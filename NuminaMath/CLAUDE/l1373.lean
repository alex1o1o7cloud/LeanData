import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l1373_137305

theorem problem_solution (a r : ℝ) (h1 : a * r = 24) (h2 : a * r^4 = 3) : a = 48 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1373_137305


namespace NUMINAMATH_CALUDE_circle_bisection_and_symmetric_points_l1373_137347

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- Define the line l
def line_l (x y k : ℝ) : Prop := y = k*x - 1

-- Define symmetry with respect to a line
def symmetric_wrt_line (x1 y1 x2 y2 k : ℝ) : Prop :=
  (x1 + x2) * (k + 1/k) = (y1 + y2) * (1 - 1/k)

-- Define perpendicularity
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem circle_bisection_and_symmetric_points :
  -- Part 1: The line y = -x - 1 bisects the circle
  (∀ x y : ℝ, circle_C x y → line_l x y (-1)) ∧
  -- Part 2: There exist points A and B on the circle satisfying the conditions
  ∃ x1 y1 x2 y2 : ℝ,
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    symmetric_wrt_line x1 y1 x2 y2 (-1) ∧
    perpendicular x1 y1 x2 y2 ∧
    ((x1 - y1 + 1 = 0 ∧ x2 - y2 + 1 = 0) ∨ (x1 - y1 - 4 = 0 ∧ x2 - y2 - 4 = 0)) :=
sorry

end NUMINAMATH_CALUDE_circle_bisection_and_symmetric_points_l1373_137347


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1373_137385

/-- An ellipse with given properties -/
structure Ellipse where
  /-- The distance between the foci -/
  focal_distance : ℝ
  /-- The distance from the center to the line connecting a focus and the endpoint of the minor axis -/
  center_to_focus_minor_line : ℝ

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ :=
  sorry

/-- Theorem: The eccentricity of the ellipse with given properties is √5/3 -/
theorem ellipse_eccentricity (e : Ellipse) 
    (h1 : e.focal_distance = 3) 
    (h2 : e.center_to_focus_minor_line = 1) : 
  eccentricity e = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1373_137385


namespace NUMINAMATH_CALUDE_student_contribution_l1373_137332

theorem student_contribution 
  (total_raised : ℕ) 
  (num_students : ℕ) 
  (cost_per_student : ℕ) 
  (remaining_funds : ℕ) : 
  total_raised = 50 → 
  num_students = 20 → 
  cost_per_student = 7 → 
  remaining_funds = 10 → 
  (total_raised - remaining_funds) / num_students = 5 :=
by sorry

end NUMINAMATH_CALUDE_student_contribution_l1373_137332


namespace NUMINAMATH_CALUDE_max_regions_40_parabolas_l1373_137351

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure VerticalParabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure HorizontalParabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the maximum number of regions created by a set of vertical and horizontal parabolas -/
def max_regions (vertical_parabolas : Finset VerticalParabola) (horizontal_parabolas : Finset HorizontalParabola) : ℕ :=
  sorry

/-- Theorem stating the maximum number of regions created by 20 vertical and 20 horizontal parabolas -/
theorem max_regions_40_parabolas :
  ∀ (v : Finset VerticalParabola) (h : Finset HorizontalParabola),
  v.card = 20 → h.card = 20 →
  max_regions v h = 2422 :=
by sorry

end NUMINAMATH_CALUDE_max_regions_40_parabolas_l1373_137351


namespace NUMINAMATH_CALUDE_cindy_calculation_l1373_137339

theorem cindy_calculation (x : ℚ) : (x - 7) / 5 = 25 → (x - 5) / 7 = 18 + 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l1373_137339


namespace NUMINAMATH_CALUDE_floor_sqrt_30_squared_l1373_137358

theorem floor_sqrt_30_squared : ⌊Real.sqrt 30⌋^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_30_squared_l1373_137358


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1373_137304

theorem complex_equation_solution (a b : ℝ) : 
  (1 - Complex.I) * (a + 2 * Complex.I) = b * Complex.I → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1373_137304


namespace NUMINAMATH_CALUDE_earthquake_ratio_l1373_137372

def initial_collapse : ℕ := 4
def total_earthquakes : ℕ := 4
def total_collapsed : ℕ := 60

def geometric_sum (a : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem earthquake_ratio :
  ∃ (r : ℚ), 
    r > 0 ∧ 
    geometric_sum initial_collapse r total_earthquakes = total_collapsed ∧
    r = 2 := by
  sorry

end NUMINAMATH_CALUDE_earthquake_ratio_l1373_137372


namespace NUMINAMATH_CALUDE_geometric_sequence_statements_l1373_137355

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_statements
    (a : ℕ → ℝ) (q : ℝ) (h : GeometricSequence a q) :
    (¬ (q > 1 → IncreasingSequence a)) ∧
    (¬ (IncreasingSequence a → q > 1)) ∧
    (¬ (q ≤ 1 → ¬IncreasingSequence a)) ∧
    (¬ (¬IncreasingSequence a → q ≤ 1)) :=
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_statements_l1373_137355


namespace NUMINAMATH_CALUDE_tan_30_degrees_l1373_137396

theorem tan_30_degrees : Real.tan (30 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_30_degrees_l1373_137396


namespace NUMINAMATH_CALUDE_final_price_is_20_70_l1373_137368

/-- The price of one kilogram of cucumbers in dollars -/
def cucumber_price : ℝ := 5

/-- The price of one kilogram of tomatoes in dollars -/
def tomato_price : ℝ := cucumber_price * (1 - 0.2)

/-- The number of kilograms of tomatoes bought -/
def tomato_kg : ℝ := 2

/-- The number of kilograms of cucumbers bought -/
def cucumber_kg : ℝ := 3

/-- The discount rate applied to the total cost -/
def discount_rate : ℝ := 0.1

/-- The final price paid for the items after discount -/
def final_price : ℝ := (tomato_price * tomato_kg + cucumber_price * cucumber_kg) * (1 - discount_rate)

theorem final_price_is_20_70 : final_price = 20.70 := by
  sorry

end NUMINAMATH_CALUDE_final_price_is_20_70_l1373_137368


namespace NUMINAMATH_CALUDE_sin_bounds_l1373_137375

theorem sin_bounds :
  (∀ x : ℝ, -5 ≤ 2 * Real.sin x - 3 ∧ 2 * Real.sin x - 3 ≤ -1) ∧
  (∃ x y : ℝ, 2 * Real.sin x - 3 = -5 ∧ 2 * Real.sin y - 3 = -1) := by sorry

end NUMINAMATH_CALUDE_sin_bounds_l1373_137375


namespace NUMINAMATH_CALUDE_pentagon_coverage_l1373_137334

-- Define a pentagon as a set of 5 points in 2D space
def Pentagon : Type := Fin 5 → ℝ × ℝ

-- Define a function to check if a pentagon is convex
def isConvex (p : Pentagon) : Prop := sorry

-- Define a function to check if all interior angles of a pentagon are obtuse
def allAnglesObtuse (p : Pentagon) : Prop := sorry

-- Define a function to check if a point is inside or on a circle
def isInsideOrOnCircle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop := sorry

-- Define a function to check if a circle covers a point of the pentagon
def circleCoversPoint (p : Pentagon) (diagonal : Fin 5 × Fin 5) (point : Fin 5) : Prop := sorry

-- Main theorem
theorem pentagon_coverage (p : Pentagon) 
  (h_convex : isConvex p) 
  (h_obtuse : allAnglesObtuse p) : 
  ∃ (d1 d2 : Fin 5 × Fin 5), ∀ (point : Fin 5), 
    circleCoversPoint p d1 point ∨ circleCoversPoint p d2 point := by
  sorry

end NUMINAMATH_CALUDE_pentagon_coverage_l1373_137334


namespace NUMINAMATH_CALUDE_binary_addition_correct_l1373_137381

/-- Represents a binary number as a list of bits (least significant bit first) -/
def BinaryNumber := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNumber) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- The four binary numbers given in the problem -/
def num1 : BinaryNumber := [true, false, true, true]
def num2 : BinaryNumber := [false, true, true]
def num3 : BinaryNumber := [true, true, false, true]
def num4 : BinaryNumber := [false, false, true, true, true]

/-- The expected sum in binary -/
def expected_sum : BinaryNumber := [true, false, true, false, false, true]

theorem binary_addition_correct :
  binary_to_decimal num1 + binary_to_decimal num2 + 
  binary_to_decimal num3 + binary_to_decimal num4 = 
  binary_to_decimal expected_sum := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_correct_l1373_137381


namespace NUMINAMATH_CALUDE_segment_length_l1373_137327

theorem segment_length : 
  let endpoints := {x : ℝ | |x - (27 : ℝ)^(1/3)| = 5}
  ∃ a b : ℝ, a ∈ endpoints ∧ b ∈ endpoints ∧ |b - a| = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_segment_length_l1373_137327


namespace NUMINAMATH_CALUDE_class_size_calculation_l1373_137398

theorem class_size_calculation (E T B N : ℕ) 
  (h1 : E = 55)
  (h2 : T = 85)
  (h3 : N = 30)
  (h4 : B = 20) :
  E + T - B + N = 150 := by
  sorry

end NUMINAMATH_CALUDE_class_size_calculation_l1373_137398


namespace NUMINAMATH_CALUDE_smallest_positive_sum_l1373_137323

theorem smallest_positive_sum (x y : ℝ) : 
  (Real.sin x + Real.cos y) * (Real.cos x - Real.sin y) = 1 + Real.sin (x - y) * Real.cos (x + y) →
  ∃ (k : ℤ), x + y = 2 * π * (k : ℝ) ∧ 
  (∀ (m : ℤ), x + y = 2 * π * (m : ℝ) → k ≤ m) ∧
  0 < 2 * π * (k : ℝ) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_sum_l1373_137323


namespace NUMINAMATH_CALUDE_power_sum_l1373_137345

theorem power_sum (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m+n) = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_l1373_137345


namespace NUMINAMATH_CALUDE_fraction_simplification_l1373_137392

theorem fraction_simplification (n : ℕ+) : (n : ℚ) * (3 : ℚ)^(n : ℕ) / (3 : ℚ)^(n : ℕ) = n := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1373_137392


namespace NUMINAMATH_CALUDE_price_decrease_revenue_unchanged_l1373_137350

theorem price_decrease_revenue_unchanged (P U : ℝ) (h_positive : P > 0 ∧ U > 0) :
  let new_price := 0.8 * P
  let new_units := U / 0.8
  let percent_decrease_price := 20
  let percent_increase_units := (new_units - U) / U * 100
  P * U = new_price * new_units →
  percent_increase_units / percent_decrease_price = 1.25 := by
sorry

end NUMINAMATH_CALUDE_price_decrease_revenue_unchanged_l1373_137350


namespace NUMINAMATH_CALUDE_stadium_length_feet_l1373_137346

/-- Converts yards to feet -/
def yards_to_feet (yards : ℕ) : ℕ := yards * 3

/-- The length of the sports stadium in yards -/
def stadium_length_yards : ℕ := 80

theorem stadium_length_feet :
  yards_to_feet stadium_length_yards = 240 := by
  sorry

end NUMINAMATH_CALUDE_stadium_length_feet_l1373_137346


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l1373_137308

/-- Given a school with a boy-to-girl ratio of 5:13 and 50 boys, prove that there are 80 more girls than boys. -/
theorem more_girls_than_boys (num_boys : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) : 
  num_boys = 50 →
  ratio_boys = 5 →
  ratio_girls = 13 →
  (ratio_girls * num_boys / ratio_boys) - num_boys = 80 :=
by sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l1373_137308


namespace NUMINAMATH_CALUDE_range_of_a_l1373_137312

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x^3 + x^2 + 1 else Real.exp (a * x)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) 3, f a x ≤ 2) ∧
  (∃ x ∈ Set.Icc (-2) 3, f a x = 2) →
  a ≤ (1/3) * Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1373_137312


namespace NUMINAMATH_CALUDE_snowboard_discount_proof_l1373_137367

theorem snowboard_discount_proof (original_price : ℝ) (friday_discount : ℝ) (monday_discount : ℝ) :
  original_price = 120 →
  friday_discount = 0.4 →
  monday_discount = 0.2 →
  let friday_price := original_price * (1 - friday_discount)
  let final_price := friday_price * (1 - monday_discount)
  final_price = 57.6 := by
sorry

end NUMINAMATH_CALUDE_snowboard_discount_proof_l1373_137367


namespace NUMINAMATH_CALUDE_unique_stamp_denomination_l1373_137369

/-- Given stamps of denominations 6, n, and n+2 cents, 
    this function returns the greatest postage that cannot be formed. -/
def greatest_unattainable_postage (n : ℕ) : ℕ :=
  6 * n * (n + 2) - (6 + n + (n + 2))

/-- This theorem states that there exists a unique positive integer n 
    such that the greatest unattainable postage is 120 cents, 
    and this n is equal to 8. -/
theorem unique_stamp_denomination :
  ∃! n : ℕ, n > 0 ∧ greatest_unattainable_postage n = 120 ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_unique_stamp_denomination_l1373_137369


namespace NUMINAMATH_CALUDE_intersection_A_B_l1373_137378

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {y | ∃ x ∈ A, y = 2 * x - 1}

theorem intersection_A_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1373_137378


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l1373_137315

def num_islands : ℕ := 8
def num_treasure_islands : ℕ := 4
def prob_treasure : ℚ := 1/3
def prob_trap : ℚ := 1/6
def prob_neither : ℚ := 1/2

theorem pirate_treasure_probability :
  (Nat.choose num_islands num_treasure_islands : ℚ) *
  prob_treasure ^ num_treasure_islands *
  prob_neither ^ (num_islands - num_treasure_islands) =
  35/648 := by sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l1373_137315


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l1373_137353

/-- Given an arithmetic sequence {a_n} where a_1 = 2, a_2 = 4, and a_3 = 6, 
    prove that the fourth term a_4 = 8. -/
theorem arithmetic_sequence_fourth_term 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 2) 
  (h2 : a 2 = 4) 
  (h3 : a 3 = 6) 
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) : 
  a 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l1373_137353


namespace NUMINAMATH_CALUDE_sequence_with_special_sums_l1373_137356

theorem sequence_with_special_sums : ∃ (seq : Fin 20 → ℝ),
  (∀ i : Fin 18, seq i + seq (i + 1) + seq (i + 2) > 0) ∧
  (Finset.sum Finset.univ seq < 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_with_special_sums_l1373_137356


namespace NUMINAMATH_CALUDE_young_photographer_club_l1373_137341

theorem young_photographer_club (total_children : ℕ) (total_groups : ℕ) (group_size : ℕ)
  (boy_boy_photos : ℕ) (girl_girl_photos : ℕ) :
  total_children = 300 →
  total_groups = 100 →
  group_size = 3 →
  total_children = total_groups * group_size →
  boy_boy_photos = 100 →
  girl_girl_photos = 56 →
  ∃ (mixed_groups : ℕ),
    mixed_groups = 72 ∧
    mixed_groups * 2 + boy_boy_photos + girl_girl_photos = total_groups * group_size :=
by sorry


end NUMINAMATH_CALUDE_young_photographer_club_l1373_137341


namespace NUMINAMATH_CALUDE_even_function_property_l1373_137374

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_property 
  (f : ℝ → ℝ) 
  (h_even : even_function f)
  (h_increasing : ∀ x y, x < y → x < 0 → y < 0 → f x < f y)
  (x₁ x₂ : ℝ)
  (h_x₁_neg : x₁ < 0)
  (h_x₂_pos : x₂ > 0)
  (h_abs : abs x₁ < abs x₂) :
  f (-x₁) > f (-x₂) := by
sorry

end NUMINAMATH_CALUDE_even_function_property_l1373_137374


namespace NUMINAMATH_CALUDE_balanced_domino_config_exists_l1373_137399

/-- A domino configuration on an n × n board. -/
structure DominoConfig (n : ℕ) where
  /-- The number of dominoes in the configuration. -/
  num_dominoes : ℕ
  /-- Predicate that the configuration is balanced. -/
  is_balanced : Prop

/-- The minimum number of dominoes needed for a balanced configuration. -/
def min_dominoes (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 * n / 3 else 2 * n

/-- Theorem stating the existence of a balanced configuration and the minimum number of dominoes needed. -/
theorem balanced_domino_config_exists (n : ℕ) (h : n ≥ 3) :
  ∃ (config : DominoConfig n), config.is_balanced ∧ config.num_dominoes = min_dominoes n :=
by sorry

end NUMINAMATH_CALUDE_balanced_domino_config_exists_l1373_137399


namespace NUMINAMATH_CALUDE_seating_theorem_l1373_137333

/-- The number of seats in the row -/
def total_seats : ℕ := 8

/-- The number of people to be seated -/
def people_to_seat : ℕ := 3

/-- A function that calculates the number of seating arrangements -/
def seating_arrangements (seats : ℕ) (people : ℕ) : ℕ :=
  -- The actual implementation is not provided in the problem
  sorry

/-- Theorem stating that the number of seating arrangements is 24 -/
theorem seating_theorem : seating_arrangements total_seats people_to_seat = 24 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l1373_137333


namespace NUMINAMATH_CALUDE_polar_curve_is_parabola_l1373_137394

/-- The curve defined by the polar equation r = 1 / (1 - sin θ) is a parabola -/
theorem polar_curve_is_parabola :
  ∀ r θ x y : ℝ,
  (r = 1 / (1 - Real.sin θ)) →
  (x = r * Real.cos θ) →
  (y = r * Real.sin θ) →
  ∃ a b : ℝ, x^2 = a * y + b :=
by sorry

end NUMINAMATH_CALUDE_polar_curve_is_parabola_l1373_137394


namespace NUMINAMATH_CALUDE_vertical_throw_meeting_conditions_l1373_137357

/-- Two objects thrown vertically upwards meet under specific conditions -/
theorem vertical_throw_meeting_conditions 
  (g a b τ : ℝ) (τ' : ℝ) (h_g_pos : g > 0) (h_a_pos : a > 0) (h_τ_pos : τ > 0) (h_τ'_pos : τ' > 0) :
  (b > a - g * τ) ∧ 
  (b > a + (g * τ^2 / 2) / (a/g - τ)) ∧ 
  (b ≥ a / Real.sqrt 2) ∧
  (b ≥ -g * τ' / 2 + Real.sqrt (2 * a^2 - g^2 * τ'^2) / 2) := by
  sorry


end NUMINAMATH_CALUDE_vertical_throw_meeting_conditions_l1373_137357


namespace NUMINAMATH_CALUDE_stephen_pizza_percentage_l1373_137379

theorem stephen_pizza_percentage (total_slices : ℕ) (stephen_percentage : ℚ) (pete_percentage : ℚ) (remaining_slices : ℕ) : 
  total_slices = 24 →
  pete_percentage = 1/2 →
  remaining_slices = 9 →
  (1 - stephen_percentage) * total_slices * (1 - pete_percentage) = remaining_slices →
  stephen_percentage = 1/4 := by
sorry

end NUMINAMATH_CALUDE_stephen_pizza_percentage_l1373_137379


namespace NUMINAMATH_CALUDE_existence_of_n_l1373_137344

theorem existence_of_n (p a k : ℕ) (h_prime : Nat.Prime p) (h_pos_a : a > 0) (h_pos_k : k > 0)
  (h_lower : p^a < k) (h_upper : k < 2*p^a) :
  ∃ n : ℕ, n < p^(2*a) ∧ (Nat.choose n k : ZMod (p^a)) = n ∧ (n : ZMod (p^a)) = k := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_l1373_137344


namespace NUMINAMATH_CALUDE_ratio_q_p_l1373_137324

/-- The number of cards in the box -/
def total_cards : ℕ := 60

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 15

/-- The number of cards for each number -/
def cards_per_number : ℕ := 4

/-- The number of cards drawn -/
def cards_drawn : ℕ := 4

/-- The probability of drawing four cards with the same number -/
def p' : ℚ := (distinct_numbers * 1) / Nat.choose total_cards cards_drawn

/-- The probability of drawing three cards with one number and one card with a different number -/
def q' : ℚ := (distinct_numbers * (distinct_numbers - 1) * Nat.choose cards_per_number 3 * Nat.choose cards_per_number 1) / Nat.choose total_cards cards_drawn

/-- The main theorem stating the ratio of q' to p' -/
theorem ratio_q_p : q' / p' = 224 := by sorry

end NUMINAMATH_CALUDE_ratio_q_p_l1373_137324


namespace NUMINAMATH_CALUDE_total_hunt_is_21_l1373_137360

/-- The number of animals hunted by Sam in a day -/
def sam_hunt : ℕ := 6

/-- The number of animals hunted by Rob in a day -/
def rob_hunt : ℕ := sam_hunt / 2

/-- The number of animals hunted by Mark in a day -/
def mark_hunt : ℕ := (sam_hunt + rob_hunt) / 3

/-- The number of animals hunted by Peter in a day -/
def peter_hunt : ℕ := 3 * mark_hunt

/-- The total number of animals hunted by all four in a day -/
def total_hunt : ℕ := sam_hunt + rob_hunt + mark_hunt + peter_hunt

theorem total_hunt_is_21 : total_hunt = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_hunt_is_21_l1373_137360


namespace NUMINAMATH_CALUDE_product_of_symmetric_complex_numbers_l1373_137389

theorem product_of_symmetric_complex_numbers :
  ∀ (z₁ z₂ : ℂ),
  (z₁.im = -z₂.im) →  -- Symmetry with respect to real axis
  (z₁.re = z₂.re) →   -- Symmetry with respect to real axis
  (z₁ = 1 + I) →      -- Given condition
  z₁ * z₂ = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_symmetric_complex_numbers_l1373_137389


namespace NUMINAMATH_CALUDE_dice_probability_l1373_137310

def num_dice : ℕ := 8
def num_faces : ℕ := 6
def num_pairs : ℕ := 3

def total_outcomes : ℕ := num_faces ^ num_dice

def favorable_outcomes : ℕ :=
  Nat.choose num_faces num_pairs *
  Nat.choose (num_faces - num_pairs) (num_dice - 2 * num_pairs) *
  Nat.factorial num_pairs *
  Nat.factorial (num_dice - 2 * num_pairs) *
  Nat.choose num_dice 2 *
  Nat.choose (num_dice - 2) 2 *
  Nat.choose (num_dice - 4) 2

theorem dice_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 525 / 972 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l1373_137310


namespace NUMINAMATH_CALUDE_second_diagonal_unrestricted_l1373_137311

/-- Represents a convex quadrilateral with specific properties -/
structure ConvexQuadrilateral where
  /-- The area of the quadrilateral in cm² -/
  area : ℝ
  /-- The length of the first diagonal in cm -/
  diagonal1 : ℝ
  /-- The length of the second diagonal in cm -/
  diagonal2 : ℝ
  /-- The sum of two opposite sides in cm -/
  opposite_sides_sum : ℝ
  /-- The area is positive -/
  area_pos : area > 0
  /-- Both diagonals are positive -/
  diag1_pos : diagonal1 > 0
  diag2_pos : diagonal2 > 0
  /-- The sum of opposite sides is non-negative -/
  opp_sides_sum_nonneg : opposite_sides_sum ≥ 0
  /-- The area is 32 cm² -/
  area_is_32 : area = 32
  /-- The sum of one diagonal and two opposite sides is 16 cm -/
  sum_is_16 : diagonal1 + opposite_sides_sum = 16

/-- Theorem stating that the second diagonal can be any positive real number -/
theorem second_diagonal_unrestricted (q : ConvexQuadrilateral) : 
  ∀ x : ℝ, x > 0 → ∃ q' : ConvexQuadrilateral, q'.diagonal2 = x := by
  sorry

end NUMINAMATH_CALUDE_second_diagonal_unrestricted_l1373_137311


namespace NUMINAMATH_CALUDE_no_all_ones_quadratic_l1373_137319

/-- A natural number whose decimal representation consists only of ones -/
def all_ones (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (10^k - 1) / 9

/-- The property that a natural number's decimal representation consists only of ones -/
def has_all_ones_representation (n : ℕ) : Prop :=
  all_ones n

/-- A quadratic polynomial with integer coefficients -/
def is_quadratic_polynomial (P : ℕ → ℕ) : Prop :=
  ∃ a b c : ℤ, ∀ x : ℕ, P x = a * x^2 + b * x + c

theorem no_all_ones_quadratic :
  ∀ P : ℕ → ℕ, is_quadratic_polynomial P →
    ∃ n : ℕ, has_all_ones_representation n ∧ ¬(has_all_ones_representation (P n)) :=
sorry

end NUMINAMATH_CALUDE_no_all_ones_quadratic_l1373_137319


namespace NUMINAMATH_CALUDE_kaleb_final_amount_l1373_137301

def kaleb_lawn_business (spring_earnings summer_earnings supply_costs : ℕ) : ℕ :=
  spring_earnings + summer_earnings - supply_costs

theorem kaleb_final_amount :
  kaleb_lawn_business 4 50 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_final_amount_l1373_137301


namespace NUMINAMATH_CALUDE_women_married_fraction_l1373_137371

theorem women_married_fraction (total : ℕ) (h1 : total > 0) :
  let women := (64 : ℚ) / 100 * total
  let married := (60 : ℚ) / 100 * total
  let men := total - women
  let single_men := (2 : ℚ) / 3 * men
  let married_men := men - single_men
  let married_women := married - married_men
  married_women / women = (3 : ℚ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_women_married_fraction_l1373_137371


namespace NUMINAMATH_CALUDE_n_solution_approx_l1373_137382

def n_equation (n : ℝ) : Prop :=
  (n + 2 * 1.5) ^ 5 = (1 + 3 * 1.5) ^ 4

theorem n_solution_approx : ∃ n : ℝ, n_equation n ∧ abs (n - 0.72) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_n_solution_approx_l1373_137382


namespace NUMINAMATH_CALUDE_table_movement_l1373_137329

theorem table_movement (table_width : ℝ) (table_length : ℝ) : 
  table_width = 8 ∧ table_length = 10 →
  ∃ (S : ℕ), S = 13 ∧ 
  (∀ (T : ℕ), T < S → Real.sqrt (table_width^2 + table_length^2) > T) ∧
  Real.sqrt (table_width^2 + table_length^2) ≤ S :=
by sorry

end NUMINAMATH_CALUDE_table_movement_l1373_137329


namespace NUMINAMATH_CALUDE_integer_triple_sum_product_l1373_137320

theorem integer_triple_sum_product : 
  ∀ a b c : ℕ+, 
    (a + b + c = 25 ∧ a * b * c = 360) ↔ 
    ((a = 4 ∧ b = 6 ∧ c = 15) ∨ 
     (a = 3 ∧ b = 10 ∧ c = 12) ∨
     (a = 4 ∧ b = 15 ∧ c = 6) ∨ 
     (a = 6 ∧ b = 4 ∧ c = 15) ∨
     (a = 6 ∧ b = 15 ∧ c = 4) ∨
     (a = 15 ∧ b = 4 ∧ c = 6) ∨
     (a = 15 ∧ b = 6 ∧ c = 4) ∨
     (a = 3 ∧ b = 12 ∧ c = 10) ∨
     (a = 10 ∧ b = 3 ∧ c = 12) ∨
     (a = 10 ∧ b = 12 ∧ c = 3) ∨
     (a = 12 ∧ b = 3 ∧ c = 10) ∨
     (a = 12 ∧ b = 10 ∧ c = 3)) :=
by sorry

end NUMINAMATH_CALUDE_integer_triple_sum_product_l1373_137320


namespace NUMINAMATH_CALUDE_probability_is_one_fifth_l1373_137370

/-- The probability of finding the last defective product on the fourth inspection -/
def probability_last_defective_fourth_inspection (total : ℕ) (qualified : ℕ) (defective : ℕ) : ℚ :=
  let p1 := qualified / total * (qualified - 1) / (total - 1) * defective / (total - 2) * 1 / (total - 3)
  let p2 := qualified / total * defective / (total - 1) * (qualified - 1) / (total - 2) * 1 / (total - 3)
  let p3 := defective / total * qualified / (total - 1) * (qualified - 1) / (total - 2) * 1 / (total - 3)
  p1 + p2 + p3

/-- Theorem stating that the probability is 1/5 for the given conditions -/
theorem probability_is_one_fifth :
  probability_last_defective_fourth_inspection 6 4 2 = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_fifth_l1373_137370


namespace NUMINAMATH_CALUDE_acid_solution_dilution_l1373_137380

theorem acid_solution_dilution (m : ℝ) (x : ℝ) (h : m > 25) :
  (m * m / 100 = (m - 15) / 100 * (m + x)) → x = 15 * m / (m - 15) := by
  sorry

end NUMINAMATH_CALUDE_acid_solution_dilution_l1373_137380


namespace NUMINAMATH_CALUDE_soccer_match_goals_l1373_137322

/-- Calculates the total number of goals scored in a soccer match -/
def total_goals (kickers_first : ℕ) : ℕ := 
  let kickers_second : ℕ := 2 * kickers_first
  let spiders_first : ℕ := kickers_first / 2
  let spiders_second : ℕ := 2 * kickers_second
  kickers_first + kickers_second + spiders_first + spiders_second

/-- Theorem stating that given the conditions of the soccer match, the total goals scored is 15 -/
theorem soccer_match_goals : total_goals 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_soccer_match_goals_l1373_137322


namespace NUMINAMATH_CALUDE_club_members_count_l1373_137343

theorem club_members_count :
  ∃! n : ℕ, 150 ≤ n ∧ n ≤ 300 ∧ n % 10 = 6 ∧ n % 11 = 6 ∧ n = 226 := by
  sorry

end NUMINAMATH_CALUDE_club_members_count_l1373_137343


namespace NUMINAMATH_CALUDE_smallest_multiple_l1373_137390

theorem smallest_multiple (n : ℕ) : n = 187 ↔ 
  n > 0 ∧ 
  17 ∣ n ∧ 
  n % 53 = 7 ∧ 
  ∀ m : ℕ, m > 0 → 17 ∣ m → m % 53 = 7 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l1373_137390


namespace NUMINAMATH_CALUDE_hansol_weight_l1373_137361

/-- Given two people, Hanbyul and Hansol, with the following conditions:
    1. The sum of their weights is 88 kg.
    2. Hanbyul weighs 4 kg more than Hansol.
    Prove that Hansol weighs 42 kg. -/
theorem hansol_weight (hanbyul hansol : ℝ) 
    (sum_weight : hanbyul + hansol = 88)
    (weight_diff : hanbyul = hansol + 4) : 
  hansol = 42 := by
  sorry

end NUMINAMATH_CALUDE_hansol_weight_l1373_137361


namespace NUMINAMATH_CALUDE_tens_digit_of_19_power_2023_l1373_137317

theorem tens_digit_of_19_power_2023 : ∃ n : ℕ, 19^2023 ≡ 10 * n + 5 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_19_power_2023_l1373_137317


namespace NUMINAMATH_CALUDE_skateboard_bicycle_problem_l1373_137313

theorem skateboard_bicycle_problem (skateboards bicycles : ℕ) : 
  (skateboards : ℚ) / bicycles = 7 / 4 →
  skateboards = bicycles + 12 →
  skateboards + bicycles = 44 := by
sorry

end NUMINAMATH_CALUDE_skateboard_bicycle_problem_l1373_137313


namespace NUMINAMATH_CALUDE_difference_max_min_both_l1373_137331

/-- The total number of students at the university -/
def total_students : ℕ := 2500

/-- The number of students studying German -/
def german_students : ℕ → Prop :=
  λ g => 1750 ≤ g ∧ g ≤ 1875

/-- The number of students studying Russian -/
def russian_students : ℕ → Prop :=
  λ r => 625 ≤ r ∧ r ≤ 875

/-- The number of students studying both German and Russian -/
def both_languages (g r b : ℕ) : Prop :=
  g + r - b = total_students

/-- The minimum number of students studying both languages -/
def min_both (m : ℕ) : Prop :=
  ∃ g r, german_students g ∧ russian_students r ∧ both_languages g r m ∧
  ∀ b, (∃ g' r', german_students g' ∧ russian_students r' ∧ both_languages g' r' b) → m ≤ b

/-- The maximum number of students studying both languages -/
def max_both (M : ℕ) : Prop :=
  ∃ g r, german_students g ∧ russian_students r ∧ both_languages g r M ∧
  ∀ b, (∃ g' r', german_students g' ∧ russian_students r' ∧ both_languages g' r' b) → b ≤ M

theorem difference_max_min_both :
  ∃ m M, min_both m ∧ max_both M ∧ M - m = 375 := by
  sorry

end NUMINAMATH_CALUDE_difference_max_min_both_l1373_137331


namespace NUMINAMATH_CALUDE_computer_price_increase_l1373_137309

theorem computer_price_increase (c : ℝ) (h : 2 * c = 540) : 
  c * (1 + 0.3) = 351 := by
sorry

end NUMINAMATH_CALUDE_computer_price_increase_l1373_137309


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_train_passing_jogger_time_approx_l1373_137359

/-- Time for a train to pass a jogger -/
theorem train_passing_jogger_time 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (train_length : ℝ) 
  (initial_distance : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- The time for the train to pass the jogger is approximately 38.75 seconds -/
theorem train_passing_jogger_time_approx :
  ∃ ε > 0, abs (train_passing_jogger_time 8 60 200 360 - 38.75) < ε :=
sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_train_passing_jogger_time_approx_l1373_137359


namespace NUMINAMATH_CALUDE_point_outside_circle_l1373_137340

theorem point_outside_circle 
  (a b : ℝ) 
  (line_intersects_circle : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ 
    a * x₁ + b * y₁ = 1 ∧ a * x₂ + b * y₂ = 1 ∧
    x₁^2 + y₁^2 = 1 ∧ x₂^2 + y₂^2 = 1) :
  a^2 + b^2 > 1 := by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l1373_137340


namespace NUMINAMATH_CALUDE_decimal_addition_l1373_137335

theorem decimal_addition : (7.15 : ℝ) + 2.639 = 9.789 := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_l1373_137335


namespace NUMINAMATH_CALUDE_cubic_equation_sum_l1373_137307

theorem cubic_equation_sum (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a = 12 →
  b^3 - 6*b^2 + 11*b = 12 →
  c^3 - 6*c^2 + 11*c = 12 →
  a*b/c + b*c/a + c*a/b = -23/12 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_l1373_137307


namespace NUMINAMATH_CALUDE_equal_division_of_trout_l1373_137300

theorem equal_division_of_trout (total_trout : ℕ) (num_people : ℕ) (trout_per_person : ℕ) :
  total_trout = 18 →
  num_people = 2 →
  trout_per_person = total_trout / num_people →
  trout_per_person = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_equal_division_of_trout_l1373_137300


namespace NUMINAMATH_CALUDE_cube_edge_from_volume_l1373_137363

theorem cube_edge_from_volume (volume : ℝ) (edge : ℝ) :
  volume = 3375 ∧ volume = edge ^ 3 → edge = 15 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_from_volume_l1373_137363


namespace NUMINAMATH_CALUDE_locus_is_ellipse_l1373_137306

/-- The locus of points (x, y) in the complex plane satisfying 
    |z-2+i| + |z+3-i| = 6 is an ellipse -/
theorem locus_is_ellipse (z : ℂ) :
  let x := z.re
  let y := z.im
  (Complex.abs (z - (2 - Complex.I)) + Complex.abs (z - (-3 + Complex.I)) = 6) ↔
  ∃ (a b : ℝ) (h : 0 < b ∧ b < a),
    (x^2 / a^2) + (y^2 / b^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_is_ellipse_l1373_137306


namespace NUMINAMATH_CALUDE_abc_and_fourth_power_sum_l1373_137395

theorem abc_and_fourth_power_sum (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a^2 + b^2 + c^2 = 2) 
  (h3 : a^3 + b^3 + c^3 = 3) : 
  a * b * c = 1/6 ∧ a^4 + b^4 + c^4 = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_abc_and_fourth_power_sum_l1373_137395


namespace NUMINAMATH_CALUDE_circle_triangle_perimeter_l1373_137348

structure Circle :=
  (points : Fin 6 → ℝ × ℝ)

structure Triangle :=
  (vertices : Fin 3 → ℝ × ℝ)

def perimeter (t : Triangle) : ℝ := sorry

theorem circle_triangle_perimeter
  (c : Circle)
  (x y z : ℝ × ℝ)
  (h1 : x ∈ Set.Icc (c.points 0) (c.points 3) ∩ Set.Icc (c.points 1) (c.points 4))
  (h2 : y ∈ Set.Icc (c.points 0) (c.points 3) ∩ Set.Icc (c.points 2) (c.points 5))
  (h3 : z ∈ Set.Icc (c.points 2) (c.points 5) ∩ Set.Icc (c.points 1) (c.points 4))
  (h4 : x ∈ Set.Icc z (c.points 1))
  (h5 : x ∈ Set.Icc y (c.points 0))
  (h6 : y ∈ Set.Icc z (c.points 2))
  (h7 : dist (c.points 0) x = 3)
  (h8 : dist (c.points 1) x = 2)
  (h9 : dist (c.points 2) y = 4)
  (h10 : dist (c.points 3) y = 10)
  (h11 : dist (c.points 4) z = 16)
  (h12 : dist (c.points 5) z = 12)
  : perimeter { vertices := ![x, y, z] } = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_triangle_perimeter_l1373_137348


namespace NUMINAMATH_CALUDE_g_negative_two_equals_negative_fifteen_l1373_137397

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 7
def g (a b x : ℝ) : ℝ := f a b x + 2

-- State the theorem
theorem g_negative_two_equals_negative_fifteen 
  (a b : ℝ) (h : f a b 2 = 3) : g a b (-2) = -15 := by
  sorry

end NUMINAMATH_CALUDE_g_negative_two_equals_negative_fifteen_l1373_137397


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l1373_137364

/-- A regular polygon with exterior angle 20° and side length 10 has perimeter 180 -/
theorem regular_polygon_perimeter (n : ℕ) (exterior_angle : ℝ) (side_length : ℝ) : 
  n > 2 →
  exterior_angle = 20 →
  side_length = 10 →
  n * exterior_angle = 360 →
  n * side_length = 180 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l1373_137364


namespace NUMINAMATH_CALUDE_solution_set_inequality1_solution_set_inequality2_l1373_137386

-- Problem 1
def inequality1 (x : ℝ) : Prop := abs (x - 2) + abs (2 * x - 3) < 4

theorem solution_set_inequality1 :
  {x : ℝ | inequality1 x} = {x : ℝ | 1/3 < x ∧ x < 3} :=
by sorry

-- Problem 2
def inequality2 (x : ℝ) : Prop := (x^2 - 3*x) / (x^2 - x - 2) ≤ x

theorem solution_set_inequality2 :
  {x : ℝ | inequality2 x} = 
    {x : ℝ | -1 < x ∧ x ≤ 0} ∪ {1} ∪ {x : ℝ | x > 2} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality1_solution_set_inequality2_l1373_137386


namespace NUMINAMATH_CALUDE_deck_cost_is_32_l1373_137366

/-- Calculates the total cost of Tom's deck of cards. -/
def deck_cost : ℝ :=
  let rare_count : ℕ := 19
  let uncommon_count : ℕ := 11
  let common_count : ℕ := 30
  let rare_cost : ℝ := 1
  let uncommon_cost : ℝ := 0.5
  let common_cost : ℝ := 0.25
  rare_count * rare_cost + uncommon_count * uncommon_cost + common_count * common_cost

/-- Proves that the total cost of Tom's deck is $32. -/
theorem deck_cost_is_32 : deck_cost = 32 := by
  sorry

end NUMINAMATH_CALUDE_deck_cost_is_32_l1373_137366


namespace NUMINAMATH_CALUDE_fraction_equality_sum_l1373_137354

theorem fraction_equality_sum (P Q : ℚ) : 
  (4 : ℚ) / 7 = P / 49 ∧ (4 : ℚ) / 7 = 56 / Q → P + Q = 126 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_sum_l1373_137354


namespace NUMINAMATH_CALUDE_intersection_equality_l1373_137365

theorem intersection_equality (m : ℝ) : 
  let A : Set ℝ := {2, 5, m^2 - m}
  let B : Set ℝ := {2, m + 3}
  A ∩ B = B → m = 3 := by
sorry

end NUMINAMATH_CALUDE_intersection_equality_l1373_137365


namespace NUMINAMATH_CALUDE_greatest_power_of_seven_in_50_factorial_l1373_137342

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def highest_power_of_seven (n : ℕ) : ℕ :=
  if n < 7 then 0
  else (n / 7) + highest_power_of_seven (n / 7)

theorem greatest_power_of_seven_in_50_factorial :
  ∃ (z : ℕ), z = highest_power_of_seven 50 ∧
  (7^z : ℕ) ∣ factorial 50 ∧
  ∀ (y : ℕ), y > z → ¬((7^y : ℕ) ∣ factorial 50) :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_seven_in_50_factorial_l1373_137342


namespace NUMINAMATH_CALUDE_ellipse_and_line_theorem_l1373_137314

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  eccentricity : ℝ
  passes_through : ℝ × ℝ

/-- A line passing through a given point -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- The dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem ellipse_and_line_theorem (C : Ellipse) (l : Line) : 
  C.center = (0, 0) ∧ 
  C.foci_on_x_axis = true ∧ 
  C.eccentricity = 1/2 ∧ 
  C.passes_through = (1, 3/2) ∧
  l.point = (2, 1) →
  (∃ (A B : ℝ × ℝ), 
    -- C has equation x^2/4 + y^2/3 = 1
    (A.1^2/4 + A.2^2/3 = 1 ∧ B.1^2/4 + B.2^2/3 = 1) ∧
    -- A and B are on line l
    (A.2 - l.point.2 = l.slope * (A.1 - l.point.1) ∧ 
     B.2 - l.point.2 = l.slope * (B.1 - l.point.1)) ∧
    -- A and B are distinct
    A ≠ B ∧
    -- PA · PB = PM^2
    dot_product (A.1 - l.point.1, A.2 - l.point.2) (B.1 - l.point.1, B.2 - l.point.2) = 
    dot_product (1 - 2, 3/2 - 1) (1 - 2, 3/2 - 1) ∧
    -- l has equation y = (1/2)x
    l.slope = 1/2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_theorem_l1373_137314


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l1373_137387

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c m n : ℝ) : Prop :=
  ∀ x, f a b c x > 0 ↔ m < x ∧ x < n

-- State the theorem
theorem quadratic_inequality_properties
  (a b c m n : ℝ)
  (h_sol : solution_set a b c m n)
  (h_m_pos : m > 0)
  (h_n_gt_m : n > m) :
  a < 0 ∧
  b > 0 ∧
  (∀ x, f c b a x > 0 ↔ 1/n < x ∧ x < 1/m) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l1373_137387


namespace NUMINAMATH_CALUDE_equation_equivalent_to_lines_l1373_137326

-- Define the original equation
def original_equation (x y : ℝ) : Prop :=
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = -x - 2
def line2 (x y : ℝ) : Prop := y = -2 * x + 1

-- Theorem statement
theorem equation_equivalent_to_lines :
  ∀ x y : ℝ, original_equation x y ↔ (line1 x y ∨ line2 x y) :=
sorry

end NUMINAMATH_CALUDE_equation_equivalent_to_lines_l1373_137326


namespace NUMINAMATH_CALUDE_triangle_isosceles_if_c_eq_2a_cos_B_l1373_137362

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Side lengths

-- Define the property of being isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- State the theorem
theorem triangle_isosceles_if_c_eq_2a_cos_B (t : Triangle) 
  (h : t.c = 2 * t.a * Real.cos t.B) : isIsosceles t :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_if_c_eq_2a_cos_B_l1373_137362


namespace NUMINAMATH_CALUDE_june_birth_percentage_l1373_137352

/-- The total number of scientists -/
def total_scientists : ℕ := 150

/-- The number of scientists born in June -/
def june_scientists : ℕ := 15

/-- The percentage of scientists born in June -/
def june_percentage : ℚ := (june_scientists : ℚ) / (total_scientists : ℚ) * 100

theorem june_birth_percentage :
  june_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_june_birth_percentage_l1373_137352


namespace NUMINAMATH_CALUDE_bird_migration_difference_l1373_137349

/-- The number of bird families that flew away for the winter -/
def flew_away : ℕ := 86

/-- The number of bird families initially living near the mountain -/
def initial_families : ℕ := 45

/-- The difference between the number of bird families that flew away and those that stayed behind -/
def difference : ℕ := flew_away - initial_families

theorem bird_migration_difference :
  difference = 41 :=
sorry

end NUMINAMATH_CALUDE_bird_migration_difference_l1373_137349


namespace NUMINAMATH_CALUDE_total_fruits_bought_l1373_137376

/-- The total number of fruits bought given the cost and quantity constraints -/
theorem total_fruits_bought
  (total_cost : ℕ)
  (plum_cost peach_cost : ℕ)
  (plum_quantity : ℕ)
  (h1 : total_cost = 52)
  (h2 : plum_cost = 2)
  (h3 : peach_cost = 1)
  (h4 : plum_quantity = 20)
  (h5 : plum_cost * plum_quantity + peach_cost * (total_cost - plum_cost * plum_quantity) = total_cost) :
  plum_quantity + (total_cost - plum_cost * plum_quantity) = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_bought_l1373_137376


namespace NUMINAMATH_CALUDE_fraction_inequality_l1373_137337

theorem fraction_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1373_137337


namespace NUMINAMATH_CALUDE_lemonade_stand_boys_l1373_137325

theorem lemonade_stand_boys (initial_group : ℕ) : 
  let initial_boys : ℕ := (6 * initial_group) / 10
  let final_group : ℕ := initial_group
  let final_boys : ℕ := initial_boys - 3
  (6 * initial_group = 10 * initial_boys) ∧ 
  (2 * final_boys = final_group) →
  initial_boys = 18 := by
sorry

end NUMINAMATH_CALUDE_lemonade_stand_boys_l1373_137325


namespace NUMINAMATH_CALUDE_lion_meeting_day_l1373_137383

/-- Represents days of the week -/
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the day after a given day -/
def nextDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

/-- Returns true if the lion lies on the given day according to his pattern -/
def lionLies (d : Day) : Prop :=
  d = Day.Tuesday ∨ d = Day.Friday ∨ d = Day.Saturday

/-- The day Alice met the lion -/
def meetingDay : Day := Day.Monday

theorem lion_meeting_day :
  (lionLies (nextDay (nextDay meetingDay)) ∧
   lionLies (nextDay (nextDay (nextDay meetingDay))) ∧
   ¬lionLies (nextDay meetingDay)) ∧
  ¬(lionLies (nextDay (nextDay (nextDay meetingDay))) ∧
    lionLies (nextDay (nextDay (nextDay (nextDay meetingDay)))) ∧
    ¬lionLies (nextDay (nextDay (nextDay (nextDay (nextDay meetingDay))))) ∧
    meetingDay ≠ Day.Monday) :=
by sorry


end NUMINAMATH_CALUDE_lion_meeting_day_l1373_137383


namespace NUMINAMATH_CALUDE_logarithm_inequality_l1373_137302

theorem logarithm_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.log (1 + Real.sqrt (a * b)) ≤ (Real.log (1 + a) + Real.log (1 + b)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l1373_137302


namespace NUMINAMATH_CALUDE_trapezoid_length_in_divided_square_l1373_137393

/-- Given a square with side length 2 meters, divided into two congruent trapezoids and a quadrilateral,
    where the trapezoids have bases on two sides of the square and their other bases meet at the square's center,
    and all three shapes have equal areas, the length of the longer parallel side of each trapezoid is 5/3 meters. -/
theorem trapezoid_length_in_divided_square :
  let square_side : ℝ := 2
  let total_area : ℝ := square_side ^ 2
  let shape_area : ℝ := total_area / 3
  let shorter_base : ℝ := square_side / 2
  ∃ (longer_base : ℝ),
    longer_base = 5 / 3 ∧
    shape_area = (longer_base + shorter_base) * square_side / 4 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_length_in_divided_square_l1373_137393


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_to_both_red_l1373_137391

/-- Represents the color of a card -/
inductive Color
  | Red
  | Green
  | Blue

/-- Represents a pair of cards drawn from the bag -/
structure DrawnCards :=
  (first : Color)
  (second : Color)

/-- The bag containing 2 red, 2 green, and 2 blue cards -/
def bag : Multiset Color := 
  2 • {Color.Red} + 2 • {Color.Green} + 2 • {Color.Blue}

/-- Event: Both cards are red -/
def bothRed (draw : DrawnCards) : Prop :=
  draw.first = Color.Red ∧ draw.second = Color.Red

/-- Event: Neither of the 2 cards is red -/
def neitherRed (draw : DrawnCards) : Prop :=
  draw.first ≠ Color.Red ∧ draw.second ≠ Color.Red

/-- Event: Exactly one card is blue -/
def exactlyOneBlue (draw : DrawnCards) : Prop :=
  (draw.first = Color.Blue ∧ draw.second ≠ Color.Blue) ∨
  (draw.first ≠ Color.Blue ∧ draw.second = Color.Blue)

/-- Event: Both cards are green -/
def bothGreen (draw : DrawnCards) : Prop :=
  draw.first = Color.Green ∧ draw.second = Color.Green

theorem events_mutually_exclusive_to_both_red :
  ∀ (draw : DrawnCards),
    (bothRed draw → ¬(neitherRed draw)) ∧
    (bothRed draw → ¬(exactlyOneBlue draw)) ∧
    (bothRed draw → ¬(bothGreen draw)) :=
  sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_to_both_red_l1373_137391


namespace NUMINAMATH_CALUDE_frames_cost_l1373_137373

theorem frames_cost (lens_cost : ℝ) (insurance_coverage : ℝ) (coupon : ℝ) (total_cost : ℝ)
  (h1 : lens_cost = 500)
  (h2 : insurance_coverage = 0.8)
  (h3 : coupon = 50)
  (h4 : total_cost = 250)
  : ∃ (frame_cost : ℝ), frame_cost = 200 ∧
    total_cost = (frame_cost - coupon) + (lens_cost * (1 - insurance_coverage)) := by
  sorry

end NUMINAMATH_CALUDE_frames_cost_l1373_137373


namespace NUMINAMATH_CALUDE_trillion_to_scientific_notation_l1373_137316

/-- Represents the value of one trillion -/
def trillion : ℕ := 1000000000000

/-- Proves that 6.13 trillion is equal to 6.13 × 10^12 -/
theorem trillion_to_scientific_notation : 
  (6.13 : ℝ) * (trillion : ℝ) = 6.13 * (10 : ℝ)^12 := by
  sorry

end NUMINAMATH_CALUDE_trillion_to_scientific_notation_l1373_137316


namespace NUMINAMATH_CALUDE_thomas_run_conversion_l1373_137303

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * 7^3 + d₂ * 7^2 + d₁ * 7^1 + d₀ * 7^0

/-- Thomas's run in base 7 -/
def thomasRunBase7 : ℕ × ℕ × ℕ × ℕ := (4, 2, 1, 3)

theorem thomas_run_conversion :
  let (d₃, d₂, d₁, d₀) := thomasRunBase7
  base7ToBase10 d₃ d₂ d₁ d₀ = 1480 := by
sorry

end NUMINAMATH_CALUDE_thomas_run_conversion_l1373_137303


namespace NUMINAMATH_CALUDE_at_least_two_consecutive_successes_l1373_137384

def probability_success : ℚ := 2 / 5

def probability_failure : ℚ := 1 - probability_success

def number_of_attempts : ℕ := 4

theorem at_least_two_consecutive_successes :
  let p_success := probability_success
  let p_failure := probability_failure
  let n := number_of_attempts
  (1 : ℚ) - (p_failure^n + n * p_success * p_failure^(n-1) + 3 * p_success^2 * p_failure^2) = 44 / 125 := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_consecutive_successes_l1373_137384


namespace NUMINAMATH_CALUDE_rectangle_area_l1373_137328

/-- Proves that the area of a rectangle with length 3 times its width and width of 4 inches is 48 square inches -/
theorem rectangle_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 4 →
  length = 3 * width →
  area = length * width →
  area = 48 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1373_137328


namespace NUMINAMATH_CALUDE_f_has_zero_in_interval_l1373_137318

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the function f in terms of g
def f (x : ℝ) : ℝ := (x^2 - 3*x + 2) * g x + 3*x - 4

-- State the theorem
theorem f_has_zero_in_interval (hg : Continuous g) :
  ∃ c ∈ Set.Ioo 1 2, f g c = 0 := by
  sorry


end NUMINAMATH_CALUDE_f_has_zero_in_interval_l1373_137318


namespace NUMINAMATH_CALUDE_red_surface_area_fraction_is_three_fourths_l1373_137388

/-- Represents a cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  small_cube_count : ℕ
  red_cube_count : ℕ
  blue_cube_count : ℕ
  blue_corners_per_face : ℕ

/-- The fraction of the surface area of the large cube that is red -/
def red_surface_area_fraction (c : LargeCube) : ℚ :=
  sorry

/-- The given large cube constructed from smaller cubes -/
def given_cube : LargeCube :=
  { edge_length := 4
  , small_cube_count := 64
  , red_cube_count := 32
  , blue_cube_count := 32
  , blue_corners_per_face := 4 }

theorem red_surface_area_fraction_is_three_fourths :
  red_surface_area_fraction given_cube = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_red_surface_area_fraction_is_three_fourths_l1373_137388


namespace NUMINAMATH_CALUDE_point_inside_circle_l1373_137338

/-- A point is inside a circle if its distance from the center is less than the radius -/
def is_inside_circle (center_distance radius : ℝ) : Prop :=
  center_distance < radius

/-- Given a circle with radius 5 and a point A at distance 4 from the center,
    prove that point A is inside the circle -/
theorem point_inside_circle (center_distance radius : ℝ)
  (h1 : center_distance = 4)
  (h2 : radius = 5) :
  is_inside_circle center_distance radius := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_l1373_137338


namespace NUMINAMATH_CALUDE_quadrilateral_AD_length_l1373_137336

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Real × Real)

-- Define the conditions of the problem
def is_convex (q : Quadrilateral) : Prop := sorry

def angle_BAC_eq_BDA (q : Quadrilateral) : Prop := sorry

def angle_BAD_eq_60 (q : Quadrilateral) : Prop := sorry

def angle_ADC_eq_60 (q : Quadrilateral) : Prop := sorry

def length_AB_eq_14 (q : Quadrilateral) : Real := sorry

def length_CD_eq_6 (q : Quadrilateral) : Real := sorry

def length_AD (q : Quadrilateral) : Real := sorry

-- Theorem statement
theorem quadrilateral_AD_length 
  (q : Quadrilateral) 
  (h1 : is_convex q)
  (h2 : angle_BAC_eq_BDA q)
  (h3 : angle_BAD_eq_60 q)
  (h4 : angle_ADC_eq_60 q)
  (h5 : length_AB_eq_14 q = 14)
  (h6 : length_CD_eq_6 q = 6) :
  length_AD q = 20 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_AD_length_l1373_137336


namespace NUMINAMATH_CALUDE_abs_y_bound_l1373_137330

theorem abs_y_bound (x y : ℝ) (h1 : |x + y| < 1/3) (h2 : |2*x - y| < 1/6) : |y| < 5/18 := by
  sorry

end NUMINAMATH_CALUDE_abs_y_bound_l1373_137330


namespace NUMINAMATH_CALUDE_business_partnership_problem_l1373_137321

/-- A business partnership problem -/
theorem business_partnership_problem
  (x_capital y_capital z_capital : ℕ)
  (total_profit z_profit : ℕ)
  (x_months y_months z_months : ℕ)
  (h1 : x_capital = 20000)
  (h2 : z_capital = 30000)
  (h3 : total_profit = 50000)
  (h4 : z_profit = 14000)
  (h5 : x_months = 12)
  (h6 : y_months = 12)
  (h7 : z_months = 7)
  (h8 : (z_capital * z_months) / (x_capital * x_months + y_capital * y_months + z_capital * z_months) = z_profit / total_profit) :
  y_capital = 25000 := by
  sorry


end NUMINAMATH_CALUDE_business_partnership_problem_l1373_137321


namespace NUMINAMATH_CALUDE_lauren_earnings_l1373_137377

/-- Represents the earnings for a single day --/
structure DayEarnings where
  commercial_rate : ℝ
  subscription_rate : ℝ
  commercial_views : ℕ
  subscriptions : ℕ

/-- Calculates the total earnings for a single day --/
def day_total (d : DayEarnings) : ℝ :=
  d.commercial_rate * d.commercial_views + d.subscription_rate * d.subscriptions

/-- Represents the earnings for the weekend --/
structure WeekendEarnings where
  merchandise_sales : ℝ
  merchandise_rate : ℝ

/-- Calculates the total earnings for the weekend --/
def weekend_total (w : WeekendEarnings) : ℝ :=
  w.merchandise_sales * w.merchandise_rate

/-- Represents Lauren's earnings for the entire period --/
structure PeriodEarnings where
  monday : DayEarnings
  tuesday : DayEarnings
  weekend : WeekendEarnings

/-- Calculates the total earnings for the entire period --/
def period_total (p : PeriodEarnings) : ℝ :=
  day_total p.monday + day_total p.tuesday + weekend_total p.weekend

/-- Theorem stating that Lauren's total earnings for the period equal $140.00 --/
theorem lauren_earnings :
  let p : PeriodEarnings := {
    monday := {
      commercial_rate := 0.40,
      subscription_rate := 0.80,
      commercial_views := 80,
      subscriptions := 20
    },
    tuesday := {
      commercial_rate := 0.50,
      subscription_rate := 1.00,
      commercial_views := 100,
      subscriptions := 27
    },
    weekend := {
      merchandise_sales := 150,
      merchandise_rate := 0.10
    }
  }
  period_total p = 140
:= by sorry

end NUMINAMATH_CALUDE_lauren_earnings_l1373_137377
