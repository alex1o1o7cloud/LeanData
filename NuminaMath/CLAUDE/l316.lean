import Mathlib

namespace NUMINAMATH_CALUDE_officer_average_salary_l316_31662

/-- Proves that the average salary of officers is 450 Rs/month --/
theorem officer_average_salary
  (total_avg : ℝ)
  (non_officer_avg : ℝ)
  (officer_count : ℕ)
  (non_officer_count : ℕ)
  (h_total_avg : total_avg = 120)
  (h_non_officer_avg : non_officer_avg = 110)
  (h_officer_count : officer_count = 15)
  (h_non_officer_count : non_officer_count = 495) :
  (total_avg * (officer_count + non_officer_count) - non_officer_avg * non_officer_count) / officer_count = 450 :=
by sorry

end NUMINAMATH_CALUDE_officer_average_salary_l316_31662


namespace NUMINAMATH_CALUDE_expected_value_of_product_l316_31683

def marbles : Finset ℕ := {1, 2, 3, 4, 5, 6}

def product_sum : ℕ := (marbles.powerset.filter (fun s => s.card = 2)).sum (fun s => s.prod id)

def total_combinations : ℕ := Nat.choose 6 2

theorem expected_value_of_product :
  (product_sum : ℚ) / total_combinations = 35 / 3 :=
sorry

end NUMINAMATH_CALUDE_expected_value_of_product_l316_31683


namespace NUMINAMATH_CALUDE_solution_of_equation_l316_31616

theorem solution_of_equation (x : ℝ) : 
  (3 / (x + 2) - 1 / x = 0) ↔ (x = 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_equation_l316_31616


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l316_31623

theorem complex_magnitude_problem (x y : ℝ) (h : (Complex.I + 2) * y = x + y * Complex.I) (hy : y ≠ 0) :
  Complex.abs ((x / y) + Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l316_31623


namespace NUMINAMATH_CALUDE_bus_seat_capacity_l316_31602

theorem bus_seat_capacity :
  let left_seats : ℕ := 15
  let right_seats : ℕ := left_seats - 3
  let back_seat_capacity : ℕ := 12
  let total_capacity : ℕ := 93
  let seat_capacity : ℕ := (total_capacity - back_seat_capacity) / (left_seats + right_seats)
  seat_capacity = 3 := by sorry

end NUMINAMATH_CALUDE_bus_seat_capacity_l316_31602


namespace NUMINAMATH_CALUDE_particle_probability_l316_31647

def move_probability (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (1/3) * move_probability (x-1) y +
       (1/3) * move_probability x (y-1) +
       (1/3) * move_probability (x-1) (y-1)

theorem particle_probability :
  move_probability 4 4 = 245 / 3^7 :=
sorry

end NUMINAMATH_CALUDE_particle_probability_l316_31647


namespace NUMINAMATH_CALUDE_triangle_perimeter_proof_l316_31675

theorem triangle_perimeter_proof :
  ∀ a : ℕ,
  a % 2 = 0 →
  2 < a →
  a < 14 →
  6 + 8 + a = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_proof_l316_31675


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l316_31657

/-- Given a circle with center O and radius r, where arc PQ is half the circle,
    the perimeter of the shaded region formed by OP, OQ, and arc PQ is 2r + πr. -/
theorem shaded_region_perimeter (r : ℝ) (h : r > 0) : 
  2 * r + π * r = 2 * r + (1 / 2) * (2 * π * r) := by sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l316_31657


namespace NUMINAMATH_CALUDE_min_value_theorem_l316_31635

def f (x : ℝ) := |x - 2| + |x + 1|

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_min : ∀ x, f x ≥ m + n) (h_exists : ∃ x, f x = m + n) :
  (∃ x, f x = 3) ∧ 
  (m^2 + n^2 ≥ 9/2) ∧
  (m^2 + n^2 = 9/2 ↔ m = 3/2 ∧ n = 3/2) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l316_31635


namespace NUMINAMATH_CALUDE_bridge_length_calculation_bridge_length_proof_l316_31665

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * time_to_pass
  let bridge_length := total_distance - train_length
  bridge_length

/-- Proves that the bridge length is approximately 140 meters --/
theorem bridge_length_proof :
  let train_length : ℝ := 360
  let train_speed_kmh : ℝ := 56
  let time_to_pass : ℝ := 32.142857142857146
  let calculated_bridge_length := bridge_length_calculation train_length train_speed_kmh time_to_pass
  ∃ ε > 0, |calculated_bridge_length - 140| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_bridge_length_proof_l316_31665


namespace NUMINAMATH_CALUDE_shifted_sine_equals_cosine_l316_31606

theorem shifted_sine_equals_cosine (φ : Real) (h : 0 < φ ∧ φ < π) :
  (∀ x, 2 * Real.sin (2 * x - π / 3 + φ) = 2 * Real.cos (2 * x)) ↔ φ = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_shifted_sine_equals_cosine_l316_31606


namespace NUMINAMATH_CALUDE_right_triangles_with_perimeter_equal_area_l316_31695

/-- A right triangle with integer side lengths. -/
structure RightTriangle where
  a : ℕ  -- First leg
  b : ℕ  -- Second leg
  c : ℕ  -- Hypotenuse
  right_angle : a^2 + b^2 = c^2

/-- The perimeter of a right triangle. -/
def perimeter (t : RightTriangle) : ℕ :=
  t.a + t.b + t.c

/-- The area of a right triangle. -/
def area (t : RightTriangle) : ℕ :=
  t.a * t.b / 2

/-- The property that the perimeter equals the area. -/
def perimeter_equals_area (t : RightTriangle) : Prop :=
  perimeter t = area t

theorem right_triangles_with_perimeter_equal_area :
  {t : RightTriangle | perimeter_equals_area t} =
  {⟨5, 12, 13, by sorry⟩, ⟨6, 8, 10, by sorry⟩} :=
by sorry

end NUMINAMATH_CALUDE_right_triangles_with_perimeter_equal_area_l316_31695


namespace NUMINAMATH_CALUDE_no_squarish_numbers_l316_31615

/-- A number is squarish if it satisfies all the given conditions -/
def is_squarish (n : ℕ) : Prop :=
  -- Six-digit number
  100000 ≤ n ∧ n < 1000000 ∧
  -- Each digit between 1 and 8
  (∀ d, d ∈ n.digits 10 → 1 ≤ d ∧ d ≤ 8) ∧
  -- Perfect square
  ∃ x, n = x^2 ∧
  -- First two digits are a perfect square
  ∃ y, (n / 10000) = y^2 ∧
  -- Middle two digits are a perfect square and divisible by 2
  ∃ z, ((n / 100) % 100) = z^2 ∧ ((n / 100) % 100) % 2 = 0 ∧
  -- Last two digits are a perfect square
  ∃ w, (n % 100) = w^2

theorem no_squarish_numbers : ¬∃ n, is_squarish n := by
  sorry

end NUMINAMATH_CALUDE_no_squarish_numbers_l316_31615


namespace NUMINAMATH_CALUDE_melanie_dimes_count_l316_31641

/-- The total number of dimes Melanie has after receiving gifts from her parents -/
def total_dimes (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) : ℕ :=
  initial + from_dad + from_mom

/-- Theorem: Given Melanie's initial dimes and the dimes given by her parents, 
    the total number of dimes Melanie has now is 83. -/
theorem melanie_dimes_count : total_dimes 19 39 25 = 83 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_count_l316_31641


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l316_31690

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 * x + 11) = 9 → x = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l316_31690


namespace NUMINAMATH_CALUDE_equals_2022_l316_31620

theorem equals_2022 : 1 - (-2021) = 2022 := by
  sorry

end NUMINAMATH_CALUDE_equals_2022_l316_31620


namespace NUMINAMATH_CALUDE_fourth_term_of_sequence_l316_31624

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem fourth_term_of_sequence (a : ℕ → ℝ) 
    (h1 : a 1 = 2)
    (h2 : ∀ n : ℕ, a (n + 1) = 3 * a n) :
  a 4 = 54 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_sequence_l316_31624


namespace NUMINAMATH_CALUDE_triangle_median_theorem_l316_31637

/-- Represents a triangle with given side lengths and medians -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ
  area : ℝ

/-- The theorem stating the properties of the specific triangle -/
theorem triangle_median_theorem (t : Triangle) 
  (h1 : t.a = 5)
  (h2 : t.b = 8)
  (h3 : t.m_a = 4)
  (h4 : t.area = 12) :
  t.m_b = 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_median_theorem_l316_31637


namespace NUMINAMATH_CALUDE_zoo_recovery_time_l316_31610

/-- The total time spent recovering escaped animals from a zoo. -/
def total_recovery_time (lions rhinos recovery_time_per_animal : ℕ) : ℕ :=
  (lions + rhinos) * recovery_time_per_animal

/-- Theorem stating that given 3 lions, 2 rhinos, and 2 hours recovery time per animal,
    the total recovery time is 10 hours. -/
theorem zoo_recovery_time :
  total_recovery_time 3 2 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_zoo_recovery_time_l316_31610


namespace NUMINAMATH_CALUDE_solution_set_inequality_inequality_with_parameter_l316_31688

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part I
theorem solution_set_inequality (x : ℝ) :
  (f x + f (x - 1) ≤ 2) ↔ (1/2 ≤ x ∧ x ≤ 5/2) :=
sorry

-- Theorem for part II
theorem inequality_with_parameter (a x : ℝ) (h : a > 0) :
  f (a * x) - a * f x ≤ f a :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_inequality_with_parameter_l316_31688


namespace NUMINAMATH_CALUDE_only_proposition2_correct_l316_31692

-- Define the propositions
def proposition1 : Prop := ∀ (p q : Prop), (p ∧ q) → (p ∨ q) ∧ ¬((p ∨ q) → (p ∧ q))

def proposition2 : Prop :=
  let p := ∃ x : ℝ, x^2 + 2*x ≤ 0
  let not_p := ∀ x : ℝ, x^2 + 2*x > 0
  (¬p) ↔ not_p

def proposition3 : Prop := ∀ (p q : Prop), p ∧ ¬q → (p ∧ ¬q) ∧ (¬p ∨ q)

def proposition4 : Prop := ∀ (p q : Prop), (¬p → q) ↔ (p → ¬q)

-- Theorem stating that only proposition2 is correct
theorem only_proposition2_correct :
  ¬proposition1 ∧ proposition2 ∧ ¬proposition3 ∧ ¬proposition4 :=
sorry

end NUMINAMATH_CALUDE_only_proposition2_correct_l316_31692


namespace NUMINAMATH_CALUDE_not_hyperbola_equation_l316_31604

/-- A hyperbola with given properties -/
structure Hyperbola where
  center_at_origin : Bool
  symmetric_about_axes : Bool
  eccentricity : ℝ
  focus_to_asymptote_distance : ℝ

/-- The equation of a hyperbola -/
def hyperbola_equation (a b : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x^2 / a - y^2 / b = 1

/-- Theorem stating that the given equation cannot be the equation of the hyperbola with the specified properties -/
theorem not_hyperbola_equation (M : Hyperbola) 
  (h1 : M.center_at_origin = true)
  (h2 : M.symmetric_about_axes = true)
  (h3 : M.eccentricity = Real.sqrt 3)
  (h4 : M.focus_to_asymptote_distance = 2) :
  ¬(hyperbola_equation 4 2 = fun x y => x^2 / 4 - y^2 / 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_not_hyperbola_equation_l316_31604


namespace NUMINAMATH_CALUDE_subtraction_addition_equality_l316_31660

theorem subtraction_addition_equality : ∃ x : ℤ, 100 - 70 = 70 + x ∧ x = -40 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_addition_equality_l316_31660


namespace NUMINAMATH_CALUDE_cyclists_problem_l316_31614

/-- Two cyclists problem -/
theorem cyclists_problem (x : ℝ) 
  (h1 : x > 0) -- Distance between A and B is positive
  (h2 : 70 + x + 90 = 3 * 70) -- Equation derived from the problem conditions
  : x = 120 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_problem_l316_31614


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l316_31671

/-- A quadratic equation with roots 1 and -2 -/
def quadratic_equation (x : ℝ) : Prop :=
  x^2 + x - 2 = 0

theorem roots_of_quadratic : 
  (quadratic_equation 1) ∧ (quadratic_equation (-2)) := by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l316_31671


namespace NUMINAMATH_CALUDE_reflection_changes_color_l316_31605

/-- Determines if a number is red (can be expressed as 81x + 100y for positive integers x and y) -/
def isRed (n : ℤ) : Prop :=
  ∃ x y : ℕ+, n = 81 * x + 100 * y

/-- The point P -/
def P : ℤ := 3960

/-- Reflects a point T with respect to P -/
def reflect (T : ℤ) : ℤ := 2 * P - T

theorem reflection_changes_color :
  ∀ T : ℤ, isRed T ≠ isRed (reflect T) :=
sorry

end NUMINAMATH_CALUDE_reflection_changes_color_l316_31605


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l316_31661

def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

def possible_b_values : Set ℝ := {-2, -1, 0, 2}

theorem point_in_second_quadrant (b : ℝ) :
  is_in_second_quadrant (-3) b ∧ b ∈ possible_b_values → b = 2 :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l316_31661


namespace NUMINAMATH_CALUDE_rbc_divisibility_l316_31669

theorem rbc_divisibility (r b c : ℕ) : 
  r < 10 → b < 10 → c < 10 →
  (523 * 100 + r * 10 + b) * 10 + c ≡ 0 [MOD 7] →
  (523 * 100 + r * 10 + b) * 10 + c ≡ 0 [MOD 89] →
  r * b * c = 36 := by
sorry

end NUMINAMATH_CALUDE_rbc_divisibility_l316_31669


namespace NUMINAMATH_CALUDE_quadratic_value_range_l316_31676

-- Define the set of x that satisfies the inequality
def S : Set ℝ := {x : ℝ | x^2 - 7*x + 12 < 0}

-- State the theorem
theorem quadratic_value_range : 
  ∀ x ∈ S, 0 < x^2 - 5*x + 6 ∧ x^2 - 5*x + 6 < 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_value_range_l316_31676


namespace NUMINAMATH_CALUDE_equilateral_triangle_rotation_volume_l316_31622

/-- The volume of a solid obtained by rotating an equilateral triangle around a line parallel to its altitude -/
theorem equilateral_triangle_rotation_volume (a : ℝ) (ha : a > 0) :
  let h := a * Real.sqrt 3 / 2
  let r := a / 2
  (1 / 3) * Real.pi * r^2 * h = (Real.pi * a^3 * Real.sqrt 3) / 24 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_rotation_volume_l316_31622


namespace NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_sq_eq_one_l316_31670

theorem x_eq_one_sufficient_not_necessary_for_x_sq_eq_one :
  (∀ x : ℝ, x = 1 → x^2 = 1) ∧
  (∃ x : ℝ, x ≠ 1 ∧ x^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_sq_eq_one_l316_31670


namespace NUMINAMATH_CALUDE_initial_ribbon_amount_l316_31684

/-- The number of gifts Josh is preparing -/
def num_gifts : ℕ := 6

/-- The amount of ribbon used for each gift in yards -/
def ribbon_per_gift : ℕ := 2

/-- The amount of ribbon left after preparing the gifts in yards -/
def leftover_ribbon : ℕ := 6

/-- Theorem: Josh initially has 18 yards of ribbon -/
theorem initial_ribbon_amount :
  num_gifts * ribbon_per_gift + leftover_ribbon = 18 := by
  sorry

end NUMINAMATH_CALUDE_initial_ribbon_amount_l316_31684


namespace NUMINAMATH_CALUDE_onion_weight_problem_l316_31680

/-- Given 40 onions weighing 7.68 kg, and 35 of these onions having an average weight of 190 grams,
    the average weight of the remaining 5 onions is 206 grams. -/
theorem onion_weight_problem (total_weight : Real) (remaining_avg : Real) :
  total_weight = 7.68 →
  remaining_avg = 190 →
  (total_weight * 1000 - 35 * remaining_avg) / 5 = 206 := by
sorry

end NUMINAMATH_CALUDE_onion_weight_problem_l316_31680


namespace NUMINAMATH_CALUDE_cubic_equation_natural_roots_l316_31674

theorem cubic_equation_natural_roots (p : ℝ) : 
  (∃ x y : ℕ, x ≠ y ∧ 
    5 * (x : ℝ)^3 - 5*(p+1)*(x : ℝ)^2 + (71*p-1)*(x : ℝ) + 1 = 66*p ∧
    5 * (y : ℝ)^3 - 5*(p+1)*(y : ℝ)^2 + (71*p-1)*(y : ℝ) + 1 = 66*p) 
  ↔ p = 76 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_natural_roots_l316_31674


namespace NUMINAMATH_CALUDE_min_value_x2_plus_2y2_l316_31663

theorem min_value_x2_plus_2y2 (x y : ℝ) (h : x^2 - 2*x*y + 2*y^2 = 2) :
  ∃ (m : ℝ), (∀ (a b : ℝ), a^2 - 2*a*b + 2*b^2 = 2 → a^2 + 2*b^2 ≥ m) ∧
             (∃ (c d : ℝ), c^2 - 2*c*d + 2*d^2 = 2 ∧ c^2 + 2*d^2 = m) ∧
             (m = 4 - 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_x2_plus_2y2_l316_31663


namespace NUMINAMATH_CALUDE_equation_system_solution_l316_31691

theorem equation_system_solution :
  ∀ (x y z : ℝ),
    z ≠ 0 →
    3 * x - 5 * y - z = 0 →
    2 * x + 4 * y - 16 * z = 0 →
    (x^2 + 4*x*y) / (2*y^2 + z^2) = 4.35 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l316_31691


namespace NUMINAMATH_CALUDE_max_value_of_expression_l316_31654

theorem max_value_of_expression (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 3) : 
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 729/108 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l316_31654


namespace NUMINAMATH_CALUDE_product_factors_count_l316_31608

/-- A natural number with exactly three factors is a perfect square of a prime. -/
def has_three_factors (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p^2

/-- The main theorem statement -/
theorem product_factors_count
  (a b c d : ℕ)
  (ha : has_three_factors a)
  (hb : has_three_factors b)
  (hc : has_three_factors c)
  (hd : has_three_factors d)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d)
  (hbc : b ≠ c) (hbd : b ≠ d)
  (hcd : c ≠ d) :
  (Nat.factors (a^3 * b^2 * c^4 * d^5)).length = 3465 :=
sorry

end NUMINAMATH_CALUDE_product_factors_count_l316_31608


namespace NUMINAMATH_CALUDE_selling_price_calculation_l316_31609

def cost_price : ℝ := 225
def profit_percentage : ℝ := 20

theorem selling_price_calculation :
  let profit := (profit_percentage / 100) * cost_price
  let selling_price := cost_price + profit
  selling_price = 270 := by
sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l316_31609


namespace NUMINAMATH_CALUDE_transformed_system_solution_l316_31667

theorem transformed_system_solution 
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) 
  (h₁ : a₁ * 3 - b₁ * 5 = c₁) 
  (h₂ : a₂ * 3 + b₂ * 5 = c₂) :
  a₁ * (11 - 2) - b₁ * 15 = 3 * c₁ ∧ 
  a₂ * (11 - 2) + b₂ * 15 = 3 * c₂ := by
  sorry

end NUMINAMATH_CALUDE_transformed_system_solution_l316_31667


namespace NUMINAMATH_CALUDE_train_length_l316_31601

/-- The length of a train given its speed and time to pass a fixed point -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 50 → -- speed in km/h
  time = 5.399568034557236 → -- time in seconds
  length = speed * 1000 / 3600 * time → -- length in meters
  length = 75 := by sorry

end NUMINAMATH_CALUDE_train_length_l316_31601


namespace NUMINAMATH_CALUDE_function_equation_implies_identity_l316_31686

/-- A function satisfying the given functional equation is the identity function. -/
theorem function_equation_implies_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f y + x^2 + 1) + 2*x = y + (f (x + 1))^2) : 
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_function_equation_implies_identity_l316_31686


namespace NUMINAMATH_CALUDE_mans_swimming_speed_l316_31613

/-- The speed of the stream in km/h -/
def stream_speed : ℝ := 1.6666666666666667

/-- Proves that the man's swimming speed in still water is 5 km/h -/
theorem mans_swimming_speed (t : ℝ) (h : t > 0) : 
  let downstream_time := t
  let upstream_time := 2 * t
  let mans_speed : ℝ := stream_speed * 3
  upstream_time * (mans_speed - stream_speed) = downstream_time * (mans_speed + stream_speed) →
  mans_speed = 5 := by
sorry

end NUMINAMATH_CALUDE_mans_swimming_speed_l316_31613


namespace NUMINAMATH_CALUDE_correct_replacement_l316_31600

/-- Represents a digit in the addition problem -/
inductive Digit
| zero | one | two | three | four | five | six | seven | eight | nine

/-- Represents whether a digit is correct or potentially incorrect -/
inductive DigitStatus
| correct
| potentiallyIncorrect

/-- Function to get the status of a digit -/
def digitStatus (d : Digit) : DigitStatus :=
  match d with
  | Digit.zero | Digit.one | Digit.three | Digit.four | Digit.five | Digit.six | Digit.eight => DigitStatus.correct
  | Digit.two | Digit.seven | Digit.nine => DigitStatus.potentiallyIncorrect

/-- Function to check if replacing a digit corrects the addition -/
def replacementCorrects (d : Digit) (replacement : Digit) : Prop :=
  match d, replacement with
  | Digit.two, Digit.six => True
  | _, _ => False

/-- Theorem stating that replacing 2 with 6 corrects the addition -/
theorem correct_replacement :
  ∃ (d : Digit) (replacement : Digit),
    digitStatus d = DigitStatus.potentiallyIncorrect ∧
    digitStatus replacement = DigitStatus.correct ∧
    replacementCorrects d replacement :=
by sorry

end NUMINAMATH_CALUDE_correct_replacement_l316_31600


namespace NUMINAMATH_CALUDE_abc_sum_product_bound_l316_31687

theorem abc_sum_product_bound (a b c : ℝ) (h : a + b + c = 1) :
  ∃ (M : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (a' b' c' : ℝ),
    a' + b' + c' = 1 ∧ 
    ab + ac + bc ≤ 1/2 ∧
    a' * b' + a' * c' + b' * c' < -M + ε :=
sorry

end NUMINAMATH_CALUDE_abc_sum_product_bound_l316_31687


namespace NUMINAMATH_CALUDE_max_altitude_product_l316_31650

/-- 
Given a triangle ABC with base AB = 1 and altitude from C of length h,
this theorem states the maximum product of the three altitudes and the
triangle configuration that achieves it.
-/
theorem max_altitude_product (h : ℝ) (h_pos : h > 0) :
  let max_product := if h ≤ 1/2 then h^2 else h^3 / (h^2 + 1/4)
  let optimal_triangle := if h ≤ 1/2 then "right triangle at C" else "isosceles triangle with AC = BC"
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    (a * b * h ≤ max_product ∧
    (a * b * h = max_product ↔ 
      (h ≤ 1/2 ∧ c^2 = a^2 + b^2) ∨ 
      (h > 1/2 ∧ a = b))) :=
by sorry


end NUMINAMATH_CALUDE_max_altitude_product_l316_31650


namespace NUMINAMATH_CALUDE_function_equation_implies_linear_l316_31651

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x * f x - y * f y = (x - y) * f (x + y)

/-- The theorem stating that any function satisfying the equation must be linear -/
theorem function_equation_implies_linear (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b := by
  sorry


end NUMINAMATH_CALUDE_function_equation_implies_linear_l316_31651


namespace NUMINAMATH_CALUDE_factor_implies_b_equals_one_l316_31696

theorem factor_implies_b_equals_one (a b : ℤ) :
  (∃ c d : ℤ, ∀ x, (x^2 + x - 2) * (c*x + d) = a*x^3 - b*x^2 + x + 2) →
  b = 1 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_b_equals_one_l316_31696


namespace NUMINAMATH_CALUDE_train_crossing_time_l316_31658

/-- Time for a train to cross another train moving in the same direction -/
theorem train_crossing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 420)
  (h2 : length2 = 640)
  (h3 : speed1 = 72)
  (h4 : speed2 = 36) :
  (length1 + length2) / ((speed1 - speed2) * (1000 / 3600)) = 106 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l316_31658


namespace NUMINAMATH_CALUDE_counterexample_exists_l316_31642

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l316_31642


namespace NUMINAMATH_CALUDE_alyssa_grape_cost_l316_31628

/-- The amount Alyssa paid for cherries in dollars -/
def cherry_cost : ℚ := 9.85

/-- The total amount Alyssa spent in dollars -/
def total_spent : ℚ := 21.93

/-- The amount Alyssa paid for grapes in dollars -/
def grape_cost : ℚ := total_spent - cherry_cost

theorem alyssa_grape_cost : grape_cost = 12.08 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_grape_cost_l316_31628


namespace NUMINAMATH_CALUDE_smallest_common_multiple_tutors_smallest_group_l316_31656

theorem smallest_common_multiple (n : ℕ) : n > 0 ∧ n % 14 = 0 ∧ n % 10 = 0 ∧ n % 15 = 0 → n ≥ 210 := by
  sorry

theorem tutors_smallest_group : ∃ (n : ℕ), n > 0 ∧ n % 14 = 0 ∧ n % 10 = 0 ∧ n % 15 = 0 ∧ n = 210 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_tutors_smallest_group_l316_31656


namespace NUMINAMATH_CALUDE_hundredth_odd_and_plus_ten_l316_31677

/-- The nth odd positive integer -/
def nthOddPositive (n : ℕ) : ℕ := 2 * n - 1

theorem hundredth_odd_and_plus_ten :
  (nthOddPositive 100 = 199) ∧ (nthOddPositive 100 + 10 = 209) := by
  sorry

end NUMINAMATH_CALUDE_hundredth_odd_and_plus_ten_l316_31677


namespace NUMINAMATH_CALUDE_caterpillar_insane_bill_sane_l316_31617

-- Define the mental state of a character
inductive MentalState
| Sane
| Insane

-- Define the characters
structure Character where
  name : String
  state : MentalState

-- Define the Caterpillar's belief
def caterpillarBelief (caterpillar : Character) (bill : Character) : Prop :=
  caterpillar.state = MentalState.Insane ∧ bill.state = MentalState.Insane

-- Theorem statement
theorem caterpillar_insane_bill_sane 
  (caterpillar : Character) 
  (bill : Character) 
  (h : caterpillarBelief caterpillar bill) : 
  caterpillar.state = MentalState.Insane ∧ bill.state = MentalState.Sane :=
sorry

end NUMINAMATH_CALUDE_caterpillar_insane_bill_sane_l316_31617


namespace NUMINAMATH_CALUDE_monotonicity_of_f_l316_31634

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.exp x + a * (x - 1)^2

theorem monotonicity_of_f (a : ℝ) :
  (∀ x y, x < 1 → y < 1 → x < y → f a x > f a y) ∧ 
  (∀ x y, x > 1 → y > 1 → x < y → f a x < f a y) ∨
  (a = -Real.exp 1 / 2 ∧ ∀ x y, x < y → f a x < f a y) ∨
  (a < -Real.exp 1 / 2 ∧ 
    (∀ x y, x < 1 → y < 1 → x < y → f a x < f a y) ∧
    (∀ x y, 1 < x → x < Real.log (-2*a) → 1 < y → y < Real.log (-2*a) → x < y → f a x > f a y) ∧
    (∀ x y, x > Real.log (-2*a) → y > Real.log (-2*a) → x < y → f a x < f a y)) ∨
  (-Real.exp 1 / 2 < a ∧ a < 0 ∧
    (∀ x y, x < Real.log (-2*a) → y < Real.log (-2*a) → x < y → f a x < f a y) ∧
    (∀ x y, Real.log (-2*a) < x → x < 1 → Real.log (-2*a) < y → y < 1 → x < y → f a x > f a y) ∧
    (∀ x y, x > 1 → y > 1 → x < y → f a x < f a y)) :=
sorry

end

end NUMINAMATH_CALUDE_monotonicity_of_f_l316_31634


namespace NUMINAMATH_CALUDE_cubic_sum_inequality_l316_31631

theorem cubic_sum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a^3 + b^3 = 2) :
  ((a + b) * (a^5 + b^5) ≥ 4) ∧ (a + b ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_inequality_l316_31631


namespace NUMINAMATH_CALUDE_total_weight_loss_l316_31648

def weight_loss_problem (seth_loss jerome_loss veronica_loss total_loss : ℝ) : Prop :=
  seth_loss = 17.5 ∧
  jerome_loss = 3 * seth_loss ∧
  veronica_loss = seth_loss + 1.5 ∧
  total_loss = seth_loss + jerome_loss + veronica_loss

theorem total_weight_loss :
  ∃ (seth_loss jerome_loss veronica_loss total_loss : ℝ),
    weight_loss_problem seth_loss jerome_loss veronica_loss total_loss ∧
    total_loss = 89 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_loss_l316_31648


namespace NUMINAMATH_CALUDE_shaded_area_of_partitioned_triangle_l316_31633

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  leg_length_pos : leg_length > 0

/-- Represents a partition of a triangle -/
structure TrianglePartition where
  num_parts : ℕ
  num_parts_pos : num_parts > 0

theorem shaded_area_of_partitioned_triangle
  (t : IsoscelesRightTriangle)
  (p : TrianglePartition)
  (h1 : t.leg_length = 10)
  (h2 : p.num_parts = 25)
  (num_shaded : ℕ)
  (h3 : num_shaded = 15) :
  num_shaded * (t.leg_length^2 / 2) / p.num_parts = 30 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_partitioned_triangle_l316_31633


namespace NUMINAMATH_CALUDE_total_sightings_first_quarter_l316_31649

/-- The total number of animal sightings in the first three months of the year. -/
def total_sightings (january_sightings : ℕ) : ℕ :=
  let february_sightings := 3 * january_sightings
  let march_sightings := february_sightings / 2
  january_sightings + february_sightings + march_sightings

/-- Theorem stating that the total number of animal sightings in the first three months is 143,
    given that there were 26 sightings in January. -/
theorem total_sightings_first_quarter (h : total_sightings 26 = 143) : total_sightings 26 = 143 := by
  sorry

end NUMINAMATH_CALUDE_total_sightings_first_quarter_l316_31649


namespace NUMINAMATH_CALUDE_marys_max_earnings_l316_31681

/-- Calculates the maximum weekly earnings for a worker with the given parameters. -/
def max_weekly_earnings (max_hours : ℕ) (regular_hours : ℕ) (regular_rate : ℚ) (overtime_rate_increase : ℚ) : ℚ :=
  let overtime_hours := max_hours - regular_hours
  let overtime_rate := regular_rate * (1 + overtime_rate_increase)
  regular_rate * regular_hours + overtime_rate * overtime_hours

/-- Theorem stating that Mary's maximum weekly earnings are $410 -/
theorem marys_max_earnings :
  max_weekly_earnings 45 20 8 (1/4) = 410 := by
  sorry

#eval max_weekly_earnings 45 20 8 (1/4)

end NUMINAMATH_CALUDE_marys_max_earnings_l316_31681


namespace NUMINAMATH_CALUDE_bee_swarm_puzzle_l316_31653

theorem bee_swarm_puzzle :
  ∃ (x : ℚ),
    x > 0 ∧
    (x / 5 + x / 3 + 3 * (x / 3 - x / 5) + 1 = x) ∧
    x = 15 :=
by sorry

end NUMINAMATH_CALUDE_bee_swarm_puzzle_l316_31653


namespace NUMINAMATH_CALUDE_bus_problem_l316_31640

theorem bus_problem (initial : ℕ) (got_on : ℕ) (final : ℕ) : 
  initial = 28 → got_on = 82 → final = 30 → 
  ∃ (got_off : ℕ), got_on - got_off = 2 ∧ initial + got_on - got_off = final :=
by sorry

end NUMINAMATH_CALUDE_bus_problem_l316_31640


namespace NUMINAMATH_CALUDE_min_distance_sum_l316_31603

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 7 = 1

-- Define the left focus F
def left_focus : ℝ × ℝ := (-4, 0)

-- Define the fixed point A
def point_A : ℝ × ℝ := (1, 4)

-- Define a point on the right branch of the hyperbola
def is_on_right_branch (P : ℝ × ℝ) : Prop :=
  is_on_hyperbola P.1 P.2 ∧ P.1 > 0

-- Theorem statement
theorem min_distance_sum (P : ℝ × ℝ) (h : is_on_right_branch P) :
  dist P left_focus + dist P point_A ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_min_distance_sum_l316_31603


namespace NUMINAMATH_CALUDE_tank_fill_time_l316_31645

/-- Represents the time (in minutes) it takes for a pipe to fill or empty the tank -/
structure PipeRate where
  rate : ℚ
  filling : Bool

/-- Represents the state of the tank -/
structure TankState where
  filled : ℚ  -- Fraction of the tank that is filled

/-- Represents the system of pipes and the tank -/
structure PipeSystem where
  pipes : Fin 4 → PipeRate
  cycle_time : ℚ
  cycle_effect : ℚ

def apply_pipe (p : PipeRate) (t : TankState) (duration : ℚ) : TankState :=
  if p.filling then
    { filled := t.filled + duration / p.rate }
  else
    { filled := t.filled - duration / p.rate }

def apply_cycle (s : PipeSystem) (t : TankState) : TankState :=
  { filled := t.filled + s.cycle_effect }

def time_to_fill (s : PipeSystem) : ℚ :=
  s.cycle_time * (1 / s.cycle_effect)

theorem tank_fill_time (s : PipeSystem) (h1 : s.pipes 0 = ⟨20, true⟩)
    (h2 : s.pipes 1 = ⟨30, true⟩) (h3 : s.pipes 2 = ⟨15, false⟩)
    (h4 : s.pipes 3 = ⟨40, true⟩) (h5 : s.cycle_time = 16)
    (h6 : s.cycle_effect = 1/10) : time_to_fill s = 160 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_l316_31645


namespace NUMINAMATH_CALUDE_corrected_mean_problem_l316_31630

/-- Calculates the corrected mean of observations after fixing an error --/
def corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * original_mean + (correct_value - incorrect_value)) / n

/-- Theorem stating the corrected mean for the given problem --/
theorem corrected_mean_problem :
  corrected_mean 50 36 23 34 = 36.22 := by
  sorry

#eval corrected_mean 50 36 23 34

end NUMINAMATH_CALUDE_corrected_mean_problem_l316_31630


namespace NUMINAMATH_CALUDE_square_side_length_l316_31689

theorem square_side_length (area : ℝ) (side : ℝ) (h1 : area = 81) (h2 : area = side ^ 2) :
  side = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l316_31689


namespace NUMINAMATH_CALUDE_perpendicular_lines_l316_31679

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, ax + 2*y + 6 = 0 ∧ x + a*(a+1)*y + (a^2-1) = 0 → 
    (a = -3/2 ∨ a = 0)) ∧
  (a = -3/2 ∨ a = 0 → 
    ∀ x y : ℝ, ax + 2*y + 6 = 0 ∧ x + a*(a+1)*y + (a^2-1) = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l316_31679


namespace NUMINAMATH_CALUDE_arbitrary_sign_sum_odd_l316_31694

theorem arbitrary_sign_sum_odd (n : ℕ) (h : n = 2005) : 
  ∀ (f : ℕ → ℤ), (∀ i ∈ Finset.range n, f i = i + 1 ∨ f i = -(i + 1)) → 
  Odd (Finset.sum (Finset.range n) f) :=
by
  sorry

end NUMINAMATH_CALUDE_arbitrary_sign_sum_odd_l316_31694


namespace NUMINAMATH_CALUDE_min_y_value_l316_31652

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 8*x + 54*y) :
  ∃ (y_min : ℝ), y_min = 27 - Real.sqrt 745 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 = 8*x' + 54*y' → y' ≥ y_min :=
sorry

end NUMINAMATH_CALUDE_min_y_value_l316_31652


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l316_31638

theorem least_positive_integer_with_remainders : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (x % 2 = 1) ∧ 
  (x % 5 = 2) ∧ 
  (x % 6 = 3) ∧ 
  (∀ y : ℕ, y > 0 ∧ y % 2 = 1 ∧ y % 5 = 2 ∧ y % 6 = 3 → x ≤ y) ∧
  x = 27 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l316_31638


namespace NUMINAMATH_CALUDE_max_value_quadratic_swap_l316_31611

/-- Given real numbers a, b, and c where |ax^2 + bx + c| has a maximum value of 1 
    on the interval x ∈ [-1,1], the maximum possible value of |cx^2 + bx + a| 
    on the interval x ∈ [-1,1] is 2. -/
theorem max_value_quadratic_swap (a b c : ℝ) 
  (h : ∀ x ∈ Set.Icc (-1) 1, |a * x^2 + b * x + c| ≤ 1) :
  (⨆ x ∈ Set.Icc (-1) 1, |c * x^2 + b * x + a|) = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_swap_l316_31611


namespace NUMINAMATH_CALUDE_line_segment_sum_l316_31619

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = -3/4 * x + 9

/-- Point P is on the x-axis -/
def P : ℝ × ℝ := (12, 0)

/-- Point Q is on the y-axis -/
def Q : ℝ × ℝ := (0, 9)

/-- Point T is on the line segment PQ -/
def T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (r, s) = (1 - t) • P + t • Q

/-- The area of triangle POQ is three times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs ((P.1 * Q.2 - Q.1 * P.2) / 2) = 3 * abs ((P.1 * s - r * P.2) / 2)

/-- The main theorem -/
theorem line_segment_sum (r s : ℝ) :
  line_equation r s → T_on_PQ r s → area_condition r s → r + s = 11 := by sorry

end NUMINAMATH_CALUDE_line_segment_sum_l316_31619


namespace NUMINAMATH_CALUDE_total_days_1996_to_2000_l316_31612

/-- The number of days in a regular year -/
def regularYearDays : ℕ := 365

/-- The number of additional days in a leap year -/
def leapYearExtraDays : ℕ := 1

/-- The start year of our range -/
def startYear : ℕ := 1996

/-- The end year of our range -/
def endYear : ℕ := 2000

/-- The number of leap years in our range -/
def leapYearsCount : ℕ := 2

/-- Theorem: The total number of days from 1996 to 2000 (inclusive) is 1827 -/
theorem total_days_1996_to_2000 : 
  (endYear - startYear + 1) * regularYearDays + leapYearsCount * leapYearExtraDays = 1827 := by
  sorry

end NUMINAMATH_CALUDE_total_days_1996_to_2000_l316_31612


namespace NUMINAMATH_CALUDE_expand_and_simplify_l316_31682

theorem expand_and_simplify (x : ℝ) (h : x ≠ 0) :
  3 / 7 * (7 / x + 14 * x^3) = 3 / x + 6 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l316_31682


namespace NUMINAMATH_CALUDE_reflected_tetrahedron_volume_l316_31643

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  A : ℝ × ℝ × ℝ
  B : ℝ × ℝ × ℝ
  C : ℝ × ℝ × ℝ
  D : ℝ × ℝ × ℝ

/-- Calculates the volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- Reflects a point with respect to another point -/
def reflect (point center : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Calculates the centroid of a triangle -/
def centroid (a b c : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Creates a new tetrahedron by reflecting each vertex of the original tetrahedron
    with respect to the centroid of the opposite face -/
def reflectedTetrahedron (t : Tetrahedron) : Tetrahedron :=
  let A' := reflect t.A (centroid t.B t.C t.D)
  let B' := reflect t.B (centroid t.A t.C t.D)
  let C' := reflect t.C (centroid t.A t.B t.D)
  let D' := reflect t.D (centroid t.A t.B t.C)
  ⟨A', B', C', D'⟩

/-- Theorem: The volume of the reflected tetrahedron is 125/27 times the volume of the original tetrahedron -/
theorem reflected_tetrahedron_volume (t : Tetrahedron) :
  volume (reflectedTetrahedron t) = (125 / 27) * volume t := by sorry

end NUMINAMATH_CALUDE_reflected_tetrahedron_volume_l316_31643


namespace NUMINAMATH_CALUDE_count_distinct_cube_colorings_l316_31632

/-- The number of distinct colorings of a cube with six colors -/
def distinct_cube_colorings : ℕ := 30

/-- The number of faces on a cube -/
def cube_faces : ℕ := 6

/-- The number of rotational symmetries of a cube -/
def cube_rotations : ℕ := 24

/-- Theorem stating the number of distinct colorings of a cube -/
theorem count_distinct_cube_colorings :
  distinct_cube_colorings = (cube_faces * (cube_faces - 1) * (cube_faces - 2) / 2) := by
  sorry

#check count_distinct_cube_colorings

end NUMINAMATH_CALUDE_count_distinct_cube_colorings_l316_31632


namespace NUMINAMATH_CALUDE_sum_bounds_l316_31607

def A : Set ℕ := {n | n ≤ 2018}

theorem sum_bounds (x y z : ℕ) (hx : x ∈ A) (hy : y ∈ A) (hz : z ∈ A) 
  (h : x^2 + y^2 - z^2 = 2019^2) : 
  2181 ≤ x + y + z ∧ x + y + z ≤ 5781 := by
  sorry

end NUMINAMATH_CALUDE_sum_bounds_l316_31607


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l316_31655

/-- Given a quadratic inequality with specific properties, prove certain statements about its coefficients and solutions. -/
theorem quadratic_inequality_properties
  (a b : ℝ) (d : ℝ)
  (h_a_pos : a > 0)
  (h_solution_set : ∀ x : ℝ, x^2 + a*x + b > 0 ↔ x ≠ d) :
  (a^2 = 4*b) ∧
  (a^2 + 1/b ≥ 4) ∧
  (∀ c x₁ x₂ : ℝ, (∀ x : ℝ, x^2 + a*x + b < c ↔ x₁ < x ∧ x < x₂) →
    |x₁ - x₂| = 4 → c = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l316_31655


namespace NUMINAMATH_CALUDE_max_sundays_in_fifty_days_l316_31621

/-- The number of days in a week -/
def daysInWeek : ℕ := 7

/-- The number of days we're considering -/
def daysConsidered : ℕ := 50

/-- The maximum number of Sundays in the first 50 days of any year -/
def maxSundays : ℕ := daysConsidered / daysInWeek

theorem max_sundays_in_fifty_days :
  maxSundays = 7 := by sorry

end NUMINAMATH_CALUDE_max_sundays_in_fifty_days_l316_31621


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l316_31693

/-- An isosceles triangle with congruent sides of length 10 and perimeter 35 has a base of length 15 -/
theorem isosceles_triangle_base_length : ℝ → Prop :=
  fun base =>
    let congruentSide := (10 : ℝ)
    let perimeter := (35 : ℝ)
    (2 * congruentSide + base = perimeter) →
    (base = 15)

/-- Proof of the theorem -/
theorem isosceles_triangle_base_length_proof : isosceles_triangle_base_length 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l316_31693


namespace NUMINAMATH_CALUDE_nested_square_root_simplification_l316_31678

theorem nested_square_root_simplification :
  Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 25 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_simplification_l316_31678


namespace NUMINAMATH_CALUDE_simplify_fraction_l316_31685

theorem simplify_fraction (x : ℝ) (h : x > 0) :
  (Real.sqrt x * 3 * x^2) / (x * 6 * x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l316_31685


namespace NUMINAMATH_CALUDE_original_strip_length_is_57_l316_31698

/-- Represents the folded strip configuration -/
structure FoldedStrip where
  width : ℝ
  folded_length : ℝ
  trapezium_count : ℕ

/-- Calculates the length of the original strip before folding -/
def original_strip_length (fs : FoldedStrip) : ℝ :=
  sorry

/-- Theorem stating the length of the original strip -/
theorem original_strip_length_is_57 (fs : FoldedStrip) 
  (h_width : fs.width = 3)
  (h_folded_length : fs.folded_length = 27)
  (h_trapezium_count : fs.trapezium_count = 4) :
  original_strip_length fs = 57 :=
sorry

end NUMINAMATH_CALUDE_original_strip_length_is_57_l316_31698


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l316_31673

theorem triangle_angle_calculation (D E F : ℝ) : 
  D + E + F = 180 →  -- Sum of angles in a triangle is 180°
  E = F →            -- Angle E is congruent to Angle F
  F = 3 * D →        -- Angle F is three times Angle D
  E = 540 / 7 :=     -- Measure of Angle E is 540/7 degrees
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l316_31673


namespace NUMINAMATH_CALUDE_lawn_mowing_payment_l316_31636

theorem lawn_mowing_payment (payment_rate : ℚ) (lawns_mowed : ℚ) : 
  payment_rate = 13/3 → lawns_mowed = 11/4 → payment_rate * lawns_mowed = 143/12 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_payment_l316_31636


namespace NUMINAMATH_CALUDE_reflection_distance_C_l316_31666

/-- The length of the segment from a point to its reflection over the x-axis --/
def reflection_distance (p : ℝ × ℝ) : ℝ :=
  2 * |p.2|

theorem reflection_distance_C : reflection_distance (-3, 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_reflection_distance_C_l316_31666


namespace NUMINAMATH_CALUDE_escalator_length_l316_31627

/-- The length of an escalator given specific conditions -/
theorem escalator_length : 
  ∀ (escalator_speed person_speed time length : ℝ),
  escalator_speed = 12 →
  person_speed = 3 →
  time = 10 →
  length = (escalator_speed + person_speed) * time →
  length = 150 := by
sorry

end NUMINAMATH_CALUDE_escalator_length_l316_31627


namespace NUMINAMATH_CALUDE_total_cost_packages_A_and_B_l316_31639

/-- Represents a subscription package with monthly cost, duration, and discount rate -/
structure Package where
  monthlyCost : ℝ
  duration : ℕ
  discountRate : ℝ

/-- Calculates the discounted cost of a package -/
def discountedCost (p : Package) : ℝ :=
  p.monthlyCost * p.duration * (1 - p.discountRate)

/-- The newspaper subscription packages -/
def packageA : Package := { monthlyCost := 10, duration := 6, discountRate := 0.1 }
def packageB : Package := { monthlyCost := 12, duration := 9, discountRate := 0.15 }

/-- Theorem stating the total cost of subscribing to Package A followed by Package B -/
theorem total_cost_packages_A_and_B :
  discountedCost packageA + discountedCost packageB = 145.80 := by
  sorry

#eval discountedCost packageA + discountedCost packageB

end NUMINAMATH_CALUDE_total_cost_packages_A_and_B_l316_31639


namespace NUMINAMATH_CALUDE_contact_list_count_is_38_l316_31697

/-- The number of people on Jerome's contact list at the end of the month -/
def contact_list_count : ℕ :=
  let classmates : ℕ := 20
  let out_of_school_friends : ℕ := classmates / 2
  let immediate_family : ℕ := 3
  let added_contacts : ℕ := 5 + 7
  let removed_contacts : ℕ := 3 + 4
  classmates + out_of_school_friends + immediate_family + added_contacts - removed_contacts

/-- Theorem stating that the number of people on Jerome's contact list at the end of the month is 38 -/
theorem contact_list_count_is_38 : contact_list_count = 38 := by
  sorry

end NUMINAMATH_CALUDE_contact_list_count_is_38_l316_31697


namespace NUMINAMATH_CALUDE_problem_solution_l316_31668

theorem problem_solution (x : ℚ) : (2 * x + 10 - 2) / 7 = 15 → x = 97 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l316_31668


namespace NUMINAMATH_CALUDE_initial_observations_count_l316_31699

theorem initial_observations_count 
  (initial_avg : ℝ) 
  (new_obs : ℝ) 
  (avg_decrease : ℝ) 
  (h1 : initial_avg = 12)
  (h2 : new_obs = 5)
  (h3 : avg_decrease = 1) :
  ∃ n : ℕ, 
    (n : ℝ) * initial_avg = ((n : ℝ) + 1) * (initial_avg - avg_decrease) - new_obs ∧ 
    n = 6 :=
by sorry

end NUMINAMATH_CALUDE_initial_observations_count_l316_31699


namespace NUMINAMATH_CALUDE_max_cylinder_volume_in_cube_l316_31646

/-- The maximum volume of a cylinder inscribed in a cube with side length √3,
    where the cylinder's axis is along a diagonal of the cube. -/
theorem max_cylinder_volume_in_cube :
  let cube_side : ℝ := Real.sqrt 3
  let max_volume : ℝ := π / 2
  ∀ (cylinder_volume : ℝ),
    (∃ (cylinder_radius height : ℝ),
      cylinder_volume = π * cylinder_radius^2 * height ∧
      0 < cylinder_radius ∧
      0 < height ∧
      2 * Real.sqrt 2 * cylinder_radius + height = cube_side) →
    cylinder_volume ≤ max_volume :=
by sorry

end NUMINAMATH_CALUDE_max_cylinder_volume_in_cube_l316_31646


namespace NUMINAMATH_CALUDE_pirate_treasure_division_l316_31629

theorem pirate_treasure_division (N : ℕ) (h : 3000 ≤ N ∧ N ≤ 4000) :
  let remaining1 := (3 * N - 6) / 4
  let remaining2 := (9 * N - 42) / 16
  let remaining3 := (108 * N - 888) / 256
  let remaining4 := (82944 * N - 876400) / 262144
  let share1 := (N + 6) / 4
  let share2 := (3 * N + 18) / 16
  let share3 := (9 * N + 54) / 64
  let share4 := (108 * N + 648) / 1024
  let final_share := remaining4 / 4
  (share1 + final_share = 1178) ∧
  (share2 + final_share = 954) ∧
  (share3 + final_share = 786) ∧
  (share4 + final_share = 660) := by
sorry

end NUMINAMATH_CALUDE_pirate_treasure_division_l316_31629


namespace NUMINAMATH_CALUDE_difference_of_differences_l316_31664

theorem difference_of_differences (a b c : ℤ) 
  (hab : a - b = 2) (hbc : b - c = -3) : a - c = -1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_differences_l316_31664


namespace NUMINAMATH_CALUDE_always_in_range_l316_31659

theorem always_in_range (k : ℝ) : ∃ x : ℝ, x^2 + 2*k*x + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_always_in_range_l316_31659


namespace NUMINAMATH_CALUDE_oil_quantity_function_correct_l316_31644

/-- Represents the remaining oil quantity in a tank after a given time of outflow. -/
def Q (t : ℝ) : ℝ := 40 - 0.2 * t

/-- The initial quantity of oil in the tank. -/
def initial_quantity : ℝ := 40

/-- The rate at which oil flows out of the tank. -/
def flow_rate : ℝ := 0.2

theorem oil_quantity_function_correct (t : ℝ) (h : t ≥ 0) :
  Q t = initial_quantity - flow_rate * t :=
by sorry

end NUMINAMATH_CALUDE_oil_quantity_function_correct_l316_31644


namespace NUMINAMATH_CALUDE_bench_capacity_l316_31672

theorem bench_capacity (num_benches : ℕ) (people_sitting : ℕ) (spaces_available : ℕ) 
  (h1 : num_benches = 50)
  (h2 : people_sitting = 80)
  (h3 : spaces_available = 120) :
  (num_benches * 4 = people_sitting + spaces_available) ∧ 
  (4 = (people_sitting + spaces_available) / num_benches) :=
by sorry

end NUMINAMATH_CALUDE_bench_capacity_l316_31672


namespace NUMINAMATH_CALUDE_team_selection_ways_eq_103950_l316_31618

/-- The number of ways to select a team of 8 people, consisting of 4 boys from a group of 10 boys
    and 4 girls from a group of 12 girls. -/
def team_selection_ways : ℕ :=
  Nat.choose 10 4 * Nat.choose 12 4

/-- Theorem stating that the number of ways to select the team is 103950. -/
theorem team_selection_ways_eq_103950 : team_selection_ways = 103950 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_ways_eq_103950_l316_31618


namespace NUMINAMATH_CALUDE_matrix_addition_and_scalar_multiplication_l316_31625

theorem matrix_addition_and_scalar_multiplication :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 2, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-2, 1; 0, 3]
  A + 3 • B = !![-2, 0; 2, 14] := by sorry

end NUMINAMATH_CALUDE_matrix_addition_and_scalar_multiplication_l316_31625


namespace NUMINAMATH_CALUDE_complex_square_plus_self_l316_31626

theorem complex_square_plus_self : 
  let z : ℂ := 1 + Complex.I
  z^2 + z = 1 + 3 * Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_square_plus_self_l316_31626
