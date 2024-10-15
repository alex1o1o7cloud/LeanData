import Mathlib

namespace NUMINAMATH_CALUDE_unique_prime_divisor_l5_584

theorem unique_prime_divisor : 
  ∃! p : ℕ, p ≥ 5 ∧ Prime p ∧ (p ∣ (p + 3)^(p-3) + (p + 5)^(p-5)) ∧ p = 2813 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_divisor_l5_584


namespace NUMINAMATH_CALUDE_matrix_product_is_zero_l5_549

def matrix_product_zero (d e f : ℝ) : Prop :=
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![0, d, -e; -d, 0, f; e, -f, 0]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![d^2, d*e, d*f; d*e, e^2, e*f; d*f, e*f, f^2]
  A * B = 0

theorem matrix_product_is_zero (d e f : ℝ) : matrix_product_zero d e f := by
  sorry

end NUMINAMATH_CALUDE_matrix_product_is_zero_l5_549


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l5_552

theorem inequality_system_solution_set : 
  {x : ℝ | (5 - 2*x ≤ 1) ∧ (x - 4 < 0)} = {x : ℝ | 2 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l5_552


namespace NUMINAMATH_CALUDE_power_five_2023_mod_11_l5_509

theorem power_five_2023_mod_11 : 5^2023 ≡ 4 [ZMOD 11] := by sorry

end NUMINAMATH_CALUDE_power_five_2023_mod_11_l5_509


namespace NUMINAMATH_CALUDE_student_mistake_difference_l5_535

theorem student_mistake_difference : 
  let number := 384
  let correct_fraction := 5 / 16
  let incorrect_fraction := 5 / 6
  let correct_answer := correct_fraction * number
  let incorrect_answer := incorrect_fraction * number
  incorrect_answer - correct_answer = 200 := by
sorry

end NUMINAMATH_CALUDE_student_mistake_difference_l5_535


namespace NUMINAMATH_CALUDE_parabola_intercept_sum_l5_541

theorem parabola_intercept_sum : ∃ (a b c : ℝ),
  (∀ y : ℝ, 3 * y^2 - 9 * y + 5 = a ↔ y = 0) ∧
  (3 * b^2 - 9 * b + 5 = 0) ∧
  (3 * c^2 - 9 * c + 5 = 0) ∧
  (b ≠ c) ∧
  (a + b + c = 8) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intercept_sum_l5_541


namespace NUMINAMATH_CALUDE_base_4_last_digit_l5_543

theorem base_4_last_digit (n : ℕ) (h : n = 389) : n % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_base_4_last_digit_l5_543


namespace NUMINAMATH_CALUDE_bacteria_population_correct_l5_536

def bacteria_population (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    2^(n/2 + 1)
  else
    2^((n+1)/2)

theorem bacteria_population_correct :
  ∀ n : ℕ,
  (bacteria_population n = 2^(n/2 + 1) ∧ n % 2 = 0) ∨
  (bacteria_population n = 2^((n+1)/2) ∧ n % 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_bacteria_population_correct_l5_536


namespace NUMINAMATH_CALUDE_beanie_babies_total_l5_559

/-- The number of beanie babies Lori has -/
def lori_beanie_babies : ℕ := 300

/-- The number of beanie babies Sydney has -/
def sydney_beanie_babies : ℕ := lori_beanie_babies / 15

/-- The initial number of beanie babies Jake has -/
def jake_initial_beanie_babies : ℕ := 2 * sydney_beanie_babies

/-- The number of additional beanie babies Jake gained -/
def jake_additional_beanie_babies : ℕ := (jake_initial_beanie_babies * 20) / 100

/-- The total number of beanie babies Jake has after gaining more -/
def jake_total_beanie_babies : ℕ := jake_initial_beanie_babies + jake_additional_beanie_babies

/-- The total number of beanie babies all three have -/
def total_beanie_babies : ℕ := lori_beanie_babies + sydney_beanie_babies + jake_total_beanie_babies

theorem beanie_babies_total : total_beanie_babies = 368 := by
  sorry

end NUMINAMATH_CALUDE_beanie_babies_total_l5_559


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l5_569

theorem no_solution_for_equation : ¬∃ (x : ℝ), (x - 1) / (x - 3) = 2 - 2 / (3 - x) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l5_569


namespace NUMINAMATH_CALUDE_probability_4H_before_3T_is_4_57_l5_551

/-- The probability of encountering 4 heads before 3 consecutive tails in fair coin flips -/
def probability_4H_before_3T : ℚ :=
  4 / 57

/-- Theorem stating that the probability of encountering 4 heads before 3 consecutive tails
    in fair coin flips is equal to 4/57 -/
theorem probability_4H_before_3T_is_4_57 :
  probability_4H_before_3T = 4 / 57 := by
  sorry

end NUMINAMATH_CALUDE_probability_4H_before_3T_is_4_57_l5_551


namespace NUMINAMATH_CALUDE_dot_product_bounds_l5_520

theorem dot_product_bounds (a b : ℝ) :
  let v : ℝ × ℝ := (a, b)
  let u : ℝ → ℝ × ℝ := fun θ ↦ (Real.cos θ, Real.sin θ)
  ∀ θ, -Real.sqrt (a^2 + b^2) ≤ (v.1 * (u θ).1 + v.2 * (u θ).2) ∧
       (v.1 * (u θ).1 + v.2 * (u θ).2) ≤ Real.sqrt (a^2 + b^2) ∧
       (∃ θ₁, v.1 * (u θ₁).1 + v.2 * (u θ₁).2 = Real.sqrt (a^2 + b^2)) ∧
       (∃ θ₂, v.1 * (u θ₂).1 + v.2 * (u θ₂).2 = -Real.sqrt (a^2 + b^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_dot_product_bounds_l5_520


namespace NUMINAMATH_CALUDE_triangle_side_length_l5_560

theorem triangle_side_length 
  (A B C : ℝ) 
  (h1 : Real.cos (A - 2*B) + Real.sin (2*A + B) = 2)
  (h2 : 0 < A ∧ A < π)
  (h3 : 0 < B ∧ B < π)
  (h4 : 0 < C ∧ C < π)
  (h5 : A + B + C = π)
  (h6 : BC = 6) :
  AB = 3 * (Real.sqrt 5 + 1) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l5_560


namespace NUMINAMATH_CALUDE_prime_congruence_problem_l5_504

theorem prime_congruence_problem (p q : Nat) (n : Nat) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hn : n > 1)
  (hpOdd : Odd p) (hqOdd : Odd q)
  (hcong1 : q^(n+2) ≡ 3^(n+2) [MOD p^n])
  (hcong2 : p^(n+2) ≡ 3^(n+2) [MOD q^n]) :
  p = 3 ∧ q = 3 := by
sorry

end NUMINAMATH_CALUDE_prime_congruence_problem_l5_504


namespace NUMINAMATH_CALUDE_root_implies_m_value_l5_555

theorem root_implies_m_value (m : ℝ) : 
  (Complex.I + 1)^2 + m * (Complex.I + 1) + 2 = 0 → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_m_value_l5_555


namespace NUMINAMATH_CALUDE_complex_number_sum_parts_l5_522

theorem complex_number_sum_parts (a : ℝ) : 
  let z : ℂ := a / (2 - Complex.I) + (3 - 4 * Complex.I) / 5
  (z.re + z.im = 1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_sum_parts_l5_522


namespace NUMINAMATH_CALUDE_solve_equation_l5_590

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 5 / 3 → x = -27 / 2 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l5_590


namespace NUMINAMATH_CALUDE_max_area_difference_l5_529

/-- A rectangle with integer dimensions and perimeter 160 cm -/
structure Rectangle where
  length : ℕ
  width : ℕ
  perimeter_constraint : length + width = 80

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The theorem statement -/
theorem max_area_difference : 
  ∃ (r1 r2 : Rectangle), 
    (∀ r : Rectangle, area r ≤ area r1 ∧ area r ≥ area r2) ∧
    area r1 - area r2 = 1521 ∧
    r1.length = 40 ∧ r1.width = 40 ∧
    r2.length = 1 ∧ r2.width = 79 := by
  sorry


end NUMINAMATH_CALUDE_max_area_difference_l5_529


namespace NUMINAMATH_CALUDE_cone_slant_height_is_10_l5_594

/-- The slant height of a cone, given its base radius and that its lateral surface unfolds into a semicircle. -/
def slant_height (base_radius : ℝ) : ℝ :=
  2 * base_radius

theorem cone_slant_height_is_10 :
  let base_radius : ℝ := 5
  slant_height base_radius = 10 :=
by sorry

end NUMINAMATH_CALUDE_cone_slant_height_is_10_l5_594


namespace NUMINAMATH_CALUDE_man_lot_ownership_l5_572

theorem man_lot_ownership (lot_value : ℝ) (sold_fraction : ℝ) (sold_value : ℝ) :
  lot_value = 9200 →
  sold_fraction = 1 / 10 →
  sold_value = 460 →
  (sold_value / sold_fraction) / lot_value = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_man_lot_ownership_l5_572


namespace NUMINAMATH_CALUDE_log_ride_cost_l5_553

def ferris_wheel_cost : ℕ := 6
def roller_coaster_cost : ℕ := 5
def initial_tickets : ℕ := 2
def additional_tickets_needed : ℕ := 16

theorem log_ride_cost :
  ferris_wheel_cost + roller_coaster_cost + (additional_tickets_needed + initial_tickets - ferris_wheel_cost - roller_coaster_cost) = additional_tickets_needed + initial_tickets :=
by sorry

end NUMINAMATH_CALUDE_log_ride_cost_l5_553


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l5_523

theorem quadratic_is_square_of_binomial (a : ℝ) : 
  a = 4 → ∃ (r s : ℝ), a * x^2 + 16 * x + 16 = (r * x + s)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l5_523


namespace NUMINAMATH_CALUDE_range_of_a_l5_566

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| > 2 → |x| > a) ∧ 
  (∃ x : ℝ, |x| > a ∧ |x + 1| ≤ 2) → 
  a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l5_566


namespace NUMINAMATH_CALUDE_odd_function_constant_term_zero_l5_505

def f (a b c x : ℝ) : ℝ := a * x^3 - b * x + c

theorem odd_function_constant_term_zero (a b c : ℝ) :
  (∀ x, f a b c x = -f a b c (-x)) → c = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_constant_term_zero_l5_505


namespace NUMINAMATH_CALUDE_molecular_weight_CaCO3_is_100_l5_577

/-- The molecular weight of CaCO3 in grams per mole -/
def molecular_weight_CaCO3 : ℝ := 100

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 9

/-- The total molecular weight of the given number of moles in grams -/
def given_total_weight : ℝ := 900

/-- Theorem stating that the molecular weight of CaCO3 is 100 grams/mole -/
theorem molecular_weight_CaCO3_is_100 :
  molecular_weight_CaCO3 = given_total_weight / given_moles :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_CaCO3_is_100_l5_577


namespace NUMINAMATH_CALUDE_shadow_length_l5_547

/-- Given two similar right triangles, if one has height 2 and base 4,
    and the other has height 2.5, then the base of the second triangle is 5. -/
theorem shadow_length (h1 h2 b1 b2 : ℝ) : 
  h1 = 2 → h2 = 2.5 → b1 = 4 → h1 / b1 = h2 / b2 → b2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_shadow_length_l5_547


namespace NUMINAMATH_CALUDE_money_distribution_l5_579

/-- Given a distribution of money in the ratio 3 : 5 : 7 among three people,
    where the second person's share is 1500, 
    the difference between the first and third person's shares is 1200. -/
theorem money_distribution (total : ℕ) (f v r : ℕ) : 
  (f + v + r = total) →
  (3 * v = 5 * f) →
  (5 * r = 7 * v) →
  (v = 1500) →
  (r - f = 1200) :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l5_579


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l5_587

theorem arithmetic_series_sum : 
  ∀ (a₁ aₙ d : ℚ) (n : ℕ),
  a₁ = 16 → 
  aₙ = 32 → 
  d = 1/3 → 
  aₙ = a₁ + (n - 1) * d →
  (n * (a₁ + aₙ)) / 2 = 1176 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l5_587


namespace NUMINAMATH_CALUDE_quadratic_one_root_l5_599

theorem quadratic_one_root (a : ℝ) : 
  (∃! x : ℝ, a * x^2 - 2 * a * x - 1 = 0) → a = -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l5_599


namespace NUMINAMATH_CALUDE_total_is_260_l5_510

/-- Represents the ratio of money shared among four people -/
structure MoneyRatio :=
  (a b c d : ℕ)

/-- Calculates the total amount of money shared given a ratio and the first person's share -/
def totalShared (ratio : MoneyRatio) (firstShare : ℕ) : ℕ :=
  firstShare * (ratio.a + ratio.b + ratio.c + ratio.d)

/-- Theorem stating that for the given ratio and first share, the total is 260 -/
theorem total_is_260 (ratio : MoneyRatio) (h1 : ratio.a = 1) (h2 : ratio.b = 2) 
    (h3 : ratio.c = 7) (h4 : ratio.d = 3) (h5 : firstShare = 20) : 
    totalShared ratio firstShare = 260 := by
  sorry


end NUMINAMATH_CALUDE_total_is_260_l5_510


namespace NUMINAMATH_CALUDE_complex_number_equality_l5_589

theorem complex_number_equality : Complex.I - (1 / Complex.I) = 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l5_589


namespace NUMINAMATH_CALUDE_pennies_spent_l5_567

/-- Given that Sam initially had 98 pennies and now has 5 pennies left,
    prove that the number of pennies Sam spent is 93. -/
theorem pennies_spent (initial : Nat) (left : Nat) (spent : Nat)
    (h1 : initial = 98)
    (h2 : left = 5)
    (h3 : spent = initial - left) :
  spent = 93 := by
  sorry

end NUMINAMATH_CALUDE_pennies_spent_l5_567


namespace NUMINAMATH_CALUDE_a_6_value_l5_546

/-- An arithmetic sequence where a_2 and a_10 are roots of 2x^2 - x - 7 = 0 -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  2 * (a 2)^2 - (a 2) - 7 = 0 ∧
  2 * (a 10)^2 - (a 10) - 7 = 0

theorem a_6_value (a : ℕ → ℚ) (h : ArithmeticSequence a) : a 6 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_a_6_value_l5_546


namespace NUMINAMATH_CALUDE_dormitory_students_count_unique_solution_l5_562

/-- Represents the number of students in the dormitory -/
def n : ℕ := 6

/-- Represents the number of administrators -/
def m : ℕ := 3

/-- The total number of greeting cards used -/
def total_cards : ℕ := 51

/-- Theorem stating that the number of students in the dormitory is 6 -/
theorem dormitory_students_count :
  (n * (n - 1)) / 2 + n * m + m = total_cards :=
by sorry

/-- Theorem stating that n is the unique solution for the given conditions -/
theorem unique_solution (k : ℕ) :
  (k * (k - 1)) / 2 + k * m + m = total_cards → k = n :=
by sorry

end NUMINAMATH_CALUDE_dormitory_students_count_unique_solution_l5_562


namespace NUMINAMATH_CALUDE_absolute_value_equality_l5_519

theorem absolute_value_equality (x : ℝ) : |x - 3| = |x + 2| → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l5_519


namespace NUMINAMATH_CALUDE_rectangle_area_l5_512

/-- The area of a rectangle with length 20 cm and width 25 cm is 500 cm² -/
theorem rectangle_area : 
  ∀ (rectangle : Set ℝ) (length width area : ℝ),
  length = 20 →
  width = 25 →
  area = length * width →
  area = 500 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l5_512


namespace NUMINAMATH_CALUDE_second_number_calculation_l5_581

theorem second_number_calculation (A B : ℝ) : 
  A = 3200 → 
  0.1 * A = 0.2 * B + 190 → 
  B = 650 := by
sorry

end NUMINAMATH_CALUDE_second_number_calculation_l5_581


namespace NUMINAMATH_CALUDE_speed_conversion_equivalence_l5_573

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- The given speed in meters per second -/
def given_speed_mps : ℝ := 35.0028

/-- The calculated speed in kilometers per hour -/
def calculated_speed_kmph : ℝ := 126.01008

theorem speed_conversion_equivalence : 
  given_speed_mps * mps_to_kmph = calculated_speed_kmph := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_equivalence_l5_573


namespace NUMINAMATH_CALUDE_first_day_rain_l5_586

/-- The amount of rain Greg experienced while camping, given the known conditions -/
def camping_rain (first_day : ℝ) : ℝ := first_day + 6 + 5

/-- The amount of rain at Greg's house during the same week -/
def house_rain : ℝ := 26

/-- The difference in rain between Greg's house and his camping experience -/
def rain_difference : ℝ := 12

theorem first_day_rain : 
  ∃ (x : ℝ), camping_rain x = house_rain - rain_difference ∧ x = 3 :=
sorry

end NUMINAMATH_CALUDE_first_day_rain_l5_586


namespace NUMINAMATH_CALUDE_dot_product_of_specific_vectors_l5_585

theorem dot_product_of_specific_vectors :
  let a : ℝ × ℝ := (-2, 4)
  let b : ℝ × ℝ := (1, 2)
  (a.1 * b.1 + a.2 * b.2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_of_specific_vectors_l5_585


namespace NUMINAMATH_CALUDE_trapezoid_x_squared_l5_580

/-- A trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  x : ℝ
  shorter_base_length : shorter_base = 50
  longer_base_length : longer_base = shorter_base + 50
  midpoint_ratio : (shorter_base + (shorter_base + longer_base) / 2) / ((shorter_base + longer_base) / 2 + longer_base) = 1 / 2
  equal_area : x > shorter_base ∧ x < longer_base ∧ 
    (x - shorter_base) / (longer_base - shorter_base) = 
    (x - shorter_base) * (x + shorter_base) / ((longer_base - shorter_base) * (longer_base + shorter_base))

theorem trapezoid_x_squared (t : Trapezoid) : t.x^2 = 6875 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_x_squared_l5_580


namespace NUMINAMATH_CALUDE_equation_system_solution_l5_578

theorem equation_system_solution : ∃ (x y z : ℝ),
  (2 * x - 3 * y - z = 0) ∧
  (x + 3 * y - 14 * z = 0) ∧
  (z = 2) ∧
  ((x^2 + 3*x*y) / (y^2 + z^2) = 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l5_578


namespace NUMINAMATH_CALUDE_quadratic_radical_range_l5_502

theorem quadratic_radical_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 3 - x) ↔ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_range_l5_502


namespace NUMINAMATH_CALUDE_vampire_survival_l5_544

/-- The number of people a vampire needs to suck blood from each day to survive -/
def vampire_daily_victims : ℕ :=
  let gallons_per_week : ℕ := 7
  let pints_per_gallon : ℕ := 8
  let pints_per_person : ℕ := 2
  let days_per_week : ℕ := 7
  (gallons_per_week * pints_per_gallon) / (pints_per_person * days_per_week)

theorem vampire_survival : vampire_daily_victims = 4 := by
  sorry

end NUMINAMATH_CALUDE_vampire_survival_l5_544


namespace NUMINAMATH_CALUDE_ice_cream_bill_calculation_l5_518

/-- The final bill for four ice cream sundaes with a 20% tip -/
def final_bill (sundae1 sundae2 sundae3 sundae4 : ℝ) (tip_percentage : ℝ) : ℝ :=
  let total_cost := sundae1 + sundae2 + sundae3 + sundae4
  let tip := tip_percentage * total_cost
  total_cost + tip

/-- Theorem stating that the final bill for the given sundae prices and tip percentage is $42.00 -/
theorem ice_cream_bill_calculation :
  final_bill 7.50 10.00 8.50 9.00 0.20 = 42.00 := by
  sorry


end NUMINAMATH_CALUDE_ice_cream_bill_calculation_l5_518


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l5_527

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_through_point :
  let l1 : Line := { a := 2, b := -3, c := 9 }
  let l2 : Line := { a := 3, b := 2, c := -1 }
  let p : Point := { x := -1, y := 2 }
  perpendicular l1 l2 ∧ pointOnLine p l2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l5_527


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l5_538

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  GeometricSequence a →
  a 4 - a 2 = 6 →
  a 5 - a 1 = 15 →
  a 3 = 4 ∨ a 3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l5_538


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l5_540

theorem geometric_sequence_fourth_term 
  (a : ℝ) -- first term
  (h1 : a ≠ 0) -- ensure first term is non-zero for division
  (h2 : (3*a + 3) / a = (6*a + 6) / (3*a + 3)) -- condition for geometric sequence
  (h3 : 3*a + 3 = a * ((3*a + 3) / a)) -- second term definition
  (h4 : 6*a + 6 = (3*a + 3) * ((3*a + 3) / a)) -- third term definition
  : a * ((3*a + 3) / a)^3 = -24 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l5_540


namespace NUMINAMATH_CALUDE_no_real_roots_l5_545

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the quadratic equation
def quadratic_equation (t : Triangle) (x : ℝ) : Prop :=
  t.a^2 * x^2 + (t.b^2 - t.a^2 - t.c^2) * x + t.c^2 = 0

-- Theorem statement
theorem no_real_roots (t : Triangle) : ¬∃ x : ℝ, quadratic_equation t x := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l5_545


namespace NUMINAMATH_CALUDE_largest_common_term_correct_l5_571

/-- First arithmetic progression -/
def seq1 (n : ℕ) : ℕ := 7 * (n + 1)

/-- Second arithmetic progression -/
def seq2 (m : ℕ) : ℕ := 8 + 12 * m

/-- Predicate for common terms -/
def isCommonTerm (a : ℕ) : Prop :=
  ∃ n m : ℕ, seq1 n = a ∧ seq2 m = a

/-- The largest common term less than 500 -/
def largestCommonTerm : ℕ := 476

theorem largest_common_term_correct :
  isCommonTerm largestCommonTerm ∧
  largestCommonTerm < 500 ∧
  ∀ x : ℕ, isCommonTerm x → x < 500 → x ≤ largestCommonTerm :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_correct_l5_571


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l5_565

theorem rectangular_to_polar_conversion :
  let x : ℝ := Real.sqrt 3
  let y : ℝ := -1
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 2 * Real.pi - Real.arctan (1 / Real.sqrt 3)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ r = 2 ∧ θ = 11 * Real.pi / 6 :=
by
  sorry

#check rectangular_to_polar_conversion

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l5_565


namespace NUMINAMATH_CALUDE_building_height_l5_521

/-- Given a flagpole and a building casting shadows under similar conditions,
    calculate the height of the building. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (flagpole_height_pos : 0 < flagpole_height)
  (flagpole_shadow_pos : 0 < flagpole_shadow)
  (building_shadow_pos : 0 < building_shadow)
  (h : flagpole_height = 18)
  (s1 : flagpole_shadow = 45)
  (s2 : building_shadow = 60) :
  flagpole_height / flagpole_shadow * building_shadow = 24 := by
sorry


end NUMINAMATH_CALUDE_building_height_l5_521


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_33_l5_525

theorem modular_inverse_of_5_mod_33 :
  ∃ x : ℕ, x ≥ 0 ∧ x ≤ 32 ∧ (5 * x) % 33 = 1 ∧ x = 20 := by sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_33_l5_525


namespace NUMINAMATH_CALUDE_delta_max_success_ratio_l5_530

/-- Represents a participant's score for a single day -/
structure DayScore where
  scored : ℕ
  attempted : ℕ

/-- Represents a participant's scores for two days -/
structure TwoDayScore where
  day1 : DayScore
  day2 : DayScore

def success_ratio (score : DayScore) : ℚ :=
  score.scored / score.attempted

def overall_success_ratio (score : TwoDayScore) : ℚ :=
  (score.day1.scored + score.day2.scored) / (score.day1.attempted + score.day2.attempted)

theorem delta_max_success_ratio 
  (gamma : TwoDayScore)
  (delta : TwoDayScore)
  (h1 : gamma.day1 = ⟨210, 350⟩)
  (h2 : gamma.day2 = ⟨150, 250⟩)
  (h3 : delta.day1.attempted + delta.day2.attempted = 600)
  (h4 : delta.day1.attempted ≠ 350)
  (h5 : delta.day1.scored > 0 ∧ delta.day2.scored > 0)
  (h6 : success_ratio delta.day1 < success_ratio gamma.day1)
  (h7 : success_ratio delta.day2 < success_ratio gamma.day2)
  (h8 : overall_success_ratio gamma = 3/5) :
  overall_success_ratio delta ≤ 359/600 := by
sorry

end NUMINAMATH_CALUDE_delta_max_success_ratio_l5_530


namespace NUMINAMATH_CALUDE_min_value_of_f_range_of_x_when_f_leq_5_l5_548

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 4| + |x - 1|

-- Theorem for the minimum value of f(x)
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 3 :=
sorry

-- Theorem for the range of x when f(x) ≤ 5
theorem range_of_x_when_f_leq_5 :
  ∀ x, f x ≤ 5 ↔ 0 ≤ x ∧ x ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_range_of_x_when_f_leq_5_l5_548


namespace NUMINAMATH_CALUDE_distance_to_focus_l5_595

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def P (y : ℝ) : ℝ × ℝ := (4, y)

-- Theorem statement
theorem distance_to_focus (y : ℝ) (h : parabola 4 y) : 
  Real.sqrt ((P y).1 - focus.1)^2 + ((P y).2 - focus.2)^2 = 5 := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l5_595


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l5_537

theorem sufficient_condition_for_inequality (x : ℝ) : 
  1 < x ∧ x < 2 → (x + 1) / (x - 1) > 2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l5_537


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l5_506

theorem arithmetic_evaluation : 4 * (9 - 6) - 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l5_506


namespace NUMINAMATH_CALUDE_multiply_decimals_l5_542

theorem multiply_decimals : 3.6 * 0.3 = 1.08 := by
  sorry

end NUMINAMATH_CALUDE_multiply_decimals_l5_542


namespace NUMINAMATH_CALUDE_completing_square_transformation_l5_550

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 8*x - 1 = 0) ↔ ((x - 4)^2 = 17) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l5_550


namespace NUMINAMATH_CALUDE_intersection_M_N_l5_528

def M : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | 2 * x - 1 > 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l5_528


namespace NUMINAMATH_CALUDE_intersection_range_l5_507

/-- The line equation y = a(x + 2) -/
def line (a x : ℝ) : ℝ := a * (x + 2)

/-- The curve equation x^2 - y|y| = 1 -/
def curve (x y : ℝ) : Prop := x^2 - y * abs y = 1

/-- The number of intersection points between the line and the curve -/
def intersection_count (a : ℝ) : ℕ := sorry

/-- The theorem stating the range of a for exactly 2 intersection points -/
theorem intersection_range :
  ∀ a : ℝ, intersection_count a = 2 ↔ a ∈ Set.Ioo (-Real.sqrt 3 / 3) 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l5_507


namespace NUMINAMATH_CALUDE_line_segment_no_intersection_l5_514

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line segment between two points -/
structure LineSegment where
  p1 : Point
  p2 : Point

/-- Checks if a line segment intersects both x and y axes -/
def intersectsBothAxes (l : LineSegment) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    ((l.p1.x + t * (l.p2.x - l.p1.x) = 0 ∧ l.p1.y + t * (l.p2.y - l.p1.y) ≠ 0) ∨
     (l.p1.x + t * (l.p2.x - l.p1.x) ≠ 0 ∧ l.p1.y + t * (l.p2.y - l.p1.y) = 0))

theorem line_segment_no_intersection :
  let p1 : Point := ⟨-3, 4⟩
  let p2 : Point := ⟨-5, 1⟩
  let segment : LineSegment := ⟨p1, p2⟩
  ¬(intersectsBothAxes segment) :=
by
  sorry

end NUMINAMATH_CALUDE_line_segment_no_intersection_l5_514


namespace NUMINAMATH_CALUDE_sum_of_number_and_reverse_is_99_l5_563

/-- Definition of a two-digit number -/
def TwoDigitNumber (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

/-- The property that the difference between the number and its reverse
    is 7 times the sum of its digits -/
def SatisfiesEquation (a b : ℕ) : Prop :=
  (10 * a + b) - (10 * b + a) = 7 * (a + b)

/-- Theorem stating that for a two-digit number satisfying the given equation,
    the sum of the number and its reverse is 99 -/
theorem sum_of_number_and_reverse_is_99 (a b : ℕ) 
  (h1 : TwoDigitNumber a b) (h2 : SatisfiesEquation a b) : 
  (10 * a + b) + (10 * b + a) = 99 := by
  sorry

#check sum_of_number_and_reverse_is_99

end NUMINAMATH_CALUDE_sum_of_number_and_reverse_is_99_l5_563


namespace NUMINAMATH_CALUDE_johns_memory_card_cost_l5_598

/-- Calculates the total cost of memory cards for John's photography habit -/
theorem johns_memory_card_cost :
  let pictures_per_day : ℕ := 25
  let years : ℕ := 6
  let days_per_year : ℕ := 365
  let images_per_card : ℕ := 40
  let cost_per_card : ℕ := 75
  let total_pictures : ℕ := pictures_per_day * years * days_per_year
  let cards_needed : ℕ := (total_pictures + images_per_card - 1) / images_per_card
  cards_needed * cost_per_card = 102675 :=
by
  sorry


end NUMINAMATH_CALUDE_johns_memory_card_cost_l5_598


namespace NUMINAMATH_CALUDE_triangle_base_length_l5_570

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 9 →
  height = 6 →
  area = (base * height) / 2 →
  base = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_base_length_l5_570


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l5_500

theorem power_fraction_simplification : (2^2020 + 2^2018) / (2^2020 - 2^2018) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l5_500


namespace NUMINAMATH_CALUDE_payment_difference_l5_515

/-- The original price of the dish -/
def original_price : Float := 24.00000000000002

/-- The discount percentage -/
def discount_percent : Float := 0.10

/-- The tip percentage -/
def tip_percent : Float := 0.15

/-- The discounted price of the dish -/
def discounted_price : Float := original_price * (1 - discount_percent)

/-- John's tip amount -/
def john_tip : Float := original_price * tip_percent

/-- Jane's tip amount -/
def jane_tip : Float := discounted_price * tip_percent

/-- John's total payment -/
def john_total : Float := discounted_price + john_tip

/-- Jane's total payment -/
def jane_total : Float := discounted_price + jane_tip

/-- Theorem stating the difference between John's and Jane's payments -/
theorem payment_difference : john_total - jane_total = 0.3600000000000003 := by
  sorry

end NUMINAMATH_CALUDE_payment_difference_l5_515


namespace NUMINAMATH_CALUDE_initial_volume_proof_l5_501

/-- Given a solution with initial volume V and 5% alcohol concentration,
    adding 2.5 liters of alcohol and 7.5 liters of water results in a
    9% alcohol concentration. Prove that V must be 40 liters. -/
theorem initial_volume_proof (V : ℝ) : 
  (0.05 * V + 2.5) / (V + 10) = 0.09 → V = 40 := by sorry

end NUMINAMATH_CALUDE_initial_volume_proof_l5_501


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l5_561

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- State the theorem
theorem f_even_and_increasing :
  (∀ x, f x = f (-x)) ∧  -- f is an even function
  (∀ x y, 0 < x → x < y → f x < f y) -- f is monotonically increasing on (0,+∞)
  := by sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l5_561


namespace NUMINAMATH_CALUDE_no_right_triangle_perimeter_twice_hypotenuse_l5_532

theorem no_right_triangle_perimeter_twice_hypotenuse :
  ¬∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b + Real.sqrt (a^2 + b^2) = 2 * Real.sqrt (a^2 + b^2) :=
by sorry

end NUMINAMATH_CALUDE_no_right_triangle_perimeter_twice_hypotenuse_l5_532


namespace NUMINAMATH_CALUDE_complex_multiplication_l5_588

theorem complex_multiplication (z : ℂ) : z = 2 - I → I^3 * z = -1 - 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l5_588


namespace NUMINAMATH_CALUDE_sum_of_pairwise_quotients_geq_three_halves_l5_508

theorem sum_of_pairwise_quotients_geq_three_halves 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_pairwise_quotients_geq_three_halves_l5_508


namespace NUMINAMATH_CALUDE_largest_root_is_three_l5_582

-- Define the cubic polynomial
def cubic (x : ℝ) : ℝ := x^3 - 3*x^2 - 8*x + 15

-- Define the conditions for p, q, and r
def root_conditions (p q r : ℝ) : Prop :=
  p + q + r = 3 ∧ p*q + p*r + q*r = -8 ∧ p*q*r = -15

-- Theorem statement
theorem largest_root_is_three :
  ∃ (p q r : ℝ), root_conditions p q r ∧
  (cubic p = 0 ∧ cubic q = 0 ∧ cubic r = 0) ∧
  (∀ x : ℝ, cubic x = 0 → x ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_largest_root_is_three_l5_582


namespace NUMINAMATH_CALUDE_min_cards_for_even_product_l5_556

def is_even (n : Nat) : Bool := n % 2 = 0

theorem min_cards_for_even_product :
  ∀ (S : Finset Nat),
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 16) →
  (Finset.card S = 16) →
  (∃ (T : Finset Nat), T ⊆ S ∧ Finset.card T = 9 ∧ ∃ n ∈ T, is_even n) ∧
  (∀ (U : Finset Nat), U ⊆ S → Finset.card U < 9 → ∀ n ∈ U, ¬is_even n) :=
by sorry

end NUMINAMATH_CALUDE_min_cards_for_even_product_l5_556


namespace NUMINAMATH_CALUDE_baker_cake_difference_l5_554

/-- Given the initial number of cakes, the number of cakes sold, and the number of cakes bought,
    prove that the difference between cakes sold and cakes bought is 47. -/
theorem baker_cake_difference (initial : ℕ) (sold : ℕ) (bought : ℕ) 
    (h1 : initial = 170) (h2 : sold = 78) (h3 : bought = 31) : 
    sold - bought = 47 := by
  sorry

end NUMINAMATH_CALUDE_baker_cake_difference_l5_554


namespace NUMINAMATH_CALUDE_sum_of_products_l5_575

theorem sum_of_products : 1234 * 2 + 2341 * 2 + 3412 * 2 + 4123 * 2 = 22220 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_l5_575


namespace NUMINAMATH_CALUDE_roots_sum_square_l5_531

/-- Given that α and β are the two roots of the equation x^2 - 7x + 3 = 0 and α > β,
    prove that α^2 + 7β = 46 -/
theorem roots_sum_square (α β : ℝ) : 
  α^2 - 7*α + 3 = 0 → 
  β^2 - 7*β + 3 = 0 → 
  α > β →
  α^2 + 7*β = 46 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_square_l5_531


namespace NUMINAMATH_CALUDE_slope_range_for_intersection_l5_534

/-- The set of possible slopes for a line with y-intercept (0,3) that intersects the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m ≤ -Real.sqrt (1/20) ∨ m ≥ Real.sqrt (1/20)}

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  4 * x^2 + 25 * y^2 = 100

/-- The line equation with slope m and y-intercept 3 -/
def line_equation (m x : ℝ) : ℝ :=
  m * x + 3

theorem slope_range_for_intersection :
  ∀ m : ℝ, (∃ x : ℝ, ellipse_equation x (line_equation m x)) ↔ m ∈ possible_slopes :=
by sorry

end NUMINAMATH_CALUDE_slope_range_for_intersection_l5_534


namespace NUMINAMATH_CALUDE_pencils_given_to_dorothy_l5_526

/-- Given that Josh had a certain number of pencils initially and was left with
    a smaller number after giving some to Dorothy, prove that the number of
    pencils he gave to Dorothy is the difference between the initial and final amounts. -/
theorem pencils_given_to_dorothy
  (initial_pencils : ℕ)
  (remaining_pencils : ℕ)
  (h1 : initial_pencils = 142)
  (h2 : remaining_pencils = 111)
  (h3 : remaining_pencils < initial_pencils) :
  initial_pencils - remaining_pencils = 31 :=
by sorry

end NUMINAMATH_CALUDE_pencils_given_to_dorothy_l5_526


namespace NUMINAMATH_CALUDE_min_value_F_l5_574

/-- The function F(x, y) -/
def F (x y : ℝ) : ℝ := x^2 + 8*y + y^2 + 14*x - 6

/-- The constraint equation -/
def constraint (x y : ℝ) : Prop := x^2 + y^2 + 25 = 10*(x + y)

/-- Theorem stating that the minimum value of F(x, y) is 29 under the given constraint -/
theorem min_value_F :
  ∃ (m : ℝ), m = 29 ∧
  (∀ x y : ℝ, constraint x y → F x y ≥ m) ∧
  (∃ x y : ℝ, constraint x y ∧ F x y = m) :=
sorry

end NUMINAMATH_CALUDE_min_value_F_l5_574


namespace NUMINAMATH_CALUDE_convention_handshakes_count_l5_558

/-- The number of handshakes at the Interregional Mischief Convention --/
def convention_handshakes (n_gremlins n_imps n_disagreeing_imps n_affected_gremlins : ℕ) : ℕ :=
  let gremlin_handshakes := n_gremlins * (n_gremlins - 1) / 2
  let normal_imp_gremlin_handshakes := (n_imps - n_disagreeing_imps) * n_gremlins
  let affected_imp_gremlin_handshakes := n_disagreeing_imps * (n_gremlins - n_affected_gremlins)
  gremlin_handshakes + normal_imp_gremlin_handshakes + affected_imp_gremlin_handshakes

/-- Theorem stating the number of handshakes at the convention --/
theorem convention_handshakes_count : convention_handshakes 30 20 5 10 = 985 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_count_l5_558


namespace NUMINAMATH_CALUDE_sin_cos_sum_47_43_l5_517

theorem sin_cos_sum_47_43 : Real.sin (47 * π / 180) * Real.cos (43 * π / 180) + Real.cos (47 * π / 180) * Real.sin (43 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_47_43_l5_517


namespace NUMINAMATH_CALUDE_average_of_x_and_y_l5_539

theorem average_of_x_and_y (x y : ℝ) : 
  (2 + 6 + 10 + x + y) / 5 = 18 → (x + y) / 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_average_of_x_and_y_l5_539


namespace NUMINAMATH_CALUDE_fixed_fee_calculation_l5_511

theorem fixed_fee_calculation (feb_bill march_bill : ℝ) 
  (h : feb_bill = 18.72 ∧ march_bill = 33.78) :
  ∃ (fixed_fee hourly_rate : ℝ),
    fixed_fee + hourly_rate = feb_bill ∧
    fixed_fee + 3 * hourly_rate = march_bill ∧
    fixed_fee = 11.19 := by
sorry

end NUMINAMATH_CALUDE_fixed_fee_calculation_l5_511


namespace NUMINAMATH_CALUDE_digit_47_is_6_l5_513

/-- The decimal representation of 1/17 as a list of digits -/
def decimal_rep_1_17 : List Nat := [0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7]

/-- The length of the repeating cycle in the decimal representation of 1/17 -/
def cycle_length : Nat := 16

/-- The 47th digit after the decimal point in the decimal representation of 1/17 -/
def digit_47 : Nat := decimal_rep_1_17[(47 - 1) % cycle_length]

theorem digit_47_is_6 : digit_47 = 6 := by sorry

end NUMINAMATH_CALUDE_digit_47_is_6_l5_513


namespace NUMINAMATH_CALUDE_four_sacks_filled_l5_557

/-- Calculates the number of sacks filled given the total pieces of wood and capacity per sack -/
def sacks_filled (total_wood : ℕ) (wood_per_sack : ℕ) : ℕ :=
  total_wood / wood_per_sack

/-- Theorem: Given 80 pieces of wood and sacks that can hold 20 pieces each, 4 sacks will be filled -/
theorem four_sacks_filled : sacks_filled 80 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_sacks_filled_l5_557


namespace NUMINAMATH_CALUDE_coffee_mixture_proof_l5_593

/-- The cost of Colombian coffee beans per pound -/
def colombian_cost : ℝ := 5.50

/-- The cost of Peruvian coffee beans per pound -/
def peruvian_cost : ℝ := 4.25

/-- The total weight of the mixture in pounds -/
def total_weight : ℝ := 40

/-- The desired cost per pound of the mixture -/
def mixture_cost : ℝ := 4.60

/-- The amount of Colombian coffee beans in the mixture -/
def colombian_amount : ℝ := 11.2

theorem coffee_mixture_proof :
  colombian_amount * colombian_cost + (total_weight - colombian_amount) * peruvian_cost = 
  mixture_cost * total_weight :=
by sorry

end NUMINAMATH_CALUDE_coffee_mixture_proof_l5_593


namespace NUMINAMATH_CALUDE_semicircle_area_theorem_l5_596

noncomputable def semicircle_area (P Q R S T U : Point) : ℝ :=
  let PQ_radius := 2
  let PS_length := Real.sqrt 2
  let QS_length := Real.sqrt 2
  let PT_radius := PQ_radius / 2
  let QU_radius := PQ_radius / 2
  let TU_radius := PS_length
  let triangle_PQS_area := PQ_radius * (Real.sqrt 2) / 2
  (PT_radius^2 * Real.pi / 2) + (QU_radius^2 * Real.pi / 2) + (TU_radius^2 * Real.pi / 2) - triangle_PQS_area

theorem semicircle_area_theorem (P Q R S T U : Point) :
  semicircle_area P Q R S T U = 9 * Real.pi - 2 :=
sorry

end NUMINAMATH_CALUDE_semicircle_area_theorem_l5_596


namespace NUMINAMATH_CALUDE_total_travel_ways_problem_solution_l5_503

/-- Represents the number of transportation options between two cities -/
structure TransportOptions where
  buses : Nat
  trains : Nat
  ferries : Nat

/-- Calculates the total number of ways to travel between two cities -/
def totalWays (options : TransportOptions) : Nat :=
  options.buses + options.trains + options.ferries

/-- Theorem: The total number of ways to travel from A to C via B is the product
    of the number of ways to travel from A to B and from B to C -/
theorem total_travel_ways
  (optionsAB : TransportOptions)
  (optionsBC : TransportOptions) :
  totalWays optionsAB * totalWays optionsBC =
  (optionsAB.buses + optionsAB.trains) * (optionsBC.buses + optionsBC.ferries) :=
by sorry

/-- Given the specific transportation options in the problem -/
def morningOptions : TransportOptions :=
  { buses := 5, trains := 2, ferries := 0 }

def afternoonOptions : TransportOptions :=
  { buses := 3, trains := 0, ferries := 2 }

/-- The main theorem that proves the total number of ways for the specific problem -/
theorem problem_solution :
  totalWays morningOptions * totalWays afternoonOptions = 35 :=
by sorry

end NUMINAMATH_CALUDE_total_travel_ways_problem_solution_l5_503


namespace NUMINAMATH_CALUDE_sarah_apple_ratio_l5_564

theorem sarah_apple_ratio : 
  let sarah_apples : ℕ := 45
  let brother_apples : ℕ := 9
  (sarah_apples : ℚ) / brother_apples = 5 := by sorry

end NUMINAMATH_CALUDE_sarah_apple_ratio_l5_564


namespace NUMINAMATH_CALUDE_debt_settlement_possible_l5_576

theorem debt_settlement_possible (vasya_coin_value : ℕ) (petya_coin_value : ℕ) 
  (debt : ℕ) (h1 : vasya_coin_value = 49) (h2 : petya_coin_value = 99) (h3 : debt = 1) :
  ∃ (n m : ℕ), vasya_coin_value * n - petya_coin_value * m = debt :=
by sorry

end NUMINAMATH_CALUDE_debt_settlement_possible_l5_576


namespace NUMINAMATH_CALUDE_line_segment_intersection_k_range_l5_597

/-- Given points A and B and a line y = kx + 1 that intersects line segment AB, 
    the range of k is [1/2, 1] -/
theorem line_segment_intersection_k_range 
  (A B : ℝ × ℝ) 
  (hA : A = (1, 2)) 
  (hB : B = (2, 1)) 
  (k : ℝ) 
  (h_intersect : ∃ (x y : ℝ), 
    y = k * x + 1 ∧ 
    (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
      x = A.1 + t * (B.1 - A.1) ∧ 
      y = A.2 + t * (B.2 - A.2))) :
  1/2 ≤ k ∧ k ≤ 1 := by sorry

end NUMINAMATH_CALUDE_line_segment_intersection_k_range_l5_597


namespace NUMINAMATH_CALUDE_no_linear_factor_with_integer_coefficients_l5_524

theorem no_linear_factor_with_integer_coefficients :
  ∀ (a b c d : ℤ), (∀ (x y z : ℝ), 
    a*x + b*y + c*z + d ≠ 0 ∨ 
    x^2 - y^2 - z^2 + 3*y*z + x + 2*y - z ≠ (a*x + b*y + c*z + d) * 
      ((x^2 - y^2 - z^2 + 3*y*z + x + 2*y - z) / (a*x + b*y + c*z + d))) :=
by sorry

end NUMINAMATH_CALUDE_no_linear_factor_with_integer_coefficients_l5_524


namespace NUMINAMATH_CALUDE_pencil_pen_combinations_l5_533

theorem pencil_pen_combinations (pencil_types : Nat) (pen_types : Nat) :
  pencil_types = 4 → pen_types = 3 → pencil_types * pen_types = 12 := by
  sorry

end NUMINAMATH_CALUDE_pencil_pen_combinations_l5_533


namespace NUMINAMATH_CALUDE_factory_production_constraints_l5_591

/-- Given a factory producing two products A and B, this theorem states the constraint
conditions for maximizing the total monthly profit. -/
theorem factory_production_constraints
  (a₁ a₂ b₁ b₂ c₁ c₂ d₁ d₂ : ℝ)
  (x y : ℝ) -- Monthly production of products A and B in kg
  (h_pos_a₁ : a₁ > 0) (h_pos_a₂ : a₂ > 0)
  (h_pos_b₁ : b₁ > 0) (h_pos_b₂ : b₂ > 0)
  (h_pos_c₁ : c₁ > 0) (h_pos_c₂ : c₂ > 0)
  (h_pos_d₁ : d₁ > 0) (h_pos_d₂ : d₂ > 0) :
  (∃ z : ℝ, z = d₁ * x + d₂ * y ∧ -- Total monthly profit
    a₁ * x + a₂ * y ≤ c₁ ∧       -- Constraint on raw material A
    b₁ * x + b₂ * y ≤ c₂ ∧       -- Constraint on raw material B
    x ≥ 0 ∧ y ≥ 0) →             -- Non-negative production constraints
  (a₁ * x + a₂ * y ≤ c₁ ∧
   b₁ * x + b₂ * y ≤ c₂ ∧
   x ≥ 0 ∧ y ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_factory_production_constraints_l5_591


namespace NUMINAMATH_CALUDE_tiles_for_18_24_room_l5_583

/-- Calculates the number of tiles needed for a rectangular room with a double border --/
def tilesNeeded (length width : ℕ) : ℕ :=
  let borderTiles := 2 * (length - 2) + 2 * (length - 4) + 2 * (width - 2) + 2 * (width - 4) + 8
  let innerLength := length - 4
  let innerWidth := width - 4
  let innerArea := innerLength * innerWidth
  let innerTiles := (innerArea + 8) / 9  -- Ceiling division
  borderTiles + innerTiles

/-- The theorem states that for an 18 by 24 foot room, 183 tiles are needed --/
theorem tiles_for_18_24_room : tilesNeeded 24 18 = 183 := by
  sorry

end NUMINAMATH_CALUDE_tiles_for_18_24_room_l5_583


namespace NUMINAMATH_CALUDE_cos_225_degrees_l5_516

theorem cos_225_degrees : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l5_516


namespace NUMINAMATH_CALUDE_expansion_properties_l5_568

def binomial_sum (n : ℕ) : ℕ := 2^n

theorem expansion_properties (x : ℝ) :
  let n : ℕ := 8
  let binomial_sum_diff : ℕ := 128
  let largest_coeff_term : ℝ := 70 * x^4
  let x_power_7_term : ℝ := -56 * x^7
  (binomial_sum n - binomial_sum 7 = binomial_sum_diff) ∧
  (∀ k, 0 ≤ k ∧ k ≤ n → |(-1)^k * (n.choose k) * x^(2*n - 3*k)| ≤ |largest_coeff_term|) ∧
  ((-1)^3 * (n.choose 3) * x^(2*n - 3*3) = x_power_7_term) :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l5_568


namespace NUMINAMATH_CALUDE_divisibility_by_six_l5_592

theorem divisibility_by_six (n : ℕ) 
  (div_by_two : ∃ k : ℕ, n = 2 * k) 
  (div_by_three : ∃ m : ℕ, n = 3 * m) : 
  ∃ p : ℕ, n = 6 * p := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_six_l5_592
