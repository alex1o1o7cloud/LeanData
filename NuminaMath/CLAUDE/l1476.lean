import Mathlib

namespace NUMINAMATH_CALUDE_debelyn_gave_two_dolls_l1476_147619

/-- Represents the number of dolls each person has --/
structure DollCount where
  debelyn_initial : ℕ
  christel_initial : ℕ
  christel_to_andrena : ℕ
  debelyn_to_andrena : ℕ

/-- The conditions of the problem --/
def problem_conditions (d : DollCount) : Prop :=
  d.debelyn_initial = 20 ∧
  d.christel_initial = 24 ∧
  d.christel_to_andrena = 5 ∧
  d.debelyn_initial - d.debelyn_to_andrena + 3 = d.christel_initial - d.christel_to_andrena + 2

theorem debelyn_gave_two_dolls (d : DollCount) 
  (h : problem_conditions d) : d.debelyn_to_andrena = 2 := by
  sorry

#check debelyn_gave_two_dolls

end NUMINAMATH_CALUDE_debelyn_gave_two_dolls_l1476_147619


namespace NUMINAMATH_CALUDE_problem_statement_l1476_147648

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_property (f : ℝ → ℝ) : Prop := ∀ x, f (2 + x) = f (-x)

theorem problem_statement (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_prop : has_property f)
  (h_f1 : f 1 = 2) : 
  f 2018 + f 2019 = -2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1476_147648


namespace NUMINAMATH_CALUDE_y_coordinate_of_P_l1476_147645

/-- The y-coordinate of a point P satisfying certain conditions -/
theorem y_coordinate_of_P (A B C D P : ℝ × ℝ) : 
  A = (-4, 0) →
  B = (-1, 2) →
  C = (1, 2) →
  D = (4, 0) →
  dist P A + dist P D = 10 →
  dist P B + dist P C = 10 →
  P.2 = (-12 + 16 * Real.sqrt 16.5) / 5 :=
by sorry

end NUMINAMATH_CALUDE_y_coordinate_of_P_l1476_147645


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1476_147664

-- Define the universe set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 2, 3}

-- Define set B
def B : Set ℕ := {1, 4}

-- Theorem statement
theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {4} :=
sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1476_147664


namespace NUMINAMATH_CALUDE_unique_solution_l1476_147688

def system_equations (n : ℕ) (x : ℕ → ℝ) : Prop :=
  n ≥ 2 ∧
  (∀ i : ℕ, i ∈ Finset.range n → 
    max (i + 1 : ℝ) (x i) = if i + 1 = n then n * x 0 else x (i + 1))

theorem unique_solution (n : ℕ) (x : ℕ → ℝ) :
  system_equations n x → (∀ i : ℕ, i ∈ Finset.range n → x i = 1) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1476_147688


namespace NUMINAMATH_CALUDE_chocolate_cost_proof_l1476_147694

/-- The cost of the chocolate -/
def chocolate_cost : ℝ := 3

/-- The cost of the candy bar -/
def candy_bar_cost : ℝ := 6

theorem chocolate_cost_proof :
  chocolate_cost = 3 ∧
  candy_bar_cost = 6 ∧
  candy_bar_cost = chocolate_cost + 3 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_cost_proof_l1476_147694


namespace NUMINAMATH_CALUDE_x_power_27_minus_reciprocal_l1476_147609

theorem x_power_27_minus_reciprocal (x : ℂ) (h : x - 1/x = Complex.I * Real.sqrt 3) :
  x^27 - 1/(x^27) = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_power_27_minus_reciprocal_l1476_147609


namespace NUMINAMATH_CALUDE_shaded_triangle_probability_l1476_147611

/-- Given a set of triangles with equal selection probability, 
    this function calculates the probability of selecting a shaded triangle -/
def probability_shaded_triangle (total_triangles : ℕ) (shaded_triangles : ℕ) : ℚ :=
  shaded_triangles / total_triangles

/-- Theorem: The probability of selecting a shaded triangle 
    given 6 total triangles and 2 shaded triangles is 1/3 -/
theorem shaded_triangle_probability :
  probability_shaded_triangle 6 2 = 1/3 := by
  sorry

#eval probability_shaded_triangle 6 2

end NUMINAMATH_CALUDE_shaded_triangle_probability_l1476_147611


namespace NUMINAMATH_CALUDE_rational_function_property_l1476_147613

theorem rational_function_property (f : ℚ → ℝ) 
  (h : ∀ r s : ℚ, ∃ n : ℤ, f (r + s) - f r - f s = n) :
  ∃ (q : ℕ+) (p : ℤ), |f (1 / q) - p| ≤ 1 / 2012 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_property_l1476_147613


namespace NUMINAMATH_CALUDE_factorization_equality_l1476_147653

theorem factorization_equality (a b : ℝ) : 2 * a^2 - 4 * a * b + 2 * b^2 = 2 * (a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1476_147653


namespace NUMINAMATH_CALUDE_pascal_triangle_52nd_number_l1476_147679

/-- The number of elements in the row of Pascal's triangle we're considering --/
def row_length : ℕ := 55

/-- The index of the number we're looking for in the row (0-indexed) --/
def target_index : ℕ := 51

/-- The row number in Pascal's triangle (0-indexed) --/
def row_number : ℕ := row_length - 1

/-- The binomial coefficient we need to calculate --/
def pascal_number : ℕ := Nat.choose row_number target_index

theorem pascal_triangle_52nd_number : pascal_number = 24804 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_52nd_number_l1476_147679


namespace NUMINAMATH_CALUDE_flagpole_height_l1476_147615

/-- Given a flagpole and a building under similar shadow-casting conditions,
    calculate the height of the flagpole. -/
theorem flagpole_height
  (flagpole_shadow : ℝ)
  (building_height : ℝ)
  (building_shadow : ℝ)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_height : building_height = 22)
  (h_building_shadow : building_shadow = 55)
  : ∃ (flagpole_height : ℝ),
    flagpole_height / flagpole_shadow = building_height / building_shadow ∧
    flagpole_height = 18 := by
  sorry

end NUMINAMATH_CALUDE_flagpole_height_l1476_147615


namespace NUMINAMATH_CALUDE_impossible_to_raise_average_l1476_147681

def current_scores : List ℝ := [82, 75, 88, 91, 78]
def max_score : ℝ := 100
def target_increase : ℝ := 5

theorem impossible_to_raise_average (scores : List ℝ) (max_score : ℝ) (target_increase : ℝ) :
  let current_avg := scores.sum / scores.length
  let new_sum := scores.sum + max_score
  let new_avg := new_sum / (scores.length + 1)
  new_avg < current_avg + target_increase :=
by sorry

end NUMINAMATH_CALUDE_impossible_to_raise_average_l1476_147681


namespace NUMINAMATH_CALUDE_square_of_1024_l1476_147621

theorem square_of_1024 : (1024 : ℕ)^2 = 1048576 := by
  have h1 : (1024 : ℕ) = 10^3 + 24 := by sorry
  sorry

end NUMINAMATH_CALUDE_square_of_1024_l1476_147621


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_equality_l1476_147671

/-- The minimum value of 1/m + 3/n given the conditions -/
theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hmn : m * n > 0) (h_line : m * 2 + n * 2 = 1) : 
  (1 / m + 3 / n : ℝ) ≥ 5 + 2 * Real.sqrt 6 := by
  sorry

/-- The conditions for equality in the minimum value theorem -/
theorem min_value_equality (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hmn : m * n > 0) (h_line : m * 2 + n * 2 = 1) : 
  (1 / m + 3 / n : ℝ) = 5 + 2 * Real.sqrt 6 ↔ m = Real.sqrt 3 / 3 ∧ n = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_equality_l1476_147671


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l1476_147699

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Theorem statement
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : EvenFunction f) :
  EvenFunction (f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l1476_147699


namespace NUMINAMATH_CALUDE_subset_existence_l1476_147690

theorem subset_existence (X : Finset ℕ) (hX : X.card = 20) :
  ∀ (f : Finset ℕ → ℕ),
  (∀ S : Finset ℕ, S ⊆ X → S.card = 9 → f S ∈ X) →
  ∃ Y : Finset ℕ, Y ⊆ X ∧ Y.card = 10 ∧
  ∀ k ∈ Y, f (Y \ {k}) ≠ k :=
by sorry

end NUMINAMATH_CALUDE_subset_existence_l1476_147690


namespace NUMINAMATH_CALUDE_plane_contains_points_and_normalized_l1476_147685

def point1 : ℝ × ℝ × ℝ := (2, -1, 3)
def point2 : ℝ × ℝ × ℝ := (0, 3, 1)
def point3 : ℝ × ℝ × ℝ := (-1, 2, 4)

def plane_equation (x y z : ℝ) := 5*x + 2*y + 3*z - 17

theorem plane_contains_points_and_normalized :
  (plane_equation point1.1 point1.2.1 point1.2.2 = 0) ∧
  (plane_equation point2.1 point2.2.1 point2.2.2 = 0) ∧
  (plane_equation point3.1 point3.2.1 point3.2.2 = 0) ∧
  (5 > 0) ∧
  (Nat.gcd (Nat.gcd (Nat.gcd 5 2) 3) 17 = 1) := by
  sorry

end NUMINAMATH_CALUDE_plane_contains_points_and_normalized_l1476_147685


namespace NUMINAMATH_CALUDE_car_dealership_hourly_wage_l1476_147670

/-- Calculates the hourly wage for employees in a car dealership --/
theorem car_dealership_hourly_wage :
  let fiona_weekly_hours : ℕ := 40
  let john_weekly_hours : ℕ := 30
  let jeremy_weekly_hours : ℕ := 25
  let weeks_per_month : ℕ := 4
  let total_monthly_pay : ℕ := 7600

  let total_monthly_hours : ℕ := 
    (fiona_weekly_hours + john_weekly_hours + jeremy_weekly_hours) * weeks_per_month

  (total_monthly_pay : ℚ) / total_monthly_hours = 20 := by
  sorry


end NUMINAMATH_CALUDE_car_dealership_hourly_wage_l1476_147670


namespace NUMINAMATH_CALUDE_simplify_sqrt_x_squared_y_l1476_147642

theorem simplify_sqrt_x_squared_y (x y : ℝ) (h : x * y < 0) :
  Real.sqrt (x^2 * y) = -x * Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_x_squared_y_l1476_147642


namespace NUMINAMATH_CALUDE_sum_of_five_integers_l1476_147627

theorem sum_of_five_integers (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (4 - a) * (4 - b) * (4 - c) * (4 - d) * (4 - e) = 12 →
  a + b + c + d + e = 17 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_five_integers_l1476_147627


namespace NUMINAMATH_CALUDE_power_division_l1476_147654

theorem power_division (a : ℝ) : a^6 / a^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l1476_147654


namespace NUMINAMATH_CALUDE_fraction_simplification_l1476_147669

theorem fraction_simplification :
  ((3^12)^2 - (3^10)^2) / ((3^11)^2 - (3^9)^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1476_147669


namespace NUMINAMATH_CALUDE_female_half_marathon_count_half_marathon_probability_no_significant_relation_l1476_147647

/-- Represents the number of students in each category --/
structure StudentCounts where
  male_half_marathon : ℕ
  male_mini_run : ℕ
  female_half_marathon : ℕ
  female_mini_run : ℕ

/-- The given student counts --/
def given_counts : StudentCounts := {
  male_half_marathon := 20,
  male_mini_run := 10,
  female_half_marathon := 10,  -- This is 'a', which we'll prove
  female_mini_run := 10
}

/-- The ratio of male to female students --/
def male_female_ratio : ℚ := 3 / 2

/-- Theorem stating the correct number of female students in half marathon --/
theorem female_half_marathon_count :
  given_counts.female_half_marathon = 10 := by sorry

/-- Theorem stating the probability of choosing half marathon --/
theorem half_marathon_probability :
  (given_counts.male_half_marathon + given_counts.female_half_marathon : ℚ) /
  (given_counts.male_half_marathon + given_counts.male_mini_run +
   given_counts.female_half_marathon + given_counts.female_mini_run) = 3 / 5 := by sorry

/-- Chi-square statistic calculation --/
def chi_square (c : StudentCounts) : ℚ :=
  let n := c.male_half_marathon + c.male_mini_run + c.female_half_marathon + c.female_mini_run
  let ad := c.male_half_marathon * c.female_mini_run
  let bc := c.male_mini_run * c.female_half_marathon
  n * (ad - bc)^2 / ((c.male_half_marathon + c.male_mini_run) *
                     (c.female_half_marathon + c.female_mini_run) *
                     (c.male_half_marathon + c.female_half_marathon) *
                     (c.male_mini_run + c.female_mini_run))

/-- Theorem stating that the chi-square statistic is less than the critical value --/
theorem no_significant_relation :
  chi_square given_counts < 2706 / 1000 := by sorry

end NUMINAMATH_CALUDE_female_half_marathon_count_half_marathon_probability_no_significant_relation_l1476_147647


namespace NUMINAMATH_CALUDE_rational_function_equation_l1476_147643

theorem rational_function_equation (f : ℚ → ℚ) :
  (∀ x y : ℚ, f (x + f y) = f x + y) →
  (∀ x : ℚ, f x = x ∨ f x = -x) := by
sorry

end NUMINAMATH_CALUDE_rational_function_equation_l1476_147643


namespace NUMINAMATH_CALUDE_boys_girls_difference_l1476_147667

theorem boys_girls_difference (x y : ℕ) (a b : ℚ) : 
  x > y → 
  x * a + y * b = x * b + y * a - 1 → 
  x = y + 1 := by
sorry

end NUMINAMATH_CALUDE_boys_girls_difference_l1476_147667


namespace NUMINAMATH_CALUDE_liza_age_is_14_liza_older_than_nastya_liza_triple_nastya_two_years_ago_l1476_147624

/-- The age difference between Liza and Nastya -/
def age_difference : ℕ := 8

/-- Liza's current age -/
def liza_age : ℕ := 14

/-- Nastya's current age -/
def nastya_age : ℕ := liza_age - age_difference

theorem liza_age_is_14 : liza_age = 14 := by sorry

theorem liza_older_than_nastya : liza_age = nastya_age + age_difference := by sorry

theorem liza_triple_nastya_two_years_ago : 
  liza_age - 2 = 3 * (nastya_age - 2) := by sorry

end NUMINAMATH_CALUDE_liza_age_is_14_liza_older_than_nastya_liza_triple_nastya_two_years_ago_l1476_147624


namespace NUMINAMATH_CALUDE_unique_four_digit_reverse_9multiple_l1476_147687

/-- Reverses a four-digit number -/
def reverse (n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d3 := n / 1000
  d0 * 1000 + d1 * 100 + d2 * 10 + d3

/-- A four-digit number is a natural number between 1000 and 9999 -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem unique_four_digit_reverse_9multiple :
  ∃! n : ℕ, is_four_digit n ∧ 9 * n = reverse n :=
by
  -- The proof goes here
  sorry

#eval reverse 1089  -- Expected output: 9801
#eval 9 * 1089      -- Expected output: 9801

end NUMINAMATH_CALUDE_unique_four_digit_reverse_9multiple_l1476_147687


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1476_147662

/-- The parabola y = 2x^2 -/
def parabola (x y : ℝ) : Prop := y = 2 * x^2

/-- The point A(-1, 2) -/
def point_A : ℝ × ℝ := (-1, 2)

/-- The line l: 4x + y + 2 = 0 -/
def line_l (x y : ℝ) : Prop := 4 * x + y + 2 = 0

/-- Theorem stating that the line l passes through point A and is tangent to the parabola -/
theorem line_tangent_to_parabola :
  line_l point_A.1 point_A.2 ∧
  parabola point_A.1 point_A.2 ∧
  ∃ (t : ℝ), t ≠ point_A.1 ∧
    (∀ (x y : ℝ), x ≠ point_A.1 → line_l x y → parabola x y → x = t) :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1476_147662


namespace NUMINAMATH_CALUDE_rational_opposite_and_number_line_order_l1476_147675

-- Define the concept of opposite for rational numbers
def opposite (a : ℚ) : ℚ := -a

-- Define a property for the order of numbers on a number line
def left_of (x y : ℝ) : Prop := x < y

theorem rational_opposite_and_number_line_order :
  (∀ a : ℚ, opposite a = -a) ∧
  (∀ x y : ℝ, x ≠ y → (left_of x y ↔ x < y)) :=
sorry

end NUMINAMATH_CALUDE_rational_opposite_and_number_line_order_l1476_147675


namespace NUMINAMATH_CALUDE_infinitely_many_commuting_functions_l1476_147676

/-- A bijective function from ℝ to ℝ -/
def BijectiveFunc := {f : ℝ → ℝ // Function.Bijective f}

/-- The set of functions g that satisfy f(g(x)) = g(f(x)) for all x -/
def CommutingFunctions (f : BijectiveFunc) :=
  {g : ℝ → ℝ | ∀ x, f.val (g x) = g (f.val x)}

/-- The theorem stating that there are infinitely many commuting functions -/
theorem infinitely_many_commuting_functions (f : BijectiveFunc) :
  Set.Infinite (CommutingFunctions f) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_commuting_functions_l1476_147676


namespace NUMINAMATH_CALUDE_donation_conversion_l1476_147660

theorem donation_conversion (usd_donation : ℝ) (exchange_rate : ℝ) (cny_donation : ℝ) : 
  usd_donation = 1.2 →
  exchange_rate = 6.25 →
  cny_donation = usd_donation * exchange_rate →
  cny_donation = 7.5 :=
by sorry

end NUMINAMATH_CALUDE_donation_conversion_l1476_147660


namespace NUMINAMATH_CALUDE_factorization_equality_l1476_147657

theorem factorization_equality (x : ℝ) : 3 * x^2 * (x - 5) + 5 * (x - 5) = (3 * x^2 + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1476_147657


namespace NUMINAMATH_CALUDE_gcd_9011_4403_l1476_147637

theorem gcd_9011_4403 : Nat.gcd 9011 4403 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_9011_4403_l1476_147637


namespace NUMINAMATH_CALUDE_bennys_cards_l1476_147696

theorem bennys_cards (x : ℕ) : 
  (x + 4) / 2 = 34 → x = 68 := by sorry

end NUMINAMATH_CALUDE_bennys_cards_l1476_147696


namespace NUMINAMATH_CALUDE_no_rational_roots_l1476_147673

theorem no_rational_roots :
  ∀ (q : ℚ), 3 * q^4 - 2 * q^3 - 15 * q^2 + 6 * q + 3 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_l1476_147673


namespace NUMINAMATH_CALUDE_expression_simplification_l1476_147604

theorem expression_simplification (a c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) (hac : a ≠ c) :
  (c^2 - a^2) / (c * a) - (c * a - c^2) / (c * a - a^2) = (2 * c^2 - a^2) / (c * a) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1476_147604


namespace NUMINAMATH_CALUDE_parabola_chord_length_l1476_147622

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line
def line (x y : ℝ) : Prop := y = x - 1

-- Theorem statement
theorem parabola_chord_length :
  ∀ p : ℝ,
  p > 0 →
  (∃ x y : ℝ, parabola p x y ∧ x = 1 ∧ y = 0) →  -- Focus at (1, 0)
  (∃ A B : ℝ × ℝ,
    parabola p A.1 A.2 ∧ 
    parabola p B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    A ≠ B) →
  (∀ x y : ℝ, parabola p x y ↔ y^2 = 2*x) ∧     -- Standard equation
  (∃ A B : ℝ × ℝ,
    parabola p A.1 A.2 ∧ 
    parabola p B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4) -- Chord length
  := by sorry

end NUMINAMATH_CALUDE_parabola_chord_length_l1476_147622


namespace NUMINAMATH_CALUDE_smallest_d_inequality_l1476_147635

theorem smallest_d_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  ∃ (d : ℝ), d > 0 ∧ 
  (∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → Real.sqrt (x^2 * y^2) + d * |x^2 - y^2| + x + y ≥ x^2 + y^2) ∧
  (∀ (d' : ℝ), d' > 0 → 
    (∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → Real.sqrt (x^2 * y^2) + d' * |x^2 - y^2| + x + y ≥ x^2 + y^2) → 
    d ≤ d') ∧
  d = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_inequality_l1476_147635


namespace NUMINAMATH_CALUDE_simplify_expression_l1476_147682

theorem simplify_expression (x y : ℚ) (hx : x = 10) (hy : y = -1/25) :
  ((x * y + 2) * (x * y - 2) - 2 * x^2 * y^2 + 4) / (x * y) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1476_147682


namespace NUMINAMATH_CALUDE_reflection_x_axis_example_l1476_147641

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Reflection of a point across the x-axis -/
def reflect_x_axis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

/-- The theorem stating that the reflection of (3, 4, -5) across the x-axis is (3, -4, 5) -/
theorem reflection_x_axis_example : 
  reflect_x_axis { x := 3, y := 4, z := -5 } = { x := 3, y := -4, z := 5 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_x_axis_example_l1476_147641


namespace NUMINAMATH_CALUDE_expression_value_l1476_147632

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 2) :
  3 * x^2 - 4 * y + 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1476_147632


namespace NUMINAMATH_CALUDE_min_value_of_bisecting_line_l1476_147607

/-- A line that bisects the circumference of a circle -/
structure BisectingLine where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  bisects : ∀ x y : ℝ, a * x + 2 * b * y - 2 = 0 → x^2 + y^2 - 4*x - 2*y - 8 = 0

/-- The minimum value of 1/a + 2/b for a bisecting line -/
theorem min_value_of_bisecting_line (l : BisectingLine) : 
  ∃ (m : ℝ), (∀ a b : ℝ, a > 0 → b > 0 → 
    (∀ x y : ℝ, a * x + 2 * b * y - 2 = 0 → x^2 + y^2 - 4*x - 2*y - 8 = 0) → 
    1/a + 2/b ≥ m) ∧ 
  1/l.a + 2/l.b = m ∧ 
  m = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_bisecting_line_l1476_147607


namespace NUMINAMATH_CALUDE_cos_pi_twelfth_l1476_147659

theorem cos_pi_twelfth : Real.cos (π / 12) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_twelfth_l1476_147659


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l1476_147672

/-- A decagon is a polygon with 10 vertices -/
def Decagon : ℕ := 10

/-- The number of ways to choose 3 distinct vertices from a decagon -/
def TotalChoices : ℕ := Nat.choose Decagon 3

/-- The number of ways to choose 3 distinct vertices that form a triangle with sides as edges -/
def FavorableChoices : ℕ := Decagon

/-- The probability of choosing 3 distinct vertices that form a triangle with sides as edges -/
def ProbabilityOfTriangle : ℚ := FavorableChoices / TotalChoices

theorem decagon_triangle_probability :
  ProbabilityOfTriangle = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l1476_147672


namespace NUMINAMATH_CALUDE_sum_of_terms_l1476_147698

theorem sum_of_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  (∀ n : ℕ, S n = n^2 + n + 1) →
  (∀ n : ℕ, S (n + 1) - S n = a (n + 1)) →
  a 8 + a 9 + a 10 + a 11 + a 12 = 100 := by
sorry

end NUMINAMATH_CALUDE_sum_of_terms_l1476_147698


namespace NUMINAMATH_CALUDE_count_three_digit_even_numbers_l1476_147689

/-- The set of available digits -/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- A function that checks if a number is even -/
def isEven (n : Nat) : Bool := n % 2 = 0

/-- A function that checks if a number has three distinct digits -/
def hasThreeDistinctDigits (n : Nat) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

/-- The main theorem -/
theorem count_three_digit_even_numbers : 
  (Finset.filter (fun n => n ≥ 100 ∧ n < 1000 ∧ isEven n ∧ hasThreeDistinctDigits n ∧ 
    (∀ d, d ∈ digits → (n / 100 = d ∨ (n / 10) % 10 = d ∨ n % 10 = d)))
    (Finset.range 1000)).card = 52 := by
  sorry

end NUMINAMATH_CALUDE_count_three_digit_even_numbers_l1476_147689


namespace NUMINAMATH_CALUDE_right_triangle_properties_l1476_147617

theorem right_triangle_properties (a b c h : ℝ) : 
  a = 12 → b = 5 → c^2 = a^2 + b^2 → (1/2) * a * b = (1/2) * c * h →
  c = 13 ∧ h = 60/13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_properties_l1476_147617


namespace NUMINAMATH_CALUDE_maddies_mom_coffee_cost_l1476_147680

/-- Represents the weekly coffee consumption and cost for Maddie's mom -/
structure CoffeeConsumption where
  cups_per_day : ℕ
  beans_per_cup : ℚ
  beans_per_bag : ℚ
  cost_per_bag : ℚ
  milk_per_week : ℚ
  cost_per_gallon_milk : ℚ

/-- Calculates the weekly cost of coffee -/
def weekly_coffee_cost (c : CoffeeConsumption) : ℚ :=
  let beans_per_week := c.cups_per_day * c.beans_per_cup * 7
  let bags_per_week := beans_per_week / c.beans_per_bag
  let coffee_cost := bags_per_week * c.cost_per_bag
  let milk_cost := c.milk_per_week * c.cost_per_gallon_milk
  coffee_cost + milk_cost

/-- Theorem stating that Maddie's mom's weekly coffee cost is $18 -/
theorem maddies_mom_coffee_cost :
  let c : CoffeeConsumption := {
    cups_per_day := 2,
    beans_per_cup := 3/2,
    beans_per_bag := 21/2,
    cost_per_bag := 8,
    milk_per_week := 1/2,
    cost_per_gallon_milk := 4
  }
  weekly_coffee_cost c = 18 := by
  sorry

end NUMINAMATH_CALUDE_maddies_mom_coffee_cost_l1476_147680


namespace NUMINAMATH_CALUDE_task_force_count_l1476_147658

theorem task_force_count (total : ℕ) (executives : ℕ) (task_force_size : ℕ) (min_executives : ℕ) :
  total = 12 →
  executives = 5 →
  task_force_size = 5 →
  min_executives = 2 →
  (Nat.choose total task_force_size -
   (Nat.choose (total - executives) task_force_size +
    Nat.choose executives 1 * Nat.choose (total - executives) (task_force_size - 1))) = 596 := by
  sorry

end NUMINAMATH_CALUDE_task_force_count_l1476_147658


namespace NUMINAMATH_CALUDE_moores_law_2010_l1476_147677

/-- Moore's law doubling period in months -/
def doubling_period : ℕ := 18

/-- Initial number of transistors in 1995 -/
def initial_transistors : ℕ := 2500000

/-- Number of months between 1995 and 2010 -/
def months_elapsed : ℕ := (2010 - 1995) * 12

/-- Number of doublings that occurred between 1995 and 2010 -/
def num_doublings : ℕ := months_elapsed / doubling_period

/-- Calculates the number of transistors after a given number of doublings -/
def transistors_after_doublings (initial : ℕ) (doublings : ℕ) : ℕ :=
  initial * (2^doublings)

theorem moores_law_2010 :
  transistors_after_doublings initial_transistors num_doublings = 2560000000 := by
  sorry

end NUMINAMATH_CALUDE_moores_law_2010_l1476_147677


namespace NUMINAMATH_CALUDE_robin_gum_count_l1476_147638

/-- Calculates the total number of gum pieces given the number of packages, pieces per package, and extra pieces. -/
def total_gum_pieces (packages : ℕ) (pieces_per_package : ℕ) (extra_pieces : ℕ) : ℕ :=
  packages * pieces_per_package + extra_pieces

/-- Proves that Robin has 997 pieces of gum given the specified conditions. -/
theorem robin_gum_count :
  total_gum_pieces 43 23 8 = 997 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_count_l1476_147638


namespace NUMINAMATH_CALUDE_scalene_to_right_triangle_l1476_147602

/-- 
For any scalene triangle with sides a < b < c, there exists a real number x 
such that the new triangle with sides (a+x), (b+x), and (c+x) is a right triangle.
-/
theorem scalene_to_right_triangle 
  (a b c : ℝ) 
  (h_scalene : a < b ∧ b < c) : 
  ∃ x : ℝ, (a + x)^2 + (b + x)^2 = (c + x)^2 := by
  sorry

end NUMINAMATH_CALUDE_scalene_to_right_triangle_l1476_147602


namespace NUMINAMATH_CALUDE_existence_of_intersection_point_l1476_147633

/-- Represents a circle on a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Represents a point on a plane -/
def Point : Type := ℝ × ℝ

/-- Checks if a point is outside a circle -/
def is_outside (p : Point) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 > c.radius^2

/-- Represents a line on a plane -/
structure Line where
  point : Point
  direction : ℝ × ℝ
  non_zero : direction ≠ (0, 0)

/-- Checks if a line intersects a circle -/
def intersects (l : Line) (c : Circle) : Prop :=
  ∃ t : ℝ, is_outside (l.point.1 + t * l.direction.1, l.point.2 + t * l.direction.2) c = false

/-- Main theorem: There exists a point outside both circles such that 
    any line passing through it intersects at least one of the circles -/
theorem existence_of_intersection_point (c1 c2 : Circle) 
  (h : ∀ p : Point, ¬(is_outside p c1 ∧ is_outside p c2)) : 
  ∃ p : Point, is_outside p c1 ∧ is_outside p c2 ∧ 
    ∀ l : Line, l.point = p → (intersects l c1 ∨ intersects l c2) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_intersection_point_l1476_147633


namespace NUMINAMATH_CALUDE_line_slope_45_degrees_l1476_147608

theorem line_slope_45_degrees (y : ℝ) : 
  (∃ (line : Set (ℝ × ℝ)), 
    (⟨4, y⟩ ∈ line) ∧ 
    (⟨2, -3⟩ ∈ line) ∧ 
    (∀ (x₁ y₁ x₂ y₂ : ℝ), ⟨x₁, y₁⟩ ∈ line → ⟨x₂, y₂⟩ ∈ line → x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) = 1)) →
  y = -1 := by
sorry

end NUMINAMATH_CALUDE_line_slope_45_degrees_l1476_147608


namespace NUMINAMATH_CALUDE_butterfat_percentage_in_cream_l1476_147625

/-- The percentage of butterfat in cream when mixed with skim milk to achieve a target butterfat percentage -/
theorem butterfat_percentage_in_cream 
  (cream_volume : ℝ) 
  (skim_milk_volume : ℝ) 
  (skim_milk_butterfat : ℝ) 
  (final_mixture_butterfat : ℝ) 
  (h1 : cream_volume = 1)
  (h2 : skim_milk_volume = 3)
  (h3 : skim_milk_butterfat = 5.5)
  (h4 : final_mixture_butterfat = 6.5)
  (h5 : cream_volume + skim_milk_volume = 4) :
  ∃ (cream_butterfat : ℝ), 
    cream_butterfat = 9.5 ∧ 
    cream_butterfat * cream_volume + skim_milk_butterfat * skim_milk_volume = 
    final_mixture_butterfat * (cream_volume + skim_milk_volume) := by
  sorry


end NUMINAMATH_CALUDE_butterfat_percentage_in_cream_l1476_147625


namespace NUMINAMATH_CALUDE_square_difference_l1476_147650

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 45) (h2 : x * y = 10) :
  (x - y)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1476_147650


namespace NUMINAMATH_CALUDE_random_variable_iff_preimage_singleton_l1476_147656

variable {Ω : Type*} [MeasurableSpace Ω]
variable {E : Set ℝ} (hE : Countable E)
variable (ξ : Ω → ℝ) (hξ : ∀ ω, ξ ω ∈ E)

theorem random_variable_iff_preimage_singleton :
  Measurable ξ ↔ ∀ x ∈ E, MeasurableSet {ω | ξ ω = x} := by
  sorry

end NUMINAMATH_CALUDE_random_variable_iff_preimage_singleton_l1476_147656


namespace NUMINAMATH_CALUDE_quadratic_roots_l1476_147616

-- Define the quadratic expression
def quadratic (c : ℝ) (x : ℝ) : ℝ := -x^2 + c*x + 8

-- State the theorem
theorem quadratic_roots (c : ℝ) : 
  (∀ x, quadratic c x > 0 ↔ x < -2 ∨ x > 4) → c = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1476_147616


namespace NUMINAMATH_CALUDE_range_of_m_l1476_147618

/-- The quadratic function p(x) = x^2 + 2x - m -/
def p (m : ℝ) (x : ℝ) : Prop := x^2 + 2*x - m > 0

/-- Given p(x): x^2 + 2x - m > 0, if p(1) is false and p(2) is true, 
    then the range of values for m is [3, 8) -/
theorem range_of_m (m : ℝ) : 
  (¬ p m 1) ∧ (p m 2) → 3 ≤ m ∧ m < 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1476_147618


namespace NUMINAMATH_CALUDE_tan_half_sum_l1476_147693

theorem tan_half_sum (a b : ℝ) 
  (h1 : Real.cos a + Real.cos b = (1 : ℝ) / 2)
  (h2 : Real.sin a + Real.sin b = (3 : ℝ) / 11) : 
  Real.tan ((a + b) / 2) = (6 : ℝ) / 11 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_sum_l1476_147693


namespace NUMINAMATH_CALUDE_opposite_of_negative_four_l1476_147655

theorem opposite_of_negative_four :
  ∀ x : ℤ, x + (-4) = 0 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_four_l1476_147655


namespace NUMINAMATH_CALUDE_total_fish_fillets_l1476_147606

theorem total_fish_fillets (team1 team2 team3 : ℕ) 
  (h1 : team1 = 189) 
  (h2 : team2 = 131) 
  (h3 : team3 = 180) : 
  team1 + team2 + team3 = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_fillets_l1476_147606


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_l1476_147684

theorem x_squared_plus_y_squared (x y : ℝ) : 
  x * y = 8 → x^2 * y + x * y^2 + x + y = 80 → x^2 + y^2 = 5104 / 81 := by
sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_l1476_147684


namespace NUMINAMATH_CALUDE_window_purchase_savings_l1476_147644

/-- Calculates the cost of windows given the quantity and the store's offer -/
def windowCost (quantity : ℕ) : ℕ :=
  let regularPrice := 100
  let freeWindowsPer4 := quantity / 4
  (quantity - freeWindowsPer4) * regularPrice

/-- Calculates the savings when purchasing windows together vs separately -/
def calculateSavings (dave_windows : ℕ) (doug_windows : ℕ) : ℕ :=
  let separate_cost := windowCost dave_windows + windowCost doug_windows
  let joint_cost := windowCost (dave_windows + doug_windows)
  separate_cost - joint_cost

theorem window_purchase_savings :
  calculateSavings 7 8 = 100 := by
  sorry

end NUMINAMATH_CALUDE_window_purchase_savings_l1476_147644


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1476_147695

theorem fraction_equation_solution :
  ∃! x : ℚ, (x - 3) / (x + 2) + (3*x - 9) / (x - 3) = 2 ∧ x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1476_147695


namespace NUMINAMATH_CALUDE_intersection_A_B_l1476_147649

def A : Set ℕ := {1, 2, 3, 4, 5}

def B : Set ℕ := {x : ℕ | (x - 1) * (x - 4) < 0}

theorem intersection_A_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1476_147649


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l1476_147629

/-- Given a geometric sequence of real numbers -1, a, b, c, -9, prove that b = -3 -/
theorem geometric_sequence_middle_term
  (a b c : ℝ)
  (h : ∃ (q : ℝ), q ≠ 0 ∧ 
       -1 * q = a ∧
       a * q = b ∧
       b * q = c ∧
       c * q = -9) :
  b = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l1476_147629


namespace NUMINAMATH_CALUDE_compound_has_two_hydrogen_l1476_147691

/-- Represents a chemical compound with hydrogen, carbon, and oxygen atoms. -/
structure Compound where
  hydrogen : ℕ
  carbon : ℕ
  oxygen : ℕ
  molecular_weight : ℕ

/-- Atomic weights of elements in g/mol -/
def atomic_weight (element : String) : ℕ :=
  match element with
  | "H" => 1
  | "C" => 12
  | "O" => 16
  | _ => 0

/-- Calculates the molecular weight of a compound based on its composition -/
def calculate_weight (c : Compound) : ℕ :=
  c.hydrogen * atomic_weight "H" +
  c.carbon * atomic_weight "C" +
  c.oxygen * atomic_weight "O"

/-- Theorem stating that a compound with 1 Carbon, 3 Oxygen, and 62 g/mol molecular weight has 2 Hydrogen atoms -/
theorem compound_has_two_hydrogen :
  ∀ (c : Compound),
    c.carbon = 1 →
    c.oxygen = 3 →
    c.molecular_weight = 62 →
    calculate_weight c = c.molecular_weight →
    c.hydrogen = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_compound_has_two_hydrogen_l1476_147691


namespace NUMINAMATH_CALUDE_investment_problem_l1476_147634

/-- Simple interest calculation function -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem statement for the investment problem -/
theorem investment_problem (P : ℝ) :
  (∃ r : ℝ, simple_interest P r 2 = 520 ∧ simple_interest P r 7 = 820) →
  P = 400 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l1476_147634


namespace NUMINAMATH_CALUDE_max_min_on_interval_l1476_147646

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

theorem max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = max) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = min) ∧
    max = 5 ∧ min = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_min_on_interval_l1476_147646


namespace NUMINAMATH_CALUDE_min_value_linear_program_l1476_147626

theorem min_value_linear_program :
  ∀ x y : ℝ,
  (2 * x + y - 2 ≥ 0) →
  (x - 2 * y + 4 ≥ 0) →
  (x - 1 ≤ 0) →
  ∃ (z : ℝ), z = 3 * x + 2 * y ∧ z ≥ 3 ∧ (∀ x' y' : ℝ, 
    (2 * x' + y' - 2 ≥ 0) →
    (x' - 2 * y' + 4 ≥ 0) →
    (x' - 1 ≤ 0) →
    3 * x' + 2 * y' ≥ z) :=
by sorry

end NUMINAMATH_CALUDE_min_value_linear_program_l1476_147626


namespace NUMINAMATH_CALUDE_freshman_class_size_l1476_147651

theorem freshman_class_size (N : ℕ) 
  (h1 : N > 0) 
  (h2 : 90 ≤ N) 
  (h3 : 100 ≤ N) :
  (90 : ℝ) / N * (20 : ℝ) / 100 = (20 : ℝ) / N → N = 450 := by
  sorry

end NUMINAMATH_CALUDE_freshman_class_size_l1476_147651


namespace NUMINAMATH_CALUDE_cube_coloring_l1476_147665

theorem cube_coloring (n : ℕ) (h : n > 0) : 
  (∃ (W B : ℕ), W + B = n^3 ∧ 
   3 * W = 3 * B ∧ 
   2 * W = n^3) → 
  ∃ k : ℕ, n = 2 * k :=
by sorry

end NUMINAMATH_CALUDE_cube_coloring_l1476_147665


namespace NUMINAMATH_CALUDE_f_2005_of_2_pow_2006_l1476_147661

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- f₁(k) is the square of the sum of digits of k -/
def f₁ (k : ℕ) : ℕ := (sum_of_digits k) ^ 2

/-- fₙ₊₁(k) = f₁(fₙ(k)) for n ≥ 1 -/
def f (n : ℕ) (k : ℕ) : ℕ :=
  match n with
  | 0 => k
  | n + 1 => f₁ (f n k)

/-- The main theorem to prove -/
theorem f_2005_of_2_pow_2006 : f 2005 (2^2006) = 169 := by sorry

end NUMINAMATH_CALUDE_f_2005_of_2_pow_2006_l1476_147661


namespace NUMINAMATH_CALUDE_dealership_sales_prediction_l1476_147600

/-- Represents the sales ratio of different car types -/
structure SalesRatio where
  sports : ℕ
  sedans : ℕ
  suvs : ℕ

/-- Represents the expected sales of different car types -/
structure ExpectedSales where
  sports : ℕ
  sedans : ℕ
  suvs : ℕ

/-- Given a sales ratio and expected sports car sales, calculates the expected sales of all car types -/
def calculateExpectedSales (ratio : SalesRatio) (expectedSports : ℕ) : ExpectedSales :=
  { sports := expectedSports,
    sedans := expectedSports * ratio.sedans / ratio.sports,
    suvs := expectedSports * ratio.suvs / ratio.sports }

theorem dealership_sales_prediction 
  (ratio : SalesRatio)
  (expectedSports : ℕ)
  (h1 : ratio.sports = 5)
  (h2 : ratio.sedans = 8)
  (h3 : ratio.suvs = 3)
  (h4 : expectedSports = 35) :
  let expected := calculateExpectedSales ratio expectedSports
  expected.sedans = 56 ∧ expected.suvs = 21 := by
  sorry

end NUMINAMATH_CALUDE_dealership_sales_prediction_l1476_147600


namespace NUMINAMATH_CALUDE_increasing_order_x_z_y_l1476_147663

theorem increasing_order_x_z_y (x : ℝ) (hx : 0.8 < x ∧ x < 0.9) :
  x < x^(x^x) ∧ x^(x^x) < x^x := by
  sorry

end NUMINAMATH_CALUDE_increasing_order_x_z_y_l1476_147663


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l1476_147603

/-- Geometric sequence with specified properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_arithmetic_mean : 2 * a 1 = a 2 + a 3)
  (h_a1 : a 1 = 1) :
  (∃ q : ℝ, q = -2 ∧ ∀ n : ℕ, a (n + 1) = q * a n) ∧
  (∀ n : ℕ, (Finset.range n).sum (fun i => (i + 1 : ℝ) * a (i + 1)) = (1 - (1 + 3 * n) * (-2)^n) / 9) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l1476_147603


namespace NUMINAMATH_CALUDE_unique_triangle_l1476_147686

/-- A triangle with integer side lengths a, b, c, where a ≤ b ≤ c -/
structure IntegerTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  a_le_b : a ≤ b
  b_le_c : b ≤ c

/-- The set of all integer triangles with perimeter 10 satisfying the triangle inequality -/
def ValidTriangles : Set IntegerTriangle :=
  {t : IntegerTriangle | t.a + t.b + t.c = 10 ∧ t.a + t.b > t.c}

theorem unique_triangle : ∃! t : IntegerTriangle, t ∈ ValidTriangles :=
sorry

end NUMINAMATH_CALUDE_unique_triangle_l1476_147686


namespace NUMINAMATH_CALUDE_function_extrema_implies_a_range_l1476_147630

-- Define the function f(x) = x^2 - 2x + 3
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the theorem
theorem function_extrema_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 0 a, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 a, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 a, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 a, f x = 2) ↔
  a ∈ Set.Icc 1 2 :=
sorry

end NUMINAMATH_CALUDE_function_extrema_implies_a_range_l1476_147630


namespace NUMINAMATH_CALUDE_domain_of_g_l1476_147605

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc 1 5

-- Define the new function g(x) = f(2x - 3)
def g (x : ℝ) : ℝ := f (2 * x - 3)

-- State the theorem
theorem domain_of_g :
  {x : ℝ | g x ∈ Set.range f} = Set.Icc 2 4 := by sorry

end NUMINAMATH_CALUDE_domain_of_g_l1476_147605


namespace NUMINAMATH_CALUDE_new_people_weight_l1476_147628

def group_size : ℕ := 14
def weight_increase : ℚ := 2.7
def replaced_weights : List ℚ := [35, 45, 56]

theorem new_people_weight (W : ℚ) :
  (W - (List.sum replaced_weights) + (3 * ((W / group_size) + weight_increase))) / group_size = 
  (W / group_size) + weight_increase →
  3 * ((W / group_size) + weight_increase) = 173.8 := by
  sorry

end NUMINAMATH_CALUDE_new_people_weight_l1476_147628


namespace NUMINAMATH_CALUDE_parking_space_painted_sides_sum_l1476_147652

/-- A rectangular parking space with three painted sides -/
structure ParkingSpace where
  /-- The length of the unpainted side in feet -/
  unpainted_side : ℝ
  /-- The area of the parking space in square feet -/
  area : ℝ
  /-- The width of the parking space in feet -/
  width : ℝ
  /-- Assertion that the area equals length times width -/
  area_eq : area = unpainted_side * width

/-- The sum of the lengths of the painted sides of a parking space -/
def sum_painted_sides (p : ParkingSpace) : ℝ :=
  2 * p.width + p.unpainted_side

/-- Theorem stating that for a parking space with an unpainted side of 9 feet
    and an area of 125 square feet, the sum of the painted sides is 37 feet -/
theorem parking_space_painted_sides_sum :
  ∀ (p : ParkingSpace),
  p.unpainted_side = 9 ∧ p.area = 125 →
  sum_painted_sides p = 37 :=
by
  sorry

end NUMINAMATH_CALUDE_parking_space_painted_sides_sum_l1476_147652


namespace NUMINAMATH_CALUDE_carla_school_distance_l1476_147623

theorem carla_school_distance (grocery_distance : ℝ) (soccer_distance : ℝ) 
  (mpg : ℝ) (gas_price : ℝ) (gas_spent : ℝ) :
  grocery_distance = 8 →
  soccer_distance = 12 →
  mpg = 25 →
  gas_price = 2.5 →
  gas_spent = 5 →
  ∃ (school_distance : ℝ),
    grocery_distance + school_distance + soccer_distance + 2 * school_distance = 
      (gas_spent / gas_price) * mpg ∧
    school_distance = 10 := by
  sorry

end NUMINAMATH_CALUDE_carla_school_distance_l1476_147623


namespace NUMINAMATH_CALUDE_inclination_of_vertical_line_l1476_147620

-- Define a vertical line
def vertical_line (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = a}

-- Define the inclination angle
def inclination_angle (l : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem inclination_of_vertical_line (a : ℝ) :
  inclination_angle (vertical_line a) = 90 :=
sorry

end NUMINAMATH_CALUDE_inclination_of_vertical_line_l1476_147620


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l1476_147666

/-- 
Given an arithmetic sequence where:
- a₇ is the 7th term
- d is the common difference
- a₁ is the first term
- a₂ is the second term

This theorem states that if a₇ = 17 and d = 2, then a₁ * a₂ = 35.
-/
theorem arithmetic_sequence_product (a : ℕ → ℝ) (d : ℝ) :
  (a 7 = 17) → (∀ n, a (n + 1) - a n = d) → (d = 2) → (a 1 * a 2 = 35) := by
  sorry

#check arithmetic_sequence_product

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l1476_147666


namespace NUMINAMATH_CALUDE_factor_36_minus_9x_squared_l1476_147610

theorem factor_36_minus_9x_squared (x : ℝ) : 36 - 9 * x^2 = 9 * (2 - x) * (2 + x) := by
  sorry

end NUMINAMATH_CALUDE_factor_36_minus_9x_squared_l1476_147610


namespace NUMINAMATH_CALUDE_largest_four_digit_negative_congruent_to_3_mod_29_l1476_147636

theorem largest_four_digit_negative_congruent_to_3_mod_29 :
  ∀ n : ℤ, -9999 ≤ n ∧ n < -999 ∧ n ≡ 3 [ZMOD 29] → n ≤ -1012 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_negative_congruent_to_3_mod_29_l1476_147636


namespace NUMINAMATH_CALUDE_triangle_division_possibility_l1476_147678

/-- Given a triangle ABC with sides a, b, c (where c > b > a), it is possible to construct 
    a line that divides the triangle into a quadrilateral with 2/3 of the triangle's area 
    and a smaller triangle with 1/3 of the area, if and only if c ≤ 3a. -/
theorem triangle_division_possibility (a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) :
  (∃ (x y : ℝ), 0 < x ∧ x < c ∧ 0 < y ∧ y < b ∧ 
    (x * y) / 2 = (2/3) * (a * b) / 2) ↔ c ≤ 3 * a :=
by sorry

end NUMINAMATH_CALUDE_triangle_division_possibility_l1476_147678


namespace NUMINAMATH_CALUDE_sixth_train_departure_l1476_147692

def train_departure_time (start_time : Nat) (interval : Nat) (n : Nat) : Nat :=
  start_time + (n - 1) * interval

theorem sixth_train_departure :
  let start_time := 10 * 60  -- 10:00 AM in minutes
  let interval := 30         -- 30 minutes
  let sixth_train := 6
  train_departure_time start_time interval sixth_train = 12 * 60 + 30  -- 12:30 PM in minutes
  := by sorry

end NUMINAMATH_CALUDE_sixth_train_departure_l1476_147692


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1476_147614

theorem necessary_not_sufficient_condition (a b : ℝ) : 
  (∀ a b : ℝ, a < b → a < b + 1) ∧ 
  (∃ a b : ℝ, a < b + 1 ∧ ¬(a < b)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1476_147614


namespace NUMINAMATH_CALUDE_angle_A_is_120_degrees_l1476_147639

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the condition given in the problem
def satisfiesCondition (t : Triangle) : Prop :=
  2 * t.a * Real.sin t.A = (2 * t.b + t.c) * Real.sin t.B + (2 * t.c + t.b) * Real.sin t.C

-- Theorem statement
theorem angle_A_is_120_degrees (t : Triangle) 
  (h : satisfiesCondition t) : t.A = 2 * π / 3 := by
  sorry


end NUMINAMATH_CALUDE_angle_A_is_120_degrees_l1476_147639


namespace NUMINAMATH_CALUDE_balls_per_color_l1476_147612

theorem balls_per_color 
  (total_balls : ℕ) 
  (num_colors : ℕ) 
  (h1 : total_balls = 350) 
  (h2 : num_colors = 10) 
  (h3 : total_balls % num_colors = 0) : 
  total_balls / num_colors = 35 := by
sorry

end NUMINAMATH_CALUDE_balls_per_color_l1476_147612


namespace NUMINAMATH_CALUDE_sum_of_roots_is_twelve_l1476_147640

/-- A function satisfying the symmetry property g(3+x) = g(3-x) -/
def SymmetricAboutThree (g : ℝ → ℝ) : Prop :=
  ∀ x, g (3 + x) = g (3 - x)

/-- The property that a function has exactly four distinct real roots -/
def HasFourDistinctRealRoots (g : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧
    (∀ x, g x = 0 → x = a ∨ x = b ∨ x = c ∨ x = d)

/-- The main theorem statement -/
theorem sum_of_roots_is_twelve (g : ℝ → ℝ) 
  (h1 : SymmetricAboutThree g) (h2 : HasFourDistinctRealRoots g) :
  ∃ (a b c d : ℝ), (HasFourDistinctRealRoots g → g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧
    a + b + c + d = 12 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_twelve_l1476_147640


namespace NUMINAMATH_CALUDE_milburg_grown_ups_l1476_147697

/-- The population of Milburg -/
def total_population : ℕ := 8243

/-- The number of children in Milburg -/
def children : ℕ := 2987

/-- The number of grown-ups in Milburg -/
def grown_ups : ℕ := total_population - children

/-- Theorem stating that the number of grown-ups in Milburg is 5256 -/
theorem milburg_grown_ups : grown_ups = 5256 := by
  sorry

end NUMINAMATH_CALUDE_milburg_grown_ups_l1476_147697


namespace NUMINAMATH_CALUDE_divisibility_by_16_l1476_147631

theorem divisibility_by_16 (x : ℤ) : 
  16 ∣ (9*x^2 + 29*x + 62) ↔ ∃ t : ℤ, (x = 16*t + 6 ∨ x = 16*t + 5) :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_16_l1476_147631


namespace NUMINAMATH_CALUDE_even_painted_faces_count_l1476_147668

/-- Represents a rectangular block -/
structure Block where
  length : Nat
  width : Nat
  height : Nat

/-- Counts the number of cubes with even number of painted faces in a painted block -/
def countEvenPaintedFaces (b : Block) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem even_painted_faces_count (b : Block) :
  b.length = 6 → b.width = 4 → b.height = 2 →
  countEvenPaintedFaces b = 32 := by
  sorry

end NUMINAMATH_CALUDE_even_painted_faces_count_l1476_147668


namespace NUMINAMATH_CALUDE_brownie_pan_dimensions_l1476_147683

theorem brownie_pan_dimensions :
  ∀ m n : ℕ,
    m * n = 48 →
    (m - 2) * (n - 2) = 2 * (2 * m + 2 * n - 4) →
    ((m = 4 ∧ n = 12) ∨ (m = 12 ∧ n = 4) ∨ (m = 6 ∧ n = 8) ∨ (m = 8 ∧ n = 6)) :=
by sorry

end NUMINAMATH_CALUDE_brownie_pan_dimensions_l1476_147683


namespace NUMINAMATH_CALUDE_ahsme_unanswered_questions_l1476_147674

/-- Represents the scoring system for AHSME -/
structure ScoringSystem where
  initial : ℕ
  correct : ℕ
  wrong : ℤ
  unanswered : ℕ

/-- Calculates the score based on the given scoring system and number of questions -/
def calculate_score (system : ScoringSystem) (correct wrong unanswered : ℕ) : ℤ :=
  system.initial + system.correct * correct + system.wrong * wrong + system.unanswered * unanswered

theorem ahsme_unanswered_questions 
  (new_system : ScoringSystem)
  (old_system : ScoringSystem)
  (total_questions : ℕ)
  (new_score : ℕ)
  (old_score : ℕ)
  (h_new_system : new_system = ⟨0, 5, 0, 2⟩)
  (h_old_system : old_system = ⟨30, 4, -1, 0⟩)
  (h_total_questions : total_questions = 30)
  (h_new_score : new_score = 93)
  (h_old_score : old_score = 84) :
  ∃ (correct wrong unanswered : ℕ),
    correct + wrong + unanswered = total_questions ∧
    calculate_score new_system correct wrong unanswered = new_score ∧
    calculate_score old_system correct wrong unanswered = old_score ∧
    unanswered = 9 :=
by sorry


end NUMINAMATH_CALUDE_ahsme_unanswered_questions_l1476_147674


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l1476_147601

theorem triangle_side_lengths (x y z k : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (k_ge_2 : k ≥ 2) 
  (prod_cond : x * y * z ≤ 2) 
  (sum_cond : 1 / x^2 + 1 / y^2 + 1 / z^2 < k) :
  (∃ a b c : ℝ, a = x ∧ b = y ∧ c = z ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b) ↔ 
  (2 ≤ k ∧ k ≤ 9/4) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l1476_147601
