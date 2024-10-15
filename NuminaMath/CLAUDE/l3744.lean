import Mathlib

namespace NUMINAMATH_CALUDE_seniority_ordering_l3744_374409

-- Define the colleagues
inductive Colleague
| Tom
| Jerry
| Sam

-- Define the seniority relation
def more_senior (a b : Colleague) : Prop := sorry

-- Define the statements
def statement_I : Prop := more_senior Colleague.Jerry Colleague.Tom ∧ more_senior Colleague.Jerry Colleague.Sam
def statement_II : Prop := more_senior Colleague.Sam Colleague.Tom ∨ more_senior Colleague.Sam Colleague.Jerry
def statement_III : Prop := more_senior Colleague.Jerry Colleague.Tom ∨ more_senior Colleague.Sam Colleague.Tom

-- Theorem statement
theorem seniority_ordering :
  -- Exactly one statement is true
  (statement_I ∧ ¬statement_II ∧ ¬statement_III) ∨
  (¬statement_I ∧ statement_II ∧ ¬statement_III) ∨
  (¬statement_I ∧ ¬statement_II ∧ statement_III) →
  -- Seniority relation is transitive
  (∀ a b c : Colleague, more_senior a b → more_senior b c → more_senior a c) →
  -- Seniority relation is asymmetric
  (∀ a b : Colleague, more_senior a b → ¬more_senior b a) →
  -- All colleagues have different seniorities
  (∀ a b : Colleague, a ≠ b → more_senior a b ∨ more_senior b a) →
  -- The correct seniority ordering
  more_senior Colleague.Jerry Colleague.Tom ∧ more_senior Colleague.Tom Colleague.Sam :=
sorry

end NUMINAMATH_CALUDE_seniority_ordering_l3744_374409


namespace NUMINAMATH_CALUDE_fraction_inequality_l3744_374430

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3744_374430


namespace NUMINAMATH_CALUDE_fault_line_movement_l3744_374443

/-- The total movement of a fault line over two years, given its movement in each year -/
def total_movement (movement_year1 : ℝ) (movement_year2 : ℝ) : ℝ :=
  movement_year1 + movement_year2

/-- Theorem stating that the total movement of the fault line over two years is 6.50 inches -/
theorem fault_line_movement :
  let movement_year1 : ℝ := 1.25
  let movement_year2 : ℝ := 5.25
  total_movement movement_year1 movement_year2 = 6.50 := by
  sorry

end NUMINAMATH_CALUDE_fault_line_movement_l3744_374443


namespace NUMINAMATH_CALUDE_conic_is_ellipse_iff_l3744_374401

/-- A conic section represented by the equation x^2 + 9y^2 - 6x + 27y = k --/
def conic (k : ℝ) (x y : ℝ) : Prop :=
  x^2 + 9*y^2 - 6*x + 27*y = k

/-- Predicate for a non-degenerate ellipse --/
def is_nondegenerate_ellipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b h k : ℝ), a > 0 ∧ b > 0 ∧ 
    ∀ x y, f x y ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

theorem conic_is_ellipse_iff (k : ℝ) :
  is_nondegenerate_ellipse (conic k) ↔ k > -117/4 :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_iff_l3744_374401


namespace NUMINAMATH_CALUDE_line_direction_vector_l3744_374403

/-- Given a line passing through two points and its direction vector, prove the value of 'a' -/
theorem line_direction_vector (p1 p2 : ℝ × ℝ) (a : ℝ) : 
  p1 = (-3, 7) → p2 = (2, -1) → ∃ k : ℝ, k ≠ 0 ∧ k • (p2.1 - p1.1, p2.2 - p1.2) = (a, -2) → a = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_line_direction_vector_l3744_374403


namespace NUMINAMATH_CALUDE_arctg_sum_implies_product_sum_l3744_374405

/-- Given that arctg x + arctg y + arctg z = π/2, prove that xy + yz + zx = 1 -/
theorem arctg_sum_implies_product_sum (x y z : ℝ) 
  (h : Real.arctan x + Real.arctan y + Real.arctan z = π / 2) : 
  x * y + y * z + x * z = 1 := by
  sorry

end NUMINAMATH_CALUDE_arctg_sum_implies_product_sum_l3744_374405


namespace NUMINAMATH_CALUDE_star_property_l3744_374455

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation *
def star : Element → Element → Element
  | Element.one, Element.one => Element.one
  | Element.one, Element.two => Element.three
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.four
  | Element.two, Element.one => Element.three
  | Element.two, Element.two => Element.one
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.four
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.one

theorem star_property :
  star (star Element.three Element.two) (star Element.four Element.one) = Element.one := by
  sorry

end NUMINAMATH_CALUDE_star_property_l3744_374455


namespace NUMINAMATH_CALUDE_average_age_of_nine_students_l3744_374421

theorem average_age_of_nine_students 
  (total_students : ℕ) 
  (average_age_all : ℝ) 
  (students_group1 : ℕ) 
  (average_age_group1 : ℝ) 
  (age_last_student : ℕ) 
  (h1 : total_students = 20)
  (h2 : average_age_all = 20)
  (h3 : students_group1 = 10)
  (h4 : average_age_group1 = 24)
  (h5 : age_last_student = 61) :
  let students_group2 := total_students - students_group1 - 1
  let total_age_all := average_age_all * total_students
  let total_age_group1 := average_age_group1 * students_group1
  let total_age_group2 := total_age_all - total_age_group1 - age_last_student
  (total_age_group2 / students_group2 : ℝ) = 11 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_nine_students_l3744_374421


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_inv_x_l3744_374442

theorem max_value_of_x_plus_inv_x (x : ℝ) (h : 15 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 17 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_inv_x_l3744_374442


namespace NUMINAMATH_CALUDE_factorization_equality_l3744_374415

theorem factorization_equality (x y : ℝ) : 
  3 * x^3 - 6 * x^2 * y + 3 * x * y^2 = 3 * x * (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3744_374415


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3744_374481

-- Problem 1
theorem problem_1 (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x + 2*y)^2 - (-2*x*y^2)^2 / (x*y^3) = x^2 + 4*y^2 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) (hx1 : x ≠ 1) (hx3 : x ≠ 3) :
  (x - 1) / (x - 3) * (2 - x + 2 / (x - 1)) = -x := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3744_374481


namespace NUMINAMATH_CALUDE_sqrt_cube_root_power_six_l3744_374458

theorem sqrt_cube_root_power_six : (Real.sqrt ((Real.sqrt 3) ^ 4)) ^ 6 = 729 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_cube_root_power_six_l3744_374458


namespace NUMINAMATH_CALUDE_stripe_width_for_equal_areas_l3744_374478

/-- Given a rectangle with dimensions 40 cm × 20 cm and two perpendicular stripes of equal width,
    prove that the width of the stripes for equal white and gray areas is 30 - 5√5 cm. -/
theorem stripe_width_for_equal_areas : ∃ (x : ℝ),
  x > 0 ∧ x < 20 ∧
  (40 * x + 20 * x - x^2 = (40 * 20) / 2) ∧
  x = 30 - 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_stripe_width_for_equal_areas_l3744_374478


namespace NUMINAMATH_CALUDE_p_sufficient_t_l3744_374450

-- Define the propositions
variable (p q r s t : Prop)

-- Define the conditions
axiom p_r_sufficient_q : (p → q) ∧ (r → q)
axiom s_necessary_sufficient_q : (q ↔ s)
axiom t_necessary_s : (s → t)
axiom t_sufficient_r : (t → r)

-- Theorem to prove
theorem p_sufficient_t : p → t := by sorry

end NUMINAMATH_CALUDE_p_sufficient_t_l3744_374450


namespace NUMINAMATH_CALUDE_prob_red_black_correct_l3744_374489

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)
  (h_total : total_cards = 52)
  (h_red : red_cards = 26)
  (h_black : black_cards = 26)
  (h_sum : red_cards + black_cards = total_cards)

/-- The probability of drawing one red card and one black card in the first two draws -/
def prob_red_black (d : Deck) : ℚ :=
  26 / 51

theorem prob_red_black_correct (d : Deck) : 
  prob_red_black d = 26 / 51 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_black_correct_l3744_374489


namespace NUMINAMATH_CALUDE_households_with_car_l3744_374414

theorem households_with_car (total : ℕ) (without_car_or_bike : ℕ) (with_both : ℕ) (with_bike_only : ℕ) 
  (h1 : total = 90)
  (h2 : without_car_or_bike = 11)
  (h3 : with_both = 16)
  (h4 : with_bike_only = 35) :
  total - without_car_or_bike - with_bike_only + with_both = 60 := by
  sorry

#check households_with_car

end NUMINAMATH_CALUDE_households_with_car_l3744_374414


namespace NUMINAMATH_CALUDE_hypotenuse_increase_bound_l3744_374432

theorem hypotenuse_increase_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let c := Real.sqrt (x^2 + y^2)
  let c' := Real.sqrt ((x + 1)^2 + (y + 1)^2)
  c' - c ≤ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_hypotenuse_increase_bound_l3744_374432


namespace NUMINAMATH_CALUDE_cookie_sheet_width_l3744_374453

/-- Represents a rectangle with given width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.width + r.length)

/-- Theorem: A rectangle with length 2 and perimeter 24 has width 10 -/
theorem cookie_sheet_width : 
  ∀ (r : Rectangle), r.length = 2 → r.perimeter = 24 → r.width = 10 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sheet_width_l3744_374453


namespace NUMINAMATH_CALUDE_population_growth_percentage_l3744_374404

theorem population_growth_percentage (a b c : ℝ) : 
  let growth_factor_1 := 1 + a / 100
  let growth_factor_2 := 1 + b / 100
  let growth_factor_3 := 1 + c / 100
  let total_growth := growth_factor_1 * growth_factor_2 * growth_factor_3
  (total_growth - 1) * 100 = a + b + c + (a * b + a * c + b * c) / 100 + a * b * c / 10000 := by
sorry

end NUMINAMATH_CALUDE_population_growth_percentage_l3744_374404


namespace NUMINAMATH_CALUDE_multiplier_value_l3744_374402

theorem multiplier_value (n : ℕ) (increase : ℕ) (result : ℕ) : 
  n = 14 → increase = 196 → result = 15 → n * result = n + increase :=
by
  sorry

end NUMINAMATH_CALUDE_multiplier_value_l3744_374402


namespace NUMINAMATH_CALUDE_smallest_n_for_divisible_sum_of_squares_l3744_374436

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem smallest_n_for_divisible_sum_of_squares :
  ∀ n : ℕ, n > 0 → (sum_of_squares n % 100 = 0 → n ≥ 24) ∧
  (sum_of_squares 24 % 100 = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_divisible_sum_of_squares_l3744_374436


namespace NUMINAMATH_CALUDE_complex_sum_l3744_374413

theorem complex_sum (z : ℂ) (h : z = (1/2 : ℂ) + (Complex.I * (Real.sqrt 3)/2)) :
  z + 2*z^2 + 3*z^3 + 4*z^4 + 5*z^5 + 6*z^6 = 3 - Complex.I * 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_l3744_374413


namespace NUMINAMATH_CALUDE_bottles_per_case_l3744_374477

/-- The number of bottles a case can hold, given the total daily production and number of cases required. -/
theorem bottles_per_case (total_bottles : ℕ) (total_cases : ℕ) 
  (h1 : total_bottles = 72000) 
  (h2 : total_cases = 8000) : 
  total_bottles / total_cases = 9 := by
sorry

end NUMINAMATH_CALUDE_bottles_per_case_l3744_374477


namespace NUMINAMATH_CALUDE_probability_of_middle_position_l3744_374451

theorem probability_of_middle_position (n : ℕ) (h : n = 3) :
  (2 : ℚ) / (n.factorial : ℚ) = (1 : ℚ) / 3 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_middle_position_l3744_374451


namespace NUMINAMATH_CALUDE_soda_difference_l3744_374447

theorem soda_difference (regular_soda : ℕ) (diet_soda : ℕ) 
  (h1 : regular_soda = 79) (h2 : diet_soda = 53) : 
  regular_soda - diet_soda = 26 := by
  sorry

end NUMINAMATH_CALUDE_soda_difference_l3744_374447


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l3744_374433

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define a point on the parabola in the first quadrant
def point_on_parabola (Q : ℝ × ℝ) : Prop :=
  parabola Q.1 Q.2 ∧ Q.1 > 0 ∧ Q.2 > 0

-- Define the condition for vector PQ and QF
def vector_condition (P Q : ℝ × ℝ) : Prop :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 2 * ((Q.1 - focus.1)^2 + (Q.2 - focus.2)^2)

-- Main theorem
theorem parabola_focus_distance 
  (Q : ℝ × ℝ) 
  (h1 : point_on_parabola Q) 
  (h2 : ∃ P, vector_condition P Q) : 
  (Q.1 - focus.1)^2 + (Q.2 - focus.2)^2 = (8 + 4*Real.sqrt 2)^2 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l3744_374433


namespace NUMINAMATH_CALUDE_difference_of_squares_24_13_l3744_374444

theorem difference_of_squares_24_13 : (24 + 13)^2 - (24 - 13)^2 = 407 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_24_13_l3744_374444


namespace NUMINAMATH_CALUDE_smallest_number_l3744_374462

def numbers : Finset ℚ := {5, -1/3, 0, -2}

theorem smallest_number : 
  ∀ x ∈ numbers, -2 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3744_374462


namespace NUMINAMATH_CALUDE_expression_value_l3744_374452

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |m| = 3) :
  c * d + m - (a + b) / m = 4 ∨ c * d + m - (a + b) / m = -2 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l3744_374452


namespace NUMINAMATH_CALUDE_selection_from_three_female_two_male_l3744_374408

/-- The number of ways to select one person from a group of female and male students. -/
def selection_methods (num_female : ℕ) (num_male : ℕ) : ℕ :=
  num_female + num_male

/-- Theorem: The number of ways to select one person from 3 female students and 2 male students is 5. -/
theorem selection_from_three_female_two_male :
  selection_methods 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_selection_from_three_female_two_male_l3744_374408


namespace NUMINAMATH_CALUDE_max_value_ab_squared_l3744_374439

theorem max_value_ab_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ∃ (max : ℝ), max = (4 * Real.sqrt 6) / 9 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 2 → x * y^2 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_ab_squared_l3744_374439


namespace NUMINAMATH_CALUDE_daniels_driving_speed_l3744_374472

/-- Proves that given the conditions of Daniel's driving scenario, 
    the speed for the first 32 miles on Monday is 2x miles per hour. -/
theorem daniels_driving_speed (x : ℝ) (y : ℝ) : 
  x > 0 → -- Ensure x is positive for valid speed
  96 / x = (32 / y) + (128 / x) * (3/2) →
  y = 2 * x :=
by sorry

end NUMINAMATH_CALUDE_daniels_driving_speed_l3744_374472


namespace NUMINAMATH_CALUDE_function_property_l3744_374437

def IsMonotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem function_property (f : ℝ → ℝ) (h_monotonic : IsMonotonic f) 
    (h_property : ∀ x > 0, f (f x + 2 / x) = 1) : 
  f 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3744_374437


namespace NUMINAMATH_CALUDE_fraction_calculation_l3744_374457

theorem fraction_calculation : (1 - 1/4) / (1 - 1/5) = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3744_374457


namespace NUMINAMATH_CALUDE_prob_two_red_without_replacement_prob_two_red_with_replacement_prob_at_least_one_red_with_replacement_l3744_374445

def total_balls : ℕ := 5
def red_balls : ℕ := 3
def white_balls : ℕ := 2

/-- Probability of drawing exactly 2 red balls without replacement -/
theorem prob_two_red_without_replacement :
  (Nat.choose red_balls 2 : ℚ) / (Nat.choose total_balls 2) = 3 / 10 := by sorry

/-- Probability of drawing exactly 2 red balls with replacement -/
theorem prob_two_red_with_replacement :
  (red_balls : ℚ) / total_balls * (red_balls : ℚ) / total_balls = 9 / 25 := by sorry

/-- Probability of drawing at least 1 red ball with replacement -/
theorem prob_at_least_one_red_with_replacement :
  1 - ((white_balls : ℚ) / total_balls) ^ 2 = 21 / 25 := by sorry

end NUMINAMATH_CALUDE_prob_two_red_without_replacement_prob_two_red_with_replacement_prob_at_least_one_red_with_replacement_l3744_374445


namespace NUMINAMATH_CALUDE_triangle_sine_cosine_inequality_l3744_374499

theorem triangle_sine_cosine_inequality (A B C : ℝ) (h : A + B + C = π) :
  (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) < 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_cosine_inequality_l3744_374499


namespace NUMINAMATH_CALUDE_fly_distance_l3744_374475

/-- The distance traveled by a fly between two runners --/
theorem fly_distance (joe_speed maria_speed fly_speed initial_distance : ℝ) :
  joe_speed = 10 ∧ 
  maria_speed = 8 ∧ 
  fly_speed = 15 ∧ 
  initial_distance = 3 →
  (fly_speed * initial_distance) / (joe_speed + maria_speed) = 5/2 := by
  sorry

#check fly_distance

end NUMINAMATH_CALUDE_fly_distance_l3744_374475


namespace NUMINAMATH_CALUDE_tan_135_degrees_l3744_374419

/-- Tangent of 135 degrees is -1 -/
theorem tan_135_degrees : Real.tan (135 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_135_degrees_l3744_374419


namespace NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l3744_374498

/-- A geometric sequence with positive terms and common ratio 2 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem geometric_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_prod : a 3 * a 13 = 16) :
  a 9 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l3744_374498


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3744_374474

/-- The quadratic function f(x) = x^2 + 2px + p^2 -/
def f (p : ℝ) (x : ℝ) : ℝ := x^2 + 2*p*x + p^2

theorem quadratic_minimum (p : ℝ) (hp : p > 0) (hp2 : 2*p + p^2 = 10) :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f p x_min ≤ f p x ∧ x_min = -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3744_374474


namespace NUMINAMATH_CALUDE_intersection_B_complement_A_l3744_374424

def I : Finset Nat := {1, 2, 3, 4, 5}
def A : Finset Nat := {2, 3, 5}
def B : Finset Nat := {1, 3}

theorem intersection_B_complement_A : B ∩ (I \ A) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_B_complement_A_l3744_374424


namespace NUMINAMATH_CALUDE_min_value_function_l3744_374418

theorem min_value_function (x : ℝ) (hx : x > 1) : 
  let m := (Real.tan (22.5 * π / 180)) / (1 - (Real.tan (22.5 * π / 180))^2)
  let y := 2 * m * x + 3 / (x - 1) + 1
  y ≥ 2 + 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_function_l3744_374418


namespace NUMINAMATH_CALUDE_min_value_of_z_l3744_374482

theorem min_value_of_z (x y : ℝ) :
  x^2 + 3*y^2 + 8*x - 6*y + 30 ≥ 11 ∧
  ∃ (x y : ℝ), x^2 + 3*y^2 + 8*x - 6*y + 30 = 11 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_z_l3744_374482


namespace NUMINAMATH_CALUDE_age_difference_is_four_l3744_374487

/-- The difference between the ages of Albert's parents -/
def age_difference (albert_age : ℕ) : ℕ :=
  let father_age := albert_age + 48
  let brother_age := albert_age - 2
  let mother_age := brother_age + 46
  father_age - mother_age

/-- Theorem stating that the difference between the ages of Albert's parents is 4 years -/
theorem age_difference_is_four (albert_age : ℕ) (h : albert_age ≥ 2) :
  age_difference albert_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_four_l3744_374487


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_eq_6_l3744_374427

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
axiom perpendicular_lines_slope_product (m₁ m₂ : ℝ) : 
  m₁ * m₂ = -1 ↔ (∃ (a b c d e f : ℝ), a ≠ 0 ∧ d ≠ 0 ∧ 
    (∀ x y, a*x + b*y + c = 0 ↔ y = m₁*x + (c/b)) ∧
    (∀ x y, d*x + e*y + f = 0 ↔ y = m₂*x + (f/e)))

/-- The theorem to be proved -/
theorem perpendicular_lines_m_eq_6 :
  ∀ (m : ℝ), (∀ x y, x - 2*y - 3 = 0 ↔ y = (1/2)*x - 3/2) →
             (∀ x y, m*x + 3*y - 6 = 0 ↔ y = (-m/3)*x + 2) →
             (∃ (m₁ m₂ : ℝ), m₁ * m₂ = -1 ∧ 
               (∀ x y, x - 2*y - 3 = 0 ↔ y = m₁*x - 3/2) ∧
               (∀ x y, m*x + 3*y - 6 = 0 ↔ y = m₂*x + 2)) →
             m = 6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_eq_6_l3744_374427


namespace NUMINAMATH_CALUDE_orange_distribution_theorem_l3744_374467

/-- Represents the number of oranges each person has at a given stage -/
structure OrangeDistribution :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Defines the redistribution rules for oranges -/
def redistribute (d : OrangeDistribution) : OrangeDistribution :=
  let d1 := OrangeDistribution.mk (d.a / 2) (d.b + d.a / 2) d.c
  let d2 := OrangeDistribution.mk d1.a (d1.b * 4 / 5) (d1.c + d1.b / 5)
  OrangeDistribution.mk (d2.a + d2.c / 7) d2.b (d2.c * 6 / 7)

theorem orange_distribution_theorem (initial : OrangeDistribution) :
  initial.a + initial.b + initial.c = 108 →
  let final := redistribute initial
  final.a = final.b ∧ final.b = final.c →
  initial = OrangeDistribution.mk 72 9 27 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_theorem_l3744_374467


namespace NUMINAMATH_CALUDE_percentage_of_girls_in_class_l3744_374479

theorem percentage_of_girls_in_class (B G : ℝ) :
  B > 0 →
  G > 0 →
  G + 0.5 * B = 1.5 * (0.5 * B) →
  (G / (B + G)) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_girls_in_class_l3744_374479


namespace NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l3744_374440

open Complex

theorem max_imaginary_part_of_roots (z : ℂ) (k : ℤ) :
  z^12 - z^9 + z^6 - z^3 + 1 = 0 →
  z = exp (I * Real.pi * (1/15 + 2/15 * k)) →
  ∃ θ : ℝ, -Real.pi/2 ≤ θ ∧ θ ≤ Real.pi/2 ∧
    (∀ w : ℂ, w^12 - w^9 + w^6 - w^3 + 1 = 0 →
      Complex.abs (Complex.im w) ≤ Real.sin (7*Real.pi/30)) :=
by sorry

end NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l3744_374440


namespace NUMINAMATH_CALUDE_circular_segment_area_l3744_374461

theorem circular_segment_area (r a : ℝ) (hr : r > 0) (ha : 0 < a ∧ a < 2*r) :
  let segment_area := r^2 * Real.arcsin (a / (2*r)) - (a/4) * Real.sqrt (4*r^2 - a^2)
  segment_area = r^2 * Real.arcsin (a / (2*r)) - (a/4) * Real.sqrt (4*r^2 - a^2) :=
by sorry

end NUMINAMATH_CALUDE_circular_segment_area_l3744_374461


namespace NUMINAMATH_CALUDE_binomial_identities_l3744_374446

theorem binomial_identities (n k : ℕ) (h : k ≤ n) :
  (n.factorial = n.choose k * k.factorial * (n - k).factorial) ∧
  (n.choose k = (n - 1).choose k + (n - 1).choose (k - 1)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_identities_l3744_374446


namespace NUMINAMATH_CALUDE_sum_of_possible_S_values_l3744_374441

theorem sum_of_possible_S_values : ∃ (a b c x y z : ℕ+) (S : ℕ),
  (a^2 - 2 : ℚ) / x = (b^2 - 37 : ℚ) / y ∧
  (b^2 - 37 : ℚ) / y = (c^2 - 41 : ℚ) / z ∧
  (c^2 - 41 : ℚ) / z = (a + b + c : ℚ) ∧
  S = a + b + c + x + y + z ∧
  (∀ (a' b' c' x' y' z' : ℕ+) (S' : ℕ),
    ((a'^2 - 2 : ℚ) / x' = (b'^2 - 37 : ℚ) / y' ∧
     (b'^2 - 37 : ℚ) / y' = (c'^2 - 41 : ℚ) / z' ∧
     (c'^2 - 41 : ℚ) / z' = (a' + b' + c' : ℚ) ∧
     S' = a' + b' + c' + x' + y' + z') →
    S = 98 ∨ S = 211) ∧
  (∃ (a₁ b₁ c₁ x₁ y₁ z₁ : ℕ+) (S₁ : ℕ),
    (a₁^2 - 2 : ℚ) / x₁ = (b₁^2 - 37 : ℚ) / y₁ ∧
    (b₁^2 - 37 : ℚ) / y₁ = (c₁^2 - 41 : ℚ) / z₁ ∧
    (c₁^2 - 41 : ℚ) / z₁ = (a₁ + b₁ + c₁ : ℚ) ∧
    S₁ = a₁ + b₁ + c₁ + x₁ + y₁ + z₁ ∧
    S₁ = 98) ∧
  (∃ (a₂ b₂ c₂ x₂ y₂ z₂ : ℕ+) (S₂ : ℕ),
    (a₂^2 - 2 : ℚ) / x₂ = (b₂^2 - 37 : ℚ) / y₂ ∧
    (b₂^2 - 37 : ℚ) / y₂ = (c₂^2 - 41 : ℚ) / z₂ ∧
    (c₂^2 - 41 : ℚ) / z₂ = (a₂ + b₂ + c₂ : ℚ) ∧
    S₂ = a₂ + b₂ + c₂ + x₂ + y₂ + z₂ ∧
    S₂ = 211) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_S_values_l3744_374441


namespace NUMINAMATH_CALUDE_bridge_length_l3744_374468

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 180 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 195 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_l3744_374468


namespace NUMINAMATH_CALUDE_friends_initial_money_l3744_374491

theorem friends_initial_money (your_initial_money : ℕ) (your_weekly_savings : ℕ) 
  (friend_weekly_savings : ℕ) (weeks : ℕ) :
  your_initial_money = 160 →
  your_weekly_savings = 7 →
  friend_weekly_savings = 5 →
  weeks = 25 →
  ∃ (friend_initial_money : ℕ),
    your_initial_money + your_weekly_savings * weeks = 
    friend_initial_money + friend_weekly_savings * weeks ∧
    friend_initial_money = 210 :=
by sorry

end NUMINAMATH_CALUDE_friends_initial_money_l3744_374491


namespace NUMINAMATH_CALUDE_no_cubic_linear_terms_implies_value_l3744_374426

theorem no_cubic_linear_terms_implies_value (m n : ℝ) :
  (∀ x : ℝ, m * x^3 - 2 * x^2 + 3 * x - 4 * x^3 + 5 * x^2 - n * x = 3 * x^2) →
  m^2 - 2 * m * n + n^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_cubic_linear_terms_implies_value_l3744_374426


namespace NUMINAMATH_CALUDE_simplify_expression_l3744_374410

theorem simplify_expression (x y : ℝ) : 3*x^2 - 3*(2*x^2 + 4*y) + 2*(x^2 - y) = -x^2 - 14*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3744_374410


namespace NUMINAMATH_CALUDE_units_digit_of_n_l3744_374425

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_n (m n : ℕ) :
  m * n = 23^5 →
  units_digit m = 4 →
  units_digit n = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l3744_374425


namespace NUMINAMATH_CALUDE_hilt_bee_count_l3744_374434

/-- The number of bees Mrs. Hilt saw on the first day -/
def first_day_bees : ℕ := 144

/-- The multiplier for the number of bees on the second day -/
def bee_multiplier : ℕ := 3

/-- The number of bees Mrs. Hilt saw on the second day -/
def second_day_bees : ℕ := first_day_bees * bee_multiplier

theorem hilt_bee_count : second_day_bees = 432 := by
  sorry

end NUMINAMATH_CALUDE_hilt_bee_count_l3744_374434


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l3744_374485

/-- Proves that given a bus with a speed of 42 kmph including stoppages
    and stopping for 9.6 minutes per hour, the speed excluding stoppages is 50 kmph. -/
theorem bus_speed_excluding_stoppages
  (speed_with_stoppages : ℝ)
  (stoppage_time : ℝ)
  (h1 : speed_with_stoppages = 42)
  (h2 : stoppage_time = 9.6)
  : (speed_with_stoppages * 60) / (60 - stoppage_time) = 50 := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l3744_374485


namespace NUMINAMATH_CALUDE_cat_grooming_time_is_640_l3744_374423

/-- Represents the time taken to groom a cat -/
def catGroomingTime (
  clipTime : ℕ)  -- Time to clip one nail in seconds
  (cleanEarTime : ℕ)  -- Time to clean one ear in seconds
  (shampooTime : ℕ)  -- Time to shampoo in minutes
  (clawsPerFoot : ℕ)  -- Number of claws per foot
  (feetCount : ℕ)  -- Number of feet
  (earCount : ℕ)  -- Number of ears
  (secondsPerMinute : ℕ)  -- Number of seconds in a minute
  : ℕ :=
  (clipTime * clawsPerFoot * feetCount) +  -- Time for clipping nails
  (cleanEarTime * earCount) +  -- Time for cleaning ears
  (shampooTime * secondsPerMinute)  -- Time for shampooing

theorem cat_grooming_time_is_640 :
  catGroomingTime 10 90 5 4 4 2 60 = 640 := by
  sorry

#eval catGroomingTime 10 90 5 4 4 2 60

end NUMINAMATH_CALUDE_cat_grooming_time_is_640_l3744_374423


namespace NUMINAMATH_CALUDE_max_value_expression_l3744_374497

theorem max_value_expression (a b c : ℝ) (h1 : b > a) (h2 : a > c) (h3 : b ≠ 0) :
  ((2*a + 3*b)^2 + (b - c)^2 + (2*c - a)^2) / b^2 ≤ 27 := by
sorry

end NUMINAMATH_CALUDE_max_value_expression_l3744_374497


namespace NUMINAMATH_CALUDE_intersection_point_determines_intercept_l3744_374473

/-- Given two lines that intersect at the same point as a third line, find the y-intercept of the third line -/
theorem intersection_point_determines_intercept 
  (line1 : ℝ → ℝ) (line2 : ℝ → ℝ) (line3 : ℝ → ℝ → ℝ) 
  (h1 : ∀ x, line1 x = 3 * x + 7)
  (h2 : ∀ x, line2 x = -4 * x + 1)
  (h3 : ∀ x k, line3 x k = 2 * x + k)
  (h_intersect : ∃ x y k, line1 x = y ∧ line2 x = y ∧ line3 x k = y) :
  ∃ k, k = 43 / 7 ∧ ∀ x, line3 x k = line1 x ∧ line3 x k = line2 x := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_determines_intercept_l3744_374473


namespace NUMINAMATH_CALUDE_prob_at_most_one_girl_l3744_374470

/-- The probability of selecting at most one girl when randomly choosing 3 people
    from a group of 4 boys and 2 girls is 4/5. -/
theorem prob_at_most_one_girl (total : Nat) (boys : Nat) (girls : Nat) (selected : Nat) :
  total = boys + girls →
  boys = 4 →
  girls = 2 →
  selected = 3 →
  (Nat.choose boys selected + Nat.choose boys (selected - 1) * Nat.choose girls 1) /
    Nat.choose total selected = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_most_one_girl_l3744_374470


namespace NUMINAMATH_CALUDE_two_solutions_cubic_equation_l3744_374484

theorem two_solutions_cubic_equation : 
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ s ↔ x^3 + y^2 = 2*y + 1) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_solutions_cubic_equation_l3744_374484


namespace NUMINAMATH_CALUDE_square_fraction_l3744_374435

theorem square_fraction (a b : ℕ+) (h : (a.val * b.val + 1) ∣ (a.val^2 + b.val^2)) :
  ∃ (k : ℕ), (a.val^2 + b.val^2) / (a.val * b.val + 1) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_fraction_l3744_374435


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l3744_374422

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem fourth_rectangle_area 
  (large : Rectangle) 
  (small1 small2 small3 small4 : Rectangle) 
  (h1 : small1.length + small3.length = large.length)
  (h2 : small1.width + small2.width = large.width)
  (h3 : small1.length = small2.length)
  (h4 : small1.width = small3.width)
  (h5 : area large = area small1 + area small2 + area small3 + area small4) :
  area small4 = small2.width * small3.length := by
  sorry

#check fourth_rectangle_area

end NUMINAMATH_CALUDE_fourth_rectangle_area_l3744_374422


namespace NUMINAMATH_CALUDE_system_solution_l3744_374486

-- Define the system of equations
def system_equations (n : ℕ) (x : ℕ → ℝ) : Prop :=
  (n ≥ 3) ∧
  (∀ i ∈ Finset.range (n - 1), x i ^ 3 = x ((i + 1) % n) + x ((i + 2) % n) + 1) ∧
  (x (n - 1) ^ 3 = x 0 + x 1 + 1)

-- Define the solution set
def solution_set : Set ℝ :=
  {-1, (1 + Real.sqrt 5) / 2, (1 - Real.sqrt 5) / 2}

-- Theorem statement
theorem system_solution (n : ℕ) (x : ℕ → ℝ) :
  system_equations n x →
  (∀ i ∈ Finset.range n, x i ∈ solution_set) ∧
  (∃ t ∈ solution_set, ∀ i ∈ Finset.range n, x i = t) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3744_374486


namespace NUMINAMATH_CALUDE_negative_two_to_fourth_power_l3744_374456

theorem negative_two_to_fourth_power : -2 * 2 * 2 * 2 = -2^4 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_to_fourth_power_l3744_374456


namespace NUMINAMATH_CALUDE_gym_distance_difference_l3744_374476

/-- The distance from Anthony's apartment to work in miles -/
def distance_to_work : ℝ := 10

/-- The distance from Anthony's apartment to the gym in miles -/
def distance_to_gym : ℝ := 7

/-- The distance to the gym is more than half the distance to work -/
axiom gym_further_than_half : distance_to_gym > distance_to_work / 2

theorem gym_distance_difference : distance_to_gym - distance_to_work / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gym_distance_difference_l3744_374476


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l3744_374429

theorem circumscribed_sphere_surface_area (cube_volume : ℝ) (h : cube_volume = 27) :
  let cube_side := cube_volume ^ (1/3)
  let sphere_diameter := cube_side * Real.sqrt 3
  let sphere_radius := sphere_diameter / 2
  4 * Real.pi * sphere_radius ^ 2 = 27 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l3744_374429


namespace NUMINAMATH_CALUDE_ryan_quiz_goal_l3744_374406

theorem ryan_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ)
  (mid_year_quizzes : ℕ) (mid_year_as : ℕ) (h1 : total_quizzes = 60)
  (h2 : goal_percentage = 3/4) (h3 : mid_year_quizzes = 40) (h4 : mid_year_as = 30) :
  ∃ (max_lower_grade : ℕ),
    max_lower_grade = 5 ∧
    (mid_year_as + (total_quizzes - mid_year_quizzes - max_lower_grade) : ℚ) / total_quizzes ≥ goal_percentage :=
by sorry

end NUMINAMATH_CALUDE_ryan_quiz_goal_l3744_374406


namespace NUMINAMATH_CALUDE_lisa_marble_distribution_l3744_374449

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed for Lisa's problem -/
theorem lisa_marble_distribution (num_friends : ℕ) (initial_marbles : ℕ)
  (h1 : num_friends = 14)
  (h2 : initial_marbles = 50) :
  min_additional_marbles num_friends initial_marbles = 55 := by
  sorry

end NUMINAMATH_CALUDE_lisa_marble_distribution_l3744_374449


namespace NUMINAMATH_CALUDE_max_value_of_g_l3744_374469

def g (x : ℝ) : ℝ := 4*x - x^4

theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc (-1 : ℝ) 2 ∧
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 2 → g x ≤ g c) ∧
  g c = 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l3744_374469


namespace NUMINAMATH_CALUDE_double_inequality_solution_l3744_374411

theorem double_inequality_solution (x : ℝ) : 
  (3 * x + 2 < (x + 2)^2 ∧ (x + 2)^2 < 8 * x + 1) ↔ (1 < x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l3744_374411


namespace NUMINAMATH_CALUDE_fuelUsageTheorem_l3744_374492

/-- Calculates the total fuel usage over four weeks given initial usage and percentage changes -/
def totalFuelUsage (initialUsage : ℝ) (week2Change : ℝ) (week3Change : ℝ) (week4Change : ℝ) : ℝ :=
  let week1 := initialUsage
  let week2 := week1 * (1 + week2Change)
  let week3 := week2 * (1 - week3Change)
  let week4 := week3 * (1 + week4Change)
  week1 + week2 + week3 + week4

/-- Theorem stating that the total fuel usage over four weeks is 94.85 gallons -/
theorem fuelUsageTheorem :
  totalFuelUsage 25 0.1 0.3 0.2 = 94.85 := by
  sorry

end NUMINAMATH_CALUDE_fuelUsageTheorem_l3744_374492


namespace NUMINAMATH_CALUDE_no_proper_divisor_sum_set_equality_l3744_374417

theorem no_proper_divisor_sum_set_equality (n : ℕ) : 
  (∃ (d₁ d₂ d₃ : ℕ), 1 < d₁ ∧ d₁ < n ∧ d₁ ∣ n ∧
                     1 < d₂ ∧ d₂ < n ∧ d₂ ∣ n ∧
                     1 < d₃ ∧ d₃ < n ∧ d₃ ∣ n ∧
                     d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃) →
  ¬∃ (m : ℕ), {x : ℕ | ∃ (a b : ℕ), 1 < a ∧ a < n ∧ a ∣ n ∧
                                   1 < b ∧ b < n ∧ b ∣ n ∧
                                   x = a + b} =
              {y : ℕ | 1 < y ∧ y < m ∧ y ∣ m} :=
by sorry

end NUMINAMATH_CALUDE_no_proper_divisor_sum_set_equality_l3744_374417


namespace NUMINAMATH_CALUDE_certain_number_exists_l3744_374459

theorem certain_number_exists : ∃ x : ℝ, ((x + 10) * 2 / 2)^3 - 2 = 120 / 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l3744_374459


namespace NUMINAMATH_CALUDE_tan_product_simplification_l3744_374480

theorem tan_product_simplification :
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_simplification_l3744_374480


namespace NUMINAMATH_CALUDE_min_abs_z_plus_one_l3744_374466

theorem min_abs_z_plus_one (z : ℂ) (h : Complex.abs (z^2 + 1) = Complex.abs (z * (z + Complex.I))) :
  ∃ (w : ℂ), ∀ (z : ℂ), Complex.abs (z^2 + 1) = Complex.abs (z * (z + Complex.I)) →
    Complex.abs (w + 1) ≤ Complex.abs (z + 1) ∧ Complex.abs (w + 1) = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_plus_one_l3744_374466


namespace NUMINAMATH_CALUDE_sin_minus_cos_eq_one_solution_set_l3744_374464

theorem sin_minus_cos_eq_one_solution_set :
  {x : ℝ | Real.sin (x / 2) - Real.cos (x / 2) = 1} =
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + Real.pi / 4 ∨ x = k * Real.pi + Real.pi / 2} := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_cos_eq_one_solution_set_l3744_374464


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3744_374471

theorem purely_imaginary_complex_number (a : ℝ) :
  let z : ℂ := Complex.mk (a - 1) 1
  (∀ x : ℝ, z = Complex.I * x) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3744_374471


namespace NUMINAMATH_CALUDE_probability_at_least_one_correct_l3744_374463

/-- The probability of getting at least one question right when randomly guessing 5 questions,
    each with 6 answer choices. -/
theorem probability_at_least_one_correct (n : ℕ) (choices : ℕ) : 
  n = 5 → choices = 6 → (1 - (choices - 1 : ℚ) / choices ^ n) = 4651 / 7776 := by
  sorry

#check probability_at_least_one_correct

end NUMINAMATH_CALUDE_probability_at_least_one_correct_l3744_374463


namespace NUMINAMATH_CALUDE_divisibility_by_five_l3744_374416

theorem divisibility_by_five (d : Nat) : 
  d ≤ 9 → (41830 + d) % 5 = 0 ↔ d = 0 ∨ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l3744_374416


namespace NUMINAMATH_CALUDE_max_sin_product_right_triangle_l3744_374448

/-- For any right triangle ABC with angle C = 90°, the maximum value of sin A sin B is 1/2. -/
theorem max_sin_product_right_triangle (A B C : Real) : 
  0 ≤ A ∧ 0 ≤ B ∧ -- Angles are non-negative
  A + B + C = π ∧ -- Sum of angles in a triangle is π
  C = π / 2 → -- Right angle at C
  ∀ (x y : Real), 0 ≤ x ∧ 0 ≤ y ∧ x + y + π/2 = π → 
    Real.sin A * Real.sin B ≤ Real.sin x * Real.sin y ∧
    Real.sin A * Real.sin B ≤ (1 : Real) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_sin_product_right_triangle_l3744_374448


namespace NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l3744_374454

open Real

theorem least_positive_t_for_geometric_progression (α : ℝ) (h : 0 < α ∧ α < π / 2) :
  ∃ t : ℝ, t > 0 ∧
  (∀ r : ℝ, r > 0 →
    (arcsin (sin α) = r * α ∧
     arcsin (sin (3 * α)) = r^2 * α ∧
     arcsin (sin (5 * α)) = r^3 * α ∧
     arcsin (sin (t * α)) = r^4 * α)) ∧
  (∀ s : ℝ, s > 0 →
    (∃ r : ℝ, r > 0 ∧
      arcsin (sin α) = r * α ∧
      arcsin (sin (3 * α)) = r^2 * α ∧
      arcsin (sin (5 * α)) = r^3 * α ∧
      arcsin (sin (s * α)) = r^4 * α) →
    t ≤ s) ∧
  t = 3 * (π - 5 * α) / (π - 3 * α) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l3744_374454


namespace NUMINAMATH_CALUDE_tan_sum_reciprocal_l3744_374412

theorem tan_sum_reciprocal (x y : ℝ) 
  (h1 : Real.sin x / Real.cos y + Real.sin y / Real.cos x = 2)
  (h2 : Real.cos x / Real.sin y + Real.cos y / Real.sin x = 4) :
  Real.tan x / Real.tan y + Real.tan y / Real.tan x = 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_reciprocal_l3744_374412


namespace NUMINAMATH_CALUDE_product_equality_l3744_374438

theorem product_equality : 500 * 2468 * 0.2468 * 100 = 30485120 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l3744_374438


namespace NUMINAMATH_CALUDE_heart_properties_l3744_374494

def heart (x y : ℝ) : ℝ := |x - y|

theorem heart_properties :
  ∀ x y : ℝ,
  (heart x y = heart y x) ∧
  (3 * heart x y = heart (3 * x) (3 * y)) ∧
  (heart (x + 1) (y + 1) = heart x y) ∧
  (heart x x = 0) ∧
  (heart x y ≥ 0) ∧
  (heart x y > 0 ↔ x ≠ y) := by
  sorry

end NUMINAMATH_CALUDE_heart_properties_l3744_374494


namespace NUMINAMATH_CALUDE_recipe_sugar_requirement_l3744_374490

/-- The number of cups of sugar Mary has already added to the cake. -/
def sugar_added : ℕ := 10

/-- The number of cups of sugar Mary still needs to add to the cake. -/
def sugar_to_add : ℕ := 1

/-- The total number of cups of sugar required by the recipe. -/
def total_sugar : ℕ := sugar_added + sugar_to_add

/-- The number of cups of flour required by the recipe. -/
def flour_required : ℕ := 9

/-- The number of cups of flour Mary has already added to the cake. -/
def flour_added : ℕ := 12

theorem recipe_sugar_requirement :
  total_sugar = 11 := by sorry

end NUMINAMATH_CALUDE_recipe_sugar_requirement_l3744_374490


namespace NUMINAMATH_CALUDE_sum4_equivalence_l3744_374428

-- Define the type for a die
def Die := Fin 6

-- Define the sum of two dice
def diceSum (d1 d2 : Die) : Nat := d1.val + d2.val + 2

-- Define the event where the sum is 4
def sumIs4 (d1 d2 : Die) : Prop := diceSum d1 d2 = 4

-- Define the event where one die is 3 and the other is 1
def oneThreeOneOne (d1 d2 : Die) : Prop :=
  (d1.val = 2 ∧ d2.val = 0) ∨ (d1.val = 0 ∧ d2.val = 2)

-- Define the event where both dice show 2
def bothTwo (d1 d2 : Die) : Prop := d1.val = 1 ∧ d2.val = 1

-- Theorem stating the equivalence
theorem sum4_equivalence (d1 d2 : Die) :
  sumIs4 d1 d2 ↔ oneThreeOneOne d1 d2 ∨ bothTwo d1 d2 := by
  sorry


end NUMINAMATH_CALUDE_sum4_equivalence_l3744_374428


namespace NUMINAMATH_CALUDE_economic_output_equals_scientific_notation_l3744_374488

/-- Represents the economic output in yuan -/
def economic_output : ℝ := 4500 * 1000000000

/-- The scientific notation representation of the economic output -/
def scientific_notation : ℝ := 4.5 * (10 ^ 12)

/-- Theorem stating that the economic output is equal to its scientific notation representation -/
theorem economic_output_equals_scientific_notation : 
  economic_output = scientific_notation := by sorry

end NUMINAMATH_CALUDE_economic_output_equals_scientific_notation_l3744_374488


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3744_374407

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 2 * x^2 - 5 * x + 2
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = 1/2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_equation_solution_l3744_374407


namespace NUMINAMATH_CALUDE_smallest_value_of_floor_sum_l3744_374493

theorem smallest_value_of_floor_sum (a b c : ℕ+) 
  (hab : (a : ℚ) / b = 2)
  (hbc : (b : ℚ) / c = 2)
  (hca : (c : ℚ) / a = 1 / 4) :
  ⌊(a + b : ℚ) / c⌋ + ⌊(b + c : ℚ) / a⌋ + ⌊(c + a : ℚ) / b⌋ = 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_floor_sum_l3744_374493


namespace NUMINAMATH_CALUDE_divisible_by_two_l3744_374483

theorem divisible_by_two (m n : ℕ) : 
  2 ∣ (5*m + n + 1) * (3*m - n + 4) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_two_l3744_374483


namespace NUMINAMATH_CALUDE_point_in_region_l3744_374400

theorem point_in_region (m : ℝ) :
  (m^2 - 3*m + 2 > 0) ↔ (m < 1 ∨ m > 2) :=
by sorry

end NUMINAMATH_CALUDE_point_in_region_l3744_374400


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_sum_of_digits_l3744_374431

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem largest_four_digit_divisible_by_sum_of_digits :
  ∃ (n : ℕ), is_four_digit n ∧ n % (sum_of_digits n) = 0 ∧
  ∀ (m : ℕ), is_four_digit m ∧ m % (sum_of_digits m) = 0 → m ≤ n :=
by
  use 9990
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_sum_of_digits_l3744_374431


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l3744_374496

/-- Given a positive geometric sequence {a_n} where a_2, a_3/2, a_1 form an arithmetic sequence,
    prove that (a_4 + a_5) / (a_3 + a_4) = (1 + √5) / 2 -/
theorem geometric_arithmetic_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_arithmetic : ∃ d : ℝ, a 2 - a 3 / 2 = a 3 / 2 - a 1) :
  (a 4 + a 5) / (a 3 + a 4) = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l3744_374496


namespace NUMINAMATH_CALUDE_farthest_vertex_distance_l3744_374420

/-- Given a rectangle ABCD with area 48 and diagonal 10, and a point O such that OB = OD = 13,
    the distance from O to the farthest vertex of the rectangle is 7√(29/5). -/
theorem farthest_vertex_distance (A B C D O : ℝ × ℝ) : 
  let area := abs ((B.1 - A.1) * (D.2 - A.2))
  let diagonal := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let OB_dist := Real.sqrt ((O.1 - B.1)^2 + (O.2 - B.2)^2)
  let OD_dist := Real.sqrt ((O.1 - D.1)^2 + (O.2 - D.2)^2)
  area = 48 ∧ diagonal = 10 ∧ OB_dist = 13 ∧ OD_dist = 13 →
  max (Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2))
      (max (Real.sqrt ((O.1 - B.1)^2 + (O.2 - B.2)^2))
           (max (Real.sqrt ((O.1 - C.1)^2 + (O.2 - C.2)^2))
                (Real.sqrt ((O.1 - D.1)^2 + (O.2 - D.2)^2))))
  = 7 * Real.sqrt (29/5) :=
by
  sorry


end NUMINAMATH_CALUDE_farthest_vertex_distance_l3744_374420


namespace NUMINAMATH_CALUDE_nice_people_count_l3744_374465

/-- Represents the proportion of nice people for each name --/
def nice_proportion (name : String) : ℚ :=
  match name with
  | "Barry" => 1
  | "Kevin" => 1/2
  | "Julie" => 3/4
  | "Joe" => 1/10
  | _ => 0

/-- Represents the number of people with each name in the crowd --/
def crowd_count (name : String) : ℕ :=
  match name with
  | "Barry" => 24
  | "Kevin" => 20
  | "Julie" => 80
  | "Joe" => 50
  | _ => 0

/-- Calculates the number of nice people for a given name --/
def nice_count (name : String) : ℕ :=
  (nice_proportion name * crowd_count name).num.toNat

/-- The total number of nice people in the crowd --/
def total_nice_people : ℕ :=
  nice_count "Barry" + nice_count "Kevin" + nice_count "Julie" + nice_count "Joe"

/-- Theorem stating that the total number of nice people in the crowd is 99 --/
theorem nice_people_count : total_nice_people = 99 := by
  sorry

end NUMINAMATH_CALUDE_nice_people_count_l3744_374465


namespace NUMINAMATH_CALUDE_geoff_total_spending_l3744_374495

/-- Geoff's spending on sneakers over three days -/
def geoff_spending (x : ℝ) : ℝ := x + 4*x + 5*x

/-- Theorem: Geoff's total spending over three days equals 10 times his Monday spending -/
theorem geoff_total_spending (x : ℝ) : geoff_spending x = 10 * x := by
  sorry

end NUMINAMATH_CALUDE_geoff_total_spending_l3744_374495


namespace NUMINAMATH_CALUDE_camille_bird_count_l3744_374460

/-- The number of birds Camille saw while bird watching -/
def total_birds (cardinals robins blue_jays sparrows : ℕ) : ℕ :=
  cardinals + robins + blue_jays + sparrows

/-- Theorem stating the total number of birds Camille saw -/
theorem camille_bird_count :
  ∃ (cardinals robins blue_jays sparrows : ℕ),
    cardinals = 3 ∧
    robins = 4 * cardinals ∧
    blue_jays = 2 * cardinals ∧
    sparrows = 3 * cardinals + 1 ∧
    total_birds cardinals robins blue_jays sparrows = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_camille_bird_count_l3744_374460
