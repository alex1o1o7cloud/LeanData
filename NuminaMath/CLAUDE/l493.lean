import Mathlib

namespace NUMINAMATH_CALUDE_inequality_problem_l493_49364

/-- Given an inequality and its solution set, prove the values of a and b and solve another inequality -/
theorem inequality_problem (a b c : ℝ) : 
  (∀ x, (a * x^2 - 3*x + 6 > 4) ↔ (x < 1 ∨ x > b)) →
  c > 2 →
  (a = 1 ∧ b = 2) ∧
  (∀ x, (a * x^2 - (a*c + b)*x + b*c < 0) ↔ (2 < x ∧ x < c)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l493_49364


namespace NUMINAMATH_CALUDE_sequence_correctness_l493_49367

def a (n : ℕ) : ℤ := (-1 : ℤ)^(n + 1) * n^2

theorem sequence_correctness : 
  (a 1 = 1) ∧ (a 2 = -4) ∧ (a 3 = 9) ∧ (a 4 = -16) ∧ (a 5 = 25) := by
  sorry

end NUMINAMATH_CALUDE_sequence_correctness_l493_49367


namespace NUMINAMATH_CALUDE_log_6_15_in_terms_of_a_b_l493_49319

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the logarithm with arbitrary base
noncomputable def log (base x : ℝ) : ℝ := Real.log x / Real.log base

-- Theorem statement
theorem log_6_15_in_terms_of_a_b (a b : ℝ) (h1 : lg 2 = a) (h2 : lg 3 = b) :
  log 6 15 = (b + 1 - a) / (a + b) := by
  sorry


end NUMINAMATH_CALUDE_log_6_15_in_terms_of_a_b_l493_49319


namespace NUMINAMATH_CALUDE_vector_properties_l493_49312

def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (-3, -4)

def a : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def b : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
def c : ℝ × ℝ := (A.1 - C.1, A.2 - C.2)

def M : ℝ × ℝ := (C.1 + 3*c.1, C.2 + 3*c.2)
def N : ℝ × ℝ := (C.1 - 2*b.1, C.2 - 2*b.2)

theorem vector_properties :
  (3*a.1 + b.1 - 3*c.1 = 6 ∧ 3*a.2 + b.2 - 3*c.2 = -42) ∧
  (a = (-b.1 - c.1, -b.2 - c.2)) ∧
  (M = (0, 20) ∧ N = (9, 2) ∧ (M.1 - N.1 = 9 ∧ M.2 - N.2 = -18)) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l493_49312


namespace NUMINAMATH_CALUDE_notebook_distribution_l493_49399

theorem notebook_distribution (total_notebooks : ℕ) (initial_students : ℕ) : 
  total_notebooks = 512 →
  total_notebooks = initial_students * (initial_students / 8) →
  (total_notebooks / (initial_students / 2) : ℕ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_notebook_distribution_l493_49399


namespace NUMINAMATH_CALUDE_arrange_four_men_five_women_l493_49333

/-- The number of ways to arrange people into groups -/
def arrange_groups (num_men : ℕ) (num_women : ℕ) : ℕ :=
  let three_person_group := Nat.choose num_men 2 * Nat.choose num_women 1
  let first_two_person_group := Nat.choose (num_men - 2) 1 * Nat.choose (num_women - 1) 1
  three_person_group * first_two_person_group * 1

/-- Theorem stating the number of ways to arrange 4 men and 5 women into specific groups -/
theorem arrange_four_men_five_women :
  arrange_groups 4 5 = 240 := by
  sorry


end NUMINAMATH_CALUDE_arrange_four_men_five_women_l493_49333


namespace NUMINAMATH_CALUDE_parabola_directrix_l493_49363

/-- The equation of the directrix of the parabola y² = 8x is x = -2 -/
theorem parabola_directrix (x y : ℝ) : 
  (∀ x y, y^2 = 8*x → ∃ p, p = 4 ∧ x = -p/2) → 
  ∃ k, k = -2 ∧ (∀ x y, y^2 = 8*x → x = k) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l493_49363


namespace NUMINAMATH_CALUDE_integral_extrema_l493_49360

open Real MeasureTheory

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ∫ t in (x - a)..(x + a), sqrt (4 * a^2 - t^2)

theorem integral_extrema (a : ℝ) (ha : a > 0) :
  (∀ x : ℝ, |x| ≤ a → f a x ≤ 2 * π * a^2) ∧
  (∀ x : ℝ, |x| ≤ a → f a x ≥ π * a^2) ∧
  (∃ x : ℝ, |x| ≤ a ∧ f a x = 2 * π * a^2) ∧
  (∃ x : ℝ, |x| ≤ a ∧ f a x = π * a^2) :=
sorry

end NUMINAMATH_CALUDE_integral_extrema_l493_49360


namespace NUMINAMATH_CALUDE_least_cube_divisible_by_17280_l493_49345

theorem least_cube_divisible_by_17280 (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬(17280 ∣ y^3)) ∧ (17280 ∣ x^3) ↔ x = 120 := by
  sorry

end NUMINAMATH_CALUDE_least_cube_divisible_by_17280_l493_49345


namespace NUMINAMATH_CALUDE_irrational_arithmetic_properties_l493_49330

-- Define irrational numbers
def IsIrrational (x : ℝ) : Prop := ¬ (∃ (q : ℚ), (x : ℝ) = q)

-- Theorem statement
theorem irrational_arithmetic_properties :
  (∃ (a b : ℝ), IsIrrational a ∧ IsIrrational b ∧ IsIrrational (a + b)) ∧
  (∃ (a b : ℝ), IsIrrational a ∧ IsIrrational b ∧ ∃ (q : ℚ), (a - b : ℝ) = q) ∧
  (∃ (a b : ℝ), IsIrrational a ∧ IsIrrational b ∧ ∃ (q : ℚ), (a * b : ℝ) = q) ∧
  (∃ (a b : ℝ), IsIrrational a ∧ IsIrrational b ∧ b ≠ 0 ∧ ∃ (q : ℚ), (a / b : ℝ) = q) :=
by sorry

end NUMINAMATH_CALUDE_irrational_arithmetic_properties_l493_49330


namespace NUMINAMATH_CALUDE_spade_calculation_l493_49343

-- Define the ⬥ operation
def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

-- State the theorem
theorem spade_calculation : spade 3 (spade 6 5) = -112 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l493_49343


namespace NUMINAMATH_CALUDE_study_time_problem_l493_49347

/-- The study time problem -/
theorem study_time_problem 
  (kwame_time : ℝ) 
  (lexia_time : ℝ) 
  (h1 : kwame_time = 2.5)
  (h2 : lexia_time = 97 / 60)
  (h3 : kwame_time + connor_time = lexia_time + 143 / 60) :
  connor_time = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_study_time_problem_l493_49347


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l493_49349

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ b : ℝ, (m^2 + Complex.I) * (1 + m * Complex.I) = Complex.I * b) → m = 0 ∨ m = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l493_49349


namespace NUMINAMATH_CALUDE_line_increase_theorem_l493_49361

/-- Represents a line in a Cartesian plane -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The increase in y for a given increase in x -/
def y_increase (l : Line) (x_increase : ℝ) : ℝ :=
  l.slope * x_increase

/-- Theorem: For a line with the given properties, an increase of 20 units in x
    from the point (1, 2) results in an increase of 41.8 units in y -/
theorem line_increase_theorem (l : Line) 
    (h1 : l.slope = 11 / 5)
    (h2 : 2 = l.slope * 1 + l.y_intercept) : 
    y_increase l 20 = 41.8 := by
  sorry

end NUMINAMATH_CALUDE_line_increase_theorem_l493_49361


namespace NUMINAMATH_CALUDE_magic_forest_coin_difference_l493_49393

theorem magic_forest_coin_difference :
  ∀ (x y : ℕ),
  let trees_with_no_coins := 2 * x
  let trees_with_one_coin := y
  let trees_with_two_coins := 3
  let trees_with_three_coins := x
  let trees_with_four_coins := 4
  let total_coins := y + 3 * x + 22
  let total_trees := 3 * x + y + 7
  total_coins - total_trees = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_magic_forest_coin_difference_l493_49393


namespace NUMINAMATH_CALUDE_dan_licks_l493_49337

/-- The number of licks it takes for each person to get to the center of a lollipop -/
structure LollipopLicks where
  michael : ℕ
  sam : ℕ
  david : ℕ
  lance : ℕ
  dan : ℕ

/-- The average number of licks for all five people -/
def average (l : LollipopLicks) : ℚ :=
  (l.michael + l.sam + l.david + l.lance + l.dan) / 5

/-- Theorem stating that Dan takes 58 licks to get to the center of a lollipop -/
theorem dan_licks (l : LollipopLicks) : 
  l.michael = 63 → l.sam = 70 → l.david = 70 → l.lance = 39 → average l = 60 → l.dan = 58 := by
  sorry

end NUMINAMATH_CALUDE_dan_licks_l493_49337


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l493_49346

theorem ceiling_floor_sum : ⌈(7:ℚ)/3⌉ + ⌊-(7:ℚ)/3⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l493_49346


namespace NUMINAMATH_CALUDE_complement_of_A_l493_49311

/-- Given that the universal set U is the set of real numbers and 
    A is the set of real numbers x such that 1 < x ≤ 3,
    prove that the complement of A with respect to U 
    is the set of real numbers x such that x ≤ 1 or x > 3 -/
theorem complement_of_A (U : Set ℝ) (A : Set ℝ) 
  (h_U : U = Set.univ)
  (h_A : A = {x : ℝ | 1 < x ∧ x ≤ 3}) :
  U \ A = {x : ℝ | x ≤ 1 ∨ x > 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l493_49311


namespace NUMINAMATH_CALUDE_square_perimeter_l493_49317

theorem square_perimeter (s : ℝ) (h1 : s > 0) : 
  (∃ strip_perimeter : ℝ, 
    strip_perimeter = 2 * (s + s / 4) ∧ 
    strip_perimeter = 40) →
  4 * s = 64 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l493_49317


namespace NUMINAMATH_CALUDE_cubic_roots_coefficients_relation_l493_49308

theorem cubic_roots_coefficients_relation 
  (a b c d : ℝ) (x₁ x₂ x₃ : ℝ) 
  (h : a ≠ 0) 
  (h_roots : ∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) : 
  (x₁ + x₂ + x₃ = -b / a) ∧ 
  (x₁ * x₂ + x₁ * x₃ + x₂ * x₃ = c / a) ∧ 
  (x₁ * x₂ * x₃ = -d / a) := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_coefficients_relation_l493_49308


namespace NUMINAMATH_CALUDE_stripe_area_theorem_l493_49302

/-- Represents a cylindrical silo -/
structure Cylinder where
  diameter : ℝ
  height : ℝ

/-- Represents a stripe wrapped around a cylinder -/
structure Stripe where
  width : ℝ
  revolutions : ℕ

/-- Calculates the area of a stripe wrapped around a cylinder -/
def stripeArea (c : Cylinder) (s : Stripe) : ℝ :=
  s.width * c.height

theorem stripe_area_theorem (c : Cylinder) (s : Stripe) :
  stripeArea c s = s.width * c.height := by sorry

end NUMINAMATH_CALUDE_stripe_area_theorem_l493_49302


namespace NUMINAMATH_CALUDE_alexander_payment_l493_49362

/-- The cost of tickets at an amusement park -/
def ticket_cost (child_cost adult_cost : ℕ) (alexander_child alexander_adult anna_child anna_adult : ℕ) : Prop :=
  let alexander_total := child_cost * alexander_child + adult_cost * alexander_adult
  let anna_total := child_cost * anna_child + adult_cost * anna_adult
  (child_cost = 600) ∧
  (alexander_child = 2) ∧
  (alexander_adult = 3) ∧
  (anna_child = 3) ∧
  (anna_adult = 2) ∧
  (alexander_total = anna_total + 200)

theorem alexander_payment :
  ∀ (child_cost adult_cost : ℕ),
  ticket_cost child_cost adult_cost 2 3 3 2 →
  child_cost * 2 + adult_cost * 3 = 3600 :=
by
  sorry

end NUMINAMATH_CALUDE_alexander_payment_l493_49362


namespace NUMINAMATH_CALUDE_max_value_of_y_l493_49365

noncomputable section

def angle_alpha : ℝ := Real.arctan (-Real.sqrt 3 / 3)

def point_P : ℝ × ℝ := (-3, Real.sqrt 3)

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

def f (x : ℝ) : ℝ := 
  determinant (Real.cos (x + angle_alpha)) (-Real.sin angle_alpha) (Real.sin (x + angle_alpha)) (Real.cos angle_alpha)

def y (x : ℝ) : ℝ := Real.sqrt 3 * f (Real.pi / 2 - 2 * x) + 2 * f x ^ 2

theorem max_value_of_y :
  ∃ (x : ℝ), x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4) ∧ 
  y x = 3 ∧ 
  ∀ (z : ℝ), z ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4) → y z ≤ y x :=
sorry

end

end NUMINAMATH_CALUDE_max_value_of_y_l493_49365


namespace NUMINAMATH_CALUDE_three_in_M_l493_49352

def U : Set ℤ := {x | x^2 - 6*x < 0}

theorem three_in_M (M : Set ℤ) (h : (U \ M) = {1, 2}) : 3 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_three_in_M_l493_49352


namespace NUMINAMATH_CALUDE_candy_comparison_l493_49351

/-- Represents a person with their candy bags -/
structure Person where
  name : String
  bags : List Nat

/-- Calculates the total candy for a person -/
def totalCandy (p : Person) : Nat :=
  p.bags.sum

theorem candy_comparison (sandra roger emily : Person)
  (h_sandra : sandra.bags = [6, 6])
  (h_roger : roger.bags = [11, 3])
  (h_emily : emily.bags = [4, 7, 5]) :
  totalCandy emily > totalCandy roger ∧
  totalCandy roger > totalCandy sandra ∧
  totalCandy sandra = 12 := by
  sorry

#eval totalCandy { name := "Sandra", bags := [6, 6] }
#eval totalCandy { name := "Roger", bags := [11, 3] }
#eval totalCandy { name := "Emily", bags := [4, 7, 5] }

end NUMINAMATH_CALUDE_candy_comparison_l493_49351


namespace NUMINAMATH_CALUDE_triangle_value_l493_49397

theorem triangle_value (triangle p : ℤ) 
  (h1 : triangle + p = 85)
  (h2 : (triangle + p) + 3 * p = 154) : 
  triangle = 62 := by
  sorry

end NUMINAMATH_CALUDE_triangle_value_l493_49397


namespace NUMINAMATH_CALUDE_f_negative_a_eq_zero_l493_49375

/-- Given a real-valued function f(x) = x³ + x + 1 and a real number a such that f(a) = 2,
    prove that f(-a) = 0. -/
theorem f_negative_a_eq_zero (a : ℝ) (h : a^3 + a + 1 = 2) :
  (-a)^3 + (-a) + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_eq_zero_l493_49375


namespace NUMINAMATH_CALUDE_remainder_3973_div_28_l493_49355

theorem remainder_3973_div_28 : 3973 % 28 = 9 := by sorry

end NUMINAMATH_CALUDE_remainder_3973_div_28_l493_49355


namespace NUMINAMATH_CALUDE_opposite_and_absolute_value_l493_49381

theorem opposite_and_absolute_value (x y : ℤ) :
  (- x = 3 ∧ |y| = 5) → (x + y = 2 ∨ x + y = -8) :=
by sorry

end NUMINAMATH_CALUDE_opposite_and_absolute_value_l493_49381


namespace NUMINAMATH_CALUDE_stating_special_numeral_satisfies_condition_l493_49379

/-- 
A numeral with two 1's where the difference between their place values is 99.99.
-/
def special_numeral : ℝ := 1.11

/-- 
The difference between the place values of the two 1's in the special numeral.
-/
def place_value_difference : ℝ := 99.99

/-- 
Theorem stating that the special_numeral satisfies the required condition.
-/
theorem special_numeral_satisfies_condition : 
  (100 : ℝ) - (1 / 100 : ℝ) = place_value_difference :=
by sorry

end NUMINAMATH_CALUDE_stating_special_numeral_satisfies_condition_l493_49379


namespace NUMINAMATH_CALUDE_marble_remainder_l493_49378

theorem marble_remainder (r p : ℕ) 
  (hr : r % 8 = 5) 
  (hp : p % 8 = 7) : 
  (r + p) % 8 = 4 := by
sorry

end NUMINAMATH_CALUDE_marble_remainder_l493_49378


namespace NUMINAMATH_CALUDE_range_of_a_l493_49339

-- Define the condition that |x-3|+|x+5|>a holds for any x ∈ ℝ
def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, |x - 3| + |x + 5| > a

-- State the theorem
theorem range_of_a :
  {a : ℝ | condition a} = Set.Iio 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l493_49339


namespace NUMINAMATH_CALUDE_y_intercept_for_specific_line_l493_49342

/-- A line in two-dimensional space. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept.1) + l.x_intercept.2)

/-- Theorem: For a line with slope 3 and x-intercept (4, 0), the y-intercept is (0, -12). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := 3, x_intercept := (4, 0) }
  y_intercept l = (0, -12) := by sorry

end NUMINAMATH_CALUDE_y_intercept_for_specific_line_l493_49342


namespace NUMINAMATH_CALUDE_stratified_sample_size_l493_49326

theorem stratified_sample_size 
  (total_population : ℕ) 
  (elderly_population : ℕ) 
  (elderly_sample : ℕ) 
  (n : ℕ) 
  (h1 : total_population = 162) 
  (h2 : elderly_population = 27) 
  (h3 : elderly_sample = 6) 
  (h4 : elderly_population * n = total_population * elderly_sample) : 
  n = 36 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l493_49326


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l493_49396

theorem imaginary_part_of_z (z : ℂ) (h : z + (3 - 4*I) = 1) : z.im = 4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l493_49396


namespace NUMINAMATH_CALUDE_factorization_equality_l493_49395

theorem factorization_equality (a b c : ℝ) : a^2 - 2*a*b + b^2 - c^2 = (a - b + c) * (a - b - c) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l493_49395


namespace NUMINAMATH_CALUDE_ophelias_current_age_l493_49324

/-- Represents a person's age -/
structure Person where
  age : ℕ

/-- Given Lennon's current age and the relationship between Ophelia and Lennon's ages in two years,
    prove that Ophelia's current age is 38 years. -/
theorem ophelias_current_age 
  (lennon ophelia : Person)
  (lennon_current_age : lennon.age = 8)
  (future_age_relation : ophelia.age + 2 = 4 * (lennon.age + 2)) :
  ophelia.age = 38 := by
  sorry

end NUMINAMATH_CALUDE_ophelias_current_age_l493_49324


namespace NUMINAMATH_CALUDE_equal_digit_probability_l493_49357

def num_dice : ℕ := 6
def sides_per_die : ℕ := 20
def one_digit_outcomes : ℕ := 9
def two_digit_outcomes : ℕ := 11

def prob_one_digit : ℚ := one_digit_outcomes / sides_per_die
def prob_two_digit : ℚ := two_digit_outcomes / sides_per_die

def equal_digit_prob : ℚ := (num_dice.choose (num_dice / 2)) *
  (prob_one_digit ^ (num_dice / 2)) * (prob_two_digit ^ (num_dice / 2))

theorem equal_digit_probability :
  equal_digit_prob = 4851495 / 16000000 := by sorry

end NUMINAMATH_CALUDE_equal_digit_probability_l493_49357


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l493_49370

theorem set_equality_implies_sum (a b : ℝ) : 
  ({4, a} : Set ℝ) = ({2, a * b} : Set ℝ) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l493_49370


namespace NUMINAMATH_CALUDE_negation_at_most_one_obtuse_angle_l493_49332

/-- Definition of a triangle -/
def Triangle : Type := Unit

/-- Definition of an obtuse angle in a triangle -/
def HasObtuseAngle (t : Triangle) : Prop := sorry

/-- Statement: There is at most one obtuse angle in a triangle -/
def AtMostOneObtuseAngle : Prop :=
  ∀ t : Triangle, ∃! a : ℕ, a ≤ 3 ∧ HasObtuseAngle t

/-- Theorem: The negation of "There is at most one obtuse angle in a triangle"
    is equivalent to "There are at least two obtuse angles." -/
theorem negation_at_most_one_obtuse_angle :
  ¬AtMostOneObtuseAngle ↔ ∃ t : Triangle, ∃ a b : ℕ, a ≠ b ∧ a ≤ 3 ∧ b ≤ 3 ∧ HasObtuseAngle t ∧ HasObtuseAngle t :=
by sorry

end NUMINAMATH_CALUDE_negation_at_most_one_obtuse_angle_l493_49332


namespace NUMINAMATH_CALUDE_monotonic_increase_interval_l493_49382

/-- Given a function f with period π and its left-translated version g, 
    prove the interval of monotonic increase for g. -/
theorem monotonic_increase_interval
  (f : ℝ → ℝ)
  (ω : ℝ)
  (h_ω_pos : ω > 0)
  (h_f_def : ∀ x, f x = Real.sin (ω * x - π / 4))
  (h_f_period : ∀ x, f (x + π) = f x)
  (g : ℝ → ℝ)
  (h_g_def : ∀ x, g x = f (x + π / 4)) :
  ∀ k : ℤ, StrictMonoOn g (Set.Icc (-3 * π / 8 + k * π) (π / 8 + k * π)) :=
sorry

end NUMINAMATH_CALUDE_monotonic_increase_interval_l493_49382


namespace NUMINAMATH_CALUDE_min_distance_ellipse_line_is_zero_l493_49348

/-- The minimum distance between a point on the ellipse x²/8 + y²/4 = 1 
    and the line x - √2 y - 4 = 0 is 0. -/
theorem min_distance_ellipse_line_is_zero :
  let ellipse := {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 4 = 1}
  let line := {p : ℝ × ℝ | p.1 - Real.sqrt 2 * p.2 - 4 = 0}
  (∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ ellipse ∧ q ∈ line ∧ ‖p - q‖ = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_line_is_zero_l493_49348


namespace NUMINAMATH_CALUDE_salmon_trip_count_l493_49377

theorem salmon_trip_count (male_salmon female_salmon : ℕ) 
  (h1 : male_salmon = 712261)
  (h2 : female_salmon = 259378) : 
  male_salmon + female_salmon = 971639 := by
  sorry

end NUMINAMATH_CALUDE_salmon_trip_count_l493_49377


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l493_49328

theorem max_value_cos_sin (x : ℝ) : 2 * Real.cos x + 3 * Real.sin x ≤ Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l493_49328


namespace NUMINAMATH_CALUDE_arrangement_theorem_l493_49309

def number_of_arrangements (num_men : ℕ) (num_women : ℕ) : ℕ :=
  let group_of_four_two_men := Nat.choose num_men 2 * Nat.choose num_women 2
  let group_of_four_one_man := Nat.choose num_men 1 * Nat.choose num_women 3
  group_of_four_two_men + group_of_four_one_man

theorem arrangement_theorem :
  number_of_arrangements 5 4 = 80 :=
by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l493_49309


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l493_49374

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicularLines (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The problem statement -/
theorem perpendicular_line_through_point :
  let givenLine : Line2D := { a := 2, b := 1, c := -3 }
  let pointA : Point2D := { x := 0, y := 4 }
  let resultLine : Line2D := { a := 1, b := -2, c := 8 }
  perpendicularLines givenLine resultLine ∧
  pointOnLine pointA resultLine := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l493_49374


namespace NUMINAMATH_CALUDE_logistics_problem_l493_49394

/-- Represents the problem of transporting goods using two types of trucks -/
theorem logistics_problem (total_goods : ℕ) (type_a_capacity : ℕ) (type_b_capacity : ℕ) (num_type_a : ℕ) :
  total_goods = 300 →
  type_a_capacity = 20 →
  type_b_capacity = 15 →
  num_type_a = 7 →
  ∃ (num_type_b : ℕ),
    num_type_b ≥ 11 ∧
    num_type_a * type_a_capacity + num_type_b * type_b_capacity ≥ total_goods ∧
    ∀ (m : ℕ), m < num_type_b →
      num_type_a * type_a_capacity + m * type_b_capacity < total_goods :=
by
  sorry


end NUMINAMATH_CALUDE_logistics_problem_l493_49394


namespace NUMINAMATH_CALUDE_percentage_relation_l493_49384

theorem percentage_relation (x y : ℝ) (h : 0.15 * x = 0.2 * y) : y = 0.75 * x := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l493_49384


namespace NUMINAMATH_CALUDE_triangle_third_side_l493_49323

theorem triangle_third_side (a b h : ℝ) (ha : a = 25) (hb : b = 30) (hh : h = 24) :
  ∃ c, (c = 25 ∨ c = 11) ∧ 
  (∃ s, s * h = a * b ∧ 
   ((c + s) * (c - s) = a^2 - b^2 ∨ (c + s) * (c - s) = b^2 - a^2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_l493_49323


namespace NUMINAMATH_CALUDE_lewis_harvest_earnings_l493_49336

/-- Calculates the total earnings during a harvest season given regular weekly earnings, overtime weekly earnings, and the number of weeks. -/
def total_harvest_earnings (regular_weekly : ℕ) (overtime_weekly : ℕ) (weeks : ℕ) : ℕ :=
  (regular_weekly + overtime_weekly) * weeks

/-- Theorem stating that Lewis's total earnings during the harvest season equal $1,055,497 -/
theorem lewis_harvest_earnings :
  total_harvest_earnings 28 939 1091 = 1055497 := by
  sorry

#eval total_harvest_earnings 28 939 1091

end NUMINAMATH_CALUDE_lewis_harvest_earnings_l493_49336


namespace NUMINAMATH_CALUDE_tangent_line_and_extrema_l493_49301

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem tangent_line_and_extrema :
  let tangent_line (x : ℝ) := 1
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ f 0) ∧
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ f (Real.pi / 2)) ∧
  (HasDerivAt f (tangent_line 0 - f 0) 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_extrema_l493_49301


namespace NUMINAMATH_CALUDE_base4_product_l493_49318

-- Define a function to convert from base 4 to decimal
def base4ToDecimal (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (4 ^ i)) 0

-- Define a function to convert from decimal to base 4
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

-- Define the two base 4 numbers
def num1 : List Nat := [1, 3, 2]  -- 132₄
def num2 : List Nat := [1, 2]     -- 12₄

-- State the theorem
theorem base4_product :
  decimalToBase4 (base4ToDecimal num1 * base4ToDecimal num2) = [2, 3, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_base4_product_l493_49318


namespace NUMINAMATH_CALUDE_opposite_of_negative_l493_49389

theorem opposite_of_negative (a : ℝ) : -(- a) = a := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_l493_49389


namespace NUMINAMATH_CALUDE_initial_sum_calculation_l493_49371

/-- The initial sum that earns a specific total simple interest over 4 years with varying interest rates -/
def initial_sum (total_interest : ℚ) (rate1 rate2 rate3 rate4 : ℚ) : ℚ :=
  total_interest / (rate1 + rate2 + rate3 + rate4)

/-- Theorem stating that given the specified conditions, the initial sum is 5000/9 -/
theorem initial_sum_calculation :
  initial_sum 100 (3/100) (5/100) (4/100) (6/100) = 5000/9 := by
  sorry

#eval initial_sum 100 (3/100) (5/100) (4/100) (6/100)

end NUMINAMATH_CALUDE_initial_sum_calculation_l493_49371


namespace NUMINAMATH_CALUDE_intersection_implies_m_equals_one_l493_49398

def A (m : ℝ) : Set ℝ := {0, m}
def B : Set ℝ := {1, 2}

theorem intersection_implies_m_equals_one (m : ℝ) :
  A m ∩ B = {1} → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_equals_one_l493_49398


namespace NUMINAMATH_CALUDE_existence_of_special_integer_l493_49331

theorem existence_of_special_integer : ∃ n : ℕ+, 
  (Nat.card {p : ℕ | Nat.Prime p ∧ p ∣ n} = 2000) ∧ 
  (n ∣ 2^(n : ℕ) + 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_integer_l493_49331


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l493_49322

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line2 (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0
def line3 (x y : ℝ) : Prop := 2 * x + 3 * y + 5 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (a1 b1 c1 a2 b2 c2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

-- The theorem to prove
theorem perpendicular_line_equation :
  ∃ (x y : ℝ), intersection_point x y ∧
  perpendicular 2 3 5 2 3 (-7) ∧
  (2 * x + 3 * y - 7 = 0) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l493_49322


namespace NUMINAMATH_CALUDE_exists_valid_road_configuration_l493_49329

/-- A configuration of roads connecting four villages at the vertices of a square -/
structure RoadConfiguration where
  /-- The side length of the square -/
  side_length : ℝ
  /-- The total length of roads in the configuration -/
  total_length : ℝ
  /-- Ensure that all villages are connected -/
  all_connected : Bool

/-- Theorem stating that there exists a valid road configuration with total length less than 5.5 km -/
theorem exists_valid_road_configuration :
  ∃ (config : RoadConfiguration),
    config.side_length = 2 ∧
    config.all_connected = true ∧
    config.total_length < 5.5 := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_road_configuration_l493_49329


namespace NUMINAMATH_CALUDE_rationalize_denominator_l493_49356

theorem rationalize_denominator : 
  (50 + Real.sqrt 8) / (Real.sqrt 50 + Real.sqrt 8) = 
  (50 * (Real.sqrt 50 - Real.sqrt 8) + 12) / 42 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l493_49356


namespace NUMINAMATH_CALUDE_percentage_problem_l493_49353

theorem percentage_problem (x : ℝ) (P : ℝ) : 
  x = 150 → 
  P * x = 0.20 * 487.50 → 
  P = 0.65 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l493_49353


namespace NUMINAMATH_CALUDE_parallel_perpendicular_implication_l493_49373

/-- Two lines are parallel -/
def parallel (m n : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular (l : Line) (p : Plane) : Prop := sorry

theorem parallel_perpendicular_implication 
  (m n : Line) (β : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular m β) : 
  perpendicular n β := sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_implication_l493_49373


namespace NUMINAMATH_CALUDE_power_product_equality_l493_49392

theorem power_product_equality (a : ℝ) : a * a^2 * (-a)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l493_49392


namespace NUMINAMATH_CALUDE_hundredth_term_equals_981_l493_49321

/-- Sequence of powers of 3 or sums of distinct powers of 3 -/
def PowerOf3Sequence : ℕ → ℕ :=
  sorry

/-- The 100th term of the PowerOf3Sequence -/
def HundredthTerm : ℕ := PowerOf3Sequence 100

theorem hundredth_term_equals_981 : HundredthTerm = 981 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_term_equals_981_l493_49321


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l493_49304

theorem geometric_series_first_term (a₁ q : ℝ) : 
  (∀ n : ℕ, n ≥ 1 → ∃ (aₙ : ℝ), aₙ = a₁ * q^(n-1)) →  -- Geometric series condition
  (-1 < q) →                                         -- Convergence condition
  (q < 1) →                                          -- Convergence condition
  (q ≠ 0) →                                          -- Non-zero common ratio
  (a₁ / (1 - q) = 1) →                               -- Sum of series is 1
  (|a₁| / (1 - |q|) = 2) →                           -- Sum of absolute values is 2
  a₁ = 4/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l493_49304


namespace NUMINAMATH_CALUDE_ravi_mobile_price_l493_49303

/-- The purchase price of Ravi's mobile phone -/
def mobile_price : ℝ :=
  -- Define the variable for the mobile phone price
  sorry

/-- The selling price of the refrigerator -/
def fridge_sell_price : ℝ :=
  15000 * (1 - 0.04)

/-- The selling price of the mobile phone -/
def mobile_sell_price : ℝ :=
  mobile_price * 1.10

/-- The total selling price of both items -/
def total_sell_price : ℝ :=
  fridge_sell_price + mobile_sell_price

/-- The total purchase price of both items plus profit -/
def total_purchase_plus_profit : ℝ :=
  15000 + mobile_price + 200

theorem ravi_mobile_price :
  (total_sell_price = total_purchase_plus_profit) →
  mobile_price = 6000 :=
by sorry

end NUMINAMATH_CALUDE_ravi_mobile_price_l493_49303


namespace NUMINAMATH_CALUDE_largest_satisfying_n_l493_49385

/-- A rectangle with sides parallel to the coordinate axes -/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- Two rectangles are disjoint -/
def disjoint (r1 r2 : Rectangle) : Prop :=
  r1.x_max ≤ r2.x_min ∨ r2.x_max ≤ r1.x_min ∨
  r1.y_max ≤ r2.y_min ∨ r2.y_max ≤ r1.y_min

/-- Two rectangles have a common point -/
def have_common_point (r1 r2 : Rectangle) : Prop :=
  ¬(disjoint r1 r2)

/-- The property described in the problem -/
def satisfies_property (n : ℕ) : Prop :=
  ∃ (A B : Fin n → Rectangle),
    (∀ i : Fin n, disjoint (A i) (B i)) ∧
    (∀ i j : Fin n, i ≠ j → have_common_point (A i) (B j))

/-- The main theorem: The largest positive integer satisfying the property is 4 -/
theorem largest_satisfying_n :
  (∃ n : ℕ, n > 0 ∧ satisfies_property n) ∧
  (∀ n : ℕ, satisfies_property n → n ≤ 4) ∧
  satisfies_property 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_satisfying_n_l493_49385


namespace NUMINAMATH_CALUDE_remainder_2019_pow_2018_mod_100_l493_49390

theorem remainder_2019_pow_2018_mod_100 : 2019^2018 ≡ 41 [ZMOD 100] := by sorry

end NUMINAMATH_CALUDE_remainder_2019_pow_2018_mod_100_l493_49390


namespace NUMINAMATH_CALUDE_arcsin_sin_eq_x_div_3_l493_49335

theorem arcsin_sin_eq_x_div_3 (x : ℝ) :
  -3 * π / 2 ≤ x ∧ x ≤ 3 * π / 2 →
  (Real.arcsin (Real.sin x) = x / 3 ↔ 
    x = -3 * π / 2 ∨ x = 0 ∨ x = 3 * π / 4 ∨ x = 3 * π / 2) := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sin_eq_x_div_3_l493_49335


namespace NUMINAMATH_CALUDE_orange_profit_maximization_l493_49387

/-- Represents the cost and selling prices of oranges --/
structure OrangePrices where
  cost_a : ℝ
  sell_a : ℝ
  cost_b : ℝ
  sell_b : ℝ

/-- Represents a purchasing plan for oranges --/
structure PurchasePlan where
  kg_a : ℕ
  kg_b : ℕ

/-- Calculates the total cost of a purchase plan --/
def total_cost (prices : OrangePrices) (plan : PurchasePlan) : ℝ :=
  prices.cost_a * plan.kg_a + prices.cost_b * plan.kg_b

/-- Calculates the profit of a purchase plan --/
def profit (prices : OrangePrices) (plan : PurchasePlan) : ℝ :=
  (prices.sell_a - prices.cost_a) * plan.kg_a + (prices.sell_b - prices.cost_b) * plan.kg_b

/-- The main theorem to prove --/
theorem orange_profit_maximization (prices : OrangePrices) 
    (h1 : prices.sell_a = 16)
    (h2 : prices.sell_b = 24)
    (h3 : total_cost prices {kg_a := 15, kg_b := 20} = 430)
    (h4 : total_cost prices {kg_a := 10, kg_b := 8} = 212)
    (h5 : ∀ plan : PurchasePlan, plan.kg_a + plan.kg_b = 100 → 
      1160 ≤ total_cost prices plan ∧ total_cost prices plan ≤ 1168) :
  prices.cost_a = 10 ∧ 
  prices.cost_b = 14 ∧
  (∀ plan : PurchasePlan, plan.kg_a + plan.kg_b = 100 → 
    profit prices plan ≤ profit prices {kg_a := 58, kg_b := 42}) ∧
  profit prices {kg_a := 58, kg_b := 42} = 768 := by
  sorry


end NUMINAMATH_CALUDE_orange_profit_maximization_l493_49387


namespace NUMINAMATH_CALUDE_committees_with_restriction_l493_49372

def total_students : ℕ := 9
def committee_size : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem committees_with_restriction (total : ℕ) (size : ℕ) : 
  total = total_students → size = committee_size → 
  (choose total size) - (choose (total - 2) (size - 2)) = 91 := by
  sorry

end NUMINAMATH_CALUDE_committees_with_restriction_l493_49372


namespace NUMINAMATH_CALUDE_remainder_sum_l493_49325

theorem remainder_sum (c d : ℤ) 
  (hc : c % 80 = 72)
  (hd : d % 120 = 112) :
  (c + d) % 40 = 24 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l493_49325


namespace NUMINAMATH_CALUDE_wickets_before_last_match_value_l493_49380

/-- Represents the bowling statistics of a cricket player -/
structure BowlingStats where
  initial_average : ℝ
  initial_wickets : ℕ
  new_wickets : ℕ
  new_runs : ℕ
  average_decrease : ℝ

/-- Calculates the number of wickets taken before the last match -/
def wickets_before_last_match (stats : BowlingStats) : ℕ :=
  stats.initial_wickets

/-- Theorem stating the number of wickets taken before the last match -/
theorem wickets_before_last_match_value (stats : BowlingStats) 
  (h1 : stats.initial_average = 12.4)
  (h2 : stats.new_wickets = 3)
  (h3 : stats.new_runs = 26)
  (h4 : stats.average_decrease = 0.4)
  (h5 : stats.initial_wickets = wickets_before_last_match stats) :
  wickets_before_last_match stats = 25 := by
  sorry

#eval wickets_before_last_match { 
  initial_average := 12.4, 
  initial_wickets := 25, 
  new_wickets := 3, 
  new_runs := 26, 
  average_decrease := 0.4 
}

end NUMINAMATH_CALUDE_wickets_before_last_match_value_l493_49380


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l493_49305

/-- Given that x and y are inversely proportional, prove that when x = -12, y = -56.25 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 60) (h3 : x = 3 * y) :
  x = -12 → y = -56.25 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l493_49305


namespace NUMINAMATH_CALUDE_twin_prime_power_theorem_l493_49310

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_twin_prime (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ q = p + 2

theorem twin_prime_power_theorem :
  ∀ n : ℕ, (∃ p q : ℕ, is_twin_prime p q ∧ is_twin_prime (2^n + p) (2^n + q)) ↔ n = 1 ∨ n = 3 :=
sorry

end NUMINAMATH_CALUDE_twin_prime_power_theorem_l493_49310


namespace NUMINAMATH_CALUDE_mark_change_factor_l493_49344

theorem mark_change_factor (n : ℕ) (original_avg new_avg : ℝ) (h1 : n = 12) (h2 : original_avg = 36) (h3 : new_avg = 72) :
  ∃ (factor : ℝ), factor * (n * original_avg) = n * new_avg ∧ factor = 2 := by
  sorry

end NUMINAMATH_CALUDE_mark_change_factor_l493_49344


namespace NUMINAMATH_CALUDE_first_digit_is_one_l493_49306

def base_three_number : List Nat := [1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 2, 2, 1, 0, 1, 2, 2, 1, 0, 2]

def base_three_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3^i)) 0

def decimal_to_base_nine (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 9) ((m % 9) :: acc)
    aux n []

theorem first_digit_is_one :
  (decimal_to_base_nine (base_three_to_decimal base_three_number)).head? = some 1 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_is_one_l493_49306


namespace NUMINAMATH_CALUDE_min_value_absolute_difference_l493_49307

theorem min_value_absolute_difference (x : ℝ) :
  ((2 * x - 1) / 3 - 1 ≥ x - (5 - 3 * x) / 2) →
  (∃ y : ℝ, y = |x - 1| - |x + 3| ∧ 
   (∀ z : ℝ, ((2 * z - 1) / 3 - 1 ≥ z - (5 - 3 * z) / 2) → y ≤ |z - 1| - |z + 3|) ∧
   y = -2 - 8 / 11) :=
by sorry

end NUMINAMATH_CALUDE_min_value_absolute_difference_l493_49307


namespace NUMINAMATH_CALUDE_intersection_M_N_l493_49386

def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N : Set ℝ := {x | x > 1}

theorem intersection_M_N : M ∩ N = Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l493_49386


namespace NUMINAMATH_CALUDE_modified_lucas_60th_term_mod_5_l493_49320

def modifiedLucas : ℕ → ℤ
  | 0 => 2
  | 1 => 5
  | n + 2 => modifiedLucas n + modifiedLucas (n + 1)

theorem modified_lucas_60th_term_mod_5 :
  modifiedLucas 59 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modified_lucas_60th_term_mod_5_l493_49320


namespace NUMINAMATH_CALUDE_power_six_2045_mod_13_l493_49391

theorem power_six_2045_mod_13 : 6^2045 ≡ 2 [ZMOD 13] := by sorry

end NUMINAMATH_CALUDE_power_six_2045_mod_13_l493_49391


namespace NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l493_49314

theorem square_plus_one_geq_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l493_49314


namespace NUMINAMATH_CALUDE_expression_simplification_l493_49359

theorem expression_simplification (q : ℚ) : 
  ((7 * q - 2) + 2 * q * 3) * 4 + (5 + 2 / 2) * (4 * q - 6) = 76 * q - 44 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l493_49359


namespace NUMINAMATH_CALUDE_specific_arithmetic_sequence_sum_l493_49338

/-- Sum of an arithmetic sequence with given parameters -/
def arithmeticSequenceSum (a₁ aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem stating the sum of the specific arithmetic sequence -/
theorem specific_arithmetic_sequence_sum :
  arithmeticSequenceSum 1000 5000 4 = 3003000 := by
  sorry

#eval arithmeticSequenceSum 1000 5000 4

end NUMINAMATH_CALUDE_specific_arithmetic_sequence_sum_l493_49338


namespace NUMINAMATH_CALUDE_target_heart_rate_for_sprinting_is_156_l493_49350

-- Define the athlete's age
def age : ℕ := 30

-- Define the maximum heart rate calculation
def max_heart_rate (a : ℕ) : ℕ := 225 - a

-- Define the target heart rate for jogging
def target_heart_rate_jogging (mhr : ℕ) : ℕ := (mhr * 3) / 4

-- Define the target heart rate for sprinting
def target_heart_rate_sprinting (thr_jogging : ℕ) : ℕ := thr_jogging + 10

-- Theorem to prove
theorem target_heart_rate_for_sprinting_is_156 : 
  target_heart_rate_sprinting (target_heart_rate_jogging (max_heart_rate age)) = 156 := by
  sorry

end NUMINAMATH_CALUDE_target_heart_rate_for_sprinting_is_156_l493_49350


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_solution_l493_49334

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℚ × ℚ := (1/8, -3/4)

/-- First line equation: y = -6x -/
def line1 (x y : ℚ) : Prop := y = -6 * x

/-- Second line equation: y + 3 = 18x -/
def line2 (x y : ℚ) : Prop := y + 3 = 18 * x

/-- Theorem stating that the intersection_point satisfies both line equations
    and is the unique solution -/
theorem intersection_point_is_unique_solution :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ (x' y' : ℚ), line1 x' y' → line2 x' y' → (x', y') = intersection_point := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_solution_l493_49334


namespace NUMINAMATH_CALUDE_min_value_expression_l493_49341

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + 2*b^2 + 2/(a + 2*b)^2 ≥ 2 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀^2 + 2*b₀^2 + 2/(a₀ + 2*b₀)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l493_49341


namespace NUMINAMATH_CALUDE_largest_whole_number_less_than_150_over_9_l493_49366

theorem largest_whole_number_less_than_150_over_9 :
  ∀ x : ℕ, x ≤ 16 ↔ 9 * x < 150 :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_less_than_150_over_9_l493_49366


namespace NUMINAMATH_CALUDE_distribute_five_objects_l493_49358

/-- The number of ways to distribute n distinguishable objects into 2 indistinguishable containers -/
def distribute_objects (n : ℕ) : ℕ :=
  (2^n - 2) / 2 + 2

/-- Theorem: There are 17 ways to distribute 5 distinguishable objects into 2 indistinguishable containers -/
theorem distribute_five_objects : distribute_objects 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_objects_l493_49358


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l493_49300

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + (a - 1) * x + (a - 1) < 0) ↔ a < -1/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l493_49300


namespace NUMINAMATH_CALUDE_merry_go_round_area_l493_49376

theorem merry_go_round_area (diameter : Real) (h : diameter = 2) :
  let radius : Real := diameter / 2
  let area : Real := π * radius ^ 2
  area = π := by
  sorry

end NUMINAMATH_CALUDE_merry_go_round_area_l493_49376


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l493_49383

theorem max_value_of_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  ∃ (max : ℝ), max = Real.sqrt 15 ∧ x + 1/x ≤ max ∧ ∃ (y : ℝ), 13 = y^2 + 1/y^2 ∧ y + 1/y = max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l493_49383


namespace NUMINAMATH_CALUDE_blood_expiration_date_l493_49388

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- The number of days in a non-leap year -/
def days_per_year : ℕ := 365

/-- The number of days in January -/
def days_in_january : ℕ := 31

/-- The number of days in February (non-leap year) -/
def days_in_february : ℕ := 28

/-- Calculate the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The expiration time of blood in seconds -/
def blood_expiration_time : ℕ := factorial 11

theorem blood_expiration_date :
  let total_days : ℕ := blood_expiration_time / seconds_per_day
  let days_in_second_year : ℕ := total_days - days_per_year
  let days_after_january : ℕ := days_in_second_year - days_in_january
  days_after_january = days_in_february + 8 :=
by sorry

end NUMINAMATH_CALUDE_blood_expiration_date_l493_49388


namespace NUMINAMATH_CALUDE_jills_total_earnings_l493_49316

/-- Calculates Jill's earnings over three months given her work schedule --/
def jills_earnings (first_month_daily_rate : ℕ) (days_per_month : ℕ) : ℕ :=
  let first_month := first_month_daily_rate * days_per_month
  let second_month := (2 * first_month_daily_rate) * days_per_month
  let third_month := (2 * first_month_daily_rate) * (days_per_month / 2)
  first_month + second_month + third_month

/-- Theorem stating that Jill's earnings over three months equal $1200 --/
theorem jills_total_earnings :
  jills_earnings 10 30 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_jills_total_earnings_l493_49316


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l493_49313

/-- A line passing through (1, 2) with equal intercepts on both axes -/
structure EqualInterceptLine where
  -- The slope of the line
  k : ℝ
  -- The line passes through (1, 2)
  point_condition : 2 = k * (1 - 1) + 2
  -- The line has equal intercepts on both axes
  equal_intercepts : 2 - k = 1 - 2 / k

/-- The equation of the line is either x + y - 3 = 0 or 2x - y = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (∀ x y, x + y - 3 = 0 ↔ y - 2 = l.k * (x - 1)) ∨
  (∀ x y, 2 * x - y = 0 ↔ y - 2 = l.k * (x - 1)) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l493_49313


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_1_over_113_l493_49369

theorem first_nonzero_digit_of_1_over_113 : ∃ (n : ℕ) (r : ℚ), 
  (1 : ℚ) / 113 = n / 10 + r ∧ 
  0 < r ∧ 
  r < 1 / 10 ∧ 
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_1_over_113_l493_49369


namespace NUMINAMATH_CALUDE_increasing_quadratic_condition_l493_49340

/-- If f(x) = x^2 + 2(a - 1)x + 2 is an increasing function on the interval (4, +∞), then a ≥ -3 -/
theorem increasing_quadratic_condition (a : ℝ) : 
  (∀ x > 4, Monotone (fun x => x^2 + 2*(a - 1)*x + 2)) → a ≥ -3 := by
sorry

end NUMINAMATH_CALUDE_increasing_quadratic_condition_l493_49340


namespace NUMINAMATH_CALUDE_flowerbed_fence_length_l493_49315

/-- Calculates the perimeter of a rectangular flowerbed with given width and length rule -/
def flowerbed_perimeter (width : ℝ) : ℝ :=
  let length := 2 * width - 1
  2 * (width + length)

/-- Theorem stating that a rectangular flowerbed with width 4 meters and length 1 meter less than twice its width has a perimeter of 22 meters -/
theorem flowerbed_fence_length : flowerbed_perimeter 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_flowerbed_fence_length_l493_49315


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l493_49354

theorem two_digit_number_puzzle :
  ∀ (x : ℕ),
  x < 10 →
  let original := 21 * x
  let reversed := 12 * x
  original < 100 →
  original - reversed = 27 →
  original = 63 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l493_49354


namespace NUMINAMATH_CALUDE_probability_third_draw_defective_10_3_l493_49327

/-- Given a set of products with some defective ones, this function calculates
    the probability of drawing a defective product on the third draw, given
    that the first draw was defective. -/
def probability_third_draw_defective (total_products : ℕ) (defective_products : ℕ) : ℚ :=
  if total_products < 3 ∨ defective_products < 1 ∨ defective_products > total_products then 0
  else
    let remaining_after_first := total_products - 1
    let defective_after_first := defective_products - 1
    let numerator := (remaining_after_first - defective_after_first) * defective_after_first +
                     defective_after_first * (defective_after_first - 1)
    let denominator := remaining_after_first * (remaining_after_first - 1)
    ↑numerator / ↑denominator

/-- Theorem stating that for 10 products with 3 defective ones, the probability
    of drawing a defective product on the third draw, given that the first
    draw was defective, is 2/9. -/
theorem probability_third_draw_defective_10_3 :
  probability_third_draw_defective 10 3 = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_third_draw_defective_10_3_l493_49327


namespace NUMINAMATH_CALUDE_line_intersects_or_tangent_circle_l493_49368

/-- A line in 2D space defined by the equation (x+1)m + (y-1)n = 0 --/
structure Line where
  m : ℝ
  n : ℝ

/-- A circle in 2D space defined by the equation x^2 + y^2 = 2 --/
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 2}

/-- The point (-1, 1) --/
def M : ℝ × ℝ := (-1, 1)

/-- Theorem stating that the line either intersects or is tangent to the circle --/
theorem line_intersects_or_tangent_circle (l : Line) : 
  (∃ p : ℝ × ℝ, p ∈ Circle ∧ (p.1 + 1) * l.m + (p.2 - 1) * l.n = 0) := by
  sorry

#check line_intersects_or_tangent_circle

end NUMINAMATH_CALUDE_line_intersects_or_tangent_circle_l493_49368
