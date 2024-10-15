import Mathlib

namespace NUMINAMATH_CALUDE_harry_owns_three_geckos_l2578_257832

/-- Represents the number of geckos Harry owns -/
def num_geckos : ℕ := 3

/-- Represents the number of iguanas Harry owns -/
def num_iguanas : ℕ := 2

/-- Represents the number of snakes Harry owns -/
def num_snakes : ℕ := 4

/-- Represents the monthly feeding cost per snake in dollars -/
def snake_cost : ℕ := 10

/-- Represents the monthly feeding cost per iguana in dollars -/
def iguana_cost : ℕ := 5

/-- Represents the monthly feeding cost per gecko in dollars -/
def gecko_cost : ℕ := 15

/-- Represents the total annual feeding cost for all pets in dollars -/
def total_annual_cost : ℕ := 1140

/-- Theorem stating that the number of geckos Harry owns is 3 -/
theorem harry_owns_three_geckos :
  num_geckos = 3 ∧
  num_geckos * gecko_cost * 12 + num_iguanas * iguana_cost * 12 + num_snakes * snake_cost * 12 = total_annual_cost :=
by sorry

end NUMINAMATH_CALUDE_harry_owns_three_geckos_l2578_257832


namespace NUMINAMATH_CALUDE_factorization_sum_l2578_257889

theorem factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 9*x + 14 = (x + a)*(x + b)) →
  (∀ x : ℝ, x^2 + 4*x - 21 = (x + b)*(x - c)) →
  a + b + c = 12 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l2578_257889


namespace NUMINAMATH_CALUDE_sequence_sum_exp_l2578_257860

/-- Given a sequence {a_n} where the sum of the first n terms is S_n = ln(1 + 1/n),
    prove that e^(a_7 + a_8 + a_9) = 20/21 -/
theorem sequence_sum_exp (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n, S n = Real.log (1 + 1 / n)) :
  Real.exp (a 7 + a 8 + a 9) = 20 / 21 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_exp_l2578_257860


namespace NUMINAMATH_CALUDE_solutions_of_quadratic_equation_l2578_257854

theorem solutions_of_quadratic_equation :
  ∀ x : ℝ, x * (2 * x + 1) = 0 ↔ x = 0 ∨ x = -1/2 := by sorry

end NUMINAMATH_CALUDE_solutions_of_quadratic_equation_l2578_257854


namespace NUMINAMATH_CALUDE_quadratic_equation_k_value_l2578_257857

theorem quadratic_equation_k_value (x1 x2 k : ℝ) : 
  x1^2 - 6*x1 + k = 0 →
  x2^2 - 6*x2 + k = 0 →
  1/x1 + 1/x2 = 3 →
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_value_l2578_257857


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l2578_257830

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3}
def N : Set Nat := {1, 4}

theorem complement_intersection_equals_set :
  (U \ M) ∩ (U \ N) = {5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l2578_257830


namespace NUMINAMATH_CALUDE_cheolsu_weight_l2578_257814

/-- Proves that Cheolsu's weight is 36 kg given the conditions stated in the problem -/
theorem cheolsu_weight :
  ∀ (cheolsu_weight mother_weight : ℝ),
    cheolsu_weight = (2 / 3) * mother_weight →
    cheolsu_weight + 72 = 2 * mother_weight →
    cheolsu_weight = 36 := by
  sorry

end NUMINAMATH_CALUDE_cheolsu_weight_l2578_257814


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l2578_257801

theorem floor_ceil_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.1 : ℝ)⌉ = 31 := by sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l2578_257801


namespace NUMINAMATH_CALUDE_donut_distribution_ways_l2578_257892

/-- The number of types of donuts available --/
def num_types : ℕ := 5

/-- The total number of donuts to be purchased --/
def total_donuts : ℕ := 8

/-- The number of donuts that must be purchased of the first type --/
def first_type_min : ℕ := 2

/-- The number of donuts that must be purchased of each other type --/
def other_types_min : ℕ := 1

/-- The number of remaining donuts to be distributed after mandatory purchases --/
def remaining_donuts : ℕ := total_donuts - (first_type_min + (num_types - 1) * other_types_min)

theorem donut_distribution_ways : 
  (Nat.choose (remaining_donuts + num_types - 1) (num_types - 1)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_donut_distribution_ways_l2578_257892


namespace NUMINAMATH_CALUDE_connect_to_inaccessible_intersection_l2578_257890

-- Define the basic types
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define a line as a point and a direction vector
structure Line (V : Type*) [NormedAddCommGroup V] where
  point : V
  direction : V

-- Define the problem setup
variable (l₁ l₂ : Line V) (M : V)

-- State the theorem
theorem connect_to_inaccessible_intersection :
  ∃ (L : Line V), L.point = M ∧ 
    ∃ (t : ℝ), M + t • L.direction ∈ {x | ∃ (s₁ s₂ : ℝ), 
      x = l₁.point + s₁ • l₁.direction ∧ 
      x = l₂.point + s₂ • l₂.direction} :=
sorry

end NUMINAMATH_CALUDE_connect_to_inaccessible_intersection_l2578_257890


namespace NUMINAMATH_CALUDE_fifth_term_of_specific_sequence_l2578_257877

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℝ
  second : ℝ

/-- Get the nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first + (n - 1) * (seq.second - seq.first)

/-- The theorem to prove -/
theorem fifth_term_of_specific_sequence :
  let seq := ArithmeticSequence.mk 3 8
  nthTerm seq 5 = 23 := by sorry

end NUMINAMATH_CALUDE_fifth_term_of_specific_sequence_l2578_257877


namespace NUMINAMATH_CALUDE_jerrys_age_l2578_257834

theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 18 →
  mickey_age = 2 * jerry_age - 6 →
  jerry_age = 12 :=
by sorry

end NUMINAMATH_CALUDE_jerrys_age_l2578_257834


namespace NUMINAMATH_CALUDE_ten_people_round_table_with_pair_l2578_257810

/-- The number of ways to arrange n people around a round table -/
def roundTableArrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to arrange n people around a round table
    when two specific people must sit next to each other -/
def roundTableArrangementsWithPair (n : ℕ) : ℕ :=
  2 * roundTableArrangements (n - 1)

/-- Theorem: There are 80,640 ways to arrange 10 people around a round table
    when two specific people must sit next to each other -/
theorem ten_people_round_table_with_pair :
  roundTableArrangementsWithPair 10 = 80640 := by
  sorry

end NUMINAMATH_CALUDE_ten_people_round_table_with_pair_l2578_257810


namespace NUMINAMATH_CALUDE_right_triangle_tan_l2578_257827

theorem right_triangle_tan (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a = 40) (h3 : c = 41) :
  b / a = 9 / 40 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_tan_l2578_257827


namespace NUMINAMATH_CALUDE_trig_identity_special_case_l2578_257852

theorem trig_identity_special_case : 
  Real.cos (60 * π / 180 + 30 * π / 180) * Real.cos (60 * π / 180 - 30 * π / 180) + 
  Real.sin (60 * π / 180 + 30 * π / 180) * Real.sin (60 * π / 180 - 30 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_special_case_l2578_257852


namespace NUMINAMATH_CALUDE_evaluate_power_l2578_257886

theorem evaluate_power : (64 : ℝ) ^ (3/4) = 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_power_l2578_257886


namespace NUMINAMATH_CALUDE_girls_in_school_l2578_257816

theorem girls_in_school (total_students : ℕ) (sample_size : ℕ) (girls_sampled : ℕ) 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : girls_sampled = 95) : 
  (girls_sampled : ℚ) / (total_girls : ℚ) = (sample_size : ℚ) / (total_students : ℚ) → 
  total_girls = 760 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_school_l2578_257816


namespace NUMINAMATH_CALUDE_sum_of_absolute_roots_l2578_257831

theorem sum_of_absolute_roots (m : ℤ) (a b c : ℤ) : 
  (∀ x : ℝ, x^3 - 2011*x + m = (x - a) * (x - b) * (x - c)) →
  |a| + |b| + |c| = 98 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_roots_l2578_257831


namespace NUMINAMATH_CALUDE_initial_position_of_moving_point_l2578_257876

theorem initial_position_of_moving_point (M : ℝ) : 
  (M - 7) + 4 = 0 → M = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_position_of_moving_point_l2578_257876


namespace NUMINAMATH_CALUDE_units_digit_theorem_l2578_257800

theorem units_digit_theorem : ∃ n : ℕ, (33 * 219^89 + 89^19) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_theorem_l2578_257800


namespace NUMINAMATH_CALUDE_inverse_f_sum_l2578_257807

-- Define the function f
def f (x : ℝ) : ℝ := x^2 * abs x

-- State the theorem
theorem inverse_f_sum : (∃ y₁ y₂ : ℝ, f y₁ = 8 ∧ f y₂ = -27 ∧ y₁ + y₂ = -1) := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_sum_l2578_257807


namespace NUMINAMATH_CALUDE_delegates_with_female_count_l2578_257817

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose delegates with at least one female student. -/
def delegates_with_female (male_count female_count delegate_count : ℕ) : ℕ :=
  (choose female_count 1 * choose male_count (delegate_count - 1)) +
  (choose female_count 2 * choose male_count (delegate_count - 2)) +
  (choose female_count 3 * choose male_count (delegate_count - 3))

theorem delegates_with_female_count :
  delegates_with_female 4 3 3 = 31 := by sorry

end NUMINAMATH_CALUDE_delegates_with_female_count_l2578_257817


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l2578_257844

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (interest : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : interest = 4016.25)
  (h2 : rate = 12)
  (h3 : time = 5) :
  ∃ principal : ℝ,
    interest = principal * rate * time / 100 ∧
    principal = 6693.75 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l2578_257844


namespace NUMINAMATH_CALUDE_parallelogram_smallest_angle_l2578_257884

theorem parallelogram_smallest_angle (a b c d : ℝ) : 
  -- Conditions
  a + b + c + d = 360 →  -- Sum of angles in a quadrilateral is 360°
  a = c →                -- Opposite angles are equal
  b = d →                -- Opposite angles are equal
  max a b - min a b = 100 →  -- Largest angle is 100° greater than smallest
  -- Conclusion
  min a b = 40 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_smallest_angle_l2578_257884


namespace NUMINAMATH_CALUDE_greatest_lower_bound_sum_squares_roots_l2578_257875

/-- A monic polynomial of degree n with real coefficients -/
def MonicPoly (n : ℕ) := Polynomial ℝ

/-- The coefficient of x^(n-1) in a monic polynomial -/
def a_n_minus_1 (p : MonicPoly n) : ℝ := p.coeff (n - 1)

/-- The coefficient of x^(n-2) in a monic polynomial -/
def a_n_minus_2 (p : MonicPoly n) : ℝ := p.coeff (n - 2)

/-- The sum of squares of roots of a polynomial -/
noncomputable def sum_of_squares_of_roots (p : MonicPoly n) : ℝ := 
  (p.roots.map (λ r => r^2)).sum

/-- Theorem: The greatest lower bound on the sum of squares of roots -/
theorem greatest_lower_bound_sum_squares_roots (n : ℕ) (p : MonicPoly n) 
  (h : a_n_minus_1 p = 2 * a_n_minus_2 p) :
  ∃ (lb : ℝ), lb = (1/4 : ℝ) ∧ 
    ∀ (q : MonicPoly n), a_n_minus_1 q = 2 * a_n_minus_2 q → 
      lb ≤ sum_of_squares_of_roots q :=
by sorry

end NUMINAMATH_CALUDE_greatest_lower_bound_sum_squares_roots_l2578_257875


namespace NUMINAMATH_CALUDE_quadratic_max_abs_value_bound_l2578_257826

/-- For any quadratic function f(x) = x^2 + px + q, 
    the maximum absolute value of f(1), f(2), and f(3) 
    is greater than or equal to 1/2. -/
theorem quadratic_max_abs_value_bound (p q : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + p*x + q
  ∃ i : Fin 3, |f (i.val + 1)| ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_abs_value_bound_l2578_257826


namespace NUMINAMATH_CALUDE_intersection_of_complement_and_Q_l2578_257896

-- Define the sets P and Q
def P : Set ℝ := {x | x - 1 ≤ 0}
def Q : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Define the complement of P in ℝ
def C_R_P : Set ℝ := {x | ¬(x ∈ P)}

-- State the theorem
theorem intersection_of_complement_and_Q : 
  (C_R_P ∩ Q) = {x : ℝ | 1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_complement_and_Q_l2578_257896


namespace NUMINAMATH_CALUDE_sqrt_product_is_eight_l2578_257818

theorem sqrt_product_is_eight :
  Real.sqrt (9 - Real.sqrt 77) * Real.sqrt 2 * (Real.sqrt 11 - Real.sqrt 7) * (9 + Real.sqrt 77) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_is_eight_l2578_257818


namespace NUMINAMATH_CALUDE_sum_of_rational_roots_l2578_257872

/-- The polynomial p(x) = x^3 - 8x^2 + 17x - 10 -/
def p (x : ℚ) : ℚ := x^3 - 8*x^2 + 17*x - 10

/-- A number is a root of p if p(x) = 0 -/
def is_root (x : ℚ) : Prop := p x = 0

/-- The sum of the rational roots of p(x) is 8 -/
theorem sum_of_rational_roots :
  ∃ (S : Finset ℚ), (∀ x ∈ S, is_root x) ∧ (∀ x : ℚ, is_root x → x ∈ S) ∧ (S.sum id = 8) :=
sorry

end NUMINAMATH_CALUDE_sum_of_rational_roots_l2578_257872


namespace NUMINAMATH_CALUDE_license_plate_theorem_l2578_257843

def license_plate_combinations : ℕ :=
  let alphabet_size : ℕ := 26
  let plate_length : ℕ := 5
  let repeated_letters : ℕ := 2
  let non_zero_digits : ℕ := 9

  let choose_repeated_letters := Nat.choose alphabet_size repeated_letters
  let assign_first_repeat := Nat.choose plate_length repeated_letters
  let assign_second_repeat := Nat.choose (plate_length - repeated_letters) repeated_letters
  let remaining_letter_choices := alphabet_size - repeated_letters
  
  choose_repeated_letters * assign_first_repeat * assign_second_repeat * remaining_letter_choices * non_zero_digits

theorem license_plate_theorem : license_plate_combinations = 210600 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l2578_257843


namespace NUMINAMATH_CALUDE_arithmetic_sequence_lower_bound_l2578_257863

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The property a₁² + a₂ₙ₊₁² = 1 -/
def SequenceProperty (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a 1 ^ 2 + a (2 * n + 1) ^ 2 = 1

theorem arithmetic_sequence_lower_bound
  (a : ℕ → ℝ) (h1 : ArithmeticSequence a) (h2 : SequenceProperty a) :
  ∀ n : ℕ, a (n + 1) ^ 2 + a (3 * n + 1) ^ 2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_lower_bound_l2578_257863


namespace NUMINAMATH_CALUDE_largest_band_size_l2578_257835

/-- Represents a rectangular band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ

/-- Represents the band and its formations --/
structure Band where
  totalMembers : ℕ
  firstFormation : BandFormation
  secondFormation : BandFormation

/-- Checks if a band satisfies all given conditions --/
def satisfiesConditions (band : Band) : Prop :=
  band.totalMembers < 100 ∧
  band.totalMembers = band.firstFormation.rows * band.firstFormation.membersPerRow + 3 ∧
  band.totalMembers = band.secondFormation.rows * band.secondFormation.membersPerRow ∧
  band.secondFormation.rows = band.firstFormation.rows - 3 ∧
  band.secondFormation.membersPerRow = band.firstFormation.membersPerRow + 1

/-- The theorem stating that 75 is the largest possible number of band members --/
theorem largest_band_size :
  ∀ band : Band, satisfiesConditions band → band.totalMembers ≤ 75 :=
by sorry

end NUMINAMATH_CALUDE_largest_band_size_l2578_257835


namespace NUMINAMATH_CALUDE_list_number_fraction_l2578_257809

theorem list_number_fraction (list : List ℝ) (n : ℝ) :
  list.length = 21 →
  n ∈ list →
  list.Pairwise (·≠·) →
  n = 4 * ((list.sum - n) / 20) →
  n = (1 / 6) * list.sum :=
by sorry

end NUMINAMATH_CALUDE_list_number_fraction_l2578_257809


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2578_257878

-- Define the triangle DEF
structure Triangle (D E F : ℝ × ℝ) : Prop where
  right_angle : (E.1 - D.1) * (F.1 - D.1) + (E.2 - D.2) * (F.2 - D.2) = 0
  de_length : (E.1 - D.1)^2 + (E.2 - D.2)^2 = 15^2

-- Define the squares DEFG and EFHI
structure OuterSquares (D E F G H I : ℝ × ℝ) : Prop where
  square_defg : (G.1 - D.1) = (E.1 - D.1) ∧ (G.2 - D.2) = (E.2 - D.2)
  square_efhi : (I.1 - E.1) = (F.1 - E.1) ∧ (I.2 - E.2) = (F.2 - E.2)

-- Define the circle passing through G, H, I, F
structure CircleGHIF (G H I F : ℝ × ℝ) : Prop where
  on_circle : ∃ (center : ℝ × ℝ) (radius : ℝ),
    (G.1 - center.1)^2 + (G.2 - center.2)^2 = radius^2 ∧
    (H.1 - center.1)^2 + (H.2 - center.2)^2 = radius^2 ∧
    (I.1 - center.1)^2 + (I.2 - center.2)^2 = radius^2 ∧
    (F.1 - center.1)^2 + (F.2 - center.2)^2 = radius^2

-- Define the point J on DF
def PointJ (D F J : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, J = (D.1 + t * (F.1 - D.1), D.2 + t * (F.2 - D.2)) ∧ t ≠ 1

theorem triangle_perimeter 
  (D E F G H I J : ℝ × ℝ)
  (triangle : Triangle D E F)
  (squares : OuterSquares D E F G H I)
  (circle : CircleGHIF G H I F)
  (j_on_df : PointJ D F J)
  (jf_length : (J.1 - F.1)^2 + (J.2 - F.2)^2 = 3^2) :
  let de := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let ef := Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)
  let fd := Real.sqrt ((D.1 - F.1)^2 + (D.2 - F.2)^2)
  de + ef + fd = 15 + 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2578_257878


namespace NUMINAMATH_CALUDE_joan_balloons_l2578_257806

/-- Given that Joan initially has 9 blue balloons and loses 2 balloons,
    prove that she has 7 blue balloons remaining. -/
theorem joan_balloons : 
  let initial_balloons : ℕ := 9
  let lost_balloons : ℕ := 2
  initial_balloons - lost_balloons = 7 := by
sorry

end NUMINAMATH_CALUDE_joan_balloons_l2578_257806


namespace NUMINAMATH_CALUDE_power_product_equals_l2578_257815

theorem power_product_equals : 3^5 * 4^5 = 248832 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_l2578_257815


namespace NUMINAMATH_CALUDE_sum_of_fractions_geq_one_l2578_257869

theorem sum_of_fractions_geq_one (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_geq_one_l2578_257869


namespace NUMINAMATH_CALUDE_angle_complement_from_supplement_l2578_257842

theorem angle_complement_from_supplement (angle : ℝ) : 
  (180 - angle = 130) → (90 - angle = 40) := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_from_supplement_l2578_257842


namespace NUMINAMATH_CALUDE_perpendicular_vectors_tan_theta_l2578_257851

theorem perpendicular_vectors_tan_theta :
  ∀ θ : ℝ,
  let a : ℝ × ℝ := (Real.cos θ, Real.sin θ)
  let b : ℝ × ℝ := (Real.sqrt 3, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  Real.tan θ = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_tan_theta_l2578_257851


namespace NUMINAMATH_CALUDE_max_intersections_count_l2578_257839

/-- The number of points on the positive x-axis -/
def num_x_points : ℕ := 15

/-- The number of points on the positive y-axis -/
def num_y_points : ℕ := 10

/-- The total number of segments connecting points on x-axis to points on y-axis -/
def num_segments : ℕ := num_x_points * num_y_points

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections : ℕ := (num_x_points.choose 2) * (num_y_points.choose 2)

theorem max_intersections_count :
  max_intersections = 4725 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_count_l2578_257839


namespace NUMINAMATH_CALUDE_compute_expression_l2578_257871

theorem compute_expression : 5 + 7 * (2 - 9)^2 = 348 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2578_257871


namespace NUMINAMATH_CALUDE_silverware_reduction_l2578_257882

theorem silverware_reduction (initial_per_type : ℕ) (num_types : ℕ) (total_purchased : ℕ) :
  initial_per_type = 15 →
  num_types = 4 →
  total_purchased = 44 →
  (initial_per_type * num_types - total_purchased) / num_types = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_silverware_reduction_l2578_257882


namespace NUMINAMATH_CALUDE_factorization_x4_minus_81_l2578_257880

theorem factorization_x4_minus_81 : 
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_81_l2578_257880


namespace NUMINAMATH_CALUDE_basketball_shot_expectation_l2578_257833

theorem basketball_shot_expectation (a b : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (h_exp : 3 * a + 2 * b = 2) :
  (∀ x y : ℝ, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ 3 * x + 2 * y = 2 → 
    2 / a + 1 / (3 * b) ≤ 2 / x + 1 / (3 * y)) ∧
  2 / a + 1 / (3 * b) = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_basketball_shot_expectation_l2578_257833


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_eq_product_l2578_257824

theorem sqrt_sum_squares_eq_product (a b c : ℝ) : 
  (Real.sqrt (a^2 + b^2) = a * b) ∧ (a + b + c = 0) → (a = 0 ∧ b = 0 ∧ c = 0) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_eq_product_l2578_257824


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l2578_257862

theorem pizza_toppings_combinations : Nat.choose 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l2578_257862


namespace NUMINAMATH_CALUDE_jelly_bean_distribution_l2578_257838

theorem jelly_bean_distribution (total_jelly_beans : ℕ) (leftover_jelly_beans : ℕ) : 
  total_jelly_beans = 726 →
  leftover_jelly_beans = 4 →
  ∃ (girls : ℕ),
    let boys := girls + 3
    let students := girls + boys
    let distributed_jelly_beans := boys * boys + girls * (2 * girls + 1)
    distributed_jelly_beans = total_jelly_beans - leftover_jelly_beans →
    students = 31 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_distribution_l2578_257838


namespace NUMINAMATH_CALUDE_a_faster_than_b_l2578_257837

/-- Represents a person sawing wood -/
structure Sawyer where
  name : String
  sections : ℕ
  pieces : ℕ

/-- Calculates the number of cuts required for a single piece of wood -/
def cuts (s : Sawyer) : ℕ := s.sections - 1

/-- Calculates the total number of cuts made by a sawyer -/
def totalCuts (s : Sawyer) : ℕ := (s.pieces / s.sections) * cuts s

/-- Defines what it means for one sawyer to be faster than another -/
def isFasterThan (s1 s2 : Sawyer) : Prop := totalCuts s1 > totalCuts s2

theorem a_faster_than_b :
  let a : Sawyer := ⟨"A", 3, 24⟩
  let b : Sawyer := ⟨"B", 2, 28⟩
  isFasterThan a b := by sorry

end NUMINAMATH_CALUDE_a_faster_than_b_l2578_257837


namespace NUMINAMATH_CALUDE_wendy_picture_upload_l2578_257870

/-- The number of pictures Wendy uploaded to Facebook -/
def total_pictures : ℕ := 79

/-- The number of pictures in the first album -/
def first_album_pictures : ℕ := 44

/-- The number of additional albums -/
def additional_albums : ℕ := 5

/-- The number of pictures in each additional album -/
def pictures_per_additional_album : ℕ := 7

/-- Theorem stating that the total number of pictures is correct -/
theorem wendy_picture_upload :
  total_pictures = first_album_pictures + additional_albums * pictures_per_additional_album :=
by sorry

end NUMINAMATH_CALUDE_wendy_picture_upload_l2578_257870


namespace NUMINAMATH_CALUDE_mike_baseball_cards_l2578_257823

/-- 
Given that Mike initially has 87 baseball cards and Sam buys 13 of them,
prove that Mike will have 74 baseball cards remaining.
-/
theorem mike_baseball_cards (initial_cards : ℕ) (bought_cards : ℕ) (remaining_cards : ℕ) :
  initial_cards = 87 →
  bought_cards = 13 →
  remaining_cards = initial_cards - bought_cards →
  remaining_cards = 74 := by
sorry

end NUMINAMATH_CALUDE_mike_baseball_cards_l2578_257823


namespace NUMINAMATH_CALUDE_unique_prime_solution_l2578_257894

theorem unique_prime_solution :
  ∀ (p q r : ℕ),
    Prime p → Prime q → Prime r →
    p + q^2 = r^4 →
    p = 7 ∧ q = 3 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l2578_257894


namespace NUMINAMATH_CALUDE_leila_spending_l2578_257828

/-- The amount Leila spent at the supermarket -/
def supermarket_cost : ℝ := 100

/-- The cost of fixing Leila's automobile -/
def automobile_cost : ℝ := 350

/-- The total amount Leila spent -/
def total_cost : ℝ := supermarket_cost + automobile_cost

theorem leila_spending :
  (automobile_cost = 3 * supermarket_cost + 50) →
  total_cost = 450 := by
  sorry

end NUMINAMATH_CALUDE_leila_spending_l2578_257828


namespace NUMINAMATH_CALUDE_largest_non_sum_of_30multiple_and_composite_l2578_257822

/-- A function that checks if a number is composite -/
def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- The statement to be proved -/
theorem largest_non_sum_of_30multiple_and_composite :
  ∀ n : ℕ, n > 93 →
    ∃ (k : ℕ) (c : ℕ), k > 0 ∧ isComposite c ∧ n = 30 * k + c :=
by sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_30multiple_and_composite_l2578_257822


namespace NUMINAMATH_CALUDE_homework_time_decrease_l2578_257888

theorem homework_time_decrease (x : ℝ) : 
  (∀ t : ℝ, t > 0 → (t * (1 - x))^2 = t * (1 - x)^2) →
  100 * (1 - x)^2 = 70 :=
by sorry

end NUMINAMATH_CALUDE_homework_time_decrease_l2578_257888


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_l2578_257866

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_non_factor_product (a b : ℕ) : 
  a ≠ b →
  a > 0 →
  b > 0 →
  is_factor a 48 →
  is_factor b 48 →
  ¬ is_factor (a * b) 48 →
  (∀ x y : ℕ, x ≠ y → x > 0 → y > 0 → is_factor x 48 → is_factor y 48 → 
    ¬ is_factor (x * y) 48 → a * b ≤ x * y) →
  a * b = 32 := by sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_l2578_257866


namespace NUMINAMATH_CALUDE_rick_ironing_rate_l2578_257847

/-- The number of dress shirts Rick can iron in an hour -/
def shirts_per_hour : ℕ := sorry

/-- The number of dress pants Rick can iron in an hour -/
def pants_per_hour : ℕ := 3

/-- The number of hours Rick spent ironing dress shirts -/
def hours_ironing_shirts : ℕ := 3

/-- The number of hours Rick spent ironing dress pants -/
def hours_ironing_pants : ℕ := 5

/-- The total number of pieces of clothing Rick ironed -/
def total_pieces : ℕ := 27

theorem rick_ironing_rate : shirts_per_hour = 4 := by
  sorry

end NUMINAMATH_CALUDE_rick_ironing_rate_l2578_257847


namespace NUMINAMATH_CALUDE_cubic_extrema_l2578_257845

/-- A cubic function f(x) = ax³ + bx² where a > 0 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2

theorem cubic_extrema (a b : ℝ) (h₁ : a > 0) :
  (∀ x, f a b x ≤ f a b 0) ∧  -- maximum at x = 0
  (∀ x, f a b x ≥ f a b (1/3)) -- minimum at x = 1/3
  → a + 2*b = 0 := by sorry

end NUMINAMATH_CALUDE_cubic_extrema_l2578_257845


namespace NUMINAMATH_CALUDE_grapes_purchased_l2578_257867

theorem grapes_purchased (grape_price mango_price mango_weight total_paid : ℕ) 
  (h1 : grape_price = 80)
  (h2 : mango_price = 55)
  (h3 : mango_weight = 9)
  (h4 : total_paid = 1135)
  : ∃ (grape_weight : ℕ), grape_weight * grape_price + mango_weight * mango_price = total_paid ∧ grape_weight = 8 := by
  sorry

end NUMINAMATH_CALUDE_grapes_purchased_l2578_257867


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2578_257855

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

-- Define the point P
def P : ℝ × ℝ := (2, 4)

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := x + 2*y - 10 = 0

theorem tangent_line_to_circle :
  (∀ x y, C x y → ¬(tangent_line x y)) ∧
  tangent_line P.1 P.2 ∧
  ∃! p : ℝ × ℝ, C p.1 p.2 ∧ tangent_line p.1 p.2 ∧ p ≠ P :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2578_257855


namespace NUMINAMATH_CALUDE_dave_has_least_money_l2578_257819

-- Define the set of people
inductive Person : Type
  | Alice : Person
  | Ben : Person
  | Carol : Person
  | Dave : Person
  | Ethan : Person

-- Define a function to represent the amount of money each person has
variable (money : Person → ℕ)

-- Define the conditions
axiom different_amounts : ∀ (p q : Person), p ≠ q → money p ≠ money q
axiom ethan_less_than_alice : money Person.Ethan < money Person.Alice
axiom ben_more_than_dave : money Person.Dave < money Person.Ben
axiom carol_more_than_dave : money Person.Dave < money Person.Carol
axiom alice_between_dave_and_ben : money Person.Dave < money Person.Alice ∧ money Person.Alice < money Person.Ben
axiom carol_between_ethan_and_alice : money Person.Ethan < money Person.Carol ∧ money Person.Carol < money Person.Alice

-- Theorem to prove
theorem dave_has_least_money :
  ∀ (p : Person), p ≠ Person.Dave → money Person.Dave < money p :=
sorry

end NUMINAMATH_CALUDE_dave_has_least_money_l2578_257819


namespace NUMINAMATH_CALUDE_expected_faces_six_die_six_rolls_l2578_257874

/-- The number of sides on the die -/
def n : ℕ := 6

/-- The number of rolls -/
def k : ℕ := 6

/-- The probability of a specific face not appearing in a single roll -/
def p : ℚ := (n - 1) / n

/-- The expected number of different faces appearing when rolling an n-sided die k times -/
def expected_faces : ℚ := n * (1 - p^k)

/-- Theorem: The expected number of different faces appearing when a fair six-sided die 
    is rolled six times is equal to (6^6 - 5^6) / 6^5 -/
theorem expected_faces_six_die_six_rolls : 
  expected_faces = (n^k - (n-1)^k) / n^(k-1) := by
  sorry

#eval expected_faces

end NUMINAMATH_CALUDE_expected_faces_six_die_six_rolls_l2578_257874


namespace NUMINAMATH_CALUDE_even_periodic_function_derivative_zero_l2578_257881

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem even_periodic_function_derivative_zero
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_diff : Differentiable ℝ f)
  (h_period : has_period f 5) :
  deriv f 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_periodic_function_derivative_zero_l2578_257881


namespace NUMINAMATH_CALUDE_additive_multiplicative_inverses_l2578_257803

theorem additive_multiplicative_inverses 
  (x y p q : ℝ) 
  (h1 : x + y = 0)  -- x and y are additive inverses
  (h2 : p * q = 1)  -- p and q are multiplicative inverses
  : (x + y) - 2 * p * q = -2 := by
sorry

end NUMINAMATH_CALUDE_additive_multiplicative_inverses_l2578_257803


namespace NUMINAMATH_CALUDE_largest_solution_floor_equation_l2578_257864

theorem largest_solution_floor_equation :
  let floor_eq (x : ℝ) := ⌊x⌋ = 7 + 50 * (x - ⌊x⌋)
  ∃ (max_sol : ℝ), floor_eq max_sol ∧
    ∀ (y : ℝ), floor_eq y → y ≤ max_sol ∧
    max_sol = 2849 / 50
  := by sorry

end NUMINAMATH_CALUDE_largest_solution_floor_equation_l2578_257864


namespace NUMINAMATH_CALUDE_expression_equals_503_l2578_257820

theorem expression_equals_503 : 2015 * (1999/2015) * (1/4) - 2011/2015 = 503 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_503_l2578_257820


namespace NUMINAMATH_CALUDE_rectangle_length_equals_eight_l2578_257856

theorem rectangle_length_equals_eight
  (square_perimeter : ℝ)
  (rectangle_width : ℝ)
  (triangle_height : ℝ)
  (h1 : square_perimeter = 64)
  (h2 : rectangle_width = 8)
  (h3 : triangle_height = 64)
  (h4 : (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_height * rectangle_length) :
  rectangle_length = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_equals_eight_l2578_257856


namespace NUMINAMATH_CALUDE_smallest_w_l2578_257841

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : 
  w > 0 →
  is_factor (2^5) (936 * w) →
  is_factor (3^3) (936 * w) →
  is_factor (12^2) (936 * w) →
  936 = 2^3 * 3^1 * 13^1 →
  (∀ v : ℕ, v > 0 → 
    is_factor (2^5) (936 * v) → 
    is_factor (3^3) (936 * v) → 
    is_factor (12^2) (936 * v) → 
    w ≤ v) →
  w = 36 := by
sorry

end NUMINAMATH_CALUDE_smallest_w_l2578_257841


namespace NUMINAMATH_CALUDE_sarah_meal_combinations_l2578_257848

/-- Represents the number of options for each meal component -/
structure MealOptions where
  appetizers : Nat
  mainCourses : Nat
  drinks : Nat
  desserts : Nat

/-- Represents the constraint on drink options when fries are chosen -/
def drinkOptionsWithFries (options : MealOptions) : Nat :=
  options.drinks - 1

/-- Calculates the number of meal combinations -/
def calculateMealCombinations (options : MealOptions) : Nat :=
  let mealsWithFries := 1 * options.mainCourses * (drinkOptionsWithFries options) * options.desserts
  let mealsWithoutFries := (options.appetizers - 1) * options.mainCourses * options.drinks * options.desserts
  mealsWithFries + mealsWithoutFries

/-- The main theorem stating the number of distinct meals Sarah can buy -/
theorem sarah_meal_combinations (options : MealOptions) 
  (h1 : options.appetizers = 3)
  (h2 : options.mainCourses = 3)
  (h3 : options.drinks = 3)
  (h4 : options.desserts = 2) : 
  calculateMealCombinations options = 48 := by
  sorry

#eval calculateMealCombinations { appetizers := 3, mainCourses := 3, drinks := 3, desserts := 2 }

end NUMINAMATH_CALUDE_sarah_meal_combinations_l2578_257848


namespace NUMINAMATH_CALUDE_road_cost_calculation_l2578_257873

theorem road_cost_calculation (lawn_length lawn_width road_length_width road_width_width : ℕ)
  (cost_length cost_width : ℚ) : 
  lawn_length = 80 ∧ 
  lawn_width = 60 ∧ 
  road_length_width = 12 ∧ 
  road_width_width = 15 ∧ 
  cost_length = 3 ∧ 
  cost_width = (5/2) →
  (lawn_length * road_length_width * cost_length + 
   lawn_width * road_width_width * cost_width : ℚ) = 5130 :=
by sorry

end NUMINAMATH_CALUDE_road_cost_calculation_l2578_257873


namespace NUMINAMATH_CALUDE_geometric_series_constant_l2578_257883

/-- A geometric series with sum of first n terms given by S_n = 3^(n+1) + a -/
def GeometricSeries (a : ℝ) : ℕ → ℝ := fun n ↦ 3^(n+1) + a

/-- The sum of the first n terms of the geometric series -/
def SeriesSum (a : ℝ) : ℕ → ℝ := fun n ↦ GeometricSeries a n

theorem geometric_series_constant (a : ℝ) : a = -3 :=
  sorry

end NUMINAMATH_CALUDE_geometric_series_constant_l2578_257883


namespace NUMINAMATH_CALUDE_milk_water_ratio_problem_l2578_257879

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- The problem statement -/
theorem milk_water_ratio_problem 
  (initial : CanContents)
  (h_capacity : initial.milk + initial.water + 20 = 60)
  (h_ratio_after : (initial.milk + 20) / initial.water = 3) :
  initial.milk / initial.water = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_milk_water_ratio_problem_l2578_257879


namespace NUMINAMATH_CALUDE_three_queens_or_at_least_one_jack_probability_l2578_257898

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Jacks in a standard deck -/
def num_jacks : ℕ := 4

/-- The number of Queens in a standard deck -/
def num_queens : ℕ := 4

/-- The number of cards drawn -/
def cards_drawn : ℕ := 3

/-- The probability of drawing either three queens or at least one jack -/
def probability : ℚ := 142 / 1105

theorem three_queens_or_at_least_one_jack_probability :
  let total_combinations := (deck_size.choose cards_drawn : ℚ)
  let three_queens_prob := (num_queens.choose cards_drawn : ℚ) / total_combinations
  let at_least_one_jack_prob := 1 - ((deck_size - num_jacks).choose cards_drawn : ℚ) / total_combinations
  three_queens_prob + at_least_one_jack_prob - (three_queens_prob * at_least_one_jack_prob) = probability :=
by sorry

end NUMINAMATH_CALUDE_three_queens_or_at_least_one_jack_probability_l2578_257898


namespace NUMINAMATH_CALUDE_length_PS_specific_quadrilateral_l2578_257858

/-- A quadrilateral with two right angles and specified side lengths -/
structure RightQuadrilateral where
  PQ : ℝ
  QR : ℝ
  RS : ℝ
  angle_Q_is_right : Bool
  angle_R_is_right : Bool

/-- The length of PS in a right quadrilateral PQRS -/
def length_PS (quad : RightQuadrilateral) : ℝ :=
  sorry

/-- Theorem: In a right quadrilateral PQRS where PQ = 7, QR = 10, RS = 25, 
    and angles Q and R are right angles, the length of PS is 2√106 -/
theorem length_PS_specific_quadrilateral :
  let quad : RightQuadrilateral := {
    PQ := 7,
    QR := 10,
    RS := 25,
    angle_Q_is_right := true,
    angle_R_is_right := true
  }
  length_PS quad = 2 * Real.sqrt 106 := by
  sorry

end NUMINAMATH_CALUDE_length_PS_specific_quadrilateral_l2578_257858


namespace NUMINAMATH_CALUDE_logarithm_simplification_l2578_257811

theorem logarithm_simplification
  (p q r s t u : ℝ)
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0) :
  Real.log (p / q) - Real.log (q / r) + Real.log (r / s) + Real.log ((s * t) / (p * u)) = Real.log (t / u) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l2578_257811


namespace NUMINAMATH_CALUDE_sphere_volume_circumscribing_rectangular_solid_l2578_257805

/-- The volume of a sphere that circumscribes a rectangular solid with dimensions 3, 2, and 1 -/
theorem sphere_volume_circumscribing_rectangular_solid :
  let length : ℝ := 3
  let width : ℝ := 2
  let height : ℝ := 1
  let radius : ℝ := Real.sqrt (length^2 + width^2 + height^2) / 2
  let volume : ℝ := (4 / 3) * Real.pi * radius^3
  volume = (7 * Real.sqrt 14 * Real.pi) / 3 := by
  sorry

#check sphere_volume_circumscribing_rectangular_solid

end NUMINAMATH_CALUDE_sphere_volume_circumscribing_rectangular_solid_l2578_257805


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l2578_257887

theorem sqrt_product_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (15 * p^3) * Real.sqrt (8 * p) * Real.sqrt (12 * p^5) = 60 * p^4 * Real.sqrt (2 * p) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l2578_257887


namespace NUMINAMATH_CALUDE_min_crossing_time_for_four_people_l2578_257849

/-- Represents a person with their crossing time -/
structure Person where
  crossingTime : ℕ

/-- Represents the state of the bridge crossing problem -/
structure BridgeState where
  leftSide : List Person
  rightSide : List Person

/-- Calculates the minimum time required for all people to cross the bridge -/
def minCrossingTime (people : List Person) : ℕ :=
  sorry

/-- Theorem stating the minimum crossing time for the given problem -/
theorem min_crossing_time_for_four_people :
  let people := [
    { crossingTime := 2 },
    { crossingTime := 4 },
    { crossingTime := 6 },
    { crossingTime := 8 }
  ]
  minCrossingTime people = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_crossing_time_for_four_people_l2578_257849


namespace NUMINAMATH_CALUDE_distributive_property_implies_fraction_additivity_l2578_257829

theorem distributive_property_implies_fraction_additivity 
  {a b c : ℝ} (h1 : c ≠ 0) (h2 : (a + b) * c = a * c + b * c) :
  (a + b) / c = a / c + b / c :=
sorry

end NUMINAMATH_CALUDE_distributive_property_implies_fraction_additivity_l2578_257829


namespace NUMINAMATH_CALUDE_batsman_average_increase_l2578_257859

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  totalRuns : ℕ
  innings : ℕ
  average : ℚ

/-- Calculates the increase in average after a new inning -/
def averageIncrease (initialStats : BatsmanStats) (newInningRuns : ℕ) : ℚ :=
  let newStats : BatsmanStats := {
    totalRuns := initialStats.totalRuns + newInningRuns,
    innings := initialStats.innings + 1,
    average := (initialStats.totalRuns + newInningRuns : ℚ) / (initialStats.innings + 1)
  }
  newStats.average - initialStats.average

/-- Theorem: The batsman's average increased by 3 runs per inning -/
theorem batsman_average_increase :
  ∀ (initialStats : BatsmanStats),
    initialStats.innings = 16 →
    averageIncrease initialStats 88 = 40 →
    averageIncrease initialStats 88 = 3 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l2578_257859


namespace NUMINAMATH_CALUDE_article_cost_l2578_257808

/-- Proves that the cost of an article is 120, given the selling prices and gain difference --/
theorem article_cost (sp1 sp2 : ℕ) (gain_diff : ℚ) :
  sp1 = 380 →
  sp2 = 420 →
  gain_diff = 8 / 100 →
  sp2 - (sp1 - (sp2 - sp1)) = 120 := by
sorry

end NUMINAMATH_CALUDE_article_cost_l2578_257808


namespace NUMINAMATH_CALUDE_smallest_prime_perfect_square_plus_20_l2578_257897

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to check if a number is a perfect square plus 20
def isPerfectSquarePlus20 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 + 20

-- Theorem statement
theorem smallest_prime_perfect_square_plus_20 :
  isPrime 29 ∧ isPerfectSquarePlus20 29 ∧
  ∀ m : ℕ, m < 29 → ¬(isPrime m ∧ isPerfectSquarePlus20 m) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perfect_square_plus_20_l2578_257897


namespace NUMINAMATH_CALUDE_least_sum_of_four_primes_l2578_257802

def is_sum_of_four_primes (n : ℕ) : Prop :=
  ∃ p₁ p₂ p₃ p₄ : ℕ, 
    p₁.Prime ∧ p₂.Prime ∧ p₃.Prime ∧ p₄.Prime ∧
    p₁ > 10 ∧ p₂ > 10 ∧ p₃ > 10 ∧ p₄ > 10 ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n = p₁ + p₂ + p₃ + p₄

theorem least_sum_of_four_primes : 
  (is_sum_of_four_primes 60) ∧ (∀ m < 60, ¬(is_sum_of_four_primes m)) :=
by sorry

end NUMINAMATH_CALUDE_least_sum_of_four_primes_l2578_257802


namespace NUMINAMATH_CALUDE_volume_of_rotated_composite_region_l2578_257865

/-- The volume of a solid formed by rotating a composite region about the y-axis -/
theorem volume_of_rotated_composite_region :
  let square_side : ℝ := 4
  let rectangle_width : ℝ := 5
  let rectangle_height : ℝ := 3
  let volume_square : ℝ := π * (square_side / 2)^2 * square_side
  let volume_rectangle : ℝ := π * (rectangle_height / 2)^2 * rectangle_width
  let total_volume : ℝ := volume_square + volume_rectangle
  total_volume = (109 * π) / 4 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_rotated_composite_region_l2578_257865


namespace NUMINAMATH_CALUDE_two_digit_number_property_l2578_257813

theorem two_digit_number_property (x y : ℕ) : 
  x < 10 → y < 10 → x ≠ 0 →
  x^2 + y^2 = 10*x + x*y →
  10*x + y - 36 = 10*y + x →
  10*x + y = 48 ∨ 10*x + y = 37 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l2578_257813


namespace NUMINAMATH_CALUDE_amount_problem_l2578_257821

theorem amount_problem (a b : ℝ) 
  (h1 : a + b = 1210)
  (h2 : (4/5) * a = (2/3) * b) :
  b = 453.75 := by
sorry

end NUMINAMATH_CALUDE_amount_problem_l2578_257821


namespace NUMINAMATH_CALUDE_shortest_side_range_l2578_257840

/-- An obtuse triangle with sides x, x+1, and x+2 -/
structure ObtuseTriangle where
  x : ℝ
  is_obtuse : 0 < x ∧ x < x + 1 ∧ x + 1 < x + 2

/-- The range of the shortest side in an obtuse triangle -/
theorem shortest_side_range (t : ObtuseTriangle) : 1 < t.x ∧ t.x < 3 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_range_l2578_257840


namespace NUMINAMATH_CALUDE_jungkook_balls_left_l2578_257891

/-- The number of balls left in a box after removing some balls -/
def balls_left (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

/-- Theorem: When Jungkook removes 3 balls from a box containing 10 balls, 7 balls are left -/
theorem jungkook_balls_left : balls_left 10 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_balls_left_l2578_257891


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2578_257895

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + 3*b = 1) :
  1/a + 3/b ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + 3*b₀ = 1 ∧ 1/a₀ + 3/b₀ = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2578_257895


namespace NUMINAMATH_CALUDE_percentage_of_150_to_60_prove_percentage_l2578_257846

theorem percentage_of_150_to_60 : Real → Prop :=
  fun x => (150 / 60) * 100 = x

theorem prove_percentage :
  ∃ x, percentage_of_150_to_60 x ∧ x = 250 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_of_150_to_60_prove_percentage_l2578_257846


namespace NUMINAMATH_CALUDE_salt_mixing_theorem_l2578_257885

def salt_mixing_problem (x : ℚ) : Prop :=
  let known_salt_weight : ℚ := 40
  let known_salt_price : ℚ := 25 / 100
  let unknown_salt_weight : ℚ := 60
  let total_weight : ℚ := known_salt_weight + unknown_salt_weight
  let selling_price : ℚ := 48 / 100
  let profit_percentage : ℚ := 20 / 100
  let total_cost : ℚ := known_salt_weight * known_salt_price + unknown_salt_weight * x
  let selling_revenue : ℚ := total_weight * selling_price
  selling_revenue = total_cost * (1 + profit_percentage) ∧ x = 50 / 100

theorem salt_mixing_theorem : ∃ x : ℚ, salt_mixing_problem x :=
  sorry

end NUMINAMATH_CALUDE_salt_mixing_theorem_l2578_257885


namespace NUMINAMATH_CALUDE_shorter_can_radius_l2578_257825

/-- Given two cylindrical cans with equal volume, where one can's height is twice 
    the other's and the taller can's radius is 10 units, the radius of the shorter 
    can is 10√2 units. -/
theorem shorter_can_radius (h : ℝ) (r : ℝ) : 
  h > 0 → -- height is positive
  π * (10^2) * (2*h) = π * r^2 * h → -- volumes are equal
  r = 10 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_shorter_can_radius_l2578_257825


namespace NUMINAMATH_CALUDE_base_seven_digits_of_1234_digits_in_base_seven_1234_l2578_257804

theorem base_seven_digits_of_1234 : ∃ n : ℕ, n > 0 ∧ 7^(n-1) ≤ 1234 ∧ 1234 < 7^n :=
by
  -- The proof would go here
  sorry

theorem digits_in_base_seven_1234 : ∃ n : ℕ, n > 0 ∧ 7^(n-1) ≤ 1234 ∧ 1234 < 7^n ∧ n = 4 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_base_seven_digits_of_1234_digits_in_base_seven_1234_l2578_257804


namespace NUMINAMATH_CALUDE_highest_probability_greater_than_2_l2578_257861

-- Define a fair dice
def FairDice := Finset (Fin 6)

-- Define the probability of an event on a fair dice
def probability (event : Finset (Fin 6)) (dice : FairDice) : ℚ :=
  (event.card : ℚ) / (dice.card : ℚ)

-- Define the events
def less_than_2 (dice : FairDice) : Finset (Fin 6) :=
  dice.filter (λ x => x < 2)

def greater_than_2 (dice : FairDice) : Finset (Fin 6) :=
  dice.filter (λ x => x > 2)

def even_number (dice : FairDice) : Finset (Fin 6) :=
  dice.filter (λ x => x % 2 = 0)

-- Theorem statement
theorem highest_probability_greater_than_2 (dice : FairDice) :
  probability (greater_than_2 dice) dice > probability (even_number dice) dice ∧
  probability (greater_than_2 dice) dice > probability (less_than_2 dice) dice :=
sorry

end NUMINAMATH_CALUDE_highest_probability_greater_than_2_l2578_257861


namespace NUMINAMATH_CALUDE_athlete_B_most_stable_l2578_257812

-- Define the athletes
inductive Athlete : Type
  | A : Athlete
  | B : Athlete
  | C : Athlete

-- Define the variance for each athlete
def variance (a : Athlete) : ℝ :=
  match a with
  | Athlete.A => 0.78
  | Athlete.B => 0.2
  | Athlete.C => 1.28

-- Define the concept of most stable performance
def most_stable (a : Athlete) : Prop :=
  ∀ b : Athlete, variance a ≤ variance b

-- Theorem statement
theorem athlete_B_most_stable :
  most_stable Athlete.B :=
sorry

end NUMINAMATH_CALUDE_athlete_B_most_stable_l2578_257812


namespace NUMINAMATH_CALUDE_function_always_positive_l2578_257850

/-- The function f(x) = (2-a^2)x + a is always positive in the interval [0,1] if and only if 0 < a < 2 -/
theorem function_always_positive (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, (2 - a^2) * x + a > 0) ↔ (0 < a ∧ a < 2) := by
  sorry

end NUMINAMATH_CALUDE_function_always_positive_l2578_257850


namespace NUMINAMATH_CALUDE_alley_width_l2578_257853

theorem alley_width (l k h w : Real) : 
  l > 0 → 
  k > 0 → 
  h > 0 → 
  w > 0 → 
  k = l * Real.sin (π / 3) → 
  h = l * Real.sin (π / 6) → 
  w = k / Real.sqrt 3 → 
  w = h * Real.sqrt 3 → 
  w = l / 2 := by sorry

end NUMINAMATH_CALUDE_alley_width_l2578_257853


namespace NUMINAMATH_CALUDE_train_length_approx_l2578_257836

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length_approx (speed : ℝ) (time : ℝ) : 
  speed = 100 → time = 3.6 → ∃ (length : ℝ), 
  (abs (length - (speed * 1000 / 3600 * time)) < 0.5) ∧ 
  (round length = 100) := by
  sorry

#check train_length_approx

end NUMINAMATH_CALUDE_train_length_approx_l2578_257836


namespace NUMINAMATH_CALUDE_mirror_wall_height_l2578_257893

def hall_of_mirrors (wall1_width wall2_width wall3_width total_area : ℝ) : Prop :=
  ∃ (height : ℝ),
    wall1_width * height + wall2_width * height + wall3_width * height = total_area

theorem mirror_wall_height :
  hall_of_mirrors 30 30 20 960 →
  ∃ (height : ℝ), height = 12 := by
sorry

end NUMINAMATH_CALUDE_mirror_wall_height_l2578_257893


namespace NUMINAMATH_CALUDE_sector_circumference_l2578_257899

/-- Given a circular sector with area 2 and central angle 4 radians, 
    its circumference is 6. -/
theorem sector_circumference (area : ℝ) (angle : ℝ) (circumference : ℝ) : 
  area = 2 → angle = 4 → circumference = 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_circumference_l2578_257899


namespace NUMINAMATH_CALUDE_remainder_eight_pow_215_mod_9_l2578_257868

theorem remainder_eight_pow_215_mod_9 : 8^215 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_eight_pow_215_mod_9_l2578_257868
