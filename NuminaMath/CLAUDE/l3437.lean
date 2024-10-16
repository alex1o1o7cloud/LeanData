import Mathlib

namespace NUMINAMATH_CALUDE_system_solutions_l3437_343714

theorem system_solutions :
  let f (x y : ℝ) := y^2 = x^3 - 3*x^2 + 2*x
  let g (x y : ℝ) := x^2 = y^3 - 3*y^2 + 2*y
  ∀ x y : ℝ, (f x y ∧ g x y) ↔ 
    ((x = 0 ∧ y = 0) ∨ 
     (x = 2 - Real.sqrt 2 ∧ y = 2 - Real.sqrt 2) ∨ 
     (x = 2 + Real.sqrt 2 ∧ y = 2 + Real.sqrt 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_system_solutions_l3437_343714


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3437_343706

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {1, 2}

-- Define set B
def B : Finset Nat := {2, 3, 4}

-- Theorem statement
theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3437_343706


namespace NUMINAMATH_CALUDE_cubic_equations_common_root_implies_three_real_roots_l3437_343743

/-- Given distinct nonzero real numbers a, b, c, if the equations ax³ + bx + c = 0, 
    bx³ + cx + a = 0, and cx³ + ax + b = 0 have a common root, then at least one of 
    these equations has three real roots. -/
theorem cubic_equations_common_root_implies_three_real_roots 
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (h_common_root : ∃ x : ℝ, a * x^3 + b * x + c = 0 ∧ 
                            b * x^3 + c * x + a = 0 ∧ 
                            c * x^3 + a * x + b = 0) :
  (∃ x y z : ℝ, a * x^3 + b * x + c = 0 ∧ 
               a * y^3 + b * y + c = 0 ∧ 
               a * z^3 + b * z + c = 0) ∨
  (∃ x y z : ℝ, b * x^3 + c * x + a = 0 ∧ 
               b * y^3 + c * y + a = 0 ∧ 
               b * z^3 + c * z + a = 0) ∨
  (∃ x y z : ℝ, c * x^3 + a * x + b = 0 ∧ 
               c * y^3 + a * y + b = 0 ∧ 
               c * z^3 + a * z + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equations_common_root_implies_three_real_roots_l3437_343743


namespace NUMINAMATH_CALUDE_four_tangent_circles_l3437_343722

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are tangent if the distance between their centers equals the sum of their radii --/
def are_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

/-- A circle is tangent to two other circles --/
def is_tangent_to_both (c c1 c2 : Circle) : Prop :=
  are_tangent c c1 ∧ are_tangent c c2

theorem four_tangent_circles (c1 c2 : Circle)
  (h1 : c1.radius = 2)
  (h2 : c2.radius = 2)
  (h3 : are_tangent c1 c2) :
  ∃! (s : Finset Circle), s.card = 4 ∧ ∀ c ∈ s, c.radius = 3 ∧ is_tangent_to_both c c1 c2 :=
sorry

end NUMINAMATH_CALUDE_four_tangent_circles_l3437_343722


namespace NUMINAMATH_CALUDE_bucket_weight_bucket_weight_proof_l3437_343799

/-- Given a bucket with the following properties:
  1. When three-quarters full of water, it weighs c kilograms (including the water).
  2. When one-third full of water, it weighs d kilograms (including the water).
  This theorem states that when the bucket is completely full of water, 
  its total weight is (8/5)c - (7/5)d kilograms. -/
theorem bucket_weight (c d : ℝ) : ℝ :=
  let three_quarters_full := c
  let one_third_full := d
  let full_weight := (8/5) * c - (7/5) * d
  full_weight

/-- Proof of the bucket_weight theorem -/
theorem bucket_weight_proof (c d : ℝ) : 
  bucket_weight c d = (8/5) * c - (7/5) * d :=
by sorry

end NUMINAMATH_CALUDE_bucket_weight_bucket_weight_proof_l3437_343799


namespace NUMINAMATH_CALUDE_cab_cost_for_event_l3437_343779

/-- Calculates the total cost of cab rides for a one-week event -/
def total_cab_cost (event_duration : ℕ) (distance : ℝ) (fare_per_mile : ℝ) (rides_per_day : ℕ) : ℝ :=
  event_duration * distance * fare_per_mile * rides_per_day

/-- Proves that the total cost of cab rides for the given conditions is $7000 -/
theorem cab_cost_for_event : 
  total_cab_cost 7 200 2.5 2 = 7000 := by sorry

end NUMINAMATH_CALUDE_cab_cost_for_event_l3437_343779


namespace NUMINAMATH_CALUDE_rotation_equivalence_l3437_343719

/-- Given that a point A is rotated 450 degrees clockwise and y degrees counterclockwise
    about the same center point B, both rotations resulting in the same final position C,
    and y < 360, prove that y = 270. -/
theorem rotation_equivalence (y : ℝ) : 
  (450 % 360 : ℝ) = (360 - y) % 360 → y < 360 → y = 270 := by sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l3437_343719


namespace NUMINAMATH_CALUDE_point_on_unit_circle_l3437_343716

theorem point_on_unit_circle (Q : ℝ × ℝ) : 
  (Q.1^2 + Q.2^2 = 1) →  -- Q is on the unit circle
  (Q.1 = -1/2 ∧ Q.2 = -Real.sqrt 3/2) ↔ 
  (∃ θ : ℝ, θ = -2*Real.pi/3 ∧ Q.1 = Real.cos θ ∧ Q.2 = Real.sin θ) :=
by sorry

end NUMINAMATH_CALUDE_point_on_unit_circle_l3437_343716


namespace NUMINAMATH_CALUDE_parabola_reflection_y_axis_l3437_343726

/-- Represents a parabola in the form y = a(x - h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Reflects a parabola along the y-axis --/
def reflect_y (p : Parabola) : Parabola :=
  { a := p.a, h := -p.h, k := p.k }

theorem parabola_reflection_y_axis :
  let original := Parabola.mk 2 1 (-4)
  let reflected := reflect_y original
  reflected = Parabola.mk 2 (-1) (-4) := by sorry

end NUMINAMATH_CALUDE_parabola_reflection_y_axis_l3437_343726


namespace NUMINAMATH_CALUDE_adams_book_purchase_l3437_343735

/-- Represents a bookcase with a given number of shelves and average books per shelf. -/
structure Bookcase where
  shelves : ℕ
  avgBooksPerShelf : ℕ

/-- Calculates the total capacity of a bookcase. -/
def Bookcase.capacity (b : Bookcase) : ℕ := b.shelves * b.avgBooksPerShelf

theorem adams_book_purchase (
  adam_bookcase : Bookcase
  ) (adam_bookcase_shelves : adam_bookcase.shelves = 4)
    (adam_bookcase_avg : adam_bookcase.avgBooksPerShelf = 20)
    (initial_books : ℕ) (initial_books_count : initial_books = 56)
    (books_left_over : ℕ) (books_left_over_count : books_left_over = 2) :
  adam_bookcase.capacity + books_left_over - initial_books = 26 := by
  sorry

end NUMINAMATH_CALUDE_adams_book_purchase_l3437_343735


namespace NUMINAMATH_CALUDE_no_nonnegative_solutions_l3437_343732

theorem no_nonnegative_solutions : ¬∃ x : ℝ, x ≥ 0 ∧ x^2 + 6*x + 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nonnegative_solutions_l3437_343732


namespace NUMINAMATH_CALUDE_factor_implies_a_value_l3437_343754

theorem factor_implies_a_value (a b : ℤ) :
  (∀ x : ℝ, (x^2 - x - 1 = 0) → (a*x^19 + b*x^18 + 1 = 0)) →
  a = 1597 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_a_value_l3437_343754


namespace NUMINAMATH_CALUDE_solve_scooter_price_l3437_343709

def scooter_price_problem (upfront_percentage : ℚ) (upfront_amount : ℚ) (num_installments : ℕ) : Prop :=
  let total_price : ℚ := upfront_amount / upfront_percentage * 100
  let remaining_amount : ℚ := total_price * (1 - upfront_percentage)
  let installment_amount : ℚ := remaining_amount / num_installments
  (upfront_percentage = 20/100) ∧ 
  (upfront_amount = 240) ∧ 
  (num_installments = 12) ∧
  (total_price = 1200) ∧ 
  (installment_amount = 80)

theorem solve_scooter_price : 
  ∃ (upfront_percentage : ℚ) (upfront_amount : ℚ) (num_installments : ℕ),
    scooter_price_problem upfront_percentage upfront_amount num_installments :=
by
  sorry

end NUMINAMATH_CALUDE_solve_scooter_price_l3437_343709


namespace NUMINAMATH_CALUDE_cube_sum_geq_triple_product_l3437_343701

theorem cube_sum_geq_triple_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 ≥ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_geq_triple_product_l3437_343701


namespace NUMINAMATH_CALUDE_solution_to_exponential_equation_l3437_343792

theorem solution_to_exponential_equation :
  ∃ y : ℝ, (3 : ℝ)^(y - 2) = (9 : ℝ)^(y + 2) ∧ y = -6 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_exponential_equation_l3437_343792


namespace NUMINAMATH_CALUDE_female_officers_count_l3437_343741

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_percent : ℚ) (female_percent_of_total : ℚ) :
  total_on_duty = 150 →
  female_on_duty_percent = 25 / 100 →
  female_percent_of_total = 40 / 100 →
  ∃ (total_female : ℕ), total_female = 240 ∧ 
    (female_on_duty_percent * total_female : ℚ) = (female_percent_of_total * total_on_duty : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l3437_343741


namespace NUMINAMATH_CALUDE_equation_solutions_l3437_343747

theorem equation_solutions (m : ℕ+) :
  let f := fun (x y : ℕ) => x^2 + y^2 + 2*x*y - m*x - m*y - m - 1
  (∃! s : Finset (ℕ × ℕ), s.card = m ∧ 
    ∀ (p : ℕ × ℕ), p ∈ s ↔ (f p.1 p.2 = 0 ∧ p.1 > 0 ∧ p.2 > 0)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3437_343747


namespace NUMINAMATH_CALUDE_trig_fraction_equality_l3437_343721

theorem trig_fraction_equality (x : ℝ) (h : (1 + Real.sin x) / Real.cos x = -1/2) :
  Real.cos x / (Real.sin x - 1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_equality_l3437_343721


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3437_343744

theorem greatest_divisor_with_remainders (a b r1 r2 : ℕ) (h1 : a = 60) (h2 : b = 190) (h3 : r1 = 6) (h4 : r2 = 10) :
  Nat.gcd (a - r1) (b - r2) = 18 :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3437_343744


namespace NUMINAMATH_CALUDE_jack_total_travel_time_l3437_343718

/-- Represents the time spent in a country during travel -/
structure CountryTime where
  customsHours : ℕ
  quarantineDays : ℕ

/-- Calculates the total hours spent in a country -/
def totalHoursInCountry (ct : CountryTime) : ℕ :=
  ct.customsHours + 24 * ct.quarantineDays

/-- The time Jack spent in each country -/
def jackTravelTime : List CountryTime := [
  { customsHours := 20, quarantineDays := 14 },  -- Canada
  { customsHours := 15, quarantineDays := 10 },  -- Australia
  { customsHours := 10, quarantineDays := 7 }    -- Japan
]

/-- Theorem stating the total time Jack spent in customs and quarantine -/
theorem jack_total_travel_time :
  List.foldl (λ acc ct => acc + totalHoursInCountry ct) 0 jackTravelTime = 789 :=
by sorry

end NUMINAMATH_CALUDE_jack_total_travel_time_l3437_343718


namespace NUMINAMATH_CALUDE_yanna_apples_kept_l3437_343781

def apples_kept (initial : ℕ) (given_to_zenny : ℕ) (given_to_andrea : ℕ) : ℕ :=
  initial - given_to_zenny - given_to_andrea

theorem yanna_apples_kept :
  apples_kept 60 18 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_yanna_apples_kept_l3437_343781


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3437_343755

theorem sum_of_numbers (a b : ℝ) (h1 : a - b = 5) (h2 : max a b = 25) : a + b = 45 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3437_343755


namespace NUMINAMATH_CALUDE_root_between_alpha_beta_l3437_343761

theorem root_between_alpha_beta (p q α β : ℝ) 
  (h_alpha : α^2 + p*α + q = 0)
  (h_beta : -β^2 + p*β + q = 0) :
  ∃ γ : ℝ, (min α β < γ ∧ γ < max α β) ∧ (1/2 * γ^2 + p*γ + q = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_between_alpha_beta_l3437_343761


namespace NUMINAMATH_CALUDE_blank_expression_proof_l3437_343764

theorem blank_expression_proof (x y : ℝ) : 2 * x * (-3 * x^2 * y) = -6 * x^3 * y := by
  sorry

end NUMINAMATH_CALUDE_blank_expression_proof_l3437_343764


namespace NUMINAMATH_CALUDE_product_94_106_l3437_343795

theorem product_94_106 : 94 * 106 = 9964 := by
  sorry

end NUMINAMATH_CALUDE_product_94_106_l3437_343795


namespace NUMINAMATH_CALUDE_y_divisibility_l3437_343757

def y : ℕ := 80 + 120 + 160 + 240 + 360 + 400 + 3600

theorem y_divisibility :
  (∃ k : ℕ, y = 5 * k) ∧
  (∃ k : ℕ, y = 10 * k) ∧
  (∃ k : ℕ, y = 20 * k) ∧
  ¬(∃ k : ℕ, y = 40 * k) :=
by sorry

end NUMINAMATH_CALUDE_y_divisibility_l3437_343757


namespace NUMINAMATH_CALUDE_eight_beads_two_identical_arrangements_l3437_343748

/-- The number of unique arrangements of n distinct beads, including k identical beads, on a bracelet, considering rotational and reflectional symmetry -/
def uniqueBraceletArrangements (n : ℕ) (k : ℕ) : ℕ :=
  (n.factorial / k.factorial) / (2 * n)

theorem eight_beads_two_identical_arrangements :
  uniqueBraceletArrangements 8 2 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_eight_beads_two_identical_arrangements_l3437_343748


namespace NUMINAMATH_CALUDE_polynomial_with_negative_integer_roots_l3437_343725

/-- A polynomial of degree 4 with integer coefficients -/
structure Polynomial4 where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- The polynomial function corresponding to a Polynomial4 -/
def poly_func (p : Polynomial4) : ℝ → ℝ :=
  fun x ↦ x^4 + p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- Predicate stating that all roots of a polynomial are negative integers -/
def all_roots_negative_integers (p : Polynomial4) : Prop :=
  ∀ x : ℝ, poly_func p x = 0 → (∃ n : ℤ, x = ↑n ∧ n < 0)

theorem polynomial_with_negative_integer_roots
  (p : Polynomial4)
  (h_roots : all_roots_negative_integers p)
  (h_sum : p.a + p.b + p.c + p.d = 2009) :
  p.d = 528 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_with_negative_integer_roots_l3437_343725


namespace NUMINAMATH_CALUDE_solution_set_inequalities_l3437_343715

theorem solution_set_inequalities :
  {x : ℝ | x - 2 > 1 ∧ x < 4} = {x : ℝ | 3 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_solution_set_inequalities_l3437_343715


namespace NUMINAMATH_CALUDE_order_independent_divisibility_criterion_only_for_3_and_9_l3437_343700

/-- A divisibility criterion for a positive integer that depends only on its digits. -/
def DigitDivisibilityCriterion (n : ℕ+) : Type :=
  (digits : List ℕ) → Bool

/-- The property that a divisibility criterion is independent of digit order. -/
def OrderIndependent (n : ℕ+) (criterion : DigitDivisibilityCriterion n) : Prop :=
  ∀ (digits₁ digits₂ : List ℕ), Multiset.ofList digits₁ = Multiset.ofList digits₂ →
    criterion digits₁ = criterion digits₂

/-- Theorem stating that order-independent digit divisibility criteria exist only for 3 and 9. -/
theorem order_independent_divisibility_criterion_only_for_3_and_9 (n : ℕ+) :
    (∃ (criterion : DigitDivisibilityCriterion n), OrderIndependent n criterion) →
    n = 3 ∨ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_order_independent_divisibility_criterion_only_for_3_and_9_l3437_343700


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l3437_343751

/-- Represents a hyperbola with foci on the y-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  focal_distance : ℝ
  focus_to_asymptote : ℝ
  h_positive : a > 0
  b_positive : b > 0
  h_focal_distance : focal_distance = 2 * Real.sqrt 3
  h_focus_to_asymptote : focus_to_asymptote = Real.sqrt 2
  h_c : c = Real.sqrt 3
  h_relation : c^2 = a^2 + b^2
  h_asymptote : b * c / Real.sqrt (a^2 + b^2) = focus_to_asymptote

/-- The standard equation of the hyperbola is y² - x²/2 = 1 -/
theorem hyperbola_standard_equation (h : Hyperbola) :
  h.a = 1 ∧ h.b = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l3437_343751


namespace NUMINAMATH_CALUDE_factorial_ratio_simplification_l3437_343736

theorem factorial_ratio_simplification :
  (11 * Nat.factorial 10 * Nat.factorial 7 * Nat.factorial 3) / 
  (Nat.factorial 10 * Nat.factorial 8) = 11 / 56 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_simplification_l3437_343736


namespace NUMINAMATH_CALUDE_max_value_when_a_zero_range_of_a_for_local_max_l3437_343717

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / Real.exp x + a * (x - 1)^2

-- Theorem for part 1
theorem max_value_when_a_zero :
  ∃ (x : ℝ), ∀ (y : ℝ), f 0 y ≤ f 0 x ∧ f 0 x = 1 / Real.exp 1 :=
sorry

-- Theorem for part 2
theorem range_of_a_for_local_max :
  ∀ (a : ℝ), (∃ (x : ℝ), ∀ (y : ℝ), f a y ≤ f a x ∧ f a x ≤ 1/2) ↔
  (a < 1 / (2 * Real.exp 1) ∨ (a > 1 / (2 * Real.exp 1) ∧ a ≤ 1/2)) :=
sorry

end NUMINAMATH_CALUDE_max_value_when_a_zero_range_of_a_for_local_max_l3437_343717


namespace NUMINAMATH_CALUDE_mindmaster_codes_l3437_343771

/-- The number of colors available for the Mindmaster game -/
def num_colors : ℕ := 5

/-- The number of slots in the Mindmaster game -/
def num_slots : ℕ := 5

/-- The total number of possible secret codes in the Mindmaster game -/
def total_codes : ℕ := num_colors ^ num_slots

/-- Theorem stating that the total number of possible secret codes is 3125 -/
theorem mindmaster_codes : total_codes = 3125 := by
  sorry

end NUMINAMATH_CALUDE_mindmaster_codes_l3437_343771


namespace NUMINAMATH_CALUDE_sugar_consumption_reduction_l3437_343797

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (initial_price_positive : initial_price > 0)
  (new_price_positive : new_price > 0)
  (price_increase : new_price > initial_price) :
  let reduction_percentage := (1 - initial_price / new_price) * 100
  reduction_percentage = 60 :=
by sorry

end NUMINAMATH_CALUDE_sugar_consumption_reduction_l3437_343797


namespace NUMINAMATH_CALUDE_max_y_coordinate_polar_curve_l3437_343786

theorem max_y_coordinate_polar_curve (θ : Real) :
  let r := λ θ : Real => Real.cos (2 * θ)
  let x := λ θ : Real => (r θ) * Real.cos θ
  let y := λ θ : Real => (r θ) * Real.sin θ
  (∀ θ', |y θ'| ≤ |y θ|) → y θ = Real.sqrt (30 * Real.sqrt 6) / 9 :=
by sorry

end NUMINAMATH_CALUDE_max_y_coordinate_polar_curve_l3437_343786


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l3437_343720

theorem fifteenth_student_age 
  (total_students : Nat) 
  (average_age : ℝ) 
  (group1_size : Nat) 
  (group1_average : ℝ) 
  (group2_size : Nat) 
  (group2_average : ℝ) 
  (h1 : total_students = 15)
  (h2 : average_age = 15)
  (h3 : group1_size = 5)
  (h4 : group1_average = 14)
  (h5 : group2_size = 9)
  (h6 : group2_average = 16)
  (h7 : group1_size + group2_size + 1 = total_students) :
  ∃ (fifteenth_age : ℝ),
    fifteenth_age = total_students * average_age - (group1_size * group1_average + group2_size * group2_average) ∧
    fifteenth_age = 11 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l3437_343720


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3437_343758

theorem quadratic_inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | a * x^2 + (2 - a) * x - 2 < 0}
  (a = 0 → S = {x : ℝ | x < 1}) ∧
  (-2 < a ∧ a < 0 → S = {x : ℝ | x < 1 ∨ x > -2/a}) ∧
  (a = -2 → S = {x : ℝ | x ≠ 1}) ∧
  (a < -2 → S = {x : ℝ | x < -2/a ∨ x > 1}) ∧
  (a > 0 → S = {x : ℝ | -2/a < x ∧ x < 1}) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3437_343758


namespace NUMINAMATH_CALUDE_find_number_l3437_343745

theorem find_number : ∃ x : ℝ, 0.3 * ((x / 2.5) - 10.5) = 5.85 ∧ x = 75 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3437_343745


namespace NUMINAMATH_CALUDE_point_side_line_range_l3437_343774

/-- Given that the points (3,-1) and (-4,-3) are on the same side of the line 3x-2y+a=0,
    prove that the range of values for a is (-∞,-11) ∪ (6,+∞). -/
theorem point_side_line_range (a : ℝ) : 
  (3 * 3 - 2 * (-1) + a) * (3 * (-4) - 2 * (-3) + a) > 0 ↔ 
  a ∈ Set.Iio (-11) ∪ Set.Ioi 6 :=
sorry

end NUMINAMATH_CALUDE_point_side_line_range_l3437_343774


namespace NUMINAMATH_CALUDE_smallest_sector_angle_l3437_343756

def circle_sectors (n : ℕ) (a : ℕ → ℕ) : Prop :=
  (∀ i, i ∈ Finset.range n → a i > 0) ∧
  (∀ i j k, i < j ∧ j < k → a j - a i = a k - a j) ∧
  (Finset.sum (Finset.range n) a = 360)

theorem smallest_sector_angle :
  ∀ a : ℕ → ℕ, circle_sectors 16 a → ∃ i, a i = 15 ∧ ∀ j, a j ≥ a i := by
  sorry

end NUMINAMATH_CALUDE_smallest_sector_angle_l3437_343756


namespace NUMINAMATH_CALUDE_clarissa_manuscript_cost_l3437_343740

/-- Calculate the total cost for printing, binding, and processing multiple copies of a manuscript with specified requirements. -/
def manuscript_cost (total_pages : ℕ) (color_pages : ℕ) (bw_cost : ℚ) (color_cost : ℚ) 
                    (binding_cost : ℚ) (index_cost : ℚ) (copies : ℕ) (rush_copies : ℕ) 
                    (rush_cost : ℚ) : ℚ :=
  let bw_pages := total_pages - color_pages
  let print_cost := (bw_pages : ℚ) * bw_cost + (color_pages : ℚ) * color_cost
  let additional_cost := binding_cost + index_cost
  let total_per_copy := print_cost + additional_cost
  let total_before_rush := (copies : ℚ) * total_per_copy
  let rush_fee := (rush_copies : ℚ) * rush_cost
  total_before_rush + rush_fee

/-- The total cost for Clarissa's manuscript printing job is $310.00. -/
theorem clarissa_manuscript_cost :
  manuscript_cost 400 50 (5/100) (10/100) 5 2 10 5 3 = 310 := by
  sorry

end NUMINAMATH_CALUDE_clarissa_manuscript_cost_l3437_343740


namespace NUMINAMATH_CALUDE_smallest_n_with_three_pairs_l3437_343737

/-- The function f(n) returns the number of distinct ordered pairs of positive integers (a, b) 
    such that a^2 + b^2 = n -/
def f (n : ℕ) : ℕ := (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 50 is the smallest positive integer n for which f(n) = 3 -/
theorem smallest_n_with_three_pairs : ∀ k : ℕ, 0 < k → k < 50 → f k ≠ 3 ∧ f 50 = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_with_three_pairs_l3437_343737


namespace NUMINAMATH_CALUDE_power_function_k_values_l3437_343749

/-- A function f(x) = ax^n is a power function if a ≠ 0 and n is a non-zero constant. -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ n ≠ 0 ∧ ∀ x, f x = a * x^n

/-- The main theorem: if y = (k^2-k-5)x^2 is a power function, then k = 3 or k = -2 -/
theorem power_function_k_values (k : ℝ) :
  is_power_function (λ x => (k^2 - k - 5) * x^2) → k = 3 ∨ k = -2 := by
  sorry


end NUMINAMATH_CALUDE_power_function_k_values_l3437_343749


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l3437_343727

theorem smallest_number_divisibility (h : ℕ) : 
  (∀ k < 259, ¬(∃ n : ℕ, k + 5 = 8 * n ∧ k + 5 = 11 * n ∧ k + 5 = 3 * n)) ∧
  (∃ n : ℕ, 259 + 5 = 8 * n) ∧
  (∃ n : ℕ, 259 + 5 = 11 * n) ∧
  (∃ n : ℕ, 259 + 5 = 3 * n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l3437_343727


namespace NUMINAMATH_CALUDE_baker_pastry_cake_difference_l3437_343763

/-- The number of cakes made by the baker -/
def cakes_made : ℕ := 19

/-- The number of pastries made by the baker -/
def pastries_made : ℕ := 131

/-- The difference between pastries and cakes made by the baker -/
def pastry_cake_difference : ℕ := pastries_made - cakes_made

theorem baker_pastry_cake_difference :
  pastry_cake_difference = 112 := by sorry

end NUMINAMATH_CALUDE_baker_pastry_cake_difference_l3437_343763


namespace NUMINAMATH_CALUDE_sum_and_ratio_implies_difference_l3437_343742

theorem sum_and_ratio_implies_difference (x y : ℝ) : 
  x + y = 540 → x / y = 0.75 → y - x = 77.143 := by
sorry

end NUMINAMATH_CALUDE_sum_and_ratio_implies_difference_l3437_343742


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_l3437_343738

theorem triangle_angle_ratio (right_angle top_angle left_angle : ℝ) : 
  right_angle = 60 →
  top_angle = 70 →
  left_angle + right_angle + top_angle = 180 →
  left_angle / right_angle = 5 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_l3437_343738


namespace NUMINAMATH_CALUDE_coaches_in_conference_l3437_343762

theorem coaches_in_conference (rowers : ℕ) (votes_per_rower : ℕ) (votes_per_coach : ℕ) 
  (h1 : rowers = 60)
  (h2 : votes_per_rower = 3)
  (h3 : votes_per_coach = 5) :
  (rowers * votes_per_rower) / votes_per_coach = 36 :=
by sorry

end NUMINAMATH_CALUDE_coaches_in_conference_l3437_343762


namespace NUMINAMATH_CALUDE_cost_of_planting_flowers_l3437_343702

/-- The cost of planting flowers given the prices of flowers, clay pot, and soil bag. -/
theorem cost_of_planting_flowers
  (flower_cost : ℕ)
  (clay_pot_cost : ℕ)
  (soil_bag_cost : ℕ)
  (h1 : flower_cost = 9)
  (h2 : clay_pot_cost = flower_cost + 20)
  (h3 : soil_bag_cost = flower_cost - 2) :
  flower_cost + clay_pot_cost + soil_bag_cost = 45 := by
  sorry

#check cost_of_planting_flowers

end NUMINAMATH_CALUDE_cost_of_planting_flowers_l3437_343702


namespace NUMINAMATH_CALUDE_multiplication_problem_l3437_343787

-- Define a custom type for single digits
def Digit := { n : Nat // n < 10 }

-- Define a function to convert a two-digit number to a natural number
def twoDigitToNat (d1 d2 : Digit) : Nat := 10 * d1.val + d2.val

-- Define a function to convert a three-digit number to a natural number
def threeDigitToNat (d1 d2 d3 : Digit) : Nat := 100 * d1.val + 10 * d2.val + d3.val

-- Define a function to convert a four-digit number to a natural number
def fourDigitToNat (d1 d2 d3 d4 : Digit) : Nat := 1000 * d1.val + 100 * d2.val + 10 * d3.val + d4.val

theorem multiplication_problem (A B C E F : Digit) :
  A ≠ B → A ≠ C → A ≠ E → A ≠ F →
  B ≠ C → B ≠ E → B ≠ F →
  C ≠ E → C ≠ F →
  E ≠ F →
  Nat.Prime (twoDigitToNat E F) →
  threeDigitToNat A B C * twoDigitToNat E F = fourDigitToNat E F E F →
  A.val + B.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_l3437_343787


namespace NUMINAMATH_CALUDE_left_of_origin_abs_value_l3437_343773

theorem left_of_origin_abs_value (a : ℝ) : 
  (a < 0) → (|a| = 4.5) → (a = -4.5) := by sorry

end NUMINAMATH_CALUDE_left_of_origin_abs_value_l3437_343773


namespace NUMINAMATH_CALUDE_square_perimeter_32cm_l3437_343796

theorem square_perimeter_32cm (side_length : ℝ) (h : side_length = 8) : 
  4 * side_length = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_32cm_l3437_343796


namespace NUMINAMATH_CALUDE_inequality_proof_l3437_343728

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (2 * a^2 / (1 + a + a * b)^2 + 
   2 * b^2 / (1 + b + b * c)^2 + 
   2 * c^2 / (1 + c + c * a)^2 + 
   9 / ((1 + a + a * b) * (1 + b + b * c) * (1 + c + c * a))) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3437_343728


namespace NUMINAMATH_CALUDE_first_equation_is_golden_second_equation_root_l3437_343794

-- Definition of a golden equation
def is_golden_equation (a b c : ℝ) : Prop := a ≠ 0 ∧ a - b + c = 0

-- Theorem 1: 4x^2 + 11x + 7 = 0 is a golden equation
theorem first_equation_is_golden : is_golden_equation 4 11 7 := by sorry

-- Theorem 2: If 3x^2 - mx + n = 0 is a golden equation and m is a root, then m = -1 or m = 3/2
theorem second_equation_root (m n : ℝ) :
  is_golden_equation 3 (-m) n →
  (3 * m^2 - m * m + n = 0) →
  (m = -1 ∨ m = 3/2) := by sorry

end NUMINAMATH_CALUDE_first_equation_is_golden_second_equation_root_l3437_343794


namespace NUMINAMATH_CALUDE_natural_solutions_count_l3437_343729

theorem natural_solutions_count :
  ∃! (s : Finset (ℕ × ℕ)), s.card = 4 ∧ ∀ (x y : ℕ), (x, y) ∈ s ↔ 2 * x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_natural_solutions_count_l3437_343729


namespace NUMINAMATH_CALUDE_product_of_cubic_fractions_l3437_343760

theorem product_of_cubic_fractions :
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 3) * (f 4) * (f 5) * (f 6) * (f 7) = 57 / 84 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cubic_fractions_l3437_343760


namespace NUMINAMATH_CALUDE_tan_pi_plus_2alpha_l3437_343790

theorem tan_pi_plus_2alpha (α : Real) 
  (h1 : α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi))
  (h2 : Real.sin (Real.pi / 2 + α) = 1 / 3) : 
  Real.tan (Real.pi + 2 * α) = 4 * Real.sqrt 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_plus_2alpha_l3437_343790


namespace NUMINAMATH_CALUDE_distance_between_vertices_l3437_343710

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := x^2 - 6*x + 13
def parabola2 (x : ℝ) : ℝ := x^2 + 2*x + 4

-- Define the vertex of a parabola
def vertex (f : ℝ → ℝ) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem distance_between_vertices : 
  distance (vertex parabola1) (vertex parabola2) = Real.sqrt 17 := by sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l3437_343710


namespace NUMINAMATH_CALUDE_ppf_combination_l3437_343707

/-- Production Possibility Frontier (PPF) for a single female -/
def individual_ppf (K : ℝ) : ℝ := 40 - 2 * K

/-- Combined Production Possibility Frontier (PPF) for two females -/
def combined_ppf (K : ℝ) : ℝ := 80 - 2 * K

theorem ppf_combination (K : ℝ) (h : K ≤ 40) :
  combined_ppf K = individual_ppf (K / 2) + individual_ppf (K / 2) :=
by sorry

#check ppf_combination

end NUMINAMATH_CALUDE_ppf_combination_l3437_343707


namespace NUMINAMATH_CALUDE_basketball_spectators_l3437_343723

/-- Proves the number of children at a basketball match -/
theorem basketball_spectators (total : ℕ) (men : ℕ) (women : ℕ) (children : ℕ) : 
  total = 10000 →
  men = 7000 →
  children = 5 * women →
  total = men + women + children →
  children = 2500 := by
sorry

end NUMINAMATH_CALUDE_basketball_spectators_l3437_343723


namespace NUMINAMATH_CALUDE_dormitory_second_year_fraction_l3437_343798

theorem dormitory_second_year_fraction :
  ∀ (F S : ℚ),
  F + S = 1 →
  (4 : ℚ) / 5 * F = F - (1 : ℚ) / 5 * F →
  (1 : ℚ) / 3 * ((1 : ℚ) / 5 * F) = (1 : ℚ) / 15 * S →
  (14 : ℚ) / 15 * S = (7 : ℚ) / 15 →
  S = (1 : ℚ) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_dormitory_second_year_fraction_l3437_343798


namespace NUMINAMATH_CALUDE_dodecahedron_colorings_l3437_343713

/-- The number of faces in a regular dodecahedron -/
def num_faces : ℕ := 12

/-- The number of rotational symmetries of a regular dodecahedron -/
def num_symmetries : ℕ := 60

/-- The number of distinguishable colorings of a regular dodecahedron -/
def num_colorings : ℕ := (Nat.factorial (num_faces - 1)) / num_symmetries

theorem dodecahedron_colorings :
  num_colorings = 665280 :=
sorry

end NUMINAMATH_CALUDE_dodecahedron_colorings_l3437_343713


namespace NUMINAMATH_CALUDE_books_sold_total_l3437_343785

/-- The total number of books sold by three salespeople over three days -/
def total_books_sold (matias_monday olivia_monday luke_monday : ℕ) : ℕ :=
  let matias_tuesday := 2 * matias_monday
  let olivia_tuesday := 3 * olivia_monday
  let luke_tuesday := luke_monday / 2
  let matias_wednesday := 3 * matias_tuesday
  let olivia_wednesday := 4 * olivia_tuesday
  let luke_wednesday := luke_tuesday
  matias_monday + matias_tuesday + matias_wednesday +
  olivia_monday + olivia_tuesday + olivia_wednesday +
  luke_monday + luke_tuesday + luke_wednesday

/-- Theorem stating the total number of books sold by Matias, Olivia, and Luke over three days -/
theorem books_sold_total : total_books_sold 7 5 12 = 167 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_total_l3437_343785


namespace NUMINAMATH_CALUDE_semicircle_three_part_area_l3437_343703

/-- Given a semicircle of radius r divided into three equal parts, the area bounded by two adjacent 
    chords (connecting division points to one end of the diameter) and the arc between them is πr²/6. -/
theorem semicircle_three_part_area (r : ℝ) (hr : r > 0) : 
  let semicircle_area := π * r^2 / 2
  let sector_angle := π / 3
  let sector_area := sector_angle / (2 * π) * π * r^2
  sector_area = π * r^2 / 6 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_three_part_area_l3437_343703


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3437_343780

theorem tan_alpha_value (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3)
  (h2 : Real.tan β = 2) : 
  Real.tan α = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3437_343780


namespace NUMINAMATH_CALUDE_light_path_in_cube_l3437_343759

theorem light_path_in_cube (cube_side : ℝ) (reflect_point_dist1 : ℝ) (reflect_point_dist2 : ℝ) :
  cube_side = 12 ∧ reflect_point_dist1 = 7 ∧ reflect_point_dist2 = 5 →
  ∃ (m n : ℕ), 
    (m = 12 ∧ n = 218) ∧
    (∀ (p : ℕ), Prime p → ¬(p^2 ∣ n)) ∧
    (m * Real.sqrt n = 12 * cube_side) :=
by sorry

end NUMINAMATH_CALUDE_light_path_in_cube_l3437_343759


namespace NUMINAMATH_CALUDE_complementary_angles_l3437_343791

theorem complementary_angles (A B : ℝ) : 
  A + B = 90 →  -- A and B are complementary
  A = 7 * B →   -- A is 7 times B
  A = 78.75 :=  -- A is 78.75°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_l3437_343791


namespace NUMINAMATH_CALUDE_equation_solution_l3437_343793

theorem equation_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) :
  y = 1 + Real.sqrt 2 ∨ y = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3437_343793


namespace NUMINAMATH_CALUDE_sample_size_is_70_l3437_343782

/-- Represents the ratio of quantities for products A, B, and C -/
structure ProductRatio where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the sample sizes for products A, B, and C -/
structure SampleSize where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total sample size given the sample sizes for each product -/
def totalSampleSize (s : SampleSize) : ℕ := s.a + s.b + s.c

/-- Theorem stating that given the product ratio and the sample size of product A, 
    the total sample size is 70 -/
theorem sample_size_is_70 (ratio : ProductRatio) (sample : SampleSize) :
  ratio = ⟨3, 4, 7⟩ → sample.a = 15 → totalSampleSize sample = 70 := by
  sorry

#check sample_size_is_70

end NUMINAMATH_CALUDE_sample_size_is_70_l3437_343782


namespace NUMINAMATH_CALUDE_building_shadow_length_l3437_343746

/-- Given a flagpole and a building under similar conditions, 
    calculate the length of the shadow cast by the building -/
theorem building_shadow_length 
  (flagpole_height : ℝ) 
  (flagpole_shadow : ℝ) 
  (building_height : ℝ) 
  (h1 : flagpole_height = 18)
  (h2 : flagpole_shadow = 45)
  (h3 : building_height = 26)
  : ∃ building_shadow : ℝ, building_shadow = 65 := by
  sorry

#check building_shadow_length

end NUMINAMATH_CALUDE_building_shadow_length_l3437_343746


namespace NUMINAMATH_CALUDE_nine_digit_integer_count_l3437_343783

/-- The number of digits in the integers we're counting -/
def num_digits : ℕ := 9

/-- The count of possible digits for the first position (1-9) -/
def first_digit_choices : ℕ := 9

/-- The count of possible digits for each remaining position (0-9) -/
def other_digit_choices : ℕ := 10

/-- The number of 9-digit positive integers that do not start with 0 -/
def count_9digit_integers : ℕ := first_digit_choices * (other_digit_choices ^ (num_digits - 1))

theorem nine_digit_integer_count :
  count_9digit_integers = 900000000 := by
  sorry

end NUMINAMATH_CALUDE_nine_digit_integer_count_l3437_343783


namespace NUMINAMATH_CALUDE_decagon_adjacent_vertex_probability_l3437_343708

/-- A decagon is a polygon with 10 vertices -/
def Decagon : ℕ := 10

/-- The number of vertices adjacent to any given vertex in a decagon -/
def AdjacentVertices : ℕ := 2

/-- The probability of selecting two adjacent vertices when choosing two distinct vertices at random from a decagon -/
theorem decagon_adjacent_vertex_probability : 
  (AdjacentVertices : ℚ) / (Decagon - 1 : ℚ) = 2 / 9 := by sorry

end NUMINAMATH_CALUDE_decagon_adjacent_vertex_probability_l3437_343708


namespace NUMINAMATH_CALUDE_base9_sum_and_subtract_l3437_343750

/-- Converts a base 9 number represented as a list of digits to a natural number. -/
def base9ToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 9 * acc + d) 0

/-- Converts a natural number to its base 9 representation as a list of digits. -/
def natToBase9 (n : Nat) : List Nat :=
  if n < 9 then [n]
  else (n % 9) :: natToBase9 (n / 9)

theorem base9_sum_and_subtract :
  let a := base9ToNat [1, 5, 3]  -- 351₉
  let b := base9ToNat [5, 6, 4]  -- 465₉
  let c := base9ToNat [2, 3, 1]  -- 132₉
  let d := base9ToNat [7, 4, 1]  -- 147₉
  natToBase9 (a + b + c - d) = [7, 4, 8] := by
  sorry

end NUMINAMATH_CALUDE_base9_sum_and_subtract_l3437_343750


namespace NUMINAMATH_CALUDE_angle_through_point_l3437_343734

theorem angle_through_point (α : Real) : 
  0 ≤ α ∧ α ≤ 2 * Real.pi → 
  (∃ r : Real, r > 0 ∧ r * Real.cos α = Real.cos (2 * Real.pi / 3) ∧ 
                      r * Real.sin α = Real.sin (2 * Real.pi / 3)) →
  α = 5 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_angle_through_point_l3437_343734


namespace NUMINAMATH_CALUDE_abc_value_l3437_343724

theorem abc_value (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + 1/b = 5)
  (h2 : b + 1/c = 2)
  (h3 : c + 1/a = 9/4) :
  a * b * c = (7 + Real.sqrt 21) / 8 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l3437_343724


namespace NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l3437_343711

/-- A parabola with equation y = x^2 - 8x + m has its vertex on the x-axis if and only if m = 16 -/
theorem parabola_vertex_on_x_axis (m : ℝ) :
  (∃ x : ℝ, x^2 - 8*x + m = 0 ∧ 
   ∀ t : ℝ, t^2 - 8*t + m ≥ 0) ↔ 
  m = 16 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l3437_343711


namespace NUMINAMATH_CALUDE_mean_of_smallest_elements_l3437_343712

/-- The arithmetic mean of the smallest elements of all r-element subsets of {1, 2, ..., n} -/
def F (n r : ℕ) : ℚ :=
  sorry

/-- Theorem stating that F(n,r) = (n+1)/(r+1) for 1 ≤ r ≤ n -/
theorem mean_of_smallest_elements (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) :
  F n r = (n + 1 : ℚ) / (r + 1 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_mean_of_smallest_elements_l3437_343712


namespace NUMINAMATH_CALUDE_parallel_plane_intersection_lines_parallel_l3437_343775

-- Define the concept of a plane
variable (Plane : Type)

-- Define the concept of a line
variable (Line : Type)

-- Define the parallel relation between planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the intersection relation between a plane and a line
variable (intersects : Plane → Plane → Line → Prop)

-- Theorem statement
theorem parallel_plane_intersection_lines_parallel 
  (P1 P2 P3 : Plane) (l1 l2 : Line) :
  parallel_planes P1 P2 →
  intersects P3 P1 l1 →
  intersects P3 P2 l2 →
  -- Conclusion: l1 and l2 are parallel
  parallel_planes P1 P2 := by sorry

end NUMINAMATH_CALUDE_parallel_plane_intersection_lines_parallel_l3437_343775


namespace NUMINAMATH_CALUDE_f_g_5_l3437_343776

def g (x : ℝ) : ℝ := 4 * x - 5

def f (x : ℝ) : ℝ := 6 * x + 11

theorem f_g_5 : f (g 5) = 101 := by
  sorry

end NUMINAMATH_CALUDE_f_g_5_l3437_343776


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l3437_343770

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 10) : x^2 + 1/x^2 = 98 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l3437_343770


namespace NUMINAMATH_CALUDE_coinciding_white_pairs_l3437_343768

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCount where
  red : Nat
  blue : Nat
  white : Nat

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  redRed : Nat
  blueBlue : Nat
  redWhite : Nat
  whiteWhite : Nat

/-- The main theorem that proves the number of coinciding white pairs -/
theorem coinciding_white_pairs
  (initial_count : TriangleCount)
  (coinciding : CoincidingPairs)
  (h1 : initial_count.red = 2)
  (h2 : initial_count.blue = 4)
  (h3 : initial_count.white = 6)
  (h4 : coinciding.redRed = 1)
  (h5 : coinciding.blueBlue = 2)
  (h6 : coinciding.redWhite = 2)
  : coinciding.whiteWhite = 4 := by
  sorry

end NUMINAMATH_CALUDE_coinciding_white_pairs_l3437_343768


namespace NUMINAMATH_CALUDE_incorrect_average_calculation_l3437_343784

theorem incorrect_average_calculation (n : ℕ) (correct_avg : ℚ) (error : ℚ) :
  n = 10 ∧ correct_avg = 17 ∧ error = 10 →
  (n * correct_avg - error) / n = 16 := by
sorry

end NUMINAMATH_CALUDE_incorrect_average_calculation_l3437_343784


namespace NUMINAMATH_CALUDE_golden_raisin_cost_l3437_343733

/-- The cost per scoop of natural seedless raisins -/
def natural_cost : ℝ := 3.45

/-- The number of scoops of natural seedless raisins -/
def natural_scoops : ℕ := 20

/-- The number of scoops of golden seedless raisins -/
def golden_scoops : ℕ := 20

/-- The cost per scoop of the mixture -/
def mixture_cost : ℝ := 3

/-- The cost per scoop of golden seedless raisins -/
def golden_cost : ℝ := 2.55

theorem golden_raisin_cost :
  (natural_cost * natural_scoops + golden_cost * golden_scoops) / (natural_scoops + golden_scoops) = mixture_cost :=
sorry

end NUMINAMATH_CALUDE_golden_raisin_cost_l3437_343733


namespace NUMINAMATH_CALUDE_min_value_x_plus_4y_l3437_343739

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 1) : 
  x + 4*y ≥ 3 + 2*Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_4y_l3437_343739


namespace NUMINAMATH_CALUDE_decimal_representation_of_225_999_l3437_343766

theorem decimal_representation_of_225_999 :
  ∃ (d : ℕ → ℕ), 
    (∀ n, d n < 10) ∧ 
    (∀ n, d (n + 3) = d n) ∧
    (d 0 = 2 ∧ d 1 = 2 ∧ d 2 = 5) ∧
    (d 80 = 5) ∧
    (225 : ℚ) / 999 = ∑' n, (d n : ℚ) / 10 ^ (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_decimal_representation_of_225_999_l3437_343766


namespace NUMINAMATH_CALUDE_faculty_marriage_percentage_l3437_343704

theorem faculty_marriage_percentage (total : ℕ) (total_pos : 0 < total) : 
  let women := (70 : ℚ) / 100 * total
  let men := total - women
  let single_men := (1 : ℚ) / 3 * men
  let married_men := (2 : ℚ) / 3 * men
  (married_men : ℚ) / total ≥ (20 : ℚ) / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_faculty_marriage_percentage_l3437_343704


namespace NUMINAMATH_CALUDE_sin_function_range_l3437_343777

open Real

theorem sin_function_range (ω : ℝ) (h_ω_pos : ω > 0) :
  (∀ x ∈ Set.Icc (-π/3) (π/4), 2 * sin (ω * x) ≥ -2) ∧
  (∃ x ∈ Set.Icc (-π/3) (π/4), 2 * sin (ω * x) = -2) ∧
  (∀ x ∈ Set.Icc (-π/3) (π/4), 2 * sin (ω * x) < 2) →
  3/2 ≤ ω ∧ ω < 2 := by
sorry

end NUMINAMATH_CALUDE_sin_function_range_l3437_343777


namespace NUMINAMATH_CALUDE_polynomial_properties_l3437_343772

/-- A polynomial of the form f(x) = ax^5 + bx^3 + 4x + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + 4 * x + c

theorem polynomial_properties :
  ∀ (a b c : ℝ),
    (f a b c 0 = 6 → c = 6) ∧
    (f a b c 0 = -2 ∧ f a b c 1 = 5 → f a b c (-1) = -9) ∧
    (f a b c 5 + f a b c (-5) = 6 ∧ f a b c 2 = 8 → f a b c (-2) = -2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_properties_l3437_343772


namespace NUMINAMATH_CALUDE_sportswear_processing_equation_l3437_343769

/-- Represents the clothing processing factory problem --/
theorem sportswear_processing_equation 
  (total_sportswear : ℕ) 
  (processed_before_tech : ℕ) 
  (efficiency_increase : ℚ) 
  (total_time : ℚ) 
  (x : ℚ) 
  (h1 : total_sportswear = 400)
  (h2 : processed_before_tech = 160)
  (h3 : efficiency_increase = 1/5)
  (h4 : total_time = 18)
  (h5 : x > 0) :
  (processed_before_tech / x) + ((total_sportswear - processed_before_tech) / ((1 + efficiency_increase) * x)) = total_time :=
sorry

end NUMINAMATH_CALUDE_sportswear_processing_equation_l3437_343769


namespace NUMINAMATH_CALUDE_mn_square_value_l3437_343705

theorem mn_square_value (m n : ℤ) 
  (h1 : |m - n| = n - m) 
  (h2 : |m| = 4) 
  (h3 : |n| = 3) : 
  (m + n)^2 = 1 ∨ (m + n)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_mn_square_value_l3437_343705


namespace NUMINAMATH_CALUDE_runners_meet_time_l3437_343752

/-- Represents a runner with a constant speed -/
structure Runner where
  speed : ℝ

/-- Represents the circular track -/
structure Track where
  length : ℝ

/-- Calculates the time when all runners meet again -/
def meeting_time (track : Track) (runners : List Runner) : ℝ :=
  sorry

theorem runners_meet_time (track : Track) (runners : List Runner) :
  track.length = 600 ∧
  runners = [
    Runner.mk 4.5,
    Runner.mk 4.9,
    Runner.mk 5.1
  ] →
  meeting_time track runners = 3000 := by
  sorry

end NUMINAMATH_CALUDE_runners_meet_time_l3437_343752


namespace NUMINAMATH_CALUDE_banana_groups_l3437_343789

theorem banana_groups (total_bananas : ℕ) (num_groups : ℕ) 
  (h1 : total_bananas = 392) 
  (h2 : num_groups = 196) : 
  total_bananas / num_groups = 2 := by
  sorry

end NUMINAMATH_CALUDE_banana_groups_l3437_343789


namespace NUMINAMATH_CALUDE_original_plan_calculation_l3437_343753

def thursday_sales : ℕ := 210
def friday_sales : ℕ := 2 * thursday_sales
def saturday_sales : ℕ := 130
def sunday_sales : ℕ := saturday_sales / 2
def excess_sales : ℕ := 325

def total_sales : ℕ := thursday_sales + friday_sales + saturday_sales + sunday_sales

theorem original_plan_calculation :
  total_sales - excess_sales = 500 := by sorry

end NUMINAMATH_CALUDE_original_plan_calculation_l3437_343753


namespace NUMINAMATH_CALUDE_part1_part2_l3437_343778

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - a) * (x - a + 1) ≤ 0}
def B : Set ℝ := {x | x^2 + x - 2 < 0}

-- Define the proposition p
def p (m : ℝ) : Prop := ∃ x ∈ B, x^2 + (2*m + 1)*x + m^2 - m > 8

-- Theorem for part 1
theorem part1 : 
  (∀ x, x ∈ A a → x ∈ B) ∧ (∃ x, x ∈ B ∧ x ∉ A a) → 
  a > -1 ∧ a < 1 :=
sorry

-- Theorem for part 2
theorem part2 : 
  (¬ p m) → m ≥ -1 ∧ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l3437_343778


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3437_343765

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem solution_set_of_inequality
  (h_decreasing : ∀ x y, x < y → f y < f x)
  (h_point_A : f 0 = 3)
  (h_point_B : f 3 = -1) :
  {x : ℝ | |f (x + 1) - 1| < 2} = Set.Ioo (-1 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3437_343765


namespace NUMINAMATH_CALUDE_curve_transformation_l3437_343731

theorem curve_transformation (x y : ℝ) : 
  y = Real.sin (π / 2 + 2 * x) → 
  y = -Real.cos (5 * π / 6 - 3 * ((2 / 3) * x - π / 18)) := by
sorry

end NUMINAMATH_CALUDE_curve_transformation_l3437_343731


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3437_343730

theorem min_value_quadratic (x : ℝ) :
  ∃ (y_min : ℝ), ∀ (y : ℝ), y = 4 * x^2 + 8 * x + 12 → y ≥ y_min ∧ ∃ (x_0 : ℝ), 4 * x_0^2 + 8 * x_0 + 12 = y_min :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3437_343730


namespace NUMINAMATH_CALUDE_inequality_proof_l3437_343788

theorem inequality_proof (x : ℝ) (h : x > 0) : x + (2016^2016)/x^2016 ≥ 2017 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3437_343788


namespace NUMINAMATH_CALUDE_labourer_income_l3437_343767

/-- Prove that the monthly income of a labourer is 75 --/
theorem labourer_income :
  ∀ (avg_expenditure_6m : ℝ) (debt : ℝ) (expenditure_4m : ℝ) (savings : ℝ),
    avg_expenditure_6m = 80 →
    debt > 0 →
    expenditure_4m = 60 →
    savings = 30 →
    ∃ (income : ℝ),
      income * 6 - debt + income * 4 = avg_expenditure_6m * 6 + expenditure_4m * 4 + debt + savings ∧
      income = 75 := by
  sorry

end NUMINAMATH_CALUDE_labourer_income_l3437_343767
