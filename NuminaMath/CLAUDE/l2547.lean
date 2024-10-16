import Mathlib

namespace NUMINAMATH_CALUDE_withdraw_300_from_two_banks_in_20_bills_l2547_254720

/-- Calculates the number of bills received when withdrawing money from two banks -/
def number_of_bills (amount_per_bank : ℕ) (num_banks : ℕ) (bill_value : ℕ) : ℕ :=
  (amount_per_bank * num_banks) / bill_value

/-- Proves that withdrawing $300 from each of two banks in $20 bills results in 30 bills -/
theorem withdraw_300_from_two_banks_in_20_bills : 
  number_of_bills 300 2 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_withdraw_300_from_two_banks_in_20_bills_l2547_254720


namespace NUMINAMATH_CALUDE_unique_solution_l2547_254743

def is_valid_digit (d : ℕ) : Prop := d > 0 ∧ d ≤ 9

def are_distinct (a b c d e f g : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
  e ≠ f ∧ e ≠ g ∧
  f ≠ g

def to_six_digit_number (a b c d e f : ℕ) : ℕ :=
  100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f

theorem unique_solution :
  ∀ A B : ℕ,
    is_valid_digit A →
    is_valid_digit B →
    are_distinct 1 2 3 4 5 A B →
    (to_six_digit_number A 1 2 3 4 5) % B = 0 →
    (to_six_digit_number 1 2 3 4 5 A) % B = 0 →
    A = 9 ∧ B = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2547_254743


namespace NUMINAMATH_CALUDE_treaty_of_paris_preliminary_articles_l2547_254763

/-- Represents days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Calculates the day of the week given a number of days before a known day -/
def daysBefore (knownDay : DayOfWeek) (daysBefore : Nat) : DayOfWeek :=
  sorry

theorem treaty_of_paris_preliminary_articles :
  let treatyDay : DayOfWeek := DayOfWeek.Thursday
  let daysBetween : Nat := 621
  daysBefore treatyDay daysBetween = DayOfWeek.Tuesday :=
sorry

end NUMINAMATH_CALUDE_treaty_of_paris_preliminary_articles_l2547_254763


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2547_254739

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 7*a^2 + 5*a + 2 = 0) →
  (b^3 - 7*b^2 + 5*b + 2 = 0) →
  (c^3 - 7*c^2 + 5*c + 2 = 0) →
  (a / (b*c + 1) + b / (a*c + 1) + c / (a*b + 1) = 15/2) := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2547_254739


namespace NUMINAMATH_CALUDE_gcd_seven_digit_special_l2547_254772

def seven_digit_special (n : ℕ) : ℕ := 1001000 * n + n / 100

theorem gcd_seven_digit_special :
  ∃ (k : ℕ), k > 0 ∧ ∀ (n : ℕ), 100 ≤ n ∧ n < 1000 → 
    (k ∣ seven_digit_special n) ∧
    ∀ (m : ℕ), m > k ∧ (∀ (p : ℕ), 100 ≤ p ∧ p < 1000 → m ∣ seven_digit_special p) → False :=
by sorry

end NUMINAMATH_CALUDE_gcd_seven_digit_special_l2547_254772


namespace NUMINAMATH_CALUDE_square_minus_twice_plus_one_equals_three_l2547_254752

theorem square_minus_twice_plus_one_equals_three :
  let x : ℝ := Real.sqrt 3 + 1
  x^2 - 2*x + 1 = 3 := by sorry

end NUMINAMATH_CALUDE_square_minus_twice_plus_one_equals_three_l2547_254752


namespace NUMINAMATH_CALUDE_paul_min_correct_answers_l2547_254769

def min_correct_answers (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) (unanswered_points : ℕ) (attempted : ℕ) (min_score : ℕ) : ℕ :=
  let unanswered := total_questions - attempted
  let unanswered_score := unanswered * unanswered_points
  let required_attempted_score := min_score - unanswered_score
  ((required_attempted_score + incorrect_points * attempted - 1) / (correct_points + incorrect_points)) + 1

theorem paul_min_correct_answers :
  min_correct_answers 25 7 2 2 18 90 = 13 := by
  sorry

end NUMINAMATH_CALUDE_paul_min_correct_answers_l2547_254769


namespace NUMINAMATH_CALUDE_pedal_triangle_largest_angle_l2547_254785

/-- Represents an acute triangle with vertices A, B, C and corresponding angles α, β, γ. -/
structure AcuteTriangle where
  α : Real
  β : Real
  γ : Real
  acute_angles : α ≤ β ∧ β ≤ γ ∧ γ < Real.pi / 2
  angle_sum : α + β + γ = Real.pi

/-- Represents the pedal triangle of an acute triangle. -/
def PedalTriangle (t : AcuteTriangle) : Prop :=
  ∃ (largest_pedal_angle : Real),
    largest_pedal_angle = Real.pi - 2 * t.α ∧
    largest_pedal_angle ≥ t.γ

/-- 
Theorem: The largest angle in the pedal triangle of an acute triangle is at least 
as large as the largest angle in the original triangle. Equality holds when the 
original triangle is isosceles with the equal angles at least 60°.
-/
theorem pedal_triangle_largest_angle (t : AcuteTriangle) : 
  PedalTriangle t ∧ 
  (Real.pi - 2 * t.α = t.γ ↔ t.α = t.β ∧ t.γ ≥ Real.pi / 3) := by
  sorry


end NUMINAMATH_CALUDE_pedal_triangle_largest_angle_l2547_254785


namespace NUMINAMATH_CALUDE_sum_remainder_mod_9_l2547_254741

theorem sum_remainder_mod_9 : (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_9_l2547_254741


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l2547_254709

theorem min_value_of_function (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 :=
sorry

theorem equality_condition (x : ℝ) (h : x > 1) : 
  x + 1 / (x - 1) = 3 ↔ x = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l2547_254709


namespace NUMINAMATH_CALUDE_negative_sqrt_four_equals_negative_two_l2547_254788

theorem negative_sqrt_four_equals_negative_two : -Real.sqrt 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_four_equals_negative_two_l2547_254788


namespace NUMINAMATH_CALUDE_unique_triangle_exists_l2547_254703

/-- Triangle ABC with given side lengths and angle -/
structure Triangle where
  a : ℝ
  b : ℝ
  A : ℝ

/-- Predicate for a valid triangle satisfying the given conditions -/
def is_valid_triangle (t : Triangle) : Prop :=
  t.a = Real.sqrt 3 ∧ t.b = 1 ∧ t.A = 130 * (Real.pi / 180)

/-- Theorem stating that there exists exactly one valid triangle -/
theorem unique_triangle_exists : ∃! t : Triangle, is_valid_triangle t :=
sorry

end NUMINAMATH_CALUDE_unique_triangle_exists_l2547_254703


namespace NUMINAMATH_CALUDE_scientific_notation_170000_l2547_254795

theorem scientific_notation_170000 :
  170000 = 1.7 * (10 : ℝ)^5 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_170000_l2547_254795


namespace NUMINAMATH_CALUDE_max_value_implies_m_l2547_254700

/-- The function f(x) = -x^3 + 6x^2 + m has a maximum value of 12 -/
def has_max_12 (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∃ (x_max : ℝ), f x_max = 12 ∧ ∀ (x : ℝ), f x ≤ 12

/-- The function f(x) = -x^3 + 6x^2 + m -/
def f (m : ℝ) (x : ℝ) : ℝ := -x^3 + 6*x^2 + m

theorem max_value_implies_m (m : ℝ) :
  has_max_12 (f m) m → m = -20 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_m_l2547_254700


namespace NUMINAMATH_CALUDE_coefficient_x4_is_negative_seven_l2547_254722

/-- The coefficient of x^4 in the expanded expression -/
def coefficient_x4 (a b c d e f g : ℤ) : ℤ :=
  5 * a - 3 * 0 + 4 * (-3)

/-- The expression to be expanded -/
def expression (x : ℚ) : ℚ :=
  5 * (x^4 - 2*x^3 + x^2) - 3 * (x^2 - x + 1) + 4 * (x^6 - 3*x^4 + x^3)

theorem coefficient_x4_is_negative_seven :
  coefficient_x4 1 (-2) 1 0 (-1) 1 = -7 := by sorry

end NUMINAMATH_CALUDE_coefficient_x4_is_negative_seven_l2547_254722


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2547_254794

theorem polynomial_evaluation : 
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x - 10 = 0 ∧ x^3 - 3*x^2 - 9*x + 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2547_254794


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2547_254737

/-- For a quadratic equation 9x^2 + kx + 49 = 0 to have exactly one real solution,
    the positive value of k must be 42. -/
theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + k * x + 49 = 0) ↔ k = 42 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2547_254737


namespace NUMINAMATH_CALUDE_bakers_cakes_l2547_254747

/-- Baker's cake selling problem -/
theorem bakers_cakes (initial_cakes : ℕ) (cakes_left : ℕ) (h1 : initial_cakes = 48) (h2 : cakes_left = 4) :
  initial_cakes - cakes_left = 44 := by
  sorry

end NUMINAMATH_CALUDE_bakers_cakes_l2547_254747


namespace NUMINAMATH_CALUDE_x_twelve_equals_one_l2547_254710

theorem x_twelve_equals_one (x : ℝ) (h : x + 1/x = 2) : x^12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_twelve_equals_one_l2547_254710


namespace NUMINAMATH_CALUDE_expression_equality_l2547_254784

theorem expression_equality : 5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2547_254784


namespace NUMINAMATH_CALUDE_altered_detergent_amount_is_180_l2547_254793

/-- Represents the ratio of components in a cleaning solution -/
structure CleaningSolution :=
  (bleach : ℚ)
  (detergent : ℚ)
  (fabricSoftener : ℚ)
  (water : ℚ)

/-- Calculates the amount of detergent in the altered solution -/
def alteredDetergentAmount (original : CleaningSolution) (alteredWaterAmount : ℚ) : ℚ :=
  let bleachToDetergentRatio := original.bleach / original.detergent * 3
  let fabricSoftenerToDetergentRatio := (original.fabricSoftener / original.detergent) / 2
  let detergentToWaterRatio := (original.detergent / original.water) * (2/3)
  
  let newDetergentToWaterRatio := detergentToWaterRatio * alteredWaterAmount
  
  newDetergentToWaterRatio

/-- Theorem stating that the altered solution contains 180 liters of detergent -/
theorem altered_detergent_amount_is_180 :
  let original := CleaningSolution.mk 4 40 60 100
  let alteredWaterAmount := 300
  alteredDetergentAmount original alteredWaterAmount = 180 := by
  sorry

end NUMINAMATH_CALUDE_altered_detergent_amount_is_180_l2547_254793


namespace NUMINAMATH_CALUDE_z_squared_abs_l2547_254745

-- Define the complex number z
def z : ℂ := sorry

-- State the theorem
theorem z_squared_abs : z * (1 + Complex.I) = 1 + 3 * Complex.I → Complex.abs (z^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_z_squared_abs_l2547_254745


namespace NUMINAMATH_CALUDE_sum_abcd_equals_negative_28_over_3_l2547_254736

theorem sum_abcd_equals_negative_28_over_3 
  (a b c d : ℚ) 
  (h : a + 3 = b + 7 ∧ a + 3 = c + 5 ∧ a + 3 = d + 9 ∧ a + 3 = a + b + c + d + 13) : 
  a + b + c + d = -28/3 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_negative_28_over_3_l2547_254736


namespace NUMINAMATH_CALUDE_book_club_groups_l2547_254756

theorem book_club_groups (n m : ℕ) (hn : n = 7) (hm : m = 4) :
  Nat.choose n m = 35 := by
  sorry

end NUMINAMATH_CALUDE_book_club_groups_l2547_254756


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2547_254787

theorem quadratic_two_distinct_roots (a b c : ℝ) (h : 2016 + a^2 + a*c < a*b) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a*x₁^2 + b*x₁ + c = 0 ∧ a*x₂^2 + b*x₂ + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2547_254787


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2547_254738

/-- Given a hyperbola C: x²/a² - y²/b² = 1 (a > 0, b > 0), 
    if a focus F and its symmetric point with respect to one asymptote 
    lies on the other asymptote, then the eccentricity e of the hyperbola is 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}
  let asymptote₁ := {p : ℝ × ℝ | p.2 = (b / a) * p.1}
  let asymptote₂ := {p : ℝ × ℝ | p.2 = -(b / a) * p.1}
  ∃ (F : ℝ × ℝ), F ∈ C ∧ 
    (∃ (S : ℝ × ℝ), S ∈ asymptote₂ ∧ 
      (∀ (p : ℝ × ℝ), p ∈ asymptote₁ → 
        ((F.1 + S.1) / 2 = p.1 ∧ (F.2 + S.2) / 2 = p.2))) →
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2547_254738


namespace NUMINAMATH_CALUDE_almond_walnut_ratio_is_five_to_two_l2547_254704

/-- Represents a mixture of almonds and walnuts -/
structure NutMixture where
  total_weight : ℝ
  almond_weight : ℝ
  almond_parts : ℝ
  walnut_parts : ℝ

/-- The ratio of almonds to walnuts in a nut mixture -/
def almond_to_walnut_ratio (mix : NutMixture) : ℝ × ℝ :=
  (mix.almond_parts, mix.walnut_parts)

/-- Theorem stating the ratio of almonds to walnuts in the specific mixture -/
theorem almond_walnut_ratio_is_five_to_two 
  (mix : NutMixture)
  (h1 : mix.total_weight = 210)
  (h2 : mix.almond_weight = 150)
  (h3 : mix.almond_parts = 5)
  (h4 : mix.almond_parts + mix.walnut_parts = mix.total_weight / (mix.almond_weight / mix.almond_parts)) :
  almond_to_walnut_ratio mix = (5, 2) := by
  sorry


end NUMINAMATH_CALUDE_almond_walnut_ratio_is_five_to_two_l2547_254704


namespace NUMINAMATH_CALUDE_least_k_for_convergence_l2547_254766

def sequence_u : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => (3/2) * sequence_u n - (3/2) * (sequence_u n)^2

def M : ℚ := 2/3

theorem least_k_for_convergence :
  (∀ k : ℕ, k < 10 → |sequence_u k - M| > 1/2^1000) ∧
  |sequence_u 10 - M| ≤ 1/2^1000 := by sorry

end NUMINAMATH_CALUDE_least_k_for_convergence_l2547_254766


namespace NUMINAMATH_CALUDE_distance_between_points_l2547_254762

theorem distance_between_points : 
  ∀ (A B : ℝ), A = -1 ∧ B = 2020 → |A - B| = 2021 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2547_254762


namespace NUMINAMATH_CALUDE_determinant_trig_matrix_equals_one_l2547_254757

theorem determinant_trig_matrix_equals_one (α β γ : ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := λ i j ↦ 
    match i, j with
    | 0, 0 => Real.cos (α + γ) * Real.cos β
    | 0, 1 => Real.cos (α + γ) * Real.sin β
    | 0, 2 => -Real.sin (α + γ)
    | 1, 0 => -Real.sin β
    | 1, 1 => Real.cos β
    | 1, 2 => 0
    | 2, 0 => Real.sin (α + γ) * Real.cos β
    | 2, 1 => Real.sin (α + γ) * Real.sin β
    | 2, 2 => Real.cos (α + γ)
  Matrix.det M = 1 := by sorry

end NUMINAMATH_CALUDE_determinant_trig_matrix_equals_one_l2547_254757


namespace NUMINAMATH_CALUDE_h_properties_l2547_254712

-- Define the functions
noncomputable def g (x : ℝ) : ℝ := 2^x

-- f is symmetric to g with respect to y = x
def f_symmetric_to_g (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- Define h in terms of f
def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f (1 - |x|)

-- Main theorem
theorem h_properties (f : ℝ → ℝ) (hf : f_symmetric_to_g f) :
  (∀ x, h f x = h f (-x)) ∧ 
  (∀ x, h f x ≤ 0 ∧ h f 0 = 0) :=
sorry

end NUMINAMATH_CALUDE_h_properties_l2547_254712


namespace NUMINAMATH_CALUDE_trapezoid_diagonal_length_l2547_254729

/-- Represents a trapezoid ABCD with specific side lengths and angle -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  BC : ℝ
  cos_BCD : ℝ
  h_AB : AB = 27
  h_CD : CD = 28
  h_BC : BC = 5
  h_cos_BCD : cos_BCD = -2/7

/-- The length of the diagonal AC in the trapezoid -/
def diagonal_AC (t : Trapezoid) : Set ℝ :=
  {28, 2 * Real.sqrt 181}

/-- Theorem stating that the diagonal AC of the trapezoid is either 28 or 2√181 -/
theorem trapezoid_diagonal_length (t : Trapezoid) :
  ∃ x ∈ diagonal_AC t, x = (Real.sqrt ((t.AB - t.BC)^2 + (t.CD * Real.sqrt (1 - t.cos_BCD^2))^2)) :=
sorry

end NUMINAMATH_CALUDE_trapezoid_diagonal_length_l2547_254729


namespace NUMINAMATH_CALUDE_min_value_of_sum_squares_l2547_254768

theorem min_value_of_sum_squares (x y z : ℝ) 
  (eq1 : x + 2*y - 5*z = 3)
  (eq2 : x - 2*y - z = -5) :
  ∃ (min : ℝ), min = 54/11 ∧ ∀ (x' y' z' : ℝ), 
    x' + 2*y' - 5*z' = 3 → x' - 2*y' - z' = -5 → 
    x'^2 + y'^2 + z'^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_squares_l2547_254768


namespace NUMINAMATH_CALUDE_sector_perimeter_and_area_l2547_254730

/-- Given a circular sector with radius 6 cm and central angle π/4 radians,
    prove that its perimeter is 12 + 3π/2 cm and its area is 9π/2 cm². -/
theorem sector_perimeter_and_area :
  let r : ℝ := 6
  let θ : ℝ := π / 4
  let perimeter : ℝ := 2 * r + r * θ
  let area : ℝ := (1 / 2) * r^2 * θ
  perimeter = 12 + 3 * π / 2 ∧ area = 9 * π / 2 := by
  sorry


end NUMINAMATH_CALUDE_sector_perimeter_and_area_l2547_254730


namespace NUMINAMATH_CALUDE_original_number_before_increase_l2547_254734

theorem original_number_before_increase (x : ℝ) : x * 1.3 = 650 → x = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_number_before_increase_l2547_254734


namespace NUMINAMATH_CALUDE_intersection_line_of_given_circles_l2547_254750

/-- Circle with center and radius --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Line equation of the form ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The intersection line of two circles --/
def intersection_line (c1 c2 : Circle) : Line :=
  sorry

theorem intersection_line_of_given_circles :
  let c1 : Circle := { center := (1, 5), radius := 7 }
  let c2 : Circle := { center := (-2, -1), radius := 5 * Real.sqrt 2 }
  let l : Line := intersection_line c1 c2
  l.a = 1 ∧ l.b = 1 ∧ l.c = 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_line_of_given_circles_l2547_254750


namespace NUMINAMATH_CALUDE_complex_number_theorem_l2547_254753

def complex_number_problem (z₁ z₂ : ℂ) : Prop :=
  Complex.abs (z₁ * z₂) = 3 ∧ z₁ + z₂ = Complex.I * 2

theorem complex_number_theorem (z₁ z₂ : ℂ) 
  (h : complex_number_problem z₁ z₂) :
  (∀ w₁ w₂ : ℂ, complex_number_problem w₁ w₂ → Complex.abs w₁ ≤ 3) ∧
  (∀ w₁ w₂ : ℂ, complex_number_problem w₁ w₂ → Complex.abs w₁ ≥ 1) ∧
  (∃ w₁ w₂ : ℂ, complex_number_problem w₁ w₂ ∧ Complex.abs w₁ = 3) ∧
  (∃ w₁ w₂ : ℂ, complex_number_problem w₁ w₂ ∧ Complex.abs w₁ = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l2547_254753


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l2547_254705

theorem billion_to_scientific_notation :
  ∀ (x : ℝ), x = 508 → (x * (10^9 : ℝ)) = 5.08 * (10^11 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l2547_254705


namespace NUMINAMATH_CALUDE_min_value_given_max_l2547_254749

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- State the theorem
theorem min_value_given_max (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≥ f a y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 20) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ f a y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = -7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_given_max_l2547_254749


namespace NUMINAMATH_CALUDE_shared_vertex_angle_measure_l2547_254725

/-- The measure of the angle at the common vertex formed by a side of an equilateral triangle
    and a side of a regular pentagon, both inscribed in a circle. -/
def common_vertex_angle : ℝ := 24

/-- A regular pentagon inscribed in a circle -/
structure RegularPentagonInCircle where
  -- Add necessary fields

/-- An equilateral triangle inscribed in a circle -/
structure EquilateralTriangleInCircle where
  -- Add necessary fields

/-- Configuration of a regular pentagon and an equilateral triangle inscribed in a circle
    with a shared vertex -/
structure SharedVertexConfiguration where
  pentagon : RegularPentagonInCircle
  triangle : EquilateralTriangleInCircle
  -- Add field to represent the shared vertex

theorem shared_vertex_angle_measure (config : SharedVertexConfiguration) :
  common_vertex_angle = 24 := by
  sorry

end NUMINAMATH_CALUDE_shared_vertex_angle_measure_l2547_254725


namespace NUMINAMATH_CALUDE_triangle_area_l2547_254778

-- Define the triangle ABC and point K
variable (A B C K : ℝ × ℝ)

-- Define the conditions
def is_on_line (P Q R : ℝ × ℝ) : Prop := sorry
def is_altitude (P Q R S : ℝ × ℝ) : Prop := sorry
def distance (P Q : ℝ × ℝ) : ℝ := sorry
def area (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem triangle_area :
  is_on_line K B C →
  is_altitude A K B C →
  distance A C = 12 →
  distance B K = 9 →
  distance B C = 18 →
  area A B C = 27 * Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2547_254778


namespace NUMINAMATH_CALUDE_angle_sum_in_circle_l2547_254744

theorem angle_sum_in_circle (x : ℝ) : 6 * x + 3 * x + 4 * x + x + 2 * x = 360 → x = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_circle_l2547_254744


namespace NUMINAMATH_CALUDE_december_23_is_saturday_l2547_254780

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day of the week
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ k => nextDay (advanceDay d k)

-- Theorem statement
theorem december_23_is_saturday (thanksgiving : DayOfWeek) 
  (h : thanksgiving = DayOfWeek.Thursday) : 
  advanceDay thanksgiving 30 = DayOfWeek.Saturday := by
  sorry


end NUMINAMATH_CALUDE_december_23_is_saturday_l2547_254780


namespace NUMINAMATH_CALUDE_ratio_problem_l2547_254776

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 1 / 3) :
  a / c = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2547_254776


namespace NUMINAMATH_CALUDE_meet_time_opposite_directions_l2547_254719

/-- Represents an athlete running on a track -/
structure Athlete where
  lap_time : ℝ
  speed : ℝ

/-- Represents a closed track -/
structure Track where
  length : ℝ

/-- The scenario of two athletes running on a track -/
def running_scenario (t : Track) (a1 a2 : Athlete) : Prop :=
  a1.speed = t.length / a1.lap_time ∧
  a2.speed = t.length / a2.lap_time ∧
  a2.lap_time = a1.lap_time + 5 ∧
  30 * a1.speed - 30 * a2.speed = t.length

theorem meet_time_opposite_directions 
  (t : Track) (a1 a2 : Athlete) 
  (h : running_scenario t a1 a2) : 
  t.length / (a1.speed + a2.speed) = 6 := by
  sorry


end NUMINAMATH_CALUDE_meet_time_opposite_directions_l2547_254719


namespace NUMINAMATH_CALUDE_complement_A_in_S_l2547_254792

def S : Set ℕ := {x | 0 ≤ x ∧ x ≤ 5}
def A : Set ℕ := {x | 1 < x ∧ x < 5}

theorem complement_A_in_S : 
  (S \ A) = {0, 1, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_S_l2547_254792


namespace NUMINAMATH_CALUDE_min_marks_group_a_l2547_254723

/-- Represents the number of marks for each question in a group -/
structure GroupMarks where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the number of questions in each group -/
structure GroupQuestions where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The examination setup -/
structure Examination where
  marks : GroupMarks
  questions : GroupQuestions
  total_questions : ℕ
  total_marks : ℕ

/-- Conditions for the examination -/
def valid_examination (e : Examination) : Prop :=
  e.total_questions = 100 ∧
  e.questions.a + e.questions.b + e.questions.c = e.total_questions ∧
  e.questions.b = 23 ∧
  e.questions.c = 1 ∧
  e.marks.b = 2 ∧
  e.marks.c = 3 ∧
  e.total_marks = e.questions.a * e.marks.a + e.questions.b * e.marks.b + e.questions.c * e.marks.c ∧
  e.questions.a * e.marks.a ≥ (60 * e.total_marks) / 100

theorem min_marks_group_a (e : Examination) (h : valid_examination e) :
  e.marks.a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_min_marks_group_a_l2547_254723


namespace NUMINAMATH_CALUDE_roots_difference_squared_l2547_254721

theorem roots_difference_squared (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + 4 * x₁ - 3 = 0) →
  (2 * x₂^2 + 4 * x₂ - 3 = 0) →
  (x₁ - x₂)^2 = 10 := by
sorry

end NUMINAMATH_CALUDE_roots_difference_squared_l2547_254721


namespace NUMINAMATH_CALUDE_system_solution_iff_b_in_range_l2547_254783

/-- The system of equations has a solution for any real a if and only if 0 ≤ b ≤ 2 -/
theorem system_solution_iff_b_in_range (b : ℝ) :
  (∀ a : ℝ, ∃ x y : ℝ, x^2 - 2*x + y^2 = 0 ∧ a*x + y = a*b) ↔ 0 ≤ b ∧ b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_iff_b_in_range_l2547_254783


namespace NUMINAMATH_CALUDE_cube_edge_length_l2547_254728

theorem cube_edge_length (surface_area : ℝ) (edge_length : ℝ) :
  surface_area = 96 ∧ surface_area = 6 * edge_length^2 → edge_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l2547_254728


namespace NUMINAMATH_CALUDE_number_equation_l2547_254771

theorem number_equation (x : ℝ) : 0.833 * x = -60 → x = -72 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l2547_254771


namespace NUMINAMATH_CALUDE_machine_production_l2547_254773

/-- Given that 4 machines produce x units in 6 days at a constant rate,
    prove that 16 machines will produce 2x units in 3 days at the same rate. -/
theorem machine_production (x : ℝ) : 
  (∃ (rate : ℝ), rate > 0 ∧ 4 * rate * 6 = x) →
  (∃ (output : ℝ), 16 * (x / (4 * 6)) * 3 = output ∧ output = 2 * x) :=
by sorry

end NUMINAMATH_CALUDE_machine_production_l2547_254773


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_sum_of_reciprocals_of_roots_l2547_254716

-- Define the coefficients of the first equation: 2x^2 - 5x + 1 = 0
def a₁ : ℚ := 2
def b₁ : ℚ := -5
def c₁ : ℚ := 1

-- Define the coefficients of the second equation: 2x^2 - 11x + 13 = 0
def a₂ : ℚ := 2
def b₂ : ℚ := -11
def c₂ : ℚ := 13

-- Theorem for the sum of cubes of roots
theorem sum_of_cubes_of_roots :
  let x₁ := (-b₁ + Real.sqrt (b₁^2 - 4*a₁*c₁)) / (2*a₁)
  let x₂ := (-b₁ - Real.sqrt (b₁^2 - 4*a₁*c₁)) / (2*a₁)
  x₁^3 + x₂^3 = 95/8 := by sorry

-- Theorem for the sum of reciprocals of roots
theorem sum_of_reciprocals_of_roots :
  let y₁ := (-b₂ + Real.sqrt (b₂^2 - 4*a₂*c₂)) / (2*a₂)
  let y₂ := (-b₂ - Real.sqrt (b₂^2 - 4*a₂*c₂)) / (2*a₂)
  y₁/y₂ + y₂/y₁ = 69/26 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_sum_of_reciprocals_of_roots_l2547_254716


namespace NUMINAMATH_CALUDE_tree_distance_l2547_254755

/-- Given a yard of length 250 meters with 51 trees planted at equal distances,
    including one at each end, the distance between consecutive trees is 5 meters. -/
theorem tree_distance (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 250 →
  num_trees = 51 →
  yard_length / (num_trees - 1) = 5 :=
by sorry

end NUMINAMATH_CALUDE_tree_distance_l2547_254755


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l2547_254791

theorem arithmetic_geometric_sequence_problem (a b : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) - a n = d) →  -- a_n is arithmetic with common difference d
  d ≠ 0 →  -- d is not equal to 0
  a 2046 + a 1978 - (a 2012)^2 = 0 →  -- given condition
  (∃ r, ∀ n, b (n + 1) = r * b n) →  -- b_n is geometric
  b 2012 = a 2012 →  -- given condition
  b 2010 * b 2014 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l2547_254791


namespace NUMINAMATH_CALUDE_ellipse_and_line_equations_l2547_254706

/-- Given an ellipse with the specified properties, prove its standard equation and the equations of the intersecting line. -/
theorem ellipse_and_line_equations 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (e : ℝ) 
  (h_e : e = Real.sqrt 2 / 2) 
  (h_point : a^2 * (1/2)^2 + b^2 * (Real.sqrt 2 / 2)^2 = 1) 
  (k : ℝ) 
  (h_intersection : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / (2 * a^2) + y₁^2 / (2 * b^2) = 1 ∧
    x₂^2 / (2 * a^2) + y₂^2 / (2 * b^2) = 1 ∧
    y₁ = k * (x₁ + 1) ∧
    y₂ = k * (x₂ + 1) ∧
    ((x₁ - 1)^2 + y₁^2 + (x₂ - 1)^2 + y₂^2 + 2 * ((x₁ - 1) * (x₂ - 1) + y₁ * y₂))^(1/2) = 2 * Real.sqrt 26 / 3) :
  (a^2 = 2 ∧ b^2 = 1) ∧ (k = 1 ∨ k = -1) := by
  sorry


end NUMINAMATH_CALUDE_ellipse_and_line_equations_l2547_254706


namespace NUMINAMATH_CALUDE_sum_expression_l2547_254781

theorem sum_expression (x y z k : ℝ) (h1 : y = 3 * x) (h2 : z = k * y) :
  x + y + z = (4 + 3 * k) * x := by
  sorry

end NUMINAMATH_CALUDE_sum_expression_l2547_254781


namespace NUMINAMATH_CALUDE_optimal_route_unchanged_for_given_network_l2547_254799

/-- Represents the transportation network of a country -/
structure TransportNetwork where
  num_cities : Nat
  capital_travel_time : Real
  city_connection_time : Real
  initial_transfer_time : Real
  reduced_transfer_time : Real

/-- Calculates the travel time via the capital -/
def time_via_capital (network : TransportNetwork) (transfer_time : Real) : Real :=
  2 * network.capital_travel_time + transfer_time

/-- Calculates the maximum travel time via cyclic connections -/
def time_via_cycle (network : TransportNetwork) (transfer_time : Real) : Real :=
  5 * network.city_connection_time + 4 * transfer_time

/-- Determines if the optimal route remains unchanged after reducing transfer time -/
def optimal_route_unchanged (network : TransportNetwork) : Prop :=
  let initial_time_via_capital := time_via_capital network network.initial_transfer_time
  let initial_time_via_cycle := time_via_cycle network network.initial_transfer_time
  let reduced_time_via_capital := time_via_capital network network.reduced_transfer_time
  let reduced_time_via_cycle := time_via_cycle network network.reduced_transfer_time
  (initial_time_via_capital ≤ initial_time_via_cycle) ∧
  (reduced_time_via_capital ≤ reduced_time_via_cycle)

theorem optimal_route_unchanged_for_given_network :
  optimal_route_unchanged
    { num_cities := 11
    , capital_travel_time := 7
    , city_connection_time := 3
    , initial_transfer_time := 2
    , reduced_transfer_time := 1.5 } := by
  sorry

end NUMINAMATH_CALUDE_optimal_route_unchanged_for_given_network_l2547_254799


namespace NUMINAMATH_CALUDE_quarter_difference_zero_l2547_254740

/-- Represents a coin collection with nickels, dimes, and quarters. -/
structure CoinCollection where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- The total number of coins in the collection. -/
def CoinCollection.total (c : CoinCollection) : ℕ :=
  c.nickels + c.dimes + c.quarters

/-- The total value of the collection in cents. -/
def CoinCollection.value (c : CoinCollection) : ℕ :=
  5 * c.nickels + 10 * c.dimes + 25 * c.quarters

/-- Predicate for a valid coin collection according to the problem conditions. -/
def isValidCollection (c : CoinCollection) : Prop :=
  c.total = 150 ∧ c.value = 2000

/-- The theorem to be proved. -/
theorem quarter_difference_zero :
  ∀ c₁ c₂ : CoinCollection, isValidCollection c₁ → isValidCollection c₂ →
  c₁.quarters = c₂.quarters :=
sorry

end NUMINAMATH_CALUDE_quarter_difference_zero_l2547_254740


namespace NUMINAMATH_CALUDE_license_plate_count_l2547_254779

/-- The number of digits used in the license plate -/
def num_digits : ℕ := 4

/-- The number of letters used in the license plate -/
def num_letters : ℕ := 3

/-- The number of possible digits (0-9) -/
def digit_choices : ℕ := 10

/-- The number of letters in the alphabet -/
def letter_choices : ℕ := 32

/-- The maximum number of different car license plates -/
def max_license_plates : ℕ := digit_choices ^ num_digits * letter_choices ^ num_letters

theorem license_plate_count : max_license_plates = 327680000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2547_254779


namespace NUMINAMATH_CALUDE_seashells_given_to_joan_l2547_254751

theorem seashells_given_to_joan (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 8) 
  (h2 : remaining_seashells = 2) : 
  initial_seashells - remaining_seashells = 6 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_to_joan_l2547_254751


namespace NUMINAMATH_CALUDE_simple_interest_time_period_l2547_254714

theorem simple_interest_time_period (P : ℝ) (P_pos : P > 0) : 
  let R : ℝ := 4
  let SI : ℝ := (2/5) * P
  let T : ℝ := SI * 100 / (P * R)
  T = 10 := by sorry

end NUMINAMATH_CALUDE_simple_interest_time_period_l2547_254714


namespace NUMINAMATH_CALUDE_largest_number_of_three_l2547_254718

theorem largest_number_of_three (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (sum_prod_eq : a * b + a * c + b * c = -10)
  (prod_eq : a * b * c = -18) :
  max a (max b c) = -1 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_of_three_l2547_254718


namespace NUMINAMATH_CALUDE_equation_to_lines_l2547_254789

/-- The set of points satisfying the given equation is equivalent to the union of two lines -/
theorem equation_to_lines : 
  ∀ x y : ℝ, (2*x^2 + y^2 + 3*x*y + 3*x + y = 2) ↔ 
  (y = -x - 2 ∨ y = -2*x + 1) := by sorry

end NUMINAMATH_CALUDE_equation_to_lines_l2547_254789


namespace NUMINAMATH_CALUDE_reduction_equivalence_original_value_proof_l2547_254748

theorem reduction_equivalence (original : ℝ) (reduced : ℝ) : 
  reduced = original * (1 / 1000) ↔ reduced = original * 0.001 :=
by sorry

theorem original_value_proof : 
  ∃ (original : ℝ), 16.9 * (1 / 1000) = 0.0169 ∧ original = 16.9 :=
by sorry

end NUMINAMATH_CALUDE_reduction_equivalence_original_value_proof_l2547_254748


namespace NUMINAMATH_CALUDE_log_sqrt10_1000sqrt10_l2547_254764

theorem log_sqrt10_1000sqrt10 : Real.log (1000 * Real.sqrt 10) / Real.log (Real.sqrt 10) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt10_1000sqrt10_l2547_254764


namespace NUMINAMATH_CALUDE_commission_change_point_l2547_254732

/-- The sales amount where the commission rate changes -/
def X : ℝ := 1822.98

/-- The total sales amount -/
def total_sales : ℝ := 15885.42

/-- The amount remitted to the parent company -/
def remitted_amount : ℝ := 15000

/-- The commission rate for sales up to X -/
def low_rate : ℝ := 0.10

/-- The commission rate for sales exceeding X -/
def high_rate : ℝ := 0.05

theorem commission_change_point : 
  X * low_rate + (total_sales - X) * high_rate = total_sales - remitted_amount :=
sorry

end NUMINAMATH_CALUDE_commission_change_point_l2547_254732


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_plus_i_l2547_254797

theorem imaginary_part_of_i_times_one_plus_i (i : ℂ) : 
  i * i = -1 → Complex.im (i * (1 + i)) = 1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_plus_i_l2547_254797


namespace NUMINAMATH_CALUDE_coin_stack_arrangements_l2547_254731

/-- Represents a coin with a color and a face side -/
inductive Coin
  | Gold : Bool → Coin
  | Silver : Bool → Coin

/-- A stack of coins -/
def CoinStack := List Coin

/-- Checks if two adjacent coins are not face to face -/
def validAdjacent : Coin → Coin → Bool
  | Coin.Gold true, Coin.Gold true => false
  | Coin.Gold true, Coin.Silver true => false
  | Coin.Silver true, Coin.Gold true => false
  | Coin.Silver true, Coin.Silver true => false
  | _, _ => true

/-- Checks if a stack of coins is valid (no adjacent face to face) -/
def validStack : CoinStack → Bool
  | [] => true
  | [_] => true
  | (c1 :: c2 :: rest) => validAdjacent c1 c2 && validStack (c2 :: rest)

/-- Counts the number of gold coins in a stack -/
def countGold : CoinStack → Nat
  | [] => 0
  | (Coin.Gold _) :: rest => 1 + countGold rest
  | _ :: rest => countGold rest

/-- Counts the number of silver coins in a stack -/
def countSilver : CoinStack → Nat
  | [] => 0
  | (Coin.Silver _) :: rest => 1 + countSilver rest
  | _ :: rest => countSilver rest

/-- The main theorem to prove -/
theorem coin_stack_arrangements :
  (∃ (validStacks : List CoinStack),
    (∀ s ∈ validStacks, validStack s = true) ∧
    (∀ s ∈ validStacks, countGold s = 5) ∧
    (∀ s ∈ validStacks, countSilver s = 5) ∧
    validStacks.length = 2772) := by
  sorry

end NUMINAMATH_CALUDE_coin_stack_arrangements_l2547_254731


namespace NUMINAMATH_CALUDE_infinite_solutions_equation_l2547_254796

theorem infinite_solutions_equation :
  ∀ n : ℕ+, ∃ a b c : ℕ+,
    (a : ℝ) ^ 2 + (b : ℝ) ^ 5 = (c : ℝ) ^ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_equation_l2547_254796


namespace NUMINAMATH_CALUDE_blue_ridge_elementary_calculation_l2547_254702

theorem blue_ridge_elementary_calculation (num_classrooms : ℕ) 
  (students_per_classroom : ℕ) (turtles_per_classroom : ℕ) (teachers_per_classroom : ℕ) : 
  num_classrooms = 6 →
  students_per_classroom = 22 →
  turtles_per_classroom = 2 →
  teachers_per_classroom = 1 →
  num_classrooms * students_per_classroom - 
  (num_classrooms * turtles_per_classroom + num_classrooms * teachers_per_classroom) = 114 := by
  sorry

#check blue_ridge_elementary_calculation

end NUMINAMATH_CALUDE_blue_ridge_elementary_calculation_l2547_254702


namespace NUMINAMATH_CALUDE_cubic_root_identity_l2547_254770

theorem cubic_root_identity (a b : ℝ) (h1 : a ≠ b) (h2 : (Real.rpow a (1/3) + Real.rpow b (1/3))^3 = a^2 * b^2) : 
  (3*a + 1)*(3*b + 1) - 3*a^2*b^2 = 1 := by sorry

end NUMINAMATH_CALUDE_cubic_root_identity_l2547_254770


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2547_254774

/-- Calculates the area of a rectangular field given specific fencing conditions -/
theorem rectangular_field_area 
  (uncovered_side : ℝ) 
  (total_fencing : ℝ) 
  (h1 : uncovered_side = 20) 
  (h2 : total_fencing = 88) : 
  uncovered_side * ((total_fencing - uncovered_side) / 2) = 680 :=
by
  sorry

#check rectangular_field_area

end NUMINAMATH_CALUDE_rectangular_field_area_l2547_254774


namespace NUMINAMATH_CALUDE_one_tangent_line_l2547_254708

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y - 26 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 4 = 0

-- Define a function to count the number of tangent lines
def count_tangent_lines (C1 C2 : ℝ → ℝ → Prop) : ℕ := sorry

-- Theorem stating that there is exactly one tangent line
theorem one_tangent_line : count_tangent_lines C1 C2 = 1 := by sorry

end NUMINAMATH_CALUDE_one_tangent_line_l2547_254708


namespace NUMINAMATH_CALUDE_smallest_largest_a_sum_l2547_254754

theorem smallest_largest_a_sum (a b c : ℝ) (sum_eq : a + b + c = 5) (sum_sq_eq : a^2 + b^2 + c^2 = 8) :
  (∃ (a_min a_max : ℝ), 
    (∀ x : ℝ, (∃ y z : ℝ, x + y + z = 5 ∧ x^2 + y^2 + z^2 = 8) → a_min ≤ x ∧ x ≤ a_max) ∧
    a_min = 1 ∧ 
    a_max = 3 ∧ 
    a_min + a_max = 4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_largest_a_sum_l2547_254754


namespace NUMINAMATH_CALUDE_range_of_a_l2547_254765

/-- Given a line l: x + y + a = 0 and a point A(0, 2), if there exists a point M on line l 
    such that |MA|^2 + |MO|^2 = 10 (where O is the origin), then -2√2 - 1 ≤ a ≤ 2√2 - 1 -/
theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 + (-x-a)^2 + x^2 + (-x-a-2)^2 = 10) → 
  -2 * Real.sqrt 2 - 1 ≤ a ∧ a ≤ 2 * Real.sqrt 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2547_254765


namespace NUMINAMATH_CALUDE_phi_value_l2547_254707

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is decreasing on an interval [a, b] if for all x, y in [a, b],
    x < y implies f(x) > f(y) -/
def IsDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x > f y

theorem phi_value (f : ℝ → ℝ) (φ : ℝ) 
    (h1 : f = λ x => 2 * Real.sin (2 * x + φ + π / 3))
    (h2 : IsOdd f)
    (h3 : IsDecreasingOn f 0 (π / 4)) :
    φ = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_phi_value_l2547_254707


namespace NUMINAMATH_CALUDE_distinct_hands_count_l2547_254726

def special_deck_size : ℕ := 60
def hand_size : ℕ := 13

theorem distinct_hands_count : (special_deck_size.choose hand_size) = 75287520 := by
  sorry

end NUMINAMATH_CALUDE_distinct_hands_count_l2547_254726


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2547_254758

/-- Given that N(4,10) is the midpoint of CD and C(14,6), prove that the sum of D's coordinates is 8 -/
theorem midpoint_coordinate_sum (N C D : ℝ × ℝ) : 
  N = (4, 10) → 
  C = (14, 6) → 
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2547_254758


namespace NUMINAMATH_CALUDE_similar_triangles_height_l2547_254701

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 →
  area_ratio = 9 →
  ∃ h_large : ℝ, h_large = h_small * Real.sqrt area_ratio :=
by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l2547_254701


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2547_254760

theorem perfect_square_trinomial (m : ℝ) :
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + 2*(m-1)*x + 16 = (a*x + b)^2) →
  m = 5 ∨ m = -3 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2547_254760


namespace NUMINAMATH_CALUDE_inequality_proof_l2547_254735

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hln : Real.log a * Real.log b > 0) :
  a^(b - 1) < b^(a - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2547_254735


namespace NUMINAMATH_CALUDE_quadratic_roots_isosceles_triangle_l2547_254798

theorem quadratic_roots_isosceles_triangle (b : ℝ) (α β : ℝ) :
  (∀ x, x^2 + b*x + 1 = 0 ↔ x = α ∨ x = β) →
  α > β →
  (α^2 + β^2 = 3*α - 3*β ∧ α^2 + β^2 = α*β) ∨
  (α^2 + β^2 = 3*α - 3*β ∧ 3*α - 3*β = α*β) ∨
  (3*α - 3*β = α*β ∧ α*β = α^2 + β^2) →
  b = Real.sqrt 5 ∨ b = -Real.sqrt 5 ∨ b = Real.sqrt 8 ∨ b = -Real.sqrt 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_isosceles_triangle_l2547_254798


namespace NUMINAMATH_CALUDE_right_triangle_area_l2547_254746

theorem right_triangle_area (h : ℝ) (angle : ℝ) :
  h = 10 * Real.sqrt 3 →
  angle = 30 * π / 180 →
  let s := h / 2
  let l := Real.sqrt 3 / 2 * h
  0.5 * s * l = 37.5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2547_254746


namespace NUMINAMATH_CALUDE_margo_round_trip_distance_l2547_254733

/-- Calculates the total distance covered in a round trip given the time for each leg and the average speed -/
def total_distance (outward_time return_time avg_speed : ℚ) : ℚ :=
  avg_speed * (outward_time + return_time) / 60

/-- Proves that the total distance covered in the given scenario is 4 miles -/
theorem margo_round_trip_distance :
  total_distance (15 : ℚ) (25 : ℚ) (6 : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_margo_round_trip_distance_l2547_254733


namespace NUMINAMATH_CALUDE_point_relation_l2547_254761

-- Define the line equation
def line_equation (x y b : ℝ) : Prop := y = -Real.sqrt 2 * x + b

-- Define the theorem
theorem point_relation (m n b : ℝ) 
  (h1 : line_equation (-2) m b)
  (h2 : line_equation 3 n b) : 
  m > n := by sorry

end NUMINAMATH_CALUDE_point_relation_l2547_254761


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l2547_254790

theorem polygon_interior_angles (P : ℕ) (h1 : P > 2) : 
  (∃ (a d : ℝ), 
    a = 20 ∧ 
    a + (P - 1) * d = 160 ∧ 
    (P / 2 : ℝ) * (a + (a + (P - 1) * d)) = 180 * (P - 2)) → 
  P = 4 := by
sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l2547_254790


namespace NUMINAMATH_CALUDE_sam_seashells_l2547_254759

/-- Given that Mary found 47 seashells and the total number of seashells
    found by Sam and Mary is 65, prove that Sam found 18 seashells. -/
theorem sam_seashells (mary_seashells : ℕ) (total_seashells : ℕ)
    (h1 : mary_seashells = 47)
    (h2 : total_seashells = 65) :
    total_seashells - mary_seashells = 18 := by
  sorry

end NUMINAMATH_CALUDE_sam_seashells_l2547_254759


namespace NUMINAMATH_CALUDE_rectangle_area_l2547_254786

theorem rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 3 → ratio = 3 → 2 * r * (ratio + 1) = 108 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2547_254786


namespace NUMINAMATH_CALUDE_probability_coprime_pairs_l2547_254713

def S : Finset Nat := Finset.range 8

theorem probability_coprime_pairs (a b : Nat) (h : a ∈ S ∧ b ∈ S ∧ a ≠ b) :
  (Finset.filter (fun p : Nat × Nat => p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 ≠ p.2 ∧ Nat.gcd p.1 p.2 = 1) 
    (S.product S)).card / (Finset.filter (fun p : Nat × Nat => p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 ≠ p.2) 
    (S.product S)).card = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_coprime_pairs_l2547_254713


namespace NUMINAMATH_CALUDE_balloon_sum_l2547_254782

theorem balloon_sum (x y : ℝ) (hx : x = 7.5) (hy : y = 5.2) : x + y = 12.7 := by
  sorry

end NUMINAMATH_CALUDE_balloon_sum_l2547_254782


namespace NUMINAMATH_CALUDE_min_correct_answers_for_environmental_quiz_l2547_254724

/-- Represents a quiz with scoring rules -/
structure Quiz where
  totalQuestions : ℕ
  correctScore : ℕ
  incorrectDeduction : ℕ

/-- Calculates the score for a given number of correct answers -/
def calculateScore (quiz : Quiz) (correctAnswers : ℕ) : ℤ :=
  (quiz.correctScore * correctAnswers : ℤ) - 
  (quiz.incorrectDeduction * (quiz.totalQuestions - correctAnswers) : ℤ)

/-- The minimum number of correct answers needed to exceed the target score -/
def minCorrectAnswers (quiz : Quiz) (targetScore : ℤ) : ℕ :=
  quiz.totalQuestions.succ

theorem min_correct_answers_for_environmental_quiz :
  let quiz : Quiz := ⟨30, 10, 5⟩
  let targetScore : ℤ := 90
  minCorrectAnswers quiz targetScore = 17 ∧
  ∀ (x : ℕ), x ≥ minCorrectAnswers quiz targetScore → calculateScore quiz x > targetScore :=
by sorry

end NUMINAMATH_CALUDE_min_correct_answers_for_environmental_quiz_l2547_254724


namespace NUMINAMATH_CALUDE_factorization_and_simplification_l2547_254717

theorem factorization_and_simplification (x : ℝ) (h : x^2 ≠ 3 ∧ x^2 ≠ -1) :
  (12 * x^6 + 36 * x^4 - 9) / (3 * x^4 - 9 * x^2 - 9) =
  (4 * x^4 * (x^2 + 3) - 3) / ((x^2 - 3) * (x^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_and_simplification_l2547_254717


namespace NUMINAMATH_CALUDE_lorelai_jellybeans_count_l2547_254777

/-- The number of jellybeans Gigi has -/
def gigi_jellybeans : ℕ := 15

/-- The number of extra jellybeans Rory has compared to Gigi -/
def rory_extra_jellybeans : ℕ := 30

/-- The number of jellybeans Rory has -/
def rory_jellybeans : ℕ := gigi_jellybeans + rory_extra_jellybeans

/-- The total number of jellybeans both girls have -/
def total_girls_jellybeans : ℕ := gigi_jellybeans + rory_jellybeans

/-- The number of times Lorelai has eaten compared to both girls -/
def lorelai_multiplier : ℕ := 3

/-- The number of jellybeans Lorelai has eaten -/
def lorelai_jellybeans : ℕ := total_girls_jellybeans * lorelai_multiplier

theorem lorelai_jellybeans_count : lorelai_jellybeans = 180 := by
  sorry

end NUMINAMATH_CALUDE_lorelai_jellybeans_count_l2547_254777


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2547_254767

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) (m : ℕ) :
  geometric_sequence a q →
  a 1 = 1 →
  q ≠ 1 →
  q ≠ -1 →
  a m = a 1 * a 2 * a 3 * a 4 * a 5 →
  m = 11 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2547_254767


namespace NUMINAMATH_CALUDE_pentagonal_sum_theorem_l2547_254742

def pentagonal_layer_sum (n : ℕ) : ℕ := 4 * (3^(n-1) - 1)

theorem pentagonal_sum_theorem (n : ℕ) :
  n ≥ 1 →
  (pentagonal_layer_sum 1 = 0) →
  (∀ k : ℕ, k ≥ 1 → pentagonal_layer_sum (k+1) = 3 * pentagonal_layer_sum k + 4) →
  pentagonal_layer_sum n = 4 * (3^(n-1) - 1) :=
by sorry

end NUMINAMATH_CALUDE_pentagonal_sum_theorem_l2547_254742


namespace NUMINAMATH_CALUDE_diagonals_are_space_l2547_254775

/-- A cube with diagonals forming a 60-degree angle --/
structure CubeWithDiagonals where
  /-- The measure of the angle between two diagonals --/
  angle : ℝ
  /-- The angle between the diagonals is 60 degrees --/
  angle_is_60 : angle = 60

/-- The types of diagonals in a cube --/
inductive DiagonalType
  | Face
  | Space

/-- Theorem: If the angle between two diagonals of a cube is 60 degrees,
    then these diagonals are space diagonals --/
theorem diagonals_are_space (c : CubeWithDiagonals) :
  ∃ (d : DiagonalType), d = DiagonalType.Space :=
sorry

end NUMINAMATH_CALUDE_diagonals_are_space_l2547_254775


namespace NUMINAMATH_CALUDE_equation_solutions_l2547_254711

theorem equation_solutions (x : ℝ) (y : ℝ) : 
  x^2 + 6 * (x / (x - 3))^2 = 81 →
  y = ((x - 3)^2 * (x + 4)) / (3*x - 4) →
  (y = -9 ∨ y = 225/176) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2547_254711


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l2547_254715

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of rectangles in a square grid -/
def rectanglesInGrid (n : ℕ) : ℕ := (choose n 2) ^ 2

theorem rectangles_in_5x5_grid :
  rectanglesInGrid 5 = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l2547_254715


namespace NUMINAMATH_CALUDE_positive_number_square_sum_l2547_254727

theorem positive_number_square_sum (n : ℝ) : n > 0 → n^2 + n = 245 → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_square_sum_l2547_254727
