import Mathlib

namespace NUMINAMATH_CALUDE_linear_function_decreasing_l921_92147

theorem linear_function_decreasing (k : ℝ) :
  (∀ x y : ℝ, x < y → ((k + 2) * x + 1) > ((k + 2) * y + 1)) ↔ k < -2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_l921_92147


namespace NUMINAMATH_CALUDE_oil_per_cylinder_l921_92189

theorem oil_per_cylinder (cylinders : ℕ) (oil_added : ℕ) (oil_needed : ℕ) :
  cylinders = 6 →
  oil_added = 16 →
  oil_needed = 32 →
  (oil_added + oil_needed) / cylinders = 8 := by
  sorry

end NUMINAMATH_CALUDE_oil_per_cylinder_l921_92189


namespace NUMINAMATH_CALUDE_inequality_implication_l921_92106

theorem inequality_implication (a b c d : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) 
  (h : a + b * Real.sqrt 5 < c + d * Real.sqrt 5) : 
  a < c ∧ b < d := by sorry

end NUMINAMATH_CALUDE_inequality_implication_l921_92106


namespace NUMINAMATH_CALUDE_fraction_problem_l921_92184

theorem fraction_problem (N : ℝ) (F : ℝ) 
  (h1 : (1/4 : ℝ) * (1/3 : ℝ) * F * N = 35)
  (h2 : (40/100 : ℝ) * N = 420) : 
  F = 2/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l921_92184


namespace NUMINAMATH_CALUDE_intersection_of_lines_l921_92145

theorem intersection_of_lines :
  let x : ℚ := 77 / 32
  let y : ℚ := 57 / 20
  (8 * x - 5 * y = 10) ∧ (9 * x + y^2 = 25) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l921_92145


namespace NUMINAMATH_CALUDE_article_cost_proof_l921_92146

theorem article_cost_proof (sp1 sp2 : ℝ) (gain_percentage : ℝ) :
  sp1 = 348 ∧ sp2 = 350 ∧ gain_percentage = 0.05 →
  ∃ (cost gain : ℝ),
    sp1 = cost + gain ∧
    sp2 = cost + gain + gain_percentage * gain ∧
    cost = 308 :=
by sorry

end NUMINAMATH_CALUDE_article_cost_proof_l921_92146


namespace NUMINAMATH_CALUDE_books_borrowed_second_day_l921_92134

def initial_books : ℕ := 100
def people_first_day : ℕ := 5
def books_per_person : ℕ := 2
def remaining_books : ℕ := 70

theorem books_borrowed_second_day :
  initial_books - people_first_day * books_per_person - remaining_books = 20 :=
by sorry

end NUMINAMATH_CALUDE_books_borrowed_second_day_l921_92134


namespace NUMINAMATH_CALUDE_integer_division_problem_l921_92161

theorem integer_division_problem (D d q r : ℤ) 
  (h1 : D = q * d + r) 
  (h2 : D + 65 = q * (d + 5) + r) : q = 13 := by
  sorry

end NUMINAMATH_CALUDE_integer_division_problem_l921_92161


namespace NUMINAMATH_CALUDE_product_repeating_decimal_9_and_8_l921_92120

/-- The repeating decimal 0.999... -/
def repeating_decimal_9 : ℝ := 0.999999

/-- Theorem: The product of 0.999... and 8 is equal to 8 -/
theorem product_repeating_decimal_9_and_8 : repeating_decimal_9 * 8 = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_repeating_decimal_9_and_8_l921_92120


namespace NUMINAMATH_CALUDE_ticket_price_possibilities_l921_92168

def is_valid_price (y : ℕ) : Prop :=
  y > 0 ∧ 90 % y = 0 ∧ 100 % y = 0

theorem ticket_price_possibilities :
  ∃! (n : ℕ), n > 0 ∧ (∃ (S : Finset ℕ), S.card = n ∧ ∀ y ∈ S, is_valid_price y) :=
sorry

end NUMINAMATH_CALUDE_ticket_price_possibilities_l921_92168


namespace NUMINAMATH_CALUDE_determinant_scaling_l921_92195

theorem determinant_scaling (a b c d : ℝ) :
  Matrix.det !![a, b; c, d] = 5 →
  Matrix.det !![3*a, 3*b; 2*c, 2*d] = 30 := by
  sorry

end NUMINAMATH_CALUDE_determinant_scaling_l921_92195


namespace NUMINAMATH_CALUDE_lolita_weekday_milk_l921_92125

/-- The number of milk boxes Lolita drinks on a single weekday -/
def weekday_milk : ℕ := 3

/-- The number of milk boxes Lolita drinks on Saturday -/
def saturday_milk : ℕ := 2 * weekday_milk

/-- The number of milk boxes Lolita drinks on Sunday -/
def sunday_milk : ℕ := 3 * weekday_milk

/-- The total number of milk boxes Lolita drinks in a week -/
def total_weekly_milk : ℕ := 30

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

theorem lolita_weekday_milk :
  weekdays * weekday_milk = 15 ∧
  weekdays * weekday_milk + saturday_milk + sunday_milk = total_weekly_milk :=
sorry

end NUMINAMATH_CALUDE_lolita_weekday_milk_l921_92125


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_negative_two_l921_92129

theorem sqrt_expression_equals_negative_two :
  Real.sqrt 24 + (Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2) - (Real.sqrt 3 + Real.sqrt 2)^2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_negative_two_l921_92129


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l921_92127

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_N :
  M ∩ N = {(3, -1)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l921_92127


namespace NUMINAMATH_CALUDE_complex_equation_solution_l921_92170

theorem complex_equation_solution (a : ℝ) :
  (2 + a * Complex.I) / (1 + Real.sqrt 2 * Complex.I) = -(Real.sqrt 2) * Complex.I →
  a = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l921_92170


namespace NUMINAMATH_CALUDE_meeting_2015_same_as_first_l921_92105

/-- Represents a point on a line segment --/
structure Point :=
  (position : ℝ)

/-- Represents a person moving on a line segment --/
structure Person :=
  (speed : ℝ)
  (startPosition : Point)
  (startTime : ℝ)

/-- Represents a meeting between two people --/
structure Meeting :=
  (position : Point)
  (time : ℝ)

/-- The theorem stating that the 2015th meeting occurs at the same point as the first meeting --/
theorem meeting_2015_same_as_first 
  (a b : Person) 
  (segment : Set Point) 
  (first_meeting last_meeting : Meeting) :
  first_meeting.position = last_meeting.position :=
sorry

end NUMINAMATH_CALUDE_meeting_2015_same_as_first_l921_92105


namespace NUMINAMATH_CALUDE_billy_ice_cubes_l921_92155

/-- The number of ice cubes in each tray -/
def cubes_per_tray : ℕ := 25

/-- The number of trays Billy has -/
def number_of_trays : ℕ := 15

/-- The total number of ice cubes Billy can make -/
def total_ice_cubes : ℕ := cubes_per_tray * number_of_trays

theorem billy_ice_cubes : total_ice_cubes = 375 := by sorry

end NUMINAMATH_CALUDE_billy_ice_cubes_l921_92155


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l921_92128

theorem max_sum_of_factors (x y : ℕ+) (h : x * y = 48) :
  x + y ≤ 49 ∧ ∃ (a b : ℕ+), a * b = 48 ∧ a + b = 49 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l921_92128


namespace NUMINAMATH_CALUDE_vector_dot_product_l921_92166

/-- Given two vectors a and b in ℝ², prove that their dot product is 25 -/
theorem vector_dot_product (a b : ℝ × ℝ) : 
  a = (1, 2) → a - (1/5 : ℝ) • b = (-2, 1) → a.1 * b.1 + a.2 * b.2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l921_92166


namespace NUMINAMATH_CALUDE_square_plus_self_even_l921_92160

theorem square_plus_self_even (n : ℤ) : Even (n^2 + n) := by sorry

end NUMINAMATH_CALUDE_square_plus_self_even_l921_92160


namespace NUMINAMATH_CALUDE_problem_solution_l921_92198

theorem problem_solution :
  ∀ n : ℤ, 3 ≤ n ∧ n ≤ 9 ∧ n ≡ 6557 [ZMOD 7] → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l921_92198


namespace NUMINAMATH_CALUDE_cassie_water_refills_l921_92158

/-- Represents the number of cups of water Cassie aims to drink daily -/
def daily_cups : ℕ := 12

/-- Represents the capacity of Cassie's water bottle in ounces -/
def bottle_capacity : ℕ := 16

/-- Represents the number of ounces in a cup -/
def ounces_per_cup : ℕ := 8

/-- Calculates the number of times Cassie needs to refill her water bottle -/
def refills_needed : ℕ := (daily_cups * ounces_per_cup) / bottle_capacity

theorem cassie_water_refills :
  refills_needed = 6 :=
sorry

end NUMINAMATH_CALUDE_cassie_water_refills_l921_92158


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l921_92132

theorem triangle_angle_calculation (x : ℝ) : 
  x > 0 ∧ 
  40 + 3 * x + x = 180 →
  x = 35 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l921_92132


namespace NUMINAMATH_CALUDE_angle_A_value_range_of_b_squared_plus_c_squared_l921_92130

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  (t.a + t.b) * (Real.sin t.A - Real.sin t.B) = t.c * (Real.sin t.C - Real.sin t.B)

-- Theorem 1: Prove that A = π/3
theorem angle_A_value (t : Triangle) (h : given_condition t) : t.A = π / 3 := by
  sorry

-- Theorem 2: Prove the range of b² + c² when a = 4
theorem range_of_b_squared_plus_c_squared (t : Triangle) (h1 : given_condition t) (h2 : t.a = 4) :
  16 < t.b^2 + t.c^2 ∧ t.b^2 + t.c^2 ≤ 32 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_value_range_of_b_squared_plus_c_squared_l921_92130


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l921_92186

theorem partial_fraction_decomposition (M₁ M₂ : ℚ) : 
  (∀ x : ℚ, x ≠ 1 → x ≠ 3 → (45 * x - 34) / (x^2 - 4*x + 3) = M₁ / (x - 1) + M₂ / (x - 3)) →
  M₁ * M₂ = -1111 / 4 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l921_92186


namespace NUMINAMATH_CALUDE_opposite_of_difference_l921_92192

theorem opposite_of_difference (a b : ℝ) : -(a - b) = b - a := by sorry

end NUMINAMATH_CALUDE_opposite_of_difference_l921_92192


namespace NUMINAMATH_CALUDE_stripe_ordering_exists_l921_92126

/-- A stripe on the rectangular desk -/
structure Stripe where
  direction : Bool  -- True for horizontal, False for vertical

/-- The configuration of stripes on the desk -/
structure StripeConfiguration where
  stripes : Set Stripe
  above : Stripe → Stripe → Prop
  valid_configuration : ∀ (s1 s2 s3 s4 : Stripe),
    s1 ∈ stripes → s2 ∈ stripes → s3 ∈ stripes → s4 ∈ stripes →
    s1.direction = s2.direction →
    s3.direction = s4.direction →
    s1.direction ≠ s3.direction →
    (above s1 s3 ∧ above s1 s4) ∨
    (above s2 s3 ∧ above s2 s4) ∨
    (above s3 s1 ∧ above s3 s2) ∨
    (above s4 s1 ∧ above s4 s2)

/-- The theorem to be proved -/
theorem stripe_ordering_exists (config : StripeConfiguration) :
  ∃ (order : List Stripe),
    (∀ s, s ∈ config.stripes ↔ s ∈ order) ∧
    (∀ i j, i < j → config.above (order.get ⟨j, sorry⟩) (order.get ⟨i, sorry⟩)) :=
  sorry

end NUMINAMATH_CALUDE_stripe_ordering_exists_l921_92126


namespace NUMINAMATH_CALUDE_triangle_construction_with_two_angles_and_perimeter_l921_92135

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define perimeter
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem triangle_construction_with_two_angles_and_perimeter 
  (A B P : ℝ) 
  (h_angles : 0 < A ∧ 0 < B ∧ A + B < π) 
  (h_perimeter : P > 0) :
  ∃ (t : Triangle), 
    t.angleA = A ∧ 
    t.angleB = B ∧ 
    perimeter t = P := by
  sorry

end NUMINAMATH_CALUDE_triangle_construction_with_two_angles_and_perimeter_l921_92135


namespace NUMINAMATH_CALUDE_alcohol_solution_proof_l921_92199

/-- Proves that adding 14.285714285714286 liters of pure alcohol to a 100-liter solution
    results in a 30% alcohol solution if and only if the initial alcohol percentage was 20% -/
theorem alcohol_solution_proof (initial_percentage : ℝ) : 
  (initial_percentage / 100) * 100 + 14.285714285714286 = 0.30 * (100 + 14.285714285714286) ↔ 
  initial_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_alcohol_solution_proof_l921_92199


namespace NUMINAMATH_CALUDE_remaining_shirt_cost_l921_92177

/-- Given a set of shirts with known prices, calculate the price of the remaining shirts -/
theorem remaining_shirt_cost (total_shirts : ℕ) (total_cost : ℚ) (known_shirt_count : ℕ) (known_shirt_cost : ℚ) :
  total_shirts = 5 →
  total_cost = 85 →
  known_shirt_count = 3 →
  known_shirt_cost = 15 →
  (total_cost - (known_shirt_count * known_shirt_cost)) / (total_shirts - known_shirt_count) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_shirt_cost_l921_92177


namespace NUMINAMATH_CALUDE_art_collection_cost_l921_92164

/-- The total cost of John's art collection --/
def total_cost (first_three_cost : ℝ) (fourth_piece_cost : ℝ) : ℝ :=
  first_three_cost + fourth_piece_cost

/-- The cost of the fourth piece of art --/
def fourth_piece_cost (single_piece_cost : ℝ) : ℝ :=
  single_piece_cost * 1.5

theorem art_collection_cost :
  ∀ (single_piece_cost : ℝ),
    single_piece_cost > 0 →
    single_piece_cost * 3 = 45000 →
    total_cost (single_piece_cost * 3) (fourth_piece_cost single_piece_cost) = 67500 := by
  sorry

end NUMINAMATH_CALUDE_art_collection_cost_l921_92164


namespace NUMINAMATH_CALUDE_water_consumption_difference_l921_92116

/-- The yearly water consumption difference between two schools -/
theorem water_consumption_difference 
  (chunlei_daily : ℕ) -- Daily water consumption of Chunlei Central Elementary School
  (days_per_year : ℕ) -- Number of days in a year
  (h1 : chunlei_daily = 111) -- Chunlei's daily consumption is 111 kg
  (h2 : days_per_year = 365) -- A year has 365 days
  : 
  chunlei_daily * days_per_year - (chunlei_daily / 3) * days_per_year = 26910 :=
by sorry

end NUMINAMATH_CALUDE_water_consumption_difference_l921_92116


namespace NUMINAMATH_CALUDE_shorter_leg_length_in_30_60_90_triangle_l921_92165

theorem shorter_leg_length_in_30_60_90_triangle (median_length : ℝ) :
  median_length = 5 * Real.sqrt 3 →
  ∃ (shorter_leg hypotenuse : ℝ),
    shorter_leg = 5 ∧
    hypotenuse = 2 * shorter_leg ∧
    median_length = hypotenuse / 2 :=
by sorry

end NUMINAMATH_CALUDE_shorter_leg_length_in_30_60_90_triangle_l921_92165


namespace NUMINAMATH_CALUDE_harkamal_purchase_l921_92196

/-- The total cost of a purchase given the quantity and price per unit -/
def totalCost (quantity : ℕ) (pricePerUnit : ℕ) : ℕ :=
  quantity * pricePerUnit

theorem harkamal_purchase : 
  let grapeQuantity : ℕ := 8
  let grapePrice : ℕ := 70
  let mangoQuantity : ℕ := 9
  let mangoPrice : ℕ := 60
  totalCost grapeQuantity grapePrice + totalCost mangoQuantity mangoPrice = 1100 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_purchase_l921_92196


namespace NUMINAMATH_CALUDE_fraction_subtraction_l921_92190

theorem fraction_subtraction : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l921_92190


namespace NUMINAMATH_CALUDE_inverse_variation_cube_fourth_l921_92174

/-- Given that a³ varies inversely with b⁴, and a = 2 when b = 4,
    prove that a = 1/∛2 when b = 8 -/
theorem inverse_variation_cube_fourth (a b : ℝ) (k : ℝ) :
  (∀ a b, a^3 * b^4 = k) →  -- a³ varies inversely with b⁴
  (2^3 * 4^4 = k) →         -- a = 2 when b = 4
  (a^3 * 8^4 = k) →         -- condition for b = 8
  a = 1 / (2^(1/3)) :=      -- a = 1/∛2 when b = 8
by sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_fourth_l921_92174


namespace NUMINAMATH_CALUDE_complex_equation_solution_l921_92124

theorem complex_equation_solution (z : ℂ) : z * (2 + I) = 1 + 3 * I → z = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l921_92124


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_a1_l921_92144

/-- An arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) ^ 2 = a n * a (n + 2)

theorem arithmetic_geometric_sequence_a1 (a : ℕ → ℚ) 
  (h_seq : ArithmeticGeometricSequence a) 
  (h_sum : a 1 + a 6 = 11) 
  (h_prod : a 3 * a 4 = 32 / 9) : 
  a 1 = 32 / 3 ∨ a 1 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_a1_l921_92144


namespace NUMINAMATH_CALUDE_function_inequality_l921_92148

/-- Given a continuous function f: ℝ → ℝ such that xf'(x) < 0 for all x in ℝ,
    prove that f(-1) + f(1) < 2f(0). -/
theorem function_inequality (f : ℝ → ℝ) 
    (hf_cont : Continuous f) 
    (hf_deriv : ∀ x : ℝ, x * (deriv f x) < 0) : 
    f (-1) + f 1 < 2 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l921_92148


namespace NUMINAMATH_CALUDE_f_properties_l921_92111

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - x - 1/a) * Real.exp (a * x)

theorem f_properties (h : a ≠ 0) :
  -- Part I
  (a = 1/2 → (f a x = 0 ↔ x = -1 ∨ x = 2)) ∧
  -- Part II
  (∀ x, f a x = 0 → x = 1 ∨ x = -2/a) ∧
  -- Part III
  (a > 0 → (∀ x, f a x + 2/a ≥ 0) ↔ 0 < a ∧ a ≤ Real.log 2) :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l921_92111


namespace NUMINAMATH_CALUDE_system_solution_l921_92139

theorem system_solution (x y b : ℝ) : 
  (4 * x + 2 * y = b) → 
  (3 * x + 7 * y = 3 * b) → 
  (x = 3) → 
  (b = 66) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l921_92139


namespace NUMINAMATH_CALUDE_polynomial_perfect_square_count_l921_92151

def p (x : ℤ) : ℤ := 4*x^4 - 12*x^3 + 17*x^2 - 6*x - 14

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m^2

theorem polynomial_perfect_square_count :
  ∃! (S : Finset ℤ), S.card = 2 ∧ ∀ x : ℤ, x ∈ S ↔ is_perfect_square (p x) :=
sorry

end NUMINAMATH_CALUDE_polynomial_perfect_square_count_l921_92151


namespace NUMINAMATH_CALUDE_value_of_a_l921_92162

theorem value_of_a (a b : ℝ) (h1 : |a| = 5) (h2 : b = 4) (h3 : a < b) : a = -5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l921_92162


namespace NUMINAMATH_CALUDE_irrational_equation_root_l921_92108

theorem irrational_equation_root (m : ℝ) : 
  (∃ x : ℝ, x = 1 ∧ Real.sqrt (2 * x + m) = x) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_irrational_equation_root_l921_92108


namespace NUMINAMATH_CALUDE_xy_sum_theorem_l921_92115

theorem xy_sum_theorem (x y : ℤ) (h : 2*x*y + x + y = 83) : 
  x + y = 83 ∨ x + y = -85 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_theorem_l921_92115


namespace NUMINAMATH_CALUDE_correct_answers_count_l921_92133

/-- Represents a mathematics contest with scoring rules and results. -/
structure MathContest where
  total_problems : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  answered_problems : ℕ
  total_score : ℤ

/-- Theorem stating that given the contest conditions, 11 correct answers result in a score of 54. -/
theorem correct_answers_count (contest : MathContest) 
  (h1 : contest.total_problems = 15)
  (h2 : contest.correct_points = 6)
  (h3 : contest.incorrect_points = -3)
  (h4 : contest.answered_problems = contest.total_problems)
  (h5 : contest.total_score = 54) :
  ∃ (correct : ℕ), correct = 11 ∧ 
    contest.correct_points * correct + contest.incorrect_points * (contest.total_problems - correct) = contest.total_score :=
by sorry

end NUMINAMATH_CALUDE_correct_answers_count_l921_92133


namespace NUMINAMATH_CALUDE_correct_algebraic_simplification_l921_92138

theorem correct_algebraic_simplification (x y : ℝ) : 3 * x^2 * y - 8 * y * x^2 = -5 * x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_algebraic_simplification_l921_92138


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l921_92175

theorem root_sum_reciprocal (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 2 = 0 → 
  x₂^2 - 3*x₂ + 2 = 0 → 
  x₁ ≠ x₂ →
  (1/x₁) + (1/x₂) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l921_92175


namespace NUMINAMATH_CALUDE_friends_receiving_pebbles_l921_92159

def pebbles_per_dozen : ℕ := 12

theorem friends_receiving_pebbles 
  (total_dozens : ℕ) 
  (pebbles_per_friend : ℕ) 
  (h1 : total_dozens = 3) 
  (h2 : pebbles_per_friend = 4) : 
  (total_dozens * pebbles_per_dozen) / pebbles_per_friend = 9 := by
  sorry

end NUMINAMATH_CALUDE_friends_receiving_pebbles_l921_92159


namespace NUMINAMATH_CALUDE_sum_pairwise_ratios_lower_bound_l921_92102

theorem sum_pairwise_ratios_lower_bound {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_pairwise_ratios_lower_bound_l921_92102


namespace NUMINAMATH_CALUDE_quadratic_minimum_l921_92104

/-- The quadratic function f(x) = x^2 - 8x + 18 -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 18

theorem quadratic_minimum :
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min) ∧
  (∃ (x_min : ℝ), f x_min = 2) ∧
  (∀ (x : ℝ), f x = 2 → x = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l921_92104


namespace NUMINAMATH_CALUDE_only_solution_is_two_l921_92123

theorem only_solution_is_two : 
  ∀ n : ℕ, n > 0 → ((n + 1) ∣ (2 * n^2 + 5 * n)) ↔ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_only_solution_is_two_l921_92123


namespace NUMINAMATH_CALUDE_some_number_value_l921_92110

theorem some_number_value (x : ℝ) (some_number : ℝ) 
  (h1 : (27 / 4) * x - some_number = 3 * x + 27) 
  (h2 : x = 12) : 
  some_number = 18 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l921_92110


namespace NUMINAMATH_CALUDE_abc_inequality_l921_92183

theorem abc_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (h : a^2 + b^2 + c^2 + a*b*c = 4) :
  0 ≤ a*b + b*c + c*a - a*b*c ∧ a*b + b*c + c*a - a*b*c ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l921_92183


namespace NUMINAMATH_CALUDE_problems_left_to_grade_l921_92140

/-- Given the number of total worksheets, graded worksheets, and problems per worksheet,
    calculate the number of problems left to grade. -/
theorem problems_left_to_grade 
  (total_worksheets : ℕ) 
  (graded_worksheets : ℕ) 
  (problems_per_worksheet : ℕ) 
  (h1 : total_worksheets = 15)
  (h2 : graded_worksheets = 7)
  (h3 : problems_per_worksheet = 3)
  (h4 : graded_worksheets ≤ total_worksheets) :
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 24 :=
by sorry

end NUMINAMATH_CALUDE_problems_left_to_grade_l921_92140


namespace NUMINAMATH_CALUDE_locus_and_fixed_point_l921_92109

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the locus C
def C : Set (ℝ × ℝ) := {p | p.1^2/4 - p.2^2 = 1 ∧ p.1 ≠ 2 ∧ p.1 ≠ -2}

-- Define the line x = 1
def line_x_eq_1 : Set (ℝ × ℝ) := {p | p.1 = 1}

-- Define the property of point M
def is_valid_M (M : ℝ × ℝ) : Prop :=
  let slope_AM := (M.2 - A.2) / (M.1 - A.1)
  let slope_BM := (M.2 - B.2) / (M.1 - B.1)
  slope_AM * slope_BM = 1/4

-- Main theorem
theorem locus_and_fixed_point :
  (∀ M, is_valid_M M → M ∈ C) ∧
  (∀ T ∈ line_x_eq_1, 
    ∃ P Q, P ∈ C ∧ Q ∈ C ∧ 
    (P.2 - A.2) / (P.1 - A.1) = (T.2 - A.2) / (T.1 - A.1) ∧
    (Q.2 - B.2) / (Q.1 - B.1) = (T.2 - B.2) / (T.1 - B.1) ∧
    (Q.2 - P.2) / (Q.1 - P.1) = (0 - P.2) / (4 - P.1)) :=
sorry

end NUMINAMATH_CALUDE_locus_and_fixed_point_l921_92109


namespace NUMINAMATH_CALUDE_quadruple_equation_solutions_l921_92117

theorem quadruple_equation_solutions :
  ∀ (a b c d : ℝ),
  (b + c + d)^2010 = 3 * a ∧
  (a + c + d)^2010 = 3 * b ∧
  (a + b + d)^2010 = 3 * c ∧
  (a + b + c)^2010 = 3 * d →
  ((a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨
   (a = 1/3 ∧ b = 1/3 ∧ c = 1/3 ∧ d = 1/3)) :=
by sorry

end NUMINAMATH_CALUDE_quadruple_equation_solutions_l921_92117


namespace NUMINAMATH_CALUDE_xiaohong_mother_age_l921_92150

/-- Xiaohong's age when her mother was her current age -/
def xiaohong_age_then : ℕ := 3

/-- Xiaohong's mother's future age when Xiaohong will be her mother's current age -/
def mother_age_future : ℕ := 78

/-- The age difference between Xiaohong and her mother -/
def age_difference : ℕ := mother_age_future - xiaohong_age_then

/-- Xiaohong's current age -/
def xiaohong_age_now : ℕ := age_difference + xiaohong_age_then

/-- Xiaohong's mother's current age -/
def mother_age_now : ℕ := mother_age_future - age_difference

theorem xiaohong_mother_age : mother_age_now = 53 := by
  sorry

#eval mother_age_now

end NUMINAMATH_CALUDE_xiaohong_mother_age_l921_92150


namespace NUMINAMATH_CALUDE_sara_apples_l921_92193

theorem sara_apples (total : ℕ) (ali_ratio : ℕ) (sara_apples : ℕ) : 
  total = 80 →
  ali_ratio = 4 →
  total = sara_apples + ali_ratio * sara_apples →
  sara_apples = 16 := by
  sorry

end NUMINAMATH_CALUDE_sara_apples_l921_92193


namespace NUMINAMATH_CALUDE_frustum_small_cone_height_l921_92103

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  height : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

/-- Calculates the height of the small cone removed from a frustum -/
def small_cone_height (f : Frustum) : ℝ :=
  f.height

/-- Theorem: The height of the small cone removed from a frustum with given dimensions is 30 cm -/
theorem frustum_small_cone_height :
  ∀ (f : Frustum),
    f.height = 30 ∧
    f.lower_base_area = 400 * Real.pi ∧
    f.upper_base_area = 100 * Real.pi →
    small_cone_height f = 30 := by
  sorry

end NUMINAMATH_CALUDE_frustum_small_cone_height_l921_92103


namespace NUMINAMATH_CALUDE_sweater_fraction_is_one_fourth_l921_92173

/-- The amount Leila spent on the sweater -/
def sweater_cost : ℕ := 40

/-- The amount Leila had left after buying jewelry -/
def remaining_money : ℕ := 20

/-- The additional amount Leila spent on jewelry compared to the sweater -/
def jewelry_additional_cost : ℕ := 60

/-- Leila's total initial money -/
def total_money : ℕ := sweater_cost + remaining_money + sweater_cost + jewelry_additional_cost

/-- The fraction of total money spent on the sweater -/
def sweater_fraction : ℚ := sweater_cost / total_money

theorem sweater_fraction_is_one_fourth : sweater_fraction = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sweater_fraction_is_one_fourth_l921_92173


namespace NUMINAMATH_CALUDE_anne_cleaning_time_l921_92141

-- Define the cleaning rates
def bruce_rate : ℝ := sorry
def anne_rate : ℝ := sorry

-- Define the conditions
axiom combined_rate : bruce_rate + anne_rate = 1 / 4
axiom doubled_anne_rate : bruce_rate + 2 * anne_rate = 1 / 3

-- Theorem to prove
theorem anne_cleaning_time :
  1 / anne_rate = 12 := by sorry

end NUMINAMATH_CALUDE_anne_cleaning_time_l921_92141


namespace NUMINAMATH_CALUDE_days_to_complete_correct_l921_92136

/-- The number of days required for a given number of men to complete a work,
    given that 12 men can do it in 80 days and 16 men can do it in 60 days. -/
def days_to_complete (num_men : ℕ) : ℚ :=
  960 / num_men

/-- Theorem stating that the number of days required for any number of men
    to complete the work is correctly given by the days_to_complete function,
    based on the given conditions. -/
theorem days_to_complete_correct (num_men : ℕ) (num_men_pos : 0 < num_men) :
  days_to_complete num_men * num_men = 960 ∧
  days_to_complete 12 = 80 ∧
  days_to_complete 16 = 60 :=
by sorry

end NUMINAMATH_CALUDE_days_to_complete_correct_l921_92136


namespace NUMINAMATH_CALUDE_complete_square_with_integer_l921_92118

theorem complete_square_with_integer (y : ℝ) : 
  ∃ (k : ℤ) (a : ℝ), y^2 + 12*y + 44 = (y + a)^2 + k ∧ k = 8 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_with_integer_l921_92118


namespace NUMINAMATH_CALUDE_problem_solving_probability_l921_92100

theorem problem_solving_probability 
  (arthur_prob : ℚ) 
  (bella_prob : ℚ) 
  (xavier_prob : ℚ) 
  (yvonne_prob : ℚ) 
  (zelda_prob : ℚ) 
  (h_arthur : arthur_prob = 1/4)
  (h_bella : bella_prob = 3/10)
  (h_xavier : xavier_prob = 1/6)
  (h_yvonne : yvonne_prob = 1/2)
  (h_zelda : zelda_prob = 5/8)
  (h_independent : True)  -- Assumption of independence
  : arthur_prob * bella_prob * xavier_prob * yvonne_prob * (1 - zelda_prob) = 9/3840 :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l921_92100


namespace NUMINAMATH_CALUDE_emma_popsicle_production_l921_92119

/-- Emma's popsicle production problem -/
theorem emma_popsicle_production 
  (p h : ℝ) 
  (h_positive : h > 0)
  (p_def : p = 3/2 * h) :
  p * h - (p + 2) * (h - 3) = 7/2 * h + 6 := by
  sorry

end NUMINAMATH_CALUDE_emma_popsicle_production_l921_92119


namespace NUMINAMATH_CALUDE_largest_integer_b_for_all_real_domain_l921_92153

theorem largest_integer_b_for_all_real_domain : 
  ∀ b : ℤ, (∀ x : ℝ, x^2 + b*x + 12 ≠ 0) → b ≤ 6 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_b_for_all_real_domain_l921_92153


namespace NUMINAMATH_CALUDE_henrys_money_l921_92176

theorem henrys_money (initial_amount : ℕ) (birthday_gift : ℕ) (game_cost : ℕ) : 
  initial_amount = 11 → birthday_gift = 18 → game_cost = 10 →
  initial_amount + birthday_gift - game_cost = 19 := by
  sorry

end NUMINAMATH_CALUDE_henrys_money_l921_92176


namespace NUMINAMATH_CALUDE_angle4_is_60_l921_92157

/-- Represents a quadrilateral with specific angle properties -/
structure SpecialQuadrilateral where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  angle5 : ℝ
  sum_property : angle1 + angle2 + angle3 = 360
  equal_angles : angle3 = angle4 ∧ angle3 = angle5
  angle1_value : angle1 = 100
  angle2_value : angle2 = 80

/-- Theorem: In a SpecialQuadrilateral, angle4 equals 60 degrees -/
theorem angle4_is_60 (q : SpecialQuadrilateral) : q.angle4 = 60 := by
  sorry


end NUMINAMATH_CALUDE_angle4_is_60_l921_92157


namespace NUMINAMATH_CALUDE_root_product_plus_one_l921_92156

theorem root_product_plus_one (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 10 = 0) →
  (b^3 - 15*b^2 + 25*b - 10 = 0) →
  (c^3 - 15*c^2 + 25*c - 10 = 0) →
  (1 + a) * (1 + b) * (1 + c) = 51 := by
sorry

end NUMINAMATH_CALUDE_root_product_plus_one_l921_92156


namespace NUMINAMATH_CALUDE_overtime_rate_increase_l921_92167

def regular_rate : ℚ := 16
def regular_hours : ℕ := 40
def total_compensation : ℚ := 1340
def total_hours : ℕ := 65

theorem overtime_rate_increase :
  let overtime_hours : ℕ := total_hours - regular_hours
  let regular_pay : ℚ := regular_rate * regular_hours
  let overtime_pay : ℚ := total_compensation - regular_pay
  let overtime_rate : ℚ := overtime_pay / overtime_hours
  let rate_increase : ℚ := (overtime_rate - regular_rate) / regular_rate
  rate_increase = 3/4 := by sorry

end NUMINAMATH_CALUDE_overtime_rate_increase_l921_92167


namespace NUMINAMATH_CALUDE_base_16_to_binary_bits_l921_92180

/-- The base-16 number represented as 66666 --/
def base_16_num : ℕ := 6 * 16^4 + 6 * 16^3 + 6 * 16^2 + 6 * 16 + 6

/-- The number of bits in the binary representation of a natural number --/
def num_bits (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.log2 n + 1

theorem base_16_to_binary_bits :
  num_bits base_16_num = 19 := by
  sorry

end NUMINAMATH_CALUDE_base_16_to_binary_bits_l921_92180


namespace NUMINAMATH_CALUDE_point_on_line_l921_92107

/-- Given points A and B in the Cartesian plane, if point C satisfies the vector equation,
    then C lies on the line passing through A and B. -/
theorem point_on_line (A B C : ℝ × ℝ) (α β : ℝ) :
  A = (3, 1) →
  B = (-1, 3) →
  α + β = 1 →
  C = (α * A.1 + β * B.1, α * A.2 + β * B.2) →
  C.1 + 2 * C.2 - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l921_92107


namespace NUMINAMATH_CALUDE_set_equality_l921_92152

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set ℕ := {1, 3, 5}

-- Define set B
def B : Set ℕ := {3, 5}

-- Theorem statement
theorem set_equality : U = A ∪ (U \ B) := by sorry

end NUMINAMATH_CALUDE_set_equality_l921_92152


namespace NUMINAMATH_CALUDE_bucket_weight_l921_92114

/-- 
Given:
- p: weight when bucket is three-quarters full
- q: weight when bucket is one-third full
- r: weight of empty bucket
Prove: weight of full bucket is (4p - r) / 3
-/
theorem bucket_weight (p q r : ℝ) : ℝ :=
  let three_quarters_full := p
  let one_third_full := q
  let empty_bucket := r
  let full_bucket := (4 * p - r) / 3
  full_bucket

#check bucket_weight

end NUMINAMATH_CALUDE_bucket_weight_l921_92114


namespace NUMINAMATH_CALUDE_value_of_x_l921_92169

theorem value_of_x :
  ∀ (x a b c d : ℤ),
    x = a + 7 →
    a = b + 9 →
    b = c + 15 →
    c = d + 25 →
    d = 60 →
    x = 116 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l921_92169


namespace NUMINAMATH_CALUDE_robin_water_consumption_l921_92191

theorem robin_water_consumption 
  (morning : ℝ) 
  (afternoon : ℝ) 
  (evening : ℝ) 
  (night : ℝ) 
  (m : ℝ) 
  (e : ℝ) 
  (t : ℝ) 
  (h1 : morning = 7.5) 
  (h2 : afternoon = 9.25) 
  (h3 : evening = 5.75) 
  (h4 : night = 3.5) 
  (h5 : m = morning + afternoon) 
  (h6 : e = evening + night) 
  (h7 : t = m + e) : 
  t = 16.75 + 9.25 := by
  sorry

end NUMINAMATH_CALUDE_robin_water_consumption_l921_92191


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l921_92113

theorem fixed_point_on_line (a b : ℝ) (h : a + 2 * b = 1) :
  a * (1/2) + 3 * (-1/6) + b = 0 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l921_92113


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l921_92187

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : Nat
  totalRuns : Nat
  averageBeforeLastInning : Rat
  lastInningScore : Nat
  averageIncrease : Rat

/-- Calculates the new average after the last inning -/
def newAverage (stats : BatsmanStats) : Rat :=
  (stats.totalRuns + stats.lastInningScore) / stats.innings

/-- Theorem: Given the conditions, prove that the new average is 23 -/
theorem batsman_average_theorem (stats : BatsmanStats) 
  (h1 : stats.innings = 17)
  (h2 : stats.lastInningScore = 87)
  (h3 : stats.averageIncrease = 4)
  (h4 : newAverage stats = stats.averageBeforeLastInning + stats.averageIncrease) :
  newAverage stats = 23 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_theorem_l921_92187


namespace NUMINAMATH_CALUDE_quadratic_inequality_l921_92149

theorem quadratic_inequality (x : ℝ) : 2 * x^2 - 6 * x - 56 > 0 ↔ x < -4 ∨ x > 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l921_92149


namespace NUMINAMATH_CALUDE_balls_distribution_theorem_l921_92163

def distribute_balls (n : ℕ) (k : ℕ) (min : ℕ) (max : ℕ) : ℕ :=
  sorry

theorem balls_distribution_theorem :
  distribute_balls 6 2 1 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_balls_distribution_theorem_l921_92163


namespace NUMINAMATH_CALUDE_fraction_equality_l921_92122

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 8)
  (h2 : s / r = 4)
  (h3 : s / t = 1 / 3) :
  t / q = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l921_92122


namespace NUMINAMATH_CALUDE_gcd_triple_characterization_l921_92194

theorem gcd_triple_characterization (a b c : ℕ+) :
  (Nat.gcd a 20 = b) →
  (Nat.gcd b 15 = c) →
  (Nat.gcd a c = 5) →
  (∃ t : ℕ+, (a = 20 * t ∧ b = 20 ∧ c = 5) ∨
             (a = 20 * t - 10 ∧ b = 10 ∧ c = 5) ∨
             (a = 10 * t - 5 ∧ b = 5 ∧ c = 5)) :=
by sorry

end NUMINAMATH_CALUDE_gcd_triple_characterization_l921_92194


namespace NUMINAMATH_CALUDE_percent_calculation_l921_92101

theorem percent_calculation (x : ℝ) (h : 0.20 * x = 200) : 1.20 * x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_percent_calculation_l921_92101


namespace NUMINAMATH_CALUDE_problem_solution_l921_92121

theorem problem_solution (m n t : ℝ) 
  (h1 : 2^m = t)
  (h2 : 5^n = t)
  (h3 : t > 0)
  (h4 : t ≠ 1)
  (h5 : 1/m + 1/n = 3) :
  t = (10 : ℝ)^(1/3) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l921_92121


namespace NUMINAMATH_CALUDE_circle_area_circumscribed_square_l921_92181

theorem circle_area_circumscribed_square (s : ℝ) (h : s = 12) :
  let r := s * Real.sqrt 2 / 2
  π * r^2 = 72 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_circumscribed_square_l921_92181


namespace NUMINAMATH_CALUDE_bananas_count_l921_92112

/-- Represents the contents of a fruit bowl -/
structure FruitBowl where
  apples : ℕ
  pears : ℕ
  bananas : ℕ

/-- Conditions for the fruit bowl problem -/
def fruitBowlConditions (bowl : FruitBowl) : Prop :=
  bowl.pears = bowl.apples + 2 ∧
  bowl.bananas = bowl.pears + 3 ∧
  bowl.apples + bowl.pears + bowl.bananas = 19

/-- Theorem stating that under the given conditions, the number of bananas is 9 -/
theorem bananas_count (bowl : FruitBowl) : 
  fruitBowlConditions bowl → bowl.bananas = 9 := by
  sorry


end NUMINAMATH_CALUDE_bananas_count_l921_92112


namespace NUMINAMATH_CALUDE_sets_A_and_B_proof_l921_92171

def U : Set Nat := {x | x ≤ 20 ∧ Nat.Prime x}

theorem sets_A_and_B_proof (A B : Set Nat) 
  (h1 : A ∩ (U \ B) = {3, 5})
  (h2 : B ∩ (U \ A) = {7, 19})
  (h3 : (U \ A) ∩ (U \ B) = {2, 17}) :
  A = {3, 5, 11, 13} ∧ B = {7, 11, 13, 19} := by
sorry

end NUMINAMATH_CALUDE_sets_A_and_B_proof_l921_92171


namespace NUMINAMATH_CALUDE_m_properties_l921_92142

/-- The smallest positive integer with both 5 and 6 as digits, each appearing at least once, and divisible by both 3 and 7 -/
def m : ℕ := 5665665660

/-- Checks if a natural number contains both 5 and 6 as digits -/
def has_five_and_six (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a * 10 + 5 ∧ ∃ (c d : ℕ), n = c * 10 + 6

/-- Returns the last four digits of a natural number -/
def last_four_digits (n : ℕ) : ℕ :=
  n % 10000

theorem m_properties :
  has_five_and_six m ∧ 
  m % 3 = 0 ∧ 
  m % 7 = 0 ∧ 
  ∀ k < m, ¬(has_five_and_six k ∧ k % 3 = 0 ∧ k % 7 = 0) ∧
  last_four_digits m = 5660 :=
sorry

end NUMINAMATH_CALUDE_m_properties_l921_92142


namespace NUMINAMATH_CALUDE_twentyFirstTerm_l921_92185

/-- The nth term of an arithmetic progression -/
def arithmeticProgressionTerm (a d n : ℕ) : ℕ :=
  a + (n - 1) * d

/-- Theorem: The 21st term of an arithmetic progression with first term 3 and common difference 5 is 103 -/
theorem twentyFirstTerm :
  arithmeticProgressionTerm 3 5 21 = 103 := by
  sorry

end NUMINAMATH_CALUDE_twentyFirstTerm_l921_92185


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequences_l921_92179

theorem arithmetic_geometric_sequences :
  -- Arithmetic sequence
  ∃ (a : ℕ → ℝ) (S : ℕ → ℝ),
    (a 8 = 6 ∧ a 10 = 0) →
    (∀ n, a n = 30 - 3 * n) ∧
    (∀ n, S n = -3/2 * n^2 + 57/2 * n) ∧
    (∀ n, n ≠ 9 ∧ n ≠ 10 → S n < S 9) ∧
  -- Geometric sequence
  ∃ (b : ℕ → ℝ) (T : ℕ → ℝ),
    (b 1 = 1/2 ∧ b 4 = 4) →
    (∀ n, b n = 2^(n-2)) ∧
    (∀ n, T n = 2^(n-1) - 1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequences_l921_92179


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l921_92178

theorem arithmetic_square_root_of_four :
  Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l921_92178


namespace NUMINAMATH_CALUDE_inequality_equivalence_l921_92182

theorem inequality_equivalence (x : ℝ) :
  (x - 3) / (2 - x) ≥ 0 ↔ (3 - x) / (x - 2) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l921_92182


namespace NUMINAMATH_CALUDE_equation_solution_l921_92137

theorem equation_solution : ∃! x : ℝ, (10 : ℝ)^(2*x) * (100 : ℝ)^x = (1000 : ℝ)^4 :=
  by
    use 3
    constructor
    · -- Proof that x = 3 satisfies the equation
      sorry
    · -- Proof of uniqueness
      sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l921_92137


namespace NUMINAMATH_CALUDE_inequality_proof_l921_92154

theorem inequality_proof (a b : ℝ) (h1 : a ≥ b) (h2 : b ≥ 0) :
  Real.sqrt (a^2 + b^2) + (a^3 + b^3)^(1/3) + (a^4 + b^4)^(1/4) ≤ 3*a + b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l921_92154


namespace NUMINAMATH_CALUDE_power_sum_problem_l921_92197

theorem power_sum_problem (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 25)
  (h4 : a * x^4 + b * y^4 = 59) :
  a * x^5 + b * y^5 = 145 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_problem_l921_92197


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l921_92131

/-- Given two circular pulleys with an uncrossed belt, prove the distance between their centers. -/
theorem pulley_centers_distance (r₁ r₂ contact_distance : ℝ) 
  (h₁ : r₁ = 14)
  (h₂ : r₂ = 4)
  (h₃ : contact_distance = 24) :
  Real.sqrt ((r₁ - r₂)^2 + contact_distance^2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l921_92131


namespace NUMINAMATH_CALUDE_f_not_mapping_l921_92188

-- Define set A
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}

-- Define set B
def B : Set ℝ := {y : ℝ | 0 ≤ y ∧ y ≤ 8}

-- Define the correspondence rule f
def f (x : ℝ) : ℝ := 4

-- Theorem stating that f is not a mapping from A to B
theorem f_not_mapping : ¬(∀ x ∈ A, f x ∈ B) :=
sorry

end NUMINAMATH_CALUDE_f_not_mapping_l921_92188


namespace NUMINAMATH_CALUDE_school_teachers_count_l921_92172

theorem school_teachers_count (total : ℕ) (sample_size : ℕ) (sampled_students : ℕ) :
  total = 2400 →
  sample_size = 160 →
  sampled_students = 150 →
  total - (total * sampled_students / sample_size) = 150 :=
by sorry

end NUMINAMATH_CALUDE_school_teachers_count_l921_92172


namespace NUMINAMATH_CALUDE_christines_dog_weight_l921_92143

/-- The weight of Christine's dog given the weights of her two cats -/
def dogs_weight (cat1_weight cat2_weight : ℕ) : ℕ :=
  2 * (cat1_weight + cat2_weight)

/-- Theorem stating that Christine's dog weighs 34 pounds -/
theorem christines_dog_weight :
  dogs_weight 7 10 = 34 := by
  sorry

end NUMINAMATH_CALUDE_christines_dog_weight_l921_92143
