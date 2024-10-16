import Mathlib

namespace NUMINAMATH_CALUDE_kims_candy_bars_l4156_415678

/-- Calculates the number of weeks passed given the number of candy bars saved, 
    the number of candy bars received per week, and the number of weeks between eating candy bars. -/
def weeks_passed (candy_bars_saved : ℕ) (candy_bars_per_week : ℕ) (weeks_between_eating : ℕ) : ℕ :=
  let candy_bars_saved_per_cycle := candy_bars_per_week * weeks_between_eating - 1
  candy_bars_saved / candy_bars_saved_per_cycle * weeks_between_eating

/-- Theorem stating that given the conditions from Kim's candy bar problem, 
    the number of weeks passed is 16. -/
theorem kims_candy_bars : 
  weeks_passed 28 2 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_kims_candy_bars_l4156_415678


namespace NUMINAMATH_CALUDE_midpoint_xy_product_l4156_415623

/-- Given that C = (3, 5) is the midpoint of AB, where A = (1, 8) and B = (x, y), prove that xy = 10 -/
theorem midpoint_xy_product (x y : ℝ) : 
  (3 : ℝ) = (1 + x) / 2 ∧ (5 : ℝ) = (8 + y) / 2 → x * y = 10 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_xy_product_l4156_415623


namespace NUMINAMATH_CALUDE_sum_due_proof_l4156_415663

/-- Banker's discount (BD) is the simple interest on the face value (FV) of a bill for the unexpired time -/
def bankers_discount (face_value : ℝ) : ℝ := 288

/-- True discount (TD) is the simple interest on the present value (PV) of the bill for the unexpired time -/
def true_discount (face_value : ℝ) : ℝ := 240

/-- The relationship between banker's discount, true discount, and face value -/
def discount_relationship (face_value : ℝ) : Prop :=
  bankers_discount face_value = true_discount face_value + (true_discount face_value)^2 / face_value

theorem sum_due_proof : 
  ∃ (face_value : ℝ), face_value = 1200 ∧ discount_relationship face_value := by
  sorry

end NUMINAMATH_CALUDE_sum_due_proof_l4156_415663


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l4156_415646

def n : ℕ := 5
def k : ℕ := 2
def p : ℚ := 2/5

theorem magic_8_ball_probability :
  (n.choose k) * p^k * (1 - p)^(n - k) = 216/625 := by
  sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l4156_415646


namespace NUMINAMATH_CALUDE_system_solution_l4156_415615

theorem system_solution :
  ∃! (x y : ℝ), 
    x^2 - 4 * Real.sqrt (3 * x - 2) + 10 = 2 * y ∧
    y^2 - 6 * Real.sqrt (4 * y - 3) + 11 = x ∧
    x = 2 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4156_415615


namespace NUMINAMATH_CALUDE_saltwater_aquariums_l4156_415636

theorem saltwater_aquariums (total_saltwater_animals : ℕ) (animals_per_aquarium : ℕ) 
  (h1 : total_saltwater_animals = 1012)
  (h2 : animals_per_aquarium = 46) :
  total_saltwater_animals / animals_per_aquarium = 22 := by
  sorry

end NUMINAMATH_CALUDE_saltwater_aquariums_l4156_415636


namespace NUMINAMATH_CALUDE_family_admission_price_l4156_415643

/-- The total price for a family's admission to an amusement park --/
def total_price (adult_price child_price : ℕ) (num_adults num_children : ℕ) : ℕ :=
  adult_price * num_adults + child_price * num_children

/-- Theorem: The total price for a family of 2 adults and 2 children,
    with adult tickets costing $22 and child tickets costing $7, is $58 --/
theorem family_admission_price :
  total_price 22 7 2 2 = 58 := by
  sorry

end NUMINAMATH_CALUDE_family_admission_price_l4156_415643


namespace NUMINAMATH_CALUDE_green_peaches_count_l4156_415627

/-- Given a basket of peaches with a total of 10 peaches and 4 red peaches,
    prove that there are 6 green peaches in the basket. -/
theorem green_peaches_count (total_peaches : ℕ) (red_peaches : ℕ) (baskets : ℕ) :
  total_peaches = 10 → red_peaches = 4 → baskets = 1 →
  total_peaches - red_peaches = 6 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l4156_415627


namespace NUMINAMATH_CALUDE_fraction_equality_l4156_415679

theorem fraction_equality (x y : ℝ) (h : (1/x + 1/y) / (1/x + 2/y) = 4) :
  (x + y) / (x + 2*y) = 4/11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4156_415679


namespace NUMINAMATH_CALUDE_simplify_expression_l4156_415600

theorem simplify_expression (a b : ℝ) : 5 * a + 2 * b + (a - 3 * b) = 6 * a - b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4156_415600


namespace NUMINAMATH_CALUDE_max_tan_MPN_l4156_415624

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4/25
def C2 (x y θ : ℝ) : Prop := (x - 3 - Real.cos θ)^2 + (y - Real.sin θ)^2 = 1/25

-- Define a point P on C2
def P_on_C2 (x y θ : ℝ) : Prop := C2 x y θ

-- Define tangent points M and N on C1
def tangent_points (xm ym xn yn : ℝ) : Prop := C1 xm ym ∧ C1 xn yn

-- Define the angle MPN
def angle_MPN (xp yp xm ym xn yn : ℝ) : ℝ := sorry

-- Theorem statement
theorem max_tan_MPN :
  ∃ (xp yp θ xm ym xn yn : ℝ),
    P_on_C2 xp yp θ ∧
    tangent_points xm ym xn yn ∧
    (∀ (xp' yp' θ' xm' ym' xn' yn' : ℝ),
      P_on_C2 xp' yp' θ' →
      tangent_points xm' ym' xn' yn' →
      Real.tan (angle_MPN xp yp xm ym xn yn) ≥ Real.tan (angle_MPN xp' yp' xm' ym' xn' yn')) ∧
    Real.tan (angle_MPN xp yp xm ym xn yn) = 4 * Real.sqrt 2 / 7 :=
sorry

end NUMINAMATH_CALUDE_max_tan_MPN_l4156_415624


namespace NUMINAMATH_CALUDE_probability_of_negative_product_l4156_415669

def set_m : Finset Int := {-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4}
def set_t : Finset Int := {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7}

def negative_product_pairs : Finset (Int × Int) :=
  (set_m.filter (λ x => x < 0) ×ˢ set_t.filter (λ y => y > 0)) ∪
  (set_m.filter (λ x => x > 0) ×ˢ set_t.filter (λ y => y < 0))

theorem probability_of_negative_product :
  (negative_product_pairs.card : ℚ) / ((set_m.card * set_t.card) : ℚ) = 65 / 144 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_negative_product_l4156_415669


namespace NUMINAMATH_CALUDE_circle_division_l4156_415642

theorem circle_division (OA : ℝ) (OA_pos : OA > 0) :
  ∃ (OC OB : ℝ),
    OC = (OA * Real.sqrt 3) / 3 ∧
    OB = (OA * Real.sqrt 6) / 3 ∧
    π * OC^2 = π * (OB^2 - OC^2) ∧
    π * (OB^2 - OC^2) = π * (OA^2 - OB^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_division_l4156_415642


namespace NUMINAMATH_CALUDE_seed_germination_problem_l4156_415607

theorem seed_germination_problem (x : ℝ) : 
  x > 0 ∧ 
  0.15 * x + 0.35 * 200 = 0.23 * (x + 200) → 
  x = 300 := by
sorry

end NUMINAMATH_CALUDE_seed_germination_problem_l4156_415607


namespace NUMINAMATH_CALUDE_math_books_count_l4156_415670

theorem math_books_count (total_books : ℕ) (math_cost history_cost total_price : ℚ)
  (h1 : total_books = 80)
  (h2 : math_cost = 4)
  (h3 : history_cost = 5)
  (h4 : total_price = 390) :
  ∃ (math_books : ℕ),
    math_books * math_cost + (total_books - math_books) * history_cost = total_price ∧
    math_books = 10 := by
  sorry

end NUMINAMATH_CALUDE_math_books_count_l4156_415670


namespace NUMINAMATH_CALUDE_sum_c_d_equals_five_l4156_415603

theorem sum_c_d_equals_five (a b c d : ℝ) 
  (h1 : a + b = 4)
  (h2 : b + c = 7)
  (h3 : a + d = 2) :
  c + d = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_c_d_equals_five_l4156_415603


namespace NUMINAMATH_CALUDE_max_value_of_expression_l4156_415652

theorem max_value_of_expression (x y z : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) → 
  (10 ≤ y ∧ y ≤ 99) → 
  (10 ≤ z ∧ z ≤ 99) → 
  ((x + y + z) / 3 = 60) → 
  ((x + y) / z ≤ 17) ∧ (∃ x' y' z' : ℕ, (10 ≤ x' ∧ x' ≤ 99) ∧ (10 ≤ y' ∧ y' ≤ 99) ∧ (10 ≤ z' ∧ z' ≤ 99) ∧ ((x' + y' + z') / 3 = 60) ∧ ((x' + y') / z' = 17)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l4156_415652


namespace NUMINAMATH_CALUDE_walnut_chestnut_cost_l4156_415695

/-- The total cost of buying walnuts and chestnuts -/
def total_cost (m n : ℝ) : ℝ :=
  2 * m + 3 * n

/-- Theorem: The total cost of buying 2 kg of walnuts at m yuan/kg and 3 kg of chestnuts at n yuan/kg is (2m + 3n) yuan -/
theorem walnut_chestnut_cost (m n : ℝ) :
  total_cost m n = 2 * m + 3 * n :=
by sorry

end NUMINAMATH_CALUDE_walnut_chestnut_cost_l4156_415695


namespace NUMINAMATH_CALUDE_ms_jones_class_size_l4156_415608

theorem ms_jones_class_size :
  ∀ (total_students : ℕ),
    (total_students : ℝ) * 0.3 * (1/3) * 10 = 50 →
    total_students = 50 := by
  sorry

end NUMINAMATH_CALUDE_ms_jones_class_size_l4156_415608


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_distance_l4156_415638

theorem isosceles_right_triangle_distance (a : ℝ) (h : a = 8) :
  Real.sqrt (a^2 + a^2) = a * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_distance_l4156_415638


namespace NUMINAMATH_CALUDE_polynomial_division_degree_l4156_415650

open Polynomial

theorem polynomial_division_degree (f d q r : ℝ[X]) : 
  degree f = 15 →
  f = d * q + r →
  degree q = 9 →
  degree r = 4 →
  degree r < degree d →
  degree d = 6 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_degree_l4156_415650


namespace NUMINAMATH_CALUDE_smallest_valid_number_l4156_415619

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000000 ∧ n < 1000000000) ∧
  (n % 11 = 0) ∧
  (∀ d : ℕ, d ≥ 1 ∧ d ≤ 9 → (∃! p : ℕ, p ≥ 0 ∧ p < 9 ∧ (n / 10^p) % 10 = d))

theorem smallest_valid_number :
  is_valid_number 123475869 ∧
  ∀ m : ℕ, is_valid_number m → m ≥ 123475869 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l4156_415619


namespace NUMINAMATH_CALUDE_cubic_factorization_l4156_415685

theorem cubic_factorization (a : ℝ) : a^3 - 4*a = a*(a+2)*(a-2) := by sorry

end NUMINAMATH_CALUDE_cubic_factorization_l4156_415685


namespace NUMINAMATH_CALUDE_tan_alpha_two_implies_fraction_l4156_415668

theorem tan_alpha_two_implies_fraction (α : Real) (h : Real.tan α = 2) :
  (3 * Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_two_implies_fraction_l4156_415668


namespace NUMINAMATH_CALUDE_montoya_budget_allocation_l4156_415614

/-- The Montoya family's budget allocation problem -/
theorem montoya_budget_allocation :
  ∀ (groceries eating_out transportation rent utilities : ℝ),
    groceries = 0.6 →
    eating_out = 0.2 →
    transportation = 0.1 →
    rent = 0.05 →
    utilities = 0.05 →
    groceries + eating_out + transportation + rent + utilities = 1 :=
by sorry

end NUMINAMATH_CALUDE_montoya_budget_allocation_l4156_415614


namespace NUMINAMATH_CALUDE_ellipse_equation_l4156_415622

/-- An ellipse with the given properties has the standard equation x²/4 + y² = 1 -/
theorem ellipse_equation (a b c : ℝ) (h1 : a + b = c * Real.sqrt 3) 
  (h2 : c = Real.sqrt 3) : 
  ∃ (x y : ℝ), x^2 / 4 + y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l4156_415622


namespace NUMINAMATH_CALUDE_beth_class_size_l4156_415657

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Calculates the final number of students in Beth's class after n years -/
def finalStudents (initialStudents : ℕ) (joiningStart : ℕ) (joiningDiff : ℕ) 
                  (leavingStart : ℕ) (leavingDiff : ℕ) (years : ℕ) : ℕ :=
  initialStudents + 
  (arithmeticSum joiningStart joiningDiff years) - 
  (arithmeticSum leavingStart leavingDiff years)

theorem beth_class_size :
  finalStudents 150 30 5 15 3 4 = 222 := by
  sorry

end NUMINAMATH_CALUDE_beth_class_size_l4156_415657


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l4156_415699

theorem polynomial_divisibility (n : ℕ) (hn : n > 0) :
  ∃ Q : Polynomial ℚ, (n^2 * X^(n+2) - (2*n^2 + 2*n - 1) * X^(n+1) + (n+1)^2 * X^n - X - 1) = (X - 1)^3 * Q := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l4156_415699


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l4156_415635

/-- A sector OAB is a third of a circle with radius 6 cm. 
    An inscribed circle is tangent to the sector at three points. -/
def sector_with_inscribed_circle (r : ℝ) : Prop :=
  r > 0 ∧ 
  ∃ (R : ℝ), R = 6 ∧
  ∃ (θ : ℝ), θ = 2 * Real.pi / 3 ∧
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧
  x = R * Real.sin θ ∧
  y = R * (1 - Real.cos θ)

/-- The radius of the inscribed circle in the sector described above is 6√2 - 6 cm. -/
theorem inscribed_circle_radius :
  ∀ r : ℝ, sector_with_inscribed_circle r → r = 6 * (Real.sqrt 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l4156_415635


namespace NUMINAMATH_CALUDE_probability_of_twin_primes_l4156_415654

/-- The set of prime numbers not exceeding 30 -/
def primes_le_30 : Finset Nat := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

/-- A pair of primes (p, q) is considered a twin prime pair if q = p + 2 -/
def is_twin_prime_pair (p q : Nat) : Prop :=
  p ∈ primes_le_30 ∧ q ∈ primes_le_30 ∧ q = p + 2

/-- The set of twin prime pairs among primes not exceeding 30 -/
def twin_prime_pairs : Finset (Nat × Nat) :=
  {(3, 5), (5, 7), (11, 13), (17, 19)}

theorem probability_of_twin_primes :
  (twin_prime_pairs.card : Rat) / (Nat.choose primes_le_30.card 2 : Rat) = 4 / 45 :=
sorry

end NUMINAMATH_CALUDE_probability_of_twin_primes_l4156_415654


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l4156_415655

theorem absolute_value_inequality (x y : ℝ) (h : x * y < 0) : 
  |x + y| < |x - y| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l4156_415655


namespace NUMINAMATH_CALUDE_min_sum_squares_l4156_415697

theorem min_sum_squares (x y z t : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : 0 ≤ t)
  (h5 : |x - y| + |y - z| + |z - t| + |t - x| = 4) :
  2 ≤ x^2 + y^2 + z^2 + t^2 ∧ ∃ (a b c d : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
    |a - b| + |b - c| + |c - d| + |d - a| = 4 ∧ a^2 + b^2 + c^2 + d^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l4156_415697


namespace NUMINAMATH_CALUDE_midpoint_intersection_l4156_415676

/-- Given a convex quadrilateral ABCD with midpoints K, L, M, and N,
    prove that the intersection point O of KM and LN is the midpoint of KM, LN,
    and the segment connecting the midpoints of diagonals AC and BD. -/
theorem midpoint_intersection (A B C D K L M N O : ℝ × ℝ) : 
  (K = (A + B) / 2) →  -- K is midpoint of AB
  (L = (B + C) / 2) →  -- L is midpoint of BC
  (M = (C + D) / 2) →  -- M is midpoint of CD
  (N = (D + A) / 2) →  -- N is midpoint of DA
  (∃ t : ℝ, O = K + t • (M - K) ∧ O = L + t • (N - L)) →  -- O is on both KM and LN
  (O = (K + M) / 2) ∧  -- O is midpoint of KM
  (O = (L + N) / 2) ∧  -- O is midpoint of LN
  (O = ((A + C) / 2 + (B + D) / 2) / 2)  -- O is midpoint of diagonal midpoints
  := by sorry

end NUMINAMATH_CALUDE_midpoint_intersection_l4156_415676


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_of_polynomial_l4156_415660

theorem pure_imaginary_solutions_of_polynomial :
  let p (x : ℂ) := x^4 - 4*x^3 + 10*x^2 - 40*x - 100
  ∀ x : ℂ, (∃ a : ℝ, x = Complex.I * a) ∧ p x = 0 ↔ x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_of_polynomial_l4156_415660


namespace NUMINAMATH_CALUDE_expenditure_for_specific_hall_l4156_415653

/-- Calculates the total expenditure for covering a rectangular floor with a mat. -/
def total_expenditure (length width cost_per_sqm : ℝ) : ℝ :=
  length * width * cost_per_sqm

/-- Proves that the total expenditure for covering a specific rectangular floor is 3000. -/
theorem expenditure_for_specific_hall : 
  total_expenditure 20 15 10 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_for_specific_hall_l4156_415653


namespace NUMINAMATH_CALUDE_sequence_problem_l4156_415601

/-- Given a sequence {aₙ} where a₂ = 3, a₄ = 15, and {aₙ₊₁} is a geometric sequence, prove that a₆ = 63. -/
theorem sequence_problem (a : ℕ → ℝ) 
  (h1 : a 2 = 3)
  (h2 : a 4 = 15)
  (h3 : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) + 1 = (a n + 1) * q) :
  a 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l4156_415601


namespace NUMINAMATH_CALUDE_c_profit_share_l4156_415612

/-- Calculates the share of profit for a partner in a business partnership --/
def calculate_profit_share (total_investment : ℕ) (partner_investment : ℕ) (total_profit : ℕ) : ℕ :=
  (partner_investment * total_profit) / total_investment

theorem c_profit_share :
  let a_investment : ℕ := 5000
  let b_investment : ℕ := 8000
  let c_investment : ℕ := 9000
  let total_investment : ℕ := a_investment + b_investment + c_investment
  let total_profit : ℕ := 88000
  calculate_profit_share total_investment c_investment total_profit = 36000 := by
  sorry

end NUMINAMATH_CALUDE_c_profit_share_l4156_415612


namespace NUMINAMATH_CALUDE_roy_pens_count_l4156_415605

/-- The total number of pens Roy has -/
def total_pens (blue : ℕ) (black : ℕ) (red : ℕ) : ℕ :=
  blue + black + red

/-- The number of blue pens Roy has -/
def blue_pens : ℕ := 2

/-- The number of black pens Roy has -/
def black_pens : ℕ := 2 * blue_pens

/-- The number of red pens Roy has -/
def red_pens : ℕ := 2 * black_pens - 2

theorem roy_pens_count :
  total_pens blue_pens black_pens red_pens = 12 := by
  sorry

end NUMINAMATH_CALUDE_roy_pens_count_l4156_415605


namespace NUMINAMATH_CALUDE_range_of_a_l4156_415611

/-- The function f(x) = x|x^2 - 12| -/
def f (x : ℝ) : ℝ := x * abs (x^2 - 12)

theorem range_of_a (m : ℝ) (h_m : m > 0) :
  (∃ (a : ℝ), ∀ (y : ℝ), y ∈ Set.range (fun x => f x) ↔ y ∈ Set.Icc 0 (a * m^2)) →
  (∃ (a : ℝ), a ≥ 1 ∧ ∀ (b : ℝ), b ≥ 1 → ∃ (m : ℝ), m > 0 ∧
    (∀ (y : ℝ), y ∈ Set.range (fun x => f x) ↔ y ∈ Set.Icc 0 (b * m^2))) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4156_415611


namespace NUMINAMATH_CALUDE_trigonometric_product_sqrt_l4156_415648

theorem trigonometric_product_sqrt (h1 : Real.sin (π / 6) = 1 / 2)
                                   (h2 : Real.sin (π / 4) = Real.sqrt 2 / 2)
                                   (h3 : Real.sin (π / 3) = Real.sqrt 3 / 2) :
  Real.sqrt ((2 - (Real.sin (π / 6))^2) * (2 - (Real.sin (π / 4))^2) * (2 - (Real.sin (π / 3))^2)) = Real.sqrt 210 / 8 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_sqrt_l4156_415648


namespace NUMINAMATH_CALUDE_distributive_law_example_l4156_415659

theorem distributive_law_example :
  (7 + 125) * 8 = 7 * 8 + 125 * 8 := by sorry

end NUMINAMATH_CALUDE_distributive_law_example_l4156_415659


namespace NUMINAMATH_CALUDE_student_marks_l4156_415658

theorem student_marks (M P C : ℕ) : 
  C = P + 20 →
  (M + C) / 2 = 20 →
  M + P = 20 :=
by sorry

end NUMINAMATH_CALUDE_student_marks_l4156_415658


namespace NUMINAMATH_CALUDE_limit_x_cubed_minus_eight_over_x_minus_two_l4156_415672

theorem limit_x_cubed_minus_eight_over_x_minus_two : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → |((x^3 - 8) / (x - 2)) - 12| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_x_cubed_minus_eight_over_x_minus_two_l4156_415672


namespace NUMINAMATH_CALUDE_burger_cost_is_five_l4156_415644

/-- The cost of a burger meal -/
def burger_meal_cost : ℝ := 9.50

/-- The cost of a kid's meal -/
def kids_meal_cost : ℝ := 5

/-- The cost of french fries -/
def fries_cost : ℝ := 3

/-- The cost of a soft drink -/
def drink_cost : ℝ := 3

/-- The cost of a kid's burger -/
def kids_burger_cost : ℝ := 3

/-- The cost of kid's french fries -/
def kids_fries_cost : ℝ := 2

/-- The cost of a kid's juice box -/
def kids_juice_cost : ℝ := 2

/-- The amount saved by buying meals instead of individual items -/
def savings : ℝ := 10

theorem burger_cost_is_five (burger_cost : ℝ) : burger_cost = 5 :=
  by sorry

end NUMINAMATH_CALUDE_burger_cost_is_five_l4156_415644


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l4156_415688

open Real

theorem sufficient_but_not_necessary (θ : ℝ) : 
  (∀ θ, |θ - π/12| < π/12 → sin θ < 1/2) ∧ 
  (∃ θ, sin θ < 1/2 ∧ |θ - π/12| ≥ π/12) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l4156_415688


namespace NUMINAMATH_CALUDE_diana_candies_l4156_415651

/-- The number of candies Diana took out of a box -/
def candies_taken (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem diana_candies :
  let initial_candies : ℕ := 88
  let remaining_candies : ℕ := 82
  candies_taken initial_candies remaining_candies = 6 := by
sorry

end NUMINAMATH_CALUDE_diana_candies_l4156_415651


namespace NUMINAMATH_CALUDE_sum_of_coefficients_cube_expansion_l4156_415677

theorem sum_of_coefficients_cube_expansion : 
  ∃ (a b c d e : ℚ), 
    (∀ x, 1000 * x^3 + 27 = (a*x + b) * (c*x^2 + d*x + e)) ∧
    a + b + c + d + e = 92 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_cube_expansion_l4156_415677


namespace NUMINAMATH_CALUDE_function_inequality_l4156_415656

/-- Given a function f: ℝ → ℝ satisfying certain conditions, prove that f(-x₁) > f(-x₂) -/
theorem function_inequality (f : ℝ → ℝ) (x₁ x₂ : ℝ)
  (h1 : ∀ x, f (x + 1) = f (-x - 1))
  (h2 : ∀ x₁ x₂, x₁ ≥ 1 ∧ x₂ ≥ 1 ∧ x₁ < x₂ → f x₁ < f x₂)
  (h3 : x₁ < 0)
  (h4 : x₂ > 0)
  (h5 : x₁ + x₂ < -2) :
  f (-x₁) > f (-x₂) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l4156_415656


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l4156_415633

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (l m n : Line) (α : Plane) :
  parallel l m → parallel m n → perpendicular l α → perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l4156_415633


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l4156_415613

theorem a_eq_one_sufficient_not_necessary :
  ∃ (a : ℝ), a ^ 2 = a ∧ a ≠ 1 ∧
  ∀ (b : ℝ), b = 1 → b ^ 2 = b :=
by sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l4156_415613


namespace NUMINAMATH_CALUDE_ellipse_b_squared_value_l4156_415681

/-- The squared semi-minor axis of an ellipse with equation (x^2/25) + (y^2/b^2) = 1,
    which has the same foci as a hyperbola with equation (x^2/225) - (y^2/144) = 1/36 -/
def ellipse_b_squared : ℝ := 14.75

/-- The equation of the ellipse -/
def is_on_ellipse (x y b : ℝ) : Prop :=
  x^2 / 25 + y^2 / b^2 = 1

/-- The equation of the hyperbola -/
def is_on_hyperbola (x y : ℝ) : Prop :=
  x^2 / 225 - y^2 / 144 = 1 / 36

/-- The foci of the ellipse and hyperbola coincide -/
axiom foci_coincide : ∃ c : ℝ,
  c^2 = 25 - ellipse_b_squared ∧
  c^2 = 225 / 36 - 144 / 36

theorem ellipse_b_squared_value :
  ellipse_b_squared = 14.75 := by sorry

end NUMINAMATH_CALUDE_ellipse_b_squared_value_l4156_415681


namespace NUMINAMATH_CALUDE_marble_difference_l4156_415629

/-- Given information about marbles owned by Amanda, Katrina, and Mabel -/
theorem marble_difference (amanda katrina mabel : ℕ) 
  (h1 : amanda + 12 = 2 * katrina)
  (h2 : mabel = 5 * katrina)
  (h3 : mabel = 85) :
  mabel - amanda = 63 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l4156_415629


namespace NUMINAMATH_CALUDE_total_books_count_l4156_415687

/-- The number of bookshelves -/
def num_bookshelves : ℕ := 1250

/-- The number of books on each bookshelf -/
def books_per_shelf : ℕ := 45

/-- The total number of books on all shelves -/
def total_books : ℕ := num_bookshelves * books_per_shelf

theorem total_books_count : total_books = 56250 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l4156_415687


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_two_million_l4156_415616

/-- The smallest positive integer n such that the nth term of a geometric sequence
    with first term 5/8 and second term 25 is divisible by 2,000,000 is 7. -/
theorem smallest_n_divisible_by_two_million (a₁ a₂ : ℚ) (h₁ : a₁ = 5/8) (h₂ : a₂ = 25) : 
  ∃ n : ℕ+, (∀ k : ℕ+, k < n → ¬(∃ m : ℤ, a₁ * (a₂ / a₁)^(k.val - 1) = 2000000 * m)) ∧ 
  (∃ m : ℤ, a₁ * (a₂ / a₁)^(n.val - 1) = 2000000 * m) ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_two_million_l4156_415616


namespace NUMINAMATH_CALUDE_triangle_angles_l4156_415666

theorem triangle_angles (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) * (a + b - c) = 3 * a * b →
  Real.sin A ^ 2 = Real.sin B ^ 2 + Real.sin C ^ 2 →
  A + B + C = π →
  a * Real.sin B = b * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  c * Real.sin A = a * Real.sin C →
  A = π / 6 ∧ B = π / 3 ∧ C = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_l4156_415666


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l4156_415631

def a : ℝ × ℝ := (1, 3)
def b (m : ℝ) : ℝ × ℝ := (-2, m)

theorem perpendicular_vectors (m : ℝ) : 
  (a.1 * (a.1 + 2 * (b m).1) + a.2 * (a.2 + 2 * (b m).2) = 0) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l4156_415631


namespace NUMINAMATH_CALUDE_days_from_thursday_l4156_415609

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advance_days (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | m + 1 => next_day (advance_days start m)

theorem days_from_thursday :
  advance_days DayOfWeek.Thursday 53 = DayOfWeek.Monday := by
  sorry


end NUMINAMATH_CALUDE_days_from_thursday_l4156_415609


namespace NUMINAMATH_CALUDE_bridge_brick_ratio_l4156_415602

theorem bridge_brick_ratio (total_bricks : ℕ) (type_a_bricks : ℕ) (other_bricks : ℕ) : 
  total_bricks = 150 →
  type_a_bricks = 40 →
  other_bricks = 90 →
  ∃ (type_b_bricks : ℕ), 
    type_a_bricks + type_b_bricks + other_bricks = total_bricks ∧
    type_b_bricks * 2 = type_a_bricks :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_brick_ratio_l4156_415602


namespace NUMINAMATH_CALUDE_valid_arrangements_l4156_415617

/-- The number of ways to arrange students in a classroom. -/
def arrange_students : ℕ :=
  let num_students : ℕ := 30
  let num_rows : ℕ := 5
  let num_cols : ℕ := 6
  let num_boys : ℕ := 15
  let num_girls : ℕ := 15
  2 * (Nat.factorial num_boys) * (Nat.factorial num_girls)

/-- Theorem stating the number of valid arrangements of students. -/
theorem valid_arrangements (num_students num_rows num_cols num_boys num_girls : ℕ) 
  (h1 : num_students = 30)
  (h2 : num_rows = 5)
  (h3 : num_cols = 6)
  (h4 : num_boys = 15)
  (h5 : num_girls = 15)
  (h6 : num_students = num_boys + num_girls)
  (h7 : num_students = num_rows * num_cols) :
  arrange_students = 2 * (Nat.factorial num_boys) * (Nat.factorial num_girls) :=
by
  sorry

#eval arrange_students

end NUMINAMATH_CALUDE_valid_arrangements_l4156_415617


namespace NUMINAMATH_CALUDE_product_of_numbers_l4156_415604

theorem product_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 16) 
  (sum_squares_eq : x^2 + y^2 = 200) : 
  x * y = 28 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l4156_415604


namespace NUMINAMATH_CALUDE_coefficient_implies_a_value_l4156_415686

theorem coefficient_implies_a_value (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = (1 + a * x)^5 * (1 - 2*x)^4) ∧
   (∃ c : ℝ → ℝ, (∀ x, f x = c 0 + c 1 * x + c 2 * x^2 + c 3 * x^3 + c 4 * x^4 + c 5 * x^5 + c 6 * x^6 + c 7 * x^7 + c 8 * x^8 + c 9 * x^9) ∧
    c 2 = -16)) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_coefficient_implies_a_value_l4156_415686


namespace NUMINAMATH_CALUDE_upgraded_sensor_fraction_l4156_415680

/-- Represents a satellite with modular units and sensors. -/
structure Satellite where
  units : ℕ
  non_upgraded_per_unit : ℕ
  total_upgraded : ℕ
  non_upgraded_ratio : non_upgraded_per_unit = total_upgraded / 4

/-- The fraction of upgraded sensors on the satellite is 1/7. -/
theorem upgraded_sensor_fraction (s : Satellite) (h : s.units = 24) :
  s.total_upgraded / (s.units * s.non_upgraded_per_unit + s.total_upgraded) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_upgraded_sensor_fraction_l4156_415680


namespace NUMINAMATH_CALUDE_perimeter_decreases_to_convex_hull_outer_perimeter_not_smaller_l4156_415683

-- Define a polygon as a list of points in 2D space
def Polygon := List (ℝ × ℝ)

-- Define a function to calculate the perimeter of a polygon
def perimeter (p : Polygon) : ℝ := sorry

-- Define a predicate to check if a polygon is convex
def is_convex (p : Polygon) : Prop := sorry

-- Define the convex hull of a polygon
def convex_hull (p : Polygon) : Polygon := sorry

-- Define a predicate to check if one polygon is completely inside another
def is_inside (a b : Polygon) : Prop := sorry

theorem perimeter_decreases_to_convex_hull (p : Polygon) : 
  perimeter (convex_hull p) < perimeter p := sorry

theorem outer_perimeter_not_smaller (a b : Polygon) 
  (h1 : is_convex a) (h2 : is_convex b) (h3 : is_inside a b) : 
  perimeter b ≥ perimeter a := sorry

end NUMINAMATH_CALUDE_perimeter_decreases_to_convex_hull_outer_perimeter_not_smaller_l4156_415683


namespace NUMINAMATH_CALUDE_dice_roll_probability_l4156_415692

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of favorable outcomes on the first die (less than 3) -/
def favorableFirst : ℕ := 2

/-- The number of favorable outcomes on the second die (greater than 3) -/
def favorableSecond : ℕ := 3

/-- The probability of the desired outcome when rolling two dice -/
def probability : ℚ := (favorableFirst / numSides) * (favorableSecond / numSides)

theorem dice_roll_probability :
  probability = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l4156_415692


namespace NUMINAMATH_CALUDE_subset_implies_a_leq_one_l4156_415630

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 1}
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

-- State the theorem
theorem subset_implies_a_leq_one (a : ℝ) : A ⊆ B a → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_leq_one_l4156_415630


namespace NUMINAMATH_CALUDE_prob_at_least_two_same_carriage_l4156_415626

/-- The number of carriages in the train -/
def num_carriages : ℕ := 10

/-- The number of acquaintances boarding the train -/
def num_people : ℕ := 3

/-- The probability that at least two people board the same carriage -/
def prob_same_carriage : ℚ := 7/25

/-- Theorem stating the probability of at least two people boarding the same carriage -/
theorem prob_at_least_two_same_carriage : 
  1 - (num_carriages.descFactorial num_people : ℚ) / (num_carriages ^ num_people : ℚ) = prob_same_carriage := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_same_carriage_l4156_415626


namespace NUMINAMATH_CALUDE_locus_of_right_triangle_vertex_l4156_415675

-- Define the points M and N
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

-- Define the property of point P forming a right-angled triangle with MN as hypotenuse
def is_right_triangle_vertex (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16

-- Theorem statement
theorem locus_of_right_triangle_vertex :
  ∀ (P : ℝ × ℝ), is_right_triangle_vertex P →
    (P.1^2 + P.2^2 = 4 ∧ P.1 ≠ 2 ∧ P.1 ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_right_triangle_vertex_l4156_415675


namespace NUMINAMATH_CALUDE_min_value_theorem_l4156_415640

theorem min_value_theorem (a b : ℝ) (h : a * b > 0) :
  a^2 + 4*b^2 + 1/(a*b) ≥ 4 ∧
  (a^2 + 4*b^2 + 1/(a*b) = 4 ↔ a = 1/Real.rpow 2 (1/4) ∧ b = 1/Real.rpow 2 (1/4)) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4156_415640


namespace NUMINAMATH_CALUDE_solve_system_l4156_415673

-- Define the variables x and y
variable (x y : ℤ)

-- State the theorem
theorem solve_system : 
  (3:ℝ)^x = 27^(y+1) → (16:ℝ)^y = 2^(x-8) → 2*x + y = -29 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l4156_415673


namespace NUMINAMATH_CALUDE_markup_is_100_percent_l4156_415618

/-- Calculates the markup percentage given wholesale price, initial price, and price increase. -/
def markup_percentage (wholesale_price initial_price price_increase : ℚ) : ℚ :=
  let new_price := initial_price + price_increase
  (new_price - wholesale_price) / wholesale_price * 100

/-- Proves that the markup percentage is 100% given the specified conditions. -/
theorem markup_is_100_percent (wholesale_price initial_price price_increase : ℚ) 
  (h1 : wholesale_price = 20)
  (h2 : initial_price = 34)
  (h3 : price_increase = 6) :
  markup_percentage wholesale_price initial_price price_increase = 100 := by
  sorry

#eval markup_percentage 20 34 6

end NUMINAMATH_CALUDE_markup_is_100_percent_l4156_415618


namespace NUMINAMATH_CALUDE_total_weight_of_mixtures_l4156_415639

/-- Represents a mixture of vegetable ghee -/
structure Mixture where
  ratio_a : ℚ
  ratio_b : ℚ
  total_volume : ℚ

/-- Calculates the weight of a mixture in kg -/
def mixture_weight (m : Mixture) (weight_a weight_b : ℚ) : ℚ :=
  let total_ratio := m.ratio_a + m.ratio_b
  let volume_a := (m.ratio_a / total_ratio) * m.total_volume
  let volume_b := (m.ratio_b / total_ratio) * m.total_volume
  (volume_a * weight_a + volume_b * weight_b) / 1000

def mixture1 : Mixture := ⟨3, 2, 6⟩
def mixture2 : Mixture := ⟨5, 3, 4⟩
def mixture3 : Mixture := ⟨9, 4, 6.5⟩

def weight_a : ℚ := 900
def weight_b : ℚ := 750

theorem total_weight_of_mixtures :
  mixture_weight mixture1 weight_a weight_b +
  mixture_weight mixture2 weight_a weight_b +
  mixture_weight mixture3 weight_a weight_b = 13.965 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_mixtures_l4156_415639


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l4156_415625

/-- Given an arithmetic sequence {a_n} with non-zero common difference d,
    if a_2 + a_3 = a_6, then (a_1 + a_2) / (a_3 + a_4 + a_5) = 1/3 -/
theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : a 2 + a 3 = a 6) :
  (a 1 + a 2) / (a 3 + a 4 + a 5) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l4156_415625


namespace NUMINAMATH_CALUDE_total_tickets_is_91_l4156_415696

/-- The total number of tickets needed for Janet's family's amusement park visits -/
def total_tickets : ℕ :=
  let family_size : ℕ := 4
  let adults : ℕ := 2
  let children : ℕ := 2
  let roller_coaster_adult : ℕ := 7
  let roller_coaster_child : ℕ := 5
  let giant_slide_adult : ℕ := 4
  let giant_slide_child : ℕ := 3
  let adult_roller_coaster_rides : ℕ := 3
  let child_roller_coaster_rides : ℕ := 2
  let adult_giant_slide_rides : ℕ := 5
  let child_giant_slide_rides : ℕ := 3

  let roller_coaster_tickets := 
    adults * roller_coaster_adult * adult_roller_coaster_rides +
    children * roller_coaster_child * child_roller_coaster_rides
  
  let giant_slide_tickets :=
    1 * giant_slide_adult * adult_giant_slide_rides +
    1 * giant_slide_child * child_giant_slide_rides

  roller_coaster_tickets + giant_slide_tickets

theorem total_tickets_is_91 : total_tickets = 91 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_is_91_l4156_415696


namespace NUMINAMATH_CALUDE_compound_interest_duration_l4156_415606

theorem compound_interest_duration (P A r : ℝ) (h_P : P = 979.0209790209791) (h_A : A = 1120) (h_r : r = 0.06) :
  ∃ t : ℝ, A = P * (1 + r) ^ t := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_duration_l4156_415606


namespace NUMINAMATH_CALUDE_rachel_books_total_l4156_415671

/-- The number of books Rachel has in total -/
def total_books (mystery_shelves picture_shelves scifi_shelves bio_shelves books_per_shelf : ℕ) : ℕ :=
  (mystery_shelves + picture_shelves + scifi_shelves + bio_shelves) * books_per_shelf

/-- Theorem stating that Rachel has 135 books in total -/
theorem rachel_books_total :
  total_books 6 2 3 4 9 = 135 := by
  sorry

end NUMINAMATH_CALUDE_rachel_books_total_l4156_415671


namespace NUMINAMATH_CALUDE_solution_theorem_l4156_415694

-- Define the function f(x) = x^2023 + x
def f (x : ℝ) := x^2023 + x

-- State the theorem
theorem solution_theorem (x y : ℝ) :
  (3*x + y)^2023 + x^2023 + 4*x + y = 0 → 4*x + y = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_theorem_l4156_415694


namespace NUMINAMATH_CALUDE_harmonic_mean_of_2_3_6_l4156_415610

theorem harmonic_mean_of_2_3_6 : 
  3 = 3 / (1 / 2 + 1 / 3 + 1 / 6) := by sorry

end NUMINAMATH_CALUDE_harmonic_mean_of_2_3_6_l4156_415610


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l4156_415691

/-- Given a point p and a line l, this function returns the equation of the line parallel to l that passes through p. -/
def parallel_line_equation (p : ℝ × ℝ) (l : ℝ → ℝ → ℝ → Prop) : ℝ → ℝ → ℝ → Prop :=
  sorry

theorem parallel_line_through_point :
  let p : ℝ × ℝ := (-1, 3)
  let l : ℝ → ℝ → ℝ → Prop := fun x y z ↦ x - 2*y + z = 0
  let result : ℝ → ℝ → ℝ → Prop := fun x y z ↦ x - 2*y + 7 = 0
  parallel_line_equation p l = result :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l4156_415691


namespace NUMINAMATH_CALUDE_machine_value_after_two_years_l4156_415634

/-- Calculates the market value of a machine after a given number of years,
    given its initial value and yearly depreciation rate. -/
def marketValue (initialValue : ℝ) (depreciationRate : ℝ) (years : ℕ) : ℝ :=
  initialValue - (depreciationRate * initialValue * years)

/-- Theorem stating that a machine with an initial value of $8,000 and a yearly
    depreciation of 30% of its purchase price will have a market value of $3,200
    after 2 years. -/
theorem machine_value_after_two_years :
  marketValue 8000 0.3 2 = 3200 := by
  sorry

end NUMINAMATH_CALUDE_machine_value_after_two_years_l4156_415634


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4156_415620

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (3 + 4 * Complex.I) = 43/25 + (1/25) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4156_415620


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l4156_415645

theorem imaginary_part_of_z (z : ℂ) : (2 - Complex.I) * z = 5 → Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l4156_415645


namespace NUMINAMATH_CALUDE_system_solution_l4156_415628

theorem system_solution : ∃ (x y : ℚ), 
  (x + 4*y = 14) ∧ 
  ((x - 3) / 4 - (y - 3) / 3 = 1 / 12) ∧ 
  (x = 3) ∧ 
  (y = 11 / 4) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4156_415628


namespace NUMINAMATH_CALUDE_power_of_power_three_l4156_415664

theorem power_of_power_three : (3^3)^(3^3) = 27^27 := by sorry

end NUMINAMATH_CALUDE_power_of_power_three_l4156_415664


namespace NUMINAMATH_CALUDE_greatest_power_under_500_l4156_415641

theorem greatest_power_under_500 (a b : ℕ) :
  a > 0 → b > 1 → a^b < 500 →
  (∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → x^y ≤ a^b) →
  a + b = 24 := by
sorry

end NUMINAMATH_CALUDE_greatest_power_under_500_l4156_415641


namespace NUMINAMATH_CALUDE_problem_solution_l4156_415649

def M : Set ℝ := {x | x^2 - 3*x ≤ 10}
def N (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2*a + 1}

theorem problem_solution :
  (∀ x, x ∈ (Set.univ \ M) ∪ (N 2) ↔ x > 5 ∨ x < -2) ∧
  (∀ a, M ∪ N a = M ↔ a < -2 ∨ (-1 ≤ a ∧ a ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l4156_415649


namespace NUMINAMATH_CALUDE_paperclip_capacity_l4156_415674

/-- Given that a box of volume 18 cm³ can hold 60 paperclips, and the storage density
    decreases by 10% in larger boxes, prove that a box of volume 72 cm³ can hold 216 paperclips. -/
theorem paperclip_capacity (small_volume small_capacity large_volume : ℝ) 
    (h1 : small_volume = 18)
    (h2 : small_capacity = 60)
    (h3 : large_volume = 72)
    (h4 : large_volume > small_volume) :
    let density_ratio := large_volume / small_volume
    let unadjusted_capacity := small_capacity * density_ratio
    let adjusted_capacity := unadjusted_capacity * 0.9
    adjusted_capacity = 216 := by
  sorry


end NUMINAMATH_CALUDE_paperclip_capacity_l4156_415674


namespace NUMINAMATH_CALUDE_gold_distribution_l4156_415667

/-- Given an arithmetic sequence with 10 terms, if the sum of the first 3 terms
    is 4 and the sum of the last 4 terms is 3, then the common difference
    of the sequence is 7/78. -/
theorem gold_distribution (a : ℕ → ℚ) :
  (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence
  (∀ n, n ≥ 10 → a n = 0) →             -- 10 terms
  a 9 + a 8 + a 7 = 4 →                 -- sum of first 3 terms is 4
  a 0 + a 1 + a 2 + a 3 = 3 →           -- sum of last 4 terms is 3
  a 1 - a 0 = 7 / 78 :=                 -- common difference is 7/78
by sorry

end NUMINAMATH_CALUDE_gold_distribution_l4156_415667


namespace NUMINAMATH_CALUDE_monthly_salary_is_1000_l4156_415661

/-- Calculates the monthly salary given savings rate, expense increase, and new savings amount -/
def calculate_salary (savings_rate : ℚ) (expense_increase : ℚ) (new_savings : ℚ) : ℚ :=
  new_savings / (savings_rate - (1 - savings_rate) * expense_increase)

/-- Theorem stating that under the given conditions, the monthly salary is 1000 -/
theorem monthly_salary_is_1000 : 
  let savings_rate : ℚ := 25 / 100
  let expense_increase : ℚ := 10 / 100
  let new_savings : ℚ := 175
  calculate_salary savings_rate expense_increase new_savings = 1000 := by
  sorry

#eval calculate_salary (25/100) (10/100) 175

end NUMINAMATH_CALUDE_monthly_salary_is_1000_l4156_415661


namespace NUMINAMATH_CALUDE_julia_tag_playmates_l4156_415632

/-- The number of kids Julia played tag with on Monday, Tuesday, and in total. -/
structure TagPlaymates where
  monday : ℕ
  tuesday : ℕ
  total : ℕ

/-- Given that Julia played tag with 20 kids in total and 13 kids on Tuesday,
    prove that she played tag with 7 kids on Monday. -/
theorem julia_tag_playmates : ∀ (j : TagPlaymates),
  j.total = 20 → j.tuesday = 13 → j.monday = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_julia_tag_playmates_l4156_415632


namespace NUMINAMATH_CALUDE_initial_machines_count_l4156_415637

/-- The number of bottles produced per minute by the initial number of machines -/
def initial_production_rate : ℕ := 270

/-- The number of machines used in the second scenario -/
def second_scenario_machines : ℕ := 20

/-- The number of bottles produced in the second scenario -/
def second_scenario_production : ℕ := 3600

/-- The time in minutes for the second scenario -/
def second_scenario_time : ℕ := 4

/-- The number of machines running initially -/
def initial_machines : ℕ := 6

theorem initial_machines_count :
  initial_machines * initial_production_rate = second_scenario_machines * (second_scenario_production / second_scenario_time) :=
by sorry

end NUMINAMATH_CALUDE_initial_machines_count_l4156_415637


namespace NUMINAMATH_CALUDE_unique_solution_abc_l4156_415621

theorem unique_solution_abc (a b c : ℝ) 
  (ha : a > 2) (hb : b > 2) (hc : c > 2)
  (heq : (a+1)^2 / (b+c-1) + (b+2)^2 / (c+a-3) + (c+3)^2 / (a+b-5) = 32) :
  a = 8 ∧ b = 6 ∧ c = 5 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_abc_l4156_415621


namespace NUMINAMATH_CALUDE_inequality_solutions_l4156_415662

theorem inequality_solutions :
  (∀ x : ℝ, 2 + 3*x - 2*x^2 > 0 ↔ -1/2 < x ∧ x < 2) ∧
  (∀ x : ℝ, x*(3-x) ≤ x*(x+2) - 1 ↔ x ≤ -1/2 ∨ x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solutions_l4156_415662


namespace NUMINAMATH_CALUDE_max_value_sin_cos_function_l4156_415698

theorem max_value_sin_cos_function :
  let f : ℝ → ℝ := λ x => Real.sin (π / 2 + x) * Real.cos (π / 6 - x)
  ∃ M : ℝ, (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = (2 + Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_function_l4156_415698


namespace NUMINAMATH_CALUDE_age_problem_l4156_415690

theorem age_problem (a b : ℕ) (h1 : a - 10 = (b - 10) / 2) (h2 : 4 * a = 3 * b) :
  a + b = 35 := by sorry

end NUMINAMATH_CALUDE_age_problem_l4156_415690


namespace NUMINAMATH_CALUDE_total_vehicles_l4156_415693

theorem total_vehicles (lanes : Nat) (trucks_per_lane : Nat) : 
  lanes = 4 → 
  trucks_per_lane = 60 → 
  (lanes * trucks_per_lane * 2 + lanes * trucks_per_lane) = 2160 :=
by
  sorry

end NUMINAMATH_CALUDE_total_vehicles_l4156_415693


namespace NUMINAMATH_CALUDE_division_problem_l4156_415689

theorem division_problem (n : ℕ) : n / 20 = 10 ∧ n % 20 = 10 ↔ n = 210 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4156_415689


namespace NUMINAMATH_CALUDE_remainder_of_product_product_remainder_l4156_415647

theorem remainder_of_product (a b c : ℕ) : (a * b * c) % 12 = ((a % 12) * (b % 12) * (c % 12)) % 12 := by sorry

theorem product_remainder : (1625 * 1627 * 1629) % 12 = 3 := by
  have h1 : 1625 % 12 = 5 := by sorry
  have h2 : 1627 % 12 = 7 := by sorry
  have h3 : 1629 % 12 = 9 := by sorry
  have h4 : (5 * 7 * 9) % 12 = 3 := by sorry
  exact calc
    (1625 * 1627 * 1629) % 12 = ((1625 % 12) * (1627 % 12) * (1629 % 12)) % 12 := by apply remainder_of_product
    _ = (5 * 7 * 9) % 12 := by rw [h1, h2, h3]
    _ = 3 := by exact h4

end NUMINAMATH_CALUDE_remainder_of_product_product_remainder_l4156_415647


namespace NUMINAMATH_CALUDE_triangle_max_area_l4156_415684

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  (Real.sin A - Real.sin B) / Real.sin C = (c - b) / (2 + b) →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  ∃ (area : ℝ), area ≤ Real.sqrt 3 ∧
    area = (1/2) * a * b * Real.sin C ∧
    ∀ (area' : ℝ), area' = (1/2) * a * b * Real.sin C → area' ≤ area :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l4156_415684


namespace NUMINAMATH_CALUDE_square_of_sum_fifteen_seven_l4156_415682

theorem square_of_sum_fifteen_seven : 15^2 + 2*(15*7) + 7^2 = 484 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_fifteen_seven_l4156_415682


namespace NUMINAMATH_CALUDE_twopirsquared_is_standard_l4156_415665

/-- Represents a mathematical expression -/
inductive MathExpression
  | Constant (c : ℝ)
  | Variable (v : String)
  | Multiplication (e1 e2 : MathExpression)
  | Exponentiation (base : MathExpression) (exponent : ℕ)

/-- Checks if an expression follows standard mathematical notation -/
def isStandardNotation : MathExpression → Bool
  | MathExpression.Constant _ => true
  | MathExpression.Variable _ => true
  | MathExpression.Multiplication e1 e2 => 
      match e1, e2 with
      | MathExpression.Constant _, _ => isStandardNotation e2
      | _, _ => false
  | MathExpression.Exponentiation base _ => isStandardNotation base

/-- Represents the expression 2πr² -/
def twopirsquared : MathExpression :=
  MathExpression.Multiplication
    (MathExpression.Constant 2)
    (MathExpression.Multiplication
      (MathExpression.Variable "π")
      (MathExpression.Exponentiation (MathExpression.Variable "r") 2))

/-- Theorem stating that 2πr² follows standard mathematical notation -/
theorem twopirsquared_is_standard : isStandardNotation twopirsquared = true := by
  sorry

end NUMINAMATH_CALUDE_twopirsquared_is_standard_l4156_415665
