import Mathlib

namespace NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_l1577_157764

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  all_positive : ∀ i, angles i > 0

-- Define an obtuse angle
def is_obtuse (angle : ℝ) : Prop := angle > 90

-- Theorem statement
theorem triangle_at_most_one_obtuse (t : Triangle) : 
  ¬(∃ i j : Fin 3, i ≠ j ∧ is_obtuse (t.angles i) ∧ is_obtuse (t.angles j)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_l1577_157764


namespace NUMINAMATH_CALUDE_work_completion_time_l1577_157780

theorem work_completion_time (a_time b_time : ℕ) (remaining_fraction : ℚ) : 
  a_time = 15 → b_time = 20 → remaining_fraction = 8/15 → 
  (1 - remaining_fraction) / ((1 / a_time) + (1 / b_time)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1577_157780


namespace NUMINAMATH_CALUDE_birds_on_fence_l1577_157765

/-- Given that there are initially 12 birds on a fence and after more birds land
    there are a total of 20 birds, prove that 8 birds landed on the fence. -/
theorem birds_on_fence (initial_birds : ℕ) (total_birds : ℕ) (h1 : initial_birds = 12) (h2 : total_birds = 20) :
  total_birds - initial_birds = 8 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l1577_157765


namespace NUMINAMATH_CALUDE_johns_spending_l1577_157704

theorem johns_spending (total : ℚ) 
  (h1 : total = 24)
  (h2 : total * (1/4) + total * (1/3) + 6 + bakery = total) : 
  bakery / total = 1/6 :=
by sorry

end NUMINAMATH_CALUDE_johns_spending_l1577_157704


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l1577_157768

theorem imaginary_part_of_complex_product : Complex.im ((1 - Complex.I * Real.sqrt 2) * Complex.I) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l1577_157768


namespace NUMINAMATH_CALUDE_f_prime_at_i_l1577_157788

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the function f
def f (x : ℂ) : ℂ := x^4 - x^2

-- State the theorem
theorem f_prime_at_i : 
  (deriv f) i = -6 * i := by sorry

end NUMINAMATH_CALUDE_f_prime_at_i_l1577_157788


namespace NUMINAMATH_CALUDE_probability_between_X_and_Z_l1577_157793

/-- Given a line segment XW where XW = 4XZ = 8YW, the probability of selecting a point between X and Z is 1/4 -/
theorem probability_between_X_and_Z (XW XZ YW : ℝ) 
  (h1 : XW = 4 * XZ) 
  (h2 : XW = 8 * YW) 
  (h3 : XW > 0) : 
  XZ / XW = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_between_X_and_Z_l1577_157793


namespace NUMINAMATH_CALUDE_number_problem_l1577_157721

theorem number_problem : ∃ n : ℝ, 8 * n - 4 = 17 ∧ n = 2.625 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1577_157721


namespace NUMINAMATH_CALUDE_min_value_problem_l1577_157743

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let line := {(x, y) : ℝ × ℝ | a * x - b * y + 2 = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 + 4*x - 4*y - 1 = 0}
  (∃ (p q : ℝ × ℝ), p ∈ line ∧ q ∈ line ∧ p ∈ circle ∧ q ∈ circle ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 6) →
  (2 / a + 3 / b ≥ 5 + 2 * Real.sqrt 6 ∧ 
   ∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ 
     2 / a' + 3 / b' = 5 + 2 * Real.sqrt 6 ∧
     ∃ (p q : ℝ × ℝ), p ∈ {(x, y) : ℝ × ℝ | a' * x - b' * y + 2 = 0} ∧ 
       q ∈ {(x, y) : ℝ × ℝ | a' * x - b' * y + 2 = 0} ∧
       p ∈ circle ∧ q ∈ circle ∧
       Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l1577_157743


namespace NUMINAMATH_CALUDE_remainder_theorem_l1577_157786

def polynomial (x : ℝ) : ℝ := 5*x^5 - 12*x^4 + 3*x^3 - 7*x + 15

def divisor (x : ℝ) : ℝ := 3*x - 6

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * q x + (-7) :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1577_157786


namespace NUMINAMATH_CALUDE_sticker_sharing_l1577_157779

theorem sticker_sharing (total_stickers : ℕ) (andrew_final : ℕ) : 
  total_stickers = 1500 →
  andrew_final = 900 →
  (2 : ℚ) / 3 = (andrew_final - total_stickers / 5) / (3 * total_stickers / 5) :=
by sorry

end NUMINAMATH_CALUDE_sticker_sharing_l1577_157779


namespace NUMINAMATH_CALUDE_factorization_theorem_l1577_157737

theorem factorization_theorem (a : ℝ) : 4 * a^2 - 4 = 4 * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l1577_157737


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1577_157776

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x > -1}

-- Define set B
def B : Set ℝ := {x | x > 2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | -1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1577_157776


namespace NUMINAMATH_CALUDE_old_edition_pages_l1577_157763

/-- The number of pages in the new edition of the Geometry book -/
def new_edition_pages : ℕ := 450

/-- The difference between twice the number of pages in the old edition and the new edition -/
def page_difference : ℕ := 230

/-- Theorem stating that the old edition of the Geometry book had 340 pages -/
theorem old_edition_pages : 
  ∃ (x : ℕ), 2 * x - page_difference = new_edition_pages ∧ x = 340 := by
  sorry

end NUMINAMATH_CALUDE_old_edition_pages_l1577_157763


namespace NUMINAMATH_CALUDE_guessing_game_score_sum_l1577_157711

/-- The guessing game score problem -/
theorem guessing_game_score_sum :
  ∀ (hajar_score farah_score : ℕ),
  hajar_score = 24 →
  farah_score - hajar_score = 21 →
  farah_score > hajar_score →
  hajar_score + farah_score = 69 :=
by
  sorry

end NUMINAMATH_CALUDE_guessing_game_score_sum_l1577_157711


namespace NUMINAMATH_CALUDE_first_employee_wage_is_12_l1577_157716

/-- The hourly wage of the first employee -/
def first_employee_wage : ℝ := sorry

/-- The hourly wage of the second employee -/
def second_employee_wage : ℝ := 22

/-- The hourly subsidy for hiring the second employee -/
def hourly_subsidy : ℝ := 6

/-- The number of hours worked per week -/
def hours_per_week : ℝ := 40

/-- The weekly savings by hiring the first employee -/
def weekly_savings : ℝ := 160

theorem first_employee_wage_is_12 :
  first_employee_wage = 12 :=
by
  have h1 : hours_per_week * (second_employee_wage - hourly_subsidy) - 
            hours_per_week * first_employee_wage = weekly_savings := by sorry
  sorry

end NUMINAMATH_CALUDE_first_employee_wage_is_12_l1577_157716


namespace NUMINAMATH_CALUDE_polynomial_value_at_three_l1577_157758

theorem polynomial_value_at_three : 
  let x : ℝ := 3
  (x^5 + 5*x^3 + 2*x) = 384 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_three_l1577_157758


namespace NUMINAMATH_CALUDE_student_factor_problem_l1577_157708

theorem student_factor_problem (n : ℝ) (f : ℝ) : n = 124 → n * f - 138 = 110 → f = 2 := by
  sorry

end NUMINAMATH_CALUDE_student_factor_problem_l1577_157708


namespace NUMINAMATH_CALUDE_greenhouse_optimization_l1577_157781

/-- Given a rectangle with area 800 m², prove that the maximum area of the inner rectangle
    formed by subtracting a 1 m border on three sides and a 3 m border on one side
    is 648 m², achieved when the original rectangle has dimensions 40 m × 20 m. -/
theorem greenhouse_optimization (a b : ℝ) :
  a > 0 ∧ b > 0 ∧ a * b = 800 →
  (a - 2) * (b - 4) ≤ 648 ∧
  (a - 2) * (b - 4) = 648 ↔ a = 40 ∧ b = 20 :=
by sorry

end NUMINAMATH_CALUDE_greenhouse_optimization_l1577_157781


namespace NUMINAMATH_CALUDE_g_difference_l1577_157715

-- Define the function g
noncomputable def g (n : ℤ) : ℝ :=
  (7 + 4 * Real.sqrt 7) / 14 * ((2 + Real.sqrt 7) / 3) ^ n +
  (7 - 4 * Real.sqrt 7) / 14 * ((2 - Real.sqrt 7) / 3) ^ n +
  3

-- Theorem statement
theorem g_difference (n : ℤ) : g (n + 1) - g (n - 1) = g n := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l1577_157715


namespace NUMINAMATH_CALUDE_correct_division_l1577_157745

theorem correct_division (dividend : ℕ) (incorrect_divisor correct_divisor incorrect_quotient : ℕ) 
  (h1 : incorrect_divisor = 72)
  (h2 : correct_divisor = 36)
  (h3 : incorrect_quotient = 24)
  (h4 : dividend = incorrect_divisor * incorrect_quotient) :
  dividend / correct_divisor = 48 := by
  sorry

end NUMINAMATH_CALUDE_correct_division_l1577_157745


namespace NUMINAMATH_CALUDE_line_contains_point_l1577_157770

theorem line_contains_point (k : ℝ) : 
  (1 - 3 * k * (1/3) = -2 * 4) ↔ (k = 9) := by sorry

end NUMINAMATH_CALUDE_line_contains_point_l1577_157770


namespace NUMINAMATH_CALUDE_division_value_problem_l1577_157766

theorem division_value_problem (x : ℝ) : 
  (1376 / x) - 160 = 12 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_value_problem_l1577_157766


namespace NUMINAMATH_CALUDE_factorial_products_squares_l1577_157701

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (fun i => i + 1)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem factorial_products_squares :
  (is_perfect_square (factorial 7 * factorial 8)) ∧
  (¬ is_perfect_square (factorial 5 * factorial 6)) ∧
  (¬ is_perfect_square (factorial 5 * factorial 7)) ∧
  (¬ is_perfect_square (factorial 6 * factorial 7)) ∧
  (¬ is_perfect_square (factorial 6 * factorial 8)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_products_squares_l1577_157701


namespace NUMINAMATH_CALUDE_xiaoxia_exceeds_xiaoming_l1577_157762

theorem xiaoxia_exceeds_xiaoming (n : ℕ) : 
  let xiaoxia_initial : ℤ := 52
  let xiaoming_initial : ℤ := 70
  let xiaoxia_monthly : ℤ := 15
  let xiaoming_monthly : ℤ := 12
  let xiaoxia_savings : ℤ := xiaoxia_initial + xiaoxia_monthly * n
  let xiaoming_savings : ℤ := xiaoming_initial + xiaoming_monthly * n
  xiaoxia_savings > xiaoming_savings ↔ 52 + 15 * n > 70 + 12 * n :=
by sorry

end NUMINAMATH_CALUDE_xiaoxia_exceeds_xiaoming_l1577_157762


namespace NUMINAMATH_CALUDE_system_solution_l1577_157738

theorem system_solution : 
  let x : ℚ := 8 / 47
  let y : ℚ := 138 / 47
  (7 * x = 10 - 3 * y) ∧ (4 * x = 5 * y - 14) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1577_157738


namespace NUMINAMATH_CALUDE_infinite_solutions_l1577_157748

theorem infinite_solutions (p : Nat) (hp : p.Prime) (hp_gt_7 : p > 7) :
  ∃ f : Nat → Nat,
    Function.Injective f ∧
    ∀ k : Nat, 
      (f k ≡ 1 [MOD 2016]) ∧ 
      (p ∣ (2^(f k) + f k)) :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_l1577_157748


namespace NUMINAMATH_CALUDE_expression_factorization_l1577_157703

theorem expression_factorization (x : ℝ) : 
  (18 * x^6 + 50 * x^4 - 8) - (2 * x^6 - 6 * x^4 - 8) = 8 * x^4 * (2 * x^2 + 7) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1577_157703


namespace NUMINAMATH_CALUDE_negative_inequality_l1577_157746

theorem negative_inequality (a b : ℝ) (h : a > b) : -b > -a := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_l1577_157746


namespace NUMINAMATH_CALUDE_original_purchase_cups_l1577_157706

/-- The cost of a single paper plate -/
def plate_cost : ℝ := sorry

/-- The cost of a single paper cup -/
def cup_cost : ℝ := sorry

/-- The number of paper cups in the original purchase -/
def num_cups : ℕ := sorry

/-- The total cost of 100 paper plates and some paper cups is $6.00 -/
axiom total_cost : 100 * plate_cost + num_cups * cup_cost = 6

/-- The total cost of 20 plates and 40 cups is $1.20 -/
axiom partial_cost : 20 * plate_cost + 40 * cup_cost = 1.2

theorem original_purchase_cups : num_cups = 200 := by
  sorry

end NUMINAMATH_CALUDE_original_purchase_cups_l1577_157706


namespace NUMINAMATH_CALUDE_soda_cost_l1577_157714

theorem soda_cost (alice_burgers alice_sodas alice_total bill_burgers bill_sodas bill_total : ℕ)
  (h_alice : 4 * alice_burgers + 3 * alice_sodas = alice_total)
  (h_bill : 3 * bill_burgers + 2 * bill_sodas = bill_total)
  (h_alice_total : alice_total = 500)
  (h_bill_total : bill_total = 370)
  (h_same_prices : alice_burgers = bill_burgers ∧ alice_sodas = bill_sodas) :
  alice_sodas = 20 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_l1577_157714


namespace NUMINAMATH_CALUDE_arctan_sum_equation_l1577_157791

theorem arctan_sum_equation (n : ℕ+) : 
  (Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/6) + Real.arctan (1/(n : ℝ)) = π/4) ↔ n = 57 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equation_l1577_157791


namespace NUMINAMATH_CALUDE_train_crossing_time_l1577_157719

/-- Proves that a train of given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 200 →
  train_speed_kmh = 180 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 4 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1577_157719


namespace NUMINAMATH_CALUDE_factorization_equality_l1577_157761

theorem factorization_equality (x y : ℝ) : 3 * x^2 + 6 * x * y + 3 * y^2 = 3 * (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1577_157761


namespace NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l1577_157710

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def digit_factorial_sum (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  factorial hundreds + factorial tens + factorial ones

theorem unique_three_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
  (∃ d, d ∈ [n / 100, (n / 10) % 10, n % 10] ∧ d = 6) ∧
  n = digit_factorial_sum n :=
by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l1577_157710


namespace NUMINAMATH_CALUDE_ryan_lost_leaves_l1577_157759

theorem ryan_lost_leaves (initial_leaves : ℕ) (broken_leaves : ℕ) (remaining_leaves : ℕ) : 
  initial_leaves = 89 → broken_leaves = 43 → remaining_leaves = 22 → 
  initial_leaves - (initial_leaves - remaining_leaves - broken_leaves) - broken_leaves = remaining_leaves :=
by
  sorry

end NUMINAMATH_CALUDE_ryan_lost_leaves_l1577_157759


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1577_157782

theorem coefficient_x_squared_in_expansion :
  let n : ℕ := 6
  let a : ℤ := 1
  let b : ℤ := -3
  (Finset.sum (Finset.range (n + 1)) (fun k => (n.choose k) * a^(n - k) * b^k * (if k = 2 then 1 else 0))) = 135 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1577_157782


namespace NUMINAMATH_CALUDE_leadership_structure_count_is_correct_l1577_157789

def tribe_size : ℕ := 15
def num_kings : ℕ := 1
def num_knights : ℕ := 2
def squires_per_knight : ℕ := 3

def leadership_structure_count : ℕ :=
  tribe_size * (tribe_size - 1).choose num_knights *
  (tribe_size - num_kings - num_knights).choose squires_per_knight *
  (tribe_size - num_kings - num_knights - squires_per_knight).choose squires_per_knight

theorem leadership_structure_count_is_correct :
  leadership_structure_count = 27392400 := by sorry

end NUMINAMATH_CALUDE_leadership_structure_count_is_correct_l1577_157789


namespace NUMINAMATH_CALUDE_investment_interest_proof_l1577_157777

/-- Calculates the total interest earned on an investment -/
def totalInterestEarned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * ((1 + rate) ^ years - 1)

/-- Proves that the total interest earned on $2,000 invested at 5% annually for 5 years is $552.56 -/
theorem investment_interest_proof :
  let principal := 2000
  let rate := 0.05
  let years := 5
  ∃ ε > 0, abs (totalInterestEarned principal rate years - 552.56) < ε :=
by sorry

end NUMINAMATH_CALUDE_investment_interest_proof_l1577_157777


namespace NUMINAMATH_CALUDE_circle_tangent_and_symmetric_points_l1577_157725

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y - 6 = 0

-- Define point M
def point_M : ℝ × ℝ := (-5, 11)

-- Define the line equation
def line_eq (m : ℝ) (x y : ℝ) : Prop := x + m*y + 4 = 0

-- Define the dot product of OP and OQ
def dot_product_OP_OQ (P Q : ℝ × ℝ) : ℝ := P.1 * Q.1 + P.2 * Q.2

-- Theorem statement
theorem circle_tangent_and_symmetric_points :
  ∃ (P Q : ℝ × ℝ) (m : ℝ),
    (∀ x y, circle_C x y ↔ (x + 1)^2 + (y - 3)^2 = 16) ∧
    (∀ x y, (x = -5 ∨ 3*x + 4*y - 29 = 0) ↔ 
      (circle_C x y ∧ ∃ t, x = point_M.1 + t * (x - point_M.1) ∧ 
                           y = point_M.2 + t * (y - point_M.2) ∧ 
                           t ≠ 0)) ∧
    circle_C P.1 P.2 ∧ 
    circle_C Q.1 Q.2 ∧
    line_eq m P.1 P.2 ∧
    line_eq m Q.1 Q.2 ∧
    dot_product_OP_OQ P Q = -7 ∧
    m = -1 ∧
    (∀ x y, (y = -x ∨ y = -x + 2) ↔ 
      (∃ t, x = P.1 + t * (Q.1 - P.1) ∧ y = P.2 + t * (Q.2 - P.2))) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_and_symmetric_points_l1577_157725


namespace NUMINAMATH_CALUDE_equation_solution_l1577_157707

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 1 →
  ((3 * x + 6) / (x^2 + 6 * x - 7) = (3 - x) / (x - 1)) ↔ (x = -5 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1577_157707


namespace NUMINAMATH_CALUDE_vector_perpendicular_condition_l1577_157753

/-- Given vectors a and b in R², if a + b is perpendicular to b, then the second component of a is 8. -/
theorem vector_perpendicular_condition (m : ℝ) : 
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (3, -2)
  (a.1 + b.1) * b.1 + (a.2 + b.2) * b.2 = 0 → m = 8 := by
sorry

end NUMINAMATH_CALUDE_vector_perpendicular_condition_l1577_157753


namespace NUMINAMATH_CALUDE_administrative_staff_sample_size_l1577_157771

/-- Represents the number of administrative staff to be drawn in a stratified sample -/
def administrative_staff_in_sample (total_population : ℕ) (sample_size : ℕ) (administrative_staff : ℕ) : ℕ :=
  (administrative_staff * sample_size) / total_population

/-- Theorem stating that the number of administrative staff to be drawn is 4 -/
theorem administrative_staff_sample_size :
  administrative_staff_in_sample 160 20 32 = 4 := by
  sorry

end NUMINAMATH_CALUDE_administrative_staff_sample_size_l1577_157771


namespace NUMINAMATH_CALUDE_remainder_17_63_mod_7_l1577_157752

theorem remainder_17_63_mod_7 : 17^63 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_63_mod_7_l1577_157752


namespace NUMINAMATH_CALUDE_vector_equation_l1577_157734

/-- Given vectors a, b, and c in ℝ², prove that if a = (5, 2), b = (-4, -3), 
    and 3a - 2b + c = 0, then c = (-23, -12). -/
theorem vector_equation (a b c : ℝ × ℝ) : 
  a = (5, 2) → 
  b = (-4, -3) → 
  3 • a - 2 • b + c = (0, 0) → 
  c = (-23, -12) := by sorry

end NUMINAMATH_CALUDE_vector_equation_l1577_157734


namespace NUMINAMATH_CALUDE_quadratic_polynomial_problem_l1577_157760

theorem quadratic_polynomial_problem (b c : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ →                             -- two distinct roots
  x₁^2 + b*x₁ + c = 0 →                 -- x₁ is a root
  x₂^2 + b*x₂ + c = 0 →                 -- x₂ is a root
  b + c + x₁ + x₂ = -3 →                -- sum condition
  b * c * x₁ * x₂ = 36 →                -- product condition
  b = 4 ∧ c = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_problem_l1577_157760


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1577_157790

theorem inequality_solution_set (x : ℝ) :
  (1 / (x^2 + 2) > 5 / x + 21 / 10) ↔ (-2 < x ∧ x < 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1577_157790


namespace NUMINAMATH_CALUDE_journey_feasibility_l1577_157736

/-- Proves that a journey can be completed in the given time at the given average speed -/
theorem journey_feasibility 
  (total_distance : ℝ) 
  (segment1 : ℝ) 
  (segment2 : ℝ) 
  (total_time : ℝ) 
  (average_speed : ℝ) 
  (h1 : total_distance = segment1 + segment2)
  (h2 : total_distance = 693)
  (h3 : segment1 = 420)
  (h4 : segment2 = 273)
  (h5 : total_time = 11)
  (h6 : average_speed = 63)
  : total_distance / average_speed = total_time :=
by sorry

#check journey_feasibility

end NUMINAMATH_CALUDE_journey_feasibility_l1577_157736


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1577_157718

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 0; 0, 1]
def B : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 0, 5]

theorem matrix_equation_solution :
  ∀ X : Matrix (Fin 2) (Fin 1) ℝ,
  B⁻¹ * A⁻¹ * X = !![5; 1] →
  X = !![28; 5] := by
sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l1577_157718


namespace NUMINAMATH_CALUDE_dave_tshirts_l1577_157756

def white_packs : ℕ := 3
def blue_packs : ℕ := 2
def red_packs : ℕ := 4
def green_packs : ℕ := 1

def white_per_pack : ℕ := 6
def blue_per_pack : ℕ := 4
def red_per_pack : ℕ := 5
def green_per_pack : ℕ := 3

def total_tshirts : ℕ := 
  white_packs * white_per_pack + 
  blue_packs * blue_per_pack + 
  red_packs * red_per_pack + 
  green_packs * green_per_pack

theorem dave_tshirts : total_tshirts = 49 := by
  sorry

end NUMINAMATH_CALUDE_dave_tshirts_l1577_157756


namespace NUMINAMATH_CALUDE_l1_fixed_point_min_distance_intersection_l1577_157799

-- Define the lines and circle
def l1 (m : ℝ) (x y : ℝ) : Prop := m * x - (m + 1) * y - 2 = 0
def l2 (x y : ℝ) : Prop := x + 2 * y + 1 = 0
def l3 (x y : ℝ) : Prop := y = x - 2

def circle_C (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 12}

-- Theorem 1: l1 always passes through (-2, -2)
theorem l1_fixed_point (m : ℝ) : l1 m (-2) (-2) := by sorry

-- Theorem 2: Minimum distance between intersection points
theorem min_distance_intersection :
  let center := (1, -1)  -- Intersection of l2 and l3
  ∃ (m : ℝ), ∀ (A B : ℝ × ℝ),
    A ∈ circle_C center → B ∈ circle_C center →
    l1 m A.1 A.2 → l1 m B.1 B.2 →
    (A.1 - B.1)^2 + (A.2 - B.2)^2 ≥ 8 := by sorry

end NUMINAMATH_CALUDE_l1_fixed_point_min_distance_intersection_l1577_157799


namespace NUMINAMATH_CALUDE_cube_sum_eq_prime_product_solution_l1577_157792

theorem cube_sum_eq_prime_product_solution :
  ∀ (x y p : ℕ+), 
    x^3 + y^3 = p * (x * y + p) ∧ Nat.Prime p.val →
    ((x = 8 ∧ y = 1 ∧ p = 19) ∨
     (x = 1 ∧ y = 8 ∧ p = 19) ∨
     (x = 7 ∧ y = 2 ∧ p = 13) ∨
     (x = 2 ∧ y = 7 ∧ p = 13) ∨
     (x = 5 ∧ y = 4 ∧ p = 7) ∨
     (x = 4 ∧ y = 5 ∧ p = 7)) :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_eq_prime_product_solution_l1577_157792


namespace NUMINAMATH_CALUDE_juliet_younger_than_ralph_l1577_157749

/-- Represents the ages of three siblings -/
structure SiblingAges where
  juliet : ℕ
  maggie : ℕ
  ralph : ℕ

/-- The conditions given in the problem -/
def problem_conditions (ages : SiblingAges) : Prop :=
  ages.juliet = ages.maggie + 3 ∧
  ages.juliet < ages.ralph ∧
  ages.juliet = 10 ∧
  ages.maggie + ages.ralph = 19

/-- The theorem to be proved -/
theorem juliet_younger_than_ralph (ages : SiblingAges) 
  (h : problem_conditions ages) : ages.ralph - ages.juliet = 2 := by
  sorry


end NUMINAMATH_CALUDE_juliet_younger_than_ralph_l1577_157749


namespace NUMINAMATH_CALUDE_triangle_possibilities_l1577_157740

-- Define a matchstick as a unit length
def matchstick_length : ℝ := 1

-- Define the total number of matchsticks
def total_matchsticks : ℕ := 12

-- Define a function to check if three lengths can form a triangle
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the types of triangles
def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∧ a ≠ c) ∨ (b = c ∧ b ≠ a) ∨ (c = a ∧ c ≠ b)

def is_equilateral (a b c : ℝ) : Prop :=
  a = b ∧ b = c

def is_right_angled (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

-- Theorem statement
theorem triangle_possibilities :
  ∃ (a b c : ℝ),
    a + b + c = total_matchsticks * matchstick_length ∧
    is_triangle a b c ∧
    (is_isosceles a b c ∧
     ∃ (d e f : ℝ), d + e + f = total_matchsticks * matchstick_length ∧
       is_triangle d e f ∧ is_equilateral d e f ∧
     ∃ (g h i : ℝ), g + h + i = total_matchsticks * matchstick_length ∧
       is_triangle g h i ∧ is_right_angled g h i) :=
by sorry

end NUMINAMATH_CALUDE_triangle_possibilities_l1577_157740


namespace NUMINAMATH_CALUDE_quadratic_root_property_l1577_157769

theorem quadratic_root_property (a b c x₁ x₂ : ℝ) (ha : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁^4 + 4*x₁^3*x₂ + 6*x₁^2*x₂^2 + 4*x₁*x₂^3 + x₂^4)^(1/4) = -b/a := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l1577_157769


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l1577_157797

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬p x) :=
by sorry

theorem negation_of_inequality :
  (¬∃ x : ℝ, 2^x ≥ 2*x + 1) ↔ (∀ x : ℝ, 2^x < 2*x + 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l1577_157797


namespace NUMINAMATH_CALUDE_bird_problem_equations_l1577_157787

/-- Represents the cost of each type of bird in coins -/
structure BirdCosts where
  rooster : ℚ
  hen : ℚ
  chick : ℚ

/-- Represents the quantities of each type of bird -/
structure BirdQuantities where
  roosters : ℕ
  hens : ℕ
  chicks : ℕ

/-- The problem constraints -/
def bird_problem (costs : BirdCosts) (quantities : BirdQuantities) : Prop :=
  costs.rooster = 5 ∧
  costs.hen = 3 ∧
  costs.chick = 1/3 ∧
  quantities.roosters = 8 ∧
  quantities.roosters + quantities.hens + quantities.chicks = 100

/-- The system of equations representing the problem -/
def problem_equations (costs : BirdCosts) (quantities : BirdQuantities) : Prop :=
  costs.rooster * quantities.roosters + costs.hen * quantities.hens + costs.chick * quantities.chicks = 100 ∧
  quantities.roosters + quantities.hens + quantities.chicks = 100

/-- Theorem stating that the problem constraints imply the system of equations -/
theorem bird_problem_equations (costs : BirdCosts) (quantities : BirdQuantities) :
  bird_problem costs quantities → problem_equations costs quantities :=
by
  sorry


end NUMINAMATH_CALUDE_bird_problem_equations_l1577_157787


namespace NUMINAMATH_CALUDE_existence_of_n_l1577_157732

theorem existence_of_n : ∃ n : ℕ+, 
  ∀ k : ℕ, 2 ≤ k ∧ k ≤ 10 → 
    ∃ p : ℕ+, 
      (↑p + 2015/10000 : ℝ)^k < n ∧ n < (↑p + 2016/10000 : ℝ)^k := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_l1577_157732


namespace NUMINAMATH_CALUDE_trading_cards_theorem_l1577_157712

/-- The number of cards in a partially filled box -/
def partially_filled_box (total_cards : ℕ) (cards_per_box : ℕ) : ℕ :=
  total_cards % cards_per_box

theorem trading_cards_theorem :
  let pokemon_cards := 65
  let magic_cards := 55
  let yugioh_cards := 40
  let pokemon_per_box := 8
  let magic_per_box := 10
  let yugioh_per_box := 12
  (partially_filled_box pokemon_cards pokemon_per_box = 1) ∧
  (partially_filled_box magic_cards magic_per_box = 5) ∧
  (partially_filled_box yugioh_cards yugioh_per_box = 4) :=
by sorry

end NUMINAMATH_CALUDE_trading_cards_theorem_l1577_157712


namespace NUMINAMATH_CALUDE_parking_space_difference_l1577_157795

/-- Represents a parking garage with four levels -/
structure ParkingGarage where
  level1 : Nat
  level2 : Nat
  level3 : Nat
  level4 : Nat

/-- Theorem stating the difference in parking spaces between the third and fourth levels -/
theorem parking_space_difference (garage : ParkingGarage) : 
  garage.level1 = 90 →
  garage.level2 = garage.level1 + 8 →
  garage.level3 = garage.level2 + 12 →
  garage.level1 + garage.level2 + garage.level3 + garage.level4 = 299 →
  garage.level3 - garage.level4 = 109 := by
  sorry

end NUMINAMATH_CALUDE_parking_space_difference_l1577_157795


namespace NUMINAMATH_CALUDE_abs_inequality_necessary_not_sufficient_l1577_157785

theorem abs_inequality_necessary_not_sufficient (x : ℝ) :
  (x * (x - 2) < 0 → abs (x - 1) < 2) ∧
  ¬(abs (x - 1) < 2 → x * (x - 2) < 0) := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_necessary_not_sufficient_l1577_157785


namespace NUMINAMATH_CALUDE_acid_solution_volume_l1577_157774

/-- Given a volume of pure acid in a solution with a known concentration,
    calculate the total volume of the solution. -/
theorem acid_solution_volume (pure_acid : ℝ) (concentration : ℝ) 
    (h1 : pure_acid = 4.8)
    (h2 : concentration = 0.4) : 
    pure_acid / concentration = 12 := by
  sorry

end NUMINAMATH_CALUDE_acid_solution_volume_l1577_157774


namespace NUMINAMATH_CALUDE_interval_intersection_l1577_157733

theorem interval_intersection (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1/2 < x ∧ x < 0.6) := by
  sorry

end NUMINAMATH_CALUDE_interval_intersection_l1577_157733


namespace NUMINAMATH_CALUDE_expand_expression_l1577_157750

theorem expand_expression (a b : ℝ) : (a - 2) * (a - 2*b) = a^2 - 2*a + 4*b := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1577_157750


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l1577_157767

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem quadratic_inequality_negation : 
  (¬ ∃ x : ℝ, x^2 + 4*x + 5 ≤ 0) ↔ (∀ x : ℝ, x^2 + 4*x + 5 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l1577_157767


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l1577_157726

theorem square_perimeter_ratio (d D s S : ℝ) : 
  d > 0 → s > 0 → 
  d = s * Real.sqrt 2 → 
  D = S * Real.sqrt 2 → 
  D = 11 * d → 
  (4 * S) / (4 * s) = 11 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l1577_157726


namespace NUMINAMATH_CALUDE_square_of_two_minus_x_l1577_157722

theorem square_of_two_minus_x (x : ℝ) : (2 - x)^2 = 4 - 4*x + x^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_two_minus_x_l1577_157722


namespace NUMINAMATH_CALUDE_x_intercepts_count_l1577_157709

theorem x_intercepts_count : 
  let f (x : ℝ) := (x - 3) * (x^2 + 4*x + 4)
  ∃ (a b : ℝ), a ≠ b ∧ 
    (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l1577_157709


namespace NUMINAMATH_CALUDE_equation_solutions_count_l1577_157720

theorem equation_solutions_count : 
  (Finset.filter (fun p : ℕ × ℕ => 4 * p.1 + 7 * p.2 = 588 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 589) (Finset.range 589))).card = 21 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l1577_157720


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1577_157702

/-- The sum of a finite geometric series -/
def geometricSum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_series_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometricSum a r n = 4/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1577_157702


namespace NUMINAMATH_CALUDE_plant_arrangement_theorem_l1577_157798

/-- Represents the number of ways to arrange plants under lamps -/
def plant_arrangement_count : ℕ :=
  let basil_count : ℕ := 3
  let aloe_count : ℕ := 2
  let white_lamp_count : ℕ := 3
  let red_lamp_count : ℕ := 3
  sorry

/-- Theorem stating that the number of plant arrangements is 128 -/
theorem plant_arrangement_theorem : plant_arrangement_count = 128 := by
  sorry

end NUMINAMATH_CALUDE_plant_arrangement_theorem_l1577_157798


namespace NUMINAMATH_CALUDE_particle_speeds_l1577_157775

-- Define the distance between points A and B in centimeters
def distance : ℝ := 301

-- Define the time when m2 starts moving after m1 leaves A
def start_time : ℝ := 11

-- Define the times of the two meetings after m2 starts moving
def first_meeting : ℝ := 10
def second_meeting : ℝ := 45

-- Define the speeds of particles m1 and m2
def speed_m1 : ℝ := 11
def speed_m2 : ℝ := 7

-- Theorem statement
theorem particle_speeds :
  -- Condition: At the first meeting, the total distance covered equals the initial distance
  (distance - start_time * speed_m1 = first_meeting * (speed_m1 + speed_m2)) ∧
  -- Condition: The relative movement between the two meetings
  (2 * first_meeting * speed_m2 = (second_meeting - first_meeting) * (speed_m1 - speed_m2)) →
  -- Conclusion: The speeds are correct
  speed_m1 = 11 ∧ speed_m2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_particle_speeds_l1577_157775


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_equality_l1577_157739

theorem quadratic_roots_sum_equality (b₁ b₂ b₃ : ℝ) : ∃ (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ),
  (x₁ = (-b₁ + 1) / 2 ∧ y₁ = (-b₁ - 1) / 2) ∧
  (x₂ = (-b₂ + 2) / 2 ∧ y₂ = (-b₂ - 2) / 2) ∧
  (x₃ = (-b₃ + 3) / 2 ∧ y₃ = (-b₃ - 3) / 2) ∧
  x₁ + x₂ + x₃ = y₁ + y₂ + y₃ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_equality_l1577_157739


namespace NUMINAMATH_CALUDE_smallest_three_digit_congruence_l1577_157751

theorem smallest_three_digit_congruence :
  ∃ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    (75 * n) % 345 = 225 ∧
    (∀ m : ℕ, 100 ≤ m ∧ m < n → (75 * m) % 345 ≠ 225) ∧
    n = 118 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_congruence_l1577_157751


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l1577_157796

/-- A trinomial ax^2 + bx + c is a perfect square if there exist real numbers p and q
    such that ax^2 + bx + c = (px + q)^2 for all real x. -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

/-- If x^2 + kx + 9 is a perfect square trinomial, then k = 6 or k = -6. -/
theorem perfect_square_trinomial_condition (k : ℝ) :
  is_perfect_square_trinomial 1 k 9 → k = 6 ∨ k = -6 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l1577_157796


namespace NUMINAMATH_CALUDE_divisibility_by_five_l1577_157755

theorem divisibility_by_five (a b : ℕ) (h : 5 ∣ (a * b)) : 5 ∣ a ∨ 5 ∣ b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l1577_157755


namespace NUMINAMATH_CALUDE_last_defective_on_fifth_draw_l1577_157700

def number_of_arrangements (n_total : ℕ) (n_genuine : ℕ) (n_defective : ℕ) : ℕ :=
  (n_total.choose (n_defective - 1)) * (n_defective.factorial) * n_genuine

theorem last_defective_on_fifth_draw :
  let n_total := 9
  let n_genuine := 5
  let n_defective := 4
  let n_draws := 5
  number_of_arrangements n_draws n_genuine n_defective = 480 :=
by sorry

end NUMINAMATH_CALUDE_last_defective_on_fifth_draw_l1577_157700


namespace NUMINAMATH_CALUDE_area_of_gray_part_l1577_157773

/-- Given two overlapping rectangles, prove the area of the gray part -/
theorem area_of_gray_part (rect1_width rect1_height rect2_width rect2_height black_area : ℕ) 
  (h1 : rect1_width = 8)
  (h2 : rect1_height = 10)
  (h3 : rect2_width = 12)
  (h4 : rect2_height = 9)
  (h5 : black_area = 37) : 
  rect2_width * rect2_height - (rect1_width * rect1_height - black_area) = 65 := by
  sorry

#check area_of_gray_part

end NUMINAMATH_CALUDE_area_of_gray_part_l1577_157773


namespace NUMINAMATH_CALUDE_parts_per_day_to_finish_ahead_l1577_157728

theorem parts_per_day_to_finish_ahead (total_parts : ℕ) (total_days : ℕ) (initial_days : ℕ) (initial_parts_per_day : ℕ) :
  total_parts = 408 →
  total_days = 15 →
  initial_days = 3 →
  initial_parts_per_day = 24 →
  ∃ (x : ℕ), x = 29 ∧ 
    (initial_days * initial_parts_per_day + (total_days - initial_days) * x > total_parts) ∧
    ∀ (y : ℕ), y < x → (initial_days * initial_parts_per_day + (total_days - initial_days) * y ≤ total_parts) :=
by sorry

end NUMINAMATH_CALUDE_parts_per_day_to_finish_ahead_l1577_157728


namespace NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l1577_157772

theorem floor_plus_self_unique_solution :
  ∃! s : ℝ, ⌊s⌋ + s = 22.7 :=
by sorry

end NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l1577_157772


namespace NUMINAMATH_CALUDE_max_guaranteed_amount_100_cards_l1577_157754

/-- Represents a set of bank cards with amounts from 1 to n rubles -/
def BankCards (n : ℕ) := Finset (Fin n)

/-- The strategy of requesting a fixed amount from each card -/
def Strategy (n : ℕ) := ℕ

/-- The amount guaranteed to be collected given a strategy -/
def guaranteedAmount (n : ℕ) (s : Strategy n) : ℕ := sorry

/-- The maximum guaranteed amount that can be collected -/
def maxGuaranteedAmount (n : ℕ) : ℕ := sorry

theorem max_guaranteed_amount_100_cards :
  maxGuaranteedAmount 100 = 2550 := by sorry

end NUMINAMATH_CALUDE_max_guaranteed_amount_100_cards_l1577_157754


namespace NUMINAMATH_CALUDE_inequalities_given_sum_positive_l1577_157731

/-- Given two real numbers a and b such that a + b > 0, 
    the following statements are true:
    1. a^5 * b^2 + a^4 * b^3 ≥ 0
    2. a^21 + b^21 > 0
    3. (a+2)*(b+2) > a*b
-/
theorem inequalities_given_sum_positive (a b : ℝ) (h : a + b > 0) :
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧ 
  (a^21 + b^21 > 0) ∧ 
  ((a+2)*(b+2) > a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_given_sum_positive_l1577_157731


namespace NUMINAMATH_CALUDE_negative_reals_inequality_l1577_157783

theorem negative_reals_inequality (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (Real.sqrt (a / (b + c)) + 1 / Real.sqrt 2) ^ 2 +
  (Real.sqrt (b / (c + a)) + 1 / Real.sqrt 2) ^ 2 +
  (Real.sqrt (c / (a + b)) + 1 / Real.sqrt 2) ^ 2 ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_reals_inequality_l1577_157783


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_6_divisible_by_11_l1577_157717

def is_smallest_positive_integer_ending_in_6_divisible_by_11 (n : ℕ) : Prop :=
  n > 0 ∧ n % 10 = 6 ∧ n % 11 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 6 → m % 11 = 0 → m ≥ n

theorem smallest_positive_integer_ending_in_6_divisible_by_11 :
  is_smallest_positive_integer_ending_in_6_divisible_by_11 116 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_6_divisible_by_11_l1577_157717


namespace NUMINAMATH_CALUDE_freds_allowance_l1577_157729

/-- Proves that Fred's weekly allowance is 16 dollars given the problem conditions -/
theorem freds_allowance (spent_on_movies : ℝ) (car_wash_earnings : ℝ) (final_amount : ℝ) :
  spent_on_movies = car_wash_earnings - 6 →
  final_amount = 14 →
  spent_on_movies = 8 →
  spent_on_movies * 2 = 16 :=
by
  sorry

#check freds_allowance

end NUMINAMATH_CALUDE_freds_allowance_l1577_157729


namespace NUMINAMATH_CALUDE_inequality_solution_l1577_157784

theorem inequality_solution (x y : ℝ) : 
  2^y - 2 * Real.cos x + Real.sqrt (y - x^2 - 1) ≤ 0 ↔ x = 0 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1577_157784


namespace NUMINAMATH_CALUDE_v_closed_under_multiplication_l1577_157794

/-- The set of cubes of positive integers -/
def v : Set ℕ := {n | ∃ m : ℕ+, n = m^3}

/-- Proof that v is closed under multiplication -/
theorem v_closed_under_multiplication :
  ∀ a b : ℕ, a ∈ v → b ∈ v → (a * b) ∈ v := by
  sorry

end NUMINAMATH_CALUDE_v_closed_under_multiplication_l1577_157794


namespace NUMINAMATH_CALUDE_students_needed_to_fill_buses_l1577_157705

theorem students_needed_to_fill_buses (total_students : ℕ) (bus_capacity : ℕ) : 
  total_students = 254 → bus_capacity = 30 → 
  (((total_students + 16) / bus_capacity : ℕ) * bus_capacity = total_students + 16) ∧
  (((total_students + 15) / bus_capacity : ℕ) * bus_capacity < total_students + 15) := by
  sorry


end NUMINAMATH_CALUDE_students_needed_to_fill_buses_l1577_157705


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1577_157742

theorem sqrt_product_equality : 3 * Real.sqrt 2 * Real.sqrt 6 = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1577_157742


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1577_157724

theorem solution_set_quadratic_inequality :
  let f : ℝ → ℝ := λ x => -x^2 + 3*x - 2
  {x : ℝ | f x ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by
sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1577_157724


namespace NUMINAMATH_CALUDE_symmetric_point_correct_l1577_157730

/-- The line of symmetry -/
def line_of_symmetry (x y : ℝ) : Prop := x + 3 * y - 10 = 0

/-- The original point -/
def original_point : ℝ × ℝ := (3, 9)

/-- The symmetric point -/
def symmetric_point : ℝ × ℝ := (-1, -3)

/-- Predicate to check if a point is symmetric to another point with respect to a line -/
def is_symmetric (p1 p2 : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  line midpoint.1 midpoint.2 ∧
  (p2.2 - p1.2) * (p2.1 - p1.1) = -(1 / 3)

theorem symmetric_point_correct : 
  is_symmetric original_point symmetric_point line_of_symmetry :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_correct_l1577_157730


namespace NUMINAMATH_CALUDE_expression_values_l1577_157747

theorem expression_values : 
  (0.64^(-1/2) - (-1/8)^0 + 8^(2/3) + (9/16)^(1/2) = 6) ∧ 
  (Real.log 2^2 + Real.log 2 * Real.log 5 + Real.log 5 = 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l1577_157747


namespace NUMINAMATH_CALUDE_polygon_interior_exterior_angles_equal_l1577_157778

theorem polygon_interior_exterior_angles_equal (n : ℕ) : 
  (n - 2) * 180 = 360 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_exterior_angles_equal_l1577_157778


namespace NUMINAMATH_CALUDE_train_speed_with_stops_l1577_157741

/-- Calculates the average speed of a train with stoppages -/
def averageSpeedWithStoppages (distanceKm : ℝ) (speedWithoutStops : ℝ) (stopTimePerHour : ℝ) : ℝ :=
  let movingTimeRatio := 1 - stopTimePerHour
  speedWithoutStops * movingTimeRatio

theorem train_speed_with_stops :
  ∀ (distanceKm : ℝ),
    distanceKm > 0 →
    averageSpeedWithStoppages distanceKm 250 0.5 = 125 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_with_stops_l1577_157741


namespace NUMINAMATH_CALUDE_x_squared_minus_four_y_squared_l1577_157727

theorem x_squared_minus_four_y_squared (x y : ℝ) 
  (eq1 : x + 2*y = 4) 
  (eq2 : x - 2*y = -1) : 
  x^2 - 4*y^2 = -4 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_four_y_squared_l1577_157727


namespace NUMINAMATH_CALUDE_product_of_three_terms_l1577_157757

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem product_of_three_terms 
  (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : a 5 = 4) : 
  a 4 * a 5 * a 6 = 64 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_terms_l1577_157757


namespace NUMINAMATH_CALUDE_otimes_nested_l1577_157723

/-- Custom binary operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^2 - 2*y

/-- Theorem stating the result of k ⊗ (k ⊗ k) -/
theorem otimes_nested (k : ℝ) : otimes k (otimes k k) = -k^2 + 4*k := by
  sorry

end NUMINAMATH_CALUDE_otimes_nested_l1577_157723


namespace NUMINAMATH_CALUDE_one_student_in_all_activities_l1577_157713

/-- Represents the number of students participating in various combinations of activities -/
structure ActivityParticipation where
  total : ℕ
  chess : ℕ
  soccer : ℕ
  music : ℕ
  atLeastTwo : ℕ

/-- The conditions of the problem -/
def clubConditions : ActivityParticipation where
  total := 30
  chess := 15
  soccer := 18
  music := 12
  atLeastTwo := 14

/-- Theorem stating that exactly one student participates in all three activities -/
theorem one_student_in_all_activities (ap : ActivityParticipation) 
  (h1 : ap = clubConditions) : 
  ∃! x : ℕ, x = (ap.chess + ap.soccer + ap.music) - (2 * ap.atLeastTwo) + ap.total - ap.atLeastTwo :=
by sorry

end NUMINAMATH_CALUDE_one_student_in_all_activities_l1577_157713


namespace NUMINAMATH_CALUDE_unpainted_area_proof_l1577_157744

def board_width_1 : ℝ := 4
def board_width_2 : ℝ := 6
def intersection_angle : ℝ := 60

theorem unpainted_area_proof :
  let parallelogram_base := board_width_2 / Real.sin (intersection_angle * Real.pi / 180)
  let parallelogram_height := board_width_1
  parallelogram_base * parallelogram_height = 16 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_area_proof_l1577_157744


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l1577_157735

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- A predicate that checks if a number is a three-digit number -/
def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem unique_three_digit_number : 
  ∃! n : ℕ, isThreeDigit n ∧ n = 12 * sumOfDigits n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l1577_157735
