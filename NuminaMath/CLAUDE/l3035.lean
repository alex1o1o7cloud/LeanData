import Mathlib

namespace NUMINAMATH_CALUDE_cubic_root_sum_l3035_303535

theorem cubic_root_sum (a b c d : ℝ) (ha : a ≠ 0) : 
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 4 ∨ x = -3) →
  (b + c) / a = -13 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3035_303535


namespace NUMINAMATH_CALUDE_q_equals_six_l3035_303516

/-- Represents a digit from 4 to 9 -/
def Digit := {n : ℕ // 4 ≤ n ∧ n ≤ 9}

/-- The theorem stating that Q must be 6 given the conditions -/
theorem q_equals_six 
  (P Q R S T U : Digit) 
  (unique : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ P ≠ U ∧
            Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ Q ≠ U ∧
            R ≠ S ∧ R ≠ T ∧ R ≠ U ∧
            S ≠ T ∧ S ≠ U ∧
            T ≠ U)
  (sum_constraint : P.val + Q.val + S.val + 
                    T.val + U.val + R.val + 
                    P.val + T.val + S.val + 
                    R.val + Q.val + S.val + 
                    P.val + U.val = 100) : 
  Q.val = 6 := by
  sorry

end NUMINAMATH_CALUDE_q_equals_six_l3035_303516


namespace NUMINAMATH_CALUDE_total_profit_is_100_l3035_303519

/-- Calculates the total profit given investments and A's profit share -/
def calculate_total_profit (a_investment : ℕ) (a_months : ℕ) (b_investment : ℕ) (b_months : ℕ) (a_profit_share : ℕ) : ℕ :=
  let a_investment_ratio := a_investment * a_months
  let b_investment_ratio := b_investment * b_months
  let total_investment_ratio := a_investment_ratio + b_investment_ratio
  (a_profit_share * total_investment_ratio) / a_investment_ratio

/-- Proves that the total profit is $100 given the specified investments and A's profit share -/
theorem total_profit_is_100 :
  calculate_total_profit 100 12 200 6 50 = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_100_l3035_303519


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l3035_303591

/-- Given a sphere and a right circular cone, if the volume of the cone is one-third
    that of the sphere, and the radius of the base of the cone is twice the radius of the sphere,
    then the ratio of the altitude of the cone to the radius of its base is 1/6. -/
theorem cone_sphere_ratio (r : ℝ) (h : ℝ) (h_pos : 0 < r) : 
  (4 / 3 * Real.pi * r^3) / 3 = 1 / 3 * Real.pi * (2 * r)^2 * h →
  h / (2 * r) = 1 / 6 := by
  sorry

#check cone_sphere_ratio

end NUMINAMATH_CALUDE_cone_sphere_ratio_l3035_303591


namespace NUMINAMATH_CALUDE_sqrt_29_between_consecutive_integers_product_l3035_303571

theorem sqrt_29_between_consecutive_integers_product (n m : ℤ) :
  n < m ∧ m = n + 1 ∧ (n : ℝ) < Real.sqrt 29 ∧ Real.sqrt 29 < (m : ℝ) →
  n * m = 30 :=
sorry

end NUMINAMATH_CALUDE_sqrt_29_between_consecutive_integers_product_l3035_303571


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l3035_303507

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem no_prime_roots_for_quadratic :
  ¬∃ k : ℤ, ∃ p q : ℕ, 
    is_prime p ∧ is_prime q ∧ 
    p ≠ q ∧
    (p : ℤ) * (q : ℤ) = k ∧ 
    (p : ℤ) + (q : ℤ) = 58 :=
sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l3035_303507


namespace NUMINAMATH_CALUDE_plate_on_square_table_l3035_303558

/-- The distance from the edge of a round plate to the bottom edge of a square table -/
def plate_to_bottom_edge (top_margin left_margin right_margin : ℝ) : ℝ :=
  left_margin + right_margin - top_margin

theorem plate_on_square_table 
  (top_margin left_margin right_margin : ℝ) 
  (h_top : top_margin = 10)
  (h_left : left_margin = 63)
  (h_right : right_margin = 20) :
  plate_to_bottom_edge top_margin left_margin right_margin = 73 := by
sorry

end NUMINAMATH_CALUDE_plate_on_square_table_l3035_303558


namespace NUMINAMATH_CALUDE_least_possible_difference_l3035_303596

theorem least_possible_difference (x y z : ℤ) : 
  x < y → y < z → 
  y - x > 5 → 
  Even x → 
  Odd y → 
  Odd z → 
  (∀ d : ℤ, d = z - x → d ≥ 9) ∧ (∃ x' y' z' : ℤ, x' < y' ∧ y' < z' ∧ y' - x' > 5 ∧ Even x' ∧ Odd y' ∧ Odd z' ∧ z' - x' = 9) :=
by sorry

end NUMINAMATH_CALUDE_least_possible_difference_l3035_303596


namespace NUMINAMATH_CALUDE_closest_fraction_l3035_303532

def medals_won : ℚ := 23 / 120

def fractions : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]

theorem closest_fraction :
  ∃ (x : ℚ), x ∈ fractions ∧
  ∀ (y : ℚ), y ∈ fractions → |medals_won - x| ≤ |medals_won - y| ∧
  x = 1/5 :=
sorry

end NUMINAMATH_CALUDE_closest_fraction_l3035_303532


namespace NUMINAMATH_CALUDE_red_balls_count_l3035_303546

theorem red_balls_count (yellow_balls : ℕ) (total_balls : ℕ) 
  (yellow_prob : ℚ) (h1 : yellow_balls = 4) 
  (h2 : yellow_prob = 1 / 5) 
  (h3 : yellow_prob = yellow_balls / total_balls) : 
  total_balls - yellow_balls = 16 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l3035_303546


namespace NUMINAMATH_CALUDE_weeks_of_papayas_l3035_303567

def jake_papayas_per_week : ℕ := 3
def brother_papayas_per_week : ℕ := 5
def father_papayas_per_week : ℕ := 4
def total_papayas_bought : ℕ := 48

theorem weeks_of_papayas : 
  (total_papayas_bought / (jake_papayas_per_week + brother_papayas_per_week + father_papayas_per_week) = 4) := by
  sorry

end NUMINAMATH_CALUDE_weeks_of_papayas_l3035_303567


namespace NUMINAMATH_CALUDE_matrix_commutation_fraction_l3035_303553

/-- Given two matrices A and B, where A is fixed and B has variable entries,
    prove that if A * B = B * A and 3b ≠ c, then (a - d) / (c - 3b) = 1. -/
theorem matrix_commutation_fraction (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 3, 4]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A * B = B * A) → (3 * b ≠ c) → ((a - d) / (c - 3 * b) = 1) := by
  sorry

end NUMINAMATH_CALUDE_matrix_commutation_fraction_l3035_303553


namespace NUMINAMATH_CALUDE_absent_men_count_l3035_303548

/-- Proves the number of absent men in a work group --/
theorem absent_men_count (total_men : ℕ) (original_days : ℕ) (actual_days : ℕ) 
  (h1 : total_men = 20)
  (h2 : original_days = 20)
  (h3 : actual_days = 40)
  (h4 : total_men * original_days = (total_men - absent_men) * actual_days) :
  absent_men = 10 := by
  sorry

#check absent_men_count

end NUMINAMATH_CALUDE_absent_men_count_l3035_303548


namespace NUMINAMATH_CALUDE_reciprocal_of_neg_tan_60_l3035_303555

theorem reciprocal_of_neg_tan_60 :
  (-(Real.tan (60 * π / 180)))⁻¹ = -((3 : ℝ).sqrt / 3) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_neg_tan_60_l3035_303555


namespace NUMINAMATH_CALUDE_C_power_50_is_identity_l3035_303540

def C : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 2],
    ![-8, -5]]

theorem C_power_50_is_identity :
  C ^ 50 = (1 : Matrix (Fin 2) (Fin 2) ℤ) := by
  sorry

end NUMINAMATH_CALUDE_C_power_50_is_identity_l3035_303540


namespace NUMINAMATH_CALUDE_range_of_a_l3035_303592

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x > 0 ∧ 2 * x * (x - a) < 1) → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3035_303592


namespace NUMINAMATH_CALUDE_bread_slices_left_l3035_303523

theorem bread_slices_left (
  initial_slices : Nat) 
  (days_in_week : Nat)
  (slices_per_sandwich : Nat)
  (extra_sandwiches : Nat) :
  initial_slices = 22 →
  days_in_week = 7 →
  slices_per_sandwich = 2 →
  extra_sandwiches = 1 →
  initial_slices - (days_in_week + extra_sandwiches) * slices_per_sandwich = 6 :=
by sorry

end NUMINAMATH_CALUDE_bread_slices_left_l3035_303523


namespace NUMINAMATH_CALUDE_solution_to_equation_l3035_303503

theorem solution_to_equation : ∃ x : ℝ, 4*x + 9*x = 360 - 9*(x - 4) ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3035_303503


namespace NUMINAMATH_CALUDE_max_triangle_side_l3035_303508

theorem max_triangle_side (a b c : ℕ) : 
  a < b → b < c →  -- Ensure different side lengths
  a + b + c = 24 →  -- Perimeter condition
  a + b > c →  -- Triangle inequality
  a + c > b →  -- Triangle inequality
  b + c > a →  -- Triangle inequality
  c ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_side_l3035_303508


namespace NUMINAMATH_CALUDE_sphere_radius_is_correct_l3035_303583

/-- A truncated cone with a tangent sphere -/
structure TruncatedConeWithSphere where
  r_bottom : ℝ
  r_top : ℝ
  sphere_radius : ℝ
  is_tangent : Bool

/-- The specific truncated cone with tangent sphere from the problem -/
def problem_cone : TruncatedConeWithSphere :=
  { r_bottom := 20
  , r_top := 5
  , sphere_radius := 10
  , is_tangent := true }

/-- Theorem stating that the sphere radius is correct -/
theorem sphere_radius_is_correct (c : TruncatedConeWithSphere) :
  c.r_bottom = 20 ∧ c.r_top = 5 ∧ c.is_tangent = true → c.sphere_radius = 10 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_is_correct_l3035_303583


namespace NUMINAMATH_CALUDE_problem_solution_l3035_303544

/-- The function f(x) defined in the problem -/
def f (a b x : ℝ) : ℝ := x^3 + (1-a)*x^2 - a*(a+2)*x + b

/-- The derivative of f(x) with respect to x -/
def f_derivative (a x : ℝ) : ℝ := 3*x^2 + 2*(1-a)*x - a*(a+2)

theorem problem_solution :
  (∀ a b : ℝ, f a b 0 = 0 ∧ f_derivative a 0 = -3 → (a = -3 ∨ a = 1) ∧ b = 0) ∧
  (∀ a b : ℝ, (∃ x y : ℝ, x ≠ y ∧ f_derivative a x = 0 ∧ f_derivative a y = 0) →
    a < -1/2 ∨ a > -1/2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3035_303544


namespace NUMINAMATH_CALUDE_complex_number_location_l3035_303586

theorem complex_number_location (z : ℂ) (h : (3 - 2*I)*z = 4 + 3*I) :
  0 < z.re ∧ 0 < z.im :=
sorry

end NUMINAMATH_CALUDE_complex_number_location_l3035_303586


namespace NUMINAMATH_CALUDE_line_perp_to_plane_perp_to_line_in_plane_l3035_303501

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- State the theorem
theorem line_perp_to_plane_perp_to_line_in_plane
  (a b : Line) (α : Plane)
  (h1 : perp_line_plane a α)
  (h2 : subset_line_plane b α) :
  perp_line_line a b :=
sorry

end NUMINAMATH_CALUDE_line_perp_to_plane_perp_to_line_in_plane_l3035_303501


namespace NUMINAMATH_CALUDE_cube_difference_l3035_303539

theorem cube_difference (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 35) : a^3 - b^3 = 200 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l3035_303539


namespace NUMINAMATH_CALUDE_nickels_count_l3035_303584

/-- Proves that given 70 coins consisting of nickels and dimes with a total value of $5.55, the number of nickels is 29. -/
theorem nickels_count (total_coins : ℕ) (total_value : ℚ) (nickels : ℕ) (dimes : ℕ) :
  total_coins = 70 →
  total_value = 555/100 →
  total_coins = nickels + dimes →
  total_value = (5/100 : ℚ) * nickels + (10/100 : ℚ) * dimes →
  nickels = 29 := by
sorry

end NUMINAMATH_CALUDE_nickels_count_l3035_303584


namespace NUMINAMATH_CALUDE_successive_discounts_equivalence_l3035_303522

/-- Proves that three successive discounts are equivalent to a single discount -/
theorem successive_discounts_equivalence (original_price : ℝ) 
  (discount1 discount2 discount3 : ℝ) (equivalent_discount : ℝ) : 
  original_price = 60 ∧ 
  discount1 = 0.15 ∧ 
  discount2 = 0.10 ∧ 
  discount3 = 0.20 ∧ 
  equivalent_discount = 0.388 →
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = 
  original_price * (1 - equivalent_discount) :=
by sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalence_l3035_303522


namespace NUMINAMATH_CALUDE_stewart_farm_horse_food_consumption_l3035_303595

/-- Calculates the total daily horse food consumption on the Stewart farm -/
theorem stewart_farm_horse_food_consumption
  (sheep_to_horse_ratio : ℚ)
  (sheep_count : ℕ)
  (food_per_horse : ℕ) :
  sheep_to_horse_ratio = 5 / 7 →
  sheep_count = 40 →
  food_per_horse = 230 →
  (sheep_count * (7 / 5) : ℚ).num * food_per_horse = 12880 := by
  sorry

#eval (40 * (7 / 5) : ℚ).num * 230

end NUMINAMATH_CALUDE_stewart_farm_horse_food_consumption_l3035_303595


namespace NUMINAMATH_CALUDE_complex_arithmetic_problem_l3035_303513

theorem complex_arithmetic_problem : ((6^2 - 4^2) + 2)^3 / 2 = 5324 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_problem_l3035_303513


namespace NUMINAMATH_CALUDE_parallel_segment_length_l3035_303502

/-- A trapezoid with bases a and b -/
structure Trapezoid (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- A line segment within a trapezoid -/
def ParallelSegment (t : Trapezoid a b) := ℝ

/-- The property that a line divides a trapezoid into two similar trapezoids -/
def DividesSimilarly (t : Trapezoid a b) (s : ParallelSegment t) : Prop :=
  sorry

/-- Theorem: If a line parallel to the bases divides a trapezoid into two similar trapezoids,
    then the length of the segment is the square root of the product of the bases -/
theorem parallel_segment_length (a b : ℝ) (t : Trapezoid a b) (s : ParallelSegment t) :
  DividesSimilarly t s → s = Real.sqrt (a * b) :=
sorry

end NUMINAMATH_CALUDE_parallel_segment_length_l3035_303502


namespace NUMINAMATH_CALUDE_sarah_amount_l3035_303585

-- Define the total amount Bridge and Sarah have
def total : ℕ := 300

-- Define the difference between Bridget's and Sarah's amounts
def difference : ℕ := 50

-- Theorem to prove
theorem sarah_amount : ∃ (s : ℕ), s + (s + difference) = total ∧ s = 125 := by
  sorry

end NUMINAMATH_CALUDE_sarah_amount_l3035_303585


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3035_303561

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3035_303561


namespace NUMINAMATH_CALUDE_division_remainder_proof_l3035_303564

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (h1 : dividend = 176) (h2 : divisor = 19) (h3 : quotient = 9) :
  dividend - divisor * quotient = 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l3035_303564


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l3035_303560

theorem geometric_arithmetic_sequence_sum (x y : ℝ) : 
  5 < x ∧ x < y ∧ y < 15 →
  (∃ r : ℝ, r > 0 ∧ x = 5 * r ∧ y = 5 * r^2) →
  (∃ d : ℝ, y = x + d ∧ 15 = y + d) →
  x + y = 10 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l3035_303560


namespace NUMINAMATH_CALUDE_plan2_cheaper_l3035_303588

/-- Represents a payment plan with number of installments and months between payments -/
structure PaymentPlan where
  installments : ℕ
  months_between : ℕ

/-- Calculates the total payment amount for a given payment plan -/
def totalPayment (price : ℝ) (rate : ℝ) (plan : PaymentPlan) : ℝ :=
  price * (1 + rate) ^ (plan.installments * plan.months_between)

theorem plan2_cheaper (price : ℝ) (rate : ℝ) (plan1 plan2 : PaymentPlan) :
  price > 0 →
  rate > 0 →
  plan1.installments = 3 →
  plan1.months_between = 4 →
  plan2.installments = 12 →
  plan2.months_between = 1 →
  totalPayment price rate plan2 ≤ totalPayment price rate plan1 := by
  sorry

#check plan2_cheaper

end NUMINAMATH_CALUDE_plan2_cheaper_l3035_303588


namespace NUMINAMATH_CALUDE_sock_matching_probability_l3035_303593

def total_socks : ℕ := 18
def gray_socks : ℕ := 10
def white_socks : ℕ := 8

def total_combinations : ℕ := total_socks.choose 2
def matching_gray_combinations : ℕ := gray_socks.choose 2
def matching_white_combinations : ℕ := white_socks.choose 2
def matching_combinations : ℕ := matching_gray_combinations + matching_white_combinations

theorem sock_matching_probability :
  (matching_combinations : ℚ) / total_combinations = 73 / 153 := by sorry

end NUMINAMATH_CALUDE_sock_matching_probability_l3035_303593


namespace NUMINAMATH_CALUDE_square_area_ratio_l3035_303559

theorem square_area_ratio (x : ℝ) (h : x > 0) : 
  (x^2) / ((4*x)^2) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3035_303559


namespace NUMINAMATH_CALUDE_band_sections_fraction_l3035_303534

theorem band_sections_fraction (trumpet_fraction trombone_fraction : ℝ) 
  (h1 : trumpet_fraction = 0.5)
  (h2 : trombone_fraction = 0.125) :
  trumpet_fraction + trombone_fraction = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_band_sections_fraction_l3035_303534


namespace NUMINAMATH_CALUDE_replacement_process_terminates_l3035_303511

/-- A finite sequence of zeros and ones -/
def BinarySequence := List Bool

/-- The operation of replacing "01" with "1000" in a binary sequence -/
def replace01With1000 (seq : BinarySequence) : BinarySequence :=
  match seq with
  | [] => []
  | [x] => [x]
  | false :: true :: xs => true :: false :: false :: false :: xs
  | x :: xs => x :: replace01With1000 xs

/-- The weight of a binary sequence -/
def weight (seq : BinarySequence) : Nat :=
  seq.foldl (λ acc x => if x then 4 * acc else acc + 1) 0

/-- Theorem: The replacement process will eventually terminate -/
theorem replacement_process_terminates (seq : BinarySequence) :
  ∃ n : Nat, ∀ m : Nat, m ≥ n → replace01With1000^[m] seq = replace01With1000^[n] seq :=
sorry

end NUMINAMATH_CALUDE_replacement_process_terminates_l3035_303511


namespace NUMINAMATH_CALUDE_part_one_part_two_l3035_303545

noncomputable section

-- Define the functions
def f (b : ℝ) (x : ℝ) : ℝ := (2*x + b) * Real.exp x
def F (b : ℝ) (x : ℝ) : ℝ := b*x - Real.log x
def g (b : ℝ) (x : ℝ) : ℝ := b*x^2 - 2*x - F b x

-- Part 1
theorem part_one (b : ℝ) :
  b < 0 ∧ 
  (∃ (M : Set ℝ), ∀ (x y : ℝ), x ∈ M → y ∈ M → x < y → 
    ((f b x < f b y ↔ F b x < F b y) ∨ (f b x > f b y ↔ F b x > F b y))) →
  b < -2 :=
sorry

-- Part 2
theorem part_two (b : ℝ) :
  b > 0 ∧ 
  (∀ x ∈ Set.Icc 1 (Real.exp 1), g b x ≥ -2) ∧
  (∃ x ∈ Set.Icc 1 (Real.exp 1), g b x = -2) →
  b ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3035_303545


namespace NUMINAMATH_CALUDE_octal_67_equals_ternary_2001_l3035_303572

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to ternary --/
def decimal_to_ternary (n : ℕ) : ℕ := sorry

theorem octal_67_equals_ternary_2001 :
  decimal_to_ternary (octal_to_decimal 67) = 2001 := by sorry

end NUMINAMATH_CALUDE_octal_67_equals_ternary_2001_l3035_303572


namespace NUMINAMATH_CALUDE_sum_of_squares_l3035_303527

theorem sum_of_squares (x y : ℝ) (hx : x^2 = 8*x + y) (hy : y^2 = x + 8*y) (hxy : x ≠ y) :
  x^2 + y^2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3035_303527


namespace NUMINAMATH_CALUDE_inequality_proof_l3035_303531

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (1 + 2*a)) + (1 / (1 + 2*b)) + (1 / (1 + 2*c)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3035_303531


namespace NUMINAMATH_CALUDE_height_difference_l3035_303550

theorem height_difference (height_A : ℝ) (initial_ratio : ℝ) (growth : ℝ) : 
  height_A = 72 →
  initial_ratio = 2/3 →
  growth = 10 →
  height_A - (initial_ratio * height_A + growth) = 14 := by
sorry

end NUMINAMATH_CALUDE_height_difference_l3035_303550


namespace NUMINAMATH_CALUDE_solution_set_theorem_l3035_303568

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ) (f' : ℝ → ℝ)

-- Define the domain of f
def domain : Set ℝ := { x | x > 0 }

-- State the theorem
theorem solution_set_theorem 
  (h_deriv : ∀ x ∈ domain, HasDerivAt f (f' x) x)
  (h_ineq : ∀ x ∈ domain, f x < -x * f' x) :
  { x ∈ domain | f (x + 1) > (x - 1) * f (x^2 - 1) } = { x | x > 2 } := by
  sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l3035_303568


namespace NUMINAMATH_CALUDE_unique_magnitude_of_complex_roots_l3035_303541

theorem unique_magnitude_of_complex_roots : ∃! r : ℝ, ∃ z : ℂ, z^2 - 8*z + 45 = 0 ∧ Complex.abs z = r := by
  sorry

end NUMINAMATH_CALUDE_unique_magnitude_of_complex_roots_l3035_303541


namespace NUMINAMATH_CALUDE_exponent_equation_solution_l3035_303575

theorem exponent_equation_solution (a b : ℝ) (m n : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^m * b * b^n)^3 = a^6 * b^15 → m = 2 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equation_solution_l3035_303575


namespace NUMINAMATH_CALUDE_ac_circuit_current_l3035_303556

def V : ℂ := 2 + 2*Complex.I
def Z : ℂ := 2 - 2*Complex.I

theorem ac_circuit_current : V = Complex.I * Z := by sorry

end NUMINAMATH_CALUDE_ac_circuit_current_l3035_303556


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3035_303537

theorem arithmetic_sequence_length (a₁ aₙ d : ℤ) (h : a₁ = 165 ∧ aₙ = 45 ∧ d = -5) :
  (a₁ - aₙ) / (-d) + 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3035_303537


namespace NUMINAMATH_CALUDE_exists_points_D_E_l3035_303570

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a function to check if a point is on a line segment
def isOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop := sorry

-- Define a function to calculate distance between two points
def distance (P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem exists_points_D_E (ABC : Triangle) : 
  ∃ (D E : ℝ × ℝ), 
    isOnSegment D ABC.A ABC.B ∧ 
    isOnSegment E ABC.A ABC.C ∧ 
    distance ABC.A D = distance D E ∧ 
    distance D E = distance E ABC.C := by
  sorry

end NUMINAMATH_CALUDE_exists_points_D_E_l3035_303570


namespace NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l3035_303529

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem third_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a2 : a 2 = 8)
  (h_a5 : a 5 = 64) :
  a 3 = 16 := by
sorry

end NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l3035_303529


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_vertices_l3035_303542

/-- The area of a square with adjacent vertices at (1, -2) and (-3, 5) is 65 -/
theorem square_area_from_adjacent_vertices : 
  let p1 : ℝ × ℝ := (1, -2)
  let p2 : ℝ × ℝ := (-3, 5)
  let side_length := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  side_length^2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_vertices_l3035_303542


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3035_303551

theorem simplify_and_evaluate (x : ℤ) (h1 : -2 < x) (h2 : x < 3) :
  (x / (x + 1) - 3 * x / (x - 1)) / (x / (x^2 - 1)) = -8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3035_303551


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l3035_303574

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel m n → 
  contains β n → 
  perpendicularPlanes α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l3035_303574


namespace NUMINAMATH_CALUDE_smallest_sum_with_real_roots_l3035_303526

theorem smallest_sum_with_real_roots (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x : ℝ, x^2 + a*x + 3*b = 0) → 
  (∃ x : ℝ, x^2 + 3*b*x + a = 0) → 
  a + b ≥ 7 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
    (∃ x : ℝ, x^2 + a₀*x + 3*b₀ = 0) ∧ 
    (∃ x : ℝ, x^2 + 3*b₀*x + a₀ = 0) ∧ 
    a₀ + b₀ = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_with_real_roots_l3035_303526


namespace NUMINAMATH_CALUDE_probability_even_product_l3035_303590

/-- Spinner C with numbers 1 through 6 -/
def spinner_C : Finset ℕ := Finset.range 6

/-- Spinner D with numbers 1 through 4 -/
def spinner_D : Finset ℕ := Finset.range 4

/-- Function to check if a number is even -/
def is_even (n : ℕ) : Bool := n % 2 = 0

/-- Function to check if the product of two numbers is even -/
def product_is_even (x y : ℕ) : Bool := is_even (x * y)

/-- Total number of possible outcomes -/
def total_outcomes : ℕ := (Finset.card spinner_C) * (Finset.card spinner_D)

/-- Number of outcomes where the product is even -/
def even_product_outcomes : ℕ := Finset.card (Finset.filter (λ (pair : ℕ × ℕ) => product_is_even pair.1 pair.2) (spinner_C.product spinner_D))

/-- Theorem stating the probability of getting an even product -/
theorem probability_even_product :
  (even_product_outcomes : ℚ) / total_outcomes = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_probability_even_product_l3035_303590


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3035_303525

-- Define the function f(x) = x^3 + x
def f (x : ℝ) := x^3 + x

-- Define the derivative of f(x)
def f' (x : ℝ) := 3 * x^2 + 1

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (4 * x - y - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3035_303525


namespace NUMINAMATH_CALUDE_vasyas_birthday_vasyas_birthday_was_thursday_l3035_303530

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to get the day after tomorrow
def dayAfterTomorrow (d : DayOfWeek) : DayOfWeek :=
  nextDay (nextDay d)

theorem vasyas_birthday (today : DayOfWeek) 
  (h1 : dayAfterTomorrow today = DayOfWeek.Sunday) 
  (h2 : nextDay today ≠ DayOfWeek.Sunday) : 
  nextDay (nextDay (nextDay today)) = DayOfWeek.Sunday := by
  sorry

-- The main theorem
theorem vasyas_birthday_was_thursday : 
  ∃ (today : DayOfWeek), 
    dayAfterTomorrow today = DayOfWeek.Sunday ∧ 
    nextDay today ≠ DayOfWeek.Sunday ∧
    nextDay (nextDay (nextDay today)) = DayOfWeek.Sunday := by
  sorry

end NUMINAMATH_CALUDE_vasyas_birthday_vasyas_birthday_was_thursday_l3035_303530


namespace NUMINAMATH_CALUDE_industrial_machine_output_l3035_303578

/-- An industrial machine that makes shirts -/
structure ShirtMachine where
  totalShirts : ℕ
  workingMinutes : ℕ

/-- Calculate the shirts per minute for a given machine -/
def shirtsPerMinute (machine : ShirtMachine) : ℚ :=
  machine.totalShirts / machine.workingMinutes

theorem industrial_machine_output (machine : ShirtMachine) 
  (h1 : machine.totalShirts = 6)
  (h2 : machine.workingMinutes = 2) : 
  shirtsPerMinute machine = 3 := by
  sorry

end NUMINAMATH_CALUDE_industrial_machine_output_l3035_303578


namespace NUMINAMATH_CALUDE_geometric_log_arithmetic_l3035_303521

open Real

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A sequence is geometric if there exists a non-zero real number q such that
    for all n, a(n+1) = q * a(n) -/
def IsGeometric (a : Sequence) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- A sequence is arithmetic if there exists a real number d such that
    for all n, a(n+1) - a(n) = d -/
def IsArithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The main theorem: If a sequence of positive terms is geometric,
    then the sequence of logarithms plus 1 is arithmetic,
    but the converse is not always true -/
theorem geometric_log_arithmetic (a : Sequence) :
  (∀ n : ℕ, a n > 0) →
  IsGeometric a →
  IsArithmetic (fun n => log (a n) + 1) ∧
  ¬(IsArithmetic (fun n => log (a n) + 1) → IsGeometric a) :=
sorry

end NUMINAMATH_CALUDE_geometric_log_arithmetic_l3035_303521


namespace NUMINAMATH_CALUDE_james_cheezits_consumption_l3035_303533

/-- Represents the number of bags of Cheezits James ate -/
def bags_of_cheezits : ℕ := sorry

/-- Represents the weight of each bag of Cheezits in ounces -/
def bag_weight : ℕ := 2

/-- Represents the number of calories in an ounce of Cheezits -/
def calories_per_ounce : ℕ := 150

/-- Represents the duration of James' run in minutes -/
def run_duration : ℕ := 40

/-- Represents the number of calories burned per minute during the run -/
def calories_burned_per_minute : ℕ := 12

/-- Represents the excess calories James consumed -/
def excess_calories : ℕ := 420

theorem james_cheezits_consumption :
  bags_of_cheezits * (bag_weight * calories_per_ounce) - 
  (run_duration * calories_burned_per_minute) = excess_calories ∧
  bags_of_cheezits = 3 := by sorry

end NUMINAMATH_CALUDE_james_cheezits_consumption_l3035_303533


namespace NUMINAMATH_CALUDE_lunch_total_amount_l3035_303579

/-- The total amount spent on lunch given the conditions -/
theorem lunch_total_amount (your_spending friend_spending : ℕ) 
  (h1 : friend_spending = 10)
  (h2 : friend_spending = your_spending + 3) : 
  your_spending + friend_spending = 17 := by
  sorry

end NUMINAMATH_CALUDE_lunch_total_amount_l3035_303579


namespace NUMINAMATH_CALUDE_tara_ice_cream_purchase_l3035_303554

/-- The number of cartons of ice cream Tara bought -/
def ice_cream_cartons : ℕ := sorry

/-- The number of cartons of yoghurt Tara bought -/
def yoghurt_cartons : ℕ := 4

/-- The cost of one carton of ice cream in dollars -/
def ice_cream_cost : ℕ := 7

/-- The cost of one carton of yoghurt in dollars -/
def yoghurt_cost : ℕ := 1

/-- Theorem stating that Tara bought 19 cartons of ice cream -/
theorem tara_ice_cream_purchase :
  ice_cream_cartons = 19 ∧
  ice_cream_cartons * ice_cream_cost = yoghurt_cartons * yoghurt_cost + 129 :=
by sorry

end NUMINAMATH_CALUDE_tara_ice_cream_purchase_l3035_303554


namespace NUMINAMATH_CALUDE_cyclist_speed_ratio_l3035_303509

theorem cyclist_speed_ratio : 
  ∀ (v_A v_B v_C : ℝ),
    v_A > 0 → v_B > 0 → v_C > 0 →
    (v_A - v_B) * 4 = 20 →
    (v_A + v_C) * 2 = 30 →
    v_A / v_B = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_ratio_l3035_303509


namespace NUMINAMATH_CALUDE_m_range_l3035_303518

-- Define the propositions
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x + 2 ≥ m

def q (m : ℝ) : Prop := ∀ x : ℝ, (-(7 - 3*m))^(x+1) < (-(7 - 3*m))^x

-- State the theorem
theorem m_range (m : ℝ) : 
  (p m ∧ ¬(q m)) ∨ (¬(p m) ∧ q m) → 1 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l3035_303518


namespace NUMINAMATH_CALUDE_mailbox_probability_l3035_303594

-- Define the number of mailboxes
def num_mailboxes : ℕ := 2

-- Define the number of letters
def num_letters : ℕ := 3

-- Define the function to calculate the total number of ways to distribute letters
def total_ways : ℕ := 2^num_letters

-- Define the function to calculate the number of favorable ways
def favorable_ways : ℕ := (num_letters.choose (num_letters - 1)) * (num_mailboxes^(num_mailboxes - 1))

-- Define the probability
def probability : ℚ := favorable_ways / total_ways

-- Theorem statement
theorem mailbox_probability : probability = 3/4 := by sorry

end NUMINAMATH_CALUDE_mailbox_probability_l3035_303594


namespace NUMINAMATH_CALUDE_inequality_solution_system_of_equations_solution_l3035_303577

-- Part 1: Inequality
theorem inequality_solution (x : ℝ) :
  (5 * x - 12 ≤ 2 * (4 * x - 3)) ↔ (x ≥ -2) := by sorry

-- Part 2: System of equations
theorem system_of_equations_solution (x y : ℝ) :
  (x - y = 5 ∧ 2 * x + y = 4) → (x = 3 ∧ y = -2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_system_of_equations_solution_l3035_303577


namespace NUMINAMATH_CALUDE_max_value_g_geq_seven_l3035_303500

theorem max_value_g_geq_seven (a b : ℝ) (h_a : a ≤ -1) : 
  let f := fun x : ℝ => Real.exp x * (x^2 + a*x + 1)
  let g := fun x : ℝ => 2*x^3 + 3*(b+1)*x^2 + 6*b*x + 6
  let x_min_f := -(a + 1)
  (∀ x : ℝ, g x ≥ g x_min_f) → 
  (∀ x : ℝ, f x ≥ f x_min_f) → 
  ∃ x : ℝ, g x ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_max_value_g_geq_seven_l3035_303500


namespace NUMINAMATH_CALUDE_mountain_height_theorem_l3035_303517

-- Define the measurement points and the peak
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the measurement setup
structure MountainMeasurement where
  A : Point3D
  B : Point3D
  C : Point3D
  peak : Point3D
  AB : ℝ
  BC : ℝ
  angle_ABC : ℝ
  elevation_A : ℝ
  elevation_C : ℝ
  angle_BAT : ℝ

-- Define the theorem
theorem mountain_height_theorem (m : MountainMeasurement) 
  (h_AB : m.AB = 100)
  (h_BC : m.BC = 150)
  (h_angle_ABC : m.angle_ABC = 130 * π / 180)
  (h_elevation_A : m.elevation_A = 20 * π / 180)
  (h_elevation_C : m.elevation_C = 22 * π / 180)
  (h_angle_BAT : m.angle_BAT = 93 * π / 180) :
  ∃ (h1 h2 : ℝ), 
    (abs (h1 - 93.4) < 0.1 ∧ abs (h2 - 390.9) < 0.1) ∧
    ((m.peak.z - m.A.z = h1) ∨ (m.peak.z - m.A.z = h2)) := by
  sorry

end NUMINAMATH_CALUDE_mountain_height_theorem_l3035_303517


namespace NUMINAMATH_CALUDE_diagonal_lengths_and_t_value_l3035_303505

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (-1, -2)
def C : ℝ × ℝ := (-2, -1)

-- Define vectors
def vec_AB : ℝ × ℝ := (A.1 - B.1, A.2 - B.2)
def vec_OC : ℝ × ℝ := C

-- Calculate the fourth point D
def D : ℝ × ℝ := (A.1 + C.1 - B.1, A.2 + C.2 - B.2)

-- Define diagonals
def vec_AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def vec_BD : ℝ × ℝ := (D.1 - B.1, D.2 - B.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statements
theorem diagonal_lengths_and_t_value : 
  (vec_AC.1^2 + vec_AC.2^2 = 32) ∧ 
  (vec_BD.1^2 + vec_BD.2^2 = 40) ∧ 
  (∃ t : ℝ, t = -11/5 ∧ dot_product (vec_AB.1 + t * vec_OC.1, vec_AB.2 + t * vec_OC.2) vec_OC = 0) :=
sorry

end NUMINAMATH_CALUDE_diagonal_lengths_and_t_value_l3035_303505


namespace NUMINAMATH_CALUDE_max_sum_of_squares_max_sum_of_squares_value_exact_max_sum_of_squares_l3035_303565

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 18 →
  a * b + c + d = 95 →
  a * d + b * c = 180 →
  c * d = 105 →
  ∀ (w x y z : ℝ), 
    w + x = 18 →
    w * x + y + z = 95 →
    w * z + x * y = 180 →
    y * z = 105 →
    a^2 + b^2 + c^2 + d^2 ≥ w^2 + x^2 + y^2 + z^2 :=
by
  sorry

theorem max_sum_of_squares_value (a b c d : ℝ) :
  a + b = 18 →
  a * b + c + d = 95 →
  a * d + b * c = 180 →
  c * d = 105 →
  a^2 + b^2 + c^2 + d^2 ≤ 1486 :=
by
  sorry

theorem exact_max_sum_of_squares (a b c d : ℝ) :
  a + b = 18 →
  a * b + c + d = 95 →
  a * d + b * c = 180 →
  c * d = 105 →
  ∃ (w x y z : ℝ),
    w + x = 18 ∧
    w * x + y + z = 95 ∧
    w * z + x * y = 180 ∧
    y * z = 105 ∧
    w^2 + x^2 + y^2 + z^2 = 1486 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_max_sum_of_squares_value_exact_max_sum_of_squares_l3035_303565


namespace NUMINAMATH_CALUDE_fathers_children_l3035_303569

theorem fathers_children (father_age : ℕ) (children_sum : ℕ) (n : ℕ) : 
  father_age = 75 →
  father_age = children_sum →
  children_sum + 15 * n = 2 * (father_age + 15) →
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_fathers_children_l3035_303569


namespace NUMINAMATH_CALUDE_diana_box_capacity_l3035_303536

/-- Represents a box with dimensions and jellybean capacity -/
structure Box where
  height : ℝ
  width : ℝ
  length : ℝ
  capacity : ℕ

/-- Calculates the volume of a box -/
def boxVolume (b : Box) : ℝ :=
  b.height * b.width * b.length

/-- Theorem: A box with triple height, double width, and quadruple length of Bert's box
    that holds 150 jellybeans will hold 3600 jellybeans -/
theorem diana_box_capacity (bert_box : Box)
    (h1 : bert_box.capacity = 150)
    (diana_box : Box)
    (h2 : diana_box.height = 3 * bert_box.height)
    (h3 : diana_box.width = 2 * bert_box.width)
    (h4 : diana_box.length = 4 * bert_box.length) :
    diana_box.capacity = 3600 := by
  sorry


end NUMINAMATH_CALUDE_diana_box_capacity_l3035_303536


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_halves_l3035_303543

theorem one_thirds_in_nine_halves : (9 : ℚ) / 2 / (1 / 3) = 27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_halves_l3035_303543


namespace NUMINAMATH_CALUDE_parallel_transitivity_l3035_303581

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary properties for a line in 3D space
  -- This is a simplified representation
  mk :: 

-- Define parallelism for lines in 3D space
def parallel (l1 l2 : Line3D) : Prop :=
  -- Define what it means for two lines to be parallel
  sorry

-- State the theorem
theorem parallel_transitivity (a b c : Line3D) :
  parallel a c → parallel b c → parallel a b := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l3035_303581


namespace NUMINAMATH_CALUDE_tom_games_owned_before_l3035_303547

/-- The number of games Tom owned before purchasing new games -/
def games_owned_before : ℕ := 0

/-- The cost of the Batman game in dollars -/
def batman_game_cost : ℚ := 13.60

/-- The cost of the Superman game in dollars -/
def superman_game_cost : ℚ := 5.06

/-- The total amount Tom spent on video games in dollars -/
def total_spent : ℚ := 18.66

theorem tom_games_owned_before :
  games_owned_before = 0 ∧
  batman_game_cost + superman_game_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_tom_games_owned_before_l3035_303547


namespace NUMINAMATH_CALUDE_x_range_l3035_303512

-- Define the condition
def satisfies_equation (x y : ℝ) : Prop :=
  x - 6 * Real.sqrt y - 4 * Real.sqrt (x - y) + 12 = 0

-- Define the theorem
theorem x_range (x y : ℝ) (h : satisfies_equation x y) :
  14 - 2 * Real.sqrt 13 ≤ x ∧ x ≤ 14 + 2 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_x_range_l3035_303512


namespace NUMINAMATH_CALUDE_soda_duration_problem_l3035_303580

/-- Given the number of soda and water bottles, and the daily consumption ratio,
    calculate the number of days the soda bottles will last. -/
def sodaDuration (sodaCount waterCount : ℕ) (sodaRatio waterRatio : ℕ) : ℕ :=
  min (sodaCount / sodaRatio) (waterCount / waterRatio)

/-- Theorem stating that with 360 soda bottles and 162 water bottles,
    consumed in a 3:2 ratio, the soda bottles will last for 81 days. -/
theorem soda_duration_problem :
  sodaDuration 360 162 3 2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_soda_duration_problem_l3035_303580


namespace NUMINAMATH_CALUDE_least_odd_prime_factor_1234_10_plus_1_l3035_303563

theorem least_odd_prime_factor_1234_10_plus_1 : 
  (Nat.minFac (1234^10 + 1)) = 61 := by sorry

end NUMINAMATH_CALUDE_least_odd_prime_factor_1234_10_plus_1_l3035_303563


namespace NUMINAMATH_CALUDE_polygon_properties_l3035_303589

/-- Represents a convex polygon with properties as described in the problem -/
structure ConvexPolygon where
  n : ℕ                             -- number of sides
  interior_angle_sum : ℝ             -- sum of interior angles minus one unknown angle
  triangle_area : ℝ                  -- area of triangle formed by three adjacent vertices
  triangle_side : ℝ                  -- length of one side of the triangle
  triangle_opposite_angle : ℝ        -- angle opposite to the known side in the triangle

/-- The theorem to be proved -/
theorem polygon_properties (p : ConvexPolygon) 
  (h1 : p.interior_angle_sum = 3240)
  (h2 : p.triangle_area = 150)
  (h3 : p.triangle_side = 15)
  (h4 : p.triangle_opposite_angle = 60) :
  p.n = 20 ∧ (180 * (p.n - 2) - p.interior_angle_sum = 0) := by
  sorry


end NUMINAMATH_CALUDE_polygon_properties_l3035_303589


namespace NUMINAMATH_CALUDE_president_vice_selection_ways_l3035_303552

/-- The number of ways to choose a president and vice-president from a club with the given conditions -/
def choose_president_and_vice (total_members boys girls : ℕ) : ℕ :=
  (boys * (boys - 1)) + (girls * (girls - 1))

/-- Theorem stating the number of ways to choose a president and vice-president under the given conditions -/
theorem president_vice_selection_ways :
  let total_members : ℕ := 30
  let boys : ℕ := 18
  let girls : ℕ := 12
  choose_president_and_vice total_members boys girls = 438 := by
  sorry

#eval choose_president_and_vice 30 18 12

end NUMINAMATH_CALUDE_president_vice_selection_ways_l3035_303552


namespace NUMINAMATH_CALUDE_map_scale_l3035_303549

theorem map_scale (map_length : ℝ) (actual_distance : ℝ) :
  (15 : ℝ) * actual_distance = 90 * map_length →
  (20 : ℝ) * actual_distance = 120 * map_length :=
by sorry

end NUMINAMATH_CALUDE_map_scale_l3035_303549


namespace NUMINAMATH_CALUDE_cupcake_frosting_problem_l3035_303514

/-- Represents the number of cupcakes frosted in a given time -/
def cupcakes_frosted (rate : ℚ) (time : ℚ) : ℚ := rate * time

/-- Represents the combined rate of two people frosting cupcakes -/
def combined_rate (rate1 : ℚ) (rate2 : ℚ) : ℚ := 1 / (1 / rate1 + 1 / rate2)

theorem cupcake_frosting_problem :
  let cagney_rate : ℚ := 1 / 18  -- Cagney's frosting rate (cupcakes per second)
  let lacey_rate : ℚ := 1 / 40   -- Lacey's frosting rate (cupcakes per second)
  let total_time : ℚ := 6 * 60   -- Total time in seconds
  let lacey_delay : ℚ := 60      -- Lacey's delay in seconds

  let cagney_solo_time := lacey_delay
  let combined_time := total_time - lacey_delay
  let combined_frosting_rate := combined_rate cagney_rate lacey_rate

  let total_cupcakes := 
    cupcakes_frosted cagney_rate cagney_solo_time + 
    cupcakes_frosted combined_frosting_rate combined_time

  ⌊total_cupcakes⌋ = 27 :=
by sorry

end NUMINAMATH_CALUDE_cupcake_frosting_problem_l3035_303514


namespace NUMINAMATH_CALUDE_women_who_left_l3035_303598

theorem women_who_left (initial_men initial_women : ℕ) : 
  (initial_men : ℚ) / initial_women = 4 / 5 →
  initial_men + 2 = 14 →
  ∃ (left : ℕ), 2 * (initial_women - left) = 24 ∧ left = 3 :=
by sorry

end NUMINAMATH_CALUDE_women_who_left_l3035_303598


namespace NUMINAMATH_CALUDE_only_tiger_and_leopard_can_participate_l3035_303506

-- Define the animals
inductive Animal : Type
| Lion : Animal
| Tiger : Animal
| Leopard : Animal
| Elephant : Animal

-- Define a function to represent selection
def isSelected : Animal → Prop := sorry

-- Define the conditions
def conditions (isSelected : Animal → Prop) : Prop :=
  (isSelected Animal.Lion → isSelected Animal.Tiger) ∧
  (¬isSelected Animal.Leopard → ¬isSelected Animal.Tiger) ∧
  (isSelected Animal.Leopard → ¬isSelected Animal.Elephant) ∧
  (∃ (a b : Animal), a ≠ b ∧ isSelected a ∧ isSelected b ∧
    ∀ (c : Animal), c ≠ a ∧ c ≠ b → ¬isSelected c)

-- Theorem statement
theorem only_tiger_and_leopard_can_participate :
  ∀ (isSelected : Animal → Prop),
    conditions isSelected →
    isSelected Animal.Tiger ∧ isSelected Animal.Leopard ∧
    ¬isSelected Animal.Lion ∧ ¬isSelected Animal.Elephant :=
sorry

end NUMINAMATH_CALUDE_only_tiger_and_leopard_can_participate_l3035_303506


namespace NUMINAMATH_CALUDE_arithmetic_combination_l3035_303520

theorem arithmetic_combination : (2 + 4 / 10) * 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_combination_l3035_303520


namespace NUMINAMATH_CALUDE_same_solution_k_value_l3035_303557

theorem same_solution_k_value (k : ℝ) : 
  (∀ x : ℝ, 5 * x + 3 = 0 ↔ 5 * x + 3 * k = 21) → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_k_value_l3035_303557


namespace NUMINAMATH_CALUDE_min_distance_complex_main_theorem_l3035_303582

-- Define inductive reasoning
def inductiveReasoning : String := "reasoning from specific to general"

-- Define deductive reasoning
def deductiveReasoning : String := "reasoning from general to specific"

-- Theorem for the complex number part
theorem min_distance_complex (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ (w : ℂ), Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≥ min_val :=
sorry

-- Main theorem combining all parts
theorem main_theorem :
  inductiveReasoning = "reasoning from specific to general" ∧
  deductiveReasoning = "reasoning from general to specific" ∧
  ∀ (z : ℂ), Complex.abs (z + 2 - 2*I) = 1 →
    ∃ (min_val : ℝ), min_val = 3 ∧ ∀ (w : ℂ), Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_main_theorem_l3035_303582


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l3035_303504

def is_valid_arrangement (arr : List Nat) : Prop :=
  ∀ i, i < arr.length - 1 → (10 * arr[i]! + arr[i+1]!) % 7 = 0

theorem no_valid_arrangement : 
  ¬ ∃ (arr : List Nat), arr.toFinset = {1, 2, 3, 4, 5, 6, 8, 9} ∧ is_valid_arrangement arr :=
by sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l3035_303504


namespace NUMINAMATH_CALUDE_cylinder_increase_equality_l3035_303597

theorem cylinder_increase_equality (x : ℝ) : 
  x > 0 → 
  π * (8 + x)^2 * 3 = π * 8^2 * (3 + x) → 
  x = 16/3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_increase_equality_l3035_303597


namespace NUMINAMATH_CALUDE_sum_remainder_theorem_l3035_303562

theorem sum_remainder_theorem :
  (9256 + 9257 + 9258 + 9259 + 9260) % 13 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_theorem_l3035_303562


namespace NUMINAMATH_CALUDE_cookies_per_bag_l3035_303538

/-- Given 33 cookies distributed equally among 3 bags, prove that each bag contains 11 cookies. -/
theorem cookies_per_bag :
  ∀ (total_cookies : ℕ) (num_bags : ℕ) (cookies_per_bag : ℕ),
    total_cookies = 33 →
    num_bags = 3 →
    total_cookies = num_bags * cookies_per_bag →
    cookies_per_bag = 11 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l3035_303538


namespace NUMINAMATH_CALUDE_sally_fries_theorem_l3035_303599

def sally_fries_problem (sally_initial : ℕ) (mark_total : ℕ) (jessica_total : ℕ) : Prop :=
  let mark_share := mark_total / 3
  let jessica_share := jessica_total / 2
  sally_initial + mark_share + jessica_share = 38

theorem sally_fries_theorem :
  sally_fries_problem 14 36 24 :=
by
  sorry

end NUMINAMATH_CALUDE_sally_fries_theorem_l3035_303599


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_eq_one_l3035_303528

/-- Given two real numbers x and y satisfying specific equations, prove that cos(x + 2y) = 1 -/
theorem cos_x_plus_2y_eq_one (x y : ℝ) 
  (hx : x^3 + Real.cos x + x - 2 = 0)
  (hy : 8 * y^3 - 2 * (Real.cos y)^2 + 2 * y + 3 = 0) :
  Real.cos (x + 2 * y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_x_plus_2y_eq_one_l3035_303528


namespace NUMINAMATH_CALUDE_opposite_expressions_theorem_l3035_303573

theorem opposite_expressions_theorem (a : ℚ) : 
  (3 * a + 1 = -(3 * (a - 1))) → a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_opposite_expressions_theorem_l3035_303573


namespace NUMINAMATH_CALUDE_product_remainder_zero_l3035_303587

theorem product_remainder_zero (n : ℕ) :
  (2021 * 2022 * 2023 * 2024 * 2025) % 23 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_product_remainder_zero_l3035_303587


namespace NUMINAMATH_CALUDE_gifted_books_count_l3035_303515

def books_per_month : ℕ := 2
def months_per_year : ℕ := 12
def bought_books : ℕ := 8
def reread_old_books : ℕ := 4

def borrowed_books : ℕ := bought_books - 2

def total_books_needed : ℕ := books_per_month * months_per_year
def new_books_needed : ℕ := total_books_needed - reread_old_books
def new_books_acquired : ℕ := bought_books + borrowed_books

theorem gifted_books_count : new_books_needed - new_books_acquired = 6 := by
  sorry

end NUMINAMATH_CALUDE_gifted_books_count_l3035_303515


namespace NUMINAMATH_CALUDE_ivans_chess_claim_impossible_l3035_303510

theorem ivans_chess_claim_impossible : ¬ ∃ (n : ℕ), n > 0 ∧ n + 3*n + 6*n = 64 := by sorry

end NUMINAMATH_CALUDE_ivans_chess_claim_impossible_l3035_303510


namespace NUMINAMATH_CALUDE_original_speed_is_30_l3035_303524

/-- Represents the driving scenario with given conditions -/
def DrivingScenario (original_speed : ℝ) : Prop :=
  let total_distance : ℝ := 100
  let breakdown_time : ℝ := 2
  let repair_time : ℝ := 0.5
  let speed_increase_factor : ℝ := 1.6
  
  -- Time equation: total time = time before breakdown + repair time + time after repair
  total_distance / original_speed = 
    breakdown_time + repair_time + 
    (total_distance - breakdown_time * original_speed) / (speed_increase_factor * original_speed)

/-- Theorem stating that the original speed satisfying the driving scenario is 30 km/h -/
theorem original_speed_is_30 : 
  ∃ (speed : ℝ), DrivingScenario speed ∧ speed = 30 := by
  sorry

end NUMINAMATH_CALUDE_original_speed_is_30_l3035_303524


namespace NUMINAMATH_CALUDE_simplify_expression_l3035_303576

theorem simplify_expression (b : ℝ) (h : b ≠ 1) :
  1 - 1 / (2 + b / (1 - b)) = 1 / (2 - b) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3035_303576


namespace NUMINAMATH_CALUDE_candy_distribution_l3035_303566

theorem candy_distribution (num_clowns num_children initial_candies remaining_candies : ℕ) 
  (h1 : num_clowns = 4)
  (h2 : num_children = 30)
  (h3 : initial_candies = 700)
  (h4 : remaining_candies = 20)
  (h5 : ∃ (candies_per_person : ℕ), 
    (num_clowns + num_children) * candies_per_person = initial_candies - remaining_candies) :
  ∃ (candies_per_person : ℕ), candies_per_person = 20 ∧ 
    (num_clowns + num_children) * candies_per_person = initial_candies - remaining_candies :=
by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3035_303566
