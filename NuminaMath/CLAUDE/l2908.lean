import Mathlib

namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2908_290843

/-- Given a line segment from (4, 3) to (x, 9) with length 15 and x < 0, prove x = 4 - √189 -/
theorem line_segment_endpoint (x : ℝ) : 
  x < 0 → 
  ((x - 4)^2 + (9 - 3)^2 : ℝ) = 15^2 → 
  x = 4 - Real.sqrt 189 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2908_290843


namespace NUMINAMATH_CALUDE_recurrence_solution_l2908_290854

def recurrence_relation (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n - 3 * a (n - 1) - 10 * a (n - 2) = 28 * (5 ^ n)

def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 0 = 25 ∧ a 1 = 120

def general_term (n : ℕ) : ℝ :=
  (20 * n + 10) * (5 ^ n) + 15 * ((-2) ^ n)

theorem recurrence_solution (a : ℕ → ℝ) :
  recurrence_relation a ∧ initial_conditions a →
  ∀ n : ℕ, a n = general_term n := by
  sorry

end NUMINAMATH_CALUDE_recurrence_solution_l2908_290854


namespace NUMINAMATH_CALUDE_ballet_slipper_price_fraction_l2908_290879

/-- The price of one pair of high heels in dollars -/
def high_heels_price : ℚ := 60

/-- The number of pairs of ballet slippers bought -/
def ballet_slippers_count : ℕ := 5

/-- The total amount paid in dollars -/
def total_paid : ℚ := 260

/-- The fraction of the high heels price paid for each pair of ballet slippers -/
def ballet_slipper_fraction : ℚ := 2/3

theorem ballet_slipper_price_fraction :
  high_heels_price + ballet_slippers_count * (ballet_slipper_fraction * high_heels_price) = total_paid :=
by sorry

end NUMINAMATH_CALUDE_ballet_slipper_price_fraction_l2908_290879


namespace NUMINAMATH_CALUDE_equation_solution_l2908_290845

theorem equation_solution (x : ℝ) : 
  (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
   1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 4) ↔ 
  (x = 5 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2908_290845


namespace NUMINAMATH_CALUDE_tan_405_degrees_l2908_290858

theorem tan_405_degrees : Real.tan (405 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_405_degrees_l2908_290858


namespace NUMINAMATH_CALUDE_count_divisible_integers_l2908_290895

theorem count_divisible_integers : 
  ∃ (S : Finset Nat), 
    (∀ n ∈ S, n > 0 ∧ (8 * n) % ((n * (n + 1)) / 2) = 0) ∧ 
    (∀ n : Nat, n > 0 → (8 * n) % ((n * (n + 1)) / 2) = 0 → n ∈ S) ∧ 
    Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_integers_l2908_290895


namespace NUMINAMATH_CALUDE_triple_base_square_exponent_l2908_290857

theorem triple_base_square_exponent 
  (a b y : ℝ) 
  (hb : b ≠ 0) 
  (hr : (3 * a) ^ (2 * b) = a ^ b * y ^ b) : 
  y = 9 * a := 
sorry

end NUMINAMATH_CALUDE_triple_base_square_exponent_l2908_290857


namespace NUMINAMATH_CALUDE_sams_candy_count_l2908_290849

/-- Represents the candy count for each friend -/
structure CandyCounts where
  bob : ℕ
  mary : ℕ
  john : ℕ
  sue : ℕ
  sam : ℕ

/-- The total candy count for all friends -/
def totalCandy : ℕ := 50

/-- The given candy counts for Bob, Mary, John, and Sue -/
def givenCounts : CandyCounts where
  bob := 10
  mary := 5
  john := 5
  sue := 20
  sam := 0  -- We don't know Sam's count yet, so we initialize it to 0

/-- Theorem stating that Sam's candy count is equal to the total minus the sum of others -/
theorem sams_candy_count (c : CandyCounts) (h : c = givenCounts) :
  c.sam = totalCandy - (c.bob + c.mary + c.john + c.sue) :=
by
  sorry

#check sams_candy_count

end NUMINAMATH_CALUDE_sams_candy_count_l2908_290849


namespace NUMINAMATH_CALUDE_glee_club_female_members_l2908_290892

theorem glee_club_female_members 
  (total_members : ℕ) 
  (female_ratio : ℕ) 
  (male_ratio : ℕ) 
  (h1 : total_members = 18)
  (h2 : female_ratio = 2)
  (h3 : male_ratio = 1)
  (h4 : female_ratio * male_members + male_ratio * male_members = total_members)
  : female_ratio * male_members = 12 :=
by
  sorry

#check glee_club_female_members

end NUMINAMATH_CALUDE_glee_club_female_members_l2908_290892


namespace NUMINAMATH_CALUDE_complex_modulus_l2908_290827

theorem complex_modulus (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : 
  Complex.abs z = 2 * Real.sqrt 313 / 13 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_l2908_290827


namespace NUMINAMATH_CALUDE_total_money_is_36000_l2908_290871

/-- The number of phones Vivienne has -/
def vivienne_phones : ℕ := 40

/-- The difference in number of phones between Aliyah and Vivienne -/
def phone_difference : ℕ := 10

/-- The price of each phone -/
def phone_price : ℕ := 400

/-- The total amount of money Aliyah and Vivienne have together after selling their phones -/
def total_money : ℕ := (vivienne_phones + (vivienne_phones + phone_difference)) * phone_price

theorem total_money_is_36000 : total_money = 36000 := by
  sorry

end NUMINAMATH_CALUDE_total_money_is_36000_l2908_290871


namespace NUMINAMATH_CALUDE_fermat_prime_condition_l2908_290835

theorem fermat_prime_condition (n : ℕ) :
  Nat.Prime (2^n + 1) → (n = 0 ∨ ∃ α : ℕ, n = 2^α) :=
by sorry

end NUMINAMATH_CALUDE_fermat_prime_condition_l2908_290835


namespace NUMINAMATH_CALUDE_mrs_hilt_pizza_slices_l2908_290825

theorem mrs_hilt_pizza_slices : ∀ (num_pizzas slices_per_pizza : ℕ),
  num_pizzas = 2 →
  slices_per_pizza = 8 →
  num_pizzas * slices_per_pizza = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_pizza_slices_l2908_290825


namespace NUMINAMATH_CALUDE_age_problem_l2908_290836

theorem age_problem (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 72 → a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l2908_290836


namespace NUMINAMATH_CALUDE_circle_origin_range_l2908_290896

theorem circle_origin_range (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*m*x + 2*m*y + 2*m^2 - 4 = 0 → x^2 + y^2 < 4) → 
  -Real.sqrt 2 < m ∧ m < Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_circle_origin_range_l2908_290896


namespace NUMINAMATH_CALUDE_log_base_is_two_range_of_m_l2908_290810

noncomputable section

-- Define the logarithm function with base a
def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := log_base a x

-- Theorem 1: If f(x) = log_a(x), a > 0, a ≠ 1, and f(2) = 1, then f(x) = log_2(x)
theorem log_base_is_two (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 1) :
  ∀ x > 0, f a x = log_base 2 x :=
sorry

-- Theorem 2: For f(x) = log_2(x), the set of real numbers m satisfying f(m^2 - m) < 1 is (-1,0) ∪ (1,2)
theorem range_of_m (m : ℝ) :
  log_base 2 (m^2 - m) < 1 ↔ (m > -1 ∧ m < 0) ∨ (m > 1 ∧ m < 2) :=
sorry

end

end NUMINAMATH_CALUDE_log_base_is_two_range_of_m_l2908_290810


namespace NUMINAMATH_CALUDE_chord_length_l2908_290888

/-- The parabola M: y^2 = 2px where p > 0 -/
def parabola_M (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

/-- The circle C: x^2 + (y-4)^2 = a^2 -/
def circle_C (a : ℝ) (x y : ℝ) : Prop := x^2 + (y-4)^2 = a^2

/-- Point A is in the first quadrant -/
def point_A_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- Distance from A to focus of parabola M is a -/
def distance_A_to_focus (p a x y : ℝ) : Prop := (x - p/2)^2 + y^2 = a^2

/-- Sum of distances from a point on M to its directrix and to point C has a maximum of 2 -/
def max_distance_sum (p : ℝ) : Prop := ∃ (x y : ℝ), parabola_M p x y ∧ (x + p/2) + ((x - 0)^2 + (y - 4)^2).sqrt ≤ 2

/-- The theorem: Length of chord intercepted by line OA on circle C is 7√2/3 -/
theorem chord_length (p a x y : ℝ) : 
  parabola_M p x y → 
  circle_C a x y → 
  point_A_first_quadrant x y → 
  distance_A_to_focus p a x y → 
  max_distance_sum p → 
  ((2 * a)^2 - (8/3)^2).sqrt = 7 * Real.sqrt 2 / 3 := by sorry

end NUMINAMATH_CALUDE_chord_length_l2908_290888


namespace NUMINAMATH_CALUDE_registration_count_l2908_290868

/-- The number of ways two students can register for universities --/
def registration_possibilities : ℕ :=
  let n_universities := 3
  let n_students := 2
  let choose_one := n_universities
  let choose_two := n_universities.choose 2
  (choose_one ^ n_students) + 
  (choose_two ^ n_students) + 
  (2 * choose_one * choose_two)

theorem registration_count : registration_possibilities = 36 := by
  sorry

end NUMINAMATH_CALUDE_registration_count_l2908_290868


namespace NUMINAMATH_CALUDE_loss_60_l2908_290885

/-- Represents the financial recording of a transaction amount in dollars -/
def record_transaction (amount : Int) : Int := amount

/-- Records a profit of $370 as +370 dollars -/
axiom profit_370 : record_transaction 370 = 370

/-- Proves that a loss of $60 is recorded as -60 dollars -/
theorem loss_60 : record_transaction (-60) = -60 := by sorry

end NUMINAMATH_CALUDE_loss_60_l2908_290885


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l2908_290842

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 3 / 4 → 
  Nat.gcd a b = 8 → 
  Nat.lcm a b = 96 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l2908_290842


namespace NUMINAMATH_CALUDE_simplify_and_add_square_roots_l2908_290833

theorem simplify_and_add_square_roots :
  let x := Real.sqrt 726 / Real.sqrt 484
  let y := Real.sqrt 245 / Real.sqrt 147
  let z := Real.sqrt 1089 / Real.sqrt 441
  x + y + z = (87 + 14 * Real.sqrt 15) / 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_add_square_roots_l2908_290833


namespace NUMINAMATH_CALUDE_percentage_equality_l2908_290853

theorem percentage_equality (x y : ℝ) (p : ℝ) (h1 : x / y = 4) (h2 : p / 100 * x = 20 / 100 * y) : p = 5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l2908_290853


namespace NUMINAMATH_CALUDE_cookie_sales_revenue_l2908_290886

-- Define the sales data for each girl on each day
def robyn_day1_packs : ℕ := 25
def robyn_day1_price : ℚ := 4
def lucy_day1_packs : ℕ := 17
def lucy_day1_price : ℚ := 5

def robyn_day2_packs : ℕ := 15
def robyn_day2_price : ℚ := 7/2
def lucy_day2_packs : ℕ := 9
def lucy_day2_price : ℚ := 9/2

def robyn_day3_packs : ℕ := 23
def robyn_day3_price : ℚ := 9/2
def lucy_day3_packs : ℕ := 20
def lucy_day3_price : ℚ := 7/2

-- Define the total revenue calculation
def total_revenue : ℚ :=
  robyn_day1_packs * robyn_day1_price +
  lucy_day1_packs * lucy_day1_price +
  robyn_day2_packs * robyn_day2_price +
  lucy_day2_packs * lucy_day2_price +
  robyn_day3_packs * robyn_day3_price +
  lucy_day3_packs * lucy_day3_price

-- Theorem statement
theorem cookie_sales_revenue :
  total_revenue = 451.5 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sales_revenue_l2908_290886


namespace NUMINAMATH_CALUDE_fraction_equality_l2908_290875

theorem fraction_equality (m n : ℚ) (h : m / n = 2 / 3) : m / (m + n) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2908_290875


namespace NUMINAMATH_CALUDE_container_volume_tripled_l2908_290811

theorem container_volume_tripled (original_volume : ℝ) (h : original_volume = 2) :
  let new_volume := original_volume * 3 * 3 * 3
  new_volume = 54 := by
sorry

end NUMINAMATH_CALUDE_container_volume_tripled_l2908_290811


namespace NUMINAMATH_CALUDE_ac_length_l2908_290822

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  -- Points A, B, C
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Length conditions
  ab_length : dist A B = 7
  bc_length : dist B C = 24
  -- Area condition
  area : abs ((A.1 - C.1) * (B.2 - A.2) - (A.2 - C.2) * (B.1 - A.1)) / 2 = 84
  -- Median condition
  median_length : dist A ((B.1 + C.1) / 2, (B.2 + C.2) / 2) = 12.5

/-- Theorem about the length of AC in the special triangle -/
theorem ac_length (t : SpecialTriangle) : dist t.A t.C = 25 := by
  sorry

end NUMINAMATH_CALUDE_ac_length_l2908_290822


namespace NUMINAMATH_CALUDE_smallest_number_for_trailing_zeros_l2908_290839

def has_four_trailing_zeros (n : ℕ) : Prop :=
  n % 10000 = 0

theorem smallest_number_for_trailing_zeros :
  ∃ (n : ℕ), has_four_trailing_zeros (225 * 525 * n) ∧
  ∀ (m : ℕ), m < n → ¬has_four_trailing_zeros (225 * 525 * m) :=
by
  use 16
  sorry

end NUMINAMATH_CALUDE_smallest_number_for_trailing_zeros_l2908_290839


namespace NUMINAMATH_CALUDE_cricket_average_l2908_290834

theorem cricket_average (innings : ℕ) (next_runs : ℕ) (increase : ℕ) (current_average : ℕ) : 
  innings = 10 → 
  next_runs = 79 → 
  increase = 4 → 
  (innings * current_average + next_runs) / (innings + 1) = current_average + increase →
  current_average = 35 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_l2908_290834


namespace NUMINAMATH_CALUDE_division_remainder_problem_l2908_290873

theorem division_remainder_problem (dividend : Nat) (divisor : Nat) (quotient : Nat) 
  (h1 : dividend = 122)
  (h2 : divisor = 20)
  (h3 : quotient = 6)
  (h4 : dividend = divisor * quotient + (dividend % divisor)) :
  dividend % divisor = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l2908_290873


namespace NUMINAMATH_CALUDE_inverse_prop_parallel_lines_interior_angles_l2908_290847

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop := sorry

/-- Definition of interior alternate angles -/
def interior_alternate_angles_equal (l1 l2 : Line) : Prop := sorry

/-- The inverse proposition of a statement "if P, then Q" is "if Q, then P" -/
def inverse_proposition (P Q : Prop) : Prop :=
  (Q → P) = (¬P → ¬Q)

/-- Theorem stating the inverse proposition of the given statement -/
theorem inverse_prop_parallel_lines_interior_angles :
  inverse_proposition
    (∀ l1 l2 : Line, parallel l1 l2 → interior_alternate_angles_equal l1 l2)
    (∀ l1 l2 : Line, interior_alternate_angles_equal l1 l2 → parallel l1 l2) :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_prop_parallel_lines_interior_angles_l2908_290847


namespace NUMINAMATH_CALUDE_ellipse_foci_l2908_290870

/-- The equation of an ellipse in the form (x²/a² + y²/b² = 1) -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- The coordinates of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given ellipse equation (x²/2 + y² = 1), prove that its foci are at (±1, 0) -/
theorem ellipse_foci (e : Ellipse) (h : e.a^2 = 2 ∧ e.b^2 = 1) :
  ∃ (p₁ p₂ : Point), p₁.x = 1 ∧ p₁.y = 0 ∧ p₂.x = -1 ∧ p₂.y = 0 ∧
  (∀ (p : Point), (p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1) →
    (p = p₁ ∨ p = p₂ → 
      (p.x - 0)^2 + (p.y - 0)^2 = (e.a^2 - e.b^2))) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_l2908_290870


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2908_290840

theorem perfect_square_trinomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 + (a - 1)*x + 25 = (x + b)^2) → (a = 11 ∨ a = -9) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2908_290840


namespace NUMINAMATH_CALUDE_product_evaluation_l2908_290897

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2908_290897


namespace NUMINAMATH_CALUDE_quadratic_decreasing_l2908_290808

def f (x : ℝ) := -2 * (x - 1)^2

theorem quadratic_decreasing (x : ℝ) (h : x > 1) : 
  ∀ y, y > x → f y < f x := by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_l2908_290808


namespace NUMINAMATH_CALUDE_inequality_proof_l2908_290815

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧
  (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2908_290815


namespace NUMINAMATH_CALUDE_pyramid_cross_section_distance_l2908_290844

/-- Given a right octagonal pyramid with two cross sections parallel to the base,
    this theorem proves the distance of the larger cross section from the apex. -/
theorem pyramid_cross_section_distance
  (area_small area_large : ℝ)
  (height_diff : ℝ)
  (h_area_small : area_small = 256 * Real.sqrt 2)
  (h_area_large : area_large = 576 * Real.sqrt 2)
  (h_height_diff : height_diff = 12) :
  ∃ (h : ℝ), h = 36 ∧ 
    (area_small / area_large = (2/3)^2) ∧
    (h - 2/3 * h = height_diff) := by
  sorry

end NUMINAMATH_CALUDE_pyramid_cross_section_distance_l2908_290844


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l2908_290894

theorem quadratic_roots_product (a b : ℝ) : 
  (3 * a ^ 2 + 9 * a - 21 = 0) → 
  (3 * b ^ 2 + 9 * b - 21 = 0) → 
  (3 * a - 4) * (6 * b - 8) = -22 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l2908_290894


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2908_290804

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 6 * a 10 + a 3 * a 5 = 26 →
  a 5 * a 7 = 5 →
  a 4 + a 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2908_290804


namespace NUMINAMATH_CALUDE_rahul_ppf_savings_l2908_290869

/-- Represents Rahul's savings in rupees -/
structure RahulSavings where
  nsc : ℕ  -- National Savings Certificate
  ppf : ℕ  -- Public Provident Fund

/-- The conditions of Rahul's savings -/
def savingsConditions (s : RahulSavings) : Prop :=
  s.nsc + s.ppf = 180000 ∧ s.nsc / 3 = s.ppf / 2

/-- Theorem stating Rahul's Public Provident Fund savings -/
theorem rahul_ppf_savings (s : RahulSavings) (h : savingsConditions s) : s.ppf = 72000 := by
  sorry

#check rahul_ppf_savings

end NUMINAMATH_CALUDE_rahul_ppf_savings_l2908_290869


namespace NUMINAMATH_CALUDE_roots_relation_l2908_290898

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4

-- Define the polynomial j(x)
def j (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

-- Theorem statement
theorem roots_relation (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧ 
    h r₁ = 0 ∧ h r₂ = 0 ∧ h r₃ = 0) →
  (∀ x : ℝ, h x = 0 → j b c d (x^3) = 0) →
  b = 6 ∧ c = 12 ∧ d = 8 :=
by sorry

end NUMINAMATH_CALUDE_roots_relation_l2908_290898


namespace NUMINAMATH_CALUDE_power_negative_two_of_five_l2908_290865

theorem power_negative_two_of_five : 5^(-2 : ℤ) = (1 : ℚ) / 25 := by sorry

end NUMINAMATH_CALUDE_power_negative_two_of_five_l2908_290865


namespace NUMINAMATH_CALUDE_midpoint_distance_theorem_l2908_290826

theorem midpoint_distance_theorem (s : ℝ) : 
  let P : ℝ × ℝ := (s - 3, 2)
  let Q : ℝ × ℝ := (1, s + 2)
  let M : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  (M.1 - P.1)^2 + (M.2 - P.2)^2 = 3 * s^2 / 4 →
  s = -5 - 5 * Real.sqrt 2 ∨ s = -5 + 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_distance_theorem_l2908_290826


namespace NUMINAMATH_CALUDE_valuable_files_after_three_rounds_l2908_290821

def first_round_files : ℕ := 1200
def first_round_delete_percent : ℚ := 80 / 100
def second_round_files : ℕ := 600
def second_round_irrelevant_fraction : ℚ := 4 / 5
def final_round_files : ℕ := 700
def final_round_not_pertinent_percent : ℚ := 65 / 100

theorem valuable_files_after_three_rounds :
  let first_round_valuable := first_round_files - (first_round_files * first_round_delete_percent).floor
  let second_round_valuable := second_round_files - (second_round_files * second_round_irrelevant_fraction).floor
  let final_round_valuable := final_round_files - (final_round_files * final_round_not_pertinent_percent).floor
  first_round_valuable + second_round_valuable + final_round_valuable = 605 := by
sorry


end NUMINAMATH_CALUDE_valuable_files_after_three_rounds_l2908_290821


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l2908_290802

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  interval_start : ℕ
  interval_end : ℕ
  (population_positive : 0 < population)
  (sample_size_positive : 0 < sample_size)
  (sample_size_le_population : sample_size ≤ population)
  (interval_valid : interval_start ≤ interval_end)
  (interval_in_population : interval_end ≤ population)

/-- Calculates the number of selected individuals in the given interval -/
def selected_in_interval (s : SystematicSample) : ℕ :=
  ((s.interval_end - s.interval_start + 1) * s.sample_size + s.population - 1) / s.population

/-- Theorem stating that for the given systematic sample, 11 individuals are selected from the interval -/
theorem systematic_sample_theorem (s : SystematicSample) 
  (h1 : s.population = 640)
  (h2 : s.sample_size = 32)
  (h3 : s.interval_start = 161)
  (h4 : s.interval_end = 380) : 
  selected_in_interval s = 11 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l2908_290802


namespace NUMINAMATH_CALUDE_complement_of_29_45_l2908_290806

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

/-- The complement of an angle is the angle that when added to the original angle results in 90° -/
def complement (a : Angle) : Angle :=
  sorry

theorem complement_of_29_45 :
  complement ⟨29, 45⟩ = ⟨60, 15⟩ :=
sorry

end NUMINAMATH_CALUDE_complement_of_29_45_l2908_290806


namespace NUMINAMATH_CALUDE_inequality_proof_l2908_290852

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) :
  (a*f - c*d)^2 ≤ (a*e - b*d)^2 + (b*f - c*e)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2908_290852


namespace NUMINAMATH_CALUDE_number_of_hens_l2908_290882

/-- Given the following conditions:
  - The total cost of pigs and hens is 1200 Rs
  - There are 3 pigs
  - The average price of a hen is 30 Rs
  - The average price of a pig is 300 Rs
  Prove that the number of hens bought is 10. -/
theorem number_of_hens (total_cost : ℕ) (num_pigs : ℕ) (hen_price : ℕ) (pig_price : ℕ) 
  (h1 : total_cost = 1200)
  (h2 : num_pigs = 3)
  (h3 : hen_price = 30)
  (h4 : pig_price = 300) :
  ∃ (num_hens : ℕ), num_hens * hen_price + num_pigs * pig_price = total_cost ∧ num_hens = 10 :=
by sorry

end NUMINAMATH_CALUDE_number_of_hens_l2908_290882


namespace NUMINAMATH_CALUDE_sum_256_130_in_base6_l2908_290837

-- Define a function to convert a number from base 10 to base 6
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

-- State the theorem
theorem sum_256_130_in_base6 :
  toBase6 (256 + 130) = [1, 0, 4, 2] :=
sorry

end NUMINAMATH_CALUDE_sum_256_130_in_base6_l2908_290837


namespace NUMINAMATH_CALUDE_only_second_expression_always_true_l2908_290877

theorem only_second_expression_always_true :
  (∀ n a : ℝ, n * a^n = a) = False ∧
  (∀ a : ℝ, (a^2 - 3*a + 3)^0 = 1) = True ∧
  (3 - 3 = 6*(-3)^2) = False := by
sorry

end NUMINAMATH_CALUDE_only_second_expression_always_true_l2908_290877


namespace NUMINAMATH_CALUDE_percentage_of_girls_l2908_290859

theorem percentage_of_girls (total_students : ℕ) (num_boys : ℕ) : 
  total_students = 400 → num_boys = 80 → 
  (((total_students - num_boys : ℚ) / total_students) * 100 : ℚ) = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_girls_l2908_290859


namespace NUMINAMATH_CALUDE_cone_angle_l2908_290823

/-- Given a cone where the ratio of its lateral surface area to the area of the section through its axis
    is 2√3π/3, the angle between its generatrix and axis is π/6. -/
theorem cone_angle (r h l : ℝ) (θ : ℝ) : 
  r > 0 → h > 0 → l > 0 →
  (π * r * l) / (r * h) = 2 * Real.sqrt 3 * π / 3 →
  l = Real.sqrt ((r^2) + (h^2)) →
  θ = Real.arccos (h / l) →
  θ = π / 6 := by
  sorry

#check cone_angle

end NUMINAMATH_CALUDE_cone_angle_l2908_290823


namespace NUMINAMATH_CALUDE_cosine_like_properties_l2908_290883

-- Define the cosine-like function
def cosine_like (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y

-- State the theorem
theorem cosine_like_properties (f : ℝ → ℝ) 
  (h1 : cosine_like f) 
  (h2 : f 1 = 5/4)
  (h3 : ∀ t : ℝ, t ≠ 0 → f t > 1) :
  (f 0 = 1 ∧ f 2 = 17/8) ∧ 
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x₁ x₂ : ℚ, |x₁| < |x₂| → f x₁ < f x₂) := by
  sorry


end NUMINAMATH_CALUDE_cosine_like_properties_l2908_290883


namespace NUMINAMATH_CALUDE_equal_x_y_l2908_290863

theorem equal_x_y (x y z : ℝ) (h1 : x = 6 - y) (h2 : z^2 = x*y - 9) : x = y := by
  sorry

end NUMINAMATH_CALUDE_equal_x_y_l2908_290863


namespace NUMINAMATH_CALUDE_negative_one_minus_two_times_negative_two_l2908_290899

theorem negative_one_minus_two_times_negative_two : -1 - 2 * (-2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_minus_two_times_negative_two_l2908_290899


namespace NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_is_multiple_of_3_l2908_290830

theorem sum_of_multiples_of_6_and_9_is_multiple_of_3 (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 6 * k) (hb : ∃ m : ℤ, b = 9 * m) : 
  ∃ n : ℤ, a + b = 3 * n := by
sorry

end NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_is_multiple_of_3_l2908_290830


namespace NUMINAMATH_CALUDE_workshop_selection_l2908_290805

/-- The number of ways to select workers for a repair job. -/
def selectWorkers (totalWorkers fitters turners masterWorkers : ℕ) : ℕ :=
  let remainingWorkers := totalWorkers - turners
  let remainingFitters := fitters + masterWorkers
  let scenario1 := Nat.choose remainingWorkers 4
  let scenario2 := Nat.choose turners 3 * Nat.choose masterWorkers 1 * Nat.choose (remainingFitters - 1) 4
  let scenario3 := Nat.choose turners 2 * Nat.choose fitters 4
  scenario1 + scenario2 + scenario3

/-- Theorem stating the number of ways to select workers for the given problem. -/
theorem workshop_selection :
  selectWorkers 11 5 4 2 = 185 := by
  sorry

end NUMINAMATH_CALUDE_workshop_selection_l2908_290805


namespace NUMINAMATH_CALUDE_inequality_proof_binomial_inequality_l2908_290880

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a / Real.sqrt b + b / Real.sqrt a > Real.sqrt a + Real.sqrt b :=
by sorry

theorem binomial_inequality (x : ℝ) (m : ℕ) (hx : x > -1) (hm : m > 0) :
  (1 + x)^m ≥ 1 + m * x :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_binomial_inequality_l2908_290880


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2908_290820

theorem quadratic_function_property (a m : ℝ) (ha : a > 0) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + x + a
  f m < 0 → f (m + 1) > 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2908_290820


namespace NUMINAMATH_CALUDE_triangle_reciprocal_sum_l2908_290817

/-- Given a triangle ABC with angle ratio A:B:C = 4:2:1, 
    prove that 1/a + 1/b = 1/c, where a, b, and c are the 
    sides opposite to angles A, B, and C respectively. -/
theorem triangle_reciprocal_sum (A B C a b c : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_ratio : A = 4 * (π / 7) ∧ B = 2 * (π / 7) ∧ C = π / 7)
  (h_sides : a = 2 * Real.sin A ∧ b = 2 * Real.sin B ∧ c = 2 * Real.sin C) :
  1 / a + 1 / b = 1 / c :=
sorry

end NUMINAMATH_CALUDE_triangle_reciprocal_sum_l2908_290817


namespace NUMINAMATH_CALUDE_tetrahedron_coloring_tetrahedron_coloring_converse_l2908_290807

/-- The number of distinct colorings of a regular tetrahedron -/
def distinct_colorings (n : ℕ) : ℚ := (n^4 + 11*n^2) / 12

/-- The theorem stating the possible values of n -/
theorem tetrahedron_coloring (n : ℕ) (hn : n > 0) :
  distinct_colorings n = n^3 → n = 1 ∨ n = 11 := by
sorry

/-- The converse of the theorem -/
theorem tetrahedron_coloring_converse (n : ℕ) (hn : n > 0) :
  (n = 1 ∨ n = 11) → distinct_colorings n = n^3 := by
sorry

end NUMINAMATH_CALUDE_tetrahedron_coloring_tetrahedron_coloring_converse_l2908_290807


namespace NUMINAMATH_CALUDE_log_stack_sum_l2908_290862

theorem log_stack_sum : ∀ (a l : ℕ) (d : ℤ),
  a = 15 ∧ l = 4 ∧ d = -1 →
  ∃ n : ℕ, n > 0 ∧ l = a + (n - 1 : ℤ) * d ∧
  (n : ℤ) * (a + l) / 2 = 114 :=
by sorry

end NUMINAMATH_CALUDE_log_stack_sum_l2908_290862


namespace NUMINAMATH_CALUDE_tan_240_degrees_l2908_290829

theorem tan_240_degrees : Real.tan (240 * Real.pi / 180) = Real.sqrt 3 := by
  sorry

#check tan_240_degrees

end NUMINAMATH_CALUDE_tan_240_degrees_l2908_290829


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2908_290893

/-- Given a geometric sequence {a_n}, prove that a_4 * a_7 = -6 
    when a_1 and a_10 are roots of x^2 - x - 6 = 0 -/
theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∀ n : ℕ, ∃ r : ℝ, a (n + 1) = a n * r) →  -- geometric sequence condition
  (a 1)^2 - (a 1) - 6 = 0 →  -- a_1 is a root of x^2 - x - 6 = 0
  (a 10)^2 - (a 10) - 6 = 0 →  -- a_10 is a root of x^2 - x - 6 = 0
  a 4 * a 7 = -6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2908_290893


namespace NUMINAMATH_CALUDE_ryan_project_average_funding_l2908_290800

/-- The average amount each person funds to Ryan's project -/
def average_funding (total_goal : ℕ) (people : ℕ) (initial_funds : ℕ) : ℚ :=
  (total_goal - initial_funds : ℚ) / people

/-- Theorem: The average funding per person for Ryan's project is $10 -/
theorem ryan_project_average_funding :
  average_funding 1000 80 200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ryan_project_average_funding_l2908_290800


namespace NUMINAMATH_CALUDE_solution_set_implies_a_minus_b_l2908_290866

/-- The solution set of a quadratic inequality -/
def SolutionSet (a b : ℝ) : Set ℝ :=
  {x | a * x^2 + b * x + 2 > 0}

/-- The theorem stating the relationship between the solution set and the value of a - b -/
theorem solution_set_implies_a_minus_b (a b : ℝ) :
  SolutionSet a b = {x | -1/2 < x ∧ x < 1/3} → a - b = -10 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_minus_b_l2908_290866


namespace NUMINAMATH_CALUDE_units_digit_of_power_plus_six_l2908_290856

theorem units_digit_of_power_plus_six (y : ℕ+) :
  (7^y.val + 6) % 10 = 9 ↔ y.val % 4 = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_power_plus_six_l2908_290856


namespace NUMINAMATH_CALUDE_quadratic_non_real_roots_l2908_290803

theorem quadratic_non_real_roots (b : ℝ) : 
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_non_real_roots_l2908_290803


namespace NUMINAMATH_CALUDE_roots_of_equation_l2908_290850

def f (x : ℝ) := x^2 - |x - 1| - 1

theorem roots_of_equation :
  ∃ (x₁ x₂ : ℝ), x₁ > x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ = 1 ∧ x₂ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2908_290850


namespace NUMINAMATH_CALUDE_sequence_properties_l2908_290832

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => (7 * a n + Nat.sqrt (45 * (a n)^2 - 36)) / 2

theorem sequence_properties :
  (∀ n : ℕ, a n > 0) ∧
  (∀ n : ℕ, ∃ k : ℕ, a n * a (n + 1) - 1 = k^2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2908_290832


namespace NUMINAMATH_CALUDE_ratio_equation_solution_l2908_290828

theorem ratio_equation_solution (x y z a : ℤ) : 
  (∃ (k : ℤ), x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) →
  z = 30 * a - 15 →
  (∀ b : ℤ, 0 < b ∧ b < a → ¬(∃ (k : ℤ), 3 * k = 30 * b - 15)) →
  (∃ (k : ℤ), 3 * k = 30 * a - 15) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_l2908_290828


namespace NUMINAMATH_CALUDE_rectangle_ratio_l2908_290855

theorem rectangle_ratio (w l : ℝ) (h1 : w = 5) (h2 : l * w = 75) : l / w = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l2908_290855


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2908_290876

/-- The equation of the tangent line to y = x^2 + 1 at (-1, 2) is 2x + y = 0 -/
theorem tangent_line_equation : 
  let f : ℝ → ℝ := λ x => x^2 + 1
  let x₀ : ℝ := -1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (2*x + y = 0) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2908_290876


namespace NUMINAMATH_CALUDE_rectangle_area_l2908_290848

/-- Given a rectangle with perimeter 28 cm and length 9 cm, its area is 45 cm² -/
theorem rectangle_area (perimeter length : ℝ) (h1 : perimeter = 28) (h2 : length = 9) :
  let width := (perimeter - 2 * length) / 2
  length * width = 45 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2908_290848


namespace NUMINAMATH_CALUDE_polar_to_rect_transformation_l2908_290884

/-- Given a point (12, 5) in rectangular coordinates and (r, θ) in polar coordinates,
    prove that the point (r³, 3θ) in polar coordinates is (5600, -325) in rectangular coordinates. -/
theorem polar_to_rect_transformation (r θ : ℝ) :
  r * Real.cos θ = 12 →
  r * Real.sin θ = 5 →
  (r^3 * Real.cos (3*θ), r^3 * Real.sin (3*θ)) = (5600, -325) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rect_transformation_l2908_290884


namespace NUMINAMATH_CALUDE_josh_age_at_marriage_l2908_290813

/-- Proves that Josh's age at marriage was 22 given the conditions of the problem -/
theorem josh_age_at_marriage :
  ∀ (josh_age_at_marriage : ℕ),
    (josh_age_at_marriage + 30 + (28 + 30) = 5 * josh_age_at_marriage) →
    josh_age_at_marriage = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_josh_age_at_marriage_l2908_290813


namespace NUMINAMATH_CALUDE_hexagon_area_sum_l2908_290864

theorem hexagon_area_sum (u v : ℤ) (hu : 0 < u) (hv : 0 < v) (huv : v < u) :
  let A : ℤ × ℤ := (u, v)
  let B : ℤ × ℤ := (v, u)
  let C : ℤ × ℤ := (-v, u)
  let D : ℤ × ℤ := (-v, -u)
  let E : ℤ × ℤ := (v, -u)
  let F : ℤ × ℤ := (-u, -v)
  let hexagon_area := 8 * u * v + |u^2 - u*v - v^2|
  hexagon_area = 802 → u + v = 27 := by
sorry

end NUMINAMATH_CALUDE_hexagon_area_sum_l2908_290864


namespace NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficient_l2908_290838

theorem quadratic_rational_root_even_coefficient 
  (a b c : ℤ) (h_a_nonzero : a ≠ 0) 
  (h_rational_root : ∃ (p q : ℤ), q ≠ 0 ∧ a * (p / q)^2 + b * (p / q) + c = 0) :
  Even a ∨ Even b ∨ Even c :=
sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficient_l2908_290838


namespace NUMINAMATH_CALUDE_axis_of_symmetry_cosine_l2908_290814

theorem axis_of_symmetry_cosine (x : ℝ) : 
  ∀ k : ℤ, (2 * x + π / 3 = k * π) ↔ x = -π / 6 := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_cosine_l2908_290814


namespace NUMINAMATH_CALUDE_congruence_problem_l2908_290887

theorem congruence_problem (a b : ℤ) (h1 : a ≡ 27 [ZMOD 60]) (h2 : b ≡ 94 [ZMOD 60]) :
  ∃ n : ℤ, 150 ≤ n ∧ n ≤ 211 ∧ (a - b) ≡ n [ZMOD 60] ∧ n = 173 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l2908_290887


namespace NUMINAMATH_CALUDE_triangle_altitude_l2908_290878

theorem triangle_altitude (A : ℝ) (b : ℝ) (h : ℝ) :
  A = 750 →
  b = 50 →
  A = (1/2) * b * h →
  h = 30 := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_l2908_290878


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l2908_290867

theorem sphere_volume_ratio (r R : ℝ) (h : r > 0) (H : R > 0) :
  (4 * Real.pi * r^2) / (4 * Real.pi * R^2) = 4 / 9 →
  ((4 / 3) * Real.pi * r^3) / ((4 / 3) * Real.pi * R^3) = 8 / 27 :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l2908_290867


namespace NUMINAMATH_CALUDE_afternoon_eggs_calculation_l2908_290824

theorem afternoon_eggs_calculation (total_eggs day_eggs morning_eggs : ℕ) 
  (h1 : total_eggs = 1339)
  (h2 : morning_eggs = 816)
  (h3 : day_eggs = total_eggs - morning_eggs) : 
  day_eggs = 523 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_eggs_calculation_l2908_290824


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l2908_290851

open Real

noncomputable def f (x : ℝ) : ℝ := exp x + 1 / (exp x)

theorem f_increasing_on_interval (x : ℝ) (h : x > 1/exp 1) : 
  deriv f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l2908_290851


namespace NUMINAMATH_CALUDE_abs_neg_2022_l2908_290819

theorem abs_neg_2022 : |(-2022 : ℤ)| = 2022 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2022_l2908_290819


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l2908_290860

theorem reciprocal_sum_theorem (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) 
  (eq : 1 / x + 1 / y = 1 / z) : z = (x * y) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l2908_290860


namespace NUMINAMATH_CALUDE_baked_goods_distribution_l2908_290846

/-- Calculates the number of items not placed in containers --/
def itemsNotPlaced (totalItems : Nat) (itemsPerContainer : Nat) : Nat :=
  totalItems % itemsPerContainer

theorem baked_goods_distribution (gingerbreadCookies sugarCookies fruitTarts : Nat) 
  (gingerbreadPerJar sugarPerBox tartsPerBox : Nat) :
  gingerbreadCookies = 47 → 
  sugarCookies = 78 → 
  fruitTarts = 36 → 
  gingerbreadPerJar = 6 → 
  sugarPerBox = 9 → 
  tartsPerBox = 4 → 
  (itemsNotPlaced gingerbreadCookies gingerbreadPerJar = 5 ∧ 
   itemsNotPlaced sugarCookies sugarPerBox = 6 ∧ 
   itemsNotPlaced fruitTarts tartsPerBox = 0) := by
  sorry

#eval itemsNotPlaced 47 6  -- Should output 5
#eval itemsNotPlaced 78 9  -- Should output 6
#eval itemsNotPlaced 36 4  -- Should output 0

end NUMINAMATH_CALUDE_baked_goods_distribution_l2908_290846


namespace NUMINAMATH_CALUDE_two_valid_plans_l2908_290809

/-- The number of valid purchasing plans for notebooks and pens -/
def valid_purchasing_plans : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ 3 * p.1 + 5 * p.2 = 35) 
    (Finset.product (Finset.range 36) (Finset.range 36))).card

/-- Theorem stating that there are exactly 2 valid purchasing plans -/
theorem two_valid_plans : valid_purchasing_plans = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_valid_plans_l2908_290809


namespace NUMINAMATH_CALUDE_parallelogram_height_l2908_290861

/-- A parallelogram with given area and base has a specific height -/
theorem parallelogram_height (area base height : ℝ) (h_area : area = 375) (h_base : base = 25) :
  area = base * height → height = 15 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l2908_290861


namespace NUMINAMATH_CALUDE_trains_crossing_time_trains_crossing_time_approx_9_seconds_l2908_290801

/-- Time for trains to cross each other -/
theorem trains_crossing_time (train1_length train2_length : ℝ) 
  (train1_speed train2_speed : ℝ) : ℝ :=
  let total_length := train1_length + train2_length
  let relative_speed_kmh := train1_speed + train2_speed
  let relative_speed_ms := relative_speed_kmh * 1000 / 3600
  total_length / relative_speed_ms

/-- Proof that the time for the trains to cross is approximately 9 seconds -/
theorem trains_crossing_time_approx_9_seconds : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |trains_crossing_time 120.00001 380.03999 120 80 - 9| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_trains_crossing_time_trains_crossing_time_approx_9_seconds_l2908_290801


namespace NUMINAMATH_CALUDE_geometric_difference_sequence_properties_l2908_290812

/-- A geometric difference sequence -/
def GeometricDifferenceSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, (a (n + 2) / a (n + 1)) - (a (n + 1) / a n) = d

theorem geometric_difference_sequence_properties
  (a : ℕ → ℚ)
  (h_gds : GeometricDifferenceSequence a)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 1)
  (h_a3 : a 3 = 3) :
  a 5 = 105 ∧ a 31 / a 29 = 3363 := by
  sorry

end NUMINAMATH_CALUDE_geometric_difference_sequence_properties_l2908_290812


namespace NUMINAMATH_CALUDE_nonzero_real_equality_l2908_290818

theorem nonzero_real_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x = 1 + 1/y) (h2 : y = 1 + 1/x) : y = x := by
  sorry

end NUMINAMATH_CALUDE_nonzero_real_equality_l2908_290818


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l2908_290874

/-- Represents the number of bottle caps in various situations -/
structure BottleCaps where
  found : ℕ
  thrown_away : ℕ
  current : ℕ

/-- Given Danny's bottle cap collection data, prove that he found 1 more than he threw away -/
theorem danny_bottle_caps (caps : BottleCaps)
  (h1 : caps.found = 36)
  (h2 : caps.thrown_away = 35)
  (h3 : caps.current = 22) :
  caps.found - caps.thrown_away = 1 := by
  sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l2908_290874


namespace NUMINAMATH_CALUDE_fashion_line_blend_pieces_l2908_290872

theorem fashion_line_blend_pieces (silk_pieces : ℕ) (cashmere_pieces : ℕ) (total_pieces : ℕ) : 
  silk_pieces = 10 →
  cashmere_pieces = silk_pieces / 2 →
  total_pieces = 13 →
  cashmere_pieces - (total_pieces - silk_pieces) = 2 :=
by sorry

end NUMINAMATH_CALUDE_fashion_line_blend_pieces_l2908_290872


namespace NUMINAMATH_CALUDE_square_garden_area_l2908_290841

/-- Represents a square garden -/
structure SquareGarden where
  side : ℝ
  area : ℝ
  perimeter : ℝ

/-- Theorem: The area of a square garden is 90.25 square feet given the conditions -/
theorem square_garden_area
  (garden : SquareGarden)
  (h1 : garden.perimeter = 38)
  (h2 : garden.area = 2 * garden.perimeter + 14.25)
  : garden.area = 90.25 := by
  sorry

end NUMINAMATH_CALUDE_square_garden_area_l2908_290841


namespace NUMINAMATH_CALUDE_shopping_trip_tax_percentage_l2908_290891

/-- Represents the spending distribution and tax rates for a shopping trip -/
structure ShoppingTrip where
  clothing_percent : ℝ
  food_percent : ℝ
  other_percent : ℝ
  clothing_tax_rate : ℝ
  food_tax_rate : ℝ
  other_tax_rate : ℝ

/-- Calculates the total tax percentage for a given shopping trip -/
def totalTaxPercentage (trip : ShoppingTrip) : ℝ :=
  trip.clothing_percent * trip.clothing_tax_rate +
  trip.food_percent * trip.food_tax_rate +
  trip.other_percent * trip.other_tax_rate

/-- Theorem stating that for the given shopping trip, the total tax is 5% of the total amount spent excluding taxes -/
theorem shopping_trip_tax_percentage :
  let trip : ShoppingTrip := {
    clothing_percent := 0.5,
    food_percent := 0.2,
    other_percent := 0.3,
    clothing_tax_rate := 0.04,
    food_tax_rate := 0,
    other_tax_rate := 0.1
  }
  totalTaxPercentage trip = 0.05 := by
  sorry


end NUMINAMATH_CALUDE_shopping_trip_tax_percentage_l2908_290891


namespace NUMINAMATH_CALUDE_k_is_negative_l2908_290889

/-- A linear function y = x + k passes through a quadrant if there exists a point (x, y) in that quadrant satisfying the equation. -/
def passes_through_quadrant (k : ℝ) (quadrant : ℕ) : Prop :=
  match quadrant with
  | 1 => ∃ x > 0, x + k > 0
  | 3 => ∃ x < 0, x + k < 0
  | 4 => ∃ x > 0, x + k < 0
  | _ => False

/-- If the graph of y = x + k passes through the first, third, and fourth quadrants, then k < 0. -/
theorem k_is_negative (k : ℝ) 
  (h1 : passes_through_quadrant k 1)
  (h3 : passes_through_quadrant k 3)
  (h4 : passes_through_quadrant k 4) : 
  k < 0 := by
  sorry


end NUMINAMATH_CALUDE_k_is_negative_l2908_290889


namespace NUMINAMATH_CALUDE_snack_eaters_final_count_l2908_290881

/-- Calculates the number of remaining snack eaters after a series of events -/
def remaining_snack_eaters (initial_people : ℕ) (initial_snack_eaters : ℕ) 
  (first_new_outsiders : ℕ) (second_new_outsiders : ℕ) (second_group_leaving : ℕ) : ℕ :=
  let total_after_first_join := initial_snack_eaters + first_new_outsiders
  let remaining_after_first_leave := total_after_first_join / 2
  let total_after_second_join := remaining_after_first_leave + second_new_outsiders
  let remaining_after_second_leave := total_after_second_join - second_group_leaving
  remaining_after_second_leave / 2

theorem snack_eaters_final_count 
  (h1 : initial_people = 200)
  (h2 : initial_snack_eaters = 100)
  (h3 : first_new_outsiders = 20)
  (h4 : second_new_outsiders = 10)
  (h5 : second_group_leaving = 30) :
  remaining_snack_eaters initial_people initial_snack_eaters first_new_outsiders second_new_outsiders second_group_leaving = 20 := by
  sorry

end NUMINAMATH_CALUDE_snack_eaters_final_count_l2908_290881


namespace NUMINAMATH_CALUDE_teacher_distribution_l2908_290890

/-- The number of ways to distribute n distinct objects among k groups, 
    with each group receiving at least one object -/
def distribute (n k : ℕ) : ℕ := sorry

theorem teacher_distribution : distribute 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_teacher_distribution_l2908_290890


namespace NUMINAMATH_CALUDE_min_value_of_f_l2908_290831

noncomputable def f (x : ℝ) : ℝ := 9 / (8 * Real.cos (2 * x) + 16) - Real.sin x ^ 2

theorem min_value_of_f (x : ℝ) :
  f x ≥ 0 ∧ (f x = 0 ↔ Real.cos (2 * x) = -1/2) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2908_290831


namespace NUMINAMATH_CALUDE_solution_for_given_condition_l2908_290816

noncomputable def f (a x : ℝ) : ℝ := (a * x - 1) / (x^2 - 1)

theorem solution_for_given_condition (a : ℝ) :
  (∀ x, f a x > 0 ↔ a > 1/3) → ∃ x, x = 3 ∧ f (1/3) x = 0 :=
by sorry

end NUMINAMATH_CALUDE_solution_for_given_condition_l2908_290816
