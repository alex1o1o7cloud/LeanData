import Mathlib

namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l3794_379447

theorem simplify_and_evaluate_expression :
  let a : ℝ := 3 - Real.sqrt 2
  let expression := (((a^2 - 1) / (a - 3) - a - 1) / ((a + 1) / (a^2 - 6*a + 9)))
  expression = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l3794_379447


namespace NUMINAMATH_CALUDE_cylinder_radius_ratio_l3794_379435

/-- Given a right circular cylinder with initial volume 6 and final volume 186,
    prove that the ratio of the new radius to the original radius is √31. -/
theorem cylinder_radius_ratio (r R h : ℝ) : 
  r > 0 → h > 0 → 
  π * r^2 * h = 6 → 
  π * R^2 * h = 186 → 
  R / r = Real.sqrt 31 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_ratio_l3794_379435


namespace NUMINAMATH_CALUDE_square_area_with_perimeter_40_l3794_379448

theorem square_area_with_perimeter_40 :
  ∀ s : ℝ, s > 0 → 4 * s = 40 → s * s = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_perimeter_40_l3794_379448


namespace NUMINAMATH_CALUDE_consecutive_numbers_equality_l3794_379470

/-- Represents a list of digits forming a number -/
def DigitList := List Nat

/-- Converts a natural number to a list of its digits -/
def toDigitList (n : Nat) : DigitList :=
  if n < 10 then [n] else (n % 10) :: toDigitList (n / 10)

/-- Concatenates a list of natural numbers into a single number -/
def concatenateNumbers (nums : List Nat) : Nat :=
  nums.foldl (fun acc n => acc * (10 ^ (toDigitList n).length) + n) 0

theorem consecutive_numbers_equality :
  ∃ (start : Nat) (perm1 : List Nat) (perm2 : List Nat),
    perm1.length = 20 ∧
    perm2.length = 21 ∧
    (∀ n, n ∈ perm1 → start ≤ n ∧ n < start + 20) ∧
    (∀ n, n ∈ perm2 → start + 1 ≤ n ∧ n < start + 22) ∧
    concatenateNumbers perm1 = concatenateNumbers perm2 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_equality_l3794_379470


namespace NUMINAMATH_CALUDE_parabola_root_difference_l3794_379473

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- Check if a point lies on the parabola -/
def lies_on (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- The roots of the quadratic equation ax^2 + bx + c = 0 -/
def roots (p : Parabola) : ℝ × ℝ := sorry

theorem parabola_root_difference :
  ∀ p : Parabola,
  vertex p = (3, -9) →
  lies_on p 5 7 →
  let (m, n) := roots p
  m - n = 3 := by sorry

end NUMINAMATH_CALUDE_parabola_root_difference_l3794_379473


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3794_379438

theorem inequality_and_equality_condition 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ+) :
  let lhs := (a₁ * b₂ + a₁ * b₃ + a₂ * b₁ + a₂ * b₃ + a₃ * b₁ + a₃ * b₂)^2
  let rhs := 4 * (a₁ * a₂ + a₂ * a₃ + a₃ * a₁) * (b₁ * b₂ + b₂ * b₃ + b₃ * b₁)
  (lhs ≥ rhs) ∧ 
  (lhs = rhs ↔ (a₁ : ℚ) / b₁ = (a₂ : ℚ) / b₂ ∧ (a₂ : ℚ) / b₂ = (a₃ : ℚ) / b₃) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3794_379438


namespace NUMINAMATH_CALUDE_complex_product_real_imag_parts_l3794_379432

theorem complex_product_real_imag_parts : ∃ (m n : ℝ), 
  let Z : ℂ := (1 + Complex.I) * (2 + Complex.I^607)
  m = Z.re ∧ n = Z.im ∧ m * n = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_imag_parts_l3794_379432


namespace NUMINAMATH_CALUDE_symmetry_wrt_origin_l3794_379449

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem symmetry_wrt_origin :
  symmetric_point (4, -1) = (-4, 1) := by sorry

end NUMINAMATH_CALUDE_symmetry_wrt_origin_l3794_379449


namespace NUMINAMATH_CALUDE_magic_square_solution_l3794_379418

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a11 : ℚ
  a12 : ℚ
  a13 : ℚ
  a21 : ℚ
  a22 : ℚ
  a23 : ℚ
  a31 : ℚ
  a32 : ℚ
  a33 : ℚ
  sum_property : ∃ s : ℚ,
    a11 + a12 + a13 = s ∧
    a21 + a22 + a23 = s ∧
    a31 + a32 + a33 = s ∧
    a11 + a21 + a31 = s ∧
    a12 + a22 + a32 = s ∧
    a13 + a23 + a33 = s ∧
    a11 + a22 + a33 = s ∧
    a13 + a22 + a31 = s

/-- The theorem stating that y = 168.5 in the given magic square -/
theorem magic_square_solution :
  ∀ (ms : MagicSquare),
    ms.a11 = ms.a11 ∧  -- y (unknown)
    ms.a12 = 25 ∧
    ms.a13 = 81 ∧
    ms.a21 = 4 →
    ms.a11 = 168.5 := by
  sorry


end NUMINAMATH_CALUDE_magic_square_solution_l3794_379418


namespace NUMINAMATH_CALUDE_three_digit_powers_of_three_l3794_379400

theorem three_digit_powers_of_three (n : ℕ) : 
  (100 ≤ 3^n ∧ 3^n ≤ 999) ↔ (n = 5 ∨ n = 6) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_powers_of_three_l3794_379400


namespace NUMINAMATH_CALUDE_total_marbles_l3794_379453

/-- Given a bag of marbles with only red, blue, and yellow marbles, where the ratio of
    red:blue:yellow is 2:3:4, and there are 24 blue marbles, prove that the total number
    of marbles is 72. -/
theorem total_marbles (red blue yellow total : ℕ) : 
  red + blue + yellow = total →
  red = 2 * n ∧ blue = 3 * n ∧ yellow = 4 * n →
  blue = 24 →
  total = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l3794_379453


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l3794_379455

theorem point_in_first_quadrant (m : ℝ) : 
  let P : ℝ × ℝ := (3, m^2 + 1)
  P.1 > 0 ∧ P.2 > 0 := by
sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l3794_379455


namespace NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l3794_379482

theorem sin_sum_arcsin_arctan :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = 11 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l3794_379482


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_collinear_vectors_k_l3794_379469

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 3)
def c (m : ℝ) : ℝ × ℝ := (-2, m)

-- Define dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector addition for 2D vectors
def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define scalar multiplication for 2D vectors
def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Define collinearity for 2D vectors
def collinear (v w : ℝ × ℝ) : Prop := ∃ (k : ℝ), v = scalar_mult k w

-- Theorem 1
theorem perpendicular_vectors_m (m : ℝ) : 
  dot_product a (vector_add b (c m)) = 0 → m = -1 := by sorry

-- Theorem 2
theorem collinear_vectors_k (k : ℝ) :
  collinear (vector_add (scalar_mult k a) b) (vector_add (scalar_mult 2 a) (scalar_mult (-1) b)) → k = -2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_collinear_vectors_k_l3794_379469


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_k_equals_five_l3794_379474

/-- Given vectors a, b, and c in ℝ², prove that if (a - c) is parallel to b, then k = 5. -/
theorem parallel_vectors_imply_k_equals_five (a b c : ℝ × ℝ) (k : ℝ) :
  a = (3, 1) →
  b = (1, 3) →
  c = (k, 7) →
  (∃ t : ℝ, (a.1 - c.1, a.2 - c.2) = t • b) →
  k = 5 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_k_equals_five_l3794_379474


namespace NUMINAMATH_CALUDE_fraction_inequality_l3794_379488

theorem fraction_inequality (a b m : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hab : a < b) : 
  (a + m) / (b + m) > a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3794_379488


namespace NUMINAMATH_CALUDE_square_difference_equality_l3794_379443

theorem square_difference_equality : 1.99^2 - 1.98 * 1.99 + 0.99^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l3794_379443


namespace NUMINAMATH_CALUDE_painting_time_with_additional_workers_l3794_379491

/-- Given that n women can paint a house in h hours, and all women paint at the same rate,
    this theorem proves that n+s women can paint the same house in (n*h)/(n+s) hours. -/
theorem painting_time_with_additional_workers 
  (n : ℕ) 
  (s : ℕ) 
  (h : ℝ) 
  (h_pos : h > 0) 
  (n_pos : n > 0) :
  let original_time := h
  let original_workers := n
  let new_workers := n + s
  let new_time := (n * h) / (n + s)
  (original_time * original_workers = new_time * new_workers) ∧ 
  (new_time > 0) := by
  sorry


end NUMINAMATH_CALUDE_painting_time_with_additional_workers_l3794_379491


namespace NUMINAMATH_CALUDE_A_intersect_B_l3794_379487

def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | |x - 2| < 2}

theorem A_intersect_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l3794_379487


namespace NUMINAMATH_CALUDE_smallest_integer_l3794_379425

theorem smallest_integer (a b : ℕ) (ha : a = 36) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 20) :
  b ≥ 45 ∧ ∃ (b' : ℕ), b' = 45 ∧ Nat.lcm a b' / Nat.gcd a b' = 20 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_l3794_379425


namespace NUMINAMATH_CALUDE_square_fraction_count_l3794_379481

theorem square_fraction_count : 
  ∃! (count : ℕ), count = 2 ∧ 
    (∀ n : ℤ, (∃ k : ℤ, n / (30 - 2*n) = k^2) ↔ (n = 0 ∨ n = 10)) := by
  sorry

end NUMINAMATH_CALUDE_square_fraction_count_l3794_379481


namespace NUMINAMATH_CALUDE_train_speed_theorem_l3794_379403

/-- Theorem: Given two trains moving in opposite directions, with one train's speed being 100 kmph,
    lengths of 500 m and 700 m, and a crossing time of 19.6347928529354 seconds,
    the speed of the faster train is 100 kmph. -/
theorem train_speed_theorem (v_slow v_fast : ℝ) (length_slow length_fast : ℝ) (crossing_time : ℝ) :
  v_fast = 100 ∧
  length_slow = 500 ∧
  length_fast = 700 ∧
  crossing_time = 19.6347928529354 ∧
  (length_slow + length_fast) / 1000 / (crossing_time / 3600) = v_slow + v_fast →
  v_fast = 100 := by
  sorry

#check train_speed_theorem

end NUMINAMATH_CALUDE_train_speed_theorem_l3794_379403


namespace NUMINAMATH_CALUDE_grocery_problem_l3794_379465

theorem grocery_problem (total_packs : ℕ) (cookie_packs : ℕ) (noodle_packs : ℕ) :
  total_packs = 28 →
  cookie_packs = 12 →
  total_packs = cookie_packs + noodle_packs →
  noodle_packs = 16 := by
sorry

end NUMINAMATH_CALUDE_grocery_problem_l3794_379465


namespace NUMINAMATH_CALUDE_dirk_amulet_selling_days_l3794_379461

/-- Represents the problem of calculating the number of days Dirk sold amulets. -/
def amulet_problem (amulets_per_day : ℕ) (selling_price : ℚ) (cost_price : ℚ) 
  (faire_cut_percentage : ℚ) (total_profit : ℚ) : Prop :=
  let revenue_per_amulet : ℚ := selling_price
  let profit_per_amulet : ℚ := selling_price - cost_price
  let faire_cut_per_amulet : ℚ := faire_cut_percentage * revenue_per_amulet
  let net_profit_per_amulet : ℚ := profit_per_amulet - faire_cut_per_amulet
  let net_profit_per_day : ℚ := net_profit_per_amulet * amulets_per_day
  let days : ℚ := total_profit / net_profit_per_day
  days = 2

/-- Theorem stating the solution to Dirk's amulet selling problem. -/
theorem dirk_amulet_selling_days : 
  amulet_problem 25 40 30 (1/10) 300 := by
  sorry

end NUMINAMATH_CALUDE_dirk_amulet_selling_days_l3794_379461


namespace NUMINAMATH_CALUDE_unique_integer_triples_l3794_379414

theorem unique_integer_triples : 
  {(a, b, c) : ℕ × ℕ × ℕ | 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a ≤ b ∧ b ≤ c ∧
    a + b + c + a*b + b*c + c*a = a*b*c + 1} 
  = {(2, 5, 8), (3, 4, 13)} := by sorry

end NUMINAMATH_CALUDE_unique_integer_triples_l3794_379414


namespace NUMINAMATH_CALUDE_lindas_savings_l3794_379493

theorem lindas_savings (savings : ℝ) : 
  (3 / 4 : ℝ) * savings + 210 = savings → savings = 840 :=
by
  sorry

end NUMINAMATH_CALUDE_lindas_savings_l3794_379493


namespace NUMINAMATH_CALUDE_jacobs_graham_crackers_l3794_379437

/-- Represents the number of graham crackers needed for one s'more -/
def graham_crackers_per_smore : ℕ := 2

/-- Represents the number of marshmallows needed for one s'more -/
def marshmallows_per_smore : ℕ := 1

/-- Represents the number of marshmallows Jacob currently has -/
def current_marshmallows : ℕ := 6

/-- Represents the number of additional marshmallows Jacob needs to buy -/
def additional_marshmallows : ℕ := 18

/-- Theorem stating the number of graham crackers Jacob has -/
theorem jacobs_graham_crackers :
  (current_marshmallows + additional_marshmallows) * graham_crackers_per_smore = 48 := by
  sorry

end NUMINAMATH_CALUDE_jacobs_graham_crackers_l3794_379437


namespace NUMINAMATH_CALUDE_palindrome_square_base_l3794_379471

theorem palindrome_square_base (r : ℕ) (x : ℕ) : 
  x = r^3 + r^2 + r + 1 →
  Even r →
  ∃ (a b c d : ℕ), 
    (x^2 = a*r^7 + b*r^6 + c*r^5 + d*r^4 + d*r^3 + c*r^2 + b*r + a) ∧
    (b + c = 24) →
  r = 26 :=
sorry

end NUMINAMATH_CALUDE_palindrome_square_base_l3794_379471


namespace NUMINAMATH_CALUDE_percentage_failed_hindi_l3794_379413

theorem percentage_failed_hindi (failed_english : ℝ) (failed_both : ℝ) (passed_both : ℝ)
  (h1 : failed_english = 42)
  (h2 : failed_both = 28)
  (h3 : passed_both = 56) :
  ∃ (failed_hindi : ℝ), failed_hindi = 30 ∧
    failed_hindi + failed_english - failed_both = 100 - passed_both :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_failed_hindi_l3794_379413


namespace NUMINAMATH_CALUDE_triangle_side_length_l3794_379408

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if the area is √3, B = 60°, and a² + c² = 3ac, then b = 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →
  (B = π/3) →
  (a^2 + c^2 = 3*a*c) →
  (b = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3794_379408


namespace NUMINAMATH_CALUDE_total_age_is_32_l3794_379440

-- Define the ages of a, b, and c
def age_b : ℕ := 12
def age_a : ℕ := age_b + 2
def age_c : ℕ := age_b / 2

-- Theorem to prove
theorem total_age_is_32 : age_a + age_b + age_c = 32 := by
  sorry


end NUMINAMATH_CALUDE_total_age_is_32_l3794_379440


namespace NUMINAMATH_CALUDE_no_damaged_pool_floats_l3794_379442

/-- Prove that the number of damaged pool floats is 0 given the following conditions:
  - Total donations: 300
  - Basketball hoops: 60
  - Half of basketball hoops came with basketballs
  - Pool floats donated: 120
  - Footballs: 50
  - Tennis balls: 40
  - Remaining donations were basketballs
-/
theorem no_damaged_pool_floats (total_donations : ℕ) (basketball_hoops : ℕ) (pool_floats : ℕ)
  (footballs : ℕ) (tennis_balls : ℕ) (h1 : total_donations = 300)
  (h2 : basketball_hoops = 60) (h3 : pool_floats = 120) (h4 : footballs = 50) (h5 : tennis_balls = 40)
  (h6 : 2 * (basketball_hoops / 2) + pool_floats + footballs + tennis_balls +
    (total_donations - (basketball_hoops + pool_floats + footballs + tennis_balls)) = total_donations) :
  total_donations - (basketball_hoops + pool_floats + footballs + tennis_balls) = pool_floats := by
  sorry

#check no_damaged_pool_floats

end NUMINAMATH_CALUDE_no_damaged_pool_floats_l3794_379442


namespace NUMINAMATH_CALUDE_wire_cutting_l3794_379429

theorem wire_cutting (wire_length : ℚ) (num_parts : ℕ) :
  wire_length = 4/5 →
  num_parts = 3 →
  (wire_length / num_parts) / wire_length = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l3794_379429


namespace NUMINAMATH_CALUDE_johann_mail_delivery_l3794_379451

theorem johann_mail_delivery (total : ℕ) (friend_delivery : ℕ) (num_friends : ℕ) :
  total = 180 →
  friend_delivery = 41 →
  num_friends = 2 →
  total - (friend_delivery * num_friends) = 98 :=
by sorry

end NUMINAMATH_CALUDE_johann_mail_delivery_l3794_379451


namespace NUMINAMATH_CALUDE_q_investment_correct_l3794_379421

/-- Represents the investment of two people in a business -/
structure Business where
  p_investment : ℕ
  q_investment : ℕ
  profit_ratio : Rat

/-- The business scenario with given conditions -/
def given_business : Business where
  p_investment := 40000
  q_investment := 60000
  profit_ratio := 2 / 3

/-- Theorem stating that q's investment is correct given the conditions -/
theorem q_investment_correct (b : Business) : 
  b.p_investment = 40000 ∧ 
  b.profit_ratio = 2 / 3 → 
  b.q_investment = 60000 := by
  sorry

#check q_investment_correct given_business

end NUMINAMATH_CALUDE_q_investment_correct_l3794_379421


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l3794_379484

/-- The minimum distance between two points on different curves with the same y-coordinate --/
theorem min_distance_between_curves : ∃ (min_dist : ℝ),
  min_dist = 3/2 ∧
  ∀ (a : ℝ) (x₁ x₂ : ℝ),
    a = 2 * (x₁ + 1) →
    a = x₂ + Real.log x₂ →
    |x₂ - x₁| ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l3794_379484


namespace NUMINAMATH_CALUDE_negation_equivalence_l3794_379462

theorem negation_equivalence :
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 - 5*x + 6 > 0)) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 5*x + 6 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3794_379462


namespace NUMINAMATH_CALUDE_binomial_30_3_l3794_379452

theorem binomial_30_3 : (30 : ℕ).choose 3 = 4060 := by sorry

end NUMINAMATH_CALUDE_binomial_30_3_l3794_379452


namespace NUMINAMATH_CALUDE_morning_rowers_count_l3794_379441

def afternoon_rowers : ℕ := 17
def total_rowers : ℕ := 32

theorem morning_rowers_count : 
  total_rowers - afternoon_rowers = 15 := by sorry

end NUMINAMATH_CALUDE_morning_rowers_count_l3794_379441


namespace NUMINAMATH_CALUDE_unique_B_for_divisibility_l3794_379445

def is_divisible_by_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def number_4BBB2 (B : ℕ) : ℕ := 40000 + 1000 * B + 100 * B + 10 * B + 2

theorem unique_B_for_divisibility :
  ∃! B : ℕ, digit B ∧ is_divisible_by_9 (number_4BBB2 B) ∧ B = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_B_for_divisibility_l3794_379445


namespace NUMINAMATH_CALUDE_heather_emily_weight_difference_l3794_379478

/-- Given the weights of Heather and Emily, prove that Heather is 78 pounds heavier than Emily. -/
theorem heather_emily_weight_difference :
  let heather_weight : ℕ := 87
  let emily_weight : ℕ := 9
  heather_weight - emily_weight = 78 := by sorry

end NUMINAMATH_CALUDE_heather_emily_weight_difference_l3794_379478


namespace NUMINAMATH_CALUDE_ofelias_to_rileys_mistakes_ratio_l3794_379495

theorem ofelias_to_rileys_mistakes_ratio 
  (total_questions : ℕ) 
  (rileys_mistakes : ℕ) 
  (team_incorrect : ℕ) 
  (h1 : total_questions = 35)
  (h2 : rileys_mistakes = 3)
  (h3 : team_incorrect = 17) :
  (team_incorrect - rileys_mistakes) / rileys_mistakes = 14 / 3 := by
sorry

end NUMINAMATH_CALUDE_ofelias_to_rileys_mistakes_ratio_l3794_379495


namespace NUMINAMATH_CALUDE_complex_square_root_expression_l3794_379407

theorem complex_square_root_expression : 
  (2 * Real.sqrt 12 - 4 * Real.sqrt 27 + 3 * Real.sqrt 75 + 7 * Real.sqrt 8 - 3 * Real.sqrt 18) * 
  (4 * Real.sqrt 48 - 3 * Real.sqrt 27 - 5 * Real.sqrt 18 + 2 * Real.sqrt 50) = 97 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_expression_l3794_379407


namespace NUMINAMATH_CALUDE_inspection_result_l3794_379417

/-- Given a set of products and a selection for inspection, 
    we define the total number of items and the sample size. -/
def inspection_setup (total_products : ℕ) (selected : ℕ) : 
  (ℕ × ℕ) :=
  (total_products, selected)

/-- Theorem stating that for 50 products with 10 selected,
    the total number of items is 50 and the sample size is 10. -/
theorem inspection_result : 
  inspection_setup 50 10 = (50, 10) := by
  sorry

end NUMINAMATH_CALUDE_inspection_result_l3794_379417


namespace NUMINAMATH_CALUDE_difference_1500th_1504th_term_l3794_379486

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem difference_1500th_1504th_term : 
  let a₁ := 3
  let d := 6
  |arithmetic_sequence a₁ d 1504 - arithmetic_sequence a₁ d 1500| = 24 := by
  sorry

end NUMINAMATH_CALUDE_difference_1500th_1504th_term_l3794_379486


namespace NUMINAMATH_CALUDE_floor_abs_neq_abs_floor_exists_floor_diff_lt_floor_eq_implies_diff_lt_one_floor_inequality_solution_set_l3794_379404

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Proposition A (negation)
theorem floor_abs_neq_abs_floor : ∃ x : ℝ, floor (|x|) ≠ |floor x| :=
sorry

-- Proposition B
theorem exists_floor_diff_lt : ∃ x y : ℝ, floor (x - y) < floor x - floor y :=
sorry

-- Proposition C
theorem floor_eq_implies_diff_lt_one :
  ∀ x y : ℝ, floor x = floor y → x - y < 1 :=
sorry

-- Proposition D
theorem floor_inequality_solution_set :
  {x : ℝ | 2 * (floor x)^2 - floor x - 3 ≥ 0} = {x : ℝ | x < 0 ∨ x ≥ 2} :=
sorry

end NUMINAMATH_CALUDE_floor_abs_neq_abs_floor_exists_floor_diff_lt_floor_eq_implies_diff_lt_one_floor_inequality_solution_set_l3794_379404


namespace NUMINAMATH_CALUDE_congress_room_arrangement_l3794_379410

/-- A type representing delegates -/
def Delegate : Type := ℕ

/-- A relation representing the ability to communicate directly -/
def CanCommunicate : Delegate → Delegate → Prop := sorry

/-- The total number of delegates -/
def totalDelegates : ℕ := 1000

theorem congress_room_arrangement 
  (delegates : Finset Delegate) 
  (h_count : delegates.card = totalDelegates)
  (h_communication : ∀ (a b c : Delegate), a ∈ delegates → b ∈ delegates → c ∈ delegates → 
    (CanCommunicate a b ∨ CanCommunicate b c ∨ CanCommunicate a c)) :
  ∃ (pairs : List (Delegate × Delegate)), 
    (∀ (pair : Delegate × Delegate), pair ∈ pairs → CanCommunicate pair.1 pair.2) ∧ 
    (pairs.length = totalDelegates / 2) ∧
    (∀ (d : Delegate), d ∈ delegates ↔ (∃ (pair : Delegate × Delegate), pair ∈ pairs ∧ (d = pair.1 ∨ d = pair.2))) :=
sorry

end NUMINAMATH_CALUDE_congress_room_arrangement_l3794_379410


namespace NUMINAMATH_CALUDE_white_surface_area_fraction_l3794_379464

theorem white_surface_area_fraction (cube_edge : ℕ) (total_cubes : ℕ) (white_cubes : ℕ) (black_cubes : ℕ) : 
  cube_edge = 4 →
  total_cubes = 64 →
  white_cubes = 48 →
  black_cubes = 16 →
  (cube_edge : ℚ) * (cube_edge : ℚ) * 6 / ((cube_edge : ℚ) * (cube_edge : ℚ) * 6) - 
  ((3 * 8 + cube_edge * cube_edge) : ℚ) / ((cube_edge : ℚ) * (cube_edge : ℚ) * 6) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_area_fraction_l3794_379464


namespace NUMINAMATH_CALUDE_equation_value_l3794_379463

theorem equation_value : 
  let Y : ℝ := (180 * 0.15 - (180 * 0.15) / 3) + 0.245 * (2 / 3 * 270) - (5.4 * 2) / (0.25^2)
  Y = -110.7 := by
  sorry

end NUMINAMATH_CALUDE_equation_value_l3794_379463


namespace NUMINAMATH_CALUDE_seventieth_pair_is_4_9_l3794_379476

/-- Represents a pair of integers -/
structure IntPair :=
  (first : ℕ)
  (second : ℕ)

/-- Calculates the sum of the numbers in a pair -/
def pairSum (p : IntPair) : ℕ := p.first + p.second

/-- Generates the nth pair in the sequence -/
def nthPair (n : ℕ) : IntPair :=
  sorry

/-- Calculates the total number of pairs up to and including pairs with sum k -/
def totalPairsUpToSum (k : ℕ) : ℕ :=
  sorry

theorem seventieth_pair_is_4_9 : nthPair 70 = IntPair.mk 4 9 := by
  sorry

end NUMINAMATH_CALUDE_seventieth_pair_is_4_9_l3794_379476


namespace NUMINAMATH_CALUDE_number_of_boys_in_school_l3794_379402

theorem number_of_boys_in_school (total : ℕ) (boys : ℕ) :
  total = 1150 →
  (total - boys : ℚ) = (boys : ℚ) * total / 100 →
  boys = 92 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_in_school_l3794_379402


namespace NUMINAMATH_CALUDE_trapezoid_division_areas_l3794_379490

/-- Given a trapezoid with base length a, parallel side length b, and height m,
    when divided into three equal parts, prove that the areas of the resulting
    trapezoids are as stated. -/
theorem trapezoid_division_areas (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 0) :
  let s := m / 3
  let x := (2 * a + b) / 3
  let y := (a + 2 * b) / 3
  let t₁ := ((a + x) / 2) * s
  let t₂ := ((x + y) / 2) * s
  let t₃ := ((y + b) / 2) * s
  (t₁ = (5 * a + b) * m / 18) ∧
  (t₂ = (a + b) * m / 6) ∧
  (t₃ = (a + 5 * b) * m / 18) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_division_areas_l3794_379490


namespace NUMINAMATH_CALUDE_binary_1011_is_11_l3794_379460

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, x) => acc + if x then 2^i else 0) 0

theorem binary_1011_is_11 :
  binary_to_decimal [true, true, false, true] = 11 := by
  sorry

end NUMINAMATH_CALUDE_binary_1011_is_11_l3794_379460


namespace NUMINAMATH_CALUDE_starting_number_is_eight_l3794_379428

def is_valid_start (n : ℕ) : Prop :=
  n ≤ 38 ∧ n % 4 = 0

def numbers_between (n : ℕ) : List ℕ :=
  (List.range ((38 - n) / 4 + 1)).map (fun i => n + 4 * i)

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem starting_number_is_eight (n : ℕ) (h1 : is_valid_start n) 
    (h2 : average (numbers_between n) = 22) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_starting_number_is_eight_l3794_379428


namespace NUMINAMATH_CALUDE_emily_egg_collection_l3794_379422

/-- The number of baskets Emily used -/
def num_baskets : ℕ := 303

/-- The number of eggs in each basket -/
def eggs_per_basket : ℕ := 28

/-- The total number of eggs Emily collected -/
def total_eggs : ℕ := num_baskets * eggs_per_basket

theorem emily_egg_collection : total_eggs = 8484 := by
  sorry

end NUMINAMATH_CALUDE_emily_egg_collection_l3794_379422


namespace NUMINAMATH_CALUDE_price_restoration_l3794_379489

theorem price_restoration (original_price : ℝ) (h : original_price > 0) :
  let reduced_price := 0.8 * original_price
  (reduced_price * 1.25 = original_price) := by
sorry

end NUMINAMATH_CALUDE_price_restoration_l3794_379489


namespace NUMINAMATH_CALUDE_arden_cricket_club_members_l3794_379479

/-- The cost of a pair of gloves in dollars -/
def glove_cost : ℕ := 6

/-- The additional cost of a cap compared to a pair of gloves in dollars -/
def cap_additional_cost : ℕ := 8

/-- The total expenditure of the club in dollars -/
def total_expenditure : ℕ := 4140

/-- The number of gloves and caps each member needs -/
def items_per_member : ℕ := 2

theorem arden_cricket_club_members :
  ∃ (n : ℕ), n * (items_per_member * (glove_cost + (glove_cost + cap_additional_cost))) = total_expenditure ∧
  n = 103 := by
  sorry

end NUMINAMATH_CALUDE_arden_cricket_club_members_l3794_379479


namespace NUMINAMATH_CALUDE_polygon_with_17_diagonals_has_8_sides_l3794_379446

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 17 diagonals has 8 sides -/
theorem polygon_with_17_diagonals_has_8_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 17 → n = 8 :=
by sorry

end NUMINAMATH_CALUDE_polygon_with_17_diagonals_has_8_sides_l3794_379446


namespace NUMINAMATH_CALUDE_square_sum_equals_90_l3794_379459

theorem square_sum_equals_90 (x y : ℝ) 
  (h1 : x * (2 * x + y) = 18) 
  (h2 : y * (2 * x + y) = 72) : 
  (2 * x + y)^2 = 90 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_90_l3794_379459


namespace NUMINAMATH_CALUDE_bones_fraction_in_beef_l3794_379430

/-- The price of beef with bones in rubles per kilogram -/
def price_beef_with_bones : ℝ := 78

/-- The price of boneless beef in rubles per kilogram -/
def price_boneless_beef : ℝ := 90

/-- The price of bones in rubles per kilogram -/
def price_bones : ℝ := 15

/-- The fraction of bones in a kilogram of beef -/
def fraction_bones : ℝ := 0.16

theorem bones_fraction_in_beef :
  price_bones * fraction_bones + price_boneless_beef * (1 - fraction_bones) = price_beef_with_bones :=
sorry

end NUMINAMATH_CALUDE_bones_fraction_in_beef_l3794_379430


namespace NUMINAMATH_CALUDE_sum_divisible_by_five_l3794_379466

theorem sum_divisible_by_five (m : ℤ) : 5 ∣ ((10 - m) + (m + 5)) := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_five_l3794_379466


namespace NUMINAMATH_CALUDE_cafe_latte_cost_correct_l3794_379406

/-- Represents the cost of a cafe latte -/
def cafe_latte_cost : ℝ := 1.50

/-- Represents the cost of a cappuccino -/
def cappuccino_cost : ℝ := 2

/-- Represents the cost of an iced tea -/
def iced_tea_cost : ℝ := 3

/-- Represents the cost of an espresso -/
def espresso_cost : ℝ := 1

/-- Represents the number of cappuccinos Sandy ordered -/
def num_cappuccinos : ℕ := 3

/-- Represents the number of iced teas Sandy ordered -/
def num_iced_teas : ℕ := 2

/-- Represents the number of cafe lattes Sandy ordered -/
def num_lattes : ℕ := 2

/-- Represents the number of espressos Sandy ordered -/
def num_espressos : ℕ := 2

/-- Represents the amount Sandy paid -/
def amount_paid : ℝ := 20

/-- Represents the change Sandy received -/
def change_received : ℝ := 3

theorem cafe_latte_cost_correct :
  cafe_latte_cost * num_lattes +
  cappuccino_cost * num_cappuccinos +
  iced_tea_cost * num_iced_teas +
  espresso_cost * num_espressos =
  amount_paid - change_received :=
by sorry

end NUMINAMATH_CALUDE_cafe_latte_cost_correct_l3794_379406


namespace NUMINAMATH_CALUDE_infinite_integers_with_noncounting_divisors_l3794_379480

theorem infinite_integers_with_noncounting_divisors :
  ∃ (S : Set ℕ), Set.Infinite S ∧ 
    (∀ a ∈ S, a ≥ 1 ∧ 
      (∀ n : ℕ, n ≥ 1 → ¬(Nat.card (Nat.divisors a) = n))) := by
  sorry

end NUMINAMATH_CALUDE_infinite_integers_with_noncounting_divisors_l3794_379480


namespace NUMINAMATH_CALUDE_product_sum_difference_theorem_l3794_379401

theorem product_sum_difference_theorem (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x * y = 2688) 
  (h4 : x = 84) : 
  (x + y) - (x - y) = 64 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_difference_theorem_l3794_379401


namespace NUMINAMATH_CALUDE_ellipse_vertices_distance_l3794_379434

/-- Given an ellipse with equation (x^2 / 45) + (y^2 / 11) = 1, 
    the distance between its vertices is 6√5 -/
theorem ellipse_vertices_distance : 
  ∀ (x y : ℝ), x^2/45 + y^2/11 = 1 → 
  ∃ (d : ℝ), d = 6 * Real.sqrt 5 ∧ d = 2 * Real.sqrt (max 45 11) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_vertices_distance_l3794_379434


namespace NUMINAMATH_CALUDE_complex_transformation_l3794_379456

/-- The result of applying a 60° counter-clockwise rotation followed by a dilation 
    with scale factor 2 to the complex number -4 + 3i -/
theorem complex_transformation : 
  let z : ℂ := -4 + 3 * Complex.I
  let rotation : ℂ := Complex.exp (Complex.I * Real.pi / 3)
  let dilation : ℝ := 2
  (dilation * rotation * z) = (-4 - 3 * Real.sqrt 3) + (3 - 4 * Real.sqrt 3) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_transformation_l3794_379456


namespace NUMINAMATH_CALUDE_box_volume_l3794_379450

theorem box_volume (x : ℕ+) :
  (5 * x) * (5 * (x + 1)) * (5 * (x + 2)) = 25 * x^3 + 50 * x^2 + 125 * x :=
by sorry

end NUMINAMATH_CALUDE_box_volume_l3794_379450


namespace NUMINAMATH_CALUDE_train_length_proof_l3794_379454

/-- Proves that a train crossing a 550-meter platform in 51 seconds and a signal pole in 18 seconds has a length of 300 meters. -/
theorem train_length_proof (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
  (h1 : platform_length = 550)
  (h2 : platform_time = 51)
  (h3 : pole_time = 18) :
  let train_length := (platform_length * pole_time) / (platform_time - pole_time)
  train_length = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l3794_379454


namespace NUMINAMATH_CALUDE_x_value_proof_l3794_379419

theorem x_value_proof (x : ℝ) (h : x^2 * 8^3 / 256 = 450) : x = 15 ∨ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3794_379419


namespace NUMINAMATH_CALUDE_number_equality_l3794_379483

theorem number_equality (x : ℝ) : (2 * x + 20 = 8 * x - 4) ↔ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l3794_379483


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3794_379405

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 1 - a 4 - a 8 - a 12 + a 15 = 2) →
  (a 3 + a 13 = -4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3794_379405


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l3794_379423

def f (x : ℝ) : ℝ := -x * abs x

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f y < f x) :=
sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l3794_379423


namespace NUMINAMATH_CALUDE_simplify_expression_l3794_379492

theorem simplify_expression :
  1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 1)) =
  ((Real.sqrt 3 - 2 * Real.sqrt 5 - 1) * (-16 - 2 * Real.sqrt 3)) / 244 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3794_379492


namespace NUMINAMATH_CALUDE_angle_measure_60_degrees_l3794_379427

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to A, B, C respectively

-- State the theorem
theorem angle_measure_60_degrees (t : Triangle) 
  (h : t.b^2 + t.c^2 - t.a^2 = t.b * t.c) : 
  t.A = 60 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_60_degrees_l3794_379427


namespace NUMINAMATH_CALUDE_clarissa_photos_count_l3794_379497

/-- The number of photos brought by Cristina -/
def cristina_photos : ℕ := 7

/-- The number of photos brought by John -/
def john_photos : ℕ := 10

/-- The number of photos brought by Sarah -/
def sarah_photos : ℕ := 9

/-- The total number of slots in the photo album -/
def album_slots : ℕ := 40

/-- The number of photos Clarissa needs to bring -/
def clarissa_photos : ℕ := album_slots - (cristina_photos + john_photos + sarah_photos)

theorem clarissa_photos_count : clarissa_photos = 14 := by
  sorry

end NUMINAMATH_CALUDE_clarissa_photos_count_l3794_379497


namespace NUMINAMATH_CALUDE_towel_shrinkage_l3794_379409

/-- Given a rectangular towel that loses 20% of its length and has a total area
    decrease of 27.999999999999993%, the percentage decrease in breadth is 10%. -/
theorem towel_shrinkage (L B : ℝ) (L' B' : ℝ) : 
  L' = 0.8 * L →
  L' * B' = 0.72 * (L * B) →
  B' = 0.9 * B :=
by sorry

end NUMINAMATH_CALUDE_towel_shrinkage_l3794_379409


namespace NUMINAMATH_CALUDE_age_difference_proof_l3794_379457

theorem age_difference_proof (younger_age elder_age : ℕ) 
  (h1 : elder_age > younger_age)
  (h2 : elder_age - 4 = 5 * (younger_age - 4))
  (h3 : younger_age = 29)
  (h4 : elder_age = 49) :
  elder_age - younger_age = 20 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l3794_379457


namespace NUMINAMATH_CALUDE_magpie_porridge_l3794_379424

/-- The amount of porridge given to each chick -/
def PorridgeDistribution (p₁ p₂ p₃ p₄ p₅ p₆ : ℝ) : Prop :=
  p₃ = p₁ + p₂ ∧
  p₄ = p₂ + p₃ ∧
  p₅ = p₃ + p₄ ∧
  p₆ = p₄ + p₅ ∧
  p₅ = 10

theorem magpie_porridge : 
  ∀ p₁ p₂ p₃ p₄ p₅ p₆ : ℝ, 
  PorridgeDistribution p₁ p₂ p₃ p₄ p₅ p₆ → 
  p₁ + p₂ + p₃ + p₄ + p₅ + p₆ = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_magpie_porridge_l3794_379424


namespace NUMINAMATH_CALUDE_merchant_markup_percentage_l3794_379468

/-- The percentage of the list price at which goods should be marked to achieve
    the desired profit and discount conditions. -/
theorem merchant_markup_percentage 
  (list_price : ℝ) 
  (purchase_discount : ℝ) 
  (selling_discount : ℝ) 
  (profit_percentage : ℝ) 
  (h1 : purchase_discount = 0.2)
  (h2 : selling_discount = 0.2)
  (h3 : profit_percentage = 0.2)
  : ∃ (markup_percentage : ℝ),
    markup_percentage = 1.25 ∧ 
    (1 - purchase_discount) * list_price = 
    (1 - profit_percentage) * ((1 - selling_discount) * (markup_percentage * list_price)) :=
by sorry

end NUMINAMATH_CALUDE_merchant_markup_percentage_l3794_379468


namespace NUMINAMATH_CALUDE_correct_algebraic_expression_l3794_379431

-- Define the set of possible expressions
inductive AlgebraicExpression
  | MixedNumber : AlgebraicExpression  -- represents 1½a
  | ExplicitMultiplication : AlgebraicExpression  -- represents a × b
  | DivisionSign : AlgebraicExpression  -- represents a ÷ b
  | ImplicitMultiplication : AlgebraicExpression  -- represents 2a

-- Define the property of being correctly written
def isCorrectlyWritten (expr : AlgebraicExpression) : Prop :=
  match expr with
  | AlgebraicExpression.ImplicitMultiplication => True
  | _ => False

-- Theorem statement
theorem correct_algebraic_expression :
  isCorrectlyWritten AlgebraicExpression.ImplicitMultiplication ∧
  ¬isCorrectlyWritten AlgebraicExpression.MixedNumber ∧
  ¬isCorrectlyWritten AlgebraicExpression.ExplicitMultiplication ∧
  ¬isCorrectlyWritten AlgebraicExpression.DivisionSign :=
sorry

end NUMINAMATH_CALUDE_correct_algebraic_expression_l3794_379431


namespace NUMINAMATH_CALUDE_difference_of_squares_l3794_379475

theorem difference_of_squares (a b : ℝ) : (a + 2*b) * (a - 2*b) = a^2 - 4*b^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3794_379475


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l3794_379467

theorem square_perimeter_ratio (a₁ a₂ p₁ p₂ : ℝ) (h_positive : a₁ > 0 ∧ a₂ > 0) 
  (h_area_ratio : a₁ / a₂ = 49 / 64) (h_perimeter₁ : p₁ = 4 * Real.sqrt a₁) 
  (h_perimeter₂ : p₂ = 4 * Real.sqrt a₂) : p₁ / p₂ = 7 / 8 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l3794_379467


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3794_379458

/-- A quadratic function f(x) = x^2 + bx + c where f(-1) = f(3) -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_inequality (b c : ℝ) :
  f b c (-1) = f b c 3 →
  f b c 1 < c ∧ c < f b c (-1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3794_379458


namespace NUMINAMATH_CALUDE_f_inequality_l3794_379472

open Real

/-- The function f(x) = x ln x -/
noncomputable def f (x : ℝ) : ℝ := x * log x

/-- The derivative of f(x) -/
noncomputable def f_deriv (x : ℝ) : ℝ := 1 + log x

theorem f_inequality (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : x₁ ≠ x₂) :
  (f x₂ - f x₁) / (x₂ - x₁) < f_deriv ((x₁ + x₂) / 2) :=
sorry

end NUMINAMATH_CALUDE_f_inequality_l3794_379472


namespace NUMINAMATH_CALUDE_books_read_total_l3794_379426

theorem books_read_total (may june july : ℕ) 
  (h_may : may = 2) 
  (h_june : june = 6) 
  (h_july : july = 10) : 
  may + june + july = 18 := by
  sorry

end NUMINAMATH_CALUDE_books_read_total_l3794_379426


namespace NUMINAMATH_CALUDE_root_difference_nonnegative_root_difference_l3794_379420

theorem root_difference (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → |r₁ - r₂| = Real.sqrt (b^2 - 4*a*c) / a :=
by sorry

theorem nonnegative_root_difference :
  let eq := fun x : ℝ ↦ x^2 + 40*x + 300
  ∃ r₁ r₂ : ℝ, eq r₁ = 0 ∧ eq r₂ = 0 ∧ |r₁ - r₂| = 20 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_nonnegative_root_difference_l3794_379420


namespace NUMINAMATH_CALUDE_exponential_increasing_condition_l3794_379498

theorem exponential_increasing_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → a^x < a^y) ↔ a > 1 := by sorry

end NUMINAMATH_CALUDE_exponential_increasing_condition_l3794_379498


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l3794_379433

theorem complex_sum_theorem (a c d e f g : ℝ) : 
  let b : ℝ := 5
  let e : ℝ := -(a + c) + g
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) = g + 9 * Complex.I →
  d + f = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l3794_379433


namespace NUMINAMATH_CALUDE_stating_min_messages_proof_l3794_379444

/-- Represents the minimum number of messages needed for information distribution -/
def min_messages (n : ℕ) : ℕ := 2 * (n - 1)

/-- 
Theorem stating that the minimum number of messages needed for n people 
to share all information is 2(n-1)
-/
theorem min_messages_proof (n : ℕ) (h : n > 0) : 
  ∀ (f : ℕ → ℕ), 
  (∀ i : ℕ, i < n → f i ≥ min_messages n) → 
  (∃ g : ℕ → ℕ → Bool, 
    (∀ i j : ℕ, i < n ∧ j < n → g i j = true) ∧ 
    (∀ i : ℕ, i < n → ∃ k : ℕ, k < f i ∧ 
      (∀ j : ℕ, j < n → ∃ m : ℕ, m ≤ k ∧ g i j = true))) :=
sorry

#check min_messages_proof

end NUMINAMATH_CALUDE_stating_min_messages_proof_l3794_379444


namespace NUMINAMATH_CALUDE_symmetric_difference_properties_l3794_379412

open Set

variable {α : Type*} [MeasurableSpace α]

def symmetricDifference (A B : Set α) : Set α := (A \ B) ∪ (B \ A)

theorem symmetric_difference_properties 
  (A B : ℕ → Set α) : 
  (symmetricDifference (A 1) (B 1) = symmetricDifference (Aᶜ 1) (Bᶜ 1)) ∧ 
  (symmetricDifference (⋃ n, A n) (⋃ n, B n) ⊆ ⋃ n, symmetricDifference (A n) (B n)) ∧
  (symmetricDifference (⋂ n, A n) (⋂ n, B n) ⊆ ⋃ n, symmetricDifference (A n) (B n)) := by
  sorry


end NUMINAMATH_CALUDE_symmetric_difference_properties_l3794_379412


namespace NUMINAMATH_CALUDE_high_school_students_l3794_379411

theorem high_school_students (total : ℕ) (ratio : ℕ) (mia zoe : ℕ) : 
  total = 2500 →
  ratio = 4 →
  mia = ratio * zoe →
  mia + zoe = total →
  mia = 2000 := by
sorry

end NUMINAMATH_CALUDE_high_school_students_l3794_379411


namespace NUMINAMATH_CALUDE_log_equation_solution_l3794_379416

theorem log_equation_solution (p q : ℝ) 
  (h1 : Real.log p + Real.log (q^2) = Real.log (p + 3*q^2))
  (h2 : p ≠ -3*q^2)
  (h3 : q ≠ 0)
  (h4 : q^2 ≠ 1) : 
  p = 3*q^2 / (q^2 - 1) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3794_379416


namespace NUMINAMATH_CALUDE_sheridan_fish_proof_l3794_379496

/-- The number of fish Mrs. Sheridan gave to her sister -/
def fish_given_away : ℝ := 22.0

/-- The number of fish Mrs. Sheridan has now -/
def fish_remaining : ℕ := 25

/-- The initial number of fish Mrs. Sheridan had -/
def initial_fish : ℝ := fish_given_away + fish_remaining

theorem sheridan_fish_proof : initial_fish = 47.0 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_fish_proof_l3794_379496


namespace NUMINAMATH_CALUDE_sequence_properties_l3794_379499

def geometric_sequence (i : ℕ) : ℕ := 7 * 3^(16 - i) * 5^(i - 1)

theorem sequence_properties :
  (∀ i ∈ Finset.range 16, geometric_sequence (i + 1) > 0) ∧
  (∀ i ∈ Finset.range 5, geometric_sequence (i + 1) ≥ 10^8 ∧ geometric_sequence (i + 1) < 10^9) ∧
  (∀ i ∈ Finset.range 5, geometric_sequence (i + 6) ≥ 10^9 ∧ geometric_sequence (i + 6) < 10^10) ∧
  (∀ i ∈ Finset.range 4, geometric_sequence (i + 11) ≥ 10^10 ∧ geometric_sequence (i + 11) < 10^11) ∧
  (∀ i ∈ Finset.range 2, geometric_sequence (i + 15) ≥ 10^11 ∧ geometric_sequence (i + 15) < 10^12) ∧
  (∀ i ∈ Finset.range 15, geometric_sequence (i + 2) / geometric_sequence (i + 1) = geometric_sequence 2 / geometric_sequence 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3794_379499


namespace NUMINAMATH_CALUDE_symmetric_point_exists_l3794_379494

def S : Set ℤ := {n : ℤ | ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ n = 19*a + 85*b}

theorem symmetric_point_exists : 
  ∃ (A : ℝ), ∀ (x y : ℤ), (x + y : ℝ) / 2 = A → 
    (x ∈ S ↔ y ∉ S) :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_exists_l3794_379494


namespace NUMINAMATH_CALUDE_train_arrival_interval_l3794_379415

def train_interval (passengers_per_hour : ℕ) (passengers_left : ℕ) (passengers_taken : ℕ) : ℕ :=
  60 / (passengers_per_hour / (passengers_left + passengers_taken))

theorem train_arrival_interval :
  train_interval 6240 200 320 = 5 := by
  sorry

end NUMINAMATH_CALUDE_train_arrival_interval_l3794_379415


namespace NUMINAMATH_CALUDE_fruit_juice_mixture_proof_l3794_379477

/-- Proves that adding 0.4 liters of pure fruit juice to a 2-liter mixture
    that is 10% pure fruit juice results in a final mixture that is 25% pure fruit juice. -/
theorem fruit_juice_mixture_proof :
  let initial_volume : ℝ := 2
  let initial_concentration : ℝ := 0.1
  let target_concentration : ℝ := 0.25
  let added_juice : ℝ := 0.4
  let final_volume : ℝ := initial_volume + added_juice
  let final_juice_amount : ℝ := initial_volume * initial_concentration + added_juice
  final_juice_amount / final_volume = target_concentration :=
by sorry

end NUMINAMATH_CALUDE_fruit_juice_mixture_proof_l3794_379477


namespace NUMINAMATH_CALUDE_cosine_product_equals_half_to_seventh_power_l3794_379439

theorem cosine_product_equals_half_to_seventh_power :
  (Real.cos (12 * π / 180)) *
  (Real.cos (24 * π / 180)) *
  (Real.cos (36 * π / 180)) *
  (Real.cos (48 * π / 180)) *
  (Real.cos (60 * π / 180)) *
  (Real.cos (72 * π / 180)) *
  (Real.cos (84 * π / 180)) = (1/2)^7 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_equals_half_to_seventh_power_l3794_379439


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3794_379485

theorem imaginary_part_of_complex_fraction (i : ℂ) : 
  i * i = -1 → Complex.im ((5 + i) / (2 - i)) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3794_379485


namespace NUMINAMATH_CALUDE_ellipse_properties_l3794_379436

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/5 + y^2 = 1

-- Define the right focus F
def right_focus : ℝ × ℝ := (2, 0)

-- Define the line l passing through F
def line_l (k : ℝ) (x : ℝ) : ℝ := k*(x - 2)

-- Define points A and B on the ellipse
def point_on_ellipse (x y : ℝ) : Prop := ellipse_C x y ∧ ∃ k, y = line_l k x

-- Define point M on y-axis
def point_M (y : ℝ) : ℝ × ℝ := (0, y)

-- Define vectors MA, MB, FA, and FB
def vector_MA (x y y0 : ℝ) : ℝ × ℝ := (x, y - y0)
def vector_MB (x y y0 : ℝ) : ℝ × ℝ := (x, y - y0)
def vector_FA (x y : ℝ) : ℝ × ℝ := (x - 2, y)
def vector_FB (x y : ℝ) : ℝ × ℝ := (x - 2, y)

theorem ellipse_properties :
  ∀ (x1 y1 x2 y2 y0 m n : ℝ),
  point_on_ellipse x1 y1 →
  point_on_ellipse x2 y2 →
  vector_MA x1 y1 y0 = m • (vector_FA x1 y1) →
  vector_MB x2 y2 y0 = n • (vector_FB x2 y2) →
  m + n = 10 :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3794_379436
