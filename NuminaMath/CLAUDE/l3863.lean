import Mathlib

namespace NUMINAMATH_CALUDE_min_distance_parabola_to_line_l3863_386344

/-- The minimum distance from a point on the parabola y² = 4x to the line x - y + 2 = 0 is √2 / 2. -/
theorem min_distance_parabola_to_line :
  let parabola := {P : ℝ × ℝ | P.2^2 = 4 * P.1}
  let line := {P : ℝ × ℝ | P.1 - P.2 + 2 = 0}
  ∀ M ∈ parabola, ∃ P ∈ line,
    ∀ Q ∈ line, dist M P ≤ dist M Q ∧ dist M P = Real.sqrt 2 / 2 := by
  sorry

#check min_distance_parabola_to_line

end NUMINAMATH_CALUDE_min_distance_parabola_to_line_l3863_386344


namespace NUMINAMATH_CALUDE_rectangle_perimeter_equal_area_l3863_386383

theorem rectangle_perimeter_equal_area (x y : ℕ) : 
  x > 0 ∧ y > 0 → 2 * x + 2 * y = x * y → (x = 3 ∧ y = 6) ∨ (x = 6 ∧ y = 3) ∨ (x = 4 ∧ y = 4) := by
  sorry

#check rectangle_perimeter_equal_area

end NUMINAMATH_CALUDE_rectangle_perimeter_equal_area_l3863_386383


namespace NUMINAMATH_CALUDE_fraction_product_power_l3863_386353

theorem fraction_product_power : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_power_l3863_386353


namespace NUMINAMATH_CALUDE_esports_gender_related_prob_select_male_expected_like_esports_l3863_386320

-- Define the survey data
def total_students : ℕ := 400
def male_like : ℕ := 120
def male_dislike : ℕ := 80
def female_like : ℕ := 100
def female_dislike : ℕ := 100

-- Define the critical value for α = 0.05
def critical_value : ℚ := 3841/1000

-- Define the chi-square statistic function
def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Theorem 1: Chi-square statistic is greater than critical value
theorem esports_gender_related :
  chi_square male_like male_dislike female_like female_dislike > critical_value := by
  sorry

-- Theorem 2: Probability of selecting at least one male student
theorem prob_select_male :
  1 - (Nat.choose 5 3 : ℚ) / (Nat.choose 9 3) = 37/42 := by
  sorry

-- Theorem 3: Expected number of students who like esports
theorem expected_like_esports :
  (10 : ℚ) * (male_like + female_like) / total_students = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_esports_gender_related_prob_select_male_expected_like_esports_l3863_386320


namespace NUMINAMATH_CALUDE_quadratic_root_transform_l3863_386369

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots x₁ and x₂,
    this theorem proves the equations with transformed roots. -/
theorem quadratic_root_transform (a b c : ℝ) (x₁ x₂ : ℝ) 
  (hroot : a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) :
  (∃ y₁ y₂ : ℝ, y₁ = 1/x₁^3 ∧ y₂ = 1/x₂^3 ∧ 
    c^3 * y₁^2 + (b^3 - 3*a*b*c) * y₁ + a^3 = 0 ∧
    c^3 * y₂^2 + (b^3 - 3*a*b*c) * y₂ + a^3 = 0) ∧
  (∃ z₁ z₂ : ℝ, z₁ = (x₁ - x₂)^2 ∧ z₂ = (x₁ + x₂)^2 ∧
    a^4 * z₁^2 + 2*a^2*(2*a*c - b^2) * z₁ + b^2*(b^2 - 4*a*c) = 0 ∧
    a^4 * z₂^2 + 2*a^2*(2*a*c - b^2) * z₂ + b^2*(b^2 - 4*a*c) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transform_l3863_386369


namespace NUMINAMATH_CALUDE_parallel_lines_theorem_l3863_386366

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (a b c d e f : ℝ) : Prop :=
  a / b = d / e

/-- Given two lines l₁: ax + 3y - 3 = 0 and l₂: 4x + 6y - 1 = 0,
    if they are parallel, then a = 2 -/
theorem parallel_lines_theorem (a : ℝ) :
  parallel_lines a 3 (-3) 4 6 (-1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_theorem_l3863_386366


namespace NUMINAMATH_CALUDE_rationalize_and_sum_l3863_386372

theorem rationalize_and_sum (a b c d s f : ℚ) (p q r : ℕ) :
  let x := (1 : ℝ) / (Real.sqrt 5 + Real.sqrt 6 + Real.sqrt 8)
  let y := (a * Real.sqrt p + b * Real.sqrt q + c * Real.sqrt r + d * Real.sqrt s) / f
  (∃ (a b c d s : ℚ) (p q r : ℕ) (f : ℚ), 
    f > 0 ∧ 
    x = y ∧
    (p = 5 ∧ q = 6 ∧ r = 2 ∧ s = 1) ∧
    (a = 9 ∧ b = 7 ∧ c = -18 ∧ d = 0)) →
  a + b + c + d + s + f = 111 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_and_sum_l3863_386372


namespace NUMINAMATH_CALUDE_part_one_part_two_l3863_386387

/-- Given c > 0 and c ≠ 1, define p and q as follows:
    p: The function y = c^x is monotonically decreasing
    q: The function f(x) = x^2 - 2cx + 1 is increasing on the interval (1/2, +∞) -/
def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y
def q (c : ℝ) : Prop := ∀ x y : ℝ, 1/2 < x ∧ x < y → x^2 - 2*c*x + 1 < y^2 - 2*c*y + 1

/-- Part 1: If p is true and ¬q is false, then 0 < c ≤ 1/2 -/
theorem part_one (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1) (h3 : p c) (h4 : ¬¬(q c)) :
  0 < c ∧ c ≤ 1/2 := by sorry

/-- Part 2: If "p AND q" is false and "p OR q" is true, then 1/2 < c < 1 -/
theorem part_two (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1) (h3 : ¬(p c ∧ q c)) (h4 : p c ∨ q c) :
  1/2 < c ∧ c < 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3863_386387


namespace NUMINAMATH_CALUDE_atm_withdrawal_cost_l3863_386328

/-- Represents the cost structure for ATM withdrawals -/
structure ATMCost where
  base_fee : ℝ
  proportional_rate : ℝ

/-- Calculates the total cost for a given withdrawal amount -/
def total_cost (c : ATMCost) (amount : ℝ) : ℝ :=
  c.base_fee + c.proportional_rate * amount

/-- The ATM cost structure satisfies the given conditions -/
def satisfies_conditions (c : ATMCost) : Prop :=
  total_cost c 40000 = 221 ∧ total_cost c 100000 = 485

theorem atm_withdrawal_cost :
  ∃ (c : ATMCost), satisfies_conditions c ∧ total_cost c 85000 = 419 := by
  sorry

end NUMINAMATH_CALUDE_atm_withdrawal_cost_l3863_386328


namespace NUMINAMATH_CALUDE_skittles_distribution_l3863_386399

/-- Given 25 Skittles distributed among 5 people, prove that each person receives 5 Skittles. -/
theorem skittles_distribution (total_skittles : ℕ) (num_people : ℕ) (skittles_per_person : ℕ) :
  total_skittles = 25 →
  num_people = 5 →
  skittles_per_person = total_skittles / num_people →
  skittles_per_person = 5 :=
by sorry

end NUMINAMATH_CALUDE_skittles_distribution_l3863_386399


namespace NUMINAMATH_CALUDE_percentage_of_sikh_boys_l3863_386378

/-- Proves that the percentage of Sikh boys in a school is 10% -/
theorem percentage_of_sikh_boys (total_boys : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (other_boys : ℕ) :
  total_boys = 850 →
  muslim_percent = 34 / 100 →
  hindu_percent = 28 / 100 →
  other_boys = 238 →
  (total_boys - (muslim_percent * total_boys + hindu_percent * total_boys + other_boys : ℚ)) / total_boys * 100 = 10 := by
sorry


end NUMINAMATH_CALUDE_percentage_of_sikh_boys_l3863_386378


namespace NUMINAMATH_CALUDE_f_properties_l3863_386352

/-- The function f(x) = x^3 - ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

/-- Theorem stating the range of a and the fixed point property -/
theorem f_properties (a : ℝ) (x₀ : ℝ) 
  (ha : a > 0)
  (hf : ∀ x y, x ≥ 1 → y ≥ 1 → x < y → f a x < f a y)
  (hx₀ : x₀ ≥ 1)
  (hfx₀ : f a x₀ ≥ 1)
  (hffx₀ : f a (f a x₀) = x₀) :
  (0 < a ∧ a ≤ 3) ∧ f a x₀ = x₀ := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3863_386352


namespace NUMINAMATH_CALUDE_b_sixth_congruence_l3863_386377

theorem b_sixth_congruence (n : ℕ+) (b : ℤ) (h : b^3 ≡ 1 [ZMOD n]) :
  b^6 ≡ 1 [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_b_sixth_congruence_l3863_386377


namespace NUMINAMATH_CALUDE_michaels_matchsticks_l3863_386315

/-- The number of matchsticks Michael had originally -/
def original_matchsticks : ℕ := 1700

/-- The number of houses Michael created -/
def houses : ℕ := 30

/-- The number of towers Michael created -/
def towers : ℕ := 20

/-- The number of bridges Michael created -/
def bridges : ℕ := 10

/-- The number of matchsticks used for each house -/
def matchsticks_per_house : ℕ := 10

/-- The number of matchsticks used for each tower -/
def matchsticks_per_tower : ℕ := 15

/-- The number of matchsticks used for each bridge -/
def matchsticks_per_bridge : ℕ := 25

/-- Theorem stating that Michael's original pile of matchsticks was 1700 -/
theorem michaels_matchsticks :
  original_matchsticks = 2 * (houses * matchsticks_per_house +
                              towers * matchsticks_per_tower +
                              bridges * matchsticks_per_bridge) :=
by sorry

end NUMINAMATH_CALUDE_michaels_matchsticks_l3863_386315


namespace NUMINAMATH_CALUDE_simplify_expression_l3863_386361

theorem simplify_expression (w : ℝ) :
  2 * w + 3 - 4 * w - 5 + 6 * w + 7 - 8 * w - 9 = -4 * w - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3863_386361


namespace NUMINAMATH_CALUDE_panteleimon_twos_count_l3863_386364

/-- Represents the grades of a student -/
structure Grades :=
  (fives : ℕ)
  (fours : ℕ)
  (threes : ℕ)
  (twos : ℕ)

/-- The total number of grades for each student -/
def total_grades : ℕ := 20

/-- Calculates the average grade -/
def average_grade (g : Grades) : ℚ :=
  (5 * g.fives + 4 * g.fours + 3 * g.threes + 2 * g.twos : ℚ) / total_grades

theorem panteleimon_twos_count 
  (p g : Grades) -- Panteleimon's and Gerasim's grades
  (h1 : p.fives + p.fours + p.threes + p.twos = total_grades)
  (h2 : g.fives + g.fours + g.threes + g.twos = total_grades)
  (h3 : p.fives = g.fours)
  (h4 : p.fours = g.threes)
  (h5 : p.threes = g.twos)
  (h6 : p.twos = g.fives)
  (h7 : average_grade p = average_grade g) :
  p.twos = 5 := by
  sorry

end NUMINAMATH_CALUDE_panteleimon_twos_count_l3863_386364


namespace NUMINAMATH_CALUDE_riverside_park_adjustment_plans_l3863_386354

/-- Represents the number of riverside theme parks -/
def total_parks : ℕ := 7

/-- Represents the number of parks to be removed -/
def parks_to_remove : ℕ := 2

/-- Represents the number of parks that can be adjusted (excluding the ends) -/
def adjustable_parks : ℕ := total_parks - 2

/-- Represents the number of adjacent park pairs that cannot be removed together -/
def adjacent_pairs : ℕ := adjustable_parks - 1

theorem riverside_park_adjustment_plans :
  (adjustable_parks.choose parks_to_remove) - adjacent_pairs = 6 := by
  sorry

end NUMINAMATH_CALUDE_riverside_park_adjustment_plans_l3863_386354


namespace NUMINAMATH_CALUDE_power_of_negative_cube_l3863_386347

theorem power_of_negative_cube (x : ℝ) : (-x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_cube_l3863_386347


namespace NUMINAMATH_CALUDE_square_divisibility_l3863_386358

theorem square_divisibility (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : ∃ k : ℕ, a^2 + b^2 = k * (a * b + 1)) : 
  ∃ n : ℕ, (a^2 + b^2) / (a * b + 1) = n^2 := by
sorry

end NUMINAMATH_CALUDE_square_divisibility_l3863_386358


namespace NUMINAMATH_CALUDE_equilateral_triangle_quadratic_ac_l3863_386336

/-- A quadratic function f(x) = ax^2 + c whose graph intersects the coordinate axes 
    at the vertices of an equilateral triangle. -/
structure EquilateralTriangleQuadratic where
  a : ℝ
  c : ℝ
  is_equilateral : ∀ (x y : ℝ), y = a * x^2 + c → 
    (x = 0 ∨ y = 0) → 
    -- The three intersection points form an equilateral triangle
    ∃ (p q r : ℝ × ℝ), 
      (p.1 = 0 ∨ p.2 = 0) ∧ 
      (q.1 = 0 ∨ q.2 = 0) ∧ 
      (r.1 = 0 ∨ r.2 = 0) ∧
      (p.2 = a * p.1^2 + c) ∧
      (q.2 = a * q.1^2 + c) ∧
      (r.2 = a * r.1^2 + c) ∧
      (p.1 - q.1)^2 + (p.2 - q.2)^2 = (q.1 - r.1)^2 + (q.2 - r.2)^2 ∧
      (q.1 - r.1)^2 + (q.2 - r.2)^2 = (r.1 - p.1)^2 + (r.2 - p.2)^2

/-- The product of a and c for an EquilateralTriangleQuadratic is -3. -/
theorem equilateral_triangle_quadratic_ac (f : EquilateralTriangleQuadratic) : 
  f.a * f.c = -3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_quadratic_ac_l3863_386336


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3863_386316

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

-- Define the asymptote of the hyperbola
def asymptote (a b : ℝ) (x y : ℝ) : Prop :=
  y = (b / a) * x ∨ y = -(b / a) * x

-- Define the axis of the parabola
def parabola_axis (p : ℝ) (x : ℝ) : Prop :=
  x = -p / 2

-- Theorem statement
theorem hyperbola_equation (a b p : ℝ) :
  a > 0 ∧ b > 0 ∧ p > 0 ∧
  (∃ x₀ y₀, asymptote a b x₀ y₀ ∧ parabola_axis p x₀ ∧ x₀ = -2 ∧ y₀ = -4) ∧
  (∃ x₁ y₁ x₂ y₂, hyperbola a b x₁ y₁ ∧ x₁ = -a ∧ y₁ = 0 ∧
                  parabola p x₂ y₂ ∧ x₂ = p ∧ y₂ = 0 ∧
                  (x₂ - x₁)^2 + (y₂ - y₁)^2 = 16) →
  a = 2 ∧ b = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3863_386316


namespace NUMINAMATH_CALUDE_continuous_at_two_l3863_386398

/-- The function f(x) = -4x^2 - 8 is continuous at x₀ = 2 -/
theorem continuous_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |(-4*x^2 - 8) - (-4*2^2 - 8)| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuous_at_two_l3863_386398


namespace NUMINAMATH_CALUDE_not_square_n5_plus_7_l3863_386317

theorem not_square_n5_plus_7 (n : ℤ) (h : n > 1) : ¬ ∃ k : ℤ, n^5 + 7 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_square_n5_plus_7_l3863_386317


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l3863_386340

theorem regular_polygon_exterior_angle (n : ℕ) :
  (360 / n : ℝ) = 72 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l3863_386340


namespace NUMINAMATH_CALUDE_exist_special_pair_l3863_386343

theorem exist_special_pair : ∃ (a b : ℕ+), 
  (¬ (7 ∣ (a.val * b.val * (a.val + b.val)))) ∧ 
  ((7^7 : ℕ) ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) ∧ 
  (((a.val = 18 ∧ b.val = 1) ∨ (a.val = 1 ∧ b.val = 18))) :=
by sorry

end NUMINAMATH_CALUDE_exist_special_pair_l3863_386343


namespace NUMINAMATH_CALUDE_crayons_to_mary_l3863_386397

def crayons_given_to_mary (new_pack : ℕ) (locker : ℕ) : ℕ :=
  let initial_total := new_pack + locker
  let from_bobby := locker / 2
  let final_total := initial_total + from_bobby
  final_total / 3

theorem crayons_to_mary :
  crayons_given_to_mary 21 36 = 25 := by
  sorry

end NUMINAMATH_CALUDE_crayons_to_mary_l3863_386397


namespace NUMINAMATH_CALUDE_photocopy_savings_theorem_l3863_386394

/-- Represents the cost structure for photocopies -/
structure CostStructure where
  base_cost : Real
  color_cost : Real
  double_sided_cost : Real
  discount_tier1 : Real
  discount_tier2 : Real
  discount_tier3 : Real

/-- Represents an order of photocopies -/
structure Order where
  bw_one_sided : Nat
  bw_double_sided : Nat
  color_one_sided : Nat
  color_double_sided : Nat

/-- Calculates the cost of an order without discount -/
def orderCost (cs : CostStructure) (o : Order) : Real := sorry

/-- Calculates the discount percentage based on the total number of copies -/
def discountPercentage (cs : CostStructure) (total_copies : Nat) : Real := sorry

/-- Calculates the total cost of combined orders with discount -/
def combinedOrderCost (cs : CostStructure) (o1 o2 : Order) : Real := sorry

/-- Calculates the savings when combining two orders -/
def savings (cs : CostStructure) (o1 o2 : Order) : Real := sorry

theorem photocopy_savings_theorem (cs : CostStructure) (steve_order dennison_order : Order) :
  cs.base_cost = 0.02 ∧
  cs.color_cost = 0.08 ∧
  cs.double_sided_cost = 0.03 ∧
  cs.discount_tier1 = 0.1 ∧
  cs.discount_tier2 = 0.2 ∧
  cs.discount_tier3 = 0.3 ∧
  steve_order.bw_one_sided = 35 ∧
  steve_order.bw_double_sided = 25 ∧
  steve_order.color_one_sided = 0 ∧
  steve_order.color_double_sided = 15 ∧
  dennison_order.bw_one_sided = 20 ∧
  dennison_order.bw_double_sided = 40 ∧
  dennison_order.color_one_sided = 12 ∧
  dennison_order.color_double_sided = 0 →
  savings cs steve_order dennison_order = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_photocopy_savings_theorem_l3863_386394


namespace NUMINAMATH_CALUDE_least_prime_factor_of_8_pow_4_minus_8_pow_3_l3863_386302

theorem least_prime_factor_of_8_pow_4_minus_8_pow_3 :
  Nat.minFac (8^4 - 8^3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_8_pow_4_minus_8_pow_3_l3863_386302


namespace NUMINAMATH_CALUDE_arlo_books_count_l3863_386368

theorem arlo_books_count (total_stationery : ℕ) (book_ratio pen_ratio : ℕ) (h1 : total_stationery = 400) (h2 : book_ratio = 7) (h3 : pen_ratio = 3) : 
  (book_ratio * total_stationery) / (book_ratio + pen_ratio) = 280 := by
sorry

end NUMINAMATH_CALUDE_arlo_books_count_l3863_386368


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_37_l3863_386391

def polynomial (x : ℝ) : ℝ := -3 * (x^8 - 2*x^5 + 4*x^3 - 6) + 5 * (x^4 + 3*x^2) - 2 * (x^6 - 5)

theorem sum_of_coefficients_is_37 : 
  (polynomial 1) = 37 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_37_l3863_386391


namespace NUMINAMATH_CALUDE_abigail_score_l3863_386389

theorem abigail_score (n : ℕ) (initial_avg final_avg : ℚ) (abigail_score : ℚ) :
  n = 20 →
  initial_avg = 85 →
  final_avg = 86 →
  (n : ℚ) * initial_avg + abigail_score = (n + 1 : ℚ) * final_avg →
  abigail_score = 106 :=
by sorry

end NUMINAMATH_CALUDE_abigail_score_l3863_386389


namespace NUMINAMATH_CALUDE_complement_of_A_l3863_386304

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := Set.Ioc (-2) 1

-- State the theorem
theorem complement_of_A : 
  Set.compl A = Set.Iic (-2) ∪ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3863_386304


namespace NUMINAMATH_CALUDE_harry_photo_reorganization_l3863_386375

/-- Represents a photo album organization system -/
structure PhotoAlbumSystem where
  initialAlbums : Nat
  pagesPerAlbum : Nat
  initialPhotosPerPage : Nat
  newPhotosPerPage : Nat
  filledAlbums : Nat

/-- Calculates the number of photos on the last page of the partially filled album -/
def photosOnLastPage (system : PhotoAlbumSystem) : Nat :=
  let totalPhotos := system.initialAlbums * system.pagesPerAlbum * system.initialPhotosPerPage
  let totalPagesNeeded := (totalPhotos + system.newPhotosPerPage - 1) / system.newPhotosPerPage
  let pagesInFilledAlbums := system.filledAlbums * system.pagesPerAlbum
  let remainingPhotos := totalPhotos - pagesInFilledAlbums * system.newPhotosPerPage
  remainingPhotos % system.newPhotosPerPage

theorem harry_photo_reorganization :
  let system : PhotoAlbumSystem := {
    initialAlbums := 10,
    pagesPerAlbum := 35,
    initialPhotosPerPage := 4,
    newPhotosPerPage := 8,
    filledAlbums := 6
  }
  photosOnLastPage system = 0 := by
  sorry

end NUMINAMATH_CALUDE_harry_photo_reorganization_l3863_386375


namespace NUMINAMATH_CALUDE_distance_circle_center_to_line_l3863_386323

/-- Given a circle with polar equation ρ = 4sin(θ) and a line with parametric equation x = √3t, y = t,
    the distance from the center of the circle to the line is √3. -/
theorem distance_circle_center_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 4*y}
  let line := {(x, y) : ℝ × ℝ | ∃ t : ℝ, x = Real.sqrt 3 * t ∧ y = t}
  let circle_center := (0, 2)
  ∃ p ∈ line, Real.sqrt ((circle_center.1 - p.1)^2 + (circle_center.2 - p.2)^2) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_distance_circle_center_to_line_l3863_386323


namespace NUMINAMATH_CALUDE_novelty_shop_costs_l3863_386370

/-- Represents the cost of items in dollars -/
structure ItemCost where
  magazine : ℝ
  chocolate : ℝ
  candy : ℝ
  toy : ℝ

/-- The conditions given in the problem -/
def shopConditions (cost : ItemCost) : Prop :=
  cost.magazine = 1 ∧
  4 * cost.chocolate = 8 * cost.magazine ∧
  2 * cost.candy + 3 * cost.toy = 5 * cost.magazine

/-- The theorem stating the cost of a dozen chocolate bars and the indeterminacy of candy and toy costs -/
theorem novelty_shop_costs (cost : ItemCost) (h : shopConditions cost) :
  12 * cost.chocolate = 24 ∧
  ∃ (c t : ℝ), c ≠ cost.candy ∧ t ≠ cost.toy ∧ shopConditions { magazine := cost.magazine, chocolate := cost.chocolate, candy := c, toy := t } :=
by sorry

end NUMINAMATH_CALUDE_novelty_shop_costs_l3863_386370


namespace NUMINAMATH_CALUDE_problem_solution_l3863_386379

theorem problem_solution (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3863_386379


namespace NUMINAMATH_CALUDE_passengers_off_in_texas_l3863_386312

/-- Represents the number of passengers at different stages of the flight --/
structure FlightPassengers where
  initial : ℕ
  texas_off : ℕ
  texas_on : ℕ
  nc_off : ℕ
  nc_on : ℕ
  final : ℕ

/-- Theorem stating that 48 passengers got off in Texas --/
theorem passengers_off_in_texas (fp : FlightPassengers) 
  (h1 : fp.initial = 124)
  (h2 : fp.texas_on = 24)
  (h3 : fp.nc_off = 47)
  (h4 : fp.nc_on = 14)
  (h5 : fp.final = 67)
  (h6 : fp.initial - fp.texas_off + fp.texas_on - fp.nc_off + fp.nc_on = fp.final) :
  fp.texas_off = 48 := by
  sorry


end NUMINAMATH_CALUDE_passengers_off_in_texas_l3863_386312


namespace NUMINAMATH_CALUDE_shaded_percentage_of_square_l3863_386362

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  sideLength : ℝ
  bottomLeft : Point

/-- Represents a shaded region -/
structure ShadedRegion where
  bottomLeft : Point
  topRight : Point

/-- Calculate the area of a square -/
def squareArea (s : Square) : ℝ := s.sideLength * s.sideLength

/-- Calculate the area of a shaded region -/
def shadedRegionArea (r : ShadedRegion) : ℝ :=
  (r.topRight.x - r.bottomLeft.x) * (r.topRight.y - r.bottomLeft.y)

/-- The main theorem -/
theorem shaded_percentage_of_square (EFGH : Square)
  (region1 region2 region3 : ShadedRegion) :
  EFGH.sideLength = 7 →
  EFGH.bottomLeft = ⟨0, 0⟩ →
  region1 = ⟨⟨0, 0⟩, ⟨1, 1⟩⟩ →
  region2 = ⟨⟨3, 0⟩, ⟨5, 5⟩⟩ →
  region3 = ⟨⟨6, 0⟩, ⟨7, 7⟩⟩ →
  (shadedRegionArea region1 + shadedRegionArea region2 + shadedRegionArea region3) /
    squareArea EFGH * 100 = 14 / 49 * 100 := by
  sorry

end NUMINAMATH_CALUDE_shaded_percentage_of_square_l3863_386362


namespace NUMINAMATH_CALUDE_polynomial_identity_coefficients_l3863_386322

theorem polynomial_identity_coefficients 
  (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ x : ℝ, x^5 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + a₄*(x+2)^4 + a₅*(x+2)^5) : 
  a₃ = 40 ∧ a₀ + a₁ + a₂ + a₄ + a₅ = -41 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_coefficients_l3863_386322


namespace NUMINAMATH_CALUDE_toy_car_cost_price_l3863_386357

/-- The cost price of a toy car given specific pricing conditions --/
theorem toy_car_cost_price :
  ∀ (cost_price : ℝ),
  let initial_price := 2 * cost_price
  let second_day_price := 0.9 * initial_price
  let final_price := second_day_price - 360
  (final_price = 1.44 * cost_price) →
  cost_price = 1000 := by
sorry

end NUMINAMATH_CALUDE_toy_car_cost_price_l3863_386357


namespace NUMINAMATH_CALUDE_power_seven_eight_mod_hundred_l3863_386324

theorem power_seven_eight_mod_hundred : 7^8 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_eight_mod_hundred_l3863_386324


namespace NUMINAMATH_CALUDE_chairs_in_clubroom_l3863_386314

/-- Represents the number of chairs in the clubroom -/
def num_chairs : ℕ := 17

/-- Represents the number of legs each chair has -/
def legs_per_chair : ℕ := 4

/-- Represents the number of legs the table has -/
def table_legs : ℕ := 3

/-- Represents the number of unoccupied chairs -/
def unoccupied_chairs : ℕ := 2

/-- Represents the total number of legs in the room -/
def total_legs : ℕ := 101

/-- Proves that the number of chairs in the clubroom is correct given the conditions -/
theorem chairs_in_clubroom :
  num_chairs * legs_per_chair + table_legs = total_legs + 2 * (num_chairs - unoccupied_chairs) :=
by sorry

end NUMINAMATH_CALUDE_chairs_in_clubroom_l3863_386314


namespace NUMINAMATH_CALUDE_polynomial_division_l3863_386311

theorem polynomial_division (z : ℝ) :
  6 * z^5 - 5 * z^4 + 2 * z^3 - 8 * z^2 + 7 * z - 3 = 
  (z^2 - 1) * (6 * z^3 + z^2 + 3 * z) + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l3863_386311


namespace NUMINAMATH_CALUDE_inverse_function_point_l3863_386363

-- Define a function f and its inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- Define the property of f_inv being the inverse of f
def is_inverse (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- State the theorem
theorem inverse_function_point 
  (h1 : is_inverse f f_inv) 
  (h2 : f 3 = -1) : 
  f_inv (-3) = 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_point_l3863_386363


namespace NUMINAMATH_CALUDE_sum_abs_difference_l3863_386392

theorem sum_abs_difference : ∀ (a b : ℤ), a = -5 ∧ b = -4 → abs a + abs b - (a + b) = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_abs_difference_l3863_386392


namespace NUMINAMATH_CALUDE_math_books_count_l3863_386360

theorem math_books_count (total_books : ℕ) (math_book_price history_book_price : ℕ) (total_price : ℕ) :
  total_books = 250 →
  math_book_price = 7 →
  history_book_price = 9 →
  total_price = 1860 →
  ∃ (math_books history_books : ℕ),
    math_books + history_books = total_books ∧
    math_book_price * math_books + history_book_price * history_books = total_price ∧
    math_books = 195 :=
by sorry

end NUMINAMATH_CALUDE_math_books_count_l3863_386360


namespace NUMINAMATH_CALUDE_ant_climb_floors_l3863_386386

-- Define the problem parameters
def time_per_floor : ℕ := 15
def total_time : ℕ := 105
def start_floor : ℕ := 1

-- State the theorem
theorem ant_climb_floors :
  ∃ (final_floor : ℕ),
    final_floor = (total_time / time_per_floor) + start_floor ∧
    final_floor = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ant_climb_floors_l3863_386386


namespace NUMINAMATH_CALUDE_cylinder_views_l3863_386390

-- Define the cylinder and its orientation
structure Cylinder where
  upright : Bool
  on_horizontal_plane : Bool

-- Define the possible view shapes
inductive ViewShape
  | Rectangle
  | Circle

-- Define the function to get the view of the cylinder
def get_cylinder_view (c : Cylinder) (view : String) : ViewShape :=
  match view with
  | "front" => ViewShape.Rectangle
  | "side" => ViewShape.Rectangle
  | "top" => ViewShape.Circle
  | _ => ViewShape.Rectangle  -- Default case, though not needed for our problem

-- Theorem statement
theorem cylinder_views (c : Cylinder) 
  (h1 : c.upright = true) 
  (h2 : c.on_horizontal_plane = true) : 
  (get_cylinder_view c "front" = ViewShape.Rectangle) ∧ 
  (get_cylinder_view c "side" = ViewShape.Rectangle) ∧ 
  (get_cylinder_view c "top" = ViewShape.Circle) := by
  sorry


end NUMINAMATH_CALUDE_cylinder_views_l3863_386390


namespace NUMINAMATH_CALUDE_square_difference_equals_product_l3863_386382

theorem square_difference_equals_product (x y : ℚ) 
  (sum_eq : x + y = 8/15) 
  (diff_eq : x - y = 2/15) : 
  x^2 - y^2 = 16/225 := by
sorry

end NUMINAMATH_CALUDE_square_difference_equals_product_l3863_386382


namespace NUMINAMATH_CALUDE_new_plan_cost_theorem_l3863_386367

def old_phone_plan_cost : ℝ := 150
def old_internet_cost : ℝ := 50
def old_calling_cost : ℝ := 30
def old_streaming_cost : ℝ := 40

def new_phone_plan_increase : ℝ := 0.30
def new_internet_increase : ℝ := 0.20
def new_calling_discount : ℝ := 0.15
def new_streaming_increase : ℝ := 0.25
def promotional_discount : ℝ := 0.10

def new_phone_plan_cost : ℝ := old_phone_plan_cost * (1 + new_phone_plan_increase)
def new_internet_cost : ℝ := old_internet_cost * (1 + new_internet_increase)
def new_calling_cost : ℝ := old_calling_cost * (1 - new_calling_discount)
def new_streaming_cost : ℝ := old_streaming_cost * (1 + new_streaming_increase)

def total_cost_before_discount : ℝ := 
  new_phone_plan_cost + new_internet_cost + new_calling_cost + new_streaming_cost

def total_cost_after_discount : ℝ := 
  total_cost_before_discount * (1 - promotional_discount)

theorem new_plan_cost_theorem : 
  total_cost_after_discount = 297.45 := by sorry

end NUMINAMATH_CALUDE_new_plan_cost_theorem_l3863_386367


namespace NUMINAMATH_CALUDE_x_minus_p_equals_two_l3863_386307

theorem x_minus_p_equals_two (x p : ℝ) (h1 : |x - 2| = p) (h2 : x > 2) : x - p = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_p_equals_two_l3863_386307


namespace NUMINAMATH_CALUDE_circular_arrangement_students_l3863_386308

/-- Given a circular arrangement of students, if the 10th and 45th positions
    are opposite each other, then the total number of students is 70. -/
theorem circular_arrangement_students (n : ℕ) : n = 70 :=
  by
  -- Assume the 10th and 45th positions are opposite each other
  have h1 : 45 - 10 = n / 2 := by sorry
  
  -- The total number of students is twice the difference between opposite positions
  have h2 : n = 2 * (45 - 10) := by sorry
  
  -- Prove that n equals 70
  sorry

end NUMINAMATH_CALUDE_circular_arrangement_students_l3863_386308


namespace NUMINAMATH_CALUDE_proposition_p_and_q_l3863_386376

def is_ellipse (m : ℝ) : Prop :=
  1 < m ∧ m < 3 ∧ m ≠ 2

def no_common_points (m : ℝ) : Prop :=
  m > Real.sqrt 5 / 2 ∨ m < -Real.sqrt 5 / 2

theorem proposition_p_and_q (m : ℝ) :
  (is_ellipse m ∧ no_common_points m) ↔ 
  (Real.sqrt 5 / 2 < m ∧ m < 2) ∨ (2 < m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_proposition_p_and_q_l3863_386376


namespace NUMINAMATH_CALUDE_fifth_girl_siblings_l3863_386365

def number_set : List ℕ := [1, 6, 10, 4, 3, 11, 3, 10]

theorem fifth_girl_siblings (mean : ℚ) (h1 : mean = 57/10) 
  (h2 : (number_set.sum + x) / 9 = mean) : x = 3 :=
sorry

end NUMINAMATH_CALUDE_fifth_girl_siblings_l3863_386365


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l3863_386395

theorem complex_fraction_equals_i : 
  let i : ℂ := Complex.I
  (1 + i^2017) / (1 - i) = i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l3863_386395


namespace NUMINAMATH_CALUDE_height_prediction_age_10_l3863_386326

/-- Regression model for height prediction -/
def height_model (age : ℝ) : ℝ := 7.19 * age + 73.93

/-- The predicted height at age 10 is approximately 145.83 cm -/
theorem height_prediction_age_10 :
  ∃ ε > 0, abs (height_model 10 - 145.83) < ε :=
sorry

end NUMINAMATH_CALUDE_height_prediction_age_10_l3863_386326


namespace NUMINAMATH_CALUDE_triangle_formation_l3863_386309

theorem triangle_formation (a b c : ℝ) : 
  a = 4 ∧ b = 9 ∧ c = 9 →
  a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l3863_386309


namespace NUMINAMATH_CALUDE_wreath_problem_l3863_386374

/-- Represents the number of flowers in a wreath -/
structure Wreath where
  dandelions : ℕ
  cornflowers : ℕ
  daisies : ℕ

/-- The problem statement -/
theorem wreath_problem (masha katya : Wreath) : 
  (masha.dandelions + masha.cornflowers + masha.daisies + 
   katya.dandelions + katya.cornflowers + katya.daisies = 70) →
  (masha.dandelions = (5 * (masha.dandelions + masha.cornflowers + masha.daisies)) / 9) →
  (katya.daisies = (7 * (katya.dandelions + katya.cornflowers + katya.daisies)) / 17) →
  (masha.dandelions = katya.dandelions) →
  (masha.daisies = katya.daisies) →
  (masha.cornflowers = 2 ∧ katya.cornflowers = 0) :=
by sorry

end NUMINAMATH_CALUDE_wreath_problem_l3863_386374


namespace NUMINAMATH_CALUDE_calculation_result_l3863_386342

theorem calculation_result : 2002 * 20032003 - 2003 * 20022002 = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l3863_386342


namespace NUMINAMATH_CALUDE_total_coughs_after_20_minutes_l3863_386329

/-- The number of coughs per minute for Georgia -/
def georgia_coughs_per_minute : ℕ := 5

/-- The number of minutes -/
def time_period : ℕ := 20

/-- Robert coughs twice as much as Georgia -/
def robert_coughs_per_minute : ℕ := 2 * georgia_coughs_per_minute

/-- The total number of coughs by both Georgia and Robert -/
def total_coughs : ℕ := 
  georgia_coughs_per_minute * time_period + robert_coughs_per_minute * time_period

theorem total_coughs_after_20_minutes : total_coughs = 300 := by
  sorry

end NUMINAMATH_CALUDE_total_coughs_after_20_minutes_l3863_386329


namespace NUMINAMATH_CALUDE_value_of_fraction_l3863_386356

-- Define the real numbers
variable (a₁ a₂ b₁ b₂ : ℝ)

-- Define the arithmetic sequence condition
def is_arithmetic_sequence : Prop :=
  ∃ d : ℝ, a₁ - (-1) = d ∧ a₂ - a₁ = d ∧ (-4) - a₂ = d

-- Define the geometric sequence condition
def is_geometric_sequence : Prop :=
  ∃ r : ℝ, b₁ / (-1) = r ∧ b₂ / b₁ = r ∧ (-8) / b₂ = r

-- Theorem statement
theorem value_of_fraction (h1 : is_arithmetic_sequence a₁ a₂)
                          (h2 : is_geometric_sequence b₁ b₂) :
  (a₂ - a₁) / b₂ = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_fraction_l3863_386356


namespace NUMINAMATH_CALUDE_factorization_m_squared_minus_3m_l3863_386345

theorem factorization_m_squared_minus_3m (m : ℝ) : m^2 - 3*m = m*(m-3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_m_squared_minus_3m_l3863_386345


namespace NUMINAMATH_CALUDE_stewart_farm_horse_food_l3863_386350

/-- Given a farm with sheep and horses, calculate the daily horse food requirement per horse -/
theorem stewart_farm_horse_food (sheep_count : ℕ) (total_horse_food : ℕ) 
  (h_sheep_count : sheep_count = 32) 
  (h_total_horse_food : total_horse_food = 12880) 
  (h_ratio : sheep_count * 7 = 32 * 4) : 
  total_horse_food / (sheep_count * 7 / 4) = 230 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_horse_food_l3863_386350


namespace NUMINAMATH_CALUDE_peri_arrival_day_l3863_386319

def travel_pattern (day : ℕ) : ℕ :=
  if day % 10 = 0 then 0 else 1

def total_distance (n : ℕ) : ℕ :=
  (List.range n).map travel_pattern |> List.sum

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed - 1) % 7 + 1

theorem peri_arrival_day :
  ∃ (n : ℕ), total_distance n = 90 ∧ day_of_week 1 n = 2 :=
sorry

end NUMINAMATH_CALUDE_peri_arrival_day_l3863_386319


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3863_386381

theorem complex_equation_solution (x y : ℕ+) 
  (h : (x - Complex.I * y) ^ 2 = 15 - 20 * Complex.I) : 
  x - Complex.I * y = 5 - 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3863_386381


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3863_386337

theorem min_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 1) : 1/x + 1/y + 1/z ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3863_386337


namespace NUMINAMATH_CALUDE_square_difference_l3863_386334

theorem square_difference (x a : ℝ) : (2*x + a)^2 - (2*x - a)^2 = 8*a*x := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3863_386334


namespace NUMINAMATH_CALUDE_min_speed_to_arrive_first_l3863_386333

/-- Proves the minimum speed required for the second person to arrive first -/
theorem min_speed_to_arrive_first (distance : ℝ) (speed_A : ℝ) (head_start : ℝ) 
  (h1 : distance = 180)
  (h2 : speed_A = 40)
  (h3 : head_start = 0.5)
  (h4 : speed_A > 0) : 
  ∃ (min_speed : ℝ), min_speed > 45 ∧ 
    ∀ (speed_B : ℝ), speed_B > min_speed → 
      distance / speed_B < distance / speed_A - head_start := by
sorry

end NUMINAMATH_CALUDE_min_speed_to_arrive_first_l3863_386333


namespace NUMINAMATH_CALUDE_easter_egg_hunt_l3863_386396

theorem easter_egg_hunt (total_eggs : ℕ) 
  (hannah_ratio : ℕ) (harry_extra : ℕ) : 
  total_eggs = 63 ∧ hannah_ratio = 2 ∧ harry_extra = 3 →
  ∃ (helen_eggs hannah_eggs harry_eggs : ℕ),
    helen_eggs = 12 ∧
    hannah_eggs = 24 ∧
    harry_eggs = 27 ∧
    hannah_eggs = hannah_ratio * helen_eggs ∧
    harry_eggs = hannah_eggs + harry_extra ∧
    helen_eggs + hannah_eggs + harry_eggs = total_eggs :=
by sorry

end NUMINAMATH_CALUDE_easter_egg_hunt_l3863_386396


namespace NUMINAMATH_CALUDE_min_nSn_l3863_386349

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  sum_def : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2
  S_10 : S 10 = 0
  S_15 : S 15 = 25

/-- The main theorem -/
theorem min_nSn (seq : ArithmeticSequence) : 
  (∃ n : ℕ, n > 0 ∧ n * seq.S n = -49) ∧ 
  (∀ m : ℕ, m > 0 → m * seq.S m ≥ -49) := by
  sorry

end NUMINAMATH_CALUDE_min_nSn_l3863_386349


namespace NUMINAMATH_CALUDE_combinations_equal_fifteen_l3863_386310

/-- The number of window treatment types available. -/
def num_treatments : ℕ := 3

/-- The number of colors available. -/
def num_colors : ℕ := 5

/-- The total number of combinations of window treatment type and color. -/
def total_combinations : ℕ := num_treatments * num_colors

/-- Theorem stating that the total number of combinations is 15. -/
theorem combinations_equal_fifteen : total_combinations = 15 := by
  sorry

end NUMINAMATH_CALUDE_combinations_equal_fifteen_l3863_386310


namespace NUMINAMATH_CALUDE_hyperbola_condition_l3863_386373

theorem hyperbola_condition (k : ℝ) : 
  (∃ x y : ℝ, x^2 / (2 - k) + y^2 / (k - 1) = 1) ∧ 
  (∀ x y : ℝ, x^2 / (2 - k) + y^2 / (k - 1) = 1 → 
    ∃ a b : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ∨ (y^2 / a^2 - x^2 / b^2 = 1)) →
  k < 1 ∨ k > 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l3863_386373


namespace NUMINAMATH_CALUDE_circle_area_difference_l3863_386338

theorem circle_area_difference : 
  let r1 : ℝ := 30
  let d2 : ℝ := 30
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 675 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l3863_386338


namespace NUMINAMATH_CALUDE_lemonade_pitcher_capacity_l3863_386348

/-- Given that 30 glasses of lemonade were served from 6 pitchers, 
    prove that each pitcher can serve 5 glasses. -/
theorem lemonade_pitcher_capacity 
  (total_glasses : ℕ) 
  (total_pitchers : ℕ) 
  (h1 : total_glasses = 30) 
  (h2 : total_pitchers = 6) : 
  total_glasses / total_pitchers = 5 := by
sorry

end NUMINAMATH_CALUDE_lemonade_pitcher_capacity_l3863_386348


namespace NUMINAMATH_CALUDE_multiplication_increase_l3863_386384

theorem multiplication_increase (n : ℕ) (x : ℚ) (h : n = 25) :
  n * x = n + 375 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_increase_l3863_386384


namespace NUMINAMATH_CALUDE_kody_age_proof_l3863_386331

/-- Kody's current age -/
def kody_age : ℕ := 32

/-- Mohamed's current age -/
def mohamed_age : ℕ := 60

/-- The time difference between now and the past reference point -/
def years_passed : ℕ := 4

theorem kody_age_proof :
  (∃ (kody_past mohamed_past : ℕ),
    kody_past = mohamed_past / 2 ∧
    kody_past + years_passed = kody_age ∧
    mohamed_past + years_passed = mohamed_age) ∧
  mohamed_age = 2 * 30 →
  kody_age = 32 := by sorry

end NUMINAMATH_CALUDE_kody_age_proof_l3863_386331


namespace NUMINAMATH_CALUDE_division_simplification_l3863_386330

theorem division_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (18 * a^2 * b - 9 * a^5 * b^2) / (-3 * a * b) = -6 * a + 3 * a^4 * b := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l3863_386330


namespace NUMINAMATH_CALUDE_field_laps_l3863_386325

theorem field_laps (length width distance : ℝ) : 
  length = 75 →
  width = 15 →
  distance = 540 →
  distance / (2 * (length + width)) = 3 := by
sorry

end NUMINAMATH_CALUDE_field_laps_l3863_386325


namespace NUMINAMATH_CALUDE_product_minus_third_lower_bound_l3863_386303

theorem product_minus_third_lower_bound 
  (x y z : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0) 
  (a : ℝ) 
  (h1 : x * y - z = a) 
  (h2 : y * z - x = a) 
  (h3 : z * x - y = a) : 
  a ≥ -1/4 := by
sorry

end NUMINAMATH_CALUDE_product_minus_third_lower_bound_l3863_386303


namespace NUMINAMATH_CALUDE_discriminant_nonnegative_m_value_when_root_difference_is_two_l3863_386385

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := x^2 - 4*m*x + 3*m^2

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (-4*m)^2 - 4*1*(3*m^2)

-- Theorem 1: The discriminant is always non-negative
theorem discriminant_nonnegative (m : ℝ) : discriminant m ≥ 0 := by
  sorry

-- Theorem 2: When m > 0 and the difference between roots is 2, m = 1
theorem m_value_when_root_difference_is_two (m : ℝ) 
  (h1 : m > 0) 
  (h2 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
                     quadratic_equation m x1 = 0 ∧ 
                     quadratic_equation m x2 = 0 ∧ 
                     x1 - x2 = 2) : 
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_nonnegative_m_value_when_root_difference_is_two_l3863_386385


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l3863_386351

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (5 * a^3 + 2003 * a + 3005 = 0) →
  (5 * b^3 + 2003 * b + 3005 = 0) →
  (5 * c^3 + 2003 * c + 3005 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l3863_386351


namespace NUMINAMATH_CALUDE_parabola_properties_l3863_386305

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Define the conditions
theorem parabola_properties :
  (parabola (-1) = 0) ∧
  (parabola 3 = 0) ∧
  (parabola 0 = -3) ∧
  (∃ (a b c : ℝ), ∀ x, parabola x = a * x^2 + b * x + c) ∧
  (let vertex := (1, -4);
   parabola vertex.1 = vertex.2 ∧
   ∀ x, parabola x ≥ parabola vertex.1) ∧
  (∀ x₁ x₂ y₁ y₂, 
    x₁ < x₂ → x₂ < 1 → 
    parabola x₁ = y₁ → parabola x₂ = y₂ → 
    y₁ < y₂) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l3863_386305


namespace NUMINAMATH_CALUDE_dogs_with_spots_l3863_386355

theorem dogs_with_spots (total_dogs : ℚ) (pointy_ears : ℚ) : ℚ :=
  by
  have h1 : pointy_ears = total_dogs / 5 := by sorry
  have h2 : total_dogs = pointy_ears * 5 := by sorry
  have h3 : total_dogs / 2 = (pointy_ears * 5) / 2 := by sorry
  exact (pointy_ears * 5) / 2

#check dogs_with_spots

end NUMINAMATH_CALUDE_dogs_with_spots_l3863_386355


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_minus_b_l3863_386388

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

-- State the theorem
theorem extreme_value_implies_a_minus_b (a b : ℝ) :
  (f a b (-1) = 0) ∧ 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a b x ≥ f a b (-1)) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a b x ≤ f a b (-1)) →
  a - b = -7 := by
  sorry


end NUMINAMATH_CALUDE_extreme_value_implies_a_minus_b_l3863_386388


namespace NUMINAMATH_CALUDE_rectangle_area_l3863_386393

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) :
  square_area = 1225 →
  rectangle_breadth = 10 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := circle_radius / 4
  let rectangle_area := rectangle_length * rectangle_breadth
  rectangle_area = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3863_386393


namespace NUMINAMATH_CALUDE_range_of_m_l3863_386300

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 1 > 0) →
  ((m + 1 ≤ 0) ∧ (∀ x : ℝ, x^2 + m*x + 1 > 0) = False) →
  ((m + 1 ≤ 0) ∨ (∀ x : ℝ, x^2 + m*x + 1 > 0) = True) →
  (m ≤ -2 ∨ (-1 < m ∧ m < 2)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3863_386300


namespace NUMINAMATH_CALUDE_cos_75_degrees_l3863_386318

theorem cos_75_degrees : Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_degrees_l3863_386318


namespace NUMINAMATH_CALUDE_provider_combinations_l3863_386301

def total_providers : ℕ := 25
def s_providers : ℕ := 6

theorem provider_combinations : 
  total_providers * s_providers * (total_providers - 2) * (total_providers - 3) = 75900 := by
  sorry

end NUMINAMATH_CALUDE_provider_combinations_l3863_386301


namespace NUMINAMATH_CALUDE_tim_income_percentage_l3863_386321

/-- Proves that Tim's income is 60% less than Juan's income given the conditions --/
theorem tim_income_percentage (tim mart juan : ℝ) 
  (h1 : mart = 1.6 * tim)  -- Mart's income is 60% more than Tim's
  (h2 : mart = 0.64 * juan)  -- Mart's income is 64% of Juan's
  : tim = 0.4 * juan :=  -- Tim's income is 40% of Juan's (equivalent to 60% less)
by
  sorry

#check tim_income_percentage

end NUMINAMATH_CALUDE_tim_income_percentage_l3863_386321


namespace NUMINAMATH_CALUDE_not_divisible_by_1000_power_minus_1_l3863_386339

theorem not_divisible_by_1000_power_minus_1 (m : ℕ) : ¬(1000^m - 1 ∣ 1978^m - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_1000_power_minus_1_l3863_386339


namespace NUMINAMATH_CALUDE_used_cd_cost_correct_l3863_386306

/-- The cost of Lakota's purchase -/
def lakota_cost : ℝ := 127.92

/-- The cost of Mackenzie's purchase -/
def mackenzie_cost : ℝ := 133.89

/-- The number of new CDs Lakota bought -/
def lakota_new : ℕ := 6

/-- The number of used CDs Lakota bought -/
def lakota_used : ℕ := 2

/-- The number of new CDs Mackenzie bought -/
def mackenzie_new : ℕ := 3

/-- The number of used CDs Mackenzie bought -/
def mackenzie_used : ℕ := 8

/-- The cost of a single used CD -/
def used_cd_cost : ℝ := 9.99

theorem used_cd_cost_correct :
  ∃ (new_cd_cost : ℝ),
    lakota_new * new_cd_cost + lakota_used * used_cd_cost = lakota_cost ∧
    mackenzie_new * new_cd_cost + mackenzie_used * used_cd_cost = mackenzie_cost :=
by sorry

end NUMINAMATH_CALUDE_used_cd_cost_correct_l3863_386306


namespace NUMINAMATH_CALUDE_S_n_perfect_square_iff_T_n_perfect_square_iff_l3863_386359

/-- Definition of S_n -/
def S_n (n : ℕ) : ℕ := n * (4 * n + 5)

/-- Definition of T_n -/
def T_n (n : ℕ) : ℕ := n * (3 * n + 2)

/-- Definition of is_perfect_square -/
def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, x = k^2

/-- Pell's equation solution -/
def is_pell_solution (l m : ℕ) : Prop := l^2 - 3 * m^2 = 1

/-- Theorem for S_n -/
theorem S_n_perfect_square_iff (n : ℕ) : 
  is_perfect_square (S_n n) ↔ n = 1 :=
sorry

/-- Theorem for T_n -/
theorem T_n_perfect_square_iff (n : ℕ) : 
  is_perfect_square (T_n n) ↔ ∃ m : ℕ, n = 2 * m^2 ∧ ∃ l : ℕ, is_pell_solution l m :=
sorry

end NUMINAMATH_CALUDE_S_n_perfect_square_iff_T_n_perfect_square_iff_l3863_386359


namespace NUMINAMATH_CALUDE_range_of_a_l3863_386341

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2*a*x - 1 + 3*a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (f a 0 < f a 1) →
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ f a x = 0) →
  (1/7 < a ∧ a < 1/5) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3863_386341


namespace NUMINAMATH_CALUDE_polygon_angle_sum_l3863_386380

theorem polygon_angle_sum (n : ℕ) : (n - 2) * 180 = 2 * 360 ↔ n = 6 := by sorry

end NUMINAMATH_CALUDE_polygon_angle_sum_l3863_386380


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_h_l3863_386335

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := 2 * x^2 + 5 * x + 3

/-- The scaled quadratic function -/
def g (x : ℝ) : ℝ := 4 * f x

/-- The vertex form of a quadratic function -/
def vertex_form (m h p : ℝ) (x : ℝ) : ℝ := m * (x - h)^2 + p

theorem quadratic_vertex_form_h :
  ∃ (m p : ℝ), ∀ x, g x = vertex_form m (-5/4) p x :=
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_h_l3863_386335


namespace NUMINAMATH_CALUDE_train_distance_theorem_l3863_386313

/-- The distance between two trains traveling in opposite directions -/
def distance_between_trains (speed_a speed_b : ℝ) (time_a time_b : ℝ) : ℝ :=
  speed_a * time_a + speed_b * time_b

/-- Theorem: The distance between the trains is 1284 miles -/
theorem train_distance_theorem :
  distance_between_trains 56 23 18 12 = 1284 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_theorem_l3863_386313


namespace NUMINAMATH_CALUDE_general_position_lines_regions_l3863_386346

/-- 
A configuration of lines in general position.
-/
structure GeneralPositionLines where
  n : ℕ
  no_parallel : True  -- Represents the condition that no two lines are parallel
  no_concurrent : True -- Represents the condition that no three lines are concurrent

/-- 
The number of regions created by n lines in general position.
-/
def num_regions (lines : GeneralPositionLines) : ℕ :=
  1 + (lines.n * (lines.n + 1)) / 2

/-- 
Theorem: n lines in general position divide a plane into 1 + (1/2) * n * (n + 1) regions.
-/
theorem general_position_lines_regions (lines : GeneralPositionLines) :
  num_regions lines = 1 + (lines.n * (lines.n + 1)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_general_position_lines_regions_l3863_386346


namespace NUMINAMATH_CALUDE_sport_water_amount_l3863_386332

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation ratio -/
def sport_ratio : DrinkRatio :=
  { flavoring := 1,
    corn_syrup := 4 * standard_ratio.flavoring / standard_ratio.corn_syrup,
    water := 2 * standard_ratio.water / standard_ratio.flavoring }

/-- Amount of corn syrup in the sport formulation bottle (in ounces) -/
def sport_corn_syrup : ℚ := 7

/-- Theorem stating the amount of water in the sport formulation bottle -/
theorem sport_water_amount :
  (sport_ratio.water / sport_ratio.corn_syrup) * sport_corn_syrup = 105 := by
  sorry

end NUMINAMATH_CALUDE_sport_water_amount_l3863_386332


namespace NUMINAMATH_CALUDE_student_number_problem_l3863_386371

theorem student_number_problem (x : ℤ) : (8 * x - 138 = 102) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l3863_386371


namespace NUMINAMATH_CALUDE_cylinder_dimensions_l3863_386327

/-- Represents a cylinder formed by rotating a rectangle around one of its sides. -/
structure Cylinder where
  height : ℝ
  radius : ℝ

/-- Theorem: Given a cylinder formed by rotating a rectangle with a diagonal of 26 cm
    around one of its sides, if a perpendicular plane equidistant from the bases has
    a total surface area of 2720 cm², then the height of the cylinder is 24 cm and
    its base radius is 10 cm. -/
theorem cylinder_dimensions (c : Cylinder) :
  c.height ^ 2 + c.radius ^ 2 = 26 ^ 2 →
  8 * c.radius ^ 2 + 8 * c.radius * c.height = 2720 →
  c.height = 24 ∧ c.radius = 10 := by
  sorry

#check cylinder_dimensions

end NUMINAMATH_CALUDE_cylinder_dimensions_l3863_386327
