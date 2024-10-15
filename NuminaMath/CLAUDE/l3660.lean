import Mathlib

namespace NUMINAMATH_CALUDE_no_three_digit_special_couples_l3660_366067

/-- Definition of a special couple for three-digit numbers -/
def is_special_couple (abc cba : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧  -- Digits are single-digit natural numbers
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧     -- Digits are distinct
    abc = 100 * a + 10 * b + c ∧
    cba = 100 * c + 10 * b + a ∧
    a + b + c = 9               -- Sum of digits is 9

/-- Theorem: There are no special couples with three-digit numbers -/
theorem no_three_digit_special_couples :
  ¬ ∃ (abc cba : ℕ), is_special_couple abc cba :=
sorry

end NUMINAMATH_CALUDE_no_three_digit_special_couples_l3660_366067


namespace NUMINAMATH_CALUDE_base_b_is_four_l3660_366051

theorem base_b_is_four : 
  ∃ (b : ℕ), 
    b > 0 ∧ 
    (b - 1) * (b - 1) * b = 72 ∧ 
    b = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_b_is_four_l3660_366051


namespace NUMINAMATH_CALUDE_inequality_proof_l3660_366023

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) :
  (a*f - c*d)^2 ≤ (a*e - b*d)^2 + (b*f - c*e)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3660_366023


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l3660_366088

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 3/b = 1) :
  ∃ (min : ℝ), (∀ x y, x > 0 → y > 0 → 1/x + 3/y = 1 → x + 2*y ≥ min) ∧ (a + 2*b = min) :=
by
  -- The minimum value is 7 + 2√6
  let min := 7 + 2 * Real.sqrt 6
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l3660_366088


namespace NUMINAMATH_CALUDE_upgraded_fraction_is_one_ninth_l3660_366028

/-- Represents a satellite with modular units and sensors. -/
structure Satellite :=
  (units : ℕ)
  (non_upgraded_per_unit : ℕ)
  (total_upgraded : ℕ)

/-- The fraction of upgraded sensors on the satellite. -/
def upgraded_fraction (s : Satellite) : ℚ :=
  s.total_upgraded / (s.units * s.non_upgraded_per_unit + s.total_upgraded)

/-- Theorem stating the fraction of upgraded sensors on a specific satellite configuration. -/
theorem upgraded_fraction_is_one_ninth (s : Satellite) 
    (h1 : s.units = 24)
    (h2 : s.non_upgraded_per_unit = s.total_upgraded / 3) : 
    upgraded_fraction s = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_upgraded_fraction_is_one_ninth_l3660_366028


namespace NUMINAMATH_CALUDE_no_matrix_satisfies_condition_l3660_366009

theorem no_matrix_satisfies_condition : 
  ∀ (N : Matrix (Fin 2) (Fin 2) ℝ),
    (∀ (w x y z : ℝ), 
      N * !![w, x; y, z] = !![x, w; z, y]) → 
    N = 0 := by sorry

end NUMINAMATH_CALUDE_no_matrix_satisfies_condition_l3660_366009


namespace NUMINAMATH_CALUDE_james_cd_purchase_total_l3660_366013

def cd_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

def total_price (prices : List ℝ) (discount_rate : ℝ) : ℝ :=
  (prices.map (λ p => cd_price p discount_rate)).sum

theorem james_cd_purchase_total :
  let prices : List ℝ := [10, 10, 15, 6, 18]
  let discount_rate : ℝ := 0.1
  total_price prices discount_rate = 53.10 := by
  sorry

end NUMINAMATH_CALUDE_james_cd_purchase_total_l3660_366013


namespace NUMINAMATH_CALUDE_workers_per_team_lead_is_ten_l3660_366075

/-- Represents the hierarchical structure of a company -/
structure CompanyStructure where
  supervisors : ℕ
  workers : ℕ
  team_leads_per_supervisor : ℕ
  workers_per_team_lead : ℕ

/-- Calculates the number of workers per team lead given a company structure -/
def calculate_workers_per_team_lead (c : CompanyStructure) : ℕ :=
  c.workers / (c.supervisors * c.team_leads_per_supervisor)

/-- Theorem stating that for the given company structure, there are 10 workers per team lead -/
theorem workers_per_team_lead_is_ten :
  let c := CompanyStructure.mk 13 390 3 10
  calculate_workers_per_team_lead c = 10 := by
  sorry


end NUMINAMATH_CALUDE_workers_per_team_lead_is_ten_l3660_366075


namespace NUMINAMATH_CALUDE_max_substances_l3660_366058

/-- The number of substances generated when ethane is mixed with chlorine gas under lighting conditions -/
def num_substances : ℕ := sorry

/-- The number of isomers for monochloroethane -/
def mono_isomers : ℕ := 1

/-- The number of isomers for dichloroethane (including geometric isomers) -/
def di_isomers : ℕ := 3

/-- The number of isomers for trichloroethane -/
def tri_isomers : ℕ := 2

/-- The number of isomers for tetrachloroethane -/
def tetra_isomers : ℕ := 2

/-- The number of isomers for pentachloroethane -/
def penta_isomers : ℕ := 1

/-- The number of isomers for hexachloroethane -/
def hexa_isomers : ℕ := 1

/-- Hydrogen chloride is also formed -/
def hcl_formed : Prop := true

theorem max_substances :
  num_substances = mono_isomers + di_isomers + tri_isomers + tetra_isomers + penta_isomers + hexa_isomers + 1 ∧
  num_substances = 10 := by sorry

end NUMINAMATH_CALUDE_max_substances_l3660_366058


namespace NUMINAMATH_CALUDE_angle_bisector_construction_with_two_sided_ruler_l3660_366005

/-- A point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- An angle formed by two lines -/
structure Angle :=
  (vertex : Point)
  (line1 : Line)
  (line2 : Line)

/-- A two-sided ruler -/
structure TwoSidedRuler :=
  (length : ℝ)

/-- Definition of an angle bisector -/
def is_angle_bisector (a : Angle) (l : Line) : Prop :=
  sorry

/-- Definition of an inaccessible point -/
def is_inaccessible (p : Point) : Prop :=
  sorry

/-- Main theorem: It is possible to construct the bisector of an angle with an inaccessible vertex using only a two-sided ruler -/
theorem angle_bisector_construction_with_two_sided_ruler 
  (a : Angle) (r : TwoSidedRuler) (h : is_inaccessible a.vertex) : 
  ∃ (l : Line), is_angle_bisector a l :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_construction_with_two_sided_ruler_l3660_366005


namespace NUMINAMATH_CALUDE_andrew_fruit_purchase_cost_l3660_366033

/-- Calculates the total cost of fruits purchased by Andrew -/
theorem andrew_fruit_purchase_cost : 
  let grapes_quantity : ℕ := 14
  let grapes_price : ℕ := 54
  let mangoes_quantity : ℕ := 10
  let mangoes_price : ℕ := 62
  let pineapple_quantity : ℕ := 8
  let pineapple_price : ℕ := 40
  let kiwi_quantity : ℕ := 5
  let kiwi_price : ℕ := 30
  let total_cost := 
    grapes_quantity * grapes_price + 
    mangoes_quantity * mangoes_price + 
    pineapple_quantity * pineapple_price + 
    kiwi_quantity * kiwi_price
  total_cost = 1846 := by
  sorry


end NUMINAMATH_CALUDE_andrew_fruit_purchase_cost_l3660_366033


namespace NUMINAMATH_CALUDE_cross_product_result_l3660_366066

def vector1 : ℝ × ℝ × ℝ := (3, -4, 7)
def vector2 : ℝ × ℝ × ℝ := (2, 5, -1)

def cross_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a1, a2, a3) := v1
  let (b1, b2, b3) := v2
  (a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1)

theorem cross_product_result :
  cross_product vector1 vector2 = (-31, 17, 23) := by
  sorry

end NUMINAMATH_CALUDE_cross_product_result_l3660_366066


namespace NUMINAMATH_CALUDE_circle_equation_correct_l3660_366032

/-- The equation of a circle with center (h, k) and radius r -/
def circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The specific circle we're considering -/
def specific_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 1)^2 = 16

theorem circle_equation_correct :
  ∀ x y : ℝ, specific_circle x y ↔ circle_equation x y 3 (-1) 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l3660_366032


namespace NUMINAMATH_CALUDE_line_intersection_theorem_l3660_366078

/-- The line L in the xy-plane --/
def line_L (m : ℝ) (x y : ℝ) : Prop :=
  5 * y + (2 * m - 4) * x - 10 * m = 0

/-- The rectangle OABC --/
def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 10 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6}

/-- Point D on OA --/
def point_D (m : ℝ) : ℝ × ℝ := (0, 2 * m)

/-- Point E on BC --/
def point_E (m : ℝ) : ℝ × ℝ := (10, 8 - 2 * m)

/-- Area of quadrilateral ADEB --/
def area_ADEB (m : ℝ) : ℝ := 20

/-- Area of rectangle OABC --/
def area_OABC : ℝ := 60

/-- Parallel line that divides the rectangle into three equal areas --/
def parallel_line (m : ℝ) (x y : ℝ) : Prop :=
  y = ((4 - 2 * m) / 5) * x + (2 * m - 2)

theorem line_intersection_theorem (m : ℝ) :
  (1 ≤ m ∧ m ≤ 3) ∧
  (area_ADEB m = (1 / 3) * area_OABC) ∧
  (∀ x y, parallel_line m x y → 
    ∃ F G, F ∈ rectangle ∧ G ∈ rectangle ∧
    line_L m F.1 F.2 ∧ line_L m G.1 G.2 ∧
    area_ADEB m = area_OABC / 3) :=
sorry

end NUMINAMATH_CALUDE_line_intersection_theorem_l3660_366078


namespace NUMINAMATH_CALUDE_sum_of_triangles_eq_22_l3660_366069

/-- Represents the value of a triangle with vertices a, b, and c -/
def triangle_value (a b c : ℕ) : ℕ := a * b + c

/-- The sum of the values of two specific triangles -/
def sum_of_triangles : ℕ :=
  triangle_value 3 2 5 + triangle_value 4 1 7

theorem sum_of_triangles_eq_22 : sum_of_triangles = 22 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_triangles_eq_22_l3660_366069


namespace NUMINAMATH_CALUDE_students_interested_in_all_subjects_prove_students_interested_in_all_subjects_l3660_366046

/-- Represents the number of students interested in a combination of subjects -/
structure InterestCounts where
  total : ℕ
  biology : ℕ
  chemistry : ℕ
  physics : ℕ
  none : ℕ
  onlyBiology : ℕ
  onlyPhysics : ℕ
  biologyAndChemistry : ℕ

/-- The theorem stating the number of students interested in all three subjects -/
theorem students_interested_in_all_subjects (counts : InterestCounts) : ℕ :=
  let all_three := counts.biology + counts.chemistry + counts.physics -
    (counts.onlyBiology + counts.biologyAndChemistry + counts.onlyPhysics) - 
    (counts.total - counts.none)
  2

/-- The main theorem proving the number of students interested in all subjects -/
theorem prove_students_interested_in_all_subjects : 
  ∃ (counts : InterestCounts), 
    counts.total = 40 ∧ 
    counts.biology = 20 ∧ 
    counts.chemistry = 10 ∧ 
    counts.physics = 8 ∧ 
    counts.none = 7 ∧ 
    counts.onlyBiology = 12 ∧ 
    counts.onlyPhysics = 4 ∧ 
    counts.biologyAndChemistry = 6 ∧ 
    students_interested_in_all_subjects counts = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_students_interested_in_all_subjects_prove_students_interested_in_all_subjects_l3660_366046


namespace NUMINAMATH_CALUDE_complex_square_eq_neg_two_i_l3660_366010

theorem complex_square_eq_neg_two_i (z : ℂ) (a b : ℝ) :
  z = Complex.mk a b → z^2 = Complex.I * (-2) → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_eq_neg_two_i_l3660_366010


namespace NUMINAMATH_CALUDE_rocket_ascent_time_l3660_366082

theorem rocket_ascent_time (n : ℕ) (a₁ d : ℝ) (h₁ : a₁ = 2) (h₂ : d = 2) :
  n * a₁ + (n * (n - 1) * d) / 2 = 240 → n = 15 :=
by sorry

end NUMINAMATH_CALUDE_rocket_ascent_time_l3660_366082


namespace NUMINAMATH_CALUDE_fiftieth_term_is_ten_l3660_366063

def sequence_term (n : ℕ) : ℕ := 
  Nat.sqrt (2 * n + 1/4 : ℚ).ceil.toNat + 1

theorem fiftieth_term_is_ten : sequence_term 50 = 10 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_is_ten_l3660_366063


namespace NUMINAMATH_CALUDE_tv_installment_plan_duration_l3660_366097

theorem tv_installment_plan_duration (cash_price down_payment monthly_payment cash_savings : ℕ) : 
  cash_price = 400 →
  down_payment = 120 →
  monthly_payment = 30 →
  cash_savings = 80 →
  (cash_price + cash_savings - down_payment) / monthly_payment = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_tv_installment_plan_duration_l3660_366097


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l3660_366018

theorem no_solution_for_equation : ¬∃ (a b : ℕ+), 
  a * b + 100 = 25 * Nat.lcm a b + 15 * Nat.gcd a b := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l3660_366018


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_S_l3660_366059

/-- The product of all non-zero digits of a positive integer -/
def p (n : ℕ+) : ℕ :=
  sorry

/-- The sum of p(n) for n from 1 to 999 -/
def S : ℕ :=
  (Finset.range 999).sum (fun i => p ⟨i + 1, Nat.succ_pos i⟩)

/-- 103 is the largest prime factor of S -/
theorem largest_prime_factor_of_S :
  ∃ (m : ℕ), S = 103 * m ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ S → q ≤ 103 :=
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_S_l3660_366059


namespace NUMINAMATH_CALUDE_radish_carrot_ratio_l3660_366006

theorem radish_carrot_ratio :
  let cucumbers : ℕ := 15
  let radishes : ℕ := 3 * cucumbers
  let carrots : ℕ := 9
  radishes / carrots = 5 := by
sorry

end NUMINAMATH_CALUDE_radish_carrot_ratio_l3660_366006


namespace NUMINAMATH_CALUDE_intersection_slope_l3660_366060

/-- Given two lines p and q that intersect at (1, 1), prove that the slope of q is -3 -/
theorem intersection_slope (k : ℝ) : 
  (∀ x y : ℝ, y = -2*x + 3 → y = k*x + 4) → -- Line p: y = -2x + 3, Line q: y = kx + 4
  1 = -2*1 + 3 →                            -- (1, 1) satisfies line p
  1 = k*1 + 4 →                             -- (1, 1) satisfies line q
  k = -3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_slope_l3660_366060


namespace NUMINAMATH_CALUDE_sequence_limit_uniqueness_l3660_366077

theorem sequence_limit_uniqueness (a : ℕ → ℝ) (l₁ l₂ : ℝ) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - l₁| < ε) →
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - l₂| < ε) →
  l₁ = l₂ :=
by sorry

end NUMINAMATH_CALUDE_sequence_limit_uniqueness_l3660_366077


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_three_l3660_366025

theorem subset_implies_a_equals_three (A B : Set ℝ) (a : ℝ) : 
  A = {2, 3} → B = {1, 2, a} → A ⊆ B → a = 3 := by sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_three_l3660_366025


namespace NUMINAMATH_CALUDE_jesse_gift_amount_l3660_366008

/-- Prove that Jesse received $50 as a gift -/
theorem jesse_gift_amount (novel_cost lunch_cost remaining_amount : ℕ) : 
  novel_cost = 7 →
  lunch_cost = 2 * novel_cost →
  remaining_amount = 29 →
  novel_cost + lunch_cost + remaining_amount = 50 := by
  sorry

end NUMINAMATH_CALUDE_jesse_gift_amount_l3660_366008


namespace NUMINAMATH_CALUDE_wood_length_equation_l3660_366055

/-- Represents the length of a piece of wood that satisfies the measurement conditions. -/
def wood_length (x : ℝ) : Prop :=
  ∃ (rope_length : ℝ),
    rope_length - x = 4.5 ∧
    (rope_length / 2) - x = 1

/-- Proves that the wood length satisfies the equation from the problem. -/
theorem wood_length_equation (x : ℝ) :
  wood_length x → (x + 4.5) / 2 = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_wood_length_equation_l3660_366055


namespace NUMINAMATH_CALUDE_sphere_area_equals_volume_l3660_366038

theorem sphere_area_equals_volume (r : ℝ) (h : r > 0) :
  4 * Real.pi * r^2 = (4/3) * Real.pi * r^3 → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_area_equals_volume_l3660_366038


namespace NUMINAMATH_CALUDE_ceiling_floor_product_range_l3660_366064

theorem ceiling_floor_product_range (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 120 → -11 < y ∧ y < -10 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_range_l3660_366064


namespace NUMINAMATH_CALUDE_coconut_trips_l3660_366030

def total_coconuts : ℕ := 144
def barbie_capacity : ℕ := 4
def bruno_capacity : ℕ := 8

theorem coconut_trips : 
  (total_coconuts / (barbie_capacity + bruno_capacity) : ℕ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_coconut_trips_l3660_366030


namespace NUMINAMATH_CALUDE_wrong_number_calculation_l3660_366041

def wrong_number (n : ℕ) (initial_avg : ℚ) (correct_num : ℕ) (correct_avg : ℚ) : ℚ :=
  n * initial_avg + correct_num - (n * correct_avg)

theorem wrong_number_calculation (n : ℕ) (initial_avg correct_avg : ℚ) (correct_num : ℕ) :
  n = 10 →
  initial_avg = 5 →
  correct_num = 36 →
  correct_avg = 6 →
  wrong_number n initial_avg correct_num correct_avg = 26 := by
  sorry

end NUMINAMATH_CALUDE_wrong_number_calculation_l3660_366041


namespace NUMINAMATH_CALUDE_initial_men_count_l3660_366076

/-- Proves that the initial number of men is 760, given the food supply conditions. -/
theorem initial_men_count (M : ℕ) : 
  (M * 22 = (M + 40) * 19 + M * 2) → M = 760 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l3660_366076


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3660_366011

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 2 * x - 1 = 0) ↔ (k ≥ -1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3660_366011


namespace NUMINAMATH_CALUDE_tax_discount_commute_price_difference_is_zero_l3660_366098

/-- Proves that the order of applying tax and discount doesn't affect the final price -/
theorem tax_discount_commute (price : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) 
  (h_tax : 0 ≤ tax_rate) (h_discount : 0 ≤ discount_rate) (h_discount_max : discount_rate ≤ 1) :
  price * (1 + tax_rate) * (1 - discount_rate) = price * (1 - discount_rate) * (1 + tax_rate) :=
by sorry

/-- Calculates the difference between applying tax then discount and applying discount then tax -/
def price_difference (price : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) : ℝ :=
  price * (1 + tax_rate) * (1 - discount_rate) - price * (1 - discount_rate) * (1 + tax_rate)

/-- Proves that the price difference is always zero -/
theorem price_difference_is_zero (price : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) 
  (h_tax : 0 ≤ tax_rate) (h_discount : 0 ≤ discount_rate) (h_discount_max : discount_rate ≤ 1) :
  price_difference price tax_rate discount_rate = 0 :=
by sorry

end NUMINAMATH_CALUDE_tax_discount_commute_price_difference_is_zero_l3660_366098


namespace NUMINAMATH_CALUDE_incorrect_inequality_transformation_l3660_366062

theorem incorrect_inequality_transformation (a b : ℝ) (h : a > b) :
  ¬(1 - a > 1 - b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_transformation_l3660_366062


namespace NUMINAMATH_CALUDE_marked_price_is_correct_l3660_366085

/-- The marked price of a down jacket -/
def marked_price : ℝ := 550

/-- The cost price of the down jacket -/
def cost_price : ℝ := 350

/-- The selling price as a percentage of the marked price -/
def selling_percentage : ℝ := 0.8

/-- The profit made on the sale -/
def profit : ℝ := 90

/-- Theorem stating that the marked price is correct given the conditions -/
theorem marked_price_is_correct : 
  selling_percentage * marked_price - cost_price = profit :=
by sorry

end NUMINAMATH_CALUDE_marked_price_is_correct_l3660_366085


namespace NUMINAMATH_CALUDE_interest_rate_problem_l3660_366045

/-- Given a principal amount and an interest rate, if increasing the interest rate by 3%
    results in 210 more interest over 10 years, then the principal amount must be 700. -/
theorem interest_rate_problem (P R : ℝ) (h : P * (R + 3) * 10 / 100 = P * R * 10 / 100 + 210) :
  P = 700 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l3660_366045


namespace NUMINAMATH_CALUDE_function_inequality_and_logarithm_comparison_l3660_366073

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x - |x + 2| - |x - 3| - m

-- State the theorem
theorem function_inequality_and_logarithm_comparison (m : ℝ) 
  (h : ∀ x : ℝ, (1 / m) - 4 ≥ f m x) : 
  m > 0 ∧ Real.log (m + 2) / Real.log (m + 1) > Real.log (m + 3) / Real.log (m + 2) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_and_logarithm_comparison_l3660_366073


namespace NUMINAMATH_CALUDE_fraction_expression_equality_l3660_366036

theorem fraction_expression_equality : (3/7 + 5/8) / (5/12 + 2/9) = 531/322 := by
  sorry

end NUMINAMATH_CALUDE_fraction_expression_equality_l3660_366036


namespace NUMINAMATH_CALUDE_trishul_investment_percentage_l3660_366043

/-- Proves that Trishul invested 10% less than Raghu -/
theorem trishul_investment_percentage (vishal trishul raghu : ℝ) : 
  vishal = 1.1 * trishul →  -- Vishal invested 10% more than Trishul
  vishal + trishul + raghu = 6069 →  -- Total sum of investments
  raghu = 2100 →  -- Raghu's investment
  (raghu - trishul) / raghu = 0.1 :=  -- Trishul invested 10% less than Raghu
by sorry

end NUMINAMATH_CALUDE_trishul_investment_percentage_l3660_366043


namespace NUMINAMATH_CALUDE_denominator_divisor_not_zero_l3660_366083

theorem denominator_divisor_not_zero :
  ∀ (a : ℝ), a ≠ 0 → (∃ (b : ℝ), b / a = b / a) ∧ (∃ (c d : ℝ), c / d = c / d) :=
by sorry

end NUMINAMATH_CALUDE_denominator_divisor_not_zero_l3660_366083


namespace NUMINAMATH_CALUDE_rhombus_transformations_l3660_366050

/-- Represents a point transformation on the plane -/
def PointTransformation := (ℤ × ℤ) → (ℤ × ℤ)

/-- Transformation of type (i) -/
def transform_i (α : ℤ) : PointTransformation :=
  λ (x, y) => (x, α * x + y)

/-- Transformation of type (ii) -/
def transform_ii (α : ℤ) : PointTransformation :=
  λ (x, y) => (x + α * y, y)

/-- A rhombus with integer-coordinate vertices -/
structure IntegerRhombus :=
  (v1 v2 v3 v4 : ℤ × ℤ)

/-- Checks if a quadrilateral is a square -/
def is_square (q : IntegerRhombus) : Prop := sorry

/-- Checks if a quadrilateral is a non-square rectangle -/
def is_non_square_rectangle (q : IntegerRhombus) : Prop := sorry

/-- Applies a series of transformations to a rhombus -/
def apply_transformations (r : IntegerRhombus) (ts : List PointTransformation) : IntegerRhombus := sorry

/-- Main theorem statement -/
theorem rhombus_transformations :
  (¬ ∃ (r : IntegerRhombus) (ts : List PointTransformation),
     is_square (apply_transformations r ts)) ∧
  (∃ (r : IntegerRhombus) (ts : List PointTransformation),
     is_non_square_rectangle (apply_transformations r ts)) := by
  sorry

end NUMINAMATH_CALUDE_rhombus_transformations_l3660_366050


namespace NUMINAMATH_CALUDE_opposite_of_A_is_F_l3660_366091

-- Define the labels for the cube faces
inductive CubeFace
  | A | B | C | D | E | F

-- Define a structure for the cube
structure Cube where
  faces : Finset CubeFace
  opposite : CubeFace → CubeFace

-- Define the properties of the cube
axiom cube_has_six_faces : ∀ (c : Cube), c.faces.card = 6

axiom cube_has_unique_opposite : ∀ (c : Cube) (f : CubeFace), 
  f ∈ c.faces → c.opposite f ∈ c.faces ∧ c.opposite (c.opposite f) = f

axiom cube_opposite_distinct : ∀ (c : Cube) (f : CubeFace), 
  f ∈ c.faces → c.opposite f ≠ f

-- Theorem to prove
theorem opposite_of_A_is_F (c : Cube) : 
  CubeFace.A ∈ c.faces → c.opposite CubeFace.A = CubeFace.F := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_A_is_F_l3660_366091


namespace NUMINAMATH_CALUDE_median_squares_sum_l3660_366016

/-- Given a triangle with sides a, b, c and corresponding medians s_a, s_b, s_c,
    the sum of the squares of the medians is equal to 3/4 times the sum of the squares of the sides. -/
theorem median_squares_sum (a b c s_a s_b s_c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_s_a : s_a^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_s_b : s_b^2 = (2*a^2 + 2*c^2 - b^2) / 4)
  (h_s_c : s_c^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  s_a^2 + s_b^2 + s_c^2 = 3/4 * (a^2 + b^2 + c^2) := by
  sorry


end NUMINAMATH_CALUDE_median_squares_sum_l3660_366016


namespace NUMINAMATH_CALUDE_complex_number_location_l3660_366052

theorem complex_number_location (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (1 + i) / i
  z = 1 - i ∧ z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l3660_366052


namespace NUMINAMATH_CALUDE_quadratic_root_implies_quintic_root_l3660_366035

theorem quadratic_root_implies_quintic_root (r : ℝ) : 
  r^2 - r - 2 = 0 → r^5 - 11*r - 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_quintic_root_l3660_366035


namespace NUMINAMATH_CALUDE_courtyard_length_l3660_366054

/-- The length of a rectangular courtyard given its width and paving stones --/
theorem courtyard_length (width : ℝ) (num_stones : ℕ) (stone_length stone_width : ℝ)
  (h_width : width = 16.5)
  (h_num_stones : num_stones = 165)
  (h_stone_length : stone_length = 2.5)
  (h_stone_width : stone_width = 2) :
  width * (num_stones * stone_length * stone_width / width) = 50 := by
  sorry

#check courtyard_length

end NUMINAMATH_CALUDE_courtyard_length_l3660_366054


namespace NUMINAMATH_CALUDE_number_equation_solution_l3660_366096

theorem number_equation_solution : 
  ∃ x : ℝ, (0.75 * x + 2 = 8) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3660_366096


namespace NUMINAMATH_CALUDE_room_population_problem_l3660_366037

theorem room_population_problem (initial_men initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →
  (initial_men + 2) = 14 →
  (2 * (initial_women - 3)) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_room_population_problem_l3660_366037


namespace NUMINAMATH_CALUDE_binary_101101_equals_45_l3660_366065

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101101_equals_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_equals_45_l3660_366065


namespace NUMINAMATH_CALUDE_playground_girls_count_l3660_366056

theorem playground_girls_count (total_children boys : ℕ) 
  (h1 : total_children = 117) 
  (h2 : boys = 40) : 
  total_children - boys = 77 := by
sorry

end NUMINAMATH_CALUDE_playground_girls_count_l3660_366056


namespace NUMINAMATH_CALUDE_largest_divisor_of_factorial_l3660_366071

theorem largest_divisor_of_factorial (m n : ℕ) (hm : m ≥ 3) (hn : n > m * (m - 2)) :
  (∃ (d : ℕ), d > 0 ∧ d ∣ n.factorial ∧ ∀ k ∈ Finset.Icc m n, ¬(k ∣ d)) →
  (∃ (d : ℕ), d > 0 ∧ d ∣ n.factorial ∧ ∀ k ∈ Finset.Icc m n, ¬(k ∣ d) ∧
    ∀ d' > 0, d' ∣ n.factorial → (∀ k ∈ Finset.Icc m n, ¬(k ∣ d')) → d' ≤ d) →
  (m - 1 : ℕ) > 0 ∧ (m - 1 : ℕ) ∣ n.factorial ∧ ∀ k ∈ Finset.Icc m n, ¬(k ∣ (m - 1 : ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_factorial_l3660_366071


namespace NUMINAMATH_CALUDE_emily_final_score_l3660_366007

def emily_game (round1 round2 round3 round4 round5 round6_initial : Int) : Int :=
  let round6 := round6_initial - (2 * round5) / 3
  round1 + round2 + round3 + round4 + round5 + round6

theorem emily_final_score :
  emily_game 16 33 (-25) 46 12 30 = 104 := by
  sorry

end NUMINAMATH_CALUDE_emily_final_score_l3660_366007


namespace NUMINAMATH_CALUDE_value_calculation_l3660_366034

theorem value_calculation (number : ℕ) (value : ℕ) (h1 : number = 16) (h2 : value = 2 * number - 12) : value = 20 := by
  sorry

end NUMINAMATH_CALUDE_value_calculation_l3660_366034


namespace NUMINAMATH_CALUDE_equilateral_triangles_in_54gon_l3660_366084

/-- Represents a regular polygon with its center -/
structure RegularPolygonWithCenter (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  center : ℝ × ℝ

/-- Represents a selection of three points -/
structure TriangleSelection (n : ℕ) where
  p1 : Fin (n + 1)
  p2 : Fin (n + 1)
  p3 : Fin (n + 1)

/-- Checks if three points form an equilateral triangle -/
def isEquilateralTriangle (n : ℕ) (poly : RegularPolygonWithCenter n) (sel : TriangleSelection n) : Prop :=
  sorry

/-- Counts the number of equilateral triangles in a regular polygon with center -/
def countEquilateralTriangles (n : ℕ) (poly : RegularPolygonWithCenter n) : ℕ :=
  sorry

/-- The main theorem: there are 72 ways to select three points forming an equilateral triangle in a regular 54-gon with center -/
theorem equilateral_triangles_in_54gon :
  ∀ (poly : RegularPolygonWithCenter 54),
  countEquilateralTriangles 54 poly = 72 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangles_in_54gon_l3660_366084


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l3660_366040

/-- The number of distinct diagonals in a convex nonagon -/
def num_diagonals_nonagon : ℕ := 27

/-- A convex nonagon has 27 distinct diagonals -/
theorem nonagon_diagonals : 
  num_diagonals_nonagon = 27 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l3660_366040


namespace NUMINAMATH_CALUDE_min_wins_for_playoffs_l3660_366026

theorem min_wins_for_playoffs (total_games : ℕ) (win_points loss_points min_points : ℕ) :
  total_games = 22 →
  win_points = 2 →
  loss_points = 1 →
  min_points = 36 →
  (∃ (wins : ℕ), 
    wins ≤ total_games ∧ 
    wins * win_points + (total_games - wins) * loss_points ≥ min_points ∧
    ∀ (w : ℕ), w < wins → w * win_points + (total_games - w) * loss_points < min_points) →
  14 = (min_points - total_games * loss_points) / (win_points - loss_points) := by
sorry

end NUMINAMATH_CALUDE_min_wins_for_playoffs_l3660_366026


namespace NUMINAMATH_CALUDE_binomial_expansions_l3660_366012

theorem binomial_expansions (x a b : ℝ) : 
  ((x + 1) * (x + 2) = x^2 + 3*x + 2) ∧
  ((x + 1) * (x - 2) = x^2 - x - 2) ∧
  ((x - 1) * (x + 2) = x^2 + x - 2) ∧
  ((x - 1) * (x - 2) = x^2 - 3*x + 2) ∧
  ((x + a) * (x + b) = x^2 + (a + b)*x + a*b) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansions_l3660_366012


namespace NUMINAMATH_CALUDE_train_length_l3660_366099

/-- The length of a train given its speed and time to cross a bridge -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 45 * 1000 / 3600 →
  bridge_length = 205 →
  crossing_time = 30 →
  (train_speed * crossing_time) - bridge_length = 170 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3660_366099


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l3660_366089

-- Define sets A, B, and C
def A (a : ℝ) := {x : ℝ | a^2 - a*x + x - 1 = 0}
def B (m : ℝ) := {x : ℝ | x^2 + x + m = 0}
def C := {x : ℝ | Real.sqrt (x^2) = x}

-- Theorem for part (1)
theorem range_of_a (a : ℝ) : A a ∪ C = C → a ∈ Set.Icc (-1) 1 ∪ Set.Ioi 1 := by
  sorry

-- Theorem for part (2)
theorem range_of_m (m : ℝ) : C ∩ B m = ∅ → m ∈ Set.Ioi 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l3660_366089


namespace NUMINAMATH_CALUDE_sqrt_real_range_l3660_366080

theorem sqrt_real_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_real_range_l3660_366080


namespace NUMINAMATH_CALUDE_last_installment_theorem_l3660_366047

/-- Represents the installment payment plan for a TV set. -/
structure TVInstallmentPlan where
  total_price : ℕ
  num_installments : ℕ
  installment_amount : ℕ
  interest_rate : ℚ
  first_installment_at_purchase : Bool

/-- Calculates the value of the last installment in a TV installment plan. -/
def last_installment_value (plan : TVInstallmentPlan) : ℕ :=
  plan.installment_amount

/-- Theorem stating that the last installment value is equal to the regular installment amount. -/
theorem last_installment_theorem (plan : TVInstallmentPlan)
  (h1 : plan.total_price = 10000)
  (h2 : plan.num_installments = 20)
  (h3 : plan.installment_amount = 1000)
  (h4 : plan.interest_rate = 6 / 100)
  (h5 : plan.first_installment_at_purchase = true) :
  last_installment_value plan = 1000 := by
  sorry

#eval last_installment_value {
  total_price := 10000,
  num_installments := 20,
  installment_amount := 1000,
  interest_rate := 6 / 100,
  first_installment_at_purchase := true
}

end NUMINAMATH_CALUDE_last_installment_theorem_l3660_366047


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3660_366079

theorem complex_equation_solution (z : ℂ) :
  (1 - Complex.I)^2 / z = 1 + Complex.I → z = -1 - Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3660_366079


namespace NUMINAMATH_CALUDE_tan_half_positive_in_second_quadrant_l3660_366029

theorem tan_half_positive_in_second_quadrant (θ : Real) : 
  (π/2 < θ ∧ θ < π) → 0 < Real.tan (θ/2) := by
  sorry

end NUMINAMATH_CALUDE_tan_half_positive_in_second_quadrant_l3660_366029


namespace NUMINAMATH_CALUDE_gcf_of_lcm_equals_five_l3660_366004

theorem gcf_of_lcm_equals_five : Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcm_equals_five_l3660_366004


namespace NUMINAMATH_CALUDE_escalator_least_time_l3660_366014

/-- The least time needed for people to go up an escalator with variable speed -/
theorem escalator_least_time (n l α : ℝ) (hn : n > 0) (hl : l > 0) (hα : α > 0) :
  let speed (m : ℝ) := m ^ (-α)
  let time_one_by_one := n * l
  let time_all_together := l * n ^ α
  min time_one_by_one time_all_together = l * n ^ min α 1 := by
  sorry

end NUMINAMATH_CALUDE_escalator_least_time_l3660_366014


namespace NUMINAMATH_CALUDE_dislike_tv_and_books_l3660_366087

/-- Given a survey of people, calculate the number who dislike both TV and books -/
theorem dislike_tv_and_books (total : ℕ) (tv_dislike_percent : ℚ) (book_dislike_percent : ℚ) :
  total = 1500 →
  tv_dislike_percent = 40 / 100 →
  book_dislike_percent = 15 / 100 →
  (total * tv_dislike_percent * book_dislike_percent).floor = 90 := by
  sorry

end NUMINAMATH_CALUDE_dislike_tv_and_books_l3660_366087


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3660_366053

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = Complex.I) :
  z.im = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3660_366053


namespace NUMINAMATH_CALUDE_disjunction_truth_implication_l3660_366086

-- Define propositions p and q
variable (p q : Prop)

-- Define the statement to be proven
theorem disjunction_truth_implication (h : p ∨ q) : ¬(p ∧ q) := by
  sorry

-- This theorem states that if p ∨ q is true, it does not necessarily imply that both p and q are true.
-- It directly corresponds to showing that statement D is incorrect.

end NUMINAMATH_CALUDE_disjunction_truth_implication_l3660_366086


namespace NUMINAMATH_CALUDE_statue_final_weight_l3660_366024

/-- Calculates the final weight of a statue after three weeks of carving. -/
def final_statue_weight (initial_weight : ℝ) : ℝ :=
  let weight_after_first_week := initial_weight * (1 - 0.3)
  let weight_after_second_week := weight_after_first_week * (1 - 0.3)
  let weight_after_third_week := weight_after_second_week * (1 - 0.15)
  weight_after_third_week

/-- Theorem stating that the final weight of the statue is 124.95 kg. -/
theorem statue_final_weight :
  final_statue_weight 300 = 124.95 := by
  sorry

end NUMINAMATH_CALUDE_statue_final_weight_l3660_366024


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l3660_366022

open Real

noncomputable def f (x : ℝ) : ℝ := exp x + 1 / (exp x)

theorem f_increasing_on_interval (x : ℝ) (h : x > 1/exp 1) : 
  deriv f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l3660_366022


namespace NUMINAMATH_CALUDE_computers_needed_after_increase_problem_solution_l3660_366017

theorem computers_needed_after_increase (initial_students : ℕ) 
  (students_per_computer : ℕ) (additional_students : ℕ) : ℕ :=
  let initial_computers := initial_students / students_per_computer
  let additional_computers := additional_students / students_per_computer
  initial_computers + additional_computers

theorem problem_solution :
  computers_needed_after_increase 82 2 16 = 49 := by
  sorry

end NUMINAMATH_CALUDE_computers_needed_after_increase_problem_solution_l3660_366017


namespace NUMINAMATH_CALUDE_hundred_thousand_eq_scientific_notation_l3660_366002

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Definition of the number 100,000 -/
def hundred_thousand : ℕ := 100000

/-- The scientific notation of 100,000 -/
def hundred_thousand_scientific : ScientificNotation :=
  ⟨1, 5, by {sorry}⟩

/-- Theorem stating that 100,000 is equal to its scientific notation representation -/
theorem hundred_thousand_eq_scientific_notation :
  (hundred_thousand : ℝ) = hundred_thousand_scientific.coefficient * (10 : ℝ) ^ hundred_thousand_scientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_hundred_thousand_eq_scientific_notation_l3660_366002


namespace NUMINAMATH_CALUDE_tangent_slope_angle_at_zero_l3660_366090

open Real

noncomputable def f (x : ℝ) : ℝ := exp x * cos x

theorem tangent_slope_angle_at_zero (α : ℝ) :
  (∀ x, HasDerivAt f (exp x * (cos x - sin x)) x) →
  HasDerivAt f 1 0 →
  0 ≤ α →
  α < π →
  tan α = 1 →
  α = π / 4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_at_zero_l3660_366090


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_2772_l3660_366044

/-- The product of the digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- Check if a number is a five-digit integer -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem largest_five_digit_with_product_2772 :
  ∀ n : ℕ, is_five_digit n → digit_product n = 2772 → n ≤ 98721 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_2772_l3660_366044


namespace NUMINAMATH_CALUDE_digit_sum_problem_l3660_366095

theorem digit_sum_problem (P Q : ℕ) (h1 : P < 10) (h2 : Q < 10) 
  (h3 : 1013 + 1000 * P + 100 * Q + 10 * P + Q = 2023) : P + Q = 1 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l3660_366095


namespace NUMINAMATH_CALUDE_range_of_squared_plus_linear_l3660_366092

theorem range_of_squared_plus_linear (a b : ℝ) (h1 : a < -2) (h2 : b > 4) :
  a^2 + b > 8 := by sorry

end NUMINAMATH_CALUDE_range_of_squared_plus_linear_l3660_366092


namespace NUMINAMATH_CALUDE_extremum_implies_f_of_2_l3660_366000

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

-- State the theorem
theorem extremum_implies_f_of_2 (a b : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 ∧ |x - 1| < ε → f a b x ≥ f a b 1) ∧
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 ∧ |x - 1| < ε → f a b x ≤ f a b 1) ∧
  f a b 1 = -2 →
  f a b 2 = 3 :=
sorry

end NUMINAMATH_CALUDE_extremum_implies_f_of_2_l3660_366000


namespace NUMINAMATH_CALUDE_one_not_identity_for_star_l3660_366021

/-- The set of all non-zero real numbers -/
def S : Set ℝ := {x : ℝ | x ≠ 0}

/-- The binary operation * on S -/
def star (a b : ℝ) : ℝ := a^2 + 2*a*b

/-- Theorem: 1 is not an identity element for * in S -/
theorem one_not_identity_for_star :
  ¬(∀ a : ℝ, a ∈ S → (star 1 a = a ∧ star a 1 = a)) :=
sorry

end NUMINAMATH_CALUDE_one_not_identity_for_star_l3660_366021


namespace NUMINAMATH_CALUDE_angle_measure_proof_l3660_366048

theorem angle_measure_proof (A B : ℝ) : 
  (A = B ∨ A + B = 180) →  -- Parallel sides condition
  A = 3 * B - 20 →         -- Relationship between A and B
  A = 10 ∨ A = 130 :=      -- Conclusion
by sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l3660_366048


namespace NUMINAMATH_CALUDE_fq_length_l3660_366068

/-- Represents a right triangle with a tangent circle -/
structure RightTriangleWithCircle where
  /-- Length of the hypotenuse -/
  df : ℝ
  /-- Length of one leg -/
  de : ℝ
  /-- Point where the circle meets the hypotenuse -/
  q : ℝ
  /-- The hypotenuse is √85 -/
  hyp_length : df = Real.sqrt 85
  /-- One leg is 7 -/
  leg_length : de = 7
  /-- The circle is tangent to both legs -/
  circle_tangent : True

/-- The length of FQ in the given configuration is 6 -/
theorem fq_length (t : RightTriangleWithCircle) : t.df - t.q = 6 := by
  sorry

end NUMINAMATH_CALUDE_fq_length_l3660_366068


namespace NUMINAMATH_CALUDE_circles_diameter_sum_l3660_366081

theorem circles_diameter_sum (D d : ℝ) (h1 : D > d) (h2 : D - d = 9) (h3 : D / 2 - 5 > 0) :
  let TO := D / 2 - 5
  let OC := (D - d) / 2
  let CT := d / 2
  TO ^ 2 + OC ^ 2 = CT ^ 2 → d + D = 91 := by
sorry

end NUMINAMATH_CALUDE_circles_diameter_sum_l3660_366081


namespace NUMINAMATH_CALUDE_kyler_wins_two_l3660_366061

/-- Represents a chess player --/
inductive Player
| Peter
| Emma
| Kyler

/-- Represents the number of games won and lost by a player --/
structure GameRecord where
  player : Player
  wins : ℕ
  losses : ℕ

/-- The total number of games in the tournament --/
def totalGames : ℕ := 6

theorem kyler_wins_two (peter_record : GameRecord) (emma_record : GameRecord) (kyler_record : GameRecord) :
  peter_record.player = Player.Peter ∧
  peter_record.wins = 5 ∧
  peter_record.losses = 4 ∧
  emma_record.player = Player.Emma ∧
  emma_record.wins = 2 ∧
  emma_record.losses = 5 ∧
  kyler_record.player = Player.Kyler ∧
  kyler_record.losses = 4 →
  kyler_record.wins = 2 := by
  sorry

end NUMINAMATH_CALUDE_kyler_wins_two_l3660_366061


namespace NUMINAMATH_CALUDE_class_average_score_l3660_366015

theorem class_average_score (total_students : Nat) (score1 score2 : Nat) (other_avg : Nat) : 
  total_students = 40 →
  score1 = 98 →
  score2 = 100 →
  other_avg = 79 →
  (other_avg * (total_students - 2) + score1 + score2) / total_students = 80 :=
by sorry

end NUMINAMATH_CALUDE_class_average_score_l3660_366015


namespace NUMINAMATH_CALUDE_power_mod_seven_l3660_366001

theorem power_mod_seven : 5^1986 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seven_l3660_366001


namespace NUMINAMATH_CALUDE_supplies_to_budget_ratio_l3660_366057

def total_budget : ℚ := 3000
def food_fraction : ℚ := 1/3
def wages : ℚ := 1250

def supplies : ℚ := total_budget - (food_fraction * total_budget) - wages

theorem supplies_to_budget_ratio : 
  supplies / total_budget = 1/4 := by sorry

end NUMINAMATH_CALUDE_supplies_to_budget_ratio_l3660_366057


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l3660_366049

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distributeBalls (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 95 ways to distribute 7 distinguishable balls into 3 indistinguishable boxes -/
theorem seven_balls_three_boxes : distributeBalls 7 3 = 95 := by sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l3660_366049


namespace NUMINAMATH_CALUDE_modified_rubiks_cube_cubie_count_l3660_366093

/-- Represents a modified Rubik's cube with 8 corner cubies removed --/
structure ModifiedRubiksCube where
  /-- The number of small cubies with 4 painted faces --/
  four_face_cubies : Nat
  /-- The number of small cubies with 1 painted face --/
  one_face_cubies : Nat
  /-- The number of small cubies with 0 painted faces --/
  zero_face_cubies : Nat

/-- Theorem stating the correct number of cubies for each type in a modified Rubik's cube --/
theorem modified_rubiks_cube_cubie_count :
  ∃ (cube : ModifiedRubiksCube),
    cube.four_face_cubies = 12 ∧
    cube.one_face_cubies = 6 ∧
    cube.zero_face_cubies = 1 :=
by sorry

end NUMINAMATH_CALUDE_modified_rubiks_cube_cubie_count_l3660_366093


namespace NUMINAMATH_CALUDE_rain_probability_three_days_l3660_366074

def prob_rain_friday : ℝ := 0.4
def prob_rain_saturday : ℝ := 0.7
def prob_rain_sunday : ℝ := 0.3

theorem rain_probability_three_days :
  let prob_all_days := prob_rain_friday * prob_rain_saturday * prob_rain_sunday
  prob_all_days = 0.084 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_three_days_l3660_366074


namespace NUMINAMATH_CALUDE_diana_paint_remaining_l3660_366020

/-- The amount of paint required for each statue in gallons -/
def paint_per_statue : ℚ := 1 / 16

/-- The number of statues Diana can paint -/
def number_of_statues : ℕ := 14

/-- The total amount of paint Diana has remaining in gallons -/
def total_paint : ℚ := paint_per_statue * number_of_statues

/-- Theorem stating that the total paint Diana has remaining is 7/8 gallon -/
theorem diana_paint_remaining : total_paint = 7 / 8 := by sorry

end NUMINAMATH_CALUDE_diana_paint_remaining_l3660_366020


namespace NUMINAMATH_CALUDE_hockey_helmets_l3660_366019

theorem hockey_helmets (red blue : ℕ) : 
  red = blue + 6 → 
  red * 3 = blue * 5 → 
  red + blue = 24 := by
sorry

end NUMINAMATH_CALUDE_hockey_helmets_l3660_366019


namespace NUMINAMATH_CALUDE_square_sum_implies_product_l3660_366031

theorem square_sum_implies_product (x : ℝ) :
  (Real.sqrt (10 + x) + Real.sqrt (15 - x) = 8) →
  ((10 + x) * (15 - x) = 1521 / 4) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_implies_product_l3660_366031


namespace NUMINAMATH_CALUDE_g_is_even_l3660_366072

open Real

/-- A function F is odd if F(-x) = -F(x) for all x -/
def IsOdd (F : ℝ → ℝ) : Prop := ∀ x, F (-x) = -F x

/-- A function G is even if G(-x) = G(x) for all x -/
def IsEven (G : ℝ → ℝ) : Prop := ∀ x, G (-x) = G x

/-- Given a > 0, a ≠ 1, and F is an odd function, prove that G(x) = F(x) * (1 / (a^x - 1) + 1/2) is an even function -/
theorem g_is_even (a : ℝ) (ha : a > 0) (hna : a ≠ 1) (F : ℝ → ℝ) (hF : IsOdd F) :
  IsEven (fun x ↦ F x * (1 / (a^x - 1) + 1/2)) := by
  sorry

end NUMINAMATH_CALUDE_g_is_even_l3660_366072


namespace NUMINAMATH_CALUDE_scores_statistics_l3660_366027

def scores : List ℕ := [85, 95, 85, 80, 80, 85]

/-- The mode of a list of natural numbers -/
def mode (l : List ℕ) : ℕ := sorry

/-- The mean of a list of natural numbers -/
def mean (l : List ℕ) : ℚ := sorry

/-- The median of a list of natural numbers -/
def median (l : List ℕ) : ℚ := sorry

/-- The range of a list of natural numbers -/
def range (l : List ℕ) : ℕ := sorry

theorem scores_statistics :
  mode scores = 85 ∧
  mean scores = 85 ∧
  median scores = 85 ∧
  range scores = 15 := by sorry

end NUMINAMATH_CALUDE_scores_statistics_l3660_366027


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3660_366039

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the asymptotic lines
def asymptotic_lines (x y : ℝ) : Prop :=
  y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

-- Theorem statement
theorem hyperbola_properties
  (a b : ℝ)
  (h_positive : a > 0 ∧ b > 0)
  (h_point : hyperbola a b (-2) (Real.sqrt 6))
  (h_asymptotic : ∀ x y, hyperbola a b x y → asymptotic_lines x y) :
  -- 1) The equation of C is x^2 - y^2/2 = 1
  (∀ x y, hyperbola a b x y ↔ x^2 - y^2/2 = 1) ∧
  -- 2) P cannot be the midpoint of any chord AB of C
  (∀ A B : ℝ × ℝ,
    (hyperbola a b A.1 A.2 ∧ hyperbola a b B.1 B.2) →
    (∃ k : ℝ, A.2 - point_P.2 = k * (A.1 - point_P.1) ∧
              B.2 - point_P.2 = k * (B.1 - point_P.1)) →
    point_P ≠ ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l3660_366039


namespace NUMINAMATH_CALUDE_womens_doubles_handshakes_l3660_366070

/-- The number of handshakes in a women's doubles tennis tournament -/
theorem womens_doubles_handshakes (num_teams : ℕ) (team_size : ℕ) : 
  num_teams = 4 → team_size = 2 → num_teams * team_size * (num_teams * team_size - team_size) / 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_womens_doubles_handshakes_l3660_366070


namespace NUMINAMATH_CALUDE_gravitational_force_on_space_station_l3660_366094

/-- Gravitational force model -/
structure GravitationalModel where
  k : ℝ
  force : ℝ → ℝ
  h_inverse_square : ∀ d, d > 0 → force d = k / (d^2)

/-- Problem statement -/
theorem gravitational_force_on_space_station
  (model : GravitationalModel)
  (h_surface : model.force 6000 = 800)
  : model.force 360000 = 2/9 := by
  sorry


end NUMINAMATH_CALUDE_gravitational_force_on_space_station_l3660_366094


namespace NUMINAMATH_CALUDE_penalty_kicks_count_l3660_366042

theorem penalty_kicks_count (total_players : ℕ) (goalies : ℕ) : 
  total_players = 25 → goalies = 4 → (total_players - goalies) * goalies = 96 := by
  sorry

end NUMINAMATH_CALUDE_penalty_kicks_count_l3660_366042


namespace NUMINAMATH_CALUDE_towels_remaining_l3660_366003

/-- The number of green towels Maria bought -/
def green_bought : ℕ := 35

/-- The number of white towels Maria bought -/
def white_bought : ℕ := 21

/-- The number of blue towels Maria bought -/
def blue_bought : ℕ := 15

/-- The number of green towels Maria gave to her mother -/
def green_given : ℕ := 22

/-- The number of white towels Maria gave to her mother -/
def white_given : ℕ := 14

/-- The number of blue towels Maria gave to her mother -/
def blue_given : ℕ := 6

/-- The total number of towels Maria gave to her mother -/
def total_given : ℕ := 42

theorem towels_remaining : 
  (green_bought + white_bought + blue_bought) - total_given = 29 := by
  sorry

end NUMINAMATH_CALUDE_towels_remaining_l3660_366003
