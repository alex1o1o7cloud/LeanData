import Mathlib

namespace root_sum_reciprocal_l2332_233285

theorem root_sum_reciprocal (x₁ x₂ : ℝ) : 
  x₁^2 + x₁ = 2 → 
  x₂^2 + x₂ = 2 → 
  x₁ ≠ x₂ → 
  1/x₁ + 1/x₂ = 1/2 := by
  sorry

end root_sum_reciprocal_l2332_233285


namespace parallelogram_area_l2332_233228

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 10 and 20 is equal to 100√3. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 20) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin (π - θ) = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_l2332_233228


namespace boxes_with_neither_l2332_233223

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (erasers : ℕ) (both : ℕ)
  (h1 : total = 12)
  (h2 : markers = 8)
  (h3 : erasers = 5)
  (h4 : both = 4) :
  total - (markers + erasers - both) = 3 := by
  sorry

end boxes_with_neither_l2332_233223


namespace cos_two_pi_thirds_minus_alpha_l2332_233214

theorem cos_two_pi_thirds_minus_alpha (α : ℝ) 
  (h : Real.sin (α - π/6) = 1/3) : 
  Real.cos ((2*π)/3 - α) = 1/3 := by
sorry

end cos_two_pi_thirds_minus_alpha_l2332_233214


namespace circle_area_from_circumference_l2332_233246

theorem circle_area_from_circumference (c : ℝ) (h : c = 24) :
  let r := c / (2 * Real.pi)
  (Real.pi * r ^ 2) = 144 / Real.pi := by
  sorry

end circle_area_from_circumference_l2332_233246


namespace calculation_proof_l2332_233274

theorem calculation_proof :
  (125 * 76 * 4 * 8 * 25 = 7600000) ∧
  ((6742 + 6743 + 6738 + 6739 + 6741 + 6743) / 6 = 6741) := by
  sorry

end calculation_proof_l2332_233274


namespace equation_solutions_count_l2332_233204

theorem equation_solutions_count :
  let f : ℝ → ℝ := fun x => (x^2 - 7)^2 + 2*x^2 - 33
  ∃! (s : Finset ℝ), (∀ x ∈ s, f x = 0) ∧ Finset.card s = 4 :=
by sorry

end equation_solutions_count_l2332_233204


namespace original_group_size_l2332_233216

theorem original_group_size (original_days : ℕ) (absent_men : ℕ) (new_days : ℕ) :
  original_days = 6 →
  absent_men = 4 →
  new_days = 12 →
  ∃ (total_men : ℕ), 
    total_men > absent_men ∧
    (1 : ℚ) / (original_days * total_men) = (1 : ℚ) / (new_days * (total_men - absent_men)) ∧
    total_men = 8 := by
  sorry

end original_group_size_l2332_233216


namespace original_price_after_discounts_l2332_233271

theorem original_price_after_discounts (price : ℝ) : 
  price * (1 - 0.2) * (1 - 0.1) * (1 - 0.05) = 6840 → price = 10000 := by
  sorry

end original_price_after_discounts_l2332_233271


namespace extremum_implies_f_2_eq_18_l2332_233229

/-- A function f with an extremum at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_implies_f_2_eq_18 (a b : ℝ) :
  f' a b 1 = 0 ∧ f a b 1 = 10 → f a b 2 = 18 := by
  sorry

#check extremum_implies_f_2_eq_18

end extremum_implies_f_2_eq_18_l2332_233229


namespace probability_two_rainy_days_l2332_233212

/-- Represents the weather condition for a day -/
inductive Weather
| Rainy
| NotRainy

/-- Represents the weather for three consecutive days -/
def ThreeDayWeather := (Weather × Weather × Weather)

/-- Checks if a ThreeDayWeather has exactly two rainy days -/
def hasTwoRainyDays (w : ThreeDayWeather) : Bool :=
  match w with
  | (Weather.Rainy, Weather.Rainy, Weather.NotRainy) => true
  | (Weather.Rainy, Weather.NotRainy, Weather.Rainy) => true
  | (Weather.NotRainy, Weather.Rainy, Weather.Rainy) => true
  | _ => false

/-- The total number of weather groups in the sample -/
def totalGroups : Nat := 20

/-- The number of groups with exactly two rainy days -/
def groupsWithTwoRainyDays : Nat := 5

/-- Theorem: The probability of exactly two rainy days out of three is 0.25 -/
theorem probability_two_rainy_days :
  (groupsWithTwoRainyDays : ℚ) / totalGroups = 1 / 4 := by
  sorry


end probability_two_rainy_days_l2332_233212


namespace solution_l2332_233219

def problem (a b c d : ℚ) : Prop :=
  a + b + c + d = 406 ∧
  a = (1/2) * b ∧
  b = (1/2) * c ∧
  d = (1/3) * c

theorem solution :
  ∃ a b c d : ℚ,
    problem a b c d ∧
    a = 48.72 ∧
    b = 97.44 ∧
    c = 194.88 ∧
    d = 64.96 :=
by sorry

end solution_l2332_233219


namespace unique_complex_solution_l2332_233201

theorem unique_complex_solution :
  ∃! (z : ℂ), Complex.abs z < 20 ∧ Complex.cos z = (z - 2) / (z + 2) := by
  sorry

end unique_complex_solution_l2332_233201


namespace binomial_9_8_l2332_233281

theorem binomial_9_8 : Nat.choose 9 8 = 9 := by
  sorry

end binomial_9_8_l2332_233281


namespace max_profit_at_grade_5_l2332_233253

def profit (x : ℕ) : ℝ :=
  (4 * (x - 1) + 8) * (60 - 6 * (x - 1))

theorem max_profit_at_grade_5 :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → profit x ≤ profit 5 :=
by sorry

end max_profit_at_grade_5_l2332_233253


namespace min_composite_with_small_factors_l2332_233284

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

def has_prime_factorization (n : ℕ) (max_factor : ℕ) : Prop :=
  ∃ (factors : List ℕ), 
    factors.length ≥ 2 ∧
    (∀ p ∈ factors, is_prime p ∧ p ≤ max_factor) ∧
    factors.prod = n

theorem min_composite_with_small_factors :
  ∀ n : ℕ, 
    ¬ is_prime n →
    has_prime_factorization n 10 →
    n ≥ 6 :=
sorry

end min_composite_with_small_factors_l2332_233284


namespace average_score_proof_l2332_233249

theorem average_score_proof (total_students : Nat) (abc_students : Nat) (de_students : Nat)
  (total_average : ℚ) (abc_average : ℚ) :
  total_students = 5 →
  abc_students = 3 →
  de_students = 2 →
  total_average = 80 →
  abc_average = 78 →
  (total_students * total_average - abc_students * abc_average) / de_students = 83 := by
  sorry

end average_score_proof_l2332_233249


namespace largest_base_for_12_4th_power_l2332_233257

def base_expansion (n : ℕ) (b : ℕ) : List ℕ :=
  sorry

def sum_digits (digits : List ℕ) : ℕ :=
  sorry

def is_largest_base (b : ℕ) : Prop :=
  (∀ k > b, sum_digits (base_expansion ((k + 2)^4) k) = 32) ∧
  sum_digits (base_expansion ((b + 2)^4) b) ≠ 32

theorem largest_base_for_12_4th_power : is_largest_base 7 :=
  sorry

end largest_base_for_12_4th_power_l2332_233257


namespace painting_gift_options_l2332_233213

theorem painting_gift_options (n : ℕ) (h : n = 10) : 
  (Finset.powerset (Finset.range n)).card - 1 = 2^n - 1 := by
  sorry

end painting_gift_options_l2332_233213


namespace shaded_area_calculation_l2332_233207

/-- The area of the shaded region formed by two intersecting rectangles minus a circular cut-out -/
theorem shaded_area_calculation (rect1_width rect1_length rect2_width rect2_length : ℝ)
  (circle_radius : ℝ) (h1 : rect1_width = 3) (h2 : rect1_length = 12)
  (h3 : rect2_width = 4) (h4 : rect2_length = 7) (h5 : circle_radius = 1) :
  let rect1_area := rect1_width * rect1_length
  let rect2_area := rect2_width * rect2_length
  let overlap_area := min rect1_width rect2_width * min rect1_length rect2_length
  let circle_area := Real.pi * circle_radius^2
  rect1_area + rect2_area - overlap_area - circle_area = 64 - Real.pi :=
by sorry

end shaded_area_calculation_l2332_233207


namespace xiaojie_purchase_solution_l2332_233258

/-- Represents the stationery purchase problem --/
structure StationeryPurchase where
  red_black_pen_price : ℕ
  black_refill_price : ℕ
  red_refill_price : ℕ
  black_discount : ℚ
  red_discount : ℚ
  red_black_pens_bought : ℕ
  total_refills_bought : ℕ
  total_spent : ℕ

/-- The specific purchase made by Xiaojie --/
def xiaojie_purchase : StationeryPurchase :=
  { red_black_pen_price := 10
  , black_refill_price := 6
  , red_refill_price := 8
  , black_discount := 1/2
  , red_discount := 3/4
  , red_black_pens_bought := 2
  , total_refills_bought := 10
  , total_spent := 74
  }

/-- Theorem stating the correct number of refills bought and amount saved --/
theorem xiaojie_purchase_solution (p : StationeryPurchase) (h : p = xiaojie_purchase) :
  ∃ (black_refills red_refills : ℕ) (savings : ℕ),
    black_refills + red_refills = p.total_refills_bought ∧
    black_refills = 2 ∧
    red_refills = 8 ∧
    savings = 22 ∧
    p.red_black_pen_price * p.red_black_pens_bought +
    (p.black_refill_price * black_refills + p.red_refill_price * red_refills) -
    p.total_spent = savings :=
  sorry

end xiaojie_purchase_solution_l2332_233258


namespace odd_terms_sum_l2332_233245

def sequence_sum (n : ℕ) : ℕ := n^2 + 2*n - 1

def arithmetic_sum (first last : ℕ) (step : ℕ) : ℕ :=
  ((last - first) / step + 1) * (first + last) / 2

theorem odd_terms_sum :
  (arithmetic_sum 1 25 2) = 350 :=
by sorry

end odd_terms_sum_l2332_233245


namespace opposite_points_theorem_l2332_233241

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Given two points on a number line, proves that if they represent opposite numbers, 
    their distance is 8, and the first point is to the left of the second, 
    then they represent -4 and 4 respectively -/
theorem opposite_points_theorem (A B : Point) : 
  A.value + B.value = 0 →  -- A and B represent opposite numbers
  |A.value - B.value| = 8 →  -- Distance between A and B is 8
  A.value < B.value →  -- A is to the left of B
  A.value = -4 ∧ B.value = 4 := by
  sorry

end opposite_points_theorem_l2332_233241


namespace min_value_theorem_l2332_233270

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^2 + 10*x + 100/x^2 ≥ 79 ∧ ∃ y > 0, y^2 + 10*y + 100/y^2 = 79 :=
by sorry

end min_value_theorem_l2332_233270


namespace arithmetic_sqrt_of_nine_l2332_233292

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

-- State the theorem
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 := by
  sorry

end arithmetic_sqrt_of_nine_l2332_233292


namespace negative_two_cubed_equality_l2332_233231

theorem negative_two_cubed_equality : (-2)^3 = -2^3 := by
  sorry

end negative_two_cubed_equality_l2332_233231


namespace speed_in_still_water_l2332_233288

/-- Given a man's upstream and downstream speeds, calculate his speed in still water -/
theorem speed_in_still_water 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (h1 : upstream_speed = 37) 
  (h2 : downstream_speed = 53) : 
  (upstream_speed + downstream_speed) / 2 = 45 := by
  sorry

end speed_in_still_water_l2332_233288


namespace parallel_lines_m_value_l2332_233240

/-- Two lines are parallel if their slopes are equal and they are not the same line -/
def are_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 * b2 = a2 * b1 ∧ a1 * c2 ≠ a2 * c1

/-- The theorem stating that if two lines (3+m)x+4y=5-3m and 2x+(5+m)y=8 are parallel, then m = -7 -/
theorem parallel_lines_m_value (m : ℝ) :
  are_parallel (3 + m) 4 (3*m - 5) 2 (5 + m) (-8) → m = -7 := by
  sorry

end parallel_lines_m_value_l2332_233240


namespace impossible_transformation_l2332_233222

/-- Represents the operation of replacing two numbers with their updated values -/
def replace_numbers (numbers : List ℕ) (x y : ℕ) : List ℕ :=
  (x - 1) :: (y + 3) :: (numbers.filter (λ n => n ≠ x ∧ n ≠ y))

/-- Checks if a list of numbers is valid according to the problem rules -/
def is_valid_list (numbers : List ℕ) : Prop :=
  numbers.length = 10 ∧ numbers.sum % 2 = 1

/-- The initial list of numbers on the board -/
def initial_numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

/-- The target list of numbers we want to achieve -/
def target_numbers : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10, 2012]

/-- Theorem stating that it's impossible to transform the initial numbers into the target numbers -/
theorem impossible_transformation :
  ¬ ∃ (n : ℕ) (operations : List (ℕ × ℕ)),
    operations.length = n ∧
    (operations.foldl (λ acc (x, y) => replace_numbers acc x y) initial_numbers) = target_numbers :=
sorry

end impossible_transformation_l2332_233222


namespace sum_calculation_l2332_233248

def sequence_S : ℕ → ℕ
  | 0 => 0
  | (n + 1) => sequence_S n + (2 * n + 1)

def sequence_n : ℕ → ℕ
  | 0 => 1
  | (n + 1) => sequence_n n + 2

theorem sum_calculation :
  ∃ k : ℕ, sequence_n k > 50 ∧ sequence_n (k - 1) ≤ 50 ∧ sequence_S (k - 1) = 625 := by
  sorry

end sum_calculation_l2332_233248


namespace inequality_solution_l2332_233206

theorem inequality_solution (x : ℕ) (h : x > 1) :
  (6 * (9 ^ (1 / x)) - 13 * (3 ^ (1 / x)) * (2 ^ (1 / x)) + 6 * (4 ^ (1 / x)) ≤ 0) ↔ x ≥ 2 := by
  sorry

end inequality_solution_l2332_233206


namespace browns_utility_bill_l2332_233227

/-- The total amount of Mrs. Brown's utility bills -/
def utility_bill_total (fifty_count : ℕ) (ten_count : ℕ) : ℕ :=
  50 * fifty_count + 10 * ten_count

/-- Theorem stating that Mrs. Brown's utility bills total $170 -/
theorem browns_utility_bill : utility_bill_total 3 2 = 170 := by
  sorry

end browns_utility_bill_l2332_233227


namespace cube_side_length_ratio_l2332_233272

/-- Given two cubes of the same material, if the weight of the second cube
    is 8 times the weight of the first cube, then the ratio of the side length
    of the second cube to the side length of the first cube is 2:1. -/
theorem cube_side_length_ratio (s1 s2 : ℝ) (w1 w2 : ℝ) : 
  s1 > 0 → s2 > 0 → w1 > 0 → w2 > 0 →
  (w2 = 8 * w1) →
  (w1 = s1^3) →
  (w2 = s2^3) →
  s2 / s1 = 2 := by sorry

end cube_side_length_ratio_l2332_233272


namespace gcd_2197_2209_l2332_233276

theorem gcd_2197_2209 : Nat.gcd 2197 2209 = 1 := by
  sorry

end gcd_2197_2209_l2332_233276


namespace add_negative_numbers_add_positive_negative_add_negative_positive_inverse_subtract_larger_from_smaller_subtract_negative_add_negative_positive_real_abs_value_add_negative_multiply_negative_mixed_multiply_two_negatives_l2332_233251

-- 1. (-51) + (-37) = -88
theorem add_negative_numbers : (-51) + (-37) = -88 := by sorry

-- 2. (+2) + (-11) = -9
theorem add_positive_negative : (2 : Int) + (-11) = -9 := by sorry

-- 3. (-12) + (+12) = 0
theorem add_negative_positive_inverse : (-12) + (12 : Int) = 0 := by sorry

-- 4. 8 - 14 = -6
theorem subtract_larger_from_smaller : (8 : Int) - 14 = -6 := by sorry

-- 5. 15 - (-8) = 23
theorem subtract_negative : (15 : Int) - (-8) = 23 := by sorry

-- 6. (-3.4) + 4.3 = 0.9
theorem add_negative_positive_real : (-3.4) + 4.3 = 0.9 := by sorry

-- 7. |-2.25| + (-0.5) = 1.75
theorem abs_value_add_negative : |(-2.25 : ℝ)| + (-0.5) = 1.75 := by sorry

-- 8. -4 * 1.5 = -6
theorem multiply_negative_mixed : (-4 : ℝ) * 1.5 = -6 := by sorry

-- 9. -3 * (-6) = 18
theorem multiply_two_negatives : (-3 : Int) * (-6) = 18 := by sorry

end add_negative_numbers_add_positive_negative_add_negative_positive_inverse_subtract_larger_from_smaller_subtract_negative_add_negative_positive_real_abs_value_add_negative_multiply_negative_mixed_multiply_two_negatives_l2332_233251


namespace sum_of_common_elements_l2332_233221

def arithmetic_progression (n : ℕ) : ℕ := 4 + 3 * n

def geometric_progression (k : ℕ) : ℕ := 10 * 2^k

def common_elements (m : ℕ) : ℕ := 10 * 4^m

theorem sum_of_common_elements : 
  (Finset.range 10).sum (λ i => common_elements i) = 3495250 := by sorry

end sum_of_common_elements_l2332_233221


namespace gcd_154_90_l2332_233247

theorem gcd_154_90 : Nat.gcd 154 90 = 2 := by
  sorry

end gcd_154_90_l2332_233247


namespace student_event_arrangements_l2332_233200

theorem student_event_arrangements (n m : ℕ) (h1 : n = 7) (h2 : m = 5) : 
  (n.choose m * m.factorial) - ((n - 1).choose m * m.factorial) = 1800 := by
  sorry

end student_event_arrangements_l2332_233200


namespace circle_tangent_to_line_l2332_233254

/-- The value of m for which the circle x^2 + y^2 = 4m is tangent to the line x + y = 2√m -/
theorem circle_tangent_to_line (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4*m ∧ x + y = 2*Real.sqrt m → 
    (∃! p : ℝ × ℝ, p.1^2 + p.2^2 = 4*m ∧ p.1 + p.2 = 2*Real.sqrt m)) → 
  m = 0 :=
sorry

end circle_tangent_to_line_l2332_233254


namespace larger_number_proof_l2332_233299

theorem larger_number_proof (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 - y^2 = 39) : x = 8 := by
  sorry

end larger_number_proof_l2332_233299


namespace negative_fractions_comparison_l2332_233268

theorem negative_fractions_comparison : (-1/2 : ℚ) < -1/3 := by
  sorry

end negative_fractions_comparison_l2332_233268


namespace college_student_count_l2332_233217

theorem college_student_count (boys girls : ℕ) (h1 : boys = 2 * girls) (h2 : girls = 200) :
  boys + girls = 600 := by
  sorry

end college_student_count_l2332_233217


namespace right_triangle_power_equation_l2332_233297

theorem right_triangle_power_equation (a b c n : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  n > 2 →
  a^2 + b^2 = c^2 →
  (a^n + b^n + c^n)^2 = 2*(a^(2*n) + b^(2*n) + c^(2*n)) →
  n = 4 := by
sorry

end right_triangle_power_equation_l2332_233297


namespace xy_inequality_and_equality_l2332_233238

theorem xy_inequality_and_equality (x y : ℝ) (h : (x + 1) * (y + 2) = 8) :
  ((x * y - 10)^2 ≥ 64) ∧
  ((x * y - 10)^2 = 64 ↔ (x = 1 ∧ y = 2) ∨ (x = -3 ∧ y = -6)) := by
  sorry

end xy_inequality_and_equality_l2332_233238


namespace reptiles_in_swamps_l2332_233237

theorem reptiles_in_swamps (num_swamps : ℕ) (reptiles_per_swamp : ℕ) :
  num_swamps = 4 →
  reptiles_per_swamp = 356 →
  num_swamps * reptiles_per_swamp = 1424 := by
  sorry

end reptiles_in_swamps_l2332_233237


namespace tanner_has_16_berries_l2332_233205

/-- The number of berries each person has -/
structure Berries where
  skylar : ℕ
  steve : ℕ
  stacy : ℕ
  tanner : ℕ

/-- Calculate the number of berries each person has based on the given conditions -/
def calculate_berries : Berries :=
  let skylar := 20
  let steve := 4 * (skylar / 3)^2
  let stacy := 2 * steve + 50
  let tanner := (8 * stacy) / (skylar + steve)
  { skylar := skylar, steve := steve, stacy := stacy, tanner := tanner }

/-- Theorem stating that Tanner has 16 berries -/
theorem tanner_has_16_berries : (calculate_berries.tanner) = 16 := by
  sorry

#eval calculate_berries.tanner

end tanner_has_16_berries_l2332_233205


namespace sqrt_product_equation_l2332_233252

theorem sqrt_product_equation (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (8 * x) * Real.sqrt (10 * x) * Real.sqrt (3 * x) * Real.sqrt (15 * x) = 15) : 
  x = 1 / 2 := by
sorry

end sqrt_product_equation_l2332_233252


namespace trig_identity_proof_l2332_233202

theorem trig_identity_proof : 
  Real.cos (28 * π / 180) * Real.cos (17 * π / 180) - 
  Real.sin (28 * π / 180) * Real.cos (73 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end trig_identity_proof_l2332_233202


namespace circle_equations_correct_l2332_233295

-- Define the points A, B, and D
def A : ℝ × ℝ := (-1, 5)
def B : ℝ × ℝ := (5, 5)
def D : ℝ × ℝ := (5, -1)

-- Define circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 18

-- Define circle M
def circle_M (x y : ℝ) : Prop :=
  (x - 6)^2 + (y - 6)^2 = 2

-- Theorem statement
theorem circle_equations_correct :
  (circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ circle_C D.1 D.2) ∧
  (∃ (t : ℝ), circle_C (B.1 + t) (B.2 + t) ∧ circle_M (B.1 + t) (B.2 + t)) ∧
  (∀ (x y : ℝ), circle_M x y → (x - B.1)^2 + (y - B.2)^2 = 2) :=
by sorry

end circle_equations_correct_l2332_233295


namespace union_complement_equals_set_l2332_233262

def U : Set ℤ := {x | -3 < x ∧ x < 3}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem union_complement_equals_set : A ∪ (U \ B) = {0, 1, 2} := by sorry

end union_complement_equals_set_l2332_233262


namespace factor_x4_minus_64_l2332_233218

theorem factor_x4_minus_64 (x : ℝ) : x^4 - 64 = (x^2 + 8) * (x^2 - 8) := by
  sorry

end factor_x4_minus_64_l2332_233218


namespace correct_systematic_sample_l2332_233234

def total_missiles : ℕ := 60
def selected_missiles : ℕ := 6

def systematic_sample (total : ℕ) (select : ℕ) : List ℕ :=
  let interval := total / select
  List.range select |>.map (fun i => i * interval + interval / 2 + 1)

theorem correct_systematic_sample :
  systematic_sample total_missiles selected_missiles = [3, 13, 23, 33, 43, 53] :=
sorry

end correct_systematic_sample_l2332_233234


namespace fraction_irreducible_l2332_233230

theorem fraction_irreducible (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 := by
  sorry

end fraction_irreducible_l2332_233230


namespace hyperbola_asymptote_angle_l2332_233267

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > b,
    if the angle between its asymptotes is 45°, then a/b = √2 -/
theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ t : ℝ, ∃ x y : ℝ, y = (b / a) * x ∨ y = -(b / a) * x) →
  (Real.pi / 4 : ℝ) = Real.arctan ((b / a - (-b / a)) / (1 + (b / a) * (-b / a))) →
  a / b = Real.sqrt 2 := by
  sorry

end hyperbola_asymptote_angle_l2332_233267


namespace sin_2010_degrees_l2332_233286

theorem sin_2010_degrees : Real.sin (2010 * π / 180) = -1/2 := by
  sorry

end sin_2010_degrees_l2332_233286


namespace greatest_possible_median_l2332_233243

theorem greatest_possible_median (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 10 →
  k < m → m < r → r < s → s < t →
  t = 20 →
  r ≤ 13 ∧ ∃ (k' m' r' s' : ℕ), 
    k' > 0 ∧ m' > 0 ∧ r' > 0 ∧ s' > 0 ∧
    (k' + m' + r' + s' + 20) / 5 = 10 ∧
    k' < m' ∧ m' < r' ∧ r' < s' ∧ s' < 20 ∧
    r' = 13 :=
by
  sorry

end greatest_possible_median_l2332_233243


namespace complexity_theorem_l2332_233283

/-- Complexity of an integer is the number of prime factors in its prime decomposition -/
def complexity (n : ℕ) : ℕ := sorry

/-- n is a power of two -/
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem complexity_theorem (n : ℕ) (h : n > 1) :
  (∀ m : ℕ, n < m ∧ m ≤ 2*n → complexity m ≤ complexity n) ↔ is_power_of_two n ∧
  ¬∃ n : ℕ, ∀ m : ℕ, n < m ∧ m ≤ 2*n → complexity m < complexity n :=
sorry

end complexity_theorem_l2332_233283


namespace min_phase_shift_l2332_233239

/-- Given a sinusoidal function with a phase shift, prove that under certain symmetry conditions, 
    the smallest possible absolute value of the phase shift is π/4. -/
theorem min_phase_shift (φ : ℝ) : 
  (∀ x, 3 * Real.sin (3 * (x - π/4) + φ) = 3 * Real.sin (3 * (2*π/3 - x) + φ)) →
  (∃ k : ℤ, φ = k * π - π/4) →
  ∃ ψ : ℝ, abs ψ = π/4 ∧ (∀ θ : ℝ, (∃ k : ℤ, θ = k * π - π/4) → abs θ ≥ abs ψ) :=
by sorry

end min_phase_shift_l2332_233239


namespace expression_value_l2332_233220

theorem expression_value (p q : ℚ) (h : p / q = 4 / 5) :
  25 / 7 + (2 * q - p) / (2 * q + p) = 4 := by
  sorry

end expression_value_l2332_233220


namespace vector_problem_proof_l2332_233289

def vector_problem (a b : ℝ × ℝ) : Prop :=
  a = (1, 1) ∧
  (b.1^2 + b.2^2) = 16 ∧
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2)) = -2 →
  ((3*a.1 - b.1)^2 + (3*a.2 - b.2)^2) = 10

theorem vector_problem_proof : ∀ a b : ℝ × ℝ, vector_problem a b :=
by
  sorry

end vector_problem_proof_l2332_233289


namespace tank_emptying_time_l2332_233233

/-- Given a tank with specified capacity, leak rate, and inlet rate, 
    calculate the time it takes to empty when both leak and inlet are open. -/
theorem tank_emptying_time 
  (tank_capacity : ℝ) 
  (leak_time : ℝ) 
  (inlet_rate_per_minute : ℝ) 
  (h1 : tank_capacity = 1440) 
  (h2 : leak_time = 3) 
  (h3 : inlet_rate_per_minute = 6) : 
  (tank_capacity / (tank_capacity / leak_time - inlet_rate_per_minute * 60)) = 12 :=
by
  sorry

#check tank_emptying_time

end tank_emptying_time_l2332_233233


namespace consecutive_integers_square_sum_l2332_233225

theorem consecutive_integers_square_sum : ∃ (a : ℕ), 
  (a > 0) ∧ 
  ((a - 1) * a * (a + 1) = 8 * (3 * a)) ∧ 
  ((a - 1)^2 + a^2 + (a + 1)^2 = 77) := by
  sorry

end consecutive_integers_square_sum_l2332_233225


namespace replaced_man_age_l2332_233226

theorem replaced_man_age (n : ℕ) (avg_increase : ℝ) (man1_age : ℕ) (women_avg_age : ℝ) :
  n = 10 ∧ 
  avg_increase = 6 ∧ 
  man1_age = 18 ∧ 
  women_avg_age = 50 → 
  ∃ (original_avg : ℝ) (man2_age : ℕ),
    n * (original_avg + avg_increase) = n * original_avg + 2 * women_avg_age - (man1_age + man2_age) ∧
    man2_age = 22 := by
  sorry

#check replaced_man_age

end replaced_man_age_l2332_233226


namespace round_trip_distance_l2332_233293

/-- Calculates the total distance of a round trip given the times for each leg and the average speed -/
theorem round_trip_distance (t1 t2 : ℚ) (avg_speed : ℚ) (h1 : t1 = 15/60) (h2 : t2 = 25/60) (h3 : avg_speed = 3) :
  (t1 + t2) * avg_speed = 2 := by
  sorry

#check round_trip_distance

end round_trip_distance_l2332_233293


namespace product_of_two_numbers_l2332_233279

theorem product_of_two_numbers (x y : ℝ) 
  (sum_condition : x + y = 24) 
  (sum_squares_condition : x^2 + y^2 = 400) : 
  x * y = 88 := by
  sorry

end product_of_two_numbers_l2332_233279


namespace x_squared_minus_y_squared_l2332_233265

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 16) (h2 : x - y = 2) : x^2 - y^2 = 32 := by
  sorry

end x_squared_minus_y_squared_l2332_233265


namespace chips_price_calculation_l2332_233263

/-- Given a discount and a final price, calculate the original price --/
def original_price (discount : ℝ) (final_price : ℝ) : ℝ :=
  discount + final_price

theorem chips_price_calculation :
  let discount : ℝ := 17
  let final_price : ℝ := 18
  original_price discount final_price = 35 := by
sorry

end chips_price_calculation_l2332_233263


namespace gumballs_last_days_l2332_233250

def gumballs_per_earring : ℕ := 9
def first_day_earrings : ℕ := 3
def second_day_earrings : ℕ := 2 * first_day_earrings
def third_day_earrings : ℕ := second_day_earrings - 1
def daily_consumption : ℕ := 3

def total_earrings : ℕ := first_day_earrings + second_day_earrings + third_day_earrings
def total_gumballs : ℕ := total_earrings * gumballs_per_earring

theorem gumballs_last_days : total_gumballs / daily_consumption = 42 := by
  sorry

end gumballs_last_days_l2332_233250


namespace triangle_properties_l2332_233210

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : (t.b^2 + t.c^2 - t.a^2) / Real.cos t.A = 2)
  (h2 : (t.a * Real.cos t.B - t.b * Real.cos t.A) / (t.a * Real.cos t.B + t.b * Real.cos t.A) - t.b / t.c = 1) :
  t.b * t.c = 1 ∧ 
  (1/2 : ℝ) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 4 := by
  sorry

end triangle_properties_l2332_233210


namespace lunch_cost_calculation_l2332_233244

/-- Calculates the total cost of lunch for all students in an elementary school --/
theorem lunch_cost_calculation (third_grade_classes : ℕ) (third_grade_students_per_class : ℕ)
  (fourth_grade_classes : ℕ) (fourth_grade_students_per_class : ℕ)
  (fifth_grade_classes : ℕ) (fifth_grade_students_per_class : ℕ)
  (hamburger_cost : ℚ) (carrots_cost : ℚ) (cookie_cost : ℚ) :
  third_grade_classes = 5 →
  third_grade_students_per_class = 30 →
  fourth_grade_classes = 4 →
  fourth_grade_students_per_class = 28 →
  fifth_grade_classes = 4 →
  fifth_grade_students_per_class = 27 →
  hamburger_cost = 2.1 →
  carrots_cost = 0.5 →
  cookie_cost = 0.2 →
  (third_grade_classes * third_grade_students_per_class +
   fourth_grade_classes * fourth_grade_students_per_class +
   fifth_grade_classes * fifth_grade_students_per_class) *
  (hamburger_cost + carrots_cost + cookie_cost) = 1036 :=
by sorry

end lunch_cost_calculation_l2332_233244


namespace unique_integer_value_of_expression_l2332_233261

theorem unique_integer_value_of_expression 
  (m n p : ℕ) 
  (hm : 2 ≤ m ∧ m ≤ 9) 
  (hn : 2 ≤ n ∧ n ≤ 9) 
  (hp : 2 ≤ p ∧ p ≤ 9) 
  (hdiff : m ≠ n ∧ m ≠ p ∧ n ≠ p) : 
  (∃ k : ℤ, (m + n + p : ℚ) / (m + n) = k) → (m + n + p : ℚ) / (m + n) = 1 :=
sorry

end unique_integer_value_of_expression_l2332_233261


namespace min_distance_parabola_point_l2332_233294

/-- The minimum distance from a point on the parabola to Q plus its y-coordinate -/
theorem min_distance_parabola_point (x y : ℝ) (h : x^2 = -4*y) :
  ∃ (min : ℝ), min = 2 ∧ 
  ∀ (x' y' : ℝ), x'^2 = -4*y' → 
    abs y + Real.sqrt ((x' + 2*Real.sqrt 2)^2 + y'^2) ≥ min :=
sorry

end min_distance_parabola_point_l2332_233294


namespace total_toes_on_bus_l2332_233282

/-- Represents a race of beings on Popton -/
inductive Race
| Hoopit
| Neglart

/-- Number of hands for each race -/
def hands (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 4
  | Race.Neglart => 5

/-- Number of toes per hand for each race -/
def toes_per_hand (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 3
  | Race.Neglart => 2

/-- Number of students of each race on the bus -/
def students (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 7
  | Race.Neglart => 8

/-- Total number of toes for a single being of a given race -/
def toes_per_being (r : Race) : ℕ :=
  hands r * toes_per_hand r

/-- Total number of toes for all students of a given race on the bus -/
def total_toes_per_race (r : Race) : ℕ :=
  students r * toes_per_being r

/-- Theorem: The total number of toes on the Popton school bus is 164 -/
theorem total_toes_on_bus :
  total_toes_per_race Race.Hoopit + total_toes_per_race Race.Neglart = 164 := by
  sorry

end total_toes_on_bus_l2332_233282


namespace smallest_four_digit_divisible_by_33_l2332_233290

theorem smallest_four_digit_divisible_by_33 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 33 = 0 → n ≥ 1023 := by
  sorry

end smallest_four_digit_divisible_by_33_l2332_233290


namespace consecutive_integers_around_sqrt_28_l2332_233255

theorem consecutive_integers_around_sqrt_28 (a b : ℤ) : 
  (b = a + 1) → (↑a < Real.sqrt 28 ∧ Real.sqrt 28 < ↑b) → a + b = 11 := by
  sorry

end consecutive_integers_around_sqrt_28_l2332_233255


namespace triangle_sine_ratio_l2332_233287

/-- Given a triangle ABC where the ratio of sines of angles is 5:7:8, 
    prove the ratio of sides and the measure of angle B -/
theorem triangle_sine_ratio (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sine_ratio : ∃ k : ℝ, k > 0 ∧ Real.sin A = 5*k ∧ Real.sin B = 7*k ∧ Real.sin C = 8*k) :
  (∃ m : ℝ, m > 0 ∧ a = 5*m ∧ b = 7*m ∧ c = 8*m) ∧ B = π/3 := by
  sorry

end triangle_sine_ratio_l2332_233287


namespace toy_car_growth_l2332_233236

theorem toy_car_growth (initial_count : ℕ) (growth_factor : ℚ) (final_multiplier : ℕ) : 
  initial_count = 50 → growth_factor = 5/2 → final_multiplier = 3 →
  (↑initial_count * growth_factor * ↑final_multiplier : ℚ) = 375 := by
  sorry

end toy_car_growth_l2332_233236


namespace sachins_age_l2332_233275

theorem sachins_age (sachin rahul : ℕ) 
  (age_difference : rahul = sachin + 4)
  (age_ratio : sachin * 9 = rahul * 7) : 
  sachin = 14 := by
  sorry

end sachins_age_l2332_233275


namespace sum_of_segments_equals_radius_l2332_233269

/-- A regular (4k+2)-gon inscribed in a circle -/
structure RegularPolygon (k : ℕ) where
  /-- The radius of the circumscribed circle -/
  R : ℝ
  /-- The center of the circle -/
  O : ℝ × ℝ
  /-- The vertices of the polygon -/
  vertices : Fin (4*k+2) → ℝ × ℝ
  /-- Condition that the polygon is regular and inscribed -/
  regular_inscribed : ∀ i : Fin (4*k+2), dist O (vertices i) = R

/-- The sum of segments cut by a central angle on diagonals -/
def sum_of_segments (p : RegularPolygon k) : ℝ := sorry

/-- Theorem: The sum of segments equals the radius -/
theorem sum_of_segments_equals_radius (k : ℕ) (p : RegularPolygon k) :
  sum_of_segments p = p.R := by sorry

end sum_of_segments_equals_radius_l2332_233269


namespace difference_of_squares_special_case_l2332_233296

theorem difference_of_squares_special_case : (527 : ℤ) * 527 - 526 * 528 = 1 := by
  sorry

end difference_of_squares_special_case_l2332_233296


namespace average_weight_calculation_l2332_233215

/-- Given the average weights of pairs of individuals and the weight of one individual,
    calculate the average weight of all three individuals. -/
theorem average_weight_calculation
  (avg_ab avg_bc b_weight : ℝ)
  (h_avg_ab : (a + b_weight) / 2 = avg_ab)
  (h_avg_bc : (b_weight + c) / 2 = avg_bc)
  (h_b : b_weight = 37)
  (a c : ℝ) :
  (a + b_weight + c) / 3 = 45 :=
by sorry


end average_weight_calculation_l2332_233215


namespace green_balls_count_l2332_233259

def bag_problem (blue_balls : ℕ) (prob_blue : ℚ) (red_balls : ℕ) (green_balls : ℕ) : Prop :=
  blue_balls = 10 ∧ 
  prob_blue = 2/7 ∧ 
  red_balls = 2 * blue_balls ∧
  prob_blue = blue_balls / (blue_balls + red_balls + green_balls)

theorem green_balls_count : 
  ∃ (green_balls : ℕ), bag_problem 10 (2/7) 20 green_balls → green_balls = 5 := by
  sorry

end green_balls_count_l2332_233259


namespace theta_range_l2332_233211

theorem theta_range (θ : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, x^2 * Real.cos θ - x*(1-x) + (1-x)^2 * Real.sin θ > 0) →
  ∃ k : ℤ, 2*k*Real.pi + Real.pi/12 < θ ∧ θ < 2*k*Real.pi + 5*Real.pi/12 :=
by sorry

end theta_range_l2332_233211


namespace function_range_l2332_233291

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (Real.exp x + x - a)

theorem function_range (a : ℝ) :
  (∃ y₀ : ℝ, y₀ ∈ Set.Icc (-1) 1 ∧ f a (f a y₀) = y₀) →
  a ∈ Set.Icc 1 (Real.exp 1) :=
by sorry

end function_range_l2332_233291


namespace cone_volume_l2332_233278

/-- The volume of a cone with given slant height and lateral surface angle -/
theorem cone_volume (s : ℝ) (θ : ℝ) (h : s = 6) (h' : θ = 2 * π / 3) :
  ∃ (v : ℝ), v = (16 * Real.sqrt 2 / 3) * π ∧ v = (1/3) * π * (s * θ / (2 * π))^2 * Real.sqrt (s^2 - (s * θ / (2 * π))^2) :=
by sorry

end cone_volume_l2332_233278


namespace ellipse_intersection_theorem_l2332_233203

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 12 + y^2 / 4 = 1

-- Define the line l
def line_l (x : ℝ) : Prop := x = -2 * Real.sqrt 2

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop := ellipse x y

-- Define a point on line l
def point_on_line_l (x y : ℝ) : Prop := line_l x

-- Define the perpendicular line l' through P
def line_l_prime (x y x_p y_p : ℝ) : Prop :=
  y - y_p = -(3 * y_p) / (2 * Real.sqrt 2) * (x - x_p)

theorem ellipse_intersection_theorem :
  ∀ (x_p y_p : ℝ),
    point_on_line_l x_p y_p →
    ∃ (x_m y_m x_n y_n : ℝ),
      point_on_ellipse x_m y_m ∧
      point_on_ellipse x_n y_n ∧
      (x_p - x_m)^2 + (y_p - y_m)^2 = (x_p - x_n)^2 + (y_p - y_n)^2 →
      line_l_prime (-4 * Real.sqrt 2 / 3) 0 x_p y_p :=
by sorry

end ellipse_intersection_theorem_l2332_233203


namespace subset_implies_a_leq_two_l2332_233260

def A : Set ℝ := {x | x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

theorem subset_implies_a_leq_two (a : ℝ) (h : A ⊆ B a) : a ≤ 2 := by
  sorry

end subset_implies_a_leq_two_l2332_233260


namespace three_Z_five_equals_eight_l2332_233242

-- Define the operation Z
def Z (a b : ℝ) : ℝ := b + 10 * a - 3 * a^2

-- Theorem to prove
theorem three_Z_five_equals_eight : Z 3 5 = 8 := by sorry

end three_Z_five_equals_eight_l2332_233242


namespace ferris_wheel_cost_l2332_233256

def tickets_bought : ℕ := 13
def tickets_left : ℕ := 4
def ticket_cost : ℕ := 9

theorem ferris_wheel_cost : (tickets_bought - tickets_left) * ticket_cost = 81 := by
  sorry

end ferris_wheel_cost_l2332_233256


namespace unique_quadratic_family_l2332_233208

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The property that the product of roots equals the sum of coefficients -/
def roots_product_equals_coeff_sum (p : QuadraticPolynomial) : Prop :=
  ∃ r s : ℝ, r * s = p.a + p.b + p.c ∧ p.a * r^2 + p.b * r + p.c = 0 ∧ p.a * s^2 + p.b * s + p.c = 0

/-- The theorem stating that there's exactly one family of quadratic polynomials satisfying the condition -/
theorem unique_quadratic_family :
  ∃! f : ℝ → QuadraticPolynomial,
    (∀ c : ℝ, (f c).a = 1 ∧ (f c).b = -1 ∧ (f c).c = c) ∧
    (∀ p : QuadraticPolynomial, roots_product_equals_coeff_sum p ↔ ∃ c : ℝ, p = f c) :=
  sorry

end unique_quadratic_family_l2332_233208


namespace percentage_without_fulltime_jobs_is_19_l2332_233264

/-- The percentage of parents who do not hold full-time jobs -/
def percentage_without_fulltime_jobs (total_parents : ℕ) 
  (mother_ratio : ℚ) (father_ratio : ℚ) (women_ratio : ℚ) : ℚ :=
  let mothers := (women_ratio * total_parents).floor
  let fathers := total_parents - mothers
  let mothers_with_jobs := (mother_ratio * mothers).floor
  let fathers_with_jobs := (father_ratio * fathers).floor
  let parents_without_jobs := total_parents - mothers_with_jobs - fathers_with_jobs
  (parents_without_jobs : ℚ) / total_parents * 100

/-- Theorem stating that given the conditions in the problem, 
    the percentage of parents without full-time jobs is 19% -/
theorem percentage_without_fulltime_jobs_is_19 :
  ∀ n : ℕ, n > 0 → 
  percentage_without_fulltime_jobs n (9/10) (3/4) (2/5) = 19 := by
  sorry

end percentage_without_fulltime_jobs_is_19_l2332_233264


namespace budget_supplies_percent_l2332_233235

theorem budget_supplies_percent (salaries research_dev utilities equipment transportation : ℝ)
  (h1 : salaries = 60)
  (h2 : research_dev = 9)
  (h3 : utilities = 5)
  (h4 : equipment = 4)
  (h5 : transportation = 72 * 100 / 360)
  (h6 : salaries + research_dev + utilities + equipment + transportation < 100) :
  100 - (salaries + research_dev + utilities + equipment + transportation) = 2 := by
  sorry

end budget_supplies_percent_l2332_233235


namespace rabbit_turning_point_theorem_l2332_233266

/-- The point where the rabbit starts moving away from the fox -/
def rabbit_turning_point : ℝ × ℝ := (2.8, 5.6)

/-- The location of the fox -/
def fox_location : ℝ × ℝ := (10, 8)

/-- The slope of the rabbit's path -/
def rabbit_path_slope : ℝ := -3

/-- The y-intercept of the rabbit's path -/
def rabbit_path_intercept : ℝ := 14

/-- The equation of the rabbit's path: y = mx + b -/
def rabbit_path (x : ℝ) : ℝ := rabbit_path_slope * x + rabbit_path_intercept

theorem rabbit_turning_point_theorem :
  let (c, d) := rabbit_turning_point
  let (fx, fy) := fox_location
  -- The turning point lies on the rabbit's path
  d = rabbit_path c ∧
  -- The line from the fox to the turning point is perpendicular to the rabbit's path
  (d - fy) / (c - fx) = -1 / rabbit_path_slope := by
  sorry

end rabbit_turning_point_theorem_l2332_233266


namespace smallest_solution_congruence_l2332_233277

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 17 % 31 ∧ ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 17 % 31 → x ≤ y :=
by sorry

end smallest_solution_congruence_l2332_233277


namespace certain_event_good_product_l2332_233280

theorem certain_event_good_product (total : Nat) (good : Nat) (defective : Nat) (draw : Nat) :
  total = good + defective →
  good = 10 →
  defective = 2 →
  draw = 3 →
  Fintype.card {s : Finset (Fin total) // s.card = draw ∧ (∃ i ∈ s, i.val < good)} / Fintype.card {s : Finset (Fin total) // s.card = draw} = 1 :=
sorry

end certain_event_good_product_l2332_233280


namespace range_of_k_for_decreasing_proportional_function_l2332_233232

/-- A proportional function y = (k+4)x where y decreases as x increases -/
def decreasing_proportional_function (k : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → (k + 4) * x₁ > (k + 4) * x₂

/-- The range of k for a decreasing proportional function y = (k+4)x -/
theorem range_of_k_for_decreasing_proportional_function :
  ∀ k : ℝ, decreasing_proportional_function k → k < -4 :=
by
  sorry

end range_of_k_for_decreasing_proportional_function_l2332_233232


namespace equation_solution_l2332_233298

theorem equation_solution (n : ℤ) : n + (n + 1) + (n + 2) = 15 → n = 4 := by
  sorry

end equation_solution_l2332_233298


namespace fourth_root_of_256_l2332_233224

theorem fourth_root_of_256 (m : ℝ) : (256 : ℝ) ^ (1/4) = 4^m → m = 1 := by
  sorry

end fourth_root_of_256_l2332_233224


namespace large_coin_equivalent_mass_l2332_233209

theorem large_coin_equivalent_mass (large_coin_mass : ℝ) (pound_coin_mass : ℝ) :
  large_coin_mass = 100000 →
  pound_coin_mass = 10 →
  (large_coin_mass / pound_coin_mass : ℝ) = 10000 := by
  sorry

end large_coin_equivalent_mass_l2332_233209


namespace right_triangle_hypotenuse_l2332_233273

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 24 →
  a^2 + b^2 + c^2 = 2500 →
  a^2 + b^2 = c^2 →
  c = 25 * Real.sqrt 2 := by
  sorry

end right_triangle_hypotenuse_l2332_233273
