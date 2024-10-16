import Mathlib

namespace NUMINAMATH_CALUDE_union_and_complement_l2882_288204

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {2, 4, 5}
def B : Set Nat := {2, 7}

theorem union_and_complement :
  (A ∪ B = {2, 4, 5, 7}) ∧ (Aᶜ = {1, 3, 6, 7}) := by
  sorry

end NUMINAMATH_CALUDE_union_and_complement_l2882_288204


namespace NUMINAMATH_CALUDE_expansion_equality_l2882_288238

-- Define a positive integer n
variable (n : ℕ+)

-- Define the condition that the coefficient of x^3 is the same in both expansions
def coefficient_equality (n : ℕ+) : Prop :=
  (Nat.choose (2 * n) 3) = 2 * (Nat.choose n 1)

-- Theorem statement
theorem expansion_equality (n : ℕ+) (h : coefficient_equality n) :
  n = 2 ∧ 
  ∀ k : ℕ, k ≤ n → 2 * (Nat.choose n k) ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_expansion_equality_l2882_288238


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2882_288281

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  q > 1 →
  (4 * (a 2011)^2 - 8 * (a 2011) + 3 = 0) →
  (4 * (a 2012)^2 - 8 * (a 2012) + 3 = 0) →
  a 2013 + a 2014 = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2882_288281


namespace NUMINAMATH_CALUDE_water_in_bucket_l2882_288245

/-- The amount of water remaining in a bucket after pouring out some water. -/
def water_remaining (initial : ℝ) (poured_out : ℝ) : ℝ :=
  initial - poured_out

/-- Theorem stating that given 0.8 gallon initially and 0.2 gallon poured out, 
    the remaining water is 0.6 gallon. -/
theorem water_in_bucket : water_remaining 0.8 0.2 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_water_in_bucket_l2882_288245


namespace NUMINAMATH_CALUDE_ratio_sum_squares_l2882_288289

theorem ratio_sum_squares : 
  ∀ (x y z : ℝ), 
    y = 2 * x → 
    z = 3 * x → 
    x + y + z = 12 → 
    x^2 + y^2 + z^2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_squares_l2882_288289


namespace NUMINAMATH_CALUDE_subset_inequality_l2882_288205

-- Define the set S_n
def S_n (n : ℕ) : Set ℕ := {i | 1 ≤ i ∧ i ≤ n}

-- Define the properties of function f
def is_valid_f (n : ℕ) (f : Set ℕ → ℝ) : Prop :=
  (∀ A : Set ℕ, A ⊆ S_n n → f A > 0) ∧
  (∀ A : Set ℕ, ∀ x y : ℕ, A ⊆ S_n n → x ∈ S_n n → y ∈ S_n n → x ≠ y →
    f (A ∪ {x}) * f (A ∪ {y}) ≤ f (A ∪ {x, y}) * f A)

-- State the theorem
theorem subset_inequality (n : ℕ) (f : Set ℕ → ℝ) (h : is_valid_f n f) :
  ∀ A B : Set ℕ, A ⊆ S_n n → B ⊆ S_n n →
    f A * f B ≤ f (A ∪ B) * f (A ∩ B) :=
sorry

end NUMINAMATH_CALUDE_subset_inequality_l2882_288205


namespace NUMINAMATH_CALUDE_sum_of_53_odd_numbers_l2882_288230

theorem sum_of_53_odd_numbers : 
  (Finset.range 53).sum (fun n => 2 * n + 1) = 2809 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_53_odd_numbers_l2882_288230


namespace NUMINAMATH_CALUDE_lateral_edge_angle_l2882_288247

/-- Regular quadrilateral pyramid with a specific cross-section -/
structure RegularPyramid where
  -- Base edge length
  a : ℝ
  -- Angle between lateral edge and base plane
  θ : ℝ

/-- Properties of the cross-section -/
def CrossSection (p : RegularPyramid) : Prop :=
  -- The plane passes through a vertex of the base
  -- The plane intersects the opposite lateral edge at a right angle
  -- The area of the cross-section is half the area of the base
  ∃ (crossSectionArea baseArea : ℝ),
    crossSectionArea = baseArea / 2 ∧
    baseArea = 4 * p.a^2

/-- Theorem: Angle between lateral edge and base plane -/
theorem lateral_edge_angle (p : RegularPyramid) 
    (h : CrossSection p) : 
    p.θ = Real.arcsin ((1 + Real.sqrt 33) / 8) := by
  sorry

end NUMINAMATH_CALUDE_lateral_edge_angle_l2882_288247


namespace NUMINAMATH_CALUDE_racket_sales_total_l2882_288236

/-- The total amount for which rackets were sold, given the average price per pair and the number of pairs sold. -/
theorem racket_sales_total (avg_price : ℝ) (num_pairs : ℕ) (h1 : avg_price = 9.8) (h2 : num_pairs = 70) :
  avg_price * (num_pairs : ℝ) = 686 := by
  sorry

end NUMINAMATH_CALUDE_racket_sales_total_l2882_288236


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2882_288224

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = (1 : ℝ) / 4 →
  a 3 * a 5 = 4 * (a 4 - 1) →
  a 2 = (1 : ℝ) / 2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l2882_288224


namespace NUMINAMATH_CALUDE_team_division_probabilities_l2882_288219

/-- The total number of teams -/
def total_teams : ℕ := 8

/-- The number of weak teams -/
def weak_teams : ℕ := 3

/-- The number of teams in each group -/
def group_size : ℕ := 4

/-- The probability that one group has exactly two weak teams -/
def prob_two_weak : ℚ := 6/7

/-- The probability that group A has at least two weak teams -/
def prob_A_at_least_two : ℚ := 1/2

/-- Theorem stating the probabilities for the team division problem -/
theorem team_division_probabilities :
  (prob_two_weak = 6/7) ∧ (prob_A_at_least_two = 1/2) := by sorry

end NUMINAMATH_CALUDE_team_division_probabilities_l2882_288219


namespace NUMINAMATH_CALUDE_f_properties_l2882_288286

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x ≥ 0 then Real.log (x^2 - 2*x + 2)
  else Real.log (x^2 + 2*x + 2)

-- State the theorem
theorem f_properties :
  (∀ x : ℝ, f x = f (-x)) ∧  -- f is even
  (∀ x : ℝ, x < 0 → f x = Real.log (x^2 + 2*x + 2)) ∧  -- expression for x < 0
  (StrictMonoOn f (Set.Ioo (-1) 0) ∧ StrictMonoOn f (Set.Ioi 1)) := by
  sorry


end NUMINAMATH_CALUDE_f_properties_l2882_288286


namespace NUMINAMATH_CALUDE_plastic_for_rulers_l2882_288221

theorem plastic_for_rulers (plastic_per_ruler : ℕ) (rulers_made : ℕ) : 
  plastic_per_ruler = 8 → rulers_made = 103 → plastic_per_ruler * rulers_made = 824 := by
  sorry

end NUMINAMATH_CALUDE_plastic_for_rulers_l2882_288221


namespace NUMINAMATH_CALUDE_pen_cost_calculation_l2882_288283

/-- Calculates the cost of each pen given the initial amount, notebook cost, and remaining amount --/
theorem pen_cost_calculation (initial_amount : ℚ) (notebook_cost : ℚ) (num_notebooks : ℕ) 
  (num_pens : ℕ) (remaining_amount : ℚ) : 
  initial_amount = 15 → 
  notebook_cost = 4 → 
  num_notebooks = 2 → 
  num_pens = 2 → 
  remaining_amount = 4 → 
  (initial_amount - remaining_amount - (num_notebooks : ℚ) * notebook_cost) / (num_pens : ℚ) = 1.5 := by
  sorry

#eval (15 : ℚ) - 4 - 2 * 4

#eval ((15 : ℚ) - 4 - 2 * 4) / 2

end NUMINAMATH_CALUDE_pen_cost_calculation_l2882_288283


namespace NUMINAMATH_CALUDE_subtract_correction_l2882_288287

theorem subtract_correction (x : ℤ) (h : x - 42 = 50) : x - 24 = 68 := by
  sorry

end NUMINAMATH_CALUDE_subtract_correction_l2882_288287


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2882_288240

theorem polynomial_factorization (a : ℝ) : a^3 + a^2 - a - 1 = (a - 1) * (a + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2882_288240


namespace NUMINAMATH_CALUDE_min_M_n_value_l2882_288282

def M_n (n k : ℕ+) : ℚ :=
  max (40 / n) (max (80 / (k * n)) (60 / (200 - n - k * n)))

theorem min_M_n_value :
  ∀ k : ℕ+, (∃ n : ℕ+, n + k * n ≤ 200) →
    (∀ n : ℕ+, n + k * n ≤ 200 → M_n n k ≥ 10/11) ∧
    (∃ n : ℕ+, n + k * n ≤ 200 ∧ M_n n k = 10/11) :=
by sorry

end NUMINAMATH_CALUDE_min_M_n_value_l2882_288282


namespace NUMINAMATH_CALUDE_min_abs_z_min_abs_z_achievable_l2882_288206

open Complex

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 5*I) + Complex.abs (z - 6) = 7) : 
  Complex.abs z ≥ 30 / Real.sqrt 61 := by
  sorry

theorem min_abs_z_achievable : ∃ z : ℂ, 
  (Complex.abs (z - 5*I) + Complex.abs (z - 6) = 7) ∧ 
  (Complex.abs z = 30 / Real.sqrt 61) := by
  sorry

end NUMINAMATH_CALUDE_min_abs_z_min_abs_z_achievable_l2882_288206


namespace NUMINAMATH_CALUDE_unique_five_digit_numbers_l2882_288297

/-- Represents a five-digit number -/
def FiveDigitNumber := { n : ℕ // 10000 ≤ n ∧ n < 100000 }

/-- Checks if a number starts with a specific digit -/
def starts_with (n : FiveDigitNumber) (d : ℕ) : Prop :=
  n.val / 10000 = d

/-- Moves the first digit of a number to the last position -/
def move_first_to_last (n : FiveDigitNumber) : ℕ :=
  (n.val % 10000) * 10 + (n.val / 10000)

/-- The main theorem stating the unique solution to the problem -/
theorem unique_five_digit_numbers :
  ∃! (n₁ n₂ : FiveDigitNumber),
    starts_with n₁ 2 ∧
    starts_with n₂ 4 ∧
    move_first_to_last n₁ = n₁.val + n₂.val ∧
    move_first_to_last n₂ = n₁.val - n₂.val ∧
    n₁.val = 26829 ∧
    n₂.val = 41463 := by
  sorry


end NUMINAMATH_CALUDE_unique_five_digit_numbers_l2882_288297


namespace NUMINAMATH_CALUDE_double_average_l2882_288211

theorem double_average (n : ℕ) (initial_avg : ℚ) (new_avg : ℚ) :
  n = 11 →
  initial_avg = 36 →
  new_avg = 2 * initial_avg →
  new_avg = 72 :=
by sorry

end NUMINAMATH_CALUDE_double_average_l2882_288211


namespace NUMINAMATH_CALUDE_average_equals_x_l2882_288259

theorem average_equals_x (x : ℝ) : 
  (2 + 5 + x + 14 + 15) / 5 = x → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_equals_x_l2882_288259


namespace NUMINAMATH_CALUDE_exam_type_a_time_l2882_288217

/-- Represents the examination setup -/
structure Exam where
  totalTime : ℕ  -- Total time in minutes
  totalQuestions : ℕ
  typeAQuestions : ℕ
  typeAMultiplier : ℕ  -- How many times longer type A questions take compared to type B

/-- Calculates the time spent on type A problems -/
def timeOnTypeA (e : Exam) : ℚ :=
  let totalTypeB := e.totalQuestions - e.typeAQuestions
  let x := e.totalTime / (e.typeAQuestions * e.typeAMultiplier + totalTypeB)
  e.typeAQuestions * e.typeAMultiplier * x

/-- Theorem stating that for the given exam setup, 40 minutes should be spent on type A problems -/
theorem exam_type_a_time :
  let e : Exam := {
    totalTime := 180,  -- 3 hours * 60 minutes
    totalQuestions := 200,
    typeAQuestions := 25,
    typeAMultiplier := 2
  }
  timeOnTypeA e = 40 := by sorry


end NUMINAMATH_CALUDE_exam_type_a_time_l2882_288217


namespace NUMINAMATH_CALUDE_math_marks_calculation_l2882_288267

theorem math_marks_calculation (english physics chemistry biology : ℕ)
  (average : ℕ) (total_subjects : ℕ) (h1 : english = 73)
  (h2 : physics = 92) (h3 : chemistry = 64) (h4 : biology = 82)
  (h5 : average = 76) (h6 : total_subjects = 5) :
  average * total_subjects - (english + physics + chemistry + biology) = 69 :=
by sorry

end NUMINAMATH_CALUDE_math_marks_calculation_l2882_288267


namespace NUMINAMATH_CALUDE_range_of_a_l2882_288285

theorem range_of_a (x a : ℝ) : 
  (∀ x, (-3 ≤ x ∧ x ≤ 3) ↔ x < a) →
  a ∈ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2882_288285


namespace NUMINAMATH_CALUDE_quadratic_no_fixed_points_l2882_288228

/-- A quadratic function f(x) = x^2 + ax + 1 has no fixed points if and only if -1 < a < 3 -/
theorem quadratic_no_fixed_points (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 1 ≠ x) ↔ -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_fixed_points_l2882_288228


namespace NUMINAMATH_CALUDE_triangle_angle_B_l2882_288266

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) : 
  a = Real.sqrt 3 →
  b = Real.sqrt 6 →
  A = π / 6 →
  (B = π / 4 ∨ B = 3 * π / 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l2882_288266


namespace NUMINAMATH_CALUDE_ultra_squarish_exists_l2882_288261

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

def digits_nonzero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0

def first_three_digits (n : ℕ) : ℕ :=
  (n / 10000) % 1000

def middle_two_digits (n : ℕ) : ℕ :=
  (n / 100) % 100

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

theorem ultra_squarish_exists :
  ∃ M : ℕ,
    1000000 ≤ M ∧ M < 10000000 ∧
    is_perfect_square M ∧
    digits_nonzero M ∧
    is_perfect_square (first_three_digits M) ∧
    is_perfect_square (middle_two_digits M) ∧
    is_perfect_square (last_two_digits M) :=
  sorry

end NUMINAMATH_CALUDE_ultra_squarish_exists_l2882_288261


namespace NUMINAMATH_CALUDE_equation_positive_root_implies_m_eq_neg_one_l2882_288271

-- Define the equation
def equation (x m : ℝ) : Prop :=
  x / (x - 1) - m / (1 - x) = 2

-- State the theorem
theorem equation_positive_root_implies_m_eq_neg_one :
  ∃ (x : ℝ), x > 0 ∧ equation x m → m = -1 :=
sorry

end NUMINAMATH_CALUDE_equation_positive_root_implies_m_eq_neg_one_l2882_288271


namespace NUMINAMATH_CALUDE_triangle_is_acute_l2882_288250

theorem triangle_is_acute (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C)
  (h_ratio : (Real.sin A + Real.sin B) / (Real.sin B + Real.sin C) = 9 / 11 ∧
             (Real.sin B + Real.sin C) / (Real.sin C + Real.sin A) = 11 / 10) :
  A < π/2 ∧ B < π/2 ∧ C < π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_acute_l2882_288250


namespace NUMINAMATH_CALUDE_roberto_outfits_l2882_288244

/-- The number of different outfits Roberto can put together -/
def number_of_outfits (trousers shirts jackets : ℕ) (incompatible_combinations : ℕ) : ℕ :=
  trousers * shirts * jackets - incompatible_combinations * shirts

/-- Theorem stating the number of outfits Roberto can put together -/
theorem roberto_outfits :
  let trousers : ℕ := 5
  let shirts : ℕ := 7
  let jackets : ℕ := 4
  let incompatible_combinations : ℕ := 1
  number_of_outfits trousers shirts jackets incompatible_combinations = 133 := by
sorry

end NUMINAMATH_CALUDE_roberto_outfits_l2882_288244


namespace NUMINAMATH_CALUDE_shoes_sales_goal_l2882_288207

/-- Given a monthly goal and the number of shoes sold in two weeks, 
    calculate the additional pairs needed to meet the goal -/
def additional_pairs_needed (monthly_goal : ℕ) (sold_week1 : ℕ) (sold_week2 : ℕ) : ℕ :=
  monthly_goal - (sold_week1 + sold_week2)

/-- Theorem: Given the specific values from the problem, 
    the additional pairs needed is 41 -/
theorem shoes_sales_goal :
  additional_pairs_needed 80 27 12 = 41 := by
  sorry

end NUMINAMATH_CALUDE_shoes_sales_goal_l2882_288207


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2882_288256

theorem contrapositive_equivalence (x : ℝ) :
  (x > 1 → x^2 > 1) ↔ (x^2 ≤ 1 → x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2882_288256


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_in_open_interval_l2882_288265

/-- A piecewise function f defined on ℝ -/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x < 1 then (3 - a) * x - 4 * a else Real.log x / Real.log a

/-- Theorem stating the range of a for which f is increasing on ℝ -/
theorem f_increasing_iff_a_in_open_interval :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_in_open_interval_l2882_288265


namespace NUMINAMATH_CALUDE_abc_product_is_one_l2882_288215

theorem abc_product_is_one (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_eq : a^2 + 1/b^2 = b^2 + 1/c^2 ∧ b^2 + 1/c^2 = c^2 + 1/a^2) :
  |a * b * c| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_is_one_l2882_288215


namespace NUMINAMATH_CALUDE_cryptarithmetic_solution_l2882_288262

def is_valid_assignment (w h i t e a r p c n : Nat) : Prop :=
  w < 10 ∧ h < 10 ∧ i < 10 ∧ t < 10 ∧ e < 10 ∧ a < 10 ∧ r < 10 ∧ p < 10 ∧ c < 10 ∧ n < 10 ∧
  w ≠ h ∧ w ≠ i ∧ w ≠ t ∧ w ≠ e ∧ w ≠ a ∧ w ≠ r ∧ w ≠ p ∧ w ≠ c ∧ w ≠ n ∧
  h ≠ i ∧ h ≠ t ∧ h ≠ e ∧ h ≠ a ∧ h ≠ r ∧ h ≠ p ∧ h ≠ c ∧ h ≠ n ∧
  i ≠ t ∧ i ≠ e ∧ i ≠ a ∧ i ≠ r ∧ i ≠ p ∧ i ≠ c ∧ i ≠ n ∧
  t ≠ e ∧ t ≠ a ∧ t ≠ r ∧ t ≠ p ∧ t ≠ c ∧ t ≠ n ∧
  e ≠ a ∧ e ≠ r ∧ e ≠ p ∧ e ≠ c ∧ e ≠ n ∧
  a ≠ r ∧ a ≠ p ∧ a ≠ c ∧ a ≠ n ∧
  r ≠ p ∧ r ≠ c ∧ r ≠ n ∧
  p ≠ c ∧ p ≠ n ∧
  c ≠ n

def white_plus_water_equals_picnic (w h i t e a r p c n : Nat) : Prop :=
  10000 * w + 1000 * h + 100 * i + 10 * t + e +
  10000 * w + 1000 * a + 100 * t + 10 * e + r =
  100000 * p + 10000 * i + 1000 * c + 100 * n + 10 * i + c

theorem cryptarithmetic_solution :
  ∃! (w h i t e a r p c n : Nat),
    is_valid_assignment w h i t e a r p c n ∧
    white_plus_water_equals_picnic w h i t e a r p c n ∧
    100000 * p + 10000 * i + 1000 * c + 100 * n + 10 * i + c = 169069 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithmetic_solution_l2882_288262


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2882_288227

theorem simple_interest_rate_calculation (principal : ℝ) (h : principal > 0) :
  let final_amount := (7 / 6 : ℝ) * principal
  let time := 4
  let interest := final_amount - principal
  let rate := (interest / (principal * time)) * 100
  rate = 100 / 24 := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2882_288227


namespace NUMINAMATH_CALUDE_initial_boarders_l2882_288252

theorem initial_boarders (initial_boarders day_students new_boarders : ℕ) : 
  initial_boarders > 0 →
  day_students > 0 →
  new_boarders = 44 →
  initial_boarders * 12 = day_students * 5 →
  (initial_boarders + new_boarders) * 2 = day_students * 1 →
  initial_boarders = 220 := by
sorry

end NUMINAMATH_CALUDE_initial_boarders_l2882_288252


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2882_288280

theorem no_integer_solutions : ¬∃ (x y z : ℤ), 
  (x^2 - 2*x*y + 3*y^2 - z^2 = 45) ∧ 
  (-x^2 + 5*y*z + 3*z^2 = 28) ∧ 
  (x^2 - x*y + 9*z^2 = 140) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2882_288280


namespace NUMINAMATH_CALUDE_cyclic_inequality_l2882_288257

def cyclic_system (n : ℕ) (p q : ℝ) (x y z : ℝ) : Prop :=
  y = x^n + p*x + q ∧ z = y^n + p*y + q ∧ x = z^n + p*z + q

theorem cyclic_inequality (n : ℕ) (p q : ℝ) (x y z : ℝ) 
  (h_sys : cyclic_system n p q x y z) 
  (h_n : n = 2 ∨ n = 2010) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) :
  x^2 * y + y^2 * z + z^2 * x ≥ x^2 * z + y^2 * x + z^2 * y := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l2882_288257


namespace NUMINAMATH_CALUDE_parallel_to_plane_not_always_parallel_l2882_288260

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and between a line and a plane
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_to_plane_not_always_parallel 
  (l m : Line) (α : Plane) : 
  ¬(∀ l m α, parallel_line_plane l α → parallel_line_plane m α → parallel_lines l m) :=
sorry

end NUMINAMATH_CALUDE_parallel_to_plane_not_always_parallel_l2882_288260


namespace NUMINAMATH_CALUDE_min_value_expression_l2882_288213

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y^2 * z = 64) :
  x^2 + 8*x*y + 8*y^2 + 4*z^2 ≥ 1536 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y^2 * z = 64 ∧ x^2 + 8*x*y + 8*y^2 + 4*z^2 = 1536 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2882_288213


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2882_288258

theorem min_value_quadratic (x : ℝ) : 
  ∃ (min_z : ℝ), min_z = 5 ∧ ∀ z : ℝ, z = 5*x^2 + 20*x + 25 → z ≥ min_z := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2882_288258


namespace NUMINAMATH_CALUDE_nth_equation_pattern_l2882_288295

theorem nth_equation_pattern (n : ℕ) : (n + 1)^2 - 1 = n * (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_pattern_l2882_288295


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l2882_288208

theorem pulley_centers_distance (r₁ r₂ contact_distance : ℝ) 
  (hr₁ : r₁ = 10) 
  (hr₂ : r₂ = 6) 
  (hd : contact_distance = 30) : 
  ∃ (center_distance : ℝ), center_distance = 2 * Real.sqrt 229 :=
by sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l2882_288208


namespace NUMINAMATH_CALUDE_combined_return_percentage_l2882_288214

theorem combined_return_percentage (investment1 investment2 return1 return2 : ℝ) :
  investment1 = 500 →
  investment2 = 1500 →
  return1 = 0.07 →
  return2 = 0.19 →
  (investment1 * return1 + investment2 * return2) / (investment1 + investment2) = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_combined_return_percentage_l2882_288214


namespace NUMINAMATH_CALUDE_nested_multiplication_l2882_288284

theorem nested_multiplication : 3 * (3 * (3 * (3 * (3 * (3 * 2) * 2) * 2) * 2) * 2) * 2 = 1458 := by
  sorry

end NUMINAMATH_CALUDE_nested_multiplication_l2882_288284


namespace NUMINAMATH_CALUDE_marbles_lost_l2882_288246

/-- Given that Josh had 16 marbles initially and now has 9 marbles, 
    prove that he lost 7 marbles. -/
theorem marbles_lost (initial : ℕ) (remaining : ℕ) (lost : ℕ) 
    (h1 : initial = 16) 
    (h2 : remaining = 9) 
    (h3 : lost = initial - remaining) : lost = 7 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_l2882_288246


namespace NUMINAMATH_CALUDE_max_balls_in_cube_l2882_288216

theorem max_balls_in_cube (cube_side : ℝ) (ball_radius : ℝ) : 
  cube_side = 8 → 
  ball_radius = 1.5 → 
  ⌊(cube_side^3) / ((4/3) * π * ball_radius^3)⌋ = 36 := by
  sorry

end NUMINAMATH_CALUDE_max_balls_in_cube_l2882_288216


namespace NUMINAMATH_CALUDE_rectangular_field_path_area_and_cost_l2882_288291

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def path_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

theorem rectangular_field_path_area_and_cost 
  (field_length field_width path_width cost_per_unit : ℝ) 
  (h1 : field_length = 85)
  (h2 : field_width = 55)
  (h3 : path_width = 2.5)
  (h4 : cost_per_unit = 2) :
  path_area field_length field_width path_width = 725 ∧ 
  path_cost (path_area field_length field_width path_width) cost_per_unit = 1450 := by
  sorry

#check rectangular_field_path_area_and_cost

end NUMINAMATH_CALUDE_rectangular_field_path_area_and_cost_l2882_288291


namespace NUMINAMATH_CALUDE_rosa_flowers_total_l2882_288255

theorem rosa_flowers_total (initial : ℝ) (gift : ℝ) (total : ℝ) 
    (h1 : initial = 67.5) 
    (h2 : gift = 90.75) 
    (h3 : total = initial + gift) : 
  total = 158.25 := by
sorry

end NUMINAMATH_CALUDE_rosa_flowers_total_l2882_288255


namespace NUMINAMATH_CALUDE_skew_iff_b_neq_4_l2882_288272

def line1 (b t : ℝ) : ℝ × ℝ × ℝ := (2 + 3*t, 3 + 4*t, b + 5*t)
def line2 (u : ℝ) : ℝ × ℝ × ℝ := (5 + 6*u, 2 + 3*u, 1 + 2*u)

def are_skew (b : ℝ) : Prop :=
  ∀ t u : ℝ, line1 b t ≠ line2 u

theorem skew_iff_b_neq_4 (b : ℝ) :
  are_skew b ↔ b ≠ 4 :=
sorry

end NUMINAMATH_CALUDE_skew_iff_b_neq_4_l2882_288272


namespace NUMINAMATH_CALUDE_complex_square_second_quadrant_l2882_288294

/-- Given that (1+2i)^2 = a+bi where a and b are real numbers,
    prove that the point P(a,b) lies in the second quadrant. -/
theorem complex_square_second_quadrant :
  ∃ (a b : ℝ), (1 + 2 * Complex.I) ^ 2 = a + b * Complex.I ∧
  a < 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_second_quadrant_l2882_288294


namespace NUMINAMATH_CALUDE_eleven_rays_max_regions_l2882_288277

/-- The maximum number of regions into which n rays can split a plane -/
def max_regions (n : ℕ) : ℕ := (n^2 - n + 2) / 2

/-- Theorem: 11 rays can split a plane into a maximum of 56 regions -/
theorem eleven_rays_max_regions : max_regions 11 = 56 := by
  sorry

end NUMINAMATH_CALUDE_eleven_rays_max_regions_l2882_288277


namespace NUMINAMATH_CALUDE_intersection_sum_l2882_288242

/-- Two circles intersecting at (1, 3) and (m, n) with centers on x - y - 2 = 0 --/
structure IntersectingCircles where
  m : ℝ
  n : ℝ
  centers_on_line : ∀ (x y : ℝ), (x - y - 2 = 0) → (∃ (r : ℝ), (x - 1)^2 + (y - 3)^2 = r^2 ∧ (x - m)^2 + (y - n)^2 = r^2)

/-- The sum of coordinates of the second intersection point is 4 --/
theorem intersection_sum (c : IntersectingCircles) : c.m + c.n = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l2882_288242


namespace NUMINAMATH_CALUDE_slope_range_for_intersecting_line_l2882_288237

/-- The range of possible slopes for a line passing through a given point and intersecting a line segment -/
theorem slope_range_for_intersecting_line (M P Q : ℝ × ℝ) :
  M = (-1, 2) →
  P = (-4, -1) →
  Q = (3, 0) →
  let slope_range := {k : ℝ | k ≤ -1/2 ∨ k ≥ 1}
  ∀ k : ℝ,
    (∃ (x y : ℝ), (k * (x - M.1) = y - M.2 ∧
      ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
        x = P.1 + t * (Q.1 - P.1) ∧
        y = P.2 + t * (Q.2 - P.2))) ↔
    k ∈ slope_range :=
by sorry

end NUMINAMATH_CALUDE_slope_range_for_intersecting_line_l2882_288237


namespace NUMINAMATH_CALUDE_sector_area_special_case_l2882_288231

/-- The area of a sector with arc length and central angle both equal to 5 is 5/2 -/
theorem sector_area_special_case :
  ∀ (l α : ℝ), l = 5 → α = 5 → (1/2) * l * (l / α) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_special_case_l2882_288231


namespace NUMINAMATH_CALUDE_min_value_theorem_l2882_288296

/-- The function f(x) defined as |x-a| + |x+b| -/
def f (a b x : ℝ) : ℝ := |x - a| + |x + b|

/-- The theorem stating the minimum value of (a^2/b + b^2/a) given conditions on f(x) -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f a b x ≥ 3) (hequal : ∃ x, f a b x = 3) :
  ∀ c d, c > 0 → d > 0 → c^2 / d + d^2 / c ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2882_288296


namespace NUMINAMATH_CALUDE_valid_triangle_l2882_288200

/-- A triangle with side lengths a, b, and c satisfies the triangle inequality theorem -/
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of numbers (2, 3, 4) forms a valid triangle -/
theorem valid_triangle : is_triangle 2 3 4 := by
  sorry

end NUMINAMATH_CALUDE_valid_triangle_l2882_288200


namespace NUMINAMATH_CALUDE_solve_equation_l2882_288274

theorem solve_equation (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2882_288274


namespace NUMINAMATH_CALUDE_fraction_cracked_pots_is_two_fifths_l2882_288218

/-- The fraction of cracked pots given the initial number of pots,
    the revenue from selling non-cracked pots, and the price per pot. -/
def fraction_cracked_pots (initial_pots : ℕ) (revenue : ℕ) (price_per_pot : ℕ) : ℚ :=
  1 - (revenue / (initial_pots * price_per_pot) : ℚ)

/-- Theorem stating that the fraction of cracked pots is 2/5 given the problem conditions. -/
theorem fraction_cracked_pots_is_two_fifths :
  fraction_cracked_pots 80 1920 40 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_cracked_pots_is_two_fifths_l2882_288218


namespace NUMINAMATH_CALUDE_g_of_3_eq_6_l2882_288292

/-- The function g(x) = x^3 - 3x^2 + 2x -/
def g (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

/-- Theorem: g(3) = 6 -/
theorem g_of_3_eq_6 : g 3 = 6 := by sorry

end NUMINAMATH_CALUDE_g_of_3_eq_6_l2882_288292


namespace NUMINAMATH_CALUDE_negation_equivalence_l2882_288279

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 - 2*x - 3 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x - 3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2882_288279


namespace NUMINAMATH_CALUDE_largest_number_of_three_l2882_288293

theorem largest_number_of_three (p q r : ℝ) 
  (sum_eq : p + q + r = 3)
  (sum_prod_eq : p * q + p * r + q * r = -8)
  (prod_eq : p * q * r = -20) :
  max p (max q r) = (-1 + Real.sqrt 21) / 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_of_three_l2882_288293


namespace NUMINAMATH_CALUDE_august_tips_fraction_l2882_288268

theorem august_tips_fraction (total_months : ℕ) (other_months : ℕ) (august_multiplier : ℕ) :
  total_months = other_months + 1 →
  august_multiplier = 8 →
  (august_multiplier : ℚ) / (august_multiplier + other_months : ℚ) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_august_tips_fraction_l2882_288268


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l2882_288288

theorem sum_of_x_and_y_is_two (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 28) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l2882_288288


namespace NUMINAMATH_CALUDE_polynomial_value_impossibility_l2882_288222

theorem polynomial_value_impossibility
  (P : ℤ → ℤ)  -- P is a function from integers to integers
  (h_poly : ∃ (Q : ℤ → ℤ), ∀ x, P x = Q x)  -- P is a polynomial
  (a b c d : ℤ)  -- a, b, c, d are integers
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)  -- a, b, c, d are distinct
  (h_values : P a = 5 ∧ P b = 5 ∧ P c = 5 ∧ P d = 5)  -- P(a) = P(b) = P(c) = P(d) = 5
  : ¬ ∃ (k : ℤ), P k = 8 :=  -- There is no integer k such that P(k) = 8
by sorry

end NUMINAMATH_CALUDE_polynomial_value_impossibility_l2882_288222


namespace NUMINAMATH_CALUDE_pages_left_to_read_l2882_288269

/-- Given a book with 400 pages, prove that after reading 20% of it, 320 pages are left to read. -/
theorem pages_left_to_read (total_pages : ℕ) (percentage_read : ℚ) 
  (h1 : total_pages = 400)
  (h2 : percentage_read = 20 / 100) : 
  total_pages - (total_pages * percentage_read).floor = 320 := by
  sorry

#eval (400 : ℕ) - ((400 : ℕ) * (20 / 100 : ℚ)).floor

end NUMINAMATH_CALUDE_pages_left_to_read_l2882_288269


namespace NUMINAMATH_CALUDE_maurice_previous_rides_l2882_288235

/-- Represents the horseback riding scenario of Maurice and Matt -/
structure RidingScenario where
  maurice_prev_rides : ℕ
  maurice_prev_horses : ℕ
  matt_total_horses : ℕ
  maurice_visit_rides : ℕ
  matt_rides_with_maurice : ℕ
  matt_solo_rides : ℕ
  matt_solo_horses : ℕ

/-- The specific scenario described in the problem -/
def problem_scenario : RidingScenario :=
  { maurice_prev_rides := 0,  -- to be determined
    maurice_prev_horses := 2,
    matt_total_horses := 4,
    maurice_visit_rides := 8,
    matt_rides_with_maurice := 8,
    matt_solo_rides := 16,
    matt_solo_horses := 2 }

/-- Theorem stating the number of Maurice's previous rides -/
theorem maurice_previous_rides (s : RidingScenario) :
  s.maurice_prev_horses = 2 ∧
  s.matt_total_horses = 4 ∧
  s.maurice_visit_rides = 8 ∧
  s.matt_rides_with_maurice = 8 ∧
  s.matt_solo_rides = 16 ∧
  s.matt_solo_horses = 2 ∧
  s.maurice_visit_rides = s.maurice_prev_rides ∧
  (s.matt_rides_with_maurice + s.matt_solo_rides) = 3 * s.maurice_prev_rides →
  s.maurice_prev_rides = 8 := by
  sorry

#check maurice_previous_rides problem_scenario

end NUMINAMATH_CALUDE_maurice_previous_rides_l2882_288235


namespace NUMINAMATH_CALUDE_square_difference_identity_l2882_288239

theorem square_difference_identity : (50 + 12)^2 - (12^2 + 50^2) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l2882_288239


namespace NUMINAMATH_CALUDE_chord_length_line_ellipse_intersection_l2882_288299

/-- The length of the chord formed by the intersection of a line and an ellipse -/
theorem chord_length_line_ellipse_intersection :
  let line : ℝ → ℝ × ℝ := λ t ↦ (1 + t, -2 + t)
  let ellipse : ℝ × ℝ → Prop := λ p ↦ p.1^2 + 2*p.2^2 = 8
  ∃ A B : ℝ × ℝ, A ≠ B ∧ 
    (∃ t₁, line t₁ = A) ∧ 
    (∃ t₂, line t₂ = B) ∧
    ellipse A ∧ 
    ellipse B ∧
    dist A B = 4 * Real.sqrt 3 / 3 :=
by sorry


end NUMINAMATH_CALUDE_chord_length_line_ellipse_intersection_l2882_288299


namespace NUMINAMATH_CALUDE_unique_double_rectangle_with_perimeter_72_l2882_288251

/-- A rectangle with integer dimensions where one side is twice the other. -/
structure DoubleRectangle where
  shorter : ℕ
  longer : ℕ
  longer_is_double : longer = 2 * shorter

/-- The perimeter of a DoubleRectangle. -/
def perimeter (r : DoubleRectangle) : ℕ := 2 * (r.shorter + r.longer)

/-- The set of all DoubleRectangles with a perimeter of 72 inches. -/
def rectangles_with_perimeter_72 : Set DoubleRectangle :=
  {r : DoubleRectangle | perimeter r = 72}

theorem unique_double_rectangle_with_perimeter_72 :
  ∃! (r : DoubleRectangle), r ∈ rectangles_with_perimeter_72 := by
  sorry

#check unique_double_rectangle_with_perimeter_72

end NUMINAMATH_CALUDE_unique_double_rectangle_with_perimeter_72_l2882_288251


namespace NUMINAMATH_CALUDE_gcd_228_1995_l2882_288212

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l2882_288212


namespace NUMINAMATH_CALUDE_largest_k_inequality_l2882_288270

theorem largest_k_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  (∃ k : ℕ+, k = 4 ∧
    (∀ m : ℕ+, (1 / (a - b) + 1 / (b - c) ≥ (m : ℝ) / (a - c)) → m ≤ k) ∧
    (1 / (a - b) + 1 / (b - c) ≥ (k : ℝ) / (a - c))) :=
sorry

end NUMINAMATH_CALUDE_largest_k_inequality_l2882_288270


namespace NUMINAMATH_CALUDE_sparse_characterization_l2882_288241

/-- A number s grows to r if there exists some integer n > 0 such that s^n = r -/
def GrowsTo (s r : ℝ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ s^n = r

/-- A real number r is sparse if there are only finitely many real numbers s that grow to r -/
def Sparse (r : ℝ) : Prop :=
  Set.Finite {s : ℝ | GrowsTo s r}

/-- The characterization of sparse real numbers -/
theorem sparse_characterization (r : ℝ) : Sparse r ↔ r = -1 ∨ r = 0 ∨ r = 1 := by
  sorry

end NUMINAMATH_CALUDE_sparse_characterization_l2882_288241


namespace NUMINAMATH_CALUDE_square_side_bounds_l2882_288201

/-- A triangle with an inscribed square and circle -/
structure TriangleWithInscriptions where
  /-- The side length of the inscribed square -/
  s : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The square is inscribed such that two vertices are on the base and two on the sides -/
  square_inscribed : True
  /-- The circle is inscribed in the triangle -/
  circle_inscribed : True
  /-- Both s and r are positive -/
  s_pos : 0 < s
  r_pos : 0 < r

/-- The side of the inscribed square is bounded by √2r and 2r -/
theorem square_side_bounds (t : TriangleWithInscriptions) : Real.sqrt 2 * t.r < t.s ∧ t.s < 2 * t.r := by
  sorry

end NUMINAMATH_CALUDE_square_side_bounds_l2882_288201


namespace NUMINAMATH_CALUDE_complex_product_theorem_l2882_288263

theorem complex_product_theorem : 
  let z₁ : ℂ := 2 + Complex.I
  let z₂ : ℂ := 1 - Complex.I
  z₁ * z₂ = 3 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l2882_288263


namespace NUMINAMATH_CALUDE_tuesday_rainfall_l2882_288273

/-- Rainfall recorded over three days -/
def total_rainfall : ℝ := 0.67

/-- Rainfall recorded on Monday -/
def monday_rainfall : ℝ := 0.17

/-- Rainfall recorded on Wednesday -/
def wednesday_rainfall : ℝ := 0.08

/-- Theorem stating that the rainfall on Tuesday is 0.42 cm -/
theorem tuesday_rainfall : 
  total_rainfall - (monday_rainfall + wednesday_rainfall) = 0.42 := by sorry

end NUMINAMATH_CALUDE_tuesday_rainfall_l2882_288273


namespace NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l2882_288233

-- Define the set A
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define the set B
def B : Set ℝ := {x | x * (x - 3) < 0}

-- Define the result set
def result : Set ℝ := {x | x ≤ 2 ∨ x ≥ 3}

-- Theorem statement
theorem union_of_A_and_complement_of_B : A ∪ (Set.univ \ B) = result := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l2882_288233


namespace NUMINAMATH_CALUDE_sum_of_decimals_l2882_288202

theorem sum_of_decimals : 5.27 + 4.19 = 9.46 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l2882_288202


namespace NUMINAMATH_CALUDE_circle_perimeter_special_radius_l2882_288253

/-- The perimeter of a circle with radius 4 / π cm is 8 cm. -/
theorem circle_perimeter_special_radius :
  let r : ℝ := 4 / Real.pi
  2 * Real.pi * r = 8 := by sorry

end NUMINAMATH_CALUDE_circle_perimeter_special_radius_l2882_288253


namespace NUMINAMATH_CALUDE_square_area_PS_l2882_288276

-- Define the triangles and their properties
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  right_angle : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0

-- Define the configuration
def Configuration (P Q R S : ℝ × ℝ) : Prop :=
  Triangle P Q R ∧ Triangle P R S ∧
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 25 ∧
  (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = 4 ∧
  (R.1 - P.1)^2 + (R.2 - P.2)^2 = 49

-- State the theorem
theorem square_area_PS (P Q R S : ℝ × ℝ) 
  (h : Configuration P Q R S) : 
  (S.1 - P.1)^2 + (S.2 - P.2)^2 = 53 := by
  sorry

end NUMINAMATH_CALUDE_square_area_PS_l2882_288276


namespace NUMINAMATH_CALUDE_kevin_born_1984_l2882_288234

/-- The year of the first AMC 8 competition -/
def first_amc8_year : ℕ := 1988

/-- The year Kevin took the AMC 8 -/
def kevins_amc8_year : ℕ := first_amc8_year + 9

/-- Kevin's age when he took the AMC 8 -/
def kevins_age : ℕ := 13

/-- Kevin's birth year -/
def kevins_birth_year : ℕ := kevins_amc8_year - kevins_age

theorem kevin_born_1984 : kevins_birth_year = 1984 := by
  sorry

end NUMINAMATH_CALUDE_kevin_born_1984_l2882_288234


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l2882_288275

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem reflection_across_x_axis :
  let original : Point2D := { x := 2, y := 3 }
  reflectAcrossXAxis original = { x := 2, y := -3 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l2882_288275


namespace NUMINAMATH_CALUDE_third_circle_radius_l2882_288220

theorem third_circle_radius (r₁ r₂ r₃ : ℝ) : 
  r₁ = 21 → r₂ = 35 → 
  π * r₃^2 = π * r₂^2 - π * r₁^2 → 
  r₃ = 28 := by sorry

end NUMINAMATH_CALUDE_third_circle_radius_l2882_288220


namespace NUMINAMATH_CALUDE_income_percentage_l2882_288223

theorem income_percentage (juan tim mart : ℝ) 
  (h1 : tim = juan * 0.5) 
  (h2 : mart = tim * 1.6) : 
  mart = juan * 0.8 := by
  sorry

end NUMINAMATH_CALUDE_income_percentage_l2882_288223


namespace NUMINAMATH_CALUDE_gaussian_guardians_score_l2882_288264

/-- The total points scored by the Gaussian Guardians basketball team -/
def total_points (daniel curtis sid emily kalyn hyojeong ty winston : ℕ) : ℕ :=
  daniel + curtis + sid + emily + kalyn + hyojeong + ty + winston

/-- Theorem stating that the total points scored by the Gaussian Guardians is 54 -/
theorem gaussian_guardians_score :
  total_points 7 8 2 11 6 12 1 7 = 54 := by
  sorry

end NUMINAMATH_CALUDE_gaussian_guardians_score_l2882_288264


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l2882_288225

/-- Represents the number of handshakes involving coaches -/
def coach_handshakes (nA nB : ℕ) : ℕ := 
  620 - (nA.choose 2 + nB.choose 2 + nA * nB)

/-- The main theorem to prove -/
theorem min_coach_handshakes : 
  ∃ (nA nB : ℕ), 
    nA = nB + 2 ∧ 
    nA > 0 ∧ 
    nB > 0 ∧
    ∀ (mA mB : ℕ), 
      mA = mB + 2 → 
      mA > 0 → 
      mB > 0 → 
      coach_handshakes nA nB ≤ coach_handshakes mA mB ∧
      coach_handshakes nA nB = 189 :=
sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l2882_288225


namespace NUMINAMATH_CALUDE_min_value_expression_l2882_288232

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 + 1 / (a + b)^3 ≥ (4 : ℝ)^(1/3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2882_288232


namespace NUMINAMATH_CALUDE_exam_score_problem_l2882_288290

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℕ) (wrong_penalty : ℕ) (total_score : ℤ) :
  total_questions = 75 ∧ correct_score = 4 ∧ wrong_penalty = 1 ∧ total_score = 125 →
  ∃ (correct_answers : ℕ),
    correct_answers * correct_score - (total_questions - correct_answers) * wrong_penalty = total_score ∧
    correct_answers = 40 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_problem_l2882_288290


namespace NUMINAMATH_CALUDE_additive_function_is_scalar_multiple_l2882_288278

/-- A function from rationals to rationals satisfying the given additive property -/
def AdditiveFunction (f : ℚ → ℚ) : Prop :=
  ∀ x y : ℚ, f (x + y) = f x + f y

/-- The theorem stating that any additive function on rationals is a scalar multiple -/
theorem additive_function_is_scalar_multiple :
  ∀ f : ℚ → ℚ, AdditiveFunction f → ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
by sorry

end NUMINAMATH_CALUDE_additive_function_is_scalar_multiple_l2882_288278


namespace NUMINAMATH_CALUDE_race_track_distance_squared_l2882_288254

theorem race_track_distance_squared (inner_radius outer_radius : ℝ) 
  (h_inner : inner_radius = 11) 
  (h_outer : outer_radius = 12) 
  (separation_angle : ℝ) 
  (h_angle : separation_angle = 30 * π / 180) : 
  (inner_radius^2 + outer_radius^2 - 2 * inner_radius * outer_radius * Real.cos separation_angle) 
  = 265 - 132 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_race_track_distance_squared_l2882_288254


namespace NUMINAMATH_CALUDE_inequality_proof_l2882_288229

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2882_288229


namespace NUMINAMATH_CALUDE_net_folds_to_partial_cube_l2882_288203

/-- Represents a net that can be folded into a cube -/
structure Net where
  faces : Finset (Fin 6)
  edges : Finset (Fin 12)
  holes : Finset (Fin 12)

/-- Represents a partial cube -/
structure PartialCube where
  faces : Finset (Fin 6)
  edges : Finset (Fin 12)
  holes : Finset (Fin 12)

/-- A net can be folded into a partial cube -/
def canFoldInto (n : Net) (pc : PartialCube) : Prop :=
  n.faces = pc.faces ∧ n.edges = pc.edges ∧ n.holes = pc.holes

/-- The given partial cube has holes on the edges of four different faces -/
axiom partial_cube_property (pc : PartialCube) :
  ∃ (f1 f2 f3 f4 : Fin 6) (e1 e2 e3 e4 : Fin 12),
    f1 ≠ f2 ∧ f1 ≠ f3 ∧ f1 ≠ f4 ∧ f2 ≠ f3 ∧ f2 ≠ f4 ∧ f3 ≠ f4 ∧
    e1 ∈ pc.holes ∧ e2 ∈ pc.holes ∧ e3 ∈ pc.holes ∧ e4 ∈ pc.holes

/-- Theorem: A net can be folded into the given partial cube if and only if
    it has holes on the edges of four different faces -/
theorem net_folds_to_partial_cube (n : Net) (pc : PartialCube) :
  canFoldInto n pc ↔
    ∃ (f1 f2 f3 f4 : Fin 6) (e1 e2 e3 e4 : Fin 12),
      f1 ≠ f2 ∧ f1 ≠ f3 ∧ f1 ≠ f4 ∧ f2 ≠ f3 ∧ f2 ≠ f4 ∧ f3 ≠ f4 ∧
      e1 ∈ n.holes ∧ e2 ∈ n.holes ∧ e3 ∈ n.holes ∧ e4 ∈ n.holes :=
by sorry

end NUMINAMATH_CALUDE_net_folds_to_partial_cube_l2882_288203


namespace NUMINAMATH_CALUDE_tile_area_calculation_l2882_288226

/-- Given a rectangular room and tiles covering a fraction of it, calculate the area of each tile. -/
theorem tile_area_calculation (room_length room_width : ℝ) (num_tiles : ℕ) (fraction_covered : ℚ) :
  room_length = 12 →
  room_width = 20 →
  num_tiles = 40 →
  fraction_covered = 1/6 →
  (room_length * room_width * fraction_covered) / num_tiles = 1 := by
  sorry

end NUMINAMATH_CALUDE_tile_area_calculation_l2882_288226


namespace NUMINAMATH_CALUDE_initial_state_is_winning_starting_player_wins_starting_player_always_wins_l2882_288249

/-- Represents a pile of matches -/
structure Pile :=
  (count : Nat)

/-- Represents the game state -/
structure GameState :=
  (piles : List Pile)

/-- Checks if a game state is a winning position for the current player -/
def isWinningPosition (state : GameState) : Prop :=
  ∃ (n m : Nat), n < m ∧
    ∃ (a b c : Nat), 
      state.piles = [Pile.mk (2^n * a), Pile.mk (2^n * b), Pile.mk (2^m * c)] ∧
      Odd a ∧ Odd b ∧ Odd c

/-- The initial game state -/
def initialState : GameState :=
  { piles := [Pile.mk 100, Pile.mk 200, Pile.mk 300] }

/-- Theorem stating that the initial state is a winning position -/
theorem initial_state_is_winning : isWinningPosition initialState := by
  sorry

/-- Theorem stating that the starting player has a winning strategy -/
theorem starting_player_wins (state : GameState) :
  isWinningPosition state → ∃ (nextState : GameState), 
    (∃ (move : GameState → GameState), nextState = move state) ∧
    ¬isWinningPosition nextState := by
  sorry

/-- Main theorem: The starting player wins with correct play -/
theorem starting_player_always_wins : 
  ∃ (strategy : GameState → GameState), 
    ∀ (state : GameState), 
      isWinningPosition state → 
      ¬isWinningPosition (strategy state) := by
  sorry

end NUMINAMATH_CALUDE_initial_state_is_winning_starting_player_wins_starting_player_always_wins_l2882_288249


namespace NUMINAMATH_CALUDE_basketball_probability_l2882_288248

/-- The number of basketballs -/
def total_balls : ℕ := 8

/-- The number of new basketballs -/
def new_balls : ℕ := 4

/-- The number of old basketballs -/
def old_balls : ℕ := 4

/-- The number of balls selected in each training session -/
def selected_balls : ℕ := 2

/-- The probability of selecting exactly one new ball in the second training session -/
def prob_one_new_second : ℚ := 51 / 98

theorem basketball_probability :
  total_balls = new_balls + old_balls →
  prob_one_new_second = (
    (Nat.choose old_balls selected_balls * Nat.choose new_balls 1 * Nat.choose old_balls 1 +
     Nat.choose new_balls 1 * Nat.choose old_balls 1 * Nat.choose (new_balls - 1) 1 * Nat.choose (old_balls + 1) 1 +
     Nat.choose new_balls selected_balls * Nat.choose (new_balls - selected_balls) 1 * Nat.choose (old_balls + selected_balls) 1) /
    (Nat.choose total_balls selected_balls * Nat.choose total_balls selected_balls)
  ) := by sorry

end NUMINAMATH_CALUDE_basketball_probability_l2882_288248


namespace NUMINAMATH_CALUDE_tangent_line_at_one_one_l2882_288243

/-- The equation of the tangent line to y = x^2 at (1, 1) is 2x - y - 1 = 0 -/
theorem tangent_line_at_one_one :
  let f : ℝ → ℝ := λ x ↦ x^2
  let point : ℝ × ℝ := (1, 1)
  let tangent_line : ℝ → ℝ → Prop := λ x y ↦ 2*x - y - 1 = 0
  (∀ x, HasDerivAt f (2*x) x) →
  tangent_line point.1 point.2 ∧
  ∀ x y, tangent_line x y ↔ y - point.2 = (2 * point.1) * (x - point.1) :=
by
  sorry


end NUMINAMATH_CALUDE_tangent_line_at_one_one_l2882_288243


namespace NUMINAMATH_CALUDE_inverse_mod_53_l2882_288210

theorem inverse_mod_53 (h : (17⁻¹ : ZMod 53) = 23) : (36⁻¹ : ZMod 53) = 30 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_53_l2882_288210


namespace NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l2882_288298

-- Define a fair coin toss
def fair_coin_toss : Type := Bool

-- Define the number of tosses
def num_tosses : Nat := 8

-- Define the number of heads we're looking for
def target_heads : Nat := 3

-- Define the probability of getting exactly 'target_heads' in 'num_tosses'
def probability_exact_heads : ℚ :=
  (Nat.choose num_tosses target_heads : ℚ) / (2 ^ num_tosses : ℚ)

-- Theorem statement
theorem probability_three_heads_in_eight_tosses :
  probability_exact_heads = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l2882_288298


namespace NUMINAMATH_CALUDE_assignment_ways_l2882_288209

def total_students : ℕ := 30
def selected_students : ℕ := 10
def group_size : ℕ := 5

def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem assignment_ways :
  (combination total_students selected_students * combination selected_students group_size) / 2 =
  (combination total_students selected_students * combination selected_students group_size) / 2 := by
  sorry

end NUMINAMATH_CALUDE_assignment_ways_l2882_288209
