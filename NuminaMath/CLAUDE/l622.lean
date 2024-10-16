import Mathlib

namespace NUMINAMATH_CALUDE_share_ratio_a_to_b_l622_62295

/-- Prove that the ratio of A's share to B's share is 4:1 -/
theorem share_ratio_a_to_b (total amount : ℕ) (a_share b_share c_share : ℕ) :
  amount = 578 →
  b_share = c_share / 4 →
  a_share = 408 →
  b_share = 102 →
  c_share = 68 →
  a_share + b_share + c_share = amount →
  a_share / b_share = 4 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_a_to_b_l622_62295


namespace NUMINAMATH_CALUDE_cat_meow_ratio_l622_62255

/-- Given three cats meowing, prove the ratio of meows per minute of the third cat to the second cat -/
theorem cat_meow_ratio :
  ∀ (cat1_rate cat2_rate cat3_rate : ℚ),
  cat1_rate = 3 →
  cat2_rate = 2 * cat1_rate →
  5 * (cat1_rate + cat2_rate + cat3_rate) = 55 →
  cat3_rate / cat2_rate = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_cat_meow_ratio_l622_62255


namespace NUMINAMATH_CALUDE_tabitha_current_age_l622_62212

def tabitha_hair_colors (age : ℕ) : ℕ :=
  age - 13

theorem tabitha_current_age : 
  ∃ (current_age : ℕ), 
    tabitha_hair_colors current_age = 5 ∧ 
    tabitha_hair_colors (current_age + 3) = 8 ∧ 
    current_age = 18 := by
  sorry

end NUMINAMATH_CALUDE_tabitha_current_age_l622_62212


namespace NUMINAMATH_CALUDE_fair_coin_toss_probability_sum_l622_62256

/-- Represents a fair coin --/
structure FairCoin where
  prob_heads : ℚ
  fair : prob_heads = 1/2

/-- Calculates the probability of getting exactly k heads in n tosses --/
def binomial_probability (c : FairCoin) (n k : ℕ) : ℚ :=
  (n.choose k) * c.prob_heads^k * (1 - c.prob_heads)^(n-k)

/-- The main theorem --/
theorem fair_coin_toss_probability_sum :
  ∀ (c : FairCoin),
  (binomial_probability c 5 1 = binomial_probability c 5 2) →
  ∃ (i j : ℕ),
    (binomial_probability c 5 3 = i / j) ∧
    (∀ (a b : ℕ), (a / b = i / j) → (a ≤ i ∧ b ≤ j)) ∧
    i + j = 283 :=
sorry

end NUMINAMATH_CALUDE_fair_coin_toss_probability_sum_l622_62256


namespace NUMINAMATH_CALUDE_total_work_experience_approx_l622_62206

def daysPerYear : ℝ := 365
def daysPerMonth : ℝ := 30.44
def daysPerWeek : ℝ := 7

def bartenderYears : ℝ := 9
def bartenderMonths : ℝ := 8

def managerYears : ℝ := 3
def managerMonths : ℝ := 6

def salesMonths : ℝ := 11

def coordinatorYears : ℝ := 2
def coordinatorMonths : ℝ := 5
def coordinatorWeeks : ℝ := 3

def totalWorkExperience : ℝ :=
  (bartenderYears * daysPerYear + bartenderMonths * daysPerMonth) +
  (managerYears * daysPerYear + managerMonths * daysPerMonth) +
  (salesMonths * daysPerMonth) +
  (coordinatorYears * daysPerYear + coordinatorMonths * daysPerMonth + coordinatorWeeks * daysPerWeek)

theorem total_work_experience_approx :
  ⌊totalWorkExperience⌋ = 6044 := by sorry

end NUMINAMATH_CALUDE_total_work_experience_approx_l622_62206


namespace NUMINAMATH_CALUDE_unique_common_root_value_l622_62235

theorem unique_common_root_value (m : ℝ) : 
  m > 5 →
  (∃! x : ℝ, x^2 - 5*x + 6 = 0 ∧ x^2 + 2*x - 2*m + 1 = 0) →
  m = 8 := by
sorry

end NUMINAMATH_CALUDE_unique_common_root_value_l622_62235


namespace NUMINAMATH_CALUDE_repetend_of_five_seventeenths_l622_62217

/-- The decimal representation of 5/17 has a 6-digit repetend equal to 294117 -/
theorem repetend_of_five_seventeenths :
  ∃ (a b : ℕ), (5 : ℚ) / 17 = (a : ℚ) + (b : ℚ) / 999999 ∧ b = 294117 := by
  sorry

end NUMINAMATH_CALUDE_repetend_of_five_seventeenths_l622_62217


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l622_62258

theorem smallest_positive_solution :
  ∃! (x y : ℝ), x > 0 ∧ y > 0 ∧
  x^2 - 3*x + 2.5 = Real.sin y - 0.75 ∧
  (∀ (x' y' : ℝ), x' > 0 → y' > 0 → x'^2 - 3*x' + 2.5 = Real.sin y' - 0.75 → x ≤ x' ∧ y ≤ y') ∧
  x = 3/2 ∧ y = Real.pi/2 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l622_62258


namespace NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l622_62244

theorem cylinder_minus_cones_volume (r h₁ h₂ : ℝ) (hr : r = 10) (hh₁ : h₁ = 15) (hh₂ : h₂ = 30) :
  let v_cyl := π * r^2 * h₂
  let v_cone := (1/3) * π * r^2 * h₁
  v_cyl - 2 * v_cone = 2000 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l622_62244


namespace NUMINAMATH_CALUDE_delivery_fee_percentage_l622_62240

def toy_organizer_cost : ℝ := 78
def gaming_chair_cost : ℝ := 83
def toy_organizer_sets : ℕ := 3
def gaming_chairs : ℕ := 2
def total_paid : ℝ := 420

def total_before_fee : ℝ := toy_organizer_cost * toy_organizer_sets + gaming_chair_cost * gaming_chairs

def delivery_fee : ℝ := total_paid - total_before_fee

theorem delivery_fee_percentage : (delivery_fee / total_before_fee) * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_delivery_fee_percentage_l622_62240


namespace NUMINAMATH_CALUDE_E_parity_2021_2022_2023_l622_62262

def E : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 => E (n + 2) + E (n + 1) + E n

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem E_parity_2021_2022_2023 :
  is_even (E 2021) ∧ ¬is_even (E 2022) ∧ ¬is_even (E 2023) := by
  sorry

end NUMINAMATH_CALUDE_E_parity_2021_2022_2023_l622_62262


namespace NUMINAMATH_CALUDE_f_range_l622_62259

-- Define the function f
def f (x : ℝ) : ℝ := 3 * (x + 4)

-- State the theorem
theorem f_range :
  Set.range f = {y : ℝ | y < 18 ∨ y > 18} :=
by
  sorry


end NUMINAMATH_CALUDE_f_range_l622_62259


namespace NUMINAMATH_CALUDE_range_of_a_l622_62269

/-- The piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then 2^x + a else x + a^2

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → 
  (a ≤ -1 ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l622_62269


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l622_62266

/-- Given a geometric sequence {a_n} with a_1 = 1, a_2 = 2, and a_3 = 4, prove that a_6 = 32 -/
theorem geometric_sequence_sixth_term (a : ℕ → ℝ) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 2) 
  (h3 : a 3 = 4) 
  (h_geom : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n) : 
  a 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l622_62266


namespace NUMINAMATH_CALUDE_student_mistake_difference_l622_62218

theorem student_mistake_difference (n : ℚ) (h : n = 480) : 5/6 * n - 5/16 * n = 250 := by
  sorry

end NUMINAMATH_CALUDE_student_mistake_difference_l622_62218


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l622_62209

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {2, 4, 5, 7, 8}

theorem intersection_of_A_and_B : A ∩ B = {2, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l622_62209


namespace NUMINAMATH_CALUDE_quotient_problem_l622_62278

theorem quotient_problem (L S Q : ℕ) : 
  L - S = 1365 → 
  L = 1620 → 
  L = S * Q + 15 → 
  Q = 6 := by sorry

end NUMINAMATH_CALUDE_quotient_problem_l622_62278


namespace NUMINAMATH_CALUDE_division_result_l622_62243

theorem division_result : (3486 : ℝ) / 189 = 18.444444444444443 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l622_62243


namespace NUMINAMATH_CALUDE_parabola_properties_l622_62287

/-- A parabola with specific properties -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  h_a_pos : a > 0
  h_axis : b = 2 * a
  h_intercept : a * m^2 + b * m + c = 0
  h_m_bounds : 0 < m ∧ m < 1

/-- Theorem stating properties of the parabola -/
theorem parabola_properties (p : Parabola) :
  (4 * p.a + p.c > 0) ∧
  (∀ t : ℝ, p.a - p.b * t ≤ p.a * t^2 + p.b) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l622_62287


namespace NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_odd_terms_l622_62237

theorem largest_divisor_of_five_consecutive_odd_terms (n : ℕ) (h : Odd n) :
  (∃ (k : ℕ), (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) = 15 * k) ∧
  (∀ (m : ℕ), m > 15 → ∃ (l : ℕ), Odd l ∧
    ¬(∃ (k : ℕ), (l + 3) * (l + 5) * (l + 7) * (l + 9) * (l + 11) = m * k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_odd_terms_l622_62237


namespace NUMINAMATH_CALUDE_evaluate_expression_l622_62246

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 2 * y^x = 533 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l622_62246


namespace NUMINAMATH_CALUDE_product_of_divisors_equals_3_30_5_40_l622_62231

/-- The product of divisors function -/
def productOfDivisors (n : ℕ) : ℕ := sorry

/-- Theorem: If the product of all divisors of N equals 3^30 * 5^40, then N = 3^3 * 5^4 -/
theorem product_of_divisors_equals_3_30_5_40 (N : ℕ) :
  productOfDivisors N = 3^30 * 5^40 → N = 3^3 * 5^4 := by sorry

end NUMINAMATH_CALUDE_product_of_divisors_equals_3_30_5_40_l622_62231


namespace NUMINAMATH_CALUDE_p_shape_points_for_10cm_square_l622_62281

/-- Calculates the number of points on a "P" shape formed from a square --/
def count_points_on_p_shape (square_side_length : ℕ) : ℕ :=
  let points_per_side := square_side_length + 1
  let total_sides := 3
  let overlapping_vertices := 2
  points_per_side * total_sides - overlapping_vertices

/-- Theorem stating that a "P" shape formed from a 10 cm square has 31 points --/
theorem p_shape_points_for_10cm_square :
  count_points_on_p_shape 10 = 31 := by
  sorry

#eval count_points_on_p_shape 10  -- Should output 31

end NUMINAMATH_CALUDE_p_shape_points_for_10cm_square_l622_62281


namespace NUMINAMATH_CALUDE_average_words_per_page_l622_62280

/-- Proves that for a book with given specifications, the average number of words per page is 1250 --/
theorem average_words_per_page
  (sheets : ℕ)
  (total_words : ℕ)
  (pages_per_sheet : ℕ)
  (h1 : sheets = 12)
  (h2 : total_words = 240000)
  (h3 : pages_per_sheet = 16) :
  total_words / (sheets * pages_per_sheet) = 1250 :=
by sorry

end NUMINAMATH_CALUDE_average_words_per_page_l622_62280


namespace NUMINAMATH_CALUDE_games_given_to_neil_l622_62201

theorem games_given_to_neil (henry_initial : ℕ) (neil_initial : ℕ) (games_given : ℕ) : 
  henry_initial = 33 →
  neil_initial = 2 →
  henry_initial - games_given = 4 * (neil_initial + games_given) →
  games_given = 5 := by
sorry

end NUMINAMATH_CALUDE_games_given_to_neil_l622_62201


namespace NUMINAMATH_CALUDE_quadratic_root_bounds_l622_62291

theorem quadratic_root_bounds (a b : ℝ) (α β : ℝ) : 
  (α^2 + a*α + b = 0) → 
  (β^2 + a*β + b = 0) → 
  (∀ x, x^2 + a*x + b = 0 → x = α ∨ x = β) →
  (|α| < 2 ∧ |β| < 2 ↔ 2*|a| < 4 + b ∧ |b| < 4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_bounds_l622_62291


namespace NUMINAMATH_CALUDE_majka_numbers_unique_l622_62294

/-- A three-digit number with alternating odd-even-odd digits -/
structure FunnyNumber :=
  (hundreds : Nat) (tens : Nat) (ones : Nat)
  (hundreds_odd : Odd hundreds)
  (tens_even : Even tens)
  (ones_odd : Odd ones)
  (is_three_digit : 100 ≤ hundreds * 100 + tens * 10 + ones ∧ hundreds * 100 + tens * 10 + ones < 1000)

/-- A three-digit number with alternating even-odd-even digits -/
structure CheerfulNumber :=
  (hundreds : Nat) (tens : Nat) (ones : Nat)
  (hundreds_even : Even hundreds)
  (tens_odd : Odd tens)
  (ones_even : Even ones)
  (is_three_digit : 100 ≤ hundreds * 100 + tens * 10 + ones ∧ hundreds * 100 + tens * 10 + ones < 1000)

/-- Convert a FunnyNumber to a natural number -/
def FunnyNumber.toNat (n : FunnyNumber) : Nat :=
  n.hundreds * 100 + n.tens * 10 + n.ones

/-- Convert a CheerfulNumber to a natural number -/
def CheerfulNumber.toNat (n : CheerfulNumber) : Nat :=
  n.hundreds * 100 + n.tens * 10 + n.ones

/-- The main theorem stating the unique solution to Majka's problem -/
theorem majka_numbers_unique (f : FunnyNumber) (c : CheerfulNumber) 
  (sum_eq : f.toNat + c.toNat = 1617)
  (product_ends_40 : (f.toNat * c.toNat) % 100 = 40)
  (all_digits_different : f.hundreds ≠ f.tens ∧ f.hundreds ≠ f.ones ∧ f.tens ≠ f.ones ∧
                          c.hundreds ≠ c.tens ∧ c.hundreds ≠ c.ones ∧ c.tens ≠ c.ones ∧
                          f.hundreds ≠ c.hundreds ∧ f.hundreds ≠ c.tens ∧ f.hundreds ≠ c.ones ∧
                          f.tens ≠ c.hundreds ∧ f.tens ≠ c.tens ∧ f.tens ≠ c.ones ∧
                          f.ones ≠ c.hundreds ∧ f.ones ≠ c.tens ∧ f.ones ≠ c.ones)
  (all_digits_nonzero : f.hundreds ≠ 0 ∧ f.tens ≠ 0 ∧ f.ones ≠ 0 ∧
                        c.hundreds ≠ 0 ∧ c.tens ≠ 0 ∧ c.ones ≠ 0) :
  f.toNat = 945 ∧ c.toNat = 672 ∧ f.toNat * c.toNat = 635040 := by
  sorry


end NUMINAMATH_CALUDE_majka_numbers_unique_l622_62294


namespace NUMINAMATH_CALUDE_X_related_Y_probability_l622_62227

/-- The probability of k² being greater than or equal to 10.83 under the null hypothesis -/
def p_k_squared_ge_10_83 : ℝ := 0.001

/-- The null hypothesis states that variable X is unrelated to variable Y -/
def H₀ : Prop := sorry

/-- The probability that variable X is related to variable Y -/
def p_X_related_Y : ℝ := sorry

/-- Theorem stating the relationship between p_X_related_Y and p_k_squared_ge_10_83 -/
theorem X_related_Y_probability : 
  p_X_related_Y = 1 - p_k_squared_ge_10_83 := by sorry

end NUMINAMATH_CALUDE_X_related_Y_probability_l622_62227


namespace NUMINAMATH_CALUDE_area_covered_by_strips_l622_62214

/-- The area covered by four overlapping rectangular strips on a table -/
def area_covered (length width : ℝ) : ℝ :=
  4 * length * width - 4 * width * width

/-- Theorem stating that the area covered by four overlapping rectangular strips,
    each 16 cm long and 2 cm wide, is 112 cm² -/
theorem area_covered_by_strips :
  area_covered 16 2 = 112 := by sorry

end NUMINAMATH_CALUDE_area_covered_by_strips_l622_62214


namespace NUMINAMATH_CALUDE_circle_equation_implies_value_l622_62241

theorem circle_equation_implies_value (x y : ℝ) : 
  x^2 + y^2 - 12*x + 16*y + 100 = 0 → (x - 7)^(-y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_implies_value_l622_62241


namespace NUMINAMATH_CALUDE_even_function_theorem_l622_62285

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The functional equation satisfied by f -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y - 2 * x * y - 1

theorem even_function_theorem (f : ℝ → ℝ) 
    (heven : EvenFunction f) 
    (heq : SatisfiesFunctionalEquation f) : 
    ∀ x, f x = -x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_theorem_l622_62285


namespace NUMINAMATH_CALUDE_complex_number_properties_l622_62283

theorem complex_number_properties (z : ℂ) (h : z * Complex.I = -3 + 2 * Complex.I) :
  z.im = 3 ∧ Complex.abs z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l622_62283


namespace NUMINAMATH_CALUDE_integer_solutions_of_quadratic_equation_l622_62222

theorem integer_solutions_of_quadratic_equation :
  ∀ x y : ℤ, x^2 = y^2 * (x + y^4 + 2*y^2) →
  (x = 0 ∧ y = 0) ∨ (x = 12 ∧ y = 2) ∨ (x = -8 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_quadratic_equation_l622_62222


namespace NUMINAMATH_CALUDE_variable_value_l622_62293

theorem variable_value (w x v : ℝ) 
  (h1 : 2 / w + 2 / x = 2 / v) 
  (h2 : w * x = v) 
  (h3 : (w + x) / 2 = 0.5) : 
  v = 0.25 := by
sorry

end NUMINAMATH_CALUDE_variable_value_l622_62293


namespace NUMINAMATH_CALUDE_basketball_game_equations_l622_62219

/-- Represents a basketball team's game results -/
structure BasketballTeam where
  gamesWon : ℕ
  gamesLost : ℕ

/-- Calculates the total points earned by a basketball team -/
def totalPoints (team : BasketballTeam) : ℕ :=
  2 * team.gamesWon + team.gamesLost

theorem basketball_game_equations (team : BasketballTeam) 
  (h1 : team.gamesWon + team.gamesLost = 12) 
  (h2 : totalPoints team = 20) : 
  (team.gamesWon + team.gamesLost = 12) ∧ (2 * team.gamesWon + team.gamesLost = 20) := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_equations_l622_62219


namespace NUMINAMATH_CALUDE_triangle_angle_B_l622_62273

theorem triangle_angle_B (A B C : ℝ) (a b c : ℝ) : 
  A = π/4 → a = 6 → b = 3 * Real.sqrt 2 → 
  0 < A ∧ A < π → 0 < B ∧ B < π → 0 < C ∧ C < π →
  a * Real.sin B = b * Real.sin A → 
  a > 0 ∧ b > 0 ∧ c > 0 →
  A + B + C = π →
  B = π/6 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l622_62273


namespace NUMINAMATH_CALUDE_max_average_raise_l622_62267

theorem max_average_raise (R S C A : ℝ) : 
  0.05 < R ∧ R < 0.10 →
  0.07 < S ∧ S < 0.12 →
  0.04 < C ∧ C < 0.09 →
  0.06 < A ∧ A < 0.15 →
  (R + S + C + A) / 4 ≤ 0.085 →
  ∃ (R' S' C' A' : ℝ),
    0.05 < R' ∧ R' < 0.10 ∧
    0.07 < S' ∧ S' < 0.12 ∧
    0.04 < C' ∧ C' < 0.09 ∧
    0.06 < A' ∧ A' < 0.15 ∧
    (R' + S' + C' + A') / 4 = 0.085 :=
by sorry

end NUMINAMATH_CALUDE_max_average_raise_l622_62267


namespace NUMINAMATH_CALUDE_rectangle_longer_side_length_l622_62242

/-- Given a circle and rectangle with specific properties, prove the length of the rectangle's longer side --/
theorem rectangle_longer_side_length (r : ℝ) (circle_area rectangle_area : ℝ) (shorter_side longer_side : ℝ) : 
  r = 6 →  -- Circle radius is 6 cm
  circle_area = π * r^2 →  -- Area of the circle
  rectangle_area = 3 * circle_area →  -- Rectangle area is three times circle area
  shorter_side = 2 * r →  -- Shorter side is twice the radius
  rectangle_area = shorter_side * longer_side →  -- Rectangle area formula
  longer_side = 9 * π := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_length_l622_62242


namespace NUMINAMATH_CALUDE_max_value_expression_l622_62264

theorem max_value_expression (a b c d : ℝ) 
  (ha : -6.5 ≤ a ∧ a ≤ 6.5)
  (hb : -6.5 ≤ b ∧ b ≤ 6.5)
  (hc : -6.5 ≤ c ∧ c ≤ 6.5)
  (hd : -6.5 ≤ d ∧ d ≤ 6.5) :
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 182 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l622_62264


namespace NUMINAMATH_CALUDE_fractional_equation_positive_root_l622_62213

theorem fractional_equation_positive_root (m : ℝ) : 
  (∃ x : ℝ, x > 2 ∧ (3 / (x - 2) + (x + m) / (2 - x) = 1)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_positive_root_l622_62213


namespace NUMINAMATH_CALUDE_product_of_real_parts_of_complex_solutions_l622_62216

theorem product_of_real_parts_of_complex_solutions : ∃ (z₁ z₂ : ℂ),
  (z₁^2 + 2*z₁ = Complex.I) ∧
  (z₂^2 + 2*z₂ = Complex.I) ∧
  (z₁ ≠ z₂) ∧
  (Complex.re z₁ * Complex.re z₂ = (1 - Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_product_of_real_parts_of_complex_solutions_l622_62216


namespace NUMINAMATH_CALUDE_equation_solutions_l622_62277

def solutions : Set (ℤ × ℤ) := {(-13,-2), (-4,-1), (-1,0), (2,3), (3,6), (4,15), (6,-21), (7,-12), (8,-9), (11,-6), (14,-5), (23,-4)}

theorem equation_solutions :
  ∀ (x y : ℤ), (x * y + 3 * x - 5 * y = -3) ↔ (x, y) ∈ solutions :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l622_62277


namespace NUMINAMATH_CALUDE_mn_perpendicular_pq_l622_62272

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Line : Type :=
  (a b : Point)

-- Define the quadrilateral and its properties
structure Quadrilateral : Type :=
  (A B C D : Point)
  (convex : Bool)

-- Define the intersection point of diagonals
def intersectionPoint (q : Quadrilateral) : Point :=
  sorry

-- Define centroid of a triangle
def centroid (p1 p2 p3 : Point) : Point :=
  sorry

-- Define orthocenter of a triangle
def orthocenter (p1 p2 p3 : Point) : Point :=
  sorry

-- Define perpendicularity of lines
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

-- Theorem statement
theorem mn_perpendicular_pq (q : Quadrilateral) :
  let O := intersectionPoint q
  let M := centroid q.A O q.B
  let N := centroid q.C O q.D
  let P := orthocenter q.B O q.C
  let Q := orthocenter q.D O q.A
  perpendicular (Line.mk M N) (Line.mk P Q) :=
sorry

end NUMINAMATH_CALUDE_mn_perpendicular_pq_l622_62272


namespace NUMINAMATH_CALUDE_count_distinct_prime_factors_30_factorial_l622_62261

/-- The number of distinct prime factors of 30! -/
def distinct_prime_factors_30_factorial : ℕ := sorry

/-- Theorem stating that the number of distinct prime factors of 30! is 10 -/
theorem count_distinct_prime_factors_30_factorial :
  distinct_prime_factors_30_factorial = 10 := by sorry

end NUMINAMATH_CALUDE_count_distinct_prime_factors_30_factorial_l622_62261


namespace NUMINAMATH_CALUDE_jisha_walking_speed_l622_62297

/-- Jisha's walking problem -/
theorem jisha_walking_speed :
  -- Day 1 conditions
  let day1_distance : ℝ := 18
  let day1_speed : ℝ := 3
  let day1_hours : ℝ := day1_distance / day1_speed

  -- Day 2 conditions
  let day2_hours : ℝ := day1_hours - 1

  -- Day 3 conditions
  let day3_hours : ℝ := day1_hours

  -- Total distance
  let total_distance : ℝ := 62

  -- Unknown speed for Day 2 and 3
  ∀ day2_speed : ℝ,
    -- Total distance equation
    day1_distance + day2_speed * day2_hours + day2_speed * day3_hours = total_distance →
    -- Conclusion: Day 2 speed is 4 mph
    day2_speed = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_jisha_walking_speed_l622_62297


namespace NUMINAMATH_CALUDE_paco_ate_fifteen_sweet_cookies_l622_62298

/-- The number of sweet cookies Paco ate -/
def sweet_cookies_eaten (initial_sweet : ℕ) (sweet_left : ℕ) : ℕ :=
  initial_sweet - sweet_left

/-- Theorem stating that Paco ate 15 sweet cookies -/
theorem paco_ate_fifteen_sweet_cookies : 
  sweet_cookies_eaten 34 19 = 15 := by
  sorry

end NUMINAMATH_CALUDE_paco_ate_fifteen_sweet_cookies_l622_62298


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l622_62275

/-- A geometric sequence of positive integers -/
def GeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fourth_term
  (a : ℕ → ℕ)
  (h_geom : GeometricSequence a)
  (h_first : a 1 = 5)
  (h_fifth : a 5 = 1280) :
  a 4 = 320 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l622_62275


namespace NUMINAMATH_CALUDE_circle_on_parabola_passes_through_focus_l622_62247

/-- A circle with center on a parabola y^2 = 4x and tangent to x = -1 passes through (1, 0) -/
theorem circle_on_parabola_passes_through_focus (C : ℝ × ℝ) (r : ℝ) :
  C.2^2 = 4 * C.1 →  -- Center C is on the parabola y^2 = 4x
  abs (C.1 + 1) = r →  -- Circle is tangent to x = -1
  (1 - C.1)^2 + C.2^2 = r^2  -- Circle passes through (1, 0)
  := by sorry

end NUMINAMATH_CALUDE_circle_on_parabola_passes_through_focus_l622_62247


namespace NUMINAMATH_CALUDE_a_value_is_two_l622_62207

/-- The quadratic function we're considering -/
def f (a : ℝ) (x : ℝ) : ℝ := -2 * x^2 + a * x + 6

/-- The condition that f(a, x) > 0 only when x ∈ (-∞, -2) ∪ (3, ∞) -/
def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x > 0 ↔ (x < -2 ∨ x > 3)

/-- The theorem stating that under the given condition, a = 2 -/
theorem a_value_is_two :
  ∃ a : ℝ, condition a ∧ a = 2 := by sorry

end NUMINAMATH_CALUDE_a_value_is_two_l622_62207


namespace NUMINAMATH_CALUDE_side_altitude_inequality_l622_62268

/-- Triangle ABC with side lengths and altitudes -/
structure Triangle where
  a : ℝ
  b : ℝ
  hₐ : ℝ
  hb : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_hₐ : 0 < hₐ
  pos_hb : 0 < hb

/-- Theorem: In a triangle, a ≥ b if and only if a + hₐ ≥ b + hb -/
theorem side_altitude_inequality (t : Triangle) : t.a ≥ t.b ↔ t.a + t.hₐ ≥ t.b + t.hb := by
  sorry

end NUMINAMATH_CALUDE_side_altitude_inequality_l622_62268


namespace NUMINAMATH_CALUDE_friend_lunch_cost_l622_62200

theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 15 → difference = 1 → friend_cost = (total + difference) / 2 → friend_cost = 8 := by
  sorry

end NUMINAMATH_CALUDE_friend_lunch_cost_l622_62200


namespace NUMINAMATH_CALUDE_expression_factorization_l622_62249

theorem expression_factorization (x : ℝ) : 
  (16 * x^4 + 36 * x^2 - 9) - (4 * x^4 - 6 * x^2 - 9) = 6 * x^2 * (2 * x^2 + 7) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l622_62249


namespace NUMINAMATH_CALUDE_karting_routes_count_l622_62210

/-- Represents the number of routes ending at point A after n minutes -/
def M_n_A : ℕ → ℕ
| 0 => 0
| 1 => 0
| 2 => 1
| (n+3) => M_n_A (n+1) + M_n_A n

/-- The race duration in minutes -/
def race_duration : ℕ := 10

/-- Theorem stating that the number of routes ending at A after 10 minutes
    is equal to the 10th number in the defined Fibonacci-like sequence -/
theorem karting_routes_count : M_n_A race_duration = 34 := by
  sorry

end NUMINAMATH_CALUDE_karting_routes_count_l622_62210


namespace NUMINAMATH_CALUDE_advisory_panel_combinations_l622_62276

theorem advisory_panel_combinations (n : ℕ) (k : ℕ) : n = 30 → k = 5 → Nat.choose n k = 142506 := by
  sorry

end NUMINAMATH_CALUDE_advisory_panel_combinations_l622_62276


namespace NUMINAMATH_CALUDE_absolute_value_equation_extrema_l622_62228

theorem absolute_value_equation_extrema :
  ∀ x : ℝ, |x - 3| = 10 → (∃ y : ℝ, |y - 3| = 10 ∧ y ≥ x) ∧ (∃ z : ℝ, |z - 3| = 10 ∧ z ≤ x) ∧
  (∀ w : ℝ, |w - 3| = 10 → w ≤ 13 ∧ w ≥ -7) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_extrema_l622_62228


namespace NUMINAMATH_CALUDE_geometric_sequence_bounded_ratio_counterexample_l622_62279

theorem geometric_sequence_bounded_ratio_counterexample :
  ¬ (∀ (a₁ : ℝ) (q : ℝ) (a : ℝ),
    (a₁ > 0 ∧ q > 0) →
    (∀ n : ℕ, a₁ * q^(n - 1) < a) →
    (q > 0 ∧ q < 1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_bounded_ratio_counterexample_l622_62279


namespace NUMINAMATH_CALUDE_percentage_in_accounting_l622_62250

def accountant_years : ℕ := 25
def manager_years : ℕ := 15
def total_lifespan : ℕ := 80

def accounting_years : ℕ := accountant_years + manager_years

theorem percentage_in_accounting : 
  (accounting_years : ℚ) / total_lifespan * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_in_accounting_l622_62250


namespace NUMINAMATH_CALUDE_trinomial_fourth_power_l622_62226

theorem trinomial_fourth_power (a b c : ℤ) : 
  (∀ x : ℤ, ∃ y : ℤ, a * x^2 + b * x + c = y^4) → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_trinomial_fourth_power_l622_62226


namespace NUMINAMATH_CALUDE_power_sum_sequence_l622_62211

theorem power_sum_sequence (a b : ℝ) : 
  a + b = 1 → 
  a^2 + b^2 = 3 → 
  a^3 + b^3 = 4 → 
  a^4 + b^4 = 7 → 
  a^5 + b^5 = 11 → 
  a^10 + b^10 = 123 := by
sorry

end NUMINAMATH_CALUDE_power_sum_sequence_l622_62211


namespace NUMINAMATH_CALUDE_jacobs_age_multiple_l622_62203

/-- Proves that Jacob's age will be 3 times his son's age in five years -/
theorem jacobs_age_multiple (jacob_age son_age : ℕ) : 
  jacob_age = 40 →
  son_age = 10 →
  jacob_age - 5 = 7 * (son_age - 5) →
  (jacob_age + 5) = 3 * (son_age + 5) := by
  sorry

end NUMINAMATH_CALUDE_jacobs_age_multiple_l622_62203


namespace NUMINAMATH_CALUDE_line_parallel_to_intersection_l622_62233

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel_line_line : Line → Line → Prop)

-- Define the intersection of planes
variable (intersection : Plane → Plane → Line)

-- Theorem statement
theorem line_parallel_to_intersection
  (a b : Line) (α β : Plane)
  (h1 : a ≠ b)
  (h2 : α ≠ β)
  (h3 : parallel_line_plane a α)
  (h4 : parallel_line_plane a β)
  (h5 : intersection α β = b) :
  parallel_line_line a b :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_intersection_l622_62233


namespace NUMINAMATH_CALUDE_nell_initial_cards_l622_62271

theorem nell_initial_cards (cards_given_to_jeff cards_left : ℕ) 
  (h1 : cards_given_to_jeff = 301)
  (h2 : cards_left = 154) :
  cards_given_to_jeff + cards_left = 455 :=
by sorry

end NUMINAMATH_CALUDE_nell_initial_cards_l622_62271


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l622_62288

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 27 ∧ x - y = 7 → x * y = 170 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l622_62288


namespace NUMINAMATH_CALUDE_triangular_pyramid_not_circular_top_view_l622_62225

-- Define the types of solids
inductive Solid
  | Sphere
  | Cylinder
  | Cone
  | TriangularPyramid

-- Define a property for having a circular top view
def has_circular_top_view (s : Solid) : Prop :=
  match s with
  | Solid.Sphere => True
  | Solid.Cylinder => True
  | Solid.Cone => True
  | Solid.TriangularPyramid => False

-- Theorem statement
theorem triangular_pyramid_not_circular_top_view :
  ∀ s : Solid, ¬(has_circular_top_view s) ↔ s = Solid.TriangularPyramid :=
by sorry

end NUMINAMATH_CALUDE_triangular_pyramid_not_circular_top_view_l622_62225


namespace NUMINAMATH_CALUDE_mixture_composition_l622_62289

theorem mixture_composition (x y : ℝ) :
  x + y = 100 →
  0.1 * x + 0.2 * y = 12 →
  x = 80 := by
sorry

end NUMINAMATH_CALUDE_mixture_composition_l622_62289


namespace NUMINAMATH_CALUDE_expression_evaluation_l622_62239

theorem expression_evaluation : 5 * 7 + 9 * 4 - (15 / 3)^2 = 46 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l622_62239


namespace NUMINAMATH_CALUDE_inequality_proof_l622_62232

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) : 
  a * b ≤ 1/8 ∧ 1/a + 2/b ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l622_62232


namespace NUMINAMATH_CALUDE_oplus_roots_l622_62234

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := a^2 - 5*a + 2*b

-- State the theorem
theorem oplus_roots : 
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = 3 ∧ 
  (∀ x : ℝ, oplus x 3 = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_oplus_roots_l622_62234


namespace NUMINAMATH_CALUDE_rose_purchase_problem_l622_62274

theorem rose_purchase_problem :
  ∃! (x y : ℤ), 
    (y = 1) ∧ 
    (x > 0) ∧
    (100 / x : ℚ) - (200 / (x + 10) : ℚ) = 80 / 12 ∧
    x = 5 ∧
    y = 1 := by
  sorry

end NUMINAMATH_CALUDE_rose_purchase_problem_l622_62274


namespace NUMINAMATH_CALUDE_factorization_equality_l622_62260

theorem factorization_equality (a b : ℝ) : a^2 * b - a^3 = a^2 * (b - a) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l622_62260


namespace NUMINAMATH_CALUDE_abc_sum_mod_five_l622_62265

theorem abc_sum_mod_five (a b c : ℕ) : 
  0 < a ∧ a < 5 ∧ 
  0 < b ∧ b < 5 ∧ 
  0 < c ∧ c < 5 ∧ 
  (a * b * c) % 5 = 1 ∧ 
  (4 * c) % 5 = 3 ∧ 
  (3 * b) % 5 = (2 + b) % 5 → 
  (a + b + c) % 5 = 3 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_mod_five_l622_62265


namespace NUMINAMATH_CALUDE_bees_after_14_days_l622_62230

/-- Calculates the total number of bees in a hive after a given number of days -/
def totalBeesAfterDays (initialBees : ℕ) (beesHatchedPerDay : ℕ) (beesLostPerDay : ℕ) (days : ℕ) : ℕ :=
  initialBees + (beesHatchedPerDay - beesLostPerDay) * days + 1

/-- Theorem: Given the specified conditions, the total number of bees after 14 days is 64801 -/
theorem bees_after_14_days :
  totalBeesAfterDays 20000 5000 1800 14 = 64801 := by
  sorry

#eval totalBeesAfterDays 20000 5000 1800 14

end NUMINAMATH_CALUDE_bees_after_14_days_l622_62230


namespace NUMINAMATH_CALUDE_problem_statement_l622_62224

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  std_dev_pos : 0 < std_dev

/-- The value that is a given number of standard deviations below the mean -/
def value_below_mean (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

/-- The problem statement -/
theorem problem_statement (d : NormalDistribution) 
  (h1 : d.mean = 17.5)
  (h2 : d.std_dev = 2.5) :
  value_below_mean d 2.7 = 10.75 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l622_62224


namespace NUMINAMATH_CALUDE_ellipse_foci_l622_62208

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 25 = 1

/-- The coordinates of a focus of the ellipse -/
def is_focus (x y : ℝ) : Prop :=
  (x = 0 ∧ y = 3) ∨ (x = 0 ∧ y = -3)

/-- Theorem stating that the given coordinates are the foci of the ellipse -/
theorem ellipse_foci :
  ∀ x y : ℝ, ellipse_equation x y → is_focus x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_l622_62208


namespace NUMINAMATH_CALUDE_simple_interest_difference_l622_62248

/-- Simple interest calculation and comparison --/
theorem simple_interest_difference (principal rate time : ℕ) : 
  principal = 3000 → 
  rate = 4 → 
  time = 5 → 
  principal - (principal * rate * time) / 100 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_difference_l622_62248


namespace NUMINAMATH_CALUDE_molecular_weight_Al2S3_proof_l622_62254

/-- The molecular weight of Al2S3 in g/mol -/
def molecular_weight_Al2S3 : ℝ := 150

/-- The number of moles used in the given condition -/
def moles : ℝ := 3

/-- The total weight of the given number of moles in grams -/
def total_weight : ℝ := 450

/-- Theorem: The molecular weight of Al2S3 is 150 g/mol, given that 3 moles weigh 450 grams -/
theorem molecular_weight_Al2S3_proof : 
  molecular_weight_Al2S3 = total_weight / moles := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_Al2S3_proof_l622_62254


namespace NUMINAMATH_CALUDE_profit_percentage_l622_62223

theorem profit_percentage (selling_price cost_price : ℝ) (h : cost_price = 0.81 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100/81 - 1) * 100 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l622_62223


namespace NUMINAMATH_CALUDE_triangle_angle_identity_l622_62299

theorem triangle_angle_identity (α β γ : Real) (h : α + β + γ = π) :
  (Real.sin β)^2 + (Real.sin γ)^2 - 2 * (Real.sin β) * (Real.sin γ) * (Real.cos α) = (Real.sin α)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_identity_l622_62299


namespace NUMINAMATH_CALUDE_triangle_properties_l622_62220

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) :
  (2 * Real.sin t.B * Real.cos t.A = Real.sin t.A * Real.cos t.C + Real.cos t.A * Real.sin t.C) →
  (t.A = π / 3) ∧
  (t.A = π / 3 ∧ t.a = 6) →
  ∃ p : Real, 12 < p ∧ p ≤ 18 ∧ p = t.a + t.b + t.c :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l622_62220


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l622_62290

theorem simplify_complex_fraction (x : ℝ) (h : x ≠ 2) :
  ((x + 1) / (x - 2) - 1) / ((x^2 - 2*x) / (x^2 - 4*x + 4)) = 3 / x :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l622_62290


namespace NUMINAMATH_CALUDE_reflect_point_across_y_axis_l622_62221

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point) : Point :=
  ⟨-p.x, p.y⟩

theorem reflect_point_across_y_axis :
  let P : Point := ⟨5, -1⟩
  reflectAcrossYAxis P = ⟨-5, -1⟩ := by
  sorry

end NUMINAMATH_CALUDE_reflect_point_across_y_axis_l622_62221


namespace NUMINAMATH_CALUDE_property_tax_increase_l622_62263

/-- Calculates the property tax increase when the assessed value changes, given a fixed tax rate. -/
theorem property_tax_increase 
  (tax_rate : ℝ) 
  (initial_value : ℝ) 
  (new_value : ℝ) 
  (h1 : tax_rate = 0.1)
  (h2 : initial_value = 20000)
  (h3 : new_value = 28000) :
  new_value * tax_rate - initial_value * tax_rate = 800 :=
by sorry

end NUMINAMATH_CALUDE_property_tax_increase_l622_62263


namespace NUMINAMATH_CALUDE_trapezoid_ed_length_l622_62251

/-- Represents a trapezoid ABCD with given properties -/
structure Trapezoid where
  /-- Length of base AB -/
  base : ℝ
  /-- Length of top base CD -/
  top_base : ℝ
  /-- Length of non-parallel sides AD and BC -/
  side : ℝ
  /-- E is the midpoint of diagonal AC -/
  e_midpoint : Bool
  /-- AED is a right triangle -/
  aed_right : Bool
  /-- D lies on extended line segment AE -/
  d_on_ae : Bool

/-- Theorem stating the length of ED in the given trapezoid -/
theorem trapezoid_ed_length (t : Trapezoid) 
  (h1 : t.base = 8) 
  (h2 : t.top_base = 6) 
  (h3 : t.side = 5) 
  (h4 : t.e_midpoint) 
  (h5 : t.aed_right) 
  (h6 : t.d_on_ae) : 
  ∃ (ed : ℝ), ed = Real.sqrt 6.5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_ed_length_l622_62251


namespace NUMINAMATH_CALUDE_rectangle_area_change_l622_62245

theorem rectangle_area_change (L W : ℝ) (h1 : L * W = 625) : 
  (1.2 * L) * (0.8 * W) = 600 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l622_62245


namespace NUMINAMATH_CALUDE_optimal_robot_purchase_l622_62229

/-- Represents the robot purchase problem -/
structure RobotPurchase where
  cost_A : ℕ  -- Cost of A robot in yuan
  cost_B : ℕ  -- Cost of B robot in yuan
  capacity_A : ℕ  -- Daily capacity of A robot in tons
  capacity_B : ℕ  -- Daily capacity of B robot in tons
  total_robots : ℕ  -- Total number of robots to purchase
  min_capacity : ℕ  -- Minimum daily capacity required

/-- The optimal solution minimizes the total cost -/
def optimal_solution (rp : RobotPurchase) : ℕ × ℕ × ℕ :=
  sorry

/-- Theorem stating the optimal solution for the given problem -/
theorem optimal_robot_purchase :
  let rp : RobotPurchase := {
    cost_A := 12000,
    cost_B := 20000,
    capacity_A := 90,
    capacity_B := 100,
    total_robots := 30,
    min_capacity := 2830
  }
  let (num_A, num_B, total_cost) := optimal_solution rp
  num_A = 17 ∧ num_B = 13 ∧ total_cost = 464000 :=
by sorry

end NUMINAMATH_CALUDE_optimal_robot_purchase_l622_62229


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l622_62205

theorem purely_imaginary_complex_number (a : ℝ) : 
  (2 : ℂ) + Complex.I * ((1 : ℂ) - a + a * Complex.I) = Complex.I * (Complex.I.im * ((1 : ℂ) - a + a * Complex.I)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l622_62205


namespace NUMINAMATH_CALUDE_no_polyhedron_with_seven_edges_l622_62284

-- Define a polyhedron structure
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  euler_formula : vertices - edges + faces = 2
  min_edges_per_vertex : edges ≥ (3 * vertices) / 2

-- Theorem statement
theorem no_polyhedron_with_seven_edges : 
  ∀ p : Polyhedron, p.edges ≠ 7 := by
  sorry

end NUMINAMATH_CALUDE_no_polyhedron_with_seven_edges_l622_62284


namespace NUMINAMATH_CALUDE_total_coins_is_21_l622_62282

/-- The number of quarters in the wallet -/
def num_quarters : ℕ := 8

/-- The number of nickels in the wallet -/
def num_nickels : ℕ := 13

/-- The total number of coins in the wallet -/
def total_coins : ℕ := num_quarters + num_nickels

/-- Theorem stating that the total number of coins is 21 -/
theorem total_coins_is_21 : total_coins = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_is_21_l622_62282


namespace NUMINAMATH_CALUDE_ratio_c_to_a_is_sqrt2_l622_62270

/-- A configuration of four points on a plane -/
structure PointConfiguration where
  /-- The length of four segments -/
  a : ℝ
  /-- The length of the longest segment -/
  longest : ℝ
  /-- The length of the remaining segment -/
  c : ℝ
  /-- The longest segment is twice the length of a -/
  longest_eq_2a : longest = 2 * a
  /-- The configuration contains a 45-45-90 triangle -/
  has_45_45_90_triangle : True
  /-- The hypotenuse of the 45-45-90 triangle is the longest segment -/
  hypotenuse_is_longest : True
  /-- All points are distinct -/
  points_distinct : True

/-- The ratio of c to a in the given point configuration is √2 -/
theorem ratio_c_to_a_is_sqrt2 (config : PointConfiguration) : 
  config.c / config.a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_c_to_a_is_sqrt2_l622_62270


namespace NUMINAMATH_CALUDE_rectangle_length_l622_62202

theorem rectangle_length (P w l A : ℝ) : 
  P > 0 → w > 0 → l > 0 → A > 0 →
  P = 2 * (l + w) →
  P / w = 5 →
  A = l * w →
  A = 150 →
  l = 15 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_l622_62202


namespace NUMINAMATH_CALUDE_fish_population_changes_l622_62286

def initial_salmon : ℕ := 500
def initial_halibut : ℕ := 800
def initial_trout : ℕ := 700

def salmon_increase_factor : ℕ := 10
def salmon_loss_rate : ℚ := 25 / 150
def halibut_reduction_rate : ℚ := 1 / 10
def trout_reduction_rate : ℚ := 1 / 20

def final_salmon : ℕ := 4175
def final_halibut : ℕ := 720
def final_trout : ℕ := 665

theorem fish_population_changes :
  (initial_salmon * salmon_increase_factor - 
   (initial_salmon * salmon_increase_factor / 150 : ℚ).floor * 25 = final_salmon) ∧
  (initial_halibut - (initial_halibut : ℚ) * halibut_reduction_rate = final_halibut) ∧
  (initial_trout - (initial_trout : ℚ) * trout_reduction_rate = final_trout) := by
  sorry

end NUMINAMATH_CALUDE_fish_population_changes_l622_62286


namespace NUMINAMATH_CALUDE_baker_eggs_theorem_l622_62204

/-- Calculates the number of eggs needed for a given amount of flour, based on a recipe ratio. -/
def eggs_needed (recipe_flour : ℚ) (recipe_eggs : ℚ) (available_flour : ℚ) : ℚ :=
  (available_flour / recipe_flour) * recipe_eggs

theorem baker_eggs_theorem (recipe_flour : ℚ) (recipe_eggs : ℚ) (available_flour : ℚ) 
  (h1 : recipe_flour = 2)
  (h2 : recipe_eggs = 3)
  (h3 : available_flour = 6) :
  eggs_needed recipe_flour recipe_eggs available_flour = 9 := by
  sorry

#eval eggs_needed 2 3 6

end NUMINAMATH_CALUDE_baker_eggs_theorem_l622_62204


namespace NUMINAMATH_CALUDE_gigi_additional_batches_l622_62296

/-- Represents the number of cups of flour required for one batch of cookies -/
def flour_per_batch : ℕ := 2

/-- Represents the number of batches Gigi has already baked -/
def batches_baked : ℕ := 3

/-- Represents the total amount of flour in Gigi's bag -/
def total_flour : ℕ := 20

/-- Calculates the number of additional batches Gigi can make with the remaining flour -/
def additional_batches : ℕ := (total_flour - batches_baked * flour_per_batch) / flour_per_batch

/-- Proves that Gigi can make 7 more batches of cookies with the remaining flour -/
theorem gigi_additional_batches : additional_batches = 7 := by
  sorry

end NUMINAMATH_CALUDE_gigi_additional_batches_l622_62296


namespace NUMINAMATH_CALUDE_min_balls_same_color_l622_62257

/-- Represents the number of different colors of balls in the bag -/
def num_colors : ℕ := 2

/-- Represents the minimum number of balls to draw -/
def min_balls : ℕ := 3

/-- Theorem stating that given a bag with balls of two colors, 
    the minimum number of balls that must be drawn to ensure 
    at least two balls of the same color is 3 -/
theorem min_balls_same_color :
  ∀ (n : ℕ), n ≥ min_balls → 
  ∃ (color : Fin num_colors), (n.choose 2) > 0 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_same_color_l622_62257


namespace NUMINAMATH_CALUDE_de_morgan_laws_l622_62253

universe u

theorem de_morgan_laws {α : Type u} (A B : Set α) : 
  ((A ∪ B)ᶜ = Aᶜ ∩ Bᶜ) ∧ ((A ∩ B)ᶜ = Aᶜ ∪ Bᶜ) := by
  sorry

end NUMINAMATH_CALUDE_de_morgan_laws_l622_62253


namespace NUMINAMATH_CALUDE_alice_work_problem_l622_62215

/-- Alice's work problem -/
theorem alice_work_problem (total_days : ℕ) (daily_wage : ℕ) (daily_loss : ℕ) (total_earnings : ℤ) :
  total_days = 20 →
  daily_wage = 80 →
  daily_loss = 40 →
  total_earnings = 880 →
  ∃ (days_not_worked : ℕ),
    days_not_worked = 6 ∧
    days_not_worked ≤ total_days ∧
    (daily_wage * (total_days - days_not_worked) : ℤ) - (daily_loss * days_not_worked : ℤ) = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_alice_work_problem_l622_62215


namespace NUMINAMATH_CALUDE_line_relations_l622_62236

-- Define the concept of a line in space
structure Line3D where
  -- You might define a line using a point and a direction vector
  -- But for simplicity, we'll just use an opaque type
  dummy : Unit

-- Define the relations between lines
def parallel (l1 l2 : Line3D) : Prop := sorry
def intersects (l1 l2 : Line3D) : Prop := sorry
def skew (l1 l2 : Line3D) : Prop := sorry

-- State the theorem
theorem line_relations (a b c : Line3D) : 
  parallel a b → intersects a c → (intersects b c ∨ skew b c) := by sorry

end NUMINAMATH_CALUDE_line_relations_l622_62236


namespace NUMINAMATH_CALUDE_equation_D_is_linear_l622_62238

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 2x - 3 = 5 -/
def equation_D (x : ℝ) : ℝ := 2 * x - 3

theorem equation_D_is_linear : is_linear_equation equation_D := by
  sorry

end NUMINAMATH_CALUDE_equation_D_is_linear_l622_62238


namespace NUMINAMATH_CALUDE_some_number_value_l622_62252

theorem some_number_value (x : ℝ) : (50 + 20/x) * x = 4520 → x = 90 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l622_62252


namespace NUMINAMATH_CALUDE_triangle_CSE_is_equilateral_l622_62292

-- Define the circle k
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the chord AB
def Chord (k : Set (ℝ × ℝ)) (A B : ℝ × ℝ) : Prop :=
  A ∈ k ∧ B ∈ k

-- Define the perpendicular bisector
def PerpendicularBisector (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {X : ℝ × ℝ | (X.1 - P.1)^2 + (X.2 - P.2)^2 = (X.1 - Q.1)^2 + (X.2 - Q.2)^2}

-- Define the line through two points
def LineThroughPoints (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {X : ℝ × ℝ | (X.2 - P.2) * (Q.1 - P.1) = (X.1 - P.1) * (Q.2 - P.2)}

theorem triangle_CSE_is_equilateral
  (k : Set (ℝ × ℝ))
  (r : ℝ)
  (A B S C D E : ℝ × ℝ)
  (h1 : k = Circle (0, 0) r)
  (h2 : Chord k A B)
  (h3 : S ∈ LineThroughPoints A B)
  (h4 : (S.1 - A.1)^2 + (S.2 - A.2)^2 = r^2)
  (h5 : (B.1 - A.1)^2 + (B.2 - A.2)^2 > r^2)
  (h6 : C ∈ k ∧ C ∈ PerpendicularBisector B S)
  (h7 : D ∈ k ∧ D ∈ PerpendicularBisector B S)
  (h8 : E ∈ k ∧ E ∈ LineThroughPoints D S) :
  (C.1 - S.1)^2 + (C.2 - S.2)^2 = (C.1 - E.1)^2 + (C.2 - E.2)^2 ∧
  (C.1 - S.1)^2 + (C.2 - S.2)^2 = (E.1 - S.1)^2 + (E.2 - S.2)^2 :=
sorry

end NUMINAMATH_CALUDE_triangle_CSE_is_equilateral_l622_62292
