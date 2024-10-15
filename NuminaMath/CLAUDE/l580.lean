import Mathlib

namespace NUMINAMATH_CALUDE_summit_academy_contestants_l580_58016

theorem summit_academy_contestants (s j : ℕ) (h : s / 3 = j * 3 / 4) : s = 4 * j := by
  sorry

end NUMINAMATH_CALUDE_summit_academy_contestants_l580_58016


namespace NUMINAMATH_CALUDE_complex_number_properties_l580_58067

/-- Given a complex number z and a real number m, where z = m^2 - m - 2 + (5m^2 - 20)i -/
theorem complex_number_properties (m : ℝ) (z : ℂ) 
  (h : z = (m^2 - m - 2 : ℝ) + (5 * m^2 - 20 : ℝ) * Complex.I) :
  (z.im = 0 ↔ m = 2 ∨ m = -2) ∧ 
  (z.re = 0 ∧ z.im ≠ 0 ↔ m = -1) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l580_58067


namespace NUMINAMATH_CALUDE_exists_monochromatic_triangle_l580_58005

/-- A type representing the scientists -/
def Scientist : Type := Fin 17

/-- A type representing the topics -/
def Topic : Type := Fin 3

/-- A function representing the correspondence between scientists on a specific topic -/
def corresponds (s1 s2 : Scientist) : Topic :=
  sorry

/-- The main theorem stating that there exists a monochromatic triangle -/
theorem exists_monochromatic_triangle :
  ∃ (s1 s2 s3 : Scientist) (t : Topic),
    s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 ∧
    corresponds s1 s2 = t ∧
    corresponds s2 s3 = t ∧
    corresponds s1 s3 = t :=
  sorry

end NUMINAMATH_CALUDE_exists_monochromatic_triangle_l580_58005


namespace NUMINAMATH_CALUDE_patent_agency_employment_relation_l580_58043

/-- Data for graduate students and their preferences for patent agency employment --/
structure GraduateData where
  total : ℕ
  male_like : ℕ
  male_dislike : ℕ
  female_like : ℕ
  female_dislike : ℕ

/-- Calculate the probability of selecting at least 2 students who like employment
    in patent agency when 3 are selected --/
def probability_at_least_two (data : GraduateData) : ℚ :=
  let p := (data.male_like + data.female_like : ℚ) / data.total
  3 * p^2 * (1 - p) + p^3

/-- Calculate the chi-square statistic for the given data --/
def chi_square (data : GraduateData) : ℚ :=
  let n := data.total
  let a := data.male_like
  let b := data.male_dislike
  let c := data.female_like
  let d := data.female_dislike
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The main theorem to be proved --/
theorem patent_agency_employment_relation (data : GraduateData)
  (h_total : data.total = 200)
  (h_male_like : data.male_like = 60)
  (h_male_dislike : data.male_dislike = 40)
  (h_female_like : data.female_like = 80)
  (h_female_dislike : data.female_dislike = 20) :
  probability_at_least_two data = 98/125 ∧ chi_square data > 7879/1000 :=
sorry


end NUMINAMATH_CALUDE_patent_agency_employment_relation_l580_58043


namespace NUMINAMATH_CALUDE_repeated_sequence_result_l580_58020

/-- Represents one cycle of operations -/
def cycle_increment : Int := 15 - 12 + 3

/-- Calculates the number of complete cycles in n steps -/
def complete_cycles (n : Nat) : Nat := n / 3

/-- Calculates the number of remaining steps after complete cycles -/
def remaining_steps (n : Nat) : Nat := n % 3

/-- Calculates the increment from remaining steps -/
def remaining_increment (steps : Nat) : Int :=
  if steps = 1 then 15
  else if steps = 2 then 15 - 12
  else 0

/-- Theorem stating the result of the repeated operation sequence -/
theorem repeated_sequence_result :
  let initial_value : Int := 100
  let total_steps : Nat := 26
  let cycles : Nat := complete_cycles total_steps
  let remaining : Nat := remaining_steps total_steps
  initial_value + cycles * cycle_increment + remaining_increment remaining = 151 := by
  sorry

end NUMINAMATH_CALUDE_repeated_sequence_result_l580_58020


namespace NUMINAMATH_CALUDE_function_value_ordering_l580_58036

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function f is increasing on [0, +∞) if f(x) ≤ f(y) for all 0 ≤ x ≤ y -/
def IncreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem function_value_ordering (f : ℝ → ℝ) 
    (heven : EvenFunction f) (hincr : IncreasingOnNonnegative f) :
    f 1 < f (-2) ∧ f (-2) < f (-3) := by
  sorry

end NUMINAMATH_CALUDE_function_value_ordering_l580_58036


namespace NUMINAMATH_CALUDE_system_of_inequalities_l580_58062

theorem system_of_inequalities (x : ℝ) :
  (2 + x < 6 - 3 * x) ∧ (x ≤ (4 + x) / 2) → x < 1 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l580_58062


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l580_58093

structure Grid :=
  (size : Nat)
  (shaded : List (Nat × Nat))

def isExternal (g : Grid) (pos : Nat × Nat) : Bool :=
  let (x, y) := pos
  x = 1 ∨ x = g.size ∨ y = 1 ∨ y = g.size

def countExternalEdges (g : Grid) : Nat :=
  g.shaded.foldl (fun acc pos =>
    acc + (if isExternal g pos then
             (if pos.1 = 1 then 1 else 0) +
             (if pos.1 = g.size then 1 else 0) +
             (if pos.2 = 1 then 1 else 0) +
             (if pos.2 = g.size then 1 else 0)
           else 0)
  ) 0

theorem shaded_region_perimeter (g : Grid) :
  g.size = 3 ∧
  g.shaded = [(1,2), (2,1), (2,3), (3,2)] →
  countExternalEdges g = 10 := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l580_58093


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_PQR_l580_58069

/-- The orthocenter of a triangle in 3D space. -/
def orthocenter (P Q R : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem: The orthocenter of triangle PQR is (3/2, 13/2, 5) -/
theorem orthocenter_of_triangle_PQR :
  let P : ℝ × ℝ × ℝ := (2, 3, 4)
  let Q : ℝ × ℝ × ℝ := (6, 4, 2)
  let R : ℝ × ℝ × ℝ := (4, 5, 6)
  orthocenter P Q R = (3/2, 13/2, 5) :=
by
  sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_PQR_l580_58069


namespace NUMINAMATH_CALUDE_problem_statement_l580_58087

/-- Given M = 6021 ÷ 4, N = 2M, and X = N - M + 500, prove that X = 3005.25 -/
theorem problem_statement (M N X : ℚ) 
  (hM : M = 6021 / 4)
  (hN : N = 2 * M)
  (hX : X = N - M + 500) :
  X = 3005.25 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l580_58087


namespace NUMINAMATH_CALUDE_root_value_range_l580_58055

theorem root_value_range (a : ℝ) (h : a^2 + 3*a - 1 = 0) :
  2 < a^2 + 3*a + Real.sqrt 3 ∧ a^2 + 3*a + Real.sqrt 3 < 3 := by
  sorry

end NUMINAMATH_CALUDE_root_value_range_l580_58055


namespace NUMINAMATH_CALUDE_expression_equality_l580_58045

theorem expression_equality : 2⁻¹ - Real.sqrt 3 * Real.tan (60 * π / 180) + (π - 2011)^0 + |(-1/2)| = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l580_58045


namespace NUMINAMATH_CALUDE_triangle_height_l580_58001

theorem triangle_height (x y : ℝ) (x_pos : 0 < x) (y_pos : 0 < y) : 
  let area := (x^3 * y)^2
  let side := (2 * x * y)^2
  let height := (1/2) * x^4
  area = (1/2) * side * height := by
sorry

end NUMINAMATH_CALUDE_triangle_height_l580_58001


namespace NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l580_58059

theorem sum_of_quadratic_solutions : 
  let f (x : ℝ) := x^2 - 6*x - 22 - (2*x + 18)
  let roots := {x : ℝ | f x = 0}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ roots ∧ x₂ ∈ roots ∧ x₁ + x₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l580_58059


namespace NUMINAMATH_CALUDE_max_value_of_f_l580_58064

-- Define the quadratic function
def f (x : ℝ) : ℝ := -8 * x^2 + 32 * x - 1

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M) ∧ M = 31 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l580_58064


namespace NUMINAMATH_CALUDE_batsman_average_l580_58042

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℕ) :
  total_innings = 12 →
  last_innings_score = 65 →
  average_increase = 2 →
  (∃ (prev_average : ℕ),
    (prev_average * (total_innings - 1) + last_innings_score) / total_innings = prev_average + average_increase) →
  (((total_innings - 1) * ((last_innings_score + (total_innings - 1) * average_increase) / total_innings - average_increase) + last_innings_score) / total_innings) = 43 :=
by sorry

end NUMINAMATH_CALUDE_batsman_average_l580_58042


namespace NUMINAMATH_CALUDE_temperature_range_l580_58075

/-- Given the highest and lowest temperatures on a certain day, 
    prove that the range of temperature change is between these two values, inclusive. -/
theorem temperature_range (highest lowest t : ℝ) 
  (h_highest : highest = 26) 
  (h_lowest : lowest = 12) 
  (h_range : lowest ≤ t ∧ t ≤ highest) : 
  12 ≤ t ∧ t ≤ 26 := by
  sorry

end NUMINAMATH_CALUDE_temperature_range_l580_58075


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_exists_185_is_greatest_l580_58089

theorem greatest_integer_with_gcf_five (n : ℕ) : n < 200 ∧ Nat.gcd n 30 = 5 → n ≤ 185 :=
by
  sorry

theorem exists_185 : 185 < 200 ∧ Nat.gcd 185 30 = 5 :=
by
  sorry

theorem is_greatest : ∀ m : ℕ, m < 200 ∧ Nat.gcd m 30 = 5 → m ≤ 185 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_exists_185_is_greatest_l580_58089


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l580_58007

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 8 * x * y) : 1 / x + 1 / y = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l580_58007


namespace NUMINAMATH_CALUDE_complex_equation_roots_l580_58066

theorem complex_equation_roots : 
  let z₁ : ℂ := (1 + 2 * Real.sqrt 7 - Complex.I * Real.sqrt 7) / 2
  let z₂ : ℂ := (1 - 2 * Real.sqrt 7 + Complex.I * Real.sqrt 7) / 2
  (z₁ ^ 2 - z₁ = 3 - 7 * Complex.I) ∧ (z₂ ^ 2 - z₂ = 3 - 7 * Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_roots_l580_58066


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l580_58002

theorem gcd_lcm_sum : Nat.gcd 42 70 + Nat.lcm 8 32 = 46 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l580_58002


namespace NUMINAMATH_CALUDE_factor_expression_l580_58015

theorem factor_expression (x : ℝ) : 81 * x^3 + 27 * x^2 = 27 * x^2 * (3 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l580_58015


namespace NUMINAMATH_CALUDE_permutations_count_l580_58047

def word_length : ℕ := 12
def repeated_letter_count : ℕ := 2

theorem permutations_count :
  (word_length.factorial / repeated_letter_count.factorial) = 239500800 := by
  sorry

end NUMINAMATH_CALUDE_permutations_count_l580_58047


namespace NUMINAMATH_CALUDE_triangle_problem_l580_58023

theorem triangle_problem (a b c A B C : ℝ) (h1 : a = 3) 
  (h2 : (a + b) * Real.sin B = (Real.sin A + Real.sin C) * (a + b - c))
  (h3 : a * Real.cos B + b * Real.cos A = Real.sqrt 3) :
  A = π / 3 ∧ (1 / 2 : ℝ) * a * c = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l580_58023


namespace NUMINAMATH_CALUDE_andrew_payment_l580_58081

/-- The total amount Andrew paid to the shopkeeper -/
def total_amount (grape_price grape_quantity mango_price mango_quantity : ℕ) : ℕ :=
  grape_price * grape_quantity + mango_price * mango_quantity

/-- Theorem: Andrew paid 975 to the shopkeeper -/
theorem andrew_payment : total_amount 74 6 59 9 = 975 := by
  sorry

end NUMINAMATH_CALUDE_andrew_payment_l580_58081


namespace NUMINAMATH_CALUDE_union_M_N_is_half_open_interval_l580_58027

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2}
def N : Set ℝ := {y | ∃ x, y = 2^x ∧ x < 0}

-- State the theorem
theorem union_M_N_is_half_open_interval :
  M ∪ N = Set.Icc 0 1 \ {1} :=
sorry

end NUMINAMATH_CALUDE_union_M_N_is_half_open_interval_l580_58027


namespace NUMINAMATH_CALUDE_remainder_1999_div_7_l580_58034

theorem remainder_1999_div_7 : 1999 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1999_div_7_l580_58034


namespace NUMINAMATH_CALUDE_potatoes_cooked_l580_58061

theorem potatoes_cooked (total : ℕ) (cooking_time : ℕ) (remaining_time : ℕ) : 
  total = 15 → 
  cooking_time = 8 → 
  remaining_time = 72 → 
  total - (remaining_time / cooking_time) = 6 := by
sorry

end NUMINAMATH_CALUDE_potatoes_cooked_l580_58061


namespace NUMINAMATH_CALUDE_two_eggs_remain_l580_58071

/-- The number of eggs remaining unsold when packaging a given number of eggs into cartons of a specific size -/
def remaining_eggs (debra_eggs eli_eggs fiona_eggs carton_size : ℕ) : ℕ :=
  (debra_eggs + eli_eggs + fiona_eggs) % carton_size

/-- Theorem stating that given the specified number of eggs and carton size, 2 eggs will remain unsold -/
theorem two_eggs_remain :
  remaining_eggs 45 58 19 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_eggs_remain_l580_58071


namespace NUMINAMATH_CALUDE_divisible_by_sixteen_l580_58092

theorem divisible_by_sixteen (m n : ℤ) : ∃ k : ℤ, (5*m + 3*n + 1)^5 * (3*m + n + 4)^4 = 16*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_sixteen_l580_58092


namespace NUMINAMATH_CALUDE_committee_selection_l580_58041

theorem committee_selection (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) :
  Nat.choose n k = 1140 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l580_58041


namespace NUMINAMATH_CALUDE_janet_number_problem_l580_58053

theorem janet_number_problem (x : ℝ) : ((x - 3) * 3 + 3) / 3 = 10 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_janet_number_problem_l580_58053


namespace NUMINAMATH_CALUDE_gcd_840_1764_l580_58038

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l580_58038


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l580_58033

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12 → x.val + y.val ≥ 49 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l580_58033


namespace NUMINAMATH_CALUDE_rumor_day_seven_l580_58044

def rumor_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

theorem rumor_day_seven :
  (∀ k < 7, rumor_spread k < 3280) ∧ rumor_spread 7 ≥ 3280 := by
  sorry

end NUMINAMATH_CALUDE_rumor_day_seven_l580_58044


namespace NUMINAMATH_CALUDE_system_solution_l580_58068

theorem system_solution (x y a : ℝ) : 
  x + 2 * y = a ∧ 
  x - 2 * y = 2 ∧ 
  x = 4 → 
  a = 6 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l580_58068


namespace NUMINAMATH_CALUDE_builder_project_l580_58065

/-- A builder's project involving bolts and nuts -/
theorem builder_project (bolt_boxes : ℕ) (bolts_per_box : ℕ) (nut_boxes : ℕ) (nuts_per_box : ℕ)
  (leftover_bolts : ℕ) (leftover_nuts : ℕ) :
  bolt_boxes = 7 →
  bolts_per_box = 11 →
  nut_boxes = 3 →
  nuts_per_box = 15 →
  leftover_bolts = 3 →
  leftover_nuts = 6 →
  (bolt_boxes * bolts_per_box - leftover_bolts) + (nut_boxes * nuts_per_box - leftover_nuts) = 113 :=
by sorry

end NUMINAMATH_CALUDE_builder_project_l580_58065


namespace NUMINAMATH_CALUDE_world_cup_investment_scientific_notation_l580_58006

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem world_cup_investment_scientific_notation :
  toScientificNotation 220000000000 = ScientificNotation.mk 2.2 11 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_world_cup_investment_scientific_notation_l580_58006


namespace NUMINAMATH_CALUDE_multiply_algebraic_expression_l580_58048

theorem multiply_algebraic_expression (a b : ℝ) : -3 * a * b * (2 * a) = -6 * a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_multiply_algebraic_expression_l580_58048


namespace NUMINAMATH_CALUDE_p_current_age_l580_58063

theorem p_current_age (p q : ℕ) : 
  (p - 3) / (q - 3) = 4 / 3 →
  (p + 6) / (q + 6) = 7 / 6 →
  p = 15 := by
sorry

end NUMINAMATH_CALUDE_p_current_age_l580_58063


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l580_58049

theorem arithmetic_geometric_mean_inequality {x y : ℝ} (hx : x ≥ 0) (hy : y ≥ 0) :
  (x + y) / 2 ≥ Real.sqrt (x * y) ∧
  ((x + y) / 2 = Real.sqrt (x * y) ↔ x = y) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l580_58049


namespace NUMINAMATH_CALUDE_quadratic_max_l580_58096

/-- The quadratic function f(x) = -2x^2 + 8x - 6 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

theorem quadratic_max :
  (∃ (x_max : ℝ), ∀ (x : ℝ), f x ≤ f x_max) ∧
  (∃ (x_max : ℝ), f x_max = 2) ∧
  (∀ (x : ℝ), f x = 2 → x = 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_l580_58096


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l580_58031

theorem trigonometric_equation_solution (x y : ℝ) : 
  x = π / 6 → 
  Real.sin x * Real.cos x * y - 2 * Real.sin x * Real.sin x * y + Real.cos x * y = 1/2 → 
  y = (6 * Real.sqrt 3 + 4) / 23 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l580_58031


namespace NUMINAMATH_CALUDE_quadratic_transformation_l580_58074

theorem quadratic_transformation (a b c : ℝ) (ha : a ≠ 0) :
  ∃ (h k r : ℝ) (hr : r ≠ 0), ∀ x : ℝ,
    a * x^2 + b * x + c = r^2 * ((x / r - h)^2) + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l580_58074


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l580_58083

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Random
  | Systematic
  | Stratified

/-- Represents a survey with its characteristics -/
structure Survey where
  total_population : ℕ
  strata : List ℕ
  sample_size : ℕ

/-- Determines the appropriate sampling method for a given survey -/
def appropriate_sampling_method (s : Survey) : SamplingMethod :=
  if s.strata.length > 1 then SamplingMethod.Stratified else SamplingMethod.Random

/-- The two surveys from the problem -/
def survey1 : Survey :=
  { total_population := 500
  , strata := [125, 280, 95]
  , sample_size := 100 }

def survey2 : Survey :=
  { total_population := 12
  , strata := [12]
  , sample_size := 3 }

/-- Theorem stating the correct sampling methods for the given surveys -/
theorem correct_sampling_methods :
  (appropriate_sampling_method survey1 = SamplingMethod.Stratified) ∧
  (appropriate_sampling_method survey2 = SamplingMethod.Random) := by
  sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l580_58083


namespace NUMINAMATH_CALUDE_first_term_to_common_difference_ratio_l580_58094

/-- An arithmetic progression where the sum of the first 14 terms is three times the sum of the first 7 terms -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_condition : (14 * a + 91 * d) = 3 * (7 * a + 21 * d)

/-- The ratio of the first term to the common difference is 4:1 -/
theorem first_term_to_common_difference_ratio 
  (ap : ArithmeticProgression) : ap.a / ap.d = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_term_to_common_difference_ratio_l580_58094


namespace NUMINAMATH_CALUDE_double_counted_page_l580_58008

theorem double_counted_page (n : ℕ) : 
  (n * (n + 1)) / 2 + 80 = 2550 → 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 
  (n * (n + 1)) / 2 + k = 2550 → 
  k = 80 := by
sorry

end NUMINAMATH_CALUDE_double_counted_page_l580_58008


namespace NUMINAMATH_CALUDE_continuous_fraction_equality_l580_58052

theorem continuous_fraction_equality : 1 + 2 / (3 + 6/7) = 41/27 := by
  sorry

end NUMINAMATH_CALUDE_continuous_fraction_equality_l580_58052


namespace NUMINAMATH_CALUDE_carrie_iphone_weeks_l580_58032

/-- Proves that Carrie needs to work 7 weeks to buy the iPhone -/
theorem carrie_iphone_weeks : 
  ∀ (iphone_cost trade_in_value weekly_earnings : ℕ),
    iphone_cost = 800 →
    trade_in_value = 240 →
    weekly_earnings = 80 →
    (iphone_cost - trade_in_value) / weekly_earnings = 7 := by
  sorry

end NUMINAMATH_CALUDE_carrie_iphone_weeks_l580_58032


namespace NUMINAMATH_CALUDE_monster_hunt_proof_l580_58095

/-- The sum of a geometric sequence with initial term 2, common ratio 2, and 5 terms -/
def monster_sum : ℕ := 
  List.range 5
  |> List.map (fun n => 2 * 2^n)
  |> List.sum

theorem monster_hunt_proof : monster_sum = 62 := by
  sorry

end NUMINAMATH_CALUDE_monster_hunt_proof_l580_58095


namespace NUMINAMATH_CALUDE_large_cube_pieces_l580_58097

/-- The number of wire pieces needed for a cube framework -/
def wire_pieces (n : ℕ) : ℕ := 3 * (n + 1)^2 * n

/-- The fact that a 2 × 2 × 2 cube uses 54 wire pieces -/
axiom small_cube_pieces : wire_pieces 2 = 54

/-- Theorem: The number of wire pieces needed for a 10 × 10 × 10 cube is 3630 -/
theorem large_cube_pieces : wire_pieces 10 = 3630 := by
  sorry

end NUMINAMATH_CALUDE_large_cube_pieces_l580_58097


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l580_58030

theorem range_of_a_minus_b (a b : ℝ) (ha : -2 < a ∧ a < 1) (hb : 0 < b ∧ b < 4) :
  ∀ x, (∃ y z, -2 < y ∧ y < 1 ∧ 0 < z ∧ z < 4 ∧ x = y - z) ↔ -6 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l580_58030


namespace NUMINAMATH_CALUDE_square_minus_one_divisible_by_three_l580_58091

theorem square_minus_one_divisible_by_three (n : ℕ) (h : ¬ 3 ∣ n) : 3 ∣ (n^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_square_minus_one_divisible_by_three_l580_58091


namespace NUMINAMATH_CALUDE_slope_interpretation_l580_58070

/-- Regression line equation for poverty and education data -/
def regression_line (x : ℝ) : ℝ := 0.8 * x + 4.6

/-- Theorem stating the relationship between changes in x and y -/
theorem slope_interpretation (x₁ x₂ : ℝ) (h : x₂ = x₁ + 1) :
  regression_line x₂ - regression_line x₁ = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_slope_interpretation_l580_58070


namespace NUMINAMATH_CALUDE_no_perfect_square_in_range_l580_58026

theorem no_perfect_square_in_range : 
  ¬ ∃ (n : ℕ), 4 ≤ n ∧ n ≤ 13 ∧ ∃ (m : ℕ), n^2 + n + 1 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_in_range_l580_58026


namespace NUMINAMATH_CALUDE_dog_weight_ratio_l580_58000

/-- Represents the weight of a dog at different ages --/
structure DogWeight where
  week7 : ℝ
  week9 : ℝ
  month3 : ℝ
  month5 : ℝ
  year1 : ℝ

/-- Theorem stating the ratio of a dog's weight at 9 weeks to 7 weeks --/
theorem dog_weight_ratio (w : DogWeight) 
  (h1 : w.week7 = 6)
  (h2 : w.month3 = 2 * w.week9)
  (h3 : w.month5 = 2 * w.month3)
  (h4 : w.year1 = w.month5 + 30)
  (h5 : w.year1 = 78) :
  w.week9 / w.week7 = 2 := by
  sorry

#check dog_weight_ratio

end NUMINAMATH_CALUDE_dog_weight_ratio_l580_58000


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l580_58018

theorem angle_measure_in_triangle (y : ℝ) : 
  let angle_ABC : ℝ := 180
  let angle_CBD : ℝ := 115
  let angle_BAD : ℝ := 31
  angle_ABC = 180 ∧ 
  angle_CBD = 115 ∧ 
  angle_BAD = 31 ∧
  y + angle_BAD + (angle_ABC - angle_CBD) = 180
  → y = 84 := by sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l580_58018


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_right_angle_point_isosceles_trapezoid_point_distances_l580_58084

/-- An isosceles trapezoid with bases a and b, and height h -/
structure IsoscelesTrapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  h_pos : 0 < h

/-- Point P on the axis of symmetry of the trapezoid -/
structure PointP (t : IsoscelesTrapezoid) where
  x : ℝ  -- Distance from P to one base
  y : ℝ  -- Distance from P to the other base
  sum_eq_h : x + y = t.h
  product_eq_ab_div_4 : x * y = t.a * t.b / 4

theorem isosceles_trapezoid_right_angle_point 
  (t : IsoscelesTrapezoid) : 
  (∃ p : PointP t, True) ↔ t.h^2 ≥ t.a * t.b :=
sorry

theorem isosceles_trapezoid_point_distances 
  (t : IsoscelesTrapezoid) 
  (h : t.h^2 ≥ t.a * t.b) :
  ∃ p : PointP t, 
    (p.x = (t.h + Real.sqrt (t.h^2 - t.a * t.b)) / 2 ∧ 
     p.y = (t.h - Real.sqrt (t.h^2 - t.a * t.b)) / 2) ∨
    (p.x = (t.h - Real.sqrt (t.h^2 - t.a * t.b)) / 2 ∧ 
     p.y = (t.h + Real.sqrt (t.h^2 - t.a * t.b)) / 2) :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_right_angle_point_isosceles_trapezoid_point_distances_l580_58084


namespace NUMINAMATH_CALUDE_coin_payment_difference_l580_58011

/-- Represents the available coin denominations in cents -/
inductive Coin : Type
  | OneCent : Coin
  | TenCent : Coin
  | TwentyCent : Coin

/-- The value of a coin in cents -/
def coin_value : Coin → ℕ
  | Coin.OneCent => 1
  | Coin.TenCent => 10
  | Coin.TwentyCent => 20

/-- A function that returns true if a list of coins sums to the target amount -/
def sum_to_target (coins : List Coin) (target : ℕ) : Prop :=
  (coins.map coin_value).sum = target

/-- The proposition to be proved -/
theorem coin_payment_difference (target : ℕ := 50) :
  ∃ (min_coins max_coins : List Coin),
    sum_to_target min_coins target ∧
    sum_to_target max_coins target ∧
    (max_coins.length - min_coins.length = 47) :=
  sorry

end NUMINAMATH_CALUDE_coin_payment_difference_l580_58011


namespace NUMINAMATH_CALUDE_ellipse_dimensions_l580_58009

/-- Given an ellipse and a parabola with specific properties, prove the dimensions of the ellipse. -/
theorem ellipse_dimensions (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (∀ x y, x^2 / m^2 + y^2 / n^2 = 1) →  -- Ellipse equation
  (∃ x₀, ∀ y, y^2 = 8*x₀ ∧ x₀ = 2) →   -- Parabola focus
  (let c := Real.sqrt (m^2 - n^2);
   c / m = 1 / 2) →                    -- Eccentricity
  m^2 = 16 ∧ n^2 = 12 := by
sorry

end NUMINAMATH_CALUDE_ellipse_dimensions_l580_58009


namespace NUMINAMATH_CALUDE_parabola_directrix_l580_58082

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2

-- Define the line containing the focus
def focus_line (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem parabola_directrix (a : ℝ) :
  (∃ x y, focus_line x y ∧ (x = 0 ∨ y = 0)) →
  (∃ x, ∀ y, y = parabola a x ↔ y + 1 = 2 * (parabola a (x/2))) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l580_58082


namespace NUMINAMATH_CALUDE_exact_defective_selection_l580_58017

def total_products : ℕ := 100
def defective_products : ℕ := 3
def products_to_select : ℕ := 4
def defective_to_select : ℕ := 2

theorem exact_defective_selection :
  (Nat.choose defective_products defective_to_select) *
  (Nat.choose (total_products - defective_products) (products_to_select - defective_to_select)) = 13968 := by
  sorry

end NUMINAMATH_CALUDE_exact_defective_selection_l580_58017


namespace NUMINAMATH_CALUDE_function_existence_l580_58073

theorem function_existence (k : ℤ) (hk : k ≠ 0) :
  ∃ f : ℤ → ℤ, ∀ a b : ℤ, k * (f (a + b)) + f (a * b) = f a * f b + k :=
by sorry

end NUMINAMATH_CALUDE_function_existence_l580_58073


namespace NUMINAMATH_CALUDE_zero_of_f_l580_58054

def f (x : ℝ) : ℝ := 4 * x - 2

theorem zero_of_f : ∃ x : ℝ, f x = 0 ∧ x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_zero_of_f_l580_58054


namespace NUMINAMATH_CALUDE_maximal_arithmetic_progression_1996_maximal_arithmetic_progression_1997_l580_58079

/-- The set of reciprocals of natural numbers -/
def S : Set ℚ := {q : ℚ | ∃ n : ℕ, q = 1 / n}

/-- An arithmetic progression in S -/
def is_arithmetic_progression (a : ℕ → ℚ) (n : ℕ) : Prop :=
  ∃ (first d : ℚ), ∀ i < n, a i = first + i • d ∧ a i ∈ S

/-- A maximal arithmetic progression in S -/
def is_maximal_arithmetic_progression (a : ℕ → ℚ) (n : ℕ) : Prop :=
  is_arithmetic_progression a n ∧
  ¬∃ (b : ℕ → ℚ) (m : ℕ), m > n ∧ is_arithmetic_progression b m ∧
    (∀ i < n, a i = b i)

theorem maximal_arithmetic_progression_1996 :
  ∃ (a : ℕ → ℚ), is_maximal_arithmetic_progression a 1996 :=
sorry

theorem maximal_arithmetic_progression_1997 :
  ∃ (a : ℕ → ℚ), is_maximal_arithmetic_progression a 1997 :=
sorry

end NUMINAMATH_CALUDE_maximal_arithmetic_progression_1996_maximal_arithmetic_progression_1997_l580_58079


namespace NUMINAMATH_CALUDE_apple_lovers_l580_58090

structure FruitPreferences where
  total : ℕ
  apple : ℕ
  orange : ℕ
  mango : ℕ
  banana : ℕ
  grapes : ℕ
  orange_mango_not_apple : ℕ
  mango_apple_not_orange : ℕ
  all_three : ℕ
  banana_grapes_only : ℕ
  apple_banana_grapes_not_others : ℕ

def room : FruitPreferences := {
  total := 60,
  apple := 40,
  orange := 17,
  mango := 23,
  banana := 12,
  grapes := 9,
  orange_mango_not_apple := 7,
  mango_apple_not_orange := 10,
  all_three := 4,
  banana_grapes_only := 6,
  apple_banana_grapes_not_others := 3
}

theorem apple_lovers (pref : FruitPreferences) : pref.apple = 40 :=
  sorry

end NUMINAMATH_CALUDE_apple_lovers_l580_58090


namespace NUMINAMATH_CALUDE_andrew_worked_300_days_l580_58088

/-- Represents the company's vacation policy and Andrew's vacation usage --/
structure VacationData where
  /-- The number of work days required to earn one vacation day --/
  work_days_per_vacation_day : ℕ
  /-- Vacation days taken in March --/
  march_vacation : ℕ
  /-- Vacation days taken in September --/
  september_vacation : ℕ
  /-- Remaining vacation days --/
  remaining_vacation : ℕ

/-- Calculates the total number of days worked given the vacation data --/
def days_worked (data : VacationData) : ℕ :=
  sorry

/-- Theorem stating that given the specific vacation data, Andrew worked 300 days --/
theorem andrew_worked_300_days : 
  let data : VacationData := {
    work_days_per_vacation_day := 10,
    march_vacation := 5,
    september_vacation := 10,
    remaining_vacation := 15
  }
  days_worked data = 300 := by
  sorry

end NUMINAMATH_CALUDE_andrew_worked_300_days_l580_58088


namespace NUMINAMATH_CALUDE_sphere_volume_after_drilling_l580_58057

/-- The remaining volume of a sphere after drilling cylindrical holes -/
theorem sphere_volume_after_drilling (sphere_diameter : ℝ) 
  (hole1_depth hole1_diameter : ℝ) 
  (hole2_depth hole2_diameter : ℝ) 
  (hole3_depth hole3_diameter : ℝ) : 
  sphere_diameter = 24 →
  hole1_depth = 10 → hole1_diameter = 3 →
  hole2_depth = 10 → hole2_diameter = 3 →
  hole3_depth = 5 → hole3_diameter = 4 →
  (4 / 3 * π * (sphere_diameter / 2)^3) - 
  (π * (hole1_diameter / 2)^2 * hole1_depth) - 
  (π * (hole2_diameter / 2)^2 * hole2_depth) - 
  (π * (hole3_diameter / 2)^2 * hole3_depth) = 2239 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_after_drilling_l580_58057


namespace NUMINAMATH_CALUDE_debate_team_formations_l580_58080

def num_boys : ℕ := 3
def num_girls : ℕ := 3
def num_debaters : ℕ := 4
def boy_a_exists : Prop := true

theorem debate_team_formations :
  (num_boys + num_girls - 1) * (num_boys + num_girls - 1) * (num_boys + num_girls - 2) * (num_boys + num_girls - 3) = 300 :=
by sorry

end NUMINAMATH_CALUDE_debate_team_formations_l580_58080


namespace NUMINAMATH_CALUDE_order_of_integrals_l580_58029

theorem order_of_integrals : 
  let a : ℝ := ∫ x in (0:ℝ)..2, x^2
  let b : ℝ := ∫ x in (0:ℝ)..2, x^3
  let c : ℝ := ∫ x in (0:ℝ)..2, Real.sin x
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_order_of_integrals_l580_58029


namespace NUMINAMATH_CALUDE_nine_integer_segments_l580_58037

/-- Right triangle XYZ with integer leg lengths -/
structure RightTriangle where
  xy : ℕ
  yz : ℕ

/-- The number of different integer length line segments from Y to XZ -/
def countIntegerSegments (t : RightTriangle) : ℕ :=
  sorry

/-- The specific right triangle with XY = 15 and YZ = 20 -/
def specialTriangle : RightTriangle :=
  { xy := 15, yz := 20 }

/-- Theorem stating that the number of integer length segments is 9 -/
theorem nine_integer_segments :
  countIntegerSegments specialTriangle = 9 :=
sorry

end NUMINAMATH_CALUDE_nine_integer_segments_l580_58037


namespace NUMINAMATH_CALUDE_milk_water_ratio_after_addition_l580_58013

/-- Calculates the final ratio of milk to water after adding water to a mixture -/
theorem milk_water_ratio_after_addition
  (initial_volume : ℚ)
  (initial_milk_ratio : ℚ)
  (initial_water_ratio : ℚ)
  (added_water : ℚ)
  (h1 : initial_volume = 45)
  (h2 : initial_milk_ratio = 4)
  (h3 : initial_water_ratio = 1)
  (h4 : added_water = 21) :
  let initial_milk := initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)
  let initial_water := initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)
  let final_water := initial_water + added_water
  let final_milk_ratio := initial_milk / final_water
  let final_water_ratio := final_water / final_water
  (final_milk_ratio : ℚ) / (final_water_ratio : ℚ) = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_milk_water_ratio_after_addition_l580_58013


namespace NUMINAMATH_CALUDE_lewis_earnings_l580_58051

/-- Lewis's earnings during harvest season --/
theorem lewis_earnings (weekly_earnings weekly_rent : ℕ) (harvest_weeks : ℕ) : 
  weekly_earnings = 403 → 
  weekly_rent = 49 → 
  harvest_weeks = 233 → 
  (weekly_earnings * harvest_weeks) - (weekly_rent * harvest_weeks) = 82482 := by
  sorry

end NUMINAMATH_CALUDE_lewis_earnings_l580_58051


namespace NUMINAMATH_CALUDE_min_rubles_to_win_l580_58014

/-- Represents the state of the game --/
structure GameState :=
  (score : ℕ)
  (rubles_spent : ℕ)

/-- The rules of the game --/
def apply_rule (state : GameState) (coin : ℕ) : GameState :=
  match coin with
  | 1 => ⟨state.score + 1, state.rubles_spent + 1⟩
  | 2 => ⟨state.score * 2, state.rubles_spent + 2⟩
  | _ => state

/-- Check if the game is won --/
def is_won (state : GameState) : Bool :=
  state.score = 50

/-- Check if the game is lost --/
def is_lost (state : GameState) : Bool :=
  state.score > 50

/-- The main theorem to prove --/
theorem min_rubles_to_win :
  ∃ (sequence : List ℕ),
    let final_state := sequence.foldl apply_rule ⟨0, 0⟩
    is_won final_state ∧
    final_state.rubles_spent = 11 ∧
    (∀ (other_sequence : List ℕ),
      let other_final_state := other_sequence.foldl apply_rule ⟨0, 0⟩
      is_won other_final_state →
      other_final_state.rubles_spent ≥ 11) :=
by sorry

end NUMINAMATH_CALUDE_min_rubles_to_win_l580_58014


namespace NUMINAMATH_CALUDE_summer_mowing_times_l580_58098

/-- The number of times Kale mowed his lawn in the summer -/
def summer_mowing : ℕ := 5

/-- The number of times Kale mowed his lawn in the spring -/
def spring_mowing : ℕ := 8

/-- The difference between spring and summer mowing times -/
def mowing_difference : ℕ := 3

theorem summer_mowing_times : 
  spring_mowing - summer_mowing = mowing_difference := by sorry

end NUMINAMATH_CALUDE_summer_mowing_times_l580_58098


namespace NUMINAMATH_CALUDE_solution_set_of_inequalities_l580_58050

theorem solution_set_of_inequalities :
  let S := { x : ℝ | x - 2 > 1 ∧ x < 4 }
  S = { x : ℝ | 3 < x ∧ x < 4 } := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequalities_l580_58050


namespace NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l580_58012

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on x and y axes
def equalIntercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ (-l.c / l.a = -l.c / l.b)

-- The main theorem
theorem line_through_point_equal_intercepts :
  ∃ (l₁ l₂ : Line),
    (pointOnLine ⟨3, -2⟩ l₁) ∧
    (pointOnLine ⟨3, -2⟩ l₂) ∧
    (equalIntercepts l₁ ∨ (l₁.a = 0 ∧ l₁.b = 0)) ∧
    (equalIntercepts l₂ ∨ (l₂.a = 0 ∧ l₂.b = 0)) ∧
    ((l₁.a = 2 ∧ l₁.b = 3 ∧ l₁.c = 0) ∨ (l₂.a = 1 ∧ l₂.b = 1 ∧ l₂.c = -1)) :=
  sorry

end NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l580_58012


namespace NUMINAMATH_CALUDE_no_rain_probability_l580_58039

theorem no_rain_probability (p : ℚ) (h : p = 2/3) : (1 - p)^4 = 1/81 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l580_58039


namespace NUMINAMATH_CALUDE_minimum_sum_of_parameters_l580_58086

theorem minimum_sum_of_parameters (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 / a + 1 / b = 1) → (a + b ≥ 4) ∧ (∃ a b, 1 / a + 1 / b = 1 ∧ a + b = 4) :=
sorry

end NUMINAMATH_CALUDE_minimum_sum_of_parameters_l580_58086


namespace NUMINAMATH_CALUDE_margies_change_l580_58056

/-- Calculates the change Margie receives after buying oranges -/
def margieChange (numOranges : ℕ) (costPerOrange : ℚ) (amountPaid : ℚ) : ℚ :=
  amountPaid - (numOranges : ℚ) * costPerOrange

/-- Theorem stating that Margie's change is $8.50 -/
theorem margies_change :
  margieChange 5 (30 / 100) 10 = 17 / 2 := by
  sorry

#eval margieChange 5 (30 / 100) 10

end NUMINAMATH_CALUDE_margies_change_l580_58056


namespace NUMINAMATH_CALUDE_division_problem_l580_58010

theorem division_problem : ∃ A : ℕ, 23 = 6 * A + 5 ∧ A = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l580_58010


namespace NUMINAMATH_CALUDE_total_paintable_area_l580_58028

def bedroom_type1_length : ℝ := 14
def bedroom_type1_width : ℝ := 11
def bedroom_type1_height : ℝ := 9
def bedroom_type2_length : ℝ := 13
def bedroom_type2_width : ℝ := 12
def bedroom_type2_height : ℝ := 9
def num_bedrooms : ℕ := 4
def unpaintable_area : ℝ := 70

def wall_area (length width height : ℝ) : ℝ :=
  2 * (length * height + width * height)

def paintable_area (total_area unpaintable_area : ℝ) : ℝ :=
  total_area - unpaintable_area

theorem total_paintable_area :
  let type1_area := wall_area bedroom_type1_length bedroom_type1_width bedroom_type1_height
  let type2_area := wall_area bedroom_type2_length bedroom_type2_width bedroom_type2_height
  let total_area := (num_bedrooms / 2) * (paintable_area type1_area unpaintable_area + 
                                          paintable_area type2_area unpaintable_area)
  total_area = 1520 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_l580_58028


namespace NUMINAMATH_CALUDE_postage_for_5_5_ounces_l580_58025

/-- Calculates the postage for a letter given its weight and the rate structure -/
def calculatePostage (weight : ℚ) : ℚ :=
  let baseRate : ℚ := 25 / 100  -- 25 cents
  let additionalRate : ℚ := 18 / 100  -- 18 cents
  let overweightSurcharge : ℚ := 10 / 100  -- 10 cents
  let overweightThreshold : ℚ := 3  -- 3 ounces
  
  let additionalWeight := max (weight - 1) 0
  let additionalCharges := ⌈additionalWeight⌉
  
  let cost := baseRate + additionalRate * additionalCharges
  if weight > overweightThreshold then
    cost + overweightSurcharge
  else
    cost

/-- Theorem stating that the postage for a 5.5 ounce letter is $1.25 -/
theorem postage_for_5_5_ounces :
  calculatePostage (11/2) = 5/4 := by sorry

end NUMINAMATH_CALUDE_postage_for_5_5_ounces_l580_58025


namespace NUMINAMATH_CALUDE_complex_distance_sum_constant_l580_58058

theorem complex_distance_sum_constant (w : ℂ) (h : Complex.abs (w - (3 + 2*I)) = 3) :
  Complex.abs (w - (2 - 3*I))^2 + Complex.abs (w - (4 + 5*I))^2 = 71 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_sum_constant_l580_58058


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l580_58072

theorem trigonometric_equation_solution (x : ℝ) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (2 * Real.cos (5 * x) + 2 * Real.cos (4 * x) + 2 * Real.cos (3 * x) +
   2 * Real.cos (2 * x) + 2 * Real.cos x + 1 = 0) ↔
  (∃ k : ℕ, k ∈ Finset.range 10 ∧ x = 2 * k * Real.pi / 11) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l580_58072


namespace NUMINAMATH_CALUDE_julia_jonny_stairs_fraction_l580_58060

theorem julia_jonny_stairs_fraction (jonny_stairs : ℕ) (total_stairs : ℕ) 
  (h1 : jonny_stairs = 1269)
  (h2 : total_stairs = 1685) :
  (total_stairs - jonny_stairs : ℚ) / jonny_stairs = 416 / 1269 := by
  sorry

end NUMINAMATH_CALUDE_julia_jonny_stairs_fraction_l580_58060


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l580_58077

theorem isosceles_right_triangle_hypotenuse (a c : ℝ) : 
  a > 0 → 
  c > 0 → 
  c^2 = 2 * a^2 →  -- isosceles right triangle condition
  a^2 + a^2 + c^2 = 1452 →  -- sum of squares condition
  c = Real.sqrt 726 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l580_58077


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l580_58046

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | x < 0}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l580_58046


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l580_58099

theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a * b) / (a * b + a^5 + b^5) + (b * c) / (b * c + b^5 + c^5) + (c * a) / (c * a + c^5 + a^5) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l580_58099


namespace NUMINAMATH_CALUDE_cookie_boxes_problem_l580_58019

theorem cookie_boxes_problem (type1_per_box type3_per_box : ℕ)
  (type1_boxes type2_boxes type3_boxes : ℕ)
  (total_cookies : ℕ)
  (h1 : type1_per_box = 12)
  (h2 : type3_per_box = 16)
  (h3 : type1_boxes = 50)
  (h4 : type2_boxes = 80)
  (h5 : type3_boxes = 70)
  (h6 : total_cookies = 3320)
  (h7 : type1_per_box * type1_boxes + type2_boxes * type2_per_box + type3_per_box * type3_boxes = total_cookies) :
  type2_per_box = 20 := by
  sorry


end NUMINAMATH_CALUDE_cookie_boxes_problem_l580_58019


namespace NUMINAMATH_CALUDE_path_length_eq_three_times_PQ_l580_58040

/-- The length of the segment PQ -/
def PQ_length : ℝ := 73

/-- The length of the path along the squares constructed on the segments of PQ -/
def path_length : ℝ := 3 * PQ_length

theorem path_length_eq_three_times_PQ : path_length = 3 * PQ_length := by
  sorry

end NUMINAMATH_CALUDE_path_length_eq_three_times_PQ_l580_58040


namespace NUMINAMATH_CALUDE_defective_shipped_percentage_l580_58022

theorem defective_shipped_percentage 
  (defective_rate : Real) 
  (shipped_rate : Real) 
  (h1 : defective_rate = 0.08) 
  (h2 : shipped_rate = 0.04) : 
  defective_rate * shipped_rate = 0.0032 := by
  sorry

#check defective_shipped_percentage

end NUMINAMATH_CALUDE_defective_shipped_percentage_l580_58022


namespace NUMINAMATH_CALUDE_mn_max_and_m2n2_min_l580_58085

/-- Given real numbers m and n, where m > 0, n > 0, and 2m + n = 1,
    prove that the maximum value of mn is 1/8 and
    the minimum value of 4m^2 + n^2 is 1/2 -/
theorem mn_max_and_m2n2_min (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) :
  (∀ x y, x > 0 → y > 0 → 2 * x + y = 1 → m * n ≥ x * y) ∧
  (∀ x y, x > 0 → y > 0 → 2 * x + y = 1 → 4 * m^2 + n^2 ≤ 4 * x^2 + y^2) ∧
  m * n = 1/8 ∧ 4 * m^2 + n^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_mn_max_and_m2n2_min_l580_58085


namespace NUMINAMATH_CALUDE_total_problems_is_550_l580_58024

/-- The total number of math problems practiced by Marvin, Arvin, and Kevin over two days -/
def totalProblems (marvinYesterday : ℕ) : ℕ :=
  let marvinToday := 3 * marvinYesterday
  let arvinYesterday := 2 * marvinYesterday
  let arvinToday := 2 * marvinToday
  let kevinYesterday := 30
  let kevinToday := kevinYesterday + 10
  (marvinYesterday + marvinToday) + (arvinYesterday + arvinToday) + (kevinYesterday + kevinToday)

/-- Theorem stating that the total number of problems practiced is 550 -/
theorem total_problems_is_550 : totalProblems 40 = 550 := by
  sorry

end NUMINAMATH_CALUDE_total_problems_is_550_l580_58024


namespace NUMINAMATH_CALUDE_smallest_d_for_g_range_three_l580_58021

/-- The function g(x) defined as x^2 + 4x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

/-- Theorem stating that 7 is the smallest value of d for which 3 is in the range of g(x) -/
theorem smallest_d_for_g_range_three :
  (∃ (d : ℝ), (∃ (x : ℝ), g d x = 3) ∧ (∀ (d' : ℝ), d' < d → ¬∃ (x : ℝ), g d' x = 3)) ∧
  (∃ (x : ℝ), g 7 x = 3) :=
sorry

end NUMINAMATH_CALUDE_smallest_d_for_g_range_three_l580_58021


namespace NUMINAMATH_CALUDE_subtraction_to_sum_equality_l580_58076

theorem subtraction_to_sum_equality : 3 - 10 - 7 = 3 + (-10) + (-7) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_to_sum_equality_l580_58076


namespace NUMINAMATH_CALUDE_container_volume_ratio_l580_58003

theorem container_volume_ratio : 
  ∀ (volume_container1 volume_container2 : ℚ),
  volume_container1 > 0 →
  volume_container2 > 0 →
  (4 / 5 : ℚ) * volume_container1 = (2 / 3 : ℚ) * volume_container2 →
  volume_container1 / volume_container2 = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l580_58003


namespace NUMINAMATH_CALUDE_grunters_win_probability_l580_58035

theorem grunters_win_probability (num_games : ℕ) (win_prob : ℚ) :
  num_games = 6 →
  win_prob = 3/5 →
  (win_prob ^ num_games : ℚ) = 729/15625 := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l580_58035


namespace NUMINAMATH_CALUDE_interest_rate_is_ten_percent_l580_58078

/-- The interest rate at which A lent money to B, given the conditions of the problem -/
def interest_rate_A_to_B (principal : ℚ) (rate_B_to_C : ℚ) (time : ℚ) (B_gain : ℚ) : ℚ :=
  let interest_from_C := principal * rate_B_to_C * time
  let interest_to_A := interest_from_C - B_gain
  (interest_to_A / (principal * time)) * 100

/-- Theorem stating that the interest rate from A to B is 10% under the given conditions -/
theorem interest_rate_is_ten_percent :
  interest_rate_A_to_B 3500 0.13 3 315 = 10 := by sorry

end NUMINAMATH_CALUDE_interest_rate_is_ten_percent_l580_58078


namespace NUMINAMATH_CALUDE_april_plant_arrangement_l580_58004

/-- The number of ways to arrange plants with specific conditions -/
def plant_arrangements (n_basil : ℕ) (n_tomato : ℕ) : ℕ :=
  (n_basil + n_tomato - 1).factorial * (n_basil - 1).factorial

/-- Theorem stating the number of arrangements for the given problem -/
theorem april_plant_arrangement :
  plant_arrangements 5 3 = 576 := by
  sorry

end NUMINAMATH_CALUDE_april_plant_arrangement_l580_58004
