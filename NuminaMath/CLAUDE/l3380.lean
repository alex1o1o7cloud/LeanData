import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l3380_338021

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x + 36 / (x - 3)
  {x : ℝ | f x = -12} = {0, -9} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3380_338021


namespace NUMINAMATH_CALUDE_shekar_average_marks_l3380_338080

def shekar_marks : List ℕ := [76, 65, 82, 67, 85]

theorem shekar_average_marks :
  (shekar_marks.sum / shekar_marks.length : ℚ) = 75 := by
  sorry

end NUMINAMATH_CALUDE_shekar_average_marks_l3380_338080


namespace NUMINAMATH_CALUDE_last_three_sum_l3380_338007

theorem last_three_sum (a : Fin 7 → ℝ) 
  (h1 : (a 0 + a 1 + a 2 + a 3) / 4 = 13)
  (h2 : (a 3 + a 4 + a 5 + a 6) / 4 = 15)
  (h3 : a 3 ^ 2 = a 6)
  (h4 : a 6 = 25) :
  a 4 + a 5 + a 6 = 55 := by
sorry

end NUMINAMATH_CALUDE_last_three_sum_l3380_338007


namespace NUMINAMATH_CALUDE_catherine_caps_proof_l3380_338032

/-- The number of bottle caps Nicholas starts with -/
def initial_caps : ℕ := 8

/-- The number of bottle caps Nicholas ends up with -/
def final_caps : ℕ := 93

/-- The number of bottle caps Catherine gave to Nicholas -/
def catherine_caps : ℕ := final_caps - initial_caps

theorem catherine_caps_proof : catherine_caps = 85 := by
  sorry

end NUMINAMATH_CALUDE_catherine_caps_proof_l3380_338032


namespace NUMINAMATH_CALUDE_exam_analysis_theorem_l3380_338075

structure StatisticalAnalysis where
  population_size : ℕ
  sample_size : ℕ
  sample_is_subset : sample_size ≤ population_size

def is_population (sa : StatisticalAnalysis) (n : ℕ) : Prop :=
  n = sa.population_size

def is_sample (sa : StatisticalAnalysis) (n : ℕ) : Prop :=
  n = sa.sample_size

def is_sample_size (sa : StatisticalAnalysis) (n : ℕ) : Prop :=
  n = sa.sample_size

-- The statement we want to prove incorrect
def each_examinee_is_individual_unit (sa : StatisticalAnalysis) : Prop :=
  False  -- This is set to False to represent that the statement is incorrect

theorem exam_analysis_theorem (sa : StatisticalAnalysis) 
  (h_pop : sa.population_size = 13000)
  (h_sample : sa.sample_size = 500) :
  is_population sa 13000 ∧ 
  is_sample sa 500 ∧ 
  is_sample_size sa 500 ∧ 
  ¬(each_examinee_is_individual_unit sa) := by
  sorry

#check exam_analysis_theorem

end NUMINAMATH_CALUDE_exam_analysis_theorem_l3380_338075


namespace NUMINAMATH_CALUDE_farmer_potatoes_l3380_338001

theorem farmer_potatoes (initial_tomatoes picked_tomatoes total_left : ℕ) 
  (h1 : initial_tomatoes = 177)
  (h2 : picked_tomatoes = 53)
  (h3 : total_left = 136) :
  initial_tomatoes - picked_tomatoes + (total_left - (initial_tomatoes - picked_tomatoes)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_farmer_potatoes_l3380_338001


namespace NUMINAMATH_CALUDE_limit_point_theorem_l3380_338069

def is_limit_point (X : Set ℝ) (x₀ : ℝ) : Prop :=
  ∀ a > 0, ∃ x ∈ X, 0 < |x - x₀| ∧ |x - x₀| < a

def set1 : Set ℝ := {x | ∃ n : ℤ, n ≥ 0 ∧ x = n / (n + 1)}
def set2 : Set ℝ := {x : ℝ | x ≠ 0}
def set3 : Set ℝ := {x | ∃ n : ℤ, n ≠ 0 ∧ x = 1 / n}
def set4 : Set ℝ := {x : ℝ | ∃ n : ℤ, x = n}

theorem limit_point_theorem :
  ¬(is_limit_point set1 0) ∧
  (is_limit_point set2 0) ∧
  (is_limit_point set3 0) ∧
  ¬(is_limit_point set4 0) := by sorry

end NUMINAMATH_CALUDE_limit_point_theorem_l3380_338069


namespace NUMINAMATH_CALUDE_reasoning_is_inductive_l3380_338015

/-- Represents different types of reasoning methods -/
inductive ReasoningMethod
  | Analogical
  | Inductive
  | Deductive
  | Analytical

/-- Represents a metal -/
structure Metal where
  name : String

/-- Represents the property of conducting electricity -/
def conductsElectricity (m : Metal) : Prop := sorry

/-- The set of metals mentioned in the statement -/
def mentionedMetals : List Metal := [
  { name := "Gold" },
  { name := "Silver" },
  { name := "Copper" },
  { name := "Iron" }
]

/-- The statement that all mentioned metals conduct electricity -/
def allMentionedMetalsConduct : Prop :=
  ∀ m ∈ mentionedMetals, conductsElectricity m

/-- The conclusion that all metals conduct electricity -/
def allMetalsConduct : Prop :=
  ∀ m : Metal, conductsElectricity m

/-- The reasoning method used in the given statement -/
def reasoningMethodUsed : ReasoningMethod := sorry

/-- Theorem stating that the reasoning method used is inductive -/
theorem reasoning_is_inductive :
  allMentionedMetalsConduct →
  reasoningMethodUsed = ReasoningMethod.Inductive :=
sorry

end NUMINAMATH_CALUDE_reasoning_is_inductive_l3380_338015


namespace NUMINAMATH_CALUDE_smallest_a_satisfying_equation_l3380_338024

theorem smallest_a_satisfying_equation :
  ∃ a : ℝ, (a = -Real.sqrt (62/5)) ∧
    (∀ b : ℝ, (8*Real.sqrt ((3*b)^2 + 2^2) - 5*b^2 - 2) / (Real.sqrt (2 + 5*b^2) + 4) = 3 → a ≤ b) ∧
    (8*Real.sqrt ((3*a)^2 + 2^2) - 5*a^2 - 2) / (Real.sqrt (2 + 5*a^2) + 4) = 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_satisfying_equation_l3380_338024


namespace NUMINAMATH_CALUDE_ali_spending_ratio_l3380_338063

theorem ali_spending_ratio :
  ∀ (initial_amount food_cost glasses_cost remaining : ℕ),
  initial_amount = 480 →
  glasses_cost = (initial_amount - food_cost) / 3 →
  remaining = initial_amount - food_cost - glasses_cost →
  remaining = 160 →
  food_cost * 2 = initial_amount :=
λ initial_amount food_cost glasses_cost remaining
  h_initial h_glasses h_remaining h_final =>
sorry

end NUMINAMATH_CALUDE_ali_spending_ratio_l3380_338063


namespace NUMINAMATH_CALUDE_office_paper_duration_l3380_338088

/-- The number of days printer paper will last given the number of packs, sheets per pack, and daily usage. -/
def printer_paper_duration (packs : ℕ) (sheets_per_pack : ℕ) (daily_usage : ℕ) : ℕ :=
  (packs * sheets_per_pack) / daily_usage

/-- Theorem stating that two packs of 240-sheet paper will last 6 days when using 80 sheets per day. -/
theorem office_paper_duration :
  printer_paper_duration 2 240 80 = 6 := by
  sorry

end NUMINAMATH_CALUDE_office_paper_duration_l3380_338088


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_72_8_l3380_338066

/-- Calculates the interval between segments for systematic sampling -/
def systematicSamplingInterval (populationSize : ℕ) (sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The systematic sampling interval for 72 students with a sample size of 8 is 9 -/
theorem systematic_sampling_interval_72_8 :
  systematicSamplingInterval 72 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_72_8_l3380_338066


namespace NUMINAMATH_CALUDE_sequence_properties_l3380_338018

def sequence_a (n : ℕ+) : ℚ := (1 / 3) ^ n.val

def sum_S (n : ℕ+) : ℚ := (1 / 2) * (1 - (1 / 3) ^ n.val)

def arithmetic_sequence_condition (t : ℚ) : Prop :=
  let S₁ := sum_S 1
  let S₂ := sum_S 2
  let S₃ := sum_S 3
  S₁ + 3 * (S₂ + S₃) = 2 * (S₁ + S₂) * t

theorem sequence_properties :
  (∀ n : ℕ+, sum_S (n + 1) - sum_S n = (1 / 3) ^ (n + 1).val) →
  (∀ n : ℕ+, sequence_a n = (1 / 3) ^ n.val) ∧
  (∀ n : ℕ+, sum_S n = (1 / 2) * (1 - (1 / 3) ^ n.val)) ∧
  (∃ t : ℚ, arithmetic_sequence_condition t ∧ t = 2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l3380_338018


namespace NUMINAMATH_CALUDE_sqrt2_minus_2_properties_l3380_338028

theorem sqrt2_minus_2_properties :
  let x : ℝ := Real.sqrt 2 - 2
  (- x = 2 - Real.sqrt 2) ∧ (|x| = 2 - Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_sqrt2_minus_2_properties_l3380_338028


namespace NUMINAMATH_CALUDE_inequality_proof_l3380_338025

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3380_338025


namespace NUMINAMATH_CALUDE_point_count_on_curve_l3380_338058

theorem point_count_on_curve : 
  ∃! (points : Finset (ℤ × ℤ)), 
    points.card = 6 ∧ 
    ∀ p : ℤ × ℤ, p ∈ points ↔ 
      let m := p.1
      let n := p.2
      n^2 = (m^2 - 4) * (m^2 + 12*m + 32) + 4 := by
  sorry

end NUMINAMATH_CALUDE_point_count_on_curve_l3380_338058


namespace NUMINAMATH_CALUDE_square_less_than_triple_l3380_338048

theorem square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_triple_l3380_338048


namespace NUMINAMATH_CALUDE_valid_arrangements_five_people_l3380_338089

/-- The number of people in the arrangement -/
def n : ℕ := 5

/-- The number of ways to arrange n people such that at least one of two specific people (A and B) is at one of the ends -/
def validArrangements (n : ℕ) : ℕ :=
  n.factorial - (n - 2).factorial * (n - 2).factorial

theorem valid_arrangements_five_people :
  validArrangements n = 84 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_five_people_l3380_338089


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3380_338060

/-- A line defined by the equation (m-1)x-y+2m+1=0 for any real number m -/
def line (m : ℝ) (x y : ℝ) : Prop :=
  (m - 1) * x - y + 2 * m + 1 = 0

/-- The fixed point (-2, 3) -/
def fixed_point : ℝ × ℝ := (-2, 3)

/-- Theorem stating that the line passes through the fixed point for any real m -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line m (fixed_point.1) (fixed_point.2) :=
by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3380_338060


namespace NUMINAMATH_CALUDE_homologous_functions_count_l3380_338064

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the range
def range : Set ℝ := {1, 3}

-- Define a valid domain
def is_valid_domain (D : Set ℝ) : Prop :=
  (∀ x ∈ D, f x ∈ range) ∧ (∀ y ∈ range, ∃ x ∈ D, f x = y)

-- Theorem statement
theorem homologous_functions_count :
  ∃! (domains : Finset (Set ℝ)), 
    domains.card = 3 ∧ 
    (∀ D ∈ domains, is_valid_domain D) ∧
    (∀ D : Set ℝ, is_valid_domain D → D ∈ domains) :=
sorry

end NUMINAMATH_CALUDE_homologous_functions_count_l3380_338064


namespace NUMINAMATH_CALUDE_perimeter_area_bisector_coincide_l3380_338092

/-- An isosceles triangle with side lengths 5, 5, and 6 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : a = b ∧ a = 5 ∧ c = 6

/-- A line bisecting the perimeter of the triangle -/
def perimeterBisector (t : IsoscelesTriangle) : Set (ℝ × ℝ) :=
  sorry

/-- A line bisecting the area of the triangle -/
def areaBisector (t : IsoscelesTriangle) : Set (ℝ × ℝ) :=
  sorry

/-- Theorem stating that the perimeter bisector coincides with the area bisector -/
theorem perimeter_area_bisector_coincide (t : IsoscelesTriangle) :
  perimeterBisector t = areaBisector t :=
sorry

end NUMINAMATH_CALUDE_perimeter_area_bisector_coincide_l3380_338092


namespace NUMINAMATH_CALUDE_exactly_one_even_l3380_338095

theorem exactly_one_even (a b c : ℕ) : 
  (a % 2 = 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0) ∨ 
  (a % 2 ≠ 0 ∧ b % 2 = 0 ∧ c % 2 ≠ 0) ∨ 
  (a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 = 0) :=
by
  sorry

#check exactly_one_even

end NUMINAMATH_CALUDE_exactly_one_even_l3380_338095


namespace NUMINAMATH_CALUDE_baseball_cost_value_l3380_338082

/-- The amount Mike spent on toys -/
def total_spent : ℚ := 20.52

/-- The cost of marbles -/
def marbles_cost : ℚ := 9.05

/-- The cost of the football -/
def football_cost : ℚ := 4.95

/-- The cost of the baseball -/
def baseball_cost : ℚ := total_spent - (marbles_cost + football_cost)

theorem baseball_cost_value : baseball_cost = 6.52 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cost_value_l3380_338082


namespace NUMINAMATH_CALUDE_ice_cream_permutations_l3380_338056

/-- The number of permutations of n distinct elements -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- There are 4 distinct ice cream flavors -/
def num_flavors : ℕ := 4

/-- Theorem: The number of permutations of 4 distinct elements is 24 -/
theorem ice_cream_permutations : permutations num_flavors = 24 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_permutations_l3380_338056


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l3380_338050

theorem pasta_preference_ratio (total_students : ℕ) 
  (fettuccine_preference : ℕ) (tortellini_preference : ℕ) 
  (penne_preference : ℕ) (fusilli_preference : ℕ) : 
  total_students = 800 →
  total_students = fettuccine_preference + tortellini_preference + penne_preference + fusilli_preference →
  fettuccine_preference = 2 * tortellini_preference →
  (fettuccine_preference : ℚ) / tortellini_preference = 2 := by
sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_l3380_338050


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_value_l3380_338047

theorem quadratic_root_implies_k_value (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k - 1 = 0 ∧ x = -1) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_value_l3380_338047


namespace NUMINAMATH_CALUDE_gas_volume_at_20C_l3380_338011

/-- Represents the volume of a gas at a given temperature -/
structure GasVolume where
  temp : ℝ  -- temperature in Celsius
  vol : ℝ   -- volume in cubic centimeters

/-- Represents the relationship between temperature change and volume change -/
structure VolumeChange where
  temp_change : ℝ  -- temperature change in Celsius
  vol_change : ℝ   -- volume change in cubic centimeters

theorem gas_volume_at_20C 
  (initial : GasVolume)
  (change : VolumeChange)
  (h1 : initial.temp = 30)
  (h2 : initial.vol = 36)
  (h3 : change.temp_change = 2)
  (h4 : change.vol_change = 3) :
  ∃ (final : GasVolume), 
    final.temp = 20 ∧ 
    final.vol = 21 :=
sorry

end NUMINAMATH_CALUDE_gas_volume_at_20C_l3380_338011


namespace NUMINAMATH_CALUDE_miser_knight_theorem_l3380_338079

theorem miser_knight_theorem (n : ℕ) (h : n > 0) :
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 76 → ∃ m : ℕ, n = m * k) →
  ∃ m : ℕ, n = m * 77 :=
by sorry

end NUMINAMATH_CALUDE_miser_knight_theorem_l3380_338079


namespace NUMINAMATH_CALUDE_anne_solo_cleaning_time_l3380_338078

/-- Represents the time it takes Anne to clean the house alone -/
def anne_solo_time : ℝ := 12

/-- Represents Bruce's cleaning rate (houses per hour) -/
noncomputable def bruce_rate : ℝ := sorry

/-- Represents Anne's cleaning rate (houses per hour) -/
noncomputable def anne_rate : ℝ := sorry

theorem anne_solo_cleaning_time :
  (∀ (bruce_rate anne_rate : ℝ),
    bruce_rate > 0 ∧ anne_rate > 0 →
    (bruce_rate + anne_rate) * 4 = 1 →
    (bruce_rate + 2 * anne_rate) * 3 = 1 →
    1 / anne_rate = anne_solo_time) :=
by sorry

end NUMINAMATH_CALUDE_anne_solo_cleaning_time_l3380_338078


namespace NUMINAMATH_CALUDE_f_2008_eq_zero_l3380_338042

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2008_eq_zero
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_f2 : f 2 = 0)
  (h_periodic : ∀ x, f (x + 4) = f x + f 4) :
  f 2008 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2008_eq_zero_l3380_338042


namespace NUMINAMATH_CALUDE_multiple_of_larger_integer_l3380_338046

theorem multiple_of_larger_integer (s l : ℤ) (m : ℚ) : 
  s + l = 30 →
  s = 10 →
  m * l = 5 * s - 10 →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_larger_integer_l3380_338046


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3380_338096

theorem simplify_and_evaluate (m : ℝ) (h : m = 4 * Real.sqrt 3) :
  (1 - m / (m - 3)) / ((m^2 - 3*m) / (m^2 - 6*m + 9)) = -(Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3380_338096


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3380_338013

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 8| + 3 * y = 12 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3380_338013


namespace NUMINAMATH_CALUDE_lauryn_company_employees_l3380_338057

/-- The number of men working for Lauryn's company -/
def num_men : ℕ := 80

/-- The difference between the number of women and men -/
def women_men_diff : ℕ := 20

/-- The total number of people working for Lauryn's company -/
def total_employees : ℕ := num_men + (num_men + women_men_diff)

theorem lauryn_company_employees :
  total_employees = 180 :=
by sorry

end NUMINAMATH_CALUDE_lauryn_company_employees_l3380_338057


namespace NUMINAMATH_CALUDE_k_range_for_empty_intersection_l3380_338037

-- Define the sets A and B
def A (k : ℝ) : Set ℝ := {x | k * x^2 - (k + 3) * x - 1 ≥ 0}
def B : Set ℝ := {y | ∃ x, y = 2 * x + 1}

-- State the theorem
theorem k_range_for_empty_intersection :
  (∀ k : ℝ, (A k ∩ B = ∅)) ↔ (∀ k : ℝ, -9 < k ∧ k < -1) :=
sorry

end NUMINAMATH_CALUDE_k_range_for_empty_intersection_l3380_338037


namespace NUMINAMATH_CALUDE_fold_and_punch_theorem_l3380_338004

/-- Represents a rectangular piece of paper -/
structure Paper :=
  (width : ℕ)
  (height : ℕ)

/-- Represents the state of the paper after folding and punching -/
inductive FoldedPaper
  | Unfolded (p : Paper)
  | FoldedOnce (p : Paper)
  | FoldedTwice (p : Paper)
  | FoldedThrice (p : Paper)
  | Punched (p : Paper)

/-- Folds the paper from bottom to top -/
def foldBottomToTop (p : Paper) : FoldedPaper :=
  FoldedPaper.FoldedOnce p

/-- Folds the paper from right to left -/
def foldRightToLeft (p : FoldedPaper) : FoldedPaper :=
  match p with
  | FoldedPaper.FoldedOnce p => FoldedPaper.FoldedTwice p
  | _ => p

/-- Folds the paper from top to bottom -/
def foldTopToBottom (p : FoldedPaper) : FoldedPaper :=
  match p with
  | FoldedPaper.FoldedTwice p => FoldedPaper.FoldedThrice p
  | _ => p

/-- Punches a hole in the center of the folded paper -/
def punchHole (p : FoldedPaper) : FoldedPaper :=
  match p with
  | FoldedPaper.FoldedThrice p => FoldedPaper.Punched p
  | _ => p

/-- Counts the number of holes in the unfolded paper -/
def countHoles (p : FoldedPaper) : ℕ :=
  match p with
  | FoldedPaper.Punched _ => 8
  | _ => 0

/-- Theorem stating that folding a rectangular paper three times and punching a hole results in 8 holes when unfolded -/
theorem fold_and_punch_theorem (p : Paper) :
  countHoles (punchHole (foldTopToBottom (foldRightToLeft (foldBottomToTop p)))) = 8 := by
  sorry


end NUMINAMATH_CALUDE_fold_and_punch_theorem_l3380_338004


namespace NUMINAMATH_CALUDE_triple_root_values_l3380_338005

/-- A polynomial with integer coefficients of the form x^5 + b₄x^4 + b₃x^3 + b₂x^2 + b₁x + 24 -/
def IntPolynomial (b₄ b₃ b₂ b₁ : ℤ) (x : ℤ) : ℤ :=
  x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + 24

/-- r is a triple root of the polynomial if (x - r)^3 divides the polynomial -/
def IsTripleRoot (r : ℤ) (b₄ b₃ b₂ b₁ : ℤ) : Prop :=
  ∃ (q : ℤ → ℤ), ∀ x, IntPolynomial b₄ b₃ b₂ b₁ x = (x - r)^3 * q x

theorem triple_root_values (r : ℤ) :
  (∃ b₄ b₃ b₂ b₁ : ℤ, IsTripleRoot r b₄ b₃ b₂ b₁) ↔ r ∈ ({-2, -1, 1, 2} : Set ℤ) :=
sorry

end NUMINAMATH_CALUDE_triple_root_values_l3380_338005


namespace NUMINAMATH_CALUDE_bicentric_quadrilateral_theorem_l3380_338038

/-- A bicentric quadrilateral is a quadrilateral that has both an inscribed circle and a circumscribed circle. -/
structure BicentricQuadrilateral where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The radius of the circumscribed circle -/
  ρ : ℝ
  /-- The distance between the centers of the inscribed and circumscribed circles -/
  h : ℝ
  /-- Ensure r, ρ, and h are positive -/
  r_pos : r > 0
  ρ_pos : ρ > 0
  h_pos : h > 0
  /-- Ensure h is less than ρ (as the incenter must be inside the circumcircle) -/
  h_lt_ρ : h < ρ

/-- The main theorem about bicentric quadrilaterals -/
theorem bicentric_quadrilateral_theorem (q : BicentricQuadrilateral) :
  1 / (q.ρ + q.h)^2 + 1 / (q.ρ - q.h)^2 = 1 / q.r^2 := by
  sorry

end NUMINAMATH_CALUDE_bicentric_quadrilateral_theorem_l3380_338038


namespace NUMINAMATH_CALUDE_shawn_red_pebbles_l3380_338040

/-- The number of red pebbles in Shawn's collection -/
def red_pebbles (total blue yellow : ℕ) : ℕ :=
  total - (blue + 3 * yellow)

/-- Theorem stating the number of red pebbles Shawn painted -/
theorem shawn_red_pebbles :
  ∃ (yellow : ℕ),
    red_pebbles 40 13 yellow = 9 ∧
    13 - yellow = 7 :=
by sorry

end NUMINAMATH_CALUDE_shawn_red_pebbles_l3380_338040


namespace NUMINAMATH_CALUDE_parallel_vectors_problem_l3380_338076

/-- Given vectors a, b, and c in ℝ², prove that if a is parallel to c and c = a + 3b, then x = 4 -/
theorem parallel_vectors_problem (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -4]
  let b : Fin 2 → ℝ := ![-1, x]
  let c : Fin 2 → ℝ := a + 3 • b
  (∃ (k : ℝ), c = k • a) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_problem_l3380_338076


namespace NUMINAMATH_CALUDE_average_difference_l3380_338099

def num_students : ℕ := 120
def num_teachers : ℕ := 6
def class_enrollments : List ℕ := [60, 30, 20, 5, 3, 2]

def t : ℚ := (class_enrollments.sum : ℚ) / num_teachers

def s : ℚ := (class_enrollments.map (λ n => n * n)).sum / num_students

theorem average_difference : t - s = -21151/1000 := by sorry

end NUMINAMATH_CALUDE_average_difference_l3380_338099


namespace NUMINAMATH_CALUDE_sin_theta_value_l3380_338016

theorem sin_theta_value (θ : Real) 
  (h1 : 10 * Real.tan θ = 4 * Real.cos θ) 
  (h2 : 0 < θ) 
  (h3 : θ < Real.pi) : 
  Real.sin θ = (-5 + Real.sqrt 41) / 4 := by
sorry

end NUMINAMATH_CALUDE_sin_theta_value_l3380_338016


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3380_338008

/-- Given a geometric sequence {a_n}, prove that a_4 * a_7 = -6 
    when a_1 and a_10 are roots of x^2 - x - 6 = 0 -/
theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∀ n : ℕ, ∃ r : ℝ, a (n + 1) = a n * r) →  -- geometric sequence condition
  (a 1)^2 - (a 1) - 6 = 0 →  -- a_1 is a root of x^2 - x - 6 = 0
  (a 10)^2 - (a 10) - 6 = 0 →  -- a_10 is a root of x^2 - x - 6 = 0
  a 4 * a 7 = -6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3380_338008


namespace NUMINAMATH_CALUDE_pythagorean_triple_with_primes_l3380_338023

theorem pythagorean_triple_with_primes (x y z : ℤ) :
  x^2 + y^2 = z^2 →
  (Prime y ∧ y > 5 ∧ Prime z ∧ z > 5) ∨
  (Prime x ∧ x > 5 ∧ Prime z ∧ z > 5) ∨
  (Prime x ∧ x > 5 ∧ Prime y ∧ y > 5) →
  60 ∣ x ∨ 60 ∣ y ∨ 60 ∣ z :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_with_primes_l3380_338023


namespace NUMINAMATH_CALUDE_relationship_depends_on_b_relationship_only_b_l3380_338009

theorem relationship_depends_on_b (a b : ℝ) : 
  (a + b) - (a - b) = 2 * b :=
sorry

theorem relationship_only_b (a b : ℝ) : 
  (a + b > a - b ↔ b > 0) ∧
  (a + b < a - b ↔ b < 0) ∧
  (a + b = a - b ↔ b = 0) :=
sorry

end NUMINAMATH_CALUDE_relationship_depends_on_b_relationship_only_b_l3380_338009


namespace NUMINAMATH_CALUDE_reverse_digits_square_diff_l3380_338029

/-- Given two-digit integers x and y where y is the reverse of x, and x^2 - y^2 = m^2 for some positive integer m, prove that x + y + m = 154 -/
theorem reverse_digits_square_diff (x y m : ℕ) : 
  (10 ≤ x ∧ x < 100) →  -- x is a two-digit integer
  (10 ≤ y ∧ y < 100) →  -- y is a two-digit integer
  (∃ (a b : ℕ), x = 10 * a + b ∧ y = 10 * b + a ∧ 0 < a ∧ a < 10 ∧ 0 < b ∧ b < 10) →  -- y is obtained by reversing the digits of x
  (x^2 - y^2 = m^2) →  -- x^2 - y^2 = m^2
  (0 < m) →  -- m is positive
  (x + y + m = 154) := by
sorry

end NUMINAMATH_CALUDE_reverse_digits_square_diff_l3380_338029


namespace NUMINAMATH_CALUDE_brownie_calories_l3380_338073

-- Define the parameters of the problem
def cake_slices : ℕ := 8
def calories_per_cake_slice : ℕ := 347
def brownies : ℕ := 6
def calorie_difference : ℕ := 526

-- Define the function to calculate calories per brownie
def calories_per_brownie : ℕ :=
  ((cake_slices * calories_per_cake_slice - calorie_difference) / brownies : ℕ)

-- Theorem statement
theorem brownie_calories :
  calories_per_brownie = 375 := by
  sorry

end NUMINAMATH_CALUDE_brownie_calories_l3380_338073


namespace NUMINAMATH_CALUDE_wax_calculation_l3380_338052

/-- The amount of wax required for the feathers -/
def required_wax : ℕ := 166

/-- The additional amount of wax needed -/
def additional_wax : ℕ := 146

/-- The current amount of wax -/
def current_wax : ℕ := required_wax - additional_wax

theorem wax_calculation : current_wax = 20 := by
  sorry

end NUMINAMATH_CALUDE_wax_calculation_l3380_338052


namespace NUMINAMATH_CALUDE_conjunction_false_l3380_338061

theorem conjunction_false (p q : Prop) (hp : p) (hq : ¬q) : ¬(p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_conjunction_false_l3380_338061


namespace NUMINAMATH_CALUDE_solution_quadratic_equation_l3380_338071

theorem solution_quadratic_equation : 
  ∀ x : ℝ, (x - 2) * (x + 3) = 0 ↔ x = 2 ∨ x = -3 := by sorry

end NUMINAMATH_CALUDE_solution_quadratic_equation_l3380_338071


namespace NUMINAMATH_CALUDE_worksheet_problems_l3380_338059

theorem worksheet_problems (total_worksheets graded_worksheets remaining_problems : ℕ) 
  (h1 : total_worksheets = 9)
  (h2 : graded_worksheets = 5)
  (h3 : remaining_problems = 16) :
  (total_worksheets - graded_worksheets) * (remaining_problems / (total_worksheets - graded_worksheets)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_worksheet_problems_l3380_338059


namespace NUMINAMATH_CALUDE_solve_equation_l3380_338039

/-- Proves that the solution to the equation 4.7 × 13.26 + 4.7 × 9.43 + 4.7 × x = 470 is x = 77.31 -/
theorem solve_equation : 
  ∃ x : ℝ, 4.7 * 13.26 + 4.7 * 9.43 + 4.7 * x = 470 ∧ x = 77.31 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3380_338039


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3380_338012

theorem absolute_value_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3380_338012


namespace NUMINAMATH_CALUDE_minimum_time_is_110_l3380_338093

/-- Represents the time taken by each teacher to examine one student -/
structure TeacherTime where
  time : ℕ

/-- Represents the problem of finding the minimum examination time -/
structure ExaminationProblem where
  teacher1 : TeacherTime
  teacher2 : TeacherTime
  totalStudents : ℕ

/-- Calculates the minimum examination time for the given problem -/
def minimumExaminationTime (problem : ExaminationProblem) : ℕ :=
  sorry

/-- Theorem stating that the minimum examination time for the given problem is 110 minutes -/
theorem minimum_time_is_110 (problem : ExaminationProblem) 
  (h1 : problem.teacher1.time = 12)
  (h2 : problem.teacher2.time = 7)
  (h3 : problem.totalStudents = 25) :
  minimumExaminationTime problem = 110 := by
  sorry

end NUMINAMATH_CALUDE_minimum_time_is_110_l3380_338093


namespace NUMINAMATH_CALUDE_skylar_starting_donation_age_l3380_338067

/-- The age at which Skylar started donating -/
def starting_age (annual_donation : ℕ) (total_donation : ℕ) (current_age : ℕ) : ℕ :=
  current_age - (total_donation / annual_donation)

/-- Theorem stating the age at which Skylar started donating -/
theorem skylar_starting_donation_age :
  starting_age 5000 105000 33 = 12 := by
  sorry

end NUMINAMATH_CALUDE_skylar_starting_donation_age_l3380_338067


namespace NUMINAMATH_CALUDE_investment_interest_theorem_l3380_338003

/-- Calculates the total interest paid in an 18-month investment contract with specified conditions -/
def totalInterest (initialInvestment : ℝ) : ℝ :=
  let interestRate1 := 0.02
  let interestRate2 := 0.03
  let interestRate3 := 0.04
  
  let interest1 := initialInvestment * interestRate1
  let newInvestment1 := initialInvestment + interest1
  
  let interest2 := newInvestment1 * interestRate2
  let newInvestment2 := newInvestment1 + interest2
  
  let interest3 := newInvestment2 * interestRate3
  
  interest1 + interest2 + interest3

/-- Theorem stating that the total interest paid in the given investment scenario is $926.24 -/
theorem investment_interest_theorem :
  totalInterest 10000 = 926.24 := by sorry

end NUMINAMATH_CALUDE_investment_interest_theorem_l3380_338003


namespace NUMINAMATH_CALUDE_equation_equality_l3380_338002

theorem equation_equality 
  (p q r x y z a b c : ℝ) 
  (h1 : p / x = q / y ∧ q / y = r / z) 
  (h2 : x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1) :
  p^2 / a^2 + q^2 / b^2 + r^2 / c^2 = (p^2 + q^2 + r^2) / (x^2 + y^2 + z^2) := by
sorry

end NUMINAMATH_CALUDE_equation_equality_l3380_338002


namespace NUMINAMATH_CALUDE_optimal_zongzi_purchase_l3380_338010

/-- Represents the unit price and quantity of zongzi --/
structure Zongzi where
  unit_price : ℝ
  quantity : ℕ

/-- Represents the shopping mall's zongzi purchase plan --/
structure ZongziPurchasePlan where
  zongzi_a : Zongzi
  zongzi_b : Zongzi

/-- Defines the conditions of the zongzi purchase problem --/
def zongzi_problem (plan : ZongziPurchasePlan) : Prop :=
  let a := plan.zongzi_a
  let b := plan.zongzi_b
  (3000 / a.unit_price - 3360 / b.unit_price = 40) ∧
  (b.unit_price = 1.2 * a.unit_price) ∧
  (a.quantity + b.quantity = 2200) ∧
  (a.unit_price * a.quantity ≤ b.unit_price * b.quantity)

/-- Theorem stating the optimal solution to the zongzi purchase problem --/
theorem optimal_zongzi_purchase :
  ∃ (plan : ZongziPurchasePlan),
    zongzi_problem plan ∧
    plan.zongzi_a.unit_price = 5 ∧
    plan.zongzi_b.unit_price = 6 ∧
    plan.zongzi_a.quantity = 1200 ∧
    plan.zongzi_b.quantity = 1000 ∧
    plan.zongzi_a.unit_price * plan.zongzi_a.quantity +
    plan.zongzi_b.unit_price * plan.zongzi_b.quantity = 12000 :=
  sorry

end NUMINAMATH_CALUDE_optimal_zongzi_purchase_l3380_338010


namespace NUMINAMATH_CALUDE_sequence_sum_implies_general_term_l3380_338019

/-- Given a sequence (aₙ) with sum Sₙ = (2/3)aₙ + 1/3, prove aₙ = (-2)^(n-1) -/
theorem sequence_sum_implies_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n : ℕ, n ≥ 1 → S n = (2/3) * a n + 1/3) :
  ∀ n : ℕ, n ≥ 1 → a n = (-2)^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_implies_general_term_l3380_338019


namespace NUMINAMATH_CALUDE_grandpa_to_uncle_ratio_l3380_338081

/-- Represents the number of toy cars in various scenarios --/
structure ToyCars where
  initial : ℕ
  final : ℕ
  fromDad : ℕ
  fromMum : ℕ
  fromAuntie : ℕ
  fromUncle : ℕ
  fromGrandpa : ℕ

/-- Theorem stating the ratio of Grandpa's gift to Uncle's gift --/
theorem grandpa_to_uncle_ratio (cars : ToyCars)
  (h1 : cars.initial = 150)
  (h2 : cars.final = 196)
  (h3 : cars.fromDad = 10)
  (h4 : cars.fromMum = cars.fromDad + 5)
  (h5 : cars.fromAuntie = 6)
  (h6 : cars.fromUncle = cars.fromAuntie - 1)
  (h7 : cars.final = cars.initial + cars.fromDad + cars.fromMum + cars.fromAuntie + cars.fromUncle + cars.fromGrandpa) :
  cars.fromGrandpa = 2 * cars.fromUncle := by
  sorry

#check grandpa_to_uncle_ratio

end NUMINAMATH_CALUDE_grandpa_to_uncle_ratio_l3380_338081


namespace NUMINAMATH_CALUDE_work_completion_time_l3380_338090

/-- Represents the work rate of one person per hour -/
structure WorkRate where
  man : ℝ
  woman : ℝ

/-- Represents a work scenario -/
structure WorkScenario where
  men : ℕ
  women : ℕ
  hours_per_day : ℝ
  days : ℝ

def total_work (rate : WorkRate) (scenario : WorkScenario) : ℝ :=
  (scenario.men * rate.man + scenario.women * rate.woman) * scenario.hours_per_day * scenario.days

theorem work_completion_time 
  (rate : WorkRate)
  (scenario1 : WorkScenario)
  (scenario2 : WorkScenario)
  (scenario3 : WorkScenario) :
  scenario1.men = 1 →
  scenario1.women = 3 →
  scenario1.hours_per_day = 7 →
  scenario1.days = 5 →
  scenario2.men = 4 →
  scenario2.women = 4 →
  scenario2.hours_per_day = 3 →
  scenario3.men = 7 →
  scenario3.women = 0 →
  scenario3.hours_per_day = 4 →
  scenario3.days = 5.000000000000001 →
  total_work rate scenario1 = total_work rate scenario3 →
  scenario2.days = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3380_338090


namespace NUMINAMATH_CALUDE_smallest_absolute_value_l3380_338098

theorem smallest_absolute_value (x : ℝ) : |x| ≥ 0 ∧ (|x| = 0 ↔ x = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_l3380_338098


namespace NUMINAMATH_CALUDE_g_zero_eq_one_l3380_338085

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x + y) = g x + g y - 1

/-- Theorem stating that g(0) = 1 for any function satisfying the functional equation -/
theorem g_zero_eq_one (g : ℝ → ℝ) (h : FunctionalEquation g) : g 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_eq_one_l3380_338085


namespace NUMINAMATH_CALUDE_negation_equivalence_l3380_338041

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3380_338041


namespace NUMINAMATH_CALUDE_fruit_salad_red_grapes_l3380_338000

theorem fruit_salad_red_grapes (green_grapes : ℕ) : 
  let red_grapes := 3 * green_grapes + 7
  let raspberries := green_grapes - 5
  green_grapes + red_grapes + raspberries = 102 →
  red_grapes = 67 := by
  sorry

end NUMINAMATH_CALUDE_fruit_salad_red_grapes_l3380_338000


namespace NUMINAMATH_CALUDE_math_team_selection_l3380_338074

theorem math_team_selection (boys girls : ℕ) (h1 : boys = 10) (h2 : girls = 12) :
  (Nat.choose boys 5) * (Nat.choose girls 3) = 55440 := by
  sorry

end NUMINAMATH_CALUDE_math_team_selection_l3380_338074


namespace NUMINAMATH_CALUDE_speed_conversion_l3380_338055

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- The given speed in m/s -/
def given_speed_mps : ℝ := 15.556799999999999

/-- The speed in km/h we want to prove -/
def speed_kmph : ℝ := 56.00448

theorem speed_conversion : given_speed_mps * mps_to_kmph = speed_kmph := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l3380_338055


namespace NUMINAMATH_CALUDE_total_leaves_l3380_338014

/-- The number of leaves Sabrina needs for her poultice --/
structure HerbLeaves where
  basil : ℕ
  sage : ℕ
  verbena : ℕ
  chamomile : ℕ
  lavender : ℕ

/-- The conditions for Sabrina's herb collection --/
def validHerbCollection (h : HerbLeaves) : Prop :=
  h.basil = 3 * h.sage ∧
  h.verbena = h.sage + 8 ∧
  h.chamomile = 2 * h.sage + 7 ∧
  h.lavender = (h.basil + h.chamomile + 1) / 2 ∧
  h.basil = 48

/-- The theorem stating the total number of leaves needed --/
theorem total_leaves (h : HerbLeaves) (hvalid : validHerbCollection h) :
  h.basil + h.sage + h.verbena + h.chamomile + h.lavender = 171 := by
  sorry

#check total_leaves

end NUMINAMATH_CALUDE_total_leaves_l3380_338014


namespace NUMINAMATH_CALUDE_roots_relation_l3380_338091

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4
def g (x b c d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

-- State the theorem
theorem roots_relation (b c d : ℝ) : 
  (∀ x y : ℝ, x ≠ y → f x = 0 → f y = 0 → x ≠ y) →  -- f has distinct roots
  (∀ r : ℝ, f r = 0 → g (r^3) b c d = 0) →          -- roots of g are cubes of roots of f
  b = -8 ∧ c = -36 ∧ d = -64 := by
  sorry

end NUMINAMATH_CALUDE_roots_relation_l3380_338091


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l3380_338030

-- Define the curve equation
def curve (x y k : ℝ) : Prop := x^2 / (k + 4) + y^2 / (k - 1) = 1

-- Define what it means for the curve to be a hyperbola
def is_hyperbola (k : ℝ) : Prop := ∃ x y, curve x y k ∧ (k + 4) * (k - 1) < 0

-- Theorem statement
theorem hyperbola_k_range :
  ∀ k : ℝ, is_hyperbola k → k ∈ Set.Ioo (-4 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l3380_338030


namespace NUMINAMATH_CALUDE_locus_is_ray_l3380_338051

/-- The locus of points P satisfying |PM| - |PN| = 4, where M(-2,0) and N(2,0) are fixed points -/
def locus_of_P (P : ℝ × ℝ) : Prop :=
  let M : ℝ × ℝ := (-2, 0)
  let N : ℝ × ℝ := (2, 0)
  Real.sqrt ((P.1 + 2)^2 + P.2^2) - Real.sqrt ((P.1 - 2)^2 + P.2^2) = 4

/-- The ray starting from the midpoint of MN and extending to the right -/
def ray_from_midpoint (P : ℝ × ℝ) : Prop :=
  P.1 ≥ 0 ∧ P.2 = 0

theorem locus_is_ray :
  ∀ P, locus_of_P P ↔ ray_from_midpoint P :=
sorry

end NUMINAMATH_CALUDE_locus_is_ray_l3380_338051


namespace NUMINAMATH_CALUDE_profit_function_and_maximum_profit_constraint_and_price_l3380_338077

/-- Weekly profit function -/
def W (x : ℝ) : ℝ := -10 * x^2 + 200 * x + 15000

/-- Initial cost per box in yuan -/
def initial_cost : ℝ := 70

/-- Initial selling price per box in yuan -/
def initial_price : ℝ := 120

/-- Initial weekly sales volume in boxes -/
def initial_sales : ℝ := 300

/-- Sales increase per yuan of price reduction -/
def sales_increase_rate : ℝ := 10

theorem profit_function_and_maximum (x : ℝ) :
  W x = -10 * x^2 + 200 * x + 15000 ∧
  (∀ y : ℝ, W y ≤ W 10) ∧
  W 10 = 16000 := by sorry

theorem profit_constraint_and_price (x : ℝ) :
  W x = 15960 →
  x ≤ 12 →
  initial_price - 12 = 108 := by sorry

end NUMINAMATH_CALUDE_profit_function_and_maximum_profit_constraint_and_price_l3380_338077


namespace NUMINAMATH_CALUDE_quadratic_function_m_condition_l3380_338045

/-- A function f: ℝ → ℝ is quadratic if it can be written as f(x) = ax² + bx + c where a ≠ 0 -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function y = (m+1)x² + 2x + 1 -/
def f (m : ℝ) : ℝ → ℝ := λ x ↦ (m + 1) * x^2 + 2 * x + 1

theorem quadratic_function_m_condition :
  ∀ m : ℝ, is_quadratic (f m) ↔ m ≠ -1 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_m_condition_l3380_338045


namespace NUMINAMATH_CALUDE_rahul_work_time_l3380_338084

-- Define the work completion time for Rajesh
def rajesh_time : ℝ := 2

-- Define the total payment
def total_payment : ℝ := 170

-- Define Rahul's share
def rahul_share : ℝ := 68

-- Define Rahul's work completion time (to be proved)
def rahul_time : ℝ := 3

-- Theorem statement
theorem rahul_work_time :
  -- Given conditions
  (rajesh_time = 2) →
  (total_payment = 170) →
  (rahul_share = 68) →
  -- Proof goal
  (rahul_time = 3) := by
    sorry -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_rahul_work_time_l3380_338084


namespace NUMINAMATH_CALUDE_symmetry_of_lines_l3380_338049

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := y = -x

-- Define the symmetric line l₂
def l₂ (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Theorem statement
theorem symmetry_of_lines :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    l₁ x₁ y₁ →
    line_of_symmetry ((x₁ + x₂) / 2) ((y₁ + y₂) / 2) →
    (x₂ = 2 * ((x₁ + x₂) / 2) - x₁ ∧ y₂ = 2 * ((y₁ + y₂) / 2) - y₁) →
    l₂ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_symmetry_of_lines_l3380_338049


namespace NUMINAMATH_CALUDE_platform_length_l3380_338054

/-- The length of a platform passed by an accelerating train -/
theorem platform_length (l a t : ℝ) (h1 : l > 0) (h2 : a > 0) (h3 : t > 0) : ∃ P : ℝ,
  (l = (1/2) * a * t^2) →
  (l + P = (1/2) * a * (6*t)^2) →
  P = 17 * l := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l3380_338054


namespace NUMINAMATH_CALUDE_counterfeit_weight_equals_net_profit_l3380_338036

/-- Represents a dishonest dealer's selling strategy -/
structure DishonestDealer where
  /-- The percentage of impurities added to the product -/
  impurities : ℝ
  /-- The net profit percentage achieved by the dealer -/
  net_profit : ℝ

/-- Calculates the percentage by which the counterfeit weight is less than the real weight -/
def counterfeit_weight_percentage (dealer : DishonestDealer) : ℝ :=
  dealer.net_profit

/-- Theorem stating that under specific conditions, the counterfeit weight percentage
    equals the net profit percentage -/
theorem counterfeit_weight_equals_net_profit 
  (dealer : DishonestDealer) 
  (h1 : dealer.impurities = 35)
  (h2 : dealer.net_profit = 68.75) :
  counterfeit_weight_percentage dealer = 68.75 := by
  sorry

end NUMINAMATH_CALUDE_counterfeit_weight_equals_net_profit_l3380_338036


namespace NUMINAMATH_CALUDE_system_solution_l3380_338053

theorem system_solution : ∃ (x y : ℝ), (2 * x - y = 5) ∧ (7 * x - 3 * y = 20) ∧ (x = 5) ∧ (y = 5) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3380_338053


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l3380_338043

theorem smallest_positive_solution :
  ∀ x : ℝ, x > 0 ∧ Real.sqrt x = 9 * x^2 → x ≥ 1/81 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l3380_338043


namespace NUMINAMATH_CALUDE_b_minus_a_value_l3380_338034

theorem b_minus_a_value (a b : ℝ) (h1 : |a| = 1) (h2 : |b| = 3) (h3 : a < b) :
  b - a = 2 ∨ b - a = 4 := by
  sorry

end NUMINAMATH_CALUDE_b_minus_a_value_l3380_338034


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3380_338062

-- Problem 1
theorem problem_1 (x : ℝ) : 
  x / (2 * x - 3) + 5 / (3 - 2 * x) = 4 ↔ x = 1 :=
sorry

-- Problem 2
theorem problem_2 : 
  ¬∃ (x : ℝ), (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 :=
sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3380_338062


namespace NUMINAMATH_CALUDE_tax_savings_calculation_l3380_338031

/-- Calculates the differential savings when tax rate is lowered -/
def differential_savings (income : ℝ) (old_rate new_rate : ℝ) : ℝ :=
  income * (old_rate - new_rate)

/-- Theorem: The differential savings for a taxpayer with an annual income
    of $42,400, when the tax rate is reduced from 42% to 32%, is $4,240 -/
theorem tax_savings_calculation :
  differential_savings 42400 0.42 0.32 = 4240 := by
  sorry

end NUMINAMATH_CALUDE_tax_savings_calculation_l3380_338031


namespace NUMINAMATH_CALUDE_cururu_jump_theorem_l3380_338094

/-- Represents the number of jumps of each type -/
structure JumpCount where
  typeI : ℕ
  typeII : ℕ

/-- Checks if a given jump count reaches the target position -/
def reachesTarget (jumps : JumpCount) (targetEast targetNorth : ℤ) : Prop :=
  10 * jumps.typeI - 20 * jumps.typeII = targetEast ∧
  30 * jumps.typeI - 40 * jumps.typeII = targetNorth

theorem cururu_jump_theorem :
  (∃ jumps : JumpCount, reachesTarget jumps 190 950) ∧
  (¬ ∃ jumps : JumpCount, reachesTarget jumps 180 950) := by
  sorry

#check cururu_jump_theorem

end NUMINAMATH_CALUDE_cururu_jump_theorem_l3380_338094


namespace NUMINAMATH_CALUDE_line_equation_through_points_l3380_338087

/-- Given two points P(3,2) and Q(4,7), prove that the equation 5x - y - 13 = 0
    represents the line passing through these points. -/
theorem line_equation_through_points (x y : ℝ) :
  let P : ℝ × ℝ := (3, 2)
  let Q : ℝ × ℝ := (4, 7)
  (5 * x - y - 13 = 0) ↔ 
    (∃ t : ℝ, (x, y) = ((1 - t) • P.1 + t • Q.1, (1 - t) • P.2 + t • Q.2)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l3380_338087


namespace NUMINAMATH_CALUDE_hannah_dog_food_l3380_338033

/-- The amount of dog food Hannah needs to prepare daily for her five dogs -/
def total_dog_food (dog1_meal : ℝ) (dog1_freq : ℕ) (dog2_ratio : ℝ) (dog2_freq : ℕ)
  (dog3_extra : ℝ) (dog3_freq : ℕ) (dog4_ratio : ℝ) (dog4_freq : ℕ)
  (dog5_ratio : ℝ) (dog5_freq : ℕ) : ℝ :=
  (dog1_meal * dog1_freq) +
  (dog1_meal * dog2_ratio * dog2_freq) +
  ((dog1_meal * dog2_ratio + dog3_extra) * dog3_freq) +
  (dog4_ratio * (dog1_meal * dog2_ratio + dog3_extra) * dog4_freq) +
  (dog5_ratio * dog1_meal * dog5_freq)

/-- Theorem stating that Hannah needs to prepare 40.5 cups of dog food daily -/
theorem hannah_dog_food : total_dog_food 1.5 2 2 1 2.5 3 1.2 2 0.8 4 = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_hannah_dog_food_l3380_338033


namespace NUMINAMATH_CALUDE_triangle_altitude_l3380_338072

theorem triangle_altitude (area : ℝ) (base : ℝ) (altitude : ℝ) :
  area = 960 →
  base = 48 →
  area = (1 / 2) * base * altitude →
  altitude = 40 := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_l3380_338072


namespace NUMINAMATH_CALUDE_matrix_det_plus_five_l3380_338017

theorem matrix_det_plus_five (M : Matrix (Fin 2) (Fin 2) ℤ) :
  M = ![![7, -2], ![-3, 6]] →
  M.det + 5 = 41 := by
sorry

end NUMINAMATH_CALUDE_matrix_det_plus_five_l3380_338017


namespace NUMINAMATH_CALUDE_smallest_number_l3380_338068

theorem smallest_number (a b c d : ℤ) (ha : a = -1) (hb : b = -2) (hc : c = 1) (hd : d = 2) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3380_338068


namespace NUMINAMATH_CALUDE_solution_count_l3380_338097

def is_solution (a b : ℕ+) : Prop :=
  (1 : ℚ) / a.val - (1 : ℚ) / b.val = (1 : ℚ) / 2018

theorem solution_count :
  ∃! (s : Finset (ℕ+ × ℕ+)), 
    (∀ (p : ℕ+ × ℕ+), p ∈ s ↔ is_solution p.1 p.2) ∧ 
    s.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_solution_count_l3380_338097


namespace NUMINAMATH_CALUDE_box_cost_is_111_kopecks_l3380_338086

/-- The cost of a box of matches in kopecks -/
def box_cost : ℕ := sorry

/-- Nine boxes cost more than 9 rubles but less than 10 rubles -/
axiom nine_boxes_cost : 900 < 9 * box_cost ∧ 9 * box_cost < 1000

/-- Ten boxes cost more than 11 rubles but less than 12 rubles -/
axiom ten_boxes_cost : 1100 < 10 * box_cost ∧ 10 * box_cost < 1200

/-- The cost of one box of matches is 1 ruble 11 kopecks -/
theorem box_cost_is_111_kopecks : box_cost = 111 := by sorry

end NUMINAMATH_CALUDE_box_cost_is_111_kopecks_l3380_338086


namespace NUMINAMATH_CALUDE_machine_output_for_68_l3380_338026

def number_machine (x : ℕ) : ℕ := x + 15 - 6

theorem machine_output_for_68 : number_machine 68 = 77 := by
  sorry

end NUMINAMATH_CALUDE_machine_output_for_68_l3380_338026


namespace NUMINAMATH_CALUDE_max_hiking_time_l3380_338020

/-- Calculates the maximum hiking time for Violet and her dog given their water consumption rates and the total water carried. -/
theorem max_hiking_time (violet_rate : ℝ) (dog_rate : ℝ) (total_water : ℝ) :
  violet_rate = 800 →
  dog_rate = 400 →
  total_water = 4800 →
  (total_water / (violet_rate + dog_rate) : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_hiking_time_l3380_338020


namespace NUMINAMATH_CALUDE_chocolate_distribution_l3380_338083

theorem chocolate_distribution (total_chocolate : ℚ) (num_packages : ℕ) (neighbor_packages : ℕ) :
  total_chocolate = 72 / 7 →
  num_packages = 6 →
  neighbor_packages = 2 →
  (total_chocolate / num_packages) * neighbor_packages = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l3380_338083


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3380_338006

theorem polynomial_simplification (x : ℝ) :
  (10 * x^3 - 30 * x^2 + 40 * x - 5) - (3 * x^3 - 7 * x^2 - 5 * x + 10) =
  7 * x^3 - 23 * x^2 + 45 * x - 15 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3380_338006


namespace NUMINAMATH_CALUDE_parcel_weight_l3380_338065

theorem parcel_weight (x y z : ℝ) 
  (h1 : x + y = 110) 
  (h2 : y + z = 140) 
  (h3 : z + x = 130) : 
  x + y + z = 190 := by
sorry

end NUMINAMATH_CALUDE_parcel_weight_l3380_338065


namespace NUMINAMATH_CALUDE_traffic_accident_emergency_number_correct_l3380_338044

def emergency_numbers : List ℕ := [122, 110, 120, 114]

def traffic_accident_emergency_number : ℕ := 122

theorem traffic_accident_emergency_number_correct :
  traffic_accident_emergency_number ∈ emergency_numbers ∧
  traffic_accident_emergency_number = 122 := by
  sorry

end NUMINAMATH_CALUDE_traffic_accident_emergency_number_correct_l3380_338044


namespace NUMINAMATH_CALUDE_saturday_zoo_visitors_l3380_338027

theorem saturday_zoo_visitors (friday_visitors : ℕ) (saturday_multiplier : ℕ) : 
  friday_visitors = 1250 →
  saturday_multiplier = 3 →
  friday_visitors * saturday_multiplier = 3750 :=
by
  sorry

end NUMINAMATH_CALUDE_saturday_zoo_visitors_l3380_338027


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l3380_338070

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, 2 * a * x^2 + 10 * x + c = 0) →
  a + c = 12 →
  a < c →
  a = 1.15 ∧ c = 10.85 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l3380_338070


namespace NUMINAMATH_CALUDE_prob_at_least_two_same_dice_l3380_338035

-- Define the number of dice and sides
def num_dice : ℕ := 5
def num_sides : ℕ := 6

-- Define the total number of outcomes
def total_outcomes : ℕ := num_sides ^ num_dice

-- Define the number of outcomes with all different numbers
def all_different_outcomes : ℕ := num_sides * (num_sides - 1) * (num_sides - 2) * (num_sides - 3) * (num_sides - 4)

-- Define the probability of at least two dice showing the same number
def prob_at_least_two_same : ℚ := 1 - (all_different_outcomes : ℚ) / total_outcomes

-- Theorem statement
theorem prob_at_least_two_same_dice :
  prob_at_least_two_same = 7056 / 7776 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_two_same_dice_l3380_338035


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_35_degree_angle_l3380_338022

/-- The degree measure of the supplement of the complement of a 35-degree angle is 125°. -/
theorem supplement_of_complement_of_35_degree_angle : 
  let angle : ℝ := 35
  let complement := 90 - angle
  let supplement := 180 - complement
  supplement = 125 := by sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_35_degree_angle_l3380_338022
