import Mathlib

namespace NUMINAMATH_CALUDE_extra_coverage_area_l3995_399528

/-- Represents the area covered by one bag of grass seed in square feet. -/
def bag_coverage : ℕ := 250

/-- Represents the length of the lawn from house to curb in feet. -/
def lawn_length : ℕ := 22

/-- Represents the width of the lawn from side to side in feet. -/
def lawn_width : ℕ := 36

/-- Represents the number of bags of grass seed bought. -/
def bags_bought : ℕ := 4

/-- Calculates the extra area that can be covered by leftover grass seed after reseeding the lawn. -/
theorem extra_coverage_area : 
  bags_bought * bag_coverage - lawn_length * lawn_width = 208 := by
  sorry

end NUMINAMATH_CALUDE_extra_coverage_area_l3995_399528


namespace NUMINAMATH_CALUDE_percent_increase_l3995_399504

theorem percent_increase (P : ℝ) (Q : ℝ) (h : Q = P + (1/3) * P) :
  (Q - P) / P * 100 = 100/3 := by
  sorry

end NUMINAMATH_CALUDE_percent_increase_l3995_399504


namespace NUMINAMATH_CALUDE_correct_classification_l3995_399551

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical

-- Define the structure of a reasoning process
structure ReasoningProcess where
  description : String
  correct_type : ReasoningType

-- Define the three reasoning processes
def process1 : ReasoningProcess :=
  { description := "The probability of a coin landing heads up is determined to be 0.5 through numerous trials",
    correct_type := ReasoningType.Inductive }

def process2 : ReasoningProcess :=
  { description := "The function f(x) = x^2 - |x| is an even function",
    correct_type := ReasoningType.Deductive }

def process3 : ReasoningProcess :=
  { description := "Scientists invented the electronic eagle eye by studying the eyes of eagles",
    correct_type := ReasoningType.Analogical }

-- Theorem to prove
theorem correct_classification :
  (process1.correct_type = ReasoningType.Inductive) ∧
  (process2.correct_type = ReasoningType.Deductive) ∧
  (process3.correct_type = ReasoningType.Analogical) :=
by sorry

end NUMINAMATH_CALUDE_correct_classification_l3995_399551


namespace NUMINAMATH_CALUDE_det_A_positive_iff_x_gt_one_l3995_399554

/-- Definition of a 2x2 matrix A with elements dependent on x -/
def A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![2, 3 - x; 1, x]

/-- Definition of determinant for 2x2 matrix -/
def det2x2 (M : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
  M 0 0 * M 1 1 - M 0 1 * M 1 0

/-- Theorem stating that det(A) > 0 iff x > 1 -/
theorem det_A_positive_iff_x_gt_one :
  ∀ x : ℝ, det2x2 (A x) > 0 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_det_A_positive_iff_x_gt_one_l3995_399554


namespace NUMINAMATH_CALUDE_equation_two_solutions_l3995_399544

def equation (a x : ℝ) : Prop :=
  (Real.cos (2 * x) + 14 * Real.cos x - 14 * a)^7 - (6 * a * Real.cos x - 4 * a^2 - 1)^7 = 
  (6 * a - 14) * Real.cos x + 2 * Real.sin x^2 - 4 * a^2 + 14 * a - 2

theorem equation_two_solutions :
  ∃ (S₁ S₂ : Set ℝ),
    (S₁ = {a : ℝ | 3.25 ≤ a ∧ a < 4}) ∧
    (S₂ = {a : ℝ | -0.5 ≤ a ∧ a < 1}) ∧
    (∀ a ∈ S₁ ∪ S₂, ∃ (x₁ x₂ : ℝ),
      x₁ ≠ x₂ ∧
      -2 * Real.pi / 3 ≤ x₁ ∧ x₁ ≤ Real.pi ∧
      -2 * Real.pi / 3 ≤ x₂ ∧ x₂ ≤ Real.pi ∧
      equation a x₁ ∧
      equation a x₂ ∧
      (∀ x, -2 * Real.pi / 3 ≤ x ∧ x ≤ Real.pi ∧ equation a x → x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_two_solutions_l3995_399544


namespace NUMINAMATH_CALUDE_triangle_properties_l3995_399556

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  -- Triangle ABC with sides a, b, c and angles A, B, C
  -- Vectors (a-b, 1) and (a-c, 2) are collinear
  (a - b) / (a - c) = 1 / 2 →
  -- Angle A is 120°
  A = 2 * π / 3 →
  -- Circumradius is 14
  R = 14 →
  -- Ratio a:b:c is 7:5:3
  ∃ (k : ℝ), a = 7 * k ∧ b = 5 * k ∧ c = 3 * k ∧
  -- Area of triangle ABC is 45√3
  1/2 * b * c * Real.sin A = 45 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3995_399556


namespace NUMINAMATH_CALUDE_history_paper_pages_l3995_399564

/-- Calculates the total number of pages in a paper given the number of days and pages per day. -/
def total_pages (days : ℕ) (pages_per_day : ℕ) : ℕ :=
  days * pages_per_day

/-- Proves that a paper due in 3 days, requiring 27 pages per day, has 81 pages in total. -/
theorem history_paper_pages : total_pages 3 27 = 81 := by
  sorry

end NUMINAMATH_CALUDE_history_paper_pages_l3995_399564


namespace NUMINAMATH_CALUDE_right_triangle_area_l3995_399575

theorem right_triangle_area (a : ℝ) (h : a > 0) :
  ∃ (R r : ℝ), R > 0 ∧ r > 0 ∧ R = (5/2) * r ∧
  (∃ (b c : ℝ), b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧
    (1/2 * a * b = (Real.sqrt 21 * a^2) / 6 ∨
     1/2 * a * b = (Real.sqrt 19 * a^2) / 22)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3995_399575


namespace NUMINAMATH_CALUDE_committee_selections_theorem_l3995_399587

/-- The number of ways to select a committee with at least one former member -/
def committee_selections_with_former (total_candidates : ℕ) (former_members : ℕ) (committee_size : ℕ) : ℕ :=
  Nat.choose total_candidates committee_size - Nat.choose (total_candidates - former_members) committee_size

/-- Theorem stating the number of committee selections with at least one former member -/
theorem committee_selections_theorem :
  committee_selections_with_former 15 6 4 = 1239 := by
  sorry

end NUMINAMATH_CALUDE_committee_selections_theorem_l3995_399587


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3995_399588

theorem quadratic_solution_sum (c d : ℝ) : 
  (c^2 - 6*c + 13 = 25) ∧ 
  (d^2 - 6*d + 13 = 25) ∧ 
  (c ≥ d) →
  c + 2*d = 9 - Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3995_399588


namespace NUMINAMATH_CALUDE_no_integer_solution_l3995_399531

theorem no_integer_solution : ¬∃ (a b : ℕ+), 
  (Real.sqrt a.val + Real.sqrt b.val = 10) ∧ 
  (Real.sqrt a.val * Real.sqrt b.val = 18) := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3995_399531


namespace NUMINAMATH_CALUDE_cousins_ages_sum_l3995_399585

def is_single_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

theorem cousins_ages_sum :
  ∀ (a b c d e : ℕ),
    is_single_digit a ∧ is_single_digit b ∧ is_single_digit c ∧ is_single_digit d ∧ is_single_digit e →
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e →
    (a * b = 36 ∨ a * c = 36 ∨ a * d = 36 ∨ a * e = 36 ∨ b * c = 36 ∨ b * d = 36 ∨ b * e = 36 ∨ c * d = 36 ∨ c * e = 36 ∨ d * e = 36) →
    (a * b = 40 ∨ a * c = 40 ∨ a * d = 40 ∨ a * e = 40 ∨ b * c = 40 ∨ b * d = 40 ∨ b * e = 40 ∨ c * d = 40 ∨ c * e = 40 ∨ d * e = 40) →
    a + b + c + d + e = 33 :=
by
  sorry

#check cousins_ages_sum

end NUMINAMATH_CALUDE_cousins_ages_sum_l3995_399585


namespace NUMINAMATH_CALUDE_cube_root_simplification_l3995_399503

theorem cube_root_simplification : (5488000 : ℝ)^(1/3) = 140 * 2^(1/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l3995_399503


namespace NUMINAMATH_CALUDE_three_digit_multiples_of_6_and_8_l3995_399539

theorem three_digit_multiples_of_6_and_8 : 
  (Finset.filter (fun n => n % 6 = 0 ∧ n % 8 = 0) (Finset.range 900 ∪ {999})).card = 37 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_multiples_of_6_and_8_l3995_399539


namespace NUMINAMATH_CALUDE_min_area_triangle_containing_unit_square_l3995_399567

/-- A triangle that contains a unit square. -/
structure TriangleContainingUnitSquare where
  /-- The area of the triangle. -/
  area : ℝ
  /-- The triangle contains a unit square. -/
  contains_unit_square : True

/-- The minimum area of a triangle containing a unit square is 2. -/
theorem min_area_triangle_containing_unit_square :
  ∀ t : TriangleContainingUnitSquare, t.area ≥ 2 ∧ ∃ t' : TriangleContainingUnitSquare, t'.area = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_area_triangle_containing_unit_square_l3995_399567


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3995_399507

theorem fraction_subtraction : (18 : ℚ) / 45 - 3 / 8 = 1 / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3995_399507


namespace NUMINAMATH_CALUDE_substitution_remainder_l3995_399570

/-- Represents the number of players in a soccer team --/
def total_players : ℕ := 22

/-- Represents the number of starting players --/
def starting_players : ℕ := 11

/-- Represents the number of substitute players --/
def substitute_players : ℕ := 11

/-- Represents the maximum number of substitutions allowed --/
def max_substitutions : ℕ := 4

/-- Calculates the number of ways to make substitutions in a soccer game --/
def substitution_ways : ℕ := 
  1 + 11^2 + 11^2 * 10^2 + 11^2 * 10^2 * 9^2 + 11^2 * 10^2 * 9^2 * 8^2

/-- Theorem stating that the remainder when the number of substitution ways
    is divided by 1000 is 722 --/
theorem substitution_remainder :
  substitution_ways % 1000 = 722 := by sorry

end NUMINAMATH_CALUDE_substitution_remainder_l3995_399570


namespace NUMINAMATH_CALUDE_largest_three_digit_geometric_l3995_399515

/-- Checks if a number is a three-digit integer -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Extracts the hundreds digit of a three-digit number -/
def hundredsDigit (n : ℕ) : ℕ := n / 100

/-- Extracts the tens digit of a three-digit number -/
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

/-- Extracts the ones digit of a three-digit number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- Checks if the digits of a three-digit number are distinct -/
def hasDistinctDigits (n : ℕ) : Prop :=
  let h := hundredsDigit n
  let t := tensDigit n
  let o := onesDigit n
  h ≠ t ∧ t ≠ o ∧ h ≠ o

/-- Checks if the digits of a three-digit number form a geometric sequence -/
def isGeometricSequence (n : ℕ) : Prop :=
  let h := hundredsDigit n
  let t := tensDigit n
  let o := onesDigit n
  ∃ r : ℚ, r ≠ 0 ∧ t = h / r ∧ o = t / r

theorem largest_three_digit_geometric : 
  ∀ n : ℕ, isThreeDigit n → hasDistinctDigits n → isGeometricSequence n → hundredsDigit n = 8 → n ≤ 842 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_geometric_l3995_399515


namespace NUMINAMATH_CALUDE_circle_tangent_to_y_axis_l3995_399577

def circle_equation (x y : ℝ) := (x - 2)^2 + (y - 1)^2 = 4

def is_tangent_to_y_axis (equation : ℝ → ℝ → Prop) : Prop :=
  ∃ y : ℝ, equation 0 y ∧ ∀ x y : ℝ, x ≠ 0 → equation x y → (x - 0)^2 + (y - y)^2 > 0

theorem circle_tangent_to_y_axis :
  is_tangent_to_y_axis circle_equation ∧
  ∀ x y : ℝ, circle_equation x y → (x - 2)^2 + (y - 1)^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_y_axis_l3995_399577


namespace NUMINAMATH_CALUDE_total_pears_theorem_l3995_399523

/-- Calculates the total number of pears picked over three days given the number of pears picked by each person in one day -/
def total_pears_over_three_days (jason keith mike alicia tina nicola : ℕ) : ℕ :=
  3 * (jason + keith + mike + alicia + tina + nicola)

/-- Theorem stating that given the specific number of pears picked by each person,
    the total number of pears picked over three days is 654 -/
theorem total_pears_theorem :
  total_pears_over_three_days 46 47 12 28 33 52 = 654 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_theorem_l3995_399523


namespace NUMINAMATH_CALUDE_zoo_birds_count_l3995_399541

theorem zoo_birds_count (non_bird_animals : ℕ) : 
  (5 * non_bird_animals = non_bird_animals + 360) → 
  (5 * non_bird_animals = 450) := by
sorry

end NUMINAMATH_CALUDE_zoo_birds_count_l3995_399541


namespace NUMINAMATH_CALUDE_parallel_condition_l3995_399548

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition for two lines to be parallel -/
def parallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- The first line: ax + y - a + 1 = 0 -/
def line1 (a : ℝ) : Line2D :=
  ⟨a, 1, -a + 1⟩

/-- The second line: 4x + ay - 2 = 0 -/
def line2 (a : ℝ) : Line2D :=
  ⟨4, a, -2⟩

/-- Statement: a = ±2 is a necessary but not sufficient condition for the lines to be parallel -/
theorem parallel_condition (a : ℝ) :
  (parallel (line1 a) (line2 a) → a = 2 ∨ a = -2) ∧
  ¬(a = 2 ∨ a = -2 → parallel (line1 a) (line2 a)) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l3995_399548


namespace NUMINAMATH_CALUDE_chocolate_ratio_l3995_399563

theorem chocolate_ratio (initial_chocolates : ℕ) (num_sisters : ℕ) (given_to_mother : ℕ) (eaten_by_father : ℕ) (left_with_father : ℕ) :
  initial_chocolates = 20 →
  num_sisters = 4 →
  given_to_mother = 3 →
  eaten_by_father = 2 →
  left_with_father = 5 →
  ∃ (chocolates_per_person : ℕ) (given_to_father : ℕ),
    chocolates_per_person * (num_sisters + 1) = initial_chocolates ∧
    given_to_father = left_with_father + given_to_mother + eaten_by_father ∧
    given_to_father * 2 = initial_chocolates :=
by sorry

end NUMINAMATH_CALUDE_chocolate_ratio_l3995_399563


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3995_399579

theorem sufficient_but_not_necessary (x : ℝ) : 
  (∀ x, 2 < x ∧ x < 4 → Real.log x < Real.exp 1) ∧
  (∃ x, Real.log x < Real.exp 1 ∧ ¬(2 < x ∧ x < 4)) := by
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3995_399579


namespace NUMINAMATH_CALUDE_inequality_proof_l3995_399598

theorem inequality_proof (a b c : ℝ) (h : a + b + c = 3) :
  1 / (5 * a^2 - 4 * a + 11) + 1 / (5 * b^2 - 4 * b + 11) + 1 / (5 * c^2 - 4 * c + 11) ≤ 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3995_399598


namespace NUMINAMATH_CALUDE_margaret_egg_collection_l3995_399566

/-- The number of groups Margaret's eggs can be organized into -/
def num_groups : ℕ := 5

/-- The number of eggs in each group -/
def eggs_per_group : ℕ := 7

/-- The total number of eggs in Margaret's collection -/
def total_eggs : ℕ := num_groups * eggs_per_group

theorem margaret_egg_collection : total_eggs = 35 := by
  sorry

end NUMINAMATH_CALUDE_margaret_egg_collection_l3995_399566


namespace NUMINAMATH_CALUDE_megan_snacks_l3995_399553

/-- The number of snacks Megan has in a given time period -/
def num_snacks (snack_interval : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / snack_interval

theorem megan_snacks : num_snacks 20 220 = 11 := by
  sorry

end NUMINAMATH_CALUDE_megan_snacks_l3995_399553


namespace NUMINAMATH_CALUDE_remaining_cards_l3995_399573

def initial_cards : ℕ := 13
def cards_given_away : ℕ := 9

theorem remaining_cards : initial_cards - cards_given_away = 4 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cards_l3995_399573


namespace NUMINAMATH_CALUDE_largest_angle_of_triangle_l3995_399518

theorem largest_angle_of_triangle (a b c : ℝ) : 
  a = 70 → b = 80 → c = 180 - a - b → a + b + c = 180 → max a (max b c) = 80 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_of_triangle_l3995_399518


namespace NUMINAMATH_CALUDE_labourer_income_l3995_399582

/-- Represents the financial situation of a labourer over a 10-month period. -/
structure LabourerFinances where
  monthly_income : ℝ
  first_period_length : ℕ := 6
  second_period_length : ℕ := 4
  first_period_expense : ℝ := 75
  second_period_expense : ℝ := 60
  savings : ℝ := 30

/-- The labourer's finances satisfy the given conditions. -/
def satisfies_conditions (f : LabourerFinances) : Prop :=
  (f.first_period_length * f.monthly_income < f.first_period_length * f.first_period_expense) ∧
  (f.second_period_length * f.monthly_income = 
    f.second_period_length * f.second_period_expense + 
    (f.first_period_length * f.first_period_expense - f.first_period_length * f.monthly_income) + 
    f.savings)

/-- The labourer's monthly income is 72 given the conditions. -/
theorem labourer_income (f : LabourerFinances) (h : satisfies_conditions f) : 
  f.monthly_income = 72 := by
  sorry


end NUMINAMATH_CALUDE_labourer_income_l3995_399582


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3995_399502

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) : 
  (∀ x, (1 - 3*x)^9 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                      a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| + |a₉| = 7^9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3995_399502


namespace NUMINAMATH_CALUDE_g_evaluation_l3995_399558

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem g_evaluation : 5 * g 2 + 4 * g (-2) = 186 := by
  sorry

end NUMINAMATH_CALUDE_g_evaluation_l3995_399558


namespace NUMINAMATH_CALUDE_coin_arrangement_exists_l3995_399549

/-- Represents a coin with mass, diameter, and minting year -/
structure Coin where
  mass : ℝ
  diameter : ℝ
  year : ℕ

/-- Represents a 3x3x3 arrangement of coins -/
def Arrangement := Fin 3 → Fin 3 → Fin 3 → Coin

/-- Checks if the arrangement satisfies the required conditions -/
def is_valid_arrangement (arr : Arrangement) : Prop :=
  ∀ i j k,
    (k < 2 → (arr i j k).mass < (arr i j (k+1)).mass) ∧
    (j < 2 → (arr i j k).diameter < (arr i (j+1) k).diameter) ∧
    (i < 2 → (arr i j k).year > (arr (i+1) j k).year)

theorem coin_arrangement_exists (coins : Fin 27 → Coin) 
  (h_distinct : ∀ i j, i ≠ j → coins i ≠ coins j) :
  ∃ (arr : Arrangement), is_valid_arrangement arr ∧ 
    ∀ i : Fin 27, ∃ x y z, arr x y z = coins i :=
sorry

end NUMINAMATH_CALUDE_coin_arrangement_exists_l3995_399549


namespace NUMINAMATH_CALUDE_walters_age_2001_l3995_399594

theorem walters_age_2001 (walter_age_1996 : ℕ) (grandmother_age_1996 : ℕ) :
  (grandmother_age_1996 = 3 * walter_age_1996) →
  (1996 - walter_age_1996 + 1996 - grandmother_age_1996 = 3864) →
  (walter_age_1996 + (2001 - 1996) = 37) :=
by sorry

end NUMINAMATH_CALUDE_walters_age_2001_l3995_399594


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_3_pow_4_l3995_399535

theorem units_digit_of_7_pow_3_pow_4 : ∃ n : ℕ, 7^(3^4) ≡ 7 [ZMOD 10] ∧ n < 10 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_3_pow_4_l3995_399535


namespace NUMINAMATH_CALUDE_equation_solution_l3995_399522

theorem equation_solution (x y : ℝ) :
  (x^4 + 1) * (y^4 + 1) = 4 * x^2 * y^2 ↔ (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3995_399522


namespace NUMINAMATH_CALUDE_arun_weight_upper_limit_l3995_399568

/-- The upper limit of Arun's weight according to his own opinion -/
def U : ℝ := sorry

/-- Arun's actual weight -/
def arun_weight : ℝ := sorry

/-- Arun's opinion: his weight is greater than 62 kg but less than U -/
axiom arun_opinion : 62 < arun_weight ∧ arun_weight < U

/-- Arun's brother's opinion: Arun's weight is greater than 60 kg but less than 70 kg -/
axiom brother_opinion : 60 < arun_weight ∧ arun_weight < 70

/-- Arun's mother's opinion: Arun's weight cannot be greater than 65 kg -/
axiom mother_opinion : arun_weight ≤ 65

/-- The average of different probable weights of Arun is 64 kg -/
axiom average_weight : (62 + U) / 2 = 64

theorem arun_weight_upper_limit : U = 65 := by sorry

end NUMINAMATH_CALUDE_arun_weight_upper_limit_l3995_399568


namespace NUMINAMATH_CALUDE_infinite_solutions_implies_integer_root_l3995_399574

/-- A polynomial of degree 3 with integer coefficients -/
def IntPolynomial3 : Type := ℤ → ℤ

/-- The property that xP(x) = yP(y) has infinitely many solutions for distinct integers x and y -/
def HasInfiniteSolutions (P : IntPolynomial3) : Prop :=
  ∀ n : ℕ, ∃ (x y : ℤ), x ≠ y ∧ x * P x = y * P y ∧ (∀ m < n, x ≠ m ∧ y ≠ m)

/-- The existence of an integer root for a polynomial -/
def HasIntegerRoot (P : IntPolynomial3) : Prop :=
  ∃ k : ℤ, P k = 0

/-- Main theorem: If a polynomial of degree 3 with integer coefficients has infinitely many
    solutions for xP(x) = yP(y) with distinct integers x and y, then it has an integer root -/
theorem infinite_solutions_implies_integer_root (P : IntPolynomial3) 
  (h : HasInfiniteSolutions P) : HasIntegerRoot P := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_implies_integer_root_l3995_399574


namespace NUMINAMATH_CALUDE_abc_signs_l3995_399547

theorem abc_signs (a b c : ℝ) 
  (h1 : (a > 0 ∧ b < 0 ∧ c = 0) ∨ 
        (a > 0 ∧ b = 0 ∧ c < 0) ∨ 
        (a < 0 ∧ b > 0 ∧ c = 0) ∨ 
        (a < 0 ∧ b = 0 ∧ c > 0) ∨ 
        (a = 0 ∧ b > 0 ∧ c < 0) ∨ 
        (a = 0 ∧ b < 0 ∧ c > 0))
  (h2 : a * b^2 * (a + c) * (b + c) < 0) :
  a > 0 ∧ b < 0 ∧ c = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_signs_l3995_399547


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l3995_399581

-- Define the hexagon
structure Hexagon :=
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (DE : ℝ)
  (EF : ℝ)
  (AC : ℝ)
  (AD : ℝ)
  (AE : ℝ)
  (AF : ℝ)

-- Define the theorem
theorem hexagon_perimeter (h : Hexagon) :
  h.AB = 1 →
  h.BC = 2 →
  h.CD = 2 →
  h.DE = 2 →
  h.EF = 3 →
  h.AC^2 = h.AB^2 + h.BC^2 →
  h.AD^2 = h.AC^2 + h.CD^2 →
  h.AE^2 = h.AD^2 + h.DE^2 →
  h.AF^2 = h.AE^2 + h.EF^2 →
  h.AB + h.BC + h.CD + h.DE + h.EF + h.AF = 10 + Real.sqrt 22 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l3995_399581


namespace NUMINAMATH_CALUDE_bakers_sales_l3995_399501

/-- Baker's cake and pastry sales problem -/
theorem bakers_sales (cakes_sold : ℕ) (pastries_sold : ℕ) 
  (h1 : cakes_sold = 158) 
  (h2 : cakes_sold = pastries_sold + 11) : 
  pastries_sold = 147 := by
  sorry

end NUMINAMATH_CALUDE_bakers_sales_l3995_399501


namespace NUMINAMATH_CALUDE_line_through_P_and_origin_line_through_P_perpendicular_to_l₃_l3995_399559

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x + y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2*x + y + 2 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-4, 6)

-- Define line l₃
def l₃ (x y : ℝ) : Prop := x - 3*y - 1 = 0

-- Theorem for the first line
theorem line_through_P_and_origin :
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧
  (∀ x y, a*x + b*y + c = 0 ↔ (x = P.1 ∧ y = P.2 ∨ x = 0 ∧ y = 0)) ∧
  a = 3 ∧ b = 2 ∧ c = 0 :=
sorry

-- Theorem for the second line
theorem line_through_P_perpendicular_to_l₃ :
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧
  (∀ x y, a*x + b*y + c = 0 ↔ (x = P.1 ∧ y = P.2)) ∧
  (a*(1 : ℝ) + b*(-3 : ℝ) = 0) ∧
  a = 3 ∧ b = 1 ∧ c = 6 :=
sorry

end NUMINAMATH_CALUDE_line_through_P_and_origin_line_through_P_perpendicular_to_l₃_l3995_399559


namespace NUMINAMATH_CALUDE_third_circle_radius_value_l3995_399519

/-- A sequence of six circles tangent to each other and to two parallel lines -/
structure TangentCircles where
  radii : Fin 6 → ℝ
  smallest_radius : radii 0 = 10
  largest_radius : radii 5 = 20
  tangent : ∀ i : Fin 5, radii i < radii (i + 1)

/-- The radius of the third circle from the smallest in the sequence -/
def third_circle_radius (tc : TangentCircles) : ℝ := tc.radii 2

/-- The theorem stating that the radius of the third circle is 10 · ⁵√4 -/
theorem third_circle_radius_value (tc : TangentCircles) :
  third_circle_radius tc = 10 * (4 : ℝ) ^ (1/5) :=
sorry

end NUMINAMATH_CALUDE_third_circle_radius_value_l3995_399519


namespace NUMINAMATH_CALUDE_rotation_result_l3995_399596

/-- Applies a 270° counter-clockwise rotation to a complex number -/
def rotate270 (z : ℂ) : ℂ := -Complex.I * z

/-- The initial complex number -/
def initial : ℂ := 4 - 2 * Complex.I

/-- The result of rotating the initial complex number by 270° counter-clockwise -/
def rotated : ℂ := rotate270 initial

/-- Theorem stating that rotating 4 - 2i by 270° counter-clockwise results in -4i - 2 -/
theorem rotation_result : rotated = -4 * Complex.I - 2 := by sorry

end NUMINAMATH_CALUDE_rotation_result_l3995_399596


namespace NUMINAMATH_CALUDE_count_sevens_1_to_100_l3995_399500

/-- Count of digit 7 in numbers from 1 to 100 -/
def countSevens : ℕ → ℕ
| 0 => 0
| (n + 1) => (if n + 1 < 101 then (if (n + 1) % 10 = 7 || (n + 1) / 10 = 7 then 1 else 0) else 0) + countSevens n

theorem count_sevens_1_to_100 : countSevens 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_count_sevens_1_to_100_l3995_399500


namespace NUMINAMATH_CALUDE_max_students_distribution_l3995_399509

theorem max_students_distribution (pens pencils notebooks erasers : ℕ) 
  (h1 : pens = 891) (h2 : pencils = 810) (h3 : notebooks = 1080) (h4 : erasers = 972) : 
  Nat.gcd pens (Nat.gcd pencils (Nat.gcd notebooks erasers)) = 27 := by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_l3995_399509


namespace NUMINAMATH_CALUDE_problem_statement_l3995_399557

theorem problem_statement (a b c : ℝ) (h : a + b + c = 0) :
  (a = 0 ∧ b = 0 ∧ c = 0 ↔ a * b + b * c + a * c = 0) ∧
  (a * b * c = 1 ∧ a ≥ b ∧ b ≥ c → c ≤ -Real.rpow 4 (1/3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3995_399557


namespace NUMINAMATH_CALUDE_line_segment_param_sum_l3995_399529

/-- Given a line segment connecting points (-3,10) and (4,16) represented by
    parametric equations x = at + b and y = ct + d where 0 ≤ t ≤ 1,
    and t = 0 corresponds to (-3,10), prove that a² + b² + c² + d² = 194 -/
theorem line_segment_param_sum (a b c d : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (b = -3 ∧ d = 10) →
  (a + b = 4 ∧ c + d = 16) →
  a^2 + b^2 + c^2 + d^2 = 194 := by
sorry

end NUMINAMATH_CALUDE_line_segment_param_sum_l3995_399529


namespace NUMINAMATH_CALUDE_acai_juice_cost_per_litre_l3995_399512

/-- The cost per litre of the superfruit juice cocktail -/
def cocktail_cost : ℝ := 1399.45

/-- The cost per litre of the mixed fruit juice -/
def mixed_juice_cost : ℝ := 262.85

/-- The volume of mixed fruit juice in litres -/
def mixed_juice_volume : ℝ := 32

/-- The volume of açaí berry juice in litres -/
def acai_juice_volume : ℝ := 21.333333333333332

/-- The total volume of the cocktail in litres -/
def total_volume : ℝ := mixed_juice_volume + acai_juice_volume

theorem acai_juice_cost_per_litre : 
  ∃ (acai_cost : ℝ),
    acai_cost = 3105.00 ∧
    mixed_juice_cost * mixed_juice_volume + acai_cost * acai_juice_volume = 
    cocktail_cost * total_volume :=
by sorry

end NUMINAMATH_CALUDE_acai_juice_cost_per_litre_l3995_399512


namespace NUMINAMATH_CALUDE_share_distribution_l3995_399591

theorem share_distribution (total : ℚ) (a b c : ℚ) : 
  total = 578 →
  a = (2/3) * b →
  b = (1/4) * c →
  a + b + c = total →
  a = 68 := by sorry

end NUMINAMATH_CALUDE_share_distribution_l3995_399591


namespace NUMINAMATH_CALUDE_comic_book_percentage_l3995_399534

theorem comic_book_percentage (total_books : ℕ) (novel_percentage : ℚ) (graphic_novels : ℕ) : 
  total_books = 120 →
  novel_percentage = 65 / 100 →
  graphic_novels = 18 →
  (total_books - (total_books * novel_percentage).floor - graphic_novels) / total_books = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_comic_book_percentage_l3995_399534


namespace NUMINAMATH_CALUDE_interval_equivalence_l3995_399530

-- Define the intervals as sets
def openRightInf (a : ℝ) : Set ℝ := {x | x > a}
def closedRightInf (a : ℝ) : Set ℝ := {x | x ≥ a}
def openLeftInf (b : ℝ) : Set ℝ := {x | x < b}
def closedLeftInf (b : ℝ) : Set ℝ := {x | x ≤ b}

-- State the theorem
theorem interval_equivalence (a b : ℝ) :
  (∀ x, x ∈ openRightInf a ↔ x > a) ∧
  (∀ x, x ∈ closedRightInf a ↔ x ≥ a) ∧
  (∀ x, x ∈ openLeftInf b ↔ x < b) ∧
  (∀ x, x ∈ closedLeftInf b ↔ x ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_interval_equivalence_l3995_399530


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3995_399586

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  1 / a + 3 / b ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 1 ∧ 1 / a₀ + 3 / b₀ = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3995_399586


namespace NUMINAMATH_CALUDE_symmetrical_cubic_function_l3995_399524

-- Define the function f(x) with parameters a and b
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + (a - 1) * x^2 + 48 * (a - 2) * x + b

-- Define the property of symmetry about the origin
def symmetrical_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem symmetrical_cubic_function
  (a b : ℝ)
  (h_symmetry : symmetrical_about_origin (f a b)) :
  (a = 1 ∧ b = 0) ∧
  (∀ x, f a b x = x^3 - 48*x) ∧
  (∀ x, -4 ≤ x ∧ x ≤ 4 → (∀ y, x < y → f a b x > f a b y)) ∧
  (∀ x, (x < -4 ∨ x > 4) → (∀ y, x < y → f a b x < f a b y)) ∧
  (f a b (-4) = 128) ∧
  (f a b 4 = -128) ∧
  (∀ x, f a b x ≤ 128) ∧
  (∀ x, f a b x ≥ -128) :=
by sorry

end NUMINAMATH_CALUDE_symmetrical_cubic_function_l3995_399524


namespace NUMINAMATH_CALUDE_bob_overspent_l3995_399517

theorem bob_overspent (necklace_cost book_cost total_spent limit : ℕ) : 
  necklace_cost = 34 →
  book_cost = necklace_cost + 5 →
  total_spent = necklace_cost + book_cost →
  limit = 70 →
  total_spent - limit = 3 := by
  sorry

end NUMINAMATH_CALUDE_bob_overspent_l3995_399517


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3995_399572

theorem sufficient_not_necessary (a : ℝ) : 
  (a < -1 → |a| > 1) ∧ ¬(|a| > 1 → a < -1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3995_399572


namespace NUMINAMATH_CALUDE_sphere_surface_area_circumscribed_cube_l3995_399571

theorem sphere_surface_area_circumscribed_cube (edge_length : ℝ) 
  (h : edge_length = 2) : 
  4 * Real.pi * (edge_length * Real.sqrt 3 / 2) ^ 2 = 12 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_circumscribed_cube_l3995_399571


namespace NUMINAMATH_CALUDE_lowest_degree_is_four_l3995_399593

/-- A polynomial with coefficients in ℤ -/
def IntPolynomial := ℕ → ℤ

/-- The degree of an IntPolynomial -/
def degree (p : IntPolynomial) : ℕ := sorry

/-- The set of coefficients of an IntPolynomial -/
def coeffSet (p : IntPolynomial) : Set ℤ := sorry

/-- Predicate for a polynomial satisfying the given conditions -/
def satisfiesCondition (p : IntPolynomial) : Prop :=
  ∃ b : ℤ, (∃ x ∈ coeffSet p, x < b) ∧ 
           (∃ y ∈ coeffSet p, y > b) ∧ 
           b ∉ coeffSet p

/-- The main theorem statement -/
theorem lowest_degree_is_four :
  ∃ p : IntPolynomial, satisfiesCondition p ∧ degree p = 4 ∧
  ∀ q : IntPolynomial, satisfiesCondition q → degree q ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_lowest_degree_is_four_l3995_399593


namespace NUMINAMATH_CALUDE_complex_number_location_l3995_399542

theorem complex_number_location :
  let z : ℂ := Complex.I / (3 - 3 * Complex.I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l3995_399542


namespace NUMINAMATH_CALUDE_custom_op_result_l3995_399513

/-- Custom operation € -/
def custom_op (x y : ℝ) : ℝ := 3 * x * y - x - y

/-- Theorem stating the result of the custom operation -/
theorem custom_op_result : 
  let x : ℝ := 6
  let y : ℝ := 4
  let z : ℝ := 2
  custom_op x (custom_op y z) = 300 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l3995_399513


namespace NUMINAMATH_CALUDE_symmetric_trapezoid_construction_l3995_399589

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a trapezoid
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define symmetry for a trapezoid
def isSymmetricTrapezoid (t : Trapezoid) : Prop :=
  -- Add conditions for symmetry here
  sorry

-- Define the construction function
def constructSymmetricTrapezoid (c : Circle) (sideLength : ℝ) : Trapezoid :=
  sorry

-- Theorem statement
theorem symmetric_trapezoid_construction
  (c : Circle) (sideLength : ℝ) :
  isSymmetricTrapezoid (constructSymmetricTrapezoid c sideLength) :=
sorry

end NUMINAMATH_CALUDE_symmetric_trapezoid_construction_l3995_399589


namespace NUMINAMATH_CALUDE_area_eq_product_segments_l3995_399580

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithIncircle where
  /-- The length of one leg of the right triangle -/
  a : ℝ
  /-- The length of the other leg of the right triangle -/
  b : ℝ
  /-- The length of the hypotenuse of the right triangle -/
  c : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of one segment of the hypotenuse divided by the point of tangency -/
  m : ℝ
  /-- The length of the other segment of the hypotenuse divided by the point of tangency -/
  n : ℝ
  /-- The hypotenuse is the sum of its segments -/
  hyp_sum : c = m + n
  /-- The triangle satisfies the Pythagorean theorem -/
  pythagorean : a^2 + b^2 = c^2
  /-- All lengths are positive -/
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  pos_r : r > 0
  pos_m : m > 0
  pos_n : n > 0

/-- The area of a right triangle with an inscribed circle is equal to the product of the 
    lengths of the segments into which the hypotenuse is divided by the point of tangency 
    with the incircle -/
theorem area_eq_product_segments (t : RightTriangleWithIncircle) : 
  (1/2) * t.a * t.b = t.m * t.n := by
  sorry

end NUMINAMATH_CALUDE_area_eq_product_segments_l3995_399580


namespace NUMINAMATH_CALUDE_fraction_depends_on_z_l3995_399540

theorem fraction_depends_on_z (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (4 * x + 2 * y) / (x - 4 * y) = 3) :
  ∃ z₁ z₂ : ℝ, z₁ ≠ z₂ ∧ 
    (x + 4 * y + z₁) / (4 * x - y - z₁) ≠ (x + 4 * y + z₂) / (4 * x - y - z₂) :=
by sorry

end NUMINAMATH_CALUDE_fraction_depends_on_z_l3995_399540


namespace NUMINAMATH_CALUDE_clock_hands_right_angles_l3995_399527

/-- Represents the number of times clock hands are at right angles in one hour -/
def right_angles_per_hour : ℕ := 2

/-- Represents the number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Represents the number of days -/
def num_days : ℕ := 5

/-- Theorem: The hands of a clock are at right angles 240 times in 5 days -/
theorem clock_hands_right_angles :
  right_angles_per_hour * hours_per_day * num_days = 240 := by
  sorry

end NUMINAMATH_CALUDE_clock_hands_right_angles_l3995_399527


namespace NUMINAMATH_CALUDE_no_four_digit_numbers_sum_10_div_9_l3995_399555

theorem no_four_digit_numbers_sum_10_div_9 : 
  ¬∃ (n : ℕ), 
    1000 ≤ n ∧ n < 10000 ∧ 
    (∃ (a b c d : ℕ), n = 1000*a + 100*b + 10*c + d ∧ a + b + c + d = 10) ∧
    n % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_four_digit_numbers_sum_10_div_9_l3995_399555


namespace NUMINAMATH_CALUDE_point_satisfies_conditions_l3995_399525

def point (m : ℝ) : ℝ × ℝ := (2 - m, 2 * m - 1)

def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  |p.1|

theorem point_satisfies_conditions (m : ℝ) :
  in_fourth_quadrant (point m) ∧
  distance_to_y_axis (point m) = 3 →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_point_satisfies_conditions_l3995_399525


namespace NUMINAMATH_CALUDE_no_solution_rebus_l3995_399505

theorem no_solution_rebus :
  ¬ ∃ (K U S Y : ℕ),
    K ≠ U ∧ K ≠ S ∧ K ≠ Y ∧ U ≠ S ∧ U ≠ Y ∧ S ≠ Y ∧
    K < 10 ∧ U < 10 ∧ S < 10 ∧ Y < 10 ∧
    1000 ≤ (1000 * K + 100 * U + 10 * S + Y) ∧
    (1000 * K + 100 * U + 10 * S + Y) < 10000 ∧
    1000 ≤ (1000 * U + 100 * K + 10 * S + Y) ∧
    (1000 * U + 100 * K + 10 * S + Y) < 10000 ∧
    10000 ≤ (10000 * U + 1000 * K + 100 * S + 10 * U + S) ∧
    (10000 * U + 1000 * K + 100 * S + 10 * U + S) < 100000 ∧
    (1000 * K + 100 * U + 10 * S + Y) + (1000 * U + 100 * K + 10 * S + Y) =
    (10000 * U + 1000 * K + 100 * S + 10 * U + S) :=
by
  sorry

end NUMINAMATH_CALUDE_no_solution_rebus_l3995_399505


namespace NUMINAMATH_CALUDE_certain_number_problem_l3995_399536

theorem certain_number_problem (y : ℝ) : 
  (0.25 * 780 = 0.15 * y - 30) → y = 1500 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3995_399536


namespace NUMINAMATH_CALUDE_smallest_integer_gcd_18_is_6_l3995_399599

theorem smallest_integer_gcd_18_is_6 : 
  ∃ n : ℕ, n > 100 ∧ n.gcd 18 = 6 ∧ ∀ m : ℕ, m > 100 ∧ m.gcd 18 = 6 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_gcd_18_is_6_l3995_399599


namespace NUMINAMATH_CALUDE_sum_of_integers_and_squares_l3995_399546

-- Define the sum of integers from a to b, inclusive
def sumIntegers (a b : Int) : Int :=
  (b - a + 1) * (a + b) / 2

-- Define the sum of squares from a to b, inclusive
def sumSquares (a b : Int) : Int :=
  (b * (b + 1) * (2 * b + 1) - (a - 1) * a * (2 * a - 1)) / 6

theorem sum_of_integers_and_squares : 
  sumIntegers (-50) 40 + sumSquares 10 40 = 21220 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_and_squares_l3995_399546


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l3995_399597

theorem max_value_of_trig_function :
  let f (x : ℝ) := Real.tan (x + 3 * Real.pi / 4) - Real.tan x + Real.sin (x + Real.pi / 4)
  ∀ x ∈ Set.Icc (-3 * Real.pi / 4) (-Real.pi / 2),
    f x ≤ 0 ∧ ∃ x₀ ∈ Set.Icc (-3 * Real.pi / 4) (-Real.pi / 2), f x₀ = 0 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l3995_399597


namespace NUMINAMATH_CALUDE_x_minus_y_values_l3995_399569

theorem x_minus_y_values (x y : ℝ) (h1 : |x| = 4) (h2 : |y| = 5) (h3 : x > y) :
  x - y = 9 ∨ x - y = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l3995_399569


namespace NUMINAMATH_CALUDE_problem_solution_l3995_399552

theorem problem_solution (x : ℝ) (h : 5 * x - 7 = 15 * x + 13) : 3 * (x + 4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3995_399552


namespace NUMINAMATH_CALUDE_min_m_value_l3995_399583

/-- Given a function f(x) = 2^(|x-a|) where a ∈ ℝ, if f(1+x) = f(1-x) for all x ∈ ℝ 
    and f(x) is monotonically increasing on [m,+∞), then the minimum value of m is 1. -/
theorem min_m_value (a : ℝ) (f : ℝ → ℝ) (m : ℝ) 
    (h1 : ∀ x, f x = 2^(|x - a|))
    (h2 : ∀ x, f (1 + x) = f (1 - x))
    (h3 : MonotoneOn f (Set.Ici m)) :
  ∃ m₀ : ℝ, m₀ = 1 ∧ ∀ m' : ℝ, (∀ x ≥ m', MonotoneOn f (Set.Ici x)) → m' ≥ m₀ :=
sorry

end NUMINAMATH_CALUDE_min_m_value_l3995_399583


namespace NUMINAMATH_CALUDE_alex_cell_phone_cost_l3995_399562

/-- Represents the cell phone plan cost structure and usage --/
structure CellPhonePlan where
  baseCost : ℝ
  textCost : ℝ
  extraMinuteCost : ℝ
  freeHours : ℝ
  textsSent : ℝ
  hoursUsed : ℝ

/-- Calculates the total cost of the cell phone plan --/
def totalCost (plan : CellPhonePlan) : ℝ :=
  plan.baseCost +
  plan.textCost * plan.textsSent +
  plan.extraMinuteCost * (plan.hoursUsed - plan.freeHours) * 60

/-- Theorem stating that Alex's total cost is $45.00 --/
theorem alex_cell_phone_cost :
  let plan : CellPhonePlan := {
    baseCost := 30
    textCost := 0.04
    extraMinuteCost := 0.15
    freeHours := 25
    textsSent := 150
    hoursUsed := 26
  }
  totalCost plan = 45 := by
  sorry


end NUMINAMATH_CALUDE_alex_cell_phone_cost_l3995_399562


namespace NUMINAMATH_CALUDE_max_container_weight_for_transport_l3995_399584

/-- Represents a container with a weight in tons -/
structure Container where
  weight : ℕ

/-- Represents a platform with a maximum load capacity -/
structure Platform where
  capacity : ℕ

/-- Represents a train with a number of platforms -/
structure Train where
  platforms : List Platform

/-- Checks if a given configuration of containers can be loaded onto a train -/
def canLoad (containers : List Container) (train : Train) : Prop :=
  sorry

/-- The main theorem stating that 26 is the maximum container weight that guarantees
    1500 tons can be transported -/
theorem max_container_weight_for_transport
  (total_weight : ℕ)
  (num_platforms : ℕ)
  (platform_capacity : ℕ)
  (h_total_weight : total_weight = 1500)
  (h_num_platforms : num_platforms = 25)
  (h_platform_capacity : platform_capacity = 80)
  : (∃ k : ℕ, k = 26 ∧
    (∀ containers : List Container,
      (∀ c ∈ containers, c.weight ≤ k ∧ c.weight > 0) →
      (containers.map (λ c => c.weight)).sum = total_weight →
      canLoad containers (Train.mk (List.replicate num_platforms (Platform.mk platform_capacity)))) ∧
    (∀ k' > k, ∃ containers : List Container,
      (∀ c ∈ containers, c.weight ≤ k' ∧ c.weight > 0) ∧
      (containers.map (λ c => c.weight)).sum = total_weight ∧
      ¬canLoad containers (Train.mk (List.replicate num_platforms (Platform.mk platform_capacity))))) :=
  sorry

end NUMINAMATH_CALUDE_max_container_weight_for_transport_l3995_399584


namespace NUMINAMATH_CALUDE_soccer_substitutions_remainder_l3995_399538

/-- The number of ways to make substitutions in a soccer game -/
def substitution_ways (total_players start_players max_substitutions : ℕ) : ℕ :=
  sorry

/-- The main theorem about the remainder of substitution ways when divided by 1000 -/
theorem soccer_substitutions_remainder :
  let total_players := 22
  let start_players := 11
  let max_substitutions := 4
  (substitution_ways total_players start_players max_substitutions) % 1000 = 122 := by
  sorry

end NUMINAMATH_CALUDE_soccer_substitutions_remainder_l3995_399538


namespace NUMINAMATH_CALUDE_two_dogs_weekly_distance_l3995_399550

/-- The total distance walked by two dogs in a week, given their daily walking distances -/
def total_weekly_distance (dog1_daily : ℕ) (dog2_daily : ℕ) : ℕ :=
  (dog1_daily * 7) + (dog2_daily * 7)

/-- Theorem: The total distance walked by two dogs in a week is 70 miles -/
theorem two_dogs_weekly_distance :
  total_weekly_distance 2 8 = 70 := by
  sorry

end NUMINAMATH_CALUDE_two_dogs_weekly_distance_l3995_399550


namespace NUMINAMATH_CALUDE_factorization_m_squared_plus_3m_l3995_399506

theorem factorization_m_squared_plus_3m (m : ℝ) : m^2 + 3*m = m*(m+3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_m_squared_plus_3m_l3995_399506


namespace NUMINAMATH_CALUDE_triangle_area_implies_p_value_l3995_399543

/-- Given a triangle ABC with vertices A(3, 15), B(15, 0), and C(0, p),
    prove that if the area of the triangle is 35, then p = 77.5/6 -/
theorem triangle_area_implies_p_value (p : ℝ) : 
  let A : ℝ × ℝ := (3, 15)
  let B : ℝ × ℝ := (15, 0)
  let C : ℝ × ℝ := (0, p)
  let triangle_area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  triangle_area = 35 → p = 77.5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_implies_p_value_l3995_399543


namespace NUMINAMATH_CALUDE_count_equality_l3995_399514

/-- The count of natural numbers from 1 to 3998 that are divisible by 4 -/
def count_divisible_by_4 : ℕ := 999

/-- The count of natural numbers from 1 to 3998 whose digit sum is divisible by 4 -/
def count_digit_sum_divisible_by_4 : ℕ := 999

/-- The upper bound of the range of natural numbers being considered -/
def upper_bound : ℕ := 3998

theorem count_equality :
  count_divisible_by_4 = count_digit_sum_divisible_by_4 ∧
  count_divisible_by_4 = (upper_bound / 4 : ℕ) :=
sorry

end NUMINAMATH_CALUDE_count_equality_l3995_399514


namespace NUMINAMATH_CALUDE_cubic_three_zeros_l3995_399533

/-- A function f(x) = x^3 - 3x + a has three distinct zeros if and only if -2 < a < 2 -/
theorem cubic_three_zeros (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 3*x + a = 0 ∧ y^3 - 3*y + a = 0 ∧ z^3 - 3*z + a = 0) ↔
  -2 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_three_zeros_l3995_399533


namespace NUMINAMATH_CALUDE_spatial_relations_theorem_l3995_399578

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Parallel relation between a line and a plane -/
def line_parallel_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between two planes -/
def plane_parallel_plane (p1 p2 : Plane3D) : Prop :=
  sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between two lines -/
def line_parallel_line (l1 l2 : Line3D) : Prop :=
  sorry

/-- Perpendicular relation between a line and a plane -/
def line_perp_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

theorem spatial_relations_theorem 
  (m n : Line3D) 
  (α β : Plane3D) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (∀ l : Line3D, line_parallel_plane m α → 
    (line_in_plane l α → line_parallel_line m l)) ∧
  (¬ (plane_parallel_plane α β → line_in_plane m α → 
    line_in_plane n β → line_parallel_line m n)) ∧
  (line_perp_plane m α → line_perp_plane n β → 
    line_parallel_line m n → plane_parallel_plane α β) ∧
  (plane_parallel_plane α β → line_in_plane m α → 
    line_parallel_plane m β) :=
by sorry

end NUMINAMATH_CALUDE_spatial_relations_theorem_l3995_399578


namespace NUMINAMATH_CALUDE_susan_gave_eight_apples_l3995_399592

/-- The number of apples Susan gave to Sean -/
def apples_from_susan (initial_apples final_apples : ℕ) : ℕ :=
  final_apples - initial_apples

/-- Theorem stating that Susan gave Sean 8 apples -/
theorem susan_gave_eight_apples (initial_apples final_apples : ℕ) 
  (h1 : initial_apples = 9)
  (h2 : final_apples = 17) :
  apples_from_susan initial_apples final_apples = 8 := by
  sorry

end NUMINAMATH_CALUDE_susan_gave_eight_apples_l3995_399592


namespace NUMINAMATH_CALUDE_arrangements_theorem_l3995_399511

def number_of_arrangements (n : ℕ) (a_not_first : Bool) (b_not_last : Bool) : ℕ :=
  if n = 5 ∧ a_not_first ∧ b_not_last then
    78
  else
    0

theorem arrangements_theorem :
  ∀ (n : ℕ) (a_not_first b_not_last : Bool),
    n = 5 → a_not_first → b_not_last →
    number_of_arrangements n a_not_first b_not_last = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_arrangements_theorem_l3995_399511


namespace NUMINAMATH_CALUDE_regular_star_points_l3995_399595

/-- A p-pointed regular star with specific angle properties -/
structure RegularStar where
  p : ℕ
  angle_d : ℝ
  angle_c : ℝ
  angle_c_minus_d : angle_c = angle_d + 15
  sum_of_angles : p * angle_c + p * angle_d = 360

/-- The number of points in a regular star with given properties is 24 -/
theorem regular_star_points (star : RegularStar) : star.p = 24 := by
  sorry

end NUMINAMATH_CALUDE_regular_star_points_l3995_399595


namespace NUMINAMATH_CALUDE_father_current_age_l3995_399565

/-- The father's age at the son's birth equals the son's current age -/
def father_age_at_son_birth (father_age_now son_age_now : ℕ) : Prop :=
  father_age_now - son_age_now = son_age_now

/-- The son's age 5 years ago was 26 -/
def son_age_five_years_ago (son_age_now : ℕ) : Prop :=
  son_age_now - 5 = 26

/-- Theorem stating that the father's current age is 62 years -/
theorem father_current_age :
  ∀ (father_age_now son_age_now : ℕ),
    father_age_at_son_birth father_age_now son_age_now →
    son_age_five_years_ago son_age_now →
    father_age_now = 62 :=
by
  sorry

end NUMINAMATH_CALUDE_father_current_age_l3995_399565


namespace NUMINAMATH_CALUDE_paint_usage_l3995_399537

theorem paint_usage (initial_paint : ℝ) (first_week_fraction : ℝ) (second_week_fraction : ℝ)
  (h1 : initial_paint = 360)
  (h2 : first_week_fraction = 1/4)
  (h3 : second_week_fraction = 1/3) :
  let first_week_usage := first_week_fraction * initial_paint
  let remaining_paint := initial_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  let total_usage := first_week_usage + second_week_usage
  total_usage = 180 := by
  sorry

end NUMINAMATH_CALUDE_paint_usage_l3995_399537


namespace NUMINAMATH_CALUDE_no_y_intercepts_l3995_399545

/-- The parabola equation -/
def parabola_equation (y : ℝ) : ℝ := 3 * y^2 - 5 * y + 9

/-- Theorem: The parabola x = 3y^2 - 5y + 9 has no y-intercepts -/
theorem no_y_intercepts : ¬ ∃ y : ℝ, parabola_equation y = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_y_intercepts_l3995_399545


namespace NUMINAMATH_CALUDE_zero_in_interval_l3995_399576

def f (x : ℝ) := x^3 + 2*x - 2

theorem zero_in_interval :
  ∃ c ∈ Set.Icc 0 1, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l3995_399576


namespace NUMINAMATH_CALUDE_pairing_count_l3995_399520

/-- The number of bowls -/
def num_bowls : ℕ := 6

/-- The number of glasses -/
def num_glasses : ℕ := 4

/-- The number of fixed pairings -/
def num_fixed_pairings : ℕ := 1

/-- The number of remaining bowls after fixed pairing -/
def num_remaining_bowls : ℕ := num_bowls - num_fixed_pairings

/-- The number of remaining glasses after fixed pairing -/
def num_remaining_glasses : ℕ := num_glasses - num_fixed_pairings

/-- The total number of possible pairings -/
def total_pairings : ℕ := num_remaining_bowls * num_remaining_glasses + num_fixed_pairings

theorem pairing_count : total_pairings = 16 := by
  sorry

end NUMINAMATH_CALUDE_pairing_count_l3995_399520


namespace NUMINAMATH_CALUDE_canada_population_1998_l3995_399521

/-- The population of Canada in 1998 in millions -/
def canada_population_millions : ℝ := 30.3

/-- One million in standard form -/
def million : ℕ := 1000000

/-- Theorem: The population of Canada in 1998 was 30,300,000 -/
theorem canada_population_1998 : 
  (canada_population_millions * million : ℝ) = 30300000 := by sorry

end NUMINAMATH_CALUDE_canada_population_1998_l3995_399521


namespace NUMINAMATH_CALUDE_order_of_trigonometric_functions_l3995_399590

theorem order_of_trigonometric_functions : 
  let a := Real.sin (Real.sin (2008 * π / 180))
  let b := Real.sin (Real.cos (2008 * π / 180))
  let c := Real.cos (Real.sin (2008 * π / 180))
  let d := Real.cos (Real.cos (2008 * π / 180))
  b < a ∧ a < d ∧ d < c := by sorry

end NUMINAMATH_CALUDE_order_of_trigonometric_functions_l3995_399590


namespace NUMINAMATH_CALUDE_carla_drink_problem_l3995_399560

/-- The amount of water Carla drank in ounces -/
def water_amount : ℝ := 15

/-- The total amount of liquid Carla drank in ounces -/
def total_amount : ℝ := 54

/-- The amount of soda Carla drank in ounces -/
def soda_amount (x : ℝ) : ℝ := 3 * water_amount - x

theorem carla_drink_problem :
  ∃ x : ℝ, x = 6 ∧ water_amount + soda_amount x = total_amount := by sorry

end NUMINAMATH_CALUDE_carla_drink_problem_l3995_399560


namespace NUMINAMATH_CALUDE_signal_count_theorem_l3995_399508

/-- Represents the number of indicator lights --/
def num_lights : Nat := 6

/-- Represents the number of lights that light up each time --/
def lights_lit : Nat := 3

/-- Represents the number of possible colors for each light --/
def num_colors : Nat := 3

/-- Calculates the total number of different signals that can be displayed --/
def total_signals : Nat :=
  -- The actual calculation is not provided, so we use a placeholder
  324

/-- Theorem stating that the total number of different signals is 324 --/
theorem signal_count_theorem :
  total_signals = 324 := by
  sorry

end NUMINAMATH_CALUDE_signal_count_theorem_l3995_399508


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l3995_399526

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- Theorem statement
theorem opposite_of_negative_five :
  opposite (-5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l3995_399526


namespace NUMINAMATH_CALUDE_l_structure_surface_area_l3995_399516

/-- Represents the L-shaped structure -/
structure LStructure where
  bottom_length : ℕ
  bottom_width : ℕ
  stack_height : ℕ

/-- Calculates the surface area of the L-shaped structure -/
def surface_area (l : LStructure) : ℕ :=
  let bottom_area := l.bottom_length * l.bottom_width
  let bottom_perimeter := 2 * l.bottom_length + l.bottom_width
  let stack_side_area := 2 * l.stack_height
  let stack_top_area := 1
  bottom_area + bottom_perimeter + stack_side_area + stack_top_area

/-- The specific L-shaped structure in the problem -/
def problem_structure : LStructure :=
  { bottom_length := 3
  , bottom_width := 3
  , stack_height := 6 }

theorem l_structure_surface_area :
  surface_area problem_structure = 29 := by
  sorry

end NUMINAMATH_CALUDE_l_structure_surface_area_l3995_399516


namespace NUMINAMATH_CALUDE_f_zero_at_three_l3995_399532

-- Define the function f
def f (x r : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 5 * x + r

-- State the theorem
theorem f_zero_at_three (r : ℝ) : f 3 r = 0 ↔ r = -273 := by sorry

end NUMINAMATH_CALUDE_f_zero_at_three_l3995_399532


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3995_399561

theorem inequality_solution_set (x : ℝ) : 
  (x - 20) / (x + 16) ≤ 0 ↔ -16 < x ∧ x ≤ 20 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3995_399561


namespace NUMINAMATH_CALUDE_range_of_m_l3995_399510

-- Define the sets
def set1 (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - 4 ≤ 0 ∧ p.2 ≥ 0 ∧ m * p.1 - p.2 ≥ 0 ∧ m > 0}

def set2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 2)^2 ≤ 8}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (set1 m ⊆ set2) → (0 < m ∧ m ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3995_399510
