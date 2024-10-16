import Mathlib

namespace NUMINAMATH_CALUDE_wand_cost_proof_l3752_375267

/-- The cost of each wand --/
def wand_cost : ℚ := 115 / 3

/-- The number of wands Kate bought --/
def num_wands : ℕ := 3

/-- The additional amount Kate charged when selling each wand --/
def additional_charge : ℚ := 5

/-- The total amount Kate collected after selling all wands --/
def total_collected : ℚ := 130

theorem wand_cost_proof : 
  num_wands * (wand_cost + additional_charge) = total_collected :=
sorry

end NUMINAMATH_CALUDE_wand_cost_proof_l3752_375267


namespace NUMINAMATH_CALUDE_john_jury_duty_days_l3752_375247

/-- The number of days John spends on jury duty -/
def jury_duty_days (jury_selection_days : ℕ) (trial_multiplier : ℕ) 
  (deliberation_days : ℕ) (deliberation_hours_per_day : ℕ) (hours_per_day : ℕ) : ℕ :=
  jury_selection_days + 
  (trial_multiplier * jury_selection_days) + 
  (deliberation_days * deliberation_hours_per_day) / hours_per_day

/-- Theorem stating that John spends 14 days on jury duty -/
theorem john_jury_duty_days : 
  jury_duty_days 2 4 6 16 24 = 14 := by
  sorry

end NUMINAMATH_CALUDE_john_jury_duty_days_l3752_375247


namespace NUMINAMATH_CALUDE_infinite_triples_theorem_l3752_375259

def is_sum_of_two_squares (n : ℕ) : Prop := ∃ a b : ℕ, n = a^2 + b^2

theorem infinite_triples_theorem :
  (∃ f : ℕ → ℕ, ∀ m : ℕ,
    is_sum_of_two_squares (2 * 100^(f m)) ∧
    ¬is_sum_of_two_squares (2 * 100^(f m) - 1) ∧
    ¬is_sum_of_two_squares (2 * 100^(f m) + 1)) ∧
  (∃ g : ℕ → ℕ, ∀ m : ℕ,
    is_sum_of_two_squares (2 * (g m^2 - g m)^2 + 1) ∧
    is_sum_of_two_squares (2 * (g m^2 - g m)^2) ∧
    is_sum_of_two_squares (2 * (g m^2 - g m)^2 + 2)) :=
by sorry

end NUMINAMATH_CALUDE_infinite_triples_theorem_l3752_375259


namespace NUMINAMATH_CALUDE_scientific_notation_45400_l3752_375260

theorem scientific_notation_45400 : 
  ∃ (a : ℝ) (n : ℤ), 45400 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 4.54 ∧ n = 4 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_45400_l3752_375260


namespace NUMINAMATH_CALUDE_remainder_mod_seven_l3752_375234

theorem remainder_mod_seven : (9^5 + 8^4 + 7^9) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_mod_seven_l3752_375234


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3752_375227

theorem max_value_sqrt_sum (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_eq_two : x + y + z = 2) : 
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 2 ∧ 
    Real.sqrt a + Real.sqrt (2 * b) + Real.sqrt (3 * c) > Real.sqrt x + Real.sqrt (2 * y) + Real.sqrt (3 * z)) 
  ∨ 
  Real.sqrt x + Real.sqrt (2 * y) + Real.sqrt (3 * z) = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3752_375227


namespace NUMINAMATH_CALUDE_cat_food_sale_calculation_l3752_375215

/-- Theorem: Cat Food Sale Calculation
Given:
- 20 people bought cat food
- First 8 customers bought 3 cases each
- Next 4 customers bought 2 cases each
- Last 8 customers bought 1 case each

Prove: The total number of cases of cat food sold is 40.
-/
theorem cat_food_sale_calculation (total_customers : Nat) 
  (first_group_size : Nat) (first_group_cases : Nat)
  (second_group_size : Nat) (second_group_cases : Nat)
  (third_group_size : Nat) (third_group_cases : Nat)
  (h1 : total_customers = 20)
  (h2 : first_group_size = 8)
  (h3 : first_group_cases = 3)
  (h4 : second_group_size = 4)
  (h5 : second_group_cases = 2)
  (h6 : third_group_size = 8)
  (h7 : third_group_cases = 1)
  (h8 : total_customers = first_group_size + second_group_size + third_group_size) :
  first_group_size * first_group_cases + 
  second_group_size * second_group_cases + 
  third_group_size * third_group_cases = 40 := by
  sorry

end NUMINAMATH_CALUDE_cat_food_sale_calculation_l3752_375215


namespace NUMINAMATH_CALUDE_percentage_employed_females_l3752_375272

/-- Given that 64% of the population are employed and 50% of the population are employed males,
    prove that 21.875% of the employed people are females. -/
theorem percentage_employed_females
  (total_population : ℝ)
  (employed_percentage : ℝ)
  (employed_males_percentage : ℝ)
  (h1 : employed_percentage = 64)
  (h2 : employed_males_percentage = 50)
  : (employed_percentage - employed_males_percentage) / employed_percentage * 100 = 21.875 := by
  sorry

end NUMINAMATH_CALUDE_percentage_employed_females_l3752_375272


namespace NUMINAMATH_CALUDE_xyz_sum_sqrt_l3752_375265

theorem xyz_sum_sqrt (x y z : ℝ) 
  (eq1 : y + z = 15)
  (eq2 : z + x = 17)
  (eq3 : x + y = 16) :
  Real.sqrt (x * y * z * (x + y + z)) = 72 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_sqrt_l3752_375265


namespace NUMINAMATH_CALUDE_subtract_decimals_l3752_375246

theorem subtract_decimals : (145.23 : ℝ) - 0.07 = 145.16 := by
  sorry

end NUMINAMATH_CALUDE_subtract_decimals_l3752_375246


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3752_375299

theorem quadratic_factorization (a b : ℤ) :
  (∀ x : ℝ, 20 * x^2 - 90 * x - 22 = (5 * x + a) * (4 * x + b)) →
  a + 3 * b = -65 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3752_375299


namespace NUMINAMATH_CALUDE_monic_cubic_polynomial_uniqueness_l3752_375286

/-- A monic cubic polynomial with real coefficients -/
def MonicCubicPolynomial (a b c : ℝ) (x : ℂ) : ℂ := x^3 + a*x^2 + b*x + c

theorem monic_cubic_polynomial_uniqueness (a b c : ℝ) :
  let q := MonicCubicPolynomial a b c
  (q (2 - I) = 0) → (q 0 = -40) →
  (∀ x, q x = x^3 - (61/4)*x^2 + (305/4)*x - 225/4) :=
sorry

end NUMINAMATH_CALUDE_monic_cubic_polynomial_uniqueness_l3752_375286


namespace NUMINAMATH_CALUDE_quadratic_root_range_l3752_375204

theorem quadratic_root_range (m : ℝ) : 
  (∃ x y : ℝ, x < -1 ∧ y > 1 ∧ 
   x^2 + (m-1)*x + m^2 - 2 = 0 ∧
   y^2 + (m-1)*y + m^2 - 2 = 0) →
  0 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l3752_375204


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3752_375298

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (m + 1, -3)
  let b : ℝ × ℝ := (2, 3)
  parallel a b → m = -3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3752_375298


namespace NUMINAMATH_CALUDE_scientific_notation_of_20_8_billion_l3752_375257

/-- Expresses 20.8 billion in scientific notation -/
theorem scientific_notation_of_20_8_billion :
  20.8 * (10 : ℝ)^9 = 2.08 * (10 : ℝ)^9 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_20_8_billion_l3752_375257


namespace NUMINAMATH_CALUDE_closest_years_with_property_l3752_375250

def has_property (year : ℕ) : Prop :=
  let a := year / 1000
  let b := (year / 100) % 10
  let c := (year / 10) % 10
  let d := year % 10
  10 * a + b + 10 * c + d = 10 * b + c

theorem closest_years_with_property : 
  (∀ y : ℕ, 1868 < y ∧ y < 1978 → ¬(has_property y)) ∧ 
  (∀ y : ℕ, 1978 < y ∧ y < 2307 → ¬(has_property y)) ∧
  has_property 1868 ∧ 
  has_property 2307 :=
sorry

end NUMINAMATH_CALUDE_closest_years_with_property_l3752_375250


namespace NUMINAMATH_CALUDE_extremum_point_implies_a_eq_3_f_increasing_when_a_le_2_max_m_value_l3752_375271

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x^2 - a*x

theorem extremum_point_implies_a_eq_3 :
  ∀ a : ℝ, (∀ h : ℝ, h ≠ 0 → (f a (1 + h) - f a 1) / h = 0) → a = 3 :=
sorry

theorem f_increasing_when_a_le_2 :
  ∀ a : ℝ, 0 < a → a ≤ 2 → StrictMono (f a) :=
sorry

theorem max_m_value :
  ∃ m : ℝ, m = -(Real.log 2)⁻¹ ∧
  (∀ a x₀ : ℝ, 1 < a → a < 2 → 1 ≤ x₀ → x₀ ≤ 2 → f a x₀ > m * Real.log a) ∧
  (∀ m' : ℝ, m' > m → ∃ a x₀ : ℝ, 1 < a ∧ a < 2 ∧ 1 ≤ x₀ ∧ x₀ ≤ 2 ∧ f a x₀ ≤ m' * Real.log a) :=
sorry

end NUMINAMATH_CALUDE_extremum_point_implies_a_eq_3_f_increasing_when_a_le_2_max_m_value_l3752_375271


namespace NUMINAMATH_CALUDE_sum_of_odd_terms_l3752_375245

theorem sum_of_odd_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  (∀ n, S n = n^2 + n) → 
  a 1 + a 3 + a 5 + a 7 + a 9 = 50 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_odd_terms_l3752_375245


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3752_375284

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (x + 1)^2 * (x^2 - 7)^3 = a₀ + a₁*(x + 2) + a₂*(x + 2)^2 + a₃*(x + 2)^3 + 
                                       a₄*(x + 2)^4 + a₅*(x + 2)^5 + a₆*(x + 2)^6 + 
                                       a₇*(x + 2)^7 + a₈*(x + 2)^8) →
  a₁ - a₂ + a₃ - a₄ + a₅ - a₆ + a₇ = -58 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3752_375284


namespace NUMINAMATH_CALUDE_eggs_laid_per_dove_l3752_375218

/-- The number of eggs laid by each dove -/
def eggs_per_dove : ℕ := 3

/-- The initial number of female doves -/
def initial_doves : ℕ := 20

/-- The fraction of eggs that hatched -/
def hatch_rate : ℚ := 3/4

/-- The total number of doves after hatching -/
def total_doves : ℕ := 65

theorem eggs_laid_per_dove :
  eggs_per_dove * initial_doves * hatch_rate = total_doves - initial_doves :=
sorry

end NUMINAMATH_CALUDE_eggs_laid_per_dove_l3752_375218


namespace NUMINAMATH_CALUDE_problem_solution_l3752_375236

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x + 8

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * (a + 1) * x + 6 * a

theorem problem_solution (a : ℝ) :
  (f' a 3 = 0 → a = 3) ∧
  (∀ x < 0, Monotone (f a) ↔ a ≥ 0) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3752_375236


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3752_375203

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  first : ℕ
  last : ℕ
  diff : ℕ
  h_first : first = 17
  h_last : last = 95
  h_diff : diff = 4

/-- The number of terms in the sequence -/
def numTerms (seq : ArithmeticSequence) : ℕ :=
  (seq.last - seq.first) / seq.diff + 1

/-- The sum of all terms in the sequence -/
def sumTerms (seq : ArithmeticSequence) : ℕ :=
  (numTerms seq * (seq.first + seq.last)) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  numTerms seq = 20 ∧ sumTerms seq = 1100 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3752_375203


namespace NUMINAMATH_CALUDE_symmetric_line_fixed_point_l3752_375200

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the symmetry relation
def symmetric_about (l1 l2 : Line) (p : ℝ × ℝ) : Prop :=
  ∀ (x y : ℝ), (y = l1.slope * (x - 4)) → 
  ∃ (x' y' : ℝ), (y' = l2.slope * x' + l2.intercept) ∧ 
  ((x + x') / 2 = p.1) ∧ ((y + y') / 2 = p.2)

-- Define when a line passes through a point
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

-- Theorem statement
theorem symmetric_line_fixed_point (k : ℝ) :
  ∀ (l2 : Line),
  symmetric_about (Line.mk k (-4*k)) l2 (2, 1) →
  passes_through l2 (0, 2) := by sorry

end NUMINAMATH_CALUDE_symmetric_line_fixed_point_l3752_375200


namespace NUMINAMATH_CALUDE_symmetric_points_product_l3752_375256

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are opposites of each other -/
def symmetric_wrt_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

theorem symmetric_points_product (a b : ℝ) :
  symmetric_wrt_x_axis (2, a) (b + 1, 3) → a * b = -3 := by
  sorry

#check symmetric_points_product

end NUMINAMATH_CALUDE_symmetric_points_product_l3752_375256


namespace NUMINAMATH_CALUDE_jar_weight_percentage_l3752_375205

theorem jar_weight_percentage (jar_weight bean_weight : ℝ) 
  (h1 : jar_weight + 0.5 * bean_weight = 0.6 * (jar_weight + bean_weight)) : 
  jar_weight / (jar_weight + bean_weight) = 0.2 := by
sorry

end NUMINAMATH_CALUDE_jar_weight_percentage_l3752_375205


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_quadratic_roots_l3752_375258

theorem sum_of_reciprocals_of_quadratic_roots :
  ∀ (p q : ℝ), 
    p^2 - 11*p + 6 = 0 →
    q^2 - 11*q + 6 = 0 →
    p ≠ 0 →
    q ≠ 0 →
    1/p + 1/q = 11/6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_quadratic_roots_l3752_375258


namespace NUMINAMATH_CALUDE_dave_pays_more_than_doug_l3752_375262

/-- Represents the cost and composition of a pizza -/
structure Pizza where
  slices : Nat
  base_cost : Nat
  olive_slices : Nat
  olive_cost : Nat
  mushroom_slices : Nat
  mushroom_cost : Nat

/-- Calculates the total cost of the pizza -/
def total_cost (p : Pizza) : Nat :=
  p.base_cost + p.olive_cost + p.mushroom_cost

/-- Calculates the cost of a given number of slices -/
def slice_cost (p : Pizza) (n : Nat) (with_olive : Nat) (with_mushroom : Nat) : Nat :=
  let base := n * p.base_cost / p.slices
  let olive := with_olive * p.olive_cost / p.olive_slices
  let mushroom := with_mushroom * p.mushroom_cost / p.mushroom_slices
  base + olive + mushroom

/-- Theorem: Dave pays 10 dollars more than Doug -/
theorem dave_pays_more_than_doug (p : Pizza) 
    (h1 : p.slices = 12)
    (h2 : p.base_cost = 12)
    (h3 : p.olive_slices = 3)
    (h4 : p.olive_cost = 3)
    (h5 : p.mushroom_slices = 6)
    (h6 : p.mushroom_cost = 4) :
  slice_cost p 8 2 6 - slice_cost p 4 0 0 = 10 := by
  sorry


end NUMINAMATH_CALUDE_dave_pays_more_than_doug_l3752_375262


namespace NUMINAMATH_CALUDE_max_value_theorem_l3752_375208

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (2 * x + y)) + (y / (x + 2 * y)) ≤ 2/3 := by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3752_375208


namespace NUMINAMATH_CALUDE_ophelias_current_age_l3752_375252

/-- Represents a person's age -/
structure Person where
  age : ℕ

/-- Given Lennon's current age and the relationship between Ophelia and Lennon's ages in two years,
    prove that Ophelia's current age is 38 years. -/
theorem ophelias_current_age 
  (lennon ophelia : Person)
  (lennon_current_age : lennon.age = 8)
  (future_age_relation : ophelia.age + 2 = 4 * (lennon.age + 2)) :
  ophelia.age = 38 := by
  sorry

end NUMINAMATH_CALUDE_ophelias_current_age_l3752_375252


namespace NUMINAMATH_CALUDE_negation_relationship_l3752_375288

theorem negation_relationship (x : ℝ) :
  (¬(|x + 1| > 2) → ¬(5*x - 6 > x^2)) ∧
  ¬(¬(5*x - 6 > x^2) → ¬(|x + 1| > 2)) :=
by sorry

end NUMINAMATH_CALUDE_negation_relationship_l3752_375288


namespace NUMINAMATH_CALUDE_max_base_eight_digit_sum_l3752_375276

def base_eight_digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
  aux n []

theorem max_base_eight_digit_sum (n : ℕ) (h : n < 1728) :
  (base_eight_digits n).sum ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_base_eight_digit_sum_l3752_375276


namespace NUMINAMATH_CALUDE_anca_rest_time_l3752_375206

-- Define the constants
def bruce_speed : ℝ := 50
def anca_speed : ℝ := 60
def total_distance : ℝ := 200

-- Define the theorem
theorem anca_rest_time :
  let bruce_time := total_distance / bruce_speed
  let anca_drive_time := total_distance / anca_speed
  let rest_time := bruce_time - anca_drive_time
  rest_time * 60 = 40 := by
  sorry

end NUMINAMATH_CALUDE_anca_rest_time_l3752_375206


namespace NUMINAMATH_CALUDE_a_must_be_positive_l3752_375255

theorem a_must_be_positive
  (a b c d : ℝ)
  (h1 : b ≠ 0)
  (h2 : d ≠ 0)
  (h3 : d > 0)
  (h4 : a / b > -(3 / (2 * d))) :
  a > 0 :=
by sorry

end NUMINAMATH_CALUDE_a_must_be_positive_l3752_375255


namespace NUMINAMATH_CALUDE_solve_for_y_l3752_375269

theorem solve_for_y (x y : ℤ) (h1 : x^2 - x + 6 = y + 2) (h2 : x = -5) : y = 34 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3752_375269


namespace NUMINAMATH_CALUDE_perpendicular_lines_k_value_l3752_375235

/-- A line in 2D space represented by parametric equations --/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Check if two lines are perpendicular --/
def perpendicular (l1 l2 : ParametricLine) : Prop :=
  ∃ m1 m2 : ℝ, (∀ t : ℝ, l1.y t = m1 * l1.x t + (l1.y 0 - m1 * l1.x 0)) ∧
              (∀ s : ℝ, l2.y s = m2 * l2.x s + (l2.y 0 - m2 * l2.x 0)) ∧
              m1 * m2 = -1

theorem perpendicular_lines_k_value :
  ∀ k : ℝ,
  let l1 : ParametricLine := {
    x := λ t => 1 - 2*t,
    y := λ t => 2 + k*t
  }
  let l2 : ParametricLine := {
    x := λ s => s,
    y := λ s => 1 - 2*s
  }
  perpendicular l1 l2 → k = -1 := by
  sorry

#check perpendicular_lines_k_value

end NUMINAMATH_CALUDE_perpendicular_lines_k_value_l3752_375235


namespace NUMINAMATH_CALUDE_chess_team_selection_l3752_375243

theorem chess_team_selection (total_players : Nat) (quadruplets : Nat) (team_size : Nat) :
  total_players = 18 →
  quadruplets = 4 →
  team_size = 8 →
  Nat.choose (total_players - quadruplets) (team_size - quadruplets) = 1001 :=
by sorry

end NUMINAMATH_CALUDE_chess_team_selection_l3752_375243


namespace NUMINAMATH_CALUDE_egyptian_fraction_for_odd_n_l3752_375240

theorem egyptian_fraction_for_odd_n (n : ℕ) 
  (h_odd : Odd n) 
  (h_gt3 : n > 3) 
  (h_not_div3 : ¬(3 ∣ n)) : 
  ∃ (a b c : ℕ), 
    Odd a ∧ Odd b ∧ Odd c ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (3 : ℚ) / n = 1 / a + 1 / b + 1 / c := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_for_odd_n_l3752_375240


namespace NUMINAMATH_CALUDE_green_ball_probability_l3752_375239

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- Calculates the probability of selecting a green ball from a container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- Theorem: The probability of selecting a green ball is 26/45 -/
theorem green_ball_probability :
  let containerA : Container := ⟨5, 7⟩
  let containerB : Container := ⟨4, 5⟩
  let containerC : Container := ⟨7, 3⟩
  let totalContainers : ℕ := 3
  let probA : ℚ := 1 / totalContainers * greenProbability containerA
  let probB : ℚ := 1 / totalContainers * greenProbability containerB
  let probC : ℚ := 1 / totalContainers * greenProbability containerC
  probA + probB + probC = 26 / 45 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l3752_375239


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3752_375223

/-- 
Proves that in an arithmetic sequence with given conditions, 
the common difference is 3.
-/
theorem arithmetic_sequence_common_difference 
  (a : ℝ) (an : ℝ) (n : ℕ) (sum : ℝ) :
  a = 2 →
  an = 50 →
  sum = 442 →
  an = a + (n - 1) * (3 : ℝ) →
  sum = (n / 2) * (a + an) →
  (3 : ℝ) = (an - a) / (n - 1) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3752_375223


namespace NUMINAMATH_CALUDE_distribute_four_students_three_universities_l3752_375292

/-- The number of ways to distribute n distinct students among k distinct universities,
    with each university receiving at least one student. -/
def distribute_students (n k : ℕ) : ℕ :=
  sorry

/-- The theorem stating that distributing 4 students among 3 universities results in 36 ways. -/
theorem distribute_four_students_three_universities :
  distribute_students 4 3 = 36 :=
sorry

end NUMINAMATH_CALUDE_distribute_four_students_three_universities_l3752_375292


namespace NUMINAMATH_CALUDE_trivia_contest_probability_l3752_375277

def num_questions : ℕ := 4
def num_choices : ℕ := 4
def min_correct : ℕ := 3

def probability_correct_guess : ℚ := 1 / num_choices

def probability_winning : ℚ :=
  (num_questions.choose min_correct) * (probability_correct_guess ^ min_correct) * ((1 - probability_correct_guess) ^ (num_questions - min_correct)) +
  (num_questions.choose (min_correct + 1)) * (probability_correct_guess ^ (min_correct + 1)) * ((1 - probability_correct_guess) ^ (num_questions - (min_correct + 1)))

theorem trivia_contest_probability : probability_winning = 13 / 256 := by
  sorry

end NUMINAMATH_CALUDE_trivia_contest_probability_l3752_375277


namespace NUMINAMATH_CALUDE_function_value_at_pi_sixth_l3752_375214

/-- Given a function f(x) = 3sin(ωx + φ) that satisfies f(π/3 + x) = f(-x) for any x,
    prove that f(π/6) = -3 or f(π/6) = 3 -/
theorem function_value_at_pi_sixth (ω φ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin (ω * x + φ)
  (∀ x, f (π / 3 + x) = f (-x)) →
  f (π / 6) = -3 ∨ f (π / 6) = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_pi_sixth_l3752_375214


namespace NUMINAMATH_CALUDE_find_b_value_l3752_375290

theorem find_b_value (a b c : ℤ) : 
  a + b + c = 111 → 
  (a + 10 = b - 10) ∧ (b - 10 = 3 * c) → 
  b = 58 :=
by sorry

end NUMINAMATH_CALUDE_find_b_value_l3752_375290


namespace NUMINAMATH_CALUDE_dave_book_cost_l3752_375254

/-- The cost per book given the total number of books and total amount spent -/
def cost_per_book (total_books : ℕ) (total_spent : ℚ) : ℚ :=
  total_spent / total_books

theorem dave_book_cost :
  let total_books : ℕ := 8 + 6 + 3
  let total_spent : ℚ := 102
  cost_per_book total_books total_spent = 6 := by
  sorry

end NUMINAMATH_CALUDE_dave_book_cost_l3752_375254


namespace NUMINAMATH_CALUDE_unique_digit_subtraction_l3752_375207

theorem unique_digit_subtraction :
  ∃! (I K S : ℕ),
    I < 10 ∧ K < 10 ∧ S < 10 ∧
    100 * K + 10 * I + S ≥ 100 ∧
    100 * S + 10 * I + K ≥ 100 ∧
    100 * S + 10 * K + I ≥ 100 ∧
    (100 * K + 10 * I + S) - (100 * S + 10 * I + K) = 100 * S + 10 * K + I :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_subtraction_l3752_375207


namespace NUMINAMATH_CALUDE_work_efficiency_l3752_375274

theorem work_efficiency (sakshi_days tanya_days : ℝ) : 
  tanya_days = 16 →
  sakshi_days / 1.25 = tanya_days →
  sakshi_days = 20 := by
sorry

end NUMINAMATH_CALUDE_work_efficiency_l3752_375274


namespace NUMINAMATH_CALUDE_prob_at_least_twice_avg_profit_two_visits_prob_select_one_twice_l3752_375293

/-- Represents the charge ratio for each visit -/
def charge_ratio : ℕ → ℝ
  | 1 => 1
  | 2 => 0.95
  | 3 => 0.90
  | 4 => 0.85
  | _ => 0.80

/-- Represents the frequency of visits -/
def visit_frequency : ℕ → ℕ
  | 1 => 60
  | 2 => 20
  | 3 => 10
  | 4 => 5
  | _ => 5

/-- The initial charge for the first visit -/
def initial_charge : ℝ := 200

/-- The cost per service -/
def service_cost : ℝ := 150

/-- Total number of members surveyed -/
def total_members : ℕ := 100

/-- Number of members who consumed at least twice -/
def members_at_least_twice : ℕ := visit_frequency 2 + visit_frequency 3 + visit_frequency 4 + visit_frequency 5

/-- Theorem: The probability of a member consuming at least twice is 0.4 -/
theorem prob_at_least_twice : (members_at_least_twice : ℝ) / total_members = 0.4 := by sorry

/-- Profit from the first visit -/
def profit_first_visit : ℝ := initial_charge - service_cost

/-- Profit from the second visit -/
def profit_second_visit : ℝ := initial_charge * charge_ratio 2 - service_cost

/-- Theorem: The average profit from two visits is $45 -/
theorem avg_profit_two_visits : (profit_first_visit + profit_second_visit) / 2 = 45 := by sorry

/-- Number of ways to select 2 customers from 6 -/
def total_selections : ℕ := 15

/-- Number of ways to select exactly 1 customer who visited twice -/
def favorable_selections : ℕ := 8

/-- Theorem: The probability of selecting exactly 1 customer who visited twice out of 2 selected customers is 8/15 -/
theorem prob_select_one_twice : (favorable_selections : ℝ) / total_selections = 8 / 15 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_twice_avg_profit_two_visits_prob_select_one_twice_l3752_375293


namespace NUMINAMATH_CALUDE_negative_eight_meters_westward_l3752_375229

-- Define the direction type
inductive Direction
| East
| West

-- Define a function to convert meters to a direction and magnitude
def metersToDirection (x : ℤ) : Direction × ℕ :=
  if x ≥ 0 then
    (Direction.East, x.natAbs)
  else
    (Direction.West, (-x).natAbs)

-- State the theorem
theorem negative_eight_meters_westward :
  metersToDirection (-8) = (Direction.West, 8) :=
sorry

end NUMINAMATH_CALUDE_negative_eight_meters_westward_l3752_375229


namespace NUMINAMATH_CALUDE_range_of_g_l3752_375295

noncomputable def g (x : ℝ) : ℝ := (Real.arccos x) * (Real.arcsin x)

theorem range_of_g :
  ∀ x ∈ Set.Icc (-1 : ℝ) 1,
    Real.arccos x + Real.arcsin x = π / 2 →
    ∃ y ∈ Set.Icc 0 (π^2 / 8), g y = g x ∧
    ∀ z ∈ Set.Icc (-1 : ℝ) 1, g z ≤ π^2 / 8 ∧ g z ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l3752_375295


namespace NUMINAMATH_CALUDE_a_range_l3752_375224

/-- A quadratic function f(x) = x² + 2(a-1)x + 5 that is increasing on (4, +∞) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 5

/-- The property that f is increasing on (4, +∞) -/
def f_increasing (a : ℝ) : Prop :=
  ∀ x y, x > 4 → y > x → f a y > f a x

/-- The theorem stating the range of a -/
theorem a_range (a : ℝ) (h : f_increasing a) : a ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_a_range_l3752_375224


namespace NUMINAMATH_CALUDE_square_root_squared_specific_case_l3752_375222

theorem square_root_squared (n : ℝ) (h : 0 ≤ n) : (Real.sqrt n)^2 = n := by sorry

theorem specific_case : (Real.sqrt 987654)^2 = 987654 := by sorry

end NUMINAMATH_CALUDE_square_root_squared_specific_case_l3752_375222


namespace NUMINAMATH_CALUDE_edward_baseball_cards_l3752_375221

/-- The number of binders Edward has -/
def num_binders : ℕ := 7

/-- The number of cards in each binder -/
def cards_per_binder : ℕ := 109

/-- The total number of baseball cards Edward has -/
def total_cards : ℕ := num_binders * cards_per_binder

theorem edward_baseball_cards : total_cards = 763 := by
  sorry

end NUMINAMATH_CALUDE_edward_baseball_cards_l3752_375221


namespace NUMINAMATH_CALUDE_apple_percentage_after_adding_oranges_l3752_375228

/-- Given a basket with apples and oranges, calculate the percentage of apples after adding more oranges. -/
theorem apple_percentage_after_adding_oranges 
  (initial_apples initial_oranges added_oranges : ℕ) : 
  initial_apples = 10 → 
  initial_oranges = 5 → 
  added_oranges = 5 → 
  (initial_apples : ℚ) / (initial_apples + initial_oranges + added_oranges) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_apple_percentage_after_adding_oranges_l3752_375228


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3752_375244

theorem imaginary_part_of_z (z : ℂ) (h : z + (3 - 4*I) = 1) : z.im = 4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3752_375244


namespace NUMINAMATH_CALUDE_rationalization_factor_l3752_375249

theorem rationalization_factor (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  (Real.sqrt a + Real.sqrt b) * (Real.sqrt a - Real.sqrt b) = a - b ∧
  (Real.sqrt a + Real.sqrt b) * (Real.sqrt b - Real.sqrt a) = b - a :=
by sorry

end NUMINAMATH_CALUDE_rationalization_factor_l3752_375249


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l3752_375261

-- Define the set of cards
inductive Card : Type
| Red : Card
| Yellow : Card
| Blue : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define a distribution of cards
def Distribution := Person → Card

-- Define the events
def EventA (d : Distribution) : Prop := d Person.A = Card.Red
def EventB (d : Distribution) : Prop := d Person.B = Card.Red

-- State the theorem
theorem events_mutually_exclusive_not_contradictory :
  -- The events are mutually exclusive
  (∀ d : Distribution, ¬(EventA d ∧ EventB d)) ∧
  -- The events are not contradictory
  (∃ d : Distribution, EventA d) ∧
  (∃ d : Distribution, EventB d) ∧
  -- There exists a distribution where neither event occurs
  (∃ d : Distribution, ¬EventA d ∧ ¬EventB d) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l3752_375261


namespace NUMINAMATH_CALUDE_min_value_of_a_min_value_is_one_l3752_375285

theorem min_value_of_a (a : ℝ) : 
  (∀ x : ℝ, (2 * x) / (x^2 + 1) ≤ a) → a ≥ 1 := by
  sorry

theorem min_value_is_one :
  ∃ a : ℝ, (∀ x : ℝ, (2 * x) / (x^2 + 1) ≤ a) ∧ 
    (∀ b : ℝ, (∀ x : ℝ, (2 * x) / (x^2 + 1) ≤ b) → a ≤ b) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_min_value_is_one_l3752_375285


namespace NUMINAMATH_CALUDE_inequality_proof_l3752_375281

theorem inequality_proof (a b : ℝ) (n : ℕ+) 
  (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  (a + b)^(n : ℝ) - a^(n : ℝ) - b^(n : ℝ) ≥ 2^(2*(n : ℝ)) - 2^((n : ℝ) + 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3752_375281


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l3752_375296

theorem min_value_sum_squares (x y z : ℝ) :
  x - 1 = 2 * (y + 1) ∧ x - 1 = 3 * (z + 2) →
  ∀ a b c : ℝ, a - 1 = 2 * (b + 1) ∧ a - 1 = 3 * (c + 2) →
  x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 ∧
  ∃ x₀ y₀ z₀ : ℝ, x₀ - 1 = 2 * (y₀ + 1) ∧ x₀ - 1 = 3 * (z₀ + 2) ∧
                  x₀^2 + y₀^2 + z₀^2 = 293 / 49 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l3752_375296


namespace NUMINAMATH_CALUDE_cathy_commission_l3752_375291

theorem cathy_commission (x : ℝ) : 
  0.15 * (x - 15) = 0.25 * (x - 25) → 
  0.1 * (x - 10) = 3 := by
sorry

end NUMINAMATH_CALUDE_cathy_commission_l3752_375291


namespace NUMINAMATH_CALUDE_real_part_of_z_l3752_375219

theorem real_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Complex.abs (1 - Complex.I * Real.sqrt 3) + Complex.I) :
  z.re = 1/2 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3752_375219


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3752_375226

theorem sum_of_fractions (x y z : ℕ) : 
  (Nat.gcd x 9 = 1) → 
  (Nat.gcd y 15 = 1) → 
  (Nat.gcd z 14 = 1) → 
  (x * y * z : ℚ) / (9 * 15 * 14) = 1 / 6 → 
  x + y + z = 21 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3752_375226


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l3752_375279

/-- An arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 2) = a (n + 1) * r

theorem arithmetic_geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_ag : ArithmeticGeometricSequence a)
  (h_eq : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25) :
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l3752_375279


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l3752_375280

theorem quadratic_roots_properties :
  let a : ℝ := 1
  let b : ℝ := 4
  let c : ℝ := -42
  let product_of_roots := c / a
  let sum_of_roots := -b / a
  product_of_roots = -42 ∧ sum_of_roots = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l3752_375280


namespace NUMINAMATH_CALUDE_inequality_proof_l3752_375232

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a|

-- State the theorem
theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_solution_set : Set.Icc 0 2 = {x | f x ((1/m) + (1/(2*n))) ≤ 1}) : 
  m + 4*n ≥ 2*Real.sqrt 2 + 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3752_375232


namespace NUMINAMATH_CALUDE_odd_functions_max_min_l3752_375213

-- Define the property of being an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the function F
def F (f g : ℝ → ℝ) (x : ℝ) : ℝ := f x + g x + 2

-- State the theorem
theorem odd_functions_max_min (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsOdd g) 
  (hmax : ∃ M, M = 8 ∧ ∀ x > 0, F f g x ≤ M) :
  ∃ m, m = -4 ∧ ∀ x < 0, F f g x ≥ m :=
sorry

end NUMINAMATH_CALUDE_odd_functions_max_min_l3752_375213


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3752_375225

def U : Set ℕ := {2, 3, 5, 7, 8}
def A : Set ℕ := {2, 8}
def B : Set ℕ := {3, 5, 8}

theorem complement_intersection_theorem : (U \ A) ∩ B = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3752_375225


namespace NUMINAMATH_CALUDE_smallest_w_l3752_375278

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : 
  w > 0 → 
  is_factor (2^5) (936 * w) → 
  is_factor (3^3) (936 * w) → 
  is_factor (13^2) (936 * w) → 
  ∀ v : ℕ, v > 0 → 
    is_factor (2^5) (936 * v) → 
    is_factor (3^3) (936 * v) → 
    is_factor (13^2) (936 * v) → 
    w ≤ v → 
  w = 156 := by sorry

end NUMINAMATH_CALUDE_smallest_w_l3752_375278


namespace NUMINAMATH_CALUDE_quadratic_roots_l3752_375263

theorem quadratic_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 - 4*y + m = 0 ∧ y = 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3752_375263


namespace NUMINAMATH_CALUDE_ordered_pairs_satisfying_equations_l3752_375233

theorem ordered_pairs_satisfying_equations :
  ∀ (x y : ℝ), x^2 * y = 3 ∧ x + x*y = 4 ↔ (x = 1 ∧ y = 3) ∨ (x = 3 ∧ y = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_ordered_pairs_satisfying_equations_l3752_375233


namespace NUMINAMATH_CALUDE_factor_3x_squared_minus_75_l3752_375275

theorem factor_3x_squared_minus_75 (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_3x_squared_minus_75_l3752_375275


namespace NUMINAMATH_CALUDE_notebook_distribution_l3752_375217

theorem notebook_distribution (total_notebooks : ℕ) (initial_students : ℕ) : 
  total_notebooks = 512 →
  total_notebooks = initial_students * (initial_students / 8) →
  (total_notebooks / (initial_students / 2) : ℕ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_notebook_distribution_l3752_375217


namespace NUMINAMATH_CALUDE_quadratic_sum_l3752_375273

/-- A quadratic function g(x) = ax^2 + bx + c satisfying g(1) = 2 and g(2) = 3 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- Theorem: For a quadratic function g(x) = ax^2 + bx + c, if g(1) = 2 and g(2) = 3, then a + 2b + 3c = 7 -/
theorem quadratic_sum (a b c : ℝ) :
  (QuadraticFunction a b c 1 = 2) →
  (QuadraticFunction a b c 2 = 3) →
  a + 2 * b + 3 * c = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3752_375273


namespace NUMINAMATH_CALUDE_sqrt_1600_minus_24_form_l3752_375238

theorem sqrt_1600_minus_24_form (a b : ℕ+) :
  (Real.sqrt 1600 - 24 : ℝ) = ((Real.sqrt a.val - b.val) : ℝ)^2 →
  a.val + b.val = 102 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_1600_minus_24_form_l3752_375238


namespace NUMINAMATH_CALUDE_sum_of_digits_of_difference_of_squares_l3752_375264

def a : ℕ := 6666666
def b : ℕ := 3333333

-- Function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_difference_of_squares :
  sum_of_digits ((a ^ 2) - (b ^ 2)) = 63 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_difference_of_squares_l3752_375264


namespace NUMINAMATH_CALUDE_complex_equation_result_l3752_375241

theorem complex_equation_result (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_result_l3752_375241


namespace NUMINAMATH_CALUDE_evaluate_expression_l3752_375212

theorem evaluate_expression (x y : ℤ) (hx : x = 5) (hy : y = -3) :
  y * (y - 2 * x + 1) = 36 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3752_375212


namespace NUMINAMATH_CALUDE_inequality_property_l3752_375268

theorem inequality_property (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a * b > b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_property_l3752_375268


namespace NUMINAMATH_CALUDE_circle_properties_l3752_375289

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 4

-- Define the center of the circle
def center : ℝ × ℝ := (1, -2)

-- Define the radius of the circle
def radius : ℝ := 2

-- Theorem statement
theorem circle_properties :
  ∀ x y : ℝ, circle_equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3752_375289


namespace NUMINAMATH_CALUDE_line_symmetry_l3752_375270

-- Define the lines
def l (x y : ℝ) : Prop := x - y - 1 = 0
def l₁ (x y : ℝ) : Prop := 2*x - y - 2 = 0
def l₂ (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define symmetry with respect to a line
def symmetric_wrt (f g h : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y → ∃ x' y', g x' y' ∧ h x y ∧
    ((x + x') / 2, (y + y') / 2) ∈ {(a, b) | f a b}

-- Theorem statement
theorem line_symmetry :
  symmetric_wrt l l₁ l₂ :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l3752_375270


namespace NUMINAMATH_CALUDE_customers_per_table_l3752_375201

theorem customers_per_table 
  (initial_customers : ℕ) 
  (left_customers : ℕ) 
  (num_tables : ℕ) 
  (h1 : initial_customers = 21)
  (h2 : left_customers = 12)
  (h3 : num_tables = 3)
  (h4 : num_tables > 0)
  : (initial_customers - left_customers) / num_tables = 3 := by
sorry

end NUMINAMATH_CALUDE_customers_per_table_l3752_375201


namespace NUMINAMATH_CALUDE_group_size_calculation_l3752_375230

theorem group_size_calculation (T : ℕ) (L : ℕ) : 
  T = L + 90 → -- Total is sum of young and old
  (L : ℚ) / T = 1/4 → -- Probability of selecting young person
  T = 120 := by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l3752_375230


namespace NUMINAMATH_CALUDE_reasoning_is_deductive_l3752_375210

-- Define the universe of discourse
variable (Person : Type)

-- Define the property of making mistakes
variable (makesMistakes : Person → Prop)

-- Define Mr. Wang as a person
variable (mrWang : Person)

-- State the premises
variable (everyone_makes_mistakes : ∀ (p : Person), makesMistakes p)
variable (mr_wang_is_person : Person)

-- Define deductive reasoning
def isDeductiveReasoning (premise1 premise2 conclusion : Prop) : Prop :=
  (premise1 ∧ premise2) → conclusion

-- Theorem statement
theorem reasoning_is_deductive :
  isDeductiveReasoning
    (∀ (p : Person), makesMistakes p)
    (mrWang = mr_wang_is_person)
    (makesMistakes mrWang) :=
by sorry

end NUMINAMATH_CALUDE_reasoning_is_deductive_l3752_375210


namespace NUMINAMATH_CALUDE_study_time_problem_l3752_375237

/-- The study time problem -/
theorem study_time_problem 
  (kwame_time : ℝ) 
  (lexia_time : ℝ) 
  (h1 : kwame_time = 2.5)
  (h2 : lexia_time = 97 / 60)
  (h3 : kwame_time + connor_time = lexia_time + 143 / 60) :
  connor_time = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_study_time_problem_l3752_375237


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3752_375294

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 2 < x ∧ x < 4}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3752_375294


namespace NUMINAMATH_CALUDE_f_comp_f_three_roots_l3752_375209

/-- The function f(x) = x^2 - 4x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + c

/-- The composition of f with itself -/
def f_comp_f (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- Theorem stating that f(f(x)) has exactly three distinct real roots when c = 8 -/
theorem f_comp_f_three_roots :
  ∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    (∀ x : ℝ, f_comp_f 8 x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) :=
sorry

end NUMINAMATH_CALUDE_f_comp_f_three_roots_l3752_375209


namespace NUMINAMATH_CALUDE_sum_of_real_and_imag_parts_l3752_375287

theorem sum_of_real_and_imag_parts (z : ℂ) (h : z * (2 + Complex.I) = 2 * Complex.I - 1) :
  z.re + z.im = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_real_and_imag_parts_l3752_375287


namespace NUMINAMATH_CALUDE_intersection_implies_m_equals_one_l3752_375216

def A (m : ℝ) : Set ℝ := {0, m}
def B : Set ℝ := {1, 2}

theorem intersection_implies_m_equals_one (m : ℝ) :
  A m ∩ B = {1} → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_equals_one_l3752_375216


namespace NUMINAMATH_CALUDE_overtime_rate_is_correct_l3752_375283

/-- Calculates the overtime rate given the following conditions:
  * 6 working days in a regular week
  * 10 working hours per day
  * Rs. 2.10 per hour for regular work
  * Total earnings: Rs. 525 in 4 weeks
  * Total hours worked: 245 hours
-/
def calculate_overtime_rate (
  working_days_per_week : ℕ)
  (working_hours_per_day : ℕ)
  (regular_rate : ℚ)
  (total_earnings : ℚ)
  (total_hours : ℕ)
  (num_weeks : ℕ) : ℚ :=
  let regular_hours := working_days_per_week * working_hours_per_day * num_weeks
  let overtime_hours := total_hours - regular_hours
  let regular_earnings := regular_hours * regular_rate
  let overtime_earnings := total_earnings - regular_earnings
  overtime_earnings / overtime_hours

/-- Theorem stating that the overtime rate is 4.20 given the problem conditions -/
theorem overtime_rate_is_correct :
  calculate_overtime_rate 6 10 (21/10) 525 245 4 = 21/5 := by
  sorry

end NUMINAMATH_CALUDE_overtime_rate_is_correct_l3752_375283


namespace NUMINAMATH_CALUDE_triangle_third_side_l3752_375251

theorem triangle_third_side (a b h : ℝ) (ha : a = 25) (hb : b = 30) (hh : h = 24) :
  ∃ c, (c = 25 ∨ c = 11) ∧ 
  (∃ s, s * h = a * b ∧ 
   ((c + s) * (c - s) = a^2 - b^2 ∨ (c + s) * (c - s) = b^2 - a^2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_l3752_375251


namespace NUMINAMATH_CALUDE_number_of_boys_l3752_375211

theorem number_of_boys (total_amount : ℕ) (total_children : ℕ) (boy_amount : ℕ) (girl_amount : ℕ) 
  (h1 : total_amount = 460)
  (h2 : total_children = 41)
  (h3 : boy_amount = 12)
  (h4 : girl_amount = 8) :
  ∃ (boys : ℕ), boys = 33 ∧ 
    boys * boy_amount + (total_children - boys) * girl_amount = total_amount :=
by sorry

end NUMINAMATH_CALUDE_number_of_boys_l3752_375211


namespace NUMINAMATH_CALUDE_drivers_distance_difference_l3752_375242

/-- Proves that the difference in distance traveled by two drivers meeting on a highway is 140 km -/
theorem drivers_distance_difference (initial_distance : ℝ) (speed_a : ℝ) (speed_b : ℝ) (delay : ℝ) : 
  initial_distance = 940 →
  speed_a = 90 →
  speed_b = 80 →
  delay = 1 →
  let remaining_distance := initial_distance - speed_a * delay
  let meeting_time := remaining_distance / (speed_a + speed_b)
  let distance_a := speed_a * (meeting_time + delay)
  let distance_b := speed_b * meeting_time
  distance_a - distance_b = 140 := by
  sorry

end NUMINAMATH_CALUDE_drivers_distance_difference_l3752_375242


namespace NUMINAMATH_CALUDE_no_negative_roots_l3752_375253

theorem no_negative_roots :
  ∀ x : ℝ, x < 0 → x^4 - 4*x^3 - 6*x^2 - 3*x + 9 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_negative_roots_l3752_375253


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l3752_375282

theorem complex_modulus_equality (x : ℝ) :
  x > 0 → (Complex.abs (5 + x * Complex.I) = 13 ↔ x = 12) := by sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l3752_375282


namespace NUMINAMATH_CALUDE_missing_number_proof_l3752_375220

theorem missing_number_proof (numbers : List ℕ) (h_count : numbers.length = 9) 
  (h_sum : numbers.sum = 744 + 745 + 747 + 749 + 752 + 752 + 753 + 755 + 755) 
  (h_avg : (numbers.sum + missing) / 10 = 750) : missing = 1748 := by
  sorry

#check missing_number_proof

end NUMINAMATH_CALUDE_missing_number_proof_l3752_375220


namespace NUMINAMATH_CALUDE_total_rewards_distributed_l3752_375266

theorem total_rewards_distributed (students_A students_B students_C : ℕ)
  (rewards_per_student_A rewards_per_student_B rewards_per_student_C : ℕ) :
  students_A = students_B + 4 →
  students_B = students_C + 4 →
  rewards_per_student_A + 3 = rewards_per_student_B →
  rewards_per_student_B + 5 = rewards_per_student_C →
  students_A * rewards_per_student_A = students_B * rewards_per_student_B + 3 →
  students_B * rewards_per_student_B = students_C * rewards_per_student_C + 5 →
  students_A * rewards_per_student_A +
  students_B * rewards_per_student_B +
  students_C * rewards_per_student_C = 673 :=
by sorry

end NUMINAMATH_CALUDE_total_rewards_distributed_l3752_375266


namespace NUMINAMATH_CALUDE_same_terminal_side_l3752_375202

/-- Proves that 375° has the same terminal side as α = π/12 + 2kπ, where k is an integer -/
theorem same_terminal_side (k : ℤ) : ∃ (n : ℤ), 375 * π / 180 = π / 12 + 2 * k * π + 2 * n * π := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_l3752_375202


namespace NUMINAMATH_CALUDE_class_assignment_arrangements_l3752_375231

theorem class_assignment_arrangements :
  let num_teachers : ℕ := 3
  let num_classes : ℕ := 6
  let classes_per_teacher : ℕ := 2
  let total_arrangements : ℕ := (Nat.choose num_classes classes_per_teacher) *
                                (Nat.choose (num_classes - classes_per_teacher) classes_per_teacher) *
                                (Nat.choose (num_classes - 2 * classes_per_teacher) classes_per_teacher) /
                                (Nat.factorial num_teachers)
  total_arrangements = 90 := by
  sorry

end NUMINAMATH_CALUDE_class_assignment_arrangements_l3752_375231


namespace NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_6_5_l3752_375297

theorem greatest_three_digit_divisible_by_3_6_5 : 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 3 ∣ n ∧ 6 ∣ n ∧ 5 ∣ n → n ≤ 990 := by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_6_5_l3752_375297


namespace NUMINAMATH_CALUDE_coprime_polynomials_l3752_375248

theorem coprime_polynomials (n : ℕ) : 
  Nat.gcd (n^5 + 4*n^3 + 3*n) (n^4 + 3*n^2 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_coprime_polynomials_l3752_375248
