import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1071_107165

theorem sum_of_three_numbers : ∀ (n₁ n₂ n₃ : ℕ),
  n₂ = 72 →
  n₁ = 2 * n₂ →
  n₃ = n₁ / 3 →
  n₁ + n₂ + n₃ = 264 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1071_107165


namespace NUMINAMATH_CALUDE_f_derivative_positive_implies_a_bound_l1071_107129

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * exp x - 2 * x^2

theorem f_derivative_positive_implies_a_bound (a : ℝ) :
  (∀ x₀ ∈ Set.Ioo 0 1, ∀ ε > 0, ∃ δ > 0, ∀ x ∈ Set.Ioo (x₀ - δ) (x₀ + δ),
    x ≠ x₀ → (f a x - f a x₀ - x + x₀) / (x - x₀) > 0) →
  a > 4 / exp (3/4) :=
sorry

end NUMINAMATH_CALUDE_f_derivative_positive_implies_a_bound_l1071_107129


namespace NUMINAMATH_CALUDE_patel_family_concert_cost_l1071_107157

theorem patel_family_concert_cost : 
  let regular_ticket_price : ℚ := 7.50 / (1 - 0.20)
  let children_ticket_price : ℚ := regular_ticket_price * (1 - 0.60)
  let senior_ticket_price : ℚ := 7.50
  let num_tickets_per_generation : ℕ := 2
  let handling_fee : ℚ := 5

  (num_tickets_per_generation * senior_ticket_price + 
   num_tickets_per_generation * regular_ticket_price + 
   num_tickets_per_generation * children_ticket_price + 
   handling_fee) = 46.25 := by
sorry


end NUMINAMATH_CALUDE_patel_family_concert_cost_l1071_107157


namespace NUMINAMATH_CALUDE_binomial_distribution_problem_l1071_107144

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The random variable X following a binomial distribution -/
def X (b : BinomialDistribution) : ℝ := sorry

/-- Expectation of a random variable -/
def expectation (X : ℝ) : ℝ := sorry

/-- Variance of a random variable -/
def variance (X : ℝ) : ℝ := sorry

theorem binomial_distribution_problem (b : BinomialDistribution) 
  (h2 : expectation (3 * X b - 9) = 27)
  (h3 : variance (3 * X b - 9) = 27) :
  b.n = 16 ∧ b.p = 3/4 := by sorry

end NUMINAMATH_CALUDE_binomial_distribution_problem_l1071_107144


namespace NUMINAMATH_CALUDE_simplify_fourth_root_l1071_107114

theorem simplify_fourth_root (a b : ℕ+) : 
  (2^6 * 5^5 : ℝ)^(1/4) = a * (b : ℝ)^(1/4) ∧ a + b = 30 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fourth_root_l1071_107114


namespace NUMINAMATH_CALUDE_a_b_product_l1071_107173

def a : ℕ → ℚ
  | 0 => 1/2
  | n + 1 => 2 * a n / (1 + (a n)^2)

def b : ℕ → ℚ
  | 0 => 4
  | n + 1 => b n ^ 2 - 2 * b n + 2

def b_product : ℕ → ℚ
  | 0 => b 0
  | n + 1 => b_product n * b (n + 1)

theorem a_b_product (n : ℕ) : a (n + 1) * b (n + 1) = 2 * b_product n := by
  sorry

end NUMINAMATH_CALUDE_a_b_product_l1071_107173


namespace NUMINAMATH_CALUDE_paperboy_delivery_ways_l1071_107108

/-- Represents the number of ways to deliver newspapers to n houses without missing four consecutive houses. -/
def E : ℕ → ℕ
  | 0 => 0  -- Define E(0) as 0 for completeness
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 4 => 15
  | n + 5 => E (n + 4) + E (n + 3) + E (n + 2) + E (n + 1)

/-- Theorem stating that there are 2872 ways for a paperboy to deliver newspapers to 12 houses without missing four consecutive houses. -/
theorem paperboy_delivery_ways : E 12 = 2872 := by
  sorry

end NUMINAMATH_CALUDE_paperboy_delivery_ways_l1071_107108


namespace NUMINAMATH_CALUDE_yvonne_word_count_l1071_107196

/-- Proves that Yvonne wrote 400 words given the conditions of the research paper problem -/
theorem yvonne_word_count 
  (total_required : Nat) 
  (janna_extra : Nat) 
  (words_removed : Nat) 
  (words_to_add : Nat) 
  (h1 : total_required = 1000)
  (h2 : janna_extra = 150)
  (h3 : words_removed = 20)
  (h4 : words_to_add = 30) : 
  ∃ (yvonne_words : Nat), 
    yvonne_words + (yvonne_words + janna_extra) - words_removed + 2 * words_removed + words_to_add = total_required ∧ 
    yvonne_words = 400 := by
  sorry

#check yvonne_word_count

end NUMINAMATH_CALUDE_yvonne_word_count_l1071_107196


namespace NUMINAMATH_CALUDE_altitude_and_median_equations_l1071_107147

/-- Triangle with vertices A(4,0), B(6,7), and C(0,3) -/
structure Triangle where
  A : ℝ × ℝ := (4, 0)
  B : ℝ × ℝ := (6, 7)
  C : ℝ × ℝ := (0, 3)

/-- Line represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Altitude from B to side AC -/
def altitude (t : Triangle) : Line :=
  { a := 3, b := 2, c := -12 }

/-- Median from B to side AC -/
def median (t : Triangle) : Line :=
  { a := 4, b := -6, c := 1 }

/-- Theorem stating that the altitude and median equations are correct -/
theorem altitude_and_median_equations (t : Triangle) : 
  (altitude t = { a := 3, b := 2, c := -12 }) ∧ 
  (median t = { a := 4, b := -6, c := 1 }) := by
  sorry

end NUMINAMATH_CALUDE_altitude_and_median_equations_l1071_107147


namespace NUMINAMATH_CALUDE_eli_calculation_l1071_107186

theorem eli_calculation (x : ℝ) (h : (8 * x - 7) / 5 = 63) : (5 * x - 7) / 8 = 24.28125 := by
  sorry

end NUMINAMATH_CALUDE_eli_calculation_l1071_107186


namespace NUMINAMATH_CALUDE_complex_sum_of_powers_of_i_l1071_107158

theorem complex_sum_of_powers_of_i : Complex.I + Complex.I^2 + Complex.I^3 + Complex.I^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_powers_of_i_l1071_107158


namespace NUMINAMATH_CALUDE_computer_program_output_l1071_107123

theorem computer_program_output (x : ℝ) (y : ℝ) : 
  x = Real.sqrt 3 - 2 → y = Real.sqrt ((x^2).sqrt - 2) → y = -Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_computer_program_output_l1071_107123


namespace NUMINAMATH_CALUDE_f_at_four_is_zero_l1071_107195

/-- A function f satisfying the given property for all real x -/
def f : ℝ → ℝ := sorry

/-- The main property of the function f -/
axiom f_property : ∀ x : ℝ, x * f x = 2 * f (2 - x) + 1

/-- The theorem to be proved -/
theorem f_at_four_is_zero : f 4 = 0 := by sorry

end NUMINAMATH_CALUDE_f_at_four_is_zero_l1071_107195


namespace NUMINAMATH_CALUDE_trip_distance_l1071_107113

theorem trip_distance (speed1 speed2 time_saved : ℝ) (h1 : speed1 = 50) (h2 : speed2 = 60) (h3 : time_saved = 4) :
  let distance := speed1 * speed2 * time_saved / (speed2 - speed1)
  distance = 1200 := by sorry

end NUMINAMATH_CALUDE_trip_distance_l1071_107113


namespace NUMINAMATH_CALUDE_max_product_f_value_l1071_107116

-- Define the function f
def f (a b x : ℝ) : ℝ := 2 * a * x + b

-- State the theorem
theorem max_product_f_value :
  ∀ a b : ℝ,
    a > 0 →
    b > 0 →
    (∀ x : ℝ, x ∈ Set.Icc (-1/2) (1/2) → |f a b x| ≤ 2) →
    (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
      (∀ x : ℝ, x ∈ Set.Icc (-1/2) (1/2) → |f a' b' x| ≤ 2) → 
      a * b ≥ a' * b') →
    f a b 2017 = 4035 :=
by
  sorry


end NUMINAMATH_CALUDE_max_product_f_value_l1071_107116


namespace NUMINAMATH_CALUDE_remainder_theorem_l1071_107185

-- Define the polynomial Q(x)
variable (Q : ℝ → ℝ)

-- Define the conditions
axiom Q_remainder_15 : ∃ P : ℝ → ℝ, ∀ x, Q x = P x * (x - 15) + 10
axiom Q_remainder_12 : ∃ P : ℝ → ℝ, ∀ x, Q x = P x * (x - 12) + 2

-- Theorem statement
theorem remainder_theorem :
  ∃ R : ℝ → ℝ, ∀ x, Q x = R x * ((x - 12) * (x - 15)) + (8/3 * x - 30) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1071_107185


namespace NUMINAMATH_CALUDE_pizza_slices_per_pizza_pizza_problem_l1071_107181

theorem pizza_slices_per_pizza (num_pizzas : ℕ) (total_cost : ℚ) (slices_sample : ℕ) (cost_sample : ℚ) : ℚ :=
  let cost_per_slice : ℚ := cost_sample / slices_sample
  let cost_per_pizza : ℚ := total_cost / num_pizzas
  cost_per_pizza / cost_per_slice

theorem pizza_problem : pizza_slices_per_pizza 3 72 5 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_per_pizza_pizza_problem_l1071_107181


namespace NUMINAMATH_CALUDE_no_five_linked_country_with_46_airlines_l1071_107192

theorem no_five_linked_country_with_46_airlines :
  ¬ ∃ (n : ℕ), n > 0 ∧ (5 * n) / 2 = 46 := by
  sorry

end NUMINAMATH_CALUDE_no_five_linked_country_with_46_airlines_l1071_107192


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_l1071_107100

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3 * x - 7 ≥ 8 - 2 * x}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem for the union of A and complement of B
theorem union_A_complement_B : A ∪ (Set.univ \ B) = {x : ℝ | x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_l1071_107100


namespace NUMINAMATH_CALUDE_decimal_multiplication_equivalence_l1071_107122

theorem decimal_multiplication_equivalence (given : 268 * 74 = 19832) :
  ∃ x : ℝ, 2.68 * x = 1.9832 ∧ x = 0.74 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_equivalence_l1071_107122


namespace NUMINAMATH_CALUDE_committee_count_theorem_l1071_107172

/-- The number of ways to choose a committee with at least one female member -/
def committee_count (total_members : ℕ) (committee_size : ℕ) (female_members : ℕ) : ℕ :=
  Nat.choose total_members committee_size - Nat.choose (total_members - female_members) committee_size

theorem committee_count_theorem :
  committee_count 30 5 12 = 133938 := by
  sorry

end NUMINAMATH_CALUDE_committee_count_theorem_l1071_107172


namespace NUMINAMATH_CALUDE_books_bought_two_years_ago_l1071_107175

/-- Represents the number of books in a library over time --/
structure LibraryBooks where
  initial : ℕ  -- Initial number of books 5 years ago
  bought_two_years_ago : ℕ  -- Books bought 2 years ago
  bought_last_year : ℕ  -- Books bought last year
  donated : ℕ  -- Books donated this year
  current : ℕ  -- Current number of books

/-- Theorem stating the number of books bought two years ago --/
theorem books_bought_two_years_ago 
  (lib : LibraryBooks) 
  (h1 : lib.initial = 500)
  (h2 : lib.bought_last_year = lib.bought_two_years_ago + 100)
  (h3 : lib.donated = 200)
  (h4 : lib.current = 1000)
  (h5 : lib.current = lib.initial + lib.bought_two_years_ago + lib.bought_last_year - lib.donated) :
  lib.bought_two_years_ago = 300 := by
  sorry

#check books_bought_two_years_ago

end NUMINAMATH_CALUDE_books_bought_two_years_ago_l1071_107175


namespace NUMINAMATH_CALUDE_F_3_f_4_equals_7_l1071_107112

def f (a : ℝ) : ℝ := a - 2

def F (a b : ℝ) : ℝ := b^2 + a

theorem F_3_f_4_equals_7 : F 3 (f 4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_F_3_f_4_equals_7_l1071_107112


namespace NUMINAMATH_CALUDE_range_of_a_l1071_107183

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x a : ℝ) : Prop := x > a

-- Define the property that ¬q is sufficient but not necessary for ¬p
def not_q_sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, ¬(q x a) → ¬(p x)) ∧ ¬(∀ x, ¬(p x) → ¬(q x a))

-- Theorem statement
theorem range_of_a (a : ℝ) (h : not_q_sufficient_not_necessary a) : a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1071_107183


namespace NUMINAMATH_CALUDE_solve_for_c_l1071_107137

theorem solve_for_c (c d : ℚ) 
  (eq1 : (c - 34) / 2 = (2 * d - 8) / 7)
  (eq2 : d = c + 9) : 
  c = 86 := by
sorry

end NUMINAMATH_CALUDE_solve_for_c_l1071_107137


namespace NUMINAMATH_CALUDE_y_derivative_at_zero_l1071_107128

noncomputable def y (x : ℝ) : ℝ := Real.exp (Real.sin x) * Real.cos (Real.sin x)

theorem y_derivative_at_zero : 
  deriv y 0 = 1 := by sorry

end NUMINAMATH_CALUDE_y_derivative_at_zero_l1071_107128


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1071_107118

theorem solution_set_inequality (x : ℝ) : 
  (0 < x ∧ x < 2) ↔ (4 / x > |x| ∧ x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1071_107118


namespace NUMINAMATH_CALUDE_zero_additive_identity_for_integers_l1071_107177

theorem zero_additive_identity_for_integers : 
  ∃! y : ℤ, ∀ x : ℤ, y + x = x :=
by sorry

end NUMINAMATH_CALUDE_zero_additive_identity_for_integers_l1071_107177


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l1071_107134

theorem quadratic_solution_difference : ∃ (x₁ x₂ : ℝ),
  (x₁^2 - 5*x₁ + 15 = x₁ + 55) ∧
  (x₂^2 - 5*x₂ + 15 = x₂ + 55) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l1071_107134


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l1071_107169

/-- The standard equation of a hyperbola given its asymptotes and shared foci with an ellipse -/
theorem hyperbola_standard_equation
  (asymptote_slope : ℝ)
  (ellipse_a : ℝ)
  (ellipse_b : ℝ)
  (h_asymptote : asymptote_slope = 2)
  (h_ellipse : ellipse_a^2 = 49 ∧ ellipse_b^2 = 24) :
  ∃ (a b : ℝ), a^2 = 25 ∧ b^2 = 100 ∧
    ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l1071_107169


namespace NUMINAMATH_CALUDE_seeds_per_flower_bed_l1071_107151

theorem seeds_per_flower_bed 
  (total_seeds : ℕ) 
  (num_flower_beds : ℕ) 
  (h1 : total_seeds = 60) 
  (h2 : num_flower_beds = 6) 
  : total_seeds / num_flower_beds = 10 := by
  sorry

end NUMINAMATH_CALUDE_seeds_per_flower_bed_l1071_107151


namespace NUMINAMATH_CALUDE_inequality_proof_l1071_107164

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  2 * (a + b + c) + 9 / ((a * b + b * c + c * a) ^ 2) ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1071_107164


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1071_107191

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 1 = 3 →  -- First term is 3
  a 1 + a 2 + a 3 = 21 →  -- Sum of first three terms is 21
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1071_107191


namespace NUMINAMATH_CALUDE_proposition_variants_l1071_107189

theorem proposition_variants (a b : ℝ) : 
  (∀ a b, a ≤ b → a - 2 ≤ b - 2) ∧ 
  (∀ a b, a - 2 > b - 2 → a > b) ∧ 
  (∀ a b, a - 2 ≤ b - 2 → a ≤ b) ∧ 
  ¬(∀ a b, a > b → a - 2 ≤ b - 2) := by
  sorry

end NUMINAMATH_CALUDE_proposition_variants_l1071_107189


namespace NUMINAMATH_CALUDE_snow_fall_time_l1071_107182

/-- Given that snow falls at a rate of 1 mm every 6 minutes, prove that it takes 100 hours for 1 m of snow to fall. -/
theorem snow_fall_time (rate : ℝ) (h1 : rate = 1 / 6) : (1000 / rate) / 60 = 100 := by
  sorry

end NUMINAMATH_CALUDE_snow_fall_time_l1071_107182


namespace NUMINAMATH_CALUDE_intersection_area_l1071_107180

/-- The area of a region formed by the intersection of four circles -/
theorem intersection_area (r : ℝ) (h : r = 5) : 
  ∃ (A : ℝ), A = 50 * (π - 2) ∧ 
  A = 8 * (((π * r^2) / 4) - ((r^2) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_intersection_area_l1071_107180


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1071_107190

/-- Given that i² = -1, prove that (2-i)/(1+4i) = -2/17 - 9/17*i -/
theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) :
  (2 - i) / (1 + 4*i) = -2/17 - 9/17*i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1071_107190


namespace NUMINAMATH_CALUDE_intersection_angle_l1071_107184

theorem intersection_angle (φ : Real) 
  (h1 : 0 ≤ φ) (h2 : φ < π)
  (h3 : 2 * Real.cos (π/3) = 2 * Real.sin (2 * (π/3) + φ)) :
  φ = π/6 := by sorry

end NUMINAMATH_CALUDE_intersection_angle_l1071_107184


namespace NUMINAMATH_CALUDE_three_digit_cube_divisible_by_8_and_9_l1071_107115

theorem three_digit_cube_divisible_by_8_and_9 :
  ∃! n : ℕ, 100 ≤ n^3 ∧ n^3 ≤ 999 ∧ 6 ∣ n ∧ 8 ∣ n^3 ∧ 9 ∣ n^3 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_cube_divisible_by_8_and_9_l1071_107115


namespace NUMINAMATH_CALUDE_tree_increase_factor_l1071_107187

theorem tree_increase_factor (initial_maples : ℝ) (initial_lindens : ℝ) 
  (spring_total : ℝ) (autumn_total : ℝ) : 
  initial_maples / (initial_maples + initial_lindens) = 3/5 →
  initial_maples / spring_total = 1/5 →
  initial_maples / autumn_total = 3/5 →
  autumn_total / (initial_maples + initial_lindens) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_increase_factor_l1071_107187


namespace NUMINAMATH_CALUDE_arrangement_count_proof_l1071_107160

/-- The number of ways to arrange 4 men and 4 women into two indistinguishable groups
    of two (each containing one man and one woman) and one group of four
    (containing the remaining two men and two women) -/
def arrangement_count : ℕ := 72

/-- The number of ways to choose one man from 4 men -/
def choose_man : ℕ := 4

/-- The number of ways to choose one woman from 4 women -/
def choose_woman : ℕ := 4

/-- The number of ways to choose one man from 3 remaining men -/
def choose_remaining_man : ℕ := 3

/-- The number of ways to choose one woman from 3 remaining women -/
def choose_remaining_woman : ℕ := 3

/-- The number of ways to arrange two indistinguishable groups -/
def indistinguishable_groups : ℕ := 2

theorem arrangement_count_proof :
  arrangement_count = (choose_man * choose_woman * choose_remaining_man * choose_remaining_woman) / indistinguishable_groups :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_proof_l1071_107160


namespace NUMINAMATH_CALUDE_median_in_60_64_interval_l1071_107146

/-- Represents the score intervals in the histogram --/
inductive ScoreInterval
| I50_54
| I55_59
| I60_64
| I65_69
| I70_74

/-- The frequency of scores in each interval --/
def frequency : ScoreInterval → Nat
| ScoreInterval.I50_54 => 3
| ScoreInterval.I55_59 => 5
| ScoreInterval.I60_64 => 10
| ScoreInterval.I65_69 => 15
| ScoreInterval.I70_74 => 20

/-- The total number of students --/
def totalStudents : Nat := 100

/-- The position of the median in the ordered list of scores --/
def medianPosition : Nat := (totalStudents + 1) / 2

/-- Theorem stating that the median score is in the interval 60-64 --/
theorem median_in_60_64_interval :
  ∃ k : Nat, k ≤ medianPosition ∧
  (frequency ScoreInterval.I70_74 + frequency ScoreInterval.I65_69 + frequency ScoreInterval.I60_64) ≥ k ∧
  (frequency ScoreInterval.I70_74 + frequency ScoreInterval.I65_69) < k :=
by sorry

end NUMINAMATH_CALUDE_median_in_60_64_interval_l1071_107146


namespace NUMINAMATH_CALUDE_gcf_75_90_l1071_107168

theorem gcf_75_90 : Nat.gcd 75 90 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_75_90_l1071_107168


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1071_107198

theorem quadratic_expression_value (a b : ℝ) : 
  (2 : ℝ)^2 + a * 2 - 6 = 0 ∧ 
  b^2 + a * b - 6 = 0 → 
  (2 * a + b)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1071_107198


namespace NUMINAMATH_CALUDE_prob_blue_twelve_sided_die_l1071_107150

/-- A die with a specified number of sides and blue faces -/
structure Die where
  sides : ℕ
  blue_faces : ℕ
  blue_faces_le_sides : blue_faces ≤ sides

/-- The probability of rolling a blue face on a given die -/
def prob_blue (d : Die) : ℚ :=
  d.blue_faces / d.sides

/-- Theorem: The probability of rolling a blue face on a 12-sided die with 4 blue faces is 1/3 -/
theorem prob_blue_twelve_sided_die :
  ∃ d : Die, d.sides = 12 ∧ d.blue_faces = 4 ∧ prob_blue d = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_blue_twelve_sided_die_l1071_107150


namespace NUMINAMATH_CALUDE_congruence_solution_l1071_107171

theorem congruence_solution : ∃! n : ℕ, n < 47 ∧ (13 * n) % 47 = 9 % 47 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1071_107171


namespace NUMINAMATH_CALUDE_volunteer_hours_per_time_l1071_107199

/-- The number of times John volunteers per month -/
def volunteering_frequency : ℕ := 2

/-- The total number of hours John volunteers per year -/
def total_hours_per_year : ℕ := 72

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- Theorem stating how many hours John volunteers at a time -/
theorem volunteer_hours_per_time :
  total_hours_per_year / (volunteering_frequency * months_per_year) = 3 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_hours_per_time_l1071_107199


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1071_107156

theorem purely_imaginary_complex_number (m : ℝ) : 
  (Complex.I * (m + 1) : ℂ).re = 0 ∧ (Complex.I * (m + 1) : ℂ).im ≠ 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1071_107156


namespace NUMINAMATH_CALUDE_evaluate_expression_l1071_107149

theorem evaluate_expression : 3000 * (3000 ^ 3000) = 3000 ^ 3001 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1071_107149


namespace NUMINAMATH_CALUDE_problem_statement_l1071_107163

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (∀ x y, x > 0 ∧ y > 0 ∧ x + y = 1 → x * y ≤ a * b) ∧
  (a^2 + b^2 ≥ 1/2) ∧
  (4/a + 1/b ≥ 9) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1071_107163


namespace NUMINAMATH_CALUDE_car_banker_speed_ratio_l1071_107105

/-- The ratio of car speed to banker speed given specific timing conditions -/
theorem car_banker_speed_ratio :
  ∀ (T : ℝ) (Vb Vc : ℝ) (d : ℝ),
    Vb > 0 →
    Vc > 0 →
    d > 0 →
    (Vb * 60 = Vc * 5) →
    (Vc / Vb = 12) := by
  sorry

end NUMINAMATH_CALUDE_car_banker_speed_ratio_l1071_107105


namespace NUMINAMATH_CALUDE_charity_event_selection_methods_l1071_107153

def total_students : ℕ := 10
def selected_students : ℕ := 4
def special_students : ℕ := 2  -- A and B

-- Function to calculate the number of ways to select k items from n items
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem charity_event_selection_methods :
  (choose (total_students - special_students) (selected_students - special_students) +
   choose (total_students - special_students) (selected_students - 1) * special_students) = 140 :=
by sorry

end NUMINAMATH_CALUDE_charity_event_selection_methods_l1071_107153


namespace NUMINAMATH_CALUDE_house_pets_problem_l1071_107117

theorem house_pets_problem (total : Nat) (dogs cats turtles : Nat)
  (h_total : total = 2017)
  (h_dogs : dogs = 1820)
  (h_cats : cats = 1651)
  (h_turtles : turtles = 1182)
  (h_dogs_le : dogs ≤ total)
  (h_cats_le : cats ≤ total)
  (h_turtles_le : turtles ≤ total) :
  ∃ (max min : Nat),
    (max ≤ turtles) ∧
    (min ≥ dogs + cats + turtles - 2 * total) ∧
    (max - min = 563) := by
  sorry

end NUMINAMATH_CALUDE_house_pets_problem_l1071_107117


namespace NUMINAMATH_CALUDE_fraction_simplification_l1071_107197

theorem fraction_simplification (m : ℝ) (h : m ≠ 3 ∧ m ≠ -3) :
  (m^2 - 3*m) / (9 - m^2) = -m / (m + 3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1071_107197


namespace NUMINAMATH_CALUDE_g_inequality_range_l1071_107104

noncomputable def g (x : ℝ) : ℝ := 2^x + 2^(-x) + |x|

theorem g_inequality_range : 
  {x : ℝ | g (2*x - 1) < g 3} = Set.Ioo (-1) 2 := by sorry

end NUMINAMATH_CALUDE_g_inequality_range_l1071_107104


namespace NUMINAMATH_CALUDE_fourth_ball_black_prob_l1071_107179

/-- A box containing red and black balls -/
structure Box where
  red_balls : ℕ
  black_balls : ℕ

/-- The probability of selecting a black ball from a box -/
def prob_black_ball (box : Box) : ℚ :=
  box.black_balls / (box.red_balls + box.black_balls)

/-- The theorem stating the probability of the fourth ball being black -/
theorem fourth_ball_black_prob (box : Box) 
  (h1 : box.red_balls = 3) 
  (h2 : box.black_balls = 3) : 
  prob_black_ball box = 1/2 := by
  sorry

#eval prob_black_ball { red_balls := 3, black_balls := 3 }

end NUMINAMATH_CALUDE_fourth_ball_black_prob_l1071_107179


namespace NUMINAMATH_CALUDE_area_cosine_plus_one_l1071_107103

/-- The area enclosed by y = 1 + cos x and the x-axis over [-π, π] is 2π -/
theorem area_cosine_plus_one : 
  (∫ x in -π..π, (1 + Real.cos x)) = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_area_cosine_plus_one_l1071_107103


namespace NUMINAMATH_CALUDE_tangent_point_circle_properties_l1071_107174

/-- The equation of a circle with center (x₀, y₀) and radius r -/
def circle_equation (x₀ y₀ r : ℝ) (x y : ℝ) : Prop :=
  (x - x₀)^2 + (y - y₀)^2 = r^2

/-- The equation of the line 2x + y = 0 -/
def line_center (x y : ℝ) : Prop :=
  2 * x + y = 0

/-- The equation of the line x + y - 1 = 0 -/
def line_tangent (x y : ℝ) : Prop :=
  x + y - 1 = 0

/-- The point (2, -1) lies on the tangent line -/
theorem tangent_point : line_tangent 2 (-1) := by sorry

theorem circle_properties (x₀ y₀ r : ℝ) :
  line_center x₀ y₀ →
  (∀ x y, circle_equation x₀ y₀ r x y ↔ (x - 1)^2 + (y + 2)^2 = 2) →
  (∃ x y, circle_equation x₀ y₀ r x y ∧ line_tangent x y) →
  circle_equation x₀ y₀ r 2 (-1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_circle_properties_l1071_107174


namespace NUMINAMATH_CALUDE_mascot_problem_solution_l1071_107140

/-- Represents the sales data for a week -/
structure WeekSales where
  bing : ℕ
  shuey : ℕ
  revenue : ℕ

/-- Solves for mascot prices and maximum purchase given sales data and budget -/
def solve_mascot_problem (week1 week2 : WeekSales) (total_budget total_mascots : ℕ) :
  (ℕ × ℕ × ℕ) :=
sorry

/-- Theorem stating the correctness of the solution -/
theorem mascot_problem_solution :
  let week1 : WeekSales := ⟨3, 5, 1800⟩
  let week2 : WeekSales := ⟨4, 10, 3100⟩
  let (bing_price, shuey_price, max_bing) := solve_mascot_problem week1 week2 6700 30
  bing_price = 250 ∧ shuey_price = 210 ∧ max_bing = 10 :=
sorry

end NUMINAMATH_CALUDE_mascot_problem_solution_l1071_107140


namespace NUMINAMATH_CALUDE_no_valid_A_l1071_107194

theorem no_valid_A : ¬∃ (A : ℕ), A ≤ 9 ∧ 45 % A = 0 ∧ (456204 + A * 10) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_A_l1071_107194


namespace NUMINAMATH_CALUDE_opposite_numbers_sum_l1071_107188

theorem opposite_numbers_sum (a b : ℤ) : (a + b = 0) → (2006 * a + 2006 * b = 0) := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_sum_l1071_107188


namespace NUMINAMATH_CALUDE_lawrence_county_kids_count_l1071_107126

/-- The number of kids in Lawrence county who stayed home -/
def kids_stayed_home : ℕ := 644997

/-- The number of kids in Lawrence county who went to camp -/
def kids_went_to_camp : ℕ := 893835

/-- The number of kids from outside the county who attended the camp -/
def outside_kids_in_camp : ℕ := 78

/-- The total number of kids in Lawrence county -/
def total_kids_in_county : ℕ := kids_stayed_home + kids_went_to_camp

theorem lawrence_county_kids_count : total_kids_in_county = 1538832 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_count_l1071_107126


namespace NUMINAMATH_CALUDE_winnie_balloon_distribution_l1071_107130

/-- Calculates the number of balloons left after equal distribution --/
def balloons_left (red blue green purple friends : ℕ) : ℕ :=
  (red + blue + green + purple) % friends

/-- Proves that Winnie has 0 balloons left after distribution --/
theorem winnie_balloon_distribution :
  balloons_left 22 44 66 88 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_winnie_balloon_distribution_l1071_107130


namespace NUMINAMATH_CALUDE_lottery_distribution_l1071_107131

/-- The total amount received by 100 students, each getting one-thousandth of $155250 -/
def total_amount (lottery_win : ℚ) (num_students : ℕ) : ℚ :=
  (lottery_win / 1000) * num_students

theorem lottery_distribution :
  total_amount 155250 100 = 15525 := by
  sorry

end NUMINAMATH_CALUDE_lottery_distribution_l1071_107131


namespace NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l1071_107145

theorem order_of_logarithmic_expressions :
  let a : ℝ := (Real.log (Real.sqrt 2)) / 2
  let b : ℝ := (Real.log 3) / 6
  let c : ℝ := 1 / (2 * Real.exp 1)
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l1071_107145


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1071_107125

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 + 2*x + 3 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1071_107125


namespace NUMINAMATH_CALUDE_gretchen_scuba_trips_l1071_107178

/-- The minimum number of trips required to transport a given number of objects,
    where each trip can carry a fixed number of objects. -/
def min_trips (total_objects : ℕ) (objects_per_trip : ℕ) : ℕ :=
  (total_objects + objects_per_trip - 1) / objects_per_trip

/-- Theorem stating that 6 trips are required to transport 17 objects
    when carrying 3 objects per trip. -/
theorem gretchen_scuba_trips :
  min_trips 17 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gretchen_scuba_trips_l1071_107178


namespace NUMINAMATH_CALUDE_quadratic_function_range_l1071_107109

/-- Given a quadratic function f(x) = ax² - c satisfying certain conditions,
    prove that f(3) is within a specific range. -/
theorem quadratic_function_range (a c : ℝ) (f : ℝ → ℝ) 
    (h_def : ∀ x, f x = a * x^2 - c)
    (h_1 : -4 ≤ f 1 ∧ f 1 ≤ -1)
    (h_2 : -1 ≤ f 2 ∧ f 2 ≤ 5) :
  -1 ≤ f 3 ∧ f 3 ≤ 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l1071_107109


namespace NUMINAMATH_CALUDE_probability_two_girls_l1071_107166

theorem probability_two_girls (total : Nat) (girls : Nat) (selected : Nat) : 
  total = 6 → girls = 4 → selected = 2 →
  (Nat.choose girls selected : Rat) / (Nat.choose total selected : Rat) = 2/5 := by
sorry

end NUMINAMATH_CALUDE_probability_two_girls_l1071_107166


namespace NUMINAMATH_CALUDE_ian_says_smallest_unclaimed_number_l1071_107132

/-- Represents a student in the counting game -/
structure Student where
  name : String
  index : Nat

/-- The list of students in alphabetical order -/
def students : List Student := [
  ⟨"Alice", 0⟩, ⟨"Barbara", 1⟩, ⟨"Candice", 2⟩, ⟨"Debbie", 3⟩,
  ⟨"Eliza", 4⟩, ⟨"Fatima", 5⟩, ⟨"Greg", 6⟩, ⟨"Helen", 7⟩
]

/-- The maximum number in the counting sequence -/
def maxNumber : Nat := 1200

/-- Determines if a student says a given number -/
def saysNumber (s : Student) (n : Nat) : Prop :=
  n ≤ maxNumber ∧ n % (4 * 4^s.index) ≠ 0

/-- The number that Ian says -/
def iansNumber : Nat := 1021

/-- Theorem stating that Ian's number is the smallest not said by any other student -/
theorem ian_says_smallest_unclaimed_number :
  (∀ n < iansNumber, ∃ s ∈ students, saysNumber s n) ∧
  (∀ s ∈ students, ¬saysNumber s iansNumber) :=
sorry

end NUMINAMATH_CALUDE_ian_says_smallest_unclaimed_number_l1071_107132


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l1071_107120

theorem min_value_of_fraction (x : ℝ) (h : x ≥ 3/2) :
  (2*x^2 - 2*x + 1) / (x - 1) ≥ 2*Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l1071_107120


namespace NUMINAMATH_CALUDE_complex_fraction_power_simplification_l1071_107102

theorem complex_fraction_power_simplification :
  (((3 : ℂ) + 4*I) / ((3 : ℂ) - 4*I))^8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_power_simplification_l1071_107102


namespace NUMINAMATH_CALUDE_range_of_a_l1071_107135

-- Define propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

-- Define the theorem
theorem range_of_a :
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∀ a : ℝ, (a < 0 ∨ (1/4 < a ∧ a < 4)) ↔ (p a ∨ q a) ∧ ¬(p a ∧ q a)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1071_107135


namespace NUMINAMATH_CALUDE_decimal_multiplication_l1071_107176

theorem decimal_multiplication (a b : ℚ) (n m : ℕ) :
  a = 0.125 →
  b = 3.84 →
  (a * 10^3).num * (b * 10^2).num = 48000 →
  a * b = 0.48 := by
sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l1071_107176


namespace NUMINAMATH_CALUDE_diamond_self_not_always_zero_l1071_107139

-- Define the diamond operator
def diamond (x y : ℝ) : ℝ := |x - 2*y|

-- Theorem stating that the statement "For all real x, x ◇ x = 0" is false
theorem diamond_self_not_always_zero : ¬ ∀ x : ℝ, diamond x x = 0 := by
  sorry

end NUMINAMATH_CALUDE_diamond_self_not_always_zero_l1071_107139


namespace NUMINAMATH_CALUDE_rocket_fuel_ratio_l1071_107133

theorem rocket_fuel_ratio (m M : ℝ) (h : m > 0) :
  2000 * Real.log (1 + M / m) = 12000 → M / m = Real.exp 6 - 1 := by
  sorry

end NUMINAMATH_CALUDE_rocket_fuel_ratio_l1071_107133


namespace NUMINAMATH_CALUDE_simplify_square_roots_l1071_107159

theorem simplify_square_roots : Real.sqrt 81 - Real.sqrt 144 = -3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l1071_107159


namespace NUMINAMATH_CALUDE_polynomial_on_unit_circle_l1071_107111

/-- A polynomial p(z) with complex coefficients a and b -/
def p (a b z : ℂ) : ℂ := z^2 + a*z + b

/-- The property that |p(z)| = 1 on the unit circle -/
def unit_circle_property (a b : ℂ) : Prop :=
  ∀ z : ℂ, Complex.abs z = 1 → Complex.abs (p a b z) = 1

/-- Theorem: If |p(z)| = 1 on the unit circle, then a = 0 and b = 0 -/
theorem polynomial_on_unit_circle (a b : ℂ) 
  (h : unit_circle_property a b) : a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_on_unit_circle_l1071_107111


namespace NUMINAMATH_CALUDE_smallest_difference_in_triangle_l1071_107142

theorem smallest_difference_in_triangle (PQ PR QR : ℕ) : 
  PQ + PR + QR = 2021 →  -- Perimeter condition
  PQ < PR →              -- PQ < PR condition
  PR = (3 * PQ) / 2 →    -- PR = 1.5 × PQ condition
  PQ > 0 ∧ PR > 0 ∧ QR > 0 →  -- Positive side lengths
  PQ + QR > PR ∧ PR + QR > PQ ∧ PQ + PR > QR →  -- Triangle inequality
  PR - PQ ≥ 204 :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_in_triangle_l1071_107142


namespace NUMINAMATH_CALUDE_exists_functions_with_even_product_l1071_107106

-- Define the type for real-valued functions
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be even
def IsEven (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define what it means for a function to be odd
def IsOdd (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define what it means for a function to be neither odd nor even
def NeitherOddNorEven (f : RealFunction) : Prop :=
  ¬(IsEven f) ∧ ¬(IsOdd f)

-- State the theorem
theorem exists_functions_with_even_product :
  ∃ (f g : RealFunction),
    NeitherOddNorEven f ∧
    NeitherOddNorEven g ∧
    IsEven (fun x ↦ f x * g x) :=
by sorry

end NUMINAMATH_CALUDE_exists_functions_with_even_product_l1071_107106


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1071_107141

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1071_107141


namespace NUMINAMATH_CALUDE_quadratic_value_relation_l1071_107136

theorem quadratic_value_relation (x : ℝ) (h : x^2 + x + 1 = 8) : 4*x^2 + 4*x + 9 = 37 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_relation_l1071_107136


namespace NUMINAMATH_CALUDE_central_angle_common_chord_l1071_107143

/-- The central angle corresponding to the common chord of two circles -/
theorem central_angle_common_chord (x y : ℝ) : 
  let circle1 := {(x, y) | (x - 2)^2 + y^2 = 4}
  let circle2 := {(x, y) | x^2 + (y - 2)^2 = 4}
  let center1 := (2, 0)
  let center2 := (0, 2)
  let radius := 2
  let center_distance := Real.sqrt ((2 - 0)^2 + (0 - 2)^2)
  let chord_distance := center_distance / 2
  let cos_half_angle := chord_distance / radius
  let central_angle := 2 * Real.arccos cos_half_angle
  central_angle = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_common_chord_l1071_107143


namespace NUMINAMATH_CALUDE_photo_difference_l1071_107119

theorem photo_difference (claire_photos : ℕ) (lisa_photos : ℕ) (robert_photos : ℕ) : 
  claire_photos = 12 →
  lisa_photos = 3 * claire_photos →
  robert_photos = lisa_photos →
  robert_photos - claire_photos = 24 := by sorry

end NUMINAMATH_CALUDE_photo_difference_l1071_107119


namespace NUMINAMATH_CALUDE_permutation_inequality_solution_l1071_107152

def A (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

theorem permutation_inequality_solution :
  ∃! x : ℕ+, A 8 x < 6 * A 8 (x - 2) ∧ x = 8 := by sorry

end NUMINAMATH_CALUDE_permutation_inequality_solution_l1071_107152


namespace NUMINAMATH_CALUDE_rider_pedestrian_problem_l1071_107148

/-- A problem about a rider and a pedestrian traveling between two points. -/
theorem rider_pedestrian_problem
  (total_time : ℝ) -- Total time for the rider's journey
  (time_difference : ℝ) -- Time difference between rider and pedestrian arriving at B
  (meeting_distance : ℝ) -- Distance from B where rider meets pedestrian on return
  (h_total_time : total_time = 100 / 60) -- Total time is 1 hour 40 minutes (100 minutes)
  (h_time_difference : time_difference = 50 / 60) -- Rider arrives 50 minutes earlier
  (h_meeting_distance : meeting_distance = 2) -- They meet 2 km from B
  : ∃ (distance speed_rider speed_pedestrian : ℝ),
    distance = 6 ∧ 
    speed_rider = 7.2 ∧ 
    speed_pedestrian = 3.6 :=
by sorry

end NUMINAMATH_CALUDE_rider_pedestrian_problem_l1071_107148


namespace NUMINAMATH_CALUDE_tangent_at_one_l1071_107162

/-- A polynomial function of degree 4 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^3 + 1

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 3 * b * x^2

theorem tangent_at_one (a b : ℝ) : 
  (f a b 1 = 0 ∧ f' a b 1 = 0) ↔ (a = 3 ∧ b = -4) := by sorry

end NUMINAMATH_CALUDE_tangent_at_one_l1071_107162


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l1071_107167

-- Define the parabola
def parabola (x : ℝ) : ℝ := (x - 5)^2 + 4

-- State the theorem
theorem axis_of_symmetry :
  ∀ x : ℝ, parabola (5 + x) = parabola (5 - x) := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l1071_107167


namespace NUMINAMATH_CALUDE_third_term_value_l1071_107155

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem third_term_value
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_first : a 1 = -11)
  (h_sum : a 4 + a 6 = -6) :
  a 3 = -7 :=
sorry

end NUMINAMATH_CALUDE_third_term_value_l1071_107155


namespace NUMINAMATH_CALUDE_machine_production_time_l1071_107124

/-- Proves that a machine producing 360 items in 2 hours takes 1/3 minute to produce one item. -/
theorem machine_production_time 
  (items_produced : ℕ) 
  (production_hours : ℕ) 
  (minutes_per_hour : ℕ) 
  (h1 : items_produced = 360)
  (h2 : production_hours = 2)
  (h3 : minutes_per_hour = 60) :
  (production_hours * minutes_per_hour) / items_produced = 1 / 3 := by
  sorry

#check machine_production_time

end NUMINAMATH_CALUDE_machine_production_time_l1071_107124


namespace NUMINAMATH_CALUDE_max_value_problem_l1071_107101

theorem max_value_problem (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 3) :
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_problem_l1071_107101


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l1071_107110

/-- A rhombus with side length 35 units and shorter diagonal 42 units has a longer diagonal of 56 units. -/
theorem rhombus_longer_diagonal (s d_short : ℝ) (h1 : s = 35) (h2 : d_short = 42) :
  let d_long := Real.sqrt (4 * s^2 - d_short^2)
  d_long = 56 := by sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l1071_107110


namespace NUMINAMATH_CALUDE_min_distance_circle_to_line_l1071_107154

theorem min_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*x - 2*y + 1 = 0}
  let line := {(x, y) : ℝ × ℝ | 3*x + 4*y + 8 = 0}
  (∀ p ∈ circle, ∃ q ∈ line, ∀ r ∈ line, dist p q ≤ dist p r) ∧
  (∃ p ∈ circle, ∃ q ∈ line, dist p q = 2) ∧
  (∀ p ∈ circle, ∀ q ∈ line, dist p q ≥ 2) :=
by sorry

where
  dist : ℝ × ℝ → ℝ × ℝ → ℝ := λ (x₁, y₁) (x₂, y₂) => Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

end NUMINAMATH_CALUDE_min_distance_circle_to_line_l1071_107154


namespace NUMINAMATH_CALUDE_pirate_treasure_year_l1071_107127

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The base-8 representation of the year --/
def year_base8 : List Nat := [7, 6, 3]

/-- The claimed base-10 equivalent of the year --/
def year_base10 : Nat := 247

/-- Theorem stating that the base-8 year converts to the claimed base-10 year --/
theorem pirate_treasure_year : base8_to_base10 year_base8 = year_base10 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_year_l1071_107127


namespace NUMINAMATH_CALUDE_third_competitor_hotdogs_l1071_107161

/-- The number of hotdogs the third competitor can eat in a given time -/
def hotdogs_eaten_by_third (first_rate : ℕ) (second_multiplier third_multiplier time : ℕ) : ℕ :=
  first_rate * second_multiplier * third_multiplier * time

/-- Theorem: The third competitor eats 300 hotdogs in 5 minutes -/
theorem third_competitor_hotdogs :
  hotdogs_eaten_by_third 10 3 2 5 = 300 := by
  sorry

#eval hotdogs_eaten_by_third 10 3 2 5

end NUMINAMATH_CALUDE_third_competitor_hotdogs_l1071_107161


namespace NUMINAMATH_CALUDE_three_subset_M_l1071_107170

def M : Set ℤ := {x | ∃ n : ℤ, x = 4 * n - 1}

theorem three_subset_M : {3} ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_three_subset_M_l1071_107170


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1071_107107

/-- The quadratic equation x^2 - 4mx + 3m^2 = 0 -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 - 4*m*x + 3*m^2 = 0

theorem quadratic_equation_properties :
  ∀ m : ℝ,
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_equation x1 m ∧ quadratic_equation x2 m) ∧
  (∀ x1 x2 : ℝ, x1 > x2 → quadratic_equation x1 m → quadratic_equation x2 m → x1 - x2 = 2 → m = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1071_107107


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l1071_107138

theorem least_three_digit_multiple_of_13 : ∃ (n : ℕ), n = 104 ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 13 ∣ m → n ≤ m) ∧ 
  100 ≤ n ∧ n < 1000 ∧ 13 ∣ n :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l1071_107138


namespace NUMINAMATH_CALUDE_unique_function_satisfying_condition_l1071_107193

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, is_perfect_square (f a * f (a + b) - a * b)

theorem unique_function_satisfying_condition :
  ∃! f : ℕ → ℕ, satisfies_condition f ∧ ∀ x : ℕ, f x = x :=
sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_condition_l1071_107193


namespace NUMINAMATH_CALUDE_exists_sum_of_digits_div_11_l1071_107121

/-- Sum of digits function in base 10 -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: Among 39 consecutive natural numbers, there is always one whose sum of digits (in base 10) is divisible by 11 -/
theorem exists_sum_of_digits_div_11 (n : ℕ) : 
  ∃ k ∈ Finset.range 39, (sumOfDigits (n + k)) % 11 = 0 := by sorry

end NUMINAMATH_CALUDE_exists_sum_of_digits_div_11_l1071_107121
