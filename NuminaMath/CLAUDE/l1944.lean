import Mathlib

namespace NUMINAMATH_CALUDE_abhay_speed_l1944_194483

theorem abhay_speed (distance : ℝ) (a s : ℝ → ℝ) :
  distance = 30 →
  (∀ x, a x > 0 ∧ s x > 0) →
  (∀ x, distance / (a x) = distance / (s x) + 2) →
  (∀ x, distance / (2 * a x) = distance / (s x) - 1) →
  (∃ x, a x = 5 * Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_abhay_speed_l1944_194483


namespace NUMINAMATH_CALUDE_grid_arithmetic_progression_l1944_194455

def is_arithmetic_progression (a b c : ℚ) : Prop :=
  b - a = c - b

theorem grid_arithmetic_progression :
  ∀ x : ℚ,
  let pos_3_4 := 2*x - 103
  let pos_1_4 := 251 - 2*x
  let pos_1_3 := 2/3*(51 - 2*x)
  let pos_3_3 := x
  (is_arithmetic_progression pos_1_3 pos_3_3 pos_3_4 ∧
   is_arithmetic_progression pos_1_4 pos_3_3 pos_3_4) →
  x = 60 := by
sorry

end NUMINAMATH_CALUDE_grid_arithmetic_progression_l1944_194455


namespace NUMINAMATH_CALUDE_xiaobing_jumps_189_ropes_per_minute_l1944_194434

/-- The number of ropes Xiaohan jumps per minute -/
def xiaohan_ropes_per_minute : ℕ := 168

/-- The number of ropes Xiaobing jumps per minute -/
def xiaobing_ropes_per_minute : ℕ := xiaohan_ropes_per_minute + 21

/-- The number of ropes Xiaobing jumps in the given time -/
def xiaobing_ropes : ℕ := 135

/-- The number of ropes Xiaohan jumps in the given time -/
def xiaohan_ropes : ℕ := 120

theorem xiaobing_jumps_189_ropes_per_minute :
  (xiaobing_ropes : ℚ) / xiaobing_ropes_per_minute = (xiaohan_ropes : ℚ) / xiaohan_ropes_per_minute →
  xiaobing_ropes_per_minute = 189 := by
  sorry

end NUMINAMATH_CALUDE_xiaobing_jumps_189_ropes_per_minute_l1944_194434


namespace NUMINAMATH_CALUDE_subset_relation_and_complement_l1944_194461

open Set

theorem subset_relation_and_complement (S A B : Set α) :
  (∀ x, x ∈ (S \ A) → x ∈ B) →
  (A ⊇ (S \ B) ∧ A ≠ (S \ B)) :=
by sorry

end NUMINAMATH_CALUDE_subset_relation_and_complement_l1944_194461


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l1944_194456

theorem complex_expression_simplification :
  (7 - 3*Complex.I) - 4*(2 + 5*Complex.I) + 3*(1 - 4*Complex.I) = 2 - 35*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l1944_194456


namespace NUMINAMATH_CALUDE_cos_alpha_terminal_point_l1944_194422

/-- Given a point P(-12, 5) on the terminal side of angle α, prove that cos α = -12/13 -/
theorem cos_alpha_terminal_point (α : Real) :
  let P : Real × Real := (-12, 5)
  (P.1 = -12 ∧ P.2 = 5) → -- Point P is (-12, 5)
  (P.1 = -12 * Real.cos α ∧ P.2 = -12 * Real.sin α) → -- P is on the terminal side of α
  Real.cos α = -12/13 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_terminal_point_l1944_194422


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_l1944_194415

/-- The trajectory of a point equidistant from a fixed point and a line is a parabola -/
theorem trajectory_is_parabola (M : ℝ × ℝ) :
  (∀ (x y : ℝ), M = (x, y) →
    dist M (0, -3) = |y - 3|) →
  ∃ (x y : ℝ), M = (x, y) ∧ x^2 = -12*y :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_l1944_194415


namespace NUMINAMATH_CALUDE_haploid_12_pairs_implies_tetraploid_l1944_194421

/-- Represents the ploidy level of a plant -/
inductive Ploidy
  | Diploid
  | Triploid
  | Tetraploid
  | Hexaploid

/-- Represents a potato plant -/
structure PotatoPlant where
  ploidy : Ploidy

/-- Represents a haploid plant derived from anther culture -/
structure HaploidPlant where
  chromosomePairs : Nat

/-- Function to determine the ploidy of the original plant based on the haploid plant's chromosome pairs -/
def determinePloidy (haploid : HaploidPlant) : Ploidy :=
  if haploid.chromosomePairs = 12 then Ploidy.Tetraploid else Ploidy.Diploid

/-- Theorem stating that if a haploid plant derived from anther culture forms 12 chromosome pairs,
    then the original potato plant is tetraploid -/
theorem haploid_12_pairs_implies_tetraploid (haploid : HaploidPlant) (original : PotatoPlant) :
  haploid.chromosomePairs = 12 → original.ploidy = Ploidy.Tetraploid :=
by
  sorry


end NUMINAMATH_CALUDE_haploid_12_pairs_implies_tetraploid_l1944_194421


namespace NUMINAMATH_CALUDE_sin_plus_cos_value_l1944_194480

theorem sin_plus_cos_value (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) 
  (h3 : Real.sin α * Real.cos α = 1 / 8) : 
  Real.sin α + Real.cos α = Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_plus_cos_value_l1944_194480


namespace NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l1944_194449

/-- The volume of a cylinder minus two cones -/
theorem cylinder_minus_cones_volume (r h : ℝ) (hr : r = 15) (hh : h = 30) :
  π * r^2 * h - 2 * (1/3 * π * r^2 * (h/2)) = 4500 * π := by
  sorry

#check cylinder_minus_cones_volume

end NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l1944_194449


namespace NUMINAMATH_CALUDE_angela_age_in_five_years_l1944_194491

/-- Given that Angela is four times as old as Beth, and five years ago the sum of their ages was 45 years, prove that Angela will be 49 years old in five years. -/
theorem angela_age_in_five_years (angela beth : ℕ) 
  (h1 : angela = 4 * beth) 
  (h2 : angela - 5 + beth - 5 = 45) : 
  angela + 5 = 49 := by
  sorry

end NUMINAMATH_CALUDE_angela_age_in_five_years_l1944_194491


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1944_194487

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 28 x = Nat.choose 28 (2*x - 1)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1944_194487


namespace NUMINAMATH_CALUDE_expression_evaluation_l1944_194472

theorem expression_evaluation :
  let x : ℚ := 1/2
  x^2 * (x - 1) - x * (x^2 + x - 1) = 0 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1944_194472


namespace NUMINAMATH_CALUDE_mom_bought_39_shirts_l1944_194408

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 13

/-- The number of packages mom bought -/
def packages_bought : ℕ := 3

/-- The total number of t-shirts mom bought -/
def total_shirts : ℕ := shirts_per_package * packages_bought

theorem mom_bought_39_shirts : total_shirts = 39 := by
  sorry

end NUMINAMATH_CALUDE_mom_bought_39_shirts_l1944_194408


namespace NUMINAMATH_CALUDE_union_equals_real_l1944_194495

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 5}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- State the theorem
theorem union_equals_real : A ∪ B = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_equals_real_l1944_194495


namespace NUMINAMATH_CALUDE_missing_number_l1944_194468

theorem missing_number (x : ℤ) : x + 64 = 16 → x = -48 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_l1944_194468


namespace NUMINAMATH_CALUDE_rock_skipping_theorem_l1944_194436

/-- The number of times Bob can skip a rock -/
def bob_skips : ℕ := 12

/-- The number of times Jim can skip a rock -/
def jim_skips : ℕ := 15

/-- The number of rocks each person skipped -/
def rocks_skipped : ℕ := 10

/-- The total number of skips for both Bob and Jim -/
def total_skips : ℕ := bob_skips * rocks_skipped + jim_skips * rocks_skipped

theorem rock_skipping_theorem : total_skips = 270 := by
  sorry

end NUMINAMATH_CALUDE_rock_skipping_theorem_l1944_194436


namespace NUMINAMATH_CALUDE_number_of_divisors_of_36_l1944_194428

theorem number_of_divisors_of_36 : Nat.card {d : ℕ | d > 0 ∧ 36 % d = 0} = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_36_l1944_194428


namespace NUMINAMATH_CALUDE_special_line_properties_l1944_194493

/-- A line passing through (-2, 3) with x-intercept twice the y-intercept -/
def special_line (x y : ℝ) : Prop :=
  x + 2 * y - 4 = 0

theorem special_line_properties :
  (special_line (-2) 3) ∧
  (∃ a : ℝ, a ≠ 0 ∧ special_line (2 * a) 0 ∧ special_line 0 a) :=
by sorry

end NUMINAMATH_CALUDE_special_line_properties_l1944_194493


namespace NUMINAMATH_CALUDE_correlation_coefficient_comparison_l1944_194489

def X : List ℝ := [16, 18, 20, 22]
def Y : List ℝ := [15.10, 12.81, 9.72, 3.21]
def U : List ℝ := [10, 20, 30]
def V : List ℝ := [7.5, 9.5, 16.6]

def r₁ : ℝ := sorry
def r₂ : ℝ := sorry

theorem correlation_coefficient_comparison : r₁ < 0 ∧ 0 < r₂ := by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_comparison_l1944_194489


namespace NUMINAMATH_CALUDE_animal_biscuit_problem_l1944_194411

theorem animal_biscuit_problem :
  ∀ (dogs cats : ℕ),
  dogs + cats = 10 →
  6 * dogs + 5 * cats = 56 →
  dogs = 6 ∧ cats = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_animal_biscuit_problem_l1944_194411


namespace NUMINAMATH_CALUDE_min_value_inequality_l1944_194438

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  (1 / x + 4 / y) ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1944_194438


namespace NUMINAMATH_CALUDE_multiplication_proof_l1944_194477

theorem multiplication_proof : 
  ∃ (a b : ℕ), 
    a * b = 4485 ∧
    a = 23 ∧
    b = 195 ∧
    (b % 10) * a = 115 ∧
    ((b / 10) % 10) * a = 207 ∧
    (b / 100) * a = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_multiplication_proof_l1944_194477


namespace NUMINAMATH_CALUDE_abs_even_and_decreasing_l1944_194478

def f (x : ℝ) := abs x

theorem abs_even_and_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f y ≤ f x) :=
by sorry

end NUMINAMATH_CALUDE_abs_even_and_decreasing_l1944_194478


namespace NUMINAMATH_CALUDE_prob_three_spades_l1944_194417

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- A standard deck of 52 cards with 4 suits and 13 cards per suit -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_suits := 4,
    cards_per_suit := 13,
    h_total := rfl }

/-- The probability of drawing three spades in a row from a standard deck -/
theorem prob_three_spades (d : Deck) (h : d = standard_deck) :
  (d.cards_per_suit : ℚ) / d.total_cards *
  (d.cards_per_suit - 1) / (d.total_cards - 1) *
  (d.cards_per_suit - 2) / (d.total_cards - 2) = 33 / 2550 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_spades_l1944_194417


namespace NUMINAMATH_CALUDE_solution_set_nonempty_l1944_194445

theorem solution_set_nonempty (a : ℝ) : 
  ∃ x : ℝ, a * x^2 - (a - 2) * x - 2 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_nonempty_l1944_194445


namespace NUMINAMATH_CALUDE_unique_solution_square_equation_l1944_194453

theorem unique_solution_square_equation :
  ∃! x : ℝ, (2010 + x)^2 = x^2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_square_equation_l1944_194453


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l1944_194425

/-- Given a parallelogram with area 128 sq m and altitude twice the base, prove the base is 8 m -/
theorem parallelogram_base_length :
  ∀ (base altitude : ℝ),
  base > 0 →
  altitude > 0 →
  altitude = 2 * base →
  base * altitude = 128 →
  base = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l1944_194425


namespace NUMINAMATH_CALUDE_opposite_numbers_l1944_194424

theorem opposite_numbers : -5^2 = -((-5)^2) := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_l1944_194424


namespace NUMINAMATH_CALUDE_raisin_mixture_problem_l1944_194446

theorem raisin_mixture_problem (raisin_cost nut_cost : ℝ) (raisin_weight : ℝ) :
  nut_cost = 3 * raisin_cost →
  raisin_weight * raisin_cost = 0.29411764705882354 * (raisin_weight * raisin_cost + 4 * nut_cost) →
  raisin_weight = 5 := by
sorry

end NUMINAMATH_CALUDE_raisin_mixture_problem_l1944_194446


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1944_194463

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose 12 n = Nat.choose 12 (2*n - 3)) → (n = 3 ∨ n = 5) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1944_194463


namespace NUMINAMATH_CALUDE_one_two_five_th_number_l1944_194418

def digit_sum (n : ℕ) : ℕ := sorry

def nth_number_with_digit_sum_5 (n : ℕ) : ℕ := sorry

theorem one_two_five_th_number : nth_number_with_digit_sum_5 125 = 41000 := by sorry

end NUMINAMATH_CALUDE_one_two_five_th_number_l1944_194418


namespace NUMINAMATH_CALUDE_one_meeting_before_return_l1944_194466

/-- Represents a runner on a rectangular track -/
structure Runner where
  speed : ℝ
  direction : Bool -- True for clockwise, False for counterclockwise

/-- Represents the rectangular track -/
def track_perimeter : ℝ := 140

/-- Calculates the number of meetings between two runners -/
def meetings (runner1 runner2 : Runner) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem one_meeting_before_return (runner1 runner2 : Runner) 
  (h1 : runner1.speed = 6)
  (h2 : runner2.speed = 10)
  (h3 : runner1.direction ≠ runner2.direction) : 
  meetings runner1 runner2 = 1 :=
sorry

end NUMINAMATH_CALUDE_one_meeting_before_return_l1944_194466


namespace NUMINAMATH_CALUDE_correct_quotient_proof_l1944_194490

theorem correct_quotient_proof (D : ℕ) (h1 : D - 1000 = 1200 * 4900) : D / 2100 = 2800 := by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_proof_l1944_194490


namespace NUMINAMATH_CALUDE_parallel_vector_scalar_l1944_194401

/-- Given two 2D vectors a and b, find the scalar m such that m*a + b is parallel to a - 2*b -/
theorem parallel_vector_scalar (a b : ℝ × ℝ) (h1 : a = (2, 3)) (h2 : b = (-1, 2)) :
  ∃ m : ℝ, m * a.1 + b.1 = k * (a.1 - 2 * b.1) ∧ 
           m * a.2 + b.2 = k * (a.2 - 2 * b.2) ∧ 
           m = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vector_scalar_l1944_194401


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1944_194414

theorem complex_modulus_problem (z : ℂ) : 
  ((1 + Complex.I) / (1 - Complex.I)) * z = 3 + 4 * Complex.I → Complex.abs z = 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1944_194414


namespace NUMINAMATH_CALUDE_expected_abs_difference_10_days_l1944_194420

/-- Represents the outcome of a single day --/
inductive DailyOutcome
| CatWins
| FoxWins
| BothLose

/-- Probability distribution for daily outcomes --/
def dailyProbability (outcome : DailyOutcome) : ℝ :=
  match outcome with
  | DailyOutcome.CatWins => 0.25
  | DailyOutcome.FoxWins => 0.25
  | DailyOutcome.BothLose => 0.5

/-- Expected value of the absolute difference in wealth after n days --/
def expectedAbsDifference (n : ℕ) : ℝ :=
  sorry

/-- Theorem stating the expected absolute difference after 10 days is 1 --/
theorem expected_abs_difference_10_days :
  expectedAbsDifference 10 = 1 :=
sorry

end NUMINAMATH_CALUDE_expected_abs_difference_10_days_l1944_194420


namespace NUMINAMATH_CALUDE_scientific_notation_of_57277000_l1944_194459

theorem scientific_notation_of_57277000 :
  (57277000 : ℝ) = 5.7277 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_57277000_l1944_194459


namespace NUMINAMATH_CALUDE_star_op_and_comparison_l1944_194402

-- Define the * operation for non-zero integers
def star_op (a b : ℤ) : ℚ := (a⁻¹ : ℚ) + (b⁻¹ : ℚ)

-- Theorem statement
theorem star_op_and_comparison 
  (a b : ℤ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : a + b = 10) 
  (h4 : a * b = 24) : 
  star_op a b = 5 / 12 ∧ a * b > a + b := by
  sorry

end NUMINAMATH_CALUDE_star_op_and_comparison_l1944_194402


namespace NUMINAMATH_CALUDE_function_properties_l1944_194481

def f (x : ℝ) : ℝ := |2*x + 2| - 5

def g (m : ℝ) (x : ℝ) : ℝ := f x + |x - m|

theorem function_properties (m : ℝ) (h : m > 0) :
  (∀ x, f x - |x - 1| ≥ 0 ↔ x ∈ Set.Iic (-8) ∪ Set.Ici 2) ∧
  (∃ a b c : ℝ, a < b ∧ b < c ∧
    (∀ x, x < a → g m x < 0) ∧
    (∀ x, a < x ∧ x < c → g m x > 0) ∧
    g m a = 0 ∧ g m c = 0) ↔
  3/2 ≤ m ∧ m < 4 :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1944_194481


namespace NUMINAMATH_CALUDE_inequality_solution_l1944_194430

theorem inequality_solution (x : ℝ) : 
  (x - 1) / ((x - 3)^2) < 0 ↔ x < 1 ∧ x ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1944_194430


namespace NUMINAMATH_CALUDE_tangent_semiperimeter_median_inequality_l1944_194423

variable (a b c : ℝ)
variable (s : ℝ)
variable (ta tb tc : ℝ)
variable (ma mb mc : ℝ)

/-- Triangle inequality --/
axiom triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Semi-perimeter definition --/
axiom semi_perimeter : s = (a + b + c) / 2

/-- Tangent line definitions --/
axiom tangent_a : ta = (2 / (b + c)) * Real.sqrt (b * c * s * (s - a))
axiom tangent_b : tb = (2 / (a + c)) * Real.sqrt (a * c * s * (s - b))
axiom tangent_c : tc = (2 / (a + b)) * Real.sqrt (a * b * s * (s - c))

/-- Median line definitions --/
axiom median_a : ma^2 = (2 * b^2 + 2 * c^2 - a^2) / 4
axiom median_b : mb^2 = (2 * a^2 + 2 * c^2 - b^2) / 4
axiom median_c : mc^2 = (2 * a^2 + 2 * b^2 - c^2) / 4

/-- Theorem: Tangent-Semiperimeter-Median Inequality --/
theorem tangent_semiperimeter_median_inequality :
  ta^2 + tb^2 + tc^2 ≤ s^2 ∧ s^2 ≤ ma^2 + mb^2 + mc^2 :=
sorry

end NUMINAMATH_CALUDE_tangent_semiperimeter_median_inequality_l1944_194423


namespace NUMINAMATH_CALUDE_unit_vectors_equality_iff_sum_magnitude_two_l1944_194479

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem unit_vectors_equality_iff_sum_magnitude_two
  (a b : E) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) :
  a = b ↔ ‖a + b‖ = 2 := by sorry

end NUMINAMATH_CALUDE_unit_vectors_equality_iff_sum_magnitude_two_l1944_194479


namespace NUMINAMATH_CALUDE_annulus_area_l1944_194482

/-- An annulus is the region between two concentric circles. -/
structure Annulus where
  b : ℝ
  c : ℝ
  a : ℝ
  h1 : b > c
  h2 : a^2 + c^2 = b^2

/-- The area of an annulus is πa², where a is the length of a line segment
    tangent to the inner circle and extending from the outer circle to the
    point of tangency. -/
theorem annulus_area (ann : Annulus) : Real.pi * ann.a^2 = Real.pi * (ann.b^2 - ann.c^2) := by
  sorry

end NUMINAMATH_CALUDE_annulus_area_l1944_194482


namespace NUMINAMATH_CALUDE_cone_base_circumference_l1944_194416

/-- The circumference of the base of a right circular cone formed by gluing together
    the edges of a 180° sector cut from a circle with radius 6 inches is equal to 6π. -/
theorem cone_base_circumference (r : ℝ) (h : r = 6) : 
  (2 * π * r) / 2 = 6 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l1944_194416


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_inequality_holds_l1944_194437

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 1} = {x : ℝ | x > 1/2} := by sorry

-- Part 2
theorem range_of_a_when_inequality_holds :
  ∀ a : ℝ, (∀ x ∈ Set.Ioo 0 1, f a x > x) ↔ a ∈ Set.Ioc 0 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_inequality_holds_l1944_194437


namespace NUMINAMATH_CALUDE_rabbit_apple_collection_l1944_194448

theorem rabbit_apple_collection (rabbit_apples_per_basket deer_apples_per_basket : ℕ)
  (rabbit_baskets deer_baskets total_apples : ℕ) :
  rabbit_apples_per_basket = 5 →
  deer_apples_per_basket = 6 →
  rabbit_baskets = deer_baskets + 3 →
  rabbit_apples_per_basket * rabbit_baskets = total_apples →
  deer_apples_per_basket * deer_baskets = total_apples →
  rabbit_apples_per_basket * rabbit_baskets = 90 :=
by sorry

end NUMINAMATH_CALUDE_rabbit_apple_collection_l1944_194448


namespace NUMINAMATH_CALUDE_technicians_count_l1944_194447

/-- Represents the workshop scenario with given salary information --/
structure Workshop where
  totalWorkers : ℕ
  avgSalaryAll : ℚ
  avgSalaryTech : ℚ
  avgSalaryNonTech : ℚ

/-- Calculates the number of technicians in the workshop --/
def numTechnicians (w : Workshop) : ℚ :=
  (w.avgSalaryAll * w.totalWorkers - w.avgSalaryNonTech * w.totalWorkers) /
  (w.avgSalaryTech - w.avgSalaryNonTech)

/-- Theorem stating that the number of technicians is 7 --/
theorem technicians_count (w : Workshop) 
  (h1 : w.totalWorkers = 22)
  (h2 : w.avgSalaryAll = 850)
  (h3 : w.avgSalaryTech = 1000)
  (h4 : w.avgSalaryNonTech = 780) :
  numTechnicians w = 7 := by
  sorry

#eval numTechnicians ⟨22, 850, 1000, 780⟩

end NUMINAMATH_CALUDE_technicians_count_l1944_194447


namespace NUMINAMATH_CALUDE_garden_perimeter_l1944_194498

/-- Given a square garden with a pond, if the pond area is 20 square meters
    and the remaining garden area is 124 square meters,
    then the perimeter of the garden is 48 meters. -/
theorem garden_perimeter (s : ℝ) : 
  s > 0 → 
  s^2 = 20 + 124 → 
  4 * s = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l1944_194498


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1944_194470

/-- The remainder when x^4 + 2x^3 is divided by x^2 + 3x + 2 is x^2 + 2x -/
theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 + 2*x^3 = (x^2 + 3*x + 2) * q + (x^2 + 2*x) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1944_194470


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1944_194499

/-- Simple interest calculation -/
def simple_interest (principal time rate : ℝ) : ℝ := principal * time * rate

/-- Theorem: Given the conditions, prove the rate of interest is 0.06 -/
theorem interest_rate_calculation (principal time interest : ℝ) 
  (h_principal : principal = 15000)
  (h_time : time = 3)
  (h_interest : interest = 2700) :
  ∃ rate : ℝ, simple_interest principal time rate = interest ∧ rate = 0.06 := by
  sorry

#check interest_rate_calculation

end NUMINAMATH_CALUDE_interest_rate_calculation_l1944_194499


namespace NUMINAMATH_CALUDE_negative_quadratic_range_l1944_194431

theorem negative_quadratic_range (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ (a > 2 ∨ a < -2) := by sorry

end NUMINAMATH_CALUDE_negative_quadratic_range_l1944_194431


namespace NUMINAMATH_CALUDE_jury_duty_days_is_25_l1944_194460

/-- Calculates the total number of days spent on jury duty -/
def totalJuryDutyDays (
  jurySelectionDays : ℕ)
  (trialMultiplier : ℕ)
  (trialDailyHours : ℕ)
  (deliberationDays : List ℕ)
  (deliberationDailyHours : ℕ) : ℕ :=
  let trialDays := jurySelectionDays * trialMultiplier
  let totalDeliberationHours := deliberationDays.sum * (deliberationDailyHours - 2)
  let totalDeliberationDays := (totalDeliberationHours + deliberationDailyHours - 1) / deliberationDailyHours
  jurySelectionDays + trialDays + totalDeliberationDays

/-- Theorem stating that the total jury duty days is 25 -/
theorem jury_duty_days_is_25 :
  totalJuryDutyDays 2 4 9 [6, 4, 5] 14 = 25 := by
  sorry

#eval totalJuryDutyDays 2 4 9 [6, 4, 5] 14

end NUMINAMATH_CALUDE_jury_duty_days_is_25_l1944_194460


namespace NUMINAMATH_CALUDE_sum_of_doubles_l1944_194494

theorem sum_of_doubles (a b c d e f : ℚ) 
  (h1 : a / b = 2) 
  (h2 : c / d = 2) 
  (h3 : e / f = 2) 
  (h4 : b + d + f = 5) : 
  a + c + e = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_doubles_l1944_194494


namespace NUMINAMATH_CALUDE_three_x_squared_y_squared_l1944_194441

theorem three_x_squared_y_squared (x y : ℤ) 
  (h : y^2 + 3*x^2*y^2 = 30*x^2 + 553) : 
  3*x^2*y^2 = 2886 := by
  sorry

end NUMINAMATH_CALUDE_three_x_squared_y_squared_l1944_194441


namespace NUMINAMATH_CALUDE_negative_expressions_count_l1944_194450

theorem negative_expressions_count : 
  let expressions := [-(-5), -|(-5)|, -(5^2), (-5)^2, 1/(-5)]
  (expressions.filter (λ x => x < 0)).length = 3 := by
sorry

end NUMINAMATH_CALUDE_negative_expressions_count_l1944_194450


namespace NUMINAMATH_CALUDE_min_value_inequality_l1944_194458

theorem min_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : b + c ≥ a) :
  b / c + c / (a + b) ≥ Real.sqrt 2 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1944_194458


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1944_194419

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {y | ∃ x, y = Real.cos x}

theorem intersection_of_M_and_N : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1944_194419


namespace NUMINAMATH_CALUDE_cookie_sales_value_l1944_194440

theorem cookie_sales_value (total_boxes : ℝ) (plain_boxes : ℝ) 
  (choc_chip_price : ℝ) (plain_price : ℝ) 
  (h1 : total_boxes = 1585) 
  (h2 : plain_boxes = 793.125) 
  (h3 : choc_chip_price = 1.25) 
  (h4 : plain_price = 0.75) : 
  (total_boxes - plain_boxes) * choc_chip_price + plain_boxes * plain_price = 1584.6875 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sales_value_l1944_194440


namespace NUMINAMATH_CALUDE_kenya_has_133_peanuts_l1944_194444

-- Define the number of peanuts Jose has
def jose_peanuts : ℕ := 85

-- Define the difference in peanuts between Kenya and Jose
def peanut_difference : ℕ := 48

-- Define Kenya's peanuts in terms of Jose's peanuts and the difference
def kenya_peanuts : ℕ := jose_peanuts + peanut_difference

-- Theorem stating that Kenya has 133 peanuts
theorem kenya_has_133_peanuts : kenya_peanuts = 133 := by
  sorry

end NUMINAMATH_CALUDE_kenya_has_133_peanuts_l1944_194444


namespace NUMINAMATH_CALUDE_difference_number_and_three_fifths_l1944_194410

theorem difference_number_and_three_fifths (n : ℚ) : n = 160 → n - (3 / 5 * n) = 64 := by
  sorry

end NUMINAMATH_CALUDE_difference_number_and_three_fifths_l1944_194410


namespace NUMINAMATH_CALUDE_identical_rows_from_increasing_sums_l1944_194400

theorem identical_rows_from_increasing_sums 
  (n : ℕ) 
  (row1 row2 : Fin n → ℝ) 
  (distinct : ∀ i j, i ≠ j → row1 i ≠ row1 j) 
  (increasing_row1 : ∀ i j, i < j → row1 i < row1 j) 
  (same_elements : ∀ x, ∃ i, row1 i = x ↔ ∃ j, row2 j = x) 
  (increasing_sums : ∀ i j, i < j → row1 i + row2 i < row1 j + row2 j) : 
  ∀ i, row1 i = row2 i :=
sorry

end NUMINAMATH_CALUDE_identical_rows_from_increasing_sums_l1944_194400


namespace NUMINAMATH_CALUDE_complex_square_one_minus_i_l1944_194451

-- Define the complex number i
def i : ℂ := Complex.I

-- Theorem statement
theorem complex_square_one_minus_i : (1 - i)^2 = -2*i := by sorry

end NUMINAMATH_CALUDE_complex_square_one_minus_i_l1944_194451


namespace NUMINAMATH_CALUDE_average_score_is_two_average_score_independent_of_class_size_l1944_194452

/-- Represents the score distribution for a test -/
structure ScoreDistribution where
  threePoints : Real
  twoPoints : Real
  onePoint : Real
  zeroPoints : Real

/-- Calculates the average score given a score distribution -/
def averageScore (dist : ScoreDistribution) : Real :=
  3 * dist.threePoints + 2 * dist.twoPoints + 1 * dist.onePoint + 0 * dist.zeroPoints

/-- Theorem: The average score is 2 for the given score distribution -/
theorem average_score_is_two (dist : ScoreDistribution) 
    (h1 : dist.threePoints = 0.3)
    (h2 : dist.twoPoints = 0.5)
    (h3 : dist.onePoint = 0.1)
    (h4 : dist.zeroPoints = 0.1)
    (h5 : dist.threePoints + dist.twoPoints + dist.onePoint + dist.zeroPoints = 1) :
    averageScore dist = 2 := by
  sorry

/-- Corollary: The average score is independent of the number of students -/
theorem average_score_independent_of_class_size (n : Nat) (dist : ScoreDistribution) 
    (h1 : dist.threePoints = 0.3)
    (h2 : dist.twoPoints = 0.5)
    (h3 : dist.onePoint = 0.1)
    (h4 : dist.zeroPoints = 0.1)
    (h5 : dist.threePoints + dist.twoPoints + dist.onePoint + dist.zeroPoints = 1) :
    averageScore dist = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_two_average_score_independent_of_class_size_l1944_194452


namespace NUMINAMATH_CALUDE_equation_describes_ellipse_and_hyperbola_l1944_194488

/-- A conic section type -/
inductive ConicSection
  | Ellipse
  | Hyperbola
  | Parabola
  | Circle
  | Point
  | Line
  | CrossedLines

/-- Represents the equation y^4 - 8x^4 = 8y^2 - 4 -/
def equation (x y : ℝ) : Prop :=
  y^4 - 8*x^4 = 8*y^2 - 4

/-- The set of conic sections described by the equation -/
def describedConicSections : Set ConicSection :=
  {ConicSection.Ellipse, ConicSection.Hyperbola}

/-- Theorem stating that the equation describes the union of an ellipse and a hyperbola -/
theorem equation_describes_ellipse_and_hyperbola :
  ∀ x y : ℝ, equation x y → 
  ∃ (c : ConicSection), c ∈ describedConicSections :=
sorry

end NUMINAMATH_CALUDE_equation_describes_ellipse_and_hyperbola_l1944_194488


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1944_194406

theorem polynomial_divisibility (n : ℕ) :
  ∃ q : Polynomial ℤ, (X + 1 : Polynomial ℤ)^(2*n + 1) + X^(n + 2) = (X^2 + X + 1) * q := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1944_194406


namespace NUMINAMATH_CALUDE_total_rectangles_is_176_l1944_194492

/-- The number of gray cells in the picture -/
def total_gray_cells : ℕ := 40

/-- The number of blue cells (more frequent gray cells) -/
def blue_cells : ℕ := 36

/-- The number of red cells (less frequent gray cells) -/
def red_cells : ℕ := 4

/-- The number of rectangles containing each blue cell -/
def rectangles_per_blue_cell : ℕ := 4

/-- The number of rectangles containing each red cell -/
def rectangles_per_red_cell : ℕ := 8

/-- The total number of rectangles containing exactly one gray cell -/
def total_rectangles : ℕ := blue_cells * rectangles_per_blue_cell + red_cells * rectangles_per_red_cell

theorem total_rectangles_is_176 : total_rectangles = 176 := by
  sorry

end NUMINAMATH_CALUDE_total_rectangles_is_176_l1944_194492


namespace NUMINAMATH_CALUDE_m_2_sufficient_m_2_not_necessary_m_2_sufficient_but_not_necessary_l1944_194484

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m : ℝ) : Prop :=
  (m - 1) / m = (m - 1) / 2

/-- m = 2 is a sufficient condition for the lines to be parallel -/
theorem m_2_sufficient : are_parallel 2 := by sorry

/-- m = 2 is not a necessary condition for the lines to be parallel -/
theorem m_2_not_necessary : ∃ m : ℝ, m ≠ 2 ∧ are_parallel m := by sorry

/-- m = 2 is a sufficient but not necessary condition for the lines to be parallel -/
theorem m_2_sufficient_but_not_necessary : 
  (are_parallel 2) ∧ (∃ m : ℝ, m ≠ 2 ∧ are_parallel m) := by sorry

end NUMINAMATH_CALUDE_m_2_sufficient_m_2_not_necessary_m_2_sufficient_but_not_necessary_l1944_194484


namespace NUMINAMATH_CALUDE_integral_f_equals_five_sixths_l1944_194404

-- Define the piecewise function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x ≥ 0 ∧ x ≤ 1 then x^2
  else if x > 1 ∧ x ≤ 2 then 2 - x
  else 0  -- Define a value for x outside [0,2] to make f total

-- State the theorem
theorem integral_f_equals_five_sixths :
  ∫ x in (0)..(2), f x = 5/6 := by sorry

end NUMINAMATH_CALUDE_integral_f_equals_five_sixths_l1944_194404


namespace NUMINAMATH_CALUDE_investment_interest_theorem_l1944_194433

/-- Calculates the total interest paid in an 18-month investment contract with specific interest rates and reinvestment -/
def total_interest (initial_investment : ℝ) : ℝ :=
  let interest_6m := initial_investment * 0.02
  let balance_10m := initial_investment + interest_6m
  let interest_10m := balance_10m * 0.03
  let balance_18m := balance_10m + interest_10m
  let interest_18m := balance_18m * 0.04
  interest_6m + interest_10m + interest_18m

/-- Theorem stating that the total interest paid in the given investment scenario is $926.24 -/
theorem investment_interest_theorem :
  total_interest 10000 = 926.24 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_theorem_l1944_194433


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1944_194462

theorem max_value_of_expression (x : ℝ) :
  x^4 / (x^8 + 4*x^6 - 8*x^4 + 16*x^2 + 64) ≤ 1/24 ∧
  ∃ y : ℝ, y^4 / (y^8 + 4*y^6 - 8*y^4 + 16*y^2 + 64) = 1/24 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1944_194462


namespace NUMINAMATH_CALUDE_dog_tricks_conversion_l1944_194412

/-- Converts a number from base 9 to base 10 -/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

theorem dog_tricks_conversion :
  base9ToBase10 [1, 2, 5] = 424 := by
  sorry

end NUMINAMATH_CALUDE_dog_tricks_conversion_l1944_194412


namespace NUMINAMATH_CALUDE_star_five_three_l1944_194496

def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

theorem star_five_three : star 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_star_five_three_l1944_194496


namespace NUMINAMATH_CALUDE_no_solution_exists_l1944_194473

theorem no_solution_exists (k m : ℕ) : k.factorial + 48 ≠ 48 * (k + 1) ^ m := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1944_194473


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l1944_194465

/-- Parabola y² = x -/
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = p.1

/-- Point lies on x-axis -/
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

/-- Line passing through two points -/
def line_through (p q : ℝ × ℝ) (r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r = (p.1 + t * (q.1 - p.1), p.2 + t * (q.2 - p.2))

theorem parabola_intersection_theorem 
  (O : ℝ × ℝ)  -- Origin
  (P S T : ℝ × ℝ)  -- Points on x-axis
  (A₁ B₁ A₂ B₂ : ℝ × ℝ)  -- Points on parabola
  (h_O : O = (0, 0))
  (h_P : on_x_axis P)
  (h_S : on_x_axis S)
  (h_T : on_x_axis T)
  (h_A₁ : parabola A₁)
  (h_B₁ : parabola B₁)
  (h_A₂ : parabola A₂)
  (h_B₂ : parabola B₂)
  (h_line₁ : line_through A₁ B₁ P)
  (h_line₂ : line_through A₂ B₂ P)
  (h_line₃ : line_through A₁ B₂ S)
  (h_line₄ : line_through A₂ B₁ T) :
  (S.1 - O.1) * (T.1 - O.1) = (P.1 - O.1)^2 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l1944_194465


namespace NUMINAMATH_CALUDE_boys_playing_cards_l1944_194439

/-- Given a class where some boys love to play marbles and some love to play cards,
    prove that the number of boys who love to play cards is 0 under certain conditions. -/
theorem boys_playing_cards (total_marbles : ℕ) (marbles_per_boy : ℕ) (boys_playing_marbles : ℕ) :
  total_marbles = 26 →
  marbles_per_boy = 2 →
  boys_playing_marbles = 13 →
  boys_playing_marbles * marbles_per_boy = total_marbles →
  0 = total_marbles / marbles_per_boy - boys_playing_marbles :=
by
  sorry

end NUMINAMATH_CALUDE_boys_playing_cards_l1944_194439


namespace NUMINAMATH_CALUDE_six_arts_competition_l1944_194407

theorem six_arts_competition (a b c : ℕ) (h_abc : a > b ∧ b > c) :
  (∃ (x y z : ℕ),
    x + y + z = 6 ∧
    a * x + b * y + c * z = 26 ∧
    (∃ (p q r : ℕ),
      p + q + r = 6 ∧
      a * p + b * q + c * r = 11 ∧
      p = 1 ∧
      (∃ (u v w : ℕ),
        u + v + w = 6 ∧
        a * u + b * v + c * w = 11 ∧
        a + b + c = 8))) →
  (∃ (p q r : ℕ),
    p + q + r = 6 ∧
    a * p + b * q + c * r = 11 ∧
    p = 1 ∧
    r = 4) :=
by sorry

end NUMINAMATH_CALUDE_six_arts_competition_l1944_194407


namespace NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l1944_194435

theorem consecutive_integers_product_812_sum_57 :
  ∀ x y : ℕ+, 
    x.val + 1 = y.val →
    x * y = 812 →
    x + y = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l1944_194435


namespace NUMINAMATH_CALUDE_derivative_of_f_l1944_194442

/-- Given a function f(x) = (x^2 + 2x - 1)e^(2-x), this theorem states its derivative. -/
theorem derivative_of_f (x : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (x^2 + 2*x - 1) * Real.exp (2 - x)
  deriv f x = (3 - x^2) * Real.exp (2 - x) := by
sorry

end NUMINAMATH_CALUDE_derivative_of_f_l1944_194442


namespace NUMINAMATH_CALUDE_science_problem_time_l1944_194432

/-- Calculates the time taken for each science problem given the number of problems and time constraints. -/
theorem science_problem_time 
  (math_problems : ℕ) 
  (social_studies_problems : ℕ) 
  (science_problems : ℕ) 
  (math_time_per_problem : ℚ) 
  (social_studies_time_per_problem : ℚ) 
  (total_time : ℚ) 
  (h1 : math_problems = 15)
  (h2 : social_studies_problems = 6)
  (h3 : science_problems = 10)
  (h4 : math_time_per_problem = 2)
  (h5 : social_studies_time_per_problem = 1/2)
  (h6 : total_time = 48) :
  (total_time - (math_problems * math_time_per_problem + social_studies_problems * social_studies_time_per_problem)) / science_problems = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_science_problem_time_l1944_194432


namespace NUMINAMATH_CALUDE_weighted_die_odd_sum_probability_l1944_194497

/-- Represents a six-sided die where even numbers are twice as likely as odd numbers -/
structure WeightedDie :=
  (prob_odd : ℝ)
  (prob_even : ℝ)
  (h1 : prob_odd > 0)
  (h2 : prob_even > 0)
  (h3 : prob_even = 2 * prob_odd)
  (h4 : 3 * prob_odd + 3 * prob_even = 1)

/-- The probability of rolling an odd sum with two rolls of the weighted die -/
def prob_odd_sum (d : WeightedDie) : ℝ :=
  2 * d.prob_odd * d.prob_even

theorem weighted_die_odd_sum_probability (d : WeightedDie) :
  prob_odd_sum d = 1 / 9 := by
  sorry

#check weighted_die_odd_sum_probability

end NUMINAMATH_CALUDE_weighted_die_odd_sum_probability_l1944_194497


namespace NUMINAMATH_CALUDE_cylinder_height_comparison_l1944_194403

/-- Theorem: Comparing cylinder heights with equal volumes and different radii -/
theorem cylinder_height_comparison (r₁ h₁ r₂ h₂ : ℝ) 
  (volume_eq : r₁ ^ 2 * h₁ = r₂ ^ 2 * h₂)
  (radius_relation : r₂ = 1.2 * r₁) :
  h₁ = 1.44 * h₂ :=
sorry

end NUMINAMATH_CALUDE_cylinder_height_comparison_l1944_194403


namespace NUMINAMATH_CALUDE_intersection_M_N_l1944_194476

def M : Set ℝ := {x : ℝ | |x + 1| < 3}
def N : Set ℝ := {0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1944_194476


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l1944_194426

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^13 + i^113 = 2*i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l1944_194426


namespace NUMINAMATH_CALUDE_sphere_volume_increase_l1944_194469

theorem sphere_volume_increase (r : ℝ) (h : r > 0) : 
  (4 / 3 * Real.pi * (2 * r)^3) / (4 / 3 * Real.pi * r^3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_increase_l1944_194469


namespace NUMINAMATH_CALUDE_trapezoid_bisector_length_l1944_194443

/-- 
Given a trapezoid with parallel sides of length a and c,
the length of a segment parallel to these sides that bisects the trapezoid's area
is √((a² + c²) / 2).
-/
theorem trapezoid_bisector_length (a c : ℝ) (ha : a > 0) (hc : c > 0) :
  ∃ x : ℝ, x > 0 ∧ x^2 = (a^2 + c^2) / 2 ∧
  (∀ m : ℝ, m > 0 → (a + c) * m / 2 = (x + c) * (2 * m / (c + x)) / 2 + (x + a) * (2 * m / (a + x)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_bisector_length_l1944_194443


namespace NUMINAMATH_CALUDE_paulas_aunt_money_l1944_194475

/-- The amount of money Paula's aunt gave her -/
def aunt_money (shirt_price : ℕ) (num_shirts : ℕ) (pants_price : ℕ) (money_left : ℕ) : ℕ :=
  shirt_price * num_shirts + pants_price + money_left

/-- Theorem stating the total amount of money Paula's aunt gave her -/
theorem paulas_aunt_money :
  aunt_money 11 2 13 74 = 109 := by
  sorry

end NUMINAMATH_CALUDE_paulas_aunt_money_l1944_194475


namespace NUMINAMATH_CALUDE_system_solution_l1944_194486

theorem system_solution (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃! (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    x + y + z = a + b + c ∧
    4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = a * b * c ∧
    x = (b + c) / 2 ∧ y = (c + a) / 2 ∧ z = (a + b) / 2 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1944_194486


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1944_194427

theorem absolute_value_equation_solution (x y : ℝ) :
  |x - Real.log (y^2)| = x + Real.log (y^2) →
  x = 0 ∧ (y = 1 ∨ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1944_194427


namespace NUMINAMATH_CALUDE_pascal_contest_certificates_l1944_194464

theorem pascal_contest_certificates 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (percent_boys_cert : ℚ) 
  (percent_girls_cert : ℚ) 
  (h1 : num_boys = 30)
  (h2 : num_girls = 20)
  (h3 : percent_boys_cert = 30 / 100)
  (h4 : percent_girls_cert = 40 / 100) :
  (num_boys * percent_boys_cert + num_girls * percent_girls_cert) / (num_boys + num_girls) = 34 / 100 := by
sorry


end NUMINAMATH_CALUDE_pascal_contest_certificates_l1944_194464


namespace NUMINAMATH_CALUDE_y_days_to_finish_work_l1944_194457

/-- The number of days x needs to finish the work alone -/
def x_days : ℕ := 36

/-- The number of days y worked before leaving -/
def y_worked : ℕ := 12

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℕ := 18

/-- The work rate of x (portion of work completed per day) -/
def x_rate : ℚ := 1 / x_days

/-- The total amount of work to be done -/
def total_work : ℚ := 1

/-- The amount of work completed by x after y left -/
def x_completed : ℚ := x_rate * x_remaining

theorem y_days_to_finish_work : ℕ := by
  sorry

end NUMINAMATH_CALUDE_y_days_to_finish_work_l1944_194457


namespace NUMINAMATH_CALUDE_smallest_multiple_l1944_194405

theorem smallest_multiple (n : ℕ) : 
  (∃ k : ℕ, n = 32 * k) ∧ 
  (∃ m : ℕ, n - 6 = 97 * m) ∧
  (∀ x : ℕ, x < n → ¬((∃ k : ℕ, x = 32 * k) ∧ (∃ m : ℕ, x - 6 = 97 * m))) →
  n = 2528 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_l1944_194405


namespace NUMINAMATH_CALUDE_cousins_distribution_eq_52_l1944_194485

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 cousins into 4 rooms -/
def cousins_distribution : ℕ := distribute 5 4

theorem cousins_distribution_eq_52 : cousins_distribution = 52 := by sorry

end NUMINAMATH_CALUDE_cousins_distribution_eq_52_l1944_194485


namespace NUMINAMATH_CALUDE_smallest_three_digit_pq2r_l1944_194413

theorem smallest_three_digit_pq2r : ∃ (p q r : ℕ), 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  126 = p * q^2 * r ∧
  (∀ (x p' q' r' : ℕ), 
    100 ≤ x ∧ x < 126 →
    Nat.Prime p' → Nat.Prime q' → Nat.Prime r' →
    p' ≠ q' → p' ≠ r' → q' ≠ r' →
    x ≠ p' * q'^2 * r') :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_pq2r_l1944_194413


namespace NUMINAMATH_CALUDE_als_original_portion_l1944_194454

theorem als_original_portion (al betty clare : ℕ) : 
  al + betty + clare = 1200 →
  al ≠ betty →
  al ≠ clare →
  betty ≠ clare →
  al - 150 + 3 * betty + 3 * clare = 1800 →
  al = 825 := by
sorry

end NUMINAMATH_CALUDE_als_original_portion_l1944_194454


namespace NUMINAMATH_CALUDE_firefighter_ratio_l1944_194467

theorem firefighter_ratio (doug kai eli : ℕ) : 
  doug = 20 →
  eli = kai / 2 →
  doug + kai + eli = 110 →
  kai / doug = 3 := by
sorry

end NUMINAMATH_CALUDE_firefighter_ratio_l1944_194467


namespace NUMINAMATH_CALUDE_marias_trip_distance_l1944_194474

theorem marias_trip_distance :
  ∀ (D : ℝ),
  (D / 2) / 4 + 180 = D / 2 →
  D = 480 :=
by
  sorry

end NUMINAMATH_CALUDE_marias_trip_distance_l1944_194474


namespace NUMINAMATH_CALUDE_trig_equation_proof_l1944_194409

theorem trig_equation_proof (x y : Real) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 4) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_equation_proof_l1944_194409


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1944_194429

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (2 * x + 15) = 12) ∧ (x = 64.5) := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1944_194429


namespace NUMINAMATH_CALUDE_job_completion_multiple_l1944_194471

/-- Given workers A and B, and their work rates, calculate the multiple of the original job they complete when working together for a given number of days. -/
theorem job_completion_multiple 
  (days_A : ℝ) 
  (days_B : ℝ) 
  (work_days : ℝ) 
  (h1 : days_A > 0) 
  (h2 : days_B > 0) 
  (h3 : work_days > 0) : 
  work_days * (1 / days_A + 1 / days_B) = 4 := by
  sorry

#check job_completion_multiple 45 30 72

end NUMINAMATH_CALUDE_job_completion_multiple_l1944_194471
