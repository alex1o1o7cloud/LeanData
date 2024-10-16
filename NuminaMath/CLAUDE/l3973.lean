import Mathlib

namespace NUMINAMATH_CALUDE_no_four_integers_product_plus_2002_square_l3973_397391

theorem no_four_integers_product_plus_2002_square : 
  ¬ ∃ (n₁ n₂ n₃ n₄ : ℕ+), 
    (∀ (i j : Fin 4), i ≠ j → ∃ (m : ℕ), (n₁ :: n₂ :: n₃ :: n₄ :: []).get i * (n₁ :: n₂ :: n₃ :: n₄ :: []).get j + 2002 = m^2) :=
by sorry

end NUMINAMATH_CALUDE_no_four_integers_product_plus_2002_square_l3973_397391


namespace NUMINAMATH_CALUDE_m_range_l3973_397331

def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x * (x - 1) > 6}
def C (m : ℝ) : Set ℝ := {x | -1 + m < x ∧ x < 2 * m}

theorem m_range (m : ℝ) : 
  (C m).Nonempty ∧ C m ⊆ (A ∩ (Set.univ \ B)) → -1 < m ∧ m ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l3973_397331


namespace NUMINAMATH_CALUDE_circle_equation_from_parabola_intersection_l3973_397353

/-- Parabola type -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Circle type -/
structure Circle where
  center : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- Theorem: Given a parabola and a circle with specific properties, 
    prove the equation of the circle -/
theorem circle_equation_from_parabola_intersection 
  (C₁ : Parabola) 
  (C₂ : Circle) 
  (F : ℝ × ℝ) 
  (A B C D : ℝ × ℝ) :
  C₁.equation = fun x y ↦ x^2 = 2*y →  -- Parabola equation
  C₂.center = F →                      -- Circle center at focus
  C₂.equation A.1 A.2 →                -- Circle intersects parabola at A
  C₂.equation B.1 B.2 →                -- Circle intersects parabola at B
  C₂.equation C.1 C.2 →                -- Circle intersects directrix at C
  C₂.equation D.1 D.2 →                -- Circle intersects directrix at D
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - D.1)^2 + (B.2 - D.2)^2 →  -- ABCD is rectangle
  C₂.equation = fun x y ↦ x^2 + (y - 1/2)^2 = 4 :=  -- Conclusion: Circle equation
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_parabola_intersection_l3973_397353


namespace NUMINAMATH_CALUDE_normal_symmetric_prob_l3973_397329

/-- A random variable following a normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- Probability of a normal random variable being less than a given value -/
noncomputable def prob_less (ξ : NormalRV) (x : ℝ) : ℝ := sorry

/-- Probability of a normal random variable being greater than a given value -/
noncomputable def prob_greater (ξ : NormalRV) (x : ℝ) : ℝ := sorry

/-- Theorem: If P(ξ < 1) = P(ξ > 3) for a normal random variable ξ, then P(ξ > 2) = 1/2 -/
theorem normal_symmetric_prob (ξ : NormalRV) 
  (h : prob_less ξ 1 = prob_greater ξ 3) : prob_greater ξ 2 = 1/2 := by sorry

end NUMINAMATH_CALUDE_normal_symmetric_prob_l3973_397329


namespace NUMINAMATH_CALUDE_cyclists_meet_time_l3973_397332

/-- Two cyclists on a circular track meet at the starting point -/
theorem cyclists_meet_time (v1 v2 circumference : ℝ) (h1 : v1 = 7) (h2 : v2 = 8) (h3 : circumference = 300) : 
  (circumference / (v1 + v2) = 20) := by
  sorry

end NUMINAMATH_CALUDE_cyclists_meet_time_l3973_397332


namespace NUMINAMATH_CALUDE_prank_combinations_l3973_397364

theorem prank_combinations (choices : List Nat) : 
  choices = [2, 3, 0, 6, 1] → List.prod choices = 0 := by
  sorry

end NUMINAMATH_CALUDE_prank_combinations_l3973_397364


namespace NUMINAMATH_CALUDE_set_membership_equivalence_l3973_397313

theorem set_membership_equivalence (C M N : Set α) (x : α) :
  x ∈ C ∪ (M ∩ N) ↔ (x ∈ C ∪ M ∨ x ∈ C ∪ N) := by sorry

end NUMINAMATH_CALUDE_set_membership_equivalence_l3973_397313


namespace NUMINAMATH_CALUDE_b_200_equals_179101_l3973_397314

/-- Sequence a_n defined as n(n+1)/2 -/
def a (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Predicate to check if a number is not divisible by 3 -/
def not_divisible_by_three (n : ℕ) : Prop := n % 3 ≠ 0

/-- Sequence b_n derived from a_n by removing terms divisible by 3 and rearranging -/
def b (n : ℕ) : ℕ := a (3 * n - 2)

/-- Theorem stating that the 200th term of sequence b_n is 179101 -/
theorem b_200_equals_179101 : b 200 = 179101 := by sorry

end NUMINAMATH_CALUDE_b_200_equals_179101_l3973_397314


namespace NUMINAMATH_CALUDE_perpendicular_vectors_implies_m_equals_three_l3973_397381

/-- Given vectors a and b in ℝ², if a is perpendicular to (a - b), then the second component of b is -1 and m = 3. -/
theorem perpendicular_vectors_implies_m_equals_three (a b : ℝ × ℝ) (m : ℝ) 
    (h1 : a = (2, 1))
    (h2 : b = (m, -1))
    (h3 : a • (a - b) = 0) :
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_implies_m_equals_three_l3973_397381


namespace NUMINAMATH_CALUDE_al2co3_3_weight_l3973_397346

/-- The molecular weight of a compound given its composition and atomic weights -/
def molecular_weight (al_weight c_weight o_weight : ℝ) (num_moles : ℝ) : ℝ :=
  let co3_weight := c_weight + 3 * o_weight
  let al2co3_3_weight := 2 * al_weight + 3 * co3_weight
  num_moles * al2co3_3_weight

/-- Theorem stating the molecular weight of 6 moles of Al2(CO3)3 -/
theorem al2co3_3_weight : 
  molecular_weight 26.98 12.01 16.00 6 = 1403.94 := by
  sorry

end NUMINAMATH_CALUDE_al2co3_3_weight_l3973_397346


namespace NUMINAMATH_CALUDE_flower_purchase_solution_l3973_397300

/-- Represents the flower purchase problem -/
structure FlowerPurchase where
  priceA : ℕ  -- Price of type A flower
  priceB : ℕ  -- Price of type B flower
  totalPlants : ℕ  -- Total number of plants to purchase
  (first_purchase : 30 * priceA + 15 * priceB = 675)
  (second_purchase : 12 * priceA + 5 * priceB = 265)
  (total_constraint : totalPlants = 31)
  (type_b_constraint : ∀ m : ℕ, m ≤ totalPlants → totalPlants - m < 2 * m)

/-- Theorem stating the solution to the flower purchase problem -/
theorem flower_purchase_solution (fp : FlowerPurchase) :
  fp.priceA = 20 ∧ fp.priceB = 5 ∧
  ∃ (m : ℕ), m = 11 ∧ fp.totalPlants - m = 20 ∧
  20 * m + 5 * (fp.totalPlants - m) = 320 ∧
  ∀ (n : ℕ), n ≤ fp.totalPlants → 
    20 * n + 5 * (fp.totalPlants - n) ≥ 20 * m + 5 * (fp.totalPlants - m) :=
by sorry


end NUMINAMATH_CALUDE_flower_purchase_solution_l3973_397300


namespace NUMINAMATH_CALUDE_luke_sticker_count_l3973_397327

/-- Represents the number of stickers Luke has at different stages -/
structure StickerCount where
  initial : ℕ
  afterBuying : ℕ
  afterGift : ℕ
  afterGiving : ℕ
  afterUsing : ℕ
  final : ℕ

/-- Theorem stating the relationship between Luke's initial and final sticker counts -/
theorem luke_sticker_count (s : StickerCount) 
  (hbuy : s.afterBuying = s.initial + 12)
  (hgift : s.afterGift = s.afterBuying + 20)
  (hgive : s.afterGiving = s.afterGift - 5)
  (huse : s.afterUsing = s.afterGiving - 8)
  (hfinal : s.final = s.afterUsing)
  (h_final_count : s.final = 39) :
  s.initial = 20 := by
  sorry

end NUMINAMATH_CALUDE_luke_sticker_count_l3973_397327


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3973_397374

theorem system_of_equations_solution :
  ∃! (x y : ℝ), 4*x + 3*y = 6.4 ∧ 5*x - 6*y = -1.5 ∧ x = 11.3/13 ∧ y = 2.9232/3 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3973_397374


namespace NUMINAMATH_CALUDE_fraction_property_l3973_397350

theorem fraction_property (a : ℕ) (h : a > 1) :
  let b := 2 * a - 1
  0 < a ∧ a < b ∧ (a - 1) / (b - 1) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_property_l3973_397350


namespace NUMINAMATH_CALUDE_square_ratio_theorem_l3973_397388

theorem square_ratio_theorem : 
  ∃ (a b c : ℕ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (75 : ℚ) / 27 = (a * (b.sqrt : ℚ) / c) ^ 2 ∧
    a + b + c = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_theorem_l3973_397388


namespace NUMINAMATH_CALUDE_probability_multiple_of_three_l3973_397397

/-- A fair cubic die with 6 faces numbered 1 to 6 -/
def FairDie : Finset ℕ := Finset.range 6

/-- The set of outcomes that are multiples of 3 -/
def MultiplesOfThree : Finset ℕ := Finset.filter (fun n => n % 3 = 0) FairDie

/-- The probability of an event in a finite sample space -/
def probability (event : Finset ℕ) (sampleSpace : Finset ℕ) : ℚ :=
  (event.card : ℚ) / (sampleSpace.card : ℚ)

theorem probability_multiple_of_three : 
  probability MultiplesOfThree FairDie = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_multiple_of_three_l3973_397397


namespace NUMINAMATH_CALUDE_cubic_function_derivative_l3973_397360

/-- Given a cubic function f(x) = ax³ + bx² + 3 where b = f'(2), 
    if f'(1) = -5, then f'(2) = -4 -/
theorem cubic_function_derivative (a b : ℝ) : 
  let f := fun x : ℝ => a * x^3 + b * x^2 + 3
  let f' := fun x : ℝ => 3 * a * x^2 + 2 * b * x
  (f' 1 = -5 ∧ b = f' 2) → f' 2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_derivative_l3973_397360


namespace NUMINAMATH_CALUDE_circle_radius_l3973_397370

theorem circle_radius (C : ℝ) (h : C = 72 * Real.pi) : C / (2 * Real.pi) = 36 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3973_397370


namespace NUMINAMATH_CALUDE_triangle_circumradius_l3973_397337

theorem triangle_circumradius (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) :
  let s := (a + b + c) / 2
  (a * b * c) / (4 * Real.sqrt (s * (s - a) * (s - b) * (s - c))) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumradius_l3973_397337


namespace NUMINAMATH_CALUDE_smallest_constant_D_l3973_397362

theorem smallest_constant_D :
  ∃ (D : ℝ), D = Real.sqrt (8 / 17) ∧
  (∀ (x y : ℝ), x^2 + 2*y^2 + 5 ≥ D*(2*x + 3*y) + 4) ∧
  (∀ (D' : ℝ), (∀ (x y : ℝ), x^2 + 2*y^2 + 5 ≥ D'*(2*x + 3*y) + 4) → D' ≥ D) :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_D_l3973_397362


namespace NUMINAMATH_CALUDE_y_divisibility_l3973_397341

def y : ℕ := 81 + 243 + 729 + 1458 + 2187 + 6561 + 19683

theorem y_divisibility :
  (∃ k : ℕ, y = 3 * k) ∧
  (∃ k : ℕ, y = 9 * k) ∧
  (∃ k : ℕ, y = 27 * k) ∧
  (∃ k : ℕ, y = 81 * k) :=
by sorry

end NUMINAMATH_CALUDE_y_divisibility_l3973_397341


namespace NUMINAMATH_CALUDE_weeks_per_month_l3973_397377

theorem weeks_per_month (months : ℕ) (weekly_rate : ℚ) (monthly_rate : ℚ) (savings : ℚ) :
  months = 3 ∧ 
  weekly_rate = 280 ∧ 
  monthly_rate = 1000 ∧
  savings = 360 →
  (months * monthly_rate + savings) / (months * weekly_rate) = 4 := by
sorry

end NUMINAMATH_CALUDE_weeks_per_month_l3973_397377


namespace NUMINAMATH_CALUDE_emma_widget_production_difference_l3973_397305

/-- 
Given Emma's widget production rates and working hours on Monday and Tuesday, 
prove the difference in total widgets produced.
-/
theorem emma_widget_production_difference 
  (w t : ℕ) 
  (h1 : w = 3 * t) : 
  w * t - (w + 5) * (t - 3) = 4 * t + 15 := by
  sorry

end NUMINAMATH_CALUDE_emma_widget_production_difference_l3973_397305


namespace NUMINAMATH_CALUDE_min_value_constraint_l3973_397399

theorem min_value_constraint (a b : ℝ) (h : a + b^2 = 2) :
  ∃ (m : ℝ), m = 3 ∧ ∀ (x y : ℝ), x + y^2 = 2 → a^2 + 6*b^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_constraint_l3973_397399


namespace NUMINAMATH_CALUDE_slope_problem_l3973_397312

theorem slope_problem (m : ℝ) (h1 : m > 0) 
  (h2 : (m - 4) / (2 - m) = m) : m = (1 + Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_problem_l3973_397312


namespace NUMINAMATH_CALUDE_part_a_part_b_l3973_397324

-- Define a type for our set of 100 positive numbers
def PositiveSet := Fin 100 → ℝ

-- Define the property that all numbers in the set are positive
def AllPositive (s : PositiveSet) : Prop :=
  ∀ i, s i > 0

-- Define the property that the sum of any 7 numbers is less than 7
def SumOfSevenLessThanSeven (s : PositiveSet) : Prop :=
  ∀ (i₁ i₂ i₃ i₄ i₅ i₆ i₇ : Fin 100),
    i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₁ ≠ i₄ ∧ i₁ ≠ i₅ ∧ i₁ ≠ i₆ ∧ i₁ ≠ i₇ ∧
    i₂ ≠ i₃ ∧ i₂ ≠ i₄ ∧ i₂ ≠ i₅ ∧ i₂ ≠ i₆ ∧ i₂ ≠ i₇ ∧
    i₃ ≠ i₄ ∧ i₃ ≠ i₅ ∧ i₃ ≠ i₆ ∧ i₃ ≠ i₇ ∧
    i₄ ≠ i₅ ∧ i₄ ≠ i₆ ∧ i₄ ≠ i₇ ∧
    i₅ ≠ i₆ ∧ i₅ ≠ i₇ ∧
    i₆ ≠ i₇ →
    s i₁ + s i₂ + s i₃ + s i₄ + s i₅ + s i₆ + s i₇ < 7

-- Define the property that the sum of any 10 numbers is less than 10
def SumOfTenLessThanTen (s : PositiveSet) : Prop :=
  ∀ (i₁ i₂ i₃ i₄ i₅ i₆ i₇ i₈ i₉ i₁₀ : Fin 100),
    i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₁ ≠ i₄ ∧ i₁ ≠ i₅ ∧ i₁ ≠ i₆ ∧ i₁ ≠ i₇ ∧ i₁ ≠ i₈ ∧ i₁ ≠ i₉ ∧ i₁ ≠ i₁₀ ∧
    i₂ ≠ i₃ ∧ i₂ ≠ i₄ ∧ i₂ ≠ i₅ ∧ i₂ ≠ i₆ ∧ i₂ ≠ i₇ ∧ i₂ ≠ i₈ ∧ i₂ ≠ i₉ ∧ i₂ ≠ i₁₀ ∧
    i₃ ≠ i₄ ∧ i₃ ≠ i₅ ∧ i₃ ≠ i₆ ∧ i₃ ≠ i₇ ∧ i₃ ≠ i₈ ∧ i₃ ≠ i₉ ∧ i₃ ≠ i₁₀ ∧
    i₄ ≠ i₅ ∧ i₄ ≠ i₆ ∧ i₄ ≠ i₇ ∧ i₄ ≠ i₈ ∧ i₄ ≠ i₉ ∧ i₄ ≠ i₁₀ ∧
    i₅ ≠ i₆ ∧ i₅ ≠ i₇ ∧ i₅ ≠ i₈ ∧ i₅ ≠ i₉ ∧ i₅ ≠ i₁₀ ∧
    i₆ ≠ i₇ ∧ i₆ ≠ i₈ ∧ i₆ ≠ i₉ ∧ i₆ ≠ i₁₀ ∧
    i₇ ≠ i₈ ∧ i₇ ≠ i₉ ∧ i₇ ≠ i₁₀ ∧
    i₈ ≠ i₉ ∧ i₈ ≠ i₁₀ ∧
    i₉ ≠ i₁₀ →
    s i₁ + s i₂ + s i₃ + s i₄ + s i₅ + s i₆ + s i₇ + s i₈ + s i₉ + s i₁₀ < 10

-- Theorem for part (a)
theorem part_a (s : PositiveSet) (h₁ : AllPositive s) (h₂ : SumOfSevenLessThanSeven s) :
  SumOfTenLessThanTen s := by
  sorry

-- Theorem for part (b)
theorem part_b :
  ¬∀ (s : PositiveSet), AllPositive s → SumOfTenLessThanTen s → SumOfSevenLessThanSeven s := by
  sorry

end NUMINAMATH_CALUDE_part_a_part_b_l3973_397324


namespace NUMINAMATH_CALUDE_barry_dime_value_l3973_397304

def dime_value : ℕ := 10

theorem barry_dime_value (dan_dimes : ℕ) (barry_dimes : ℕ) : 
  dan_dimes = 52 ∧ 
  dan_dimes = barry_dimes / 2 + 2 →
  barry_dimes * dime_value = 1000 := by
sorry

end NUMINAMATH_CALUDE_barry_dime_value_l3973_397304


namespace NUMINAMATH_CALUDE_inequality_proof_l3973_397343

theorem inequality_proof (a b c : ℝ) (θ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * Real.cos θ ^ 2 + b * Real.sin θ ^ 2 < c) :
  Real.sqrt a * Real.cos θ ^ 2 + Real.sqrt b * Real.sin θ ^ 2 < Real.sqrt c := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3973_397343


namespace NUMINAMATH_CALUDE_a_equals_b_l3973_397307

theorem a_equals_b (a b : ℝ) : 
  a = Real.sqrt 5 + Real.sqrt 6 → 
  b = 1 / (Real.sqrt 6 - Real.sqrt 5) → 
  a = b := by sorry

end NUMINAMATH_CALUDE_a_equals_b_l3973_397307


namespace NUMINAMATH_CALUDE_emily_songs_l3973_397384

theorem emily_songs (x : ℕ) : x + 7 = 13 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_emily_songs_l3973_397384


namespace NUMINAMATH_CALUDE_pavilion_pillar_height_l3973_397390

-- Define the regular octagon
structure RegularOctagon where
  side_length : ℝ
  center : ℝ × ℝ

-- Define a pillar
structure Pillar where
  base : ℝ × ℝ
  height : ℝ

-- Define the pavilion
structure Pavilion where
  octagon : RegularOctagon
  pillars : Fin 8 → Pillar

-- Define the theorem
theorem pavilion_pillar_height 
  (pav : Pavilion) 
  (h_a : (pav.pillars 0).height = 15)
  (h_b : (pav.pillars 1).height = 11)
  (h_c : (pav.pillars 2).height = 13) :
  (pav.pillars 5).height = 32 :=
sorry

end NUMINAMATH_CALUDE_pavilion_pillar_height_l3973_397390


namespace NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l3973_397328

theorem shaded_fraction_of_rectangle : ∀ (length width : ℕ) (shaded_fraction : ℚ),
  length = 15 →
  width = 20 →
  shaded_fraction = 1/4 →
  (shaded_fraction * (1/2 : ℚ)) * (length * width : ℚ) = (1/8 : ℚ) * (length * width : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l3973_397328


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3973_397359

theorem inequality_solution_set (k : ℝ) : 
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 3) ↔ |k * x - 4| ≤ 2) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3973_397359


namespace NUMINAMATH_CALUDE_f_nonnegative_range_l3973_397371

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (Real.exp x - a) - a^2 * x

theorem f_nonnegative_range (a : ℝ) :
  (∀ x, f a x ≥ 0) ↔ a ∈ Set.Icc (-2 * Real.exp (3/4)) 1 :=
by sorry

end NUMINAMATH_CALUDE_f_nonnegative_range_l3973_397371


namespace NUMINAMATH_CALUDE_random_selection_probability_l3973_397335

theorem random_selection_probability (a : ℝ) : a > 0 → (∃ m : ℝ, 0 ≤ m ∧ m ≤ a) → (2 / a = 1 / 3) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_random_selection_probability_l3973_397335


namespace NUMINAMATH_CALUDE_division_problem_l3973_397309

theorem division_problem (dividend : ℕ) (remainder : ℕ) (quotient : ℕ) (divisor : ℕ) (n : ℕ) :
  dividend = 251 →
  remainder = 8 →
  divisor = 3 * quotient →
  divisor = 3 * remainder + n →
  dividend = divisor * quotient + remainder →
  n = 3 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l3973_397309


namespace NUMINAMATH_CALUDE_river_bank_bottom_width_l3973_397354

/-- Given a trapezium-shaped cross-section of a river bank, prove that the bottom width is 8 meters -/
theorem river_bank_bottom_width
  (top_width : ℝ)
  (depth : ℝ)
  (area : ℝ)
  (h1 : top_width = 12)
  (h2 : depth = 50)
  (h3 : area = 500)
  (h4 : area = (top_width + bottom_width) * depth / 2) :
  bottom_width = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_river_bank_bottom_width_l3973_397354


namespace NUMINAMATH_CALUDE_f_value_at_3_l3973_397330

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 6

-- State the theorem
theorem f_value_at_3 (a b c : ℝ) :
  f a b c (-3) = -12 → f a b c 3 = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_f_value_at_3_l3973_397330


namespace NUMINAMATH_CALUDE_next_perfect_square_formula_l3973_397319

/-- A perfect square is an integer that is the square of another integer. -/
def isPerfectSquare (x : ℤ) : Prop := ∃ n : ℤ, x = n^2

/-- The next perfect square after a given perfect square. -/
def nextPerfectSquare (x : ℤ) : ℤ := x + 2 * Int.sqrt x + 1

/-- Theorem: For any perfect square x, the next perfect square is x + 2√x + 1. -/
theorem next_perfect_square_formula (x : ℤ) (h : isPerfectSquare x) :
  isPerfectSquare (nextPerfectSquare x) ∧ 
  ∀ y, isPerfectSquare y ∧ y > x → y ≥ nextPerfectSquare x :=
by sorry

end NUMINAMATH_CALUDE_next_perfect_square_formula_l3973_397319


namespace NUMINAMATH_CALUDE_sequence_formula_l3973_397368

/-- Given a sequence {a_n} where the sum of the first n terms S_n satisfies S_n = 2a_n + 1,
    prove that the general formula for a_n is -2^(n-1) -/
theorem sequence_formula (a : ℕ → ℤ) (S : ℕ → ℤ) 
    (h : ∀ n, S n = 2 * a n + 1) :
  ∀ n, a n = -2^(n-1) := by
sorry

end NUMINAMATH_CALUDE_sequence_formula_l3973_397368


namespace NUMINAMATH_CALUDE_triangle_property_l3973_397325

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : (3 * t.b - t.c) * Real.cos t.A = t.a * Real.cos t.C)
  (h2 : 1/2 * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 2) : 
  Real.cos t.A = 1/3 ∧ 
  ∃ (p : ℝ), p = t.a + t.b + t.c ∧ 
  p ≥ 2 * Real.sqrt 6 + 2 * Real.sqrt 2 ∧
  (p = 2 * Real.sqrt 6 + 2 * Real.sqrt 2 ↔ t.a = t.b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l3973_397325


namespace NUMINAMATH_CALUDE_solution_equation_l3973_397347

theorem solution_equation (x : ℝ) : (5 * 12) / (x / 3) + 80 = 81 ↔ x = 180 :=
by sorry

end NUMINAMATH_CALUDE_solution_equation_l3973_397347


namespace NUMINAMATH_CALUDE_length_width_difference_l3973_397342

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  length : ℕ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.width + r.length)

theorem length_width_difference (r : Rectangle) 
  (h1 : perimeter r = 150)
  (h2 : r.length > r.width)
  (h3 : r.width = 45)
  (h4 : r.length = 60) :
  r.length - r.width = 15 := by sorry

end NUMINAMATH_CALUDE_length_width_difference_l3973_397342


namespace NUMINAMATH_CALUDE_pentagon_area_l3973_397387

/-- The area of a pentagon with specific properties -/
theorem pentagon_area : 
  ∀ (s₁ s₂ s₃ s₄ s₅ : ℝ),
  s₁ = 16 ∧ s₂ = 25 ∧ s₃ = 30 ∧ s₄ = 26 ∧ s₅ = 25 →
  ∃ (triangle_area trapezoid_area : ℝ),
    triangle_area = (1/2) * s₁ * s₂ ∧
    trapezoid_area = (1/2) * (s₄ + s₅) * s₃ ∧
    triangle_area + trapezoid_area = 965 :=
by sorry


end NUMINAMATH_CALUDE_pentagon_area_l3973_397387


namespace NUMINAMATH_CALUDE_front_view_of_given_stack_map_l3973_397349

/-- Represents a column of stacked cubes -/
def Column := List Nat

/-- Represents a stack map as a list of columns -/
def StackMap := List Column

/-- Calculates the maximum height of a column -/
def maxHeight (column : Column) : Nat :=
  column.foldl max 0

/-- Calculates the front view heights of a stack map -/
def frontView (stackMap : StackMap) : List Nat :=
  stackMap.map maxHeight

/-- The given stack map from the problem -/
def givenStackMap : StackMap :=
  [[1, 3], [2, 4, 2], [3, 5], [2]]

theorem front_view_of_given_stack_map :
  frontView givenStackMap = [3, 4, 5, 2] := by
  sorry

end NUMINAMATH_CALUDE_front_view_of_given_stack_map_l3973_397349


namespace NUMINAMATH_CALUDE_no_five_digit_flippy_divisible_by_33_l3973_397311

-- Define a flippy number
def is_flippy (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ 
    (n = a * 10000 + b * 1000 + a * 100 + b * 10 + a ∨
     n = b * 10000 + a * 1000 + b * 100 + a * 10 + b)

-- Define a five-digit number
def is_five_digit (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000

-- Theorem statement
theorem no_five_digit_flippy_divisible_by_33 :
  ¬ ∃ (n : ℕ), is_five_digit n ∧ is_flippy n ∧ n % 33 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_five_digit_flippy_divisible_by_33_l3973_397311


namespace NUMINAMATH_CALUDE_equation_solutions_l3973_397382

theorem equation_solutions : 
  -- Equation 1
  (∃ x : ℝ, 4 * (x - 1)^2 - 8 = 0) ∧
  (∀ x : ℝ, 4 * (x - 1)^2 - 8 = 0 → (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2)) ∧
  -- Equation 2
  (∃ x : ℝ, 2 * x * (x - 3) = x - 3) ∧
  (∀ x : ℝ, 2 * x * (x - 3) = x - 3 → (x = 3 ∨ x = 1/2)) ∧
  -- Equation 3
  (∃ x : ℝ, x^2 - 10*x + 16 = 0) ∧
  (∀ x : ℝ, x^2 - 10*x + 16 = 0 → (x = 8 ∨ x = 2)) ∧
  -- Equation 4
  (∃ x : ℝ, 2*x^2 + 3*x - 1 = 0) ∧
  (∀ x : ℝ, 2*x^2 + 3*x - 1 = 0 → (x = (Real.sqrt 17 - 3) / 4 ∨ x = -(Real.sqrt 17 + 3) / 4)) := by
  sorry


end NUMINAMATH_CALUDE_equation_solutions_l3973_397382


namespace NUMINAMATH_CALUDE_circumscribed_polygon_similarity_l3973_397306

/-- A circumscribed n-gon (n > 3) divided by non-intersecting diagonals into triangles -/
structure CircumscribedPolygon (n : ℕ) :=
  (n_gt_three : n > 3)
  (divided_into_triangles : Bool)
  (non_intersecting_diagonals : Bool)

/-- Predicate to check if all triangles are similar to at least one other triangle -/
def all_triangles_similar (p : CircumscribedPolygon n) : Prop := sorry

/-- The set of possible n values for which the described situation is possible -/
def possible_n_values : Set ℕ := {n | n = 4 ∨ n > 5}

/-- Theorem stating the possible values of n for which the described situation is possible -/
theorem circumscribed_polygon_similarity (n : ℕ) (p : CircumscribedPolygon n) :
  all_triangles_similar p ↔ n ∈ possible_n_values :=
sorry

end NUMINAMATH_CALUDE_circumscribed_polygon_similarity_l3973_397306


namespace NUMINAMATH_CALUDE_equation_solution_l3973_397356

theorem equation_solution :
  let x : ℚ := 4 / 7
  (7 * x + 3) / (x + 5) - 5 / (x + 5) = 2 / (x + 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3973_397356


namespace NUMINAMATH_CALUDE_max_earnings_ali_baba_l3973_397369

/-- Represents the weight of the bag when filled with only diamonds -/
def diamond_full_weight : ℝ := 40

/-- Represents the weight of the bag when filled with only gold -/
def gold_full_weight : ℝ := 200

/-- Represents the maximum weight Ali Baba can carry -/
def max_carry_weight : ℝ := 100

/-- Represents the cost of 1 kg of diamonds in dinars -/
def diamond_cost : ℝ := 60

/-- Represents the cost of 1 kg of gold in dinars -/
def gold_cost : ℝ := 20

/-- Represents the objective function to maximize -/
def objective_function (x y : ℝ) : ℝ := diamond_cost * x + gold_cost * y

/-- Theorem stating that the maximum value of the objective function
    under given constraints is 3000 dinars -/
theorem max_earnings_ali_baba :
  ∃ x y : ℝ,
    x ≥ 0 ∧
    y ≥ 0 ∧
    x + y ≤ max_carry_weight ∧
    (x / diamond_full_weight + y / gold_full_weight) ≤ 1 ∧
    objective_function x y = 3000 ∧
    ∀ x' y' : ℝ,
      x' ≥ 0 →
      y' ≥ 0 →
      x' + y' ≤ max_carry_weight →
      (x' / diamond_full_weight + y' / gold_full_weight) ≤ 1 →
      objective_function x' y' ≤ 3000 :=
by sorry

end NUMINAMATH_CALUDE_max_earnings_ali_baba_l3973_397369


namespace NUMINAMATH_CALUDE_max_pyramid_volume_l3973_397380

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a pyramid with vertex O and base ABC -/
structure Pyramid where
  O : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

/-- Calculates the angle between two vectors -/
def angle (v1 v2 : Point3D) : ℝ := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Calculates the volume of a pyramid -/
def pyramidVolume (p : Pyramid) : ℝ := sorry

/-- Checks if a point is on the surface of a sphere -/
def isOnSphere (center : Point3D) (radius : ℝ) (point : Point3D) : Prop :=
  distance center point = radius

theorem max_pyramid_volume (p : Pyramid) (r : ℝ) :
  r = 3 →
  isOnSphere p.O r p.A →
  isOnSphere p.O r p.B →
  isOnSphere p.O r p.C →
  angle (p.A) (p.B) = 150 * π / 180 →
  ∀ (q : Pyramid), 
    isOnSphere p.O r q.A →
    isOnSphere p.O r q.B →
    isOnSphere p.O r q.C →
    pyramidVolume q ≤ 9/2 :=
by sorry

end NUMINAMATH_CALUDE_max_pyramid_volume_l3973_397380


namespace NUMINAMATH_CALUDE_complex_number_problem_l3973_397301

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  (z₁ - 2) * (1 + Complex.I) = 1 - Complex.I →
  z₂.im = 2 →
  ∃ (r : ℝ), z₁ * z₂ = r →
  z₂ = 4 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3973_397301


namespace NUMINAMATH_CALUDE_a_in_M_sufficient_not_necessary_for_a_in_N_l3973_397338

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 < 4}
def N : Set ℝ := {x : ℝ | x < 2}

-- Statement to prove
theorem a_in_M_sufficient_not_necessary_for_a_in_N :
  (∀ a : ℝ, a ∈ M → a ∈ N) ∧ (∃ a : ℝ, a ∈ N ∧ a ∉ M) :=
sorry

end NUMINAMATH_CALUDE_a_in_M_sufficient_not_necessary_for_a_in_N_l3973_397338


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l3973_397373

theorem least_k_for_inequality (k : ℤ) : 
  (∀ m : ℤ, m < k → (0.000010101 * (10 : ℝ)^m ≤ 10000)) →
  (0.000010101 * (10 : ℝ)^k > 10000) →
  k = 9 := by
sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l3973_397373


namespace NUMINAMATH_CALUDE_wall_breadth_l3973_397336

/-- Proves that the breadth of a wall with given proportions and volume is 0.4 meters -/
theorem wall_breadth (V h l b : ℝ) (hV : V = 12.8) (hh : h = 5 * b) (hl : l = 8 * h) 
  (hvolume : V = l * b * h) : b = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_wall_breadth_l3973_397336


namespace NUMINAMATH_CALUDE_f_max_at_neg_two_l3973_397315

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := -2 * x^2 - 8 * x + 18

/-- The statement that f attains its maximum at x = -2 with a value of 26 -/
theorem f_max_at_neg_two :
  (∃ (a : ℝ), ∀ (x : ℝ), f x ≤ f a) ∧
  (∀ (x : ℝ), f x ≤ 26) ∧
  (f (-2) = 26) :=
sorry

end NUMINAMATH_CALUDE_f_max_at_neg_two_l3973_397315


namespace NUMINAMATH_CALUDE_sum_consecutive_triangular_numbers_l3973_397320

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_consecutive_triangular_numbers (n : ℕ) :
  triangular_number n + triangular_number (n + 1) = (n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_consecutive_triangular_numbers_l3973_397320


namespace NUMINAMATH_CALUDE_hotel_floors_l3973_397378

theorem hotel_floors (available_rooms : ℕ) (rooms_per_floor : ℕ) (unavailable_floors : ℕ) : 
  available_rooms = 90 → rooms_per_floor = 10 → unavailable_floors = 1 →
  (available_rooms / rooms_per_floor + unavailable_floors = 10) := by
sorry

end NUMINAMATH_CALUDE_hotel_floors_l3973_397378


namespace NUMINAMATH_CALUDE_solve_lawn_mowing_problem_l3973_397365

/-- Kaleb's lawn mowing business finances --/
def lawn_mowing_problem (spring_earnings summer_earnings final_amount : ℕ) : Prop :=
  let total_earnings := spring_earnings + summer_earnings
  let supplies_cost := total_earnings - final_amount
  supplies_cost = total_earnings - final_amount

theorem solve_lawn_mowing_problem :
  lawn_mowing_problem 4 50 50 = true :=
sorry

end NUMINAMATH_CALUDE_solve_lawn_mowing_problem_l3973_397365


namespace NUMINAMATH_CALUDE_valid_street_distances_l3973_397351

/-- Represents the position of a house on a street. -/
structure House where
  position : ℝ

/-- The street with four houses. -/
structure Street where
  andrei : House
  borya : House
  vova : House
  gleb : House

/-- The distance between two houses. -/
def distance (h1 h2 : House) : ℝ :=
  |h1.position - h2.position|

/-- A street satisfying the given conditions. -/
def validStreet (s : Street) : Prop :=
  distance s.andrei s.borya = 600 ∧
  distance s.vova s.gleb = 600 ∧
  distance s.andrei s.gleb = 3 * distance s.borya s.vova

theorem valid_street_distances (s : Street) (h : validStreet s) :
  distance s.andrei s.gleb = 900 ∨ distance s.andrei s.gleb = 1800 :=
sorry

end NUMINAMATH_CALUDE_valid_street_distances_l3973_397351


namespace NUMINAMATH_CALUDE_complex_function_property_l3973_397344

theorem complex_function_property (a b : ℝ) :
  (∀ z : ℂ, Complex.abs ((a + b * Complex.I) * z^2 - z) = Complex.abs ((a + b * Complex.I) * z^2)) →
  Complex.abs (a + b * Complex.I) = 5 →
  b^2 = 99/4 := by sorry

end NUMINAMATH_CALUDE_complex_function_property_l3973_397344


namespace NUMINAMATH_CALUDE_inequality_comparison_l3973_397333

theorem inequality_comparison (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a < b + c) (hbc : b < c + a) : 
  let K := a^4 + b^4 + c^4 - 2*(a^2*b^2 + b^2*c^2 + c^2*a^2)
  (K < 0 ↔ c < a + b) ∧ 
  (K = 0 ↔ c = a + b) ∧ 
  (K > 0 ↔ c > a + b) := by
sorry

end NUMINAMATH_CALUDE_inequality_comparison_l3973_397333


namespace NUMINAMATH_CALUDE_line_k_value_l3973_397366

/-- A line passes through the points (0, 3), (7, k), and (21, 2) -/
def line_passes_through (k : ℚ) : Prop :=
  ∃ m b : ℚ, 
    (3 = m * 0 + b) ∧ 
    (k = m * 7 + b) ∧ 
    (2 = m * 21 + b)

/-- Theorem: If a line passes through (0, 3), (7, k), and (21, 2), then k = 8/3 -/
theorem line_k_value : 
  ∀ k : ℚ, line_passes_through k → k = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_line_k_value_l3973_397366


namespace NUMINAMATH_CALUDE_eight_number_sequence_proof_l3973_397376

theorem eight_number_sequence_proof :
  ∀ (a : Fin 8 → ℕ),
  (a 0 = 20) →
  (a 7 = 16) →
  (∀ i : Fin 6, a i + a (i + 1) + a (i + 2) = 100) →
  (∀ i : Fin 8, a i = [20, 16, 64, 20, 16, 64, 20, 16].get i) :=
by
  sorry

end NUMINAMATH_CALUDE_eight_number_sequence_proof_l3973_397376


namespace NUMINAMATH_CALUDE_double_layer_cake_cost_double_layer_cake_cost_is_seven_l3973_397392

theorem double_layer_cake_cost (single_layer_cost : ℝ) 
                               (single_layer_quantity : ℕ) 
                               (double_layer_quantity : ℕ) 
                               (total_paid : ℝ) 
                               (change_received : ℝ) : ℝ :=
  let total_spent := total_paid - change_received
  let single_layer_total := single_layer_cost * single_layer_quantity
  let double_layer_total := total_spent - single_layer_total
  double_layer_total / double_layer_quantity

theorem double_layer_cake_cost_is_seven :
  double_layer_cake_cost 4 7 5 100 37 = 7 := by
  sorry

end NUMINAMATH_CALUDE_double_layer_cake_cost_double_layer_cake_cost_is_seven_l3973_397392


namespace NUMINAMATH_CALUDE_square_difference_fourth_power_l3973_397398

theorem square_difference_fourth_power : (7^2 - 5^2)^4 = 331776 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_fourth_power_l3973_397398


namespace NUMINAMATH_CALUDE_stickers_given_to_lucy_l3973_397302

/-- Given that Gary initially had 99 stickers, gave 26 stickers to Alex, 
    and had 31 stickers left afterwards, prove that Gary gave 42 stickers to Lucy. -/
theorem stickers_given_to_lucy (initial_stickers : ℕ) (stickers_to_alex : ℕ) (stickers_left : ℕ) :
  initial_stickers = 99 →
  stickers_to_alex = 26 →
  stickers_left = 31 →
  initial_stickers - stickers_to_alex - stickers_left = 42 :=
by sorry

end NUMINAMATH_CALUDE_stickers_given_to_lucy_l3973_397302


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_sum_l3973_397367

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_minimum_sum 
  (a : ℕ → ℝ) 
  (h_geom : GeometricSequence a) 
  (h_positive : ∀ n, a n > 0) 
  (h_product : a 3 * a 5 = 12) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x * y = 12 → x + y ≥ 4 * Real.sqrt 3) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = 12 ∧ x + y = 4 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_sum_l3973_397367


namespace NUMINAMATH_CALUDE_janes_age_l3973_397386

/-- Jane's babysitting problem -/
theorem janes_age :
  ∀ (jane_start_age : ℕ) 
    (years_since_stopped : ℕ) 
    (oldest_babysat_age : ℕ),
  jane_start_age = 16 →
  years_since_stopped = 10 →
  oldest_babysat_age = 24 →
  ∃ (jane_current_age : ℕ),
    jane_current_age = 38 ∧
    (∀ (child_age : ℕ),
      child_age ≤ oldest_babysat_age →
      child_age ≤ (jane_current_age - years_since_stopped) / 2) :=
by sorry

end NUMINAMATH_CALUDE_janes_age_l3973_397386


namespace NUMINAMATH_CALUDE_geometric_sequence_decreasing_l3973_397358

def geometric_sequence (n : ℕ) : ℝ := 4 * (3 ^ (1 - n))

theorem geometric_sequence_decreasing :
  ∀ n : ℕ, geometric_sequence (n + 1) < geometric_sequence n :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_decreasing_l3973_397358


namespace NUMINAMATH_CALUDE_ahn_max_number_l3973_397385

theorem ahn_max_number : ∃ (max : ℕ), max = 650 ∧ 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 3 * (300 - n) + 50 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_ahn_max_number_l3973_397385


namespace NUMINAMATH_CALUDE_lines_2_3_parallel_l3973_397340

-- Define the slopes of the lines
def slope1 : ℚ := 3 / 4
def slope2 : ℚ := -3 / 4
def slope3 : ℚ := -3 / 4
def slope4 : ℚ := -4 / 3

-- Define the equations of the lines
def line1 (x y : ℚ) : Prop := 4 * y - 3 * x = 16
def line2 (x y : ℚ) : Prop := -3 * x - 4 * y = 15
def line3 (x y : ℚ) : Prop := 4 * y + 3 * x = 16
def line4 (x y : ℚ) : Prop := 3 * y + 4 * x = 15

-- Theorem: Lines 2 and 3 are parallel
theorem lines_2_3_parallel : 
  ∀ (x1 y1 x2 y2 : ℚ), 
    line2 x1 y1 → line3 x2 y2 → 
    slope2 = slope3 ∧ slope2 ≠ slope1 ∧ slope2 ≠ slope4 := by
  sorry

end NUMINAMATH_CALUDE_lines_2_3_parallel_l3973_397340


namespace NUMINAMATH_CALUDE_sum_of_greatest_b_values_l3973_397352

theorem sum_of_greatest_b_values (c : ℝ) (h : c ≠ 0) :
  ∃ (b₁ b₂ : ℝ), b₁ > b₂ ∧ b₂ > 0 ∧
  (4 * b₁^4 - 41 * b₁^2 + 100) * c = 0 ∧
  (4 * b₂^4 - 41 * b₂^2 + 100) * c = 0 ∧
  ∀ (b : ℝ), (4 * b^4 - 41 * b^2 + 100) * c = 0 → b ≤ b₁ ∧
  b₁ + b₂ = 4.5 :=
sorry

end NUMINAMATH_CALUDE_sum_of_greatest_b_values_l3973_397352


namespace NUMINAMATH_CALUDE_root_equation_value_l3973_397395

theorem root_equation_value (m : ℝ) : m^2 - m - 1 = 0 → m^2 - m + 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l3973_397395


namespace NUMINAMATH_CALUDE_johns_former_apartment_cost_l3973_397375

/-- Proves that the cost per square foot of John's former apartment was $2 -/
theorem johns_former_apartment_cost (former_size : ℝ) (new_rent : ℝ) (savings : ℝ) : 
  former_size = 750 →
  new_rent = 2800 →
  savings = 1200 →
  ∃ (cost_per_sqft : ℝ), 
    cost_per_sqft = 2 ∧ 
    former_size * cost_per_sqft * 12 = (new_rent / 2) * 12 + savings :=
by sorry

end NUMINAMATH_CALUDE_johns_former_apartment_cost_l3973_397375


namespace NUMINAMATH_CALUDE_plum_pies_count_l3973_397355

theorem plum_pies_count (total : ℕ) (ratio_r : ℕ) (ratio_p : ℕ) (ratio_m : ℕ) 
  (h_total : total = 30)
  (h_ratio : ratio_r = 2 ∧ ratio_p = 5 ∧ ratio_m = 3) :
  (total * ratio_m) / (ratio_r + ratio_p + ratio_m) = 9 := by
sorry

end NUMINAMATH_CALUDE_plum_pies_count_l3973_397355


namespace NUMINAMATH_CALUDE_specific_profit_calculation_l3973_397310

/-- Given an item cost, markup percentage, discount percentage, and number of items sold,
    calculates the total profit. -/
def totalProfit (a : ℝ) (markup discount : ℝ) (m : ℝ) : ℝ :=
  let sellingPrice := a * (1 + markup)
  let discountedPrice := sellingPrice * (1 - discount)
  m * (discountedPrice - a)

/-- Theorem stating that under specific conditions, the total profit is 0.08am -/
theorem specific_profit_calculation (a m : ℝ) :
  totalProfit a 0.2 0.1 m = 0.08 * a * m :=
by sorry

end NUMINAMATH_CALUDE_specific_profit_calculation_l3973_397310


namespace NUMINAMATH_CALUDE_smallest_positive_integer_e_l3973_397393

theorem smallest_positive_integer_e (a b c d e : ℤ) : 
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x = -3 ∨ x = 7 ∨ x = 11 ∨ x = -1/4) →
  e > 0 →
  (∀ e' : ℤ, e' > 0 ∧ 
    (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e' = 0 ↔ x = -3 ∨ x = 7 ∨ x = 11 ∨ x = -1/4) →
    e' ≥ e) →
  e = 231 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_e_l3973_397393


namespace NUMINAMATH_CALUDE_prob_six_odd_in_eight_rolls_l3973_397345

/-- The probability of rolling an odd number on a fair 6-sided die -/
def prob_odd : ℚ := 1 / 2

/-- The number of rolls -/
def num_rolls : ℕ := 8

/-- The number of desired odd rolls -/
def num_odd : ℕ := 6

/-- The probability of getting exactly 6 odd numbers in 8 rolls of a fair 6-sided die -/
theorem prob_six_odd_in_eight_rolls : 
  (Nat.choose num_rolls num_odd : ℚ) * prob_odd ^ num_odd * (1 - prob_odd) ^ (num_rolls - num_odd) = 7 / 64 := by
  sorry

end NUMINAMATH_CALUDE_prob_six_odd_in_eight_rolls_l3973_397345


namespace NUMINAMATH_CALUDE_lunch_packing_days_l3973_397363

/-- Represents the number of school days for each school -/
structure SchoolDays where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Represents the number of days a student packs lunch -/
structure LunchDays where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ

/-- Given the conditions of the problem, prove the correct expressions for lunch packing days -/
theorem lunch_packing_days (sd : SchoolDays) : 
  ∃ (ld : LunchDays), 
    ld.A = (3 * sd.A) / 5 ∧
    ld.B = (3 * sd.B) / 20 ∧
    ld.C = (3 * sd.C) / 10 ∧
    ld.D = sd.A / 3 := by
  sorry

end NUMINAMATH_CALUDE_lunch_packing_days_l3973_397363


namespace NUMINAMATH_CALUDE_shyne_eggplant_packets_l3973_397379

/-- The number of eggplants that can be grown from one seed packet -/
def eggplants_per_packet : ℕ := 14

/-- The number of sunflowers that can be grown from one seed packet -/
def sunflowers_per_packet : ℕ := 10

/-- The number of sunflower seed packets Shyne bought -/
def sunflower_packets : ℕ := 6

/-- The total number of plants Shyne can grow in her backyard -/
def total_plants : ℕ := 116

/-- The number of eggplant seed packets Shyne bought -/
def eggplant_packets : ℕ := 4

theorem shyne_eggplant_packets : 
  eggplant_packets * eggplants_per_packet + sunflower_packets * sunflowers_per_packet = total_plants :=
by sorry

end NUMINAMATH_CALUDE_shyne_eggplant_packets_l3973_397379


namespace NUMINAMATH_CALUDE_rational_powers_imply_rational_irrational_with_rational_powers_l3973_397348

-- Define rationality for real numbers
def IsRational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

-- Part a
theorem rational_powers_imply_rational (x : ℝ) 
  (h1 : IsRational (x^7)) (h2 : IsRational (x^12)) : 
  IsRational x := by sorry

-- Part b
theorem irrational_with_rational_powers : 
  ∃ (x : ℝ), IsRational (x^9) ∧ IsRational (x^12) ∧ ¬IsRational x := by sorry

end NUMINAMATH_CALUDE_rational_powers_imply_rational_irrational_with_rational_powers_l3973_397348


namespace NUMINAMATH_CALUDE_sin_75_cos_45_minus_cos_75_sin_45_l3973_397303

theorem sin_75_cos_45_minus_cos_75_sin_45 :
  Real.sin (75 * π / 180) * Real.cos (45 * π / 180) -
  Real.cos (75 * π / 180) * Real.sin (45 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_cos_45_minus_cos_75_sin_45_l3973_397303


namespace NUMINAMATH_CALUDE_max_child_age_fraction_is_five_eighths_l3973_397321

/-- The maximum fraction of Jane's age that a child she babysat could be -/
def max_child_age_fraction : ℚ :=
  let jane_current_age : ℕ := 34
  let years_since_stopped : ℕ := 10
  let jane_age_when_stopped : ℕ := jane_current_age - years_since_stopped
  let oldest_child_current_age : ℕ := 25
  let oldest_child_age_when_jane_stopped : ℕ := oldest_child_current_age - years_since_stopped
  (oldest_child_age_when_jane_stopped : ℚ) / jane_age_when_stopped

/-- Theorem stating that the maximum fraction of Jane's age that a child she babysat could be is 5/8 -/
theorem max_child_age_fraction_is_five_eighths :
  max_child_age_fraction = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_max_child_age_fraction_is_five_eighths_l3973_397321


namespace NUMINAMATH_CALUDE_largest_difference_l3973_397322

theorem largest_difference (A B C D E F : ℕ) : 
  A = 3 * 2005^2006 →
  B = 2005^2006 →
  C = 2004 * 2005^2005 →
  D = 3 * 2005^2005 →
  E = 2005^2005 →
  F = 2005^2004 →
  (A - B > B - C) ∧ 
  (A - B > C - D) ∧ 
  (A - B > D - E) ∧ 
  (A - B > E - F) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_difference_l3973_397322


namespace NUMINAMATH_CALUDE_solution_set_length_l3973_397383

theorem solution_set_length (a b : ℝ) : 
  (∃ l : ℝ, l = 12 ∧ l = (b - 4) / 3 - (a - 4) / 3) → b - a = 36 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_length_l3973_397383


namespace NUMINAMATH_CALUDE_first_job_earnings_is_52_l3973_397323

/-- Represents Mike's weekly wages --/
def TotalWages : ℝ := 160

/-- Represents the hours Mike works at his second job --/
def SecondJobHours : ℝ := 12

/-- Represents the hourly rate for Mike's second job --/
def SecondJobRate : ℝ := 9

/-- Calculates the amount Mike earns from his second job --/
def SecondJobEarnings : ℝ := SecondJobHours * SecondJobRate

/-- Represents the amount Mike earns from his first job --/
def FirstJobEarnings : ℝ := TotalWages - SecondJobEarnings

/-- Proves that Mike's earnings from his first job is $52 --/
theorem first_job_earnings_is_52 : FirstJobEarnings = 52 := by
  sorry

end NUMINAMATH_CALUDE_first_job_earnings_is_52_l3973_397323


namespace NUMINAMATH_CALUDE_area_is_72_l3973_397372

/-- A square with side length 12 and a right triangle in a plane -/
structure Configuration :=
  (square_side : ℝ)
  (square_lower_right : ℝ × ℝ)
  (triangle_base : ℝ)
  (hypotenuse_end : ℝ × ℝ)

/-- The area of the region formed by the portion of the square below the diagonal of the triangle -/
def area_below_diagonal (config : Configuration) : ℝ :=
  sorry

/-- The theorem stating the area is 72 square units -/
theorem area_is_72 (config : Configuration) 
  (h1 : config.square_side = 12)
  (h2 : config.square_lower_right = (12, 0))
  (h3 : config.triangle_base = 12)
  (h4 : config.hypotenuse_end = (24, 0)) :
  area_below_diagonal config = 72 :=
sorry

end NUMINAMATH_CALUDE_area_is_72_l3973_397372


namespace NUMINAMATH_CALUDE_l_shaped_figure_perimeter_l3973_397357

/-- Represents an L-shaped figure formed by a 3x3 square with a 2x2 square attached to one side -/
structure LShapedFigure :=
  (base_side : ℕ)
  (extension_side : ℕ)
  (unit_length : ℝ)
  (h_base : base_side = 3)
  (h_extension : extension_side = 2)
  (h_unit : unit_length = 1)

/-- Calculates the perimeter of the L-shaped figure -/
def perimeter (figure : LShapedFigure) : ℝ :=
  2 * (figure.base_side + figure.extension_side + figure.base_side) * figure.unit_length

/-- Theorem stating that the perimeter of the L-shaped figure is 15 units -/
theorem l_shaped_figure_perimeter :
  ∀ (figure : LShapedFigure), perimeter figure = 15 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_figure_perimeter_l3973_397357


namespace NUMINAMATH_CALUDE_largest_angle_in_345_ratio_triangle_l3973_397316

theorem largest_angle_in_345_ratio_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 →
    b = (4/3) * a →
    c = (5/3) * a →
    a + b + c = 180 →
    c = 75 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_345_ratio_triangle_l3973_397316


namespace NUMINAMATH_CALUDE_arithmetic_equalities_l3973_397317

theorem arithmetic_equalities : 
  (10 - 20 - (-7) + |(-2)|) = -1 ∧ 
  (48 * (-1/4) - (-36) / 4) = -3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_equalities_l3973_397317


namespace NUMINAMATH_CALUDE_curve_C_properties_l3973_397339

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (x - 1)^2 + y^2 = 4 * ((x - 4)^2 + y^2)}

-- Define the line l
def l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               x - y + 3 = 0}

-- Theorem statement
theorem curve_C_properties :
  -- 1. The equation of C is x^2 + y^2 = 4
  (∀ p : ℝ × ℝ, p ∈ C ↔ let (x, y) := p; x^2 + y^2 = 4) ∧
  -- 2. The minimum distance from C to l is (3√2)/2 - 2
  (∃ d_min : ℝ, d_min = 3 * Real.sqrt 2 / 2 - 2 ∧
    (∀ p ∈ C, ∀ q ∈ l, dist p q ≥ d_min) ∧
    (∃ p ∈ C, ∃ q ∈ l, dist p q = d_min)) ∧
  -- 3. The maximum distance from C to l is 2 + (3√2)/2
  (∃ d_max : ℝ, d_max = 2 + 3 * Real.sqrt 2 / 2 ∧
    (∀ p ∈ C, ∀ q ∈ l, dist p q ≤ d_max) ∧
    (∃ p ∈ C, ∃ q ∈ l, dist p q = d_max)) :=
by sorry

end NUMINAMATH_CALUDE_curve_C_properties_l3973_397339


namespace NUMINAMATH_CALUDE_rooster_count_l3973_397318

/-- Given a chicken farm with roosters and hens, proves the number of roosters -/
theorem rooster_count (total : ℕ) (ratio : ℚ) (rooster_count : ℕ) : 
  total = 9000 →
  ratio = 2 / 1 →
  rooster_count = total * (ratio / (1 + ratio)) →
  rooster_count = 6000 := by
  sorry


end NUMINAMATH_CALUDE_rooster_count_l3973_397318


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l3973_397396

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 12 * x + c = 0) →  -- exactly one solution
  (a + c = 14) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 7 - Real.sqrt 31 ∧ c = 7 + Real.sqrt 31) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l3973_397396


namespace NUMINAMATH_CALUDE_eighth_term_value_l3973_397394

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the theorem
theorem eighth_term_value 
  (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_sum : a 3 + 3 * a 8 + a 13 = 120) : 
  a 8 = 24 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_value_l3973_397394


namespace NUMINAMATH_CALUDE_quadrilateral_perpendicular_diagonals_l3973_397389

/-- Given a quadrilateral ABCD in the complex plane, construct points O₁, O₂, O₃, O₄
    and prove that O₁O₃ is perpendicular and equal to O₂O₄ -/
theorem quadrilateral_perpendicular_diagonals
  (a b c d : ℂ) : 
  let g₁ : ℂ := (a + d) / 2
  let g₂ : ℂ := (b + a) / 2
  let g₃ : ℂ := (c + b) / 2
  let g₄ : ℂ := (d + c) / 2
  let o₁ : ℂ := g₁ + (d - a) / 2 * Complex.I
  let o₂ : ℂ := g₂ + (a - b) / 2 * Complex.I
  let o₃ : ℂ := g₃ + (c - b) / 2 * Complex.I
  let o₄ : ℂ := g₄ + (d - c) / 2 * Complex.I
  (o₃ - o₁) = (o₄ - o₂) * Complex.I ∧ Complex.abs (o₃ - o₁) = Complex.abs (o₄ - o₂) :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perpendicular_diagonals_l3973_397389


namespace NUMINAMATH_CALUDE_combine_like_terms_l3973_397334

theorem combine_like_terms (a : ℝ) : 2 * a - 5 * a = -3 * a := by
  sorry

end NUMINAMATH_CALUDE_combine_like_terms_l3973_397334


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3973_397308

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 6 + a 10 = 20)
  (h_a4 : a 4 = 2) :
  a 12 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3973_397308


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_for_given_equation_l3973_397361

theorem sum_of_x_and_y_for_given_equation (x y : ℝ) : 
  2 * x^2 - 4 * x * y + 4 * y^2 + 6 * x + 9 = 0 → x + y = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_for_given_equation_l3973_397361


namespace NUMINAMATH_CALUDE_inequality_proof_l3973_397326

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a * b / (c - 1)) + (b * c / (a - 1)) + (c * a / (b - 1)) ≥ 12 ∧
  ((a * b / (c - 1)) + (b * c / (a - 1)) + (c * a / (b - 1)) = 12 ↔ a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l3973_397326
