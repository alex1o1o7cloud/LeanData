import Mathlib

namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l3370_337018

theorem smallest_lcm_with_gcd_5 :
  ∃ (k l : ℕ), 
    1000 ≤ k ∧ k < 10000 ∧
    1000 ≤ l ∧ l < 10000 ∧
    Nat.gcd k l = 5 ∧
    Nat.lcm k l = 203010 ∧
    ∀ (m n : ℕ), 
      1000 ≤ m ∧ m < 10000 ∧
      1000 ≤ n ∧ n < 10000 ∧
      Nat.gcd m n = 5 →
      Nat.lcm m n ≥ 203010 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l3370_337018


namespace NUMINAMATH_CALUDE_otimes_inequality_range_l3370_337028

-- Define the ⊗ operation
def otimes (x y : ℝ) := x * (2 - y)

-- Theorem statement
theorem otimes_inequality_range (m : ℝ) :
  (∀ x : ℝ, otimes (x + m) x < 1) ↔ -4 < m ∧ m < 0 := by sorry

end NUMINAMATH_CALUDE_otimes_inequality_range_l3370_337028


namespace NUMINAMATH_CALUDE_problem_solution_l3370_337010

noncomputable def f (x : ℝ) := x + 1 + abs (3 - x)

theorem problem_solution :
  (∀ x ≥ -1, f x ≤ 6 ↔ -1 ≤ x ∧ x ≤ 4) ∧
  (∀ x ≥ -1, f x ≥ 4) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 8 * a * b = a + 2 * b → 2 * a + b ≥ 9/8) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3370_337010


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l3370_337063

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l3370_337063


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3370_337039

theorem fixed_point_on_line (a : ℝ) : 
  let line := fun (x y : ℝ) => a * x - y + 1 = 0
  line 0 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3370_337039


namespace NUMINAMATH_CALUDE_sugar_in_recipe_l3370_337025

theorem sugar_in_recipe (sugar_already_in : ℕ) (sugar_to_add : ℕ) : 
  sugar_already_in = 2 → sugar_to_add = 11 → sugar_already_in + sugar_to_add = 13 := by
  sorry

end NUMINAMATH_CALUDE_sugar_in_recipe_l3370_337025


namespace NUMINAMATH_CALUDE_min_value_theorem_l3370_337031

theorem min_value_theorem (x A B C : ℝ) (hx : x > 0) (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (hxA : x^2 + 1/x^2 = A)
  (hxB : x - 1/x = B)
  (hxC : x^3 - 1/x^3 = C) :
  ∃ (m : ℝ), m = 6.4 ∧ ∀ (A' B' C' x' : ℝ), 
    x' > 0 → A' > 0 → B' > 0 → C' > 0 →
    x'^2 + 1/x'^2 = A' →
    x' - 1/x' = B' →
    x'^3 - 1/x'^3 = C' →
    A'^3 / C' ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3370_337031


namespace NUMINAMATH_CALUDE_backpack_price_l3370_337020

theorem backpack_price (t_shirt_price cap_price discount total_after_discount : ℕ) 
  (ht : t_shirt_price = 30)
  (hc : cap_price = 5)
  (hd : discount = 2)
  (ht : total_after_discount = 43) :
  ∃ backpack_price : ℕ, 
    t_shirt_price + backpack_price + cap_price - discount = total_after_discount ∧ 
    backpack_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_backpack_price_l3370_337020


namespace NUMINAMATH_CALUDE_unique_intersection_l3370_337052

-- Define the two functions
def f (x : ℝ) : ℝ := |3 * x + 6|
def g (x : ℝ) : ℝ := -|2 * x - 1|

-- State the theorem
theorem unique_intersection :
  ∃! p : ℝ × ℝ, 
    f p.1 = g p.1 ∧ 
    p.1 = -1 ∧ 
    p.2 = -3 := by sorry

end NUMINAMATH_CALUDE_unique_intersection_l3370_337052


namespace NUMINAMATH_CALUDE_blue_parrots_count_l3370_337032

theorem blue_parrots_count (total : ℕ) (green_fraction : ℚ) (blue_parrots : ℕ) : 
  total = 160 →
  green_fraction = 5/8 →
  blue_parrots = total - (green_fraction * total).num →
  blue_parrots = 60 := by
sorry

end NUMINAMATH_CALUDE_blue_parrots_count_l3370_337032


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3370_337093

theorem complex_modulus_problem (z : ℂ) (h : z * (1 + 2 * Complex.I) = 3 - Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3370_337093


namespace NUMINAMATH_CALUDE_expression_simplification_l3370_337095

theorem expression_simplification : 
  ((0.3 * 0.8) / 0.2) + (0.1 * 0.5) ^ 2 - 1 / (0.5 * 0.8)^2 = -5.0475 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3370_337095


namespace NUMINAMATH_CALUDE_inequality_proof_l3370_337053

theorem inequality_proof (a b c : ℝ) 
  (ha : a = 17/18)
  (hb : b = Real.cos (1/3))
  (hc : c = 3 * Real.sin (1/3)) :
  c > b ∧ b > a :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3370_337053


namespace NUMINAMATH_CALUDE_divisibility_property_l3370_337057

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, b * k = a

def sequence_property (A : ℕ → ℕ) : Prop :=
  ∀ n k : ℕ, is_divisible (A (n + k) - A k) (A n)

def B (A : ℕ → ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => B A n * A (n + 1)

theorem divisibility_property (A : ℕ → ℕ) (h : sequence_property A) :
  ∀ n k : ℕ, is_divisible (B A (n + k)) ((B A n) * (B A k)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_property_l3370_337057


namespace NUMINAMATH_CALUDE_M_invertible_iff_square_free_l3370_337026

def M (n : ℕ+) : Matrix (Fin n) (Fin n) ℤ :=
  Matrix.of (fun i j => if (i.val + 1) % j.val = 0 then 1 else 0)

def square_free (m : ℕ) : Prop :=
  ∀ k : ℕ, k > 1 → m % (k * k) ≠ 0

theorem M_invertible_iff_square_free (n : ℕ+) :
  IsUnit (M n).det ↔ square_free (n + 1) :=
sorry

end NUMINAMATH_CALUDE_M_invertible_iff_square_free_l3370_337026


namespace NUMINAMATH_CALUDE_f_2013_l3370_337005

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 6) ≤ f (x + 2) + 4) ∧
  (∀ x : ℝ, f (x + 4) ≥ f (x + 2) + 2) ∧
  (f 1 = 1)

theorem f_2013 (f : ℝ → ℝ) (h : f_properties f) : f 2013 = 2013 := by
  sorry

end NUMINAMATH_CALUDE_f_2013_l3370_337005


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_under_1000_l3370_337009

/-- A function that checks if a number has only even digits -/
def hasOnlyEvenDigits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

/-- The largest positive integer with only even digits that is less than 1000 and is a multiple of 9 -/
def largestEvenDigitMultipleOf9Under1000 : ℕ := 864

theorem largest_even_digit_multiple_of_9_under_1000 :
  (largestEvenDigitMultipleOf9Under1000 < 1000) ∧
  (largestEvenDigitMultipleOf9Under1000 % 9 = 0) ∧
  (hasOnlyEvenDigits largestEvenDigitMultipleOf9Under1000) ∧
  (∀ n : ℕ, n < 1000 → n % 9 = 0 → hasOnlyEvenDigits n → n ≤ largestEvenDigitMultipleOf9Under1000) :=
by sorry

#eval largestEvenDigitMultipleOf9Under1000

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_under_1000_l3370_337009


namespace NUMINAMATH_CALUDE_egg_collection_sum_l3370_337092

theorem egg_collection_sum (n : ℕ) (a₁ : ℕ) (d : ℕ) (S : ℕ) : 
  n = 12 → a₁ = 25 → d = 5 → S = n * (2 * a₁ + (n - 1) * d) / 2 → S = 630 := by sorry

end NUMINAMATH_CALUDE_egg_collection_sum_l3370_337092


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l3370_337081

/-- A sequence a_n is geometric if there exists a constant r such that a_{n+1} = r * a_n for all n. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- A sequence a_n is increasing if a_n < a_{n+1} for all n. -/
def IsIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- The condition a₁ < a₂ < a₄ for a sequence a_n. -/
def Condition (a : ℕ → ℝ) : Prop :=
  a 1 < a 2 ∧ a 2 < a 4

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) 
  (h : IsGeometric a) :
  (IsIncreasing a → Condition a) ∧ 
  ¬(Condition a → IsIncreasing a) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l3370_337081


namespace NUMINAMATH_CALUDE_smallest_M_bound_l3370_337050

theorem smallest_M_bound : ∃ (M : ℕ),
  (∀ (a b c : ℝ), (∀ (x : ℝ), |x| ≤ 1 → |a*x^2 + b*x + c| ≤ 1) →
    (∀ (x : ℝ), |x| ≤ 1 → |2*a*x + b| ≤ M)) ∧
  (∀ (N : ℕ), N < M →
    ∃ (a b c : ℝ), (∀ (x : ℝ), |x| ≤ 1 → |a*x^2 + b*x + c| ≤ 1) ∧
      (∃ (x : ℝ), |x| ≤ 1 ∧ |2*a*x + b| > N)) ∧
  M = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_M_bound_l3370_337050


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l3370_337085

theorem stratified_sampling_size (undergrads : ℕ) (masters : ℕ) (doctorates : ℕ) 
  (doctoral_sample : ℕ) (n : ℕ) : 
  undergrads = 12000 →
  masters = 1000 →
  doctorates = 200 →
  doctoral_sample = 20 →
  n = (undergrads + masters + doctorates) * doctoral_sample / doctorates →
  n = 1320 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_size_l3370_337085


namespace NUMINAMATH_CALUDE_remaining_region_area_l3370_337023

/-- Represents a rectangle divided into five regions -/
structure DividedRectangle where
  total_area : ℝ
  region1_area : ℝ
  region2_area : ℝ
  region3_area : ℝ
  region4_area : ℝ
  region5_area : ℝ
  area_sum : total_area = region1_area + region2_area + region3_area + region4_area + region5_area

/-- The theorem stating that one of the remaining regions has an area of 27 square units -/
theorem remaining_region_area (rect : DividedRectangle) 
    (h1 : rect.total_area = 72)
    (h2 : rect.region1_area = 15)
    (h3 : rect.region2_area = 12)
    (h4 : rect.region3_area = 18) :
    rect.region4_area = 27 ∨ rect.region5_area = 27 :=
  sorry

end NUMINAMATH_CALUDE_remaining_region_area_l3370_337023


namespace NUMINAMATH_CALUDE_class_size_is_40_l3370_337038

/-- Represents the heights of rectangles in a histogram --/
structure HistogramHeights where
  ratios : List Nat
  first_frequency : Nat

/-- Calculates the total number of students represented by a histogram --/
def totalStudents (h : HistogramHeights) : Nat :=
  let unit_frequency := h.first_frequency / h.ratios.head!
  unit_frequency * h.ratios.sum

/-- Theorem stating that for the given histogram, the total number of students is 40 --/
theorem class_size_is_40 (h : HistogramHeights) 
    (height_ratio : h.ratios = [4, 3, 7, 6]) 
    (first_freq : h.first_frequency = 8) : 
  totalStudents h = 40 := by
  sorry

#eval totalStudents { ratios := [4, 3, 7, 6], first_frequency := 8 }

end NUMINAMATH_CALUDE_class_size_is_40_l3370_337038


namespace NUMINAMATH_CALUDE_marathon_average_time_l3370_337017

/-- Calculates the average time per mile for a marathon --/
def average_time_per_mile (distance : ℕ) (hours : ℕ) (minutes : ℕ) : ℚ :=
  (hours * 60 + minutes : ℚ) / distance

/-- Theorem: The average time per mile for a 24-mile marathon completed in 3 hours and 36 minutes is 9 minutes --/
theorem marathon_average_time :
  average_time_per_mile 24 3 36 = 9 := by
  sorry

end NUMINAMATH_CALUDE_marathon_average_time_l3370_337017


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3370_337011

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel
  (α : Plane) (a b : Line)
  (ha : perpendicular a α)
  (hb : perpendicular b α) :
  parallel a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3370_337011


namespace NUMINAMATH_CALUDE_hurricane_damage_in_cad_l3370_337088

/-- Converts American dollars to Canadian dollars given a conversion rate -/
def convert_usd_to_cad (usd : ℝ) (rate : ℝ) : ℝ := usd * rate

/-- The damage caused by the hurricane in American dollars -/
def damage_usd : ℝ := 60000000

/-- The conversion rate from American dollars to Canadian dollars -/
def usd_to_cad_rate : ℝ := 1.25

/-- Theorem stating the equivalent damage in Canadian dollars -/
theorem hurricane_damage_in_cad :
  convert_usd_to_cad damage_usd usd_to_cad_rate = 75000000 := by
  sorry

end NUMINAMATH_CALUDE_hurricane_damage_in_cad_l3370_337088


namespace NUMINAMATH_CALUDE_master_title_possibilities_l3370_337056

/-- Represents a chess tournament with the given rules --/
structure ChessTournament where
  num_players : Nat
  points_for_win : Rat
  points_for_draw : Rat
  points_for_loss : Rat
  master_threshold : Rat

/-- Determines if it's possible for a given number of players to earn the Master of Sports title --/
def can_earn_master_title (t : ChessTournament) (num_masters : Nat) : Prop :=
  num_masters ≤ t.num_players ∧
  ∃ (point_distribution : Fin t.num_players → Rat),
    (∀ i, point_distribution i ≥ (t.num_players - 1 : Rat) * t.points_for_win * t.master_threshold) ∧
    (∀ i j, i ≠ j → point_distribution i + point_distribution j ≤ t.points_for_win)

/-- The specific tournament described in the problem --/
def tournament : ChessTournament :=
  { num_players := 12
  , points_for_win := 1
  , points_for_draw := 1/2
  , points_for_loss := 0
  , master_threshold := 7/10 }

theorem master_title_possibilities :
  (can_earn_master_title tournament 7) ∧
  ¬(can_earn_master_title tournament 8) := by sorry

end NUMINAMATH_CALUDE_master_title_possibilities_l3370_337056


namespace NUMINAMATH_CALUDE_number_ratio_l3370_337070

theorem number_ratio (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 143) (h4 : y = 104) :
  y / x = 8 / 3 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l3370_337070


namespace NUMINAMATH_CALUDE_min_value_of_function_l3370_337002

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  (x^2 + 3*x + 1) / x ≥ 5 ∧
  ((x^2 + 3*x + 1) / x = 5 ↔ x = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3370_337002


namespace NUMINAMATH_CALUDE_cubic_equation_root_squared_l3370_337061

theorem cubic_equation_root_squared (r : ℝ) : 
  r^3 - r + 3 = 0 → (r^2)^3 - 2*(r^2)^2 + r^2 - 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_root_squared_l3370_337061


namespace NUMINAMATH_CALUDE_dagger_example_l3370_337046

/-- The dagger operation on rational numbers -/
def dagger (a b : ℚ) : ℚ :=
  (a.num ^ 2 : ℚ) * b * (b.den : ℚ) / (a.den : ℚ)

/-- Theorem stating that 5/11 † 9/4 = 225/11 -/
theorem dagger_example : dagger (5 / 11) (9 / 4) = 225 / 11 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l3370_337046


namespace NUMINAMATH_CALUDE_equilateral_triangle_existence_l3370_337098

/-- Represents a color: Blue, White, or Red -/
inductive Color
| Blue
| White
| Red

/-- Represents a point in the triangle -/
structure Point where
  x : ℤ
  y : ℤ

/-- The set of points S satisfying the given conditions -/
def S (h : ℕ) : Set Point :=
  {p : Point | 0 ≤ p.x ∧ 0 ≤ p.y ∧ p.x + p.y ≤ h}

/-- A coloring function that respects the given constraints -/
def coloringFunction (h : ℕ) (p : Point) : Color :=
  sorry

/-- Checks if three points form an equilateral triangle with side length 1 -/
def isUnitEquilateralTriangle (p1 p2 p3 : Point) : Prop :=
  sorry

theorem equilateral_triangle_existence (h : ℕ) (h_pos : h > 0) :
  ∃ (p1 p2 p3 : Point),
    p1 ∈ S h ∧ p2 ∈ S h ∧ p3 ∈ S h ∧
    isUnitEquilateralTriangle p1 p2 p3 ∧
    coloringFunction h p1 ≠ coloringFunction h p2 ∧
    coloringFunction h p2 ≠ coloringFunction h p3 ∧
    coloringFunction h p3 ≠ coloringFunction h p1 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_existence_l3370_337098


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_n_equals_three_l3370_337041

/-- Two vectors in R² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a = (3, 2) and b = (2, n), if a is perpendicular to b, then n = 3 -/
theorem perpendicular_vectors_imply_n_equals_three :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ → ℝ × ℝ := λ n => (2, n)
  ∀ n : ℝ, perpendicular a (b n) → n = 3 := by
sorry


end NUMINAMATH_CALUDE_perpendicular_vectors_imply_n_equals_three_l3370_337041


namespace NUMINAMATH_CALUDE_independence_test_type_I_error_l3370_337019

/-- Represents the observed value of the χ² statistic -/
def k : ℝ := sorry

/-- Represents the probability of making a Type I error -/
def type_I_error_prob : ℝ → ℝ := sorry

/-- States that as k decreases, the probability of Type I error increases -/
theorem independence_test_type_I_error (h : k₁ < k₂) :
  type_I_error_prob k₁ > type_I_error_prob k₂ := by sorry

end NUMINAMATH_CALUDE_independence_test_type_I_error_l3370_337019


namespace NUMINAMATH_CALUDE_line_equation_through_points_l3370_337083

/-- The equation x - y + 1 = 0 represents the line passing through the points (-1, 0) and (0, 1) -/
theorem line_equation_through_points : 
  ∀ (x y : ℝ), (x = -1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) → x - y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l3370_337083


namespace NUMINAMATH_CALUDE_divisors_of_36_l3370_337048

/-- The number of integer divisors of 36 -/
def num_divisors_36 : ℕ := 18

/-- A function that counts the number of integer divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ i => n % i = 0) (Finset.range (n + 1))).card * 2

theorem divisors_of_36 :
  count_divisors 36 = num_divisors_36 := by
  sorry

#eval count_divisors 36

end NUMINAMATH_CALUDE_divisors_of_36_l3370_337048


namespace NUMINAMATH_CALUDE_xu_jun_current_age_l3370_337067

-- Define Xu Jun's current age
def xu_jun_age : ℕ := sorry

-- Define the teacher's current age
def teacher_age : ℕ := sorry

-- Condition 1: Two years ago, the teacher's age was 3 times Xu Jun's age
axiom condition1 : teacher_age - 2 = 3 * (xu_jun_age - 2)

-- Condition 2: In 8 years, the teacher's age will be twice Xu Jun's age
axiom condition2 : teacher_age + 8 = 2 * (xu_jun_age + 8)

-- Theorem to prove
theorem xu_jun_current_age : xu_jun_age = 12 := by sorry

end NUMINAMATH_CALUDE_xu_jun_current_age_l3370_337067


namespace NUMINAMATH_CALUDE_morgan_pens_count_l3370_337068

/-- The number of pens Morgan has -/
def total_pens (red blue black green purple : ℕ) : ℕ :=
  red + blue + black + green + purple

/-- Theorem: Morgan has 231 pens in total -/
theorem morgan_pens_count : total_pens 65 45 58 36 27 = 231 := by
  sorry

end NUMINAMATH_CALUDE_morgan_pens_count_l3370_337068


namespace NUMINAMATH_CALUDE_second_term_value_l3370_337082

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem second_term_value (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 2 + a 3 = 9) →
  a 1 = 1 →
  a 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_second_term_value_l3370_337082


namespace NUMINAMATH_CALUDE_cricketer_average_increase_l3370_337074

/-- Represents a cricketer's batting statistics -/
structure CricketerStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an additional inning -/
def newAverage (stats : CricketerStats) (newInningRuns : ℕ) : ℚ :=
  (stats.totalRuns + newInningRuns) / (stats.innings + 1)

/-- Theorem: If a cricketer's average increases by 8 after scoring 140 in the 15th inning, 
    the new average is 28 -/
theorem cricketer_average_increase 
  (stats : CricketerStats) 
  (h1 : stats.innings = 14)
  (h2 : newAverage stats 140 = stats.average + 8)
  : newAverage stats 140 = 28 := by
  sorry

#check cricketer_average_increase

end NUMINAMATH_CALUDE_cricketer_average_increase_l3370_337074


namespace NUMINAMATH_CALUDE_joes_investment_rate_l3370_337080

/-- Represents a simple interest bond investment -/
structure SimpleInterestBond where
  initialValue : ℝ
  interestRate : ℝ

/-- Calculates the value of a simple interest bond after a given number of years -/
def bondValue (bond : SimpleInterestBond) (years : ℝ) : ℝ :=
  bond.initialValue * (1 + bond.interestRate * years)

/-- Theorem: Given the conditions of Joe's investment, the interest rate is 1/13 -/
theorem joes_investment_rate : ∃ (bond : SimpleInterestBond),
  bondValue bond 3 = 260 ∧
  bondValue bond 8 = 360 ∧
  bond.interestRate = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_joes_investment_rate_l3370_337080


namespace NUMINAMATH_CALUDE_son_age_l3370_337014

theorem son_age (son_age man_age : ℕ) : 
  man_age = son_age + 24 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_son_age_l3370_337014


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_l3370_337006

def is_inverse_undefined (a n : ℕ) : Prop :=
  ¬ (∃ b : ℕ, a * b ≡ 1 [MOD n])

theorem smallest_undefined_inverse : 
  (∀ a : ℕ, 0 < a → a < 10 → 
    ¬(is_inverse_undefined a 55 ∧ is_inverse_undefined a 66)) ∧ 
  (is_inverse_undefined 10 55 ∧ is_inverse_undefined 10 66) := by
  sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_l3370_337006


namespace NUMINAMATH_CALUDE_susan_ate_six_candies_l3370_337090

/-- The number of candies Susan ate during the week -/
def candies_eaten (bought_tuesday bought_thursday bought_friday remaining : ℕ) : ℕ :=
  bought_tuesday + bought_thursday + bought_friday - remaining

/-- Theorem: Susan ate 6 candies during the week -/
theorem susan_ate_six_candies : candies_eaten 3 5 2 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_susan_ate_six_candies_l3370_337090


namespace NUMINAMATH_CALUDE_bus_line_count_l3370_337044

theorem bus_line_count (people_in_front people_behind : ℕ) 
  (h1 : people_in_front = 6) 
  (h2 : people_behind = 5) : 
  people_in_front + 1 + people_behind = 12 := by
  sorry

end NUMINAMATH_CALUDE_bus_line_count_l3370_337044


namespace NUMINAMATH_CALUDE_missing_number_in_mean_l3370_337060

theorem missing_number_in_mean (known_numbers : List ℤ) (mean : ℚ) : 
  known_numbers = [22, 23, 24, 25, 26, 27, 2] ∧ 
  mean = 20 ∧ 
  (List.sum known_numbers + (missing_number : ℤ)) / 7 = mean →
  missing_number = -9 :=
by
  sorry

end NUMINAMATH_CALUDE_missing_number_in_mean_l3370_337060


namespace NUMINAMATH_CALUDE_time_to_fill_cistern_l3370_337073

/-- Given a cistern that can be partially filled by a pipe, this theorem proves
    the time required to fill the entire cistern. -/
theorem time_to_fill_cistern (partial_fill_time : ℝ) (partial_fill_fraction : ℝ) 
    (h1 : partial_fill_time = 4)
    (h2 : partial_fill_fraction = 1 / 11) : 
  partial_fill_time / partial_fill_fraction = 44 := by
  sorry

#check time_to_fill_cistern

end NUMINAMATH_CALUDE_time_to_fill_cistern_l3370_337073


namespace NUMINAMATH_CALUDE_revenue_increase_80_percent_l3370_337016

/-- Represents the change in revenue given a price decrease and sales increase --/
def revenue_change (price_decrease : ℝ) (sales_increase_ratio : ℝ) : ℝ :=
  let new_price_factor := 1 - price_decrease
  let sales_increase := price_decrease * sales_increase_ratio
  let new_quantity_factor := 1 + sales_increase
  new_price_factor * new_quantity_factor - 1

/-- 
Theorem: Given a 10% price decrease and a sales increase ratio of 10,
the total revenue will increase by 80%
-/
theorem revenue_increase_80_percent :
  revenue_change 0.1 10 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_revenue_increase_80_percent_l3370_337016


namespace NUMINAMATH_CALUDE_contacts_in_sphere_tetrahedron_l3370_337013

/-- The number of contacts in a tetrahedral stack of spheres -/
def tetrahedron_contacts (n : ℕ) : ℕ := n^3 - n

/-- 
Theorem: In a tetrahedron formed by stacking identical spheres, 
where each edge has n spheres, the total number of points of 
tangency between the spheres is n³ - n.
-/
theorem contacts_in_sphere_tetrahedron (n : ℕ) : 
  tetrahedron_contacts n = n^3 - n := by
  sorry

end NUMINAMATH_CALUDE_contacts_in_sphere_tetrahedron_l3370_337013


namespace NUMINAMATH_CALUDE_sculpture_cost_in_rupees_l3370_337021

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_namibian : ℝ := 5

/-- Exchange rate from US dollars to Indian rupees -/
def usd_to_rupees : ℝ := 8

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_namibian : ℝ := 200

/-- Theorem stating the cost of the sculpture in Indian rupees -/
theorem sculpture_cost_in_rupees :
  (sculpture_cost_namibian / usd_to_namibian) * usd_to_rupees = 320 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_rupees_l3370_337021


namespace NUMINAMATH_CALUDE_daisies_per_bouquet_l3370_337000

/-- Represents a flower shop with rose and daisy bouquets -/
structure FlowerShop where
  roses_per_bouquet : ℕ
  total_bouquets : ℕ
  rose_bouquets : ℕ
  daisy_bouquets : ℕ
  total_flowers : ℕ

/-- Theorem stating the number of daisies in each daisy bouquet -/
theorem daisies_per_bouquet (shop : FlowerShop) 
  (h1 : shop.roses_per_bouquet = 12)
  (h2 : shop.total_bouquets = 20)
  (h3 : shop.rose_bouquets = 10)
  (h4 : shop.daisy_bouquets = 10)
  (h5 : shop.total_flowers = 190)
  (h6 : shop.rose_bouquets + shop.daisy_bouquets = shop.total_bouquets) :
  (shop.total_flowers - shop.roses_per_bouquet * shop.rose_bouquets) / shop.daisy_bouquets = 7 := by
  sorry

end NUMINAMATH_CALUDE_daisies_per_bouquet_l3370_337000


namespace NUMINAMATH_CALUDE_triangle_side_length_l3370_337029

open Real

theorem triangle_side_length 
  (g : ℝ → ℝ)
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : ∀ x, g x = cos (2 * x + π / 6))
  (h2 : (1/2) * b * c * sin A = 2)
  (h3 : b = 2)
  (h4 : g A = -1/2)
  (h5 : a < c) :
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3370_337029


namespace NUMINAMATH_CALUDE_greatest_solution_of_equation_l3370_337043

theorem greatest_solution_of_equation (x : ℝ) : 
  (((5*x - 20)/(4*x - 5))^2 + ((5*x - 20)/(4*x - 5)) = 20) → x ≤ 9/5 :=
by sorry

end NUMINAMATH_CALUDE_greatest_solution_of_equation_l3370_337043


namespace NUMINAMATH_CALUDE_cos_equality_proof_l3370_337007

theorem cos_equality_proof (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : Real.cos (n * π / 180) = Real.cos (317 * π / 180)) : n = 43 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_proof_l3370_337007


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3370_337078

theorem inequality_and_equality_condition (x y : ℝ) 
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  8 * x * y ≤ 5 * x * (1 - x) + 5 * y * (1 - y) ∧
  (8 * x * y = 5 * x * (1 - x) + 5 * y * (1 - y) ↔ x = 1/2 ∧ y = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3370_337078


namespace NUMINAMATH_CALUDE_expression_value_l3370_337012

theorem expression_value :
  let a : ℤ := 3
  let b : ℤ := 2
  let c : ℤ := 1
  (a + (b + c)^2) - ((a + b)^2 - c) = -12 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l3370_337012


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3370_337027

theorem arithmetic_calculation : 2546 + 240 / 60 - 346 = 2204 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3370_337027


namespace NUMINAMATH_CALUDE_squares_containing_a_l3370_337033

/-- Represents a square in a grid -/
structure Square where
  size : Nat
  contains_a : Bool

/-- Represents a 4x4 grid -/
def Grid := Array (Array Square)

/-- Creates a 4x4 grid with A in one cell -/
def create_grid : Grid := sorry

/-- Counts the total number of squares in the grid -/
def total_squares (grid : Grid) : Nat := sorry

/-- Counts the number of squares containing A -/
def squares_with_a (grid : Grid) : Nat := sorry

/-- Theorem stating that there are 13 squares containing A in a 4x4 grid with A in one cell -/
theorem squares_containing_a (grid : Grid) :
  total_squares grid = 20 → squares_with_a grid = 13 := by sorry

end NUMINAMATH_CALUDE_squares_containing_a_l3370_337033


namespace NUMINAMATH_CALUDE_initial_marbles_proof_l3370_337037

/-- The number of marbles Connie gave to Juan -/
def marbles_to_juan : ℕ := 73

/-- The number of marbles Connie gave to Maria -/
def marbles_to_maria : ℕ := 45

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 70

/-- The initial number of marbles Connie had -/
def initial_marbles : ℕ := marbles_to_juan + marbles_to_maria + marbles_left

theorem initial_marbles_proof : initial_marbles = 188 := by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_proof_l3370_337037


namespace NUMINAMATH_CALUDE_product_equality_l3370_337069

theorem product_equality (x y : ℤ) (h1 : 24 * x = 173 * y) (h2 : 173 * y = 1730) : y = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l3370_337069


namespace NUMINAMATH_CALUDE_flagpole_height_l3370_337094

/-- The height of a flagpole given specific measurements of surrounding stakes -/
theorem flagpole_height (AB OC OD OH : ℝ) (hAB : AB = 120) 
  (hHC : OH^2 + OC^2 = 170^2) (hHD : OH^2 + OD^2 = 100^2) (hCD : OC^2 + OD^2 = AB^2) :
  OH = 50 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_flagpole_height_l3370_337094


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3370_337086

/-- A rhombus with side length 65 and shorter diagonal 72 has a longer diagonal of 108 -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diag : ℝ) (longer_diag : ℝ) : 
  side = 65 → shorter_diag = 72 → longer_diag = 108 → 
  side^2 = (shorter_diag/2)^2 + (longer_diag/2)^2 := by
  sorry

#check rhombus_longer_diagonal

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3370_337086


namespace NUMINAMATH_CALUDE_max_diagonals_theorem_l3370_337079

/-- The maximum number of non-intersecting or perpendicular diagonals in a regular n-gon -/
def max_diagonals (n : ℕ) : ℕ :=
  if n % 2 = 0 then n - 2 else n - 3

/-- Theorem stating the maximum number of non-intersecting or perpendicular diagonals in a regular n-gon -/
theorem max_diagonals_theorem (n : ℕ) (h : n ≥ 3) :
  max_diagonals n = if n % 2 = 0 then n - 2 else n - 3 :=
by sorry

end NUMINAMATH_CALUDE_max_diagonals_theorem_l3370_337079


namespace NUMINAMATH_CALUDE_max_triangle_area_l3370_337045

theorem max_triangle_area (a b c : Real) (h1 : 0 < a) (h2 : a ≤ 1) (h3 : 1 ≤ b) 
  (h4 : b ≤ 2) (h5 : 2 ≤ c) (h6 : c ≤ 3) :
  ∃ (area : Real), area ≤ 1 ∧ 
    ∀ (A : Real), (∃ (α : Real), A = 1/2 * a * b * Real.sin α ∧ 
      a + b > c ∧ b + c > a ∧ c + a > b) → A ≤ area :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_area_l3370_337045


namespace NUMINAMATH_CALUDE_range_of_even_function_l3370_337071

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

theorem range_of_even_function (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) →
  (∀ x, x ∈ Set.Icc (a - 3) (2 * a) ↔ f a b x ≠ 0) →
  Set.range (f a b) = Set.Icc 3 7 := by
  sorry

#check range_of_even_function

end NUMINAMATH_CALUDE_range_of_even_function_l3370_337071


namespace NUMINAMATH_CALUDE_radio_contest_winner_l3370_337062

theorem radio_contest_winner (n : ℕ) 
  (h1 : 35 % 5 = 0)
  (h2 : 35 % n = 0)
  (h3 : ∀ m : ℕ, m > 0 ∧ m < 35 → ¬(m % 5 = 0 ∧ m % n = 0)) : 
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_radio_contest_winner_l3370_337062


namespace NUMINAMATH_CALUDE_problem_statement_l3370_337064

def p (m : ℝ) : Prop := ∀ x ∈ Set.Icc 0 2, m ≤ x^2 - 2*x

def q (m : ℝ) : Prop := ∃ x ≥ 0, 2^x + 3 = m

theorem problem_statement :
  (∀ m : ℝ, p m ↔ m ∈ Set.Iic (-1)) ∧
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ Set.Iic (-1) ∪ Set.Ici 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3370_337064


namespace NUMINAMATH_CALUDE_five_dice_not_same_l3370_337042

theorem five_dice_not_same (n : ℕ) (h : n = 8) : 
  (1 - (n : ℚ)/(n^5 : ℚ)) = 4095/4096 := by
  sorry

end NUMINAMATH_CALUDE_five_dice_not_same_l3370_337042


namespace NUMINAMATH_CALUDE_total_score_is_26_l3370_337087

-- Define the scores for Keith, Larry, and Danny
def keith_score : ℕ := 3
def larry_score : ℕ := 3 * keith_score
def danny_score : ℕ := larry_score + 5

-- Define the total score
def total_score : ℕ := keith_score + larry_score + danny_score

-- Theorem to prove
theorem total_score_is_26 : total_score = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_score_is_26_l3370_337087


namespace NUMINAMATH_CALUDE_sisters_contribution_l3370_337022

/-- The amount of money Miranda's sister gave her to buy heels -/
theorem sisters_contribution (months_saved : ℕ) (monthly_savings : ℕ) (total_cost : ℕ) : 
  months_saved = 3 → monthly_savings = 70 → total_cost = 260 →
  total_cost - (months_saved * monthly_savings) = 50 := by
  sorry

end NUMINAMATH_CALUDE_sisters_contribution_l3370_337022


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l3370_337065

theorem rectangular_box_volume (x : ℕ) (h : x > 0) :
  (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
   a = x ∧ b = 3*x ∧ c = 4*x ∧
   a * b * c = 96) ↔ x = 2 :=
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l3370_337065


namespace NUMINAMATH_CALUDE_work_completion_time_l3370_337001

theorem work_completion_time (a b t : ℝ) (ha : a > 0) (hb : b > 0) (ht : t > 0) :
  1 / a + 1 / b = 1 / t → b = 18 → t = 7.2 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3370_337001


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l3370_337075

theorem fruit_seller_apples (initial_apples : ℕ) : 
  (initial_apples : ℝ) * 0.6 = 420 → initial_apples = 700 :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l3370_337075


namespace NUMINAMATH_CALUDE_babylonian_conversion_l3370_337030

/-- Converts a Babylonian sexagesimal number to its decimal representation -/
def babylonian_to_decimal (a b : ℕ) : ℕ :=
  60^a + 10 * 60^b

/-- The Babylonian number 60^8 + 10 * 60^7 in decimal form -/
theorem babylonian_conversion :
  babylonian_to_decimal 8 7 = 195955200000000 := by
  sorry

#eval babylonian_to_decimal 8 7

end NUMINAMATH_CALUDE_babylonian_conversion_l3370_337030


namespace NUMINAMATH_CALUDE_parabola_coefficients_l3370_337055

/-- A parabola with vertex (h, k) passing through point (x₀, y₀) has equation y = a(x - h)² + k -/
def is_parabola (a h k x₀ y₀ : ℝ) : Prop :=
  y₀ = a * (x₀ - h)^2 + k

/-- The general form of a parabola y = ax² + bx + c can be derived from the vertex form -/
def general_form (a h k : ℝ) : ℝ × ℝ × ℝ :=
  (a, -2*a*h, a*h^2 + k)

theorem parabola_coefficients :
  ∀ (a : ℝ), is_parabola a 4 (-1) 2 3 →
  general_form a 4 (-1) = (1, -8, 15) := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l3370_337055


namespace NUMINAMATH_CALUDE_girls_in_class_l3370_337058

theorem girls_in_class (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 70 → 
  4 * boys = 3 * girls → 
  total = girls + boys → 
  girls = 40 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l3370_337058


namespace NUMINAMATH_CALUDE_henry_total_games_l3370_337084

def wins : ℕ := 2
def losses : ℕ := 2
def draws : ℕ := 10

theorem henry_total_games : wins + losses + draws = 14 := by
  sorry

end NUMINAMATH_CALUDE_henry_total_games_l3370_337084


namespace NUMINAMATH_CALUDE_camera_pictures_l3370_337059

def picture_problem (total_albums : ℕ) (pics_per_album : ℕ) (pics_from_phone : ℕ) : Prop :=
  let total_pics := total_albums * pics_per_album
  total_pics - pics_from_phone = 13

theorem camera_pictures :
  picture_problem 5 4 7 := by
  sorry

end NUMINAMATH_CALUDE_camera_pictures_l3370_337059


namespace NUMINAMATH_CALUDE_sum_of_products_inequality_l3370_337015

theorem sum_of_products_inequality (a b c d : ℝ) (h : a + b + c + d = 1) :
  a * b + b * c + c * d + d * a ≤ 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_inequality_l3370_337015


namespace NUMINAMATH_CALUDE_opposite_of_2023_l3370_337008

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l3370_337008


namespace NUMINAMATH_CALUDE_cream_ratio_proof_l3370_337077

-- Define the given constants
def servings : ℕ := 4
def fat_per_cup : ℕ := 88
def fat_per_serving : ℕ := 11

-- Define the ratio we want to prove
def cream_ratio : ℚ := 1 / 2

-- Theorem statement
theorem cream_ratio_proof :
  (servings * fat_per_serving : ℚ) / fat_per_cup = cream_ratio := by
  sorry

end NUMINAMATH_CALUDE_cream_ratio_proof_l3370_337077


namespace NUMINAMATH_CALUDE_exists_diff_same_num_prime_divisors_l3370_337003

/-- The number of distinct prime divisors of a natural number -/
def numDistinctPrimeDivisors (n : ℕ) : ℕ := sorry

/-- For any natural number n, there exist natural numbers a and b such that
    n = a - b and they have the same number of distinct prime divisors -/
theorem exists_diff_same_num_prime_divisors (n : ℕ) :
  ∃ a b : ℕ, n = a - b ∧ numDistinctPrimeDivisors a = numDistinctPrimeDivisors b := by
  sorry

end NUMINAMATH_CALUDE_exists_diff_same_num_prime_divisors_l3370_337003


namespace NUMINAMATH_CALUDE_marble_box_problem_l3370_337089

theorem marble_box_problem :
  ∀ (red blue : ℕ),
  red = blue →
  20 + red + blue - 2 * (20 - blue) = 40 →
  20 + red + blue = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_box_problem_l3370_337089


namespace NUMINAMATH_CALUDE_lcm_12_18_l3370_337099

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_18_l3370_337099


namespace NUMINAMATH_CALUDE_steve_salary_calculation_l3370_337096

def steve_take_home_pay (salary : ℝ) (tax_rate : ℝ) (healthcare_rate : ℝ) (union_dues : ℝ) : ℝ :=
  salary - (salary * tax_rate) - (salary * healthcare_rate) - union_dues

theorem steve_salary_calculation :
  steve_take_home_pay 40000 0.20 0.10 800 = 27200 := by
  sorry

end NUMINAMATH_CALUDE_steve_salary_calculation_l3370_337096


namespace NUMINAMATH_CALUDE_zongzi_profit_maximization_l3370_337054

/-- Problem statement for zongzi profit maximization --/
theorem zongzi_profit_maximization 
  (cost_A cost_B : ℚ)  -- Cost prices of type A and B zongzi
  (sell_A sell_B : ℚ)  -- Selling prices of type A and B zongzi
  (total : ℕ)          -- Total number of zongzi to purchase
  :
  (cost_B = cost_A + 2) →  -- Condition 1
  (1000 / cost_A = 1200 / cost_B) →  -- Condition 2
  (sell_A = 12) →  -- Condition 5
  (sell_B = 15) →  -- Condition 6
  (total = 200) →  -- Condition 3
  ∃ (m : ℕ),  -- Number of type A zongzi purchased
    (m ≥ 2 * (total - m)) ∧  -- Condition 4
    (m < total) ∧
    (∀ (n : ℕ), n ≥ 2 * (total - n) → n < total →
      (sell_A - cost_A) * m + (sell_B - cost_B) * (total - m) ≥
      (sell_A - cost_A) * n + (sell_B - cost_B) * (total - n)) ∧
    ((sell_A - cost_A) * m + (sell_B - cost_B) * (total - m) = 466) ∧
    (m = 134) :=
by sorry

end NUMINAMATH_CALUDE_zongzi_profit_maximization_l3370_337054


namespace NUMINAMATH_CALUDE_apollonius_circle_minimum_l3370_337034

/-- Given points A, B, D, and a moving point P in a 2D plane, 
    prove that the minimum value of 2|PD|+|PB| is 2√10 when |PA|/|PB| = 1/2 -/
theorem apollonius_circle_minimum (A B D P : EuclideanSpace ℝ (Fin 2)) :
  A = ![(1 : ℝ), 0] →
  B = ![4, 0] →
  D = ![0, 3] →
  dist P A / dist P B = (1 : ℝ) / 2 →
  ∃ (P : EuclideanSpace ℝ (Fin 2)), 2 * dist P D + dist P B ≥ 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_apollonius_circle_minimum_l3370_337034


namespace NUMINAMATH_CALUDE_max_side_length_l3370_337040

/-- A triangle with three different integer side lengths and a perimeter of 20 units -/
structure TriangleWithConstraints where
  a : ℕ
  b : ℕ
  c : ℕ
  different : a ≠ b ∧ b ≠ c ∧ a ≠ c
  perimeter : a + b + c = 20

/-- The maximum length of any side in a TriangleWithConstraints is 9 -/
theorem max_side_length (t : TriangleWithConstraints) :
  max t.a (max t.b t.c) = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_side_length_l3370_337040


namespace NUMINAMATH_CALUDE_sum_seven_more_likely_than_eight_l3370_337066

def dice_sum_probability (sum : Nat) : Rat :=
  (Finset.filter (fun (x, y) => x + y = sum) (Finset.product (Finset.range 6) (Finset.range 6))).card / 36

theorem sum_seven_more_likely_than_eight :
  dice_sum_probability 7 > dice_sum_probability 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_more_likely_than_eight_l3370_337066


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3370_337097

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + m - 2 = 0 ∧ x = -3) → 
  (m = -1 ∧ ∃ y : ℝ, y^2 + 2*y + m - 2 = 0 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3370_337097


namespace NUMINAMATH_CALUDE_sale_price_ratio_l3370_337024

theorem sale_price_ratio (c x y : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) :
  y / x = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sale_price_ratio_l3370_337024


namespace NUMINAMATH_CALUDE_sqrt_product_l3370_337076

theorem sqrt_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_l3370_337076


namespace NUMINAMATH_CALUDE_commission_percentage_proof_l3370_337091

def commission_percentage_below_threshold (total_sales : ℚ) (threshold : ℚ) (remitted_amount : ℚ) (commission_percentage_above_threshold : ℚ) : ℚ :=
  let sales_above_threshold := total_sales - threshold
  let commission_above_threshold := sales_above_threshold * commission_percentage_above_threshold / 100
  let total_commission := total_sales - remitted_amount
  let commission_below_threshold := total_commission - commission_above_threshold
  commission_below_threshold / threshold * 100

theorem commission_percentage_proof :
  let total_sales : ℚ := 32500
  let threshold : ℚ := 10000
  let remitted_amount : ℚ := 31100
  let commission_percentage_above_threshold : ℚ := 4
  commission_percentage_below_threshold total_sales threshold remitted_amount commission_percentage_above_threshold = 5 := by
  sorry

#eval commission_percentage_below_threshold 32500 10000 31100 4

end NUMINAMATH_CALUDE_commission_percentage_proof_l3370_337091


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3370_337051

-- Define the function f(x) = x³ + 1
def f (x : ℝ) : ℝ := x^3 + 1

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 3 * x^2

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_derivative x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ 3 * x - y - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3370_337051


namespace NUMINAMATH_CALUDE_ratio_is_zero_l3370_337004

-- Define the rectangle JKLM
def Rectangle (J L M : ℝ × ℝ) : Prop :=
  (L.1 - J.1 = 8) ∧ (L.2 - M.2 = 6) ∧ (J.2 = L.2) ∧ (J.1 = M.1)

-- Define points N, P, Q
def PointN (J L N : ℝ × ℝ) : Prop :=
  N.2 = J.2 ∧ L.1 - N.1 = 2

def PointP (L M P : ℝ × ℝ) : Prop :=
  P.1 = L.1 ∧ M.2 - P.2 = 2

def PointQ (M J Q : ℝ × ℝ) : Prop :=
  Q.2 = J.2 ∧ M.1 - Q.1 = 3

-- Define the intersection points R and S
def IntersectionR (J P N Q R : ℝ × ℝ) : Prop :=
  (R.2 - J.2) / (R.1 - J.1) = (P.2 - J.2) / (P.1 - J.1) ∧
  R.2 = N.2

def IntersectionS (J M N Q S : ℝ × ℝ) : Prop :=
  (S.2 - J.2) / (S.1 - J.1) = (M.2 - J.2) / (M.1 - J.1) ∧
  S.2 = N.2

-- Theorem statement
theorem ratio_is_zero
  (J K L M N P Q R S : ℝ × ℝ)
  (h_rect : Rectangle J L M)
  (h_N : PointN J L N)
  (h_P : PointP L M P)
  (h_Q : PointQ M J Q)
  (h_R : IntersectionR J P N Q R)
  (h_S : IntersectionS J M N Q S) :
  (R.1 - S.1) / (N.1 - Q.1) = 0 :=
sorry

end NUMINAMATH_CALUDE_ratio_is_zero_l3370_337004


namespace NUMINAMATH_CALUDE_cheese_bread_solution_l3370_337036

/-- Represents the problem of buying cheese bread for a group of people. -/
structure CheeseBreadProblem where
  cost_per_100g : ℚ  -- Cost in R$ per 100g of cheese bread
  pieces_per_100g : ℕ  -- Number of pieces in 100g of cheese bread
  pieces_per_person : ℕ  -- Average number of pieces eaten per person
  total_people : ℕ  -- Total number of people
  scale_precision : ℕ  -- Precision of the bakery's scale in grams

/-- Calculates the amount to buy, cost, and leftover pieces for a given CheeseBreadProblem. -/
def solve_cheese_bread_problem (p : CheeseBreadProblem) :
  (ℕ × ℚ × ℕ) :=
  sorry

/-- Theorem stating the correct solution for the given problem. -/
theorem cheese_bread_solution :
  let problem := CheeseBreadProblem.mk 3.2 10 5 23 100
  let (amount, cost, leftover) := solve_cheese_bread_problem problem
  amount = 1200 ∧ cost = 38.4 ∧ leftover = 5 :=
sorry

end NUMINAMATH_CALUDE_cheese_bread_solution_l3370_337036


namespace NUMINAMATH_CALUDE_pentagon_perimeter_eq_sum_of_coefficients_l3370_337035

/-- The perimeter of a pentagon with specified vertices -/
def pentagon_perimeter : ℝ := sorry

/-- Theorem stating that the perimeter of the specified pentagon equals 2 + 2√10 -/
theorem pentagon_perimeter_eq : pentagon_perimeter = 2 + 2 * Real.sqrt 10 := by sorry

/-- Corollary showing that when expressed as p + q√10 + r√13, p + q + r = 4 -/
theorem sum_of_coefficients : ∃ (p q r : ℤ), 
  pentagon_perimeter = ↑p + ↑q * Real.sqrt 10 + ↑r * Real.sqrt 13 ∧ p + q + r = 4 := by sorry

end NUMINAMATH_CALUDE_pentagon_perimeter_eq_sum_of_coefficients_l3370_337035


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3370_337047

/-- Tetrahedron PQRS with given edge lengths -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- The volume of a tetrahedron given its edge lengths -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem: The volume of the specific tetrahedron is 10.25 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 4,
    PR := 5,
    PS := 6,
    QR := 3,
    QS := Real.sqrt 37,
    RS := 7
  }
  volume t = 10.25 := by sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3370_337047


namespace NUMINAMATH_CALUDE_initial_concentration_proof_l3370_337049

theorem initial_concentration_proof (volume_replaced : ℝ) 
  (replacement_concentration : ℝ) (final_concentration : ℝ) :
  volume_replaced = 0.7142857142857143 →
  replacement_concentration = 0.25 →
  final_concentration = 0.35 →
  ∃ initial_concentration : ℝ,
    initial_concentration = 0.6 ∧
    (1 - volume_replaced) * initial_concentration + 
      volume_replaced * replacement_concentration = final_concentration :=
by
  sorry

end NUMINAMATH_CALUDE_initial_concentration_proof_l3370_337049


namespace NUMINAMATH_CALUDE_pump_emptying_time_l3370_337072

theorem pump_emptying_time (time_B time_together : ℝ) 
  (hB : time_B = 6)
  (hTogether : time_together = 2.4)
  (h_rate : 1 / time_A + 1 / time_B = 1 / time_together) :
  time_A = 4 := by
  sorry

end NUMINAMATH_CALUDE_pump_emptying_time_l3370_337072
