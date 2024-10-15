import Mathlib

namespace NUMINAMATH_CALUDE_restaurant_pie_days_l880_88079

/-- Given a restaurant that sells a constant number of pies per day,
    calculate the number of days based on the total pies sold. -/
theorem restaurant_pie_days (pies_per_day : ℕ) (total_pies : ℕ) (h1 : pies_per_day = 8) (h2 : total_pies = 56) :
  total_pies / pies_per_day = 7 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_pie_days_l880_88079


namespace NUMINAMATH_CALUDE_school_departments_l880_88028

/-- Given a school with departments where each department has 20 teachers and there are 140 teachers in total, prove that the number of departments is 7. -/
theorem school_departments (total_teachers : ℕ) (teachers_per_dept : ℕ) (h1 : total_teachers = 140) (h2 : teachers_per_dept = 20) :
  total_teachers / teachers_per_dept = 7 := by
  sorry

end NUMINAMATH_CALUDE_school_departments_l880_88028


namespace NUMINAMATH_CALUDE_jason_potato_eating_time_l880_88087

/-- Given that Jason eats 27 potatoes in 3 hours, prove that it takes him 20 minutes to eat 3 potatoes. -/
theorem jason_potato_eating_time :
  ∀ (total_potatoes total_hours potatoes_to_eat : ℕ) (minutes_per_hour : ℕ),
    total_potatoes = 27 →
    total_hours = 3 →
    potatoes_to_eat = 3 →
    minutes_per_hour = 60 →
    (potatoes_to_eat * total_hours * minutes_per_hour) / total_potatoes = 20 := by
  sorry

end NUMINAMATH_CALUDE_jason_potato_eating_time_l880_88087


namespace NUMINAMATH_CALUDE_paperboy_delivery_12_l880_88040

def paperboy_delivery (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | m + 4 => paperboy_delivery m + paperboy_delivery (m + 1) + paperboy_delivery (m + 2) + paperboy_delivery (m + 3)

theorem paperboy_delivery_12 : paperboy_delivery 12 = 2873 := by
  sorry

end NUMINAMATH_CALUDE_paperboy_delivery_12_l880_88040


namespace NUMINAMATH_CALUDE_floor_sum_eval_l880_88017

theorem floor_sum_eval : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_eval_l880_88017


namespace NUMINAMATH_CALUDE_polynomial_factor_l880_88042

-- Define the polynomials
def P (c : ℝ) (x : ℝ) : ℝ := 3 * x^3 + c * x + 12
def Q (q : ℝ) (x : ℝ) : ℝ := x^2 + q * x + 2

-- Theorem statement
theorem polynomial_factor (c : ℝ) :
  (∃ q r : ℝ, ∀ x : ℝ, P c x = Q q x * (r * x + (12 / r))) → c = 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l880_88042


namespace NUMINAMATH_CALUDE_quadrilaterals_with_fixed_point_l880_88000

theorem quadrilaterals_with_fixed_point (n : ℕ) (k : ℕ) :
  n = 11 ∧ k = 3 → Nat.choose n k = 165 := by sorry

end NUMINAMATH_CALUDE_quadrilaterals_with_fixed_point_l880_88000


namespace NUMINAMATH_CALUDE_complement_of_intersection_l880_88081

/-- The universal set U -/
def U : Set Nat := {0, 1, 2, 3}

/-- The set M -/
def M : Set Nat := {0, 1, 2}

/-- The set N -/
def N : Set Nat := {1, 2, 3}

/-- Theorem stating that the complement of M ∩ N in U is {0, 3} -/
theorem complement_of_intersection (U M N : Set Nat) (hU : U = {0, 1, 2, 3}) (hM : M = {0, 1, 2}) (hN : N = {1, 2, 3}) :
  (M ∩ N)ᶜ = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l880_88081


namespace NUMINAMATH_CALUDE_greatest_k_value_l880_88008

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 85) →
  k ≤ Real.sqrt 117 :=
sorry

end NUMINAMATH_CALUDE_greatest_k_value_l880_88008


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l880_88062

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

def has_additional_prime_factors (n : ℕ) (count : ℕ) : Prop :=
  ∃ (factors : List ℕ), 
    factors.length = count ∧ 
    (∀ p ∈ factors, Nat.Prime p) ∧
    (∀ p ∈ factors, p ∣ n) ∧
    (∀ p ∈ factors, p ≠ 2 ∧ p ≠ 5)

theorem smallest_number_with_conditions : 
  (∀ n : ℕ, n < 840 → 
    ¬(is_divisible_by n 8 ∧ 
      is_divisible_by n 5 ∧ 
      has_additional_prime_factors n 2)) ∧
  (is_divisible_by 840 8 ∧ 
   is_divisible_by 840 5 ∧ 
   has_additional_prime_factors 840 2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l880_88062


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l880_88094

theorem simplify_and_evaluate (x y : ℚ) 
  (hx : x = 1) (hy : y = 1/2) : 
  (3*x + 2*y) * (3*x - 2*y) - (x - y)^2 = 31/4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l880_88094


namespace NUMINAMATH_CALUDE_a_ln_a_gt_b_ln_b_l880_88091

theorem a_ln_a_gt_b_ln_b (a b : ℝ) (h1 : a > b) (h2 : b > 1) : a * Real.log a > b * Real.log b := by
  sorry

end NUMINAMATH_CALUDE_a_ln_a_gt_b_ln_b_l880_88091


namespace NUMINAMATH_CALUDE_passes_through_origin_parallel_to_line_intersects_below_l880_88054

/-- Linear function definition -/
def linear_function (m x : ℝ) : ℝ := (2*m + 1)*x + m - 3

/-- Theorem for when the function passes through the origin -/
theorem passes_through_origin (m : ℝ) : 
  linear_function m 0 = 0 ↔ m = 3 := by sorry

/-- Theorem for when the function is parallel to y = 3x - 3 -/
theorem parallel_to_line (m : ℝ) :
  (2*m + 1 = 3) ↔ m = 1 := by sorry

/-- Theorem for when the function intersects y-axis below x-axis -/
theorem intersects_below (m : ℝ) :
  (linear_function m 0 < 0 ∧ 2*m + 1 ≠ 0) ↔ (m < 3 ∧ m ≠ -1/2) := by sorry

end NUMINAMATH_CALUDE_passes_through_origin_parallel_to_line_intersects_below_l880_88054


namespace NUMINAMATH_CALUDE_point_in_region_l880_88047

def satisfies_inequality (x y : ℝ) : Prop := 3 + 2*y < 6

theorem point_in_region :
  satisfies_inequality 1 1 ∧
  ¬(satisfies_inequality 0 0 ∧ satisfies_inequality 0 2 ∧ satisfies_inequality 2 0) :=
by sorry

end NUMINAMATH_CALUDE_point_in_region_l880_88047


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l880_88085

theorem complex_modulus_problem (i : ℂ) (h : i^2 = -1) :
  Complex.abs (i / (1 + i^3)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l880_88085


namespace NUMINAMATH_CALUDE_election_majority_proof_l880_88023

theorem election_majority_proof (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 440 →
  winning_percentage = 70 / 100 →
  (winning_percentage * total_votes : ℚ).num - ((1 - winning_percentage) * total_votes : ℚ).num = 176 := by
  sorry

end NUMINAMATH_CALUDE_election_majority_proof_l880_88023


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l880_88004

/-- Theater ticket sales problem -/
theorem theater_ticket_sales
  (orchestra_price : ℕ)
  (balcony_price : ℕ)
  (total_tickets : ℕ)
  (balcony_orchestra_diff : ℕ)
  (ho : orchestra_price = 12)
  (hb : balcony_price = 8)
  (ht : total_tickets = 340)
  (hd : balcony_orchestra_diff = 40) :
  let orchestra_tickets := (total_tickets - balcony_orchestra_diff) / 2
  let balcony_tickets := total_tickets - orchestra_tickets
  orchestra_tickets * orchestra_price + balcony_tickets * balcony_price = 3320 :=
by sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l880_88004


namespace NUMINAMATH_CALUDE_disrespectful_polynomial_max_value_l880_88078

/-- A quadratic polynomial with real coefficients and leading coefficient 1 -/
structure QuadraticPolynomial where
  b : ℝ
  c : ℝ

/-- Evaluation of a quadratic polynomial at a point x -/
def evaluate (q : QuadraticPolynomial) (x : ℝ) : ℝ :=
  x^2 + q.b * x + q.c

/-- A quadratic polynomial is disrespectful if q(q(x)) = 0 has exactly three distinct real roots -/
def isDisrespectful (q : QuadraticPolynomial) : Prop :=
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    evaluate q (evaluate q x) = 0 ∧
    evaluate q (evaluate q y) = 0 ∧
    evaluate q (evaluate q z) = 0 ∧
    ∀ w : ℝ, evaluate q (evaluate q w) = 0 → w = x ∨ w = y ∨ w = z

theorem disrespectful_polynomial_max_value :
  ∀ q : QuadraticPolynomial, isDisrespectful q → evaluate q 2 ≤ 45/16 :=
sorry

end NUMINAMATH_CALUDE_disrespectful_polynomial_max_value_l880_88078


namespace NUMINAMATH_CALUDE_pentagon_side_length_l880_88010

/-- A five-sided figure with equal side lengths -/
structure Pentagon where
  side_length : ℝ
  perimeter : ℝ
  side_count : ℕ := 5
  all_sides_equal : perimeter = side_count * side_length

/-- Theorem: Given a pentagon with perimeter 23.4 cm, the length of one side is 4.68 cm -/
theorem pentagon_side_length (p : Pentagon) (h : p.perimeter = 23.4) : p.side_length = 4.68 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_side_length_l880_88010


namespace NUMINAMATH_CALUDE_price_increase_percentage_l880_88039

theorem price_increase_percentage (x : ℝ) : 
  (∀ (P : ℝ), P > 0 → P * (1 + x / 100) * (1 - 20 / 100) = P) → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l880_88039


namespace NUMINAMATH_CALUDE_right_triangle_inequalities_l880_88045

-- Define a structure for a right-angled triangle with height to hypotenuse
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ
  right_angle : a^2 + b^2 = c^2
  height_def : 2 * h * c = a * b

theorem right_triangle_inequalities (t : RightTriangle) :
  (t.a^2 + t.b^2 < t.c^2 + t.h^2) ∧ (t.a^4 + t.b^4 < t.c^4 + t.h^4) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_inequalities_l880_88045


namespace NUMINAMATH_CALUDE_moles_of_CO₂_equals_two_l880_88083

/-- Represents a chemical compound --/
inductive Compound
| HNO₃
| NaHCO₃
| NH₄Cl
| NaNO₃
| H₂O
| CO₂
| NH₄NO₃
| HCl

/-- Represents a chemical reaction --/
structure Reaction :=
(reactants : List (Compound × ℕ))
(products : List (Compound × ℕ))

/-- Represents the two-step reaction process --/
def two_step_reaction : List Reaction :=
[
  { reactants := [(Compound.NaHCO₃, 1), (Compound.HNO₃, 1)],
    products := [(Compound.NaNO₃, 1), (Compound.H₂O, 1), (Compound.CO₂, 1)] },
  { reactants := [(Compound.NH₄Cl, 1), (Compound.HNO₃, 1)],
    products := [(Compound.NH₄NO₃, 1), (Compound.HCl, 1)] }
]

/-- Initial amounts of compounds --/
def initial_amounts : List (Compound × ℕ) :=
[(Compound.HNO₃, 2), (Compound.NaHCO₃, 2), (Compound.NH₄Cl, 1)]

/-- Calculates the moles of CO₂ formed in the two-step reaction --/
def moles_of_CO₂_formed (reactions : List Reaction) (initial : List (Compound × ℕ)) : ℕ :=
sorry

/-- Theorem stating that the moles of CO₂ formed is 2 --/
theorem moles_of_CO₂_equals_two :
  moles_of_CO₂_formed two_step_reaction initial_amounts = 2 :=
sorry

end NUMINAMATH_CALUDE_moles_of_CO₂_equals_two_l880_88083


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_minus_b_zero_l880_88030

-- Define the curve
def curve (a b x : ℝ) : ℝ := x^2 + a*x + b

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Theorem statement
theorem tangent_line_implies_a_minus_b_zero (a b : ℝ) :
  (∃ x y, tangent_line x y ∧ y = curve a b x) →
  (tangent_line 0 b) →
  a - b = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_minus_b_zero_l880_88030


namespace NUMINAMATH_CALUDE_quadratic_root_m_value_l880_88050

theorem quadratic_root_m_value :
  ∀ m : ℝ, (1 : ℝ)^2 + m * 1 + 2 = 0 → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_m_value_l880_88050


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l880_88051

theorem quadratic_roots_sum_of_squares (p q r s : ℝ) : 
  (∀ x, x^2 - 2*p*x + 3*q = 0 ↔ x = r ∨ x = s) → 
  r^2 + s^2 = 4*p^2 - 6*q := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l880_88051


namespace NUMINAMATH_CALUDE_no_extreme_points_iff_l880_88097

/-- A cubic function parameterized by a real number a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * a * x^2 + (a + 1) * x

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 4 * a * x + (a + 1)

/-- The discriminant of f_deriv -/
def discriminant (a : ℝ) : ℝ := 4 * a^2 - 12 * a

/-- Theorem stating that f has no extreme points iff 0 ≤ a ≤ 3 -/
theorem no_extreme_points_iff (a : ℝ) :
  (∀ x : ℝ, f_deriv a x ≠ 0) ↔ 0 ≤ a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_no_extreme_points_iff_l880_88097


namespace NUMINAMATH_CALUDE_share_difference_l880_88056

/-- Given a distribution ratio and Vasim's share, calculate the difference between Ranjith's and Faruk's shares -/
theorem share_difference (faruk_ratio vasim_ratio ranjith_ratio vasim_share : ℕ) : 
  faruk_ratio = 3 → 
  vasim_ratio = 3 → 
  ranjith_ratio = 7 → 
  vasim_share = 1500 → 
  (ranjith_ratio * vasim_share / vasim_ratio) - (faruk_ratio * vasim_share / vasim_ratio) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_share_difference_l880_88056


namespace NUMINAMATH_CALUDE_water_added_amount_l880_88018

def initial_volume : ℝ := 340
def initial_water_percent : ℝ := 0.75
def initial_kola_percent : ℝ := 0.05
def initial_sugar_percent : ℝ := 1 - initial_water_percent - initial_kola_percent
def added_sugar : ℝ := 3.2
def added_kola : ℝ := 6.8
def final_sugar_percent : ℝ := 0.1966850828729282

theorem water_added_amount (added_water : ℝ) : 
  let initial_sugar := initial_volume * initial_sugar_percent
  let total_sugar := initial_sugar + added_sugar
  let final_volume := initial_volume + added_sugar + added_kola + added_water
  total_sugar / final_volume = final_sugar_percent →
  added_water = 12 := by sorry

end NUMINAMATH_CALUDE_water_added_amount_l880_88018


namespace NUMINAMATH_CALUDE_syllogism_conclusion_l880_88086

-- Define the sets
variable (U : Type) -- Universe set
variable (Mem : Set U) -- Set of Mems
variable (En : Set U) -- Set of Ens
variable (Veen : Set U) -- Set of Veens

-- Define the hypotheses
variable (h1 : Mem ⊆ En) -- All Mems are Ens
variable (h2 : ∃ x, x ∈ En ∩ Veen) -- Some Ens are Veens

-- Define the conclusions to be proved
def conclusion1 : Prop := ∃ x, x ∈ Mem ∩ Veen -- Some Mems are Veens
def conclusion2 : Prop := ∃ x, x ∈ Veen \ Mem -- Some Veens are not Mems

-- Theorem statement
theorem syllogism_conclusion (U : Type) (Mem En Veen : Set U) 
  (h1 : Mem ⊆ En) (h2 : ∃ x, x ∈ En ∩ Veen) : 
  conclusion1 U Mem Veen ∧ conclusion2 U Mem Veen := by
  sorry

end NUMINAMATH_CALUDE_syllogism_conclusion_l880_88086


namespace NUMINAMATH_CALUDE_inequality_equivalence_l880_88044

theorem inequality_equivalence (x : ℝ) : 3 * x + 4 < 5 * x - 6 ↔ x > 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l880_88044


namespace NUMINAMATH_CALUDE_median_and_mode_are_50_l880_88060

/-- Represents a speed measurement and its frequency --/
structure SpeedData where
  speed : ℕ
  frequency : ℕ

/-- The dataset of vehicle speeds and their frequencies --/
def speedDataset : List SpeedData := [
  ⟨48, 5⟩,
  ⟨49, 4⟩,
  ⟨50, 8⟩,
  ⟨51, 2⟩,
  ⟨52, 1⟩
]

/-- Calculates the median of the dataset --/
def calculateMedian (data : List SpeedData) : ℕ := sorry

/-- Calculates the mode of the dataset --/
def calculateMode (data : List SpeedData) : ℕ := sorry

/-- Theorem stating that the median and mode of the dataset are both 50 --/
theorem median_and_mode_are_50 :
  calculateMedian speedDataset = 50 ∧ calculateMode speedDataset = 50 := by sorry

end NUMINAMATH_CALUDE_median_and_mode_are_50_l880_88060


namespace NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l880_88075

/-- 
Given a real number m < 1, prove that the complex number 1 + (m-1)i 
is located in the fourth quadrant of the complex plane.
-/
theorem complex_in_fourth_quadrant (m : ℝ) (h : m < 1) : 
  let z : ℂ := 1 + (m - 1) * I
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l880_88075


namespace NUMINAMATH_CALUDE_total_cost_of_three_items_l880_88066

/-- The total cost of three items is the sum of their individual costs -/
theorem total_cost_of_three_items
  (cost_watch : ℕ)
  (cost_bracelet : ℕ)
  (cost_necklace : ℕ)
  (h1 : cost_watch = 144)
  (h2 : cost_bracelet = 250)
  (h3 : cost_necklace = 190) :
  cost_watch + cost_bracelet + cost_necklace = 584 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_of_three_items_l880_88066


namespace NUMINAMATH_CALUDE_line_relationship_undetermined_l880_88007

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parabola defined by y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a parabola -/
def pointOnParabola (p : Point) (para : Parabola) : Prop :=
  p.y = para.a * p.x^2 + para.b * p.x + para.c

/-- The relationship between two lines -/
inductive LineRelationship
  | Parallel
  | Perpendicular
  | Intersecting

/-- Theorem: The relationship between AD and BC cannot be determined -/
theorem line_relationship_undetermined 
  (A B C D : Point) 
  (para : Parabola) 
  (h_a_nonzero : para.a ≠ 0)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_on_parabola : pointOnParabola A para ∧ pointOnParabola B para ∧ 
                   pointOnParabola C para ∧ pointOnParabola D para)
  (h_x_sum : A.x + D.x - B.x + C.x = 0) :
  ∀ r : LineRelationship, ∃ para' : Parabola, 
    para'.a ≠ 0 ∧
    (pointOnParabola A para' ∧ pointOnParabola B para' ∧ 
     pointOnParabola C para' ∧ pointOnParabola D para') ∧
    A.x + D.x - B.x + C.x = 0 :=
  sorry

end NUMINAMATH_CALUDE_line_relationship_undetermined_l880_88007


namespace NUMINAMATH_CALUDE_win_sector_area_l880_88013

/-- Given a circular spinner with radius 8 cm and a probability of winning 1/4,
    prove that the area of the WIN sector is 16π square centimeters. -/
theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 1/4) :
  p * π * r^2 = 16 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l880_88013


namespace NUMINAMATH_CALUDE_walking_problem_l880_88036

/-- Represents the walking problem from "The Nine Chapters on the Mathematical Art" -/
theorem walking_problem (x : ℝ) :
  (∀ d : ℝ, d > 0 → 100 * (d / 60) = d) →  -- Good walker takes 100 steps for every 60 steps of bad walker
  x = 100 + (60 / 100) * x →               -- The equation to be proved
  x = (100 * 100) / 40                     -- The solution (not given in the original problem, but included for completeness)
  := by sorry

end NUMINAMATH_CALUDE_walking_problem_l880_88036


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l880_88041

theorem like_terms_exponent_sum (m n : ℤ) : 
  (m + 2 = 6 ∧ n + 1 = 3) → (-m)^3 + n^2 = -60 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l880_88041


namespace NUMINAMATH_CALUDE_a_range_l880_88034

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, a ≥ 2^x

-- Define proposition q
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 4*x + a = 0

-- Theorem statement
theorem a_range (a : ℝ) (hp : p a) (hq : q a) : 2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l880_88034


namespace NUMINAMATH_CALUDE_like_terms_imply_a_minus_b_eq_two_l880_88090

/-- Two algebraic expressions are like terms if they have the same variables raised to the same powers. -/
def are_like_terms (expr1 expr2 : ℝ → ℝ → ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ) (m n : ℕ), 
    (∀ x y, expr1 x y = c₁ * x^m * y^n) ∧ 
    (∀ x y, expr2 x y = c₂ * x^m * y^n)

/-- Given that -2.5x^(a+b)y^(a-1) and 3x^2y are like terms, prove that a - b = 2 -/
theorem like_terms_imply_a_minus_b_eq_two 
  (a b : ℝ) 
  (h : are_like_terms (λ x y => -2.5 * x^(a+b) * y^(a-1)) (λ x y => 3 * x^2 * y)) : 
  a - b = 2 := by
sorry


end NUMINAMATH_CALUDE_like_terms_imply_a_minus_b_eq_two_l880_88090


namespace NUMINAMATH_CALUDE_final_F_position_l880_88037

-- Define the letter F as a type with base and stem directions
inductive LetterF
  | mk (base : ℝ × ℝ) (stem : ℝ × ℝ)

-- Define the initial position of F
def initial_F : LetterF := LetterF.mk (-1, 0) (0, -1)

-- Define the transformations
def rotate_180 (f : LetterF) : LetterF :=
  match f with
  | LetterF.mk (x, y) (a, b) => LetterF.mk (-x, -y) (-a, -b)

def reflect_y_axis (f : LetterF) : LetterF :=
  match f with
  | LetterF.mk (x, y) (a, b) => LetterF.mk (-x, y) (-a, b)

def rotate_90 (f : LetterF) : LetterF :=
  match f with
  | LetterF.mk (x, y) (a, b) => LetterF.mk (y, -x) (b, -a)

-- Define the final transformation as a composition of the three transformations
def final_transformation (f : LetterF) : LetterF :=
  rotate_90 (reflect_y_axis (rotate_180 f))

-- Theorem: The final position of F after transformations
theorem final_F_position :
  final_transformation initial_F = LetterF.mk (0, -1) (-1, 0) :=
by sorry

end NUMINAMATH_CALUDE_final_F_position_l880_88037


namespace NUMINAMATH_CALUDE_conference_men_count_l880_88053

/-- The number of men at a climate conference -/
def number_of_men : ℕ := 700

/-- The number of women at the conference -/
def number_of_women : ℕ := 500

/-- The number of children at the conference -/
def number_of_children : ℕ := 800

/-- The percentage of men who were Indian -/
def indian_men_percentage : ℚ := 20 / 100

/-- The percentage of women who were Indian -/
def indian_women_percentage : ℚ := 40 / 100

/-- The percentage of children who were Indian -/
def indian_children_percentage : ℚ := 10 / 100

/-- The percentage of people who were not Indian -/
def non_indian_percentage : ℚ := 79 / 100

theorem conference_men_count :
  let total_people := number_of_men + number_of_women + number_of_children
  let indian_people := (indian_men_percentage * number_of_men) + 
                       (indian_women_percentage * number_of_women) + 
                       (indian_children_percentage * number_of_children)
  (1 - non_indian_percentage) * total_people = indian_people →
  number_of_men = 700 := by sorry

end NUMINAMATH_CALUDE_conference_men_count_l880_88053


namespace NUMINAMATH_CALUDE_calvin_insect_collection_l880_88064

/-- Calculates the total number of insects in Calvin's collection. -/
def total_insects (roaches scorpions : ℕ) : ℕ :=
  let crickets := roaches / 2
  let caterpillars := scorpions * 2
  roaches + scorpions + crickets + caterpillars

/-- Proves that Calvin has 27 insects in his collection. -/
theorem calvin_insect_collection : total_insects 12 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_calvin_insect_collection_l880_88064


namespace NUMINAMATH_CALUDE_jake_tuesday_watching_time_l880_88098

def monday_hours : ℝ := 12
def wednesday_hours : ℝ := 6
def friday_hours : ℝ := 19
def total_show_length : ℝ := 52

def tuesday_hours : ℝ := 4

theorem jake_tuesday_watching_time :
  let mon_to_wed := monday_hours + tuesday_hours + wednesday_hours
  let thursday_hours := mon_to_wed / 2
  monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours = total_show_length :=
by sorry

end NUMINAMATH_CALUDE_jake_tuesday_watching_time_l880_88098


namespace NUMINAMATH_CALUDE_figure_18_to_square_l880_88093

/-- Represents a figure on a graph paper -/
structure Figure where
  area : ℕ

/-- Represents a cut of the figure -/
structure Cut where
  parts : ℕ

/-- Represents the result of rearranging the cut parts -/
structure Rearrangement where
  is_square : Bool

/-- Function to determine if a figure can be cut and rearranged into a square -/
def can_form_square (f : Figure) (c : Cut) : Prop :=
  ∃ (r : Rearrangement), r.is_square = true

/-- Theorem stating that a figure with area 18 can be cut into 3 parts and rearranged into a square -/
theorem figure_18_to_square :
  ∀ (f : Figure) (c : Cut), 
    f.area = 18 → c.parts = 3 → can_form_square f c :=
by sorry

end NUMINAMATH_CALUDE_figure_18_to_square_l880_88093


namespace NUMINAMATH_CALUDE_paint_set_cost_l880_88027

def total_spent : ℕ := 80
def num_classes : ℕ := 6
def folders_per_class : ℕ := 1
def pencils_per_class : ℕ := 3
def pencils_per_eraser : ℕ := 6
def folder_cost : ℕ := 6
def pencil_cost : ℕ := 2
def eraser_cost : ℕ := 1

def total_folders : ℕ := num_classes * folders_per_class
def total_pencils : ℕ := num_classes * pencils_per_class
def total_erasers : ℕ := total_pencils / pencils_per_eraser

def supplies_cost : ℕ := 
  total_folders * folder_cost + 
  total_pencils * pencil_cost + 
  total_erasers * eraser_cost

theorem paint_set_cost : total_spent - supplies_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_paint_set_cost_l880_88027


namespace NUMINAMATH_CALUDE_complement_union_equality_l880_88035

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3, 4}

-- Define set A
def A : Set ℕ := {0, 3, 4}

-- Define set B
def B : Set ℕ := {1, 3}

-- Theorem statement
theorem complement_union_equality : (U \ A) ∪ B = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_equality_l880_88035


namespace NUMINAMATH_CALUDE_division_remainder_problem_l880_88071

theorem division_remainder_problem (N : ℕ) (Q2 : ℕ) :
  (∃ R1 : ℕ, N = 44 * 432 + R1) ∧ 
  (∃ Q2 : ℕ, N = 38 * Q2 + 8) →
  N % 44 = 0 :=
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l880_88071


namespace NUMINAMATH_CALUDE_movie_marathon_duration_l880_88067

def movie_marathon (first_movie : ℝ) (second_movie_percentage : ℝ) (third_movie_difference : ℝ) : ℝ :=
  let second_movie := first_movie * (1 + second_movie_percentage)
  let third_movie := first_movie + second_movie - third_movie_difference
  first_movie + second_movie + third_movie

theorem movie_marathon_duration :
  movie_marathon 2 0.5 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_movie_marathon_duration_l880_88067


namespace NUMINAMATH_CALUDE_P_not_factorable_l880_88063

/-- The polynomial P(x,y) = x^n + xy + y^n -/
def P (n : ℕ) (x y : ℝ) : ℝ := x^n + x*y + y^n

/-- Theorem stating that P(x,y) cannot be factored into two non-constant real polynomials -/
theorem P_not_factorable (n : ℕ) :
  ¬∃ (G H : ℝ → ℝ → ℝ), 
    (∀ x y, P n x y = G x y * H x y) ∧ 
    (∃ a b c d, G a b ≠ G c d) ∧ 
    (∃ a b c d, H a b ≠ H c d) :=
sorry

end NUMINAMATH_CALUDE_P_not_factorable_l880_88063


namespace NUMINAMATH_CALUDE_complex_product_PRS_l880_88032

theorem complex_product_PRS : 
  let P : ℂ := 3 + 4 * Complex.I
  let R : ℂ := 2 * Complex.I
  let S : ℂ := 3 - 4 * Complex.I
  P * R * S = 50 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_product_PRS_l880_88032


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt35_l880_88012

theorem rationalize_denominator_sqrt35 : 
  (35 : ℝ) / Real.sqrt 35 = Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt35_l880_88012


namespace NUMINAMATH_CALUDE_carla_chicken_farm_problem_l880_88002

/-- The percentage of chickens that died in Carla's farm -/
def percentage_died (initial_chickens final_chickens : ℕ) : ℚ :=
  let bought_chickens := final_chickens - initial_chickens
  let died_chickens := bought_chickens / 10
  (died_chickens : ℚ) / initial_chickens * 100

theorem carla_chicken_farm_problem :
  percentage_died 400 1840 = 40 := by
  sorry

end NUMINAMATH_CALUDE_carla_chicken_farm_problem_l880_88002


namespace NUMINAMATH_CALUDE_intersection_range_l880_88070

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the property of having three distinct intersection points
def has_three_distinct_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
  f x₁ = a ∧ f x₂ = a ∧ f x₃ = a

-- Theorem statement
theorem intersection_range (a : ℝ) :
  has_three_distinct_intersections a → -2 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l880_88070


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l880_88099

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l880_88099


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l880_88026

theorem min_value_sum_reciprocals (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (sum_one : x + y + z + w = 1) :
  1/(x+y) + 1/(x+z) + 1/(x+w) + 1/(y+z) + 1/(y+w) + 1/(z+w) ≥ 18 ∧
  (1/(x+y) + 1/(x+z) + 1/(x+w) + 1/(y+z) + 1/(y+w) + 1/(z+w) = 18 ↔ x = 1/4 ∧ y = 1/4 ∧ z = 1/4 ∧ w = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l880_88026


namespace NUMINAMATH_CALUDE_zero_point_in_interval_l880_88095

def f (x : ℝ) := -x^3 - 3*x + 5

theorem zero_point_in_interval :
  (∀ x y, x < y → f x > f y) →  -- f is monotonically decreasing
  Continuous f →
  f 1 > 0 →
  f 2 < 0 →
  ∃ c, c ∈ Set.Ioo 1 2 ∧ f c = 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_point_in_interval_l880_88095


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l880_88024

theorem quadratic_coefficient (a b c : ℤ) :
  (∀ x : ℝ, a * x^2 + b * x + c = a * (x - 2)^2 + 3) →
  a * 1^2 + b * 1 + c = 5 →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l880_88024


namespace NUMINAMATH_CALUDE_inequality_addition_l880_88065

theorem inequality_addition (a b c : ℝ) : a > b → a + c > b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_addition_l880_88065


namespace NUMINAMATH_CALUDE_jason_shampoo_time_l880_88003

theorem jason_shampoo_time :
  ∀ (J : ℝ),
  J > 0 →
  (1 / J + 1 / 6 = 1 / 2) →
  J = 3 :=
by sorry

end NUMINAMATH_CALUDE_jason_shampoo_time_l880_88003


namespace NUMINAMATH_CALUDE_at_least_one_non_negative_l880_88009

theorem at_least_one_non_negative (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) 
  (h₁ : a₁ ≠ 0) (h₂ : a₂ ≠ 0) (h₃ : a₃ ≠ 0) (h₄ : a₄ ≠ 0) 
  (h₅ : a₅ ≠ 0) (h₆ : a₆ ≠ 0) (h₇ : a₇ ≠ 0) (h₈ : a₈ ≠ 0) : 
  max (a₁ * a₃ + a₂ * a₄) (max (a₁ * a₅ + a₂ * a₆) (max (a₁ * a₇ + a₂ * a₈) 
    (max (a₃ * a₅ + a₄ * a₆) (max (a₃ * a₇ + a₄ * a₈) (a₅ * a₇ + a₆ * a₈))))) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_non_negative_l880_88009


namespace NUMINAMATH_CALUDE_division_remainder_problem_solution_l880_88029

theorem division_remainder (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = divisor * quotient + remainder →
  remainder < divisor →
  dividend / divisor = quotient →
  dividend % divisor = remainder :=
by sorry

theorem problem_solution :
  14 / 3 = 4 →
  14 % 3 = 2 :=
by sorry

end NUMINAMATH_CALUDE_division_remainder_problem_solution_l880_88029


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l880_88052

theorem quadratic_no_real_roots (m : ℝ) : 
  (∀ x : ℝ, x^2 + 3*x + m ≠ 0) → m > 9/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l880_88052


namespace NUMINAMATH_CALUDE_x_mod_105_l880_88016

theorem x_mod_105 (x : ℤ) 
  (h1 : (3 + x) % (3^3) = 2^2 % (3^3))
  (h2 : (5 + x) % (5^3) = 3^2 % (5^3))
  (h3 : (7 + x) % (7^3) = 5^2 % (7^3)) :
  x % 105 = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_mod_105_l880_88016


namespace NUMINAMATH_CALUDE_composite_probability_is_point_68_l880_88014

/-- The number of natural numbers from 1 to 50 -/
def total_numbers : ℕ := 50

/-- The number of composite numbers from 1 to 50 -/
def composite_count : ℕ := 34

/-- The probability of selecting a composite number from the first 50 natural numbers -/
def composite_probability : ℚ := composite_count / total_numbers

/-- Theorem: The probability of selecting a composite number from the first 50 natural numbers is 0.68 -/
theorem composite_probability_is_point_68 : composite_probability = 68 / 100 := by
  sorry

end NUMINAMATH_CALUDE_composite_probability_is_point_68_l880_88014


namespace NUMINAMATH_CALUDE_billys_age_l880_88038

theorem billys_age (billy joe : ℕ) 
  (h1 : billy = 3 * joe)
  (h2 : billy + joe = 60)
  (h3 : billy > 30) :
  billy = 45 := by
sorry

end NUMINAMATH_CALUDE_billys_age_l880_88038


namespace NUMINAMATH_CALUDE_matrix_subtraction_l880_88057

theorem matrix_subtraction : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 6, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![1, -8; 3, 7]
  let C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 5; 3, -2]
  A - B = C := by sorry

end NUMINAMATH_CALUDE_matrix_subtraction_l880_88057


namespace NUMINAMATH_CALUDE_employment_percentage_l880_88046

theorem employment_percentage (total_population : ℝ) (employed_population : ℝ) :
  (employed_population / total_population = 0.5 / 0.78125) ↔
  (0.5 * total_population = employed_population * (1 - 0.21875)) :=
by sorry

end NUMINAMATH_CALUDE_employment_percentage_l880_88046


namespace NUMINAMATH_CALUDE_cube_folding_l880_88069

/-- Represents the squares on the flat sheet --/
inductive Square
| A | B | C | D | E | F

/-- Represents the adjacency of squares on the flat sheet --/
def adjacent : Square → Square → Prop :=
  sorry

/-- Represents the opposite faces of the cube after folding --/
def opposite : Square → Square → Prop :=
  sorry

/-- The theorem to be proved --/
theorem cube_folding (h1 : adjacent Square.B Square.A)
                     (h2 : adjacent Square.C Square.B)
                     (h3 : adjacent Square.C Square.A)
                     (h4 : adjacent Square.D Square.C)
                     (h5 : adjacent Square.E Square.A)
                     (h6 : adjacent Square.F Square.D)
                     (h7 : adjacent Square.F Square.E) :
  opposite Square.A Square.D :=
sorry

end NUMINAMATH_CALUDE_cube_folding_l880_88069


namespace NUMINAMATH_CALUDE_sum_distinct_digits_mod_1000_l880_88021

/-- The sum of all four-digit positive integers with distinct digits -/
def T : ℕ := sorry

/-- Predicate to check if a number has distinct digits -/
def has_distinct_digits (n : ℕ) : Prop := sorry

theorem sum_distinct_digits_mod_1000 : 
  T % 1000 = 400 :=
sorry

end NUMINAMATH_CALUDE_sum_distinct_digits_mod_1000_l880_88021


namespace NUMINAMATH_CALUDE_integer_root_of_cubic_l880_88049

/-- Given a cubic polynomial x^3 + bx + c = 0 with rational coefficients b and c,
    if 5 - √2 is a root, then -10 is also a root. -/
theorem integer_root_of_cubic (b c : ℚ) : 
  (5 - Real.sqrt 2)^3 + b*(5 - Real.sqrt 2) + c = 0 →
  (-10)^3 + b*(-10) + c = 0 := by
sorry

end NUMINAMATH_CALUDE_integer_root_of_cubic_l880_88049


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l880_88005

theorem cone_generatrix_length (r : ℝ) (h1 : r = Real.sqrt 2) :
  let l := 2 * Real.sqrt 2
  (2 * Real.pi * r = Real.pi * l) → l = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cone_generatrix_length_l880_88005


namespace NUMINAMATH_CALUDE_milton_books_l880_88096

theorem milton_books (total : ℕ) (zoology : ℕ) (botany : ℕ) : 
  total = 960 → 
  botany = 7 * zoology → 
  total = zoology + botany → 
  zoology = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_milton_books_l880_88096


namespace NUMINAMATH_CALUDE_polynomial_remainder_l880_88059

def polynomial (x : ℝ) : ℝ := 4*x^8 - 3*x^7 + 2*x^6 - 8*x^4 + 5*x^3 - 9

def divisor (x : ℝ) : ℝ := 3*x - 6

theorem polynomial_remainder : 
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * q x + 671 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l880_88059


namespace NUMINAMATH_CALUDE_expression_equality_l880_88084

theorem expression_equality : (3 / 5 : ℚ) * ((2 / 3 + 3 / 8) / 2) - 1 / 16 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l880_88084


namespace NUMINAMATH_CALUDE_sugar_left_l880_88077

/-- Given that Pamela bought 9.8 ounces of sugar and spilled 5.2 ounces,
    prove that the amount of sugar left is 4.6 ounces. -/
theorem sugar_left (bought spilled : ℝ) (h1 : bought = 9.8) (h2 : spilled = 5.2) :
  bought - spilled = 4.6 := by
  sorry

end NUMINAMATH_CALUDE_sugar_left_l880_88077


namespace NUMINAMATH_CALUDE_abs_neg_four_equals_four_l880_88048

theorem abs_neg_four_equals_four : |(-4 : ℤ)| = 4 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_four_equals_four_l880_88048


namespace NUMINAMATH_CALUDE_cookies_per_bag_l880_88001

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (cookies_per_bag : ℕ) 
  (h1 : total_cookies = 14)
  (h2 : num_bags = 7)
  (h3 : total_cookies = num_bags * cookies_per_bag) :
  cookies_per_bag = 2 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l880_88001


namespace NUMINAMATH_CALUDE_katya_magic_pen_problem_l880_88025

theorem katya_magic_pen_problem (katya_prob : ℚ) (pen_prob : ℚ) (total_problems : ℕ) (min_correct : ℕ) :
  katya_prob = 4/5 →
  pen_prob = 1/2 →
  total_problems = 20 →
  min_correct = 13 →
  ∃ x : ℕ, x ≥ 10 ∧
    (x : ℚ) * katya_prob + (total_problems - x : ℚ) * pen_prob ≥ min_correct ∧
    ∀ y : ℕ, y < 10 →
      (y : ℚ) * katya_prob + (total_problems - y : ℚ) * pen_prob < min_correct :=
by sorry

end NUMINAMATH_CALUDE_katya_magic_pen_problem_l880_88025


namespace NUMINAMATH_CALUDE_choir_members_count_l880_88015

theorem choir_members_count : ∃! n : ℕ, 
  200 ≤ n ∧ n ≤ 300 ∧ 
  n % 10 = 4 ∧ 
  n % 11 = 5 ∧ 
  n = 234 := by sorry

end NUMINAMATH_CALUDE_choir_members_count_l880_88015


namespace NUMINAMATH_CALUDE_games_in_23_team_tournament_l880_88082

/-- Represents a single-elimination tournament -/
structure Tournament where
  num_teams : ℕ
  no_ties : Bool

/-- The number of games played in a single-elimination tournament -/
def games_played (t : Tournament) : ℕ := t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 23 teams and no ties, 
    the number of games played is 22 -/
theorem games_in_23_team_tournament (t : Tournament) 
  (h1 : t.num_teams = 23) (h2 : t.no_ties = true) : 
  games_played t = 22 := by
  sorry

end NUMINAMATH_CALUDE_games_in_23_team_tournament_l880_88082


namespace NUMINAMATH_CALUDE_range_of_3x_minus_2y_l880_88068

theorem range_of_3x_minus_2y (x y : ℝ) 
  (h1 : -1 ≤ x + y ∧ x + y ≤ 1) 
  (h2 : 1 ≤ x - y ∧ x - y ≤ 5) : 
  2 ≤ 3*x - 2*y ∧ 3*x - 2*y ≤ 13 := by
  sorry

end NUMINAMATH_CALUDE_range_of_3x_minus_2y_l880_88068


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l880_88092

theorem floor_equation_solutions : 
  (∃ (S : Finset ℤ), S.card = 30 ∧ 
    (∀ x ∈ S, 0 ≤ x ∧ x < 30 ∧ x = ⌊x/2⌋ + ⌊x/3⌋ + ⌊x/5⌋) ∧
    (∀ x : ℤ, 0 ≤ x ∧ x < 30 ∧ x = ⌊x/2⌋ + ⌊x/3⌋ + ⌊x/5⌋ → x ∈ S)) :=
by sorry


end NUMINAMATH_CALUDE_floor_equation_solutions_l880_88092


namespace NUMINAMATH_CALUDE_planes_parallel_transitive_l880_88089

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation for planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_transitive 
  (α β γ : Plane) 
  (h1 : parallel α γ) 
  (h2 : parallel γ β) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_transitive_l880_88089


namespace NUMINAMATH_CALUDE_turban_price_turban_price_proof_l880_88058

/-- The price of a turban given the following conditions:
  - The total salary for one year is Rs. 90 plus one turban
  - The servant leaves after 9 months
  - The servant receives Rs. 40 and the turban after 9 months
-/
theorem turban_price : ℝ → Prop :=
  fun price =>
    let total_salary : ℝ := 90 + price
    let months_worked : ℝ := 9
    let total_months : ℝ := 12
    let received_amount : ℝ := 40 + price
    (months_worked / total_months) * total_salary = received_amount →
    price = 110

/-- Proof of the turban price theorem -/
theorem turban_price_proof : ∃ price, turban_price price := by
  sorry

end NUMINAMATH_CALUDE_turban_price_turban_price_proof_l880_88058


namespace NUMINAMATH_CALUDE_gcd_of_256_180_600_l880_88043

theorem gcd_of_256_180_600 : Nat.gcd 256 (Nat.gcd 180 600) = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_256_180_600_l880_88043


namespace NUMINAMATH_CALUDE_orange_stock_proof_l880_88033

/-- Represents the original stock of oranges in kg -/
def original_stock : ℝ := 2700

/-- Represents the percentage of stock remaining after sale -/
def remaining_percentage : ℝ := 0.25

/-- Represents the amount of oranges remaining after sale in kg -/
def remaining_stock : ℝ := 675

theorem orange_stock_proof :
  remaining_percentage * original_stock = remaining_stock :=
sorry

end NUMINAMATH_CALUDE_orange_stock_proof_l880_88033


namespace NUMINAMATH_CALUDE_connie_calculation_l880_88080

theorem connie_calculation (x : ℝ) : 4 * x = 200 → x / 4 + 10 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_connie_calculation_l880_88080


namespace NUMINAMATH_CALUDE_m_mobile_additional_line_cost_l880_88011

/-- Represents a mobile phone plan with a base cost and additional line cost -/
structure MobilePlan where
  baseCost : ℕ  -- Cost for first two lines
  addLineCost : ℕ  -- Cost for each additional line

/-- Calculates the total cost for a given number of lines -/
def totalCost (plan : MobilePlan) (lines : ℕ) : ℕ :=
  plan.baseCost + plan.addLineCost * (lines - 2)

theorem m_mobile_additional_line_cost :
  ∃ (mMobileAddCost : ℕ),
    let tMobile : MobilePlan := ⟨50, 16⟩
    let mMobile : MobilePlan := ⟨45, mMobileAddCost⟩
    totalCost tMobile 5 - totalCost mMobile 5 = 11 →
    mMobileAddCost = 14 := by
  sorry


end NUMINAMATH_CALUDE_m_mobile_additional_line_cost_l880_88011


namespace NUMINAMATH_CALUDE_airplane_stop_time_l880_88006

/-- The distance function for an airplane after landing -/
def distance (t : ℝ) : ℝ := 75 * t - 1.5 * t^2

/-- The time at which the airplane stops -/
def stop_time : ℝ := 25

theorem airplane_stop_time :
  (∀ t : ℝ, distance t ≤ distance stop_time) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ t, |t - stop_time| < δ → distance stop_time - distance t < ε) :=
sorry

end NUMINAMATH_CALUDE_airplane_stop_time_l880_88006


namespace NUMINAMATH_CALUDE_sum_of_integers_l880_88020

theorem sum_of_integers (a b : ℤ) (h : 6 * a * b = 9 * a - 10 * b + 16) : a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l880_88020


namespace NUMINAMATH_CALUDE_chess_tournament_l880_88074

theorem chess_tournament (n : ℕ) : 
  (n ≥ 3) →  -- Ensure at least 3 players (2 who withdraw + 1 more)
  ((n - 2) * (n - 3) / 2 + 3 = 81) →  -- Total games equation
  (n = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_l880_88074


namespace NUMINAMATH_CALUDE_mushroom_count_l880_88019

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Theorem: Given the conditions, the number of mushrooms is 950 -/
theorem mushroom_count : ∃ (N : ℕ), 
  100 ≤ N ∧ N < 1000 ∧  -- N is a three-digit number
  sumOfDigits N = 14 ∧  -- sum of digits is 14
  N % 50 = 0 ∧  -- N is divisible by 50
  N % 100 = 50 ∧  -- N ends in 50
  N = 950 := by
sorry

end NUMINAMATH_CALUDE_mushroom_count_l880_88019


namespace NUMINAMATH_CALUDE_max_abs_z_on_circle_l880_88088

open Complex

theorem max_abs_z_on_circle (z : ℂ) : 
  (abs (z - 2*I) = 1) → (abs z ≤ 3) ∧ ∃ w : ℂ, abs (w - 2*I) = 1 ∧ abs w = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_z_on_circle_l880_88088


namespace NUMINAMATH_CALUDE_johns_quarters_l880_88072

theorem johns_quarters (quarters dimes nickels : ℕ) : 
  quarters + dimes + nickels = 63 →
  dimes = quarters + 3 →
  nickels = quarters - 6 →
  quarters = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_quarters_l880_88072


namespace NUMINAMATH_CALUDE_train_final_speed_train_final_speed_zero_l880_88076

/-- Proves that a train with given initial speed and deceleration comes to a stop before traveling 4 km -/
theorem train_final_speed (v_i : Real) (a : Real) (d : Real) :
  v_i = 189 * (1000 / 3600) →
  a = -0.5 →
  d = 4000 →
  v_i^2 + 2 * a * d < 0 →
  ∃ (d_stop : Real), d_stop < d ∧ v_i^2 + 2 * a * d_stop = 0 :=
by sorry

/-- Proves that the final speed of the train after traveling 4 km is 0 m/s -/
theorem train_final_speed_zero (v_i : Real) (a : Real) (d : Real) (v_f : Real) :
  v_i = 189 * (1000 / 3600) →
  a = -0.5 →
  d = 4000 →
  v_f^2 = v_i^2 + 2 * a * d →
  v_f = 0 :=
by sorry

end NUMINAMATH_CALUDE_train_final_speed_train_final_speed_zero_l880_88076


namespace NUMINAMATH_CALUDE_negative_reciprocal_of_0125_l880_88055

def negative_reciprocal (a b : ℝ) : Prop := a * b = -1

theorem negative_reciprocal_of_0125 :
  negative_reciprocal 0.125 (-8) := by
  sorry

end NUMINAMATH_CALUDE_negative_reciprocal_of_0125_l880_88055


namespace NUMINAMATH_CALUDE_midpoint_coordinates_l880_88061

/-- Given point A and vector AB, prove that the midpoint of segment AB has specific coordinates -/
theorem midpoint_coordinates (A B : ℝ × ℝ) (h1 : A = (-3, 2)) (h2 : B - A = (6, 0)) :
  (A.1 + B.1) / 2 = 0 ∧ (A.2 + B.2) / 2 = 2 := by
  sorry

#check midpoint_coordinates

end NUMINAMATH_CALUDE_midpoint_coordinates_l880_88061


namespace NUMINAMATH_CALUDE_value_of_expression_l880_88031

theorem value_of_expression (x : ℝ) (h : x^2 - 2*x = 3) : 3*x^2 - 6*x - 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l880_88031


namespace NUMINAMATH_CALUDE_square_sum_from_diff_and_product_l880_88073

theorem square_sum_from_diff_and_product (p q : ℝ) 
  (h1 : p - q = 4) 
  (h2 : p * q = -2) : 
  p^2 + q^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_diff_and_product_l880_88073


namespace NUMINAMATH_CALUDE_all_nines_multiple_l880_88022

theorem all_nines_multiple (p : Nat) (hp : Nat.Prime p) (hp2 : p ≠ 2) (hp5 : p ≠ 5) :
  ∃ k : Nat, k > 0 ∧ (((10^k - 1) / 9) % p = 0) := by
  sorry

end NUMINAMATH_CALUDE_all_nines_multiple_l880_88022
