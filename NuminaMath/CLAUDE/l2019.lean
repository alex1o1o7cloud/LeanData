import Mathlib

namespace NUMINAMATH_CALUDE_fox_initial_money_l2019_201906

/-- The amount of money the fox has after n bridge crossings -/
def fox_money (a₀ : ℕ) : ℕ → ℕ
  | 0 => a₀
  | n + 1 => 2 * fox_money a₀ n - 2^2019

theorem fox_initial_money :
  ∀ a₀ : ℕ, fox_money a₀ 2019 = 0 → a₀ = 2^2019 - 1 := by
  sorry

#check fox_initial_money

end NUMINAMATH_CALUDE_fox_initial_money_l2019_201906


namespace NUMINAMATH_CALUDE_semicircle_area_ratio_l2019_201942

theorem semicircle_area_ratio (R : ℝ) (h : R > 0) :
  let r := (3 : ℝ) / 5 * R
  (π * r^2 / 2) / (π * R^2 / 2) = 9 / 25 := by sorry

end NUMINAMATH_CALUDE_semicircle_area_ratio_l2019_201942


namespace NUMINAMATH_CALUDE_negation_equivalence_l2019_201966

-- Define the curve
def is_curve (m : ℕ) (x y : ℝ) : Prop := x^2 / m + y^2 = 1

-- Define what it means for the curve to be an ellipse (this is a placeholder definition)
def is_ellipse (m : ℕ) : Prop := ∃ x y : ℝ, is_curve m x y

-- The theorem to prove
theorem negation_equivalence :
  (¬ ∃ m : ℕ, is_ellipse m) ↔ (∀ m : ℕ, ¬ is_ellipse m) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2019_201966


namespace NUMINAMATH_CALUDE_diagonals_in_polygon_l2019_201909

/-- The number of diagonals in a convex k-sided polygon. -/
def num_diagonals (k : ℕ) : ℕ := k * (k - 3) / 2

/-- Theorem stating that the number of diagonals in a convex k-sided polygon
    (where k > 3) is equal to k(k-3)/2. -/
theorem diagonals_in_polygon (k : ℕ) (h : k > 3) :
  num_diagonals k = k * (k - 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_diagonals_in_polygon_l2019_201909


namespace NUMINAMATH_CALUDE_smallest_power_congruence_l2019_201913

/-- For any integer r ≥ 3, the smallest positive integer d₀ such that 7^d₀ ≡ 1 (mod 2^r) is 2^(r-2) -/
theorem smallest_power_congruence (r : ℕ) (hr : r ≥ 3) :
  (∃ (d₀ : ℕ), d₀ > 0 ∧ 7^d₀ ≡ 1 [MOD 2^r] ∧
    ∀ (d : ℕ), d > 0 → 7^d ≡ 1 [MOD 2^r] → d₀ ≤ d) ∧
  (∀ (d₀ : ℕ), d₀ > 0 → 7^d₀ ≡ 1 [MOD 2^r] ∧
    (∀ (d : ℕ), d > 0 → 7^d ≡ 1 [MOD 2^r] → d₀ ≤ d) →
    d₀ = 2^(r-2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_power_congruence_l2019_201913


namespace NUMINAMATH_CALUDE_treatment_volume_is_120_ml_l2019_201961

/-- Calculates the total volume of treatment received from a saline drip. -/
def total_treatment_volume (drops_per_minute : ℕ) (treatment_hours : ℕ) (ml_per_100_drops : ℕ) : ℕ :=
  let minutes_per_hour : ℕ := 60
  let drops_per_100 : ℕ := 100
  let total_minutes : ℕ := treatment_hours * minutes_per_hour
  let total_drops : ℕ := drops_per_minute * total_minutes
  (total_drops * ml_per_100_drops) / drops_per_100

/-- The theorem stating that the total treatment volume is 120 ml under given conditions. -/
theorem treatment_volume_is_120_ml :
  total_treatment_volume 20 2 5 = 120 :=
by
  sorry

#eval total_treatment_volume 20 2 5

end NUMINAMATH_CALUDE_treatment_volume_is_120_ml_l2019_201961


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l2019_201912

theorem sqrt_difference_equality : Real.sqrt (64 + 81) - Real.sqrt (49 - 36) = Real.sqrt 145 - Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l2019_201912


namespace NUMINAMATH_CALUDE_tan_2x_value_l2019_201964

theorem tan_2x_value (f : ℝ → ℝ) (x : ℝ) 
  (h1 : ∀ x, f x = Real.sin x + Real.cos x)
  (h2 : ∀ x, deriv f x = 3 * f x) : 
  Real.tan (2 * x) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_2x_value_l2019_201964


namespace NUMINAMATH_CALUDE_sqrt_fraction_equals_sixteen_l2019_201986

theorem sqrt_fraction_equals_sixteen :
  let eight : ℕ := 2^3
  let four : ℕ := 2^2
  ∀ x : ℝ, x = (((eight^10 + four^10) : ℝ) / (eight^4 + four^11 : ℝ))^(1/2) → x = 16 := by
sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equals_sixteen_l2019_201986


namespace NUMINAMATH_CALUDE_james_fish_catch_l2019_201943

/-- The amount of trout James caught in pounds -/
def trout : ℝ := 200

/-- The amount of salmon James caught in pounds -/
def salmon : ℝ := 1.5 * trout

/-- The amount of tuna James caught in pounds -/
def tuna : ℝ := 2 * trout

/-- The total amount of fish James caught in pounds -/
def total_fish : ℝ := trout + salmon + tuna

theorem james_fish_catch : total_fish = 900 := by
  sorry

end NUMINAMATH_CALUDE_james_fish_catch_l2019_201943


namespace NUMINAMATH_CALUDE_range_of_a_l2019_201970

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x + 4 / x - 1 - a^2 + 2*a > 0) → 
  -1 < a ∧ a < 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2019_201970


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2019_201999

theorem greatest_divisor_with_remainders : Nat.gcd (1657 - 6) (2037 - 5) = 127 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2019_201999


namespace NUMINAMATH_CALUDE_angle_through_point_l2019_201991

theorem angle_through_point (α : Real) :
  (∃ (x y : Real), x = -1 ∧ y = 2 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin α = 2 * Real.sqrt 5 / 5 ∧
  Real.cos α = -Real.sqrt 5 / 5 ∧
  Real.tan α = -2 ∧
  Real.tan (α - Real.pi/4) = 3 := by
sorry

end NUMINAMATH_CALUDE_angle_through_point_l2019_201991


namespace NUMINAMATH_CALUDE_inverse_matrices_sum_l2019_201929

open Matrix

theorem inverse_matrices_sum (x y z w p q r s : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![x, 2, y; 3, 4, 5; z, 6, w]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![-7, p, -13; q, -15, r; 3, s, 6]
  A * B = 1 → x + y + z + w + p + q + r + s = -5.5 := by
sorry

end NUMINAMATH_CALUDE_inverse_matrices_sum_l2019_201929


namespace NUMINAMATH_CALUDE_sqrt_sum_rational_implies_components_rational_l2019_201918

theorem sqrt_sum_rational_implies_components_rational
  (A B : ℚ) (h : ∃ (r : ℚ), r = Real.sqrt A + Real.sqrt B) :
  (∃ (s : ℚ), s = Real.sqrt A) ∧ (∃ (t : ℚ), t = Real.sqrt B) :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_rational_implies_components_rational_l2019_201918


namespace NUMINAMATH_CALUDE_wife_account_percentage_l2019_201958

def income : ℝ := 1000000

def children_percentage : ℝ := 0.2
def num_children : ℕ := 3
def orphan_house_percentage : ℝ := 0.05
def final_amount : ℝ := 50000

theorem wife_account_percentage : 
  let children_total := children_percentage * num_children * income
  let remaining_after_children := income - children_total
  let orphan_house_donation := orphan_house_percentage * remaining_after_children
  let remaining_after_donation := remaining_after_children - orphan_house_donation
  let wife_account := remaining_after_donation - final_amount
  (wife_account / income) * 100 = 33 := by sorry

end NUMINAMATH_CALUDE_wife_account_percentage_l2019_201958


namespace NUMINAMATH_CALUDE_monroe_family_children_l2019_201963

/-- Given the total number of granola bars, the number eaten by parents, and the number given to each child,
    calculate the number of children in the family. -/
def number_of_children (total_bars : ℕ) (eaten_by_parents : ℕ) (bars_per_child : ℕ) : ℕ :=
  (total_bars - eaten_by_parents) / bars_per_child

/-- Theorem stating that the number of children in Monroe's family is 6. -/
theorem monroe_family_children :
  number_of_children 200 80 20 = 6 := by
  sorry

end NUMINAMATH_CALUDE_monroe_family_children_l2019_201963


namespace NUMINAMATH_CALUDE_huahuan_initial_cards_l2019_201952

/-- Represents the card distribution among the three players -/
structure CardDistribution where
  huahuan : ℕ
  yingying : ℕ
  nini : ℕ

/-- Represents one round of operations -/
def performRound (dist : CardDistribution) : CardDistribution :=
  sorry

/-- Check if the distribution forms an arithmetic sequence -/
def isArithmeticSequence (dist : CardDistribution) : Prop :=
  dist.yingying - dist.huahuan = dist.nini - dist.yingying

/-- The main theorem -/
theorem huahuan_initial_cards 
  (initial : CardDistribution)
  (h1 : initial.huahuan + initial.yingying + initial.nini = 2712)
  (h2 : ∃ (final : CardDistribution), 
    (performRound^[50] initial = final) ∧ 
    (isArithmeticSequence final)) :
  initial.huahuan = 754 := by
  sorry


end NUMINAMATH_CALUDE_huahuan_initial_cards_l2019_201952


namespace NUMINAMATH_CALUDE_unique_solution_square_equation_l2019_201983

theorem unique_solution_square_equation :
  ∃! x : ℚ, (2015 + x)^2 = x^2 ∧ x = -2015/2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_square_equation_l2019_201983


namespace NUMINAMATH_CALUDE_exponent_problem_l2019_201962

theorem exponent_problem (x m n : ℝ) (hm : x^m = 5) (hn : x^n = 10) :
  x^(2*m - n) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_problem_l2019_201962


namespace NUMINAMATH_CALUDE_find_A_l2019_201910

theorem find_A : ∃ A : ℕ, A = 23 ∧ A / 8 = 2 ∧ A % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l2019_201910


namespace NUMINAMATH_CALUDE_right_triangle_with_special_point_l2019_201904

theorem right_triangle_with_special_point (A B C P : ℝ × ℝ) 
  (h_right : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0)
  (h_AP : Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) = 2)
  (h_BP : Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 1)
  (h_CP : Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2) = Real.sqrt 5)
  (h_inside : ∃ (t u : ℝ), t > 0 ∧ u > 0 ∧ t + u < 1 ∧ 
    P.1 = t * B.1 + u * C.1 + (1 - t - u) * A.1 ∧
    P.2 = t * B.2 + u * C.2 + (1 - t - u) * A.2) :
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 5 := by
  sorry

#check right_triangle_with_special_point

end NUMINAMATH_CALUDE_right_triangle_with_special_point_l2019_201904


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2019_201990

theorem no_integer_solutions : ¬∃ (x y : ℤ), 0 < x ∧ x < y ∧ Real.sqrt 4096 = Real.sqrt x + Real.sqrt y + Real.sqrt (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2019_201990


namespace NUMINAMATH_CALUDE_inequality_proof_l2019_201953

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≤ a + b + c) ∧
  (a + b + c = 1 → (2 * a * b) / (a + b) + (2 * b * c) / (b + c) + (2 * a * c) / (a + c) ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2019_201953


namespace NUMINAMATH_CALUDE_sales_profit_equation_max_profit_selling_price_range_l2019_201956

-- Define the cost to produce each item
def production_cost : ℝ := 50

-- Define the daily sales volume as a function of price
def sales_volume (x : ℝ) : ℝ := 50 + 5 * (100 - x)

-- Define the daily sales profit function
def sales_profit (x : ℝ) : ℝ := (x - production_cost) * sales_volume x

-- Theorem 1: The daily sales profit function
theorem sales_profit_equation (x : ℝ) :
  sales_profit x = -5 * x^2 + 800 * x - 27500 := by sorry

-- Theorem 2: The maximum daily sales profit
theorem max_profit :
  ∃ (x : ℝ), x = 80 ∧ sales_profit x = 4500 ∧
  ∀ (y : ℝ), 50 ≤ y ∧ y ≤ 100 → sales_profit y ≤ sales_profit x := by sorry

-- Theorem 3: The range of selling prices satisfying the conditions
theorem selling_price_range :
  ∀ (x : ℝ), (sales_profit x ≥ 4000 ∧ production_cost * sales_volume x ≤ 7000) ↔
  (82 ≤ x ∧ x ≤ 90) := by sorry

end NUMINAMATH_CALUDE_sales_profit_equation_max_profit_selling_price_range_l2019_201956


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l2019_201941

theorem roots_quadratic_equation (x₁ x₂ : ℝ) : 
  (x₁^2 + 3*x₁ - 2 = 0) → 
  (x₂^2 + 3*x₂ - 2 = 0) → 
  (x₁^2 + 2*x₁ - x₂ = 5) := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l2019_201941


namespace NUMINAMATH_CALUDE_baseball_card_pages_l2019_201987

def number_of_pages (packs : ℕ) (cards_per_pack : ℕ) (cards_per_page : ℕ) : ℕ :=
  (packs * cards_per_pack) / cards_per_page

theorem baseball_card_pages : number_of_pages 60 7 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_baseball_card_pages_l2019_201987


namespace NUMINAMATH_CALUDE_soccer_league_female_fraction_l2019_201907

theorem soccer_league_female_fraction :
  ∀ (male_last_year female_last_year : ℕ)
    (total_this_year : ℚ)
    (male_this_year female_this_year : ℚ),
  male_last_year = 15 →
  total_this_year = 1.15 * (male_last_year + female_last_year) →
  male_this_year = 1.1 * male_last_year →
  female_this_year = 2 * female_last_year →
  female_this_year / total_this_year = 5 / 51 :=
by sorry

end NUMINAMATH_CALUDE_soccer_league_female_fraction_l2019_201907


namespace NUMINAMATH_CALUDE_second_discount_percentage_l2019_201988

theorem second_discount_percentage
  (normal_price : ℝ)
  (first_discount_rate : ℝ)
  (final_price : ℝ)
  (h1 : normal_price = 174.99999999999997)
  (h2 : first_discount_rate = 0.1)
  (h3 : final_price = 126) :
  let price_after_first_discount := normal_price * (1 - first_discount_rate)
  let second_discount_rate := (price_after_first_discount - final_price) / price_after_first_discount
  second_discount_rate = 0.2 := by
sorry

#eval (174.99999999999997 * 0.9 - 126) / (174.99999999999997 * 0.9)

end NUMINAMATH_CALUDE_second_discount_percentage_l2019_201988


namespace NUMINAMATH_CALUDE_recurrence_sequence_b8_l2019_201992

/-- An increasing sequence of positive integers satisfying the given recurrence relation -/
def RecurrenceSequence (b : ℕ → ℕ) : Prop :=
  (∀ n, b n < b (n + 1)) ∧ 
  (∀ n, 1 ≤ n → b (n + 2) = b (n + 1) + b n)

/-- The theorem statement -/
theorem recurrence_sequence_b8 (b : ℕ → ℕ) 
  (h : RecurrenceSequence b) (h7 : b 7 = 198) : b 8 = 321 := by
  sorry

end NUMINAMATH_CALUDE_recurrence_sequence_b8_l2019_201992


namespace NUMINAMATH_CALUDE_triangle_area_from_sides_and_median_l2019_201976

/-- Given a triangle PQR with side lengths and median, calculate its area -/
theorem triangle_area_from_sides_and_median 
  (PQ PR PM : ℝ) 
  (h_PQ : PQ = 8) 
  (h_PR : PR = 18) 
  (h_PM : PM = 12) : 
  ∃ (area : ℝ), area = Real.sqrt 2975 ∧ area > 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_sides_and_median_l2019_201976


namespace NUMINAMATH_CALUDE_roger_bike_distance_l2019_201938

/-- Calculates the total distance Roger rode his bike over three sessions -/
theorem roger_bike_distance (morning_distance : ℝ) (evening_multiplier : ℝ) (km_per_mile : ℝ) : 
  morning_distance = 2 →
  evening_multiplier = 5 →
  km_per_mile = 1.6 →
  morning_distance + (evening_multiplier * morning_distance) + 
    (2 * morning_distance * km_per_mile / km_per_mile) = 16 := by
  sorry


end NUMINAMATH_CALUDE_roger_bike_distance_l2019_201938


namespace NUMINAMATH_CALUDE_total_cost_is_75_l2019_201903

/-- Calculates the total cost for two siblings attending a music school with a sibling discount -/
def total_cost_for_siblings (regular_tuition : ℕ) (sibling_discount : ℕ) : ℕ :=
  regular_tuition + (regular_tuition - sibling_discount)

/-- Theorem stating that the total cost for two siblings is $75 given the specific tuition and discount -/
theorem total_cost_is_75 :
  total_cost_for_siblings 45 15 = 75 := by
  sorry

#eval total_cost_for_siblings 45 15

end NUMINAMATH_CALUDE_total_cost_is_75_l2019_201903


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l2019_201936

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence :=
  (a : ℤ)  -- First term
  (d : ℤ)  -- Common difference

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.a + (n - 1) * seq.d

theorem arithmetic_sequence_10th_term
  (seq : ArithmeticSequence)
  (h4 : seq.nthTerm 4 = 23)
  (h8 : seq.nthTerm 8 = 55) :
  seq.nthTerm 10 = 71 := by
  sorry

#check arithmetic_sequence_10th_term

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l2019_201936


namespace NUMINAMATH_CALUDE_rectangle_not_unique_l2019_201995

/-- A rectangle is a quadrilateral with four right angles -/
structure Rectangle where
  side1 : ℝ
  side2 : ℝ
  angle : ℝ
  h1 : side1 > 0
  h2 : side2 > 0
  h3 : angle = Real.pi / 2

/-- Given an angle and a side length, there exist multiple distinct rectangles -/
theorem rectangle_not_unique (a : ℝ) (s : ℝ) (h1 : a = Real.pi / 2) (h2 : s > 0) :
  ∃ (r1 r2 : Rectangle), r1 ≠ r2 ∧ r1.angle = a ∧ r1.side1 = s ∧ r2.angle = a ∧ r2.side1 = s :=
sorry

end NUMINAMATH_CALUDE_rectangle_not_unique_l2019_201995


namespace NUMINAMATH_CALUDE_water_evaporation_proof_l2019_201974

theorem water_evaporation_proof (initial_mass : ℝ) (initial_water_percentage : ℝ) 
  (final_water_percentage : ℝ) (evaporated_water : ℝ) : 
  initial_mass = 500 →
  initial_water_percentage = 0.85 →
  final_water_percentage = 0.75 →
  evaporated_water = 200 →
  (initial_mass * initial_water_percentage - evaporated_water) / (initial_mass - evaporated_water) = final_water_percentage :=
by
  sorry

end NUMINAMATH_CALUDE_water_evaporation_proof_l2019_201974


namespace NUMINAMATH_CALUDE_sports_meeting_formation_l2019_201954

/-- The number of performers in the initial formation -/
def initial_performers : ℕ := sorry

/-- The number of performers after adding 16 -/
def after_addition : ℕ := initial_performers + 16

/-- The number of performers after 15 leave -/
def after_leaving : ℕ := after_addition - 15

theorem sports_meeting_formation :
  (∃ n : ℕ, initial_performers = 8 * n) ∧ 
  (∃ m : ℕ, after_addition = m * m) ∧
  (∃ k : ℕ, after_leaving = k * k) →
  initial_performers = 48 := by sorry

end NUMINAMATH_CALUDE_sports_meeting_formation_l2019_201954


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2019_201930

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n : ℚ) / d = 0.714714714 ∧ 
  (∀ (n' d' : ℕ), (n' : ℚ) / d' = 0.714714714 → n ≤ n' ∧ d ≤ d') ∧
  n + d = 571 :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2019_201930


namespace NUMINAMATH_CALUDE_sphere_radii_difference_l2019_201947

theorem sphere_radii_difference (R r : ℝ) : 
  R > r → 
  4 * Real.pi * R^2 - 4 * Real.pi * r^2 = 48 * Real.pi → 
  2 * Real.pi * R + 2 * Real.pi * r = 12 * Real.pi → 
  R - r = 2 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radii_difference_l2019_201947


namespace NUMINAMATH_CALUDE_solution_to_equation_l2019_201914

theorem solution_to_equation (x : ℝ) (h : 1/4 - 1/5 = 1/x) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2019_201914


namespace NUMINAMATH_CALUDE_fraction_inequality_l2019_201980

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : d > c) (h4 : c > 0) : 
  a / c > b / d := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2019_201980


namespace NUMINAMATH_CALUDE_upper_limit_proof_l2019_201940

theorem upper_limit_proof (x : ℝ) (upper_limit : ℝ) : 
  (3 < x ∧ x < 8) → (6 < x ∧ x < upper_limit) → x = 7 → upper_limit > 7 := by
sorry

end NUMINAMATH_CALUDE_upper_limit_proof_l2019_201940


namespace NUMINAMATH_CALUDE_parabola_slope_theorem_l2019_201967

/-- A parabola with equation y² = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- A point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given a parabola and three points on it, prove that the slopes of the lines
    formed by these points satisfy a specific equation -/
theorem parabola_slope_theorem (C : Parabola) (A B P M N : Point) 
  (h_A : A.y^2 = 2 * C.p * A.x) 
  (h_A_x : A.x = 1)
  (h_B : B.y = 0 ∧ B.x = -C.p/2)
  (h_AB : (A.x - B.x)^2 + (A.y - B.y)^2 = 8)
  (h_P : P.y^2 = 2 * C.p * P.x ∧ P.y = 2)
  (h_M : M.y^2 = 2 * C.p * M.x)
  (h_N : N.y^2 = 2 * C.p * N.x)
  (k₁ k₂ k₃ : ℝ)
  (h_k₁ : k₁ = (M.y - P.y) / (M.x - P.x))
  (h_k₂ : k₂ = (N.y - P.y) / (N.x - P.x))
  (h_k₃ : k₃ = (N.y - M.y) / (N.x - M.x)) :
  1/k₁ + 1/k₂ - 1/k₃ = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_slope_theorem_l2019_201967


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2019_201981

theorem perfect_square_condition (n : ℕ+) : 
  ∃ (m : ℕ), 2^n.val + 12^n.val + 2011^n.val = m^2 ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2019_201981


namespace NUMINAMATH_CALUDE_max_intersection_points_l2019_201935

/-- The number of points on the positive x-axis -/
def num_x_points : ℕ := 12

/-- The number of points on the positive y-axis -/
def num_y_points : ℕ := 6

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections : ℕ := 990

/-- Theorem stating the maximum number of intersection points -/
theorem max_intersection_points :
  (num_x_points.choose 2) * (num_y_points.choose 2) = max_intersections := by
  sorry

end NUMINAMATH_CALUDE_max_intersection_points_l2019_201935


namespace NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l2019_201978

theorem vector_subtraction_scalar_multiplication :
  (3 : ℝ) • (((⟨-3, 2, -5⟩ : ℝ × ℝ × ℝ) - ⟨1, 6, 2⟩) : ℝ × ℝ × ℝ) = ⟨-12, -12, -21⟩ := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l2019_201978


namespace NUMINAMATH_CALUDE_polygon_sides_l2019_201946

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 4 * 360 + 180) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2019_201946


namespace NUMINAMATH_CALUDE_division_problem_l2019_201993

theorem division_problem (divisor : ℕ) : 
  (83 / divisor = 9) ∧ (83 % divisor = 2) → divisor = 9 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l2019_201993


namespace NUMINAMATH_CALUDE_valentines_distribution_l2019_201927

theorem valentines_distribution (initial_valentines : Real) (additional_valentines : Real) (num_students : Nat) :
  initial_valentines = 58.0 →
  additional_valentines = 16.0 →
  num_students = 74 →
  (initial_valentines + additional_valentines) / num_students = 1 := by
  sorry

end NUMINAMATH_CALUDE_valentines_distribution_l2019_201927


namespace NUMINAMATH_CALUDE_elizabeth_stickers_l2019_201994

/-- The number of stickers Elizabeth uses on her water bottles -/
def total_stickers (initial_bottles : ℕ) (lost_bottles : ℕ) (stolen_bottles : ℕ) (stickers_per_bottle : ℕ) : ℕ :=
  (initial_bottles - lost_bottles - stolen_bottles) * stickers_per_bottle

/-- Theorem: Elizabeth uses 21 stickers in total -/
theorem elizabeth_stickers :
  total_stickers 10 2 1 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_stickers_l2019_201994


namespace NUMINAMATH_CALUDE_ott_final_fraction_l2019_201902

-- Define the friends
inductive Friend
| Moe
| Loki
| Nick
| Ott
| Pat

-- Define the function for initial money
def initialMoney (f : Friend) : ℚ :=
  match f with
  | Friend.Moe => 14
  | Friend.Loki => 10
  | Friend.Nick => 8
  | Friend.Pat => 12
  | Friend.Ott => 0

-- Define the function for the fraction given by each friend
def fractionGiven (f : Friend) : ℚ :=
  match f with
  | Friend.Moe => 1/7
  | Friend.Loki => 1/5
  | Friend.Nick => 1/4
  | Friend.Pat => 1/6
  | Friend.Ott => 0

-- Define the amount given by each friend
def amountGiven : ℚ := 2

-- Theorem statement
theorem ott_final_fraction :
  let totalInitial := (initialMoney Friend.Moe) + (initialMoney Friend.Loki) + 
                      (initialMoney Friend.Nick) + (initialMoney Friend.Pat)
  let totalGiven := 4 * amountGiven
  (totalGiven / (totalInitial + totalGiven)) = 2/11 := by sorry

end NUMINAMATH_CALUDE_ott_final_fraction_l2019_201902


namespace NUMINAMATH_CALUDE_increasing_root_m_value_l2019_201957

theorem increasing_root_m_value (m : ℝ) : 
  (∃ x : ℝ, (2 * x + 1) / (x - 3) = m / (3 - x) + 1 ∧ 
   ∀ y : ℝ, y > x → (2 * y + 1) / (y - 3) > m / (3 - y) + 1) → 
  m = -7 := by
sorry

end NUMINAMATH_CALUDE_increasing_root_m_value_l2019_201957


namespace NUMINAMATH_CALUDE_james_barrels_l2019_201924

/-- The number of barrels James has -/
def number_of_barrels : ℕ := 3

/-- The capacity of a cask in gallons -/
def cask_capacity : ℕ := 20

/-- The capacity of a barrel in gallons -/
def barrel_capacity : ℕ := 2 * cask_capacity + 3

/-- The total storage capacity in gallons -/
def total_capacity : ℕ := 172

/-- Proof that James has 3 barrels -/
theorem james_barrels :
  number_of_barrels * barrel_capacity + cask_capacity = total_capacity :=
by sorry

end NUMINAMATH_CALUDE_james_barrels_l2019_201924


namespace NUMINAMATH_CALUDE_problem_statement_l2019_201984

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / Real.log x - a * x

theorem problem_statement :
  (∃ (a : ℝ), ∀ (x y : ℝ), 1 < x ∧ x < y → f a y ≤ f a x) ∧
  (∃ (a : ℝ), ∀ (x₁ x₂ : ℝ), Real.exp 1 ≤ x₁ ∧ x₁ ≤ Real.exp 2 ∧
                              Real.exp 1 ≤ x₂ ∧ x₂ ≤ Real.exp 2 →
                              f a x₁ ≤ (deriv (f a)) x₂ + a) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2019_201984


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l2019_201948

/-- Given a square and a circle intersecting such that each side of the square contains
    a chord of the circle with length equal to half the radius of the circle,
    the ratio of the area of the square to the area of the circle is 3/(4π). -/
theorem square_circle_area_ratio (r : ℝ) (h : r > 0) :
  let s := r * Real.sqrt 3 / 2
  (s^2) / (π * r^2) = 3 / (4 * π) := by sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l2019_201948


namespace NUMINAMATH_CALUDE_shoebox_plausibility_l2019_201969

/-- Represents a rectangular prism object -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents possible objects that could match the given dimensions -/
inductive PossibleObject
  | PencilCase
  | MathTextbook
  | Bookshelf
  | Shoebox

/-- Determines if the given dimensions are plausible for a shoebox -/
def is_plausible_shoebox (prism : RectangularPrism) : Prop :=
  prism.length = 35 ∧ prism.width = 20 ∧ prism.height = 15

/-- Theorem stating that a rectangular prism with given dimensions could be a shoebox -/
theorem shoebox_plausibility (prism : RectangularPrism) 
  (h : is_plausible_shoebox prism) : 
  ∃ obj : PossibleObject, obj = PossibleObject.Shoebox := by
  sorry

end NUMINAMATH_CALUDE_shoebox_plausibility_l2019_201969


namespace NUMINAMATH_CALUDE_quadratic_equation_c_value_l2019_201911

theorem quadratic_equation_c_value (b c : ℝ) : 
  (∀ x : ℝ, x^2 - b*x + c = 0 → 
    ∃ y : ℝ, y^2 - b*y + c = 0 ∧ x ≠ y ∧ x * y = 20 ∧ x + y = 12) →
  c = 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_c_value_l2019_201911


namespace NUMINAMATH_CALUDE_two_digit_numbers_with_gcd_lcm_l2019_201955

theorem two_digit_numbers_with_gcd_lcm (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 →
  Nat.gcd a b = 8 →
  Nat.lcm a b = 96 →
  a + b = 56 := by
sorry

end NUMINAMATH_CALUDE_two_digit_numbers_with_gcd_lcm_l2019_201955


namespace NUMINAMATH_CALUDE_largest_c_for_negative_three_in_range_l2019_201979

-- Define the function f
def f (x c : ℝ) : ℝ := x^2 + 5*x + c

-- State the theorem
theorem largest_c_for_negative_three_in_range :
  (∃ (c : ℝ), ∀ (d : ℝ), 
    (∃ (x : ℝ), f x c = -3) → 
    (∃ (y : ℝ), f y d = -3) → 
    d ≤ c) ∧
  (∃ (x : ℝ), f x (13/4) = -3) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_negative_three_in_range_l2019_201979


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2019_201905

theorem simplify_sqrt_expression :
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 242 / Real.sqrt 121) = (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2019_201905


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2019_201971

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n / d : ℚ) = 34 / 99 ∧ 
  (∀ (a b : ℕ), (a / b : ℚ) = 34 / 99 → b ≤ d) ∧ 
  n + d = 133 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2019_201971


namespace NUMINAMATH_CALUDE_prob_different_numbers_is_three_fourths_l2019_201939

/-- Men's team has 3 players -/
def num_men : ℕ := 3

/-- Women's team has 4 players -/
def num_women : ℕ := 4

/-- Total number of possible outcomes when selecting one player from each team -/
def total_outcomes : ℕ := num_men * num_women

/-- Number of outcomes where players have the same number -/
def same_number_outcomes : ℕ := min num_men num_women

/-- Probability of selecting players with different numbers -/
def prob_different_numbers : ℚ := 1 - (same_number_outcomes : ℚ) / total_outcomes

theorem prob_different_numbers_is_three_fourths : 
  prob_different_numbers = 3/4 := by sorry

end NUMINAMATH_CALUDE_prob_different_numbers_is_three_fourths_l2019_201939


namespace NUMINAMATH_CALUDE_triangle_properties_l2019_201997

/-- Given a triangle ABC with angle A = π/3 and perimeter 6, 
    prove the relation between sides and find the maximum area -/
theorem triangle_properties (b c : ℝ) (h_perimeter : b + c ≤ 6) : 
  b * c + 12 = 4 * (b + c) ∧ 
  (∀ (b' c' : ℝ), b' + c' ≤ 6 → 
    (1/2 : ℝ) * b' * c' * Real.sqrt 3 ≤ Real.sqrt 3) := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_triangle_properties_l2019_201997


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l2019_201937

-- Define the parabolas
def Parabola1 (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def Parabola2 (d e f : ℝ) (y : ℝ) : ℝ := d * y^2 + e * y + f

-- Define the intersection points
def IntersectionPoints (a b c d e f : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | Parabola1 a b c x = y ∧ Parabola2 d e f y = x}

-- Define a circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | (x - center.1)^2 + (y - center.2)^2 = radius^2}

-- Theorem statement
theorem intersection_points_on_circle 
  (a b c d e f : ℝ) (ha : a > 0) (hd : d > 0) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    IntersectionPoints a b c d e f ⊆ Circle center radius :=
sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l2019_201937


namespace NUMINAMATH_CALUDE_factorial_ratio_plus_two_l2019_201972

theorem factorial_ratio_plus_two : Nat.factorial 50 / Nat.factorial 48 + 2 = 2452 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_plus_two_l2019_201972


namespace NUMINAMATH_CALUDE_ab_range_l2019_201931

def f (x : ℝ) : ℝ := |2 - x^2|

theorem ab_range (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  0 < a * b ∧ a * b < 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_range_l2019_201931


namespace NUMINAMATH_CALUDE_correct_electric_bicycle_volumes_l2019_201998

/-- Represents the parking data for a day --/
structure ParkingData where
  totalVolume : ℕ
  regularFeeBefore : ℚ
  electricFeeBefore : ℚ
  regularFeeAfter : ℚ
  electricFeeAfter : ℚ
  regularVolumeBefore : ℕ
  regularVolumeAfter : ℕ
  incomeFactor : ℚ

/-- Theorem stating the correct parking volumes for electric bicycles --/
theorem correct_electric_bicycle_volumes (data : ParkingData)
  (h1 : data.totalVolume = 6882)
  (h2 : data.regularFeeBefore = 1/5)
  (h3 : data.electricFeeBefore = 1/2)
  (h4 : data.regularFeeAfter = 2/5)
  (h5 : data.electricFeeAfter = 1)
  (h6 : data.regularVolumeBefore = 5180)
  (h7 : data.regularVolumeAfter = 335)
  (h8 : data.incomeFactor = 3/2) :
  ∃ (x y : ℕ),
    x + y = data.totalVolume - data.regularVolumeBefore - data.regularVolumeAfter ∧
    data.regularFeeBefore * data.regularVolumeBefore +
    data.regularFeeAfter * data.regularVolumeAfter +
    data.electricFeeBefore * x + data.electricFeeAfter * y =
    data.incomeFactor * (data.electricFeeBefore * x + data.electricFeeAfter * y) ∧
    x = 1174 ∧ y = 193 := by
  sorry


end NUMINAMATH_CALUDE_correct_electric_bicycle_volumes_l2019_201998


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2019_201944

theorem inequality_solution_set :
  {x : ℝ | 3 ≤ |5 - 2*x| ∧ |5 - 2*x| < 9} = {x : ℝ | -2 < x ∧ x ≤ 1 ∨ 4 ≤ x ∧ x < 7} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2019_201944


namespace NUMINAMATH_CALUDE_min_value_A_l2019_201917

theorem min_value_A (x y z : ℝ) 
  (hx : x ∈ Set.Ioc 0 1) (hy : y ∈ Set.Ioc 0 1) (hz : z ∈ Set.Ioc 0 1) : 
  let A := ((x + 2*y) * Real.sqrt (x + y - x*y) + 
            (y + 2*z) * Real.sqrt (y + z - y*z) + 
            (z + 2*x) * Real.sqrt (z + x - z*x)) / (x*y + y*z + z*x)
  A ≥ 3 ∧ (A = 3 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_A_l2019_201917


namespace NUMINAMATH_CALUDE_dealer_profit_percentage_l2019_201977

/-- Calculates the profit percentage for a dealer's transaction -/
def profit_percentage (purchase_quantity : ℕ) (purchase_price : ℚ) 
                      (sale_quantity : ℕ) (sale_price : ℚ) : ℚ :=
  let cost_per_article := purchase_price / purchase_quantity
  let sale_per_article := sale_price / sale_quantity
  let profit_per_article := sale_per_article - cost_per_article
  (profit_per_article / cost_per_article) * 100

/-- The profit percentage for the given dealer transaction is approximately 89.99% -/
theorem dealer_profit_percentage :
  let result := profit_percentage 15 25 12 38
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 / 100) ∧ |result - 8999 / 100| < ε :=
sorry

end NUMINAMATH_CALUDE_dealer_profit_percentage_l2019_201977


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_2010_l2019_201960

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_2010 :
  unitsDigit (sumFactorials 2010) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_2010_l2019_201960


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2019_201908

theorem quadratic_expression_value (x : ℝ) (h : 3 * x^2 + 5 * x + 1 = 0) :
  (x + 2)^2 + x * (2 * x + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2019_201908


namespace NUMINAMATH_CALUDE_bad_oranges_l2019_201973

theorem bad_oranges (total_oranges : ℕ) (num_students : ℕ) (reduction : ℕ) : 
  total_oranges = 108 →
  num_students = 12 →
  reduction = 3 →
  (total_oranges / num_students - reduction) * num_students = total_oranges - 36 :=
by sorry

end NUMINAMATH_CALUDE_bad_oranges_l2019_201973


namespace NUMINAMATH_CALUDE_total_earnings_1200_l2019_201923

/-- Represents the prices for services of a car model -/
structure ModelPrices where
  oil_change : ℕ
  repair : ℕ
  car_wash : ℕ

/-- Represents the number of services performed for a car model -/
structure ModelServices where
  oil_changes : ℕ
  repairs : ℕ
  car_washes : ℕ

/-- Calculates the total earnings for a single car model -/
def modelEarnings (prices : ModelPrices) (services : ModelServices) : ℕ :=
  prices.oil_change * services.oil_changes +
  prices.repair * services.repairs +
  prices.car_wash * services.car_washes

/-- Theorem stating that the total earnings for the day is $1200 -/
theorem total_earnings_1200 
  (prices_A : ModelPrices)
  (prices_B : ModelPrices)
  (prices_C : ModelPrices)
  (services_A : ModelServices)
  (services_B : ModelServices)
  (services_C : ModelServices)
  (h1 : prices_A = ⟨20, 30, 5⟩)
  (h2 : prices_B = ⟨25, 40, 8⟩)
  (h3 : prices_C = ⟨30, 50, 10⟩)
  (h4 : services_A = ⟨5, 10, 15⟩)
  (h5 : services_B = ⟨3, 4, 10⟩)
  (h6 : services_C = ⟨2, 6, 5⟩) :
  modelEarnings prices_A services_A + 
  modelEarnings prices_B services_B + 
  modelEarnings prices_C services_C = 1200 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_1200_l2019_201923


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2019_201926

theorem arithmetic_calculation : (10 - 9 + 8) * 7 + 6 - 5 * (4 - 3 + 2) - 1 = 53 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2019_201926


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2019_201949

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^5 + 3 * x^4 + x^3 - 5 * x + 2) + (x^5 - 3 * x^4 - 2 * x^3 + x^2 + 5 * x - 7) = 
  3 * x^5 - x^3 + x^2 - 5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2019_201949


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2019_201996

theorem complex_modulus_problem (m : ℝ) :
  (Complex.I : ℂ) * Complex.I = -1 →
  (↑1 + m * Complex.I) * (↑3 + Complex.I) = Complex.I * (Complex.im ((↑1 + m * Complex.I) * (↑3 + Complex.I))) →
  Complex.abs ((↑m + ↑3 * Complex.I) / (↑1 - Complex.I)) = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2019_201996


namespace NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_two_l2019_201965

theorem no_solution_iff_n_eq_neg_two (n : ℤ) :
  (∀ x y : ℚ, 2 * x = 1 + n * y ∧ n * x = 1 + 2 * y) ↔ n = -2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_two_l2019_201965


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2019_201959

theorem complex_modulus_problem (z : ℂ) (h : (3 + 4 * Complex.I) * z = 1) : Complex.abs z = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2019_201959


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2019_201920

/-- A parabola is defined by its equation y² = x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  is_parabola : equation = fun y x => y^2 = x

/-- The distance between the focus and directrix of a parabola -/
def focus_directrix_distance (p : Parabola) : ℝ := sorry

/-- Theorem: The distance between the focus and directrix of the parabola y² = x is 0.5 -/
theorem parabola_focus_directrix_distance :
  ∀ p : Parabola, p.equation = fun y x => y^2 = x → focus_directrix_distance p = 0.5 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2019_201920


namespace NUMINAMATH_CALUDE_computer_discount_theorem_l2019_201919

theorem computer_discount_theorem (saved : ℝ) (paid : ℝ) (additional_discount : ℝ) :
  saved = 120 →
  paid = 1080 →
  additional_discount = 0.05 →
  let original_price := saved + paid
  let first_discount_percentage := saved / original_price
  let second_discount_amount := additional_discount * paid
  let total_saved := saved + second_discount_amount
  let total_percentage_saved := total_saved / original_price
  total_percentage_saved = 0.145 := by
  sorry

end NUMINAMATH_CALUDE_computer_discount_theorem_l2019_201919


namespace NUMINAMATH_CALUDE_probability_of_selecting_two_specific_elements_l2019_201901

theorem probability_of_selecting_two_specific_elements 
  (total_elements : Nat) 
  (elements_to_select : Nat) 
  (h1 : total_elements = 6) 
  (h2 : elements_to_select = 2) :
  (1 : ℚ) / (Nat.choose total_elements elements_to_select) = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selecting_two_specific_elements_l2019_201901


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l2019_201975

theorem rectangle_diagonal (x y a b : ℝ) (h1 : π * x^2 * y = a) (h2 : π * y^2 * x = b) :
  (x^2 + y^2).sqrt = ((a^2 + b^2) / (a * b)).sqrt * ((a * b) / π^2)^(1/6) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l2019_201975


namespace NUMINAMATH_CALUDE_coeff_4th_term_of_1_minus_2x_to_15_l2019_201928

/-- The coefficient of the 4th term in the expansion of (1-2x)^15 -/
def coeff_4th_term : ℤ := -3640

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem coeff_4th_term_of_1_minus_2x_to_15 :
  coeff_4th_term = (-2)^3 * (binomial 15 3) := by sorry

end NUMINAMATH_CALUDE_coeff_4th_term_of_1_minus_2x_to_15_l2019_201928


namespace NUMINAMATH_CALUDE_right_triangle_acute_angle_theorem_l2019_201951

theorem right_triangle_acute_angle_theorem :
  ∀ (a b : ℝ), 
  a > 0 ∧ b > 0 →  -- Ensuring positive angles
  a = 2 * b →      -- One acute angle is twice the other
  a + b = 90 →     -- Sum of acute angles in a right triangle is 90°
  a = 60 :=        -- The larger acute angle is 60°
by sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angle_theorem_l2019_201951


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l2019_201921

theorem part_to_whole_ratio (N : ℝ) (h1 : (1/3) * (2/5) * N = 17) (h2 : 0.4 * N = 204) : 
  17 / N = 1 / 30 := by
sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l2019_201921


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l2019_201989

theorem min_value_of_fraction (a : ℝ) (h : a > 1) : 
  ∀ x : ℝ, x > 1 → (x^2 - x + 1) / (x - 1) ≥ (a^2 - a + 1) / (a - 1) → 
  (a^2 - a + 1) / (a - 1) = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l2019_201989


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2019_201985

def P (x : ℝ) : ℝ := x^3 - 7*x^2 + 14*x - 8

theorem roots_of_polynomial :
  (∀ x : ℝ, P x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 4) ∧
  (∀ x : ℝ, (x - 1) * (x - 2) * (x - 4) = P x) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2019_201985


namespace NUMINAMATH_CALUDE_composite_shape_perimeter_l2019_201916

/-- A figure composed of two unit squares and one unit equilateral triangle. -/
structure CompositeShape where
  /-- The side length of each square -/
  square_side : ℝ
  /-- The side length of the equilateral triangle -/
  triangle_side : ℝ
  /-- Assertion that both squares and the triangle have unit side length -/
  h_unit_sides : square_side = 1 ∧ triangle_side = 1

/-- The perimeter of the composite shape -/
def perimeter (shape : CompositeShape) : ℝ :=
  3 * shape.square_side + 2 * shape.triangle_side

/-- Theorem stating that the perimeter of the composite shape is 5 units -/
theorem composite_shape_perimeter (shape : CompositeShape) :
  perimeter shape = 5 :=
sorry

end NUMINAMATH_CALUDE_composite_shape_perimeter_l2019_201916


namespace NUMINAMATH_CALUDE_not_cube_of_integer_l2019_201932

theorem not_cube_of_integer : ¬ ∃ (k : ℤ), 10^202 + 5 * 10^101 + 1 = k^3 := by sorry

end NUMINAMATH_CALUDE_not_cube_of_integer_l2019_201932


namespace NUMINAMATH_CALUDE_average_decrease_l2019_201982

theorem average_decrease (n : ℕ) (initial_avg : ℚ) (new_obs : ℚ) : 
  n = 6 → 
  initial_avg = 11 → 
  new_obs = 4 → 
  (n * initial_avg + new_obs) / (n + 1) = initial_avg - 1 := by
sorry

end NUMINAMATH_CALUDE_average_decrease_l2019_201982


namespace NUMINAMATH_CALUDE_weekly_training_cost_l2019_201945

/-- Proves that the weekly training cost is $250, given the adoption fee, training duration, certification cost, insurance coverage, and total out-of-pocket cost. -/
theorem weekly_training_cost
  (adoption_fee : ℝ)
  (training_weeks : ℕ)
  (certification_cost : ℝ)
  (insurance_coverage : ℝ)
  (total_out_of_pocket : ℝ)
  (h1 : adoption_fee = 150)
  (h2 : training_weeks = 12)
  (h3 : certification_cost = 3000)
  (h4 : insurance_coverage = 0.9)
  (h5 : total_out_of_pocket = 3450)
  : ∃ (weekly_cost : ℝ),
    weekly_cost = 250 ∧
    total_out_of_pocket = adoption_fee + training_weeks * weekly_cost + (1 - insurance_coverage) * certification_cost :=
by sorry

end NUMINAMATH_CALUDE_weekly_training_cost_l2019_201945


namespace NUMINAMATH_CALUDE_average_weight_problem_l2019_201950

/-- Given the average weight of three people and two of them, prove the average weight of two of them. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 43 →  -- The average weight of a, b, and c is 43 kg
  (a + b) / 2 = 40 →      -- The average weight of a and b is 40 kg
  b = 37 →                -- The weight of b is 37 kg
  (b + c) / 2 = 43        -- The average weight of b and c is 43 kg
  := by sorry

end NUMINAMATH_CALUDE_average_weight_problem_l2019_201950


namespace NUMINAMATH_CALUDE_sqrt_a_power_b_equals_three_l2019_201900

theorem sqrt_a_power_b_equals_three (a b : ℝ) 
  (h : a^2 - 6*a + Real.sqrt (2*b - 4) = -9) : 
  Real.sqrt (a^b) = 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_a_power_b_equals_three_l2019_201900


namespace NUMINAMATH_CALUDE_sum_of_digits_c_plus_d_l2019_201915

/-- The sum of digits of c + d, where c and d are defined as follows:
    c = 10^1986 - 1
    d = 6(10^1986 - 1)/9 -/
theorem sum_of_digits_c_plus_d : ℕ :=
  let c : ℕ := 10^1986 - 1
  let d : ℕ := 6 * (10^1986 - 1) / 9
  9931

#check sum_of_digits_c_plus_d

end NUMINAMATH_CALUDE_sum_of_digits_c_plus_d_l2019_201915


namespace NUMINAMATH_CALUDE_passengers_boarded_in_north_carolina_l2019_201968

/-- Represents the number of passengers at different stages of the flight --/
structure FlightPassengers where
  initial : Nat
  afterTexas : Nat
  afterNorthCarolina : Nat
  final : Nat

/-- Represents the changes in passenger numbers during layovers --/
structure LayoverChanges where
  texasOff : Nat
  texasOn : Nat
  northCarolinaOff : Nat

/-- The main theorem about the flight --/
theorem passengers_boarded_in_north_carolina 
  (fp : FlightPassengers) 
  (lc : LayoverChanges) 
  (crew : Nat) 
  (h1 : fp.initial = 124)
  (h2 : lc.texasOff = 58)
  (h3 : lc.texasOn = 24)
  (h4 : lc.northCarolinaOff = 47)
  (h5 : crew = 10)
  (h6 : fp.final + crew = 67)
  (h7 : fp.afterTexas = fp.initial - lc.texasOff + lc.texasOn)
  (h8 : fp.afterNorthCarolina = fp.afterTexas - lc.northCarolinaOff)
  : fp.final - fp.afterNorthCarolina = 14 := by
  sorry

#check passengers_boarded_in_north_carolina

end NUMINAMATH_CALUDE_passengers_boarded_in_north_carolina_l2019_201968


namespace NUMINAMATH_CALUDE_initial_green_papayas_l2019_201922

/-- The number of green papayas that turned yellow on Friday -/
def friday_yellow : ℕ := 2

/-- The number of green papayas that turned yellow on Sunday -/
def sunday_yellow : ℕ := 2 * friday_yellow

/-- The number of green papayas left on the tree -/
def remaining_green : ℕ := 8

/-- The initial number of green papayas on the tree -/
def initial_green : ℕ := remaining_green + friday_yellow + sunday_yellow

theorem initial_green_papayas : initial_green = 14 := by
  sorry

end NUMINAMATH_CALUDE_initial_green_papayas_l2019_201922


namespace NUMINAMATH_CALUDE_m_less_than_n_l2019_201925

/-- Represents a quadratic function f(x) = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines N based on the coefficients of a quadratic function -/
def N (f : QuadraticFunction) : ℝ :=
  |f.a + f.b + f.c| + |2*f.a - f.b|

/-- Defines M based on the coefficients of a quadratic function -/
def M (f : QuadraticFunction) : ℝ :=
  |f.a - f.b + f.c| + |2*f.a + f.b|

/-- Theorem stating that M < N for a quadratic function satisfying certain conditions -/
theorem m_less_than_n (f : QuadraticFunction)
  (h1 : f.a + f.b + f.c < 0)
  (h2 : f.a - f.b + f.c > 0)
  (h3 : f.a > 0)
  (h4 : -f.b / (2 * f.a) > 1) :
  M f < N f := by
  sorry

end NUMINAMATH_CALUDE_m_less_than_n_l2019_201925


namespace NUMINAMATH_CALUDE_cos_A_from_tan_A_l2019_201933

theorem cos_A_from_tan_A (A : Real) (h : Real.tan A = 2/3) : 
  Real.cos A = 3 * Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_cos_A_from_tan_A_l2019_201933


namespace NUMINAMATH_CALUDE_second_train_length_l2019_201934

-- Define constants
def train1_speed : Real := 60  -- km/hr
def train2_speed : Real := 40  -- km/hr
def crossing_time : Real := 11.159107271418288  -- seconds
def train1_length : Real := 140  -- meters

-- Define the theorem
theorem second_train_length :
  let relative_speed := (train1_speed + train2_speed) * (5/18)  -- Convert km/hr to m/s
  let total_distance := relative_speed * crossing_time
  let train2_length := total_distance - train1_length
  train2_length = 170 := by
  sorry

end NUMINAMATH_CALUDE_second_train_length_l2019_201934
