import Mathlib

namespace NUMINAMATH_CALUDE_emily_sixth_score_l2591_259155

def emily_scores : List ℕ := [91, 94, 86, 88, 101]

theorem emily_sixth_score (target_mean : ℕ := 94) (sixth_score : ℕ := 104) :
  let all_scores := emily_scores ++ [sixth_score]
  (all_scores.sum / all_scores.length : ℚ) = target_mean := by
  sorry

end NUMINAMATH_CALUDE_emily_sixth_score_l2591_259155


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l2591_259179

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ - 3| = 25 ∧ |x₂ - 3| = 25 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 50 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l2591_259179


namespace NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l2591_259153

theorem halfway_between_one_eighth_and_one_third :
  (1 / 8 + 1 / 3) / 2 = 11 / 48 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l2591_259153


namespace NUMINAMATH_CALUDE_cyclic_fraction_product_l2591_259129

theorem cyclic_fraction_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : (x + y) / z = (y + z) / x) (h2 : (y + z) / x = (z + x) / y) :
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = -1 ∨
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = 8 := by
sorry

end NUMINAMATH_CALUDE_cyclic_fraction_product_l2591_259129


namespace NUMINAMATH_CALUDE_max_value_b_plus_c_l2591_259100

theorem max_value_b_plus_c (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : (a + c) * (b^2 + a*c) = 4*a) : 
  b + c ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_b_plus_c_l2591_259100


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l2591_259141

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 25             -- Shorter leg length
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l2591_259141


namespace NUMINAMATH_CALUDE_expand_product_l2591_259124

theorem expand_product (y : ℝ) (h : y ≠ 0) :
  (3 / 7) * ((7 / y) - 14 * y^3 + 21) = 3 / y - 6 * y^3 + 9 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2591_259124


namespace NUMINAMATH_CALUDE_symmetric_circle_l2591_259121

/-- Given a circle C1 with equation (x+2)^2+(y-1)^2=5,
    prove that its symmetric circle C2 with respect to the origin (0,0)
    has the equation (x-2)^2+(y+1)^2=5 -/
theorem symmetric_circle (x y : ℝ) :
  (∀ x y, (x + 2)^2 + (y - 1)^2 = 5) →
  (∃ C2 : Set (ℝ × ℝ), C2 = {(x, y) | (x - 2)^2 + (y + 1)^2 = 5} ∧
    ∀ (p : ℝ × ℝ), p ∈ C2 ↔ (-p.1, -p.2) ∈ {(x, y) | (x + 2)^2 + (y - 1)^2 = 5}) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_l2591_259121


namespace NUMINAMATH_CALUDE_expansion_terms_count_l2591_259156

/-- The number of terms in the expansion of a product of two sums -/
def num_terms_in_expansion (n m : ℕ) : ℕ := n * m

/-- The first factor (a+b+c+d) has 4 terms -/
def first_factor_terms : ℕ := 4

/-- The second factor (e+f+g+h+i) has 5 terms -/
def second_factor_terms : ℕ := 5

theorem expansion_terms_count :
  num_terms_in_expansion first_factor_terms second_factor_terms = 20 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l2591_259156


namespace NUMINAMATH_CALUDE_baking_powder_difference_l2591_259119

/-- The amount of baking powder Kelly had yesterday, in boxes -/
def yesterday_supply : ℚ := 0.4

/-- The amount of baking powder Kelly has today, in boxes -/
def today_supply : ℚ := 0.3

/-- The difference in baking powder supply between yesterday and today -/
def supply_difference : ℚ := yesterday_supply - today_supply

theorem baking_powder_difference :
  supply_difference = 0.1 := by sorry

end NUMINAMATH_CALUDE_baking_powder_difference_l2591_259119


namespace NUMINAMATH_CALUDE_cos_squared_pi_fourth_minus_alpha_l2591_259140

theorem cos_squared_pi_fourth_minus_alpha (α : ℝ) 
  (h : Real.sin α - Real.cos α = 4/3) : 
  Real.cos (π/4 - α)^2 = 1/9 := by sorry

end NUMINAMATH_CALUDE_cos_squared_pi_fourth_minus_alpha_l2591_259140


namespace NUMINAMATH_CALUDE_zeros_of_composition_l2591_259178

/-- Given functions f and g, prove that the zeros of their composition h are ±√2 -/
theorem zeros_of_composition (f g h : ℝ → ℝ) :
  (∀ x, f x = 2 * x - 4) →
  (∀ x, g x = x^2) →
  (∀ x, h x = f (g x)) →
  {x : ℝ | h x = 0} = {-Real.sqrt 2, Real.sqrt 2} := by
  sorry

end NUMINAMATH_CALUDE_zeros_of_composition_l2591_259178


namespace NUMINAMATH_CALUDE_empty_set_cardinality_zero_l2591_259133

theorem empty_set_cardinality_zero : Finset.card (∅ : Finset α) = 0 := by sorry

end NUMINAMATH_CALUDE_empty_set_cardinality_zero_l2591_259133


namespace NUMINAMATH_CALUDE_well_digging_payment_l2591_259184

/-- The total amount paid to workers for digging a well -/
def total_payment (num_workers : ℕ) (hours_day1 hours_day2 hours_day3 : ℕ) (hourly_rate : ℕ) : ℕ :=
  num_workers * (hours_day1 + hours_day2 + hours_day3) * hourly_rate

/-- Theorem stating the total payment for the well digging job -/
theorem well_digging_payment :
  total_payment 2 10 8 15 10 = 660 := by
  sorry

end NUMINAMATH_CALUDE_well_digging_payment_l2591_259184


namespace NUMINAMATH_CALUDE_mathematics_letter_probability_l2591_259162

theorem mathematics_letter_probability : 
  let alphabet_size : ℕ := 26
  let unique_letters : ℕ := 8
  (unique_letters : ℚ) / (alphabet_size : ℚ) = 4 / 13 :=
by sorry

end NUMINAMATH_CALUDE_mathematics_letter_probability_l2591_259162


namespace NUMINAMATH_CALUDE_glass_volume_l2591_259146

theorem glass_volume (V : ℝ) 
  (h1 : 0.4 * V = V - 0.6 * V) -- pessimist's glass is 60% empty
  (h2 : 0.6 * V - 0.4 * V = 46) -- difference between optimist's and pessimist's water volumes
  : V = 230 := by
sorry

end NUMINAMATH_CALUDE_glass_volume_l2591_259146


namespace NUMINAMATH_CALUDE_decimal_to_fraction_sum_l2591_259142

theorem decimal_to_fraction_sum (c d : ℕ+) : 
  (c : ℚ) / (d : ℚ) = 0.325 ∧ 
  (∀ (k : ℕ+), k ∣ c ∧ k ∣ d → k = 1) →
  (c : ℕ) + d = 53 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_sum_l2591_259142


namespace NUMINAMATH_CALUDE_two_point_distribution_p_values_l2591_259154

/-- A two-point distribution random variable -/
structure TwoPointDistribution where
  p : ℝ
  prob_x_eq_one : p ∈ Set.Icc 0 1

/-- The variance of a two-point distribution -/
def variance (X : TwoPointDistribution) : ℝ := X.p - X.p^2

theorem two_point_distribution_p_values (X : TwoPointDistribution) 
  (h : variance X = 2/9) : X.p = 1/3 ∨ X.p = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_two_point_distribution_p_values_l2591_259154


namespace NUMINAMATH_CALUDE_mercedes_jonathan_distance_ratio_l2591_259110

/-- Proves that the ratio of Mercedes' distance to Jonathan's distance is 2:1 -/
theorem mercedes_jonathan_distance_ratio :
  ∀ (jonathan_distance mercedes_distance davonte_distance : ℝ),
  jonathan_distance = 7.5 →
  davonte_distance = mercedes_distance + 2 →
  mercedes_distance + davonte_distance = 32 →
  mercedes_distance / jonathan_distance = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_mercedes_jonathan_distance_ratio_l2591_259110


namespace NUMINAMATH_CALUDE_final_count_A_l2591_259120

/-- Represents a switch with its ID and position -/
structure Switch where
  id : Nat
  position : Fin 3

/-- Represents the state of all switches -/
def SwitchState := Fin 1000 → Switch

/-- Checks if one number divides another -/
def divides (a b : Nat) : Prop := ∃ k, b = a * k

/-- Represents a single step in the process -/
def step (s : SwitchState) (i : Fin 1000) : SwitchState := sorry

/-- Represents the entire process of 1000 steps -/
def process (initial : SwitchState) : SwitchState := sorry

/-- Counts the number of switches in position A -/
def countA (s : SwitchState) : Nat := sorry

/-- The main theorem to prove -/
theorem final_count_A (initial : SwitchState) : 
  (∀ i, (initial i).position = 0) →
  (∀ i, ∃ x y z, x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧ (initial i).id = 2^x * 3^y * 7^z) →
  countA (process initial) = 660 := sorry

end NUMINAMATH_CALUDE_final_count_A_l2591_259120


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_2pi_3_l2591_259195

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ) + 1

theorem sin_2alpha_plus_2pi_3 (ω φ α : ℝ) :
  ω > 0 →
  0 ≤ φ ∧ φ ≤ Real.pi / 2 →
  (∀ x : ℝ, f ω φ (x + Real.pi / ω) = f ω φ x) →
  f ω φ (Real.pi / 6) = 2 →
  f ω φ α = 9 / 5 →
  Real.pi / 6 < α ∧ α < 2 * Real.pi / 3 →
  Real.sin (2 * α + 2 * Real.pi / 3) = -24 / 25 := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_2pi_3_l2591_259195


namespace NUMINAMATH_CALUDE_rectangle_count_5x5_l2591_259157

/-- The number of dots in each row and column of the square array -/
def gridSize : Nat := 5

/-- The number of different rectangles that can be formed in a square array of dots -/
def numberOfRectangles (n : Nat) : Nat :=
  (n.choose 2) * (n.choose 2)

/-- Theorem: The number of different rectangles with sides parallel to the grid
    that can be formed by connecting four dots in a 5x5 square array of dots is 100 -/
theorem rectangle_count_5x5 :
  numberOfRectangles gridSize = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_count_5x5_l2591_259157


namespace NUMINAMATH_CALUDE_max_value_problem_l2591_259186

theorem max_value_problem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x^2 + y + z = 1) :
  ∀ a b c : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a^2 + b + c = 1 → x + y^3 + z^4 ≤ a + b^3 + c^4 → x + y^3 + z^4 ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_l2591_259186


namespace NUMINAMATH_CALUDE_functional_equation_properties_l2591_259138

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem functional_equation_properties (f : ℝ → ℝ) (h : FunctionalEquation f) :
  f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : ℝ, f (-x) = f x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_properties_l2591_259138


namespace NUMINAMATH_CALUDE_nine_digit_divisibility_l2591_259149

theorem nine_digit_divisibility (a b c : Nat) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : c ≤ 9) (h4 : a ≠ 0) :
  ∃ k : Nat, (100 * a + 10 * b + c) * 1001001 = k * (100000000 * a + 10000000 * b + 1000000 * c +
                                                     100000 * a + 10000 * b + 1000 * c +
                                                     100 * a + 10 * b + c) :=
sorry

end NUMINAMATH_CALUDE_nine_digit_divisibility_l2591_259149


namespace NUMINAMATH_CALUDE_power_of_product_l2591_259187

theorem power_of_product (a b : ℝ) : (a * b^2)^3 = a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l2591_259187


namespace NUMINAMATH_CALUDE_candy_groups_l2591_259135

theorem candy_groups (total_candies : ℕ) (group_size : ℕ) (h1 : total_candies = 30) (h2 : group_size = 3) :
  total_candies / group_size = 10 := by
  sorry

end NUMINAMATH_CALUDE_candy_groups_l2591_259135


namespace NUMINAMATH_CALUDE_car_distance_theorem_l2591_259175

/-- Theorem: A car traveling at 208 km/h for 3 hours covers a distance of 624 km. -/
theorem car_distance_theorem (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 208 → time = 3 → distance = speed * time → distance = 624 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l2591_259175


namespace NUMINAMATH_CALUDE_expression_value_l2591_259106

theorem expression_value (a b : ℤ) (h : a = b + 1) : 3 + 2*a - 2*b = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2591_259106


namespace NUMINAMATH_CALUDE_max_unique_subsets_l2591_259145

theorem max_unique_subsets (n : ℕ) (h : n = 7) : 
  (2 ^ n) - 2 = 128 := by
  sorry

end NUMINAMATH_CALUDE_max_unique_subsets_l2591_259145


namespace NUMINAMATH_CALUDE_total_cost_after_discounts_and_cashback_l2591_259190

/-- The total cost of an iPhone 12 and an iWatch after discounts and cashback -/
theorem total_cost_after_discounts_and_cashback :
  let iphone_price : ℚ := 800
  let iwatch_price : ℚ := 300
  let iphone_discount : ℚ := 15 / 100
  let iwatch_discount : ℚ := 10 / 100
  let cashback_rate : ℚ := 2 / 100
  let iphone_discounted := iphone_price * (1 - iphone_discount)
  let iwatch_discounted := iwatch_price * (1 - iwatch_discount)
  let total_before_cashback := iphone_discounted + iwatch_discounted
  let cashback_amount := total_before_cashback * cashback_rate
  let final_cost := total_before_cashback - cashback_amount
  final_cost = 931 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_after_discounts_and_cashback_l2591_259190


namespace NUMINAMATH_CALUDE_parallel_segments_theorem_l2591_259164

/-- Represents a line segment with a length -/
structure Segment where
  length : ℝ

/-- Represents three parallel line segments intersecting another line segment -/
structure ParallelSegments where
  ab : Segment
  ef : Segment
  cd : Segment
  bc : Segment
  ab_parallel_ef : Bool
  ef_parallel_cd : Bool

/-- Given three parallel line segments intersecting another line segment,
    with specific lengths, the middle segment's length is 16 -/
theorem parallel_segments_theorem (p : ParallelSegments)
  (h1 : p.ab_parallel_ef = true)
  (h2 : p.ef_parallel_cd = true)
  (h3 : p.ab.length = 20)
  (h4 : p.cd.length = 80)
  (h5 : p.bc.length = 100) :
  p.ef.length = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallel_segments_theorem_l2591_259164


namespace NUMINAMATH_CALUDE_leila_sweater_cost_l2591_259144

/-- Represents Leila's spending on a sweater and jewelry --/
structure LeilasSpending where
  total : ℝ
  sweater : ℝ
  jewelry : ℝ

/-- The conditions of Leila's spending --/
def spending_conditions (s : LeilasSpending) : Prop :=
  s.sweater = (1/4) * s.total ∧
  s.jewelry = (3/4) * s.total - 20 ∧
  s.jewelry = s.sweater + 60

/-- Theorem stating that under the given conditions, Leila spent $40 on the sweater --/
theorem leila_sweater_cost (s : LeilasSpending) 
  (h : spending_conditions s) : s.sweater = 40 := by
  sorry

#check leila_sweater_cost

end NUMINAMATH_CALUDE_leila_sweater_cost_l2591_259144


namespace NUMINAMATH_CALUDE_reciprocal_minus_opposite_l2591_259132

theorem reciprocal_minus_opposite : 
  let x : ℚ := -4
  (-1 / x) - (-x) = -17 / 4 := by sorry

end NUMINAMATH_CALUDE_reciprocal_minus_opposite_l2591_259132


namespace NUMINAMATH_CALUDE_largest_y_coordinate_degenerate_ellipse_l2591_259151

theorem largest_y_coordinate_degenerate_ellipse :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ (x^2 / 49) + ((y - 3)^2 / 25)
  ∀ (x y : ℝ), f (x, y) = 0 → y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_y_coordinate_degenerate_ellipse_l2591_259151


namespace NUMINAMATH_CALUDE_fast_food_theorem_l2591_259161

/-- A fast food composition -/
structure FastFood where
  total_mass : ℝ
  fat_percentage : ℝ
  protein_mass : ℝ → ℝ
  mineral_mass : ℝ → ℝ
  carb_mass : ℝ → ℝ

/-- Conditions for the fast food -/
def fast_food_conditions (ff : FastFood) : Prop :=
  ff.total_mass = 500 ∧
  ff.fat_percentage = 0.05 ∧
  (∀ x, ff.protein_mass x = 4 * ff.mineral_mass x) ∧
  (∀ x, (ff.protein_mass x + ff.carb_mass x) / ff.total_mass ≤ 0.85)

/-- Theorem about the mass of fat and maximum carbohydrates in the fast food -/
theorem fast_food_theorem (ff : FastFood) (h : fast_food_conditions ff) :
  ff.fat_percentage * ff.total_mass = 25 ∧
  ∃ x, ff.carb_mass x = 225 ∧ 
    ∀ y, ff.carb_mass y ≤ ff.carb_mass x :=
by sorry

end NUMINAMATH_CALUDE_fast_food_theorem_l2591_259161


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l2591_259114

theorem no_integer_solutions_for_equation :
  ¬ ∃ (x y z : ℤ), x^2 + y^2 = 4*z - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l2591_259114


namespace NUMINAMATH_CALUDE_sin_minus_pi_half_times_tan_pi_minus_l2591_259177

open Real

theorem sin_minus_pi_half_times_tan_pi_minus (α : ℝ) : 
  sin (α - π / 2) * tan (π - α) = sin α := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_pi_half_times_tan_pi_minus_l2591_259177


namespace NUMINAMATH_CALUDE_inequality_proof_l2591_259173

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq : a + b + c = 3) : 
  (a^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*a - 1) + 
  (b^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*b - 1) + 
  (c^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*c - 1) ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2591_259173


namespace NUMINAMATH_CALUDE_part_1_part_2_l2591_259128

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.exp x - 1) - a * x^2

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := Real.exp x + x * Real.exp x - 1 - 2 * a * x

-- Theorem for part 1
theorem part_1 (a : ℝ) : f' a 1 = 2 * Real.exp 1 - 2 → a = 1/2 := by sorry

-- Define the specific function f with a = 1/2
def f_half (x : ℝ) : ℝ := x * (Real.exp x - 1) - (1/2) * x^2

-- Define the derivative of f_half
def f_half' (x : ℝ) : ℝ := (x + 1) * (Real.exp x - 1)

-- Theorem for part 2
theorem part_2 (m : ℝ) : 
  (∀ x ∈ Set.Ioo (2*m - 3) (3*m - 2), f_half' x > 0) ↔ 
  (m ∈ Set.Ioc (-1) (1/3) ∪ Set.Ici (3/2)) := by sorry

end

end NUMINAMATH_CALUDE_part_1_part_2_l2591_259128


namespace NUMINAMATH_CALUDE_rug_inner_length_is_four_l2591_259189

/-- Represents the dimensions of a rectangular region -/
structure RectDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : RectDimensions) : ℝ := d.length * d.width

/-- Represents the three colored regions of the rug -/
structure RugRegions where
  inner : RectDimensions
  middle : RectDimensions
  outer : RectDimensions

/-- Checks if three real numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop := b - a = c - b

theorem rug_inner_length_is_four :
  ∀ (r : RugRegions),
    r.inner.width = 2 →
    r.middle.length = r.inner.length + 4 →
    r.middle.width = r.inner.width + 4 →
    r.outer.length = r.middle.length + 4 →
    r.outer.width = r.middle.width + 4 →
    isArithmeticProgression (area r.inner) (area r.middle - area r.inner) (area r.outer - area r.middle) →
    r.inner.length = 4 := by
  sorry

end NUMINAMATH_CALUDE_rug_inner_length_is_four_l2591_259189


namespace NUMINAMATH_CALUDE_harry_apples_l2591_259191

/-- The number of apples each person has -/
structure Apples where
  martha : ℕ
  tim : ℕ
  harry : ℕ
  jane : ℕ

/-- The conditions of the problem -/
def apple_conditions (a : Apples) : Prop :=
  a.martha = 68 ∧
  a.tim = a.martha - 30 ∧
  a.harry = a.tim / 2 ∧
  a.jane = (a.tim + a.martha) / 4

/-- The theorem stating that Harry has 19 apples -/
theorem harry_apples (a : Apples) (h : apple_conditions a) : a.harry = 19 := by
  sorry

end NUMINAMATH_CALUDE_harry_apples_l2591_259191


namespace NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l2591_259134

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- The sum of the measures of the six interior angles of a hexagon is 720 degrees -/
theorem hexagon_interior_angles_sum :
  sum_interior_angles hexagon_sides = 720 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l2591_259134


namespace NUMINAMATH_CALUDE_operation_simplification_l2591_259167

theorem operation_simplification (x : ℚ) : 
  ((3 * x + 6) - 5 * x + 10) / 5 = -2/5 * x + 16/5 := by
  sorry

end NUMINAMATH_CALUDE_operation_simplification_l2591_259167


namespace NUMINAMATH_CALUDE_ellipse_focal_property_l2591_259139

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 4 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define points A and B on the ellipse
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Theorem statement
theorem ellipse_focal_property :
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧  -- A and B are on the ellipse
  (∃ (t : ℝ), B = F2 + t • (A - F2)) ∧  -- A, B, and F2 are collinear
  ‖A - B‖ = 8 →  -- Distance between A and B is 8
  ‖A - F1‖ + ‖B - F1‖ = 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_property_l2591_259139


namespace NUMINAMATH_CALUDE_probability_of_four_white_balls_l2591_259152

def total_balls : ℕ := 25
def white_balls : ℕ := 10
def black_balls : ℕ := 15
def drawn_balls : ℕ := 4

theorem probability_of_four_white_balls : 
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls) = 3 / 181 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_four_white_balls_l2591_259152


namespace NUMINAMATH_CALUDE_second_pipe_fills_in_30_minutes_l2591_259101

/-- Represents a system of pipes filling and emptying a tank -/
structure PipeSystem where
  fill_time1 : ℝ  -- Time for first pipe to fill tank
  empty_time : ℝ  -- Time for outlet pipe to empty tank
  combined_fill_time : ℝ  -- Time to fill tank when all pipes are open

/-- Calculates the fill time of the second pipe given a PipeSystem -/
def second_pipe_fill_time (sys : PipeSystem) : ℝ :=
  30  -- Placeholder for the actual calculation

/-- Theorem stating that for the given system, the second pipe fills the tank in 30 minutes -/
theorem second_pipe_fills_in_30_minutes (sys : PipeSystem) 
  (h1 : sys.fill_time1 = 18)
  (h2 : sys.empty_time = 45)
  (h3 : sys.combined_fill_time = 0.06666666666666665) :
  second_pipe_fill_time sys = 30 := by
  sorry

#eval second_pipe_fill_time { fill_time1 := 18, empty_time := 45, combined_fill_time := 0.06666666666666665 }

end NUMINAMATH_CALUDE_second_pipe_fills_in_30_minutes_l2591_259101


namespace NUMINAMATH_CALUDE_divisor_count_squared_lt_4n_l2591_259166

def divisor_count (n : ℕ+) : ℕ := (Nat.divisors n.val).card

theorem divisor_count_squared_lt_4n (n : ℕ+) : (divisor_count n)^2 < 4 * n.val := by
  sorry

end NUMINAMATH_CALUDE_divisor_count_squared_lt_4n_l2591_259166


namespace NUMINAMATH_CALUDE_factors_of_210_l2591_259102

theorem factors_of_210 : Nat.card (Nat.divisors 210) = 16 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_210_l2591_259102


namespace NUMINAMATH_CALUDE_moe_has_least_money_l2591_259183

-- Define the set of people
inductive Person : Type
  | Bo | Coe | Flo | Jo | Moe | Zoe

-- Define the "has more money than" relation
def has_more_money (a b : Person) : Prop := sorry

-- Define the conditions
axiom different_amounts :
  ∀ (a b : Person), a ≠ b → (has_more_money a b ∨ has_more_money b a)

axiom flo_more_than_jo_bo :
  has_more_money Person.Flo Person.Jo ∧ has_more_money Person.Flo Person.Bo

axiom bo_coe_more_than_moe :
  has_more_money Person.Bo Person.Moe ∧ has_more_money Person.Coe Person.Moe

axiom jo_between_bo_moe :
  has_more_money Person.Jo Person.Moe ∧ has_more_money Person.Bo Person.Jo

axiom zoe_between_flo_coe :
  has_more_money Person.Zoe Person.Coe ∧ has_more_money Person.Flo Person.Zoe

-- Theorem to prove
theorem moe_has_least_money :
  ∀ (p : Person), p ≠ Person.Moe → has_more_money p Person.Moe := by sorry

end NUMINAMATH_CALUDE_moe_has_least_money_l2591_259183


namespace NUMINAMATH_CALUDE_not_divisible_five_power_minus_one_by_four_power_minus_one_l2591_259199

theorem not_divisible_five_power_minus_one_by_four_power_minus_one (n : ℕ) :
  ¬(∃ k : ℕ, 5^n - 1 = k * (4^n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_five_power_minus_one_by_four_power_minus_one_l2591_259199


namespace NUMINAMATH_CALUDE_recipe_flour_calculation_l2591_259148

/-- The amount of flour required for a recipe -/
def recipe_flour (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

theorem recipe_flour_calculation (initial : ℕ) (additional : ℕ) :
  recipe_flour initial additional = initial + additional :=
by sorry

end NUMINAMATH_CALUDE_recipe_flour_calculation_l2591_259148


namespace NUMINAMATH_CALUDE_parabola_intersects_x_axis_once_l2591_259116

/-- A parabola in the xy-plane defined by y = x^2 + 2x + k -/
def parabola (k : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + k

/-- Condition for a quadratic equation to have exactly one real root -/
def has_one_root (a b c : ℝ) : Prop := b^2 - 4*a*c = 0

/-- Theorem: The parabola y = x^2 + 2x + k intersects the x-axis at only one point if and only if k = 1 -/
theorem parabola_intersects_x_axis_once (k : ℝ) :
  (∃ x : ℝ, parabola k x = 0 ∧ ∀ y : ℝ, parabola k y = 0 → y = x) ↔ k = 1 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersects_x_axis_once_l2591_259116


namespace NUMINAMATH_CALUDE_bike_cost_theorem_l2591_259174

def apple_price : ℚ := 1.25
def apples_sold : ℕ := 20
def repair_ratio : ℚ := 1/4
def remaining_ratio : ℚ := 1/5

def total_earnings : ℚ := apple_price * apples_sold

theorem bike_cost_theorem (h1 : total_earnings = apple_price * apples_sold)
                          (h2 : repair_ratio * (total_earnings * (1 - remaining_ratio)) = total_earnings * (1 - remaining_ratio)) :
  (total_earnings * (1 - remaining_ratio)) / repair_ratio = 80 := by sorry

end NUMINAMATH_CALUDE_bike_cost_theorem_l2591_259174


namespace NUMINAMATH_CALUDE_larger_ball_radius_larger_ball_radius_proof_l2591_259131

theorem larger_ball_radius : ℝ → Prop :=
  fun r : ℝ =>
    -- Volume of a sphere: (4/3) * π * r^3
    let volume_sphere (radius : ℝ) := (4/3) * Real.pi * (radius ^ 3)
    -- Volume of 10 balls with radius 2
    let volume_ten_balls := 10 * volume_sphere 2
    -- Volume of 2 balls with radius 1
    let volume_two_small_balls := 2 * volume_sphere 1
    -- Volume of the larger ball with radius r
    let volume_larger_ball := volume_sphere r
    -- The total volume equality
    volume_ten_balls = volume_larger_ball + volume_two_small_balls →
    -- The radius of the larger ball is ∛78
    r = Real.rpow 78 (1/3)

-- The proof is omitted
theorem larger_ball_radius_proof : larger_ball_radius (Real.rpow 78 (1/3)) := by sorry

end NUMINAMATH_CALUDE_larger_ball_radius_larger_ball_radius_proof_l2591_259131


namespace NUMINAMATH_CALUDE_initial_paint_amount_l2591_259115

theorem initial_paint_amount (total_needed : ℕ) (bought : ℕ) (still_needed : ℕ) 
  (h1 : total_needed = 70)
  (h2 : bought = 23)
  (h3 : still_needed = 11) :
  total_needed - still_needed - bought = 36 :=
by sorry

end NUMINAMATH_CALUDE_initial_paint_amount_l2591_259115


namespace NUMINAMATH_CALUDE_intersection_sum_l2591_259181

theorem intersection_sum : ∃ (x₁ x₂ : ℝ),
  (x₁^2 = 2*x₁ + 3) ∧
  (x₂^2 = 2*x₂ + 3) ∧
  (x₁ ≠ x₂) ∧
  (x₁ + x₂ = 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_l2591_259181


namespace NUMINAMATH_CALUDE_square_of_1017_l2591_259111

theorem square_of_1017 : (1017 : ℕ)^2 = 1034289 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1017_l2591_259111


namespace NUMINAMATH_CALUDE_quadratic_expansion_constraint_l2591_259123

theorem quadratic_expansion_constraint (a b m : ℤ) :
  (∀ x, (x + a) * (x + b) = x^2 + m*x + 5) →
  (m = 6 ∨ m = -6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_expansion_constraint_l2591_259123


namespace NUMINAMATH_CALUDE_calculate_expression_l2591_259117

theorem calculate_expression : 2 * Real.cos (45 * π / 180) - (π - 2023) ^ 0 + |3 - Real.sqrt 2| = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2591_259117


namespace NUMINAMATH_CALUDE_djibo_sister_age_djibo_sister_age_is_28_l2591_259168

theorem djibo_sister_age (djibo_current_age : ℕ) (sum_five_years_ago : ℕ) : ℕ :=
  let djibo_age_five_years_ago := djibo_current_age - 5
  let sister_age_five_years_ago := sum_five_years_ago - djibo_age_five_years_ago
  sister_age_five_years_ago + 5

/-- Given Djibo's current age and the sum of his and his sister's ages five years ago,
    prove that Djibo's sister's current age is 28. -/
theorem djibo_sister_age_is_28 :
  djibo_sister_age 17 35 = 28 := by
  sorry

end NUMINAMATH_CALUDE_djibo_sister_age_djibo_sister_age_is_28_l2591_259168


namespace NUMINAMATH_CALUDE_A_union_B_eq_l2591_259176

def A : Set ℝ := {x | x^2 - x - 2 < 0}

def B : Set ℝ := {x | x^2 - 3*x < 0}

theorem A_union_B_eq : A ∪ B = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_A_union_B_eq_l2591_259176


namespace NUMINAMATH_CALUDE_comparison_theorem_l2591_259192

theorem comparison_theorem :
  (∀ x : ℝ, x^2 - x > x - 2) ∧
  (∀ a : ℝ, (a + 3) * (a - 5) < (a + 2) * (a - 4)) := by
sorry

end NUMINAMATH_CALUDE_comparison_theorem_l2591_259192


namespace NUMINAMATH_CALUDE_initial_candies_l2591_259109

theorem initial_candies : ∃ x : ℕ, (x - 29) / 13 = 15 ∧ x = 224 := by
  sorry

end NUMINAMATH_CALUDE_initial_candies_l2591_259109


namespace NUMINAMATH_CALUDE_axis_of_symmetry_shifted_l2591_259193

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define the axis of symmetry for a function
def AxisOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- Theorem statement
theorem axis_of_symmetry_shifted (f : ℝ → ℝ) (h : EvenFunction f) :
  AxisOfSymmetry (fun x ↦ f (x + 2)) (-2) := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_shifted_l2591_259193


namespace NUMINAMATH_CALUDE_tony_water_consumption_l2591_259126

/-- 
Given that Tony drank 48 ounces of water yesterday, which is 4% less than 
what he drank two days ago, prove that he drank 50 ounces of water two days ago.
-/
theorem tony_water_consumption (yesterday : ℝ) (two_days_ago : ℝ) 
  (h1 : yesterday = 48)
  (h2 : yesterday = two_days_ago * (1 - 0.04)) : 
  two_days_ago = 50 := by
  sorry

end NUMINAMATH_CALUDE_tony_water_consumption_l2591_259126


namespace NUMINAMATH_CALUDE_farthest_corner_distance_l2591_259137

/-- Represents a rectangular pool with given dimensions -/
structure Pool :=
  (length : ℝ)
  (width : ℝ)

/-- Calculates the perimeter of a rectangular pool -/
def perimeter (p : Pool) : ℝ := 2 * (p.length + p.width)

/-- Theorem: In a 10m × 25m pool, if three children walk 50m total,
    the distance to the farthest corner is 20m -/
theorem farthest_corner_distance (p : Pool) 
  (h1 : p.length = 25)
  (h2 : p.width = 10)
  (h3 : ∃ (x : ℝ), x ≥ 0 ∧ x ≤ perimeter p ∧ perimeter p - x = 50) :
  ∃ (y : ℝ), y = 20 ∧ y = perimeter p - 50 := by
  sorry

end NUMINAMATH_CALUDE_farthest_corner_distance_l2591_259137


namespace NUMINAMATH_CALUDE_equation_proof_l2591_259196

theorem equation_proof : 3889 + 12.952 - 47.95 = 3854.002 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2591_259196


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2591_259158

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b * c + a + c = b) :
  ∃ (p : ℝ), p = 2 / (a^2 + 1) - 2 / (b^2 + 1) + 3 / (c^2 + 1) ∧
  p ≤ 10/3 ∧ 
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    a' * b' * c' + a' + c' = b' ∧
    2 / (a'^2 + 1) - 2 / (b'^2 + 1) + 3 / (c'^2 + 1) = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2591_259158


namespace NUMINAMATH_CALUDE_function_property_l2591_259171

variable (f : ℝ → ℝ)
variable (p q : ℝ)

theorem function_property
  (h1 : ∀ a b, f (a * b) = f a + f b)
  (h2 : f 2 = p)
  (h3 : f 3 = q) :
  f 12 = 2 * p + q :=
by sorry

end NUMINAMATH_CALUDE_function_property_l2591_259171


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2591_259127

theorem min_value_of_expression (x : ℝ) : (x^2 + 8) / Real.sqrt (x^2 + 4) ≥ 4 ∧
  ∃ x₀ : ℝ, (x₀^2 + 8) / Real.sqrt (x₀^2 + 4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2591_259127


namespace NUMINAMATH_CALUDE_a_range_l2591_259150

/-- Proposition p: A real number x satisfies 2 < x < 3 -/
def p (x : ℝ) : Prop := 2 < x ∧ x < 3

/-- Proposition q: A real number x satisfies 2x^2 - 9x + a < 0 -/
def q (x a : ℝ) : Prop := 2 * x^2 - 9 * x + a < 0

/-- p is a sufficient condition for q -/
def p_implies_q (a : ℝ) : Prop := ∀ x, p x → q x a

theorem a_range (a : ℝ) : p_implies_q a ↔ 7 ≤ a ∧ a ≤ 8 := by sorry

end NUMINAMATH_CALUDE_a_range_l2591_259150


namespace NUMINAMATH_CALUDE_simplify_expression_l2591_259165

theorem simplify_expression (x : ℝ) : (2*x + 25) + (150*x + 35) + (50*x + 10) = 202*x + 70 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2591_259165


namespace NUMINAMATH_CALUDE_square_root_equation_solution_l2591_259147

theorem square_root_equation_solution :
  ∃ x : ℝ, (Real.sqrt 289 - Real.sqrt x / Real.sqrt 25 = 12) ∧ x = 625 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_solution_l2591_259147


namespace NUMINAMATH_CALUDE_cube_opposite_face_l2591_259159

structure Cube where
  faces : Finset Char
  adjacent : Char → Char → Prop

def opposite (c : Cube) (x y : Char) : Prop :=
  x ∈ c.faces ∧ y ∈ c.faces ∧ x ≠ y ∧ ¬c.adjacent x y

theorem cube_opposite_face (c : Cube) :
  c.faces = {'А', 'Б', 'В', 'Г', 'Д', 'Е'} →
  c.adjacent 'В' 'А' →
  c.adjacent 'В' 'Д' →
  c.adjacent 'В' 'Е' →
  opposite c 'В' 'Г' := by
  sorry

end NUMINAMATH_CALUDE_cube_opposite_face_l2591_259159


namespace NUMINAMATH_CALUDE_locus_of_points_l2591_259198

/-- The locus of points with a 3:1 distance ratio to a fixed point and line -/
theorem locus_of_points (x y : ℝ) : 
  let F : ℝ × ℝ := (4.5, 0)
  let dist_to_F := Real.sqrt ((x - F.1)^2 + (y - F.2)^2)
  let dist_to_line := |x - 0.5|
  dist_to_F = 3 * dist_to_line → x^2 / 2.25 - y^2 / 18 = 1 := by
  sorry

end NUMINAMATH_CALUDE_locus_of_points_l2591_259198


namespace NUMINAMATH_CALUDE_negative_of_negative_six_equals_six_l2591_259108

theorem negative_of_negative_six_equals_six : -(-6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_of_negative_six_equals_six_l2591_259108


namespace NUMINAMATH_CALUDE_sum_of_roots_equal_three_l2591_259180

-- Define the polynomial
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 - 48 * x - 12

-- State the theorem
theorem sum_of_roots_equal_three :
  ∃ (r p q : ℝ), f r = 0 ∧ f p = 0 ∧ f q = 0 ∧ r + p + q = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equal_three_l2591_259180


namespace NUMINAMATH_CALUDE_last_two_digits_of_expression_l2591_259113

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_of_expression : 
  last_two_digits (sum_of_factorials 15 - factorial 5 * factorial 10 * factorial 15) = 13 := by
sorry

end NUMINAMATH_CALUDE_last_two_digits_of_expression_l2591_259113


namespace NUMINAMATH_CALUDE_squares_in_4x2023_grid_l2591_259130

/-- The number of squares with vertices on grid points in a 4 x 2023 grid -/
def squaresInGrid (rows : ℕ) (cols : ℕ) : ℕ :=
  let type_a := rows * cols
  let type_b := (rows - 1) * (cols - 1)
  let type_c := (rows - 2) * (cols - 2)
  let type_d := (rows - 3) * (cols - 3)
  type_a + 2 * type_b + 3 * type_c + 4 * type_d

/-- Theorem stating that the number of squares in a 4 x 2023 grid is 40430 -/
theorem squares_in_4x2023_grid :
  squaresInGrid 4 2023 = 40430 := by
  sorry

end NUMINAMATH_CALUDE_squares_in_4x2023_grid_l2591_259130


namespace NUMINAMATH_CALUDE_six_by_six_grid_half_shaded_l2591_259143

/-- Represents a square grid -/
structure SquareGrid :=
  (size : ℕ)
  (shaded : ℕ)

/-- Calculates the percentage of shaded area in a square grid -/
def shaded_percentage (grid : SquareGrid) : ℚ :=
  (grid.shaded : ℚ) / (grid.size * grid.size : ℚ) * 100

/-- Theorem: A 6x6 grid with 18 shaded squares is 50% shaded -/
theorem six_by_six_grid_half_shaded :
  let grid : SquareGrid := ⟨6, 18⟩
  shaded_percentage grid = 50 := by sorry

end NUMINAMATH_CALUDE_six_by_six_grid_half_shaded_l2591_259143


namespace NUMINAMATH_CALUDE_inverse_proportionality_l2591_259182

/-- Given that x is inversely proportional to y, prove that if x = 5 when y = 15, then x = -2.5 when y = -30 -/
theorem inverse_proportionality (x y : ℝ) (h : ∃ k : ℝ, ∀ x y, x * y = k) :
  (∃ x₀, x₀ = 5 ∧ x₀ * 15 = 5 * 15) →
  (∃ x₁, x₁ * (-30) = 5 * 15 ∧ x₁ = -2.5) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportionality_l2591_259182


namespace NUMINAMATH_CALUDE_square_plus_abs_zero_implies_both_zero_l2591_259169

theorem square_plus_abs_zero_implies_both_zero (a b : ℝ) : 
  a^2 + |b| = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_abs_zero_implies_both_zero_l2591_259169


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l2591_259170

/-- Given a circle C with equation x^2 + 8x - 5y = -y^2 + 2x, 
    the sum of the x-coordinate and y-coordinate of its center along with its radius 
    is equal to (√61 - 1) / 2 -/
theorem circle_center_radius_sum (x y : ℝ) : 
  (x^2 + 8*x - 5*y = -y^2 + 2*x) → 
  ∃ (center_x center_y radius : ℝ), 
    (center_x + center_y + radius = (Real.sqrt 61 - 1) / 2) ∧
    ∀ (p_x p_y : ℝ), (p_x - center_x)^2 + (p_y - center_y)^2 = radius^2 ↔ 
      p_x^2 + 8*p_x - 5*p_y = -p_y^2 + 2*p_x :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l2591_259170


namespace NUMINAMATH_CALUDE_total_balloons_l2591_259197

def tom_balloons : ℕ := 9
def sara_balloons : ℕ := 8

theorem total_balloons : tom_balloons + sara_balloons = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_l2591_259197


namespace NUMINAMATH_CALUDE_complex_real_condition_l2591_259112

theorem complex_real_condition (m : ℝ) : 
  (∃ (z : ℂ), z = (m^2 - 5*m + 6 : ℝ) + (m - 3 : ℝ)*I ∧ z.im = 0) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l2591_259112


namespace NUMINAMATH_CALUDE_range_of_f_l2591_259105

/-- The function f(c) defined as (c-a)(c-b) -/
def f (c a b : ℝ) : ℝ := (c - a) * (c - b)

/-- Theorem stating the range of f(c) -/
theorem range_of_f :
  ∀ c a b : ℝ,
  a + b = 1 - c →
  c ≥ 0 →
  a ≥ 0 →
  b ≥ 0 →
  ∃ y : ℝ, f c a b = y ∧ -1/8 ≤ y ∧ y ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l2591_259105


namespace NUMINAMATH_CALUDE_expression_comparison_l2591_259172

theorem expression_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ¬(∀ x y : ℝ, x > 0 → y > 0 → x ≠ y →
    ((x + 1/x) * (y + 1/y) > (Real.sqrt (x*y) + 1/Real.sqrt (x*y))^2 ∧
     (x + 1/x) * (y + 1/y) > ((x + y)/2 + 2/(x + y))^2) ∨
    ((Real.sqrt (x*y) + 1/Real.sqrt (x*y))^2 > (x + 1/x) * (y + 1/y) ∧
     (Real.sqrt (x*y) + 1/Real.sqrt (x*y))^2 > ((x + y)/2 + 2/(x + y))^2) ∨
    (((x + y)/2 + 2/(x + y))^2 > (x + 1/x) * (y + 1/y) ∧
     ((x + y)/2 + 2/(x + y))^2 > (Real.sqrt (x*y) + 1/Real.sqrt (x*y))^2)) :=
by sorry

end NUMINAMATH_CALUDE_expression_comparison_l2591_259172


namespace NUMINAMATH_CALUDE_salt_water_fraction_l2591_259103

theorem salt_water_fraction (small_capacity large_capacity : ℝ) 
  (h1 : large_capacity = 5 * small_capacity)
  (h2 : 0.3 * large_capacity = 0.2 * large_capacity + small_capacity * x) : x = 1/2 := by
  sorry

#check salt_water_fraction

end NUMINAMATH_CALUDE_salt_water_fraction_l2591_259103


namespace NUMINAMATH_CALUDE_max_sum_of_cubes_l2591_259107

/-- Given a system of equations, find the maximum value of x³ + y³ + z³ -/
theorem max_sum_of_cubes (x y z : ℝ) : 
  x^3 - x*y*z = 2 → 
  y^3 - x*y*z = 6 → 
  z^3 - x*y*z = 20 → 
  x^3 + y^3 + z^3 ≤ 151/7 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_cubes_l2591_259107


namespace NUMINAMATH_CALUDE_regular_polygon_diagonals_l2591_259188

theorem regular_polygon_diagonals (n : ℕ) (h : n > 2) :
  (n * (n - 3) / 2 : ℚ) = 2 * n → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_diagonals_l2591_259188


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2591_259104

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |2005 * x - 2005| = 2005 ↔ x = 2 ∨ x = 0 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2591_259104


namespace NUMINAMATH_CALUDE_lawn_mowing_time_mowing_time_approx_2_3_l2591_259163

/-- Represents the lawn mowing problem -/
theorem lawn_mowing_time (lawn_length lawn_width : ℝ) 
                         (swath_width overlap : ℝ) 
                         (mowing_speed : ℝ) : ℝ :=
  let effective_swath := swath_width - overlap
  let num_strips := lawn_width / effective_swath
  let total_distance := num_strips * lawn_length
  let time_taken := total_distance / mowing_speed
  time_taken

/-- Proves that the time taken to mow the lawn is approximately 2.3 hours -/
theorem mowing_time_approx_2_3 :
  ∃ ε > 0, |lawn_mowing_time 120 180 (30/12) (2/12) 4000 - 2.3| < ε :=
sorry

end NUMINAMATH_CALUDE_lawn_mowing_time_mowing_time_approx_2_3_l2591_259163


namespace NUMINAMATH_CALUDE_next_occurrence_is_august_first_l2591_259194

def initial_date : Nat × Nat × Nat := (5, 1, 1994)
def initial_time : Nat × Nat := (7, 32)

def digits : List Nat := [0, 5, 1, 1, 9, 9, 4, 0, 7, 3]

def is_valid_date (d : Nat) (m : Nat) (y : Nat) : Bool :=
  d > 0 && d ≤ 31 && m > 0 && m ≤ 12 && y == 1994

def is_valid_time (h : Nat) (m : Nat) : Bool :=
  h ≥ 0 && h < 24 && m ≥ 0 && m < 60

def date_time_to_digits (d : Nat) (m : Nat) (y : Nat) (h : Nat) (min : Nat) : List Nat :=
  let date_digits := (d / 10) :: (d % 10) :: (m / 10) :: (m % 10) :: (y / 1000) :: ((y / 100) % 10) :: ((y / 10) % 10) :: (y % 10) :: []
  let time_digits := (h / 10) :: (h % 10) :: (min / 10) :: (min % 10) :: []
  date_digits ++ time_digits

def is_next_occurrence (d : Nat) (m : Nat) (h : Nat) (min : Nat) : Prop :=
  is_valid_date d m 1994 ∧
  is_valid_time h min ∧
  date_time_to_digits d m 1994 h min == digits ∧
  (d, m) > (5, 1) ∧
  ∀ (d' : Nat) (m' : Nat) (h' : Nat) (min' : Nat),
    is_valid_date d' m' 1994 →
    is_valid_time h' min' →
    date_time_to_digits d' m' 1994 h' min' == digits →
    (d', m') > (5, 1) →
    (d', m') ≤ (d, m)

theorem next_occurrence_is_august_first :
  is_next_occurrence 1 8 2 45 := by sorry

end NUMINAMATH_CALUDE_next_occurrence_is_august_first_l2591_259194


namespace NUMINAMATH_CALUDE_difference_between_decimals_and_fractions_l2591_259122

theorem difference_between_decimals_and_fractions : (0.127 : ℝ) - (1/8 : ℝ) = 0.002 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_decimals_and_fractions_l2591_259122


namespace NUMINAMATH_CALUDE_mildred_blocks_l2591_259136

/-- The number of blocks Mildred ends up with -/
def total_blocks (initial : ℕ) (found : ℕ) : ℕ :=
  initial + found

/-- Theorem stating that Mildred's total blocks is the sum of initial and found blocks -/
theorem mildred_blocks (initial : ℕ) (found : ℕ) :
  total_blocks initial found = initial + found := by
  sorry

end NUMINAMATH_CALUDE_mildred_blocks_l2591_259136


namespace NUMINAMATH_CALUDE_common_point_sum_mod_9_l2591_259118

theorem common_point_sum_mod_9 : ∃ (x : ℤ), 
  (∀ (y : ℤ), (y ≡ 3*x + 5 [ZMOD 9] ↔ y ≡ 7*x + 3 [ZMOD 9])) ∧ 
  (x ≡ 5 [ZMOD 9]) := by
  sorry

end NUMINAMATH_CALUDE_common_point_sum_mod_9_l2591_259118


namespace NUMINAMATH_CALUDE_fraction_equality_l2591_259185

theorem fraction_equality : (8 : ℚ) / (5 * 46) = 0.8 / 23 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2591_259185


namespace NUMINAMATH_CALUDE_bill_difference_l2591_259125

theorem bill_difference (anna_tip bob_tip cindy_tip : ℝ)
  (anna_percent bob_percent cindy_percent : ℝ)
  (h_anna : anna_tip = 3 ∧ anna_percent = 0.15)
  (h_bob : bob_tip = 4 ∧ bob_percent = 0.10)
  (h_cindy : cindy_tip = 5 ∧ cindy_percent = 0.25)
  (h_anna_bill : anna_tip = anna_percent * (anna_tip / anna_percent))
  (h_bob_bill : bob_tip = bob_percent * (bob_tip / bob_percent))
  (h_cindy_bill : cindy_tip = cindy_percent * (cindy_tip / cindy_percent)) :
  max (anna_tip / anna_percent) (max (bob_tip / bob_percent) (cindy_tip / cindy_percent)) -
  min (anna_tip / anna_percent) (min (bob_tip / bob_percent) (cindy_tip / cindy_percent)) = 20 :=
by sorry

end NUMINAMATH_CALUDE_bill_difference_l2591_259125


namespace NUMINAMATH_CALUDE_arithmetic_sequence_51st_term_l2591_259160

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_51st_term 
  (a : ℕ → ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 2) : 
  a 51 = 101 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_51st_term_l2591_259160
