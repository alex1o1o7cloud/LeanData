import Mathlib

namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2054_205420

theorem fractional_equation_solution :
  ∃ x : ℚ, (x + 1) / (4 * (x - 1)) = 2 / (3 * x - 3) - 1 ↔ x = 17 / 15 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2054_205420


namespace NUMINAMATH_CALUDE_class_size_proof_l2054_205409

theorem class_size_proof (total_average : ℝ) (excluded_average : ℝ) (remaining_average : ℝ) 
  (excluded_count : ℕ) (h1 : total_average = 80) (h2 : excluded_average = 60) 
  (h3 : remaining_average = 90) (h4 : excluded_count = 5) : 
  ∃ (n : ℕ), n = 15 ∧ 
  (n : ℝ) * total_average = 
    ((n : ℝ) - excluded_count) * remaining_average + (excluded_count : ℝ) * excluded_average :=
by
  sorry

#check class_size_proof

end NUMINAMATH_CALUDE_class_size_proof_l2054_205409


namespace NUMINAMATH_CALUDE_cymbal_triangle_sync_l2054_205401

theorem cymbal_triangle_sync (cymbal_beats triangle_beats : ℕ) 
  (h1 : cymbal_beats = 7) (h2 : triangle_beats = 2) : 
  Nat.lcm cymbal_beats triangle_beats = 14 := by
  sorry

end NUMINAMATH_CALUDE_cymbal_triangle_sync_l2054_205401


namespace NUMINAMATH_CALUDE_min_coefficient_value_l2054_205454

theorem min_coefficient_value (a b Box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 15 * x^2 + Box * x + 15) →
  a ≠ b ∧ b ≠ Box ∧ a ≠ Box →
  Box ≥ 34 ∧ 
  (Box = 34 ↔ ((a = 3 ∧ b = 5) ∨ (a = -3 ∧ b = -5) ∨ (a = 5 ∧ b = 3) ∨ (a = -5 ∧ b = -3))) :=
by sorry

end NUMINAMATH_CALUDE_min_coefficient_value_l2054_205454


namespace NUMINAMATH_CALUDE_number_equation_proof_l2054_205445

theorem number_equation_proof : ∃ n : ℝ, n + 11.95 - 596.95 = 3054 ∧ n = 3639 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_proof_l2054_205445


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2054_205431

def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 4}

theorem intersection_of_A_and_B : A ∩ B = {3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2054_205431


namespace NUMINAMATH_CALUDE_sum_ge_sum_of_sqrt_products_l2054_205448

theorem sum_ge_sum_of_sqrt_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) := by
  sorry

end NUMINAMATH_CALUDE_sum_ge_sum_of_sqrt_products_l2054_205448


namespace NUMINAMATH_CALUDE_division_remainder_l2054_205484

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 686 →
  divisor = 36 →
  quotient = 19 →
  dividend = divisor * quotient + remainder →
  remainder = 2 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l2054_205484


namespace NUMINAMATH_CALUDE_max_segments_proof_l2054_205472

/-- Given n consecutive points on a line with total length 1, 
    this function returns the maximum number of segments with length ≥ a,
    where 0 ≤ a ≤ 1/(n-1) -/
def max_segments (n : ℕ) (a : ℝ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem stating that for n consecutive points on a line with total length 1,
    and 0 ≤ a ≤ 1/(n-1), the maximum number of segments with length ≥ a is n(n-1)/2 -/
theorem max_segments_proof (n : ℕ) (a : ℝ) 
    (h1 : n > 1) 
    (h2 : 0 ≤ a) 
    (h3 : a ≤ 1 / (n - 1)) : 
  max_segments n a = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_segments_proof_l2054_205472


namespace NUMINAMATH_CALUDE_max_teams_advancing_l2054_205499

/-- Represents a football tournament with the given conditions -/
structure FootballTournament where
  num_teams : Nat
  min_points_to_advance : Nat
  points_for_win : Nat
  points_for_draw : Nat
  points_for_loss : Nat

/-- Calculate the total number of games in the tournament -/
def total_games (t : FootballTournament) : Nat :=
  t.num_teams * (t.num_teams - 1) / 2

/-- Calculate the maximum total points that can be distributed in the tournament -/
def max_total_points (t : FootballTournament) : Nat :=
  (total_games t) * t.points_for_win

/-- Theorem stating the maximum number of teams that can advance -/
theorem max_teams_advancing (t : FootballTournament) 
  (h1 : t.num_teams = 6)
  (h2 : t.min_points_to_advance = 12)
  (h3 : t.points_for_win = 3)
  (h4 : t.points_for_draw = 1)
  (h5 : t.points_for_loss = 0) :
  ∃ (n : Nat), n ≤ 3 ∧ 
    n * t.min_points_to_advance ≤ max_total_points t ∧
    ∀ (m : Nat), m * t.min_points_to_advance ≤ max_total_points t → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_max_teams_advancing_l2054_205499


namespace NUMINAMATH_CALUDE_min_disks_needed_l2054_205408

/-- Represents the capacity of each disk in MB -/
def diskCapacity : ℚ := 1.44

/-- Represents the file sizes in MB -/
def fileSizes : List ℚ := [0.9, 0.6, 0.45, 0.3]

/-- Represents the quantity of each file size -/
def fileQuantities : List ℕ := [5, 10, 10, 5]

/-- Calculates the total storage required for all files in MB -/
def totalStorage : ℚ :=
  List.sum (List.zipWith (· * ·) (List.map (λ x => (x : ℚ)) fileQuantities) fileSizes)

/-- Theorem: The minimum number of disks needed is 15 -/
theorem min_disks_needed : 
  ∃ (n : ℕ), n = 15 ∧ 
  n * diskCapacity ≥ totalStorage ∧
  ∀ m : ℕ, m * diskCapacity ≥ totalStorage → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_min_disks_needed_l2054_205408


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_two_l2054_205475

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = (x+a)(x-2) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * (x - 2)

/-- If f(x) = (x+a)(x-2) is an even function, then a = 2 -/
theorem even_function_implies_a_equals_two :
  ∀ a : ℝ, IsEven (f a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_two_l2054_205475


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2054_205465

theorem polynomial_divisibility (a b c d m : ℤ) 
  (h1 : (5 : ℤ) ∣ (a * m^3 + b * m^2 + c * m + d))
  (h2 : ¬((5 : ℤ) ∣ d)) :
  ∃ n : ℤ, (5 : ℤ) ∣ (d * n^3 + c * n^2 + b * n + a) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2054_205465


namespace NUMINAMATH_CALUDE_framed_photo_border_area_l2054_205421

/-- The area of the border of a framed rectangular photograph. -/
theorem framed_photo_border_area 
  (photo_height : ℝ) 
  (photo_width : ℝ) 
  (border_width : ℝ) 
  (h1 : photo_height = 12) 
  (h2 : photo_width = 15) 
  (h3 : border_width = 3) : 
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - 
  photo_height * photo_width = 198 := by
  sorry

end NUMINAMATH_CALUDE_framed_photo_border_area_l2054_205421


namespace NUMINAMATH_CALUDE_B_equals_zero_one_two_l2054_205405

def B : Set ℤ := {x | -3 < 2*x - 1 ∧ 2*x - 1 < 5}

theorem B_equals_zero_one_two : B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_B_equals_zero_one_two_l2054_205405


namespace NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l2054_205413

theorem largest_x_sqrt_3x_eq_5x : 
  (∃ (x : ℝ), x > 0 ∧ Real.sqrt (3 * x) = 5 * x) →
  (∀ (y : ℝ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y → y ≤ 3/25) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l2054_205413


namespace NUMINAMATH_CALUDE_max_value_of_f_l2054_205464

-- Define the function
def f (x : ℝ) : ℝ := x * (3 - 2 * x)

-- Define the domain
def domain (x : ℝ) : Prop := 0 < x ∧ x ≤ 1

-- State the theorem
theorem max_value_of_f :
  ∃ (max : ℝ), max = 9/8 ∧ ∀ (x : ℝ), domain x → f x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2054_205464


namespace NUMINAMATH_CALUDE_cinnamon_swirl_sharing_l2054_205498

theorem cinnamon_swirl_sharing (total_pieces : ℕ) (jane_pieces : ℕ) (h1 : total_pieces = 12) (h2 : jane_pieces = 4) :
  total_pieces / jane_pieces = 3 :=
by
  sorry

#check cinnamon_swirl_sharing

end NUMINAMATH_CALUDE_cinnamon_swirl_sharing_l2054_205498


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l2054_205473

theorem minimum_value_theorem (a b m n : ℝ) : 
  (∀ x, (x + 2) / (x + 1) < 0 ↔ a < x ∧ x < b) →
  m * a + n * b + 1 = 0 →
  m * n > 0 →
  (∀ m' n', m' * n' > 0 → m' * a + n' * b + 1 = 0 → 2 / m' + 1 / n' ≥ 2 / m + 1 / n) →
  2 / m + 1 / n = 9 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l2054_205473


namespace NUMINAMATH_CALUDE_point_b_coordinates_l2054_205486

/-- Given a circle and two points A and B, if the squared distance from any point on the circle to A
    is twice the squared distance to B, then B has coordinates (1, 1). -/
theorem point_b_coordinates (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4 → (x - 2)^2 + (y - 2)^2 = 2*((x - a)^2 + (y - b)^2)) →
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_point_b_coordinates_l2054_205486


namespace NUMINAMATH_CALUDE_cold_drink_pitcher_l2054_205492

/-- Represents the recipe for a cold drink -/
structure Recipe where
  iced_tea : Rat
  lemonade : Rat

/-- Calculates the total amount of drink for a given recipe -/
def total_drink (r : Recipe) : Rat :=
  r.iced_tea + r.lemonade

/-- Represents the contents of a pitcher -/
structure Pitcher where
  lemonade : Rat
  total : Rat

/-- The theorem to be proved -/
theorem cold_drink_pitcher (r : Recipe) (p : Pitcher) :
  r.iced_tea = 1/4 →
  r.lemonade = 5/4 →
  p.lemonade = 15 →
  p.total = 18 :=
by sorry

end NUMINAMATH_CALUDE_cold_drink_pitcher_l2054_205492


namespace NUMINAMATH_CALUDE_price_reduction_options_best_discount_percentage_impossibility_of_higher_profit_l2054_205480

-- Define constants
def cost_price : ℝ := 240
def original_price : ℝ := 400
def initial_sales : ℝ := 200
def sales_increase_rate : ℝ := 4
def target_profit : ℝ := 41600
def impossible_profit : ℝ := 50000

-- Define function for weekly profit based on price reduction
def weekly_profit (price_reduction : ℝ) : ℝ :=
  (original_price - price_reduction - cost_price) * (initial_sales + sales_increase_rate * price_reduction)

-- Theorem 1: Price reduction options
theorem price_reduction_options :
  ∃ (x y : ℝ), x ≠ y ∧ weekly_profit x = target_profit ∧ weekly_profit y = target_profit :=
sorry

-- Theorem 2: Best discount percentage
theorem best_discount_percentage :
  ∃ (best_reduction : ℝ), weekly_profit best_reduction = target_profit ∧
    ∀ (other_reduction : ℝ), weekly_profit other_reduction = target_profit →
      best_reduction ≥ other_reduction ∧
      (original_price - best_reduction) / original_price = 0.8 :=
sorry

-- Theorem 3: Impossibility of higher profit
theorem impossibility_of_higher_profit :
  ∀ (price_reduction : ℝ), weekly_profit price_reduction ≠ impossible_profit :=
sorry

end NUMINAMATH_CALUDE_price_reduction_options_best_discount_percentage_impossibility_of_higher_profit_l2054_205480


namespace NUMINAMATH_CALUDE_winter_migration_l2054_205438

/-- The number of bird families living near the mountain -/
def mountain_families : ℕ := 18

/-- The number of bird families that flew to Africa -/
def africa_families : ℕ := 38

/-- The number of bird families that flew to Asia -/
def asia_families : ℕ := 80

/-- The total number of bird families that flew away for the winter -/
def total_migrated_families : ℕ := africa_families + asia_families

theorem winter_migration :
  total_migrated_families = 118 :=
by sorry

end NUMINAMATH_CALUDE_winter_migration_l2054_205438


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_and_formula_l2054_205424

/-- Geometric sequence a_n with sum S_n -/
def geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))

theorem geometric_sequence_sum_and_formula 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) :
  geometric_sequence a S →
  S 3 = 21 →
  S 6 = 189 →
  (∀ n : ℕ, b n = (-1)^n * a n) →
  (∀ n : ℕ, n > 0 → a n = 3 * 2^(n-1)) ∧
  (∀ n : ℕ, n > 0 → T n = -1 + (-2)^n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_and_formula_l2054_205424


namespace NUMINAMATH_CALUDE_number_equation_l2054_205442

theorem number_equation (x : ℝ) : 3 * x - 6 = 2 * x ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l2054_205442


namespace NUMINAMATH_CALUDE_expression_simplification_l2054_205436

theorem expression_simplification (a c x y : ℝ) (h : c*x^2 + c*y^2 ≠ 0) :
  (c*x^2*(a^2*x^3 + 3*a^2*y^3 + c^2*y^3) + c*y^2*(a^2*x^3 + 3*c^2*x^3 + c^2*y^3)) / (c*x^2 + c*y^2)
  = a^2*x^3 + 3*c*x^3 + c^2*y^3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2054_205436


namespace NUMINAMATH_CALUDE_ray_nickels_left_l2054_205435

def nickel_value : ℕ := 5
def initial_cents : ℕ := 95
def cents_to_peter : ℕ := 25

theorem ray_nickels_left : 
  let initial_nickels := initial_cents / nickel_value
  let nickels_to_peter := cents_to_peter / nickel_value
  let cents_to_randi := 2 * cents_to_peter
  let nickels_to_randi := cents_to_randi / nickel_value
  initial_nickels - nickels_to_peter - nickels_to_randi = 4 := by
  sorry

end NUMINAMATH_CALUDE_ray_nickels_left_l2054_205435


namespace NUMINAMATH_CALUDE_range_of_a_l2054_205416

/-- The function f(x) defined in the problem -/
def f (a x : ℝ) : ℝ := a * x^2 - (3 - a) * x + 1

/-- The function g(x) defined in the problem -/
def g (x : ℝ) : ℝ := x

/-- The theorem stating the range of a -/
theorem range_of_a : 
  {a : ℝ | ∀ x, max (f a x) (g x) > 0} = Set.Icc 0 9 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2054_205416


namespace NUMINAMATH_CALUDE_banana_bread_flour_calculation_hannahs_banana_bread_flour_l2054_205470

/-- Given the ratio of flour to banana mush, bananas per cup of mush, and total bananas used,
    calculate the number of cups of flour needed. -/
theorem banana_bread_flour_calculation 
  (flour_to_mush_ratio : ℚ) 
  (bananas_per_mush : ℕ) 
  (total_bananas : ℕ) : ℚ :=
  by
  sorry

/-- Prove that for Hannah's banana bread recipe, she needs 15 cups of flour. -/
theorem hannahs_banana_bread_flour : 
  banana_bread_flour_calculation 3 4 20 = 15 :=
  by
  sorry

end NUMINAMATH_CALUDE_banana_bread_flour_calculation_hannahs_banana_bread_flour_l2054_205470


namespace NUMINAMATH_CALUDE_sequence_existence_l2054_205468

theorem sequence_existence (n : ℕ) (hn : n ≥ 3) :
  (∃ (a : ℕ → ℝ), 
    (∀ i ∈ Finset.range n, a i * a (i + 1) + 1 = a (i + 2)) ∧
    (a (n + 1) = a 1) ∧
    (a (n + 2) = a 2)) ↔ 
  (3 ∣ n) :=
sorry

end NUMINAMATH_CALUDE_sequence_existence_l2054_205468


namespace NUMINAMATH_CALUDE_james_lifting_ratio_l2054_205412

theorem james_lifting_ratio :
  let initial_total : ℝ := 2200
  let initial_weight : ℝ := 245
  let total_gain_percent : ℝ := 0.15
  let weight_gain : ℝ := 8
  let final_total : ℝ := initial_total * (1 + total_gain_percent)
  let final_weight : ℝ := initial_weight + weight_gain
  final_total / final_weight = 10
  := by sorry

end NUMINAMATH_CALUDE_james_lifting_ratio_l2054_205412


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_l2054_205450

theorem abs_inequality_equivalence (x : ℝ) : 
  |((x + 4) / 2)| < 3 ↔ -10 < x ∧ x < 2 := by
sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_l2054_205450


namespace NUMINAMATH_CALUDE_shanes_remaining_gum_is_eight_l2054_205429

/-- The number of pieces of gum Shane has left after a series of exchanges and consumption --/
def shanes_remaining_gum : ℕ :=
  let elyses_initial_gum : ℕ := 100
  let ricks_gum : ℕ := elyses_initial_gum / 2
  let shanes_initial_gum : ℕ := ricks_gum / 3
  let shanes_gum_after_cousin : ℕ := shanes_initial_gum + 10
  let shanes_gum_after_chewing : ℕ := shanes_gum_after_cousin - 11
  let gum_shared_with_sarah : ℕ := shanes_gum_after_chewing / 2
  shanes_gum_after_chewing - gum_shared_with_sarah

theorem shanes_remaining_gum_is_eight :
  shanes_remaining_gum = 8 := by
  sorry

end NUMINAMATH_CALUDE_shanes_remaining_gum_is_eight_l2054_205429


namespace NUMINAMATH_CALUDE_find_a_l2054_205451

theorem find_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = a * x + 6) →
  f (-1) = 8 →
  a = -2 := by sorry

end NUMINAMATH_CALUDE_find_a_l2054_205451


namespace NUMINAMATH_CALUDE_abs_three_plus_one_l2054_205404

theorem abs_three_plus_one (a : ℝ) : 
  (|a| = 3) → (a + 1 = 4 ∨ a + 1 = -2) := by sorry

end NUMINAMATH_CALUDE_abs_three_plus_one_l2054_205404


namespace NUMINAMATH_CALUDE_rectangular_field_width_l2054_205497

/-- Proves that for a rectangular field with length 7/5 of its width and perimeter 336 meters, the width is 70 meters -/
theorem rectangular_field_width (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = (7/5) * width →
  perimeter = 336 →
  perimeter = 2 * length + 2 * width →
  width = 70 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l2054_205497


namespace NUMINAMATH_CALUDE_special_ellipse_properties_l2054_205493

/-- An ellipse with eccentricity √3/2 and maximum triangle area of 1 -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 3/4
  h_max_area : a * b = 2

/-- A line intersecting the ellipse -/
structure IntersectingLine (E : SpecialEllipse) where
  k : ℝ
  m : ℝ
  h_slope_product : (5 : ℝ) / 4 = k^2 + m^2

/-- The theorem statement -/
theorem special_ellipse_properties (E : SpecialEllipse) (L : IntersectingLine E) :
  (E.a = 2 ∧ E.b = 1) ∧
  (∃ (S : ℝ), S = 1 ∧ ∀ (k m : ℝ), (5 : ℝ) / 4 = k^2 + m^2 → S ≥ 
    ((5 - 4*k^2) * (20*k^2 - 1)) / (2 * (4*k^2 + 1))) :=
  sorry

end NUMINAMATH_CALUDE_special_ellipse_properties_l2054_205493


namespace NUMINAMATH_CALUDE_exam_score_calculation_l2054_205432

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) (total_marks : ℕ) :
  total_questions = 60 →
  correct_answers = 36 →
  total_marks = 120 →
  ∃ (score_per_correct : ℕ),
    score_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧
    score_per_correct = 4 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l2054_205432


namespace NUMINAMATH_CALUDE_valid_solution_l2054_205411

/-- A number is a perfect square if it's the square of an integer. -/
def IsPerfectSquare (x : ℕ) : Prop := ∃ y : ℕ, x = y^2

/-- A function to check if a number is prime. -/
def IsPrime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

/-- The theorem stating that 900 is a valid solution for n. -/
theorem valid_solution :
  ∃ m : ℕ, IsPerfectSquare m ∧ IsPerfectSquare 900 ∧ IsPrime (m - 900) :=
by sorry

end NUMINAMATH_CALUDE_valid_solution_l2054_205411


namespace NUMINAMATH_CALUDE_ab_value_l2054_205495

theorem ab_value (a b : ℕ+) (h1 : a + b = 24) (h2 : 2 * a * b + 10 * a = 3 * b + 222) : a * b = 108 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2054_205495


namespace NUMINAMATH_CALUDE_two_digit_number_proof_l2054_205437

theorem two_digit_number_proof : 
  ∀ (n : ℕ), 
  (n ≥ 10 ∧ n < 100) →  -- n is a two-digit number
  (n % 10 + n / 10 = 9) →  -- sum of digits is 9
  (10 * (n % 10) + n / 10 = n - 9) →  -- swapping digits results in n - 9
  n = 54 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_proof_l2054_205437


namespace NUMINAMATH_CALUDE_john_lawyer_payment_l2054_205488

/-- Calculates John's payment for lawyer fees --/
def johnPayment (upfrontFee courtHours hourlyRate prepTimeFactor paperworkFee transportCost : ℕ) : ℕ :=
  let totalHours := courtHours * (1 + prepTimeFactor)
  let totalFee := upfrontFee + (totalHours * hourlyRate) + paperworkFee + transportCost
  totalFee / 2

theorem john_lawyer_payment :
  johnPayment 1000 50 100 2 500 300 = 8400 := by
  sorry

end NUMINAMATH_CALUDE_john_lawyer_payment_l2054_205488


namespace NUMINAMATH_CALUDE_min_value_inequality_l2054_205461

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) * (1 / x + 1 / y) ≥ 4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (a + b) * (1 / a + 1 / b) = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2054_205461


namespace NUMINAMATH_CALUDE_max_diagonal_sum_l2054_205494

/-- A rhombus with side length 5 -/
structure Rhombus where
  side_length : ℝ
  side_length_eq : side_length = 5

/-- The diagonals of the rhombus -/
structure RhombusDiagonals (r : Rhombus) where
  d1 : ℝ
  d2 : ℝ
  d1_le_6 : d1 ≤ 6
  d2_ge_6 : d2 ≥ 6

/-- The sum of the diagonals -/
def diagonal_sum (r : Rhombus) (d : RhombusDiagonals r) : ℝ := d.d1 + d.d2

/-- The theorem stating the maximum sum of diagonals -/
theorem max_diagonal_sum (r : Rhombus) :
  ∃ (d : RhombusDiagonals r), ∀ (d' : RhombusDiagonals r), diagonal_sum r d ≥ diagonal_sum r d' ∧ diagonal_sum r d = 14 :=
sorry

end NUMINAMATH_CALUDE_max_diagonal_sum_l2054_205494


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l2054_205446

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (corrected_mean : ℝ) :
  n = 50 ∧ 
  wrong_value = 23 ∧ 
  correct_value = 46 ∧ 
  corrected_mean = 36.5 →
  (n * corrected_mean - (correct_value - wrong_value)) / n = 36.04 := by
  sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l2054_205446


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2054_205423

theorem complex_equation_solution (z : ℂ) : (1 + 2 * Complex.I) * z = 3 - Complex.I → z = 1/5 - 7/5 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2054_205423


namespace NUMINAMATH_CALUDE_equation_solution_l2054_205476

theorem equation_solution : ∃ x : ℝ, (x / (2 * x - 5) + 5 / (5 - 2 * x) = 1) ∧ 
                            (2 * x - 5 ≠ 0) ∧ (5 - 2 * x ≠ 0) ∧ (x = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2054_205476


namespace NUMINAMATH_CALUDE_tree_calculation_l2054_205406

theorem tree_calculation (T P R : ℝ) (h1 : T = 400) (h2 : P = 0.20) (h3 : R = 5) :
  T - (P * T) + (P * T * R) = 720 :=
by sorry

end NUMINAMATH_CALUDE_tree_calculation_l2054_205406


namespace NUMINAMATH_CALUDE_probability_not_hearing_favorite_song_l2054_205444

-- Define the number of songs
def num_songs : ℕ := 12

-- Define the length of the shortest song in seconds
def shortest_song : ℕ := 45

-- Define the common difference between song lengths
def song_length_diff : ℕ := 45

-- Define the length of the favorite song in seconds
def favorite_song_length : ℕ := 375

-- Define the total listening time in seconds
def total_listening_time : ℕ := 420

-- Function to calculate the length of the nth song
def song_length (n : ℕ) : ℕ := shortest_song + (n - 1) * song_length_diff

-- Theorem stating the probability of not hearing the entire favorite song
theorem probability_not_hearing_favorite_song :
  let total_orderings := num_songs.factorial
  let favorable_orderings := 3 * (num_songs - 1).factorial
  (total_orderings - favorable_orderings) / total_orderings = 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_probability_not_hearing_favorite_song_l2054_205444


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_of_seven_l2054_205441

theorem least_four_digit_multiple_of_seven : 
  (∀ n : ℕ, n < 1001 → n % 7 ≠ 0 ∨ n < 1000) ∧ 1001 % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_of_seven_l2054_205441


namespace NUMINAMATH_CALUDE_product_remainder_l2054_205434

theorem product_remainder (a b m : ℕ) (ha : a = 103) (hb : b = 107) (hm : m = 13) :
  (a * b) % m = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l2054_205434


namespace NUMINAMATH_CALUDE_remainder_theorem_l2054_205410

theorem remainder_theorem : ∃ q : ℕ, 3^303 + 303 = (3^151 + 3^76 + 1) * q + 303 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2054_205410


namespace NUMINAMATH_CALUDE_log_sum_equals_ten_l2054_205478

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_sum_equals_ten :
  log 3 243 - log 3 (1/27) + log 3 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_ten_l2054_205478


namespace NUMINAMATH_CALUDE_total_volume_is_114_l2054_205491

/-- The volume of a cube with side length s -/
def cube_volume (s : ℝ) : ℝ := s^3

/-- The number of Carl's cubes -/
def carl_cubes : ℕ := 4

/-- The side length of Carl's cubes -/
def carl_side_length : ℝ := 3

/-- The number of Kate's cubes -/
def kate_cubes : ℕ := 6

/-- The side length of Kate's cubes -/
def kate_side_length : ℝ := 1

/-- The total volume of all cubes -/
def total_volume : ℝ :=
  (carl_cubes : ℝ) * cube_volume carl_side_length +
  (kate_cubes : ℝ) * cube_volume kate_side_length

theorem total_volume_is_114 : total_volume = 114 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_is_114_l2054_205491


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2054_205430

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular polygon satisfies the given condition if the number of its diagonals
    plus 6 equals twice the number of its sides -/
def satisfies_condition (n : ℕ) : Prop :=
  num_diagonals n + 6 = 2 * n

theorem regular_polygon_sides :
  ∃ (n : ℕ), n > 2 ∧ satisfies_condition n ∧ n = 4 :=
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2054_205430


namespace NUMINAMATH_CALUDE_fraction_reducibility_l2054_205427

theorem fraction_reducibility (n : ℕ) :
  (∃ k : ℕ, k > 1 ∧ (n^2 + 1).gcd (n + 1) = k) ↔ n % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_reducibility_l2054_205427


namespace NUMINAMATH_CALUDE_percentage_relationship_l2054_205469

theorem percentage_relationship (A B n c : ℝ) : 
  A > 0 → B > 0 → B > A → 
  A * (1 + n / 100) = B → B * (1 - c / 100) = A →
  A * Real.sqrt (100 + n) = B * Real.sqrt (100 - c) := by
sorry

end NUMINAMATH_CALUDE_percentage_relationship_l2054_205469


namespace NUMINAMATH_CALUDE_return_amount_calculation_l2054_205453

-- Define the borrowed amount
def borrowed_amount : ℝ := 100

-- Define the interest rate
def interest_rate : ℝ := 0.10

-- Theorem to prove
theorem return_amount_calculation :
  borrowed_amount * (1 + interest_rate) = 110 := by
  sorry

end NUMINAMATH_CALUDE_return_amount_calculation_l2054_205453


namespace NUMINAMATH_CALUDE_initial_queue_size_l2054_205489

theorem initial_queue_size (n : ℕ) : 
  (∀ A : ℕ, A = 41 * n) →  -- Current total age
  (A + 69 = 45 * (n + 1)) → -- New total age after 7th person joins
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_initial_queue_size_l2054_205489


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2054_205485

theorem quadratic_equation_solution (a b m : ℤ) : 
  (∀ x, a * x^2 + 24 * x + b = (m * x - 3)^2) → 
  (a = 16 ∧ b = 9 ∧ m = -4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2054_205485


namespace NUMINAMATH_CALUDE_inequality_proof_root_inequality_l2054_205496

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hne : ¬(a = b ∧ b = c)) : 
  a*(b^2 + c^2) + b*(c^2 + a^2) + c*(a^2 + b^2) > 6*a*b*c :=
sorry

theorem root_inequality : Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_root_inequality_l2054_205496


namespace NUMINAMATH_CALUDE_complex_number_equality_l2054_205455

theorem complex_number_equality : Complex.abs ((1 - Complex.I) / (1 + Complex.I)) + 2 * Complex.I = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l2054_205455


namespace NUMINAMATH_CALUDE_smallest_w_value_l2054_205467

theorem smallest_w_value (w : ℕ+) : 
  (∃ k : ℕ+, (2547 * w) = k * (2^6 * 3^5 * 5^4 * 7^3 * 13^4)) → 
  w ≥ 1592010000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_w_value_l2054_205467


namespace NUMINAMATH_CALUDE_function_increment_l2054_205463

theorem function_increment (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 19) ≤ f x + 19) 
  (h2 : ∀ x, f (x + 94) ≥ f x + 94) : 
  ∀ x, f (x + 1) = f x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_increment_l2054_205463


namespace NUMINAMATH_CALUDE_logarithm_equation_solution_l2054_205447

theorem logarithm_equation_solution (x : ℝ) (hx_pos : x > 0) (hx_neq_one : x ≠ 1) :
  (Real.log 2 / Real.log x) * (Real.log 2 / Real.log (2 * x)) = Real.log 2 / Real.log (4 * x) →
  x = 2 ^ Real.sqrt 2 ∨ x = 2 ^ (-Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equation_solution_l2054_205447


namespace NUMINAMATH_CALUDE_range_of_a_in_linear_inequality_l2054_205449

/-- The range of values for parameter 'a' in the inequality 2x - y + a > 0,
    given that only one point among (0,0) and (1,1) is inside the region. -/
theorem range_of_a_in_linear_inequality :
  ∃ (a : ℝ), (∀ x y : ℝ, 2*x - y + a > 0 →
    ((x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) →
    (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) ∧
  (-1 < a ∧ a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_in_linear_inequality_l2054_205449


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2054_205407

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^10 + x^4 + 1 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
    a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9 + a₁₀*(x+1)^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2054_205407


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l2054_205419

-- Define the quadratic function
def f (a b x : ℝ) := a * x^2 - (a + 1) * x + b

-- Define the solution set condition
def solution_set (a b : ℝ) :=
  ∀ x, f a b x < 0 ↔ (x < -1/2 ∨ x > 1)

-- Define the inequality for part II
def g (x m : ℝ) := x^2 + (m - 4) * x + 3 - m

-- Main theorem
theorem quadratic_inequality_problem :
  ∃ a b : ℝ,
    solution_set a b ∧
    a = -2 ∧
    b = 1 ∧
    (∀ x, (∀ m ∈ Set.Icc 0 4, g x m ≥ 0) ↔ 
      (x ≤ -1 ∨ x = 1 ∨ x ≥ 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l2054_205419


namespace NUMINAMATH_CALUDE_undefined_condition_l2054_205400

theorem undefined_condition (y : ℝ) : 
  ¬(∃ x : ℝ, x = (3 * y^3 + 5) / (y^2 - 18*y + 81)) ↔ y = 9 := by
  sorry

end NUMINAMATH_CALUDE_undefined_condition_l2054_205400


namespace NUMINAMATH_CALUDE_power_sum_equality_l2054_205458

theorem power_sum_equality : (-1)^45 + 2^(3^2 + 5^2 - 4^2) = 262143 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2054_205458


namespace NUMINAMATH_CALUDE_unique_solution_l2054_205481

/-- Two lines in a 2D plane -/
structure TwoLines where
  l₁ : ℝ → ℝ → ℝ  -- represents ax - by + 4 = 0
  l₂ : ℝ → ℝ → ℝ  -- represents (a - 1)x + y + b = 0
  a : ℝ
  b : ℝ

/-- Condition that l₁ is perpendicular to l₂ -/
def perpendicular (lines : TwoLines) : Prop :=
  lines.a * (lines.a - 1) - lines.b = 0

/-- Condition that l₁ passes through point (-3, -1) -/
def passes_through (lines : TwoLines) : Prop :=
  lines.l₁ (-3) (-1) = 0

/-- Condition that l₁ is parallel to l₂ -/
def parallel (lines : TwoLines) : Prop :=
  lines.a / lines.b = 1 - lines.a

/-- Condition that the distance from origin to both lines is equal -/
def equal_distance (lines : TwoLines) : Prop :=
  4 / lines.b = -lines.b

/-- The main theorem -/
theorem unique_solution (lines : TwoLines) :
  perpendicular lines ∧ passes_through lines →
  parallel lines ∧ equal_distance lines →
  lines.a = 2 ∧ lines.b = -2 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2054_205481


namespace NUMINAMATH_CALUDE_soap_bars_per_pack_l2054_205471

/-- Given that Nancy bought 6 packs of soap and 30 bars of soap in total,
    prove that the number of bars in each pack is 5. -/
theorem soap_bars_per_pack :
  ∀ (total_packs : ℕ) (total_bars : ℕ),
    total_packs = 6 →
    total_bars = 30 →
    total_bars / total_packs = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_soap_bars_per_pack_l2054_205471


namespace NUMINAMATH_CALUDE_radish_count_l2054_205487

theorem radish_count (total : ℕ) (difference : ℕ) (radishes : ℕ) : 
  total = 100 →
  difference = 24 →
  radishes = total - difference / 2 →
  radishes = 62 := by
sorry

end NUMINAMATH_CALUDE_radish_count_l2054_205487


namespace NUMINAMATH_CALUDE_probability_distribution_problem_l2054_205474

theorem probability_distribution_problem (m n : ℝ) 
  (sum_prob : 0.1 + m + n + 0.1 = 1)
  (condition : m + 2 * n = 1.2) : n = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_probability_distribution_problem_l2054_205474


namespace NUMINAMATH_CALUDE_susan_missed_pay_l2054_205433

/-- Calculates the pay Susan will miss during her three-week vacation --/
def missed_pay (regular_rate : ℚ) (overtime_rate : ℚ) (sunday_rate : ℚ) 
               (regular_hours : ℕ) (overtime_hours : ℕ) (sunday_hours : ℕ)
               (sunday_count : List ℕ) (vacation_days : ℕ) (workweek_days : ℕ) : ℚ :=
  let weekly_pay := regular_rate * regular_hours + overtime_rate * overtime_hours
  let sunday_pay := sunday_rate * sunday_hours * (sunday_count.sum)
  let total_pay := weekly_pay * 3 + sunday_pay
  let paid_vacation_pay := regular_rate * regular_hours * (vacation_days / workweek_days)
  total_pay - paid_vacation_pay

/-- The main theorem stating that Susan will miss $2160 during her vacation --/
theorem susan_missed_pay : 
  missed_pay 15 20 25 40 8 8 [1, 2, 0] 6 5 = 2160 := by
  sorry

end NUMINAMATH_CALUDE_susan_missed_pay_l2054_205433


namespace NUMINAMATH_CALUDE_magnitude_equality_not_implies_vector_equality_l2054_205459

-- Define vectors a and b in a real vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (a b : V)

-- State the theorem
theorem magnitude_equality_not_implies_vector_equality :
  ∃ a b : V, (‖a‖ = 3 * ‖b‖) ∧ (a ≠ 3 • b) ∧ (a ≠ -3 • b) := by
  sorry

end NUMINAMATH_CALUDE_magnitude_equality_not_implies_vector_equality_l2054_205459


namespace NUMINAMATH_CALUDE_hare_wins_by_10_meters_l2054_205456

-- Define the race parameters
def race_duration : ℕ := 50
def hare_initial_speed : ℕ := 12
def hare_later_speed : ℕ := 1
def tortoise_speed : ℕ := 3

-- Define the function to calculate the hare's distance
def hare_distance (initial_time : ℕ) : ℕ :=
  (initial_time * hare_initial_speed) + ((race_duration - initial_time) * hare_later_speed)

-- Define the function to calculate the tortoise's distance
def tortoise_distance : ℕ := race_duration * tortoise_speed

-- Theorem statement
theorem hare_wins_by_10_meters :
  ∃ (initial_time : ℕ), initial_time < race_duration ∧ 
  hare_distance initial_time = tortoise_distance + 10 :=
sorry

end NUMINAMATH_CALUDE_hare_wins_by_10_meters_l2054_205456


namespace NUMINAMATH_CALUDE_alcohol_dilution_l2054_205482

/-- Proves that adding 3 litres of water to 18 litres of a 20% alcohol mixture 
    results in a new mixture with 17.14285714285715% alcohol. -/
theorem alcohol_dilution (initial_volume : ℝ) (initial_percentage : ℝ) 
    (water_added : ℝ) (final_percentage : ℝ) : 
    initial_volume = 18 →
    initial_percentage = 0.20 →
    water_added = 3 →
    final_percentage = 0.1714285714285715 →
    (initial_volume * initial_percentage) / (initial_volume + water_added) = final_percentage := by
  sorry


end NUMINAMATH_CALUDE_alcohol_dilution_l2054_205482


namespace NUMINAMATH_CALUDE_makeup_problem_solution_l2054_205440

/-- Represents the makeup problem with given parameters -/
structure MakeupProblem where
  people_per_tube : ℕ
  total_people : ℕ
  num_tubs : ℕ

/-- Calculates the number of tubes per tub for a given makeup problem -/
def tubes_per_tub (p : MakeupProblem) : ℕ :=
  (p.total_people / p.people_per_tube) / p.num_tubs

/-- Theorem stating that for the given problem, the number of tubes per tub is 2 -/
theorem makeup_problem_solution :
  let p : MakeupProblem := ⟨3, 36, 6⟩
  tubes_per_tub p = 2 := by
  sorry

end NUMINAMATH_CALUDE_makeup_problem_solution_l2054_205440


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2054_205466

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 15 + 20 + 6 + y) / 5 = 12 → y = 11 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2054_205466


namespace NUMINAMATH_CALUDE_brass_composition_ratio_l2054_205428

theorem brass_composition_ratio (total_mass zinc_mass : ℝ) 
  (h_total : total_mass = 100)
  (h_zinc : zinc_mass = 35) :
  (total_mass - zinc_mass) / zinc_mass = 13 / 7 := by
  sorry

end NUMINAMATH_CALUDE_brass_composition_ratio_l2054_205428


namespace NUMINAMATH_CALUDE_regular_1001_gon_labeling_existence_l2054_205490

theorem regular_1001_gon_labeling_existence :
  ∃ f : Fin 1001 → Fin 1001,
    Function.Bijective f ∧
    ∀ (r : Fin 1001) (b : Bool),
      ∃ i : Fin 1001,
        f ((i + r) % 1001) = if b then i else (1001 - i) % 1001 := by
  sorry

end NUMINAMATH_CALUDE_regular_1001_gon_labeling_existence_l2054_205490


namespace NUMINAMATH_CALUDE_monotone_decreasing_range_l2054_205422

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - x else -x^2 + a

theorem monotone_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_monotone_decreasing_range_l2054_205422


namespace NUMINAMATH_CALUDE_previous_year_300th_day_l2054_205479

/-- Represents the days of the week -/
inductive DayOfWeek
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.monday => DayOfWeek.tuesday
  | DayOfWeek.tuesday => DayOfWeek.wednesday
  | DayOfWeek.wednesday => DayOfWeek.thursday
  | DayOfWeek.thursday => DayOfWeek.friday
  | DayOfWeek.friday => DayOfWeek.saturday
  | DayOfWeek.saturday => DayOfWeek.sunday
  | DayOfWeek.sunday => DayOfWeek.monday

/-- Returns the day of the week after n days -/
def addDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => nextDay (addDays d n)

theorem previous_year_300th_day 
  (current_year_200th_day : DayOfWeek)
  (next_year_100th_day : DayOfWeek)
  (h1 : current_year_200th_day = DayOfWeek.sunday)
  (h2 : next_year_100th_day = DayOfWeek.sunday)
  : addDays DayOfWeek.monday 299 = current_year_200th_day :=
by sorry

#check previous_year_300th_day

end NUMINAMATH_CALUDE_previous_year_300th_day_l2054_205479


namespace NUMINAMATH_CALUDE_blue_face_probability_l2054_205426

/-- The probability of rolling a blue face on a 12-sided die with 4 blue faces is 1/3 -/
theorem blue_face_probability (total_sides : ℕ) (blue_faces : ℕ) 
  (h1 : total_sides = 12) (h2 : blue_faces = 4) : 
  (blue_faces : ℚ) / total_sides = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_face_probability_l2054_205426


namespace NUMINAMATH_CALUDE_equation_rewrite_l2054_205477

theorem equation_rewrite (x y : ℝ) : 5 * x + 3 * y = 1 ↔ y = (1 - 5 * x) / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_rewrite_l2054_205477


namespace NUMINAMATH_CALUDE_missing_digit_is_seven_l2054_205417

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def insert_digit (d : ℕ) : ℕ := 351000 + d * 100 + 92

theorem missing_digit_is_seven :
  ∃! d : ℕ, d < 10 ∧ is_divisible_by_9 (insert_digit d) :=
by
  sorry

#check missing_digit_is_seven

end NUMINAMATH_CALUDE_missing_digit_is_seven_l2054_205417


namespace NUMINAMATH_CALUDE_max_n_is_14_l2054_205460

/-- A function that divides a list of integers into two groups -/
def divide_into_groups (n : ℕ) : (List ℕ) × (List ℕ) := sorry

/-- Predicate to check if a list contains no pair of numbers that sum to a perfect square -/
def no_square_sum (l : List ℕ) : Prop := sorry

/-- Predicate to check if two lists have no common elements -/
def no_common_elements (l1 l2 : List ℕ) : Prop := sorry

/-- The main theorem stating that 14 is the maximum value of n satisfying the conditions -/
theorem max_n_is_14 : 
  ∀ n : ℕ, n > 14 → 
  ¬∃ (g1 g2 : List ℕ), 
    (divide_into_groups n = (g1, g2)) ∧ 
    (no_square_sum g1) ∧ 
    (no_square_sum g2) ∧ 
    (no_common_elements g1 g2) ∧ 
    (g1.length + g2.length = n) ∧
    (∀ i : ℕ, i ∈ g1 ∨ i ∈ g2 ↔ 1 ≤ i ∧ i ≤ n) :=
sorry

end NUMINAMATH_CALUDE_max_n_is_14_l2054_205460


namespace NUMINAMATH_CALUDE_median_inequality_l2054_205457

/-- Given a triangle ABC with sides a, b, c and medians s_a, s_b, s_c,
    if a < (b+c)/2, then s_a > (s_b + s_c)/2 -/
theorem median_inequality (a b c s_a s_b s_c : ℝ) 
    (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
    (h_medians : s_a = (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2) ∧
                 s_b = (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2) ∧
                 s_c = (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2))
    (h_cond : a < (b + c) / 2) :
  s_a > (s_b + s_c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_median_inequality_l2054_205457


namespace NUMINAMATH_CALUDE_min_games_for_prediction_l2054_205418

/-- Represents the chess tournament setup -/
structure ChessTournament where
  white_rook_students : ℕ
  black_elephant_students : ℕ
  total_games : ℕ
  games_per_white_student : ℕ

/-- Defines the specific chess tournament in the problem -/
def problem_tournament : ChessTournament :=
  { white_rook_students := 15,
    black_elephant_students := 20,
    total_games := 300,
    games_per_white_student := 20 }

/-- Theorem stating the minimum number of games for Sasha's prediction -/
theorem min_games_for_prediction (t : ChessTournament) 
  (h1 : t.white_rook_students * t.black_elephant_students = t.total_games)
  (h2 : t.games_per_white_student = t.black_elephant_students) :
  t.total_games - (t.white_rook_students - 1) * t.games_per_white_student = 280 :=
sorry

end NUMINAMATH_CALUDE_min_games_for_prediction_l2054_205418


namespace NUMINAMATH_CALUDE_polynomial_degree_product_l2054_205414

-- Define the polynomials
def p (x : ℝ) := 5*x^3 - 4*x + 7
def q (x : ℝ) := 2*x^2 + 9

-- State the theorem
theorem polynomial_degree_product : 
  Polynomial.degree ((Polynomial.monomial 0 1 + Polynomial.X)^10 * (Polynomial.monomial 0 1 + Polynomial.X)^5) = 40 := by
  sorry


end NUMINAMATH_CALUDE_polynomial_degree_product_l2054_205414


namespace NUMINAMATH_CALUDE_probability_three_blue_jellybeans_is_two_nineteenths_l2054_205425

/-- The probability of drawing 3 blue jellybeans in succession without replacement
    from a bag containing 10 blue and 10 red jellybeans -/
def probability_three_blue_jellybeans : ℚ :=
  let total_jellybeans : ℕ := 20
  let blue_jellybeans : ℕ := 10
  let red_jellybeans : ℕ := 10
  let draws : ℕ := 3
  (blue_jellybeans * (blue_jellybeans - 1) * (blue_jellybeans - 2)) /
  (total_jellybeans * (total_jellybeans - 1) * (total_jellybeans - 2))

theorem probability_three_blue_jellybeans_is_two_nineteenths :
  probability_three_blue_jellybeans = 2 / 19 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_blue_jellybeans_is_two_nineteenths_l2054_205425


namespace NUMINAMATH_CALUDE_parabola_vertex_l2054_205415

/-- The vertex of the parabola y = x^2 - 2 is at (0, -2) -/
theorem parabola_vertex :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2
  ∃! (h k : ℝ), (∀ x, f x = (x - h)^2 + k) ∧ h = 0 ∧ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2054_205415


namespace NUMINAMATH_CALUDE_total_gold_value_proof_l2054_205462

/-- The value of one gold bar in dollars -/
def gold_bar_value : ℕ := 2200

/-- The number of gold bars Legacy has -/
def legacy_bars : ℕ := 5

/-- The number of gold bars Aleena has -/
def aleena_bars : ℕ := legacy_bars - 2

/-- The total value of gold for Legacy and Aleena -/
def total_gold_value : ℕ := gold_bar_value * (legacy_bars + aleena_bars)

theorem total_gold_value_proof : total_gold_value = 17600 := by
  sorry

end NUMINAMATH_CALUDE_total_gold_value_proof_l2054_205462


namespace NUMINAMATH_CALUDE_function_value_at_six_l2054_205443

/-- Given a function f such that f(4x+2) = x^2 - x + 1 for all real x, prove that f(6) = 1/2 -/
theorem function_value_at_six (f : ℝ → ℝ) (h : ∀ x : ℝ, f (4 * x + 2) = x^2 - x + 1) : f 6 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_six_l2054_205443


namespace NUMINAMATH_CALUDE_museum_tour_time_l2054_205483

theorem museum_tour_time (total_students : ℕ) (num_groups : ℕ) (time_per_student : ℕ) 
  (h1 : total_students = 18)
  (h2 : num_groups = 3)
  (h3 : time_per_student = 4)
  (h4 : total_students % num_groups = 0) : -- Ensuring equal groups
  (total_students / num_groups) * time_per_student = 24 := by
  sorry

end NUMINAMATH_CALUDE_museum_tour_time_l2054_205483


namespace NUMINAMATH_CALUDE_coin_toss_experiment_l2054_205403

theorem coin_toss_experiment (total_tosses : ℕ) (heads_frequency : ℚ) 
  (h1 : total_tosses = 100)
  (h2 : heads_frequency = 49/100) :
  total_tosses - (total_tosses * heads_frequency).num = 51 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_experiment_l2054_205403


namespace NUMINAMATH_CALUDE_expression_equality_l2054_205439

theorem expression_equality : 1 - (1 / (1 + Real.sqrt 2)) + (1 / (1 - Real.sqrt 2)) = 1 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2054_205439


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2054_205452

theorem inequality_system_solution_set :
  {x : ℝ | x + 2 ≤ 3 ∧ 1 + x > -2} = {x : ℝ | -3 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2054_205452


namespace NUMINAMATH_CALUDE_circles_intersect_l2054_205402

/-- Two circles are intersecting if the distance between their centers is greater than the absolute
    difference of their radii and less than the sum of their radii. -/
def are_circles_intersecting (r1 r2 d : ℝ) : Prop :=
  abs (r1 - r2) < d ∧ d < r1 + r2

/-- Given two circles with radii 4 and 3, and a distance of 5 between their centers,
    prove that they are intersecting. -/
theorem circles_intersect : are_circles_intersecting 4 3 5 := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l2054_205402
