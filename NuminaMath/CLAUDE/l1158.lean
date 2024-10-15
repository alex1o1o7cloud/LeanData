import Mathlib

namespace NUMINAMATH_CALUDE_second_fold_perpendicular_l1158_115823

/-- Represents a point on a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line on a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Represents a sheet of paper with one straight edge -/
structure Paper :=
  (straight_edge : Line)

/-- Represents a fold on the paper -/
structure Fold :=
  (line : Line)
  (paper : Paper)

/-- Checks if a point is on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Theorem: The second fold creates a line perpendicular to the initial crease -/
theorem second_fold_perpendicular 
  (paper : Paper) 
  (initial_fold : Fold)
  (A : Point)
  (second_fold : Fold)
  (h1 : point_on_line A paper.straight_edge)
  (h2 : point_on_line A initial_fold.line)
  (h3 : point_on_line A second_fold.line)
  (h4 : ∃ (p q : Point), 
    point_on_line p paper.straight_edge ∧ 
    point_on_line q paper.straight_edge ∧
    point_on_line p second_fold.line ∧
    point_on_line q initial_fold.line) :
  perpendicular initial_fold.line second_fold.line :=
sorry

end NUMINAMATH_CALUDE_second_fold_perpendicular_l1158_115823


namespace NUMINAMATH_CALUDE_intensity_for_three_breaks_l1158_115866

/-- Represents the relationship between breaks and intensity -/
def inverse_proportional (breaks intensity : ℝ) (k : ℝ) : Prop :=
  breaks * intensity = k

theorem intensity_for_three_breaks 
  (k : ℝ) 
  (h1 : inverse_proportional 4 6 k) 
  (h2 : inverse_proportional 3 8 k) : 
  True :=
sorry

end NUMINAMATH_CALUDE_intensity_for_three_breaks_l1158_115866


namespace NUMINAMATH_CALUDE_maria_coin_difference_l1158_115852

/-- Represents the number of coins of each denomination -/
structure CoinCollection where
  five_cent : ℕ
  ten_cent : ℕ
  twenty_cent : ℕ
  twenty_five_cent : ℕ

/-- The conditions of Maria's coin collection -/
def maria_collection (c : CoinCollection) : Prop :=
  c.five_cent + c.ten_cent + c.twenty_cent + c.twenty_five_cent = 30 ∧
  c.ten_cent = 2 * c.five_cent ∧
  5 * c.five_cent + 10 * c.ten_cent + 20 * c.twenty_cent + 25 * c.twenty_five_cent = 410

theorem maria_coin_difference (c : CoinCollection) : 
  maria_collection c → c.twenty_five_cent - c.twenty_cent = 1 := by
  sorry

end NUMINAMATH_CALUDE_maria_coin_difference_l1158_115852


namespace NUMINAMATH_CALUDE_least_possible_QR_length_l1158_115844

theorem least_possible_QR_length (PQ PR SR QS : ℝ) (hPQ : PQ = 7)
  (hPR : PR = 15) (hSR : SR = 10) (hQS : QS = 24) :
  ∃ (QR : ℕ), QR ≥ 14 ∧ ∀ (n : ℕ), n ≥ 14 → 
  (QR : ℝ) > PR - PQ ∧ (QR : ℝ) > QS - SR := by
  sorry

end NUMINAMATH_CALUDE_least_possible_QR_length_l1158_115844


namespace NUMINAMATH_CALUDE_quadratic_root_property_l1158_115841

theorem quadratic_root_property (m : ℝ) : 
  m^2 + 2*m - 1 = 0 → 2*m^2 + 4*m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l1158_115841


namespace NUMINAMATH_CALUDE_rug_area_l1158_115875

/-- The area of a rug on a rectangular floor with uncovered strips along the edges -/
theorem rug_area (floor_length floor_width strip_width : ℝ) 
  (h1 : floor_length = 12)
  (h2 : floor_width = 10)
  (h3 : strip_width = 3)
  (h4 : floor_length > 0)
  (h5 : floor_width > 0)
  (h6 : strip_width > 0)
  (h7 : 2 * strip_width < floor_length)
  (h8 : 2 * strip_width < floor_width) :
  (floor_length - 2 * strip_width) * (floor_width - 2 * strip_width) = 24 := by
  sorry

end NUMINAMATH_CALUDE_rug_area_l1158_115875


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l1158_115873

/-- The function f(x) = x^2(x-2) + 1 -/
def f (x : ℝ) : ℝ := x^2 * (x - 2) + 1

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 2*x*(x - 2) + x^2

theorem tangent_line_at_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x + y - 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l1158_115873


namespace NUMINAMATH_CALUDE_no_18_pretty_below_1500_l1158_115838

def is_m_pretty (n m : ℕ+) : Prop :=
  (Nat.divisors n).card = m ∧ m ∣ n

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem no_18_pretty_below_1500 :
  ∀ n : ℕ+,
  n < 1500 →
  (∃ (a b : ℕ) (k : ℕ+), 
    n = 2^a * 7^b * k ∧
    a ≥ 1 ∧
    b ≥ 1 ∧
    is_coprime k.val 14 ∧
    (Nat.divisors n.val).card = 18) →
  ¬(is_m_pretty n 18) :=
sorry

end NUMINAMATH_CALUDE_no_18_pretty_below_1500_l1158_115838


namespace NUMINAMATH_CALUDE_square_area_above_line_l1158_115877

/-- Given a square with vertices at (2,1), (7,1), (7,6), and (2,6),
    and a line connecting points (2,1) and (7,3),
    the fraction of the square's area above this line is 4/5. -/
theorem square_area_above_line : 
  let square_vertices : List (ℝ × ℝ) := [(2,1), (7,1), (7,6), (2,6)]
  let line_points : List (ℝ × ℝ) := [(2,1), (7,3)]
  let total_area : ℝ := 25
  let area_above_line : ℝ := 20
  (area_above_line / total_area) = 4/5 := by sorry

end NUMINAMATH_CALUDE_square_area_above_line_l1158_115877


namespace NUMINAMATH_CALUDE_sum_of_digits_of_N_l1158_115854

def N : ℕ := 10^3 + 10^4 + 10^5 + 10^6 + 10^7 + 10^8 + 10^9

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_N : sum_of_digits N = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_N_l1158_115854


namespace NUMINAMATH_CALUDE_total_bacon_needed_l1158_115855

/-- The number of eggs on each breakfast plate -/
def eggs_per_plate : ℕ := 2

/-- The number of customers ordering breakfast plates -/
def num_customers : ℕ := 14

/-- The number of bacon strips on each breakfast plate -/
def bacon_per_plate : ℕ := 2 * eggs_per_plate

/-- The total number of bacon strips needed -/
def total_bacon : ℕ := num_customers * bacon_per_plate

theorem total_bacon_needed : total_bacon = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_bacon_needed_l1158_115855


namespace NUMINAMATH_CALUDE_is_integer_division_l1158_115813

theorem is_integer_division : ∃ k : ℤ, (19^92 - 91^29) / 90 = k := by
  sorry

end NUMINAMATH_CALUDE_is_integer_division_l1158_115813


namespace NUMINAMATH_CALUDE_linear_function_straight_line_l1158_115861

-- Define a linear function
def LinearFunction (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * x + b

-- Define the property of having a straight line graph
def HasStraightLineGraph (f : ℝ → ℝ) : Prop := 
  ∃ a b : ℝ, ∀ x y : ℝ, f y - f x = a * (y - x)

-- Define our specific function
def f (x : ℝ) : ℝ := 2 * x + 5

-- State the theorem
theorem linear_function_straight_line :
  (∀ g : ℝ → ℝ, LinearFunction g → HasStraightLineGraph g) →
  LinearFunction f →
  HasStraightLineGraph f :=
by
  sorry

end NUMINAMATH_CALUDE_linear_function_straight_line_l1158_115861


namespace NUMINAMATH_CALUDE_range_of_a_l1158_115837

-- Define a decreasing function on (-1, 1)
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y

-- State the theorem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : DecreasingFunction f) 
  (h2 : f (1 - a) < f (2 * a - 1)) :
  0 < a ∧ a < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1158_115837


namespace NUMINAMATH_CALUDE_barefoot_kids_l1158_115867

theorem barefoot_kids (total : ℕ) (with_socks : ℕ) (with_shoes : ℕ) (with_both : ℕ) : 
  total = 22 →
  with_socks = 12 →
  with_shoes = 8 →
  with_both = 6 →
  total - (with_socks + with_shoes - with_both) = 8 :=
by sorry

end NUMINAMATH_CALUDE_barefoot_kids_l1158_115867


namespace NUMINAMATH_CALUDE_fraction_simplification_l1158_115878

theorem fraction_simplification (x y z : ℝ) (hx : x = 3) (hy : y = 2) (hz : z = 5) :
  (15 * x^2 * y^3) / (9 * x * y^2 * z) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1158_115878


namespace NUMINAMATH_CALUDE_competition_results_l1158_115810

/-- Represents the score for a single competition -/
structure CompetitionScore where
  first : ℕ+
  second : ℕ+
  third : ℕ+
  first_gt_second : first > second
  second_gt_third : second > third

/-- Represents the results of all competitions -/
structure CompetitionResults where
  score : CompetitionScore
  num_competitions : ℕ+
  a_total_score : ℕ
  b_total_score : ℕ
  c_total_score : ℕ
  b_first_place_count : ℕ

/-- The main theorem statement -/
theorem competition_results 
  (res : CompetitionResults)
  (h1 : res.num_competitions = 6)
  (h2 : res.a_total_score = 26)
  (h3 : res.b_total_score = 11)
  (h4 : res.c_total_score = 11)
  (h5 : res.b_first_place_count = 1) :
  ∃ (b_third_place_count : ℕ), b_third_place_count = 4 := by
  sorry

end NUMINAMATH_CALUDE_competition_results_l1158_115810


namespace NUMINAMATH_CALUDE_tan_roots_expression_value_l1158_115899

theorem tan_roots_expression_value (α β : ℝ) :
  (∃ x y : ℝ, x^2 - 4*x - 2 = 0 ∧ y^2 - 4*y - 2 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  Real.cos (α + β)^2 + 2 * Real.sin (α + β) * Real.cos (α + β) - 3 * Real.sin (α + β)^2 = -3/5 :=
by sorry

end NUMINAMATH_CALUDE_tan_roots_expression_value_l1158_115899


namespace NUMINAMATH_CALUDE_sister_reams_proof_l1158_115816

/-- The number of reams of paper bought for Haley -/
def reams_for_haley : ℕ := 2

/-- The total number of reams of paper bought -/
def total_reams : ℕ := 5

/-- The number of reams of paper bought for Haley's sister -/
def reams_for_sister : ℕ := total_reams - reams_for_haley

theorem sister_reams_proof : reams_for_sister = 3 := by
  sorry

end NUMINAMATH_CALUDE_sister_reams_proof_l1158_115816


namespace NUMINAMATH_CALUDE_total_cost_is_21_16_l1158_115828

def sandwich_price : ℝ := 2.49
def soda_price : ℝ := 1.87
def chips_price : ℝ := 1.25
def chocolate_price : ℝ := 0.99

def sandwich_quantity : ℕ := 2
def soda_quantity : ℕ := 4
def chips_quantity : ℕ := 3
def chocolate_quantity : ℕ := 5

def total_cost : ℝ :=
  sandwich_price * sandwich_quantity +
  soda_price * soda_quantity +
  chips_price * chips_quantity +
  chocolate_price * chocolate_quantity

theorem total_cost_is_21_16 : total_cost = 21.16 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_21_16_l1158_115828


namespace NUMINAMATH_CALUDE_card_58_is_six_l1158_115871

/-- Represents a playing card value -/
inductive CardValue
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Converts a natural number to a card value -/
def natToCardValue (n : ℕ) : CardValue :=
  match n % 13 with
  | 0 => CardValue.Ace
  | 1 => CardValue.Two
  | 2 => CardValue.Three
  | 3 => CardValue.Four
  | 4 => CardValue.Five
  | 5 => CardValue.Six
  | 6 => CardValue.Seven
  | 7 => CardValue.Eight
  | 8 => CardValue.Nine
  | 9 => CardValue.Ten
  | 10 => CardValue.Jack
  | 11 => CardValue.Queen
  | _ => CardValue.King

theorem card_58_is_six :
  natToCardValue 57 = CardValue.Six :=
by sorry

end NUMINAMATH_CALUDE_card_58_is_six_l1158_115871


namespace NUMINAMATH_CALUDE_prime_sum_divisibility_l1158_115856

theorem prime_sum_divisibility (p : ℕ) : 
  Prime p → (7^p - 6^p + 2) % 43 = 0 → p = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_divisibility_l1158_115856


namespace NUMINAMATH_CALUDE_point_on_line_l1158_115874

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line :
  let p1 : Point := ⟨4, 8⟩
  let p2 : Point := ⟨0, -4⟩
  let p3 : Point := ⟨2, 2⟩
  collinear p1 p2 p3 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l1158_115874


namespace NUMINAMATH_CALUDE_f_sum_derivative_equals_two_l1158_115885

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem f_sum_derivative_equals_two :
  let f' := deriv f
  f 2017 + f' 2017 + f (-2017) - f' (-2017) = 2 := by sorry

end NUMINAMATH_CALUDE_f_sum_derivative_equals_two_l1158_115885


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1158_115801

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := 5*x - 6 > x^2

-- Define the relationship between ¬p and ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  (∃ x, ¬(p x) ∧ q x) := by
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1158_115801


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_l1158_115896

theorem quadratic_root_ratio (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : ∃ (x y : ℝ), x = 2022 * y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) :
  (2023 * a * c) / (b^2) = 2022 / 2023 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_l1158_115896


namespace NUMINAMATH_CALUDE_interest_rate_proof_l1158_115827

theorem interest_rate_proof (P : ℝ) (n : ℕ) (diff : ℝ) (r : ℝ) : 
  P = 5399.999999999995 →
  n = 2 →
  P * ((1 + r)^n - 1) - P * r * n = diff →
  diff = 216 →
  r = 0.2 :=
sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l1158_115827


namespace NUMINAMATH_CALUDE_sqrt_588_simplification_l1158_115872

theorem sqrt_588_simplification : Real.sqrt 588 = 14 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_588_simplification_l1158_115872


namespace NUMINAMATH_CALUDE_sin_cos_cube_sum_l1158_115820

theorem sin_cos_cube_sum (θ : ℝ) (h : 4 * Real.sin θ * Real.cos θ - 5 * Real.sin θ - 5 * Real.cos θ - 1 = 0) :
  Real.sin θ ^ 3 + Real.cos θ ^ 3 = -11/16 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_cube_sum_l1158_115820


namespace NUMINAMATH_CALUDE_problem_statement_l1158_115825

theorem problem_statement (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_eq1 : x + 1 / z = 5)
  (h_eq2 : y + 1 / x = 29) :
  z + 1 / y = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1158_115825


namespace NUMINAMATH_CALUDE_triangle_problem_l1158_115889

theorem triangle_problem (A B C : Real) (a b c S : Real) :
  -- Given conditions
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  a = Real.sqrt 3 →
  S = Real.sqrt 3 / 2 →
  -- Triangle properties
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  S = 1/2 * b * c * Real.sin A →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  -- Conclusion
  A = π/3 ∧ b + c = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1158_115889


namespace NUMINAMATH_CALUDE_watermelon_seeds_l1158_115809

/-- Given a watermelon cut into 40 slices, with each slice having an equal number of black and white seeds,
    and a total of 1,600 seeds in the watermelon, prove that there are 20 black seeds in each slice. -/
theorem watermelon_seeds (slices : ℕ) (total_seeds : ℕ) (black_seeds_per_slice : ℕ) :
  slices = 40 →
  total_seeds = 1600 →
  total_seeds = 2 * slices * black_seeds_per_slice →
  black_seeds_per_slice = 20 := by
  sorry

#check watermelon_seeds

end NUMINAMATH_CALUDE_watermelon_seeds_l1158_115809


namespace NUMINAMATH_CALUDE_bananas_arrangements_l1158_115860

def word_length : ℕ := 7
def a_count : ℕ := 3
def n_count : ℕ := 2

theorem bananas_arrangements : 
  (word_length.factorial) / (a_count.factorial * n_count.factorial) = 420 := by
  sorry

end NUMINAMATH_CALUDE_bananas_arrangements_l1158_115860


namespace NUMINAMATH_CALUDE_positive_X_value_l1158_115839

-- Define the # relation
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- Theorem statement
theorem positive_X_value (X : ℝ) (h : hash X 7 = 250) : X = Real.sqrt 201 :=
by sorry

end NUMINAMATH_CALUDE_positive_X_value_l1158_115839


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1158_115807

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1158_115807


namespace NUMINAMATH_CALUDE_xyz_sum_bounds_l1158_115891

theorem xyz_sum_bounds (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) :
  let m := x*y + y*z + z*x
  (∃ (k : ℝ), ∀ (a b c : ℝ), a^2 + b^2 + c^2 = 1 → x*y + y*z + z*x ≤ k) ∧
  (∃ (l : ℝ), ∀ (a b c : ℝ), a^2 + b^2 + c^2 = 1 → l ≤ x*y + y*z + z*x) ∧
  (∃ (a b c : ℝ), a^2 + b^2 + c^2 = 1 ∧ x*y + y*z + z*x = 1) ∧
  (∃ (a b c : ℝ), a^2 + b^2 + c^2 = 1 ∧ x*y + y*z + z*x = -1/2) :=
sorry

end NUMINAMATH_CALUDE_xyz_sum_bounds_l1158_115891


namespace NUMINAMATH_CALUDE_mixed_box_weight_l1158_115868

/-- The weight of a box with 100 aluminum balls -/
def weight_aluminum : ℝ := 510

/-- The weight of a box with 100 plastic balls -/
def weight_plastic : ℝ := 490

/-- The number of aluminum balls in the mixed box -/
def num_aluminum : ℕ := 20

/-- The number of plastic balls in the mixed box -/
def num_plastic : ℕ := 80

/-- The total number of balls in each box -/
def total_balls : ℕ := 100

theorem mixed_box_weight : 
  (num_aluminum : ℝ) / total_balls * weight_aluminum + 
  (num_plastic : ℝ) / total_balls * weight_plastic = 494 := by
  sorry

end NUMINAMATH_CALUDE_mixed_box_weight_l1158_115868


namespace NUMINAMATH_CALUDE_point_coordinates_l1158_115851

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem stating the coordinates of point P given the conditions -/
theorem point_coordinates (p : Point) 
  (h1 : isInSecondQuadrant p)
  (h2 : distanceToXAxis p = 4)
  (h3 : distanceToYAxis p = 5) :
  p = Point.mk (-5) 4 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1158_115851


namespace NUMINAMATH_CALUDE_optimal_well_position_l1158_115835

open Real

/-- Represents the positions of 6 houses along a road -/
structure HousePositions where
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  x₄ : ℝ
  x₅ : ℝ
  x₆ : ℝ
  h₁₂ : x₁ < x₂
  h₂₃ : x₂ < x₃
  h₃₄ : x₃ < x₄
  h₄₅ : x₄ < x₅
  h₅₆ : x₅ < x₆

/-- The sum of absolute distances from a point x to all house positions -/
def sumOfDistances (hp : HousePositions) (x : ℝ) : ℝ :=
  |x - hp.x₁| + |x - hp.x₂| + |x - hp.x₃| + |x - hp.x₄| + |x - hp.x₅| + |x - hp.x₆|

/-- The theorem stating that the optimal well position is the average of x₃ and x₄ -/
theorem optimal_well_position (hp : HousePositions) :
  ∃ (x : ℝ), ∀ (y : ℝ), sumOfDistances hp x ≤ sumOfDistances hp y ∧ x = (hp.x₃ + hp.x₄) / 2 :=
sorry

end NUMINAMATH_CALUDE_optimal_well_position_l1158_115835


namespace NUMINAMATH_CALUDE_specific_polyhedron_space_diagonals_l1158_115814

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- The number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  (Q.vertices.choose 2) - Q.edges - 2 * Q.quadrilateral_faces

/-- Theorem: A convex polyhedron Q with 30 vertices, 72 edges, 44 faces
    (of which 30 are triangular and 14 are quadrilateral) has 335 space diagonals -/
theorem specific_polyhedron_space_diagonals :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 44,
    triangular_faces := 30,
    quadrilateral_faces := 14
  }
  space_diagonals Q = 335 := by sorry

end NUMINAMATH_CALUDE_specific_polyhedron_space_diagonals_l1158_115814


namespace NUMINAMATH_CALUDE_remaining_speed_calculation_l1158_115821

/-- Calculates the average speed for the remaining part of a trip given:
    - The fraction of the trip completed in the first part
    - The speed of the first part of the trip
    - The average speed for the entire trip
-/
theorem remaining_speed_calculation 
  (first_part_fraction : Real) 
  (first_part_speed : Real) 
  (total_average_speed : Real) :
  first_part_fraction = 0.4 →
  first_part_speed = 40 →
  total_average_speed = 50 →
  (1 - first_part_fraction) * total_average_speed / 
    (1 - first_part_fraction * total_average_speed / first_part_speed) = 60 := by
  sorry

#check remaining_speed_calculation

end NUMINAMATH_CALUDE_remaining_speed_calculation_l1158_115821


namespace NUMINAMATH_CALUDE_min_value_of_x_l1158_115802

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ Real.log 3 + (2/3) * Real.log x) : x ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_x_l1158_115802


namespace NUMINAMATH_CALUDE_toothpicks_per_card_l1158_115815

theorem toothpicks_per_card 
  (total_cards : ℕ) 
  (unused_cards : ℕ) 
  (toothpick_boxes : ℕ) 
  (toothpicks_per_box : ℕ) 
  (h1 : total_cards = 52) 
  (h2 : unused_cards = 16) 
  (h3 : toothpick_boxes = 6) 
  (h4 : toothpicks_per_box = 450) :
  (toothpick_boxes * toothpicks_per_box) / (total_cards - unused_cards) = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_toothpicks_per_card_l1158_115815


namespace NUMINAMATH_CALUDE_polynomial_root_problem_l1158_115880

theorem polynomial_root_problem (a b : ℝ) : 
  (∀ x : ℝ, a*x^4 + (a + b)*x^3 + (b - 2*a)*x^2 + 5*b*x + (12 - a) = 0 ↔ 
    x = 1 ∨ x = -3 ∨ x = 4 ∨ x = -92/297) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_root_problem_l1158_115880


namespace NUMINAMATH_CALUDE_perfume_price_increase_l1158_115887

theorem perfume_price_increase (x : ℝ) : 
  let original_price : ℝ := 1200
  let increased_price : ℝ := original_price * (1 + x / 100)
  let final_price : ℝ := increased_price * (1 - 15 / 100)
  final_price = original_price - 78 → x = 10 :=
by sorry

end NUMINAMATH_CALUDE_perfume_price_increase_l1158_115887


namespace NUMINAMATH_CALUDE_chim_tu_survival_days_l1158_115843

/-- The number of distinct T-shirts --/
def n : ℕ := 4

/-- The number of days between outfit changes --/
def days_per_outfit : ℕ := 3

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The factorial of a natural number --/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of distinct outfits with exactly k T-shirts --/
def outfits_with_k (k : ℕ) : ℕ := choose n k * factorial k

/-- The total number of distinct outfits --/
def total_outfits : ℕ := outfits_with_k 3 + outfits_with_k 4

/-- The number of days Chim Tu can wear a unique outfit --/
def survival_days : ℕ := total_outfits * days_per_outfit

theorem chim_tu_survival_days : survival_days = 144 := by
  sorry

end NUMINAMATH_CALUDE_chim_tu_survival_days_l1158_115843


namespace NUMINAMATH_CALUDE_statement_consistency_l1158_115882

def Statement : Type := Bool

def statementA (a b c d e : Statement) : Prop :=
  (a = true ∨ b = true ∨ c = true ∨ d = true ∨ e = true) ∧
  ¬(a = true ∧ b = true) ∧ ¬(a = true ∧ c = true) ∧ ¬(a = true ∧ d = true) ∧ ¬(a = true ∧ e = true) ∧
  ¬(b = true ∧ c = true) ∧ ¬(b = true ∧ d = true) ∧ ¬(b = true ∧ e = true) ∧
  ¬(c = true ∧ d = true) ∧ ¬(c = true ∧ e = true) ∧
  ¬(d = true ∧ e = true)

def statementC (a b c d e : Statement) : Prop :=
  a = true ∧ b = true ∧ c = true ∧ d = true ∧ e = true

def statementE (a : Statement) : Prop :=
  a = true

theorem statement_consistency :
  ∀ (a b c d e : Statement),
  (statementA a b c d e ↔ a = true) →
  (statementC a b c d e ↔ c = true) →
  (statementE a ↔ e = true) →
  (a = false ∧ b = true ∧ c = false ∧ d = true ∧ e = false) :=
by sorry

end NUMINAMATH_CALUDE_statement_consistency_l1158_115882


namespace NUMINAMATH_CALUDE_problem_statement_l1158_115826

/-- Given m > 0, p, and q as defined, prove the conditions for m and x. -/
theorem problem_statement (m : ℝ) (h_m : m > 0) : 
  -- Define p
  let p := fun x : ℝ => (x + 1) * (x - 5) ≤ 0
  -- Define q
  let q := fun x : ℝ => 1 - m ≤ x ∧ x ≤ 1 + m
  -- Part 1: When p is a sufficient condition for q, m ≥ 4
  ((∀ x : ℝ, p x → q x) → m ≥ 4) ∧
  -- Part 2: When m = 5 and (p or q) is true but (p and q) is false, 
  --         x is in the specified range
  (m = 5 → 
    ∀ x : ℝ, ((p x ∨ q x) ∧ ¬(p x ∧ q x)) → 
      ((-4 ≤ x ∧ x < -1) ∨ (5 < x ∧ x < 6))) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1158_115826


namespace NUMINAMATH_CALUDE_f_inequality_l1158_115830

open Real

-- Define the function f and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State the theorem
theorem f_inequality (hf : ∀ x, x ∈ Set.Ici 0 → (x + 1) * f x + x * f' x ≥ 0)
  (hf_not_const : ¬∀ x y, f x = f y) :
  f 1 < 2 * ℯ * f 2 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1158_115830


namespace NUMINAMATH_CALUDE_koolaid_mixture_l1158_115800

theorem koolaid_mixture (W : ℝ) : 
  W > 4 →
  (2 : ℝ) / (2 + 4 * (W - 4)) = 0.04 →
  W = 16 := by
sorry

end NUMINAMATH_CALUDE_koolaid_mixture_l1158_115800


namespace NUMINAMATH_CALUDE_fraction_division_and_addition_l1158_115848

theorem fraction_division_and_addition :
  (5 : ℚ) / 6 / (9 : ℚ) / 10 + 1 / 15 = 402 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_and_addition_l1158_115848


namespace NUMINAMATH_CALUDE_peanuts_problem_l1158_115893

/-- The number of peanuts remaining in the jar after a series of distributions and consumptions -/
def peanuts_remaining (initial : ℕ) : ℕ :=
  let brock_ate := initial / 3
  let after_brock := initial - brock_ate
  let per_family := after_brock / 3
  let bonita_per_family := (2 * per_family) / 5
  let after_bonita_per_family := per_family - bonita_per_family
  let after_bonita_total := after_bonita_per_family * 3
  let carlos_ate := after_bonita_total / 5
  after_bonita_total - carlos_ate

/-- Theorem stating that given the initial conditions, 216 peanuts remain in the jar -/
theorem peanuts_problem : peanuts_remaining 675 = 216 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_problem_l1158_115893


namespace NUMINAMATH_CALUDE_vector_to_line_parallel_l1158_115897

/-- A line parameterized by t -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Check if a point lies on a parameterized line -/
def pointOnLine (p : ℝ × ℝ) (l : ParametricLine) : Prop :=
  ∃ t : ℝ, l.x t = p.1 ∧ l.y t = p.2

/-- Check if two vectors are parallel -/
def vectorsParallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_to_line_parallel (l : ParametricLine) (v : ℝ × ℝ) :
  l.x t = 5 * t + 3 ∧ l.y t = 2 * t - 1 →
  pointOnLine v l ∧ vectorsParallel v (5, 2) →
  v = (-2.5, -1) :=
sorry

end NUMINAMATH_CALUDE_vector_to_line_parallel_l1158_115897


namespace NUMINAMATH_CALUDE_barbaras_candy_count_l1158_115862

/-- Given Barbara's initial candy count and the number of candies she bought,
    prove that her total candy count is the sum of these two quantities. -/
theorem barbaras_candy_count (initial_candies bought_candies : ℕ) :
  initial_candies = 9 →
  bought_candies = 18 →
  initial_candies + bought_candies = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_barbaras_candy_count_l1158_115862


namespace NUMINAMATH_CALUDE_messages_cleared_in_seven_days_l1158_115803

/-- Given the initial number of unread messages, messages read per day,
    and new messages received per day, calculate the number of days
    required to read all unread messages. -/
def days_to_read_messages (initial_messages : ℕ) (messages_read_per_day : ℕ) (new_messages_per_day : ℕ) : ℕ :=
  initial_messages / (messages_read_per_day - new_messages_per_day)

/-- Theorem stating that it takes 7 days to read all unread messages
    under the given conditions. -/
theorem messages_cleared_in_seven_days :
  days_to_read_messages 98 20 6 = 7 := by
  sorry

#eval days_to_read_messages 98 20 6

end NUMINAMATH_CALUDE_messages_cleared_in_seven_days_l1158_115803


namespace NUMINAMATH_CALUDE_perfect_square_fraction_solutions_l1158_115836

theorem perfect_square_fraction_solutions :
  ∀ m n p : ℕ+,
  p.val.Prime →
  (∃ k : ℕ+, ((5^(m.val) + 2^(n.val) * p.val) : ℚ) / (5^(m.val) - 2^(n.val) * p.val) = (k.val : ℚ)^2) →
  ((m = 1 ∧ n = 1 ∧ p = 2) ∨ (m = 2 ∧ n = 3 ∧ p = 3) ∨ (m = 2 ∧ n = 2 ∧ p = 5)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_fraction_solutions_l1158_115836


namespace NUMINAMATH_CALUDE_z_properties_l1158_115818

def z : ℂ := -(2 * Complex.I + 6) * Complex.I

theorem z_properties : 
  (z.re > 0 ∧ z.im < 0) ∧ 
  ∃ (y : ℝ), z - 2 = y * Complex.I :=
sorry

end NUMINAMATH_CALUDE_z_properties_l1158_115818


namespace NUMINAMATH_CALUDE_flower_bed_distance_l1158_115858

/-- The perimeter of a rectangle -/
def rectangle_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- The total distance walked around a rectangle multiple times -/
def total_distance (length width : ℝ) (times : ℕ) : ℝ :=
  (rectangle_perimeter length width) * times

theorem flower_bed_distance :
  total_distance 5 3 3 = 30 := by sorry

end NUMINAMATH_CALUDE_flower_bed_distance_l1158_115858


namespace NUMINAMATH_CALUDE_zongzi_sales_theorem_l1158_115859

/-- Represents the sales and profit model for zongzi boxes -/
structure ZongziSales where
  cost : ℝ             -- Cost per box
  min_price : ℝ        -- Minimum selling price
  base_sales : ℝ       -- Base sales at minimum price
  price_sensitivity : ℝ -- Decrease in sales per unit price increase
  max_price : ℝ        -- Maximum allowed selling price
  min_profit : ℝ       -- Minimum desired daily profit

/-- The main theorem about zongzi sales and profit -/
theorem zongzi_sales_theorem (z : ZongziSales)
  (h_cost : z.cost = 40)
  (h_min_price : z.min_price = 45)
  (h_base_sales : z.base_sales = 700)
  (h_price_sensitivity : z.price_sensitivity = 20)
  (h_max_price : z.max_price = 58)
  (h_min_profit : z.min_profit = 6000) :
  (∃ (sales_eq : ℝ → ℝ),
    (∀ x, sales_eq x = -20 * x + 1600) ∧
    (∃ (optimal_price : ℝ) (max_profit : ℝ),
      optimal_price = 60 ∧
      max_profit = 8000 ∧
      (∀ p, z.min_price ≤ p → p ≤ z.max_price →
        (p - z.cost) * (sales_eq p) ≤ max_profit)) ∧
    (∃ (min_boxes : ℝ),
      min_boxes = 440 ∧
      (z.max_price - z.cost) * min_boxes ≥ z.min_profit)) :=
sorry

end NUMINAMATH_CALUDE_zongzi_sales_theorem_l1158_115859


namespace NUMINAMATH_CALUDE_complex_number_coordinates_l1158_115864

/-- Given a complex number z = (1 + 2i^3) / (2 + i), prove that its coordinates in the complex plane are (0, -1) -/
theorem complex_number_coordinates :
  let i : ℂ := Complex.I
  let z : ℂ := (1 + 2 * i^3) / (2 + i)
  z.re = 0 ∧ z.im = -1 := by sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_l1158_115864


namespace NUMINAMATH_CALUDE_H2O_weight_not_72_l1158_115834

/-- The molecular weight of H2O in g/mol -/
def molecular_weight_H2O : ℝ := 18.016

/-- The given incorrect molecular weight in g/mol -/
def given_weight : ℝ := 72

/-- Theorem stating that the molecular weight of H2O is not equal to the given weight -/
theorem H2O_weight_not_72 : molecular_weight_H2O ≠ given_weight := by
  sorry

end NUMINAMATH_CALUDE_H2O_weight_not_72_l1158_115834


namespace NUMINAMATH_CALUDE_trigonometric_product_l1158_115895

theorem trigonometric_product (α : Real) (h : Real.tan α = -2) : 
  Real.sin (π/2 + α) * Real.cos (π + α) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_l1158_115895


namespace NUMINAMATH_CALUDE_square_of_binomial_p_l1158_115857

/-- If 9x^2 + 24x + p is the square of a binomial, then p = 16 -/
theorem square_of_binomial_p (p : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9*x^2 + 24*x + p = (a*x + b)^2) → p = 16 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_p_l1158_115857


namespace NUMINAMATH_CALUDE_souvenir_price_increase_l1158_115805

theorem souvenir_price_increase (original_price final_price : ℝ) 
  (h1 : original_price = 76.8)
  (h2 : final_price = 120)
  (h3 : ∃ x : ℝ, original_price * (1 + x)^2 = final_price) :
  ∃ x : ℝ, original_price * (1 + x)^2 = final_price ∧ x = 0.25 := by
sorry

end NUMINAMATH_CALUDE_souvenir_price_increase_l1158_115805


namespace NUMINAMATH_CALUDE_max_sequence_length_l1158_115819

theorem max_sequence_length (x : ℕ → ℕ) (n : ℕ) : 
  (∀ k, k < n - 1 → x k < x (k + 1)) →
  (∀ k, k ≤ n - 2 → x k ∣ x (k + 2)) →
  x n = 1000 →
  n ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_max_sequence_length_l1158_115819


namespace NUMINAMATH_CALUDE_college_graduates_scientific_notation_l1158_115884

theorem college_graduates_scientific_notation :
  ∃ (x : ℝ) (n : ℤ), 
    x ≥ 1 ∧ x < 10 ∧ 
    116000000 = x * (10 : ℝ) ^ n ∧
    x = 1.16 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_college_graduates_scientific_notation_l1158_115884


namespace NUMINAMATH_CALUDE_page_lines_increase_l1158_115879

theorem page_lines_increase (L : ℕ) (h1 : (60 : ℝ) / L = 1 / 3) : L + 60 = 240 := by
  sorry

end NUMINAMATH_CALUDE_page_lines_increase_l1158_115879


namespace NUMINAMATH_CALUDE_expected_left_handed_students_l1158_115806

theorem expected_left_handed_students
  (total_students : ℕ)
  (left_handed_proportion : ℚ)
  (h1 : total_students = 32)
  (h2 : left_handed_proportion = 3 / 8) :
  ↑total_students * left_handed_proportion = 12 :=
by sorry

end NUMINAMATH_CALUDE_expected_left_handed_students_l1158_115806


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l1158_115865

theorem negative_fraction_comparison : -3/4 > -6/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l1158_115865


namespace NUMINAMATH_CALUDE_unique_polynomial_satisfying_conditions_l1158_115824

/-- A polynomial function of degree at most 3 -/
def PolynomialDegree3 (g : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, g x = a * x^3 + b * x^2 + c * x + d

/-- The conditions that g must satisfy -/
def SatisfiesConditions (g : ℝ → ℝ) : Prop :=
  (∀ x, g (x^2) = (g x)^2) ∧ 
  (∀ x, g (x^2) = g (g x)) ∧ 
  g 1 = 1

theorem unique_polynomial_satisfying_conditions :
  ∃! g : ℝ → ℝ, PolynomialDegree3 g ∧ SatisfiesConditions g ∧ (∀ x, g x = x^2) := by
  sorry

end NUMINAMATH_CALUDE_unique_polynomial_satisfying_conditions_l1158_115824


namespace NUMINAMATH_CALUDE_function_properties_l1158_115817

theorem function_properties :
  (∃ x : ℝ, (10 : ℝ) ^ x = x) ∧
  (∃ x : ℝ, (10 : ℝ) ^ x = x ^ 2) ∧
  (¬ ∀ x : ℝ, (10 : ℝ) ^ x > x) ∧
  (¬ ∀ x : ℝ, x > 0 → (10 : ℝ) ^ x > x ^ 2) ∧
  (¬ ∃ x y : ℝ, x ≠ y ∧ (10 : ℝ) ^ x = -x ∧ (10 : ℝ) ^ y = -y) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1158_115817


namespace NUMINAMATH_CALUDE_opposite_of_neg_six_l1158_115888

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℤ) : ℤ := -a

/-- Theorem: The opposite of -6 is 6. -/
theorem opposite_of_neg_six : opposite (-6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_six_l1158_115888


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1158_115822

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x => 2*x^2 - 3*x - (1 - 2*x)
  (f 1 = 0) ∧ (f (-1/2) = 0) ∧ 
  (∀ x : ℝ, f x = 0 → x = 1 ∨ x = -1/2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1158_115822


namespace NUMINAMATH_CALUDE_scientific_notation_of_number_l1158_115890

def number : ℝ := 308000000

theorem scientific_notation_of_number :
  ∃ (a : ℝ) (n : ℤ), number = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.08 ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_number_l1158_115890


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1158_115894

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 2*x

-- Define the point of tangency
def x₀ : ℝ := 2

-- Define the slope of the tangent line
def m : ℝ := 3 * x₀^2 - 2

-- Define the y-intercept of the tangent line
def b : ℝ := f x₀ - m * x₀

-- Theorem statement
theorem tangent_line_equation :
  ∀ x y : ℝ, y = m * x + b ↔ y - f x₀ = m * (x - x₀) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1158_115894


namespace NUMINAMATH_CALUDE_unknown_number_solution_l1158_115869

theorem unknown_number_solution : 
  ∃ x : ℝ, (4.7 * 13.26 + 4.7 * x + 4.7 * 77.31 = 470) ∧ (abs (x - 9.43) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_solution_l1158_115869


namespace NUMINAMATH_CALUDE_purple_valley_skirts_l1158_115898

/-- The number of skirts in Azure Valley -/
def azure_skirts : ℕ := 60

/-- The number of skirts in Seafoam Valley -/
def seafoam_skirts : ℕ := (2 * azure_skirts) / 3

/-- The number of skirts in Purple Valley -/
def purple_skirts : ℕ := seafoam_skirts / 4

/-- Theorem stating that Purple Valley has 10 skirts -/
theorem purple_valley_skirts : purple_skirts = 10 := by
  sorry

end NUMINAMATH_CALUDE_purple_valley_skirts_l1158_115898


namespace NUMINAMATH_CALUDE_square_area_increase_l1158_115849

theorem square_area_increase (s : ℝ) (h1 : s^2 = 256) (h2 : s > 0) :
  (s + 2)^2 - s^2 = 68 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l1158_115849


namespace NUMINAMATH_CALUDE_expand_expression_l1158_115840

theorem expand_expression (x : ℝ) : (7 * x - 3) * (3 * x^2) = 21 * x^3 - 9 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1158_115840


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l1158_115832

def m : Fin 2 → ℝ := ![(-1), 2]
def n (b : ℝ) : Fin 2 → ℝ := ![2, b]

theorem vector_difference_magnitude (b : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ m = k • n b) →
  ‖m - n b‖ = 3 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l1158_115832


namespace NUMINAMATH_CALUDE_sum_of_divisors_143_l1158_115870

theorem sum_of_divisors_143 : (Finset.filter (· ∣ 143) (Finset.range 144)).sum id = 168 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_143_l1158_115870


namespace NUMINAMATH_CALUDE_permutations_of_three_objects_l1158_115853

theorem permutations_of_three_objects (n : ℕ) (h : n = 3) : Nat.factorial n = 6 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_three_objects_l1158_115853


namespace NUMINAMATH_CALUDE_f_47_mod_17_l1158_115804

def f (n : ℕ) : ℕ := 3^n + 7^n

theorem f_47_mod_17 : f 47 % 17 = 10 := by
  sorry

end NUMINAMATH_CALUDE_f_47_mod_17_l1158_115804


namespace NUMINAMATH_CALUDE_always_two_real_roots_root_geq_7_implies_k_leq_neg_5_l1158_115850

-- Define the quadratic equation
def quadratic_equation (x k : ℝ) : ℝ := x^2 + (k+1)*x + 3*k - 6

-- Theorem 1: The quadratic equation always has two real roots
theorem always_two_real_roots (k : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ k = 0 ∧ quadratic_equation x₂ k = 0 :=
sorry

-- Theorem 2: If one root is not less than 7, then k ≤ -5
theorem root_geq_7_implies_k_leq_neg_5 (k : ℝ) :
  (∃ x : ℝ, quadratic_equation x k = 0 ∧ x ≥ 7) → k ≤ -5 :=
sorry

end NUMINAMATH_CALUDE_always_two_real_roots_root_geq_7_implies_k_leq_neg_5_l1158_115850


namespace NUMINAMATH_CALUDE_star_properties_l1158_115829

-- Define the * operation
def star (x y : ℝ) : ℝ := x - y

-- State the theorem
theorem star_properties :
  (∀ x : ℝ, star x x = 0) ∧
  (∀ x y z : ℝ, star x (star y z) = star x y + z) ∧
  (star 1993 1935 = 58) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l1158_115829


namespace NUMINAMATH_CALUDE_container_volume_transformation_l1158_115892

/-- A cuboid container with volume measured in gallons -/
structure Container where
  height : ℝ
  length : ℝ
  width : ℝ
  volume : ℝ
  volume_eq : volume = height * length * width

/-- Theorem stating that if a container with 3 gallon volume has its height doubled and length tripled, its new volume will be 18 gallons -/
theorem container_volume_transformation (c : Container) 
  (h_volume : c.volume = 3) :
  let new_container := Container.mk 
    (2 * c.height) 
    (3 * c.length) 
    c.width 
    ((2 * c.height) * (3 * c.length) * c.width)
    (by simp)
  new_container.volume = 18 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_transformation_l1158_115892


namespace NUMINAMATH_CALUDE_multiples_of_six_or_eight_l1158_115883

def count_multiples (n : ℕ) (m : ℕ) : ℕ :=
  (n - 1) / m

theorem multiples_of_six_or_eight (upper_bound : ℕ) (h : upper_bound = 151) : 
  (count_multiples upper_bound 6 + count_multiples upper_bound 8 - 2 * count_multiples upper_bound 24) = 31 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_six_or_eight_l1158_115883


namespace NUMINAMATH_CALUDE_apple_quantity_proof_l1158_115833

/-- Calculates the final quantity of apples given initial quantity, sold quantity, and purchased quantity. -/
def final_quantity (initial : ℕ) (sold : ℕ) (purchased : ℕ) : ℕ :=
  initial - sold + purchased

/-- Theorem stating that given the specific quantities in the problem, the final quantity is 293 kg. -/
theorem apple_quantity_proof :
  final_quantity 280 132 145 = 293 := by
  sorry

end NUMINAMATH_CALUDE_apple_quantity_proof_l1158_115833


namespace NUMINAMATH_CALUDE_total_peaches_is_450_l1158_115845

/-- Represents the number of baskets in the fruit shop -/
def num_baskets : ℕ := 15

/-- Represents the initial number of red peaches in each basket -/
def initial_red : ℕ := 19

/-- Represents the initial number of green peaches in each basket -/
def initial_green : ℕ := 4

/-- Represents the number of moldy peaches in each basket -/
def moldy : ℕ := 6

/-- Represents the number of red peaches removed from each basket -/
def removed_red : ℕ := 3

/-- Represents the number of green peaches removed from each basket -/
def removed_green : ℕ := 1

/-- Represents the number of freshly harvested peaches added to each basket -/
def added_fresh : ℕ := 5

/-- Calculates the total number of peaches in all baskets after adjustments -/
def total_peaches_after_adjustment : ℕ :=
  num_baskets * ((initial_red - removed_red) + (initial_green - removed_green) + moldy + added_fresh)

/-- Theorem stating that the total number of peaches after adjustments is 450 -/
theorem total_peaches_is_450 : total_peaches_after_adjustment = 450 := by
  sorry

end NUMINAMATH_CALUDE_total_peaches_is_450_l1158_115845


namespace NUMINAMATH_CALUDE_yarn_length_proof_l1158_115831

theorem yarn_length_proof (green_length red_length total_length : ℕ) : 
  green_length = 156 ∧ 
  red_length = 3 * green_length + 8 →
  total_length = green_length + red_length →
  total_length = 632 := by
sorry

end NUMINAMATH_CALUDE_yarn_length_proof_l1158_115831


namespace NUMINAMATH_CALUDE_fourth_number_in_expression_l1158_115812

theorem fourth_number_in_expression (x : ℝ) : 
  0.3 * 0.8 + 0.1 * x = 0.29 → x = 0.5 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_in_expression_l1158_115812


namespace NUMINAMATH_CALUDE_game_lives_theorem_l1158_115811

/-- Given a game with initial players, players who quit, and total remaining lives,
    calculate the number of lives per remaining player. -/
def lives_per_player (initial_players : ℕ) (players_quit : ℕ) (total_lives : ℕ) : ℕ :=
  total_lives / (initial_players - players_quit)

/-- Theorem: In a game with 16 initial players, 7 players quitting, and 72 total remaining lives,
    each remaining player has 8 lives. -/
theorem game_lives_theorem :
  lives_per_player 16 7 72 = 8 := by
  sorry

end NUMINAMATH_CALUDE_game_lives_theorem_l1158_115811


namespace NUMINAMATH_CALUDE_present_age_of_B_l1158_115863

/-- Given three people A, B, and C, whose ages satisfy certain conditions,
    prove that the present age of B is 30 years. -/
theorem present_age_of_B (A B C : ℕ) : 
  A + B + C = 90 →  -- Total present age is 90
  (A - 10) = 1 * x ∧ (B - 10) = 2 * x ∧ (C - 10) = 3 * x →  -- Age ratio 10 years ago
  B = 30 := by
sorry


end NUMINAMATH_CALUDE_present_age_of_B_l1158_115863


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1158_115876

/-- Given a triangle ABC with an arbitrary point inside it, and three lines drawn through
    this point parallel to the sides of the triangle, dividing it into six parts including
    three triangles with areas S₁, S₂, and S₃, the area of triangle ABC is (√S₁ + √S₂ + √S₃)². -/
theorem triangle_area_theorem (S₁ S₂ S₃ : ℝ) (h₁ : 0 < S₁) (h₂ : 0 < S₂) (h₃ : 0 < S₃) :
  ∃ (S : ℝ), S > 0 ∧ S = (Real.sqrt S₁ + Real.sqrt S₂ + Real.sqrt S₃)^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1158_115876


namespace NUMINAMATH_CALUDE_problem_statement_l1158_115808

noncomputable def f (x : ℝ) : ℝ := (x / (x + 4)) * Real.exp (x + 2)

noncomputable def g (a x : ℝ) : ℝ := (Real.exp (x + 2) - a * x - 3 * a) / ((x + 2)^2)

theorem problem_statement :
  (∀ x > -2, x * Real.exp (x + 2) + x + 4 > 0) ∧
  (∀ a ∈ Set.Icc 0 1, ∃ min_x > -2, ∀ x > -2, g a min_x ≤ g a x) ∧
  (∃ h : ℝ → ℝ, (∀ a ∈ Set.Icc 0 1, ∃ min_x > -2, h a = g a min_x) ∧
    Set.range h = Set.Ioo (1/2) (Real.exp 2 / 4)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1158_115808


namespace NUMINAMATH_CALUDE_number_of_cats_l1158_115886

/-- Represents the number of cats on the ship. -/
def cats : ℕ := sorry

/-- Represents the number of sailors on the ship. -/
def sailors : ℕ := sorry

/-- Represents the number of cooks on the ship. -/
def cooks : ℕ := 1

/-- Represents the number of captains on the ship. -/
def captains : ℕ := 1

/-- The total number of heads on the ship. -/
def total_heads : ℕ := 16

/-- The total number of legs on the ship. -/
def total_legs : ℕ := 41

/-- Theorem stating that the number of cats on the ship is 5. -/
theorem number_of_cats : cats = 5 := by
  have head_count : cats + sailors + cooks + captains = total_heads := sorry
  have leg_count : 4 * cats + 2 * sailors + 2 * cooks + captains = total_legs := sorry
  sorry

end NUMINAMATH_CALUDE_number_of_cats_l1158_115886


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_and_3_l1158_115846

theorem smallest_perfect_square_divisible_by_2_and_3 :
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m^2) ∧ 2 ∣ n ∧ 3 ∣ n ∧
  ∀ k : ℕ, k > 0 → (∃ l : ℕ, k = l^2) → 2 ∣ k → 3 ∣ k → n ≤ k :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_and_3_l1158_115846


namespace NUMINAMATH_CALUDE_inequality_proof_l1158_115842

theorem inequality_proof (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x^2 + y^2 + 1 = (x*y - 1)^2) : 
  (x + y ≥ 4) ∧ (x^2 + y^2 ≥ 8) ∧ (x + 4*y ≥ 9) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1158_115842


namespace NUMINAMATH_CALUDE_vectors_collinear_l1158_115847

def a : ℝ × ℝ × ℝ := (-1, 2, 8)
def b : ℝ × ℝ × ℝ := (3, 7, -1)
def c₁ : ℝ × ℝ × ℝ := (4 * a.1 - 3 * b.1, 4 * a.2.1 - 3 * b.2.1, 4 * a.2.2 - 3 * b.2.2)
def c₂ : ℝ × ℝ × ℝ := (9 * b.1 - 12 * a.1, 9 * b.2.1 - 12 * a.2.1, 9 * b.2.2 - 12 * a.2.2)

theorem vectors_collinear : ∃ (k : ℝ), c₁ = (k * c₂.1, k * c₂.2.1, k * c₂.2.2) := by
  sorry

end NUMINAMATH_CALUDE_vectors_collinear_l1158_115847


namespace NUMINAMATH_CALUDE_relationship_abc_l1158_115881

theorem relationship_abc (a b c : ℝ) : 
  a = (2/5)^(2/5) → 
  b = (3/5)^(2/5) → 
  c = Real.log (2/5) / Real.log (3/5) → 
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l1158_115881
