import Mathlib

namespace NUMINAMATH_CALUDE_darias_remaining_debt_l636_63650

def savings : ℕ := 500
def couch_price : ℕ := 750
def table_price : ℕ := 100
def lamp_price : ℕ := 50

theorem darias_remaining_debt : 
  (couch_price + table_price + lamp_price) - savings = 400 := by
  sorry

end NUMINAMATH_CALUDE_darias_remaining_debt_l636_63650


namespace NUMINAMATH_CALUDE_count_pairs_eq_five_l636_63649

/-- The number of pairs of natural numbers (a, b) satisfying the given conditions -/
def count_pairs : ℕ := 5

/-- Predicate to check if a pair of natural numbers satisfies the equation -/
def satisfies_equation (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 6

/-- The main theorem stating that there are exactly 5 pairs satisfying the conditions -/
theorem count_pairs_eq_five :
  (∃! (s : Finset (ℕ × ℕ)), s.card = count_pairs ∧ 
    (∀ (p : ℕ × ℕ), p ∈ s ↔ 
      (p.1 ≥ p.2 ∧ satisfies_equation p.1 p.2))) :=
by sorry

end NUMINAMATH_CALUDE_count_pairs_eq_five_l636_63649


namespace NUMINAMATH_CALUDE_pink_yards_calculation_l636_63662

/-- The total number of yards dyed for the order -/
def total_yards : ℕ := 111421

/-- The number of yards dyed green -/
def green_yards : ℕ := 61921

/-- The number of yards dyed pink -/
def pink_yards : ℕ := total_yards - green_yards

theorem pink_yards_calculation : pink_yards = 49500 := by
  sorry

end NUMINAMATH_CALUDE_pink_yards_calculation_l636_63662


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l636_63631

theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 - 1)
  (z.re = 0 ∧ z.im ≠ 0) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l636_63631


namespace NUMINAMATH_CALUDE_stones_combine_l636_63686

/-- Definition of similar sizes -/
def similar (a b : ℕ) : Prop := a ≤ b ∧ b ≤ 2 * a

/-- A step in the combining process -/
inductive CombineStep (n : ℕ)
  | combine (i j : Fin n) (h : i.val < j.val) (hsimilar : similar (Fin.val i) (Fin.val j)) : CombineStep n

/-- A sequence of combining steps -/
def CombineSequence (n : ℕ) := List (CombineStep n)

/-- The final state after combining -/
def FinalState (n : ℕ) : Prop := ∃ (seq : CombineSequence n), 
  (∀ i : Fin n, i.val = 1) → (∃ j : Fin n, j.val = n ∧ ∀ k : Fin n, k ≠ j → k.val = 0)

/-- The main theorem -/
theorem stones_combine (n : ℕ) : FinalState n := by sorry

end NUMINAMATH_CALUDE_stones_combine_l636_63686


namespace NUMINAMATH_CALUDE_parabola_point_order_l636_63694

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop := y = (x - 1)^2 - 2

theorem parabola_point_order (a b c d : ℝ) :
  parabola a 2 →
  parabola b 6 →
  parabola c d →
  d < 1 →
  a < 0 →
  b > 0 →
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_order_l636_63694


namespace NUMINAMATH_CALUDE_tangent_line_equation_l636_63678

-- Define the function f
def f (x : ℝ) : ℝ := 8 * x^3 - 6 * x

-- State the theorem
theorem tangent_line_equation :
  (∀ x, f (x / 2) = x^3 - 3 * x) →
  ∃ m b, m * 1 - f 1 + b = 0 ∧
         ∀ x, m * x - f x + b = 0 → x = 1 ∧
         m = 18 ∧ b = -16 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l636_63678


namespace NUMINAMATH_CALUDE_sad_children_count_l636_63695

theorem sad_children_count (total : ℕ) (happy : ℕ) (neither : ℕ) 
  (boys : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) (neither_boys : ℕ)
  (h1 : total = 60)
  (h2 : happy = 30)
  (h3 : neither = 20)
  (h4 : boys = 19)
  (h5 : girls = 41)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neither_boys = 7)
  (h9 : total = happy + neither + (total - happy - neither)) :
  total - happy - neither = 10 := by
sorry

end NUMINAMATH_CALUDE_sad_children_count_l636_63695


namespace NUMINAMATH_CALUDE_odd_sum_floor_power_l636_63611

theorem odd_sum_floor_power (n : ℕ+) : 
  Odd (n + ⌊(Real.sqrt 2 + 1)^(n : ℝ)⌋) := by sorry

end NUMINAMATH_CALUDE_odd_sum_floor_power_l636_63611


namespace NUMINAMATH_CALUDE_shifted_roots_l636_63635

variable (x : ℝ)

-- Define the original polynomial
def original_poly (x : ℝ) : ℝ := x^3 - 5*x + 7

-- Define the roots a, b, c of the original polynomial
axiom roots_exist : ∃ a b c : ℝ, original_poly a = 0 ∧ original_poly b = 0 ∧ original_poly c = 0

-- Define the shifted polynomial
def shifted_poly (x : ℝ) : ℝ := x^3 + 6*x^2 + 7*x + 5

theorem shifted_roots (a b c : ℝ) : 
  original_poly a = 0 → original_poly b = 0 → original_poly c = 0 →
  shifted_poly (a - 2) = 0 ∧ shifted_poly (b - 2) = 0 ∧ shifted_poly (c - 2) = 0 :=
sorry

end NUMINAMATH_CALUDE_shifted_roots_l636_63635


namespace NUMINAMATH_CALUDE_intersection_area_theorem_l636_63637

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  center : Point
  width : ℝ
  height : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a right triangle in 2D space -/
structure RightTriangle where
  vertex : Point
  leg1 : ℝ
  leg2 : ℝ

/-- Calculate the area of intersection between a rectangle, circle, and right triangle -/
def areaOfIntersection (rect : Rectangle) (circ : Circle) (tri : RightTriangle) : ℝ :=
  sorry

theorem intersection_area_theorem (rect : Rectangle) (circ : Circle) (tri : RightTriangle) :
  rect.width = 10 →
  rect.height = 4 →
  circ.radius = 4 →
  rect.center = circ.center →
  tri.leg1 = 3 →
  tri.leg2 = 3 →
  -- Assuming the triangle is positioned correctly
  areaOfIntersection rect circ tri = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_area_theorem_l636_63637


namespace NUMINAMATH_CALUDE_rose_puzzle_l636_63657

theorem rose_puzzle : ∃! n : ℕ, 
  300 ≤ n ∧ n ≤ 400 ∧ 
  n % 21 = 13 ∧ 
  n % 15 = 7 ∧ 
  n = 307 := by sorry

end NUMINAMATH_CALUDE_rose_puzzle_l636_63657


namespace NUMINAMATH_CALUDE_untouchable_area_of_cube_l636_63639

-- Define the cube and sphere
def cube_edge_length : ℝ := 4
def sphere_radius : ℝ := 1

-- Theorem statement
theorem untouchable_area_of_cube (cube_edge_length : ℝ) (sphere_radius : ℝ) 
  (h1 : cube_edge_length = 4) (h2 : sphere_radius = 1) : 
  (6 * (cube_edge_length ^ 2 - (cube_edge_length - 2 * sphere_radius) ^ 2)) = 72 := by
  sorry

end NUMINAMATH_CALUDE_untouchable_area_of_cube_l636_63639


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l636_63621

theorem quadratic_form_sum (x : ℝ) : ∃ (b c : ℝ), 
  2*x^2 - 28*x + 50 = (x + b)^2 + c ∧ b + c = -55 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l636_63621


namespace NUMINAMATH_CALUDE_sqrt_y_squared_range_l636_63640

theorem sqrt_y_squared_range (y : ℝ) (h : (y + 16) ^ (1/3) - (y - 16) ^ (1/3) = 4) :
  15 < Real.sqrt (y^2) ∧ Real.sqrt (y^2) < 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_y_squared_range_l636_63640


namespace NUMINAMATH_CALUDE_square_sum_problem_l636_63617

theorem square_sum_problem (x y : ℝ) (h1 : x + 3*y = 6) (h2 : x*y = -12) : 
  x^2 + 6*y^2 = 108 := by sorry

end NUMINAMATH_CALUDE_square_sum_problem_l636_63617


namespace NUMINAMATH_CALUDE_total_marbles_is_27_l636_63634

/-- The total number of green and red marbles owned by Sara, Tom, and Lisa -/
def total_green_red_marbles (sara_green sara_red : ℕ) (tom_green tom_red : ℕ) (lisa_green lisa_red : ℕ) : ℕ :=
  sara_green + sara_red + tom_green + tom_red + lisa_green + lisa_red

/-- Theorem stating that the total number of green and red marbles is 27 -/
theorem total_marbles_is_27 :
  total_green_red_marbles 3 5 4 7 5 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_is_27_l636_63634


namespace NUMINAMATH_CALUDE_nested_radical_twenty_l636_63605

theorem nested_radical_twenty (x : ℝ) (h : x > 0) (eq : x = Real.sqrt (20 + x)) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_twenty_l636_63605


namespace NUMINAMATH_CALUDE_art_collection_cost_l636_63627

/-- The total cost of John's art collection --/
def total_cost (first_3_price : ℚ) : ℚ :=
  -- Cost of first 3 pieces
  3 * first_3_price +
  -- Cost of next 2 pieces (25% more expensive)
  2 * (first_3_price * (1 + 1/4)) +
  -- Cost of last 3 pieces (50% more expensive)
  3 * (first_3_price * (1 + 1/2))

/-- Theorem stating the total cost of John's art collection --/
theorem art_collection_cost :
  ∃ (first_3_price : ℚ),
    first_3_price > 0 ∧
    3 * first_3_price = 45000 ∧
    total_cost first_3_price = 150000 := by
  sorry


end NUMINAMATH_CALUDE_art_collection_cost_l636_63627


namespace NUMINAMATH_CALUDE_min_value_theorem_l636_63660

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (4 * x) / (x + 3 * y) + (3 * y) / x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l636_63660


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_64_l636_63600

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_64_l636_63600


namespace NUMINAMATH_CALUDE_repeating_six_equals_two_thirds_l636_63608

/-- The decimal representation of a real number with a single repeating digit. -/
def repeatingDecimal (digit : ℕ) : ℚ :=
  (digit : ℚ) / 9

/-- Prove that the repeating decimal 0.666... is equal to 2/3 -/
theorem repeating_six_equals_two_thirds :
  repeatingDecimal 6 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_six_equals_two_thirds_l636_63608


namespace NUMINAMATH_CALUDE_gcd_product_is_square_l636_63633

theorem gcd_product_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, (Nat.gcd x y).gcd z * x * y * z = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_product_is_square_l636_63633


namespace NUMINAMATH_CALUDE_fraction_product_proof_l636_63607

theorem fraction_product_proof :
  (8 / 4) * (10 / 25) * (27 / 18) * (16 / 24) * (35 / 21) * (30 / 50) * (14 / 7) * (20 / 40) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_proof_l636_63607


namespace NUMINAMATH_CALUDE_mom_t_shirt_purchase_l636_63681

/-- The number of packages of white t-shirts Mom bought -/
def num_packages : ℕ := 14

/-- The number of white t-shirts in each package -/
def t_shirts_per_package : ℕ := 5

/-- The total number of white t-shirts Mom bought -/
def total_t_shirts : ℕ := num_packages * t_shirts_per_package

theorem mom_t_shirt_purchase : total_t_shirts = 70 := by
  sorry

end NUMINAMATH_CALUDE_mom_t_shirt_purchase_l636_63681


namespace NUMINAMATH_CALUDE_spatial_quadrilateral_angle_sum_l636_63664

-- Define a spatial quadrilateral
structure SpatialQuadrilateral :=
  (A B C D : Real)

-- State the theorem
theorem spatial_quadrilateral_angle_sum 
  (q : SpatialQuadrilateral) : q.A + q.B + q.C + q.D ≤ 360 := by
  sorry

end NUMINAMATH_CALUDE_spatial_quadrilateral_angle_sum_l636_63664


namespace NUMINAMATH_CALUDE_five_hundredth_term_is_negative_one_l636_63606

def sequence_term (n : ℕ) : ℚ :=
  match n % 3 with
  | 1 => 2
  | 2 => -1
  | 0 => 1/2
  | _ => 0 -- This case should never occur

theorem five_hundredth_term_is_negative_one :
  sequence_term 500 = -1 := by
  sorry

end NUMINAMATH_CALUDE_five_hundredth_term_is_negative_one_l636_63606


namespace NUMINAMATH_CALUDE_square_13_on_top_l636_63619

/-- Represents a 5x5 grid of numbers -/
def Grid := Fin 5 → Fin 5 → Fin 25

/-- The initial configuration of the grid -/
def initial_grid : Grid :=
  fun i j => ⟨i.val * 5 + j.val + 1, by sorry⟩

/-- Represents a folding operation on the grid -/
def Fold := Grid → Grid

/-- Fold the top half over the bottom half -/
def fold1 : Fold := sorry

/-- Fold the bottom half over the top half -/
def fold2 : Fold := sorry

/-- Fold the left half over the right half -/
def fold3 : Fold := sorry

/-- Fold the right half over the left half -/
def fold4 : Fold := sorry

/-- Fold diagonally from bottom left to top right -/
def fold5 : Fold := sorry

/-- The final configuration after all folds -/
def final_grid : Grid :=
  fold5 (fold4 (fold3 (fold2 (fold1 initial_grid))))

/-- The theorem stating that square 13 is on top after all folds -/
theorem square_13_on_top :
  final_grid 0 0 = ⟨13, by sorry⟩ := by sorry

end NUMINAMATH_CALUDE_square_13_on_top_l636_63619


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l636_63659

theorem least_number_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((1056 + y) % 35 = 0 ∧ (1056 + y) % 51 = 0)) ∧
  ((1056 + x) % 35 = 0 ∧ (1056 + x) % 51 = 0) →
  x = 729 := by
sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l636_63659


namespace NUMINAMATH_CALUDE_door_pole_equation_l636_63613

/-- 
Given a rectangular door and a pole:
- The door's diagonal length is x
- The pole's length is x
- When placed horizontally, the pole extends 4 feet beyond the door's width
- When placed vertically, the pole extends 2 feet beyond the door's height

This theorem proves that the equation (x-2)^2 + (x-4)^2 = x^2 holds true for this configuration.
-/
theorem door_pole_equation (x : ℝ) : (x - 2)^2 + (x - 4)^2 = x^2 := by
  sorry

end NUMINAMATH_CALUDE_door_pole_equation_l636_63613


namespace NUMINAMATH_CALUDE_goods_train_speed_calculation_l636_63658

/-- The speed of the man's train in km/h -/
def man_train_speed : ℝ := 60

/-- The length of the goods train in meters -/
def goods_train_length : ℝ := 280

/-- The time it takes for the goods train to pass the man in seconds -/
def passing_time : ℝ := 9

/-- The speed of the goods train in km/h -/
def goods_train_speed : ℝ := 52

theorem goods_train_speed_calculation :
  (man_train_speed + goods_train_speed) * passing_time / 3600 = goods_train_length / 1000 :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_calculation_l636_63658


namespace NUMINAMATH_CALUDE_sum_of_digits_of_second_smallest_divisible_by_all_less_than_8_l636_63699

def is_divisible_by_all_less_than_8 (n : ℕ) : Prop :=
  ∀ k : ℕ, 0 < k ∧ k < 8 → n % k = 0

def second_smallest (P : ℕ → Prop) (n : ℕ) : Prop :=
  P n ∧ ∃ m : ℕ, P m ∧ m < n ∧ ∀ k : ℕ, P k → k = m ∨ n ≤ k

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_second_smallest_divisible_by_all_less_than_8 :
  ∃ N : ℕ, second_smallest is_divisible_by_all_less_than_8 N ∧ sum_of_digits N = 12 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_second_smallest_divisible_by_all_less_than_8_l636_63699


namespace NUMINAMATH_CALUDE_playlist_duration_l636_63629

/-- Given a playlist with three songs of durations 3, 2, and 3 minutes respectively,
    prove that listening to this playlist 5 times takes 40 minutes. -/
theorem playlist_duration (song1 song2 song3 : ℕ) (repetitions : ℕ) :
  song1 = 3 ∧ song2 = 2 ∧ song3 = 3 ∧ repetitions = 5 →
  (song1 + song2 + song3) * repetitions = 40 :=
by sorry

end NUMINAMATH_CALUDE_playlist_duration_l636_63629


namespace NUMINAMATH_CALUDE_function_inequality_l636_63693

open Real

theorem function_inequality (f : ℝ → ℝ) (h : ∀ x > 0, 2 * f x < x * (deriv f x) ∧ x * (deriv f x) < 3 * f x) :
  4 < f 2 / f 1 ∧ f 2 / f 1 < 8 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l636_63693


namespace NUMINAMATH_CALUDE_expression_simplification_l636_63612

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (2 * x - 6) / (x - 2) / (5 / (x - 2) - x - 2) = Real.sqrt 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l636_63612


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l636_63638

/-- The line (2a+b)x + (a+b)y + a - b = 0 passes through (-2, 3) for all real a and b -/
theorem line_passes_through_fixed_point :
  ∀ (a b x y : ℝ), (2*a + b)*x + (a + b)*y + a - b = 0 ↔ (x = -2 ∧ y = 3) ∨ (x ≠ -2 ∨ y ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l636_63638


namespace NUMINAMATH_CALUDE_number_puzzle_l636_63669

theorem number_puzzle : ∃! x : ℝ, 0.8 * x + 20 = x := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l636_63669


namespace NUMINAMATH_CALUDE_hyperbola_equation_l636_63685

/-- The equation of a hyperbola with given parameters -/
theorem hyperbola_equation (a c : ℝ) (h1 : a > 0) (h2 : c > a) :
  let e := c / a
  let b := Real.sqrt (c^2 - a^2)
  ∀ x y : ℝ, 2 * a = 8 → e = 5/4 →
    (x^2 / a^2 - y^2 / b^2 = 1) ↔ (x^2 / 16 - y^2 / 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l636_63685


namespace NUMINAMATH_CALUDE_exponential_decreasing_implies_cubic_increasing_l636_63698

theorem exponential_decreasing_implies_cubic_increasing
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x y : ℝ, x < y → a^x > a^y) :
  (∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3) ∧
  (∃ b : ℝ, b > 0 ∧ b ≠ 1 ∧
    (∀ x y : ℝ, x < y → (2 - b) * x^3 < (2 - b) * y^3) ∧
    ¬(∀ x y : ℝ, x < y → b^x > b^y)) :=
by sorry

end NUMINAMATH_CALUDE_exponential_decreasing_implies_cubic_increasing_l636_63698


namespace NUMINAMATH_CALUDE_gcd_12a_18b_min_l636_63641

theorem gcd_12a_18b_min (a b : ℕ+) (h : Nat.gcd a b = 9) :
  (∃ (a' b' : ℕ+), Nat.gcd a' b' = 9 ∧ Nat.gcd (12 * a') (18 * b') = 54) ∧
  (Nat.gcd (12 * a) (18 * b) ≥ 54) :=
sorry

end NUMINAMATH_CALUDE_gcd_12a_18b_min_l636_63641


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l636_63632

theorem sum_of_x_and_y_on_circle (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y - 48) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l636_63632


namespace NUMINAMATH_CALUDE_power_of_one_third_l636_63679

theorem power_of_one_third (a b : ℕ) : 
  (2^a : ℕ) * 5^b = 200 ∧ 
  ∀ k > a, ¬(2^k : ℕ) ∣ 200 ∧ 
  ∀ m > b, ¬(5^m : ℕ) ∣ 200 → 
  (1/3 : ℚ)^(b - a) = 3 := by
sorry

end NUMINAMATH_CALUDE_power_of_one_third_l636_63679


namespace NUMINAMATH_CALUDE_cosine_equality_condition_l636_63673

theorem cosine_equality_condition (x y : ℝ) : 
  (x = y → Real.cos x = Real.cos y) ∧ 
  ∃ a b : ℝ, Real.cos a = Real.cos b ∧ a ≠ b :=
by sorry

end NUMINAMATH_CALUDE_cosine_equality_condition_l636_63673


namespace NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l636_63615

-- Define the sets M and N
def M : Set ℝ := {x | x > 0}
def N : Set ℝ := {x | x > 1}

-- Theorem statement
theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a : ℝ, a ∈ N → a ∈ M) ∧ 
  (∃ a : ℝ, a ∈ M ∧ a ∉ N) :=
by sorry

end NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l636_63615


namespace NUMINAMATH_CALUDE_some_mythical_creatures_are_magical_beings_l636_63668

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Dragon : U → Prop)
variable (MythicalCreature : U → Prop)
variable (MagicalBeing : U → Prop)

-- State the theorem
theorem some_mythical_creatures_are_magical_beings
  (h1 : ∀ x, Dragon x → MythicalCreature x)
  (h2 : ∃ x, MagicalBeing x ∧ Dragon x) :
  ∃ x, MythicalCreature x ∧ MagicalBeing x :=
by
  sorry


end NUMINAMATH_CALUDE_some_mythical_creatures_are_magical_beings_l636_63668


namespace NUMINAMATH_CALUDE_constant_fifth_term_implies_n_six_l636_63674

/-- 
Given a positive integer n, and considering the binomial expansion of (x^2 + 1/x)^n,
if the fifth term is a constant (i.e., the exponent of x is 0), then n must equal 6.
-/
theorem constant_fifth_term_implies_n_six (n : ℕ+) : 
  (∃ k : ℕ, k > 0 ∧ 2*n - 3*(k+1) = 0) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_constant_fifth_term_implies_n_six_l636_63674


namespace NUMINAMATH_CALUDE_complex_multiplication_l636_63672

theorem complex_multiplication (i : ℂ) : i^2 = -1 → i * (2 - i) = 1 + 2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l636_63672


namespace NUMINAMATH_CALUDE_integer_triple_sum_equation_l636_63675

theorem integer_triple_sum_equation : ∃ (x y z : ℕ),
  1000 < x ∧ x < y ∧ y < z ∧ z < 2000 ∧
  (1 : ℚ) / 2 + 1 / 3 + 1 / 7 + 1 / x + 1 / y + 1 / z + 1 / 45 = 1 ∧
  x = 1806 ∧ y = 1892 ∧ z = 1980 := by
  sorry

end NUMINAMATH_CALUDE_integer_triple_sum_equation_l636_63675


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l636_63609

theorem quadratic_inequality_solution (x : ℝ) : 
  -3 * x^2 + 5 * x + 4 < 0 ↔ -4/3 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l636_63609


namespace NUMINAMATH_CALUDE_third_day_temperature_l636_63688

/-- Given three temperatures with a known average and two known values, 
    calculate the third temperature. -/
theorem third_day_temperature 
  (avg : ℚ) 
  (temp1 temp2 : ℚ) 
  (h_avg : avg = -7)
  (h_temp1 : temp1 = -14)
  (h_temp2 : temp2 = -8)
  (h_sum : 3 * avg = temp1 + temp2 + temp3) :
  temp3 = 1 := by
  sorry

#check third_day_temperature

end NUMINAMATH_CALUDE_third_day_temperature_l636_63688


namespace NUMINAMATH_CALUDE_fraction_equivalence_l636_63655

theorem fraction_equivalence : 
  ∀ (n : ℚ), (3 + n) / (5 + n) = 5 / 6 → n = 7 := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l636_63655


namespace NUMINAMATH_CALUDE_yellow_balls_count_l636_63630

/-- Given a bag with 50 balls of two colors, if the frequency of picking one color (yellow)
    stabilizes around 0.3, then the number of yellow balls is 15. -/
theorem yellow_balls_count (total_balls : ℕ) (yellow_frequency : ℚ) 
  (h1 : total_balls = 50)
  (h2 : yellow_frequency = 3/10) : 
  ∃ (yellow_balls : ℕ), yellow_balls = 15 ∧ yellow_balls / total_balls = yellow_frequency := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l636_63630


namespace NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l636_63677

theorem quadratic_equation_two_distinct_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 2 * x₁^2 - m * x₁ - 1 = 0 ∧ 2 * x₂^2 - m * x₂ - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l636_63677


namespace NUMINAMATH_CALUDE_marcia_blouses_l636_63671

/-- Calculates the number of blouses Marcia can add to her wardrobe given the following conditions:
  * Marcia needs 3 skirts, 2 pairs of pants, and some blouses
  * Skirts cost $20.00 each
  * Blouses cost $15.00 each
  * Pants cost $30.00 each
  * There's a sale on pants: buy 1 pair get 1 pair 1/2 off
  * Total budget is $180.00
-/
def calculate_blouses (skirt_count : Nat) (skirt_price : Nat) (pant_count : Nat) (pant_price : Nat) (blouse_price : Nat) (total_budget : Nat) : Nat :=
  let skirt_total := skirt_count * skirt_price
  let pant_total := pant_price + (pant_price / 2)
  let remaining_budget := total_budget - skirt_total - pant_total
  remaining_budget / blouse_price

theorem marcia_blouses :
  calculate_blouses 3 20 2 30 15 180 = 5 := by
  sorry

end NUMINAMATH_CALUDE_marcia_blouses_l636_63671


namespace NUMINAMATH_CALUDE_tangled_legs_scenario_l636_63626

/-- The number of legs tangled in leashes when two dog walkers meet --/
def tangled_legs (dogs_group1 : ℕ) (dogs_group2 : ℕ) (legs_per_dog : ℕ) (walkers : ℕ) (legs_per_walker : ℕ) : ℕ :=
  (dogs_group1 + dogs_group2) * legs_per_dog + walkers * legs_per_walker

/-- Theorem stating the number of legs tangled in leashes in the given scenario --/
theorem tangled_legs_scenario : tangled_legs 5 3 4 2 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_tangled_legs_scenario_l636_63626


namespace NUMINAMATH_CALUDE_set_operation_proof_l636_63667

theorem set_operation_proof (A B C : Set ℕ) : 
  A = {1, 2} → B = {1, 2, 3} → C = {2, 3, 4} → 
  (A ∩ B) ∪ C = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_proof_l636_63667


namespace NUMINAMATH_CALUDE_quadratic_inequalities_solution_sets_l636_63696

theorem quadratic_inequalities_solution_sets 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | x^2 - a*x - b < 0}) : 
  Set.Ioo (-1/2) (-1/3) = {x : ℝ | b*x^2 - a*x - 1 > 0} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_solution_sets_l636_63696


namespace NUMINAMATH_CALUDE_product_bounds_l636_63624

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  x + y = a ∧ x^2 + y^2 = -a^2 + 2

-- Define the product function
def product (x y : ℝ) : ℝ := x * y

-- Theorem statement
theorem product_bounds :
  ∀ x y a : ℝ, system x y a → 
    (∃ x' y' a' : ℝ, system x' y' a' ∧ product x' y' = 1/3) ∧
    (∃ x'' y'' a'' : ℝ, system x'' y'' a'' ∧ product x'' y'' = -1) ∧
    (∀ x''' y''' a''' : ℝ, system x''' y''' a''' → 
      -1 ≤ product x''' y''' ∧ product x''' y''' ≤ 1/3) :=
sorry

end NUMINAMATH_CALUDE_product_bounds_l636_63624


namespace NUMINAMATH_CALUDE_shelter_dogs_l636_63683

/-- The number of dogs in an animal shelter given specific ratios -/
theorem shelter_dogs (d c : ℕ) (h1 : d * 7 = c * 15) (h2 : d * 11 = (c + 20) * 15) : d = 175 := by
  sorry

end NUMINAMATH_CALUDE_shelter_dogs_l636_63683


namespace NUMINAMATH_CALUDE_target_has_six_more_tools_l636_63661

/-- The number of tools in the Walmart multitool -/
def walmart_tools : ℕ := 1 + 3 + 2

/-- The number of tools in the Target multitool -/
def target_tools : ℕ := 1 + (2 * 3) + 3 + 1

/-- The difference in the number of tools between Target and Walmart multitools -/
def tool_difference : ℕ := target_tools - walmart_tools

theorem target_has_six_more_tools : tool_difference = 6 := by
  sorry

end NUMINAMATH_CALUDE_target_has_six_more_tools_l636_63661


namespace NUMINAMATH_CALUDE_megan_files_added_l636_63652

theorem megan_files_added 
  (initial_files : ℝ) 
  (files_per_folder : ℝ) 
  (num_folders : ℝ) 
  (h1 : initial_files = 93.0) 
  (h2 : files_per_folder = 8.0) 
  (h3 : num_folders = 14.25) : 
  num_folders * files_per_folder - initial_files = 21.0 := by
sorry

end NUMINAMATH_CALUDE_megan_files_added_l636_63652


namespace NUMINAMATH_CALUDE_pentadecagon_diagonals_l636_63625

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A pentadecagon is a 15-sided polygon -/
def pentadecagon_sides : ℕ := 15

/-- Theorem: The number of diagonals in a pentadecagon is 90 -/
theorem pentadecagon_diagonals : num_diagonals pentadecagon_sides = 90 := by
  sorry

end NUMINAMATH_CALUDE_pentadecagon_diagonals_l636_63625


namespace NUMINAMATH_CALUDE_students_walking_home_l636_63610

theorem students_walking_home (bus automobile skateboard bicycle : ℚ)
  (h_bus : bus = 1 / 3)
  (h_auto : automobile = 1 / 5)
  (h_skate : skateboard = 1 / 8)
  (h_bike : bicycle = 1 / 10)
  (h_total : bus + automobile + skateboard + bicycle < 1) :
  1 - (bus + automobile + skateboard + bicycle) = 29 / 120 := by
  sorry

end NUMINAMATH_CALUDE_students_walking_home_l636_63610


namespace NUMINAMATH_CALUDE_max_value_abc_l636_63689

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + 2*b + c = 2) :
  ∃ (max : ℝ), max = 1/2 + Real.sqrt 3 + (3/32)^(1/3) ∧
  ∀ (x : ℝ), x = a + 2*Real.sqrt (a*b) + (a*b*c)^(1/3) → x ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l636_63689


namespace NUMINAMATH_CALUDE_seventh_selected_number_l636_63653

def random_sequence : List ℕ := [6572, 0802, 6319, 8702, 4369, 9728, 0198, 3204, 9243, 4935, 8200, 3623, 4869, 6938, 7481]

def is_valid (n : ℕ) : Bool := 1 ≤ n ∧ n ≤ 500

def select_valid_numbers (seq : List ℕ) : List ℕ :=
  seq.filter (λ n => is_valid (n % 1000))

theorem seventh_selected_number :
  (select_valid_numbers random_sequence).nthLe 6 sorry = 320 := by sorry

end NUMINAMATH_CALUDE_seventh_selected_number_l636_63653


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l636_63618

-- Define the points
def p1 : ℝ × ℝ := (1, 3)
def p2 : ℝ × ℝ := (5, -1)
def p3 : ℝ × ℝ := (10, 3)
def p4 : ℝ × ℝ := (5, 7)

-- Define the ellipse
def ellipse (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  let center := ((p1.1 + p3.1) / 2, (p1.2 + p3.2) / 2)
  let a := (p3.1 - p1.1) / 2
  let b := (p4.2 - p2.2) / 2
  (center.1 = (p2.1 + p4.1) / 2) ∧ 
  (center.2 = (p2.2 + p4.2) / 2) ∧
  (a > b) ∧ (b > 0)

-- Theorem statement
theorem ellipse_foci_distance (h : ellipse p1 p2 p3 p4) :
  let a := (p3.1 - p1.1) / 2
  let b := (p4.2 - p2.2) / 2
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 2 * Real.sqrt 4.25 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l636_63618


namespace NUMINAMATH_CALUDE_quadratic_factorization_l636_63616

theorem quadratic_factorization (y A B : ℤ) : 
  (15 * y^2 - 94 * y + 56 = (A * y - 7) * (B * y - 8)) → 
  (A * B + A = 20) := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l636_63616


namespace NUMINAMATH_CALUDE_polly_lunch_time_l636_63645

/-- Represents the cooking time for a week -/
structure CookingTime where
  breakfast : ℕ  -- Time spent on breakfast daily
  dinner_short : ℕ  -- Time spent on dinner for short days
  dinner_long : ℕ  -- Time spent on dinner for long days
  short_days : ℕ  -- Number of days with short dinner time
  total : ℕ  -- Total cooking time for the week

/-- Calculates the time spent on lunch given the cooking time for other meals -/
def lunch_time (c : CookingTime) : ℕ :=
  c.total - (7 * c.breakfast + c.short_days * c.dinner_short + (7 - c.short_days) * c.dinner_long)

/-- Theorem stating that Polly spends 35 minutes cooking lunch -/
theorem polly_lunch_time :
  ∃ (c : CookingTime),
    c.breakfast = 20 ∧
    c.dinner_short = 10 ∧
    c.dinner_long = 30 ∧
    c.short_days = 4 ∧
    c.total = 305 ∧
    lunch_time c = 35 := by
  sorry

end NUMINAMATH_CALUDE_polly_lunch_time_l636_63645


namespace NUMINAMATH_CALUDE_larger_number_problem_l636_63665

theorem larger_number_problem (L S : ℕ) 
  (h1 : L - S = 1325)
  (h2 : L = 5 * S + 5) :
  L = 1655 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l636_63665


namespace NUMINAMATH_CALUDE_P_equals_set_l636_63623

def P : Set ℝ := {x | x^2 = 1}

theorem P_equals_set : P = {-1, 1} := by
  sorry

end NUMINAMATH_CALUDE_P_equals_set_l636_63623


namespace NUMINAMATH_CALUDE_cycle_sale_gain_percent_l636_63687

/-- Calculates the gain percent for a cycle sale given the original price, discount percentage, refurbishing cost, and selling price. -/
def cycleGainPercent (originalPrice discountPercent refurbishCost sellingPrice : ℚ) : ℚ :=
  let discountAmount := (discountPercent / 100) * originalPrice
  let purchasePrice := originalPrice - discountAmount
  let totalCostPrice := purchasePrice + refurbishCost
  let gain := sellingPrice - totalCostPrice
  (gain / totalCostPrice) * 100

/-- Theorem stating that the gain percent for the given cycle sale scenario is 62.5% -/
theorem cycle_sale_gain_percent :
  cycleGainPercent 1200 25 300 1950 = 62.5 := by
  sorry

#eval cycleGainPercent 1200 25 300 1950

end NUMINAMATH_CALUDE_cycle_sale_gain_percent_l636_63687


namespace NUMINAMATH_CALUDE_subtraction_from_percentage_l636_63680

theorem subtraction_from_percentage (n : ℝ) : n = 100 → (0.8 * n - 20 = 60) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_from_percentage_l636_63680


namespace NUMINAMATH_CALUDE_derivative_at_zero_l636_63614

theorem derivative_at_zero (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x*(deriv f (-1))) :
  deriv f 0 = 4 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_zero_l636_63614


namespace NUMINAMATH_CALUDE_equation_roots_l636_63691

-- Define the equation
def equation (x : ℝ) : Prop :=
  (21 / (x^2 - 9) - 3 / (x - 3) = 1)

-- Define the roots
def roots : Set ℝ := {-3, 7}

-- Theorem statement
theorem equation_roots :
  ∀ x : ℝ, x ∈ roots ↔ equation x ∧ x ≠ 3 ∧ x ≠ -3 :=
sorry

end NUMINAMATH_CALUDE_equation_roots_l636_63691


namespace NUMINAMATH_CALUDE_underdog_wins_in_nine_games_l636_63602

/- Define the probability of the favored team winning a single game -/
def p : ℚ := 3/4

/- Define the number of games needed to win the series -/
def games_to_win : ℕ := 5

/- Define the maximum number of games in the series -/
def max_games : ℕ := 9

/- Define the probability of the underdog team winning a single game -/
def q : ℚ := 1 - p

/- Define the number of ways to choose 4 games out of 8 -/
def ways_to_choose : ℕ := Nat.choose 8 4

theorem underdog_wins_in_nine_games :
  (ways_to_choose : ℚ) * q^4 * p^4 * q = 5670/262144 := by
  sorry

end NUMINAMATH_CALUDE_underdog_wins_in_nine_games_l636_63602


namespace NUMINAMATH_CALUDE_log_equation_implies_sum_l636_63684

theorem log_equation_implies_sum (x y : ℝ) 
  (h1 : x > 1) (h2 : y > 1) 
  (h3 : (Real.log x / Real.log 4)^3 + (Real.log y / Real.log 5)^3 + 6 = 
        6 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) : 
  x^Real.sqrt 3 + y^Real.sqrt 3 = 189 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_implies_sum_l636_63684


namespace NUMINAMATH_CALUDE_thief_catch_time_l636_63636

/-- The time it takes for the passenger to catch the thief -/
def catchUpTime (thief_speed : ℝ) (passenger_speed : ℝ) (bus_speed : ℝ) (stop_time : ℝ) : ℝ :=
  stop_time

theorem thief_catch_time :
  ∀ (thief_speed : ℝ),
    thief_speed > 0 →
    let passenger_speed := 2 * thief_speed
    let bus_speed := 10 * thief_speed
    let stop_time := 40
    catchUpTime thief_speed passenger_speed bus_speed stop_time = 40 :=
by
  sorry

#check thief_catch_time

end NUMINAMATH_CALUDE_thief_catch_time_l636_63636


namespace NUMINAMATH_CALUDE_concert_ticket_price_l636_63666

theorem concert_ticket_price (total_people : Nat) (discount_group1 : Nat) (discount_group2 : Nat)
  (discount1 : Real) (discount2 : Real) (total_revenue : Real) :
  total_people = 56 →
  discount_group1 = 10 →
  discount_group2 = 20 →
  discount1 = 0.4 →
  discount2 = 0.15 →
  total_revenue = 980 →
  ∃ (original_price : Real),
    original_price = 20 ∧
    total_revenue = (discount_group1 * (1 - discount1) * original_price) +
                    (discount_group2 * (1 - discount2) * original_price) +
                    ((total_people - discount_group1 - discount_group2) * original_price) :=
by sorry


end NUMINAMATH_CALUDE_concert_ticket_price_l636_63666


namespace NUMINAMATH_CALUDE_birthday_count_theorem_l636_63628

/-- Represents a date with year, month, and day -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents a birthday -/
structure Birthday where
  month : ℕ
  day : ℕ

def startDate : Date := ⟨2012, 12, 26⟩

def dadBirthday : Birthday := ⟨5, 1⟩
def chunchunBirthday : Birthday := ⟨7, 1⟩

def daysToCount : ℕ := 2013

/-- Counts the number of birthdays between two dates -/
def countBirthdays (start : Date) (days : ℕ) (birthday : Birthday) : ℕ :=
  sorry

theorem birthday_count_theorem :
  countBirthdays startDate daysToCount dadBirthday +
  countBirthdays startDate daysToCount chunchunBirthday = 11 :=
by sorry

end NUMINAMATH_CALUDE_birthday_count_theorem_l636_63628


namespace NUMINAMATH_CALUDE_total_puppies_l636_63692

/-- Given an initial number of puppies and a number of additional puppies,
    prove that the total number of puppies is equal to the sum of the initial number
    and the additional number. -/
theorem total_puppies (initial_puppies additional_puppies : ℝ) :
  initial_puppies + additional_puppies = initial_puppies + additional_puppies :=
by sorry

end NUMINAMATH_CALUDE_total_puppies_l636_63692


namespace NUMINAMATH_CALUDE_weightlifting_total_capacity_l636_63622

/-- Represents a weightlifter's lifting capacities -/
structure LiftingCapacity where
  cleanAndJerk : ℝ
  snatch : ℝ

/-- Calculates the new lifting capacity after applying percentage increases -/
def newCapacity (initial : LiftingCapacity) (cjIncrease snatchIncrease : ℝ) : LiftingCapacity :=
  { cleanAndJerk := initial.cleanAndJerk * (1 + cjIncrease)
  , snatch := initial.snatch * (1 + snatchIncrease) }

/-- Calculates the total lifting capacity for a weightlifter -/
def totalCapacity (capacity : LiftingCapacity) : ℝ :=
  capacity.cleanAndJerk + capacity.snatch

/-- The theorem to be proved -/
theorem weightlifting_total_capacity : 
  let john_initial := LiftingCapacity.mk 80 50
  let alice_initial := LiftingCapacity.mk 90 55
  let mark_initial := LiftingCapacity.mk 100 65
  
  let john_final := newCapacity john_initial 1 0.8
  let alice_final := newCapacity alice_initial 0.5 0.9
  let mark_final := newCapacity mark_initial 0.75 0.7
  
  totalCapacity john_final + totalCapacity alice_final + totalCapacity mark_final = 775 := by
  sorry

end NUMINAMATH_CALUDE_weightlifting_total_capacity_l636_63622


namespace NUMINAMATH_CALUDE_sum_three_consecutive_divisible_by_three_l636_63697

theorem sum_three_consecutive_divisible_by_three (n : ℕ) :
  ∃ k : ℕ, n + (n + 1) + (n + 2) = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_three_consecutive_divisible_by_three_l636_63697


namespace NUMINAMATH_CALUDE_program_size_calculation_l636_63663

/-- Calculates the size of a downloaded program given the download speed and time -/
theorem program_size_calculation (download_speed : ℝ) (download_time : ℝ) : 
  download_speed = 50 → download_time = 2 → 
  download_speed * download_time * 60 * 60 / 1024 = 351.5625 := by
  sorry

#check program_size_calculation

end NUMINAMATH_CALUDE_program_size_calculation_l636_63663


namespace NUMINAMATH_CALUDE_starting_lineups_count_l636_63648

def total_players : ℕ := 15
def lineup_size : ℕ := 6
def injured_players : ℕ := 1
def incompatible_players : ℕ := 2

theorem starting_lineups_count :
  (Nat.choose (total_players - incompatible_players - injured_players + 1) (lineup_size - 1)) * 2 +
  (Nat.choose (total_players - incompatible_players - injured_players) lineup_size) = 3498 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineups_count_l636_63648


namespace NUMINAMATH_CALUDE_ball_max_height_l636_63670

/-- The height function of the ball -/
def h (t : ℝ) : ℝ := -20 * t^2 + 70 * t + 20

/-- The maximum height reached by the ball -/
def max_height : ℝ := 81.25

theorem ball_max_height :
  ∃ t : ℝ, h t = max_height ∧ ∀ s : ℝ, h s ≤ h t :=
by sorry

end NUMINAMATH_CALUDE_ball_max_height_l636_63670


namespace NUMINAMATH_CALUDE_horner_method_v₃_l636_63603

def f (x : ℝ) : ℝ := 2*x^5 - 5*x^4 - 4*x^3 + 3*x^2 - 6*x + 7

def horner_v₃ (x v v₁ : ℝ) : ℝ :=
  let v₂ := v₁ * x - 4
  v₂ * x + 3

theorem horner_method_v₃ :
  let x : ℝ := 5
  let v : ℝ := 2
  let v₁ : ℝ := 5
  horner_v₃ x v v₁ = 108 := by sorry

end NUMINAMATH_CALUDE_horner_method_v₃_l636_63603


namespace NUMINAMATH_CALUDE_minimum_total_tests_l636_63604

/-- Represents the test data for a student -/
structure StudentData where
  name : String
  numTests : ℕ
  avgScore : ℕ
  totalScore : ℕ

/-- The problem statement -/
theorem minimum_total_tests (k m r : StudentData) : 
  k.name = "Michael K" →
  m.name = "Michael M" →
  r.name = "Michael R" →
  k.avgScore = 90 →
  m.avgScore = 91 →
  r.avgScore = 92 →
  k.numTests > m.numTests →
  m.numTests > r.numTests →
  m.totalScore > r.totalScore →
  r.totalScore > k.totalScore →
  k.totalScore = k.numTests * k.avgScore →
  m.totalScore = m.numTests * m.avgScore →
  r.totalScore = r.numTests * r.avgScore →
  k.numTests + m.numTests + r.numTests ≥ 413 :=
by sorry

end NUMINAMATH_CALUDE_minimum_total_tests_l636_63604


namespace NUMINAMATH_CALUDE_garden_perimeter_l636_63676

/-- The perimeter of a rectangle with length l and breadth b is 2 * (l + b) -/
def rectanglePerimeter (l b : ℝ) : ℝ := 2 * (l + b)

/-- The perimeter of a rectangular garden with length 500 m and breadth 400 m is 1800 m -/
theorem garden_perimeter :
  rectanglePerimeter 500 400 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l636_63676


namespace NUMINAMATH_CALUDE_adam_orchard_apples_l636_63656

/-- Represents the number of apples Adam collected from his orchard -/
def total_apples (daily_apples : ℕ) (days : ℕ) (remaining_apples : ℕ) : ℕ :=
  daily_apples * days + remaining_apples

/-- Theorem stating the total number of apples Adam collected -/
theorem adam_orchard_apples :
  total_apples 4 30 230 = 350 := by
  sorry

end NUMINAMATH_CALUDE_adam_orchard_apples_l636_63656


namespace NUMINAMATH_CALUDE_range_of_a_l636_63654

theorem range_of_a (a : ℝ) : 
  (∀ n : ℕ+, ((1 - a) * n - a) * Real.log a < 0) ↔ (0 < a ∧ a < 1/2) ∨ a > 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l636_63654


namespace NUMINAMATH_CALUDE_scientific_notation_10500_l636_63620

/-- Theorem: 10500 is equal to 1.05 × 10^4 in scientific notation -/
theorem scientific_notation_10500 :
  (10500 : ℝ) = 1.05 * (10 : ℝ)^4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_10500_l636_63620


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_345_triangle_l636_63643

/-- A triangle with side lengths 3, 4, and 5 has an inscribed circle with radius 1. -/
theorem inscribed_circle_radius_345_triangle :
  ∀ (a b c r : ℝ),
  a = 3 ∧ b = 4 ∧ c = 5 →
  (a + b + c) / 2 * r = (a * b) / 2 →
  r = 1 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_345_triangle_l636_63643


namespace NUMINAMATH_CALUDE_min_distinct_values_l636_63644

theorem min_distinct_values (n : ℕ) (mode_count : ℕ) (total_count : ℕ) :
  n = 2017 →
  mode_count = 11 →
  total_count = n →
  (∃ (distinct_values : ℕ), 
    distinct_values ≥ 202 ∧
    ∀ (m : ℕ), m < 202 → 
      ¬(∃ (list : List ℕ),
        list.length = total_count ∧
        (∃ (mode : ℕ), list.count mode = mode_count ∧
          ∀ (x : ℕ), x ≠ mode → list.count x < mode_count) ∧
        list.toFinset.card = m)) :=
sorry

end NUMINAMATH_CALUDE_min_distinct_values_l636_63644


namespace NUMINAMATH_CALUDE_bridge_length_proof_l636_63682

/-- The length of a bridge given train parameters -/
theorem bridge_length_proof (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 275 :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_proof_l636_63682


namespace NUMINAMATH_CALUDE_white_closed_under_add_mul_l636_63651

/-- A color type representing black and white --/
inductive Color
| Black
| White

/-- A function that assigns a color to each positive integer --/
def coloring : ℕ+ → Color := sorry

/-- The property that the sum of two differently colored numbers is black --/
axiom sum_diff_color_black :
  ∀ (a b : ℕ+), coloring a ≠ coloring b → coloring (a + b) = Color.Black

/-- The property that there are infinitely many white numbers --/
axiom infinitely_many_white :
  ∀ (n : ℕ), ∃ (m : ℕ+), m > n ∧ coloring m = Color.White

/-- The theorem stating that the set of white numbers is closed under addition and multiplication --/
theorem white_closed_under_add_mul :
  ∀ (a b : ℕ+),
    coloring a = Color.White →
    coloring b = Color.White →
    coloring (a + b) = Color.White ∧ coloring (a * b) = Color.White :=
by sorry

end NUMINAMATH_CALUDE_white_closed_under_add_mul_l636_63651


namespace NUMINAMATH_CALUDE_jason_borrowed_amount_l636_63647

/-- Calculates the payment for a given hour based on the repeating pattern -/
def hourly_payment (hour : ℕ) : ℕ :=
  (hour - 1) % 6 + 1

/-- Calculates the total payment for a given number of hours -/
def total_payment (hours : ℕ) : ℕ :=
  (List.range hours).map hourly_payment |>.sum

/-- The problem statement -/
theorem jason_borrowed_amount :
  total_payment 39 = 132 := by
  sorry

end NUMINAMATH_CALUDE_jason_borrowed_amount_l636_63647


namespace NUMINAMATH_CALUDE_coronavirus_cases_l636_63646

theorem coronavirus_cases (initial_cases : ℕ) : 
  initial_cases > 0 →
  initial_cases + 450 + 1300 = 3750 →
  initial_cases = 2000 := by
sorry

end NUMINAMATH_CALUDE_coronavirus_cases_l636_63646


namespace NUMINAMATH_CALUDE_chocolate_cookie_percentage_l636_63601

/-- Calculates the percentage of chocolate in cookies given the initial ingredients and leftover chocolate. -/
theorem chocolate_cookie_percentage
  (dough : ℝ)
  (initial_chocolate : ℝ)
  (leftover_chocolate : ℝ)
  (h_dough : dough = 36)
  (h_initial : initial_chocolate = 13)
  (h_leftover : leftover_chocolate = 4) :
  (initial_chocolate - leftover_chocolate) / (dough + initial_chocolate - leftover_chocolate) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cookie_percentage_l636_63601


namespace NUMINAMATH_CALUDE_brian_always_wins_l636_63690

/-- Represents the game board -/
structure GameBoard :=
  (n : ℕ)

/-- Represents a player in the game -/
inductive Player := | Albus | Brian

/-- Represents a position on the game board -/
structure Position :=
  (x : ℤ)
  (y : ℤ)

/-- Represents the state of the game -/
structure GameState :=
  (board : GameBoard)
  (position : Position)
  (current_player : Player)
  (move_distance : ℕ)

/-- Checks if a position is within the game board -/
def is_valid_position (board : GameBoard) (pos : Position) : Prop :=
  abs pos.x ≤ board.n ∧ abs pos.y ≤ board.n

/-- Defines the initial game state -/
def initial_state (n : ℕ) : GameState :=
  { board := { n := n },
    position := { x := 0, y := 0 },
    current_player := Player.Albus,
    move_distance := 1 }

/-- Theorem: Brian always has a winning strategy -/
theorem brian_always_wins (n : ℕ) :
  ∃ (strategy : GameState → Position),
    ∀ (game : GameState),
      game.current_player = Player.Brian →
      is_valid_position game.board (strategy game) →
      ¬is_valid_position game.board
        {x := 2 * game.position.x - (strategy game).x,
         y := 2 * game.position.y - (strategy game).y} :=
sorry

end NUMINAMATH_CALUDE_brian_always_wins_l636_63690


namespace NUMINAMATH_CALUDE_driver_speed_proof_l636_63642

theorem driver_speed_proof (v : ℝ) : v > 0 → v / (v + 12) = 2/3 → v = 24 := by
  sorry

end NUMINAMATH_CALUDE_driver_speed_proof_l636_63642
