import Mathlib

namespace NUMINAMATH_CALUDE_pool_paint_area_calculation_l2344_234478

/-- Calculates the total area to be painted in a cuboid-shaped pool -/
def poolPaintArea (length width depth : ℝ) : ℝ :=
  2 * (length * depth + width * depth) + length * width

theorem pool_paint_area_calculation :
  let length : ℝ := 20
  let width : ℝ := 12
  let depth : ℝ := 2
  poolPaintArea length width depth = 368 := by
  sorry

end NUMINAMATH_CALUDE_pool_paint_area_calculation_l2344_234478


namespace NUMINAMATH_CALUDE_min_quotient_value_l2344_234465

theorem min_quotient_value (a b : ℝ) 
  (ha : 100 ≤ a ∧ a ≤ 300)
  (hb : 400 ≤ b ∧ b ≤ 800)
  (hab : a + b ≤ 950) :
  (∀ a' b', 100 ≤ a' ∧ a' ≤ 300 → 400 ≤ b' ∧ b' ≤ 800 → a' + b' ≤ 950 → a / b ≤ a' / b') →
  a / b = 1 / 8 :=
by sorry

end NUMINAMATH_CALUDE_min_quotient_value_l2344_234465


namespace NUMINAMATH_CALUDE_inequality_of_positive_reals_l2344_234489

theorem inequality_of_positive_reals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x / (y + z)) + Real.sqrt (y / (z + x)) + Real.sqrt (z / (x + y)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_positive_reals_l2344_234489


namespace NUMINAMATH_CALUDE_complex_real_condition_l2344_234487

theorem complex_real_condition (a : ℝ) : 
  (Complex.mk (1 / (a + 5)) (a^2 + 2*a - 15)).im = 0 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l2344_234487


namespace NUMINAMATH_CALUDE_a_spending_percentage_l2344_234496

/-- Proves that A spends 95% of his salary given the conditions of the problem -/
theorem a_spending_percentage 
  (total_salary : ℝ) 
  (a_salary : ℝ) 
  (b_spending_percentage : ℝ) 
  (h1 : total_salary = 7000)
  (h2 : a_salary = 5250)
  (h3 : b_spending_percentage = 0.85)
  (h4 : ∃ (a_spending_percentage : ℝ), 
    a_salary * (1 - a_spending_percentage) = (total_salary - a_salary) * (1 - b_spending_percentage)) :
  ∃ (a_spending_percentage : ℝ), a_spending_percentage = 0.95 := by
sorry

end NUMINAMATH_CALUDE_a_spending_percentage_l2344_234496


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2344_234499

theorem inscribed_squares_ratio : 
  ∀ x y : ℝ,
  (x > 0) →
  (y > 0) →
  (x^2 + x * 5 = 5 * 12) →
  (8/5 * y^2 + y^2 + 3/5 * y^2 = 10 * 2) →
  x / y = 96 / 85 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2344_234499


namespace NUMINAMATH_CALUDE_daily_production_is_1100_l2344_234472

/-- A factory produces toys with the following characteristics:
  * Produces a total of 5500 toys per week
  * Works 5 days per week
  * Produces the same number of toys each working day
-/
def ToyFactory : Type :=
  { weekly_production : ℕ // weekly_production = 5500 }
  × { working_days : ℕ // working_days = 5 }

/-- Calculate the daily toy production for a given toy factory -/
def daily_production (factory : ToyFactory) : ℕ :=
  factory.1 / factory.2

/-- Theorem stating that the daily production of toys is 1100 -/
theorem daily_production_is_1100 (factory : ToyFactory) :
  daily_production factory = 1100 := by
  sorry


end NUMINAMATH_CALUDE_daily_production_is_1100_l2344_234472


namespace NUMINAMATH_CALUDE_gas_fill_friday_l2344_234432

/-- Calculates the number of liters of gas Mr. Deane will fill on Friday given the conditions of the problem. -/
theorem gas_fill_friday 
  (today_liters : ℝ) 
  (today_price : ℝ) 
  (price_rollback : ℝ) 
  (total_cost : ℝ) 
  (total_liters : ℝ) 
  (h1 : today_liters = 10)
  (h2 : today_price = 1.4)
  (h3 : price_rollback = 0.4)
  (h4 : total_cost = 39)
  (h5 : total_liters = 35) :
  total_liters - today_liters = 25 := by
sorry

end NUMINAMATH_CALUDE_gas_fill_friday_l2344_234432


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2344_234447

theorem absolute_value_inequality (x : ℝ) : 
  x ≠ 3 → (|(3 * x + 2) / (x - 3)| < 4 ↔ (10/7 < x ∧ x < 3) ∨ (3 < x ∧ x < 14)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2344_234447


namespace NUMINAMATH_CALUDE_rectangle_area_l2344_234416

theorem rectangle_area (x : ℝ) : 
  (2 * (x + 4) + 2 * (x - 2) = 56) → 
  ((x + 4) * (x - 2) = 187) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2344_234416


namespace NUMINAMATH_CALUDE_equation_solution_l2344_234407

theorem equation_solution : ∃ y : ℚ, (40 / 60 = Real.sqrt (y / 60)) ∧ y = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2344_234407


namespace NUMINAMATH_CALUDE_v_closed_under_cube_l2344_234479

def v : Set ℕ := {n : ℕ | ∃ m : ℕ, n = m^4}

theorem v_closed_under_cube (x : ℕ) (hx : x ∈ v) : x^3 ∈ v := by
  sorry

end NUMINAMATH_CALUDE_v_closed_under_cube_l2344_234479


namespace NUMINAMATH_CALUDE_jerome_money_problem_l2344_234414

theorem jerome_money_problem (certain_amount : ℕ) : 
  (2 * certain_amount - (8 + 3 * 8) = 54) → 
  certain_amount = 43 := by
  sorry

end NUMINAMATH_CALUDE_jerome_money_problem_l2344_234414


namespace NUMINAMATH_CALUDE_prob_two_diff_numbers_correct_l2344_234458

/-- The number of faces on a standard die -/
def num_faces : ℕ := 6

/-- The number of dice being rolled -/
def num_dice : ℕ := 3

/-- The probability of getting exactly two different numbers when rolling three standard six-sided dice -/
def prob_two_diff_numbers : ℚ := sorry

theorem prob_two_diff_numbers_correct :
  prob_two_diff_numbers = 
    (num_faces.choose 2 * num_dice * (num_faces - 2)) / (num_faces ^ num_dice) :=
by sorry

end NUMINAMATH_CALUDE_prob_two_diff_numbers_correct_l2344_234458


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2344_234436

/-- A triangle with side lengths x, x+1, and x-1 has a perimeter of 21 if and only if x = 7 -/
theorem triangle_perimeter (x : ℝ) : 
  x > 0 ∧ x + 1 > 0 ∧ x - 1 > 0 → 
  (x + (x + 1) + (x - 1) = 21 ↔ x = 7) := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2344_234436


namespace NUMINAMATH_CALUDE_local_minimum_at_one_l2344_234463

-- Define the function f
def f (x m : ℝ) : ℝ := x * (x - m)^2

-- State the theorem
theorem local_minimum_at_one (m : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), f x m ≥ f 1 m) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_local_minimum_at_one_l2344_234463


namespace NUMINAMATH_CALUDE_shortest_side_in_triangle_l2344_234491

/-- Given a triangle with side lengths a, b, and c, if a^2 + b^2 > 5c^2, then c is the length of the shortest side. -/
theorem shortest_side_in_triangle (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_inequality : a^2 + b^2 > 5*c^2) : 
  c ≤ a ∧ c ≤ b :=
sorry

end NUMINAMATH_CALUDE_shortest_side_in_triangle_l2344_234491


namespace NUMINAMATH_CALUDE_min_consecutive_even_numbers_divisible_by_384_l2344_234419

-- Define a function that checks if a number is even
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define a function that generates a list of consecutive even numbers
def consecutiveEvenNumbers (start : ℕ) (count : ℕ) : List ℕ :=
  List.range count |>.map (λ i => start + 2 * i)

-- Define a function that calculates the product of a list of numbers
def productOfList (l : List ℕ) : ℕ :=
  l.foldl (·*·) 1

-- The main theorem
theorem min_consecutive_even_numbers_divisible_by_384 :
  ∀ n : ℕ, n ≥ 7 →
    ∀ start : ℕ, isEven start →
      384 ∣ productOfList (consecutiveEvenNumbers start n) ∧
      ∀ m : ℕ, m < 7 →
        ∃ s : ℕ, isEven s ∧ ¬(384 ∣ productOfList (consecutiveEvenNumbers s m)) :=
by sorry


end NUMINAMATH_CALUDE_min_consecutive_even_numbers_divisible_by_384_l2344_234419


namespace NUMINAMATH_CALUDE_star_equality_implies_x_equals_6_l2344_234466

/-- Binary operation ★ on ordered pairs of integers -/
def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b + d)

/-- Theorem stating that if (3,3) ★ (0,0) = (x,y) ★ (3,2), then x = 6 -/
theorem star_equality_implies_x_equals_6 (x y : ℤ) :
  star 3 3 0 0 = star x y 3 2 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_star_equality_implies_x_equals_6_l2344_234466


namespace NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_neg_two_l2344_234460

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Definition of the first line -/
def line1 (k : ℝ) : Line3D :=
  { point := (-2, 4, 2),
    direction := (1, -k, k) }

/-- Definition of the second line -/
def line2 : Line3D :=
  { point := (0, 2, 3),
    direction := (1, 2, -1) }

/-- Two lines are coplanar if their direction vectors and the vector connecting their points are linearly dependent -/
def are_coplanar (l1 l2 : Line3D) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∧
    a • l1.direction + b • l2.direction = c • (l2.point - l1.point)

/-- Theorem stating that the lines are coplanar if and only if k = -2 -/
theorem lines_coplanar_iff_k_eq_neg_two (k : ℝ) :
  are_coplanar (line1 k) line2 ↔ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_neg_two_l2344_234460


namespace NUMINAMATH_CALUDE_carrie_tshirt_purchase_l2344_234423

/-- The number of t-shirts Carrie bought -/
def num_tshirts : ℕ := 24

/-- The cost of each t-shirt in dollars -/
def cost_per_tshirt : ℚ := 9.95

/-- The total amount Carrie spent in dollars -/
def total_spent : ℚ := 248

/-- Theorem stating that the number of t-shirts Carrie bought is correct -/
theorem carrie_tshirt_purchase : 
  (↑num_tshirts : ℚ) * cost_per_tshirt ≤ total_spent ∧ 
  (↑(num_tshirts + 1) : ℚ) * cost_per_tshirt > total_spent :=
by sorry

end NUMINAMATH_CALUDE_carrie_tshirt_purchase_l2344_234423


namespace NUMINAMATH_CALUDE_equation_solutions_l2344_234408

theorem equation_solutions :
  let f (x : ℝ) := 3 / (Real.sqrt (x - 5) - 7) + 2 / (Real.sqrt (x - 5) - 3) +
                   9 / (Real.sqrt (x - 5) + 3) + 15 / (Real.sqrt (x - 5) + 7)
  ∀ x : ℝ, f x = 0 ↔ x = 54 ∨ x = 846 / 29 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2344_234408


namespace NUMINAMATH_CALUDE_count_propositions_with_connectives_l2344_234444

-- Define a proposition type
inductive Proposition
| feb14_2010 : Proposition
| multiple_10_5 : Proposition
| trapezoid_rectangle : Proposition

-- Define a function to check if a proposition uses a logical connective
def uses_logical_connective (p : Proposition) : Bool :=
  match p with
  | Proposition.feb14_2010 => true  -- Uses "and"
  | Proposition.multiple_10_5 => false
  | Proposition.trapezoid_rectangle => true  -- Uses "not"

-- Define the list of propositions
def propositions : List Proposition :=
  [Proposition.feb14_2010, Proposition.multiple_10_5, Proposition.trapezoid_rectangle]

-- Theorem statement
theorem count_propositions_with_connectives :
  (propositions.filter uses_logical_connective).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_propositions_with_connectives_l2344_234444


namespace NUMINAMATH_CALUDE_james_total_spending_l2344_234430

def club_entry_fee : ℕ := 20
def friends_count : ℕ := 5
def rounds_for_friends : ℕ := 2
def james_drinks : ℕ := 6
def drink_cost : ℕ := 6
def food_cost : ℕ := 14
def tip_percentage : ℚ := 30 / 100

def total_drinks : ℕ := friends_count * rounds_for_friends + james_drinks

def order_cost : ℕ := total_drinks * drink_cost + food_cost

def tip_amount : ℚ := (order_cost : ℚ) * tip_percentage

def total_spending : ℚ := (club_entry_fee : ℚ) + (order_cost : ℚ) + tip_amount

theorem james_total_spending :
  total_spending = 163 := by sorry

end NUMINAMATH_CALUDE_james_total_spending_l2344_234430


namespace NUMINAMATH_CALUDE_cos_difference_x1_x2_l2344_234438

theorem cos_difference_x1_x2 (x₁ x₂ : ℝ) 
  (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < π)
  (h4 : Real.sin (2 * x₁ - π / 3) = 4 / 5)
  (h5 : Real.sin (2 * x₂ - π / 3) = 4 / 5) :
  Real.cos (x₁ - x₂) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_x1_x2_l2344_234438


namespace NUMINAMATH_CALUDE_greatest_integer_c_for_all_real_domain_l2344_234495

theorem greatest_integer_c_for_all_real_domain : 
  (∃ c : ℤ, (∀ x : ℝ, x^2 + c * x + 10 ≠ 0) ∧ 
   (∀ c' : ℤ, c' > c → ∃ x : ℝ, x^2 + c' * x + 10 = 0)) → 
  (∃ c : ℤ, c = 6 ∧ (∀ x : ℝ, x^2 + c * x + 10 ≠ 0) ∧ 
   (∀ c' : ℤ, c' > c → ∃ x : ℝ, x^2 + c' * x + 10 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_c_for_all_real_domain_l2344_234495


namespace NUMINAMATH_CALUDE_marble_problem_l2344_234440

theorem marble_problem (atticus jensen cruz harper : ℕ) : 
  4 * (atticus + jensen + cruz + harper) = 120 →
  atticus = jensen / 2 →
  atticus = 4 →
  jensen = 2 * harper →
  cruz = 14 := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l2344_234440


namespace NUMINAMATH_CALUDE_negation_equivalence_l2344_234429

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 4*x₀ + 1 < 0) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 4*x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2344_234429


namespace NUMINAMATH_CALUDE_intersection_A_B_when_a_neg_one_complement_A_intersect_B_empty_iff_a_gt_three_l2344_234476

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (a - x)}
def B : Set ℝ := {x | x^2 - x - 6 = 0}

-- Part 1
theorem intersection_A_B_when_a_neg_one :
  A (-1) ∩ B = {-2} := by sorry

-- Part 2
theorem complement_A_intersect_B_empty_iff_a_gt_three (a : ℝ) :
  (Set.univ \ A a) ∩ B = ∅ ↔ a > 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_when_a_neg_one_complement_A_intersect_B_empty_iff_a_gt_three_l2344_234476


namespace NUMINAMATH_CALUDE_pyramid_intersection_theorem_l2344_234401

structure Pyramid where
  base : Rectangle
  side_edge : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Given a pyramid with a rectangular base and equal side edges, when a plane intersects
    the side edges cutting off segments a, b, c, and d, the equation 1/a + 1/c = 1/b + 1/d holds. -/
theorem pyramid_intersection_theorem (p : Pyramid) (ha : p.a > 0) (hb : p.b > 0) (hc : p.c > 0) (hd : p.d > 0) :
  1 / p.a + 1 / p.c = 1 / p.b + 1 / p.d := by
  sorry

end NUMINAMATH_CALUDE_pyramid_intersection_theorem_l2344_234401


namespace NUMINAMATH_CALUDE_race_outcomes_l2344_234480

theorem race_outcomes (n : ℕ) (k : ℕ) (h : n = 6 ∧ k = 4) :
  Nat.factorial n / Nat.factorial (n - k) = 360 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_l2344_234480


namespace NUMINAMATH_CALUDE_combined_transformation_correct_l2344_234446

/-- A dilation centered at the origin with scale factor k -/
def dilation (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.diagonal (λ _ => k)

/-- A reflection across the x-axis -/
def reflectionX : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.diagonal (λ i => if i = 0 then 1 else -1)

/-- The combined transformation matrix -/
def combinedTransformation : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.diagonal (λ i => if i = 0 then 5 else -5)

theorem combined_transformation_correct :
  combinedTransformation = reflectionX * dilation 5 := by
  sorry


end NUMINAMATH_CALUDE_combined_transformation_correct_l2344_234446


namespace NUMINAMATH_CALUDE_highest_power_of_three_for_concatenated_range_l2344_234439

def concatenate_range (a b : ℕ) : ℕ := sorry

def highest_power_of_three (n : ℕ) : ℕ := sorry

theorem highest_power_of_three_for_concatenated_range :
  let N := concatenate_range 31 73
  highest_power_of_three N = 1 := by sorry

end NUMINAMATH_CALUDE_highest_power_of_three_for_concatenated_range_l2344_234439


namespace NUMINAMATH_CALUDE_greatest_value_l2344_234422

theorem greatest_value (p : ℝ) (a b c d : ℝ) 
  (h1 : a + 1 = p) 
  (h2 : b - 2 = p) 
  (h3 : c + 3 = p) 
  (h4 : d - 4 = p) : 
  d > a ∧ d > b ∧ d > c :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_l2344_234422


namespace NUMINAMATH_CALUDE_product_mod_500_l2344_234454

theorem product_mod_500 : (1502 * 2021) % 500 = 42 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_500_l2344_234454


namespace NUMINAMATH_CALUDE_r_plus_s_equals_12_l2344_234402

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = -1/2 * x + 8

-- Define points P and Q
def P : ℝ × ℝ := (16, 0)
def Q : ℝ × ℝ := (0, 8)

-- Define point T
def T (r s : ℝ) : ℝ × ℝ := (r, s)

-- Define that T is on line segment PQ
def T_on_PQ (r s : ℝ) : Prop :=
  line_equation r s ∧ 0 ≤ r ∧ r ≤ 16

-- Define the area of triangle POQ
def area_POQ : ℝ := 64

-- Define the area of triangle TOP
def area_TOP (s : ℝ) : ℝ := 8 * s

-- Theorem statement
theorem r_plus_s_equals_12 (r s : ℝ) :
  T_on_PQ r s → area_POQ = 2 * area_TOP s → r + s = 12 :=
sorry

end NUMINAMATH_CALUDE_r_plus_s_equals_12_l2344_234402


namespace NUMINAMATH_CALUDE_sin_two_phi_l2344_234490

theorem sin_two_phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_phi_l2344_234490


namespace NUMINAMATH_CALUDE_cut_pentagon_area_l2344_234468

/-- Represents a pentagon created by cutting a triangular corner from a rectangular sheet. -/
structure CutPentagon where
  sides : Finset ℕ
  area : ℕ

/-- The theorem stating that a pentagon with specific side lengths has an area of 770. -/
theorem cut_pentagon_area : ∃ (p : CutPentagon), p.sides = {14, 21, 22, 28, 35} ∧ p.area = 770 := by
  sorry

end NUMINAMATH_CALUDE_cut_pentagon_area_l2344_234468


namespace NUMINAMATH_CALUDE_centripetal_acceleration_proportionality_l2344_234404

/-- Centripetal acceleration proportionality -/
theorem centripetal_acceleration_proportionality
  (a v r ω T : ℝ) (h1 : a = v^2 / r) (h2 : a = r * ω^2) (h3 : a = 4 * Real.pi^2 * r / T^2) :
  (∃ k1 : ℝ, a = k1 * (v^2 / r)) ∧
  (∃ k2 : ℝ, a = k2 * (r * ω^2)) ∧
  (∃ k3 : ℝ, a = k3 * (r / T^2)) :=
by sorry

end NUMINAMATH_CALUDE_centripetal_acceleration_proportionality_l2344_234404


namespace NUMINAMATH_CALUDE_merry_go_round_time_l2344_234418

/-- The time taken for the second horse to travel the same distance as the first horse -/
theorem merry_go_round_time (r₁ r₂ : ℝ) (rev : ℕ) (v₁ v₂ : ℝ) : 
  r₁ = 30 → r₂ = 15 → rev = 40 → v₁ = 3 → v₂ = 6 → 
  (2 * Real.pi * r₂ * (rev * 2 * Real.pi * r₁) / v₂) = (400 * Real.pi) := by
  sorry

#check merry_go_round_time

end NUMINAMATH_CALUDE_merry_go_round_time_l2344_234418


namespace NUMINAMATH_CALUDE_matthews_water_bottle_size_l2344_234415

/-- Calculates the size of Matthew's water bottle based on his drinking habits -/
theorem matthews_water_bottle_size 
  (glasses_per_day : ℕ) 
  (ounces_per_glass : ℕ) 
  (fills_per_week : ℕ) 
  (h1 : glasses_per_day = 4)
  (h2 : ounces_per_glass = 5)
  (h3 : fills_per_week = 4) :
  (glasses_per_day * ounces_per_glass * 7) / fills_per_week = 35 := by
  sorry

end NUMINAMATH_CALUDE_matthews_water_bottle_size_l2344_234415


namespace NUMINAMATH_CALUDE_arrangement_count_l2344_234497

/-- Represents the number of students -/
def num_students : ℕ := 4

/-- Represents the number of schools -/
def num_schools : ℕ := 3

/-- Represents the total number of arrangements without restrictions -/
def total_arrangements : ℕ := (num_students.choose 2) * num_schools.factorial

/-- Represents the number of arrangements where A and B are in the same school -/
def arrangements_ab_together : ℕ := num_schools.factorial

/-- Represents the number of valid arrangements -/
def valid_arrangements : ℕ := total_arrangements - arrangements_ab_together

theorem arrangement_count : valid_arrangements = 30 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_l2344_234497


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2344_234488

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 - 2*x + 3

-- Theorem statement
theorem quadratic_function_properties :
  (∃ (a : ℝ), f x = a * (x + 1)^2 + 4) ∧ -- Vertex form with vertex at (-1, 4)
  f 2 = -5 := by -- Passes through (2, -5)
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2344_234488


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2344_234453

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 9*x + 18 = 0

-- Define the isosceles triangle
structure IsoscelesTriangle :=
  (base : ℝ)
  (leg : ℝ)
  (base_is_root : quadratic_equation base)
  (leg_is_root : quadratic_equation leg)

-- Theorem statement
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : 
  t.base + 2 * t.leg = 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2344_234453


namespace NUMINAMATH_CALUDE_quadruple_count_l2344_234484

/-- The number of ordered quadruples of positive even integers that sum to 104 -/
def n : ℕ := sorry

/-- Predicate for a quadruple of positive even integers -/
def is_valid_quadruple (x₁ x₂ x₃ x₄ : ℕ) : Prop :=
  x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
  Even x₁ ∧ Even x₂ ∧ Even x₃ ∧ Even x₄ ∧
  x₁ + x₂ + x₃ + x₄ = 104

/-- The main theorem stating that n/100 equals 208.25 -/
theorem quadruple_count : (n : ℚ) / 100 = 208.25 := by sorry

end NUMINAMATH_CALUDE_quadruple_count_l2344_234484


namespace NUMINAMATH_CALUDE_sophies_daily_oranges_l2344_234492

/-- The number of oranges Sophie's mom gives her every day -/
def sophies_oranges : ℕ := 20

/-- The number of grapes Hannah eats per day -/
def hannahs_grapes : ℕ := 40

/-- The number of days in the observation period -/
def observation_days : ℕ := 30

/-- The total number of fruits eaten by Sophie and Hannah during the observation period -/
def total_fruits : ℕ := 1800

/-- Theorem stating that Sophie's mom gives her 20 oranges per day -/
theorem sophies_daily_oranges :
  sophies_oranges * observation_days + hannahs_grapes * observation_days = total_fruits :=
by sorry

end NUMINAMATH_CALUDE_sophies_daily_oranges_l2344_234492


namespace NUMINAMATH_CALUDE_initial_snack_eaters_l2344_234428

/-- The number of snack eaters after a series of events -/
def final_snack_eaters (S : ℕ) : ℕ :=
  ((S + 20) / 2 + 10 - 30) / 2

/-- Theorem stating that the initial number of snack eaters was 100 -/
theorem initial_snack_eaters :
  ∃ S : ℕ, final_snack_eaters S = 20 ∧ S = 100 := by
  sorry

end NUMINAMATH_CALUDE_initial_snack_eaters_l2344_234428


namespace NUMINAMATH_CALUDE_coin_collection_value_johns_collection_value_l2344_234481

/-- Proves the value of a coin collection given certain conditions -/
theorem coin_collection_value
  (total_coins : ℕ)
  (sample_coins : ℕ)
  (sample_value : ℚ)
  (h1 : total_coins = 24)
  (h2 : sample_coins = 8)
  (h3 : sample_value = 20)
  : ℚ
:=
by
  -- The value of the entire collection
  sorry

/-- The main theorem stating the value of John's coin collection -/
theorem johns_collection_value : coin_collection_value 24 8 20 rfl rfl rfl = 60 := by sorry

end NUMINAMATH_CALUDE_coin_collection_value_johns_collection_value_l2344_234481


namespace NUMINAMATH_CALUDE_x_gt_y_necessary_not_sufficient_l2344_234434

theorem x_gt_y_necessary_not_sufficient (x y : ℝ) (hx : x > 0) :
  (∀ y, x > |y| → x > y) ∧ ¬(∀ y, x > y → x > |y|) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_y_necessary_not_sufficient_l2344_234434


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2344_234461

theorem complex_magnitude_problem (w z : ℂ) :
  w * z = 20 - 15 * I ∧ Complex.abs w = Real.sqrt 20 →
  Complex.abs z = (5 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2344_234461


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2344_234486

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 5) / 10 = 5 / (x - 10) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2344_234486


namespace NUMINAMATH_CALUDE_factorization_equality_l2344_234445

theorem factorization_equality (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2*b) * (a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2344_234445


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2344_234464

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 →
  (x + 1 - 3 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1)) = (x + 2) / (x - 2) ∧
  (0 + 2) / (0 - 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2344_234464


namespace NUMINAMATH_CALUDE_volunteer_selection_count_l2344_234420

/-- The number of ways to select 3 volunteers from 5 boys and 2 girls, with at least 1 girl selected -/
def select_volunteers (num_boys : ℕ) (num_girls : ℕ) (total_selected : ℕ) : ℕ :=
  Nat.choose num_girls 1 * Nat.choose num_boys 2 +
  Nat.choose num_girls 2 * Nat.choose num_boys 1

/-- Theorem stating that the number of ways to select 3 volunteers from 5 boys and 2 girls, 
    with at least 1 girl selected, is equal to 25 -/
theorem volunteer_selection_count :
  select_volunteers 5 2 3 = 25 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_selection_count_l2344_234420


namespace NUMINAMATH_CALUDE_sqrt_k_squared_minus_pk_integer_l2344_234494

theorem sqrt_k_squared_minus_pk_integer (p : ℕ) (hp : Prime p) :
  ∀ k : ℤ, (∃ n : ℕ+, (k^2 - p * k : ℤ) = n^2) ↔ 
    (p ≠ 2 ∧ (k = ((p + 1) / 2)^2 ∨ k = -((p - 1) / 2)^2)) ∨ 
    (p = 2 ∧ False) := by
  sorry

#check sqrt_k_squared_minus_pk_integer

end NUMINAMATH_CALUDE_sqrt_k_squared_minus_pk_integer_l2344_234494


namespace NUMINAMATH_CALUDE_cone_volume_l2344_234449

/-- The volume of a cone with slant height 5 and lateral area 20π is 16π -/
theorem cone_volume (s : ℝ) (lateral_area : ℝ) (h : s = 5) (h' : lateral_area = 20 * Real.pi) :
  (1 / 3 : ℝ) * Real.pi * (lateral_area / (Real.pi * s))^2 * Real.sqrt (s^2 - (lateral_area / (Real.pi * s))^2) = 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l2344_234449


namespace NUMINAMATH_CALUDE_even_monotone_increasing_range_l2344_234426

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem even_monotone_increasing_range 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0)) :
  {x : ℝ | f x < f 1} = Set.Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_even_monotone_increasing_range_l2344_234426


namespace NUMINAMATH_CALUDE_original_selling_price_l2344_234450

theorem original_selling_price (P : ℝ) (S : ℝ) (S_new : ℝ) : 
  S = 1.1 * P →
  S_new = 1.3 * (0.9 * P) →
  S_new = S + 35 →
  S = 550 := by
sorry

end NUMINAMATH_CALUDE_original_selling_price_l2344_234450


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l2344_234471

theorem binomial_coefficient_ratio (a b : ℕ) : 
  (a = Nat.choose 6 3) → 
  (b = Nat.choose 6 4 * 2^4) → 
  (∀ k, 0 ≤ k ∧ k ≤ 6 → Nat.choose 6 k ≤ a) → 
  (∀ k, 0 ≤ k ∧ k ≤ 6 → Nat.choose 6 k * 2^k ≤ b) → 
  b / a = 12 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l2344_234471


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l2344_234448

/-- Given two similar right triangles, where one has side lengths 6, 8, and 10,
    and the other has its shortest side equal to 15,
    prove that the perimeter of the larger triangle is 60. -/
theorem similar_triangle_perimeter :
  ∀ (a b c d e f : ℝ),
  a = 6 ∧ b = 8 ∧ c = 10 ∧  -- First triangle side lengths
  d = 15 ∧                  -- Shortest side of the similar triangle
  a^2 + b^2 = c^2 ∧         -- Pythagorean theorem for the first triangle
  (d / a) * b = e ∧         -- Similar triangles proportion for the second side
  (d / a) * c = f →         -- Similar triangles proportion for the third side
  d + e + f = 60 :=
by
  sorry


end NUMINAMATH_CALUDE_similar_triangle_perimeter_l2344_234448


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2344_234485

theorem complex_fraction_equality : 
  let x : ℂ := (1 + Complex.I * Real.sqrt 3) / 3
  1 / (x^2 + x) = 9/76 - (45 * Complex.I * Real.sqrt 3)/76 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2344_234485


namespace NUMINAMATH_CALUDE_quadratic_roots_transformation_l2344_234467

/-- Given that u and v are roots of 2x^2 + 5x + 3 = 0, prove that x^2 - x + 6 = 0 has roots 2u + 3 and 2v + 3 -/
theorem quadratic_roots_transformation (u v : ℝ) :
  (2 * u^2 + 5 * u + 3 = 0) →
  (2 * v^2 + 5 * v + 3 = 0) →
  ∀ x : ℝ, (x^2 - x + 6 = 0) ↔ (x = 2*u + 3 ∨ x = 2*v + 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_transformation_l2344_234467


namespace NUMINAMATH_CALUDE_estimate_students_in_range_l2344_234482

/-- Given a histogram of student heights with two adjacent rectangles, 
    estimate the number of students in the combined range. -/
theorem estimate_students_in_range 
  (total_students : ℕ) 
  (rectangle_width : ℝ) 
  (height_a : ℝ) 
  (height_b : ℝ) 
  (h_total : total_students = 1500)
  (h_width : rectangle_width = 5) :
  (rectangle_width * height_a + rectangle_width * height_b) * total_students = 
    7500 * (height_a + height_b) := by
  sorry

end NUMINAMATH_CALUDE_estimate_students_in_range_l2344_234482


namespace NUMINAMATH_CALUDE_evaluate_F_of_f_l2344_234451

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 - 1
def F (a b : ℝ) : ℝ := b^3 - a

-- State the theorem
theorem evaluate_F_of_f : F 2 (f 3) = 510 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_F_of_f_l2344_234451


namespace NUMINAMATH_CALUDE_green_pill_cost_proof_l2344_234443

/-- The cost of a green pill in dollars -/
def green_pill_cost : ℚ := 41 / 3

/-- The cost of a pink pill in dollars -/
def pink_pill_cost : ℚ := green_pill_cost - 1

/-- The number of days in the treatment period -/
def treatment_days : ℕ := 21

/-- The total cost of medication for the treatment period -/
def total_cost : ℚ := 819

theorem green_pill_cost_proof :
  (green_pill_cost + 2 * pink_pill_cost) * treatment_days = total_cost :=
sorry

end NUMINAMATH_CALUDE_green_pill_cost_proof_l2344_234443


namespace NUMINAMATH_CALUDE_chocolate_ratio_simplification_l2344_234435

theorem chocolate_ratio_simplification :
  let white_chocolate : ℕ := 20
  let dark_chocolate : ℕ := 15
  let gcd := Nat.gcd white_chocolate dark_chocolate
  (white_chocolate / gcd : ℚ) / (dark_chocolate / gcd : ℚ) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_ratio_simplification_l2344_234435


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_l2344_234470

/-- Given a quadratic function f(x) = 2x^2 - x + 7, when shifted 6 units to the right,
    the resulting function g(x) = ax^2 + bx + c satisfies a + b + c = 62 -/
theorem quadratic_shift_sum (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 2 * x^2 - x + 7) →
  (∀ x, g x = f (x - 6)) →
  (∀ x, g x = a * x^2 + b * x + c) →
  a + b + c = 62 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_sum_l2344_234470


namespace NUMINAMATH_CALUDE_height_increase_per_decade_l2344_234459

/-- Proves that the height increase per decade is 90 meters, given that the total increase in height over 2 centuries is 1800 meters. -/
theorem height_increase_per_decade : 
  ∀ (increase_per_decade : ℝ),
  (20 * increase_per_decade = 1800) →
  increase_per_decade = 90 := by
sorry

end NUMINAMATH_CALUDE_height_increase_per_decade_l2344_234459


namespace NUMINAMATH_CALUDE_compound_weight_l2344_234442

/-- The atomic weight of Aluminum-27 in atomic mass units -/
def aluminum_weight : ℕ := 27

/-- The atomic weight of Iodine-127 in atomic mass units -/
def iodine_weight : ℕ := 127

/-- The atomic weight of Oxygen-16 in atomic mass units -/
def oxygen_weight : ℕ := 16

/-- The number of Aluminum-27 atoms in the compound -/
def aluminum_count : ℕ := 1

/-- The number of Iodine-127 atoms in the compound -/
def iodine_count : ℕ := 3

/-- The number of Oxygen-16 atoms in the compound -/
def oxygen_count : ℕ := 2

/-- The molecular weight of the compound -/
def molecular_weight : ℕ := 
  aluminum_count * aluminum_weight + 
  iodine_count * iodine_weight + 
  oxygen_count * oxygen_weight

theorem compound_weight : molecular_weight = 440 := by
  sorry

end NUMINAMATH_CALUDE_compound_weight_l2344_234442


namespace NUMINAMATH_CALUDE_heartsuit_ratio_l2344_234477

-- Define the ♡ operation
def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

-- Theorem statement
theorem heartsuit_ratio : (heartsuit 3 5) / (heartsuit 5 3) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_ratio_l2344_234477


namespace NUMINAMATH_CALUDE_binomial_expansion_ratio_l2344_234462

theorem binomial_expansion_ratio (n : ℕ) (a b c : ℝ) :
  n ≥ 3 →
  (∀ x : ℝ, (x + 2)^n = x^n + a * x^3 + b * x^2 + c * x + 2^n) →
  a / b = 3 / 2 →
  n = 11 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_ratio_l2344_234462


namespace NUMINAMATH_CALUDE_max_perimeter_of_special_triangle_l2344_234469

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a * Real.sin t.A - t.c * Real.sin t.C = (t.a - t.b) * Real.sin t.B

/-- The perimeter of the triangle -/
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- The theorem to be proved -/
theorem max_perimeter_of_special_triangle :
  ∀ t : Triangle,
    satisfiesCondition t →
    t.c = Real.sqrt 3 →
    ∃ maxPerimeter : ℝ,
      maxPerimeter = 3 * Real.sqrt 3 ∧
      ∀ t' : Triangle,
        satisfiesCondition t' →
        t'.c = Real.sqrt 3 →
        perimeter t' ≤ maxPerimeter :=
by sorry

end NUMINAMATH_CALUDE_max_perimeter_of_special_triangle_l2344_234469


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_l2344_234437

/-- The general solution to the differential equation (4y - 3x - 5)y' + 7x - 3y + 2 = 0 -/
def general_solution (x y : ℝ) (C : ℝ) : Prop :=
  2 * y^2 - 3 * x * y + (7/2) * x^2 + 2 * x - 5 * y = C

/-- The differential equation (4y - 3x - 5)y' + 7x - 3y + 2 = 0 -/
def differential_equation (x y : ℝ) (y' : ℝ → ℝ) : Prop :=
  (4 * y - 3 * x - 5) * (y' x) + 7 * x - 3 * y + 2 = 0

theorem solution_satisfies_equation :
  ∀ (x y : ℝ) (C : ℝ),
  general_solution x y C →
  ∃ (y' : ℝ → ℝ), differential_equation x y y' :=
sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_l2344_234437


namespace NUMINAMATH_CALUDE_emily_age_l2344_234406

theorem emily_age :
  ∀ (e m : ℕ),
  e = m - 18 →
  e + m = 54 →
  e = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_emily_age_l2344_234406


namespace NUMINAMATH_CALUDE_soccer_match_ratio_l2344_234425

def soccer_match (kickers_second_period : ℕ) : Prop :=
  let kickers_first_period : ℕ := 2
  let spiders_first_period : ℕ := kickers_first_period / 2
  let spiders_second_period : ℕ := 2 * kickers_second_period
  let total_goals : ℕ := 15
  (kickers_first_period + kickers_second_period + spiders_first_period + spiders_second_period = total_goals) ∧
  (kickers_second_period : ℚ) / (kickers_first_period : ℚ) = 2 / 1

theorem soccer_match_ratio : ∃ (kickers_second_period : ℕ), soccer_match kickers_second_period := by
  sorry

end NUMINAMATH_CALUDE_soccer_match_ratio_l2344_234425


namespace NUMINAMATH_CALUDE_inheritance_satisfies_tax_conditions_inheritance_uniqueness_l2344_234417

/-- The inheritance amount that satisfies the tax conditions -/
def inheritance : ℝ := 41379

/-- The total tax paid -/
def total_tax : ℝ := 15000

/-- Federal tax rate -/
def federal_tax_rate : ℝ := 0.25

/-- State tax rate -/
def state_tax_rate : ℝ := 0.15

/-- Theorem stating that the inheritance amount satisfies the tax conditions -/
theorem inheritance_satisfies_tax_conditions :
  federal_tax_rate * inheritance + 
  state_tax_rate * (inheritance - federal_tax_rate * inheritance) = 
  total_tax := by sorry

/-- Theorem stating that the inheritance amount is unique -/
theorem inheritance_uniqueness (x : ℝ) :
  federal_tax_rate * x + 
  state_tax_rate * (x - federal_tax_rate * x) = 
  total_tax →
  x = inheritance := by sorry

end NUMINAMATH_CALUDE_inheritance_satisfies_tax_conditions_inheritance_uniqueness_l2344_234417


namespace NUMINAMATH_CALUDE_average_transformation_l2344_234424

theorem average_transformation (t b c : ℝ) :
  (t + b + c + 14 + 15) / 5 = 12 →
  (t + b + c + 29) / 4 = 15 := by
sorry

end NUMINAMATH_CALUDE_average_transformation_l2344_234424


namespace NUMINAMATH_CALUDE_largest_divisor_of_60_36_divisible_by_3_l2344_234433

theorem largest_divisor_of_60_36_divisible_by_3 :
  ∃ (n : ℕ), n > 0 ∧ n ∣ 60 ∧ n ∣ 36 ∧ 3 ∣ n ∧
  ∀ (m : ℕ), m > n → (m ∣ 60 ∧ m ∣ 36 ∧ 3 ∣ m) → False :=
by
  use 12
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_60_36_divisible_by_3_l2344_234433


namespace NUMINAMATH_CALUDE_percentage_difference_l2344_234413

theorem percentage_difference : (45 / 100 * 60) - (35 / 100 * 40) = 13 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2344_234413


namespace NUMINAMATH_CALUDE_pickle_barrel_problem_l2344_234431

theorem pickle_barrel_problem (B M T G S : ℚ) : 
  M + T + G + S = B →
  B - M / 2 = B / 10 →
  B - T / 2 = B / 8 →
  B - G / 2 = B / 4 →
  B - S / 2 = B / 40 := by
sorry

end NUMINAMATH_CALUDE_pickle_barrel_problem_l2344_234431


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2344_234409

def M : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem sufficient_not_necessary : 
  (∀ a, a ∈ M → a ∈ N) ∧ (∃ a, a ∈ N ∧ a ∉ M) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2344_234409


namespace NUMINAMATH_CALUDE_circle1_correct_circle2_correct_l2344_234403

-- Define the points
def M : ℝ × ℝ := (-5, 3)
def A1 : ℝ × ℝ := (-8, -1)
def A2 : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (-1, 3)
def C : ℝ × ℝ := (2, 6)

-- Define the circle equations
def circle1_eq (x y : ℝ) : Prop := (x + 5)^2 + (y - 3)^2 = 25
def circle2_eq (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 5

-- Theorem for the first circle
theorem circle1_correct : 
  (∀ x y : ℝ, circle1_eq x y ↔ 
    ((x, y) = M ∨ (∃ t : ℝ, (x, y) = M + t • (A1 - M) ∧ 0 < t ∧ t < 1))) := by sorry

-- Theorem for the second circle
theorem circle2_correct : 
  (∀ x y : ℝ, circle2_eq x y ↔ 
    ((x, y) = A2 ∨ (x, y) = B ∨ (x, y) = C ∨ 
    (∃ t : ℝ, ((x, y) = A2 + t • (B - A2) ∨ 
               (x, y) = B + t • (C - B) ∨ 
               (x, y) = C + t • (A2 - C)) ∧ 
    0 < t ∧ t < 1))) := by sorry

end NUMINAMATH_CALUDE_circle1_correct_circle2_correct_l2344_234403


namespace NUMINAMATH_CALUDE_exists_perfect_square_2022_not_perfect_square_for_a_2_l2344_234441

-- Part (a)
theorem exists_perfect_square_2022 : ∃ n : ℕ, ∃ k : ℕ, n * (n + 2022) + 2 = k^2 := by
  sorry

-- Part (b)
theorem not_perfect_square_for_a_2 : ∀ n : ℕ, ¬∃ k : ℕ, n * (n + 2) + 2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_perfect_square_2022_not_perfect_square_for_a_2_l2344_234441


namespace NUMINAMATH_CALUDE_fraction_equality_l2344_234457

theorem fraction_equality : 
  (14/10 : ℚ) = 7/5 ∧ 
  (1 + 2/5 : ℚ) = 7/5 ∧ 
  (1 + 7/35 : ℚ) ≠ 7/5 ∧ 
  (1 + 4/20 : ℚ) ≠ 7/5 ∧ 
  (1 + 3/15 : ℚ) ≠ 7/5 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2344_234457


namespace NUMINAMATH_CALUDE_grace_constant_reading_rate_l2344_234474

/-- Grace's reading rate in pages per hour -/
def reading_rate (pages : ℕ) (hours : ℕ) : ℚ :=
  pages / hours

theorem grace_constant_reading_rate :
  let rate1 := reading_rate 200 20
  let rate2 := reading_rate 250 25
  rate1 = rate2 ∧ rate1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_grace_constant_reading_rate_l2344_234474


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2344_234411

/-- Given a quadratic equation with coefficients a, b, and c, returns true if it has exactly one solution -/
def has_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

theorem quadratic_equation_solution (b : ℝ) :
  has_one_solution 3 15 b →
  b + 3 = 36 →
  b > 3 →
  b = 33 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2344_234411


namespace NUMINAMATH_CALUDE_min_difference_of_sine_bounds_l2344_234421

open Real

theorem min_difference_of_sine_bounds (a b : ℝ) :
  (∀ x ∈ Set.Ioo 0 (π / 2), a * x < sin x ∧ sin x < b * x) →
  1 - 2 / π ≤ b - a :=
by sorry

end NUMINAMATH_CALUDE_min_difference_of_sine_bounds_l2344_234421


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2344_234427

theorem quadratic_factorization :
  ∃ (x : ℝ), x^2 + 6*x - 2 = 0 ↔ ∃ (x : ℝ), (x + 3)^2 = 11 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2344_234427


namespace NUMINAMATH_CALUDE_ball_placement_count_ball_placement_count_is_30_l2344_234475

/-- Number of ways to place 4 balls in 3 boxes with constraints -/
theorem ball_placement_count : ℕ :=
  let total_balls : ℕ := 4
  let num_boxes : ℕ := 3
  let ways_to_choose_two : ℕ := Nat.choose total_balls 2
  let ways_to_arrange_three : ℕ := Nat.factorial num_boxes
  let invalid_arrangements : ℕ := 6
  ways_to_choose_two * ways_to_arrange_three - invalid_arrangements

/-- Proof that the number of valid arrangements is 30 -/
theorem ball_placement_count_is_30 : ball_placement_count = 30 := by
  sorry

end NUMINAMATH_CALUDE_ball_placement_count_ball_placement_count_is_30_l2344_234475


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2344_234473

theorem right_triangle_hypotenuse (x y z : ℝ) : 
  x > 0 → 
  y > 0 → 
  z > 0 → 
  y = 3 * x - 3 → 
  (1 / 2) * x * y = 72 → 
  x^2 + y^2 = z^2 → 
  z = Real.sqrt 505 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2344_234473


namespace NUMINAMATH_CALUDE_students_in_both_clubs_l2344_234412

/-- The number of students in both drama and science clubs at Lincoln High School -/
theorem students_in_both_clubs 
  (total_students : ℕ)
  (drama_club : ℕ)
  (science_club : ℕ)
  (either_club : ℕ)
  (h1 : total_students = 250)
  (h2 : drama_club = 100)
  (h3 : science_club = 130)
  (h4 : either_club = 210) :
  drama_club + science_club - either_club = 20 := by
sorry

end NUMINAMATH_CALUDE_students_in_both_clubs_l2344_234412


namespace NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l2344_234483

theorem smallest_solution_absolute_value_equation :
  let f : ℝ → ℝ := λ x => x * |x| - 3 * x + 2
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → x ≤ y ∧ x = (-3 - Real.sqrt 17) / 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l2344_234483


namespace NUMINAMATH_CALUDE_first_shift_members_l2344_234405

-- Define the number of shifts
def num_shifts : ℕ := 3

-- Define the total number of workers in the company
def total_workers (shift1 : ℕ) : ℕ := shift1 + 50 + 40

-- Define the participation rate for each shift
def participation_rate1 : ℚ := 1/5
def participation_rate2 : ℚ := 2/5
def participation_rate3 : ℚ := 1/10

-- Define the total number of participants in the pension program
def total_participants (shift1 : ℕ) : ℚ :=
  participation_rate1 * shift1 + participation_rate2 * 50 + participation_rate3 * 40

-- State the theorem
theorem first_shift_members :
  ∃ (shift1 : ℕ), 
    shift1 > 0 ∧
    (total_participants shift1) / (total_workers shift1) = 6/25 ∧
    shift1 = 60 :=
by sorry

end NUMINAMATH_CALUDE_first_shift_members_l2344_234405


namespace NUMINAMATH_CALUDE_joes_steakhouse_wages_l2344_234498

/-- Proves that the manager's hourly wage is $6.50 given the conditions from Joe's Steakhouse --/
theorem joes_steakhouse_wages (manager_wage dishwasher_wage chef_wage : ℝ) :
  chef_wage = dishwasher_wage + 0.2 * dishwasher_wage →
  dishwasher_wage = 0.5 * manager_wage →
  chef_wage = manager_wage - 2.6 →
  manager_wage = 6.5 := by
sorry

end NUMINAMATH_CALUDE_joes_steakhouse_wages_l2344_234498


namespace NUMINAMATH_CALUDE_problem_statement_l2344_234455

theorem problem_statement (a : ℝ) (h : (a + 1/a)^3 = 3) :
  a^4 + 1/a^4 = Real.rpow 9 (1/3) - 4 * Real.rpow 3 (1/3) + 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2344_234455


namespace NUMINAMATH_CALUDE_smallest_value_of_expression_l2344_234456

theorem smallest_value_of_expression (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a ≠ 0) :
  ((a + b)^2 + (b + c)^2 + (c + a)^2) / a^2 ≥ 6 ∧
  ∃ (a' b' c' : ℝ), a' > b' ∧ b' > c' ∧ a' ≠ 0 ∧
    ((a' + b')^2 + (b' + c')^2 + (c' + a')^2) / a'^2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_expression_l2344_234456


namespace NUMINAMATH_CALUDE_first_part_speed_l2344_234452

theorem first_part_speed (total_distance : ℝ) (first_part_distance : ℝ) (second_part_speed : ℝ) (average_speed : ℝ) 
  (h1 : total_distance = 60)
  (h2 : first_part_distance = 12)
  (h3 : second_part_speed = 48)
  (h4 : average_speed = 40)
  (h5 : total_distance = first_part_distance + (total_distance - first_part_distance))
  (h6 : average_speed = total_distance / (first_part_distance / v + (total_distance - first_part_distance) / second_part_speed)) :
  v = 24 := by
  sorry

end NUMINAMATH_CALUDE_first_part_speed_l2344_234452


namespace NUMINAMATH_CALUDE_polynomial_sign_intervals_l2344_234410

theorem polynomial_sign_intervals (x : ℝ) :
  x > 0 → ((x - 1) * (x - 2) * (x - 3) < 0 ↔ (x > 0 ∧ x < 1) ∨ (x > 2 ∧ x < 3)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_sign_intervals_l2344_234410


namespace NUMINAMATH_CALUDE_ear_muffs_proof_l2344_234493

/-- The number of ear muffs bought before December -/
def ear_muffs_before_december : ℕ := 7790 - 6444

/-- The total number of ear muffs bought -/
def total_ear_muffs : ℕ := 7790

/-- The number of ear muffs bought during December -/
def ear_muffs_during_december : ℕ := 6444

theorem ear_muffs_proof :
  ear_muffs_before_december = 1346 ∧
  total_ear_muffs = ear_muffs_before_december + ear_muffs_during_december :=
by sorry

end NUMINAMATH_CALUDE_ear_muffs_proof_l2344_234493


namespace NUMINAMATH_CALUDE_divide_and_add_problem_l2344_234400

theorem divide_and_add_problem (x : ℝ) : (48 / x) + 7 = 15 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_divide_and_add_problem_l2344_234400
