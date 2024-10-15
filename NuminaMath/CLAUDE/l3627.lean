import Mathlib

namespace NUMINAMATH_CALUDE_sqrt6_custom_op_approx_l3627_362779

/-- Custom binary operation ¤ -/
def custom_op (x y : ℝ) : ℝ := x^2 + y^2 + 12

/-- Theorem stating that √6 ¤ √6 ≈ 23.999999999999996 -/
theorem sqrt6_custom_op_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1e-14 ∧ |custom_op (Real.sqrt 6) (Real.sqrt 6) - 23.999999999999996| < ε :=
sorry

end NUMINAMATH_CALUDE_sqrt6_custom_op_approx_l3627_362779


namespace NUMINAMATH_CALUDE_total_animals_count_l3627_362710

/-- The total number of dangerous animals pointed out by the teacher in the swamp area -/
def total_dangerous_animals : ℕ := 250

/-- The number of crocodiles observed -/
def crocodiles : ℕ := 42

/-- The number of alligators observed -/
def alligators : ℕ := 35

/-- The number of vipers observed -/
def vipers : ℕ := 10

/-- The number of water moccasins observed -/
def water_moccasins : ℕ := 28

/-- The number of cottonmouth snakes observed -/
def cottonmouth_snakes : ℕ := 15

/-- The number of piranha fish in the school -/
def piranha_fish : ℕ := 120

/-- Theorem stating that the total number of dangerous animals is the sum of all observed species -/
theorem total_animals_count :
  total_dangerous_animals = crocodiles + alligators + vipers + water_moccasins + cottonmouth_snakes + piranha_fish :=
by
  sorry

end NUMINAMATH_CALUDE_total_animals_count_l3627_362710


namespace NUMINAMATH_CALUDE_rug_purchase_price_l3627_362700

/-- Proves that the purchase price per rug is $40 given the selling price, number of rugs, and total profit -/
theorem rug_purchase_price
  (selling_price : ℝ)
  (num_rugs : ℕ)
  (total_profit : ℝ)
  (h1 : selling_price = 60)
  (h2 : num_rugs = 20)
  (h3 : total_profit = 400) :
  (selling_price * num_rugs - total_profit) / num_rugs = 40 := by
  sorry

end NUMINAMATH_CALUDE_rug_purchase_price_l3627_362700


namespace NUMINAMATH_CALUDE_expression_evaluation_l3627_362741

theorem expression_evaluation (a b : ℚ) (h1 : a = -3) (h2 : b = 1/3) :
  (a - 3*b) * (a + 3*b) + (a - 3*b)^2 = 24 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3627_362741


namespace NUMINAMATH_CALUDE_multiplication_of_powers_l3627_362750

theorem multiplication_of_powers (a : ℝ) : 2 * a^2 * a = 2 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_of_powers_l3627_362750


namespace NUMINAMATH_CALUDE_allyson_age_l3627_362799

theorem allyson_age (hiram_age : ℕ) (allyson_age : ℕ) 
  (h1 : hiram_age = 40)
  (h2 : hiram_age + 12 = 2 * allyson_age - 4) :
  allyson_age = 28 := by
  sorry

end NUMINAMATH_CALUDE_allyson_age_l3627_362799


namespace NUMINAMATH_CALUDE_vec_b_is_correct_l3627_362766

def vec_a : ℝ × ℝ := (6, -8)
def vec_b : ℝ × ℝ := (-4, -3)
def vec_c : ℝ × ℝ := (1, 0)

theorem vec_b_is_correct : 
  (vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 = 0) ∧ 
  (vec_b.1^2 + vec_b.2^2 = 25) ∧
  (vec_b.1 * vec_c.1 + vec_b.2 * vec_c.2 < 0) ∧
  (∀ x y : ℝ, (vec_a.1 * x + vec_a.2 * y = 0) ∧ 
              (x^2 + y^2 = 25) ∧ 
              (x * vec_c.1 + y * vec_c.2 < 0) → 
              (x, y) = vec_b) :=
by sorry

end NUMINAMATH_CALUDE_vec_b_is_correct_l3627_362766


namespace NUMINAMATH_CALUDE_lcm_48_180_l3627_362746

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_180_l3627_362746


namespace NUMINAMATH_CALUDE_david_pushups_count_l3627_362735

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 7

/-- The additional number of push-ups David did compared to Zachary -/
def david_extra_pushups : ℕ := 30

/-- The number of push-ups David did -/
def david_pushups : ℕ := zachary_pushups + david_extra_pushups

theorem david_pushups_count : david_pushups = 37 := by sorry

end NUMINAMATH_CALUDE_david_pushups_count_l3627_362735


namespace NUMINAMATH_CALUDE_newspaper_delivery_difference_l3627_362751

/-- Calculates the difference in monthly newspaper deliveries between Miranda and Jake -/
def monthly_delivery_difference (jake_weekly : ℕ) (miranda_multiplier : ℕ) (weeks_per_month : ℕ) : ℕ :=
  (jake_weekly * miranda_multiplier - jake_weekly) * weeks_per_month

/-- Proves that the difference in monthly newspaper deliveries between Miranda and Jake is 936 -/
theorem newspaper_delivery_difference :
  monthly_delivery_difference 234 2 4 = 936 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_delivery_difference_l3627_362751


namespace NUMINAMATH_CALUDE_jellybean_count_l3627_362728

/-- The number of jellybeans in a dozen -/
def dozen : ℕ := 12

/-- The number of jellybeans Caleb has -/
def caleb_jellybeans : ℕ := 3 * dozen

/-- The number of jellybeans Sophie has -/
def sophie_jellybeans : ℕ := caleb_jellybeans / 2

/-- The number of jellybeans Max has -/
def max_jellybeans : ℕ := sophie_jellybeans + 2 * dozen

/-- The total number of jellybeans -/
def total_jellybeans : ℕ := caleb_jellybeans + sophie_jellybeans + max_jellybeans

theorem jellybean_count : total_jellybeans = 96 := by sorry

end NUMINAMATH_CALUDE_jellybean_count_l3627_362728


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3627_362798

-- Define an isosceles triangle with side lengths 6 and 14
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = 14 ∧ b = 14 ∧ c = 6) ∨ (a = 6 ∧ b = 6 ∧ c = 14)

-- Define the perimeter of a triangle
def Perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, IsoscelesTriangle a b c → Perimeter a b c = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3627_362798


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l3627_362758

/-- Given three points A, B, and C in 2D space, 
    returns true if they are collinear -/
def collinear (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1)

/-- The main theorem: if A(-1,-2), B(4,8), and C(5,x) are collinear, 
    then x = 10 -/
theorem collinear_points_x_value :
  collinear (-1, -2) (4, 8) (5, x) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_x_value_l3627_362758


namespace NUMINAMATH_CALUDE_remainder_a_sixth_mod_n_l3627_362731

theorem remainder_a_sixth_mod_n (n : ℕ+) (a : ℤ) (h : a^3 ≡ 1 [ZMOD n]) :
  a^6 ≡ 1 [ZMOD n] := by sorry

end NUMINAMATH_CALUDE_remainder_a_sixth_mod_n_l3627_362731


namespace NUMINAMATH_CALUDE_multiple_of_seven_l3627_362721

theorem multiple_of_seven : (2222^5555 + 5555^2222) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_seven_l3627_362721


namespace NUMINAMATH_CALUDE_percentage_problem_l3627_362720

/-- Given a number N and a percentage P, this theorem proves that P is 20%
    when N is 580 and P% of N equals 30% of 120 plus 80. -/
theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 580 → 
  (P / 100) * N = (30 / 100) * 120 + 80 → 
  P = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3627_362720


namespace NUMINAMATH_CALUDE_correct_chest_contents_l3627_362732

-- Define the types of coins
inductive CoinType
| Gold
| Silver
| Copper

-- Define the chests
structure Chest where
  label : CoinType
  content : CoinType

-- Define the problem setup
def setup : List Chest := [
  { label := CoinType.Gold, content := CoinType.Silver },
  { label := CoinType.Silver, content := CoinType.Gold },
  { label := CoinType.Gold, content := CoinType.Copper }
]

-- Theorem statement
theorem correct_chest_contents :
  ∀ (chests : List Chest),
  (chests.length = 3) →
  (∃! c, c ∈ chests ∧ c.content = CoinType.Gold) →
  (∃! c, c ∈ chests ∧ c.content = CoinType.Silver) →
  (∃! c, c ∈ chests ∧ c.content = CoinType.Copper) →
  (∀ c ∈ chests, c.label ≠ c.content) →
  (chests = setup) :=
by sorry

end NUMINAMATH_CALUDE_correct_chest_contents_l3627_362732


namespace NUMINAMATH_CALUDE_peaches_per_basket_proof_l3627_362747

/-- The number of peaches in each basket originally -/
def peaches_per_basket : ℕ := 25

/-- The number of baskets -/
def num_baskets : ℕ := 5

/-- The number of peaches eaten by farmers -/
def eaten_peaches : ℕ := 5

/-- The number of peaches in each small box after packing -/
def peaches_per_box : ℕ := 15

/-- The number of small boxes after packing -/
def num_boxes : ℕ := 8

theorem peaches_per_basket_proof :
  peaches_per_basket * num_baskets = 
    num_boxes * peaches_per_box + eaten_peaches :=
by sorry

end NUMINAMATH_CALUDE_peaches_per_basket_proof_l3627_362747


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l3627_362761

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d
  h3 : a 1 = 1
  h4 : (a 1) * (a 5) = (a 2) ^ 2

/-- The nth term of the arithmetic sequence is 2n - 1 -/
theorem arithmetic_sequence_nth_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l3627_362761


namespace NUMINAMATH_CALUDE_triangle_radii_product_l3627_362714

theorem triangle_radii_product (a b c : ℝ) (ha : a = 26) (hb : b = 28) (hc : c = 30) :
  let p := (a + b + c) / 2
  let s := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let r := s / p
  let R := (a * b * c) / (4 * s)
  R * r = 130 := by sorry

end NUMINAMATH_CALUDE_triangle_radii_product_l3627_362714


namespace NUMINAMATH_CALUDE_B_power_101_l3627_362757

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_101 : B^101 = B^2 := by sorry

end NUMINAMATH_CALUDE_B_power_101_l3627_362757


namespace NUMINAMATH_CALUDE_binary_110101_to_base7_l3627_362788

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem binary_110101_to_base7 :
  decimal_to_base7 (binary_to_decimal [true, false, true, false, true, true]) = [1, 0, 4] :=
sorry

end NUMINAMATH_CALUDE_binary_110101_to_base7_l3627_362788


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3627_362754

theorem imaginary_part_of_complex_fraction : Complex.im ((2 + Complex.I) / (1 - 2 * Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3627_362754


namespace NUMINAMATH_CALUDE_fraction_inequality_l3627_362709

theorem fraction_inequality (x : ℝ) (h : x ≠ 2) :
  (x + 1) / (x - 2) ≥ 0 ↔ x ≤ -1 ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3627_362709


namespace NUMINAMATH_CALUDE_tromino_bounds_l3627_362763

/-- A tromino is a 1 x 3 rectangle that covers exactly three squares on a board. -/
structure Tromino

/-- The board is an n x n grid where trominoes can be placed. -/
structure Board (n : ℕ) where
  size : n > 0

/-- f(n) is the smallest number of trominoes required to stop any more being placed on an n x n board. -/
noncomputable def f (n : ℕ) : ℕ :=
  sorry

/-- For all positive n, there exist real numbers h and k such that
    (n^2 / 7) + hn ≤ f(n) ≤ (n^2 / 5) + kn -/
theorem tromino_bounds (n : ℕ) (b : Board n) :
  ∃ (h k : ℝ), (n^2 / 7 : ℝ) + h * n ≤ f n ∧ (f n : ℝ) ≤ n^2 / 5 + k * n :=
sorry

end NUMINAMATH_CALUDE_tromino_bounds_l3627_362763


namespace NUMINAMATH_CALUDE_determinant_equation_solution_l3627_362748

-- Define the determinant operation
def determinant (a b c d : ℝ) : ℝ := a * d - b * c

-- State the theorem
theorem determinant_equation_solution :
  ∃ (x : ℝ), determinant (x + 1) x (2*x - 6) (2*(x - 1)) = 10 ∧ x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_determinant_equation_solution_l3627_362748


namespace NUMINAMATH_CALUDE_sum_equals_negative_two_thirds_l3627_362767

theorem sum_equals_negative_two_thirds 
  (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 4) : 
  a + b + c + d = -2/3 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_negative_two_thirds_l3627_362767


namespace NUMINAMATH_CALUDE_thousand_ring_date_l3627_362742

/-- Represents a time with hour and minute components -/
structure Time where
  hour : Nat
  minute : Nat

/-- Represents a date with year, month, and day components -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

/-- Counts the number of bell rings from a given start time and date until the nth ring -/
def countBellRings (startTime : Time) (startDate : Date) (n : Nat) : Date :=
  sorry

/-- The bell ringing pattern: once at 45 minutes past each hour and according to the hour every hour -/
axiom bell_pattern : ∀ (t : Time), (t.minute = 45 ∧ t.hour ≠ 0) ∨ (t.minute = 0 ∧ t.hour ≠ 0)

/-- The starting time is 10:30 AM on January 1, 2021 -/
def startTime : Time := { hour := 10, minute := 30 }
def startDate : Date := { year := 2021, month := 1, day := 1 }

/-- The theorem to prove -/
theorem thousand_ring_date : 
  countBellRings startTime startDate 1000 = { year := 2021, month := 1, day := 11 } :=
sorry

end NUMINAMATH_CALUDE_thousand_ring_date_l3627_362742


namespace NUMINAMATH_CALUDE_stating_prize_distribution_orders_l3627_362755

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 6

/-- Represents the number of games played in the tournament -/
def num_games : ℕ := num_players - 1

/-- 
Theorem stating that the number of possible prize distribution orders
in a tournament with 6 players following the described elimination format is 32
-/
theorem prize_distribution_orders :
  (2 : ℕ) ^ num_games = 32 := by
  sorry

end NUMINAMATH_CALUDE_stating_prize_distribution_orders_l3627_362755


namespace NUMINAMATH_CALUDE_dot_product_bounds_l3627_362718

-- Define the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 4) + (P.2^2 / 3) = 1

-- Define the circle
def is_on_circle (Q : ℝ × ℝ) : Prop :=
  (Q.1 + 1)^2 + Q.2^2 = 1

-- Define a tangent line from a point to the circle
def is_tangent (P A : ℝ × ℝ) : Prop :=
  is_on_circle A ∧ ((P.1 - A.1) * (A.1 + 1) + (P.2 - A.2) * A.2 = 0)

-- Define the dot product of two vectors
def dot_product (P A B : ℝ × ℝ) : ℝ :=
  (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2)

-- The main theorem
theorem dot_product_bounds (P A B : ℝ × ℝ) :
  is_on_ellipse P → is_tangent P A → is_tangent P B →
  2 * Real.sqrt 2 - 3 ≤ dot_product P A B ∧ dot_product P A B ≤ 56 / 9 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_bounds_l3627_362718


namespace NUMINAMATH_CALUDE_fraction_equality_l3627_362717

theorem fraction_equality (a b c : ℝ) 
  (h1 : a / b = 20) 
  (h2 : b / c = 10) : 
  (a + b) / (b + c) = 210 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3627_362717


namespace NUMINAMATH_CALUDE_arc_angle_proof_l3627_362745

/-- Given a circle with radius 3 cm and an arc length of π/2 cm, 
    prove that the corresponding central angle is 30°. -/
theorem arc_angle_proof (r : ℝ) (l : ℝ) (θ : ℝ) : 
  r = 3 → l = π / 2 → θ = (l * 180) / (π * r) → θ = 30 := by
  sorry

end NUMINAMATH_CALUDE_arc_angle_proof_l3627_362745


namespace NUMINAMATH_CALUDE_system_solution_and_M_minimum_l3627_362729

-- Define the system of equations
def system (x y t : ℝ) : Prop :=
  x - 3*y = 4 - t ∧ x + y = 3*t

-- Define the range of t
def t_range (t : ℝ) : Prop :=
  -3 ≤ t ∧ t ≤ 1

-- Define M
def M (x y t : ℝ) : ℝ :=
  2*x - y - t

theorem system_solution_and_M_minimum :
  (∃ t, t_range t ∧ system 1 (-1) t) ∧
  (∀ x y t, t_range t → system x y t → M x y t ≥ -3) ∧
  (∃ x y t, t_range t ∧ system x y t ∧ M x y t = -3) :=
sorry

end NUMINAMATH_CALUDE_system_solution_and_M_minimum_l3627_362729


namespace NUMINAMATH_CALUDE_equality_of_fractions_l3627_362749

theorem equality_of_fractions (x y z k : ℝ) :
  (9 / (x + y) = k / (x + z)) ∧ (k / (x + z) = 15 / (z - y)) → k = 24 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_fractions_l3627_362749


namespace NUMINAMATH_CALUDE_white_balls_count_l3627_362792

theorem white_balls_count (total : ℕ) (yellow_probability : ℚ) : 
  total = 20 → yellow_probability = 3/5 → total - (total * yellow_probability).num = 8 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l3627_362792


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l3627_362723

/-- Given a hyperbola and a parabola with specific properties, 
    prove that the focal length of the hyperbola is 2√5 -/
theorem hyperbola_focal_length 
  (a b p : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hp : p > 0) 
  (h_distance : p/2 + a = 4) 
  (h_intersection : -1 = -2*b/a ∧ -2 = -p/2) : 
  2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l3627_362723


namespace NUMINAMATH_CALUDE_find_a_value_l3627_362770

noncomputable section

open Set Real

def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

theorem find_a_value : ∃ (a : ℝ), 
  (∀ x ∈ (Ioo 0 2), StrictAntiOn (f a) (Ioo 0 2)) ∧
  (∀ x ∈ (Ioi 2), StrictMonoOn (f a) (Ioi 2)) ∧
  a = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_find_a_value_l3627_362770


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l3627_362784

theorem geometric_sequence_ratio_sum (m : ℝ) (a₂ a₃ b₂ b₃ : ℝ) :
  m ≠ 0 →
  (∃ x : ℝ, x ≠ 1 ∧ a₂ = m * x ∧ a₃ = m * x^2) →
  (∃ y : ℝ, y ≠ 1 ∧ b₂ = m * y ∧ b₃ = m * y^2) →
  (∀ x y : ℝ, a₂ = m * x ∧ a₃ = m * x^2 ∧ b₂ = m * y ∧ b₃ = m * y^2 → x ≠ y) →
  a₃ - b₃ = 3 * (a₂ - b₂) →
  ∃ x y : ℝ, (a₂ = m * x ∧ a₃ = m * x^2 ∧ b₂ = m * y ∧ b₃ = m * y^2) ∧ x + y = 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l3627_362784


namespace NUMINAMATH_CALUDE_sum_of_degrees_l3627_362782

/-- Represents the degrees of four people in a specific ratio -/
structure DegreeRatio :=
  (a b c d : ℕ)
  (ratio : a = 5 ∧ b = 4 ∧ c = 6 ∧ d = 3)

/-- The theorem stating the sum of degrees given the ratio and highest degree -/
theorem sum_of_degrees (r : DegreeRatio) (highest_degree : ℕ) 
  (h : highest_degree = 150) : 
  (r.a + r.b + r.c + r.d) * (highest_degree / r.c) = 450 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_degrees_l3627_362782


namespace NUMINAMATH_CALUDE_ellipse_properties_l3627_362774

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The equation of the ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The sum of distances from a point to the foci of the ellipse -/
def Ellipse.foci_distance_sum (e : Ellipse) (x y : ℝ) : ℝ :=
  2 * e.a

theorem ellipse_properties (e : Ellipse) 
    (h_point : e.equation 0 (Real.sqrt 3))
    (h_sum : e.foci_distance_sum 0 (Real.sqrt 3) = 4) :
  e.a = 2 ∧ e.b = Real.sqrt 3 ∧ 
  (∀ x y, e.equation x y ↔ x^2/4 + y^2/3 = 1) ∧
  e.b * 2 = 2 * Real.sqrt 3 ∧
  2 * Real.sqrt (e.a^2 - e.b^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3627_362774


namespace NUMINAMATH_CALUDE_contrapositive_inequality_l3627_362739

theorem contrapositive_inequality (a b c : ℝ) :
  (¬(a < b) → ¬(a + c < b + c)) ↔ (a + c ≥ b + c → a ≥ b) := by sorry

end NUMINAMATH_CALUDE_contrapositive_inequality_l3627_362739


namespace NUMINAMATH_CALUDE_impossibleConfiguration_l3627_362765

/-- Represents a configuration of points on a circle -/
structure CircleConfiguration where
  numPoints : ℕ
  circumference : ℕ

/-- Checks if a configuration satisfies the arc length condition -/
def satisfiesArcLengthCondition (config : CircleConfiguration) : Prop :=
  ∃ (points : Fin config.numPoints → ℝ),
    (∀ i, 0 ≤ points i ∧ points i < config.circumference) ∧
    (∀ l : ℕ, 1 ≤ l ∧ l < config.circumference →
      ∃ i j, (points j - points i + config.circumference) % config.circumference = l)

/-- The main theorem stating the impossibility of the configuration -/
theorem impossibleConfiguration :
  ¬ satisfiesArcLengthCondition ⟨10, 90⟩ := by
  sorry


end NUMINAMATH_CALUDE_impossibleConfiguration_l3627_362765


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3627_362759

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → x = -4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3627_362759


namespace NUMINAMATH_CALUDE_worker_count_l3627_362719

theorem worker_count : ∃ (W : ℕ), 
  (W > 0) ∧ 
  (∃ (C : ℚ), C > 0 ∧ W * C = 300000) ∧
  (W * (C + 50) = 325000) ∧
  W = 500 := by
  sorry

end NUMINAMATH_CALUDE_worker_count_l3627_362719


namespace NUMINAMATH_CALUDE_mean_calculation_l3627_362785

def set1 : List ℝ := [28, 42, 78, 104]
def set2 : List ℝ := [128, 255, 511, 1023]

theorem mean_calculation (x : ℝ) :
  (List.sum set1 + x) / 5 = 90 →
  (List.sum set2 + x) / 5 = 423 := by
  sorry

end NUMINAMATH_CALUDE_mean_calculation_l3627_362785


namespace NUMINAMATH_CALUDE_circles_are_externally_tangent_l3627_362772

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii. -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

/-- The first circle: x^2 + y^2 = 4 -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The second circle: x^2 + y^2 - 10x + 16 = 0 -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x + 16 = 0

theorem circles_are_externally_tangent :
  externally_tangent (0, 0) (5, 0) 2 3 :=
sorry

end NUMINAMATH_CALUDE_circles_are_externally_tangent_l3627_362772


namespace NUMINAMATH_CALUDE_triangle_angle_identity_l3627_362701

theorem triangle_angle_identity (α β γ : Real) (h : α + β + γ = π) :
  2 * Real.sin α * Real.sin β * Real.cos γ = Real.sin α ^ 2 + Real.sin β ^ 2 - Real.sin γ ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_identity_l3627_362701


namespace NUMINAMATH_CALUDE_dachshund_starting_weight_l3627_362795

theorem dachshund_starting_weight :
  ∀ (labrador_start dachshund_start : ℝ),
    labrador_start = 40 →
    (labrador_start * 1.25 - dachshund_start * 1.25 = 35) →
    dachshund_start = 12 := by
  sorry

end NUMINAMATH_CALUDE_dachshund_starting_weight_l3627_362795


namespace NUMINAMATH_CALUDE_maria_test_scores_l3627_362704

def test_scores : List ℤ := [94, 92, 91, 75, 68]

theorem maria_test_scores :
  let scores := test_scores
  (scores.length = 5) ∧
  (scores.take 3 = [91, 75, 68]) ∧
  (scores.sum / scores.length = 84) ∧
  (∀ s ∈ scores, s < 95) ∧
  (∀ s ∈ scores, s ≥ 65) ∧
  scores.Nodup ∧
  scores.Sorted (· ≥ ·) :=
by sorry

end NUMINAMATH_CALUDE_maria_test_scores_l3627_362704


namespace NUMINAMATH_CALUDE_pencils_bought_l3627_362797

/-- 
Given:
- Amy initially had 3 pencils
- Amy now has a total of 10 pencils
Prove that Amy bought 7 pencils at the school store
-/
theorem pencils_bought (initial_pencils : ℕ) (total_pencils : ℕ) (bought_pencils : ℕ) : 
  initial_pencils = 3 → 
  total_pencils = 10 → 
  bought_pencils = total_pencils - initial_pencils → 
  bought_pencils = 7 := by
  sorry

end NUMINAMATH_CALUDE_pencils_bought_l3627_362797


namespace NUMINAMATH_CALUDE_student_signup_combinations_l3627_362706

theorem student_signup_combinations :
  let num_students : ℕ := 3
  let num_groups : ℕ := 4
  num_groups ^ num_students = 64 :=
by sorry

end NUMINAMATH_CALUDE_student_signup_combinations_l3627_362706


namespace NUMINAMATH_CALUDE_zoo_trip_students_l3627_362787

theorem zoo_trip_students (buses : Nat) (students_per_bus : Nat) (car_students : Nat) :
  buses = 7 →
  students_per_bus = 53 →
  car_students = 4 →
  buses * students_per_bus + car_students = 375 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_students_l3627_362787


namespace NUMINAMATH_CALUDE_problem_statement_l3627_362790

def a₁ (n : ℕ+) : ℤ := n.val^2 - 10*n.val + 23
def a₂ (n : ℕ+) : ℤ := n.val^2 - 9*n.val + 31
def a₃ (n : ℕ+) : ℤ := n.val^2 - 12*n.val + 46

theorem problem_statement :
  (∀ n : ℕ+, Even (a₁ n + a₂ n + a₃ n)) ∧
  (∀ n : ℕ+, (Prime (a₁ n) ∧ Prime (a₂ n) ∧ Prime (a₃ n)) ↔ n = 7) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3627_362790


namespace NUMINAMATH_CALUDE_lucas_units_digit_l3627_362753

-- Define Lucas numbers
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem lucas_units_digit :
  unitsDigit (lucas (lucas 15)) = 7 := by sorry

end NUMINAMATH_CALUDE_lucas_units_digit_l3627_362753


namespace NUMINAMATH_CALUDE_farm_entrance_fee_for_students_l3627_362702

theorem farm_entrance_fee_for_students :
  let num_students : ℕ := 35
  let num_adults : ℕ := 4
  let adult_fee : ℚ := 6
  let total_cost : ℚ := 199
  let student_fee : ℚ := (total_cost - num_adults * adult_fee) / num_students
  student_fee = 5 := by sorry

end NUMINAMATH_CALUDE_farm_entrance_fee_for_students_l3627_362702


namespace NUMINAMATH_CALUDE_remaining_eggs_eggs_after_three_days_l3627_362793

/-- Calculates the remaining eggs after consumption --/
theorem remaining_eggs (initial : ℕ) (consumed : ℕ) (h : initial ≥ consumed) : 
  initial - consumed = 75 - 49 → initial - consumed = 26 := by
  sorry

/-- Proves that 26 eggs remain after 3 days --/
theorem eggs_after_three_days : 
  ∃ (initial consumed : ℕ), initial = 75 ∧ consumed = 49 ∧ initial - consumed = 26 := by
  sorry

end NUMINAMATH_CALUDE_remaining_eggs_eggs_after_three_days_l3627_362793


namespace NUMINAMATH_CALUDE_josh_marbles_after_gift_l3627_362703

/-- The number of marbles Josh has after receiving marbles from Jack -/
theorem josh_marbles_after_gift (original : ℝ) (gift : ℝ) (total : ℝ)
  (h1 : original = 22.5)
  (h2 : gift = 20.75)
  (h3 : total = original + gift) :
  total = 43.25 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_after_gift_l3627_362703


namespace NUMINAMATH_CALUDE_pepper_remaining_l3627_362730

theorem pepper_remaining (initial : Real) (used : Real) (remaining : Real) : 
  initial = 0.25 → used = 0.16 → remaining = initial - used → remaining = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_pepper_remaining_l3627_362730


namespace NUMINAMATH_CALUDE_hamilton_marching_band_max_members_l3627_362716

theorem hamilton_marching_band_max_members :
  ∀ n : ℕ,
  (∃ k : ℕ, 30 * n = 34 * k + 2) →
  30 * n < 1500 →
  (∀ m : ℕ, (∃ j : ℕ, 30 * m = 34 * j + 2) → 30 * m < 1500 → 30 * m ≤ 30 * n) →
  30 * n = 1260 :=
by sorry

end NUMINAMATH_CALUDE_hamilton_marching_band_max_members_l3627_362716


namespace NUMINAMATH_CALUDE_car_speed_problem_l3627_362769

theorem car_speed_problem (D : ℝ) (D_pos : D > 0) : 
  let total_time := D / 40
  let first_part_time := (0.75 * D) / 60
  let second_part_time := total_time - first_part_time
  let s := (0.25 * D) / second_part_time
  s = 20 := by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3627_362769


namespace NUMINAMATH_CALUDE_group_average_before_new_member_l3627_362791

theorem group_average_before_new_member (group : Finset ℕ) (group_sum : ℕ) (new_member : ℕ) :
  Finset.card group = 7 →
  group_sum / Finset.card group = 20 →
  new_member = 56 →
  group_sum / Finset.card group = 20 := by
sorry

end NUMINAMATH_CALUDE_group_average_before_new_member_l3627_362791


namespace NUMINAMATH_CALUDE_prize_interval_is_1000_l3627_362711

/-- Represents the prize structure of an international competition --/
structure PrizeStructure where
  totalPrize : ℕ
  firstPrize : ℕ
  numPositions : ℕ
  hasPrizeInterval : Bool

/-- Calculates the interval between prizes --/
def calculatePrizeInterval (ps : PrizeStructure) : ℕ :=
  sorry

/-- Theorem stating that the prize interval is 1000 given the specific conditions --/
theorem prize_interval_is_1000 (ps : PrizeStructure) 
  (h1 : ps.totalPrize = 15000)
  (h2 : ps.firstPrize = 5000)
  (h3 : ps.numPositions = 5)
  (h4 : ps.hasPrizeInterval = true) : 
  calculatePrizeInterval ps = 1000 := by
  sorry

end NUMINAMATH_CALUDE_prize_interval_is_1000_l3627_362711


namespace NUMINAMATH_CALUDE_cubic_quartic_system_solution_l3627_362705

theorem cubic_quartic_system_solution (x y : ℝ) 
  (h1 : x^3 + y^3 = 1) 
  (h2 : x^4 + y^4 = 1) : 
  (x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_quartic_system_solution_l3627_362705


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3627_362726

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3627_362726


namespace NUMINAMATH_CALUDE_sequence_convergence_comparison_l3627_362727

/-- Given sequences (aₙ) and (bₙ) defined by the recurrence relations
    aₙ₊₁ = (aₙ + 1) / 2 and bₙ₊₁ = bₙᵏ, where 0 < k < 1/2 and
    a₀, b₀ ∈ (0, 1), there exists an N such that for all n ≥ N, aₙ < bₙ. -/
theorem sequence_convergence_comparison
  (k : ℝ) (h_k_pos : 0 < k) (h_k_bound : k < 1/2)
  (a₀ b₀ : ℝ) (h_a₀ : 0 < a₀ ∧ a₀ < 1) (h_b₀ : 0 < b₀ ∧ b₀ < 1)
  (a : ℕ → ℝ) (h_a : ∀ n, a (n + 1) = (a n + 1) / 2)
  (b : ℕ → ℝ) (h_b : ∀ n, b (n + 1) = (b n) ^ k)
  (h_a_init : a 0 = a₀) (h_b_init : b 0 = b₀) :
  ∃ N, ∀ n ≥ N, a n < b n :=
sorry

end NUMINAMATH_CALUDE_sequence_convergence_comparison_l3627_362727


namespace NUMINAMATH_CALUDE_painter_rooms_problem_l3627_362737

theorem painter_rooms_problem (time_per_room : ℕ) (rooms_painted : ℕ) (time_remaining : ℕ) :
  time_per_room = 8 →
  rooms_painted = 8 →
  time_remaining = 16 →
  rooms_painted + (time_remaining / time_per_room) = 10 :=
by sorry

end NUMINAMATH_CALUDE_painter_rooms_problem_l3627_362737


namespace NUMINAMATH_CALUDE_medal_winners_combinations_l3627_362771

theorem medal_winners_combinations (semifinalists : ℕ) (eliminated : ℕ) (medals : ℕ) : 
  semifinalists = 8 →
  eliminated = 2 →
  medals = 3 →
  Nat.choose (semifinalists - eliminated) medals = 20 :=
by sorry

end NUMINAMATH_CALUDE_medal_winners_combinations_l3627_362771


namespace NUMINAMATH_CALUDE_student_meeting_distance_l3627_362780

theorem student_meeting_distance (initial_distance : ℝ) (time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  initial_distance = 350 →
  time = 100 →
  speed1 = 1.6 →
  speed2 = 1.9 →
  speed2 * time = 190 :=
by sorry

end NUMINAMATH_CALUDE_student_meeting_distance_l3627_362780


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_l3627_362744

/-- Represents a systematic sample from a population --/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  start : ℕ

/-- Checks if a number is in the systematic sample --/
def inSample (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.start + k * s.interval ∧ n ≤ s.population_size

/-- The theorem to be proved --/
theorem systematic_sample_fourth_element :
  ∀ s : SystematicSample,
    s.population_size = 48 →
    s.sample_size = 4 →
    inSample s 5 →
    inSample s 29 →
    inSample s 41 →
    inSample s 17 :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_element_l3627_362744


namespace NUMINAMATH_CALUDE_quadratic_trinomial_not_factor_l3627_362712

theorem quadratic_trinomial_not_factor (r : ℕ) (p : Polynomial ℤ) :
  (∀ i, |p.coeff i| < r) →
  p ≠ 0 →
  ¬ (X^2 - r • X - 1 : Polynomial ℤ) ∣ p :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_not_factor_l3627_362712


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3627_362786

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3627_362786


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l3627_362752

-- Define an arithmetic sequence of integers
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define an increasing sequence
def is_increasing_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

-- Theorem statement
theorem arithmetic_sequence_product (a : ℕ → ℤ) :
  is_arithmetic_sequence a →
  is_increasing_sequence a →
  a 4 * a 5 = 13 →
  a 3 * a 6 = -275 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l3627_362752


namespace NUMINAMATH_CALUDE_vector_equality_l3627_362776

/-- Given vectors a, b, and c in ℝ², prove that c = 1/2 * a - 3/2 * b -/
theorem vector_equality (a b c : Fin 2 → ℝ) 
  (ha : a = ![1, 1])
  (hb : b = ![1, -1])
  (hc : c = ![-1, 2]) :
  c = 1/2 • a - 3/2 • b := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_l3627_362776


namespace NUMINAMATH_CALUDE_delta_phi_equals_negative_one_l3627_362734

def δ (x : ℚ) : ℚ := 5 * x + 6

def φ (x : ℚ) : ℚ := 6 * x + 5

theorem delta_phi_equals_negative_one (x : ℚ) : 
  δ (φ x) = -1 ↔ x = -16/15 := by sorry

end NUMINAMATH_CALUDE_delta_phi_equals_negative_one_l3627_362734


namespace NUMINAMATH_CALUDE_closest_multiple_of_18_to_2021_l3627_362707

def closest_multiple (n m : ℕ) : ℕ :=
  let q := n / m
  let r := n % m
  if r ≤ m / 2 then q * m else (q + 1) * m

theorem closest_multiple_of_18_to_2021 :
  closest_multiple 2021 18 = 2016 :=
sorry

end NUMINAMATH_CALUDE_closest_multiple_of_18_to_2021_l3627_362707


namespace NUMINAMATH_CALUDE_oranges_used_proof_l3627_362768

/-- Calculates the total number of oranges used to make juice -/
def total_oranges (oranges_per_glass : ℕ) (glasses : ℕ) : ℕ :=
  oranges_per_glass * glasses

/-- Proves that the total number of oranges used is 12 -/
theorem oranges_used_proof (oranges_per_glass : ℕ) (glasses : ℕ)
  (h1 : oranges_per_glass = 2)
  (h2 : glasses = 6) :
  total_oranges oranges_per_glass glasses = 12 := by
  sorry

end NUMINAMATH_CALUDE_oranges_used_proof_l3627_362768


namespace NUMINAMATH_CALUDE_translation_theorem_l3627_362708

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally and vertically -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

theorem translation_theorem :
  let A : Point := { x := -2, y := 3 }
  let A' : Point := translate (translate A 0 (-3)) 4 0
  A'.x = 2 ∧ A'.y = 0 := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l3627_362708


namespace NUMINAMATH_CALUDE_lg_calculation_l3627_362736

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_calculation : lg 25 - 2 * lg (1/2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_lg_calculation_l3627_362736


namespace NUMINAMATH_CALUDE_total_flowers_and_stems_l3627_362738

def roses : ℕ := 12
def carnations : ℕ := 15
def lilies : ℕ := 10
def tulips : ℕ := 8
def daisies : ℕ := 5
def orchids : ℕ := 3
def babys_breath : ℕ := 10

theorem total_flowers_and_stems :
  roses + carnations + lilies + tulips + daisies + orchids + babys_breath = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_and_stems_l3627_362738


namespace NUMINAMATH_CALUDE_rectangle_fourth_vertex_l3627_362740

-- Define a structure for a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a structure for a rectangle
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the theorem
theorem rectangle_fourth_vertex 
  (ABCD : Rectangle)
  (h1 : ABCD.A = ⟨0, 1⟩)
  (h2 : ABCD.B = ⟨1, 0⟩)
  (h3 : ABCD.C = ⟨3, 2⟩)
  : ABCD.D = ⟨2, 3⟩ := by
  sorry

end NUMINAMATH_CALUDE_rectangle_fourth_vertex_l3627_362740


namespace NUMINAMATH_CALUDE_reach_all_integers_l3627_362783

/-- Represents the allowed operations on positive integers -/
inductive Operation
  | append4 : Operation
  | append0 : Operation
  | divideBy2 : Operation

/-- Applies an operation to a positive integer -/
def applyOperation (n : ℕ+) (op : Operation) : ℕ+ :=
  match op with
  | Operation.append4 => ⟨10 * n.val + 4, by sorry⟩
  | Operation.append0 => ⟨10 * n.val, by sorry⟩
  | Operation.divideBy2 => if n.val % 2 = 0 then ⟨n.val / 2, by sorry⟩ else n

/-- Applies a sequence of operations to a positive integer -/
def applyOperations (n : ℕ+) (ops : List Operation) : ℕ+ :=
  ops.foldl applyOperation n

/-- Theorem stating that any positive integer can be reached from 4 using the allowed operations -/
theorem reach_all_integers (n : ℕ+) : 
  ∃ (ops : List Operation), applyOperations ⟨4, by norm_num⟩ ops = n := by
  sorry

end NUMINAMATH_CALUDE_reach_all_integers_l3627_362783


namespace NUMINAMATH_CALUDE_wilmas_garden_red_flowers_l3627_362794

/-- Wilma's Garden Flower Count Theorem -/
theorem wilmas_garden_red_flowers :
  let total_flowers : ℕ := 6 * 13
  let yellow_flowers : ℕ := 12
  let green_flowers : ℕ := 2 * yellow_flowers
  let red_flowers : ℕ := total_flowers - (yellow_flowers + green_flowers)
  red_flowers = 42 := by sorry

end NUMINAMATH_CALUDE_wilmas_garden_red_flowers_l3627_362794


namespace NUMINAMATH_CALUDE_mark_marbles_count_l3627_362764

def connie_marbles : ℕ := 323
def juan_marbles : ℕ := connie_marbles + 175
def mark_marbles : ℕ := juan_marbles * 3

theorem mark_marbles_count : mark_marbles = 1494 := by
  sorry

end NUMINAMATH_CALUDE_mark_marbles_count_l3627_362764


namespace NUMINAMATH_CALUDE_sequence_properties_l3627_362778

def sequence_a (n : ℕ) : ℝ := 2^n

def sum_S (n : ℕ) : ℝ := 2 * sequence_a n - 2

def sequence_b (n : ℕ) : ℝ := (n + 1 : ℝ) * sequence_a n

def sum_T (n : ℕ) : ℝ := n * 2^(n + 1)

theorem sequence_properties (n : ℕ) :
  (∀ k, sum_S k = 2 * sequence_a k - 2) →
  (sequence_a n = 2^n ∧
   sum_T n = n * 2^(n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l3627_362778


namespace NUMINAMATH_CALUDE_equal_temperament_sequence_l3627_362789

theorem equal_temperament_sequence (a : ℕ → ℝ) :
  (∀ n, 1 ≤ n → n ≤ 13 → a n > 0) →
  (∀ n, 1 ≤ n → n < 13 → a (n + 1) / a n = a 2 / a 1) →
  a 1 = 1 →
  a 13 = 2 →
  a 3 = 2^(1/6) :=
by sorry

end NUMINAMATH_CALUDE_equal_temperament_sequence_l3627_362789


namespace NUMINAMATH_CALUDE_max_rectangles_in_square_l3627_362713

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square grid -/
structure Grid where
  size : ℕ

/-- Defines a 4×1 rectangle -/
def fourByOne : Rectangle := { width := 4, height := 1 }

/-- Defines a 6×6 grid -/
def sixBySix : Grid := { size := 6 }

/-- 
  Theorem: The maximum number of 4×1 rectangles that can be placed 
  in a 6×6 square without crossing cell boundaries is 8.
-/
theorem max_rectangles_in_square : 
  ∃ (n : ℕ), n = 8 ∧ 
  (∀ (m : ℕ), m > n → 
    ¬ (∃ (arrangement : List (ℕ × ℕ)), 
      arrangement.length = m ∧
      (∀ (pos : ℕ × ℕ), pos ∈ arrangement → 
        pos.1 + fourByOne.width ≤ sixBySix.size ∧ 
        pos.2 + fourByOne.height ≤ sixBySix.size) ∧
      (∀ (pos1 pos2 : ℕ × ℕ), pos1 ∈ arrangement → pos2 ∈ arrangement → pos1 ≠ pos2 → 
        ¬ (pos1.1 < pos2.1 + fourByOne.width ∧ 
           pos2.1 < pos1.1 + fourByOne.width ∧ 
           pos1.2 < pos2.2 + fourByOne.height ∧ 
           pos2.2 < pos1.2 + fourByOne.height)))) :=
by
  sorry

end NUMINAMATH_CALUDE_max_rectangles_in_square_l3627_362713


namespace NUMINAMATH_CALUDE_mean_of_playground_counts_l3627_362775

def playground_counts : List ℕ := [6, 12, 1, 12, 7, 3, 8]

theorem mean_of_playground_counts :
  (playground_counts.sum : ℚ) / playground_counts.length = 7 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_playground_counts_l3627_362775


namespace NUMINAMATH_CALUDE_semiperimeter_equals_diagonal_l3627_362781

/-- A rectangle inscribed in a square --/
structure InscribedRectangle where
  /-- Side length of the square --/
  a : ℝ
  /-- Width of the rectangle --/
  b : ℝ
  /-- Height of the rectangle --/
  c : ℝ
  /-- The rectangle is not a square --/
  not_square : b ≠ c
  /-- The rectangle is inscribed in the square --/
  inscribed : b + c = a * Real.sqrt 2

/-- The semiperimeter of the inscribed rectangle equals the diagonal of the square --/
theorem semiperimeter_equals_diagonal (rect : InscribedRectangle) :
  (rect.b + rect.c) / 2 = rect.a * Real.sqrt 2 / 2 := by
  sorry

#check semiperimeter_equals_diagonal

end NUMINAMATH_CALUDE_semiperimeter_equals_diagonal_l3627_362781


namespace NUMINAMATH_CALUDE_line_equation_l3627_362733

/-- The circle with center (-1, 2) and radius √(5-a) --/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 2)^2 = 5 - a}

/-- The line l --/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}

/-- The midpoint of the chord AB --/
def M : ℝ × ℝ := (0, 1)

theorem line_equation (a : ℝ) (h : a < 3) :
  ∃ A B : ℝ × ℝ, A ∈ Circle a ∧ B ∈ Circle a ∧
  A ∈ Line ∧ B ∈ Line ∧
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  Line = {p : ℝ × ℝ | p.1 - p.2 + 1 = 0} := by
sorry

end NUMINAMATH_CALUDE_line_equation_l3627_362733


namespace NUMINAMATH_CALUDE_diamond_paths_count_l3627_362762

/-- Represents a position in the diamond grid -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents a move in the diamond grid -/
inductive Move
  | Right
  | Down
  | DiagonalDown

/-- The diamond-shaped grid containing the word "DIAMOND" -/
def diamond_grid : List (List Char) := sorry

/-- Check if a move is valid in the diamond grid -/
def is_valid_move (grid : List (List Char)) (pos : Position) (move : Move) : Bool := sorry

/-- Get the next position after a move -/
def next_position (pos : Position) (move : Move) : Position := sorry

/-- Check if a path spells "DIAMOND" -/
def spells_diamond (grid : List (List Char)) (path : List Move) : Bool := sorry

/-- Count the number of valid paths spelling "DIAMOND" -/
def count_diamond_paths (grid : List (List Char)) : ℕ := sorry

theorem diamond_paths_count :
  count_diamond_paths diamond_grid = 64 := by sorry

end NUMINAMATH_CALUDE_diamond_paths_count_l3627_362762


namespace NUMINAMATH_CALUDE_circle_radius_with_secant_l3627_362722

/-- Represents a circle with an external point and a secant --/
structure CircleWithSecant where
  -- Radius of the circle
  r : ℝ
  -- Distance from external point P to center
  distPC : ℝ
  -- Length of external segment PQ
  lenPQ : ℝ
  -- Length of segment QR
  lenQR : ℝ
  -- Condition: P is outside the circle
  h_outside : distPC > r
  -- Condition: PQ is external segment
  h_external : lenPQ < distPC

/-- The radius of the circle given the specified conditions --/
theorem circle_radius_with_secant (c : CircleWithSecant)
    (h_distPC : c.distPC = 17)
    (h_lenPQ : c.lenPQ = 12)
    (h_lenQR : c.lenQR = 8) :
    c.r = 7 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_with_secant_l3627_362722


namespace NUMINAMATH_CALUDE_josh_marbles_l3627_362777

/-- The number of marbles Josh has after losing some -/
def remaining_marbles (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem: If Josh had 9 marbles initially and lost 5, he now has 4 marbles -/
theorem josh_marbles : remaining_marbles 9 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_l3627_362777


namespace NUMINAMATH_CALUDE_solve_system_l3627_362715

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 7 * q = 20) 
  (eq2 : 7 * p + 5 * q = 26) : 
  q = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3627_362715


namespace NUMINAMATH_CALUDE_rectangle_width_l3627_362760

/-- 
Given a rectangle where:
  - The length is 3 cm shorter than the width
  - The perimeter is 54 cm
Prove that the width of the rectangle is 15 cm
-/
theorem rectangle_width (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = width - 3 →
  perimeter = 54 →
  perimeter = 2 * width + 2 * length →
  width = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l3627_362760


namespace NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l3627_362725

/-- Given an arithmetic sequence where the sum of the first and third terms is 10,
    prove that the second term is 5. -/
theorem arithmetic_sequence_second_term 
  (a : ℝ) -- First term of the arithmetic sequence
  (d : ℝ) -- Common difference of the arithmetic sequence
  (h : a + (a + 2*d) = 10) -- Sum of first and third terms is 10
  : a + d = 5 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l3627_362725


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3627_362773

/-- The limiting sum of the geometric series 4 - 8/3 + 16/9 - ... equals 2.4 -/
theorem geometric_series_sum : 
  let a : ℝ := 4
  let r : ℝ := -2/3
  let s : ℝ := a / (1 - r)
  s = 2.4 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3627_362773


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3627_362724

theorem tangent_line_to_circle (r : ℝ) : 
  r > 0 → 
  (∃ (x y : ℝ), x + 2*y = r ∧ x^2 + y^2 = 2*r^2) →
  (∀ (x y : ℝ), x + 2*y = r → x^2 + y^2 ≥ 2*r^2) →
  r = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3627_362724


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1800_l3627_362796

theorem largest_perfect_square_factor_of_1800 : 
  ∃ (n : ℕ), n * n = 3600 ∧ 
  3600 ∣ 1800 ∧
  ∀ (m : ℕ), m * m ∣ 1800 → m * m ≤ 3600 :=
sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1800_l3627_362796


namespace NUMINAMATH_CALUDE_polygon_count_l3627_362743

/-- The number of points marked on the circle -/
def n : ℕ := 12

/-- The number of distinct convex polygons with 3 or more sides 
    that can be drawn using some or all of n points marked on a circle as vertices -/
def num_polygons (n : ℕ) : ℕ := 2^n - (n.choose 0 + n.choose 1 + n.choose 2)

theorem polygon_count : num_polygons n = 4017 := by
  sorry

end NUMINAMATH_CALUDE_polygon_count_l3627_362743


namespace NUMINAMATH_CALUDE_red_car_speed_is_10_l3627_362756

/-- The speed of the black car in miles per hour -/
def black_car_speed : ℝ := 50

/-- The initial distance between the cars in miles -/
def initial_distance : ℝ := 20

/-- The time it takes for the black car to overtake the red car in hours -/
def overtake_time : ℝ := 0.5

/-- The speed of the red car in miles per hour -/
def red_car_speed : ℝ := 10

theorem red_car_speed_is_10 :
  red_car_speed = 10 :=
by sorry

end NUMINAMATH_CALUDE_red_car_speed_is_10_l3627_362756
