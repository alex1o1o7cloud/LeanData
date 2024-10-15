import Mathlib

namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l819_81906

theorem no_positive_integer_solution :
  ¬ ∃ (a b c d : ℕ+), a + b + c + d - 3 = a * b + c * d := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l819_81906


namespace NUMINAMATH_CALUDE_existence_of_equal_elements_l819_81930

theorem existence_of_equal_elements
  (p q n : ℕ+)
  (h_sum : p + q < n)
  (x : Fin (n + 1) → ℤ)
  (h_boundary : x 0 = 0 ∧ x n = 0)
  (h_diff : ∀ i : Fin n, x (i + 1) - x i = p ∨ x (i + 1) - x i = -q) :
  ∃ (i j : Fin (n + 1)), i ≠ j ∧ (i, j) ≠ (0, n) ∧ x i = x j :=
by sorry

end NUMINAMATH_CALUDE_existence_of_equal_elements_l819_81930


namespace NUMINAMATH_CALUDE_product_of_squares_l819_81900

theorem product_of_squares (x : ℝ) : 
  (Real.sqrt (6 + x) + Real.sqrt (21 - x) = 8) → 
  ((6 + x) * (21 - x) = 1369 / 4) := by
sorry

end NUMINAMATH_CALUDE_product_of_squares_l819_81900


namespace NUMINAMATH_CALUDE_intersection_perpendicular_points_l819_81918

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def l (x y m : ℝ) : Prop := y = 2 * x + m

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem intersection_perpendicular_points (m : ℝ) : 
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    C x₁ y₁ ∧ C x₂ y₂ ∧ 
    l x₁ y₁ m ∧ l x₂ y₂ m ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    perpendicular x₁ y₁ x₂ y₂ ↔ 
    m = 2 ∨ m = -2 := by sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_points_l819_81918


namespace NUMINAMATH_CALUDE_remainder_2673_base12_div_9_l819_81949

/-- Converts a base-12 integer to decimal --/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

/-- The base-12 representation of 2673 --/
def base12_2673 : List Nat := [2, 6, 7, 3]

theorem remainder_2673_base12_div_9 :
  (base12ToDecimal base12_2673) % 9 = 8 := by sorry

end NUMINAMATH_CALUDE_remainder_2673_base12_div_9_l819_81949


namespace NUMINAMATH_CALUDE_large_rectangle_perimeter_l819_81987

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Calculates the perimeter of a square -/
def Square.perimeter (s : Square) : ℝ := 4 * s.side

/-- Theorem stating the perimeter of the large rectangle -/
theorem large_rectangle_perimeter 
  (square : Square) 
  (small_rect : Rectangle) 
  (h1 : square.perimeter = 24) 
  (h2 : small_rect.perimeter = 16) 
  (h3 : small_rect.length = square.side) :
  let large_rect := Rectangle.mk (3 * square.side + small_rect.length) (square.side + small_rect.width)
  large_rect.perimeter = 52 := by
  sorry


end NUMINAMATH_CALUDE_large_rectangle_perimeter_l819_81987


namespace NUMINAMATH_CALUDE_cards_in_play_l819_81958

/-- The number of cards in a standard deck --/
def standard_deck : ℕ := 52

/-- The number of cards kept away --/
def cards_kept_away : ℕ := 2

/-- Theorem: The number of cards being played with is 50 --/
theorem cards_in_play (deck : ℕ) (kept_away : ℕ) 
  (h1 : deck = standard_deck) (h2 : kept_away = cards_kept_away) : 
  deck - kept_away = 50 := by
  sorry

end NUMINAMATH_CALUDE_cards_in_play_l819_81958


namespace NUMINAMATH_CALUDE_total_bags_l819_81920

theorem total_bags (points_per_bag : ℕ) (total_points : ℕ) (unrecycled_bags : ℕ) : 
  points_per_bag = 5 → total_points = 45 → unrecycled_bags = 8 →
  (total_points / points_per_bag + unrecycled_bags : ℕ) = 17 := by
sorry

end NUMINAMATH_CALUDE_total_bags_l819_81920


namespace NUMINAMATH_CALUDE_equal_pair_proof_l819_81923

theorem equal_pair_proof : 
  ((-3 : ℤ)^2 = Int.sqrt 81) ∧ 
  (|(-3 : ℤ)| ≠ -3) ∧ 
  (-|(-4 : ℤ)| ≠ (-2 : ℤ)^2) ∧ 
  (Int.sqrt ((-4 : ℤ)^2) ≠ -4) :=
by sorry

end NUMINAMATH_CALUDE_equal_pair_proof_l819_81923


namespace NUMINAMATH_CALUDE_real_part_of_z_l819_81967

/-- Given that z = (1+i)(1-2i)(i) where i is the imaginary unit, prove that the real part of z is 3 -/
theorem real_part_of_z (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (1 + i) * (1 - 2*i) * i
  (z.re : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l819_81967


namespace NUMINAMATH_CALUDE_power_function_passes_through_one_l819_81964

theorem power_function_passes_through_one (α : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x ^ α
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_power_function_passes_through_one_l819_81964


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l819_81962

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l819_81962


namespace NUMINAMATH_CALUDE_sum_min_period_length_l819_81916

def min_period_length (x : ℚ) : ℕ :=
  sorry

theorem sum_min_period_length (A B : ℚ) :
  min_period_length A = 6 →
  min_period_length B = 12 →
  min_period_length (A + B) = 12 ∨ min_period_length (A + B) = 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_min_period_length_l819_81916


namespace NUMINAMATH_CALUDE_dot_product_range_l819_81905

/-- Given points A and B in a 2D Cartesian coordinate system,
    and P on the curve y = √(1-x²), prove that the dot product
    BP · BA is bounded by 0 and 1+√2. -/
theorem dot_product_range (x y : ℝ) :
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (0, -1)
  let P : ℝ × ℝ := (x, y)
  y = Real.sqrt (1 - x^2) →
  0 ≤ ((P.1 - B.1) * (A.1 - B.1) + (P.2 - B.2) * (A.2 - B.2)) ∧
  ((P.1 - B.1) * (A.1 - B.1) + (P.2 - B.2) * (A.2 - B.2)) ≤ 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_range_l819_81905


namespace NUMINAMATH_CALUDE_parabola_kite_theorem_l819_81944

/-- Represents a parabola of the form y = ax^2 + c -/
structure Parabola where
  a : ℝ
  c : ℝ

/-- Represents a kite formed by the intersection points of two parabolas with the coordinate axes -/
structure Kite where
  p1 : Parabola
  p2 : Parabola

/-- The area of a kite -/
def kite_area (k : Kite) : ℝ := sorry

/-- The sum of the coefficients of the x^2 terms in the two parabolas forming the kite -/
def coeff_sum (k : Kite) : ℝ := k.p1.a + k.p2.a

theorem parabola_kite_theorem (k : Kite) :
  k.p1 = Parabola.mk a (-4) ∧
  k.p2 = Parabola.mk (-b) 8 ∧
  kite_area k = 24 →
  coeff_sum k = 3 := by sorry

end NUMINAMATH_CALUDE_parabola_kite_theorem_l819_81944


namespace NUMINAMATH_CALUDE_division_multiplication_problem_l819_81939

theorem division_multiplication_problem : 
  let x : ℝ := 7.5
  let y : ℝ := 6
  let z : ℝ := 12
  (x / y) * z = 15 := by sorry

end NUMINAMATH_CALUDE_division_multiplication_problem_l819_81939


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l819_81926

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1 / a + 4 / b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 1 / a₀ + 4 / b₀ = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l819_81926


namespace NUMINAMATH_CALUDE_tax_rate_above_40k_l819_81917

/-- Proves that the tax rate on income above $40,000 is 20% given the conditions --/
theorem tax_rate_above_40k (total_income : ℝ) (total_tax : ℝ) :
  total_income = 58000 →
  total_tax = 8000 →
  (∃ (rate_above_40k : ℝ),
    total_tax = 0.11 * 40000 + rate_above_40k * (total_income - 40000) ∧
    rate_above_40k = 0.20) :=
by
  sorry

end NUMINAMATH_CALUDE_tax_rate_above_40k_l819_81917


namespace NUMINAMATH_CALUDE_walkway_diameter_l819_81954

theorem walkway_diameter (water_diameter : Real) (tile_width : Real) (walkway_width : Real) :
  water_diameter = 16 →
  tile_width = 12 →
  walkway_width = 10 →
  2 * (water_diameter / 2 + tile_width + walkway_width) = 60 := by
  sorry

end NUMINAMATH_CALUDE_walkway_diameter_l819_81954


namespace NUMINAMATH_CALUDE_sequence_existence_and_extension_l819_81983

theorem sequence_existence_and_extension (m : ℕ) (h : m ≥ 2) :
  (∃ x : ℕ → ℤ, ∀ i ∈ Finset.range m, x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1) ∧
  (∀ x : ℕ → ℤ, (∀ i ∈ Finset.range m, x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1) →
    ∃ y : ℤ → ℤ, (∀ k : ℤ, y k * y (m + k) = y (k + 1) * y (m + k - 1) + 1) ∧
               (∀ i ∈ Finset.range (2 * m), y i = x i)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_existence_and_extension_l819_81983


namespace NUMINAMATH_CALUDE_lilly_and_rosy_fish_l819_81980

/-- The number of fish Lilly and Rosy have together -/
def total_fish (lilly_fish rosy_fish : ℕ) : ℕ := lilly_fish + rosy_fish

/-- Theorem: Lilly and Rosy have 22 fish in total -/
theorem lilly_and_rosy_fish : total_fish 10 12 = 22 := by
  sorry

end NUMINAMATH_CALUDE_lilly_and_rosy_fish_l819_81980


namespace NUMINAMATH_CALUDE_max_equilateral_triangles_l819_81925

-- Define the number of line segments
def num_segments : ℕ := 6

-- Define the length of each segment
def segment_length : ℝ := 2

-- Define the side length of the equilateral triangles
def triangle_side_length : ℝ := 2

-- State the theorem
theorem max_equilateral_triangles :
  ∃ (n : ℕ), n ≤ 4 ∧
  (∀ (m : ℕ), (∃ (arrangement : List (List ℕ)),
    (∀ triangle ∈ arrangement, triangle.length = 3 ∧
     (∀ side ∈ triangle, side ≤ num_segments) ∧
     arrangement.length = m) →
    m ≤ n)) ∧
  (∃ (arrangement : List (List ℕ)),
    (∀ triangle ∈ arrangement, triangle.length = 3 ∧
     (∀ side ∈ triangle, side ≤ num_segments) ∧
     arrangement.length = 4)) :=
by sorry

end NUMINAMATH_CALUDE_max_equilateral_triangles_l819_81925


namespace NUMINAMATH_CALUDE_g_one_equals_three_l819_81977

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem g_one_equals_three (f g : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_even : is_even_function g) 
  (h_eq1 : f (-1) + g 1 = 2) 
  (h_eq2 : f 1 + g (-1) = 4) : 
  g 1 = 3 := by sorry

end NUMINAMATH_CALUDE_g_one_equals_three_l819_81977


namespace NUMINAMATH_CALUDE_new_years_appetizer_l819_81929

/-- The number of bags of chips Alex bought for his New Year's Eve appetizer -/
def num_bags : ℕ := 3

/-- The cost of each bag of chips in dollars -/
def cost_per_bag : ℚ := 1

/-- The cost of creme fraiche in dollars -/
def cost_creme_fraiche : ℚ := 5

/-- The cost of caviar in dollars -/
def cost_caviar : ℚ := 73

/-- The total cost per person in dollars -/
def cost_per_person : ℚ := 27

theorem new_years_appetizer :
  (cost_per_bag * num_bags + cost_creme_fraiche + cost_caviar) / num_bags = cost_per_person :=
by
  sorry

end NUMINAMATH_CALUDE_new_years_appetizer_l819_81929


namespace NUMINAMATH_CALUDE_cos_equality_theorem_l819_81970

theorem cos_equality_theorem :
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (942 * π / 180) ∧ n = 138 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_theorem_l819_81970


namespace NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l819_81935

def U : Set Int := {x | -3 < x ∧ x < 3}
def A : Set Int := {1, 2}
def B : Set Int := {-2, -1, 2}

theorem union_of_A_and_complement_of_B :
  A ∪ (U \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l819_81935


namespace NUMINAMATH_CALUDE_factorial_ratio_evaluation_l819_81931

theorem factorial_ratio_evaluation : (Nat.factorial 10 * Nat.factorial 4 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 5 / 21 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_evaluation_l819_81931


namespace NUMINAMATH_CALUDE_rectangle_area_change_l819_81910

/-- Given a rectangle with dimensions 4 and 6, if shortening one side by 1
    results in an area of 18, then shortening the other side by 1
    results in an area of 20. -/
theorem rectangle_area_change (l w : ℝ) : 
  l = 4 ∧ w = 6 ∧ 
  ((l - 1) * w = 18 ∨ l * (w - 1) = 18) →
  (l * (w - 1) = 20 ∨ (l - 1) * w = 20) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l819_81910


namespace NUMINAMATH_CALUDE_stewart_farm_horse_food_l819_81992

/-- The Stewart farm problem -/
theorem stewart_farm_horse_food (sheep_count : ℕ) (total_horse_food : ℕ) 
  (sheep_to_horse_ratio : ℚ) : 
  sheep_count = 8 →
  total_horse_food = 12880 →
  sheep_to_horse_ratio = 1 / 7 →
  (total_horse_food : ℚ) / ((sheep_count : ℚ) / sheep_to_horse_ratio) = 230 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_horse_food_l819_81992


namespace NUMINAMATH_CALUDE_total_spent_is_correct_l819_81904

def clothes_price : ℝ := 250
def clothes_discount : ℝ := 0.15
def movie_ticket_price : ℝ := 24
def movie_tickets : ℕ := 3
def movie_discount : ℝ := 0.10
def beans_price : ℝ := 1.25
def beans_quantity : ℕ := 20
def cucumber_price : ℝ := 2.50
def cucumber_quantity : ℕ := 5
def tomato_price : ℝ := 5.00
def tomato_quantity : ℕ := 3
def pineapple_price : ℝ := 6.50
def pineapple_quantity : ℕ := 2

def total_spent : ℝ := 
  clothes_price * (1 - clothes_discount) +
  (movie_ticket_price * movie_tickets) * (1 - movie_discount) +
  (beans_price * beans_quantity) +
  (cucumber_price * cucumber_quantity) +
  (tomato_price * tomato_quantity) +
  (pineapple_price * pineapple_quantity)

theorem total_spent_is_correct : total_spent = 342.80 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_correct_l819_81904


namespace NUMINAMATH_CALUDE_inequality_proof_l819_81913

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + a + 1) * (b^2 + b + 1) * (c^2 + c + 1) / (a * b * c) ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l819_81913


namespace NUMINAMATH_CALUDE_root_sum_quotient_l819_81947

theorem root_sum_quotient (p q r s t : ℝ) (hp : p ≠ 0) 
  (h1 : p * 6^4 + q * 6^3 + r * 6^2 + s * 6 + t = 0)
  (h2 : p * (-4)^4 + q * (-4)^3 + r * (-4)^2 + s * (-4) + t = 0)
  (h3 : t = 0) :
  (q + s) / p = 48 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_quotient_l819_81947


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l819_81996

theorem average_of_a_and_b (a b c : ℝ) : 
  (a + b) / 2 = 45 ∧ (b + c) / 2 = 60 ∧ c - a = 30 → (a + b) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l819_81996


namespace NUMINAMATH_CALUDE_cat_distribution_theorem_l819_81903

/-- Represents the number of segments white cats are divided into by black cats -/
inductive X
| one
| two
| three
| four

/-- The probability distribution of X -/
def P (x : X) : ℚ :=
  match x with
  | X.one => 1 / 30
  | X.two => 9 / 30
  | X.three => 15 / 30
  | X.four => 5 / 30

theorem cat_distribution_theorem :
  (∀ x : X, 0 ≤ P x ∧ P x ≤ 1) ∧
  (P X.one + P X.two + P X.three + P X.four = 1) :=
sorry

end NUMINAMATH_CALUDE_cat_distribution_theorem_l819_81903


namespace NUMINAMATH_CALUDE_tom_payment_l819_81934

/-- The amount Tom paid to the shopkeeper -/
def total_amount (apple_quantity apple_rate mango_quantity mango_rate : ℕ) : ℕ :=
  apple_quantity * apple_rate + mango_quantity * mango_rate

/-- Proof that Tom paid 1055 to the shopkeeper -/
theorem tom_payment : total_amount 8 70 9 55 = 1055 := by
  sorry

end NUMINAMATH_CALUDE_tom_payment_l819_81934


namespace NUMINAMATH_CALUDE_square_sum_calculation_l819_81976

theorem square_sum_calculation (a b c : ℝ) 
  (h1 : a * b + b * c + c * a = 10)
  (h2 : a + b + c = 31) : 
  a^2 + b^2 + c^2 = 941 := by
sorry

end NUMINAMATH_CALUDE_square_sum_calculation_l819_81976


namespace NUMINAMATH_CALUDE_marble_problem_solution_l819_81990

/-- Represents a jar of marbles -/
structure Jar where
  blue : ℕ
  green : ℕ

/-- The problem setup -/
def marble_problem : Prop :=
  ∃ (jar1 jar2 : Jar),
    -- Both jars have the same total number of marbles
    jar1.blue + jar1.green = jar2.blue + jar2.green
    -- Ratio of blue to green in Jar 1 is 9:1
    ∧ 9 * jar1.green = jar1.blue
    -- Ratio of blue to green in Jar 2 is 7:2
    ∧ 7 * jar2.green = 2 * jar2.blue
    -- Total number of green marbles is 108
    ∧ jar1.green + jar2.green = 108
    -- The difference in blue marbles between Jar 1 and Jar 2 is 38
    ∧ jar1.blue - jar2.blue = 38

/-- The theorem to prove -/
theorem marble_problem_solution : marble_problem := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_solution_l819_81990


namespace NUMINAMATH_CALUDE_trick_decks_total_spend_l819_81951

/-- The total amount spent by Victor and his friend on trick decks -/
def totalSpent (deckCost : ℕ) (victorDecks : ℕ) (friendDecks : ℕ) : ℕ :=
  deckCost * (victorDecks + friendDecks)

/-- Theorem stating the total amount spent by Victor and his friend -/
theorem trick_decks_total_spend :
  totalSpent 8 6 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_trick_decks_total_spend_l819_81951


namespace NUMINAMATH_CALUDE_fraction_power_approximation_l819_81999

theorem fraction_power_approximation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.000000000000000001 ∧ 
  |((1 : ℝ) / 9)^2 - 0.012345679012345678| < ε :=
sorry

end NUMINAMATH_CALUDE_fraction_power_approximation_l819_81999


namespace NUMINAMATH_CALUDE_mirror_number_max_k_value_l819_81975

/-- Definition of a mirror number -/
def is_mirror_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n % 10 ≠ 0) ∧ (n / 10 % 10 ≠ 0) ∧ (n / 100 % 10 ≠ 0) ∧ (n / 1000 ≠ 0) ∧
  (n % 10 ≠ n / 10 % 10) ∧ (n % 10 ≠ n / 100 % 10) ∧ (n % 10 ≠ n / 1000) ∧
  (n / 10 % 10 ≠ n / 100 % 10) ∧ (n / 10 % 10 ≠ n / 1000) ∧ (n / 100 % 10 ≠ n / 1000) ∧
  (n % 10 + n / 1000 = n / 10 % 10 + n / 100 % 10)

/-- Definition of F(m) -/
def F (m : ℕ) : ℚ :=
  let m₁ := (m % 10) * 1000 + (m / 10 % 10) * 100 + (m / 100 % 10) * 10 + (m / 1000)
  let m₂ := (m / 1000) * 1000 + (m / 100 % 10) * 100 + (m / 10 % 10) * 10 + (m % 10)
  (m₁ + m₂ : ℚ) / 1111

/-- Main theorem -/
theorem mirror_number_max_k_value 
  (s t : ℕ) 
  (x y e f : ℕ)
  (hs : is_mirror_number s)
  (ht : is_mirror_number t)
  (hx : 1 ≤ x ∧ x ≤ 9)
  (hy : 1 ≤ y ∧ y ≤ 9)
  (he : 1 ≤ e ∧ e ≤ 9)
  (hf : 1 ≤ f ∧ f ≤ 9)
  (hs_def : s = 1000 * x + 100 * y + 32)
  (ht_def : t = 1500 + 10 * e + f)
  (h_sum : F s + F t = 19)
  : (F s / F t) ≤ 11 / 8 :=
sorry

end NUMINAMATH_CALUDE_mirror_number_max_k_value_l819_81975


namespace NUMINAMATH_CALUDE_line_through_circle_center_parallel_to_line_l819_81912

/-- The equation of a line passing through the center of a circle and parallel to another line -/
theorem line_through_circle_center_parallel_to_line :
  let circle : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 1}
  let center : ℝ × ℝ := (2, 0)
  let parallel_line : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 + 1 = 0}
  let result_line : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 - 4 = 0}
  (center ∈ circle) →
  (∀ p ∈ result_line, ∃ q ∈ parallel_line, (p.2 - q.2) / (p.1 - q.1) = (center.2 - q.2) / (center.1 - q.1)) →
  (center ∈ result_line) :=
by
  sorry

end NUMINAMATH_CALUDE_line_through_circle_center_parallel_to_line_l819_81912


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l819_81988

theorem units_digit_of_fraction : (30 * 31 * 32 * 33 * 34 * 35) / 7200 ≡ 6 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l819_81988


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_expression_3_simplify_expression_4_l819_81941

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) :
  4 * a^3 + 2 * b - 2 * a^3 + b = 2 * a^3 + 3 * b := by sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) :
  2 * x^2 + 6 * x - 6 - (-2 * x^2 + 4 * x + 1) = 4 * x^2 + 2 * x - 7 := by sorry

-- Problem 3
theorem simplify_expression_3 (a b : ℝ) :
  3 * (3 * a^2 - 2 * a * b) - 2 * (4 * a^2 - a * b) = a^2 - 4 * a * b := by sorry

-- Problem 4
theorem simplify_expression_4 (x y : ℝ) :
  6 * x * y^2 - (2 * x - (1/2) * (2 * x - 4 * x * y^2) - x * y^2) = 5 * x * y^2 - x := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_expression_3_simplify_expression_4_l819_81941


namespace NUMINAMATH_CALUDE_even_function_extension_l819_81907

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_function_extension
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_nonpos : ∀ x ≤ 0, f x = x^3 - x^2) :
  ∀ x > 0, f x = -x^3 - x^2 :=
by sorry

end NUMINAMATH_CALUDE_even_function_extension_l819_81907


namespace NUMINAMATH_CALUDE_cookie_radius_l819_81993

-- Define the cookie equation
def cookie_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 36 = 6*x + 24*y

-- Theorem statement
theorem cookie_radius :
  ∃ (h k r : ℝ), r = Real.sqrt 117 ∧
  ∀ (x y : ℝ), cookie_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_cookie_radius_l819_81993


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l819_81927

theorem purely_imaginary_complex_number (a : ℝ) :
  let z := (a + 2 * Complex.I) / (3 - 4 * Complex.I)
  (∃ (b : ℝ), z = b * Complex.I) → a = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l819_81927


namespace NUMINAMATH_CALUDE_sum_of_xyz_l819_81919

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 24) (hxz : x * z = 48) (hyz : y * z = 72) :
  x + y + z = 22 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l819_81919


namespace NUMINAMATH_CALUDE_juliet_younger_than_ralph_l819_81989

/-- Represents the ages of three siblings -/
structure SiblingAges where
  juliet : ℕ
  maggie : ℕ
  ralph : ℕ

/-- The conditions given in the problem -/
def problem_conditions (ages : SiblingAges) : Prop :=
  ages.juliet = ages.maggie + 3 ∧
  ages.juliet < ages.ralph ∧
  ages.juliet = 10 ∧
  ages.maggie + ages.ralph = 19

/-- The theorem to be proved -/
theorem juliet_younger_than_ralph (ages : SiblingAges) 
  (h : problem_conditions ages) : ages.ralph - ages.juliet = 2 := by
  sorry


end NUMINAMATH_CALUDE_juliet_younger_than_ralph_l819_81989


namespace NUMINAMATH_CALUDE_max_volume_at_eight_l819_81932

/-- The volume of the box as a function of the side length of the removed square -/
def boxVolume (x : ℝ) : ℝ := (48 - 2*x)^2 * x

/-- The derivative of the box volume with respect to x -/
def boxVolumeDerivative (x : ℝ) : ℝ := (48 - 2*x) * (48 - 6*x)

theorem max_volume_at_eight :
  ∃ (x : ℝ), 0 < x ∧ x < 24 ∧
  (∀ (y : ℝ), 0 < y ∧ y < 24 → boxVolume y ≤ boxVolume x) ∧
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_max_volume_at_eight_l819_81932


namespace NUMINAMATH_CALUDE_parabola_symmetry_axis_l819_81945

/-- A parabola defined by y = 2x^2 + bx + c -/
structure Parabola where
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The theorem stating that for a parabola y = 2x^2 + bx + c passing through
    points A(2,5) and B(4,5), the axis of symmetry is x = 3 -/
theorem parabola_symmetry_axis (p : Parabola) (A B : Point)
    (hA : A.y = 2 * A.x^2 + p.b * A.x + p.c)
    (hB : B.y = 2 * B.x^2 + p.b * B.x + p.c)
    (hAx : A.x = 2) (hAy : A.y = 5)
    (hBx : B.x = 4) (hBy : B.y = 5) :
    (A.x + B.x) / 2 = 3 := by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_axis_l819_81945


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_cos_2beta_l819_81950

theorem cos_2alpha_plus_cos_2beta (α β : Real) 
  (h1 : Real.sin α + Real.sin β = 1)
  (h2 : Real.cos α + Real.cos β = 0) :
  Real.cos (2 * α) + Real.cos (2 * β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_cos_2beta_l819_81950


namespace NUMINAMATH_CALUDE_vector_on_line_and_parallel_l819_81971

/-- A line parameterized by x = 5t + 3 and y = 2t + 3 -/
def parameterized_line (t : ℝ) : ℝ × ℝ := (5 * t + 3, 2 * t + 3)

/-- The vector we want to prove is on the line and parallel to (5, 2) -/
def vector : ℝ × ℝ := (-1.5, -0.6)

/-- The direction vector we want our vector to be parallel to -/
def direction : ℝ × ℝ := (5, 2)

theorem vector_on_line_and_parallel :
  ∃ (t : ℝ), parameterized_line t = vector ∧
  ∃ (k : ℝ), vector.1 = k * direction.1 ∧ vector.2 = k * direction.2 := by
  sorry

end NUMINAMATH_CALUDE_vector_on_line_and_parallel_l819_81971


namespace NUMINAMATH_CALUDE_symmetry_xoz_of_point_l819_81952

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Performs symmetry about the xOz plane -/
def symmetryXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

theorem symmetry_xoz_of_point :
  let A : Point3D := { x := 9, y := 8, z := 5 }
  symmetryXOZ A = { x := 9, y := -8, z := 5 } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_xoz_of_point_l819_81952


namespace NUMINAMATH_CALUDE_max_result_ahn_max_result_ahn_achievable_l819_81942

theorem max_result_ahn (n : ℕ) : 
  (100 ≤ n ∧ n ≤ 999) → 3 * (200 + n) ≤ 3597 := by
  sorry

theorem max_result_ahn_achievable : 
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 3 * (200 + n) = 3597 := by
  sorry

end NUMINAMATH_CALUDE_max_result_ahn_max_result_ahn_achievable_l819_81942


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l819_81982

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (n : ℕ), n > 0 ∧ (∃ (p₁ p₂ p₃ p₄ : ℕ), 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0) ∧
  (∀ m : ℕ, m > 0 → (∃ (q₁ q₂ q₃ q₄ : ℕ),
    Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧
    q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
    m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0) → m ≥ n) ∧
  n = 210 :=
sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l819_81982


namespace NUMINAMATH_CALUDE_math_city_intersections_l819_81995

/-- Represents a city with streets and intersections -/
structure City where
  num_streets : ℕ
  num_non_intersections : ℕ

/-- Calculates the number of intersections in a city -/
def num_intersections (c : City) : ℕ :=
  Nat.choose c.num_streets 2 - c.num_non_intersections

/-- Theorem: A city with 10 streets and 2 non-intersections has 43 intersections -/
theorem math_city_intersections :
  let c : City := { num_streets := 10, num_non_intersections := 2 }
  num_intersections c = 43 := by sorry

end NUMINAMATH_CALUDE_math_city_intersections_l819_81995


namespace NUMINAMATH_CALUDE_trapezoid_area_l819_81981

/-- Represents a trapezoid ABCD with point E at the intersection of diagonals -/
structure Trapezoid :=
  (A B C D E : ℝ × ℝ)

/-- The area of a triangle given its vertices -/
def triangle_area (p q r : ℝ × ℝ) : ℝ := sorry

theorem trapezoid_area (ABCD : Trapezoid) : 
  (ABCD.A.1 = ABCD.B.1) ∧  -- AB is parallel to CD (same x-coordinate)
  (ABCD.C.1 = ABCD.D.1) ∧
  (triangle_area ABCD.A ABCD.B ABCD.E = 60) ∧  -- Area of ABE is 60
  (triangle_area ABCD.A ABCD.D ABCD.E = 30) →  -- Area of ADE is 30
  (triangle_area ABCD.A ABCD.B ABCD.C) + 
  (triangle_area ABCD.A ABCD.C ABCD.D) = 135 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l819_81981


namespace NUMINAMATH_CALUDE_cylinder_volume_scale_l819_81978

/-- Given a cylinder with volume V, radius r, and height h, 
    if the radius is tripled and the height is quadrupled, 
    then the new volume V' is 36 times the original volume V. -/
theorem cylinder_volume_scale (V r h : ℝ) (h1 : V = π * r^2 * h) : 
  let V' := π * (3*r)^2 * (4*h)
  V' = 36 * V := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_scale_l819_81978


namespace NUMINAMATH_CALUDE_negation_equivalence_l819_81928

-- Define a triangle
structure Triangle where
  -- Add necessary fields for a triangle

-- Define an obtuse angle
def isObtuseAngle (angle : Real) : Prop := angle > Real.pi / 2

-- Define the property of having at most one obtuse angle
def atMostOneObtuseAngle (t : Triangle) : Prop :=
  ∃ (a b c : Real), isObtuseAngle a → ¬(isObtuseAngle b ∨ isObtuseAngle c)

-- Define the property of having at least two obtuse angles
def atLeastTwoObtuseAngles (t : Triangle) : Prop :=
  ∃ (a b : Real), isObtuseAngle a ∧ isObtuseAngle b

-- Theorem stating that the negation of "at most one obtuse angle" 
-- is equivalent to "at least two obtuse angles"
theorem negation_equivalence (t : Triangle) : 
  ¬(atMostOneObtuseAngle t) ↔ atLeastTwoObtuseAngles t := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l819_81928


namespace NUMINAMATH_CALUDE_max_consecutive_matching_terms_l819_81946

/-- Given two sequences with periods 7 and 13, prove that the maximum number of
consecutive matching terms is the LCM of their periods. -/
theorem max_consecutive_matching_terms
  (a b : ℕ → ℕ)  -- Two sequences of natural numbers
  (ha : ∀ n, a (n + 7) = a n)  -- a has period 7
  (hb : ∀ n, b (n + 13) = b n)  -- b has period 13
  : (∃ k, ∀ i ≤ k, a i = b i) ↔ (∃ k, k = Nat.lcm 7 13 ∧ ∀ i ≤ k, a i = b i) :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_matching_terms_l819_81946


namespace NUMINAMATH_CALUDE_one_percent_as_decimal_l819_81943

theorem one_percent_as_decimal : (1 : ℚ) / 100 = (1 : ℚ) / 100 := by sorry

end NUMINAMATH_CALUDE_one_percent_as_decimal_l819_81943


namespace NUMINAMATH_CALUDE_g_difference_l819_81965

-- Define the function g
noncomputable def g (n : ℤ) : ℝ :=
  (7 + 4 * Real.sqrt 7) / 14 * ((2 + Real.sqrt 7) / 3) ^ n +
  (7 - 4 * Real.sqrt 7) / 14 * ((2 - Real.sqrt 7) / 3) ^ n +
  3

-- Theorem statement
theorem g_difference (n : ℤ) : g (n + 1) - g (n - 1) = g n := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l819_81965


namespace NUMINAMATH_CALUDE_product_first_10000_trailing_zeros_l819_81960

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- The product of the first 10000 natural numbers has 2499 trailing zeros -/
theorem product_first_10000_trailing_zeros :
  trailingZeros 10000 = 2499 := by
  sorry

end NUMINAMATH_CALUDE_product_first_10000_trailing_zeros_l819_81960


namespace NUMINAMATH_CALUDE_base9_85_to_decimal_l819_81921

/-- Converts a two-digit number in base 9 to its decimal representation -/
def base9_to_decimal (tens : Nat) (ones : Nat) : Nat :=
  tens * 9 + ones

/-- States that 85 in base 9 is equal to 77 in decimal -/
theorem base9_85_to_decimal : base9_to_decimal 8 5 = 77 := by
  sorry

end NUMINAMATH_CALUDE_base9_85_to_decimal_l819_81921


namespace NUMINAMATH_CALUDE_expression_simplification_l819_81966

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (1 - 2 / (x + 1)) / ((x^2 - 1) / (3 * x + 3)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l819_81966


namespace NUMINAMATH_CALUDE_excluded_students_average_mark_l819_81937

theorem excluded_students_average_mark
  (N : ℕ)  -- Total number of students
  (A : ℝ)  -- Average mark of all students
  (X : ℕ)  -- Number of excluded students
  (R : ℝ)  -- Average mark of remaining students
  (h1 : N = 10)
  (h2 : A = 80)
  (h3 : X = 5)
  (h4 : R = 90)
  : ∃ E : ℝ,  -- Average mark of excluded students
    N * A = X * E + (N - X) * R ∧ E = 70 :=
by sorry

end NUMINAMATH_CALUDE_excluded_students_average_mark_l819_81937


namespace NUMINAMATH_CALUDE_s_128_eq_one_half_l819_81986

/-- Best decomposition of a positive integer -/
def BestDecomposition (n : ℕ+) : ℕ+ × ℕ+ :=
  sorry

/-- S function for a positive integer -/
def S (n : ℕ+) : ℚ :=
  let (p, q) := BestDecomposition n
  p.val / q.val

/-- Theorem: S(128) = 1/2 -/
theorem s_128_eq_one_half : S 128 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_s_128_eq_one_half_l819_81986


namespace NUMINAMATH_CALUDE_lunks_for_dozen_apples_l819_81914

-- Define the exchange rates
def lunks_per_kunk : ℚ := 7 / 4
def apples_per_kunk : ℚ := 5 / 3

-- Define a dozen
def dozen : ℕ := 12

-- Theorem statement
theorem lunks_for_dozen_apples : 
  ∃ (l : ℚ), l = dozen * (lunks_per_kunk / apples_per_kunk) ∧ l = 12.6 := by
sorry

end NUMINAMATH_CALUDE_lunks_for_dozen_apples_l819_81914


namespace NUMINAMATH_CALUDE_total_energy_calculation_l819_81997

def light_energy (base_watts : ℕ) (multiplier : ℕ) (hours : ℕ) : ℕ :=
  base_watts * multiplier * hours

theorem total_energy_calculation (base_watts : ℕ) (hours : ℕ) 
  (h1 : base_watts = 6)
  (h2 : hours = 2) :
  light_energy base_watts 1 hours + 
  light_energy base_watts 3 hours + 
  light_energy base_watts 4 hours = 96 :=
by sorry

end NUMINAMATH_CALUDE_total_energy_calculation_l819_81997


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l819_81940

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 100) 
  (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l819_81940


namespace NUMINAMATH_CALUDE_engine_problem_solution_l819_81994

/-- Represents the fuel consumption and operation time of two engines -/
structure EnginePair where
  first_consumption : ℝ
  second_consumption : ℝ
  time_difference : ℝ
  consumption_difference : ℝ

/-- Determines if the given fuel consumption rates satisfy the conditions for the two engines -/
def is_valid_solution (pair : EnginePair) (first_rate second_rate : ℝ) : Prop :=
  first_rate > 0 ∧
  second_rate > 0 ∧
  first_rate = second_rate + pair.consumption_difference ∧
  pair.first_consumption / first_rate - pair.second_consumption / second_rate = pair.time_difference

/-- Theorem stating that the given solution satisfies the engine problem conditions -/
theorem engine_problem_solution (pair : EnginePair) 
    (h1 : pair.first_consumption = 300)
    (h2 : pair.second_consumption = 192)
    (h3 : pair.time_difference = 2)
    (h4 : pair.consumption_difference = 6) :
    is_valid_solution pair 30 24 := by
  sorry


end NUMINAMATH_CALUDE_engine_problem_solution_l819_81994


namespace NUMINAMATH_CALUDE_min_sum_of_equal_powers_l819_81973

theorem min_sum_of_equal_powers (x y z : ℕ+) (h : 2^(x:ℕ) = 5^(y:ℕ) ∧ 5^(y:ℕ) = 6^(z:ℕ)) :
  (x:ℕ) + (y:ℕ) + (z:ℕ) ≥ 26 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_equal_powers_l819_81973


namespace NUMINAMATH_CALUDE_election_majority_l819_81936

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 5200 → 
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ).floor - ((1 - winning_percentage) * total_votes : ℚ).floor = 1040 :=
by sorry

end NUMINAMATH_CALUDE_election_majority_l819_81936


namespace NUMINAMATH_CALUDE_divisibility_of_subset_products_l819_81959

def P (A : Finset Nat) : Nat := A.prod id

theorem divisibility_of_subset_products :
  let S : Finset Nat := Finset.range 2010
  let n : Nat := Nat.choose 2010 99
  let subsets : Finset (Finset Nat) := S.powerset.filter (fun A => A.card = 99)
  2010 ∣ subsets.sum P := by sorry

end NUMINAMATH_CALUDE_divisibility_of_subset_products_l819_81959


namespace NUMINAMATH_CALUDE_min_dot_product_hyperbola_l819_81908

/-- The minimum dot product of two vectors from the origin to points on the right branch of x² - y² = 1 is 1 -/
theorem min_dot_product_hyperbola (x₁ y₁ x₂ y₂ : ℝ) : 
  x₁ > 0 → x₂ > 0 → x₁^2 - y₁^2 = 1 → x₂^2 - y₂^2 = 1 → x₁*x₂ + y₁*y₂ ≥ 1 := by
  sorry

#check min_dot_product_hyperbola

end NUMINAMATH_CALUDE_min_dot_product_hyperbola_l819_81908


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l819_81991

theorem sufficient_but_not_necessary (a : ℝ) :
  (∀ x : ℝ, x > 1 → x > a) ∧ (∃ x : ℝ, x > a ∧ x ≤ 1) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l819_81991


namespace NUMINAMATH_CALUDE_sum_bounds_l819_81985

theorem sum_bounds (r s t u : ℝ) 
  (eq : 5*r + 4*s + 3*t + 6*u = 100)
  (h1 : r ≥ s) (h2 : s ≥ t) (h3 : t ≥ u) (h4 : u ≥ 0) :
  20 ≤ r + s + t + u ∧ r + s + t + u ≤ 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_bounds_l819_81985


namespace NUMINAMATH_CALUDE_minimum_pigs_on_farm_l819_81955

theorem minimum_pigs_on_farm (total : ℕ) (pigs : ℕ) : 
  (pigs : ℝ) / total ≥ 0.54 ∧ (pigs : ℝ) / total ≤ 0.57 → pigs ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_minimum_pigs_on_farm_l819_81955


namespace NUMINAMATH_CALUDE_vector_loop_closure_l819_81911

variable {V : Type*} [AddCommGroup V]

theorem vector_loop_closure (A B C : V) :
  (B - A) - (B - C) + (A - C) = (0 : V) := by
  sorry

end NUMINAMATH_CALUDE_vector_loop_closure_l819_81911


namespace NUMINAMATH_CALUDE_star_calculation_l819_81901

-- Define the star operation
def star (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the theorem
theorem star_calculation :
  star (star (star 3 5) 2) 7 = -11/10 :=
by sorry

end NUMINAMATH_CALUDE_star_calculation_l819_81901


namespace NUMINAMATH_CALUDE_difference_theorem_l819_81968

def difference (a b c : ℚ) : ℚ :=
  min (a - b) (min ((a - c) / 2) ((b - c) / 3))

theorem difference_theorem :
  (difference (-2) (-4) 1 = -5/3) ∧
  (2/3 = max
    (max (difference (-2) (-4) 1) (difference (-2) 1 (-4)))
    (max (difference (-4) (-2) 1) (max (difference (-4) 1 (-2)) (max (difference 1 (-4) (-2)) (difference 1 (-2) (-4)))))) ∧
  (∀ x : ℚ, difference (-1) 6 x = 2 ↔ (x = -7 ∨ x = 8)) :=
by sorry

end NUMINAMATH_CALUDE_difference_theorem_l819_81968


namespace NUMINAMATH_CALUDE_ant_distance_l819_81998

def ant_path (n : ℕ) : ℝ × ℝ := 
  let rec path_sum (k : ℕ) : ℝ × ℝ := 
    if k = 0 then (0, 0)
    else 
      let (x, y) := path_sum (k-1)
      match k % 4 with
      | 0 => (x - k, y)
      | 1 => (x, y + k)
      | 2 => (x + k, y)
      | _ => (x, y - k)
  path_sum n

theorem ant_distance : 
  let (x, y) := ant_path 41
  Real.sqrt (x^2 + y^2) = Real.sqrt 221 := by sorry

end NUMINAMATH_CALUDE_ant_distance_l819_81998


namespace NUMINAMATH_CALUDE_brown_eyed_brunettes_l819_81974

theorem brown_eyed_brunettes (total : ℕ) (blue_eyed_blondes : ℕ) (brunettes : ℕ) (brown_eyed : ℕ) 
  (h1 : total = 60)
  (h2 : blue_eyed_blondes = 20)
  (h3 : brunettes = 36)
  (h4 : brown_eyed = 25) :
  total - brunettes - blue_eyed_blondes + brown_eyed = 21 :=
by sorry

end NUMINAMATH_CALUDE_brown_eyed_brunettes_l819_81974


namespace NUMINAMATH_CALUDE_point_Q_in_third_quadrant_l819_81969

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Determine if a point is in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The theorem statement -/
theorem point_Q_in_third_quadrant (m : ℝ) :
  let P : Point := ⟨m + 3, 2 * m + 4⟩
  let Q : Point := ⟨m - 3, m⟩
  (P.y = 0) → isInThirdQuadrant Q :=
by sorry

end NUMINAMATH_CALUDE_point_Q_in_third_quadrant_l819_81969


namespace NUMINAMATH_CALUDE_triangle_properties_l819_81933

/-- Given a triangle ABC with the following properties:
    1. f(x) = sin(2x + B) + √3 cos(2x + B) is an even function
    2. b = f(π/12)
    3. a = 3
    Prove that b = √3 and the area S of triangle ABC is either (3√3)/2 or (3√3)/4 -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  (∀ x, Real.sin (2 * x + B) + Real.sqrt 3 * Real.cos (2 * x + B) =
        Real.sin (2 * -x + B) + Real.sqrt 3 * Real.cos (2 * -x + B)) →
  b = Real.sin (2 * (π / 12) + B) + Real.sqrt 3 * Real.cos (2 * (π / 12) + B) →
  a = 3 →
  b = Real.sqrt 3 ∧ (
    (1/2 * a * b = (3 * Real.sqrt 3) / 2) ∨
    (1/2 * a * b = (3 * Real.sqrt 3) / 4)
  ) := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l819_81933


namespace NUMINAMATH_CALUDE_common_measure_proof_l819_81972

theorem common_measure_proof (a b : ℚ) (ha : a = 4/15) (hb : b = 8/21) :
  ∃ (m : ℚ), m > 0 ∧ ∃ (k₁ k₂ : ℕ), a = k₁ * m ∧ b = k₂ * m :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_common_measure_proof_l819_81972


namespace NUMINAMATH_CALUDE_solution_of_functional_equation_l819_81961

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, x + f x = f (f x)

/-- The theorem stating that the only solution to f(f(x)) = 0 is x = 0 -/
theorem solution_of_functional_equation (f : ℝ → ℝ) (h : FunctionalEquation f) :
  {x : ℝ | f (f x) = 0} = {0} := by
  sorry

end NUMINAMATH_CALUDE_solution_of_functional_equation_l819_81961


namespace NUMINAMATH_CALUDE_proposition_b_proposition_c_proposition_d_l819_81924

-- Define the types for planes and lines
variable (α β : Set (ℝ × ℝ × ℝ))
variable (m n : Set (ℝ × ℝ × ℝ))

-- Define the perpendicular and parallel relations
def perpendicular (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry
def parallel (a b : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the subset relation
def subset (a b : Set (ℝ × ℝ × ℝ)) : Prop := ∀ x, x ∈ a → x ∈ b

-- Define the angle between a line and a plane
def angle (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Theorem B
theorem proposition_b (h1 : perpendicular m α) (h2 : parallel n α) :
  perpendicular m n := sorry

-- Theorem C
theorem proposition_c (h1 : parallel α β) (h2 : subset m α) :
  parallel m β := sorry

-- Theorem D
theorem proposition_d (h1 : parallel m n) (h2 : parallel α β) :
  angle m α = angle n β := sorry

end NUMINAMATH_CALUDE_proposition_b_proposition_c_proposition_d_l819_81924


namespace NUMINAMATH_CALUDE_seed_without_water_impossible_l819_81948

/-- An event is a phenomenon that may or may not occur under certain conditions. -/
structure Event where
  description : String

/-- An impossible event is one that cannot occur under certain conditions. -/
def Event.impossible (e : Event) : Prop := sorry

/-- A certain event is one that will definitely occur under certain conditions. -/
def Event.certain (e : Event) : Prop := sorry

/-- A random event is one that may or may not occur under certain conditions. -/
def Event.random (e : Event) : Prop := sorry

def conductor_heating : Event :=
  { description := "A conductor heats up when conducting electricity" }

def three_points_plane : Event :=
  { description := "Three non-collinear points determine a plane" }

def seed_without_water : Event :=
  { description := "A seed germinates without water" }

def consecutive_lottery : Event :=
  { description := "Someone wins the lottery for two consecutive weeks" }

theorem seed_without_water_impossible :
  Event.impossible seed_without_water ∧
  ¬Event.impossible conductor_heating ∧
  ¬Event.impossible three_points_plane ∧
  ¬Event.impossible consecutive_lottery :=
by sorry

end NUMINAMATH_CALUDE_seed_without_water_impossible_l819_81948


namespace NUMINAMATH_CALUDE_limit_p_n_sqrt_n_l819_81963

/-- The probability that the sum of two randomly selected integers from {1,2,...,n} is a perfect square -/
def p (n : ℕ) : ℝ := sorry

/-- The main theorem stating that the limit of p_n√n as n approaches infinity is 2/3 -/
theorem limit_p_n_sqrt_n :
  ∃ (L : ℝ), L = 2/3 ∧ ∀ ε > 0, ∃ N, ∀ n ≥ N, |p n * Real.sqrt n - L| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_p_n_sqrt_n_l819_81963


namespace NUMINAMATH_CALUDE_fraction_simplification_l819_81902

theorem fraction_simplification (x y : ℚ) 
  (hx : x = 4 / 6) 
  (hy : y = 5 / 8) : 
  (6 * x + 8 * y) / (48 * x * y) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l819_81902


namespace NUMINAMATH_CALUDE_distance_to_line_implies_ab_bound_l819_81909

theorem distance_to_line_implies_ab_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let P : ℝ × ℝ := (1, 1)
  let line (x y : ℝ) := (a + 1) * x + (b + 1) * y - 2 = 0
  let distance_to_line := |((a + 1) * P.1 + (b + 1) * P.2 - 2)| / Real.sqrt ((a + 1)^2 + (b + 1)^2)
  distance_to_line = 1 → a * b ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_line_implies_ab_bound_l819_81909


namespace NUMINAMATH_CALUDE_regression_line_estimate_estimated_y_value_l819_81938

/-- Given a regression line equation and an x-value, calculate the estimated y-value -/
theorem regression_line_estimate (slope intercept x : ℝ) :
  let regression_line := fun (x : ℝ) => slope * x + intercept
  regression_line x = slope * x + intercept := by sorry

/-- The estimated y-value for the given regression line when x = 28 is 390 -/
theorem estimated_y_value :
  let slope : ℝ := 4.75
  let intercept : ℝ := 257
  let x : ℝ := 28
  let regression_line := fun (x : ℝ) => slope * x + intercept
  regression_line x = 390 := by sorry

end NUMINAMATH_CALUDE_regression_line_estimate_estimated_y_value_l819_81938


namespace NUMINAMATH_CALUDE_houses_with_one_pet_l819_81984

/-- Represents the number of houses with different pet combinations in a neighborhood --/
structure PetHouses where
  total : ℕ
  dogs : ℕ
  cats : ℕ
  birds : ℕ
  dogsCats : ℕ
  catsBirds : ℕ
  dogsBirds : ℕ

/-- Theorem stating the number of houses with only one type of pet --/
theorem houses_with_one_pet (h : PetHouses) 
  (h_total : h.total = 75)
  (h_dogs : h.dogs = 40)
  (h_cats : h.cats = 30)
  (h_birds : h.birds = 8)
  (h_dogs_cats : h.dogsCats = 10)
  (h_cats_birds : h.catsBirds = 5)
  (h_dogs_birds : h.dogsBirds = 0) :
  h.dogs + h.cats + h.birds - h.dogsCats - h.catsBirds - h.dogsBirds = 48 := by
  sorry


end NUMINAMATH_CALUDE_houses_with_one_pet_l819_81984


namespace NUMINAMATH_CALUDE_bus_delay_l819_81957

/-- Proves that walking at 4/5 of usual speed results in a 5-minute delay -/
theorem bus_delay (usual_time : ℝ) (h : usual_time = 20) : 
  usual_time * (5/4) - usual_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_bus_delay_l819_81957


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l819_81915

theorem complex_number_in_fourth_quadrant (a b : ℝ) : 
  let z : ℂ := (a^2 - 6*a + 10) + (-b^2 + 4*b - 5)*I
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l819_81915


namespace NUMINAMATH_CALUDE_algebraic_fraction_simplification_l819_81979

theorem algebraic_fraction_simplification (x : ℝ) 
  (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 3) (h4 : x ≠ 5) : 
  (x^2 - 4*x + 3) / (x^2 - 3*x + 2) / ((x^2 - 6*x + 9) / (x^2 - 7*x + 10)) = (x - 5) / (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_fraction_simplification_l819_81979


namespace NUMINAMATH_CALUDE_hexagons_in_50th_ring_l819_81953

/-- Represents the number of hexagons in a ring of a hexagonal arrangement -/
def hexagonsInRing (n : ℕ) : ℕ := 6 * n

/-- The hexagonal arrangement has the following properties:
    1. The center is a regular hexagon of unit side length
    2. Surrounded by rings of unit hexagons
    3. The first ring consists of 6 unit hexagons
    4. The second ring contains 12 unit hexagons -/
axiom hexagonal_arrangement_properties : True

theorem hexagons_in_50th_ring : 
  hexagonsInRing 50 = 300 := by sorry

end NUMINAMATH_CALUDE_hexagons_in_50th_ring_l819_81953


namespace NUMINAMATH_CALUDE_inverse_f_at_46_l819_81922

def f (x : ℝ) : ℝ := 5 * x^3 + 6

theorem inverse_f_at_46 : f⁻¹ 46 = 2 := by sorry

end NUMINAMATH_CALUDE_inverse_f_at_46_l819_81922


namespace NUMINAMATH_CALUDE_remainder_theorem_l819_81956

/-- The polynomial p(x) = 3x^5 + 2x^3 - 5x + 8 -/
def p (x : ℝ) : ℝ := 3 * x^5 + 2 * x^3 - 5 * x + 8

/-- The divisor polynomial d(x) = x^2 - 2x + 1 -/
def d (x : ℝ) : ℝ := x^2 - 2 * x + 1

/-- The remainder polynomial r(x) = 16x - 8 -/
def r (x : ℝ) : ℝ := 16 * x - 8

/-- The quotient polynomial q(x) -/
noncomputable def q (x : ℝ) : ℝ := (p x - r x) / (d x)

theorem remainder_theorem : ∀ x : ℝ, p x = d x * q x + r x := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l819_81956
