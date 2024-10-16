import Mathlib

namespace NUMINAMATH_CALUDE_product_trailing_zeros_l748_74836

def trailing_zeros (n : ℕ) : ℕ := sorry

theorem product_trailing_zeros : 
  let a : ℕ := 35
  let b : ℕ := 4900
  let a_factorization := 5 * 7
  let b_factorization := 2^2 * 5^2 * 7^2
  trailing_zeros (a * b) = 2 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l748_74836


namespace NUMINAMATH_CALUDE_angle_identity_l748_74879

theorem angle_identity (A B C : ℝ) (h : A + B + C = π) :
  Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 - 2 * Real.cos A * Real.cos B * Real.cos C = 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_identity_l748_74879


namespace NUMINAMATH_CALUDE_unique_4digit_number_l748_74845

def is_3digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_4digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem unique_4digit_number :
  ∃! n : ℕ, 
    is_4digit n ∧ 
    (∃ a : ℕ, is_3digit (400 + 10*a + 3) ∧ n = (400 + 10*a + 3) + 984) ∧
    n % 11 = 0 ∧
    (∃ h : ℕ, 10 ≤ h ∧ h ≤ 19 ∧ a + (h - 10) = 10 ∧ n = 1000*h + (n % 1000)) ∧
    n = 1397 :=
sorry

end NUMINAMATH_CALUDE_unique_4digit_number_l748_74845


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_fifteen_fourths_l748_74847

theorem floor_plus_self_eq_fifteen_fourths :
  ∃! (x : ℚ), (⌊x⌋ : ℚ) + x = 15/4 :=
by sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_fifteen_fourths_l748_74847


namespace NUMINAMATH_CALUDE_day2_sale_is_1043_l748_74830

/-- Represents the sales data for a grocer over 5 days -/
structure SalesData where
  average : ℕ
  day1 : ℕ
  day3 : ℕ
  day4 : ℕ
  day5 : ℕ

/-- Calculates the sale on the second day given the sales data -/
def calculateDay2Sale (data : SalesData) : ℕ :=
  5 * data.average - (data.day1 + data.day3 + data.day4 + data.day5)

/-- Proves that the sale on the second day is 1043 given the specified sales data -/
theorem day2_sale_is_1043 (data : SalesData) 
    (h1 : data.average = 625)
    (h2 : data.day1 = 435)
    (h3 : data.day3 = 855)
    (h4 : data.day4 = 230)
    (h5 : data.day5 = 562) :
    calculateDay2Sale data = 1043 := by
  sorry

end NUMINAMATH_CALUDE_day2_sale_is_1043_l748_74830


namespace NUMINAMATH_CALUDE_rex_cards_left_l748_74821

-- Define the number of cards each person has
def nicole_cards : ℕ := 700
def cindy_cards : ℕ := (3 * nicole_cards + (40 * 3 * nicole_cards) / 100)
def tim_cards : ℕ := (4 * cindy_cards) / 5
def rex_joe_cards : ℕ := ((60 * (nicole_cards + cindy_cards + tim_cards)) / 100)

-- Define the number of people sharing Rex and Joe's cards
def num_sharing_people : ℕ := 9

-- Theorem to prove
theorem rex_cards_left : 
  (rex_joe_cards / num_sharing_people) = 399 := by sorry

end NUMINAMATH_CALUDE_rex_cards_left_l748_74821


namespace NUMINAMATH_CALUDE_mike_seashells_l748_74897

/-- The number of seashells Mike found -/
def total_seashells (unbroken_seashells broken_seashells : ℕ) : ℕ :=
  unbroken_seashells + broken_seashells

/-- Theorem stating that Mike found 6 seashells in total -/
theorem mike_seashells : total_seashells 2 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_mike_seashells_l748_74897


namespace NUMINAMATH_CALUDE_factorization_equality_l748_74823

theorem factorization_equality (x y : ℝ) :
  x * (x - y) + y * (y - x) = (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l748_74823


namespace NUMINAMATH_CALUDE_odot_computation_l748_74863

def odot (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem odot_computation : odot 2 (odot 3 (odot 4 5)) = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_odot_computation_l748_74863


namespace NUMINAMATH_CALUDE_inequality_proof_l748_74887

theorem inequality_proof (b c : ℝ) (hb : b > 0) (hc : c > 0) :
  (b - c)^2011 * (b + c)^2011 * (c - b)^2011 ≥ (b^2011 - c^2011) * (b^2011 + c^2011) * (c^2011 - b^2011) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l748_74887


namespace NUMINAMATH_CALUDE_percentage_markup_l748_74825

theorem percentage_markup (cost_price selling_price : ℝ) : 
  cost_price = 7000 →
  selling_price = 8400 →
  (selling_price - cost_price) / cost_price * 100 = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_markup_l748_74825


namespace NUMINAMATH_CALUDE_divisible_by_2000_arrangement_l748_74838

theorem divisible_by_2000_arrangement (nums : Vector ℕ 23) :
  ∃ (arrangement : List (Sum (Prod ℕ ℕ) ℕ)),
    (arrangement.foldl (λ acc x => match x with
      | Sum.inl (a, b) => acc * (a * b)
      | Sum.inr a => acc + a
    ) 0) % 2000 = 0 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_2000_arrangement_l748_74838


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l748_74841

/-- Two arithmetic sequences a and b with their respective sums S and T -/
structure ArithmeticSequences where
  a : ℕ → ℚ
  b : ℕ → ℚ
  S : ℕ → ℚ
  T : ℕ → ℚ

/-- The ratio of sums S_n and T_n for any n -/
def sum_ratio (seq : ArithmeticSequences) : ℕ → ℚ :=
  fun n => seq.S n / seq.T n

/-- The given condition that S_n / T_n = (2n + 1) / (3n + 2) -/
def sum_ratio_condition (seq : ArithmeticSequences) : Prop :=
  ∀ n : ℕ, sum_ratio seq n = (2 * n + 1) / (3 * n + 2)

/-- The theorem to be proved -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequences) 
  (h : sum_ratio_condition seq) : 
  (seq.a 3 + seq.a 11 + seq.a 19) / (seq.b 7 + seq.b 15) = 129 / 130 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l748_74841


namespace NUMINAMATH_CALUDE_soccer_committee_combinations_l748_74868

theorem soccer_committee_combinations : Nat.choose 6 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_soccer_committee_combinations_l748_74868


namespace NUMINAMATH_CALUDE_divisibility_property_l748_74832

theorem divisibility_property (p : ℕ) (h1 : Even p) (h2 : p > 2) :
  ∃ k : ℤ, (p + 1) ^ (p / 2) - 1 = k * p := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l748_74832


namespace NUMINAMATH_CALUDE_weight_problem_l748_74839

/-- Given three weights A, B, and C, prove that their average weights satisfy certain conditions -/
theorem weight_problem (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)  -- The average weight of A, B, and C is 45 kg
  (h2 : (B + C) / 2 = 43)      -- The average weight of B and C is 43 kg
  (h3 : B = 31)                -- The weight of B is 31 kg
  : (A + B) / 2 = 40 :=        -- The average weight of A and B is 40 kg
by sorry

end NUMINAMATH_CALUDE_weight_problem_l748_74839


namespace NUMINAMATH_CALUDE_scientific_notation_of_1_59_million_l748_74884

/-- Expresses 1.59 million in scientific notation -/
theorem scientific_notation_of_1_59_million :
  (1.59 : ℝ) * 1000000 = 1.59 * (10 : ℝ) ^ 6 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1_59_million_l748_74884


namespace NUMINAMATH_CALUDE_remainder_problem_l748_74857

theorem remainder_problem (G : ℕ) (a b : ℕ) (h1 : G = 127) (h2 : a = 1661) (h3 : b = 2045) 
  (h4 : b % G = 13) (h5 : ∀ d : ℕ, d > G → (a % d ≠ 0 ∨ b % d ≠ 0)) :
  a % G = 10 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l748_74857


namespace NUMINAMATH_CALUDE_parabola_c_value_l748_74849

/-- A parabola with vertex (h, k) passing through point (x₀, y₀) has c = 12.5 -/
theorem parabola_c_value (a b c h k x₀ y₀ : ℝ) : 
  (∀ x y, y = a * x^2 + b * x + c) →  -- parabola equation
  (h = 3 ∧ k = -1) →                  -- vertex at (3, -1)
  (x₀ = 1 ∧ y₀ = 5) →                 -- point (1, 5) on parabola
  (∀ x, a * (x - h)^2 + k = a * x^2 + b * x + c) →  -- vertex form equals general form
  (y₀ = a * x₀^2 + b * x₀ + c) →      -- point (1, 5) satisfies equation
  c = 12.5 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l748_74849


namespace NUMINAMATH_CALUDE_inequality_proof_l748_74842

theorem inequality_proof (x y : ℝ) : x^4 + y^4 + 8 ≥ 8*x*y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l748_74842


namespace NUMINAMATH_CALUDE_odd_digits_181_base4_l748_74861

/-- Converts a natural number from base 10 to base 8 --/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a natural number from base 8 to base 4 --/
def base8ToBase4 (n : List ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers --/
def countOddDigits (n : List ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of odd digits in the base 4 representation of 181 (base 10),
    when converted through base 8, is equal to 5 --/
theorem odd_digits_181_base4 : 
  countOddDigits (base8ToBase4 (toBase8 181)) = 5 :=
by sorry

end NUMINAMATH_CALUDE_odd_digits_181_base4_l748_74861


namespace NUMINAMATH_CALUDE_equation_solution_l748_74864

theorem equation_solution (x : ℚ) : 
  (5 * x - 3) / (6 * x - 12) = 4 / 3 → x = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l748_74864


namespace NUMINAMATH_CALUDE_rectangleB_is_leftmost_l748_74822

-- Define a structure for rectangles
structure Rectangle where
  name : Char
  w : Int
  x : Int
  y : Int
  z : Int

-- Define the five rectangles
def rectangleA : Rectangle := ⟨'A', 5, 2, 8, 10⟩
def rectangleB : Rectangle := ⟨'B', 2, 1, 6, 9⟩
def rectangleC : Rectangle := ⟨'C', 4, 7, 3, 0⟩
def rectangleD : Rectangle := ⟨'D', 9, 6, 5, 11⟩
def rectangleE : Rectangle := ⟨'E', 10, 4, 7, 2⟩

-- Define a list of all rectangles
def allRectangles : List Rectangle := [rectangleA, rectangleB, rectangleC, rectangleD, rectangleE]

-- Define a function to check if a rectangle is leftmost
def isLeftmost (r : Rectangle) (rectangles : List Rectangle) : Prop :=
  ∀ other ∈ rectangles, r.w ≤ other.w

-- Theorem statement
theorem rectangleB_is_leftmost :
  isLeftmost rectangleB allRectangles :=
sorry

end NUMINAMATH_CALUDE_rectangleB_is_leftmost_l748_74822


namespace NUMINAMATH_CALUDE_smallest_n_trailing_zeros_l748_74889

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- The smallest integer n ≥ 48 for which the number of trailing zeros in n! is exactly n - 48 -/
theorem smallest_n_trailing_zeros : ∀ n : ℕ, n ≥ 48 → (trailingZeros n = n - 48 → n ≥ 62) ∧ trailingZeros 62 = 62 - 48 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_trailing_zeros_l748_74889


namespace NUMINAMATH_CALUDE_last_digit_product_l748_74818

theorem last_digit_product : (3^65 * 6^59 * 7^71) % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_last_digit_product_l748_74818


namespace NUMINAMATH_CALUDE_triangle_property_l748_74804

theorem triangle_property (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  (A + B + C = π) →
  -- Given condition
  (2 * Real.cos B * Real.cos C + 1 = 2 * Real.sin B * Real.sin C) →
  (b + c = 4) →
  -- Conclusions
  (A = π / 3) ∧
  (∀ (area : Real), area = 1/2 * b * c * Real.sin A → area ≤ Real.sqrt 3) ∧
  (∃ (area : Real), area = 1/2 * b * c * Real.sin A ∧ area = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l748_74804


namespace NUMINAMATH_CALUDE_power_of_three_squared_l748_74890

theorem power_of_three_squared : 3^2 = 9 := by sorry

end NUMINAMATH_CALUDE_power_of_three_squared_l748_74890


namespace NUMINAMATH_CALUDE_circles_intersect_l748_74813

/-- Two circles are intersecting if the distance between their centers is less than the sum of their radii
    and greater than the absolute difference of their radii. -/
def are_intersecting (r₁ r₂ d : ℝ) : Prop :=
  d < r₁ + r₂ ∧ d > |r₁ - r₂|

/-- Given two circles with radii 3 and 5, whose centers are 2 units apart, prove they are intersecting. -/
theorem circles_intersect : are_intersecting 3 5 2 := by
  sorry

#check circles_intersect

end NUMINAMATH_CALUDE_circles_intersect_l748_74813


namespace NUMINAMATH_CALUDE_forest_trees_count_l748_74880

/-- The side length of the square-shaped street in meters -/
def street_side_length : ℝ := 100

/-- The area of the square-shaped street in square meters -/
def street_area : ℝ := street_side_length ^ 2

/-- The area of the forest in square meters -/
def forest_area : ℝ := 3 * street_area

/-- The number of trees per square meter in the forest -/
def trees_per_square_meter : ℝ := 4

/-- The total number of trees in the forest -/
def total_trees : ℝ := forest_area * trees_per_square_meter

theorem forest_trees_count : total_trees = 120000 := by
  sorry

end NUMINAMATH_CALUDE_forest_trees_count_l748_74880


namespace NUMINAMATH_CALUDE_necklace_price_calculation_l748_74867

def polo_shirt_price : ℕ := 26
def polo_shirt_quantity : ℕ := 3
def necklace_quantity : ℕ := 2
def computer_game_price : ℕ := 90
def computer_game_quantity : ℕ := 1
def rebate : ℕ := 12
def total_cost_after_rebate : ℕ := 322

theorem necklace_price_calculation (necklace_price : ℕ) : 
  polo_shirt_price * polo_shirt_quantity + 
  necklace_price * necklace_quantity + 
  computer_game_price * computer_game_quantity - 
  rebate = total_cost_after_rebate → 
  necklace_price = 83 := by sorry

end NUMINAMATH_CALUDE_necklace_price_calculation_l748_74867


namespace NUMINAMATH_CALUDE_cube_dimension_ratio_l748_74848

theorem cube_dimension_ratio (v1 v2 : ℝ) (h1 : v1 = 27) (h2 : v2 = 216) :
  (v2 / v1) ^ (1/3 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_dimension_ratio_l748_74848


namespace NUMINAMATH_CALUDE_sock_count_proof_l748_74840

def total_socks (john_initial mary_initial kate_initial : ℕ)
                (john_thrown john_bought : ℕ)
                (mary_thrown mary_bought : ℕ)
                (kate_thrown kate_bought : ℕ) : ℕ :=
  (john_initial - john_thrown + john_bought) +
  (mary_initial - mary_thrown + mary_bought) +
  (kate_initial - kate_thrown + kate_bought)

theorem sock_count_proof :
  total_socks 33 20 15 19 13 6 10 5 8 = 69 := by
  sorry

end NUMINAMATH_CALUDE_sock_count_proof_l748_74840


namespace NUMINAMATH_CALUDE_gcd_840_1764_l748_74870

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l748_74870


namespace NUMINAMATH_CALUDE_triangle_angle_relation_minimum_l748_74855

theorem triangle_angle_relation_minimum (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hSum : A + B + C = π) (hTriangle : 3 * (Real.cos (2 * A) - Real.cos (2 * C)) = 1 - Real.cos (2 * B)) :
  ∃ (x : ℝ), ∀ (y : ℝ), 
    (Real.sin C / (Real.sin A * Real.sin B) + Real.cos C / Real.sin C) ≥ y → 
    y ≥ 2 * Real.sqrt 7 / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_minimum_l748_74855


namespace NUMINAMATH_CALUDE_inscribed_circle_area_isosceles_trapezoid_l748_74814

/-- The area of a circle inscribed in an isosceles trapezoid -/
theorem inscribed_circle_area_isosceles_trapezoid 
  (a : ℝ) 
  (h_positive : a > 0) 
  (h_isosceles : IsoscelesTrapezoid) 
  (h_angle : AngleAtSmallerBase = 120) : 
  AreaOfInscribedCircle = π * a^2 / 12 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_isosceles_trapezoid_l748_74814


namespace NUMINAMATH_CALUDE_roots_sum_and_product_l748_74872

theorem roots_sum_and_product (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 4 = 0 → 
  x₂^2 - 2*x₂ - 4 = 0 → 
  x₁ + x₂ + x₁*x₂ = -2 :=
by sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_l748_74872


namespace NUMINAMATH_CALUDE_dwarf_system_stabilizes_l748_74819

-- Define the color of a dwarf's house
inductive Color
| Red
| White

-- Define the state of the dwarf system
structure DwarfSystem :=
  (houses : Fin 12 → Color)
  (friends : Fin 12 → Set (Fin 12))

-- Define a single step in the system
def step (sys : DwarfSystem) (i : Fin 12) : DwarfSystem := sorry

-- Define the relation between two states
def reaches (initial final : DwarfSystem) : Prop := sorry

-- Theorem statement
theorem dwarf_system_stabilizes (initial : DwarfSystem) :
  ∃ (final : DwarfSystem), reaches initial final ∧ ∀ i, step final i = final :=
sorry

end NUMINAMATH_CALUDE_dwarf_system_stabilizes_l748_74819


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l748_74859

theorem hexagon_angle_measure (F I U R G E : ℝ) : 
  -- Hexagon angle sum is 720°
  F + I + U + R + G + E = 720 →
  -- Four angles are congruent
  F = I ∧ F = U ∧ F = R →
  -- G and E are supplementary
  G + E = 180 →
  -- Prove that E is 45°
  E = 45 := by
sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l748_74859


namespace NUMINAMATH_CALUDE_function_bound_l748_74886

theorem function_bound (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x : ℝ, f x = Real.sqrt 3 * Real.sin (3 * x) + Real.cos (3 * x)) →
  (∀ x : ℝ, |f x| ≤ a) →
  a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_function_bound_l748_74886


namespace NUMINAMATH_CALUDE_division_problem_l748_74802

theorem division_problem (x : ℝ) (h : 82.04 / x = 28) : x = 2.93 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l748_74802


namespace NUMINAMATH_CALUDE_even_perfect_square_ablab_l748_74874

theorem even_perfect_square_ablab : 
  ∃! n : ℕ, 
    (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ 
      n = 10000 * a + 1000 * b + 100 + 10 * a + b) ∧ 
    (∃ m : ℕ, n = m^2) ∧ 
    (∃ k : ℕ, n = 2 * k) ∧
    n = 76176 :=
by
  sorry

end NUMINAMATH_CALUDE_even_perfect_square_ablab_l748_74874


namespace NUMINAMATH_CALUDE_smallest_n_cookie_boxes_l748_74860

theorem smallest_n_cookie_boxes : ∃ (n : ℕ), n > 0 ∧ 12 ∣ (17 * n - 1) ∧ ∀ (m : ℕ), m > 0 ∧ 12 ∣ (17 * m - 1) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_cookie_boxes_l748_74860


namespace NUMINAMATH_CALUDE_second_month_sale_l748_74894

/-- Represents the sales data for a grocery shop over 6 months -/
structure GrocerySales where
  month1 : ℕ
  month2 : ℕ
  month3 : ℕ
  month4 : ℕ
  month5 : ℕ
  month6 : ℕ

/-- Calculates the average sale over 6 months -/
def average_sale (sales : GrocerySales) : ℚ :=
  (sales.month1 + sales.month2 + sales.month3 + sales.month4 + sales.month5 + sales.month6) / 6

/-- Theorem stating the conditions and the result to be proved -/
theorem second_month_sale 
  (sales : GrocerySales)
  (h1 : sales.month1 = 6435)
  (h2 : sales.month3 = 7230)
  (h3 : sales.month4 = 6562)
  (h4 : sales.month6 = 4991)
  (h5 : average_sale sales = 6500) :
  sales.month2 = 13782 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_l748_74894


namespace NUMINAMATH_CALUDE_sequence_general_term_l748_74846

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n + 1) = a n + 2) :
  ∀ n : ℕ, a n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l748_74846


namespace NUMINAMATH_CALUDE_tom_initial_investment_l748_74898

/-- Represents the initial investment of Tom in rupees -/
def tom_investment : ℕ := 30000

/-- Represents Jose's investment in rupees -/
def jose_investment : ℕ := 45000

/-- Represents the total profit after one year in rupees -/
def total_profit : ℕ := 63000

/-- Represents Jose's share of the profit in rupees -/
def jose_profit : ℕ := 35000

/-- Represents the number of months Tom invested -/
def tom_months : ℕ := 12

/-- Represents the number of months Jose invested -/
def jose_months : ℕ := 10

theorem tom_initial_investment :
  tom_investment * tom_months * jose_profit = jose_investment * jose_months * (total_profit - jose_profit) :=
sorry

end NUMINAMATH_CALUDE_tom_initial_investment_l748_74898


namespace NUMINAMATH_CALUDE_marys_number_l748_74871

theorem marys_number (n : ℕ) : 
  150 ∣ n → 
  45 ∣ n → 
  1000 ≤ n → 
  n ≤ 3000 → 
  n = 1350 ∨ n = 1800 ∨ n = 2250 ∨ n = 2700 := by
sorry

end NUMINAMATH_CALUDE_marys_number_l748_74871


namespace NUMINAMATH_CALUDE_smaller_fraction_l748_74878

theorem smaller_fraction (x y : ℚ) (sum_eq : x + y = 13/14) (prod_eq : x * y = 1/8) :
  min x y = 1/6 := by sorry

end NUMINAMATH_CALUDE_smaller_fraction_l748_74878


namespace NUMINAMATH_CALUDE_no_positive_roots_l748_74820

theorem no_positive_roots :
  ∀ x : ℝ, x > 0 → x^3 + 6*x^2 + 11*x + 6 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_roots_l748_74820


namespace NUMINAMATH_CALUDE_max_value_of_z_l748_74844

-- Define the system of inequalities and z
def system (x y : ℝ) : Prop :=
  x + y - Real.sqrt 2 ≤ 0 ∧
  x - y + Real.sqrt 2 ≥ 0 ∧
  y ≥ 0

def z (x y : ℝ) : ℝ := 2 * x - y

-- State the theorem
theorem max_value_of_z :
  ∃ (max_z : ℝ) (x_max y_max : ℝ),
    system x_max y_max ∧
    z x_max y_max = max_z ∧
    max_z = 2 * Real.sqrt 2 ∧
    x_max = Real.sqrt 2 ∧
    y_max = 0 ∧
    ∀ (x y : ℝ), system x y → z x y ≤ max_z :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_z_l748_74844


namespace NUMINAMATH_CALUDE_phyllis_gardens_tomato_percentage_l748_74877

/-- Represents a garden with a total number of plants and a fraction of tomato plants -/
structure Garden where
  total_plants : ℕ
  tomato_fraction : ℚ

/-- Calculates the percentage of tomato plants in two gardens combined -/
def combined_tomato_percentage (g1 g2 : Garden) : ℚ :=
  let total_plants := g1.total_plants + g2.total_plants
  let total_tomatoes := g1.total_plants * g1.tomato_fraction + g2.total_plants * g2.tomato_fraction
  (total_tomatoes / total_plants) * 100

/-- Theorem stating that the percentage of tomato plants in Phyllis's two gardens is 20% -/
theorem phyllis_gardens_tomato_percentage :
  let garden1 : Garden := { total_plants := 20, tomato_fraction := 1/10 }
  let garden2 : Garden := { total_plants := 15, tomato_fraction := 1/3 }
  combined_tomato_percentage garden1 garden2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_phyllis_gardens_tomato_percentage_l748_74877


namespace NUMINAMATH_CALUDE_share_of_y_is_36_l748_74826

/-- The share of y in rupees when a sum is divided among x, y, and z -/
def share_of_y (total : ℚ) (x_share : ℚ) (y_share : ℚ) (z_share : ℚ) : ℚ :=
  (y_share / x_share) * (total / (1 + y_share / x_share + z_share / x_share))

/-- Theorem: The share of y is 36 rupees given the problem conditions -/
theorem share_of_y_is_36 :
  share_of_y 156 1 (45/100) (1/2) = 36 := by
  sorry

#eval share_of_y 156 1 (45/100) (1/2)

end NUMINAMATH_CALUDE_share_of_y_is_36_l748_74826


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_ab_value_l748_74850

theorem ellipse_hyperbola_ab_value (a b : ℝ) : 
  (∃ (c : ℝ), c = 5 ∧ c^2 = b^2 - a^2) →
  (∃ (d : ℝ), d = 8 ∧ d^2 = a^2 + b^2) →
  |a * b| = Real.sqrt 3471 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_ab_value_l748_74850


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l748_74851

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 13)
  (sum_products_eq : a * b + a * c + b * c = 32) :
  a^3 + b^3 + c^3 - 3*a*b*c = 949 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l748_74851


namespace NUMINAMATH_CALUDE_intersection_is_empty_l748_74817

open Set

def A : Set ℝ := Ioc (-1) 3
def B : Set ℝ := {2, 4}

theorem intersection_is_empty : A ∩ B = ∅ := by sorry

end NUMINAMATH_CALUDE_intersection_is_empty_l748_74817


namespace NUMINAMATH_CALUDE_no_primes_in_range_l748_74892

theorem no_primes_in_range (n : ℕ) (h : n > 2) :
  ∀ p, Prime p → ¬(n! + 2 < p ∧ p < n! + n + 1) :=
sorry

end NUMINAMATH_CALUDE_no_primes_in_range_l748_74892


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l748_74835

theorem ceiling_floor_sum : ⌈(7 : ℚ) / 3⌉ + ⌊-(7 : ℚ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l748_74835


namespace NUMINAMATH_CALUDE_price_restoration_l748_74801

theorem price_restoration (original_price : ℝ) (markup_percentage : ℝ) (reduction_percentage : ℝ) : 
  markup_percentage = 25 →
  reduction_percentage = 20 →
  original_price * (1 + markup_percentage / 100) * (1 - reduction_percentage / 100) = original_price :=
by
  sorry

end NUMINAMATH_CALUDE_price_restoration_l748_74801


namespace NUMINAMATH_CALUDE_choose_four_different_suits_standard_deck_l748_74828

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (h1 : cards = suits * cards_per_suit)

/-- The number of ways to choose 4 cards from different suits in a standard deck -/
def choose_four_different_suits (d : Deck) : Nat :=
  d.cards_per_suit ^ d.suits

/-- Theorem stating that the number of ways to choose 4 cards from different suits
    in a standard deck of 52 cards is 28,561 -/
theorem choose_four_different_suits_standard_deck :
  ∃ (d : Deck), d.cards = 52 ∧ d.suits = 4 ∧ d.cards_per_suit = 13 ∧
  choose_four_different_suits d = 28561 :=
sorry

end NUMINAMATH_CALUDE_choose_four_different_suits_standard_deck_l748_74828


namespace NUMINAMATH_CALUDE_power_of_three_equality_l748_74876

theorem power_of_three_equality (x : ℕ) :
  3^x = 3^20 * 3^20 * 3^18 + 3^19 * 3^20 * 3^19 + 3^18 * 3^21 * 3^19 → x = 59 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_equality_l748_74876


namespace NUMINAMATH_CALUDE_base7_243_to_base10_l748_74852

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (d2 d1 d0 : ℕ) : ℕ :=
  d0 + d1 * 7 + d2 * 7^2

/-- The base 10 equivalent of 243 in base 7 --/
theorem base7_243_to_base10 : base7ToBase10 2 4 3 = 129 := by
  sorry

end NUMINAMATH_CALUDE_base7_243_to_base10_l748_74852


namespace NUMINAMATH_CALUDE_note_count_l748_74869

theorem note_count (total_amount : ℕ) (denomination_1 : ℕ) (denomination_5 : ℕ) (denomination_10 : ℕ) :
  total_amount = 192 ∧
  denomination_1 = 1 ∧
  denomination_5 = 5 ∧
  denomination_10 = 10 ∧
  (∃ (x : ℕ), x * denomination_1 + x * denomination_5 + x * denomination_10 = total_amount) →
  (∃ (x : ℕ), x * 3 = 36 ∧ x * denomination_1 + x * denomination_5 + x * denomination_10 = total_amount) :=
by sorry

end NUMINAMATH_CALUDE_note_count_l748_74869


namespace NUMINAMATH_CALUDE_lisa_hourly_wage_l748_74831

/-- Calculates the hourly wage of Lisa given Greta's work hours, hourly rate, and Lisa's equivalent work hours -/
theorem lisa_hourly_wage (greta_hours : ℕ) (greta_rate : ℚ) (lisa_hours : ℕ) : 
  greta_hours = 40 → 
  greta_rate = 12 → 
  lisa_hours = 32 → 
  (greta_hours * greta_rate) / lisa_hours = 15 := by
sorry

end NUMINAMATH_CALUDE_lisa_hourly_wage_l748_74831


namespace NUMINAMATH_CALUDE_hasan_plates_removal_l748_74899

/-- The weight of each plate in ounces -/
def plate_weight : ℕ := 10

/-- The weight limit for each box in pounds -/
def box_weight_limit : ℕ := 20

/-- The number of plates initially packed in the box -/
def initial_plates : ℕ := 38

/-- The number of ounces in a pound -/
def ounces_per_pound : ℕ := 16

/-- The number of plates Hasan needs to remove from the box -/
def plates_to_remove : ℕ := 6

theorem hasan_plates_removal :
  plates_to_remove = 
    (initial_plates * plate_weight - box_weight_limit * ounces_per_pound) / plate_weight :=
by sorry

end NUMINAMATH_CALUDE_hasan_plates_removal_l748_74899


namespace NUMINAMATH_CALUDE_valid_numbers_l748_74808

def is_valid_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  (n / 10000 % 10) * 5 = (n / 1000 % 10) ∧
  (n / 10000 % 10) * (n / 1000 % 10) * (n / 100 % 10) * (n / 10 % 10) * (n % 10) = 1000

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n ↔ n = 15558 ∨ n = 15585 ∨ n = 15855 :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l748_74808


namespace NUMINAMATH_CALUDE_toothpick_grid_30_50_l748_74806

/-- Represents a grid of toothpicks -/
structure ToothpickGrid where
  length : ℕ
  width : ℕ

/-- Calculates the total number of toothpicks in a grid -/
def total_toothpicks (grid : ToothpickGrid) : ℕ :=
  (grid.length + 1) * grid.width + (grid.width + 1) * grid.length

/-- Calculates the area enclosed by a grid -/
def enclosed_area (grid : ToothpickGrid) : ℕ :=
  grid.length * grid.width

/-- Theorem stating the properties of a 30x50 toothpick grid -/
theorem toothpick_grid_30_50 :
  let grid : ToothpickGrid := ⟨30, 50⟩
  total_toothpicks grid = 3080 ∧ enclosed_area grid = 1500 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_grid_30_50_l748_74806


namespace NUMINAMATH_CALUDE_tangent_line_equation_l748_74837

/-- The curve S defined by y = x³ + 4 -/
def S : ℝ → ℝ := fun x ↦ x^3 + 4

/-- The point A -/
def A : ℝ × ℝ := (1, 5)

/-- The first possible tangent line equation: 3x - y - 2 = 0 -/
def tangent1 (x y : ℝ) : Prop := 3 * x - y - 2 = 0

/-- The second possible tangent line equation: 3x - 4y + 17 = 0 -/
def tangent2 (x y : ℝ) : Prop := 3 * x - 4 * y + 17 = 0

/-- Theorem: The tangent line to curve S passing through point A
    is either tangent1 or tangent2 -/
theorem tangent_line_equation :
  ∃ (x y : ℝ), (y = S x ∧ (x, y) ≠ A) →
  (∀ (h k : ℝ), tangent1 h k ∨ tangent2 h k ↔ 
    (k - A.2) / (h - A.1) = 3 * x^2 ∧ k = S h) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l748_74837


namespace NUMINAMATH_CALUDE_log_expression_equals_one_l748_74810

-- Define the logarithm base 2 function
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- Define the common logarithm (base 10) function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_one :
  2 * lg (Real.sqrt 2) + log2 5 * lg 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_one_l748_74810


namespace NUMINAMATH_CALUDE_negative_two_cubed_minus_squared_l748_74896

theorem negative_two_cubed_minus_squared : (-2)^3 - (-2)^2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_cubed_minus_squared_l748_74896


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l748_74815

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 3 + a 5 + a 11 + a 13 = 80 →
  a 8 = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l748_74815


namespace NUMINAMATH_CALUDE_circle_center_l748_74882

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 2*y - 15 = 0

/-- The center of a circle given by its coordinates -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Theorem stating that the center of the circle with the given equation is (3, -1) -/
theorem circle_center : 
  ∃ (center : CircleCenter), center.x = 3 ∧ center.y = -1 ∧
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - center.x)^2 + (y - center.y)^2 = 25 :=
sorry

end NUMINAMATH_CALUDE_circle_center_l748_74882


namespace NUMINAMATH_CALUDE_no_positive_integer_sequence_exists_positive_irrational_sequence_l748_74843

-- Part 1: No sequence of positive integers satisfying the condition
theorem no_positive_integer_sequence :
  ¬ ∃ (a : ℕ → ℕ+), ∀ n : ℕ, (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2)) := by sorry

-- Part 2: Existence of a sequence of positive irrational numbers satisfying the condition
theorem exists_positive_irrational_sequence :
  ∃ (a : ℕ → ℝ), (∀ n : ℕ, Irrational (a n) ∧ a n > 0) ∧
    (∀ n : ℕ, (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2))) := by sorry

end NUMINAMATH_CALUDE_no_positive_integer_sequence_exists_positive_irrational_sequence_l748_74843


namespace NUMINAMATH_CALUDE_saree_original_price_l748_74888

/-- The original price of sarees given successive discounts -/
theorem saree_original_price (final_price : ℝ) 
  (h1 : final_price = 380.16) 
  (h2 : final_price = 0.9 * 0.8 * original_price) : 
  original_price = 528 :=
by
  sorry

#check saree_original_price

end NUMINAMATH_CALUDE_saree_original_price_l748_74888


namespace NUMINAMATH_CALUDE_square_even_implies_even_sqrt_2_irrational_l748_74809

-- Part 1: If p² is even, then p is even
theorem square_even_implies_even (p : ℤ) : Even (p^2) → Even p := by sorry

-- Part 2: √2 is irrational
theorem sqrt_2_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a : ℚ) / b = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_square_even_implies_even_sqrt_2_irrational_l748_74809


namespace NUMINAMATH_CALUDE_ratio_equality_l748_74827

theorem ratio_equality (x y : ℚ) (h : x / (2 * y) = 27) :
  (7 * x + 6 * y) / (x - 2 * y) = 96 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l748_74827


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l748_74885

theorem cubic_root_sum_cubes (p q r : ℝ) : 
  (p^3 - 2*p^2 + 3*p - 4 = 0) ∧ 
  (q^3 - 2*q^2 + 3*q - 4 = 0) ∧ 
  (r^3 - 2*r^2 + 3*r - 4 = 0) →
  p^3 + q^3 + r^3 = 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l748_74885


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l748_74854

-- Define the markup and discount percentages
def markup : ℝ := 0.40
def discount : ℝ := 0.15

-- Theorem statement
theorem merchant_profit_percentage :
  let marked_price := 1 + markup
  let selling_price := marked_price * (1 - discount)
  (selling_price - 1) * 100 = 19 := by
  sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l748_74854


namespace NUMINAMATH_CALUDE_sprint_medal_awarding_ways_l748_74866

/-- The number of ways to award medals in an international sprint final --/
def medalAwardingWays (totalSprinters : ℕ) (americanSprinters : ℕ) (medals : ℕ) : ℕ :=
  -- We'll define this function without implementation
  sorry

/-- Theorem stating the number of ways to award medals under given conditions --/
theorem sprint_medal_awarding_ways :
  medalAwardingWays 10 4 3 = 696 :=
by
  sorry

end NUMINAMATH_CALUDE_sprint_medal_awarding_ways_l748_74866


namespace NUMINAMATH_CALUDE_remainder_3_pow_2000_mod_17_l748_74873

theorem remainder_3_pow_2000_mod_17 : 3^2000 % 17 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_2000_mod_17_l748_74873


namespace NUMINAMATH_CALUDE_find_b_l748_74853

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - b else 2^x

theorem find_b : ∃ b : ℝ, f b (f b (5/6)) = 4 ∧ b = 11/8 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l748_74853


namespace NUMINAMATH_CALUDE_snackies_leftover_l748_74833

theorem snackies_leftover (m : ℕ) (h : m % 8 = 5) : (4 * m) % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_snackies_leftover_l748_74833


namespace NUMINAMATH_CALUDE_circle_on_parabola_tangent_to_axes_l748_74824

/-- A circle whose center lies on a parabola and is tangent to the parabola's axis and y-axis -/
theorem circle_on_parabola_tangent_to_axes :
  ∃ (x₀ y₀ r : ℝ),
    (x₀ < 0) ∧                             -- Center is on the left side of y-axis
    (y₀ = (1/2) * x₀^2) ∧                  -- Center lies on the parabola
    (∀ x y : ℝ,
      (x + 1)^2 + (y - 1/2)^2 = 1 ↔        -- Equation of the circle
      (x - x₀)^2 + (y - y₀)^2 = r^2) ∧     -- Standard form of circle equation
    (r = |x₀|) ∧                           -- Circle is tangent to y-axis
    (r = |y₀ - 1/2|)                       -- Circle is tangent to parabola's axis
  := by sorry

end NUMINAMATH_CALUDE_circle_on_parabola_tangent_to_axes_l748_74824


namespace NUMINAMATH_CALUDE_gcd_problems_l748_74803

theorem gcd_problems : 
  (Nat.gcd 120 168 = 24) ∧ (Nat.gcd 459 357 = 51) := by
  sorry

end NUMINAMATH_CALUDE_gcd_problems_l748_74803


namespace NUMINAMATH_CALUDE_hyperbola_equation_l748_74807

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ c : ℝ, c = Real.sqrt 5 ∧ c^2 = a^2 + b^2) → 
  (b / a = 1 / 2) → 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 4 - y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l748_74807


namespace NUMINAMATH_CALUDE_min_value_trig_expression_equality_achieved_l748_74811

theorem min_value_trig_expression (x : ℝ) :
  (4 * Real.sin x * Real.cos x + 3) / (Real.cos x)^2 ≥ 5/3 :=
by sorry

theorem equality_achieved :
  ∃ x : ℝ, (4 * Real.sin x * Real.cos x + 3) / (Real.cos x)^2 = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_equality_achieved_l748_74811


namespace NUMINAMATH_CALUDE_expression_value_l748_74816

theorem expression_value (x y : ℝ) (h1 : x + y = 17) (h2 : x * y = 17) :
  (x^2 - 17*x) * (y + 17/y) = -289 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l748_74816


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l748_74858

theorem arithmetic_sequence_terms (a₁ aₙ : ℤ) (n : ℕ) : 
  a₁ = -1 → aₙ = 89 → aₙ = a₁ + (n - 1) * ((aₙ - a₁) / (n - 1)) → n = 46 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l748_74858


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l748_74865

theorem concentric_circles_radii_difference
  (r R : ℝ) -- radii of the smaller and larger circles
  (h : r > 0) -- radius is positive
  (area_ratio : π * R^2 = 4 * (π * r^2)) -- area ratio is 1:4
  : R - r = r := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l748_74865


namespace NUMINAMATH_CALUDE_waitress_tips_fraction_l748_74893

theorem waitress_tips_fraction (salary : ℝ) (tips : ℝ) (h : tips = 2/4 * salary) :
  tips / (salary + tips) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_waitress_tips_fraction_l748_74893


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l748_74856

theorem triangle_angle_problem (A B C : ℝ) 
  (h1 : B = A + 10)
  (h2 : C = B + 10)
  (h3 : A + B + C = 180) :
  B = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l748_74856


namespace NUMINAMATH_CALUDE_unclaimed_books_fraction_l748_74875

/-- Represents the fraction of books each person takes -/
def take_books (total : ℚ) (fraction : ℚ) (remaining : ℚ) : ℚ :=
  fraction * remaining

/-- The fraction of books that goes unclaimed after all four people take their share -/
def unclaimed_fraction : ℚ :=
  let total := 1
  let al_takes := take_books total (2/5) total
  let bert_takes := take_books total (3/10) (total - al_takes)
  let carl_takes := take_books total (1/5) (total - al_takes - bert_takes)
  let dan_takes := take_books total (1/10) (total - al_takes - bert_takes - carl_takes)
  total - (al_takes + bert_takes + carl_takes + dan_takes)

theorem unclaimed_books_fraction :
  unclaimed_fraction = 1701 / 2500 :=
sorry

end NUMINAMATH_CALUDE_unclaimed_books_fraction_l748_74875


namespace NUMINAMATH_CALUDE_triangle_solution_l748_74834

/-- Given a triangle ABC with side lengths a, b, c and angles α, β, γ,
    if a : b = 1 : 2, α : β = 1 : 3, and c = 5 cm,
    then a = 5√3/3 cm, b = 10√3/3 cm, α = 30°, β = 90°, and γ = 60°. -/
theorem triangle_solution (a b c : ℝ) (α β γ : ℝ) : 
  a / b = 1 / 2 →
  α / β = 1 / 3 →
  c = 5 →
  a = 5 * Real.sqrt 3 / 3 ∧
  b = 10 * Real.sqrt 3 / 3 ∧
  α = Real.pi / 6 ∧
  β = Real.pi / 2 ∧
  γ = Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_solution_l748_74834


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_negative_48_to_0_l748_74800

def arithmeticSequenceSum (a l d : ℤ) : ℤ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem arithmetic_sequence_sum_negative_48_to_0 :
  arithmeticSequenceSum (-48) 0 2 = -600 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_negative_48_to_0_l748_74800


namespace NUMINAMATH_CALUDE_card_arrangement_probability_l748_74895

theorem card_arrangement_probability : 
  let n : ℕ := 8  -- total number of cards
  let k : ℕ := 3  -- number of identical cards (О in this case)
  let total_permutations : ℕ := n.factorial
  let favorable_permutations : ℕ := k.factorial
  (favorable_permutations : ℚ) / total_permutations = 1 / 6720 :=
by sorry

end NUMINAMATH_CALUDE_card_arrangement_probability_l748_74895


namespace NUMINAMATH_CALUDE_addition_subtraction_elimination_not_factorization_l748_74829

-- Define the type for factorization methods
inductive FactorizationMethod
  | TakeOutCommonFactor
  | CrossMultiplication
  | Formula
  | AdditionSubtractionElimination

-- Define a predicate to check if a method is a factorization method
def is_factorization_method : FactorizationMethod → Prop
  | FactorizationMethod.TakeOutCommonFactor => true
  | FactorizationMethod.CrossMultiplication => true
  | FactorizationMethod.Formula => true
  | FactorizationMethod.AdditionSubtractionElimination => false

-- Theorem statement
theorem addition_subtraction_elimination_not_factorization :
  ¬(is_factorization_method FactorizationMethod.AdditionSubtractionElimination) :=
by sorry

end NUMINAMATH_CALUDE_addition_subtraction_elimination_not_factorization_l748_74829


namespace NUMINAMATH_CALUDE_first_boy_speed_l748_74805

/-- The speed of the second boy in km/h -/
def second_boy_speed : ℝ := 7.5

/-- The time the boys walk in hours -/
def walking_time : ℝ := 16

/-- The distance between the boys after walking in km -/
def final_distance : ℝ := 32

/-- Theorem stating the speed of the first boy -/
theorem first_boy_speed (x : ℝ) : 
  (x - second_boy_speed) * walking_time = final_distance → x = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_first_boy_speed_l748_74805


namespace NUMINAMATH_CALUDE_combined_average_score_l748_74883

/-- Given three classes with average scores and student ratios, prove the combined average score -/
theorem combined_average_score 
  (score_U score_B score_C : ℝ)
  (ratio_U ratio_B ratio_C : ℕ)
  (h1 : score_U = 65)
  (h2 : score_B = 80)
  (h3 : score_C = 77)
  (h4 : ratio_U = 4)
  (h5 : ratio_B = 6)
  (h6 : ratio_C = 5) :
  (score_U * ratio_U + score_B * ratio_B + score_C * ratio_C) / (ratio_U + ratio_B + ratio_C) = 75 := by
  sorry

end NUMINAMATH_CALUDE_combined_average_score_l748_74883


namespace NUMINAMATH_CALUDE_laura_park_time_percentage_l748_74812

theorem laura_park_time_percentage :
  let num_trips : ℕ := 6
  let park_time_per_trip : ℝ := 2
  let walking_time_per_trip : ℝ := 0.5
  let total_time_per_trip : ℝ := park_time_per_trip + walking_time_per_trip
  let total_time_all_trips : ℝ := total_time_per_trip * num_trips
  let total_park_time : ℝ := park_time_per_trip * num_trips
  total_park_time / total_time_all_trips = 0.8
  := by sorry

end NUMINAMATH_CALUDE_laura_park_time_percentage_l748_74812


namespace NUMINAMATH_CALUDE_sandy_younger_than_molly_l748_74891

theorem sandy_younger_than_molly (sandy_age molly_age : ℕ) : 
  sandy_age = 63 → 
  sandy_age * 9 = molly_age * 7 → 
  molly_age - sandy_age = 18 := by
sorry

end NUMINAMATH_CALUDE_sandy_younger_than_molly_l748_74891


namespace NUMINAMATH_CALUDE_units_digit_7_pow_million_l748_74881

def units_digit_cycle_7 : List Nat := [7, 9, 3, 1]

theorem units_digit_7_pow_million :
  ∃ (n : Nat), n < 10 ∧ (7^(10^6 : Nat)) % 10 = n ∧ n = 1 :=
by
  sorry

#check units_digit_7_pow_million

end NUMINAMATH_CALUDE_units_digit_7_pow_million_l748_74881


namespace NUMINAMATH_CALUDE_triangle_side_length_l748_74862

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Conditions
  a = Real.sqrt 3 →
  b = 1 →
  A = 2 * B →
  -- Triangle properties
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  -- Sine law
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  -- Question/Conclusion
  c = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l748_74862
