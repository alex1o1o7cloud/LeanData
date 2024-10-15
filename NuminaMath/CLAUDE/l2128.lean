import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_ratio_l2128_212807

/-- Given a quadratic polynomial of the form x^2 + 1800x + 2700,
    prove that when written as (x + b)^2 + c, the ratio c/b equals -897 -/
theorem quadratic_ratio (x : ℝ) :
  let f := fun x => x^2 + 1800*x + 2700
  ∃ b c : ℝ, (∀ x, f x = (x + b)^2 + c) ∧ c / b = -897 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l2128_212807


namespace NUMINAMATH_CALUDE_back_seat_capacity_is_nine_l2128_212809

/-- Represents the seating capacity of a bus -/
structure BusSeats where
  leftSeats : Nat
  rightSeats : Nat
  peoplePerSeat : Nat
  totalCapacity : Nat

/-- Calculates the number of people who can sit at the back seat of the bus -/
def backSeatCapacity (bus : BusSeats) : Nat :=
  bus.totalCapacity - (bus.leftSeats + bus.rightSeats) * bus.peoplePerSeat

/-- Theorem stating the back seat capacity of the given bus configuration -/
theorem back_seat_capacity_is_nine :
  let bus : BusSeats := {
    leftSeats := 15,
    rightSeats := 12,
    peoplePerSeat := 3,
    totalCapacity := 90
  }
  backSeatCapacity bus = 9 := by sorry

end NUMINAMATH_CALUDE_back_seat_capacity_is_nine_l2128_212809


namespace NUMINAMATH_CALUDE_min_value_of_a_l2128_212863

theorem min_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x / (x^2 + 3*x + 1) ≤ a) → a ≥ 1/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_l2128_212863


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_halves_l2128_212878

theorem one_thirds_in_nine_halves : (9 / 2) / (1 / 3) = 27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_halves_l2128_212878


namespace NUMINAMATH_CALUDE_numbers_with_five_in_range_l2128_212820

def count_numbers_with_five (n : ℕ) : ℕ :=
  n - (6 * 9 * 9)

theorem numbers_with_five_in_range :
  count_numbers_with_five 700 = 214 := by
  sorry

end NUMINAMATH_CALUDE_numbers_with_five_in_range_l2128_212820


namespace NUMINAMATH_CALUDE_playlist_composition_l2128_212866

theorem playlist_composition (initial_hip_hop_ratio : Real) 
  (country_percentage : Real) (hip_hop_percentage : Real) : 
  initial_hip_hop_ratio = 0.65 →
  country_percentage = 0.4 →
  hip_hop_percentage = (1 - country_percentage) * initial_hip_hop_ratio →
  hip_hop_percentage = 0.39 := by
  sorry

end NUMINAMATH_CALUDE_playlist_composition_l2128_212866


namespace NUMINAMATH_CALUDE_min_value_cos_squared_plus_sin_l2128_212855

theorem min_value_cos_squared_plus_sin (f : ℝ → ℝ) :
  (∀ x, -π/4 ≤ x ∧ x ≤ π/4 → f x = Real.cos x ^ 2 + Real.sin x) →
  ∃ x₀, -π/4 ≤ x₀ ∧ x₀ ≤ π/4 ∧ f x₀ = (1 - Real.sqrt 2) / 2 ∧
  ∀ x, -π/4 ≤ x ∧ x ≤ π/4 → f x₀ ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_min_value_cos_squared_plus_sin_l2128_212855


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2128_212837

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) : ℕ+ → ℤ := fun n => a₁ + d * (n - 1)

theorem arithmetic_sequence_properties
  (a : ℕ+ → ℤ)
  (h₁ : a 1 = -60)
  (h₂ : a 17 = -12) :
  let d := (a 17 - a 1) / 16
  ∃ (S T : ℕ+ → ℤ),
    (∀ n : ℕ+, a n = arithmetic_sequence (-60) d n) ∧
    (∀ n : ℕ+, n < 22 → a n ≤ 0) ∧
    (a 22 > 0) ∧
    (∀ n : ℕ+, S n = n * (a 1 + a n) / 2) ∧
    (S 20 = S 21) ∧
    (S 20 = -630) ∧
    (∀ n : ℕ+, n ≤ 21 → T n = n * (123 - 3 * n) / 2) ∧
    (∀ n : ℕ+, n ≥ 22 → T n = (3 * n^2 - 123 * n + 2520) / 2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2128_212837


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_condition_l2128_212851

theorem arithmetic_geometric_mean_inequality_condition (a b : ℝ) : 
  (a > 0 ∧ b > 0 → (a + b) / 2 ≥ Real.sqrt (a * b)) ∧ 
  ∃ a b : ℝ, ¬(a > 0 ∧ b > 0) ∧ (a + b) / 2 ≥ Real.sqrt (a * b) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_condition_l2128_212851


namespace NUMINAMATH_CALUDE_merry_sunday_boxes_l2128_212897

/-- Represents the number of apples in each box -/
def apples_per_box : ℕ := 10

/-- Represents the number of boxes Merry had on Saturday -/
def saturday_boxes : ℕ := 50

/-- Represents the total number of apples sold on Saturday and Sunday -/
def total_apples_sold : ℕ := 720

/-- Represents the number of boxes left after selling -/
def boxes_left : ℕ := 3

/-- Represents the number of boxes Merry had on Sunday -/
def sunday_boxes : ℕ := 25

theorem merry_sunday_boxes :
  sunday_boxes = 25 :=
by sorry

end NUMINAMATH_CALUDE_merry_sunday_boxes_l2128_212897


namespace NUMINAMATH_CALUDE_cube_surface_area_l2128_212808

/-- The surface area of a cube with edge length 3a is 54a² -/
theorem cube_surface_area (a : ℝ) : 
  6 * (3 * a)^2 = 54 * a^2 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2128_212808


namespace NUMINAMATH_CALUDE_odd_function_properties_l2128_212814

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define an increasing function on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Define the minimum value of a function on an interval
def HasMinValueOn (f : ℝ → ℝ) (v : ℝ) (a b : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → v ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = v)

-- Define the maximum value of a function on an interval
def HasMaxValueOn (f : ℝ → ℝ) (v : ℝ) (a b : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ v) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = v)

-- Theorem statement
theorem odd_function_properties (f : ℝ → ℝ) :
  OddFunction f →
  IncreasingOn f 1 3 →
  HasMinValueOn f 0 1 3 →
  IncreasingOn f (-3) (-1) ∧ HasMaxValueOn f 0 (-3) (-1) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l2128_212814


namespace NUMINAMATH_CALUDE_flag_design_count_l2128_212830

/-- Represents the number of available colors for the flag stripes -/
def num_colors : ℕ := 3

/-- Represents the number of stripes in the flag -/
def num_stripes : ℕ := 3

/-- Calculates the number of possible flag designs -/
def num_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem: The number of unique three-stripe flags that can be created
    using three colors, where adjacent stripes may be the same color, is 27 -/
theorem flag_design_count :
  num_flag_designs = 27 := by sorry

end NUMINAMATH_CALUDE_flag_design_count_l2128_212830


namespace NUMINAMATH_CALUDE_base12_addition_theorem_l2128_212828

-- Define a custom type for base-12 digits
inductive Base12Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B

-- Define a type for base-12 numbers
def Base12Number := List Base12Digit

-- Define the two numbers we're adding
def num1 : Base12Number := [Base12Digit.D5, Base12Digit.D2, Base12Digit.D8]
def num2 : Base12Number := [Base12Digit.D2, Base12Digit.D7, Base12Digit.D3]

-- Define the expected result
def expected_result : Base12Number := [Base12Digit.D7, Base12Digit.D9, Base12Digit.B]

-- Function to add two base-12 numbers
def add_base12 (a b : Base12Number) : Base12Number :=
  sorry

theorem base12_addition_theorem :
  add_base12 num1 num2 = expected_result :=
sorry

end NUMINAMATH_CALUDE_base12_addition_theorem_l2128_212828


namespace NUMINAMATH_CALUDE_integer_root_of_cubic_l2128_212865

theorem integer_root_of_cubic (a b c : ℚ) :
  let f : ℝ → ℝ := λ x => x^3 + a*x^2 + b*x + c
  (f (3 - Real.sqrt 5) = 0) →
  (∃ r : ℤ, f r = 0) →
  (∃ r : ℤ, f r = 0 ∧ r = -6) :=
by sorry

end NUMINAMATH_CALUDE_integer_root_of_cubic_l2128_212865


namespace NUMINAMATH_CALUDE_function_properties_l2128_212859

noncomputable def f (x : ℝ) := Real.cos (2 * x + 2 * Real.pi / 3) + 2 * (Real.cos x) ^ 2

theorem function_properties :
  (∀ x, f x ≤ 2) ∧
  (∀ k : ℤ, f (k * Real.pi - Real.pi / 6) = 2) ∧
  (∀ A B C a b c : ℝ,
    0 < A ∧ A < Real.pi →
    0 < B ∧ B < Real.pi →
    0 < C ∧ C < Real.pi →
    A + B + C = Real.pi →
    a > 0 ∧ b > 0 ∧ c > 0 →
    a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
    f A = 3 / 2 →
    b + c = 2 →
    a ≥ Real.sqrt 3) := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_function_properties_l2128_212859


namespace NUMINAMATH_CALUDE_f_values_l2128_212864

def f (x : ℝ) : ℝ := x^2 + x + 1

theorem f_values : f 2 = 7 ∧ f (f 1) = 13 := by sorry

end NUMINAMATH_CALUDE_f_values_l2128_212864


namespace NUMINAMATH_CALUDE_point_not_in_region_iff_m_in_interval_l2128_212800

/-- The function representing the left side of the inequality -/
def f (m x y : ℝ) : ℝ := x - (m^2 - 2*m + 4)*y - 6

/-- The theorem stating the equivalence between the point (-1, -1) not being in the region
    and m being in the interval [-1, 3] -/
theorem point_not_in_region_iff_m_in_interval :
  ∀ m : ℝ, f m (-1) (-1) ≤ 0 ↔ -1 ≤ m ∧ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_point_not_in_region_iff_m_in_interval_l2128_212800


namespace NUMINAMATH_CALUDE_sum_in_base_9_l2128_212803

/-- Converts a base-9 number to base-10 --/
def base9To10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * 9^i) 0

/-- Converts a base-10 number to base-9 --/
def base10To9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- The sum of 263₉, 504₉, and 72₉ in base 9 is 850₉ --/
theorem sum_in_base_9 :
  base10To9 (base9To10 [3, 6, 2] + base9To10 [4, 0, 5] + base9To10 [2, 7]) = [0, 5, 8] :=
by sorry

end NUMINAMATH_CALUDE_sum_in_base_9_l2128_212803


namespace NUMINAMATH_CALUDE_rectangular_box_dimension_sum_square_l2128_212810

/-- Given a rectangular box with dimensions a, b, c, where a = b + c + 10,
    prove that the square of the sum of dimensions is equal to 4(b+c)^2 + 40(b+c) + 100 -/
theorem rectangular_box_dimension_sum_square (b c : ℝ) :
  let a : ℝ := b + c + 10
  let D : ℝ := a + b + c
  D^2 = 4*(b+c)^2 + 40*(b+c) + 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_dimension_sum_square_l2128_212810


namespace NUMINAMATH_CALUDE_fourth_root_16_times_sixth_root_64_l2128_212829

theorem fourth_root_16_times_sixth_root_64 : (16 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/6) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_16_times_sixth_root_64_l2128_212829


namespace NUMINAMATH_CALUDE_shoe_savings_l2128_212854

theorem shoe_savings (max_budget : ℝ) (original_price : ℝ) (discount_percent : ℝ) 
  (h1 : max_budget = 130)
  (h2 : original_price = 120)
  (h3 : discount_percent = 30) : 
  max_budget - (original_price * (1 - discount_percent / 100)) = 46 := by
  sorry

end NUMINAMATH_CALUDE_shoe_savings_l2128_212854


namespace NUMINAMATH_CALUDE_max_total_points_l2128_212883

/-- Represents the types of buckets in the ring toss game -/
inductive Bucket
| Red
| Green
| Blue

/-- Represents the game state -/
structure GameState where
  money : ℕ
  points : ℕ
  rings_per_play : ℕ
  red_points : ℕ
  green_points : ℕ
  blue_points : ℕ
  blue_success_rate : ℚ
  red_buckets_hit : ℕ
  green_buckets_hit : ℕ
  blue_buckets_hit : ℕ

/-- Calculates the maximum points achievable in one play -/
def max_points_per_play (gs : GameState) : ℕ :=
  gs.rings_per_play * max gs.red_points (max gs.green_points gs.blue_points)

/-- Calculates the total points from already hit buckets -/
def current_points (gs : GameState) : ℕ :=
  gs.red_buckets_hit * gs.red_points +
  gs.green_buckets_hit * gs.green_points +
  gs.blue_buckets_hit * gs.blue_points

/-- Theorem: The maximum total points Tiffany can achieve in three games is 43 -/
theorem max_total_points (gs : GameState)
  (h1 : gs.money = 3)
  (h2 : gs.rings_per_play = 5)
  (h3 : gs.red_points = 2)
  (h4 : gs.green_points = 3)
  (h5 : gs.blue_points = 5)
  (h6 : gs.blue_success_rate = 1/10)
  (h7 : gs.red_buckets_hit = 4)
  (h8 : gs.green_buckets_hit = 5)
  (h9 : gs.blue_buckets_hit = 1) :
  current_points gs + max_points_per_play gs = 43 :=
by sorry

end NUMINAMATH_CALUDE_max_total_points_l2128_212883


namespace NUMINAMATH_CALUDE_stock_market_value_l2128_212874

/-- Given a stock with a 10% dividend rate and an 8% yield, its market value is $125. -/
theorem stock_market_value (face_value : ℝ) (dividend_rate : ℝ) (yield : ℝ) : 
  dividend_rate = 0.1 → yield = 0.08 → (dividend_rate * face_value) / yield = 125 := by
  sorry

end NUMINAMATH_CALUDE_stock_market_value_l2128_212874


namespace NUMINAMATH_CALUDE_carrots_equal_fifteen_l2128_212899

/-- The price relationship between apples, bananas, and carrots -/
structure FruitPrices where
  apple_banana_ratio : ℚ
  banana_carrot_ratio : ℚ
  apple_banana_eq : apple_banana_ratio = 10 / 5
  banana_carrot_eq : banana_carrot_ratio = 2 / 5

/-- The number of carrots that can be bought for the price of 12 apples -/
def carrots_for_apples (prices : FruitPrices) : ℚ :=
  12 * (prices.banana_carrot_ratio / prices.apple_banana_ratio)

theorem carrots_equal_fifteen (prices : FruitPrices) :
  carrots_for_apples prices = 15 := by
  sorry

end NUMINAMATH_CALUDE_carrots_equal_fifteen_l2128_212899


namespace NUMINAMATH_CALUDE_domino_coloring_properties_l2128_212845

/-- Definition of the number of possible colorings for a domino of length n -/
def A (n : ℕ) : ℕ := 2^n

/-- Definition of the number of valid colorings (no adjacent painted squares) for a domino of length n -/
def F : ℕ → ℕ
  | 0 => 1  -- Base case for convenience
  | 1 => 2
  | 2 => 3
  | (n+3) => F (n+2) + F (n+1)

theorem domino_coloring_properties :
  (∀ n : ℕ, A n = 2^n) ∧
  F 1 = 2 ∧ F 2 = 3 ∧ F 3 = 5 ∧ F 4 = 8 ∧
  (∀ n : ℕ, n ≥ 3 → F n = F (n-1) + F (n-2)) ∧
  (∀ n p : ℕ+, F (n + p + 1) = F n * F p + F (n-1) * F (p-1)) := by
  sorry

end NUMINAMATH_CALUDE_domino_coloring_properties_l2128_212845


namespace NUMINAMATH_CALUDE_probability_no_dessert_l2128_212802

def probability_dessert : ℝ := 0.60
def probability_dessert_no_coffee : ℝ := 0.20

theorem probability_no_dessert :
  1 - probability_dessert = 0.40 :=
sorry

end NUMINAMATH_CALUDE_probability_no_dessert_l2128_212802


namespace NUMINAMATH_CALUDE_value_of_b_l2128_212892

theorem value_of_b (a b c : ℝ) 
  (h1 : a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1))
  (h2 : 6 * b * 11 = 1) : 
  b = 15 := by
sorry

end NUMINAMATH_CALUDE_value_of_b_l2128_212892


namespace NUMINAMATH_CALUDE_circle_intersection_properties_l2128_212835

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the intersection points
def intersectionPoints (A B : ℝ × ℝ) : Prop :=
  C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧ A ≠ B

-- Theorem statement
theorem circle_intersection_properties
  (A B : ℝ × ℝ) (h : intersectionPoints A B) :
  -- 1. Equation of the line containing chord AB
  (∀ x y : ℝ, (x - y - 3 = 0) ↔ (∃ t : ℝ, x = A.1 + t * (B.1 - A.1) ∧ y = A.2 + t * (B.2 - A.2))) ∧
  -- 2. Length of the common chord AB
  Real.sqrt 2 = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ∧
  -- 3. Equation of the perpendicular bisector of AB
  (∀ x y : ℝ, (x + y = 0) ↔ ((x - (A.1 + B.1) / 2)^2 + (y - (A.2 + B.2) / 2)^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4)) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_properties_l2128_212835


namespace NUMINAMATH_CALUDE_f_value_at_pi_24_max_monotone_interval_exists_max_monotone_interval_l2128_212861

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2

theorem f_value_at_pi_24 : f (Real.pi / 24) = Real.sqrt 2 + 1 := by sorry

theorem max_monotone_interval : 
  ∀ m : ℝ, (∀ x y : ℝ, -m ≤ x ∧ x < y ∧ y ≤ m → f x < f y) → m ≤ Real.pi / 6 := by sorry

theorem exists_max_monotone_interval : 
  ∃ m : ℝ, m = Real.pi / 6 ∧ 
    (∀ x y : ℝ, -m ≤ x ∧ x < y ∧ y ≤ m → f x < f y) ∧
    (∀ m' : ℝ, m' > Real.pi / 6 → ¬(∀ x y : ℝ, -m' ≤ x ∧ x < y ∧ y ≤ m' → f x < f y)) := by sorry

end NUMINAMATH_CALUDE_f_value_at_pi_24_max_monotone_interval_exists_max_monotone_interval_l2128_212861


namespace NUMINAMATH_CALUDE_pants_gross_profit_l2128_212826

/-- Calculates the gross profit for a store selling pants -/
theorem pants_gross_profit (purchase_price : ℝ) (markup_percent : ℝ) (price_decrease : ℝ) :
  purchase_price = 210 ∧ 
  markup_percent = 0.25 ∧ 
  price_decrease = 0.20 →
  let original_price := purchase_price / (1 - markup_percent)
  let new_price := original_price * (1 - price_decrease)
  new_price - purchase_price = 14 := by
  sorry

end NUMINAMATH_CALUDE_pants_gross_profit_l2128_212826


namespace NUMINAMATH_CALUDE_mary_took_three_crayons_l2128_212876

/-- Given an initial number of crayons and the number left after some are taken,
    calculate the number of crayons taken. -/
def crayons_taken (initial : ℕ) (left : ℕ) : ℕ := initial - left

theorem mary_took_three_crayons :
  let initial_crayons : ℕ := 7
  let crayons_left : ℕ := 4
  crayons_taken initial_crayons crayons_left = 3 := by
sorry

end NUMINAMATH_CALUDE_mary_took_three_crayons_l2128_212876


namespace NUMINAMATH_CALUDE_sphere_to_hemisphere_volume_ratio_l2128_212856

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_to_hemisphere_volume_ratio (p : ℝ) (p_pos : p > 0) :
  (4 / 3 * Real.pi * p ^ 3) / (1 / 2 * 4 / 3 * Real.pi * (3 * p) ^ 3) = 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_sphere_to_hemisphere_volume_ratio_l2128_212856


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l2128_212873

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l2128_212873


namespace NUMINAMATH_CALUDE_complex_coordinate_l2128_212823

/-- Given zi = 2-i, prove that z = -1 - 2i -/
theorem complex_coordinate (z : ℂ) : z * Complex.I = 2 - Complex.I → z = -1 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_coordinate_l2128_212823


namespace NUMINAMATH_CALUDE_sum_independence_and_value_l2128_212843

theorem sum_independence_and_value (a : ℝ) (h : a ≥ -3/4) :
  let s := (((a + 1) / 2 + (a + 3) / 6 * Real.sqrt ((4 * a + 3) / 3)) ^ (1/3 : ℝ) : ℝ)
  let t := (((a + 1) / 2 - (a + 3) / 6 * Real.sqrt ((4 * a + 3) / 3)) ^ (1/3 : ℝ) : ℝ)
  s + t = 1 := by sorry

end NUMINAMATH_CALUDE_sum_independence_and_value_l2128_212843


namespace NUMINAMATH_CALUDE_room_area_square_inches_l2128_212872

-- Define the number of inches in a foot
def inches_per_foot : ℕ := 12

-- Define the length of the room in feet
def room_length_feet : ℕ := 10

-- Theorem: The area of the room in square inches is 14400
theorem room_area_square_inches :
  (room_length_feet * inches_per_foot) ^ 2 = 14400 := by
  sorry

end NUMINAMATH_CALUDE_room_area_square_inches_l2128_212872


namespace NUMINAMATH_CALUDE_gcd_459_357_l2128_212806

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l2128_212806


namespace NUMINAMATH_CALUDE_vincent_book_expenditure_l2128_212898

/-- The total amount Vincent spent on books -/
def total_spent (animal_books train_books space_books book_price : ℕ) : ℕ :=
  (animal_books + train_books + space_books) * book_price

/-- Theorem stating that Vincent spent $224 on books -/
theorem vincent_book_expenditure :
  total_spent 10 3 1 16 = 224 := by
  sorry

end NUMINAMATH_CALUDE_vincent_book_expenditure_l2128_212898


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2128_212839

theorem simplify_and_evaluate : 
  let a : ℚ := 1/2
  let b : ℚ := 1/3
  5 * (3 * a^2 * b - a * b^2) - (a * b^2 + 3 * a^2 * b) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2128_212839


namespace NUMINAMATH_CALUDE_calculation_proof_l2128_212894

theorem calculation_proof :
  ((-1 : ℝ)^2023 + |(-3 : ℝ)| - (π - 7)^0 + 2^4 * (1/2 : ℝ)^4 = 2) ∧
  (∀ (a b : ℝ), 6*a^3*b^2 / (3*a^2*b^2) + (2*a*b^3)^2 / (a*b)^2 = 2*a + 4*b^4) :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l2128_212894


namespace NUMINAMATH_CALUDE_recurrence_initial_values_l2128_212889

/-- A sequence of real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (x : ℤ → ℝ) : Prop :=
  ∀ n : ℤ, x (n + 1) = (x n ^ 2 + 10) / 7

/-- The property of being bounded above -/
def BoundedAbove (x : ℤ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ n : ℤ, x n ≤ M

/-- The set of possible initial values for bounded sequences satisfying the recurrence -/
def PossibleInitialValues : Set ℝ :=
  {x₀ : ℝ | ∃ x : ℤ → ℝ, RecurrenceSequence x ∧ BoundedAbove x ∧ x 0 = x₀}

theorem recurrence_initial_values :
    PossibleInitialValues = Set.Icc 2 5 := by
  sorry

end NUMINAMATH_CALUDE_recurrence_initial_values_l2128_212889


namespace NUMINAMATH_CALUDE_age_ratio_problem_l2128_212879

theorem age_ratio_problem (a b : ℕ) 
  (h1 : a = 2 * b)  -- Present age ratio 6:3 simplifies to 2:1
  (h2 : a - 4 = b + 4)  -- A's age 4 years ago equals B's age 4 years hence
  : (a + 4) / (b - 4) = 5 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l2128_212879


namespace NUMINAMATH_CALUDE_iron_nickel_percentage_l2128_212825

/-- Represents the exchange of quarters for nickels, including special iron nickels --/
def nickel_exchange (num_quarters : ℕ) (total_value : ℚ) (iron_nickel_value : ℚ) : Prop :=
  ∃ (num_iron_nickels : ℕ),
    let num_nickels : ℕ := num_quarters * 5
    let regular_nickel_value : ℚ := 1/20
    (num_iron_nickels : ℚ) * iron_nickel_value + 
    ((num_nickels - num_iron_nickels) : ℚ) * regular_nickel_value = total_value ∧
    (num_iron_nickels : ℚ) / (num_nickels : ℚ) = 1/5

theorem iron_nickel_percentage 
  (h : nickel_exchange 20 64 3) : 
  ∃ (num_iron_nickels : ℕ), 
    (num_iron_nickels : ℚ) / 100 = 1/5 :=
by
  sorry

#check iron_nickel_percentage

end NUMINAMATH_CALUDE_iron_nickel_percentage_l2128_212825


namespace NUMINAMATH_CALUDE_arcsin_one_half_l2128_212846

theorem arcsin_one_half : Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_l2128_212846


namespace NUMINAMATH_CALUDE_coordinates_sum_of_point_b_l2128_212886

/-- Given two points A and B, where A is at the origin and B is on the line y=5,
    and the slope of segment AB is 3/4, prove that the sum of the x- and y-coordinates of B is 35/3 -/
theorem coordinates_sum_of_point_b (A B : ℝ × ℝ) : 
  A = (0, 0) →
  B.2 = 5 →
  (B.2 - A.2) / (B.1 - A.1) = 3 / 4 →
  B.1 + B.2 = 35 / 3 := by
  sorry

end NUMINAMATH_CALUDE_coordinates_sum_of_point_b_l2128_212886


namespace NUMINAMATH_CALUDE_necklace_cuts_l2128_212813

theorem necklace_cuts (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 8) :
  Nat.choose n k = 145422675 :=
by sorry

end NUMINAMATH_CALUDE_necklace_cuts_l2128_212813


namespace NUMINAMATH_CALUDE_equal_digits_probability_l2128_212804

/-- The number of sides on each die -/
def num_sides : ℕ := 20

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The number of one-digit outcomes on a die -/
def one_digit_outcomes : ℕ := 9

/-- The number of two-digit outcomes on a die -/
def two_digit_outcomes : ℕ := 11

/-- The probability of rolling an equal number of one-digit and two-digit numbers on 6 20-sided dice -/
def equal_digits_prob : ℚ := 4851495 / 16000000

theorem equal_digits_probability : 
  let p_one_digit := one_digit_outcomes / num_sides
  let p_two_digit := two_digit_outcomes / num_sides
  let combinations := Nat.choose num_dice (num_dice / 2)
  combinations * (p_one_digit ^ (num_dice / 2)) * (p_two_digit ^ (num_dice / 2)) = equal_digits_prob := by
  sorry

end NUMINAMATH_CALUDE_equal_digits_probability_l2128_212804


namespace NUMINAMATH_CALUDE_max_area_region_S_l2128_212801

/-- A circle in a plane -/
structure Circle where
  radius : ℝ
  center : ℝ × ℝ

/-- The line to which the circles are tangent -/
def TangentLine : Set (ℝ × ℝ) := sorry

/-- The point at which the circles are tangent to the line -/
def TangentPoint : ℝ × ℝ := sorry

/-- The set of four circles with radii 1, 3, 5, and 7 -/
def FourCircles : Set Circle := sorry

/-- The region S composed of all points that lie within one of the four circles -/
def RegionS (circles : Set Circle) : Set (ℝ × ℝ) := sorry

/-- The area of a set in the plane -/
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the maximum area of region S is 65π -/
theorem max_area_region_S :
  ∃ (c : Set Circle), c = FourCircles ∧
    (∀ circle ∈ c, ∃ p ∈ TangentLine, p = TangentPoint ∧
      (circle.center.1 - p.1)^2 + (circle.center.2 - p.2)^2 = circle.radius^2) ∧
    (∀ arrangement : Set Circle, arrangement = FourCircles →
      area (RegionS arrangement) ≤ area (RegionS c)) ∧
    area (RegionS c) = 65 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_max_area_region_S_l2128_212801


namespace NUMINAMATH_CALUDE_customers_who_tipped_l2128_212891

/-- The number of customers who left a tip at 'The Gourmet Kitchen' restaurant --/
theorem customers_who_tipped (total_customers early_morning_customers priority_customers regular_evening_customers : ℕ)
  (h_total : total_customers = 215)
  (h_early : early_morning_customers = 20)
  (h_priority : priority_customers = 60)
  (h_regular : regular_evening_customers = 22)
  (h_early_no_tip : ⌊early_morning_customers * (30 : ℚ) / 100⌋ = 6)
  (h_priority_no_tip : ⌊priority_customers * (60 : ℚ) / 100⌋ = 36)
  (h_regular_no_tip : ⌊regular_evening_customers * (50 : ℚ) / 100⌋ = 11)
  (h_remaining : total_customers - early_morning_customers - priority_customers - regular_evening_customers = 113)
  (h_remaining_no_tip : ⌊113 * (25 : ℚ) / 100⌋ = 28) :
  total_customers - (6 + 36 + 11 + 28) = 134 := by
  sorry

#check customers_who_tipped

end NUMINAMATH_CALUDE_customers_who_tipped_l2128_212891


namespace NUMINAMATH_CALUDE_parabola_zeros_difference_l2128_212818

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_zeros_difference (p : Parabola) :
  p.y_coord 3 = -9 →   -- vertex condition
  p.y_coord 6 = 27 →   -- point condition
  ∃ (x1 x2 : ℝ), 
    p.y_coord x1 = 0 ∧ 
    p.y_coord x2 = 0 ∧ 
    |x1 - x2| = 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_zeros_difference_l2128_212818


namespace NUMINAMATH_CALUDE_first_sales_amount_l2128_212890

/-- The amount of the first sales in millions of dollars -/
def first_sales : ℝ := sorry

/-- The profit on the first sales in millions of dollars -/
def first_profit : ℝ := 5

/-- The profit on the next $30 million in sales in millions of dollars -/
def second_profit : ℝ := 12

/-- The amount of the second sales in millions of dollars -/
def second_sales : ℝ := 30

/-- The increase in profit ratio from the first to the second sales -/
def profit_ratio_increase : ℝ := 0.2000000000000001

theorem first_sales_amount :
  (first_profit / first_sales) * (1 + profit_ratio_increase) = second_profit / second_sales ∧
  first_sales = 15 := by
  sorry

end NUMINAMATH_CALUDE_first_sales_amount_l2128_212890


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l2128_212881

/-- The trajectory of point Q given the conditions of the problem -/
theorem trajectory_of_Q (A P Q : ℝ × ℝ) : 
  A = (4, 0) →
  (P.1^2 + P.2^2 = 4) →
  (Q.1 - A.1, Q.2 - A.2) = (2*(P.1 - Q.1), 2*(P.2 - Q.2)) →
  (Q.1 - 4/3)^2 + Q.2^2 = 16/9 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l2128_212881


namespace NUMINAMATH_CALUDE_seahawks_touchdowns_l2128_212812

theorem seahawks_touchdowns (final_score : ℕ) (field_goals : ℕ) 
  (h1 : final_score = 37)
  (h2 : field_goals = 3) :
  (final_score - field_goals * 3) / 7 = 4 := by
  sorry

#check seahawks_touchdowns

end NUMINAMATH_CALUDE_seahawks_touchdowns_l2128_212812


namespace NUMINAMATH_CALUDE_xy_equals_five_l2128_212869

theorem xy_equals_five (x y : ℝ) (h : x * (x + 2*y) = x^2 + 10) : x * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_five_l2128_212869


namespace NUMINAMATH_CALUDE_total_cost_for_two_rides_l2128_212860

def base_fare : ℚ := 2
def per_mile_charge : ℚ := (3 : ℚ) / 10
def first_ride_distance : ℕ := 8
def second_ride_distance : ℕ := 5

def ride_cost (distance : ℕ) : ℚ :=
  base_fare + per_mile_charge * distance

theorem total_cost_for_two_rides :
  ride_cost first_ride_distance + ride_cost second_ride_distance = (79 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_for_two_rides_l2128_212860


namespace NUMINAMATH_CALUDE_functional_equation_problem_l2128_212862

/-- The functional equation problem -/
theorem functional_equation_problem (α : ℝ) (hα : α ≠ 0) :
  (∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x^2 + y + f y) = (f x)^2 + α * y) ↔ α = 2 :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_problem_l2128_212862


namespace NUMINAMATH_CALUDE_eg_fh_ratio_l2128_212824

/-- Given points E, F, G, and H on a line in that order, prove that EG:FH = 10:17 -/
theorem eg_fh_ratio (E F G H : ℝ) (h_order : E ≤ F ∧ F ≤ G ∧ G ≤ H) 
  (h_ef : F - E = 3) (h_fg : G - F = 7) (h_eh : H - E = 20) :
  (G - E) / (H - F) = 10 / 17 := by
  sorry

end NUMINAMATH_CALUDE_eg_fh_ratio_l2128_212824


namespace NUMINAMATH_CALUDE_auction_bid_ratio_l2128_212880

/-- Auction bidding problem --/
theorem auction_bid_ratio :
  let start_price : ℕ := 300
  let harry_first_bid : ℕ := start_price + 200
  let second_bid : ℕ := 2 * harry_first_bid
  let harry_final_bid : ℕ := 4000
  let third_bid : ℕ := harry_final_bid - 1500
  (third_bid : ℚ) / harry_first_bid = 5 := by sorry

end NUMINAMATH_CALUDE_auction_bid_ratio_l2128_212880


namespace NUMINAMATH_CALUDE_reciprocal_complement_sum_square_l2128_212836

theorem reciprocal_complement_sum_square (p q r : ℝ) (h : p * q * r = 1) :
  (1 / (1 - p))^2 + (1 / (1 - q))^2 + (1 / (1 - r))^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_complement_sum_square_l2128_212836


namespace NUMINAMATH_CALUDE_cos_pi_plus_2alpha_l2128_212815

theorem cos_pi_plus_2alpha (α : Real) 
  (h : Real.sin (Real.pi / 2 - α) = 1 / 3) : 
  Real.cos (Real.pi + 2 * α) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_plus_2alpha_l2128_212815


namespace NUMINAMATH_CALUDE_quadrupled_container_volume_l2128_212896

/-- A container with an initial volume and a scale factor for its dimensions. -/
structure Container :=
  (initial_volume : ℝ)
  (scale_factor : ℝ)

/-- The new volume of a container after scaling its dimensions. -/
def new_volume (c : Container) : ℝ :=
  c.initial_volume * c.scale_factor^3

/-- Theorem stating that a container with 5 gallons initial volume and dimensions quadrupled results in 320 gallons. -/
theorem quadrupled_container_volume :
  let c := Container.mk 5 4
  new_volume c = 320 := by
  sorry

end NUMINAMATH_CALUDE_quadrupled_container_volume_l2128_212896


namespace NUMINAMATH_CALUDE_crayon_difference_is_1040_l2128_212884

/-- The number of crayons Willy and Lucy have combined, minus the number of crayons Max has -/
def crayon_difference (willy_crayons lucy_crayons max_crayons : ℕ) : ℕ :=
  (willy_crayons + lucy_crayons) - max_crayons

/-- Theorem stating that the difference in crayons is 1040 -/
theorem crayon_difference_is_1040 :
  crayon_difference 1400 290 650 = 1040 := by
  sorry

end NUMINAMATH_CALUDE_crayon_difference_is_1040_l2128_212884


namespace NUMINAMATH_CALUDE_ball_placement_count_is_144_l2128_212849

/-- The number of ways to place four different balls into four numbered boxes with exactly one box remaining empty -/
def ballPlacementCount : ℕ := 144

/-- The number of different balls -/
def numBalls : ℕ := 4

/-- The number of boxes -/
def numBoxes : ℕ := 4

/-- Theorem stating that the number of ways to place four different balls into four numbered boxes with exactly one box remaining empty is 144 -/
theorem ball_placement_count_is_144 : ballPlacementCount = 144 := by sorry

end NUMINAMATH_CALUDE_ball_placement_count_is_144_l2128_212849


namespace NUMINAMATH_CALUDE_cost_per_tire_to_produce_l2128_212888

/-- Proves that the cost per tire to produce is $8 given the specified conditions --/
theorem cost_per_tire_to_produce
  (fixed_cost : ℝ)
  (selling_price : ℝ)
  (batch_size : ℝ)
  (profit_per_tire : ℝ)
  (h1 : fixed_cost = 22500)
  (h2 : selling_price = 20)
  (h3 : batch_size = 15000)
  (h4 : profit_per_tire = 10.5) :
  ∃ (cost_per_tire : ℝ),
    cost_per_tire = 8 ∧
    batch_size * (selling_price - cost_per_tire) - fixed_cost = batch_size * profit_per_tire :=
by sorry

end NUMINAMATH_CALUDE_cost_per_tire_to_produce_l2128_212888


namespace NUMINAMATH_CALUDE_derivative_at_three_l2128_212827

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem derivative_at_three : 
  (deriv f) 3 = 6 := by sorry

end NUMINAMATH_CALUDE_derivative_at_three_l2128_212827


namespace NUMINAMATH_CALUDE_circle_center_l2128_212847

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 - 8*y - 16 = 0

-- Define the center of a circle
def is_center (h k : ℝ) (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ r, ∀ x y, eq x y ↔ (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem circle_center :
  is_center 3 4 circle_equation :=
sorry

end NUMINAMATH_CALUDE_circle_center_l2128_212847


namespace NUMINAMATH_CALUDE_stockholm_uppsala_distance_l2128_212850

/-- Represents the distance between two cities on a map and in reality. -/
structure MapDistance where
  /-- Distance on the map in centimeters -/
  map_distance : ℝ
  /-- Map scale: 1 cm on map represents this many km in reality -/
  scale : ℝ

/-- Calculates the real-world distance in meters given a MapDistance -/
def real_distance (d : MapDistance) : ℝ :=
  d.map_distance * d.scale * 1000

/-- The distance between Stockholm and Uppsala -/
def stockholm_uppsala : MapDistance :=
  { map_distance := 55
  , scale := 30 }

/-- Theorem stating that the distance between Stockholm and Uppsala is 1650000 meters -/
theorem stockholm_uppsala_distance :
  real_distance stockholm_uppsala = 1650000 := by
  sorry

end NUMINAMATH_CALUDE_stockholm_uppsala_distance_l2128_212850


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l2128_212833

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 4*y - 16 = -y^2 + 24*x + 16

-- Define the center and radius of the circle
def is_center_radius (a b r : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_center_radius_sum :
  ∃ a b r : ℝ, is_center_radius a b r ∧ a + b + r = 10 + 2 * Real.sqrt 41 :=
sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l2128_212833


namespace NUMINAMATH_CALUDE_circle_radius_l2128_212885

theorem circle_radius (M N : ℝ) (h1 : M > 0) (h2 : N > 0) (h3 : M / N = 40) :
  ∃ (r : ℝ), r > 0 ∧ M = π * r^2 ∧ N = 2 * π * r ∧ r = 80 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l2128_212885


namespace NUMINAMATH_CALUDE_calculate_expressions_l2128_212867

theorem calculate_expressions : 
  (1 - 1^2 + (64 : ℝ)^(1/3) - (-2) * (9 : ℝ)^(1/2) = 9) ∧ 
  ((-1/2 : ℝ) * (-2)^2 - (-1/8 : ℝ)^(1/3) + ((-1/2 : ℝ)^2)^(1/2) = -1) := by
  sorry

end NUMINAMATH_CALUDE_calculate_expressions_l2128_212867


namespace NUMINAMATH_CALUDE_weight_of_new_person_l2128_212816

theorem weight_of_new_person (initial_count : ℕ) (avg_increase : ℝ) (old_weight : ℝ) :
  initial_count = 15 →
  avg_increase = 2.3 →
  old_weight = 80 →
  let total_increase := initial_count * avg_increase
  let new_weight := old_weight + total_increase
  new_weight = 114.5 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l2128_212816


namespace NUMINAMATH_CALUDE_kendall_driving_distance_l2128_212852

/-- The distance Kendall drove with her mother -/
def distance_with_mother : ℝ := 0.67 - 0.5

/-- The total distance Kendall drove -/
def total_distance : ℝ := 0.67

/-- The distance Kendall drove with her father -/
def distance_with_father : ℝ := 0.5

theorem kendall_driving_distance :
  distance_with_mother = 0.17 :=
by sorry

end NUMINAMATH_CALUDE_kendall_driving_distance_l2128_212852


namespace NUMINAMATH_CALUDE_polygon_sides_l2128_212838

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 + 360 = 900 → n = 5 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l2128_212838


namespace NUMINAMATH_CALUDE_jared_popcorn_order_l2128_212832

/-- Calculate the number of servings of popcorn needed for a group -/
def popcorn_servings (pieces_per_serving : ℕ) (jared_pieces : ℕ) (friend_count : ℕ) (friend_pieces : ℕ) : ℕ :=
  (jared_pieces + friend_count * friend_pieces) / pieces_per_serving

theorem jared_popcorn_order :
  popcorn_servings 30 90 3 60 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jared_popcorn_order_l2128_212832


namespace NUMINAMATH_CALUDE_smallest_b_for_five_not_in_range_l2128_212834

theorem smallest_b_for_five_not_in_range :
  ∃ (b : ℤ), (∀ x : ℝ, x^2 + b*x + 10 ≠ 5) ∧
             (∀ c : ℤ, c < b → ∃ x : ℝ, x^2 + c*x + 10 = 5) ∧
             b = -4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_five_not_in_range_l2128_212834


namespace NUMINAMATH_CALUDE_sample_mean_inequality_l2128_212819

theorem sample_mean_inequality (n m : ℕ) (x_bar y_bar z_bar α : ℝ) :
  x_bar ≠ y_bar →
  0 < α →
  α < 1 / 2 →
  z_bar = α * x_bar + (1 - α) * y_bar →
  z_bar = (n * x_bar + m * y_bar) / (n + m) →
  n < m :=
by sorry

end NUMINAMATH_CALUDE_sample_mean_inequality_l2128_212819


namespace NUMINAMATH_CALUDE_cubic_polynomial_q_value_l2128_212868

/-- A cubic polynomial Q(x) = x^3 + px^2 + qx + d -/
def cubicPolynomial (p q d : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + q*x + d

theorem cubic_polynomial_q_value 
  (p q d : ℝ) 
  (h1 : -(p/3) = q) -- mean of zeros equals product of zeros taken two at a time
  (h2 : q = 1 + p + q + d) -- product of zeros taken two at a time equals sum of coefficients
  (h3 : d = 7) -- y-intercept is 7
  : q = 8/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_q_value_l2128_212868


namespace NUMINAMATH_CALUDE_pool_swimmers_l2128_212857

theorem pool_swimmers (total : ℕ) (first_day : ℕ) (extra_second_day : ℕ) 
  (h1 : total = 246)
  (h2 : first_day = 79)
  (h3 : extra_second_day = 47) :
  ∃ (third_day : ℕ), 
    third_day = 60 ∧ 
    first_day + (third_day + extra_second_day) + third_day = total :=
by
  sorry

end NUMINAMATH_CALUDE_pool_swimmers_l2128_212857


namespace NUMINAMATH_CALUDE_rectangle_composition_l2128_212877

theorem rectangle_composition (total_width total_height : ℕ) 
  (h_width : total_width = 3322) (h_height : total_height = 2020) : ∃ (r s : ℕ),
  2 * r + s = total_height ∧ 2 * r + 3 * s = total_width ∧ s = 651 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_composition_l2128_212877


namespace NUMINAMATH_CALUDE_expression_evaluation_l2128_212882

theorem expression_evaluation : (3 * 4 * 5) * (1/3 + 1/4 + 1/5 - 1/6) = 37 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2128_212882


namespace NUMINAMATH_CALUDE_right_triangle_max_ratio_l2128_212870

theorem right_triangle_max_ratio :
  ∀ (x y z : ℝ), 
    x > 0 → y > 0 → z > 0 →
    x^2 + y^2 = z^2 →
    (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a^2 + b^2 = c^2 → (a + 2*b) / c ≤ (x + 2*y) / z) →
    (x + 2*y) / z = 3 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_max_ratio_l2128_212870


namespace NUMINAMATH_CALUDE_binomial_22_12_l2128_212821

theorem binomial_22_12 (h1 : Nat.choose 20 10 = 184756)
                        (h2 : Nat.choose 20 11 = 167960)
                        (h3 : Nat.choose 20 12 = 125970) :
  Nat.choose 22 12 = 646646 := by
  sorry

end NUMINAMATH_CALUDE_binomial_22_12_l2128_212821


namespace NUMINAMATH_CALUDE_factorization_equality_l2128_212842

theorem factorization_equality (a b : ℝ) : a * b^2 - 2 * a * b + a = a * (b - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2128_212842


namespace NUMINAMATH_CALUDE_cos_minus_sin_identity_l2128_212858

theorem cos_minus_sin_identity (θ : Real) (a b : Real) 
  (h_acute : 0 < θ ∧ θ < π / 2)
  (h_sin : Real.sin (2 * θ) = a)
  (h_cos : Real.cos (2 * θ) = b) :
  Real.cos θ - Real.sin θ = Real.sqrt (1 - a) :=
sorry

end NUMINAMATH_CALUDE_cos_minus_sin_identity_l2128_212858


namespace NUMINAMATH_CALUDE_max_value_of_product_sum_l2128_212805

theorem max_value_of_product_sum (w x y z : ℝ) 
  (nonneg_w : w ≥ 0) (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_eq_200 : w + x + y + z = 200) : 
  wx + xy + yz ≤ 10000 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_product_sum_l2128_212805


namespace NUMINAMATH_CALUDE_collaborative_work_theorem_work_result_l2128_212822

/-- Represents the time taken to complete a task -/
structure TaskTime where
  days : ℝ
  hours_per_day : ℝ := 24
  total_hours : ℝ := days * hours_per_day

/-- Represents a worker's productivity -/
structure Worker where
  name : String
  task_time : TaskTime

/-- Represents a collaborative work scenario -/
structure CollaborativeWork where
  john : Worker
  jane : Worker
  total_time : TaskTime
  jane_indisposed_time : ℝ

/-- The main theorem to prove -/
theorem collaborative_work_theorem (work : CollaborativeWork) : 
  work.john.task_time.days = 18 →
  work.jane.task_time.days = 12 →
  work.total_time.days = 10.8 →
  work.jane_indisposed_time = 6 := by
  sorry

/-- An instance of the collaborative work scenario -/
def work_scenario : CollaborativeWork := {
  john := { name := "John", task_time := { days := 18 } }
  jane := { name := "Jane", task_time := { days := 12 } }
  total_time := { days := 10.8 }
  jane_indisposed_time := 6
}

/-- The main result -/
theorem work_result : work_scenario.jane_indisposed_time = 6 := by
  apply collaborative_work_theorem work_scenario
  · rfl
  · rfl
  · rfl

end NUMINAMATH_CALUDE_collaborative_work_theorem_work_result_l2128_212822


namespace NUMINAMATH_CALUDE_difference_nonnegative_equivalence_l2128_212831

theorem difference_nonnegative_equivalence (x : ℝ) :
  (x - 8 ≥ 0) ↔ (∃ (y : ℝ), y ≥ 0 ∧ x - 8 = y) :=
by sorry

end NUMINAMATH_CALUDE_difference_nonnegative_equivalence_l2128_212831


namespace NUMINAMATH_CALUDE_second_number_proof_l2128_212853

theorem second_number_proof (x : ℕ) : 
  (1255 % 29 = 8) → (x % 29 = 11) → x = 1287 := by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l2128_212853


namespace NUMINAMATH_CALUDE_unique_value_of_2n_plus_m_l2128_212840

theorem unique_value_of_2n_plus_m (n m : ℤ) 
  (h1 : 3*n - m < 5)
  (h2 : n + m > 26)
  (h3 : 3*m - 2*n < 46) :
  2*n + m = 36 := by
  sorry

end NUMINAMATH_CALUDE_unique_value_of_2n_plus_m_l2128_212840


namespace NUMINAMATH_CALUDE_evaluate_expression_l2128_212893

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 4) : y * (y - 2 * x + 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2128_212893


namespace NUMINAMATH_CALUDE_fraction_sum_positive_l2128_212841

theorem fraction_sum_positive (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  (1 / (a - b) + 1 / (b - c) + 1 / (c - a)) > 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_positive_l2128_212841


namespace NUMINAMATH_CALUDE_rain_probability_l2128_212811

/-- Probability of rain on at least one of two days -/
theorem rain_probability (p1 p2 p2_given_r1 : ℝ) 
  (h1 : p1 = 0.3) 
  (h2 : p2 = 0.4) 
  (h3 : p2_given_r1 = 0.7) 
  (h4 : 0 ≤ p1 ∧ p1 ≤ 1)
  (h5 : 0 ≤ p2 ∧ p2 ≤ 1)
  (h6 : 0 ≤ p2_given_r1 ∧ p2_given_r1 ≤ 1) : 
  1 - ((1 - p1) * (1 - p2) + p1 * (1 - p2_given_r1)) = 0.49 := by
sorry

end NUMINAMATH_CALUDE_rain_probability_l2128_212811


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2128_212817

theorem right_triangle_hypotenuse (base height hypotenuse : ℝ) : 
  base = 8 →
  (1/2) * base * height = 24 →
  base^2 + height^2 = hypotenuse^2 →
  hypotenuse = 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2128_212817


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l2128_212848

theorem triangle_sine_inequality (α β γ : Real) (h : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) :
  (Real.sin α + Real.sin β + Real.sin γ)^2 > 9 * Real.sin α * Real.sin β * Real.sin γ := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l2128_212848


namespace NUMINAMATH_CALUDE_consecutive_heads_probability_l2128_212895

/-- The number of coin flips -/
def n : ℕ := 12

/-- The number of desired heads -/
def k : ℕ := 9

/-- The probability of getting heads on a single flip of a fair coin -/
def p : ℚ := 1/2

/-- The number of ways to arrange k consecutive heads in n flips -/
def consecutive_arrangements : ℕ := n - k + 1

theorem consecutive_heads_probability :
  (consecutive_arrangements : ℚ) * p^k * (1-p)^(n-k) = 1/1024 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_heads_probability_l2128_212895


namespace NUMINAMATH_CALUDE_some_magical_creatures_are_mystical_beings_l2128_212875

-- Define our sets
variable (U : Type) -- Universe set
variable (D : Set U) -- Set of dragons
variable (M : Set U) -- Set of magical creatures
variable (B : Set U) -- Set of mystical beings

-- Define our premises
variable (h1 : D ⊆ M) -- All dragons are magical creatures
variable (h2 : ∃ x, x ∈ B ∩ D) -- Some mystical beings are dragons

-- Theorem to prove
theorem some_magical_creatures_are_mystical_beings : 
  ∃ x, x ∈ M ∩ B := by sorry

end NUMINAMATH_CALUDE_some_magical_creatures_are_mystical_beings_l2128_212875


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l2128_212844

/-- Two lines are parallel if and only if their slopes are equal -/
def are_parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

/-- The first line: (k-1)x + y + 2 = 0 -/
def line1 (k x y : ℝ) : Prop :=
  (k - 1) * x + y + 2 = 0

/-- The second line: 8x + (k+1)y + k - 1 = 0 -/
def line2 (k x y : ℝ) : Prop :=
  8 * x + (k + 1) * y + k - 1 = 0

theorem parallel_lines_k_value (k : ℝ) :
  (∀ x y : ℝ, are_parallel (k - 1) 1 8 (k + 1)) →
  k = 3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l2128_212844


namespace NUMINAMATH_CALUDE_linear_function_point_range_l2128_212887

theorem linear_function_point_range (x y : ℝ) : 
  y = 4 - 3 * x → y > -5 → x < 3 := by sorry

end NUMINAMATH_CALUDE_linear_function_point_range_l2128_212887


namespace NUMINAMATH_CALUDE_intersection_difference_is_zero_l2128_212871

noncomputable def f (x : ℝ) : ℝ := 2 - x^3 + x^4
noncomputable def g (x : ℝ) : ℝ := 1 + 2*x^3 + x^4

theorem intersection_difference_is_zero :
  ∀ x y : ℝ, f x = g x → f y = g y → |f x - g y| = 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_difference_is_zero_l2128_212871
