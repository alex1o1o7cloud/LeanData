import Mathlib

namespace NUMINAMATH_CALUDE_price_change_l2967_296726

/-- Theorem: Price change after 50% decrease and 60% increase --/
theorem price_change (P : ℝ) (P_pos : P > 0) :
  P * (1 - 0.5) * (1 + 0.6) = P * 0.8 := by
  sorry

#check price_change

end NUMINAMATH_CALUDE_price_change_l2967_296726


namespace NUMINAMATH_CALUDE_area_of_region_l2967_296740

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 23 ∧ 
   A = Real.pi * (Real.sqrt ((x + 3)^2 + (y - 2)^2))^2 ∧
   x^2 + y^2 + 6*x - 4*y - 10 = 0) := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l2967_296740


namespace NUMINAMATH_CALUDE_tylers_meal_combinations_l2967_296783

theorem tylers_meal_combinations (meat_types : ℕ) (vegetable_types : ℕ) (dessert_types : ℕ) 
  (h1 : meat_types = 4)
  (h2 : vegetable_types = 5)
  (h3 : dessert_types = 5) :
  meat_types * (vegetable_types.choose 3) * (dessert_types.choose 2) = 400 := by
  sorry

end NUMINAMATH_CALUDE_tylers_meal_combinations_l2967_296783


namespace NUMINAMATH_CALUDE_pool_water_volume_l2967_296719

/-- The volume of water in a cylindrical pool with a cylindrical column inside -/
theorem pool_water_volume 
  (pool_diameter : ℝ) 
  (pool_depth : ℝ) 
  (column_diameter : ℝ) 
  (column_depth : ℝ) 
  (h_pool_diameter : pool_diameter = 20)
  (h_pool_depth : pool_depth = 6)
  (h_column_diameter : column_diameter = 4)
  (h_column_depth : column_depth = pool_depth) :
  let pool_radius : ℝ := pool_diameter / 2
  let column_radius : ℝ := column_diameter / 2
  let pool_volume : ℝ := π * pool_radius^2 * pool_depth
  let column_volume : ℝ := π * column_radius^2 * column_depth
  pool_volume - column_volume = 576 * π := by
sorry


end NUMINAMATH_CALUDE_pool_water_volume_l2967_296719


namespace NUMINAMATH_CALUDE_units_digit_7_pow_2023_l2967_296714

def units_digit (n : ℕ) : ℕ := n % 10

def power_7_units_digit_pattern : List ℕ := [7, 9, 3, 1]

theorem units_digit_7_pow_2023 :
  units_digit (7^2023) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_7_pow_2023_l2967_296714


namespace NUMINAMATH_CALUDE_dumpling_storage_temp_l2967_296796

def storage_temp_range (x : ℝ) : Prop := -20 ≤ x ∧ x ≤ -16

theorem dumpling_storage_temp :
  (storage_temp_range (-17)) ∧
  (storage_temp_range (-18)) ∧
  (storage_temp_range (-19)) ∧
  (¬ storage_temp_range (-22)) :=
by sorry

end NUMINAMATH_CALUDE_dumpling_storage_temp_l2967_296796


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l2967_296774

def A : Set ℝ := {x | (x - 3) / (x - 7) < 0}
def B : Set ℝ := {x | x^2 - 12*x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

theorem set_operations_and_subset :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)}) ∧
  (∀ a : ℝ, C a ⊆ (A ∪ B) ↔ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l2967_296774


namespace NUMINAMATH_CALUDE_range_of_z_l2967_296709

theorem range_of_z (x y : ℝ) 
  (h1 : -4 ≤ x - y ∧ x - y ≤ -1)
  (h2 : -1 ≤ 4*x - y ∧ 4*x - y ≤ 5) :
  ∃ (z : ℝ), z = 9*x - y ∧ -1 ≤ z ∧ z ≤ 20 :=
sorry

end NUMINAMATH_CALUDE_range_of_z_l2967_296709


namespace NUMINAMATH_CALUDE_chocolate_distribution_l2967_296771

theorem chocolate_distribution (x y : ℕ) : 
  (y = x + 1) →  -- If each person is given 1 chocolate, then 1 chocolate is left
  (y = 2 * (x - 1)) →  -- If each person is given 2 chocolates, then 1 person will be left
  (x + y = 7) :=  -- The sum of persons and chocolates is 7
by
  sorry

#check chocolate_distribution

end NUMINAMATH_CALUDE_chocolate_distribution_l2967_296771


namespace NUMINAMATH_CALUDE_pool_capacity_l2967_296770

theorem pool_capacity (initial_percentage : ℚ) (final_percentage : ℚ) (added_water : ℚ) :
  initial_percentage = 0.4 →
  final_percentage = 0.8 →
  added_water = 300 →
  (∃ (total_capacity : ℚ), 
    total_capacity * final_percentage = total_capacity * initial_percentage + added_water ∧
    total_capacity = 750) :=
by sorry

end NUMINAMATH_CALUDE_pool_capacity_l2967_296770


namespace NUMINAMATH_CALUDE_window_offer_savings_l2967_296745

/-- Represents the window offer structure -/
structure WindowOffer where
  normalPrice : ℕ
  purchaseCount : ℕ
  freeCount : ℕ

/-- Calculates the cost for a given number of windows under the offer -/
def costUnderOffer (offer : WindowOffer) (windowCount : ℕ) : ℕ :=
  let fullSets := windowCount / (offer.purchaseCount + offer.freeCount)
  let remainingWindows := windowCount % (offer.purchaseCount + offer.freeCount)
  (fullSets * offer.purchaseCount + min remainingWindows offer.purchaseCount) * offer.normalPrice

/-- Calculates the savings when purchasing windows together vs separately -/
def calculateSavings (offer : WindowOffer) (dave : ℕ) (doug : ℕ) : ℕ :=
  let separateCost := costUnderOffer offer dave + costUnderOffer offer doug
  let combinedCost := costUnderOffer offer (dave + doug)
  (dave + doug) * offer.normalPrice - combinedCost

/-- The main theorem stating the savings amount -/
theorem window_offer_savings :
  let offer : WindowOffer := ⟨100, 6, 2⟩
  let davesWindows : ℕ := 9
  let dougsWindows : ℕ := 10
  calculateSavings offer davesWindows dougsWindows = 400 := by
  sorry

end NUMINAMATH_CALUDE_window_offer_savings_l2967_296745


namespace NUMINAMATH_CALUDE_cereal_eating_time_l2967_296754

/-- The time taken for three people to eat a certain amount of cereal together -/
def time_to_eat (fat_rate thin_rate medium_rate total_cereal : ℚ) : ℚ :=
  total_cereal / (fat_rate + thin_rate + medium_rate)

theorem cereal_eating_time :
  let fat_rate : ℚ := 1 / 15
  let thin_rate : ℚ := 1 / 35
  let medium_rate : ℚ := 1 / 25
  let total_cereal : ℚ := 5
  time_to_eat fat_rate thin_rate medium_rate total_cereal = 2625 / 71 :=
by sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l2967_296754


namespace NUMINAMATH_CALUDE_certain_number_proof_l2967_296765

theorem certain_number_proof : ∃ x : ℝ, (0.60 * x = 0.50 * 600) ∧ (x = 500) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2967_296765


namespace NUMINAMATH_CALUDE_unique_fixed_point_for_rotation_invariant_function_l2967_296769

/-- A function is invariant under π rotation around the origin -/
def RotationInvariant (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (-x) = -y

/-- The main theorem -/
theorem unique_fixed_point_for_rotation_invariant_function (f : ℝ → ℝ) 
    (h : RotationInvariant f) : 
    ∃! x, f x = x :=
  sorry

end NUMINAMATH_CALUDE_unique_fixed_point_for_rotation_invariant_function_l2967_296769


namespace NUMINAMATH_CALUDE_black_pens_count_l2967_296744

theorem black_pens_count (total_pens : ℕ) (red_pens black_pens : ℕ) : 
  (3 : ℚ) / 10 * total_pens = red_pens →
  (1 : ℚ) / 5 * total_pens = black_pens →
  red_pens = 12 →
  black_pens = 8 := by
  sorry

end NUMINAMATH_CALUDE_black_pens_count_l2967_296744


namespace NUMINAMATH_CALUDE_rajans_position_l2967_296706

/-- Given a row of boys, this theorem proves Rajan's position from the left end. -/
theorem rajans_position 
  (total_boys : ℕ) 
  (vinays_position_from_right : ℕ) 
  (boys_between : ℕ) 
  (h1 : total_boys = 24) 
  (h2 : vinays_position_from_right = 10) 
  (h3 : boys_between = 8) : 
  total_boys - (vinays_position_from_right - 1 + boys_between + 1) = 6 := by
sorry

end NUMINAMATH_CALUDE_rajans_position_l2967_296706


namespace NUMINAMATH_CALUDE_chinese_chess_pieces_sum_l2967_296799

theorem chinese_chess_pieces_sum :
  ∀ (Rook Knight Cannon : ℕ),
    Rook / Knight = 2 →
    Cannon / Rook = 4 →
    Cannon - Knight = 56 →
    Rook + Knight + Cannon = 88 :=
by
  sorry

end NUMINAMATH_CALUDE_chinese_chess_pieces_sum_l2967_296799


namespace NUMINAMATH_CALUDE_min_value_expression_l2967_296751

theorem min_value_expression (a b c : ℝ) (h1 : b > a) (h2 : a > c) (h3 : b ≠ 0) :
  ((a + b)^2 + (b + c)^2 + (c + a)^2) / b^2 ≥ 5.5 ∧
  ∃ (a' b' c' : ℝ), b' > a' ∧ a' > c' ∧ b' ≠ 0 ∧
    ((a' + b')^2 + (b' + c')^2 + (c' + a')^2) / b'^2 = 5.5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2967_296751


namespace NUMINAMATH_CALUDE_total_amount_is_952_20_l2967_296767

/-- Calculate the total amount paid for three items with given original prices, discounts, and sales taxes. -/
def total_amount_paid (vase_price teacups_price plate_price : ℝ)
                      (vase_discount teacups_discount : ℝ)
                      (vase_tax teacups_tax plate_tax : ℝ) : ℝ :=
  let vase_sale_price := vase_price * (1 - vase_discount)
  let teacups_sale_price := teacups_price * (1 - teacups_discount)
  let vase_total := vase_sale_price * (1 + vase_tax)
  let teacups_total := teacups_sale_price * (1 + teacups_tax)
  let plate_total := plate_price * (1 + plate_tax)
  vase_total + teacups_total + plate_total

/-- The total amount paid for the three porcelain items is $952.20. -/
theorem total_amount_is_952_20 :
  total_amount_paid 200 300 500 0.35 0.20 0.10 0.08 0.10 = 952.20 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_952_20_l2967_296767


namespace NUMINAMATH_CALUDE_farmer_sold_two_ducks_l2967_296705

/-- Represents the farmer's market scenario -/
structure FarmerMarket where
  duck_price : ℕ
  chicken_price : ℕ
  chickens_sold : ℕ
  wheelbarrow_profit : ℕ

/-- Calculates the number of ducks sold given the market conditions -/
def ducks_sold (market : FarmerMarket) : ℕ :=
  let total_earnings := 2 * market.wheelbarrow_profit
  let chicken_earnings := market.chicken_price * market.chickens_sold
  (total_earnings - chicken_earnings) / market.duck_price

/-- Theorem stating that the number of ducks sold is 2 -/
theorem farmer_sold_two_ducks : 
  ∀ (market : FarmerMarket), 
  market.duck_price = 10 ∧ 
  market.chicken_price = 8 ∧ 
  market.chickens_sold = 5 ∧ 
  market.wheelbarrow_profit = 60 →
  ducks_sold market = 2 := by
  sorry


end NUMINAMATH_CALUDE_farmer_sold_two_ducks_l2967_296705


namespace NUMINAMATH_CALUDE_line_intercept_form_l2967_296759

/-- Given a line with equation 3x - 2y = 4, its intercept form is x/(4/3) + y/(-2) = 1 -/
theorem line_intercept_form :
  ∀ (x y : ℝ), 3*x - 2*y = 4 → x/(4/3) + y/(-2) = 1 := by sorry

end NUMINAMATH_CALUDE_line_intercept_form_l2967_296759


namespace NUMINAMATH_CALUDE_half_power_inequality_l2967_296711

theorem half_power_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (1/2 : ℝ)^a < (1/2 : ℝ)^b := by sorry

end NUMINAMATH_CALUDE_half_power_inequality_l2967_296711


namespace NUMINAMATH_CALUDE_special_triangle_all_angles_60_l2967_296730

/-- A triangle with angles in arithmetic progression and sides in geometric progression -/
structure SpecialTriangle where
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Sides of the triangle opposite to angles A, B, C respectively
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles form an arithmetic progression
  angle_progression : ∃ (d : ℝ), B - A = C - B
  -- One angle is 60°
  one_angle_60 : A = 60 ∨ B = 60 ∨ C = 60
  -- Sum of angles is 180°
  angle_sum : A + B + C = 180
  -- Sides form a geometric progression
  side_progression : b^2 = a * c
  -- Triangle inequality
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  -- All sides are positive
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

/-- Theorem: In a SpecialTriangle, all angles are 60° -/
theorem special_triangle_all_angles_60 (t : SpecialTriangle) : t.A = 60 ∧ t.B = 60 ∧ t.C = 60 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_all_angles_60_l2967_296730


namespace NUMINAMATH_CALUDE_special_triangle_side_length_l2967_296728

/-- A triangle with special median properties -/
structure SpecialTriangle where
  /-- The length of side EF -/
  EF : ℝ
  /-- The length of side DF -/
  DF : ℝ
  /-- The median from D is perpendicular to the median from E -/
  medians_perpendicular : Bool

/-- Theorem: In a special triangle with EF = 10, DF = 8, and perpendicular medians, DE = 18 -/
theorem special_triangle_side_length (t : SpecialTriangle) 
  (h1 : t.EF = 10) 
  (h2 : t.DF = 8) 
  (h3 : t.medians_perpendicular = true) : 
  ∃ DE : ℝ, DE = 18 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_side_length_l2967_296728


namespace NUMINAMATH_CALUDE_partition_spread_bound_l2967_296729

/-- The number of partitions of a natural number -/
def P (n : ℕ) : ℕ := sorry

/-- The spread of a partition -/
def spread (partition : List ℕ) : ℕ := sorry

/-- The sum of spreads of all partitions of a natural number -/
def Q (n : ℕ) : ℕ := sorry

/-- Theorem: Q(n) ≤ √(2n) · P(n) for all natural numbers n -/
theorem partition_spread_bound (n : ℕ) : Q n ≤ Real.sqrt (2 * n) * P n := by sorry

end NUMINAMATH_CALUDE_partition_spread_bound_l2967_296729


namespace NUMINAMATH_CALUDE_rectangle_division_exists_l2967_296725

/-- A rectangle in a 2D plane --/
structure Rectangle where
  x : ℝ
  y : ℝ
  width : ℝ
  height : ℝ

/-- Predicate to check if a set of points forms a rectangle --/
def IsRectangle (s : Set (ℝ × ℝ)) : Prop := sorry

/-- A division of a rectangle into smaller rectangles --/
def RectangleDivision (r : Rectangle) (divisions : List Rectangle) : Prop := sorry

/-- Check if the union of two rectangles forms a rectangle --/
def UnionIsRectangle (r1 r2 : Rectangle) : Prop := sorry

/-- Main theorem: There exists a division of a rectangle into 5 smaller rectangles
    such that the union of any two of them is not a rectangle --/
theorem rectangle_division_exists :
  ∃ (r : Rectangle) (divisions : List Rectangle),
    RectangleDivision r divisions ∧
    divisions.length = 5 ∧
    ∀ (r1 r2 : Rectangle), r1 ∈ divisions → r2 ∈ divisions → r1 ≠ r2 →
      ¬UnionIsRectangle r1 r2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_exists_l2967_296725


namespace NUMINAMATH_CALUDE_union_necessary_not_sufficient_l2967_296781

theorem union_necessary_not_sufficient (A B : Set α) :
  (∀ x, x ∈ A ∩ B → x ∈ A ∪ B) ∧
  (∃ x, x ∈ A ∪ B ∧ x ∉ A ∩ B) := by
  sorry

end NUMINAMATH_CALUDE_union_necessary_not_sufficient_l2967_296781


namespace NUMINAMATH_CALUDE_order_of_numbers_l2967_296749

theorem order_of_numbers : Real.log 0.76 < 0.76 ∧ 0.76 < 60.7 := by sorry

end NUMINAMATH_CALUDE_order_of_numbers_l2967_296749


namespace NUMINAMATH_CALUDE_triangle_side_length_squared_l2967_296734

theorem triangle_side_length_squared (A B C : ℝ × ℝ) :
  let area := 10
  let tan_ABC := 5
  area = (1/2) * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1) →
  tan_ABC = (B.2 - A.2) / (B.1 - A.1) →
  ∃ (AC_squared : ℝ), AC_squared = (C.1 - A.1)^2 + (C.2 - A.2)^2 ∧
    AC_squared ≥ -8 + 8 * Real.sqrt 26 :=
by sorry

#check triangle_side_length_squared

end NUMINAMATH_CALUDE_triangle_side_length_squared_l2967_296734


namespace NUMINAMATH_CALUDE_tan_strictly_increasing_interval_l2967_296712

theorem tan_strictly_increasing_interval (k : ℤ) :
  StrictMonoOn (fun x ↦ Real.tan (2 * x - π / 3))
    (Set.Ioo (k * π / 2 - π / 12) (k * π / 2 + 5 * π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_tan_strictly_increasing_interval_l2967_296712


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_23_l2967_296733

theorem modular_inverse_of_5_mod_23 :
  ∃ x : ℕ, x < 23 ∧ (5 * x) % 23 = 1 ∧ x = 14 := by
sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_23_l2967_296733


namespace NUMINAMATH_CALUDE_double_base_exponent_l2967_296741

theorem double_base_exponent (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (2 * a) ^ (2 * b) = (a ^ 2) ^ b * x ^ b → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_double_base_exponent_l2967_296741


namespace NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_8_ending_4_l2967_296708

theorem greatest_three_digit_divisible_by_8_ending_4 : ∃ n : ℕ, 
  n = 984 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  n % 8 = 0 ∧ 
  n % 10 = 4 ∧ 
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 8 = 0 ∧ m % 10 = 4 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_8_ending_4_l2967_296708


namespace NUMINAMATH_CALUDE_friends_assignment_l2967_296750

-- Define the types for names, surnames, and grades
inductive Name : Type
  | Petya | Kolya | Alyosha | Misha | Dima | Borya | Vasya

inductive Surname : Type
  | Ivanov | Petrov | Krylov | Orlov

inductive Grade : Type
  | First | Second | Third | Fourth

-- Define a function to represent the assignment of names, surnames, and grades
def Assignment := Name → Surname × Grade

-- Define the conditions
def not_first_grader (a : Assignment) (n : Name) : Prop :=
  (a n).2 ≠ Grade.First

def different_streets (a : Assignment) (n1 n2 : Name) : Prop :=
  (a n1).1 ≠ (a n2).1

def one_year_older (a : Assignment) (n1 n2 : Name) : Prop :=
  match (a n1).2, (a n2).2 with
  | Grade.Second, Grade.First => True
  | Grade.Third, Grade.Second => True
  | Grade.Fourth, Grade.Third => True
  | _, _ => False

def neighbors (a : Assignment) (n1 n2 : Name) : Prop :=
  (a n1).1 = (a n2).1

def met_year_ago_first_grade (a : Assignment) (n : Name) : Prop :=
  (a n).2 = Grade.Second

def gave_last_year_textbook (a : Assignment) (n1 n2 : Name) : Prop :=
  match (a n1).2, (a n2).2 with
  | Grade.Second, Grade.First => True
  | Grade.Third, Grade.Second => True
  | Grade.Fourth, Grade.Third => True
  | _, _ => False

-- Define the theorem
theorem friends_assignment (a : Assignment) :
  not_first_grader a Name.Borya
  ∧ different_streets a Name.Vasya Name.Dima
  ∧ one_year_older a Name.Misha Name.Dima
  ∧ neighbors a Name.Borya Name.Vasya
  ∧ met_year_ago_first_grade a Name.Misha
  ∧ gave_last_year_textbook a Name.Vasya Name.Borya
  → a Name.Dima = (Surname.Ivanov, Grade.First)
  ∧ a Name.Misha = (Surname.Krylov, Grade.Second)
  ∧ a Name.Borya = (Surname.Petrov, Grade.Third)
  ∧ a Name.Vasya = (Surname.Orlov, Grade.Fourth) :=
by
  sorry

end NUMINAMATH_CALUDE_friends_assignment_l2967_296750


namespace NUMINAMATH_CALUDE_total_area_three_shapes_l2967_296763

theorem total_area_three_shapes 
  (rect_area square_area tri_area : ℝ)
  (rect_square_overlap rect_tri_overlap square_tri_overlap : ℝ)
  (all_overlap : ℝ) :
  let total_area := rect_area + square_area + tri_area - 
                    rect_square_overlap - rect_tri_overlap - square_tri_overlap + 
                    all_overlap
  total_area = 66 :=
by sorry

end NUMINAMATH_CALUDE_total_area_three_shapes_l2967_296763


namespace NUMINAMATH_CALUDE_sally_payment_l2967_296768

/-- The amount Sally needs to pay out of pocket to buy books for her students -/
def sally_out_of_pocket (budget : ℚ) (num_students : ℕ) (reading_book_price : ℚ) 
  (math_book_price : ℚ) (discount_rate : ℚ) (discount_threshold : ℕ) : ℚ :=
  let total_reading_books := num_students * reading_book_price
  let discounted_reading_books := if num_students ≥ discount_threshold
    then total_reading_books * (1 - discount_rate)
    else total_reading_books
  let total_math_books := num_students * math_book_price
  let total_cost := discounted_reading_books + total_math_books
  max (total_cost - budget) 0

/-- Theorem stating that Sally needs to pay $467.50 out of pocket -/
theorem sally_payment : 
  sally_out_of_pocket 320 35 15 9 (1/10) 25 = 467.5 := by
  sorry

end NUMINAMATH_CALUDE_sally_payment_l2967_296768


namespace NUMINAMATH_CALUDE_x_minus_y_values_l2967_296704

theorem x_minus_y_values (x y : ℝ) (h1 : x^2 = 9) (h2 : |y| = 4) (h3 : x < y) :
  x - y = -7 ∨ x - y = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l2967_296704


namespace NUMINAMATH_CALUDE_total_cards_is_690_l2967_296732

/-- The number of get well cards Mariela received in the hospital -/
def cards_in_hospital : ℕ := 403

/-- The number of get well cards Mariela received at home -/
def cards_at_home : ℕ := 287

/-- The total number of get well cards Mariela received -/
def total_cards : ℕ := cards_in_hospital + cards_at_home

/-- Theorem stating that the total number of get well cards Mariela received is 690 -/
theorem total_cards_is_690 : total_cards = 690 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_is_690_l2967_296732


namespace NUMINAMATH_CALUDE_base6_addition_theorem_l2967_296787

/-- Converts a base 6 number represented as a list of digits to base 10 -/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 represented as a list of digits -/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (n : Nat) (acc : List Nat) : List Nat :=
      if n = 0 then acc else aux (n / 6) ((n % 6) :: acc)
    aux n []

/-- Adds two base 6 numbers represented as lists of digits -/
def addBase6 (a b : List Nat) : List Nat :=
  let sum := base6ToBase10 a + base6ToBase10 b
  base10ToBase6 sum

theorem base6_addition_theorem :
  let a := [2, 4, 5, 3]  -- 2453₆
  let b := [1, 6, 4, 3, 2]  -- 16432₆
  addBase6 a b = [2, 5, 5, 4, 5] ∧  -- 25545₆
  base6ToBase10 (addBase6 a b) = 3881 := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_theorem_l2967_296787


namespace NUMINAMATH_CALUDE_victoria_beacon_ratio_l2967_296701

/-- The population of Richmond -/
def richmond_population : ℕ := 3000

/-- The population of Beacon -/
def beacon_population : ℕ := 500

/-- The difference between Richmond's and Victoria's populations -/
def richmond_victoria_diff : ℕ := 1000

/-- The population of Victoria -/
def victoria_population : ℕ := richmond_population - richmond_victoria_diff

theorem victoria_beacon_ratio : 
  (victoria_population : ℚ) / (beacon_population : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_victoria_beacon_ratio_l2967_296701


namespace NUMINAMATH_CALUDE_inequalities_from_sqrt_reciprocal_l2967_296738

theorem inequalities_from_sqrt_reciprocal (a b : ℝ) (h : 1 / Real.sqrt a > 1 / Real.sqrt b) :
  (b / (a + b) + a / (2 * b) ≥ (2 * Real.sqrt 2 - 1) / 2) ∧
  ((b + 1) / (a + 1) < b / a) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_from_sqrt_reciprocal_l2967_296738


namespace NUMINAMATH_CALUDE_comic_stacking_arrangements_l2967_296798

def batman_comics : ℕ := 8
def superman_comics : ℕ := 6
def wonder_woman_comics : ℕ := 3

theorem comic_stacking_arrangements :
  (batman_comics.factorial * superman_comics.factorial * wonder_woman_comics.factorial) *
  (batman_comics + superman_comics + wonder_woman_comics).choose 3 = 1040486400 :=
by sorry

end NUMINAMATH_CALUDE_comic_stacking_arrangements_l2967_296798


namespace NUMINAMATH_CALUDE_tan_identities_l2967_296776

theorem tan_identities (α : Real) (h : Real.tan (π / 4 + α) = 3) :
  (Real.tan α = 1 / 2) ∧
  (Real.tan (2 * α) = 4 / 3) ∧
  ((2 * Real.sin α * Real.cos α + 3 * Real.cos (2 * α)) / 
   (5 * Real.cos (2 * α) - 3 * Real.sin (2 * α)) = 13 / 3) := by
  sorry

end NUMINAMATH_CALUDE_tan_identities_l2967_296776


namespace NUMINAMATH_CALUDE_probability_under20_is_one_tenth_l2967_296723

/-- Represents a group of people with age distribution --/
structure AgeGroup where
  total : ℕ
  over30 : ℕ
  under20 : ℕ
  h1 : total = over30 + under20
  h2 : over30 < total

/-- Calculates the probability of selecting a person under 20 from the group --/
def probabilityUnder20 (group : AgeGroup) : ℚ :=
  group.under20 / group.total

/-- The main theorem to prove --/
theorem probability_under20_is_one_tenth
  (group : AgeGroup)
  (h3 : group.total = 100)
  (h4 : group.over30 = 90) :
  probabilityUnder20 group = 1/10 := by
  sorry


end NUMINAMATH_CALUDE_probability_under20_is_one_tenth_l2967_296723


namespace NUMINAMATH_CALUDE_new_year_markup_verify_new_year_markup_l2967_296718

/-- Calculates the New Year season markup percentage given other price adjustments and final profit -/
theorem new_year_markup (initial_markup : ℝ) (february_discount : ℝ) (final_profit : ℝ) : ℝ :=
  let new_year_markup := 
    ((1 + final_profit) / ((1 + initial_markup) * (1 - february_discount)) - 1) * 100
  by
    -- The proof would go here
    sorry

/-- Verifies that the New Year markup is 25% given the problem conditions -/
theorem verify_new_year_markup : 
  new_year_markup 0.20 0.09 0.365 = 25 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_new_year_markup_verify_new_year_markup_l2967_296718


namespace NUMINAMATH_CALUDE_johns_age_l2967_296791

theorem johns_age : ∃ (j : ℝ), j = 22.5 ∧ (j - 10 = (1 / 3) * (j + 15)) := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l2967_296791


namespace NUMINAMATH_CALUDE_annual_population_increase_rate_l2967_296792

theorem annual_population_increase_rate (initial_population final_population : ℕ) 
  (h : initial_population = 14000 ∧ final_population = 16940) : 
  ∃ r : ℝ, initial_population * (1 + r)^2 = final_population := by
  sorry

end NUMINAMATH_CALUDE_annual_population_increase_rate_l2967_296792


namespace NUMINAMATH_CALUDE_m_range_l2967_296731

/-- The range of m given the specified conditions -/
theorem m_range (m : ℝ) : 
  (¬ ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ∧ 
  ((∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ∨ 
   (∀ x : ℝ, x ≥ 2 → x + m/x - 2 > 0)) →
  0 < m ∧ m ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_m_range_l2967_296731


namespace NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l2967_296715

theorem quadratic_roots_reciprocal_sum (x₁ x₂ : ℝ) :
  x₁^2 - 4*x₁ - 2 = 0 →
  x₂^2 - 4*x₂ - 2 = 0 →
  x₁ ≠ x₂ →
  (1/x₁) + (1/x₂) = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l2967_296715


namespace NUMINAMATH_CALUDE_solve_equation_l2967_296795

theorem solve_equation (x : ℝ) (h : x ≠ 0) :
  (2 / x + (3 / x) / (6 / x) = 1.25) → x = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2967_296795


namespace NUMINAMATH_CALUDE_sequence_2011th_term_l2967_296760

def sequence_term (n : ℕ) : ℕ → ℕ
  | 0 => 52
  | (m + 1) => 
    let prev := sequence_term n m
    let last_digit := prev % 10
    let remaining := prev / 10
    last_digit ^ 2 + 2 * remaining

def is_cyclic (seq : ℕ → ℕ) (start : ℕ) (length : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ start → seq k = seq (k % length + start)

theorem sequence_2011th_term :
  ∃ (start length : ℕ),
    start > 0 ∧
    length > 0 ∧
    is_cyclic (sequence_term 0) start length ∧
    sequence_term 0 2010 = 18 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2011th_term_l2967_296760


namespace NUMINAMATH_CALUDE_a_fourth_plus_inverse_a_fourth_l2967_296762

theorem a_fourth_plus_inverse_a_fourth (a : ℝ) (h : a - 1/a = -2) : 
  a^4 + 1/a^4 = 34 := by
sorry

end NUMINAMATH_CALUDE_a_fourth_plus_inverse_a_fourth_l2967_296762


namespace NUMINAMATH_CALUDE_union_of_sets_l2967_296702

theorem union_of_sets : 
  let A : Set ℕ := {0, 1, 2, 3}
  let B : Set ℕ := {1, 2, 4}
  A ∪ B = {0, 1, 2, 3, 4} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l2967_296702


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2967_296737

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := fun x ↦ (x - 1)^2 - 4
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = -1 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
    ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2967_296737


namespace NUMINAMATH_CALUDE_batsman_average_excluding_extremes_l2967_296742

def batting_average : ℝ := 60
def num_innings : ℕ := 46
def highest_score : ℕ := 194
def score_difference : ℕ := 180

theorem batsman_average_excluding_extremes :
  let total_runs : ℝ := batting_average * num_innings
  let lowest_score : ℕ := highest_score - score_difference
  let runs_excluding_extremes : ℝ := total_runs - highest_score - lowest_score
  let innings_excluding_extremes : ℕ := num_innings - 2
  (runs_excluding_extremes / innings_excluding_extremes : ℝ) = 58 := by sorry

end NUMINAMATH_CALUDE_batsman_average_excluding_extremes_l2967_296742


namespace NUMINAMATH_CALUDE_average_chapters_per_book_l2967_296752

theorem average_chapters_per_book 
  (total_chapters : Float) 
  (total_books : Float) 
  (h1 : total_chapters = 17.0) 
  (h2 : total_books = 4.0) : 
  total_chapters / total_books = 4.25 := by
sorry

end NUMINAMATH_CALUDE_average_chapters_per_book_l2967_296752


namespace NUMINAMATH_CALUDE_inequality_part1_inequality_part2_l2967_296782

-- Part 1
theorem inequality_part1 (x p : ℝ) :
  (∀ p, |p| ≤ 2 → x^2 + p*x + 1 > 2*x + p) → x < -1 ∨ x > 3 :=
sorry

-- Part 2
theorem inequality_part2 (x p : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 4 → x^2 + p*x + 1 > 2*x + p) → p > -1 :=
sorry

end NUMINAMATH_CALUDE_inequality_part1_inequality_part2_l2967_296782


namespace NUMINAMATH_CALUDE_pens_given_to_sharon_l2967_296747

/-- The number of pens given to Sharon -/
def pens_to_sharon (initial_pens : ℕ) (mike_pens : ℕ) (final_pens : ℕ) : ℕ :=
  2 * (initial_pens + mike_pens) - final_pens

theorem pens_given_to_sharon :
  pens_to_sharon 20 22 65 = 19 := by
  sorry

end NUMINAMATH_CALUDE_pens_given_to_sharon_l2967_296747


namespace NUMINAMATH_CALUDE_complement_union_equals_divisible_by_3_l2967_296746

-- Define the universal set U as the set of all integers
def U : Set ℤ := Set.univ

-- Define set A
def A : Set ℤ := {x | ∃ k : ℤ, x = 3*k + 1}

-- Define set B
def B : Set ℤ := {x | ∃ k : ℤ, x = 3*k + 2}

-- Define the set of integers divisible by 3
def DivisibleBy3 : Set ℤ := {x | ∃ k : ℤ, x = 3*k}

-- Theorem statement
theorem complement_union_equals_divisible_by_3 :
  (U \ (A ∪ B)) = DivisibleBy3 :=
sorry

end NUMINAMATH_CALUDE_complement_union_equals_divisible_by_3_l2967_296746


namespace NUMINAMATH_CALUDE_largest_n_binomial_equality_l2967_296721

theorem largest_n_binomial_equality : ∃ (n : ℕ), (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) ∧ 
  (∀ (m : ℕ), Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_n_binomial_equality_l2967_296721


namespace NUMINAMATH_CALUDE_opposite_of_2023_l2967_296764

-- Define the concept of opposite for integers
def opposite (n : ℤ) : ℤ := -n

-- Theorem statement
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l2967_296764


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2967_296756

theorem sum_of_fractions : (1 : ℚ) / 6 + (5 : ℚ) / 12 = (7 : ℚ) / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2967_296756


namespace NUMINAMATH_CALUDE_concatenated_square_exists_l2967_296727

theorem concatenated_square_exists : ∃ (A : ℕ), ∃ (n : ℕ), ∃ (B : ℕ), 
  (10^n + 1) * A = B^2 ∧ A > 0 ∧ A < 10^n := by
  sorry

end NUMINAMATH_CALUDE_concatenated_square_exists_l2967_296727


namespace NUMINAMATH_CALUDE_six_and_neg_six_are_opposite_l2967_296736

/-- Two real numbers are opposite if one is the negative of the other -/
def are_opposite (a b : ℝ) : Prop := b = -a

/-- 6 and -6 are opposite numbers -/
theorem six_and_neg_six_are_opposite : are_opposite 6 (-6) := by
  sorry

end NUMINAMATH_CALUDE_six_and_neg_six_are_opposite_l2967_296736


namespace NUMINAMATH_CALUDE_standard_deviation_of_data_set_l2967_296720

def data_set : List ℝ := [10, 5, 4, 2, 2, 1]

theorem standard_deviation_of_data_set :
  let x := data_set[2]
  ∀ (mode median : ℝ),
    x ≠ 5 →
    mode = 2 →
    median = (x + 2) / 2 →
    mode = 2/3 * median →
    let mean := (data_set.sum) / (data_set.length : ℝ)
    let variance := (data_set.map (λ y => (y - mean)^2)).sum / (data_set.length : ℝ)
    Real.sqrt variance = 3 := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_of_data_set_l2967_296720


namespace NUMINAMATH_CALUDE_wedding_catering_ratio_l2967_296784

/-- Represents the catering problem for Jenny's wedding --/
def CateringProblem (total_guests : ℕ) (steak_cost chicken_cost : ℚ) (total_budget : ℚ) : Prop :=
  ∃ (steak_guests chicken_guests : ℕ),
    steak_guests + chicken_guests = total_guests ∧
    steak_cost * steak_guests + chicken_cost * chicken_guests = total_budget ∧
    steak_guests = 3 * chicken_guests

/-- Theorem stating that the given conditions result in a 3:1 ratio of steak to chicken guests --/
theorem wedding_catering_ratio :
  CateringProblem 80 25 18 1860 :=
by
  sorry

end NUMINAMATH_CALUDE_wedding_catering_ratio_l2967_296784


namespace NUMINAMATH_CALUDE_tv_selection_problem_l2967_296748

theorem tv_selection_problem (type_a : ℕ) (type_b : ℕ) (total_selection : ℕ) :
  type_a = 4 →
  type_b = 5 →
  total_selection = 3 →
  (Nat.choose type_a 2 * Nat.choose type_b 1) + (Nat.choose type_a 1 * Nat.choose type_b 2) = 70 :=
by sorry

end NUMINAMATH_CALUDE_tv_selection_problem_l2967_296748


namespace NUMINAMATH_CALUDE_cos_300_degrees_l2967_296755

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l2967_296755


namespace NUMINAMATH_CALUDE_symmetric_second_quadrant_condition_l2967_296780

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of symmetry about the origin -/
def symmetricAboutOrigin (p : Point2D) : Prop :=
  ∃ q : Point2D, q.x = -p.x ∧ q.y = -p.y

/-- Definition of a point being in the second quadrant -/
def inSecondQuadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem stating the condition for m -/
theorem symmetric_second_quadrant_condition (m : ℝ) :
  symmetricAboutOrigin ⟨-m, m-3⟩ ∧ inSecondQuadrant ⟨m, 3-m⟩ → m < 0 :=
sorry

end NUMINAMATH_CALUDE_symmetric_second_quadrant_condition_l2967_296780


namespace NUMINAMATH_CALUDE_quadrilateral_ratio_theorem_l2967_296788

-- Define the quadrilateral and points
variable (A B C D K L M N P : ℝ × ℝ)
variable (α β : ℝ)

-- Define the conditions
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def point_on_side (X Y Z : ℝ × ℝ) : Prop := sorry

def ratio_equals (A B X Y : ℝ × ℝ) (r : ℝ) : Prop := sorry

def intersection_point (K M L N P : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem quadrilateral_ratio_theorem 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_K : point_on_side K A B)
  (h_L : point_on_side L B C)
  (h_M : point_on_side M C D)
  (h_N : point_on_side N D A)
  (h_AK_KB : ratio_equals A K K B α)
  (h_DM_MC : ratio_equals D M M C α)
  (h_BL_LC : ratio_equals B L L C β)
  (h_AN_ND : ratio_equals A N N D β)
  (h_P : intersection_point K M L N P) :
  ratio_equals N P P L α ∧ ratio_equals K P P M β := by sorry

end NUMINAMATH_CALUDE_quadrilateral_ratio_theorem_l2967_296788


namespace NUMINAMATH_CALUDE_problem_solution_l2967_296753

theorem problem_solution (a b : ℚ) 
  (h1 : 5 + a = 3 - b) 
  (h2 : 3 + b = 8 + a) : 
  5 - a = 17 / 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2967_296753


namespace NUMINAMATH_CALUDE_prove_sales_tax_percentage_l2967_296710

def total_spent : ℝ := 184.80
def tip_percentage : ℝ := 20
def food_price : ℝ := 140

def sales_tax_percentage : ℝ := 10

theorem prove_sales_tax_percentage :
  let price_with_tax := food_price * (1 + sales_tax_percentage / 100)
  let total_with_tip := price_with_tax * (1 + tip_percentage / 100)
  total_with_tip = total_spent :=
by sorry

end NUMINAMATH_CALUDE_prove_sales_tax_percentage_l2967_296710


namespace NUMINAMATH_CALUDE_watermelon_price_per_pound_l2967_296743

/-- The price per pound of watermelons sold by Farmer Kent -/
def price_per_pound (watermelon_weight : ℕ) (num_watermelons : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / (watermelon_weight * num_watermelons)

/-- Theorem stating that the price per pound of Farmer Kent's watermelons is $2 -/
theorem watermelon_price_per_pound :
  price_per_pound 23 18 828 = 2 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_price_per_pound_l2967_296743


namespace NUMINAMATH_CALUDE_double_divide_four_equals_twelve_l2967_296766

theorem double_divide_four_equals_twelve (x : ℝ) : (2 * x) / 4 = 12 → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_double_divide_four_equals_twelve_l2967_296766


namespace NUMINAMATH_CALUDE_unique_solution_for_difference_of_squares_l2967_296757

theorem unique_solution_for_difference_of_squares : 
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 - y^2 = 204 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_difference_of_squares_l2967_296757


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2967_296739

theorem absolute_value_equation_solution :
  ∃! n : ℝ, |n + 6| = 2 - n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2967_296739


namespace NUMINAMATH_CALUDE_square_diameter_double_area_l2967_296793

theorem square_diameter_double_area (d₁ : ℝ) (d₂ : ℝ) : 
  d₁ = 4 * Real.sqrt 2 →
  (d₂ / 2)^2 = 2 * (d₁ / 2)^2 →
  d₂ = 8 :=
by sorry

end NUMINAMATH_CALUDE_square_diameter_double_area_l2967_296793


namespace NUMINAMATH_CALUDE_fourth_guard_distance_theorem_l2967_296778

/-- Represents a rectangular classified area with guards -/
structure ClassifiedArea where
  length : ℝ
  width : ℝ
  perimeter : ℝ
  guard_count : ℕ
  three_guards_distance : ℝ

/-- Calculates the distance run by the fourth guard -/
def fourth_guard_distance (area : ClassifiedArea) : ℝ :=
  area.perimeter - area.three_guards_distance

/-- Theorem stating the distance run by the fourth guard -/
theorem fourth_guard_distance_theorem (area : ClassifiedArea) 
  (h1 : area.length = 200)
  (h2 : area.width = 300)
  (h3 : area.perimeter = 2 * (area.length + area.width))
  (h4 : area.guard_count = 4)
  (h5 : area.three_guards_distance = 850)
  : fourth_guard_distance area = 150 := by
  sorry

end NUMINAMATH_CALUDE_fourth_guard_distance_theorem_l2967_296778


namespace NUMINAMATH_CALUDE_plant_branches_theorem_l2967_296794

theorem plant_branches_theorem : ∃ (x : ℕ), x > 0 ∧ 1 + x + x^2 = 57 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_plant_branches_theorem_l2967_296794


namespace NUMINAMATH_CALUDE_complement_intersection_eq_set_l2967_296790

def U : Finset ℕ := {1,2,3,4,5}
def A : Finset ℕ := {1,2,3}
def B : Finset ℕ := {3,4,5}

theorem complement_intersection_eq_set : 
  (U \ (A ∩ B)) = {1,2,4,5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_eq_set_l2967_296790


namespace NUMINAMATH_CALUDE_stating_correct_deposit_equation_l2967_296724

/-- Represents the annual interest rate as a decimal -/
def annual_rate : ℝ := 0.0369

/-- Represents the number of years for the fixed deposit -/
def years : ℕ := 3

/-- Represents the tax rate on interest as a decimal -/
def tax_rate : ℝ := 0.2

/-- Represents the final withdrawal amount in yuan -/
def final_amount : ℝ := 5442.8

/-- 
Theorem stating the correct equation for calculating the initial deposit amount,
given the annual interest rate, number of years, tax rate, and final withdrawal amount.
-/
theorem correct_deposit_equation (x : ℝ) :
  x + x * annual_rate * (years : ℝ) * (1 - tax_rate) = final_amount :=
sorry

end NUMINAMATH_CALUDE_stating_correct_deposit_equation_l2967_296724


namespace NUMINAMATH_CALUDE_absolute_difference_100th_terms_l2967_296772

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem absolute_difference_100th_terms :
  let C := arithmetic_sequence 35 7
  let D := arithmetic_sequence 35 (-7)
  |C 100 - D 100| = 1386 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_100th_terms_l2967_296772


namespace NUMINAMATH_CALUDE_not_perfect_square_if_last_two_digits_odd_l2967_296717

-- Define a function to get the last two digits of an integer
def lastTwoDigits (n : ℤ) : ℤ × ℤ :=
  let d₁ := n % 10
  let d₂ := (n / 10) % 10
  (d₂, d₁)

-- Define a predicate for an integer being odd
def isOdd (n : ℤ) : Prop := n % 2 ≠ 0

-- Theorem statement
theorem not_perfect_square_if_last_two_digits_odd (n : ℤ) :
  let (d₂, d₁) := lastTwoDigits n
  isOdd d₂ ∧ isOdd d₁ → ¬∃ (m : ℤ), n = m ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_not_perfect_square_if_last_two_digits_odd_l2967_296717


namespace NUMINAMATH_CALUDE_derivative_f_at_4_l2967_296716

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt x

theorem derivative_f_at_4 : 
  deriv f 4 = -1/16 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_4_l2967_296716


namespace NUMINAMATH_CALUDE_hyundai_dodge_ratio_l2967_296761

theorem hyundai_dodge_ratio (total : ℕ) (dodge : ℕ) (kia : ℕ) (hyundai : ℕ) :
  total = 400 →
  dodge = total / 2 →
  kia = 100 →
  hyundai = total - dodge - kia →
  (hyundai : ℚ) / dodge = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_hyundai_dodge_ratio_l2967_296761


namespace NUMINAMATH_CALUDE_machinery_spending_l2967_296797

/-- Represents the financial breakdown of Kanul's spending --/
structure KanulSpending where
  total : ℝ
  rawMaterials : ℝ
  cash : ℝ
  machinery : ℝ

/-- Theorem stating the amount spent on machinery --/
theorem machinery_spending (k : KanulSpending) 
  (h1 : k.total = 1000)
  (h2 : k.rawMaterials = 500)
  (h3 : k.cash = 0.1 * k.total)
  (h4 : k.total = k.rawMaterials + k.cash + k.machinery) :
  k.machinery = 400 := by
  sorry

end NUMINAMATH_CALUDE_machinery_spending_l2967_296797


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l2967_296779

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 4 ∧ 
  (∀ (x y z : ℝ), (x^2 + 2*y^2 + z^2)^2 ≤ n*(x^4 + 3*y^4 + z^4)) ∧ 
  (∀ (m : ℕ), m < n → ∃ (x y z : ℝ), (x^2 + 2*y^2 + z^2)^2 > m*(x^4 + 3*y^4 + z^4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l2967_296779


namespace NUMINAMATH_CALUDE_greatest_integer_b_for_quadratic_range_l2967_296707

theorem greatest_integer_b_for_quadratic_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 5 > -25) ↔ b ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_b_for_quadratic_range_l2967_296707


namespace NUMINAMATH_CALUDE_sqrt_product_property_sqrt_40_in_terms_of_a_b_l2967_296775

theorem sqrt_product_property (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  Real.sqrt x * Real.sqrt y = Real.sqrt (x * y) := by sorry

theorem sqrt_40_in_terms_of_a_b (a b : ℝ) (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 10) :
  Real.sqrt 40 = Real.sqrt 2 * a * b := by sorry

end NUMINAMATH_CALUDE_sqrt_product_property_sqrt_40_in_terms_of_a_b_l2967_296775


namespace NUMINAMATH_CALUDE_fib_last_digit_periodic_l2967_296703

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Period of Fibonacci sequence modulo 10 -/
def fibPeriod : ℕ := 60

/-- Theorem: The last digit of Fibonacci numbers repeats with period 60 -/
theorem fib_last_digit_periodic (n : ℕ) : fib n % 10 = fib (n + fibPeriod) % 10 := by
  sorry

end NUMINAMATH_CALUDE_fib_last_digit_periodic_l2967_296703


namespace NUMINAMATH_CALUDE_ambulance_cost_calculation_l2967_296785

/-- Calculates the cost of an ambulance ride given hospital stay details and total bill --/
def ambulance_ride_cost (days : ℕ) (bed_cost_per_day : ℕ) (specialist_cost_per_hour : ℕ) 
  (specialist_time_minutes : ℕ) (num_specialists : ℕ) (total_bill : ℕ) : ℕ :=
  let bed_cost := days * bed_cost_per_day
  let specialist_cost := (specialist_cost_per_hour * specialist_time_minutes * num_specialists) / 60
  let known_costs := bed_cost + specialist_cost
  total_bill - known_costs

/-- Proves that the ambulance ride cost is $1675 given the specific conditions --/
theorem ambulance_cost_calculation : 
  ambulance_ride_cost 3 900 250 15 2 4625 = 1675 := by
  sorry

end NUMINAMATH_CALUDE_ambulance_cost_calculation_l2967_296785


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2967_296722

theorem arithmetic_calculations :
  (-1^2 + |(-3)| + 5 / (-5) = 1) ∧
  (2 * (-3)^2 + 24 * (1/4 - 3/8 - 1/12) = 4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2967_296722


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2967_296777

theorem trigonometric_identity : 
  (Real.sin (65 * π / 180) + Real.sin (15 * π / 180) * Real.sin (10 * π / 180)) / 
  (Real.sin (25 * π / 180) - Real.cos (15 * π / 180) * Real.cos (80 * π / 180)) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2967_296777


namespace NUMINAMATH_CALUDE_classroom_pencils_l2967_296735

theorem classroom_pencils (num_children : ℕ) (pencils_per_child : ℕ) 
  (h1 : num_children = 10) (h2 : pencils_per_child = 5) : 
  num_children * pencils_per_child = 50 := by
  sorry

end NUMINAMATH_CALUDE_classroom_pencils_l2967_296735


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2967_296786

/-- The inequality x^2 - 2ax + a > 0 has ℝ as its solution set -/
def has_real_solution_set (a : ℝ) : Prop :=
  ∀ x, x^2 - 2*a*x + a > 0

/-- 0 < a < 1 -/
def a_in_open_unit_interval (a : ℝ) : Prop :=
  0 < a ∧ a < 1

theorem sufficient_not_necessary :
  (∀ a : ℝ, has_real_solution_set a → a_in_open_unit_interval a) ∧
  (∃ a : ℝ, a_in_open_unit_interval a ∧ ¬has_real_solution_set a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2967_296786


namespace NUMINAMATH_CALUDE_equation_transformation_l2967_296700

theorem equation_transformation (x : ℝ) : 
  ((x - 1) / 2 - 1 = (3 * x + 1) / 3) ↔ (3 * (x - 1) - 6 = 2 * (3 * x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_transformation_l2967_296700


namespace NUMINAMATH_CALUDE_sum_of_squares_l2967_296773

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (cube_eq_seventh : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2967_296773


namespace NUMINAMATH_CALUDE_largest_N_is_120_l2967_296789

/-- A type representing a 6 × N table with entries from 1 to 6 -/
def Table (N : ℕ) := Fin 6 → Fin N → Fin 6

/-- Predicate to check if a column is a permutation of 1 to 6 -/
def IsPermutation (t : Table N) (col : Fin N) : Prop :=
  ∀ i : Fin 6, ∃ j : Fin 6, t j col = i

/-- Predicate to check if any two columns have a common entry in some row -/
def HasCommonEntry (t : Table N) : Prop :=
  ∀ i j : Fin N, i ≠ j → ∃ r : Fin 6, t r i = t r j

/-- Predicate to check if any two columns have a different entry in some row -/
def HasDifferentEntry (t : Table N) : Prop :=
  ∀ i j : Fin N, i ≠ j → ∃ s : Fin 6, t s i ≠ t s j

/-- The main theorem stating the largest possible N -/
theorem largest_N_is_120 :
  (∃ N : ℕ, N > 0 ∧ ∃ t : Table N,
    (∀ col, IsPermutation t col) ∧
    HasCommonEntry t ∧
    HasDifferentEntry t) ∧
  (∀ M : ℕ, M > 120 →
    ¬∃ t : Table M,
      (∀ col, IsPermutation t col) ∧
      HasCommonEntry t ∧
      HasDifferentEntry t) :=
sorry

end NUMINAMATH_CALUDE_largest_N_is_120_l2967_296789


namespace NUMINAMATH_CALUDE_capital_city_free_after_year_l2967_296713

/-- Represents the state of a city (under spell or not) -/
inductive SpellState
| Free
| UnderSpell

/-- Represents the kingdom with 12 cities -/
structure Kingdom where
  cities : Fin 12 → SpellState

/-- Represents the magician's action on a city -/
def magicianAction (s : SpellState) : SpellState :=
  match s with
  | SpellState.Free => SpellState.UnderSpell
  | SpellState.UnderSpell => SpellState.Free

/-- Applies the magician's transformation to the kingdom -/
def monthlyTransformation (k : Kingdom) (startCity : Fin 12) : Kingdom :=
  { cities := λ i => 
      if i.val < startCity.val 
      then k.cities i 
      else magicianAction (k.cities i) }

/-- The state of the kingdom after 12 months -/
def afterTwelveMonths (k : Kingdom) : Kingdom :=
  (List.range 12).foldl (λ acc i => monthlyTransformation acc i) k

/-- The theorem to be proved -/
theorem capital_city_free_after_year (k : Kingdom) (capitalCity : Fin 12) :
  k.cities capitalCity = SpellState.Free →
  (afterTwelveMonths k).cities capitalCity = SpellState.Free :=
sorry

end NUMINAMATH_CALUDE_capital_city_free_after_year_l2967_296713


namespace NUMINAMATH_CALUDE_solution_product_l2967_296758

theorem solution_product (p q : ℝ) : 
  (p - 6) * (3 * p + 8) = p^2 - 15 * p + 54 →
  (q - 6) * (3 * q + 8) = q^2 - 15 * q + 54 →
  p ≠ q →
  (p + 4) * (q + 4) = 130 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l2967_296758
