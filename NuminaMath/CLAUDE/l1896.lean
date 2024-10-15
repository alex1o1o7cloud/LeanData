import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_existence_l1896_189655

/-- The system of equations has at least one solution for some 'a' if and only if 'b' is in [-11, 2) -/
theorem system_solution_existence (b : ℝ) : 
  (∃ a x y : ℝ, x^2 + y^2 + 2*b*(b - x + y) = 4 ∧ y = 9 / ((x + a)^2 + 1)) ↔ 
  -11 ≤ b ∧ b < 2 := by sorry

end NUMINAMATH_CALUDE_system_solution_existence_l1896_189655


namespace NUMINAMATH_CALUDE_intersection_condition_l1896_189634

open Set Real

def A : Set ℝ := {x | (x - 1) / (x + 1) < 0}
def B (b : ℝ) : Set ℝ := {x | (x - b)^2 < 1}

theorem intersection_condition (b : ℝ) : A ∩ B b ≠ ∅ ↔ -2 < b ∧ b < 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l1896_189634


namespace NUMINAMATH_CALUDE_rational_sequence_to_integer_l1896_189667

theorem rational_sequence_to_integer (x : ℚ) : 
  ∃ (f : ℕ → ℚ), 
    f 0 = x ∧ 
    (∀ n : ℕ, n ≥ 1 → (f n = 2 * f (n - 1) ∨ f n = 2 * f (n - 1) + 1 / n)) ∧
    (∃ k : ℕ, ∃ m : ℤ, f k = m) := by
  sorry

end NUMINAMATH_CALUDE_rational_sequence_to_integer_l1896_189667


namespace NUMINAMATH_CALUDE_egg_laying_hens_l1896_189631

theorem egg_laying_hens (total_chickens roosters non_laying_hens : ℕ) 
  (h1 : total_chickens = 325)
  (h2 : roosters = 28)
  (h3 : non_laying_hens = 20) :
  total_chickens - roosters - non_laying_hens = 277 := by
  sorry

#check egg_laying_hens

end NUMINAMATH_CALUDE_egg_laying_hens_l1896_189631


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1896_189673

theorem trigonometric_identities (α : Real) (h : Real.tan α = 3) :
  ((Real.sin α + 3 * Real.cos α) / (2 * Real.sin α + 5 * Real.cos α) = 6/11) ∧
  (Real.sin α ^ 2 + Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 6) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1896_189673


namespace NUMINAMATH_CALUDE_polygon_side_theorem_l1896_189690

def polygon_side_proof (total_area : ℝ) (rect1_length rect1_width : ℝ) 
  (rect2_length rect2_width : ℝ) (unknown_side_min unknown_side_max : ℝ) : Prop :=
  let rect1_area := rect1_length * rect1_width
  let rect2_area := rect2_length * rect2_width
  let unknown_rect_area := total_area - rect1_area - rect2_area
  ∃ (x : ℝ), (x = 7 ∨ x = 6) ∧ 
             unknown_rect_area = x * (unknown_rect_area / x) ∧
             x > unknown_side_min ∧ x < unknown_side_max

theorem polygon_side_theorem : 
  polygon_side_proof 72 10 1 5 4 5 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_side_theorem_l1896_189690


namespace NUMINAMATH_CALUDE_password_probability_l1896_189696

/-- The set of possible first characters in the password -/
def first_char : Finset Char := {'M', 'I', 'N'}

/-- The set of possible second characters in the password -/
def second_char : Finset Char := {'1', '2', '3', '4', '5'}

/-- The type representing a two-character password -/
def Password := Char × Char

/-- The set of all possible passwords -/
def all_passwords : Finset Password :=
  first_char.product second_char

theorem password_probability :
  (Finset.card all_passwords : ℚ) = 15 ∧
  (1 : ℚ) / (Finset.card all_passwords : ℚ) = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_password_probability_l1896_189696


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_18_l1896_189600

def is_four_digit (n : ℕ) : Prop := n ≥ 1000 ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

theorem largest_four_digit_sum_18 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 18 → n ≤ 9720 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_18_l1896_189600


namespace NUMINAMATH_CALUDE_min_value_of_2a_plus_b_l1896_189616

/-- Given a line x/a + y/b = 1 where a > 0 and b > 0, and the line passes through the point (1, 2),
    the minimum value of 2a + b is 8. -/
theorem min_value_of_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : ∀ x y : ℝ, x / a + y / b = 1 → x = 1 ∧ y = 2) :
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∀ x y : ℝ, x / a' + y / b' = 1 → x = 1 ∧ y = 2) → 
    2 * a' + b' ≥ 2 * a + b) ∧
  2 * a + b = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_2a_plus_b_l1896_189616


namespace NUMINAMATH_CALUDE_john_rejection_rate_proof_l1896_189611

/-- The percentage of products Jane rejected -/
def jane_rejection_rate : ℝ := 0.7

/-- The total percentage of products rejected -/
def total_rejection_rate : ℝ := 0.75

/-- The ratio of products Jane inspected compared to John -/
def jane_inspection_ratio : ℝ := 1.25

/-- John's rejection rate -/
def john_rejection_rate : ℝ := 0.8125

theorem john_rejection_rate_proof :
  let total_products := 1 + jane_inspection_ratio
  jane_rejection_rate * jane_inspection_ratio + john_rejection_rate = total_rejection_rate * total_products :=
by sorry

end NUMINAMATH_CALUDE_john_rejection_rate_proof_l1896_189611


namespace NUMINAMATH_CALUDE_comparison_of_expressions_l1896_189683

theorem comparison_of_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (∃ a b, (a + 1/a) * (b + 1/b) > (Real.sqrt (a*b) + 1/Real.sqrt (a*b))^2) ∧
  (∃ a b, (a + 1/a) * (b + 1/b) > ((a+b)/2 + 2/(a+b))^2) ∧
  (∃ a b, ((a+b)/2 + 2/(a+b))^2 > (a + 1/a) * (b + 1/b)) :=
by sorry

end NUMINAMATH_CALUDE_comparison_of_expressions_l1896_189683


namespace NUMINAMATH_CALUDE_princess_daphne_jewelry_cost_l1896_189691

/-- The cost of Princess Daphne's jewelry purchase -/
def total_cost : ℕ := 240000

/-- The cost of a single necklace -/
def necklace_cost : ℕ := 40000

/-- The cost of the earrings -/
def earrings_cost : ℕ := 3 * necklace_cost

theorem princess_daphne_jewelry_cost :
  3 * necklace_cost + earrings_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_princess_daphne_jewelry_cost_l1896_189691


namespace NUMINAMATH_CALUDE_remainder_x_plus_one_2025_mod_x_squared_plus_one_l1896_189609

theorem remainder_x_plus_one_2025_mod_x_squared_plus_one (x : ℤ) :
  (x + 1) ^ 2025 % (x^2 + 1) = 0 := by
sorry

end NUMINAMATH_CALUDE_remainder_x_plus_one_2025_mod_x_squared_plus_one_l1896_189609


namespace NUMINAMATH_CALUDE_mean_age_of_friends_l1896_189647

theorem mean_age_of_friends (age_group1 : ℕ) (age_group2 : ℕ) 
  (h1 : age_group1 = 12 * 12 + 3)  -- 12 years and 3 months in months
  (h2 : age_group2 = 13 * 12 + 5)  -- 13 years and 5 months in months
  : (3 * age_group1 + 4 * age_group2) / 7 = 155 := by
  sorry

end NUMINAMATH_CALUDE_mean_age_of_friends_l1896_189647


namespace NUMINAMATH_CALUDE_ideal_gas_pressure_change_l1896_189669

/-- Given an ideal gas at constant temperature, calculate the new pressure when the volume changes. -/
theorem ideal_gas_pressure_change (V1 V2 P1 P2 : ℝ) (hV1 : V1 = 4.56) (hV2 : V2 = 2.28) (hP1 : P1 = 10) :
  V1 * P1 = V2 * P2 → P2 = 20 := by
  sorry

#check ideal_gas_pressure_change

end NUMINAMATH_CALUDE_ideal_gas_pressure_change_l1896_189669


namespace NUMINAMATH_CALUDE_rabbit_count_l1896_189664

/-- Calculates the number of rabbits given land dimensions and clearing rates -/
theorem rabbit_count (land_width : ℝ) (land_length : ℝ) (rabbit_clear_rate : ℝ) (days_to_clear : ℝ) : 
  land_width = 200 ∧ 
  land_length = 900 ∧ 
  rabbit_clear_rate = 10 ∧ 
  days_to_clear = 20 → 
  (land_width * land_length) / 9 / (rabbit_clear_rate * days_to_clear) = 100 := by
  sorry

#check rabbit_count

end NUMINAMATH_CALUDE_rabbit_count_l1896_189664


namespace NUMINAMATH_CALUDE_max_product_distances_l1896_189684

/-- Given two perpendicular lines passing through fixed points A and B,
    prove that the maximum value of the product of distances from their
    intersection point P to A and B is |AB|²/2 -/
theorem max_product_distances (m : ℝ) : ∃ (P : ℝ × ℝ),
  (P.1 + m * P.2 = 0) ∧
  (m * P.1 - P.2 - m + 3 = 0) →
  ∀ (Q : ℝ × ℝ),
    (Q.1 + m * Q.2 = 0) ∧
    (m * Q.1 - Q.2 - m + 3 = 0) →
    (Q.1 - 0)^2 + (Q.2 - 0)^2 * ((Q.1 - 1)^2 + (Q.2 - 3)^2) ≤ 25 :=
sorry

end NUMINAMATH_CALUDE_max_product_distances_l1896_189684


namespace NUMINAMATH_CALUDE_students_on_field_trip_l1896_189628

/-- The number of seats on each school bus -/
def seats_per_bus : ℕ := 10

/-- The number of buses needed for the trip -/
def number_of_buses : ℕ := 6

/-- Theorem stating the number of students going on the field trip -/
theorem students_on_field_trip : seats_per_bus * number_of_buses = 60 := by
  sorry

end NUMINAMATH_CALUDE_students_on_field_trip_l1896_189628


namespace NUMINAMATH_CALUDE_complex_number_quadrant_z_in_second_quadrant_l1896_189618

theorem complex_number_quadrant : Complex → Prop :=
  fun z => ∃ (a b : ℝ), z = Complex.mk a b ∧ a < 0 ∧ b > 0

def i : Complex := Complex.I

def z : Complex := (1 + 2 * i) * i

theorem z_in_second_quadrant : complex_number_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_z_in_second_quadrant_l1896_189618


namespace NUMINAMATH_CALUDE_gcd_153_119_l1896_189638

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_153_119_l1896_189638


namespace NUMINAMATH_CALUDE_function_properties_l1896_189639

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

theorem function_properties (a : ℝ) :
  (∀ x : ℝ, x > 0 → f a x = a * x^2 - (a + 2) * x + Real.log x) →
  (a = 1 → ∀ x : ℝ, x > 0 → (f 1 x - f 1 1) = 0 * (x - 1)) ∧
  (a > 0 → (∀ x : ℝ, 1 ≤ x → x ≤ Real.exp 1 → f a x ≥ -2) → (∀ x : ℝ, 1 ≤ x → x ≤ Real.exp 1 → f a x = -2) → a ≥ 1) ∧
  ((∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (f a x₁ + 2 * x₁ - (f a x₂ + 2 * x₂)) / (x₁ - x₂) > 0) → 0 ≤ a ∧ a ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1896_189639


namespace NUMINAMATH_CALUDE_max_value_quadratic_inequality_l1896_189672

theorem max_value_quadratic_inequality :
  let f : ℝ → ℝ := λ x => -2 * x^2 + 9 * x - 7
  ∃ (max_x : ℝ), max_x = 3.5 ∧
    (∀ x : ℝ, f x ≤ 0 → x ≤ max_x) ∧
    f max_x ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_inequality_l1896_189672


namespace NUMINAMATH_CALUDE_frank_to_betty_bill_ratio_l1896_189699

/-- The number of oranges Betty picked -/
def betty_oranges : ℕ := 15

/-- The number of oranges Bill picked -/
def bill_oranges : ℕ := 12

/-- The number of seeds Frank planted from each orange -/
def seeds_per_orange : ℕ := 2

/-- The number of oranges on each tree -/
def oranges_per_tree : ℕ := 5

/-- The total number of oranges Philip can pick -/
def philip_oranges : ℕ := 810

/-- Theorem stating the ratio of Frank's oranges to Betty and Bill's combined oranges -/
theorem frank_to_betty_bill_ratio :
  ∃ (frank_oranges : ℕ),
    frank_oranges > 0 ∧
    philip_oranges = frank_oranges * seeds_per_orange * oranges_per_tree ∧
    frank_oranges = 3 * (betty_oranges + bill_oranges) := by
  sorry

end NUMINAMATH_CALUDE_frank_to_betty_bill_ratio_l1896_189699


namespace NUMINAMATH_CALUDE_transformation_result_l1896_189623

/-- Rotation of 180 degrees counterclockwise around a point -/
def rotate180 (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - point.1, 2 * center.2 - point.2)

/-- Reflection about the line y = -x -/
def reflectAboutNegativeXEqualsY (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.2, point.1)

/-- The main theorem -/
theorem transformation_result (a b : ℝ) :
  let P : ℝ × ℝ := (a, b)
  let center : ℝ × ℝ := (1, 5)
  let transformed := reflectAboutNegativeXEqualsY (rotate180 center P)
  transformed = (7, -3) → b - a = -2 := by
  sorry

end NUMINAMATH_CALUDE_transformation_result_l1896_189623


namespace NUMINAMATH_CALUDE_two_number_problem_l1896_189697

theorem two_number_problem (A B n : ℕ) : 
  B > 0 → 
  A > B → 
  A = 10 * B + n → 
  0 ≤ n → 
  n ≤ 9 → 
  A + B = 2022 → 
  A = 1839 ∧ B = 183 := by
sorry

end NUMINAMATH_CALUDE_two_number_problem_l1896_189697


namespace NUMINAMATH_CALUDE_square_side_lengths_average_l1896_189622

theorem square_side_lengths_average (a b c : ℝ) (ha : a = 25) (hb : b = 64) (hc : c = 144) :
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_lengths_average_l1896_189622


namespace NUMINAMATH_CALUDE_trader_profit_l1896_189661

theorem trader_profit (C : ℝ) (C_pos : C > 0) : 
  let markup := 0.12
  let discount := 0.09821428571428571
  let marked_price := C * (1 + markup)
  let final_price := marked_price * (1 - discount)
  (final_price - C) / C = 0.01 := by
sorry

end NUMINAMATH_CALUDE_trader_profit_l1896_189661


namespace NUMINAMATH_CALUDE_unique_gcd_triplet_l1896_189666

theorem unique_gcd_triplet :
  ∃! (x y z : ℕ),
    (∃ (a b c : ℕ), x = Nat.gcd a b ∧ y = Nat.gcd b c ∧ z = Nat.gcd c a) ∧
    x ∈ ({6, 8, 12, 18, 24} : Set ℕ) ∧
    y ∈ ({14, 20, 28, 44, 56} : Set ℕ) ∧
    z ∈ ({5, 15, 18, 27, 42} : Set ℕ) ∧
    x = 8 ∧ y = 14 ∧ z = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_gcd_triplet_l1896_189666


namespace NUMINAMATH_CALUDE_yura_catches_lena_l1896_189610

/-- The time it takes for Yura to catch up with Lena -/
def catchUpTime : ℝ := 5

/-- Lena's walking speed -/
def lenaSpeed : ℝ := 1

/-- The time difference between Lena and Yura's start -/
def timeDifference : ℝ := 5

theorem yura_catches_lena :
  ∀ (t : ℝ),
  t = catchUpTime →
  (lenaSpeed * (t + timeDifference)) = (2 * lenaSpeed * t) :=
by sorry

end NUMINAMATH_CALUDE_yura_catches_lena_l1896_189610


namespace NUMINAMATH_CALUDE_strawberry_bucket_problem_l1896_189630

/-- Proves that the number of buckets used is 5 given the conditions of the strawberry problem -/
theorem strawberry_bucket_problem (total_strawberries : ℕ) (removed_per_bucket : ℕ) (remaining_per_bucket : ℕ) 
  (h1 : total_strawberries = 300)
  (h2 : removed_per_bucket = 20)
  (h3 : remaining_per_bucket = 40) :
  (total_strawberries / (removed_per_bucket + remaining_per_bucket) : ℕ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_bucket_problem_l1896_189630


namespace NUMINAMATH_CALUDE_certain_number_proof_l1896_189619

theorem certain_number_proof (h : 2994 / 14.5 = 171) : 
  ∃ x : ℝ, x / 1.45 = 17.1 ∧ x = 24.795 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1896_189619


namespace NUMINAMATH_CALUDE_function_inequality_l1896_189677

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -2 * Real.log x + a / x^2 + 1

theorem function_inequality (a : ℝ) (x₁ x₂ x₀ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₀ : x₀ > 0)
  (hz₁ : f a x₁ = 0) (hz₂ : f a x₂ = 0) (hx₁₂ : x₁ ≠ x₂)
  (hextremum : ∀ x > 0, f a x₀ ≥ f a x) :
  1 / x₁^2 + 1 / x₂^2 > 2 * f a x₀ :=
sorry

end NUMINAMATH_CALUDE_function_inequality_l1896_189677


namespace NUMINAMATH_CALUDE_remainder_of_sum_divided_by_256_l1896_189653

theorem remainder_of_sum_divided_by_256 :
  (1234567 + 890123) % 256 = 74 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_divided_by_256_l1896_189653


namespace NUMINAMATH_CALUDE_dunbar_bouquets_l1896_189660

/-- The number of table decorations needed --/
def num_table_decorations : ℕ := 7

/-- The number of white roses used in each table decoration --/
def roses_per_table_decoration : ℕ := 12

/-- The number of white roses used in each bouquet --/
def roses_per_bouquet : ℕ := 5

/-- The total number of white roses needed for all bouquets and table decorations --/
def total_roses : ℕ := 109

/-- The number of bouquets Mrs. Dunbar needs to make --/
def num_bouquets : ℕ := (total_roses - num_table_decorations * roses_per_table_decoration) / roses_per_bouquet

theorem dunbar_bouquets : num_bouquets = 5 := by
  sorry

end NUMINAMATH_CALUDE_dunbar_bouquets_l1896_189660


namespace NUMINAMATH_CALUDE_tommys_profit_l1896_189643

/-- Represents a type of crate --/
structure Crate where
  capacity : ℕ
  quantity : ℕ
  cost : ℕ
  rotten : ℕ
  price : ℕ

/-- Calculates the profit from selling tomatoes --/
def calculateProfit (crateA crateB crateC : Crate) : ℕ :=
  let totalCost := crateA.cost + crateB.cost + crateC.cost
  let revenueA := (crateA.capacity * crateA.quantity - crateA.rotten) * crateA.price
  let revenueB := (crateB.capacity * crateB.quantity - crateB.rotten) * crateB.price
  let revenueC := (crateC.capacity * crateC.quantity - crateC.rotten) * crateC.price
  let totalRevenue := revenueA + revenueB + revenueC
  totalRevenue - totalCost

/-- Tommy's profit from selling tomatoes is $14 --/
theorem tommys_profit :
  let crateA : Crate := ⟨20, 2, 220, 4, 5⟩
  let crateB : Crate := ⟨25, 3, 375, 5, 6⟩
  let crateC : Crate := ⟨30, 1, 180, 3, 7⟩
  calculateProfit crateA crateB crateC = 14 := by
  sorry


end NUMINAMATH_CALUDE_tommys_profit_l1896_189643


namespace NUMINAMATH_CALUDE_tuesday_spending_multiple_l1896_189680

/-- Represents the spending on sneakers over three days -/
structure SneakerSpending where
  monday : ℕ
  tuesday_multiple : ℕ
  wednesday_multiple : ℕ
  total : ℕ

/-- The spending satisfies the given conditions -/
def valid_spending (s : SneakerSpending) : Prop :=
  s.monday = 60 ∧
  s.wednesday_multiple = 5 ∧
  s.total = 600 ∧
  s.monday + s.monday * s.tuesday_multiple + s.monday * s.wednesday_multiple = s.total

/-- The theorem to be proved -/
theorem tuesday_spending_multiple (s : SneakerSpending) :
  valid_spending s → s.tuesday_multiple = 4 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_spending_multiple_l1896_189680


namespace NUMINAMATH_CALUDE_rectangle_area_l1896_189608

/-- Represents a square with a given side length -/
structure Square where
  side : ℝ
  area : ℝ := side * side

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ
  area : ℝ := width * height

/-- The problem statement -/
theorem rectangle_area (s1 s2 s3 s4 : Square) (r : Rectangle) :
  s1.area = 4 ∧ s2.area = 4 ∧ s3.area = 1 ∧ s4.side = 2 * s3.side ∧
  r.width = s1.side + s4.side ∧ r.height = s1.side + s3.side →
  r.area = 12 := by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_rectangle_area_l1896_189608


namespace NUMINAMATH_CALUDE_tank_depth_l1896_189688

theorem tank_depth (length width : ℝ) (cost_per_sqm total_cost : ℝ) (d : ℝ) :
  length = 25 →
  width = 12 →
  cost_per_sqm = 0.3 →
  total_cost = 223.2 →
  cost_per_sqm * (length * width + 2 * (length * d) + 2 * (width * d)) = total_cost →
  d = 6 := by
sorry

end NUMINAMATH_CALUDE_tank_depth_l1896_189688


namespace NUMINAMATH_CALUDE_polygon_area_l1896_189698

-- Define a point in 2D space
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define the polygon
def polygon : List Point := [
  ⟨0, 0⟩, ⟨12, 0⟩, ⟨24, 12⟩, ⟨24, 0⟩, ⟨36, 0⟩,
  ⟨36, 24⟩, ⟨24, 36⟩, ⟨12, 36⟩, ⟨0, 36⟩, ⟨0, 24⟩
]

-- Function to calculate the area of the polygon
def calculateArea (vertices : List Point) : ℤ :=
  sorry

-- Theorem stating that the area of the polygon is 1008 square units
theorem polygon_area : calculateArea polygon = 1008 :=
  sorry

end NUMINAMATH_CALUDE_polygon_area_l1896_189698


namespace NUMINAMATH_CALUDE_product_of_roots_l1896_189658

theorem product_of_roots (x : ℝ) : 
  (∃ r₁ r₂ r₃ r₄ : ℝ, x^4 - 12*x^3 + 50*x^2 + 48*x - 35 = (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄)) →
  (∃ r₁ r₂ r₃ r₄ : ℝ, x^4 - 12*x^3 + 50*x^2 + 48*x - 35 = (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄) ∧
                       r₁ * r₂ * r₃ * r₄ = 35) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l1896_189658


namespace NUMINAMATH_CALUDE_expression_evaluation_l1896_189604

theorem expression_evaluation : |-3| - 2 * Real.tan (π / 3) + (1 / 2)⁻¹ + Real.sqrt 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1896_189604


namespace NUMINAMATH_CALUDE_second_guldin_theorem_l1896_189657

/-- A plate with an arbitrary boundary -/
structure ArbitraryPlate where
  area : ℝ
  centerOfMass : ℝ × ℝ × ℝ

/-- The volume generated by rotating an arbitrary plate around an axis -/
def rotationVolume (plate : ArbitraryPlate) (axisDistance : ℝ) : ℝ := sorry

/-- The second Guldin's theorem for an arbitrary plate -/
theorem second_guldin_theorem (plate : ArbitraryPlate) (axisDistance : ℝ) :
  rotationVolume plate axisDistance = 2 * Real.pi * plate.area * axisDistance := by
  sorry

end NUMINAMATH_CALUDE_second_guldin_theorem_l1896_189657


namespace NUMINAMATH_CALUDE_not_possible_N_l1896_189602

-- Define the set M
def M : Set ℝ := {x | x^2 - 6*x - 16 < 0}

-- Define the theorem
theorem not_possible_N (N : Set ℝ) (h1 : M ∩ N = N) : N ≠ Set.Icc (-1 : ℝ) 8 := by
  sorry

end NUMINAMATH_CALUDE_not_possible_N_l1896_189602


namespace NUMINAMATH_CALUDE_triangle_ratio_proof_l1896_189607

theorem triangle_ratio_proof (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  A = π / 3 ∧
  b = 1 ∧
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 →
  (a + 2 * b - 3 * c) / (Real.sin A + 2 * Real.sin B - 3 * Real.sin C) = 2 * Real.sqrt 39 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ratio_proof_l1896_189607


namespace NUMINAMATH_CALUDE_supplement_of_complement_65_l1896_189652

def complement (α : ℝ) : ℝ := 90 - α

def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_65 :
  supplement (complement 65) = 155 := by sorry

end NUMINAMATH_CALUDE_supplement_of_complement_65_l1896_189652


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1896_189627

def complex_number : ℂ := Complex.I + Complex.I^2

theorem complex_number_in_second_quadrant :
  complex_number.re < 0 ∧ complex_number.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1896_189627


namespace NUMINAMATH_CALUDE_train_length_l1896_189674

/-- Calculates the length of a train given its speed and the time and distance it takes to cross a bridge -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 57.6 * (1000 / 3600) →
  bridge_length = 150 →
  crossing_time = 25 →
  (train_speed * crossing_time) - bridge_length = 250 := by
sorry

end NUMINAMATH_CALUDE_train_length_l1896_189674


namespace NUMINAMATH_CALUDE_reverse_clock_theorem_l1896_189645

/-- Represents a clock with a reverse-moving minute hand -/
structure ReverseClock :=
  (hour : ℝ)
  (minute : ℝ)

/-- Converts a ReverseClock time to a standard clock time -/
def to_standard_time (c : ReverseClock) : ℝ := sorry

/-- Checks if the hands of a ReverseClock coincide -/
def hands_coincide (c : ReverseClock) : Prop := sorry

theorem reverse_clock_theorem :
  ∀ (c : ReverseClock),
    4 < c.hour ∧ c.hour < 5 →
    hands_coincide c →
    to_standard_time c = 4 + 36 / 60 + 12 / (13 * 60) :=
by sorry

end NUMINAMATH_CALUDE_reverse_clock_theorem_l1896_189645


namespace NUMINAMATH_CALUDE_cos_cubed_minus_sin_cubed_l1896_189676

theorem cos_cubed_minus_sin_cubed (θ : ℝ) :
  Real.cos θ ^ 3 - Real.sin θ ^ 3 = (Real.cos θ - Real.sin θ) * (1 + Real.cos θ * Real.sin θ) := by
  sorry

end NUMINAMATH_CALUDE_cos_cubed_minus_sin_cubed_l1896_189676


namespace NUMINAMATH_CALUDE_average_equation_solution_l1896_189662

theorem average_equation_solution (x : ℝ) : 
  (1/3 : ℝ) * ((2*x + 8) + (5*x + 3) + (3*x + 9)) = 5*x - 10 → x = 10 := by
sorry

end NUMINAMATH_CALUDE_average_equation_solution_l1896_189662


namespace NUMINAMATH_CALUDE_rectangle_construction_l1896_189651

/-- Given a length b and a sum s, prove the existence of a rectangle with side lengths a and b,
    such that s equals the sum of the diagonal and side b. -/
theorem rectangle_construction (b : ℝ) (s : ℝ) (h_pos : b > 0 ∧ s > b) :
  ∃ (a : ℝ), a > 0 ∧ s = a + (a^2 + b^2).sqrt := by
  sorry

#check rectangle_construction

end NUMINAMATH_CALUDE_rectangle_construction_l1896_189651


namespace NUMINAMATH_CALUDE_point_b_coordinates_l1896_189671

/-- Given a circle with center (0,0) and radius 2, points A(2,2) and B(a,b),
    if for any point P on the circle, |PA|/|PB| = √2, then B = (1,1) -/
theorem point_b_coordinates (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4 → 
    ((x - 2)^2 + (y - 2)^2) / ((x - a)^2 + (y - b)^2) = 2) → 
  a = 1 ∧ b = 1 := by sorry

end NUMINAMATH_CALUDE_point_b_coordinates_l1896_189671


namespace NUMINAMATH_CALUDE_divisor_problem_l1896_189620

theorem divisor_problem (initial_number : ℕ) (added_number : ℕ) (divisor : ℕ) : 
  initial_number = 8679921 →
  added_number = 72 →
  divisor = 69 →
  (initial_number + added_number) % divisor = 0 :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l1896_189620


namespace NUMINAMATH_CALUDE_one_and_one_third_problem_l1896_189621

theorem one_and_one_third_problem :
  ∀ x : ℝ, (4/3 : ℝ) * x = 45 ↔ x = 33.75 := by sorry

end NUMINAMATH_CALUDE_one_and_one_third_problem_l1896_189621


namespace NUMINAMATH_CALUDE_ten_dollar_combinations_l1896_189640

def coin_combinations (target : ℕ) (coins : List ℕ) : ℕ :=
  sorry

theorem ten_dollar_combinations :
  coin_combinations 10 [1, 2, 5] = 10 :=
sorry

end NUMINAMATH_CALUDE_ten_dollar_combinations_l1896_189640


namespace NUMINAMATH_CALUDE_certain_number_is_three_l1896_189648

theorem certain_number_is_three :
  ∀ certain_number : ℕ,
  (2^14 : ℕ) - (2^12 : ℕ) = certain_number * (2^12 : ℕ) →
  certain_number = 3 := by
sorry

end NUMINAMATH_CALUDE_certain_number_is_three_l1896_189648


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l1896_189693

theorem polynomial_value_theorem (m n : ℝ) 
  (h1 : 2*m + n + 2 = m + 2*n) 
  (h2 : m - n + 2 ≠ 0) : 
  let x := 3*(m + n + 1)
  (x^2 + 4*x + 6 : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l1896_189693


namespace NUMINAMATH_CALUDE_town_distance_proof_l1896_189615

/-- The distance between two towns on a map in inches -/
def map_distance : ℝ := 18

/-- The scale of the map in inches -/
def scale_inches : ℝ := 0.5

/-- The scale of the map in miles -/
def scale_miles : ℝ := 6

/-- The actual distance between the towns in miles -/
def actual_distance : ℝ := 216

theorem town_distance_proof :
  (map_distance * scale_miles) / scale_inches = actual_distance := by
  sorry

end NUMINAMATH_CALUDE_town_distance_proof_l1896_189615


namespace NUMINAMATH_CALUDE_product_of_positive_reals_l1896_189603

theorem product_of_positive_reals (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h1 : r^2 + s^2 = 2) (h2 : r^4 + s^4 = 15/8) : r * s = Real.sqrt 17 / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_positive_reals_l1896_189603


namespace NUMINAMATH_CALUDE_specific_convention_handshakes_l1896_189650

/-- The number of handshakes in a convention with multiple companies -/
def convention_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_participants := num_companies * reps_per_company
  let handshakes_per_person := total_participants - reps_per_company
  (total_participants * handshakes_per_person) / 2

/-- Theorem stating the number of handshakes in the specific convention scenario -/
theorem specific_convention_handshakes :
  convention_handshakes 5 5 = 250 := by
  sorry

#eval convention_handshakes 5 5

end NUMINAMATH_CALUDE_specific_convention_handshakes_l1896_189650


namespace NUMINAMATH_CALUDE_gold_bar_worth_l1896_189626

/-- Proves that the worth of each gold bar is $20,000 given the specified conditions -/
theorem gold_bar_worth (rows : ℕ) (bars_per_row : ℕ) (total_worth : ℕ) : ℕ :=
  by
  -- Define the given conditions
  have h1 : rows = 4 := by sorry
  have h2 : bars_per_row = 20 := by sorry
  have h3 : total_worth = 1600000 := by sorry

  -- Calculate the total number of gold bars
  let total_bars := rows * bars_per_row

  -- Calculate the worth of each gold bar
  let bar_worth := total_worth / total_bars

  -- Prove that bar_worth equals 20000
  sorry

-- The theorem statement
#check gold_bar_worth

end NUMINAMATH_CALUDE_gold_bar_worth_l1896_189626


namespace NUMINAMATH_CALUDE_tan_ratio_theorem_l1896_189617

theorem tan_ratio_theorem (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 8) :
  (Real.tan x * Real.tan y) / (Real.tan x / Real.tan y + Real.tan y / Real.tan x) = 31 / 13 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_theorem_l1896_189617


namespace NUMINAMATH_CALUDE_count_odd_sum_numbers_l1896_189663

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- A function that checks if a number is odd -/
def isOdd (n : Nat) : Bool := n % 2 = 1

/-- A function that returns the sum of digits of a three-digit number -/
def digitSum (n : Nat) : Nat :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

/-- The set of all three-digit numbers formed by the given digits without repetition -/
def threeDigitNumbers : Finset Nat :=
  Finset.filter (fun n => n ≥ 100 ∧ n < 1000 ∧ (Finset.card (Finset.filter (fun d => d ∈ digits) (Finset.range 10))) = 3) (Finset.range 1000)

theorem count_odd_sum_numbers :
  Finset.card (Finset.filter (fun n => isOdd (digitSum n)) threeDigitNumbers) = 24 := by sorry

end NUMINAMATH_CALUDE_count_odd_sum_numbers_l1896_189663


namespace NUMINAMATH_CALUDE_equation_solution_l1896_189636

theorem equation_solution (k : ℝ) : 
  ((-2 : ℝ)^2 + 4*k*(-2) + 2*k^2 = 4) → (k = 0 ∨ k = 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1896_189636


namespace NUMINAMATH_CALUDE_least_perimeter_triangle_l1896_189659

theorem least_perimeter_triangle (a b c : ℕ) : 
  a = 45 → b = 53 → c > 0 → 
  (a + b > c) → (a + c > b) → (b + c > a) →
  ∀ x : ℕ, (x > 0 ∧ (a + x > b) ∧ (b + x > a) ∧ (a + b > x)) → (a + b + c ≤ a + b + x) →
  a + b + c = 107 := by
sorry

end NUMINAMATH_CALUDE_least_perimeter_triangle_l1896_189659


namespace NUMINAMATH_CALUDE_expression_evaluation_l1896_189685

theorem expression_evaluation :
  (2 * Real.sqrt 2 - Real.pi) ^ 0 - 4 * Real.cos (60 * π / 180) + |Real.sqrt 2 - 2| - Real.sqrt 18 = 1 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1896_189685


namespace NUMINAMATH_CALUDE_collectible_toy_price_changes_l1896_189654

/-- Represents the months of the year --/
inductive Month
  | january
  | february
  | march
  | april
  | may
  | june

/-- The price change for each month --/
def price_change : Month → ℝ
  | Month.january => -1.00
  | Month.february => 3.50
  | Month.march => -3.00
  | Month.april => 4.50
  | Month.may => -1.50
  | Month.june => -3.50

/-- The month with the greatest price drop --/
def greatest_drop : Month := Month.june

/-- The month with the greatest price increase --/
def greatest_increase : Month := Month.april

theorem collectible_toy_price_changes :
  (∀ m : Month, price_change greatest_drop ≤ price_change m) ∧
  (∀ m : Month, price_change m ≤ price_change greatest_increase) :=
by sorry

end NUMINAMATH_CALUDE_collectible_toy_price_changes_l1896_189654


namespace NUMINAMATH_CALUDE_triangle_area_formulas_l1896_189694

theorem triangle_area_formulas (R r : ℝ) (A B C : ℝ) :
  let T := R * r * (Real.sin A + Real.sin B + Real.sin C)
  T = 2 * R^2 * Real.sin A * Real.sin B * Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_formulas_l1896_189694


namespace NUMINAMATH_CALUDE_equality_of_powers_l1896_189641

theorem equality_of_powers (a b c d e f : ℕ+) 
  (h1 : 20^21 = 2^(a:ℕ) * 5^(b:ℕ))
  (h2 : 20^21 = 4^(c:ℕ) * 5^(d:ℕ))
  (h3 : 20^21 = 8^(e:ℕ) * 5^(f:ℕ)) :
  100 * (b:ℕ) * (d:ℕ) * (f:ℕ) / ((a:ℕ) * (c:ℕ) * (e:ℕ)) = 75 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_powers_l1896_189641


namespace NUMINAMATH_CALUDE_max_value_fraction_l1896_189675

theorem max_value_fraction (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hbc : b + c ≤ a) :
  ∃ (max : ℝ), max = 1/8 ∧ ∀ x, x = b * c / (a^2 + 2*a*b + b^2) → x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1896_189675


namespace NUMINAMATH_CALUDE_original_number_proof_l1896_189656

theorem original_number_proof : ∃ (n : ℕ), n ≥ 129 ∧ (n - 30) % 99 = 0 ∧ ∀ (m : ℕ), m < 129 → (m - 30) % 99 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l1896_189656


namespace NUMINAMATH_CALUDE_gumball_ratio_l1896_189649

/-- Gumball machine problem -/
theorem gumball_ratio : 
  ∀ (red green blue : ℕ),
  red = 16 →
  green = 4 * blue →
  red + green + blue = 56 →
  blue * 2 = red :=
by
  sorry

end NUMINAMATH_CALUDE_gumball_ratio_l1896_189649


namespace NUMINAMATH_CALUDE_decagon_area_ratio_l1896_189689

theorem decagon_area_ratio (decagon_area : ℝ) (below_PQ_square_area : ℝ) (triangle_base : ℝ) (XQ QY : ℝ) :
  decagon_area = 12 →
  below_PQ_square_area = 1 →
  triangle_base = 6 →
  XQ + QY = 6 →
  (decagon_area / 2 = below_PQ_square_area + (1/2 * triangle_base * ((decagon_area / 2) - below_PQ_square_area) / triangle_base)) →
  XQ / QY = 2 := by
  sorry

end NUMINAMATH_CALUDE_decagon_area_ratio_l1896_189689


namespace NUMINAMATH_CALUDE_south_movement_l1896_189613

-- Define a type for direction
inductive Direction
  | North
  | South

-- Define a function to represent movement
def movement (d : Direction) (distance : ℝ) : ℝ :=
  match d with
  | Direction.North => distance
  | Direction.South => -distance

-- Theorem statement
theorem south_movement :
  movement Direction.North 8 = 8 →
  movement Direction.South 5 = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_south_movement_l1896_189613


namespace NUMINAMATH_CALUDE_rectangle_area_y_l1896_189614

theorem rectangle_area_y (y : ℝ) : y > 0 →
  let E : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (0, 6)
  let G : ℝ × ℝ := (y, 6)
  let H : ℝ × ℝ := (y, 0)
  let area := (G.1 - E.1) * (F.2 - E.2)
  area = 42 →
  y = 7 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_y_l1896_189614


namespace NUMINAMATH_CALUDE_increase_by_fifty_percent_l1896_189681

theorem increase_by_fifty_percent : 
  let initial : ℝ := 100
  let percentage : ℝ := 50
  let increase : ℝ := initial * (percentage / 100)
  let final : ℝ := initial + increase
  final = 150
  := by sorry

end NUMINAMATH_CALUDE_increase_by_fifty_percent_l1896_189681


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l1896_189633

theorem trig_expression_simplification :
  let num := Real.sin (20 * π / 180) + Real.sin (40 * π / 180) + Real.sin (60 * π / 180) + Real.sin (80 * π / 180) +
             Real.sin (100 * π / 180) + Real.sin (120 * π / 180) + Real.sin (140 * π / 180) + Real.sin (160 * π / 180)
  let den := Real.cos (15 * π / 180) * Real.cos (30 * π / 180) * Real.cos (45 * π / 180)
  num / den = (16 * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) * Real.cos (10 * π / 180)) /
              (Real.cos (15 * π / 180) * Real.cos (30 * π / 180) * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_trig_expression_simplification_l1896_189633


namespace NUMINAMATH_CALUDE_function_composition_theorem_l1896_189678

theorem function_composition_theorem (a b : ℤ) :
  (∃ f g : ℤ → ℤ, ∀ x : ℤ, f (g x) = x + a ∧ g (f x) = x + b) ↔ (a = b ∨ a = -b) :=
by sorry

end NUMINAMATH_CALUDE_function_composition_theorem_l1896_189678


namespace NUMINAMATH_CALUDE_find_third_number_l1896_189692

def third_number (a b n : ℕ) : Prop :=
  (Nat.gcd a (Nat.gcd b n) = 8) ∧
  (Nat.lcm a (Nat.lcm b n) = 2^4 * 3^2 * 17 * 7)

theorem find_third_number :
  third_number 136 144 7 :=
by sorry

end NUMINAMATH_CALUDE_find_third_number_l1896_189692


namespace NUMINAMATH_CALUDE_bake_sale_chips_l1896_189644

/-- The number of cups of chocolate chips needed for one recipe -/
def chips_per_recipe : ℕ := 2

/-- The number of recipes to be made -/
def num_recipes : ℕ := 23

/-- The total number of cups of chocolate chips needed -/
def total_chips : ℕ := chips_per_recipe * num_recipes

theorem bake_sale_chips : total_chips = 46 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_chips_l1896_189644


namespace NUMINAMATH_CALUDE_pet_store_siamese_cats_l1896_189601

theorem pet_store_siamese_cats 
  (house_cats : ℕ) 
  (cats_sold : ℕ) 
  (cats_remaining : ℕ) 
  (h1 : house_cats = 25)
  (h2 : cats_sold = 45)
  (h3 : cats_remaining = 18) :
  ∃ (initial_siamese : ℕ), 
    initial_siamese + house_cats = cats_sold + cats_remaining ∧ 
    initial_siamese = 38 := by
sorry

end NUMINAMATH_CALUDE_pet_store_siamese_cats_l1896_189601


namespace NUMINAMATH_CALUDE_joan_picked_37_oranges_l1896_189665

/-- The number of oranges picked by Sara -/
def sara_oranges : ℕ := 10

/-- The total number of oranges picked -/
def total_oranges : ℕ := 47

/-- The number of oranges picked by Joan -/
def joan_oranges : ℕ := total_oranges - sara_oranges

theorem joan_picked_37_oranges : joan_oranges = 37 := by
  sorry

end NUMINAMATH_CALUDE_joan_picked_37_oranges_l1896_189665


namespace NUMINAMATH_CALUDE_journey_rate_problem_l1896_189637

/-- Proves that given a 640-mile journey split into two equal halves, 
    where the second half takes 200% longer than the first half, 
    and the average rate for the entire trip is 40 miles per hour, 
    the average rate for the first half of the trip is 80 miles per hour. -/
theorem journey_rate_problem (total_distance : ℝ) (first_half_rate : ℝ) :
  total_distance = 640 →
  (total_distance / 2) / first_half_rate + 3 * ((total_distance / 2) / first_half_rate) = total_distance / 40 →
  first_half_rate = 80 := by
  sorry

end NUMINAMATH_CALUDE_journey_rate_problem_l1896_189637


namespace NUMINAMATH_CALUDE_cost_of_candies_l1896_189668

/-- The cost of buying lollipops and chocolates -/
theorem cost_of_candies (lollipop_cost : ℕ) (chocolate_cost : ℕ) 
  (lollipop_count : ℕ) (chocolate_count : ℕ) : 
  lollipop_cost = 3 →
  chocolate_cost = 2 →
  lollipop_count = 500 →
  chocolate_count = 300 →
  (lollipop_cost * lollipop_count + chocolate_cost * chocolate_count : ℕ) / 100 = 21 :=
by
  sorry

#check cost_of_candies

end NUMINAMATH_CALUDE_cost_of_candies_l1896_189668


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1896_189646

/-- The length of the major axis of an ellipse with given foci and tangent line -/
theorem ellipse_major_axis_length : ∀ (F₁ F₂ : ℝ × ℝ) (y₀ : ℝ),
  F₁ = (4, 10) →
  F₂ = (34, 40) →
  y₀ = -5 →
  ∃ (X : ℝ × ℝ), X.2 = y₀ ∧ 
    (∀ (P : ℝ × ℝ), P.2 = y₀ → 
      Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) + 
      Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2) ≥ 
      Real.sqrt ((X.1 - F₁.1)^2 + (X.2 - F₁.2)^2) + 
      Real.sqrt ((X.1 - F₂.1)^2 + (X.2 - F₂.2)^2)) →
  30 * Real.sqrt 5 = 
    Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - (2 * y₀ - F₁.2))^2) := by
  sorry

#check ellipse_major_axis_length

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1896_189646


namespace NUMINAMATH_CALUDE_water_added_to_tank_l1896_189612

theorem water_added_to_tank (tank_capacity : ℚ) 
  (initial_fraction : ℚ) (final_fraction : ℚ) : 
  tank_capacity = 72 ∧ initial_fraction = 3/4 ∧ final_fraction = 7/8 →
  (final_fraction - initial_fraction) * tank_capacity = 9 := by
  sorry

end NUMINAMATH_CALUDE_water_added_to_tank_l1896_189612


namespace NUMINAMATH_CALUDE_parabola_ellipse_intersection_l1896_189670

/-- Represents a parabola with equation y² = -4x -/
structure Parabola where
  equation : ∀ x y, y^2 = -4*x

/-- Represents an ellipse with equation x²/4 + y²/b² = 1, where b > 0 -/
structure Ellipse where
  b : ℝ
  b_pos : b > 0
  equation : ∀ x y, x^2/4 + y^2/b^2 = 1

/-- The x-coordinate of the latus rectum for a parabola y² = -4x -/
def latus_rectum_x (p : Parabola) : ℝ := 1

/-- The x-coordinate of the focus for an ellipse x²/4 + y²/b² = 1 -/
def focus_x (e : Ellipse) : ℝ := 1

/-- Theorem stating that if the latus rectum of the parabola passes through
    the focus of the ellipse, then b = √3 -/
theorem parabola_ellipse_intersection
  (p : Parabola) (e : Ellipse)
  (h : latus_rectum_x p = focus_x e) :
  e.b = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_parabola_ellipse_intersection_l1896_189670


namespace NUMINAMATH_CALUDE_jerry_fireworks_l1896_189624

theorem jerry_fireworks (firecrackers sparklers : ℕ) 
  (h1 : firecrackers = 48)
  (h2 : sparklers = 30)
  (confiscated_firecrackers : ℕ := firecrackers / 4)
  (confiscated_sparklers : ℕ := sparklers / 10)
  (remaining_firecrackers : ℕ := firecrackers - confiscated_firecrackers)
  (remaining_sparklers : ℕ := sparklers - confiscated_sparklers)
  (defective_firecrackers : ℕ := remaining_firecrackers / 6)
  (defective_sparklers : ℕ := remaining_sparklers / 4)
  (good_firecrackers : ℕ := remaining_firecrackers - defective_firecrackers)
  (good_sparklers : ℕ := remaining_sparklers - defective_sparklers)
  (set_off_firecrackers : ℕ := good_firecrackers / 2)
  (set_off_sparklers : ℕ := good_sparklers * 2 / 3) :
  set_off_firecrackers + set_off_sparklers = 29 :=
by sorry

end NUMINAMATH_CALUDE_jerry_fireworks_l1896_189624


namespace NUMINAMATH_CALUDE_trig_simplification_l1896_189629

theorem trig_simplification (α : ℝ) : 
  (Real.sin (135 * π / 180 - 2 * α))^2 - 
  (Real.sin (210 * π / 180 - 2 * α))^2 - 
  Real.sin (195 * π / 180) * Real.cos (165 * π / 180 - 4 * α) = 
  Real.sin (4 * α) := by
sorry

end NUMINAMATH_CALUDE_trig_simplification_l1896_189629


namespace NUMINAMATH_CALUDE_zinc_in_mixture_l1896_189635

/-- Given a mixture of zinc and copper with a ratio of 9:11 and a total weight of 74 kg,
    prove that the amount of zinc in the mixture is 33.3 kg. -/
theorem zinc_in_mixture (ratio_zinc : ℚ) (ratio_copper : ℚ) (total_weight : ℚ) :
  ratio_zinc = 9 →
  ratio_copper = 11 →
  total_weight = 74 →
  (ratio_zinc / (ratio_zinc + ratio_copper)) * total_weight = 33.3 := by
  sorry

end NUMINAMATH_CALUDE_zinc_in_mixture_l1896_189635


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1896_189687

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = -b / a) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 - 3*x + 2 - 12
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = 3) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1896_189687


namespace NUMINAMATH_CALUDE_white_pairs_coincide_l1896_189642

/-- Represents the number of triangles of each color in each half of the figure -/
structure HalfFigure where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs when the figure is folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_white : ℕ
  white_white : ℕ

/-- The main theorem stating that 5 white pairs coincide -/
theorem white_pairs_coincide (half : HalfFigure) (pairs : CoincidingPairs) :
  half.red = 4 ∧ half.blue = 6 ∧ half.white = 10 ∧
  pairs.red_red = 3 ∧ pairs.blue_blue = 4 ∧ pairs.red_white = 3 →
  pairs.white_white = 5 := by
  sorry

end NUMINAMATH_CALUDE_white_pairs_coincide_l1896_189642


namespace NUMINAMATH_CALUDE_range_of_k_l1896_189695

-- Define the equation
def equation (x k : ℝ) : Prop := |x| / (x - 2) = k * x

-- Define the property of having three distinct real roots
def has_three_distinct_roots (k : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    equation x₁ k ∧ equation x₂ k ∧ equation x₃ k

-- Theorem statement
theorem range_of_k (k : ℝ) :
  has_three_distinct_roots k ↔ 0 < k ∧ k < 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l1896_189695


namespace NUMINAMATH_CALUDE_selene_purchase_total_l1896_189605

/-- The price of an instant camera -/
def camera_price : ℝ := 110

/-- The price of a digital photo frame -/
def frame_price : ℝ := 120

/-- The number of cameras purchased -/
def num_cameras : ℕ := 2

/-- The number of frames purchased -/
def num_frames : ℕ := 3

/-- The discount rate as a decimal -/
def discount_rate : ℝ := 0.05

/-- The total amount Selene pays -/
def total_paid : ℝ := 551

theorem selene_purchase_total :
  (camera_price * num_cameras + frame_price * num_frames) * (1 - discount_rate) = total_paid := by
  sorry

end NUMINAMATH_CALUDE_selene_purchase_total_l1896_189605


namespace NUMINAMATH_CALUDE_min_Q_value_l1896_189682

def is_special_number (m : ℕ) : Prop :=
  10 ≤ m ∧ m < 100 ∧ m % 10 ≠ m / 10 ∧ m % 10 ≠ 0 ∧ m / 10 ≠ 0

def swap_digits (m : ℕ) : ℕ :=
  (m % 10) * 10 + m / 10

def F (m : ℕ) : ℚ :=
  (m * 100 + swap_digits m - (swap_digits m * 100 + m)) / 99

def Q (s t : ℕ) : ℚ :=
  (t - s : ℚ) / s

theorem min_Q_value (a b x y : ℕ) (h1 : 1 ≤ b) (h2 : b < a) (h3 : a ≤ 7)
    (h4 : 1 ≤ x) (h5 : x ≤ 8) (h6 : 1 ≤ y) (h7 : y ≤ 8)
    (hs : is_special_number (10 * a + b)) (ht : is_special_number (10 * x + y))
    (hFs : F (10 * a + b) % 5 = 1)
    (hFt : F (10 * x + y) - F (10 * a + b) + 18 * x = 36) :
    ∃ (s t : ℕ), is_special_number s ∧ is_special_number t ∧
      Q s t = -42 / 73 ∧ ∀ (s' t' : ℕ), is_special_number s' → is_special_number t' →
        Q s' t' ≥ -42 / 73 :=
  sorry

end NUMINAMATH_CALUDE_min_Q_value_l1896_189682


namespace NUMINAMATH_CALUDE_total_go_stones_l1896_189686

theorem total_go_stones (white_stones black_stones : ℕ) : 
  white_stones = 954 →
  white_stones = black_stones + 468 →
  white_stones + black_stones = 1440 :=
by
  sorry

end NUMINAMATH_CALUDE_total_go_stones_l1896_189686


namespace NUMINAMATH_CALUDE_coefficient_sum_after_shift_l1896_189606

def original_function (x : ℝ) : ℝ := 2 * x^2 - x + 7

def shifted_function (x : ℝ) : ℝ := original_function (x - 4)

def quadratic_form (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem coefficient_sum_after_shift :
  ∃ (a b c : ℝ), (∀ x, shifted_function x = quadratic_form a b c x) ∧ a + b + c = 28 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_sum_after_shift_l1896_189606


namespace NUMINAMATH_CALUDE_dave_apps_remaining_l1896_189632

/-- Calculates the number of apps remaining after deletion -/
def apps_remaining (initial : Nat) (deleted : Nat) : Nat :=
  initial - deleted

theorem dave_apps_remaining :
  apps_remaining 16 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dave_apps_remaining_l1896_189632


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1896_189679

theorem polynomial_factorization (m : ℤ) : 
  (∃ (A B C D E F : ℤ), 
    (A * x + B * y + C) * (D * x + E * y + F) = x^2 + 4*x*y + 2*x + m*y + m^2 - 16) ↔ 
  (m = 5 ∨ m = -6) := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1896_189679


namespace NUMINAMATH_CALUDE_g_composition_of_2_l1896_189625

def g (x : ℕ) : ℕ :=
  if x % 3 = 0 then x / 3 else 5 * x + 1

theorem g_composition_of_2 : g (g (g (g 2))) = 1406 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_2_l1896_189625
