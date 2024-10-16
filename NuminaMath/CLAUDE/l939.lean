import Mathlib

namespace NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l939_93972

/-- The surface area of a cube with the same volume as a rectangular prism of dimensions 10 inches by 5 inches by 20 inches is 600 square inches. -/
theorem cube_surface_area_equal_volume (prism_length prism_width prism_height : ℝ)
  (h1 : prism_length = 10)
  (h2 : prism_width = 5)
  (h3 : prism_height = 20) :
  (6 : ℝ) * ((prism_length * prism_width * prism_height) ^ (1/3 : ℝ))^2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l939_93972


namespace NUMINAMATH_CALUDE_sum_of_threes_plus_product_of_fours_l939_93903

theorem sum_of_threes_plus_product_of_fours (m n : ℕ) :
  (List.replicate m 3).sum + (List.replicate n 4).prod = 3 * m + 4^n := by
  sorry

end NUMINAMATH_CALUDE_sum_of_threes_plus_product_of_fours_l939_93903


namespace NUMINAMATH_CALUDE_paving_cost_is_111405_l939_93908

/-- Calculates the total cost of paving three rooms given their dimensions and paving costs. -/
def total_paving_cost (room1_length room1_width room1_cost_per_sqm
                       room2_length room2_width room2_cost_per_sqm
                       room3_side room3_cost_per_sqm : ℝ) : ℝ :=
  let room1_area := room1_length * room1_width
  let room2_area := room2_length * room2_width
  let room3_area := room3_side * room3_side
  let room1_cost := room1_area * room1_cost_per_sqm
  let room2_cost := room2_area * room2_cost_per_sqm
  let room3_cost := room3_area * room3_cost_per_sqm
  room1_cost + room2_cost + room3_cost

/-- Theorem stating that the total cost of paving the three rooms is 111405. -/
theorem paving_cost_is_111405 :
  total_paving_cost 5.5 3.75 1400 6.4 4.5 1600 4.5 1800 = 111405 := by
  sorry


end NUMINAMATH_CALUDE_paving_cost_is_111405_l939_93908


namespace NUMINAMATH_CALUDE_min_value_theorem_l939_93963

theorem min_value_theorem (a b c : ℝ) 
  (h : ∀ x y : ℝ, x + 2*y - 3 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ x + 2*y + 3) : 
  ∃ m : ℝ, m = a + 2*b - 3*c ∧ ∀ a' b' c' : ℝ, 
    (∀ x y : ℝ, x + 2*y - 3 ≤ a'*x + b'*y + c' ∧ a'*x + b'*y + c' ≤ x + 2*y + 3) →
    m ≤ a' + 2*b' - 3*c' :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l939_93963


namespace NUMINAMATH_CALUDE_exists_set_with_divisibility_property_l939_93958

theorem exists_set_with_divisibility_property (n : ℕ) :
  ∃ (S : Finset ℕ), S.card = n ∧
    ∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b →
      (max a b - min a b) ∣ max a b :=
sorry

end NUMINAMATH_CALUDE_exists_set_with_divisibility_property_l939_93958


namespace NUMINAMATH_CALUDE_smallest_integer_2011m_55555n_l939_93974

theorem smallest_integer_2011m_55555n :
  ∃ (k : ℕ), k > 0 ∧ (∀ (j : ℕ), j > 0 → (∃ (m n : ℤ), j = 2011*m + 55555*n) → k ≤ j) ∧
  (∃ (m n : ℤ), k = 2011*m + 55555*n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_2011m_55555n_l939_93974


namespace NUMINAMATH_CALUDE_sodium_chloride_moles_l939_93928

-- Define the chemical reaction components
structure ChemicalReaction where
  NaCl : ℕ  -- moles of Sodium chloride
  HNO3 : ℕ  -- moles of Nitric acid
  NaNO3 : ℕ  -- moles of Sodium nitrate
  HCl : ℕ   -- moles of Hydrochloric acid

-- Define the theorem
theorem sodium_chloride_moles (reaction : ChemicalReaction) :
  reaction.NaNO3 = 2 →  -- Condition 1
  reaction.HCl = 2 →    -- Condition 2
  reaction.HNO3 = reaction.NaNO3 →  -- Condition 3
  reaction.NaCl = 2 :=  -- Conclusion
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_sodium_chloride_moles_l939_93928


namespace NUMINAMATH_CALUDE_power_quotient_23_l939_93909

theorem power_quotient_23 : (23 ^ 11) / (23 ^ 8) = 12167 := by sorry

end NUMINAMATH_CALUDE_power_quotient_23_l939_93909


namespace NUMINAMATH_CALUDE_average_string_length_l939_93925

theorem average_string_length : 
  let string1 : ℚ := 5/2
  let string2 : ℚ := 11/2
  let string3 : ℚ := 7/2
  let total_length := string1 + string2 + string3
  let num_strings := 3
  (total_length / num_strings) = 23/6 := by
sorry

end NUMINAMATH_CALUDE_average_string_length_l939_93925


namespace NUMINAMATH_CALUDE_polynomial_roots_product_l939_93992

theorem polynomial_roots_product (d e : ℤ) : 
  (∀ r : ℝ, r^2 = r + 1 → r^6 = d*r + e) → d*e = 40 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_product_l939_93992


namespace NUMINAMATH_CALUDE_cats_total_l939_93931

theorem cats_total (initial_cats bought_cats : Float) : 
  initial_cats = 11.0 → bought_cats = 43.0 → initial_cats + bought_cats = 54.0 := by
  sorry

end NUMINAMATH_CALUDE_cats_total_l939_93931


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l939_93971

theorem vector_magnitude_problem (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  (a.1 * b.1 + a.2 * b.2 = 5) →
  ((a.1 - b.1)^2 + (a.2 - b.2)^2 = 20) →
  (b.1^2 + b.2^2 = 25) :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l939_93971


namespace NUMINAMATH_CALUDE_soup_feeding_problem_l939_93951

theorem soup_feeding_problem (initial_cans : ℕ) (adults_per_can : ℕ) (children_per_can : ℕ) 
  (children_to_feed : ℕ) (adults_fed : ℕ) : 
  initial_cans = 8 → 
  adults_per_can = 4 → 
  children_per_can = 6 → 
  children_to_feed = 24 → 
  adults_fed = (initial_cans - (children_to_feed / children_per_can)) * adults_per_can → 
  adults_fed = 16 := by
sorry

end NUMINAMATH_CALUDE_soup_feeding_problem_l939_93951


namespace NUMINAMATH_CALUDE_yuko_wins_l939_93964

theorem yuko_wins (yuri_total yuko_known x y : ℕ) : 
  yuri_total = 17 → yuko_known = 6 → yuko_known + x + y > yuri_total → x + y > 11 := by
  sorry

end NUMINAMATH_CALUDE_yuko_wins_l939_93964


namespace NUMINAMATH_CALUDE_computer_pricing_l939_93922

theorem computer_pricing (selling_price_40 : ℝ) (profit_percentage_40 : ℝ) 
  (selling_price_50 : ℝ) (profit_percentage_50 : ℝ) :
  selling_price_40 = 2240 ∧ 
  profit_percentage_40 = 0.4 ∧ 
  selling_price_50 = 2400 ∧ 
  profit_percentage_50 = 0.5 →
  let cost := selling_price_40 / (1 + profit_percentage_40)
  selling_price_50 = cost * (1 + profit_percentage_50) := by
  sorry


end NUMINAMATH_CALUDE_computer_pricing_l939_93922


namespace NUMINAMATH_CALUDE_delta_y_value_l939_93900

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 + 1

-- State the theorem
theorem delta_y_value (x Δx : ℝ) (hx : x = 1) (hΔx : Δx = 0.1) :
  f (x + Δx) - f x = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_delta_y_value_l939_93900


namespace NUMINAMATH_CALUDE_twenty_three_in_base_two_l939_93905

theorem twenty_three_in_base_two : 23 = 1*2^4 + 0*2^3 + 1*2^2 + 1*2^1 + 1*2^0 := by
  sorry

end NUMINAMATH_CALUDE_twenty_three_in_base_two_l939_93905


namespace NUMINAMATH_CALUDE_remainder_is_224_l939_93967

/-- The polynomial f(x) = x^5 - 8x^4 + 16x^3 + 25x^2 - 50x + 24 -/
def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 16*x^3 + 25*x^2 - 50*x + 24

/-- The remainder when f(x) is divided by (x - 4) -/
def remainder : ℝ := f 4

theorem remainder_is_224 : remainder = 224 := by
  sorry

end NUMINAMATH_CALUDE_remainder_is_224_l939_93967


namespace NUMINAMATH_CALUDE_rhombus_area_l939_93993

/-- The area of a rhombus with side length 5 cm and an interior angle of 60 degrees is 12.5√3 square centimeters. -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 5) (h2 : θ = π / 3) :
  s * s * Real.sin θ = 25 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l939_93993


namespace NUMINAMATH_CALUDE_simplify_expression_l939_93983

theorem simplify_expression : 18 * (7 / 12) * (1 / 6) + 1 / 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l939_93983


namespace NUMINAMATH_CALUDE_goods_train_speed_l939_93961

/-- The speed of a goods train passing a man on another train -/
theorem goods_train_speed (man_train_speed : ℝ) (passing_time : ℝ) (goods_train_length : ℝ) :
  man_train_speed = 40 →
  passing_time = 9 / 3600 →
  goods_train_length = 280 / 1000 →
  ∃ (goods_train_speed : ℝ), goods_train_speed = 72 :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_l939_93961


namespace NUMINAMATH_CALUDE_abs_neg_three_l939_93995

theorem abs_neg_three : abs (-3 : ℤ) = 3 := by sorry

end NUMINAMATH_CALUDE_abs_neg_three_l939_93995


namespace NUMINAMATH_CALUDE_sum_172_83_base4_l939_93981

/-- Converts a natural number to its base 4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Checks if a list of natural numbers represents a valid base 4 number -/
def isValidBase4 (l : List ℕ) : Prop :=
  ∀ d ∈ l, d < 4

theorem sum_172_83_base4 :
  toBase4 (172 + 83) = [3, 3, 3, 3, 3] ∧ isValidBase4 [3, 3, 3, 3, 3] := by
  sorry

end NUMINAMATH_CALUDE_sum_172_83_base4_l939_93981


namespace NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l939_93917

/-- 
Given an arithmetic sequence where:
  - The first term is 2
  - The common difference is 4
Prove that the 150th term of this sequence is 598
-/
theorem arithmetic_sequence_150th_term : 
  ∀ (a : ℕ → ℕ), 
  (a 1 = 2) →  -- First term is 2
  (∀ n, a (n + 1) = a n + 4) →  -- Common difference is 4
  a 150 = 598 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l939_93917


namespace NUMINAMATH_CALUDE_right_triangle_30_perpendicular_segment_l939_93933

/-- In a right triangle with one angle of 30°, the segment of the perpendicular
    from the hypotenuse midpoint to the longer leg is one-third of the longer leg. -/
theorem right_triangle_30_perpendicular_segment (A B C : ℝ × ℝ) 
  (h_right : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (h_30deg : Real.cos (Real.arccos ((C.1 - A.1) / Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))) = Real.sqrt 3 / 2)
  (M : ℝ × ℝ)
  (h_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (K : ℝ × ℝ)
  (h_perpendicular : (K.1 - M.1) * (B.1 - A.1) + (K.2 - M.2) * (B.2 - A.2) = 0)
  (h_on_leg : ∃ t : ℝ, K = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))) :
  Real.sqrt ((K.1 - M.1)^2 + (K.2 - M.2)^2) = 
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) / 3 :=
by sorry


end NUMINAMATH_CALUDE_right_triangle_30_perpendicular_segment_l939_93933


namespace NUMINAMATH_CALUDE_shaded_area_is_one_third_l939_93904

/-- Two rectangles with dimensions 10 × 20 overlap to form a 20 × 30 rectangle. -/
structure OverlappingRectangles where
  small_width : ℝ
  small_height : ℝ
  large_width : ℝ
  large_height : ℝ
  small_width_eq : small_width = 10
  small_height_eq : small_height = 20
  large_width_eq : large_width = 20
  large_height_eq : large_height = 30

/-- The shaded area is the overlap of the two smaller rectangles. -/
def shaded_area (r : OverlappingRectangles) : ℝ :=
  r.small_width * r.small_height

/-- The area of the larger rectangle. -/
def large_area (r : OverlappingRectangles) : ℝ :=
  r.large_width * r.large_height

/-- The theorem stating that the shaded area is 1/3 of the larger rectangle's area. -/
theorem shaded_area_is_one_third (r : OverlappingRectangles) :
    shaded_area r / large_area r = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_one_third_l939_93904


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l939_93947

theorem right_triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) :
  area = (1/2) * leg1 * leg2 →
  leg1 = 30 →
  area = 150 →
  leg2^2 + leg1^2 = hypotenuse^2 →
  leg1 + leg2 + hypotenuse = 40 + 10 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l939_93947


namespace NUMINAMATH_CALUDE_thirty_five_power_ab_equals_R_power_b_times_S_power_a_l939_93937

theorem thirty_five_power_ab_equals_R_power_b_times_S_power_a
  (a b : ℤ) (R S : ℝ) (hR : R = 5^a) (hS : S = 7^b) :
  35^(a*b) = R^b * S^a := by
  sorry

end NUMINAMATH_CALUDE_thirty_five_power_ab_equals_R_power_b_times_S_power_a_l939_93937


namespace NUMINAMATH_CALUDE_max_pages_for_budget_l939_93914

-- Define the cost per page in cents
def cost_per_page : ℕ := 3

-- Define the budget in dollars
def budget : ℕ := 25

-- Define the function to calculate the maximum number of pages
def max_pages (cost : ℕ) (budget : ℕ) : ℕ :=
  (budget * 100) / cost

-- Theorem statement
theorem max_pages_for_budget :
  max_pages cost_per_page budget = 833 := by
  sorry

end NUMINAMATH_CALUDE_max_pages_for_budget_l939_93914


namespace NUMINAMATH_CALUDE_problem_solution_l939_93938

def f (x a : ℝ) := |x - a| * x + |x - 2| * (x - a)

theorem problem_solution :
  (∀ x, f x 1 < 0 ↔ x ∈ Set.Iio 1) ∧
  (∀ a, (∀ x, x ∈ Set.Iio 1 → f x a < 0) ↔ a ∈ Set.Ici 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l939_93938


namespace NUMINAMATH_CALUDE_magazine_purchasing_methods_l939_93913

/-- Represents the number of magazine types priced at 2 yuan -/
def magazines_2yuan : ℕ := 8

/-- Represents the number of magazine types priced at 1 yuan -/
def magazines_1yuan : ℕ := 3

/-- Represents the total amount spent -/
def total_spent : ℕ := 10

/-- Calculates the number of ways to buy magazines -/
def number_of_ways : ℕ := 
  Nat.choose magazines_2yuan 5 + 
  Nat.choose magazines_2yuan 4 * Nat.choose magazines_1yuan 2

theorem magazine_purchasing_methods :
  number_of_ways = 266 := by sorry

end NUMINAMATH_CALUDE_magazine_purchasing_methods_l939_93913


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l939_93921

theorem absolute_value_inequality (x : ℝ) :
  |x - 1| - |x - 5| < 2 ↔ x < 4 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l939_93921


namespace NUMINAMATH_CALUDE_triangle_inequality_l939_93932

/-- Given a triangle ABC with sides a and b, and a point E on side AB such that AE:EB = n:m, 
    prove that CE < (ma + mb) / (m + n). -/
theorem triangle_inequality (A B C E : ℝ × ℝ) (a b : ℝ) (m n : ℝ) :
  let AB := dist A B
  let BC := dist B C
  let CA := dist C A
  let AE := dist A E
  let EB := dist E B
  let CE := dist C E
  (AB = a) →
  (BC = b) →
  (E.1 - A.1) / (B.1 - E.1) = n / m →
  (E.2 - A.2) / (B.2 - E.2) = n / m →
  CE < (m * a + m * b) / (m + n) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l939_93932


namespace NUMINAMATH_CALUDE_log_inequality_l939_93919

theorem log_inequality (a b c : ℝ) (h1 : 0 < c) (h2 : c < 1) (h3 : b > 1) (h4 : a > b) :
  Real.log c / Real.log a > Real.log c / Real.log b :=
sorry

end NUMINAMATH_CALUDE_log_inequality_l939_93919


namespace NUMINAMATH_CALUDE_log_less_than_square_l939_93984

theorem log_less_than_square (x : ℝ) (h : x > 0) : Real.log (1 + x) < x^2 := by
  sorry

end NUMINAMATH_CALUDE_log_less_than_square_l939_93984


namespace NUMINAMATH_CALUDE_existence_of_divisor_l939_93918

def f : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 23 * f (n + 1) + f n

theorem existence_of_divisor (m : ℕ) : ∃ d : ℕ, ∀ n : ℕ, m ∣ f (f n) ↔ d ∣ n := by
  sorry

end NUMINAMATH_CALUDE_existence_of_divisor_l939_93918


namespace NUMINAMATH_CALUDE_money_distribution_problem_l939_93953

/-- Represents the money distribution problem among three friends --/
structure MoneyDistribution where
  total : ℝ  -- Total amount to distribute
  neha : ℝ   -- Neha's share
  sabi : ℝ   -- Sabi's share
  mahi : ℝ   -- Mahi's share
  x : ℝ      -- Amount removed from Sabi's share

/-- The conditions of the problem --/
def problemConditions (d : MoneyDistribution) : Prop :=
  d.total = 1100 ∧
  d.mahi = 102 ∧
  d.neha + d.sabi + d.mahi = d.total ∧
  (d.neha - 5) / (d.sabi - d.x) = 1/4 ∧
  (d.neha - 5) / (d.mahi - 4) = 1/3

/-- The theorem to prove --/
theorem money_distribution_problem (d : MoneyDistribution) 
  (h : problemConditions d) : d.x = 829.67 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_problem_l939_93953


namespace NUMINAMATH_CALUDE_amy_flash_drive_files_l939_93956

/-- Calculates the number of remaining files on Amy's flash drive -/
def remainingFiles (musicFiles videoFiles deletedFiles : ℕ) : ℕ :=
  musicFiles + videoFiles - deletedFiles

/-- Theorem stating the number of remaining files on Amy's flash drive -/
theorem amy_flash_drive_files : remainingFiles 26 36 48 = 14 := by
  sorry

end NUMINAMATH_CALUDE_amy_flash_drive_files_l939_93956


namespace NUMINAMATH_CALUDE_square_sum_of_xy_l939_93941

theorem square_sum_of_xy (x y : ℕ+) 
  (h1 : x * y + x + y = 83)
  (h2 : x^2 * y + x * y^2 = 1056) : 
  x^2 + y^2 = 458 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_xy_l939_93941


namespace NUMINAMATH_CALUDE_a_range_when_p_false_l939_93901

-- Define the proposition p
def p (a : ℝ) : Prop := ∀ x ∈ Set.Ioo 2 3, x^2 + 5 > a*x

-- Define the range of a
def a_range : Set ℝ := Set.Ici (2 * Real.sqrt 5)

-- Theorem statement
theorem a_range_when_p_false :
  (∃ a : ℝ, ¬(p a)) ↔ ∃ a ∈ a_range, True :=
sorry

end NUMINAMATH_CALUDE_a_range_when_p_false_l939_93901


namespace NUMINAMATH_CALUDE_max_value_of_expression_l939_93985

theorem max_value_of_expression (t : ℝ) :
  (∃ (max : ℝ), max = (1 / 8) ∧
    ∀ (t : ℝ), ((3^t - 2*t^2)*t) / (9^t) ≤ max ∧
    ∃ (t_max : ℝ), ((3^t_max - 2*t_max^2)*t_max) / (9^t_max) = max) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l939_93985


namespace NUMINAMATH_CALUDE_candy_distribution_l939_93906

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) :
  total_candy = 15 →
  num_bags = 5 →
  total_candy = num_bags * candy_per_bag →
  candy_per_bag = 3 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l939_93906


namespace NUMINAMATH_CALUDE_bakery_rolls_combinations_l939_93980

theorem bakery_rolls_combinations :
  let total_rolls : ℕ := 9
  let kinds_of_rolls : ℕ := 4
  let min_per_kind : ℕ := 1
  let remaining_rolls : ℕ := total_rolls - kinds_of_rolls * min_per_kind
  Nat.choose (kinds_of_rolls + remaining_rolls - 1) remaining_rolls = 56 := by
  sorry

end NUMINAMATH_CALUDE_bakery_rolls_combinations_l939_93980


namespace NUMINAMATH_CALUDE_inner_outer_hexagon_area_ratio_is_three_fourths_l939_93979

/-- A regular hexagon -/
structure RegularHexagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The ratio of the area of the inner hexagon to the area of the outer hexagon -/
def inner_outer_hexagon_area_ratio (h : RegularHexagon) : ℚ :=
  3 / 4

/-- The theorem stating that the ratio of the areas is 3/4 -/
theorem inner_outer_hexagon_area_ratio_is_three_fourths (h : RegularHexagon) :
  inner_outer_hexagon_area_ratio h = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inner_outer_hexagon_area_ratio_is_three_fourths_l939_93979


namespace NUMINAMATH_CALUDE_sprocket_production_problem_l939_93950

/-- Represents a machine that produces sprockets -/
structure Machine where
  productionRate : ℝ
  timeToProduce660 : ℝ

/-- Given the conditions of the problem -/
theorem sprocket_production_problem 
  (machineA machineP machineQ : Machine)
  (h1 : machineA.productionRate = 6)
  (h2 : machineQ.productionRate = 1.1 * machineA.productionRate)
  (h3 : machineQ.timeToProduce660 = 660 / machineQ.productionRate)
  (h4 : machineP.timeToProduce660 > machineQ.timeToProduce660)
  (h5 : machineP.timeToProduce660 = machineQ.timeToProduce660 + (machineP.timeToProduce660 - machineQ.timeToProduce660)) :
  ¬ ∃ (x : ℝ), machineP.timeToProduce660 - machineQ.timeToProduce660 = x :=
sorry

end NUMINAMATH_CALUDE_sprocket_production_problem_l939_93950


namespace NUMINAMATH_CALUDE_waiter_customers_l939_93978

/-- Given a waiter with 9 tables, each having 7 women and 3 men, prove that the total number of customers is 90. -/
theorem waiter_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) 
  (h1 : num_tables = 9) 
  (h2 : women_per_table = 7) 
  (h3 : men_per_table = 3) : 
  num_tables * (women_per_table + men_per_table) = 90 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l939_93978


namespace NUMINAMATH_CALUDE_matthews_income_l939_93910

/-- Represents the state income tax calculation function -/
def state_tax (q : ℝ) (income : ℝ) : ℝ :=
  0.01 * q * 50000 + 0.01 * (q + 3) * (income - 50000)

/-- Represents the condition that the total tax is (q + 0.5)% of the total income -/
def tax_condition (q : ℝ) (income : ℝ) : Prop :=
  state_tax q income = 0.01 * (q + 0.5) * income

/-- Theorem stating that given the tax calculation method and condition, 
    Matthew's annual income is $60000 -/
theorem matthews_income (q : ℝ) : 
  ∃ (income : ℝ), tax_condition q income ∧ income = 60000 := by
  sorry

end NUMINAMATH_CALUDE_matthews_income_l939_93910


namespace NUMINAMATH_CALUDE_uncovered_side_length_l939_93965

/-- Represents a rectangular field with three sides fenced --/
structure FencedField where
  length : ℝ
  width : ℝ
  area : ℝ
  fencing : ℝ

/-- The uncovered side of a fenced field --/
def uncovered_side (field : FencedField) : ℝ :=
  field.length

theorem uncovered_side_length
  (field : FencedField)
  (h_area : field.area = 50)
  (h_fencing : field.fencing = 25)
  (h_rect : field.area = field.length * field.width)
  (h_fence : field.fencing = 2 * field.width + field.length) :
  uncovered_side field = 20 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_side_length_l939_93965


namespace NUMINAMATH_CALUDE_symmetrical_line_sum_l939_93949

/-- Given a line y = mx + b that is symmetrical to the line x - 3y + 11 = 0
    with respect to the x-axis, prove that m + b = -4 -/
theorem symmetrical_line_sum (m b : ℝ) : 
  (∀ x y, y = m * x + b ↔ x + 3 * y + 11 = 0) → m + b = -4 := by
  sorry

end NUMINAMATH_CALUDE_symmetrical_line_sum_l939_93949


namespace NUMINAMATH_CALUDE_davids_crunches_l939_93940

theorem davids_crunches (zachary_crunches : ℕ) (david_less_crunches : ℕ) 
  (h1 : zachary_crunches = 17)
  (h2 : david_less_crunches = 13) :
  zachary_crunches - david_less_crunches = 4 := by
  sorry

end NUMINAMATH_CALUDE_davids_crunches_l939_93940


namespace NUMINAMATH_CALUDE_complex_root_sum_l939_93944

theorem complex_root_sum (w : ℂ) (hw : w^4 + w^2 + 1 = 0) :
  w^120 + w^121 + w^122 + w^123 + w^124 = w - 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_sum_l939_93944


namespace NUMINAMATH_CALUDE_square_root_divided_by_15_equals_4_l939_93969

theorem square_root_divided_by_15_equals_4 (n : ℝ) : 
  (Real.sqrt n) / 15 = 4 → n = 3600 := by sorry

end NUMINAMATH_CALUDE_square_root_divided_by_15_equals_4_l939_93969


namespace NUMINAMATH_CALUDE_equation_solution_l939_93927

theorem equation_solution : 
  ∃! x : ℚ, (x - 17) / 3 = (3 * x + 4) / 8 ∧ x = -148 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l939_93927


namespace NUMINAMATH_CALUDE_max_profit_theorem_l939_93989

/-- Represents the daily sales volume as a function of unit price -/
def sales_volume (x : ℝ) : ℝ := -100 * x + 5000

/-- Represents the daily profit as a function of unit price -/
def daily_profit (x : ℝ) : ℝ := (sales_volume x) * (x - 6)

/-- The theorem stating the maximum profit and the price at which it occurs -/
theorem max_profit_theorem :
  let x_min : ℝ := 6
  let x_max : ℝ := 32
  ∀ x ∈ Set.Icc x_min x_max,
    daily_profit x ≤ daily_profit 28 ∧
    daily_profit 28 = 48400 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l939_93989


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l939_93982

/-- Given a circle D with equation x^2 - 8x + y^2 + 14y = -28,
    prove that the sum of its center coordinates and radius is -3 + √37 -/
theorem circle_center_radius_sum :
  let D : Set (ℝ × ℝ) := {p | (p.1^2 - 8*p.1 + p.2^2 + 14*p.2 = -28)}
  ∃ (c d s : ℝ),
    (∀ (x y : ℝ), (x, y) ∈ D ↔ (x - c)^2 + (y - d)^2 = s^2) ∧
    c + d + s = -3 + Real.sqrt 37 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l939_93982


namespace NUMINAMATH_CALUDE_original_number_l939_93935

theorem original_number (x : ℝ) : 1 + 1 / x = 9 / 4 → x = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l939_93935


namespace NUMINAMATH_CALUDE_min_difference_f_g_l939_93960

noncomputable def f (x : ℝ) : ℝ := Real.exp (3 * x - 1)
noncomputable def g (x : ℝ) : ℝ := 1 / 3 + Real.log x

theorem min_difference_f_g (m n : ℝ) (h : f m = g n) :
  ∃ (d : ℝ), d = (2 + Real.log 3) / 3 ∧ n - m ≥ d ∧ ∃ (m' n' : ℝ), f m' = g n' ∧ n' - m' = d :=
by sorry

end NUMINAMATH_CALUDE_min_difference_f_g_l939_93960


namespace NUMINAMATH_CALUDE_polynomial_factorization_l939_93998

theorem polynomial_factorization (y : ℝ) : 
  y^8 - 4*y^6 + 6*y^4 - 4*y^2 + 1 = (y-1)^4 * (y+1)^4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l939_93998


namespace NUMINAMATH_CALUDE_geometric_distribution_sum_to_one_l939_93939

/-- The probability mass function for a geometric distribution -/
def geometric_pmf (p : ℝ) (m : ℕ) : ℝ := (1 - p) ^ (m - 1) * p

/-- Theorem: The sum of probabilities for a geometric distribution equals 1 -/
theorem geometric_distribution_sum_to_one (p : ℝ) (hp : 0 < p) (hp' : p < 1) :
  ∑' m : ℕ, geometric_pmf p m = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_distribution_sum_to_one_l939_93939


namespace NUMINAMATH_CALUDE_writing_speed_ratio_l939_93923

/-- Jacob and Nathan's writing speeds -/
def writing_problem (jacob_speed nathan_speed : ℚ) : Prop :=
  nathan_speed = 25 ∧ 
  jacob_speed + nathan_speed = 75 ∧
  jacob_speed / nathan_speed = 2

theorem writing_speed_ratio : ∃ (jacob_speed nathan_speed : ℚ), 
  writing_problem jacob_speed nathan_speed :=
sorry

end NUMINAMATH_CALUDE_writing_speed_ratio_l939_93923


namespace NUMINAMATH_CALUDE_green_blue_difference_after_double_border_l939_93912

/-- Represents a hexagonal figure with blue and green tiles -/
structure HexagonalFigure where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Calculates the number of tiles in a single border layer of a hexagon -/
def border_layer_tiles (layer : ℕ) : ℕ :=
  6 * (2 * layer + 1)

/-- Adds a double border of green tiles to a hexagonal figure -/
def add_double_border (figure : HexagonalFigure) : HexagonalFigure :=
  { blue_tiles := figure.blue_tiles,
    green_tiles := figure.green_tiles + border_layer_tiles 1 + border_layer_tiles 2 }

/-- Theorem: The difference between green and blue tiles after adding a double border is 50 -/
theorem green_blue_difference_after_double_border (initial_figure : HexagonalFigure)
    (h_blue : initial_figure.blue_tiles = 20)
    (h_green : initial_figure.green_tiles = 10) :
    let final_figure := add_double_border initial_figure
    final_figure.green_tiles - final_figure.blue_tiles = 50 := by
  sorry


end NUMINAMATH_CALUDE_green_blue_difference_after_double_border_l939_93912


namespace NUMINAMATH_CALUDE_solution_pair_l939_93994

theorem solution_pair : ∃ (x y : ℤ), 
  Real.sqrt (4 - 3 * Real.sin (30 * π / 180)) = x + y * (1 / Real.sin (30 * π / 180)) ∧ 
  x = 0 ∧ y = 1 := by
  sorry

#check solution_pair

end NUMINAMATH_CALUDE_solution_pair_l939_93994


namespace NUMINAMATH_CALUDE_max_profit_allocation_l939_93945

/-- Represents the allocation of raw materials to workshops --/
structure Allocation :=
  (workshop_a : ℕ)
  (workshop_b : ℕ)

/-- Calculates the profit for a given allocation --/
def profit (a : Allocation) : ℝ :=
  let total_boxes := 60
  let box_cost := 80
  let water_cost := 5
  let product_price := 30
  let workshop_a_production := 12
  let workshop_b_production := 10
  let workshop_a_water := 4
  let workshop_b_water := 2
  30 * (workshop_a_production * a.workshop_a + workshop_b_production * a.workshop_b) -
  box_cost * total_boxes -
  water_cost * (workshop_a_water * a.workshop_a + workshop_b_water * a.workshop_b)

/-- Checks if an allocation satisfies the water consumption constraint --/
def water_constraint (a : Allocation) : Prop :=
  4 * a.workshop_a + 2 * a.workshop_b ≤ 200

/-- Checks if an allocation uses exactly 60 boxes --/
def total_boxes_constraint (a : Allocation) : Prop :=
  a.workshop_a + a.workshop_b = 60

/-- The theorem stating that the given allocation maximizes profit --/
theorem max_profit_allocation :
  ∀ a : Allocation,
  water_constraint a →
  total_boxes_constraint a →
  profit a ≤ profit { workshop_a := 40, workshop_b := 20 } :=
sorry

end NUMINAMATH_CALUDE_max_profit_allocation_l939_93945


namespace NUMINAMATH_CALUDE_triangle_lattice_distance_product_l939_93996

theorem triangle_lattice_distance_product (x y : ℝ) 
  (hx : ∃ (a b : ℤ), x^2 = a^2 + a*b + b^2)
  (hy : ∃ (c d : ℤ), y^2 = c^2 + c*d + d^2) :
  ∃ (e f : ℤ), (x*y)^2 = e^2 + e*f + f^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_lattice_distance_product_l939_93996


namespace NUMINAMATH_CALUDE_parallel_lines_b_value_l939_93902

-- Define the slopes of the two lines
def slope1 (b : ℝ) : ℝ := 4
def slope2 (b : ℝ) : ℝ := b - 3

-- Define the condition for parallel lines
def are_parallel (b : ℝ) : Prop := slope1 b = slope2 b

-- Theorem statement
theorem parallel_lines_b_value :
  ∃ b : ℝ, are_parallel b ∧ b = 7 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_b_value_l939_93902


namespace NUMINAMATH_CALUDE_sum_of_preceding_terms_l939_93976

def arithmetic_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ d : ℕ, ∀ i : ℕ, i < n → a (i + 1) = a i + d

theorem sum_of_preceding_terms (a : ℕ → ℕ) (n : ℕ) :
  arithmetic_sequence a n →
  a 0 = 3 →
  a (n - 1) = 39 →
  n ≥ 3 →
  a (n - 2) + a (n - 3) = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_preceding_terms_l939_93976


namespace NUMINAMATH_CALUDE_find_tuesday_date_l939_93990

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a date in a month -/
structure Date where
  day : ℕ
  month : ℕ
  dayOfWeek : DayOfWeek

/-- Given conditions of the problem -/
def problemConditions (tuesdayDate : Date) (thirdFridayDate : Date) : Prop :=
  tuesdayDate.dayOfWeek = DayOfWeek.Tuesday ∧
  thirdFridayDate.dayOfWeek = DayOfWeek.Friday ∧
  thirdFridayDate.day = 15 ∧
  thirdFridayDate.day + 3 = 18

/-- The theorem to prove -/
theorem find_tuesday_date (tuesdayDate : Date) (thirdFridayDate : Date) :
  problemConditions tuesdayDate thirdFridayDate →
  tuesdayDate.day = 29 ∧ tuesdayDate.month + 1 = thirdFridayDate.month :=
by sorry

end NUMINAMATH_CALUDE_find_tuesday_date_l939_93990


namespace NUMINAMATH_CALUDE_inequality_solution_l939_93926

def solution_set (m : ℝ) : Set ℝ :=
  if m = 0 then Set.univ
  else if m > 0 then {x | -3/m < x ∧ x < 1/m}
  else {x | 1/m < x ∧ x < -3/m}

theorem inequality_solution (m : ℝ) :
  {x : ℝ | m^2 * x^2 + 2*m*x - 3 < 0} = solution_set m :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l939_93926


namespace NUMINAMATH_CALUDE_train_speed_l939_93977

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 160) (h2 : time = 16) :
  length / time = 10 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l939_93977


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l939_93916

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (1 - 1 / (x + 1)) * ((x^2 - 1) / x) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l939_93916


namespace NUMINAMATH_CALUDE_collinear_points_imply_b_value_l939_93946

/-- Given three points in 2D space, this function checks if they are collinear -/
def are_collinear (x1 y1 x2 y2 x3 y3 : ℚ) : Prop :=
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

/-- Theorem stating that if the given points are collinear, then b = -3/13 -/
theorem collinear_points_imply_b_value (b : ℚ) :
  are_collinear 4 (-6) ((-b) + 3) 4 (3*b + 4) 3 → b = -3/13 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_imply_b_value_l939_93946


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l939_93962

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 10| = |x + 4| := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l939_93962


namespace NUMINAMATH_CALUDE_unique_valid_integer_l939_93987

def is_valid_integer (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a ≥ 1 ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 ∧
    a + b + c + d = 18 ∧
    b + c = 11 ∧
    a - d = 3 ∧
    n % 9 = 0

theorem unique_valid_integer : ∃! n : ℕ, is_valid_integer n ∧ n = 5472 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_integer_l939_93987


namespace NUMINAMATH_CALUDE_function_max_min_difference_l939_93968

theorem function_max_min_difference (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := fun x => a^x
  let max_val := max (f 1) (f 2)
  let min_val := min (f 1) (f 2)
  max_val = min_val + a / 3 → a = 4/3 ∨ a = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_function_max_min_difference_l939_93968


namespace NUMINAMATH_CALUDE_complex_modulus_l939_93970

theorem complex_modulus (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 26/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l939_93970


namespace NUMINAMATH_CALUDE_consecutive_numbers_digit_sum_exists_l939_93915

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem consecutive_numbers_digit_sum_exists : ∃ (n : ℕ), 
  sumOfDigits n = 52 ∧ 
  sumOfDigits (n + 4) = 20 ∧ 
  n > 0 :=
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_digit_sum_exists_l939_93915


namespace NUMINAMATH_CALUDE_initial_pennies_equation_l939_93991

/-- Given that Sam spent some pennies and has some left, prove that his initial number of pennies
    is equal to the sum of pennies spent and pennies left. -/
theorem initial_pennies_equation (initial spent left : ℕ) : 
  spent = 93 → left = 5 → initial = spent + left := by sorry

end NUMINAMATH_CALUDE_initial_pennies_equation_l939_93991


namespace NUMINAMATH_CALUDE_fixed_distance_point_l939_93955

open Real

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]

noncomputable def s : ℝ := 9/8
noncomputable def v : ℝ := 1/8

theorem fixed_distance_point (a c p : n) 
  (h : ‖p - c‖ = 3 * ‖p - a‖) : 
  ∃ (k : ℝ), ‖p - (s • a + v • c)‖ = k := by
  sorry

end NUMINAMATH_CALUDE_fixed_distance_point_l939_93955


namespace NUMINAMATH_CALUDE_sequence_length_l939_93952

theorem sequence_length (n : ℕ+) (b : ℕ → ℝ) : 
  b 0 = 41 →
  b 1 = 76 →
  b n = 0 →
  (∀ k : ℕ, 1 ≤ k ∧ k < n → b (k + 1) = b (k - 1) - 4 / b k) →
  n = 777 :=
by sorry

end NUMINAMATH_CALUDE_sequence_length_l939_93952


namespace NUMINAMATH_CALUDE_min_sum_complementary_events_l939_93920

theorem min_sum_complementary_events (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hcomp : 4/x + 1/y = 1) : 
  x + y ≥ 9 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 4/x + 1/y = 1 ∧ x + y = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_complementary_events_l939_93920


namespace NUMINAMATH_CALUDE_no_overlap_in_intervals_l939_93942

theorem no_overlap_in_intervals (x : ℝ) : 
  50 ≤ x ∧ x ≤ 150 ∧ Int.floor (Real.sqrt x) = 11 → 
  Int.floor (Real.sqrt (50 * x)) ≠ 110 := by
sorry

end NUMINAMATH_CALUDE_no_overlap_in_intervals_l939_93942


namespace NUMINAMATH_CALUDE_min_sum_squares_l939_93924

theorem min_sum_squares (x₁ x₂ x₃ x₄ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) 
  (h_sum : 2*x₁ + 3*x₂ + 4*x₃ + 5*x₄ = 120) : 
  x₁^2 + x₂^2 + x₃^2 + x₄^2 ≥ 800/3 ∧ 
  ∃ y₁ y₂ y₃ y₄ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ y₄ > 0 ∧ 
    2*y₁ + 3*y₂ + 4*y₃ + 5*y₄ = 120 ∧ 
    y₁^2 + y₂^2 + y₃^2 + y₄^2 = 800/3 :=
by sorry


end NUMINAMATH_CALUDE_min_sum_squares_l939_93924


namespace NUMINAMATH_CALUDE_ruiStateSurvey2016_sampleSize_l939_93934

/-- Represents a survey about student heights -/
structure HeightSurvey where
  city : String
  year : Nat
  sampleCount : Nat

/-- Definition of sample size for a height survey -/
def sampleSize (survey : HeightSurvey) : Nat := survey.sampleCount

/-- The specific survey conducted in Rui State City in 2016 -/
def ruiStateSurvey2016 : HeightSurvey := {
  city := "Rui State City"
  year := 2016
  sampleCount := 200
}

/-- Theorem stating that the sample size of the Rui State City survey in 2016 is 200 -/
theorem ruiStateSurvey2016_sampleSize :
  sampleSize ruiStateSurvey2016 = 200 := by
  sorry


end NUMINAMATH_CALUDE_ruiStateSurvey2016_sampleSize_l939_93934


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l939_93954

theorem cube_root_equation_solution :
  ∃ y : ℝ, (5 - 2 / y)^(1/3 : ℝ) = -3 ↔ y = 1/16 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l939_93954


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l939_93975

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (Complex.I - 1) ∧ z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l939_93975


namespace NUMINAMATH_CALUDE_intersection_line_equation_l939_93929

/-- Given two lines L₁ and L₂ in the plane, and a third line L that intersects both L₁ and L₂,
    if the midpoint of the line segment formed by these intersections is the origin,
    then L has the equation x + 6y = 0. -/
theorem intersection_line_equation (L₁ L₂ L : Set (ℝ × ℝ)) :
  L₁ = {p : ℝ × ℝ | 4 * p.1 + p.2 + 6 = 0} →
  L₂ = {p : ℝ × ℝ | 3 * p.1 - 5 * p.2 - 6 = 0} →
  (∃ A B : ℝ × ℝ, A ∈ L ∩ L₁ ∧ B ∈ L ∩ L₂ ∧ (A.1 + B.1) / 2 = 0 ∧ (A.2 + B.2) / 2 = 0) →
  L = {p : ℝ × ℝ | p.1 + 6 * p.2 = 0} :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l939_93929


namespace NUMINAMATH_CALUDE_ratio_chain_l939_93959

theorem ratio_chain (a b c d e : ℚ) 
  (h1 : a / b = 3 / 4)
  (h2 : b / c = 7 / 9)
  (h3 : c / d = 5 / 7)
  (h4 : d / e = 11 / 13) :
  a / e = 165 / 468 := by
  sorry

end NUMINAMATH_CALUDE_ratio_chain_l939_93959


namespace NUMINAMATH_CALUDE_steak_distribution_l939_93943

theorem steak_distribution (family_members : Nat) (steak_size : Nat) (steaks_needed : Nat) :
  family_members = 5 →
  steak_size = 20 →
  steaks_needed = 4 →
  (steak_size * steaks_needed) / family_members = 16 :=
by sorry

end NUMINAMATH_CALUDE_steak_distribution_l939_93943


namespace NUMINAMATH_CALUDE_f_positive_range_l939_93986

/-- A function f that is strictly increasing for x > 0 and symmetric about the y-axis -/
noncomputable def f (a b : ℝ) : ℝ → ℝ := λ x => (a * Real.exp x + b) * (x - 2)

/-- The theorem stating the range of m for which f(2-m) > 0 -/
theorem f_positive_range (a b : ℝ) :
  (∀ x > 0, Monotone (f a b)) →
  (∀ x, f a b x = f a b (-x)) →
  {m : ℝ | f a b (2 - m) > 0} = {m : ℝ | m < 0 ∨ m > 4} := by
  sorry

end NUMINAMATH_CALUDE_f_positive_range_l939_93986


namespace NUMINAMATH_CALUDE_magnitude_eighth_power_complex_l939_93988

theorem magnitude_eighth_power_complex (z : ℂ) (h : z = (4/5 : ℂ) + (3/5 : ℂ) * I) : 
  Complex.abs (z^8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_eighth_power_complex_l939_93988


namespace NUMINAMATH_CALUDE_percentage_difference_l939_93930

theorem percentage_difference (x y : ℝ) (h : x = 0.65 * y) : y = (1 + 0.35) * x := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l939_93930


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l939_93973

theorem arithmetic_calculation : 8 / 4 - 3 - 9 + 3 * 9 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l939_93973


namespace NUMINAMATH_CALUDE_unique_integer_point_implies_c_value_l939_93911

/-- The x-coordinate of the first point -/
def x1 : ℚ := 22

/-- The y-coordinate of the first point -/
def y1 : ℚ := 38/3

/-- The y-coordinate of the second point -/
def y2 : ℚ := 53/3

/-- The number of integer points on the line segment -/
def num_integer_points : ℕ := 1

/-- The x-coordinate of the second point -/
def c : ℚ := 23

theorem unique_integer_point_implies_c_value :
  (∃! p : ℤ × ℤ, (x1 : ℚ) < p.1 ∧ p.1 < c ∧
    (p.2 : ℚ) = y1 + (y2 - y1) / (c - x1) * ((p.1 : ℚ) - x1)) →
  c = 23 := by sorry

end NUMINAMATH_CALUDE_unique_integer_point_implies_c_value_l939_93911


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l939_93936

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20412 → n + (n + 1) = 287 ∧ 
  ∀ m : ℕ, m > 0 ∧ m * (m + 1) = 20412 → m + (m + 1) ≥ 287 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l939_93936


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l939_93966

open Set
open Function
open Real

theorem solution_set_of_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h_deriv : ∀ x, deriv f x > f x) :
  {x : ℝ | f x / Real.exp x > f 1 / Real.exp 1} = Ioi 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l939_93966


namespace NUMINAMATH_CALUDE_coefficient_of_x4_in_expansion_l939_93957

def binomial_coefficient (n k : ℕ) : ℕ := sorry

def expansion_coefficient (n : ℕ) : ℕ :=
  binomial_coefficient n 4

theorem coefficient_of_x4_in_expansion : 
  expansion_coefficient 5 + expansion_coefficient 6 + expansion_coefficient 7 = 55 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x4_in_expansion_l939_93957


namespace NUMINAMATH_CALUDE_fiftieth_day_previous_year_is_wednesday_l939_93948

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific day in a year -/
structure DayInYear where
  year : Int
  dayNumber : Nat

/-- Returns the day of the week for a given day in a year -/
def dayOfWeek (d : DayInYear) : DayOfWeek :=
  sorry

theorem fiftieth_day_previous_year_is_wednesday
  (N : Int)
  (h1 : dayOfWeek ⟨N, 250⟩ = DayOfWeek.Friday)
  (h2 : dayOfWeek ⟨N + 1, 150⟩ = DayOfWeek.Friday) :
  dayOfWeek ⟨N - 1, 50⟩ = DayOfWeek.Wednesday :=
sorry

end NUMINAMATH_CALUDE_fiftieth_day_previous_year_is_wednesday_l939_93948


namespace NUMINAMATH_CALUDE_valid_selling_price_l939_93997

/-- Represents the business model for Oleg's water heater production --/
structure WaterHeaterBusiness where
  units_sold : ℕ
  variable_cost : ℕ
  fixed_cost : ℕ
  desired_profit : ℕ
  selling_price : ℕ

/-- Calculates the total revenue given the number of units sold and the selling price --/
def total_revenue (b : WaterHeaterBusiness) : ℕ :=
  b.units_sold * b.selling_price

/-- Calculates the total cost given the number of units sold, variable cost, and fixed cost --/
def total_cost (b : WaterHeaterBusiness) : ℕ :=
  b.units_sold * b.variable_cost + b.fixed_cost

/-- Checks if the selling price satisfies the business requirements --/
def is_valid_price (b : WaterHeaterBusiness) : Prop :=
  total_revenue b ≥ total_cost b + b.desired_profit

/-- Theorem stating that the calculated selling price satisfies the business requirements --/
theorem valid_selling_price :
  let b : WaterHeaterBusiness := {
    units_sold := 5000,
    variable_cost := 800,
    fixed_cost := 1000000,
    desired_profit := 1500000,
    selling_price := 1300
  }
  is_valid_price b ∧ b.selling_price ≥ 0 :=
by sorry


end NUMINAMATH_CALUDE_valid_selling_price_l939_93997


namespace NUMINAMATH_CALUDE_debby_water_bottles_l939_93999

/-- The number of water bottles Debby drank per day -/
def bottles_per_day : ℕ := 109

/-- The number of days the bottles lasted -/
def days_lasted : ℕ := 74

/-- The total number of bottles Debby bought -/
def total_bottles : ℕ := bottles_per_day * days_lasted

theorem debby_water_bottles : total_bottles = 8066 := by
  sorry

end NUMINAMATH_CALUDE_debby_water_bottles_l939_93999


namespace NUMINAMATH_CALUDE_calculation_proof_l939_93907

theorem calculation_proof : (-3)^2 - Real.sqrt 4 + (1/2)⁻¹ = 9 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l939_93907
