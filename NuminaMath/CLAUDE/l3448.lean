import Mathlib

namespace NUMINAMATH_CALUDE_mode_and_median_of_data_set_l3448_344809

def data_set : List ℝ := [1, 1, 4, 5, 5, 5]

/-- The mode of a list of real numbers -/
def mode (l : List ℝ) : ℝ := sorry

/-- The median of a list of real numbers -/
def median (l : List ℝ) : ℝ := sorry

theorem mode_and_median_of_data_set :
  mode data_set = 5 ∧ median data_set = 4.5 := by sorry

end NUMINAMATH_CALUDE_mode_and_median_of_data_set_l3448_344809


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3448_344831

theorem sum_of_coefficients (a b : ℝ) : 
  (∃ x y : ℝ, a * x + b * y = 3 ∧ b * x + a * y = 2) →
  (3 * a + 2 * b = 3 ∧ 3 * b + 2 * a = 2) →
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3448_344831


namespace NUMINAMATH_CALUDE_two_valid_positions_l3448_344824

/-- Represents a square in the polygon arrangement -/
structure Square :=
  (id : Char)

/-- Represents the flat arrangement of squares -/
def FlatArrangement := List Square

/-- Represents a position where an additional square can be attached -/
inductive AttachmentPosition
  | Right : Square → AttachmentPosition
  | Left : Square → AttachmentPosition

/-- Checks if a given attachment position allows folding into a cube with two opposite faces missing -/
def allows_cube_folding (arrangement : FlatArrangement) (pos : AttachmentPosition) : Prop :=
  sorry

/-- The main theorem stating that there are exactly two valid attachment positions -/
theorem two_valid_positions (arrangement : FlatArrangement) :
  (arrangement.length = 4) →
  (∃ A B C D : Square, arrangement = [A, B, C, D]) →
  (∃! (pos1 pos2 : AttachmentPosition),
    pos1 ≠ pos2 ∧
    allows_cube_folding arrangement pos1 ∧
    allows_cube_folding arrangement pos2 ∧
    (∀ pos, allows_cube_folding arrangement pos → (pos = pos1 ∨ pos = pos2))) :=
  sorry

end NUMINAMATH_CALUDE_two_valid_positions_l3448_344824


namespace NUMINAMATH_CALUDE_absolute_value_of_negative_not_negative_l3448_344898

theorem absolute_value_of_negative_not_negative (x : ℝ) (h : x < 0) : |x| ≠ x := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_negative_not_negative_l3448_344898


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l3448_344841

/-- Proves that the profit percent is 140% when selling an article at a certain price,
    given that selling it at 1/3 of that price results in a 20% loss. -/
theorem profit_percent_calculation (C S : ℝ) (h : (1/3) * S = 0.8 * C) :
  (S - C) / C * 100 = 140 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l3448_344841


namespace NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_cos_x_l3448_344827

open Set
open MeasureTheory
open Real

theorem integral_sqrt_one_minus_x_squared_plus_x_cos_x (f : ℝ → ℝ) :
  (∫ x in (-1)..1, (Real.sqrt (1 - x^2) + x * Real.cos x)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_cos_x_l3448_344827


namespace NUMINAMATH_CALUDE_linear_function_value_l3448_344848

theorem linear_function_value (f : ℝ → ℝ) (a b : ℝ) 
    (h1 : f = fun x ↦ a * x + b)
    (h2 : f 1 = 2017)
    (h3 : f 2 = 2018) : 
  f 2019 = 4035 := by
sorry

end NUMINAMATH_CALUDE_linear_function_value_l3448_344848


namespace NUMINAMATH_CALUDE_peaches_left_l3448_344821

def total_peaches : ℕ := 250
def fresh_percentage : ℚ := 60 / 100
def small_peaches : ℕ := 15

theorem peaches_left : 
  (total_peaches * fresh_percentage).floor - small_peaches = 135 := by
  sorry

end NUMINAMATH_CALUDE_peaches_left_l3448_344821


namespace NUMINAMATH_CALUDE_farmer_sold_two_ducks_l3448_344876

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


end NUMINAMATH_CALUDE_farmer_sold_two_ducks_l3448_344876


namespace NUMINAMATH_CALUDE_angle_y_value_l3448_344813

-- Define the angles in the diagram
variable (x y : ℝ)

-- Define the conditions given in the problem
axiom AB_parallel_CD : True  -- We can't directly represent parallel lines, so we use this as a placeholder
axiom angle_BMN : x = 2 * x
axiom angle_MND : x = 70
axiom angle_NMP : x = 70

-- Theorem to prove
theorem angle_y_value : y = 55 := by
  sorry

end NUMINAMATH_CALUDE_angle_y_value_l3448_344813


namespace NUMINAMATH_CALUDE_surprise_combinations_for_week_l3448_344816

/-- The number of combinations for surprise gift placement --/
def surprise_combinations (monday tuesday wednesday thursday friday : ℕ) : ℕ :=
  monday * tuesday * wednesday * thursday * friday

/-- Theorem stating the total number of combinations for the given week --/
theorem surprise_combinations_for_week :
  surprise_combinations 2 1 1 4 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_surprise_combinations_for_week_l3448_344816


namespace NUMINAMATH_CALUDE_kyles_rent_calculation_l3448_344859

def monthly_income : ℕ := 3200

def utilities : ℕ := 150
def retirement_savings : ℕ := 400
def groceries_eating_out : ℕ := 300
def insurance : ℕ := 200
def miscellaneous : ℕ := 200
def car_payment : ℕ := 350
def gas_maintenance : ℕ := 350

def total_expenses : ℕ :=
  utilities + retirement_savings + groceries_eating_out + insurance +
  miscellaneous + car_payment + gas_maintenance

def rent : ℕ := monthly_income - total_expenses

theorem kyles_rent_calculation :
  rent = 1250 :=
sorry

end NUMINAMATH_CALUDE_kyles_rent_calculation_l3448_344859


namespace NUMINAMATH_CALUDE_student_count_problem_l3448_344870

theorem student_count_problem : 
  ∃ n : ℕ, n > 1 ∧ 
  (n - 1) % 2 = 1 ∧ 
  (n - 1) % 7 = 1 ∧ 
  (∀ m : ℕ, m > 1 ∧ m < n → (m - 1) % 2 ≠ 1 ∨ (m - 1) % 7 ≠ 1) ∧
  n = 44 := by
sorry

end NUMINAMATH_CALUDE_student_count_problem_l3448_344870


namespace NUMINAMATH_CALUDE_cube_minus_cylinder_volume_l3448_344881

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem cube_minus_cylinder_volume (cube_side : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ)
  (h1 : cube_side = 6)
  (h2 : cylinder_radius = 3)
  (h3 : cylinder_height = cube_side) :
  cube_side^3 - π * cylinder_radius^2 * cylinder_height = 216 - 54*π :=
by sorry

end NUMINAMATH_CALUDE_cube_minus_cylinder_volume_l3448_344881


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3448_344886

-- Define the sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x^2 ≤ 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3448_344886


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l3448_344882

theorem max_sum_on_circle (x y : ℤ) : 
  x > 0 → y > 0 → x^2 + y^2 = 49 → x + y ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l3448_344882


namespace NUMINAMATH_CALUDE_primitive_root_modulo_power_of_prime_l3448_344887

theorem primitive_root_modulo_power_of_prime
  (p : Nat) (x α : Nat)
  (h_prime : Nat.Prime p)
  (h_alpha : α ≥ 2)
  (h_primitive_root : IsPrimitiveRoot x p)
  (h_not_congruent : ¬ (x^(p^(α-2)*(p-1)) ≡ 1 [MOD p^α])) :
  IsPrimitiveRoot x (p^α) :=
sorry

end NUMINAMATH_CALUDE_primitive_root_modulo_power_of_prime_l3448_344887


namespace NUMINAMATH_CALUDE_students_in_both_clubs_l3448_344867

theorem students_in_both_clubs
  (total_students : ℕ)
  (drama_club : ℕ)
  (science_club : ℕ)
  (in_either_club : ℕ)
  (h1 : total_students = 300)
  (h2 : drama_club = 100)
  (h3 : science_club = 140)
  (h4 : in_either_club = 220) :
  drama_club + science_club - in_either_club = 20 :=
by sorry

end NUMINAMATH_CALUDE_students_in_both_clubs_l3448_344867


namespace NUMINAMATH_CALUDE_rectangle_length_l3448_344842

/-- Proves that a rectangle with perimeter to width ratio of 5:1 and area 150 has length 15 -/
theorem rectangle_length (w l : ℝ) (h1 : w > 0) (h2 : l > 0) : 
  (2 * l + 2 * w) / w = 5 → l * w = 150 → l = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l3448_344842


namespace NUMINAMATH_CALUDE_ratio_problem_l3448_344894

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 3) (h2 : c/b = 2) : 
  (a + b) / (b + c) = 4/9 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3448_344894


namespace NUMINAMATH_CALUDE_queen_heart_jack_probability_l3448_344883

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of queens in a standard deck -/
def QueenCount : ℕ := 4

/-- Number of hearts in a standard deck -/
def HeartCount : ℕ := 13

/-- Number of jacks in a standard deck -/
def JackCount : ℕ := 4

/-- Probability of drawing a queen as the first card, a heart as the second card, 
    and a jack as the third card from a standard 52-card deck -/
def probabilityQueenHeartJack : ℚ := 1 / 663

theorem queen_heart_jack_probability :
  probabilityQueenHeartJack = 
    (QueenCount / StandardDeck) * 
    (HeartCount / (StandardDeck - 1)) * 
    (JackCount / (StandardDeck - 2)) := by
  sorry

end NUMINAMATH_CALUDE_queen_heart_jack_probability_l3448_344883


namespace NUMINAMATH_CALUDE_emily_quiz_score_l3448_344890

def emily_scores : List ℝ := [96, 88, 90, 85, 94]

theorem emily_quiz_score (target_mean : ℝ) (sixth_score : ℝ) :
  target_mean = 92 ∧ sixth_score = 99 →
  (emily_scores.sum + sixth_score) / 6 = target_mean := by
  sorry

end NUMINAMATH_CALUDE_emily_quiz_score_l3448_344890


namespace NUMINAMATH_CALUDE_fraction_of_states_1790s_l3448_344844

/-- The number of states that joined the union during 1790-1799 -/
def states_joined_1790s : ℕ := 7

/-- The total number of states considered -/
def total_states : ℕ := 30

/-- The fraction of states that joined during 1790-1799 out of the first 30 states -/
theorem fraction_of_states_1790s :
  (states_joined_1790s : ℚ) / total_states = 7 / 30 := by sorry

end NUMINAMATH_CALUDE_fraction_of_states_1790s_l3448_344844


namespace NUMINAMATH_CALUDE_negation_of_positive_product_l3448_344855

theorem negation_of_positive_product (x y : ℝ) :
  ¬(x > 0 ∧ y > 0 → x * y > 0) ↔ (x ≤ 0 ∨ y ≤ 0 → x * y ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_positive_product_l3448_344855


namespace NUMINAMATH_CALUDE_modified_lucas_units_digit_l3448_344897

/-- Modified Lucas sequence -/
def M : ℕ → ℕ
  | 0 => 3
  | 1 => 1
  | n + 2 => M (n + 1) + M n + 2

/-- Units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

theorem modified_lucas_units_digit :
  unitsDigit (M (M 6)) = unitsDigit (M 11) :=
sorry

end NUMINAMATH_CALUDE_modified_lucas_units_digit_l3448_344897


namespace NUMINAMATH_CALUDE_time_conversion_not_100_l3448_344815

/-- Represents the conversion rate between adjacent time units -/
def time_conversion_rate : ℕ := 60

/-- The set of standard time units -/
inductive TimeUnit
| Hour
| Minute
| Second

theorem time_conversion_not_100 : time_conversion_rate ≠ 100 := by
  sorry

end NUMINAMATH_CALUDE_time_conversion_not_100_l3448_344815


namespace NUMINAMATH_CALUDE_factorization_2x_squared_minus_8_l3448_344806

theorem factorization_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_2x_squared_minus_8_l3448_344806


namespace NUMINAMATH_CALUDE_max_value_condition_l3448_344818

noncomputable def f (x a : ℝ) : ℝ := -(Real.sin x + a/2)^2 + 3 + a^2/4

theorem max_value_condition (a : ℝ) :
  (∀ x, f x a ≤ 5) ∧ (∃ x, f x a = 5) ↔ a = 3 ∨ a = -3 := by sorry

end NUMINAMATH_CALUDE_max_value_condition_l3448_344818


namespace NUMINAMATH_CALUDE_coefficient_of_x2y_div_3_l3448_344899

/-- Definition of a coefficient in a monomial -/
def coefficient (term : ℚ × (ℕ → ℕ)) : ℚ := term.1

/-- The monomial x^2 * y / 3 -/
def monomial : ℚ × (ℕ → ℕ) := (1/3, fun n => if n = 1 then 2 else if n = 2 then 1 else 0)

/-- Theorem: The coefficient of x^2 * y / 3 is 1/3 -/
theorem coefficient_of_x2y_div_3 : coefficient monomial = 1/3 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x2y_div_3_l3448_344899


namespace NUMINAMATH_CALUDE_base_k_representation_of_fraction_l3448_344837

theorem base_k_representation_of_fraction (k : ℕ) (h : k = 18) :
  let series_sum := (1 / k + 6 / k^2) / (1 - 1 / k^2)
  series_sum = 8 / 63 := by
  sorry

end NUMINAMATH_CALUDE_base_k_representation_of_fraction_l3448_344837


namespace NUMINAMATH_CALUDE_wire_length_ratio_l3448_344868

/-- The length of each wire piece used by Bonnie, in inches -/
def bonnie_wire_length : ℕ := 8

/-- The number of wire pieces used by Bonnie -/
def bonnie_wire_count : ℕ := 12

/-- The length of each wire piece used by Clyde, in inches -/
def clyde_wire_length : ℕ := 2

/-- The side length of Clyde's unit cubes, in inches -/
def clyde_cube_side : ℕ := 1

/-- The number of wire pieces needed for one cube frame -/
def wire_pieces_per_cube : ℕ := 12

theorem wire_length_ratio :
  (bonnie_wire_length * bonnie_wire_count : ℚ) / 
  (clyde_wire_length * wire_pieces_per_cube * bonnie_wire_length ^ 3) = 1 / 128 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l3448_344868


namespace NUMINAMATH_CALUDE_cos_theta_for_point_l3448_344874

/-- If the terminal side of angle θ passes through point P(-12, 5), then cos θ = -12/13 -/
theorem cos_theta_for_point (θ : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos θ = -12 ∧ r * Real.sin θ = 5) → 
  Real.cos θ = -12/13 := by
sorry

end NUMINAMATH_CALUDE_cos_theta_for_point_l3448_344874


namespace NUMINAMATH_CALUDE_inequalities_with_distinct_positive_reals_l3448_344865

theorem inequalities_with_distinct_positive_reals 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  (a^4 + b^4 > a^3*b + a*b^3) ∧ (a^5 + b^5 > a^3*b^2 + a^2*b^3) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_with_distinct_positive_reals_l3448_344865


namespace NUMINAMATH_CALUDE_tank_capacity_l3448_344878

theorem tank_capacity (x : ℝ) 
  (h1 : x / 3 + 180 = 2 * x / 3) : x = 540 := by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l3448_344878


namespace NUMINAMATH_CALUDE_exists_different_degree_same_characteristic_l3448_344856

/-- A characteristic of a polynomial -/
def characteristic (P : Polynomial ℝ) : ℝ := sorry

/-- Theorem: There exist two polynomials with different degrees but the same characteristic -/
theorem exists_different_degree_same_characteristic :
  ∃ (P1 P2 : Polynomial ℝ), 
    (Polynomial.degree P1 ≠ Polynomial.degree P2) ∧ 
    (characteristic P1 = characteristic P2) := by
  sorry

end NUMINAMATH_CALUDE_exists_different_degree_same_characteristic_l3448_344856


namespace NUMINAMATH_CALUDE_ab_value_l3448_344896

theorem ab_value (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 48) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l3448_344896


namespace NUMINAMATH_CALUDE_abs_f_range_l3448_344893

/-- A function whose range is [-2, 3] -/
def f : ℝ → ℝ :=
  sorry

/-- The range of f is [-2, 3] -/
axiom f_range : Set.range f = Set.Icc (-2) 3

/-- Theorem: If the range of f(x) is [-2, 3], then the range of |f(x)| is [0, 3] -/
theorem abs_f_range :
  Set.range (fun x ↦ |f x|) = Set.Icc 0 3 :=
sorry

end NUMINAMATH_CALUDE_abs_f_range_l3448_344893


namespace NUMINAMATH_CALUDE_arithmetic_sequence_odd_numbers_l3448_344860

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_odd_numbers :
  ∀ n : ℕ, n > 0 → arithmetic_sequence 1 2 n = 2 * n - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_odd_numbers_l3448_344860


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l3448_344839

theorem average_of_a_and_b (a b : ℝ) : 
  (4 + 6 + 8 + a + b) / 5 = 20 → (a + b) / 2 = 41 := by
sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l3448_344839


namespace NUMINAMATH_CALUDE_point_distance_to_line_l3448_344836

theorem point_distance_to_line (a : ℝ) (h1 : a > 0) : 
  (|a - 2 + 3| / Real.sqrt 2 = 1) → a = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_point_distance_to_line_l3448_344836


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_435_l3448_344823

/-- The sum of the digits in the binary representation of 435 is 6 -/
theorem sum_of_binary_digits_435 : 
  (Nat.digits 2 435).sum = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_435_l3448_344823


namespace NUMINAMATH_CALUDE_simplify_inverse_sum_l3448_344834

theorem simplify_inverse_sum (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (y * z + x * z + x * y) / (x * y * z * (x + y + z)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_inverse_sum_l3448_344834


namespace NUMINAMATH_CALUDE_negation_of_all_divisible_by_two_are_even_l3448_344803

theorem negation_of_all_divisible_by_two_are_even :
  (¬ ∀ n : ℤ, n % 2 = 0 → Even n) ↔ (∃ n : ℤ, n % 2 = 0 ∧ ¬ Even n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_divisible_by_two_are_even_l3448_344803


namespace NUMINAMATH_CALUDE_function_characterization_l3448_344852

theorem function_characterization (f : ℝ → ℝ) 
  (h1 : f 1 = 1)
  (h2 : ∀ x y : ℝ, f (x + y) = 3^y * f x + 2^x * f y) :
  ∀ x : ℝ, f x = 3^x - 2^x := by
sorry

end NUMINAMATH_CALUDE_function_characterization_l3448_344852


namespace NUMINAMATH_CALUDE_second_quarter_profit_l3448_344833

def annual_profit : ℕ := 8000
def first_quarter_profit : ℕ := 1500
def third_quarter_profit : ℕ := 3000
def fourth_quarter_profit : ℕ := 2000

theorem second_quarter_profit :
  annual_profit - (first_quarter_profit + third_quarter_profit + fourth_quarter_profit) = 1500 :=
by sorry

end NUMINAMATH_CALUDE_second_quarter_profit_l3448_344833


namespace NUMINAMATH_CALUDE_brand_z_percentage_l3448_344854

theorem brand_z_percentage (tank_capacity : ℝ) (brand_z_amount : ℝ) (brand_x_amount : ℝ)
  (h1 : tank_capacity > 0)
  (h2 : brand_z_amount = 1/8 * tank_capacity)
  (h3 : brand_x_amount = 7/8 * tank_capacity)
  (h4 : brand_z_amount + brand_x_amount = tank_capacity) :
  (brand_z_amount / tank_capacity) * 100 = 12.5 := by
sorry

end NUMINAMATH_CALUDE_brand_z_percentage_l3448_344854


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l3448_344838

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → 
  (∀ m : ℕ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) →
  n + (n + 1) = 43 := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l3448_344838


namespace NUMINAMATH_CALUDE_vacation_book_pairs_l3448_344819

/-- The number of ways to choose two books of different genres -/
def different_genre_pairs (mystery fantasy biography : ℕ) : ℕ :=
  mystery * fantasy + mystery * biography + fantasy * biography

/-- Theorem stating that choosing two books of different genres from the given collection results in 33 pairs -/
theorem vacation_book_pairs :
  different_genre_pairs 3 4 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_vacation_book_pairs_l3448_344819


namespace NUMINAMATH_CALUDE_sum_of_ages_is_twelve_l3448_344817

/-- The sum of ages of four children born at one-year intervals -/
def sum_of_ages (youngest_age : ℝ) : ℝ :=
  youngest_age + (youngest_age + 1) + (youngest_age + 2) + (youngest_age + 3)

/-- Theorem: The sum of ages of four children, where the youngest is 1.5 years old
    and each subsequent child is 1 year older, is 12 years. -/
theorem sum_of_ages_is_twelve :
  sum_of_ages 1.5 = 12 := by sorry

end NUMINAMATH_CALUDE_sum_of_ages_is_twelve_l3448_344817


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l3448_344877

/-- Converts a base-5 number (represented as a list of digits) to decimal --/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (5 ^ i)) 0

/-- The base-5 representation of the number in question --/
def base5Number : List Nat := [1, 2, 0, 1, 0, 2, 0, 1]

/-- The decimal representation of the number --/
def decimalNumber : Nat := base5ToDecimal base5Number

/-- Proposition: The largest prime divisor of the given number is 139 --/
theorem largest_prime_divisor :
  ∃ (d : Nat), d.Prime ∧ d ∣ decimalNumber ∧ d = 139 ∧ ∀ (p : Nat), p.Prime → p ∣ decimalNumber → p ≤ d :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_l3448_344877


namespace NUMINAMATH_CALUDE_smallest_number_l3448_344895

-- Define the numbers in their respective bases
def num_decimal : ℕ := 75
def num_binary : ℕ := 63  -- 111111₍₂₎ in decimal
def num_base_6 : ℕ := 2 * 6^2 + 1 * 6  -- 210₍₆₎
def num_base_9 : ℕ := 8 * 9 + 5  -- 85₍₉₎

-- Theorem statement
theorem smallest_number :
  num_binary < num_decimal ∧
  num_binary < num_base_6 ∧
  num_binary < num_base_9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3448_344895


namespace NUMINAMATH_CALUDE_min_value_theorem_range_theorem_l3448_344857

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

-- Define the left focus F₁
def F₁ : ℝ × ℝ := (-2, 0)

-- Define point P
def P : ℝ × ℝ := (1, 1)

-- Define a point M on the ellipse
def M : ℝ × ℝ := sorry

-- Distance between two points
def distance (p₁ p₂ : ℝ × ℝ) : ℝ := sorry

-- Statement for the minimum value
theorem min_value_theorem :
  ∀ M, is_on_ellipse M.1 M.2 →
  ∃ m : ℝ, m = distance M P + (3/2) * distance M F₁ ∧
  m ≥ 11/2 ∧
  ∃ M₀, is_on_ellipse M₀.1 M₀.2 ∧ distance M₀ P + (3/2) * distance M₀ F₁ = 11/2 :=
sorry

-- Statement for the range of values
theorem range_theorem :
  ∀ M, is_on_ellipse M.1 M.2 →
  ∃ r : ℝ, r = distance M P + distance M F₁ ∧
  6 - Real.sqrt 2 < r ∧ r < 6 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_range_theorem_l3448_344857


namespace NUMINAMATH_CALUDE_work_completion_proof_l3448_344805

/-- The number of days it takes x to complete the work -/
def x_total_days : ℝ := 40

/-- The number of days it takes y to complete the work -/
def y_total_days : ℝ := 35

/-- The number of days y worked to finish the work after x stopped -/
def y_actual_days : ℝ := 28

/-- The number of days x worked before y took over -/
def x_worked_days : ℝ := 8

theorem work_completion_proof :
  x_worked_days / x_total_days + y_actual_days / y_total_days = 1 := by
  sorry


end NUMINAMATH_CALUDE_work_completion_proof_l3448_344805


namespace NUMINAMATH_CALUDE_one_meter_per_minute_not_implies_uniform_speed_l3448_344832

/-- A snail's movement over time -/
structure SnailMovement where
  /-- The distance traveled by the snail in meters -/
  distance : ℝ → ℝ
  /-- The property that the snail travels 1 meter every minute -/
  travels_one_meter_per_minute : ∀ t : ℝ, distance (t + 1) - distance t = 1

/-- Definition of uniform speed -/
def UniformSpeed (s : SnailMovement) : Prop :=
  ∃ v : ℝ, ∀ t₁ t₂ : ℝ, s.distance t₂ - s.distance t₁ = v * (t₂ - t₁)

/-- Theorem stating that traveling 1 meter per minute does not imply uniform speed -/
theorem one_meter_per_minute_not_implies_uniform_speed :
  ¬(∀ s : SnailMovement, UniformSpeed s) :=
sorry

end NUMINAMATH_CALUDE_one_meter_per_minute_not_implies_uniform_speed_l3448_344832


namespace NUMINAMATH_CALUDE_green_pill_cost_l3448_344843

/-- The cost of Al's pills for three weeks of treatment --/
def total_cost : ℚ := 1092

/-- The number of days in the treatment period --/
def treatment_days : ℕ := 21

/-- The number of times Al takes a blue pill --/
def blue_pill_count : ℕ := 10

/-- The cost difference between a green pill and a pink pill --/
def green_pink_diff : ℚ := 2

/-- The cost of a pink pill --/
def pink_cost : ℚ := 1050 / 62

/-- The cost of a green pill --/
def green_cost : ℚ := pink_cost + green_pink_diff

/-- Theorem stating the cost of a green pill --/
theorem green_pill_cost : green_cost = 587 / 31 := by sorry

end NUMINAMATH_CALUDE_green_pill_cost_l3448_344843


namespace NUMINAMATH_CALUDE_hallies_reading_l3448_344884

/-- Proves that given the conditions of Hallie's reading pattern, she read 63 pages on the first day -/
theorem hallies_reading (total_pages : ℕ) (day1 : ℕ) : 
  total_pages = 354 → 
  day1 + 2 * day1 + (2 * day1 + 10) + 29 = total_pages → 
  day1 = 63 := by
  sorry

#check hallies_reading

end NUMINAMATH_CALUDE_hallies_reading_l3448_344884


namespace NUMINAMATH_CALUDE_least_number_with_remainders_l3448_344845

theorem least_number_with_remainders : ∃! n : ℕ,
  n > 0 ∧
  n % 56 = 3 ∧
  n % 78 = 3 ∧
  n % 9 = 0 ∧
  ∀ m : ℕ, m > 0 ∧ m % 56 = 3 ∧ m % 78 = 3 ∧ m % 9 = 0 → n ≤ m :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainders_l3448_344845


namespace NUMINAMATH_CALUDE_next_signal_time_l3448_344872

def factory_interval : ℕ := 18
def train_interval : ℕ := 24
def lighthouse_interval : ℕ := 36
def start_time : ℕ := 480  -- 8:00 AM in minutes since midnight

def next_simultaneous_signal (f t l s : ℕ) : ℕ :=
  s + Nat.lcm (Nat.lcm f t) l

theorem next_signal_time :
  next_simultaneous_signal factory_interval train_interval lighthouse_interval start_time = 552 := by
  sorry

#eval next_simultaneous_signal factory_interval train_interval lighthouse_interval start_time

end NUMINAMATH_CALUDE_next_signal_time_l3448_344872


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3448_344869

theorem perfect_square_trinomial (k : ℝ) : 
  (∀ x, ∃ a b : ℝ, x^2 - (k-1)*x + 25 = (a*x + b)^2) ↔ (k = 11 ∨ k = -9) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3448_344869


namespace NUMINAMATH_CALUDE_polygon_sides_count_l3448_344866

-- Define a convex polygon with n sides
def ConvexPolygon (n : ℕ) := n ≥ 3

-- Define the sum of interior angles of a polygon
def SumOfInteriorAngles (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the theorem
theorem polygon_sides_count 
  (n : ℕ) 
  (h_convex : ConvexPolygon n) 
  (h_sum : SumOfInteriorAngles n - (2 * (SumOfInteriorAngles n / (n - 1)) - 20) = 2790) :
  n = 18 := by sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l3448_344866


namespace NUMINAMATH_CALUDE_squirrel_journey_time_l3448_344873

theorem squirrel_journey_time : 
  let first_leg_distance : ℝ := 2
  let first_leg_speed : ℝ := 5
  let second_leg_distance : ℝ := 3
  let second_leg_speed : ℝ := 3
  let first_leg_time : ℝ := first_leg_distance / first_leg_speed
  let second_leg_time : ℝ := second_leg_distance / second_leg_speed
  let total_time_hours : ℝ := first_leg_time + second_leg_time
  let total_time_minutes : ℝ := total_time_hours * 60
  total_time_minutes = 84 := by
sorry


end NUMINAMATH_CALUDE_squirrel_journey_time_l3448_344873


namespace NUMINAMATH_CALUDE_parabola_focus_to_line_distance_l3448_344858

/-- The distance from the focus of the parabola y² = 2x to the line x - √3y = 0 is 1/4 -/
theorem parabola_focus_to_line_distance : 
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 2*x}
  let line := {(x, y) : ℝ × ℝ | x - Real.sqrt 3 * y = 0}
  let focus : ℝ × ℝ := (1/2, 0)
  ∃ d : ℝ, d = 1/4 ∧ ∀ (p : ℝ × ℝ), p ∈ line → Real.sqrt ((p.1 - focus.1)^2 + (p.2 - focus.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_to_line_distance_l3448_344858


namespace NUMINAMATH_CALUDE_derivative_at_negative_third_l3448_344804

theorem derivative_at_negative_third (f : ℝ → ℝ) 
  (h : ∀ x, f x = x^2 + 2 * (deriv f (-1/3)) * x) : 
  deriv f (-1/3) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_negative_third_l3448_344804


namespace NUMINAMATH_CALUDE_four_digit_with_five_or_seven_l3448_344850

theorem four_digit_with_five_or_seven (total_four_digit : Nat) (without_five_or_seven : Nat) :
  total_four_digit = 9000 →
  without_five_or_seven = 3584 →
  total_four_digit - without_five_or_seven = 5416 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_with_five_or_seven_l3448_344850


namespace NUMINAMATH_CALUDE_sphere_surface_area_in_cube_l3448_344820

theorem sphere_surface_area_in_cube (edge_length : Real) (surface_area : Real) :
  edge_length = 2 →
  surface_area = 4 * Real.pi →
  ∃ (r : Real),
    r = edge_length / 2 ∧
    surface_area = 4 * Real.pi * r^2 :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_in_cube_l3448_344820


namespace NUMINAMATH_CALUDE_problem_solution_l3448_344853

theorem problem_solution (a b c : ℝ) (h1 : a - b = 2) (h2 : a + c = 6) :
  (2*a + b + c) - 2*(a - b - c) = 12 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3448_344853


namespace NUMINAMATH_CALUDE_jenny_basket_eggs_l3448_344879

def is_valid_basket_size (n : ℕ) : Prop :=
  n ≥ 5 ∧ 30 % n = 0 ∧ 42 % n = 0

theorem jenny_basket_eggs : ∃! n : ℕ, is_valid_basket_size n ∧ ∀ m : ℕ, is_valid_basket_size m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_jenny_basket_eggs_l3448_344879


namespace NUMINAMATH_CALUDE_jester_count_l3448_344849

theorem jester_count (total_legs total_heads : ℕ) 
  (jester_legs jester_heads elephant_legs elephant_heads : ℕ) : 
  total_legs = 50 → 
  total_heads = 18 → 
  jester_legs = 3 → 
  jester_heads = 1 → 
  elephant_legs = 4 → 
  elephant_heads = 1 → 
  ∃ (num_jesters num_elephants : ℕ), 
    num_jesters * jester_legs + num_elephants * elephant_legs = total_legs ∧
    num_jesters * jester_heads + num_elephants * elephant_heads = total_heads ∧
    num_jesters = 22 :=
by sorry

end NUMINAMATH_CALUDE_jester_count_l3448_344849


namespace NUMINAMATH_CALUDE_max_pages_copied_l3448_344885

/-- The cost in cents to copy one page -/
def cost_per_page : ℕ := 3

/-- The available amount in dollars -/
def available_dollars : ℕ := 25

/-- Convert dollars to cents -/
def dollars_to_cents (dollars : ℕ) : ℕ := dollars * 100

/-- Calculate the number of full pages that can be copied -/
def pages_copied (cents : ℕ) (cost : ℕ) : ℕ := cents / cost

theorem max_pages_copied : 
  pages_copied (dollars_to_cents available_dollars) cost_per_page = 833 := by
  sorry

end NUMINAMATH_CALUDE_max_pages_copied_l3448_344885


namespace NUMINAMATH_CALUDE_min_value_theorem_l3448_344801

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / a + 2 / b = 1) : 
  ∃ (m : ℝ), m = 2 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1 / x + 2 / y = 1 → 
  2 / (x - 1) + 1 / (y - 2) ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3448_344801


namespace NUMINAMATH_CALUDE_triangular_prism_volume_l3448_344830

/-- The volume of a triangular prism with a right triangle base and specific conditions -/
theorem triangular_prism_volume (PQ PR h θ : ℝ) : 
  PQ = Real.sqrt 5 →
  PR = Real.sqrt 5 →
  Real.tan θ = h / Real.sqrt 5 →
  Real.sin θ = 3 / 5 →
  (1 / 2 * PQ * PR) * h = 15 * Real.sqrt 5 / 8 := by
  sorry

#check triangular_prism_volume

end NUMINAMATH_CALUDE_triangular_prism_volume_l3448_344830


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3448_344875

theorem quadratic_inequality_condition (x : ℝ) : x^2 - 4*x ≥ 0 ↔ x ≥ 4 ∨ x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3448_344875


namespace NUMINAMATH_CALUDE_exists_divisible_pair_l3448_344822

def u : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | (n + 2) => u (n + 1) + u n + 1

theorem exists_divisible_pair :
  ∃ n : ℕ, n ≥ 1 ∧ (2011^2012 ∣ u n) ∧ (2011^2012 ∣ u (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_pair_l3448_344822


namespace NUMINAMATH_CALUDE_smallest_divisor_of_Q_l3448_344891

def die_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def Q (visible : Finset ℕ) : ℕ := 
  visible.prod id

theorem smallest_divisor_of_Q : 
  ∀ visible : Finset ℕ, visible ⊆ die_numbers → visible.card = 7 → 
    (∃ k : ℕ, Q visible = 192 * k) ∧ 
    ∀ m : ℕ, m < 192 → (∃ v : Finset ℕ, v ⊆ die_numbers ∧ v.card = 7 ∧ ¬(∃ k : ℕ, Q v = m * k)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_of_Q_l3448_344891


namespace NUMINAMATH_CALUDE_fraction_simplification_l3448_344840

theorem fraction_simplification : 
  (1 - 2 + 4 - 8 + 16 - 32 + 64 - 128 + 256) / 
  (2 - 4 + 8 - 16 + 32 - 64 + 128 - 256 + 512) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3448_344840


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3448_344800

/-- Given a geometric sequence {a_n} where a_1 + a_2 = 324 and a_3 + a_4 = 36,
    prove that a_5 + a_6 = 4 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →
  a 1 + a 2 = 324 →
  a 3 + a 4 = 36 →
  a 5 + a 6 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3448_344800


namespace NUMINAMATH_CALUDE_investment_percentage_problem_l3448_344810

theorem investment_percentage_problem (x y : ℝ) (P : ℝ) : 
  x + y = 2000 →
  y = 600 →
  0.1 * x - (P / 100) * y = 92 →
  P = 8 := by
sorry

end NUMINAMATH_CALUDE_investment_percentage_problem_l3448_344810


namespace NUMINAMATH_CALUDE_problem_solution_l3448_344889

theorem problem_solution (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2)
  (h2 : y^2 / z = 3)
  (h3 : z^2 / x = 4) :
  x = 576^(1/7) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3448_344889


namespace NUMINAMATH_CALUDE_local_minimum_at_two_l3448_344862

/-- The function f(x) = x^3 - 12x --/
def f (x : ℝ) : ℝ := x^3 - 12*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 12

theorem local_minimum_at_two :
  ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → f x ≥ f 2 := by
  sorry

end NUMINAMATH_CALUDE_local_minimum_at_two_l3448_344862


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3448_344871

theorem fractional_equation_solution :
  ∀ x : ℝ, x ≠ 3 → ((x - 2) / (x - 3) = 2 / (x - 3) ↔ x = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3448_344871


namespace NUMINAMATH_CALUDE_bisected_areas_correct_l3448_344846

/-- A rectangle with sides 2 meters and 4 meters, divided by angle bisectors -/
structure BisectedRectangle where
  /-- The length of the shorter side of the rectangle -/
  short_side : ℝ
  /-- The length of the longer side of the rectangle -/
  long_side : ℝ
  /-- The short side is 2 meters -/
  short_side_eq : short_side = 2
  /-- The long side is 4 meters -/
  long_side_eq : long_side = 4
  /-- The angle bisectors are drawn from angles adjacent to the longer side -/
  bisectors_from_long_side : Bool

/-- The areas into which the rectangle is divided by the angle bisectors -/
def bisected_areas (rect : BisectedRectangle) : List ℝ :=
  [2, 2, 4]

/-- Theorem stating that the bisected areas are correct -/
theorem bisected_areas_correct (rect : BisectedRectangle) :
  bisected_areas rect = [2, 2, 4] := by
  sorry

end NUMINAMATH_CALUDE_bisected_areas_correct_l3448_344846


namespace NUMINAMATH_CALUDE_triangle_side_length_l3448_344847

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if the area is √3, angle B is 60°, and a² + c² = 3ac, then the length of side b is 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (1/2 * b * c * Real.sin A = Real.sqrt 3) →  -- Area of triangle is √3
  (B = Real.pi / 3) →  -- Angle B is 60°
  (a^2 + c^2 = 3 * a * c) →  -- Given equation
  (b = 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3448_344847


namespace NUMINAMATH_CALUDE_ball_box_arrangement_l3448_344828

/-- The number of ways to place n different balls into k boxes -/
def total_arrangements (n k : ℕ) : ℕ := k^n

/-- The number of ways to place n different balls into k boxes, 
    with at least one ball in a specific box -/
def arrangements_with_specific_box (n k : ℕ) : ℕ := 
  total_arrangements n k - total_arrangements n (k-1)

theorem ball_box_arrangement : 
  arrangements_with_specific_box 3 6 = 91 := by sorry

end NUMINAMATH_CALUDE_ball_box_arrangement_l3448_344828


namespace NUMINAMATH_CALUDE_museum_ticket_fraction_l3448_344851

theorem museum_ticket_fraction (total_money : ℚ) (sandwich_fraction : ℚ) (book_fraction : ℚ) (leftover : ℚ) : 
  total_money = 120 →
  sandwich_fraction = 1/5 →
  book_fraction = 1/2 →
  leftover = 16 →
  (total_money - (sandwich_fraction * total_money + book_fraction * total_money + leftover)) / total_money = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_museum_ticket_fraction_l3448_344851


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3448_344825

/-- The repeating decimal 4.6̄ -/
def repeating_decimal : ℚ := 4 + 6/9

/-- The fraction 14/3 -/
def fraction : ℚ := 14/3

/-- Theorem: The repeating decimal 4.6̄ is equal to the fraction 14/3 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3448_344825


namespace NUMINAMATH_CALUDE_cyclic_win_sets_count_l3448_344829

/-- Represents a round-robin tournament -/
structure Tournament where
  n : ℕ  -- number of teams
  wins : ℕ  -- number of wins for each team
  losses : ℕ  -- number of losses for each team

/-- Conditions for the tournament -/
def tournament_conditions (t : Tournament) : Prop :=
  t.n * (t.n - 1) / 2 = t.wins * t.n ∧ 
  t.wins = 12 ∧ 
  t.losses = 8 ∧ 
  t.wins + t.losses = t.n - 1

/-- The number of sets of three teams with cyclic wins -/
def cyclic_win_sets (t : Tournament) : ℕ := sorry

/-- Main theorem -/
theorem cyclic_win_sets_count (t : Tournament) : 
  tournament_conditions t → cyclic_win_sets t = 868 := by sorry

end NUMINAMATH_CALUDE_cyclic_win_sets_count_l3448_344829


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3448_344812

theorem max_value_sqrt_sum (x : ℝ) (h : x ∈ Set.Icc (-36) 36) :
  Real.sqrt (36 + x) + Real.sqrt (36 - x) ≤ 12 ∧
  ∃ y ∈ Set.Icc (-36) 36, Real.sqrt (36 + y) + Real.sqrt (36 - y) = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3448_344812


namespace NUMINAMATH_CALUDE_rectangular_field_equation_l3448_344807

theorem rectangular_field_equation (x : ℝ) : 
  (((60 - x) / 2) * ((60 + x) / 2) = 864) ↔ 
  (∃ (length width : ℝ), 
    length * width = 864 ∧ 
    length + width = 60 ∧ 
    length = width + x) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_equation_l3448_344807


namespace NUMINAMATH_CALUDE_book_organizing_group_size_l3448_344808

/-- Represents the number of hours of work for one person to complete the task -/
def total_hours : ℕ := 40

/-- Represents the number of hours worked by the initial group -/
def initial_hours : ℕ := 2

/-- Represents the number of hours worked by the remaining group -/
def remaining_hours : ℕ := 4

/-- Represents the number of people who left the group -/
def people_left : ℕ := 2

theorem book_organizing_group_size :
  ∃ (initial_group : ℕ),
    (initial_hours : ℚ) / total_hours * initial_group + 
    (remaining_hours : ℚ) / total_hours * (initial_group - people_left) = 1 ∧
    initial_group = 8 := by
  sorry

end NUMINAMATH_CALUDE_book_organizing_group_size_l3448_344808


namespace NUMINAMATH_CALUDE_exam_result_proof_l3448_344864

/-- Represents the result of an examination --/
structure ExamResult where
  total_questions : ℕ
  correct_score : ℤ
  wrong_score : ℤ
  unanswered_score : ℤ
  total_score : ℤ
  correct_answers : ℕ
  wrong_answers : ℕ
  unanswered : ℕ

/-- Theorem stating the correct number of answers for the given exam conditions --/
theorem exam_result_proof (exam : ExamResult) : 
  exam.total_questions = 75 ∧ 
  exam.correct_score = 5 ∧ 
  exam.wrong_score = -2 ∧ 
  exam.unanswered_score = -1 ∧ 
  exam.total_score = 215 ∧
  exam.correct_answers + exam.wrong_answers + exam.unanswered = exam.total_questions ∧
  exam.correct_score * exam.correct_answers + exam.wrong_score * exam.wrong_answers + exam.unanswered_score * exam.unanswered = exam.total_score →
  exam.correct_answers = 52 ∧ exam.wrong_answers = 23 ∧ exam.unanswered = 0 := by
  sorry

end NUMINAMATH_CALUDE_exam_result_proof_l3448_344864


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_1_plus_2i_l3448_344802

theorem imaginary_part_of_i_times_1_plus_2i (i : ℂ) (h : i * i = -1) :
  Complex.im (i * (1 + 2*i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_1_plus_2i_l3448_344802


namespace NUMINAMATH_CALUDE_princess_puff_whisker_count_l3448_344880

/-- The number of whiskers Princess Puff has -/
def princess_puff_whiskers : ℕ := 14

/-- The number of whiskers Catman Do has -/
def catman_do_whiskers : ℕ := 22

theorem princess_puff_whisker_count :
  princess_puff_whiskers = 14 ∧
  catman_do_whiskers = 22 ∧
  catman_do_whiskers = 2 * princess_puff_whiskers - 6 :=
by sorry

end NUMINAMATH_CALUDE_princess_puff_whisker_count_l3448_344880


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l3448_344863

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x = Real.sqrt 194 ∧
    (∀ y : ℝ, y < x → ⌊y^2⌋ - ⌊y⌋^2 ≠ 25) ∧
    ⌊x^2⌋ - ⌊x⌋^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l3448_344863


namespace NUMINAMATH_CALUDE_blue_jellybean_probability_l3448_344811

/-- The probability of drawing 3 blue jellybeans in succession from a bag 
    containing 10 red and 10 blue jellybeans, without replacement. -/
theorem blue_jellybean_probability : 
  let total_jellybeans : ℕ := 20
  let blue_jellybeans : ℕ := 10
  let draws : ℕ := 3
  
  -- The probability is calculated as the product of individual probabilities
  (blue_jellybeans / total_jellybeans) * 
  ((blue_jellybeans - 1) / (total_jellybeans - 1)) * 
  ((blue_jellybeans - 2) / (total_jellybeans - 2)) = 2 / 19 := by
sorry


end NUMINAMATH_CALUDE_blue_jellybean_probability_l3448_344811


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3448_344835

/-- The quadratic function f(x) = x^2 + 2x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

/-- f has a root -/
def has_root (m : ℝ) : Prop := ∃ x, f m x = 0

/-- m < 1 is sufficient but not necessary for f to have a root -/
theorem sufficient_not_necessary :
  (∀ m, m < 1 → has_root m) ∧ 
  (∃ m, ¬(m < 1) ∧ has_root m) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3448_344835


namespace NUMINAMATH_CALUDE_min_squares_to_remove_202x202_l3448_344826

/-- Represents a T-tetromino -/
structure TTetromino :=
  (shape : List (Int × Int))

/-- Represents a grid -/
structure Grid :=
  (width : Nat)
  (height : Nat)

/-- Represents a tiling of a grid with T-tetrominoes -/
def Tiling (g : Grid) := List TTetromino

/-- The number of squares that need to be removed for a valid tiling -/
def SquaresToRemove (g : Grid) (t : Tiling g) : Nat :=
  g.width * g.height - 4 * t.length

/-- Theorem: The minimum number of squares to remove from a 202x202 grid for T-tetromino tiling is 4 -/
theorem min_squares_to_remove_202x202 :
  ∀ (g : Grid) (t : Tiling g), g.width = 202 → g.height = 202 →
  SquaresToRemove g t ≥ 4 ∧ ∃ (t' : Tiling g), SquaresToRemove g t' = 4 :=
sorry

end NUMINAMATH_CALUDE_min_squares_to_remove_202x202_l3448_344826


namespace NUMINAMATH_CALUDE_f_minimum_value_l3448_344814

def f (x : ℝ) := |2*x + 1| + |x - 1|

theorem f_minimum_value :
  (∀ x, f x ≥ 3/2) ∧ (∃ x, f x = 3/2) := by sorry

end NUMINAMATH_CALUDE_f_minimum_value_l3448_344814


namespace NUMINAMATH_CALUDE_cubic_expression_value_l3448_344861

theorem cubic_expression_value (a b : ℝ) :
  (a * 1^3 + b * 1 + 1 = 5) → (a * (-1)^3 + b * (-1) + 1 = -3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l3448_344861


namespace NUMINAMATH_CALUDE_distance_A_to_C_l3448_344888

/-- Given four collinear points A, B, C, and D in that order, with specific distance relationships,
    prove that the distance from A to C is 15. -/
theorem distance_A_to_C (A B C D : ℝ) : 
  A < B ∧ B < C ∧ C < D →  -- Points are on a line in order
  D - A = 24 →             -- Distance from A to D is 24
  D - B = 3 * (B - A) →    -- Distance from B to D is 3 times the distance from A to B
  C - B = (D - B) / 2 →    -- C is halfway between B and D
  C - A = 15 := by         -- Distance from A to C is 15
sorry

end NUMINAMATH_CALUDE_distance_A_to_C_l3448_344888


namespace NUMINAMATH_CALUDE_toy_blocks_difference_l3448_344892

theorem toy_blocks_difference (red_blocks yellow_blocks blue_blocks : ℕ) : 
  red_blocks = 18 →
  yellow_blocks = red_blocks + 7 →
  red_blocks + yellow_blocks + blue_blocks = 75 →
  blue_blocks - red_blocks = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_blocks_difference_l3448_344892
