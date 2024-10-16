import Mathlib

namespace NUMINAMATH_CALUDE_asterisk_replacement_l1618_161839

theorem asterisk_replacement : ∃ x : ℝ, (x / 18) * (x / 72) = 1 ∧ x = 36 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l1618_161839


namespace NUMINAMATH_CALUDE_base3_to_base10_conversion_l1618_161805

/-- Converts a base 3 number to base 10 --/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number --/
def base3Number : List Nat := [1, 2, 1, 0, 2]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Number = 178 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_conversion_l1618_161805


namespace NUMINAMATH_CALUDE_fraction_equality_l1618_161899

theorem fraction_equality (x y : ℝ) (h : x ≠ y) :
  (x - y)^2 / (x^2 - y^2) = (x - y) / (x + y) := by
  sorry

#check fraction_equality

end NUMINAMATH_CALUDE_fraction_equality_l1618_161899


namespace NUMINAMATH_CALUDE_definite_integral_equals_six_ln_five_l1618_161835

theorem definite_integral_equals_six_ln_five :
  ∫ x in (π / 4)..(Real.arccos (1 / Real.sqrt 26)),
    36 / ((6 - Real.tan x) * Real.sin (2 * x)) = 6 * Real.log 5 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_equals_six_ln_five_l1618_161835


namespace NUMINAMATH_CALUDE_irrational_floor_inequality_l1618_161891

theorem irrational_floor_inequality :
  ∃ (a b : ℝ), Irrational a ∧ Irrational b ∧ a > 1 ∧ b > 1 ∧
  ∀ (m n : ℕ), ⌊a ^ m⌋ ≠ ⌊b ^ n⌋ := by
  sorry

end NUMINAMATH_CALUDE_irrational_floor_inequality_l1618_161891


namespace NUMINAMATH_CALUDE_biased_coin_probability_l1618_161804

theorem biased_coin_probability : ∀ h : ℝ,
  0 < h ∧ h < 1 →
  (Nat.choose 6 2 : ℝ) * h^2 * (1 - h)^4 = (Nat.choose 6 3 : ℝ) * h^3 * (1 - h)^3 →
  (Nat.choose 6 4 : ℝ) * h^4 * (1 - h)^2 = 19440 / 117649 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l1618_161804


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1618_161873

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  x₁^2 + x₁ - 3 = 0 ∧ x₂^2 + x₂ - 3 = 0 → x₁^3 - 4*x₂^2 + 19 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1618_161873


namespace NUMINAMATH_CALUDE_roses_cut_is_difference_l1618_161808

/-- The number of roses Mary cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Theorem stating that the number of roses Mary cut is the difference between the final and initial number of roses -/
theorem roses_cut_is_difference (initial_roses final_roses : ℕ) 
  (h : final_roses ≥ initial_roses) : 
  roses_cut initial_roses final_roses = final_roses - initial_roses :=
by
  sorry

#eval roses_cut 6 16  -- Should evaluate to 10

end NUMINAMATH_CALUDE_roses_cut_is_difference_l1618_161808


namespace NUMINAMATH_CALUDE_quadratic_root_sum_power_l1618_161838

/-- Given a quadratic equation x^2 + mx + 3 = 0 with roots 1 and n, prove (m + n)^2023 = -1 -/
theorem quadratic_root_sum_power (m n : ℝ) : 
  (1 : ℝ) ^ 2 + m * 1 + 3 = 0 → 
  n ^ 2 + m * n + 3 = 0 → 
  (m + n) ^ 2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_power_l1618_161838


namespace NUMINAMATH_CALUDE_turtle_race_ratio_l1618_161809

theorem turtle_race_ratio : 
  ∀ (greta_time george_time gloria_time : ℕ),
    greta_time = 6 →
    george_time = greta_time - 2 →
    gloria_time = 8 →
    (gloria_time : ℚ) / (george_time : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_turtle_race_ratio_l1618_161809


namespace NUMINAMATH_CALUDE_smallest_covering_circle_l1618_161836

-- Define the plane region
def PlaneRegion (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ x + 2*y - 4 ≤ 0

-- Define the circle equation
def Circle (a b r : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem smallest_covering_circle :
  ∃ (C : ℝ → ℝ → Prop),
    (∀ x y, PlaneRegion x y → C x y) ∧
    (∀ D : ℝ → ℝ → Prop, (∀ x y, PlaneRegion x y → D x y) → 
      ∃ a b r, C = Circle a b r ∧ 
      ∀ a' b' r', D = Circle a' b' r' → r ≤ r') ∧
    C = Circle 2 1 (Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_smallest_covering_circle_l1618_161836


namespace NUMINAMATH_CALUDE_two_arrows_balance_l1618_161807

/-- A polygon with arrows on its sides -/
structure ArrowPolygon where
  n : ℕ  -- number of sides/vertices
  incoming : Fin n → Fin 2  -- number of incoming arrows for each vertex (0, 1, or 2)
  outgoing : Fin n → Fin 2  -- number of outgoing arrows for each vertex (0, 1, or 2)

/-- The sum of incoming arrows equals the number of sides -/
axiom total_arrows_incoming (p : ArrowPolygon) : 
  (Finset.univ.sum p.incoming) = p.n

/-- The sum of outgoing arrows equals the number of sides -/
axiom total_arrows_outgoing (p : ArrowPolygon) : 
  (Finset.univ.sum p.outgoing) = p.n

/-- Theorem: The number of vertices with two incoming arrows equals the number of vertices with two outgoing arrows -/
theorem two_arrows_balance (p : ArrowPolygon) :
  (Finset.univ.filter (fun i => p.incoming i = 2)).card = 
  (Finset.univ.filter (fun i => p.outgoing i = 2)).card := by
  sorry

end NUMINAMATH_CALUDE_two_arrows_balance_l1618_161807


namespace NUMINAMATH_CALUDE_assistant_professor_pencils_l1618_161885

theorem assistant_professor_pencils :
  ∀ (A B P : ℕ),
    A + B = 7 →
    2 * A + P * B = 10 →
    A + 2 * B = 11 →
    P = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_assistant_professor_pencils_l1618_161885


namespace NUMINAMATH_CALUDE_bruce_egg_count_after_loss_l1618_161893

/-- Given Bruce's initial egg count and the number of eggs he loses,
    calculate Bruce's final egg count. -/
def bruces_final_egg_count (initial_count : ℕ) (eggs_lost : ℕ) : ℕ :=
  initial_count - eggs_lost

/-- Theorem stating that given Bruce's initial egg count of 215 and a loss of 137 eggs,
    Bruce's final egg count is 78. -/
theorem bruce_egg_count_after_loss :
  bruces_final_egg_count 215 137 = 78 := by
  sorry

end NUMINAMATH_CALUDE_bruce_egg_count_after_loss_l1618_161893


namespace NUMINAMATH_CALUDE_mika_stickers_total_l1618_161851

/-- The total number of stickers Mika has after receiving stickers from various sources -/
theorem mika_stickers_total : 
  let initial : ℝ := 20.5
  let bought : ℝ := 26.3
  let birthday : ℝ := 19.75
  let sister : ℝ := 6.25
  let mother : ℝ := 57.65
  let cousin : ℝ := 15.8
  initial + bought + birthday + sister + mother + cousin = 146.25 := by
  sorry

end NUMINAMATH_CALUDE_mika_stickers_total_l1618_161851


namespace NUMINAMATH_CALUDE_alpha_plus_beta_equals_113_l1618_161812

theorem alpha_plus_beta_equals_113 (α β : ℝ) : 
  (∀ x : ℝ, x ≠ 45 → (x - α) / (x + β) = (x^2 - 90*x + 1981) / (x^2 + 63*x - 3420)) →
  α + β = 113 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_equals_113_l1618_161812


namespace NUMINAMATH_CALUDE_union_of_sets_l1618_161888

theorem union_of_sets (S T : Set ℕ) (h1 : S = {0, 1}) (h2 : T = {0}) : 
  S ∪ T = {0, 1} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1618_161888


namespace NUMINAMATH_CALUDE_exists_positive_m_for_field_l1618_161802

/-- The dimensions of a rectangular field -/
def field_length (m : ℝ) : ℝ := 4*m + 6

/-- The width of a rectangular field -/
def field_width (m : ℝ) : ℝ := 2*m - 5

/-- The area of the rectangular field -/
def field_area : ℝ := 159

/-- Theorem stating that there exists a positive real number m that satisfies the field dimensions and area -/
theorem exists_positive_m_for_field : ∃ m : ℝ, m > 0 ∧ field_length m * field_width m = field_area := by
  sorry

end NUMINAMATH_CALUDE_exists_positive_m_for_field_l1618_161802


namespace NUMINAMATH_CALUDE_no_square_divisor_of_power_sum_l1618_161880

theorem no_square_divisor_of_power_sum (a b : ℕ) (α : ℕ) (ha : Odd a) (hb : Odd b) 
  (ha_gt1 : a > 1) (hb_gt1 : b > 1) (hsum : a + b = 2^α) (hα : α ≥ 1) :
  ¬ ∃ (k : ℕ), k > 1 ∧ k^2 ∣ (a^k + b^k) := by
  sorry

end NUMINAMATH_CALUDE_no_square_divisor_of_power_sum_l1618_161880


namespace NUMINAMATH_CALUDE_probability_collinear_dots_l1618_161894

/-- Represents a rectangular array of dots -/
structure DotArray where
  rows : ℕ
  cols : ℕ
  total_dots : ℕ

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the number of collinear sets of 4 dots in a vertical line -/
def vertical_collinear_sets (arr : DotArray) : ℕ := 
  arr.cols * choose arr.rows 4

/-- Main theorem: Probability of choosing 4 collinear dots -/
theorem probability_collinear_dots (arr : DotArray) 
  (h1 : arr.rows = 5) 
  (h2 : arr.cols = 4) 
  (h3 : arr.total_dots = 20) : 
  (vertical_collinear_sets arr : ℚ) / (choose arr.total_dots 4) = 4 / 969 := by
  sorry

end NUMINAMATH_CALUDE_probability_collinear_dots_l1618_161894


namespace NUMINAMATH_CALUDE_sum_of_angles_l1618_161803

-- Define the angles as real numbers
variable (A B C D F G EDC ECD : ℝ)

-- Define the conditions
variable (h1 : A + B + C + D = 360) -- ABCD is a quadrilateral
variable (h2 : G + F = EDC + ECD)   -- Given condition

-- Theorem statement
theorem sum_of_angles : A + B + C + D + F + G = 360 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_l1618_161803


namespace NUMINAMATH_CALUDE_hawks_score_l1618_161820

/-- Given the total score and winning margin in a basketball game, 
    calculate the score of the losing team. -/
theorem hawks_score (total_score winning_margin : ℕ) : 
  total_score = 58 → winning_margin = 12 → 
  (total_score - winning_margin) / 2 = 23 := by
  sorry

#check hawks_score

end NUMINAMATH_CALUDE_hawks_score_l1618_161820


namespace NUMINAMATH_CALUDE_f_properties_l1618_161842

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.cos x)^2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3 * (Real.sin x)^2

theorem f_properties :
  (∀ x, f (x + π/12) = f (π/12 - x)) ∧
  (∀ x, f (π/3 + x) = -f (π/3 - x)) ∧
  (∃ x₁ x₂, |f x₁ - f x₂| ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1618_161842


namespace NUMINAMATH_CALUDE_room_width_calculation_l1618_161817

theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (width : ℝ) : 
  length = 8 →
  cost_per_sqm = 900 →
  total_cost = 34200 →
  width = total_cost / cost_per_sqm / length →
  width = 4.75 :=
by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l1618_161817


namespace NUMINAMATH_CALUDE_total_fruits_is_107_l1618_161871

/-- The number of fruits picked by George and Amelia -/
def total_fruits (george_oranges amelia_apples : ℕ) : ℕ :=
  let george_apples := amelia_apples + 5
  let amelia_oranges := george_oranges - 18
  (george_oranges + amelia_oranges) + (george_apples + amelia_apples)

/-- Theorem stating that the total number of fruits picked is 107 -/
theorem total_fruits_is_107 :
  total_fruits 45 15 = 107 := by sorry

end NUMINAMATH_CALUDE_total_fruits_is_107_l1618_161871


namespace NUMINAMATH_CALUDE_point_c_coordinates_l1618_161860

/-- Given points A and B in ℝ³, if vector AC is half of vector AB, then C has specific coordinates -/
theorem point_c_coordinates (A B C : ℝ × ℝ × ℝ) : 
  A = (2, 2, 7) → 
  B = (-2, 4, 3) → 
  C - A = (1 / 2 : ℝ) • (B - A) → 
  C = (0, 3, 5) := by
  sorry

end NUMINAMATH_CALUDE_point_c_coordinates_l1618_161860


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1618_161890

theorem fraction_multiplication : (1 : ℚ) / 3 * (1 : ℚ) / 2 * (3 : ℚ) / 4 * (5 : ℚ) / 6 = (5 : ℚ) / 48 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1618_161890


namespace NUMINAMATH_CALUDE_remainder_2022_power_2023_power_2024_mod_19_l1618_161806

theorem remainder_2022_power_2023_power_2024_mod_19 :
  (2022 ^ (2023 ^ 2024)) % 19 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2022_power_2023_power_2024_mod_19_l1618_161806


namespace NUMINAMATH_CALUDE_ages_correct_l1618_161854

-- Define the current ages as variables
def hans_age : ℕ := 8
def annika_age : ℕ := 25
def emil_age : ℕ := 5
def frida_age : ℕ := 20

-- Define the conditions
axiom future_annika : annika_age + 4 = 3 * (hans_age + 4)
axiom future_emil : emil_age + 4 = 2 * (hans_age + 4)
axiom future_age_diff : (emil_age + 4) - (hans_age + 4) = (frida_age + 4) / 2
axiom sum_of_ages : hans_age + annika_age + emil_age + frida_age = 58
axiom frida_annika_diff : frida_age + 5 = annika_age

-- Theorem to prove
theorem ages_correct : 
  annika_age = 25 ∧ emil_age = 5 ∧ frida_age = 20 :=
by sorry

end NUMINAMATH_CALUDE_ages_correct_l1618_161854


namespace NUMINAMATH_CALUDE_duck_to_pig_water_ratio_l1618_161855

/-- Proves that the ratio of water needed by each duck to water needed by each pig is 1:16 --/
theorem duck_to_pig_water_ratio :
  let pump_rate := 3 -- gallons per minute
  let pump_time := 25 -- minutes
  let corn_rows := 4
  let corn_plants_per_row := 15
  let water_per_corn_plant := 0.5 -- gallons
  let num_pigs := 10
  let water_per_pig := 4 -- gallons
  let num_ducks := 20

  let total_water := pump_rate * pump_time
  let corn_water := corn_rows * corn_plants_per_row * water_per_corn_plant
  let pig_water := num_pigs * water_per_pig
  let duck_water := total_water - corn_water - pig_water
  let water_per_duck := duck_water / num_ducks

  water_per_duck / water_per_pig = 1 / 16 := by
    sorry

end NUMINAMATH_CALUDE_duck_to_pig_water_ratio_l1618_161855


namespace NUMINAMATH_CALUDE_equation_solution_l1618_161826

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (8 * x)^16 = (32 * x)^8 → x = 1/2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1618_161826


namespace NUMINAMATH_CALUDE_hexagon_angle_sum_l1618_161878

-- Define the hexagon and its properties
structure Hexagon where
  A : ℝ
  B : ℝ
  C : ℝ
  x : ℝ
  y : ℝ
  h : A = 34
  i : B = 74
  j : C = 32

-- State the theorem
theorem hexagon_angle_sum (H : Hexagon) : H.x + H.y = 40 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_sum_l1618_161878


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1618_161819

theorem arithmetic_equality : 3 * 12 + 3 * 13 + 3 * 16 + 11 = 134 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1618_161819


namespace NUMINAMATH_CALUDE_diamond_value_l1618_161892

theorem diamond_value (diamond : ℕ) (h1 : diamond < 10) 
  (h2 : diamond * 9 + 5 = diamond * 10 + 2) : diamond = 3 := by
  sorry

end NUMINAMATH_CALUDE_diamond_value_l1618_161892


namespace NUMINAMATH_CALUDE_remainder_after_adding_3006_l1618_161898

theorem remainder_after_adding_3006 (n : ℤ) (h : n % 6 = 1) : (n + 3006) % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_3006_l1618_161898


namespace NUMINAMATH_CALUDE_rest_of_body_length_l1618_161887

theorem rest_of_body_length (total_height : ℝ) (leg_ratio : ℝ) (head_ratio : ℝ) 
  (h1 : total_height = 60)
  (h2 : leg_ratio = 1/3)
  (h3 : head_ratio = 1/4) :
  total_height - (leg_ratio * total_height) - (head_ratio * total_height) = 25 := by
  sorry

end NUMINAMATH_CALUDE_rest_of_body_length_l1618_161887


namespace NUMINAMATH_CALUDE_system_solution_l1618_161847

theorem system_solution (a b : ℚ) : 
  (∃ b, 2 * 1 - b * 2 = 1) →
  (∃ a, a * 1 + 1 = 2) →
  (∃! x y : ℚ, a * x + y = 2 ∧ 2 * x - b * y = 1 ∧ x = 4/5 ∧ y = 6/5) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1618_161847


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l1618_161865

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  AD : ℝ
  height : ℝ
  is_trapezoid : True
  AB_eq_CD : AB = CD
  BC_eq_10 : BC = 10
  AD_eq_22 : AD = 22
  height_eq_5 : height = 5

/-- The perimeter of the trapezoid ABCD is 2√61 + 32 -/
theorem trapezoid_perimeter (t : Trapezoid) : 
  t.AB + t.BC + t.CD + t.AD = 2 * Real.sqrt 61 + 32 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_perimeter_l1618_161865


namespace NUMINAMATH_CALUDE_move_point_right_point_B_position_l1618_161813

def point_on_number_line (x : ℤ) := x

theorem move_point_right (start : ℤ) (distance : ℕ) :
  point_on_number_line (start + distance) = point_on_number_line start + distance :=
by sorry

theorem point_B_position :
  let point_A := point_on_number_line (-3)
  let move_distance := 4
  let point_B := point_on_number_line (point_A + move_distance)
  point_B = 1 :=
by sorry

end NUMINAMATH_CALUDE_move_point_right_point_B_position_l1618_161813


namespace NUMINAMATH_CALUDE_stratified_sampling_results_l1618_161858

theorem stratified_sampling_results (junior_students senior_students sample_size : ℕ) 
  (h1 : junior_students = 400)
  (h2 : senior_students = 200)
  (h3 : sample_size = 60) :
  let junior_sample := (junior_students * sample_size) / (junior_students + senior_students)
  let senior_sample := sample_size - junior_sample
  Nat.choose junior_students junior_sample * Nat.choose senior_students senior_sample =
  Nat.choose 400 40 * Nat.choose 200 20 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_results_l1618_161858


namespace NUMINAMATH_CALUDE_expression_factorization_l1618_161861

theorem expression_factorization (x : ℝ) :
  (3 * x^3 + 70 * x^2 - 5) - (-4 * x^3 + 2 * x^2 - 5) = 7 * x^2 * (x + 68/7) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1618_161861


namespace NUMINAMATH_CALUDE_vector_addition_l1618_161818

theorem vector_addition (A B C : ℝ × ℝ) : 
  (B.1 - A.1, B.2 - A.2) = (0, 1) →
  (C.1 - B.1, C.2 - B.2) = (1, 0) →
  (C.1 - A.1, C.2 - A.2) = (1, 1) := by
sorry

end NUMINAMATH_CALUDE_vector_addition_l1618_161818


namespace NUMINAMATH_CALUDE_apple_distribution_l1618_161889

/-- Represents the number of apples each person receives when evenly distributing
    a given number of apples among a given number of people. -/
def apples_per_person (total_apples : ℕ) (num_people : ℕ) : ℕ :=
  total_apples / num_people

/-- Theorem stating that when 15 apples are evenly distributed among 3 people,
    each person receives 5 apples. -/
theorem apple_distribution :
  apples_per_person 15 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l1618_161889


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_one_l1618_161884

theorem sqrt_difference_equals_one : Real.sqrt 25 - Real.sqrt 16 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_one_l1618_161884


namespace NUMINAMATH_CALUDE_problem_solution_l1618_161832

noncomputable def problem (a b c k x y z : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ k ≠ 0 ∧
  x * y / (x + y) = a ∧
  x * z / (x + z) = b ∧
  y * z / (y + z) = c ∧
  x * y * z / (x + y + z) = k

theorem problem_solution (a b c k x y z : ℝ) (h : problem a b c k x y z) :
  x = 2 * k * a * b / (a * b + b * c - a * c) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1618_161832


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1618_161822

theorem polynomial_divisibility (n : ℕ) (h : n > 1) :
  ∃ Q : Polynomial ℂ, x^(4*n+3) + x^(4*n+1) + x^(4*n-2) + x^8 = (x^2 + 1) * Q := by
  sorry

#check polynomial_divisibility

end NUMINAMATH_CALUDE_polynomial_divisibility_l1618_161822


namespace NUMINAMATH_CALUDE_regression_line_at_12_l1618_161841

def regression_line (x_mean y_mean slope : ℝ) (x : ℝ) : ℝ :=
  slope * (x - x_mean) + y_mean

theorem regression_line_at_12 
  (x_mean : ℝ) 
  (y_mean : ℝ) 
  (slope : ℝ) 
  (h1 : x_mean = 10) 
  (h2 : y_mean = 4) 
  (h3 : slope = 0.6) :
  regression_line x_mean y_mean slope 12 = 5.2 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_at_12_l1618_161841


namespace NUMINAMATH_CALUDE_garcia_fourth_quarter_shots_l1618_161821

/-- Represents the number of shots taken and made in a basketball game --/
structure GameStats :=
  (shots_taken : ℕ)
  (shots_made : ℕ)

/-- Calculates the shooting accuracy as a rational number --/
def accuracy (stats : GameStats) : ℚ :=
  stats.shots_made / stats.shots_taken

theorem garcia_fourth_quarter_shots 
  (first_two_quarters : GameStats)
  (third_quarter : GameStats)
  (fourth_quarter : GameStats)
  (h1 : first_two_quarters.shots_taken = 20)
  (h2 : first_two_quarters.shots_made = 12)
  (h3 : third_quarter.shots_taken = 10)
  (h4 : accuracy third_quarter = (1/2) * accuracy first_two_quarters)
  (h5 : accuracy fourth_quarter = (4/3) * accuracy third_quarter)
  (h6 : accuracy (GameStats.mk 
    (first_two_quarters.shots_taken + third_quarter.shots_taken + fourth_quarter.shots_taken)
    (first_two_quarters.shots_made + third_quarter.shots_made + fourth_quarter.shots_made)) = 46/100)
  : fourth_quarter.shots_made = 8 := by
  sorry

end NUMINAMATH_CALUDE_garcia_fourth_quarter_shots_l1618_161821


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l1618_161829

theorem smallest_gcd_bc (a b c : ℕ+) (h1 : Nat.gcd a b = 240) (h2 : Nat.gcd a c = 1001) :
  ∃ (b' c' : ℕ+), Nat.gcd b'.val c'.val = 1 ∧ 
    ∀ (b'' c'' : ℕ+), Nat.gcd a b''.val = 240 → Nat.gcd a c''.val = 1001 → 
      Nat.gcd b''.val c''.val ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l1618_161829


namespace NUMINAMATH_CALUDE_james_to_remaining_ratio_l1618_161883

def total_slices : ℕ := 8
def friend_eats : ℕ := 2
def james_eats : ℕ := 3

def slices_after_friend : ℕ := total_slices - friend_eats

theorem james_to_remaining_ratio :
  (james_eats : ℚ) / slices_after_friend = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_james_to_remaining_ratio_l1618_161883


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_ninety_five_satisfies_conditions_largest_integer_is_95_l1618_161844

theorem largest_integer_with_remainder (n : ℕ) : n < 100 ∧ n % 7 = 4 → n ≤ 95 :=
by
  sorry

theorem ninety_five_satisfies_conditions : 95 < 100 ∧ 95 % 7 = 4 :=
by
  sorry

theorem largest_integer_is_95 : ∀ (n : ℕ), n < 100 ∧ n % 7 = 4 → n ≤ 95 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_ninety_five_satisfies_conditions_largest_integer_is_95_l1618_161844


namespace NUMINAMATH_CALUDE_tangent_line_proof_l1618_161825

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 5*x^2 - 5

-- Define the given line
def l₁ (z y : ℝ) : Prop := 2*z - 6*y + 1 = 0

-- Define the tangent line
def l₂ (x y : ℝ) : Prop := 3*x + y + 6 = 0

-- Theorem statement
theorem tangent_line_proof :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) lies on the curve
    f x₀ = y₀ ∧
    -- The tangent line passes through (x₀, y₀)
    l₂ x₀ y₀ ∧
    -- The slope of the tangent line at (x₀, y₀) is the derivative of f at x₀
    (3*x₀^2 + 10*x₀ = -3) ∧
    -- The two lines are perpendicular
    ∀ (z₁ y₁ z₂ y₂ : ℝ),
      l₁ z₁ y₁ ∧ l₁ z₂ y₂ ∧ z₁ ≠ z₂ →
      (y₁ - y₂) / (z₁ - z₂) * (-1/3) = -1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_proof_l1618_161825


namespace NUMINAMATH_CALUDE_exponent_addition_l1618_161886

theorem exponent_addition (a : ℝ) (m n : ℕ) : a^m * a^n = a^(m + n) := by sorry

end NUMINAMATH_CALUDE_exponent_addition_l1618_161886


namespace NUMINAMATH_CALUDE_runner_solution_l1618_161816

def runner_problem (t : ℕ) : Prop :=
  let first_runner := 2
  let second_runner := 4
  let third_runner := t
  let meeting_time := 44
  (meeting_time % first_runner = 0) ∧
  (meeting_time % second_runner = 0) ∧
  (meeting_time % third_runner = 0) ∧
  (first_runner < third_runner) ∧
  (second_runner < third_runner) ∧
  (∀ t' < meeting_time, t' % first_runner = 0 → t' % second_runner = 0 → t' % third_runner ≠ 0)

theorem runner_solution : runner_problem 11 := by
  sorry

end NUMINAMATH_CALUDE_runner_solution_l1618_161816


namespace NUMINAMATH_CALUDE_sequence_max_value_l1618_161857

/-- The sequence a_n defined by -2n^2 + 29n + 3 for positive integers n has a maximum value of 108 -/
theorem sequence_max_value :
  ∃ (M : ℕ), ∀ (n : ℕ), n > 0 → (-2 * n^2 + 29 * n + 3 : ℤ) ≤ M ∧
  ∃ (k : ℕ), k > 0 ∧ (-2 * k^2 + 29 * k + 3 : ℤ) = M ∧ M = 108 :=
by
  sorry


end NUMINAMATH_CALUDE_sequence_max_value_l1618_161857


namespace NUMINAMATH_CALUDE_michael_work_time_l1618_161827

/-- Given that:
    - Michael and Adam can complete a work together in 20 days
    - They work together for 18 days, then Michael stops
    - Adam completes the remaining work in 10 days
    Prove that Michael can complete the work separately in 25 days -/
theorem michael_work_time (total_work : ℝ) (michael_rate : ℝ) (adam_rate : ℝ)
  (h1 : michael_rate + adam_rate = total_work / 20)
  (h2 : 18 * (michael_rate + adam_rate) = 9 / 10 * total_work)
  (h3 : adam_rate = total_work / 100) :
  michael_rate = total_work / 25 := by
  sorry

end NUMINAMATH_CALUDE_michael_work_time_l1618_161827


namespace NUMINAMATH_CALUDE_complementary_angle_of_30_28_l1618_161840

/-- Represents an angle in degrees and minutes -/
structure DegreeMinute where
  degree : ℕ
  minute : ℕ

/-- Converts DegreeMinute to a rational number -/
def DegreeMinute.toRational (dm : DegreeMinute) : ℚ :=
  dm.degree + dm.minute / 60

/-- Theorem: The complementary angle of 30°28' is 59°32' -/
theorem complementary_angle_of_30_28 :
  let angle1 : DegreeMinute := ⟨30, 28⟩
  let complement : DegreeMinute := ⟨59, 32⟩
  DegreeMinute.toRational angle1 + DegreeMinute.toRational complement = 90 := by
  sorry


end NUMINAMATH_CALUDE_complementary_angle_of_30_28_l1618_161840


namespace NUMINAMATH_CALUDE_first_player_wins_l1618_161833

/-- Represents a position on the rectangular table -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents the state of the game -/
structure GameState :=
  (table : Set Position)
  (occupied : Set Position)
  (currentPlayer : Bool)

/-- Checks if a position is valid on the table -/
def isValidPosition (state : GameState) (pos : Position) : Prop :=
  pos ∈ state.table ∧ pos ∉ state.occupied

/-- Represents a move in the game -/
def makeMove (state : GameState) (pos : Position) : GameState :=
  { state with
    occupied := state.occupied ∪ {pos},
    currentPlayer := ¬state.currentPlayer
  }

/-- Checks if the game is over (no more valid moves) -/
def isGameOver (state : GameState) : Prop :=
  ∀ pos, pos ∈ state.table → pos ∈ state.occupied

/-- Theorem: The first player has a winning strategy -/
theorem first_player_wins :
  ∀ (initialState : GameState),
  initialState.currentPlayer = true →
  ∃ (strategy : GameState → Position),
  (∀ state, isValidPosition state (strategy state)) →
  (∀ state, ¬isGameOver state → 
    ∃ (opponentMove : Position),
    isValidPosition state opponentMove →
    isGameOver (makeMove (makeMove state (strategy state)) opponentMove)) :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l1618_161833


namespace NUMINAMATH_CALUDE_function_properties_l1618_161850

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the function g
def g (a b c x : ℝ) : ℝ := |f a b c x - a|

-- Theorem statement
theorem function_properties
  (a b c : ℝ)
  (h1 : b > a)
  (h2 : ∀ x : ℝ, f a b c x ≥ 0) :
  (∀ x : ℝ, f 1 b c x = (x + 2)^2) ∧
  (0 < a ∧ a ≤ (2 + Real.sqrt 3) / 2 ↔
    ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-3*a) (-a) → x₂ ∈ Set.Icc (-3*a) (-a) →
      |g a b c x₁ - g a b c x₂| ≤ 2*a) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1618_161850


namespace NUMINAMATH_CALUDE_value_of_expression_l1618_161868

theorem value_of_expression (x : ℝ) (h : x^2 - 3*x - 12 = 0) : 3*x^2 - 9*x + 5 = 41 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1618_161868


namespace NUMINAMATH_CALUDE_sector_angle_when_length_equals_area_l1618_161845

/-- Theorem: For a circular sector with arc length and area both equal to 6,
    the central angle in radians is 3. -/
theorem sector_angle_when_length_equals_area (r : ℝ) (θ : ℝ) : 
  r * θ = 6 → -- arc length = r * θ = 6
  (1/2) * r^2 * θ = 6 → -- area = (1/2) * r^2 * θ = 6
  θ = 3 := by sorry

end NUMINAMATH_CALUDE_sector_angle_when_length_equals_area_l1618_161845


namespace NUMINAMATH_CALUDE_battery_charging_time_l1618_161846

/-- Represents the charging characteristics of a mobile battery -/
structure BatteryCharging where
  initial_rate : ℝ  -- Percentage charged per hour
  initial_time : ℝ  -- Time for initial charge in minutes
  additional_time : ℝ  -- Additional time to reach certain percentage in minutes

/-- Calculates the total charging time for a mobile battery -/
def total_charging_time (b : BatteryCharging) : ℝ :=
  b.initial_time + b.additional_time

/-- Theorem: The total charging time for the given battery is 255 minutes -/
theorem battery_charging_time :
  let b : BatteryCharging := {
    initial_rate := 20,
    initial_time := 60,
    additional_time := 195
  }
  total_charging_time b = 255 := by
  sorry

end NUMINAMATH_CALUDE_battery_charging_time_l1618_161846


namespace NUMINAMATH_CALUDE_trig_expression_equality_l1618_161867

theorem trig_expression_equality : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l1618_161867


namespace NUMINAMATH_CALUDE_fraction_division_specific_fraction_division_l1618_161895

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem specific_fraction_division :
  (3 : ℚ) / 7 / ((4 : ℚ) / 5) = 15 / 28 :=
by sorry

end NUMINAMATH_CALUDE_fraction_division_specific_fraction_division_l1618_161895


namespace NUMINAMATH_CALUDE_net_amount_correct_l1618_161849

/-- Maria's transactions at the fair -/
structure FairTransactions where
  initial_amount : ℕ
  spent_rides : ℕ
  won_booth : ℕ
  spent_food : ℕ
  found : ℕ
  final_amount : ℕ

/-- Calculate the net amount spent or gained at the fair -/
def net_amount (t : FairTransactions) : ℤ :=
  t.initial_amount - t.final_amount

/-- The theorem stating that the net amount is correct -/
theorem net_amount_correct (t : FairTransactions) 
  (h : t.initial_amount = 87 ∧ t.spent_rides = 25 ∧ t.won_booth = 10 ∧ 
       t.spent_food = 12 ∧ t.found = 5 ∧ t.final_amount = 16) : 
  net_amount t = 71 := by
  sorry

end NUMINAMATH_CALUDE_net_amount_correct_l1618_161849


namespace NUMINAMATH_CALUDE_bug_probability_after_seven_steps_l1618_161852

-- Define the probability function
def probability (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | m + 1 => 1/3 * (1 - probability m)

-- State the theorem
theorem bug_probability_after_seven_steps :
  probability 7 = 182 / 729 :=
sorry

end NUMINAMATH_CALUDE_bug_probability_after_seven_steps_l1618_161852


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l1618_161882

-- Define the variables and conditions
variable (x y a : ℝ)
variable (h1 : x * y ∈ Set.Icc (-π/4) (π/4))
variable (h2 : x^3 + Real.sin x - 2*a = 0)
variable (h3 : 4*y^3 + (1/2) * Real.sin (2*y) - a = 0)

-- State the theorem
theorem cos_x_plus_2y_equals_one : Real.cos (x + 2*y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l1618_161882


namespace NUMINAMATH_CALUDE_pipe_sale_result_l1618_161869

theorem pipe_sale_result : 
  ∀ (price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ),
    price = 1.20 →
    profit_percent = 20 →
    loss_percent = 20 →
    let profit_pipe_cost := price / (1 + profit_percent / 100)
    let loss_pipe_cost := price / (1 - loss_percent / 100)
    let total_cost := profit_pipe_cost + loss_pipe_cost
    let total_revenue := 2 * price
    total_revenue - total_cost = -0.10 := by
  sorry

end NUMINAMATH_CALUDE_pipe_sale_result_l1618_161869


namespace NUMINAMATH_CALUDE_christophers_speed_l1618_161831

/-- Given a distance of 5 miles and a time of 1.25 hours, the speed is 4 miles per hour -/
theorem christophers_speed (distance : ℝ) (time : ℝ) (speed : ℝ) :
  distance = 5 → time = 1.25 → speed = distance / time → speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_christophers_speed_l1618_161831


namespace NUMINAMATH_CALUDE_three_digit_segment_sum_l1618_161830

/-- Represents the number of horizontal and vertical segments for a digit --/
structure DigitSegments where
  horizontal : Nat
  vertical : Nat

/-- The set of all digits and their corresponding segment counts --/
def digit_segments : Fin 10 → DigitSegments := fun d =>
  match d with
  | 0 => ⟨2, 4⟩
  | 1 => ⟨0, 2⟩
  | 2 => ⟨2, 3⟩
  | 3 => ⟨3, 3⟩
  | 4 => ⟨1, 3⟩
  | 5 => ⟨2, 2⟩
  | 6 => ⟨1, 3⟩
  | 7 => ⟨1, 2⟩
  | 8 => ⟨3, 4⟩
  | 9 => ⟨2, 3⟩

theorem three_digit_segment_sum :
  ∃ (a b c : Fin 10),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (digit_segments a).horizontal + (digit_segments b).horizontal + (digit_segments c).horizontal = 5 ∧
    (digit_segments a).vertical + (digit_segments b).vertical + (digit_segments c).vertical = 10 ∧
    a.val + b.val + c.val = 9 :=
by sorry


end NUMINAMATH_CALUDE_three_digit_segment_sum_l1618_161830


namespace NUMINAMATH_CALUDE_set_B_when_a_is_2_A_equals_B_when_a_is_negative_one_l1618_161876

-- Define set A
def setA (a : ℝ) : Set ℝ := {x | (x - 2) * (x - 3*a - 1) < 0}

-- Define set B (domain of log(x))
def setB : Set ℝ := {x | x > 0}

-- Theorem 1: When a=2, B = {x | 2 < x < 7}
theorem set_B_when_a_is_2 :
  setB = {x : ℝ | 2 < x ∧ x < 7} ∧ ∀ x ∈ setB, (x - 2) * (x - 7) < 0 :=
sorry

-- Theorem 2: A = B only when a = -1
theorem A_equals_B_when_a_is_negative_one :
  ∃! a : ℝ, setA a = setB ∧ a = -1 :=
sorry

end NUMINAMATH_CALUDE_set_B_when_a_is_2_A_equals_B_when_a_is_negative_one_l1618_161876


namespace NUMINAMATH_CALUDE_problem_solution_l1618_161848

theorem problem_solution : ∃ x : ℝ, 4 * x - 4 = 2 * 4 + 20 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1618_161848


namespace NUMINAMATH_CALUDE_linear_inequality_m_value_l1618_161815

theorem linear_inequality_m_value (m : ℝ) : 
  (∀ x, ∃ a b, (m - 2) * x^(|m - 1|) - 3 > 6 ↔ a * x + b > 0) → 
  (m - 2 ≠ 0) → 
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_linear_inequality_m_value_l1618_161815


namespace NUMINAMATH_CALUDE_pauls_crayons_l1618_161872

theorem pauls_crayons (birthday_crayons : Float) (school_year_crayons : Float) (neighbor_crayons : Float)
  (h1 : birthday_crayons = 479.0)
  (h2 : school_year_crayons = 134.0)
  (h3 : neighbor_crayons = 256.0) :
  birthday_crayons + school_year_crayons + neighbor_crayons = 869.0 := by
  sorry

end NUMINAMATH_CALUDE_pauls_crayons_l1618_161872


namespace NUMINAMATH_CALUDE_carly_dog_grooming_l1618_161837

theorem carly_dog_grooming (total_nails : ℕ) (three_legged_dogs : ℕ) :
  total_nails = 164 →
  three_legged_dogs = 3 →
  ∃ (total_dogs : ℕ),
    total_dogs * 4 * 4 - three_legged_dogs * 4 = total_nails ∧
    total_dogs = 11 :=
by sorry

end NUMINAMATH_CALUDE_carly_dog_grooming_l1618_161837


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1618_161823

theorem solve_exponential_equation : 
  ∃ x : ℝ, (125 : ℝ)^4 = 5^x ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1618_161823


namespace NUMINAMATH_CALUDE_medicine_supply_duration_l1618_161879

/-- Represents the duration of the medicine supply in months -/
def medicine_duration (pills : ℕ) (pill_fraction : ℚ) (days_between_doses : ℕ) : ℚ :=
  let days_per_pill := (days_between_doses : ℚ) / pill_fraction
  let total_days := (pills : ℚ) * days_per_pill
  let days_per_month := 30
  total_days / days_per_month

/-- The theorem stating that the given medicine supply lasts 18 months -/
theorem medicine_supply_duration :
  medicine_duration 60 (1/3) 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_medicine_supply_duration_l1618_161879


namespace NUMINAMATH_CALUDE_product_of_reciprocals_l1618_161800

theorem product_of_reciprocals (a b : ℝ) : 
  a = 1 / (2 - Real.sqrt 3) → 
  b = 1 / (2 + Real.sqrt 3) → 
  a * b = 1 := by
sorry

end NUMINAMATH_CALUDE_product_of_reciprocals_l1618_161800


namespace NUMINAMATH_CALUDE_associates_hired_l1618_161810

theorem associates_hired (initial_partners initial_associates : ℕ) 
  (new_associates : ℕ) (hired_associates : ℕ) : 
  initial_partners = 18 →
  initial_partners * 63 = 2 * initial_associates →
  (initial_partners) * 34 = (initial_associates + hired_associates) →
  hired_associates = 45 := by sorry

end NUMINAMATH_CALUDE_associates_hired_l1618_161810


namespace NUMINAMATH_CALUDE_probability_factor_less_than_eight_l1618_161843

theorem probability_factor_less_than_eight (n : ℕ) (h : n = 90) :
  let factors := {d : ℕ | d > 0 ∧ n % d = 0}
  let factors_less_than_eight := {d ∈ factors | d < 8}
  Nat.card factors_less_than_eight / Nat.card factors = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_factor_less_than_eight_l1618_161843


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_for_modified_quadratic_l1618_161870

theorem no_positive_integer_solutions_for_modified_quadratic :
  ∀ A : ℕ, 1 ≤ A → A ≤ 9 →
    ¬∃ x : ℕ, x > 0 ∧ x^2 - (10 * A + 1) * x + (10 * A + A) = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_for_modified_quadratic_l1618_161870


namespace NUMINAMATH_CALUDE_max_books_borrowed_l1618_161866

theorem max_books_borrowed (total_students : Nat) (zero_books : Nat) (one_book : Nat) 
  (two_books : Nat) (three_books : Nat) (avg_books : Nat) (max_books : Nat) :
  total_students = 50 →
  zero_books = 4 →
  one_book = 15 →
  two_books = 9 →
  three_books = 7 →
  avg_books = 3 →
  max_books = 10 →
  ∃ (max_single : Nat),
    max_single ≤ max_books ∧
    max_single = 40 ∧
    (total_students * avg_books - (one_book + 2 * two_books + 3 * three_books)) % 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l1618_161866


namespace NUMINAMATH_CALUDE_dozen_pens_cost_l1618_161881

/-- The cost of a pen in rupees -/
def pen_cost : ℝ := sorry

/-- The cost of a pencil in rupees -/
def pencil_cost : ℝ := sorry

/-- The cost ratio of a pen to a pencil is 5:1 -/
axiom cost_ratio : pen_cost = 5 * pencil_cost

/-- The cost of 3 pens and 5 pencils is Rs. 200 -/
axiom total_cost : 3 * pen_cost + 5 * pencil_cost = 200

/-- The cost of one dozen pens is Rs. 600 -/
theorem dozen_pens_cost : 12 * pen_cost = 600 := by sorry

end NUMINAMATH_CALUDE_dozen_pens_cost_l1618_161881


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1618_161896

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ := seq.a 2 - seq.a 1

theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : 
  common_difference seq = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1618_161896


namespace NUMINAMATH_CALUDE_wendy_unrecycled_bags_l1618_161862

/-- Proves that Wendy did not recycle 2 bags given the problem conditions --/
theorem wendy_unrecycled_bags :
  ∀ (total_bags : ℕ) (points_per_bag : ℕ) (total_possible_points : ℕ),
    total_bags = 11 →
    points_per_bag = 5 →
    total_possible_points = 45 →
    total_bags - (total_possible_points / points_per_bag) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_wendy_unrecycled_bags_l1618_161862


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l1618_161874

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between a line and a plane -/
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicularity between two planes -/
def perpendicular_planes (p1 p2 : Plane3D) : Prop :=
  sorry

theorem line_perp_parallel_implies_planes_perp 
  (m : Line3D) (α β : Plane3D) :
  perpendicular m α → parallel m β → perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l1618_161874


namespace NUMINAMATH_CALUDE_min_a_value_l1618_161875

theorem min_a_value (a : ℝ) : 
  (∀ x : ℝ, |x + a| - |x + 1| ≤ 2*a) → a ≥ 1/3 :=
by
  sorry

end NUMINAMATH_CALUDE_min_a_value_l1618_161875


namespace NUMINAMATH_CALUDE_sin_150_degrees_l1618_161877

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l1618_161877


namespace NUMINAMATH_CALUDE_three_distinct_zeros_l1618_161811

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + a

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

/-- Theorem: For f to have three distinct real zeros, a must be in (1/4, +∞) -/
theorem three_distinct_zeros (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔
  a > (1/4 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_three_distinct_zeros_l1618_161811


namespace NUMINAMATH_CALUDE_no_real_solutions_l1618_161897

theorem no_real_solutions : 
  ∀ x : ℝ, (x - 3*x + 7)^2 + 1 ≠ -abs x := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1618_161897


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1618_161859

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ P : ℝ × ℝ, (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧
       (|P.1 - (-c)| + |P.1 - c| = 3*b) ∧
       (|P.1 - (-c)| * |P.1 - c| = 9/4 * a * b))
  (h4 : c^2 = a^2 + b^2) : 
  (c / a : ℝ) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1618_161859


namespace NUMINAMATH_CALUDE_shortest_segment_right_triangle_l1618_161801

theorem shortest_segment_right_triangle (a b c : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) 
  (h4 : a^2 + b^2 = c^2) : 
  ∃ (t : ℝ), t = 2 * Real.sqrt 3 ∧ 
  ∀ (x y : ℝ), x * y = (a * b) / 2 → 
  t ≤ Real.sqrt (x^2 + y^2 - 2 * x * y * (b / c)) := by
  sorry

end NUMINAMATH_CALUDE_shortest_segment_right_triangle_l1618_161801


namespace NUMINAMATH_CALUDE_track_length_l1618_161834

/-- The length of a circular track given specific running conditions -/
theorem track_length : 
  ∀ (L : ℝ),
  (∃ (d₁ d₂ : ℝ),
    d₁ = 100 ∧
    d₂ = 100 ∧
    d₁ + (L / 2 - d₁) = L / 2 ∧
    (L - d₁) + (L / 2 - d₁ + d₂) = L) →
  L = 200 :=
by sorry

end NUMINAMATH_CALUDE_track_length_l1618_161834


namespace NUMINAMATH_CALUDE_projected_strings_intersection_criterion_l1618_161863

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral in 2D space -/
structure Quadrilateral2D where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Calculates the ratio of two line segments -/
def segmentRatio (P Q R : Point2D) : ℝ := sorry

/-- Determines if two projected strings intersect in 3D space -/
def stringsIntersect (quad : Quadrilateral2D) (P Q R S : Point2D) : Prop :=
  let ratio1 := segmentRatio quad.A P quad.B
  let ratio2 := segmentRatio quad.B Q quad.C
  let ratio3 := segmentRatio quad.C R quad.D
  let ratio4 := segmentRatio quad.D S quad.A
  ratio1 * ratio2 * ratio3 * ratio4 = 1

/-- Theorem: Projected strings intersect in 3D iff their segment ratio product is 1 -/
theorem projected_strings_intersection_criterion 
  (quad : Quadrilateral2D) (P Q R S : Point2D) : 
  stringsIntersect quad P Q R S ↔ 
  segmentRatio quad.A P quad.B * 
  segmentRatio quad.B Q quad.C * 
  segmentRatio quad.C R quad.D * 
  segmentRatio quad.D S quad.A = 1 := by sorry

#check projected_strings_intersection_criterion

end NUMINAMATH_CALUDE_projected_strings_intersection_criterion_l1618_161863


namespace NUMINAMATH_CALUDE_linear_function_proof_l1618_161853

theorem linear_function_proof (f : ℝ → ℝ) :
  (∀ x y : ℝ, ∃ k b : ℝ, f x = k * x + b) →
  (∀ x : ℝ, 3 * f (x + 1) - f x = 2 * x + 9) →
  (∀ x : ℝ, f x = x + 3) := by
sorry

end NUMINAMATH_CALUDE_linear_function_proof_l1618_161853


namespace NUMINAMATH_CALUDE_coin_array_digit_sum_l1618_161814

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The total number of coins in a triangular array with n rows -/
def triangular_sum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem coin_array_digit_sum :
  ∃ (N : ℕ), triangular_sum N = 3003 ∧ sum_of_digits N = 14 := by
  sorry

end NUMINAMATH_CALUDE_coin_array_digit_sum_l1618_161814


namespace NUMINAMATH_CALUDE_non_monotonic_k_range_l1618_161856

-- Define the function f(x)
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- Define the interval
def interval : Set ℝ := Set.Icc 5 8

-- Define the property of non-monotonicity
def is_non_monotonic (g : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∃ (x y z : ℝ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x < y ∧ y < z ∧ 
  ((g x < g y ∧ g y > g z) ∨ (g x > g y ∧ g y < g z))

-- State the theorem
theorem non_monotonic_k_range :
  ∀ k : ℝ, is_non_monotonic (f k) interval ↔ k ∈ Set.Ioo 40 64 :=
sorry

end NUMINAMATH_CALUDE_non_monotonic_k_range_l1618_161856


namespace NUMINAMATH_CALUDE_quadratic_roots_in_fourth_quadrant_l1618_161864

/-- A point in the fourth quadrant -/
structure FourthQuadrantPoint where
  x : ℝ
  y : ℝ
  x_pos : 0 < x
  y_neg : y < 0

/-- Quadratic equation coefficients -/
structure QuadraticCoeffs where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The quadratic equation has two distinct real roots -/
def has_two_distinct_real_roots (q : QuadraticCoeffs) : Prop :=
  0 < q.b ^ 2 - 4 * q.a * q.c

theorem quadratic_roots_in_fourth_quadrant 
  (p : FourthQuadrantPoint) (q : QuadraticCoeffs) 
  (h : p.x = q.a ∧ p.y = q.c) : 
  has_two_distinct_real_roots q := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_in_fourth_quadrant_l1618_161864


namespace NUMINAMATH_CALUDE_brown_ball_weight_l1618_161828

theorem brown_ball_weight (blue_weight : ℝ) (total_weight : ℝ) (brown_weight : ℝ) :
  blue_weight = 6 →
  total_weight = 9.12 →
  total_weight = blue_weight + brown_weight →
  brown_weight = 3.12 :=
by
  sorry

end NUMINAMATH_CALUDE_brown_ball_weight_l1618_161828


namespace NUMINAMATH_CALUDE_difference_percentages_l1618_161824

theorem difference_percentages : (800 * 75 / 100) - (1200 * 7 / 8) = 450 := by
  sorry

end NUMINAMATH_CALUDE_difference_percentages_l1618_161824
