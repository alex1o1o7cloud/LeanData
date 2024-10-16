import Mathlib

namespace NUMINAMATH_CALUDE_mrs_hilt_impressed_fans_l1532_153260

/-- The number of sets of bleachers -/
def num_bleachers : ℕ := 3

/-- The number of fans on each set of bleachers -/
def fans_per_bleacher : ℕ := 812

/-- The total number of fans Mrs. Hilt impressed -/
def total_fans : ℕ := num_bleachers * fans_per_bleacher

theorem mrs_hilt_impressed_fans : total_fans = 2436 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_impressed_fans_l1532_153260


namespace NUMINAMATH_CALUDE_cubic_function_extrema_l1532_153263

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 27

def f_derivative (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_extrema (a b : ℝ) :
  f_derivative a b (-1) = 0 ∧ f_derivative a b 3 = 0 → a = -3 ∧ b = -9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_l1532_153263


namespace NUMINAMATH_CALUDE_min_value_theorem_l1532_153223

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 3/2) :
  (∀ x y : ℝ, x > 0 → y > 1 → x + y = 3/2 → 2/x + 1/(y-1) ≥ 2/a + 1/(b-1)) ∧
  2/a + 1/(b-1) = 6 + 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1532_153223


namespace NUMINAMATH_CALUDE_morse_code_symbols_l1532_153216

/-- The number of possible symbols (dot, dash, space) -/
def num_symbols : ℕ := 3

/-- The maximum length of a sequence -/
def max_length : ℕ := 3

/-- Calculates the number of distinct sequences for a given length -/
def sequences_of_length (n : ℕ) : ℕ := num_symbols ^ n

/-- The total number of distinct symbols that can be represented -/
def total_distinct_symbols : ℕ :=
  (sequences_of_length 1) + (sequences_of_length 2) + (sequences_of_length 3)

/-- Theorem: The total number of distinct symbols that can be represented is 39 -/
theorem morse_code_symbols : total_distinct_symbols = 39 := by
  sorry

end NUMINAMATH_CALUDE_morse_code_symbols_l1532_153216


namespace NUMINAMATH_CALUDE_inequality_proof_l1532_153273

theorem inequality_proof (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) : 
  x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1) ≤ 1 - (1 - x) * (1 - y) * (1 - z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1532_153273


namespace NUMINAMATH_CALUDE_unique_solution_A7B_l1532_153204

/-- Given a three-digit number of the form A7B, where A and B are single digits,
    prove that B = 2 is the unique solution satisfying A7B + 23 = 695 and 27B is a three-digit number. -/
theorem unique_solution_A7B : ∃! B : ℕ, 
  (∃ A : ℕ, A < 10 ∧ B < 10 ∧ (100 * A + 70 + B) + 23 = 695) ∧ 
  200 ≤ 27 * B ∧ 27 * B < 1000 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_A7B_l1532_153204


namespace NUMINAMATH_CALUDE_ages_sum_five_years_ago_l1532_153232

/-- Proves that the sum of Angela's and Beth's ages 5 years ago was 39 years -/
theorem ages_sum_five_years_ago : 
  ∀ (angela_age beth_age : ℕ),
  angela_age = 4 * beth_age →
  angela_age + 5 = 44 →
  (angela_age - 5) + (beth_age - 5) = 39 :=
by
  sorry

end NUMINAMATH_CALUDE_ages_sum_five_years_ago_l1532_153232


namespace NUMINAMATH_CALUDE_cube_cut_surface_area_l1532_153262

/-- Calculates the total surface area of small blocks after cutting a cube -/
def total_surface_area (edge_length : ℝ) (horizontal_cuts : ℕ) (vertical_cuts : ℕ) : ℝ :=
  let original_surface_area := 6 * edge_length^2
  let horizontal_new_area := 2 * edge_length^2 * (2 * horizontal_cuts)
  let vertical_new_area := 2 * edge_length^2 * (2 * vertical_cuts)
  original_surface_area + horizontal_new_area + vertical_new_area

/-- Theorem: The total surface area of all small blocks after cutting a cube with edge length 2,
    4 horizontal cuts, and 5 vertical cuts, is equal to 96 square units -/
theorem cube_cut_surface_area :
  total_surface_area 2 4 5 = 96 := by sorry

end NUMINAMATH_CALUDE_cube_cut_surface_area_l1532_153262


namespace NUMINAMATH_CALUDE_gasoline_price_change_l1532_153280

theorem gasoline_price_change (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  let new_price := 1.25 * P
  let new_quantity := 0.88 * Q
  let original_spending := P * Q
  let new_spending := new_price * new_quantity
  (new_spending - original_spending) / original_spending = 0.1 := by
sorry

end NUMINAMATH_CALUDE_gasoline_price_change_l1532_153280


namespace NUMINAMATH_CALUDE_max_areas_formula_max_areas_for_n_3_l1532_153269

/-- Represents a circular disk divided by radii and secant lines -/
structure DividedDisk where
  n : ℕ
  radii_count : ℕ
  secant_lines : ℕ
  h_positive : n > 0
  h_radii : radii_count = 2 * n
  h_secants : secant_lines = 2

/-- Calculates the maximum number of non-overlapping areas in a divided disk -/
def max_areas (d : DividedDisk) : ℕ :=
  4 * d.n + 4

/-- Theorem stating the maximum number of non-overlapping areas -/
theorem max_areas_formula (d : DividedDisk) :
  max_areas d = 4 * d.n + 4 :=
by sorry

/-- Specific case for n = 3 -/
theorem max_areas_for_n_3 :
  ∃ (d : DividedDisk), d.n = 3 ∧ max_areas d = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_areas_formula_max_areas_for_n_3_l1532_153269


namespace NUMINAMATH_CALUDE_zeros_of_f_with_fixed_points_range_of_b_with_no_fixed_points_l1532_153236

-- Define the function f(x)
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- Theorem 1
theorem zeros_of_f_with_fixed_points (b c : ℝ) :
  (f b c (-3) = -3) ∧ (f b c 2 = 2) →
  (∃ x : ℝ, f b c x = 0) ∧ 
  (∀ x : ℝ, f b c x = 0 ↔ (x = -1 + Real.sqrt 7 ∨ x = -1 - Real.sqrt 7)) :=
sorry

-- Theorem 2
theorem range_of_b_with_no_fixed_points :
  (∀ b : ℝ, ∀ x : ℝ, f b (b^2/4) x ≠ x) →
  (∀ b : ℝ, (b < -1 ∨ b > 1/3) ↔ (∀ x : ℝ, f b (b^2/4) x ≠ x)) :=
sorry

end NUMINAMATH_CALUDE_zeros_of_f_with_fixed_points_range_of_b_with_no_fixed_points_l1532_153236


namespace NUMINAMATH_CALUDE_f_properties_l1532_153200

open Real

noncomputable def f (x : ℝ) : ℝ := 2 / x + log x

theorem f_properties :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f x ≥ f 2) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f x₁ = f x₂ → x₁ + x₂ > 4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1532_153200


namespace NUMINAMATH_CALUDE_bert_toy_phones_l1532_153220

/-- 
Proves that Bert sold 8 toy phones given the conditions of the problem.
-/
theorem bert_toy_phones :
  ∀ (bert_phones : ℕ),
  (18 * bert_phones = 20 * 7 + 4) →
  bert_phones = 8 := by
  sorry

end NUMINAMATH_CALUDE_bert_toy_phones_l1532_153220


namespace NUMINAMATH_CALUDE_correct_contribution_l1532_153258

/-- Represents the amount spent by each person -/
structure Expenses where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Represents the contribution from Person C to others -/
structure Contribution where
  to_a : ℚ
  to_b : ℚ

def calculate_contribution (e : Expenses) : Contribution :=
  { to_a := 6,
    to_b := 3 }

theorem correct_contribution (e : Expenses) :
  e.b = 12/13 * e.a ∧ 
  e.c = 2/3 * e.b ∧ 
  calculate_contribution e = { to_a := 6, to_b := 3 } :=
by sorry

#check correct_contribution

end NUMINAMATH_CALUDE_correct_contribution_l1532_153258


namespace NUMINAMATH_CALUDE_tan_sum_17_28_l1532_153274

theorem tan_sum_17_28 : 
  (Real.tan (17 * π / 180) + Real.tan (28 * π / 180)) / 
  (1 - Real.tan (17 * π / 180) * Real.tan (28 * π / 180)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_tan_sum_17_28_l1532_153274


namespace NUMINAMATH_CALUDE_product_equals_243_l1532_153237

theorem product_equals_243 : 
  (1 / 3 : ℚ) * 9 * (1 / 27 : ℚ) * 81 * (1 / 243 : ℚ) * 729 * (1 / 2187 : ℚ) * 6561 * (1 / 19683 : ℚ) * 59049 = 243 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_243_l1532_153237


namespace NUMINAMATH_CALUDE_fraction_equality_l1532_153252

theorem fraction_equality : (1 : ℚ) / 2 = 4 / 8 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1532_153252


namespace NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_range_of_a_l1532_153284

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Theorem 1: A ∪ B = {x | -1 ≤ x}
theorem union_A_B : A ∪ B = {x : ℝ | -1 ≤ x} := by sorry

-- Theorem 2: A ∩ (Cᴿ B) = {x | -1 ≤ x < 2}
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

-- Theorem 3: If B ∪ C = C, then a ≤ 3
theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_range_of_a_l1532_153284


namespace NUMINAMATH_CALUDE_sin_zero_degrees_l1532_153210

theorem sin_zero_degrees : Real.sin (0 * π / 180) = 0 := by sorry

end NUMINAMATH_CALUDE_sin_zero_degrees_l1532_153210


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1532_153254

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 100 ∧ x - y = 8 → x * y = 2484 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1532_153254


namespace NUMINAMATH_CALUDE_horizontal_distance_P_Q_l1532_153249

/-- The curve function -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

/-- Theorem stating the horizontal distance between P and Q -/
theorem horizontal_distance_P_Q : 
  ∀ (xp xq : ℝ), 
  f xp = 8 → 
  f xq = -1 → 
  (∀ x : ℝ, f x = -1 → |x - xp| ≥ |xq - xp|) → 
  |xq - xp| = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_horizontal_distance_P_Q_l1532_153249


namespace NUMINAMATH_CALUDE_min_abc_value_l1532_153278

-- Define the set M
def M : Set ℝ := {x | 2/3 < x ∧ x < 2}

-- Define t as the largest positive integer in M
def t : ℕ := 1

-- Theorem statement
theorem min_abc_value (a b c : ℝ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (h_abc : (a - 1) * (b - 1) * (c - 1) = t) :
  a * b * c ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_min_abc_value_l1532_153278


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l1532_153299

theorem modular_congruence_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 27 ∧ n ≡ -3456 [ZMOD 28] ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l1532_153299


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l1532_153224

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- A function that checks if three numbers form a valid triangle -/
def isValidTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- The theorem stating the smallest possible perimeter of a scalene triangle
    with prime side lengths greater than 3 and prime perimeter -/
theorem smallest_prime_perimeter_scalene_triangle :
  ∃ (a b c : ℕ),
    a < b ∧ b < c ∧
    isPrime a ∧ isPrime b ∧ isPrime c ∧
    a > 3 ∧ b > 3 ∧ c > 3 ∧
    isValidTriangle a b c ∧
    isPrime (a + b + c) ∧
    (∀ (x y z : ℕ),
      x < y ∧ y < z ∧
      isPrime x ∧ isPrime y ∧ isPrime z ∧
      x > 3 ∧ y > 3 ∧ z > 3 ∧
      isValidTriangle x y z ∧
      isPrime (x + y + z) →
      a + b + c ≤ x + y + z) ∧
    a + b + c = 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l1532_153224


namespace NUMINAMATH_CALUDE_cookies_remaining_l1532_153267

theorem cookies_remaining (total_taken : ℕ) (h1 : total_taken = 11) 
  (h2 : total_taken * 2 = total_taken + total_taken) : 
  total_taken = total_taken * 2 - total_taken := by
  sorry

end NUMINAMATH_CALUDE_cookies_remaining_l1532_153267


namespace NUMINAMATH_CALUDE_house_store_transaction_loss_l1532_153282

theorem house_store_transaction_loss (house_price store_price : ℝ) : 
  house_price * (1 - 0.2) = 12000 →
  store_price * (1 + 0.2) = 12000 →
  house_price + store_price - 2 * 12000 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_house_store_transaction_loss_l1532_153282


namespace NUMINAMATH_CALUDE_map_distance_calculation_l1532_153201

/-- Given a map with a scale of 1:1000000 and two points A and B that are 8 cm apart on the map,
    the actual distance between A and B is 80 km. -/
theorem map_distance_calculation (scale : ℚ) (map_distance : ℚ) (actual_distance : ℚ) :
  scale = 1 / 1000000 →
  map_distance = 8 →
  actual_distance = map_distance / scale →
  actual_distance = 80 * 100000 := by
  sorry


end NUMINAMATH_CALUDE_map_distance_calculation_l1532_153201


namespace NUMINAMATH_CALUDE_dihedral_angle_line_relationship_l1532_153229

/-- A dihedral angle with edge l and planes α and β -/
structure DihedralAngle where
  l : Line
  α : Plane
  β : Plane

/-- A right dihedral angle -/
def is_right_dihedral (d : DihedralAngle) : Prop := sorry

/-- A line a in plane α -/
def line_in_plane_α (d : DihedralAngle) (a : Line) : Prop := sorry

/-- A line b in plane β -/
def line_in_plane_β (d : DihedralAngle) (b : Line) : Prop := sorry

/-- Line not perpendicular to edge l -/
def not_perp_to_edge (d : DihedralAngle) (m : Line) : Prop := sorry

/-- Two lines are parallel -/
def are_parallel (m n : Line) : Prop := sorry

/-- Two lines are perpendicular -/
def are_perpendicular (m n : Line) : Prop := sorry

theorem dihedral_angle_line_relationship (d : DihedralAngle) (a b : Line) 
  (h_right : is_right_dihedral d)
  (h_a_in_α : line_in_plane_α d a)
  (h_b_in_β : line_in_plane_β d b)
  (h_a_not_perp : not_perp_to_edge d a)
  (h_b_not_perp : not_perp_to_edge d b) :
  (∃ (a' b' : Line), line_in_plane_α d a' ∧ line_in_plane_β d b' ∧ 
    not_perp_to_edge d a' ∧ not_perp_to_edge d b' ∧ are_parallel a' b') ∧ 
  (∀ (a' b' : Line), line_in_plane_α d a' → line_in_plane_β d b' → 
    not_perp_to_edge d a' → not_perp_to_edge d b' → ¬ are_perpendicular a' b') :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_line_relationship_l1532_153229


namespace NUMINAMATH_CALUDE_weekend_sales_total_l1532_153209

/-- Calculates the total money made from jewelry sales --/
def total_money_made (necklace_price bracelet_price earring_pair_price ensemble_price : ℚ)
  (necklaces_sold bracelets_sold earring_pairs_sold ensembles_sold : ℕ) : ℚ :=
  necklace_price * necklaces_sold +
  bracelet_price * bracelets_sold +
  earring_pair_price * earring_pairs_sold +
  ensemble_price * ensembles_sold

/-- Proves that the total money made over the weekend is $565.00 --/
theorem weekend_sales_total :
  let necklace_price : ℚ := 25
  let bracelet_price : ℚ := 15
  let earring_pair_price : ℚ := 10
  let ensemble_price : ℚ := 45
  let necklaces_sold : ℕ := 5
  let bracelets_sold : ℕ := 10
  let earring_pairs_sold : ℕ := 20
  let ensembles_sold : ℕ := 2
  total_money_made necklace_price bracelet_price earring_pair_price ensemble_price
    necklaces_sold bracelets_sold earring_pairs_sold ensembles_sold = 565 := by
  sorry

end NUMINAMATH_CALUDE_weekend_sales_total_l1532_153209


namespace NUMINAMATH_CALUDE_ken_steak_change_l1532_153298

/-- Calculates the change Ken will receive when buying steak -/
def calculate_change (price_per_pound : ℕ) (pounds_bought : ℕ) (payment : ℕ) : ℕ :=
  payment - (price_per_pound * pounds_bought)

/-- Proves that Ken will receive $6 in change -/
theorem ken_steak_change :
  let price_per_pound : ℕ := 7
  let pounds_bought : ℕ := 2
  let payment : ℕ := 20
  calculate_change price_per_pound pounds_bought payment = 6 := by
sorry

end NUMINAMATH_CALUDE_ken_steak_change_l1532_153298


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l1532_153231

-- Define the points
variable (A B C D O M N : ℝ × ℝ)

-- Define the triangle area function
def triangle_area (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem quadrilateral_inequality 
  (h_convex : sorry) -- ABCD is a convex quadrilateral
  (h_intersect : sorry) -- AC and BD intersect at O
  (h_line : sorry) -- Line through O intersects AB at M and CD at N
  (h_ineq1 : triangle_area O M B > triangle_area O N D)
  (h_ineq2 : triangle_area O C N > triangle_area O A M) :
  triangle_area O A M + triangle_area O B C + triangle_area O N D >
  triangle_area O D A + triangle_area O M B + triangle_area O C N :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l1532_153231


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_equals_256_l1532_153239

/-- Given a cubic polynomial with roots p, q, r, prove that the sum of reciprocals of 
    partial fraction decomposition coefficients equals 256. -/
theorem sum_of_reciprocals_equals_256 
  (p q r : ℝ) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (h_roots : x^3 - 27*x^2 + 98*x - 72 = (x - p) * (x - q) * (x - r)) 
  (A B C : ℝ) 
  (h_partial_fraction : ∀ (s : ℝ), s ≠ p → s ≠ q → s ≠ r → 
    1 / (s^3 - 27*s^2 + 98*s - 72) = A / (s - p) + B / (s - q) + C / (s - r)) :
  1/A + 1/B + 1/C = 256 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_equals_256_l1532_153239


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l1532_153291

theorem cubic_root_equation_solution :
  ∀ x : ℝ, (x^(1/3) = 15 / (8 - x^(1/3))) ↔ (x = 27 ∨ x = 125) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l1532_153291


namespace NUMINAMATH_CALUDE_equation_equivalence_l1532_153255

-- Define the original equation
def original_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + y^2) + Real.sqrt ((x + 4)^2 + y^2) = 10

-- Define the simplified equation
def simplified_equation (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

-- Theorem statement
theorem equation_equivalence :
  ∀ x y : ℝ, original_equation x y ↔ simplified_equation x y :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1532_153255


namespace NUMINAMATH_CALUDE_last_eight_digits_of_product_l1532_153246

theorem last_eight_digits_of_product : ∃ n : ℕ, 
  11 * 101 * 1001 * 10001 * 100001 * 1000001 * 111 ≡ 19754321 [MOD 100000000] :=
by sorry

end NUMINAMATH_CALUDE_last_eight_digits_of_product_l1532_153246


namespace NUMINAMATH_CALUDE_coin_combination_theorem_l1532_153275

/-- Represents the number of coins of each denomination -/
structure CoinCounts where
  fiveCent : ℕ
  tenCent : ℕ
  twentyFiveCent : ℕ

/-- Calculates the number of different values obtainable from a given set of coins -/
def differentValues (coins : CoinCounts) : ℕ := sorry

theorem coin_combination_theorem (coins : CoinCounts) :
  coins.fiveCent + coins.tenCent + coins.twentyFiveCent = 15 →
  differentValues coins = 23 →
  coins.twentyFiveCent = 3 := by sorry

end NUMINAMATH_CALUDE_coin_combination_theorem_l1532_153275


namespace NUMINAMATH_CALUDE_fraction_product_l1532_153286

theorem fraction_product : (2 : ℚ) / 3 * (4 : ℚ) / 7 * (9 : ℚ) / 13 = (24 : ℚ) / 91 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l1532_153286


namespace NUMINAMATH_CALUDE_flowerbed_length_difference_l1532_153250

theorem flowerbed_length_difference (width length : ℝ) : 
  width = 4 →
  2 * length + 2 * width = 22 →
  2 * width - length = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_flowerbed_length_difference_l1532_153250


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l1532_153214

/-- A line in the xy-plane represented by its slope and y-intercept. -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Checks if two lines are parallel. -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Checks if a point lies on a line. -/
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.y_intercept

/-- The given line y = -3x + 7 -/
def given_line : Line :=
  { slope := -3, y_intercept := 7 }

theorem y_intercept_of_parallel_line :
  ∀ (b : Line),
    are_parallel b given_line →
    point_on_line b 5 (-2) →
    b.y_intercept = 13 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l1532_153214


namespace NUMINAMATH_CALUDE_cinema_renovation_unique_solution_l1532_153206

theorem cinema_renovation_unique_solution :
  ∃! (x y : ℕ), 
    x > 0 ∧ 
    y > 20 ∧ 
    y * (2 * x + y - 1) = 4008 := by
  sorry

end NUMINAMATH_CALUDE_cinema_renovation_unique_solution_l1532_153206


namespace NUMINAMATH_CALUDE_students_per_row_l1532_153202

theorem students_per_row (total_students : ℕ) (rows : ℕ) (leftover : ℕ) 
  (h1 : total_students = 45)
  (h2 : rows = 11)
  (h3 : leftover = 1)
  (h4 : total_students = rows * (total_students / rows) + leftover) :
  total_students / rows = 4 := by
  sorry

end NUMINAMATH_CALUDE_students_per_row_l1532_153202


namespace NUMINAMATH_CALUDE_intersection_point_l1532_153245

/-- A parabola in the xy-plane defined by y^2 - 4y + x = 6 -/
def parabola (x y : ℝ) : Prop := y^2 - 4*y + x = 6

/-- A vertical line in the xy-plane defined by x = k -/
def vertical_line (k x : ℝ) : Prop := x = k

/-- The condition for a quadratic equation ay^2 + by + c = 0 to have exactly one solution -/
def has_unique_solution (a b c : ℝ) : Prop := b^2 - 4*a*c = 0

theorem intersection_point (k : ℝ) : 
  (∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ vertical_line k p.1) ↔ k = 10 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l1532_153245


namespace NUMINAMATH_CALUDE_solution_is_i_div_3_l1532_153234

/-- The imaginary unit i, where i^2 = -1 -/
noncomputable def i : ℂ := Complex.I

/-- The equation to be solved -/
def equation (x : ℂ) : Prop := 3 + i * x = 5 - 2 * i * x

/-- The theorem stating that i/3 is the solution to the equation -/
theorem solution_is_i_div_3 : equation (i / 3) := by
  sorry

end NUMINAMATH_CALUDE_solution_is_i_div_3_l1532_153234


namespace NUMINAMATH_CALUDE_labourer_income_is_78_l1532_153261

/-- Represents the financial situation of a labourer over a 10-month period. -/
structure LabourerFinances where
  monthly_income : ℝ
  initial_debt : ℝ
  first_period_months : ℕ := 6
  second_period_months : ℕ := 4
  first_period_monthly_expense : ℝ := 85
  second_period_monthly_expense : ℝ := 60
  final_savings : ℝ := 30

/-- The labourer's financial situation satisfies the given conditions. -/
def satisfies_conditions (f : LabourerFinances) : Prop :=
  f.first_period_months * f.monthly_income - f.initial_debt = 
    f.first_period_months * f.first_period_monthly_expense ∧
  f.second_period_months * f.monthly_income = 
    f.second_period_months * f.second_period_monthly_expense + f.initial_debt + f.final_savings

/-- The labourer's monthly income is 78 given the conditions. -/
theorem labourer_income_is_78 (f : LabourerFinances) 
  (h : satisfies_conditions f) : f.monthly_income = 78 := by
  sorry

end NUMINAMATH_CALUDE_labourer_income_is_78_l1532_153261


namespace NUMINAMATH_CALUDE_problem_solution_l1532_153211

theorem problem_solution (a b : ℝ) (h1 : a + b = 4) (h2 : a * b = 1) : 
  (a - b)^2 = 12 ∧ a^5*b - 2*a^4*b^4 + a*b^5 = 192 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1532_153211


namespace NUMINAMATH_CALUDE_crystal_mass_ratio_l1532_153297

theorem crystal_mass_ratio (x y : ℝ) 
  (h1 : x * 0.04 = y * 0.05 * 7 / 3)  -- x increases in 3 months as much as y does in 7 months
  (h2 : x * 1.04 = x + x * 0.04)      -- x increases by 4% in a year
  (h3 : y * 1.05 = y + y * 0.05)      -- y increases by 5% in a year
  : x / y = 35 / 12 := by
  sorry

end NUMINAMATH_CALUDE_crystal_mass_ratio_l1532_153297


namespace NUMINAMATH_CALUDE_complex_magnitude_l1532_153266

theorem complex_magnitude (z : ℂ) (h : z * (1 - Complex.I) = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1532_153266


namespace NUMINAMATH_CALUDE_time_for_type_A_is_60_l1532_153279

/-- Represents the time allocation for an examination with different problem types. -/
structure ExamTime where
  totalQuestions : ℕ
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ
  totalTime : ℕ
  lastHour : ℕ

/-- Calculates the time spent on Type A problems in the examination. -/
def timeForTypeA (e : ExamTime) : ℕ :=
  let typeB_time := (e.lastHour * 2) / e.typeC
  (e.typeA * typeB_time * 2)

/-- Theorem stating that the time spent on Type A problems is 60 minutes. -/
theorem time_for_type_A_is_60 (e : ExamTime) 
  (h1 : e.totalQuestions = 200)
  (h2 : e.typeA = 20)
  (h3 : e.typeB = 100)
  (h4 : e.typeC = 80)
  (h5 : e.totalTime = 180)
  (h6 : e.lastHour = 60) :
  timeForTypeA e = 60 := by
  sorry

end NUMINAMATH_CALUDE_time_for_type_A_is_60_l1532_153279


namespace NUMINAMATH_CALUDE_divisibility_property_l1532_153256

theorem divisibility_property (m : ℕ) (hm : m > 0) :
  ∃ q : Polynomial ℤ, (x + 1)^(2*m) - x^(2*m) - 2*x - 1 = x * (x + 1) * (2*x + 1) * q :=
sorry

end NUMINAMATH_CALUDE_divisibility_property_l1532_153256


namespace NUMINAMATH_CALUDE_function_property_l1532_153221

/-- Given a function f: ℕ → ℕ satisfying the property that
    for all positive integers a, b, n such that a + b = 3^n,
    f(a) + f(b) = 2n^2, prove that f(3003) = 44 -/
theorem function_property (f : ℕ → ℕ) 
  (h : ∀ (a b n : ℕ), 0 < a → 0 < b → 0 < n → a + b = 3^n → f a + f b = 2*n^2) :
  f 3003 = 44 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1532_153221


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1532_153253

theorem inequality_equivalence (x : ℝ) : 
  |((8 - x) / 4)|^2 < 4 ↔ 0 < x ∧ x < 16 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1532_153253


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1532_153271

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 6 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

-- Define the point the hyperbola passes through
def passes_through : Prop := hyperbola 3 (-2 * Real.sqrt 3)

-- Define the intersection line
def intersection_line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 3)

-- State the theorem
theorem hyperbola_properties :
  (∀ x y, asymptotes x y → hyperbola x y) ∧
  passes_through ∧
  (∃ A B : ℝ × ℝ,
    hyperbola A.1 A.2 ∧
    hyperbola B.1 B.2 ∧
    intersection_line A.1 A.2 ∧
    intersection_line B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1532_153271


namespace NUMINAMATH_CALUDE_rectangle_width_l1532_153207

/-- Given a rectangle with area 300 square meters and perimeter 70 meters, prove its width is 15 meters. -/
theorem rectangle_width (length width : ℝ) : 
  length * width = 300 ∧ 
  2 * (length + width) = 70 → 
  width = 15 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l1532_153207


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l1532_153257

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    where a line passing through its left vertex with slope k 
    intersects the ellipse at a point whose x-coordinate is the 
    distance from the center to the focus, prove that the 
    eccentricity e of the ellipse is between 1/2 and 2/3 
    when k is between 1/3 and 1/2. -/
theorem ellipse_eccentricity_range (a b : ℝ) (k : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : 1/3 < k) (h4 : k < 1/2) :
  let e := Real.sqrt (1 - b^2 / a^2)
  ∃ (x y : ℝ), 
    x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    y = k * (x + a) ∧
    x = a * e ∧
    1/2 < e ∧ e < 2/3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l1532_153257


namespace NUMINAMATH_CALUDE_gcd_459_357_l1532_153244

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l1532_153244


namespace NUMINAMATH_CALUDE_sum_interior_angles_formula_l1532_153251

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: For a polygon with n sides (n > 2), the sum of interior angles is 180° × (n-2) -/
theorem sum_interior_angles_formula {n : ℕ} (h : n > 2) :
  sum_interior_angles n = 180 * (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_formula_l1532_153251


namespace NUMINAMATH_CALUDE_total_age_l1532_153276

def kate_age : ℕ := 19
def maggie_age : ℕ := 17
def sue_age : ℕ := 12

theorem total_age : kate_age + maggie_age + sue_age = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_age_l1532_153276


namespace NUMINAMATH_CALUDE_f_properties_l1532_153233

-- Define the function f
def f (x : ℝ) : ℝ := |x + 7| + |x - 1|

-- Theorem statement
theorem f_properties :
  (∀ x : ℝ, f x ≥ 8) ∧
  (∀ x : ℝ, |x - 3| - 2*x ≤ 4 ↔ x ≥ -1/3) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1532_153233


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l1532_153288

theorem solve_cubic_equation (x : ℝ) : 
  (x^3 * 6^3) / 432 = 864 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l1532_153288


namespace NUMINAMATH_CALUDE_paper_completion_days_l1532_153270

theorem paper_completion_days (total_pages : ℕ) (pages_per_day : ℕ) (days : ℕ) : 
  total_pages = 81 → pages_per_day = 27 → days * pages_per_day = total_pages → days = 3 := by
  sorry

end NUMINAMATH_CALUDE_paper_completion_days_l1532_153270


namespace NUMINAMATH_CALUDE_erroneous_product_l1532_153238

/-- Given two positive integers a and b, where a is a two-digit number,
    if reversing the digits of a and multiplying by b, then adding 2, results in 240,
    then the actual product of a and b is 301. -/
theorem erroneous_product (a b : ℕ) : 
  a > 9 ∧ a < 100 →  -- a is a two-digit number
  b > 0 →  -- b is positive
  (((a % 10) * 10 + (a / 10)) * b + 2 = 240) →  -- erroneous calculation
  a * b = 301 := by
sorry

end NUMINAMATH_CALUDE_erroneous_product_l1532_153238


namespace NUMINAMATH_CALUDE_cubic_function_property_l1532_153240

/-- Given a cubic function f(x) = ax³ + bx + 1 where ab ≠ 0, 
    if f(2016) = k, then f(-2016) = 2-k -/
theorem cubic_function_property (a b k : ℝ) (h1 : a * b ≠ 0) :
  let f := λ x : ℝ => a * x^3 + b * x + 1
  f 2016 = k → f (-2016) = 2 - k := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l1532_153240


namespace NUMINAMATH_CALUDE_substitution_ways_mod_1000_l1532_153296

/-- Represents the number of players in a soccer team --/
def total_players : ℕ := 22

/-- Represents the number of starting players --/
def starting_players : ℕ := 11

/-- Represents the maximum number of substitutions allowed --/
def max_substitutions : ℕ := 4

/-- Calculates the number of ways to make substitutions in a soccer game --/
def substitution_ways : ℕ := 
  1 + 
  (starting_players * starting_players) + 
  (starting_players^3 * (starting_players - 1)) + 
  (starting_players^5 * (starting_players - 1) * (starting_players - 2)) + 
  (starting_players^7 * (starting_players - 1) * (starting_players - 2) * (starting_players - 3))

/-- Theorem stating that the number of substitution ways modulo 1000 is 712 --/
theorem substitution_ways_mod_1000 : 
  substitution_ways % 1000 = 712 := by sorry

end NUMINAMATH_CALUDE_substitution_ways_mod_1000_l1532_153296


namespace NUMINAMATH_CALUDE_value_of_x_l1532_153225

theorem value_of_x : ∃ x : ℝ, 3 * x + 15 = (1 / 3) * (7 * x + 45) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1532_153225


namespace NUMINAMATH_CALUDE_evaluate_polynomial_l1532_153268

theorem evaluate_polynomial (a b : ℤ) (h : b = a + 2) :
  b^3 - a*b^2 - a^2*b + a^3 = 8*(a + 1) := by
  sorry

end NUMINAMATH_CALUDE_evaluate_polynomial_l1532_153268


namespace NUMINAMATH_CALUDE_intersection_on_y_axis_l1532_153247

/-- Given two lines l₁ and l₂, prove that if their intersection is on the y-axis, then C = -4 -/
theorem intersection_on_y_axis (A : ℝ) :
  ∃ (x y : ℝ),
    (A * x + 3 * y + C = 0) ∧
    (2 * x - 3 * y + 4 = 0) ∧
    (x = 0) →
    C = -4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_on_y_axis_l1532_153247


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l1532_153295

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 2/b + 3/c = 2) :
  a + 2*b + 3*c ≥ 18 :=
by sorry

theorem min_value_achieved (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 2/b + 3/c = 2) :
  (a + 2*b + 3*c = 18) ↔ (a = 3 ∧ b = 3 ∧ c = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l1532_153295


namespace NUMINAMATH_CALUDE_square_orientation_after_1011_transformations_l1532_153208

/-- Represents the possible orientations of the square -/
inductive SquareOrientation
  | ABCD
  | DABC
  | BADC
  | DCBA

/-- Applies the 90-degree clockwise rotation -/
def rotate90 (s : SquareOrientation) : SquareOrientation :=
  match s with
  | SquareOrientation.ABCD => SquareOrientation.DABC
  | SquareOrientation.DABC => SquareOrientation.BADC
  | SquareOrientation.BADC => SquareOrientation.DCBA
  | SquareOrientation.DCBA => SquareOrientation.ABCD

/-- Applies the 180-degree rotation -/
def rotate180 (s : SquareOrientation) : SquareOrientation :=
  match s with
  | SquareOrientation.ABCD => SquareOrientation.BADC
  | SquareOrientation.DABC => SquareOrientation.BADC
  | SquareOrientation.BADC => SquareOrientation.ABCD
  | SquareOrientation.DCBA => SquareOrientation.ABCD

/-- Applies both rotations in sequence -/
def applyTransformations (s : SquareOrientation) : SquareOrientation :=
  rotate180 (rotate90 s)

/-- Applies the transformations n times -/
def applyNTimes (s : SquareOrientation) (n : Nat) : SquareOrientation :=
  match n with
  | 0 => s
  | n + 1 => applyTransformations (applyNTimes s n)

theorem square_orientation_after_1011_transformations :
  applyNTimes SquareOrientation.ABCD 1011 = SquareOrientation.DCBA := by
  sorry


end NUMINAMATH_CALUDE_square_orientation_after_1011_transformations_l1532_153208


namespace NUMINAMATH_CALUDE_plot_breadth_is_8_l1532_153228

/-- A rectangular plot with the given properties. -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area_is_18_times_breadth : length * breadth = 18 * breadth
  length_breadth_difference : length - breadth = 10

/-- The breadth of the rectangular plot is 8 meters. -/
theorem plot_breadth_is_8 (plot : RectangularPlot) : plot.breadth = 8 := by
  sorry

end NUMINAMATH_CALUDE_plot_breadth_is_8_l1532_153228


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_sqrt_equation_unique_solution_sqrt_equation_boundary_solution_sqrt_equation_no_solution_l1532_153242

/-- The equation √(x+1) - √(2x+1) = m has solutions as described -/
theorem sqrt_equation_solutions (m : ℝ) :
  (∃ x : ℝ, Real.sqrt (x + 1) - Real.sqrt (2 * x + 1) = m) ↔ 
  m ≤ Real.sqrt 2 / 2 :=
sorry

/-- The equation √(x+1) - √(2x+1) = m has exactly one solution when m < √2/2 -/
theorem sqrt_equation_unique_solution (m : ℝ) (h : m < Real.sqrt 2 / 2) :
  ∃! x : ℝ, Real.sqrt (x + 1) - Real.sqrt (2 * x + 1) = m :=
sorry

/-- The equation √(x+1) - √(2x+1) = m has exactly one solution when m = √2/2 -/
theorem sqrt_equation_boundary_solution :
  ∃! x : ℝ, Real.sqrt (x + 1) - Real.sqrt (2 * x + 1) = Real.sqrt 2 / 2 :=
sorry

/-- The equation √(x+1) - √(2x+1) = m has no solutions when m > √2/2 -/
theorem sqrt_equation_no_solution (m : ℝ) (h : m > Real.sqrt 2 / 2) :
  ¬∃ x : ℝ, Real.sqrt (x + 1) - Real.sqrt (2 * x + 1) = m :=
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_sqrt_equation_unique_solution_sqrt_equation_boundary_solution_sqrt_equation_no_solution_l1532_153242


namespace NUMINAMATH_CALUDE_part_one_solution_set_part_two_minimum_value_l1532_153248

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one_solution_set (x : ℝ) : 
  (f 1 x ≥ 4 - |x - 3|) ↔ (x ≤ 0 ∨ x ≥ 4) := by sorry

-- Part II
theorem part_two_minimum_value (a m n : ℝ) (h1 : m > 0) (h2 : n > 0) :
  (Set.Icc 0 2 = {x | f a x ≤ 1}) → 
  (1 / m + 1 / (2 * n) = a) → 
  (∀ k l, k > 0 → l > 0 → 1 / k + 1 / (2 * l) = a → m * n ≤ k * l) →
  m * n = 2 := by sorry

end NUMINAMATH_CALUDE_part_one_solution_set_part_two_minimum_value_l1532_153248


namespace NUMINAMATH_CALUDE_minnow_count_l1532_153217

theorem minnow_count (total : ℕ) (red green white : ℕ) : 
  (red : ℚ) / total = 2/5 →
  (green : ℚ) / total = 3/10 →
  white + red + green = total →
  red = 20 →
  white = 15 := by
sorry

end NUMINAMATH_CALUDE_minnow_count_l1532_153217


namespace NUMINAMATH_CALUDE_line_angle_problem_l1532_153264

theorem line_angle_problem (a : ℝ) : 
  let line1 := {(x, y) : ℝ × ℝ | a * x - y + 3 = 0}
  let line2 := {(x, y) : ℝ × ℝ | x - 2 * y + 4 = 0}
  let angle := Real.arccos (Real.sqrt 5 / 5)
  (∃ (θ : ℝ), θ = angle ∧ 
    θ = Real.arccos ((1 + a * (1/2)) / Real.sqrt ((1 + a^2) * (1 + (1/2)^2))))
  → a = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_line_angle_problem_l1532_153264


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1532_153227

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ),
  X^4 + 1 = (X^2 - 3*X + 5) * q + r ∧
  r.degree < (X^2 - 3*X + 5).degree ∧
  r = -3*X - 19 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1532_153227


namespace NUMINAMATH_CALUDE_smallest_class_number_l1532_153222

theorem smallest_class_number (total_classes : Nat) (selected_classes : Nat) (sum_selected : Nat) : 
  total_classes = 24 →
  selected_classes = 4 →
  sum_selected = 48 →
  ∃ x : Nat, 
    x ≥ 1 ∧ 
    x ≤ total_classes ∧ 
    x + (x + (total_classes / selected_classes)) + 
    (x + 2 * (total_classes / selected_classes)) + 
    (x + 3 * (total_classes / selected_classes)) = sum_selected ∧
    x = 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_class_number_l1532_153222


namespace NUMINAMATH_CALUDE_solution_set_part1_a_upper_bound_l1532_153287

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 2 x ≥ 3} = {x : ℝ | x ≤ -1 ∨ x ≥ 3} :=
sorry

-- Part 2
theorem a_upper_bound (a : ℝ) :
  (∀ x ∈ Set.Ici 1, f a x ≥ -x^2 - 2) → a ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_a_upper_bound_l1532_153287


namespace NUMINAMATH_CALUDE_ratio_of_numbers_with_special_average_l1532_153285

theorem ratio_of_numbers_with_special_average (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (h4 : (a + b) / 2 = a - b) : a / b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_with_special_average_l1532_153285


namespace NUMINAMATH_CALUDE_first_expression_second_expression_l1532_153290

-- First expression
theorem first_expression (x y : ℝ) : (-3 * x^2 * y)^3 = -27 * x^6 * y^3 := by sorry

-- Second expression
theorem second_expression (a : ℝ) : (-2*a - 1) * (2*a - 1) = 1 - 4*a^2 := by sorry

end NUMINAMATH_CALUDE_first_expression_second_expression_l1532_153290


namespace NUMINAMATH_CALUDE_cos_three_halves_pi_l1532_153213

theorem cos_three_halves_pi : Real.cos (3 * π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_three_halves_pi_l1532_153213


namespace NUMINAMATH_CALUDE_max_sum_of_distances_squared_l1532_153265

def A : ℝ × ℝ := (-2, -2)
def B : ℝ × ℝ := (-2, 6)
def C : ℝ × ℝ := (4, -2)

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

def sum_of_distances_squared (P : ℝ × ℝ) : ℝ :=
  distance_squared P A + distance_squared P B + distance_squared P C

def on_circle (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 = 4

theorem max_sum_of_distances_squared :
  ∃ (max : ℝ), max = 88 ∧
  ∀ (P : ℝ × ℝ), on_circle P → sum_of_distances_squared P ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_distances_squared_l1532_153265


namespace NUMINAMATH_CALUDE_locus_characterization_l1532_153281

def has_solution (u v : ℝ) (n : ℕ) : Prop :=
  ∃ x y : ℝ, (Real.sin x)^(2*n) + (Real.cos y)^(2*n) = u ∧ (Real.sin x)^n + (Real.cos y)^n = v

theorem locus_characterization (u v : ℝ) (n : ℕ) :
  has_solution u v n ↔ 
    (v^2 ≤ 2*u ∧ (v - 1)^2 ≥ (u - 1)) ∧
    ((n % 2 = 0 → (0 ≤ v ∧ v ≤ 2 ∧ v^2 ≥ u)) ∧
     (n % 2 = 1 → (-2 ≤ v ∧ v ≤ 2 ∧ (v + 1)^2 ≥ (u - 1)))) :=
by sorry

end NUMINAMATH_CALUDE_locus_characterization_l1532_153281


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_range_l1532_153243

/-- The range of b for which the ellipse C: x²/4 + y²/b = 1 always intersects with any line l: y = mx + 1 -/
theorem ellipse_line_intersection_range :
  ∀ (b : ℝ),
  (∀ (m : ℝ), ∃ (x y : ℝ), x^2/4 + y^2/b = 1 ∧ y = m*x + 1) →
  (b ∈ Set.Icc 1 4 ∪ Set.Ioi 4) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_range_l1532_153243


namespace NUMINAMATH_CALUDE_radical_simplification_l1532_153230

theorem radical_simplification (y : ℝ) (h : y > 0) :
  Real.sqrt (50 * y) * Real.sqrt (5 * y) * Real.sqrt (45 * y) = 15 * y * Real.sqrt (10 * y) :=
by sorry

end NUMINAMATH_CALUDE_radical_simplification_l1532_153230


namespace NUMINAMATH_CALUDE_koala_fiber_consumption_l1532_153219

theorem koala_fiber_consumption (absorption_rate : ℝ) (absorbed_amount : ℝ) (total_consumed : ℝ) :
  absorption_rate = 0.25 →
  absorbed_amount = 10.5 →
  absorbed_amount = absorption_rate * total_consumed →
  total_consumed = 42 := by
  sorry

end NUMINAMATH_CALUDE_koala_fiber_consumption_l1532_153219


namespace NUMINAMATH_CALUDE_reflect_x_coordinates_l1532_153205

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the reflection across x-axis operation
def reflect_x (p : Point) : Point :=
  (p.1, -p.2)

-- Theorem statement
theorem reflect_x_coordinates (x y : ℝ) :
  reflect_x (x, y) = (x, -y) := by
  sorry

end NUMINAMATH_CALUDE_reflect_x_coordinates_l1532_153205


namespace NUMINAMATH_CALUDE_sum_ten_smallest_multiples_of_eight_l1532_153203

theorem sum_ten_smallest_multiples_of_eight : 
  (Finset.range 10).sum (fun i => 8 * (i + 1)) = 440 := by
  sorry

end NUMINAMATH_CALUDE_sum_ten_smallest_multiples_of_eight_l1532_153203


namespace NUMINAMATH_CALUDE_equation_equivalence_l1532_153212

theorem equation_equivalence (x y : ℝ) :
  (3 * x^2 + 9 * x + 7 * y + 2 = 0) ∧ (3 * x + 2 * y + 5 = 0) →
  4 * y^2 + 23 * y - 14 = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1532_153212


namespace NUMINAMATH_CALUDE_back_wheel_circumference_l1532_153294

/-- Given a cart with front and back wheels, this theorem proves the circumference of the back wheel
    based on the given conditions. -/
theorem back_wheel_circumference
  (front_circumference : ℝ)
  (distance : ℝ)
  (revolution_difference : ℕ)
  (h1 : front_circumference = 30)
  (h2 : distance = 1650)
  (h3 : revolution_difference = 5) :
  ∃ (back_circumference : ℝ),
    back_circumference * (distance / front_circumference - revolution_difference) = distance ∧
    back_circumference = 33 :=
by sorry

end NUMINAMATH_CALUDE_back_wheel_circumference_l1532_153294


namespace NUMINAMATH_CALUDE_triangle_similarity_equivalence_l1532_153241

theorem triangle_similarity_equivalence 
  (a b c a₁ b₁ c₁ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha₁ : a₁ > 0) (hb₁ : b₁ > 0) (hc₁ : c₁ > 0) :
  (∃ k : ℝ, k > 0 ∧ a = k * a₁ ∧ b = k * b₁ ∧ c = k * c₁) ↔ 
  (Real.sqrt (a * a₁) + Real.sqrt (b * b₁) + Real.sqrt (c * c₁) = 
   Real.sqrt ((a + b + c) * (a₁ + b₁ + c₁))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_equivalence_l1532_153241


namespace NUMINAMATH_CALUDE_king_middle_school_teachers_l1532_153259

theorem king_middle_school_teachers (total_students : ℕ) 
  (classes_per_student : ℕ) (classes_per_teacher : ℕ) 
  (students_per_class : ℕ) :
  total_students = 1500 →
  classes_per_student = 6 →
  classes_per_teacher = 5 →
  students_per_class = 25 →
  (total_students * classes_per_student) / (students_per_class * classes_per_teacher) = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_king_middle_school_teachers_l1532_153259


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1532_153283

theorem inequality_system_solution_set :
  ∀ x : ℝ, (2 * x ≤ -1 ∧ x > -1) ↔ (-1 < x ∧ x ≤ -1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1532_153283


namespace NUMINAMATH_CALUDE_intersection_right_angle_coordinates_l1532_153226

-- Define the line and parabola
def line (x y : ℝ) : Prop := x - 2*y - 1 = 0
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define points A and B as intersections
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line A.1 A.2 ∧ parabola A.1 A.2 ∧
  line B.1 B.2 ∧ parabola B.1 B.2 ∧
  A ≠ B

-- Define point C on the parabola
def point_on_parabola (C : ℝ × ℝ) : Prop := parabola C.1 C.2

-- Define right angle ACB
def right_angle (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0

-- Theorem statement
theorem intersection_right_angle_coordinates :
  ∀ A B C : ℝ × ℝ,
  intersection_points A B →
  point_on_parabola C →
  right_angle A B C →
  (C = (1, -2) ∨ C = (9, -6)) :=
sorry

end NUMINAMATH_CALUDE_intersection_right_angle_coordinates_l1532_153226


namespace NUMINAMATH_CALUDE_tall_mirror_passes_l1532_153293

/-- The number of times Sarah and Ellie passed through the room with tall mirrors -/
def T : ℕ := sorry

/-- Sarah's reflections in tall mirrors -/
def sarah_tall : ℕ := 10

/-- Sarah's reflections in wide mirrors -/
def sarah_wide : ℕ := 5

/-- Ellie's reflections in tall mirrors -/
def ellie_tall : ℕ := 6

/-- Ellie's reflections in wide mirrors -/
def ellie_wide : ℕ := 3

/-- Number of times they passed through the wide mirrors room -/
def wide_passes : ℕ := 5

/-- Total number of reflections seen by Sarah and Ellie -/
def total_reflections : ℕ := 88

theorem tall_mirror_passes :
  T * (sarah_tall + ellie_tall) + wide_passes * (sarah_wide + ellie_wide) = total_reflections ∧
  T = 3 := by sorry

end NUMINAMATH_CALUDE_tall_mirror_passes_l1532_153293


namespace NUMINAMATH_CALUDE_percentage_of_female_officers_on_duty_l1532_153235

def total_officers_on_duty : ℕ := 160
def total_female_officers : ℕ := 500

def female_officers_on_duty : ℕ := total_officers_on_duty / 2

def percentage_on_duty : ℚ := (female_officers_on_duty : ℚ) / total_female_officers * 100

theorem percentage_of_female_officers_on_duty :
  percentage_on_duty = 16 := by sorry

end NUMINAMATH_CALUDE_percentage_of_female_officers_on_duty_l1532_153235


namespace NUMINAMATH_CALUDE_price_calculation_equivalence_l1532_153277

theorem price_calculation_equivalence 
  (initial_price tax_rate discount_rate : ℝ) 
  (tax_rate_pos : 0 < tax_rate) 
  (discount_rate_pos : 0 < discount_rate) 
  (tax_rate_bound : tax_rate < 1) 
  (discount_rate_bound : discount_rate < 1) :
  initial_price * (1 + tax_rate) * (1 - discount_rate) = 
  initial_price * (1 - discount_rate) * (1 + tax_rate) :=
by sorry

end NUMINAMATH_CALUDE_price_calculation_equivalence_l1532_153277


namespace NUMINAMATH_CALUDE_nonAttackingRooksPlacementCount_l1532_153292

/-- The size of the chessboard -/
def boardSize : Nat := 8

/-- The total number of squares on the chessboard -/
def totalSquares : Nat := boardSize * boardSize

/-- The number of squares a rook attacks in its row and column, excluding itself -/
def attackedSquares : Nat := 2 * (boardSize - 1)

/-- The number of ways to place two rooks on a chessboard so they don't attack each other -/
def nonAttackingRooksPlacement : Nat := totalSquares * (totalSquares - 1 - attackedSquares)

theorem nonAttackingRooksPlacementCount : nonAttackingRooksPlacement = 3136 := by
  sorry

end NUMINAMATH_CALUDE_nonAttackingRooksPlacementCount_l1532_153292


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_expression_3_l1532_153215

-- Define variables
variable (a b x y : ℝ)

-- Theorem 1
theorem simplify_expression_1 : 2*a - 3*b + a - 5*b = 3*a - 8*b := by sorry

-- Theorem 2
theorem simplify_expression_2 : (a^2 - 6*a) - 3*(a^2 - 2*a + 1) + 3 = -2*a^2 := by sorry

-- Theorem 3
theorem simplify_expression_3 : 4*(x^2*y - 2*x*y^2) - 3*(-x*y^2 + 2*x^2*y) = -2*x^2*y - 5*x*y^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_expression_3_l1532_153215


namespace NUMINAMATH_CALUDE_store_profit_percentage_l1532_153218

/-- Calculates the profit percentage on items sold in February given the markups and discount -/
theorem store_profit_percentage (initial_markup : ℝ) (new_year_markup : ℝ) (february_discount : ℝ) :
  initial_markup = 0.20 →
  new_year_markup = 0.25 →
  february_discount = 0.20 →
  (1 + initial_markup + new_year_markup * (1 + initial_markup)) * (1 - february_discount) - 1 = 0.20 := by
  sorry

#check store_profit_percentage

end NUMINAMATH_CALUDE_store_profit_percentage_l1532_153218


namespace NUMINAMATH_CALUDE_matching_pair_probability_for_sue_l1532_153289

/-- Represents the number of pairs of shoes for each color --/
structure ShoePairs :=
  (black : Nat)
  (brown : Nat)
  (gray : Nat)
  (red : Nat)

/-- Calculates the probability of picking a matching pair of shoes --/
def matchingPairProbability (shoes : ShoePairs) : Rat :=
  let totalShoes := 2 * (shoes.black + shoes.brown + shoes.gray + shoes.red)
  let matchingPairs := 
    shoes.black * (shoes.black - 1) + 
    shoes.brown * (shoes.brown - 1) + 
    shoes.gray * (shoes.gray - 1) + 
    shoes.red * (shoes.red - 1)
  matchingPairs / (totalShoes * (totalShoes - 1))

theorem matching_pair_probability_for_sue : 
  let sueShoes : ShoePairs := { black := 7, brown := 4, gray := 3, red := 2 }
  matchingPairProbability sueShoes = 39 / 248 := by
  sorry

end NUMINAMATH_CALUDE_matching_pair_probability_for_sue_l1532_153289


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1532_153272

/-- The eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (A B D : ℝ × ℝ) (c : ℝ),
    -- Right focus of the hyperbola
    c > 0 ∧
    -- Equation of the hyperbola
    (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}) ∧
    -- A and B are on the hyperbola and on a line perpendicular to x-axis through the right focus
    A ∈ {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1} ∧
    B ∈ {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1} ∧
    A.1 = c ∧ B.1 = c ∧
    -- D is on the imaginary axis
    D = (0, b) ∧
    -- ABD is a right-angled triangle
    (A.1 - D.1) * (B.1 - D.1) + (A.2 - D.2) * (B.2 - D.2) = 0 →
    -- The eccentricity is either √2 or √(2 + √2)
    c / a = Real.sqrt 2 ∨ c / a = Real.sqrt (2 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1532_153272
