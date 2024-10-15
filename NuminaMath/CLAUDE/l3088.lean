import Mathlib

namespace NUMINAMATH_CALUDE_bell_pepper_ratio_l3088_308827

/-- Represents the number of bell peppers --/
def num_peppers : ℕ := 5

/-- Represents the number of large slices per bell pepper --/
def slices_per_pepper : ℕ := 20

/-- Represents the total number of slices and pieces in the meal --/
def total_pieces : ℕ := 200

/-- Represents the number of smaller pieces each large slice is cut into --/
def pieces_per_slice : ℕ := 3

/-- Calculates the total number of large slices --/
def total_large_slices : ℕ := num_peppers * slices_per_pepper

/-- Theorem stating the ratio of large slices cut into smaller pieces to total large slices --/
theorem bell_pepper_ratio : 
  ∃ (x : ℕ), x * pieces_per_slice + (total_large_slices - x) = total_pieces ∧ 
             x = 33 ∧
             (x : ℚ) / (total_large_slices : ℚ) = 33 / 100 := by
  sorry

end NUMINAMATH_CALUDE_bell_pepper_ratio_l3088_308827


namespace NUMINAMATH_CALUDE_m_minus_n_eq_neg_reals_l3088_308820

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x) ∧ -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1}

-- Define the set difference operation
def setDifference (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- State the theorem
theorem m_minus_n_eq_neg_reals : 
  setDifference M N = {x : ℝ | x < 0} := by sorry

end NUMINAMATH_CALUDE_m_minus_n_eq_neg_reals_l3088_308820


namespace NUMINAMATH_CALUDE_secant_triangle_area_l3088_308880

theorem secant_triangle_area (r : ℝ) (d : ℝ) (θ : ℝ) (S_ABC : ℝ) :
  r = 3 →
  d = 5 →
  θ = 30 * π / 180 →
  S_ABC = 10 →
  ∃ (S_AKL : ℝ), S_AKL = 8 / 5 :=
by sorry

end NUMINAMATH_CALUDE_secant_triangle_area_l3088_308880


namespace NUMINAMATH_CALUDE_system_solution_unique_l3088_308889

theorem system_solution_unique :
  ∃! (x y : ℚ), 37 * x + 92 * y = 5043 ∧ 92 * x + 37 * y = 2568 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3088_308889


namespace NUMINAMATH_CALUDE_sand_weight_formula_l3088_308825

/-- Given a number of bags n, where each full bag contains 65 pounds of sand,
    and one bag is not full containing 42 pounds of sand,
    the total weight of sand W is (n-1) * 65 + 42 pounds. -/
theorem sand_weight_formula (n : ℕ) (W : ℕ) : W = (n - 1) * 65 + 42 :=
by sorry

end NUMINAMATH_CALUDE_sand_weight_formula_l3088_308825


namespace NUMINAMATH_CALUDE_complement_A_intersection_B_range_l3088_308837

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

-- Define set B with parameter a
def B (a : ℝ) : Set ℝ := {x : ℝ | |x| > a}

-- Define the complement of A with respect to U
def complementA : Set ℝ := Set.Icc (-1) 3

theorem complement_A_intersection_B_range :
  (∀ a : ℝ, (complementA ∩ B a).Nonempty) ↔ a ∈ Set.Icc 0 2 :=
sorry

end NUMINAMATH_CALUDE_complement_A_intersection_B_range_l3088_308837


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l3088_308831

/-- The function f(x) = x^3 + 2 -/
def f (x : ℝ) : ℝ := x^3 + 2

/-- Theorem: The derivative of f at x = 2 is equal to 12 -/
theorem f_derivative_at_2 : 
  deriv f 2 = 12 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l3088_308831


namespace NUMINAMATH_CALUDE_people_on_boats_l3088_308835

theorem people_on_boats (total_boats : Nat) (boats_with_four : Nat) (boats_with_five : Nat)
  (h1 : total_boats = 7)
  (h2 : boats_with_four = 4)
  (h3 : boats_with_five = 3)
  (h4 : total_boats = boats_with_four + boats_with_five) :
  boats_with_four * 4 + boats_with_five * 5 = 31 := by
  sorry

end NUMINAMATH_CALUDE_people_on_boats_l3088_308835


namespace NUMINAMATH_CALUDE_largest_n_for_factorable_quadratic_l3088_308852

/-- A structure representing a quadratic expression ax^2 + bx + c -/
structure Quadratic where
  a : ℤ
  b : ℤ
  c : ℤ

/-- A structure representing a linear factor ax + b -/
structure LinearFactor where
  a : ℤ
  b : ℤ

/-- Function to check if a quadratic can be factored into two linear factors -/
def isFactorable (q : Quadratic) (l1 l2 : LinearFactor) : Prop :=
  q.a = l1.a * l2.a ∧
  q.b = l1.a * l2.b + l1.b * l2.a ∧
  q.c = l1.b * l2.b

/-- The main theorem stating the largest value of n -/
theorem largest_n_for_factorable_quadratic :
  ∃ (n : ℤ),
    n = 451 ∧
    (∀ m : ℤ, m > n → 
      ¬∃ (l1 l2 : LinearFactor), 
        isFactorable ⟨5, m, 90⟩ l1 l2) ∧
    (∃ (l1 l2 : LinearFactor), 
      isFactorable ⟨5, n, 90⟩ l1 l2) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_factorable_quadratic_l3088_308852


namespace NUMINAMATH_CALUDE_inscribed_circle_distance_l3088_308864

theorem inscribed_circle_distance (a b : ℝ) (h1 : a = 6) (h2 : b = 8) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a + b + c) / 2
  let m := s - b
  2 * Real.sqrt ((a^2 + m^2 - 2 * a * m * (a / c)) / 5) = 2 * Real.sqrt (29 / 5) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_distance_l3088_308864


namespace NUMINAMATH_CALUDE_square_floor_tiles_l3088_308862

theorem square_floor_tiles (diagonal_tiles : ℕ) (total_tiles : ℕ) : 
  diagonal_tiles = 37 → total_tiles = 361 → 
  (∃ (side_length : ℕ), 
    2 * side_length - 1 = diagonal_tiles ∧ 
    side_length * side_length = total_tiles) := by
  sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l3088_308862


namespace NUMINAMATH_CALUDE_fraction_sum_equals_ten_thirds_l3088_308894

theorem fraction_sum_equals_ten_thirds (a b : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (h1 : a = 2) (h2 : b = 1) : 
  (a + b) / (a - b) + (a - b) / (a + b) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_ten_thirds_l3088_308894


namespace NUMINAMATH_CALUDE_eric_park_time_ratio_l3088_308817

/-- The ratio of Eric's return time to his time to reach the park is 3:1 -/
theorem eric_park_time_ratio :
  let time_to_park : ℕ := 20 + 10  -- Time to reach the park (running + jogging)
  let time_to_return : ℕ := 90     -- Time to return home
  (time_to_return : ℚ) / time_to_park = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_eric_park_time_ratio_l3088_308817


namespace NUMINAMATH_CALUDE_roots_log_sum_l3088_308815

-- Define the equation
def equation (x : ℝ) : Prop := (Real.log x)^2 - Real.log (x^2) = 2

-- Define α and β as the roots of the equation
axiom α : ℝ
axiom β : ℝ
axiom α_pos : α > 0
axiom β_pos : β > 0
axiom α_root : equation α
axiom β_root : equation β

-- State the theorem
theorem roots_log_sum : Real.log β / Real.log α + Real.log α / Real.log β = -4 := by
  sorry

end NUMINAMATH_CALUDE_roots_log_sum_l3088_308815


namespace NUMINAMATH_CALUDE_school_teachers_count_l3088_308814

theorem school_teachers_count (total : ℕ) (sample_size : ℕ) (sampled_students : ℕ) :
  total = 2400 →
  sample_size = 160 →
  sampled_students = 150 →
  total - (total * sampled_students / sample_size) = 150 :=
by sorry

end NUMINAMATH_CALUDE_school_teachers_count_l3088_308814


namespace NUMINAMATH_CALUDE_square_area_and_perimeter_l3088_308887

/-- Given a square with diagonal length 12√2 cm, prove its area and perimeter -/
theorem square_area_and_perimeter (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  (s ^ 2 = 144) ∧ (4 * s = 48) := by sorry

end NUMINAMATH_CALUDE_square_area_and_perimeter_l3088_308887


namespace NUMINAMATH_CALUDE_triangle_construction_with_two_angles_and_perimeter_l3088_308822

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define perimeter
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem triangle_construction_with_two_angles_and_perimeter 
  (A B P : ℝ) 
  (h_angles : 0 < A ∧ 0 < B ∧ A + B < π) 
  (h_perimeter : P > 0) :
  ∃ (t : Triangle), 
    t.angleA = A ∧ 
    t.angleB = B ∧ 
    perimeter t = P := by
  sorry

end NUMINAMATH_CALUDE_triangle_construction_with_two_angles_and_perimeter_l3088_308822


namespace NUMINAMATH_CALUDE_x_equals_two_l3088_308845

-- Define the * operation
def star (a b : ℕ) : ℕ := 
  Finset.sum (Finset.range b) (λ i => a + i)

-- State the theorem
theorem x_equals_two : 
  ∃ x : ℕ, star x 10 = 65 ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_x_equals_two_l3088_308845


namespace NUMINAMATH_CALUDE_table_relationship_l3088_308839

def f (x : ℝ) : ℝ := 200 - 3*x - 6*x^2

theorem table_relationship : 
  (f 0 = 200) ∧ 
  (f 2 = 152) ∧ 
  (f 4 = 80) ∧ 
  (f 6 = -16) ∧ 
  (f 8 = -128) := by
  sorry

end NUMINAMATH_CALUDE_table_relationship_l3088_308839


namespace NUMINAMATH_CALUDE_units_digit_of_square_l3088_308871

theorem units_digit_of_square (n : ℕ) : (n ^ 2) % 10 ≠ 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_square_l3088_308871


namespace NUMINAMATH_CALUDE_effective_CAGR_l3088_308802

/-- The effective Compound Annual Growth Rate (CAGR) for an investment with stepped interest rates, inflation, and currency exchange rate changes. -/
theorem effective_CAGR 
  (R1 R2 R3 R4 I C : ℝ) 
  (h_growth : (3/5 : ℝ) = (1 + R1/100)^(5/2) * (1 + R2/100)^(5/2) * (1 + R3/100)^(5/2) * (1 + R4/100)^(5/2)) :
  ∃ CAGR : ℝ, 
    CAGR = ((1 + R1/100)^(5/2) * (1 + R2/100)^(5/2) * (1 + R3/100)^(5/2) * (1 + R4/100)^(5/2) / (1 + I/100)^10 * (1 + C/100)^10)^(1/10) - 1 := by
  sorry

end NUMINAMATH_CALUDE_effective_CAGR_l3088_308802


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l3088_308800

theorem ice_cream_flavors (cone_types : ℕ) (total_combinations : ℕ) (h1 : cone_types = 2) (h2 : total_combinations = 8) :
  total_combinations / cone_types = 4 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l3088_308800


namespace NUMINAMATH_CALUDE_intimate_functions_range_l3088_308848

theorem intimate_functions_range (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x ∈ Set.Icc 2 3, f x = x^3 - 2*x + 7) →
  (∀ x ∈ Set.Icc 2 3, g x = x + m) →
  (∀ x ∈ Set.Icc 2 3, |f x - g x| ≤ 10) →
  15 ≤ m ∧ m ≤ 19 := by
sorry

end NUMINAMATH_CALUDE_intimate_functions_range_l3088_308848


namespace NUMINAMATH_CALUDE_bucket_weight_l3088_308809

/-- 
Given:
- p: weight when bucket is three-quarters full
- q: weight when bucket is one-third full
- r: weight of empty bucket
Prove: weight of full bucket is (4p - r) / 3
-/
theorem bucket_weight (p q r : ℝ) : ℝ :=
  let three_quarters_full := p
  let one_third_full := q
  let empty_bucket := r
  let full_bucket := (4 * p - r) / 3
  full_bucket

#check bucket_weight

end NUMINAMATH_CALUDE_bucket_weight_l3088_308809


namespace NUMINAMATH_CALUDE_tan_45_degrees_l3088_308885

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l3088_308885


namespace NUMINAMATH_CALUDE_max_rectangle_area_l3088_308838

/-- The equation that the vertices' coordinates must satisfy -/
def vertex_equation (x y : ℝ) : Prop :=
  |y + 1| * (y^2 + 2*y + 28) + |x - 2| = 9 * (y^2 + 2*y + 4)

/-- The area function of the rectangle -/
def rectangle_area (x : ℝ) : ℝ :=
  -4 * x * (x - 3)^3

/-- The theorem stating the maximum area of the rectangle -/
theorem max_rectangle_area :
  ∃ (x y : ℝ), vertex_equation x y ∧
  ∀ (x' y' : ℝ), vertex_equation x' y' → rectangle_area x ≥ rectangle_area x' ∧
  rectangle_area x = 34.171875 := by
sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l3088_308838


namespace NUMINAMATH_CALUDE_pauls_money_duration_l3088_308879

/-- Represents the duration (in weeks) that money lasts given earnings and weekly spending. -/
def money_duration (lawn_earnings weed_eating_earnings weekly_spending : ℚ) : ℚ :=
  (lawn_earnings + weed_eating_earnings) / weekly_spending

/-- Theorem stating that Paul's money lasts for 2 weeks given his earnings and spending. -/
theorem pauls_money_duration :
  money_duration 3 3 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_pauls_money_duration_l3088_308879


namespace NUMINAMATH_CALUDE_attractions_permutations_l3088_308816

theorem attractions_permutations : Nat.factorial 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_attractions_permutations_l3088_308816


namespace NUMINAMATH_CALUDE_min_value_theorem_l3088_308881

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.exp x = y * Real.log x + y * Real.log y) : 
  ∃ (m : ℝ), ∀ (x' y' : ℝ) (hx' : x' > 0) (hy' : y' > 0) 
  (h' : Real.exp x' = y' * Real.log x' + y' * Real.log y'), 
  (Real.exp x' / x' - Real.log y') ≥ m ∧ 
  (Real.exp x / x - Real.log y = m) ∧ 
  m = Real.exp 1 - 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3088_308881


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l3088_308804

theorem arithmetic_sequence_inequality (a₁ : ℝ) (d : ℝ) :
  (∀ n : Fin 8, a₁ + (n : ℕ) * d > 0) →
  d ≠ 0 →
  (a₁ * (a₁ + 7 * d)) < ((a₁ + 3 * d) * (a₁ + 4 * d)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l3088_308804


namespace NUMINAMATH_CALUDE_ab_in_terms_of_m_and_n_l3088_308803

theorem ab_in_terms_of_m_and_n (a b m n : ℝ) 
  (h1 : (a + b)^2 = m) 
  (h2 : (a - b)^2 = n) : 
  a * b = (m - n) / 4 := by
sorry

end NUMINAMATH_CALUDE_ab_in_terms_of_m_and_n_l3088_308803


namespace NUMINAMATH_CALUDE_no_prime_solution_l3088_308840

theorem no_prime_solution (p : ℕ) (hp : Prime p) : ¬(2^p + p ∣ 3^p + p) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_l3088_308840


namespace NUMINAMATH_CALUDE_decimal_3_is_binary_11_binary_11_is_decimal_3_l3088_308892

/-- Converts a natural number to its binary representation as a list of bits (0s and 1s) --/
def toBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else go (m / 2) ((m % 2) :: acc)
    go n []

/-- Converts a list of bits (0s and 1s) to its decimal representation --/
def fromBinary (bits : List ℕ) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + b) 0

theorem decimal_3_is_binary_11 : toBinary 3 = [1, 1] := by
  sorry

theorem binary_11_is_decimal_3 : fromBinary [1, 1] = 3 := by
  sorry

end NUMINAMATH_CALUDE_decimal_3_is_binary_11_binary_11_is_decimal_3_l3088_308892


namespace NUMINAMATH_CALUDE_prob_product_multiple_of_four_is_two_fifths_l3088_308896

/-- A fair 12-sided die -/
def dodecahedral_die := Finset.range 12

/-- A fair 10-sided die -/
def ten_sided_die := Finset.range 10

/-- The probability of an event occurring when rolling a fair n-sided die -/
def prob_fair_die (event : Finset ℕ) (die : Finset ℕ) : ℚ :=
  (event ∩ die).card / die.card

/-- The event of rolling a multiple of 4 -/
def multiple_of_four (die : Finset ℕ) : Finset ℕ :=
  die.filter (λ x => x % 4 = 0)

/-- The probability that the product of two rolls is a multiple of 4 -/
def prob_product_multiple_of_four : ℚ :=
  1 - (1 - prob_fair_die (multiple_of_four dodecahedral_die) dodecahedral_die) *
      (1 - prob_fair_die (multiple_of_four ten_sided_die) ten_sided_die)

theorem prob_product_multiple_of_four_is_two_fifths :
  prob_product_multiple_of_four = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_product_multiple_of_four_is_two_fifths_l3088_308896


namespace NUMINAMATH_CALUDE_bicycle_selling_price_l3088_308829

/-- The final selling price of a bicycle given initial cost and profit percentages -/
theorem bicycle_selling_price 
  (initial_cost : ℝ) 
  (profit_a_percent : ℝ) 
  (profit_b_percent : ℝ) 
  (h1 : initial_cost = 150)
  (h2 : profit_a_percent = 20)
  (h3 : profit_b_percent = 25) : 
  initial_cost * (1 + profit_a_percent / 100) * (1 + profit_b_percent / 100) = 225 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_selling_price_l3088_308829


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l3088_308833

theorem greatest_power_of_two_factor (n : ℕ) : 
  ∃ (k : ℕ), (2^459 : ℕ) ∣ (9^456 - 3^684) ∧ 
  ∀ m > 459, ¬((2^m : ℕ) ∣ (9^456 - 3^684)) := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l3088_308833


namespace NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l3088_308861

theorem product_of_sum_and_cube_sum (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a^3 + b^3 = 136) : 
  a * b = -6 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l3088_308861


namespace NUMINAMATH_CALUDE_proportional_segments_l3088_308884

theorem proportional_segments (a b c d : ℝ) :
  b = 3 → c = 4 → d = 6 → (a / b = c / d) → a = 2 := by sorry

end NUMINAMATH_CALUDE_proportional_segments_l3088_308884


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3088_308868

theorem fraction_inequality_solution_set :
  {x : ℝ | (x - 1) / x < 0} = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3088_308868


namespace NUMINAMATH_CALUDE_range_of_a_l3088_308865

-- Define the sets p and q
def p (x : ℝ) : Prop := x^2 - x - 2 ≥ 0
def q (a x : ℝ) : Prop := 2*a - 1 ≤ x ∧ x ≤ a + 3

-- Define the property that ¬p is a necessary but not sufficient condition for q
def not_p_necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, q a x → ¬(p x)) ∧ (∃ x, ¬(p x) ∧ ¬(q a x))

-- State the theorem
theorem range_of_a (a : ℝ) :
  not_p_necessary_not_sufficient a ↔ -1 ≤ a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3088_308865


namespace NUMINAMATH_CALUDE_mass_of_man_on_boat_l3088_308824

/-- The mass of a man who causes a boat to sink by a certain amount -/
def mass_of_man (boat_length boat_breadth boat_sink_height water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sink_height * water_density

/-- Theorem stating the mass of the man in the given problem -/
theorem mass_of_man_on_boat :
  let boat_length : ℝ := 8
  let boat_breadth : ℝ := 2
  let boat_sink_height : ℝ := 0.01  -- 1 cm in meters
  let water_density : ℝ := 1000     -- kg/m³
  mass_of_man boat_length boat_breadth boat_sink_height water_density = 160 := by
  sorry

#check mass_of_man_on_boat

end NUMINAMATH_CALUDE_mass_of_man_on_boat_l3088_308824


namespace NUMINAMATH_CALUDE_car_distance_calculation_l3088_308856

/-- The distance covered by the car in kilometers -/
def car_distance_km : ℝ := 2.2

/-- The distance covered by Amar in meters -/
def amar_distance_m : ℝ := 880

/-- The ratio of Amar's speed to the car's speed -/
def speed_ratio : ℚ := 2 / 5

theorem car_distance_calculation :
  car_distance_km = (amar_distance_m / speed_ratio) / 1000 := by
  sorry

#check car_distance_calculation

end NUMINAMATH_CALUDE_car_distance_calculation_l3088_308856


namespace NUMINAMATH_CALUDE_circle_tangent_and_shortest_chord_l3088_308801

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define points P and M
def P : ℝ × ℝ := (2, 5)
def M : ℝ × ℝ := (5, 0)

-- Define the line with shortest chord length
def shortest_chord_line (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the tangent lines
def tangent_line1 (x y : ℝ) : Prop := 3*x + 4*y - 15 = 0
def tangent_line2 (x : ℝ) : Prop := x = 5

theorem circle_tangent_and_shortest_chord :
  (∀ x y, C x y → shortest_chord_line x y → (x, y) = P ∨ (C x y ∧ shortest_chord_line x y)) ∧
  (∀ x y, C x y → tangent_line1 x y → (x, y) = M ∨ (C x y ∧ tangent_line1 x y)) ∧
  (∀ x y, C x y → tangent_line2 x → (x, y) = M ∨ (C x y ∧ tangent_line2 x)) := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_and_shortest_chord_l3088_308801


namespace NUMINAMATH_CALUDE_cylinder_prism_pyramid_elements_l3088_308836

/-- Represents a cylinder unwrapped into a prism with a pyramid attached -/
structure CylinderPrismPyramid where
  /-- Number of faces in the original prism -/
  prism_faces : ℕ
  /-- Number of edges in the original prism -/
  prism_edges : ℕ
  /-- Number of vertices in the original prism -/
  prism_vertices : ℕ
  /-- Number of faces added by the pyramid -/
  pyramid_faces : ℕ
  /-- Number of edges added by the pyramid -/
  pyramid_edges : ℕ
  /-- Number of vertices added by the pyramid -/
  pyramid_vertices : ℕ

/-- The total number of geometric elements in the CylinderPrismPyramid -/
def total_elements (cpp : CylinderPrismPyramid) : ℕ :=
  cpp.prism_faces + cpp.prism_edges + cpp.prism_vertices +
  cpp.pyramid_faces + cpp.pyramid_edges + cpp.pyramid_vertices

/-- Theorem stating that the total number of elements is 31 -/
theorem cylinder_prism_pyramid_elements :
  ∀ cpp : CylinderPrismPyramid,
  cpp.prism_faces = 5 ∧ 
  cpp.prism_edges = 10 ∧ 
  cpp.prism_vertices = 8 ∧
  cpp.pyramid_faces = 3 ∧
  cpp.pyramid_edges = 4 ∧
  cpp.pyramid_vertices = 1 →
  total_elements cpp = 31 := by
  sorry

#check cylinder_prism_pyramid_elements

end NUMINAMATH_CALUDE_cylinder_prism_pyramid_elements_l3088_308836


namespace NUMINAMATH_CALUDE_trapezoid_segment_property_l3088_308886

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midpoint_segment : ℝ
  equal_area_segment : ℝ
  base_difference : shorter_base + 150 = longer_base
  midpoint_area_ratio : (shorter_base + midpoint_segment) / (longer_base + midpoint_segment) = 3 / 4
  equal_area_condition : (shorter_base + equal_area_segment) * (height / 2) = 
                         (shorter_base + longer_base) * height / 2

/-- The theorem statement -/
theorem trapezoid_segment_property (t : Trapezoid) : 
  ⌊t.equal_area_segment^2 / 150⌋ = 304 := by sorry

end NUMINAMATH_CALUDE_trapezoid_segment_property_l3088_308886


namespace NUMINAMATH_CALUDE_parabola_directrix_l3088_308898

/-- Given a parabola with equation y = (x^2 - 4x + 3) / 8, its directrix is y = -9/8 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = (x^2 - 4*x + 3) / 8) → 
  (∃ (d : ℝ), d = -9/8 ∧ 
    ∀ (p : ℝ × ℝ), 
      p.1 = x ∧ p.2 = y → 
      ∃ (f : ℝ × ℝ), 
        (f.1 - p.1)^2 + (f.2 - p.2)^2 = (p.2 - d)^2 ∧
        ∀ (q : ℝ × ℝ), q.2 = d → 
          (f.1 - p.1)^2 + (f.2 - p.2)^2 ≤ (q.1 - p.1)^2 + (q.2 - p.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3088_308898


namespace NUMINAMATH_CALUDE_inhabitable_earth_surface_inhabitable_earth_surface_proof_l3088_308883

theorem inhabitable_earth_surface : Real → Prop :=
  λ x =>
    let total_surface := 1
    let land_fraction := 1 / 4
    let inhabitable_land_fraction := 1 / 2
    x = land_fraction * inhabitable_land_fraction ∧ 
    x = 1 / 8

-- Proof
theorem inhabitable_earth_surface_proof : inhabitable_earth_surface (1 / 8) := by
  sorry

end NUMINAMATH_CALUDE_inhabitable_earth_surface_inhabitable_earth_surface_proof_l3088_308883


namespace NUMINAMATH_CALUDE_candy_bars_total_l3088_308855

theorem candy_bars_total (people : Float) (bars_per_person : Float) : 
  people = 3.0 → 
  bars_per_person = 1.66666666699999 → 
  people * bars_per_person = 5.0 := by
  sorry

end NUMINAMATH_CALUDE_candy_bars_total_l3088_308855


namespace NUMINAMATH_CALUDE_total_flowers_l3088_308843

theorem total_flowers (num_pots : ℕ) (flowers_per_pot : ℕ) 
  (h1 : num_pots = 2150) 
  (h2 : flowers_per_pot = 128) : 
  num_pots * flowers_per_pot = 275200 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_l3088_308843


namespace NUMINAMATH_CALUDE_food_drive_problem_l3088_308842

/-- Represents the food drive problem in Ms. Perez's class -/
theorem food_drive_problem (total_students : ℕ) (total_cans : ℕ) 
  (students_with_four_cans : ℕ) (students_with_zero_cans : ℕ) :
  total_students = 30 →
  total_cans = 232 →
  students_with_four_cans = 13 →
  students_with_zero_cans = 2 →
  2 * (total_students - students_with_four_cans - students_with_zero_cans) = total_students →
  (total_cans - 4 * students_with_four_cans) / 
    (total_students - students_with_four_cans - students_with_zero_cans) = 12 :=
by sorry

end NUMINAMATH_CALUDE_food_drive_problem_l3088_308842


namespace NUMINAMATH_CALUDE_simplify_expression_l3088_308841

theorem simplify_expression (a : ℝ) : (a + 4) * (a - 4) - (a - 1)^2 = 2 * a - 17 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3088_308841


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_one_l3088_308878

theorem no_solution_implies_a_leq_one :
  (∀ x : ℝ, ¬(x + 2 > 3 ∧ x < a)) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_one_l3088_308878


namespace NUMINAMATH_CALUDE_roundness_of_eight_million_l3088_308811

def roundness (n : ℕ) : ℕ := sorry

theorem roundness_of_eight_million : roundness 8000000 = 15 := by sorry

end NUMINAMATH_CALUDE_roundness_of_eight_million_l3088_308811


namespace NUMINAMATH_CALUDE_chairs_for_play_l3088_308805

theorem chairs_for_play (rows : ℕ) (chairs_per_row : ℕ) 
  (h1 : rows = 27) (h2 : chairs_per_row = 16) : 
  rows * chairs_per_row = 432 := by
  sorry

end NUMINAMATH_CALUDE_chairs_for_play_l3088_308805


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3088_308808

theorem fixed_point_on_line (a b : ℝ) (h : a + 2 * b = 1) :
  a * (1/2) + 3 * (-1/6) + b = 0 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3088_308808


namespace NUMINAMATH_CALUDE_linda_total_sales_eq_366_9_l3088_308810

/-- Calculates the total sales for Linda's store given the following conditions:
  * Jeans are sold at $22 each
  * Tees are sold at $15 each
  * Jackets are sold at $37 each
  * 10% discount on jackets during the first half of the day
  * 7 tees sold
  * 4 jeans sold
  * 5 jackets sold in total
  * 3 jackets sold during the discount period
-/
def lindaTotalSales : ℝ :=
  let jeanPrice : ℝ := 22
  let teePrice : ℝ := 15
  let jacketPrice : ℝ := 37
  let jacketDiscount : ℝ := 0.1
  let teesSold : ℕ := 7
  let jeansSold : ℕ := 4
  let jacketsSold : ℕ := 5
  let discountedJackets : ℕ := 3
  let fullPriceJackets : ℕ := jacketsSold - discountedJackets
  let discountedJacketPrice : ℝ := jacketPrice * (1 - jacketDiscount)
  
  jeanPrice * jeansSold +
  teePrice * teesSold +
  jacketPrice * fullPriceJackets +
  discountedJacketPrice * discountedJackets

/-- Theorem stating that Linda's total sales at the end of the day equal $366.9 -/
theorem linda_total_sales_eq_366_9 : lindaTotalSales = 366.9 := by
  sorry

end NUMINAMATH_CALUDE_linda_total_sales_eq_366_9_l3088_308810


namespace NUMINAMATH_CALUDE_smallest_block_with_360_hidden_l3088_308899

/-- Given a rectangular block made of unit cubes, this function calculates
    the number of hidden cubes when three surfaces are visible. -/
def hidden_cubes (l m n : ℕ) : ℕ := (l - 1) * (m - 1) * (n - 1)

/-- The total number of cubes in the rectangular block. -/
def total_cubes (l m n : ℕ) : ℕ := l * m * n

/-- Theorem stating that the smallest possible number of cubes in a rectangular block
    with 360 hidden cubes when three surfaces are visible is 560. -/
theorem smallest_block_with_360_hidden : 
  (∃ l m n : ℕ, 
    l > 1 ∧ m > 1 ∧ n > 1 ∧ 
    hidden_cubes l m n = 360 ∧
    (∀ l' m' n' : ℕ, 
      l' > 1 → m' > 1 → n' > 1 → 
      hidden_cubes l' m' n' = 360 → 
      total_cubes l m n ≤ total_cubes l' m' n')) ∧
  (∀ l m n : ℕ,
    l > 1 → m > 1 → n > 1 →
    hidden_cubes l m n = 360 →
    total_cubes l m n ≥ 560) := by
  sorry

end NUMINAMATH_CALUDE_smallest_block_with_360_hidden_l3088_308899


namespace NUMINAMATH_CALUDE_range_of_a_l3088_308876

-- Define sets A and B
def A : Set ℝ := {x | x > 5}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem range_of_a (h : ∀ x, x ∈ A → x ∈ B a) 
                   (h_not_nec : ∃ x, x ∈ B a ∧ x ∉ A) : 
  a > 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3088_308876


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l3088_308874

open Set

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | (x - 2) / (x + 1) ≤ 0}
def N : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- Define the interval (-1, 2]
def interval : Set ℝ := Ioc (-1) 2

-- Theorem statement
theorem intersection_equals_interval : M ∩ N = interval := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l3088_308874


namespace NUMINAMATH_CALUDE_cubic_polynomial_coefficient_l3088_308863

/-- 
Given a cubic polynomial y = ax³ + bx² + cx + d, 
if (1, y₁) and (-1, y₂) lie on its graph and y₁ - y₂ = -8, 
then a = -4.
-/
theorem cubic_polynomial_coefficient (a b c d y₁ y₂ : ℝ) : 
  y₁ = a + b + c + d → 
  y₂ = -a + b - c + d → 
  y₁ - y₂ = -8 → 
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_coefficient_l3088_308863


namespace NUMINAMATH_CALUDE_cubic_function_extrema_l3088_308870

/-- A function f with two extremum points on ℝ -/
def has_two_extrema (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ y : ℝ, (f y ≤ f x₁ ∨ f y ≥ f x₁) ∧ (f y ≤ f x₂ ∨ f y ≥ f x₂))

/-- The main theorem -/
theorem cubic_function_extrema (a : ℝ) :
  has_two_extrema (λ x : ℝ => x^3 + a*x) → a < 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_l3088_308870


namespace NUMINAMATH_CALUDE_wombat_claws_l3088_308867

theorem wombat_claws (num_wombats num_rheas total_claws : ℕ) 
  (h1 : num_wombats = 9)
  (h2 : num_rheas = 3)
  (h3 : total_claws = 39) :
  ∃ (wombat_claws : ℕ), 
    wombat_claws * num_wombats + num_rheas = total_claws ∧ 
    wombat_claws = 4 := by
  sorry

end NUMINAMATH_CALUDE_wombat_claws_l3088_308867


namespace NUMINAMATH_CALUDE_coefficients_of_equation_l3088_308849

/-- Given a quadratic equation ax² + bx + c = 0, this function returns its coefficients (a, b, c) -/
def quadratic_coefficients (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

/-- The coefficients of the quadratic equation 4x² - 6x + 1 = 0 are (4, -6, 1) -/
theorem coefficients_of_equation : quadratic_coefficients 4 (-6) 1 = (4, -6, 1) := by sorry

end NUMINAMATH_CALUDE_coefficients_of_equation_l3088_308849


namespace NUMINAMATH_CALUDE_trigonometric_equality_l3088_308847

theorem trigonometric_equality (α β : ℝ) 
  (h : (Real.cos α)^6 / (Real.cos β)^3 + (Real.sin α)^6 / (Real.sin β)^3 = 1) :
  (Real.sin β)^6 / (Real.sin α)^3 + (Real.cos β)^6 / (Real.cos α)^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l3088_308847


namespace NUMINAMATH_CALUDE_fence_painted_fraction_l3088_308897

/-- The fraction of a fence Tom paints while Jerry digs a hole -/
def fence_fraction (tom_rate jerry_rate : ℚ) : ℚ :=
  (jerry_rate / tom_rate)

theorem fence_painted_fraction :
  fence_fraction (1 / 60) (1 / 40) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fence_painted_fraction_l3088_308897


namespace NUMINAMATH_CALUDE_right_triangle_area_l3088_308850

/-- A line passing through two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- A right triangle formed by x-axis, y-axis, and a line -/
structure RightTriangle where
  line : Line

/-- Calculate the area of a right triangle -/
def area (t : RightTriangle) : ℝ :=
  sorry

theorem right_triangle_area :
  let l := Line.mk (-4, 8) (-8, 4)
  let t := RightTriangle.mk l
  area t = 72 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3088_308850


namespace NUMINAMATH_CALUDE_binary_1010101_conversion_l3088_308873

/-- Converts a binary number represented as a list of bits to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation as a list of digits. -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

/-- The binary representation of 1010101₂ -/
def binary_1010101 : List Bool := [true, false, true, false, true, false, true]

theorem binary_1010101_conversion :
  (binary_to_decimal binary_1010101 = 85) ∧
  (decimal_to_octal 85 = [5, 2, 1]) := by
sorry

end NUMINAMATH_CALUDE_binary_1010101_conversion_l3088_308873


namespace NUMINAMATH_CALUDE_pie_sugar_percentage_l3088_308875

/-- Given a pie weighing 200 grams with 50 grams of sugar, 
    prove that 75% of the pie is not sugar. -/
theorem pie_sugar_percentage (total_weight : ℝ) (sugar_weight : ℝ) 
    (h1 : total_weight = 200) 
    (h2 : sugar_weight = 50) : 
    (total_weight - sugar_weight) / total_weight * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_pie_sugar_percentage_l3088_308875


namespace NUMINAMATH_CALUDE_equal_squares_count_l3088_308891

/-- Represents a cell in the grid -/
inductive Cell
| White
| Black

/-- Represents the 5x5 grid -/
def Grid := Fin 5 → Fin 5 → Cell

/-- Defines the specific coloring pattern of the grid -/
def initial_grid : Grid :=
  fun i j => 
    if (i = 2 ∧ j = 2) ∨ 
       (i = 1 ∧ j = 3) ∨ 
       (i = 3 ∧ j = 1) ∨ 
       (i = 3 ∧ j = 3) ∨ 
       (i = 3 ∧ j = 5) 
    then Cell.Black 
    else Cell.White

/-- Checks if a square in the grid has equal number of black and white cells -/
def has_equal_cells (g : Grid) (top_left_i top_left_j size : Nat) : Bool :=
  sorry

/-- Counts the number of squares with equal black and white cells -/
def count_equal_squares (g : Grid) : Nat :=
  sorry

/-- The main theorem stating that there are exactly 16 squares with equal black and white cells -/
theorem equal_squares_count : count_equal_squares initial_grid = 16 :=
  sorry

end NUMINAMATH_CALUDE_equal_squares_count_l3088_308891


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3088_308806

/-- Given a line L1 with equation x - 2y + 3 = 0, prove that the line L2 with equation 2x + y - 3 = 0
    passes through the point (1, 1) and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  (x - 2*y + 3 = 0) →  -- L1 equation
  (2*1 + 1 - 3 = 0) ∧  -- L2 passes through (1, 1)
  (1 * 2 + (-2) * 1 = 0)  -- L1 and L2 are perpendicular (slope product = -1)
  :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3088_308806


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_planes_parallel_perpendicular_implies_perpendicular_l3088_308895

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (parallelLines : Line → Line → Prop)

-- Theorem 1: If m ⟂ α and m ∥ β, then α ⟂ β
theorem perpendicular_parallel_implies_perpendicular_planes
  (m : Line) (α β : Plane) :
  perpendicular m α → parallel m β → perpendicularPlanes α β :=
sorry

-- Theorem 2: If m ∥ n and m ⟂ α, then α ⟂ n
theorem parallel_perpendicular_implies_perpendicular
  (m n : Line) (α : Plane) :
  parallelLines m n → perpendicular m α → perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_planes_parallel_perpendicular_implies_perpendicular_l3088_308895


namespace NUMINAMATH_CALUDE_clara_three_times_anna_age_l3088_308812

/-- Proves that Clara was three times Anna's age 41 years ago -/
theorem clara_three_times_anna_age : ∃ x : ℕ, x = 41 ∧ 
  (80 : ℝ) - x = 3 * ((54 : ℝ) - x) := by
  sorry

end NUMINAMATH_CALUDE_clara_three_times_anna_age_l3088_308812


namespace NUMINAMATH_CALUDE_sandys_remaining_nickels_l3088_308877

/-- Given an initial number of nickels and a number of borrowed nickels,
    calculate the remaining nickels. -/
def remaining_nickels (initial : ℕ) (borrowed : ℕ) : ℕ :=
  initial - borrowed

/-- Theorem stating that Sandy's remaining nickels is 11 -/
theorem sandys_remaining_nickels :
  remaining_nickels 31 20 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sandys_remaining_nickels_l3088_308877


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3088_308893

theorem quadratic_real_roots (k : ℕ) : 
  (∃ x : ℝ, k * x^2 - 3 * x + 2 = 0) ↔ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3088_308893


namespace NUMINAMATH_CALUDE_subset_union_existence_l3088_308890

theorem subset_union_existence (n m : ℕ) (h1 : n ≥ 3) (h2 : m ≥ 2^(n-1) + 1) :
  ∀ (A : Fin m → Set (Fin n)), 
    (∀ j, A j ≠ ∅) → 
    (∀ i j, i ≠ j → A i ≠ A j) → 
    ∃ i j k, A i ∪ A j = A k := by
  sorry

end NUMINAMATH_CALUDE_subset_union_existence_l3088_308890


namespace NUMINAMATH_CALUDE_sandbox_side_length_l3088_308819

/-- Represents the properties of a square sandbox. -/
structure Sandbox where
  sandPerArea : Real  -- Pounds of sand per square inch
  totalSand : Real    -- Total pounds of sand needed
  sideLength : Real   -- Length of each side in inches

/-- 
Theorem: Given a square sandbox where 30 pounds of sand fills 80 square inches,
and 600 pounds of sand fills the entire sandbox, the length of each side is 40 inches.
-/
theorem sandbox_side_length (sb : Sandbox)
  (h1 : sb.sandPerArea = 30 / 80)
  (h2 : sb.totalSand = 600) :
  sb.sideLength = 40 := by
  sorry


end NUMINAMATH_CALUDE_sandbox_side_length_l3088_308819


namespace NUMINAMATH_CALUDE_hash_property_l3088_308818

/-- Operation # for non-negative integers -/
def hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

/-- Theorem stating that if a # b = 100, then (a + b) + 6 = 11 -/
theorem hash_property (a b : ℕ) (h : hash a b = 100) : (a + b) + 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_hash_property_l3088_308818


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l3088_308857

theorem hemisphere_surface_area (r : ℝ) (h : r = 5) :
  let sphere_area (r : ℝ) := 4 * π * r^2
  let hemisphere_curved_area (r : ℝ) := (sphere_area r) / 2
  let base_area (r : ℝ) := π * r^2
  hemisphere_curved_area r + base_area r = 75 * π := by
sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l3088_308857


namespace NUMINAMATH_CALUDE_vloggers_earnings_per_view_l3088_308828

/-- Represents the earnings and viewership of a vlogger -/
structure Vlogger where
  name : String
  daily_viewers : ℕ
  weekly_earnings : ℚ

/-- Calculates the earnings per view for a vlogger -/
def earnings_per_view (v : Vlogger) : ℚ :=
  v.weekly_earnings / (v.daily_viewers * 7)

theorem vloggers_earnings_per_view 
  (voltaire leila : Vlogger)
  (h1 : voltaire.daily_viewers = 50)
  (h2 : leila.daily_viewers = 2 * voltaire.daily_viewers)
  (h3 : leila.weekly_earnings = 350) :
  earnings_per_view voltaire = earnings_per_view leila ∧ 
  earnings_per_view voltaire = 1/2 := by
  sorry

#check vloggers_earnings_per_view

end NUMINAMATH_CALUDE_vloggers_earnings_per_view_l3088_308828


namespace NUMINAMATH_CALUDE_weekly_earnings_is_1454000_l3088_308826

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the number of computers produced on a given day -/
def production_rate (d : Day) : ℕ :=
  match d with
  | Day.Monday    => 1200
  | Day.Tuesday   => 1500
  | Day.Wednesday => 1800
  | Day.Thursday  => 1600
  | Day.Friday    => 1400
  | Day.Saturday  => 1000
  | Day.Sunday    => 800

/-- Returns the selling price per computer on a given day -/
def selling_price (d : Day) : ℕ :=
  match d with
  | Day.Monday    => 150
  | Day.Tuesday   => 160
  | Day.Wednesday => 170
  | Day.Thursday  => 155
  | Day.Friday    => 145
  | Day.Saturday  => 165
  | Day.Sunday    => 140

/-- Calculates the earnings for a given day -/
def daily_earnings (d : Day) : ℕ :=
  production_rate d * selling_price d

/-- Calculates the total earnings for the week -/
def total_weekly_earnings : ℕ :=
  daily_earnings Day.Monday +
  daily_earnings Day.Tuesday +
  daily_earnings Day.Wednesday +
  daily_earnings Day.Thursday +
  daily_earnings Day.Friday +
  daily_earnings Day.Saturday +
  daily_earnings Day.Sunday

/-- Theorem stating that the total weekly earnings is $1,454,000 -/
theorem weekly_earnings_is_1454000 :
  total_weekly_earnings = 1454000 := by
  sorry

end NUMINAMATH_CALUDE_weekly_earnings_is_1454000_l3088_308826


namespace NUMINAMATH_CALUDE_days_to_complete_correct_l3088_308823

/-- The number of days required for a given number of men to complete a work,
    given that 12 men can do it in 80 days and 16 men can do it in 60 days. -/
def days_to_complete (num_men : ℕ) : ℚ :=
  960 / num_men

/-- Theorem stating that the number of days required for any number of men
    to complete the work is correctly given by the days_to_complete function,
    based on the given conditions. -/
theorem days_to_complete_correct (num_men : ℕ) (num_men_pos : 0 < num_men) :
  days_to_complete num_men * num_men = 960 ∧
  days_to_complete 12 = 80 ∧
  days_to_complete 16 = 60 :=
by sorry

end NUMINAMATH_CALUDE_days_to_complete_correct_l3088_308823


namespace NUMINAMATH_CALUDE_complex_expression_value_l3088_308844

theorem complex_expression_value : 
  let expr := (4.7 * 13.26 + 4.7 * 9.43 + 4.7 * 77.31) * Real.exp 3.5 + Real.log (Real.sin 0.785)
  ∃ ε > 0, |expr - 15563.91492641| < ε :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_value_l3088_308844


namespace NUMINAMATH_CALUDE_wrong_mark_value_l3088_308858

/-- Proves that the wrongly entered mark is 85 given the conditions of the problem -/
theorem wrong_mark_value (n : ℕ) (correct_mark : ℕ) (average_increase : ℚ) 
  (h1 : n = 104) 
  (h2 : correct_mark = 33) 
  (h3 : average_increase = 1/2) : 
  ∃ x : ℕ, x = 85 ∧ (x - correct_mark : ℚ) = average_increase * n := by
  sorry

end NUMINAMATH_CALUDE_wrong_mark_value_l3088_308858


namespace NUMINAMATH_CALUDE_min_apples_in_basket_l3088_308866

theorem min_apples_in_basket : ∃ n : ℕ, n ≥ 23 ∧ 
  (∃ a b c : ℕ, 
    n + 4 = 3 * a ∧
    2 * a + 4 = 3 * b ∧
    2 * b + 4 = 3 * c) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ a b c : ℕ, 
      m + 4 = 3 * a ∧
      2 * a + 4 = 3 * b ∧
      2 * b + 4 = 3 * c)) :=
by sorry

end NUMINAMATH_CALUDE_min_apples_in_basket_l3088_308866


namespace NUMINAMATH_CALUDE_cone_volume_l3088_308882

/-- A cone with base area π and lateral surface in the shape of a semicircle has volume (√3 / 3)π -/
theorem cone_volume (r h l : ℝ) : 
  r > 0 → h > 0 → l > 0 →
  π * r^2 = π →
  π * l = 2 * π * r →
  h^2 + r^2 = l^2 →
  (1/3) * π * r^2 * h = (Real.sqrt 3 / 3) * π := by
sorry

end NUMINAMATH_CALUDE_cone_volume_l3088_308882


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l3088_308832

/-- The perimeter of a T shape formed by two rectangles -/
def t_perimeter (width height overlap : ℝ) : ℝ :=
  2 * (width + height - 2 * overlap) + 2 * height

/-- Theorem: The perimeter of the T shape is 20 inches -/
theorem t_shape_perimeter :
  let width := 3
  let height := 5
  let overlap := 1.5
  t_perimeter width height overlap = 20 := by
sorry

#eval t_perimeter 3 5 1.5

end NUMINAMATH_CALUDE_t_shape_perimeter_l3088_308832


namespace NUMINAMATH_CALUDE_sets_A_and_B_proof_l3088_308813

def U : Set Nat := {x | x ≤ 20 ∧ Nat.Prime x}

theorem sets_A_and_B_proof (A B : Set Nat) 
  (h1 : A ∩ (U \ B) = {3, 5})
  (h2 : B ∩ (U \ A) = {7, 19})
  (h3 : (U \ A) ∩ (U \ B) = {2, 17}) :
  A = {3, 5, 11, 13} ∧ B = {7, 11, 13, 19} := by
sorry

end NUMINAMATH_CALUDE_sets_A_and_B_proof_l3088_308813


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l3088_308854

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- Represents a trapezoid -/
structure Trapezoid :=
  (H : Point) (I : Point) (J : Point) (K : Point)

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Checks if two lines are parallel -/
def areParallel (p1 p2 q1 q2 : Point) : Prop := sorry

/-- Checks if four points are equally spaced on a line -/
def equallySpaced (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Calculates the area of a trapezoid -/
def trapezoidArea (t : Trapezoid) : ℝ := sorry

theorem area_ratio_theorem (ABC : Triangle) (HIJK : Trapezoid) 
  (D E F G : Point) :
  isEquilateral ABC →
  areParallel D E B C →
  areParallel F G B C →
  areParallel HIJK.H HIJK.I B C →
  areParallel HIJK.J HIJK.K B C →
  equallySpaced ABC.A D F HIJK.H →
  equallySpaced ABC.A D F HIJK.J →
  trapezoidArea HIJK / triangleArea ABC = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l3088_308854


namespace NUMINAMATH_CALUDE_midpoint_x_sum_eq_vertex_x_sum_l3088_308872

/-- Given a triangle in the Cartesian plane, the sum of the x-coordinates of the midpoints
    of its sides is equal to the sum of the x-coordinates of its vertices. -/
theorem midpoint_x_sum_eq_vertex_x_sum (a b c : ℝ) : 
  let vertex_sum := a + b + c
  let midpoint_sum := (a + b) / 2 + (a + c) / 2 + (b + c) / 2
  midpoint_sum = vertex_sum :=
by sorry

end NUMINAMATH_CALUDE_midpoint_x_sum_eq_vertex_x_sum_l3088_308872


namespace NUMINAMATH_CALUDE_sara_pumpkins_left_l3088_308834

/-- Given that Sara grew 43 pumpkins and rabbits ate 23 pumpkins, 
    prove that Sara has 20 pumpkins left. -/
theorem sara_pumpkins_left : 
  let total_grown : ℕ := 43
  let eaten_by_rabbits : ℕ := 23
  let pumpkins_left := total_grown - eaten_by_rabbits
  pumpkins_left = 20 := by sorry

end NUMINAMATH_CALUDE_sara_pumpkins_left_l3088_308834


namespace NUMINAMATH_CALUDE_calculation_proof_l3088_308851

theorem calculation_proof :
  (1 : ℚ) * (5 / 7 : ℚ) * (-4 - 2/3 : ℚ) / (1 + 2/3 : ℚ) = -2 ∧
  (-2 - 1/7 : ℚ) / (-1.2 : ℚ) * (-1 - 2/5 : ℚ) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3088_308851


namespace NUMINAMATH_CALUDE_bug_meeting_point_l3088_308846

theorem bug_meeting_point (PQ QR RP : ℝ) (h1 : PQ = 7) (h2 : QR = 8) (h3 : RP = 9) :
  let perimeter := PQ + QR + RP
  let distance_traveled := 10
  let QS := distance_traveled - PQ
  QS = 3 := by sorry

end NUMINAMATH_CALUDE_bug_meeting_point_l3088_308846


namespace NUMINAMATH_CALUDE_red_shells_count_l3088_308853

theorem red_shells_count (total : ℕ) (green : ℕ) (not_red_or_green : ℕ) 
  (h1 : total = 291)
  (h2 : green = 49)
  (h3 : not_red_or_green = 166) :
  total - green - not_red_or_green = 76 := by
sorry

end NUMINAMATH_CALUDE_red_shells_count_l3088_308853


namespace NUMINAMATH_CALUDE_larry_dog_time_l3088_308860

/-- The number of minutes in half an hour -/
def half_hour : ℕ := 30

/-- The number of minutes spent feeding the dog daily -/
def feeding_time : ℕ := 12

/-- The total number of minutes Larry spends on his dog daily -/
def total_time : ℕ := 72

/-- The number of sessions Larry spends walking and playing with his dog daily -/
def walking_playing_sessions : ℕ := 2

theorem larry_dog_time :
  half_hour * walking_playing_sessions + feeding_time = total_time :=
sorry

end NUMINAMATH_CALUDE_larry_dog_time_l3088_308860


namespace NUMINAMATH_CALUDE_max_digit_sum_three_digit_number_l3088_308830

theorem max_digit_sum_three_digit_number (a b c : ℕ) : 
  a < 10 → b < 10 → c < 10 →
  (100 * a + 10 * b + c) + (100 * a + 10 * c + b) = 1732 →
  a + b + c ≤ 20 := by
sorry

end NUMINAMATH_CALUDE_max_digit_sum_three_digit_number_l3088_308830


namespace NUMINAMATH_CALUDE_books_borrowed_second_day_l3088_308821

def initial_books : ℕ := 100
def people_first_day : ℕ := 5
def books_per_person : ℕ := 2
def remaining_books : ℕ := 70

theorem books_borrowed_second_day :
  initial_books - people_first_day * books_per_person - remaining_books = 20 :=
by sorry

end NUMINAMATH_CALUDE_books_borrowed_second_day_l3088_308821


namespace NUMINAMATH_CALUDE_inequality_proof_l3088_308888

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  Real.sqrt (3 * x^2 + x * y) + Real.sqrt (3 * y^2 + y * z) + Real.sqrt (3 * z^2 + z * x) ≤ 2 * (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3088_308888


namespace NUMINAMATH_CALUDE_candy_store_revenue_l3088_308869

/-- Calculates the revenue of a candy store given specific sales conditions --/
theorem candy_store_revenue :
  let fudge_pounds : ℕ := 37
  let fudge_price : ℚ := 5/2
  let truffle_count : ℕ := 82
  let truffle_price : ℚ := 3/2
  let pretzel_count : ℕ := 48
  let pretzel_price : ℚ := 2
  let fudge_discount : ℚ := 1/10
  let sales_tax : ℚ := 1/20
  let truffle_promo : ℕ := 3  -- buy 3, get 1 free

  let fudge_revenue := (1 - fudge_discount) * (fudge_pounds : ℚ) * fudge_price
  let truffle_revenue := (truffle_count - truffle_count / (truffle_promo + 1)) * truffle_price
  let pretzel_revenue := (pretzel_count : ℚ) * pretzel_price
  
  let total_before_tax := fudge_revenue + truffle_revenue + pretzel_revenue
  let total_after_tax := total_before_tax * (1 + sales_tax)

  total_after_tax = 28586 / 100
  := by sorry

end NUMINAMATH_CALUDE_candy_store_revenue_l3088_308869


namespace NUMINAMATH_CALUDE_bananas_count_l3088_308807

/-- Represents the contents of a fruit bowl -/
structure FruitBowl where
  apples : ℕ
  pears : ℕ
  bananas : ℕ

/-- Conditions for the fruit bowl problem -/
def fruitBowlConditions (bowl : FruitBowl) : Prop :=
  bowl.pears = bowl.apples + 2 ∧
  bowl.bananas = bowl.pears + 3 ∧
  bowl.apples + bowl.pears + bowl.bananas = 19

/-- Theorem stating that under the given conditions, the number of bananas is 9 -/
theorem bananas_count (bowl : FruitBowl) : 
  fruitBowlConditions bowl → bowl.bananas = 9 := by
  sorry


end NUMINAMATH_CALUDE_bananas_count_l3088_308807


namespace NUMINAMATH_CALUDE_matrix_determinant_l3088_308859

theorem matrix_determinant (x y : ℝ) : 
  Matrix.det ![![x, x, y], ![x, y, x], ![y, x, x]] = 3 * x^2 * y - 2 * x^3 - y^3 := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_l3088_308859
