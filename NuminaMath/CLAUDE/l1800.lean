import Mathlib

namespace NUMINAMATH_CALUDE_bianca_extra_flowers_l1800_180000

/-- The number of extra flowers Bianca picked -/
def extra_flowers (tulips roses used : ℕ) : ℕ :=
  tulips + roses - used

/-- Theorem stating that Bianca picked 7 extra flowers -/
theorem bianca_extra_flowers :
  extra_flowers 39 49 81 = 7 := by
  sorry

end NUMINAMATH_CALUDE_bianca_extra_flowers_l1800_180000


namespace NUMINAMATH_CALUDE_higher_variance_greater_fluctuations_l1800_180094

-- Define the properties of the two data sets
def mean_A : ℝ := 5
def mean_B : ℝ := 5
def variance_A : ℝ := 0.1
def variance_B : ℝ := 0.2

-- Define a function to represent fluctuations based on variance
def fluctuations (variance : ℝ) : ℝ := variance

-- Theorem stating that higher variance implies greater fluctuations
theorem higher_variance_greater_fluctuations :
  variance_A < variance_B →
  fluctuations variance_A < fluctuations variance_B :=
by sorry

end NUMINAMATH_CALUDE_higher_variance_greater_fluctuations_l1800_180094


namespace NUMINAMATH_CALUDE_angle_with_double_supplement_is_60_degrees_l1800_180061

theorem angle_with_double_supplement_is_60_degrees (α : Real) :
  (180 - α = 2 * α) → α = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_double_supplement_is_60_degrees_l1800_180061


namespace NUMINAMATH_CALUDE_odd_expression_proof_l1800_180081

theorem odd_expression_proof (p q : ℕ) (hp : Odd p) (hq : Odd q) (hp_pos : p > 0) (hq_pos : q > 0) :
  Odd (4 * p^2 + 2 * q^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_expression_proof_l1800_180081


namespace NUMINAMATH_CALUDE_emily_phone_bill_l1800_180076

/-- Calculates the total cost of a cell phone plan based on usage. -/
def calculate_total_cost (base_cost : ℚ) (included_hours : ℚ) (text_cost : ℚ) 
  (extra_minute_cost : ℚ) (data_cost : ℚ) (texts_sent : ℚ) (hours_used : ℚ) 
  (data_used : ℚ) : ℚ :=
  base_cost + 
  (text_cost * texts_sent) + 
  (max (hours_used - included_hours) 0 * 60 * extra_minute_cost) + 
  (data_cost * data_used)

theorem emily_phone_bill :
  let base_cost : ℚ := 25
  let included_hours : ℚ := 25
  let text_cost : ℚ := 0.1
  let extra_minute_cost : ℚ := 0.15
  let data_cost : ℚ := 2
  let texts_sent : ℚ := 150
  let hours_used : ℚ := 26
  let data_used : ℚ := 3
  calculate_total_cost base_cost included_hours text_cost extra_minute_cost 
    data_cost texts_sent hours_used data_used = 55 := by
  sorry

end NUMINAMATH_CALUDE_emily_phone_bill_l1800_180076


namespace NUMINAMATH_CALUDE_shepherd_sheep_problem_l1800_180031

/-- The number of sheep with the shepherd boy on the mountain -/
def x : ℕ := 20

/-- The number of sheep with the shepherd boy at the foot of the mountain -/
def y : ℕ := 12

/-- Theorem stating that the given numbers of sheep satisfy the problem conditions -/
theorem shepherd_sheep_problem :
  (x - 4 = y + 4) ∧ (x + 4 = 3 * (y - 4)) := by
  sorry

#check shepherd_sheep_problem

end NUMINAMATH_CALUDE_shepherd_sheep_problem_l1800_180031


namespace NUMINAMATH_CALUDE_bride_groom_age_sum_l1800_180024

theorem bride_groom_age_sum :
  ∀ (groom_age bride_age : ℕ),
    groom_age = 83 →
    bride_age = groom_age + 19 →
    groom_age + bride_age = 185 :=
by
  sorry

end NUMINAMATH_CALUDE_bride_groom_age_sum_l1800_180024


namespace NUMINAMATH_CALUDE_intersection_points_l1800_180006

/-- The intersection points of two cubic and quadratic functions -/
theorem intersection_points
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let g (x : ℝ) := -a * x^3 + b * x + c
  ∃ (x₁ x₂ : ℝ), (x₁ = 0 ∧ f x₁ = g x₁) ∧ (x₂ = -1 ∧ f x₂ = g x₂) ∧
    ∀ (x : ℝ), f x = g x → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_l1800_180006


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l1800_180030

-- Define the repeating decimal
def repeating_decimal : ℚ := 7 + 17 / 990

-- Theorem statement
theorem repeating_decimal_fraction_sum :
  (repeating_decimal = 710 / 99) ∧
  (710 + 99 = 809) :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l1800_180030


namespace NUMINAMATH_CALUDE_factorization_3ax2_minus_3ay2_l1800_180084

theorem factorization_3ax2_minus_3ay2 (a x y : ℝ) : 3*a*x^2 - 3*a*y^2 = 3*a*(x+y)*(x-y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3ax2_minus_3ay2_l1800_180084


namespace NUMINAMATH_CALUDE_right_triangle_equality_l1800_180088

/-- In a right triangle ABC with point M on the hypotenuse, if BM + MA = BC + CA,
    MB = x, CB = 2h, and CA = d, then x = hd / (2h + d) -/
theorem right_triangle_equality (h d x : ℝ) :
  h > 0 → d > 0 →
  x > 0 →
  x + Real.sqrt ((x + 2*h)^2 + d^2) = 2*h + d →
  x = h * d / (2*h + d) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_equality_l1800_180088


namespace NUMINAMATH_CALUDE_a4_to_a5_booklet_l1800_180097

theorem a4_to_a5_booklet (n : ℕ) (h : 2 * n + 2 = 74) : n / 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_a4_to_a5_booklet_l1800_180097


namespace NUMINAMATH_CALUDE_least_integer_with_divisibility_l1800_180008

def n : ℕ := 2329089562800

theorem least_integer_with_divisibility (k : ℕ) (hk : k < n) : 
  (∀ i ∈ Finset.range 18, n % (i + 1) = 0) ∧ 
  (∀ i ∈ Finset.range 10, n % (i + 21) = 0) ∧ 
  n % 19 ≠ 0 ∧ 
  n % 20 ≠ 0 → 
  ¬(∀ i ∈ Finset.range 18, k % (i + 1) = 0) ∨ 
  ¬(∀ i ∈ Finset.range 10, k % (i + 21) = 0) ∨ 
  k % 19 = 0 ∨ 
  k % 20 = 0 :=
by sorry

#check least_integer_with_divisibility

end NUMINAMATH_CALUDE_least_integer_with_divisibility_l1800_180008


namespace NUMINAMATH_CALUDE_dice_probability_l1800_180078

/-- The number of sides on a standard die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The number of dice re-rolled -/
def numReRolled : ℕ := 3

/-- The probability of a single re-rolled die matching the set-aside pair -/
def probSingleMatch : ℚ := 1 / numSides

/-- The probability of all re-rolled dice matching the set-aside pair -/
def probAllMatch : ℚ := probSingleMatch ^ numReRolled

theorem dice_probability : probAllMatch = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l1800_180078


namespace NUMINAMATH_CALUDE_f_min_value_a_range_l1800_180059

-- Define the function f
def f (x : ℝ) : ℝ := 2 * abs (x - 2) - x + 5

-- Theorem for the minimum value of f
theorem f_min_value :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 3 :=
sorry

-- Theorem for the range of a
theorem a_range (a : ℝ) :
  (∀ x, |x - a| + |x + 2| ≥ 3) → (a ≤ -5 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_f_min_value_a_range_l1800_180059


namespace NUMINAMATH_CALUDE_mrs_martin_bagels_l1800_180057

/-- The cost of one bagel in dollars -/
def bagel_cost : ℚ := 3/2

/-- Mrs. Martin's purchase -/
def mrs_martin_purchase (coffee_cost bagels : ℚ) : Prop :=
  3 * coffee_cost + bagels * bagel_cost = 51/4

/-- Mr. Martin's purchase -/
def mr_martin_purchase (coffee_cost : ℚ) : Prop :=
  2 * coffee_cost + 5 * bagel_cost = 14

theorem mrs_martin_bagels :
  ∃ (coffee_cost : ℚ), mr_martin_purchase coffee_cost →
    mrs_martin_purchase coffee_cost 2 := by sorry

end NUMINAMATH_CALUDE_mrs_martin_bagels_l1800_180057


namespace NUMINAMATH_CALUDE_riverview_village_l1800_180062

theorem riverview_village (p h s c d : ℕ) : 
  p = 4 * h → 
  s = 5 * c → 
  d = 4 * p → 
  ¬∃ (h c : ℕ), 52 = 21 * h + 6 * c :=
by sorry

end NUMINAMATH_CALUDE_riverview_village_l1800_180062


namespace NUMINAMATH_CALUDE_replaced_person_weight_l1800_180085

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (average_increase : ℝ) (weight_of_new_person : ℝ) : ℝ :=
  weight_of_new_person - 2 * average_increase

/-- Theorem stating that the weight of the replaced person is 65 kg -/
theorem replaced_person_weight :
  weight_of_replaced_person 4.5 74 = 65 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l1800_180085


namespace NUMINAMATH_CALUDE_stock_transaction_l1800_180064

/-- Represents the number of shares for each stock --/
structure StockHoldings where
  v : ℕ
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the range of a set of numbers --/
def range (s : StockHoldings) : ℕ :=
  max s.v (max s.w (max s.x (max s.y s.z))) - min s.v (min s.w (min s.x (min s.y s.z)))

/-- Theorem representing the stock transaction problem --/
theorem stock_transaction (initial : StockHoldings) 
  (h1 : initial.v = 68)
  (h2 : initial.w = 112)
  (h3 : initial.x = 56)
  (h4 : initial.y = 94)
  (h5 : initial.z = 45)
  (bought_y : ℕ)
  (h6 : bought_y = 23)
  (range_increase : ℕ)
  (h7 : range_increase = 14)
  : ∃ (sold_x : ℕ), 
    let final := StockHoldings.mk 
      initial.v 
      initial.w 
      (initial.x - sold_x)
      (initial.y + bought_y)
      initial.z
    range final = range initial + range_increase ∧ sold_x = 20 := by
  sorry


end NUMINAMATH_CALUDE_stock_transaction_l1800_180064


namespace NUMINAMATH_CALUDE_als_original_portion_l1800_180035

theorem als_original_portion (al betty clare : ℕ) : 
  al + betty + clare = 1200 →
  al ≠ betty →
  al ≠ clare →
  betty ≠ clare →
  al - 150 + 3 * betty + 3 * clare = 1800 →
  al = 825 := by
sorry

end NUMINAMATH_CALUDE_als_original_portion_l1800_180035


namespace NUMINAMATH_CALUDE_first_term_of_constant_ratio_l1800_180005

def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a + (n - 1 : ℚ) * d) / 2

theorem first_term_of_constant_ratio (d : ℚ) (h : d = 5) :
  (∃ k : ℚ, ∀ n : ℕ, n > 0 → 
    arithmetic_sum a d (4*n) / arithmetic_sum a d n = k) →
  a = -5/2 :=
sorry

end NUMINAMATH_CALUDE_first_term_of_constant_ratio_l1800_180005


namespace NUMINAMATH_CALUDE_circular_seating_arrangement_l1800_180017

/-- Given a circular seating arrangement where the 7th person is directly opposite the 27th person,
    prove that the total number of people in the circle is 40. -/
theorem circular_seating_arrangement (n : ℕ) : n = 40 := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_arrangement_l1800_180017


namespace NUMINAMATH_CALUDE_pentagon_area_form_pentagon_area_sum_l1800_180003

/-- A pentagon constructed from 15 line segments of length 3 -/
structure Pentagon :=
  (F G H I J : ℝ × ℝ)
  (segments : List (ℝ × ℝ))
  (segment_length : ℝ)
  (segment_count : ℕ)
  (is_valid : segment_count = 15 ∧ segment_length = 3)

/-- The area of the pentagon -/
def pentagon_area (p : Pentagon) : ℝ := sorry

/-- The area can be expressed as √p + √q where p and q are positive integers -/
theorem pentagon_area_form (p : Pentagon) : 
  ∃ (a b : ℕ), pentagon_area p = Real.sqrt a + Real.sqrt b ∧ a > 0 ∧ b > 0 := sorry

/-- The sum of p and q is 48 -/
theorem pentagon_area_sum (p : Pentagon) :
  ∃ (a b : ℕ), pentagon_area p = Real.sqrt a + Real.sqrt b ∧ a > 0 ∧ b > 0 ∧ a + b = 48 := sorry

end NUMINAMATH_CALUDE_pentagon_area_form_pentagon_area_sum_l1800_180003


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1800_180049

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 + x - 12 ≥ 0 ↔ x ≤ -4 ∨ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1800_180049


namespace NUMINAMATH_CALUDE_second_number_value_l1800_180092

theorem second_number_value (A B C : ℚ) 
  (sum_eq : A + B + C = 98)
  (ratio_AB : A = (2/3) * B)
  (ratio_BC : C = (8/5) * B) : 
  B = 30 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l1800_180092


namespace NUMINAMATH_CALUDE_concession_stand_soda_cost_l1800_180060

/-- Proves that the cost of each soda is $0.50 given the conditions of the concession stand problem -/
theorem concession_stand_soda_cost 
  (total_revenue : ℝ)
  (total_items : ℕ)
  (hot_dogs_sold : ℕ)
  (hot_dog_cost : ℝ)
  (h1 : total_revenue = 78.50)
  (h2 : total_items = 87)
  (h3 : hot_dogs_sold = 35)
  (h4 : hot_dog_cost = 1.50) :
  let soda_cost := (total_revenue - hot_dogs_sold * hot_dog_cost) / (total_items - hot_dogs_sold)
  soda_cost = 0.50 := by
    sorry

#check concession_stand_soda_cost

end NUMINAMATH_CALUDE_concession_stand_soda_cost_l1800_180060


namespace NUMINAMATH_CALUDE_determinant_equality_l1800_180034

theorem determinant_equality (a x y : ℝ) : 
  Matrix.det ![![1, x^2, y], ![1, a*x + y, y^2], ![1, x^2, a*x + y]] = 
    a^2*x^2 + 2*a*x*y + y^2 - a*x^3 - x*y^2 := by sorry

end NUMINAMATH_CALUDE_determinant_equality_l1800_180034


namespace NUMINAMATH_CALUDE_triangle_properties_triangle_is_equilateral_l1800_180043

-- Define the triangle ABC
structure Triangle where
  a : ℝ  -- side opposite to angle A
  b : ℝ  -- side opposite to angle B
  c : ℝ  -- side opposite to angle C
  angleA : ℝ  -- measure of angle A
  angleB : ℝ  -- measure of angle B
  angleC : ℝ  -- measure of angle C

-- Define the theorem
theorem triangle_properties (t : Triangle)
  (h1 : (t.a + t.b + t.c) * (t.a - t.b - t.c) + 3 * t.b * t.c = 0)
  (h2 : t.a = 2 * t.c * Real.cos t.angleB) :
  t.angleA = π / 3 ∧ t.angleB = t.angleC := by
  sorry

-- Define what it means for a triangle to be equilateral
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- Prove that the triangle is equilateral
theorem triangle_is_equilateral (t : Triangle)
  (h1 : (t.a + t.b + t.c) * (t.a - t.b - t.c) + 3 * t.b * t.c = 0)
  (h2 : t.a = 2 * t.c * Real.cos t.angleB) :
  is_equilateral t := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_triangle_is_equilateral_l1800_180043


namespace NUMINAMATH_CALUDE_company_fund_problem_l1800_180095

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) : 
  (initial_fund = 60 * n - 10) →  -- The fund was $10 short of giving $60 to each employee
  (initial_fund = 50 * n + 150) → -- After giving $50 to each employee, $150 remained
  initial_fund = 950 := by
  sorry

end NUMINAMATH_CALUDE_company_fund_problem_l1800_180095


namespace NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l1800_180020

theorem min_value_x_plus_four_over_x (x : ℝ) (hx : x > 0) :
  x + 4 / x ≥ 4 ∧ ∃ y > 0, y + 4 / y = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l1800_180020


namespace NUMINAMATH_CALUDE_count_self_inverse_pairs_l1800_180058

/-- A 2x2 matrix of the form [[a, 4], [-9, d]] -/
def special_matrix (a d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![a, 4; -9, d]

/-- The identity matrix of size 2x2 -/
def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 0; 0, 1]

/-- Predicate to check if a matrix is its own inverse -/
def is_self_inverse (a d : ℝ) : Prop :=
  special_matrix a d * special_matrix a d = identity_matrix

/-- The set of all pairs (a, d) where the special matrix is its own inverse -/
def self_inverse_pairs : Set (ℝ × ℝ) :=
  {p | is_self_inverse p.1 p.2}

theorem count_self_inverse_pairs :
  ∃ (s : Finset (ℝ × ℝ)), s.card = 2 ∧ ↑s = self_inverse_pairs :=
sorry

end NUMINAMATH_CALUDE_count_self_inverse_pairs_l1800_180058


namespace NUMINAMATH_CALUDE_artists_contemporary_probability_l1800_180090

/-- Represents the birth year of an artist, measured in years ago --/
def BirthYear := Fin 301

/-- Represents the lifetime of an artist --/
structure Lifetime where
  birth : BirthYear
  death : BirthYear
  age_constraint : death.val = birth.val + 80

/-- Two artists are contemporaries if their lifetimes overlap --/
def are_contemporaries (a b : Lifetime) : Prop :=
  (a.birth.val ≤ b.death.val ∧ b.birth.val ≤ a.death.val) ∨
  (b.birth.val ≤ a.death.val ∧ a.birth.val ≤ b.death.val)

/-- The probability of two artists being contemporaries --/
def probability_contemporaries : ℚ :=
  209 / 225

theorem artists_contemporary_probability :
  probability_contemporaries = 209 / 225 := by sorry


end NUMINAMATH_CALUDE_artists_contemporary_probability_l1800_180090


namespace NUMINAMATH_CALUDE_nitrogen_electron_count_hydrazine_N2O4_reaction_hydrazine_combustion_l1800_180048

-- Define the chemical reactions and their enthalpies
def reaction1_enthalpy : ℝ := -19.5
def reaction2_enthalpy : ℝ := -534.2
def reaction3_enthalpy : ℝ := 44.0

-- Define the number of electrons in the L shell of a nitrogen atom
def nitrogen_L_shell_electrons : ℕ := 5

-- Define the enthalpy of the reaction between hydrazine and N₂O₄
def hydrazine_N2O4_reaction_enthalpy : ℝ := -1048.9

-- Define the combustion heat of hydrazine
def hydrazine_combustion_heat : ℝ := -622.2

-- Theorem statements
theorem nitrogen_electron_count :
  nitrogen_L_shell_electrons = 5 := by sorry

theorem hydrazine_N2O4_reaction :
  hydrazine_N2O4_reaction_enthalpy = 2 * reaction2_enthalpy - reaction1_enthalpy := by sorry

theorem hydrazine_combustion :
  hydrazine_combustion_heat = reaction2_enthalpy - 2 * reaction3_enthalpy := by sorry

end NUMINAMATH_CALUDE_nitrogen_electron_count_hydrazine_N2O4_reaction_hydrazine_combustion_l1800_180048


namespace NUMINAMATH_CALUDE_milburg_adults_l1800_180045

theorem milburg_adults (total_population children : ℕ) 
  (h1 : total_population = 5256)
  (h2 : children = 2987) :
  total_population - children = 2269 := by
  sorry

end NUMINAMATH_CALUDE_milburg_adults_l1800_180045


namespace NUMINAMATH_CALUDE_range_of_a_perpendicular_case_l1800_180077

-- Define the line and hyperbola
def line (a : ℝ) (x : ℝ) : ℝ := a * x + 1
def hyperbola (x y : ℝ) : Prop := 3 * x^2 - y^2 = 1

-- Define the intersection condition
def intersects (a : ℝ) : Prop := ∃ x y, hyperbola x y ∧ y = line a x

-- Define the range of a
def valid_range (a : ℝ) : Prop := -Real.sqrt 6 < a ∧ a < Real.sqrt 6 ∧ a ≠ Real.sqrt 3 ∧ a ≠ -Real.sqrt 3

-- Define the perpendicularity condition
def perpendicular (a : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂, 
  hyperbola x₁ y₁ ∧ y₁ = line a x₁ ∧
  hyperbola x₂ y₂ ∧ y₂ = line a x₂ ∧
  x₁ * x₂ + y₁ * y₂ = 0

-- Theorem 1: Range of a
theorem range_of_a : ∀ a : ℝ, intersects a ↔ valid_range a :=
sorry

-- Theorem 2: Perpendicular case
theorem perpendicular_case : ∀ a : ℝ, perpendicular a ↔ (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_perpendicular_case_l1800_180077


namespace NUMINAMATH_CALUDE_test_question_points_l1800_180007

theorem test_question_points : 
  ∀ (other_point_value : ℕ),
    (40 : ℕ) = 10 + (100 - 10 * 4) / other_point_value →
    other_point_value = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_test_question_points_l1800_180007


namespace NUMINAMATH_CALUDE_x_squared_mod_20_l1800_180033

theorem x_squared_mod_20 (x : ℤ) 
  (h1 : 4 * x ≡ 8 [ZMOD 20]) 
  (h2 : 3 * x ≡ 16 [ZMOD 20]) : 
  x^2 ≡ 4 [ZMOD 20] := by
sorry

end NUMINAMATH_CALUDE_x_squared_mod_20_l1800_180033


namespace NUMINAMATH_CALUDE_pencils_per_row_l1800_180022

/-- Given 12 pencils distributed equally among 3 rows, prove that there are 4 pencils in each row. -/
theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) 
  (h1 : total_pencils = 12) 
  (h2 : num_rows = 3) 
  (h3 : total_pencils = num_rows * pencils_per_row) : 
  pencils_per_row = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l1800_180022


namespace NUMINAMATH_CALUDE_bryan_total_books_l1800_180055

/-- The number of bookshelves Bryan has -/
def num_bookshelves : ℕ := 9

/-- The number of books in each bookshelf -/
def books_per_shelf : ℕ := 56

/-- The total number of books Bryan has -/
def total_books : ℕ := num_bookshelves * books_per_shelf

/-- Theorem stating that the total number of books Bryan has is 504 -/
theorem bryan_total_books : total_books = 504 := by sorry

end NUMINAMATH_CALUDE_bryan_total_books_l1800_180055


namespace NUMINAMATH_CALUDE_central_angle_unchanged_l1800_180053

/-- Theorem: When a circle's radius is doubled and the arc length is doubled, the central angle of the sector remains unchanged. -/
theorem central_angle_unchanged 
  (r : ℝ) 
  (l : ℝ) 
  (h_positive_r : r > 0) 
  (h_positive_l : l > 0) : 
  (l / r) = ((2 * l) / (2 * r)) := by 
sorry

end NUMINAMATH_CALUDE_central_angle_unchanged_l1800_180053


namespace NUMINAMATH_CALUDE_incorrect_number_calculation_l1800_180044

theorem incorrect_number_calculation (n : ℕ) (correct_num incorrect_num : ℝ) 
  (incorrect_avg correct_avg : ℝ) :
  n = 10 ∧ 
  correct_num = 75 ∧
  n * incorrect_avg = n * correct_avg - (correct_num - incorrect_num) →
  incorrect_num = 25 :=
by sorry

end NUMINAMATH_CALUDE_incorrect_number_calculation_l1800_180044


namespace NUMINAMATH_CALUDE_infinite_binary_decimal_divisible_by_2019_l1800_180001

/-- A number composed only of 0 and 1 in decimal form -/
def BinaryDecimal (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The set of numbers composed only of 0 and 1 in decimal form that are divisible by 2019 -/
def BinaryDecimalDivisibleBy2019 : Set ℕ :=
  {n : ℕ | BinaryDecimal n ∧ 2019 ∣ n}

/-- The set of numbers composed only of 0 and 1 in decimal form that are divisible by 2019 is infinite -/
theorem infinite_binary_decimal_divisible_by_2019 :
    Set.Infinite BinaryDecimalDivisibleBy2019 :=
  sorry

end NUMINAMATH_CALUDE_infinite_binary_decimal_divisible_by_2019_l1800_180001


namespace NUMINAMATH_CALUDE_smaller_circle_circumference_l1800_180042

theorem smaller_circle_circumference :
  ∀ (r R s d : ℝ),
  s^2 = 784 →
  s = 2 * R →
  d = r + R →
  R = (7/3) * r →
  2 * π * r = 12 * π :=
by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_circumference_l1800_180042


namespace NUMINAMATH_CALUDE_eliminate_x_from_system_l1800_180086

theorem eliminate_x_from_system : ∀ x y : ℝ,
  (2 * x - 3 * y = 11 ∧ 2 * x + 5 * y = -5) →
  -8 * y = 16 := by
  sorry

end NUMINAMATH_CALUDE_eliminate_x_from_system_l1800_180086


namespace NUMINAMATH_CALUDE_multiplication_after_division_l1800_180050

theorem multiplication_after_division (x y : ℝ) : x = 6 → (x / 6) * y = 12 → y = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_after_division_l1800_180050


namespace NUMINAMATH_CALUDE_x_satisfies_quadratic_l1800_180010

theorem x_satisfies_quadratic (x y : ℝ) 
  (h1 : x^2 - y = 10) 
  (h2 : x + y = 14) : 
  x^2 + x - 24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_satisfies_quadratic_l1800_180010


namespace NUMINAMATH_CALUDE_paul_needs_21_cans_l1800_180098

/-- Represents the amount of frosting needed for different baked goods -/
structure FrostingNeeds where
  layerCake : ℕ  -- number of layer cakes
  cupcakesDozens : ℕ  -- number of dozens of cupcakes
  singleCakes : ℕ  -- number of single cakes
  browniePans : ℕ  -- number of brownie pans

/-- Calculates the total number of cans of frosting needed -/
def totalFrostingCans (needs : FrostingNeeds) : ℕ :=
  needs.layerCake + (needs.cupcakesDozens + needs.singleCakes + needs.browniePans) / 2

/-- Paul's specific frosting needs for Saturday -/
def paulsFrostingNeeds : FrostingNeeds :=
  { layerCake := 3
  , cupcakesDozens := 6
  , singleCakes := 12
  , browniePans := 18 }

/-- Theorem stating that Paul needs 21 cans of frosting -/
theorem paul_needs_21_cans : totalFrostingCans paulsFrostingNeeds = 21 := by
  sorry

end NUMINAMATH_CALUDE_paul_needs_21_cans_l1800_180098


namespace NUMINAMATH_CALUDE_triangle_properties_l1800_180040

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (6, 4)
def C : ℝ × ℝ := (4, 0)

-- Define the perpendicular bisector equation
def perpendicular_bisector (x y : ℝ) : Prop :=
  2 * x - y - 3 = 0

-- Define the circumcircle equation
def circumcircle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 3)^2 = 10

-- Theorem statement
theorem triangle_properties :
  (∀ x y : ℝ, perpendicular_bisector x y ↔ 
    (x - (A.1 + C.1) / 2)^2 + (y - (A.2 + C.2) / 2)^2 = 
    ((A.1 - C.1)^2 + (A.2 - C.2)^2) / 4) ∧
  (∀ x y : ℝ, circumcircle x y ↔ 
    (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
    (x - B.1)^2 + (y - B.2)^2 = (x - C.1)^2 + (y - C.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1800_180040


namespace NUMINAMATH_CALUDE_fraction_property_l1800_180052

theorem fraction_property (a : ℕ) (h : a > 1) :
  let b := 2 * a - 1
  0 < a ∧ a < b ∧ (a - 1) / (b - 1) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_property_l1800_180052


namespace NUMINAMATH_CALUDE_science_problem_time_l1800_180021

/-- Calculates the time taken for each science problem given the number of problems and time constraints. -/
theorem science_problem_time 
  (math_problems : ℕ) 
  (social_studies_problems : ℕ) 
  (science_problems : ℕ) 
  (math_time_per_problem : ℚ) 
  (social_studies_time_per_problem : ℚ) 
  (total_time : ℚ) 
  (h1 : math_problems = 15)
  (h2 : social_studies_problems = 6)
  (h3 : science_problems = 10)
  (h4 : math_time_per_problem = 2)
  (h5 : social_studies_time_per_problem = 1/2)
  (h6 : total_time = 48) :
  (total_time - (math_problems * math_time_per_problem + social_studies_problems * social_studies_time_per_problem)) / science_problems = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_science_problem_time_l1800_180021


namespace NUMINAMATH_CALUDE_count_blocks_with_three_differences_l1800_180066

-- Define the properties of a block
structure BlockProperty where
  material : Fin 2
  size : Fin 2
  color : Fin 4
  shape : Fin 4
  pattern : Fin 2

-- Define the set of all possible blocks
def AllBlocks : Finset BlockProperty := sorry

-- Define a function to count the differences between two blocks
def countDifferences (b1 b2 : BlockProperty) : Nat := sorry

-- Define the reference block (plastic large red circle striped)
def referenceBlock : BlockProperty := sorry

-- Theorem statement
theorem count_blocks_with_three_differences :
  (AllBlocks.filter (fun b => countDifferences b referenceBlock = 3)).card = 21 := by sorry

end NUMINAMATH_CALUDE_count_blocks_with_three_differences_l1800_180066


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1800_180071

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x ≥ 1 → x^2 - 1 < 0) ↔ (∃ x : ℝ, x ≥ 1 ∧ x^2 - 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1800_180071


namespace NUMINAMATH_CALUDE_partial_fraction_product_l1800_180073

/-- Given a rational function and its partial fraction decomposition, prove that the product of the numerator coefficients is zero. -/
theorem partial_fraction_product (x : ℝ) (A B C : ℝ) : 
  (x^2 - 25) / (x^3 - x^2 - 7*x + 15) = A / (x - 3) + B / (x + 3) + C / (x - 5) →
  A * B * C = 0 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_product_l1800_180073


namespace NUMINAMATH_CALUDE_carls_marbles_l1800_180032

theorem carls_marbles (initial_marbles : ℕ) : 
  (initial_marbles / 2 + 10 + 25 = 41) → initial_marbles = 12 := by
  sorry

end NUMINAMATH_CALUDE_carls_marbles_l1800_180032


namespace NUMINAMATH_CALUDE_unique_residue_mod_11_l1800_180012

theorem unique_residue_mod_11 :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [ZMOD 11] := by
  sorry

end NUMINAMATH_CALUDE_unique_residue_mod_11_l1800_180012


namespace NUMINAMATH_CALUDE_monic_polynomial_theorem_l1800_180016

def is_monic_degree_7 (p : ℝ → ℝ) : Prop :=
  ∃ a b c d e f g : ℝ, ∀ x, p x = x^7 + a*x^6 + b*x^5 + c*x^4 + d*x^3 + e*x^2 + f*x + g

def satisfies_conditions (p : ℝ → ℝ) : Prop :=
  p 1 = 1 ∧ p 2 = 2 ∧ p 3 = 3 ∧ p 4 = 4 ∧ p 5 = 5 ∧ p 6 = 6 ∧ p 7 = 7

theorem monic_polynomial_theorem (p : ℝ → ℝ) 
  (h1 : is_monic_degree_7 p) 
  (h2 : satisfies_conditions p) : 
  p 8 = 5048 := by
  sorry

end NUMINAMATH_CALUDE_monic_polynomial_theorem_l1800_180016


namespace NUMINAMATH_CALUDE_number_of_divisors_of_36_l1800_180019

theorem number_of_divisors_of_36 : Nat.card {d : ℕ | d > 0 ∧ 36 % d = 0} = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_36_l1800_180019


namespace NUMINAMATH_CALUDE_center_is_midpoint_distance_between_foci_l1800_180028

/-- The equation of an ellipse with foci at (6, -3) and (-4, 5) -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 6)^2 + (y + 3)^2) + Real.sqrt ((x + 4)^2 + (y - 5)^2) = 24

/-- The center of the ellipse -/
def center : ℝ × ℝ := (1, 1)

/-- The first focus of the ellipse -/
def focus1 : ℝ × ℝ := (6, -3)

/-- The second focus of the ellipse -/
def focus2 : ℝ × ℝ := (-4, 5)

/-- The center is the midpoint of the foci -/
theorem center_is_midpoint : center = ((focus1.1 + focus2.1) / 2, (focus1.2 + focus2.2) / 2) := by sorry

/-- The distance between the foci of the ellipse is 2√41 -/
theorem distance_between_foci : 
  Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2) = 2 * Real.sqrt 41 := by sorry

end NUMINAMATH_CALUDE_center_is_midpoint_distance_between_foci_l1800_180028


namespace NUMINAMATH_CALUDE_midpoint_between_fractions_l1800_180046

theorem midpoint_between_fractions :
  (1 / 12 + 1 / 15) / 2 = 3 / 40 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_between_fractions_l1800_180046


namespace NUMINAMATH_CALUDE_divisible_by_4_or_5_count_l1800_180056

def count_divisible (n : ℕ) : ℕ :=
  (n / 4) + (n / 5) - (n / 20)

theorem divisible_by_4_or_5_count :
  count_divisible 60 = 24 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_4_or_5_count_l1800_180056


namespace NUMINAMATH_CALUDE_intersection_points_are_two_and_eight_l1800_180011

/-- The set of k values for which |z - 4| = 3|z + 4| intersects |z| = k at exactly one point -/
def intersection_points : Set ℝ :=
  {k : ℝ | ∃! z : ℂ, Complex.abs (z - 4) = 3 * Complex.abs (z + 4) ∧ Complex.abs z = k}

/-- Theorem stating that the intersection_points set contains only 2 and 8 -/
theorem intersection_points_are_two_and_eight :
  intersection_points = {2, 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_are_two_and_eight_l1800_180011


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l1800_180099

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 8 * x + c = 0) →
  a + 2 * c = 14 →
  a < c →
  (a = (7 - Real.sqrt 17) / 2 ∧ c = (7 + Real.sqrt 17) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l1800_180099


namespace NUMINAMATH_CALUDE_lines_2_3_parallel_l1800_180067

-- Define the slopes of the lines
def slope1 : ℚ := 3 / 4
def slope2 : ℚ := -3 / 4
def slope3 : ℚ := -3 / 4
def slope4 : ℚ := -4 / 3

-- Define the equations of the lines
def line1 (x y : ℚ) : Prop := 4 * y - 3 * x = 16
def line2 (x y : ℚ) : Prop := -3 * x - 4 * y = 15
def line3 (x y : ℚ) : Prop := 4 * y + 3 * x = 16
def line4 (x y : ℚ) : Prop := 3 * y + 4 * x = 15

-- Theorem: Lines 2 and 3 are parallel
theorem lines_2_3_parallel : 
  ∀ (x1 y1 x2 y2 : ℚ), 
    line2 x1 y1 → line3 x2 y2 → 
    slope2 = slope3 ∧ slope2 ≠ slope1 ∧ slope2 ≠ slope4 := by
  sorry

end NUMINAMATH_CALUDE_lines_2_3_parallel_l1800_180067


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1800_180047

theorem diophantine_equation_solution : 
  {(x, y) : ℕ+ × ℕ+ | x.val - y.val - (x.val / y.val) - (x.val^3 / y.val^3) + (x.val^4 / y.val^4) = 2017} = 
  {(⟨2949, by norm_num⟩, ⟨983, by norm_num⟩), (⟨4022, by norm_num⟩, ⟨2011, by norm_num⟩)} :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1800_180047


namespace NUMINAMATH_CALUDE_program_output_is_one_l1800_180063

/-- Represents the state of the program -/
structure ProgramState :=
  (S : ℕ)
  (n : ℕ)

/-- The update function for the program state -/
def updateState (state : ProgramState) : ProgramState :=
  if state.n > 1 then
    { S := state.S + state.n, n := state.n - 1 }
  else
    state

/-- The termination condition for the program -/
def isTerminated (state : ProgramState) : Prop :=
  state.S ≥ 17 ∧ state.n ≤ 1

/-- The initial state of the program -/
def initialState : ProgramState :=
  { S := 0, n := 5 }

/-- The theorem stating that the program terminates with n = 1 -/
theorem program_output_is_one :
  ∃ (finalState : ProgramState), 
    (∃ (k : ℕ), finalState = (updateState^[k] initialState)) ∧
    isTerminated finalState ∧
    finalState.n = 1 := by
  sorry

end NUMINAMATH_CALUDE_program_output_is_one_l1800_180063


namespace NUMINAMATH_CALUDE_max_min_product_l1800_180027

theorem max_min_product (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 12) (h_prod_sum : a * b + b * c + c * a = 30) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 2 ∧ 
  ∀ (m' : ℝ), m' = min (a * b) (min (b * c) (c * a)) → m' ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_min_product_l1800_180027


namespace NUMINAMATH_CALUDE_exactly_three_non_congruent_triangles_l1800_180091

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if two triangles are congruent -/
def congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of all triangles with perimeter 11 -/
def triangles_with_perimeter_11 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 11}

/-- The theorem to be proved -/
theorem exactly_three_non_congruent_triangles :
  ∃ (t1 t2 t3 : IntTriangle),
    t1 ∈ triangles_with_perimeter_11 ∧
    t2 ∈ triangles_with_perimeter_11 ∧
    t3 ∈ triangles_with_perimeter_11 ∧
    ¬congruent t1 t2 ∧ ¬congruent t1 t3 ∧ ¬congruent t2 t3 ∧
    ∀ (t : IntTriangle),
      t ∈ triangles_with_perimeter_11 →
      (congruent t t1 ∨ congruent t t2 ∨ congruent t t3) :=
by sorry

end NUMINAMATH_CALUDE_exactly_three_non_congruent_triangles_l1800_180091


namespace NUMINAMATH_CALUDE_souvenir_problem_l1800_180096

/-- Represents the cost and quantity of souvenirs --/
structure SouvenirPlan where
  costA : ℕ
  costB : ℕ
  quantityA : ℕ
  quantityB : ℕ

/-- Checks if a souvenir plan satisfies all conditions --/
def isValidPlan (plan : SouvenirPlan) : Prop :=
  plan.costA + 20 = plan.costB ∧
  9 * plan.costA = 7 * plan.costB ∧
  plan.quantityB = 2 * plan.quantityA + 5 ∧
  plan.quantityA ≥ 18 ∧
  plan.costA * plan.quantityA + plan.costB * plan.quantityB ≤ 5450

/-- The correct costs and possible purchasing plans --/
def correctSolution : Prop :=
  ∃ (plan : SouvenirPlan),
    isValidPlan plan ∧
    plan.costA = 70 ∧
    plan.costB = 90 ∧
    (plan.quantityA = 18 ∧ plan.quantityB = 41) ∨
    (plan.quantityA = 19 ∧ plan.quantityB = 43) ∨
    (plan.quantityA = 20 ∧ plan.quantityB = 45)

theorem souvenir_problem : correctSolution := by
  sorry

end NUMINAMATH_CALUDE_souvenir_problem_l1800_180096


namespace NUMINAMATH_CALUDE_eraser_price_correct_l1800_180036

/-- The price of an eraser given the following conditions:
  1. 3 erasers and 5 pencils cost 10.6 yuan
  2. 4 erasers and 4 pencils cost 12 yuan -/
def eraser_price : ℝ := 2.2

/-- The price of a pencil (to be determined) -/
def pencil_price : ℝ := sorry

theorem eraser_price_correct :
  3 * eraser_price + 5 * pencil_price = 10.6 ∧
  4 * eraser_price + 4 * pencil_price = 12 :=
by sorry

end NUMINAMATH_CALUDE_eraser_price_correct_l1800_180036


namespace NUMINAMATH_CALUDE_min_socks_for_twelve_pairs_l1800_180015

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (black : ℕ)
  (yellow : ℕ)

/-- Represents the problem setup -/
def initialDrawer : SockDrawer :=
  { red := 120
  , green := 90
  , blue := 70
  , black := 50
  , yellow := 30 }

/-- The number of pairs we want to guarantee -/
def requiredPairs : ℕ := 12

/-- Function to calculate the minimum number of socks needed to guarantee the required pairs -/
def minSocksForPairs (drawer : SockDrawer) (pairs : ℕ) : ℕ :=
  sorry

/-- Theorem stating that 28 socks are needed to guarantee 12 pairs -/
theorem min_socks_for_twelve_pairs :
  minSocksForPairs initialDrawer requiredPairs = 28 :=
sorry

end NUMINAMATH_CALUDE_min_socks_for_twelve_pairs_l1800_180015


namespace NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l1800_180025

/-- The volume of a cylinder minus two cones -/
theorem cylinder_minus_cones_volume (r h : ℝ) (hr : r = 15) (hh : h = 30) :
  π * r^2 * h - 2 * (1/3 * π * r^2 * (h/2)) = 4500 * π := by
  sorry

#check cylinder_minus_cones_volume

end NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l1800_180025


namespace NUMINAMATH_CALUDE_largest_inclination_angle_l1800_180004

-- Define the inclination angle function
noncomputable def inclinationAngle (m : ℝ) : ℝ := Real.arctan m

-- Define the lines
def line1 (x : ℝ) : ℝ := -x + 1
def line2 (x : ℝ) : ℝ := x + 1
def line3 (x : ℝ) : ℝ := 2*x + 1
def line4 : ℝ → Prop := λ x => x = 1

-- Theorem statement
theorem largest_inclination_angle :
  ∀ (θ1 θ2 θ3 θ4 : ℝ),
    θ1 = inclinationAngle (-1) →
    θ2 = inclinationAngle 1 →
    θ3 = inclinationAngle 2 →
    θ4 = Real.pi / 2 →
    θ1 > θ2 ∧ θ1 > θ3 ∧ θ1 > θ4 :=
sorry

end NUMINAMATH_CALUDE_largest_inclination_angle_l1800_180004


namespace NUMINAMATH_CALUDE_extended_morse_code_symbols_l1800_180009

-- Define the function to calculate the number of sequences for a given length
def sequencesForLength (n : ℕ) : ℕ := 2^n

-- Define the total number of sequences for lengths 1 to 5
def totalSequences : ℕ :=
  (sequencesForLength 1) + (sequencesForLength 2) + (sequencesForLength 3) +
  (sequencesForLength 4) + (sequencesForLength 5)

-- Theorem statement
theorem extended_morse_code_symbols :
  totalSequences = 62 := by
  sorry

end NUMINAMATH_CALUDE_extended_morse_code_symbols_l1800_180009


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l1800_180072

-- Statement 1
theorem inequality_one (a b : ℝ) : a^2 + b^2 ≥ a*b + a + b - 1 := by sorry

-- Statement 2
theorem inequality_two {a b : ℝ} (ha : a > 0) (hb : b > 0) : 
  Real.sqrt ((a^2 + b^2) / 2) ≥ (a + b) / 2 := by sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l1800_180072


namespace NUMINAMATH_CALUDE_negative_expressions_count_l1800_180026

theorem negative_expressions_count : 
  let expressions := [-(-5), -|(-5)|, -(5^2), (-5)^2, 1/(-5)]
  (expressions.filter (λ x => x < 0)).length = 3 := by
sorry

end NUMINAMATH_CALUDE_negative_expressions_count_l1800_180026


namespace NUMINAMATH_CALUDE_electrocardiogram_is_line_chart_l1800_180041

/-- Represents different types of charts --/
inductive ChartType
  | BarChart
  | LineChart
  | PieChart

/-- Represents a chart that can display data --/
structure Chart where
  type : ChartType
  representsChangesOverTime : Bool

/-- Defines an electrocardiogram as a chart --/
def Electrocardiogram : Chart :=
  { type := ChartType.LineChart,
    representsChangesOverTime := true }

/-- Theorem stating that an electrocardiogram is a line chart --/
theorem electrocardiogram_is_line_chart : 
  Electrocardiogram.type = ChartType.LineChart :=
by
  sorry


end NUMINAMATH_CALUDE_electrocardiogram_is_line_chart_l1800_180041


namespace NUMINAMATH_CALUDE_second_discount_percentage_l1800_180039

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  original_price = 400 →
  first_discount = 10 →
  final_price = 331.2 →
  ∃ (second_discount : ℝ),
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    second_discount = 8 :=
by sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l1800_180039


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l1800_180074

theorem nested_fraction_equality : (1 / (1 + 1 / (4 + 1 / 5))) = 21 / 26 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l1800_180074


namespace NUMINAMATH_CALUDE_front_view_of_given_stack_map_l1800_180051

/-- Represents a column of stacked cubes -/
def Column := List Nat

/-- Represents a stack map as a list of columns -/
def StackMap := List Column

/-- Calculates the maximum height of a column -/
def maxHeight (column : Column) : Nat :=
  column.foldl max 0

/-- Calculates the front view heights of a stack map -/
def frontView (stackMap : StackMap) : List Nat :=
  stackMap.map maxHeight

/-- The given stack map from the problem -/
def givenStackMap : StackMap :=
  [[1, 3], [2, 4, 2], [3, 5], [2]]

theorem front_view_of_given_stack_map :
  frontView givenStackMap = [3, 4, 5, 2] := by
  sorry

end NUMINAMATH_CALUDE_front_view_of_given_stack_map_l1800_180051


namespace NUMINAMATH_CALUDE_symmetric_circle_l1800_180093

/-- Given a circle C with equation x^2 + y^2 = 25 and a point of symmetry (-3, 4),
    the symmetric circle has the equation (x + 6)^2 + (y - 8)^2 = 25 -/
theorem symmetric_circle (x y : ℝ) :
  (∀ x y, x^2 + y^2 = 25 → (x + 6)^2 + (y - 8)^2 = 25) ∧
  (∃ x₀ y₀, x₀^2 + y₀^2 = 25 ∧ 
    2 * (-3) = x₀ + (x₀ - 6) ∧
    2 * 4 = y₀ + (y₀ - (-8))) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_l1800_180093


namespace NUMINAMATH_CALUDE_system_solution_l1800_180065

theorem system_solution :
  ∀ x y : ℝ, x^2 + y^2 = 13 ∧ x * y = 6 → x = 3 ∧ y = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1800_180065


namespace NUMINAMATH_CALUDE_amusement_park_theorem_l1800_180089

/-- Represents the amusement park scenario with two roller coasters and a group of friends. -/
structure AmusementPark where
  friends : ℕ
  first_coaster_cost : ℕ
  second_coaster_cost : ℕ
  first_coaster_rides : ℕ
  second_coaster_rides : ℕ
  discount_rate : ℚ
  discount_threshold : ℕ

/-- Calculates the total number of tickets needed for the group. -/
def total_tickets (park : AmusementPark) : ℕ :=
  park.friends * (park.first_coaster_cost * park.first_coaster_rides + 
                  park.second_coaster_cost * park.second_coaster_rides)

/-- Calculates the cost difference between non-discounted and discounted tickets. -/
def cost_difference (park : AmusementPark) : ℚ :=
  let total_cost := total_tickets park
  if total_tickets park ≥ park.discount_threshold then
    (total_cost : ℚ) * park.discount_rate
  else
    0

/-- Theorem stating the correct number of tickets and cost difference for the given scenario. -/
theorem amusement_park_theorem (park : AmusementPark) 
  (h1 : park.friends = 8)
  (h2 : park.first_coaster_cost = 6)
  (h3 : park.second_coaster_cost = 8)
  (h4 : park.first_coaster_rides = 2)
  (h5 : park.second_coaster_rides = 1)
  (h6 : park.discount_rate = 15 / 100)
  (h7 : park.discount_threshold = 10) :
  total_tickets park = 160 ∧ cost_difference park = 24 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_theorem_l1800_180089


namespace NUMINAMATH_CALUDE_book_pages_l1800_180087

/-- Calculates the total number of pages in a book given reading rate and time spent reading. -/
def total_pages (pages_per_hour : ℝ) (monday_hours : ℝ) (tuesday_hours : ℝ) (remaining_hours : ℝ) : ℝ :=
  pages_per_hour * (monday_hours + tuesday_hours + remaining_hours)

/-- Theorem stating that the book has 248 pages given Joanna's reading rate and time spent. -/
theorem book_pages : 
  let pages_per_hour : ℝ := 16
  let monday_hours : ℝ := 3
  let tuesday_hours : ℝ := 6.5
  let remaining_hours : ℝ := 6
  total_pages pages_per_hour monday_hours tuesday_hours remaining_hours = 248 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_l1800_180087


namespace NUMINAMATH_CALUDE_equation_solution_l1800_180079

theorem equation_solution (m n : ℝ) : 
  (∀ x : ℝ, (2*x - 5)*(x + m) = 2*x^2 - 3*x + n) → 
  (m = 1 ∧ n = -5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1800_180079


namespace NUMINAMATH_CALUDE_complex_operations_l1800_180054

theorem complex_operations (z₁ z₂ : ℂ) 
  (h₁ : z₁ = 2 - 3 * Complex.I) 
  (h₂ : z₂ = (15 - 5 * Complex.I) / (2 + Complex.I)^2) : 
  z₁ * z₂ = -7 - 9 * Complex.I ∧ 
  z₁ / z₂ = 11/10 + 3/10 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_operations_l1800_180054


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_l1800_180023

theorem infinitely_many_pairs : 
  Set.Infinite {p : ℕ × ℕ | 2019 < (2 : ℝ)^p.1 / (3 : ℝ)^p.2 ∧ (2 : ℝ)^p.1 / (3 : ℝ)^p.2 < 2020} :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_l1800_180023


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1800_180070

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |x + 2| = 3*x - 6 ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1800_180070


namespace NUMINAMATH_CALUDE_lilia_initial_peaches_l1800_180082

/-- The number of peaches Lilia sold to friends -/
def peaches_sold_to_friends : ℕ := 10

/-- The price of each peach sold to friends -/
def price_for_friends : ℚ := 2

/-- The number of peaches Lilia sold to relatives -/
def peaches_sold_to_relatives : ℕ := 4

/-- The price of each peach sold to relatives -/
def price_for_relatives : ℚ := 5/4

/-- The number of peaches Lilia kept for herself -/
def peaches_kept : ℕ := 1

/-- The total amount of money Lilia earned -/
def total_earned : ℚ := 25

/-- The initial number of peaches Lilia had -/
def initial_peaches : ℕ := peaches_sold_to_friends + peaches_sold_to_relatives + peaches_kept

theorem lilia_initial_peaches :
  initial_peaches = 15 ∧
  total_earned = peaches_sold_to_friends * price_for_friends + peaches_sold_to_relatives * price_for_relatives :=
by sorry

end NUMINAMATH_CALUDE_lilia_initial_peaches_l1800_180082


namespace NUMINAMATH_CALUDE_afternoon_to_morning_ratio_is_two_to_one_l1800_180083

/-- Represents the sales of pears by a salesman in a day -/
structure PearSales where
  total : ℕ
  morning : ℕ
  afternoon : ℕ

/-- Theorem stating that the ratio of afternoon to morning pear sales is 2:1 -/
theorem afternoon_to_morning_ratio_is_two_to_one (sales : PearSales)
  (h_total : sales.total = 360)
  (h_morning : sales.morning = 120)
  (h_afternoon : sales.afternoon = 240) :
  sales.afternoon / sales.morning = 2 := by
  sorry

#check afternoon_to_morning_ratio_is_two_to_one

end NUMINAMATH_CALUDE_afternoon_to_morning_ratio_is_two_to_one_l1800_180083


namespace NUMINAMATH_CALUDE_function_minimum_implies_parameter_range_l1800_180029

/-- Given a function f(x) with parameter a > 0, if its minimum value is ln²(a) + 3ln(a) + 2,
    then a ≥ e^(-3/2) -/
theorem function_minimum_implies_parameter_range (a : ℝ) (h_a_pos : a > 0) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = a^2 * Real.exp (-2*x) + a * (2*x + 1) * Real.exp (-x) + x^2 + x) ∧
    (∀ x, f x ≥ Real.log a ^ 2 + 3 * Real.log a + 2) ∧
    (∃ x₀, f x₀ = Real.log a ^ 2 + 3 * Real.log a + 2)) →
  a ≥ Real.exp (-3/2) := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_implies_parameter_range_l1800_180029


namespace NUMINAMATH_CALUDE_prob_even_diagonals_eq_one_over_101_l1800_180014

/-- Represents a 3x3 grid filled with numbers 1 to 9 --/
def Grid := Fin 9 → Fin 9

/-- Checks if a given grid has even sums on both diagonals --/
def has_even_diagonal_sums (g : Grid) : Prop :=
  (g 0 + g 4 + g 8) % 2 = 0 ∧ (g 2 + g 4 + g 6) % 2 = 0

/-- The set of all valid grids --/
def all_grids : Finset Grid :=
  sorry

/-- The set of grids with even diagonal sums --/
def even_sum_grids : Finset Grid :=
  sorry

/-- The probability of having even sums on both diagonals --/
def prob_even_diagonals : ℚ :=
  (Finset.card even_sum_grids : ℚ) / (Finset.card all_grids : ℚ)

theorem prob_even_diagonals_eq_one_over_101 : 
  prob_even_diagonals = 1 / 101 :=
sorry

end NUMINAMATH_CALUDE_prob_even_diagonals_eq_one_over_101_l1800_180014


namespace NUMINAMATH_CALUDE_school_pens_problem_l1800_180080

theorem school_pens_problem (pencils : ℕ) (pen_cost pencil_cost total_cost : ℚ) :
  pencils = 38 →
  pencil_cost = 5/2 →
  pen_cost = 7/2 →
  total_cost = 291 →
  ∃ (pens : ℕ), pens * pen_cost + pencils * pencil_cost = total_cost ∧ pens = 56 := by
  sorry

end NUMINAMATH_CALUDE_school_pens_problem_l1800_180080


namespace NUMINAMATH_CALUDE_additive_fun_properties_l1800_180068

/-- A function satisfying f(x+y) = f(x) + f(y) for all x, y ∈ ℝ -/
def AdditiveFun (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

theorem additive_fun_properties
  (f : ℝ → ℝ)
  (h_additive : AdditiveFun f)
  (h_increasing : Monotone f)
  (h_f1 : f 1 = 1)
  (h_f2a : ∀ a : ℝ, f (2 * a) > f (a - 1) + 2) :
  (f 0 = 0) ∧
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ a : ℝ, a > 1) :=
by sorry

end NUMINAMATH_CALUDE_additive_fun_properties_l1800_180068


namespace NUMINAMATH_CALUDE_triangle_property_l1800_180037

theorem triangle_property (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  A < π/2 ∧ B < π/2 ∧ C < π/2 →
  S > 0 →
  S = (1/2) * b * c * Real.sin A →
  (b - c) * Real.sin B = b * Real.sin (A - C) →
  A = π/3 ∧ 
  4 * Real.sqrt 3 ≤ (a^2 + b^2 + c^2) / S ∧ 
  (a^2 + b^2 + c^2) / S < 16 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_property_l1800_180037


namespace NUMINAMATH_CALUDE_remainder_sum_l1800_180002

theorem remainder_sum (n : ℤ) (h : n % 12 = 5) : (n % 3 + n % 4 = 3) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l1800_180002


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1800_180018

theorem absolute_value_equation_solution (x y : ℝ) :
  |x - Real.log (y^2)| = x + Real.log (y^2) →
  x = 0 ∧ (y = 1 ∨ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1800_180018


namespace NUMINAMATH_CALUDE_equation_holds_iff_base_ten_l1800_180013

/-- Represents a digit in base k --/
def Digit (k : ℕ) := Fin k

/-- Converts a natural number to its representation in base k --/
def toBaseK (n : ℕ) (k : ℕ) : List (Digit k) :=
  sorry

/-- Adds two numbers represented in base k --/
def addBaseK (a b : List (Digit k)) : List (Digit k) :=
  sorry

/-- Converts a list of digits in base k to a natural number --/
def fromBaseK (digits : List (Digit k)) (k : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the equation holds iff k = 10 --/
theorem equation_holds_iff_base_ten (k : ℕ) :
  (fromBaseK (addBaseK (toBaseK 5342 k) (toBaseK 6421 k)) k = fromBaseK (toBaseK 14163 k) k) ↔ k = 10 :=
sorry

end NUMINAMATH_CALUDE_equation_holds_iff_base_ten_l1800_180013


namespace NUMINAMATH_CALUDE_halloween_candies_l1800_180069

/-- The total number of candies collected by a group of friends on Halloween. -/
def total_candies (bob : ℕ) (mary : ℕ) (john : ℕ) (sue : ℕ) (sam : ℕ) : ℕ :=
  bob + mary + john + sue + sam

/-- Theorem stating that the total number of candies collected by the friends is 50. -/
theorem halloween_candies : total_candies 10 5 5 20 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candies_l1800_180069


namespace NUMINAMATH_CALUDE_intersection_at_origin_l1800_180038

/-- A line in the coordinate plane --/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The origin point (0, 0) --/
def origin : ℝ × ℝ := (0, 0)

/-- Check if a point lies on a line --/
def pointOnLine (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.slope * p.1 + (l.point.2 - l.slope * l.point.1)

theorem intersection_at_origin 
  (k : Line)
  (l : Line)
  (hk_slope : k.slope = 1/2)
  (hk_origin : pointOnLine k origin)
  (hl_slope : l.slope = -2)
  (hl_point : l.point = (-2, 4)) :
  ∃ (p : ℝ × ℝ), pointOnLine k p ∧ pointOnLine l p ∧ p = origin :=
sorry

#check intersection_at_origin

end NUMINAMATH_CALUDE_intersection_at_origin_l1800_180038


namespace NUMINAMATH_CALUDE_tens_digit_of_nine_power_2010_l1800_180075

def last_two_digits (n : ℕ) : ℕ := n % 100

def cycle_of_nine : List ℕ := [09, 81, 29, 61, 49, 41, 69, 21, 89, 01]

theorem tens_digit_of_nine_power_2010 :
  (last_two_digits (9^2010)) / 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_nine_power_2010_l1800_180075
