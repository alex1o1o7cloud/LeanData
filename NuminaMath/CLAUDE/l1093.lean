import Mathlib

namespace NUMINAMATH_CALUDE_l_shape_area_is_55_l1093_109384

/-- The area of an "L" shape formed by cutting a smaller rectangle from a larger rectangle -/
def l_shape_area (large_width large_height small_width small_height : ℕ) : ℕ :=
  large_width * large_height - small_width * small_height

/-- Theorem: The area of the "L" shape is 55 square units -/
theorem l_shape_area_is_55 :
  l_shape_area 10 7 5 3 = 55 := by
  sorry

#eval l_shape_area 10 7 5 3

end NUMINAMATH_CALUDE_l_shape_area_is_55_l1093_109384


namespace NUMINAMATH_CALUDE_problem_statement_l1093_109398

theorem problem_statement (x y M : ℝ) (h : M / ((x * y + y^2) / (x - y)^2) = (x^2 - y^2) / y) :
  M = (x + y)^2 / (x - y) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1093_109398


namespace NUMINAMATH_CALUDE_bob_gave_terry_24_bushels_l1093_109345

/-- Represents the number of bushels Bob grew -/
def total_bushels : ℕ := 50

/-- Represents the number of ears per bushel -/
def ears_per_bushel : ℕ := 14

/-- Represents the number of ears Bob has left -/
def ears_left : ℕ := 357

/-- Calculates the number of bushels Bob gave to Terry -/
def bushels_given_to_terry : ℕ :=
  ((total_bushels * ears_per_bushel) - ears_left) / ears_per_bushel

theorem bob_gave_terry_24_bushels :
  bushels_given_to_terry = 24 := by
  sorry

end NUMINAMATH_CALUDE_bob_gave_terry_24_bushels_l1093_109345


namespace NUMINAMATH_CALUDE_sum_xy_equals_two_l1093_109365

theorem sum_xy_equals_two (w x y z : ℝ) 
  (eq1 : w + x + y = 3)
  (eq2 : x + y + z = 4)
  (eq3 : w + x + y + z = 5) :
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_xy_equals_two_l1093_109365


namespace NUMINAMATH_CALUDE_milk_packet_cost_l1093_109320

theorem milk_packet_cost (total_packets : Nat) (remaining_packets : Nat) 
  (avg_price_all : ℚ) (avg_price_remaining : ℚ) :
  total_packets = 10 →
  remaining_packets = 7 →
  avg_price_all = 25 →
  avg_price_remaining = 20 →
  (total_packets * avg_price_all - remaining_packets * avg_price_remaining : ℚ) = 110 := by
  sorry

end NUMINAMATH_CALUDE_milk_packet_cost_l1093_109320


namespace NUMINAMATH_CALUDE_system_solution_l1093_109304

theorem system_solution (a b c : ℝ) 
  (eq1 : b + c = 10 - 4*a)
  (eq2 : a + c = -16 - 4*b)
  (eq3 : a + b = 9 - 4*c) :
  2*a + 2*b + 2*c = 1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1093_109304


namespace NUMINAMATH_CALUDE_modular_inverse_of_7_mod_31_l1093_109314

theorem modular_inverse_of_7_mod_31 :
  ∃ x : ℕ, x ≤ 30 ∧ (7 * x) % 31 = 1 :=
by
  use 9
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_7_mod_31_l1093_109314


namespace NUMINAMATH_CALUDE_hyperbola_focus_distance_l1093_109324

/-- The hyperbola equation -/
def is_on_hyperbola (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

/-- The distance from a point to the left focus -/
def dist_to_left_focus (x y : ℝ) : ℝ := sorry

/-- The distance from a point to the right focus -/
def dist_to_right_focus (x y : ℝ) : ℝ := sorry

/-- Theorem: If P(x,y) is on the right branch of the hyperbola and
    its distance to the left focus is 12, then its distance to the right focus is 4 -/
theorem hyperbola_focus_distance (x y : ℝ) :
  is_on_hyperbola x y ∧ x > 0 ∧ dist_to_left_focus x y = 12 →
  dist_to_right_focus x y = 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_distance_l1093_109324


namespace NUMINAMATH_CALUDE_propositions_b_and_c_are_true_l1093_109325

theorem propositions_b_and_c_are_true :
  (∀ a b : ℝ, |a| > |b| → a^2 > b^2) ∧
  (∀ a b c : ℝ, (a - b) * c^2 > 0 → a > b) := by
  sorry

end NUMINAMATH_CALUDE_propositions_b_and_c_are_true_l1093_109325


namespace NUMINAMATH_CALUDE_apples_to_oranges_ratio_l1093_109369

/-- Represents the number of fruits of each type on the display -/
structure FruitDisplay where
  apples : ℕ
  oranges : ℕ
  bananas : ℕ

/-- Defines the conditions of the fruit display -/
def validFruitDisplay (d : FruitDisplay) : Prop :=
  d.oranges = 2 * d.bananas ∧
  d.bananas = 5 ∧
  d.apples + d.oranges + d.bananas = 35

/-- Theorem stating that for a valid fruit display, the ratio of apples to oranges is 2:1 -/
theorem apples_to_oranges_ratio (d : FruitDisplay) (h : validFruitDisplay d) :
  d.apples * 1 = d.oranges * 2 := by
  sorry

end NUMINAMATH_CALUDE_apples_to_oranges_ratio_l1093_109369


namespace NUMINAMATH_CALUDE_complex_equation_solution_pure_imaginary_condition_l1093_109321

-- Problem 1
theorem complex_equation_solution (a b : ℝ) (h : (a + Complex.I) * (1 + Complex.I) = b * Complex.I) :
  a = 1 ∧ b = 2 := by sorry

-- Problem 2
theorem pure_imaginary_condition (m : ℝ) 
  (h : ∃ (k : ℝ), Complex.mk (m^2 + m - 2) (m^2 - 1) = Complex.I * k) :
  m = -2 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_pure_imaginary_condition_l1093_109321


namespace NUMINAMATH_CALUDE_percentage_problem_l1093_109370

/-- Given that P% of 820 is 20 less than 15% of 1500, prove that P = 25 -/
theorem percentage_problem (P : ℝ) (h : P / 100 * 820 = 15 / 100 * 1500 - 20) : P = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1093_109370


namespace NUMINAMATH_CALUDE_quadratic_expression_k_value_l1093_109376

theorem quadratic_expression_k_value :
  ∀ a h k : ℝ, (∀ x : ℝ, x^2 - 8*x = a*(x - h)^2 + k) → k = -16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_k_value_l1093_109376


namespace NUMINAMATH_CALUDE_geometric_sequence_reciprocal_sum_l1093_109336

/-- Given a geometric sequence {a_n} with a₁ = 2 and a₁ + a₃ + a₅ = 14,
    prove that 1/a₁ + 1/a₃ + 1/a₅ = 7/8 -/
theorem geometric_sequence_reciprocal_sum (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 2 →
  a 1 + a 3 + a 5 = 14 →
  1 / a 1 + 1 / a 3 + 1 / a 5 = 7 / 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_reciprocal_sum_l1093_109336


namespace NUMINAMATH_CALUDE_student_arrangement_counts_l1093_109395

/-- The number of ways to arrange 5 male and 2 female students in a row --/
def arrange_students (n_male : ℕ) (n_female : ℕ) : ℕ → ℕ → ℕ
| 1 => λ _ => 1400  -- females must be next to each other
| 2 => λ _ => 3600  -- females must not be next to each other
| 3 => λ _ => 3720  -- specific placement restrictions for females
| _ => λ _ => 0     -- undefined for other cases

/-- Theorem stating the correct number of arrangements for each scenario --/
theorem student_arrangement_counts :
  let n_male := 5
  let n_female := 2
  (arrange_students n_male n_female 1 0 = 1400) ∧
  (arrange_students n_male n_female 2 0 = 3600) ∧
  (arrange_students n_male n_female 3 0 = 3720) :=
by sorry


end NUMINAMATH_CALUDE_student_arrangement_counts_l1093_109395


namespace NUMINAMATH_CALUDE_right_rectangular_prism_x_value_l1093_109355

theorem right_rectangular_prism_x_value 
  (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive for valid logarithms
  (edge1 : ℝ := Real.log x / Real.log 5)
  (edge2 : ℝ := Real.log x / Real.log 6)
  (edge3 : ℝ := Real.log x / Real.log 10)
  (surface_area : ℝ := 2 * (edge1 * edge2 + edge1 * edge3 + edge2 * edge3))
  (volume : ℝ := edge1 * edge2 * edge3)
  (h2 : surface_area = 3 * volume) :
  x = 300^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_x_value_l1093_109355


namespace NUMINAMATH_CALUDE_unique_solution_for_prime_equation_l1093_109308

theorem unique_solution_for_prime_equation :
  ∃! n : ℕ, ∃ p : ℕ, Nat.Prime p ∧ n^2 = p^2 + 3*p + 9 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_prime_equation_l1093_109308


namespace NUMINAMATH_CALUDE_afternoon_rowers_l1093_109387

theorem afternoon_rowers (total : ℕ) (morning : ℕ) (h1 : total = 60) (h2 : morning = 53) :
  total - morning = 7 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_rowers_l1093_109387


namespace NUMINAMATH_CALUDE_usual_time_is_eight_l1093_109302

-- Define the usual speed and time
variable (S : ℝ) -- Usual speed
variable (T : ℝ) -- Usual time

-- Define the theorem
theorem usual_time_is_eight
  (h1 : S > 0) -- Assume speed is positive
  (h2 : T > 0) -- Assume time is positive
  (h3 : S / (0.25 * S) = (T + 24) / T) -- Equation from the problem
  : T = 8 := by
sorry


end NUMINAMATH_CALUDE_usual_time_is_eight_l1093_109302


namespace NUMINAMATH_CALUDE_floor_sum_abcd_l1093_109358

theorem floor_sum_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a^2 + b^2 = 2010) (h2 : c^2 + d^2 = 2010) (h3 : a * c = 1020) (h4 : b * d = 1020) :
  ⌊a + b + c + d⌋ = 126 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_abcd_l1093_109358


namespace NUMINAMATH_CALUDE_range_of_a_l1093_109343

theorem range_of_a (a : ℝ) : 
  (∀ x, x > a → x^2 + x - 2 > 0) ∧ 
  (∃ x, x^2 + x - 2 > 0 ∧ x ≤ a) → 
  a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1093_109343


namespace NUMINAMATH_CALUDE_f_inequality_A_is_solution_set_l1093_109319

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Define set A
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- State the theorem
theorem f_inequality (a b : ℝ) (ha : a ∈ A) (hb : b ∈ A) :
  f (a * b) > f a - f b := by
  sorry

-- Prove that A is indeed the solution set to f(x) < 3 - |2x + 1|
theorem A_is_solution_set (x : ℝ) :
  x ∈ A ↔ f x < 3 - |2 * x + 1| := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_A_is_solution_set_l1093_109319


namespace NUMINAMATH_CALUDE_lucas_income_l1093_109322

/-- Represents the tax structure and Lucas's income --/
structure TaxSystem where
  p : ℝ  -- Base tax rate as a decimal
  income : ℝ  -- Lucas's annual income
  taxPaid : ℝ  -- Total tax paid by Lucas

/-- The tax system satisfies the given conditions --/
def validTaxSystem (ts : TaxSystem) : Prop :=
  ts.taxPaid = (0.01 * ts.p * 35000 + 0.01 * (ts.p + 4) * (ts.income - 35000))
  ∧ ts.taxPaid = 0.01 * (ts.p + 0.5) * ts.income
  ∧ ts.income ≥ 35000

/-- Theorem stating that Lucas's income is $40000 --/
theorem lucas_income (ts : TaxSystem) (h : validTaxSystem ts) : ts.income = 40000 := by
  sorry

end NUMINAMATH_CALUDE_lucas_income_l1093_109322


namespace NUMINAMATH_CALUDE_combined_tax_rate_l1093_109331

/-- Combined tax rate calculation -/
theorem combined_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.4) 
  (h2 : mindy_rate = 0.3) 
  (h3 : income_ratio = 3) :
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.325 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l1093_109331


namespace NUMINAMATH_CALUDE_sum_of_roots_l1093_109386

theorem sum_of_roots (p q : ℝ) : 
  (∀ x, x^2 - p*x + q = 0 ↔ (x = p ∨ x = q)) →
  2*p + 3*q = 6 →
  p + q = p :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1093_109386


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l1093_109348

/-- Given the purchase of blankets with specific quantities and prices, 
    prove that the unknown rate for two blankets is 225 Rs. -/
theorem unknown_blanket_rate : 
  ∀ (unknown_rate : ℕ),
  (3 * 100 + 2 * 150 + 2 * unknown_rate) / 7 = 150 →
  unknown_rate = 225 := by
sorry

end NUMINAMATH_CALUDE_unknown_blanket_rate_l1093_109348


namespace NUMINAMATH_CALUDE_kim_shirts_proof_l1093_109318

def dozen : ℕ := 12

def initial_shirts : ℕ := 4 * dozen

def shirts_given_away : ℕ := initial_shirts / 3

def remaining_shirts : ℕ := initial_shirts - shirts_given_away

theorem kim_shirts_proof : remaining_shirts = 32 := by
  sorry

end NUMINAMATH_CALUDE_kim_shirts_proof_l1093_109318


namespace NUMINAMATH_CALUDE_max_product_constraint_l1093_109344

theorem max_product_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 6 * a + 8 * b = 48) :
  a * b ≤ 12 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 6 * a₀ + 8 * b₀ = 48 ∧ a₀ * b₀ = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l1093_109344


namespace NUMINAMATH_CALUDE_negative_eight_to_four_thirds_equals_sixteen_l1093_109360

theorem negative_eight_to_four_thirds_equals_sixteen :
  (-8 : ℝ) ^ (4/3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_negative_eight_to_four_thirds_equals_sixteen_l1093_109360


namespace NUMINAMATH_CALUDE_not_even_implies_exists_unequal_l1093_109357

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define what it means for f to be not even
def NotEven (f : ℝ → ℝ) : Prop :=
  ¬(∀ x : ℝ, f (-x) = f x)

-- Theorem statement
theorem not_even_implies_exists_unequal (f : ℝ → ℝ) :
  NotEven f → ∃ x₀ : ℝ, f (-x₀) ≠ f x₀ :=
by
  sorry

end NUMINAMATH_CALUDE_not_even_implies_exists_unequal_l1093_109357


namespace NUMINAMATH_CALUDE_vector_inequality_not_always_holds_l1093_109397

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_inequality_not_always_holds :
  ∃ (a b : V), ‖a - b‖ > |‖a‖ - ‖b‖| := by sorry

end NUMINAMATH_CALUDE_vector_inequality_not_always_holds_l1093_109397


namespace NUMINAMATH_CALUDE_max_slope_product_30deg_l1093_109392

/-- The maximum product of slopes for two lines intersecting at 30° with one slope four times the other -/
theorem max_slope_product_30deg (m₁ m₂ : ℝ) : 
  m₁ ≠ 0 → m₂ ≠ 0 →  -- nonhorizontal and nonvertical lines
  m₂ = 4 * m₁ →  -- one slope is 4 times the other
  |((m₂ - m₁) / (1 + m₁ * m₂))| = 1 / Real.sqrt 3 →  -- 30° angle between lines
  m₁ * m₂ ≤ (3 * Real.sqrt 3 + Real.sqrt 11)^2 / 16 :=
by sorry

end NUMINAMATH_CALUDE_max_slope_product_30deg_l1093_109392


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l1093_109383

/-- Given a cone with a 45° angle between the generatrix and base, and height 1,
    the length of the generatrix is √2. -/
theorem cone_generatrix_length 
  (angle : ℝ) 
  (height : ℝ) 
  (h_angle : angle = Real.pi / 4) 
  (h_height : height = 1) : 
  Real.sqrt 2 = 
    Real.sqrt (height ^ 2 + height ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_cone_generatrix_length_l1093_109383


namespace NUMINAMATH_CALUDE_origin_outside_circle_l1093_109350

theorem origin_outside_circle (a : ℝ) (h : 0 < a ∧ a < 1) :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 + 2*a*x + 2*y + (a - 1)^2 = 0}
  (0, 0) ∉ circle := by
sorry

end NUMINAMATH_CALUDE_origin_outside_circle_l1093_109350


namespace NUMINAMATH_CALUDE_average_marks_of_combined_classes_l1093_109326

theorem average_marks_of_combined_classes 
  (class1_size : ℕ) (class1_avg : ℝ) 
  (class2_size : ℕ) (class2_avg : ℝ) : 
  let total_students := class1_size + class2_size
  let total_marks := class1_size * class1_avg + class2_size * class2_avg
  total_marks / total_students = (35 * 45 + 55 * 65) / (35 + 55) :=
by
  sorry

#eval (35 * 45 + 55 * 65) / (35 + 55)

end NUMINAMATH_CALUDE_average_marks_of_combined_classes_l1093_109326


namespace NUMINAMATH_CALUDE_three_numbers_sum_l1093_109382

theorem three_numbers_sum (a b c : ℝ) 
  (sum_ab : a + b = 37)
  (sum_bc : b + c = 58)
  (sum_ca : c + a = 72) :
  a + b + c - 10 = 73.5 := by sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l1093_109382


namespace NUMINAMATH_CALUDE_oreo_multiple_l1093_109339

theorem oreo_multiple (total : Nat) (jordan : Nat) (m : Nat) : 
  total = 36 → jordan = 11 → jordan + (jordan * m + 3) = total → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_oreo_multiple_l1093_109339


namespace NUMINAMATH_CALUDE_cola_price_is_three_l1093_109330

/-- Represents the cost and quantity of drinks sold in a store --/
structure DrinkSales where
  cola_price : ℝ
  cola_quantity : ℕ
  juice_price : ℝ
  juice_quantity : ℕ
  water_price : ℝ
  water_quantity : ℕ
  total_earnings : ℝ

/-- Theorem stating that the cola price is $3 given the specific sales conditions --/
theorem cola_price_is_three (sales : DrinkSales)
  (h_juice_price : sales.juice_price = 1.5)
  (h_water_price : sales.water_price = 1)
  (h_cola_quantity : sales.cola_quantity = 15)
  (h_juice_quantity : sales.juice_quantity = 12)
  (h_water_quantity : sales.water_quantity = 25)
  (h_total_earnings : sales.total_earnings = 88) :
  sales.cola_price = 3 := by
  sorry

#check cola_price_is_three

end NUMINAMATH_CALUDE_cola_price_is_three_l1093_109330


namespace NUMINAMATH_CALUDE_concert_ticket_cost_daria_concert_money_l1093_109399

theorem concert_ticket_cost (ticket_price : ℕ) (current_money : ℕ) : ℕ :=
  let total_tickets : ℕ := 4
  let total_cost : ℕ := total_tickets * ticket_price
  let additional_money_needed : ℕ := total_cost - current_money
  additional_money_needed

theorem daria_concert_money : concert_ticket_cost 90 189 = 171 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_cost_daria_concert_money_l1093_109399


namespace NUMINAMATH_CALUDE_f_composition_value_l1093_109385

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else Real.exp (x + 1) - 2

theorem f_composition_value : f (f (1 / Real.exp 1)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l1093_109385


namespace NUMINAMATH_CALUDE_remainder_seven_to_63_mod_8_l1093_109334

theorem remainder_seven_to_63_mod_8 :
  7^63 % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_to_63_mod_8_l1093_109334


namespace NUMINAMATH_CALUDE_max_ratio_squared_l1093_109371

theorem max_ratio_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≥ b) : 
  (∃ ρ : ℝ, ρ > 0 ∧ ρ^2 = 2 ∧ 
    (∀ r : ℝ, r > ρ → 
      ¬∃ x y : ℝ, 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b ∧ 
        a^2 + y^2 = (a - x)^2 + (b - y)^2 ∧ 
        a^2 + y^2 = b^2 - x^2 + y^2 ∧ 
        r = a / b)) := by
  sorry

end NUMINAMATH_CALUDE_max_ratio_squared_l1093_109371


namespace NUMINAMATH_CALUDE_min_cubes_to_remove_l1093_109310

/-- Represents the dimensions of a rectangular block. -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular block given its dimensions. -/
def blockVolume (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the side length of the largest cube that can fit within the block. -/
def largestCubeSide (d : BlockDimensions) : ℕ :=
  min d.length (min d.width d.height)

/-- Calculates the volume of the largest cube that can fit within the block. -/
def largestCubeVolume (d : BlockDimensions) : ℕ :=
  let side := largestCubeSide d
  side * side * side

/-- The main theorem stating the minimum number of cubes to remove. -/
theorem min_cubes_to_remove (d : BlockDimensions) 
    (h1 : d.length = 4) (h2 : d.width = 5) (h3 : d.height = 6) : 
    blockVolume d - largestCubeVolume d = 56 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_to_remove_l1093_109310


namespace NUMINAMATH_CALUDE_max_servings_is_56_l1093_109378

/-- Ingredients required for one serving of salad -/
structure SaladServing where
  cucumbers : ℕ := 2
  tomatoes : ℕ := 2
  brynza : ℕ := 75  -- in grams
  peppers : ℕ := 1

/-- Ingredients available in the warehouse -/
structure Warehouse where
  cucumbers : ℕ := 117
  tomatoes : ℕ := 116
  brynza : ℕ := 4200  -- converted from 4.2 kg to grams
  peppers : ℕ := 60

/-- Calculate the maximum number of servings that can be made -/
def maxServings (w : Warehouse) (s : SaladServing) : ℕ :=
  min (w.cucumbers / s.cucumbers)
      (min (w.tomatoes / s.tomatoes)
           (min (w.brynza / s.brynza)
                (w.peppers / s.peppers)))

/-- Theorem: The maximum number of servings that can be made is 56 -/
theorem max_servings_is_56 (w : Warehouse) (s : SaladServing) :
  maxServings w s = 56 := by
  sorry

end NUMINAMATH_CALUDE_max_servings_is_56_l1093_109378


namespace NUMINAMATH_CALUDE_expression_evaluation_l1093_109354

theorem expression_evaluation : (2023 - 1910 + 5)^2 / 121 = 114 + 70 / 121 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1093_109354


namespace NUMINAMATH_CALUDE_boat_travel_theorem_l1093_109303

/-- Represents the distance traveled by a boat in one hour -/
def boat_distance (boat_speed stream_speed : ℝ) : ℝ := boat_speed - stream_speed

/-- Proves that a boat traveling 11 km along the stream in one hour will travel 7 km against the stream in one hour, given its still water speed is 9 km/hr -/
theorem boat_travel_theorem (boat_speed : ℝ) (h1 : boat_speed = 9) 
  (h2 : boat_speed + (11 - boat_speed) = 11) : 
  boat_distance boat_speed (11 - boat_speed) = 7 := by
  sorry

#check boat_travel_theorem

end NUMINAMATH_CALUDE_boat_travel_theorem_l1093_109303


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l1093_109315

theorem product_remainder_mod_five : ∃ k : ℕ, 114 * 232 * 454^2 * 678 = 5 * k + 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l1093_109315


namespace NUMINAMATH_CALUDE_xy_system_implies_x2_plus_y2_l1093_109349

theorem xy_system_implies_x2_plus_y2 (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + x + y = 110) : 
  x^2 + y^2 = 80 := by
sorry

end NUMINAMATH_CALUDE_xy_system_implies_x2_plus_y2_l1093_109349


namespace NUMINAMATH_CALUDE_division_remainder_l1093_109333

theorem division_remainder :
  ∀ (dividend quotient divisor remainder : ℕ),
    dividend = divisor * quotient + remainder →
    dividend = 217 →
    divisor = 4 →
    quotient = 54 →
    remainder < divisor →
    remainder = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l1093_109333


namespace NUMINAMATH_CALUDE_correct_average_unchanged_l1093_109364

theorem correct_average_unchanged (n : ℕ) (initial_avg : ℚ) (error1 : ℚ) (wrong_num : ℚ) (correct_num : ℚ) :
  n = 10 →
  initial_avg = 40.2 →
  error1 = 18 →
  wrong_num = 13 →
  correct_num = 31 →
  (n : ℚ) * initial_avg - error1 - wrong_num + correct_num = (n : ℚ) * initial_avg :=
by sorry

end NUMINAMATH_CALUDE_correct_average_unchanged_l1093_109364


namespace NUMINAMATH_CALUDE_total_coins_luke_l1093_109391

/-- Given 5 piles of quarters, 5 piles of dimes, and 3 coins in each pile, 
    the total number of coins is 30. -/
theorem total_coins_luke (piles_quarters piles_dimes coins_per_pile : ℕ) 
  (h1 : piles_quarters = 5)
  (h2 : piles_dimes = 5)
  (h3 : coins_per_pile = 3) : 
  piles_quarters * coins_per_pile + piles_dimes * coins_per_pile = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_luke_l1093_109391


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1093_109389

theorem quadratic_factorization (p : ℕ+) :
  (∃ a b : ℤ, ∀ x : ℤ, x^2 - 5*x + p.val = (x - a) * (x - b)) →
  p.val = 4 ∨ p.val = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1093_109389


namespace NUMINAMATH_CALUDE_parking_fee_calculation_l1093_109347

/-- Calculates the parking fee based on the given fee structure and parking duration. -/
def parking_fee (initial_fee : ℕ) (additional_fee : ℕ) (initial_duration : ℕ) (increment : ℕ) (total_duration : ℕ) : ℕ :=
  let extra_duration := total_duration - initial_duration
  let extra_increments := (extra_duration + increment - 1) / increment
  initial_fee + additional_fee * extra_increments

/-- Theorem stating that the parking fee for 80 minutes is 1500 won given the specified fee structure. -/
theorem parking_fee_calculation : parking_fee 500 200 30 10 80 = 1500 := by
  sorry

#eval parking_fee 500 200 30 10 80

end NUMINAMATH_CALUDE_parking_fee_calculation_l1093_109347


namespace NUMINAMATH_CALUDE_division_problem_l1093_109388

theorem division_problem (remainder quotient divisor dividend : ℕ) : 
  remainder = 5 →
  divisor = 3 * remainder + 3 →
  dividend = 113 →
  dividend = divisor * quotient + remainder →
  (divisor : ℚ) / quotient = 3 / 1 := by sorry

end NUMINAMATH_CALUDE_division_problem_l1093_109388


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l1093_109352

def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N : Set ℝ := {x | 2*x < 2}

theorem intersection_M_complement_N : M ∩ (Set.univ \ N) = Set.Icc 1 3 := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l1093_109352


namespace NUMINAMATH_CALUDE_power_greater_than_square_l1093_109396

theorem power_greater_than_square (n : ℕ) (h : n ≥ 8) : 2^(n-1) > (n+1)^2 := by
  sorry

end NUMINAMATH_CALUDE_power_greater_than_square_l1093_109396


namespace NUMINAMATH_CALUDE_lucky_larry_coincidence_l1093_109361

theorem lucky_larry_coincidence :
  let a : ℚ := 2
  let b : ℚ := 3
  let c : ℚ := 4
  let d : ℚ := 5
  let f : ℚ := 4/5
  (a - b - c + d * f) = (a - (b - (c - (d * f)))) := by
  sorry

end NUMINAMATH_CALUDE_lucky_larry_coincidence_l1093_109361


namespace NUMINAMATH_CALUDE_trailing_zeros_310_factorial_l1093_109306

/-- The number of trailing zeros in n! --/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeros in 310! is 76 --/
theorem trailing_zeros_310_factorial :
  trailingZeros 310 = 76 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_310_factorial_l1093_109306


namespace NUMINAMATH_CALUDE_positive_sum_from_positive_difference_l1093_109316

theorem positive_sum_from_positive_difference (a b : ℝ) : a - |b| > 0 → b + a > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_from_positive_difference_l1093_109316


namespace NUMINAMATH_CALUDE_coin_distribution_theorem_l1093_109368

-- Define the number of cans
def num_cans : ℕ := 2015

-- Define the three initial configurations
def config_a (j : ℕ) : ℤ := 0
def config_b (j : ℕ) : ℤ := j
def config_c (j : ℕ) : ℤ := 2016 - j

-- Define the property that needs to be proven for each configuration
def has_solution (d : ℕ → ℤ) : Prop :=
  ∃ X : ℤ, ∀ j : ℕ, 1 ≤ j ∧ j ≤ num_cans → X ≡ d j [ZMOD j]

-- Theorem statement
theorem coin_distribution_theorem :
  has_solution config_a ∧ has_solution config_b ∧ has_solution config_c :=
sorry

end NUMINAMATH_CALUDE_coin_distribution_theorem_l1093_109368


namespace NUMINAMATH_CALUDE_candy_to_drink_ratio_l1093_109363

def deal_price : ℚ := 20
def ticket_price : ℚ := 8
def popcorn_price : ℚ := ticket_price - 3
def drink_price : ℚ := popcorn_price + 1
def savings : ℚ := 2

def normal_total : ℚ := deal_price + savings
def candy_price : ℚ := normal_total - (ticket_price + popcorn_price + drink_price)

theorem candy_to_drink_ratio : candy_price / drink_price = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_to_drink_ratio_l1093_109363


namespace NUMINAMATH_CALUDE_area_BEDC_is_30_l1093_109362

/-- Represents a parallelogram ABCD with a line DE parallel to AB -/
structure Parallelogram :=
  (AB : ℝ)
  (height : ℝ)
  (DE : ℝ)
  (is_parallelogram : Bool)
  (DE_parallel_AB : Bool)
  (E_midpoint_DC : Bool)

/-- Calculate the area of region BEDC in the given parallelogram -/
def area_BEDC (p : Parallelogram) : ℝ :=
  sorry

/-- Theorem stating that the area of region BEDC is 30 under given conditions -/
theorem area_BEDC_is_30 (p : Parallelogram) 
  (h1 : p.AB = 12)
  (h2 : p.height = 10)
  (h3 : p.DE = 6)
  (h4 : p.is_parallelogram = true)
  (h5 : p.DE_parallel_AB = true)
  (h6 : p.E_midpoint_DC = true) :
  area_BEDC p = 30 :=
sorry

end NUMINAMATH_CALUDE_area_BEDC_is_30_l1093_109362


namespace NUMINAMATH_CALUDE_p_start_time_correct_l1093_109359

/-- The time when J starts walking (in hours after midnight) -/
def j_start_time : ℝ := 12

/-- J's walking speed in km/h -/
def j_speed : ℝ := 6

/-- P's cycling speed in km/h -/
def p_speed : ℝ := 8

/-- The time when J is 3 km behind P (in hours after midnight) -/
def final_time : ℝ := 19.3

/-- The distance J is behind P at the final time (in km) -/
def distance_behind : ℝ := 3

/-- The time when P starts following J (in hours after midnight) -/
def p_start_time : ℝ := j_start_time + 1.45

theorem p_start_time_correct :
  j_speed * (final_time - j_start_time) + distance_behind =
  p_speed * (final_time - p_start_time) := by sorry

end NUMINAMATH_CALUDE_p_start_time_correct_l1093_109359


namespace NUMINAMATH_CALUDE_rainfall_problem_l1093_109335

/-- Rainfall problem -/
theorem rainfall_problem (day1 day2 day3 : ℝ) : 
  day1 = 4 →
  day2 = 5 * day1 →
  day3 = day1 + day2 - 6 →
  day3 = 18 := by
sorry

end NUMINAMATH_CALUDE_rainfall_problem_l1093_109335


namespace NUMINAMATH_CALUDE_olivias_correct_answers_l1093_109311

theorem olivias_correct_answers 
  (total_problems : ℕ) 
  (correct_points : ℤ) 
  (incorrect_points : ℤ) 
  (total_score : ℤ) 
  (h1 : total_problems = 15)
  (h2 : correct_points = 6)
  (h3 : incorrect_points = -3)
  (h4 : total_score = 45) :
  ∃ (correct_answers : ℕ),
    correct_answers * correct_points + 
    (total_problems - correct_answers) * incorrect_points = total_score ∧
    correct_answers = 10 := by
  sorry

end NUMINAMATH_CALUDE_olivias_correct_answers_l1093_109311


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1093_109313

def expansion (x : ℝ) := (1 + x^2) * (1 - x)^5

theorem coefficient_of_x_cubed : 
  (deriv (deriv (deriv expansion))) 0 / 6 = -15 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1093_109313


namespace NUMINAMATH_CALUDE_train_crossing_time_l1093_109367

/-- Proves the time taken by a train to cross a platform -/
theorem train_crossing_time (train_length platform_length : ℝ) 
  (time_to_cross_pole : ℝ) (h1 : train_length = 300) 
  (h2 : platform_length = 675) (h3 : time_to_cross_pole = 12) : 
  (train_length + platform_length) / (train_length / time_to_cross_pole) = 39 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1093_109367


namespace NUMINAMATH_CALUDE_jordan_machine_l1093_109337

theorem jordan_machine (x : ℚ) : ((3 * x - 6) / 2 + 9 = 27) → x = 14 := by
  sorry

end NUMINAMATH_CALUDE_jordan_machine_l1093_109337


namespace NUMINAMATH_CALUDE_team_allocation_proof_l1093_109346

/-- Proves that given the initial team sizes and total transfer, 
    the number of people allocated to Team A that makes its size 
    twice Team B's size is 23 -/
theorem team_allocation_proof 
  (initial_a initial_b transfer : ℕ) 
  (h_initial_a : initial_a = 31)
  (h_initial_b : initial_b = 26)
  (h_transfer : transfer = 24) :
  ∃ (x : ℕ), 
    x ≤ transfer ∧ 
    initial_a + x = 2 * (initial_b + (transfer - x)) ∧ 
    x = 23 := by
  sorry

end NUMINAMATH_CALUDE_team_allocation_proof_l1093_109346


namespace NUMINAMATH_CALUDE_exclusive_albums_count_l1093_109300

/-- The number of albums that are in either Andrew's or Bella's collection, but not both. -/
def exclusive_albums (shared : ℕ) (andrew_total : ℕ) (bella_unique : ℕ) : ℕ :=
  (andrew_total - shared) + bella_unique

/-- Theorem stating that the number of exclusive albums is 17 given the problem conditions. -/
theorem exclusive_albums_count :
  exclusive_albums 15 23 9 = 17 := by
  sorry

end NUMINAMATH_CALUDE_exclusive_albums_count_l1093_109300


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l1093_109353

theorem rectangle_area_theorem (d : ℝ) (h : d > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧
  l / w = 5 / 2 ∧
  d ^ 2 = (l / 2) ^ 2 + w ^ 2 ∧
  l * w = (5 / 13) * d ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l1093_109353


namespace NUMINAMATH_CALUDE_chopped_cube_height_l1093_109342

-- Define the cube
def cube_edge_length : ℝ := 2

-- Define the cut face as an equilateral triangle
def cut_face_is_equilateral_triangle : Prop := sorry

-- Define the remaining height
def remaining_height : ℝ := cube_edge_length - 1

-- Theorem statement
theorem chopped_cube_height :
  cut_face_is_equilateral_triangle →
  remaining_height = 1 := by sorry

end NUMINAMATH_CALUDE_chopped_cube_height_l1093_109342


namespace NUMINAMATH_CALUDE_pencil_sales_problem_l1093_109372

theorem pencil_sales_problem (eraser_price regular_price short_price : ℚ)
  (eraser_quantity short_quantity : ℕ) (total_revenue : ℚ)
  (h1 : eraser_price = 0.8)
  (h2 : regular_price = 0.5)
  (h3 : short_price = 0.4)
  (h4 : eraser_quantity = 200)
  (h5 : short_quantity = 35)
  (h6 : total_revenue = 194)
  (h7 : eraser_price * eraser_quantity + regular_price * x + short_price * short_quantity = total_revenue) :
  x = 40 := by
  sorry

#check pencil_sales_problem

end NUMINAMATH_CALUDE_pencil_sales_problem_l1093_109372


namespace NUMINAMATH_CALUDE_mural_hourly_rate_l1093_109328

/-- Calculates the hourly rate for painting a mural given its dimensions, painting rate, and total charge -/
theorem mural_hourly_rate (length width : ℝ) (paint_rate : ℝ) (total_charge : ℝ) :
  length = 20 ∧ width = 15 ∧ paint_rate = 20 ∧ total_charge = 15000 →
  total_charge / (length * width * paint_rate / 60) = 150 := by
  sorry

#check mural_hourly_rate

end NUMINAMATH_CALUDE_mural_hourly_rate_l1093_109328


namespace NUMINAMATH_CALUDE_three_right_angles_implies_rectangle_l1093_109381

/-- A quadrilateral is a polygon with four sides and four vertices. -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- An angle is right if it measures 90 degrees or π/2 radians. -/
def is_right_angle (a b c : ℝ × ℝ) : Prop := sorry

/-- A quadrilateral is a rectangle if all its angles are right angles. -/
def is_rectangle (q : Quadrilateral) : Prop :=
  ∀ i : Fin 4, is_right_angle (q.vertices i) (q.vertices (i + 1)) (q.vertices (i + 2))

/-- If a quadrilateral has three right angles, then it is a rectangle. -/
theorem three_right_angles_implies_rectangle (q : Quadrilateral) :
  (∃ i j k : Fin 4, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    is_right_angle (q.vertices i) (q.vertices (i + 1)) (q.vertices (i + 2)) ∧
    is_right_angle (q.vertices j) (q.vertices (j + 1)) (q.vertices (j + 2)) ∧
    is_right_angle (q.vertices k) (q.vertices (k + 1)) (q.vertices (k + 2)))
  → is_rectangle q :=
sorry

end NUMINAMATH_CALUDE_three_right_angles_implies_rectangle_l1093_109381


namespace NUMINAMATH_CALUDE_andrews_blue_balloons_l1093_109366

/-- Given information about Andrew's balloons, prove the number of blue balloons. -/
theorem andrews_blue_balloons :
  ∀ (total_balloons remaining_balloons purple_balloons : ℕ),
    total_balloons = 2 * remaining_balloons →
    remaining_balloons = 378 →
    purple_balloons = 453 →
    total_balloons - purple_balloons = 303 := by
  sorry

end NUMINAMATH_CALUDE_andrews_blue_balloons_l1093_109366


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1093_109332

theorem quadratic_factorization :
  ∀ x : ℝ, 12 * x^2 + 4 * x - 16 = 4 * (x - 1) * (3 * x + 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1093_109332


namespace NUMINAMATH_CALUDE_clara_final_stickers_l1093_109307

/-- Calculates the number of stickers Clara has left after a series of operations --/
def clara_stickers : ℕ :=
  let initial := 100
  let after_boy := initial - 10
  let after_teacher := after_boy + 50
  let after_classmates := after_teacher - (after_teacher * 20 / 100)
  let exchange_amount := after_classmates / 3
  let after_exchange := after_classmates - exchange_amount + (2 * exchange_amount)
  let give_to_friends := after_exchange / 4
  let remaining := after_exchange - (give_to_friends / 3 * 3)
  remaining

/-- Theorem stating that Clara ends up with 114 stickers --/
theorem clara_final_stickers : clara_stickers = 114 := by
  sorry


end NUMINAMATH_CALUDE_clara_final_stickers_l1093_109307


namespace NUMINAMATH_CALUDE_age_difference_l1093_109341

/-- Given the ages of Patrick, Michael, and Monica satisfying certain ratios and sum, 
    prove that the difference between Monica's and Patrick's ages is 33 years. -/
theorem age_difference (patrick michael monica : ℕ) : 
  patrick * 5 = michael * 3 →  -- Patrick and Michael's ages are in ratio 3:5
  michael * 4 = monica * 3 →   -- Michael and Monica's ages are in ratio 3:4
  patrick + michael + monica = 132 →  -- Sum of their ages is 132
  monica - patrick = 33 := by  -- Difference between Monica's and Patrick's ages is 33
sorry  -- Proof omitted

end NUMINAMATH_CALUDE_age_difference_l1093_109341


namespace NUMINAMATH_CALUDE_connie_red_markers_l1093_109393

/-- The number of red markers Connie has -/
def red_markers : ℕ := 3343 - 1028

/-- The total number of markers Connie has -/
def total_markers : ℕ := 3343

/-- The number of blue markers Connie has -/
def blue_markers : ℕ := 1028

theorem connie_red_markers :
  red_markers = 2315 ∧ total_markers = red_markers + blue_markers :=
sorry

end NUMINAMATH_CALUDE_connie_red_markers_l1093_109393


namespace NUMINAMATH_CALUDE_snow_white_marbles_l1093_109309

theorem snow_white_marbles (x : ℕ) (y : ℕ) : 
  (x > 0) →
  (y > 0) →
  (y ≤ 6) →
  (7 * x - (1 + 2 + 3 + 4 + 5 + 6) - y = 46) →
  (x = 10 ∧ y = 3) := by
sorry

end NUMINAMATH_CALUDE_snow_white_marbles_l1093_109309


namespace NUMINAMATH_CALUDE_excluded_students_average_mark_l1093_109375

/-- Proves that given a class of 35 students with an average mark of 80,
    if 5 students are excluded and the remaining students have an average mark of 90,
    then the average mark of the excluded students is 20. -/
theorem excluded_students_average_mark
  (total_students : ℕ)
  (class_average : ℚ)
  (remaining_students : ℕ)
  (remaining_average : ℚ)
  (h1 : total_students = 35)
  (h2 : class_average = 80)
  (h3 : remaining_students = 30)
  (h4 : remaining_average = 90) :
  let excluded_students := total_students - remaining_students
  let excluded_average := (total_students * class_average - remaining_students * remaining_average) / excluded_students
  excluded_average = 20 := by
  sorry

#check excluded_students_average_mark

end NUMINAMATH_CALUDE_excluded_students_average_mark_l1093_109375


namespace NUMINAMATH_CALUDE_initial_number_theorem_l1093_109305

theorem initial_number_theorem (x : ℤ) : (x + 2)^2 = x^2 - 2016 → x = -505 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_theorem_l1093_109305


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_abs_function_l1093_109380

-- Define the function
def f (x : ℝ) : ℝ := |x - 4| + 7

-- State the theorem
theorem minimum_point_of_translated_abs_function :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), f x ≥ f x₀) ∧ f x₀ = 7 ∧ x₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_abs_function_l1093_109380


namespace NUMINAMATH_CALUDE_least_even_perimeter_l1093_109377

theorem least_even_perimeter (a b c : ℕ) : 
  a = 24 →
  b = 37 →
  c ≥ a ∧ c ≥ b →
  a + b + c > a + b →
  Even (a + b + c) →
  (∀ x : ℕ, x < c → ¬(Even (a + b + x) ∧ a + b + x > a + b)) →
  a + b + c = 100 := by
  sorry

end NUMINAMATH_CALUDE_least_even_perimeter_l1093_109377


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1093_109301

theorem inequality_system_solution (x : ℝ) : 
  (1 / x < 1 ∧ |4 * x - 1| > 2) ↔ (x < -1/4 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1093_109301


namespace NUMINAMATH_CALUDE_no_real_solutions_l1093_109323

theorem no_real_solutions :
  ¬∃ (x : ℝ), (8 * x^2 + 150 * x - 5) / (3 * x + 50) = 4 * x + 7 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1093_109323


namespace NUMINAMATH_CALUDE_integer_k_not_dividing_binomial_coefficient_l1093_109327

theorem integer_k_not_dividing_binomial_coefficient (k : ℤ) : 
  k ≠ 1 ↔ ∃ (S : Set ℕ+), Set.Infinite S ∧ 
    ∀ n ∈ S, ¬(n + k : ℤ) ∣ (Nat.choose (2 * n) n : ℤ) :=
by sorry

end NUMINAMATH_CALUDE_integer_k_not_dividing_binomial_coefficient_l1093_109327


namespace NUMINAMATH_CALUDE_negation_absolute_value_inequality_l1093_109373

theorem negation_absolute_value_inequality :
  (¬ ∀ x : ℝ, |x - 2| < 3) ↔ (∃ x : ℝ, |x - 2| ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_absolute_value_inequality_l1093_109373


namespace NUMINAMATH_CALUDE_remainder_when_z_plus_3_div_9_is_integer_l1093_109312

theorem remainder_when_z_plus_3_div_9_is_integer (z : ℤ) :
  ∃ k : ℤ, (z + 3) / 9 = k → z ≡ 6 [ZMOD 9] := by
  sorry

end NUMINAMATH_CALUDE_remainder_when_z_plus_3_div_9_is_integer_l1093_109312


namespace NUMINAMATH_CALUDE_octagon_diagonals_l1093_109351

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l1093_109351


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l1093_109317

theorem square_circle_area_ratio (s r : ℝ) (h : s > 0) (k : r > 0) (eq : 4 * s = 4 * Real.pi * r) :
  s^2 / (Real.pi * r^2) = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l1093_109317


namespace NUMINAMATH_CALUDE_load_transport_l1093_109374

theorem load_transport (total_load : ℝ) (box_weight_max : ℝ) (num_trucks : ℕ) (truck_capacity : ℝ) :
  total_load = 13.5 →
  box_weight_max ≤ 0.35 →
  num_trucks = 11 →
  truck_capacity = 1.5 →
  ∃ (n : ℕ), n ≤ num_trucks ∧ n * truck_capacity ≥ total_load :=
by sorry

end NUMINAMATH_CALUDE_load_transport_l1093_109374


namespace NUMINAMATH_CALUDE_sin_405_degrees_l1093_109379

theorem sin_405_degrees : Real.sin (405 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_405_degrees_l1093_109379


namespace NUMINAMATH_CALUDE_salon_customers_l1093_109356

/-- The number of customers a salon has each day, given their hairspray usage and purchasing. -/
theorem salon_customers (total_cans : ℕ) (extra_cans : ℕ) (cans_per_customer : ℕ) : 
  total_cans = 33 →
  extra_cans = 5 →
  cans_per_customer = 2 →
  (total_cans - extra_cans) / cans_per_customer = 14 :=
by sorry

end NUMINAMATH_CALUDE_salon_customers_l1093_109356


namespace NUMINAMATH_CALUDE_earliest_meeting_time_is_440_l1093_109338

/-- Represents the lap time in minutes for each runner -/
structure LapTime where
  charlie : ℕ
  ben : ℕ
  laura : ℕ

/-- Calculates the earliest meeting time in minutes -/
def earliest_meeting_time (lt : LapTime) : ℕ :=
  Nat.lcm (Nat.lcm lt.charlie lt.ben) lt.laura

/-- Theorem: Given the specific lap times, the earliest meeting time is 440 minutes -/
theorem earliest_meeting_time_is_440 :
  earliest_meeting_time ⟨5, 8, 11⟩ = 440 := by
  sorry

end NUMINAMATH_CALUDE_earliest_meeting_time_is_440_l1093_109338


namespace NUMINAMATH_CALUDE_fence_cost_per_foot_l1093_109390

/-- The cost per foot of fencing a square plot -/
theorem fence_cost_per_foot 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h1 : area = 25) 
  (h2 : total_cost = 1160) : 
  total_cost / (4 * Real.sqrt area) = 58 := by
sorry


end NUMINAMATH_CALUDE_fence_cost_per_foot_l1093_109390


namespace NUMINAMATH_CALUDE_problem_solution_l1093_109340

theorem problem_solution (x : ℝ) : ((x / 4) * 5 + 10 - 12 = 48) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1093_109340


namespace NUMINAMATH_CALUDE_least_possible_difference_l1093_109329

theorem least_possible_difference (x y z : ℤ) : 
  x < y → y < z → 
  y - x > 9 → 
  Even x → Odd y → Odd z → 
  (∀ w, w = z - x → w ≥ 13) ∧ ∃ w, w = z - x ∧ w = 13 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_difference_l1093_109329


namespace NUMINAMATH_CALUDE_x_to_twenty_l1093_109394

theorem x_to_twenty (x : ℝ) (h : x + 1/x = Real.sqrt 5) : x^20 = 16163 := by
  sorry

end NUMINAMATH_CALUDE_x_to_twenty_l1093_109394
