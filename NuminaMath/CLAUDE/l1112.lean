import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l1112_111253

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l1112_111253


namespace NUMINAMATH_CALUDE_chicken_flash_sale_theorem_l1112_111284

/-- Represents the original selling price of a free-range ecological chicken -/
def original_price : ℝ := sorry

/-- Represents the flash sale price of a free-range ecological chicken -/
def flash_sale_price : ℝ := original_price - 15

/-- Represents the percentage increase in buyers every 30 minutes -/
def m : ℝ := sorry

theorem chicken_flash_sale_theorem :
  (120 / flash_sale_price = 2 * (90 / original_price)) ∧
  (50 + 50 * (1 + m / 100) + 50 * (1 + m / 100)^2 = 5460 / flash_sale_price) →
  original_price = 45 ∧ m = 20 := by sorry

end NUMINAMATH_CALUDE_chicken_flash_sale_theorem_l1112_111284


namespace NUMINAMATH_CALUDE_complex_sum_equals_z_l1112_111259

theorem complex_sum_equals_z (z : ℂ) (h : z^2 - z + 1 = 0) : 
  z^107 + z^108 + z^109 + z^110 + z^111 = z := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_z_l1112_111259


namespace NUMINAMATH_CALUDE_x_is_integer_l1112_111280

theorem x_is_integer (x : ℝ) (h1 : ∃ n : ℤ, x^3 - x = n) (h2 : ∃ m : ℤ, x^4 - x = m) : ∃ k : ℤ, x = k := by
  sorry

end NUMINAMATH_CALUDE_x_is_integer_l1112_111280


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1112_111209

open Set

universe u

def U : Set ℝ := univ

def A : Set ℝ := {x : ℝ | x^2 - 2*x < 0}

def B : Set ℝ := {x : ℝ | x ≥ 1}

theorem intersection_A_complement_B : A ∩ (U \ B) = Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1112_111209


namespace NUMINAMATH_CALUDE_triangle_equality_l1112_111292

theorem triangle_equality (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_equation : (c - b) / a + (a - c) / b + (b - a) / c = 0) :
  a = b ∨ b = c ∨ c = a :=
sorry

end NUMINAMATH_CALUDE_triangle_equality_l1112_111292


namespace NUMINAMATH_CALUDE_max_product_sum_l1112_111271

theorem max_product_sum (f g h j : ℕ) : 
  f ∈ ({7, 8, 9, 10} : Set ℕ) →
  g ∈ ({7, 8, 9, 10} : Set ℕ) →
  h ∈ ({7, 8, 9, 10} : Set ℕ) →
  j ∈ ({7, 8, 9, 10} : Set ℕ) →
  f ≠ g → f ≠ h → f ≠ j → g ≠ h → g ≠ j → h ≠ j →
  (f * g + g * h + h * j + f * j) ≤ 289 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_l1112_111271


namespace NUMINAMATH_CALUDE_cubic_polynomial_three_distinct_roots_l1112_111246

/-- A cubic polynomial with specific properties has three distinct real roots -/
theorem cubic_polynomial_three_distinct_roots 
  (f : ℝ → ℝ) (a b c : ℝ) 
  (h_cubic : ∀ x, f x = x^3 + a*x^2 + b*x + c) 
  (h_b_neg : b < 0) 
  (h_ab_9c : a * b = 9 * c) : 
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = 0 ∧ f y = 0 ∧ f z = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_three_distinct_roots_l1112_111246


namespace NUMINAMATH_CALUDE_square_area_to_cube_volume_ratio_l1112_111273

theorem square_area_to_cube_volume_ratio 
  (cube : Real → Real) 
  (square : Real → Real) 
  (h : ∀ s : Real, s > 0 → s * Real.sqrt 3 = 4 * square s) :
  ∀ s : Real, s > 0 → (square s)^2 / (cube s) = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_to_cube_volume_ratio_l1112_111273


namespace NUMINAMATH_CALUDE_recurring_decimal_sum_l1112_111266

theorem recurring_decimal_sum : 
  (∃ (x y : ℚ), x = 123 / 999 ∧ y = 123 / 999999 ∧ x + y = 154 / 1001) :=
by sorry

end NUMINAMATH_CALUDE_recurring_decimal_sum_l1112_111266


namespace NUMINAMATH_CALUDE_construct_from_blocks_l1112_111232

/-- A building block consists of 7 unit cubes in a 2x2x2 shape with one corner unit cube missing. -/
structure BuildingBlock :=
  (size : Nat)
  (unit_cubes : Nat)

/-- Definition of our specific building block -/
def specific_block : BuildingBlock :=
  { size := 2,
    unit_cubes := 7 }

/-- A cube with one unit removed -/
structure CubeWithUnitRemoved :=
  (edge_length : Nat)
  (total_units : Nat)

/-- Function to check if a cube with a unit removed can be constructed from building blocks -/
def can_construct (c : CubeWithUnitRemoved) (b : BuildingBlock) : Prop :=
  ∃ (num_blocks : Nat), c.total_units = num_blocks * b.unit_cubes

/-- Main theorem -/
theorem construct_from_blocks (n : Nat) (h : n ≥ 2) :
  let c := CubeWithUnitRemoved.mk (2^n) ((2^n)^3 - 1)
  can_construct c specific_block :=
by sorry

end NUMINAMATH_CALUDE_construct_from_blocks_l1112_111232


namespace NUMINAMATH_CALUDE_income_comparison_l1112_111249

/-- Represents the problem of calculating incomes relative to Juan's base income -/
theorem income_comparison (J : ℝ) (J_pos : J > 0) : 
  let tim_base := 0.7 * J
  let mary_total := 1.12 * J * 1.1
  let lisa_base := 0.63 * J
  let lisa_total := lisa_base * 1.03
  let alan_base := lisa_base / 1.15
  let nina_base := 1.25 * J
  let nina_total := nina_base * 1.07
  (mary_total + lisa_total + nina_total) / J = 3.2184 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l1112_111249


namespace NUMINAMATH_CALUDE_complex_power_result_l1112_111203

theorem complex_power_result : (3 * Complex.cos (π / 4) + 3 * Complex.I * Complex.sin (π / 4)) ^ 4 = (-81 : ℂ) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_result_l1112_111203


namespace NUMINAMATH_CALUDE_shelly_money_l1112_111200

/-- Calculates the total amount of money Shelly has given the number of $10 and $5 bills -/
def total_money (ten_dollar_bills : ℕ) (five_dollar_bills : ℕ) : ℕ :=
  10 * ten_dollar_bills + 5 * five_dollar_bills

/-- Proves that Shelly has $130 given the conditions -/
theorem shelly_money : 
  let ten_dollar_bills : ℕ := 10
  let five_dollar_bills : ℕ := ten_dollar_bills - 4
  total_money ten_dollar_bills five_dollar_bills = 130 := by
sorry

end NUMINAMATH_CALUDE_shelly_money_l1112_111200


namespace NUMINAMATH_CALUDE_money_difference_l1112_111268

/-- Calculates the difference between final and initial amounts given monetary transactions --/
theorem money_difference (initial chores birthday neighbor candy lost : ℕ) : 
  initial = 2 →
  chores = 5 →
  birthday = 10 →
  neighbor = 7 →
  candy = 3 →
  lost = 2 →
  (initial + chores + birthday + neighbor - candy - lost) - initial = 17 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_l1112_111268


namespace NUMINAMATH_CALUDE_fence_cost_l1112_111201

/-- The cost of building a fence around a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 56) :
  4 * Real.sqrt area * price_per_foot = 3808 := by
  sorry

#check fence_cost

end NUMINAMATH_CALUDE_fence_cost_l1112_111201


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l1112_111296

theorem ratio_a_to_b (a b : ℚ) (h : (12*a - 5*b) / (17*a - 3*b) = 4/7) : a/b = 23/16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l1112_111296


namespace NUMINAMATH_CALUDE_bd_production_l1112_111278

/-- Represents the total production of all workshops -/
def total_production : ℕ := 2800

/-- Represents the total number of units sampled for quality inspection -/
def total_sampled : ℕ := 140

/-- Represents the number of units sampled from workshops A and C combined -/
def ac_sampled : ℕ := 60

/-- Theorem stating that the total production from workshops B and D is 1600 units -/
theorem bd_production : 
  total_production - (ac_sampled * (total_production / total_sampled)) = 1600 := by
  sorry

end NUMINAMATH_CALUDE_bd_production_l1112_111278


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_progression_l1112_111252

theorem sequence_is_arithmetic_progression 
  (a : ℕ → ℝ) 
  (h : ∀ m n : ℕ, |a m + a n - a (m + n)| ≤ 1 / (m + n : ℝ)) : 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d := by
  sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_progression_l1112_111252


namespace NUMINAMATH_CALUDE_classroom_puzzle_l1112_111293

theorem classroom_puzzle (initial_boys initial_girls : ℕ) : 
  initial_boys = initial_girls →
  initial_boys = 2 * (initial_girls - 8) →
  initial_boys + initial_girls = 32 := by
sorry

end NUMINAMATH_CALUDE_classroom_puzzle_l1112_111293


namespace NUMINAMATH_CALUDE_intersection_equality_implies_m_values_l1112_111210

def A (m : ℝ) : Set ℝ := {3, m^2}
def B (m : ℝ) : Set ℝ := {-1, 3, 3*m-2}

theorem intersection_equality_implies_m_values (m : ℝ) :
  A m ∩ B m = A m → m = 1 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_m_values_l1112_111210


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l1112_111224

theorem rectangle_area_with_inscribed_circle (r : ℝ) (ratio : ℝ) : 
  r = 7 → ratio = 3 → 2 * r * ratio * 2 * r = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l1112_111224


namespace NUMINAMATH_CALUDE_grocery_value_proof_l1112_111231

def car_cost : ℝ := 14600
def initial_savings : ℝ := 14500
def trips : ℕ := 40
def fixed_charge : ℝ := 1.5
def grocery_charge_rate : ℝ := 0.05

theorem grocery_value_proof (grocery_value : ℝ) : 
  car_cost - initial_savings = trips * fixed_charge + grocery_charge_rate * grocery_value →
  grocery_value = 800 := by
  sorry

end NUMINAMATH_CALUDE_grocery_value_proof_l1112_111231


namespace NUMINAMATH_CALUDE_multiples_properties_l1112_111207

theorem multiples_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ 
  (∃ p : ℤ, a - b = 4 * p) := by
sorry

end NUMINAMATH_CALUDE_multiples_properties_l1112_111207


namespace NUMINAMATH_CALUDE_reading_homework_pages_l1112_111287

theorem reading_homework_pages (total_pages math_pages : ℕ) 
  (h1 : total_pages = 7) 
  (h2 : math_pages = 5) : 
  total_pages - math_pages = 2 := by
sorry

end NUMINAMATH_CALUDE_reading_homework_pages_l1112_111287


namespace NUMINAMATH_CALUDE_class_item_distribution_l1112_111248

/-- Calculates the total number of items distributed in a class --/
def calculate_total_items (num_children : ℕ) 
                          (initial_pencils : ℕ) 
                          (initial_erasers : ℕ) 
                          (initial_crayons : ℕ) 
                          (extra_pencils : ℕ) 
                          (extra_crayons : ℕ) 
                          (extra_erasers : ℕ) 
                          (num_children_extra_pencils_crayons : ℕ) : ℕ × ℕ × ℕ :=
  let total_pencils := num_children * initial_pencils + num_children_extra_pencils_crayons * extra_pencils
  let total_erasers := num_children * initial_erasers + (num_children - num_children_extra_pencils_crayons) * extra_erasers
  let total_crayons := num_children * initial_crayons + num_children_extra_pencils_crayons * extra_crayons
  (total_pencils, total_erasers, total_crayons)

theorem class_item_distribution :
  let num_children : ℕ := 18
  let initial_pencils : ℕ := 6
  let initial_erasers : ℕ := 3
  let initial_crayons : ℕ := 12
  let extra_pencils : ℕ := 5
  let extra_crayons : ℕ := 8
  let extra_erasers : ℕ := 2
  let num_children_extra_pencils_crayons : ℕ := 10
  
  calculate_total_items num_children initial_pencils initial_erasers initial_crayons
                        extra_pencils extra_crayons extra_erasers
                        num_children_extra_pencils_crayons = (158, 70, 296) := by
  sorry

end NUMINAMATH_CALUDE_class_item_distribution_l1112_111248


namespace NUMINAMATH_CALUDE_chord_length_l1112_111242

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by sorry

end NUMINAMATH_CALUDE_chord_length_l1112_111242


namespace NUMINAMATH_CALUDE_cricket_bat_price_l1112_111221

/-- Represents the cost and selling prices of an item -/
structure PriceData where
  cost_price_a : ℝ
  selling_price_b : ℝ
  selling_price_c : ℝ

/-- Theorem stating the relationship between the prices and profits -/
theorem cricket_bat_price (p : PriceData) 
  (profit_a : p.selling_price_b = 1.20 * p.cost_price_a)
  (profit_b : p.selling_price_c = 1.25 * p.selling_price_b)
  (final_price : p.selling_price_c = 222) :
  p.cost_price_a = 148 := by
  sorry

#check cricket_bat_price

end NUMINAMATH_CALUDE_cricket_bat_price_l1112_111221


namespace NUMINAMATH_CALUDE_class_average_weight_l1112_111213

theorem class_average_weight (students_A : ℕ) (students_B : ℕ) 
  (avg_weight_A : ℝ) (avg_weight_B : ℝ) :
  students_A = 30 →
  students_B = 20 →
  avg_weight_A = 40 →
  avg_weight_B = 35 →
  let total_students := students_A + students_B
  let total_weight := students_A * avg_weight_A + students_B * avg_weight_B
  (total_weight / total_students : ℝ) = 38 := by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l1112_111213


namespace NUMINAMATH_CALUDE_square_overlap_area_l1112_111276

/-- The area of overlapping regions in a rectangle with four squares -/
theorem square_overlap_area (total_square_area sum_individual_areas uncovered_area : ℝ) :
  total_square_area = 27.5 ∧ 
  sum_individual_areas = 30 ∧ 
  uncovered_area = 1.5 →
  sum_individual_areas - total_square_area + uncovered_area = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_overlap_area_l1112_111276


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l1112_111282

/-- A square inscribed in a right triangle with one vertex at the right angle -/
def square_in_triangle_vertex (a b c : ℝ) (x : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ (b - x) / x = a / b

/-- A square inscribed in a right triangle with one side on the hypotenuse -/
def square_in_triangle_hypotenuse (a b c : ℝ) (y : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ y / c = (b - y) / a

theorem inscribed_squares_ratio :
  ∀ (x y : ℝ),
    square_in_triangle_vertex 5 12 13 x →
    square_in_triangle_hypotenuse 6 8 10 y →
    x / y = 216 / 85 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l1112_111282


namespace NUMINAMATH_CALUDE_arctan_of_tan_difference_l1112_111251

theorem arctan_of_tan_difference (θ : Real) : 
  θ ∈ Set.Icc 0 180 → 
  Real.arctan (Real.tan (80 * π / 180) - 3 * Real.tan (30 * π / 180)) = 50 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_arctan_of_tan_difference_l1112_111251


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l1112_111229

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be represented in scientific notation -/
def number : ℕ := 2700000

/-- The scientific notation representation of the number -/
def scientific_representation : ScientificNotation :=
  { coefficient := 2.7
    exponent := 6
    property := by sorry }

/-- Theorem stating that the scientific notation representation is correct -/
theorem scientific_notation_correct :
  (scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent) = number := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l1112_111229


namespace NUMINAMATH_CALUDE_davantes_girl_friends_l1112_111206

def days_in_week : ℕ := 7

def davantes_friends (days : ℕ) : ℕ := 2 * days

def boy_friends : ℕ := 11

theorem davantes_girl_friends :
  davantes_friends days_in_week - boy_friends = 3 := by
  sorry

end NUMINAMATH_CALUDE_davantes_girl_friends_l1112_111206


namespace NUMINAMATH_CALUDE_sqrt_26_is_7th_term_l1112_111288

theorem sqrt_26_is_7th_term (a : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → a n = Real.sqrt (4 * n - 2)) →
  a 7 = Real.sqrt 26 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_26_is_7th_term_l1112_111288


namespace NUMINAMATH_CALUDE_min_value_theorem_l1112_111218

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  (2 / a) + (3 / b) ≥ 25 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + 3 * b₀ = 1 ∧ (2 / a₀) + (3 / b₀) = 25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1112_111218


namespace NUMINAMATH_CALUDE_cell_plan_comparison_l1112_111298

/-- Represents a cell phone plan with a flat fee and per-minute rate -/
structure CellPlan where
  flatFee : ℕ  -- Flat fee in cents
  perMinRate : ℕ  -- Per-minute rate in cents
  
/-- Calculates the cost of a plan for a given number of minutes -/
def planCost (plan : CellPlan) (minutes : ℕ) : ℕ :=
  plan.flatFee + plan.perMinRate * minutes

/-- The three cell phone plans -/
def planX : CellPlan := { flatFee := 0, perMinRate := 15 }
def planY : CellPlan := { flatFee := 2500, perMinRate := 7 }
def planZ : CellPlan := { flatFee := 3000, perMinRate := 6 }

theorem cell_plan_comparison :
  (∀ m : ℕ, m < 313 → planCost planX m ≤ planCost planY m) ∧
  (planCost planY 313 < planCost planX 313) ∧
  (∀ m : ℕ, m < 334 → planCost planX m ≤ planCost planZ m) ∧
  (planCost planZ 334 < planCost planX 334) :=
by sorry


end NUMINAMATH_CALUDE_cell_plan_comparison_l1112_111298


namespace NUMINAMATH_CALUDE_identify_participants_with_2k_minus_3_questions_l1112_111274

/-- Represents the type of participant -/
inductive Participant
| Chemist
| Alchemist

/-- Represents the state of the identification process -/
structure IdentificationState where
  participants : Nat
  chemists : Nat
  alchemists : Nat
  questions_asked : Nat

/-- The main theorem stating that 2k-3 questions are sufficient -/
theorem identify_participants_with_2k_minus_3_questions 
  (k : Nat) 
  (h : k > 0) 
  (more_chemists : ∃ (c a : Nat), c > a ∧ c + a = k) :
  ∃ (strategy : IdentificationState → Participant), 
    (∀ (state : IdentificationState), 
      state.participants = k → 
      state.chemists > state.alchemists → 
      state.questions_asked ≤ 2 * k - 3 → 
      (∀ p, strategy state = p → 
        (p = Participant.Chemist → state.chemists > 0) ∧ 
        (p = Participant.Alchemist → state.alchemists > 0))) :=
sorry

end NUMINAMATH_CALUDE_identify_participants_with_2k_minus_3_questions_l1112_111274


namespace NUMINAMATH_CALUDE_ellipse_left_vertex_l1112_111267

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
def ellipse_conditions (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ b = 4 ∧ (3 : ℝ)^2 = a^2 - b^2

-- Theorem statement
theorem ellipse_left_vertex (a b : ℝ) :
  ellipse_conditions a b →
  ∃ (x y : ℝ), ellipse a b x y ∧ x = -5 ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_left_vertex_l1112_111267


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l1112_111254

theorem mixed_number_calculation : 
  23 * ((1 + 2/3) + (2 + 1/4)) / ((1 + 1/2) + (1 + 1/5)) = 3 + 43/108 := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l1112_111254


namespace NUMINAMATH_CALUDE_proposition_p_and_not_q_l1112_111289

theorem proposition_p_and_not_q :
  (∃ x : ℝ, x - 2 > Real.log x) ∧ ¬(∀ x : ℝ, Real.sin x < x) := by sorry

end NUMINAMATH_CALUDE_proposition_p_and_not_q_l1112_111289


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l1112_111211

theorem count_integers_satisfying_inequality :
  (Finset.filter (fun n : ℤ => (n - 3) * (n + 2) * (n + 6) < 0)
    (Finset.Icc (-11 : ℤ) 11)).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l1112_111211


namespace NUMINAMATH_CALUDE_equation_solutions_l1112_111202

theorem equation_solutions : 
  let f (x : ℝ) := 1/((x-1)*(x-2)) + 1/((x-2)*(x-3)) + 1/((x-3)*(x-4)) + 1/((x-4)*(x-5))
  ∀ x : ℝ, f x = 1/12 ↔ x = 12 ∨ x = -4.5 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1112_111202


namespace NUMINAMATH_CALUDE_intersection_implies_difference_l1112_111297

def set_A (a : ℝ) : Set (ℝ × ℝ) := {p | p.2 = a * p.1 + 6}
def set_B : Set (ℝ × ℝ) := {p | p.2 = 5 * p.1 - 3}

theorem intersection_implies_difference (a b : ℝ) :
  (1, b) ∈ set_A a ∩ set_B → a - b = -6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_difference_l1112_111297


namespace NUMINAMATH_CALUDE_solve_equation_l1112_111286

theorem solve_equation (r : ℚ) : 4 * (r - 10) = 3 * (3 - 3 * r) + 9 → r = 58 / 13 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1112_111286


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1112_111205

theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 1/7
  let S := ∑' n, a * r^n
  S = 7/6 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1112_111205


namespace NUMINAMATH_CALUDE_fred_weekend_earnings_l1112_111245

/-- Fred's initial amount of money in dollars -/
def fred_initial : ℕ := 19

/-- Fred's final amount of money in dollars -/
def fred_final : ℕ := 40

/-- Fred's earnings over the weekend in dollars -/
def fred_earnings : ℕ := fred_final - fred_initial

theorem fred_weekend_earnings : fred_earnings = 21 := by
  sorry

end NUMINAMATH_CALUDE_fred_weekend_earnings_l1112_111245


namespace NUMINAMATH_CALUDE_committee_probability_grammar_club_committee_probability_l1112_111262

/-- The probability of selecting a committee with at least one boy and one girl -/
theorem committee_probability (total : ℕ) (boys : ℕ) (girls : ℕ) (committee_size : ℕ) :
  total = boys + girls →
  total = 25 →
  boys = 15 →
  girls = 10 →
  committee_size = 5 →
  (Nat.choose total committee_size - (Nat.choose boys committee_size + Nat.choose girls committee_size)) / 
  Nat.choose total committee_size = 195 / 208 := by
  sorry

/-- Main theorem stating the probability for the specific case -/
theorem grammar_club_committee_probability :
  (Nat.choose 25 5 - (Nat.choose 15 5 + Nat.choose 10 5)) / Nat.choose 25 5 = 195 / 208 := by
  sorry

end NUMINAMATH_CALUDE_committee_probability_grammar_club_committee_probability_l1112_111262


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l1112_111261

/-- Represents a parallelogram with given base and height -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- Theorem: The area of a parallelogram with base 22 cm and height 14 cm is 308 square centimeters -/
theorem parallelogram_area_example : 
  let p : Parallelogram := { base := 22, height := 14 }
  area p = 308 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l1112_111261


namespace NUMINAMATH_CALUDE_multiply_37_23_l1112_111263

theorem multiply_37_23 : 37 * 23 = 851 := by
  sorry

end NUMINAMATH_CALUDE_multiply_37_23_l1112_111263


namespace NUMINAMATH_CALUDE_workshop_workers_l1112_111219

theorem workshop_workers (total_average : ℕ) (tech_count : ℕ) (tech_average : ℕ) (non_tech_average : ℕ) :
  total_average = 8000 →
  tech_count = 7 →
  tech_average = 12000 →
  non_tech_average = 6000 →
  ∃ (total_workers : ℕ), 
    total_workers * total_average = tech_count * tech_average + (total_workers - tech_count) * non_tech_average ∧
    total_workers = 21 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l1112_111219


namespace NUMINAMATH_CALUDE_unequal_grandchildren_probability_l1112_111233

-- Define the number of grandchildren
def n : ℕ := 12

-- Define the probability of a child being male or female
def p : ℚ := 1/2

-- Define the probability of having an equal number of grandsons and granddaughters
def prob_equal : ℚ := (n.choose (n/2)) / (2^n)

-- Theorem statement
theorem unequal_grandchildren_probability :
  1 - prob_equal = 793/1024 := by sorry

end NUMINAMATH_CALUDE_unequal_grandchildren_probability_l1112_111233


namespace NUMINAMATH_CALUDE_escalator_time_l1112_111228

/-- The time taken for a person to cover the entire length of an escalator -/
theorem escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) 
  (h1 : escalator_speed = 20)
  (h2 : person_speed = 4)
  (h3 : escalator_length = 210) : 
  escalator_length / (escalator_speed + person_speed) = 8.75 := by
sorry

end NUMINAMATH_CALUDE_escalator_time_l1112_111228


namespace NUMINAMATH_CALUDE_savings_exceed_500_on_sunday_l1112_111264

/-- The day of the week, starting from Sunday as 0 -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculate the total savings after n days -/
def totalSavings (n : ℕ) : ℚ :=
  (3^n - 1) / 2

/-- Convert number of days to day of the week -/
def toDayOfWeek (n : ℕ) : DayOfWeek :=
  match n % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem savings_exceed_500_on_sunday :
  ∃ n : ℕ, totalSavings n > 500 ∧
    ∀ m : ℕ, m < n → totalSavings m ≤ 500 ∧
    toDayOfWeek n = DayOfWeek.Sunday :=
by sorry

end NUMINAMATH_CALUDE_savings_exceed_500_on_sunday_l1112_111264


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l1112_111212

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 5*a + 5 = 0) → (b^2 - 5*b + 5 = 0) → a^2 + b^2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l1112_111212


namespace NUMINAMATH_CALUDE_vector_equation_solution_l1112_111277

/-- Given vector a and an equation involving a and b, prove that b equals (1, -2) -/
theorem vector_equation_solution (a b : ℝ × ℝ) : 
  a = (1, 2) → 
  (2 • a) + b = (3, 2) → 
  b = (1, -2) := by
sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l1112_111277


namespace NUMINAMATH_CALUDE_even_function_negative_domain_l1112_111285

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_function_negative_domain
  (f : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_positive : ∀ x ≥ 0, f x = 2 * x + 1) :
  ∀ x < 0, f x = -2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_negative_domain_l1112_111285


namespace NUMINAMATH_CALUDE_function_equation_solution_l1112_111226

theorem function_equation_solution (f : ℚ → ℚ) 
  (h0 : f 0 = 0) 
  (h1 : ∀ x y : ℚ, f (f x + f y) = x + y) : 
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = -x) := by
sorry

end NUMINAMATH_CALUDE_function_equation_solution_l1112_111226


namespace NUMINAMATH_CALUDE_range_of_f_l1112_111241

def f (x : ℝ) : ℝ := (x - 2)^2 - 1

theorem range_of_f :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3,
  ∃ y ∈ Set.Icc (-1 : ℝ) 8,
  f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc (-1 : ℝ) 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1112_111241


namespace NUMINAMATH_CALUDE_max_NPM_value_l1112_111216

theorem max_NPM_value : 
  ∀ M : ℕ, 
  1 ≤ M ∧ M ≤ 9 →
  let MM := 10 * M + M
  let NPM := MM * M
  100 ≤ NPM ∧ NPM < 1000 →
  (∀ N P : ℕ, NPM = 100 * N + 10 * P + M → N < 10 ∧ P < 10) →
  NPM ≤ 891 :=
by sorry

end NUMINAMATH_CALUDE_max_NPM_value_l1112_111216


namespace NUMINAMATH_CALUDE_multiplicative_inverse_600_mod_3599_l1112_111227

theorem multiplicative_inverse_600_mod_3599 :
  ∃ (n : ℕ), n < 3599 ∧ (600 * n) % 3599 = 1 :=
by
  -- Define the right triangle
  let a : ℕ := 45
  let b : ℕ := 336
  let c : ℕ := 339
  
  -- Assert that a, b, c form a right triangle
  have right_triangle : a^2 + b^2 = c^2 := by sorry
  
  -- Define the multiplicative inverse
  let inverse : ℕ := 1200
  
  -- Prove that inverse is less than 3599
  have inverse_bound : inverse < 3599 := by sorry
  
  -- Prove that inverse is the multiplicative inverse of 600 modulo 3599
  have inverse_property : (600 * inverse) % 3599 = 1 := by sorry
  
  -- Combine the proofs
  exact ⟨inverse, inverse_bound, inverse_property⟩

#eval (600 * 1200) % 3599  -- Should output 1

end NUMINAMATH_CALUDE_multiplicative_inverse_600_mod_3599_l1112_111227


namespace NUMINAMATH_CALUDE_seven_zero_three_six_repeating_equals_fraction_l1112_111258

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : Int
  repeatingPart : Nat
  repeatingLength : Nat

/-- The value of 7.036̄ as a RepeatingDecimal -/
def seven_zero_three_six_repeating : RepeatingDecimal :=
  { integerPart := 7
  , repeatingPart := 36
  , repeatingLength := 3 }

/-- Converts a RepeatingDecimal to a rational number -/
def toRational (d : RepeatingDecimal) : Rat :=
  sorry

theorem seven_zero_three_six_repeating_equals_fraction :
  toRational seven_zero_three_six_repeating = 781 / 111 := by
  sorry

end NUMINAMATH_CALUDE_seven_zero_three_six_repeating_equals_fraction_l1112_111258


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l1112_111272

theorem price_reduction_percentage (initial_price final_price : ℝ) 
  (h1 : initial_price = 2000)
  (h2 : final_price = 1620)
  (h3 : initial_price > 0)
  (h4 : final_price > 0)
  (h5 : final_price < initial_price) :
  ∃ (x : ℝ), x > 0 ∧ x < 1 ∧ initial_price * (1 - x)^2 = final_price ∧ x = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l1112_111272


namespace NUMINAMATH_CALUDE_C₁_C₂_intersections_l1112_111236

/-- The polar coordinate equation of curve C₁ -/
def C₁_polar (ρ θ : ℝ) : Prop := ρ - 2 * Real.cos θ = 0

/-- The rectangular coordinate equation of curve C₁ -/
def C₁_rect (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

/-- The equation of curve C₂ -/
def C₂ (x y m : ℝ) : Prop := 2*x - y - 2*m - 1 = 0

/-- The condition for C₁ and C₂ to have two distinct intersection points -/
def has_two_intersections (m : ℝ) : Prop :=
  (1 - Real.sqrt 5) / 2 < m ∧ m < (1 + Real.sqrt 5) / 2

theorem C₁_C₂_intersections (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    C₁_rect x₁ y₁ ∧ C₁_rect x₂ y₂ ∧
    C₂ x₁ y₁ m ∧ C₂ x₂ y₂ m) ↔
  has_two_intersections m :=
sorry

end NUMINAMATH_CALUDE_C₁_C₂_intersections_l1112_111236


namespace NUMINAMATH_CALUDE_m_range_l1112_111279

def A : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1}

theorem m_range (m : ℝ) (h : A ∪ B m = A) : 1/2 ≤ m ∧ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l1112_111279


namespace NUMINAMATH_CALUDE_remainder_problem_l1112_111217

theorem remainder_problem (n : ℤ) (h : n % 18 = 10) : (2 * n) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1112_111217


namespace NUMINAMATH_CALUDE_masking_tape_wall_width_l1112_111257

theorem masking_tape_wall_width (total_tape : ℝ) (known_wall_width : ℝ) (known_wall_count : ℕ) (unknown_wall_count : ℕ) :
  total_tape = 20 →
  known_wall_width = 6 →
  known_wall_count = 2 →
  unknown_wall_count = 2 →
  (unknown_wall_count : ℝ) * (total_tape - known_wall_count * known_wall_width) / unknown_wall_count = 4 := by
sorry

end NUMINAMATH_CALUDE_masking_tape_wall_width_l1112_111257


namespace NUMINAMATH_CALUDE_find_defective_box_l1112_111295

/-- Represents the number of boxes -/
def num_boxes : ℕ := 9

/-- Represents the number of standard parts per box -/
def standard_parts_per_box : ℕ := 10

/-- Represents the number of defective parts in one box -/
def defective_parts : ℕ := 10

/-- Represents the weight of a standard part in grams -/
def standard_weight : ℕ := 100

/-- Represents the weight of a defective part in grams -/
def defective_weight : ℕ := 101

/-- Represents the total number of parts selected for weighing -/
def total_selected : ℕ := (num_boxes + 1) * num_boxes / 2

/-- Represents the expected weight if all selected parts were standard -/
def expected_weight : ℕ := total_selected * standard_weight

theorem find_defective_box (actual_weight : ℕ) :
  actual_weight > expected_weight →
  ∃ (box_number : ℕ), 
    box_number ≤ num_boxes ∧
    box_number = actual_weight - expected_weight ∧
    box_number * defective_parts = (defective_weight - standard_weight) * total_selected :=
by sorry

end NUMINAMATH_CALUDE_find_defective_box_l1112_111295


namespace NUMINAMATH_CALUDE_inequality_proof_l1112_111291

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2014 + b^2014 + c^2014 + a*b*c = 4) :
  (a^2013 + b^2013 - c)/c^2013 + (b^2013 + c^2013 - a)/a^2013 + (c^2013 + a^2013 - b)/b^2013 
  ≥ a^2012 + b^2012 + c^2012 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1112_111291


namespace NUMINAMATH_CALUDE_a_eq_3_sufficient_not_necessary_l1112_111283

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l₁ l₂ : Line2D) : Prop :=
  l₁.a * l₂.b = l₂.a * l₁.b

/-- The first line: ax - 5y - 1 = 0 -/
def l₁ (a : ℝ) : Line2D :=
  { a := a, b := -5, c := -1 }

/-- The second line: 3x - (a+2)y + 4 = 0 -/
def l₂ (a : ℝ) : Line2D :=
  { a := 3, b := -(a+2), c := 4 }

/-- The statement that a = 3 is a sufficient but not necessary condition for l₁ ∥ l₂ -/
theorem a_eq_3_sufficient_not_necessary :
  (∃ (a : ℝ), a ≠ 3 ∧ parallel (l₁ a) (l₂ a)) ∧
  (parallel (l₁ 3) (l₂ 3)) := by
  sorry

end NUMINAMATH_CALUDE_a_eq_3_sufficient_not_necessary_l1112_111283


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1112_111239

/-- Given two vectors a and b in ℝ², prove that if 2a + b is parallel to (1/2)a + kb, then k = 1/4. -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (2, 1)) 
    (h2 : b = (1, 2)) 
    (h_parallel : ∃ (t : ℝ), t • (2 • a + b) = (1/2 • a + k • b)) : 
  k = 1/4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1112_111239


namespace NUMINAMATH_CALUDE_marble_probability_difference_l1112_111237

/-- The number of red marbles in the box -/
def red : ℕ := 500

/-- The number of black marbles in the box -/
def black : ℕ := 700

/-- The number of blue marbles in the box -/
def blue : ℕ := 800

/-- The total number of marbles in the box -/
def total : ℕ := red + black + blue

/-- The probability of drawing two marbles of the same color -/
noncomputable def Ps : ℚ := 
  (red * (red - 1) + black * (black - 1) + blue * (blue - 1)) / (total * (total - 1))

/-- The probability of drawing two marbles of different colors -/
noncomputable def Pd : ℚ := 
  (red * black + red * blue + black * blue) * 2 / (total * (total - 1))

/-- Theorem stating that the absolute difference between Ps and Pd is 31/100 -/
theorem marble_probability_difference : |Ps - Pd| = 31 / 100 := by sorry

end NUMINAMATH_CALUDE_marble_probability_difference_l1112_111237


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1112_111256

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a + b + c = 3) : 
  ∀ x y z, x > 0 → y > 0 → z > 0 → x + y + z = 3 → 
  1 / (a + b) + 1 / c ≤ 1 / (x + y) + 1 / z :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1112_111256


namespace NUMINAMATH_CALUDE_combined_teaching_experience_l1112_111222

theorem combined_teaching_experience (james_experience partner_experience : ℕ) 
  (h1 : james_experience = 40)
  (h2 : partner_experience = james_experience - 10) :
  james_experience + partner_experience = 70 := by
sorry

end NUMINAMATH_CALUDE_combined_teaching_experience_l1112_111222


namespace NUMINAMATH_CALUDE_line_through_hyperbola_points_l1112_111244

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2/2 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := 2*x + 8*y + 7 = 0

/-- Theorem stating that the line passing through two points on the given hyperbola
    with midpoint (1/2, -1) has the equation 2x + 8y + 7 = 0 -/
theorem line_through_hyperbola_points :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola x₁ y₁ ∧
    hyperbola x₂ y₂ ∧
    (x₁ + x₂)/2 = 1/2 ∧
    (y₁ + y₂)/2 = -1 ∧
    (∀ (x y : ℝ), (x - x₁)*(y₂ - y₁) = (y - y₁)*(x₂ - x₁) ↔ line x y) :=
by sorry

end NUMINAMATH_CALUDE_line_through_hyperbola_points_l1112_111244


namespace NUMINAMATH_CALUDE_newspaper_conference_max_overlap_l1112_111281

theorem newspaper_conference_max_overlap (total : ℕ) (writers : ℕ) (editors : ℕ) 
  (h_total : total = 100)
  (h_writers : writers = 45)
  (h_editors : editors > 36)
  (x : ℕ)
  (h_both : x = writers + editors - total + (total - writers - editors) / 2) :
  x ≤ 18 ∧ ∃ (e : ℕ), e > 36 ∧ x = writers + e - total + (total - writers - e) / 2 ∧ x = 18 :=
by sorry

end NUMINAMATH_CALUDE_newspaper_conference_max_overlap_l1112_111281


namespace NUMINAMATH_CALUDE_square_side_length_l1112_111230

theorem square_side_length (area : ℝ) (side : ℝ) :
  area = 1 / 9 ∧ area = side ^ 2 → side = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1112_111230


namespace NUMINAMATH_CALUDE_smallest_sum_xyz_l1112_111234

theorem smallest_sum_xyz (x y z : ℕ+) 
  (eq1 : (x.val + y.val) * (y.val + z.val) = 2016)
  (eq2 : (x.val + y.val) * (z.val + x.val) = 1080) :
  (∀ a b c : ℕ+, 
    (a.val + b.val) * (b.val + c.val) = 2016 → 
    (a.val + b.val) * (c.val + a.val) = 1080 → 
    x.val + y.val + z.val ≤ a.val + b.val + c.val) ∧
  x.val + y.val + z.val = 61 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_xyz_l1112_111234


namespace NUMINAMATH_CALUDE_hyperbola_sum_l1112_111250

/-- Represents a hyperbola with center (h, k), focus (h + c, k), and vertex (h + a, k) --/
structure Hyperbola where
  h : ℝ
  k : ℝ
  a : ℝ
  c : ℝ

/-- The theorem that for a specific hyperbola, h + k + a + b = 7 --/
theorem hyperbola_sum (H : Hyperbola) (h_center : H.h = 1) (k_center : H.k = -3)
  (h_vertex : H.h + H.a = 4) (h_focus : H.h + H.c = 1 + 3 * Real.sqrt 5) :
  H.h + H.k + H.a + Real.sqrt (H.c^2 - H.a^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l1112_111250


namespace NUMINAMATH_CALUDE_percy_christmas_money_l1112_111225

/-- The amount of money Percy received at Christmas -/
def christmas_money : ℝ :=
  let playstation_cost : ℝ := 500
  let birthday_money : ℝ := 200
  let game_price : ℝ := 7.5
  let games_sold : ℕ := 20
  playstation_cost - birthday_money - (game_price * games_sold)

theorem percy_christmas_money :
  christmas_money = 150 := by
  sorry

end NUMINAMATH_CALUDE_percy_christmas_money_l1112_111225


namespace NUMINAMATH_CALUDE_parabola_focus_l1112_111290

/-- The parabola is defined by the equation x^2 = 20y -/
def parabola (x y : ℝ) : Prop := x^2 = 20 * y

/-- The focus of a parabola with equation x^2 = 4py has coordinates (0, p) -/
def is_focus (x y p : ℝ) : Prop := x = 0 ∧ y = p

/-- Theorem: The focus of the parabola x^2 = 20y has coordinates (0, 5) -/
theorem parabola_focus :
  ∃ (x y : ℝ), parabola x y ∧ is_focus x y 5 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l1112_111290


namespace NUMINAMATH_CALUDE_g_constant_value_l1112_111220

-- Define the function g
def g : ℝ → ℝ := fun x ↦ 5

-- Theorem statement
theorem g_constant_value (x : ℝ) : g (x + 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_g_constant_value_l1112_111220


namespace NUMINAMATH_CALUDE_paint_house_theorem_l1112_111223

/-- Represents the time taken to paint a house given the number of people -/
def paint_time (people : ℝ) (hours : ℝ) : Prop :=
  people * hours = 5 * 10

theorem paint_house_theorem :
  paint_time 5 10 → paint_time 4 12.5 :=
by
  sorry

end NUMINAMATH_CALUDE_paint_house_theorem_l1112_111223


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l1112_111243

theorem consecutive_numbers_sum (x : ℕ) :
  (∃ y : ℕ, 0 ≤ y ∧ y ≤ 9 ∧
    (List.sum (List.filter (λ i => i ≠ x + y) [x, x+1, x+2, x+3, x+4, x+5, x+6, x+7, x+8, x+9]) = 2002)) →
  x = 218 ∧ 
  List.filter (λ i => i ≠ 223) [x, x+1, x+2, x+3, x+4, x+5, x+6, x+7, x+8, x+9] = 
    [218, 219, 220, 221, 222, 224, 225, 226, 227] := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l1112_111243


namespace NUMINAMATH_CALUDE_election_votes_l1112_111235

theorem election_votes (votes_A : ℕ) (ratio_A ratio_B : ℕ) : 
  votes_A = 14 → ratio_A = 2 → ratio_B = 1 → 
  votes_A + (votes_A * ratio_B / ratio_A) = 21 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l1112_111235


namespace NUMINAMATH_CALUDE_unique_a_value_l1112_111255

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- State the theorem
theorem unique_a_value : ∃! a : ℝ, (A a ∩ B).Nonempty ∧ (A a ∩ C = ∅) ∧ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l1112_111255


namespace NUMINAMATH_CALUDE_jeremy_stroll_distance_l1112_111240

theorem jeremy_stroll_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 2 → time = 10 → distance = speed * time → distance = 20 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_stroll_distance_l1112_111240


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l1112_111269

theorem cubic_roots_sum_of_cubes (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) → 
  (b^3 - 2*b^2 + 3*b - 4 = 0) → 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  a^3 + b^3 + c^3 = 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l1112_111269


namespace NUMINAMATH_CALUDE_book_pages_and_reading_schedule_l1112_111270

-- Define the total number of pages in the book
variable (P : ℕ)

-- Define the number of pages read on the 4th day
variable (x : ℕ)

-- Theorem statement
theorem book_pages_and_reading_schedule :
  -- Conditions
  (2 / 3 : ℚ) * P = ((2 / 3 : ℚ) * P - (1 / 3 : ℚ) * P) + 90 ∧
  (1 / 3 : ℚ) * P = x + (x - 10) ∧
  x > 10 →
  -- Conclusions
  P = 270 ∧ x = 50 ∧ x - 10 = 40 := by
sorry

end NUMINAMATH_CALUDE_book_pages_and_reading_schedule_l1112_111270


namespace NUMINAMATH_CALUDE_min_value_expression_l1112_111265

/-- Given two positive real numbers m and n, and two vectors a and b that are perpendicular,
    prove that the minimum value of 1/m + 2/n is 3 + 2√2 -/
theorem min_value_expression (m n : ℝ) (a b : ℝ × ℝ) 
  (hm : m > 0) (hn : n > 0)
  (ha : a = (m, 1)) (hb : b = (1, n - 1))
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) :
  (∀ x y, x > 0 → y > 0 → 1/x + 2/y ≥ 1/m + 2/n) → 1/m + 2/n = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1112_111265


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1112_111247

-- Problem 1
theorem problem_1 : Real.sqrt 27 - 6 * Real.sqrt (1/3) + Real.sqrt ((-2)^2) = Real.sqrt 3 + 2 := by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x = 2 + Real.sqrt 3) (hy : y = 2 - Real.sqrt 3) :
  x^2 * y + x * y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1112_111247


namespace NUMINAMATH_CALUDE_expected_rainfall_is_19_25_l1112_111275

/-- Weather forecast for a single day -/
structure DailyForecast where
  prob_sun : ℝ
  prob_rain3 : ℝ
  prob_rain8 : ℝ
  sum_to_one : prob_sun + prob_rain3 + prob_rain8 = 1

/-- Calculate expected rainfall for a single day -/
def expected_daily_rainfall (f : DailyForecast) : ℝ :=
  f.prob_sun * 0 + f.prob_rain3 * 3 + f.prob_rain8 * 8

/-- Weather forecast for the week -/
def weekly_forecast : DailyForecast :=
  { prob_sun := 0.3
    prob_rain3 := 0.35
    prob_rain8 := 0.35
    sum_to_one := by norm_num }

/-- Number of days in the forecast -/
def num_days : ℕ := 5

/-- Expected total rainfall for the week -/
def expected_total_rainfall : ℝ :=
  (expected_daily_rainfall weekly_forecast) * num_days

theorem expected_rainfall_is_19_25 :
  expected_total_rainfall = 19.25 := by sorry

end NUMINAMATH_CALUDE_expected_rainfall_is_19_25_l1112_111275


namespace NUMINAMATH_CALUDE_program_output_l1112_111294

theorem program_output : ∃ i : ℕ, (∀ j < i, 2^j ≤ 2000) ∧ (2^i > 2000) ∧ (i - 1 = 10) := by
  sorry

end NUMINAMATH_CALUDE_program_output_l1112_111294


namespace NUMINAMATH_CALUDE_min_m_for_solution_non_monotonic_range_l1112_111208

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x^2 - 1| + x

-- Part I
theorem min_m_for_solution (a : ℝ) (h : a = 2) :
  (∃ m : ℝ, ∀ x : ℝ, f a x - m ≤ 0) ↔ m ≥ -Real.sqrt 2 / 2 :=
sorry

-- Part II
theorem non_monotonic_range (a : ℝ) :
  (∃ x y : ℝ, x ∈ Set.Icc (-3) 2 ∧ y ∈ Set.Icc (-3) 2 ∧ x < y ∧ f a x > f a y) ↔
  a < -1/6 ∨ a > 1/6 :=
sorry

end NUMINAMATH_CALUDE_min_m_for_solution_non_monotonic_range_l1112_111208


namespace NUMINAMATH_CALUDE_smallest_m_plus_n_l1112_111238

/-- The function f(x) = arcsin(log_m(nx)) has a domain that is a closed interval of length 1/1007 --/
def domain_length (m n : ℕ) : ℚ :=
  (m^2 - 1 : ℚ) / (m * n)

/-- The theorem stating the smallest possible value of m + n --/
theorem smallest_m_plus_n :
  ∃ (m n : ℕ),
    m > 1 ∧
    domain_length m n = 1/1007 ∧
    ∀ (m' n' : ℕ), m' > 1 → domain_length m' n' = 1/1007 → m + n ≤ m' + n' ∧
    m + n = 19099 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_plus_n_l1112_111238


namespace NUMINAMATH_CALUDE_solution_set_min_value_equality_condition_l1112_111214

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * abs (x - 1) + abs (x + 2)

-- Part 1: Solution set of f(x) ≤ 9
theorem solution_set (x : ℝ) : f x ≤ 9 ↔ x ∈ Set.Icc (-3) 3 := by sorry

-- Part 2: Minimum value of 4a² + b² + c²
theorem min_value (a b c : ℝ) (h : a + b + c = 3) :
  4 * a^2 + b^2 + c^2 ≥ 4 := by sorry

-- Equality condition
theorem equality_condition (a b c : ℝ) (h : a + b + c = 3) :
  4 * a^2 + b^2 + c^2 = 4 ↔ a = 1/3 ∧ b = 4/3 ∧ c = 4/3 := by sorry

end NUMINAMATH_CALUDE_solution_set_min_value_equality_condition_l1112_111214


namespace NUMINAMATH_CALUDE_carols_rectangle_length_l1112_111299

theorem carols_rectangle_length (carol_width jordan_length jordan_width : ℕ) 
  (h1 : carol_width = 15)
  (h2 : jordan_length = 6)
  (h3 : jordan_width = 30)
  (h4 : carol_width * carol_length = jordan_length * jordan_width) :
  carol_length = 12 := by
  sorry

#check carols_rectangle_length

end NUMINAMATH_CALUDE_carols_rectangle_length_l1112_111299


namespace NUMINAMATH_CALUDE_sum_of_powers_lower_bound_l1112_111260

theorem sum_of_powers_lower_bound 
  (x y z : ℝ) 
  (n : ℕ) 
  (pos_x : 0 < x) 
  (pos_y : 0 < y) 
  (pos_z : 0 < z) 
  (sum_one : x + y + z = 1) 
  (pos_n : 0 < n) : 
  x^n + y^n + z^n ≥ 1 / (3^(n-1)) := by
sorry

end NUMINAMATH_CALUDE_sum_of_powers_lower_bound_l1112_111260


namespace NUMINAMATH_CALUDE_trig_identity_l1112_111204

theorem trig_identity (α β : Real) 
  (h : (Real.cos α)^6 / (Real.cos β)^3 + (Real.sin α)^6 / (Real.sin β)^3 = 1) :
  (Real.sin β)^6 / (Real.sin α)^3 + (Real.cos β)^6 / (Real.cos α)^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1112_111204


namespace NUMINAMATH_CALUDE_bumper_car_line_count_l1112_111215

/-- Calculates the final number of people in line for bumper cars after several changes --/
def final_line_count (initial : ℕ) (left1 left2 left3 joined1 joined2 joined3 : ℕ) : ℕ :=
  initial - left1 + joined1 - left2 + joined2 - left3 + joined3

/-- Theorem stating the final number of people in line for the given scenario --/
theorem bumper_car_line_count : 
  final_line_count 31 15 8 7 12 18 25 = 56 := by
  sorry

end NUMINAMATH_CALUDE_bumper_car_line_count_l1112_111215
