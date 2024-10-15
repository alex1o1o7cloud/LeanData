import Mathlib

namespace NUMINAMATH_CALUDE_function_properties_l154_15406

def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

theorem function_properties (a b : ℝ) :
  (f a b 1 = 3) →
  ((3 * a + 2 * b) = 0) →
  (a = -6 ∧ b = 9) ∧
  (∀ x ∈ Set.Icc (-1) 2, f a b x ≤ 15) ∧
  (∀ x ∈ Set.Icc (-1) 2, f a b x ≥ -12) ∧
  (∃ x ∈ Set.Icc (-1) 2, f a b x = 15) ∧
  (∃ x ∈ Set.Icc (-1) 2, f a b x = -12) :=
by
  sorry

end NUMINAMATH_CALUDE_function_properties_l154_15406


namespace NUMINAMATH_CALUDE_two_possible_values_for_D_l154_15488

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def distinct_digits (A B C D E : ℕ) : Prop :=
  is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ is_digit E ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
  C ≠ D ∧ C ≠ E ∧
  D ≠ E

def addition_equation (A B C D E : ℕ) : Prop :=
  (A * 10000 + B * 1000 + C * 100 + D * 10 + B) +
  (B * 10000 + C * 1000 + A * 100 + D * 10 + E) =
  (E * 10000 + D * 1000 + D * 100 + E * 10 + E)

theorem two_possible_values_for_D :
  ∃ (D₁ D₂ : ℕ), D₁ ≠ D₂ ∧
  (∀ (A B C D E : ℕ), distinct_digits A B C D E → addition_equation A B C D E →
    D = D₁ ∨ D = D₂) ∧
  (∀ (A B C E : ℕ), ∃ (D : ℕ), distinct_digits A B C D E ∧ addition_equation A B C D E) :=
by sorry

end NUMINAMATH_CALUDE_two_possible_values_for_D_l154_15488


namespace NUMINAMATH_CALUDE_soft_taco_price_is_correct_l154_15433

/-- The price of a hard shell taco -/
def hard_shell_price : ℝ := 5

/-- The number of hard shell tacos bought by the family -/
def family_hard_shells : ℕ := 4

/-- The number of soft tacos bought by the family -/
def family_soft_shells : ℕ := 3

/-- The number of other customers -/
def other_customers : ℕ := 10

/-- The number of soft tacos bought by each other customer -/
def soft_tacos_per_customer : ℕ := 2

/-- The total revenue during lunch rush -/
def total_revenue : ℝ := 66

/-- The price of a soft taco -/
def soft_taco_price : ℝ := 2

theorem soft_taco_price_is_correct :
  soft_taco_price * (family_soft_shells + other_customers * soft_tacos_per_customer) +
  hard_shell_price * family_hard_shells = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_soft_taco_price_is_correct_l154_15433


namespace NUMINAMATH_CALUDE_no_valid_partition_l154_15447

-- Define a partition type
def Partition := ℤ → Fin 3

-- Define the property that n, n-50, and n+1987 belong to different subsets
def ValidPartition (p : Partition) : Prop :=
  ∀ n : ℤ, p n ≠ p (n - 50) ∧ p n ≠ p (n + 1987) ∧ p (n - 50) ≠ p (n + 1987)

-- Theorem statement
theorem no_valid_partition : ¬∃ p : Partition, ValidPartition p := by
  sorry

end NUMINAMATH_CALUDE_no_valid_partition_l154_15447


namespace NUMINAMATH_CALUDE_bottle_cap_distribution_l154_15455

theorem bottle_cap_distribution (initial : ℕ) (rebecca : ℕ) (siblings : ℕ) : 
  initial = 150 →
  rebecca = 42 →
  siblings = 5 →
  (initial + rebecca + 2 * rebecca) / (siblings + 1) = 46 :=
by sorry

end NUMINAMATH_CALUDE_bottle_cap_distribution_l154_15455


namespace NUMINAMATH_CALUDE_division_calculation_l154_15478

theorem division_calculation : (6 : ℚ) / (-1/2 + 1/3) = -36 := by sorry

end NUMINAMATH_CALUDE_division_calculation_l154_15478


namespace NUMINAMATH_CALUDE_unknown_number_value_l154_15483

theorem unknown_number_value (a x : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * x * 45 * 25) : x = 49 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_value_l154_15483


namespace NUMINAMATH_CALUDE_three_digit_number_relation_l154_15498

theorem three_digit_number_relation :
  ∀ a b c : ℕ,
  (100 ≤ 100 * a + 10 * b + c) ∧ (100 * a + 10 * b + c < 1000) →  -- three-digit number condition
  (100 * a + 10 * b + c = 56 * c) →                               -- 56 times last digit condition
  (100 * a + 10 * b + c = 112 * a) :=                             -- 112 times first digit (to prove)
by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_relation_l154_15498


namespace NUMINAMATH_CALUDE_constant_term_of_expansion_l154_15462

theorem constant_term_of_expansion (x : ℝ) (x_pos : x > 0) : 
  ∃ (c : ℝ), c = 240 ∧ 
  ∀ (k : ℕ), k ≤ 6 → 
    (Nat.choose 6 k * (2^k) * x^(6 - 3/2 * k : ℝ) = c ↔ k = 4) :=
sorry

end NUMINAMATH_CALUDE_constant_term_of_expansion_l154_15462


namespace NUMINAMATH_CALUDE_frog_eggs_theorem_l154_15408

/-- The number of eggs laid by a frog in a year -/
def eggs_laid : ℕ := 593

/-- The fraction of eggs that don't dry up -/
def not_dried_up : ℚ := 9/10

/-- The fraction of remaining eggs that are not eaten -/
def not_eaten : ℚ := 3/10

/-- The fraction of remaining eggs that hatch -/
def hatch_rate : ℚ := 1/4

/-- The number of frogs that hatch -/
def frogs_hatched : ℕ := 40

theorem frog_eggs_theorem :
  ↑frogs_hatched = ⌈(↑eggs_laid * not_dried_up * not_eaten * hatch_rate)⌉ := by sorry

end NUMINAMATH_CALUDE_frog_eggs_theorem_l154_15408


namespace NUMINAMATH_CALUDE_moving_circle_center_trajectory_l154_15404

/-- A moving circle that passes through (1, 0) and is tangent to x = -1 -/
structure MovingCircle where
  center : ℝ × ℝ
  passes_through_one_zero : (center.1 - 1)^2 + center.2^2 = (center.1 + 1)^2
  tangent_to_neg_one : True  -- This condition is implied by the equation above

/-- The trajectory of the center of the moving circle is y² = 4x -/
theorem moving_circle_center_trajectory (M : MovingCircle) : 
  M.center.2^2 = 4 * M.center.1 := by
  sorry

end NUMINAMATH_CALUDE_moving_circle_center_trajectory_l154_15404


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l154_15451

theorem binomial_coefficient_ratio : ∀ a₀ a₁ a₂ a₃ a₄ a₅ : ℤ,
  (∀ x : ℤ, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) / (a₁ + a₃) = -61 / 60 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l154_15451


namespace NUMINAMATH_CALUDE_smallest_tax_price_integer_l154_15437

theorem smallest_tax_price_integer (n : ℕ) : n = 21 ↔ 
  (n > 0 ∧ ∀ m : ℕ, m > 0 → m < n → 
    ¬∃ x : ℕ, (105 * x : ℚ) / 100 = m) ∧
  ∃ x : ℕ, (105 * x : ℚ) / 100 = n :=
by sorry

end NUMINAMATH_CALUDE_smallest_tax_price_integer_l154_15437


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_51200_l154_15422

/-- The number of factors of 51200 that are perfect squares -/
def perfect_square_factors_of_51200 : ℕ :=
  (Finset.range 6).card * (Finset.range 2).card

/-- Theorem stating that the number of factors of 51200 that are perfect squares is 12 -/
theorem count_perfect_square_factors_51200 :
  perfect_square_factors_of_51200 = 12 := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_51200_l154_15422


namespace NUMINAMATH_CALUDE_transfer_equation_l154_15493

def location_A : ℕ := 232
def location_B : ℕ := 146

theorem transfer_equation (x : ℤ) : 
  (location_A : ℤ) + x = 3 * ((location_B : ℤ) - x) ↔ 
  (location_A : ℤ) + x = 3 * ((location_B : ℤ) - x) :=
by sorry

end NUMINAMATH_CALUDE_transfer_equation_l154_15493


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l154_15419

theorem fraction_equation_solution : 
  ∃ x : ℚ, (3 / (x - 2) + 5 / (x + 2) = 8 / (x^2 - 4)) ∧ (x = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l154_15419


namespace NUMINAMATH_CALUDE_johns_total_spending_johns_spending_proof_l154_15413

/-- Calculate John's total spending on a phone and accessories, including sales tax -/
theorem johns_total_spending (online_price : ℝ) (price_increase_rate : ℝ) 
  (accessory_discount_rate : ℝ) (case_price : ℝ) (protector_price : ℝ) 
  (sales_tax_rate : ℝ) : ℝ :=
  let store_phone_price := online_price * (1 + price_increase_rate)
  let accessories_regular_price := case_price + protector_price
  let accessories_discounted_price := accessories_regular_price * (1 - accessory_discount_rate)
  let subtotal := store_phone_price + accessories_discounted_price
  let total_with_tax := subtotal * (1 + sales_tax_rate)
  total_with_tax

/-- Proof that John's total spending is $2212.75 -/
theorem johns_spending_proof : 
  johns_total_spending 2000 0.02 0.05 35 15 0.06 = 2212.75 := by
  sorry

end NUMINAMATH_CALUDE_johns_total_spending_johns_spending_proof_l154_15413


namespace NUMINAMATH_CALUDE_whitewashing_cost_is_7580_l154_15449

/-- Calculates the cost of white washing a trapezoidal room with given dimensions and conditions -/
def whitewashingCost (length width height1 height2 : ℕ) (doorCount windowCount : ℕ) 
  (doorLength doorWidth windowLength windowWidth : ℕ) (decorationArea : ℕ) (ratePerSqFt : ℕ) : ℕ :=
  let totalWallArea := 2 * (length * height1 + width * height2)
  let doorArea := doorCount * doorLength * doorWidth
  let windowArea := windowCount * windowLength * windowWidth
  let adjustedArea := totalWallArea - doorArea - windowArea - decorationArea
  adjustedArea * ratePerSqFt

/-- Theorem stating that the cost of white washing the given trapezoidal room is 7580 -/
theorem whitewashing_cost_is_7580 : 
  whitewashingCost 25 15 12 8 2 3 6 3 4 3 10 10 = 7580 := by
  sorry

end NUMINAMATH_CALUDE_whitewashing_cost_is_7580_l154_15449


namespace NUMINAMATH_CALUDE_new_cube_volume_l154_15475

theorem new_cube_volume (original_volume : ℝ) (scale_factor : ℝ) : 
  original_volume = 64 →
  scale_factor = 2 →
  (scale_factor ^ 3) * original_volume = 512 :=
by sorry

end NUMINAMATH_CALUDE_new_cube_volume_l154_15475


namespace NUMINAMATH_CALUDE_clinic_patient_count_l154_15474

theorem clinic_patient_count (original_count current_count diagnosed_count : ℕ) : 
  current_count = 2 * original_count →
  diagnosed_count = 13 →
  (4 : ℕ) * diagnosed_count = current_count →
  original_count = 26 := by
  sorry

end NUMINAMATH_CALUDE_clinic_patient_count_l154_15474


namespace NUMINAMATH_CALUDE_sprinkles_problem_l154_15414

theorem sprinkles_problem (initial_cans : ℕ) : 
  (initial_cans / 2 - 3 = 3) → initial_cans = 12 := by
  sorry

end NUMINAMATH_CALUDE_sprinkles_problem_l154_15414


namespace NUMINAMATH_CALUDE_total_legs_count_l154_15486

theorem total_legs_count (total_tables : ℕ) (four_leg_tables : ℕ) : 
  total_tables = 36 → four_leg_tables = 16 → 
  (∃ (three_leg_tables : ℕ), 
    three_leg_tables + four_leg_tables = total_tables ∧
    3 * three_leg_tables + 4 * four_leg_tables = 124) := by
  sorry

end NUMINAMATH_CALUDE_total_legs_count_l154_15486


namespace NUMINAMATH_CALUDE_pauls_remaining_crayons_l154_15446

/-- The number of crayons Paul had initially -/
def initial_crayons : ℕ := 479

/-- The number of crayons Paul lost or gave away -/
def lost_crayons : ℕ := 345

/-- The number of crayons Paul had left -/
def remaining_crayons : ℕ := initial_crayons - lost_crayons

theorem pauls_remaining_crayons : remaining_crayons = 134 := by
  sorry

end NUMINAMATH_CALUDE_pauls_remaining_crayons_l154_15446


namespace NUMINAMATH_CALUDE_pizza_not_crust_percentage_l154_15491

def pizza_weight : ℝ := 800
def crust_weight : ℝ := 200

theorem pizza_not_crust_percentage :
  (pizza_weight - crust_weight) / pizza_weight * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_pizza_not_crust_percentage_l154_15491


namespace NUMINAMATH_CALUDE_heating_plant_consumption_l154_15484

/-- Represents the fuel consumption of a heating plant -/
structure HeatingPlant where
  consumption_rate : ℝ  -- Liters per hour

/-- Given a heating plant that consumes 7 liters of fuel in 21 hours,
    prove that it will consume 30 liters of fuel in 90 hours -/
theorem heating_plant_consumption 
  (plant : HeatingPlant) 
  (h1 : plant.consumption_rate * 21 = 7) :
  plant.consumption_rate * 90 = 30 := by
  sorry

end NUMINAMATH_CALUDE_heating_plant_consumption_l154_15484


namespace NUMINAMATH_CALUDE_simplify_square_roots_l154_15467

theorem simplify_square_roots : 
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 98 / Real.sqrt 49) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l154_15467


namespace NUMINAMATH_CALUDE_purchase_percentage_l154_15485

/-- Given a 25% price increase and a net difference in expenditure of 20,
    prove that the percentage of the required amount purchased is 16%. -/
theorem purchase_percentage (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  let new_price := 1.25 * P
  let R := (500 : ℝ) / 31.25
  let new_expenditure := new_price * (R / 100) * Q
  P * Q - new_expenditure = 20 → R = 16 := by sorry

end NUMINAMATH_CALUDE_purchase_percentage_l154_15485


namespace NUMINAMATH_CALUDE_tan_alpha_value_l154_15427

theorem tan_alpha_value (α : ℝ) 
  (h : (2 * Real.sin α + 3 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 1/4) : 
  Real.tan α = -2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l154_15427


namespace NUMINAMATH_CALUDE_square_sequence_theorem_l154_15430

/-- The number of squares in figure n -/
def f (n : ℕ) : ℕ := 4 * n^2 + 1

/-- Theorem stating the properties of the sequence and the value for figure 100 -/
theorem square_sequence_theorem :
  (f 0 = 1) ∧
  (f 1 = 5) ∧
  (f 2 = 17) ∧
  (f 3 = 37) ∧
  (f 100 = 40001) := by
  sorry

end NUMINAMATH_CALUDE_square_sequence_theorem_l154_15430


namespace NUMINAMATH_CALUDE_oranges_packed_in_week_l154_15477

/-- The number of oranges packed in a full week given the daily packing rate and box capacity -/
theorem oranges_packed_in_week
  (oranges_per_box : ℕ)
  (boxes_per_day : ℕ)
  (days_in_week : ℕ)
  (h1 : oranges_per_box = 15)
  (h2 : boxes_per_day = 2150)
  (h3 : days_in_week = 7) :
  oranges_per_box * boxes_per_day * days_in_week = 225750 := by
  sorry

end NUMINAMATH_CALUDE_oranges_packed_in_week_l154_15477


namespace NUMINAMATH_CALUDE_evaluate_power_l154_15441

theorem evaluate_power (x : ℝ) (h : x = 81) : x^(5/4) = 243 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_power_l154_15441


namespace NUMINAMATH_CALUDE_same_volume_prisms_l154_15438

def edge_lengths : List ℕ := [12, 18, 20, 24, 30, 33, 70, 24, 154]

def is_valid_prism (a b c : ℕ) : Bool :=
  a ∈ edge_lengths ∧ b ∈ edge_lengths ∧ c ∈ edge_lengths

def prism_volume (a b c : ℕ) : ℕ := a * b * c

theorem same_volume_prisms :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ : ℕ),
    is_valid_prism a₁ b₁ c₁ ∧
    is_valid_prism a₂ b₂ c₂ ∧
    is_valid_prism a₃ b₃ c₃ ∧
    prism_volume a₁ b₁ c₁ = prism_volume a₂ b₂ c₂ ∧
    prism_volume a₂ b₂ c₂ = prism_volume a₃ b₃ c₃ ∧
    (a₁, b₁, c₁) ≠ (a₂, b₂, c₂) ∧
    (a₂, b₂, c₂) ≠ (a₃, b₃, c₃) ∧
    (a₁, b₁, c₁) ≠ (a₃, b₃, c₃) :=
by
  sorry

#check same_volume_prisms

end NUMINAMATH_CALUDE_same_volume_prisms_l154_15438


namespace NUMINAMATH_CALUDE_unique_cubic_zero_a_range_l154_15428

/-- A cubic function with a unique positive zero point -/
structure UniqueCubicZero where
  a : ℝ
  f : ℝ → ℝ := λ x ↦ a * x^3 - 3 * x^2 + 1
  x₀ : ℝ
  x₀_pos : x₀ > 0
  x₀_zero : f x₀ = 0
  unique_zero : ∀ x, f x = 0 → x = x₀

/-- The range of 'a' for a cubic function with a unique positive zero point -/
theorem unique_cubic_zero_a_range (c : UniqueCubicZero) : c.a < -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_cubic_zero_a_range_l154_15428


namespace NUMINAMATH_CALUDE_average_speed_two_part_journey_l154_15464

theorem average_speed_two_part_journey 
  (total_distance : ℝ) 
  (first_part_ratio : ℝ) 
  (first_part_speed : ℝ) 
  (second_part_speed : ℝ) 
  (h1 : first_part_ratio = 0.35) 
  (h2 : first_part_speed = 35) 
  (h3 : second_part_speed = 65) 
  (h4 : first_part_ratio > 0 ∧ first_part_ratio < 1) :
  let second_part_ratio := 1 - first_part_ratio
  let first_part_time := (first_part_ratio * total_distance) / first_part_speed
  let second_part_time := (second_part_ratio * total_distance) / second_part_speed
  let total_time := first_part_time + second_part_time
  let average_speed := total_distance / total_time
  average_speed = 50 := by sorry

end NUMINAMATH_CALUDE_average_speed_two_part_journey_l154_15464


namespace NUMINAMATH_CALUDE_student_sums_l154_15415

theorem student_sums (total : ℕ) (right : ℕ) (wrong : ℕ) : 
  total = 48 → 
  wrong = 3 * right → 
  total = right + wrong → 
  wrong = 36 := by sorry

end NUMINAMATH_CALUDE_student_sums_l154_15415


namespace NUMINAMATH_CALUDE_range_of_2a_minus_b_l154_15416

theorem range_of_2a_minus_b (a b : ℝ) (ha : -1 ≤ a ∧ a ≤ 3) (hb : 2 ≤ b ∧ b ≤ 4) :
  (∀ x, 2 * a - b ≤ x → x ≤ 4) ∧ (∀ y, -6 ≤ y → y ≤ 2 * a - b) :=
by sorry

end NUMINAMATH_CALUDE_range_of_2a_minus_b_l154_15416


namespace NUMINAMATH_CALUDE_rectangular_to_polar_y_equals_x_l154_15497

theorem rectangular_to_polar_y_equals_x :
  ∀ (x y ρ : ℝ) (θ : ℝ),
  (y = x) ↔ (θ = π / 4 ∧ ρ > 0) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_y_equals_x_l154_15497


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l154_15403

-- Define the propositions p and q
def p (x : ℝ) : Prop := x = 2
def q (x : ℝ) : Prop := 0 < x ∧ x < 3

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l154_15403


namespace NUMINAMATH_CALUDE_lab_items_per_tech_l154_15431

/-- Given the number of uniforms in a lab, calculate the total number of coats and uniforms per lab tech. -/
def total_per_lab_tech (num_uniforms : ℕ) : ℕ :=
  let num_coats := 6 * num_uniforms
  let total_items := num_coats + num_uniforms
  let num_lab_techs := num_uniforms / 2
  total_items / num_lab_techs

/-- Theorem stating that given 12 uniforms, each lab tech gets 14 coats and uniforms in total. -/
theorem lab_items_per_tech :
  total_per_lab_tech 12 = 14 := by
  sorry

#eval total_per_lab_tech 12

end NUMINAMATH_CALUDE_lab_items_per_tech_l154_15431


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_l154_15432

def repeating_decimal : ℚ := 7 + 17 / 99

theorem repeating_decimal_fraction :
  repeating_decimal = 710 / 99 ∧
  (Nat.gcd 710 99 = 1) ∧
  (710 + 99 = 809) := by
  sorry

#eval repeating_decimal

end NUMINAMATH_CALUDE_repeating_decimal_fraction_l154_15432


namespace NUMINAMATH_CALUDE_part1_part2_l154_15499

-- Define the quadratic function
def f (b c x : ℝ) : ℝ := x^2 + 2*b*x + c

-- Part 1
theorem part1 (b c : ℝ) : 
  (∀ x, f b c x = 0 ↔ x = -1 ∨ x = 1) → b = 0 ∧ c = -1 := by sorry

-- Part 2
theorem part2 (b : ℝ) :
  (∃ x₁ x₂, f b (b^2 + 2*b + 3) x₁ = 0 ∧ f b (b^2 + 2*b + 3) x₂ = 0 ∧ (x₁ + 1)*(x₂ + 1) = 8) →
  b = -2 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l154_15499


namespace NUMINAMATH_CALUDE_problem_statement_l154_15425

open Real

noncomputable def f (x : ℝ) := log x
noncomputable def g (a : ℝ) (x : ℝ) := a * (x - 1) / (x + 1)
noncomputable def h (a : ℝ) (x : ℝ) := f x - g a x

theorem problem_statement :
  (∀ x > 1, f x > g 2 x) ∧
  (∀ a ≤ 2, StrictMono (h a)) ∧
  (∀ a > 2, ∃ x y, x < y ∧ IsLocalMax (h a) x ∧ IsLocalMin (h a) y) ∧
  (∀ x > 0, f (x + 1) > x^2 / (exp x - 1)) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l154_15425


namespace NUMINAMATH_CALUDE_sum_of_integers_l154_15407

theorem sum_of_integers (x y z w : ℤ) 
  (eq1 : x - y + z = 10)
  (eq2 : y - z + w = 15)
  (eq3 : z - w + x = 9)
  (eq4 : w - x + y = 4) :
  x + y + z + w = 38 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l154_15407


namespace NUMINAMATH_CALUDE_rectangleA_max_sum_l154_15401

-- Define a structure for rectangles
structure Rectangle where
  w : Int
  x : Int
  y : Int
  z : Int

-- Define the five rectangles
def rectangleA : Rectangle := ⟨8, 2, 9, 5⟩
def rectangleB : Rectangle := ⟨2, 1, 5, 8⟩
def rectangleC : Rectangle := ⟨6, 9, 4, 3⟩
def rectangleD : Rectangle := ⟨4, 6, 2, 9⟩
def rectangleE : Rectangle := ⟨9, 5, 6, 1⟩

-- Define a list of all rectangles
def rectangles : List Rectangle := [rectangleA, rectangleB, rectangleC, rectangleD, rectangleE]

-- Define a function to calculate the sum of w and y
def sumWY (r : Rectangle) : Int := r.w + r.y

-- Theorem: Rectangle A has the maximum sum of w and y
theorem rectangleA_max_sum :
  ∀ r ∈ rectangles, sumWY rectangleA ≥ sumWY r := by
  sorry

end NUMINAMATH_CALUDE_rectangleA_max_sum_l154_15401


namespace NUMINAMATH_CALUDE_units_digit_of_2_pow_20_minus_1_l154_15443

theorem units_digit_of_2_pow_20_minus_1 : 
  (2^20 - 1) % 10 = 5 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_2_pow_20_minus_1_l154_15443


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l154_15423

theorem arithmetic_sequence_first_term (a : ℕ → ℤ) (d : ℤ) :
  (∀ n, a n = a 0 + n * d) →  -- arithmetic sequence definition
  (a 19 = 205) →             -- given condition a_20 = 205 (index starts at 0)
  (a 0 = 91) :=              -- prove a_1 = 91 (a_1 is a 0 in 0-indexed notation)
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l154_15423


namespace NUMINAMATH_CALUDE_zeros_arithmetic_sequence_implies_a_value_l154_15429

/-- A cubic polynomial function -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + x + a

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 + 4*x + 1

/-- Theorem: If the zeros of f form an arithmetic sequence, then a = -23/54 -/
theorem zeros_arithmetic_sequence_implies_a_value (a : ℝ) : 
  (∃ r s t : ℝ, (f a r = 0 ∧ f a s = 0 ∧ f a t = 0) ∧ 
   (s - r = t - s) ∧ (r < s ∧ s < t)) → 
  a = -23/54 := by
  sorry

end NUMINAMATH_CALUDE_zeros_arithmetic_sequence_implies_a_value_l154_15429


namespace NUMINAMATH_CALUDE_inverse_of_A_l154_15405

-- Define matrix A
def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, 0; 1, 8]

-- Define the proposed inverse of A
def A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![1/2, 0; -1/16, 1/8]

-- Theorem statement
theorem inverse_of_A : A⁻¹ = A_inv := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l154_15405


namespace NUMINAMATH_CALUDE_investment_total_l154_15459

theorem investment_total (rate1 rate2 amount1 amount2 total_income : ℚ)
  (h1 : rate1 = 85 / 1000)
  (h2 : rate2 = 64 / 1000)
  (h3 : amount1 = 3000)
  (h4 : amount2 = 5000)
  (h5 : rate1 * amount1 + rate2 * amount2 = 575) :
  amount1 + amount2 = 8000 :=
by sorry

end NUMINAMATH_CALUDE_investment_total_l154_15459


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_over_i_plus_one_l154_15445

theorem imaginary_part_of_i_over_i_plus_one :
  Complex.im (Complex.I / (Complex.I + 1)) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_over_i_plus_one_l154_15445


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l154_15487

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation x^2 - 7x + 4 = 0 -/
def a : ℝ := 1
def b : ℝ := -7
def c : ℝ := 4

theorem quadratic_discriminant : discriminant a b c = 33 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l154_15487


namespace NUMINAMATH_CALUDE_min_value_expression_l154_15454

theorem min_value_expression (x : ℝ) (hx : x > 0) :
  (4 + x) * (1 + x) / x ≥ 9 ∧ ∃ y > 0, (4 + y) * (1 + y) / y = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l154_15454


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_sequence_l154_15417

theorem arithmetic_and_geometric_sequence (a : ℕ → ℝ) : 
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence
  (∃ q : ℝ, q = 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n) := by
sorry


end NUMINAMATH_CALUDE_arithmetic_and_geometric_sequence_l154_15417


namespace NUMINAMATH_CALUDE_arithmetic_mean_set_not_full_segment_l154_15468

open Set

/-- The set of points generated by repeatedly inserting 9 arithmetic means
    between consecutive points on the segment [a, a+1] -/
def arithmeticMeanSet (a : ℕ) : Set ℝ :=
  { x | ∃ (n : ℕ) (m : ℕ), x = a + m / (10 ^ n) ∧ m < 10 ^ n }

/-- The theorem stating that the set of points generated by repeatedly inserting
    9 arithmetic means is not equal to the entire segment [a, a+1] -/
theorem arithmetic_mean_set_not_full_segment (a : ℕ) :
  ∃ x, x ∈ Icc (a : ℝ) (a + 1) ∧ x ∉ arithmeticMeanSet a :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_set_not_full_segment_l154_15468


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l154_15471

theorem complex_fraction_equality : (2 - I) / (1 - I) = 3/2 + I/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l154_15471


namespace NUMINAMATH_CALUDE_division_theorem_l154_15436

theorem division_theorem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ)
  (h1 : dividend = 132)
  (h2 : divisor = 16)
  (h3 : remainder = 4)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_theorem_l154_15436


namespace NUMINAMATH_CALUDE_sin_squared_3x_maximum_l154_15472

open Real

theorem sin_squared_3x_maximum (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin (3 * x) ^ 2) :
  ∃ x, x ∈ Set.Ioo 0 0.6 ∧ f x = 1 ∧ ∀ y ∈ Set.Ioo 0 0.6, f y ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_sin_squared_3x_maximum_l154_15472


namespace NUMINAMATH_CALUDE_fraction_subtraction_l154_15489

theorem fraction_subtraction (x : ℝ) (h : x ≠ 1) :
  x / (x - 1) - 1 / (x - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l154_15489


namespace NUMINAMATH_CALUDE_divisors_of_210_l154_15418

theorem divisors_of_210 : Finset.card (Nat.divisors 210) = 16 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_210_l154_15418


namespace NUMINAMATH_CALUDE_vector_problem_l154_15490

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v = t • w

theorem vector_problem :
  (∃ k : ℝ, perpendicular (k • a + b) (a - 3 • b) ∧ k = 19) ∧
  (∃ k : ℝ, parallel (k • a + b) (a - 3 • b) ∧ k = -1/3) :=
sorry

end NUMINAMATH_CALUDE_vector_problem_l154_15490


namespace NUMINAMATH_CALUDE_fruit_arrangement_l154_15448

theorem fruit_arrangement (n a o b p : ℕ) 
  (total : n = a + o + b + p)
  (apple : a = 4)
  (orange : o = 2)
  (banana : b = 2)
  (pear : p = 1) :
  Nat.factorial n / (Nat.factorial a * Nat.factorial o * Nat.factorial b * Nat.factorial p) = 3780 := by
  sorry

end NUMINAMATH_CALUDE_fruit_arrangement_l154_15448


namespace NUMINAMATH_CALUDE_smallest_y_for_cube_l154_15426

theorem smallest_y_for_cube (y : ℕ+) (M : ℤ) : 
  (∀ k : ℕ+, k < y → ¬∃ N : ℤ, 2520 * k = N^3) → 
  (∃ N : ℤ, 2520 * y = N^3) → 
  y = 3675 := by
sorry

end NUMINAMATH_CALUDE_smallest_y_for_cube_l154_15426


namespace NUMINAMATH_CALUDE_potatoes_per_bag_l154_15458

/-- Proves that the number of pounds of potatoes in one bag is 20 -/
theorem potatoes_per_bag (potatoes_per_person : ℝ) (num_people : ℕ) (cost_per_bag : ℝ) (total_cost : ℝ) :
  potatoes_per_person = 1.5 →
  num_people = 40 →
  cost_per_bag = 5 →
  total_cost = 15 →
  (num_people * potatoes_per_person) / (total_cost / cost_per_bag) = 20 := by
  sorry

end NUMINAMATH_CALUDE_potatoes_per_bag_l154_15458


namespace NUMINAMATH_CALUDE_hockey_league_season_games_l154_15409

/-- The number of games played in a hockey league season -/
def hockey_league_games (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) / 2 * m

/-- Theorem: In a hockey league with 16 teams, where each team faces all other teams 10 times,
    the total number of games played in the season is 1200. -/
theorem hockey_league_season_games :
  hockey_league_games 16 10 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_season_games_l154_15409


namespace NUMINAMATH_CALUDE_holds_age_ratio_l154_15476

/-- Proves that the ratio of Hold's age to her son's age today is 3:1 -/
theorem holds_age_ratio : 
  ∀ (hold_age_today hold_age_8_years_ago son_age_today son_age_8_years_ago : ℕ),
  hold_age_today = 36 →
  hold_age_8_years_ago = hold_age_today - 8 →
  son_age_8_years_ago = son_age_today - 8 →
  hold_age_8_years_ago = 7 * son_age_8_years_ago →
  (hold_age_today : ℚ) / son_age_today = 3 := by
sorry

end NUMINAMATH_CALUDE_holds_age_ratio_l154_15476


namespace NUMINAMATH_CALUDE_quadratic_maximum_value_l154_15457

/-- A quadratic function f(x) = ax² + bx + c -/
def QuadraticFunction (a b c : ℝ) := fun (x : ℝ) ↦ a * x^2 + b * x + c

/-- The derivative of a quadratic function -/
def QuadraticDerivative (a b : ℝ) := fun (x : ℝ) ↦ 2 * a * x + b

theorem quadratic_maximum_value (a b c : ℝ) :
  (∀ x : ℝ, QuadraticFunction a b c x ≥ QuadraticDerivative a b x) →
  (∃ M : ℝ, M = 2 * Real.sqrt 2 - 2 ∧
    (∀ k : ℝ, k ≤ M ↔ ∃ a' b' c' : ℝ, 
      (∀ x : ℝ, QuadraticFunction a' b' c' x ≥ QuadraticDerivative a' b' x) ∧
      k = b'^2 / (a'^2 + c'^2))) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_maximum_value_l154_15457


namespace NUMINAMATH_CALUDE_yankees_to_mets_ratio_l154_15481

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The given conditions of the problem -/
def baseball_town_conditions (fans : FanCounts) : Prop :=
  fans.yankees + fans.mets + fans.red_sox = 360 ∧
  fans.mets = 96 ∧
  5 * fans.mets = 4 * fans.red_sox

/-- The theorem to be proved -/
theorem yankees_to_mets_ratio (fans : FanCounts) 
  (h : baseball_town_conditions fans) : 
  3 * fans.mets = 2 * fans.yankees :=
sorry

end NUMINAMATH_CALUDE_yankees_to_mets_ratio_l154_15481


namespace NUMINAMATH_CALUDE_batsman_average_after_15th_inning_l154_15480

def batsman_average (total_innings : ℕ) (last_inning_score : ℕ) (average_increase : ℕ) : ℚ :=
  let previous_average := (total_innings - 1 : ℚ) * (average_increase : ℚ) + (last_inning_score : ℚ) / (total_innings : ℚ)
  previous_average + average_increase

theorem batsman_average_after_15th_inning :
  batsman_average 15 75 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_15th_inning_l154_15480


namespace NUMINAMATH_CALUDE_road_trip_total_hours_l154_15424

/-- Calculates the total hours driven on a road trip -/
def total_hours_driven (days : ℕ) (jade_hours_per_day : ℕ) (krista_hours_per_day : ℕ) : ℕ :=
  days * (jade_hours_per_day + krista_hours_per_day)

/-- Proves that the total hours driven by Jade and Krista over 3 days equals 42 hours -/
theorem road_trip_total_hours : total_hours_driven 3 8 6 = 42 := by
  sorry

#eval total_hours_driven 3 8 6

end NUMINAMATH_CALUDE_road_trip_total_hours_l154_15424


namespace NUMINAMATH_CALUDE_johns_age_l154_15494

theorem johns_age (age : ℕ) : 
  (age + 9 = 3 * (age - 11)) → age = 21 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l154_15494


namespace NUMINAMATH_CALUDE_work_completion_time_l154_15434

/-- Given workers A, B, and C, where A can complete a job in 6 days,
    B can complete it in 5 days, and together they complete it in 2 days with C's help,
    prove that C alone can complete the job in 7.5 days. -/
theorem work_completion_time (a b c : ℝ) 
  (ha : a = 6) 
  (hb : b = 5) 
  (hab : 1 / a + 1 / b + 1 / c = 1 / 2) : 
  c = 15 / 2 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l154_15434


namespace NUMINAMATH_CALUDE_ribbon_length_proof_l154_15435

theorem ribbon_length_proof (R : ℝ) : 
  (R / 2 + 2000 = R - ((R / 2 - 2000) / 2 + 2000)) → R = 12000 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_length_proof_l154_15435


namespace NUMINAMATH_CALUDE_not_unique_perpendicular_l154_15400

/-- A line in a plane --/
structure Line where
  -- We don't need to define the internals of a line for this statement
  mk :: 

/-- A plane --/
structure Plane where
  -- We don't need to define the internals of a plane for this statement
  mk ::

/-- Perpendicularity relation between two lines --/
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

/-- The statement to be proven false --/
def unique_perpendicular (p : Plane) : Prop :=
  ∃! (l : Line), ∀ (m : Line), perpendicular l m

/-- The theorem stating that the unique perpendicular line statement is false --/
theorem not_unique_perpendicular :
  ∃ (p : Plane), ¬(unique_perpendicular p) :=
sorry

end NUMINAMATH_CALUDE_not_unique_perpendicular_l154_15400


namespace NUMINAMATH_CALUDE_completing_square_result_l154_15456

theorem completing_square_result (x : ℝ) :
  (x^2 - 6*x + 5 = 0) ↔ ((x - 3)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_result_l154_15456


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l154_15411

/-- Given a boat that travels 6 km/hr along a stream and 2 km/hr against the same stream,
    its speed in still water is 4 km/hr. -/
theorem boat_speed_in_still_water (b s : ℝ) 
    (h1 : b + s = 6)  -- Speed along the stream
    (h2 : b - s = 2)  -- Speed against the stream
    : b = 4 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l154_15411


namespace NUMINAMATH_CALUDE_modular_inverse_of_two_mod_187_l154_15453

theorem modular_inverse_of_two_mod_187 : ∃ x : ℤ, 0 ≤ x ∧ x < 187 ∧ (2 * x) % 187 = 1 :=
by
  use 94
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_two_mod_187_l154_15453


namespace NUMINAMATH_CALUDE_xyz_value_l154_15410

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) : 
  x * y * z = 5 := by sorry

end NUMINAMATH_CALUDE_xyz_value_l154_15410


namespace NUMINAMATH_CALUDE_paint_usage_fraction_l154_15452

theorem paint_usage_fraction (total_paint : ℚ) (total_used : ℚ) : 
  total_paint = 360 →
  total_used = 264 →
  let first_week_fraction := (5 : ℚ) / 9
  let remaining_after_first := total_paint * (1 - first_week_fraction)
  let second_week_usage := remaining_after_first / 5
  total_used = total_paint * first_week_fraction + second_week_usage →
  first_week_fraction = 5 / 9 := by
sorry

end NUMINAMATH_CALUDE_paint_usage_fraction_l154_15452


namespace NUMINAMATH_CALUDE_three_isosceles_right_triangles_l154_15482

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := 2 * x^2 + 4 * x - y^2 = 0

-- Define an isosceles right triangle with O as the right angle
def isosceles_right_triangle (A B : ℝ × ℝ) : Prop :=
  let (xA, yA) := A
  let (xB, yB) := B
  xA * xB + yA * yB = 0 ∧ xA^2 + yA^2 = xB^2 + yB^2

-- Main theorem
theorem three_isosceles_right_triangles :
  ∃ (S : Finset (ℝ × ℝ)),
    Finset.card S = 3 ∧
    (∀ A ∈ S, hyperbola A.1 A.2) ∧
    (∀ A B, A ∈ S → B ∈ S → A ≠ B → isosceles_right_triangle A B) ∧
    (∀ A B, hyperbola A.1 A.2 → hyperbola B.1 B.2 → 
      isosceles_right_triangle A B → (A ∈ S ∧ B ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_three_isosceles_right_triangles_l154_15482


namespace NUMINAMATH_CALUDE_maltese_cross_to_square_l154_15420

/-- Represents a piece of the Maltese cross -/
structure Piece where
  area : ℝ

/-- Represents the Maltese cross -/
structure MalteseCross where
  pieces : Finset Piece
  total_area : ℝ

/-- Represents a square -/
structure Square where
  side_length : ℝ

/-- A function that checks if a set of pieces can form a square -/
def can_form_square (pieces : Finset Piece) : Prop :=
  ∃ (s : Square), s.side_length^2 = (pieces.sum (λ p => p.area))

theorem maltese_cross_to_square (cross : MalteseCross) : 
  cross.total_area = 17 → 
  (∃ (cut_pieces : Finset Piece), 
    cut_pieces.card = 7 ∧ 
    (cut_pieces.sum (λ p => p.area) = cross.total_area) ∧
    can_form_square cut_pieces) := by
  sorry

end NUMINAMATH_CALUDE_maltese_cross_to_square_l154_15420


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l154_15412

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℤ) : ℤ := -a

/-- Prove that the opposite of -2023 is 2023. -/
theorem opposite_of_negative_2023 : opposite (-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l154_15412


namespace NUMINAMATH_CALUDE_complex_equation_solution_l154_15461

theorem complex_equation_solution (a : ℝ) : 
  Complex.abs (a - 2 + (4 + 3 * Complex.I) / (1 + 2 * Complex.I)) = Real.sqrt 3 * a → 
  a = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l154_15461


namespace NUMINAMATH_CALUDE_line_segment_length_l154_15495

/-- The length of a line segment with endpoints (1, 2) and (8, 6) is √65 -/
theorem line_segment_length : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (8, 6)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_length_l154_15495


namespace NUMINAMATH_CALUDE_piano_lessons_cost_l154_15440

theorem piano_lessons_cost (piano_cost : ℝ) (num_lessons : ℕ) (lesson_cost : ℝ) (discount_rate : ℝ) :
  piano_cost = 500 →
  num_lessons = 20 →
  lesson_cost = 40 →
  discount_rate = 0.25 →
  piano_cost + (num_lessons : ℝ) * lesson_cost * (1 - discount_rate) = 1100 := by
  sorry

end NUMINAMATH_CALUDE_piano_lessons_cost_l154_15440


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l154_15442

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 4 + a 5 + a 6 + a 7 = 400) :
  a 2 + a 8 = 160 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l154_15442


namespace NUMINAMATH_CALUDE_age_inconsistency_l154_15439

/-- Given the ages of Sandy, Molly, and Noah, this theorem proves that the given conditions lead to a contradiction. -/
theorem age_inconsistency (S M N : ℕ) : 
  (M = S + 20) →  -- Sandy is younger than Molly by 20 years
  (S : ℚ) / M = 7 / 9 →  -- The ratio of Sandy's age to Molly's age is 7:9
  S + M + N = 120 →  -- The sum of their ages is 120
  (N - M : ℚ) = (1 / 2 : ℚ) * (M - S) →  -- The age difference between Noah and Molly is half that between Sandy and Molly
  False :=
by
  sorry

#eval 70 + 90  -- This evaluates to 160, which is already greater than 120


end NUMINAMATH_CALUDE_age_inconsistency_l154_15439


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l154_15421

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 1 → |x| > 1) ∧ 
  (∃ x : ℝ, |x| > 1 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l154_15421


namespace NUMINAMATH_CALUDE_illegal_parking_percentage_l154_15450

theorem illegal_parking_percentage (total_cars : ℕ) (towed_cars : ℕ) (illegal_cars : ℕ) :
  towed_cars = (2 : ℕ) * total_cars / 100 →
  (80 : ℕ) * illegal_cars / 100 = illegal_cars - towed_cars →
  illegal_cars * 100 / total_cars = 10 := by
  sorry

end NUMINAMATH_CALUDE_illegal_parking_percentage_l154_15450


namespace NUMINAMATH_CALUDE_smallest_divisor_with_remainder_l154_15444

theorem smallest_divisor_with_remainder (n : ℕ) : 
  (∃ (k : ℕ), n = 10 * k) ∧ 
  (19^19 + 19) % n = 18 ∧ 
  (∀ m : ℕ, m < n → m % 10 = 0 → (19^19 + 19) % m ≠ 18) → 
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_smallest_divisor_with_remainder_l154_15444


namespace NUMINAMATH_CALUDE_triangle_side_length_l154_15463

/-- A triangle with circumradius 1 -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (circumradius : ℝ)
  (h_circumradius : circumradius = 1)

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The circumcircle of a triangle -/
def circumcircle (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- The circle passing through two points and the orthocenter -/
def circle_through_points_and_orthocenter (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- The center of a circle -/
def center (c : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem triangle_side_length (t : Triangle) :
  center (circle_through_points_and_orthocenter t) ∈ circumcircle t →
  distance t.A t.C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l154_15463


namespace NUMINAMATH_CALUDE_committee_probability_l154_15496

/-- The probability of selecting a committee with at least one boy and one girl -/
theorem committee_probability (total : ℕ) (boys : ℕ) (girls : ℕ) (committee_size : ℕ) : 
  total = 30 →
  boys = 12 →
  girls = 18 →
  committee_size = 6 →
  (1 : ℚ) - (Nat.choose boys committee_size + Nat.choose girls committee_size : ℚ) / 
    (Nat.choose total committee_size : ℚ) = 574287 / 593775 := by
  sorry

#check committee_probability

end NUMINAMATH_CALUDE_committee_probability_l154_15496


namespace NUMINAMATH_CALUDE_product_1011_2_112_3_l154_15492

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.reverse.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a ternary number represented as a list of digits to its decimal equivalent -/
def ternary_to_decimal (digits : List ℕ) : ℕ :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

/-- The main theorem stating that the product of 1011₂ and 112₃ in base 10 is 154 -/
theorem product_1011_2_112_3 : 
  (binary_to_decimal [true, true, false, true]) * 
  (ternary_to_decimal [2, 1, 1]) = 154 := by
  sorry

#eval binary_to_decimal [true, true, false, true]  -- Should output 11
#eval ternary_to_decimal [2, 1, 1]  -- Should output 14

end NUMINAMATH_CALUDE_product_1011_2_112_3_l154_15492


namespace NUMINAMATH_CALUDE_max_largest_integer_l154_15473

theorem max_largest_integer (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℚ) / 5 = 45 →
  (max a (max b (max c (max d e)))) - (min a (min b (min c (min d e)))) = 10 →
  (max a (max b (max c (max d e)))) ≤ 215 :=
by sorry

end NUMINAMATH_CALUDE_max_largest_integer_l154_15473


namespace NUMINAMATH_CALUDE_counterexample_exists_l154_15470

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l154_15470


namespace NUMINAMATH_CALUDE_archie_marbles_l154_15465

/-- Represents the number of marbles Archie has at various stages --/
structure MarbleCount where
  initial : ℕ
  afterStreet : ℕ
  afterSewer : ℕ
  afterBush : ℕ
  final : ℕ

/-- Represents the number of Glacier marbles Archie has at various stages --/
structure GlacierMarbleCount where
  initial : ℕ
  final : ℕ

/-- The main theorem about Archie's marbles --/
theorem archie_marbles (m : MarbleCount) (g : GlacierMarbleCount) : 
  (m.afterStreet = (m.initial * 2) / 5) →
  (m.afterSewer = m.afterStreet / 2) →
  (m.afterBush = (m.afterSewer * 3) / 4) →
  (m.final = m.afterBush + 5) →
  (m.final = 15) →
  (g.final = 4) →
  (g.initial = (m.initial * 3) / 10) →
  (m.initial = 67 ∧ g.initial - g.final = 16) := by
  sorry


end NUMINAMATH_CALUDE_archie_marbles_l154_15465


namespace NUMINAMATH_CALUDE_least_value_of_d_l154_15466

theorem least_value_of_d :
  let f : ℝ → ℝ := λ d => |((3 - 2*d) / 5) + 2|
  ∃ d_min : ℝ, d_min = -1 ∧
    (∀ d : ℝ, f d ≤ 3 → d ≥ d_min) ∧
    (f d_min ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_least_value_of_d_l154_15466


namespace NUMINAMATH_CALUDE_fortieth_term_is_81_l154_15460

/-- An arithmetic sequence starting from 3 with common difference 2 -/
def arithmeticSequence (n : ℕ) : ℕ := 3 + 2 * (n - 1)

/-- The 40th term of the arithmetic sequence is 81 -/
theorem fortieth_term_is_81 : arithmeticSequence 40 = 81 := by
  sorry

end NUMINAMATH_CALUDE_fortieth_term_is_81_l154_15460


namespace NUMINAMATH_CALUDE_square_area_17m_l154_15402

theorem square_area_17m (side_length : ℝ) (h : side_length = 17) :
  side_length * side_length = 289 := by
  sorry

end NUMINAMATH_CALUDE_square_area_17m_l154_15402


namespace NUMINAMATH_CALUDE_chips_division_l154_15479

theorem chips_division (total_chips : ℕ) (ratio_small : ℕ) (ratio_large : ℕ) :
  total_chips = 100 →
  ratio_small = 4 →
  ratio_large = 6 →
  (ratio_large : ℚ) / (ratio_small + ratio_large : ℚ) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_chips_division_l154_15479


namespace NUMINAMATH_CALUDE_conference_attendees_l154_15469

theorem conference_attendees (total : ℕ) (first_known : ℕ) : 
  total = 47 → first_known = 16 → 
  ∃ (women men : ℕ), 
    women + men = total ∧ 
    men = first_known + (women - 1) ∧
    women = 16 ∧ 
    men = 31 := by
  sorry

end NUMINAMATH_CALUDE_conference_attendees_l154_15469
