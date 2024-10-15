import Mathlib

namespace NUMINAMATH_CALUDE_sequence_a_formula_sequence_a_first_term_sequence_a_second_term_sequence_a_third_term_sequence_a_fourth_term_l370_37068

def sequence_a (n : ℕ) : ℚ := (18 * n - 9) / (7 * (10^n - 1))

theorem sequence_a_formula (n : ℕ) : 
  sequence_a n = (18 * n - 9) / (7 * (10^n - 1)) :=
by sorry

theorem sequence_a_first_term : sequence_a 1 = 1 / 7 :=
by sorry

theorem sequence_a_second_term : sequence_a 2 = 3 / 77 :=
by sorry

theorem sequence_a_third_term : sequence_a 3 = 5 / 777 :=
by sorry

theorem sequence_a_fourth_term : sequence_a 4 = 7 / 7777 :=
by sorry

end NUMINAMATH_CALUDE_sequence_a_formula_sequence_a_first_term_sequence_a_second_term_sequence_a_third_term_sequence_a_fourth_term_l370_37068


namespace NUMINAMATH_CALUDE_lcm_of_54_96_120_150_l370_37061

theorem lcm_of_54_96_120_150 : Nat.lcm 54 (Nat.lcm 96 (Nat.lcm 120 150)) = 21600 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_54_96_120_150_l370_37061


namespace NUMINAMATH_CALUDE_parabola_equation_l370_37054

def is_valid_parabola (p : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ),
    (2 * x₁ + 1)^2 = 2 * p * x₁ ∧
    (2 * x₂ + 1)^2 = 2 * p * x₂ ∧
    (x₁ - x₂)^2 * 5 = 15

theorem parabola_equation :
  ∀ p : ℝ, is_valid_parabola p → (p = -2 ∨ p = 6) := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l370_37054


namespace NUMINAMATH_CALUDE_cube_root_of_decimal_l370_37005

theorem cube_root_of_decimal (x : ℚ) : x = 1/4 → x^3 = 15625/1000000 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_decimal_l370_37005


namespace NUMINAMATH_CALUDE_f_negative_one_value_l370_37080

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_negative_one_value : 
  (∀ x, f (x / (1 + x)) = x) → f (-1) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_one_value_l370_37080


namespace NUMINAMATH_CALUDE_sphere_plane_distance_l370_37007

/-- Given a sphere and a plane cutting it, this theorem relates the radius of the sphere,
    the radius of the circular section, and the distance from the sphere's center to the plane. -/
theorem sphere_plane_distance (R r d : ℝ) : R = 2 * Real.sqrt 3 → r = 2 → d ^ 2 + r ^ 2 = R ^ 2 → d = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_plane_distance_l370_37007


namespace NUMINAMATH_CALUDE_inverse_sum_simplification_l370_37070

theorem inverse_sum_simplification (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ - y⁻¹ + z⁻¹)⁻¹ = (x * y * z) / (y * z - x * z + x * y) := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_simplification_l370_37070


namespace NUMINAMATH_CALUDE_case1_exists_case2_not_exists_l370_37022

-- Define a tetrahedron as a collection of 6 edge lengths
def Tetrahedron := Fin 6 → ℝ

-- Define the property of a valid tetrahedron
def is_valid_tetrahedron (t : Tetrahedron) : Prop := sorry

-- Define the conditions for case 1
def satisfies_case1 (t : Tetrahedron) : Prop :=
  (∃ i j, i ≠ j ∧ t i < 0.01 ∧ t j < 0.01) ∧
  (∃ a b c d, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
    t a > 1000 ∧ t b > 1000 ∧ t c > 1000 ∧ t d > 1000)

-- Define the conditions for case 2
def satisfies_case2 (t : Tetrahedron) : Prop :=
  (∃ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧
    t i < 0.01 ∧ t j < 0.01 ∧ t k < 0.01 ∧ t l < 0.01) ∧
  (∃ a b, a ≠ b ∧ t a > 1000 ∧ t b > 1000)

-- Theorem for case 1
theorem case1_exists :
  ∃ t : Tetrahedron, is_valid_tetrahedron t ∧ satisfies_case1 t := by sorry

-- Theorem for case 2
theorem case2_not_exists :
  ¬ ∃ t : Tetrahedron, is_valid_tetrahedron t ∧ satisfies_case2 t := by sorry

end NUMINAMATH_CALUDE_case1_exists_case2_not_exists_l370_37022


namespace NUMINAMATH_CALUDE_tile1_in_position_B_l370_37045

-- Define a tile with numbers on its sides
structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

-- Define the four tiles
def tile1 : Tile := ⟨5, 3, 2, 4⟩
def tile2 : Tile := ⟨3, 1, 5, 2⟩
def tile3 : Tile := ⟨4, 0, 6, 5⟩
def tile4 : Tile := ⟨2, 4, 3, 0⟩

-- Define the possible positions
inductive Position
  | A | B | C | D

-- Function to check if two tiles can be adjacent
def canBeAdjacent (t1 t2 : Tile) : Bool :=
  (t1.right = t2.left) ∨ (t1.left = t2.right) ∨ (t1.top = t2.bottom) ∨ (t1.bottom = t2.top)

-- Theorem: Tile 1 must be in position B
theorem tile1_in_position_B :
  ∃ (p2 p3 p4 : Position), 
    p2 ≠ Position.B ∧ p3 ≠ Position.B ∧ p4 ≠ Position.B ∧
    p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    (canBeAdjacent tile1 tile2 → 
      (p2 = Position.A ∨ p2 = Position.C ∨ p2 = Position.D)) ∧
    (canBeAdjacent tile1 tile3 → 
      (p3 = Position.A ∨ p3 = Position.C ∨ p3 = Position.D)) ∧
    (canBeAdjacent tile1 tile4 → 
      (p4 = Position.A ∨ p4 = Position.C ∨ p4 = Position.D)) :=
by
  sorry


end NUMINAMATH_CALUDE_tile1_in_position_B_l370_37045


namespace NUMINAMATH_CALUDE_line_perpendicular_planes_parallel_l370_37095

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicularToPlane : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallelPlanes : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicularLines : Line → Line → Prop)

-- Define the "contained in" relation for a line in a plane
variable (containedIn : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_planes_parallel
  (l m : Line) (α β : Plane)
  (h1 : perpendicularToPlane l α)
  (h2 : containedIn m β) :
  (∀ x y, parallelPlanes x y → perpendicularLines l m) ∧
  ∃ x y, perpendicularLines x y ∧ ¬parallelPlanes α β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_planes_parallel_l370_37095


namespace NUMINAMATH_CALUDE_customer_count_is_twenty_l370_37002

/-- The number of customers who bought marbles from Mr Julien's store -/
def number_of_customers (initial_marbles final_marbles marbles_per_customer : ℕ) : ℕ :=
  (initial_marbles - final_marbles) / marbles_per_customer

/-- Theorem stating that the number of customers who bought marbles is 20 -/
theorem customer_count_is_twenty :
  number_of_customers 400 100 15 = 20 := by
  sorry

end NUMINAMATH_CALUDE_customer_count_is_twenty_l370_37002


namespace NUMINAMATH_CALUDE_greatest_gcd_pentagonal_l370_37023

def P (n : ℕ+) : ℕ := (n : ℕ).succ * n

theorem greatest_gcd_pentagonal (n : ℕ+) : 
  (Nat.gcd (6 * P n) (n.val - 2) : ℕ) ≤ 24 ∧ 
  ∃ m : ℕ+, (Nat.gcd (6 * P m) (m.val - 2) : ℕ) = 24 :=
sorry

end NUMINAMATH_CALUDE_greatest_gcd_pentagonal_l370_37023


namespace NUMINAMATH_CALUDE_olivia_spent_15_dollars_l370_37091

/-- The amount spent at a supermarket, given the initial amount and the amount left after spending. -/
def amount_spent (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

/-- Proves that Olivia spent 15 dollars at the supermarket. -/
theorem olivia_spent_15_dollars : amount_spent 78 63 = 15 := by
  sorry

end NUMINAMATH_CALUDE_olivia_spent_15_dollars_l370_37091


namespace NUMINAMATH_CALUDE_product_of_cosines_l370_37066

theorem product_of_cosines : 
  (1 + Real.cos (π / 12)) * (1 + Real.cos (5 * π / 12)) * 
  (1 + Real.cos (7 * π / 12)) * (1 + Real.cos (11 * π / 12)) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cosines_l370_37066


namespace NUMINAMATH_CALUDE_pages_per_day_to_finish_on_time_l370_37024

/-- Given a 66-page paper due in 6 days, prove that 11 pages per day are required to finish on time. -/
theorem pages_per_day_to_finish_on_time :
  let total_pages : ℕ := 66
  let days_until_due : ℕ := 6
  let pages_per_day : ℕ := total_pages / days_until_due
  pages_per_day = 11 := by sorry

end NUMINAMATH_CALUDE_pages_per_day_to_finish_on_time_l370_37024


namespace NUMINAMATH_CALUDE_star_arrangement_exists_l370_37077

/-- A type representing a star-like configuration with 11 positions --/
structure StarConfiguration :=
  (positions : Fin 11 → ℕ)

/-- The sum of numbers from 1 to 11 --/
def sum_1_to_11 : ℕ := (11 * 12) / 2

/-- The segments of the star configuration --/
def segments : List (Fin 11 × Fin 11 × Fin 11) := sorry

/-- The condition that all numbers from 1 to 11 are used exactly once --/
def valid_arrangement (config : StarConfiguration) : Prop :=
  (∀ n : Fin 11, ∃ p : Fin 11, config.positions p = n.val + 1) ∧
  (∀ p q : Fin 11, p ≠ q → config.positions p ≠ config.positions q)

/-- The condition that the sum of each segment is 18 --/
def segment_sum_18 (config : StarConfiguration) : Prop :=
  ∀ seg ∈ segments, 
    config.positions seg.1 + config.positions seg.2.1 + config.positions seg.2.2 = 18

/-- The main theorem: there exists a valid arrangement with segment sum 18 --/
theorem star_arrangement_exists : 
  ∃ (config : StarConfiguration), valid_arrangement config ∧ segment_sum_18 config := by
  sorry

end NUMINAMATH_CALUDE_star_arrangement_exists_l370_37077


namespace NUMINAMATH_CALUDE_same_function_shifted_possible_same_function_different_variable_power_zero_not_always_one_same_domain_range_not_same_function_l370_37050

-- Define a function type
def RealFunction := ℝ → ℝ

-- Statement 1
theorem same_function_shifted_possible : 
  ∃ (f : RealFunction), ∀ x : ℝ, f x = f (x + 1) :=
sorry

-- Statement 2
theorem same_function_different_variable (f : RealFunction) :
  ∀ x t : ℝ, f x = f t :=
sorry

-- Statement 3
theorem power_zero_not_always_one :
  ∃ x : ℝ, x^0 ≠ 1 :=
sorry

-- Statement 4
theorem same_domain_range_not_same_function :
  ∃ (f g : RealFunction), (∀ x : ℝ, ∃ y : ℝ, f x = y ∧ g x = y) ∧ f ≠ g :=
sorry

end NUMINAMATH_CALUDE_same_function_shifted_possible_same_function_different_variable_power_zero_not_always_one_same_domain_range_not_same_function_l370_37050


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l370_37071

theorem product_of_three_numbers (a b c : ℝ) 
  (sum_eq : a + b + c = 22)
  (sum_squares_eq : a^2 + b^2 + c^2 = 404)
  (sum_cubes_eq : a^3 + b^3 + c^3 = 9346) :
  a * b * c = 446 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l370_37071


namespace NUMINAMATH_CALUDE_min_cost_22_bottles_l370_37067

/-- Calculates the minimum cost to buy a given number of bottles -/
def min_cost (single_price : ℚ) (box_price : ℚ) (bottles_needed : ℕ) : ℚ :=
  let box_size := 6
  let full_boxes := bottles_needed / box_size
  let remaining_bottles := bottles_needed % box_size
  full_boxes * box_price + remaining_bottles * single_price

/-- The minimum cost to buy 22 bottles is R$ 56.20 -/
theorem min_cost_22_bottles :
  min_cost (280 / 100) (1500 / 100) 22 = 5620 / 100 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_22_bottles_l370_37067


namespace NUMINAMATH_CALUDE_jayda_spending_l370_37058

theorem jayda_spending (aitana_spending jayda_spending : ℚ) : 
  aitana_spending = jayda_spending + (2/5 : ℚ) * jayda_spending →
  aitana_spending + jayda_spending = 960 →
  jayda_spending = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_jayda_spending_l370_37058


namespace NUMINAMATH_CALUDE_unique_solution_inequality_l370_37021

theorem unique_solution_inequality (a : ℝ) : 
  (∃! x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2) ↔ (a = 1 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_inequality_l370_37021


namespace NUMINAMATH_CALUDE_binomial_10_3_l370_37027

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l370_37027


namespace NUMINAMATH_CALUDE_heejin_is_oldest_l370_37094

-- Define the ages of the three friends
def yoona_age : ℕ := 23
def miyoung_age : ℕ := 22
def heejin_age : ℕ := 24

-- Theorem stating that Heejin is the oldest
theorem heejin_is_oldest : 
  heejin_age ≥ yoona_age ∧ heejin_age ≥ miyoung_age := by
  sorry

end NUMINAMATH_CALUDE_heejin_is_oldest_l370_37094


namespace NUMINAMATH_CALUDE_goat_difference_l370_37092

-- Define the number of goats for each person
def adam_goats : ℕ := 7
def ahmed_goats : ℕ := 13

-- Define Andrew's goats in terms of Adam's
def andrew_goats : ℕ := 2 * adam_goats + 5

-- Theorem statement
theorem goat_difference : andrew_goats - ahmed_goats = 6 := by
  sorry

end NUMINAMATH_CALUDE_goat_difference_l370_37092


namespace NUMINAMATH_CALUDE_dinner_cost_theorem_l370_37090

/-- Calculate the total amount Bret spends on dinner -/
def dinner_cost : ℝ :=
  let team_a_size : ℕ := 4
  let team_b_size : ℕ := 4
  let main_meal_cost : ℝ := 12.00
  let team_a_appetizers : ℕ := 2
  let team_a_appetizer_cost : ℝ := 6.00
  let team_b_appetizers : ℕ := 3
  let team_b_appetizer_cost : ℝ := 8.00
  let sharing_plates : ℕ := 4
  let sharing_plate_cost : ℝ := 10.00
  let tip_percentage : ℝ := 0.20
  let rush_order_fee : ℝ := 5.00
  let sales_tax_rate : ℝ := 0.07

  let main_meals_cost := (team_a_size + team_b_size) * main_meal_cost
  let appetizers_cost := team_a_appetizers * team_a_appetizer_cost + team_b_appetizers * team_b_appetizer_cost
  let sharing_plates_cost := sharing_plates * sharing_plate_cost
  let food_cost := main_meals_cost + appetizers_cost + sharing_plates_cost
  let tip := food_cost * tip_percentage
  let subtotal := food_cost + tip + rush_order_fee
  let sales_tax := (food_cost + tip) * sales_tax_rate
  food_cost + tip + rush_order_fee + sales_tax

theorem dinner_cost_theorem : dinner_cost = 225.85 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cost_theorem_l370_37090


namespace NUMINAMATH_CALUDE_cards_in_same_envelope_probability_l370_37075

/-- The number of cards -/
def num_cards : ℕ := 6

/-- The number of envelopes -/
def num_envelopes : ℕ := 3

/-- The number of cards per envelope -/
def cards_per_envelope : ℕ := 2

/-- The set of all possible distributions of cards into envelopes -/
def all_distributions : Finset (Fin num_cards → Fin num_envelopes) :=
  sorry

/-- The set of distributions where cards 1 and 2 are in the same envelope -/
def favorable_distributions : Finset (Fin num_cards → Fin num_envelopes) :=
  sorry

/-- The probability of cards 1 and 2 being in the same envelope -/
def prob_same_envelope : ℚ :=
  (favorable_distributions.card : ℚ) / (all_distributions.card : ℚ)

theorem cards_in_same_envelope_probability :
  prob_same_envelope = 1 / 5 :=
sorry

end NUMINAMATH_CALUDE_cards_in_same_envelope_probability_l370_37075


namespace NUMINAMATH_CALUDE_A_intersect_B_empty_l370_37059

def A : Set ℤ := {x | ∃ n : ℕ+, x = 2*n - 1}

def B : Set ℤ := {y | ∃ x ∈ A, y = 3*x - 1}

theorem A_intersect_B_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_empty_l370_37059


namespace NUMINAMATH_CALUDE_expansion_coefficient_l370_37093

theorem expansion_coefficient (m : ℤ) : 
  (Nat.choose 6 3 : ℤ) * m^3 = -160 → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l370_37093


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l370_37031

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 5/8
  let b : ℚ := 7/8
  let c : ℚ := 3/4
  c = (a + b) / 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l370_37031


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l370_37026

theorem sum_of_roots_cubic_equation :
  let f : ℝ → ℝ := λ x => 4 * x^3 + 5 * x^2 - 8 * x
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ (r₁ + r₂ + r₃ = -1.25) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l370_37026


namespace NUMINAMATH_CALUDE_line_intercepts_l370_37012

/-- Given a line with equation 2x - 3y = 6, prove that its x-intercept is 3 and y-intercept is -2 -/
theorem line_intercepts :
  let line : ℝ → ℝ → Prop := λ x y => 2 * x - 3 * y = 6
  ∃ (x y : ℝ), (line x 0 ∧ x = 3) ∧ (line 0 y ∧ y = -2) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_l370_37012


namespace NUMINAMATH_CALUDE_min_sum_with_constraint_min_sum_achieved_l370_37098

/-- Given two natural numbers a and b satisfying 1a + 4b = 30,
    their sum is minimized when a = b = 6 -/
theorem min_sum_with_constraint (a b : ℕ) (h : a + 4 * b = 30) :
  a + b ≥ 12 := by
sorry

/-- The minimum sum of 12 is achieved when a = b = 6 -/
theorem min_sum_achieved : ∃ (a b : ℕ), a + 4 * b = 30 ∧ a + b = 12 := by
sorry

end NUMINAMATH_CALUDE_min_sum_with_constraint_min_sum_achieved_l370_37098


namespace NUMINAMATH_CALUDE_divisibility_problem_l370_37046

theorem divisibility_problem (N : ℕ) (h1 : N % 44 = 0) (h2 : N % 30 = 18) : N / 44 = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l370_37046


namespace NUMINAMATH_CALUDE_student_count_l370_37043

theorem student_count : ∃ S : ℕ, 
  S = 92 ∧ 
  (3 / 8 : ℚ) * (S - 20 : ℚ) = 27 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l370_37043


namespace NUMINAMATH_CALUDE_largest_valid_number_l370_37076

def is_valid (n : ℕ) : Prop :=
  (n ≥ 1000000 ∧ n ≤ 9999999) ∧
  ∀ i : ℕ, i ∈ [0, 1, 2, 3, 4] →
    (((n / 10^i) % 1000) % 11 = 0 ∨ ((n / 10^i) % 1000) % 13 = 0)

theorem largest_valid_number :
  is_valid 9884737 ∧ ∀ m : ℕ, is_valid m → m ≤ 9884737 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l370_37076


namespace NUMINAMATH_CALUDE_other_number_proof_l370_37082

theorem other_number_proof (A B : ℕ+) (hcf lcm : ℕ+) : 
  hcf = Nat.gcd A B →
  lcm = Nat.lcm A B →
  hcf = 12 →
  lcm = 396 →
  A = 36 →
  B = 132 := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l370_37082


namespace NUMINAMATH_CALUDE_fraction_inequality_l370_37062

theorem fraction_inequality (x : ℝ) : x / (x + 1) < 0 ↔ -1 < x ∧ x < 0 := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l370_37062


namespace NUMINAMATH_CALUDE_cos_54_degrees_l370_37015

theorem cos_54_degrees : Real.cos (54 * π / 180) = (-1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_54_degrees_l370_37015


namespace NUMINAMATH_CALUDE_f_properties_l370_37069

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x

theorem f_properties :
  (∀ x > 1, f x > 0) ∧
  (∀ x, 0 < x → x < 1 → f x < 0) ∧
  (∀ x > 0, f x ≥ -1 / (2 * Real.exp 1)) ∧
  (∀ x > 0, f x ≥ x - 1) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l370_37069


namespace NUMINAMATH_CALUDE_max_profit_is_12250_l370_37036

/-- Represents the profit function for selling humidifiers -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 300 * x + 10000

/-- Represents the selling price of a humidifier -/
def selling_price (x : ℝ) : ℝ := 100 + x

/-- Represents the daily sales volume -/
def daily_sales (x : ℝ) : ℝ := 500 - 10 * x

/-- Theorem stating that the maximum profit is 12250 yuan -/
theorem max_profit_is_12250 :
  ∃ x : ℝ, 
    (∀ y : ℝ, profit_function y ≤ profit_function x) ∧ 
    profit_function x = 12250 ∧
    selling_price x = 115 :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_12250_l370_37036


namespace NUMINAMATH_CALUDE_wall_width_proof_l370_37074

def wall_height : ℝ := 4
def wall_area : ℝ := 16

theorem wall_width_proof :
  ∃ (width : ℝ), width * wall_height = wall_area ∧ width = 4 := by
sorry

end NUMINAMATH_CALUDE_wall_width_proof_l370_37074


namespace NUMINAMATH_CALUDE_rectangle_width_l370_37003

theorem rectangle_width (perimeter length width : ℝ) : 
  perimeter = 16 ∧ 
  width = length + 2 ∧ 
  perimeter = 2 * (length + width) →
  width = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l370_37003


namespace NUMINAMATH_CALUDE_length_PS_l370_37087

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  ∃ (x y z : ℝ), P = (0, 0) ∧ Q = (x, y) ∧ R = (z, 0)

-- Define a right angle at P
def RightAngleAtP (P Q R : ℝ × ℝ) : Prop :=
  (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0

-- Define the lengths of PR and PQ
def LengthPR (P R : ℝ × ℝ) : ℝ := 3
def LengthPQ (P Q : ℝ × ℝ) : ℝ := 4

-- Define S as the point where the angle bisector of ∠QPR meets QR
def AngleBisectorS (P Q R S : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ S = (t * Q.1 + (1 - t) * R.1, t * Q.2 + (1 - t) * R.2) ∧
  (S.1 - P.1) * (Q.2 - P.2) = (S.2 - P.2) * (Q.1 - P.1) ∧
  (S.1 - P.1) * (R.2 - P.2) = (S.2 - P.2) * (R.1 - P.1)

-- Main theorem
theorem length_PS (P Q R S : ℝ × ℝ) :
  Triangle P Q R →
  RightAngleAtP P Q R →
  LengthPR P R = 3 →
  LengthPQ P Q = 4 →
  AngleBisectorS P Q R S →
  Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2) = 20/7 := by
  sorry

end NUMINAMATH_CALUDE_length_PS_l370_37087


namespace NUMINAMATH_CALUDE_sqrt_pattern_l370_37053

theorem sqrt_pattern (n : ℕ) (h : n ≥ 2) :
  Real.sqrt (n - n / (n^2 + 1)) = n * Real.sqrt (n / (n^2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_pattern_l370_37053


namespace NUMINAMATH_CALUDE_farah_order_match_sticks_l370_37060

/-- The number of boxes Farah ordered -/
def num_boxes : ℕ := 4

/-- The number of matchboxes in each box -/
def matchboxes_per_box : ℕ := 20

/-- The number of sticks in each matchbox -/
def sticks_per_matchbox : ℕ := 300

/-- The total number of match sticks Farah ordered -/
def total_match_sticks : ℕ := num_boxes * matchboxes_per_box * sticks_per_matchbox

theorem farah_order_match_sticks :
  total_match_sticks = 24000 := by
  sorry

end NUMINAMATH_CALUDE_farah_order_match_sticks_l370_37060


namespace NUMINAMATH_CALUDE_rainbow_preschool_students_l370_37042

theorem rainbow_preschool_students (half_day_percent : ℝ) (full_day_count : ℕ) : 
  half_day_percent = 0.25 →
  full_day_count = 60 →
  ∃ total_students : ℕ, 
    (1 - half_day_percent) * (total_students : ℝ) = full_day_count ∧
    total_students = 80 :=
by sorry

end NUMINAMATH_CALUDE_rainbow_preschool_students_l370_37042


namespace NUMINAMATH_CALUDE_fraction_simplification_l370_37079

theorem fraction_simplification (x : ℝ) (h : x = 5) :
  (x^6 - 25*x^3 + 144) / (x^3 - 12) = 114 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l370_37079


namespace NUMINAMATH_CALUDE_sports_stars_arrangement_l370_37049

/-- The number of ways to arrange players from multiple teams in a row, where teammates must sit together -/
def arrangement_count (team_sizes : List Nat) : Nat :=
  (Nat.factorial team_sizes.length) * (team_sizes.map Nat.factorial).prod

/-- Theorem: The number of ways to arrange 10 players from 4 teams (with 3, 3, 2, and 2 players respectively) in a row, where teammates must sit together, is 3456 -/
theorem sports_stars_arrangement :
  arrangement_count [3, 3, 2, 2] = 3456 := by
  sorry

#eval arrangement_count [3, 3, 2, 2]

end NUMINAMATH_CALUDE_sports_stars_arrangement_l370_37049


namespace NUMINAMATH_CALUDE_sqrt_three_plus_sqrt_seven_less_than_two_sqrt_five_l370_37004

theorem sqrt_three_plus_sqrt_seven_less_than_two_sqrt_five : 
  Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_plus_sqrt_seven_less_than_two_sqrt_five_l370_37004


namespace NUMINAMATH_CALUDE_largest_of_eight_consecutive_integers_l370_37011

theorem largest_of_eight_consecutive_integers (n : ℕ) : 
  (n > 0) →
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) = 3224) →
  (n + 7 = 406) := by
sorry

end NUMINAMATH_CALUDE_largest_of_eight_consecutive_integers_l370_37011


namespace NUMINAMATH_CALUDE_dividend_calculation_l370_37089

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 13)
  (h_quotient : quotient = 17)
  (h_remainder : remainder = 1) :
  divisor * quotient + remainder = 222 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l370_37089


namespace NUMINAMATH_CALUDE_sum_of_qp_values_l370_37073

def p (x : ℝ) : ℝ := |x| + 1

def q (x : ℝ) : ℝ := -|x - 1|

def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3]

theorem sum_of_qp_values :
  (x_values.map (λ x => q (p x))).sum = -21 := by sorry

end NUMINAMATH_CALUDE_sum_of_qp_values_l370_37073


namespace NUMINAMATH_CALUDE_sum_of_squared_differences_to_fourth_power_l370_37040

theorem sum_of_squared_differences_to_fourth_power :
  (6^2 - 3^2)^4 + (7^2 - 2^2)^4 = 4632066 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_differences_to_fourth_power_l370_37040


namespace NUMINAMATH_CALUDE_sam_hunts_seven_l370_37064

/-- The number of animals hunted by Sam, Rob, Mark, and Peter in a day -/
def total_animals : ℕ := 21

/-- Sam's hunt count -/
def sam_hunt : ℕ := 7

/-- Rob's hunt count in terms of Sam's -/
def rob_hunt (s : ℕ) : ℚ := s / 2

/-- Mark's hunt count in terms of Sam's -/
def mark_hunt (s : ℕ) : ℚ := (1 / 3) * (s + rob_hunt s)

/-- Peter's hunt count in terms of Sam's -/
def peter_hunt (s : ℕ) : ℚ := 3 * mark_hunt s

/-- Theorem stating that Sam hunts 7 animals given the conditions -/
theorem sam_hunts_seven :
  sam_hunt + rob_hunt sam_hunt + mark_hunt sam_hunt + peter_hunt sam_hunt = total_animals := by
  sorry

#eval sam_hunt

end NUMINAMATH_CALUDE_sam_hunts_seven_l370_37064


namespace NUMINAMATH_CALUDE_optimal_profit_l370_37099

/-- Represents the profit optimization problem for a shopping mall --/
structure ShoppingMall where
  total_boxes : ℕ
  profit_A : ℝ
  profit_B : ℝ
  profit_diff : ℝ
  price_change : ℝ
  box_change : ℝ

/-- Calculates the optimal price reduction and maximum profit --/
def optimize_profit (mall : ShoppingMall) : ℝ × ℝ :=
  sorry

/-- Theorem stating the optimal price reduction and maximum profit --/
theorem optimal_profit (mall : ShoppingMall) 
  (h1 : mall.total_boxes = 600)
  (h2 : mall.profit_A = 40000)
  (h3 : mall.profit_B = 160000)
  (h4 : mall.profit_diff = 200)
  (h5 : mall.price_change = 5)
  (h6 : mall.box_change = 2) :
  optimize_profit mall = (75, 204500) :=
sorry

end NUMINAMATH_CALUDE_optimal_profit_l370_37099


namespace NUMINAMATH_CALUDE_line_parameterization_l370_37081

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = 5 * x - 7

/-- The parametric form of the line -/
def parametric_form (s m t x y : ℝ) : Prop :=
  x = s + 2 * t ∧ y = 3 + m * t

/-- The theorem stating that s = 2 and m = 10 for the given line and parametric form -/
theorem line_parameterization :
  ∃ (s m : ℝ), (∀ (x y t : ℝ), line_equation x y ↔ parametric_form s m t x y) ∧ s = 2 ∧ m = 10 := by
  sorry


end NUMINAMATH_CALUDE_line_parameterization_l370_37081


namespace NUMINAMATH_CALUDE_shoe_selection_probability_l370_37063

def total_pairs : ℕ := 16
def black_pairs : ℕ := 8
def brown_pairs : ℕ := 4
def gray_pairs : ℕ := 2
def red_pairs : ℕ := 2

theorem shoe_selection_probability :
  let total_shoes := total_pairs * 2
  let prob_same_color_diff_foot : ℚ :=
    (black_pairs * black_pairs + brown_pairs * brown_pairs + 
     gray_pairs * gray_pairs + red_pairs * red_pairs) / 
    (total_shoes * (total_shoes - 1))
  prob_same_color_diff_foot = 11 / 62 := by sorry

end NUMINAMATH_CALUDE_shoe_selection_probability_l370_37063


namespace NUMINAMATH_CALUDE_problem_solution_l370_37018

theorem problem_solution (a b : ℤ) (ha : a = 4) (hb : b = -1) : 
  -a^2 - b^2 + a*b = -21 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l370_37018


namespace NUMINAMATH_CALUDE_ball_distribution_theorem_l370_37032

def total_balls : ℕ := 10
def red_balls : ℕ := 5
def yellow_balls : ℕ := 5
def balls_per_box : ℕ := 5
def red_in_box_A : ℕ := 3
def yellow_in_box_A : ℕ := 2
def exchanged_balls : ℕ := 3

def probability_3_red_2_yellow : ℚ := 25 / 63

def mathematical_expectation : ℚ := 12 / 5

theorem ball_distribution_theorem :
  (probability_3_red_2_yellow = 25 / 63) ∧
  (mathematical_expectation = 12 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_theorem_l370_37032


namespace NUMINAMATH_CALUDE_whales_next_year_l370_37013

/-- The number of whales last year -/
def whales_last_year : ℕ := 4000

/-- The number of whales this year -/
def whales_this_year : ℕ := 2 * whales_last_year

/-- The predicted increase in whales for next year -/
def predicted_increase : ℕ := 800

/-- The theorem stating the number of whales next year -/
theorem whales_next_year : whales_this_year + predicted_increase = 8800 := by
  sorry

end NUMINAMATH_CALUDE_whales_next_year_l370_37013


namespace NUMINAMATH_CALUDE_f_properties_l370_37028

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2*a * Real.log x + (a-2)*x

theorem f_properties :
  let f := f
  ∀ a : ℝ, ∀ x : ℝ, x > 0 →
    (∃ min_val : ℝ, a = 1 → (∀ y > 0, f 1 y ≥ f 1 2) ∧ f 1 2 = -2 * Real.log 2) ∧
    (
      (a ≥ 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f a x₁ > f a x₂) ∧
                (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) ∧
      (-2 < a ∧ a < 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < -a → f a x₁ < f a x₂) ∧
                        (∀ x₁ x₂, -a < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f a x₁ > f a x₂) ∧
                        (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) ∧
      (a = -2 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) ∧
      (a < -2 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f a x₁ < f a x₂) ∧
                (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < -a → f a x₁ > f a x₂) ∧
                (∀ x₁ x₂, -a < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂))
    ) := by sorry

end NUMINAMATH_CALUDE_f_properties_l370_37028


namespace NUMINAMATH_CALUDE_feta_cheese_price_per_pound_l370_37001

/-- Given Teresa's shopping list and total spent, calculate the price per pound of feta cheese --/
theorem feta_cheese_price_per_pound 
  (sandwich_price : ℝ) 
  (sandwich_quantity : ℕ) 
  (salami_price : ℝ) 
  (olive_price_per_pound : ℝ) 
  (olive_quantity : ℝ) 
  (bread_price : ℝ) 
  (feta_quantity : ℝ) 
  (total_spent : ℝ) 
  (h1 : sandwich_price = 7.75)
  (h2 : sandwich_quantity = 2)
  (h3 : salami_price = 4)
  (h4 : olive_price_per_pound = 10)
  (h5 : olive_quantity = 0.25)
  (h6 : bread_price = 2)
  (h7 : feta_quantity = 0.5)
  (h8 : total_spent = 40) :
  (total_spent - (sandwich_price * sandwich_quantity + salami_price + 3 * salami_price + 
  olive_price_per_pound * olive_quantity + bread_price)) / feta_quantity = 8 := by
sorry

end NUMINAMATH_CALUDE_feta_cheese_price_per_pound_l370_37001


namespace NUMINAMATH_CALUDE_correct_rainwater_collection_l370_37038

/-- Represents the water collection problem --/
structure WaterCollection where
  tankCapacity : ℕ        -- Tank capacity in liters
  riverWater : ℕ          -- Water collected from river daily in milliliters
  daysToFill : ℕ          -- Number of days to fill the tank
  rainWater : ℕ           -- Water collected from rain daily in milliliters

/-- Theorem stating the correct amount of rainwater collected daily --/
theorem correct_rainwater_collection (w : WaterCollection) 
  (h1 : w.tankCapacity = 50)
  (h2 : w.riverWater = 1700)
  (h3 : w.daysToFill = 20)
  : w.rainWater = 800 := by
  sorry

#check correct_rainwater_collection

end NUMINAMATH_CALUDE_correct_rainwater_collection_l370_37038


namespace NUMINAMATH_CALUDE_range_of_a_l370_37047

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - x + 2 * a

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, (∃ x : ℝ, f a x = y) ↔ (∃ x : ℝ, f a (f a x) = y)) →
  a ∈ Set.Ioo (1/2) 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l370_37047


namespace NUMINAMATH_CALUDE_parabola_hyperbola_equations_l370_37035

/-- A parabola and hyperbola with specific properties -/
structure ParabolaHyperbolaPair where
  -- Parabola properties
  parabola_vertex : ℝ × ℝ
  parabola_axis_through_focus : Bool
  parabola_perpendicular : Bool
  
  -- Hyperbola properties
  hyperbola_a : ℝ
  hyperbola_b : ℝ
  
  -- Intersection point
  intersection : ℝ × ℝ
  
  -- Conditions
  vertex_at_origin : parabola_vertex = (0, 0)
  axis_through_focus : parabola_axis_through_focus = true
  perpendicular_to_real_axis : parabola_perpendicular = true
  intersection_point : intersection = (3/2, Real.sqrt 6)
  hyperbola_equation : ∀ x y, x^2 / hyperbola_a^2 - y^2 / hyperbola_b^2 = 1 → 
    (x, y) ∈ Set.range (λ t : ℝ × ℝ => t)

/-- The equations of the parabola and hyperbola given the conditions -/
theorem parabola_hyperbola_equations (ph : ParabolaHyperbolaPair) :
  (∀ x y, y^2 = 4*x ↔ (x, y) ∈ Set.range (λ t : ℝ × ℝ => t)) ∧
  (∀ x y, x^2 / (1/4) - y^2 / (3/4) = 1 ↔ (x, y) ∈ Set.range (λ t : ℝ × ℝ => t)) :=
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_equations_l370_37035


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l370_37057

theorem gcd_digits_bound (a b : ℕ) : 
  (1000000 ≤ a ∧ a < 10000000) →
  (1000000 ≤ b ∧ b < 10000000) →
  (100000000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 1000000000000) →
  Nat.gcd a b < 1000 :=
by sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l370_37057


namespace NUMINAMATH_CALUDE_journal_involvement_l370_37029

theorem journal_involvement (total_students : ℕ) 
  (total_percentage : ℚ) (boys_percentage : ℚ) (girls_percentage : ℚ)
  (h1 : total_students = 75000)
  (h2 : total_percentage = 5 / 300)  -- 1 2/3% as a fraction
  (h3 : boys_percentage = 7 / 300)   -- 2 1/3% as a fraction
  (h4 : girls_percentage = 2 / 300)  -- 2/3% as a fraction
  : ∃ (boys girls : ℕ),
    boys + girls = total_students ∧
    ↑boys * boys_percentage + ↑girls * girls_percentage = ↑total_students * total_percentage ∧
    boys * boys_percentage = 700 ∧
    girls * girls_percentage = 300 :=
sorry

end NUMINAMATH_CALUDE_journal_involvement_l370_37029


namespace NUMINAMATH_CALUDE_line_through_points_2m_plus_3b_l370_37083

/-- Given a line passing through the points (-1, 1/2) and (2, -3/2), 
    prove that 2m+3b = -11/6 when the line is expressed as y = mx + b -/
theorem line_through_points_2m_plus_3b (m b : ℚ) : 
  (1/2 : ℚ) = m * (-1) + b →
  (-3/2 : ℚ) = m * 2 + b →
  2 * m + 3 * b = -11/6 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_2m_plus_3b_l370_37083


namespace NUMINAMATH_CALUDE_range_of_m_l370_37085

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (2 * y - x / Real.exp 1) * (Real.log x - Real.log y) - y / m ≤ 0) → 
  0 < m ∧ m ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l370_37085


namespace NUMINAMATH_CALUDE_power_sixteen_divided_by_eight_l370_37009

theorem power_sixteen_divided_by_eight (m : ℕ) : 
  m = 16^1000 → m / 8 = 2^3997 := by
  sorry

end NUMINAMATH_CALUDE_power_sixteen_divided_by_eight_l370_37009


namespace NUMINAMATH_CALUDE_inequality_region_l370_37034

open Real

theorem inequality_region (x y : ℝ) : 
  (x^5 - 13*x^3 + 36*x) * (x^4 - 17*x^2 + 16) / 
  ((y^5 - 13*y^3 + 36*y) * (y^4 - 17*y^2 + 16)) ≥ 0 ↔ 
  y ≠ 0 ∧ y ≠ 1 ∧ y ≠ -1 ∧ y ≠ 2 ∧ y ≠ -2 ∧ y ≠ 3 ∧ y ≠ -3 ∧ y ≠ 4 ∧ y ≠ -4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_region_l370_37034


namespace NUMINAMATH_CALUDE_cos_330_degrees_l370_37055

theorem cos_330_degrees : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l370_37055


namespace NUMINAMATH_CALUDE_area_R_specific_rhombus_l370_37006

/-- Represents a rhombus ABCD -/
structure Rhombus :=
  (side_length : ℝ)
  (angle_B : ℝ)

/-- Represents the region R inside the rhombus -/
def region_R (r : Rhombus) : Set (ℝ × ℝ) := sorry

/-- The area of region R in the rhombus -/
def area_R (r : Rhombus) : ℝ := sorry

/-- Theorem: The area of region R in a rhombus with side length 3 and angle B = 150° -/
theorem area_R_specific_rhombus :
  let r : Rhombus := { side_length := 3, angle_B := 150 }
  area_R r = (9 * (Real.sqrt 6 - Real.sqrt 2)) / 8 := by sorry

end NUMINAMATH_CALUDE_area_R_specific_rhombus_l370_37006


namespace NUMINAMATH_CALUDE_initial_hats_count_l370_37084

/-- Represents the state of hat distribution among gentlemen -/
structure HatDistribution where
  total : Nat
  withHat : Nat
  withoutHat : Nat
  givenMoreThanReceived : Nat

/-- Theorem stating that if 10 out of 20 gentlemen gave away more hats than they received,
    then the initial number of gentlemen with hats must be 10 -/
theorem initial_hats_count (dist : HatDistribution) :
  dist.total = 20 ∧
  dist.givenMoreThanReceived = 10 ∧
  dist.withHat + dist.withoutHat = dist.total →
  dist.withHat = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_hats_count_l370_37084


namespace NUMINAMATH_CALUDE_roller_plate_acceleration_l370_37041

noncomputable def g : ℝ := 10
noncomputable def R : ℝ := 1
noncomputable def r : ℝ := 0.4
noncomputable def m : ℝ := 150
noncomputable def α : ℝ := Real.arccos 0.68

theorem roller_plate_acceleration 
  (h_no_slip : True) -- Assumption of no slipping
  (h_weightless : True) -- Assumption of weightless rollers
  : ∃ (plate_acc_mag plate_acc_dir roller_acc : ℝ),
    plate_acc_mag = 4 ∧ 
    plate_acc_dir = Real.arcsin 0.4 ∧
    roller_acc = 4 := by
  sorry

end NUMINAMATH_CALUDE_roller_plate_acceleration_l370_37041


namespace NUMINAMATH_CALUDE_smallest_three_digit_sum_of_powers_l370_37033

/-- A function that checks if a number is a three-digit number -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A function that checks if a number is a one-digit positive integer -/
def isOneDigitPositive (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

/-- The main theorem statement -/
theorem smallest_three_digit_sum_of_powers :
  ∃ (K a b : ℕ), 
    isThreeDigit K ∧
    isOneDigitPositive a ∧
    isOneDigitPositive b ∧
    K = a^b + b^a ∧
    (∀ (K' a' b' : ℕ), 
      isThreeDigit K' ∧ 
      isOneDigitPositive a' ∧ 
      isOneDigitPositive b' ∧ 
      K' = a'^b' + b'^a' → 
      K ≤ K') ∧
    K = 100 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_sum_of_powers_l370_37033


namespace NUMINAMATH_CALUDE_infinite_primes_with_solutions_l370_37096

theorem infinite_primes_with_solutions : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ x y : ℤ, x^2 + x + 1 = p * y} := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_with_solutions_l370_37096


namespace NUMINAMATH_CALUDE_amys_haircut_l370_37056

/-- Amy's haircut problem -/
theorem amys_haircut (initial_length : ℝ) (final_length : ℝ) (cut_length : ℝ)
  (h1 : initial_length = 11)
  (h2 : final_length = 7)
  (h3 : cut_length = initial_length - final_length) :
  cut_length = 4 := by sorry

end NUMINAMATH_CALUDE_amys_haircut_l370_37056


namespace NUMINAMATH_CALUDE_rectangle_cutout_equals_square_area_l370_37065

theorem rectangle_cutout_equals_square_area : 
  (10 * 7 - 1 * 6 : ℕ) = 8 * 8 := by sorry

end NUMINAMATH_CALUDE_rectangle_cutout_equals_square_area_l370_37065


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l370_37048

theorem algebraic_expression_equality (a b : ℝ) (h : a - 3 * b = -3) : 
  5 - a + 3 * b = 8 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l370_37048


namespace NUMINAMATH_CALUDE_quadratic_statements_l370_37025

variable (a b c : ℝ)
variable (x₀ : ℝ)

def quadratic_equation (x : ℝ) := a * x^2 + b * x + c

theorem quadratic_statements (h : a ≠ 0) :
  (a + b + c = 0 → b^2 - 4*a*c ≥ 0) ∧
  ((∃ x y, x ≠ y ∧ a * x^2 + c = 0 ∧ a * y^2 + c = 0) →
    ∃ u v, u ≠ v ∧ quadratic_equation u = 0 ∧ quadratic_equation v = 0) ∧
  (quadratic_equation x₀ = 0 → b^2 - 4*a*c = (2*a*x₀ + b)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_statements_l370_37025


namespace NUMINAMATH_CALUDE_village_households_l370_37030

/-- The number of households in a village given water consumption data. -/
theorem village_households (water_per_household : ℕ) (total_water : ℕ) 
  (h1 : water_per_household = 200)
  (h2 : total_water = 2000)
  (h3 : total_water = water_per_household * (total_water / water_per_household)) :
  total_water / water_per_household = 10 := by
  sorry

#check village_households

end NUMINAMATH_CALUDE_village_households_l370_37030


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l370_37052

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 4| + 3 * y = 15 :=
by
  -- The unique solution is y = 4.75
  use 4.75
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l370_37052


namespace NUMINAMATH_CALUDE_decimal_repetend_of_five_thirteenth_l370_37086

/-- The decimal representation of 5/13 has a 6-digit repetend of 384615 -/
theorem decimal_repetend_of_five_thirteenth : ∃ (n : ℕ), 
  (5 : ℚ) / 13 = (384615 : ℚ) / 999999 + (n : ℚ) / 999999 := by
  sorry

end NUMINAMATH_CALUDE_decimal_repetend_of_five_thirteenth_l370_37086


namespace NUMINAMATH_CALUDE_triangle_properties_l370_37037

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- The main theorem about properties of triangle ABC -/
theorem triangle_properties (t : Triangle) :
  (t.A - t.B = t.C → ¬(t.A = Real.pi/2 ∧ t.c > t.a ∧ t.c > t.b)) ∧
  (t.a^2 = t.b^2 - t.c^2 → t.B = Real.pi/2) ∧
  (t.A / (t.A + t.B + t.C) = 1/6 ∧ t.B / (t.A + t.B + t.C) = 1/3 ∧ t.C / (t.A + t.B + t.C) = 1/2 → t.C = Real.pi/2) ∧
  (t.a^2 / (t.a^2 + t.b^2 + t.c^2) = 9/50 ∧ t.b^2 / (t.a^2 + t.b^2 + t.c^2) = 16/50 ∧ t.c^2 / (t.a^2 + t.b^2 + t.c^2) = 25/50 → t.a^2 + t.b^2 = t.c^2) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l370_37037


namespace NUMINAMATH_CALUDE_acute_triangle_properties_l370_37017

theorem acute_triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (∀ x : ℝ, x^2 - 2 * Real.sqrt 3 * x + 2 = 0 → (x = a ∨ x = b)) →
  2 * Real.sin (A + B) - Real.sqrt 3 = 0 →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  c = Real.sqrt 6 ∧
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_properties_l370_37017


namespace NUMINAMATH_CALUDE_sufficient_condition_for_line_parallel_plane_not_necessary_condition_for_line_parallel_plane_l370_37019

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (planeParallel : Plane → Plane → Prop)

-- Define the parallel relation for a line and a plane
variable (lineParallelPlane : Line → Plane → Prop)

-- Define the subset relation for a line and a plane
variable (lineInPlane : Line → Plane → Prop)

-- State the theorem
theorem sufficient_condition_for_line_parallel_plane 
  (α β : Plane) (m : Line) :
  (planeParallel α β ∧ lineInPlane m β) → lineParallelPlane m α :=
sorry

-- State that the condition is not necessary
theorem not_necessary_condition_for_line_parallel_plane 
  (α β : Plane) (m : Line) :
  ¬(lineParallelPlane m α → (planeParallel α β ∧ lineInPlane m β)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_line_parallel_plane_not_necessary_condition_for_line_parallel_plane_l370_37019


namespace NUMINAMATH_CALUDE_sum_of_real_roots_of_quartic_l370_37016

theorem sum_of_real_roots_of_quartic (x : ℝ) :
  let f : ℝ → ℝ := λ x => x^4 - 4*x - 1
  ∃ (r₁ r₂ : ℝ), (f r₁ = 0 ∧ f r₂ = 0) ∧ (∀ r : ℝ, f r = 0 → r = r₁ ∨ r = r₂) ∧ r₁ + r₂ = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_real_roots_of_quartic_l370_37016


namespace NUMINAMATH_CALUDE_system_solution_l370_37088

theorem system_solution (m n : ℝ) 
  (eq1 : m * 3 + (-7) = 5)
  (eq2 : 2 * (7/2) - n * (-2) = 13)
  : ∃ (x y : ℝ), m * x + y = 5 ∧ 2 * x - n * y = 13 ∧ x = 2 ∧ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l370_37088


namespace NUMINAMATH_CALUDE_problem_statement_l370_37020

theorem problem_statement (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49/(x - 3)^2 = 23 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l370_37020


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_tangent_circle_l370_37000

/-- The value of k for which the asymptotes of the hyperbola x^2 - y^2/k^2 = 1 
    are tangent to the circle x^2 + (y-2)^2 = 1 -/
theorem hyperbola_asymptote_tangent_circle (k : ℝ) :
  k > 0 →
  (∀ x y : ℝ, x^2 - y^2/k^2 = 1 → 
    ∃ m : ℝ, (∀ t : ℝ, (x = t ∧ y = k*t) ∨ (x = t ∧ y = -k*t)) →
      (∃ x₀ y₀ : ℝ, x₀^2 + (y₀-2)^2 = 1 ∧
        (x₀ - x)^2 + (y₀ - y)^2 = 1)) →
  k = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_tangent_circle_l370_37000


namespace NUMINAMATH_CALUDE_triangle_properties_l370_37010

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → A < π →
  B > 0 → B < π →
  C > 0 → C < π →
  a^2 - b^2 = Real.sqrt 3 * b * c →
  Real.sin C = 2 * Real.sqrt 3 * Real.sin B →
  (a * Real.sin B = b * Real.sin A) →
  (b * Real.sin C = c * Real.sin B) →
  (c * Real.sin A = a * Real.sin C) →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (b^2 = a^2 + c^2 - 2*a*c*Real.cos B) →
  (c^2 = a^2 + b^2 - 2*a*b*Real.cos C) →
  (Real.cos A = Real.sqrt 3 / 2) ∧
  (b = 1 → (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l370_37010


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l370_37008

theorem system_of_equations_solution (a b x y : ℝ) : 
  (x - y * Real.sqrt (x^2 - y^2)) / Real.sqrt (1 - x^2 + y^2) = a ∧
  (y - x * Real.sqrt (x^2 - y^2)) / Real.sqrt (1 - x^2 + y^2) = b →
  x = (a + b * Real.sqrt (a^2 - b^2)) / Real.sqrt (1 - a^2 + b^2) ∧
  y = (b + a * Real.sqrt (a^2 - b^2)) / Real.sqrt (1 - a^2 + b^2) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l370_37008


namespace NUMINAMATH_CALUDE_square_wood_weight_l370_37044

/-- Represents the properties of a piece of wood -/
structure Wood where
  length : ℝ
  width : ℝ
  weight : ℝ

/-- Calculates the area of a piece of wood -/
def area (w : Wood) : ℝ := w.length * w.width

/-- Theorem stating the weight of the square piece of wood -/
theorem square_wood_weight (rect : Wood) (square : Wood) :
  rect.length = 4 ∧ 
  rect.width = 6 ∧ 
  rect.weight = 24 ∧
  square.length = 5 ∧
  square.width = 5 →
  square.weight = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_wood_weight_l370_37044


namespace NUMINAMATH_CALUDE_coffee_shop_weekly_total_l370_37072

/-- Represents a coffee shop with its brewing characteristics -/
structure CoffeeShop where
  weekday_rate : ℕ  -- Cups brewed per hour on weekdays
  weekend_total : ℕ  -- Total cups brewed over the weekend
  daily_hours : ℕ  -- Hours open per day

/-- Calculates the total number of coffee cups brewed in one week -/
def weekly_total (shop : CoffeeShop) : ℕ :=
  (shop.weekday_rate * shop.daily_hours * 5) + shop.weekend_total

/-- Theorem stating that a coffee shop with given characteristics brews 370 cups per week -/
theorem coffee_shop_weekly_total :
  ∀ (shop : CoffeeShop),
    shop.weekday_rate = 10 ∧
    shop.weekend_total = 120 ∧
    shop.daily_hours = 5 →
    weekly_total shop = 370 := by
  sorry


end NUMINAMATH_CALUDE_coffee_shop_weekly_total_l370_37072


namespace NUMINAMATH_CALUDE_sequence_sum_zero_l370_37014

-- Define the sequence type
def Sequence := Fin 12 → ℤ

-- Define the property of sum of three consecutive terms being 40
def ConsecutiveSum (seq : Sequence) : Prop :=
  ∀ i : Fin 10, seq i + seq (i + 1) + seq (i + 2) = 40

-- Define the theorem
theorem sequence_sum_zero (seq : Sequence) 
  (h1 : ConsecutiveSum seq) 
  (h2 : seq 2 = 9) : 
  seq 0 + seq 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_sequence_sum_zero_l370_37014


namespace NUMINAMATH_CALUDE_complex_numbers_count_is_25_l370_37039

def S : Finset ℕ := {0, 1, 2, 3, 4, 5}

def complex_numbers_count : ℕ :=
  (S.filter (λ b => b ≠ 0)).card * (S.card - 1)

theorem complex_numbers_count_is_25 : complex_numbers_count = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_numbers_count_is_25_l370_37039


namespace NUMINAMATH_CALUDE_max_braking_distance_l370_37051

/-- The braking distance function for a car -/
def s (t : ℝ) : ℝ := 15 * t - 6 * t^2

/-- The maximum distance traveled by the car before stopping -/
theorem max_braking_distance :
  (∃ t : ℝ, ∀ u : ℝ, s u ≤ s t) ∧ (∃ t : ℝ, s t = 75/8) :=
sorry

end NUMINAMATH_CALUDE_max_braking_distance_l370_37051


namespace NUMINAMATH_CALUDE_nell_baseball_cards_l370_37097

theorem nell_baseball_cards (initial_cards given_to_john given_to_jeff : ℕ) 
  (h1 : initial_cards = 573)
  (h2 : given_to_john = 195)
  (h3 : given_to_jeff = 168) :
  initial_cards - (given_to_john + given_to_jeff) = 210 := by
  sorry

end NUMINAMATH_CALUDE_nell_baseball_cards_l370_37097


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l370_37078

theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let asymptotes := fun (x y : ℝ) => y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x
  e = 2 → (∀ x y, hyperbola x y ↔ asymptotes x y) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l370_37078
