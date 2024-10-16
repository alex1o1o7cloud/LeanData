import Mathlib

namespace NUMINAMATH_CALUDE_letters_with_both_in_given_alphabet_l1975_197580

/-- Represents an alphabet with letters containing dots and straight lines -/
structure Alphabet where
  total : ℕ
  line_no_dot : ℕ
  dot_no_line : ℕ
  all_have_dot_or_line : Bool

/-- The number of letters containing both a dot and a straight line -/
def letters_with_both (a : Alphabet) : ℕ :=
  a.total - a.line_no_dot - a.dot_no_line

/-- Theorem stating the number of letters with both dot and line in the given alphabet -/
theorem letters_with_both_in_given_alphabet :
  ∀ (a : Alphabet),
    a.total = 60 ∧
    a.line_no_dot = 36 ∧
    a.dot_no_line = 4 ∧
    a.all_have_dot_or_line = true →
    letters_with_both a = 20 := by
  sorry


end NUMINAMATH_CALUDE_letters_with_both_in_given_alphabet_l1975_197580


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1975_197520

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, 12 * x^2 + 7 * y^2 = 4620 ↔
    ((x = 7 ∨ x = -7) ∧ (y = 24 ∨ y = -24)) ∨
    ((x = 14 ∨ x = -14) ∧ (y = 18 ∨ y = -18)) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1975_197520


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l1975_197559

/-- A quadratic equation x^2 + bx + 25 = 0 has at least one real root if and only if b is in the set (-∞, -10] ∪ [10, ∞). -/
theorem quadratic_real_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_real_root_condition_l1975_197559


namespace NUMINAMATH_CALUDE_max_value_of_2sinx_l1975_197504

theorem max_value_of_2sinx :
  ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), 2 * Real.sin x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_2sinx_l1975_197504


namespace NUMINAMATH_CALUDE_tangent_line_at_x_squared_l1975_197566

theorem tangent_line_at_x_squared (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2
  let x₀ : ℝ := 2
  let y₀ : ℝ := f x₀
  let m : ℝ := 2 * x₀
  (λ x ↦ m * (x - x₀) + y₀) = (λ x ↦ 4 * x - 4) := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_squared_l1975_197566


namespace NUMINAMATH_CALUDE_square_perimeter_relationship_l1975_197590

/-- Given two squares C and D, where C has a perimeter of 32 cm and D has an area
    equal to one-third the area of C, the perimeter of D is (32√3)/3 cm. -/
theorem square_perimeter_relationship (C D : Real → Real → Prop) :
  (∃ (side_c : Real), C side_c side_c ∧ 4 * side_c = 32) →
  (∃ (side_d : Real), D side_d side_d ∧ side_d^2 = (side_c^2) / 3) →
  (∃ (perimeter_d : Real), perimeter_d = 32 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_relationship_l1975_197590


namespace NUMINAMATH_CALUDE_range_of_f_l1975_197571

-- Define the function f
def f (x : ℝ) := x^2 - 6*x - 9

-- State the theorem
theorem range_of_f :
  ∃ (a b : ℝ), a < b ∧
  (Set.Icc a b) = {y | ∃ x ∈ Set.Ioo 1 4, f x = y} ∧
  a = -18 ∧ b = -14 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l1975_197571


namespace NUMINAMATH_CALUDE_prize_guesses_count_l1975_197544

def digit_partitions : List (Nat × Nat × Nat) :=
  [(1,1,6), (1,2,5), (1,3,4), (1,4,3), (1,5,2), (1,6,1),
   (2,1,5), (2,2,4), (2,3,3), (2,4,2), (2,5,1),
   (3,1,4), (3,2,3), (3,3,2), (3,4,1),
   (4,1,3), (4,2,2), (4,3,1)]

def digit_arrangements : Nat := 70

theorem prize_guesses_count : 
  (List.length digit_partitions) * digit_arrangements = 1260 := by
  sorry

end NUMINAMATH_CALUDE_prize_guesses_count_l1975_197544


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l1975_197570

-- Part 1
theorem part_one : Real.sqrt 16 + (1 - Real.sqrt 3) ^ 0 - 2⁻¹ = 4.5 := by sorry

-- Part 2
def system_solution (x : ℝ) : Prop :=
  -2 * x + 6 ≥ 4 ∧ (4 * x + 1) / 3 > x - 1

theorem part_two : ∀ x : ℝ, system_solution x ↔ -4 < x ∧ x ≤ 1 := by sorry

-- Part 3
theorem part_three : {x : ℕ | system_solution x} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l1975_197570


namespace NUMINAMATH_CALUDE_road_repair_workers_l1975_197582

/-- Represents the work done by a group of workers -/
structure Work where
  persons : ℕ
  days : ℕ
  hours_per_day : ℕ

/-- Calculates the total work units -/
def total_work (w : Work) : ℕ := w.persons * w.days * w.hours_per_day

theorem road_repair_workers (first_group : Work) (second_group : Work) :
  first_group.days = 12 ∧
  first_group.hours_per_day = 5 ∧
  second_group.persons = 30 ∧
  second_group.days = 17 ∧
  second_group.hours_per_day = 6 ∧
  total_work first_group = total_work second_group →
  first_group.persons = 51 := by
  sorry

end NUMINAMATH_CALUDE_road_repair_workers_l1975_197582


namespace NUMINAMATH_CALUDE_vector_perpendicularity_l1975_197517

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vector2D) : ℝ :=
  v.x * w.x + v.y * w.y

/-- Two vectors are perpendicular if their dot product is zero -/
def is_perpendicular (v w : Vector2D) : Prop :=
  dot_product v w = 0

/-- Unit vector in positive x direction -/
def i : Vector2D :=
  ⟨1, 0⟩

/-- Unit vector in positive y direction -/
def j : Vector2D :=
  ⟨0, 1⟩

/-- Vector addition -/
def add_vectors (v w : Vector2D) : Vector2D :=
  ⟨v.x + w.x, v.y + w.y⟩

/-- Vector subtraction -/
def subtract_vectors (v w : Vector2D) : Vector2D :=
  ⟨v.x - w.x, v.y - w.y⟩

/-- Scalar multiplication of a vector -/
def scalar_mult (k : ℝ) (v : Vector2D) : Vector2D :=
  ⟨k * v.x, k * v.y⟩

theorem vector_perpendicularity :
  let a := scalar_mult 2 i
  let b := add_vectors i j
  is_perpendicular (subtract_vectors a b) b := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicularity_l1975_197517


namespace NUMINAMATH_CALUDE_penny_fountain_problem_l1975_197536

theorem penny_fountain_problem (rachelle gretchen rocky : ℕ) : 
  rachelle = 180 →
  gretchen = rachelle / 2 →
  rocky = gretchen / 3 →
  rachelle + gretchen + rocky = 300 :=
by sorry

end NUMINAMATH_CALUDE_penny_fountain_problem_l1975_197536


namespace NUMINAMATH_CALUDE_f_five_not_unique_l1975_197522

/-- A function satisfying the given functional equation for all real x and y -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x + f (3 * x + y) + 3 * x * y = f (4 * x - y) + 3 * x^2 + 2

/-- The theorem stating that f(5) cannot be uniquely determined -/
theorem f_five_not_unique : 
  ¬ ∃ (a : ℝ), ∀ (f : ℝ → ℝ), FunctionalEquation f → f 5 = a :=
sorry

end NUMINAMATH_CALUDE_f_five_not_unique_l1975_197522


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1975_197578

theorem quadratic_roots_property (m n : ℝ) : 
  (m^2 - 2*m - 2025 = 0) → 
  (n^2 - 2*n - 2025 = 0) → 
  (m^2 - 3*m - n = 2023) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1975_197578


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1975_197560

open Real

theorem sufficient_not_necessary_condition : 
  (∀ α : ℝ, ∃ k : ℤ, α = π / 6 + 2 * k * π → cos (2 * α) = 1 / 2) ∧ 
  (∃ α : ℝ, cos (2 * α) = 1 / 2 ∧ ∀ k : ℤ, α ≠ π / 6 + 2 * k * π) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1975_197560


namespace NUMINAMATH_CALUDE_room_expansion_proof_l1975_197551

theorem room_expansion_proof (initial_length initial_width increase : ℝ)
  (h1 : initial_length = 13)
  (h2 : initial_width = 18)
  (h3 : increase = 2) :
  let new_length := initial_length + increase
  let new_width := initial_width + increase
  let single_room_area := new_length * new_width
  let total_area := 4 * single_room_area + 2 * single_room_area
  total_area = 1800 := by sorry

end NUMINAMATH_CALUDE_room_expansion_proof_l1975_197551


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l1975_197540

/-- The property that a function f: ℕ → ℕ must satisfy -/
def SatisfiesProperty (f : ℕ → ℕ) : Prop :=
  ∀ n, f n + f (f n) + f (f (f n)) = 3 * n

/-- Theorem stating that the identity function is the only function satisfying the property -/
theorem unique_satisfying_function :
  ∀ f : ℕ → ℕ, SatisfiesProperty f → f = id := by
  sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l1975_197540


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1975_197592

theorem simplify_trig_expression (x : ℝ) : 
  ((1 + Real.sin x) / Real.cos x) * (Real.sin (2 * x) / (2 * (Real.cos (π/4 - x/2))^2)) = 2 * Real.sin x :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1975_197592


namespace NUMINAMATH_CALUDE_gcd_of_72_120_168_l1975_197546

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_168_l1975_197546


namespace NUMINAMATH_CALUDE_binomial_coefficient_modulo_power_of_two_l1975_197527

theorem binomial_coefficient_modulo_power_of_two 
  (n : ℕ) (r : ℕ) (h_r_odd : Odd r) :
  ∃ i : ℕ, i < 2^n ∧ Nat.choose (2^n + i) i ≡ r [MOD 2^(n+1)] := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_modulo_power_of_two_l1975_197527


namespace NUMINAMATH_CALUDE_largest_y_value_l1975_197556

theorem largest_y_value (x y : ℝ) 
  (eq1 : x^2 + 3*x*y - y^2 = 27)
  (eq2 : 3*x^2 - x*y + y^2 = 27) :
  ∃ (y_max : ℝ), y_max = 3 ∧ 
  (∀ (y' : ℝ), (∃ (x' : ℝ), x'^2 + 3*x'*y' - y'^2 = 27 ∧ 
                             3*x'^2 - x'*y' + y'^2 = 27) → 
                y' ≤ y_max) :=
sorry

end NUMINAMATH_CALUDE_largest_y_value_l1975_197556


namespace NUMINAMATH_CALUDE_vehicle_passing_condition_min_speed_for_passing_l1975_197598

-- Define the speeds and distances
def VB : ℝ := 40  -- mph
def VC : ℝ := 65  -- mph
def dist_AB : ℝ := 100  -- ft
def dist_BC : ℝ := 250  -- ft

-- Define the theorem
theorem vehicle_passing_condition (VA : ℝ) :
  VA > 2 →
  (dist_AB / (VA + VB)) < (dist_BC / (VB + VC)) :=
by
  sorry

-- Define the main theorem that answers the original question
theorem min_speed_for_passing :
  ∃ (VA : ℝ), VA > 2 ∧
  ∀ (VA' : ℝ), VA' > VA →
  (dist_AB / (VA' + VB)) < (dist_BC / (VB + VC)) :=
by
  sorry

end NUMINAMATH_CALUDE_vehicle_passing_condition_min_speed_for_passing_l1975_197598


namespace NUMINAMATH_CALUDE_parabola_tangent_point_l1975_197550

theorem parabola_tangent_point (p q : ℤ) (h : p^2 = 4*q) :
  ∃ (a b : ℤ), (a = -p ∧ b = q) ∧ a^2 = 4*b :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_point_l1975_197550


namespace NUMINAMATH_CALUDE_valid_drawings_for_ten_balls_l1975_197505

/-- The number of ways to draw balls from a box -/
def validDrawings (n k : ℕ) : ℕ :=
  Nat.choose n k - Nat.choose n (k + 1)

/-- Theorem stating the number of valid ways to draw balls -/
theorem valid_drawings_for_ten_balls :
  validDrawings 10 5 = 42 := by
  sorry

end NUMINAMATH_CALUDE_valid_drawings_for_ten_balls_l1975_197505


namespace NUMINAMATH_CALUDE_g_lower_bound_l1975_197531

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def g (x : ℝ) : ℝ := f x - Real.log x

theorem g_lower_bound : ∀ x > 0, g x > 4/3 := by
  sorry

end NUMINAMATH_CALUDE_g_lower_bound_l1975_197531


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l1975_197512

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

theorem thirtieth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 7) (h₃ : a₃ = 11) :
  arithmetic_sequence a₁ (a₂ - a₁) 30 = 119 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l1975_197512


namespace NUMINAMATH_CALUDE_congruence_problem_l1975_197507

theorem congruence_problem (y : ℤ) 
  (h1 : (2 + y) % (2^3) = 2^3 % (2^3))
  (h2 : (4 + y) % (4^3) = 2^3 % (4^3))
  (h3 : (6 + y) % (6^3) = 2^3 % (6^3)) :
  y % 24 = 6 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1975_197507


namespace NUMINAMATH_CALUDE_gcd_count_for_360_l1975_197576

theorem gcd_count_for_360 (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 360) :
  ∃! (s : Finset ℕ+), (∀ x ∈ s, ∃ a b : ℕ+, Nat.gcd a b = x ∧ Nat.lcm a b * x = 360) ∧ s.card = 6 :=
sorry

end NUMINAMATH_CALUDE_gcd_count_for_360_l1975_197576


namespace NUMINAMATH_CALUDE_boys_camp_total_l1975_197594

theorem boys_camp_total (total_boys : ℕ) : 
  (total_boys : ℝ) * 0.2 * 0.7 = 21 → total_boys = 150 := by
  sorry

end NUMINAMATH_CALUDE_boys_camp_total_l1975_197594


namespace NUMINAMATH_CALUDE_wrong_mark_value_l1975_197567

/-- Proves that the wrongly entered mark is 73 given the conditions of the problem -/
theorem wrong_mark_value (correct_mark : ℕ) (class_size : ℕ) (average_increase : ℚ) 
  (h1 : correct_mark = 63)
  (h2 : class_size = 20)
  (h3 : average_increase = 1/2) :
  ∃ x : ℕ, x = 73 ∧ (x : ℚ) - correct_mark = class_size * average_increase := by
  sorry


end NUMINAMATH_CALUDE_wrong_mark_value_l1975_197567


namespace NUMINAMATH_CALUDE_largest_number_problem_l1975_197545

theorem largest_number_problem (a b c : ℝ) : 
  a < b → b < c → 
  a + b + c = 67 → 
  c - b = 7 → 
  b - a = 3 → 
  c = 28 := by
sorry

end NUMINAMATH_CALUDE_largest_number_problem_l1975_197545


namespace NUMINAMATH_CALUDE_mean_temperature_is_84_l1975_197515

def temperatures : List ℝ := [82, 84, 83, 85, 86]

theorem mean_temperature_is_84 :
  (temperatures.sum / temperatures.length) = 84 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_is_84_l1975_197515


namespace NUMINAMATH_CALUDE_sin_value_given_tan_and_range_l1975_197558

theorem sin_value_given_tan_and_range (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) (3 * π / 2)) 
  (h2 : Real.tan α = Real.sqrt 2) : 
  Real.sin α = -(Real.sqrt 6 / 3) := by
sorry

end NUMINAMATH_CALUDE_sin_value_given_tan_and_range_l1975_197558


namespace NUMINAMATH_CALUDE_exists_valid_heptagon_arrangement_l1975_197506

/-- Represents a heptagon with numbers placed in its vertices -/
def Heptagon := Fin 7 → Nat

/-- Checks if a given heptagon arrangement satisfies the sum condition -/
def is_valid_arrangement (h : Heptagon) : Prop :=
  (∀ i : Fin 7, h i ∈ Finset.range 15 \ {0}) ∧
  (∀ i : Fin 7, h i + h ((i + 1) % 7) + h ((i + 2) % 7) = 19)

/-- Theorem stating the existence of a valid heptagon arrangement -/
theorem exists_valid_heptagon_arrangement : ∃ h : Heptagon, is_valid_arrangement h :=
sorry

end NUMINAMATH_CALUDE_exists_valid_heptagon_arrangement_l1975_197506


namespace NUMINAMATH_CALUDE_moles_of_Cu_CN_2_formed_l1975_197503

/-- Represents a chemical species in a reaction -/
inductive Species
| HCN
| CuSO4
| Cu_CN_2
| H2SO4

/-- Represents the coefficients of a balanced chemical equation -/
structure BalancedEquation :=
(reactants : Species → ℕ)
(products : Species → ℕ)

/-- Represents the available moles of each species -/
structure AvailableMoles :=
(moles : Species → ℝ)

def reaction : BalancedEquation :=
{ reactants := λ s => match s with
  | Species.HCN => 2
  | Species.CuSO4 => 1
  | _ => 0
, products := λ s => match s with
  | Species.Cu_CN_2 => 1
  | Species.H2SO4 => 1
  | _ => 0
}

def available : AvailableMoles :=
{ moles := λ s => match s with
  | Species.HCN => 2
  | Species.CuSO4 => 1
  | _ => 0
}

/-- Calculates the moles of product formed based on the limiting reactant -/
def moles_of_product (eq : BalancedEquation) (avail : AvailableMoles) (product : Species) : ℝ :=
sorry

theorem moles_of_Cu_CN_2_formed :
  moles_of_product reaction available Species.Cu_CN_2 = 1 :=
sorry

end NUMINAMATH_CALUDE_moles_of_Cu_CN_2_formed_l1975_197503


namespace NUMINAMATH_CALUDE_car_profit_percentage_l1975_197573

/-- Calculates the profit percentage on the original price of a car, given specific buying and selling conditions. -/
theorem car_profit_percentage (P : ℝ) (h : P > 0) : 
  let purchase_price := 0.95 * P
  let taxes := 0.03 * P
  let maintenance := 0.02 * P
  let total_cost := purchase_price + taxes + maintenance
  let selling_price := purchase_price * 1.6
  let profit := selling_price - total_cost
  let profit_percentage := (profit / P) * 100
  profit_percentage = 52 := by sorry

end NUMINAMATH_CALUDE_car_profit_percentage_l1975_197573


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1975_197572

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                                    a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 510 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1975_197572


namespace NUMINAMATH_CALUDE_contrapositive_evenness_l1975_197593

theorem contrapositive_evenness (a b : ℤ) : 
  (Odd (a + b) → Odd a ∨ Odd b) = False :=
sorry

end NUMINAMATH_CALUDE_contrapositive_evenness_l1975_197593


namespace NUMINAMATH_CALUDE_total_spent_on_flowers_l1975_197554

def roses_quantity : ℕ := 5
def roses_price : ℕ := 6
def daisies_quantity : ℕ := 3
def daisies_price : ℕ := 4
def tulips_quantity : ℕ := 2
def tulips_price : ℕ := 5

theorem total_spent_on_flowers :
  roses_quantity * roses_price +
  daisies_quantity * daisies_price +
  tulips_quantity * tulips_price = 52 := by
sorry

end NUMINAMATH_CALUDE_total_spent_on_flowers_l1975_197554


namespace NUMINAMATH_CALUDE_min_a₄_is_27_l1975_197525

/-- A geometric progression of four positive integers -/
structure GeometricProgression :=
  (a₁ a₂ a₃ a₄ : ℕ+)
  (ratio : ℚ)
  (is_gp : a₂ = a₁ * ratio ∧ a₃ = a₂ * ratio ∧ a₄ = a₃ * ratio)
  (ratio_gt_one : ratio > 1)
  (ratio_not_int : ∀ n : ℤ, ratio ≠ n)

/-- The minimum possible value of a₄ in a geometric progression of positive integers with ratio > 1 and not an integer is 27 -/
theorem min_a₄_is_27 : ∀ gp : GeometricProgression, gp.a₄ ≥ 27 :=
sorry

end NUMINAMATH_CALUDE_min_a₄_is_27_l1975_197525


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l1975_197534

theorem digit_sum_puzzle (c o u n t s : ℕ) : 
  c ≠ 0 → o ≠ 0 → u ≠ 0 → n ≠ 0 → t ≠ 0 → s ≠ 0 →
  c + o = u →
  u + n = t →
  t + c = s →
  o + n + s = 18 →
  t = 9 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l1975_197534


namespace NUMINAMATH_CALUDE_cubic_factorization_l1975_197500

theorem cubic_factorization (x : ℝ) : x^3 - 9*x = x*(x + 3)*(x - 3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1975_197500


namespace NUMINAMATH_CALUDE_paving_rate_calculation_l1975_197528

/-- Given a rectangular room with length and width, and the total cost of paving,
    calculate the rate of paving per square meter. -/
theorem paving_rate_calculation 
  (length width total_cost : ℝ) 
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 16500) : 
  total_cost / (length * width) = 800 := by
  sorry

end NUMINAMATH_CALUDE_paving_rate_calculation_l1975_197528


namespace NUMINAMATH_CALUDE_least_number_of_cans_l1975_197552

theorem least_number_of_cans (maaza pepsi sprite : ℕ) 
  (h_maaza : maaza = 10)
  (h_pepsi : pepsi = 144)
  (h_sprite : sprite = 368) :
  let gcd_all := Nat.gcd maaza (Nat.gcd pepsi sprite)
  ∃ (can_size : ℕ), 
    can_size = gcd_all ∧ 
    can_size > 0 ∧
    maaza % can_size = 0 ∧ 
    pepsi % can_size = 0 ∧ 
    sprite % can_size = 0 ∧
    (maaza / can_size + pepsi / can_size + sprite / can_size) = 261 :=
by sorry

end NUMINAMATH_CALUDE_least_number_of_cans_l1975_197552


namespace NUMINAMATH_CALUDE_translation_result_l1975_197542

def translate_point (x y dx dy : ℝ) : ℝ × ℝ :=
  (x + dx, y + dy)

theorem translation_result :
  let P : ℝ × ℝ := (-2, 1)
  let dx : ℝ := 3
  let dy : ℝ := 4
  let P' : ℝ × ℝ := translate_point P.1 P.2 dx dy
  P' = (1, 5) := by sorry

end NUMINAMATH_CALUDE_translation_result_l1975_197542


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l1975_197537

theorem binomial_coefficient_problem (m : ℝ) (n : ℕ) :
  (∃ k : ℕ, (Nat.choose n 3) * m^3 = 160 ∧ n = 6) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l1975_197537


namespace NUMINAMATH_CALUDE_arithmetic_sequences_bound_l1975_197533

theorem arithmetic_sequences_bound (n k b : ℕ) (d₁ d₂ : ℤ) :
  0 < b → b < n →
  (∀ i j, i ≠ j → i ≤ n → j ≤ n → ∃ (x y : ℤ), x ≠ y ∧ 
    (∃ (a r : ℤ), x = a + r * (if i ≤ b then d₁ else d₂) ∧
                  y = a + r * (if i ≤ b then d₁ else d₂)) ∧
    (∃ (a r : ℤ), x = a + r * (if j ≤ b then d₁ else d₂) ∧
                  y = a + r * (if j ≤ b then d₁ else d₂))) →
  b ≤ 2 * (k - d₂ / Int.gcd d₁ d₂) - 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_bound_l1975_197533


namespace NUMINAMATH_CALUDE_probability_of_sunflower_seed_l1975_197557

def sunflower_seeds : ℕ := 2
def green_bean_seeds : ℕ := 3
def pumpkin_seeds : ℕ := 4

def total_seeds : ℕ := sunflower_seeds + green_bean_seeds + pumpkin_seeds

theorem probability_of_sunflower_seed :
  (sunflower_seeds : ℚ) / total_seeds = 2 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_of_sunflower_seed_l1975_197557


namespace NUMINAMATH_CALUDE_cube_volume_problem_l1975_197547

theorem cube_volume_problem (reference_cube_volume : ℝ) 
  (surface_area_ratio : ℝ) (target_cube_volume : ℝ) : 
  reference_cube_volume = 8 →
  surface_area_ratio = 3 →
  (6 * (reference_cube_volume ^ (1/3))^2) * surface_area_ratio = 6 * (target_cube_volume ^ (1/3))^2 →
  target_cube_volume = 24 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l1975_197547


namespace NUMINAMATH_CALUDE_circle_line_intersection_l1975_197510

/-- The equation of circle C is x^2 + y^2 + 8x + 15 = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 8*x + 15 = 0

/-- The equation of the line is y = kx - 2 -/
def line (k x y : ℝ) : Prop := y = k*x - 2

/-- A point (x, y) is on the line y = kx - 2 -/
def point_on_line (k x y : ℝ) : Prop := line k x y

/-- The distance between two points (x1, y1) and (x2, y2) is less than or equal to r -/
def distance_le (x1 y1 x2 y2 r : ℝ) : Prop :=
  (x1 - x2)^2 + (y1 - y2)^2 ≤ r^2

theorem circle_line_intersection (k : ℝ) : 
  (∃ x y : ℝ, point_on_line k x y ∧ 
    (∃ x0 y0 : ℝ, circle_C x0 y0 ∧ distance_le x y x0 y0 1)) →
  -4/3 ≤ k ∧ k ≤ 0 := by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l1975_197510


namespace NUMINAMATH_CALUDE_marathon_remainder_l1975_197595

/-- Represents the distance of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents a runner's total distance in miles and yards -/
structure TotalDistance where
  miles : ℕ
  yards : ℕ

/-- Converts a given number of yards to miles and remaining yards -/
def yardsToMilesAndYards (totalYards : ℕ) : TotalDistance :=
  { miles := totalYards / 1760,
    yards := totalYards % 1760 }

theorem marathon_remainder (marathonDistance : Marathon) (numMarathons : ℕ) : 
  marathonDistance.miles = 26 →
  marathonDistance.yards = 395 →
  numMarathons = 15 →
  (yardsToMilesAndYards (numMarathons * (marathonDistance.miles * 1760 + marathonDistance.yards))).yards = 645 := by
  sorry

#check marathon_remainder

end NUMINAMATH_CALUDE_marathon_remainder_l1975_197595


namespace NUMINAMATH_CALUDE_point_coordinates_l1975_197518

/-- Given point M(5, -6) and vector a = (1, -2), if NM = 3a, then N has coordinates (2, 0) -/
theorem point_coordinates (M N : ℝ × ℝ) (a : ℝ × ℝ) : 
  M = (5, -6) → 
  a = (1, -2) → 
  N.1 - M.1 = 3 * a.1 ∧ N.2 - M.2 = 3 * a.2 → 
  N = (2, 0) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1975_197518


namespace NUMINAMATH_CALUDE_visits_needed_is_eleven_l1975_197539

/-- The cost of headphones in rubles -/
def headphones_cost : ℕ := 275

/-- The cost of a combined pool and sauna visit in rubles -/
def combined_cost : ℕ := 250

/-- The difference in cost between pool-only and sauna-only visits in rubles -/
def pool_sauna_diff : ℕ := 200

/-- Calculates the number of pool-only visits needed to save for headphones -/
def visits_needed : ℕ :=
  let sauna_cost := (combined_cost - pool_sauna_diff) / 2
  let pool_only_cost := sauna_cost + pool_sauna_diff
  let savings_per_visit := combined_cost - pool_only_cost
  (headphones_cost + savings_per_visit - 1) / savings_per_visit

theorem visits_needed_is_eleven : visits_needed = 11 := by
  sorry

end NUMINAMATH_CALUDE_visits_needed_is_eleven_l1975_197539


namespace NUMINAMATH_CALUDE_boys_fraction_l1975_197521

/-- In a class with boys and girls, prove that the fraction of boys is 2/3 given the conditions. -/
theorem boys_fraction (tall_boys : ℕ) (total_boys : ℕ) (girls : ℕ) 
  (h1 : tall_boys = 18)
  (h2 : tall_boys = 3 * total_boys / 4)
  (h3 : girls = 12) :
  total_boys / (total_boys + girls) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_boys_fraction_l1975_197521


namespace NUMINAMATH_CALUDE_sector_area_l1975_197562

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 10) (h2 : θ = 42) :
  (θ / 360) * π * r^2 = 35 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1975_197562


namespace NUMINAMATH_CALUDE_room_occupancy_l1975_197589

theorem room_occupancy (x y : ℕ) : 
  x + y = 76 → 
  x - 30 = y - 40 → 
  (x = 33 ∧ y = 43) ∨ (x = 43 ∧ y = 33) :=
by sorry

end NUMINAMATH_CALUDE_room_occupancy_l1975_197589


namespace NUMINAMATH_CALUDE_aleesia_weight_loss_weeks_aleesia_weight_loss_weeks_proof_l1975_197548

theorem aleesia_weight_loss_weeks : ℝ → Prop :=
  fun w =>
    let aleesia_weekly_loss : ℝ := 1.5
    let alexei_weekly_loss : ℝ := 2.5
    let alexei_weeks : ℝ := 8
    let total_loss : ℝ := 35
    (aleesia_weekly_loss * w + alexei_weekly_loss * alexei_weeks = total_loss) →
    w = 10

-- The proof would go here
theorem aleesia_weight_loss_weeks_proof : aleesia_weight_loss_weeks 10 := by
  sorry

end NUMINAMATH_CALUDE_aleesia_weight_loss_weeks_aleesia_weight_loss_weeks_proof_l1975_197548


namespace NUMINAMATH_CALUDE_average_salary_after_bonuses_and_taxes_l1975_197584

def employee_salary (name : Char) : ℕ :=
  match name with
  | 'A' => 8000
  | 'B' => 5000
  | 'C' => 11000
  | 'D' => 7000
  | 'E' => 9000
  | 'F' => 6000
  | 'G' => 10000
  | _ => 0

def apply_bonus_or_tax (salary : ℕ) (rate : ℚ) (is_bonus : Bool) : ℚ :=
  if is_bonus then
    salary + salary * rate
  else
    salary - salary * rate

def final_salary (name : Char) : ℚ :=
  match name with
  | 'A' => apply_bonus_or_tax (employee_salary 'A') (1/10) true
  | 'B' => apply_bonus_or_tax (employee_salary 'B') (1/20) false
  | 'C' => employee_salary 'C'
  | 'D' => apply_bonus_or_tax (employee_salary 'D') (1/20) false
  | 'E' => apply_bonus_or_tax (employee_salary 'E') (3/100) false
  | 'F' => apply_bonus_or_tax (employee_salary 'F') (1/20) false
  | 'G' => apply_bonus_or_tax (employee_salary 'G') (3/40) true
  | _ => 0

def total_final_salaries : ℚ :=
  (final_salary 'A') + (final_salary 'B') + (final_salary 'C') +
  (final_salary 'D') + (final_salary 'E') + (final_salary 'F') +
  (final_salary 'G')

def number_of_employees : ℕ := 7

theorem average_salary_after_bonuses_and_taxes :
  (total_final_salaries / number_of_employees) = 8054.29 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_after_bonuses_and_taxes_l1975_197584


namespace NUMINAMATH_CALUDE_min_distance_circle_line_l1975_197579

/-- The minimum distance between a circle and a line --/
theorem min_distance_circle_line :
  let circle := {p : ℝ × ℝ | (p.1 + 2)^2 + p.2^2 = 4}
  let line := {p : ℝ × ℝ | p.1 + p.2 = 4}
  (∀ p ∈ circle, ∀ q ∈ line, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ 3 * Real.sqrt 2 - 2) ∧
  (∃ p ∈ circle, ∃ q ∈ line, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 3 * Real.sqrt 2 - 2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_line_l1975_197579


namespace NUMINAMATH_CALUDE_remainder_three_power_2010_mod_8_l1975_197575

theorem remainder_three_power_2010_mod_8 : 3^2010 % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_power_2010_mod_8_l1975_197575


namespace NUMINAMATH_CALUDE_parallelogram_d_coordinates_l1975_197564

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a vector in 2D space
structure Vector2D where
  x : ℝ
  y : ℝ

-- Define a parallelogram
structure Parallelogram where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

def vector_between_points (p1 p2 : Point2D) : Vector2D :=
  { x := p2.x - p1.x, y := p2.y - p1.y }

theorem parallelogram_d_coordinates :
  ∀ (ABCD : Parallelogram),
    ABCD.A = { x := 1, y := 2 } →
    ABCD.B = { x := -2, y := 0 } →
    vector_between_points ABCD.A ABCD.C = { x := 2, y := -3 } →
    ABCD.D = { x := 6, y := 1 } :=
by
  sorry


end NUMINAMATH_CALUDE_parallelogram_d_coordinates_l1975_197564


namespace NUMINAMATH_CALUDE_water_mixture_percentage_l1975_197586

/-- Proves that adding a specific amount of water to a given mixture results in a new mixture with the expected water percentage. -/
theorem water_mixture_percentage 
  (initial_volume : ℝ) 
  (initial_water_percentage : ℝ) 
  (added_water : ℝ) 
  (h1 : initial_volume = 200)
  (h2 : initial_water_percentage = 20)
  (h3 : added_water = 13.333333333333334)
  : (initial_water_percentage / 100 * initial_volume + added_water) / (initial_volume + added_water) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_water_mixture_percentage_l1975_197586


namespace NUMINAMATH_CALUDE_runners_meet_time_l1975_197563

/-- The length of the circular track in meters -/
def track_length : ℝ := 400

/-- The speeds of the three runners in meters per second -/
def runner_speeds : Fin 3 → ℝ
  | 0 => 5
  | 1 => 5.5
  | 2 => 6

/-- The time in seconds for the runners to meet again at the starting point -/
def meeting_time : ℝ := 800

theorem runners_meet_time :
  ∀ (i : Fin 3), ∃ (n : ℕ), (runner_speeds i * meeting_time) = n * track_length :=
sorry

end NUMINAMATH_CALUDE_runners_meet_time_l1975_197563


namespace NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l1975_197591

/-- Given a geometric sequence with positive terms where a₂, ½a₃, and a₁ form an arithmetic sequence,
    the ratio (a₄ + a₅)/(a₃ + a₄) equals (1 + √5)/2. -/
theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_arith : a 2 - a 1 = (1/2 : ℝ) * a 3 - a 2) :
  (a 4 + a 5) / (a 3 + a 4) = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l1975_197591


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1975_197587

theorem simplify_fraction_product : 
  4 * (18 / 5) * (25 / -45) * (10 / 8) = -10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1975_197587


namespace NUMINAMATH_CALUDE_thomas_total_training_hours_l1975_197561

/-- Calculates the total training hours for Thomas given his training schedule --/
def total_training_hours : ℕ :=
  let first_phase_days : ℕ := 15
  let first_phase_hours_per_day : ℕ := 5
  let second_phase_days : ℕ := 15
  let second_phase_rest_days : ℕ := 3
  let third_phase_days : ℕ := 12
  let third_phase_rest_days : ℕ := 2
  let new_schedule_morning_hours : ℕ := 4
  let new_schedule_evening_hours : ℕ := 3

  let first_phase_total := first_phase_days * first_phase_hours_per_day
  let second_phase_total := (second_phase_days - second_phase_rest_days) * (new_schedule_morning_hours + new_schedule_evening_hours)
  let third_phase_total := (third_phase_days - third_phase_rest_days) * (new_schedule_morning_hours + new_schedule_evening_hours)

  first_phase_total + second_phase_total + third_phase_total

/-- Theorem stating that Thomas' total training hours is 229 --/
theorem thomas_total_training_hours :
  total_training_hours = 229 := by
  sorry

end NUMINAMATH_CALUDE_thomas_total_training_hours_l1975_197561


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1975_197516

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

theorem intersection_complement_equality : B ∩ (U \ A) = {3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1975_197516


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1975_197569

theorem matrix_equation_solution :
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 4]
  N^4 - 5 • N^3 + 9 • N^2 - 5 • N = !![6, 12; 3, 6] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l1975_197569


namespace NUMINAMATH_CALUDE_negation_of_implication_l1975_197597

theorem negation_of_implication (p q : Prop) : 
  ¬(p → q) ↔ p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1975_197597


namespace NUMINAMATH_CALUDE_triangle_OAB_and_point_C_l1975_197549

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, -1)
def B : ℝ × ℝ := (1, 2)

-- Define the area of a triangle given three points
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define perpendicularity of two vectors
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := sorry

-- Define parallelism of two vectors
def parallel (v1 v2 : ℝ × ℝ) : Prop := sorry

-- Define the vector between two points
def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem triangle_OAB_and_point_C :
  -- Part 1: Area of triangle OAB
  triangle_area O A B = 5/2 ∧
  -- Part 2: Coordinates of point C
  ∃ C : ℝ × ℝ,
    perpendicular (vector B C) (vector A B) ∧
    parallel (vector A C) (vector O B) ∧
    C = (4, 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_OAB_and_point_C_l1975_197549


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1975_197524

-- Define the propositions
variable (G : Prop) -- G represents "The goods are of high quality"
variable (C : Prop) -- C represents "The price is cheap"

-- State the theorem
theorem contrapositive_equivalence : (G → ¬C) ↔ (C → ¬G) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1975_197524


namespace NUMINAMATH_CALUDE_angle_A_measure_l1975_197581

/-- Given a geometric figure with the following properties:
  - Angle B measures 120°
  - A line divides the space opposite angle B on a straight line into two angles
  - One of these angles measures 50°
  - Angle A is vertically opposite to the angle that is not 50°
  Prove that angle A measures 130° -/
theorem angle_A_measure (B : ℝ) (angle1 : ℝ) (angle2 : ℝ) (A : ℝ) 
  (h1 : B = 120)
  (h2 : angle1 + angle2 = 180 - B)
  (h3 : angle1 = 50)
  (h4 : A = 180 - angle2) :
  A = 130 := by sorry

end NUMINAMATH_CALUDE_angle_A_measure_l1975_197581


namespace NUMINAMATH_CALUDE_f_2009_equals_one_l1975_197502

-- Define the function f
axiom f : ℝ → ℝ

-- Define the conditions
axiom func_prop : ∀ x y : ℝ, f (x * y) = f x * f y
axiom f0_nonzero : f 0 ≠ 0

-- State the theorem
theorem f_2009_equals_one : f 2009 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_2009_equals_one_l1975_197502


namespace NUMINAMATH_CALUDE_households_with_appliances_l1975_197553

theorem households_with_appliances (total : ℕ) (tv : ℕ) (fridge : ℕ) (both : ℕ) :
  total = 100 →
  tv = 65 →
  fridge = 84 →
  both = 53 →
  tv + fridge - both = 96 := by
  sorry

end NUMINAMATH_CALUDE_households_with_appliances_l1975_197553


namespace NUMINAMATH_CALUDE_w_to_twelve_power_l1975_197529

theorem w_to_twelve_power (w : ℂ) (h : w = (-Real.sqrt 3 + Complex.I) / 3) :
  w^12 = 400 / 531441 := by
  sorry

end NUMINAMATH_CALUDE_w_to_twelve_power_l1975_197529


namespace NUMINAMATH_CALUDE_not_kth_power_consecutive_product_l1975_197519

theorem not_kth_power_consecutive_product (m k : ℕ) (hk : k > 1) :
  ¬ ∃ (a : ℤ), m * (m + 1) = a^k := by
  sorry

end NUMINAMATH_CALUDE_not_kth_power_consecutive_product_l1975_197519


namespace NUMINAMATH_CALUDE_total_distance_eight_points_circle_l1975_197535

/-- The total distance traveled by 8 points on a circle visiting non-adjacent points -/
theorem total_distance_eight_points_circle (r : ℝ) (h : r = 40) :
  let n := 8
  let distance_two_apart := r * Real.sqrt 2
  let distance_three_apart := r * Real.sqrt (2 + Real.sqrt 2)
  let distance_four_apart := 2 * r
  let single_point_distance := 4 * distance_two_apart + 2 * distance_three_apart + distance_four_apart
  n * single_point_distance = 1280 * Real.sqrt 2 + 640 * Real.sqrt (2 + Real.sqrt 2) + 640 :=
by sorry

end NUMINAMATH_CALUDE_total_distance_eight_points_circle_l1975_197535


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1975_197514

theorem arithmetic_sequence_ratio (x y d₁ d₂ : ℝ) (h₁ : d₁ ≠ 0) (h₂ : d₂ ≠ 0) 
  (h₃ : x + 4 * d₁ = y) (h₄ : x + 5 * d₂ = y) : d₁ / d₂ = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1975_197514


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1975_197538

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- If 2S₃ = 3S₂ + 6 for an arithmetic sequence, then its common difference is 2 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : 
  seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1975_197538


namespace NUMINAMATH_CALUDE_mason_bricks_used_l1975_197523

/-- Calculates the total number of bricks used by a mason given the following conditions:
  * The mason needs to build 6 courses per wall
  * Each course has 10 bricks
  * He needs to build 4 walls
  * He can't finish two courses of the last wall due to lack of bricks
-/
def total_bricks_used (courses_per_wall : ℕ) (bricks_per_course : ℕ) (total_walls : ℕ) (unfinished_courses : ℕ) : ℕ :=
  let complete_walls := total_walls - 1
  let complete_wall_bricks := courses_per_wall * bricks_per_course * complete_walls
  let incomplete_wall_bricks := (courses_per_wall - unfinished_courses) * bricks_per_course
  complete_wall_bricks + incomplete_wall_bricks

theorem mason_bricks_used :
  total_bricks_used 6 10 4 2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_mason_bricks_used_l1975_197523


namespace NUMINAMATH_CALUDE_circle_center_distance_l1975_197508

/-- The distance between the center of the circle x²+y²=4x+6y+3 and the point (8,4) is √37 -/
theorem circle_center_distance : ∃ (h k : ℝ),
  (∀ x y : ℝ, x^2 + y^2 = 4*x + 6*y + 3 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 3)) ∧
  Real.sqrt ((8 - h)^2 + (4 - k)^2) = Real.sqrt 37 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_distance_l1975_197508


namespace NUMINAMATH_CALUDE_brass_price_is_correct_l1975_197577

/-- The price of copper in dollars per pound -/
def copper_price : ℚ := 65 / 100

/-- The price of zinc in dollars per pound -/
def zinc_price : ℚ := 30 / 100

/-- The total weight of brass in pounds -/
def total_weight : ℚ := 70

/-- The amount of copper used in pounds -/
def copper_weight : ℚ := 30

/-- The amount of zinc used in pounds -/
def zinc_weight : ℚ := total_weight - copper_weight

/-- The selling price of brass per pound -/
def brass_price : ℚ := (copper_price * copper_weight + zinc_price * zinc_weight) / total_weight

theorem brass_price_is_correct : brass_price = 45 / 100 := by
  sorry

end NUMINAMATH_CALUDE_brass_price_is_correct_l1975_197577


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l1975_197541

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (c : Line) (α β : Plane) :
  parallel c α → perpendicular c β → planes_perpendicular α β :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l1975_197541


namespace NUMINAMATH_CALUDE_square_side_length_l1975_197574

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 1 / 9 → side ^ 2 = area → side = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1975_197574


namespace NUMINAMATH_CALUDE_fraction_equals_98_when_x_is_3_l1975_197583

theorem fraction_equals_98_when_x_is_3 :
  let x : ℝ := 3
  (x^8 + 16*x^4 + 64 + 2*x^2) / (x^4 + 8 + x^2) = 98 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_98_when_x_is_3_l1975_197583


namespace NUMINAMATH_CALUDE_jana_kelly_height_difference_l1975_197588

/-- Proves that Jana is 5 inches taller than Kelly given the heights of Jess and Jana, and the height difference between Jess and Kelly. -/
theorem jana_kelly_height_difference :
  ∀ (jess_height jana_height kelly_height : ℕ),
    jess_height = 72 →
    jana_height = 74 →
    kelly_height = jess_height - 3 →
    jana_height - kelly_height = 5 := by
  sorry

end NUMINAMATH_CALUDE_jana_kelly_height_difference_l1975_197588


namespace NUMINAMATH_CALUDE_final_position_l1975_197543

/-- A point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally -/
def translateRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

/-- Reflect a point about the origin -/
def reflectOrigin (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- The theorem stating the final position of the point after translation and reflection -/
theorem final_position :
  let initial := Point.mk 3 2
  let translated := translateRight initial 2
  let final := reflectOrigin translated
  final = Point.mk (-5) (-2) := by sorry

end NUMINAMATH_CALUDE_final_position_l1975_197543


namespace NUMINAMATH_CALUDE_car_initial_speed_l1975_197599

/-- Represents a point on the road --/
inductive Point
| A
| B
| C
| D

/-- Represents the speed of the car at different segments --/
structure Speed where
  initial : ℝ
  fromBtoC : ℝ
  fromCtoD : ℝ

/-- Represents the distance between points --/
structure Distance where
  total : ℝ
  AtoB : ℝ
  BtoC : ℝ
  CtoD : ℝ

/-- Represents the travel time between points --/
structure TravelTime where
  BtoC : ℝ
  CtoD : ℝ

/-- The main theorem stating the conditions and the result to be proved --/
theorem car_initial_speed 
  (d : Distance)
  (s : Speed)
  (t : TravelTime)
  (h1 : d.total = 100)
  (h2 : d.total - d.AtoB = 0.5 * s.initial)
  (h3 : s.fromBtoC = s.initial - 10)
  (h4 : s.fromCtoD = s.initial - 20)
  (h5 : d.CtoD = 20)
  (h6 : t.BtoC = t.CtoD + 1/12)
  (h7 : d.BtoC / s.fromBtoC = t.BtoC)
  (h8 : d.CtoD / s.fromCtoD = t.CtoD)
  : s.initial = 100 := by
  sorry


end NUMINAMATH_CALUDE_car_initial_speed_l1975_197599


namespace NUMINAMATH_CALUDE_triathlon_speed_l1975_197532

/-- Triathlon problem -/
theorem triathlon_speed (total_time swim_dist swim_speed run_dist run_speed rest_time bike_dist : ℝ) 
  (h_total : total_time = 2.25)
  (h_swim : swim_dist = 0.5)
  (h_swim_speed : swim_speed = 2)
  (h_run : run_dist = 4)
  (h_run_speed : run_speed = 8)
  (h_rest : rest_time = 1/6)
  (h_bike : bike_dist = 20) :
  bike_dist / (total_time - (swim_dist / swim_speed + run_dist / run_speed + rest_time)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_triathlon_speed_l1975_197532


namespace NUMINAMATH_CALUDE_tank_capacity_l1975_197568

theorem tank_capacity (initial_fill : Real) (added_amount : Real) (final_fill : Real) :
  initial_fill = 3/4 →
  added_amount = 4 →
  final_fill = 9/10 →
  ∃ (capacity : Real), capacity = 80/3 ∧
    initial_fill * capacity + added_amount = final_fill * capacity :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l1975_197568


namespace NUMINAMATH_CALUDE_solve_equation_l1975_197530

-- Define the operation "*"
def star (a b : ℝ) : ℝ := 2 * a - b

-- Theorem statement
theorem solve_equation (x : ℝ) (h : star x (star 2 1) = 3) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1975_197530


namespace NUMINAMATH_CALUDE_lifting_capacity_proof_l1975_197526

/-- Calculates the total weight a person can lift with both hands after training and specializing,
    given their initial lifting capacity per hand. -/
def totalLiftingCapacity (initialCapacity : ℝ) : ℝ :=
  let doubledCapacity := initialCapacity * 2
  let specializedCapacity := doubledCapacity * 1.1
  specializedCapacity * 2

/-- Proves that given an initial lifting capacity of 80 kg per hand,
    the total weight that can be lifted with both hands after training and specializing is 352 kg. -/
theorem lifting_capacity_proof :
  totalLiftingCapacity 80 = 352 := by
  sorry

#eval totalLiftingCapacity 80

end NUMINAMATH_CALUDE_lifting_capacity_proof_l1975_197526


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1975_197585

-- Define the line equation
def line_equation (m x y : ℝ) : Prop :=
  m * x - y + 2 * m + 1 = 0

-- Theorem statement
theorem fixed_point_on_line :
  ∀ m : ℝ, line_equation m (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1975_197585


namespace NUMINAMATH_CALUDE_same_gender_leaders_count_l1975_197509

/-- Represents the number of ways to select a captain and co-captain of the same gender
    from a team with an equal number of men and women. -/
def select_same_gender_leaders (team_size : ℕ) : ℕ :=
  2 * (team_size * (team_size - 1))

/-- Theorem: In a team of 12 men and 12 women, there are 264 ways to select
    a captain and co-captain of the same gender. -/
theorem same_gender_leaders_count :
  select_same_gender_leaders 12 = 264 := by
  sorry

#eval select_same_gender_leaders 12

end NUMINAMATH_CALUDE_same_gender_leaders_count_l1975_197509


namespace NUMINAMATH_CALUDE_distance_is_134_div_7_l1975_197513

/-- The distance from a point to a plane defined by three points -/
def distance_point_to_plane (M₀ M₁ M₂ M₃ : ℝ × ℝ × ℝ) : ℝ := sorry

/-- The points given in the problem -/
def M₀ : ℝ × ℝ × ℝ := (-13, -8, 16)
def M₁ : ℝ × ℝ × ℝ := (1, 2, 0)
def M₂ : ℝ × ℝ × ℝ := (3, 0, -3)
def M₃ : ℝ × ℝ × ℝ := (5, 2, 6)

/-- The theorem stating that the distance is equal to 134/7 -/
theorem distance_is_134_div_7 : distance_point_to_plane M₀ M₁ M₂ M₃ = 134 / 7 := by sorry

end NUMINAMATH_CALUDE_distance_is_134_div_7_l1975_197513


namespace NUMINAMATH_CALUDE_smallest_number_is_57_l1975_197555

theorem smallest_number_is_57 (a b c d : ℕ) 
  (sum_abc : a + b + c = 234)
  (sum_abd : a + b + d = 251)
  (sum_acd : a + c + d = 284)
  (sum_bcd : b + c + d = 299) :
  min a (min b (min c d)) = 57 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_is_57_l1975_197555


namespace NUMINAMATH_CALUDE_expand_product_l1975_197511

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1975_197511


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1975_197501

theorem coin_flip_probability : 
  let n : ℕ := 5  -- number of coins
  let k : ℕ := 3  -- minimum number of heads we're interested in
  let total_outcomes : ℕ := 2^n  -- total number of possible outcomes
  let favorable_outcomes : ℕ := (Finset.range (n - k + 1)).sum (λ i => Nat.choose n (k + i))
  (favorable_outcomes : ℚ) / total_outcomes = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1975_197501


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1975_197565

/-- A curve y = sin(x + φ) is symmetric about the y-axis if and only if sin(x + φ) = sin(-x + φ) for all x ∈ ℝ -/
def symmetric_about_y_axis (φ : ℝ) : Prop :=
  ∀ x : ℝ, Real.sin (x + φ) = Real.sin (-x + φ)

/-- φ = π/2 is a sufficient condition for y = sin(x + φ) to be symmetric about the y-axis -/
theorem sufficient_condition (φ : ℝ) (h : φ = π / 2) : symmetric_about_y_axis φ := by
  sorry

/-- φ = π/2 is not a necessary condition for y = sin(x + φ) to be symmetric about the y-axis -/
theorem not_necessary_condition : ∃ φ : ℝ, φ ≠ π / 2 ∧ symmetric_about_y_axis φ := by
  sorry

/-- φ = π/2 is a sufficient but not necessary condition for y = sin(x + φ) to be symmetric about the y-axis -/
theorem sufficient_but_not_necessary : 
  (∀ φ : ℝ, φ = π / 2 → symmetric_about_y_axis φ) ∧ 
  (∃ φ : ℝ, φ ≠ π / 2 ∧ symmetric_about_y_axis φ) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1975_197565


namespace NUMINAMATH_CALUDE_fib_80_mod_7_l1975_197596

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The period of the Fibonacci sequence modulo 7 -/
def fib_mod7_period : ℕ := 16

theorem fib_80_mod_7 :
  fib 80 % 7 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_fib_80_mod_7_l1975_197596
