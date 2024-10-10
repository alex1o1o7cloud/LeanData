import Mathlib

namespace modular_exponentiation_difference_l1250_125012

theorem modular_exponentiation_difference (n : ℕ) :
  (51 : ℤ) ^ n - (9 : ℤ) ^ n ≡ 0 [ZMOD 6] :=
by
  sorry

end modular_exponentiation_difference_l1250_125012


namespace trig_identity_l1250_125076

theorem trig_identity (α φ : ℝ) : 
  4 * Real.cos α * Real.cos φ * Real.cos (α - φ) - 
  2 * (Real.cos (α - φ))^2 - Real.cos (2 * φ) = 
  Real.cos (2 * α) := by
  sorry

end trig_identity_l1250_125076


namespace perpendicular_vector_equation_l1250_125063

/-- Given two vectors a and b in ℝ², find the value of t such that a is perpendicular to (t * a + b) -/
theorem perpendicular_vector_equation (a b : ℝ × ℝ) (h : a = (1, 2) ∧ b = (4, 3)) :
  ∃ t : ℝ, a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0 ∧ t = -2 := by
  sorry

#check perpendicular_vector_equation

end perpendicular_vector_equation_l1250_125063


namespace intersection_point_coordinates_l1250_125023

/-- Given points A, B, C, and O in a 2D plane, prove that the intersection point P
    of line segments AC and OB has coordinates (3, 3) -/
theorem intersection_point_coordinates :
  let A : Fin 2 → ℝ := ![4, 0]
  let B : Fin 2 → ℝ := ![4, 4]
  let C : Fin 2 → ℝ := ![2, 6]
  let O : Fin 2 → ℝ := ![0, 0]
  ∃ P : Fin 2 → ℝ,
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (fun i => t * (C i - A i) + A i)) ∧
    (∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ P = (fun i => s * (B i - O i) + O i)) ∧
    P = ![3, 3] :=
by sorry

end intersection_point_coordinates_l1250_125023


namespace inequality_problem_l1250_125070

theorem inequality_problem (a b : ℝ) (h : a ≠ b) :
  (a^2 + b^2 ≥ 2*(a - b - 1)) ∧
  ¬(∀ a b : ℝ, a + b > 2*b^2) ∧
  ¬(∀ a b : ℝ, a^5 + b^5 > a^3*b^2 + a^2*b^3) ∧
  ¬(∀ a b : ℝ, b/a + a/b > 2) :=
by sorry

end inequality_problem_l1250_125070


namespace parabola_points_l1250_125062

/-- A point on a parabola with equation y² = 4x that is 3 units away from its focus -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x
  distance_from_focus : (x - 1)^2 + y^2 = 3^2

/-- The theorem stating that (2, 2√2) and (2, -2√2) are the points on the parabola y² = 4x
    that are 3 units away from its focus -/
theorem parabola_points : 
  (∃ (p : ParabolaPoint), p.x = 2 ∧ p.y = 2 * Real.sqrt 2) ∧
  (∃ (p : ParabolaPoint), p.x = 2 ∧ p.y = -2 * Real.sqrt 2) :=
by sorry

end parabola_points_l1250_125062


namespace direction_vector_x_component_l1250_125013

/-- Given a line passing through two points with a specific direction vector form, prove the value of the direction vector's x-component. -/
theorem direction_vector_x_component
  (p1 : ℝ × ℝ)
  (p2 : ℝ × ℝ)
  (h1 : p1 = (-3, 6))
  (h2 : p2 = (2, -1))
  (h3 : ∃ (a : ℝ), (a, -1) = (p2.1 - p1.1, p2.2 - p1.2)) :
  ∃ (a : ℝ), (a, -1) = (p2.1 - p1.1, p2.2 - p1.2) ∧ a = -5/7 := by
sorry


end direction_vector_x_component_l1250_125013


namespace sum_of_abs_roots_is_122_l1250_125081

/-- Represents a cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ := 1
  b : ℤ := 0
  c : ℤ := -707
  d : ℤ

/-- Predicate to check if a given integer is a root of the polynomial -/
def is_root (p : CubicPolynomial) (x : ℤ) : Prop :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d = 0

/-- Theorem stating the sum of absolute values of roots -/
theorem sum_of_abs_roots_is_122 (m : ℤ) (p q r : ℤ) :
  let poly : CubicPolynomial := { d := m }
  (is_root poly p) ∧ (is_root poly q) ∧ (is_root poly r) →
  |p| + |q| + |r| = 122 := by
  sorry

end sum_of_abs_roots_is_122_l1250_125081


namespace five_fourths_of_eight_thirds_l1250_125031

theorem five_fourths_of_eight_thirds (x : ℚ) : x = 8/3 → (5/4) * x = 10/3 := by
  sorry

end five_fourths_of_eight_thirds_l1250_125031


namespace union_of_A_and_B_l1250_125086

def A : Set ℕ := {1, 2, 6}
def B : Set ℕ := {2, 3, 6}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 6} := by
  sorry

end union_of_A_and_B_l1250_125086


namespace tension_force_in_rod_system_l1250_125059

/-- The tension force in a weightless rod system with a suspended weight. -/
theorem tension_force_in_rod_system (m g : ℝ) (T₀ T₁ T₂ : ℝ) : 
  m = 2 →
  g = 10 →
  T₂ = 1/4 * m * g →
  T₁ = 3/4 * m * g →
  T₀ * (1/4) + T₂ = T₁ * (1/2) →
  T₀ = 10 := by sorry

end tension_force_in_rod_system_l1250_125059


namespace option_b_neither_parallel_nor_perpendicular_l1250_125072

/-- Two vectors in R³ -/
structure VectorPair where
  μ : Fin 3 → ℝ
  v : Fin 3 → ℝ

/-- Check if two vectors are parallel -/
def isParallel (pair : VectorPair) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i, pair.μ i = k * pair.v i)

/-- Check if two vectors are perpendicular -/
def isPerpendicular (pair : VectorPair) : Prop :=
  (pair.μ 0 * pair.v 0 + pair.μ 1 * pair.v 1 + pair.μ 2 * pair.v 2) = 0

/-- The specific vector pair for option B -/
def optionB : VectorPair where
  μ := ![3, 0, -1]
  v := ![0, 0, 2]

/-- Theorem stating that the vectors in option B are neither parallel nor perpendicular -/
theorem option_b_neither_parallel_nor_perpendicular :
  ¬(isParallel optionB) ∧ ¬(isPerpendicular optionB) := by
  sorry


end option_b_neither_parallel_nor_perpendicular_l1250_125072


namespace divisibility_by_seven_l1250_125043

theorem divisibility_by_seven (n : ℕ) : 
  7 ∣ n ↔ 7 ∣ ((n / 10) - 2 * (n % 10)) := by
  sorry

end divisibility_by_seven_l1250_125043


namespace machine_A_time_l1250_125087

/-- The time it takes for machines A, B, and C to finish a job together -/
def combined_time : ℝ := 2.181818181818182

/-- The time it takes for machine B to finish the job alone -/
def time_B : ℝ := 12

/-- The time it takes for machine C to finish the job alone -/
def time_C : ℝ := 8

/-- Theorem stating that if machines A, B, and C working together can finish a job in 
    2.181818181818182 hours, machine B alone takes 12 hours, and machine C alone takes 8 hours, 
    then machine A alone takes 4 hours to finish the job -/
theorem machine_A_time : 
  ∃ (time_A : ℝ), 
    1 / time_A + 1 / time_B + 1 / time_C = 1 / combined_time ∧ 
    time_A = 4 := by
  sorry

end machine_A_time_l1250_125087


namespace integer_average_l1250_125055

theorem integer_average (k m r s t : ℕ) : 
  0 < k ∧ k < m ∧ m < r ∧ r < s ∧ s < t ∧ 
  t = 40 ∧ 
  r ≤ 23 ∧ 
  ∀ (k' m' r' s' t' : ℕ), 
    (0 < k' ∧ k' < m' ∧ m' < r' ∧ r' < s' ∧ s' < t' ∧ t' = 40) → r' ≤ r →
  (k + m + r + s + t) / 5 = 18 := by
sorry

end integer_average_l1250_125055


namespace louisa_travel_l1250_125010

/-- Louisa's vacation travel problem -/
theorem louisa_travel (average_speed : ℝ) (second_day_distance : ℝ) (time_difference : ℝ) :
  average_speed = 33.333333333333336 →
  second_day_distance = 350 →
  time_difference = 3 →
  ∃ (first_day_distance : ℝ),
    first_day_distance = average_speed * (second_day_distance / average_speed - time_difference) ∧
    first_day_distance = 250 :=
by sorry

end louisa_travel_l1250_125010


namespace points_collinear_iff_b_eq_neg_one_over_44_l1250_125057

/-- Three points are collinear if and only if the slopes between any two pairs of points are equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- The theorem states that the given points are collinear if and only if b = -1/44. -/
theorem points_collinear_iff_b_eq_neg_one_over_44 :
  ∀ b : ℝ, collinear 4 (-6) (2*b + 1) 4 (-3*b + 2) 1 ↔ b = -1/44 :=
by sorry

end points_collinear_iff_b_eq_neg_one_over_44_l1250_125057


namespace simplify_and_rationalize_l1250_125021

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 4) * (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 8 / Real.sqrt 9) = Real.sqrt 5 / 3 := by
  sorry

end simplify_and_rationalize_l1250_125021


namespace tan_22_5_decomposition_l1250_125034

theorem tan_22_5_decomposition :
  ∃ (a b c d : ℕ+), 
    (Real.tan (22.5 * π / 180) = (a : ℝ).sqrt - (b : ℝ).sqrt + (c : ℝ).sqrt - (d : ℝ)) ∧
    a ≥ b ∧ b ≥ c ∧ c ≥ d ∧
    a + b + c + d = 3 := by sorry

end tan_22_5_decomposition_l1250_125034


namespace rogers_expenses_l1250_125061

theorem rogers_expenses (A : ℝ) (m s p : ℝ) : 
  (m = 0.25 * (A - s - p)) →
  (s = 0.1 * (A - m - p)) →
  (p = 0.05 * (A - m - s)) →
  (A > 0) →
  (m > 0) →
  (s > 0) →
  (p > 0) →
  (abs ((m + s + p) / A - 0.32) < 0.005) := by
sorry

end rogers_expenses_l1250_125061


namespace constant_distance_vector_l1250_125077

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem constant_distance_vector (a b p : V) :
  ‖p - b‖ = 3 * ‖p - a‖ →
  ∃ (c : ℝ), ∀ (q : V), ‖p - q‖ = c ↔ q = (9/8 : ℝ) • a - (1/8 : ℝ) • b :=
sorry

end constant_distance_vector_l1250_125077


namespace line_passes_through_circle_center_l1250_125036

/-- The line equation -/
def line_equation (x y m : ℝ) : Prop := x - 2*y + m = 0

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

/-- The center of the circle -/
def circle_center (x y : ℝ) : Prop := 
  circle_equation x y ∧ ∀ x' y', circle_equation x' y' → (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2

/-- The theorem statement -/
theorem line_passes_through_circle_center :
  ∃ m : ℝ, ∀ x y : ℝ, circle_center x y → line_equation x y m := by sorry

end line_passes_through_circle_center_l1250_125036


namespace ripe_orange_harvest_l1250_125099

/-- The number of days of harvest -/
def harvest_days : ℕ := 73

/-- The number of sacks of ripe oranges harvested per day -/
def daily_ripe_harvest : ℕ := 5

/-- The total number of sacks of ripe oranges harvested over the entire period -/
def total_ripe_harvest : ℕ := harvest_days * daily_ripe_harvest

theorem ripe_orange_harvest :
  total_ripe_harvest = 365 := by
  sorry

end ripe_orange_harvest_l1250_125099


namespace book_width_average_l1250_125044

theorem book_width_average : 
  let book_widths : List ℝ := [3, 3/4, 1.2, 4, 9, 0.5, 8]
  let total_width : ℝ := book_widths.sum
  let num_books : ℕ := book_widths.length
  let average_width : ℝ := total_width / num_books
  ∃ ε > 0, |average_width - 3.8| < ε := by
sorry

end book_width_average_l1250_125044


namespace parallel_vectors_trig_expression_l1250_125020

/-- Given vectors a and b that are parallel, prove that the given expression equals 3√2 -/
theorem parallel_vectors_trig_expression (x : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (ha : a = (Real.sin x, 2)) 
  (hb : b = (Real.cos x, 1)) 
  (h_parallel : ∃ (k : ℝ), a = k • b) : 
  (2 * Real.sin (x + π/4)) / (Real.sin x - Real.cos x) = 3 * Real.sqrt 2 := by
  sorry

end parallel_vectors_trig_expression_l1250_125020


namespace reciprocal_of_negative_two_l1250_125071

theorem reciprocal_of_negative_two :
  (1 : ℚ) / (-2 : ℚ) = -1/2 := by sorry

end reciprocal_of_negative_two_l1250_125071


namespace quadratic_roots_problem_l1250_125017

theorem quadratic_roots_problem (a b m p r : ℝ) : 
  (∀ x, x^2 - m*x + 4 = 0 ↔ x = a ∨ x = b) →
  (∀ x, x^2 - p*x + r = 0 ↔ x = a + 2/b ∨ x = b + 2/a) →
  r = 9 := by
sorry

end quadratic_roots_problem_l1250_125017


namespace additional_plates_l1250_125028

/-- The number of choices for each letter position in the original license plate system -/
def original_choices : Fin 3 → Nat
  | 0 => 5  -- First position
  | 1 => 3  -- Second position
  | 2 => 4  -- Third position

/-- The total number of possible license plates in the original system -/
def original_total : Nat := (original_choices 0) * (original_choices 1) * (original_choices 2)

/-- The number of choices for each letter position after adding one letter to each set -/
def new_choices : Fin 3 → Nat
  | i => (original_choices i) + 1

/-- The total number of possible license plates in the new system -/
def new_total : Nat := (new_choices 0) * (new_choices 1) * (new_choices 2)

/-- The theorem stating the number of additional license plates -/
theorem additional_plates : new_total - original_total = 60 := by
  sorry

end additional_plates_l1250_125028


namespace card_movement_strategy_exists_no_guaranteed_ace_strategy_l1250_125007

/-- Represents a deck of cards arranged in a circle with one free spot -/
structure CircularDeck :=
  (cards : Fin 52 → Option (Fin 52))
  (free_spot : Fin 53)
  (initial_positions : Fin 52 → Fin 53)

/-- Represents a strategy for naming cards -/
def Strategy := ℕ → Fin 52

/-- Checks if a card is next to the free spot -/
def is_next_to_free_spot (deck : CircularDeck) (card : Fin 52) : Prop :=
  sorry

/-- Moves a card to the free spot if it's adjacent -/
def move_card (deck : CircularDeck) (card : Fin 52) : CircularDeck :=
  sorry

/-- Applies a strategy to a deck for a given number of steps -/
def apply_strategy (deck : CircularDeck) (strategy : Strategy) (steps : ℕ) : CircularDeck :=
  sorry

/-- Checks if all cards are not in their initial positions -/
def all_cards_moved (deck : CircularDeck) : Prop :=
  sorry

/-- Checks if the ace of spades is not next to the free spot -/
def ace_not_next_to_free (deck : CircularDeck) : Prop :=
  sorry

theorem card_movement_strategy_exists :
  ∃ (strategy : Strategy), ∀ (initial_deck : CircularDeck),
  ∃ (steps : ℕ), all_cards_moved (apply_strategy initial_deck strategy steps) :=
sorry

theorem no_guaranteed_ace_strategy :
  ¬∃ (strategy : Strategy), ∀ (initial_deck : CircularDeck),
  ∃ (steps : ℕ), ace_not_next_to_free (apply_strategy initial_deck strategy steps) :=
sorry

end card_movement_strategy_exists_no_guaranteed_ace_strategy_l1250_125007


namespace race_distance_difference_l1250_125085

theorem race_distance_difference (race_distance : ℝ) (a_time b_time : ℝ) 
  (h1 : race_distance = 120)
  (h2 : a_time = 36)
  (h3 : b_time = 45) : 
  race_distance - (race_distance / b_time * a_time) = 24 := by
  sorry

end race_distance_difference_l1250_125085


namespace profit_percentage_l1250_125042

/-- Given that the cost price of 58 articles equals the selling price of 50 articles, 
    the percent profit is 16%. -/
theorem profit_percentage (C S : ℝ) (h : 58 * C = 50 * S) : 
  (S - C) / C * 100 = 16 := by
  sorry

end profit_percentage_l1250_125042


namespace exists_valid_nail_sequence_l1250_125091

/-- Represents a nail operation -/
inductive NailOp
| Blue1 | Blue2 | Blue3
| Red1 | Red2 | Red3

/-- Represents a sequence of nail operations -/
def NailSeq := List NailOp

/-- Checks if a nail sequence becomes trivial when a specific operation is removed -/
def becomes_trivial_without (seq : NailSeq) (op : NailOp) : Prop := sorry

/-- Checks if a nail sequence becomes trivial when two specific operations are removed -/
def becomes_trivial_without_two (seq : NailSeq) (op1 op2 : NailOp) : Prop := sorry

/-- The main theorem stating the existence of a valid nail sequence -/
theorem exists_valid_nail_sequence :
  ∃ (W : NailSeq),
    (∀ blue : NailOp, blue ∈ [NailOp.Blue1, NailOp.Blue2, NailOp.Blue3] →
      becomes_trivial_without W blue) ∧
    (∀ red1 red2 : NailOp, red1 ≠ red2 →
      red1 ∈ [NailOp.Red1, NailOp.Red2, NailOp.Red3] →
      red2 ∈ [NailOp.Red1, NailOp.Red2, NailOp.Red3] →
      becomes_trivial_without_two W red1 red2) ∧
    (∀ red : NailOp, red ∈ [NailOp.Red1, NailOp.Red2, NailOp.Red3] →
      ¬becomes_trivial_without W red) :=
sorry

end exists_valid_nail_sequence_l1250_125091


namespace square_root_expression_values_l1250_125026

theorem square_root_expression_values :
  ∀ (x y z : ℝ),
  (x^2 = 25) →
  (y = 4) →
  (z^2 = 9) →
  (2*x + y - 5*z = -1) ∨ (2*x + y - 5*z = 29) :=
by
  sorry

end square_root_expression_values_l1250_125026


namespace election_ratio_l1250_125078

theorem election_ratio (R D : ℝ) : 
  R > 0 ∧ D > 0 →  -- Positive number of Republicans and Democrats
  (0.9 * R + 0.15 * D) / (R + D) = 0.7 →  -- Candidate X's vote share
  (0.1 * R + 0.85 * D) / (R + D) = 0.3 →  -- Candidate Y's vote share
  R / D = 2.75 := by
sorry

end election_ratio_l1250_125078


namespace sin_2alpha_value_l1250_125047

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (α + π/4) = 3*Real.sqrt 2/5) : 
  Real.sin (2*α) = -11/25 := by
sorry

end sin_2alpha_value_l1250_125047


namespace subtract_twice_l1250_125064

theorem subtract_twice (a : ℝ) : a - 2*a = -a := by sorry

end subtract_twice_l1250_125064


namespace evaluate_expression_l1250_125003

theorem evaluate_expression : (24^36) / (72^18) = 8^18 := by
  sorry

end evaluate_expression_l1250_125003


namespace product_and_sum_of_integers_l1250_125030

theorem product_and_sum_of_integers : ∃ (n m : ℕ), 
  m = n + 2 ∧ 
  n * m = 2720 ∧ 
  n > 0 ∧ 
  n + m = 104 := by
sorry

end product_and_sum_of_integers_l1250_125030


namespace speed_calculation_l1250_125053

theorem speed_calculation (distance : ℝ) (early_time : ℝ) (speed_reduction : ℝ) : 
  distance = 40 ∧ early_time = 4/60 ∧ speed_reduction = 5 →
  ∃ (v : ℝ), v > 0 ∧ 
    (distance / v = distance / (v - speed_reduction) - early_time) ↔ 
    v = 60 := by sorry

end speed_calculation_l1250_125053


namespace greatest_two_digit_multiple_of_17_l1250_125079

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, n = 85 ∧ 
  (∀ m : ℕ, m ≤ 99 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ n) ∧ 
  17 ∣ n ∧ n ≤ 99 ∧ n ≥ 10 := by
  sorry

end greatest_two_digit_multiple_of_17_l1250_125079


namespace sixth_employee_salary_l1250_125074

/-- Given the salaries of 5 employees and the mean salary of all 6 employees,
    prove that the salary of the sixth employee is equal to the difference between
    the total salary of all 6 employees and the sum of the known 5 salaries. -/
theorem sixth_employee_salary
  (salary1 salary2 salary3 salary4 salary5 : ℝ)
  (mean_salary : ℝ)
  (h1 : salary1 = 1000)
  (h2 : salary2 = 2500)
  (h3 : salary3 = 3100)
  (h4 : salary4 = 1500)
  (h5 : salary5 = 2000)
  (h_mean : mean_salary = 2291.67)
  : ∃ (salary6 : ℝ),
    salary6 = 6 * mean_salary - (salary1 + salary2 + salary3 + salary4 + salary5) :=
by sorry

end sixth_employee_salary_l1250_125074


namespace cube_root_sum_equals_sixty_l1250_125092

theorem cube_root_sum_equals_sixty : 
  (30^3 + 40^3 + 50^3 : ℝ)^(1/3) = 60 := by sorry

end cube_root_sum_equals_sixty_l1250_125092


namespace complex_power_225_deg_18_l1250_125025

theorem complex_power_225_deg_18 : 
  (Complex.exp (Complex.I * Real.pi * (5 / 4)))^18 = Complex.I := by sorry

end complex_power_225_deg_18_l1250_125025


namespace sphere_configuration_exists_l1250_125014

-- Define a sphere in 3D space
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define a plane in 3D space
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

-- Function to check if a plane is tangent to a sphere
def is_tangent_plane (p : Plane) (s : Sphere) : Prop :=
  -- Implementation details omitted
  sorry

-- Function to check if a plane touches a sphere
def touches_sphere (p : Plane) (s : Sphere) : Prop :=
  -- Implementation details omitted
  sorry

-- Main theorem
theorem sphere_configuration_exists : ∃ (spheres : Fin 5 → Sphere),
  ∀ i : Fin 5, ∃ (p : Plane),
    is_tangent_plane p (spheres i) ∧
    (∀ j : Fin 5, j ≠ i → touches_sphere p (spheres j)) :=
  sorry

end sphere_configuration_exists_l1250_125014


namespace triangle_area_l1250_125060

-- Define the three lines
def line1 (x : ℝ) : ℝ := 2 * x + 1
def line2 (x : ℝ) : ℝ := -x + 4
def line3 (x : ℝ) : ℝ := -1

-- Define the vertices of the triangle
def vertex1 : ℝ × ℝ := (1, 3)
def vertex2 : ℝ × ℝ := (-1, -1)
def vertex3 : ℝ × ℝ := (5, -1)

-- Theorem statement
theorem triangle_area : 
  let vertices := [vertex1, vertex2, vertex3]
  let xs := vertices.map Prod.fst
  let ys := vertices.map Prod.snd
  abs ((xs[0] * (ys[1] - ys[2]) + xs[1] * (ys[2] - ys[0]) + xs[2] * (ys[0] - ys[1])) / 2) = 12 := by
  sorry


end triangle_area_l1250_125060


namespace arithmetic_sequence_properties_l1250_125005

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  a_1_eq : a 1 = 4
  a_7_sq_eq : (a 7) ^ 2 = (a 1) * (a 10)
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of the first n terms of the sequence -/
def S_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = -1/3 * n + 13/3) ∧
  (∃ n : ℕ, S_n seq n = 26 ∧ (n = 12 ∨ n = 13) ∧ ∀ m : ℕ, S_n seq m ≤ 26) :=
sorry

end arithmetic_sequence_properties_l1250_125005


namespace greatest_root_of_g_l1250_125051

-- Define the function g(x)
def g (x : ℝ) : ℝ := 10 * x^4 - 17 * x^2 + 3

-- State the theorem
theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt 15 / 5 ∧
  (∀ x : ℝ, g x = 0 → x ≤ r) ∧
  g r = 0 := by
  sorry

end greatest_root_of_g_l1250_125051


namespace cylinder_volume_increase_l1250_125054

def R : ℝ := 10
def H : ℝ := 5

theorem cylinder_volume_increase (x : ℝ) : 
  π * (R + 2*x)^2 * H = π * R^2 * (H + 3*x) → x = 5 := by
sorry

end cylinder_volume_increase_l1250_125054


namespace negation_of_proposition_P_l1250_125027

theorem negation_of_proposition_P :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by
  sorry

end negation_of_proposition_P_l1250_125027


namespace norris_money_left_l1250_125098

def savings_september : ℕ := 29
def savings_october : ℕ := 25
def savings_november : ℕ := 31
def spending_game : ℕ := 75

theorem norris_money_left : 
  savings_september + savings_october + savings_november - spending_game = 10 := by
  sorry

end norris_money_left_l1250_125098


namespace assignment_satisfies_conditions_l1250_125032

-- Define the set of people
inductive Person : Type
| Arthur : Person
| Burton : Person
| Congreve : Person
| Downs : Person
| Ewald : Person
| Flynn : Person

-- Define the set of positions
inductive Position : Type
| President : Position
| VicePresident : Position
| Secretary : Position
| Treasurer : Position

-- Define the assignment function
def assignment : Position → Person
| Position.President => Person.Flynn
| Position.VicePresident => Person.Ewald
| Position.Secretary => Person.Congreve
| Position.Treasurer => Person.Burton

-- Define the conditions
def arthur_condition (a : Position → Person) : Prop :=
  (a Position.VicePresident ≠ Person.Arthur) ∨ (a Position.President ≠ Person.Burton ∧ a Position.VicePresident ≠ Person.Burton ∧ a Position.Secretary ≠ Person.Burton ∧ a Position.Treasurer ≠ Person.Burton)

def burton_condition (a : Position → Person) : Prop :=
  a Position.VicePresident ≠ Person.Burton ∧ a Position.Secretary ≠ Person.Burton

def congreve_condition (a : Position → Person) : Prop :=
  (a Position.President ≠ Person.Burton ∧ a Position.VicePresident ≠ Person.Burton ∧ a Position.Secretary ≠ Person.Burton ∧ a Position.Treasurer ≠ Person.Burton) ∨
  (a Position.President = Person.Flynn ∨ a Position.VicePresident = Person.Flynn ∨ a Position.Secretary = Person.Flynn ∨ a Position.Treasurer = Person.Flynn)

def downs_condition (a : Position → Person) : Prop :=
  (a Position.President ≠ Person.Ewald ∧ a Position.VicePresident ≠ Person.Ewald ∧ a Position.Secretary ≠ Person.Ewald ∧ a Position.Treasurer ≠ Person.Ewald) ∧
  (a Position.President ≠ Person.Flynn ∧ a Position.VicePresident ≠ Person.Flynn ∧ a Position.Secretary ≠ Person.Flynn ∧ a Position.Treasurer ≠ Person.Flynn)

def ewald_condition (a : Position → Person) : Prop :=
  ¬(a Position.President = Person.Arthur ∧ (a Position.VicePresident = Person.Burton ∨ a Position.Secretary = Person.Burton ∨ a Position.Treasurer = Person.Burton)) ∧
  ¬(a Position.VicePresident = Person.Arthur ∧ (a Position.President = Person.Burton ∨ a Position.Secretary = Person.Burton ∨ a Position.Treasurer = Person.Burton)) ∧
  ¬(a Position.Secretary = Person.Arthur ∧ (a Position.President = Person.Burton ∨ a Position.VicePresident = Person.Burton ∨ a Position.Treasurer = Person.Burton)) ∧
  ¬(a Position.Treasurer = Person.Arthur ∧ (a Position.President = Person.Burton ∨ a Position.VicePresident = Person.Burton ∨ a Position.Secretary = Person.Burton))

def flynn_condition (a : Position → Person) : Prop :=
  (a Position.President = Person.Flynn) → (a Position.VicePresident ≠ Person.Congreve)

-- Theorem statement
theorem assignment_satisfies_conditions :
  arthur_condition assignment ∧
  burton_condition assignment ∧
  congreve_condition assignment ∧
  downs_condition assignment ∧
  ewald_condition assignment ∧
  flynn_condition assignment :=
sorry

end assignment_satisfies_conditions_l1250_125032


namespace age_problem_l1250_125016

theorem age_problem (a b c : ℕ) : 
  (4 * a + b = 3 * c) →
  (3 * c^3 = 4 * a^3 + b^3) →
  (Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1) →
  (a^2 + b^2 + c^2 = 35) :=
by sorry

end age_problem_l1250_125016


namespace sum_90_is_neg_180_l1250_125069

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference

/-- Sum of the first n terms of an arithmetic progression -/
def sum_n (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.a + (n - 1 : ℚ) * ap.d)

/-- Theorem: For the given arithmetic progression, the sum of the first 90 terms is -180 -/
theorem sum_90_is_neg_180 (ap : ArithmeticProgression) 
  (h15 : sum_n ap 15 = 150)
  (h75 : sum_n ap 75 = 30) : 
  sum_n ap 90 = -180 := by
  sorry

end sum_90_is_neg_180_l1250_125069


namespace subset_condition_disjoint_condition_l1250_125046

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Theorem 1: B ⊆ A iff m ∈ (-∞, 3]
theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by sorry

-- Theorem 2: A ∩ B = ∅ iff m ∈ (-∞, 2) ∪ (4, +∞)
theorem disjoint_condition (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end subset_condition_disjoint_condition_l1250_125046


namespace original_gross_profit_percentage_l1250_125095

theorem original_gross_profit_percentage
  (old_price new_price : ℝ)
  (new_profit_percentage : ℝ)
  (cost : ℝ)
  (h1 : old_price = 88)
  (h2 : new_price = 92)
  (h3 : new_profit_percentage = 0.15)
  (h4 : new_price = cost * (1 + new_profit_percentage)) :
  (old_price - cost) / cost = 0.1 := by
  sorry

end original_gross_profit_percentage_l1250_125095


namespace five_b_value_l1250_125033

theorem five_b_value (a b : ℚ) (h1 : 3 * a + 4 * b = 0) (h2 : a = b - 3) : 5 * b = 45 / 7 := by
  sorry

end five_b_value_l1250_125033


namespace final_jellybean_count_l1250_125048

def jellybean_count (initial : ℕ) (first_removal : ℕ) (addition : ℕ) (second_removal : ℕ) : ℕ :=
  initial - first_removal + addition - second_removal

theorem final_jellybean_count :
  jellybean_count 37 15 5 4 = 23 := by
  sorry

end final_jellybean_count_l1250_125048


namespace quadrilateral_properties_l1250_125011

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral defined by four points -/
structure Quadrilateral where
  O : Point
  P : Point
  Q : Point
  R : Point

/-- Checks if a quadrilateral is a parallelogram -/
def isParallelogram (quad : Quadrilateral) : Prop :=
  (quad.R.x - quad.P.x = quad.Q.x - quad.O.x) ∧
  (quad.R.y - quad.P.y = quad.Q.y - quad.O.y)

/-- Checks if a quadrilateral is a rhombus -/
def isRhombus (quad : Quadrilateral) : Prop :=
  let OP := (quad.P.x - quad.O.x)^2 + (quad.P.y - quad.O.y)^2
  let OQ := (quad.Q.x - quad.O.x)^2 + (quad.Q.y - quad.O.y)^2
  let OR := (quad.R.x - quad.O.x)^2 + (quad.R.y - quad.O.y)^2
  let PQ := (quad.Q.x - quad.P.x)^2 + (quad.Q.y - quad.P.y)^2
  OP = OQ ∧ OQ = OR ∧ OR = PQ

theorem quadrilateral_properties (x₁ y₁ x₂ y₂ : ℝ) :
  let quad := Quadrilateral.mk
    (Point.mk 0 0)
    (Point.mk x₁ y₁)
    (Point.mk x₂ y₂)
    (Point.mk (2*x₁ - x₂) (2*y₁ - y₂))
  isParallelogram quad ∧ (∃ (x₁ y₁ x₂ y₂ : ℝ), isRhombus quad) := by
  sorry

end quadrilateral_properties_l1250_125011


namespace initial_guppies_count_l1250_125029

/-- Represents the fish tank scenario --/
structure FishTank where
  initialGuppies : ℕ
  initialAngelfish : ℕ
  initialTigerSharks : ℕ
  initialOscarFish : ℕ
  soldGuppies : ℕ
  soldAngelfish : ℕ
  soldTigerSharks : ℕ
  soldOscarFish : ℕ
  remainingFish : ℕ

/-- Theorem stating the initial number of guppies in Danny's fish tank --/
theorem initial_guppies_count (tank : FishTank)
    (h1 : tank.initialAngelfish = 76)
    (h2 : tank.initialTigerSharks = 89)
    (h3 : tank.initialOscarFish = 58)
    (h4 : tank.soldGuppies = 30)
    (h5 : tank.soldAngelfish = 48)
    (h6 : tank.soldTigerSharks = 17)
    (h7 : tank.soldOscarFish = 24)
    (h8 : tank.remainingFish = 198)
    (h9 : tank.remainingFish = 
      (tank.initialGuppies - tank.soldGuppies) +
      (tank.initialAngelfish - tank.soldAngelfish) +
      (tank.initialTigerSharks - tank.soldTigerSharks) +
      (tank.initialOscarFish - tank.soldOscarFish)) :
    tank.initialGuppies = 94 := by
  sorry

end initial_guppies_count_l1250_125029


namespace quadratic_inequality_solution_sets_l1250_125009

-- Define the solution set of ax^2 - 5x + b > 0
def solution_set_1 : Set ℝ := {x | -3 < x ∧ x < -2}

-- Define the quadratic expression ax^2 - 5x + b
def quadratic_1 (a b x : ℝ) : ℝ := a * x^2 - 5 * x + b

-- Define the quadratic expression bx^2 - 5x + a
def quadratic_2 (a b x : ℝ) : ℝ := b * x^2 - 5 * x + a

-- Define the solution set of bx^2 - 5x + a < 0
def solution_set_2 : Set ℝ := {x | x < -1/2 ∨ x > -1/3}

theorem quadratic_inequality_solution_sets 
  (a b : ℝ) :
  (∀ x, x ∈ solution_set_1 ↔ quadratic_1 a b x > 0) →
  (∀ x, x ∈ solution_set_2 ↔ quadratic_2 a b x < 0) :=
by sorry

end quadratic_inequality_solution_sets_l1250_125009


namespace coin_value_problem_l1250_125002

theorem coin_value_problem :
  ∃ (n d q : ℕ),
    n + d + q = 30 ∧
    5 * n + 10 * d + 25 * q = 315 ∧
    10 * n + 25 * d + 5 * q = 5 * n + 10 * d + 25 * q + 120 :=
by sorry

end coin_value_problem_l1250_125002


namespace travel_time_difference_l1250_125022

def speed_A : ℝ := 60
def speed_B : ℝ := 45
def distance : ℝ := 360

theorem travel_time_difference :
  (distance / speed_B - distance / speed_A) * 60 = 120 := by
  sorry

end travel_time_difference_l1250_125022


namespace number_comparison_l1250_125045

theorem number_comparison (A B : ℝ) (h : (3/4) * A = (2/3) * B) : A < B := by
  sorry

end number_comparison_l1250_125045


namespace closed_grid_path_even_length_l1250_125096

/-- A closed path on a grid -/
structure GridPath where
  up : ℕ
  down : ℕ
  right : ℕ
  left : ℕ
  closed : up = down ∧ right = left

/-- The length of a grid path -/
def GridPath.length (p : GridPath) : ℕ :=
  p.up + p.down + p.right + p.left

/-- Theorem: The length of any closed grid path is even -/
theorem closed_grid_path_even_length (p : GridPath) : 
  Even p.length := by
sorry

end closed_grid_path_even_length_l1250_125096


namespace regular_polygon_diagonals_l1250_125037

theorem regular_polygon_diagonals (n : ℕ) : n > 2 →
  (n * (n - 3)) / 2 = 20 → n = 8 := by sorry

end regular_polygon_diagonals_l1250_125037


namespace monotone_increasing_constraint_l1250_125073

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 3

theorem monotone_increasing_constraint (a : ℝ) :
  (∀ x y, x < y ∧ y < 4 → f a x < f a y) →
  -1/4 ≤ a ∧ a ≤ 0 :=
by sorry

end monotone_increasing_constraint_l1250_125073


namespace combined_salaries_l1250_125041

/-- Given the salary of E and the average salary of A, B, C, D, and E,
    calculate the combined salaries of A, B, C, and D. -/
theorem combined_salaries 
  (salary_E : ℕ) 
  (average_salary : ℕ) 
  (h1 : salary_E = 9000)
  (h2 : average_salary = 8600) :
  (5 * average_salary) - salary_E = 34000 :=
by sorry

end combined_salaries_l1250_125041


namespace ellipse_properties_l1250_125080

/-- Given an ellipse with equation x²/m + y²/(m/(m+3)) = 1 where m > 0,
    and eccentricity e = √3/2, prove the following properties. -/
theorem ellipse_properties (m : ℝ) (h_m : m > 0) :
  let e := Real.sqrt 3 / 2
  let a := Real.sqrt m
  let b := Real.sqrt (m / (m + 3))
  let c := Real.sqrt ((m * (m + 2)) / (m + 3))
  (e = c / a) →
  (m = 1 ∧
   2 * a = 2 ∧ 2 * b = 1 ∧
   c = Real.sqrt 3 / 2 ∧
   a = 1 ∧ b = 1 / 2) :=
by sorry

end ellipse_properties_l1250_125080


namespace parabola_focus_directrix_relation_l1250_125083

/-- Represents a parabola with equation y^2 = 8px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- The focus of a parabola -/
def focus (para : Parabola) : ℝ × ℝ := (2 * para.p, 0)

/-- The x-coordinate of the directrix of a parabola -/
def directrix_x (para : Parabola) : ℝ := -2 * para.p

/-- The distance from the focus to the directrix -/
def focus_directrix_distance (para : Parabola) : ℝ :=
  (focus para).1 - directrix_x para

theorem parabola_focus_directrix_relation (para : Parabola) :
  para.p = (1/4) * focus_directrix_distance para := by sorry

end parabola_focus_directrix_relation_l1250_125083


namespace gcd_digits_bound_l1250_125050

theorem gcd_digits_bound (a b : ℕ) : 
  (1000000 ≤ a ∧ a < 10000000) →
  (1000000 ≤ b ∧ b < 10000000) →
  (10000000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 100000000000) →
  Nat.gcd a b < 10000 := by
sorry

end gcd_digits_bound_l1250_125050


namespace unequal_gender_probability_l1250_125089

/-- The number of grandchildren --/
def n : ℕ := 12

/-- The probability of a child being male or female --/
def p : ℚ := 1/2

/-- The probability of having an unequal number of grandsons and granddaughters --/
def unequal_probability : ℚ := 793/1024

theorem unequal_gender_probability :
  (1 : ℚ) - (n.choose (n/2) : ℚ) / (2^n : ℚ) = unequal_probability :=
sorry

end unequal_gender_probability_l1250_125089


namespace remainder_of_division_l1250_125000

theorem remainder_of_division (n : ℕ) : 
  (3^302 + 302) % (3^151 + 3^101 + 1) = 302 := by
  sorry

#check remainder_of_division

end remainder_of_division_l1250_125000


namespace rectangle_length_l1250_125038

/-- Given a rectangle with perimeter 30 cm and width 10 cm, prove its length is 5 cm -/
theorem rectangle_length (perimeter width : ℝ) (h1 : perimeter = 30) (h2 : width = 10) :
  2 * (width + (perimeter / 2 - width)) = perimeter → perimeter / 2 - width = 5 := by
  sorry

end rectangle_length_l1250_125038


namespace jack_morning_emails_l1250_125018

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 8

/-- The total number of emails Jack received in the morning and afternoon -/
def total_morning_afternoon : ℕ := 13

/-- Theorem stating that Jack received 5 emails in the morning -/
theorem jack_morning_emails : 
  morning_emails + afternoon_emails = total_morning_afternoon → 
  morning_emails = 5 := by sorry

end jack_morning_emails_l1250_125018


namespace min_value_when_a_is_neg_three_a_range_when_inequality_holds_l1250_125015

-- Define the function f
def f (a x : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem for part (1)
theorem min_value_when_a_is_neg_three :
  ∃ (min : ℝ), min = 4 ∧ ∀ x, f (-3) x ≥ min :=
sorry

-- Theorem for part (2)
theorem a_range_when_inequality_holds :
  (∀ x, f a x ≤ 2*a + 2*|x - 1|) → a ≥ 1/3 :=
sorry

end min_value_when_a_is_neg_three_a_range_when_inequality_holds_l1250_125015


namespace rose_additional_money_needed_l1250_125094

/-- The amount of additional money Rose needs to buy her art supplies -/
theorem rose_additional_money_needed 
  (paintbrush_cost : ℚ)
  (paints_cost : ℚ)
  (easel_cost : ℚ)
  (rose_current_money : ℚ)
  (h1 : paintbrush_cost = 2.40)
  (h2 : paints_cost = 9.20)
  (h3 : easel_cost = 6.50)
  (h4 : rose_current_money = 7.10) :
  paintbrush_cost + paints_cost + easel_cost - rose_current_money = 11 :=
by sorry

end rose_additional_money_needed_l1250_125094


namespace joshua_share_l1250_125024

def total_amount : ℚ := 123.50
def joshua_multiplier : ℚ := 3.5
def jasmine_multiplier : ℚ := 0.75

theorem joshua_share :
  ∃ (justin_share : ℚ),
    justin_share + joshua_multiplier * justin_share + jasmine_multiplier * justin_share = total_amount ∧
    joshua_multiplier * justin_share = 82.32 :=
by sorry

end joshua_share_l1250_125024


namespace investment_difference_proof_l1250_125019

/-- Represents an investment scheme with an initial investment and a yield rate -/
structure Scheme where
  investment : ℝ
  yieldRate : ℝ

/-- Calculates the total amount in a scheme after a year -/
def totalAfterYear (s : Scheme) : ℝ :=
  s.investment + s.investment * s.yieldRate

/-- The difference in total amounts between two schemes after a year -/
def schemeDifference (s1 s2 : Scheme) : ℝ :=
  totalAfterYear s1 - totalAfterYear s2

theorem investment_difference_proof (schemeA schemeB : Scheme) 
  (h1 : schemeA.investment = 300)
  (h2 : schemeB.investment = 200)
  (h3 : schemeA.yieldRate = 0.3)
  (h4 : schemeB.yieldRate = 0.5) :
  schemeDifference schemeA schemeB = 90 := by
  sorry

end investment_difference_proof_l1250_125019


namespace money_distribution_l1250_125090

theorem money_distribution (m l n : ℚ) (h1 : m > 0) (h2 : l > 0) (h3 : n > 0) 
  (h4 : m / 5 = l / 3) (h5 : m / 5 = n / 2) : 
  (3 * (m / 5)) / (m + l + n) = 3 / 10 := by
  sorry

end money_distribution_l1250_125090


namespace constant_term_expansion_l1250_125039

theorem constant_term_expansion (x : ℝ) : 
  (∃ c : ℝ, c ≠ 0 ∧ ∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, 
    0 < |y - x| ∧ |y - x| < δ → |(1/y - y^3)^4 - c| < ε) → 
  c = -4 :=
sorry

end constant_term_expansion_l1250_125039


namespace max_sections_five_l1250_125035

/-- The maximum number of sections created by n line segments in a rectangle -/
def max_sections (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => max_sections m + m + 1

/-- Theorem: The maximum number of sections created by 5 line segments in a rectangle is 16 -/
theorem max_sections_five : max_sections 5 = 16 := by
  sorry

end max_sections_five_l1250_125035


namespace bruce_purchase_l1250_125040

/-- Calculates the total amount Bruce paid for grapes and mangoes -/
def totalAmountPaid (grapeQuantity : ℕ) (grapeRate : ℕ) (mangoQuantity : ℕ) (mangoRate : ℕ) : ℕ :=
  grapeQuantity * grapeRate + mangoQuantity * mangoRate

/-- Proves that Bruce paid 985 for his purchase of grapes and mangoes -/
theorem bruce_purchase : totalAmountPaid 7 70 9 55 = 985 := by
  sorry

end bruce_purchase_l1250_125040


namespace shape_area_theorem_l1250_125008

/-- Represents a shape in a grid --/
structure GridShape where
  wholeSquares : ℕ
  halfSquares : ℕ

/-- Calculates the area of a GridShape --/
def calculateArea (shape : GridShape) : ℚ :=
  shape.wholeSquares + shape.halfSquares / 2

theorem shape_area_theorem (shape : GridShape) :
  shape.wholeSquares = 5 → shape.halfSquares = 6 → calculateArea shape = 8 := by
  sorry

end shape_area_theorem_l1250_125008


namespace overtime_pay_fraction_l1250_125058

/-- Represents the overtime pay calculation problem --/
theorem overtime_pay_fraction (regular_wage : ℝ) (hours_per_day : ℝ) (days : ℕ) 
  (total_pay : ℝ) (regular_hours : ℝ) (overtime_fraction : ℝ) : 
  regular_wage = 18 →
  hours_per_day = 10 →
  days = 5 →
  total_pay = 990 →
  regular_hours = 8 →
  total_pay = (regular_wage * regular_hours * days) + 
    (regular_wage * (1 + overtime_fraction) * (hours_per_day - regular_hours) * days) →
  overtime_fraction = 1/2 := by
  sorry


end overtime_pay_fraction_l1250_125058


namespace student_divisor_error_l1250_125068

theorem student_divisor_error (D : ℚ) (x : ℚ) : 
  D / 36 = 48 → D / x = 24 → x = 72 := by
  sorry

end student_divisor_error_l1250_125068


namespace circular_sector_angle_l1250_125065

/-- Given a circular sector with arc length 30 and diameter 16, 
    prove that its central angle in radians is 15/4 -/
theorem circular_sector_angle (arc_length : ℝ) (diameter : ℝ) 
  (h1 : arc_length = 30) (h2 : diameter = 16) :
  arc_length / (diameter / 2) = 15 / 4 := by
  sorry

end circular_sector_angle_l1250_125065


namespace tom_speed_proof_l1250_125097

/-- Represents the speed from B to C in miles per hour -/
def speed_B_to_C : ℝ := 64.8

/-- Represents the distance between B and C in miles -/
def distance_B_to_C : ℝ := 1  -- We use 1 as a variable to represent this distance

theorem tom_speed_proof :
  let distance_W_to_B : ℝ := 2 * distance_B_to_C
  let speed_W_to_B : ℝ := 60
  let average_speed : ℝ := 36
  let total_distance : ℝ := distance_W_to_B + distance_B_to_C
  let time_W_to_B : ℝ := distance_W_to_B / speed_W_to_B
  let time_B_to_C : ℝ := distance_B_to_C / speed_B_to_C
  let total_time : ℝ := time_W_to_B + time_B_to_C
  average_speed = total_distance / total_time →
  speed_B_to_C = 64.8 := by
  sorry

#check tom_speed_proof

end tom_speed_proof_l1250_125097


namespace two_correct_implications_l1250_125093

-- Define the types for planes and lines
def Plane : Type := Unit
def Line : Type := Unit

-- Define the relations between lines and planes
def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_to_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry
def not_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem two_correct_implications 
  (α β : Plane) 
  (l : Line) 
  (h_diff : α ≠ β)
  (h_not_in_α : not_in_plane l α)
  (h_not_in_β : not_in_plane l β)
  (h1 : perpendicular_to_plane l α)
  (h2 : parallel_to_plane l β)
  (h3 : perpendicular_planes α β) :
  ∃ (P Q R : Prop),
    (P ∧ Q → R) ∧
    (P ∧ R → Q) ∧
    ¬(Q ∧ R → P) ∧
    P ≠ Q ∧ Q ≠ R ∧ R ≠ P ∧
    (P = perpendicular_to_plane l α ∨ 
     P = parallel_to_plane l β ∨ 
     P = perpendicular_planes α β) ∧
    (Q = perpendicular_to_plane l α ∨ 
     Q = parallel_to_plane l β ∨ 
     Q = perpendicular_planes α β) ∧
    (R = perpendicular_to_plane l α ∨ 
     R = parallel_to_plane l β ∨ 
     R = perpendicular_planes α β) :=
by sorry

end two_correct_implications_l1250_125093


namespace vector_equality_iff_magnitude_and_parallel_l1250_125084

/-- Two plane vectors are equal if and only if their magnitudes are equal and they are parallel. -/
theorem vector_equality_iff_magnitude_and_parallel {a b : ℝ × ℝ} :
  a = b ↔ (‖a‖ = ‖b‖ ∧ ∃ (k : ℝ), a = k • b) :=
by sorry

end vector_equality_iff_magnitude_and_parallel_l1250_125084


namespace combined_age_of_siblings_l1250_125006

-- Define the ages of the siblings
def aaron_age : ℕ := 15
def sister_age : ℕ := 3 * aaron_age
def henry_age : ℕ := 4 * sister_age
def alice_age : ℕ := aaron_age - 2

-- Theorem to prove
theorem combined_age_of_siblings : aaron_age + sister_age + henry_age + alice_age = 253 := by
  sorry

end combined_age_of_siblings_l1250_125006


namespace range_of_a_l1250_125075

-- Define the inequality system
def inequality_system (x a : ℝ) : Prop :=
  (((x + 2) / 3 - x / 2) > 1) ∧ (2 * (x - a) ≤ 0)

-- Define the solution set
def solution_set (x : ℝ) : Prop := x < -2

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ x, inequality_system x a ↔ solution_set x) →
  a ≥ -2 :=
sorry

end range_of_a_l1250_125075


namespace factorization_of_a_squared_plus_5a_l1250_125049

theorem factorization_of_a_squared_plus_5a (a : ℝ) : a^2 + 5*a = a*(a+5) := by
  sorry

end factorization_of_a_squared_plus_5a_l1250_125049


namespace intersection_A_complement_B_l1250_125056

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x < 0}

def B : Set ℝ := {x | x > 1}

theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end intersection_A_complement_B_l1250_125056


namespace matt_work_time_l1250_125052

theorem matt_work_time (total_time together_time matt_remaining_time : ℝ) 
  (h1 : total_time = 20)
  (h2 : together_time = 12)
  (h3 : matt_remaining_time = 10) : 
  (total_time * matt_remaining_time) / (total_time - together_time) = 25 := by
  sorry

#check matt_work_time

end matt_work_time_l1250_125052


namespace student_sample_total_prove_student_sample_size_l1250_125088

/-- Represents the composition of students in a high school sample -/
structure StudentSample where
  total : ℕ
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- The theorem stating the total number of students in the sample -/
theorem student_sample_total (s : StudentSample) : s.total = 800 :=
  by
  have h1 : s.juniors = (28 : ℕ) * s.total / 100 := sorry
  have h2 : s.sophomores = (25 : ℕ) * s.total / 100 := sorry
  have h3 : s.seniors = 160 := sorry
  have h4 : s.freshmen = s.sophomores + 16 := sorry
  have h5 : s.total = s.freshmen + s.sophomores + s.juniors + s.seniors := sorry
  sorry

/-- The main theorem proving the total number of students -/
theorem prove_student_sample_size : ∃ s : StudentSample, s.total = 800 :=
  by
  sorry

end student_sample_total_prove_student_sample_size_l1250_125088


namespace units_digit_34_pow_30_l1250_125066

theorem units_digit_34_pow_30 : (34^30) % 10 = 6 := by
  sorry

end units_digit_34_pow_30_l1250_125066


namespace mans_speed_with_current_l1250_125004

/-- Given a man's speed against the current and the speed of the current,
    calculate the man's speed with the current. -/
def speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Theorem stating that given the specific conditions in the problem,
    the man's speed with the current is 20 kmph. -/
theorem mans_speed_with_current :
  let speed_against_current := 18
  let current_speed := 1
  speed_with_current speed_against_current current_speed = 20 := by
  sorry

#eval speed_with_current 18 1

end mans_speed_with_current_l1250_125004


namespace complex_number_simplification_l1250_125082

theorem complex_number_simplification :
  let i : ℂ := Complex.I
  let Z : ℂ := (2 + 4 * i) / (1 + i)
  Z = 3 + i := by sorry

end complex_number_simplification_l1250_125082


namespace max_notebooks_nine_notebooks_possible_l1250_125067

/-- Represents the cost and quantity of an item --/
structure Item where
  cost : ℕ
  quantity : ℕ

/-- Represents a purchase of pens, pencils, and notebooks --/
structure Purchase where
  pens : Item
  pencils : Item
  notebooks : Item

/-- The total cost of a purchase --/
def totalCost (p : Purchase) : ℕ :=
  p.pens.cost * p.pens.quantity +
  p.pencils.cost * p.pencils.quantity +
  p.notebooks.cost * p.notebooks.quantity

/-- A purchase is valid if it meets the given conditions --/
def isValidPurchase (p : Purchase) : Prop :=
  p.pens.cost = 3 ∧
  p.pencils.cost = 4 ∧
  p.notebooks.cost = 10 ∧
  p.pens.quantity ≥ 1 ∧
  p.pencils.quantity ≥ 1 ∧
  p.notebooks.quantity ≥ 1 ∧
  totalCost p ≤ 100

theorem max_notebooks (p : Purchase) (h : isValidPurchase p) :
  p.notebooks.quantity ≤ 9 := by
  sorry

theorem nine_notebooks_possible :
  ∃ p : Purchase, isValidPurchase p ∧ p.notebooks.quantity = 9 := by
  sorry

end max_notebooks_nine_notebooks_possible_l1250_125067


namespace point_on_line_l1250_125001

/-- A complex number z represented as (a-1) + 3i, where a is a real number -/
def z (a : ℝ) : ℂ := Complex.mk (a - 1) 3

/-- The line y = x + 2 in the complex plane -/
def line (x : ℝ) : ℝ := x + 2

/-- Theorem: If z(a) is on the line y = x + 2, then a = 2 -/
theorem point_on_line (a : ℝ) : z a = Complex.mk (z a).re (line (z a).re) → a = 2 := by
  sorry

end point_on_line_l1250_125001
