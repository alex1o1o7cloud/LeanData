import Mathlib

namespace festival_remaining_money_l3977_397759

def festival_spending (total_budget food_cost : ℕ) : ℕ :=
  let ride_cost := 3 * food_cost
  let game_cost := ride_cost / 2
  total_budget - (food_cost + ride_cost + game_cost)

theorem festival_remaining_money :
  festival_spending 100 16 = 12 := by
  sorry

end festival_remaining_money_l3977_397759


namespace coat_price_calculation_l3977_397788

/-- Calculates the final price of a coat after two discounts and tax --/
def finalPrice (originalPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) (taxRate : ℝ) : ℝ :=
  let priceAfterDiscount1 := originalPrice * (1 - discount1)
  let priceAfterDiscount2 := priceAfterDiscount1 * (1 - discount2)
  priceAfterDiscount2 * (1 + taxRate)

/-- Theorem stating that the final price of the coat is approximately 84.7 --/
theorem coat_price_calculation :
  let originalPrice : ℝ := 120
  let discount1 : ℝ := 0.30
  let discount2 : ℝ := 0.10
  let taxRate : ℝ := 0.12
  abs (finalPrice originalPrice discount1 discount2 taxRate - 84.7) < 0.1 := by
  sorry

#eval finalPrice 120 0.30 0.10 0.12

end coat_price_calculation_l3977_397788


namespace inequality_range_l3977_397799

theorem inequality_range (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ m : ℝ, (3 * x) / (2 * x + y) + (3 * y) / (x + 2 * y) ≤ m^2 + m) ↔ 
  (m ≤ -2 ∨ m ≥ 1) :=
by sorry

end inequality_range_l3977_397799


namespace exists_m_divides_f_100_l3977_397776

def f (x : ℤ) : ℤ := 3 * x + 2

theorem exists_m_divides_f_100 :
  ∃ m : ℕ+, 19881 ∣ (3^100 * (m.val + 1) - 1) :=
sorry

end exists_m_divides_f_100_l3977_397776


namespace max_N_is_seven_l3977_397725

def J (k : ℕ) : ℕ := 10^(k+3) + 128

def N (k : ℕ) : ℕ := (J k).factors.count 2

theorem max_N_is_seven : ∀ k : ℕ, k > 0 → N k ≤ 7 ∧ ∃ k₀ : ℕ, k₀ > 0 ∧ N k₀ = 7 :=
sorry

end max_N_is_seven_l3977_397725


namespace quadratic_real_roots_k_range_l3977_397722

theorem quadratic_real_roots_k_range (k : ℝ) :
  (∃ x : ℝ, 2 * x^2 + 4 * x + k - 1 = 0) →
  k ≤ 3 := by
sorry

end quadratic_real_roots_k_range_l3977_397722


namespace angle_terminal_side_cosine_l3977_397772

theorem angle_terminal_side_cosine (x : ℝ) (α : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (-x, -6) ∧ P ∈ {p | ∃ t : ℝ, p = (t * Real.cos α, t * Real.sin α)}) →
  Real.cos α = 4/5 →
  x = -8 := by
  sorry

end angle_terminal_side_cosine_l3977_397772


namespace probability_black_ball_l3977_397781

def total_balls : ℕ := 2 + 3

def black_balls : ℕ := 2

theorem probability_black_ball :
  (black_balls : ℚ) / total_balls = 2 / 5 := by sorry

end probability_black_ball_l3977_397781


namespace age_difference_john_aunt_l3977_397715

/-- Represents the ages of family members --/
structure FamilyAges where
  john : ℕ
  father : ℕ
  mother : ℕ
  grandmother : ℕ
  aunt : ℕ

/-- Defines the relationships between family members' ages --/
def valid_family_ages (ages : FamilyAges) : Prop :=
  ages.john * 2 = ages.father ∧
  ages.father = ages.mother + 4 ∧
  ages.grandmother = ages.john * 3 ∧
  ages.aunt = ages.mother * 2 - 5 ∧
  ages.father = 40

/-- Theorem stating the age difference between John and his aunt --/
theorem age_difference_john_aunt (ages : FamilyAges) 
  (h : valid_family_ages ages) : ages.aunt - ages.john = 47 := by
  sorry

end age_difference_john_aunt_l3977_397715


namespace bus_problem_l3977_397796

theorem bus_problem (initial_children on_bus off_bus final_children : ℕ) :
  initial_children = 22 →
  on_bus = 40 →
  final_children = 2 →
  initial_children + on_bus - off_bus = final_children →
  off_bus = 60 :=
by sorry

end bus_problem_l3977_397796


namespace problem_statement_l3977_397754

theorem problem_statement (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 1/2) : 
  m = 100 := by
sorry

end problem_statement_l3977_397754


namespace no_solution_fractional_equation_l3977_397780

theorem no_solution_fractional_equation :
  ∀ y : ℝ, y ≠ 3 → (y - 2) / (y - 3) ≠ 2 - 1 / (3 - y) := by
  sorry

end no_solution_fractional_equation_l3977_397780


namespace no_unique_solution_l3977_397740

/-- For a system of two linear equations to have no unique solution, 
    the ratios of coefficients and constants must be equal. -/
theorem no_unique_solution (k e : ℝ) : 
  (∃ k, ¬∃! (x y : ℝ), 4 * (3 * x + 4 * y) = 48 ∧ k * x + e * y = 30) →
  e = 10 := by
sorry

end no_unique_solution_l3977_397740


namespace order_of_expressions_l3977_397777

theorem order_of_expressions : 7^(0.3 : ℝ) > (0.3 : ℝ)^7 ∧ (0.3 : ℝ)^7 > Real.log 0.3 := by
  sorry

end order_of_expressions_l3977_397777


namespace inverse_37_mod_53_l3977_397727

theorem inverse_37_mod_53 : ∃ x : ℤ, 37 * x ≡ 1 [ZMOD 53] :=
by
  use 10
  sorry

end inverse_37_mod_53_l3977_397727


namespace f_of_one_eq_two_l3977_397738

def f (x : ℝ) := 3 * x - 1

theorem f_of_one_eq_two : f 1 = 2 := by
  sorry

end f_of_one_eq_two_l3977_397738


namespace decimal_to_fraction_l3977_397792

theorem decimal_to_fraction :
  (0.34 : ℚ) = 17 / 50 := by sorry

end decimal_to_fraction_l3977_397792


namespace binomial_cube_expansion_problem_solution_l3977_397743

theorem binomial_cube_expansion (n : ℕ) : n^3 + 3*(n^2) + 3*n + 1 = (n+1)^3 := by
  sorry

theorem problem_solution : 98^3 + 3*(98^2) + 3*98 + 1 = 99^3 := by
  sorry

end binomial_cube_expansion_problem_solution_l3977_397743


namespace conditional_structure_correctness_l3977_397753

-- Define a conditional structure
structure ConditionalStructure where
  hasTwoExits : Bool
  hasOneEffectiveExit : Bool

-- Define the properties of conditional structures
def conditionalStructureProperties : ConditionalStructure where
  hasTwoExits := true
  hasOneEffectiveExit := true

-- Theorem to prove
theorem conditional_structure_correctness :
  (conditionalStructureProperties.hasTwoExits = true) ∧
  (conditionalStructureProperties.hasOneEffectiveExit = true) := by
  sorry

#check conditional_structure_correctness

end conditional_structure_correctness_l3977_397753


namespace bryce_raisins_l3977_397714

theorem bryce_raisins (bryce carter : ℕ) : 
  bryce = carter + 8 →
  carter = bryce / 3 →
  bryce + carter = 44 →
  bryce = 33 := by
sorry

end bryce_raisins_l3977_397714


namespace cos_pi_minus_alpha_l3977_397734

theorem cos_pi_minus_alpha (α : Real) (h : Real.sin (α / 2) = 2 / 3) : 
  Real.cos (Real.pi - α) = -1 / 9 := by
  sorry

end cos_pi_minus_alpha_l3977_397734


namespace task_completion_probability_l3977_397719

theorem task_completion_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 3/8) (h2 : p2 = 3/5) (h3 : p3 = 5/9) (h4 : p4 = 7/12) :
  p1 * (1 - p2) * p3 * (1 - p4) = 5/72 := by
  sorry

end task_completion_probability_l3977_397719


namespace problem_solution_l3977_397746

theorem problem_solution (x y : ℝ) 
  (h1 : x = 103) 
  (h2 : x^3*y - 4*x^2*y + 4*x*y = 515400) : 
  y = 1/2 := by
sorry

end problem_solution_l3977_397746


namespace cylinder_lateral_area_l3977_397769

/-- The lateral area of a cylinder with diameter and height both equal to 4 is 16π. -/
theorem cylinder_lateral_area : 
  ∀ (d h : ℝ), d = 4 → h = 4 → 2 * π * (d / 2) * h = 16 * π :=
by
  sorry


end cylinder_lateral_area_l3977_397769


namespace largest_integer_in_interval_l3977_397775

theorem largest_integer_in_interval : 
  ∃ (x : ℤ), x = 4 ∧ (2 : ℚ) / 7 < (x : ℚ) / 6 ∧ (x : ℚ) / 6 < 7 / 9 ∧
  ∀ (y : ℤ), y > x → ¬((2 : ℚ) / 7 < (y : ℚ) / 6 ∧ (y : ℚ) / 6 < 7 / 9) :=
by sorry

end largest_integer_in_interval_l3977_397775


namespace equilateral_triangle_sticks_l3977_397750

def canFormEquilateralTriangle (n : ℕ) : Prop :=
  ∃ (side : ℕ), 3 * side = n * (n + 1) / 2

theorem equilateral_triangle_sticks (n : ℕ) :
  canFormEquilateralTriangle n ↔ n ≥ 5 ∧ (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :=
sorry

end equilateral_triangle_sticks_l3977_397750


namespace fraction_simplification_l3977_397724

theorem fraction_simplification (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : z - 1/x ≠ 0) :
  (x*z - 1/y) / (z - 1/x) = z :=
by sorry

end fraction_simplification_l3977_397724


namespace y2_greater_than_y1_l3977_397767

/-- The parabola equation y = x² - 2x + 3 -/
def parabola (x y : ℝ) : Prop := y = x^2 - 2*x + 3

theorem y2_greater_than_y1 (y1 y2 : ℝ) 
  (h1 : parabola (-1) y1)
  (h2 : parabola (-2) y2) : 
  y2 > y1 := by
  sorry

end y2_greater_than_y1_l3977_397767


namespace gcd_2134_155_in_ternary_is_100_l3977_397730

def gcd_2134_155_in_ternary : List Nat :=
  let m := Nat.gcd 2134 155
  Nat.digits 3 m

theorem gcd_2134_155_in_ternary_is_100 : 
  gcd_2134_155_in_ternary = [1, 0, 0] := by
  sorry

end gcd_2134_155_in_ternary_is_100_l3977_397730


namespace coordinates_of_B_l3977_397756

-- Define the square OABC
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (4, 3)

-- Define the property that C is in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Define the property of being a square
def is_square (O A B C : ℝ × ℝ) : Prop :=
  let d₁ := (A.1 - O.1)^2 + (A.2 - O.2)^2
  let d₂ := (B.1 - A.1)^2 + (B.2 - A.2)^2
  let d₃ := (C.1 - B.1)^2 + (C.2 - B.2)^2
  let d₄ := (O.1 - C.1)^2 + (O.2 - C.2)^2
  d₁ = d₂ ∧ d₂ = d₃ ∧ d₃ = d₄

-- Theorem statement
theorem coordinates_of_B :
  ∃ (B C : ℝ × ℝ), is_square O A B C ∧ in_fourth_quadrant C → B = (7, -1) :=
sorry

end coordinates_of_B_l3977_397756


namespace good_permutations_congruence_l3977_397786

/-- Given a prime number p > 3, count_good_permutations p returns the number of permutations
    (a₁, a₂, ..., aₚ₋₁) of (1, 2, ..., p-1) such that p divides the sum of consecutive products. -/
def count_good_permutations (p : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the number of good permutations is congruent to p-1 modulo p(p-1). -/
theorem good_permutations_congruence (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  count_good_permutations p ≡ p - 1 [MOD p * (p - 1)] :=
sorry

end good_permutations_congruence_l3977_397786


namespace unique_solution_l3977_397783

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

/-- The equation that n must satisfy -/
def satisfies_equation (n : ℕ) : Prop :=
  factorial (n + 1) + factorial (n + 3) = factorial n * 1540

theorem unique_solution :
  ∃! n : ℕ, n > 0 ∧ satisfies_equation n ∧ n = 10 := by sorry

end unique_solution_l3977_397783


namespace equation_represents_three_lines_l3977_397721

-- Define the equation
def equation (x y : ℝ) : Prop := (x + y)^3 = x^3 + y^3

-- Define what it means for a point to be on a line
def on_line (x y a b c : ℝ) : Prop := a*x + b*y + c = 0

-- Define the three lines we expect
def line1 (x y : ℝ) : Prop := on_line x y 1 1 0  -- x + y = 0
def line2 (x y : ℝ) : Prop := on_line x y 1 0 0  -- x = 0
def line3 (x y : ℝ) : Prop := on_line x y 0 1 0  -- y = 0

-- Theorem statement
theorem equation_represents_three_lines :
  ∀ x y : ℝ, equation x y ↔ (line1 x y ∨ line2 x y ∨ line3 x y) :=
sorry

end equation_represents_three_lines_l3977_397721


namespace max_students_equal_distribution_l3977_397764

theorem max_students_equal_distribution (pens pencils : ℕ) 
  (h1 : pens = 1204) (h2 : pencils = 840) :
  Nat.gcd pens pencils = 28 := by
sorry

end max_students_equal_distribution_l3977_397764


namespace a_range_f_result_l3977_397732

noncomputable section

variables (a x x₁ x₂ t : ℝ)

def f (x : ℝ) := Real.exp x - a * x + a

def f' (x : ℝ) := Real.exp x - a

axiom a_positive : a > 0

axiom x₁_less_x₂ : x₁ < x₂

axiom f_roots : f a x₁ = 0 ∧ f a x₂ = 0

axiom t_def : Real.sqrt ((x₂ - 1) / (x₁ - 1)) = t

axiom isosceles_right_triangle : ∃ (x₀ : ℝ), x₀ ∈ Set.Ioo x₁ x₂ ∧ 
  f a x₀ = (x₁ - x₂) / 2

theorem a_range : a > Real.exp 2 := by sorry

theorem f'_negative : f' a (Real.sqrt (x₁ * x₂)) < 0 := by sorry

theorem result : (a - 1) * (t - 1) = 2 := by sorry

end a_range_f_result_l3977_397732


namespace arithmetic_sequence_seventh_term_l3977_397752

theorem arithmetic_sequence_seventh_term
  (a : ℚ) -- First term of the sequence
  (d : ℚ) -- Common difference of the sequence
  (h1 : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 20) -- Sum of first five terms
  (h2 : a + 5*d = 8) -- Sixth term
  : a + 6*d = 28/3 := by
  sorry

end arithmetic_sequence_seventh_term_l3977_397752


namespace equation_solution_l3977_397736

theorem equation_solution : ∃ x : ℝ, (x - 3)^2 = x^2 - 9 ∧ x = 3 := by
  sorry

end equation_solution_l3977_397736


namespace existence_of_xy_l3977_397707

theorem existence_of_xy (n : ℕ) (k : ℕ) (h : n = 4 * k + 1) : 
  ∃ (x y : ℤ), (x^n + y^n) ∈ {z : ℤ | ∃ (a b : ℤ), z = a^2 + n * b^2} ∧ 
  (x + y) ∉ {z : ℤ | ∃ (a b : ℤ), z = a^2 + n * b^2} := by
  sorry

end existence_of_xy_l3977_397707


namespace polynomial_factorization_l3977_397712

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) =
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) := by
  sorry

end polynomial_factorization_l3977_397712


namespace train_speed_problem_l3977_397784

/-- Given a train journey with two scenarios:
    1. The train covers a distance in 360 minutes at an unknown average speed.
    2. The same distance can be covered in 90 minutes at a speed of 20 kmph.
    This theorem proves that the average speed in the first scenario is 5 kmph. -/
theorem train_speed_problem (distance : ℝ) (avg_speed : ℝ) : 
  distance = 20 * (90 / 60) → -- The distance is covered in 90 minutes at 20 kmph
  distance = avg_speed * (360 / 60) → -- The same distance is covered in 360 minutes at avg_speed
  avg_speed = 5 := by
  sorry

end train_speed_problem_l3977_397784


namespace tims_initial_amount_l3977_397703

/-- Tim's candy bar purchase scenario -/
def candy_bar_purchase (initial_amount paid change : ℕ) : Prop :=
  initial_amount = paid + change

/-- Theorem: Tim's initial amount before buying the candy bar -/
theorem tims_initial_amount : ∃ (initial_amount : ℕ), 
  candy_bar_purchase initial_amount 45 5 ∧ initial_amount = 50 := by
  sorry

end tims_initial_amount_l3977_397703


namespace complement_M_intersect_P_l3977_397770

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | |x - 1/2| ≤ 5/2}
def P : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- State the theorem
theorem complement_M_intersect_P :
  (U \ M) ∩ P = {x | 3 < x ∧ x ≤ 4} := by sorry

end complement_M_intersect_P_l3977_397770


namespace total_spent_is_575_l3977_397787

def vacuum_original_cost : ℚ := 250
def vacuum_discount_rate : ℚ := 20 / 100
def dishwasher_cost : ℚ := 450
def combined_discount : ℚ := 75

def total_spent : ℚ :=
  (vacuum_original_cost * (1 - vacuum_discount_rate) + dishwasher_cost) - combined_discount

theorem total_spent_is_575 : total_spent = 575 := by
  sorry

end total_spent_is_575_l3977_397787


namespace mem_not_veen_l3977_397737

-- Define the sets
variable (U : Type) -- Universe set
variable (Mem En Veen : Set U)

-- Define the hypotheses
variable (h1 : Mem ⊆ En)
variable (h2 : En ∩ Veen = ∅)

-- Theorem to prove
theorem mem_not_veen :
  (∀ x, x ∈ Mem → x ∉ Veen) ∧
  (Mem ∩ Veen = ∅) :=
sorry

end mem_not_veen_l3977_397737


namespace two_red_more_likely_than_one_four_l3977_397702

/-- The number of red balls in the box -/
def red_balls : ℕ := 4

/-- The number of white balls in the box -/
def white_balls : ℕ := 2

/-- The total number of balls in the box -/
def total_balls : ℕ := red_balls + white_balls

/-- The number of faces on each die -/
def die_faces : ℕ := 6

/-- The probability of drawing two red balls from the box -/
def prob_two_red : ℚ := (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1))

/-- The probability of rolling at least one 4 with two dice -/
def prob_at_least_one_four : ℚ := 1 - (die_faces - 1)^2 / die_faces^2

/-- Theorem stating that the probability of drawing two red balls is greater than
    the probability of rolling at least one 4 with two dice -/
theorem two_red_more_likely_than_one_four : prob_two_red > prob_at_least_one_four :=
sorry

end two_red_more_likely_than_one_four_l3977_397702


namespace income_growth_and_projection_l3977_397733

/-- Represents the annual growth rate as a real number between 0 and 1 -/
def AnnualGrowthRate := { r : ℝ // 0 < r ∧ r < 1 }

/-- Calculates the future value given initial value, growth rate, and number of years -/
def futureValue (initialValue : ℝ) (rate : AnnualGrowthRate) (years : ℕ) : ℝ :=
  initialValue * (1 + rate.val) ^ years

theorem income_growth_and_projection (initialIncome : ℝ) (finalIncome : ℝ) (years : ℕ) 
  (h1 : initialIncome = 2500)
  (h2 : finalIncome = 3600)
  (h3 : years = 2) :
  ∃ (rate : AnnualGrowthRate),
    (futureValue initialIncome rate years = finalIncome) ∧ 
    (rate.val = 0.2) ∧
    (futureValue finalIncome rate 1 > 4200) := by
  sorry

#check income_growth_and_projection

end income_growth_and_projection_l3977_397733


namespace intersection_nonempty_implies_a_greater_than_one_l3977_397778

-- Define the sets A and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem statement
theorem intersection_nonempty_implies_a_greater_than_one (a : ℝ) :
  (A ∩ C a).Nonempty → a > 1 := by
  sorry

end intersection_nonempty_implies_a_greater_than_one_l3977_397778


namespace pager_fraction_l3977_397791

theorem pager_fraction (total : ℝ) (total_pos : 0 < total) : 
  let cell_phone := (2/3 : ℝ) * total
  let neither := (1/3 : ℝ) * total
  let both := (0.4 : ℝ) * total
  let pager := (0.8 : ℝ) * total
  (cell_phone + (pager - both) = total - neither) →
  (pager / total = 0.8) :=
by
  sorry

end pager_fraction_l3977_397791


namespace least_tiles_required_l3977_397748

def room_length : Real := 8.16
def room_width : Real := 4.32
def recess_width : Real := 1.24
def recess_length : Real := 2
def protrusion_width : Real := 0.48
def protrusion_length : Real := 0.96

def main_area : Real := room_length * room_width
def recess_area : Real := recess_width * recess_length
def protrusion_area : Real := protrusion_width * protrusion_length
def total_area : Real := main_area + recess_area + protrusion_area

def tile_size : Real := protrusion_width

theorem least_tiles_required :
  ∃ n : ℕ, n = ⌈total_area / (tile_size * tile_size)⌉ ∧ n = 166 := by
  sorry

end least_tiles_required_l3977_397748


namespace log_problem_l3977_397713

theorem log_problem (x k : ℝ) : 
  (Real.log 3 / Real.log 4 = x) → 
  (Real.log 27 / Real.log 2 = k * x) → 
  k = 6 := by
sorry

end log_problem_l3977_397713


namespace article_profit_percentage_l3977_397700

theorem article_profit_percentage (cost selling_price : ℚ) : 
  cost = 70 →
  (0.8 * cost) * 1.3 = selling_price - 14.70 →
  (selling_price - cost) / cost * 100 = 25 := by
sorry

end article_profit_percentage_l3977_397700


namespace no_single_intersection_point_l3977_397773

theorem no_single_intersection_point :
  ¬ ∃ (b : ℝ), b ≠ 0 ∧
    (∃! (x : ℝ), bx^2 + 2*x - 3 = 2*x + 5) :=
sorry

end no_single_intersection_point_l3977_397773


namespace area_of_XYZ_main_theorem_l3977_397717

/-- Triangle ABC with given side lengths --/
structure Triangle :=
  (AB BC CA : ℝ)

/-- Points on the triangle --/
structure TrianglePoints :=
  (A B C D P Q X Y Z : ℝ × ℝ)

/-- Given triangle ABC with altitude AD and inscribed circle tangent points --/
def given_triangle : Triangle := { AB := 13, BC := 14, CA := 15 }

/-- Theorem: Area of triangle XYZ is 25/4 --/
theorem area_of_XYZ (t : Triangle) (tp : TrianglePoints) : ℝ :=
  let triangle := given_triangle
  25 / 4

/-- Main theorem --/
theorem main_theorem (t : Triangle) (tp : TrianglePoints) : 
  t = given_triangle → 
  tp.D.1 = tp.B.1 ∨ tp.D.1 = tp.C.1 →  -- D is on BC
  (tp.A.1 - tp.D.1)^2 + (tp.A.2 - tp.D.2)^2 = (tp.B.1 - tp.D.1)^2 + (tp.B.2 - tp.D.2)^2 →  -- AD perpendicular to BC
  (tp.P.1 - tp.A.1) * (tp.D.1 - tp.A.1) + (tp.P.2 - tp.A.2) * (tp.D.2 - tp.A.2) = 0 →  -- P on AD
  (tp.Q.1 - tp.A.1) * (tp.D.1 - tp.A.1) + (tp.Q.2 - tp.A.2) * (tp.D.2 - tp.A.2) = 0 →  -- Q on AD
  tp.X.1 = tp.B.1 ∨ tp.X.1 = tp.C.1 →  -- X on BC
  tp.Y.1 = tp.B.1 ∨ tp.Y.1 = tp.C.1 →  -- Y on BC
  (tp.Z.1 - tp.P.1) * (tp.X.1 - tp.P.1) + (tp.Z.2 - tp.P.2) * (tp.X.2 - tp.P.2) = 0 →  -- Z on PX
  (tp.Z.1 - tp.Q.1) * (tp.Y.1 - tp.Q.1) + (tp.Z.2 - tp.Q.2) * (tp.Y.2 - tp.Q.2) = 0 →  -- Z on QY
  area_of_XYZ t tp = 25 / 4 := by
  sorry

end area_of_XYZ_main_theorem_l3977_397717


namespace distribute_negative_two_l3977_397795

theorem distribute_negative_two (x : ℝ) : -2 * (x + 1) = -2 * x - 2 := by
  sorry

end distribute_negative_two_l3977_397795


namespace primitive_root_modulo_p_alpha_implies_modulo_p_l3977_397741

theorem primitive_root_modulo_p_alpha_implies_modulo_p
  (p : Nat) (α : Nat) (x : Nat)
  (h_prime : Nat.Prime p)
  (h_pos : α > 0)
  (h_primitive_p_alpha : IsPrimitiveRoot x (p ^ α)) :
  IsPrimitiveRoot x p :=
sorry

end primitive_root_modulo_p_alpha_implies_modulo_p_l3977_397741


namespace parabola_equation_l3977_397762

/-- Represents a parabola with integer coefficients -/
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  a_pos : 0 < a
  gcd_one : Nat.gcd (Int.natAbs a) (Nat.gcd (Int.natAbs b) (Nat.gcd (Int.natAbs c) (Nat.gcd (Int.natAbs d) (Nat.gcd (Int.natAbs e) (Int.natAbs f))))) = 1

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (1, -2)

/-- The directrix of the parabola -/
def directrix (x y : ℝ) : Prop := 5 * x + 2 * y = 10

/-- Checks if a point is on the parabola -/
def isOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  p.a * x^2 + p.b * x * y + p.c * y^2 + p.d * x + p.e * y + p.f = 0

/-- The theorem stating the equation of the parabola -/
theorem parabola_equation : ∃ (p : Parabola), 
  ∀ (x y : ℝ), isOnParabola p x y ↔ 
    (x - focus.1)^2 + (y - focus.2)^2 = (5 * x + 2 * y - 10)^2 / 29 :=
sorry

end parabola_equation_l3977_397762


namespace total_arrangements_l3977_397735

/-- Represents the number of people in the group -/
def total_people : Nat := 6

/-- Represents the number of people who must sit together -/
def group_size : Nat := 3

/-- Calculates the number of ways to arrange the group -/
def arrange_group (n : Nat) : Nat :=
  Nat.factorial n

/-- Calculates the number of ways to choose people for the group -/
def choose_group (n : Nat) : Nat :=
  n

/-- Calculates the number of ways to insert the group -/
def insert_group (n : Nat) : Nat :=
  n * (n - 1)

/-- The main theorem stating the total number of arrangements -/
theorem total_arrangements :
  arrange_group (total_people - group_size) *
  choose_group group_size *
  insert_group (total_people - group_size + 1) = 216 :=
sorry

end total_arrangements_l3977_397735


namespace segment_length_l3977_397747

theorem segment_length : Real.sqrt 193 = Real.sqrt ((8 - 1)^2 + (14 - 2)^2) := by sorry

end segment_length_l3977_397747


namespace integral_of_even_function_l3977_397771

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The main theorem -/
theorem integral_of_even_function (a : ℝ) :
  let f := fun x => a * x^2 + (a - 2) * x + a^2
  IsEven f →
  ∫ x in (-a)..a, (x^2 + x + Real.sqrt (4 - x^2)) = 16/3 + 2 * Real.pi := by
  sorry

end integral_of_even_function_l3977_397771


namespace cloth_selling_price_l3977_397763

/-- Calculates the total selling price of cloth given the quantity, loss per meter, and cost price per meter. -/
def total_selling_price (quantity : ℕ) (loss_per_meter : ℚ) (cost_price_per_meter : ℚ) : ℚ :=
  quantity * (cost_price_per_meter - loss_per_meter)

/-- Proves that the total selling price for 400 meters of cloth is $18,000 given the specified conditions. -/
theorem cloth_selling_price :
  total_selling_price 400 5 50 = 18000 := by
  sorry

end cloth_selling_price_l3977_397763


namespace solar_eclipse_viewers_scientific_notation_l3977_397716

theorem solar_eclipse_viewers_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 2580000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.58 ∧ n = 6 := by
  sorry

end solar_eclipse_viewers_scientific_notation_l3977_397716


namespace systematic_sample_interval_count_l3977_397709

/-- Calculates the number of sampled individuals within a given interval in a systematic sample. -/
def sampledInInterval (totalPopulation : ℕ) (sampleSize : ℕ) (intervalStart : ℕ) (intervalEnd : ℕ) : ℕ :=
  let groupDistance := totalPopulation / sampleSize
  (intervalEnd - intervalStart + 1) / groupDistance

/-- Theorem stating that for the given parameters, the number of sampled individuals in the interval [61, 140] is 4. -/
theorem systematic_sample_interval_count :
  sampledInInterval 840 42 61 140 = 4 := by
  sorry

#eval sampledInInterval 840 42 61 140

end systematic_sample_interval_count_l3977_397709


namespace binary_octal_conversion_l3977_397751

/-- Converts a binary number (represented as a list of 0s and 1s) to decimal -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.reverse.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- Converts a decimal number to octal (represented as a list of digits) -/
def decimal_to_octal (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else aux (m / 8) ((m % 8) :: acc)
  aux n []

theorem binary_octal_conversion :
  let binary : List Nat := [1, 0, 1, 1, 1, 1]
  let decimal : Nat := binary_to_decimal binary
  let octal : List Nat := decimal_to_octal decimal
  decimal = 47 ∧ octal = [5, 7] := by sorry

end binary_octal_conversion_l3977_397751


namespace difference_of_squares_l3977_397749

theorem difference_of_squares (a : ℝ) : (a + 1) * (a - 1) = a^2 - 1 := by
  sorry

end difference_of_squares_l3977_397749


namespace two_red_one_blue_probability_l3977_397728

def total_marbles : ℕ := 20
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8

theorem two_red_one_blue_probability :
  let prob := (red_marbles * (red_marbles - 1) * blue_marbles + 
               red_marbles * blue_marbles * (red_marbles - 1) + 
               blue_marbles * red_marbles * (red_marbles - 1)) / 
              (total_marbles * (total_marbles - 1) * (total_marbles - 2))
  prob = 44 / 95 := by
sorry

end two_red_one_blue_probability_l3977_397728


namespace three_students_got_A_l3977_397708

structure Student :=
  (name : String)
  (gotA : Bool)

def Emily : Student := ⟨"Emily", false⟩
def Frank : Student := ⟨"Frank", false⟩
def Grace : Student := ⟨"Grace", false⟩
def Harry : Student := ⟨"Harry", false⟩

def students : List Student := [Emily, Frank, Grace, Harry]

def emilyStatement (s : List Student) : Prop :=
  (Emily.gotA = true) → (Frank.gotA = true)

def frankStatement (s : List Student) : Prop :=
  (Frank.gotA = true) → (Grace.gotA = true)

def graceStatement (s : List Student) : Prop :=
  (Grace.gotA = true) → (Harry.gotA = true)

def harryStatement (s : List Student) : Prop :=
  (Harry.gotA = true) → (Emily.gotA = false)

def exactlyThreeGotA (s : List Student) : Prop :=
  (s.filter (λ x => x.gotA)).length = 3

theorem three_students_got_A :
  ∀ s : List Student,
    s = students →
    emilyStatement s →
    frankStatement s →
    graceStatement s →
    harryStatement s →
    exactlyThreeGotA s →
    (Frank.gotA = true ∧ Grace.gotA = true ∧ Harry.gotA = true ∧ Emily.gotA = false) :=
by sorry


end three_students_got_A_l3977_397708


namespace problem_statement_l3977_397798

theorem problem_statement (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49/((x - 3)^2) = 23 := by
sorry

end problem_statement_l3977_397798


namespace least_number_of_cans_l3977_397779

def maaza_liters : ℕ := 80
def pepsi_liters : ℕ := 144
def sprite_liters : ℕ := 368

def can_size : ℕ := Nat.gcd maaza_liters (Nat.gcd pepsi_liters sprite_liters)

def total_cans : ℕ := maaza_liters / can_size + pepsi_liters / can_size + sprite_liters / can_size

theorem least_number_of_cans :
  (∀ k : ℕ, k > 0 → (maaza_liters % k = 0 ∧ pepsi_liters % k = 0 ∧ sprite_liters % k = 0) → k ≤ can_size) ∧
  total_cans = 37 :=
sorry

end least_number_of_cans_l3977_397779


namespace intersection_sum_l3977_397742

theorem intersection_sum (c d : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + c) →
  (∀ x y : ℝ, y = 5 * x + d) →
  16 = 2 * 4 + c →
  16 = 5 * 4 + d →
  c + d = 4 := by
sorry

end intersection_sum_l3977_397742


namespace triangle_area_l3977_397790

/-- Given a right isosceles triangle that shares sides with squares of areas 100, 64, and 100,
    prove that the area of the triangle is 50. -/
theorem triangle_area (a b c : ℝ) (ha : a^2 = 100) (hb : b^2 = 64) (hc : c^2 = 100)
  (right_isosceles : a = c ∧ a^2 + c^2 = b^2) : (1/2) * a * c = 50 := by
  sorry

end triangle_area_l3977_397790


namespace rectangle_area_l3977_397723

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 16 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 48 := by
  sorry

end rectangle_area_l3977_397723


namespace root_implies_k_value_l3977_397766

theorem root_implies_k_value (k : ℝ) : 
  (2^2 - 3*2 + k = 0) → k = 2 := by
  sorry

end root_implies_k_value_l3977_397766


namespace extremum_values_l3977_397794

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extremum_values (a b : ℝ) :
  f a b 1 = 10 ∧ (deriv (f a b)) 1 = 0 → a = 4 ∧ b = -11 := by
  sorry

end extremum_values_l3977_397794


namespace coopers_fence_bricks_l3977_397718

/-- Calculates the number of bricks needed for a fence with given dimensions. -/
def bricks_needed (num_walls length height depth : ℕ) : ℕ :=
  num_walls * length * height * depth

/-- Theorem stating the number of bricks needed for Cooper's fence. -/
theorem coopers_fence_bricks : 
  bricks_needed 4 20 5 2 = 800 := by
  sorry

end coopers_fence_bricks_l3977_397718


namespace neither_direct_nor_inverse_proportional_l3977_397765

/-- A function representing the relationship between x and y --/
def Relationship (f : ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ x y, y = f x → (y = k * x ∨ x * y = k)

/-- Equation A: 2x + y = 5 --/
def EquationA (x y : ℝ) : Prop := 2 * x + y = 5

/-- Equation B: 4xy = 12 --/
def EquationB (x y : ℝ) : Prop := 4 * x * y = 12

/-- Equation C: x = 4y --/
def EquationC (x y : ℝ) : Prop := x = 4 * y

/-- Equation D: 2x + 3y = 15 --/
def EquationD (x y : ℝ) : Prop := 2 * x + 3 * y = 15

/-- Equation E: x/y = 4 --/
def EquationE (x y : ℝ) : Prop := x / y = 4

theorem neither_direct_nor_inverse_proportional :
  (¬ Relationship (λ x => 5 - 2 * x)) ∧
  (Relationship (λ x => 3 / (4 * x))) ∧
  (Relationship (λ x => x / 4)) ∧
  (¬ Relationship (λ x => (15 - 2 * x) / 3)) ∧
  (Relationship (λ x => x / 4)) :=
sorry

end neither_direct_nor_inverse_proportional_l3977_397765


namespace trigonometric_equality_l3977_397797

theorem trigonometric_equality (x y z a : ℝ) 
  (h1 : (Real.sin x + Real.sin y + Real.sin z) / Real.sin (x + y + z) = a)
  (h2 : (Real.cos x + Real.cos y + Real.cos z) / Real.cos (x + y + z) = a) :
  Real.cos (x + y) + Real.cos (y + z) + Real.cos (z + x) = a := by
  sorry

end trigonometric_equality_l3977_397797


namespace solution_set_inequality_l3977_397768

-- Define the function f
def f (x : ℝ) : ℝ := -x^3

-- State the theorem
theorem solution_set_inequality (x : ℝ) :
  f (2 * x^2 - 1) < -1 ↔ x < -1 ∨ x > 1 := by
  sorry

end solution_set_inequality_l3977_397768


namespace arithmetic_sequence_sum_l3977_397782

/-- Given an arithmetic sequence {a_n} with a_1 = -2014 and S_n as the sum of first n terms,
    if S_{2012}/2012 - S_{10}/10 = 2002, then S_{2016} = 2016 -/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)) →  -- Definition of S_n
  (a 1 = -2014) →                                         -- First term condition
  (S 2012 / 2012 - S 10 / 10 = 2002) →                    -- Given condition
  (S 2016 = 2016) :=                                      -- Conclusion to prove
by sorry

end arithmetic_sequence_sum_l3977_397782


namespace line_cartesian_to_polar_l3977_397760

/-- Given a line in Cartesian coordinates x cos α + y sin α = 0,
    its equivalent polar coordinate equation is θ = α - π/2 --/
theorem line_cartesian_to_polar (α : Real) :
  ∀ x y r θ : Real,
  (x * Real.cos α + y * Real.sin α = 0) →
  (x = r * Real.cos θ) →
  (y = r * Real.sin θ) →
  (θ = α - π/2) := by
  sorry

end line_cartesian_to_polar_l3977_397760


namespace watermelon_not_necessarily_split_l3977_397739

/-- Represents a spherical watermelon with given diameter and cut depth. -/
structure Watermelon where
  diameter : ℝ
  cut_depth : ℝ

/-- Determines if the watermelon is necessarily split into at least two pieces. -/
def is_necessarily_split (w : Watermelon) : Prop :=
  ∃ (configuration : ℝ → ℝ → ℝ → Prop),
    ∀ (x y z : ℝ),
      configuration x y z →
      (x^2 + y^2 + z^2 ≤ (w.diameter/2)^2) →
      (|x| ≤ w.cut_depth ∨ |y| ≤ w.cut_depth ∨ |z| ≤ w.cut_depth)

/-- Theorem stating that a watermelon with diameter 20 cm is not necessarily split
    for cut depths of 17 cm and 18 cm. -/
theorem watermelon_not_necessarily_split :
  let w₁ : Watermelon := ⟨20, 17⟩
  let w₂ : Watermelon := ⟨20, 18⟩
  ¬(is_necessarily_split w₁) ∧ ¬(is_necessarily_split w₂) := by
  sorry

end watermelon_not_necessarily_split_l3977_397739


namespace quadratic_distinct_roots_l3977_397745

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) := by
sorry

end quadratic_distinct_roots_l3977_397745


namespace largest_cube_minus_smallest_fifth_l3977_397757

theorem largest_cube_minus_smallest_fifth : ∃ (a b : ℕ), 
  (∀ n : ℕ, n^3 < 999 → n ≤ a) ∧ 
  (a^3 < 999) ∧
  (∀ m : ℕ, m^5 > 99 → b ≤ m) ∧ 
  (b^5 > 99) ∧
  (a - b = 6) := by
sorry

end largest_cube_minus_smallest_fifth_l3977_397757


namespace sqrt_of_square_negative_eleven_l3977_397711

theorem sqrt_of_square_negative_eleven : Real.sqrt ((-11)^2) = 11 := by
  sorry

end sqrt_of_square_negative_eleven_l3977_397711


namespace tickets_left_l3977_397729

/-- Given that Paul bought eleven tickets and spent three tickets,
    prove that he has eight tickets left. -/
theorem tickets_left (total : ℕ) (spent : ℕ) (left : ℕ) 
    (h1 : total = 11)
    (h2 : spent = 3)
    (h3 : left = total - spent) : left = 8 := by
  sorry

end tickets_left_l3977_397729


namespace square_circle_circumradius_infinite_l3977_397744

/-- The radius of the circumcircle of a square with side length 1 and an inscribed circle 
    with diameter equal to the square's diagonal is infinite. -/
theorem square_circle_circumradius_infinite :
  let square : Set (ℝ × ℝ) := {p | p.1 ∈ [0, 1] ∧ p.2 ∈ [0, 1]}
  let circle : Set (ℝ × ℝ) := {p | (p.1 - 0.5)^2 + p.2^2 ≤ 0.5^2}
  let figure : Set (ℝ × ℝ) := square ∪ circle
  ¬ ∃ (r : ℝ), r > 0 ∧ ∃ (c : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ figure → (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 :=
by sorry


end square_circle_circumradius_infinite_l3977_397744


namespace election_votes_l3977_397755

theorem election_votes (total_votes : ℕ) (winner_votes loser_votes : ℕ) : 
  winner_votes = (56 : ℕ) * total_votes / 100 →
  loser_votes = total_votes - winner_votes →
  winner_votes - loser_votes = 288 →
  winner_votes = 1344 := by
sorry

end election_votes_l3977_397755


namespace symmetric_point_of_P_l3977_397761

/-- The symmetric point of P(1, 3) with respect to the line y=x is (3, 1) -/
theorem symmetric_point_of_P : ∃ (P' : ℝ × ℝ), 
  (P' = (3, 1) ∧ 
   (∀ (Q : ℝ × ℝ), Q.1 = Q.2 → (1 - Q.1)^2 + (3 - Q.2)^2 = (P'.1 - Q.1)^2 + (P'.2 - Q.2)^2)) :=
by sorry

end symmetric_point_of_P_l3977_397761


namespace infinite_solutions_abs_value_equation_l3977_397706

theorem infinite_solutions_abs_value_equation (a : ℝ) :
  (∀ x : ℝ, |x - 2| = a * x - 2) ↔ a = 1 := by
  sorry

end infinite_solutions_abs_value_equation_l3977_397706


namespace chord_count_l3977_397774

/-- The number of points on the circumference of the circle -/
def n : ℕ := 9

/-- The number of points needed to form a chord -/
def r : ℕ := 2

/-- The number of different chords that can be drawn -/
def num_chords : ℕ := Nat.choose n r

theorem chord_count : num_chords = 36 := by
  sorry

end chord_count_l3977_397774


namespace faster_walking_speed_l3977_397701

/-- Proves that given a person who walked 100 km at 10 km/hr, if they had walked at a faster speed
    for the same amount of time and covered an additional 20 km, their faster speed would be 12 km/hr. -/
theorem faster_walking_speed (actual_distance : ℝ) (actual_speed : ℝ) (additional_distance : ℝ) :
  actual_distance = 100 →
  actual_speed = 10 →
  additional_distance = 20 →
  (actual_distance + additional_distance) / (actual_distance / actual_speed) = 12 :=
by sorry

end faster_walking_speed_l3977_397701


namespace complement_of_union_l3977_397731

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Set Nat := {2, 3, 5}

-- Define set N
def N : Set Nat := {4, 5}

-- Theorem statement
theorem complement_of_union (h : Set Nat → Set Nat → Set Nat) :
  h M N = {1, 6} :=
by sorry

end complement_of_union_l3977_397731


namespace quadratic_roots_nature_l3977_397758

theorem quadratic_roots_nature (k : ℝ) : 
  (∃ x y : ℝ, x * y = 12 ∧ x^2 - 4*k*x + 3*k^2 + 1 = 0 ∧ y^2 - 4*k*y + 3*k^2 + 1 = 0) →
  (∃ x : ℝ, x^2 - 4*k*x + 3*k^2 + 1 = 0 ∧ (∀ y : ℝ, y^2 - 4*k*y + 3*k^2 + 1 = 0 → y = x)) :=
by sorry

end quadratic_roots_nature_l3977_397758


namespace expected_coffee_tea_difference_l3977_397704

/-- Represents the outcome of rolling a fair eight-sided die -/
inductive DieRoll
  | one | two | three | four | five | six | seven | eight

/-- Represents the drink Alice chooses based on her die roll -/
inductive Drink
  | coffee | tea | juice

/-- Function that determines the drink based on the die roll -/
def chooseDrink (roll : DieRoll) : Drink :=
  match roll with
  | DieRoll.one => Drink.juice
  | DieRoll.two => Drink.coffee
  | DieRoll.three => Drink.tea
  | DieRoll.four => Drink.coffee
  | DieRoll.five => Drink.tea
  | DieRoll.six => Drink.coffee
  | DieRoll.seven => Drink.tea
  | DieRoll.eight => Drink.coffee

/-- The number of days in a non-leap year -/
def daysInYear : ℕ := 365

/-- Theorem stating the expected difference between coffee and tea days -/
theorem expected_coffee_tea_difference :
  let p_coffee : ℚ := 1/2
  let p_tea : ℚ := 3/8
  let expected_coffee_days : ℚ := p_coffee * daysInYear
  let expected_tea_days : ℚ := p_tea * daysInYear
  let difference : ℚ := expected_coffee_days - expected_tea_days
  ⌊difference⌋ = 45 := by sorry


end expected_coffee_tea_difference_l3977_397704


namespace project_choices_l3977_397705

/-- The number of projects available to choose from -/
def num_projects : ℕ := 5

/-- The number of students choosing projects -/
def num_students : ℕ := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the number of permutations of k items from n items -/
def permute (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

/-- The main theorem stating the number of ways students can choose projects -/
theorem project_choices : 
  (choose num_students 2) * (permute num_projects 3) + (permute num_projects num_students) = 480 :=
sorry

end project_choices_l3977_397705


namespace mrs_hilt_books_read_l3977_397710

def chapters_per_book : ℕ := 17
def total_chapters_read : ℕ := 68

theorem mrs_hilt_books_read :
  total_chapters_read / chapters_per_book = 4 := by sorry

end mrs_hilt_books_read_l3977_397710


namespace quadratic_decreasing_condition_l3977_397793

/-- A quadratic function f(x) = -2x^2 + mx - 3 is decreasing on the interval [-1, +∞) if and only if m ≤ -4 -/
theorem quadratic_decreasing_condition (m : ℝ) :
  (∀ x : ℝ, x ≥ -1 → (∀ y : ℝ, y > x → -2*y^2 + m*y - 3 < -2*x^2 + m*x - 3)) ↔ m ≤ -4 := by
  sorry


end quadratic_decreasing_condition_l3977_397793


namespace matrix_equation_solution_l3977_397726

theorem matrix_equation_solution :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -3; 5, -1]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-30, -9; 11, 1]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![5, -8; 7/13, 35/13]
  N * A = B := by sorry

end matrix_equation_solution_l3977_397726


namespace range_of_m_l3977_397789

def has_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4*a*c ≥ 0

def p (m : ℝ) : Prop :=
  has_real_roots 1 m 1

def q (m : ℝ) : Prop :=
  ¬(has_real_roots 4 (4*(m-2)) 1)

def exactly_one_true (p q : Prop) : Prop :=
  (p ∧ ¬q) ∨ (¬p ∧ q)

theorem range_of_m : 
  {m : ℝ | exactly_one_true (p m) (q m)} = 
  {m : ℝ | m ≤ -2 ∨ (1 < m ∧ m < 2) ∨ m ≥ 3} :=
by sorry

end range_of_m_l3977_397789


namespace xy_value_l3977_397785

theorem xy_value (x y : ℚ) 
  (eq1 : 5 * x + 3 * y + 5 = 0) 
  (eq2 : 3 * x + 5 * y - 5 = 0) : 
  x * y = -25 / 4 := by
sorry

end xy_value_l3977_397785


namespace sum_of_roots_equals_fourteen_l3977_397720

theorem sum_of_roots_equals_fourteen : 
  let f : ℝ → ℝ := λ x => (x - 7)^2 - 16
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 14 :=
by sorry

end sum_of_roots_equals_fourteen_l3977_397720
