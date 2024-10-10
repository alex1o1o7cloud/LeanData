import Mathlib

namespace harvard_attendance_l1577_157790

def total_applicants : ℕ := 20000
def acceptance_rate : ℚ := 5 / 100
def attendance_rate : ℚ := 90 / 100

theorem harvard_attendance : 
  ⌊(total_applicants : ℚ) * acceptance_rate * attendance_rate⌋ = 900 := by
  sorry

end harvard_attendance_l1577_157790


namespace range_x_when_m_is_one_range_m_for_not_p_necessary_but_not_sufficient_l1577_157758

-- Define propositions p and q
def p (x m : ℝ) : Prop := |2 * x - m| ≥ 1
def q (x : ℝ) : Prop := (1 - 3 * x) / (x + 2) > 0

-- Theorem for part 1
theorem range_x_when_m_is_one :
  {x : ℝ | p x 1 ∧ q x} = {x : ℝ | -2 < x ∧ x ≤ 0} := by sorry

-- Theorem for part 2
theorem range_m_for_not_p_necessary_but_not_sufficient :
  {m : ℝ | ∀ x, q x → ¬(p x m) ∧ ∃ y, ¬(p y m) ∧ ¬(q y)} = {m : ℝ | -3 ≤ m ∧ m ≤ -1/3} := by sorry

end range_x_when_m_is_one_range_m_for_not_p_necessary_but_not_sufficient_l1577_157758


namespace division_problem_l1577_157728

theorem division_problem (N : ℕ) : 
  (N / 7 = 12 ∧ N % 7 = 4) → (N / 3 = 29) := by
  sorry

end division_problem_l1577_157728


namespace incenter_circles_theorem_l1577_157733

-- Define the basic geometric objects
variable (A B C I : Point)
variable (O₁ O₂ O₃ : Point)
variable (A' B' C' : Point)

-- Define the incenter
def is_incenter (I : Point) (A B C : Point) : Prop := sorry

-- Define circles passing through points
def circle_through (O : Point) (P Q : Point) : Prop := sorry

-- Define perpendicular intersection of circles
def perpendicular_intersection (O : Point) (I : Point) : Prop := sorry

-- Define the other intersection point of two circles
def other_intersection (O₁ O₂ : Point) (P : Point) : Point := sorry

-- Define the circumradius of a triangle
def circumradius (A B C : Point) : ℝ := sorry

-- Define the radius of a circle
def circle_radius (O : Point) : ℝ := sorry

-- State the theorem
theorem incenter_circles_theorem 
  (h_incenter : is_incenter I A B C)
  (h_O₁ : circle_through O₁ B C)
  (h_O₂ : circle_through O₂ A C)
  (h_O₃ : circle_through O₃ A B)
  (h_perp₁ : perpendicular_intersection O₁ I)
  (h_perp₂ : perpendicular_intersection O₂ I)
  (h_perp₃ : perpendicular_intersection O₃ I)
  (h_A' : A' = other_intersection O₂ O₃ A)
  (h_B' : B' = other_intersection O₁ O₃ B)
  (h_C' : C' = other_intersection O₁ O₂ C) :
  circumradius A' B' C' = (1/2) * circle_radius I := by sorry

end incenter_circles_theorem_l1577_157733


namespace min_value_of_a_l1577_157722

theorem min_value_of_a (a : ℝ) (h_a : a > 0) : 
  (∀ (x₁ : ℝ) (x₂ : ℝ), x₁ > 0 → 1 ≤ x₂ → x₂ ≤ Real.exp 1 → 
    x₁ + a^2 / x₁ ≥ x₂ - Real.log x₂) → 
  a ≥ Real.sqrt (Real.exp 1 - 2) := by
  sorry

end min_value_of_a_l1577_157722


namespace f_has_three_zeros_l1577_157747

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  -1 / (x + 1) - (a + 1) * log (x + 1) + a * x + Real.exp 1 - 2

theorem f_has_three_zeros (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    x₁ > -1 ∧ x₂ > -1 ∧ x₃ > -1 ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    ∀ x : ℝ, x > -1 → f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) ↔
  a > Real.exp 1 :=
sorry

end f_has_three_zeros_l1577_157747


namespace flower_bed_fraction_is_correct_l1577_157764

/-- Represents a rectangular yard with flower beds and a trapezoidal lawn -/
structure YardWithFlowerBeds where
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ
  trapezoid_height : ℝ
  num_flower_beds : ℕ

/-- The fraction of the yard occupied by flower beds -/
def flower_bed_fraction (yard : YardWithFlowerBeds) : ℚ :=
  25 / 324

/-- Theorem stating the fraction of the yard occupied by flower beds -/
theorem flower_bed_fraction_is_correct (yard : YardWithFlowerBeds) 
    (h1 : yard.trapezoid_short_side = 26)
    (h2 : yard.trapezoid_long_side = 36)
    (h3 : yard.trapezoid_height = 6)
    (h4 : yard.num_flower_beds = 3) : 
  flower_bed_fraction yard = 25 / 324 := by
  sorry

#check flower_bed_fraction_is_correct

end flower_bed_fraction_is_correct_l1577_157764


namespace initial_typists_count_l1577_157772

/-- The number of typists in the initial group -/
def initial_typists : ℕ := 25

/-- The number of letters the initial group can type in 20 minutes -/
def letters_in_20_min : ℕ := 60

/-- The number of typists in the second group -/
def second_group_typists : ℕ := 75

/-- The number of letters the second group can type in 60 minutes -/
def letters_in_60_min : ℕ := 540

/-- The time ratio between the two scenarios -/
def time_ratio : ℚ := 3

theorem initial_typists_count :
  initial_typists * second_group_typists * letters_in_20_min * time_ratio = 
  letters_in_60_min * initial_typists * time_ratio :=
sorry

end initial_typists_count_l1577_157772


namespace imaginary_part_of_i_minus_one_l1577_157732

theorem imaginary_part_of_i_minus_one :
  Complex.im (Complex.I - 1) = 1 :=
sorry

end imaginary_part_of_i_minus_one_l1577_157732


namespace soccer_game_scoring_l1577_157783

/-- Soccer game scoring theorem -/
theorem soccer_game_scoring
  (team_a_first_half : ℕ)
  (team_b_first_half : ℕ)
  (team_a_second_half : ℕ)
  (team_b_second_half : ℕ)
  (h1 : team_a_first_half = 8)
  (h2 : team_b_second_half = team_a_first_half)
  (h3 : team_a_second_half = team_b_second_half - 2)
  (h4 : team_a_first_half + team_b_first_half + team_a_second_half + team_b_second_half = 26) :
  team_b_first_half / team_a_first_half = 1 / 2 := by
  sorry

end soccer_game_scoring_l1577_157783


namespace hyperbola_asymptote_l1577_157734

/-- Given a hyperbola with equation x²/4 - y²/b² = 1 where b > 0,
    if the equations of its asymptotes are y = ± x/2, then b = 1 -/
theorem hyperbola_asymptote (b : ℝ) (h1 : b > 0) :
  (∀ x y : ℝ, x^2 / 4 - y^2 / b^2 = 1) →
  (∀ x : ℝ, (∃ y : ℝ, y = x / 2 ∨ y = -x / 2) → 
    x^2 / 4 - y^2 / b^2 = 1) →
  b = 1 := by sorry

end hyperbola_asymptote_l1577_157734


namespace lcm_of_1716_924_1260_l1577_157782

theorem lcm_of_1716_924_1260 : Nat.lcm (Nat.lcm 1716 924) 1260 = 13860 := by
  sorry

end lcm_of_1716_924_1260_l1577_157782


namespace company_merger_profit_l1577_157776

theorem company_merger_profit (x : ℝ) (h1 : 0.4 * x = 60000) (h2 : 0 < x) : 0.6 * x = 90000 := by
  sorry

end company_merger_profit_l1577_157776


namespace subcommittee_formation_count_l1577_157723

theorem subcommittee_formation_count :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 7
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 3
  
  (Nat.choose total_republicans subcommittee_republicans) *
  (Nat.choose total_democrats subcommittee_democrats) = 7350 := by
  sorry

end subcommittee_formation_count_l1577_157723


namespace sum_of_first_n_naturals_l1577_157700

theorem sum_of_first_n_naturals (n : ℕ) : 
  (List.range (n + 1)).sum = n * (n + 1) / 2 := by
  sorry

end sum_of_first_n_naturals_l1577_157700


namespace population_model_steps_l1577_157786

/- Define the steps as an inductive type -/
inductive ModelingStep
  | observe : ModelingStep
  | test : ModelingStep
  | propose : ModelingStep
  | express : ModelingStep

/- Define a function to represent the correct order of steps -/
def correct_order : List ModelingStep :=
  [ModelingStep.observe, ModelingStep.propose, ModelingStep.express, ModelingStep.test]

/- Define a predicate to check if a given order is correct -/
def is_correct_order (order : List ModelingStep) : Prop :=
  order = correct_order

/- Theorem stating that the specified order is correct -/
theorem population_model_steps :
  is_correct_order [ModelingStep.observe, ModelingStep.propose, ModelingStep.express, ModelingStep.test] :=
by sorry

end population_model_steps_l1577_157786


namespace cuboid_length_is_40_l1577_157702

/-- The surface area of a cuboid given its length, breadth, and height -/
def cuboidSurfaceArea (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

/-- Theorem: The length of a cuboid with surface area 2400, breadth 10, and height 16 is 40 -/
theorem cuboid_length_is_40 :
  ∃ l : ℝ, cuboidSurfaceArea l 10 16 = 2400 ∧ l = 40 :=
by sorry

end cuboid_length_is_40_l1577_157702


namespace base_conversion_1729_to_base7_l1577_157730

/-- Converts a list of digits in base 7 to a natural number in base 10 -/
def fromBase7 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Theorem: 1729 in base 10 is equal to 5020 in base 7 -/
theorem base_conversion_1729_to_base7 :
  1729 = fromBase7 [5, 0, 2, 0] := by
  sorry

#eval fromBase7 [5, 0, 2, 0]  -- Should output 1729

end base_conversion_1729_to_base7_l1577_157730


namespace points_in_quadrants_I_and_II_l1577_157726

-- Define the set of points satisfying the inequalities
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 > -1/2 * p.1 ∧ p.2 > 3 * p.1 + 6}

-- Define the quadrants
def quadrantI : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}
def quadrantII : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0}
def quadrantIII : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 < 0 ∧ p.2 < 0}
def quadrantIV : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 < 0}

-- Theorem statement
theorem points_in_quadrants_I_and_II : 
  S ⊆ quadrantI ∪ quadrantII ∧ 
  S ∩ quadrantIII = ∅ ∧ 
  S ∩ quadrantIV = ∅ := by
  sorry

end points_in_quadrants_I_and_II_l1577_157726


namespace polynomial_multiplication_l1577_157739

theorem polynomial_multiplication (x : ℝ) :
  (3 * x^2 - 2 * x + 4) * (-4 * x^2 + 3 * x - 6) =
  -12 * x^4 + 17 * x^3 - 40 * x^2 + 24 * x - 24 := by
  sorry

end polynomial_multiplication_l1577_157739


namespace complex_subtraction_multiplication_l1577_157707

theorem complex_subtraction_multiplication (i : ℂ) :
  (7 - 3 * i) - 3 * (2 + 4 * i) = 1 - 15 * i :=
by sorry

end complex_subtraction_multiplication_l1577_157707


namespace pet_store_siamese_cats_l1577_157741

theorem pet_store_siamese_cats :
  let initial_house_cats : ℝ := 5.0
  let added_cats : ℝ := 10.0
  let total_cats_after : ℕ := 28
  let initial_siamese_cats : ℝ := initial_house_cats + added_cats + total_cats_after - (initial_house_cats + added_cats)
  initial_siamese_cats = 13 :=
by sorry

end pet_store_siamese_cats_l1577_157741


namespace circle_B_radius_l1577_157780

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

def internally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius - c2.radius)^2

def congruent (c1 c2 : Circle) : Prop :=
  c1.radius = c2.radius

def passes_through_center (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = c1.radius^2

theorem circle_B_radius
  (A B C D E : Circle)
  (h1 : externally_tangent A B)
  (h2 : externally_tangent A C)
  (h3 : externally_tangent A E)
  (h4 : externally_tangent B C)
  (h5 : externally_tangent B E)
  (h6 : externally_tangent C E)
  (h7 : internally_tangent A D)
  (h8 : internally_tangent B D)
  (h9 : internally_tangent C D)
  (h10 : internally_tangent E D)
  (h11 : congruent B C)
  (h12 : congruent A E)
  (h13 : A.radius = 2)
  (h14 : passes_through_center A D)
  : B.radius = 2 := by
  sorry

end circle_B_radius_l1577_157780


namespace negation_of_union_membership_l1577_157748

theorem negation_of_union_membership {α : Type*} (A B : Set α) (x : α) :
  ¬(x ∈ A ∪ B) ↔ x ∉ A ∧ x ∉ B := by
  sorry

end negation_of_union_membership_l1577_157748


namespace middle_number_theorem_l1577_157704

theorem middle_number_theorem (x y z : ℤ) 
  (h_order : x < y ∧ y < z)
  (h_sum1 : x + y = 10)
  (h_sum2 : x + z = 21)
  (h_sum3 : y + z = 25) : 
  y = 7 := by
sorry

end middle_number_theorem_l1577_157704


namespace heating_pad_cost_per_use_l1577_157743

/-- The cost per use of a heating pad -/
def cost_per_use (total_cost : ℚ) (uses_per_week : ℕ) (num_weeks : ℕ) : ℚ :=
  total_cost / (uses_per_week * num_weeks)

/-- Theorem: The cost per use of a heating pad is $5 -/
theorem heating_pad_cost_per_use :
  cost_per_use 30 3 2 = 5 := by
  sorry

end heating_pad_cost_per_use_l1577_157743


namespace sin_15_deg_identity_l1577_157720

theorem sin_15_deg_identity : 1 - 2 * (Real.sin (15 * π / 180))^2 = Real.sqrt 3 / 2 := by
  sorry

end sin_15_deg_identity_l1577_157720


namespace inequality_implication_l1577_157735

theorem inequality_implication (a b : ℝ) (h : a > b) : 2*a - 1 > 2*b - 1 := by
  sorry

end inequality_implication_l1577_157735


namespace unique_solution_range_l1577_157746

theorem unique_solution_range (x a : ℝ) : 
  (∃! x, Real.log (4 * x^2 + 4 * a * x) - Real.log (4 * x - a + 1) = 0) ↔ 
  (1/5 ≤ a ∧ a < 1) :=
sorry

end unique_solution_range_l1577_157746


namespace inequality_solution_set_l1577_157725

theorem inequality_solution_set 
  (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_deriv : ∀ x, deriv f x > f x) : 
  {x : ℝ | f x / Real.exp x > f 1 / Real.exp 1} = Set.Ioi 1 := by
sorry

end inequality_solution_set_l1577_157725


namespace luzhou_gdp_scientific_correct_l1577_157718

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_in_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The GDP value of Luzhou City in 2022 -/
def luzhou_gdp : ℕ := 260150000000

/-- The scientific notation representation of Luzhou's GDP -/
def luzhou_gdp_scientific : ScientificNotation :=
  { coefficient := 2.6015
    exponent := 11
    coeff_in_range := by sorry }

/-- Theorem stating that the scientific notation representation is correct -/
theorem luzhou_gdp_scientific_correct :
  (luzhou_gdp_scientific.coefficient * (10 : ℝ) ^ luzhou_gdp_scientific.exponent) = luzhou_gdp := by
  sorry

end luzhou_gdp_scientific_correct_l1577_157718


namespace ratio_odd_even_divisors_l1577_157708

def M : ℕ := 39 * 48 * 77 * 150

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors M) * 62 = sum_even_divisors M :=
sorry

end ratio_odd_even_divisors_l1577_157708


namespace find_b_l1577_157767

theorem find_b (x₁ x₂ c : ℝ) (h₁ : x₁ ≠ x₂) 
  (h₂ : ∃ y, y^2 + 2*x₁*y + 2*x₂ = 0 ∧ y^2 + 2*x₂*y + 2*x₁ = 0)
  (h₃ : x₁^2 + 5*(1/10)*x₁ + c = 0)
  (h₄ : x₂^2 + 5*(1/10)*x₂ + c = 0) :
  ∃ b : ℝ, b = 1/10 ∧ x₁^2 + 5*b*x₁ + c = 0 ∧ x₂^2 + 5*b*x₂ + c = 0 :=
by sorry

end find_b_l1577_157767


namespace terry_current_age_l1577_157778

/-- Terry's current age in years -/
def terry_age : ℕ := sorry

/-- Nora's current age in years -/
def nora_age : ℕ := 10

/-- The number of years in the future when Terry's age will be 4 times Nora's current age -/
def years_future : ℕ := 10

theorem terry_current_age : 
  terry_age = 30 :=
by
  have h1 : terry_age + years_future = 4 * nora_age := sorry
  sorry

end terry_current_age_l1577_157778


namespace smallest_non_trivial_divisor_of_product_l1577_157752

def product_of_even_integers (n : ℕ) : ℕ :=
  (List.range ((n + 1) / 2)).foldl (λ acc i => acc * (2 * (i + 1))) 1

theorem smallest_non_trivial_divisor_of_product (n : ℕ) (h : n = 134) :
  ∃ (d : ℕ), d > 1 ∧ d ∣ product_of_even_integers n ∧
  ∀ (k : ℕ), 1 < k → k < d → ¬(k ∣ product_of_even_integers n) :=
by
  sorry

end smallest_non_trivial_divisor_of_product_l1577_157752


namespace equation_solution_l1577_157779

theorem equation_solution (x : ℝ) : x^2 + x = 5 + Real.sqrt 5 ↔ x = Real.sqrt 5 ∨ x = -Real.sqrt 5 - 1 := by
  sorry

end equation_solution_l1577_157779


namespace geometric_sequence_sum_l1577_157761

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n < 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36 →
  a 3 + a 5 = -6 := by
sorry

end geometric_sequence_sum_l1577_157761


namespace solution_equals_answer_l1577_157711

/-- A perfect square is an integer that is the square of another integer. -/
def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m^2

/-- The set of all integer pairs (a, b) satisfying the given conditions. -/
def solution_set : Set (ℤ × ℤ) :=
  {p : ℤ × ℤ | is_perfect_square (p.1^2 - 4*p.2) ∧ is_perfect_square (p.2^2 - 4*p.1)}

/-- The set described in the answer. -/
def answer_set : Set (ℤ × ℤ) :=
  {p : ℤ × ℤ | (∃ n : ℤ, p = (0, n^2) ∨ p = (n^2, 0)) ∨
               (p.1 > 0 ∧ p.2 = -1 - p.1) ∨
               (p.2 > 0 ∧ p.1 = -1 - p.2) ∨
               p = (4, 4) ∨ p = (5, 6) ∨ p = (6, 5)}

theorem solution_equals_answer : solution_set = answer_set :=
  sorry

end solution_equals_answer_l1577_157711


namespace min_students_with_brown_eyes_and_lunch_box_l1577_157794

/-- Given a class with the following properties:
  * There are 30 students in total
  * 12 students have brown eyes
  * 20 students have a lunch box
  This theorem proves that the minimum number of students
  who have both brown eyes and a lunch box is 2. -/
theorem min_students_with_brown_eyes_and_lunch_box
  (total_students : ℕ)
  (brown_eyes : ℕ)
  (lunch_box : ℕ)
  (h1 : total_students = 30)
  (h2 : brown_eyes = 12)
  (h3 : lunch_box = 20) :
  brown_eyes + lunch_box - total_students ≥ 2 := by
  sorry

end min_students_with_brown_eyes_and_lunch_box_l1577_157794


namespace max_reflections_is_18_l1577_157716

/-- The angle between the lines in degrees -/
def angle : ℝ := 5

/-- The maximum angle for perpendicular reflection in degrees -/
def max_angle : ℝ := 90

/-- The maximum number of reflections -/
def max_reflections : ℕ := 18

/-- Theorem stating that the maximum number of reflections is 18 -/
theorem max_reflections_is_18 :
  ∀ n : ℕ, n * angle ≤ max_angle → n ≤ max_reflections :=
by sorry

end max_reflections_is_18_l1577_157716


namespace sum_of_squares_l1577_157781

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 16) (h2 : x * y = 28) : x^2 + y^2 = 200 := by
  sorry

end sum_of_squares_l1577_157781


namespace f_is_quadratic_l1577_157798

/-- Definition of a quadratic equation in standard form -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c)

/-- The specific equation we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end f_is_quadratic_l1577_157798


namespace butterflies_in_garden_l1577_157773

theorem butterflies_in_garden (total : ℕ) (flew_away_fraction : ℚ) (left : ℕ) : 
  total = 150 →
  flew_away_fraction = 11 / 13 →
  left = total - Int.floor (↑total * flew_away_fraction) →
  left = 23 := by
  sorry

end butterflies_in_garden_l1577_157773


namespace nonnegative_root_condition_l1577_157757

/-- A polynomial of degree 4 with coefficient q -/
def polynomial (q : ℝ) (x : ℝ) : ℝ := x^4 + q*x^3 + x^2 + q*x + 4

/-- The condition for the existence of a non-negative real root -/
def has_nonnegative_root (q : ℝ) : Prop :=
  ∃ x : ℝ, x ≥ 0 ∧ polynomial q x = 0

/-- The theorem stating the condition on q for the existence of a non-negative root -/
theorem nonnegative_root_condition (q : ℝ) : 
  has_nonnegative_root q ↔ q ≤ -2 * Real.sqrt 2 :=
sorry

end nonnegative_root_condition_l1577_157757


namespace cubic_factorization_l1577_157763

theorem cubic_factorization (x : ℝ) : x^3 - 9*x = x*(x + 3)*(x - 3) := by
  sorry

end cubic_factorization_l1577_157763


namespace cubic_equation_one_solution_l1577_157740

/-- The cubic equation in x with parameter b -/
def cubic_equation (x b : ℝ) : ℝ := x^3 - b*x^2 - 3*b*x + b^2 - 4

/-- The condition for the equation to have exactly one real solution -/
def has_one_real_solution (b : ℝ) : Prop :=
  ∃! x : ℝ, cubic_equation x b = 0

theorem cubic_equation_one_solution :
  ∀ b : ℝ, has_one_real_solution b ↔ b > 3 := by sorry

end cubic_equation_one_solution_l1577_157740


namespace probability_of_even_sum_l1577_157785

theorem probability_of_even_sum (p1 p2 : ℝ) 
  (h1 : p1 = 1/2)  -- Probability of even number from first wheel
  (h2 : p2 = 1/3)  -- Probability of even number from second wheel
  : p1 * p2 + (1 - p1) * (1 - p2) = 1/2 := by
  sorry

end probability_of_even_sum_l1577_157785


namespace two_distinct_roots_condition_l1577_157784

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := k * x^2 - 2 * x - 3

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  quadratic_equation k x₁ = 0 ∧ 
  quadratic_equation k x₂ = 0

-- Theorem statement
theorem two_distinct_roots_condition (k : ℝ) :
  has_two_distinct_real_roots k ↔ k > -1/3 ∧ k ≠ 0 := by
  sorry

end two_distinct_roots_condition_l1577_157784


namespace base_7_representation_and_properties_l1577_157750

def base_10_to_base_7 (n : ℕ) : List ℕ :=
  sorry

def count_even_digits (digits : List ℕ) : ℕ :=
  sorry

def sum_even_digits (digits : List ℕ) : ℕ :=
  sorry

theorem base_7_representation_and_properties :
  let base_7_repr := base_10_to_base_7 1250
  base_7_repr = [3, 4, 3, 4] ∧
  count_even_digits base_7_repr = 2 ∧
  ¬(sum_even_digits base_7_repr % 3 = 0) :=
by
  sorry

end base_7_representation_and_properties_l1577_157750


namespace pizza_payment_difference_l1577_157799

def pizza_problem (total_slices : ℕ) (plain_cost anchovy_cost onion_cost : ℚ)
  (anchovy_slices onion_slices : ℕ) (jerry_plain_slices : ℕ) : Prop :=
  let total_cost := plain_cost + anchovy_cost + onion_cost
  let cost_per_slice := total_cost / total_slices
  let jerry_slices := anchovy_slices + onion_slices + jerry_plain_slices
  let tom_slices := total_slices - jerry_slices
  let jerry_cost := cost_per_slice * jerry_slices
  let tom_cost := cost_per_slice * tom_slices
  jerry_cost - tom_cost = 11.36

theorem pizza_payment_difference :
  pizza_problem 12 12 3 2 4 4 2 := by sorry

end pizza_payment_difference_l1577_157799


namespace sqrt_x_plus_one_real_range_l1577_157795

theorem sqrt_x_plus_one_real_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by
sorry

end sqrt_x_plus_one_real_range_l1577_157795


namespace trichotomy_of_reals_l1577_157742

theorem trichotomy_of_reals : ∀ a b : ℝ, (a > b ∨ a = b ∨ a < b) ∧ 
  (¬(a > b ∧ a = b) ∧ ¬(a > b ∧ a < b) ∧ ¬(a = b ∧ a < b)) := by
  sorry

end trichotomy_of_reals_l1577_157742


namespace total_area_of_triangular_houses_l1577_157787

/-- The total area of three similar triangular houses -/
theorem total_area_of_triangular_houses (base : ℝ) (height : ℝ) (num_houses : ℕ) :
  base = 40 ∧ height = 20 ∧ num_houses = 3 →
  num_houses * (base * height / 2) = 1200 := by
  sorry

end total_area_of_triangular_houses_l1577_157787


namespace pablo_puzzle_completion_time_l1577_157789

/-- The number of days Pablo needs to complete all puzzles -/
def days_to_complete_puzzles (
  pieces_per_hour : ℕ
  ) (
  puzzles_300 : ℕ
  ) (
  puzzles_500 : ℕ
  ) (
  max_hours_per_day : ℕ
  ) : ℕ :=
  let total_pieces := puzzles_300 * 300 + puzzles_500 * 500
  let pieces_per_day := pieces_per_hour * max_hours_per_day
  (total_pieces + pieces_per_day - 1) / pieces_per_day

theorem pablo_puzzle_completion_time :
  days_to_complete_puzzles 100 8 5 7 = 7 := by
  sorry

end pablo_puzzle_completion_time_l1577_157789


namespace soda_difference_l1577_157745

theorem soda_difference (regular_soda : ℕ) (diet_soda : ℕ) 
  (h1 : regular_soda = 79) (h2 : diet_soda = 53) : 
  regular_soda - diet_soda = 26 := by
  sorry

end soda_difference_l1577_157745


namespace oplus_problem_l1577_157737

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation ⊕
def oplus : Element → Element → Element
  | Element.one, Element.one => Element.two
  | Element.one, Element.two => Element.three
  | Element.one, Element.three => Element.four
  | Element.one, Element.four => Element.one
  | Element.two, Element.one => Element.three
  | Element.two, Element.two => Element.one
  | Element.two, Element.three => Element.two
  | Element.two, Element.four => Element.four
  | Element.three, Element.one => Element.four
  | Element.three, Element.two => Element.two
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.one
  | Element.four, Element.two => Element.four
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.two

theorem oplus_problem :
  oplus (oplus Element.three Element.two) (oplus Element.four Element.one) = Element.three :=
by sorry

end oplus_problem_l1577_157737


namespace bus_children_difference_solve_bus_problem_l1577_157714

theorem bus_children_difference : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun initial_children children_off children_on final_children =>
    initial_children - children_off + children_on = final_children →
    children_off - children_on = 24

theorem solve_bus_problem :
  bus_children_difference 36 68 (68 - 24) 12 := by
  sorry

end bus_children_difference_solve_bus_problem_l1577_157714


namespace xiao_ming_correct_answers_l1577_157793

theorem xiao_ming_correct_answers 
  (total_questions : ℕ) 
  (correct_points : ℤ) 
  (wrong_points : ℤ) 
  (total_score : ℤ) 
  (h1 : total_questions = 20)
  (h2 : correct_points = 5)
  (h3 : wrong_points = -1)
  (h4 : total_score = 76) :
  ∃ (correct_answers : ℕ), 
    correct_answers ≤ total_questions ∧ 
    correct_points * correct_answers + wrong_points * (total_questions - correct_answers) = total_score ∧
    correct_answers = 16 := by
  sorry

#check xiao_ming_correct_answers

end xiao_ming_correct_answers_l1577_157793


namespace max_a_value_l1577_157771

/-- A lattice point in an xy-coordinate system is any point (x, y) where both x and y are integers. -/
def is_lattice_point (x y : ℤ) : Prop := True

/-- The equation y = mx + 3 -/
def equation (m : ℚ) (x y : ℤ) : Prop := y = m * x + 3

/-- The condition that the equation has no lattice point solutions for 0 < x ≤ 150 -/
def no_lattice_solutions (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x → x ≤ 150 → is_lattice_point x y → ¬equation m x y

/-- The theorem stating that 101/150 is the maximum value of a satisfying the given conditions -/
theorem max_a_value : 
  (∃ a : ℚ, a = 101/150 ∧ 
    (∀ m : ℚ, 2/3 < m → m < a → no_lattice_solutions m) ∧
    (∀ b : ℚ, b > a → ∃ m : ℚ, 2/3 < m ∧ m < b ∧ ¬no_lattice_solutions m)) :=
sorry

end max_a_value_l1577_157771


namespace tan_ratio_problem_l1577_157753

theorem tan_ratio_problem (x : ℝ) (h : Real.tan (x + π/4) = 2) : 
  Real.tan x / Real.tan (2*x) = 4/9 := by
  sorry

end tan_ratio_problem_l1577_157753


namespace kyle_money_after_snowboarding_l1577_157713

theorem kyle_money_after_snowboarding (dave_money : ℕ) (kyle_initial_money : ℕ) : 
  dave_money = 46 →
  kyle_initial_money = 3 * dave_money - 12 →
  kyle_initial_money / 3 = kyle_initial_money - (kyle_initial_money / 3) →
  kyle_initial_money - (kyle_initial_money / 3) = 84 :=
by
  sorry

end kyle_money_after_snowboarding_l1577_157713


namespace time_from_velocity_and_displacement_l1577_157754

/-- 
Given two equations relating velocity (V), displacement (S), time (t), 
acceleration (g), and initial velocity (V₀), prove that t can be expressed 
in terms of S, V, and V₀.
-/
theorem time_from_velocity_and_displacement 
  (V g t V₀ S : ℝ) 
  (hV : V = g * t + V₀) 
  (hS : S = (1/2) * g * t^2 + V₀ * t) : 
  t = 2 * S / (V + V₀) := by
sorry

end time_from_velocity_and_displacement_l1577_157754


namespace max_value_theorem_l1577_157791

theorem max_value_theorem (k : ℝ) (hk : k > 0) :
  (3 * k^3 + 3 * k) / ((3/2 * k^2 + 14) * (14 * k^2 + 3/2)) ≤ Real.sqrt 21 / 175 := by
  sorry

end max_value_theorem_l1577_157791


namespace f_max_min_implies_a_range_l1577_157703

/-- The function f(x) defined in terms of a real parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

/-- The derivative of f(x) with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a+6)

/-- Theorem stating that if f(x) has both a maximum and a minimum, then a is in the specified range -/
theorem f_max_min_implies_a_range (a : ℝ) :
  (∃ (x_max x_min : ℝ), ∀ x, f a x ≤ f a x_max ∧ f a x_min ≤ f a x) →
  a < -3 ∨ a > 6 :=
sorry

end f_max_min_implies_a_range_l1577_157703


namespace banknote_problem_l1577_157706

/-- Represents the number of banknotes of each denomination -/
structure Banknotes where
  ten : ℕ
  twenty : ℕ
  fifty : ℕ

/-- The problem constraints -/
def valid_banknotes (b : Banknotes) : Prop :=
  b.ten > 0 ∧ b.twenty > 0 ∧ b.fifty > 0 ∧
  b.ten + b.twenty + b.fifty = 24 ∧
  10 * b.ten + 20 * b.twenty + 50 * b.fifty = 1000

theorem banknote_problem :
  ∃ (b : Banknotes), valid_banknotes b ∧ b.twenty = 4 :=
sorry

end banknote_problem_l1577_157706


namespace stock_sale_before_brokerage_l1577_157724

/-- Calculates the total amount before brokerage given the cash realized and brokerage rate -/
def totalBeforeBrokerage (cashRealized : ℚ) (brokerageRate : ℚ) : ℚ :=
  cashRealized / (1 - brokerageRate)

/-- Theorem stating that for a stock sale with cash realization of 106.25 
    after a 1/4% brokerage fee, the total amount before brokerage is approximately 106.515 -/
theorem stock_sale_before_brokerage :
  let cashRealized : ℚ := 106.25
  let brokerageRate : ℚ := 1 / 400
  let result := totalBeforeBrokerage cashRealized brokerageRate
  ⌊result * 1000⌋ / 1000 = 106515 / 1000 := by
  sorry

end stock_sale_before_brokerage_l1577_157724


namespace sphere_surface_area_from_pyramid_l1577_157760

/-- The surface area of a sphere given a right square pyramid inscribed in it -/
theorem sphere_surface_area_from_pyramid (h V : ℝ) (h_pos : h > 0) (V_pos : V > 0) :
  let s := Real.sqrt (3 * V / h)
  let r := Real.sqrt ((s^2 + 2 * h^2) / 4)
  h = 4 → V = 16 → 4 * Real.pi * r^2 = 24 * Real.pi :=
by sorry

end sphere_surface_area_from_pyramid_l1577_157760


namespace magic_square_solution_l1577_157738

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  g : ℕ
  h : ℕ
  i : ℕ
  sum : ℕ
  row_sums : a + b + c = sum ∧ d + e + f = sum ∧ g + h + i = sum
  col_sums : a + d + g = sum ∧ b + e + h = sum ∧ c + f + i = sum
  diag_sums : a + e + i = sum ∧ c + e + g = sum

/-- The theorem to be proved -/
theorem magic_square_solution (ms : MagicSquare) 
  (h1 : ms.b = 25)
  (h2 : ms.c = 103)
  (h3 : ms.d = 3) :
  ms.a = 214 := by
  sorry

end magic_square_solution_l1577_157738


namespace a_range_l1577_157715

/-- Set A defined as { x | a ≤ x ≤ a+3 } -/
def A (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ a + 3 }

/-- Set B defined as { x | x < -1 or x > 5 } -/
def B : Set ℝ := { x | x < -1 ∨ x > 5 }

/-- Theorem stating that if A ∪ B = B, then a is in (-∞, -4) ∪ (5, +∞) -/
theorem a_range (a : ℝ) : (A a ∪ B = B) → a < -4 ∨ a > 5 := by
  sorry

end a_range_l1577_157715


namespace gcd_of_156_and_195_l1577_157744

theorem gcd_of_156_and_195 : Nat.gcd 156 195 = 39 := by
  sorry

end gcd_of_156_and_195_l1577_157744


namespace apple_picking_l1577_157766

theorem apple_picking (minjae_apples father_apples : ℝ) 
  (h1 : minjae_apples = 2.6)
  (h2 : father_apples = 5.98) :
  minjae_apples + father_apples = 8.58 := by
  sorry

end apple_picking_l1577_157766


namespace sphere_surface_area_l1577_157769

theorem sphere_surface_area (V : ℝ) (r : ℝ) (A : ℝ) : 
  V = 72 * Real.pi →
  V = (4/3) * Real.pi * r^3 →
  A = 4 * Real.pi * r^2 →
  A = 36 * Real.pi * 2^(2/3) :=
by sorry

end sphere_surface_area_l1577_157769


namespace premium_percentage_is_twenty_percent_l1577_157770

/-- Calculates the premium percentage on shares given investment details. -/
def calculate_premium_percentage (total_investment : ℚ) (face_value : ℚ) (dividend_rate : ℚ) (dividend_received : ℚ) : ℚ :=
  let num_shares := dividend_received / (dividend_rate * face_value / 100)
  let share_cost := total_investment / num_shares
  (share_cost - face_value) / face_value * 100

/-- Proves that the premium percentage is 20% given the specified conditions. -/
theorem premium_percentage_is_twenty_percent :
  let total_investment : ℚ := 14400
  let face_value : ℚ := 100
  let dividend_rate : ℚ := 5
  let dividend_received : ℚ := 600
  calculate_premium_percentage total_investment face_value dividend_rate dividend_received = 20 := by
  sorry

end premium_percentage_is_twenty_percent_l1577_157770


namespace two_intersection_points_l1577_157731

def quadratic_function (c : ℝ) (x : ℝ) : ℝ := 2*x^2 - 3*x - c

theorem two_intersection_points (c : ℝ) (h : c > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_function c x₁ = 0 ∧ quadratic_function c x₂ = 0 ∧
  ∀ x : ℝ, quadratic_function c x = 0 → x = x₁ ∨ x = x₂ :=
sorry

end two_intersection_points_l1577_157731


namespace police_force_ratio_l1577_157736

/-- Given a police force with female officers and officers on duty, prove the ratio of female officers to total officers on duty. -/
theorem police_force_ratio (total_female : ℕ) (total_on_duty : ℕ) (female_duty_percent : ℚ) : 
  total_female = 300 →
  total_on_duty = 240 →
  female_duty_percent = 2/5 →
  (female_duty_percent * total_female) / total_on_duty = 1/2 := by
  sorry

end police_force_ratio_l1577_157736


namespace min_value_inequality_l1577_157797

theorem min_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 9) :
  (x^2 + y^2)/(3*(x + y)) + (x^2 + z^2)/(3*(x + z)) + (y^2 + z^2)/(3*(y + z)) ≥ 3 := by
  sorry

end min_value_inequality_l1577_157797


namespace henry_final_balance_l1577_157721

/-- Henry's money transactions --/
def henry_money_problem (initial_amount received_from_relatives found_in_card spent_on_game donated_to_charity : ℚ) : Prop :=
  let total_received := initial_amount + received_from_relatives + found_in_card
  let total_spent := spent_on_game + donated_to_charity
  let final_balance := total_received - total_spent
  final_balance = 21.75

/-- Theorem stating that Henry's final balance is $21.75 --/
theorem henry_final_balance :
  henry_money_problem 11.75 18.50 5.25 10.60 3.15 := by
  sorry


end henry_final_balance_l1577_157721


namespace equation_solutions_l1577_157717

theorem equation_solutions :
  (∀ x : ℝ, x^2 + 6*x - 7 = 0 ↔ x = -7 ∨ x = 1) ∧
  (∀ x : ℝ, 4*x*(2*x+1) = 3*(2*x+1) ↔ x = -1/2 ∨ x = 3/4) := by
  sorry

end equation_solutions_l1577_157717


namespace card_A_total_percent_decrease_l1577_157712

def card_A_initial_value : ℝ := 150
def card_A_decrease_year1 : ℝ := 0.20
def card_A_decrease_year2 : ℝ := 0.30
def card_A_decrease_year3 : ℝ := 0.15

def card_A_value_after_three_years : ℝ :=
  card_A_initial_value * (1 - card_A_decrease_year1) * (1 - card_A_decrease_year2) * (1 - card_A_decrease_year3)

theorem card_A_total_percent_decrease :
  (card_A_initial_value - card_A_value_after_three_years) / card_A_initial_value = 0.524 := by
  sorry

end card_A_total_percent_decrease_l1577_157712


namespace point_in_region_implies_a_range_l1577_157749

theorem point_in_region_implies_a_range (a : ℝ) :
  (1 : ℝ) + (1 : ℝ) + a < 0 → a < -2 := by
  sorry

end point_in_region_implies_a_range_l1577_157749


namespace set_inclusion_implies_m_bound_l1577_157729

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x ≤ 6}
def B : Set ℝ := {x | x^2 - 4*x < 0}
def C (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 1}

-- State the theorem
theorem set_inclusion_implies_m_bound (m : ℝ) :
  C m ⊆ (C m ∩ B) → m ≤ 5/2 := by
  sorry


end set_inclusion_implies_m_bound_l1577_157729


namespace negation_of_universal_proposition_l1577_157788

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + 1 < 0) :=
by sorry

end negation_of_universal_proposition_l1577_157788


namespace ball_box_difference_l1577_157796

theorem ball_box_difference : 
  let white_balls : ℕ := 30
  let red_balls : ℕ := 18
  let balls_per_box : ℕ := 6
  let white_boxes := white_balls / balls_per_box
  let red_boxes := red_balls / balls_per_box
  white_boxes - red_boxes = 2 := by
sorry

end ball_box_difference_l1577_157796


namespace smallest_b_value_l1577_157774

theorem smallest_b_value (a b c : ℕ) : 
  (a * b * c = 360) → 
  (1 < a) → (a < b) → (b < c) → 
  (∀ b' : ℕ, (∃ a' c' : ℕ, a' * b' * c' = 360 ∧ 1 < a' ∧ a' < b' ∧ b' < c') → b ≤ b') → 
  b = 3 := by
sorry

end smallest_b_value_l1577_157774


namespace function_symmetry_l1577_157762

/-- The function f(x) = 3cos(2x + π/6) is symmetric about the point (π/6, 0) -/
theorem function_symmetry (f : ℝ → ℝ) (h : ∀ x, f x = 3 * Real.cos (2 * x + π / 6)) :
  ∀ x, f (π / 3 - x) = f (π / 3 + x) :=
sorry

end function_symmetry_l1577_157762


namespace min_megabytes_for_plan_y_l1577_157768

/-- The cost in cents for Plan X given m megabytes -/
def plan_x_cost (m : ℕ) : ℕ := 15 * m

/-- The cost in cents for Plan Y given m megabytes -/
def plan_y_cost (m : ℕ) : ℕ := 3000 + 7 * m

/-- Predicate to check if Plan Y is cheaper for a given number of megabytes -/
def plan_y_cheaper (m : ℕ) : Prop := plan_y_cost m < plan_x_cost m

theorem min_megabytes_for_plan_y : ∀ m : ℕ, m ≥ 376 → plan_y_cheaper m ∧ ∀ n : ℕ, n < 376 → ¬plan_y_cheaper n :=
  sorry

end min_megabytes_for_plan_y_l1577_157768


namespace sum_of_integers_l1577_157719

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val^2 + y.val^2 = 193) 
  (h2 : x.val * y.val = 84) : 
  x.val + y.val = 19 := by
sorry

end sum_of_integers_l1577_157719


namespace rockham_soccer_league_members_l1577_157775

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 4

/-- The additional cost of a T-shirt compared to a pair of socks in dollars -/
def tshirt_additional_cost : ℕ := 5

/-- The total cost for all members in dollars -/
def total_cost : ℕ := 2366

/-- The number of pairs of socks each member needs -/
def socks_per_member : ℕ := 2

/-- The number of T-shirts each member needs -/
def tshirts_per_member : ℕ := 2

/-- The cost of equipment for one member -/
def member_cost : ℕ := socks_per_member * sock_cost + 
                       tshirts_per_member * (sock_cost + tshirt_additional_cost)

/-- The number of members in the Rockham Soccer League -/
def number_of_members : ℕ := total_cost / member_cost

theorem rockham_soccer_league_members : number_of_members = 91 := by
  sorry

end rockham_soccer_league_members_l1577_157775


namespace min_product_with_98_zeros_l1577_157727

/-- The number of trailing zeros in a positive integer -/
def trailingZeros (n : ℕ+) : ℕ := sorry

/-- The concatenation of two positive integers -/
def concat (a b : ℕ+) : ℕ+ := sorry

/-- The statement of the problem -/
theorem min_product_with_98_zeros :
  ∃ (m n : ℕ+),
    (∀ (x y : ℕ+), trailingZeros (x^x.val * y^y.val) = 98 → m.val * n.val ≤ x.val * y.val) ∧
    trailingZeros (m^m.val * n^n.val) = 98 ∧
    trailingZeros (concat (concat m m) (concat n n)) = 98 ∧
    m.val * n.val = 7350 := by
  sorry

end min_product_with_98_zeros_l1577_157727


namespace proposition_truth_l1577_157751

theorem proposition_truth (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : p ∨ q) : 
  (¬p) ∨ (¬q) := by
sorry

end proposition_truth_l1577_157751


namespace acme_vowel_soup_combinations_l1577_157792

/-- Represents the number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- Represents the number of times each vowel appears in the soup -/
def vowel_occurrences : ℕ := 6

/-- Represents the number of wildcard characters in the soup -/
def num_wildcards : ℕ := 1

/-- Represents the length of the words to be formed -/
def word_length : ℕ := 6

/-- Represents the total number of character choices for each position in the word -/
def choices_per_position : ℕ := num_vowels + num_wildcards

/-- Theorem stating that the number of possible six-letter words is 46656 -/
theorem acme_vowel_soup_combinations :
  choices_per_position ^ word_length = 46656 := by
  sorry

end acme_vowel_soup_combinations_l1577_157792


namespace alpha_value_l1577_157710

/-- Given that α is inversely proportional to β and directly proportional to γ,
    prove that α = 2.5 when β = 30 and γ = 6, given that α = 5 when β = 15 and γ = 3 -/
theorem alpha_value (α β γ : ℝ) (h1 : ∃ k : ℝ, α * β = k)
    (h2 : ∃ j : ℝ, α * γ = j) (h3 : α = 5 ∧ β = 15 ∧ γ = 3) :
  α = 2.5 ∧ β = 30 ∧ γ = 6 := by
  sorry

end alpha_value_l1577_157710


namespace least_common_multiple_of_band_sets_l1577_157756

theorem least_common_multiple_of_band_sets : Nat.lcm (Nat.lcm 2 9) 14 = 126 := by
  sorry

end least_common_multiple_of_band_sets_l1577_157756


namespace triangle_side_length_l1577_157755

theorem triangle_side_length (a b c : ℝ) (B : ℝ) : 
  a = 2 → B = π / 3 → c = 3 → b = Real.sqrt 7 := by
  sorry

end triangle_side_length_l1577_157755


namespace intersection_of_sets_l1577_157759

open Set

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {3, 4, 5}
  A ∩ B = {3} := by
sorry

end intersection_of_sets_l1577_157759


namespace trevor_coin_count_l1577_157709

theorem trevor_coin_count : 
  let total_coins : ℕ := 77
  let quarters : ℕ := 29
  let dimes : ℕ := total_coins - quarters
  total_coins - quarters = dimes :=
by sorry

end trevor_coin_count_l1577_157709


namespace altitude_B_correct_median_A_correct_circumcircle_correct_l1577_157701

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (0, -1)
def C : ℝ × ℝ := (-2, 1)

-- Define the altitude from B to BC
def altitude_B (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the median from A to AC
def median_A (x : ℝ) : Prop := x = -1

-- Define the circumcircle of triangle ABC
def circumcircle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 1 = 0

-- Theorem statements
theorem altitude_B_correct :
  ∀ x y : ℝ, altitude_B x y ↔ (x - y + 1 = 0) :=
sorry

theorem median_A_correct :
  ∀ x : ℝ, median_A x ↔ (x = -1) :=
sorry

theorem circumcircle_correct :
  ∀ x y : ℝ, circumcircle x y ↔ (x^2 + y^2 + 2*x - 1 = 0) :=
sorry

end altitude_B_correct_median_A_correct_circumcircle_correct_l1577_157701


namespace baker_revenue_difference_l1577_157765

/-- Baker's sales and pricing information --/
structure BakerSales where
  usual_pastries : ℕ
  usual_bread : ℕ
  today_pastries : ℕ
  today_bread : ℕ
  pastry_price : ℕ
  bread_price : ℕ

/-- Calculate the difference between daily average and today's revenue --/
def revenue_difference (sales : BakerSales) : ℕ :=
  let usual_revenue := sales.usual_pastries * sales.pastry_price + sales.usual_bread * sales.bread_price
  let today_revenue := sales.today_pastries * sales.pastry_price + sales.today_bread * sales.bread_price
  today_revenue - usual_revenue

/-- Theorem stating the revenue difference for the given sales information --/
theorem baker_revenue_difference :
  revenue_difference ⟨20, 10, 14, 25, 2, 4⟩ = 48 := by
  sorry

end baker_revenue_difference_l1577_157765


namespace quadratic_inequality_solution_l1577_157705

theorem quadratic_inequality_solution (x : ℝ) :
  (-x^2 + 5*x - 4 < 0) ↔ (1 < x ∧ x < 4) := by
  sorry

end quadratic_inequality_solution_l1577_157705


namespace equations_same_graph_l1577_157777

-- Define the three equations
def equation_I (x y : ℝ) : Prop := y = x^2 - 1
def equation_II (x y : ℝ) : Prop := x ≠ 1 → y = (x^3 - x) / (x - 1)
def equation_III (x y : ℝ) : Prop := (x - 1) * y = x^3 - x

-- Define what it means for two equations to have the same graph
def same_graph (eq1 eq2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq1 x y ↔ eq2 x y

-- Theorem statement
theorem equations_same_graph :
  (same_graph equation_II equation_III) ∧
  (¬ same_graph equation_I equation_II) ∧
  (¬ same_graph equation_I equation_III) :=
sorry

end equations_same_graph_l1577_157777
