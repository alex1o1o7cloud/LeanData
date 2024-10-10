import Mathlib

namespace expression_value_l3957_395742

theorem expression_value (a b c : ℝ) (h : a^2 + b = b^2 + c ∧ b^2 + c = c^2 + a) :
  a * (a^2 - b^2) + b * (b^2 - c^2) + c * (c^2 - a^2) = 0 := by
  sorry

end expression_value_l3957_395742


namespace special_numbers_count_l3957_395710

def count_multiples (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

def count_special_numbers (upper_bound : ℕ) : ℕ :=
  count_multiples upper_bound 5 + count_multiples upper_bound 7 - count_multiples upper_bound 35

theorem special_numbers_count :
  count_special_numbers 3000 = 943 := by
  sorry

end special_numbers_count_l3957_395710


namespace no_perfect_power_consecutive_product_l3957_395744

theorem no_perfect_power_consecutive_product : 
  ∀ n : ℕ, ¬∃ (a k : ℕ), k > 1 ∧ n * (n + 1) = a ^ k :=
sorry

end no_perfect_power_consecutive_product_l3957_395744


namespace square_rectangle_area_relation_l3957_395726

theorem square_rectangle_area_relation :
  ∀ x : ℝ,
  let square_side := x - 3
  let rect_length := x - 2
  let rect_width := x + 5
  let square_area := square_side ^ 2
  let rect_area := rect_length * rect_width
  rect_area = 3 * square_area →
  ∃ x₁ x₂ : ℝ, 
    (x₁ + x₂ = 21/2) ∧ 
    (∀ y : ℝ, rect_area = 3 * square_area → y = x₁ ∨ y = x₂) :=
by
  sorry

#check square_rectangle_area_relation

end square_rectangle_area_relation_l3957_395726


namespace coefficient_x_cubed_in_expansion_l3957_395733

/-- The coefficient of x^3 in the expansion of (1-1/x)(1+x)^5 is 5 -/
theorem coefficient_x_cubed_in_expansion : ∃ (f : ℝ → ℝ),
  (∀ x ≠ 0, f x = (1 - 1/x) * (1 + x)^5) ∧
  (∃ a b c d e g : ℝ, ∀ x ≠ 0, f x = a + b*x + c*x^2 + 5*x^3 + d*x^4 + e*x^5 + g/x) :=
by sorry

end coefficient_x_cubed_in_expansion_l3957_395733


namespace soak_time_for_marinara_stain_l3957_395724

/-- The time needed to soak clothes for grass and marinara stains -/
theorem soak_time_for_marinara_stain 
  (grass_stain_time : ℕ) 
  (num_grass_stains : ℕ) 
  (num_marinara_stains : ℕ) 
  (total_soak_time : ℕ) 
  (h1 : grass_stain_time = 4)
  (h2 : num_grass_stains = 3)
  (h3 : num_marinara_stains = 1)
  (h4 : total_soak_time = 19) :
  total_soak_time - (grass_stain_time * num_grass_stains) = 7 := by
sorry

end soak_time_for_marinara_stain_l3957_395724


namespace f_increasing_decreasing_l3957_395720

noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

theorem f_increasing_decreasing :
  let f : ℝ → ℝ := λ x => x^2 - Real.log x
  (∀ x, x > 0 → f x ∈ Set.univ) ∧
  (∀ x y, x > Real.sqrt 2 / 2 → y > Real.sqrt 2 / 2 → x < y → f x < f y) ∧
  (∀ x y, x > 0 → y > 0 → x < Real.sqrt 2 / 2 → y < Real.sqrt 2 / 2 → x < y → f x > f y) :=
by sorry

end f_increasing_decreasing_l3957_395720


namespace new_man_weight_l3957_395748

/-- Given a group of 8 men, if replacing a 60 kg man with a new man increases the average weight by 1 kg, then the new man weighs 68 kg. -/
theorem new_man_weight (initial_average : ℝ) : 
  (8 * initial_average + 68 = 8 * (initial_average + 1) + 60) → 68 = 68 := by
  sorry

end new_man_weight_l3957_395748


namespace function_zero_implies_a_bound_l3957_395754

/-- If the function f(x) = e^x - 2x + a has a zero, then a ≤ 2ln2 - 2 -/
theorem function_zero_implies_a_bound (a : ℝ) : 
  (∃ x : ℝ, Real.exp x - 2 * x + a = 0) → a ≤ 2 * Real.log 2 - 2 := by
  sorry

end function_zero_implies_a_bound_l3957_395754


namespace parabola_point_order_l3957_395758

/-- Given a parabola y = 2x² - 4x + m and three points on it, prove their y-coordinates are in ascending order -/
theorem parabola_point_order (m : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : y₁ = 2 * 3^2 - 4 * 3 + m)
  (h₂ : y₂ = 2 * 4^2 - 4 * 4 + m)
  (h₃ : y₃ = 2 * 5^2 - 4 * 5 + m) :
  y₁ < y₂ ∧ y₂ < y₃ := by
sorry

end parabola_point_order_l3957_395758


namespace monochromatic_triangle_in_17_vertex_graph_l3957_395782

/-- A coloring of edges in a complete graph -/
def EdgeColoring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A complete graph has a monochromatic triangle if there exist three vertices
    such that all edges between them have the same color -/
def has_monochromatic_triangle (n : ℕ) (coloring : EdgeColoring n) : Prop :=
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    coloring i j = coloring j k ∧ coloring j k = coloring i k

/-- In any complete graph with 17 vertices where each edge is colored in one of three colors,
    there exist three vertices such that all edges between them are the same color -/
theorem monochromatic_triangle_in_17_vertex_graph :
  ∀ (coloring : EdgeColoring 17), has_monochromatic_triangle 17 coloring :=
sorry

end monochromatic_triangle_in_17_vertex_graph_l3957_395782


namespace percentage_runs_by_running_l3957_395705

def total_runs : ℕ := 150
def boundaries : ℕ := 6
def sixes : ℕ := 4
def no_balls : ℕ := 8
def wide_balls : ℕ := 5
def leg_byes : ℕ := 2

def runs_from_boundaries : ℕ := boundaries * 4
def runs_from_sixes : ℕ := sixes * 6
def runs_not_from_bat : ℕ := no_balls + wide_balls + leg_byes

def runs_by_running : ℕ := total_runs - runs_not_from_bat - (runs_from_boundaries + runs_from_sixes)

theorem percentage_runs_by_running :
  (runs_by_running : ℚ) / total_runs * 100 = 58 := by
  sorry

end percentage_runs_by_running_l3957_395705


namespace simple_interest_rate_calculation_l3957_395747

theorem simple_interest_rate_calculation (P A T : ℝ) (h1 : P = 750) (h2 : A = 950) (h3 : T = 5) :
  let SI := A - P
  let R := (SI * 100) / (P * T)
  R = (200 * 100) / (750 * 5) := by sorry

end simple_interest_rate_calculation_l3957_395747


namespace extreme_values_of_f_l3957_395711

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 5

theorem extreme_values_of_f :
  (∃ x : ℝ, f x = 10 ∧ ∀ y : ℝ, f y ≤ f x) ∧
  (∃ x : ℝ, f x = -22 ∧ ∀ y : ℝ, f y ≥ f x) :=
sorry

end extreme_values_of_f_l3957_395711


namespace base7_to_base10_conversion_l3957_395727

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number -/
def base7Number : List Nat := [6, 3, 4, 5, 2]

theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 6740 := by
  sorry

end base7_to_base10_conversion_l3957_395727


namespace area_of_specific_quadrilateral_l3957_395700

/-- Represents a quadrilateral EFGH with specific angle and side length properties -/
structure Quadrilateral :=
  (EF : ℝ)
  (FG : ℝ)
  (GH : ℝ)
  (angle_F : ℝ)
  (angle_G : ℝ)

/-- Calculates the area of the quadrilateral EFGH -/
def area (q : Quadrilateral) : ℝ :=
  sorry

/-- Theorem stating that for a quadrilateral EFGH with given properties, its area is (77√2)/4 -/
theorem area_of_specific_quadrilateral :
  ∀ (q : Quadrilateral),
    q.EF = 5 ∧
    q.FG = 7 ∧
    q.GH = 6 ∧
    q.angle_F = 135 ∧
    q.angle_G = 135 →
    area q = (77 * Real.sqrt 2) / 4 :=
by sorry

end area_of_specific_quadrilateral_l3957_395700


namespace equation_represents_pair_of_lines_l3957_395786

/-- The equation x^2 - xy - 6y^2 = 0 represents a pair of straight lines -/
theorem equation_represents_pair_of_lines : ∃ (m₁ m₂ : ℝ),
  ∀ (x y : ℝ), x^2 - x*y - 6*y^2 = 0 ↔ (x = m₁*y ∨ x = m₂*y) :=
by sorry

end equation_represents_pair_of_lines_l3957_395786


namespace exponential_system_solution_l3957_395773

theorem exponential_system_solution (x y : ℝ) : 
  (4 : ℝ)^x = 256^(y + 1) → (27 : ℝ)^y = 3^(x - 2) → x = -4 ∧ y = -2 := by
  sorry

end exponential_system_solution_l3957_395773


namespace factorial_345_trailing_zeros_l3957_395755

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_345_trailing_zeros :
  trailing_zeros 345 = 84 := by
  sorry

end factorial_345_trailing_zeros_l3957_395755


namespace no_periodic_sum_with_given_periods_l3957_395731

/-- A function is periodic if it takes at least two different values and there exists a positive period. -/
def Periodic (f : ℝ → ℝ) : Prop :=
  (∃ x y, f x ≠ f y) ∧ (∃ p > 0, ∀ x, f (x + p) = f x)

/-- The period of a function. -/
def Period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

/-- Theorem: There do not exist periodic functions g and h with periods 2 and π/2 respectively,
    such that g + h is also a periodic function. -/
theorem no_periodic_sum_with_given_periods :
  ¬ ∃ (g h : ℝ → ℝ),
    Periodic g ∧ Periodic h ∧ Period g 2 ∧ Period h (π / 2) ∧ Periodic (g + h) :=
by sorry

end no_periodic_sum_with_given_periods_l3957_395731


namespace no_simultaneously_safe_numbers_l3957_395704

def is_p_safe (n p : ℕ) : Prop :=
  ∀ k : ℤ, |n - k * p| > 3

def simultaneously_safe (n : ℕ) : Prop :=
  is_p_safe n 5 ∧ is_p_safe n 9 ∧ is_p_safe n 11

theorem no_simultaneously_safe_numbers : 
  ¬∃ n : ℕ, n > 0 ∧ n ≤ 5000 ∧ simultaneously_safe n :=
sorry

end no_simultaneously_safe_numbers_l3957_395704


namespace quadratic_equation_roots_l3957_395751

theorem quadratic_equation_roots (k : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + (2*k + 1)*x + k^2 + 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) →
  (k > 3/4 ∧ (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ * x₂ = 5 → k = 2)) := by
  sorry

end quadratic_equation_roots_l3957_395751


namespace min_distance_to_line_l3957_395799

/-- The minimum distance from the origin to a point on the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : 
  let line := {p : ℝ × ℝ | p.1 + p.2 = 4}
  ∃ d : ℝ, d = 2 * Real.sqrt 2 ∧ 
    ∀ p ∈ line, Real.sqrt (p.1^2 + p.2^2) ≥ d ∧
    ∃ q ∈ line, Real.sqrt (q.1^2 + q.2^2) = d :=
by sorry

end min_distance_to_line_l3957_395799


namespace rajas_household_expenditure_percentage_l3957_395736

theorem rajas_household_expenditure_percentage 
  (monthly_income : ℝ) 
  (clothes_percentage : ℝ) 
  (medicines_percentage : ℝ) 
  (savings : ℝ) 
  (h1 : monthly_income = 37500) 
  (h2 : clothes_percentage = 20) 
  (h3 : medicines_percentage = 5) 
  (h4 : savings = 15000) : 
  (monthly_income - (monthly_income * clothes_percentage / 100 + 
   monthly_income * medicines_percentage / 100 + savings)) / monthly_income * 100 = 35 := by
sorry

end rajas_household_expenditure_percentage_l3957_395736


namespace werewolf_identity_l3957_395768

-- Define the inhabitants
inductive Inhabitant : Type
| A : Inhabitant
| B : Inhabitant
| C : Inhabitant

-- Define the possible states
inductive State
| Knight : State
| Liar : State
| Werewolf : State

def is_knight (i : Inhabitant) (state : Inhabitant → State) : Prop :=
  state i = State.Knight

def is_liar (i : Inhabitant) (state : Inhabitant → State) : Prop :=
  state i = State.Liar

def is_werewolf (i : Inhabitant) (state : Inhabitant → State) : Prop :=
  state i = State.Werewolf

-- A's statement: At least one of us is a knight
def A_statement (state : Inhabitant → State) : Prop :=
  ∃ i : Inhabitant, is_knight i state

-- B's statement: At least one of us is a liar
def B_statement (state : Inhabitant → State) : Prop :=
  ∃ i : Inhabitant, is_liar i state

-- Theorem to prove
theorem werewolf_identity (state : Inhabitant → State) :
  -- At least one is a werewolf
  (∃ i : Inhabitant, is_werewolf i state) →
  -- None are both knight and werewolf
  (∀ i : Inhabitant, ¬(is_knight i state ∧ is_werewolf i state)) →
  -- A's statement is true if A is a knight, false if A is a liar
  ((is_knight Inhabitant.A state → A_statement state) ∧
   (is_liar Inhabitant.A state → ¬A_statement state)) →
  -- B's statement is true if B is a knight, false if B is a liar
  ((is_knight Inhabitant.B state → B_statement state) ∧
   (is_liar Inhabitant.B state → ¬B_statement state)) →
  -- C is the werewolf
  is_werewolf Inhabitant.C state :=
by sorry

end werewolf_identity_l3957_395768


namespace black_pens_count_l3957_395794

theorem black_pens_count (green_pens red_pens : ℕ) 
  (prob_neither_red_nor_green : ℚ) :
  green_pens = 5 →
  red_pens = 7 →
  prob_neither_red_nor_green = 1/3 →
  ∃ (total_pens black_pens : ℕ),
    total_pens = green_pens + red_pens + black_pens ∧
    (black_pens : ℚ) / total_pens = prob_neither_red_nor_green ∧
    black_pens = 6 :=
by sorry

end black_pens_count_l3957_395794


namespace expression_simplification_l3957_395784

theorem expression_simplification (a b : ℝ) (h1 : a = Real.sqrt 3 - 3) (h2 : b = 3) :
  1 - (a - b) / (a + 2*b) / ((a^2 - b^2) / (a^2 + 4*a*b + 4*b^2)) = -Real.sqrt 3 := by
  sorry

end expression_simplification_l3957_395784


namespace parkway_elementary_students_l3957_395749

/-- The number of students in the fifth grade at Parkway Elementary School -/
def total_students : ℕ := 500

/-- The number of boys in the fifth grade -/
def boys : ℕ := 350

/-- The number of students playing soccer -/
def soccer_players : ℕ := 250

/-- The percentage of soccer players who are boys -/
def boys_soccer_percentage : ℚ := 86 / 100

/-- The number of girls not playing soccer -/
def girls_not_soccer : ℕ := 115

/-- Theorem stating that the total number of students is 500 -/
theorem parkway_elementary_students :
  total_students = boys + girls_not_soccer + (soccer_players - (boys_soccer_percentage * soccer_players).num) :=
by sorry

end parkway_elementary_students_l3957_395749


namespace percentage_increase_l3957_395781

theorem percentage_increase (original_earnings new_earnings : ℝ) 
  (h1 : original_earnings = 55)
  (h2 : new_earnings = 60) :
  ((new_earnings - original_earnings) / original_earnings) * 100 = 9.09 := by
  sorry

end percentage_increase_l3957_395781


namespace complex_exp_thirteen_pi_i_over_three_l3957_395714

theorem complex_exp_thirteen_pi_i_over_three :
  Complex.exp (13 * π * Complex.I / 3) = (1 / 2 : ℂ) + Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end complex_exp_thirteen_pi_i_over_three_l3957_395714


namespace train_length_calculation_l3957_395732

/-- Given a train crossing a bridge, calculate its length. -/
theorem train_length_calculation (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 255 →
  ∃ (train_length : ℝ), train_length = train_speed * crossing_time - bridge_length ∧ train_length = 120 := by
  sorry

end train_length_calculation_l3957_395732


namespace relation_between_x_and_y_l3957_395743

theorem relation_between_x_and_y (p : ℝ) :
  let x : ℝ := 3 + 3^p
  let y : ℝ := 3 + 3^(-p)
  y = (3*x - 8) / (x - 3) :=
by sorry

end relation_between_x_and_y_l3957_395743


namespace green_balloons_count_l3957_395740

theorem green_balloons_count (total : ℕ) (red : ℕ) (green : ℕ) : 
  total = 17 → red = 8 → total = red + green → green = 9 := by
  sorry

end green_balloons_count_l3957_395740


namespace intersection_sum_l3957_395738

/-- Given two equations and their intersection points, prove the sum of x-coordinates is 0 and the sum of y-coordinates is 3 -/
theorem intersection_sum (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) : 
  (y₁ = x₁^3 - 3*x₁ + 2) →
  (y₂ = x₂^3 - 3*x₂ + 2) →
  (y₃ = x₃^3 - 3*x₃ + 2) →
  (2*x₁ + 3*y₁ = 3) →
  (2*x₂ + 3*y₂ = 3) →
  (2*x₃ + 3*y₃ = 3) →
  (x₁ + x₂ + x₃ = 0 ∧ y₁ + y₂ + y₃ = 3) :=
by sorry

end intersection_sum_l3957_395738


namespace real_part_of_complex_fraction_l3957_395707

theorem real_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.re ((2 : ℂ) + i) / i = 1 := by
  sorry

end real_part_of_complex_fraction_l3957_395707


namespace trapezoid_perimeter_l3957_395760

/-- A symmetric trapezoid EFGH with given properties -/
structure SymmetricTrapezoid :=
  (EF : ℝ)  -- Length of top base EF
  (GH : ℝ)  -- Length of bottom base GH
  (height : ℝ)  -- Height from EF to GH
  (isSymmetric : Bool)  -- Is the trapezoid symmetric?
  (EFGHEqual : Bool)  -- Are EF and GH equal in length?

/-- Properties of the specific trapezoid in the problem -/
def problemTrapezoid : SymmetricTrapezoid :=
  { EF := 10,
    GH := 22,
    height := 6,
    isSymmetric := true,
    EFGHEqual := true }

/-- Theorem stating the perimeter of the trapezoid -/
theorem trapezoid_perimeter (t : SymmetricTrapezoid) 
  (h1 : t.EF = 10)
  (h2 : t.GH = t.EF + 12)
  (h3 : t.height = 6)
  (h4 : t.isSymmetric = true)
  (h5 : t.EFGHEqual = true) :
  (t.EF + t.GH + 2 * Real.sqrt (t.height^2 + ((t.GH - t.EF) / 2)^2)) = 32 + 12 * Real.sqrt 2 :=
sorry

end trapezoid_perimeter_l3957_395760


namespace subset_implies_m_range_l3957_395709

theorem subset_implies_m_range (m : ℝ) : 
  let A := Set.Iic m
  let B := Set.Ioo 1 2
  B ⊆ A → m ∈ Set.Ici 2 := by
sorry

end subset_implies_m_range_l3957_395709


namespace no_multiple_of_five_l3957_395766

theorem no_multiple_of_five : ∀ C : ℕ, C < 10 → ¬(∃ k : ℕ, 200 + 10 * C + 4 = 5 * k) := by
  sorry

end no_multiple_of_five_l3957_395766


namespace parallel_line_equation_l3957_395797

/-- Given a point A and a line L, this theorem proves that the equation
    4x + y - 14 = 0 represents the line passing through A and parallel to L. -/
theorem parallel_line_equation (A : ℝ × ℝ) (L : Set (ℝ × ℝ)) : 
  A.1 = 3 ∧ A.2 = 2 →
  L = {(x, y) | 4 * x + y - 2 = 0} →
  {(x, y) | 4 * x + y - 14 = 0} = 
    {(x, y) | ∃ (t : ℝ), x = A.1 + t ∧ y = A.2 - 4 * t} :=
by sorry

end parallel_line_equation_l3957_395797


namespace rug_area_theorem_l3957_395780

/-- Given three overlapping rugs, calculates their combined area -/
def combined_rug_area (total_floor_area two_layer_area three_layer_area : ℝ) : ℝ :=
  let one_layer_area := total_floor_area - two_layer_area - three_layer_area
  one_layer_area + 2 * two_layer_area + 3 * three_layer_area

/-- Theorem stating that the combined area of three rugs is 200 square meters
    given the specified overlapping conditions -/
theorem rug_area_theorem :
  combined_rug_area 138 24 19 = 200 := by
  sorry


end rug_area_theorem_l3957_395780


namespace quadratic_equation_single_solution_sum_l3957_395791

theorem quadratic_equation_single_solution_sum (b : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ 3 * x^2 + b * x + 6 * x + 10
  (∃! x, f x = 0) → 
  ∃ b₁ b₂, b = b₁ ∨ b = b₂ ∧ b₁ + b₂ = -12 :=
by sorry

end quadratic_equation_single_solution_sum_l3957_395791


namespace sine_cosine_difference_l3957_395721

theorem sine_cosine_difference (θ₁ θ₂ : Real) :
  Real.sin (37.5 * π / 180) * Real.cos (7.5 * π / 180) -
  Real.cos (37.5 * π / 180) * Real.sin (7.5 * π / 180) = 1/2 := by
  sorry

end sine_cosine_difference_l3957_395721


namespace greatest_three_digit_multiple_of_17_l3957_395775

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 986 ∧ 
  17 ∣ n ∧
  100 ≤ n ∧ 
  n ≤ 999 ∧
  ∀ m : ℕ, (17 ∣ m ∧ 100 ≤ m ∧ m ≤ 999) → m ≤ n :=
by sorry

end greatest_three_digit_multiple_of_17_l3957_395775


namespace rectangle_side_length_l3957_395776

/-- Given a rectangle with its bottom side on the x-axis from (-a, 0) to (a, 0),
    its top side on the parabola y = x^2, and its area equal to 81,
    prove that the length of its side parallel to the x-axis is 2∛(40.5). -/
theorem rectangle_side_length (a : ℝ) : 
  (2 * a * a^2 = 81) →  -- Area of the rectangle
  (2 * a = 2 * (40.5 : ℝ)^(1/3)) := by
  sorry

end rectangle_side_length_l3957_395776


namespace imaginary_part_of_z_l3957_395723

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = Complex.I) : 
  z.im = 1/2 := by
  sorry

end imaginary_part_of_z_l3957_395723


namespace student_assignment_count_l3957_395701

theorem student_assignment_count : ∀ (n m : ℕ),
  n = 4 ∧ m = 3 →
  (Nat.choose n 2 * (Nat.factorial m)) = (m * Nat.choose n 2 * 2) :=
by sorry

end student_assignment_count_l3957_395701


namespace inequality_solution_l3957_395769

theorem inequality_solution : 
  {x : ℕ | 2 * x - 1 < 5} = {0, 1, 2} := by sorry

end inequality_solution_l3957_395769


namespace quadratic_equation_solutions_l3957_395735

theorem quadratic_equation_solutions (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 := by
  sorry

end quadratic_equation_solutions_l3957_395735


namespace sequence_term_proof_l3957_395777

def sequence_sum (n : ℕ) : ℕ := 2^n

def sequence_term (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2^(n-1)

theorem sequence_term_proof :
  ∀ n : ℕ, n ≥ 1 →
    sequence_term n = (if n = 1 then sequence_sum 1 else sequence_sum n - sequence_sum (n-1)) :=
by sorry

end sequence_term_proof_l3957_395777


namespace root_exists_in_interval_l3957_395779

def f (x : ℝ) := x^2 + 3*x - 5

theorem root_exists_in_interval :
  ∃ x : ℝ, 1.1 < x ∧ x < 1.2 ∧ f x = 0 :=
by
  have h1 : f 1.1 < 0 := by sorry
  have h2 : f 1.2 > 0 := by sorry
  sorry

#check root_exists_in_interval

end root_exists_in_interval_l3957_395779


namespace division_simplification_l3957_395725

theorem division_simplification : 240 / (12 + 12 * 2 - 3) = 240 / 33 := by
  sorry

end division_simplification_l3957_395725


namespace largest_possible_median_l3957_395728

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  s.card = 5 ∧ (s.filter (λ x => x ≤ m)).card ≥ 3 ∧ (s.filter (λ x => x ≥ m)).card ≥ 3

theorem largest_possible_median :
  ∀ x y : ℤ, y = 2 * x →
  ∃ m : ℤ, is_median m {x, y, 3, 7, 9} ∧
    ∀ m' : ℤ, is_median m' {x, y, 3, 7, 9} → m' ≤ m ∧ m = 7 :=
by sorry

end largest_possible_median_l3957_395728


namespace incorrect_fraction_equality_l3957_395718

theorem incorrect_fraction_equality (x y : ℝ) (h : x ≠ -y) :
  ¬ ((x - y) / (x + y) = (y - x) / (y + x)) := by
  sorry

end incorrect_fraction_equality_l3957_395718


namespace vertical_angles_equal_parallel_lines_corresponding_angles_equal_l3957_395767

-- Define the concept of an angle
def Angle : Type := ℝ

-- Define the concept of a line
def Line : Type := Unit

-- Define vertical angles
def are_vertical (a b : Angle) : Prop := sorry

-- Define parallel lines
def are_parallel (l1 l2 : Line) : Prop := sorry

-- Define corresponding angles
def are_corresponding (a b : Angle) (l1 l2 : Line) : Prop := sorry

-- Theorem: Vertical angles are equal
theorem vertical_angles_equal (a b : Angle) : 
  are_vertical a b → a = b := by sorry

-- Theorem: If two lines are parallel, then corresponding angles are equal
theorem parallel_lines_corresponding_angles_equal (a b : Angle) (l1 l2 : Line) :
  are_parallel l1 l2 → are_corresponding a b l1 l2 → a = b := by sorry

end vertical_angles_equal_parallel_lines_corresponding_angles_equal_l3957_395767


namespace urn_probability_theorem_l3957_395759

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Blue

/-- Represents the state of the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents one draw operation -/
def draw (state : UrnState) : UrnState :=
  if state.red / (state.red + state.blue) > state.blue / (state.red + state.blue)
  then UrnState.mk (state.red + 1) state.blue
  else UrnState.mk state.red (state.blue + 1)

/-- Performs n draw operations -/
def performDraws (n : ℕ) (initial : UrnState) : UrnState :=
  match n with
  | 0 => initial
  | n + 1 => draw (performDraws n initial)

/-- Calculates the probability of selecting a red ball n times in a row -/
def redProbability (n : ℕ) (initial : UrnState) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => 
    let state := performDraws n initial
    (state.red : ℚ) / (state.red + state.blue) * redProbability n initial

/-- The main theorem to prove -/
theorem urn_probability_theorem :
  let initial := UrnState.mk 2 1
  let final := performDraws 5 initial
  final.red = 8 ∧ final.blue = 4 ∧ redProbability 5 initial = 2/7 :=
sorry

end urn_probability_theorem_l3957_395759


namespace rectangular_prism_diagonal_l3957_395772

theorem rectangular_prism_diagonal (a b c : ℝ) (ha : a = 12) (hb : b = 15) (hc : c = 8) :
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 433 := by
  sorry

end rectangular_prism_diagonal_l3957_395772


namespace train_length_is_415_l3957_395745

/-- Represents the problem of calculating a train's length -/
def TrainProblem (speed : ℝ) (tunnelLength : ℝ) (time : ℝ) : Prop :=
  let speedMPS := speed * 1000 / 3600
  let totalDistance := speedMPS * time
  totalDistance = tunnelLength + 415

/-- Theorem stating that given the conditions, the train length is 415 meters -/
theorem train_length_is_415 :
  TrainProblem 63 285 40 := by
  sorry

#check train_length_is_415

end train_length_is_415_l3957_395745


namespace sqrt_sqrt_81_l3957_395792

theorem sqrt_sqrt_81 : Real.sqrt (Real.sqrt 81) = 3 := by
  sorry

end sqrt_sqrt_81_l3957_395792


namespace fourth_power_congruence_divisibility_l3957_395734

theorem fourth_power_congruence_divisibility (p a b c d : ℕ) (hp : Prime p) 
  (ha : 0 < a) (hab : a < b) (hbc : b < c) (hcd : c < d) (hdp : d < p)
  (hcong : ∃ k : ℕ, a^4 % p = k ∧ b^4 % p = k ∧ c^4 % p = k ∧ d^4 % p = k) :
  (a + b + c + d) ∣ (a^2013 + b^2013 + c^2013 + d^2013) := by
  sorry

end fourth_power_congruence_divisibility_l3957_395734


namespace second_section_has_180_cars_l3957_395757

-- Define the given information
def section_g_rows : ℕ := 15
def section_g_cars_per_row : ℕ := 10
def second_section_rows : ℕ := 20
def nate_cars_per_minute : ℕ := 11
def nate_search_time : ℕ := 30

-- Define the total number of cars Nate walked past
def total_cars_walked : ℕ := nate_cars_per_minute * nate_search_time

-- Define the number of cars in Section G
def section_g_cars : ℕ := section_g_rows * section_g_cars_per_row

-- Define the number of cars in the second section
def second_section_cars : ℕ := total_cars_walked - section_g_cars

-- Theorem to prove
theorem second_section_has_180_cars :
  second_section_cars = 180 :=
sorry

end second_section_has_180_cars_l3957_395757


namespace general_solution_valid_particular_solution_valid_l3957_395730

-- Define the general solution function
def f (C : ℝ) (x : ℝ) : ℝ := x^2 + C

-- Define the particular solution function
def g (x : ℝ) : ℝ := x^2 + 1

-- Theorem for the general solution
theorem general_solution_valid (C : ℝ) : 
  ∀ x, HasDerivAt (f C) (2 * x) x :=
sorry

-- Theorem for the particular solution
theorem particular_solution_valid : 
  g 1 = 2 ∧ ∀ x, HasDerivAt g (2 * x) x :=
sorry

end general_solution_valid_particular_solution_valid_l3957_395730


namespace min_value_problem_l3957_395750

theorem min_value_problem (x y a : ℝ) (hx : x > 0) (hy : y > 0) (ha : a > 0)
  (h1 : 2 * x + y = 1)
  (h2 : ∀ x' y' : ℝ, x' > 0 → y' > 0 → 2 * x' + y' = 1 → a / x' + 1 / y' ≥ 9)
  (h3 : ∃ x' y' : ℝ, x' > 0 ∧ y' > 0 ∧ 2 * x' + y' = 1 ∧ a / x' + 1 / y' = 9) :
  a = 2 := by
sorry

end min_value_problem_l3957_395750


namespace curve_to_linear_equation_l3957_395729

/-- Given a curve parameterized by (x, y) = (3t + 6, 5t - 3), where t is a real number,
    prove that it can be expressed as the linear equation y = (5/3)x - 13. -/
theorem curve_to_linear_equation :
  ∀ (t x y : ℝ), x = 3 * t + 6 ∧ y = 5 * t - 3 →
  y = (5 / 3 : ℝ) * x - 13 :=
by
  sorry

end curve_to_linear_equation_l3957_395729


namespace time_conversions_and_difference_l3957_395788

/-- Converts 12-hour time (PM) to 24-hour format -/
def convert_pm_to_24h (hour : Nat) : Nat :=
  hour + 12

/-- Calculates the time difference in minutes between two times in 24-hour format -/
def time_diff_minutes (start_hour start_min end_hour end_min : Nat) : Nat :=
  (end_hour * 60 + end_min) - (start_hour * 60 + start_min)

theorem time_conversions_and_difference :
  (convert_pm_to_24h 5 = 17) ∧
  (convert_pm_to_24h 10 = 22) ∧
  (time_diff_minutes 16 40 17 20 = 40) :=
by sorry

end time_conversions_and_difference_l3957_395788


namespace compute_expression_l3957_395783

theorem compute_expression : 4 * 6 * 8 - 24 / 3 = 184 := by
  sorry

end compute_expression_l3957_395783


namespace area_FJGH_area_FJGH_proof_l3957_395796

/-- Represents a parallelogram EFGH with point J on side EH -/
structure Parallelogram where
  /-- Length of side EH -/
  eh : ℝ
  /-- Length of JH -/
  jh : ℝ
  /-- Height of the parallelogram from FG to EH -/
  height : ℝ
  /-- Condition that EH = 12 -/
  eh_eq : eh = 12
  /-- Condition that JH = 8 -/
  jh_eq : jh = 8
  /-- Condition that the height is 10 -/
  height_eq : height = 10

/-- The area of region FJGH in the parallelogram is 100 -/
theorem area_FJGH (p : Parallelogram) : ℝ :=
  100

#check area_FJGH

/-- Proof of the theorem -/
theorem area_FJGH_proof (p : Parallelogram) : area_FJGH p = 100 := by
  sorry

end area_FJGH_area_FJGH_proof_l3957_395796


namespace red_balls_count_l3957_395717

theorem red_balls_count (w r : ℕ) : 
  (w : ℚ) / r = 5 / 3 →  -- ratio of white to red balls
  w + 15 + r = 50 →     -- total after adding 15 white balls
  r = 12 := by           -- number of red balls
sorry

end red_balls_count_l3957_395717


namespace pastry_combinations_l3957_395787

/-- The number of ways to distribute n indistinguishable items into k distinguishable bins -/
def combinations_with_repetition (n k : ℕ) : ℕ := 
  Nat.choose (n + k - 1) k

/-- The number of pastry types available -/
def num_pastry_types : ℕ := 3

/-- The total number of pastries to be bought -/
def total_pastries : ℕ := 9

/-- Theorem stating that the number of ways to buy 9 pastries from 3 types is 55 -/
theorem pastry_combinations : 
  combinations_with_repetition total_pastries num_pastry_types = 55 := by
  sorry

end pastry_combinations_l3957_395787


namespace desk_purchase_optimization_l3957_395702

/-- The total cost function for shipping and storage fees -/
def f (x : ℕ) : ℚ := 144 / x + 4 * x

/-- The number of desks to be purchased -/
def total_desks : ℕ := 36

/-- The value of each desk -/
def desk_value : ℕ := 20

/-- The shipping fee per batch -/
def shipping_fee : ℕ := 4

/-- Available funds for shipping and storage -/
def available_funds : ℕ := 48

theorem desk_purchase_optimization :
  /- 1. The total cost function is correct -/
  (∀ x : ℕ, x > 0 → f x = 144 / x + 4 * x) ∧
  /- 2. There exists an integer x between 4 and 9 inclusive that satisfies the budget -/
  (∃ x : ℕ, 4 ≤ x ∧ x ≤ 9 ∧ f x ≤ available_funds) ∧
  /- 3. The minimum value of f(x) occurs when x = 6 -/
  (∀ x : ℕ, x > 0 → f 6 ≤ f x) :=
by sorry

end desk_purchase_optimization_l3957_395702


namespace complete_graph_edges_six_vertices_l3957_395719

theorem complete_graph_edges_six_vertices :
  let n : ℕ := 6
  let E : ℕ := n * (n - 1) / 2
  E = 15 := by sorry

end complete_graph_edges_six_vertices_l3957_395719


namespace monotonic_f_iff_a_range_l3957_395798

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + x^2 - a*x + 1

-- Define monotonically increasing
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem monotonic_f_iff_a_range :
  ∀ a : ℝ, (monotonically_increasing (f a)) ↔ a ≤ -1/3 :=
sorry

end monotonic_f_iff_a_range_l3957_395798


namespace cards_in_basketball_box_dexter_basketball_cards_l3957_395756

/-- The number of cards in each basketball card box -/
def cards_per_basketball_box (total_cards : ℕ) (basketball_boxes : ℕ) (football_boxes : ℕ) (cards_per_football_box : ℕ) : ℕ :=
  (total_cards - football_boxes * cards_per_football_box) / basketball_boxes

/-- Theorem stating the number of cards in each basketball card box -/
theorem cards_in_basketball_box :
  cards_per_basketball_box 255 9 6 20 = 15 := by
  sorry

/-- Main theorem proving the problem statement -/
theorem dexter_basketball_cards :
  ∃ (total_cards basketball_boxes football_boxes cards_per_football_box : ℕ),
    total_cards = 255 ∧
    basketball_boxes = 9 ∧
    football_boxes = basketball_boxes - 3 ∧
    cards_per_football_box = 20 ∧
    cards_per_basketball_box total_cards basketball_boxes football_boxes cards_per_football_box = 15 := by
  sorry

end cards_in_basketball_box_dexter_basketball_cards_l3957_395756


namespace usual_baking_time_l3957_395737

/-- Represents the time in hours for Matthew's cake-making process -/
structure BakingTime where
  assembly : ℝ
  baking : ℝ
  decorating : ℝ

/-- The total time for Matthew's cake-making process -/
def total_time (t : BakingTime) : ℝ := t.assembly + t.baking + t.decorating

/-- Represents the scenario when the oven fails -/
def oven_failure (normal : BakingTime) : BakingTime :=
  { assembly := normal.assembly,
    baking := 2 * normal.baking,
    decorating := normal.decorating }

theorem usual_baking_time :
  ∃ (normal : BakingTime),
    normal.assembly = 1 ∧
    normal.decorating = 1 ∧
    total_time (oven_failure normal) = 5 ∧
    normal.baking = 1.5 := by
  sorry

end usual_baking_time_l3957_395737


namespace series_convergence_l3957_395703

theorem series_convergence (a : ℕ → ℝ) :
  (∃ S : ℝ, HasSum (λ n : ℕ => a n + 2 * a (n + 1)) S) →
  (∃ T : ℝ, HasSum a T) :=
by sorry

end series_convergence_l3957_395703


namespace traffic_light_change_probability_l3957_395764

/-- Represents a traffic light cycle with durations for each color -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total duration of a traffic light cycle -/
def cycleDuration (c : TrafficLightCycle) : ℕ :=
  c.green + c.yellow + c.red

/-- Calculates the number of seconds where a color change can be observed in a 3-second interval -/
def changeObservationWindow (c : TrafficLightCycle) : ℕ :=
  3 * 3  -- 3 transitions, each with a 3-second window

/-- The probability of observing a color change in a random 3-second interval -/
def probabilityOfChange (c : TrafficLightCycle) : ℚ :=
  changeObservationWindow c / cycleDuration c

theorem traffic_light_change_probability :
  let c : TrafficLightCycle := ⟨50, 2, 40⟩
  probabilityOfChange c = 9 / 92 := by
  sorry


end traffic_light_change_probability_l3957_395764


namespace sequence_zero_at_201_l3957_395739

/-- Sequence defined by the given recurrence relation -/
def a : ℕ → ℚ
  | 0 => 134
  | 1 => 150
  | (k + 2) => a k - (k + 1) / a (k + 1)

/-- The smallest positive integer n for which a_n = 0 -/
def n : ℕ := 201

/-- Theorem stating that a_n = 0 and n is the smallest such positive integer -/
theorem sequence_zero_at_201 :
  a n = 0 ∧ ∀ m : ℕ, m > 0 ∧ m < n → a m ≠ 0 := by sorry

end sequence_zero_at_201_l3957_395739


namespace prob_B_given_A₁_pairwise_mutually_exclusive_l3957_395712

-- Define the number of balls in each can
def can_A_red : ℕ := 5
def can_A_white : ℕ := 2
def can_A_black : ℕ := 3
def can_B_red : ℕ := 4
def can_B_white : ℕ := 3
def can_B_black : ℕ := 3

-- Define the total number of balls in each can
def total_A : ℕ := can_A_red + can_A_white + can_A_black
def total_B : ℕ := can_B_red + can_B_white + can_B_black

-- Define the events
def A₁ : Set ℕ := {x | x ≤ can_A_red}
def A₂ : Set ℕ := {x | can_A_red < x ∧ x ≤ can_A_red + can_A_white}
def A₃ : Set ℕ := {x | can_A_red + can_A_white < x ∧ x ≤ total_A}
def B : Set ℕ := {x | x ≤ can_B_red + 1}

-- Define the probability measure
noncomputable def P : Set ℕ → ℝ := sorry

-- Theorem 1: P(B|A₁) = 5/11
theorem prob_B_given_A₁ : P (B ∩ A₁) / P A₁ = 5 / 11 := by sorry

-- Theorem 2: A₁, A₂, A₃ are pairwise mutually exclusive
theorem pairwise_mutually_exclusive : 
  (A₁ ∩ A₂ = ∅) ∧ (A₁ ∩ A₃ = ∅) ∧ (A₂ ∩ A₃ = ∅) := by sorry

end prob_B_given_A₁_pairwise_mutually_exclusive_l3957_395712


namespace completing_square_equivalence_l3957_395765

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 2*x = 9 ↔ (x - 1)^2 = 10 :=
by sorry

end completing_square_equivalence_l3957_395765


namespace no_threefold_decreasing_number_l3957_395762

theorem no_threefold_decreasing_number : ¬∃ (a b c : ℕ), 
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (0 ≤ b ∧ b ≤ 9) ∧ 
  (0 ≤ c ∧ c ≤ 9) ∧ 
  (100 * a + 10 * b + c = 3 * (100 * b + 10 * c + a)) := by
  sorry

end no_threefold_decreasing_number_l3957_395762


namespace vanessa_albums_l3957_395708

theorem vanessa_albums (phone_pics : ℕ) (camera_pics : ℕ) (pics_per_album : ℕ) : 
  phone_pics = 23 → 
  camera_pics = 7 → 
  pics_per_album = 6 → 
  (phone_pics + camera_pics) / pics_per_album = 5 := by
sorry

end vanessa_albums_l3957_395708


namespace fourth_root_equation_solution_l3957_395706

theorem fourth_root_equation_solution (x : ℝ) (h1 : x > 0) 
  (h2 : (1 - x^4)^(1/4) + (1 + x^4)^(1/4) = 1) : x^8 = 35/36 := by
  sorry

end fourth_root_equation_solution_l3957_395706


namespace marys_income_percentage_l3957_395785

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.7)
  (h2 : mary = tim * 1.6) :
  mary = juan * 1.12 := by
  sorry

end marys_income_percentage_l3957_395785


namespace parallel_unit_vector_l3957_395763

/-- Given a vector a = (12, 5), prove that its parallel unit vector is (12/13, 5/13) or (-12/13, -5/13) -/
theorem parallel_unit_vector (a : ℝ × ℝ) (h : a = (12, 5)) :
  ∃ u : ℝ × ℝ, (u.1 * u.1 + u.2 * u.2 = 1) ∧ 
  (∃ k : ℝ, u.1 = k * a.1 ∧ u.2 = k * a.2) ∧
  (u = (12/13, 5/13) ∨ u = (-12/13, -5/13)) :=
by sorry

end parallel_unit_vector_l3957_395763


namespace horner_method_example_l3957_395722

def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 3*x + 2

theorem horner_method_example : f (-2) = 320 := by
  sorry

end horner_method_example_l3957_395722


namespace quadratic_two_distinct_roots_l3957_395778

/-- A quadratic function f(x) = kx^2 - 4x - 2 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 4 * x - 2

/-- The discriminant of the quadratic function f(x) = kx^2 - 4x - 2 -/
def discriminant (k : ℝ) : ℝ := 16 + 8 * k

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f k x = 0 ∧ f k y = 0) ↔ k > -2 ∧ k ≠ 0 :=
sorry

end quadratic_two_distinct_roots_l3957_395778


namespace last_two_digits_product_l3957_395770

theorem last_two_digits_product (n : ℤ) : 
  (n % 100 ≥ 0) →
  (n % 8 = 0) → 
  ((n % 100) / 10 + n % 10 = 12) → 
  ((n % 100) / 10) * (n % 10) = 32 := by
sorry

end last_two_digits_product_l3957_395770


namespace product_sequence_sum_l3957_395752

theorem product_sequence_sum (a b : ℕ) (h1 : a / 3 = 16) (h2 : b = a - 1) : a + b = 95 := by
  sorry

end product_sequence_sum_l3957_395752


namespace last_year_ticket_cost_l3957_395771

/-- 
Proves that the ticket cost last year was $85, given that this year's cost 
is $102 and represents a 20% increase from last year.
-/
theorem last_year_ticket_cost : 
  ∀ (last_year_cost : ℝ), 
  (last_year_cost + 0.2 * last_year_cost = 102) → 
  last_year_cost = 85 := by
sorry

end last_year_ticket_cost_l3957_395771


namespace x_coordinate_difference_at_y_20_l3957_395774

/-- A line in a 2D coordinate system --/
structure Line where
  slope : ℚ
  y_intercept : ℚ

def Line.through_points (x1 y1 x2 y2 : ℚ) : Line where
  slope := (y2 - y1) / (x2 - x1)
  y_intercept := y1 - ((y2 - y1) / (x2 - x1)) * x1

def Line.x_at_y (l : Line) (y : ℚ) : ℚ :=
  (y - l.y_intercept) / l.slope

theorem x_coordinate_difference_at_y_20 :
  let l := Line.through_points 0 6 3 0
  let m := Line.through_points 0 3 8 0
  let x_l := l.x_at_y 20
  let x_m := m.x_at_y 20
  |x_l - x_m| = 115 / 3 := by
  sorry

end x_coordinate_difference_at_y_20_l3957_395774


namespace smallest_multiple_of_45_and_75_not_20_l3957_395741

theorem smallest_multiple_of_45_and_75_not_20 :
  ∃ n : ℕ, n > 0 ∧ 45 ∣ n ∧ 75 ∣ n ∧ ¬(20 ∣ n) ∧
  ∀ m : ℕ, m > 0 ∧ 45 ∣ m ∧ 75 ∣ m ∧ ¬(20 ∣ m) → n ≤ m :=
by sorry

end smallest_multiple_of_45_and_75_not_20_l3957_395741


namespace max_ratio_is_half_l3957_395716

/-- A hyperbola with equation x^2 - y^2 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p | p.1^2 - p.2^2 = 1}

/-- The right focus of the hyperbola -/
def RightFocus : ℝ × ℝ := sorry

/-- The right directrix of the hyperbola -/
def RightDirectrix : Set (ℝ × ℝ) := sorry

/-- The right branch of the hyperbola -/
def RightBranch : Set (ℝ × ℝ) := sorry

/-- The projection of a point onto the right directrix -/
def ProjectOntoDirectrix (p : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The distance between two points -/
def Distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The midpoint of two points -/
def Midpoint (p q : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Statement: The maximum value of |MN|/|AB| is 1/2 -/
theorem max_ratio_is_half :
  ∀ A B : ℝ × ℝ,
  A ∈ RightBranch →
  B ∈ RightBranch →
  Distance A RightFocus * Distance B RightFocus = 0 →  -- This represents AF ⟂ BF
  let M := Midpoint A B
  let N := ProjectOntoDirectrix M
  Distance M N / Distance A B ≤ 1/2 := by
sorry

end max_ratio_is_half_l3957_395716


namespace largest_solution_of_equation_l3957_395746

theorem largest_solution_of_equation (x : ℝ) : 
  (6 * (12 * x^2 + 12 * x + 11) = x * (12 * x - 44)) → x ≤ -1 :=
by sorry

end largest_solution_of_equation_l3957_395746


namespace expression_evaluation_l3957_395789

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  x^2 + y^2 - z^2 + 2*x*y + 3*z = 0 := by
  sorry

end expression_evaluation_l3957_395789


namespace video_game_points_sum_l3957_395795

theorem video_game_points_sum : 
  let paul_points : ℕ := 3103
  let cousin_points : ℕ := 2713
  paul_points + cousin_points = 5816 :=
by sorry

end video_game_points_sum_l3957_395795


namespace segment_ratio_l3957_395753

/-- Given a line segment GH with points E and F on it, where GE is 3 times EH and GF is 4 times FH,
    prove that EF is 1/20 of GH. -/
theorem segment_ratio (G E F H : Real) (GH EF : Real) : 
  E ∈ Set.Icc G H → 
  F ∈ Set.Icc G H → 
  G - E = 3 * (H - E) → 
  G - F = 4 * (H - F) → 
  GH = G - H → 
  EF = E - F → 
  EF = (1 / 20) * GH := by
sorry

end segment_ratio_l3957_395753


namespace same_side_of_line_l3957_395761

/-- Given a line x + y = a, if the origin (0, 0) and the point (1, 1) are on the same side of this line, then a < 0 or a > 2. -/
theorem same_side_of_line (a : ℝ) : 
  (0 + 0 - a) * (1 + 1 - a) > 0 → a < 0 ∨ a > 2 := by
  sorry

end same_side_of_line_l3957_395761


namespace triangle_relations_l3957_395715

/-- Given a triangle ABC with side lengths a, b, c, altitudes h_a, h_b, h_c, 
    inradius r, and exradii r_a, r_b, r_c, the following equations hold -/
theorem triangle_relations (a b c h_a h_b h_c r r_a r_b r_c : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (hr : r > 0) (hr_a : r_a > 0) (hr_b : r_b > 0) (hr_c : r_c > 0) 
    (hh_a : h_a > 0) (hh_b : h_b > 0) (hh_c : h_c > 0) :
  h_a + h_b + h_c = r * (a + b + c) * (1 / a + 1 / b + 1 / c) ∧
  1 / h_a + 1 / h_b + 1 / h_c = 1 / r ∧
  1 / r = 1 / r_a + 1 / r_b + 1 / r_c ∧
  (h_a + h_b + h_c) * (1 / h_a + 1 / h_b + 1 / h_c) = (a + b + c) * (1 / a + 1 / b + 1 / c) ∧
  (h_a + h_c) / r_a + (h_c + h_a) / r_b + (h_a + h_b) / r_c = 6 :=
by sorry

end triangle_relations_l3957_395715


namespace orange_weight_after_water_loss_orange_weight_problem_l3957_395713

/-- Calculates the new weight of oranges after water loss -/
theorem orange_weight_after_water_loss 
  (initial_weight : ℝ) 
  (initial_water_percentage : ℝ) 
  (evaporation_loss_percentage : ℝ) 
  (skin_loss_percentage : ℝ) : ℝ :=
  let initial_water_weight := initial_weight * initial_water_percentage
  let dry_weight := initial_weight - initial_water_weight
  let evaporation_loss := initial_water_weight * evaporation_loss_percentage
  let remaining_water_after_evaporation := initial_water_weight - evaporation_loss
  let skin_loss := remaining_water_after_evaporation * skin_loss_percentage
  let total_water_loss := evaporation_loss + skin_loss
  let new_water_weight := initial_water_weight - total_water_loss
  new_water_weight + dry_weight

/-- The new weight of oranges after water loss is approximately 4.67225 kg -/
theorem orange_weight_problem : 
  ∃ ε > 0, |orange_weight_after_water_loss 5 0.95 0.05 0.02 - 4.67225| < ε :=
sorry

end orange_weight_after_water_loss_orange_weight_problem_l3957_395713


namespace starting_lineup_combinations_l3957_395793

def team_size : ℕ := 12
def offensive_linemen : ℕ := 4
def positions : ℕ := 5

theorem starting_lineup_combinations :
  (offensive_linemen) *
  (team_size - 1) *
  (team_size - 2) *
  (team_size - 3) *
  (team_size - 4) = 31680 :=
by sorry

end starting_lineup_combinations_l3957_395793


namespace area_of_specific_rectangle_l3957_395790

/-- A rectangle with a diagonal divided into four equal segments -/
structure DividedRectangle where
  /-- The length of each segment of the diagonal -/
  segment_length : ℝ
  /-- The diagonal is divided into four equal segments -/
  diagonal_length : ℝ := 4 * segment_length
  /-- The parallel lines are perpendicular to the diagonal -/
  perpendicular_lines : Bool

/-- The area of a rectangle with a divided diagonal -/
def area (rect : DividedRectangle) : ℝ :=
  sorry

/-- Theorem: The area of the specific rectangle is 16√3 -/
theorem area_of_specific_rectangle :
  let rect : DividedRectangle := {
    segment_length := 2,
    perpendicular_lines := true
  }
  area rect = 16 * Real.sqrt 3 :=
by
  sorry

end area_of_specific_rectangle_l3957_395790
