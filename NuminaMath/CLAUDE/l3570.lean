import Mathlib

namespace max_value_complex_expression_l3570_357080

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 2) :
  Complex.abs ((z - 2) * (z + 2)^2) ≤ 16 * Real.sqrt 2 := by
  sorry

end max_value_complex_expression_l3570_357080


namespace tangent_length_is_six_l3570_357006

/-- Circle C with equation x^2 + y^2 - 4x - 2y + 1 = 0 -/
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- Line l with equation x + my - 1 = 0 -/
def line_l (m x y : ℝ) : Prop :=
  x + m*y - 1 = 0

/-- Point A with coordinates (-4, m) -/
def point_A (m : ℝ) : ℝ × ℝ :=
  (-4, m)

/-- Theorem stating that the length of the tangent from A to C is 6 -/
theorem tangent_length_is_six (m : ℝ) : 
  line_l m (2 : ℝ) 1 →  -- line l passes through (2, 1)
  ∃ (B : ℝ × ℝ), 
    circle_C B.1 B.2 ∧  -- B is on circle C
    (∀ (x y : ℝ), circle_C x y → ((x - (-4))^2 + (y - m)^2 ≥ (B.1 - (-4))^2 + (B.2 - m)^2)) ∧  -- AB is tangent
    ((B.1 - (-4))^2 + (B.2 - m)^2 = 36) :=  -- |AB|^2 = 6^2
  sorry

end tangent_length_is_six_l3570_357006


namespace line_intersection_problem_l3570_357012

/-- The problem statement as a theorem -/
theorem line_intersection_problem :
  ∃ (m b : ℝ),
    b ≠ 0 ∧
    (∃! k, ∃ y₁ y₂, 
      y₁ = k^2 + 4*k + 4 ∧
      y₂ = m*k + b ∧
      |y₁ - y₂| = 6) ∧
    (8 = m*2 + b) ∧
    m = 2 * Real.sqrt 6 ∧
    b = 8 - 4 * Real.sqrt 6 :=
by sorry

end line_intersection_problem_l3570_357012


namespace min_value_a_plus_b_l3570_357019

theorem min_value_a_plus_b (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a * b^2 = 4) :
  a + b ≥ 3 ∧ ∃ (a₀ b₀ : ℝ), a₀ ≠ 0 ∧ b₀ ≠ 0 ∧ a₀ * b₀^2 = 4 ∧ a₀ + b₀ = 3 :=
sorry

end min_value_a_plus_b_l3570_357019


namespace complex_equation_implies_difference_l3570_357093

theorem complex_equation_implies_difference (x y : ℝ) :
  (x * Complex.I + 2 = y - Complex.I) → (x - y = -3) := by sorry

end complex_equation_implies_difference_l3570_357093


namespace batsman_average_increase_l3570_357096

theorem batsman_average_increase 
  (total_innings : ℕ) 
  (last_innings_score : ℕ) 
  (new_average : ℚ) :
  total_innings = 17 →
  last_innings_score = 85 →
  new_average = 37 →
  (total_innings * new_average - last_innings_score) / (total_innings - 1) + 3 = new_average :=
by sorry

end batsman_average_increase_l3570_357096


namespace rhombus_diagonals_not_always_equal_l3570_357082

-- Define a rhombus
structure Rhombus :=
  (side_length : ℝ)
  (diagonal1 : ℝ)
  (diagonal2 : ℝ)
  (side_length_positive : side_length > 0)
  (diagonals_positive : diagonal1 > 0 ∧ diagonal2 > 0)

-- State the theorem
theorem rhombus_diagonals_not_always_equal :
  ∃ (r : Rhombus), r.diagonal1 ≠ r.diagonal2 :=
sorry

end rhombus_diagonals_not_always_equal_l3570_357082


namespace unique_solution_natural_system_l3570_357024

theorem unique_solution_natural_system :
  ∃! (a b c d : ℕ), a * b = c + d ∧ c * d = a + b :=
by
  -- The unique solution is (2, 2, 2, 2)
  -- Proof goes here
  sorry

end unique_solution_natural_system_l3570_357024


namespace intersection_line_canonical_equations_l3570_357070

/-- Given two planes in 3D space, this theorem states that their line of intersection
    can be represented by specific canonical equations. -/
theorem intersection_line_canonical_equations
  (plane1 : x + y + z = 2)
  (plane2 : x - y - 2*z = -2)
  : ∃ (t : ℝ), x = -t ∧ y = 3*t + 2 ∧ z = -2*t :=
sorry

end intersection_line_canonical_equations_l3570_357070


namespace custom_mul_equality_l3570_357088

/-- Custom multiplication operation for real numbers -/
def custom_mul (x y : ℝ) : ℝ := (x - y)^2

/-- Theorem stating the equality for the given expression using custom multiplication -/
theorem custom_mul_equality (x y z : ℝ) : 
  custom_mul (x - y) (y - z) = (x - 2*y + z)^2 := by
  sorry

end custom_mul_equality_l3570_357088


namespace even_function_theorem_l3570_357014

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The functional equation satisfied by f -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y - 2 * x * y - 1

theorem even_function_theorem (f : ℝ → ℝ) 
    (heven : EvenFunction f) 
    (heq : SatisfiesFunctionalEquation f) : 
    ∀ x, f x = -x^2 + 1 := by
  sorry

end even_function_theorem_l3570_357014


namespace jogger_train_distance_l3570_357095

/-- Proves the distance a jogger is ahead of a train engine given specific conditions -/
theorem jogger_train_distance (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  passing_time = 35 →
  (train_speed - jogger_speed) * passing_time = train_length + 230 := by
  sorry

#check jogger_train_distance

end jogger_train_distance_l3570_357095


namespace markup_discount_profit_l3570_357000

/-- Given a markup percentage and a discount percentage, calculate the profit percentage -/
def profit_percentage (markup : ℝ) (discount : ℝ) : ℝ :=
  let marked_price := 1 + markup
  let selling_price := marked_price * (1 - discount)
  (selling_price - 1) * 100

/-- Theorem stating that a 75% markup followed by a 30% discount results in a 22.5% profit -/
theorem markup_discount_profit : profit_percentage 0.75 0.3 = 22.5 := by
  sorry

end markup_discount_profit_l3570_357000


namespace g_of_f_minus_x_l3570_357078

theorem g_of_f_minus_x (x : ℝ) (hx : x^2 ≠ 1) :
  let f (x : ℝ) := (x^2 + 2*x + 1) / (x^2 - 2*x + 1)
  let g (x : ℝ) := x^2
  g (f (-x)) = (x^2 - 2*x + 1)^2 / (x^2 + 2*x + 1)^2 := by
  sorry

end g_of_f_minus_x_l3570_357078


namespace polynomial_factorization_l3570_357027

theorem polynomial_factorization (x : ℝ) : 
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 12) * (x^2 + 6*x + 3) := by
  sorry

end polynomial_factorization_l3570_357027


namespace students_not_in_biology_or_chemistry_l3570_357035

theorem students_not_in_biology_or_chemistry
  (total : ℕ)
  (biology_percent : ℚ)
  (chemistry_percent : ℚ)
  (both_percent : ℚ)
  (h_total : total = 880)
  (h_biology : biology_percent = 40 / 100)
  (h_chemistry : chemistry_percent = 30 / 100)
  (h_both : both_percent = 10 / 100) :
  total - (total * biology_percent + total * chemistry_percent - total * both_percent).floor = 352 :=
by sorry

end students_not_in_biology_or_chemistry_l3570_357035


namespace parabola_ratio_l3570_357055

/-- A parabola passing through points (-1, 1) and (3, 1) has a/b = -2 --/
theorem parabola_ratio (a b c : ℝ) : 
  (a * (-1)^2 + b * (-1) + c = 1) → 
  (a * 3^2 + b * 3 + c = 1) → 
  a / b = -2 := by
sorry

end parabola_ratio_l3570_357055


namespace quadratic_root_zero_l3570_357020

theorem quadratic_root_zero (a : ℝ) : 
  (∃ x, (a - 1) * x^2 + x + a^2 - 1 = 0) ∧ 
  ((a - 1) * 0^2 + 0 + a^2 - 1 = 0) →
  a = -1 := by
sorry

end quadratic_root_zero_l3570_357020


namespace prob_one_defective_out_of_two_l3570_357084

/-- The probability of selecting exactly one defective product when randomly choosing 2 out of 5 products, where 2 are defective and 3 are qualified. -/
theorem prob_one_defective_out_of_two (total : Nat) (defective : Nat) (selected : Nat) : 
  total = 5 → defective = 2 → selected = 2 → 
  (Nat.choose defective 1 * Nat.choose (total - defective) (selected - 1)) / Nat.choose total selected = 3/5 := by
sorry

end prob_one_defective_out_of_two_l3570_357084


namespace min_value_d_l3570_357018

/-- Given positive integers a, b, c, and d where a < b < c < d, and a system of equations
    with exactly one solution, the minimum value of d is 602. -/
theorem min_value_d (a b c d : ℕ+) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : ∃! (x y : ℝ), 3 * x + y = 3004 ∧ y = |x - a| + |x - b| + |x - c| + |x - d|) :
  d = 602 := by
  sorry

end min_value_d_l3570_357018


namespace not_prime_5n_plus_3_l3570_357043

theorem not_prime_5n_plus_3 (n : ℕ) (k m : ℤ) 
  (h1 : 2 * n + 1 = k^2) 
  (h2 : 3 * n + 1 = m^2) : 
  ¬ Nat.Prime (5 * n + 3) := by
sorry

end not_prime_5n_plus_3_l3570_357043


namespace cuboid_edge_length_l3570_357031

/-- Theorem: Given a cuboid with edges x cm, 5 cm, and 6 cm, and a volume of 180 cm³,
    the length of the first edge (x) is 6 cm. -/
theorem cuboid_edge_length (x : ℝ) : x * 5 * 6 = 180 → x = 6 := by
  sorry

end cuboid_edge_length_l3570_357031


namespace area_of_EFGH_l3570_357047

/-- The length of the shorter side of each smaller rectangle -/
def short_side : ℝ := 3

/-- The number of smaller rectangles used to form EFGH -/
def num_rectangles : ℕ := 4

/-- The number of rectangles placed horizontally -/
def horizontal_rectangles : ℕ := 2

/-- The number of rectangles placed vertically -/
def vertical_rectangles : ℕ := 2

/-- The ratio of the longer side to the shorter side of each smaller rectangle -/
def side_ratio : ℝ := 2

theorem area_of_EFGH : 
  let longer_side := short_side * side_ratio
  let width := short_side * horizontal_rectangles
  let length := longer_side * vertical_rectangles
  width * length = 72 := by sorry

end area_of_EFGH_l3570_357047


namespace tangent_sum_and_double_sum_l3570_357063

theorem tangent_sum_and_double_sum (α β : Real) 
  (h1 : Real.tan α = 1/7) (h2 : Real.tan β = 1/3) : 
  Real.tan (α + β) = 1/2 ∧ Real.tan (α + 2*β) = 1 := by sorry

end tangent_sum_and_double_sum_l3570_357063


namespace intersection_of_A_and_B_l3570_357008

def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}
def B : Set ℝ := {y | y^2 - 2*y - 3 ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x ≤ 3} := by sorry

end intersection_of_A_and_B_l3570_357008


namespace yogurt_combinations_l3570_357009

theorem yogurt_combinations (flavors : ℕ) (toppings : ℕ) :
  flavors = 6 → toppings = 8 →
  flavors * (toppings.choose 3) = 336 := by
  sorry

end yogurt_combinations_l3570_357009


namespace equal_area_line_equation_l3570_357003

-- Define the circle arrangement
def circle_arrangement : List (ℝ × ℝ) :=
  [(1, 1), (3, 1), (5, 1), (7, 1), (1, 3), (3, 3), (5, 3), (1, 5), (3, 5), (5, 5)]

-- Define the line with slope 2
def line_slope : ℝ := 2

-- Define the function to check if a line divides the area equally
def divides_area_equally (a b c : ℤ) : Prop := sorry

-- Define the function to check if three integers are coprime
def are_coprime (a b c : ℤ) : Prop := sorry

-- Main theorem
theorem equal_area_line_equation :
  ∃ (a b c : ℤ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    are_coprime a b c ∧
    divides_area_equally a b c ∧
    a^2 + b^2 + c^2 = 86 :=
  sorry

end equal_area_line_equation_l3570_357003


namespace cubic_identity_fraction_l3570_357058

theorem cubic_identity_fraction (x y z : ℝ) :
  ((x - y)^3 + (y - z)^3 + (z - x)^3) / (15 * (x - y) * (y - z) * (z - x)) = 1/5 := by
  sorry

end cubic_identity_fraction_l3570_357058


namespace max_dominoes_formula_l3570_357066

/-- Represents a grid of size 2n × 2n -/
structure Grid (n : ℕ+) where
  size : ℕ := 2 * n

/-- Represents a domino placement on the grid -/
structure DominoPlacement (n : ℕ+) where
  grid : Grid n
  num_dominoes : ℕ
  valid : Prop  -- This represents the validity of the placement according to the rules

/-- The maximum number of dominoes that can be placed on a 2n × 2n grid -/
def max_dominoes (n : ℕ+) : ℕ := n * (n + 1) / 2

/-- Theorem stating that the maximum number of dominoes is n(n+1)/2 -/
theorem max_dominoes_formula (n : ℕ+) :
  ∀ (p : DominoPlacement n), p.valid → p.num_dominoes ≤ max_dominoes n :=
sorry

end max_dominoes_formula_l3570_357066


namespace investment_percentage_l3570_357056

/-- The investment problem with Vishal, Trishul, and Raghu -/
theorem investment_percentage (vishal trishul raghu : ℝ) : 
  vishal = 1.1 * trishul →                  -- Vishal invested 10% more than Trishul
  raghu = 2200 →                            -- Raghu invested Rs. 2200
  vishal + trishul + raghu = 6358 →         -- Total sum of investments
  trishul < raghu →                         -- Trishul invested less than Raghu
  (raghu - trishul) / raghu * 100 = 10 :=   -- Percentage Trishul invested less than Raghu
by sorry

end investment_percentage_l3570_357056


namespace quadratic_polynomial_unique_l3570_357028

theorem quadratic_polynomial_unique (q : ℝ → ℝ) : 
  (∀ x, q x = 2 * x^2 - 6 * x - 36) →
  q (-3) = 0 ∧ q 6 = 0 ∧ q 2 = -40 := by
  sorry

end quadratic_polynomial_unique_l3570_357028


namespace dogs_in_park_l3570_357068

/-- The number of dogs in the park -/
def D : ℕ := 88

/-- The number of dogs running -/
def running : ℕ := 12

/-- The number of dogs doing nothing -/
def doing_nothing : ℕ := 10

theorem dogs_in_park :
  D = running + D / 2 + D / 4 + doing_nothing :=
sorry


end dogs_in_park_l3570_357068


namespace cube_root_equation_solution_l3570_357048

theorem cube_root_equation_solution :
  ∃! x : ℝ, (2 - x / 2) ^ (1/3 : ℝ) = -3 :=
by
  -- Proof goes here
  sorry

end cube_root_equation_solution_l3570_357048


namespace polynomial_as_sum_of_squares_l3570_357016

theorem polynomial_as_sum_of_squares (x : ℝ) :
  x^4 - 2*x^3 + 6*x^2 - 2*x + 1 = (x^2 - x)^2 + (x - 1)^2 + (2*x)^2 := by
  sorry

end polynomial_as_sum_of_squares_l3570_357016


namespace min_bullseyes_theorem_l3570_357067

/-- Represents the archery tournament scenario -/
structure ArcheryTournament where
  total_shots : Nat
  halfway_shots : Nat
  chelsea_lead : Nat
  chelsea_min_score : Nat
  opponent_min_score : Nat

/-- Calculates the minimum number of bullseyes needed for Chelsea to guarantee victory -/
def min_bullseyes_for_victory (tournament : ArcheryTournament) : Nat :=
  let remaining_shots := tournament.total_shots - tournament.halfway_shots
  let chelsea_max_score := remaining_shots * 10
  let opponent_max_score := remaining_shots * 10 - tournament.chelsea_lead
  let chelsea_guaranteed_points := remaining_shots * tournament.chelsea_min_score
  let n := (opponent_max_score - chelsea_guaranteed_points + 10 - 1) / (10 - tournament.chelsea_min_score)
  n + 1

/-- The theorem states that for the given tournament conditions, 
    the minimum number of bullseyes needed for Chelsea to guarantee victory is 87 -/
theorem min_bullseyes_theorem (tournament : ArcheryTournament) 
  (h1 : tournament.total_shots = 200)
  (h2 : tournament.halfway_shots = 100)
  (h3 : tournament.chelsea_lead = 70)
  (h4 : tournament.chelsea_min_score = 5)
  (h5 : tournament.opponent_min_score = 3) :
  min_bullseyes_for_victory tournament = 87 := by
  sorry

end min_bullseyes_theorem_l3570_357067


namespace correlation_properties_l3570_357023

/-- The linear correlation coefficient between two variables -/
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

/-- The strength of linear correlation between two variables -/
def correlation_strength (r : ℝ) : ℝ := sorry

theorem correlation_properties (x y : ℝ → ℝ) (r : ℝ) 
  (h : r = correlation_coefficient x y) :
  (r > 0 → ∀ t₁ t₂, t₁ < t₂ → x t₁ < x t₂ → y t₁ < y t₂) ∧ 
  (∀ ε > 0, ∃ δ > 0, |r| > 1 - δ → correlation_strength r > 1 - ε) ∧
  (r = 1 ∨ r = -1 → ∃ a b : ℝ, ∀ t, y t = a * x t + b) :=
sorry

end correlation_properties_l3570_357023


namespace at_op_difference_zero_l3570_357013

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - (x + y)

-- State the theorem
theorem at_op_difference_zero : at_op 7 4 - at_op 4 7 = 0 := by
  sorry

end at_op_difference_zero_l3570_357013


namespace total_exercise_hours_l3570_357049

/-- Exercise duration in minutes for each person -/
def natasha_minutes : ℕ := 30 * 7
def esteban_minutes : ℕ := 10 * 9
def charlotte_minutes : ℕ := 20 + 45 + 70 + 100

/-- Total exercise duration in minutes -/
def total_minutes : ℕ := natasha_minutes + esteban_minutes + charlotte_minutes

/-- Conversion factor from minutes to hours -/
def minutes_per_hour : ℕ := 60

/-- Total exercise duration in hours -/
def total_hours : ℚ := total_minutes / minutes_per_hour

/-- Theorem: The total hours of exercise for all three individuals is 8.92 hours -/
theorem total_exercise_hours : total_hours = 892 / 100 := by
  sorry

end total_exercise_hours_l3570_357049


namespace gcd_11121_12012_l3570_357086

theorem gcd_11121_12012 : Nat.gcd 11121 12012 = 1 := by
  sorry

end gcd_11121_12012_l3570_357086


namespace total_time_outside_class_l3570_357050

def first_recess : ℕ := 15
def second_recess : ℕ := 15
def lunch : ℕ := 30
def third_recess : ℕ := 20

theorem total_time_outside_class :
  first_recess + second_recess + lunch + third_recess = 80 := by
  sorry

end total_time_outside_class_l3570_357050


namespace isosceles_triangle_base_angle_l3570_357030

theorem isosceles_triangle_base_angle (base_angle : ℝ) (top_angle : ℝ) : 
  -- The triangle is isosceles
  -- The top angle is 20° more than twice the base angle
  top_angle = 2 * base_angle + 20 →
  -- The sum of angles in a triangle is 180°
  base_angle + base_angle + top_angle = 180 →
  -- The base angle is 40°
  base_angle = 40 := by
sorry

end isosceles_triangle_base_angle_l3570_357030


namespace unique_integer_l3570_357037

theorem unique_integer (x : ℤ) 
  (h1 : 1 < x ∧ x < 9)
  (h2 : 2 < x ∧ x < 15)
  (h3 : -1 < x ∧ x < 7)
  (h4 : 0 < x ∧ x < 4)
  (h5 : x + 1 < 5) : 
  x = 3 := by sorry

end unique_integer_l3570_357037


namespace tangent_circles_exist_l3570_357017

-- Define the circle k
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define a ray
def Ray (origin : ℝ × ℝ) (direction : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ p = (origin.1 + t * direction.1, origin.2 + t * direction.2)}

-- Define tangency between a circle and a ray
def IsTangent (c : Set (ℝ × ℝ)) (r : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ c ∧ p ∈ r ∧ ∀ q : ℝ × ℝ, q ∈ c ∩ r → q = p

-- Main theorem
theorem tangent_circles_exist
  (k : Set (ℝ × ℝ))
  (O : ℝ × ℝ)
  (r : ℝ)
  (A : ℝ × ℝ)
  (e f : Set (ℝ × ℝ))
  (hk : k = Circle O r)
  (hA : A ∈ k)
  (he : e = Ray A (1, 0))  -- Arbitrary direction for e
  (hf : f = Ray A (0, 1))  -- Arbitrary direction for f
  (hef : e ≠ f) :
  ∃ c : Set (ℝ × ℝ), ∃ center : ℝ × ℝ, ∃ radius : ℝ,
    c = Circle center radius ∧
    IsTangent c k ∧
    IsTangent c e ∧
    IsTangent c f :=
sorry

end tangent_circles_exist_l3570_357017


namespace abs_neg_two_fifths_l3570_357004

theorem abs_neg_two_fifths : |(-2 : ℚ) / 5| = 2 / 5 := by
  sorry

end abs_neg_two_fifths_l3570_357004


namespace num_al_sandwiches_l3570_357069

/-- Represents the number of different types of bread available at the deli. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available at the deli. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available at the deli. -/
def num_cheeses : ℕ := 5

/-- Represents whether ham is available at the deli. -/
def ham_available : Prop := True

/-- Represents whether turkey is available at the deli. -/
def turkey_available : Prop := True

/-- Represents whether cheddar cheese is available at the deli. -/
def cheddar_available : Prop := True

/-- Represents whether rye bread is available at the deli. -/
def rye_available : Prop := True

/-- Represents the number of sandwiches with ham and cheddar cheese combination. -/
def ham_cheddar_combos : ℕ := num_breads

/-- Represents the number of sandwiches with rye bread and turkey combination. -/
def rye_turkey_combos : ℕ := num_cheeses

/-- Theorem stating the number of different sandwiches Al could order. -/
theorem num_al_sandwiches : 
  num_breads * num_meats * num_cheeses - ham_cheddar_combos - rye_turkey_combos = 165 := by
  sorry

end num_al_sandwiches_l3570_357069


namespace quadratic_inequality_solution_set_l3570_357042

/-- Given a quadratic inequality ax² + bx + c < 0 with solution set {x | x < -2 ∨ x > -1/2},
    prove that the solution set for ax² - bx + c > 0 is {x | 1/2 < x ∧ x < 2} -/
theorem quadratic_inequality_solution_set
  (a b c : ℝ)
  (h : ∀ x : ℝ, (a * x^2 + b * x + c < 0) ↔ (x < -2 ∨ x > -(1/2))) :
  ∀ x : ℝ, (a * x^2 - b * x + c > 0) ↔ (1/2 < x ∧ x < 2) :=
sorry

end quadratic_inequality_solution_set_l3570_357042


namespace smallest_prime_dividing_sum_l3570_357053

theorem smallest_prime_dividing_sum : ∃ p : Nat, 
  Prime p ∧ 
  p ∣ (4^15 + 7^12) ∧ 
  ∀ q : Nat, Prime q → q ∣ (4^15 + 7^12) → p ≤ q :=
by sorry

end smallest_prime_dividing_sum_l3570_357053


namespace factors_of_81_l3570_357076

theorem factors_of_81 : Finset.card (Nat.divisors 81) = 5 := by
  sorry

end factors_of_81_l3570_357076


namespace magical_red_knights_fraction_l3570_357052

theorem magical_red_knights_fraction 
  (total : ℕ) 
  (red : ℕ) 
  (blue : ℕ) 
  (magical : ℕ) 
  (magical_red : ℕ) 
  (magical_blue : ℕ) 
  (h1 : red = (3 * total) / 8)
  (h2 : blue = total - red)
  (h3 : magical = total / 8)
  (h4 : magical_red * blue = 3 * magical_blue * red)
  (h5 : magical = magical_red + magical_blue) :
  magical_red * 14 = red * 3 := by
  sorry

end magical_red_knights_fraction_l3570_357052


namespace equilateral_triangle_perimeter_equilateral_triangle_perimeter_proof_l3570_357085

theorem equilateral_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
fun equilateral_side isosceles_perimeter isosceles_base =>
  let isosceles_side := equilateral_side
  let equilateral_perimeter := 3 * equilateral_side
  isosceles_perimeter = 2 * isosceles_side + isosceles_base ∧
  isosceles_perimeter = 40 ∧
  isosceles_base = 10 →
  equilateral_perimeter = 45

-- The proof would go here, but we'll skip it as requested
theorem equilateral_triangle_perimeter_proof :
  equilateral_triangle_perimeter 15 40 10 :=
sorry

end equilateral_triangle_perimeter_equilateral_triangle_perimeter_proof_l3570_357085


namespace equation_solution_l3570_357098

theorem equation_solution : ∃! x : ℝ, (1 + x) / 4 - (x - 2) / 8 = 1 := by
  use 4
  constructor
  · -- Prove that x = 4 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check equation_solution

end equation_solution_l3570_357098


namespace ball_probability_l3570_357029

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h_total : total = 60)
  (h_white : white = 22)
  (h_green : green = 18)
  (h_yellow : yellow = 17)
  (h_red : red = 3)
  (h_purple : purple = 1)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 14 / 15 := by
sorry

end ball_probability_l3570_357029


namespace f_root_exists_l3570_357001

noncomputable def f (x : ℝ) := Real.log x / Real.log 3 + x

theorem f_root_exists : ∃ x ∈ Set.Ioo 3 4, f x - 5 = 0 := by
  sorry

end f_root_exists_l3570_357001


namespace parallel_case_perpendicular_case_l3570_357083

-- Define points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 8)

-- Define vector CD
def CD (x : ℝ) : ℝ × ℝ := (x, 3)

-- Define vector AB
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Theorem for parallel case
theorem parallel_case : 
  (∃ k : ℝ, AB = k • CD 1) → 1 = 1 := by sorry

-- Theorem for perpendicular case
theorem perpendicular_case :
  (AB.1 * (CD (-9)).1 + AB.2 * (CD (-9)).2 = 0) → -9 = -9 := by sorry

end parallel_case_perpendicular_case_l3570_357083


namespace imaginary_product_implies_zero_l3570_357062

/-- If the product of (1-ai) and i is a pure imaginary number, then a = 0 -/
theorem imaginary_product_implies_zero (a : ℝ) : 
  (∃ b : ℝ, (1 - a * Complex.I) * Complex.I = b * Complex.I) → a = 0 :=
by sorry

end imaginary_product_implies_zero_l3570_357062


namespace toothpick_100th_stage_l3570_357073

/-- Arithmetic sequence with first term 4 and common difference 4 -/
def toothpick_sequence (n : ℕ) : ℕ := 4 + (n - 1) * 4

/-- The 100th term of the toothpick sequence is 400 -/
theorem toothpick_100th_stage : toothpick_sequence 100 = 400 := by
  sorry

end toothpick_100th_stage_l3570_357073


namespace firefighter_ratio_l3570_357032

theorem firefighter_ratio (doug kai eli : ℕ) : 
  doug = 20 →
  eli = kai / 2 →
  doug + kai + eli = 110 →
  kai / doug = 3 := by
sorry

end firefighter_ratio_l3570_357032


namespace probability_two_copresidents_selected_l3570_357034

def choose (n k : ℕ) : ℕ := Nat.choose n k

def prob_copresident_selected (n : ℕ) : ℚ :=
  (choose (n - 2) 2 : ℚ) / (choose n 4 : ℚ)

def total_probability : ℚ :=
  (1 : ℚ) / 3 * (prob_copresident_selected 6 + prob_copresident_selected 8 + prob_copresident_selected 9)

theorem probability_two_copresidents_selected : total_probability = 82 / 315 := by
  sorry

end probability_two_copresidents_selected_l3570_357034


namespace trig_identity_l3570_357090

theorem trig_identity (α β : Real) 
  (h : (Real.sin β)^4 / (Real.sin α)^2 + (Real.cos β)^4 / (Real.cos α)^2 = 1) :
  ∃ x, (Real.cos α)^4 / (Real.cos β)^2 + (Real.sin α)^4 / (Real.sin β)^2 = x ∧ x = 1 :=
by sorry

end trig_identity_l3570_357090


namespace sin_arctan_equation_l3570_357038

theorem sin_arctan_equation (y : ℝ) (hy : y > 0) 
  (h : Real.sin (Real.arctan y) = 1 / (2 * y)) : 
  y^2 = (1 + Real.sqrt 17) / 8 := by
  sorry

end sin_arctan_equation_l3570_357038


namespace unique_two_digit_sum_product_l3570_357051

theorem unique_two_digit_sum_product : ∃! (a b : ℕ), 
  1 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  10 * a + b = a + 2 * b + a * b :=
by sorry

end unique_two_digit_sum_product_l3570_357051


namespace supplemental_tanks_needed_l3570_357044

def total_diving_time : ℕ := 8
def primary_tank_duration : ℕ := 2
def supplemental_tank_duration : ℕ := 1

theorem supplemental_tanks_needed :
  (total_diving_time - primary_tank_duration) / supplemental_tank_duration = 6 :=
by sorry

end supplemental_tanks_needed_l3570_357044


namespace bus_stop_interval_l3570_357094

/-- Proves that the time interval between bus stops is 6 minutes -/
theorem bus_stop_interval (average_speed : ℝ) (total_distance : ℝ) (num_stops : ℕ) 
  (h1 : average_speed = 60)
  (h2 : total_distance = 30)
  (h3 : num_stops = 6) :
  (total_distance / average_speed) * 60 / (num_stops - 1) = 6 := by
  sorry

end bus_stop_interval_l3570_357094


namespace unique_remainder_sum_equal_l3570_357015

/-- The sum of distinct remainders when dividing a natural number by all smaller positive natural numbers -/
def sumDistinctRemainders (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k => n % (k + 1))

/-- Theorem stating that 3 is the only natural number equal to the sum of its distinct remainders -/
theorem unique_remainder_sum_equal : ∀ n : ℕ, n > 0 → (sumDistinctRemainders n = n ↔ n = 3) := by
  sorry

end unique_remainder_sum_equal_l3570_357015


namespace problem_statement_l3570_357061

theorem problem_statement (x : ℕ) (h : x = 3) : x + x * (Nat.factorial x)^x = 651 := by
  sorry

end problem_statement_l3570_357061


namespace solution_set_eq_four_points_l3570_357040

/-- The set of solutions to the system of equations:
    a^4 - b^4 = c
    b^4 - c^4 = a
    c^4 - a^4 = b
    where a, b, c are real numbers. -/
def SolutionSet : Set (ℝ × ℝ × ℝ) :=
  {abc | let (a, b, c) := abc
         a^4 - b^4 = c ∧
         b^4 - c^4 = a ∧
         c^4 - a^4 = b}

/-- The theorem stating that the solution set is equal to the given set of four points. -/
theorem solution_set_eq_four_points :
  SolutionSet = {(0, 0, 0), (0, 1, -1), (-1, 0, 1), (1, -1, 0)} := by
  sorry

end solution_set_eq_four_points_l3570_357040


namespace clubsuit_calculation_l3570_357064

-- Define the new operation
def clubsuit (x y : ℤ) : ℤ := x^2 - y^2

-- Theorem statement
theorem clubsuit_calculation : clubsuit 5 (clubsuit 6 7) = -144 := by
  sorry

end clubsuit_calculation_l3570_357064


namespace polygon_diagonals_l3570_357054

theorem polygon_diagonals (n : ℕ) (interior_angle : ℝ) : 
  interior_angle = 150 → (n - 2) * 180 = n * interior_angle → n - 3 = 9 := by
  sorry

end polygon_diagonals_l3570_357054


namespace line_segment_length_l3570_357060

structure Line where
  points : Fin 5 → ℝ
  consecutive : ∀ i : Fin 4, points i < points (Fin.succ i)

def Line.segment (l : Line) (i j : Fin 5) : ℝ :=
  |l.points j - l.points i|

theorem line_segment_length (l : Line) 
  (h1 : l.segment 1 2 = 3 * l.segment 2 3)
  (h2 : l.segment 3 4 = 7)
  (h3 : l.segment 0 1 = 5)
  (h4 : l.segment 0 2 = 11) :
  l.segment 0 4 = 20 := by
  sorry

end line_segment_length_l3570_357060


namespace hall_paving_l3570_357074

/-- The number of stones required to pave a rectangular hall -/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℚ :=
  (hall_length * hall_width) / (stone_length * stone_width)

/-- Theorem: 1800 stones are required to pave a 36m x 15m hall with 6dm x 5dm stones -/
theorem hall_paving :
  stones_required 36 15 0.6 0.5 = 1800 := by
  sorry

end hall_paving_l3570_357074


namespace three_circles_cross_ratio_invariance_l3570_357092

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define a line by two points
structure Line where
  p1 : Point
  p2 : Point

-- Define the cross-ratio of four points on a line
def cross_ratio (p1 p2 p3 p4 : Point) : ℝ := sorry

-- Define a function to check if a point is on a circle
def point_on_circle (p : Point) (c : Circle) : Prop := sorry

-- Define a function to check if a point is on a line
def point_on_line (p : Point) (l : Line) : Prop := sorry

-- Define a function to find the intersection points of a line and a circle
def line_circle_intersection (l : Line) (c : Circle) : Set Point := sorry

theorem three_circles_cross_ratio_invariance 
  (c1 c2 c3 : Circle) 
  (A B : Point) 
  (h1 : point_on_circle A c1 ∧ point_on_circle A c2 ∧ point_on_circle A c3)
  (h2 : point_on_circle B c1 ∧ point_on_circle B c2 ∧ point_on_circle B c3)
  (h3 : A ≠ B) :
  ∀ (l1 l2 : Line), 
  (point_on_line A l1 ∧ point_on_line A l2) →
  ∃ (P1 Q1 R1 P2 Q2 R2 : Point),
  (P1 ∈ line_circle_intersection l1 c1 ∧ 
   Q1 ∈ line_circle_intersection l1 c2 ∧ 
   R1 ∈ line_circle_intersection l1 c3 ∧
   P2 ∈ line_circle_intersection l2 c1 ∧ 
   Q2 ∈ line_circle_intersection l2 c2 ∧ 
   R2 ∈ line_circle_intersection l2 c3) →
  cross_ratio A P1 Q1 R1 = cross_ratio A P2 Q2 R2 :=
sorry

end three_circles_cross_ratio_invariance_l3570_357092


namespace expression_zero_at_two_l3570_357046

theorem expression_zero_at_two (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  x = 2 → (1 / (x - 1) + 3 / (1 - x^2)) = 0 := by
  sorry

end expression_zero_at_two_l3570_357046


namespace tan_alpha_plus_pi_sixth_l3570_357099

theorem tan_alpha_plus_pi_sixth (α : Real) 
  (h : Real.cos α + 2 * Real.cos (α + π/3) = 0) : 
  Real.tan (α + π/6) = 3 * Real.sqrt 3 := by
  sorry

end tan_alpha_plus_pi_sixth_l3570_357099


namespace inequality_solution_set_l3570_357002

theorem inequality_solution_set (x : ℝ) : (2 * x - 1) / (3 * x + 1) > 1 ↔ -2 < x ∧ x < -1/3 := by
  sorry

end inequality_solution_set_l3570_357002


namespace log_function_through_point_l3570_357010

/-- Given a logarithmic function that passes through the point (4, 2), prove that its base is 2 -/
theorem log_function_through_point (f : ℝ → ℝ) (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : ∀ x > 0, f x = Real.log x / Real.log a) 
  (h4 : f 4 = 2) : 
  a = 2 := by
  sorry

end log_function_through_point_l3570_357010


namespace fraction_simplification_l3570_357005

theorem fraction_simplification : 1 / (Real.sqrt 3 - 2) = -Real.sqrt 3 - 2 := by
  sorry

end fraction_simplification_l3570_357005


namespace jake_paid_forty_l3570_357077

/-- Calculates the amount paid before working given initial debt, hourly rate, hours worked, and that the remaining debt was paid off by working. -/
def amount_paid_before_working (initial_debt : ℕ) (hourly_rate : ℕ) (hours_worked : ℕ) : ℕ :=
  initial_debt - (hourly_rate * hours_worked)

/-- Proves that Jake paid $40 before working, given the problem conditions. -/
theorem jake_paid_forty :
  amount_paid_before_working 100 15 4 = 40 := by
  sorry

end jake_paid_forty_l3570_357077


namespace fraction_equality_l3570_357097

theorem fraction_equality (a b c d : ℚ) 
  (h1 : a/b = 8)
  (h2 : c/b = 4)
  (h3 : c/d = 2/3) :
  d/a = 3/4 := by
sorry

end fraction_equality_l3570_357097


namespace min_value_S_l3570_357045

/-- The minimum value of (x-a)^2 + (ln x - a)^2 is 1/2, where x > 0 and a is real. -/
theorem min_value_S (x a : ℝ) (hx : x > 0) : 
  ∃ (min : ℝ), min = (1/2 : ℝ) ∧ ∀ y > 0, (y - a)^2 + (Real.log y - a)^2 ≥ min :=
sorry

end min_value_S_l3570_357045


namespace intersection_k_value_l3570_357091

theorem intersection_k_value (k : ℝ) : 
  (∃ y : ℝ, 3 * (-6) - 2 * y = k ∧ -6 - 0.5 * y = 10) → k = 46 := by
  sorry

end intersection_k_value_l3570_357091


namespace quadratic_coefficient_l3570_357065

/-- Given a quadratic function y = ax² + bx + c, if (2, y₁) and (-2, y₂) are points on this function
    and y₁ - y₂ = -16, then b = -4. -/
theorem quadratic_coefficient (a b c y₁ y₂ : ℝ) : 
  y₁ = a * 2^2 + b * 2 + c →
  y₂ = a * (-2)^2 + b * (-2) + c →
  y₁ - y₂ = -16 →
  b = -4 := by
sorry

end quadratic_coefficient_l3570_357065


namespace x_values_l3570_357026

theorem x_values (x : ℝ) : x ∈ ({1, 2, x^2} : Set ℝ) → x = 0 ∨ x = 2 := by
  sorry

end x_values_l3570_357026


namespace ratio_of_numbers_l3570_357072

theorem ratio_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end ratio_of_numbers_l3570_357072


namespace choose_three_from_nine_l3570_357089

theorem choose_three_from_nine (n : ℕ) (r : ℕ) (h1 : n = 9) (h2 : r = 3) :
  Nat.choose n r = 84 := by
  sorry

end choose_three_from_nine_l3570_357089


namespace terrell_hike_distance_l3570_357057

/-- The total distance hiked by Terrell over two days -/
def total_distance (saturday_distance sunday_distance : ℝ) : ℝ :=
  saturday_distance + sunday_distance

/-- Theorem stating that Terrell's total hiking distance is 9.8 miles -/
theorem terrell_hike_distance :
  total_distance 8.2 1.6 = 9.8 := by
  sorry

end terrell_hike_distance_l3570_357057


namespace sum_of_cubes_product_l3570_357039

theorem sum_of_cubes_product : ∃ a b : ℤ, a^3 + b^3 = 35 ∧ a * b = 6 := by
  sorry

end sum_of_cubes_product_l3570_357039


namespace divisibility_implication_l3570_357087

theorem divisibility_implication (x y : ℤ) : (2*x + 1) ∣ (8*y) → (2*x + 1) ∣ y := by
  sorry

end divisibility_implication_l3570_357087


namespace office_supplies_cost_l3570_357021

def pencil_cost : ℝ := 0.5
def folder_cost : ℝ := 0.9
def pencil_quantity : ℕ := 24  -- two dozen
def folder_quantity : ℕ := 20

def total_cost : ℝ := pencil_cost * pencil_quantity + folder_cost * folder_quantity

theorem office_supplies_cost : total_cost = 30 := by
  sorry

end office_supplies_cost_l3570_357021


namespace quadratic_has_real_root_l3570_357041

theorem quadratic_has_real_root (a b : ℝ) : ∃ x : ℝ, x^2 + a*x + b = 0 := by
  sorry

end quadratic_has_real_root_l3570_357041


namespace julia_tag_total_l3570_357025

/-- The number of kids Julia played tag with on Monday -/
def monday_kids : ℕ := 16

/-- The number of kids Julia played tag with on Tuesday -/
def tuesday_kids : ℕ := 14

/-- The total number of kids Julia played tag with over two days -/
def total_kids : ℕ := monday_kids + tuesday_kids

theorem julia_tag_total : total_kids = 30 := by sorry

end julia_tag_total_l3570_357025


namespace black_lambs_count_l3570_357059

/-- The total number of lambs -/
def total_lambs : ℕ := 6048

/-- The number of white lambs -/
def white_lambs : ℕ := 193

/-- Theorem: The number of black lambs is 5855 -/
theorem black_lambs_count : total_lambs - white_lambs = 5855 := by
  sorry

end black_lambs_count_l3570_357059


namespace probability_three_white_balls_l3570_357007

theorem probability_three_white_balls (total_balls : ℕ) (white_balls : ℕ) (drawn_balls : ℕ) :
  total_balls = 15 →
  white_balls = 7 →
  drawn_balls = 3 →
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 1 / 13 := by
  sorry

end probability_three_white_balls_l3570_357007


namespace central_square_area_central_square_area_proof_l3570_357075

/-- Given a square with side length 6 composed of smaller squares with side length 2,
    the area of the central square formed by removing one small square from each corner is 20. -/
theorem central_square_area : ℕ → ℕ → ℕ → Prop :=
  fun large_side small_side central_area =>
    large_side = 6 ∧ 
    small_side = 2 ∧ 
    large_side % small_side = 0 ∧
    (large_side / small_side) ^ 2 - 4 = 5 ∧ 
    central_area = 5 * small_side ^ 2 ∧
    central_area = 20

/-- Proof of the theorem -/
theorem central_square_area_proof : central_square_area 6 2 20 := by
  sorry

end central_square_area_central_square_area_proof_l3570_357075


namespace missing_number_l3570_357033

theorem missing_number (x : ℤ) : x + 64 = 16 → x = -48 := by
  sorry

end missing_number_l3570_357033


namespace recurring_decimal_equals_two_fifteenths_l3570_357081

-- Define the recurring decimal 0.333...
def recurring_third : ℚ := 1/3

-- Define the recurring decimal 0.1333...
def recurring_decimal : ℚ := 0.1333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333

-- State the theorem
theorem recurring_decimal_equals_two_fifteenths 
  (h : recurring_third = 1/3) : 
  recurring_decimal = 2/15 := by
  sorry

end recurring_decimal_equals_two_fifteenths_l3570_357081


namespace population_decrease_rate_l3570_357071

theorem population_decrease_rate (initial_population : ℝ) (final_population : ℝ) (years : ℕ) 
  (h1 : initial_population = 8000)
  (h2 : final_population = 3920)
  (h3 : years = 2) :
  ∃ (rate : ℝ), initial_population * (1 - rate)^years = final_population ∧ rate = 0.3 := by
sorry

end population_decrease_rate_l3570_357071


namespace first_player_wins_l3570_357079

/-- Represents a board in the game -/
structure Board :=
  (m : ℕ)

/-- Represents a position on the board -/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Represents a move in the game -/
inductive Move
  | Up
  | Down
  | Left
  | Right

/-- Represents the game state -/
structure GameState :=
  (board : Board)
  (currentPosition : Position)
  (usedSegments : List (Position × Position))

/-- Checks if a move is valid -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  match move with
  | Move.Up => state.currentPosition.y < state.board.m
  | Move.Down => state.currentPosition.y > 0
  | Move.Left => state.currentPosition.x > 0
  | Move.Right => state.currentPosition.x < state.board.m - 1

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Represents a winning strategy for the first player -/
def winningStrategy (board : Board) : Prop :=
  ∃ (strategy : List Move),
    ∀ (opponentMoves : List Move),
      let finalState := (strategy ++ opponentMoves).foldl applyMove
        { board := board
        , currentPosition := ⟨0, 0⟩
        , usedSegments := []
        }
      ¬∃ (move : Move), isValidMove finalState move

/-- The main theorem: there exists a winning strategy for the first player -/
theorem first_player_wins (m : ℕ) (h : m > 1) :
  winningStrategy { m := m } :=
  sorry

end first_player_wins_l3570_357079


namespace percentage_problem_l3570_357022

/-- The problem statement --/
theorem percentage_problem (P : ℝ) : 
  (P / 100) * 200 = (60 / 100) * 50 + 30 → P = 30 := by
  sorry

end percentage_problem_l3570_357022


namespace bird_cages_count_l3570_357036

/-- The number of bird cages in a pet store -/
def num_cages : ℕ := 6

/-- The number of parrots in each cage -/
def parrots_per_cage : ℝ := 6.0

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℝ := 2.0

/-- The total number of birds in the pet store -/
def total_birds : ℕ := 48

/-- Theorem stating that the number of bird cages is correct given the conditions -/
theorem bird_cages_count :
  (parrots_per_cage + parakeets_per_cage) * num_cages = total_birds := by
  sorry

end bird_cages_count_l3570_357036


namespace condition_relationship_l3570_357011

theorem condition_relationship (a b : ℝ) : 
  (((a > 2 ∧ b > 2) → (a + b > 4)) ∧ 
   (∃ (x y : ℝ), x + y > 4 ∧ (x ≤ 2 ∨ y ≤ 2))) :=
by sorry

end condition_relationship_l3570_357011
