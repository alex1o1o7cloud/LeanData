import Mathlib

namespace NUMINAMATH_CALUDE_investment_problem_l3211_321163

/-- The investment problem -/
theorem investment_problem 
  (x_investment : ℕ) 
  (z_investment : ℕ) 
  (z_join_time : ℕ) 
  (total_profit : ℕ) 
  (z_profit_share : ℕ) 
  (total_time : ℕ) 
  (h1 : x_investment = 36000)
  (h2 : z_investment = 48000)
  (h3 : z_join_time = 4)
  (h4 : total_profit = 13860)
  (h5 : z_profit_share = 4032)
  (h6 : total_time = 12) :
  ∃ y_investment : ℕ, 
    y_investment * total_time * (total_profit - z_profit_share) = 
    x_investment * total_time * z_profit_share - 
    z_investment * (total_time - z_join_time) * (total_profit - z_profit_share) ∧
    y_investment = 25000 := by
  sorry


end NUMINAMATH_CALUDE_investment_problem_l3211_321163


namespace NUMINAMATH_CALUDE_find_x_l3211_321104

theorem find_x (x y z a b c d k : ℝ) 
  (h1 : (x * y + k) / (x + y) = a)
  (h2 : (x * z + k) / (x + z) = b)
  (h3 : (y * z + k) / (y + z) = c)
  (hk : k ≠ 0) :
  x = (2 * a * b * c * d) / (b * (a * c - k) + c * (a * b - k) - a * (b * c - k)) :=
sorry

end NUMINAMATH_CALUDE_find_x_l3211_321104


namespace NUMINAMATH_CALUDE_find_unknown_number_l3211_321189

theorem find_unknown_number : ∃ x : ℝ, (213 * 16 = 3408) ∧ (1.6 * x = 3.408) → x = 2.13 := by
  sorry

end NUMINAMATH_CALUDE_find_unknown_number_l3211_321189


namespace NUMINAMATH_CALUDE_difference_of_squares_fraction_l3211_321191

theorem difference_of_squares_fraction :
  (113^2 - 104^2) / 9 = 217 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_fraction_l3211_321191


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3211_321152

/-- An arithmetic sequence is defined by its first term and common difference. -/
structure ArithmeticSequence where
  first_term : ℤ
  common_difference : ℤ

/-- Get the nth term of an arithmetic sequence. -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first_term + (n - 1) * seq.common_difference

/-- Theorem: In an arithmetic sequence where the 4th term is 23 and the 6th term is 51, the 8th term is 79. -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h1 : seq.nthTerm 4 = 23)
    (h2 : seq.nthTerm 6 = 51) : 
    seq.nthTerm 8 = 79 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3211_321152


namespace NUMINAMATH_CALUDE_inverse_functions_theorem_l3211_321124

-- Define the set of graph labels
inductive GraphLabel
| A | B | C | D | E

-- Define a property for a function to have an inverse based on the Horizontal Line Test
def has_inverse (g : GraphLabel) : Prop :=
  match g with
  | GraphLabel.B => True
  | GraphLabel.C => True
  | _ => False

-- Define a function that checks if a graph passes the Horizontal Line Test
def passes_horizontal_line_test (g : GraphLabel) : Prop :=
  has_inverse g

-- Theorem statement
theorem inverse_functions_theorem :
  (∀ g : GraphLabel, has_inverse g ↔ passes_horizontal_line_test g) ∧
  (has_inverse GraphLabel.B ∧ has_inverse GraphLabel.C) ∧
  (¬ has_inverse GraphLabel.A ∧ ¬ has_inverse GraphLabel.D ∧ ¬ has_inverse GraphLabel.E) :=
sorry

end NUMINAMATH_CALUDE_inverse_functions_theorem_l3211_321124


namespace NUMINAMATH_CALUDE_range_of_c_l3211_321109

def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y

def q (c : ℝ) : Prop := ∀ x : ℝ, 2*c*x^2 + 2*x + 1 > 0

theorem range_of_c (c : ℝ) (h1 : c > 0) (h2 : (p c ∨ q c) ∧ ¬(p c ∧ q c)) :
  c ∈ Set.Ioo 0 (1/2) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_c_l3211_321109


namespace NUMINAMATH_CALUDE_dividend_division_theorem_l3211_321198

theorem dividend_division_theorem : ∃ (q r : ℕ), 
  220025 = (555 + 445) * q + r ∧ 
  r < (555 + 445) ∧ 
  r = 25 ∧ 
  q = 2 * (555 - 445) := by
  sorry

end NUMINAMATH_CALUDE_dividend_division_theorem_l3211_321198


namespace NUMINAMATH_CALUDE_cone_radius_l3211_321106

/-- Given a cone with surface area 6 and lateral surface that unfolds into a semicircle,
    prove that the radius of its base is √(2/π) -/
theorem cone_radius (r : ℝ) (l : ℝ) : 
  r > 0 →  -- radius is positive
  l > 0 →  -- slant height is positive
  2 * π * r = π * l →  -- lateral surface unfolds into a semicircle
  π * r^2 + π * r * l = 6 →  -- surface area is 6
  r = Real.sqrt (2 / π) := by
sorry

end NUMINAMATH_CALUDE_cone_radius_l3211_321106


namespace NUMINAMATH_CALUDE_unique_seven_numbers_sum_100_l3211_321192

theorem unique_seven_numbers_sum_100 (a₄ : ℕ) : 
  ∃! (a₁ a₂ a₃ a₅ a₆ a₇ : ℕ), 
    a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ a₄ < a₅ ∧ a₅ < a₆ ∧ a₆ < a₇ ∧
    a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 100 :=
by sorry

end NUMINAMATH_CALUDE_unique_seven_numbers_sum_100_l3211_321192


namespace NUMINAMATH_CALUDE_intersection_points_form_geometric_sequence_l3211_321118

-- Define the curve C
def curve (a : ℝ) (x y : ℝ) : Prop := y^2 = 2*a*x

-- Define the line l
def line (t : ℝ) : ℝ × ℝ := (-2 + t, -4 + t)

-- Define the point P
def P : ℝ × ℝ := (-2, -4)

-- Define the property of geometric sequence for three positive real numbers
def is_geometric_sequence (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ b^2 = a*c

-- Main theorem
theorem intersection_points_form_geometric_sequence (a : ℝ) :
  a > 0 →
  ∃ t₁ t₂ : ℝ,
    let M := line t₁
    let N := line t₂
    curve a M.1 M.2 ∧
    curve a N.1 N.2 ∧
    is_geometric_sequence (Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2))
                          (Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2))
                          (Real.sqrt ((P.1 - N.1)^2 + (P.2 - N.2)^2)) →
  a = 1 := by
  sorry


end NUMINAMATH_CALUDE_intersection_points_form_geometric_sequence_l3211_321118


namespace NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l3211_321112

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : List (List ℕ) := sorry

/-- Counts the number of ones in the first n rows of Pascal's Triangle -/
def countOnes (n : ℕ) : ℕ := sorry

/-- Counts the total number of elements in the first n rows of Pascal's Triangle -/
def totalElements (n : ℕ) : ℕ := sorry

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def probabilityOfOne (n : ℕ) : ℚ :=
  (countOnes n : ℚ) / (totalElements n : ℚ)

theorem probability_of_one_in_20_rows :
  probabilityOfOne 20 = 13 / 70 := by sorry

end NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l3211_321112


namespace NUMINAMATH_CALUDE_mabel_katrina_marble_ratio_l3211_321190

/-- Prove that Mabel has 5 times as many marbles as Katrina -/
theorem mabel_katrina_marble_ratio : 
  ∀ (amanda katrina mabel : ℕ),
  amanda + 12 = 2 * katrina →
  mabel = 85 →
  mabel = amanda + 63 →
  mabel / katrina = 5 := by
sorry

end NUMINAMATH_CALUDE_mabel_katrina_marble_ratio_l3211_321190


namespace NUMINAMATH_CALUDE_smallest_cube_root_integer_l3211_321166

theorem smallest_cube_root_integer (m n : ℕ) (r : ℝ) : 
  (∃ m : ℕ, ∃ r : ℝ, 
    m > 0 ∧ 
    r > 0 ∧ 
    r < 1/1000 ∧ 
    (m : ℝ)^(1/3 : ℝ) = n + r) → 
  n ≥ 19 :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_integer_l3211_321166


namespace NUMINAMATH_CALUDE_used_car_seller_problem_l3211_321179

theorem used_car_seller_problem (num_clients : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ) : 
  num_clients = 18 → cars_per_client = 3 → selections_per_car = 3 →
  num_clients * cars_per_client / selections_per_car = 18 := by
sorry

end NUMINAMATH_CALUDE_used_car_seller_problem_l3211_321179


namespace NUMINAMATH_CALUDE_pencil_case_solution_l3211_321177

/-- Represents the cost and quantity of pencil cases --/
structure PencilCases where
  cost_A : ℚ
  cost_B : ℚ
  quantity_A : ℕ
  quantity_B : ℕ

/-- Conditions for the pencil case problem --/
def PencilCaseProblem (p : PencilCases) : Prop :=
  p.cost_B = p.cost_A + 2 ∧
  800 / p.cost_A = 1000 / p.cost_B ∧
  p.quantity_A = 3 * p.quantity_B - 50 ∧
  p.quantity_A + p.quantity_B ≤ 910 ∧
  12 * p.quantity_A + 15 * p.quantity_B - 
  (p.cost_A * p.quantity_A + p.cost_B * p.quantity_B) > 3795

/-- The main theorem to prove --/
theorem pencil_case_solution (p : PencilCases) 
  (h : PencilCaseProblem p) : 
  p.cost_A = 8 ∧ 
  p.cost_B = 10 ∧ 
  p.quantity_B ≤ 240 ∧ 
  (∃ n : ℕ, n = 5 ∧ 
    ∀ m : ℕ, 236 ≤ m ∧ m ≤ 240 → 
      (12 * (3 * m - 50) + 15 * m - (8 * (3 * m - 50) + 10 * m) > 3795)) := by
  sorry


end NUMINAMATH_CALUDE_pencil_case_solution_l3211_321177


namespace NUMINAMATH_CALUDE_system_solution_unique_l3211_321132

theorem system_solution_unique : 
  ∃! (x y : ℝ), (2 * x - y = 5) ∧ (3 * x + 2 * y = -3) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3211_321132


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3211_321187

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {x | x > 1}

-- State the theorem
theorem complement_union_theorem :
  (A ∪ B)ᶜ = {x : ℝ | 0 ≤ x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3211_321187


namespace NUMINAMATH_CALUDE_min_value_theorem_l3211_321154

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem min_value_theorem (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 7 = a 6 + 2 * a 5 →
  (∃ m n : ℕ, a m * a n = 8 * (a 1)^2) →
  (∃ m n : ℕ, 1 / m + 4 / n = 11 / 6 ∧
    ∀ k l : ℕ, 1 / k + 4 / l ≥ 11 / 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3211_321154


namespace NUMINAMATH_CALUDE_discounted_tickets_count_l3211_321113

/-- Proves the number of discounted tickets bought given the problem conditions -/
theorem discounted_tickets_count :
  ∀ (full_price discounted_price : ℚ) 
    (total_tickets : ℕ) 
    (total_spent : ℚ),
  full_price = 2 →
  discounted_price = (8 : ℚ) / 5 →
  total_tickets = 10 →
  total_spent = (92 : ℚ) / 5 →
  ∃ (full_tickets discounted_tickets : ℕ),
    full_tickets + discounted_tickets = total_tickets ∧
    full_price * full_tickets + discounted_price * discounted_tickets = total_spent ∧
    discounted_tickets = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_discounted_tickets_count_l3211_321113


namespace NUMINAMATH_CALUDE_park_diameter_l3211_321121

/-- Given a circular park with concentric rings, calculate the diameter of the outer boundary. -/
theorem park_diameter (statue_width garden_width path_width fountain_diameter : ℝ) : 
  statue_width = 2 ∧ 
  garden_width = 10 ∧ 
  path_width = 8 ∧ 
  fountain_diameter = 12 → 
  2 * (fountain_diameter / 2 + statue_width + garden_width + path_width) = 52 := by
sorry

end NUMINAMATH_CALUDE_park_diameter_l3211_321121


namespace NUMINAMATH_CALUDE_abs_a_minus_3_l3211_321127

theorem abs_a_minus_3 (a : ℝ) (h : ∀ x : ℝ, (a - 2) * x > a - 2 ↔ x < 1) : 
  |a - 3| = 3 - a := by
  sorry

end NUMINAMATH_CALUDE_abs_a_minus_3_l3211_321127


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l3211_321180

theorem arithmetic_sequence_solution :
  let a₁ : ℚ := 3/4
  let a₂ : ℚ → ℚ := λ x => x - 2
  let a₃ : ℚ → ℚ := λ x => 5*x
  ∀ x : ℚ, (a₂ x - a₁ = a₃ x - a₂ x) → x = -19/12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l3211_321180


namespace NUMINAMATH_CALUDE_problem_solution_l3211_321146

def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |x - a|

theorem problem_solution :
  (∀ x : ℝ, f 1 x ≥ 1 ↔ x ∈ Set.Ici (1/2)) ∧
  (Set.Iio 1 = {a : ℝ | ∀ x ≥ 0, f a x < 2}) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3211_321146


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_one_l3211_321107

-- Define the distance function
def S (t : ℝ) : ℝ := t^3 - 2

-- State the theorem
theorem instantaneous_velocity_at_one (t : ℝ) : 
  (deriv S) 1 = 3 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_one_l3211_321107


namespace NUMINAMATH_CALUDE_election_winner_percentage_l3211_321197

theorem election_winner_percentage 
  (total_votes : ℕ) 
  (majority : ℕ) 
  (winning_percentage : ℚ) :
  total_votes = 6500 →
  majority = 1300 →
  winning_percentage * total_votes = (total_votes + majority) / 2 →
  winning_percentage = 60 / 100 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l3211_321197


namespace NUMINAMATH_CALUDE_min_translation_for_symmetry_l3211_321181

theorem min_translation_for_symmetry :
  let f (x m : ℝ) := Real.sin (2 * (x - m) - π / 6)
  ∀ m : ℝ, m > 0 →
    (∀ x : ℝ, f x m = f (-x) m) →
    m ≥ π / 6 :=
by sorry

end NUMINAMATH_CALUDE_min_translation_for_symmetry_l3211_321181


namespace NUMINAMATH_CALUDE_probability_even_product_l3211_321101

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

def is_even_product (a b : ℕ) : Prop := Even (a * b)

theorem probability_even_product :
  Nat.card {p : S × S | p.1 ≠ p.2 ∧ is_even_product p.1 p.2} / Nat.choose 7 2 = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_product_l3211_321101


namespace NUMINAMATH_CALUDE_theater_camp_talents_l3211_321108

theorem theater_camp_talents (total_students : ℕ) 
  (cannot_sing cannot_dance both_talents : ℕ) : 
  total_students = 120 →
  cannot_sing = 30 →
  cannot_dance = 50 →
  both_talents = 10 →
  (total_students - cannot_sing) + (total_students - cannot_dance) - both_talents = 130 :=
by sorry

end NUMINAMATH_CALUDE_theater_camp_talents_l3211_321108


namespace NUMINAMATH_CALUDE_sugar_solution_percentage_l3211_321175

theorem sugar_solution_percentage (x : ℝ) :
  (3/4 * x + 1/4 * 26 = 14) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_sugar_solution_percentage_l3211_321175


namespace NUMINAMATH_CALUDE_only_zero_function_satisfies_l3211_321134

/-- A function satisfying the given inequality for all non-zero real x and all real y -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x ≠ 0 → f (x^2 + y) ≥ (1/x + 1) * f y

/-- The main theorem stating that the only function satisfying the inequality is the zero function -/
theorem only_zero_function_satisfies :
  ∀ f : ℝ → ℝ, SatisfiesInequality f ↔ ∀ x, f x = 0 := by sorry

end NUMINAMATH_CALUDE_only_zero_function_satisfies_l3211_321134


namespace NUMINAMATH_CALUDE_cube_volume_problem_l3211_321130

theorem cube_volume_problem : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a^3 - ((a + 2) * (a - 2) * (a + 3)) = 7) ∧ 
  (a^3 = 27) := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l3211_321130


namespace NUMINAMATH_CALUDE_r_share_l3211_321126

/-- Given a total amount divided among three people P, Q, and R, with specified ratios,
    calculate R's share. -/
theorem r_share (total : ℕ) (p q r : ℕ) : 
  total = 1210 →
  5 * q = 4 * p →
  9 * r = 10 * q →
  r = 400 := by
  sorry


end NUMINAMATH_CALUDE_r_share_l3211_321126


namespace NUMINAMATH_CALUDE_kingdom_cats_and_hogs_l3211_321153

theorem kingdom_cats_and_hogs (num_hogs : ℕ) (num_cats : ℕ) : 
  num_hogs = 630 → 
  num_hogs = 7 * num_cats → 
  15 < (0.8 * (num_cats^2 : ℝ)) → 
  (0.8 * (num_cats^2 : ℝ)) - 15 = 6465 := by
sorry

end NUMINAMATH_CALUDE_kingdom_cats_and_hogs_l3211_321153


namespace NUMINAMATH_CALUDE_parametric_to_general_equation_l3211_321158

/-- Parametric equations to general equation conversion -/
theorem parametric_to_general_equation :
  ∀ θ : ℝ,
  let x : ℝ := 2 + Real.sin θ ^ 2
  let y : ℝ := -1 + Real.cos (2 * θ)
  2 * x + y - 4 = 0 ∧ x ∈ Set.Icc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_general_equation_l3211_321158


namespace NUMINAMATH_CALUDE_sweater_markup_percentage_l3211_321151

theorem sweater_markup_percentage 
  (wholesale : ℝ) 
  (retail : ℝ) 
  (h1 : wholesale > 0) 
  (h2 : retail > 0) 
  (h3 : 0.4 * retail = 1.2 * wholesale) : 
  (retail - wholesale) / wholesale * 100 = 200 := by
sorry

end NUMINAMATH_CALUDE_sweater_markup_percentage_l3211_321151


namespace NUMINAMATH_CALUDE_cubic_minus_linear_at_five_l3211_321171

theorem cubic_minus_linear_at_five : 
  let x : ℝ := 5
  (x^3 - 3*x) = 110 := by sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_at_five_l3211_321171


namespace NUMINAMATH_CALUDE_max_sum_of_primes_perfect_square_l3211_321145

theorem max_sum_of_primes_perfect_square (p q : ℕ) : 
  Prime p → Prime q → ∃ r : ℕ, p^2 + 3*p*q + q^2 = r^2 → p + q ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_primes_perfect_square_l3211_321145


namespace NUMINAMATH_CALUDE_greatest_integer_for_integer_fraction_l3211_321136

theorem greatest_integer_for_integer_fraction : 
  ∃ (x : ℤ), x = 53 ∧ 
  (∀ (y : ℤ), y > 53 → ¬(∃ (z : ℤ), (y^2 + 2*y + 13) / (y - 5) = z)) ∧
  (∃ (z : ℤ), (x^2 + 2*x + 13) / (x - 5) = z) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_for_integer_fraction_l3211_321136


namespace NUMINAMATH_CALUDE_solution_in_quadrant_IV_l3211_321122

/-- Given a system of equations x + 2y = 4 and kx - y = 1, where k is a constant,
    the solution (x, y) is in Quadrant IV if and only if -1/2 < k < 2 -/
theorem solution_in_quadrant_IV (k : ℝ) : 
  (∃ x y : ℝ, x + 2*y = 4 ∧ k*x - y = 1 ∧ x > 0 ∧ y < 0) ↔ -1/2 < k ∧ k < 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_quadrant_IV_l3211_321122


namespace NUMINAMATH_CALUDE_binomial_12_choose_2_l3211_321165

theorem binomial_12_choose_2 : Nat.choose 12 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_2_l3211_321165


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l3211_321103

/-- Given a cylinder whose radius is increased by a factor x and height is doubled,
    resulting in a volume 18 times the original, prove that x = 3 -/
theorem cylinder_volume_increase (r h x : ℝ) (hr : r > 0) (hh : h > 0) (hx : x > 0) :
  2 * x^2 * (π * r^2 * h) = 18 * (π * r^2 * h) → x = 3 := by
  sorry

#check cylinder_volume_increase

end NUMINAMATH_CALUDE_cylinder_volume_increase_l3211_321103


namespace NUMINAMATH_CALUDE_pipe_fill_time_l3211_321150

theorem pipe_fill_time (T : ℝ) (h1 : T > 0) (h2 : 1/T - 1/4.5 = 1/9) : T = 3 := by
  sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l3211_321150


namespace NUMINAMATH_CALUDE_parabola_directrix_l3211_321148

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = (x^2 - 8*x + 12) / 16

/-- The directrix equation -/
def directrix (y : ℝ) : Prop := y = -17/4

/-- Theorem: The directrix of the given parabola is y = -17/4 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), parabola x y → ∃ (d : ℝ), directrix d ∧ 
  (∀ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y → 
    (p.1 - 4)^2 + (p.2 - d)^2 = (p.2 - (d + 4))^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3211_321148


namespace NUMINAMATH_CALUDE_difference_median_mode_l3211_321131

def data : List ℕ := [36, 37, 37, 38, 40, 40, 40, 41, 42, 43, 54, 55, 57, 59, 61, 61, 65, 68, 69]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℝ := sorry

theorem difference_median_mode : 
  |median data - mode data| = 2 := by sorry

end NUMINAMATH_CALUDE_difference_median_mode_l3211_321131


namespace NUMINAMATH_CALUDE_parallel_planes_line_sufficient_not_necessary_l3211_321140

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relations
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (line_in_plane : Line → Plane → Prop)

theorem parallel_planes_line_sufficient_not_necessary 
  (α β : Plane) (l : Line) 
  (h_distinct : α ≠ β) 
  (h_l_in_α : line_in_plane l α) :
  (∀ α β l, plane_parallel α β → line_parallel_plane l β) ∧ 
  (∃ α β l, line_parallel_plane l β ∧ ¬plane_parallel α β) := by
  sorry


end NUMINAMATH_CALUDE_parallel_planes_line_sufficient_not_necessary_l3211_321140


namespace NUMINAMATH_CALUDE_triangle_obtuse_from_trig_inequality_l3211_321174

/-- Given a triangle ABC, if sin A * sin B < cos A * cos B, then ABC is an obtuse triangle -/
theorem triangle_obtuse_from_trig_inequality (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_inequality : Real.sin A * Real.sin B < Real.cos A * Real.cos B) : 
  π / 2 < C ∧ C < π :=
sorry

end NUMINAMATH_CALUDE_triangle_obtuse_from_trig_inequality_l3211_321174


namespace NUMINAMATH_CALUDE_y_minus_x_value_l3211_321178

theorem y_minus_x_value (x y : ℚ) (h1 : x + y = 8) (h2 : y - 3 * x = 7) : 
  y - x = (15 : ℚ) / 2 := by sorry

end NUMINAMATH_CALUDE_y_minus_x_value_l3211_321178


namespace NUMINAMATH_CALUDE_min_prime_divisor_of_quadratic_l3211_321128

theorem min_prime_divisor_of_quadratic : 
  ∀ p : ℕ, Prime p → (∃ n : ℕ, p ∣ (n^2 + 7*n + 23)) → p ≥ 11 :=
by sorry

end NUMINAMATH_CALUDE_min_prime_divisor_of_quadratic_l3211_321128


namespace NUMINAMATH_CALUDE_john_jane_difference_l3211_321119

/-- The width of the streets in Perfectville -/
def street_width : ℕ := 30

/-- The side length of a block in Perfectville -/
def block_side_length : ℕ := 500

/-- The side length of John's path -/
def john_path_side : ℕ := block_side_length + 2 * street_width

/-- The perimeter of Jane's path -/
def jane_perimeter : ℕ := 4 * block_side_length

/-- The perimeter of John's path -/
def john_perimeter : ℕ := 4 * john_path_side

theorem john_jane_difference :
  john_perimeter - jane_perimeter = 240 := by sorry

end NUMINAMATH_CALUDE_john_jane_difference_l3211_321119


namespace NUMINAMATH_CALUDE_rectangle_to_cylinder_volume_l3211_321164

/-- The volume of a cylinder formed by rolling a rectangle with length 6 and width 3 -/
theorem rectangle_to_cylinder_volume :
  ∃ (V : ℝ), (V = 27 / π ∨ V = 27 / (4 * π)) ∧
  ∃ (R h : ℝ), (R * h = 18 ∨ R * h = 9) ∧ V = π * R^2 * h := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_cylinder_volume_l3211_321164


namespace NUMINAMATH_CALUDE_range_of_a_l3211_321143

def A (a : ℝ) : Set ℝ := {x | a * x < 1}
def B : Set ℝ := {x | |x - 1| < 2}

theorem range_of_a (a : ℝ) (h : A a ∪ B = A a) : a ∈ Set.Icc (-1 : ℝ) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3211_321143


namespace NUMINAMATH_CALUDE_quadratic_inequality_bc_l3211_321129

/-- Given a quadratic inequality x^2 + bx + c ≤ 0 with solution set [-2, 5], 
    prove that bc = 30 -/
theorem quadratic_inequality_bc (b c : ℝ) : 
  (∀ x, x^2 + b*x + c ≤ 0 ↔ -2 ≤ x ∧ x ≤ 5) → b*c = 30 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_bc_l3211_321129


namespace NUMINAMATH_CALUDE_system_solution_l3211_321117

def solution_set : Set (ℝ × ℝ) := {(3, 2)}

theorem system_solution :
  {(x, y) : ℝ × ℝ | x + y = 5 ∧ x - y = 1} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3211_321117


namespace NUMINAMATH_CALUDE_cone_prism_volume_ratio_l3211_321173

/-- The ratio of the volume of a right circular cone to the volume of its circumscribing right rectangular prism -/
theorem cone_prism_volume_ratio :
  ∀ (r h : ℝ), r > 0 → h > 0 →
  (1 / 3 * π * r^2 * h) / (9 * r^2 * h) = π / 27 := by
sorry

end NUMINAMATH_CALUDE_cone_prism_volume_ratio_l3211_321173


namespace NUMINAMATH_CALUDE_men_to_women_percentage_l3211_321156

theorem men_to_women_percentage (men women : ℕ) (h : women = men / 2) :
  (men : ℚ) / (women : ℚ) * 100 = 200 := by
  sorry

end NUMINAMATH_CALUDE_men_to_women_percentage_l3211_321156


namespace NUMINAMATH_CALUDE_hide_and_seek_time_l3211_321155

/-- Represents the square wall in the hide and seek game -/
structure Square :=
  (side_length : ℝ)

/-- Represents a player in the hide and seek game -/
structure Player :=
  (speed : ℝ)
  (corner_pause : ℝ)

/-- Calculates the time needed for a player to see the other player -/
def time_to_see (s : Square) (a b : Player) : ℝ :=
  sorry

/-- Theorem stating that the minimum time for A to see B is 8 minutes -/
theorem hide_and_seek_time (s : Square) (a b : Player) :
  s.side_length = 100 ∧
  a.speed = 50 ∧
  b.speed = 30 ∧
  a.corner_pause = 1 ∧
  b.corner_pause = 1 →
  time_to_see s a b = 8 :=
sorry

end NUMINAMATH_CALUDE_hide_and_seek_time_l3211_321155


namespace NUMINAMATH_CALUDE_max_abs_z_quadratic_equation_l3211_321184

open Complex

theorem max_abs_z_quadratic_equation (a b c z : ℂ) 
  (h1 : abs a = 1) (h2 : abs b = 1) (h3 : abs c = 1)
  (h4 : arg c = arg a + arg b)
  (h5 : a * z^2 + b * z + c = 0) :
  abs z ≤ (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_quadratic_equation_l3211_321184


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l3211_321114

theorem range_of_a_minus_b (a b : ℝ) (ha : -2 < a ∧ a < 1) (hb : 0 < b ∧ b < 4) :
  ∀ x, (∃ (a' b' : ℝ), -2 < a' ∧ a' < 1 ∧ 0 < b' ∧ b' < 4 ∧ x = a' - b') ↔ -6 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l3211_321114


namespace NUMINAMATH_CALUDE_max_value_sum_sqrt_l3211_321125

theorem max_value_sum_sqrt (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 2 ∧ 
  (∀ a' b' c' : ℝ, 0 < a' → 0 < b' → 0 < c' → a' + b' + c' = 1 →
    Real.sqrt (3 * a' + 1) + Real.sqrt (3 * b' + 1) + Real.sqrt (3 * c' + 1) ≤ max) ∧
  Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) = max :=
sorry

end NUMINAMATH_CALUDE_max_value_sum_sqrt_l3211_321125


namespace NUMINAMATH_CALUDE_additive_function_value_l3211_321169

/-- A function satisfying f(x+y) = f(x) + f(y) for all real x and y -/
def AdditiveFunctionR (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

/-- Theorem: If f is an additive function on ℝ and f(2) = 4, then f(1) = 2 -/
theorem additive_function_value (f : ℝ → ℝ) (h1 : AdditiveFunctionR f) (h2 : f 2 = 4) : f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_additive_function_value_l3211_321169


namespace NUMINAMATH_CALUDE_probability_of_triangle_formation_l3211_321167

/-- Regular 15-gon with unit circumradius -/
def regular_15gon : Set (ℝ × ℝ) := sorry

/-- Set of all segments in the 15-gon -/
def segments (poly : Set (ℝ × ℝ)) : Set (Set (ℝ × ℝ)) := sorry

/-- Function to calculate the length of a segment -/
def segment_length (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Predicate to check if three segments form a triangle with positive area -/
def forms_triangle (s1 s2 s3 : Set (ℝ × ℝ)) : Prop := sorry

/-- The total number of ways to choose 3 segments from the 15-gon -/
def total_combinations : ℕ := Nat.choose 105 3

/-- The number of valid triangles formed by three segments -/
def valid_triangles : ℕ := sorry

theorem probability_of_triangle_formation :
  (valid_triangles : ℚ) / total_combinations = 323 / 429 := by sorry

end NUMINAMATH_CALUDE_probability_of_triangle_formation_l3211_321167


namespace NUMINAMATH_CALUDE_stating_max_fully_connected_is_N_minus_1_l3211_321116

/-- Represents a network of computers. -/
structure Network where
  N : ℕ
  not_fully_connected : ∃ (node : Fin N), ∃ (other : Fin N), node ≠ other
  N_gt_3 : N > 3

/-- The maximum number of fully connected nodes in the network. -/
def max_fully_connected (net : Network) : ℕ := net.N - 1

/-- 
Theorem stating that the maximum number of fully connected nodes 
in a network with the given conditions is N-1.
-/
theorem max_fully_connected_is_N_minus_1 (net : Network) : 
  max_fully_connected net = net.N - 1 := by
  sorry


end NUMINAMATH_CALUDE_stating_max_fully_connected_is_N_minus_1_l3211_321116


namespace NUMINAMATH_CALUDE_max_bananas_purchase_l3211_321159

def apple_cost : ℕ := 3
def orange_cost : ℕ := 5
def banana_cost : ℕ := 8
def total_budget : ℕ := 100

def is_valid_purchase (apples oranges bananas : ℕ) : Prop :=
  apples ≥ 1 ∧ oranges ≥ 1 ∧ bananas ≥ 1 ∧
  apple_cost * apples + orange_cost * oranges + banana_cost * bananas ≤ total_budget

theorem max_bananas_purchase :
  ∃ (apples oranges : ℕ),
    is_valid_purchase apples oranges 11 ∧
    ∀ (a o b : ℕ), is_valid_purchase a o b → b ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_bananas_purchase_l3211_321159


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l3211_321194

/-- The area of a triangle given its side lengths -/
noncomputable def S (a b c : ℝ) : ℝ := sorry

/-- Triangle inequality -/
def is_triangle (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_area_inequality 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (h₁ : is_triangle a₁ b₁ c₁) 
  (h₂ : is_triangle a₂ b₂ c₂) : 
  Real.sqrt (S a₁ b₁ c₁) + Real.sqrt (S a₂ b₂ c₂) ≤ Real.sqrt (S (a₁ + a₂) (b₁ + b₂) (c₁ + c₂)) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l3211_321194


namespace NUMINAMATH_CALUDE_smallest_root_of_unity_order_l3211_321186

open Complex

theorem smallest_root_of_unity_order : ∃ (n : ℕ), n > 0 ∧ 
  (∀ z : ℂ, z^5 - z^3 + 1 = 0 → z^n = 1) ∧
  (∀ m : ℕ, m > 0 → (∀ z : ℂ, z^5 - z^3 + 1 = 0 → z^m = 1) → m ≥ n) ∧
  n = 16 := by
  sorry

end NUMINAMATH_CALUDE_smallest_root_of_unity_order_l3211_321186


namespace NUMINAMATH_CALUDE_sum_a_c_equals_six_l3211_321147

theorem sum_a_c_equals_six (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 36) 
  (h2 : b + d = 6) : 
  a + c = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_a_c_equals_six_l3211_321147


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l3211_321105

theorem p_sufficient_not_necessary_q :
  (∀ x : ℝ, 0 < x ∧ x < 2 → -1 < x ∧ x < 3) ∧
  (∃ x : ℝ, -1 < x ∧ x < 3 ∧ ¬(0 < x ∧ x < 2)) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l3211_321105


namespace NUMINAMATH_CALUDE_largest_coin_through_hole_l3211_321141

-- Define the diameter of coins
def diameter (coin : String) : ℝ :=
  match coin with
  | "1 kopeck" => 1
  | "20 kopeck" => 2
  | _ => 0

-- Define a circular hole
structure CircularHole where
  diameter : ℝ

-- Define a function to check if a coin can pass through a hole when paper is folded
def canPassThroughWhenFolded (coin : String) (hole : CircularHole) : Prop :=
  diameter coin ≤ 2 * hole.diameter

theorem largest_coin_through_hole :
  let hole : CircularHole := ⟨diameter "1 kopeck"⟩
  canPassThroughWhenFolded "20 kopeck" hole := by
  sorry

end NUMINAMATH_CALUDE_largest_coin_through_hole_l3211_321141


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_foci_l3211_321138

theorem ellipse_hyperbola_foci (a b : ℝ) 
  (h1 : b^2 - a^2 = 25)  -- Condition for ellipse foci
  (h2 : a^2 + b^2 = 49)  -- Condition for hyperbola foci
  : a = 2 * Real.sqrt 3 ∧ b = Real.sqrt 37 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_foci_l3211_321138


namespace NUMINAMATH_CALUDE_adult_price_calculation_l3211_321137

/-- The daily price for adults at a public swimming pool -/
def adult_price (total_people : ℕ) (child_price : ℚ) (total_receipts : ℚ) (num_children : ℕ) : ℚ :=
  let num_adults : ℕ := total_people - num_children
  (total_receipts - (num_children : ℚ) * child_price) / (num_adults : ℚ)

/-- Theorem stating the adult price calculation for the given scenario -/
theorem adult_price_calculation :
  adult_price 754 (3/2) 1422 388 = 840 / 366 := by
  sorry

end NUMINAMATH_CALUDE_adult_price_calculation_l3211_321137


namespace NUMINAMATH_CALUDE_average_book_width_l3211_321157

theorem average_book_width :
  let book_widths : List ℚ := [5, 3/4, 3/2, 3, 29/4, 12]
  (book_widths.sum / book_widths.length : ℚ) = 59/12 := by
sorry

end NUMINAMATH_CALUDE_average_book_width_l3211_321157


namespace NUMINAMATH_CALUDE_largest_prime_divisor_xyxyxy_l3211_321172

/-- The largest prime divisor of a number in the form xyxyxy -/
theorem largest_prime_divisor_xyxyxy (x y : ℕ) (hx : x < 10) (hy : y < 10) :
  ∃ (p : ℕ), p.Prime ∧ p ∣ (100000 * x + 10000 * y + 1000 * x + 100 * y + 10 * x + y) ∧
  ∀ (q : ℕ), q.Prime → q ∣ (100000 * x + 10000 * y + 1000 * x + 100 * y + 10 * x + y) → q ≤ 97 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_xyxyxy_l3211_321172


namespace NUMINAMATH_CALUDE_loss_percentage_proof_l3211_321139

def calculate_loss_percentage (cost_prices selling_prices : List ℚ) : ℚ :=
  let total_cp := cost_prices.sum
  let total_sp := selling_prices.sum
  let loss := total_cp - total_sp
  (loss / total_cp) * 100

theorem loss_percentage_proof (cost_prices selling_prices : List ℚ) :
  cost_prices = [1200, 1500, 1800] →
  selling_prices = [800, 1300, 1500] →
  calculate_loss_percentage cost_prices selling_prices = 20 := by
  sorry

#eval calculate_loss_percentage [1200, 1500, 1800] [800, 1300, 1500]

end NUMINAMATH_CALUDE_loss_percentage_proof_l3211_321139


namespace NUMINAMATH_CALUDE_system_solution_l3211_321168

theorem system_solution :
  ∃! (x y z : ℚ),
    2 * x - 3 * y + z = 8 ∧
    4 * x - 6 * y + 2 * z = 16 ∧
    x + y - z = 1 ∧
    x = 11 / 3 ∧
    y = 1 ∧
    z = 11 / 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3211_321168


namespace NUMINAMATH_CALUDE_octagon_diagonals_l3211_321188

/-- The number of internal diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon is a polygon with 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: An octagon has 20 internal diagonals -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l3211_321188


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_y_l3211_321199

theorem max_value_of_x_plus_y (x y : ℝ) 
  (h1 : 5 * x + 3 * y ≤ 10) 
  (h2 : 3 * x + 6 * y ≤ 12) : 
  x + y ≤ 18 / 7 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_y_l3211_321199


namespace NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l3211_321135

/-- The quadratic function f(x) = -3x^2 + 24x - 45 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 24 * x - 45

/-- The same function in completed square form a(x+b)^2 + c -/
def g (x a b c : ℝ) : ℝ := a * (x + b)^2 + c

theorem quadratic_sum_of_coefficients :
  ∃ (a b c : ℝ), (∀ x, f x = g x a b c) ∧ (a + b + c = 4) := by sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l3211_321135


namespace NUMINAMATH_CALUDE_unique_fraction_representation_l3211_321170

theorem unique_fraction_representation (p : ℕ) (h_prime : Nat.Prime p) (h_greater_than_two : p > 2) :
  ∃! (x y : ℕ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ (2 : ℚ) / p = 1 / x + 1 / y := by
  sorry

end NUMINAMATH_CALUDE_unique_fraction_representation_l3211_321170


namespace NUMINAMATH_CALUDE_teacher_earnings_five_weeks_l3211_321142

/-- Calculates the teacher's earnings for piano lessons over a given number of weeks -/
def teacher_earnings (rate_per_half_hour : ℕ) (lesson_duration_hours : ℕ) (lessons_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  rate_per_half_hour * (lesson_duration_hours * 2) * lessons_per_week * num_weeks

/-- Proves that the teacher earns $100 in 5 weeks under the given conditions -/
theorem teacher_earnings_five_weeks :
  teacher_earnings 10 1 1 5 = 100 := by
  sorry

end NUMINAMATH_CALUDE_teacher_earnings_five_weeks_l3211_321142


namespace NUMINAMATH_CALUDE_ball_drawing_game_l3211_321196

/-- Represents the probability that the last ball is white in the ball-drawing game. -/
def lastBallWhiteProbability (p : ℕ) : ℚ :=
  if p % 2 = 0 then 0 else 1

/-- The ball-drawing game theorem. -/
theorem ball_drawing_game (p q : ℕ) :
  ∀ (pile : ℕ), lastBallWhiteProbability p = if p % 2 = 0 then 0 else 1 := by
  sorry

#check ball_drawing_game

end NUMINAMATH_CALUDE_ball_drawing_game_l3211_321196


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3211_321110

theorem unique_quadratic_solution (a : ℝ) :
  (∃! x : ℝ, a * x^2 + a * x + 1 = 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3211_321110


namespace NUMINAMATH_CALUDE_train_length_l3211_321182

/-- Given a train that crosses a platform in a certain time and a signal pole in another time,
    this theorem calculates the length of the train. -/
theorem train_length
  (platform_length : ℝ)
  (platform_time : ℝ)
  (pole_time : ℝ)
  (h1 : platform_length = 400)
  (h2 : platform_time = 42)
  (h3 : pole_time = 18) :
  ∃ (train_length : ℝ),
    train_length = 300 ∧
    train_length * (1 / pole_time) * platform_time = train_length + platform_length :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3211_321182


namespace NUMINAMATH_CALUDE_unique_solution_l3211_321176

-- Define the type for digits from 2 to 9
def Digit := Fin 8

-- Define a function to map letters to digits
def LetterToDigit := Char → Digit

-- Define the letters used in the problem
def Letters : List Char := ['N', 'I', 'E', 'O', 'T', 'W', 'S', 'X']

-- Define the function to convert a word to a number
def wordToNumber (f : LetterToDigit) (word : List Char) : Nat :=
  word.foldl (fun acc c => 10 * acc + (f c).val.succ.succ) 0

-- State the theorem
theorem unique_solution :
  ∃! f : LetterToDigit,
    (∀ c₁ c₂ : Char, c₁ ∈ Letters → c₂ ∈ Letters → c₁ ≠ c₂ → f c₁ ≠ f c₂) ∧
    (wordToNumber f ['O', 'N', 'E']) + 
    (wordToNumber f ['T', 'W', 'O']) + 
    (wordToNumber f ['S', 'I', 'X']) = 
    (wordToNumber f ['N', 'I', 'N', 'E']) ∧
    (wordToNumber f ['N', 'I', 'N', 'E']) = 2526 :=
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3211_321176


namespace NUMINAMATH_CALUDE_fill_jug_completely_l3211_321120

/-- The capacity of the jug in milliliters -/
def jug_capacity : ℕ := 800

/-- The capacity of a small container in milliliters -/
def container_capacity : ℕ := 48

/-- The minimum number of small containers needed to fill the jug completely -/
def min_containers : ℕ := 17

theorem fill_jug_completely :
  min_containers = (jug_capacity + container_capacity - 1) / container_capacity ∧
  min_containers * container_capacity ≥ jug_capacity ∧
  (min_containers - 1) * container_capacity < jug_capacity := by
  sorry

end NUMINAMATH_CALUDE_fill_jug_completely_l3211_321120


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3211_321133

-- Define the start point of the line segment
def start_point : ℝ × ℝ := (1, 3)

-- Define the end point of the line segment
def end_point (x : ℝ) : ℝ × ℝ := (x, -4)

-- Define the length of the line segment
def segment_length : ℝ := 15

-- Theorem statement
theorem line_segment_endpoint (x : ℝ) : 
  x < 0 → 
  (end_point x).1 - (start_point.1) = -4 * Real.sqrt 11 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3211_321133


namespace NUMINAMATH_CALUDE_shopping_money_l3211_321185

theorem shopping_money (initial_amount : ℝ) : 
  (0.7 * initial_amount = 350) → initial_amount = 500 := by
  sorry

end NUMINAMATH_CALUDE_shopping_money_l3211_321185


namespace NUMINAMATH_CALUDE_pascal_triangle_30_rows_sum_l3211_321160

/-- The number of elements in the nth row of Pascal's Triangle -/
def pascalRowElements (n : ℕ) : ℕ := n + 1

/-- The sum of elements in the first n rows of Pascal's Triangle -/
def pascalTriangleSum (n : ℕ) : ℕ := 
  (n + 1) * (n + 2) / 2

theorem pascal_triangle_30_rows_sum :
  pascalTriangleSum 29 = 465 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_30_rows_sum_l3211_321160


namespace NUMINAMATH_CALUDE_garden_area_l3211_321161

/-- Given a square garden with perimeter 48 meters and a pond of area 20 square meters inside,
    the area of the garden not taken up by the pond is 124 square meters. -/
theorem garden_area (garden_perimeter : ℝ) (pond_area : ℝ) : 
  garden_perimeter = 48 → 
  pond_area = 20 → 
  (garden_perimeter / 4) ^ 2 - pond_area = 124 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l3211_321161


namespace NUMINAMATH_CALUDE_puppy_price_is_five_l3211_321111

/-- The price of a kitten in dollars -/
def kitten_price : ℕ := 6

/-- The number of kittens sold -/
def kittens_sold : ℕ := 2

/-- The total earnings from all pets sold in dollars -/
def total_earnings : ℕ := 17

/-- The price of the puppy in dollars -/
def puppy_price : ℕ := total_earnings - (kitten_price * kittens_sold)

theorem puppy_price_is_five : puppy_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_puppy_price_is_five_l3211_321111


namespace NUMINAMATH_CALUDE_smallest_number_l3211_321102

def binary_to_decimal (n : ℕ) : ℕ := n

def base_6_to_decimal (n : ℕ) : ℕ := n

def base_4_to_decimal (n : ℕ) : ℕ := n

def base_9_to_decimal (n : ℕ) : ℕ := n

theorem smallest_number :
  let a := binary_to_decimal 111111
  let b := base_6_to_decimal 210
  let c := base_4_to_decimal 1000
  let d := base_9_to_decimal 81
  a < b ∧ a < c ∧ a < d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3211_321102


namespace NUMINAMATH_CALUDE_tea_sale_prices_l3211_321115

structure Tea where
  name : String
  quantity : ℕ
  costPrice : ℚ
  profitPercentage : ℚ

def calculateSalePrice (tea : Tea) : ℚ :=
  tea.costPrice + tea.costPrice * (tea.profitPercentage / 100)

def teaA : Tea := ⟨"A", 120, 25, 45⟩
def teaB : Tea := ⟨"B", 60, 30, 35⟩
def teaC : Tea := ⟨"C", 40, 50, 25⟩
def teaD : Tea := ⟨"D", 30, 70, 20⟩

theorem tea_sale_prices :
  calculateSalePrice teaA = 36.25 ∧
  calculateSalePrice teaB = 40.5 ∧
  calculateSalePrice teaC = 62.5 ∧
  calculateSalePrice teaD = 84 := by
  sorry

end NUMINAMATH_CALUDE_tea_sale_prices_l3211_321115


namespace NUMINAMATH_CALUDE_equation_four_solutions_l3211_321183

theorem equation_four_solutions :
  ∃! (s : Finset ℝ), s.card = 4 ∧ 
  (∀ x : ℝ, x ∈ s ↔ (x - 2) * (x + 1) * (x + 4) * (x + 7) = 19) ∧
  (s = {(-5 + Real.sqrt 85) / 2, (-5 - Real.sqrt 85) / 2, 
        (-5 + Real.sqrt 5) / 2, (-5 - Real.sqrt 5) / 2}) :=
by sorry

end NUMINAMATH_CALUDE_equation_four_solutions_l3211_321183


namespace NUMINAMATH_CALUDE_farmer_cows_distribution_l3211_321123

theorem farmer_cows_distribution (total : ℕ) : 
  (total : ℚ) / 3 + (total : ℚ) / 6 + (total : ℚ) / 8 + 15 = total → total = 40 := by
  sorry

end NUMINAMATH_CALUDE_farmer_cows_distribution_l3211_321123


namespace NUMINAMATH_CALUDE_log_sum_lower_bound_l3211_321195

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_sum_lower_bound :
  (∀ a : ℝ, a > 1 → log 2 a + log a 2 ≥ 2) ∧
  (∃ m : ℝ, m < 2 ∧ ∀ a : ℝ, a > 1 → log 2 a + log a 2 ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_log_sum_lower_bound_l3211_321195


namespace NUMINAMATH_CALUDE_perpendicular_to_same_line_are_parallel_l3211_321100

-- Define the concept of a line in a plane
def Line (P : Type) := P → P → Prop

-- Define the concept of a plane
variable {P : Type}

-- Define the perpendicular relation between lines
def Perpendicular (l₁ l₂ : Line P) : Prop := sorry

-- Define the parallel relation between lines
def Parallel (l₁ l₂ : Line P) : Prop := sorry

-- State the theorem
theorem perpendicular_to_same_line_are_parallel 
  (l₁ l₂ l₃ : Line P) 
  (h₁ : Perpendicular l₁ l₃) 
  (h₂ : Perpendicular l₂ l₃) : 
  Parallel l₁ l₂ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_same_line_are_parallel_l3211_321100


namespace NUMINAMATH_CALUDE_one_refilling_cost_l3211_321193

/-- Given that Greyson spent 40 dollars on fuel this week and refilled 4 times,
    prove that the cost of one refilling is 10 dollars. -/
theorem one_refilling_cost (total_spent : ℝ) (num_refills : ℕ) 
  (h1 : total_spent = 40)
  (h2 : num_refills = 4) :
  total_spent / num_refills = 10 := by
  sorry

end NUMINAMATH_CALUDE_one_refilling_cost_l3211_321193


namespace NUMINAMATH_CALUDE_seven_children_ages_l3211_321144

theorem seven_children_ages (n : ℕ) : 
  (∃ (a : ℕ), a + (a + 1) + (a + 2) = 42) →
  ((n + 4) + (n + 5) + (n + 6) = 54) :=
by sorry

end NUMINAMATH_CALUDE_seven_children_ages_l3211_321144


namespace NUMINAMATH_CALUDE_parity_of_f_l3211_321149

/-- A function that is not always zero -/
def NonZeroFunction (f : ℝ → ℝ) : Prop :=
  ∃ x, f x ≠ 0

/-- Definition of an odd function -/
def OddFunction (F : ℝ → ℝ) : Prop :=
  ∀ x, F (-x) = -F x

/-- Definition of an even function -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The main theorem -/
theorem parity_of_f (f : ℝ → ℝ) (h_nonzero : NonZeroFunction f) :
    let F := fun x => if x ≠ 0 then (x^3 - 2*x) * f x else 0
    OddFunction F → EvenFunction f := by
  sorry

end NUMINAMATH_CALUDE_parity_of_f_l3211_321149


namespace NUMINAMATH_CALUDE_grant_baseball_gear_sale_total_l3211_321162

def baseball_cards_price : ℝ := 25
def baseball_bat_price : ℝ := 10
def baseball_glove_original_price : ℝ := 30
def baseball_glove_discount : ℝ := 0.20
def baseball_cleats_original_price : ℝ := 10
def usd_to_eur_rate : ℝ := 0.85
def baseball_cleats_discount : ℝ := 0.15

theorem grant_baseball_gear_sale_total :
  let baseball_glove_sale_price := baseball_glove_original_price * (1 - baseball_glove_discount)
  let cleats_eur_price := baseball_cleats_original_price * usd_to_eur_rate
  let cleats_discounted_price := baseball_cleats_original_price * (1 - baseball_cleats_discount)
  baseball_cards_price + baseball_bat_price + baseball_glove_sale_price + cleats_eur_price + cleats_discounted_price = 76 := by
  sorry

end NUMINAMATH_CALUDE_grant_baseball_gear_sale_total_l3211_321162
