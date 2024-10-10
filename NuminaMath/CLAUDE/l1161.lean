import Mathlib

namespace property_length_proof_l1161_116121

/-- Given a rectangular property and a garden within it, prove the length of the property. -/
theorem property_length_proof (property_width : ℝ) (garden_area : ℝ) : 
  property_width = 1000 →
  garden_area = 28125 →
  ∃ (property_length : ℝ),
    property_length = 2250 ∧
    garden_area = (property_width / 8) * (property_length / 10) :=
by sorry

end property_length_proof_l1161_116121


namespace smallest_valid_n_l1161_116184

def is_valid_pair (m n x : ℕ+) : Prop :=
  m = 60 ∧ 
  Nat.gcd m n = x + 5 ∧ 
  Nat.lcm m n = x * (x + 5)

theorem smallest_valid_n : 
  ∃ (x : ℕ+), is_valid_pair 60 100 x ∧ 
  ∀ (y : ℕ+) (n : ℕ+), y < x → ¬ is_valid_pair 60 n y :=
sorry

end smallest_valid_n_l1161_116184


namespace correct_average_calculation_l1161_116114

/-- Given a number of tables, women, and men, calculate the average number of customers per table. -/
def averageCustomersPerTable (tables : Float) (women : Float) (men : Float) : Float :=
  (women + men) / tables

/-- Theorem stating that for the given values, the average number of customers per table is correct. -/
theorem correct_average_calculation :
  averageCustomersPerTable 9.0 7.0 3.0 = (7.0 + 3.0) / 9.0 := by
  sorry

end correct_average_calculation_l1161_116114


namespace m_range_l1161_116105

theorem m_range (m : ℝ) (h1 : m < 0) (h2 : ∀ x : ℝ, x^2 + m*x + 1 > 0) : -2 < m ∧ m < 0 := by
  sorry

end m_range_l1161_116105


namespace triangle_inequality_l1161_116108

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c ∧
  (Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) = Real.sqrt a + Real.sqrt b + Real.sqrt c ↔ a = b ∧ b = c) :=
by sorry

end triangle_inequality_l1161_116108


namespace unique_solution_quadratic_inequality_l1161_116162

theorem unique_solution_quadratic_inequality (b : ℝ) : 
  (∃! x : ℝ, |x^2 + 2*b*x + 2*b| ≤ 1) ↔ b = 1 := by
  sorry

end unique_solution_quadratic_inequality_l1161_116162


namespace binomial_square_difference_specific_case_l1161_116125

theorem binomial_square_difference (a b : ℕ) : (a + b)^2 - (a^2 + b^2) = 2 * a * b := by sorry

theorem specific_case : (45 + 15)^2 - (45^2 + 15^2) = 1350 := by sorry

end binomial_square_difference_specific_case_l1161_116125


namespace bears_captured_pieces_l1161_116117

theorem bears_captured_pieces (H B F : ℕ) : 
  (64 : ℕ) = H + B + F →
  H = B / 2 →
  H = F / 5 →
  (0 : ℕ) = 16 - B :=
by sorry

end bears_captured_pieces_l1161_116117


namespace marys_friends_marys_friends_correct_l1161_116148

theorem marys_friends (total_stickers : ℕ) (stickers_per_friend : ℕ) (stickers_per_non_friend : ℕ) 
  (stickers_left : ℕ) (total_students : ℕ) : ℕ :=
  let num_friends := (total_stickers - stickers_left - 2 * (total_students - 1)) / 
    (stickers_per_friend - stickers_per_non_friend)
  num_friends

theorem marys_friends_correct : marys_friends 50 4 2 8 17 = 5 := by
  sorry

end marys_friends_marys_friends_correct_l1161_116148


namespace sam_eats_280_apples_in_week_l1161_116109

/-- Calculates the number of apples Sam eats in a week -/
def apples_eaten_in_week (apples_per_sandwich : ℕ) (sandwiches_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  apples_per_sandwich * sandwiches_per_day * days_in_week

/-- Proves that Sam eats 280 apples in a week -/
theorem sam_eats_280_apples_in_week :
  apples_eaten_in_week 4 10 7 = 280 := by
  sorry

end sam_eats_280_apples_in_week_l1161_116109


namespace factorial_equation_solution_l1161_116164

theorem factorial_equation_solution : 
  ∃ (n : ℕ), n > 0 ∧ (Nat.factorial (n + 1) + Nat.factorial (n + 3) = Nat.factorial n * 1320) ∧ n = 10 := by
  sorry

end factorial_equation_solution_l1161_116164


namespace triangle_sine_inequality_l1161_116144

/-- Given a triangle ABC, prove that (sin A + sin B + sin C) / (sin A * sin B * sin C) ≥ 4,
    with equality if and only if the triangle is equilateral -/
theorem triangle_sine_inequality (A B C : ℝ) (h_triangle : A + B + C = π) (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) :
  (Real.sin A + Real.sin B + Real.sin C) / (Real.sin A * Real.sin B * Real.sin C) ≥ 4 ∧
  ((Real.sin A + Real.sin B + Real.sin C) / (Real.sin A * Real.sin B * Real.sin C) = 4 ↔ A = B ∧ B = C) :=
by sorry

end triangle_sine_inequality_l1161_116144


namespace parallelogram_side_length_l1161_116192

theorem parallelogram_side_length 
  (s : ℝ) 
  (area : ℝ) 
  (h1 : area = 27 * Real.sqrt 3) 
  (h2 : area = 3 * s^2 * (1/2)) : 
  s = 3 * Real.sqrt 3 := by
sorry

end parallelogram_side_length_l1161_116192


namespace triangle_area_l1161_116113

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) : 
  (1/2) * a * b = 54 := by
  sorry

end triangle_area_l1161_116113


namespace alexis_shoe_cost_l1161_116179

/-- Given Alexis' shopping scenario, prove the cost of shoes --/
theorem alexis_shoe_cost (budget : ℕ) (shirt_cost pants_cost coat_cost socks_cost belt_cost money_left : ℕ) 
  (h1 : budget = 200)
  (h2 : shirt_cost = 30)
  (h3 : pants_cost = 46)
  (h4 : coat_cost = 38)
  (h5 : socks_cost = 11)
  (h6 : belt_cost = 18)
  (h7 : money_left = 16) :
  budget - (shirt_cost + pants_cost + coat_cost + socks_cost + belt_cost + money_left) = 41 := by
  sorry

end alexis_shoe_cost_l1161_116179


namespace expand_and_simplify_l1161_116123

theorem expand_and_simplify (x : ℝ) : -2 * (4 * x^3 - 5 * x^2 + 3 * x - 7) = -8 * x^3 + 10 * x^2 - 6 * x + 14 := by
  sorry

end expand_and_simplify_l1161_116123


namespace point_relationship_l1161_116169

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

theorem point_relationship (m n : ℝ) : 
  let l : Line := { slope := -2, intercept := 1 }
  let A : Point := { x := -1, y := m }
  let B : Point := { x := 3, y := n }
  A.liesOn l ∧ B.liesOn l → m > n := by
  sorry

end point_relationship_l1161_116169


namespace product_of_22nd_and_23rd_multiples_l1161_116198

/-- The sequence of multiples of 3 greater than 0 and less than 100 -/
def multiples_of_3 : List Nat :=
  (List.range 33).map (fun n => (n + 1) * 3)

/-- The 22nd element in the sequence -/
def element_22 : Nat := multiples_of_3[21]

/-- The 23rd element in the sequence -/
def element_23 : Nat := multiples_of_3[22]

theorem product_of_22nd_and_23rd_multiples :
  element_22 * element_23 = 4554 :=
by sorry

end product_of_22nd_and_23rd_multiples_l1161_116198


namespace equation_solution_l1161_116126

theorem equation_solution (x : ℝ) : 
  1 / (x + 5) + 1 / (x - 5) = 1 / (x - 5) → x = 1/2 := by
  sorry

end equation_solution_l1161_116126


namespace a_range_l1161_116120

-- Define the line equation
def line_equation (x y a : ℝ) : Prop := 2 * x - 3 * y + a = 0

-- Define the condition for points being on opposite sides of the line
def opposite_sides (a : ℝ) : Prop :=
  (2 * 2 - 3 * 1 + a) * (2 * 4 - 3 * 3 + a) < 0

-- Theorem statement
theorem a_range (a : ℝ) :
  (∀ x y, line_equation x y a) →
  opposite_sides a →
  -1 < a ∧ a < 1 :=
sorry

end a_range_l1161_116120


namespace f_2019_value_l1161_116199

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 1/4 ∧ ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

/-- The main theorem stating that f(2019) = -1/2 for any function satisfying the conditions -/
theorem f_2019_value (f : ℝ → ℝ) (hf : special_function f) : f 2019 = -1/2 := by
  sorry

end f_2019_value_l1161_116199


namespace inscribed_circle_theorem_l1161_116181

theorem inscribed_circle_theorem (PQ PR : ℝ) (h_PQ : PQ = 6) (h_PR : PR = 8) :
  let QR := Real.sqrt (PQ^2 + PR^2)
  let s := (PQ + PR + QR) / 2
  let A := PQ * PR / 2
  let r := A / s
  let x := QR - 2*r
  x = 6 := by sorry

end inscribed_circle_theorem_l1161_116181


namespace system_solution_expression_simplification_l1161_116145

-- Part 1: System of equations
theorem system_solution :
  ∃ (x y : ℝ), 2 * x + y = 3 ∧ 3 * x + y = 5 ∧ x = 2 ∧ y = -1 := by sorry

-- Part 2: Expression calculation
theorem expression_simplification (a : ℝ) (h : a ≠ 1) :
  (a^2 / (a^2 - 2*a + 1)) * ((a - 1) / a) - (1 / (a - 1)) = 1 := by sorry

end system_solution_expression_simplification_l1161_116145


namespace winning_lines_8_cube_l1161_116195

/-- The number of straight lines containing 8 points in a 3D cubic grid --/
def winning_lines (n : ℕ) : ℕ :=
  ((n + 2)^3 - n^3) / 2

/-- Theorem: In an 8×8×8 cubic grid, the number of straight lines containing 8 points is 244 --/
theorem winning_lines_8_cube : winning_lines 8 = 244 := by
  sorry

end winning_lines_8_cube_l1161_116195


namespace solution_set_f_leq_x_plus_1_range_f_geq_inequality_l1161_116147

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 1

-- Theorem for the first part of the problem
theorem solution_set_f_leq_x_plus_1 :
  {x : ℝ | f x ≤ x + 1} = Set.Icc 0 2 :=
sorry

-- Theorem for the second part of the problem
theorem range_f_geq_inequality (a : ℝ) (ha : a ≠ 0) :
  {x : ℝ | ∀ a : ℝ, a ≠ 0 → f x ≥ (|a + 1| - |2*a - 1|) / |a|} = 
    Set.Iic (-2) ∪ Set.Ici 2 :=
sorry

end solution_set_f_leq_x_plus_1_range_f_geq_inequality_l1161_116147


namespace complex_equation_sum_l1161_116107

theorem complex_equation_sum (a b : ℝ) : 
  (a + 2 * Complex.I) / Complex.I = b + Complex.I → a + b = 1 := by
  sorry

end complex_equation_sum_l1161_116107


namespace max_d_value_l1161_116135

def is_prime (n : ℕ) : Prop := sorry

def max_list (l : List ℕ) : ℕ := sorry

def min_list (l : List ℕ) : ℕ := sorry

theorem max_d_value (a b c : ℕ) : 
  is_prime a ∧ is_prime b ∧ is_prime c ∧
  is_prime (a + b - c) ∧ is_prime (a + c - b) ∧ is_prime (b + c - a) ∧ is_prime (a + b + c) ∧
  (a + b = 800 ∨ a + c = 800 ∨ b + c = 800) ∧
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
  a ≠ (a + b - c) ∧ a ≠ (a + c - b) ∧ a ≠ (b + c - a) ∧ a ≠ (a + b + c) ∧
  b ≠ (a + b - c) ∧ b ≠ (a + c - b) ∧ b ≠ (b + c - a) ∧ b ≠ (a + b + c) ∧
  c ≠ (a + b - c) ∧ c ≠ (a + c - b) ∧ c ≠ (b + c - a) ∧ c ≠ (a + b + c) ∧
  (a + b - c) ≠ (a + c - b) ∧ (a + b - c) ≠ (b + c - a) ∧ (a + b - c) ≠ (a + b + c) ∧
  (a + c - b) ≠ (b + c - a) ∧ (a + c - b) ≠ (a + b + c) ∧
  (b + c - a) ≠ (a + b + c) →
  max_list [a, b, c, a + b - c, a + c - b, b + c - a, a + b + c] - 
  min_list [a, b, c, a + b - c, a + c - b, b + c - a, a + b + c] ≤ 
  max_list [3, 797, c, 800 - c, 3 + c, 797 + c, 800 + c] - 
  min_list [3, 797, c, 800 - c, 3 + c, 797 + c, 800 + c] := by
sorry

end max_d_value_l1161_116135


namespace min_value_fraction_sum_l1161_116154

theorem min_value_fraction_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 6) :
  (9 / a + 4 / b + 25 / c) ≥ 50 / 3 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 6 ∧ (9 / a₀ + 4 / b₀ + 25 / c₀ = 50 / 3) :=
by sorry

end min_value_fraction_sum_l1161_116154


namespace vector_collinearity_l1161_116155

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem vector_collinearity :
  let a : ℝ × ℝ := (3, 6)
  let b : ℝ × ℝ := (x, 8)
  collinear a b → x = 4 :=
by sorry

end vector_collinearity_l1161_116155


namespace table_rotation_l1161_116173

theorem table_rotation (table_width : ℝ) (table_length : ℝ) : 
  table_width = 8 ∧ table_length = 12 →
  ∃ (S : ℕ), (S : ℝ) ≥ (table_width^2 + table_length^2).sqrt ∧
  ∀ (T : ℕ), (T : ℝ) ≥ (table_width^2 + table_length^2).sqrt → S ≤ T →
  S = 15 :=
by sorry

end table_rotation_l1161_116173


namespace students_taking_one_subject_l1161_116171

theorem students_taking_one_subject (both : ℕ) (science : ℕ) (only_history : ℕ) 
  (h1 : both = 15)
  (h2 : science = 30)
  (h3 : only_history = 18) :
  science - both + only_history = 33 := by
sorry

end students_taking_one_subject_l1161_116171


namespace fractional_equation_positive_root_l1161_116178

theorem fractional_equation_positive_root (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (1 - x) / (x - 2) = a / (2 - x) - 2) → a = 1 := by
  sorry

end fractional_equation_positive_root_l1161_116178


namespace beth_peas_cans_l1161_116150

/-- The number of cans of corn Beth bought -/
def corn_cans : ℕ := 10

/-- The number of cans of peas Beth bought -/
def peas_cans : ℕ := 2 * corn_cans + 15

theorem beth_peas_cans : peas_cans = 35 := by
  sorry

end beth_peas_cans_l1161_116150


namespace larry_jogging_days_l1161_116128

theorem larry_jogging_days (daily_jog_time : ℕ) (second_week_days : ℕ) (total_time : ℕ) : 
  daily_jog_time = 30 →
  second_week_days = 5 →
  total_time = 4 * 60 →
  (total_time - second_week_days * daily_jog_time) / daily_jog_time = 3 :=
by sorry

end larry_jogging_days_l1161_116128


namespace middle_number_is_five_l1161_116115

/-- Represents a triple of positive integers in increasing order -/
structure IncreasingTriple where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  h1 : a < b
  h2 : b < c

/-- The set of all valid triples according to the problem conditions -/
def ValidTriples : Set IncreasingTriple :=
  { t : IncreasingTriple | t.a + t.b + t.c = 16 }

/-- A triple is ambiguous if there exists another valid triple with the same middle number -/
def IsAmbiguous (t : IncreasingTriple) : Prop :=
  ∃ t' : IncreasingTriple, t' ∈ ValidTriples ∧ t' ≠ t ∧ t'.b = t.b

theorem middle_number_is_five :
  ∀ t ∈ ValidTriples, IsAmbiguous t → t.b = 5 := by sorry

end middle_number_is_five_l1161_116115


namespace cookie_distribution_l1161_116175

/-- Represents the number of cookie boxes in Sonny's distribution problem -/
structure CookieBoxes where
  total : ℕ
  tobrother : ℕ
  tocousin : ℕ
  kept : ℕ
  tosister : ℕ

/-- Theorem stating the relationship between the number of cookie boxes -/
theorem cookie_distribution (c : CookieBoxes) 
  (h1 : c.total = 45)
  (h2 : c.tobrother = 12)
  (h3 : c.tocousin = 7)
  (h4 : c.kept = 17) :
  c.tosister = c.total - (c.tobrother + c.tocousin + c.kept) :=
by sorry

end cookie_distribution_l1161_116175


namespace least_sum_exponents_1540_l1161_116118

/-- The function that computes the least sum of exponents for a given number -/
def leastSumOfExponents (n : ℕ) : ℕ := sorry

/-- The theorem stating that the least sum of exponents for 1540 is 21 -/
theorem least_sum_exponents_1540 : leastSumOfExponents 1540 = 21 := by sorry

end least_sum_exponents_1540_l1161_116118


namespace cacl₂_production_l1161_116100

/-- Represents the chemical reaction: CaCO₃ + 2HCl → CaCl₂ + CO₂ + H₂O -/
structure ChemicalReaction where
  cacO₃ : ℚ  -- moles of CaCO₃
  hcl : ℚ    -- moles of HCl
  cacl₂ : ℚ  -- moles of CaCl₂ produced

/-- The stoichiometric ratio of the reaction -/
def stoichiometricRatio : ℚ := 2

/-- Calculates the amount of CaCl₂ produced based on the limiting reactant -/
def calcCaCl₂Produced (reaction : ChemicalReaction) : ℚ :=
  min reaction.cacO₃ (reaction.hcl / stoichiometricRatio)

/-- Theorem stating that 2 moles of CaCl₂ are produced when 4 moles of HCl react with 2 moles of CaCO₃ -/
theorem cacl₂_production (reaction : ChemicalReaction) 
  (h1 : reaction.cacO₃ = 2)
  (h2 : reaction.hcl = 4) :
  calcCaCl₂Produced reaction = 2 := by
  sorry

end cacl₂_production_l1161_116100


namespace complex_equation_sum_l1161_116132

theorem complex_equation_sum (a b : ℝ) : 
  (a : ℂ) + b * Complex.I = (11 - 7 * Complex.I) / (1 - 2 * Complex.I) → a + b = 8 := by
  sorry

end complex_equation_sum_l1161_116132


namespace range_of_a_l1161_116142

-- Define propositions P and Q
def P (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def Q (a : ℝ) : Prop := ∃ x y : ℝ, x^2 / a + y^2 / (a - 3) = 1 ∧ a * (a - 3) < 0

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  ((P a ∨ Q a) ∧ ¬(P a ∧ Q a)) → (a = 0 ∨ (3 ≤ a ∧ a < 4)) :=
sorry

end range_of_a_l1161_116142


namespace circle_condition_l1161_116111

theorem circle_condition (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*m*x - 2*m*y + 2*m^2 + m - 1 = 0 → 
    ∃ r : ℝ, r > 0 ∧ ∃ a b : ℝ, (x - a)^2 + (y - b)^2 = r^2) → 
  m < 1 := by
sorry

end circle_condition_l1161_116111


namespace jills_salary_solution_l1161_116101

def jills_salary_problem (net_salary : ℝ) : Prop :=
  let discretionary_income := net_salary / 5
  let vacation_fund := 0.30 * discretionary_income
  let savings := 0.20 * discretionary_income
  let eating_out := 0.35 * discretionary_income
  let fitness_classes := 0.05 * discretionary_income
  let gifts_and_charity := 99
  vacation_fund + savings + eating_out + fitness_classes + gifts_and_charity = discretionary_income ∧
  net_salary = 4950

theorem jills_salary_solution :
  ∃ (net_salary : ℝ), jills_salary_problem net_salary :=
sorry

end jills_salary_solution_l1161_116101


namespace queens_free_subgrid_l1161_116186

/-- Represents a chessboard with queens -/
structure Chessboard :=
  (size : Nat)
  (queens : Nat)

/-- Theorem: On an 8x8 chessboard with 12 queens, there always exist four rows and four columns
    such that none of the 16 cells at their intersections contain a queen -/
theorem queens_free_subgrid (board : Chessboard) 
  (h1 : board.size = 8) 
  (h2 : board.queens = 12) : 
  ∃ (rows columns : Finset Nat), 
    rows.card = 4 ∧ 
    columns.card = 4 ∧ 
    (∀ r ∈ rows, ∀ c ∈ columns, 
      ¬∃ (queen : Nat × Nat), queen.1 = r ∧ queen.2 = c) :=
sorry

end queens_free_subgrid_l1161_116186


namespace pierced_square_theorem_l1161_116188

/-- Represents a square pierced at n points and cut into triangles --/
structure PiercedSquare where
  n : ℕ  -- number of pierced points
  no_collinear_triples : True  -- represents the condition that no three points are collinear
  no_internal_piercings : True  -- represents the condition that there are no piercings inside triangles

/-- Calculates the number of triangles formed in a pierced square --/
def num_triangles (ps : PiercedSquare) : ℕ :=
  2 * (ps.n + 1)

/-- Calculates the number of cuts made in a pierced square --/
def num_cuts (ps : PiercedSquare) : ℕ :=
  (3 * num_triangles ps - 4) / 2

/-- Theorem stating the relationship between pierced points, triangles, and cuts --/
theorem pierced_square_theorem (ps : PiercedSquare) :
  (num_triangles ps = 2 * (ps.n + 1)) ∧
  (num_cuts ps = (3 * num_triangles ps - 4) / 2) := by
  sorry

end pierced_square_theorem_l1161_116188


namespace equation_proof_l1161_116110

theorem equation_proof (a b : ℚ) (h : 3 * a = 2 * b) : (a + b) / b = 5 / 3 := by
  sorry

end equation_proof_l1161_116110


namespace shaded_area_square_with_circles_l1161_116133

/-- The shaded area of a square with six inscribed circles --/
theorem shaded_area_square_with_circles (square_side : ℝ) (circle_diameter : ℝ) :
  square_side = 24 →
  circle_diameter = 8 →
  let square_area := square_side ^ 2
  let circle_area := π * (circle_diameter / 2) ^ 2
  let total_circles_area := 6 * circle_area
  let shaded_area := square_area - total_circles_area
  shaded_area = 576 - 96 * π := by
  sorry

#check shaded_area_square_with_circles

end shaded_area_square_with_circles_l1161_116133


namespace pink_to_orange_ratio_l1161_116103

theorem pink_to_orange_ratio :
  -- Define the total number of balls
  let total_balls : ℕ := 50
  -- Define the number of red balls
  let red_balls : ℕ := 20
  -- Define the number of blue balls
  let blue_balls : ℕ := 10
  -- Define the number of orange balls
  let orange_balls : ℕ := 5
  -- Define the number of pink balls
  let pink_balls : ℕ := 15
  -- Ensure that the sum of all balls equals the total
  red_balls + blue_balls + orange_balls + pink_balls = total_balls →
  -- Prove that the ratio of pink to orange balls is 3:1
  (pink_balls : ℚ) / (orange_balls : ℚ) = 3 / 1 :=
by
  sorry

end pink_to_orange_ratio_l1161_116103


namespace g_of_3_l1161_116127

def g (x : ℝ) : ℝ := 5 * x^3 - 7 * x^2 + 3 * x - 2

theorem g_of_3 : g 3 = 79 := by
  sorry

end g_of_3_l1161_116127


namespace divide_algebraic_expression_l1161_116112

theorem divide_algebraic_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  12 * a^4 * b^3 * c / (-4 * a^3 * b^2) = -3 * a * b * c :=
by sorry

end divide_algebraic_expression_l1161_116112


namespace function_monotonicity_l1161_116116

theorem function_monotonicity (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) :
  (∀ x, (x^2 - 3*x + 2) * (deriv (deriv f) x) ≤ 0) →
  (∀ x ∈ Set.Icc 1 2, f 1 ≤ f x ∧ f x ≤ f 2) :=
by sorry

end function_monotonicity_l1161_116116


namespace solution_set_inequalities_l1161_116190

theorem solution_set_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by sorry

end solution_set_inequalities_l1161_116190


namespace triangle_angle_A_l1161_116174

theorem triangle_angle_A (a b : ℝ) (A B : ℝ) :
  a = Real.sqrt 2 →
  b = 2 →
  Real.sin B + Real.cos B = Real.sqrt 2 →
  0 < A →
  A < π / 4 →
  0 < B →
  B < π →
  Real.sin A = a / b * Real.sin B →
  A = π / 6 :=
by sorry

end triangle_angle_A_l1161_116174


namespace floor_sqrt_80_l1161_116193

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end floor_sqrt_80_l1161_116193


namespace sheep_count_l1161_116106

/-- Represents the number of animals on a boat and their fate after capsizing -/
structure BoatAnimals where
  sheep : ℕ
  cows : ℕ
  dogs : ℕ
  drownedSheep : ℕ
  drownedCows : ℕ
  survivedAnimals : ℕ

/-- Theorem stating the number of sheep on the boat given the conditions -/
theorem sheep_count (b : BoatAnimals) : b.sheep = 20 :=
  by
  have h1 : b.cows = 10 := sorry
  have h2 : b.dogs = 14 := sorry
  have h3 : b.drownedSheep = 3 := sorry
  have h4 : b.drownedCows = 2 * b.drownedSheep := sorry
  have h5 : b.survivedAnimals = 35 := sorry
  have h6 : b.survivedAnimals = b.sheep - b.drownedSheep + b.cows - b.drownedCows + b.dogs := sorry
  sorry

#check sheep_count

end sheep_count_l1161_116106


namespace newspaper_distribution_l1161_116177

theorem newspaper_distribution (F : ℚ) : 
  200 * F + 0.6 * (200 - 200 * F) = 200 - 48 → F = 2/5 := by
  sorry

end newspaper_distribution_l1161_116177


namespace zero_discriminant_implies_ratio_l1161_116176

/-- Given a quadratic equation 3ax^2 + 6bx + 2c = 0 with zero discriminant,
    prove that b^2 = (2/3)ac -/
theorem zero_discriminant_implies_ratio (a b c : ℝ) :
  (6 * b)^2 - 4 * (3 * a) * (2 * c) = 0 →
  b^2 = (2/3) * a * c := by
  sorry

end zero_discriminant_implies_ratio_l1161_116176


namespace smallest_n_99n_all_threes_l1161_116172

def all_threes (n : ℕ) : Prop :=
  ∀ d, d ∈ (n.digits 10) → d = 3

theorem smallest_n_99n_all_threes :
  ∃ (N : ℕ), (N = 3367 ∧ all_threes (99 * N) ∧ ∀ n < N, ¬ all_threes (99 * n)) :=
sorry

end smallest_n_99n_all_threes_l1161_116172


namespace general_form_equation_l1161_116139

theorem general_form_equation (x : ℝ) : 
  (x - 1) * (x - 2) = 4 ↔ x^2 - 3*x - 2 = 0 := by sorry

end general_form_equation_l1161_116139


namespace may_blue_yarns_l1161_116187

/-- The number of scarves May can knit using one yarn -/
def scarves_per_yarn : ℕ := 3

/-- The number of red yarns May bought -/
def red_yarns : ℕ := 2

/-- The number of yellow yarns May bought -/
def yellow_yarns : ℕ := 4

/-- The total number of scarves May will be able to make -/
def total_scarves : ℕ := 36

/-- The number of blue yarns May bought -/
def blue_yarns : ℕ := 6

theorem may_blue_yarns : 
  scarves_per_yarn * (red_yarns + yellow_yarns + blue_yarns) = total_scarves :=
by sorry

end may_blue_yarns_l1161_116187


namespace quadratic_inequality_solution_l1161_116191

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2 := by
  sorry

end quadratic_inequality_solution_l1161_116191


namespace max_salary_for_given_constraints_l1161_116185

/-- Represents a baseball team with salary constraints -/
structure BaseballTeam where
  num_players : ℕ
  min_salary : ℕ
  max_total_salary : ℕ

/-- Calculates the maximum possible salary for a single player -/
def max_single_player_salary (team : BaseballTeam) : ℕ :=
  team.max_total_salary - (team.num_players - 1) * team.min_salary

/-- Theorem stating the maximum possible salary for a single player
    in a team with given constraints -/
theorem max_salary_for_given_constraints :
  let team : BaseballTeam := {
    num_players := 25,
    min_salary := 20000,
    max_total_salary := 800000
  }
  max_single_player_salary team = 320000 := by
  sorry

#eval max_single_player_salary {
  num_players := 25,
  min_salary := 20000,
  max_total_salary := 800000
}

end max_salary_for_given_constraints_l1161_116185


namespace time_against_walkway_l1161_116134

/-- The time it takes to walk against a moving walkway given specific conditions -/
theorem time_against_walkway 
  (walkway_length : ℝ) 
  (time_with_walkway : ℝ) 
  (time_without_movement : ℝ) 
  (h1 : walkway_length = 100) 
  (h2 : time_with_walkway = 25) 
  (h3 : time_without_movement = 42.857142857142854) :
  let person_speed := walkway_length / time_without_movement
  let walkway_speed := walkway_length / time_with_walkway - person_speed
  walkway_length / (person_speed - walkway_speed) = 150 := by
  sorry

end time_against_walkway_l1161_116134


namespace estimate_fish_population_l1161_116166

/-- Estimate the number of fish in a pond using the mark and recapture method. -/
theorem estimate_fish_population (initial_marked : ℕ) (recapture_total : ℕ) (recapture_marked : ℕ) 
  (h1 : initial_marked = 20)
  (h2 : recapture_total = 40)
  (h3 : recapture_marked = 2) :
  (initial_marked * recapture_total) / recapture_marked = 400 := by
  sorry

#check estimate_fish_population

end estimate_fish_population_l1161_116166


namespace smallest_p_value_l1161_116189

theorem smallest_p_value (p q : ℕ+) 
  (h1 : (5 : ℚ) / 8 < p / q)
  (h2 : p / q < (7 : ℚ) / 8)
  (h3 : p + q = 2005) : 
  p.val ≥ 772 ∧ (∀ m : ℕ+, m < p → ¬((5 : ℚ) / 8 < m / (2005 - m) ∧ m / (2005 - m) < (7 : ℚ) / 8)) := by
  sorry

end smallest_p_value_l1161_116189


namespace cafeteria_pies_l1161_116138

/-- Given a cafeteria with initial apples, apples handed out, and apples required per pie,
    calculates the number of pies that can be made with the remaining apples. -/
def calculate_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

/-- Proves that given 86 initial apples, after handing out 30 apples,
    and using 8 apples per pie, the number of pies that can be made is 7. -/
theorem cafeteria_pies :
  calculate_pies 86 30 8 = 7 := by
  sorry

end cafeteria_pies_l1161_116138


namespace inequality_proof_l1161_116197

theorem inequality_proof (a b c : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) 
  (h4 : a * b + b * c + c * a = 1/3) : 
  1 / (a^2 - b*c + 1) + 1 / (b^2 - c*a + 1) + 1 / (c^2 - a*b + 1) ≤ 3 := by
sorry

end inequality_proof_l1161_116197


namespace circus_ticket_sales_l1161_116140

/-- Calculates the total number of tickets sold at a circus given the prices, revenue, and number of lower seat tickets sold. -/
def total_tickets (lower_price upper_price : ℕ) (total_revenue : ℕ) (lower_tickets : ℕ) : ℕ :=
  lower_tickets + (total_revenue - lower_price * lower_tickets) / upper_price

/-- Theorem stating that given the specific conditions of the circus problem, the total number of tickets sold is 80. -/
theorem circus_ticket_sales :
  total_tickets 30 20 2100 50 = 80 := by
  sorry

end circus_ticket_sales_l1161_116140


namespace line_parabola_single_intersection_l1161_116196

theorem line_parabola_single_intersection (a : ℝ) :
  (∃! x : ℝ, a * x - 6 = x^2 + 4*x + 3) ↔ (a = -2 ∨ a = 10) :=
sorry

end line_parabola_single_intersection_l1161_116196


namespace chips_on_line_after_moves_l1161_116136

/-- Represents a configuration of chips on a plane -/
structure ChipConfiguration where
  num_chips : ℕ
  num_lines : ℕ

/-- Represents a move that can be applied to a chip configuration -/
def apply_move (config : ChipConfiguration) : ChipConfiguration :=
  { num_chips := config.num_chips,
    num_lines := min config.num_lines (2 ^ (config.num_lines - 1)) }

/-- Represents the initial configuration of chips on a convex 2000-gon -/
def initial_config : ChipConfiguration :=
  { num_chips := 2000,
    num_lines := 2000 }

/-- Applies n moves to the initial configuration -/
def apply_n_moves (n : ℕ) : ChipConfiguration :=
  (List.range n).foldl (λ config _ => apply_move config) initial_config

theorem chips_on_line_after_moves :
  (∀ n : ℕ, n ≤ 9 → (apply_n_moves n).num_lines > 1) ∧
  ∃ m : ℕ, m = 10 ∧ (apply_n_moves m).num_lines = 1 :=
sorry

end chips_on_line_after_moves_l1161_116136


namespace problem_surface_area_l1161_116158

/-- Represents a solid block formed by unit cubes -/
structure SolidBlock where
  base_width : ℕ
  base_length : ℕ
  base_height : ℕ
  top_cubes : ℕ

/-- Calculates the surface area of a SolidBlock -/
def surface_area (block : SolidBlock) : ℕ :=
  sorry

/-- The specific solid block described in the problem -/
def problem_block : SolidBlock :=
  { base_width := 3
  , base_length := 2
  , base_height := 2
  , top_cubes := 2 }

theorem problem_surface_area : surface_area problem_block = 42 := by
  sorry

end problem_surface_area_l1161_116158


namespace pentagonal_prism_lateral_angle_l1161_116141

/-- A pentagonal prism is a three-dimensional geometric shape with a pentagonal base
    and rectangular lateral faces. -/
structure PentagonalPrism where
  base : Pentagon
  height : ℝ
  height_pos : height > 0

/-- The angle between a lateral edge and the base of a pentagonal prism. -/
def lateral_angle (p : PentagonalPrism) : ℝ := sorry

/-- Theorem: The angle between any lateral edge and the base of a pentagonal prism is 90°. -/
theorem pentagonal_prism_lateral_angle (p : PentagonalPrism) :
  lateral_angle p = Real.pi / 2 := by sorry

end pentagonal_prism_lateral_angle_l1161_116141


namespace factorization_problem1_l1161_116160

theorem factorization_problem1 (a m : ℝ) : 2 * a * m^2 - 8 * a = 2 * a * (m + 2) * (m - 2) := by
  sorry

end factorization_problem1_l1161_116160


namespace isosceles_not_equilateral_l1161_116194

-- Define an isosceles triangle
def IsIsosceles (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ (a > 0 ∧ b > 0 ∧ c > 0)

-- Define an equilateral triangle
def IsEquilateral (a b c : ℝ) : Prop :=
  a = b ∧ b = c ∧ a > 0

-- Theorem: There exists an isosceles triangle that is not equilateral
theorem isosceles_not_equilateral : ∃ a b c : ℝ, IsIsosceles a b c ∧ ¬IsEquilateral a b c := by
  sorry


end isosceles_not_equilateral_l1161_116194


namespace new_supervisor_salary_l1161_116159

theorem new_supervisor_salary 
  (num_workers : ℕ) 
  (num_supervisors : ℕ) 
  (initial_avg_salary : ℚ) 
  (retiring_supervisor_salary : ℚ) 
  (new_avg_salary : ℚ) 
  (h1 : num_workers = 12) 
  (h2 : num_supervisors = 3) 
  (h3 : initial_avg_salary = 650) 
  (h4 : retiring_supervisor_salary = 1200) 
  (h5 : new_avg_salary = 675) : 
  (num_workers + num_supervisors) * new_avg_salary - 
  ((num_workers + num_supervisors) * initial_avg_salary - retiring_supervisor_salary) = 1575 :=
by sorry

end new_supervisor_salary_l1161_116159


namespace cone_base_radius_l1161_116137

theorem cone_base_radius 
  (unfolded_area : ℝ) 
  (generatrix : ℝ) 
  (h1 : unfolded_area = 15 * Real.pi) 
  (h2 : generatrix = 5) : 
  ∃ (base_radius : ℝ), base_radius = 3 ∧ unfolded_area = Real.pi * base_radius * generatrix :=
by sorry

end cone_base_radius_l1161_116137


namespace complex_modulus_one_l1161_116119

theorem complex_modulus_one (z : ℂ) (h : z * (1 + Complex.I) = 1 - Complex.I) : Complex.abs z = 1 := by
  sorry

end complex_modulus_one_l1161_116119


namespace polyline_distance_bound_l1161_116167

/-- Polyline distance between two points -/
def polyline_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := |x₁ - x₂| + |y₁ - y₂|

/-- Theorem: For any point C(x, y) with polyline distance 1 from O(0, 0), √(x² + y²) ≥ √2/2 -/
theorem polyline_distance_bound (x y : ℝ) 
  (h : polyline_distance 0 0 x y = 1) : 
  Real.sqrt (x^2 + y^2) ≥ Real.sqrt 2 / 2 := by
  sorry

end polyline_distance_bound_l1161_116167


namespace squares_in_6x6_grid_l1161_116152

/-- Calculates the number of squares in a grid with n+1 lines in each direction -/
def count_squares (n : ℕ) : ℕ := 
  (n * (n + 1) * (2 * n + 1)) / 6

/-- Theorem: In a 6x6 grid, the total number of squares is 55 -/
theorem squares_in_6x6_grid : count_squares 5 = 55 := by
  sorry

end squares_in_6x6_grid_l1161_116152


namespace distance_pole_to_line_rho_cos_theta_eq_two_l1161_116156

/-- The distance from the pole to a line in polar coordinates -/
def distance_pole_to_line (a : ℝ) : ℝ :=
  |a|

/-- Theorem: The distance from the pole to the line ρcosθ=2 is 2 -/
theorem distance_pole_to_line_rho_cos_theta_eq_two :
  distance_pole_to_line 2 = 2 := by
  sorry

end distance_pole_to_line_rho_cos_theta_eq_two_l1161_116156


namespace max_third_side_length_l1161_116151

theorem max_third_side_length (a b : ℝ) (ha : a = 7) (hb : b = 10) :
  ∃ (x : ℕ), x ≤ 16 ∧
    (∀ (y : ℕ), (y : ℝ) + a > b ∧ (y : ℝ) + b > a ∧ a + b > (y : ℝ) → y ≤ x) ∧
    ((16 : ℝ) + a > b ∧ (16 : ℝ) + b > a ∧ a + b > 16) :=
by sorry

end max_third_side_length_l1161_116151


namespace candy_distribution_l1161_116124

/-- Proves that given the candy distribution conditions, the total number of children is 40 -/
theorem candy_distribution (total_candies : ℕ) (boys girls : ℕ) : 
  total_candies = 90 →
  total_candies / 3 = boys * 3 →
  2 * total_candies / 3 = girls * 2 →
  boys + girls = 40 := by
  sorry

#check candy_distribution

end candy_distribution_l1161_116124


namespace winners_make_zeros_largest_c_winners_make_zeros_optimal_c_l1161_116131

/-- Represents a game state in Winners Make Zeros --/
structure GameState where
  m : ℕ
  n : ℕ

/-- Determines if a given game state is a winning position --/
def is_winning_position (state : GameState) : Prop :=
  sorry

/-- The largest valid choice for c that results in a winning position --/
def largest_winning_c : ℕ :=
  999

theorem winners_make_zeros_largest_c :
  ∀ c : ℕ,
    c > largest_winning_c →
    c > 0 ∧
    2007777 - c * 2007 ≥ 0 →
    ¬is_winning_position ⟨2007777 - c * 2007, 2007⟩ :=
by sorry

theorem winners_make_zeros_optimal_c :
  largest_winning_c > 0 ∧
  2007777 - largest_winning_c * 2007 ≥ 0 ∧
  is_winning_position ⟨2007777 - largest_winning_c * 2007, 2007⟩ :=
by sorry

end winners_make_zeros_largest_c_winners_make_zeros_optimal_c_l1161_116131


namespace pablo_blocks_l1161_116161

theorem pablo_blocks (stack1 stack2 stack3 stack4 : ℕ) : 
  stack1 = 5 →
  stack3 = stack2 - 5 →
  stack4 = stack3 + 5 →
  stack1 + stack2 + stack3 + stack4 = 21 →
  stack2 - stack1 = 2 :=
by sorry

end pablo_blocks_l1161_116161


namespace intersection_complement_problem_l1161_116122

def I : Set ℤ := {x | -3 < x ∧ x < 3}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem intersection_complement_problem :
  A ∩ (I \ B) = {1} := by sorry

end intersection_complement_problem_l1161_116122


namespace parallel_planes_from_parallel_intersecting_lines_parallel_planes_transitive_l1161_116157

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (parallel : Plane → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (in_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Theorem 1
theorem parallel_planes_from_parallel_intersecting_lines 
  (α β : Plane) (l1 l2 m1 m2 : Line) :
  in_plane l1 α → in_plane l2 α → intersect l1 l2 →
  in_plane m1 β → in_plane m2 β → intersect m1 m2 →
  parallel_lines l1 m1 → parallel_lines l2 m2 →
  parallel α β :=
sorry

-- Theorem 2
theorem parallel_planes_transitive (α β γ : Plane) :
  parallel α β → parallel β γ → parallel α γ :=
sorry

end parallel_planes_from_parallel_intersecting_lines_parallel_planes_transitive_l1161_116157


namespace equal_projections_imply_equal_areas_l1161_116183

/-- Represents a parabola -/
structure Parabola where
  -- Add necessary fields to define a parabola

/-- Represents a chord of a parabola -/
structure Chord (p : Parabola) where
  -- Add necessary fields to define a chord

/-- Represents the projection of a chord on the directrix -/
def projection (p : Parabola) (c : Chord p) : ℝ :=
  sorry

/-- Represents the area of the segment cut off by a chord -/
def segmentArea (p : Parabola) (c : Chord p) : ℝ :=
  sorry

/-- Theorem: If two chords of a parabola have equal projections on the directrix,
    then the areas of the segments they cut off are equal -/
theorem equal_projections_imply_equal_areas (p : Parabola) (c1 c2 : Chord p) :
  projection p c1 = projection p c2 → segmentArea p c1 = segmentArea p c2 :=
by sorry

end equal_projections_imply_equal_areas_l1161_116183


namespace line_through_two_points_l1161_116163

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem stating that the equation represents a line through two points
theorem line_through_two_points (M N : Point2D) (h : M ≠ N) :
  ∃! l : Line2D, pointOnLine M l ∧ pointOnLine N l ∧
  ∀ P : Point2D, pointOnLine P l ↔ (P.x - M.x) / (N.x - M.x) = (P.y - M.y) / (N.y - M.y) :=
sorry

end line_through_two_points_l1161_116163


namespace total_cost_pencils_erasers_l1161_116104

/-- Given the price of a pencil (a) and an eraser (b) in yuan, 
    prove that the total cost of 3 pencils and 7 erasers is 3a + 7b yuan. -/
theorem total_cost_pencils_erasers (a b : ℝ) : 3 * a + 7 * b = 3 * a + 7 * b := by
  sorry

end total_cost_pencils_erasers_l1161_116104


namespace segment_division_l1161_116129

/-- Given a segment of length a, prove that dividing it into n equal parts
    results in each part having a length of a/(n+1). -/
theorem segment_division (a : ℝ) (n : ℕ) (h : 0 < n) :
  ∃ (x : ℝ), x = a / (n + 1) ∧ n * x = a :=
by sorry

end segment_division_l1161_116129


namespace min_questions_to_identify_apartment_l1161_116102

theorem min_questions_to_identify_apartment (n : ℕ) (h : n = 80) : 
  (∀ m : ℕ, 2^m < n → m < 7) ∧ (2^7 ≥ n) := by
  sorry

end min_questions_to_identify_apartment_l1161_116102


namespace alloy_mixture_l1161_116149

theorem alloy_mixture (first_alloy_chromium_percent : Real)
                      (second_alloy_chromium_percent : Real)
                      (first_alloy_weight : Real)
                      (new_alloy_chromium_percent : Real)
                      (h1 : first_alloy_chromium_percent = 0.15)
                      (h2 : second_alloy_chromium_percent = 0.08)
                      (h3 : first_alloy_weight = 15)
                      (h4 : new_alloy_chromium_percent = 0.101) :
  ∃ (second_alloy_weight : Real),
    second_alloy_weight = 35 ∧
    new_alloy_chromium_percent * (first_alloy_weight + second_alloy_weight) =
    first_alloy_chromium_percent * first_alloy_weight +
    second_alloy_chromium_percent * second_alloy_weight :=
by
  sorry

end alloy_mixture_l1161_116149


namespace walking_distance_ratio_l1161_116130

/-- The ratio of walking distances given different walking speeds and times -/
theorem walking_distance_ratio 
  (your_speed : ℝ) 
  (harris_speed : ℝ) 
  (harris_time : ℝ) 
  (your_time : ℝ) 
  (h1 : your_speed = 2 * harris_speed) 
  (h2 : harris_time = 2) 
  (h3 : your_time = 3) : 
  (your_speed * your_time) / (harris_speed * harris_time) = 3 := by
sorry

end walking_distance_ratio_l1161_116130


namespace custard_combinations_l1161_116153

theorem custard_combinations (flavors : ℕ) (toppings : ℕ) 
  (h1 : flavors = 5) (h2 : toppings = 7) :
  flavors * (toppings.choose 2) = 105 := by
  sorry

end custard_combinations_l1161_116153


namespace solution_set_part1_solution_set_part2_l1161_116146

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Theorem for part (1)
theorem solution_set_part1 (a : ℝ) (h : a = 1) :
  {x : ℝ | f a x ≥ 3 * x + 2} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} :=
by sorry

-- Theorem for part (2)
theorem solution_set_part2 (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 0} = {x : ℝ | x ≤ -1}) → a = 2 :=
by sorry

end solution_set_part1_solution_set_part2_l1161_116146


namespace donald_oranges_l1161_116170

theorem donald_oranges (initial : ℕ) (total : ℕ) (found : ℕ) : 
  initial = 4 → total = 9 → found = total - initial → found = 5 := by sorry

end donald_oranges_l1161_116170


namespace grocery_store_soda_l1161_116182

theorem grocery_store_soda (regular_soda : ℕ) (apples : ℕ) (total_bottles : ℕ) 
  (h1 : regular_soda = 72)
  (h2 : apples = 78)
  (h3 : total_bottles = apples + 26) :
  total_bottles - regular_soda = 32 := by
  sorry

end grocery_store_soda_l1161_116182


namespace alex_walking_distance_l1161_116165

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents Alex's bike journey -/
structure BikeJourney where
  totalDistance : ℝ
  flatSpeed : ℝ
  flatTime : ℝ
  uphillSpeed : ℝ
  uphillTime : ℝ
  downhillSpeed : ℝ
  downhillTime : ℝ

/-- Calculates the distance Alex had to walk -/
def distanceToWalk (journey : BikeJourney) : ℝ :=
  journey.totalDistance - (distance journey.flatSpeed journey.flatTime +
                           distance journey.uphillSpeed journey.uphillTime +
                           distance journey.downhillSpeed journey.downhillTime)

theorem alex_walking_distance :
  let journey : BikeJourney := {
    totalDistance := 164,
    flatSpeed := 20,
    flatTime := 4.5,
    uphillSpeed := 12,
    uphillTime := 2.5,
    downhillSpeed := 24,
    downhillTime := 1.5
  }
  distanceToWalk journey = 8 := by
  sorry

end alex_walking_distance_l1161_116165


namespace odd_function_property_l1161_116168

-- Define an odd function f: ℝ → ℝ
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_property (f : ℝ → ℝ) (h_odd : isOddFunction f) (h_f_neg_three : f (-3) = 2) :
  f 3 + f 0 = -2 := by
  sorry

end odd_function_property_l1161_116168


namespace fraction_subtraction_simplification_l1161_116180

theorem fraction_subtraction_simplification (d : ℝ) :
  (5 + 4 * d) / 9 - 3 = (4 * d - 22) / 9 := by
  sorry

end fraction_subtraction_simplification_l1161_116180


namespace boat_journey_time_l1161_116143

/-- Calculates the total journey time for a boat traveling upstream and downstream in a river -/
theorem boat_journey_time 
  (river_speed : ℝ) 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (h1 : river_speed = 2)
  (h2 : boat_speed = 6)
  (h3 : distance = 64) : 
  (distance / (boat_speed - river_speed)) + (distance / (boat_speed + river_speed)) = 24 := by
  sorry

#check boat_journey_time

end boat_journey_time_l1161_116143
