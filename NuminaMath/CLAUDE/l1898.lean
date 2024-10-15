import Mathlib

namespace NUMINAMATH_CALUDE_natural_exp_inequality_l1898_189885

theorem natural_exp_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b := by
  sorry

end NUMINAMATH_CALUDE_natural_exp_inequality_l1898_189885


namespace NUMINAMATH_CALUDE_wilsons_theorem_l1898_189875

theorem wilsons_theorem (p : ℕ) (h : p > 1) :
  Nat.Prime p ↔ (Nat.factorial (p - 1) % p = p - 1) := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l1898_189875


namespace NUMINAMATH_CALUDE_min_value_a_l1898_189800

theorem min_value_a (a b : ℕ) (h : 1998 * a = b^4) : 1215672 ≤ a := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l1898_189800


namespace NUMINAMATH_CALUDE_square_root_of_one_is_one_l1898_189840

theorem square_root_of_one_is_one : Real.sqrt 1 = 1 := by sorry

end NUMINAMATH_CALUDE_square_root_of_one_is_one_l1898_189840


namespace NUMINAMATH_CALUDE_karlson_candy_theorem_l1898_189896

/-- The number of initial ones on the board -/
def initial_ones : ℕ := 39

/-- The number of minutes the process continues -/
def total_minutes : ℕ := 39

/-- The maximum number of candies Karlson could have eaten -/
def max_candies : ℕ := initial_ones.choose 2

theorem karlson_candy_theorem : 
  max_candies = (initial_ones * (initial_ones - 1)) / 2 := by
  sorry

#eval max_candies

end NUMINAMATH_CALUDE_karlson_candy_theorem_l1898_189896


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_9_and_3_digit_by_4_l1898_189826

/-- Represents a four-digit number -/
structure FourDigitNumber where
  value : Nat
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- Returns the three-digit number obtained by removing the last digit -/
def remove_last_digit (n : FourDigitNumber) : Nat :=
  n.value / 10

/-- Returns the last digit of a number -/
def last_digit (n : FourDigitNumber) : Nat :=
  n.value % 10

theorem largest_four_digit_divisible_by_9_and_3_digit_by_4 (n : FourDigitNumber) 
  (h1 : n.value % 9 = 0)
  (h2 : remove_last_digit n % 4 = 0)
  (h3 : ∀ m : FourDigitNumber, m.value % 9 = 0 → remove_last_digit m % 4 = 0 → m.value ≤ n.value) :
  last_digit n = 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_9_and_3_digit_by_4_l1898_189826


namespace NUMINAMATH_CALUDE_investment_interest_rate_l1898_189877

/-- Proves that given the specified investment conditions, the unknown interest rate is 8% --/
theorem investment_interest_rate : 
  ∀ (x y r : ℚ),
  x + y = 2000 →
  y = 650 →
  x * (1/10) - y * r = 83 →
  r = 8/100 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l1898_189877


namespace NUMINAMATH_CALUDE_fuel_used_calculation_l1898_189822

/-- Calculates the total fuel used given the initial capacity, intermediate reading, and final reading after refill -/
def total_fuel_used (initial_capacity : ℝ) (intermediate_reading : ℝ) (final_reading : ℝ) : ℝ :=
  (initial_capacity - intermediate_reading) + (initial_capacity - final_reading)

/-- Theorem stating that the total fuel used is 4582 L given the specific readings -/
theorem fuel_used_calculation :
  let initial_capacity : ℝ := 3000
  let intermediate_reading : ℝ := 180
  let final_reading : ℝ := 1238
  total_fuel_used initial_capacity intermediate_reading final_reading = 4582 := by
  sorry

#eval total_fuel_used 3000 180 1238

end NUMINAMATH_CALUDE_fuel_used_calculation_l1898_189822


namespace NUMINAMATH_CALUDE_geometric_progression_problem_l1898_189830

theorem geometric_progression_problem (b₂ b₆ : ℚ) 
  (h₂ : b₂ = 37 + 1/3) 
  (h₆ : b₆ = 2 + 1/3) : 
  ∃ (a q : ℚ), a * q = b₂ ∧ a * q^5 = b₆ ∧ a = 224/3 ∧ q = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_problem_l1898_189830


namespace NUMINAMATH_CALUDE_quadratic_equation_from_roots_l1898_189863

theorem quadratic_equation_from_roots (r s : ℝ) 
  (sum_roots : r + s = 12)
  (product_roots : r * s = 27)
  (root_relation : s = 3 * r) : 
  ∀ x : ℝ, x^2 - 12*x + 27 = (x - r) * (x - s) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_roots_l1898_189863


namespace NUMINAMATH_CALUDE_combined_savings_difference_l1898_189841

/-- The cost of a single window -/
def window_cost : ℕ := 100

/-- The number of windows purchased to get one free -/
def windows_for_free : ℕ := 4

/-- The number of windows Dave needs -/
def dave_windows : ℕ := 7

/-- The number of windows Doug needs -/
def doug_windows : ℕ := 8

/-- Calculate the cost of windows with the promotion -/
def cost_with_promotion (n : ℕ) : ℕ :=
  ((n + windows_for_free - 1) / windows_for_free) * windows_for_free * window_cost

/-- Calculate the savings for a given number of windows -/
def savings (n : ℕ) : ℕ :=
  n * window_cost - cost_with_promotion n

/-- The main theorem: combined savings minus individual savings equals $100 -/
theorem combined_savings_difference : 
  savings (dave_windows + doug_windows) - (savings dave_windows + savings doug_windows) = 100 := by
  sorry

end NUMINAMATH_CALUDE_combined_savings_difference_l1898_189841


namespace NUMINAMATH_CALUDE_tangent_line_condition_l1898_189897

-- Define the condition for a line being tangent to a circle
def is_tangent (k : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + 2 ∧ x^2 + y^2 = 1 ∧
  ∀ x' y' : ℝ, y' = k * x' + 2 → x'^2 + y'^2 ≥ 1

-- State the theorem
theorem tangent_line_condition :
  (∀ k : ℝ, ¬(k = Real.sqrt 3) → ¬(is_tangent k)) ∧
  ¬(∀ k : ℝ, ¬(is_tangent k) → ¬(k = Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_condition_l1898_189897


namespace NUMINAMATH_CALUDE_real_roots_sum_product_l1898_189848

theorem real_roots_sum_product (a b : ℝ) : 
  (a^4 - 6*a^2 - a + 2 = 0) → 
  (b^4 - 6*b^2 - b + 2 = 0) → 
  (∀ x : ℝ, x^4 - 6*x^2 - x + 2 = 0 → x = a ∨ x = b) →
  a * b + a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_sum_product_l1898_189848


namespace NUMINAMATH_CALUDE_trig_fraction_value_l1898_189827

theorem trig_fraction_value (θ : Real) (h : Real.sin θ + 2 * Real.cos θ = 1) :
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = -7 ∨
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_value_l1898_189827


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1898_189832

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 5)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → x = -4/5 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1898_189832


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_three_l1898_189804

theorem subset_implies_m_equals_three (A B : Set ℕ) (m : ℕ) :
  A = {1, 3} →
  B = {1, 2, m} →
  A ⊆ B →
  m = 3 := by sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_three_l1898_189804


namespace NUMINAMATH_CALUDE_ordering_abc_l1898_189807

theorem ordering_abc (a b c : ℝ) : 
  a = 31/32 → 
  b = Real.cos (1/4) → 
  c = 4 * Real.sin (1/4) → 
  c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_ordering_abc_l1898_189807


namespace NUMINAMATH_CALUDE_corrected_mean_problem_l1898_189853

/-- Calculates the corrected mean of a set of observations after fixing an error -/
def corrected_mean (n : ℕ) (original_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * original_mean - wrong_value + correct_value) / n

/-- Theorem stating that the corrected mean for the given problem is 36.42 -/
theorem corrected_mean_problem : 
  corrected_mean 50 36 23 44 = 36.42 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_problem_l1898_189853


namespace NUMINAMATH_CALUDE_billy_crayons_l1898_189843

theorem billy_crayons (initial_crayons eaten_crayons remaining_crayons : ℕ) :
  eaten_crayons = 52 →
  remaining_crayons = 10 →
  initial_crayons = eaten_crayons + remaining_crayons →
  initial_crayons = 62 := by
  sorry

end NUMINAMATH_CALUDE_billy_crayons_l1898_189843


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l1898_189851

theorem cubic_equation_solutions (x : ℝ) : 
  2.21 * (((5 + x)^2)^(1/3)) + 4 * (((5 - x)^2)^(1/3)) = 5 * ((25 - x)^(1/3)) ↔ 
  x = 0 ∨ x = 63/13 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l1898_189851


namespace NUMINAMATH_CALUDE_product_xyz_w_l1898_189898

theorem product_xyz_w (x y z w : ℚ) 
  (eq1 : 3 * x + 4 * y = 60)
  (eq2 : 6 * x - 4 * y = 12)
  (eq3 : 2 * x - 3 * z = 38)
  (eq4 : x + y + z = w) :
  x * y * z * w = -5104 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_w_l1898_189898


namespace NUMINAMATH_CALUDE_parabola_vertex_l1898_189876

/-- The vertex of the parabola y = 2x^2 + 16x + 50 is (-4, 18) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 2 * x^2 + 16 * x + 50 → (∃ m n : ℝ, m = -4 ∧ n = 18 ∧ 
    ∀ x, y = 2 * (x - m)^2 + n) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1898_189876


namespace NUMINAMATH_CALUDE_tenth_term_of_inverse_proportional_sequence_l1898_189872

/-- A sequence where each term after the first is inversely proportional to the preceding term -/
def InverseProportionalSequence (a : ℕ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) * a n = k

theorem tenth_term_of_inverse_proportional_sequence
  (a : ℕ → ℝ)
  (h_seq : InverseProportionalSequence a)
  (h_first : a 1 = 3)
  (h_second : a 2 = 4) :
  a 10 = 4 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_inverse_proportional_sequence_l1898_189872


namespace NUMINAMATH_CALUDE_olivias_groceries_cost_l1898_189887

/-- The total cost of Olivia's groceries is $42 -/
theorem olivias_groceries_cost (banana_cost bread_cost milk_cost apple_cost : ℕ)
  (h1 : banana_cost = 12)
  (h2 : bread_cost = 9)
  (h3 : milk_cost = 7)
  (h4 : apple_cost = 14) :
  banana_cost + bread_cost + milk_cost + apple_cost = 42 := by
  sorry

end NUMINAMATH_CALUDE_olivias_groceries_cost_l1898_189887


namespace NUMINAMATH_CALUDE_least_x_value_l1898_189821

theorem least_x_value (x p : ℕ) (h1 : x > 0) (h2 : Nat.Prime p) 
  (h3 : ∃ q : ℕ, Nat.Prime q ∧ q % 2 = 1 ∧ x = 12 * p * q) : 
  x ≥ 72 ∧ (∃ x₀ : ℕ, x₀ ≥ 72 → 
    (∃ p₀ q₀ : ℕ, Nat.Prime p₀ ∧ Nat.Prime q₀ ∧ q₀ % 2 = 1 ∧ x₀ = 12 * p₀ * q₀) → x₀ ≥ x) :=
by sorry

end NUMINAMATH_CALUDE_least_x_value_l1898_189821


namespace NUMINAMATH_CALUDE_two_digit_number_proof_l1898_189893

theorem two_digit_number_proof :
  ∀ n : ℕ,
  (10 ≤ n ∧ n < 100) →  -- two-digit number
  (n % 2 = 0) →  -- even number
  (n / 10 * (n % 10) = 20) →  -- product of digits is 20
  n = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_proof_l1898_189893


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1898_189860

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (2 + I) / (3 - I) → z.im = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1898_189860


namespace NUMINAMATH_CALUDE_vector_q_solution_l1898_189801

/-- Custom vector operation ⊗ -/
def vector_op (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 * b.1, a.2 * b.2)

theorem vector_q_solution :
  let p : ℝ × ℝ := (1, 2)
  let q : ℝ × ℝ := (-3, -2)
  vector_op p q = (-3, -4) :=
by sorry

end NUMINAMATH_CALUDE_vector_q_solution_l1898_189801


namespace NUMINAMATH_CALUDE_f_neg_five_halves_l1898_189814

-- Define the function f
def f : ℝ → ℝ := sorry

-- f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- f has a period of 2
axiom f_periodic : ∀ x, f (x + 2) = f x

-- f(x) = 2x(1-x) when 0 ≤ x ≤ 1
axiom f_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)

-- Theorem to prove
theorem f_neg_five_halves : f (-5/2) = -1/2 := sorry

end NUMINAMATH_CALUDE_f_neg_five_halves_l1898_189814


namespace NUMINAMATH_CALUDE_exists_cycle_not_div_by_three_l1898_189824

/-- A graph is a set of vertices and a set of edges between them. -/
structure Graph (V : Type) where
  edge : V → V → Prop

/-- The degree of a vertex in a graph is the number of edges connected to it. -/
def degree (G : Graph V) (v : V) : ℕ := sorry

/-- A path in a graph is a sequence of vertices where each consecutive pair is connected by an edge. -/
def is_path (G : Graph V) (path : List V) : Prop := sorry

/-- A cycle is a path that starts and ends at the same vertex. -/
def is_cycle (G : Graph V) (cycle : List V) : Prop := sorry

/-- The main theorem: In any graph where each vertex has degree at least 3,
    there exists a cycle whose length is not divisible by 3. -/
theorem exists_cycle_not_div_by_three {V : Type} (G : Graph V) :
  (∀ v : V, degree G v ≥ 3) →
  ∃ cycle : List V, is_cycle G cycle ∧ (cycle.length % 3 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_exists_cycle_not_div_by_three_l1898_189824


namespace NUMINAMATH_CALUDE_linear_system_det_proof_l1898_189813

/-- Given a linear equation system represented by an augmented matrix,
    prove that the determinant of a specific matrix using the solution is -1 -/
theorem linear_system_det_proof (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) :
  a₁ = 2 ∧ b₁ = 0 ∧ c₁ = 2 ∧ a₂ = 3 ∧ b₂ = 1 ∧ c₂ = 2 →
  ∃ x y : ℝ, a₁ * x + b₁ * y = c₁ ∧ a₂ * x + b₂ * y = c₂ →
  x * 2 - y * (-3) = -1 := by
sorry


end NUMINAMATH_CALUDE_linear_system_det_proof_l1898_189813


namespace NUMINAMATH_CALUDE_old_manufacturing_cost_calculation_l1898_189867

def selling_price : ℝ := 100
def new_profit_percentage : ℝ := 0.50
def old_profit_percentage : ℝ := 0.20
def new_manufacturing_cost : ℝ := 50

theorem old_manufacturing_cost_calculation :
  let old_manufacturing_cost := selling_price * (1 - old_profit_percentage)
  old_manufacturing_cost = 80 :=
by sorry

end NUMINAMATH_CALUDE_old_manufacturing_cost_calculation_l1898_189867


namespace NUMINAMATH_CALUDE_distribute_5_3_l1898_189817

/-- The number of ways to distribute n distinct objects into k distinct containers,
    with each container receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct objects into 3 distinct containers,
    with each container receiving at least one object, is 150. -/
theorem distribute_5_3 : distribute 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_5_3_l1898_189817


namespace NUMINAMATH_CALUDE_walk_legs_and_wheels_l1898_189831

/-- Calculates the total number of legs and wheels for a group of organisms and a wheelchair -/
def total_legs_and_wheels (humans : ℕ) (dogs : ℕ) (cats : ℕ) (horses : ℕ) (monkeys : ℕ) (wheelchair_wheels : ℕ) : ℕ :=
  humans * 2 + dogs * 4 + cats * 4 + horses * 4 + monkeys * 4 + wheelchair_wheels

/-- Proves that the total number of legs and wheels for the given group is 46 -/
theorem walk_legs_and_wheels :
  total_legs_and_wheels 9 3 1 1 1 4 = 46 := by
  sorry

end NUMINAMATH_CALUDE_walk_legs_and_wheels_l1898_189831


namespace NUMINAMATH_CALUDE_distance_between_points_l1898_189815

theorem distance_between_points : 
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (-4, 7)
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1898_189815


namespace NUMINAMATH_CALUDE_triangle_area_after_transformation_l1898_189842

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![-1, 5]
def T : Matrix (Fin 2) (Fin 2) ℝ := !![1, 1; 0, 1]

theorem triangle_area_after_transformation :
  let Ta := T.mulVec a
  let Tb := T.mulVec b
  (1/2) * abs (Ta 0 * Tb 1 - Ta 1 * Tb 0) = 8.5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_after_transformation_l1898_189842


namespace NUMINAMATH_CALUDE_lemon_pie_degrees_l1898_189864

theorem lemon_pie_degrees (total_students : ℕ) (chocolate : ℕ) (apple : ℕ) (blueberry : ℕ)
  (h1 : total_students = 40)
  (h2 : chocolate = 15)
  (h3 : apple = 9)
  (h4 : blueberry = 7)
  (h5 : (total_students - (chocolate + apple + blueberry)) % 2 = 0) :
  let remaining := total_students - (chocolate + apple + blueberry)
  let lemon := remaining / 2
  (lemon : ℚ) / total_students * 360 = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_lemon_pie_degrees_l1898_189864


namespace NUMINAMATH_CALUDE_abs_diff_sqrt_two_l1898_189806

theorem abs_diff_sqrt_two : ∀ x : ℝ, 1 < x ∧ x < 2 ∧ x^2 = 2 → |3 - x| - |x - 2| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_sqrt_two_l1898_189806


namespace NUMINAMATH_CALUDE_third_team_wins_l1898_189820

/-- Represents the amount of wood processed by a team of lumberjacks -/
structure WoodProcessed where
  amount : ℝ
  amount_pos : amount > 0

/-- The competition between three teams of lumberjacks -/
structure LumberjackCompetition where
  team1 : WoodProcessed
  team2 : WoodProcessed
  team3 : WoodProcessed
  first_third_twice_second : team1.amount + team3.amount = 2 * team2.amount
  second_third_thrice_first : team2.amount + team3.amount = 3 * team1.amount

/-- The third team processes the most wood in the competition -/
theorem third_team_wins (comp : LumberjackCompetition) : 
  comp.team3.amount > comp.team1.amount ∧ comp.team3.amount > comp.team2.amount := by
  sorry

#check third_team_wins

end NUMINAMATH_CALUDE_third_team_wins_l1898_189820


namespace NUMINAMATH_CALUDE_absolute_difference_of_numbers_l1898_189890

theorem absolute_difference_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 36) 
  (product_eq : x * y = 320) : 
  |x - y| = 4 := by sorry

end NUMINAMATH_CALUDE_absolute_difference_of_numbers_l1898_189890


namespace NUMINAMATH_CALUDE_mark_soup_donation_l1898_189871

/-- The number of homeless shelters Mark donates to -/
def num_shelters : ℕ := 6

/-- The number of people served per shelter -/
def people_per_shelter : ℕ := 30

/-- The number of cans of soup per person -/
def cans_per_person : ℕ := 10

/-- The total number of cans Mark donates -/
def total_cans : ℕ := num_shelters * people_per_shelter * cans_per_person

theorem mark_soup_donation : total_cans = 1800 := by
  sorry

end NUMINAMATH_CALUDE_mark_soup_donation_l1898_189871


namespace NUMINAMATH_CALUDE_binomial_307_307_equals_1_l1898_189852

theorem binomial_307_307_equals_1 : Nat.choose 307 307 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_307_307_equals_1_l1898_189852


namespace NUMINAMATH_CALUDE_hexagon_planes_count_l1898_189855

/-- A regular dodecahedron in three-dimensional space. -/
structure RegularDodecahedron

/-- A plane in three-dimensional space. -/
structure Plane

/-- The number of large diagonals in a regular dodecahedron. -/
def num_large_diagonals : ℕ := 10

/-- The number of planes perpendicular to each large diagonal that produce a regular hexagon slice. -/
def planes_per_diagonal : ℕ := 3

/-- A function that counts the number of planes intersecting a regular dodecahedron to produce a regular hexagon. -/
def count_hexagon_planes (d : RegularDodecahedron) : ℕ :=
  num_large_diagonals * planes_per_diagonal

/-- Theorem stating that the number of planes intersecting a regular dodecahedron to produce a regular hexagon is 30. -/
theorem hexagon_planes_count (d : RegularDodecahedron) :
  count_hexagon_planes d = 30 := by sorry

end NUMINAMATH_CALUDE_hexagon_planes_count_l1898_189855


namespace NUMINAMATH_CALUDE_boys_in_line_l1898_189823

theorem boys_in_line (n : ℕ) (h : n > 0) :
  (∃ k : ℕ, k > 0 ∧ k = 19 ∧ k = n + 1 - 19) → n = 37 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_line_l1898_189823


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_two_ninths_l1898_189847

theorem sum_of_fractions_equals_two_ninths :
  let sum := (1 : ℚ) / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8) + 1 / (8 * 9)
  sum = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_two_ninths_l1898_189847


namespace NUMINAMATH_CALUDE_max_value_on_interval_l1898_189838

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (f 1 = 2) ∧
  (∀ x y, x > 0 → y > 0 → f x < f y) ∧
  (∀ x y, f (x + y) = f x + f y)

theorem max_value_on_interval 
  (f : ℝ → ℝ) 
  (h : f_properties f) :
  ∃ x ∈ Set.Icc (-3) (-2), ∀ y ∈ Set.Icc (-3) (-2), f y ≤ f x ∧ f x = -4 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_interval_l1898_189838


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_88_l1898_189818

theorem largest_four_digit_divisible_by_88 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 88 = 0 → n ≤ 9944 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_88_l1898_189818


namespace NUMINAMATH_CALUDE_john_ben_difference_l1898_189808

/-- Represents the marble transfer problem --/
structure MarbleTransfer where
  ben_initial : ℝ
  john_initial : ℝ
  lisa_initial : ℝ
  max_initial : ℝ
  ben_to_john_percent : ℝ
  ben_to_lisa_percent : ℝ
  john_to_max_percent : ℝ
  lisa_to_john_percent : ℝ

/-- Calculates the final marble counts after all transfers --/
def finalCounts (mt : MarbleTransfer) : ℝ × ℝ × ℝ × ℝ :=
  let ben_to_john := mt.ben_initial * mt.ben_to_john_percent
  let ben_to_lisa := mt.ben_initial * mt.ben_to_lisa_percent
  let ben_final := mt.ben_initial - ben_to_john - ben_to_lisa
  let john_from_ben := ben_to_john
  let john_to_max := john_from_ben * mt.john_to_max_percent
  let lisa_with_ben := mt.lisa_initial + ben_to_lisa
  let lisa_to_john := mt.lisa_initial * mt.lisa_to_john_percent + ben_to_lisa
  let john_final := mt.john_initial + john_from_ben - john_to_max + lisa_to_john
  let max_final := mt.max_initial + john_to_max
  let lisa_final := lisa_with_ben - lisa_to_john
  (ben_final, john_final, lisa_final, max_final)

/-- Theorem stating the difference in marbles between John and Ben after transfers --/
theorem john_ben_difference (mt : MarbleTransfer) 
  (h1 : mt.ben_initial = 18)
  (h2 : mt.john_initial = 17)
  (h3 : mt.lisa_initial = 12)
  (h4 : mt.max_initial = 9)
  (h5 : mt.ben_to_john_percent = 0.5)
  (h6 : mt.ben_to_lisa_percent = 0.25)
  (h7 : mt.john_to_max_percent = 0.65)
  (h8 : mt.lisa_to_john_percent = 0.2) :
  (finalCounts mt).2.1 - (finalCounts mt).1 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_john_ben_difference_l1898_189808


namespace NUMINAMATH_CALUDE_second_number_value_l1898_189846

theorem second_number_value (A B C D : ℝ) : 
  C = 4.5 * B →
  B = 2.5 * A →
  D = 0.5 * (A + B) →
  (A + B + C + D) / 4 = 165 →
  B = 100 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l1898_189846


namespace NUMINAMATH_CALUDE_trigonometric_equation_consequences_l1898_189858

open Real

theorem trigonometric_equation_consequences (α : ℝ) 
  (h : sin (π - α) * cos (2*π - α) / (tan (π - α) * sin (π/2 + α) * cos (π/2 - α)) = 1/2) : 
  (cos α - 2*sin α) / (3*cos α + sin α) = 5 ∧ 
  1 - 2*sin α*cos α + cos α^2 = 2/5 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_consequences_l1898_189858


namespace NUMINAMATH_CALUDE_aruns_weight_estimation_l1898_189886

/-- Arun's weight estimation problem -/
theorem aruns_weight_estimation (W : ℝ) (L : ℝ) : 
  (L < W ∧ W < 72) →  -- Arun's estimation
  (60 < W ∧ W < 70) →  -- Brother's estimation
  (W ≤ 68) →  -- Mother's estimation
  (∃ (a b : ℝ), 60 < a ∧ a < b ∧ b ≤ 68 ∧ (a + b) / 2 = 67) →  -- Average condition
  L > 60 := by
sorry

end NUMINAMATH_CALUDE_aruns_weight_estimation_l1898_189886


namespace NUMINAMATH_CALUDE_tree_height_difference_l1898_189802

theorem tree_height_difference : 
  let maple_height : ℚ := 13 + 1/4
  let pine_height : ℚ := 19 + 3/8
  pine_height - maple_height = 6 + 1/8 := by
sorry

end NUMINAMATH_CALUDE_tree_height_difference_l1898_189802


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_negative_two_sqrt_ten_l1898_189892

theorem sqrt_difference_equals_negative_two_sqrt_ten :
  Real.sqrt (25 - 10 * Real.sqrt 6) - Real.sqrt (25 + 10 * Real.sqrt 6) = -2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_negative_two_sqrt_ten_l1898_189892


namespace NUMINAMATH_CALUDE_triangle_problem_l1898_189880

theorem triangle_problem (a b c A B C : ℝ) :
  (0 < A) ∧ (A < π) →
  (0 < B) ∧ (B < π) →
  (0 < C) ∧ (C < π) →
  (A + B + C = π) →
  (c = Real.sqrt 3 * a * Real.sin C - c * Real.cos A) →
  (a = 2) →
  (1/2 * b * c * Real.sin A = Real.sqrt 3) →
  (A = π/3 ∧ b = 2 ∧ c = 2) := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1898_189880


namespace NUMINAMATH_CALUDE_levels_for_110_blocks_l1898_189837

/-- The number of blocks in the nth level of the pattern -/
def blocks_in_level (n : ℕ) : ℕ := 2 + 2 * (n - 1)

/-- The total number of blocks used up to the nth level -/
def total_blocks (n : ℕ) : ℕ := n * (n + 1)

/-- The theorem stating that 10 levels are needed to use exactly 110 blocks -/
theorem levels_for_110_blocks :
  ∃ (n : ℕ), n > 0 ∧ total_blocks n = 110 ∧ 
  (∀ (m : ℕ), m > 0 ∧ m ≠ n → total_blocks m ≠ 110) :=
sorry

end NUMINAMATH_CALUDE_levels_for_110_blocks_l1898_189837


namespace NUMINAMATH_CALUDE_exists_fixed_point_l1898_189811

variable {X : Type u}
variable (μ : Set X → Set X)

axiom μ_union_disjoint {A B : Set X} (h : Disjoint A B) : μ (A ∪ B) = μ A ∪ μ B

theorem exists_fixed_point : ∃ F : Set X, μ F = F := by
  sorry

end NUMINAMATH_CALUDE_exists_fixed_point_l1898_189811


namespace NUMINAMATH_CALUDE_min_expression_upper_bound_l1898_189803

theorem min_expression_upper_bound (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min (min (min (1 / a) (2 / b)) (4 / c)) (Real.rpow (a * b * c) (1 / 3)) ≤ Real.sqrt 2 ∧
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧
    min (min (min (1 / a) (2 / b)) (4 / c)) (Real.rpow (a * b * c) (1 / 3)) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_expression_upper_bound_l1898_189803


namespace NUMINAMATH_CALUDE_divisibility_problem_l1898_189894

theorem divisibility_problem (n : ℕ) (h1 : n = 6268440) (h2 : n % 5 = 0) : n % 30 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1898_189894


namespace NUMINAMATH_CALUDE_correct_number_of_arrangements_l1898_189857

/-- The number of arrangements for 3 boys and 3 girls in a line, where students of the same gender are adjacent -/
def number_of_arrangements : ℕ := 72

/-- The number of boys -/
def num_boys : ℕ := 3

/-- The number of girls -/
def num_girls : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_boys + num_girls

/-- Theorem stating that the number of arrangements is correct -/
theorem correct_number_of_arrangements :
  number_of_arrangements = (Nat.factorial num_boys) * (Nat.factorial num_girls) * 2 :=
by sorry

end NUMINAMATH_CALUDE_correct_number_of_arrangements_l1898_189857


namespace NUMINAMATH_CALUDE_sum_of_a_and_c_l1898_189856

theorem sum_of_a_and_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 20) 
  (h2 : b + d = 4) : 
  a + c = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_c_l1898_189856


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l1898_189859

/-- A quadratic function that takes values 6, 5, and 5 for three consecutive natural numbers. -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, f n = 6 ∧ f (n + 1) = 5 ∧ f (n + 2) = 5

/-- The theorem stating that the minimum value of the quadratic function is 39/8. -/
theorem quadratic_minimum_value (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∃ x : ℝ, f x = 39/8 ∧ ∀ y : ℝ, f y ≥ 39/8 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l1898_189859


namespace NUMINAMATH_CALUDE_rectangle_rotation_path_length_l1898_189865

/-- The length of the path traveled by point P of a rectangle PQRS after two 90° rotations -/
theorem rectangle_rotation_path_length (P Q R S : ℝ × ℝ) : 
  let pq : ℝ := 2
  let rs : ℝ := 2
  let qr : ℝ := 6
  let sp : ℝ := 6
  let first_rotation_radius : ℝ := Real.sqrt (pq^2 + qr^2)
  let first_rotation_arc_length : ℝ := (π / 2) * first_rotation_radius
  let second_rotation_radius : ℝ := sp
  let second_rotation_arc_length : ℝ := (π / 2) * second_rotation_radius
  let total_path_length : ℝ := first_rotation_arc_length + second_rotation_arc_length
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = pq^2 →
  (R.1 - S.1)^2 + (R.2 - S.2)^2 = rs^2 →
  (Q.1 - R.1)^2 + (Q.2 - R.2)^2 = qr^2 →
  (S.1 - P.1)^2 + (S.2 - P.2)^2 = sp^2 →
  total_path_length = (3 + Real.sqrt 10) * π := by
sorry

end NUMINAMATH_CALUDE_rectangle_rotation_path_length_l1898_189865


namespace NUMINAMATH_CALUDE_odd_divisors_implies_perfect_square_l1898_189809

/-- The number of divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- A natural number is a perfect square if there exists an integer k such that n = k^2 -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- If a natural number has an odd number of divisors, then it is a perfect square -/
theorem odd_divisors_implies_perfect_square (n : ℕ) : 
  Odd (num_divisors n) → is_perfect_square n := by sorry

end NUMINAMATH_CALUDE_odd_divisors_implies_perfect_square_l1898_189809


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1898_189889

def M : Set ℝ := {x : ℝ | x^2 - 3*x - 28 ≤ 0}
def N : Set ℝ := {x : ℝ | x^2 - x - 6 > 0}

theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | (-4 ≤ x ∧ x ≤ -2) ∨ (3 < x ∧ x ≤ 7)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1898_189889


namespace NUMINAMATH_CALUDE_gcd_of_256_180_600_l1898_189866

theorem gcd_of_256_180_600 : Nat.gcd 256 (Nat.gcd 180 600) = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_256_180_600_l1898_189866


namespace NUMINAMATH_CALUDE_range_of_a_l1898_189805

open Set

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x < a}
def B : Set ℝ := {x | 1 < x ∧ x < 2}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (A a ∪ (Bᶜ) = univ) → a ∈ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1898_189805


namespace NUMINAMATH_CALUDE_escape_theorem_l1898_189845

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circular pond -/
structure Pond where
  center : Point
  radius : ℝ

/-- Represents a person with swimming and running speeds -/
structure Person where
  position : Point
  swimSpeed : ℝ
  runSpeed : ℝ

/-- Checks if a person can escape from another in a circular pond -/
def canEscape (pond : Pond) (escaper : Person) (chaser : Person) : Prop :=
  ∃ (t : ℝ), t > 0 ∧
  ∃ (escapePoint : Point),
    (escapePoint.x - pond.center.x)^2 + (escapePoint.y - pond.center.y)^2 > pond.radius^2 ∧
    (escapePoint.x - escaper.position.x)^2 + (escapePoint.y - escaper.position.y)^2 ≤ (escaper.swimSpeed * t)^2 ∧
    (escapePoint.x - chaser.position.x)^2 + (escapePoint.y - chaser.position.y)^2 > (chaser.runSpeed * t)^2

theorem escape_theorem (pond : Pond) (x y : Person) :
  x.position = pond.center →
  (y.position.x - pond.center.x)^2 + (y.position.y - pond.center.y)^2 = pond.radius^2 →
  y.runSpeed = 4 * x.swimSpeed →
  x.runSpeed > 4 * x.swimSpeed →
  canEscape pond x y :=
sorry

end NUMINAMATH_CALUDE_escape_theorem_l1898_189845


namespace NUMINAMATH_CALUDE_zero_in_P_l1898_189833

def P : Set ℝ := {x | x > -1}

theorem zero_in_P : (0 : ℝ) ∈ P := by sorry

end NUMINAMATH_CALUDE_zero_in_P_l1898_189833


namespace NUMINAMATH_CALUDE_director_dividends_director_dividends_calculation_l1898_189835

/-- Calculates the dividends for the General Director given the financial data of the company. -/
theorem director_dividends (revenue : ℝ) (expenses : ℝ) (tax_rate : ℝ)
                           (monthly_loan_payment : ℝ) (annual_interest : ℝ)
                           (total_shares : ℕ) (director_shares : ℕ) : ℝ :=
  let net_profit := (revenue - expenses) - (revenue - expenses) * tax_rate
  let total_loan_payments := monthly_loan_payment * 12 - annual_interest
  let profits_for_dividends := net_profit - total_loan_payments
  let dividend_per_share := profits_for_dividends / total_shares
  dividend_per_share * director_shares

/-- The General Director's dividends are 246,400.0 rubles given the specified financial conditions. -/
theorem director_dividends_calculation :
  director_dividends 1500000 674992 0.2 23914 74992 1000 550 = 246400 := by
  sorry

end NUMINAMATH_CALUDE_director_dividends_director_dividends_calculation_l1898_189835


namespace NUMINAMATH_CALUDE_sally_weekend_pages_l1898_189879

/-- The number of pages Sally reads on weekdays -/
def weekday_pages : ℕ := 10

/-- The number of weeks it takes Sally to finish the book -/
def weeks_to_finish : ℕ := 2

/-- The total number of pages in the book -/
def total_pages : ℕ := 180

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days_per_week : ℕ := 2

/-- Theorem: Sally reads 20 pages on each weekend day -/
theorem sally_weekend_pages : 
  (total_pages - weekday_pages * weekdays_per_week * weeks_to_finish) / (weekend_days_per_week * weeks_to_finish) = 20 := by
  sorry

end NUMINAMATH_CALUDE_sally_weekend_pages_l1898_189879


namespace NUMINAMATH_CALUDE_vector_sum_in_triangle_l1898_189829

-- Define the triangle ABC and points E and F
variable (A B C E F : ℝ × ℝ)

-- Define vectors
def AB : ℝ × ℝ := B - A
def AC : ℝ × ℝ := C - A
def AE : ℝ × ℝ := E - A
def CF : ℝ × ℝ := F - C
def FA : ℝ × ℝ := A - F
def EF : ℝ × ℝ := F - E

-- Define conditions
variable (h1 : AE = (1/2 : ℝ) • AB)
variable (h2 : CF = (2 : ℝ) • FA)
variable (x y : ℝ)
variable (h3 : EF = x • AB + y • AC)

-- Theorem statement
theorem vector_sum_in_triangle : x + y = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_in_triangle_l1898_189829


namespace NUMINAMATH_CALUDE_inequality_proof_l1898_189819

theorem inequality_proof (A B C ε : Real) 
  (hA : 0 ≤ A ∧ A ≤ π) 
  (hB : 0 ≤ B ∧ B ≤ π) 
  (hC : 0 ≤ C ∧ C ≤ π) 
  (hε : ε ≥ 1) : 
  ε * (Real.sin A + Real.sin B + Real.sin C) ≤ Real.sin A * Real.sin B * Real.sin C + 1 + ε^3 ∧ 
  (1 + ε + Real.sin A) * (1 + ε + Real.sin B) * (1 + ε + Real.sin C) ≥ 9 * ε * (Real.sin A + Real.sin B + Real.sin C) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1898_189819


namespace NUMINAMATH_CALUDE_symmetry_implies_b_pow_a_l1898_189873

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposite and y-coordinates are equal -/
def symmetric_y_axis (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 = -x2 ∧ y1 = y2

theorem symmetry_implies_b_pow_a (a b : ℝ) :
  symmetric_y_axis (2*a) 2 (-8) (a+b) → b^a = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_b_pow_a_l1898_189873


namespace NUMINAMATH_CALUDE_stair_cleaning_problem_l1898_189839

theorem stair_cleaning_problem (a b c : ℕ) (h1 : a > c) (h2 : 101 * (a + c) + 20 * b = 746) :
  let n := 100 * a + 10 * b + c
  (2 * n = 944) ∨ (2 * n = 1142) := by
  sorry

end NUMINAMATH_CALUDE_stair_cleaning_problem_l1898_189839


namespace NUMINAMATH_CALUDE_simplify_expressions_l1898_189825

variable (a b : ℝ)

theorem simplify_expressions :
  (4 * a^2 + 3 * b^2 + 2 * a * b - 4 * a^2 - 4 * b = 3 * b^2 + 2 * a * b - 4 * b) ∧
  (2 * (5 * a - 3 * b) - 3 = 10 * a - 6 * b - 3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l1898_189825


namespace NUMINAMATH_CALUDE_total_votes_is_330_l1898_189878

/-- Proves that the total number of votes is 330 given the specified conditions -/
theorem total_votes_is_330 :
  ∀ (total_votes votes_for votes_against : ℕ),
    votes_for = votes_against + 66 →
    votes_against = (40 * total_votes) / 100 →
    total_votes = votes_for + votes_against →
    total_votes = 330 := by
  sorry

end NUMINAMATH_CALUDE_total_votes_is_330_l1898_189878


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1898_189884

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop := y^2 / 9 - x^2 / 16 = 1

/-- The asymptote equation -/
def asymptote_eq (m x y : ℝ) : Prop := y = m * x ∨ y = -m * x

/-- Theorem: The value of m for the given hyperbola's asymptotes is 3/4 -/
theorem hyperbola_asymptote_slope :
  ∃ (m : ℝ), (∀ (x y : ℝ), hyperbola_eq x y → asymptote_eq m x y) ∧ m = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1898_189884


namespace NUMINAMATH_CALUDE_eighth_grade_class_problem_l1898_189895

theorem eighth_grade_class_problem (total students_math students_foreign : ℕ) 
  (h_total : total = 93)
  (h_math : students_math = 70)
  (h_foreign : students_foreign = 54) :
  students_math - (total - (students_math + students_foreign - total)) = 39 := by
  sorry

end NUMINAMATH_CALUDE_eighth_grade_class_problem_l1898_189895


namespace NUMINAMATH_CALUDE_count_four_digit_integers_eq_six_l1898_189870

def digits : Multiset ℕ := {2, 2, 9, 9}

/-- The number of different positive, four-digit integers that can be formed using the digits 2, 2, 9, and 9 -/
def count_four_digit_integers : ℕ :=
  Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2)

theorem count_four_digit_integers_eq_six :
  count_four_digit_integers = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_integers_eq_six_l1898_189870


namespace NUMINAMATH_CALUDE_bathtub_fill_time_l1898_189891

/-- Proves that a bathtub with given capacity filled by a tap with given flow rate takes the calculated time to fill -/
theorem bathtub_fill_time (bathtub_capacity : ℝ) (tap_volume : ℝ) (tap_time : ℝ) (fill_time : ℝ) 
    (h1 : bathtub_capacity = 140)
    (h2 : tap_volume = 15)
    (h3 : tap_time = 3)
    (h4 : fill_time = bathtub_capacity / (tap_volume / tap_time)) :
  fill_time = 28 := by
  sorry

end NUMINAMATH_CALUDE_bathtub_fill_time_l1898_189891


namespace NUMINAMATH_CALUDE_fence_cost_l1898_189844

/-- The cost of building a fence around a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 58) :
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_cost := perimeter * price_per_foot
  total_cost = 3944 := by sorry

end NUMINAMATH_CALUDE_fence_cost_l1898_189844


namespace NUMINAMATH_CALUDE_problem_solution_l1898_189854

theorem problem_solution : (10^3 - (270 * (1/3))) + Real.sqrt 144 = 922 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1898_189854


namespace NUMINAMATH_CALUDE_gcd_of_225_and_135_l1898_189816

theorem gcd_of_225_and_135 : Nat.gcd 225 135 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_225_and_135_l1898_189816


namespace NUMINAMATH_CALUDE_tangency_and_tangent_line_l1898_189836

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x = Real.sqrt (2 * y^2 + 25/2)
def C₂ (a x y : ℝ) : Prop := y = a * x^2

-- Define the tangency condition
def is_tangent (a : ℝ) : Prop :=
  ∃ x y : ℝ, C₁ x y ∧ C₂ a x y ∧
  (∀ x' y' : ℝ, C₁ x' y' → C₂ a x' y' → (x = x' ∧ y = y'))

-- State the theorem
theorem tangency_and_tangent_line :
  ∃ a : ℝ, a > 0 ∧ is_tangent a ∧
  (∀ x y : ℝ, C₁ x y ∧ C₂ a x y → x = 5 ∧ y = 5/2) ∧
  (∀ x y : ℝ, 2*x - 2*y - 5 = 0 ↔ (C₁ x y ∧ C₂ a x y ∨ (x = 5 ∧ y = 5/2))) :=
sorry

end NUMINAMATH_CALUDE_tangency_and_tangent_line_l1898_189836


namespace NUMINAMATH_CALUDE_eggs_per_meal_l1898_189869

def initial_eggs : ℕ := 24
def used_eggs : ℕ := 6
def meals : ℕ := 3

theorem eggs_per_meal :
  let remaining_after_use := initial_eggs - used_eggs
  let remaining_after_sharing := remaining_after_use / 2
  remaining_after_sharing / meals = 3 :=
by sorry

end NUMINAMATH_CALUDE_eggs_per_meal_l1898_189869


namespace NUMINAMATH_CALUDE_gcd_problem_l1898_189881

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 1061) :
  Int.gcd (3 * b^2 + 41 * b + 96) (b + 17) = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1898_189881


namespace NUMINAMATH_CALUDE_a_33_mod_33_l1898_189812

/-- The integer obtained by writing all integers from 1 to n sequentially -/
def a (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that a₃₃ mod 33 = 22 -/
theorem a_33_mod_33 : a 33 % 33 = 22 := by
  sorry

end NUMINAMATH_CALUDE_a_33_mod_33_l1898_189812


namespace NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l1898_189888

/-- The orthocenter of a triangle ABC in 3D space -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem: The orthocenter of triangle ABC with given coordinates is (5/2, 3, 7/2) -/
theorem orthocenter_of_specific_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (6, 4, 2)
  let C : ℝ × ℝ × ℝ := (4, 5, 6)
  orthocenter A B C = (5/2, 3, 7/2) :=
sorry

end NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l1898_189888


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1898_189834

theorem imaginary_part_of_complex_fraction :
  Complex.im ((3 * Complex.I + 4) / (1 + 2 * Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1898_189834


namespace NUMINAMATH_CALUDE_S_inter_T_eq_T_l1898_189861

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem S_inter_T_eq_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_S_inter_T_eq_T_l1898_189861


namespace NUMINAMATH_CALUDE_find_x_value_l1898_189868

theorem find_x_value (X : ℝ) (h : (X + 200 / 90) * 90 = 18200) : X = 18000 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l1898_189868


namespace NUMINAMATH_CALUDE_typing_service_problem_l1898_189899

/-- Typing service problem -/
theorem typing_service_problem 
  (total_pages : ℕ) 
  (pages_revised_twice : ℕ) 
  (cost_first_typing : ℕ) 
  (cost_per_revision : ℕ) 
  (total_cost : ℕ) 
  (h1 : total_pages = 200)
  (h2 : pages_revised_twice = 20)
  (h3 : cost_first_typing = 5)
  (h4 : cost_per_revision = 3)
  (h5 : total_cost = 1360) :
  ∃ (pages_revised_once : ℕ),
    pages_revised_once = 80 ∧
    total_cost = 
      total_pages * cost_first_typing + 
      pages_revised_once * cost_per_revision + 
      pages_revised_twice * cost_per_revision * 2 :=
by sorry

end NUMINAMATH_CALUDE_typing_service_problem_l1898_189899


namespace NUMINAMATH_CALUDE_painted_cube_equality_l1898_189810

theorem painted_cube_equality (n : ℝ) (h : n > 2) :
  12 * (n - 2) = (n - 2)^3 ↔ n = 2 + 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_painted_cube_equality_l1898_189810


namespace NUMINAMATH_CALUDE_factors_of_1320_eq_24_l1898_189862

/-- The number of distinct positive factors of 1320 -/
def factors_of_1320 : ℕ :=
  (3 : ℕ) * 2 * 2 * 2

/-- Theorem stating that the number of distinct positive factors of 1320 is 24 -/
theorem factors_of_1320_eq_24 : factors_of_1320 = 24 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_1320_eq_24_l1898_189862


namespace NUMINAMATH_CALUDE_total_miles_walked_l1898_189874

-- Define the number of islands
def num_islands : ℕ := 4

-- Define the number of days to explore each island
def days_per_island : ℚ := 3/2

-- Define the daily walking distances for each type of island
def miles_per_day_type1 : ℕ := 20
def miles_per_day_type2 : ℕ := 25

-- Define the number of islands for each type
def num_islands_type1 : ℕ := 2
def num_islands_type2 : ℕ := 2

-- Theorem to prove
theorem total_miles_walked :
  (num_islands_type1 * miles_per_day_type1 + num_islands_type2 * miles_per_day_type2) * days_per_island = 135 := by
  sorry


end NUMINAMATH_CALUDE_total_miles_walked_l1898_189874


namespace NUMINAMATH_CALUDE_clothing_percentage_is_fifty_percent_l1898_189850

/-- Represents the shopping breakdown and tax rates for Jill's purchases --/
structure ShoppingBreakdown where
  clothing_percentage : ℝ
  food_percentage : ℝ
  other_percentage : ℝ
  clothing_tax_rate : ℝ
  food_tax_rate : ℝ
  other_tax_rate : ℝ
  total_tax_rate : ℝ

/-- Calculates the percentage spent on clothing given the shopping breakdown --/
def calculate_clothing_percentage (sb : ShoppingBreakdown) : ℝ :=
  sb.clothing_percentage

/-- Theorem stating that the percentage spent on clothing is 50% --/
theorem clothing_percentage_is_fifty_percent (sb : ShoppingBreakdown) 
  (h1 : sb.food_percentage = 0.25)
  (h2 : sb.other_percentage = 0.25)
  (h3 : sb.clothing_tax_rate = 0.10)
  (h4 : sb.food_tax_rate = 0)
  (h5 : sb.other_tax_rate = 0.20)
  (h6 : sb.total_tax_rate = 0.10)
  (h7 : sb.clothing_percentage + sb.food_percentage + sb.other_percentage = 1) :
  calculate_clothing_percentage sb = 0.5 := by
  sorry

#eval calculate_clothing_percentage { 
  clothing_percentage := 0.5,
  food_percentage := 0.25,
  other_percentage := 0.25,
  clothing_tax_rate := 0.10,
  food_tax_rate := 0,
  other_tax_rate := 0.20,
  total_tax_rate := 0.10
}

end NUMINAMATH_CALUDE_clothing_percentage_is_fifty_percent_l1898_189850


namespace NUMINAMATH_CALUDE_symmetric_partitions_generating_function_main_theorem_l1898_189828

/-- A partition is a non-increasing sequence of positive integers. -/
def Partition := List Nat

/-- A partition is symmetric if its Ferrers diagram is symmetric with respect to the diagonal. -/
def IsSymmetric (p : Partition) : Prop := sorry

/-- A partition consists of distinct odd parts if all its parts are odd and unique. -/
def HasDistinctOddParts (p : Partition) : Prop := sorry

/-- The generating function for partitions with a given property. -/
noncomputable def GeneratingFunction (P : Partition → Prop) : ℕ → ℚ := sorry

/-- The infinite product ∏_{k=1}^{∞} (1 + x^(2k+1)) -/
noncomputable def InfiniteProduct : ℕ → ℚ := sorry

theorem symmetric_partitions_generating_function :
  GeneratingFunction IsSymmetric = GeneratingFunction HasDistinctOddParts :=
by sorry

theorem main_theorem :
  GeneratingFunction IsSymmetric = InfiniteProduct :=
by sorry

end NUMINAMATH_CALUDE_symmetric_partitions_generating_function_main_theorem_l1898_189828


namespace NUMINAMATH_CALUDE_incorrect_negation_even_multiple_of_seven_l1898_189882

theorem incorrect_negation_even_multiple_of_seven :
  ¬(∀ n : ℕ, ¬(2 * n % 7 = 0)) ↔ ∃ n : ℕ, 2 * n % 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_incorrect_negation_even_multiple_of_seven_l1898_189882


namespace NUMINAMATH_CALUDE_lcm_18_10_l1898_189883

theorem lcm_18_10 : Nat.lcm 18 10 = 36 :=
by
  have h1 : Nat.gcd 18 10 = 5 := by sorry
  sorry

end NUMINAMATH_CALUDE_lcm_18_10_l1898_189883


namespace NUMINAMATH_CALUDE_wicket_keeper_age_difference_l1898_189849

theorem wicket_keeper_age_difference (team_size : ℕ) (team_avg_age : ℝ) (remaining_players : ℕ) (age_difference : ℝ) :
  team_size = 11 →
  team_avg_age = 21 →
  remaining_players = 9 →
  age_difference = 1 →
  let total_age := team_size * team_avg_age
  let remaining_avg_age := team_avg_age - age_difference
  let remaining_total_age := remaining_players * remaining_avg_age
  let wicket_keeper_age := total_age - (remaining_total_age + team_avg_age)
  wicket_keeper_age - team_avg_age = 9 := by
  sorry

end NUMINAMATH_CALUDE_wicket_keeper_age_difference_l1898_189849
