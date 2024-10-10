import Mathlib

namespace power_of_two_equality_l4106_410612

theorem power_of_two_equality (x : ℕ) : (1 / 16 : ℝ) * (2 ^ 50) = 2 ^ x → x = 46 := by
  sorry

end power_of_two_equality_l4106_410612


namespace polynomial_division_l4106_410677

theorem polynomial_division (x : ℝ) :
  8 * x^3 - 2 * x^2 + 4 * x - 9 = (x - 3) * (8 * x^2 + 22 * x + 70) + 201 := by
  sorry

end polynomial_division_l4106_410677


namespace union_complement_equals_set_l4106_410628

def U : Finset Nat := {0, 1, 2, 4, 6, 8}
def M : Finset Nat := {0, 4, 6}
def N : Finset Nat := {0, 1, 6}

theorem union_complement_equals_set : M ∪ (U \ N) = {0, 2, 4, 6, 8} := by sorry

end union_complement_equals_set_l4106_410628


namespace max_distance_to_line_family_l4106_410669

/-- The maximum distance from a point to a family of lines --/
theorem max_distance_to_line_family (a : ℝ) : 
  let P : ℝ × ℝ := (1, -1)
  let line := {(x, y) : ℝ × ℝ | a * x + 3 * y + 2 * a - 6 = 0}
  ∃ (Q : ℝ × ℝ), Q ∈ line ∧ 
    ∀ (R : ℝ × ℝ), R ∈ line → 
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≥ Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  ∧ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 3 * Real.sqrt 2 :=
by
  sorry


end max_distance_to_line_family_l4106_410669


namespace parallel_lines_a_value_l4106_410608

/-- Two lines in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- Check if two lines are coincident -/
def are_coincident (l1 l2 : Line) : Prop :=
  are_parallel l1 l2 ∧ l1.a * l2.c = l2.a * l1.c

/-- The problem statement -/
theorem parallel_lines_a_value :
  ∃ (a : ℝ), 
    let l1 : Line := ⟨a, 3, 1⟩
    let l2 : Line := ⟨2, a+1, 1⟩
    are_parallel l1 l2 ∧ ¬are_coincident l1 l2 ∧ a = -3 := by
  sorry

end parallel_lines_a_value_l4106_410608


namespace valid_pairs_l4106_410691

def is_valid_pair (a b : ℕ+) : Prop :=
  (∃ k : ℤ, (a.val ^ 3 * b.val - 1) = k * (a.val + 1)) ∧
  (∃ m : ℤ, (b.val ^ 3 * a.val + 1) = m * (b.val - 1))

theorem valid_pairs :
  ∀ a b : ℕ+, is_valid_pair a b →
    ((a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3)) :=
by sorry

end valid_pairs_l4106_410691


namespace sum_reciprocal_squares_cubic_l4106_410661

theorem sum_reciprocal_squares_cubic (a b c : ℝ) : 
  a^3 - 12*a^2 + 20*a - 3 = 0 →
  b^3 - 12*b^2 + 20*b - 3 = 0 →
  c^3 - 12*c^2 + 20*c - 3 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 328/9 := by
  sorry

end sum_reciprocal_squares_cubic_l4106_410661


namespace tire_circumference_constant_l4106_410605

/-- The circumference of a tire remains constant given car speed and tire rotation rate -/
theorem tire_circumference_constant
  (v : ℝ) -- Car speed in km/h
  (n : ℝ) -- Tire rotation rate in rpm
  (h1 : v = 120) -- Car speed is 120 km/h
  (h2 : n = 400) -- Tire rotation rate is 400 rpm
  : ∃ (C : ℝ), C = 5 ∧ ∀ (grade : ℝ), C = 5 := by
  sorry

end tire_circumference_constant_l4106_410605


namespace point_on_hyperbola_l4106_410699

/-- A point (x, y) lies on the hyperbola y = -4/x if and only if xy = -4 -/
def lies_on_hyperbola (x y : ℝ) : Prop := x * y = -4

/-- The point (-2, 2) lies on the hyperbola y = -4/x -/
theorem point_on_hyperbola : lies_on_hyperbola (-2) 2 := by sorry

end point_on_hyperbola_l4106_410699


namespace sine_graph_transformation_l4106_410685

theorem sine_graph_transformation (x : ℝ) : 
  4 * Real.sin (2 * x + π / 5) = 4 * Real.sin ((2 * x / 2) + π / 5) := by
  sorry

end sine_graph_transformation_l4106_410685


namespace gcd_power_minus_one_l4106_410693

theorem gcd_power_minus_one (m n : ℕ+) :
  Nat.gcd ((2 : ℕ) ^ m.val - 1) ((2 : ℕ) ^ n.val - 1) = (2 : ℕ) ^ (Nat.gcd m.val n.val) - 1 := by
  sorry

end gcd_power_minus_one_l4106_410693


namespace smallest_upper_bound_for_triangle_ratio_l4106_410697

/-- Triangle sides -/
structure Triangle :=
  (a b c : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (triangle_ineq_ab : c < a + b)
  (triangle_ineq_bc : a < b + c)
  (triangle_ineq_ca : b < c + a)
  (a_neq_b : a ≠ b)

/-- The smallest upper bound for (a² + b²) / c² in any triangle with unequal sides -/
theorem smallest_upper_bound_for_triangle_ratio :
  ∃ N : ℝ, (∀ t : Triangle, (t.a^2 + t.b^2) / t.c^2 < N) ∧
           (∀ ε > 0, ∃ t : Triangle, N - ε < (t.a^2 + t.b^2) / t.c^2) :=
sorry

end smallest_upper_bound_for_triangle_ratio_l4106_410697


namespace fruit_salad_problem_l4106_410614

/-- Fruit salad problem -/
theorem fruit_salad_problem (green_grapes red_grapes raspberries blueberries pineapple : ℕ) :
  red_grapes = 3 * green_grapes + 7 →
  raspberries = green_grapes - 5 →
  blueberries = 4 * raspberries →
  pineapple = blueberries / 2 + 5 →
  green_grapes + red_grapes + raspberries + blueberries + pineapple = 350 →
  red_grapes = 100 := by
  sorry

#check fruit_salad_problem

end fruit_salad_problem_l4106_410614


namespace min_abs_sum_l4106_410648

open Complex

variable (α γ : ℂ)

def f (z : ℂ) : ℂ := (2 + 3*I)*z^2 + α*z + γ

theorem min_abs_sum (h1 : (f α γ 1).im = 0) (h2 : (f α γ I).im = 0) :
  ∃ (α₀ γ₀ : ℂ), (abs α₀ + abs γ₀ = 3) ∧ 
    ∀ (α' γ' : ℂ), (f α' γ' 1).im = 0 → (f α' γ' I).im = 0 → abs α' + abs γ' ≥ 3 :=
sorry

end min_abs_sum_l4106_410648


namespace scientific_notation_of_population_l4106_410660

theorem scientific_notation_of_population (population : ℝ) : 
  population = 2184.3 * 1000000 → 
  ∃ (a : ℝ) (n : ℤ), population = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.1843 ∧ n = 7 :=
by sorry

end scientific_notation_of_population_l4106_410660


namespace base10_89_equals_base5_324_l4106_410611

/-- Converts a natural number to its base-5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Converts a list of digits in base 5 to a natural number --/
def fromBase5 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 5 * acc + d) 0

theorem base10_89_equals_base5_324 : fromBase5 [4, 2, 3] = 89 := by
  sorry

end base10_89_equals_base5_324_l4106_410611


namespace intersection_point_y_coordinate_l4106_410635

-- Define the parabola
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Define the slope of the tangent at a point
def tangent_slope (x : ℝ) : ℝ := 8 * x

-- Define the condition for perpendicular tangents
def perpendicular_tangents (a b : ℝ) : Prop :=
  tangent_slope a * tangent_slope b = -1

-- Define the y-coordinate of the intersection point
def intersection_y (a b : ℝ) : ℝ := 4 * a * b

-- Theorem statement
theorem intersection_point_y_coordinate 
  (a b : ℝ) 
  (ha : parabola a = 4 * a^2) 
  (hb : parabola b = 4 * b^2) 
  (hperp : perpendicular_tangents a b) :
  intersection_y a b = -1/4 := by sorry

end intersection_point_y_coordinate_l4106_410635


namespace johns_initial_money_l4106_410604

theorem johns_initial_money (initial_amount : ℚ) : 
  (initial_amount * (1 - (3/8 + 3/10)) = 65) → initial_amount = 200 := by
  sorry

end johns_initial_money_l4106_410604


namespace range_of_a_l4106_410631

theorem range_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 1/x + 2/y = 1) (h_ineq : ∀ x y, x > 0 → y > 0 → 1/x + 2/y = 1 → x + 2*y > a^2 + 8*a) : 
  -4 - 2*Real.sqrt 6 < a ∧ a < -4 + 2*Real.sqrt 6 := by
  sorry

end range_of_a_l4106_410631


namespace exists_permutation_equals_sixteen_l4106_410671

-- Define the set of operations
inductive Operation
  | Div : Operation
  | Add : Operation
  | Mul : Operation

-- Define a function to apply an operation
def applyOp (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Div => a / b
  | Operation.Add => a + b
  | Operation.Mul => a * b

-- Define a function to evaluate the expression given a permutation of operations
def evaluate (ops : List Operation) : ℚ :=
  match ops with
  | [op1, op2, op3] => applyOp op3 (applyOp op2 (applyOp op1 8 2) 3) 4
  | _ => 0  -- Invalid permutation

-- Theorem statement
theorem exists_permutation_equals_sixteen :
  ∃ (ops : List Operation),
    (ops.length = 3) ∧
    (Operation.Div ∈ ops) ∧
    (Operation.Add ∈ ops) ∧
    (Operation.Mul ∈ ops) ∧
    (evaluate ops = 16) := by
  sorry

end exists_permutation_equals_sixteen_l4106_410671


namespace correct_statements_l4106_410665

-- Define the data sets
def dataSetA : List ℝ := sorry
def dataSetB : List ℝ := sorry
def dataSetC : List ℝ := [1, 2, 5, 5, 5, 3, 3]

-- Define variance function
def variance (data : List ℝ) : ℝ := sorry

-- Define median function
def median (data : List ℝ) : ℝ := sorry

-- Define mode function
def mode (data : List ℝ) : ℝ := sorry

-- Theorem to prove
theorem correct_statements :
  -- Statement A is incorrect
  ¬ (∀ (n : ℕ), n = 1000 → ∃ (h : ℕ), h = 500 ∧ h = n / 2) ∧
  -- Statement B is correct
  (variance dataSetA = 0.03 ∧ variance dataSetB = 0.1 → variance dataSetA < variance dataSetB) ∧
  -- Statement C is correct
  (median dataSetC = 3 ∧ mode dataSetC = 5) ∧
  -- Statement D is incorrect
  ¬ (∀ (population : Type) (property : population → Prop),
     (∀ x : population, property x) ↔ (∃ survey : population → Prop, ∀ x, survey x → property x))
  := by sorry

end correct_statements_l4106_410665


namespace zoo_animals_l4106_410696

theorem zoo_animals (sea_lions : ℕ) (penguins : ℕ) : 
  sea_lions = 48 →
  sea_lions * 11 = penguins * 4 →
  penguins > sea_lions →
  penguins - sea_lions = 84 := by
sorry

end zoo_animals_l4106_410696


namespace toluene_formation_l4106_410652

-- Define the chemical species involved in the reaction
structure ChemicalSpecies where
  formula : String
  moles : ℝ

-- Define the chemical reaction
def reaction (reactant1 reactant2 product1 product2 : ChemicalSpecies) : Prop :=
  reactant1.formula = "C6H6" ∧ 
  reactant2.formula = "CH4" ∧ 
  product1.formula = "C6H5CH3" ∧ 
  product2.formula = "H2" ∧
  reactant1.moles = reactant2.moles ∧
  product1.moles = product2.moles ∧
  reactant1.moles = product1.moles

-- Theorem statement
theorem toluene_formation 
  (benzene : ChemicalSpecies)
  (methane : ChemicalSpecies)
  (toluene : ChemicalSpecies)
  (hydrogen : ChemicalSpecies)
  (h1 : reaction benzene methane toluene hydrogen)
  (h2 : methane.moles = 3)
  (h3 : hydrogen.moles = 3) :
  toluene.moles = 3 :=
sorry

end toluene_formation_l4106_410652


namespace complex_sum_problem_l4106_410649

theorem complex_sum_problem (u v w x y z : ℂ) : 
  v = 2 →
  y = -u - w →
  u + v * I + w + x * I + y + z * I = 2 * I →
  x + z = 0 := by
sorry

end complex_sum_problem_l4106_410649


namespace roll_one_probability_l4106_410692

/-- A fair six-sided die -/
structure FairDie :=
  (sides : Fin 6)

/-- The probability of rolling a specific number on a fair six-sided die -/
def roll_probability (d : FairDie) (n : Fin 6) : ℚ := 1 / 6

/-- The independence of die rolls -/
axiom roll_independence (d : FairDie) (n m : Fin 6) : 
  roll_probability d n = roll_probability d n

/-- Theorem: The probability of rolling a 1 on a fair six-sided die is 1/6 -/
theorem roll_one_probability (d : FairDie) : 
  roll_probability d 0 = 1 / 6 := by sorry

end roll_one_probability_l4106_410692


namespace fractional_parts_sum_l4106_410602

theorem fractional_parts_sum (x : ℝ) (h : x^3 + 1/x^3 = 18) :
  (x - ⌊x⌋) + (1/x - ⌊1/x⌋) = 1 := by
  sorry

end fractional_parts_sum_l4106_410602


namespace cost_of_tea_cake_eclair_l4106_410647

/-- Given the costs of tea and a cake, tea and an éclair, and a cake and an éclair,
    prove that the sum of the costs of tea, a cake, and an éclair
    is equal to half the sum of all three given costs. -/
theorem cost_of_tea_cake_eclair
  (t c e : ℝ)  -- t: cost of tea, c: cost of cake, e: cost of éclair
  (h1 : t + c = 4.5)  -- cost of tea and cake
  (h2 : t + e = 4)    -- cost of tea and éclair
  (h3 : c + e = 6.5)  -- cost of cake and éclair
  : t + c + e = (4.5 + 4 + 6.5) / 2 :=
by sorry

end cost_of_tea_cake_eclair_l4106_410647


namespace power_function_value_l4106_410629

-- Define a power function that passes through (1/2, √2/2)
def f (x : ℝ) : ℝ := x^(1/2)

-- State the theorem
theorem power_function_value : f (1/4) = 1/2 := by
  sorry

end power_function_value_l4106_410629


namespace combination_permutation_equality_l4106_410626

theorem combination_permutation_equality (n : ℕ) (hn : n > 0) : 
  3 * (Nat.choose (2 * n) 3) = 5 * (Nat.factorial n / Nat.factorial (n - 3)) → n = 8 := by
  sorry

end combination_permutation_equality_l4106_410626


namespace arrangement_count_l4106_410638

/-- The number of ways to arrange 15 letters (5 A's, 5 B's, 5 C's) with restrictions -/
def restricted_arrangements : ℕ :=
  Finset.sum (Finset.range 6) (fun k => (Nat.choose 5 k) ^ 3)

/-- The conditions of the problem -/
theorem arrangement_count :
  restricted_arrangements =
    (Finset.sum (Finset.range 6) (fun k =>
      /- Number of ways to arrange k A's and (5-k) C's in the first 5 positions -/
      (Nat.choose 5 k) *
      /- Number of ways to arrange k B's and (5-k) A's in the middle 5 positions -/
      (Nat.choose 5 k) *
      /- Number of ways to arrange k C's and (5-k) B's in the last 5 positions -/
      (Nat.choose 5 k))) :=
by
  sorry

#eval restricted_arrangements

end arrangement_count_l4106_410638


namespace project_completion_time_l4106_410619

/-- The number of days A and B take working together -/
def AB_days : ℝ := 2

/-- The number of days B and C take working together -/
def BC_days : ℝ := 4

/-- The number of days C and A take working together -/
def CA_days : ℝ := 2.4

/-- The number of days A takes to complete the project alone -/
def A_days : ℝ := 3

theorem project_completion_time :
  (1 / A_days) * CA_days + (1 / BC_days - (1 / AB_days - 1 / A_days)) * CA_days = 1 :=
sorry

end project_completion_time_l4106_410619


namespace book_selection_theorem_l4106_410694

def number_of_ways_to_select_books (total_books : ℕ) (identical_books : ℕ) (different_books : ℕ) (books_to_select : ℕ) : ℕ :=
  -- Select all identical books
  (if books_to_select ≤ identical_books then 1 else 0) +
  -- Select some identical books and some different books
  (Finset.sum (Finset.range (min identical_books books_to_select + 1)) (fun i =>
    Nat.choose identical_books i * Nat.choose different_books (books_to_select - i)))

theorem book_selection_theorem :
  number_of_ways_to_select_books 9 3 6 3 = 42 := by
  sorry

end book_selection_theorem_l4106_410694


namespace weighted_arithmetic_geometric_mean_inequality_l4106_410681

theorem weighted_arithmetic_geometric_mean_inequality 
  {n : ℕ} (a w : Fin n → ℝ) (h_pos_a : ∀ i, a i > 0) (h_pos_w : ∀ i, w i > 0) :
  let W := (Finset.univ.sum w)
  (W⁻¹ * Finset.univ.sum (λ i => w i * a i)) ≥ 
    (Finset.univ.prod (λ i => (a i) ^ (w i))) ^ (W⁻¹) := by
  sorry

end weighted_arithmetic_geometric_mean_inequality_l4106_410681


namespace absolute_value_equality_l4106_410633

theorem absolute_value_equality (x y : ℝ) : 
  |x - Real.sqrt y| = x + Real.sqrt y → x = 0 ∧ y = 0 := by
  sorry

end absolute_value_equality_l4106_410633


namespace other_number_proof_l4106_410683

/-- Given two positive integers with specific HCF and LCM, prove that if one number is 36, the other is 154 -/
theorem other_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 14) (h2 : Nat.lcm a b = 396) (h3 : a = 36) : b = 154 := by
  sorry

end other_number_proof_l4106_410683


namespace optimal_layoffs_maximizes_benefit_l4106_410679

/-- Represents the number of employees to lay off for maximum economic benefit -/
def optimal_layoffs (a : ℕ) : ℚ :=
  if 70 < a ∧ a ≤ 140 then a - 70
  else if 140 < a ∧ a < 210 then a / 2
  else 0

theorem optimal_layoffs_maximizes_benefit (a b : ℕ) :
  140 < 2 * a ∧ 2 * a < 420 ∧ 
  ∃ k, a = 2 * k ∧
  (∀ x : ℚ, 0 < x ∧ x ≤ a / 2 →
    ((2 * a - x) * (b + 0.01 * b * x) - 0.4 * b * x) ≤
    ((2 * a - optimal_layoffs a) * (b + 0.01 * b * optimal_layoffs a) - 0.4 * b * optimal_layoffs a)) :=
by sorry

end optimal_layoffs_maximizes_benefit_l4106_410679


namespace f_geq_6_iff_l4106_410664

def f (x : ℝ) := |x + 1| + |2*x - 4|

theorem f_geq_6_iff (x : ℝ) : f x ≥ 6 ↔ x ≤ -1 ∨ x ≥ 3 := by sorry

end f_geq_6_iff_l4106_410664


namespace chairs_remaining_l4106_410623

theorem chairs_remaining (initial_chairs : ℕ) (difference : ℕ) (remaining_chairs : ℕ) : 
  initial_chairs = 15 → 
  difference = 12 → 
  initial_chairs - remaining_chairs = difference →
  remaining_chairs = 3 := by
  sorry

end chairs_remaining_l4106_410623


namespace justin_bought_two_striped_jerseys_l4106_410650

-- Define the cost of each type of jersey
def long_sleeved_cost : ℕ := 15
def striped_cost : ℕ := 10

-- Define the number of long-sleeved jerseys bought
def long_sleeved_count : ℕ := 4

-- Define the total amount spent
def total_spent : ℕ := 80

-- Define the number of striped jerseys as a function
def striped_jerseys : ℕ := (total_spent - long_sleeved_cost * long_sleeved_count) / striped_cost

-- Theorem to prove
theorem justin_bought_two_striped_jerseys : striped_jerseys = 2 := by
  sorry

end justin_bought_two_striped_jerseys_l4106_410650


namespace philips_farm_l4106_410670

theorem philips_farm (cows ducks pigs : ℕ) : 
  ducks = (3 * cows) / 2 →
  pigs = (cows + ducks) / 5 →
  cows + ducks + pigs = 60 →
  cows = 20 := by
sorry

end philips_farm_l4106_410670


namespace complete_square_sum_l4106_410666

theorem complete_square_sum (a b c : ℤ) : 
  (∀ x : ℝ, 25 * x^2 + 30 * x - 75 = 0 ↔ (a * x + b)^2 = c) →
  a > 0 →
  a + b + c = -58 :=
by sorry

end complete_square_sum_l4106_410666


namespace square_sum_equation_l4106_410688

theorem square_sum_equation (x y : ℝ) (h1 : x + 2*y = 8) (h2 : x*y = 1) : x^2 + 4*y^2 = 60 := by
  sorry

end square_sum_equation_l4106_410688


namespace polynomial_integer_values_l4106_410658

/-- A polynomial of degree 3 with real coefficients -/
def Polynomial3 := ℝ → ℝ

/-- Predicate to check if a number is an integer -/
def IsInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- Main theorem: If a polynomial of degree 3 takes integer values at four consecutive integers,
    then it takes integer values at all integers -/
theorem polynomial_integer_values (P : Polynomial3) (i : ℤ) 
  (h1 : IsInteger (P i))
  (h2 : IsInteger (P (i + 1)))
  (h3 : IsInteger (P (i + 2)))
  (h4 : IsInteger (P (i + 3))) :
  ∀ n : ℤ, IsInteger (P n) := by
  sorry

end polynomial_integer_values_l4106_410658


namespace quadrilateral_perimeter_l4106_410690

/-- The perimeter of a quadrilateral with sides x, x + 1, 6, and 10, where x = 3, is 23. -/
theorem quadrilateral_perimeter (x : ℝ) (h : x = 3) : x + (x + 1) + 6 + 10 = 23 := by
  sorry

#check quadrilateral_perimeter

end quadrilateral_perimeter_l4106_410690


namespace maximize_x5y2_l4106_410634

theorem maximize_x5y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 35) :
  x^5 * y^2 ≤ 25^5 * 10^2 ∧ 
  (x^5 * y^2 = 25^5 * 10^2 ↔ x = 25 ∧ y = 10) :=
by sorry

end maximize_x5y2_l4106_410634


namespace first_month_sale_l4106_410646

def last_four_months_sales : List Int := [5660, 6200, 6350, 6500]
def sixth_month_sale : Int := 8270
def average_sale : Int := 6400
def num_months : Int := 6

theorem first_month_sale :
  (num_months * average_sale) - (sixth_month_sale + last_four_months_sales.sum) = 5420 := by
  sorry

end first_month_sale_l4106_410646


namespace prob_at_least_one_2_or_3_is_7_16_l4106_410639

/-- The probability of at least one of two fair 8-sided dice showing a 2 or a 3 -/
def prob_at_least_one_2_or_3 : ℚ := 7/16

/-- Two fair 8-sided dice are rolled -/
axiom fair_8_sided_dice : True

theorem prob_at_least_one_2_or_3_is_7_16 :
  prob_at_least_one_2_or_3 = 7/16 :=
sorry

end prob_at_least_one_2_or_3_is_7_16_l4106_410639


namespace women_in_luxury_suite_l4106_410678

def total_passengers : ℕ := 300
def women_percentage : ℚ := 1/2
def luxury_suite_percentage : ℚ := 3/20

theorem women_in_luxury_suite :
  ⌊(total_passengers : ℚ) * women_percentage * luxury_suite_percentage⌋ = 23 := by
  sorry

end women_in_luxury_suite_l4106_410678


namespace average_weight_increase_l4106_410620

/-- Proves that replacing a person weighing 70 kg with a person weighing 90 kg
    in a group of 8 people increases the average weight by 2.5 kg. -/
theorem average_weight_increase
  (n : ℕ)
  (old_weight new_weight : ℝ)
  (h_n : n = 8)
  (h_old : old_weight = 70)
  (h_new : new_weight = 90) :
  (new_weight - old_weight) / n = 2.5 := by
  sorry

end average_weight_increase_l4106_410620


namespace expand_polynomial_l4106_410656

theorem expand_polynomial (x y : ℝ) : 
  (1 + x^2 + y^3) * (1 - x^3 - y^3) = 1 + x^2 - x^3 - y^3 - x^5 - x^2 * y^3 - x^3 * y^3 - y^6 := by
  sorry

end expand_polynomial_l4106_410656


namespace sector_angle_l4106_410615

/-- Given a sector of a circle with perimeter 8 and area 4, 
    prove that the absolute value of its central angle in radians is 2 -/
theorem sector_angle (r l θ : ℝ) 
  (h_perimeter : 2 * r + l = 8)
  (h_area : (1 / 2) * l * r = 4)
  (h_angle : θ = l / r)
  (h_positive : r > 0) : 
  |θ| = 2 := by
sorry

end sector_angle_l4106_410615


namespace hexagon_wire_problem_l4106_410600

/-- Calculates the remaining wire length after creating a regular hexagon. -/
def remaining_wire_length (total_wire : ℝ) (hexagon_side : ℝ) : ℝ :=
  total_wire - 6 * hexagon_side

/-- Proves that given a wire of 50 cm and a regular hexagon with side length 8 cm, 
    the remaining wire length is 2 cm. -/
theorem hexagon_wire_problem :
  remaining_wire_length 50 8 = 2 := by
  sorry

end hexagon_wire_problem_l4106_410600


namespace romeo_chocolate_profit_l4106_410632

theorem romeo_chocolate_profit :
  let num_bars : ℕ := 20
  let cost_per_bar : ℕ := 8
  let total_sales : ℕ := 240
  let packaging_cost_per_bar : ℕ := 3
  let advertising_cost : ℕ := 15
  
  let total_cost : ℕ := num_bars * cost_per_bar + num_bars * packaging_cost_per_bar + advertising_cost
  let profit : ℤ := total_sales - total_cost
  
  profit = 5 :=
by
  sorry


end romeo_chocolate_profit_l4106_410632


namespace circle_line_no_intersection_l4106_410601

theorem circle_line_no_intersection (b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 2 → y ≠ x + b) ↔ (b > 2 ∨ b < -2) :=
by sorry

end circle_line_no_intersection_l4106_410601


namespace equation_roots_existence_l4106_410674

-- Define the equation
def equation (x a : ℝ) : Prop := |x^2 - x| - a = 0

-- Define the number of different real roots for a given 'a'
def num_roots (a : ℝ) : ℕ := sorry

-- Theorem statement
theorem equation_roots_existence :
  (∃ a : ℝ, num_roots a = 2) ∧
  (∃ a : ℝ, num_roots a = 3) ∧
  (∃ a : ℝ, num_roots a = 4) ∧
  (¬ ∃ a : ℝ, num_roots a = 6) :=
sorry

end equation_roots_existence_l4106_410674


namespace no_abc_divisible_by_9_l4106_410668

theorem no_abc_divisible_by_9 :
  ∀ (a b c : ℤ), ∃ (x : ℤ), ¬ (9 ∣ ((x + a) * (x + b) * (x + c) - x^3 - 1)) :=
by sorry

end no_abc_divisible_by_9_l4106_410668


namespace geometric_sequence_constant_l4106_410675

/-- Represents a geometric sequence with sum S_n = 3^n + a -/
structure GeometricSequence where
  a : ℝ  -- The constant term in the sum formula
  -- Sequence definition: a_n = S_n - S_{n-1}
  seq : ℕ → ℝ := λ n => 3^n + a - (3^(n-1) + a)

/-- The first term of the sequence is 2 -/
axiom first_term (s : GeometricSequence) : s.seq 1 = 2

/-- The common ratio of the sequence is 3 -/
axiom common_ratio (s : GeometricSequence) : s.seq 2 = 3 * s.seq 1

/-- Theorem: The value of 'a' in the sum formula S_n = 3^n + a is -1 -/
theorem geometric_sequence_constant (s : GeometricSequence) : s.a = -1 := by
  sorry

end geometric_sequence_constant_l4106_410675


namespace initial_men_correct_l4106_410603

/-- The initial number of men working -/
def initial_men : ℕ := 450

/-- The number of hours worked per day initially -/
def initial_hours : ℕ := 8

/-- The depth dug initially in meters -/
def initial_depth : ℕ := 40

/-- The new depth to be dug in meters -/
def new_depth : ℕ := 50

/-- The new number of hours worked per day -/
def new_hours : ℕ := 6

/-- The number of extra men needed for the new task -/
def extra_men : ℕ := 30

/-- Theorem stating that the initial number of men is correct -/
theorem initial_men_correct :
  initial_men * initial_hours * initial_depth = (initial_men + extra_men) * new_hours * new_depth :=
by sorry

end initial_men_correct_l4106_410603


namespace contradictory_statements_l4106_410622

theorem contradictory_statements (a b c : ℝ) : 
  (a * b * c ≠ 0) ∧ (a * b * c = 0) ∧ (a * b ≤ 0) → False :=
sorry

end contradictory_statements_l4106_410622


namespace percentage_born_in_july_l4106_410627

theorem percentage_born_in_july (total : ℕ) (born_in_july : ℕ) 
  (h1 : total = 120) (h2 : born_in_july = 18) : 
  (born_in_july : ℚ) / (total : ℚ) * 100 = 15 := by
  sorry

end percentage_born_in_july_l4106_410627


namespace interest_rate_difference_l4106_410673

theorem interest_rate_difference 
  (principal : ℝ) 
  (time : ℝ) 
  (interest_difference : ℝ) 
  (h1 : principal = 750) 
  (h2 : time = 2) 
  (h3 : interest_difference = 60) : 
  ∃ (original_rate higher_rate : ℝ),
    principal * higher_rate * time / 100 - principal * original_rate * time / 100 = interest_difference ∧ 
    higher_rate - original_rate = 4 := by
sorry

end interest_rate_difference_l4106_410673


namespace yellow_yarns_count_l4106_410689

/-- The number of scarves that can be made from one yarn -/
def scarves_per_yarn : ℕ := 3

/-- The number of red yarns May bought -/
def red_yarns : ℕ := 2

/-- The number of blue yarns May bought -/
def blue_yarns : ℕ := 6

/-- The total number of scarves May can make -/
def total_scarves : ℕ := 36

/-- The number of yellow yarns May bought -/
def yellow_yarns : ℕ := (total_scarves - (red_yarns + blue_yarns) * scarves_per_yarn) / scarves_per_yarn

theorem yellow_yarns_count : yellow_yarns = 4 := by
  sorry

end yellow_yarns_count_l4106_410689


namespace unique_equidistant_point_l4106_410654

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
def Point := ℝ × ℝ

/-- The distance between a point and a line -/
def distancePointToLine (p : Point) (l : Line) : ℝ :=
  sorry

/-- The distance between a point and a circle -/
def distancePointToCircle (p : Point) (c : Circle) : ℝ :=
  sorry

/-- Checks if two lines are parallel -/
def areParallel (l1 l2 : Line) : Prop :=
  sorry

/-- Theorem stating that there is exactly one point equidistant from a circle
    and two parallel tangents under specific conditions -/
theorem unique_equidistant_point
  (c : Circle)
  (l1 l2 : Line)
  (h1 : c.radius = 4)
  (h2 : areParallel l1 l2)
  (h3 : distancePointToLine c.center l1 = 6)
  (h4 : distancePointToLine c.center l2 = 6) :
  ∃! p : Point,
    distancePointToCircle p c = distancePointToLine p l1 ∧
    distancePointToCircle p c = distancePointToLine p l2 :=
  sorry

end unique_equidistant_point_l4106_410654


namespace roots_transformation_l4106_410636

theorem roots_transformation (p q r : ℂ) : 
  (p^3 - 4*p^2 + 5*p + 2 = 0) ∧ 
  (q^3 - 4*q^2 + 5*q + 2 = 0) ∧ 
  (r^3 - 4*r^2 + 5*r + 2 = 0) →
  ((3*p)^3 - 12*(3*p)^2 + 45*(3*p) + 54 = 0) ∧
  ((3*q)^3 - 12*(3*q)^2 + 45*(3*q) + 54 = 0) ∧
  ((3*r)^3 - 12*(3*r)^2 + 45*(3*r) + 54 = 0) :=
by sorry

end roots_transformation_l4106_410636


namespace banana_pear_ratio_l4106_410625

theorem banana_pear_ratio : 
  ∀ (dishes bananas pears : ℕ),
  dishes = 160 →
  pears = 50 →
  dishes = bananas + 10 →
  ∃ k : ℕ, bananas = k * pears →
  bananas / pears = 3 := by
sorry

end banana_pear_ratio_l4106_410625


namespace polynomial_value_l4106_410642

def star (x y : ℤ) : ℤ := (x + 1) * (y + 1)

def star_square (x : ℤ) : ℤ := star x x

theorem polynomial_value : 
  let x := 2
  3 * (star_square x) - 2 * x + 1 = 32 := by sorry

end polynomial_value_l4106_410642


namespace count_perfect_square_factors_l4106_410657

/-- The number of positive perfect square factors of (2^14)(3^18)(7^21) -/
def perfect_square_factors : ℕ := sorry

/-- The given number -/
def given_number : ℕ := 2^14 * 3^18 * 7^21

theorem count_perfect_square_factors :
  perfect_square_factors = 880 :=
sorry

end count_perfect_square_factors_l4106_410657


namespace largest_divisor_five_consecutive_integers_l4106_410667

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, ∃ m : ℤ, m > 24 ∧ ¬(m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
  ∀ k : ℤ, k ≤ 24 → k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by sorry

end largest_divisor_five_consecutive_integers_l4106_410667


namespace intersection_sum_l4106_410640

def M : Set ℝ := {x | x^2 - 4*x < 0}
def N (m : ℝ) : Set ℝ := {x | m < x ∧ x < 5}

theorem intersection_sum (m n : ℝ) : 
  M ∩ N m = {x | 3 < x ∧ x < n} → m + n = 7 := by
  sorry

end intersection_sum_l4106_410640


namespace tickets_won_later_l4106_410695

/-- Given Cody's initial tickets, tickets spent on a beanie, and final ticket count,
    prove the number of tickets he won later. -/
theorem tickets_won_later
  (initial_tickets : ℕ)
  (tickets_spent : ℕ)
  (final_tickets : ℕ)
  (h1 : initial_tickets = 49)
  (h2 : tickets_spent = 25)
  (h3 : final_tickets = 30) :
  final_tickets - (initial_tickets - tickets_spent) = 6 := by
  sorry

end tickets_won_later_l4106_410695


namespace supermarket_spending_difference_l4106_410606

/-- 
Given:
- initial_amount: The initial amount in Olivia's wallet
- atm_amount: The amount collected from the ATM
- final_amount: The amount left after visiting the supermarket

Prove that the difference between the amount spent at the supermarket
and the amount collected from the ATM is 39 dollars.
-/
theorem supermarket_spending_difference 
  (initial_amount atm_amount final_amount : ℕ) 
  (h1 : initial_amount = 53)
  (h2 : atm_amount = 91)
  (h3 : final_amount = 14) :
  (initial_amount + atm_amount - final_amount) - atm_amount = 39 := by
  sorry

end supermarket_spending_difference_l4106_410606


namespace simplify_sqrt_expression_l4106_410617

theorem simplify_sqrt_expression :
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 245 / Real.sqrt 175) = (15 + 2 * Real.sqrt 7) / 10 := by
  sorry

end simplify_sqrt_expression_l4106_410617


namespace trigonometric_expression_equality_l4106_410687

theorem trigonometric_expression_equality : 
  (2 * Real.sin (25 * π / 180) ^ 2 - 1) / (Real.sin (20 * π / 180) * Real.cos (20 * π / 180)) = -2 := by
  sorry

end trigonometric_expression_equality_l4106_410687


namespace evaluate_g_l4106_410655

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + x + 1

-- State the theorem
theorem evaluate_g : 3 * g 2 + 2 * g (-2) = -9 := by
  sorry

end evaluate_g_l4106_410655


namespace probability_nine_heads_in_twelve_flips_l4106_410698

def coin_flips : ℕ := 12
def desired_heads : ℕ := 9

theorem probability_nine_heads_in_twelve_flips :
  (Nat.choose coin_flips desired_heads : ℚ) / (2 ^ coin_flips) = 220 / 4096 := by
  sorry

end probability_nine_heads_in_twelve_flips_l4106_410698


namespace f_neg_one_equals_five_l4106_410644

-- Define the function f
def f (x : ℝ) : ℝ := (1 - x)^2 + 1

-- State the theorem
theorem f_neg_one_equals_five : f (-1) = 5 := by
  sorry

end f_neg_one_equals_five_l4106_410644


namespace complement_intersection_theorem_l4106_410682

def U : Set ℕ := {x | x ≤ 8}
def A : Set ℕ := {1, 3, 7}
def B : Set ℕ := {2, 3, 8}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {0, 4, 5, 6} := by sorry

end complement_intersection_theorem_l4106_410682


namespace number_equals_1038_l4106_410637

theorem number_equals_1038 : ∃ n : ℝ, n * 40 = 173 * 240 ∧ n = 1038 := by
  sorry

end number_equals_1038_l4106_410637


namespace infinitely_many_M_exist_l4106_410630

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Check if a natural number has no zero digits -/
def hasNoZeroDigits (n : ℕ) : Prop := sorry

/-- The main theorem -/
theorem infinitely_many_M_exist (N : ℕ) (hN : N > 0) :
  ∀ k : ℕ, ∃ M : ℕ, M > k ∧ hasNoZeroDigits M ∧ sumOfDigits (N * M) = sumOfDigits M :=
sorry

end infinitely_many_M_exist_l4106_410630


namespace sum_a1_a5_l4106_410613

/-- For a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ) : ℕ := n^2 + 1

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℕ := S n - S (n-1)

theorem sum_a1_a5 : a 1 + a 5 = 11 := by
  sorry

end sum_a1_a5_l4106_410613


namespace problem_statement_l4106_410643

theorem problem_statement (a b : ℝ) (h : a * b + b^2 = 12) :
  (a + b)^2 - (a + b) * (a - b) = 24 := by
  sorry

end problem_statement_l4106_410643


namespace root_equation_result_l4106_410686

/-- Given two constants c and d, if the equation ((x+c)(x+d)(x-15))/((x-4)^2) = 0 has exactly 3 distinct roots,
    and the equation ((x+2c)(x-4)(x-9))/((x+d)(x-15)) = 0 has exactly 1 distinct root,
    then 100c + d = -391 -/
theorem root_equation_result (c d : ℝ) 
  (h1 : ∃! (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ 
    ∀ x, (x + c) * (x + d) * (x - 15) = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3)
  (h2 : ∃! (r : ℝ), ∀ x, (x + 2*c) * (x - 4) * (x - 9) = 0 ↔ x = r) :
  100 * c + d = -391 :=
sorry

end root_equation_result_l4106_410686


namespace rectangle_perimeter_l4106_410607

theorem rectangle_perimeter (L B : ℝ) 
  (h1 : L - B = 23) 
  (h2 : L * B = 3650) : 
  2 * L + 2 * B = 338 := by
sorry

end rectangle_perimeter_l4106_410607


namespace probability_three_primes_and_at_least_one_eight_l4106_410653

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Bool := sorry

/-- The set of prime numbers on an 8-sided die -/
def primesOnDie : Finset ℕ := {2, 3, 5, 7}

/-- The probability of rolling a prime number on a single 8-sided die -/
def probPrime : ℚ := (Finset.card primesOnDie : ℚ) / 8

/-- The probability of rolling an 8 on a single 8-sided die -/
def probEight : ℚ := 1 / 8

/-- The number of ways to choose 3 dice out of 6 -/
def chooseThreeOutOfSix : ℕ := Nat.choose 6 3

theorem probability_three_primes_and_at_least_one_eight :
  let probExactlyThreePrimes := chooseThreeOutOfSix * probPrime^3 * (1 - probPrime)^3
  let probAtLeastOneEight := 1 - (1 - probEight)^6
  probExactlyThreePrimes * probAtLeastOneEight = 2899900 / 16777216 := by
  sorry

end probability_three_primes_and_at_least_one_eight_l4106_410653


namespace initial_kittens_count_l4106_410676

/-- The number of kittens Alyssa's cat initially had -/
def initial_kittens : ℕ := sorry

/-- The number of kittens Alyssa gave to her friends -/
def given_away : ℕ := 4

/-- The number of kittens Alyssa has left -/
def kittens_left : ℕ := 4

/-- Theorem stating that the initial number of kittens is 8 -/
theorem initial_kittens_count : initial_kittens = 8 :=
by sorry

end initial_kittens_count_l4106_410676


namespace original_price_after_changes_l4106_410680

/-- Given an item with original price x, increased by q% and then reduced by r%,
    resulting in a final price of 2 dollars, prove that the original price x
    is equal to 20000 / (10000 + 100 * (q - r) - q * r) -/
theorem original_price_after_changes (q r : ℝ) (x : ℝ) 
    (h1 : x * (1 + q / 100) * (1 - r / 100) = 2) :
  x = 20000 / (10000 + 100 * (q - r) - q * r) := by
  sorry


end original_price_after_changes_l4106_410680


namespace fraction_inequality_l4106_410624

theorem fraction_inequality (x : ℝ) : (1 / x > 1) ↔ (0 < x ∧ x < 1) := by sorry

end fraction_inequality_l4106_410624


namespace inner_square_area_ratio_is_one_fourth_l4106_410672

/-- The ratio of the area of a square formed by connecting the center of a larger square
    to the midpoints of its sides, to the area of the larger square. -/
def inner_square_area_ratio : ℚ := 1 / 4

/-- Theorem stating that the ratio of the area of the inner square to the outer square is 1/4. -/
theorem inner_square_area_ratio_is_one_fourth :
  inner_square_area_ratio = 1 / 4 := by
  sorry

end inner_square_area_ratio_is_one_fourth_l4106_410672


namespace Q_iff_a_in_open_interval_P_or_Q_and_not_P_and_Q_iff_a_in_union_l4106_410659

-- Define the propositions P and Q
def P (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def Q (a : ℝ) : Prop := ∃ x y : ℝ, x^2 / (a + 1) + y^2 / (a - 2) = 1 ∧ (a + 1) * (a - 2) < 0

-- Theorem 1
theorem Q_iff_a_in_open_interval (a : ℝ) : Q a ↔ a ∈ Set.Ioo (-1) 2 := by sorry

-- Theorem 2
theorem P_or_Q_and_not_P_and_Q_iff_a_in_union (a : ℝ) : 
  (P a ∨ Q a) ∧ ¬(P a ∧ Q a) ↔ a ∈ Set.Ioo 1 2 ∪ Set.Iic (-1) := by sorry

end Q_iff_a_in_open_interval_P_or_Q_and_not_P_and_Q_iff_a_in_union_l4106_410659


namespace sum_of_squares_zero_implies_sum_l4106_410641

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) :
  (x - 2)^2 + (y - 3)^2 + (z - 6)^2 = 0 → x + y + z = 11 := by
  sorry

end sum_of_squares_zero_implies_sum_l4106_410641


namespace tim_younger_than_jenny_l4106_410651

/-- Given the ages of Tim, Rommel, and Jenny, prove that Tim is 12 years younger than Jenny. -/
theorem tim_younger_than_jenny (tim_age rommel_age jenny_age : ℕ) : 
  tim_age = 5 →
  rommel_age = 3 * tim_age →
  jenny_age = rommel_age + 2 →
  jenny_age - tim_age = 12 := by
sorry

end tim_younger_than_jenny_l4106_410651


namespace swiss_cheese_probability_l4106_410684

theorem swiss_cheese_probability :
  let cheddar : ℕ := 22
  let mozzarella : ℕ := 34
  let pepperjack : ℕ := 29
  let swiss : ℕ := 45
  let gouda : ℕ := 20
  let total : ℕ := cheddar + mozzarella + pepperjack + swiss + gouda
  (swiss : ℚ) / (total : ℚ) = 0.3 := by
  sorry

end swiss_cheese_probability_l4106_410684


namespace range_of_t_range_of_a_l4106_410610

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3|

-- Part 1
theorem range_of_t (t : ℝ) : f t + f (2 * t) < 9 ↔ -1 < t ∧ t < 5 := by sorry

-- Part 2
theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 2 4 ∧ f (2 * x) + |x + a| ≤ 3) ↔ a ∈ Set.Icc (-4) 0 := by sorry

end range_of_t_range_of_a_l4106_410610


namespace max_consecutive_odd_exponents_is_seven_l4106_410663

/-- A natural number has odd prime factor exponents if all exponents in its prime factorization are odd. -/
def has_odd_prime_factor_exponents (n : ℕ) : Prop :=
  ∀ p k, p.Prime → p ^ k ∣ n → k % 2 = 1

/-- The maximum number of consecutive natural numbers with odd prime factor exponents. -/
def max_consecutive_odd_exponents : ℕ := 7

/-- Theorem stating that the maximum number of consecutive natural numbers 
    with odd prime factor exponents is 7. -/
theorem max_consecutive_odd_exponents_is_seven :
  ∀ n : ℕ, ∃ m ∈ Finset.range 7, ¬(has_odd_prime_factor_exponents (n + m)) ∧
  ∃ k : ℕ, ∀ i ∈ Finset.range 7, has_odd_prime_factor_exponents (k + i) :=
sorry

end max_consecutive_odd_exponents_is_seven_l4106_410663


namespace max_matches_C_proof_l4106_410618

/-- Represents a player in the tournament -/
inductive Player : Type
| A : Player
| B : Player
| C : Player
| D : Player

/-- The number of matches won by a player -/
def matches_won : Player → Nat
| Player.A => 2
| Player.B => 1
| _ => 0  -- We don't know for C and D, so we set it to 0

/-- The total number of matches in a round-robin tournament with 4 players -/
def total_matches : Nat := 6

/-- The maximum number of matches C can win -/
def max_matches_C : Nat := 3

/-- Theorem stating the maximum number of matches C can win -/
theorem max_matches_C_proof :
  ∀ (c_wins : Nat),
  c_wins ≤ max_matches_C ∧
  c_wins + matches_won Player.A + matches_won Player.B ≤ total_matches :=
sorry

end max_matches_C_proof_l4106_410618


namespace smallest_two_digit_multiple_of_seven_l4106_410621

def digits : List Nat := [3, 5, 6, 7]

def isTwoDigitNumber (n : Nat) : Prop :=
  10 ≤ n ∧ n < 100

def formedFromList (n : Nat) : Prop :=
  ∃ (d1 d2 : Nat), d1 ∈ digits ∧ d2 ∈ digits ∧ d1 ≠ d2 ∧ n = 10 * d1 + d2

theorem smallest_two_digit_multiple_of_seven :
  ∃ (n : Nat), isTwoDigitNumber n ∧ formedFromList n ∧ n % 7 = 0 ∧
  (∀ (m : Nat), isTwoDigitNumber m → formedFromList m → m % 7 = 0 → n ≤ m) ∧
  n = 35 := by sorry

end smallest_two_digit_multiple_of_seven_l4106_410621


namespace cost_of_20_pencils_15_notebooks_l4106_410662

/-- The cost of a pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a notebook -/
def notebook_cost : ℝ := sorry

/-- The first condition: 9 pencils and 10 notebooks cost $5.45 -/
axiom condition1 : 9 * pencil_cost + 10 * notebook_cost = 5.45

/-- The second condition: 6 pencils and 4 notebooks cost $2.50 -/
axiom condition2 : 6 * pencil_cost + 4 * notebook_cost = 2.50

/-- Theorem: The cost of 20 pencils and 15 notebooks is $9.04 -/
theorem cost_of_20_pencils_15_notebooks :
  20 * pencil_cost + 15 * notebook_cost = 9.04 := by sorry

end cost_of_20_pencils_15_notebooks_l4106_410662


namespace smallest_solution_of_equation_l4106_410645

theorem smallest_solution_of_equation :
  let f (x : ℝ) := 3 * x / (x - 2) + (3 * x^2 - 36) / x
  ∃ (y : ℝ), y = (2 - Real.sqrt 58) / 3 ∧ 
    f y = 13 ∧ 
    ∀ (z : ℝ), f z = 13 → y ≤ z := by
  sorry

end smallest_solution_of_equation_l4106_410645


namespace f_properties_l4106_410609

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 2 / (3^x + 1)

theorem f_properties :
  ∀ (a : ℝ),
  -- 1. Range of f when a = 1
  (∀ y : ℝ, y ∈ Set.range (f 1) ↔ 1 < y ∧ y < 3) ∧
  -- 2. f is strictly decreasing
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) ∧
  -- 3. If f is odd and f(f(x)) + f(m) < 0 has solutions, then m > -1
  (((∀ x : ℝ, f a (-x) = -f a x) ∧
    (∃ x : ℝ, f a (f a x) + f a m < 0)) → m > -1) :=
by sorry

end f_properties_l4106_410609


namespace function_satisfies_conditions_l4106_410616

/-- The set of positive real numbers -/
def PositiveReals := {x : ℝ | x > 0}

/-- The function f: S³ → S -/
noncomputable def f (x y z : ℝ) : ℝ := (y + Real.sqrt (y^2 + 4*x*z)) / (2*x)

/-- The main theorem -/
theorem function_satisfies_conditions :
  ∀ (x y z k : ℝ), x ∈ PositiveReals → y ∈ PositiveReals → z ∈ PositiveReals → k ∈ PositiveReals →
  (x * f x y z = z * f z y x) ∧
  (f x (k*y) (k^2*z) = k * f x y z) ∧
  (f 1 k (k+1) = k+1) := by
  sorry

end function_satisfies_conditions_l4106_410616
