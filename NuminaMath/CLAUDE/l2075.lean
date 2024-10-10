import Mathlib

namespace route_length_l2075_207573

/-- Proves that given a round trip with total time of 1 hour, average speed of 8 miles/hour,
    and return speed of 20 miles/hour along the same path, the length of the one-way route is 4 miles. -/
theorem route_length (total_time : ℝ) (avg_speed : ℝ) (return_speed : ℝ) (route_length : ℝ) : 
  total_time = 1 →
  avg_speed = 8 →
  return_speed = 20 →
  route_length * 2 = avg_speed * total_time →
  route_length / return_speed + route_length / (route_length * 2 / total_time - return_speed) = total_time →
  route_length = 4 := by
  sorry

end route_length_l2075_207573


namespace constant_sum_property_l2075_207572

/-- Represents a triangle with numbers at its vertices -/
structure NumberedTriangle where
  a : ℝ  -- Number at vertex A
  b : ℝ  -- Number at vertex B
  c : ℝ  -- Number at vertex C

/-- The sum of a vertex number and the opposite side sum is constant -/
theorem constant_sum_property (t : NumberedTriangle) :
  t.a + (t.b + t.c) = t.b + (t.c + t.a) ∧
  t.b + (t.c + t.a) = t.c + (t.a + t.b) ∧
  t.c + (t.a + t.b) = t.a + t.b + t.c :=
sorry

end constant_sum_property_l2075_207572


namespace negation_equivalence_l2075_207548

universe u

-- Define the universe of discourse
variable {Person : Type u}

-- Define predicates
variable (Teacher : Person → Prop)
variable (ExcellentInMath : Person → Prop)
variable (PoorInMath : Person → Prop)

-- Define the theorem
theorem negation_equivalence :
  (∃ x, Teacher x ∧ PoorInMath x) ↔ ¬(∀ x, Teacher x → ExcellentInMath x) :=
sorry

end negation_equivalence_l2075_207548


namespace complex_division_result_l2075_207562

theorem complex_division_result : (5 - I) / (1 - I) = 3 + 2*I := by
  sorry

end complex_division_result_l2075_207562


namespace range_of_x_in_negative_sqrt_l2075_207512

theorem range_of_x_in_negative_sqrt (x : ℝ) :
  (3 * x + 5 ≥ 0) ↔ (x ≥ -5/3) :=
by sorry

end range_of_x_in_negative_sqrt_l2075_207512


namespace quadratic_equation_properties_l2075_207516

/-- The quadratic equation x^2 - (2m+1)x + m^2 + m = 0 -/
def quadratic_equation (m x : ℝ) : ℝ := x^2 - (2*m+1)*x + m^2 + m

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ := (2*m+1)^2 - 4*(m^2 + m)

/-- The sum of roots of the quadratic equation -/
def sum_of_roots (m : ℝ) : ℝ := 2*m + 1

/-- The product of roots of the quadratic equation -/
def product_of_roots (m : ℝ) : ℝ := m^2 + m

theorem quadratic_equation_properties (m : ℝ) :
  (discriminant m = 1) ∧
  (∃ a b : ℝ, quadratic_equation m a = 0 ∧ quadratic_equation m b = 0 ∧
    (2*a + b) * (a + 2*b) = 20 → (m = -2 ∨ m = 1)) :=
by sorry

end quadratic_equation_properties_l2075_207516


namespace george_movie_cost_l2075_207506

/-- The total cost of George's visit to the movie theater -/
def total_cost (ticket_price : ℝ) (nachos_price : ℝ) : ℝ :=
  ticket_price + nachos_price

/-- Theorem: George's total cost for the movie theater visit is $24 -/
theorem george_movie_cost :
  ∀ (ticket_price : ℝ) (nachos_price : ℝ),
    ticket_price = 16 →
    nachos_price = ticket_price / 2 →
    total_cost ticket_price nachos_price = 24 :=
by
  sorry

end george_movie_cost_l2075_207506


namespace vector_length_on_number_line_l2075_207519

theorem vector_length_on_number_line : 
  ∀ (A B : ℝ), A = -1 → B = 2 → abs (B - A) = 3 :=
by
  sorry

end vector_length_on_number_line_l2075_207519


namespace fourth_number_proof_l2075_207555

theorem fourth_number_proof (sum : ℝ) (a b c : ℝ) (h1 : sum = 221.2357) 
  (h2 : a = 217) (h3 : b = 2.017) (h4 : c = 0.217) : 
  sum - (a + b + c) = 2.0017 := by
  sorry

end fourth_number_proof_l2075_207555


namespace impossibleToMakeAllEqual_l2075_207593

/-- Represents the possible values in a cell of the table -/
inductive CellValue
  | Zero
  | One
  deriving Repr

/-- Represents a 4x4 table of cell values -/
def Table := Fin 4 → Fin 4 → CellValue

/-- Represents the initial state of the table -/
def initialTable : Table := fun i j =>
  if i = 0 ∧ j = 1 then CellValue.One else CellValue.Zero

/-- Represents the allowed operations on the table -/
inductive Operation
  | AddToRow (row : Fin 4)
  | AddToColumn (col : Fin 4)
  | AddToDiagonal (startRow startCol : Fin 4)

/-- Applies an operation to a table -/
def applyOperation (t : Table) (op : Operation) : Table :=
  sorry

/-- Checks if all values in the table are equal -/
def allEqual (t : Table) : Prop :=
  ∀ i j k l, t i j = t k l

/-- The main theorem stating that it's impossible to make all numbers equal -/
theorem impossibleToMakeAllEqual :
  ¬∃ (ops : List Operation), allEqual (ops.foldl applyOperation initialTable) :=
sorry

end impossibleToMakeAllEqual_l2075_207593


namespace min_triangle_area_l2075_207538

/-- The minimum area of a triangle with vertices A(0,0), B(30,10), and C(p,q) where p and q are integers -/
theorem min_triangle_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (30, 10)
  ∃ (min_area : ℝ), min_area = 5/2 ∧ 
    ∀ (p q : ℤ), 
      let C : ℝ × ℝ := (p, q)
      let area := (1/2) * |(-p : ℝ) + 3*q|
      area ≥ min_area :=
by sorry

end min_triangle_area_l2075_207538


namespace m_range_theorem_l2075_207594

open Set

noncomputable def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

noncomputable def Q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

def p_set : Set ℝ := {x | P x}
def q_set (m : ℝ) : Set ℝ := {x | Q x m}

theorem m_range_theorem :
  ∀ m : ℝ, (0 < m ∧ m ≤ 3) ↔ 
    (m > 0 ∧ q_set m ⊂ p_set ∧ q_set m ≠ p_set) :=
sorry

end m_range_theorem_l2075_207594


namespace no_integer_solutions_l2075_207534

theorem no_integer_solutions : ¬∃ (a b : ℤ), a^3 + 3*a^2 + 2*a = 125*b^3 + 75*b^2 + 15*b + 2 := by
  sorry

end no_integer_solutions_l2075_207534


namespace no_valid_cube_labeling_l2075_207543

/-- A cube vertex labeling is a function from vertex indices to odd numbers -/
def CubeLabeling := Fin 8 → Nat

/-- Predicate to check if two numbers are adjacent on a cube -/
def adjacent (i j : Fin 8) : Prop :=
  (i.val + j.val) % 2 = 1 ∧ i ≠ j

/-- Predicate to check if a labeling satisfies the problem conditions -/
def validLabeling (f : CubeLabeling) : Prop :=
  (∀ i, 1 ≤ f i ∧ f i ≤ 600 ∧ f i % 2 = 1) ∧
  (∀ i j, adjacent i j → ∃ d > 1, d ∣ f i ∧ d ∣ f j) ∧
  (∀ i j, ¬adjacent i j → ∀ d > 1, ¬(d ∣ f i ∧ d ∣ f j)) ∧
  (∀ i j, i ≠ j → f i ≠ f j)

theorem no_valid_cube_labeling : ¬∃ f : CubeLabeling, validLabeling f :=
sorry

end no_valid_cube_labeling_l2075_207543


namespace solution_set_l2075_207507

theorem solution_set : ∀ x y : ℝ,
  (3/20 + |x - 15/40| < 7/20 ∧ y = 2*x + 1) ↔ 
  (7/20 < x ∧ x < 2/5 ∧ 17/10 ≤ y ∧ y ≤ 11/5) :=
by sorry

end solution_set_l2075_207507


namespace original_number_proof_l2075_207503

theorem original_number_proof : ∃! N : ℤ, 
  (N - 8) % 5 = 4 ∧ 
  (N - 8) % 7 = 4 ∧ 
  (N - 8) % 9 = 4 ∧ 
  N = 326 := by
sorry

end original_number_proof_l2075_207503


namespace triangle_theorem_l2075_207584

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sinA : ℝ
  sinB : ℝ
  sinC : ℝ
  cosA : ℝ
  cosB : ℝ
  cosC : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = 2 * t.b ∧ 
  t.sinC = 3/4 ∧ 
  t.b^2 + t.b * t.c = 2 * t.a^2

-- State the theorem
theorem triangle_theorem (t : Triangle) 
  (h : triangle_conditions t) : 
  t.sinB = 3/8 ∧ t.cosB = (3 * Real.sqrt 6) / 8 := by
  sorry

end triangle_theorem_l2075_207584


namespace complex_magnitude_l2075_207586

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 3 - 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end complex_magnitude_l2075_207586


namespace quadrilateral_circumscribed_circle_l2075_207554

/-- The quadrilateral formed by four lines has a circumscribed circle -/
theorem quadrilateral_circumscribed_circle 
  (l₁ : Real → Real → Prop) 
  (l₂ : Real → Real → Real → Prop)
  (l₃ : Real → Real → Prop)
  (l₄ : Real → Real → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ x + 3*y - 15 = 0)
  (h₂ : ∀ x y k, l₂ x y k ↔ k*x - y - 6 = 0)
  (h₃ : ∀ x y, l₃ x y ↔ x + 5*y = 0)
  (h₄ : ∀ x y, l₄ x y ↔ y = 0) :
  ∃ (k : Real) (circle : Real → Real → Prop),
    k = -8/15 ∧
    (∀ x y, circle x y ↔ x^2 + y^2 - 15*x - 159*y = 0) ∧
    (∀ x y, (l₁ x y ∨ l₂ x y k ∨ l₃ x y ∨ l₄ x y) → circle x y) :=
by sorry

end quadrilateral_circumscribed_circle_l2075_207554


namespace age_ratio_problem_l2075_207530

/-- Given Sam's current age s and Tim's current age t, where:
    1. s - 4 = 4(t - 4)
    2. s - 10 = 5(t - 10)
    Prove that the number of years x until their age ratio is 3:1 is 8. -/
theorem age_ratio_problem (s t : ℕ) 
  (h1 : s - 4 = 4 * (t - 4)) 
  (h2 : s - 10 = 5 * (t - 10)) : 
  ∃ x : ℕ, x = 8 ∧ (s + x) / (t + x) = 3 := by
  sorry

end age_ratio_problem_l2075_207530


namespace function_equality_l2075_207553

theorem function_equality (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = x^2) : 
  ∀ x, f x = (x + 1)^2 := by
sorry

end function_equality_l2075_207553


namespace difference_of_squares_example_l2075_207546

theorem difference_of_squares_example : (527 : ℤ) * 527 - 526 * 528 = 1 := by
  sorry

end difference_of_squares_example_l2075_207546


namespace latus_rectum_of_parabola_l2075_207515

/-- Given a parabola with equation x = 4y², prove that its latus rectum has the equation x = -1/16 -/
theorem latus_rectum_of_parabola (y : ℝ) :
  let x := 4 * y^2
  (∃ p : ℝ, p = 1/8 ∧ x = -p) → x = -1/16 := by
  sorry

end latus_rectum_of_parabola_l2075_207515


namespace two_propositions_have_true_converse_l2075_207502

-- Define the propositions
def proposition1 := "Vertical angles are equal"
def proposition2 := "Supplementary angles of the same side are complementary, and two lines are parallel"
def proposition3 := "Corresponding angles of congruent triangles are equal"
def proposition4 := "If the squares of two real numbers are equal, then the two real numbers are equal"

-- Define a function to check if a proposition has a true converse
def hasValidConverse (p : String) : Bool :=
  match p with
  | "Vertical angles are equal" => false
  | "Supplementary angles of the same side are complementary, and two lines are parallel" => true
  | "Corresponding angles of congruent triangles are equal" => false
  | "If the squares of two real numbers are equal, then the two real numbers are equal" => true
  | _ => false

-- Theorem statement
theorem two_propositions_have_true_converse :
  (hasValidConverse proposition1).toNat +
  (hasValidConverse proposition2).toNat +
  (hasValidConverse proposition3).toNat +
  (hasValidConverse proposition4).toNat = 2 := by
  sorry

end two_propositions_have_true_converse_l2075_207502


namespace x_value_l2075_207521

-- Define the problem statement
theorem x_value (x : ℝ) : x = 70 * (1 + 0.12) → x = 78.4 := by
  sorry

end x_value_l2075_207521


namespace regression_line_equation_l2075_207526

/-- Given a regression line with slope 1.23 and a point (4, 5) on the line,
    prove that the equation of the line is y = 1.23x + 0.08 -/
theorem regression_line_equation (x y : ℝ) :
  let slope : ℝ := 1.23
  let point : ℝ × ℝ := (4, 5)
  (y - point.2 = slope * (x - point.1)) → (y = slope * x + 0.08) :=
by
  sorry

end regression_line_equation_l2075_207526


namespace no_solution_equation_l2075_207522

theorem no_solution_equation :
  ¬ ∃ (x : ℝ), 6 + 3.5 * x = 2.5 * x - 30 + x := by
  sorry

end no_solution_equation_l2075_207522


namespace smallest_k_correct_l2075_207560

/-- The smallest integer k for which kx^2 - 4x - 4 = 0 has two distinct real roots -/
def smallest_k : ℤ := 1

/-- Quadratic equation ax^2 + bx + c = 0 has two distinct real roots iff b^2 - 4ac > 0 -/
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4*a*c > 0

theorem smallest_k_correct :
  (∀ k : ℤ, k < smallest_k → ¬(has_two_distinct_real_roots k (-4) (-4))) ∧
  has_two_distinct_real_roots smallest_k (-4) (-4) :=
sorry

end smallest_k_correct_l2075_207560


namespace thirtieth_term_of_sequence_l2075_207581

def arithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem thirtieth_term_of_sequence : 
  let a₁ := 8
  let a₂ := 5
  let a₃ := 2
  let d := a₂ - a₁
  arithmeticSequence a₁ d 30 = -79 := by
sorry

end thirtieth_term_of_sequence_l2075_207581


namespace complex_fraction_simplification_l2075_207574

theorem complex_fraction_simplification :
  let z : ℂ := (3 - 2*I) / (1 + 5*I)
  z = -7/26 - 17/26*I :=
by sorry

end complex_fraction_simplification_l2075_207574


namespace tommys_coin_collection_l2075_207557

theorem tommys_coin_collection (nickels dimes quarters pennies : ℕ) : 
  nickels = 100 →
  nickels = 2 * dimes →
  quarters = 4 →
  pennies = 10 * quarters →
  dimes - pennies = 10 := by
  sorry

end tommys_coin_collection_l2075_207557


namespace linear_function_theorem_l2075_207500

/-- A linear function that intersects the x-axis at (-2, 0) and forms a triangle with area 8 with the coordinate axes -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  x_intercept : k * (-2) + b = 0
  triangle_area : |k * 2 * b / 2| = 8

/-- The two possible linear functions satisfying the given conditions -/
def possible_functions : Set LinearFunction :=
  { f | f.k = 4 ∧ f.b = 8 } ∪ { f | f.k = -4 ∧ f.b = -8 }

/-- Theorem stating that the only linear functions satisfying the conditions are y = 4x + 8 or y = -4x - 8 -/
theorem linear_function_theorem :
  ∀ f : LinearFunction, f ∈ possible_functions :=
by sorry

end linear_function_theorem_l2075_207500


namespace circle_properties_l2075_207588

/-- A circle with center on the y-axis, radius 1, passing through (1, 2) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 1

theorem circle_properties :
  ∃ (b : ℝ),
    (∀ x y : ℝ, circle_equation x y ↔ x^2 + (y - b)^2 = 1) ∧
    (circle_equation 1 2) ∧
    (∀ x y : ℝ, circle_equation x y → x^2 + y^2 = 1) :=
sorry

end circle_properties_l2075_207588


namespace divisible_by_nine_sequence_l2075_207580

theorem divisible_by_nine_sequence (start : ℕ) (h1 : start ≥ 32) (h2 : start % 9 = 0) : 
  let sequence := List.range 7
  let last_number := start + 9 * 6
  last_number = 90 :=
by sorry

end divisible_by_nine_sequence_l2075_207580


namespace min_value_reciprocal_sum_l2075_207535

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (1 / x + 4 / y) ≥ 9 / 2 := by
  sorry

end min_value_reciprocal_sum_l2075_207535


namespace problem_statement_l2075_207509

theorem problem_statement (a b : ℕ) (m : ℝ) 
  (h1 : a > 1) 
  (h2 : b > 1) 
  (h3 : a * (b + Real.sin m) = b + Real.cos m) : 
  a + b = 4 := by
  sorry

end problem_statement_l2075_207509


namespace largest_x_floor_ratio_l2075_207520

theorem largest_x_floor_ratio : 
  ∀ x : ℝ, (↑(⌊x⌋) / x = 8 / 9) → x ≤ 63 / 8 :=
by
  sorry

end largest_x_floor_ratio_l2075_207520


namespace similar_triangles_shortest_side_l2075_207508

theorem similar_triangles_shortest_side 
  (a b c : ℝ) -- sides of the first triangle
  (d e f : ℝ) -- sides of the second triangle
  (h1 : a^2 + b^2 = c^2) -- first triangle is right-angled
  (h2 : d^2 + e^2 = f^2) -- second triangle is right-angled
  (h3 : a = 24) -- first condition on first triangle
  (h4 : b = 32) -- second condition on first triangle
  (h5 : f = 80) -- condition on second triangle's hypotenuse
  (h6 : a / d = b / e) -- triangles are similar
  (h7 : b / e = c / f) -- triangles are similar
  : d = 48 := by
  sorry

end similar_triangles_shortest_side_l2075_207508


namespace onion_piece_per_student_l2075_207583

/-- Represents the pizza distribution problem --/
structure PizzaDistribution where
  students : ℕ
  pizzas : ℕ
  slices_per_pizza : ℕ
  cheese_per_student : ℕ
  leftover_cheese : ℕ
  leftover_onion : ℕ

/-- Calculates the number of onion pieces per student --/
def onion_per_student (pd : PizzaDistribution) : ℕ :=
  let total_slices := pd.pizzas * pd.slices_per_pizza
  let total_cheese := pd.students * pd.cheese_per_student
  let used_slices := total_slices - pd.leftover_cheese - pd.leftover_onion
  let onion_slices := used_slices - total_cheese
  onion_slices / pd.students

/-- Theorem stating that each student gets 1 piece of onion pizza --/
theorem onion_piece_per_student (pd : PizzaDistribution) 
  (h1 : pd.students = 32)
  (h2 : pd.pizzas = 6)
  (h3 : pd.slices_per_pizza = 18)
  (h4 : pd.cheese_per_student = 2)
  (h5 : pd.leftover_cheese = 8)
  (h6 : pd.leftover_onion = 4) :
  onion_per_student pd = 1 := by
  sorry

end onion_piece_per_student_l2075_207583


namespace f_one_upper_bound_l2075_207558

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 - m * x + 5

-- State the theorem
theorem f_one_upper_bound (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ -2 → f m x₁ > f m x₂) →
  f m 1 ≤ 15 := by
  sorry

end f_one_upper_bound_l2075_207558


namespace ship_capacity_and_tax_calculation_l2075_207544

/-- Represents the types of cargo --/
inductive CargoType
  | Steel
  | Timber
  | Electronics
  | Textiles

/-- Represents a cargo load with its type and weight --/
structure CargoLoad :=
  (type : CargoType)
  (weight : Nat)

/-- Calculates the total weight of a list of cargo loads --/
def totalWeight (loads : List CargoLoad) : Nat :=
  loads.foldl (fun acc load => acc + load.weight) 0

/-- Calculates the import tax for a single cargo load --/
def importTax (load : CargoLoad) : Nat :=
  match load.type with
  | CargoType.Steel => load.weight * 50
  | CargoType.Timber => load.weight * 75
  | CargoType.Electronics => load.weight * 100
  | CargoType.Textiles => load.weight * 40

/-- Calculates the total import tax for a list of cargo loads --/
def totalImportTax (loads : List CargoLoad) : Nat :=
  loads.foldl (fun acc load => acc + importTax load) 0

/-- The main theorem to prove --/
theorem ship_capacity_and_tax_calculation 
  (maxCapacity : Nat)
  (initialCargo : List CargoLoad)
  (additionalCargo : List CargoLoad) :
  maxCapacity = 20000 →
  initialCargo = [
    ⟨CargoType.Steel, 3428⟩,
    ⟨CargoType.Timber, 1244⟩,
    ⟨CargoType.Electronics, 1301⟩
  ] →
  additionalCargo = [
    ⟨CargoType.Steel, 3057⟩,
    ⟨CargoType.Textiles, 2364⟩,
    ⟨CargoType.Timber, 1517⟩,
    ⟨CargoType.Electronics, 1785⟩
  ] →
  totalWeight (initialCargo ++ additionalCargo) ≤ maxCapacity ∧
  totalImportTax (initialCargo ++ additionalCargo) = 934485 :=
by sorry


end ship_capacity_and_tax_calculation_l2075_207544


namespace incorrect_equality_l2075_207511

theorem incorrect_equality (h : (12.5 / 12.5) = (2.4 / 2.4)) :
  ¬ (25 * (0.5 / 0.5) = 4 * (0.6 / 0.6)) := by
  sorry

end incorrect_equality_l2075_207511


namespace unique_configuration_l2075_207595

/-- A configuration of n points in the plane with associated real numbers -/
structure PointConfiguration (n : ℕ) where
  points : Fin n → ℝ × ℝ
  r : Fin n → ℝ

/-- The area of a triangle given by three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Predicate for non-collinearity of three points -/
def non_collinear (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

/-- The main theorem: only n = 4 satisfies the conditions -/
theorem unique_configuration :
  ∀ n : ℕ, n > 3 →
  (∃ (config : PointConfiguration n),
    (∀ i j k : Fin n, i < j → j < k →
      non_collinear (config.points i) (config.points j) (config.points k)) ∧
    (∀ i j k : Fin n, i < j → j < k →
      triangle_area (config.points i) (config.points j) (config.points k) =
        config.r i + config.r j + config.r k)) →
  n = 4 := by sorry

end unique_configuration_l2075_207595


namespace probability_three_red_balls_l2075_207528

/-- The probability of picking 3 red balls from a bag containing 4 red, 5 blue, and 3 green balls -/
theorem probability_three_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ) :
  total_balls = red_balls + blue_balls + green_balls →
  red_balls = 4 →
  blue_balls = 5 →
  green_balls = 3 →
  (red_balls : ℚ) / total_balls * (red_balls - 1) / (total_balls - 1) * (red_balls - 2) / (total_balls - 2) = 1 / 55 := by
  sorry

end probability_three_red_balls_l2075_207528


namespace problem_solution_l2075_207559

theorem problem_solution (x y z : ℤ) 
  (h1 : x + y = 74)
  (h2 : (x + y) + y + z = 164)
  (h3 : z - y = 16) :
  x = 37 := by
  sorry

end problem_solution_l2075_207559


namespace ellipse_chord_ratio_theorem_l2075_207525

noncomputable section

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- The eccentricity of an ellipse -/
def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: For any ellipse satisfying the given conditions, 
    the ratio of the square of the chord length passing through the origin 
    to the chord length passing through the left focus is always 4, 
    when the slope angles of these chords sum to π -/
theorem ellipse_chord_ratio_theorem (e : Ellipse) 
    (h_focus : e.eccentricity * e.a = 1)
    (h_b_mean : e.b^2 = 3 * e.eccentricity * e.a)
    (α β : ℝ)
    (h_angle_sum : α + β = π)
    (A B D E : Point)
    (h_AB_on_ellipse : A.x^2 / e.a^2 + A.y^2 / e.b^2 = 1 ∧ 
                       B.x^2 / e.a^2 + B.y^2 / e.b^2 = 1)
    (h_DE_on_ellipse : D.x^2 / e.a^2 + D.y^2 / e.b^2 = 1 ∧ 
                       E.x^2 / e.a^2 + E.y^2 / e.b^2 = 1)
    (h_AB_through_origin : ∃ (k : ℝ), A.y = k * A.x ∧ B.y = k * B.x)
    (h_DE_through_focus : ∃ (m : ℝ), D.y = m * (D.x + 1) ∧ E.y = m * (E.x + 1))
    (h_AB_slope : ∃ (k : ℝ), k = Real.tan α)
    (h_DE_slope : ∃ (m : ℝ), m = Real.tan β) :
    (distance A B)^2 / (distance D E) = 4 := by
  sorry

end ellipse_chord_ratio_theorem_l2075_207525


namespace cube_volume_from_circumscribed_sphere_l2075_207523

/-- Given a cube with a circumscribed sphere of volume 32π/3, the volume of the cube is 64√3/9 -/
theorem cube_volume_from_circumscribed_sphere (V_sphere : ℝ) (V_cube : ℝ) :
  V_sphere = 32 / 3 * Real.pi → V_cube = 64 * Real.sqrt 3 / 9 := by
  sorry

end cube_volume_from_circumscribed_sphere_l2075_207523


namespace minimum_teams_l2075_207524

theorem minimum_teams (total_players : Nat) (max_team_size : Nat) : total_players = 30 → max_team_size = 8 → ∃ (num_teams : Nat), num_teams = 5 ∧ 
  (∃ (players_per_team : Nat), 
    players_per_team ≤ max_team_size ∧ 
    total_players = num_teams * players_per_team ∧
    ∀ (x : Nat), x < num_teams → 
      total_players % x ≠ 0 ∨ (total_players / x) > max_team_size) := by
  sorry

end minimum_teams_l2075_207524


namespace pony_daily_food_cost_l2075_207537

def annual_expenses : ℕ := 15890
def monthly_pasture_rent : ℕ := 500
def weekly_lessons : ℕ := 2
def lesson_cost : ℕ := 60
def months_per_year : ℕ := 12
def weeks_per_year : ℕ := 52
def days_per_year : ℕ := 365

theorem pony_daily_food_cost :
  (annual_expenses - (monthly_pasture_rent * months_per_year + weekly_lessons * lesson_cost * weeks_per_year)) / days_per_year = 10 :=
by sorry

end pony_daily_food_cost_l2075_207537


namespace line_vector_to_slope_intercept_l2075_207549

/-- Given a line in vector form, prove its equivalence to slope-intercept form -/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), 
  (2 : ℝ) * (x - 1) + (-1 : ℝ) * (y + 1) = 0 ↔ y = 2 * x - 3 := by
  sorry

end line_vector_to_slope_intercept_l2075_207549


namespace modulo_eleven_residue_l2075_207517

theorem modulo_eleven_residue : (178 + 4 * 28 + 8 * 62 + 3 * 21) % 11 = 2 := by
  sorry

end modulo_eleven_residue_l2075_207517


namespace sum_twenty_from_negative_nine_l2075_207504

/-- The sum of n consecutive integers starting from a given first term -/
def sumConsecutiveIntegers (n : ℕ) (first : ℤ) : ℤ :=
  n * (2 * first + n - 1) / 2

/-- Theorem: The sum of 20 consecutive integers starting from -9 is 10 -/
theorem sum_twenty_from_negative_nine :
  sumConsecutiveIntegers 20 (-9) = 10 := by
  sorry

end sum_twenty_from_negative_nine_l2075_207504


namespace positive_X_value_l2075_207547

-- Define the # relation
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- State the theorem
theorem positive_X_value (X : ℝ) (h1 : hash X 7 = 250) (h2 : X > 0) : X = Real.sqrt 201 := by
  sorry

end positive_X_value_l2075_207547


namespace geometric_sequence_problem_l2075_207532

theorem geometric_sequence_problem (a b c d e : ℕ) : 
  (2 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < 100) →
  Nat.gcd a e = 1 →
  (∃ (r : ℚ), b = a * r ∧ c = a * r^2 ∧ d = a * r^3 ∧ e = a * r^4) →
  c = 36 := by
sorry

end geometric_sequence_problem_l2075_207532


namespace f_zero_range_l2075_207550

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * Real.exp 1 * x - (Real.log x) / x + a

theorem f_zero_range (a : ℝ) :
  (∃ x > 0, f x a = 0) → a ≤ Real.exp 2 + 1 / Real.exp 1 :=
by sorry

end f_zero_range_l2075_207550


namespace triangle_isosceles_from_quadratic_equation_l2075_207591

/-- A triangle with sides a, b, and c is isosceles if the quadratic equation
    (c-b)x^2 + 2(b-a)x + (a-b) = 0 has two equal real roots. -/
theorem triangle_isosceles_from_quadratic_equation (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_eq : ∃ x : ℝ, (c - b) * x^2 + 2*(b - a)*x + (a - b) = 0 ∧ 
    ∀ y : ℝ, (c - b) * y^2 + 2*(b - a)*y + (a - b) = 0 → y = x) :
  (a = b ∧ c ≠ b) ∨ (a = c ∧ b ≠ c) ∨ (b = c ∧ a ≠ b) :=
sorry

end triangle_isosceles_from_quadratic_equation_l2075_207591


namespace angle_pcq_is_45_deg_l2075_207542

/-- Given a unit square ABCD with points P on AB and Q on AD forming
    triangle APQ with perimeter 2, angle PCQ is 45 degrees. -/
theorem angle_pcq_is_45_deg (A B C D P Q : ℝ × ℝ) : 
  -- Square ABCD is a unit square
  A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1) →
  -- P is on AB
  ∃ a : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ P = (a, 0) →
  -- Q is on AD
  ∃ b : ℝ, 0 ≤ b ∧ b ≤ 1 ∧ Q = (0, b) →
  -- Perimeter of APQ is 2
  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) +
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) +
  Real.sqrt ((A.1 - Q.1)^2 + (A.2 - Q.2)^2) = 2 →
  -- Angle PCQ is 45 degrees
  (Real.arctan ((C.2 - P.2) / (C.1 - P.1)) -
   Real.arctan ((C.1 - Q.1) / (C.2 - Q.2))) * (180 / Real.pi) = 45 := by
  sorry


end angle_pcq_is_45_deg_l2075_207542


namespace puppies_calculation_l2075_207545

/-- The number of puppies Alyssa initially had -/
def initial_puppies : ℕ := 12

/-- The number of puppies Alyssa gave away -/
def puppies_given_away : ℕ := 7

/-- The number of puppies Alyssa has now -/
def remaining_puppies : ℕ := initial_puppies - puppies_given_away

theorem puppies_calculation : remaining_puppies = 5 := by
  sorry

end puppies_calculation_l2075_207545


namespace thirty_percent_less_than_eighty_l2075_207569

theorem thirty_percent_less_than_eighty (x : ℝ) : x + x/2 = 80 * (1 - 0.3) → x = 37 := by
  sorry

end thirty_percent_less_than_eighty_l2075_207569


namespace sum_of_squares_and_products_l2075_207576

theorem sum_of_squares_and_products (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x^2 + y^2 + z^2 = 48)
  (h5 : x*y + y*z + z*x = 26) : 
  x + y + z = 10 := by
sorry

end sum_of_squares_and_products_l2075_207576


namespace loss_of_30_notation_l2075_207536

def profit_notation (amount : ℤ) : ℤ := amount

def loss_notation (amount : ℤ) : ℤ := -amount

theorem loss_of_30_notation :
  profit_notation 20 = 20 →
  loss_notation 30 = -30 :=
by
  sorry

end loss_of_30_notation_l2075_207536


namespace james_weight_vest_savings_l2075_207575

/-- The amount James saves by assembling his own weight vest -/
theorem james_weight_vest_savings : 
  let weight_vest_cost : ℝ := 250
  let weight_plates_pounds : ℝ := 200
  let weight_plates_cost_per_pound : ℝ := 1.2
  let ready_made_vest_cost : ℝ := 700
  let ready_made_vest_discount : ℝ := 100
  
  let james_vest_cost := weight_vest_cost + weight_plates_pounds * weight_plates_cost_per_pound
  let discounted_ready_made_vest_cost := ready_made_vest_cost - ready_made_vest_discount
  
  discounted_ready_made_vest_cost - james_vest_cost = 110 := by
  sorry

end james_weight_vest_savings_l2075_207575


namespace minuend_is_zero_l2075_207527

theorem minuend_is_zero (x y : ℝ) (h : x - y = -y) : x = 0 := by
  sorry

end minuend_is_zero_l2075_207527


namespace sum_of_squares_mod_13_l2075_207505

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_mod_13 : sum_of_squares 15 % 13 = 5 := by
  sorry

end sum_of_squares_mod_13_l2075_207505


namespace expression_factorization_l2075_207556

theorem expression_factorization (x : ℝ) : 
  4*x*(x-5) + 5*(x-5) + 6*x*(x-2) = (4*x+5)*(x-5) + 6*x*(x-2) := by
  sorry

end expression_factorization_l2075_207556


namespace card_sum_problem_l2075_207592

theorem card_sum_problem (a b c d e f g h : ℕ) :
  (a + b) * (c + d) * (e + f) * (g + h) = 330 →
  a + b + c + d + e + f + g + h = 21 := by
sorry

end card_sum_problem_l2075_207592


namespace complex_power_eight_l2075_207567

theorem complex_power_eight : (Complex.I + 1 : ℂ) ^ 8 / 4 = 1 := by sorry

end complex_power_eight_l2075_207567


namespace barbed_wire_height_l2075_207563

/-- Calculates the height of a barbed wire fence around a square field. -/
theorem barbed_wire_height 
  (field_area : ℝ) 
  (wire_cost_per_meter : ℝ) 
  (gate_width : ℝ) 
  (num_gates : ℕ) 
  (total_cost : ℝ) 
  (h : field_area = 3136) 
  (h1 : wire_cost_per_meter = 3.5) 
  (h2 : gate_width = 1) 
  (h3 : num_gates = 2) 
  (h4 : total_cost = 2331) : 
  Real.sqrt field_area * 4 - (gate_width * num_gates) * wire_cost_per_meter * 
    (total_cost / (Real.sqrt field_area * 4 - gate_width * num_gates) / wire_cost_per_meter) = 2331 :=
sorry

end barbed_wire_height_l2075_207563


namespace factor_expression_l2075_207571

theorem factor_expression (x : ℝ) : 
  (4 * x^4 + 128 * x^3 - 9) - (-6 * x^4 + 2 * x^3 - 9) = 2 * x^3 * (5 * x + 63) := by
sorry

end factor_expression_l2075_207571


namespace cost_price_is_4_l2075_207570

/-- The cost price of a pen in yuan. -/
def cost_price : ℝ := 4

/-- The retail price of a pen in the first scenario. -/
def retail_price1 : ℝ := 7

/-- The retail price of a pen in the second scenario. -/
def retail_price2 : ℝ := 8

/-- The number of pens sold in the first scenario. -/
def num_pens1 : ℕ := 20

/-- The number of pens sold in the second scenario. -/
def num_pens2 : ℕ := 15

theorem cost_price_is_4 : 
  num_pens1 * (retail_price1 - cost_price) = num_pens2 * (retail_price2 - cost_price) → 
  cost_price = 4 := by
  sorry

end cost_price_is_4_l2075_207570


namespace kristine_has_more_cd_difference_l2075_207568

/-- The number of CDs Dawn has -/
def dawn_cds : ℕ := 10

/-- The total number of CDs Kristine and Dawn have together -/
def total_cds : ℕ := 27

/-- Kristine's CDs -/
def kristine_cds : ℕ := total_cds - dawn_cds

/-- The statement that Kristine has more CDs than Dawn -/
theorem kristine_has_more : kristine_cds > dawn_cds := by sorry

/-- The main theorem: Kristine has 7 more CDs than Dawn -/
theorem cd_difference : kristine_cds - dawn_cds = 7 := by sorry

end kristine_has_more_cd_difference_l2075_207568


namespace sunglasses_and_caps_probability_l2075_207596

theorem sunglasses_and_caps_probability 
  (total_sunglasses : ℕ) 
  (total_caps : ℕ) 
  (prob_cap_and_sunglasses : ℚ) :
  total_sunglasses = 60 →
  total_caps = 40 →
  prob_cap_and_sunglasses = 2/5 →
  (prob_cap_and_sunglasses * total_caps) / total_sunglasses = 4/15 := by
  sorry

end sunglasses_and_caps_probability_l2075_207596


namespace product_and_closest_value_l2075_207582

def calculate_product : ℝ := 2.5 * (53.6 - 0.4)

def options : List ℝ := [120, 130, 133, 140, 150]

theorem product_and_closest_value :
  calculate_product = 133 ∧
  ∀ x ∈ options, |calculate_product - 133| ≤ |calculate_product - x| :=
by sorry

end product_and_closest_value_l2075_207582


namespace distance_to_point_l2075_207533

theorem distance_to_point : ∀ (x y : ℝ), x = 7 ∧ y = -24 →
  Real.sqrt (x^2 + y^2) = 25 := by
  sorry

end distance_to_point_l2075_207533


namespace max_gcd_sum_1023_l2075_207589

theorem max_gcd_sum_1023 :
  ∃ (c d : ℕ+), c + d = 1023 ∧
  ∀ (x y : ℕ+), x + y = 1023 → Nat.gcd x y ≤ Nat.gcd c d ∧
  Nat.gcd c d = 341 :=
sorry

end max_gcd_sum_1023_l2075_207589


namespace orange_cost_l2075_207541

/-- Given the cost of 3 dozen oranges, calculate the cost of 5 dozen oranges at the same rate -/
theorem orange_cost (cost_3_dozen : ℝ) (h : cost_3_dozen = 28.80) :
  let cost_per_dozen := cost_3_dozen / 3
  let cost_5_dozen := 5 * cost_per_dozen
  cost_5_dozen = 48 := by
  sorry

end orange_cost_l2075_207541


namespace program_output_l2075_207566

theorem program_output : 
  let initial_value := 2
  let after_multiplication := initial_value * 2
  let final_value := after_multiplication + 6
  final_value = 10 := by sorry

end program_output_l2075_207566


namespace abs_neg_three_equals_three_l2075_207510

theorem abs_neg_three_equals_three : abs (-3 : ℝ) = 3 := by
  sorry

end abs_neg_three_equals_three_l2075_207510


namespace triangle_area_l2075_207564

/-- The area of a triangle with vertices at (2,1,0), (3,3,2), and (5,8,1) is √170/2 -/
theorem triangle_area : ℝ := by
  -- Define the vertices of the triangle
  let a : Fin 3 → ℝ := ![2, 1, 0]
  let b : Fin 3 → ℝ := ![3, 3, 2]
  let c : Fin 3 → ℝ := ![5, 8, 1]

  -- Calculate the area using the cross product method
  let area := (1/2 : ℝ) * Real.sqrt ((b 0 - a 0) * (c 1 - a 1) - (b 1 - a 1) * (c 0 - a 0))^2 +
                                    ((b 1 - a 1) * (c 2 - a 2) - (b 2 - a 2) * (c 1 - a 1))^2 +
                                    ((b 2 - a 2) * (c 0 - a 0) - (b 0 - a 0) * (c 2 - a 2))^2

  -- Prove that the calculated area equals √170/2
  have : area = Real.sqrt 170 / 2 := by sorry

  exact area

end triangle_area_l2075_207564


namespace complex_point_in_first_quadrant_l2075_207577

theorem complex_point_in_first_quadrant : 
  let z : ℂ := (1 - 2*I)^3 / I
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end complex_point_in_first_quadrant_l2075_207577


namespace triangle_shape_l2075_207518

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_shape (t : Triangle) (h : t.a * Real.cos t.A = t.b * Real.cos t.B) :
  (t.A = t.B) ∨ (t.A + t.B = Real.pi / 2) :=
sorry

end triangle_shape_l2075_207518


namespace solve_exponential_equation_l2075_207578

theorem solve_exponential_equation :
  ∃ x : ℤ, (2^x : ℝ) - (2^(x-2) : ℝ) = 3 * (2^10 : ℝ) ∧ x = 12 :=
by sorry

end solve_exponential_equation_l2075_207578


namespace triangle_side_length_bound_l2075_207579

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where the area S = √3/4 * (a² + c² - b²) and b = √3,
    prove that (√3 - 1)a + 2c is bounded by (3 - √3, 2√6]. -/
theorem triangle_side_length_bound (a c : ℝ) (h_positive : a > 0 ∧ c > 0) :
  let b := Real.sqrt 3
  let S := Real.sqrt 3 / 4 * (a^2 + c^2 - b^2)
  3 - Real.sqrt 3 < (Real.sqrt 3 - 1) * a + 2 * c ∧
  (Real.sqrt 3 - 1) * a + 2 * c ≤ 2 * Real.sqrt 6 := by
  sorry

end triangle_side_length_bound_l2075_207579


namespace equation_holds_l2075_207565

theorem equation_holds (a b c : ℝ) (h : a^2 + c^2 = 2*b^2) : 
  (a+b)*(a+c) + (c+a)*(c+b) = 2*(b+a)*(b+c) := by
  sorry

end equation_holds_l2075_207565


namespace band_earnings_theorem_l2075_207540

/-- Represents a band with its earnings and gig information -/
structure Band where
  members : ℕ
  totalEarnings : ℕ
  gigs : ℕ

/-- Calculates the earnings per member per gig for a given band -/
def earningsPerMemberPerGig (b : Band) : ℚ :=
  (b.totalEarnings : ℚ) / (b.members : ℚ) / (b.gigs : ℚ)

/-- Theorem: For a band with 4 members that earned $400 after 5 gigs, 
    each member earns $20 per gig -/
theorem band_earnings_theorem (b : Band) 
    (h1 : b.members = 4) 
    (h2 : b.totalEarnings = 400) 
    (h3 : b.gigs = 5) : 
  earningsPerMemberPerGig b = 20 := by
  sorry


end band_earnings_theorem_l2075_207540


namespace contained_circle_radius_l2075_207587

/-- An isosceles trapezoid with specific dimensions -/
structure IsoscelesTrapezoid where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  isIsosceles : BC = DA
  dimensionsGiven : AB = 6 ∧ BC = 5 ∧ CD = 4

/-- Circles centered at the vertices of the trapezoid -/
structure VertexCircles where
  radiusAB : ℝ
  radiusCD : ℝ
  radiusGiven : radiusAB = 3 ∧ radiusCD = 2

/-- A circle contained within and tangent to all vertex circles -/
structure ContainedCircle where
  radius : ℝ
  isTangent : True  -- Placeholder for tangency condition

/-- The main theorem -/
theorem contained_circle_radius 
  (t : IsoscelesTrapezoid) 
  (v : VertexCircles) 
  (c : ContainedCircle) : 
  c.radius = (-60 + 48 * Real.sqrt 3) / 23 :=
sorry

end contained_circle_radius_l2075_207587


namespace equation_solutions_l2075_207599

theorem equation_solutions :
  (∀ x : ℚ, x + 1/4 = 7/4 → x = 3/2) ∧
  (∀ x : ℚ, 2/3 + x = 3/4 → x = 1/12) := by
sorry

end equation_solutions_l2075_207599


namespace evaluate_expression_l2075_207585

theorem evaluate_expression (x y : ℝ) (hx : x = 3) (hy : y = 2) :
  4 * x^y - 5 * y^x = -4 := by sorry

end evaluate_expression_l2075_207585


namespace equation_solution_l2075_207598

theorem equation_solution : ∃! x : ℝ, (x^2 + 2*x + 3) / (x^2 - 1) = x + 3 :=
by
  -- The unique solution is x = 1
  use 1
  constructor
  -- Prove that x = 1 satisfies the equation
  · sorry
  -- Prove that any other solution must equal 1
  · sorry

end equation_solution_l2075_207598


namespace circular_view_not_rectangular_prism_l2075_207514

/-- A geometric body in three-dimensional space. -/
class GeometricBody :=
(has_circular_view : Bool)

/-- A Rectangular Prism is a type of GeometricBody. -/
def RectangularPrism : GeometricBody :=
{ has_circular_view := false }

/-- Theorem: If a geometric body has a circular view from some direction, it cannot be a Rectangular Prism. -/
theorem circular_view_not_rectangular_prism (body : GeometricBody) :
  body.has_circular_view → body ≠ RectangularPrism :=
sorry

end circular_view_not_rectangular_prism_l2075_207514


namespace green_apples_count_l2075_207539

/-- Given a basket with red and green apples, prove the number of green apples. -/
theorem green_apples_count (total : ℕ) (red : ℕ) (green : ℕ) : 
  total = 9 → red = 7 → green = total - red → green = 2 := by sorry

end green_apples_count_l2075_207539


namespace max_value_constraint_l2075_207501

theorem max_value_constraint (a b c : ℝ) (h : 9*a^2 + 4*b^2 + 25*c^2 = 1) :
  (8*a + 5*b + 15*c) ≤ Real.sqrt 115 / 2 :=
by sorry

end max_value_constraint_l2075_207501


namespace toy_box_problem_l2075_207597

/-- The time taken to put all toys in the box -/
def time_to_fill_box (total_toys : ℕ) (toys_in_per_cycle : ℕ) (toys_out_per_cycle : ℕ) (cycle_time : ℕ) : ℕ := 
  sorry

/-- The problem statement -/
theorem toy_box_problem :
  let total_toys : ℕ := 50
  let toys_in_per_cycle : ℕ := 5
  let toys_out_per_cycle : ℕ := 3
  let cycle_time_seconds : ℕ := 45
  let minutes_per_hour : ℕ := 60
  time_to_fill_box total_toys toys_in_per_cycle toys_out_per_cycle cycle_time_seconds = 18 * minutes_per_hour :=
by sorry

end toy_box_problem_l2075_207597


namespace max_figures_9x9_grid_l2075_207529

/-- Represents a square grid -/
structure Grid (n : ℕ) :=
  (size : ℕ)
  (h_size : size = n)

/-- Represents a square figure -/
structure Figure (m : ℕ) :=
  (size : ℕ)
  (h_size : size = m)

/-- The maximum number of non-overlapping figures that can fit in a grid -/
def max_figures (g : Grid n) (f : Figure m) : ℕ :=
  (g.size / f.size) ^ 2

/-- Theorem: The maximum number of non-overlapping 2x2 squares in a 9x9 grid is 16 -/
theorem max_figures_9x9_grid :
  ∀ (g : Grid 9) (f : Figure 2),
    max_figures g f = 16 := by
  sorry

end max_figures_9x9_grid_l2075_207529


namespace inequality_equivalence_l2075_207561

theorem inequality_equivalence (y : ℝ) : 
  3/20 + |y - 1/5| < 1/4 ↔ y ∈ Set.Ioo (1/10 : ℝ) (3/10 : ℝ) := by
  sorry

end inequality_equivalence_l2075_207561


namespace mountain_climb_fraction_l2075_207531

theorem mountain_climb_fraction (mountain_height : ℕ) (num_trips : ℕ) (total_distance : ℕ)
  (h1 : mountain_height = 40000)
  (h2 : num_trips = 10)
  (h3 : total_distance = 600000) :
  (total_distance / (2 * num_trips)) / mountain_height = 3 / 4 := by
  sorry

end mountain_climb_fraction_l2075_207531


namespace complex_equation_solution_l2075_207513

theorem complex_equation_solution (a : ℝ) : 
  (2 + a * Complex.I) / (1 + Complex.I) = (3 : ℂ) + Complex.I → a = 4 := by
  sorry

end complex_equation_solution_l2075_207513


namespace german_students_count_l2075_207590

theorem german_students_count (total : ℕ) (french : ℕ) (both : ℕ) (neither : ℕ) :
  total = 79 →
  french = 41 →
  both = 9 →
  neither = 25 →
  ∃ german : ℕ, german = 22 ∧ 
    total = french + german - both + neither :=
by sorry

end german_students_count_l2075_207590


namespace parabola_vertex_l2075_207551

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by the equation y^2 - 8y + 4x = 12 -/
def Parabola := {p : Point | p.y^2 - 8*p.y + 4*p.x = 12}

/-- The vertex of a parabola -/
def vertex : Point := ⟨7, 4⟩

/-- Theorem stating that the vertex of the parabola is (7, 4) -/
theorem parabola_vertex : vertex ∈ Parabola ∧ ∀ p ∈ Parabola, p.x ≥ vertex.x := by
  sorry

#check parabola_vertex

end parabola_vertex_l2075_207551


namespace problem_statement_l2075_207552

open Set Real

def M (a : ℝ) : Set ℝ := {x | (x + a) * (x - 1) ≤ 0}
def N : Set ℝ := {x | 4 * x^2 - 4 * x - 3 < 0}

theorem problem_statement (a : ℝ) (h : a > 0) :
  (M a ∪ N = Icc (-2) (3/2) → a = 2) ∧
  (N ∪ (univ \ M a) = univ → 0 < a ∧ a ≤ 1/2) :=
sorry

end problem_statement_l2075_207552
