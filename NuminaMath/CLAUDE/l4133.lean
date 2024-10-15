import Mathlib

namespace NUMINAMATH_CALUDE_incorrect_statement_l4133_413358

-- Define propositions P and Q
def P : Prop := 2 + 2 = 5
def Q : Prop := 3 > 2

-- Theorem stating that the incorrect statement is "'P and Q' is false, 'not P' is false"
theorem incorrect_statement :
  ¬((P ∧ Q → False) ∧ (¬P → False)) :=
by
  sorry


end NUMINAMATH_CALUDE_incorrect_statement_l4133_413358


namespace NUMINAMATH_CALUDE_composition_central_symmetries_is_translation_translation_then_symmetry_is_symmetry_symmetry_then_translation_is_symmetry_l4133_413378

-- Define central symmetry
def central_symmetry (O : ℝ × ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  (2 * O.1 - P.1, 2 * O.2 - P.2)

-- Define parallel translation
def parallel_translation (a : ℝ × ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 + a.1, P.2 + a.2)

-- Theorem 1: Composition of two central symmetries is a parallel translation
theorem composition_central_symmetries_is_translation 
  (O₁ O₂ : ℝ × ℝ) (P : ℝ × ℝ) :
  ∃ a : ℝ × ℝ, central_symmetry O₂ (central_symmetry O₁ P) = parallel_translation a P :=
sorry

-- Theorem 2a: Composition of translation and central symmetry is a central symmetry
theorem translation_then_symmetry_is_symmetry 
  (a : ℝ × ℝ) (O : ℝ × ℝ) (P : ℝ × ℝ) :
  ∃ O' : ℝ × ℝ, central_symmetry O (parallel_translation a P) = central_symmetry O' P :=
sorry

-- Theorem 2b: Composition of central symmetry and translation is a central symmetry
theorem symmetry_then_translation_is_symmetry 
  (O : ℝ × ℝ) (a : ℝ × ℝ) (P : ℝ × ℝ) :
  ∃ O' : ℝ × ℝ, parallel_translation a (central_symmetry O P) = central_symmetry O' P :=
sorry

end NUMINAMATH_CALUDE_composition_central_symmetries_is_translation_translation_then_symmetry_is_symmetry_symmetry_then_translation_is_symmetry_l4133_413378


namespace NUMINAMATH_CALUDE_no_solution_exists_l4133_413393

-- Define the system of equations
def system (a b c d : ℝ) : Prop :=
  a^3 + c^3 = 2 ∧
  a^2*b + c^2*d = 0 ∧
  b^3 + d^3 = 1 ∧
  a*b^2 + c*d^2 = -6

-- Theorem stating that no solution exists
theorem no_solution_exists : ¬∃ (a b c d : ℝ), system a b c d := by
  sorry


end NUMINAMATH_CALUDE_no_solution_exists_l4133_413393


namespace NUMINAMATH_CALUDE_cube_angle_range_l4133_413341

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Calculates the angle between two vectors -/
def angle (v1 v2 : Point3D) : ℝ := sorry

/-- Theorem: The angle between A₁M and C₁N is in the range (π/3, π/2) -/
theorem cube_angle_range (cube : Cube) (M N : Point3D) 
  (h_M : M.x > cube.A.x ∧ M.x < cube.B.x ∧ M.y = cube.A.y ∧ M.z = cube.A.z)
  (h_N : N.x = cube.B.x ∧ N.y > cube.B.y ∧ N.y < cube.B₁.y ∧ N.z = cube.B.z)
  (h_AM_eq_B₁N : (M.x - cube.A.x)^2 = (cube.B₁.y - N.y)^2) :
  let θ := angle (Point3D.mk (cube.A₁.x - M.x) (cube.A₁.y - M.y) (cube.A₁.z - M.z))
              (Point3D.mk (cube.C₁.x - N.x) (cube.C₁.y - N.y) (cube.C₁.z - N.z))
  π/3 < θ ∧ θ < π/2 := by
  sorry

end NUMINAMATH_CALUDE_cube_angle_range_l4133_413341


namespace NUMINAMATH_CALUDE_doll_collection_increase_l4133_413397

theorem doll_collection_increase (original_count : ℕ) (increase : ℕ) (final_count : ℕ) :
  original_count + increase = final_count →
  final_count = 10 →
  increase = 2 →
  (increase : ℚ) / (original_count : ℚ) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_doll_collection_increase_l4133_413397


namespace NUMINAMATH_CALUDE_triangle_area_l4133_413379

def a : ℝ × ℝ := (7, 3)
def b : ℝ × ℝ := (-1, 5)

theorem triangle_area : 
  let det := a.1 * b.2 - a.2 * b.1
  (1/2 : ℝ) * |det| = 19 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l4133_413379


namespace NUMINAMATH_CALUDE_binomial_coeff_equality_l4133_413384

def binomial_coeff (n m : ℕ) : ℕ := Nat.choose n m

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem binomial_coeff_equality (n m : ℕ) :
  binomial_coeff n (m - 1) = binomial_coeff (n - 1) m ↔
  ∃ k : ℕ, n = fibonacci (2 * k) * fibonacci (2 * k + 1) ∧
            m = fibonacci (2 * k) * fibonacci (2 * k - 1) :=
sorry

end NUMINAMATH_CALUDE_binomial_coeff_equality_l4133_413384


namespace NUMINAMATH_CALUDE_salary_before_raise_l4133_413370

theorem salary_before_raise (new_salary : ℝ) (increase_percentage : ℝ) (old_salary : ℝ) :
  new_salary = 70 →
  increase_percentage = 16.666666666666664 →
  old_salary * (1 + increase_percentage / 100) = new_salary →
  old_salary = 60 := by
sorry

end NUMINAMATH_CALUDE_salary_before_raise_l4133_413370


namespace NUMINAMATH_CALUDE_smallest_value_l4133_413362

theorem smallest_value (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  x^3 < x ∧ x^3 < 3*x ∧ x^3 < x^(1/3) ∧ x^3 < 1/(x+1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_l4133_413362


namespace NUMINAMATH_CALUDE_direct_proportion_increases_l4133_413344

theorem direct_proportion_increases (x₁ x₂ : ℝ) (h : x₁ < x₂) : 2 * x₁ < 2 * x₂ := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_increases_l4133_413344


namespace NUMINAMATH_CALUDE_vertex_on_x_axis_l4133_413367

/-- The vertex of the parabola y = x^2 - 6x + c lies on the x-axis if and only if c = 9 -/
theorem vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + c = 0 ∧ ∀ y : ℝ, y^2 - 6*y + c ≥ x^2 - 6*x + c) ↔ c = 9 := by
  sorry

end NUMINAMATH_CALUDE_vertex_on_x_axis_l4133_413367


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l4133_413369

structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  isIsosceles : side > 0

def medianDividesDifference (t : IsoscelesTriangle) (diff : ℝ) : Prop :=
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = t.base + 2 * t.side ∧ |x - y| = diff

theorem isosceles_triangle_side_length 
  (t : IsoscelesTriangle) 
  (h_base : t.base = 7) 
  (h_median : medianDividesDifference t 3) : 
  t.side = 4 ∨ t.side = 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l4133_413369


namespace NUMINAMATH_CALUDE_odd_function_m_value_l4133_413335

/-- Given a > 0 and a ≠ 1, if f(x) = 1/(a^x + 1) - m is an odd function, then m = 1/2 -/
theorem odd_function_m_value (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := λ x => 1 / (a^x + 1) - m
  (∀ x, f x + f (-x) = 0) → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_m_value_l4133_413335


namespace NUMINAMATH_CALUDE_expression_value_l4133_413304

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 4) :
  3 * x - 5 * y + 7 = -4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4133_413304


namespace NUMINAMATH_CALUDE_hcl_formation_l4133_413326

/-- Represents a chemical compound with its coefficient in a chemical equation -/
structure Compound where
  name : String
  coefficient : ℚ

/-- Represents a chemical equation with reactants and products -/
structure ChemicalEquation where
  reactants : List Compound
  products : List Compound

/-- Calculates the number of moles of HCl formed given the initial moles of reactants -/
def molesOfHClFormed (h2so4_moles : ℚ) (nacl_moles : ℚ) (equation : ChemicalEquation) : ℚ :=
  sorry

/-- The main theorem stating that 3 moles of HCl are formed -/
theorem hcl_formation :
  let equation : ChemicalEquation := {
    reactants := [
      {name := "H₂SO₄", coefficient := 1},
      {name := "NaCl", coefficient := 2}
    ],
    products := [
      {name := "HCl", coefficient := 2},
      {name := "Na₂SO₄", coefficient := 1}
    ]
  }
  molesOfHClFormed 3 3 equation = 3 :=
by sorry

end NUMINAMATH_CALUDE_hcl_formation_l4133_413326


namespace NUMINAMATH_CALUDE_min_sum_of_product_144_l4133_413322

theorem min_sum_of_product_144 :
  ∀ c d : ℤ, c * d = 144 → (∀ x y : ℤ, x * y = 144 → c + d ≤ x + y) ∧ (∃ a b : ℤ, a * b = 144 ∧ a + b = -145) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_144_l4133_413322


namespace NUMINAMATH_CALUDE_duck_count_proof_l4133_413340

/-- The number of mallard ducks initially at the park -/
def initial_ducks : ℕ := sorry

/-- The number of geese initially at the park -/
def initial_geese : ℕ := 2 * initial_ducks - 10

/-- The number of ducks after the small flock arrives -/
def ducks_after_arrival : ℕ := initial_ducks + 4

/-- The number of geese after some leave -/
def geese_after_leaving : ℕ := initial_geese - 10

theorem duck_count_proof : 
  initial_ducks = 25 ∧ 
  geese_after_leaving = ducks_after_arrival + 1 :=
sorry

end NUMINAMATH_CALUDE_duck_count_proof_l4133_413340


namespace NUMINAMATH_CALUDE_length_of_BC_l4133_413311

-- Define the parabola
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.A = (2, 16) ∧
  parabola 2 = 16 ∧
  t.B.1 = -t.C.1 ∧
  t.B.2 = t.C.2 ∧
  (1/2 : ℝ) * |t.B.1 - t.C.1| * |t.A.2 - t.B.2| = 128

-- Theorem statement
theorem length_of_BC (t : Triangle) (h : satisfies_conditions t) : 
  |t.B.1 - t.C.1| = 8 := by sorry

end NUMINAMATH_CALUDE_length_of_BC_l4133_413311


namespace NUMINAMATH_CALUDE_tenth_digit_theorem_l4133_413308

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def some_number : ℕ := 6840

theorem tenth_digit_theorem :
  (((factorial 5 * factorial 5 - factorial 5 * factorial 3) / some_number) % 100) / 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_tenth_digit_theorem_l4133_413308


namespace NUMINAMATH_CALUDE_sum_of_thousands_and_units_digits_l4133_413317

/-- Represents a 100-digit number with a repeating pattern --/
def RepeatNumber (a b : ℕ) := ℕ

/-- The first 100-digit number: 606060606...060606 --/
def num1 : RepeatNumber 60 6 := sorry

/-- The second 100-digit number: 808080808...080808 --/
def num2 : RepeatNumber 80 8 := sorry

/-- Returns the units digit of a number --/
def unitsDigit (n : ℕ) : ℕ := sorry

/-- Returns the thousands digit of a number --/
def thousandsDigit (n : ℕ) : ℕ := sorry

/-- The product of num1 and num2 --/
def product : ℕ := sorry

theorem sum_of_thousands_and_units_digits :
  thousandsDigit product + unitsDigit product = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_thousands_and_units_digits_l4133_413317


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l4133_413359

theorem arithmetic_evaluation : 3 + 2 * (8 - 3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l4133_413359


namespace NUMINAMATH_CALUDE_neighborhood_cable_cost_l4133_413313

/-- Calculate the total cost of power cable for a neighborhood --/
theorem neighborhood_cable_cost
  (east_west_streets : ℕ)
  (north_south_streets : ℕ)
  (east_west_length : ℝ)
  (north_south_length : ℝ)
  (cable_per_street_mile : ℝ)
  (cable_cost_per_mile : ℝ)
  (h1 : east_west_streets = 18)
  (h2 : north_south_streets = 10)
  (h3 : east_west_length = 2)
  (h4 : north_south_length = 4)
  (h5 : cable_per_street_mile = 5)
  (h6 : cable_cost_per_mile = 2000) :
  east_west_streets * east_west_length * cable_per_street_mile * cable_cost_per_mile +
  north_south_streets * north_south_length * cable_per_street_mile * cable_cost_per_mile =
  760000 :=
by sorry

end NUMINAMATH_CALUDE_neighborhood_cable_cost_l4133_413313


namespace NUMINAMATH_CALUDE_distance_a_beats_b_proof_l4133_413391

/-- The distance A can beat B when running 4.5 km -/
def distance_a_beats_b (a_speed : ℝ) (time_diff : ℝ) : ℝ :=
  a_speed * time_diff

/-- Theorem stating that the distance A beats B is equal to A's speed multiplied by the time difference -/
theorem distance_a_beats_b_proof (a_speed : ℝ) (time_diff : ℝ) (a_time : ℝ) (b_time : ℝ) 
    (h1 : a_speed = 4.5 / a_time)
    (h2 : time_diff = b_time - a_time)
    (h3 : a_time = 90)
    (h4 : b_time = 180) :
  distance_a_beats_b a_speed time_diff = 4.5 := by
  sorry

#check distance_a_beats_b_proof

end NUMINAMATH_CALUDE_distance_a_beats_b_proof_l4133_413391


namespace NUMINAMATH_CALUDE_existence_of_h_l4133_413363

theorem existence_of_h : ∃ h : ℝ, ∀ n : ℕ, 
  ¬(⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋) := by sorry

end NUMINAMATH_CALUDE_existence_of_h_l4133_413363


namespace NUMINAMATH_CALUDE_unbroken_matches_count_l4133_413390

def dozen : ℕ := 12
def boxes_count : ℕ := 5 * dozen
def matches_per_box : ℕ := 20
def broken_matches_per_box : ℕ := 3

theorem unbroken_matches_count :
  boxes_count * (matches_per_box - broken_matches_per_box) = 1020 :=
by sorry

end NUMINAMATH_CALUDE_unbroken_matches_count_l4133_413390


namespace NUMINAMATH_CALUDE_volleyball_team_selection_count_l4133_413388

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 6 starters from a team of 15 players,
    including 4 quadruplets, with at least two quadruplets in the starting lineup -/
def volleyball_team_selection : ℕ :=
  choose 4 2 * choose 11 4 +
  choose 4 3 * choose 11 3 +
  choose 11 2

theorem volleyball_team_selection_count :
  volleyball_team_selection = 2695 := by sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_count_l4133_413388


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_l4133_413381

theorem fraction_sum_equals_one (x y : ℝ) (h : x + y ≠ 0) :
  x / (x + y) + y / (x + y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_l4133_413381


namespace NUMINAMATH_CALUDE_root_sum_cubes_l4133_413373

theorem root_sum_cubes (a b c : ℝ) : 
  (a^3 + 14*a^2 + 49*a + 36 = 0) → 
  (b^3 + 14*b^2 + 49*b + 36 = 0) → 
  (c^3 + 14*c^2 + 49*c + 36 = 0) → 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 686 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_cubes_l4133_413373


namespace NUMINAMATH_CALUDE_power_sum_fourth_l4133_413303

theorem power_sum_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_fourth_l4133_413303


namespace NUMINAMATH_CALUDE_modulus_of_z_l4133_413355

theorem modulus_of_z (z : ℂ) (h : z * (1 + Complex.I) = 1 + 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l4133_413355


namespace NUMINAMATH_CALUDE_system_solutions_l4133_413300

theorem system_solutions (a : ℤ) :
  let eq1 := fun (x y z : ℤ) => 5 * x + (a + 2) * y + (a + 2) * z = a
  let eq2 := fun (x y z : ℤ) => (2 * a + 4) * x + (a^2 + 3) * y + (2 * a + 2) * z = 3 * a - 1
  let eq3 := fun (x y z : ℤ) => (2 * a + 4) * x + (2 * a + 2) * y + (a^2 + 3) * z = a + 1
  (∀ x y z : ℤ, eq1 x y z ∧ eq2 x y z ∧ eq3 x y z ↔
    (a = 1 ∧ ∃ n : ℤ, x = -1 ∧ y = n ∧ z = 2 - n) ∨
    (a = -1 ∧ x = 0 ∧ y = -1 ∧ z = 0) ∨
    (a = 0 ∧ x = 0 ∧ y = -1 ∧ z = 1) ∨
    (a = 2 ∧ x = -6 ∧ y = 5 ∧ z = 3)) ∧
  (a = 3 → ¬∃ x y z : ℤ, eq1 x y z ∧ eq2 x y z ∧ eq3 x y z) ∧
  (a ≠ 1 ∧ a ≠ -1 ∧ a ≠ 0 ∧ a ≠ 2 ∧ a ≠ 3 →
    ¬∃ x y z : ℤ, eq1 x y z ∧ eq2 x y z ∧ eq3 x y z) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l4133_413300


namespace NUMINAMATH_CALUDE_vector_operation_l4133_413361

def a : Fin 2 → ℝ := ![2, 4]
def b : Fin 2 → ℝ := ![-1, 1]

theorem vector_operation : 
  (2 • a - b) = ![5, 7] := by sorry

end NUMINAMATH_CALUDE_vector_operation_l4133_413361


namespace NUMINAMATH_CALUDE_elevator_problem_l4133_413302

theorem elevator_problem (x y z w v a b c n : ℕ) : 
  x = 20 ∧ 
  y = 7 ∧ 
  z = 3^2 ∧ 
  w = 5^2 ∧ 
  v = 3^2 ∧ 
  a = 3^2 - 2 ∧ 
  b = 3 ∧ 
  c = 1^3 ∧ 
  x - y + z - w + v - a + b - c = n 
  → n = 1 := by
sorry

end NUMINAMATH_CALUDE_elevator_problem_l4133_413302


namespace NUMINAMATH_CALUDE_maria_furniture_assembly_time_l4133_413328

def total_time (num_chairs num_tables time_per_piece : ℕ) : ℕ :=
  (num_chairs + num_tables) * time_per_piece

theorem maria_furniture_assembly_time :
  total_time 2 2 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_maria_furniture_assembly_time_l4133_413328


namespace NUMINAMATH_CALUDE_whale_population_prediction_l4133_413399

theorem whale_population_prediction (whales_last_year whales_this_year whales_next_year predicted_increase : ℕ) : 
  whales_last_year = 4000 →
  whales_this_year = 2 * whales_last_year →
  predicted_increase = 800 →
  whales_next_year = whales_this_year + predicted_increase →
  whales_next_year = 8800 := by
sorry

end NUMINAMATH_CALUDE_whale_population_prediction_l4133_413399


namespace NUMINAMATH_CALUDE_bryden_payment_proof_l4133_413320

/-- The amount a collector pays for state quarters as a percentage of face value -/
def collector_payment_percentage : ℝ := 1500

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 7

/-- The face value of a single state quarter in dollars -/
def quarter_face_value : ℝ := 0.25

/-- The amount Bryden receives for his quarters in dollars -/
def bryden_payment : ℝ := 26.25

theorem bryden_payment_proof :
  (collector_payment_percentage / 100) * (bryden_quarters * quarter_face_value) = bryden_payment :=
by sorry

end NUMINAMATH_CALUDE_bryden_payment_proof_l4133_413320


namespace NUMINAMATH_CALUDE_printer_depreciation_l4133_413387

def initial_price : ℝ := 625000
def first_year_depreciation : ℝ := 0.20
def subsequent_depreciation : ℝ := 0.08
def target_value : ℝ := 400000

def resale_value (n : ℕ) : ℝ :=
  if n = 0 then initial_price
  else if n = 1 then initial_price * (1 - first_year_depreciation)
  else (resale_value (n - 1)) * (1 - subsequent_depreciation)

theorem printer_depreciation :
  resale_value 4 < target_value ∧
  ∀ k : ℕ, k < 4 → resale_value k ≥ target_value :=
sorry

end NUMINAMATH_CALUDE_printer_depreciation_l4133_413387


namespace NUMINAMATH_CALUDE_grandfather_pension_increase_l4133_413332

/-- Represents the percentage increase in family income when a member's income is doubled -/
structure IncomeIncrease where
  masha : ℝ
  mother : ℝ
  father : ℝ

/-- Calculates the percentage increase in family income when grandfather's pension is doubled -/
def grandfather_increase (i : IncomeIncrease) : ℝ :=
  100 - (i.masha + i.mother + i.father)

/-- Theorem stating that given the specified income increases for Masha, mother, and father,
    doubling grandfather's pension will increase the family income by 55% -/
theorem grandfather_pension_increase (i : IncomeIncrease) 
  (h1 : i.masha = 5)
  (h2 : i.mother = 15)
  (h3 : i.father = 25) :
  grandfather_increase i = 55 := by
  sorry

#eval grandfather_increase { masha := 5, mother := 15, father := 25 }

end NUMINAMATH_CALUDE_grandfather_pension_increase_l4133_413332


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4133_413330

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_properties
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h1 : a 5 = -1)
  (h2 : a 8 = 2)
  (m n : ℕ+)
  (h3 : m ≠ n)
  (h4 : a m = n)
  (h5 : a n = m) :
  (a 1 = -5 ∧ ∃ d : ℤ, d = 1 ∧ ∀ k : ℕ, a (k + 1) = a k + d) ∧
  a (m + n) = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4133_413330


namespace NUMINAMATH_CALUDE_sandy_has_144_marbles_l4133_413336

def dozen : ℕ := 12

def jessica_marbles : ℕ := 3 * dozen

def sandy_marbles : ℕ := 4 * jessica_marbles

theorem sandy_has_144_marbles : sandy_marbles = 144 := by
  sorry

end NUMINAMATH_CALUDE_sandy_has_144_marbles_l4133_413336


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_squared_l4133_413338

/-- The square of the side length of an equilateral triangle inscribed in a specific circle -/
theorem equilateral_triangle_side_length_squared (x y : ℝ) : 
  x^2 + y^2 = 16 →  -- Circle equation
  (0 : ℝ)^2 + 4^2 = 16 →  -- One vertex at (0, 4)
  (∃ a b : ℝ, a^2 + b^2 = 16 ∧ (0 - a)^2 + (4 - b)^2 = a^2 + (4 - b)^2) →  -- Triangle inscribed in circle
  (∃ c : ℝ, c^2 + (-3)^2 = 16) →  -- Altitude on y-axis (implied by y = -3 for other vertices)
  (0 : ℝ)^2 + 7^2 = 49 :=  -- Square of side length is 49
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_squared_l4133_413338


namespace NUMINAMATH_CALUDE_quadratic_radical_equality_l4133_413327

theorem quadratic_radical_equality :
  ∃! x : ℝ, x^2 - 2 = 2*x - 2 ∧ x^2 - 2 ≥ 0 ∧ 2*x - 2 ≥ 0 ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_radical_equality_l4133_413327


namespace NUMINAMATH_CALUDE_number_order_l4133_413305

-- Define the numbers in their respective bases
def a : ℕ := 33
def b : ℕ := 5 * 6 + 2
def c : ℕ := 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

-- Theorem statement
theorem number_order : a > b ∧ b > c :=
sorry

end NUMINAMATH_CALUDE_number_order_l4133_413305


namespace NUMINAMATH_CALUDE_line_through_points_l4133_413371

-- Define the line equation
def line_equation (a b x : ℝ) : ℝ := a * x + b

-- Define the condition that the line passes through two points
def passes_through (a b : ℝ) : Prop :=
  line_equation a b 3 = 4 ∧ line_equation a b 9 = 22

-- Theorem statement
theorem line_through_points :
  ∀ a b : ℝ, passes_through a b → a - b = 8 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l4133_413371


namespace NUMINAMATH_CALUDE_base_seven_digits_of_1234_l4133_413392

theorem base_seven_digits_of_1234 : ∃ n : ℕ, (7^n ≤ 1234 ∧ ∀ m : ℕ, 7^m ≤ 1234 → m ≤ n) ∧ n + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_digits_of_1234_l4133_413392


namespace NUMINAMATH_CALUDE_square_side_length_l4133_413382

theorem square_side_length (width height : ℝ) (h1 : width = 3320) (h2 : height = 2025) : ∃ (r s : ℝ),
  2 * r + s = height ∧
  2 * r + 3 * s = width ∧
  s = 647.5 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l4133_413382


namespace NUMINAMATH_CALUDE_fraction_equality_l4133_413329

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 3)
  (h3 : p / q = 1 / 5) :
  m / q = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l4133_413329


namespace NUMINAMATH_CALUDE_collinear_points_determine_a_l4133_413339

/-- Given three points A(1,-1), B(a,3), and C(4,5) that are collinear,
    prove that a = 3. -/
theorem collinear_points_determine_a (a : ℝ) :
  let A : ℝ × ℝ := (1, -1)
  let B : ℝ × ℝ := (a, 3)
  let C : ℝ × ℝ := (4, 5)
  (∃ (t : ℝ), B.1 = A.1 + t * (C.1 - A.1) ∧ B.2 = A.2 + t * (C.2 - A.2)) →
  a = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_collinear_points_determine_a_l4133_413339


namespace NUMINAMATH_CALUDE_x_plus_y_value_l4133_413354

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.sin y = 2023)
  (eq2 : x + 2023 * Real.cos y = 2021)
  (y_range : π/4 ≤ y ∧ y ≤ 3*π/4) :
  x + y = 2023 - Real.sqrt 2 / 2 + 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l4133_413354


namespace NUMINAMATH_CALUDE_simplify_expression_l4133_413321

theorem simplify_expression (x y : ℝ) :
  (2 * x + 25) + (150 * x + 40) + (5 * y + 10) = 152 * x + 5 * y + 75 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4133_413321


namespace NUMINAMATH_CALUDE_solve_for_y_l4133_413357

theorem solve_for_y (x y : ℤ) (h1 : x^2 = y - 2) (h2 : x = -6) : y = 38 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l4133_413357


namespace NUMINAMATH_CALUDE_triangle_property_l4133_413342

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_property (t : Triangle) 
  (h : 2 * Real.sin t.B * Real.sin t.C * Real.cos t.A = 1 - Real.cos (2 * t.A)) :
  (t.b^2 + t.c^2) / t.a^2 = 3 ∧ 
  (∀ (t' : Triangle), Real.sin t'.A ≤ Real.sqrt 5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l4133_413342


namespace NUMINAMATH_CALUDE_right_triangle_theorem_l4133_413323

/-- Right triangle DEF with given side lengths and midpoint N on hypotenuse -/
structure RightTriangle where
  /-- Length of side DE -/
  de : ℝ
  /-- Length of side DF -/
  df : ℝ
  /-- Right angle at E -/
  right_angle : de ^ 2 + df ^ 2 = (de + df) ^ 2 / 4
  /-- N is midpoint of EF -/
  n_midpoint : True

/-- Properties of the right triangle -/
def triangle_properties (t : RightTriangle) : Prop :=
  let dn := (t.de ^ 2 + t.df ^ 2).sqrt / 2
  let area := t.de * t.df / 2
  let centroid_distance := 2 * dn / 3
  dn = 5.0 ∧ area = 24.0 ∧ centroid_distance = 3.3

/-- Theorem stating the properties of the specific right triangle -/
theorem right_triangle_theorem :
  ∃ t : RightTriangle, t.de = 6 ∧ t.df = 8 ∧ triangle_properties t :=
sorry

end NUMINAMATH_CALUDE_right_triangle_theorem_l4133_413323


namespace NUMINAMATH_CALUDE_parabolas_intersection_l4133_413331

-- Define the two parabola functions
def f (x : ℝ) : ℝ := 2 * x^2 + 5 * x - 3
def g (x : ℝ) : ℝ := x^2 + 2

-- Theorem statement
theorem parabolas_intersection :
  (∃ (x y : ℝ), f x = g x ∧ y = f x) ↔
  (∃ (x y : ℝ), (x = -5 ∧ y = 27) ∨ (x = 1 ∧ y = 3)) :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l4133_413331


namespace NUMINAMATH_CALUDE_tangent_range_l4133_413383

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- The equation for the tangent line passing through (1, m) and touching the curve at x₀ --/
def tangent_equation (x₀ m : ℝ) : Prop :=
  (x₀^3 - 3*x₀ - m) / (x₀ - 1) = 3*x₀^2 - 3

/-- The condition for exactly three tangent lines --/
def three_tangents (m : ℝ) : Prop :=
  ∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    tangent_equation x₁ m ∧ tangent_equation x₂ m ∧ tangent_equation x₃ m

/-- The main theorem --/
theorem tangent_range :
  ∀ m : ℝ, m ≠ -2 → three_tangents m → -3 < m ∧ m < -2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_range_l4133_413383


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l4133_413306

/-- A line passing through (1, 3) with equal absolute intercepts has one of three specific equations -/
theorem line_through_point_with_equal_intercepts :
  ∀ (f : ℝ → ℝ),
  (f 1 = 3) →  -- Line passes through (1, 3)
  (∃ a : ℝ, a ≠ 0 ∧ f 0 = a ∧ f a = 0) →  -- Equal absolute intercepts
  (∀ x, f x = 3 * x) ∨  -- y = 3x
  (∀ x, x + f x = 4) ∨  -- x + y - 4 = 0
  (∀ x, x - f x = -2)  -- x - y + 2 = 0
  := by sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l4133_413306


namespace NUMINAMATH_CALUDE_robot_purchase_strategy_l4133_413325

/-- The problem of finding optimal robot purchase strategy -/
theorem robot_purchase_strategy 
  (price_difference : ℕ) 
  (cost_A cost_B : ℕ) 
  (total_units : ℕ) 
  (discount_rate : ℚ) : 
  price_difference = 200 →
  cost_A = 2000 →
  cost_B = 1200 →
  total_units = 40 →
  discount_rate = 1/5 →
  ∃ (price_A price_B units_A units_B min_cost : ℕ),
    -- Unit prices
    price_A = 500 ∧ 
    price_B = 300 ∧ 
    price_A = price_B + price_difference ∧
    cost_A * price_B = cost_B * price_A ∧
    -- Optimal purchase strategy
    units_A = 10 ∧
    units_B = 30 ∧
    units_A + units_B = total_units ∧
    units_B ≤ 3 * units_A ∧
    min_cost = 11200 ∧
    min_cost = (price_A * units_A + price_B * units_B) * (1 - discount_rate) ∧
    ∀ (other_A other_B : ℕ), 
      other_A + other_B = total_units →
      other_B ≤ 3 * other_A →
      min_cost ≤ (price_A * other_A + price_B * other_B) * (1 - discount_rate) :=
by sorry

end NUMINAMATH_CALUDE_robot_purchase_strategy_l4133_413325


namespace NUMINAMATH_CALUDE_garden_path_width_l4133_413396

/-- Given two concentric circles with a difference in circumference of 20π meters,
    the width of the path between them is 10 meters. -/
theorem garden_path_width (R r : ℝ) (h : 2 * Real.pi * R - 2 * Real.pi * r = 20 * Real.pi) :
  R - r = 10 := by
  sorry

end NUMINAMATH_CALUDE_garden_path_width_l4133_413396


namespace NUMINAMATH_CALUDE_triangle_inequality_range_l4133_413365

/-- The triangle operation on real numbers -/
def triangle (x y : ℝ) : ℝ := x * (2 - y)

/-- Theorem stating the range of m for which (x + m) △ x < 1 holds for all real x -/
theorem triangle_inequality_range (m : ℝ) :
  (∀ x : ℝ, triangle (x + m) x < 1) ↔ m ∈ Set.Ioo (-4 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_range_l4133_413365


namespace NUMINAMATH_CALUDE_total_hours_theorem_hours_breakdown_theorem_l4133_413319

/-- The number of hours Sangita is required to fly to earn an airplane pilot certificate -/
def required_hours : ℕ := 1320

/-- The number of hours Sangita has already completed -/
def completed_hours : ℕ := 50 + 9 + 121

/-- The number of months Sangita needs to complete her goal -/
def months : ℕ := 6

/-- The number of hours Sangita must fly per month -/
def hours_per_month : ℕ := 220

/-- Theorem stating that the total number of hours Sangita is required to fly
    is equal to the product of months and hours per month -/
theorem total_hours_theorem :
  required_hours = months * hours_per_month :=
by sorry

/-- Theorem stating that the total number of hours Sangita is required to fly
    is equal to the sum of completed hours and remaining hours -/
theorem hours_breakdown_theorem :
  required_hours = completed_hours + (required_hours - completed_hours) :=
by sorry

end NUMINAMATH_CALUDE_total_hours_theorem_hours_breakdown_theorem_l4133_413319


namespace NUMINAMATH_CALUDE_expand_polynomial_l4133_413345

theorem expand_polynomial (a b : ℝ) : (a - b) * (a + b) * (a^2 - b^2) = a^4 - 2*a^2*b^2 + b^4 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l4133_413345


namespace NUMINAMATH_CALUDE_candy_distribution_l4133_413351

theorem candy_distribution (total_children : ℕ) (absent_children : ℕ) (extra_candies : ℕ) :
  total_children = 300 →
  absent_children = 150 →
  extra_candies = 24 →
  (total_children - absent_children) * (total_children / (total_children - absent_children) + extra_candies) = 
    total_children * (48 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l4133_413351


namespace NUMINAMATH_CALUDE_ascending_order_negative_a_l4133_413301

theorem ascending_order_negative_a (a : ℝ) (h1 : -1 < a) (h2 : a < 0) :
  1 / a < a ∧ a < a^2 ∧ a^2 < |a| := by sorry

end NUMINAMATH_CALUDE_ascending_order_negative_a_l4133_413301


namespace NUMINAMATH_CALUDE_min_value_of_f_l4133_413309

/-- The function f(x, y) as defined in the problem -/
def f (x y : ℝ) : ℝ := 6 * (x^2 + y^2) * (x + y) - 4 * (x^2 + x*y + y^2) - 3 * (x + y) + 5

/-- The theorem statement -/
theorem min_value_of_f :
  (∀ x y : ℝ, x > 0 → y > 0 → f x y ≥ 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ f x y = 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l4133_413309


namespace NUMINAMATH_CALUDE_inequality_condition_l4133_413312

theorem inequality_condition (x y : ℝ) : 
  (x > 0 ∧ y > 0 → y / x + x / y ≥ 2) ∧ 
  ¬(y / x + x / y ≥ 2 → x > 0 ∧ y > 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l4133_413312


namespace NUMINAMATH_CALUDE_smallest_integer_square_triple_plus_100_l4133_413347

theorem smallest_integer_square_triple_plus_100 : 
  ∃ (x : ℤ), x^2 = 3*x + 100 ∧ ∀ (y : ℤ), y^2 = 3*y + 100 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_square_triple_plus_100_l4133_413347


namespace NUMINAMATH_CALUDE_songs_per_album_l4133_413375

/-- The number of country albums bought -/
def country_albums : ℕ := 4

/-- The number of pop albums bought -/
def pop_albums : ℕ := 5

/-- The total number of songs bought -/
def total_songs : ℕ := 72

/-- Proves that if all albums have the same number of songs, then each album contains 8 songs -/
theorem songs_per_album :
  ∀ (songs_per_album : ℕ),
  country_albums * songs_per_album + pop_albums * songs_per_album = total_songs →
  songs_per_album = 8 := by
sorry

end NUMINAMATH_CALUDE_songs_per_album_l4133_413375


namespace NUMINAMATH_CALUDE_expression_equivalence_l4133_413346

theorem expression_equivalence (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 1) / x) * ((y^2 + 1) / y) + ((x^2 - 1) / y) * ((y^2 - 1) / x) = 2 * x * y + 2 / (x * y) :=
by sorry

end NUMINAMATH_CALUDE_expression_equivalence_l4133_413346


namespace NUMINAMATH_CALUDE_token_game_ends_in_37_rounds_l4133_413372

/-- Represents a player in the token game -/
inductive Player : Type
| A
| B
| C

/-- The state of the game at any given round -/
structure GameState :=
  (tokens : Player → ℕ)

/-- The initial state of the game -/
def initial_state : GameState :=
  { tokens := fun p => match p with
    | Player.A => 15
    | Player.B => 14
    | Player.C => 13 }

/-- Simulates one round of the game -/
def play_round (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended -/
def game_ended (state : GameState) : Bool :=
  sorry

/-- Counts the number of rounds until the game ends -/
def count_rounds (state : GameState) : ℕ :=
  sorry

theorem token_game_ends_in_37_rounds :
  count_rounds initial_state = 37 :=
sorry

end NUMINAMATH_CALUDE_token_game_ends_in_37_rounds_l4133_413372


namespace NUMINAMATH_CALUDE_tangent_lines_count_l4133_413349

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Counts the number of lines tangent to two circles -/
def count_tangent_lines (c1 c2 : Circle) : ℕ :=
  sorry

theorem tangent_lines_count 
  (A B : ℝ × ℝ)
  (C_A : Circle)
  (C_B : Circle)
  (h_distance : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 7)
  (h_C_A_center : C_A.center = A)
  (h_C_B_center : C_B.center = B)
  (h_C_A_radius : C_A.radius = 3)
  (h_C_B_radius : C_B.radius = 4) :
  count_tangent_lines C_A C_B = 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_count_l4133_413349


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l4133_413386

theorem adult_ticket_cost (num_adults : ℕ) (child_ticket_cost : ℚ) (total_receipts : ℚ) :
  num_adults = 152 →
  child_ticket_cost = 5/2 →
  total_receipts = 1026 →
  ∃ A : ℚ, A * num_adults + child_ticket_cost * (num_adults / 2) = total_receipts ∧ A = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l4133_413386


namespace NUMINAMATH_CALUDE_student_absence_probability_l4133_413316

theorem student_absence_probability :
  let p_absent : ℚ := 1 / 20
  let p_present : ℚ := 1 - p_absent
  let p_two_absent_one_present : ℚ := 3 * (p_absent * p_absent * p_present)
  p_two_absent_one_present = 57 / 8000 := by
  sorry

end NUMINAMATH_CALUDE_student_absence_probability_l4133_413316


namespace NUMINAMATH_CALUDE_ratio_difference_l4133_413333

theorem ratio_difference (a b : ℕ) (ha : a > 5) (hb : b > 5) : 
  (a : ℚ) / b = 6 / 5 → (a - 5 : ℚ) / (b - 5) = 5 / 4 → a - b = 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_difference_l4133_413333


namespace NUMINAMATH_CALUDE_scientific_notation_393000_l4133_413314

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_393000 :
  toScientificNotation 393000 = ScientificNotation.mk 3.93 5 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_393000_l4133_413314


namespace NUMINAMATH_CALUDE_razorback_tshirt_sales_l4133_413352

/-- The number of t-shirts sold by the Razorback t-shirt shop during a game -/
def num_tshirts_sold (original_price discount total_revenue : ℕ) : ℕ :=
  total_revenue / (original_price - discount)

/-- Theorem stating that 130 t-shirts were sold given the problem conditions -/
theorem razorback_tshirt_sales : num_tshirts_sold 51 8 5590 = 130 := by
  sorry

end NUMINAMATH_CALUDE_razorback_tshirt_sales_l4133_413352


namespace NUMINAMATH_CALUDE_shirt_discount_problem_l4133_413380

theorem shirt_discount_problem (list_price : ℝ) (final_price : ℝ) (second_discount : ℝ) (first_discount : ℝ) : 
  list_price = 150 →
  final_price = 105 →
  second_discount = 12.5 →
  final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) →
  first_discount = 20 := by
sorry

end NUMINAMATH_CALUDE_shirt_discount_problem_l4133_413380


namespace NUMINAMATH_CALUDE_zach_score_l4133_413394

/-- Given that Ben scored 21 points in a football game and Zach scored 21 more points than Ben,
    prove that Zach scored 42 points. -/
theorem zach_score (ben_score : ℕ) (zach_ben_diff : ℕ) 
  (h1 : ben_score = 21)
  (h2 : zach_ben_diff = 21) :
  ben_score + zach_ben_diff = 42 := by
  sorry

end NUMINAMATH_CALUDE_zach_score_l4133_413394


namespace NUMINAMATH_CALUDE_highest_score_percentage_l4133_413364

/-- The percentage of correct answers on an exam with a given number of questions -/
def examPercentage (correctAnswers : ℕ) (totalQuestions : ℕ) : ℚ :=
  (correctAnswers : ℚ) / (totalQuestions : ℚ) * 100

theorem highest_score_percentage
  (totalQuestions : ℕ)
  (hannahsTarget : ℕ)
  (otherStudentWrong : ℕ)
  (hTotal : totalQuestions = 40)
  (hHannah : hannahsTarget = 39)
  (hOther : otherStudentWrong = 3)
  : examPercentage (totalQuestions - otherStudentWrong - 1) totalQuestions = 95 := by
  sorry

end NUMINAMATH_CALUDE_highest_score_percentage_l4133_413364


namespace NUMINAMATH_CALUDE_era_burger_division_l4133_413395

/-- The number of slices each of the third and fourth friends receive when Era divides her burgers. -/
def slices_per_friend (total_burgers : ℕ) (first_friend_slices second_friend_slices era_slices : ℕ) : ℕ :=
  let total_slices := total_burgers * 2
  let remaining_slices := total_slices - (first_friend_slices + second_friend_slices + era_slices)
  remaining_slices / 2

/-- Theorem stating that under the given conditions, each of the third and fourth friends receives 3 slices. -/
theorem era_burger_division :
  slices_per_friend 5 1 2 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_era_burger_division_l4133_413395


namespace NUMINAMATH_CALUDE_xyz_equals_ten_l4133_413398

theorem xyz_equals_ten (a b c x y z : ℂ) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h_a : a = (b + c) / (x - 3))
  (h_b : b = (a + c) / (y - 3))
  (h_c : c = (a + b) / (z - 3))
  (h_sum_prod : x * y + x * z + y * z = 7)
  (h_sum : x + y + z = 4) :
  x * y * z = 10 := by
sorry

end NUMINAMATH_CALUDE_xyz_equals_ten_l4133_413398


namespace NUMINAMATH_CALUDE_series_sum_equals_four_implies_x_equals_half_l4133_413310

/-- The sum of the infinite series 1 + 2x + 3x^2 + ... -/
noncomputable def S (x : ℝ) : ℝ := ∑' n, (n + 1) * x^n

/-- The theorem stating that if S(x) = 4, then x = 1/2 -/
theorem series_sum_equals_four_implies_x_equals_half :
  ∀ x : ℝ, x < 1 → S x = 4 → x = 1/2 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_four_implies_x_equals_half_l4133_413310


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l4133_413360

/-- For a constant m, x^2 + 2x + m is a perfect square trinomial if and only if m = 1 -/
theorem perfect_square_trinomial (m : ℝ) :
  (∀ x : ℝ, ∃ a : ℝ, x^2 + 2*x + m = (x + a)^2) ↔ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l4133_413360


namespace NUMINAMATH_CALUDE_fraction_ordering_l4133_413376

theorem fraction_ordering : 
  (20 : ℚ) / 16 < (18 : ℚ) / 14 ∧ (18 : ℚ) / 14 < (16 : ℚ) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l4133_413376


namespace NUMINAMATH_CALUDE_probability_of_sum_25_l4133_413389

/-- Represents a die with numbered faces and a blank face -/
structure Die where
  faces : ℕ
  numbers : List ℕ
  blank_faces : ℕ

/-- Calculates the probability of a specific sum when rolling two dice -/
def probability_of_sum (die1 die2 : Die) (target_sum : ℕ) : ℚ :=
  sorry

/-- The first die with 19 faces numbered 1 through 18 and one blank face -/
def first_die : Die :=
  { faces := 20,
    numbers := List.range 18,
    blank_faces := 1 }

/-- The second die with 19 faces numbered 2 through 9 and 11 through 21 and one blank face -/
def second_die : Die :=
  { faces := 20,
    numbers := (List.range 8).map (· + 2) ++ (List.range 11).map (· + 11),
    blank_faces := 1 }

/-- Theorem stating the probability of rolling a sum of 25 with the given dice -/
theorem probability_of_sum_25 :
  probability_of_sum first_die second_die 25 = 3 / 80 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_sum_25_l4133_413389


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4133_413307

theorem geometric_sequence_sum (a₁ a₄ r : ℚ) (h₁ : a₁ = 4096) (h₂ : a₄ = 16) (h₃ : r = 1/4) :
  a₁ * r + a₁ * r^2 = 320 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4133_413307


namespace NUMINAMATH_CALUDE_rachel_age_when_father_is_60_l4133_413368

/-- Rachel's age when the problem is stated -/
def rachel_initial_age : ℕ := 12

/-- Rachel's grandfather's age is 7 times Rachel's age -/
def grandfather_age (rachel_age : ℕ) : ℕ := 7 * rachel_age

/-- Rachel's mother's age is half her grandfather's age -/
def mother_age (grandfather_age : ℕ) : ℕ := grandfather_age / 2

/-- Rachel's father's age is 5 years older than her mother -/
def father_age (mother_age : ℕ) : ℕ := mother_age + 5

/-- The age difference between Rachel and her father -/
def age_difference : ℕ := father_age (mother_age (grandfather_age rachel_initial_age)) - rachel_initial_age

theorem rachel_age_when_father_is_60 :
  rachel_initial_age + age_difference = 25 ∧ father_age (mother_age (grandfather_age (rachel_initial_age + age_difference))) = 60 := by
  sorry


end NUMINAMATH_CALUDE_rachel_age_when_father_is_60_l4133_413368


namespace NUMINAMATH_CALUDE_girl_scouts_expenses_l4133_413356

def total_earnings : ℝ := 30

def pool_entry_cost : ℝ :=
  5 * 3.5 + 3 * 2 + 2 * 1

def transportation_cost : ℝ :=
  6 * 1.5 + 4 * 0.75

def snack_cost : ℝ :=
  3 * 3 + 4 * 2.5 + 3 * 2

def total_expenses : ℝ :=
  pool_entry_cost + transportation_cost + snack_cost

theorem girl_scouts_expenses (h : total_expenses > total_earnings) :
  total_expenses - total_earnings = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_girl_scouts_expenses_l4133_413356


namespace NUMINAMATH_CALUDE_complex_power_result_l4133_413350

theorem complex_power_result : ∃ (i : ℂ), i^2 = -1 ∧ ((1 + i) / i)^2014 = 2^1007 * i := by sorry

end NUMINAMATH_CALUDE_complex_power_result_l4133_413350


namespace NUMINAMATH_CALUDE_rain_duration_l4133_413343

theorem rain_duration (total_hours : ℕ) (no_rain_hours : ℕ) 
  (h1 : total_hours = 8) (h2 : no_rain_hours = 6) : 
  total_hours - no_rain_hours = 2 := by
  sorry

end NUMINAMATH_CALUDE_rain_duration_l4133_413343


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l4133_413353

theorem students_playing_both_sports (total : ℕ) (football : ℕ) (cricket : ℕ) (neither : ℕ) : 
  total = 420 → football = 325 → cricket = 175 → neither = 50 →
  football + cricket - (total - neither) = 130 := by
sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l4133_413353


namespace NUMINAMATH_CALUDE_prime_equation_solution_l4133_413318

theorem prime_equation_solution (p : ℕ) (x y : ℕ) 
  (h_prime : Nat.Prime p) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) 
  (h_eq : x * (y^2 - p) + y * (x^2 - p) = 5 * p) : 
  p = 2 ∨ p = 3 ∨ p = 7 := by
  sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l4133_413318


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4133_413377

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x : ℝ | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4133_413377


namespace NUMINAMATH_CALUDE_rectangle_ratio_theorem_l4133_413366

theorem rectangle_ratio_theorem (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a ≤ b) :
  let d := Real.sqrt (a^2 + b^2)
  let k := a / b
  (a / b = (a + 2*b) / d) → (k^4 - 3*k^2 - 4*k - 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_theorem_l4133_413366


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_equals_four_l4133_413324

theorem x_squared_plus_y_squared_equals_four 
  (h : (x^2 + y^2 + 1) * (x^2 + y^2 - 3) = 5) : 
  x^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_equals_four_l4133_413324


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l4133_413337

/-- The area of an isosceles right triangle with hypotenuse 6√2 is 18 -/
theorem isosceles_right_triangle_area (h : ℝ) (A : ℝ) : 
  h = 6 * Real.sqrt 2 →  -- hypotenuse is 6√2
  A = (h^2) / 4 →        -- area formula for isosceles right triangle
  A = 18 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l4133_413337


namespace NUMINAMATH_CALUDE_initial_maple_trees_l4133_413315

theorem initial_maple_trees (cut_trees : ℝ) (remaining_trees : ℕ) 
  (h1 : cut_trees = 2.0)
  (h2 : remaining_trees = 7) :
  cut_trees + remaining_trees = 9.0 := by
  sorry

end NUMINAMATH_CALUDE_initial_maple_trees_l4133_413315


namespace NUMINAMATH_CALUDE_division_result_l4133_413385

theorem division_result : (0.0204 : ℝ) / 17 = 0.0012 := by sorry

end NUMINAMATH_CALUDE_division_result_l4133_413385


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4133_413374

theorem complex_equation_solution (Z : ℂ) : (3 + Z) * Complex.I = 1 → Z = -3 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4133_413374


namespace NUMINAMATH_CALUDE_mam_mgm_difference_bound_l4133_413334

theorem mam_mgm_difference_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a < b) :
  let mam := (a^(1/3) + b^(1/3)) / 2
  let mgm := (a * b)^(1/6)
  mam - mgm < (b - a) / (2 * b) := by
  sorry

end NUMINAMATH_CALUDE_mam_mgm_difference_bound_l4133_413334


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l4133_413348

/-- Taxi fare function -/
def fare (k : ℝ) (d : ℝ) : ℝ := 20 + k * d

/-- Theorem: If the fare for 60 miles is $140, then the fare for 85 miles is $190 -/
theorem taxi_fare_calculation (k : ℝ) :
  fare k 60 = 140 → fare k 85 = 190 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_calculation_l4133_413348
