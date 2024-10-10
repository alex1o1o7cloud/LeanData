import Mathlib

namespace triangle_area_l3022_302292

/-- The area of a triangle with vertices at (3, -3), (8, 4), and (3, 4) is 17.5 square units. -/
theorem triangle_area : Real := by
  -- Define the vertices of the triangle
  let v1 : (Real × Real) := (3, -3)
  let v2 : (Real × Real) := (8, 4)
  let v3 : (Real × Real) := (3, 4)

  -- Calculate the area of the triangle
  let area : Real := 17.5

  sorry -- The proof is omitted

#check triangle_area

end triangle_area_l3022_302292


namespace smallest_n_perfect_powers_l3022_302230

theorem smallest_n_perfect_powers : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (x : ℕ), 3 * n = x^4) ∧ 
  (∃ (y : ℕ), 2 * n = y^5) ∧ 
  (∀ (m : ℕ), m > 0 → 
    (∃ (x : ℕ), 3 * m = x^4) → 
    (∃ (y : ℕ), 2 * m = y^5) → 
    n ≤ m) ∧
  n = 432 := by
sorry

end smallest_n_perfect_powers_l3022_302230


namespace units_digit_of_47_to_47_l3022_302252

theorem units_digit_of_47_to_47 : (47^47 % 10 = 3) := by
  sorry

end units_digit_of_47_to_47_l3022_302252


namespace soda_bottles_ordered_l3022_302254

/-- The number of bottles of soda ordered by a store owner in April and May -/
theorem soda_bottles_ordered (april_cases may_cases bottles_per_case : ℕ) 
  (h1 : april_cases = 20)
  (h2 : may_cases = 30)
  (h3 : bottles_per_case = 20) :
  (april_cases + may_cases) * bottles_per_case = 1000 := by
  sorry

end soda_bottles_ordered_l3022_302254


namespace subcommittee_count_l3022_302288

theorem subcommittee_count (n m k : ℕ) (hn : n = 12) (hm : m = 5) (hk : k = 5) :
  Nat.choose n k - Nat.choose (n - m) k = 771 :=
sorry

end subcommittee_count_l3022_302288


namespace three_digit_numbers_count_l3022_302286

theorem three_digit_numbers_count : 
  let digits : Finset Nat := {1, 2, 3, 4, 5}
  (digits.card : Nat) ^ 3 = 125 := by
  sorry

end three_digit_numbers_count_l3022_302286


namespace tan_sum_special_case_l3022_302239

theorem tan_sum_special_case :
  let tan55 := Real.tan (55 * π / 180)
  let tan65 := Real.tan (65 * π / 180)
  tan55 + tan65 - Real.sqrt 3 * tan55 * tan65 = -Real.sqrt 3 :=
by
  sorry

end tan_sum_special_case_l3022_302239


namespace reciprocal_of_difference_l3022_302222

-- Define repeating decimals
def repeating_decimal_1 : ℚ := 1/9
def repeating_decimal_6 : ℚ := 2/3

-- State the theorem
theorem reciprocal_of_difference : (repeating_decimal_6 - repeating_decimal_1)⁻¹ = 9/5 := by
  sorry

end reciprocal_of_difference_l3022_302222


namespace rhombus_perimeter_l3022_302287

-- Define the rhombus
structure Rhombus :=
  (side_length : ℝ)
  (diagonal_length : ℝ)

-- Define the conditions
def satisfies_equation (y : ℝ) : Prop :=
  y^2 - 7*y + 10 = 0

def is_valid_rhombus (r : Rhombus) : Prop :=
  r.diagonal_length = 6 ∧ satisfies_equation r.side_length

-- Theorem statement
theorem rhombus_perimeter (r : Rhombus) (h : is_valid_rhombus r) : 
  4 * r.side_length = 20 :=
sorry

end rhombus_perimeter_l3022_302287


namespace irene_income_l3022_302267

/-- Calculates the total income for a given number of hours worked -/
def total_income (regular_income : ℕ) (overtime_rate : ℕ) (hours_worked : ℕ) : ℕ :=
  let regular_hours := 40
  let overtime_hours := max (hours_worked - regular_hours) 0
  regular_income + overtime_rate * overtime_hours

/-- Irene's income calculation theorem -/
theorem irene_income :
  total_income 500 20 50 = 700 := by
  sorry

#eval total_income 500 20 50

end irene_income_l3022_302267


namespace no_prime_roots_for_quadratic_l3022_302295

theorem no_prime_roots_for_quadratic : ¬∃ k : ℤ, ∃ p q : ℕ,
  Prime p ∧ Prime q ∧ p + q = 95 ∧ p * q = k :=
sorry

end no_prime_roots_for_quadratic_l3022_302295


namespace solution_set_implies_a_value_l3022_302253

theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, |x - a| < 1 ↔ 2 < x ∧ x < 4) →
  a = 3 := by
sorry

end solution_set_implies_a_value_l3022_302253


namespace cubic_inequality_l3022_302234

theorem cubic_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hne : a ≠ b) :
  a^3 + b^3 > a^2*b + a*b^2 := by
  sorry

end cubic_inequality_l3022_302234


namespace magic_square_x_free_l3022_302261

/-- Represents a 3x3 magic square with given entries -/
structure MagicSquare where
  x : ℝ
  sum : ℝ
  top_middle : ℝ
  top_right : ℝ
  middle_left : ℝ
  is_magic : sum = x + top_middle + top_right
           ∧ sum = x + middle_left + (sum - x - middle_left)
           ∧ sum = top_right + (sum - top_right - (sum - x - middle_left))

/-- Theorem stating that x can be any real number in the given magic square -/
theorem magic_square_x_free (m : MagicSquare) (h : m.top_middle = 35 ∧ m.top_right = 58 ∧ m.middle_left = 8 ∧ m.sum = 85) :
  ∀ y : ℝ, ∃ m' : MagicSquare, m'.x = y ∧ m'.top_middle = m.top_middle ∧ m'.top_right = m.top_right ∧ m'.middle_left = m.middle_left ∧ m'.sum = m.sum :=
sorry

end magic_square_x_free_l3022_302261


namespace covered_number_value_l3022_302225

theorem covered_number_value : ∃ a : ℝ, 
  (∀ x : ℝ, (x - a) / 2 = x + 3 ↔ x = -7) ∧ a = 1 := by
  sorry

end covered_number_value_l3022_302225


namespace smallest_max_volume_is_500_l3022_302277

/-- Represents a cuboid with integral side lengths -/
structure Cuboid where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- Calculates the volume of a cuboid -/
def Cuboid.volume (c : Cuboid) : ℕ := c.length.val * c.width.val * c.height.val

/-- Represents the result of cutting a cube into three cuboids -/
structure CubeCut where
  cuboid1 : Cuboid
  cuboid2 : Cuboid
  cuboid3 : Cuboid

/-- Checks if a CubeCut is valid for a cube with side length 10 -/
def isValidCubeCut (cut : CubeCut) : Prop :=
  cut.cuboid1.length + cut.cuboid2.length + cut.cuboid3.length = 10 ∧
  cut.cuboid1.width = 10 ∧ cut.cuboid2.width = 10 ∧ cut.cuboid3.width = 10 ∧
  cut.cuboid1.height = 10 ∧ cut.cuboid2.height = 10 ∧ cut.cuboid3.height = 10

/-- The main theorem to prove -/
theorem smallest_max_volume_is_500 :
  ∀ (cut : CubeCut),
    isValidCubeCut cut →
    max cut.cuboid1.volume (max cut.cuboid2.volume cut.cuboid3.volume) ≥ 500 :=
by sorry

end smallest_max_volume_is_500_l3022_302277


namespace range_of_m_when_S_true_range_of_m_when_p_or_q_and_not_q_l3022_302273

-- Define the propositions
def p (m : ℝ) : Prop := ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
  ∀ x y : ℝ, x^2 / (4 - m) + y^2 / m = 1 ↔ (x / a)^2 + (y / b)^2 = 1

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 1 > 0

def S (m : ℝ) : Prop := ∃ x : ℝ, m*x^2 + 2*m*x + 2 - m = 0

-- State the theorems
theorem range_of_m_when_S_true :
  ∀ m : ℝ, S m → m < 0 ∨ m ≥ 1 :=
sorry

theorem range_of_m_when_p_or_q_and_not_q :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(q m) → 1 ≤ m ∧ m < 2 :=
sorry

end range_of_m_when_S_true_range_of_m_when_p_or_q_and_not_q_l3022_302273


namespace polynomial_simplification_l3022_302209

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^2 + 7 * x - 3) - (x^2 + 5 * x - 12) = x^2 + 2 * x + 9 := by
  sorry

end polynomial_simplification_l3022_302209


namespace max_cables_cut_l3022_302224

/-- Represents a computer network -/
structure ComputerNetwork where
  numComputers : ℕ
  numCables : ℕ
  numClusters : ℕ

/-- The initial state of the computer network -/
def initialNetwork : ComputerNetwork :=
  { numComputers := 200
  , numCables := 345
  , numClusters := 1 }

/-- The final state of the computer network after cutting cables -/
def finalNetwork : ComputerNetwork :=
  { numComputers := 200
  , numCables := 345 - 153
  , numClusters := 8 }

/-- Theorem stating the maximum number of cables that can be cut -/
theorem max_cables_cut (initial : ComputerNetwork) (final : ComputerNetwork) :
  initial.numComputers = 200 →
  initial.numCables = 345 →
  initial.numClusters = 1 →
  final.numComputers = initial.numComputers →
  final.numClusters = 8 →
  final.numCables = initial.numCables - 153 →
  ∀ n : ℕ, n > 153 → 
    ¬∃ (network : ComputerNetwork), 
      network.numComputers = initial.numComputers ∧
      network.numClusters = final.numClusters ∧
      network.numCables = initial.numCables - n :=
by sorry


end max_cables_cut_l3022_302224


namespace equation_solution_exists_l3022_302246

theorem equation_solution_exists : ∃ x : ℝ, (0.75 : ℝ) ^ x + 2 = 8 := by sorry

end equation_solution_exists_l3022_302246


namespace inequality_implies_values_l3022_302202

theorem inequality_implies_values (a b : ℤ) 
  (h : ∀ x : ℝ, x ≤ 0 → (a * x + 2) * (x^2 + 2 * b) ≤ 0) : 
  a = 1 ∧ b = -2 := by
  sorry

end inequality_implies_values_l3022_302202


namespace identify_scientists_l3022_302245

/-- Represents the type of scientist: chemist or alchemist -/
inductive ScientistType
| Chemist
| Alchemist

/-- Represents a scientist at the conference -/
structure Scientist where
  id : Nat
  type : ScientistType

/-- Represents the conference of scientists -/
structure Conference where
  scientists : List Scientist
  num_chemists : Nat
  num_alchemists : Nat
  chemists_outnumber_alchemists : num_chemists > num_alchemists

/-- Represents a question asked by the mathematician -/
def Question := Scientist → Scientist → ScientistType

/-- The main theorem to be proved -/
theorem identify_scientists (conf : Conference) :
  ∃ (questions : List Question), questions.length ≤ 2 * conf.scientists.length - 2 ∧
  (∀ s : Scientist, s ∈ conf.scientists → 
    ∃ (determined_type : ScientistType), determined_type = s.type) :=
sorry

end identify_scientists_l3022_302245


namespace rectangle_area_ratio_l3022_302278

theorem rectangle_area_ratio : 
  let length_A : ℝ := 48
  let breadth_A : ℝ := 30
  let length_B : ℝ := 60
  let breadth_B : ℝ := 35
  let area_A := length_A * breadth_A
  let area_B := length_B * breadth_B
  (area_A / area_B) = 24 / 35 := by
sorry

end rectangle_area_ratio_l3022_302278


namespace marble_selection_ways_l3022_302218

def blue_marbles : ℕ := 3
def red_marbles : ℕ := 4
def green_marbles : ℕ := 3
def total_marbles : ℕ := blue_marbles + red_marbles + green_marbles
def marbles_to_choose : ℕ := 5

theorem marble_selection_ways : 
  (Nat.choose blue_marbles 1) * (Nat.choose red_marbles 1) * (Nat.choose green_marbles 1) *
  (Nat.choose (total_marbles - 3) 2) = 756 := by
  sorry

end marble_selection_ways_l3022_302218


namespace arithmetic_sequence_problem_l3022_302231

theorem arithmetic_sequence_problem (a : ℕ → ℝ) :
  (∀ n m : ℕ, n < m → a n < a m) →  -- increasing sequence
  (a 4)^2 - 10 * (a 4) + 24 = 0 →   -- a_4 is a root
  (a 6)^2 - 10 * (a 6) + 24 = 0 →   -- a_6 is a root
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
  a 20 = 20 := by
sorry

end arithmetic_sequence_problem_l3022_302231


namespace fungi_at_128pm_l3022_302203

/-- The number of fungi at a given time, given an initial population and doubling time -/
def fungiBehavior (initialPopulation : ℕ) (doublingTime : ℕ) (elapsedTime : ℕ) : ℕ :=
  initialPopulation * 2 ^ (elapsedTime / doublingTime)

/-- Theorem stating the number of fungi at 1:28 p.m. given the initial conditions -/
theorem fungi_at_128pm (initialPopulation : ℕ) (doublingTime : ℕ) (elapsedTime : ℕ) :
  initialPopulation = 30 → doublingTime = 4 → elapsedTime = 28 →
  fungiBehavior initialPopulation doublingTime elapsedTime = 3840 := by
  sorry

#check fungi_at_128pm

end fungi_at_128pm_l3022_302203


namespace max_cross_section_area_l3022_302265

/-- A right rectangular prism with a square base and varying height -/
structure Prism where
  base_length : ℝ
  height_a : ℝ
  height_b : ℝ
  height_c : ℝ
  height_d : ℝ

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The cross-section area formed by the intersection of a prism and a plane -/
def cross_section_area (p : Prism) (pl : Plane) : ℝ := sorry

/-- The theorem stating that the maximal area of the cross-section is 110 -/
theorem max_cross_section_area (p : Prism) (pl : Plane) :
  p.base_length = 8 ∧
  p.height_a = 3 ∧ p.height_b = 2 ∧ p.height_c = 4 ∧ p.height_d = 1 ∧
  pl.a = 3 ∧ pl.b = -5 ∧ pl.c = 3 ∧ pl.d = 24 →
  cross_section_area p pl = 110 := by
  sorry

end max_cross_section_area_l3022_302265


namespace min_sum_floor_l3022_302264

theorem min_sum_floor (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  (⌊(2*x + y) / z⌋ : ℤ) + ⌊(y + 2*z) / x⌋ + ⌊(2*z + x) / y⌋ = 9 ∧
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    (⌊(2*a + b) / c⌋ : ℤ) + ⌊(b + 2*c) / a⌋ + ⌊(2*c + a) / b⌋ ≥ 9 :=
by sorry

end min_sum_floor_l3022_302264


namespace smallest_integer_larger_than_sqrt3_minus_sqrt2_to_6th_l3022_302208

theorem smallest_integer_larger_than_sqrt3_minus_sqrt2_to_6th :
  ∃ n : ℤ, (n = 133 ∧ (∀ m : ℤ, m > (Real.sqrt 3 - Real.sqrt 2)^6 → m ≥ n)) :=
by sorry

end smallest_integer_larger_than_sqrt3_minus_sqrt2_to_6th_l3022_302208


namespace trig_simplification_l3022_302229

theorem trig_simplification (α : Real) (h : α ≠ 0) (h' : α ≠ π / 2) :
  (1 / Real.sin α + 1 / Real.tan α) * (1 - Real.cos α) = Real.sin α := by
  sorry

end trig_simplification_l3022_302229


namespace two_triangles_from_tetrahedron_l3022_302217

-- Define a tetrahedron
structure Tetrahedron :=
  (A B C D : Point)
  (AB AC AD BC BD CD : ℝ)
  (longest_edge : AB ≥ max AC (max AD (max BC (max BD CD))))
  (AC_geq_BD : AC ≥ BD)

-- Define a triangle
structure Triangle :=
  (side1 side2 side3 : ℝ)

-- Theorem statement
theorem two_triangles_from_tetrahedron (t : Tetrahedron) : 
  ∃ (triangle1 triangle2 : Triangle), 
    (triangle1.side1 = t.BC ∧ triangle1.side2 = t.CD ∧ triangle1.side3 = t.BD) ∧
    (triangle2.side1 = t.AC ∧ triangle2.side2 = t.CD ∧ triangle2.side3 = t.AD) ∧
    (triangle1 ≠ triangle2) :=
sorry

end two_triangles_from_tetrahedron_l3022_302217


namespace jessica_quarters_l3022_302200

theorem jessica_quarters (initial borrowed current : ℕ) : 
  borrowed = 3 → current = 5 → initial = current + borrowed :=
by sorry

end jessica_quarters_l3022_302200


namespace monic_polynomial_divisibility_l3022_302223

open Polynomial

theorem monic_polynomial_divisibility (n k : ℕ) (h_pos_n : n > 0) (h_pos_k : k > 0) :
  ∀ (f : Polynomial ℤ),
    Monic f →
    (Polynomial.degree f = n) →
    (∀ (a : ℤ), f.eval a ≠ 0 → (f.eval a ∣ f.eval (2 * a ^ k))) →
    f = X ^ n :=
by sorry

end monic_polynomial_divisibility_l3022_302223


namespace triangle_side_value_l3022_302256

theorem triangle_side_value (m : ℝ) : m > 0 → 
  (3 + 4 > m ∧ 3 + m > 4 ∧ 4 + m > 3) →
  (m = 1 ∨ m = 5 ∨ m = 7 ∨ m = 9) →
  m = 5 := by sorry

end triangle_side_value_l3022_302256


namespace root_sum_squares_l3022_302216

theorem root_sum_squares (a b c d : ℂ) : 
  (a^4 - 24*a^3 + 50*a^2 - 35*a + 7 = 0) →
  (b^4 - 24*b^3 + 50*b^2 - 35*b + 7 = 0) →
  (c^4 - 24*c^3 + 50*c^2 - 35*c + 7 = 0) →
  (d^4 - 24*d^3 + 50*d^2 - 35*d + 7 = 0) →
  (a+b+c)^2 + (b+c+d)^2 + (c+d+a)^2 + (d+a+b)^2 = 2104 := by
  sorry

end root_sum_squares_l3022_302216


namespace condition_equivalence_l3022_302262

theorem condition_equivalence (α β : ℝ) :
  (α > β) ↔ (α + Real.sin α * Real.cos β > β + Real.sin β * Real.cos α) := by
  sorry

end condition_equivalence_l3022_302262


namespace greatest_multiple_of_four_l3022_302237

theorem greatest_multiple_of_four (x : ℕ) : 
  x % 4 = 0 → 
  x > 0 → 
  x^3 < 5000 → 
  x ≤ 16 ∧ 
  ∃ y : ℕ, y % 4 = 0 ∧ y > 0 ∧ y^3 < 5000 ∧ y = 16 :=
by sorry

end greatest_multiple_of_four_l3022_302237


namespace full_house_count_l3022_302294

theorem full_house_count :
  let n_values : ℕ := 13
  let cards_per_value : ℕ := 4
  let full_house_count := n_values * (n_values - 1) * (cards_per_value.choose 3) * (cards_per_value.choose 2)
  full_house_count = 3744 :=
by sorry

end full_house_count_l3022_302294


namespace other_root_of_quadratic_l3022_302247

theorem other_root_of_quadratic (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x = -2) → 
  (-1 : ℝ) ∈ {x : ℝ | 3 * x^2 + m * x = -2} → 
  (-2/3 : ℝ) ∈ {x : ℝ | 3 * x^2 + m * x = -2} :=
by sorry

end other_root_of_quadratic_l3022_302247


namespace sqrt_9801_minus_39_cube_l3022_302219

theorem sqrt_9801_minus_39_cube (a b : ℕ+) :
  (Real.sqrt 9801 - 39 : ℝ) = (Real.sqrt a.val - b.val : ℝ)^3 →
  a.val + b.val = 13 := by
sorry

end sqrt_9801_minus_39_cube_l3022_302219


namespace roots_of_quadratic_equation_l3022_302220

def quadratic_equation (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem roots_of_quadratic_equation :
  let a : ℝ := 1
  let b : ℝ := -7
  let c : ℝ := 12
  (quadratic_equation a b c 3 = 0) ∧ (quadratic_equation a b c 4 = 0) :=
by sorry

end roots_of_quadratic_equation_l3022_302220


namespace simplify_expression_l3022_302241

theorem simplify_expression (x : ℝ) : 5 * x + 7 * x - 3 * x = 9 * x := by
  sorry

end simplify_expression_l3022_302241


namespace common_divisors_9180_10080_l3022_302271

theorem common_divisors_9180_10080 : 
  let a := 9180
  let b := 10080
  (a % 7 = 0) → 
  (b % 7 = 0) → 
  (Finset.filter (fun d => d ∣ a ∧ d ∣ b) (Finset.range (min a b + 1))).card = 36 := by
sorry

end common_divisors_9180_10080_l3022_302271


namespace angle_A_is_obtuse_l3022_302289

/-- Triangle ABC with vertices A(2,1), B(-1,4), and C(5,3) -/
structure Triangle where
  A : ℝ × ℝ := (2, 1)
  B : ℝ × ℝ := (-1, 4)
  C : ℝ × ℝ := (5, 3)

/-- Calculate the squared distance between two points -/
def squaredDistance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Check if an angle is obtuse using the cosine law -/
def isObtuse (a b c : ℝ) : Prop :=
  a^2 > b^2 + c^2

theorem angle_A_is_obtuse (t : Triangle) : 
  isObtuse (squaredDistance t.B t.C) (squaredDistance t.A t.B) (squaredDistance t.A t.C) :=
sorry

end angle_A_is_obtuse_l3022_302289


namespace only_statement4_correct_l3022_302251

-- Define the structure of input/output statements
inductive Statement
| Input (vars : List String)
| InputAssign (var : String) (value : Nat)
| Print (expr : String)
| PrintMultiple (values : List Nat)

-- Define the rules for correct statements
def isCorrectInput (s : Statement) : Prop :=
  match s with
  | Statement.Input vars => vars.length > 0
  | Statement.InputAssign _ _ => false
  | _ => false

def isCorrectOutput (s : Statement) : Prop :=
  match s with
  | Statement.Print _ => false
  | Statement.PrintMultiple values => values.length > 0
  | _ => false

def isCorrect (s : Statement) : Prop :=
  isCorrectInput s ∨ isCorrectOutput s

-- Define the given statements
def statement1 : Statement := Statement.Input ["a", "b", "c"]
def statement2 : Statement := Statement.Print "a=1"
def statement3 : Statement := Statement.InputAssign "x" 2
def statement4 : Statement := Statement.PrintMultiple [20, 4]

-- Theorem to prove
theorem only_statement4_correct :
  ¬ isCorrect statement1 ∧
  ¬ isCorrect statement2 ∧
  ¬ isCorrect statement3 ∧
  isCorrect statement4 :=
sorry

end only_statement4_correct_l3022_302251


namespace cow_husk_consumption_l3022_302238

/-- Given that 20 cows eat 20 bags of husk in 20 days, prove that one cow will eat one bag of husk in 20 days. -/
theorem cow_husk_consumption (num_cows : ℕ) (num_bags : ℕ) (num_days : ℕ) 
  (h1 : num_cows = 20) 
  (h2 : num_bags = 20) 
  (h3 : num_days = 20) : 
  (num_days : ℚ) = (num_cows : ℚ) * (num_bags : ℚ) / ((num_cows : ℚ) * (num_bags : ℚ)) := by
  sorry

end cow_husk_consumption_l3022_302238


namespace least_four_digit_solution_l3022_302240

theorem least_four_digit_solution (x : ℕ) : x = 1163 ↔ 
  (x ≥ 1000 ∧ x < 10000) ∧
  (∀ y : ℕ, y ≥ 1000 ∧ y < 10000 →
    (5 * y ≡ 15 [ZMOD 20] ∧
     3 * y + 10 ≡ 19 [ZMOD 14] ∧
     -3 * y + 4 ≡ 2 * y [ZMOD 35] ∧
     y + 1 ≡ 0 [ZMOD 11]) →
    x ≤ y) ∧
  (5 * x ≡ 15 [ZMOD 20]) ∧
  (3 * x + 10 ≡ 19 [ZMOD 14]) ∧
  (-3 * x + 4 ≡ 2 * x [ZMOD 35]) ∧
  (x + 1 ≡ 0 [ZMOD 11]) := by
  sorry

end least_four_digit_solution_l3022_302240


namespace small_rectangle_perimeter_l3022_302207

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Represents the problem setup -/
structure ProblemSetup where
  large_rectangle : Rectangle
  num_vertical_cuts : ℕ
  num_horizontal_cuts : ℕ
  total_cut_length : ℝ

/-- Theorem stating the solution to the problem -/
theorem small_rectangle_perimeter
  (setup : ProblemSetup)
  (h1 : setup.large_rectangle.perimeter = 100)
  (h2 : setup.num_vertical_cuts = 6)
  (h3 : setup.num_horizontal_cuts = 9)
  (h4 : setup.total_cut_length = 405)
  (h5 : (setup.num_vertical_cuts + 1) * (setup.num_horizontal_cuts + 1) = 70) :
  let small_rectangle := Rectangle.mk
    (setup.large_rectangle.width / (setup.num_vertical_cuts + 1))
    (setup.large_rectangle.height / (setup.num_horizontal_cuts + 1))
  small_rectangle.perimeter = 13 := by
  sorry

end small_rectangle_perimeter_l3022_302207


namespace sqrt_equation_sum_l3022_302214

theorem sqrt_equation_sum (a t : ℝ) (ha : a > 0) (ht : t > 0) :
  Real.sqrt (6 + a / t) = 6 * Real.sqrt (a / t) → t + a = 41 := by
  sorry

end sqrt_equation_sum_l3022_302214


namespace clay_pot_flower_cost_difference_clay_pot_flower_cost_difference_proof_l3022_302258

/-- The cost difference between a clay pot and flowers -/
theorem clay_pot_flower_cost_difference : ℝ → ℝ → ℝ → Prop :=
  fun flower_cost clay_pot_cost soil_cost =>
    flower_cost = 9 ∧
    clay_pot_cost > flower_cost ∧
    soil_cost = flower_cost - 2 ∧
    flower_cost + clay_pot_cost + soil_cost = 45 →
    clay_pot_cost - flower_cost = 20

/-- Proof of the clay_pot_flower_cost_difference theorem -/
theorem clay_pot_flower_cost_difference_proof :
  ∃ (flower_cost clay_pot_cost soil_cost : ℝ),
    clay_pot_flower_cost_difference flower_cost clay_pot_cost soil_cost :=
by
  sorry

end clay_pot_flower_cost_difference_clay_pot_flower_cost_difference_proof_l3022_302258


namespace quadratic_function_nonnegative_constraint_l3022_302206

theorem quadratic_function_nonnegative_constraint (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → x^2 + a*x + 3 - a ≥ 0) → 
  a ∈ Set.Icc (-7) 2 :=
by sorry

end quadratic_function_nonnegative_constraint_l3022_302206


namespace cylinder_volume_change_l3022_302213

/-- Given a cylinder with original volume of 15 cubic feet, prove that tripling its radius and doubling its height results in a new volume of 270 cubic feet. -/
theorem cylinder_volume_change (r h : ℝ) : 
  r > 0 → h > 0 → π * r^2 * h = 15 → π * (3*r)^2 * (2*h) = 270 := by
  sorry

end cylinder_volume_change_l3022_302213


namespace toms_common_cards_l3022_302291

theorem toms_common_cards (rare_count : ℕ) (uncommon_count : ℕ) (rare_cost : ℚ) (uncommon_cost : ℚ) (common_cost : ℚ) (total_cost : ℚ) :
  rare_count = 19 →
  uncommon_count = 11 →
  rare_cost = 1 →
  uncommon_cost = 1/2 →
  common_cost = 1/4 →
  total_cost = 32 →
  (total_cost - (rare_count * rare_cost + uncommon_count * uncommon_cost)) / common_cost = 30 := by
  sorry

#eval (32 : ℚ) - (19 * 1 + 11 * (1/2 : ℚ)) / (1/4 : ℚ)

end toms_common_cards_l3022_302291


namespace decimal_to_fraction_l3022_302281

theorem decimal_to_fraction : (2.35 : ℚ) = 47 / 20 := by sorry

end decimal_to_fraction_l3022_302281


namespace expression_simplification_l3022_302227

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (x^2 + 1) / (x^2 - 1) - (x - 2) / (x - 1) / ((x - 2) / x) = -1/4 := by
  sorry

end expression_simplification_l3022_302227


namespace cubic_polynomial_satisfies_conditions_l3022_302299

theorem cubic_polynomial_satisfies_conditions :
  let q : ℝ → ℝ := λ x => -(1/3) * x^3 - x^2 - (2/3) * x - 3
  (q 1 = -5) ∧ (q 2 = -8) ∧ (q 3 = -17) ∧ (q 4 = -34) := by
  sorry

end cubic_polynomial_satisfies_conditions_l3022_302299


namespace prob_neither_correct_l3022_302284

/-- Given probabilities for answering questions correctly, calculate the probability of answering neither correctly -/
theorem prob_neither_correct (P_A P_B P_AB : ℝ) 
  (h1 : P_A = 0.65)
  (h2 : P_B = 0.55)
  (h3 : P_AB = 0.40)
  (h4 : 0 ≤ P_A ∧ P_A ≤ 1)
  (h5 : 0 ≤ P_B ∧ P_B ≤ 1)
  (h6 : 0 ≤ P_AB ∧ P_AB ≤ 1) :
  1 - (P_A + P_B - P_AB) = 0.20 := by
  sorry

end prob_neither_correct_l3022_302284


namespace exists_counterexample_1_fraction_inequality_implies_exists_counterexample_3_fraction_inequality_implies_product_l3022_302242

-- Statement 1
theorem exists_counterexample_1 : ∃ (a b c d : ℝ), a > b ∧ c = d ∧ a * c ≤ b * d := by sorry

-- Statement 2
theorem fraction_inequality_implies (a b c : ℝ) (h : c ≠ 0) : a / c^2 < b / c^2 → a < b := by sorry

-- Statement 3
theorem exists_counterexample_3 : ∃ (a b c d : ℝ), a > b ∧ c > d ∧ a - c ≤ b - d := by sorry

-- Statement 4
theorem fraction_inequality_implies_product (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : c / a > d / b → b * c > a * d := by sorry

end exists_counterexample_1_fraction_inequality_implies_exists_counterexample_3_fraction_inequality_implies_product_l3022_302242


namespace sum_is_composite_l3022_302269

theorem sum_is_composite (a b c d : ℕ+) 
  (h : a^2 - a*b + b^2 = c^2 - c*d + d^2) : 
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ a + b + c + d = x * y :=
sorry

end sum_is_composite_l3022_302269


namespace symmetric_point_x_axis_l3022_302263

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the original point P
def P : Point := (3, -5)

-- Define the symmetry operation with respect to x-axis
def symmetry_x_axis (p : Point) : Point :=
  (p.1, -p.2)

-- Theorem statement
theorem symmetric_point_x_axis :
  symmetry_x_axis P = (3, 5) := by sorry

end symmetric_point_x_axis_l3022_302263


namespace shopping_trip_tax_percentage_l3022_302236

/-- Calculate the total tax percentage given spending percentages and tax rates -/
theorem shopping_trip_tax_percentage
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (other_percent : ℝ)
  (clothing_tax_rate : ℝ)
  (food_tax_rate : ℝ)
  (other_tax_rate : ℝ)
  (h1 : clothing_percent = 0.45)
  (h2 : food_percent = 0.45)
  (h3 : other_percent = 0.1)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : clothing_tax_rate = 0.05)
  (h6 : food_tax_rate = 0)
  (h7 : other_tax_rate = 0.1) :
  clothing_percent * clothing_tax_rate +
  food_percent * food_tax_rate +
  other_percent * other_tax_rate = 0.0325 := by
  sorry

#check shopping_trip_tax_percentage

end shopping_trip_tax_percentage_l3022_302236


namespace novelist_writing_speed_l3022_302290

/-- Calculates the effective writing speed given total words, total hours, and break hours -/
def effectiveWritingSpeed (totalWords : ℕ) (totalHours : ℕ) (breakHours : ℕ) : ℕ :=
  totalWords / (totalHours - breakHours)

/-- Proves that the effective writing speed for the given conditions is 750 words per hour -/
theorem novelist_writing_speed :
  effectiveWritingSpeed 60000 100 20 = 750 := by
  sorry

end novelist_writing_speed_l3022_302290


namespace negation_equivalence_l3022_302250

theorem negation_equivalence :
  (¬ ∃ x : ℝ, Real.exp x - x - 1 < 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 ≥ 0) :=
by sorry

end negation_equivalence_l3022_302250


namespace unique_solution_triple_l3022_302293

theorem unique_solution_triple (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (2 * x^3 = 2 * y * (x^2 + 1) - (z^2 + 1)) ∧
  (2 * y^4 = 3 * z * (y^2 + 1) - 2 * (x^2 + 1)) ∧
  (2 * z^5 = 4 * x * (z^2 + 1) - 3 * (y^2 + 1)) →
  x = 1 ∧ y = 1 ∧ z = 1 :=
by sorry

end unique_solution_triple_l3022_302293


namespace sqrt_three_expression_equality_l3022_302243

theorem sqrt_three_expression_equality : 
  (Real.sqrt 3 + 1)^2 - Real.sqrt 12 + 2 * Real.sqrt (1/3) = 4 + (2 * Real.sqrt 3) / 3 := by
  sorry

end sqrt_three_expression_equality_l3022_302243


namespace probability_y_div_x_geq_4_probability_equals_one_eighth_l3022_302248

/-- The probability that y/x ≥ 4 when x and y are randomly selected from [0,2] -/
theorem probability_y_div_x_geq_4 : Real :=
  let total_area := 4
  let favorable_area := 1/2
  favorable_area / total_area

/-- The probability is equal to 1/8 -/
theorem probability_equals_one_eighth : probability_y_div_x_geq_4 = 1/8 := by
  sorry

end probability_y_div_x_geq_4_probability_equals_one_eighth_l3022_302248


namespace equation_unique_solution_l3022_302285

theorem equation_unique_solution :
  ∃! x : ℝ, (8 : ℝ)^(2*x+1) * (2 : ℝ)^(3*x+5) = (4 : ℝ)^(3*x+2) ∧ x = -4/3 := by
  sorry

end equation_unique_solution_l3022_302285


namespace equation_solution_l3022_302244

theorem equation_solution (x : ℚ) (h : 2 * x + 1 = 8) : 4 * x + 1 = 15 := by
  sorry

end equation_solution_l3022_302244


namespace combined_distance_is_122_l3022_302259

-- Define the fuel-to-distance ratios for both cars
def car_A_ratio : Rat := 4 / 7
def car_B_ratio : Rat := 3 / 5

-- Define the amount of fuel used by each car
def car_A_fuel : ℕ := 44
def car_B_fuel : ℕ := 27

-- Function to calculate distance given fuel and ratio
def calculate_distance (fuel : ℕ) (ratio : Rat) : ℚ :=
  (fuel : ℚ) * (ratio.den : ℚ) / (ratio.num : ℚ)

-- Theorem stating the combined distance is 122 miles
theorem combined_distance_is_122 :
  (calculate_distance car_A_fuel car_A_ratio + calculate_distance car_B_fuel car_B_ratio) = 122 := by
  sorry

end combined_distance_is_122_l3022_302259


namespace vet_donation_is_78_l3022_302276

/-- Represents the vet fees and adoption numbers for different animal types -/
structure AnimalAdoption where
  dog_fee : ℕ
  cat_fee : ℕ
  rabbit_fee : ℕ
  parrot_fee : ℕ
  dog_adoptions : ℕ
  cat_adoptions : ℕ
  rabbit_adoptions : ℕ
  parrot_adoptions : ℕ

/-- Calculates the total vet fees collected -/
def total_fees (a : AnimalAdoption) : ℕ :=
  a.dog_fee * a.dog_adoptions +
  a.cat_fee * a.cat_adoptions +
  a.rabbit_fee * a.rabbit_adoptions +
  a.parrot_fee * a.parrot_adoptions

/-- Calculates the amount donated by the vet -/
def vet_donation (a : AnimalAdoption) : ℕ :=
  (total_fees a + 1) / 3

/-- Theorem stating that the vet's donation is $78 given the specified conditions -/
theorem vet_donation_is_78 (a : AnimalAdoption) 
  (h1 : a.dog_fee = 15)
  (h2 : a.cat_fee = 13)
  (h3 : a.rabbit_fee = 10)
  (h4 : a.parrot_fee = 12)
  (h5 : a.dog_adoptions = 8)
  (h6 : a.cat_adoptions = 3)
  (h7 : a.rabbit_adoptions = 5)
  (h8 : a.parrot_adoptions = 2) :
  vet_donation a = 78 := by
  sorry


end vet_donation_is_78_l3022_302276


namespace quadratic_inequality_bc_value_l3022_302296

theorem quadratic_inequality_bc_value 
  (b c : ℝ) 
  (h : ∀ x : ℝ, x^2 + b*x + c < 0 ↔ 2 < x ∧ x < 4) : 
  b * c = -48 := by
  sorry

end quadratic_inequality_bc_value_l3022_302296


namespace exists_special_function_l3022_302212

theorem exists_special_function :
  ∃ f : ℕ → ℕ,
    (∀ m n : ℕ, m < n → f m < f n) ∧
    f 1 = 2 ∧
    ∀ n : ℕ, f (f n) = f n + n :=
by sorry

end exists_special_function_l3022_302212


namespace jeff_initial_pencils_l3022_302266

theorem jeff_initial_pencils (J : ℝ) : 
  J > 0 →
  (0.7 * J + 0.25 * (2 * J) = 360) →
  J = 300 := by
sorry

end jeff_initial_pencils_l3022_302266


namespace trains_meeting_problem_l3022_302233

/-- Theorem: Two trains meeting problem
    Given two trains starting 450 miles apart and traveling towards each other
    at 50 miles per hour each, the distance traveled by one train when they meet
    is 225 miles. -/
theorem trains_meeting_problem (distance_between_stations : ℝ) 
                                (speed_train_a : ℝ) 
                                (speed_train_b : ℝ) : ℝ :=
  by
  have h1 : distance_between_stations = 450 := by sorry
  have h2 : speed_train_a = 50 := by sorry
  have h3 : speed_train_b = 50 := by sorry
  
  -- Calculate the combined speed of the trains
  let combined_speed := speed_train_a + speed_train_b
  
  -- Calculate the time until the trains meet
  let time_to_meet := distance_between_stations / combined_speed
  
  -- Calculate the distance traveled by Train A
  let distance_traveled_by_a := speed_train_a * time_to_meet
  
  -- Prove that the distance traveled by Train A is 225 miles
  have h4 : distance_traveled_by_a = 225 := by sorry
  
  exact distance_traveled_by_a


end trains_meeting_problem_l3022_302233


namespace roundness_of_24300000_l3022_302201

/-- Roundness of a positive integer is the sum of the exponents of its prime factors. -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The number we're analyzing -/
def number : ℕ+ := 24300000

theorem roundness_of_24300000 : roundness number = 15 := by sorry

end roundness_of_24300000_l3022_302201


namespace ratio_and_equation_solution_l3022_302298

theorem ratio_and_equation_solution :
  ∀ (x y z b : ℤ),
  (∃ (k : ℤ), x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) →
  y = 15 * b - 5 →
  (b = 3 → (∃ (k : ℤ), x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) ∧ y = 15 * b - 5) :=
by sorry

end ratio_and_equation_solution_l3022_302298


namespace sum_of_ages_l3022_302205

theorem sum_of_ages (age1 age2 age3 : ℕ) : 
  age1 = 9 → age2 = 9 → age3 = 11 → age1 + age2 + age3 = 29 :=
by
  sorry

end sum_of_ages_l3022_302205


namespace least_value_of_x_l3022_302272

theorem least_value_of_x (x p : ℕ) : 
  x > 0 → 
  Nat.Prime p → 
  ∃ q, Nat.Prime q ∧ q % 2 = 1 ∧ x / (9 * p) = q →
  x ≥ 81 :=
sorry

end least_value_of_x_l3022_302272


namespace loss_equates_to_five_balls_l3022_302204

/-- Given the sale of 20 balls at Rs. 720 with a loss equal to the cost price of some balls,
    and the cost price of a ball being Rs. 48, prove that the loss equates to 5 balls. -/
theorem loss_equates_to_five_balls 
  (total_balls : ℕ) 
  (selling_price : ℕ) 
  (cost_price_per_ball : ℕ) 
  (h1 : total_balls = 20)
  (h2 : selling_price = 720)
  (h3 : cost_price_per_ball = 48) :
  (total_balls * cost_price_per_ball - selling_price) / cost_price_per_ball = 5 :=
by sorry

end loss_equates_to_five_balls_l3022_302204


namespace sum_of_complex_numbers_l3022_302235

theorem sum_of_complex_numbers :
  let z₁ : ℂ := 3 + 5*I
  let z₂ : ℂ := 4 - 7*I
  let z₃ : ℂ := -2 + 3*I
  z₁ + z₂ + z₃ = 5 + I := by sorry

end sum_of_complex_numbers_l3022_302235


namespace lcm_of_12_16_15_l3022_302255

theorem lcm_of_12_16_15 : Nat.lcm (Nat.lcm 12 16) 15 = 240 := by
  sorry

end lcm_of_12_16_15_l3022_302255


namespace sum_inequality_l3022_302210

theorem sum_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  let S := a / (b + c + d) + b / (a + c + d) + c / (a + b + d) + d / (a + b + c)
  0 < S ∧ S < 1 := by sorry

end sum_inequality_l3022_302210


namespace subtraction_of_like_terms_l3022_302211

theorem subtraction_of_like_terms (a b : ℝ) : 5 * a^3 * b^2 - 3 * a^3 * b^2 = 2 * a^3 * b^2 := by
  sorry

end subtraction_of_like_terms_l3022_302211


namespace volume_of_prism_with_inscribed_sphere_l3022_302283

/-- A regular triangular prism with an inscribed sphere -/
structure RegularTriangularPrism where
  -- The radius of the inscribed sphere
  sphere_radius : ℝ
  -- Assertion that the sphere is inscribed in the prism
  sphere_inscribed : sphere_radius > 0

/-- The volume of a regular triangular prism with an inscribed sphere -/
def prism_volume (p : RegularTriangularPrism) : ℝ :=
  -- Definition of volume calculation
  sorry

/-- Theorem: The volume of a regular triangular prism with an inscribed sphere of radius 2 is 48√3 -/
theorem volume_of_prism_with_inscribed_sphere :
  ∀ (p : RegularTriangularPrism), p.sphere_radius = 2 → prism_volume p = 48 * Real.sqrt 3 := by
  sorry

end volume_of_prism_with_inscribed_sphere_l3022_302283


namespace sebastians_orchestra_size_l3022_302268

/-- Represents the composition of an orchestra --/
structure Orchestra where
  percussion : Nat
  trombone : Nat
  trumpet : Nat
  french_horn : Nat
  violin : Nat
  cello : Nat
  contrabass : Nat
  clarinet : Nat
  flute : Nat
  maestro : Nat

/-- The total number of people in the orchestra --/
def Orchestra.total (o : Orchestra) : Nat :=
  o.percussion + o.trombone + o.trumpet + o.french_horn +
  o.violin + o.cello + o.contrabass +
  o.clarinet + o.flute + o.maestro

/-- The specific orchestra composition from the problem --/
def sebastians_orchestra : Orchestra :=
  { percussion := 1
  , trombone := 4
  , trumpet := 2
  , french_horn := 1
  , violin := 3
  , cello := 1
  , contrabass := 1
  , clarinet := 3
  , flute := 4
  , maestro := 1
  }

/-- Theorem stating that the total number of people in Sebastian's orchestra is 21 --/
theorem sebastians_orchestra_size :
  sebastians_orchestra.total = 21 := by
  sorry

end sebastians_orchestra_size_l3022_302268


namespace decimal_to_fraction_sum_l3022_302275

theorem decimal_to_fraction_sum (p q : ℕ+) : 
  (p : ℚ) / q = 504/1000 → 
  (∀ (a b : ℕ+), (a : ℚ) / b = p / q → a ≤ p ∧ b ≤ q) → 
  (p : ℕ) + q = 188 := by
sorry

end decimal_to_fraction_sum_l3022_302275


namespace balloon_count_l3022_302226

theorem balloon_count (green blue yellow red : ℚ) (total : ℕ) : 
  green = 2/9 →
  blue = 1/3 →
  yellow = 1/4 →
  red = 7/36 →
  green + blue + yellow + red = 1 →
  (yellow * total / 2 : ℚ) = 50 →
  total = 400 := by
sorry

end balloon_count_l3022_302226


namespace sean_charles_whistle_difference_l3022_302260

/-- 
Given that Sean has 45 whistles and Charles has 13 whistles, 
prove that Sean has 32 more whistles than Charles.
-/
theorem sean_charles_whistle_difference :
  ∀ (sean_whistles charles_whistles : ℕ),
    sean_whistles = 45 →
    charles_whistles = 13 →
    sean_whistles - charles_whistles = 32 :=
by
  sorry

end sean_charles_whistle_difference_l3022_302260


namespace unique_solution_system_l3022_302297

theorem unique_solution_system (x y : ℝ) : 
  (x + 2*y = 4 ∧ 2*x - y = 3) ↔ (x = 2 ∧ y = 1) := by sorry

end unique_solution_system_l3022_302297


namespace fraction_power_product_l3022_302249

theorem fraction_power_product : (2/3 : ℚ)^2023 * (-3/2 : ℚ)^2023 = -1 := by
  sorry

end fraction_power_product_l3022_302249


namespace pythagorean_triple_parity_l3022_302228

theorem pythagorean_triple_parity (m n : ℤ) 
  (h_succ : m = n + 1 ∨ n = m + 1)
  (a b c : ℤ) 
  (h_a : a = m^2 - n^2)
  (h_b : b = 2*m*n)
  (h_c : c = m^2 + n^2)
  (h_coprime : ¬(2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c)) :
  Odd c ∧ Even b ∧ Odd a := by
sorry

end pythagorean_triple_parity_l3022_302228


namespace nested_fraction_evaluation_l3022_302232

theorem nested_fraction_evaluation : 
  1 / (1 + 1 / (2 + 1 / (4^2))) = 33 / 49 := by
  sorry

end nested_fraction_evaluation_l3022_302232


namespace mikes_tire_changes_l3022_302270

/-- The number of tires changed by Mike in a day -/
def total_tires_changed (
  motorcycles cars bicycles trucks atvs : ℕ)
  (motorcycle_wheels car_wheels bicycle_wheels truck_wheels atv_wheels : ℕ) : ℕ :=
  motorcycles * motorcycle_wheels +
  cars * car_wheels +
  bicycles * bicycle_wheels +
  trucks * truck_wheels +
  atvs * atv_wheels

/-- Theorem stating the total number of tires changed by Mike in a day -/
theorem mikes_tire_changes :
  total_tires_changed 12 10 8 5 7 2 4 2 18 4 = 198 := by
  sorry

end mikes_tire_changes_l3022_302270


namespace relationship_abc_l3022_302215

theorem relationship_abc : ∃ (a b c : ℝ), 
  a = 2^(2/5) ∧ b = 9^(1/5) ∧ c = 3^(3/4) ∧ a < b ∧ b < c := by
  sorry

end relationship_abc_l3022_302215


namespace negPowersOfTwo_is_geometric_l3022_302274

/-- A sequence is geometric if it has a constant ratio between consecutive terms. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- A sequence of the form a_n = cq^n (where cq ≠ 0) is geometric. -/
axiom geometric_sequence_criterion (c q : ℝ) (hcq : c * q ≠ 0) :
  IsGeometricSequence (fun n => c * q ^ n)

/-- The sequence {-2^n} -/
def negPowersOfTwo (n : ℕ) : ℝ := -2 ^ n

/-- Theorem: The sequence {-2^n} is a geometric sequence -/
theorem negPowersOfTwo_is_geometric : IsGeometricSequence negPowersOfTwo := by
  sorry

end negPowersOfTwo_is_geometric_l3022_302274


namespace bean_ratio_l3022_302282

/-- Given a jar of beans with the following properties:
  - There are 572 beans in total
  - One-fourth of the beans are red
  - Half of the remaining beans after removing red are green
  - There are 143 green beans
  This theorem proves that the ratio of white beans to the remaining beans
  after removing red beans is 1:2. -/
theorem bean_ratio (total : ℕ) (red : ℕ) (green : ℕ) (white : ℕ) : 
  total = 572 →
  red = total / 4 →
  green = (total - red) / 2 →
  green = 143 →
  white = total - red - green →
  (white : ℚ) / (total - red - green : ℚ) = 1 / 2 := by
  sorry

end bean_ratio_l3022_302282


namespace eugene_model_house_l3022_302280

/-- The number of toothpicks Eugene uses for each card -/
def toothpicks_per_card : ℕ := 75

/-- The total number of cards in a deck -/
def cards_in_deck : ℕ := 52

/-- The number of cards Eugene didn't use -/
def unused_cards : ℕ := 16

/-- The number of toothpicks in each box -/
def toothpicks_per_box : ℕ := 450

/-- The number of boxes of toothpicks Eugene used -/
def boxes_used : ℕ := 6

theorem eugene_model_house :
  (cards_in_deck - unused_cards) * toothpicks_per_card / toothpicks_per_box = boxes_used :=
sorry

end eugene_model_house_l3022_302280


namespace equation_solutions_l3022_302279

theorem equation_solutions : 
  (∃ x : ℝ, x^2 - 4 = 0 ↔ x = 2 ∨ x = -2) ∧ 
  (∃ x : ℝ, (x + 3)^2 = (2*x - 1)*(x + 3) ↔ x = -3 ∨ x = 4) := by
  sorry

end equation_solutions_l3022_302279


namespace range_of_difference_l3022_302257

theorem range_of_difference (a b : ℝ) (ha : 12 < a ∧ a < 60) (hb : 15 < b ∧ b < 36) :
  -24 < a - b ∧ a - b < 45 := by
  sorry

end range_of_difference_l3022_302257


namespace square_difference_l3022_302221

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : 
  (x - y)^2 = 4 := by
  sorry

end square_difference_l3022_302221
