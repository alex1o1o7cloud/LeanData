import Mathlib

namespace complement_A_in_U_l1210_121008

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define set A
def A : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end complement_A_in_U_l1210_121008


namespace correct_answer_probability_l1210_121031

/-- The probability of correctly answering one MCQ question with 3 options -/
def mcq_prob : ℚ := 1 / 3

/-- The probability of correctly answering a true/false question -/
def tf_prob : ℚ := 1 / 2

/-- The number of true/false questions -/
def num_tf_questions : ℕ := 2

/-- The probability of correctly answering all questions -/
def total_prob : ℚ := mcq_prob * tf_prob ^ num_tf_questions

theorem correct_answer_probability :
  total_prob = 1 / 12 := by sorry

end correct_answer_probability_l1210_121031


namespace points_lost_l1210_121011

theorem points_lost (first_round : ℕ) (second_round : ℕ) (final_score : ℕ) :
  first_round = 40 →
  second_round = 50 →
  final_score = 86 →
  (first_round + second_round) - final_score = 4 :=
by
  sorry

end points_lost_l1210_121011


namespace inequality_cube_l1210_121048

theorem inequality_cube (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  (a - c)^3 > (b - c)^3 := by
  sorry

end inequality_cube_l1210_121048


namespace number_problem_l1210_121042

theorem number_problem (x : ℝ) : 5 * (x - 12) = 40 → x = 20 := by
  sorry

end number_problem_l1210_121042


namespace f_properties_l1210_121023

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (f (π / 12) = 1 / 2) ∧
  (Set.Icc 0 (3 / 2) = Set.image f (Set.Icc 0 (π / 2))) ∧
  (∃ (zeros : Finset ℝ), zeros.card = 5 ∧
    (∀ x ∈ zeros, x ∈ Set.Icc 0 (2 * π) ∧ f x = 0) ∧
    (∀ x ∈ Set.Icc 0 (2 * π), f x = 0 → x ∈ zeros)) ∧
  (∃ (zeros : Finset ℝ), zeros.card = 5 ∧
    (∀ x ∈ zeros, x ∈ Set.Icc 0 (2 * π) ∧ f x = 0) ∧
    (∀ x ∈ Set.Icc 0 (2 * π), f x = 0 → x ∈ zeros) ∧
    (zeros.sum id = 16 * π / 3)) :=
by sorry

end f_properties_l1210_121023


namespace modular_inverse_13_mod_101_l1210_121068

theorem modular_inverse_13_mod_101 : ∃ x : ℕ, x ≤ 100 ∧ (13 * x) % 101 = 1 :=
by
  use 70
  sorry

end modular_inverse_13_mod_101_l1210_121068


namespace f_is_quadratic_l1210_121051

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 3 - 5*x^2 - x

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end f_is_quadratic_l1210_121051


namespace smallest_linear_combination_3003_55555_l1210_121089

theorem smallest_linear_combination_3003_55555 :
  ∃ (k : ℕ), k > 0 ∧ (∃ (m n : ℤ), k = 3003 * m + 55555 * n) ∧
  ∀ (j : ℕ), j > 0 → (∃ (a b : ℤ), j = 3003 * a + 55555 * b) → j ≥ k :=
by
  -- The proof goes here
  sorry

end smallest_linear_combination_3003_55555_l1210_121089


namespace infinite_square_root_equals_three_l1210_121012

theorem infinite_square_root_equals_three :
  ∃ x : ℝ, x > 0 ∧ x = Real.sqrt (3 + 2 * x) → x = 3 := by
  sorry

end infinite_square_root_equals_three_l1210_121012


namespace prime_square_mod_twelve_l1210_121027

theorem prime_square_mod_twelve (p : ℕ) (hp : Nat.Prime p) (hp_gt_three : p > 3) :
  p^2 % 12 = 1 := by
  sorry

end prime_square_mod_twelve_l1210_121027


namespace complex_equation_solution_l1210_121075

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end complex_equation_solution_l1210_121075


namespace perfect_square_divisibility_l1210_121078

theorem perfect_square_divisibility (m n : ℕ) (h : m * n ∣ m^2 + n^2 + m) : 
  ∃ k : ℕ, m = k^2 := by
sorry

end perfect_square_divisibility_l1210_121078


namespace complex_power_215_36_l1210_121080

theorem complex_power_215_36 :
  (Complex.exp (215 * π / 180 * Complex.I)) ^ 36 = 1/2 - Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end complex_power_215_36_l1210_121080


namespace rhombus_area_l1210_121029

/-- Rhombus in a plane rectangular coordinate system -/
structure Rhombus where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The area of a rhombus given its vertices -/
def area (r : Rhombus) : ℝ := sorry

/-- Theorem: Area of rhombus ABCD with given conditions -/
theorem rhombus_area : 
  ∀ (r : Rhombus), 
    r.A = (-4, 0) →
    r.B = (0, -3) →
    (∃ (x y : ℝ), r.C = (x, 0) ∧ r.D = (0, y)) →  -- vertices on axes
    area r = 24 := by
  sorry

end rhombus_area_l1210_121029


namespace coefficient_of_3x_squared_l1210_121069

/-- Definition of a coefficient in a monomial term -/
def coefficient (term : ℝ → ℝ) : ℝ :=
  term 1

/-- The term 3x^2 -/
def term (x : ℝ) : ℝ := 3 * x^2

/-- Theorem: The coefficient of 3x^2 is 3 -/
theorem coefficient_of_3x_squared :
  coefficient term = 3 := by
  sorry

end coefficient_of_3x_squared_l1210_121069


namespace integer_roots_of_cubic_l1210_121096

def f (x : ℤ) : ℤ := x^3 - 4*x^2 - 7*x + 10

theorem integer_roots_of_cubic :
  {x : ℤ | f x = 0} = {1, -2, 5} := by sorry

end integer_roots_of_cubic_l1210_121096


namespace fibonacci_sum_odd_equals_next_l1210_121095

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def sum_odd_fibonacci (n : ℕ) : ℕ :=
  1 + (List.range n).foldl (λ acc i => acc + fibonacci (2 * i + 3)) 0

theorem fibonacci_sum_odd_equals_next (n : ℕ) :
  sum_odd_fibonacci n = fibonacci (2 * n + 2) := by
  sorry

#eval fibonacci 2018
#eval sum_odd_fibonacci 1008

end fibonacci_sum_odd_equals_next_l1210_121095


namespace star_op_equation_has_two_distinct_real_roots_l1210_121070

/-- Custom operation ※ -/
def star_op (a b : ℝ) : ℝ := a^2 * b + a * b - 1

/-- Theorem stating that x※1 = 0 has two distinct real roots -/
theorem star_op_equation_has_two_distinct_real_roots :
  ∃ x y : ℝ, x ≠ y ∧ star_op x 1 = 0 ∧ star_op y 1 = 0 := by
  sorry

end star_op_equation_has_two_distinct_real_roots_l1210_121070


namespace spherical_to_rectangular_conversion_l1210_121062

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 3
  let θ : ℝ := 3 * π / 2
  let φ : ℝ := π / 3
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) := by sorry

end spherical_to_rectangular_conversion_l1210_121062


namespace simplify_expression_l1210_121004

theorem simplify_expression (y : ℝ) : (5 * y)^3 + (4 * y) * (y^2) = 129 * y^3 := by
  sorry

end simplify_expression_l1210_121004


namespace pure_imaginary_condition_l1210_121086

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_condition (a : ℝ) :
  let z : ℂ := Complex.mk (a - 1) 1
  is_pure_imaginary z → a = 1 := by
  sorry

end pure_imaginary_condition_l1210_121086


namespace quadratic_inequality_solution_related_quadratic_inequality_solution_l1210_121050

/-- Solution set type -/
inductive SolutionSet
  | Empty
  | Interval (lower upper : ℝ)
  | Union (s1 s2 : SolutionSet)

/-- Solve quadratic inequality -/
noncomputable def solveQuadraticInequality (a : ℝ) : SolutionSet :=
  if a = 0 then
    SolutionSet.Empty
  else if a > 0 then
    SolutionSet.Interval (-a) (2 * a)
  else
    SolutionSet.Interval (2 * a) (-a)

/-- Theorem for part 1 -/
theorem quadratic_inequality_solution (a : ℝ) :
  solveQuadraticInequality a =
    if a = 0 then
      SolutionSet.Empty
    else if a > 0 then
      SolutionSet.Interval (-a) (2 * a)
    else
      SolutionSet.Interval (2 * a) (-a) :=
by sorry

/-- Theorem for part 2 -/
theorem related_quadratic_inequality_solution :
  ∃ (a b : ℝ), 
    (∀ x, x^2 - a*x - b < 0 ↔ -1 < x ∧ x < 2) →
    (∀ x, a*x^2 + x - b > 0 ↔ x < -2 ∨ x > 1) :=
by sorry

end quadratic_inequality_solution_related_quadratic_inequality_solution_l1210_121050


namespace focal_distance_l1210_121020

/-- Represents an ellipse with focal points and vertices -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Focal distance from center

/-- Properties of the ellipse based on given conditions -/
def ellipse_properties (e : Ellipse) : Prop :=
  e.a - e.c = 1.5 ∧  -- |F₂A| = 1.5
  2 * e.a = 5.4 ∧    -- |BC| = 5.4
  e.a^2 = e.b^2 + e.c^2

/-- Theorem stating the distance between focal points -/
theorem focal_distance (e : Ellipse) 
  (h : ellipse_properties e) : 2 * e.a - (e.a - e.c) = 13.5 := by
  sorry

#check focal_distance

end focal_distance_l1210_121020


namespace max_green_beads_l1210_121083

/-- A necklace with red, blue, and green beads. -/
structure Necklace :=
  (total : ℕ)
  (red : Finset ℕ)
  (blue : Finset ℕ)
  (green : Finset ℕ)

/-- The necklace satisfies the problem conditions. -/
def ValidNecklace (n : Necklace) : Prop :=
  n.total = 100 ∧
  n.red ∪ n.blue ∪ n.green = Finset.range n.total ∧
  (∀ i : ℕ, ∃ j ∈ n.blue, j % n.total ∈ Finset.range 5 ∪ {n.total - 1, n.total - 2, n.total - 3, n.total - 4}) ∧
  (∀ i : ℕ, ∃ j ∈ n.red, j % n.total ∈ Finset.range 7 ∪ {n.total - 1, n.total - 2, n.total - 3, n.total - 4, n.total - 5, n.total - 6})

/-- The maximum number of green beads in a valid necklace. -/
theorem max_green_beads (n : Necklace) (h : ValidNecklace n) :
  n.green.card ≤ 65 :=
sorry

end max_green_beads_l1210_121083


namespace pentagonal_figure_segment_length_pentagonal_figure_segment_length_proof_l1210_121018

/-- The length of segment AB in a folded pentagonal figure --/
theorem pentagonal_figure_segment_length : ℝ :=
  -- Define the side length of the regular pentagons
  let side_length : ℝ := 1

  -- Define the number of pentagons
  let num_pentagons : ℕ := 4

  -- Define the internal angle of a regular pentagon (in radians)
  let pentagon_angle : ℝ := 3 * Real.pi / 5

  -- Define the angle between the square base and the pentagon face (in radians)
  let folding_angle : ℝ := Real.pi / 2 - pentagon_angle / 2

  -- The length of segment AB
  2

/-- Proof of the pentagonal_figure_segment_length theorem --/
theorem pentagonal_figure_segment_length_proof :
  pentagonal_figure_segment_length = 2 := by
  sorry

end pentagonal_figure_segment_length_pentagonal_figure_segment_length_proof_l1210_121018


namespace distance_after_walk_on_hexagon_l1210_121024

/-- The distance from the starting point after walking along a regular hexagon's perimeter -/
theorem distance_after_walk_on_hexagon (side_length : ℝ) (walk_distance : ℝ) 
  (h1 : side_length = 3)
  (h2 : walk_distance = 10) :
  ∃ (end_point : ℝ × ℝ),
    (end_point.1^2 + end_point.2^2) = (3 * Real.sqrt 3)^2 := by
  sorry

end distance_after_walk_on_hexagon_l1210_121024


namespace square_fraction_count_l1210_121055

theorem square_fraction_count : 
  ∃! (S : Finset ℤ), 
    S.card = 2 ∧ 
    (∀ n ∈ S, 0 ≤ n ∧ n < 25 ∧ ∃ k : ℤ, (n : ℚ) / (25 - n) = k^2) ∧
    (∀ n : ℤ, 0 ≤ n → n < 25 → (∃ k : ℤ, (n : ℚ) / (25 - n) = k^2) → n ∈ S) :=
sorry

end square_fraction_count_l1210_121055


namespace remaining_nails_l1210_121076

def initial_nails : ℕ := 400

def kitchen_repair (n : ℕ) : ℕ := n - (n * 30 / 100)

def fence_repair (n : ℕ) : ℕ := n - (n * 70 / 100)

def table_repair (n : ℕ) : ℕ := n - (n * 50 / 100)

def floorboard_repair (n : ℕ) : ℕ := n - (n * 25 / 100)

theorem remaining_nails :
  floorboard_repair (table_repair (fence_repair (kitchen_repair initial_nails))) = 32 := by
  sorry

end remaining_nails_l1210_121076


namespace range_of_m_l1210_121049

theorem range_of_m (m : ℝ) : (∀ x ∈ Set.Icc 0 1, x^2 - 4*x ≥ m) → m ≤ -3 := by
  sorry

end range_of_m_l1210_121049


namespace license_plate_count_l1210_121092

/-- The set of odd single-digit numbers -/
def oddDigits : Finset Nat := {1, 3, 5, 7, 9}

/-- The set of prime numbers less than 10 -/
def primesLessThan10 : Finset Nat := {2, 3, 5, 7}

/-- The set of even single-digit numbers -/
def evenDigits : Finset Nat := {0, 2, 4, 6, 8}

/-- The number of letters in the alphabet -/
def alphabetSize : Nat := 26

theorem license_plate_count :
  (alphabetSize ^ 2) * oddDigits.card * primesLessThan10.card * evenDigits.card = 67600 := by
  sorry

#eval (alphabetSize ^ 2) * oddDigits.card * primesLessThan10.card * evenDigits.card

end license_plate_count_l1210_121092


namespace oranges_remaining_proof_l1210_121085

/-- The number of oranges Michaela needs to eat until she gets full -/
def michaela_oranges : ℕ := 30

/-- The number of oranges Cassandra needs to eat until she gets full -/
def cassandra_oranges : ℕ := 3 * michaela_oranges

/-- The total number of oranges picked from the farm -/
def total_oranges : ℕ := 200

/-- The number of oranges remaining after Michaela and Cassandra eat until they're full -/
def remaining_oranges : ℕ := total_oranges - (michaela_oranges + cassandra_oranges)

theorem oranges_remaining_proof : remaining_oranges = 80 := by
  sorry

end oranges_remaining_proof_l1210_121085


namespace minimum_walnuts_l1210_121030

/-- Represents the process of a child dividing and taking walnuts -/
def childProcess (n : ℕ) : ℕ := (n - 1) * 4 / 5

/-- Represents the final division process -/
def finalDivision (n : ℕ) : ℕ := n - 1

/-- Represents the entire walnut distribution process -/
def walnutDistribution (initial : ℕ) : ℕ :=
  finalDivision (childProcess (childProcess (childProcess (childProcess (childProcess initial)))))

theorem minimum_walnuts :
  ∃ (n : ℕ), n > 0 ∧ walnutDistribution n = 0 ∧ ∀ m : ℕ, m > 0 ∧ m < n → walnutDistribution m ≠ 0 :=
by sorry

end minimum_walnuts_l1210_121030


namespace log_inequality_l1210_121061

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_increasing : ∀ x y, 0 ≤ x → x < y → f x < f y

-- State the theorem
theorem log_inequality (x : ℝ) (h : f (Real.log x) < 0) : 0 < x ∧ x < 1 := by
  sorry

end log_inequality_l1210_121061


namespace swimmer_distance_l1210_121056

/-- Proves that a swimmer covers 8 km when swimming against a current for 5 hours -/
theorem swimmer_distance (swimmer_speed : ℝ) (current_speed : ℝ) (time : ℝ) :
  swimmer_speed = 3 →
  current_speed = 1.4 →
  time = 5 →
  (swimmer_speed - current_speed) * time = 8 := by
sorry

end swimmer_distance_l1210_121056


namespace division_problem_l1210_121007

theorem division_problem (total : ℝ) (a b c : ℝ) (h1 : total = 1080) 
  (h2 : a = (1/3) * (b + c)) (h3 : a = b + 30) (h4 : a + b + c = total) 
  (h5 : ∃ f : ℝ, b = f * (a + c)) : 
  ∃ f : ℝ, b = f * (a + c) ∧ f = 2/7 := by
  sorry

end division_problem_l1210_121007


namespace log_identity_l1210_121022

theorem log_identity (a : ℝ) (h : a * Real.log 4 / Real.log 3 = 1) :
  2^a + 2^(-a) = 4 * Real.sqrt 3 / 3 := by
  sorry

end log_identity_l1210_121022


namespace abs_sum_zero_implies_sum_l1210_121014

theorem abs_sum_zero_implies_sum (a b : ℝ) : 
  |a - 2| + |b + 3| = 0 → a + b = -1 := by
  sorry

end abs_sum_zero_implies_sum_l1210_121014


namespace sqrt_inequality_range_l1210_121072

theorem sqrt_inequality_range (m : ℝ) : 
  Real.sqrt (2 * m + 1) > Real.sqrt (m^2 + m - 1) ↔ 
  m ∈ Set.Ici ((Real.sqrt 5 - 1) / 2) ∩ Set.Iio 2 :=
sorry

end sqrt_inequality_range_l1210_121072


namespace odd_divisor_of_power_plus_one_l1210_121019

theorem odd_divisor_of_power_plus_one (n : ℕ) :
  n > 0 ∧ Odd n ∧ n ∣ (3^n + 1) ↔ n = 1 := by
  sorry

end odd_divisor_of_power_plus_one_l1210_121019


namespace sqrt_cos_squared_660_l1210_121045

theorem sqrt_cos_squared_660 : Real.sqrt (Real.cos (660 * π / 180) ^ 2) = 1 / 2 := by
  sorry

end sqrt_cos_squared_660_l1210_121045


namespace total_trees_l1210_121034

/-- Represents the tree-planting task for a school --/
structure TreePlantingTask where
  total : ℕ
  ninth_grade : ℕ
  eighth_grade : ℕ
  seventh_grade : ℕ

/-- The conditions of the tree-planting task --/
def tree_planting_conditions (a : ℕ) (task : TreePlantingTask) : Prop :=
  task.ninth_grade = task.total / 2 ∧
  task.eighth_grade = (task.total - task.ninth_grade) * 2 / 3 ∧
  task.seventh_grade = a ∧
  task.total = task.ninth_grade + task.eighth_grade + task.seventh_grade

/-- The theorem stating that the total number of trees is 6a --/
theorem total_trees (a : ℕ) (task : TreePlantingTask) 
  (h : tree_planting_conditions a task) : task.total = 6 * a :=
sorry

end total_trees_l1210_121034


namespace nasadkas_in_barrel_l1210_121084

/-- The volume of a barrel -/
def barrel : ℝ := sorry

/-- The volume of a nasadka -/
def nasadka : ℝ := sorry

/-- The volume of a bucket -/
def bucket : ℝ := sorry

/-- The first condition: 1 barrel + 20 buckets = 3 barrels -/
axiom condition1 : barrel + 20 * bucket = 3 * barrel

/-- The second condition: 19 barrels + 1 nasadka + 15.5 buckets = 20 barrels + 8 buckets -/
axiom condition2 : 19 * barrel + nasadka + 15.5 * bucket = 20 * barrel + 8 * bucket

/-- The theorem stating that there are 4 nasadkas in a barrel -/
theorem nasadkas_in_barrel : barrel / nasadka = 4 := by sorry

end nasadkas_in_barrel_l1210_121084


namespace two_unique_pairs_for_15_l1210_121032

/-- The number of unique pairs of nonnegative integers (a, b) satisfying a^2 - b^2 = n, for n = 15 -/
def uniquePairsCount (n : Nat) : Nat :=
  (Finset.filter (fun p : Nat × Nat => 
    p.1 ≥ p.2 ∧ p.1^2 - p.2^2 = n) (Finset.product (Finset.range (n+1)) (Finset.range (n+1)))).card

/-- Theorem stating that there are exactly 2 unique pairs for n = 15 -/
theorem two_unique_pairs_for_15 : uniquePairsCount 15 = 2 := by
  sorry

end two_unique_pairs_for_15_l1210_121032


namespace sin_product_equals_cos_over_eight_l1210_121087

theorem sin_product_equals_cos_over_eight :
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) * Real.sin (80 * π / 180) = 
  Real.cos (10 * π / 180) / 8 := by
  sorry

end sin_product_equals_cos_over_eight_l1210_121087


namespace triangle_area_theorem_l1210_121073

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : ℝ
  hb : ℝ
  hc : ℝ
  area : ℝ
  inradius : ℝ

-- Define the conditions
def validTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.ha > 0 ∧ t.hb > 0 ∧ t.hc > 0 ∧
  t.area > 0 ∧ t.inradius > 0

def integerAltitudes (t : Triangle) : Prop :=
  ∃ (n₁ n₂ n₃ : ℕ), t.ha = n₁ ∧ t.hb = n₂ ∧ t.hc = n₃

def altitudesSumLessThan20 (t : Triangle) : Prop :=
  t.ha + t.hb + t.hc < 20

def integerInradius (t : Triangle) : Prop :=
  ∃ (n : ℕ), t.inradius = n

-- State the theorem
theorem triangle_area_theorem (t : Triangle) 
  (h1 : validTriangle t)
  (h2 : integerAltitudes t)
  (h3 : altitudesSumLessThan20 t)
  (h4 : integerInradius t) :
  t.area = 6 ∨ t.area = 12 := by
  sorry

end triangle_area_theorem_l1210_121073


namespace arithmetic_sequence_length_l1210_121090

/-- 
Given an arithmetic sequence with:
- First term a = -36
- Common difference d = 6
- Last term l = 66

Prove that the number of terms in the sequence is 18.
-/
theorem arithmetic_sequence_length :
  ∀ (a d l : ℤ) (n : ℕ),
    a = -36 →
    d = 6 →
    l = 66 →
    l = a + (n - 1) * d →
    n = 18 :=
by sorry

end arithmetic_sequence_length_l1210_121090


namespace arithmetic_contains_geometric_l1210_121039

-- Define the arithmetic progression
def arithmetic_progression (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Theorem statement
theorem arithmetic_contains_geometric (a d : ℕ) (h : d > 0) :
  ∃ (r : ℕ) (b : ℕ → ℕ), 
    (∀ n : ℕ, ∃ k : ℕ, b n = arithmetic_progression a d k) ∧
    (∀ n : ℕ, b (n + 1) = r * b n) ∧
    (∀ n : ℕ, b n > 0) :=
sorry

end arithmetic_contains_geometric_l1210_121039


namespace reflection_line_correct_l1210_121067

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The reflection line that transforms one point to another -/
def reflection_line (p1 p2 : Point) : Line :=
  sorry

/-- Check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- The given points in the problem -/
def point1 : Point := ⟨5, 3⟩
def point2 : Point := ⟨1, -1⟩

/-- The proposed reflection line -/
def line_l : Line := ⟨-1, 4⟩

theorem reflection_line_correct :
  reflection_line point1 point2 = line_l ∧
  point_on_line (⟨3, 1⟩ : Point) line_l :=
sorry

end reflection_line_correct_l1210_121067


namespace average_temperature_twthf_l1210_121088

/-- The average temperature for Tuesday, Wednesday, Thursday, and Friday is 46 degrees -/
theorem average_temperature_twthf (temp_mon : ℝ) (temp_fri : ℝ) (avg_mtwth : ℝ) :
  temp_mon = 43 →
  temp_fri = 35 →
  avg_mtwth = 48 →
  let temp_twth : ℝ := (4 * avg_mtwth - temp_mon) / 3
  let avg_twthf : ℝ := (3 * temp_twth + temp_fri) / 4
  ∀ ε > 0, |avg_twthf - 46| < ε :=
by sorry

end average_temperature_twthf_l1210_121088


namespace f_geq_1_solution_set_g_max_value_l1210_121071

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Define the function g(x)
def g (x : ℝ) : ℝ := f x - x^2 + x

theorem f_geq_1_solution_set (x : ℝ) :
  f x ≥ 1 ↔ x ≥ 1 := by sorry

theorem g_max_value :
  ∃ x₀ : ℝ, ∀ x : ℝ, g x ≤ g x₀ ∧ g x₀ = 5/4 := by sorry

end f_geq_1_solution_set_g_max_value_l1210_121071


namespace all_naturals_reachable_l1210_121016

def triple_plus_one (x : ℕ) : ℕ := 3 * x + 1

def floor_half (x : ℕ) : ℕ := x / 2

def reachable (n : ℕ) : Prop :=
  ∃ (seq : List (ℕ → ℕ)), seq.foldl (λ acc f => f acc) 1 = n ∧
    ∀ f ∈ seq, f = triple_plus_one ∨ f = floor_half

theorem all_naturals_reachable : ∀ n : ℕ, reachable n := by
  sorry

end all_naturals_reachable_l1210_121016


namespace seunghye_number_l1210_121041

theorem seunghye_number (x : ℝ) : 10 * x - x = 37.35 → x = 4.15 := by
  sorry

end seunghye_number_l1210_121041


namespace calendar_cost_l1210_121053

/-- The cost of each calendar given the promotional item quantities and costs -/
theorem calendar_cost (total_items : ℕ) (num_calendars : ℕ) (num_datebooks : ℕ) 
  (datebook_cost : ℚ) (total_spent : ℚ) :
  total_items = 500 →
  num_calendars = 300 →
  num_datebooks = 200 →
  datebook_cost = 1/2 →
  total_spent = 300 →
  (total_spent - num_datebooks * datebook_cost) / num_calendars = 2/3 :=
by sorry

end calendar_cost_l1210_121053


namespace workers_wage_increase_l1210_121052

/-- Proves that if a worker's daily wage is increased by 50% to $42, then the original daily wage was $28. -/
theorem workers_wage_increase (original_wage : ℝ) (increased_wage : ℝ) : 
  increased_wage = 42 ∧ increased_wage = original_wage * 1.5 → original_wage = 28 := by
  sorry

end workers_wage_increase_l1210_121052


namespace square_roots_of_nine_l1210_121074

theorem square_roots_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end square_roots_of_nine_l1210_121074


namespace incenter_coeff_sum_specific_triangle_incenter_l1210_121010

/-- Given a triangle XYZ with sides x, y, z, the position vector of its incenter J
    can be expressed as J⃗ = p X⃗ + q Y⃗ + r Z⃗, where p, q, r are constants. -/
def incenter_position_vector (x y z : ℝ) (p q r : ℝ) : Prop :=
  p = x / (x + y + z) ∧ q = y / (x + y + z) ∧ r = z / (x + y + z)

/-- The sum of coefficients p, q, r in the incenter position vector equation is 1. -/
theorem incenter_coeff_sum (x y z : ℝ) (p q r : ℝ) 
  (h : incenter_position_vector x y z p q r) : p + q + r = 1 := by sorry

/-- For a triangle with sides 8, 11, and 5, the position vector of its incenter
    is given by (1/3, 11/24, 5/24). -/
theorem specific_triangle_incenter : 
  incenter_position_vector 8 11 5 (1/3) (11/24) (5/24) := by sorry

end incenter_coeff_sum_specific_triangle_incenter_l1210_121010


namespace tan_value_proof_l1210_121000

theorem tan_value_proof (α : Real) 
  (h1 : Real.sin α - Real.cos α = 1/5)
  (h2 : 0 < α ∧ α < Real.pi) : 
  Real.tan α = 4/3 := by
  sorry

end tan_value_proof_l1210_121000


namespace no_linear_factor_l1210_121015

theorem no_linear_factor (x y z : ℤ) : 
  ¬ ∃ (a b c d : ℤ), (a*x + b*y + c*z + d) ∣ (x^2 - y^2 - z^2 + 2*x*y + x + y - z) :=
sorry

end no_linear_factor_l1210_121015


namespace hotel_flat_fee_l1210_121013

/-- Given a hotel's pricing structure and two customer payments, prove the flat fee for the first night. -/
theorem hotel_flat_fee (f n : ℝ) 
  (ann_payment : f + n = 120)
  (bob_payment : f + 6 * n = 330) :
  f = 78 := by sorry

end hotel_flat_fee_l1210_121013


namespace max_true_statements_l1210_121059

theorem max_true_statements (a b : ℝ) : 
  ¬(∃ (s1 s2 s3 s4 : Bool), 
    (s1 → 1/a > 1/b) ∧
    (s2 → abs a > abs b) ∧
    (s3 → a > b) ∧
    (s4 → a < 0) ∧
    (¬s1 ∨ ¬s2 ∨ ¬s3 ∨ ¬s4 → b > 0) ∧
    s1 ∧ s2 ∧ s3 ∧ s4) :=
by sorry

end max_true_statements_l1210_121059


namespace shirt_tie_outfits_l1210_121037

theorem shirt_tie_outfits (shirts : ℕ) (ties : ℕ) :
  shirts = 7 → ties = 4 → shirts * ties = 28 := by
sorry

end shirt_tie_outfits_l1210_121037


namespace arithmetic_sequence_sum_remainder_l1210_121026

def arithmetic_sequence_sum (a : ℕ) (d : ℕ) (last : ℕ) : ℕ :=
  let n := (last - a) / d + 1
  n * (a + last) / 2

theorem arithmetic_sequence_sum_remainder
  (h1 : arithmetic_sequence_sum 3 6 273 % 6 = 0) : 
  arithmetic_sequence_sum 3 6 273 % 6 = 0 := by
  sorry

#check arithmetic_sequence_sum_remainder

end arithmetic_sequence_sum_remainder_l1210_121026


namespace banana_pie_angle_l1210_121077

theorem banana_pie_angle (total : ℕ) (chocolate apple blueberry : ℕ) :
  total = 48 →
  chocolate = 15 →
  apple = 10 →
  blueberry = 9 →
  let remaining := total - (chocolate + apple + blueberry)
  let banana := remaining / 2
  (banana : ℝ) / total * 360 = 52.5 := by
  sorry

end banana_pie_angle_l1210_121077


namespace cubic_root_sum_l1210_121097

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 12*a^2 + 27*a - 18 = 0 →
  b^3 - 12*b^2 + 27*b - 18 = 0 →
  c^3 - 12*c^2 + 27*c - 18 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^3 + 1/b^3 + 1/c^3 = 13/24 := by
sorry

end cubic_root_sum_l1210_121097


namespace tina_katya_difference_l1210_121001

/-- The number of glasses of lemonade sold by each person -/
structure LemonadeSales where
  katya : ℕ
  ricky : ℕ
  tina : ℕ

/-- The conditions of the lemonade sales problem -/
def lemonade_problem (sales : LemonadeSales) : Prop :=
  sales.katya = 8 ∧
  sales.ricky = 9 ∧
  sales.tina = 2 * (sales.katya + sales.ricky)

/-- The theorem stating the difference between Tina's and Katya's sales -/
theorem tina_katya_difference (sales : LemonadeSales) 
  (h : lemonade_problem sales) : sales.tina - sales.katya = 26 := by
  sorry

end tina_katya_difference_l1210_121001


namespace least_subtraction_for_divisibility_l1210_121043

theorem least_subtraction_for_divisibility : ∃ (n : ℕ), n = 8 ∧ 
  (∀ (m : ℕ), m < n → ¬(20 ∣ (50248 - m))) ∧ (20 ∣ (50248 - n)) := by
  sorry

end least_subtraction_for_divisibility_l1210_121043


namespace hundredth_training_day_l1210_121005

def training_program (start_day : Nat) (n : Nat) : Nat :=
  (start_day + (n - 1) * 8 + (n - 1) % 6) % 7

theorem hundredth_training_day :
  training_program 1 100 = 6 := by
  sorry

end hundredth_training_day_l1210_121005


namespace geometric_sequence_sum_l1210_121057

/-- Given a geometric sequence with sum of first n terms Sn = 24 and sum of first 3n terms S3n = 42,
    prove that the sum of first 2n terms S2n = 36 -/
theorem geometric_sequence_sum (n : ℕ) (Sn S2n S3n : ℝ) : 
  Sn = 24 → S3n = 42 → (S2n - Sn)^2 = Sn * (S3n - S2n) → S2n = 36 := by
sorry

end geometric_sequence_sum_l1210_121057


namespace complex_equation_solution_l1210_121099

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) * z = 2 → z = 1 - Complex.I := by
  sorry

end complex_equation_solution_l1210_121099


namespace fraction_relation_l1210_121064

theorem fraction_relation (a b c d : ℚ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c = d + 4) :
  d / a = 6 / 25 := by
  sorry

end fraction_relation_l1210_121064


namespace sixty_first_sample_number_l1210_121063

/-- Systematic sampling function -/
def systematicSample (totalItems : ℕ) (numGroups : ℕ) (firstItem : ℕ) (groupIndex : ℕ) : ℕ :=
  firstItem + (groupIndex - 1) * (totalItems / numGroups)

/-- Theorem stating the result of the 61st sample in the given conditions -/
theorem sixty_first_sample_number
  (totalItems : ℕ) (numGroups : ℕ) (firstItem : ℕ) (groupIndex : ℕ)
  (h1 : totalItems = 3000)
  (h2 : numGroups = 150)
  (h3 : firstItem = 11)
  (h4 : groupIndex = 61) :
  systematicSample totalItems numGroups firstItem groupIndex = 1211 := by
  sorry

#eval systematicSample 3000 150 11 61

end sixty_first_sample_number_l1210_121063


namespace unique_solution_l1210_121021

/-- The equation has two solutions and their sum is 12 -/
def has_two_solutions_sum_12 (a b : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ x ≠ 1 ∧ x ≠ -1 ∧ y ≠ 1 ∧ y ≠ -1 ∧
    (a * x^2 - 24 * x + b) / (x^2 - 1) = x ∧
    (a * y^2 - 24 * y + b) / (y^2 - 1) = y ∧
    x + y = 12

theorem unique_solution :
  ∀ a b : ℝ, has_two_solutions_sum_12 a b ↔ a = 35 ∧ b = -5819 :=
sorry

end unique_solution_l1210_121021


namespace polly_happy_tweets_l1210_121006

/-- Represents the number of tweets Polly makes per minute in different states -/
structure PollyTweets where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Represents the duration Polly spends in each state -/
structure Duration where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Calculates the total number of tweets given tweet rates and durations -/
def totalTweets (tweets : PollyTweets) (duration : Duration) : ℕ :=
  tweets.happy * duration.happy +
  tweets.hungry * duration.hungry +
  tweets.mirror * duration.mirror

/-- Theorem stating that Polly tweets 18 times per minute when happy -/
theorem polly_happy_tweets (tweets : PollyTweets) (duration : Duration) :
  tweets.hungry = 4 ∧
  tweets.mirror = 45 ∧
  duration.happy = 20 ∧
  duration.hungry = 20 ∧
  duration.mirror = 20 ∧
  totalTweets tweets duration = 1340 →
  tweets.happy = 18 := by
  sorry

end polly_happy_tweets_l1210_121006


namespace linear_system_solution_l1210_121054

theorem linear_system_solution (x y a : ℝ) : 
  x + 2*y = 2 → 
  2*x + y = a → 
  x + y = 5 → 
  a = 13 := by sorry

end linear_system_solution_l1210_121054


namespace existence_of_special_sequence_l1210_121040

theorem existence_of_special_sequence :
  ∃ (a : ℕ → ℕ),
    (∀ k, a k < a (k + 1)) ∧
    (∀ n : ℤ, ∃ N : ℕ, ∀ k > N, ¬ Prime (a k + n)) :=
sorry

end existence_of_special_sequence_l1210_121040


namespace prime_9k_plus_1_divides_cubic_l1210_121047

theorem prime_9k_plus_1_divides_cubic (p : Nat) (k : Nat) (h_prime : Nat.Prime p) (h_form : p = 9*k + 1) :
  ∃ n : ℤ, (n^3 - 3*n + 1) % p = 0 := by sorry

end prime_9k_plus_1_divides_cubic_l1210_121047


namespace right_triangle_side_length_l1210_121035

theorem right_triangle_side_length 
  (a b c : ℝ) 
  (hyp : a^2 + b^2 = c^2) -- Pythagorean theorem
  (hypotenuse : c = 13) 
  (side : a = 12) : 
  b = 5 := by
sorry

end right_triangle_side_length_l1210_121035


namespace actual_distance_between_towns_l1210_121046

-- Define the map distance between towns
def map_distance : ℝ := 18

-- Define the scale
def scale_inches : ℝ := 0.3
def scale_miles : ℝ := 5

-- Theorem to prove
theorem actual_distance_between_towns :
  (map_distance * scale_miles) / scale_inches = 300 := by
  sorry

end actual_distance_between_towns_l1210_121046


namespace square_floor_tiles_l1210_121066

theorem square_floor_tiles (s : ℕ) (h_odd : Odd s) (h_middle : (s + 1) / 2 = 49) :
  s * s = 9409 := by
  sorry

end square_floor_tiles_l1210_121066


namespace jackson_earnings_l1210_121033

def hourly_rate : ℝ := 5
def vacuuming_time : ℝ := 2
def vacuuming_repetitions : ℕ := 2
def dish_washing_time : ℝ := 0.5
def bathroom_cleaning_multiplier : ℕ := 3

def total_earnings : ℝ :=
  hourly_rate * (vacuuming_time * vacuuming_repetitions +
                 dish_washing_time +
                 bathroom_cleaning_multiplier * dish_washing_time)

theorem jackson_earnings :
  total_earnings = 30 := by
  sorry

end jackson_earnings_l1210_121033


namespace different_log_differences_l1210_121003

theorem different_log_differences (primes : Finset ℕ) : 
  primes = {3, 5, 7, 11} → 
  Finset.card (Finset.image (λ (p : ℕ × ℕ) => Real.log p.1 - Real.log p.2) 
    (Finset.filter (λ (p : ℕ × ℕ) => p.1 ≠ p.2) (primes.product primes))) = 12 := by
  sorry

end different_log_differences_l1210_121003


namespace first_three_terms_b_is_geometric_T_sum_l1210_121081

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

axiom S_def (n : ℕ) : S n = 2 * sequence_a n - 2 * n

theorem first_three_terms :
  sequence_a 1 = 2 ∧ sequence_a 2 = 6 ∧ sequence_a 3 = 14 := by sorry

def sequence_b (n : ℕ) : ℝ := sequence_a n + 2

theorem b_is_geometric :
  ∃ (r : ℝ), ∀ (n : ℕ), n ≥ 2 → sequence_b n = r * sequence_b (n-1) := by sorry

def T (n : ℕ) : ℝ := sorry

theorem T_sum (n : ℕ) :
  T n = (n + 1) * 2^(n + 2) + 4 - n * (n + 1) := by sorry

end first_three_terms_b_is_geometric_T_sum_l1210_121081


namespace gift_wrapping_problem_l1210_121002

/-- Given three rolls of wrapping paper where the first roll wraps 3 gifts,
    the second roll wraps 5 gifts, and the third roll wraps 4 gifts with no paper leftover,
    prove that the total number of gifts wrapped is 12. -/
theorem gift_wrapping_problem (rolls : Nat) (first_roll : Nat) (second_roll : Nat) (third_roll : Nat)
    (h1 : rolls = 3)
    (h2 : first_roll = 3)
    (h3 : second_roll = 5)
    (h4 : third_roll = 4)
    (h5 : rolls * first_roll ≥ first_roll + second_roll + third_roll) :
    first_roll + second_roll + third_roll = 12 := by
  sorry

end gift_wrapping_problem_l1210_121002


namespace probability_three_tails_l1210_121025

/-- The probability of getting exactly k successes in n trials
    with probability p for each trial -/
def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The number of coin flips -/
def numFlips : ℕ := 8

/-- The probability of getting tails -/
def probTails : ℚ := 2/3

/-- The number of tails we're interested in -/
def numTails : ℕ := 3

theorem probability_three_tails :
  binomialProbability numFlips numTails probTails = 448/6561 := by
  sorry

end probability_three_tails_l1210_121025


namespace min_operations_needed_l1210_121082

-- Define the type for letters
inductive Letter | A | B | C | D | E | F | G

-- Define the type for positions in the circle
inductive Position | Center | Top | TopRight | BottomRight | Bottom | BottomLeft | TopLeft

-- Define the configuration as a function from Position to Letter
def Configuration := Position → Letter

-- Define the initial configuration
def initial_config : Configuration := sorry

-- Define the final configuration
def final_config : Configuration := sorry

-- Define a valid operation
def valid_operation (c : Configuration) : Configuration := sorry

-- Define the number of operations needed to transform one configuration to another
def operations_needed (start finish : Configuration) : ℕ := sorry

-- The main theorem
theorem min_operations_needed :
  operations_needed initial_config final_config = 3 := by sorry

end min_operations_needed_l1210_121082


namespace unique_function_satisfying_conditions_l1210_121036

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 2 then 0 else 2 / (2 - x)

theorem unique_function_satisfying_conditions :
  (∀ x : ℝ, x ≥ 0 → f x ≥ 0) ∧
  (f 2 = 0) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x ≠ 0) ∧
  (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → f (x * f y) * f y = f (x + y)) ∧
  (∀ g : ℝ → ℝ, 
    ((∀ x : ℝ, x ≥ 0 → g x ≥ 0) ∧
     (g 2 = 0) ∧
     (∀ x : ℝ, 0 ≤ x ∧ x < 2 → g x ≠ 0) ∧
     (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → g (x * g y) * g y = g (x + y))) →
    (∀ x : ℝ, x ≥ 0 → g x = f x)) :=
by sorry

end unique_function_satisfying_conditions_l1210_121036


namespace tyler_meal_combinations_l1210_121009

/-- The number of types of meat available -/
def num_meats : ℕ := 4

/-- The number of types of vegetables available -/
def num_vegetables : ℕ := 5

/-- The number of types of desserts available -/
def num_desserts : ℕ := 5

/-- The number of types of drinks available -/
def num_drinks : ℕ := 4

/-- The number of vegetables Tyler must choose -/
def vegetables_to_choose : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The theorem stating the number of different meal combinations Tyler can choose -/
theorem tyler_meal_combinations : 
  num_meats * choose num_vegetables vegetables_to_choose * num_desserts * num_drinks = 800 := by
  sorry

end tyler_meal_combinations_l1210_121009


namespace fraction_equality_l1210_121017

theorem fraction_equality (a : ℕ+) :
  (a : ℚ) / (a + 45 : ℚ) = 3 / 4 → a = 135 := by
  sorry

end fraction_equality_l1210_121017


namespace pyramid_cross_section_theorem_l1210_121028

/-- Represents a regular pyramid -/
structure RegularPyramid where
  lateralEdgeLength : ℝ

/-- Represents a cross-section of a pyramid -/
structure CrossSection where
  areaRatio : ℝ  -- ratio of cross-section area to base area

/-- 
Given a regular pyramid with lateral edge length 3 cm, if a plane parallel to the base
creates a cross-section with an area 1/9 of the base area, then the lateral edge length
of the smaller pyramid removed is 1 cm.
-/
theorem pyramid_cross_section_theorem (p : RegularPyramid) (cs : CrossSection) :
  p.lateralEdgeLength = 3 → cs.areaRatio = 1/9 → 
  ∃ (smallerPyramid : RegularPyramid), smallerPyramid.lateralEdgeLength = 1 := by
  sorry

end pyramid_cross_section_theorem_l1210_121028


namespace prob_no_match_three_picks_correct_l1210_121038

/-- The probability of not having a matching pair after 3 picks from 3 pairs of socks -/
def prob_no_match_three_picks : ℚ := 2 / 5

/-- The number of pairs of socks -/
def num_pairs : ℕ := 3

/-- The total number of socks -/
def total_socks : ℕ := 2 * num_pairs

/-- The probability of picking a non-matching sock on the second draw -/
def prob_second_draw : ℚ := 4 / 5

/-- The probability of picking a non-matching sock on the third draw -/
def prob_third_draw : ℚ := 1 / 2

theorem prob_no_match_three_picks_correct :
  prob_no_match_three_picks = prob_second_draw * prob_third_draw :=
sorry

end prob_no_match_three_picks_correct_l1210_121038


namespace grape_rate_proof_l1210_121058

/-- The rate per kg for grapes -/
def grape_rate : ℝ := 70

/-- The rate per kg for mangoes -/
def mango_rate : ℝ := 55

/-- The quantity of grapes and mangoes in kg -/
def quantity : ℝ := 8

/-- The total cost paid to the shopkeeper -/
def total_cost : ℝ := 1000

theorem grape_rate_proof :
  grape_rate * quantity + mango_rate * quantity = total_cost :=
by sorry

end grape_rate_proof_l1210_121058


namespace even_multiple_six_sum_properties_l1210_121060

theorem even_multiple_six_sum_properties (a b : ℤ) 
  (h_a_even : Even a) (h_b_multiple_six : ∃ k, b = 6 * k) :
  Even (a + b) ∧ 
  (∃ m, a + b = 3 * m) ∧ 
  ¬(∀ (a b : ℤ), Even a → (∃ k, b = 6 * k) → ∃ n, a + b = 6 * n) ∧
  ∃ (a b : ℤ), Even a ∧ (∃ k, b = 6 * k) ∧ (∃ n, a + b = 6 * n) :=
by sorry

end even_multiple_six_sum_properties_l1210_121060


namespace rectangular_plot_area_l1210_121091

/-- Given a rectangular plot where the length is thrice the breadth and the breadth is 15 meters,
    prove that the area of the plot is 675 square meters. -/
theorem rectangular_plot_area (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  breadth = 15 →
  length = 3 * breadth →
  area = length * breadth →
  area = 675 := by
  sorry

end rectangular_plot_area_l1210_121091


namespace homes_cleaned_l1210_121098

-- Define the given conditions
def earnings_per_home : ℕ := 46
def total_earnings : ℕ := 276

-- Define the theorem to prove
theorem homes_cleaned (earnings_per_home : ℕ) (total_earnings : ℕ) : 
  total_earnings / earnings_per_home = 6 :=
sorry

end homes_cleaned_l1210_121098


namespace walking_scenario_solution_l1210_121044

/-- Represents the walking scenario between two people --/
structure WalkingScenario where
  speed_A : ℝ  -- Speed of person A in meters per minute
  time_A_start : ℝ  -- Time person A has been walking when B starts
  speed_diff : ℝ  -- Speed difference between person B and person A
  time_diff_AC_CB : ℝ  -- Time difference between A to C and C to B for person A
  time_diff_CA_BC : ℝ  -- Time difference between C to A and B to C for person B

/-- The main theorem about the walking scenario --/
theorem walking_scenario_solution (w : WalkingScenario) 
  (h_speed_diff : w.speed_diff = 30)
  (h_time_A_start : w.time_A_start = 5.5)
  (h_time_diff_AC_CB : w.time_diff_AC_CB = 4)
  (h_time_diff_CA_BC : w.time_diff_CA_BC = 3) :
  ∃ (time_A_to_C : ℝ) (distance_AB : ℝ),
    time_A_to_C = 10 ∧ distance_AB = 1440 := by
  sorry


end walking_scenario_solution_l1210_121044


namespace claire_pets_l1210_121094

theorem claire_pets (total_pets : ℕ) (male_pets : ℕ) 
  (h_total : total_pets = 90)
  (h_male : male_pets = 25) :
  ∃ (gerbils hamsters : ℕ),
    gerbils + hamsters = total_pets ∧
    (1 : ℚ) / 4 * gerbils + (1 : ℚ) / 3 * hamsters = male_pets ∧
    gerbils = 60 := by
  sorry

end claire_pets_l1210_121094


namespace mary_sheep_problem_l1210_121079

theorem mary_sheep_problem (initial_sheep : ℕ) : 
  (initial_sheep : ℚ) * (3/4) * (1/2) = 150 → initial_sheep = 400 := by
  sorry

end mary_sheep_problem_l1210_121079


namespace sum_of_coordinates_A_l1210_121093

/-- Given three points A, B, and C in a 2D plane satisfying specific conditions,
    prove that the sum of the coordinates of A is 3. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (C.1 - A.1) / (B.1 - A.1) = 1/2 →
  (C.2 - A.2) / (B.2 - A.2) = 1/2 →
  B = (2, 5) →
  C = (6, -3) →
  A.1 + A.2 = 3 := by
sorry

end sum_of_coordinates_A_l1210_121093


namespace quadratic_always_positive_range_l1210_121065

theorem quadratic_always_positive_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 2*a > 0) ↔ (0 < a ∧ a < 8) :=
sorry

end quadratic_always_positive_range_l1210_121065
