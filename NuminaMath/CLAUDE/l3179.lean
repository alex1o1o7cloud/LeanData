import Mathlib

namespace NUMINAMATH_CALUDE_opposite_numbers_linear_equation_l3179_317964

theorem opposite_numbers_linear_equation :
  ∀ x y : ℝ,
  (2 * x - 3 * y = 10) →
  (y = -x) →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_opposite_numbers_linear_equation_l3179_317964


namespace NUMINAMATH_CALUDE_exactly_seven_numbers_satisfy_conditions_l3179_317922

/-- A function that replaces a digit at position k with zero in a natural number n -/
def replace_digit_with_zero (n : ℕ) (k : ℕ) : ℕ := sorry

/-- A function that checks if a number ends with zero -/
def ends_with_zero (n : ℕ) : Prop := sorry

/-- A function that counts the number of digits in a natural number -/
def digit_count (n : ℕ) : ℕ := sorry

/-- The main theorem stating that there are exactly 7 numbers satisfying the conditions -/
theorem exactly_seven_numbers_satisfy_conditions : 
  ∃! (s : Finset ℕ), 
    (s.card = 7) ∧ 
    (∀ n ∈ s, 
      ¬ends_with_zero n ∧ 
      ∃ k, k < digit_count n ∧ 
        9 * replace_digit_with_zero n k = n) :=
by sorry

end NUMINAMATH_CALUDE_exactly_seven_numbers_satisfy_conditions_l3179_317922


namespace NUMINAMATH_CALUDE_triangle_equality_l3179_317937

theorem triangle_equality (a b c : ℝ) 
  (h1 : |a - b| ≥ |c|) 
  (h2 : |b - c| ≥ |a|) 
  (h3 : |c - a| ≥ |b|) : 
  a = b + c ∨ b = c + a ∨ c = a + b := by
  sorry

end NUMINAMATH_CALUDE_triangle_equality_l3179_317937


namespace NUMINAMATH_CALUDE_ones_digit_of_first_prime_in_sequence_l3179_317929

/-- A function that returns the ones digit of a natural number -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- Definition of an arithmetic sequence of five primes -/
def is_arithmetic_prime_sequence (p q r s t : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧ Nat.Prime t ∧
  q = p + 10 ∧ r = q + 10 ∧ s = r + 10 ∧ t = s + 10

theorem ones_digit_of_first_prime_in_sequence (p q r s t : ℕ) :
  is_arithmetic_prime_sequence p q r s t → p > 5 → ones_digit p = 1 :=
by sorry

end NUMINAMATH_CALUDE_ones_digit_of_first_prime_in_sequence_l3179_317929


namespace NUMINAMATH_CALUDE_train_speed_problem_l3179_317979

theorem train_speed_problem (initial_distance : ℝ) (speed_train1 : ℝ) (distance_before_meet : ℝ) (time_before_meet : ℝ) :
  initial_distance = 120 →
  speed_train1 = 40 →
  distance_before_meet = 70 →
  time_before_meet = 1 →
  ∃ speed_train2 : ℝ,
    speed_train2 = 30 ∧
    initial_distance - (speed_train1 + speed_train2) * time_before_meet = distance_before_meet :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3179_317979


namespace NUMINAMATH_CALUDE_not_corner_2010_l3179_317953

/-- Represents the sequence of corner numbers in the spiral -/
def corner_sequence : ℕ → ℕ
| 0 => 2
| 1 => 4
| n + 2 => corner_sequence (n + 1) + 8 * (n + 1)

/-- Checks if a number is a corner number in the spiral -/
def is_corner_number (n : ℕ) : Prop :=
  ∃ k : ℕ, corner_sequence k = n

/-- The main theorem stating that 2010 is not a corner number -/
theorem not_corner_2010 : ¬ is_corner_number 2010 := by
  sorry

#eval corner_sequence 0  -- Expected: 2
#eval corner_sequence 1  -- Expected: 4
#eval corner_sequence 2  -- Expected: 6
#eval corner_sequence 3  -- Expected: 10

end NUMINAMATH_CALUDE_not_corner_2010_l3179_317953


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l3179_317962

theorem polynomial_identity_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^4 = a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₄ + a₂ + a₀ = 41 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l3179_317962


namespace NUMINAMATH_CALUDE_investment_change_l3179_317995

/-- Proves that an investment decreasing by 25% and then increasing by 40% results in a 5% overall increase -/
theorem investment_change (initial_value : ℝ) (initial_value_pos : initial_value > 0) :
  let day1_value := initial_value * (1 - 0.25)
  let day2_value := day1_value * (1 + 0.40)
  let percent_change := (day2_value - initial_value) / initial_value * 100
  percent_change = 5 := by
  sorry

end NUMINAMATH_CALUDE_investment_change_l3179_317995


namespace NUMINAMATH_CALUDE_plumbing_equal_charge_time_l3179_317982

def pauls_visit_fee : ℚ := 55
def pauls_hourly_rate : ℚ := 35
def reliable_visit_fee : ℚ := 75
def reliable_hourly_rate : ℚ := 30

theorem plumbing_equal_charge_time : 
  ∃ h : ℚ, h > 0 ∧ (pauls_visit_fee + pauls_hourly_rate * h = reliable_visit_fee + reliable_hourly_rate * h) ∧ h = 4 := by
  sorry

end NUMINAMATH_CALUDE_plumbing_equal_charge_time_l3179_317982


namespace NUMINAMATH_CALUDE_werewolf_unreachable_l3179_317909

def is_black (x y : Int) : Bool :=
  x % 2 = y % 2

def possible_moves : List (Int × Int) :=
  [(1, 2), (2, -1), (-1, -2), (-2, 1)]

def reachable (start_x start_y end_x end_y : Int) : Prop :=
  ∃ (n : Nat), ∃ (moves : List (Int × Int)),
    moves.length = n ∧
    moves.all (λ m => m ∈ possible_moves) ∧
    (moves.foldl (λ (x, y) (dx, dy) => (x + dx, y + dy)) (start_x, start_y) = (end_x, end_y))

theorem werewolf_unreachable :
  ¬(reachable 26 10 42 2017) :=
by sorry

end NUMINAMATH_CALUDE_werewolf_unreachable_l3179_317909


namespace NUMINAMATH_CALUDE_probability_at_least_one_contract_probability_at_least_one_contract_proof_l3179_317912

theorem probability_at_least_one_contract 
  (p_hardware : ℚ) 
  (p_not_software : ℚ) 
  (p_both : ℚ) 
  (h1 : p_hardware = 3/4) 
  (h2 : p_not_software = 5/9) 
  (h3 : p_both = 71/180) -- 0.3944444444444444 ≈ 71/180
  : ℚ :=
  29/36

theorem probability_at_least_one_contract_proof 
  (p_hardware : ℚ) 
  (p_not_software : ℚ) 
  (p_both : ℚ) 
  (h1 : p_hardware = 3/4) 
  (h2 : p_not_software = 5/9) 
  (h3 : p_both = 71/180)
  : probability_at_least_one_contract p_hardware p_not_software p_both h1 h2 h3 = 29/36 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_contract_probability_at_least_one_contract_proof_l3179_317912


namespace NUMINAMATH_CALUDE_inequality_solution_l3179_317968

theorem inequality_solution (x : ℝ) : 
  (x / 4 ≤ 3 + 2 * x ∧ 3 + 2 * x < -3 * (1 + x^2)) ↔ x ≥ -12 / 7 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3179_317968


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3179_317952

theorem simplify_and_rationalize :
  ∃ (x : ℝ), (x = (Real.sqrt 2 / Real.sqrt 5) * (Real.sqrt 3 / Real.sqrt 7) * (Real.rpow 4 (1/3) / Real.sqrt 6)) ∧
             (x = (Real.rpow 4 (1/3) * Real.sqrt 35) / 35) := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3179_317952


namespace NUMINAMATH_CALUDE_special_triangle_properties_l3179_317939

/-- Represents a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Conditions for our specific triangle -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.A + t.C = 2 * Real.pi / 3 ∧
  t.b = 1 ∧
  0 < t.A ∧ t.A < Real.pi / 2 ∧
  0 < t.B ∧ t.B < Real.pi / 2 ∧
  0 < t.C ∧ t.C < Real.pi / 2

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  Real.sqrt 3 < t.a + t.c ∧ t.a + t.c ≤ 2 ∧
  ∃ (max_area : Real), max_area = Real.sqrt 3 / 4 ∧
    ∀ (area : Real), area = 1 / 2 * t.a * t.c * Real.sin t.B → area ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l3179_317939


namespace NUMINAMATH_CALUDE_trapezoid_area_l3179_317959

-- Define the lengths of the line segments
def a : ℝ := 1
def b : ℝ := 4
def c : ℝ := 4
def d : ℝ := 5

-- Define the possible areas
def area1 : ℝ := 6
def area2 : ℝ := 10

-- Statement of the theorem
theorem trapezoid_area :
  ∃ (S : ℝ), (S = area1 ∨ S = area2) ∧
  (∃ (h1 h2 base1 base2 : ℝ),
    (h1 = b ∧ h2 = c ∧ base1 = a ∧ base2 = d) ∨
    (h1 = b ∧ h2 = d ∧ base1 = a ∧ base2 = c) ∨
    (h1 = c ∧ h2 = d ∧ base1 = a ∧ base2 = b)) ∧
  S = (base1 + base2) * (h1 + h2) / 4 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3179_317959


namespace NUMINAMATH_CALUDE_converse_of_negative_square_positive_l3179_317934

theorem converse_of_negative_square_positive :
  (∀ x : ℝ, x < 0 → x^2 > 0) →
  (∀ x : ℝ, x^2 > 0 → x < 0) :=
sorry

end NUMINAMATH_CALUDE_converse_of_negative_square_positive_l3179_317934


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l3179_317987

theorem min_value_of_function (x : ℝ) (h : x > 2) : x + 4 / (x - 2) ≥ 6 := by
  sorry

theorem equality_condition (x : ℝ) (h : x > 2) : 
  ∃ x, x > 2 ∧ x + 4 / (x - 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l3179_317987


namespace NUMINAMATH_CALUDE_correct_mean_problem_l3179_317900

def correct_mean (n : ℕ) (original_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * original_mean - incorrect_value + correct_value) / n

theorem correct_mean_problem :
  correct_mean 20 36 40 25 = 35.25 := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_problem_l3179_317900


namespace NUMINAMATH_CALUDE_max_area_triangle_l3179_317975

/-- Given a triangle ABC where angle B equals angle C and 7a² + b² + c² = 4√3,
    the maximum possible area of the triangle is √5/5. -/
theorem max_area_triangle (a b c : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
    (h2 : 7 * a^2 + b^2 + c^2 = 4 * Real.sqrt 3)
    (h3 : b = c) : 
    ∃ (S : ℝ), S = Real.sqrt 5 / 5 ∧ 
    ∀ (A : ℝ), A = 1/2 * a * b * Real.sqrt (1 - (a / (2 * b))^2) → A ≤ S :=
by sorry

end NUMINAMATH_CALUDE_max_area_triangle_l3179_317975


namespace NUMINAMATH_CALUDE_pa_distance_bounds_l3179_317933

/-- Given a segment AB of length 2 and a point P satisfying |PA| + |PB| = 8,
    prove that the distance |PA| is bounded by 3 ≤ |PA| ≤ 5. -/
theorem pa_distance_bounds (A B P : EuclideanSpace ℝ (Fin 2)) 
  (h1 : dist A B = 2)
  (h2 : dist P A + dist P B = 8) :
  3 ≤ dist P A ∧ dist P A ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_pa_distance_bounds_l3179_317933


namespace NUMINAMATH_CALUDE_expression_value_l3179_317943

theorem expression_value (x y : ℝ) 
  (eq1 : x - y = -2)
  (eq2 : 2 * x + y = -1) :
  (x - y)^2 - (x - 2*y) * (x + 2*y) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3179_317943


namespace NUMINAMATH_CALUDE_sector_area_l3179_317981

/-- The area of a circular sector with central angle π/3 and radius 2 is 2π/3 -/
theorem sector_area (α : Real) (r : Real) (h1 : α = π / 3) (h2 : r = 2) :
  (1 / 2) * α * r^2 = (2 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3179_317981


namespace NUMINAMATH_CALUDE_triangle_height_relationship_l3179_317967

/-- Given two triangles A and B, proves the relationship between their heights
    when their bases and areas are related. -/
theorem triangle_height_relationship (b h : ℝ) (h_pos : 0 < h) (b_pos : 0 < b) :
  let base_A := 1.2 * b
  let area_B := (1 / 2) * b * h
  let area_A := 0.9975 * area_B
  let height_A := (2 * area_A) / base_A
  height_A / h = 0.83125 :=
by sorry

end NUMINAMATH_CALUDE_triangle_height_relationship_l3179_317967


namespace NUMINAMATH_CALUDE_person_age_l3179_317901

theorem person_age : ∃ (age : ℕ), 
  (4 * (age + 4) - 4 * (age - 4) = age) ∧ (age = 32) := by
  sorry

end NUMINAMATH_CALUDE_person_age_l3179_317901


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l3179_317926

theorem smallest_sum_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 20 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 20 → (x + y : ℕ) ≤ (a + b : ℕ)) → 
  (x + y : ℕ) = 81 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l3179_317926


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3179_317915

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_prod : a 1 * a 3 * a 11 = 8) : 
  a 2 * a 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3179_317915


namespace NUMINAMATH_CALUDE_cutting_process_result_l3179_317903

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square with side length -/
structure Square where
  side : ℕ

/-- Cuts the largest possible square from a rectangle and returns the remaining rectangle -/
def cutSquare (r : Rectangle) : Square × Rectangle :=
  if r.width ≤ r.height then
    ({ side := r.width }, { width := r.width, height := r.height - r.width })
  else
    ({ side := r.height }, { width := r.width - r.height, height := r.height })

/-- Applies the cutting process to a rectangle and returns the list of resulting squares -/
def cutProcess (r : Rectangle) : List Square :=
  sorry

/-- Theorem stating the result of applying the cutting process to a 14 × 36 rectangle -/
theorem cutting_process_result :
  let initial_rectangle : Rectangle := { width := 14, height := 36 }
  let result := cutProcess initial_rectangle
  (result.filter (λ s => s.side = 14)).length = 2 ∧
  (result.filter (λ s => s.side = 8)).length = 1 ∧
  (result.filter (λ s => s.side = 6)).length = 1 ∧
  (result.filter (λ s => s.side = 2)).length = 3 :=
sorry

end NUMINAMATH_CALUDE_cutting_process_result_l3179_317903


namespace NUMINAMATH_CALUDE_correct_calculation_l3179_317921

theorem correct_calculation (a b : ℝ) : 2 * a^2 * b - 4 * a^2 * b = -2 * a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3179_317921


namespace NUMINAMATH_CALUDE_max_value_expression_l3179_317986

theorem max_value_expression (x y z : ℝ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0) 
  (sum_constraint : x + y + z = 3) :
  (x^3 - x*y^2 + y^3) * (x^3 - x^2*z + z^3) * (y^3 - y^2*z + z^3) ≤ 1 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ ≥ 0 ∧ y₀ ≥ 0 ∧ z₀ ≥ 0 ∧ x₀ + y₀ + z₀ = 3 ∧
    (x₀^3 - x₀*y₀^2 + y₀^3) * (x₀^3 - x₀^2*z₀ + z₀^3) * (y₀^3 - y₀^2*z₀ + z₀^3) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3179_317986


namespace NUMINAMATH_CALUDE_equation_solution_l3179_317985

theorem equation_solution (x p : ℝ) : 
  (Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) = x) ↔ 
  (x = (4 - p) / Real.sqrt (8 * (2 - p)) ∧ 0 ≤ p ∧ p ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3179_317985


namespace NUMINAMATH_CALUDE_rainwater_farm_chickens_l3179_317945

/-- Represents the number of animals on Mr. Rainwater's farm -/
structure FarmAnimals where
  cows : Nat
  goats : Nat
  chickens : Nat
  ducks : Nat

/-- Defines the conditions for Mr. Rainwater's farm -/
def valid_farm (f : FarmAnimals) : Prop :=
  f.cows = 9 ∧
  f.goats = 4 * f.cows ∧
  f.goats = 2 * f.chickens ∧
  f.ducks = (3 * f.chickens) / 2 ∧
  (f.ducks - 2 * f.chickens) % 3 = 0 ∧
  f.goats + f.chickens + f.ducks ≤ 100

theorem rainwater_farm_chickens :
  ∀ f : FarmAnimals, valid_farm f → f.chickens = 18 :=
sorry

end NUMINAMATH_CALUDE_rainwater_farm_chickens_l3179_317945


namespace NUMINAMATH_CALUDE_x_plus_3y_equals_1_l3179_317974

theorem x_plus_3y_equals_1 (x y : ℝ) (h1 : x + y = 19) (h2 : x + 2*y = 10) : x + 3*y = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_3y_equals_1_l3179_317974


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3179_317994

def set_A : Set ℝ := {x | Real.sqrt (x + 1) < 2}
def set_B : Set ℝ := {x | 1 < x ∧ x < 4}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3179_317994


namespace NUMINAMATH_CALUDE_inconsistent_farm_animals_l3179_317917

theorem inconsistent_farm_animals :
  ∀ (x y z g : ℕ),
  x = 2 * y →
  y = 310 →
  z = 180 →
  x + y + z + g = 900 →
  g < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_inconsistent_farm_animals_l3179_317917


namespace NUMINAMATH_CALUDE_degree_of_x2y_l3179_317958

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree_of_monomial (exponents : List ℕ) : ℕ :=
  exponents.sum

/-- The monomial x^2y has exponents [2, 1] -/
def monomial_x2y_exponents : List ℕ := [2, 1]

theorem degree_of_x2y :
  degree_of_monomial monomial_x2y_exponents = 3 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_x2y_l3179_317958


namespace NUMINAMATH_CALUDE_roots_of_equation_l3179_317930

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => (x^2 - 5*x + 6)*(x - 3)*(x + 2)
  {x : ℝ | f x = 0} = {2, 3, -2} := by
sorry

end NUMINAMATH_CALUDE_roots_of_equation_l3179_317930


namespace NUMINAMATH_CALUDE_triangle_area_product_l3179_317940

theorem triangle_area_product (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a * x + b * y = 12) →
  (1/2 * (12/a) * (12/b) = 12) →
  a * b = 6 := by sorry

end NUMINAMATH_CALUDE_triangle_area_product_l3179_317940


namespace NUMINAMATH_CALUDE_first_few_terms_eighth_term_l3179_317920

/-- Definition of the sequence -/
def a (n : ℕ) : ℕ := n^2 + 2*n - 1

/-- The first few terms of the sequence -/
theorem first_few_terms :
  a 1 = 2 ∧ a 2 = 7 ∧ a 3 = 14 ∧ a 4 = 23 := by sorry

/-- The 8th term of the sequence is 79 -/
theorem eighth_term : a 8 = 79 := by sorry

end NUMINAMATH_CALUDE_first_few_terms_eighth_term_l3179_317920


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l3179_317906

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l3179_317906


namespace NUMINAMATH_CALUDE_smallest_k_satisfying_condition_l3179_317998

def S (n : ℕ) : ℤ := 2 * n^2 - 15 * n

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem smallest_k_satisfying_condition : 
  (∀ k < 6, a k + a (k + 1) ≤ 12) ∧ 
  (a 6 + a 7 > 12) := by sorry

end NUMINAMATH_CALUDE_smallest_k_satisfying_condition_l3179_317998


namespace NUMINAMATH_CALUDE_min_value_x_plus_4y_l3179_317977

theorem min_value_x_plus_4y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 2) : 
  x + 4*y ≥ 3/2 + Real.sqrt 2 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1/x₀ + 1/(2*y₀) = 2 ∧ 
    x₀ + 4*y₀ = 3/2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_4y_l3179_317977


namespace NUMINAMATH_CALUDE_xiaoming_relative_score_l3179_317955

def class_average : ℝ := 90
def xiaoming_score : ℝ := 85

theorem xiaoming_relative_score :
  xiaoming_score - class_average = -5 := by
sorry

end NUMINAMATH_CALUDE_xiaoming_relative_score_l3179_317955


namespace NUMINAMATH_CALUDE_bus_passenger_count_l3179_317989

def bus_passengers (initial_passengers : ℕ) (new_passengers : ℕ) : ℕ :=
  initial_passengers + new_passengers

theorem bus_passenger_count : 
  bus_passengers 4 13 = 17 := by sorry

end NUMINAMATH_CALUDE_bus_passenger_count_l3179_317989


namespace NUMINAMATH_CALUDE_two_hundred_fiftieth_term_is_331_l3179_317969

/-- The sequence function that generates the nth term of the sequence 
    by omitting perfect squares and multiples of 5 -/
def sequenceFunction (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the 250th term of the sequence is 331 -/
theorem two_hundred_fiftieth_term_is_331 : sequenceFunction 250 = 331 := by
  sorry

end NUMINAMATH_CALUDE_two_hundred_fiftieth_term_is_331_l3179_317969


namespace NUMINAMATH_CALUDE_total_days_1996_to_2000_l3179_317970

def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

def daysInYear (year : Nat) : Nat :=
  if isLeapYear year then 366 else 365

def totalDays (startYear endYear : Nat) : Nat :=
  (List.range (endYear - startYear + 1)).map (fun i => daysInYear (startYear + i))
    |> List.sum

theorem total_days_1996_to_2000 :
  totalDays 1996 2000 = 1827 := by sorry

end NUMINAMATH_CALUDE_total_days_1996_to_2000_l3179_317970


namespace NUMINAMATH_CALUDE_polynomial_property_l3179_317996

-- Define the polynomial Q(x)
def Q (x : ℝ) (d e : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + 9

-- Define the conditions
theorem polynomial_property (d e : ℝ) :
  -- The mean of zeros equals the product of zeros
  (-(d / 3) / 3 = -3) →
  -- The product of zeros equals the sum of coefficients
  (-3 = 3 + d + e + 9) →
  -- The y-intercept is 9
  (Q 0 d e = 9) →
  -- Prove that e equals -42
  e = -42 := by sorry

end NUMINAMATH_CALUDE_polynomial_property_l3179_317996


namespace NUMINAMATH_CALUDE_number_relationship_theorem_l3179_317902

theorem number_relationship_theorem :
  ∃ (x y a : ℝ), x = 6 * y - a ∧ x + y = 38 ∧
  (∀ (x' y' : ℝ), x' = 6 * y' - a ∧ x' + y' = 38 → x' = x ∧ y' = y → a = a) := by
  sorry

end NUMINAMATH_CALUDE_number_relationship_theorem_l3179_317902


namespace NUMINAMATH_CALUDE_negation_equivalence_l3179_317913

theorem negation_equivalence (m : ℤ) : 
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3179_317913


namespace NUMINAMATH_CALUDE_min_degree_for_connected_system_l3179_317999

/-- A graph representing a road system in a kingdom --/
structure RoadSystem where
  cities : Finset Nat
  roads : Finset (Nat × Nat)
  city_count : cities.card = 8
  road_symmetry : ∀ a b, (a, b) ∈ roads → (b, a) ∈ roads

/-- The maximum number of roads leading out from any city --/
def max_degree (g : RoadSystem) : Nat :=
  g.cities.sup (λ c => (g.roads.filter (λ r => r.1 = c)).card)

/-- A path between two cities with at most one intermediate city --/
def has_short_path (g : RoadSystem) (a b : Nat) : Prop :=
  (a, b) ∈ g.roads ∨ ∃ c, (a, c) ∈ g.roads ∧ (c, b) ∈ g.roads

/-- The property that any two cities are connected by a short path --/
def all_cities_connected (g : RoadSystem) : Prop :=
  ∀ a b, a ∈ g.cities → b ∈ g.cities → a ≠ b → has_short_path g a b

/-- The main theorem: the minimum degree for a connected road system is greater than 2 --/
theorem min_degree_for_connected_system (g : RoadSystem) (h : all_cities_connected g) :
  max_degree g > 2 := by
  sorry


end NUMINAMATH_CALUDE_min_degree_for_connected_system_l3179_317999


namespace NUMINAMATH_CALUDE_remaining_pages_l3179_317942

/-- Calculate the remaining pages of books after some are lost -/
theorem remaining_pages (initial_books : ℕ) (pages_per_book : ℕ) (lost_books : ℕ) 
  (h1 : initial_books ≥ lost_books) :
  (initial_books - lost_books) * pages_per_book = 
  initial_books * pages_per_book - lost_books * pages_per_book :=
by sorry

#check remaining_pages

end NUMINAMATH_CALUDE_remaining_pages_l3179_317942


namespace NUMINAMATH_CALUDE_number_1991_position_l3179_317966

/-- Represents a row in the number array -/
structure NumberArrayRow where
  startNumber : Nat
  length : Nat

/-- Defines the pattern of the number array -/
def numberArrayPattern (row : Nat) : NumberArrayRow :=
  { startNumber := row * 10,
    length := if row < 10 then row else 10 + (row - 10) * 10 }

/-- Checks if a number appears in a specific row and position -/
def appearsInRowAndPosition (n : Nat) (row : Nat) (position : Nat) : Prop :=
  let arrayRow := numberArrayPattern row
  n ≥ arrayRow.startNumber ∧ 
  n < arrayRow.startNumber + arrayRow.length ∧
  n = arrayRow.startNumber + position - 1

/-- Theorem stating that 1991 appears in the 199th row and 2nd position -/
theorem number_1991_position :
  appearsInRowAndPosition 1991 199 2 := by
  sorry


end NUMINAMATH_CALUDE_number_1991_position_l3179_317966


namespace NUMINAMATH_CALUDE_exchange_theorem_l3179_317965

/-- Represents the number of exchanges between Xiao Zhang and Xiao Li -/
def num_exchanges : ℕ := 4

/-- Xiao Zhang's initial number of pencils -/
def zhang_initial_pencils : ℕ := 200

/-- Xiao Li's initial number of fountain pens -/
def li_initial_pens : ℕ := 20

/-- Number of pencils Xiao Zhang gives in each exchange -/
def pencils_per_exchange : ℕ := 6

/-- Number of fountain pens Xiao Li gives in each exchange -/
def pens_per_exchange : ℕ := 1

/-- Xiao Zhang's final number of pencils -/
def zhang_final_pencils : ℕ := zhang_initial_pencils - num_exchanges * pencils_per_exchange

/-- Xiao Li's final number of fountain pens -/
def li_final_pens : ℕ := li_initial_pens - num_exchanges * pens_per_exchange

theorem exchange_theorem :
  zhang_final_pencils = 11 * li_final_pens :=
by sorry

end NUMINAMATH_CALUDE_exchange_theorem_l3179_317965


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3179_317988

theorem solution_set_inequality (x : ℝ) : 
  (x - 2) * (1 - 2*x) ≥ 0 ↔ 1/2 ≤ x ∧ x ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3179_317988


namespace NUMINAMATH_CALUDE_hallway_area_in_sq_yards_l3179_317992

-- Define the dimensions of the hallway
def hallway_length : ℝ := 15
def hallway_width : ℝ := 4

-- Define the conversion factor from square feet to square yards
def sq_feet_per_sq_yard : ℝ := 9

-- Theorem statement
theorem hallway_area_in_sq_yards :
  (hallway_length * hallway_width) / sq_feet_per_sq_yard = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hallway_area_in_sq_yards_l3179_317992


namespace NUMINAMATH_CALUDE_area_enclosed_circles_l3179_317938

/-- The area enclosed between the circumferences of four equal circles described about the corners of a square -/
theorem area_enclosed_circles (s : ℝ) (h : s = 14) :
  let r : ℝ := s / 2
  let square_area : ℝ := s ^ 2
  let circle_segment_area : ℝ := π * r ^ 2
  square_area - circle_segment_area = 196 - 49 * π :=
by sorry

end NUMINAMATH_CALUDE_area_enclosed_circles_l3179_317938


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3179_317923

/-- Given a quadratic inequality a*x^2 + b*x + 1 > 0 with solution set {x | -1 < x < 1/3},
    prove that the product ab equals -6. -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, a*x^2 + b*x + 1 > 0 ↔ -1 < x ∧ x < 1/3) → 
  a * b = -6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3179_317923


namespace NUMINAMATH_CALUDE_probability_of_specific_combination_l3179_317931

def total_marbles : ℕ := 12 + 8 + 5

def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def green_marbles : ℕ := 5

def marbles_drawn : ℕ := 4

def ways_to_draw_specific_combination : ℕ := (red_marbles.choose 2) * blue_marbles * green_marbles

def total_ways_to_draw : ℕ := total_marbles.choose marbles_drawn

theorem probability_of_specific_combination :
  (ways_to_draw_specific_combination : ℚ) / total_ways_to_draw = 264 / 1265 := by sorry

end NUMINAMATH_CALUDE_probability_of_specific_combination_l3179_317931


namespace NUMINAMATH_CALUDE_line_passes_through_point_l3179_317983

/-- Proves that k = 167/3 given that the line -1/3 - 3kx = 7y passes through the point (1/3, -8) -/
theorem line_passes_through_point (k : ℚ) : 
  (-1/3 : ℚ) - 3 * k * (1/3 : ℚ) = 7 * (-8 : ℚ) → k = 167/3 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l3179_317983


namespace NUMINAMATH_CALUDE_g_at_negative_one_l3179_317984

def g (x : ℚ) : ℚ := (2 * x - 3) / (5 * x + 2)

theorem g_at_negative_one : g (-1) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_g_at_negative_one_l3179_317984


namespace NUMINAMATH_CALUDE_a_has_winning_strategy_l3179_317960

/-- Represents the state of the game board -/
structure GameState where
  primes : List Nat
  product_mod_4 : Nat

/-- Represents a move in the game -/
inductive Move
  | erase_and_write (n : Nat) (erased : List Nat) (written : List Nat)

/-- The game between players A and B -/
def Game :=
  List Move

/-- Checks if a number is an odd prime -/
def is_odd_prime (n : Nat) : Prop :=
  Nat.Prime n ∧ n % 2 = 1

/-- The initial setup of the game -/
def initial_setup (primes : List Nat) : Prop :=
  primes.length = 1000 ∧ ∀ p ∈ primes, is_odd_prime p

/-- B's selection of primes -/
def b_selection (all_primes : List Nat) (selected : List Nat) : Prop :=
  selected.length = 500 ∧ ∀ p ∈ selected, p ∈ all_primes

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if a move is valid -/
def is_valid_move (state : GameState) (move : Move) : Prop :=
  sorry

/-- Checks if the game is over (board is empty) -/
def is_game_over (state : GameState) : Prop :=
  state.primes.isEmpty

/-- Player A's winning strategy -/
def a_winning_strategy (game : Game) : Prop :=
  sorry

/-- The main theorem stating that player A has a winning strategy -/
theorem a_has_winning_strategy 
  (initial_primes : List Nat)
  (h_initial : initial_setup initial_primes)
  (b_primes : List Nat)
  (h_b_selection : b_selection initial_primes b_primes) :
  ∃ (strategy : Game), a_winning_strategy strategy :=
sorry

end NUMINAMATH_CALUDE_a_has_winning_strategy_l3179_317960


namespace NUMINAMATH_CALUDE_max_value_constraint_l3179_317919

theorem max_value_constraint (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 4) :
  (10 * x + 3 * y + 15 * z)^2 ≤ 3220 / 36 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3179_317919


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l3179_317954

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := (a - 2) * x + 3 * y + a = 0
def l₂ (a x y : ℝ) : Prop := a * x + (a - 2) * y - 1 = 0

-- Define perpendicularity condition
def perpendicular (a : ℝ) : Prop := (a - 2) * a + 3 * (a - 2) = 0

-- Theorem statement
theorem perpendicular_lines_a_values :
  ∀ a : ℝ, perpendicular a → a = 2 ∨ a = -3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l3179_317954


namespace NUMINAMATH_CALUDE_shopkeeper_payment_l3179_317946

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def apply_successive_discounts (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem shopkeeper_payment (porcelain_price crystal_price : ℝ)
  (porcelain_discounts crystal_discounts : List ℝ) :
  porcelain_price = 8500 →
  crystal_price = 1500 →
  porcelain_discounts = [0.25, 0.15, 0.05] →
  crystal_discounts = [0.30, 0.10, 0.05] →
  (apply_successive_discounts porcelain_price porcelain_discounts +
   apply_successive_discounts crystal_price crystal_discounts) = 6045.56 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_payment_l3179_317946


namespace NUMINAMATH_CALUDE_investment_growth_proof_l3179_317957

/-- The initial investment amount that results in $132 after two years with given growth rates and addition --/
def initial_investment : ℝ := 80

/-- The growth rate for the first year --/
def first_year_growth_rate : ℝ := 0.15

/-- The amount added after the first year --/
def added_amount : ℝ := 28

/-- The growth rate for the second year --/
def second_year_growth_rate : ℝ := 0.10

/-- The final portfolio value after two years --/
def final_value : ℝ := 132

theorem investment_growth_proof :
  ((1 + first_year_growth_rate) * initial_investment + added_amount) * 
  (1 + second_year_growth_rate) = final_value := by
  sorry

#eval initial_investment

end NUMINAMATH_CALUDE_investment_growth_proof_l3179_317957


namespace NUMINAMATH_CALUDE_desk_rearrangement_combinations_l3179_317947

/-- The number of choices for each day of the week --/
def monday_choices : ℕ := 1
def tuesday_choices : ℕ := 3
def wednesday_choices : ℕ := 5
def thursday_choices : ℕ := 4
def friday_choices : ℕ := 1

/-- The total number of combinations --/
def total_combinations : ℕ := 
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices

/-- Theorem stating that the total number of combinations is 60 --/
theorem desk_rearrangement_combinations : total_combinations = 60 := by
  sorry

end NUMINAMATH_CALUDE_desk_rearrangement_combinations_l3179_317947


namespace NUMINAMATH_CALUDE_fifty_billion_scientific_notation_l3179_317924

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem fifty_billion_scientific_notation :
  toScientificNotation 50000000000 = ScientificNotation.mk 5 10 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_fifty_billion_scientific_notation_l3179_317924


namespace NUMINAMATH_CALUDE_soap_brand_usage_l3179_317918

theorem soap_brand_usage (total : ℕ) (neither : ℕ) (both : ℕ) :
  total = 180 →
  neither = 80 →
  both = 10 →
  ∃ (only_A only_B : ℕ),
    total = only_A + only_B + both + neither ∧
    only_B = 3 * both ∧
    only_A = 60 := by
  sorry

end NUMINAMATH_CALUDE_soap_brand_usage_l3179_317918


namespace NUMINAMATH_CALUDE_cone_base_circumference_l3179_317973

theorem cone_base_circumference (r : ℝ) (angle : ℝ) (h1 : r = 6) (h2 : angle = 120) :
  let original_circumference := 2 * π * r
  let sector_fraction := angle / 360
  let base_circumference := (1 - sector_fraction) * original_circumference
  base_circumference = 4 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l3179_317973


namespace NUMINAMATH_CALUDE_sin_graph_shift_l3179_317935

theorem sin_graph_shift (x : ℝ) :
  3 * Real.sin (2 * (x + π/8) - π/4) = 3 * Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_graph_shift_l3179_317935


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_is_32pi_l3179_317905

/-- Represents a triangular pyramid with vertex P and base ABC -/
structure TriangularPyramid where
  PA : ℝ
  AB : ℝ
  BC : ℝ
  angleABC : ℝ

/-- The surface area of the circumscribed sphere of a triangular pyramid -/
def circumscribedSphereSurfaceArea (pyramid : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem stating the surface area of the circumscribed sphere for the given pyramid -/
theorem circumscribed_sphere_surface_area_is_32pi :
  let pyramid : TriangularPyramid := {
    PA := 4,
    AB := 2,
    BC := 2,
    angleABC := 2 * Real.pi / 3  -- 120° in radians
  }
  circumscribedSphereSurfaceArea pyramid = 32 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_is_32pi_l3179_317905


namespace NUMINAMATH_CALUDE_frank_saturday_bags_l3179_317950

def total_cans : ℕ := 40
def cans_per_bag : ℕ := 5
def bags_filled_sunday : ℕ := 3

def total_bags : ℕ := total_cans / cans_per_bag

def bags_filled_saturday : ℕ := total_bags - bags_filled_sunday

theorem frank_saturday_bags :
  bags_filled_saturday = 5 :=
by sorry

end NUMINAMATH_CALUDE_frank_saturday_bags_l3179_317950


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3179_317997

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 9 → b = 12 → c^2 = a^2 + b^2 → c = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3179_317997


namespace NUMINAMATH_CALUDE_du_chin_meat_pie_business_l3179_317928

/-- Du Chin's meat pie business theorem -/
theorem du_chin_meat_pie_business 
  (pies_baked : ℕ) 
  (price_per_pie : ℚ) 
  (ingredient_cost_ratio : ℚ) 
  (h1 : pies_baked = 200)
  (h2 : price_per_pie = 20)
  (h3 : ingredient_cost_ratio = 3/5) :
  pies_baked * price_per_pie - pies_baked * price_per_pie * ingredient_cost_ratio = 1600 :=
by sorry

end NUMINAMATH_CALUDE_du_chin_meat_pie_business_l3179_317928


namespace NUMINAMATH_CALUDE_b_over_a_squared_is_seven_l3179_317927

theorem b_over_a_squared_is_seven (a : ℕ) (k : ℕ) (b : ℕ) :
  a > 1 →
  b = a * (10^k + 1) →
  k > 0 →
  a < 10^k →
  a^2 ∣ b →
  b / a^2 = 7 := by
sorry

end NUMINAMATH_CALUDE_b_over_a_squared_is_seven_l3179_317927


namespace NUMINAMATH_CALUDE_diamond_square_counts_l3179_317990

/-- Represents a diamond-shaped arrangement of colored squares -/
structure DiamondArrangement where
  sideLength : ℕ
  totalSquares : ℕ
  greenSquares : ℕ
  whiteSquares : ℕ

/-- Properties of the diamond arrangement -/
def validDiamondArrangement (d : DiamondArrangement) : Prop :=
  d.sideLength = 4 ∧
  d.totalSquares = (2 * d.sideLength + 1)^2 ∧
  d.greenSquares = (d.totalSquares + 1) / 2 ∧
  d.whiteSquares = (d.totalSquares - 1) / 2

theorem diamond_square_counts (d : DiamondArrangement) 
  (h : validDiamondArrangement d) : 
  d.whiteSquares = 40 ∧ 
  d.greenSquares = 41 ∧ 
  100 * d.whiteSquares + d.greenSquares = 4041 := by
  sorry

end NUMINAMATH_CALUDE_diamond_square_counts_l3179_317990


namespace NUMINAMATH_CALUDE_rotten_apples_percentage_l3179_317925

theorem rotten_apples_percentage (total : ℕ) (good : ℕ) 
  (h1 : total = 75) (h2 : good = 66) : 
  (((total - good : ℚ) / total) * 100 : ℚ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_rotten_apples_percentage_l3179_317925


namespace NUMINAMATH_CALUDE_estimate_black_pieces_is_twelve_l3179_317976

/-- Represents the result of drawing chess pieces -/
structure DrawResult where
  total_pieces : ℕ
  total_draws : ℕ
  black_draws : ℕ

/-- Estimates the number of black chess pieces in the bag -/
def estimate_black_pieces (result : DrawResult) : ℚ :=
  result.total_pieces * (result.black_draws : ℚ) / result.total_draws

/-- Theorem: The estimated number of black chess pieces is 12 -/
theorem estimate_black_pieces_is_twelve (result : DrawResult) 
  (h1 : result.total_pieces = 20)
  (h2 : result.total_draws = 100)
  (h3 : result.black_draws = 60) : 
  estimate_black_pieces result = 12 := by
  sorry

#eval estimate_black_pieces ⟨20, 100, 60⟩

end NUMINAMATH_CALUDE_estimate_black_pieces_is_twelve_l3179_317976


namespace NUMINAMATH_CALUDE_infinitely_many_lines_through_lattice_points_l3179_317948

/-- A line passing through the point (10, 1/2) -/
structure LineThrough10Half where
  slope : ℤ
  intercept : ℚ
  eq : intercept = 1/2 - 10 * slope

/-- A lattice point is a point with integer coordinates -/
def LatticePoint (x y : ℤ) : Prop := True

/-- A line passes through a lattice point -/
def PassesThroughLatticePoint (line : LineThrough10Half) (x y : ℤ) : Prop :=
  y = line.slope * x + line.intercept

theorem infinitely_many_lines_through_lattice_points :
  ∃ (f : ℕ → LineThrough10Half),
    (∀ n : ℕ, ∃ (x₁ y₁ x₂ y₂ : ℤ), 
      x₁ ≠ x₂ ∧ 
      LatticePoint x₁ y₁ ∧ 
      LatticePoint x₂ y₂ ∧ 
      PassesThroughLatticePoint (f n) x₁ y₁ ∧ 
      PassesThroughLatticePoint (f n) x₂ y₂) ∧
    (∀ m n : ℕ, m ≠ n → f m ≠ f n) :=
  sorry

end NUMINAMATH_CALUDE_infinitely_many_lines_through_lattice_points_l3179_317948


namespace NUMINAMATH_CALUDE_empty_solution_set_iff_a_in_range_l3179_317951

/-- The quadratic function f(x) = (a²-4)x² + (a+2)x - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 4) * x^2 + (a + 2) * x - 1

/-- The set of x that satisfy the inequality f(x) ≥ 0 -/
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | f a x ≥ 0}

/-- The theorem stating the range of a for which the solution set is empty -/
theorem empty_solution_set_iff_a_in_range :
  ∀ a : ℝ, solution_set a = ∅ ↔ a ∈ Set.Icc (-2) (6/5) := by sorry

end NUMINAMATH_CALUDE_empty_solution_set_iff_a_in_range_l3179_317951


namespace NUMINAMATH_CALUDE_initial_average_height_l3179_317956

/-- Given a class of students with an incorrect height measurement,
    prove that the initially calculated average height is 174 cm. -/
theorem initial_average_height
  (n : ℕ)  -- number of students
  (incorrect_height correct_height : ℝ)  -- heights of the misrecorded student
  (actual_average : ℝ)  -- actual average height after correction
  (h_n : n = 30)  -- there are 30 students
  (h_incorrect : incorrect_height = 151)  -- incorrectly recorded height
  (h_correct : correct_height = 136)  -- actual height of the misrecorded student
  (h_actual_avg : actual_average = 174.5)  -- actual average height
  : (n * actual_average - (incorrect_height - correct_height)) / n = 174 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_height_l3179_317956


namespace NUMINAMATH_CALUDE_day_301_is_sunday_l3179_317971

/-- Days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Function to determine the day of the week given a day number -/
def dayOfWeek (dayNumber : Nat) : DayOfWeek :=
  match dayNumber % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

/-- Theorem: If the 35th day is a Sunday, then the 301st day is also a Sunday -/
theorem day_301_is_sunday (h : dayOfWeek 35 = DayOfWeek.Sunday) :
  dayOfWeek 301 = DayOfWeek.Sunday :=
by
  sorry


end NUMINAMATH_CALUDE_day_301_is_sunday_l3179_317971


namespace NUMINAMATH_CALUDE_xy_bounds_l3179_317916

/-- Given a system of equations x + y = a and x^2 + y^2 = -a^2 + 2,
    prove that the product xy is bounded by -1 ≤ xy ≤ 1/3 -/
theorem xy_bounds (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) :
  -1 ≤ x * y ∧ x * y ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_xy_bounds_l3179_317916


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3179_317963

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3179_317963


namespace NUMINAMATH_CALUDE_point_transformation_l3179_317941

def rotate_180 (x y : ℝ) : ℝ × ℝ :=
  (4 - x, 6 - y)

def reflect_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  (reflect_y_eq_x (rotate_180 a b).1 (rotate_180 a b).2) = (2, -5) →
  a + b = 13 := by
sorry

end NUMINAMATH_CALUDE_point_transformation_l3179_317941


namespace NUMINAMATH_CALUDE_restaurant_bill_rounding_l3179_317949

theorem restaurant_bill_rounding (people : ℕ) (individual_payment : ℚ) (total_payment : ℚ) :
  people = 9 →
  individual_payment = 3491/100 →
  total_payment = 31419/100 →
  ∃ (original_bill : ℚ), 
    original_bill = 31418/100 ∧
    original_bill * people ≤ total_payment ∧
    total_payment - original_bill * people < people * (1/100) :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_rounding_l3179_317949


namespace NUMINAMATH_CALUDE_min_value_a_min_value_a_tight_l3179_317980

theorem min_value_a (a : ℝ) : 
  (∀ x > 0, x^2 + a*x + 1 ≥ 0) → a ≥ -2 :=
by sorry

theorem min_value_a_tight : 
  ∃ a : ℝ, (∀ x > 0, x^2 + a*x + 1 ≥ 0) ∧ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_min_value_a_tight_l3179_317980


namespace NUMINAMATH_CALUDE_large_duck_cost_large_duck_cost_proof_l3179_317991

/-- The cost of a large size duck given the following conditions:
  * Regular size ducks cost $3.00 each
  * 221 regular size ducks were sold
  * 185 large size ducks were sold
  * Total amount raised is $1588
-/
theorem large_duck_cost : ℝ → Prop :=
  λ large_cost : ℝ =>
    let regular_cost : ℝ := 3
    let regular_sold : ℕ := 221
    let large_sold : ℕ := 185
    let total_raised : ℝ := 1588
    (regular_cost * regular_sold + large_cost * large_sold = total_raised) →
    large_cost = 5

/-- Proof of the large duck cost theorem -/
theorem large_duck_cost_proof : large_duck_cost 5 := by
  sorry

end NUMINAMATH_CALUDE_large_duck_cost_large_duck_cost_proof_l3179_317991


namespace NUMINAMATH_CALUDE_arithmetic_contains_geometric_l3179_317944

/-- An arithmetic sequence of natural numbers -/
def ArithmeticSequence (a d : ℕ) : ℕ → ℕ := fun n => a + (n - 1) * d

theorem arithmetic_contains_geometric (a d : ℕ) (h : d > 0) :
  ∃ (r : ℚ) (f : ℕ → ℕ), 
    (∀ n, f n < f (n + 1)) ∧ 
    (∀ n, ArithmeticSequence a d (f n) * r = ArithmeticSequence a d (f (n + 1))) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_contains_geometric_l3179_317944


namespace NUMINAMATH_CALUDE_expression_evaluation_l3179_317911

theorem expression_evaluation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : 3 * a - b / 3 ≠ 0) :
  (3 * a - b / 3)⁻¹ * ((3 * a)⁻¹ + (b / 3)⁻¹) = (a * b)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3179_317911


namespace NUMINAMATH_CALUDE_fraction_bounds_l3179_317961

theorem fraction_bounds (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  0 ≤ (|x + y|^2) / (|x|^2 + |y|^2) ∧ (|x + y|^2) / (|x|^2 + |y|^2) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_bounds_l3179_317961


namespace NUMINAMATH_CALUDE_umar_age_l3179_317914

/-- Given the ages of Ali, Yusaf, and Umar, prove Umar's age -/
theorem umar_age (ali_age yusaf_age umar_age : ℕ) : 
  ali_age = 8 →
  ali_age = yusaf_age + 3 →
  umar_age = 2 * yusaf_age →
  umar_age = 10 := by
sorry

end NUMINAMATH_CALUDE_umar_age_l3179_317914


namespace NUMINAMATH_CALUDE_find_M_l3179_317904

theorem find_M : ∃ M : ℕ, (9.5 < (M : ℝ) / 4 ∧ (M : ℝ) / 4 < 10) ∧ M = 39 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l3179_317904


namespace NUMINAMATH_CALUDE_range_of_m2_plus_n2_l3179_317993

/-- An increasing function f with the property f(-x) + f(x) = 0 for all x -/
def IncreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) + f x = 0)

theorem range_of_m2_plus_n2 
  (f : ℝ → ℝ) (m n : ℝ) 
  (h_f : IncreasingOddFunction f) 
  (h_ineq : f (m^2 - 6*m + 21) + f (n^2 - 8*n) < 0) :
  9 < m^2 + n^2 ∧ m^2 + n^2 < 49 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m2_plus_n2_l3179_317993


namespace NUMINAMATH_CALUDE_constant_c_value_l3179_317907

theorem constant_c_value (b c : ℝ) : 
  (∀ x : ℝ, (x + 2) * (x + b) = x^2 + c*x + 6) → c = 5 := by
  sorry

end NUMINAMATH_CALUDE_constant_c_value_l3179_317907


namespace NUMINAMATH_CALUDE_continued_fraction_value_l3179_317972

theorem continued_fraction_value : 
  ∃ y : ℝ, y = 3 + 5 / (4 + 5 / y) ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_value_l3179_317972


namespace NUMINAMATH_CALUDE_paper_clips_remaining_l3179_317936

theorem paper_clips_remaining (initial : ℕ) (used : ℕ) (remaining : ℕ) : 
  initial = 85 → used = 59 → remaining = initial - used → remaining = 26 := by
  sorry

end NUMINAMATH_CALUDE_paper_clips_remaining_l3179_317936


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3179_317932

/-- A geometric sequence with first term a₁ and common ratio q -/
def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ :=
  λ n => a₁ * q^(n - 1)

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∃ (a₁ q : ℝ), ∀ n, a n = geometric_sequence a₁ q n)
  (h_a₁ : a 1 = 2)
  (h_a₄ : a 4 = 16) :
  ∃ q, ∀ n, a n = geometric_sequence 2 q n ∧ q = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3179_317932


namespace NUMINAMATH_CALUDE_first_group_men_count_l3179_317908

/-- Represents the amount of work that can be done by one person in one day -/
structure WorkRate where
  men : ℝ
  boys : ℝ

/-- Represents a group of workers -/
structure WorkGroup where
  men : ℕ
  boys : ℕ

/-- Represents a work scenario -/
structure WorkScenario where
  group : WorkGroup
  days : ℕ

theorem first_group_men_count (rate : WorkRate) 
  (scenario1 : WorkScenario)
  (scenario2 : WorkScenario)
  (scenario3 : WorkScenario) :
  scenario1.group.men = 6 :=
by
  sorry

#check first_group_men_count

end NUMINAMATH_CALUDE_first_group_men_count_l3179_317908


namespace NUMINAMATH_CALUDE_boxes_problem_l3179_317910

theorem boxes_problem (stan jules joseph john : ℕ) : 
  stan = 100 →
  joseph = stan / 5 →
  jules = joseph + 5 →
  john = jules + jules / 5 →
  john = 30 := by
sorry

end NUMINAMATH_CALUDE_boxes_problem_l3179_317910


namespace NUMINAMATH_CALUDE_probability_ratio_l3179_317978

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of distinct numbers on the slips -/
def distinct_numbers : ℕ := 10

/-- The number of slips for each number -/
def slips_per_number : ℕ := 5

/-- The number of slips drawn -/
def drawn_slips : ℕ := 5

/-- The probability of drawing all five slips with the same number -/
def p : ℚ := (distinct_numbers * 1) / Nat.choose total_slips drawn_slips

/-- The probability of drawing three slips with one number and two with another -/
def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 3 * Nat.choose slips_per_number 2) / Nat.choose total_slips drawn_slips

theorem probability_ratio :
  q / p = 450 := by sorry

end NUMINAMATH_CALUDE_probability_ratio_l3179_317978
