import Mathlib

namespace NUMINAMATH_CALUDE_round_trip_average_speed_l529_52962

/-- The average speed of a round trip given outbound and return speeds -/
theorem round_trip_average_speed
  (outbound_speed : ℝ)
  (return_speed : ℝ)
  (h1 : outbound_speed = 60)
  (h2 : return_speed = 40)
  : (2 / (1 / outbound_speed + 1 / return_speed)) = 48 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l529_52962


namespace NUMINAMATH_CALUDE_largest_n_for_product_1764_l529_52993

/-- Represents an arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

theorem largest_n_for_product_1764 :
  ∀ u v : ℕ,
  u ≥ 1 → v ≥ 1 → u ≤ v →
  ∃ n : ℕ, n ≥ 1 ∧
    (arithmeticSequence 3 u n) * (arithmeticSequence 3 v n) = 1764 →
  ∀ m : ℕ, m > 40 →
    (arithmeticSequence 3 u m) * (arithmeticSequence 3 v m) ≠ 1764 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_product_1764_l529_52993


namespace NUMINAMATH_CALUDE_seminar_attendees_seminar_attendees_solution_l529_52910

theorem seminar_attendees (total : ℕ) (company_a : ℕ) : ℕ :=
  let company_b := 2 * company_a
  let company_c := company_a + 10
  let company_d := company_c - 5
  let from_companies := company_a + company_b + company_c + company_d
  total - from_companies

theorem seminar_attendees_solution :
  seminar_attendees 185 30 = 20 := by
  sorry

end NUMINAMATH_CALUDE_seminar_attendees_seminar_attendees_solution_l529_52910


namespace NUMINAMATH_CALUDE_taxi_fare_distance_l529_52968

/-- Represents the fare structure and total charge for a taxi ride -/
structure TaxiFare where
  initialCharge : ℚ  -- Initial charge for the first 1/5 mile
  additionalCharge : ℚ  -- Charge for each additional 1/5 mile
  totalCharge : ℚ  -- Total charge for the ride

/-- Calculates the distance of a taxi ride given the fare structure and total charge -/
def calculateDistance (fare : TaxiFare) : ℚ :=
  let additionalDistance := (fare.totalCharge - fare.initialCharge) / fare.additionalCharge
  (additionalDistance + 1) / 5

/-- Theorem stating that for the given fare structure and total charge, the ride distance is 8 miles -/
theorem taxi_fare_distance (fare : TaxiFare) 
    (h1 : fare.initialCharge = 280/100)
    (h2 : fare.additionalCharge = 40/100)
    (h3 : fare.totalCharge = 1840/100) : 
  calculateDistance fare = 8 := by
  sorry

#eval calculateDistance { initialCharge := 280/100, additionalCharge := 40/100, totalCharge := 1840/100 }

end NUMINAMATH_CALUDE_taxi_fare_distance_l529_52968


namespace NUMINAMATH_CALUDE_no_integer_pairs_with_square_diff_150_l529_52902

theorem no_integer_pairs_with_square_diff_150 :
  ¬ ∃ (m n : ℕ), m ≥ n ∧ m^2 - n^2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_pairs_with_square_diff_150_l529_52902


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l529_52930

def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {-3, 1, 2, 4}

theorem intersection_of_A_and_B : A ∩ B = {-3, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l529_52930


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l529_52983

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 4 + a 7 = 45 →
  a 2 + a 5 + a 8 = 39 →
  a 3 + a 6 + a 9 = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l529_52983


namespace NUMINAMATH_CALUDE_smallest_n_value_l529_52998

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem smallest_n_value (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 2010 →
  is_even c →
  a.factorial * b.factorial * c.factorial = m * (10 ^ n) →
  ¬(10 ∣ m) →
  n ≥ 501 ∧ ∃ (a' b' c' m' : ℕ), 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    a' + b' + c' = 2010 ∧
    is_even c' ∧
    a'.factorial * b'.factorial * c'.factorial = m' * (10 ^ 501) ∧
    ¬(10 ∣ m') :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l529_52998


namespace NUMINAMATH_CALUDE_smallest_factorial_with_43_zeroes_l529_52941

/-- Count the number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 175 is the smallest positive integer k such that k! ends in at least 43 zeroes -/
theorem smallest_factorial_with_43_zeroes :
  (∀ k : ℕ, k > 0 → k < 175 → trailingZeroes k < 43) ∧ trailingZeroes 175 = 43 := by
  sorry

#eval trailingZeroes 175  -- Should output 43

end NUMINAMATH_CALUDE_smallest_factorial_with_43_zeroes_l529_52941


namespace NUMINAMATH_CALUDE_sugar_solution_percentage_l529_52927

theorem sugar_solution_percentage (x : ℝ) :
  (3/4 * x + 1/4 * 50 = 20) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_sugar_solution_percentage_l529_52927


namespace NUMINAMATH_CALUDE_max_profit_l529_52933

noncomputable def T (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x < 6 then (9*x - 2*x^2) / (6 - x)
  else 0

theorem max_profit :
  ∃ (x : ℝ), 1 ≤ x ∧ x < 6 ∧ T x = 3 ∧ ∀ y, T y ≤ T x :=
by
  sorry

end NUMINAMATH_CALUDE_max_profit_l529_52933


namespace NUMINAMATH_CALUDE_first_stick_length_l529_52925

theorem first_stick_length (stick1 stick2 stick3 : ℝ) : 
  stick2 = 2 * stick1 →
  stick3 = stick2 - 1 →
  stick1 + stick2 + stick3 = 14 →
  stick1 = 3 := by
sorry

end NUMINAMATH_CALUDE_first_stick_length_l529_52925


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l529_52905

/-- Two vectors in R² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (m + 1, -2)
  let b : ℝ × ℝ := (-3, 3)
  parallel a b → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l529_52905


namespace NUMINAMATH_CALUDE_proposition_implication_l529_52971

theorem proposition_implication (P : ℕ+ → Prop) 
  (h1 : ∀ k : ℕ+, P k → P (k + 1)) 
  (h2 : ¬ P 9) : 
  ¬ P 8 := by
  sorry

end NUMINAMATH_CALUDE_proposition_implication_l529_52971


namespace NUMINAMATH_CALUDE_f_of_tan_squared_l529_52985

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem f_of_tan_squared (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π/4) :
  (∀ x : ℝ, x ≠ 0 → x ≠ 1 → f (x / (x - 1)) = 1 / x) →
  f (Real.tan t ^ 2) = Real.tan t ^ 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_of_tan_squared_l529_52985


namespace NUMINAMATH_CALUDE_abs_neg_two_eq_two_l529_52929

theorem abs_neg_two_eq_two : |(-2 : ℝ)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_eq_two_l529_52929


namespace NUMINAMATH_CALUDE_relationship_between_a_b_c_l529_52937

theorem relationship_between_a_b_c : ∀ (a b c : ℝ),
  a = -(1^2) →
  b = (3 - Real.pi)^0 →
  c = (-0.25)^2023 * 4^2024 →
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_a_b_c_l529_52937


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l529_52908

theorem distance_to_y_axis (x : ℝ) :
  let P : ℝ × ℝ := (x, -5)
  (|P.2| = (1/2 : ℝ) * |P.1|) → |P.1| = 10 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l529_52908


namespace NUMINAMATH_CALUDE_church_cookie_baking_l529_52991

theorem church_cookie_baking (members : ℕ) (sheets_per_member : ℕ) (cookies_per_sheet : ℕ)
  (h1 : members = 100)
  (h2 : sheets_per_member = 10)
  (h3 : cookies_per_sheet = 16) :
  members * sheets_per_member * cookies_per_sheet = 16000 := by
  sorry

end NUMINAMATH_CALUDE_church_cookie_baking_l529_52991


namespace NUMINAMATH_CALUDE_function_range_l529_52977

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 1

-- Define the domain
def domain : Set ℝ := {x | -2 < x ∧ x < 1}

-- State the theorem
theorem function_range :
  {y | ∃ x ∈ domain, f x = y} = {y | -2 ≤ y ∧ y < 2} := by sorry

end NUMINAMATH_CALUDE_function_range_l529_52977


namespace NUMINAMATH_CALUDE_wild_animal_picture_difference_l529_52979

/-- The number of wild animal pictures Ralph has -/
def ralph_wild_animals : ℕ := 58

/-- The number of wild animal pictures Derrick has -/
def derrick_wild_animals : ℕ := 76

/-- Theorem stating the difference in wild animal pictures between Derrick and Ralph -/
theorem wild_animal_picture_difference :
  derrick_wild_animals - ralph_wild_animals = 18 := by sorry

end NUMINAMATH_CALUDE_wild_animal_picture_difference_l529_52979


namespace NUMINAMATH_CALUDE_moores_law_2010_l529_52997

def transistor_count (year : ℕ) : ℕ :=
  if year ≤ 2000 then
    2000000 * 2^((year - 1992) / 2)
  else
    2000000 * 2^4 * 4^((year - 2000) / 2)

theorem moores_law_2010 : transistor_count 2010 = 32768000000 := by
  sorry

end NUMINAMATH_CALUDE_moores_law_2010_l529_52997


namespace NUMINAMATH_CALUDE_surface_area_of_problem_structure_l529_52915

/-- Represents a solid formed by unit cubes -/
structure CubeStructure where
  base_layer : Nat
  middle_layer : Nat
  top_layer : Nat
  base_width : Nat
  base_length : Nat

/-- Calculates the surface area of the cube structure -/
def surface_area (c : CubeStructure) : Nat :=
  sorry

/-- The specific cube structure described in the problem -/
def problem_structure : CubeStructure :=
  { base_layer := 6
  , middle_layer := 4
  , top_layer := 2
  , base_width := 2
  , base_length := 3 }

theorem surface_area_of_problem_structure :
  surface_area problem_structure = 36 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_problem_structure_l529_52915


namespace NUMINAMATH_CALUDE_expression_value_l529_52936

theorem expression_value (x y : ℤ) (hx : x = -5) (hy : y = 8) :
  2 * (x - y)^2 - x * y = 378 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l529_52936


namespace NUMINAMATH_CALUDE_unique_operator_assignment_l529_52986

-- Define the arithmetic operators
inductive Operator
| Plus
| Minus
| Multiply
| Divide
| Equals

-- Define a function to apply an operator
def apply_operator (op : Operator) (a b : ℕ) : Prop :=
  match op with
  | Operator.Plus => a + b = b
  | Operator.Minus => a - b = b
  | Operator.Multiply => a * b = b
  | Operator.Divide => a / b = b
  | Operator.Equals => a = b

-- Define the theorem
theorem unique_operator_assignment :
  ∃! (A B C D E : Operator),
    apply_operator A 4 2 ∧
    apply_operator B 2 2 ∧
    apply_operator B 8 (4 * 2) ∧
    apply_operator C 4 2 ∧
    apply_operator D 2 3 ∧
    apply_operator B 5 5 ∧
    apply_operator B 4 (5 - 1) ∧
    apply_operator E 5 1 ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E :=
sorry

end NUMINAMATH_CALUDE_unique_operator_assignment_l529_52986


namespace NUMINAMATH_CALUDE_pencils_broken_l529_52943

theorem pencils_broken (initial bought found misplaced final : ℕ) : 
  initial = 20 → 
  bought = 2 → 
  found = 4 → 
  misplaced = 7 → 
  final = 16 → 
  initial + bought + found - misplaced - final = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencils_broken_l529_52943


namespace NUMINAMATH_CALUDE_johns_grocery_spending_l529_52922

theorem johns_grocery_spending (total_spent : ℚ) 
  (meat_fraction : ℚ) (bakery_fraction : ℚ) (candy_spent : ℚ) :
  total_spent = 24 →
  meat_fraction = 1/3 →
  bakery_fraction = 1/6 →
  candy_spent = 6 →
  total_spent - (meat_fraction * total_spent + bakery_fraction * total_spent) - candy_spent = 1/4 * total_spent :=
by sorry

end NUMINAMATH_CALUDE_johns_grocery_spending_l529_52922


namespace NUMINAMATH_CALUDE_xyz_equals_seven_cubed_l529_52944

theorem xyz_equals_seven_cubed 
  (x y z : ℝ) 
  (h1 : x^2 * y * z^3 = 7^4) 
  (h2 : x * y^2 = 7^5) : 
  x * y * z = 7^3 := by
  sorry

end NUMINAMATH_CALUDE_xyz_equals_seven_cubed_l529_52944


namespace NUMINAMATH_CALUDE_percentage_of_125_equal_to_70_l529_52994

theorem percentage_of_125_equal_to_70 : 
  ∃ p : ℝ, p * 125 = 70 ∧ p = 56 / 100 := by sorry

end NUMINAMATH_CALUDE_percentage_of_125_equal_to_70_l529_52994


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l529_52999

theorem reciprocal_inequality (a b : ℝ) :
  (a > b ∧ a * b > 0 → 1 / a < 1 / b) ∧
  (a > b ∧ a * b < 0 → 1 / a > 1 / b) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l529_52999


namespace NUMINAMATH_CALUDE_diagonal_crosses_820_cubes_l529_52960

/-- The number of unit cubes crossed by an internal diagonal in a rectangular solid. -/
def cubesCrossedByDiagonal (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem stating that the number of cubes crossed by the diagonal in a 200 × 330 × 360 solid is 820. -/
theorem diagonal_crosses_820_cubes :
  cubesCrossedByDiagonal 200 330 360 = 820 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_crosses_820_cubes_l529_52960


namespace NUMINAMATH_CALUDE_existence_of_special_set_l529_52965

theorem existence_of_special_set (n : ℕ) (h : n ≥ 2) :
  ∃ (S : Finset ℤ), Finset.card S = n ∧
    ∀ (a b : ℤ), a ∈ S → b ∈ S → a ≠ b → (a - b)^2 ∣ (a * b) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_set_l529_52965


namespace NUMINAMATH_CALUDE_geometry_relationships_l529_52956

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem geometry_relationships 
  (l m : Line) (a : Plane) (h_diff : l ≠ m) :
  (perpendicular l a ∧ contains a m → line_perpendicular l m) ∧
  (perpendicular l a ∧ line_parallel l m → perpendicular m a) ∧
  ¬(parallel l a ∧ contains a m → line_parallel l m) ∧
  ¬(parallel l a ∧ parallel m a → line_parallel l m) :=
sorry

end NUMINAMATH_CALUDE_geometry_relationships_l529_52956


namespace NUMINAMATH_CALUDE_max_both_writers_and_editors_l529_52948

/-- Represents the number of people at the newspaper conference --/
def total_people : ℕ := 150

/-- Represents the number of writers at the conference --/
def writers : ℕ := 50

/-- Represents the number of editors at the conference --/
def editors : ℕ := 66

/-- Represents the number of people who are both writers and editors --/
def both (x : ℕ) : ℕ := x

/-- Represents the number of people who are neither writers nor editors --/
def neither (x : ℕ) : ℕ := 3 * x

/-- States that the number of editors is more than 65 --/
axiom editors_more_than_65 : editors > 65

/-- Theorem stating that the maximum number of people who are both writers and editors is 17 --/
theorem max_both_writers_and_editors :
  ∃ (x : ℕ), x ≤ 17 ∧
  total_people = writers + editors - both x + neither x ∧
  ∀ (y : ℕ), y > x →
    total_people ≠ writers + editors - both y + neither y :=
sorry

end NUMINAMATH_CALUDE_max_both_writers_and_editors_l529_52948


namespace NUMINAMATH_CALUDE_first_group_selection_is_five_l529_52973

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  group_count : ℕ
  group_size : ℕ
  selected_number : ℕ
  selected_group : ℕ

/-- Calculates the position of a number within its group -/
def position_in_group (s : SystematicSampling) : ℕ :=
  s.selected_number - (s.selected_group - 1) * s.group_size

/-- Calculates the number selected from the first group -/
def first_group_selection (s : SystematicSampling) : ℕ :=
  position_in_group s

/-- Theorem stating the correct number selected from the first group -/
theorem first_group_selection_is_five (s : SystematicSampling) 
  (h1 : s.total_students = 160)
  (h2 : s.sample_size = 20)
  (h3 : s.group_count = 20)
  (h4 : s.group_size = 8)
  (h5 : s.selected_number = 125)
  (h6 : s.selected_group = 16) : 
  first_group_selection s = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_group_selection_is_five_l529_52973


namespace NUMINAMATH_CALUDE_prism_in_sphere_lateral_edge_l529_52964

/-- A prism with a square base and lateral edges perpendicular to the base -/
structure Prism where
  base_side : ℝ
  lateral_edge : ℝ

/-- A sphere -/
structure Sphere where
  radius : ℝ

/-- Theorem: The length of the lateral edge of a prism inscribed in a sphere -/
theorem prism_in_sphere_lateral_edge 
  (p : Prism) 
  (s : Sphere) 
  (h1 : p.base_side = 1) 
  (h2 : s.radius = 1) 
  (h3 : s.radius = Real.sqrt (p.base_side^2 + p.base_side^2 + p.lateral_edge^2) / 2) : 
  p.lateral_edge = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_prism_in_sphere_lateral_edge_l529_52964


namespace NUMINAMATH_CALUDE_vector_expression_l529_52926

/-- Given vectors a, b, and c in ℝ², prove that c = 2a + b -/
theorem vector_expression (a b c : ℝ × ℝ) 
  (h1 : a = (1, 2)) 
  (h2 : a + b = (0, 3)) 
  (h3 : c = (1, 5)) : 
  c = 2 • a + b := by sorry

end NUMINAMATH_CALUDE_vector_expression_l529_52926


namespace NUMINAMATH_CALUDE_base7_to_base49_conversion_l529_52916

/-- Converts a list of digits in base 7 to a natural number -/
def fromBase7 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Converts a natural number to a list of digits in base 49 -/
def toBase49 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 49) ((m % 49) :: acc)
    aux n []

theorem base7_to_base49_conversion :
  toBase49 (fromBase7 [6, 2, 6]) = [0, 6, 0, 2, 0, 6] := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base49_conversion_l529_52916


namespace NUMINAMATH_CALUDE_paintbrush_cost_calculation_l529_52932

/-- The cost of the paintbrush Rose wants to buy -/
def paintbrush_cost (paints_cost easel_cost rose_has rose_needs : ℚ) : ℚ :=
  (rose_has + rose_needs) - (paints_cost + easel_cost)

/-- Theorem stating the cost of the paintbrush Rose wants to buy -/
theorem paintbrush_cost_calculation :
  paintbrush_cost 9.20 6.50 7.10 11 = 2.40 := by sorry

end NUMINAMATH_CALUDE_paintbrush_cost_calculation_l529_52932


namespace NUMINAMATH_CALUDE_bag_price_problem_l529_52921

theorem bag_price_problem (P : ℝ) : 
  (P - P * 0.95 * 0.96 = 44) → P = 500 := by
  sorry

end NUMINAMATH_CALUDE_bag_price_problem_l529_52921


namespace NUMINAMATH_CALUDE_evaluate_expression_l529_52901

theorem evaluate_expression : 3000 * (3000 ^ 2999) = 3000 ^ 3000 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l529_52901


namespace NUMINAMATH_CALUDE_hat_color_game_l529_52980

/-- Represents the maximum number of correct guesses in the hat color game -/
def max_correct_guesses (n k : ℕ) : ℕ :=
  n - k - 1

/-- Theorem stating the maximum number of guaranteed correct guesses in the hat color game -/
theorem hat_color_game (n k : ℕ) (h1 : k < n) :
  max_correct_guesses n k = n - k - 1 :=
by sorry

end NUMINAMATH_CALUDE_hat_color_game_l529_52980


namespace NUMINAMATH_CALUDE_circle_chord_intersection_l529_52934

theorem circle_chord_intersection (r : ℝ) (chord_length : ℝ) :
  r = 8 →
  chord_length = 12 →
  ∃ (ak kb : ℝ),
    ak = 8 - 2 * Real.sqrt 7 ∧
    kb = 8 + 2 * Real.sqrt 7 ∧
    ak + kb = 2 * r :=
by sorry

end NUMINAMATH_CALUDE_circle_chord_intersection_l529_52934


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l529_52923

-- Define the quadratic equation
def quadratic_equation (y : ℝ) : ℝ := 3 * y^2 - 5 * y + 2

-- State the theorem
theorem parabola_y_intercepts :
  ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ quadratic_equation y₁ = 0 ∧ quadratic_equation y₂ = 0 ∧
  ∀ (y : ℝ), quadratic_equation y = 0 → y = y₁ ∨ y = y₂ :=
sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l529_52923


namespace NUMINAMATH_CALUDE_increase_amount_is_four_l529_52974

/-- Represents a set of numbers with a known size and average -/
structure NumberSet where
  size : ℕ
  average : ℝ

/-- Calculates the sum of elements in a NumberSet -/
def NumberSet.sum (s : NumberSet) : ℝ := s.size * s.average

/-- The original set of numbers -/
def original_set : NumberSet := { size := 10, average := 6.2 }

/-- The new set of numbers after increasing one element -/
def new_set : NumberSet := { size := 10, average := 6.6 }

/-- The theorem to be proved -/
theorem increase_amount_is_four :
  new_set.sum - original_set.sum = 4 := by sorry

end NUMINAMATH_CALUDE_increase_amount_is_four_l529_52974


namespace NUMINAMATH_CALUDE_complex_magnitude_from_equation_l529_52906

theorem complex_magnitude_from_equation (z : ℂ) : 
  Complex.I * (1 - z) = 1 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_from_equation_l529_52906


namespace NUMINAMATH_CALUDE_option_d_is_deductive_reasoning_l529_52967

/-- A predicate representing periodic functions --/
def IsPeriodic (f : ℝ → ℝ) : Prop := sorry

/-- A predicate representing trigonometric functions --/
def IsTrigonometric (f : ℝ → ℝ) : Prop := sorry

/-- Definition of deductive reasoning --/
def IsDeductiveReasoning (premise1 premise2 conclusion : Prop) : Prop :=
  (premise1 ∧ premise2) → conclusion

/-- The tangent function --/
noncomputable def tan : ℝ → ℝ := sorry

/-- Theorem stating that the reasoning in option D is deductive --/
theorem option_d_is_deductive_reasoning :
  IsDeductiveReasoning
    (∀ f, IsTrigonometric f → IsPeriodic f)
    (IsTrigonometric tan)
    (IsPeriodic tan) :=
sorry

end NUMINAMATH_CALUDE_option_d_is_deductive_reasoning_l529_52967


namespace NUMINAMATH_CALUDE_is_projection_matrix_l529_52918

def projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

theorem is_projection_matrix : 
  let P : Matrix (Fin 2) (Fin 2) ℚ := !![2368/2401, 16/49; 33*2401/2240, 33/49]
  projection_matrix P := by sorry

end NUMINAMATH_CALUDE_is_projection_matrix_l529_52918


namespace NUMINAMATH_CALUDE_binomial_15_4_l529_52989

theorem binomial_15_4 : Nat.choose 15 4 = 1365 := by sorry

end NUMINAMATH_CALUDE_binomial_15_4_l529_52989


namespace NUMINAMATH_CALUDE_parallelogram_vertex_sum_l529_52931

/-- A parallelogram with vertices A, B, C, D in 2D space -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The sum of coordinates of a point -/
def sumCoordinates (p : ℝ × ℝ) : ℝ := p.1 + p.2

/-- Theorem: In parallelogram ABCD with A(-1,2), B(3,-4), C(7,3), and A,C opposite, 
    the sum of coordinates of D is 12 -/
theorem parallelogram_vertex_sum (ABCD : Parallelogram) 
    (hA : ABCD.A = (-1, 2))
    (hB : ABCD.B = (3, -4))
    (hC : ABCD.C = (7, 3))
    (hAC_opposite : ABCD.A = (-ABCD.C.1, -ABCD.C.2)) :
    sumCoordinates ABCD.D = 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_sum_l529_52931


namespace NUMINAMATH_CALUDE_set_A_nonempty_iff_a_negative_l529_52963

theorem set_A_nonempty_iff_a_negative (a : ℝ) :
  (∃ x : ℝ, (Real.sqrt x)^2 ≠ a) ↔ a < 0 := by sorry

end NUMINAMATH_CALUDE_set_A_nonempty_iff_a_negative_l529_52963


namespace NUMINAMATH_CALUDE_leap_year_53_sundays_l529_52970

/-- The number of days in a leap year -/
def leap_year_days : ℕ := 366

/-- The number of complete weeks in a leap year -/
def complete_weeks : ℕ := leap_year_days / 7

/-- The number of extra days beyond complete weeks in a leap year -/
def extra_days : ℕ := leap_year_days % 7

/-- The number of possible combinations for the extra days -/
def extra_day_combinations : ℕ := 7

/-- The number of combinations that include a Sunday -/
def sunday_combinations : ℕ := 2

/-- The probability of a randomly chosen leap year having 53 Sundays -/
def prob_53_sundays : ℚ := sunday_combinations / extra_day_combinations

theorem leap_year_53_sundays : 
  prob_53_sundays = 2 / 7 :=
sorry

end NUMINAMATH_CALUDE_leap_year_53_sundays_l529_52970


namespace NUMINAMATH_CALUDE_unique_two_digit_prime_sum_reverse_l529_52912

def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem unique_two_digit_prime_sum_reverse : 
  ∃! n : ℕ, is_two_digit n ∧ Nat.Prime (n + reverse_digits n) :=
sorry

end NUMINAMATH_CALUDE_unique_two_digit_prime_sum_reverse_l529_52912


namespace NUMINAMATH_CALUDE_fraction_value_l529_52903

theorem fraction_value (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 1/2) (h3 : a > b) :
  a / b = 6 ∨ a / b = -6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l529_52903


namespace NUMINAMATH_CALUDE_weekly_sales_equals_63_l529_52935

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The average number of hamburgers sold per day -/
def avg_daily_sales : ℕ := 9

/-- The total number of hamburgers sold in a week -/
def total_weekly_sales : ℕ := days_in_week * avg_daily_sales

theorem weekly_sales_equals_63 : total_weekly_sales = 63 := by
  sorry

end NUMINAMATH_CALUDE_weekly_sales_equals_63_l529_52935


namespace NUMINAMATH_CALUDE_solve_equation_l529_52900

/-- Given the equation 19(x + y) + 17 = 19(-x + y) - z, where x = 1, prove that z = -55 -/
theorem solve_equation (y : ℝ) : 
  ∃ (z : ℝ), 19 * (1 + y) + 17 = 19 * (-1 + y) - z ∧ z = -55 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l529_52900


namespace NUMINAMATH_CALUDE_mean_correction_l529_52958

def correct_mean (n : ℕ) (original_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n : ℚ) * original_mean - wrong_value + correct_value

theorem mean_correction (n : ℕ) (original_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) 
  (h1 : n = 50)
  (h2 : original_mean = 36)
  (h3 : wrong_value = 23)
  (h4 : correct_value = 45) :
  (correct_mean n original_mean wrong_value correct_value) / n = 36.44 := by
  sorry

end NUMINAMATH_CALUDE_mean_correction_l529_52958


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l529_52954

theorem min_sum_absolute_values :
  (∀ x : ℝ, |x - 3| + |x - 1| + |x + 6| ≥ 9) ∧
  (∃ x : ℝ, |x - 3| + |x - 1| + |x + 6| = 9) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l529_52954


namespace NUMINAMATH_CALUDE_percentage_difference_l529_52987

theorem percentage_difference (p j t : ℝ) 
  (hj : j = 0.75 * p) 
  (ht : t = 0.9375 * p) : 
  (t - j) / t = 0.2 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l529_52987


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l529_52914

theorem girls_to_boys_ratio (total : ℕ) (girl_boy_diff : ℕ) : 
  total = 25 → girl_boy_diff = 3 → 
  ∃ (girls boys : ℕ), 
    girls + boys = total ∧ 
    girls = boys + girl_boy_diff ∧ 
    (girls : ℚ) / (boys : ℚ) = 14 / 11 := by
  sorry

#check girls_to_boys_ratio

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l529_52914


namespace NUMINAMATH_CALUDE_absent_boys_l529_52919

/-- Proves the number of absent boys in a class with given conditions -/
theorem absent_boys (total_students : ℕ) (girls_present : ℕ) : 
  total_students = 250 →
  girls_present = 140 →
  girls_present = 2 * (total_students - (total_students - (girls_present + girls_present / 2))) →
  total_students - (girls_present + girls_present / 2) = 40 :=
by sorry

end NUMINAMATH_CALUDE_absent_boys_l529_52919


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l529_52952

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 9 > 0} = {x : ℝ | x < -3 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l529_52952


namespace NUMINAMATH_CALUDE_wednesday_earnings_l529_52961

/-- Represents the working hours and earnings of Jack and Bob on a particular Wednesday -/
structure WorkDay where
  t : ℝ
  jack_hours : ℝ := t - 2
  jack_rate : ℝ := 3 * t - 2
  bob_hours : ℝ := 1.5 * (t - 2)
  bob_rate : ℝ := (3 * t - 2) - (2 * t - 7)
  tax : ℝ := 10

/-- The theorem stating that t = 19/3 is the only valid solution -/
theorem wednesday_earnings (w : WorkDay) : 
  (w.jack_hours * w.jack_rate - w.tax = w.bob_hours * w.bob_rate - w.tax) ∧ 
  (w.jack_hours > 0) ∧ (w.bob_hours > 0) → 
  w.t = 19/3 := by
  sorry

#check wednesday_earnings

end NUMINAMATH_CALUDE_wednesday_earnings_l529_52961


namespace NUMINAMATH_CALUDE_sum_of_first_four_terms_l529_52966

/-- Geometric sequence with given properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem sum_of_first_four_terms (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 = 9 →
  a 5 = 243 →
  (a 1) + (a 2) + (a 3) + (a 4) = 120 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_four_terms_l529_52966


namespace NUMINAMATH_CALUDE_g_50_eq_zero_l529_52982

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → x * g y + y * g x = g (x * y)

/-- The main theorem stating that g(50) = 0 for any function satisfying the functional equation -/
theorem g_50_eq_zero (g : ℝ → ℝ) (h : FunctionalEquation g) : g 50 = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_50_eq_zero_l529_52982


namespace NUMINAMATH_CALUDE_min_sum_squares_l529_52959

theorem min_sum_squares (x y : ℝ) (h : (x - 1)^2 + y^2 = 16) :
  ∃ (m : ℝ), (∀ (a b : ℝ), (a - 1)^2 + b^2 = 16 → x^2 + y^2 ≤ a^2 + b^2) ∧ m = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l529_52959


namespace NUMINAMATH_CALUDE_amount_second_shop_is_340_l529_52949

/-- The amount spent on books from the second shop -/
def amount_second_shop (books_first : ℕ) (amount_first : ℕ) (books_second : ℕ) (total_books : ℕ) (avg_price : ℕ) : ℕ :=
  total_books * avg_price - amount_first

/-- Theorem: The amount spent on the second shop is 340 -/
theorem amount_second_shop_is_340 :
  amount_second_shop 55 1500 60 115 16 = 340 := by
  sorry

end NUMINAMATH_CALUDE_amount_second_shop_is_340_l529_52949


namespace NUMINAMATH_CALUDE_function_relationship_l529_52972

def f (b c : ℝ) (x : ℝ) : ℝ := x^2 - b*x + c

theorem function_relationship (b c : ℝ) :
  (∀ x, f b c (1 + x) = f b c (1 - x)) →
  f b c 0 = 3 →
  ∀ x, f b c (3^x) ≥ f b c (2^x) := by
  sorry

end NUMINAMATH_CALUDE_function_relationship_l529_52972


namespace NUMINAMATH_CALUDE_sequence_nonpositive_l529_52992

theorem sequence_nonpositive (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0) 
  (hn : a n = 0) 
  (h_convex : ∀ k : ℕ, 1 ≤ k ∧ k < n → a (k-1) - 2*a k + a (k+1) ≥ 0) : 
  ∀ k : ℕ, k ≤ n → a k ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_sequence_nonpositive_l529_52992


namespace NUMINAMATH_CALUDE_equal_perimeter_ratio_l529_52938

/-- Given a square and an equilateral triangle with equal perimeters, 
    the ratio of the triangle's side length to the square's side length is 4/3 -/
theorem equal_perimeter_ratio (s t : ℝ) (hs : s > 0) (ht : t > 0) 
  (h_equal_perimeter : 4 * s = 3 * t) : t / s = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equal_perimeter_ratio_l529_52938


namespace NUMINAMATH_CALUDE_equation_solution_l529_52924

theorem equation_solution : ∀ x : ℚ, 
  (Real.sqrt (6 * x) / Real.sqrt (4 * (x - 1)) = 3) → x = 24 / 23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l529_52924


namespace NUMINAMATH_CALUDE_quadratic_minimum_point_l529_52928

/-- The x-coordinate of the minimum point of a quadratic function f(x) = x^2 - 2px + 4q,
    where p and q are positive real numbers, is p. -/
theorem quadratic_minimum_point (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  let f : ℝ → ℝ := fun x ↦ x^2 - 2*p*x + 4*q
  (∀ x, f p ≤ f x) ∧ (∃ x, f p < f x) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_point_l529_52928


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l529_52939

/-- A line in the 2D plane represented by its slope-intercept form -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- Defines symmetry of two lines with respect to the y-axis -/
def symmetricAboutYAxis (l₁ l₂ : Line) : Prop :=
  ∀ x y, l₁.contains x y ↔ l₂.contains (-x) y

/-- The main theorem -/
theorem symmetric_line_equation (l₁ l₂ : Line) :
  l₁.slope = 3 →
  l₁.contains 1 2 →
  symmetricAboutYAxis l₁ l₂ →
  ∀ x y, l₂.contains x y ↔ 3 * x + y + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l529_52939


namespace NUMINAMATH_CALUDE_auction_bids_l529_52946

theorem auction_bids (initial_price final_price : ℕ) (price_increase : ℕ) (num_bidders : ℕ) :
  initial_price = 15 →
  final_price = 65 →
  price_increase = 5 →
  num_bidders = 2 →
  (final_price - initial_price) / price_increase / num_bidders = 5 :=
by sorry

end NUMINAMATH_CALUDE_auction_bids_l529_52946


namespace NUMINAMATH_CALUDE_z_takes_at_most_two_values_l529_52975

/-- Given two distinct real numbers x and y with absolute values not less than 2,
    prove that z = uv + (uv)⁻¹ can take at most 2 distinct values,
    where u + u⁻¹ = x and v + v⁻¹ = y. -/
theorem z_takes_at_most_two_values (x y : ℝ) (hx : |x| ≥ 2) (hy : |y| ≥ 2) (hxy : x ≠ y) :
  ∃ (z₁ z₂ : ℝ), ∀ (u v : ℝ),
    (u + u⁻¹ = x) → (v + v⁻¹ = y) → (u * v + (u * v)⁻¹ = z₁ ∨ u * v + (u * v)⁻¹ = z₂) :=
by sorry

end NUMINAMATH_CALUDE_z_takes_at_most_two_values_l529_52975


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l529_52953

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| > 2}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- Define the complement of A in the universal set ℝ
def C_UA : Set ℝ := (Set.univ : Set ℝ) \ A

-- State the theorem
theorem complement_A_intersect_B : (C_UA ∩ B) = Set.Ioc 2 3 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l529_52953


namespace NUMINAMATH_CALUDE_triangle_side_length_l529_52940

theorem triangle_side_length (a b c : ℝ) (A : ℝ) : 
  A = Real.pi / 3 →  -- 60 degrees in radians
  a * b = 11 →
  a + b = 7 →
  a > c ∧ c > b →
  c = 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l529_52940


namespace NUMINAMATH_CALUDE_cycles_alignment_min_cycles_alignment_l529_52907

/-- The length of the letter cycle -/
def letter_cycle_length : ℕ := 6

/-- The length of the digit cycle -/
def digit_cycle_length : ℕ := 4

/-- The theorem stating when both cycles will simultaneously return to their original state -/
theorem cycles_alignment (m : ℕ) (h1 : m > 0) (h2 : m % letter_cycle_length = 0) (h3 : m % digit_cycle_length = 0) :
  m ≥ 12 :=
sorry

/-- The theorem stating that 12 is the least number satisfying the conditions -/
theorem min_cycles_alignment :
  12 % letter_cycle_length = 0 ∧ 12 % digit_cycle_length = 0 ∧
  ∀ (k : ℕ), k > 0 → k % letter_cycle_length = 0 → k % digit_cycle_length = 0 → k ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_cycles_alignment_min_cycles_alignment_l529_52907


namespace NUMINAMATH_CALUDE_fraction_division_addition_l529_52917

theorem fraction_division_addition : (5 : ℚ) / 6 / ((9 : ℚ) / 10) + (1 : ℚ) / 15 = (402 : ℚ) / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_addition_l529_52917


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l529_52988

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 - a - 2013 = 0) → (b^2 - b - 2013 = 0) → (a^2 + 2*a + 3*b - 2 = 2014) := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l529_52988


namespace NUMINAMATH_CALUDE_max_product_is_48_l529_52950

def max_product (x y z : ℕ+) : Prop :=
  (x : ℕ) + y + z = 12 ∧
  x ≤ y ∧ y ≤ z ∧
  z ≤ 3 * x ∧
  x * y * z ≤ 48

theorem max_product_is_48 :
  ∀ x y z : ℕ+, max_product x y z → x * y * z = 48 :=
sorry

end NUMINAMATH_CALUDE_max_product_is_48_l529_52950


namespace NUMINAMATH_CALUDE_intersection_when_m_2_B_subset_A_iff_l529_52920

-- Define sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 6}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 3 * m - 1}

-- Part 1: Intersection when m = 2
theorem intersection_when_m_2 : A ∩ B 2 = {x | 3 ≤ x ∧ x ≤ 5} := by
  sorry

-- Part 2: Condition for B to be a subset of A
theorem B_subset_A_iff (m : ℝ) : B m ⊆ A ↔ m ≤ 7/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_when_m_2_B_subset_A_iff_l529_52920


namespace NUMINAMATH_CALUDE_profit_sharing_ratio_l529_52981

/-- Represents the investment and time period for a partner --/
structure Partner where
  investment : ℕ
  months : ℕ

/-- Calculates the effective capital of a partner --/
def effectiveCapital (p : Partner) : ℕ := p.investment * p.months

/-- Calculates the ratio of two numbers --/
def ratio (a b : ℕ) : ℕ × ℕ :=
  let gcd := a.gcd b
  (a / gcd, b / gcd)

/-- Theorem stating the profit sharing ratio between P and Q --/
theorem profit_sharing_ratio (p q : Partner)
  (h1 : p.investment = 4000)
  (h2 : p.months = 12)
  (h3 : q.investment = 9000)
  (h4 : q.months = 8) :
  ratio (effectiveCapital p) (effectiveCapital q) = (2, 3) := by
  sorry

#check profit_sharing_ratio

end NUMINAMATH_CALUDE_profit_sharing_ratio_l529_52981


namespace NUMINAMATH_CALUDE_uncertain_roots_l529_52990

/-- Given that mx² - 2(m+2)x + m + 5 = 0 has no real roots, 
    prove that the number of real roots of (m-5)x² - 2(m+2)x + m = 0 is uncertain. -/
theorem uncertain_roots (m : ℝ) 
  (h : ∀ x : ℝ, m * x^2 - 2*(m+2)*x + m + 5 ≠ 0) : 
  ∃ m₁ m₂ : ℝ, 
    (∃! x : ℝ, (m₁-5) * x^2 - 2*(m₁+2)*x + m₁ = 0) ∧ 
    (∃ x y : ℝ, x ≠ y ∧ (m₂-5) * x^2 - 2*(m₂+2)*x + m₂ = 0 ∧ (m₂-5) * y^2 - 2*(m₂+2)*y + m₂ = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_uncertain_roots_l529_52990


namespace NUMINAMATH_CALUDE_sarah_sells_more_than_tamara_l529_52969

/-- Represents the bake sale competition between Tamara and Sarah -/
structure BakeSale where
  -- Tamara's baked goods
  tamara_brownie_pans : ℕ
  tamara_cookie_trays : ℕ
  tamara_brownie_pieces_per_pan : ℕ
  tamara_cookie_pieces_per_tray : ℕ
  tamara_small_brownie_price : ℚ
  tamara_large_brownie_price : ℚ
  tamara_cookie_price : ℚ
  tamara_small_brownies_sold : ℕ

  -- Sarah's baked goods
  sarah_cupcake_batches : ℕ
  sarah_muffin_dozens : ℕ
  sarah_cupcakes_per_batch : ℕ
  sarah_chocolate_cupcake_price : ℚ
  sarah_vanilla_cupcake_price : ℚ
  sarah_strawberry_cupcake_price : ℚ
  sarah_muffin_price : ℚ
  sarah_chocolate_cupcakes_sold : ℕ
  sarah_vanilla_cupcakes_sold : ℕ

/-- Calculates the total sales for Tamara -/
def tamara_total_sales (bs : BakeSale) : ℚ :=
  let total_brownies := bs.tamara_brownie_pans * bs.tamara_brownie_pieces_per_pan
  let large_brownies_sold := total_brownies - bs.tamara_small_brownies_sold
  let total_cookies := bs.tamara_cookie_trays * bs.tamara_cookie_pieces_per_tray
  bs.tamara_small_brownies_sold * bs.tamara_small_brownie_price +
  large_brownies_sold * bs.tamara_large_brownie_price +
  total_cookies * bs.tamara_cookie_price

/-- Calculates the total sales for Sarah -/
def sarah_total_sales (bs : BakeSale) : ℚ :=
  let total_cupcakes := bs.sarah_cupcake_batches * bs.sarah_cupcakes_per_batch
  let strawberry_cupcakes_sold := total_cupcakes - bs.sarah_chocolate_cupcakes_sold - bs.sarah_vanilla_cupcakes_sold
  let total_muffins := bs.sarah_muffin_dozens * 12
  total_muffins * bs.sarah_muffin_price +
  bs.sarah_chocolate_cupcakes_sold * bs.sarah_chocolate_cupcake_price +
  bs.sarah_vanilla_cupcakes_sold * bs.sarah_vanilla_cupcake_price +
  strawberry_cupcakes_sold * bs.sarah_strawberry_cupcake_price

/-- Theorem stating the difference in sales between Sarah and Tamara -/
theorem sarah_sells_more_than_tamara (bs : BakeSale) :
  bs.tamara_brownie_pans = 2 ∧
  bs.tamara_cookie_trays = 3 ∧
  bs.tamara_brownie_pieces_per_pan = 8 ∧
  bs.tamara_cookie_pieces_per_tray = 12 ∧
  bs.tamara_small_brownie_price = 2 ∧
  bs.tamara_large_brownie_price = 3 ∧
  bs.tamara_cookie_price = 3/2 ∧
  bs.tamara_small_brownies_sold = 4 ∧
  bs.sarah_cupcake_batches = 3 ∧
  bs.sarah_muffin_dozens = 2 ∧
  bs.sarah_cupcakes_per_batch = 10 ∧
  bs.sarah_chocolate_cupcake_price = 5/2 ∧
  bs.sarah_vanilla_cupcake_price = 2 ∧
  bs.sarah_strawberry_cupcake_price = 11/4 ∧
  bs.sarah_muffin_price = 7/4 ∧
  bs.sarah_chocolate_cupcakes_sold = 7 ∧
  bs.sarah_vanilla_cupcakes_sold = 8 →
  sarah_total_sales bs - tamara_total_sales bs = 75/4 := by
  sorry

end NUMINAMATH_CALUDE_sarah_sells_more_than_tamara_l529_52969


namespace NUMINAMATH_CALUDE_circle_area_through_DEF_l529_52904

-- Define the triangle DEF
def triangle_DEF (D E F : ℝ × ℝ) : Prop :=
  let d_e := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let d_f := Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)
  d_e = d_f ∧ d_e = 5 * Real.sqrt 3

-- Define the tangent circle
def tangent_circle (D E F : ℝ × ℝ) (G : ℝ × ℝ) : Prop :=
  let g_e := Real.sqrt ((E.1 - G.1)^2 + (E.2 - G.2)^2)
  let g_f := Real.sqrt ((F.1 - G.1)^2 + (F.2 - G.2)^2)
  g_e = 6 ∧ g_f = 6

-- Define the altitude condition
def altitude_condition (D E F : ℝ × ℝ) (G : ℝ × ℝ) : Prop :=
  let m_ef := (F.2 - E.2) / (F.1 - E.1)
  let m_dg := (G.2 - D.2) / (G.1 - D.1)
  m_ef * m_dg = -1

-- Theorem statement
theorem circle_area_through_DEF 
  (D E F : ℝ × ℝ) 
  (G : ℝ × ℝ) 
  (h1 : triangle_DEF D E F) 
  (h2 : tangent_circle D E F G) 
  (h3 : altitude_condition D E F G) :
  let R := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) / 2
  Real.pi * R^2 = 36 * Real.pi := by sorry

end NUMINAMATH_CALUDE_circle_area_through_DEF_l529_52904


namespace NUMINAMATH_CALUDE_constant_if_average_property_l529_52996

/-- A function from ℤ² to ℕ -/
def GridFunction := ℤ × ℤ → ℕ

/-- The property that f(x, y) is the average of its four neighbors -/
def HasAverageProperty (f : GridFunction) : Prop :=
  ∀ x y : ℤ, f (x, y) = (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1)) / 4

/-- Main theorem: if f has the average property, then it is constant -/
theorem constant_if_average_property (f : GridFunction) (h : HasAverageProperty f) :
  ∃ c : ℕ, ∀ x y : ℤ, f (x, y) = c := by
  sorry

end NUMINAMATH_CALUDE_constant_if_average_property_l529_52996


namespace NUMINAMATH_CALUDE_power_sum_division_equals_seventeen_l529_52945

theorem power_sum_division_equals_seventeen :
  1^234 + 4^6 / 4^4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_division_equals_seventeen_l529_52945


namespace NUMINAMATH_CALUDE_larger_number_proof_l529_52976

/-- Given two positive integers with HCF 23 and LCM factors 13 and 14, prove the larger number is 322 -/
theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 23) → 
  (∃ (k : ℕ+), Nat.lcm a b = 23 * 13 * 14 * k) → 
  (max a b = 322) := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l529_52976


namespace NUMINAMATH_CALUDE_hall_width_proof_l529_52957

/-- Given a rectangular hall with specified dimensions and cost constraints, 
    prove that the width of the hall is 17 meters. -/
theorem hall_width_proof (length height : ℝ) (cost_per_sqm total_cost : ℝ) :
  length = 20 →
  height = 5 →
  cost_per_sqm = 60 →
  total_cost = 57000 →
  ∃ w : ℝ, (2 * length * w + 2 * length * height + 2 * w * height) * cost_per_sqm = total_cost ∧ w = 17 :=
by sorry

end NUMINAMATH_CALUDE_hall_width_proof_l529_52957


namespace NUMINAMATH_CALUDE_license_plate_count_l529_52978

/-- The number of consonants available for the license plate. -/
def num_consonants : ℕ := 20

/-- The number of vowels available for the license plate. -/
def num_vowels : ℕ := 6

/-- The number of digits available for the license plate. -/
def num_digits : ℕ := 10

/-- The number of special symbols available for the license plate. -/
def num_special_symbols : ℕ := 2

/-- The total number of possible license plates. -/
def total_license_plates : ℕ := num_consonants * num_vowels * num_consonants * num_digits * num_special_symbols

/-- Theorem stating that the total number of license plates is 48,000. -/
theorem license_plate_count : total_license_plates = 48000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l529_52978


namespace NUMINAMATH_CALUDE_peanut_mixture_proof_l529_52947

/-- Given the following:
    - 10 pounds of Virginia peanuts cost $3.50 per pound
    - Spanish peanuts cost $3.00 per pound
    - The desired mixture should cost $3.40 per pound
    Prove that 2.5 pounds of Spanish peanuts should be used to create the mixture. -/
theorem peanut_mixture_proof (virginia_weight : ℝ) (virginia_price : ℝ) (spanish_price : ℝ) 
  (mixture_price : ℝ) (spanish_weight : ℝ) :
  virginia_weight = 10 →
  virginia_price = 3.5 →
  spanish_price = 3 →
  mixture_price = 3.4 →
  spanish_weight = 2.5 →
  (virginia_weight * virginia_price + spanish_weight * spanish_price) / (virginia_weight + spanish_weight) = mixture_price :=
by sorry

end NUMINAMATH_CALUDE_peanut_mixture_proof_l529_52947


namespace NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l529_52913

/-- Converts a base 8 number to decimal --/
def base8ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 8 --/
def decimalToBase8 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 3-digit base 8 number --/
def isThreeDigitBase8 (n : ℕ) : Prop := 
  100 ≤ n ∧ n ≤ 777

theorem greatest_3digit_base8_divisible_by_7 :
  ∃ (n : ℕ), isThreeDigitBase8 n ∧ 
             n % 7 = 0 ∧
             base8ToDecimal n = 511 ∧
             decimalToBase8 511 = 777 ∧
             ∀ (m : ℕ), isThreeDigitBase8 m ∧ m % 7 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l529_52913


namespace NUMINAMATH_CALUDE_correct_operation_l529_52995

theorem correct_operation (x y : ℝ) : 4 * x^3 * y^2 * (x^2 * y^3) = 4 * x^5 * y^5 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l529_52995


namespace NUMINAMATH_CALUDE_quadratic_no_rational_solution_l529_52984

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluation of a quadratic polynomial at a point x -/
def evaluate (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Statement: For any quadratic polynomial with real coefficients, 
    there exists a natural number n such that p(x) = 1/n has no rational solutions -/
theorem quadratic_no_rational_solution (p : QuadraticPolynomial) :
  ∃ n : ℕ, ∀ x : ℚ, evaluate p x ≠ 1 / n := by sorry

end NUMINAMATH_CALUDE_quadratic_no_rational_solution_l529_52984


namespace NUMINAMATH_CALUDE_unique_real_root_of_equation_l529_52911

theorem unique_real_root_of_equation :
  ∃! x : ℝ, 2 * Real.sqrt (x - 3) + 6 = x :=
by sorry

end NUMINAMATH_CALUDE_unique_real_root_of_equation_l529_52911


namespace NUMINAMATH_CALUDE_marie_erasers_l529_52955

def initial_erasers : ℕ := 95
def lost_erasers : ℕ := 42

theorem marie_erasers : initial_erasers - lost_erasers = 53 := by
  sorry

end NUMINAMATH_CALUDE_marie_erasers_l529_52955


namespace NUMINAMATH_CALUDE_min_max_sum_x_l529_52909

theorem min_max_sum_x (x y z : ℝ) 
  (sum_eq : x + y + z = 6)
  (sum_sq_eq : x^2 + y^2 + z^2 = 14) :
  ∃ (m M : ℝ), (∀ t, m ≤ t ∧ t ≤ M → ∃ u v, t + u + v = 6 ∧ t^2 + u^2 + v^2 = 14) ∧
                (∀ s, (∃ u v, s + u + v = 6 ∧ s^2 + u^2 + v^2 = 14) → m ≤ s ∧ s ≤ M) ∧
                m + M = 10/3 :=
sorry

end NUMINAMATH_CALUDE_min_max_sum_x_l529_52909


namespace NUMINAMATH_CALUDE_clinic_cats_count_l529_52951

theorem clinic_cats_count (dog_cost cat_cost dog_count total_cost : ℕ) 
  (h1 : dog_cost = 60)
  (h2 : cat_cost = 40)
  (h3 : dog_count = 20)
  (h4 : total_cost = 3600)
  : ∃ cat_count : ℕ, dog_cost * dog_count + cat_cost * cat_count = total_cost ∧ cat_count = 60 := by
  sorry

end NUMINAMATH_CALUDE_clinic_cats_count_l529_52951


namespace NUMINAMATH_CALUDE_determinant_special_matrix_l529_52942

open Matrix

theorem determinant_special_matrix (x : ℝ) :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![x + 2, x, x; x, x + 2, x; x, x, x + 2]
  det A = 16 * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_determinant_special_matrix_l529_52942
