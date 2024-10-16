import Mathlib

namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l387_38730

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem fifth_term_of_geometric_sequence (a : ℕ → ℝ) :
  IsGeometricSequence a → a 3 = 18 → a 4 = 24 → a 5 = 32 := by
  sorry


end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l387_38730


namespace NUMINAMATH_CALUDE_base_8_digit_count_l387_38785

/-- The count of numbers among the first 512 positive integers in base 8 
    that contain either 5 or 6 -/
def count_with_5_or_6 : ℕ := 296

/-- The count of numbers among the first 512 positive integers in base 8 
    that don't contain 5 or 6 -/
def count_without_5_or_6 : ℕ := 6^3

/-- The total count of numbers considered -/
def total_count : ℕ := 512

theorem base_8_digit_count : 
  count_with_5_or_6 = total_count - count_without_5_or_6 := by sorry

end NUMINAMATH_CALUDE_base_8_digit_count_l387_38785


namespace NUMINAMATH_CALUDE_negative_cube_squared_l387_38793

theorem negative_cube_squared (a : ℝ) : -(-3*a)^2 = -9*a^2 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l387_38793


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l387_38741

theorem arithmetic_calculation : (4 + 4 + 6) / 3 - 2 / 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l387_38741


namespace NUMINAMATH_CALUDE_pascal_triangle_count_l387_38751

/-- Represents a row in Pascal's Triangle -/
def PascalRow := List Nat

/-- Generates the nth row of Pascal's Triangle -/
def generatePascalRow (n : Nat) : PascalRow :=
  sorry

/-- Counts the number of even integers in a given row -/
def countEvens (row : PascalRow) : Nat :=
  sorry

/-- Counts the number of integers that are multiples of 4 in a given row -/
def countMultiplesOfFour (row : PascalRow) : Nat :=
  sorry

/-- Theorem stating the count of even integers and multiples of 4 in the first 12 rows of Pascal's Triangle -/
theorem pascal_triangle_count :
  let rows := List.range 12
  let evenCount := rows.map (fun n => countEvens (generatePascalRow n)) |>.sum
  let multiples4Count := rows.map (fun n => countMultiplesOfFour (generatePascalRow n)) |>.sum
  ∃ (e m : Nat), evenCount = e ∧ multiples4Count = m :=
by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_count_l387_38751


namespace NUMINAMATH_CALUDE_cannot_reach_target_l387_38794

/-- Represents a positive integer as a list of digits (most significant digit first) -/
def Digits := List Nat

/-- The starting number -/
def startNum : Digits := [1]

/-- The target 100-digit number -/
def targetNum : Digits := List.replicate 98 2 ++ [5, 2, 2, 2, 1]

/-- Checks if a number is valid (non-zero first digit) -/
def isValidNumber (d : Digits) : Prop := d.head? ≠ some 0

/-- Represents the operation of multiplying by 5 -/
def multiplyBy5 (d : Digits) : Digits := sorry

/-- Represents the operation of rearranging digits -/
def rearrangeDigits (d : Digits) : Digits := sorry

/-- Represents a sequence of operations -/
inductive Operation
| Multiply
| Rearrange

def applyOperation (op : Operation) (d : Digits) : Digits :=
  match op with
  | Operation.Multiply => multiplyBy5 d
  | Operation.Rearrange => rearrangeDigits d

/-- Theorem stating the impossibility of reaching the target number -/
theorem cannot_reach_target : 
  ∀ (ops : List Operation), 
    let finalNum := ops.foldl (λ acc op => applyOperation op acc) startNum
    isValidNumber finalNum → finalNum ≠ targetNum :=
by sorry

end NUMINAMATH_CALUDE_cannot_reach_target_l387_38794


namespace NUMINAMATH_CALUDE_wedge_volume_l387_38713

/-- The volume of a wedge that represents one-third of a cylindrical cheese log -/
theorem wedge_volume (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cylinder_volume := π * r^2 * h
  let wedge_volume := (1/3) * cylinder_volume
  h = 8 ∧ r = 5 → wedge_volume = (200 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_l387_38713


namespace NUMINAMATH_CALUDE_cube_volume_from_total_edge_length_l387_38782

/-- The volume of a cube with total edge length of 72 feet is 216 cubic feet. -/
theorem cube_volume_from_total_edge_length :
  ∀ (s : ℝ), (12 * s = 72) → s^3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_total_edge_length_l387_38782


namespace NUMINAMATH_CALUDE_infinitely_many_non_representable_l387_38791

theorem infinitely_many_non_representable : 
  Set.Infinite {n : ℤ | ∀ (a b c : ℕ), n ≠ 2^a + 3^b - 5^c} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_non_representable_l387_38791


namespace NUMINAMATH_CALUDE_percent_problem_l387_38768

theorem percent_problem (x : ℝ) : 
  (30 / 100 * 100 = 50 / 100 * x + 10) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l387_38768


namespace NUMINAMATH_CALUDE_housing_boom_construction_l387_38708

/-- The number of houses in Lawrence County before the housing boom -/
def houses_before : ℕ := 1426

/-- The number of houses in Lawrence County after the housing boom -/
def houses_after : ℕ := 2000

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := houses_after - houses_before

theorem housing_boom_construction : houses_built = 574 := by
  sorry

end NUMINAMATH_CALUDE_housing_boom_construction_l387_38708


namespace NUMINAMATH_CALUDE_sphere_intersection_ratio_l387_38750

/-- Two spheres with radii R₁ and R₂ are intersected by a plane P perpendicular to the line
    connecting their centers and passing through its midpoint. If P divides the surface area
    of the first sphere in ratio m:1 and the second sphere in ratio n:1 (where m > 1 and n > 1),
    then R₂/R₁ = ((m - 1)(n + 1)) / ((m + 1)(n - 1)). -/
theorem sphere_intersection_ratio (R₁ R₂ m n : ℝ) (hm : m > 1) (hn : n > 1) :
  let h₁ := (2 * R₁) / (m + 1)
  let h₂ := (2 * R₂) / (n + 1)
  R₁ - h₁ = R₂ - h₂ →
  R₂ / R₁ = ((m - 1) * (n + 1)) / ((m + 1) * (n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_sphere_intersection_ratio_l387_38750


namespace NUMINAMATH_CALUDE_lions_in_first_group_l387_38778

/-- The killing rate of lions in deers per minute -/
def killing_rate (lions : ℕ) (deers : ℕ) (minutes : ℕ) : ℚ :=
  (deers : ℚ) / (lions : ℚ) / (minutes : ℚ)

/-- The number of lions in the first group -/
def first_group_lions : ℕ := 10

theorem lions_in_first_group :
  (killing_rate first_group_lions 10 10 = killing_rate 100 100 10) →
  first_group_lions = 10 := by
  sorry

end NUMINAMATH_CALUDE_lions_in_first_group_l387_38778


namespace NUMINAMATH_CALUDE_license_plate_count_l387_38763

/-- The number of vowels (excluding Y) -/
def num_vowels : Nat := 5

/-- The number of digits between 1 and 5 -/
def num_digits : Nat := 5

/-- The number of consonants (including Y) -/
def num_consonants : Nat := 26 - num_vowels

/-- The total number of license plates meeting the specified criteria -/
def total_plates : Nat := num_vowels * num_digits * num_consonants * num_consonants * num_vowels

theorem license_plate_count : total_plates = 55125 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l387_38763


namespace NUMINAMATH_CALUDE_alexs_score_l387_38732

theorem alexs_score (total_students : ℕ) (average_without_alex : ℚ) (average_with_alex : ℚ) :
  total_students = 20 →
  average_without_alex = 75 →
  average_with_alex = 76 →
  (total_students - 1) * average_without_alex + 95 = total_students * average_with_alex :=
by sorry

end NUMINAMATH_CALUDE_alexs_score_l387_38732


namespace NUMINAMATH_CALUDE_f_min_max_values_g_negative_range_l387_38736

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 + Real.log x

def g (a : ℝ) (x : ℝ) : ℝ := f a x - 2 * a * x

-- Define the interval [1/e, e]
def interval : Set ℝ := Set.Icc (1 / Real.exp 1) (Real.exp 1)

-- Theorem 1: Minimum and maximum values of f when a = -1/2
theorem f_min_max_values :
  let f_neg_half (x : ℝ) := f (-1/2) x
  (∀ x ∈ interval, f_neg_half x ≥ 1 - Real.exp 1 ^ 2) ∧
  (∃ x ∈ interval, f_neg_half x = 1 - Real.exp 1 ^ 2) ∧
  (∀ x ∈ interval, f_neg_half x ≤ -1/2 - 1/2 * Real.log 2) ∧
  (∃ x ∈ interval, f_neg_half x = -1/2 - 1/2 * Real.log 2) := by
  sorry

-- Theorem 2: Range of a for which g(x) < 0 holds for all x > 2
theorem g_negative_range :
  {a : ℝ | ∀ x > 2, g a x < 0} = Set.Iic (1/2) := by
  sorry

end

end NUMINAMATH_CALUDE_f_min_max_values_g_negative_range_l387_38736


namespace NUMINAMATH_CALUDE_inequality_proof_l387_38733

theorem inequality_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  |b / a - b / c| + |c / a - c / b| + |b * c + 1| > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l387_38733


namespace NUMINAMATH_CALUDE_initial_men_is_ten_l387_38738

/-- The initial number of men in the camp -/
def initial_men : ℕ := sorry

/-- The number of days the food lasts for the initial number of men -/
def initial_days : ℕ := 20

/-- The number of additional men that join the camp -/
def additional_men : ℕ := 30

/-- The number of days the food lasts after additional men join -/
def final_days : ℕ := 5

/-- The total amount of food available -/
def total_food : ℕ := initial_men * initial_days

/-- Theorem stating that the initial number of men is 10 -/
theorem initial_men_is_ten : initial_men = 10 := by
  have h1 : total_food = (initial_men + additional_men) * final_days := sorry
  sorry

end NUMINAMATH_CALUDE_initial_men_is_ten_l387_38738


namespace NUMINAMATH_CALUDE_coefficient_a3_equals_80_l387_38731

theorem coefficient_a3_equals_80 :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) (x : ℝ),
  (2 * x^2 + 1)^5 = a₀ + a₁ * x^2 + a₂ * x^4 + a₃ * x^6 + a₄ * x^8 + a₅ * x^10 →
  a₃ = 80 := by
sorry

end NUMINAMATH_CALUDE_coefficient_a3_equals_80_l387_38731


namespace NUMINAMATH_CALUDE_marker_cost_l387_38783

theorem marker_cost (total_students : Nat) (buyers : Nat) (markers_per_student : Nat) (total_cost : Nat) :
  total_students = 40 →
  buyers > total_students / 2 →
  buyers ≤ total_students →
  markers_per_student % 2 = 0 →
  markers_per_student > 2 →
  total_cost = 3185 →
  ∃ (cost_per_marker : Nat),
    cost_per_marker > markers_per_student ∧
    buyers * markers_per_student * cost_per_marker = total_cost ∧
    cost_per_marker = 13 :=
by sorry

end NUMINAMATH_CALUDE_marker_cost_l387_38783


namespace NUMINAMATH_CALUDE_marble_probability_l387_38711

theorem marble_probability (total : ℕ) (p_white p_red_or_blue : ℚ) :
  total = 90 →
  p_white = 1/3 →
  p_red_or_blue = 7/15 →
  ∃ (white red blue green : ℕ),
    white + red + blue + green = total ∧
    p_white = white / total ∧
    p_red_or_blue = (red + blue) / total ∧
    green / total = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_marble_probability_l387_38711


namespace NUMINAMATH_CALUDE_parabola_maximum_l387_38703

/-- The quadratic function f(x) = -x^2 - 1 -/
def f (x : ℝ) : ℝ := -x^2 - 1

theorem parabola_maximum :
  (∀ x : ℝ, f x ≤ f 0) ∧ f 0 = -1 :=
sorry

end NUMINAMATH_CALUDE_parabola_maximum_l387_38703


namespace NUMINAMATH_CALUDE_bottles_produced_l387_38792

-- Define the production rate of 4 machines
def production_rate_4 : ℕ := 16

-- Define the number of minutes
def minutes : ℕ := 3

-- Define the number of machines in the first scenario
def machines_1 : ℕ := 4

-- Define the number of machines in the second scenario
def machines_2 : ℕ := 8

-- Theorem to prove
theorem bottles_produced :
  (machines_2 * minutes * (production_rate_4 / machines_1)) = 96 := by
  sorry

end NUMINAMATH_CALUDE_bottles_produced_l387_38792


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l387_38795

theorem opposite_of_negative_five : -((-5) : ℝ) = (5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l387_38795


namespace NUMINAMATH_CALUDE_triangle_exists_and_satisfies_inequality_l387_38725

/-- Theorem: Existence of a triangle with sides 9, 15, and 21 satisfying the triangle inequality. -/
theorem triangle_exists_and_satisfies_inequality : ∃ (a b c : ℝ),
  a = 9 ∧ b = 15 ∧ c = 21 ∧
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧
  (∃ (x : ℝ), a = 9 ∧ b = x + 6 ∧ c = 2*x + 3 ∧ a + b + c = 45) :=
by sorry

end NUMINAMATH_CALUDE_triangle_exists_and_satisfies_inequality_l387_38725


namespace NUMINAMATH_CALUDE_point_on_line_coordinates_l387_38754

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line passing through two points in 3D space -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- Function to get a point on a line given an x-coordinate -/
def pointOnLine (l : Line3D) (x : ℝ) : Point3D :=
  sorry

theorem point_on_line_coordinates (l : Line3D) :
  l.p1 = ⟨1, 3, 4⟩ →
  l.p2 = ⟨4, 2, 1⟩ →
  let p := pointOnLine l 7
  p.y = 1 ∧ p.z = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_coordinates_l387_38754


namespace NUMINAMATH_CALUDE_p_and_q_implies_m_leq_1_l387_38749

/-- Proposition p: For all x ∈ ℝ, the function y = log₂(2ˣ - m + 1) is defined. -/
def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, 2^x - m + 1 > 0

/-- Proposition q: The function f(x) = (5 - 2m)ˣ is increasing. -/
def proposition_q (m : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (5 - 2*m)^x < (5 - 2*m)^y

/-- If propositions p and q are true, then m ≤ 1. -/
theorem p_and_q_implies_m_leq_1 (m : ℝ) :
  proposition_p m ∧ proposition_q m → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_p_and_q_implies_m_leq_1_l387_38749


namespace NUMINAMATH_CALUDE_georgesBirthdayMoneyIs12_l387_38737

/-- Calculates the amount George will receive on his 25th birthday --/
def georgesBirthdayMoney (currentAge : ℕ) (startAge : ℕ) (spendPercentage : ℚ) (exchangeRate : ℚ) : ℚ :=
  let totalBills : ℕ := currentAge - startAge
  let remainingBills : ℚ := (1 - spendPercentage) * totalBills
  exchangeRate * remainingBills

/-- Theorem stating the amount George will receive --/
theorem georgesBirthdayMoneyIs12 : 
  georgesBirthdayMoney 25 15 (1/5) (3/2) = 12 := by
  sorry


end NUMINAMATH_CALUDE_georgesBirthdayMoneyIs12_l387_38737


namespace NUMINAMATH_CALUDE_wire_length_problem_l387_38772

theorem wire_length_problem (total_wires : ℕ) (avg_length : ℝ) (third_avg_length : ℝ) :
  total_wires = 6 →
  avg_length = 80 →
  third_avg_length = 70 →
  let total_length := total_wires * avg_length
  let third_wires := total_wires / 3
  let third_total_length := third_wires * third_avg_length
  let remaining_wires := total_wires - third_wires
  let remaining_length := total_length - third_total_length
  remaining_length / remaining_wires = 85 := by
sorry

end NUMINAMATH_CALUDE_wire_length_problem_l387_38772


namespace NUMINAMATH_CALUDE_initial_depth_calculation_l387_38745

theorem initial_depth_calculation (men_initial : ℕ) (hours_initial : ℕ) (men_extra : ℕ) (hours_final : ℕ) (depth_final : ℕ) :
  men_initial = 75 →
  hours_initial = 8 →
  men_extra = 65 →
  hours_final = 6 →
  depth_final = 70 →
  ∃ (depth_initial : ℕ), 
    (men_initial * hours_initial * depth_final = (men_initial + men_extra) * hours_final * depth_initial) ∧
    depth_initial = 50 := by
  sorry

#check initial_depth_calculation

end NUMINAMATH_CALUDE_initial_depth_calculation_l387_38745


namespace NUMINAMATH_CALUDE_cos420_plus_sin330_eq_zero_l387_38773

theorem cos420_plus_sin330_eq_zero :
  Real.cos (420 * π / 180) + Real.sin (330 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos420_plus_sin330_eq_zero_l387_38773


namespace NUMINAMATH_CALUDE_parabola_properties_l387_38779

-- Define the parabola
def parabola (x : ℝ) : ℝ := (x - 4)^2 - 5

-- Theorem statement
theorem parabola_properties :
  (∃ (x y : ℝ), y = parabola x ∧ ∀ (x' : ℝ), parabola x' ≥ y) ∧
  (∀ (x₁ x₂ : ℝ), x₁ < 4 ∧ x₂ > 4 → parabola x₁ > parabola 4 ∧ parabola x₂ > parabola 4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l387_38779


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_9_with_digit_sum_27_l387_38720

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem smallest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ (n : ℕ), is_three_digit n ∧ n % 9 = 0 ∧ digit_sum n = 27 ∧
  ∀ (m : ℕ), is_three_digit m ∧ m % 9 = 0 ∧ digit_sum m = 27 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_9_with_digit_sum_27_l387_38720


namespace NUMINAMATH_CALUDE_gcf_72_108_l387_38775

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by sorry

end NUMINAMATH_CALUDE_gcf_72_108_l387_38775


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l387_38764

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) :
  3 * x^2 * y + 12 * x^2 * y^2 + 12 * x * y^3 = 3 * x * y * (x + 4 * x * y + 4 * y^2) := by
  sorry

-- Problem 2
theorem factorization_problem_2 (a b : ℝ) :
  2 * a^5 * b - 2 * a * b^5 = 2 * a * b * (a^2 + b^2) * (a + b) * (a - b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l387_38764


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l387_38771

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m * n > 0) (h2 : m + n = 2) :
  (1 / m + 1 / n) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l387_38771


namespace NUMINAMATH_CALUDE_eighth_odd_multiple_of_five_l387_38714

def is_odd_multiple_of_five (n : ℕ) : Prop := n % 2 = 1 ∧ n % 5 = 0

def nth_odd_multiple_of_five (n : ℕ) : ℕ :=
  (2 * n - 1) * 5

theorem eighth_odd_multiple_of_five :
  nth_odd_multiple_of_five 8 = 75 ∧ is_odd_multiple_of_five (nth_odd_multiple_of_five 8) :=
sorry

end NUMINAMATH_CALUDE_eighth_odd_multiple_of_five_l387_38714


namespace NUMINAMATH_CALUDE_carls_garden_area_l387_38796

/-- Represents a rectangular garden with fence posts --/
structure Garden where
  total_posts : ℕ
  post_separation : ℕ
  shorter_side_posts : ℕ

/-- Calculates the area of the garden --/
def garden_area (g : Garden) : ℕ :=
  let longer_side_posts := 2 * g.shorter_side_posts
  let shorter_side_length := (g.shorter_side_posts - 1) * g.post_separation
  let longer_side_length := (longer_side_posts - 1) * g.post_separation
  shorter_side_length * longer_side_length

/-- Theorem stating that Carl's garden has an area of 900 square yards --/
theorem carls_garden_area :
  ∀ (g : Garden),
    g.total_posts = 26 ∧
    g.post_separation = 5 ∧
    g.shorter_side_posts = 5 →
    garden_area g = 900 := by
  sorry

end NUMINAMATH_CALUDE_carls_garden_area_l387_38796


namespace NUMINAMATH_CALUDE_ali_baba_walk_possible_l387_38710

/-- Represents a cell in the cave -/
structure Cell where
  row : Nat
  col : Nat
  isBlack : Bool

/-- Represents the state of the cave -/
structure CaveState where
  m : Nat
  n : Nat
  coins : Cell → Nat

/-- Represents a move in the cave -/
inductive Move
  | up
  | down
  | left
  | right

/-- Predicate to check if a move is valid -/
def isValidMove (state : CaveState) (pos : Cell) (move : Move) : Prop :=
  match move with
  | Move.up => pos.row > 0
  | Move.down => pos.row < state.m - 1
  | Move.left => pos.col > 0
  | Move.right => pos.col < state.n - 1

/-- Function to apply a move and update the cave state -/
def applyMove (state : CaveState) (pos : Cell) (move : Move) : CaveState :=
  sorry

/-- Predicate to check if the final state is correct -/
def isCorrectFinalState (state : CaveState) : Prop :=
  ∀ cell, (cell.isBlack → state.coins cell = 1) ∧ (¬cell.isBlack → state.coins cell = 0)

/-- Theorem stating that Ali Baba's walk is possible -/
theorem ali_baba_walk_possible (m n : Nat) :
  ∃ (initialState : CaveState) (moves : List Move),
    initialState.m = m ∧
    initialState.n = n ∧
    (∀ cell, initialState.coins cell = 0) ∧
    isCorrectFinalState (moves.foldl (λ s m => applyMove s (sorry) m) initialState) :=
  sorry

end NUMINAMATH_CALUDE_ali_baba_walk_possible_l387_38710


namespace NUMINAMATH_CALUDE_sum_of_ages_is_32_l387_38787

/-- Viggo's age when his brother was 2 years old -/
def viggos_initial_age : ℕ := 2 * 2 + 10

/-- The current age of Viggo's younger brother -/
def brothers_current_age : ℕ := 10

/-- The number of years that have passed since the initial condition -/
def years_passed : ℕ := brothers_current_age - 2

/-- Viggo's current age -/
def viggos_current_age : ℕ := viggos_initial_age + years_passed

/-- The sum of Viggo's and his younger brother's current ages -/
def sum_of_ages : ℕ := viggos_current_age + brothers_current_age

theorem sum_of_ages_is_32 : sum_of_ages = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_is_32_l387_38787


namespace NUMINAMATH_CALUDE_base4_division_l387_38744

/-- Convert a number from base 4 to decimal --/
def base4ToDecimal (n : List Nat) : Nat :=
  n.foldr (fun digit acc => acc * 4 + digit) 0

/-- Convert a number from decimal to base 4 --/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Theorem: 12345₄ divided by 23₄ equals 535₄ in base 4 --/
theorem base4_division :
  let dividend := base4ToDecimal [1, 2, 3, 4, 5]
  let divisor := base4ToDecimal [2, 3]
  let quotient := base4ToDecimal [5, 3, 5]
  decimalToBase4 (dividend / divisor) = [5, 3, 5] :=
by sorry

end NUMINAMATH_CALUDE_base4_division_l387_38744


namespace NUMINAMATH_CALUDE_intersection_points_parabola_and_circle_l387_38777

theorem intersection_points_parabola_and_circle (A : ℝ) (h : A > 0) :
  ∃! (points : Finset (ℝ × ℝ)), points.card = 4 ∧
    ∀ (x y : ℝ), (x, y) ∈ points ↔ 
      (y = A * x^2 ∧ y^2 + 5 = x^2 + 6 * y) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_parabola_and_circle_l387_38777


namespace NUMINAMATH_CALUDE_exp_greater_than_log_squared_l387_38799

open Real

theorem exp_greater_than_log_squared (x : ℝ) (h : x > 0) : exp x - exp 2 * log x > 0 := by
  sorry

end NUMINAMATH_CALUDE_exp_greater_than_log_squared_l387_38799


namespace NUMINAMATH_CALUDE_polynomial_inequality_l387_38797

theorem polynomial_inequality (x : ℝ) : 
  x^4 - 4*x^3 + 8*x^2 - 8*x ≤ 96 → -2 ≤ x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l387_38797


namespace NUMINAMATH_CALUDE_jersey_t_shirt_price_difference_l387_38798

/-- The price difference between a jersey and a t-shirt -/
def price_difference (jersey_profit t_shirt_profit : ℕ) : ℕ :=
  jersey_profit - t_shirt_profit

/-- Theorem stating that the price difference between a jersey and a t-shirt is $90 -/
theorem jersey_t_shirt_price_difference :
  price_difference 115 25 = 90 := by
  sorry

end NUMINAMATH_CALUDE_jersey_t_shirt_price_difference_l387_38798


namespace NUMINAMATH_CALUDE_geometric_progression_integers_l387_38742

/-- A geometric progression with first term b and common ratio r -/
def GeometricProgression (b : ℤ) (r : ℚ) : ℕ → ℚ :=
  fun n => b * r ^ (n - 1)

/-- An arithmetic progression with first term a and common difference d -/
def ArithmeticProgression (a d : ℚ) : ℕ → ℚ :=
  fun n => a + (n - 1) * d

theorem geometric_progression_integers
  (b : ℤ) (r : ℚ) (a d : ℚ)
  (h_subset : ∀ n : ℕ, ∃ m : ℕ, GeometricProgression b r n = ArithmeticProgression a d m) :
  ∀ n : ℕ, ∃ k : ℤ, GeometricProgression b r n = k :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_integers_l387_38742


namespace NUMINAMATH_CALUDE_circle_equation_l387_38748

/-- Represents a parabola in the form y^2 = 4x -/
def Parabola := { p : ℝ × ℝ | p.2^2 = 4 * p.1 }

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- The directrix of the parabola y^2 = 4x -/
def directrix : ℝ → ℝ := fun x ↦ -1

/-- Represents a circle with center (h, k) and radius r -/
def Circle (h k r : ℝ) := { p : ℝ × ℝ | (p.1 - h)^2 + (p.2 - k)^2 = r^2 }

/-- The theorem stating that the circle with the focus as its center and tangent to the directrix
    has the equation (x - 1)^2 + y^2 = 4 -/
theorem circle_equation : 
  ∃ (c : Set (ℝ × ℝ)), c = Circle focus.1 focus.2 2 ∧ 
  (∀ p ∈ c, p.1 ≠ -1) ∧
  (∃ p ∈ c, p.1 = -1) ∧
  c = { p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 4 } :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l387_38748


namespace NUMINAMATH_CALUDE_weight_loss_probability_is_0_241_l387_38790

/-- The probability of a person losing weight after taking a drug, given the total number of volunteers and the number of people who lost weight. -/
def probability_of_weight_loss (total_volunteers : ℕ) (weight_loss_count : ℕ) : ℚ :=
  weight_loss_count / total_volunteers

/-- Theorem stating that the probability of weight loss is 0.241 given the provided data. -/
theorem weight_loss_probability_is_0_241 
  (total_volunteers : ℕ) 
  (weight_loss_count : ℕ) 
  (h1 : total_volunteers = 1000) 
  (h2 : weight_loss_count = 241) : 
  probability_of_weight_loss total_volunteers weight_loss_count = 241 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_probability_is_0_241_l387_38790


namespace NUMINAMATH_CALUDE_probability_two_female_volunteers_l387_38740

/-- The probability of selecting 2 female volunteers from a group of 3 female and 2 male volunteers (5 in total) is 3/10. -/
theorem probability_two_female_volunteers :
  let total_volunteers : ℕ := 5
  let female_volunteers : ℕ := 3
  let male_volunteers : ℕ := 2
  let selected_volunteers : ℕ := 2
  let total_combinations := Nat.choose total_volunteers selected_volunteers
  let female_combinations := Nat.choose female_volunteers selected_volunteers
  (female_combinations : ℚ) / total_combinations = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_female_volunteers_l387_38740


namespace NUMINAMATH_CALUDE_triangle_inequality_l387_38756

theorem triangle_inequality (a b c S r R : ℝ) (ha hb hc : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < S ∧ 0 < r ∧ 0 < R →
  9 * r ≤ ha + hb + hc →
  ha + hb + hc ≤ 9 * R / 2 →
  1 / a + 1 / b + 1 / c = (ha + hb + hc) / (2 * S) →
  9 * r / (2 * S) ≤ 1 / a + 1 / b + 1 / c ∧ 1 / a + 1 / b + 1 / c ≤ 9 * R / (4 * S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l387_38756


namespace NUMINAMATH_CALUDE_exam_question_distribution_l387_38707

theorem exam_question_distribution :
  ∃ (P M E : ℕ),
    P + M + E = 50 ∧
    P ≥ 39 ∧ P ≤ 41 ∧
    M ≥ 7 ∧ M ≤ 8 ∧
    E ≥ 2 ∧ E ≤ 3 ∧
    P = 40 ∧ M = 7 ∧ E = 3 :=
by sorry

end NUMINAMATH_CALUDE_exam_question_distribution_l387_38707


namespace NUMINAMATH_CALUDE_triangle_area_ratio_l387_38776

theorem triangle_area_ratio (k : ℕ) (H : ℝ) (h : ℝ) :
  k > 0 →
  H > 0 →
  h > 0 →
  h / H = 1 / Real.sqrt k →
  (h / H)^2 = 1 / k :=
by sorry


end NUMINAMATH_CALUDE_triangle_area_ratio_l387_38776


namespace NUMINAMATH_CALUDE_emma_walk_distance_l387_38752

theorem emma_walk_distance
  (total_time : ℝ)
  (bike_speed : ℝ)
  (walk_speed : ℝ)
  (bike_fraction : ℝ)
  (walk_fraction : ℝ)
  (h_total_time : total_time = 1)
  (h_bike_speed : bike_speed = 20)
  (h_walk_speed : walk_speed = 6)
  (h_bike_fraction : bike_fraction = 1/3)
  (h_walk_fraction : walk_fraction = 2/3)
  (h_fractions : bike_fraction + walk_fraction = 1) :
  let total_distance := (bike_speed * bike_fraction + walk_speed * walk_fraction) * total_time
  let walk_distance := total_distance * walk_fraction
  ∃ (ε : ℝ), abs (walk_distance - 5.2) < ε ∧ ε < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_emma_walk_distance_l387_38752


namespace NUMINAMATH_CALUDE_no_rational_solution_l387_38762

theorem no_rational_solution :
  ∀ (a b c d : ℚ), (a + b * Real.sqrt 3)^4 + (c + d * Real.sqrt 3)^4 ≠ 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l387_38762


namespace NUMINAMATH_CALUDE_find_n_l387_38726

theorem find_n (e n : ℕ+) (h1 : Nat.lcm e n = 690) 
  (h2 : ¬ 3 ∣ n) (h3 : ¬ 2 ∣ e) : n = 230 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l387_38726


namespace NUMINAMATH_CALUDE_gloria_turtle_time_l387_38704

/-- The time it took for Gloria's turtle to finish the race -/
def glorias_time (gretas_time georges_time : ℕ) : ℕ :=
  2 * georges_time

theorem gloria_turtle_time : ∃ (gretas_time georges_time : ℕ),
  gretas_time = 6 ∧
  georges_time = gretas_time - 2 ∧
  glorias_time gretas_time georges_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_gloria_turtle_time_l387_38704


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l387_38786

theorem reciprocal_of_negative_2023 : 
  ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l387_38786


namespace NUMINAMATH_CALUDE_min_crystals_to_kill_120_l387_38743

structure Skill where
  name : String
  crystalCost : ℕ
  damage : ℕ
  specialEffect : Bool

def applySkill (health : ℕ) (skill : Skill) (prevWindUsed : Bool) : ℕ × ℕ :=
  let actualCost := if prevWindUsed then skill.crystalCost / 2 else skill.crystalCost
  let newHealth := 
    if skill.name = "Earth" then
      if health % 2 = 1 then (health + 1) / 2 else health / 2
    else
      if health > skill.damage then health - skill.damage else 0
  (newHealth, actualCost)

def minCrystalsToKill (initialHealth : ℕ) (water fire wind earth : Skill) : ℕ :=
  sorry

theorem min_crystals_to_kill_120 :
  let water : Skill := ⟨"Water", 4, 4, false⟩
  let fire : Skill := ⟨"Fire", 10, 11, false⟩
  let wind : Skill := ⟨"Wind", 10, 5, true⟩
  let earth : Skill := ⟨"Earth", 18, 0, false⟩
  minCrystalsToKill 120 water fire wind earth = 68 := by
  sorry

end NUMINAMATH_CALUDE_min_crystals_to_kill_120_l387_38743


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l387_38723

theorem least_positive_integer_with_remainders : ∃ b : ℕ, 
  b > 0 ∧ 
  b % 2 = 1 ∧ 
  b % 5 = 2 ∧ 
  b % 7 = 3 ∧ 
  ∀ c : ℕ, c > 0 ∧ c % 2 = 1 ∧ c % 5 = 2 ∧ c % 7 = 3 → b ≤ c :=
by
  use 17
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l387_38723


namespace NUMINAMATH_CALUDE_students_liking_sports_l387_38781

theorem students_liking_sports (total : ℕ) (basketball cricket soccer : ℕ)
  (basketball_cricket basketball_soccer cricket_soccer : ℕ) (all_three : ℕ)
  (h1 : total = 30)
  (h2 : basketball = 12)
  (h3 : cricket = 10)
  (h4 : soccer = 8)
  (h5 : basketball_cricket = 4)
  (h6 : basketball_soccer = 3)
  (h7 : cricket_soccer = 2)
  (h8 : all_three = 1) :
  basketball + cricket + soccer - (basketball_cricket + basketball_soccer + cricket_soccer) + all_three = 22 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_sports_l387_38781


namespace NUMINAMATH_CALUDE_abs_neg_two_eq_two_l387_38715

theorem abs_neg_two_eq_two : |(-2 : ℝ)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_eq_two_l387_38715


namespace NUMINAMATH_CALUDE_expected_deliveries_l387_38788

theorem expected_deliveries (packages_yesterday : ℕ) (success_rate : ℚ) :
  packages_yesterday = 80 →
  success_rate = 90 / 100 →
  (packages_yesterday * 2 : ℚ) * success_rate = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_expected_deliveries_l387_38788


namespace NUMINAMATH_CALUDE_prime_ap_difference_greater_than_30000_l387_38706

/-- An arithmetic progression of prime numbers -/
structure PrimeArithmeticProgression where
  terms : Fin 15 → ℕ
  is_prime : ∀ i, Nat.Prime (terms i)
  is_increasing : ∀ i j, i < j → terms i < terms j
  is_arithmetic : ∀ i j k, terms j - terms i = terms k - terms j ↔ j - i = k - j

/-- The common difference of an arithmetic progression -/
def common_difference (ap : PrimeArithmeticProgression) : ℕ :=
  ap.terms 1 - ap.terms 0

/-- Theorem: The common difference of an arithmetic progression of 15 primes is greater than 30000 -/
theorem prime_ap_difference_greater_than_30000 (ap : PrimeArithmeticProgression) :
  common_difference ap > 30000 := by
  sorry

end NUMINAMATH_CALUDE_prime_ap_difference_greater_than_30000_l387_38706


namespace NUMINAMATH_CALUDE_min_investment_optimal_quantities_l387_38734

/-- Represents the cost and quantity of stationery types A and B -/
structure Stationery where
  cost_A : ℕ
  cost_B : ℕ
  quantity_A : ℕ
  quantity_B : ℕ

/-- Defines the conditions of the stationery purchase problem -/
def stationery_problem (s : Stationery) : Prop :=
  s.cost_A * 2 + s.cost_B = 35 ∧
  s.cost_A + s.cost_B * 3 = 30 ∧
  s.quantity_A + s.quantity_B = 120 ∧
  975 ≤ s.cost_A * s.quantity_A + s.cost_B * s.quantity_B ∧
  s.cost_A * s.quantity_A + s.cost_B * s.quantity_B ≤ 1000

/-- Theorem stating the minimum investment for the stationery purchase -/
theorem min_investment (s : Stationery) :
  stationery_problem s →
  s.cost_A * s.quantity_A + s.cost_B * s.quantity_B ≥ 980 :=
by sorry

/-- Theorem stating the optimal purchase quantities -/
theorem optimal_quantities (s : Stationery) :
  stationery_problem s →
  s.cost_A * s.quantity_A + s.cost_B * s.quantity_B = 980 →
  s.quantity_A = 38 ∧ s.quantity_B = 82 :=
by sorry

end NUMINAMATH_CALUDE_min_investment_optimal_quantities_l387_38734


namespace NUMINAMATH_CALUDE_speed_of_X_is_60_l387_38709

-- Define the speed of person Y
def speed_Y : ℝ := 60

-- Define the time difference between X and Y's start
def time_difference : ℝ := 3

-- Define the distance ahead
def distance_ahead : ℝ := 30

-- Define the time difference between Y catching up to X and X catching up to Y
def catch_up_time_difference : ℝ := 3

-- Define the speed of person X
def speed_X : ℝ := 60

-- Theorem statement
theorem speed_of_X_is_60 :
  ∀ (t₁ t₂ : ℝ),
  t₂ - t₁ = catch_up_time_difference →
  speed_X * (time_difference + t₁) = speed_Y * t₁ + distance_ahead →
  speed_X * (time_difference + t₂) + distance_ahead = speed_Y * t₂ →
  speed_X = speed_Y :=
by sorry

end NUMINAMATH_CALUDE_speed_of_X_is_60_l387_38709


namespace NUMINAMATH_CALUDE_range_of_a_l387_38774

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ((a - 5) * x > a - 5) ↔ x < 1) → a < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l387_38774


namespace NUMINAMATH_CALUDE_ap_terms_count_l387_38757

theorem ap_terms_count (n : ℕ) (a d : ℝ) : 
  n % 2 = 0 ∧ 
  n > 0 ∧
  (n / 2 : ℝ) * (2 * a + (n - 2) * d) = 30 ∧ 
  (n / 2 : ℝ) * (2 * a + n * d) = 36 ∧ 
  a + (n - 1) * d - a = 15 → 
  n = 6 := by sorry

end NUMINAMATH_CALUDE_ap_terms_count_l387_38757


namespace NUMINAMATH_CALUDE_total_age_reaches_target_in_10_years_l387_38702

/-- Represents the number of years between each sibling's birth -/
def age_gap : ℕ := 5

/-- Represents the current age of the eldest sibling -/
def eldest_current_age : ℕ := 20

/-- Represents the target total age of all siblings -/
def target_total_age : ℕ := 75

/-- Calculates the total age of the siblings after a given number of years -/
def total_age_after (years : ℕ) : ℕ :=
  (eldest_current_age + years) + 
  (eldest_current_age - age_gap + years) + 
  (eldest_current_age - 2 * age_gap + years)

/-- Theorem stating that it takes 10 years for the total age to reach the target -/
theorem total_age_reaches_target_in_10_years : 
  total_age_after 10 = target_total_age :=
sorry

end NUMINAMATH_CALUDE_total_age_reaches_target_in_10_years_l387_38702


namespace NUMINAMATH_CALUDE_union_complement_problem_l387_38755

theorem union_complement_problem (U A B : Set Nat) :
  U = {1, 2, 3, 4} →
  A = {1, 2} →
  B = {2, 3} →
  A ∪ (U \ B) = {1, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_problem_l387_38755


namespace NUMINAMATH_CALUDE_max_video_game_hours_l387_38760

/-- Proves that given the conditions of Max's video game playing schedule,
    he must have played 2 hours on Wednesday. -/
theorem max_video_game_hours :
  ∀ x : ℝ,
  (x + x + (x + 3)) / 3 = 3 →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_max_video_game_hours_l387_38760


namespace NUMINAMATH_CALUDE_employee_salaries_calculation_l387_38758

/-- Given a total revenue and a ratio for division between employee salaries and stock purchases,
    calculate the amount spent on employee salaries. -/
def calculate_employee_salaries (total_revenue : ℚ) (salary_ratio stock_ratio : ℕ) : ℚ :=
  (salary_ratio : ℚ) / ((salary_ratio : ℚ) + (stock_ratio : ℚ)) * total_revenue

/-- Theorem stating that given a total revenue of 3000 and a division ratio of 4:11
    for employee salaries to stock purchases, the amount spent on employee salaries is 800. -/
theorem employee_salaries_calculation :
  calculate_employee_salaries 3000 4 11 = 800 := by
  sorry

#eval calculate_employee_salaries 3000 4 11

end NUMINAMATH_CALUDE_employee_salaries_calculation_l387_38758


namespace NUMINAMATH_CALUDE_log_inequality_condition_l387_38735

theorem log_inequality_condition (x y : ℝ) :
  (∀ x y, x > 0 ∧ y > 0 ∧ Real.log x > Real.log y → x > y) ∧
  ¬(∀ x y, x > y → Real.log x > Real.log y) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_condition_l387_38735


namespace NUMINAMATH_CALUDE_set_A_nonempty_iff_a_negative_l387_38747

theorem set_A_nonempty_iff_a_negative (a : ℝ) :
  (∃ x : ℝ, (Real.sqrt x)^2 ≠ a) ↔ a < 0 := by sorry

end NUMINAMATH_CALUDE_set_A_nonempty_iff_a_negative_l387_38747


namespace NUMINAMATH_CALUDE_quartic_polynomial_sum_l387_38705

/-- A quartic polynomial with specific properties -/
structure QuarticPolynomial (k : ℝ) where
  P : ℝ → ℝ
  is_quartic : ∃ (a b c d : ℝ), ∀ x, P x = a * x^4 + b * x^3 + c * x^2 + d * x + k
  at_zero : P 0 = k
  at_one : P 1 = 3 * k
  at_neg_one : P (-1) = 4 * k

/-- The sum of the polynomial evaluated at 2 and -2 equals 82k -/
theorem quartic_polynomial_sum (k : ℝ) (p : QuarticPolynomial k) :
  p.P 2 + p.P (-2) = 82 * k := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_sum_l387_38705


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l387_38765

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a4 : a 4 = 9)
  (h_sum : a 3 + a 7 = 20) :
  ∃ d : ℝ, d = 1 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l387_38765


namespace NUMINAMATH_CALUDE_moores_law_2010_l387_38753

def transistor_count (year : ℕ) : ℕ :=
  if year ≤ 2000 then
    2000000 * 2^((year - 1992) / 2)
  else
    2000000 * 2^4 * 4^((year - 2000) / 2)

theorem moores_law_2010 : transistor_count 2010 = 32768000000 := by
  sorry

end NUMINAMATH_CALUDE_moores_law_2010_l387_38753


namespace NUMINAMATH_CALUDE_chocolate_distribution_l387_38712

theorem chocolate_distribution (num_students : ℕ) (num_choices : ℕ) 
  (h1 : num_students = 211) (h2 : num_choices = 35) : 
  ∃ (group_size : ℕ), group_size ≥ 7 ∧ 
  (∀ (group : ℕ), group ≤ group_size) ∧ 
  (num_students ≤ group_size * num_choices) :=
sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l387_38712


namespace NUMINAMATH_CALUDE_gold_alloy_composition_l387_38719

/-- Proves that adding 24 ounces of pure gold to a 16-ounce alloy that is 50% gold
    will result in an alloy that is 80% gold. -/
theorem gold_alloy_composition (initial_weight : ℝ) (initial_purity : ℝ) 
    (added_gold : ℝ) (final_purity : ℝ) : 
  initial_weight = 16 →
  initial_purity = 0.5 →
  added_gold = 24 →
  final_purity = 0.8 →
  (initial_weight * initial_purity + added_gold) / (initial_weight + added_gold) = final_purity :=
by
  sorry

#check gold_alloy_composition

end NUMINAMATH_CALUDE_gold_alloy_composition_l387_38719


namespace NUMINAMATH_CALUDE_curve_self_intersection_l387_38701

/-- The x-coordinate of a point on the curve as a function of t -/
def x (t : ℝ) : ℝ := t^3 - 3*t + 1

/-- The y-coordinate of a point on the curve as a function of t -/
def y (t : ℝ) : ℝ := t^4 - 4*t^2 + 4

/-- The curve crosses itself at (1, 1) -/
theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧ x a = x b ∧ y a = y b ∧ x a = 1 ∧ y a = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l387_38701


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l387_38716

def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {-3, 1, 2, 4}

theorem intersection_of_A_and_B : A ∩ B = {-3, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l387_38716


namespace NUMINAMATH_CALUDE_functional_equation_proof_l387_38769

/-- Given a function f: ℝ → ℝ satisfying the functional equation
    f(x + y) = f(x) * f(y) for all real x and y, and f(3) = 4,
    prove that f(9) = 64. -/
theorem functional_equation_proof (f : ℝ → ℝ) 
    (h1 : ∀ x y : ℝ, f (x + y) = f x * f y) 
    (h2 : f 3 = 4) : 
  f 9 = 64 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_proof_l387_38769


namespace NUMINAMATH_CALUDE_sufficient_condition_exclusive_or_condition_l387_38717

-- Define propositions p and q
def p (x : ℝ) : Prop := (x + 1) * (x - 5) ≤ 0
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0

-- Part 1: p is a sufficient condition for q
theorem sufficient_condition (m : ℝ) :
  (∀ x, p x → q x m) → m ∈ Set.Ici 4 :=
sorry

-- Part 2: m = 5, "p or q" is true, "p and q" is false
theorem exclusive_or_condition (x : ℝ) :
  (p x ∨ q x 5) ∧ ¬(p x ∧ q x 5) → x ∈ Set.Icc (-4) (-1) ∪ Set.Ioc 5 6 :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_exclusive_or_condition_l387_38717


namespace NUMINAMATH_CALUDE_similar_triangles_height_l387_38780

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 →
  area_ratio = 9 →
  let h_large := h_small * Real.sqrt area_ratio
  h_small = 5 →
  h_large = 15 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l387_38780


namespace NUMINAMATH_CALUDE_overall_average_score_problem_solution_l387_38724

/-- Calculates the overall average score of two classes -/
theorem overall_average_score 
  (n1 : ℕ) (n2 : ℕ) (avg1 : ℝ) (avg2 : ℝ) : 
  (n1 : ℝ) * avg1 + (n2 : ℝ) * avg2 = ((n1 + n2) : ℝ) * ((n1 * avg1 + n2 * avg2) / (n1 + n2)) :=
by sorry

/-- Proves that the overall average score for the given problem is 74 -/
theorem problem_solution 
  (n1 : ℕ) (n2 : ℕ) (avg1 : ℝ) (avg2 : ℝ) 
  (h1 : n1 = 20) (h2 : n2 = 30) (h3 : avg1 = 80) (h4 : avg2 = 70) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 74 :=
by sorry

end NUMINAMATH_CALUDE_overall_average_score_problem_solution_l387_38724


namespace NUMINAMATH_CALUDE_max_sum_of_digits_is_24_max_sum_of_digits_is_achievable_l387_38770

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours < 24
  minute_valid : minutes < 60

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a given time -/
def timeSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum sum of digits possible in a 24-hour format digital watch display -/
def maxSumOfDigits : Nat := 24

/-- Theorem stating that the maximum sum of digits in a 24-hour format digital watch display is 24 -/
theorem max_sum_of_digits_is_24 :
  ∀ t : Time24, timeSumOfDigits t ≤ maxSumOfDigits :=
sorry

/-- Theorem stating that there exists a time that achieves the maximum sum of digits -/
theorem max_sum_of_digits_is_achievable :
  ∃ t : Time24, timeSumOfDigits t = maxSumOfDigits :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_is_24_max_sum_of_digits_is_achievable_l387_38770


namespace NUMINAMATH_CALUDE_students_liking_sports_l387_38789

theorem students_liking_sports (B C : Finset Nat) : 
  (B.card = 9) → 
  (C.card = 8) → 
  ((B ∩ C).card = 6) → 
  ((B ∪ C).card = 11) := by
sorry

end NUMINAMATH_CALUDE_students_liking_sports_l387_38789


namespace NUMINAMATH_CALUDE_hyperbola_condition_l387_38729

/-- Represents a hyperbola equation with parameter m -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), x^2 / a - y^2 / b = 1 ↔ x^2 / (m - 10) - y^2 / (m - 8) = 1

/-- m > 10 is a sufficient but not necessary condition for the equation to represent a hyperbola -/
theorem hyperbola_condition (m : ℝ) :
  (m > 10 → is_hyperbola m) ∧ (∃ m₀ : ℝ, m₀ ≤ 10 ∧ is_hyperbola m₀) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l387_38729


namespace NUMINAMATH_CALUDE_problem_statement_l387_38727

theorem problem_statement : (36 / (7 + 2 - 5)) * 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l387_38727


namespace NUMINAMATH_CALUDE_function_inequality_l387_38759

theorem function_inequality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) 
  (h2 : ∀ x > 0, deriv f x + f x / x > 0) :
  ∀ a b, a > 0 → b > 0 → a > b → a * f a > b * f b := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l387_38759


namespace NUMINAMATH_CALUDE_new_average_weight_l387_38700

theorem new_average_weight 
  (initial_students : ℕ) 
  (initial_average : ℚ) 
  (new_student_weight : ℚ) : 
  initial_students = 29 →
  initial_average = 28 →
  new_student_weight = 13 →
  (initial_students * initial_average + new_student_weight) / (initial_students + 1) = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l387_38700


namespace NUMINAMATH_CALUDE_neighborhood_vehicles_l387_38739

theorem neighborhood_vehicles (total : Nat) (both : Nat) (car : Nat) (bike_only : Nat)
  (h1 : total = 90)
  (h2 : both = 16)
  (h3 : car = 44)
  (h4 : bike_only = 35) :
  total - (car + bike_only) = 11 := by
  sorry

end NUMINAMATH_CALUDE_neighborhood_vehicles_l387_38739


namespace NUMINAMATH_CALUDE_fifteen_sided_polygon_diagonals_l387_38728

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 15 sides has 90 diagonals -/
theorem fifteen_sided_polygon_diagonals :
  num_diagonals 15 = 90 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_sided_polygon_diagonals_l387_38728


namespace NUMINAMATH_CALUDE_area_is_nine_halves_l387_38721

/-- The line in the Cartesian coordinate system -/
def line (x y : ℝ) : Prop := x - y = 0

/-- The curve in the Cartesian coordinate system -/
def curve (x y : ℝ) : Prop := y = x^2 - 2*x

/-- The area enclosed by the line and the curve -/
noncomputable def enclosed_area : ℝ := sorry

/-- Theorem stating that the enclosed area is equal to 9/2 -/
theorem area_is_nine_halves : enclosed_area = 9/2 := by sorry

end NUMINAMATH_CALUDE_area_is_nine_halves_l387_38721


namespace NUMINAMATH_CALUDE_mary_seth_age_ratio_l387_38761

/-- Given that Mary is 9 years older than Seth and Seth is currently 3.5 years old,
    prove that the ratio of Mary's age to Seth's age in a year is 3:1. -/
theorem mary_seth_age_ratio :
  ∀ (seth_age mary_age seth_future_age mary_future_age : ℝ),
  seth_age = 3.5 →
  mary_age = seth_age + 9 →
  seth_future_age = seth_age + 1 →
  mary_future_age = mary_age + 1 →
  mary_future_age / seth_future_age = 3 := by
sorry

end NUMINAMATH_CALUDE_mary_seth_age_ratio_l387_38761


namespace NUMINAMATH_CALUDE_min_sum_and_inequality_l387_38722

theorem min_sum_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3 * a * b) :
  (∃ (min : ℝ), min = 4/3 ∧ ∀ x y, x > 0 → y > 0 → x + y = 3 * x * y → x + y ≥ min) ∧
  (a / b + b / a ≥ 8 / (9 * a * b)) := by
sorry

end NUMINAMATH_CALUDE_min_sum_and_inequality_l387_38722


namespace NUMINAMATH_CALUDE_last_eight_digits_of_product_l387_38784

def product : ℕ := 11 * 101 * 1001 * 10001 * 1000001 * 111

theorem last_eight_digits_of_product : product % 100000000 = 87654321 := by
  sorry

end NUMINAMATH_CALUDE_last_eight_digits_of_product_l387_38784


namespace NUMINAMATH_CALUDE_no_solution_for_prime_factor_conditions_l387_38767

/-- P(n) denotes the greatest prime factor of n -/
def greatest_prime_factor (n : ℕ) : ℕ :=
  sorry

theorem no_solution_for_prime_factor_conditions : 
  ∀ n : ℕ, n > 1 → 
  ¬(greatest_prime_factor n = Real.sqrt n ∧ 
    greatest_prime_factor (n + 54) = Real.sqrt (n + 54)) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_prime_factor_conditions_l387_38767


namespace NUMINAMATH_CALUDE_abs_equation_solution_l387_38766

theorem abs_equation_solution : ∃! x : ℚ, |x - 3| = |x - 4| := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_l387_38766


namespace NUMINAMATH_CALUDE_slope_angle_45_implies_a_equals_3_l387_38718

theorem slope_angle_45_implies_a_equals_3 (a : ℝ) :
  (∃ (x y : ℝ), (a - 2) * x - y + 3 = 0 ∧ 
   Real.arctan ((a - 2) : ℝ) = π / 4) →
  a = 3 := by sorry

end NUMINAMATH_CALUDE_slope_angle_45_implies_a_equals_3_l387_38718


namespace NUMINAMATH_CALUDE_diophantine_equation_solvability_l387_38746

theorem diophantine_equation_solvability (m : ℤ) :
  ∃ (k : ℕ+) (a b c d : ℕ+), a * b - c * d = m := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solvability_l387_38746
