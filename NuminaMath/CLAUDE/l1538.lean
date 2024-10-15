import Mathlib

namespace NUMINAMATH_CALUDE_divisibility_equivalence_l1538_153843

theorem divisibility_equivalence (a b : ℤ) :
  (29 ∣ 3*a + 2*b) ↔ (29 ∣ 11*a + 17*b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l1538_153843


namespace NUMINAMATH_CALUDE_isabella_escalator_time_l1538_153833

/-- Represents the time it takes Isabella to ride an escalator under different conditions -/
def EscalatorTime (walk_time_stopped : ℝ) (walk_time_moving : ℝ) : Prop :=
  ∃ (escalator_speed : ℝ) (isabella_speed : ℝ),
    escalator_speed > 0 ∧
    isabella_speed > 0 ∧
    walk_time_stopped * isabella_speed = walk_time_moving * (isabella_speed + escalator_speed) ∧
    walk_time_stopped / escalator_speed = 45

theorem isabella_escalator_time :
  EscalatorTime 90 30 :=
sorry

end NUMINAMATH_CALUDE_isabella_escalator_time_l1538_153833


namespace NUMINAMATH_CALUDE_abe_age_sum_l1538_153827

/-- The sum of Abe's present age and his age 7 years ago is 31, given that Abe's present age is 19. -/
theorem abe_age_sum : 
  let present_age : ℕ := 19
  let years_ago : ℕ := 7
  present_age + (present_age - years_ago) = 31 := by sorry

end NUMINAMATH_CALUDE_abe_age_sum_l1538_153827


namespace NUMINAMATH_CALUDE_multiplication_fraction_product_l1538_153874

theorem multiplication_fraction_product : 11 * (1 / 17) * 34 = 22 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_fraction_product_l1538_153874


namespace NUMINAMATH_CALUDE_problem_solution_l1538_153844

def smallest_positive_integer : ℕ := 1

def opposite_is_self (b : ℤ) : Prop := -b = b

def largest_negative_integer : ℤ := -1

theorem problem_solution (a b c : ℤ) 
  (ha : a = smallest_positive_integer)
  (hb : opposite_is_self b)
  (hc : c = largest_negative_integer + 3) :
  (2*a + 3*c) * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1538_153844


namespace NUMINAMATH_CALUDE_computer_usage_difference_l1538_153860

/-- The difference in computer usage between two weeks -/
def usage_difference (last_week : ℕ) (this_week_daily : ℕ) : ℕ :=
  last_week - (this_week_daily * 7)

/-- Theorem stating the difference in computer usage -/
theorem computer_usage_difference :
  usage_difference 91 8 = 35 := by
  sorry

end NUMINAMATH_CALUDE_computer_usage_difference_l1538_153860


namespace NUMINAMATH_CALUDE_circle_condition_l1538_153822

/-- The equation x^2 + y^2 + ax - ay + 2 = 0 represents a circle if and only if a > 2 or a < -2 -/
theorem circle_condition (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + a*x - a*y + 2 = 0 ∧ 
   ∀ x' y' : ℝ, x'^2 + y'^2 + a*x' - a*y' + 2 = 0 → (x' - x)^2 + (y' - y)^2 = ((x' - x)^2 + (y' - y)^2)) 
  ↔ (a > 2 ∨ a < -2) :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l1538_153822


namespace NUMINAMATH_CALUDE_inequality_proof_l1538_153893

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hsum : a + b + c = 1) :
  a^(a^2 + 2*c*a) * b^(b^2 + 2*a*b) * c^(c^2 + 2*b*c) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1538_153893


namespace NUMINAMATH_CALUDE_building_height_l1538_153872

/-- Given a flagpole and a building casting shadows under similar conditions,
    calculate the height of the building. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagpole : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 65)
  : (flagpole_height * building_shadow) / flagpole_shadow = 26 := by
  sorry

#check building_height

end NUMINAMATH_CALUDE_building_height_l1538_153872


namespace NUMINAMATH_CALUDE_remainder_plus_three_l1538_153846

/-- f(x) represents the remainder of x divided by 3 -/
def f (x : ℕ) : ℕ := x % 3

/-- For all natural numbers x, f(x+3) = f(x) -/
theorem remainder_plus_three (x : ℕ) : f (x + 3) = f x := by
  sorry

end NUMINAMATH_CALUDE_remainder_plus_three_l1538_153846


namespace NUMINAMATH_CALUDE_square_of_binomial_l1538_153847

theorem square_of_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 4*x^2 + 12*x + a = (2*x + b)^2) → a = 9 :=
by sorry

end NUMINAMATH_CALUDE_square_of_binomial_l1538_153847


namespace NUMINAMATH_CALUDE_digit_product_sum_l1538_153858

/-- A function that checks if a number is a three-digit number with all digits the same -/
def isTripleDigit (n : Nat) : Prop :=
  ∃ d, d ∈ Finset.range 10 ∧ n = d * 100 + d * 10 + d

/-- A function that converts a two-digit number to its decimal representation -/
def twoDigitToDecimal (a b : Nat) : Nat := 10 * a + b

theorem digit_product_sum : 
  ∃ (A B C D E : Nat), 
    A ∈ Finset.range 10 ∧ 
    B ∈ Finset.range 10 ∧ 
    C ∈ Finset.range 10 ∧ 
    D ∈ Finset.range 10 ∧ 
    E ∈ Finset.range 10 ∧ 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (twoDigitToDecimal A B) * (twoDigitToDecimal C D) = E * 100 + E * 10 + E ∧
    A + B + C + D + E = 21 :=
sorry

end NUMINAMATH_CALUDE_digit_product_sum_l1538_153858


namespace NUMINAMATH_CALUDE_louisa_first_day_travel_l1538_153809

/-- Represents Louisa's travel details -/
structure LouisaTravel where
  first_day_miles : ℝ
  second_day_miles : ℝ
  average_speed : ℝ
  time_difference : ℝ

/-- Theorem stating that Louisa traveled 200 miles on the first day -/
theorem louisa_first_day_travel (t : LouisaTravel) 
  (h1 : t.second_day_miles = 350)
  (h2 : t.average_speed = 50)
  (h3 : t.time_difference = 3)
  (h4 : t.second_day_miles / t.average_speed = t.first_day_miles / t.average_speed + t.time_difference) :
  t.first_day_miles = 200 := by
  sorry

#check louisa_first_day_travel

end NUMINAMATH_CALUDE_louisa_first_day_travel_l1538_153809


namespace NUMINAMATH_CALUDE_pushup_difference_l1538_153810

-- Define the number of push-ups for each person
def zachary_pushups : ℕ := 51
def john_pushups : ℕ := 69

-- Define David's push-ups in terms of Zachary's
def david_pushups : ℕ := zachary_pushups + 22

-- Theorem to prove
theorem pushup_difference : david_pushups - john_pushups = 4 := by
  sorry

end NUMINAMATH_CALUDE_pushup_difference_l1538_153810


namespace NUMINAMATH_CALUDE_pencil_grouping_l1538_153886

theorem pencil_grouping (total_pencils : ℕ) (num_groups : ℕ) (pencils_per_group : ℕ) :
  total_pencils = 25 →
  num_groups = 5 →
  total_pencils = num_groups * pencils_per_group →
  pencils_per_group = 5 :=
by sorry

end NUMINAMATH_CALUDE_pencil_grouping_l1538_153886


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l1538_153801

def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x ≥ 1}

theorem intersection_complement_theorem :
  A ∩ (Set.univ \ B) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l1538_153801


namespace NUMINAMATH_CALUDE_prime_difference_divisibility_l1538_153828

theorem prime_difference_divisibility 
  (p₁ p₂ p₃ p₄ q₁ q₂ q₃ q₄ : ℕ) 
  (hp : p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄)
  (hq : q₁ < q₂ ∧ q₂ < q₃ ∧ q₃ < q₄)
  (hp_prime : Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄)
  (hq_prime : Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄)
  (hp_diff : p₄ - p₁ = 8)
  (hq_diff : q₄ - q₁ = 8)
  (hp_gt_5 : p₁ > 5)
  (hq_gt_5 : q₁ > 5) :
  30 ∣ (p₁ - q₁) := by
sorry

end NUMINAMATH_CALUDE_prime_difference_divisibility_l1538_153828


namespace NUMINAMATH_CALUDE_factors_360_divisible_by_3_not_5_l1538_153841

def factors_divisible_by_3_not_5 (n : ℕ) : ℕ :=
  (Finset.filter (λ x => x ∣ n ∧ 3 ∣ x ∧ ¬(5 ∣ x)) (Finset.range (n + 1))).card

theorem factors_360_divisible_by_3_not_5 :
  factors_divisible_by_3_not_5 360 = 8 := by
  sorry

end NUMINAMATH_CALUDE_factors_360_divisible_by_3_not_5_l1538_153841


namespace NUMINAMATH_CALUDE_car_trip_duration_l1538_153803

/-- Proves that a car trip with given conditions has a total duration of 15 hours -/
theorem car_trip_duration (initial_speed initial_time additional_speed average_speed : ℝ) : 
  initial_speed = 30 →
  initial_time = 5 →
  additional_speed = 42 →
  average_speed = 38 →
  (initial_speed * initial_time + additional_speed * (15 - initial_time)) / 15 = average_speed :=
by
  sorry

#check car_trip_duration

end NUMINAMATH_CALUDE_car_trip_duration_l1538_153803


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l1538_153863

def is_divisible_by_all (n : ℕ) (divisors : List ℕ) : Prop :=
  ∀ d ∈ divisors, (n + 3) % d = 0

theorem smallest_number_divisible_by_all : 
  ∀ n : ℕ, n < 6303 → ¬(is_divisible_by_all n [70, 100, 84]) ∧ 
  is_divisible_by_all 6303 [70, 100, 84] :=
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l1538_153863


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1538_153856

-- Define the inverse relationship between a^3 and b^4
def inverse_relation (a b : ℝ) : Prop := ∃ k : ℝ, a^3 * b^4 = k

-- Theorem statement
theorem inverse_variation_problem (a₀ b₀ a₁ b₁ : ℝ) 
  (h_inverse : inverse_relation a₀ b₀ ∧ inverse_relation a₁ b₁)
  (h_initial : a₀ = 2 ∧ b₀ = 4)
  (h_final_a : a₁ = 8) :
  b₁ = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1538_153856


namespace NUMINAMATH_CALUDE_line_segment_b_value_l1538_153866

/-- Given a line segment with slope -3/2 from (0, b) to (8, 0), prove b = 12 -/
theorem line_segment_b_value (b : ℝ) : 
  (∀ x y, 0 ≤ x → x ≤ 8 → y = b - (3/2) * x) → 
  (b - (3/2) * 8 = 0) → 
  b = 12 := by
  sorry


end NUMINAMATH_CALUDE_line_segment_b_value_l1538_153866


namespace NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l1538_153807

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1 -/
theorem mans_age_to_sons_age_ratio :
  ∀ (man_age son_age : ℕ),
    son_age = 18 →
    man_age = son_age + 20 →
    ∃ (k : ℕ), (man_age + 2) = k * (son_age + 2) →
    (man_age + 2) / (son_age + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l1538_153807


namespace NUMINAMATH_CALUDE_wheel_moves_200cm_per_rotation_l1538_153876

/-- Represents the properties of a rotating wheel -/
structure RotatingWheel where
  rotations_per_minute : ℕ
  distance_per_hour : ℕ

/-- Calculates the distance moved during each rotation of the wheel -/
def distance_per_rotation (wheel : RotatingWheel) : ℚ :=
  wheel.distance_per_hour / (wheel.rotations_per_minute * 60)

/-- Theorem stating that a wheel with the given properties moves 200 cm per rotation -/
theorem wheel_moves_200cm_per_rotation (wheel : RotatingWheel) 
    (h1 : wheel.rotations_per_minute = 10)
    (h2 : wheel.distance_per_hour = 120000) : 
  distance_per_rotation wheel = 200 := by
  sorry

end NUMINAMATH_CALUDE_wheel_moves_200cm_per_rotation_l1538_153876


namespace NUMINAMATH_CALUDE_frog_jumps_l1538_153839

-- Define the hexagon vertices
inductive Vertex : Type
| A | B | C | D | E | F

-- Define the neighbor relation
def isNeighbor : Vertex → Vertex → Prop :=
  sorry

-- Define the number of paths from A to C in n jumps
def numPaths (n : ℕ) : ℕ :=
  if n % 2 = 0 then (1/3) * (4^(n/2) - 1) else 0

-- Define the number of paths from A to C in n jumps avoiding D
def numPathsAvoidD (n : ℕ) : ℕ :=
  if n % 2 = 0 then 3^(n/2 - 1) else 0

-- Define the survival probability after n jumps with mine at D
def survivalProb (n : ℕ) : ℚ :=
  if n % 2 = 0 then (3/4)^(n/2 - 1) else (3/4)^((n-1)/2)

-- Define the expected lifespan
def expectedLifespan : ℚ := 9

-- Main theorem
theorem frog_jumps :
  (∀ n : ℕ, numPaths n = if n % 2 = 0 then (1/3) * (4^(n/2) - 1) else 0) ∧
  (∀ n : ℕ, numPathsAvoidD n = if n % 2 = 0 then 3^(n/2 - 1) else 0) ∧
  (∀ n : ℕ, survivalProb n = if n % 2 = 0 then (3/4)^(n/2 - 1) else (3/4)^((n-1)/2)) ∧
  expectedLifespan = 9 :=
sorry

end NUMINAMATH_CALUDE_frog_jumps_l1538_153839


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1538_153815

theorem complex_equation_solution (a : ℝ) : (a + Complex.I) * (1 - a * Complex.I) = 2 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1538_153815


namespace NUMINAMATH_CALUDE_cos_product_sevenths_pi_l1538_153816

theorem cos_product_sevenths_pi : 
  Real.cos (π / 7) * Real.cos (2 * π / 7) * Real.cos (4 * π / 7) = -1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_product_sevenths_pi_l1538_153816


namespace NUMINAMATH_CALUDE_zhang_wang_sum_difference_l1538_153829

/-- Sum of arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * a₁ + (n * (n - 1) * d) / 2

/-- Sum of 26 consecutive odd numbers starting from 27 -/
def zhang_sum : ℕ := arithmetic_sum 27 2 26

/-- Sum of 26 consecutive natural numbers starting from 26 -/
def wang_sum : ℕ := arithmetic_sum 26 1 26

theorem zhang_wang_sum_difference :
  zhang_sum - wang_sum = 351 := by sorry

end NUMINAMATH_CALUDE_zhang_wang_sum_difference_l1538_153829


namespace NUMINAMATH_CALUDE_sum_of_number_and_its_square_l1538_153821

theorem sum_of_number_and_its_square : 17 + 17^2 = 306 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_number_and_its_square_l1538_153821


namespace NUMINAMATH_CALUDE_replaced_person_weight_l1538_153811

/-- Given a group of 8 people, if replacing one person with a new person weighing 89 kg
    increases the average weight by 3 kg, then the replaced person's weight was 65 kg. -/
theorem replaced_person_weight (initial_count : Nat) (new_person_weight : ℝ) (average_increase : ℝ) :
  initial_count = 8 →
  new_person_weight = 89 →
  average_increase = 3 →
  new_person_weight - (initial_count : ℝ) * average_increase = 65 :=
by sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l1538_153811


namespace NUMINAMATH_CALUDE_check_amount_error_l1538_153895

theorem check_amount_error (x y : ℕ) : 
  x ≥ 10 ∧ x ≤ 99 ∧ y ≥ 10 ∧ y ≤ 99 →  -- x and y are two-digit numbers
  y - x = 18 →                         -- difference is $17.82
  ∃ x y : ℕ, y = 2 * x                 -- y can be twice x
:= by sorry

end NUMINAMATH_CALUDE_check_amount_error_l1538_153895


namespace NUMINAMATH_CALUDE_division_problem_l1538_153875

theorem division_problem (Ω : ℕ) : 
  Ω ≤ 9 ∧ Ω ≥ 1 →
  (∃ (n : ℕ), n ≥ 10 ∧ n < 50 ∧ 504 / Ω = n + 2 * Ω) →
  Ω = 7 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1538_153875


namespace NUMINAMATH_CALUDE_parallel_lines_a_equals_two_l1538_153869

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The first line equation: x + ay - 1 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y - 1 = 0

/-- The second line equation: ax + 4y + 2 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := a * x + 4 * y + 2 = 0

theorem parallel_lines_a_equals_two :
  ∃ a : ℝ, (∀ x y, line1 a x y ↔ line2 a x y) → a = 2 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_equals_two_l1538_153869


namespace NUMINAMATH_CALUDE_smallest_positive_angle_proof_l1538_153838

/-- The smallest positive angle with the same terminal side as 400° -/
def smallest_positive_angle : ℝ := 40

/-- The set of angles with the same terminal side as 400° -/
def angle_set (k : ℤ) : ℝ := 400 + k * 360

theorem smallest_positive_angle_proof :
  ∀ k : ℤ, angle_set k > 0 → smallest_positive_angle ≤ angle_set k :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_proof_l1538_153838


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1538_153820

theorem fraction_equals_zero (x : ℝ) : (x - 2) / (x + 5) = 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1538_153820


namespace NUMINAMATH_CALUDE_haley_recycling_cans_l1538_153855

theorem haley_recycling_cans (total_cans : ℕ) (difference : ℕ) (cans_in_bag : ℕ) :
  total_cans = 9 →
  difference = 2 →
  total_cans - cans_in_bag = difference →
  cans_in_bag = 7 := by
sorry

end NUMINAMATH_CALUDE_haley_recycling_cans_l1538_153855


namespace NUMINAMATH_CALUDE_max_xy_on_line_segment_l1538_153888

/-- The maximum value of xy for a point P(x,y) on the line segment between A(3,0) and B(0,4) is 3 -/
theorem max_xy_on_line_segment : 
  let A : ℝ × ℝ := (3, 0)
  let B : ℝ × ℝ := (0, 4)
  ∃ (M : ℝ), M = 3 ∧ 
    ∀ (P : ℝ × ℝ), (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B) → 
      P.1 * P.2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_xy_on_line_segment_l1538_153888


namespace NUMINAMATH_CALUDE_sequence_property_l1538_153871

def is_valid_sequence (s : List Nat) : Prop :=
  (∀ x ∈ s, x = 0 ∨ x = 1) ∧ 
  (∀ i j, i + 4 < s.length → j + 4 < s.length → 
    (List.take 5 (List.drop i s) ≠ List.take 5 (List.drop j s) ∨ i = j)) ∧
  (∀ x, x = 0 ∨ x = 1 → 
    ¬(∀ i j, i + 4 < (s ++ [x]).length → j + 4 < (s ++ [x]).length → 
      (List.take 5 (List.drop i (s ++ [x])) ≠ List.take 5 (List.drop j (s ++ [x])) ∨ i = j)))

theorem sequence_property (s : List Nat) (h : is_valid_sequence s) (h_length : s.length ≥ 8) :
  List.take 4 s = List.take 4 (List.reverse s) :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l1538_153871


namespace NUMINAMATH_CALUDE_oldest_to_rick_age_ratio_l1538_153842

/-- Proves that the ratio of the oldest brother's age to Rick's age is 2:1 --/
theorem oldest_to_rick_age_ratio :
  ∀ (rick_age oldest_age middle_age smallest_age youngest_age : ℕ),
    rick_age = 15 →
    ∃ (k : ℕ), oldest_age = k * rick_age →
    middle_age = oldest_age / 3 →
    smallest_age = middle_age / 2 →
    youngest_age = smallest_age - 2 →
    youngest_age = 3 →
    oldest_age / rick_age = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_oldest_to_rick_age_ratio_l1538_153842


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1538_153879

/-- Represents a parabola in the form y = a(x-h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally and vertically --/
def shift_parabola (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h + dx, k := p.k + dy }

theorem parabola_shift_theorem :
  let original := Parabola.mk (-1) 0 1
  let shifted := shift_parabola original 2 (-2)
  shifted.a = -1 ∧ shifted.h = 2 ∧ shifted.k = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1538_153879


namespace NUMINAMATH_CALUDE_workshop_workers_count_workshop_workers_count_is_49_l1538_153814

/-- Proves that the total number of workers in a workshop is 49, given the following conditions:
  * The average salary of all workers is 8000
  * There are 7 technicians with an average salary of 20000
  * The average salary of the non-technicians is 6000
-/
theorem workshop_workers_count : ℕ → Prop :=
  fun (total_workers : ℕ) =>
    let avg_salary : ℚ := 8000
    let technician_count : ℕ := 7
    let technician_avg_salary : ℚ := 20000
    let non_technician_avg_salary : ℚ := 6000
    let non_technician_count : ℕ := total_workers - technician_count
    (↑total_workers * avg_salary = 
      ↑technician_count * technician_avg_salary + 
      ↑non_technician_count * non_technician_avg_salary) →
    total_workers = 49

theorem workshop_workers_count_is_49 : workshop_workers_count 49 := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_count_workshop_workers_count_is_49_l1538_153814


namespace NUMINAMATH_CALUDE_inheritance_division_l1538_153890

/-- Proves that dividing $527,500 equally among 5 people results in each person receiving $105,500 -/
theorem inheritance_division (total_amount : ℕ) (num_people : ℕ) (individual_share : ℕ) : 
  total_amount = 527500 → num_people = 5 → individual_share = total_amount / num_people → 
  individual_share = 105500 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_division_l1538_153890


namespace NUMINAMATH_CALUDE_function_minimum_l1538_153864

theorem function_minimum (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x - 3*a - 9 ≥ 0) →
  (1^2 + a*1 - 3*a - 9 = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_function_minimum_l1538_153864


namespace NUMINAMATH_CALUDE_b_lending_rate_to_c_l1538_153881

/-- Given the lending scenario between A, B, and C, prove that B's lending rate to C is 12.5% --/
theorem b_lending_rate_to_c (principal : ℝ) (a_rate : ℝ) (b_gain : ℝ) (time : ℝ) :
  principal = 3150 →
  a_rate = 8 →
  b_gain = 283.5 →
  time = 2 →
  ∃ (b_rate : ℝ),
    b_rate = 12.5 ∧
    b_gain = (principal * b_rate / 100 * time) - (principal * a_rate / 100 * time) :=
by sorry

end NUMINAMATH_CALUDE_b_lending_rate_to_c_l1538_153881


namespace NUMINAMATH_CALUDE_maria_roses_l1538_153850

/-- The number of roses Maria bought -/
def roses : ℕ := sorry

/-- The price of each flower -/
def flower_price : ℕ := 6

/-- The number of daisies Maria bought -/
def daisies : ℕ := 3

/-- The total amount Maria spent -/
def total_spent : ℕ := 60

theorem maria_roses :
  roses * flower_price + daisies * flower_price = total_spent →
  roses = 7 := by sorry

end NUMINAMATH_CALUDE_maria_roses_l1538_153850


namespace NUMINAMATH_CALUDE_count_four_digit_snappy_divisible_by_25_l1538_153853

def is_snappy (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a * 1000 + b * 100 + b * 10 + a ∧ a < 10 ∧ b < 10

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem count_four_digit_snappy_divisible_by_25 :
  ∃! (s : Finset ℕ),
    (∀ n ∈ s, is_four_digit n ∧ is_snappy n ∧ n % 25 = 0) ∧
    s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_four_digit_snappy_divisible_by_25_l1538_153853


namespace NUMINAMATH_CALUDE_intersection_equals_N_l1538_153800

def M : Set ℝ := {x | x < 2011}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_equals_N : M ∩ N = N := by sorry

end NUMINAMATH_CALUDE_intersection_equals_N_l1538_153800


namespace NUMINAMATH_CALUDE_infinite_sum_equals_five_twentyfourths_l1538_153862

/-- The sum of the infinite series n / (n^4 - 4n^2 + 8) from n = 1 to infinity is equal to 5/24. -/
theorem infinite_sum_equals_five_twentyfourths :
  ∑' n : ℕ+, (n : ℝ) / ((n : ℝ)^4 - 4*(n : ℝ)^2 + 8) = 5 / 24 := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_five_twentyfourths_l1538_153862


namespace NUMINAMATH_CALUDE_interest_calculation_l1538_153885

/-- Calculates the simple interest and proves the interest credited is 63 cents. -/
theorem interest_calculation (initial_savings : ℝ) (interest_rate : ℝ) (time : ℝ) 
  (additional_deposit : ℝ) (total_amount : ℝ) : ℝ :=
  let interest := initial_savings * interest_rate * time
  let amount_after_interest := initial_savings + interest
  let amount_after_deposit := amount_after_interest + additional_deposit
  let interest_credited := total_amount - (initial_savings + additional_deposit)
by
  have h1 : initial_savings = 500 := by sorry
  have h2 : interest_rate = 0.03 := by sorry
  have h3 : time = 1/4 := by sorry
  have h4 : additional_deposit = 15 := by sorry
  have h5 : total_amount = 515.63 := by sorry
  
  -- Prove that the interest credited is 63 cents
  sorry

#eval (515.63 - (500 + 15)) * 100 -- Should evaluate to 63.0

end NUMINAMATH_CALUDE_interest_calculation_l1538_153885


namespace NUMINAMATH_CALUDE_sum_nonnegative_implies_one_nonnegative_l1538_153824

theorem sum_nonnegative_implies_one_nonnegative (a b : ℝ) :
  a + b ≥ 0 → (a ≥ 0 ∨ b ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_nonnegative_implies_one_nonnegative_l1538_153824


namespace NUMINAMATH_CALUDE_preschool_nap_problem_l1538_153870

theorem preschool_nap_problem (initial_kids : ℕ) (awake_after_first_round : ℕ) (awake_after_second_round : ℕ) : 
  initial_kids = 20 →
  awake_after_first_round = initial_kids - initial_kids / 2 →
  awake_after_second_round = awake_after_first_round - awake_after_first_round / 2 →
  awake_after_second_round = 5 :=
by sorry

end NUMINAMATH_CALUDE_preschool_nap_problem_l1538_153870


namespace NUMINAMATH_CALUDE_shirts_washed_l1538_153804

theorem shirts_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (not_washed : ℕ) 
  (h1 : short_sleeve = 40)
  (h2 : long_sleeve = 23)
  (h3 : not_washed = 34) :
  short_sleeve + long_sleeve - not_washed = 29 := by
  sorry

end NUMINAMATH_CALUDE_shirts_washed_l1538_153804


namespace NUMINAMATH_CALUDE_oblique_drawing_properties_l1538_153849

-- Define the intuitive drawing using the oblique method
structure ObliqueDrawing where
  x_scale : ℝ
  y_scale : ℝ
  angle : ℝ

-- Define the properties of the oblique drawing
def is_valid_oblique_drawing (d : ObliqueDrawing) : Prop :=
  d.x_scale = 1 ∧ d.y_scale = 1/2 ∧ (d.angle = 135 ∨ d.angle = 45)

-- Theorem stating the properties of oblique drawing
theorem oblique_drawing_properties (d : ObliqueDrawing) 
  (h : is_valid_oblique_drawing d) : 
  d.x_scale = 1 ∧ 
  d.y_scale = 1/2 ∧ 
  (d.angle = 135 ∨ d.angle = 45) ∧ 
  ∃ (d' : ObliqueDrawing), is_valid_oblique_drawing d' ∧ d' ≠ d :=
sorry


end NUMINAMATH_CALUDE_oblique_drawing_properties_l1538_153849


namespace NUMINAMATH_CALUDE_prime_natural_equation_solutions_l1538_153832

theorem prime_natural_equation_solutions :
  ∀ p n : ℕ,
    Prime p →
    p^2 + n^2 = 3*p*n + 1 →
    ((p = 3 ∧ n = 1) ∨ (p = 3 ∧ n = 8)) := by
  sorry

end NUMINAMATH_CALUDE_prime_natural_equation_solutions_l1538_153832


namespace NUMINAMATH_CALUDE_divides_product_l1538_153831

theorem divides_product (a b c d : ℤ) (h1 : a ∣ b) (h2 : c ∣ d) : a * c ∣ b * d := by
  sorry

end NUMINAMATH_CALUDE_divides_product_l1538_153831


namespace NUMINAMATH_CALUDE_max_students_above_median_l1538_153802

theorem max_students_above_median (n : ℕ) (h : n = 101) :
  ∃ (scores : Fin n → ℝ),
    (∃ (median : ℝ), ∀ i : Fin n, scores i ≥ median → 
      (Fintype.card {i : Fin n | scores i > median} ≤ 50)) ∧
    (∃ (median : ℝ), Fintype.card {i : Fin n | scores i > median} = 50) :=
by sorry

end NUMINAMATH_CALUDE_max_students_above_median_l1538_153802


namespace NUMINAMATH_CALUDE_no_snow_probability_l1538_153852

theorem no_snow_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^5 = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l1538_153852


namespace NUMINAMATH_CALUDE_shoe_size_age_game_l1538_153830

theorem shoe_size_age_game (shoe_size age : ℕ) : 
  let current_year := 1952
  let birth_year := current_year - age
  let game_result := ((shoe_size + 7) * 2 + 5) * 50 + 1711 - birth_year
  game_result = 5059 → shoe_size = 43 ∧ age = 50 := by
sorry

end NUMINAMATH_CALUDE_shoe_size_age_game_l1538_153830


namespace NUMINAMATH_CALUDE_inverse_fraction_ratio_l1538_153854

noncomputable def g (x : ℝ) : ℝ := (3 * x - 2) / (x + 4)

theorem inverse_fraction_ratio (a b c d : ℝ) :
  (∀ x, g (((a * x + b) / (c * x + d)) : ℝ) = x) →
  a / c = -4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_fraction_ratio_l1538_153854


namespace NUMINAMATH_CALUDE_petes_number_l1538_153845

theorem petes_number : ∃ x : ℚ, 5 * (3 * x + 15) = 245 ∧ x = 34 / 3 := by
  sorry

end NUMINAMATH_CALUDE_petes_number_l1538_153845


namespace NUMINAMATH_CALUDE_zero_in_M_l1538_153894

def M : Set ℝ := {x | x ≤ 2}

theorem zero_in_M : (0 : ℝ) ∈ M := by
  sorry

end NUMINAMATH_CALUDE_zero_in_M_l1538_153894


namespace NUMINAMATH_CALUDE_set_intersection_equality_l1538_153818

def A : Set ℝ := {x | -5 < x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x | -7 < x ∧ x < a}
def C (b : ℝ) : Set ℝ := {x | b < x ∧ x < 2}

theorem set_intersection_equality (a b : ℝ) :
  A ∩ B a = C b → a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l1538_153818


namespace NUMINAMATH_CALUDE_group_collection_problem_l1538_153898

theorem group_collection_problem (n : ℕ) (total_rupees : ℚ) : 
  (n : ℚ) * n = total_rupees * 100 →
  total_rupees = 19.36 →
  n = 44 := by
  sorry

end NUMINAMATH_CALUDE_group_collection_problem_l1538_153898


namespace NUMINAMATH_CALUDE_exists_special_function_l1538_153884

open Function Set

/-- A function f: ℝ → ℝ satisfying specific properties --/
structure SpecialFunction where
  f : ℝ → ℝ
  increasing : Monotone f
  composite_increasing : Monotone (f ∘ f)
  not_fixed_point : ∀ a : ℝ, f a ≠ a
  involutive : ∀ x : ℝ, f (f x) = x

/-- Theorem stating the existence of a function satisfying the required properties --/
theorem exists_special_function : ∃ sf : SpecialFunction, True := by
  sorry

end NUMINAMATH_CALUDE_exists_special_function_l1538_153884


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1538_153865

theorem arithmetic_equality : 1 - 0.2 + 0.03 - 0.004 = 0.826 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1538_153865


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1538_153805

theorem unique_solution_condition (b : ℝ) : 
  (∃! x : ℝ, x^3 - 2*b*x^2 + b*x + b^2 - 2 = 0) ↔ (b = 0 ∨ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1538_153805


namespace NUMINAMATH_CALUDE_root_in_interval_l1538_153899

def f (x : ℝ) : ℝ := x^3 - x - 3

theorem root_in_interval :
  ∃ c ∈ Set.Icc 1 2, f c = 0 :=
sorry

end NUMINAMATH_CALUDE_root_in_interval_l1538_153899


namespace NUMINAMATH_CALUDE_distribute_problems_l1538_153812

theorem distribute_problems (num_problems : ℕ) (num_friends : ℕ) :
  num_problems = 6 → num_friends = 15 →
  (num_friends : ℕ) ^ (num_problems : ℕ) = 11390625 := by
  sorry

end NUMINAMATH_CALUDE_distribute_problems_l1538_153812


namespace NUMINAMATH_CALUDE_min_value_theorem_l1538_153851

theorem min_value_theorem (a : ℝ) (h : a > 3) :
  a + 1 / (a - 3) ≥ 5 ∧ (a + 1 / (a - 3) = 5 ↔ a = 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1538_153851


namespace NUMINAMATH_CALUDE_cylinder_cone_volume_l1538_153873

/-- Given a cylinder and cone with the same base and height, and a combined volume of 48cm³,
    prove that the volume of the cylinder is 36cm³ and the volume of the cone is 12cm³. -/
theorem cylinder_cone_volume (cylinder_volume cone_volume : ℝ) : 
  cylinder_volume + cone_volume = 48 →
  cylinder_volume = 3 * cone_volume →
  cylinder_volume = 36 ∧ cone_volume = 12 := by
sorry

end NUMINAMATH_CALUDE_cylinder_cone_volume_l1538_153873


namespace NUMINAMATH_CALUDE_abs_inequality_abs_inequality_with_constraints_l1538_153813

-- Part I
theorem abs_inequality (x : ℝ) : 
  |x - 1| + |2*x + 1| > 3 ↔ x < -1 ∨ x > 1 := by sorry

-- Part II
theorem abs_inequality_with_constraints (a b : ℝ) 
  (ha : a ∈ Set.Icc (-1 : ℝ) 1) (hb : b ∈ Set.Icc (-1 : ℝ) 1) : 
  |1 + a*b/4| > |(a + b)/2| := by sorry

end NUMINAMATH_CALUDE_abs_inequality_abs_inequality_with_constraints_l1538_153813


namespace NUMINAMATH_CALUDE_set_operations_and_range_l1538_153897

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 2 < x ∧ x < 4}
def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

theorem set_operations_and_range :
  (A ∩ B = {x | 2 < x ∧ x ≤ 3}) ∧
  (A ∪ (U \ B) = {x | x ≤ 3 ∨ x ≥ 4}) ∧
  (∀ a : ℝ, B ∩ C a = C a → 2 < a ∧ a < 3) := by sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l1538_153897


namespace NUMINAMATH_CALUDE_binomial_30_3_l1538_153859

theorem binomial_30_3 : (30 : ℕ).choose 3 = 4060 := by sorry

end NUMINAMATH_CALUDE_binomial_30_3_l1538_153859


namespace NUMINAMATH_CALUDE_dollar_calculation_l1538_153819

-- Define the $ operation
def dollar (a b : ℝ) : ℝ := (a^2 - b^2)^2

-- Theorem statement
theorem dollar_calculation (x : ℝ) : 
  dollar (x^3 + x) (x - x^3) = 16 * x^8 := by
  sorry

end NUMINAMATH_CALUDE_dollar_calculation_l1538_153819


namespace NUMINAMATH_CALUDE_carlas_marbles_l1538_153887

theorem carlas_marbles (x : ℕ) : 
  x + 134 - 68 + 56 = 244 → x = 122 := by
  sorry

end NUMINAMATH_CALUDE_carlas_marbles_l1538_153887


namespace NUMINAMATH_CALUDE_count_valid_triples_l1538_153837

def valid_triple (x y z : ℕ+) : Prop :=
  Nat.lcm x.val y.val = 180 ∧
  Nat.lcm x.val z.val = 420 ∧
  Nat.lcm y.val z.val = 1260

theorem count_valid_triples :
  ∃! (s : Finset (ℕ+ × ℕ+ × ℕ+)),
    (∀ t ∈ s, valid_triple t.1 t.2.1 t.2.2) ∧
    s.card = 4 :=
sorry

end NUMINAMATH_CALUDE_count_valid_triples_l1538_153837


namespace NUMINAMATH_CALUDE_triangle_heights_semiperimeter_inequality_l1538_153823

/-- Given a triangle with heights m_a, m_b, m_c and semiperimeter s,
    prove that the sum of squares of the heights is less than or equal to
    the square of the semiperimeter. -/
theorem triangle_heights_semiperimeter_inequality 
  (m_a m_b m_c s : ℝ) 
  (h_pos_a : 0 < m_a) (h_pos_b : 0 < m_b) (h_pos_c : 0 < m_c) (h_pos_s : 0 < s)
  (h_heights : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    m_a = (2 * (s * (s - a) * (s - b) * (s - c))^(1/2)) / a ∧
    m_b = (2 * (s * (s - a) * (s - b) * (s - c))^(1/2)) / b ∧
    m_c = (2 * (s * (s - a) * (s - b) * (s - c))^(1/2)) / c ∧
    s = (a + b + c) / 2) :
  m_a^2 + m_b^2 + m_c^2 ≤ s^2 := by sorry

end NUMINAMATH_CALUDE_triangle_heights_semiperimeter_inequality_l1538_153823


namespace NUMINAMATH_CALUDE_wrapping_paper_fraction_l1538_153896

theorem wrapping_paper_fraction (total_fraction : ℚ) (num_presents : ℕ) 
  (h1 : total_fraction = 3 / 10)
  (h2 : num_presents = 3) :
  total_fraction / num_presents = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_fraction_l1538_153896


namespace NUMINAMATH_CALUDE_robin_candy_packages_l1538_153867

/-- Given the total number of candy pieces and the number of pieces per package,
    calculate the number of candy packages. -/
def candy_packages (total_pieces : ℕ) (pieces_per_package : ℕ) : ℕ :=
  total_pieces / pieces_per_package

/-- Theorem stating that Robin has 45 packages of candy. -/
theorem robin_candy_packages :
  candy_packages 405 9 = 45 := by
  sorry

#eval candy_packages 405 9

end NUMINAMATH_CALUDE_robin_candy_packages_l1538_153867


namespace NUMINAMATH_CALUDE_coefficient_of_x5y2_l1538_153878

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the polynomial (x^2 + 3x - y)^5
def polynomial (x y : ℤ) : ℤ := (x^2 + 3*x - y)^5

-- Theorem statement
theorem coefficient_of_x5y2 :
  ∃ (coeff : ℤ), coeff = 90 ∧
  ∀ (x y : ℤ), 
    ∃ (rest : ℤ), 
      polynomial x y = coeff * x^5 * y^2 + rest ∧ 
      (∀ (a b : ℕ), a ≤ 5 ∧ b ≤ 2 ∧ (a, b) ≠ (5, 2) → 
        ∃ (other_terms : ℤ), rest = other_terms * x^a * y^b + other_terms) :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x5y2_l1538_153878


namespace NUMINAMATH_CALUDE_redistribution_theorem_l1538_153891

/-- The number of trucks after redistribution of oil containers -/
def num_trucks_after_redistribution : ℕ :=
  let initial_trucks_1 : ℕ := 7
  let boxes_per_truck_1 : ℕ := 20
  let initial_trucks_2 : ℕ := 5
  let boxes_per_truck_2 : ℕ := 12
  let containers_per_box : ℕ := 8
  let containers_per_truck_after : ℕ := 160
  let total_boxes : ℕ := initial_trucks_1 * boxes_per_truck_1 + initial_trucks_2 * boxes_per_truck_2
  let total_containers : ℕ := total_boxes * containers_per_box
  total_containers / containers_per_truck_after

theorem redistribution_theorem :
  num_trucks_after_redistribution = 10 :=
by sorry

end NUMINAMATH_CALUDE_redistribution_theorem_l1538_153891


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1538_153835

-- Define the set P
def P : Set ℝ := {1, 2, 3, 4}

-- Define the set Q
def Q : Set ℝ := {x | 0 < x ∧ x < 5}

-- Theorem stating that "x ∈ P" is a sufficient but not necessary condition for "x ∈ Q"
theorem p_sufficient_not_necessary_for_q :
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1538_153835


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1538_153868

theorem simplify_and_evaluate (x : ℝ) (h : x = 1 / (Real.sqrt 3 - 2)) :
  x^2 + 4*x - 4 = -5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1538_153868


namespace NUMINAMATH_CALUDE_parabola_properties_l1538_153877

-- Define the parabola
def parabola (b c x : ℝ) : ℝ := -x^2 + b*x + c

-- Define the roots of the parabola
def roots (b c : ℝ) : Set ℝ := {x | parabola b c x = 0}

-- Theorem statement
theorem parabola_properties :
  ∀ b c : ℝ,
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ + x₂ = 4 ∧ {x₁, x₂} ⊆ roots b c) →
  (b = 4 ∧ c > -4) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ {x₁, x₂} ⊆ roots b c ∧ |x₁ - x₂| = 2 → c = -3) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ {x₁, x₂} ⊆ roots b c ∧
    (c = (1 + Real.sqrt 17) / 2 ∨ c = (1 - Real.sqrt 17) / 2) ∧
    |c| = c - parabola b c 2) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l1538_153877


namespace NUMINAMATH_CALUDE_total_wheels_in_lot_l1538_153840

/-- The number of wheels on a standard car -/
def wheels_per_car : ℕ := 4

/-- The number of cars in the parking lot -/
def cars_in_lot : ℕ := 12

/-- Theorem: The total number of car wheels in the parking lot is 48 -/
theorem total_wheels_in_lot : cars_in_lot * wheels_per_car = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_in_lot_l1538_153840


namespace NUMINAMATH_CALUDE_all_propositions_true_l1538_153836

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (lineparallel : Line → Plane → Prop)
variable (skew : Line → Line → Prop)
variable (equidistant : Plane → Plane → Point → Prop)
variable (noncollinear : Point → Point → Point → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- Define the theorem
theorem all_propositions_true :
  (∀ (n : Line) (α β : Plane), perpendicular n α → perpendicular n β → parallel α β) ∧
  (∀ (α β : Plane) (p q r : Point), noncollinear p q r → equidistant α β p → equidistant α β q → equidistant α β r → parallel α β) ∧
  (∀ (m n : Line) (α β : Plane), skew m n → contains α n → lineparallel n β → contains β m → lineparallel m α → parallel α β) :=
sorry

end NUMINAMATH_CALUDE_all_propositions_true_l1538_153836


namespace NUMINAMATH_CALUDE_calculate_expression_l1538_153889

theorem calculate_expression : 15 * 25 + 35 * 15 + 16 * 28 + 32 * 16 = 1860 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1538_153889


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1538_153892

theorem modulus_of_complex_fraction : Complex.abs ((2 - Complex.I) / (1 + Complex.I)) = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1538_153892


namespace NUMINAMATH_CALUDE_f_value_at_7_6_l1538_153834

def periodic_function (f : ℝ → ℝ) (period : ℝ) : Prop :=
  ∀ x, f (x + period) = f x

theorem f_value_at_7_6 (f : ℝ → ℝ) (h1 : periodic_function f 4) 
  (h2 : ∀ x ∈ Set.Icc (-2) 2, f x = x + 1) : 
  f 7.6 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_7_6_l1538_153834


namespace NUMINAMATH_CALUDE_sin_15_minus_sin_75_l1538_153883

theorem sin_15_minus_sin_75 : 
  Real.sin (15 * π / 180) - Real.sin (75 * π / 180) = -Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_minus_sin_75_l1538_153883


namespace NUMINAMATH_CALUDE_smallest_integer_above_sqrt_sum_power_l1538_153817

theorem smallest_integer_above_sqrt_sum_power : 
  ∃ n : ℕ, n = 3742 ∧ (∀ m : ℕ, m < n → (m : ℝ) ≤ (Real.sqrt 5 + Real.sqrt 3)^6) ∧
  n > (Real.sqrt 5 + Real.sqrt 3)^6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_above_sqrt_sum_power_l1538_153817


namespace NUMINAMATH_CALUDE_fish_problem_l1538_153857

theorem fish_problem (west north left : ℕ) (E : ℕ) : 
  west = 1800 →
  north = 500 →
  left = 2870 →
  (3 * E) / 5 + west / 4 + north = left →
  E = 3200 :=
by
  sorry

end NUMINAMATH_CALUDE_fish_problem_l1538_153857


namespace NUMINAMATH_CALUDE_walking_distance_l1538_153806

/-- Proves that walking at 3 miles per hour for 1.5 hours results in a distance of 4.5 miles -/
theorem walking_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 3 → time = 1.5 → distance = speed * time → distance = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_walking_distance_l1538_153806


namespace NUMINAMATH_CALUDE_f_properties_l1538_153848

noncomputable section

variables (a b : ℝ)

def f (x : ℝ) := Real.log ((a^x) - (b^x))

theorem f_properties (h1 : a > 1) (h2 : b > 0) (h3 : b < 1) :
  -- 1. Domain of f is (0, +∞)
  (∀ x > 0, (a^x) - (b^x) > 0) ∧
  -- 2. f is strictly increasing on its domain
  (∀ x y, 0 < x ∧ x < y → f a b x < f a b y) ∧
  -- 3. f(x) > 0 for all x > 1 iff a - b ≥ 1
  (∀ x > 1, f a b x > 0) ↔ a - b ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1538_153848


namespace NUMINAMATH_CALUDE_equation_solution_l1538_153825

theorem equation_solution : 
  ∃ (x₁ x₂ : ℚ), (x₁ = 1/6 ∧ x₂ = -1/4) ∧ 
  (∀ x : ℚ, 4*x*(6*x - 1) = 1 - 6*x ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1538_153825


namespace NUMINAMATH_CALUDE_cost_of_eggs_l1538_153826

/-- The amount Samantha spent on the crate of eggs -/
def cost : ℝ := 5

/-- The number of eggs in the crate -/
def total_eggs : ℕ := 30

/-- The price of each egg in dollars -/
def price_per_egg : ℝ := 0.20

/-- The number of eggs left when Samantha recovers her capital -/
def eggs_left : ℕ := 5

/-- Theorem stating that the cost of the crate is $5 -/
theorem cost_of_eggs : cost = (total_eggs - eggs_left) * price_per_egg := by
  sorry

end NUMINAMATH_CALUDE_cost_of_eggs_l1538_153826


namespace NUMINAMATH_CALUDE_parabola_vertex_locus_l1538_153880

/-- The locus of the vertex of a parabola with specific constraints -/
theorem parabola_vertex_locus (a b s t : ℝ) : 
  (∀ x y, y = a * x^2 + b * x + 1) →  -- Parabola equation
  (8 * a^2 + 4 * a * b = b^3) →       -- Constraint on a and b
  (s = -b / (2 * a)) →                -- x-coordinate of vertex
  (t = (4 * a - b^2) / (4 * a)) →     -- y-coordinate of vertex
  (s * t = 1) :=                      -- Locus equation
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_locus_l1538_153880


namespace NUMINAMATH_CALUDE_triangle_line_equations_l1538_153808

/-- Triangle ABC with vertices A(-4,0), B(0,-3), and C(-2,1) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Definition of the specific triangle in the problem -/
def triangle_ABC : Triangle :=
  { A := (-4, 0)
  , B := (0, -3)
  , C := (-2, 1) }

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating the equations of line BC and altitude from A to BC -/
theorem triangle_line_equations (t : Triangle) (h : t = triangle_ABC) :
  ∃ (line_BC altitude : LineEquation),
    line_BC = { a := 2, b := 1, c := 3 } ∧
    altitude = { a := 1, b := -2, c := 4 } := by
  sorry

end NUMINAMATH_CALUDE_triangle_line_equations_l1538_153808


namespace NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l1538_153861

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the sum of the first and third terms is 8,
    prove that the second term is 4. -/
theorem arithmetic_sequence_second_term
  (a : ℕ → ℝ)
  (h_arithmetic : IsArithmeticSequence a)
  (h_sum : a 0 + a 2 = 8) :
  a 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l1538_153861


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_21_over_10_l1538_153882

theorem sqrt_difference_equals_21_over_10 :
  Real.sqrt (25 / 4) - Real.sqrt (4 / 25) = 21 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_21_over_10_l1538_153882
