import Mathlib

namespace train_crossing_time_l1718_171827

/-- Calculates the time it takes for a train to cross a platform -/
theorem train_crossing_time (train_speed_kmph : ℝ) (man_crossing_time : ℝ) (platform_length : ℝ) : 
  train_speed_kmph = 72 →
  man_crossing_time = 18 →
  platform_length = 280 →
  (platform_length + train_speed_kmph * man_crossing_time * (5/18)) / (train_speed_kmph * (5/18)) = 32 := by
  sorry


end train_crossing_time_l1718_171827


namespace fourth_quadrant_m_range_l1718_171864

theorem fourth_quadrant_m_range (m : ℝ) :
  let z : ℂ := (1 + m * Complex.I) / (1 + Complex.I)
  (z.re > 0 ∧ z.im < 0) → -1 < m ∧ m < 1 := by
  sorry

end fourth_quadrant_m_range_l1718_171864


namespace juan_saw_three_bicycles_l1718_171852

/-- The number of bicycles Juan saw -/
def num_bicycles : ℕ := 3

/-- The number of cars Juan saw -/
def num_cars : ℕ := 15

/-- The number of pickup trucks Juan saw -/
def num_pickup_trucks : ℕ := 8

/-- The number of tricycles Juan saw -/
def num_tricycles : ℕ := 1

/-- The total number of tires on all vehicles Juan saw -/
def total_tires : ℕ := 101

/-- The number of tires on a car -/
def tires_per_car : ℕ := 4

/-- The number of tires on a pickup truck -/
def tires_per_pickup : ℕ := 4

/-- The number of tires on a tricycle -/
def tires_per_tricycle : ℕ := 3

/-- The number of tires on a bicycle -/
def tires_per_bicycle : ℕ := 2

theorem juan_saw_three_bicycles :
  num_bicycles * tires_per_bicycle + 
  num_cars * tires_per_car + 
  num_pickup_trucks * tires_per_pickup + 
  num_tricycles * tires_per_tricycle = total_tires :=
by sorry

end juan_saw_three_bicycles_l1718_171852


namespace intersection_of_M_and_N_l1718_171811

-- Define sets M and N
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | x^2 < 4}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -2 < x ∧ x < 1} := by
  sorry

end intersection_of_M_and_N_l1718_171811


namespace factorization_problem_1_factorization_problem_2_l1718_171872

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) :
  5 * x^2 + 6 * x * y - 8 * y^2 = (5 * x - 4 * y) * (x + 2 * y) := by
  sorry

-- Problem 2
theorem factorization_problem_2 (x a : ℝ) :
  x^2 + 2 * x - 15 - a * x - 5 * a = (x + 5) * (x - (3 + a)) := by
  sorry

end factorization_problem_1_factorization_problem_2_l1718_171872


namespace angle_C_is_45_degrees_l1718_171897

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ -- Area of the triangle

-- Define the vectors p and q
def p (t : Triangle) : ℝ × ℝ := (4, t.a^2 + t.b^2 - t.c^2)
def q (t : Triangle) : ℝ × ℝ := (1, t.S)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem statement
theorem angle_C_is_45_degrees (t : Triangle) 
  (h : parallel (p t) (q t)) : t.C = π/4 := by
  sorry

end angle_C_is_45_degrees_l1718_171897


namespace room_area_l1718_171892

/-- The area of a rectangular room with length 5 feet and width 2 feet is 10 square feet. -/
theorem room_area : 
  let length : ℝ := 5
  let width : ℝ := 2
  length * width = 10 := by sorry

end room_area_l1718_171892


namespace solution_characterization_l1718_171853

def is_solution (a b : ℕ+) : Prop :=
  (a.val ^ 2 * b.val ^ 2 + 208 : ℕ) = 4 * (Nat.lcm a.val b.val + Nat.gcd a.val b.val) ^ 2

theorem solution_characterization :
  ∀ a b : ℕ+, is_solution a b ↔ 
    ((a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4) ∨ (a = 2 ∧ b = 12) ∨ (a = 12 ∧ b = 2)) :=
by sorry

end solution_characterization_l1718_171853


namespace company_kw_price_percentage_l1718_171822

theorem company_kw_price_percentage (price_kw assets_a assets_b : ℝ) : 
  price_kw = 2 * assets_b →
  price_kw = 0.7878787878787878 * (assets_a + assets_b) →
  (price_kw - assets_a) / assets_a = 0.3 := by
  sorry

end company_kw_price_percentage_l1718_171822


namespace geometric_progression_solution_l1718_171854

/-- Given three terms of a geometric progression in the form (15 + x), (45 + x), and (135 + x),
    prove that x = 0 is the unique solution. -/
theorem geometric_progression_solution (x : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (45 + x) = (15 + x) * r ∧ (135 + x) = (45 + x) * r) ↔ x = 0 := by
  sorry

end geometric_progression_solution_l1718_171854


namespace triangle_angle_theorem_l1718_171836

theorem triangle_angle_theorem (A B C : ℝ) : 
  A = 32 →
  B = 3 * A →
  C = 2 * A - 12 →
  A + B + C = 180 →
  C = 52 := by
sorry

end triangle_angle_theorem_l1718_171836


namespace sum_first_8_even_numbers_l1718_171813

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2 * (i + 1))

theorem sum_first_8_even_numbers :
  (first_n_even_numbers 8).sum = 72 := by
  sorry

end sum_first_8_even_numbers_l1718_171813


namespace factorization_of_2a2_minus_8b2_l1718_171848

theorem factorization_of_2a2_minus_8b2 (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2*b) * (a - 2*b) := by
  sorry

end factorization_of_2a2_minus_8b2_l1718_171848


namespace slope_of_line_l1718_171834

theorem slope_of_line (x y : ℝ) :
  x + 2 * y - 4 = 0 → (∃ m b : ℝ, y = m * x + b ∧ m = -1/2) :=
by
  sorry

end slope_of_line_l1718_171834


namespace count_solutions_eq_288_l1718_171884

/-- The count of positive integers N less than 500 for which x^⌊x⌋ = N has a solution -/
def count_solutions : ℕ :=
  let floor_0_count := 1  -- N = 1 for ⌊x⌋ = 0
  let floor_1_count := 0  -- Already counted in floor_0_count
  let floor_2_count := 5  -- N = 4, 5, ..., 8
  let floor_3_count := 38 -- N = 27, 28, ..., 64
  let floor_4_count := 244 -- N = 256, 257, ..., 499
  floor_0_count + floor_1_count + floor_2_count + floor_3_count + floor_4_count

/-- The main theorem stating that the count of solutions is 288 -/
theorem count_solutions_eq_288 : count_solutions = 288 := by
  sorry

end count_solutions_eq_288_l1718_171884


namespace popton_bus_toes_count_l1718_171826

/-- Represents the three races on planet Popton -/
inductive Race
  | Hoopit
  | Neglart
  | Zentorian

/-- Returns the number of toes per hand for a given race -/
def toesPerHand (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 3
  | Race.Neglart => 2
  | Race.Zentorian => 4

/-- Returns the number of hands for a given race -/
def handsCount (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 4
  | Race.Neglart => 5
  | Race.Zentorian => 6

/-- Returns the number of students of a given race on the bus -/
def studentsCount (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 7
  | Race.Neglart => 8
  | Race.Zentorian => 5

/-- Calculates the total number of toes for a given race on the bus -/
def totalToesForRace (r : Race) : ℕ :=
  toesPerHand r * handsCount r * studentsCount r

/-- Theorem: The total number of toes on the Popton school bus is 284 -/
theorem popton_bus_toes_count :
  (totalToesForRace Race.Hoopit) + (totalToesForRace Race.Neglart) + (totalToesForRace Race.Zentorian) = 284 := by
  sorry

end popton_bus_toes_count_l1718_171826


namespace sin_150_cos_30_l1718_171860

theorem sin_150_cos_30 : Real.sin (150 * π / 180) * Real.cos (30 * π / 180) = Real.sqrt 3 / 4 := by
  sorry

end sin_150_cos_30_l1718_171860


namespace zebra_chase_time_l1718_171868

/-- The time (in hours) it takes for the zebra to catch up with the tiger -/
def catchup_time : ℝ := 6

/-- The speed of the zebra in km/h -/
def zebra_speed : ℝ := 55

/-- The speed of the tiger in km/h -/
def tiger_speed : ℝ := 30

/-- The time (in hours) after which the zebra starts chasing the tiger -/
def chase_start_time : ℝ := 5

theorem zebra_chase_time :
  chase_start_time * tiger_speed + catchup_time * tiger_speed = catchup_time * zebra_speed :=
sorry

end zebra_chase_time_l1718_171868


namespace notebook_pen_combinations_l1718_171847

theorem notebook_pen_combinations (notebooks : Finset α) (pens : Finset β) 
  (h1 : notebooks.card = 4) (h2 : pens.card = 5) :
  (notebooks.product pens).card = 20 := by
  sorry

end notebook_pen_combinations_l1718_171847


namespace group_frequency_number_l1718_171859

-- Define the sample capacity
def sample_capacity : ℕ := 100

-- Define the frequency of the group
def group_frequency : ℚ := 3/10

-- Define the frequency number calculation
def frequency_number (capacity : ℕ) (frequency : ℚ) : ℚ := capacity * frequency

-- Theorem statement
theorem group_frequency_number :
  frequency_number sample_capacity group_frequency = 30 := by sorry

end group_frequency_number_l1718_171859


namespace system_solution_implies_a_equals_five_l1718_171851

theorem system_solution_implies_a_equals_five 
  (x y a : ℝ) 
  (eq1 : 2 * x - y = 1) 
  (eq2 : 3 * x + y = 2 * a - 1) 
  (eq3 : 2 * y - x = 4) : 
  a = 5 := by
sorry

end system_solution_implies_a_equals_five_l1718_171851


namespace parabolas_similar_l1718_171846

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := x^2
def parabola2 (x : ℝ) : ℝ := 2 * x^2

-- Define a homothety transformation
def homothety (scale : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (scale * p.1, scale * p.2)

-- Theorem statement
theorem parabolas_similar :
  ∀ x : ℝ, homothety 2 (x, parabola2 x) = (2*x, parabola1 (2*x)) :=
by sorry

end parabolas_similar_l1718_171846


namespace number_exceeds_fraction_l1718_171833

theorem number_exceeds_fraction : ∃ x : ℚ, x = (3/8) * x + 30 ∧ x = 48 := by
  sorry

end number_exceeds_fraction_l1718_171833


namespace popsicle_stick_ratio_l1718_171814

theorem popsicle_stick_ratio : 
  ∀ (steve sid sam : ℕ),
  steve = 12 →
  sid = 2 * steve →
  sam + sid + steve = 108 →
  sam / sid = 3 :=
by
  sorry

end popsicle_stick_ratio_l1718_171814


namespace hexagon_area_l1718_171807

/-- Given a square with area 16 and a regular hexagon with perimeter 3/4 of the square's perimeter,
    the area of the hexagon is 32√3/27. -/
theorem hexagon_area (s : ℝ) (t : ℝ) : 
  s^2 = 16 → 
  4 * s = 18 * t → 
  (3 * t^2 * Real.sqrt 3) / 2 = (32 * Real.sqrt 3) / 27 := by
  sorry

end hexagon_area_l1718_171807


namespace largest_five_digit_product_120_sum_18_l1718_171821

/-- Represents a five-digit number -/
structure FiveDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  d5 : Nat
  is_valid : d1 ≥ 1 ∧ d1 ≤ 9 ∧ 
             d2 ≥ 0 ∧ d2 ≤ 9 ∧ 
             d3 ≥ 0 ∧ d3 ≤ 9 ∧ 
             d4 ≥ 0 ∧ d4 ≤ 9 ∧ 
             d5 ≥ 0 ∧ d5 ≤ 9

/-- The value of a five-digit number -/
def value (n : FiveDigitNumber) : Nat :=
  10000 * n.d1 + 1000 * n.d2 + 100 * n.d3 + 10 * n.d4 + n.d5

/-- The product of the digits of a five-digit number -/
def digit_product (n : FiveDigitNumber) : Nat :=
  n.d1 * n.d2 * n.d3 * n.d4 * n.d5

/-- The sum of the digits of a five-digit number -/
def digit_sum (n : FiveDigitNumber) : Nat :=
  n.d1 + n.d2 + n.d3 + n.d4 + n.d5

/-- Theorem: The sum of digits of the largest five-digit number 
    whose digits' product is 120 is 18 -/
theorem largest_five_digit_product_120_sum_18 :
  ∃ (N : FiveDigitNumber), 
    (∀ (M : FiveDigitNumber), digit_product M = 120 → value M ≤ value N) ∧ 
    digit_product N = 120 ∧ 
    digit_sum N = 18 := by
  sorry

end largest_five_digit_product_120_sum_18_l1718_171821


namespace quadratic_root_arithmetic_sequence_l1718_171830

theorem quadratic_root_arithmetic_sequence (p q r : ℝ) : 
  p ≥ q → q ≥ r → r ≥ 0 →  -- Conditions on p, q, r
  (∃ d : ℝ, q = p - d ∧ r = p - 2*d) →  -- Arithmetic sequence condition
  (∃! x : ℝ, p*x^2 + q*x + r = 0) →  -- Exactly one root condition
  (∃ x : ℝ, p*x^2 + q*x + r = 0 ∧ x = -2 + Real.sqrt 3) := by
sorry

end quadratic_root_arithmetic_sequence_l1718_171830


namespace line_mb_value_l1718_171810

/-- Given a line passing through points (0, -1) and (1, 1) with equation y = mx + b, prove that mb = -2 -/
theorem line_mb_value (m b : ℝ) : 
  (0 : ℝ) = m * 0 + b → -- The line passes through (0, -1)
  (1 : ℝ) = m * 1 + b → -- The line passes through (1, 1)
  m * b = -2 := by
sorry

end line_mb_value_l1718_171810


namespace investment_growth_l1718_171838

theorem investment_growth (x : ℝ) : 
  (1 + x / 100) * (1 - 30 / 100) = 1 + 11.99999999999999 / 100 → x = 60 := by
  sorry

end investment_growth_l1718_171838


namespace enclosed_area_circular_arcs_octagon_l1718_171844

/-- The area enclosed by a curve formed by circular arcs centered on a regular octagon -/
theorem enclosed_area_circular_arcs_octagon (n : ℕ) (arc_length : ℝ) (side_length : ℝ) : 
  n = 12 → 
  arc_length = 3 * π / 4 → 
  side_length = 3 → 
  ∃ (area : ℝ), area = 54 + 18 * Real.sqrt 2 + 81 * π / 64 - 54 * π / 64 - 18 * π * Real.sqrt 2 / 64 :=
by sorry

end enclosed_area_circular_arcs_octagon_l1718_171844


namespace students_in_all_classes_l1718_171870

/-- Proves that 8 students are registered for all 3 classes given the problem conditions -/
theorem students_in_all_classes (total_students : ℕ) (history_students : ℕ) (math_students : ℕ) 
  (english_students : ℕ) (two_classes_students : ℕ) : ℕ :=
by
  sorry

#check students_in_all_classes 68 19 14 26 7

end students_in_all_classes_l1718_171870


namespace min_value_theorem_l1718_171808

open Real

theorem min_value_theorem (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  let f := fun x => 1/x + 9/(1-x)
  (∀ y, 0 < y ∧ y < 1 → f y ≥ 16) ∧ (∃ z, 0 < z ∧ z < 1 ∧ f z = 16) := by
  sorry

end min_value_theorem_l1718_171808


namespace horatio_sonnets_l1718_171861

/-- Proves that Horatio wrote 12 sonnets in total -/
theorem horatio_sonnets (lines_per_sonnet : ℕ) (read_sonnets : ℕ) (unread_lines : ℕ) : 
  lines_per_sonnet = 14 → read_sonnets = 7 → unread_lines = 70 →
  read_sonnets + (unread_lines / lines_per_sonnet) = 12 := by
  sorry

end horatio_sonnets_l1718_171861


namespace parallel_vectors_m_value_l1718_171886

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (5, -2)
  are_parallel a b → m = -10 := by
sorry

end parallel_vectors_m_value_l1718_171886


namespace largest_three_digit_multiple_of_3_and_5_l1718_171835

theorem largest_three_digit_multiple_of_3_and_5 : ∃ n : ℕ, n = 990 ∧ 
  n % 3 = 0 ∧ n % 5 = 0 ∧ 
  n < 1000 ∧
  ∀ m : ℕ, m < 1000 → m % 3 = 0 → m % 5 = 0 → m ≤ n :=
by sorry

end largest_three_digit_multiple_of_3_and_5_l1718_171835


namespace magnitude_comparison_l1718_171840

theorem magnitude_comparison (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 1/2) 
  (A : ℝ) (hA : A = 1 - a^2)
  (B : ℝ) (hB : B = 1 + a^2)
  (C : ℝ) (hC : C = 1 / (1 - a))
  (D : ℝ) (hD : D = 1 / (1 + a)) :
  (1 - a > a^2) ∧ (D < A ∧ A < B ∧ B < C) := by
sorry

end magnitude_comparison_l1718_171840


namespace baker_cakes_l1718_171883

theorem baker_cakes (initial_cakes : ℕ) 
  (bought_cakes : ℕ := 103)
  (sold_cakes : ℕ := 86)
  (final_cakes : ℕ := 190)
  (h : initial_cakes + bought_cakes - sold_cakes = final_cakes) :
  initial_cakes = 173 := by
  sorry

end baker_cakes_l1718_171883


namespace village_population_l1718_171873

theorem village_population (P : ℕ) : 
  (P : ℝ) * (1 - 0.1) * (1 - 0.25) * (1 - 0.12) * (1 - 0.15) = 4136 → 
  P = 8192 := by
sorry

end village_population_l1718_171873


namespace triangle_nabla_equality_l1718_171885

-- Define the triangle operation
def triangle (a b : ℤ) : ℤ := 3 * a + 2 * b

-- Define the nabla operation
def nabla (a b : ℤ) : ℤ := 2 * a + 3 * b

-- Theorem to prove
theorem triangle_nabla_equality : triangle 2 (nabla 3 4) = 42 := by
  sorry

end triangle_nabla_equality_l1718_171885


namespace range_of_S_l1718_171845

theorem range_of_S (x₁ x₂ x₃ x₄ : ℝ) 
  (h_nonneg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0) 
  (h_sum : x₁ + x₂ - x₃ + x₄ = 1) : 
  let S := 1 - (x₁^4 + x₂^4 + x₃^4 + x₄^4) - 
    6 * (x₁*x₂ + x₁*x₃ + x₁*x₄ + x₂*x₃ + x₂*x₄ + x₃*x₄)
  0 ≤ S ∧ S ≤ 3/4 := by
sorry

end range_of_S_l1718_171845


namespace all_inhabitants_can_reach_palace_l1718_171882

-- Define the kingdom as a square
def kingdom_side_length : ℝ := 2

-- Define the speed of inhabitants
def inhabitant_speed : ℝ := 3

-- Define the available time
def available_time : ℝ := 7

-- Theorem statement
theorem all_inhabitants_can_reach_palace :
  ∀ (x y : ℝ), 
    0 ≤ x ∧ x ≤ kingdom_side_length ∧
    0 ≤ y ∧ y ≤ kingdom_side_length →
    ∃ (t : ℝ), 
      0 ≤ t ∧ t ≤ available_time ∧
      t * inhabitant_speed ≥ Real.sqrt ((x - kingdom_side_length/2)^2 + (y - kingdom_side_length/2)^2) :=
by sorry

end all_inhabitants_can_reach_palace_l1718_171882


namespace lemonade_proportion_l1718_171804

/-- Given that 40 lemons make 50 gallons of lemonade, prove that 12 lemons make 15 gallons -/
theorem lemonade_proportion :
  let lemons_for_50 : ℚ := 40
  let gallons_50 : ℚ := 50
  let gallons_15 : ℚ := 15
  let lemons_for_15 : ℚ := 12
  (lemons_for_50 / gallons_50 = lemons_for_15 / gallons_15) := by sorry

end lemonade_proportion_l1718_171804


namespace least_reducible_fraction_l1718_171817

/-- A fraction is reducible if the GCD of its numerator and denominator is greater than 1 -/
def IsReducible (n : ℕ) : Prop :=
  Nat.gcd (n - 17) (3 * n + 4) > 1

/-- The fraction (n-17)/(3n+4) is non-zero for positive n -/
def IsNonZero (n : ℕ) : Prop :=
  n > 0 ∧ n ≠ 17

theorem least_reducible_fraction :
  IsReducible 22 ∧ IsNonZero 22 ∧ ∀ n < 22, ¬(IsReducible n ∧ IsNonZero n) :=
sorry

end least_reducible_fraction_l1718_171817


namespace larger_number_proof_l1718_171867

theorem larger_number_proof (a b : ℕ) : 
  (Nat.gcd a b = 25) →
  (Nat.lcm a b = 4550) →
  (13 ∣ Nat.lcm a b) →
  (14 ∣ Nat.lcm a b) →
  (max a b = 350) :=
by sorry

end larger_number_proof_l1718_171867


namespace compound_composition_l1718_171829

/-- Atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- Atomic weight of phosphorus in g/mol -/
def atomic_weight_P : ℝ := 30.97

/-- Atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The molecular weight of the compound in g/mol -/
def compound_weight : ℝ := 122

/-- The number of oxygen atoms in the compound -/
def num_oxygen_atoms : ℕ := 4

theorem compound_composition :
  ∀ x : ℕ, 
    atomic_weight_Al + atomic_weight_P + x * atomic_weight_O = compound_weight 
    ↔ 
    x = num_oxygen_atoms :=
by sorry

end compound_composition_l1718_171829


namespace boys_at_reunion_l1718_171898

/-- The number of handshakes when n boys each shake hands once with every other boy -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: There are 7 boys at the reunion -/
theorem boys_at_reunion : ∃ n : ℕ, n > 0 ∧ handshakes n = 21 ∧ n = 7 := by
  sorry

end boys_at_reunion_l1718_171898


namespace geometric_arithmetic_sequence_l1718_171815

/-- A geometric sequence with common ratio q where the first, third, and second terms form an arithmetic sequence has q = 1 or q = -1 -/
theorem geometric_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  (a 3 - a 2 = a 2 - a 1) →    -- arithmetic sequence condition
  (q = 1 ∨ q = -1) := by sorry

end geometric_arithmetic_sequence_l1718_171815


namespace line_slope_at_minimum_l1718_171878

/-- Given a line ax - by + 2 = 0 (a > 0, b > 0) passing through (-1, 2),
    the slope is 2 when 2/a + 1/b is minimized. -/
theorem line_slope_at_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 2*b = 2) →
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 2 → 2/x + 1/y ≥ 2/a + 1/b) →
  b/a = 2 :=
by sorry

end line_slope_at_minimum_l1718_171878


namespace parallelogram_reconstruction_l1718_171843

/-- Given a parallelogram ABCD with E as the midpoint of BC and F as the midpoint of CD,
    prove that the coordinates of C can be determined from the coordinates of A, E, and F. -/
theorem parallelogram_reconstruction (A E F : ℝ × ℝ) :
  let K : ℝ × ℝ := ((E.1 + F.1) / 2, (E.2 + F.2) / 2)
  let C : ℝ × ℝ := (A.1 / 2, A.2 / 2)
  (∃ (B D : ℝ × ℝ), 
    -- ABCD is a parallelogram
    (A.1 - B.1 = D.1 - C.1 ∧ A.2 - B.2 = D.2 - C.2) ∧
    (A.1 - D.1 = B.1 - C.1 ∧ A.2 - D.2 = B.2 - C.2) ∧
    -- E is the midpoint of BC
    (E.1 = (B.1 + C.1) / 2 ∧ E.2 = (B.2 + C.2) / 2) ∧
    -- F is the midpoint of CD
    (F.1 = (C.1 + D.1) / 2 ∧ F.2 = (C.2 + D.2) / 2)) :=
by sorry

end parallelogram_reconstruction_l1718_171843


namespace distance_C_D_l1718_171895

/-- An ellipse with equation 16(x-2)^2 + 4y^2 = 64 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 16 * (p.1 - 2)^2 + 4 * p.2^2 = 64}

/-- The center of the ellipse -/
def center : ℝ × ℝ := (2, 0)

/-- The semi-major axis length -/
def a : ℝ := 4

/-- The semi-minor axis length -/
def b : ℝ := 2

/-- An endpoint of the major axis -/
def C : ℝ × ℝ := (center.1, center.2 + a)

/-- An endpoint of the minor axis -/
def D : ℝ × ℝ := (center.1 + b, center.2)

/-- The theorem stating the distance between C and D -/
theorem distance_C_D : Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 := by
  sorry

end distance_C_D_l1718_171895


namespace intersection_points_theorem_l1718_171839

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  lines : ℕ
  intersections : ℕ

/-- Predicate to check if a configuration is valid -/
def IsValidConfiguration (config : LineConfiguration) : Prop :=
  config.lines = 100 ∧ (config.intersections = 100 ∨ config.intersections = 99)

theorem intersection_points_theorem :
  ∃ (config1 config2 : LineConfiguration),
    IsValidConfiguration config1 ∧
    IsValidConfiguration config2 ∧
    config1.intersections = 100 ∧
    config2.intersections = 99 :=
sorry

end intersection_points_theorem_l1718_171839


namespace triangle_side_length_range_l1718_171812

theorem triangle_side_length_range (a : ℝ) : 
  (∃ (s₁ s₂ s₃ : ℝ), s₁ = 3*a - 1 ∧ s₂ = 4*a + 1 ∧ s₃ = 12 - a ∧ 
    s₁ + s₂ > s₃ ∧ s₁ + s₃ > s₂ ∧ s₂ + s₃ > s₁) ↔ 
  (3/2 < a ∧ a < 5) :=
by sorry

end triangle_side_length_range_l1718_171812


namespace larger_number_proof_l1718_171899

theorem larger_number_proof (S L : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 6 * S + 10) : 
  L = 1636 := by
sorry

end larger_number_proof_l1718_171899


namespace symmetric_circle_equation_l1718_171849

/-- The equation of a circle symmetric to another circle with respect to a line. -/
theorem symmetric_circle_equation (x y : ℝ) : 
  (∃ (x₀ y₀ : ℝ), (x - x₀)^2 + (y - y₀)^2 = 7 ∧ x₀ + y₀ = 4) → 
  (x^2 + y^2 = 7) := by sorry

end symmetric_circle_equation_l1718_171849


namespace unique_solution_condition_l1718_171866

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 3) ↔ d ≠ 4 := by
  sorry

end unique_solution_condition_l1718_171866


namespace new_machine_rate_proof_l1718_171856

/-- The rate of the old machine in bolts per hour -/
def old_machine_rate : ℝ := 100

/-- The time both machines work together in minutes -/
def work_time : ℝ := 84

/-- The total number of bolts produced by both machines -/
def total_bolts : ℝ := 350

/-- The rate of the new machine in bolts per hour -/
def new_machine_rate : ℝ := 150

theorem new_machine_rate_proof :
  (old_machine_rate * work_time / 60 + new_machine_rate * work_time / 60) = total_bolts :=
by sorry

end new_machine_rate_proof_l1718_171856


namespace max_teams_double_round_robin_l1718_171888

/-- A schedule for a double round robin tournament. -/
def Schedule (n : ℕ) := Fin n → Fin 4 → List (Fin n)

/-- Predicate to check if a schedule is valid according to the tournament rules. -/
def is_valid_schedule (n : ℕ) (s : Schedule n) : Prop :=
  -- Each team plays with every other team twice
  (∀ i j : Fin n, i ≠ j → (∃ w : Fin 4, i ∈ s j w) ∧ (∃ w : Fin 4, j ∈ s i w)) ∧
  -- If a team has a home game in a week, it cannot have any away games that week
  (∀ i : Fin n, ∀ w : Fin 4, (s i w).length > 0 → ∀ j : Fin n, i ∉ s j w)

/-- The maximum number of teams that can complete the tournament in 4 weeks is 6. -/
theorem max_teams_double_round_robin : 
  (∃ s : Schedule 6, is_valid_schedule 6 s) ∧ 
  (∀ s : Schedule 7, ¬ is_valid_schedule 7 s) :=
sorry

end max_teams_double_round_robin_l1718_171888


namespace power_gt_one_iff_diff_times_b_gt_zero_l1718_171876

theorem power_gt_one_iff_diff_times_b_gt_zero
  (a b : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  a^b > 1 ↔ (a - 1) * b > 0 := by
  sorry

end power_gt_one_iff_diff_times_b_gt_zero_l1718_171876


namespace lions_scored_18_l1718_171803

-- Define the total score and winning margin
def total_score : ℕ := 52
def winning_margin : ℕ := 16

-- Define the Lions' score as a function of the total score and winning margin
def lions_score (total : ℕ) (margin : ℕ) : ℕ :=
  (total - margin) / 2

-- Theorem statement
theorem lions_scored_18 :
  lions_score total_score winning_margin = 18 := by
  sorry

end lions_scored_18_l1718_171803


namespace algae_growth_l1718_171806

/-- Represents the number of cells in an algae colony after a given number of days -/
def algaeCells (initialCells : ℕ) (divisionPeriod : ℕ) (totalDays : ℕ) : ℕ :=
  initialCells * (2 ^ (totalDays / divisionPeriod))

/-- Theorem stating that an algae colony starting with 5 cells, doubling every 3 days,
    will have 20 cells after 9 days -/
theorem algae_growth : algaeCells 5 3 9 = 20 := by
  sorry


end algae_growth_l1718_171806


namespace nancy_total_games_l1718_171800

/-- The total number of games Nancy will attend over three months -/
def total_games (this_month : ℕ) (last_month : ℕ) (next_month : ℕ) : ℕ :=
  this_month + last_month + next_month

/-- Proof that Nancy will attend 24 games in total -/
theorem nancy_total_games :
  total_games 9 8 7 = 24 := by
  sorry

end nancy_total_games_l1718_171800


namespace f_derivative_l1718_171890

noncomputable def f (x : ℝ) : ℝ := Real.log (5 * x + Real.sqrt (25 * x^2 + 1)) - Real.sqrt (25 * x^2 + 1) * Real.arctan (5 * x)

theorem f_derivative (x : ℝ) : 
  deriv f x = -(25 * x * Real.arctan (5 * x)) / Real.sqrt (25 * x^2 + 1) :=
by sorry

end f_derivative_l1718_171890


namespace quiz_probabilities_l1718_171809

/-- Represents the total number of questions -/
def total_questions : ℕ := 5

/-- Represents the number of multiple-choice questions -/
def multiple_choice_questions : ℕ := 3

/-- Represents the number of true or false questions -/
def true_false_questions : ℕ := 2

/-- The probability that A draws a multiple-choice question while B draws a true or false question -/
def prob_A_multiple_B_true_false : ℚ := 3/10

/-- The probability that at least one of A and B draws a multiple-choice question -/
def prob_at_least_one_multiple : ℚ := 9/10

theorem quiz_probabilities :
  (prob_A_multiple_B_true_false = multiple_choice_questions * true_false_questions / (total_questions * (total_questions - 1))) ∧
  (prob_at_least_one_multiple = 1 - (true_false_questions * (true_false_questions - 1) / (total_questions * (total_questions - 1)))) :=
by sorry

end quiz_probabilities_l1718_171809


namespace arithmetic_sequence_sum_11_l1718_171823

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem arithmetic_sequence_sum_11 (seq : ArithmeticSequence) :
  sum_n seq 15 = 75 ∧ seq.a 3 + seq.a 4 + seq.a 5 = 12 → sum_n seq 11 = 99 / 2 := by
  sorry

end arithmetic_sequence_sum_11_l1718_171823


namespace geom_seq_306th_term_l1718_171841

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

theorem geom_seq_306th_term (a₁ a₂ : ℝ) (h1 : a₁ = 7) (h2 : a₂ = -7) :
  geometric_sequence a₁ (a₂ / a₁) 306 = -7 :=
by sorry

end geom_seq_306th_term_l1718_171841


namespace checkered_square_covering_l1718_171820

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (vertices : Fin 3 → ℝ × ℝ)

/-- Represents a 2x2 checkered square -/
def CheckeredSquare : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

/-- Represents a 1x1 square -/
def UnitSquare : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

/-- Checks if a point is inside a triangle -/
def isInTriangle (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Checks if a set is completely covered by a triangle -/
def isCompletelyCovered (s : Set (ℝ × ℝ)) (t : Triangle) : Prop :=
  ∀ p ∈ s, isInTriangle p t

/-- Checks if a set can be placed inside a triangle -/
def canBePlacedInside (s : Set (ℝ × ℝ)) (t : Triangle) : Prop := sorry

theorem checkered_square_covering (t1 t2 : Triangle) 
  (h : ∀ p ∈ CheckeredSquare, isInTriangle p t1 ∨ isInTriangle p t2) :
  (∃ cell : Set (ℝ × ℝ), cell ⊆ CheckeredSquare ∧ 
    ¬(isCompletelyCovered cell t1 ∨ isCompletelyCovered cell t2)) ∧
  (canBePlacedInside UnitSquare t1 ∨ canBePlacedInside UnitSquare t2) := by
  sorry

end checkered_square_covering_l1718_171820


namespace susie_earnings_l1718_171877

def slice_price : ℕ := 3
def whole_pizza_price : ℕ := 15
def slices_sold : ℕ := 24
def whole_pizzas_sold : ℕ := 3

theorem susie_earnings : 
  slice_price * slices_sold + whole_pizza_price * whole_pizzas_sold = 117 := by
  sorry

end susie_earnings_l1718_171877


namespace profit_scenario_theorem_l1718_171889

/-- Represents the profit scenarios for Bill's product sales -/
structure ProfitScenarios where
  original_purchase_price : ℝ
  original_profit_rate : ℝ
  second_purchase_discount : ℝ
  second_profit_rate : ℝ
  second_additional_profit : ℝ
  third_purchase_discount : ℝ
  third_profit_rate : ℝ
  third_additional_profit : ℝ

/-- Calculates the selling prices for each scenario given the profit conditions -/
def calculate_selling_prices (s : ProfitScenarios) : ℝ × ℝ × ℝ :=
  let original_selling_price := s.original_purchase_price * (1 + s.original_profit_rate)
  let second_selling_price := original_selling_price + s.second_additional_profit
  let third_selling_price := original_selling_price + s.third_additional_profit
  (original_selling_price, second_selling_price, third_selling_price)

/-- Theorem stating that given the profit conditions, the selling prices are as calculated -/
theorem profit_scenario_theorem (s : ProfitScenarios) 
  (h1 : s.original_profit_rate = 0.1)
  (h2 : s.second_purchase_discount = 0.1)
  (h3 : s.second_profit_rate = 0.3)
  (h4 : s.second_additional_profit = 35)
  (h5 : s.third_purchase_discount = 0.15)
  (h6 : s.third_profit_rate = 0.5)
  (h7 : s.third_additional_profit = 70) :
  calculate_selling_prices s = (550, 585, 620) := by
  sorry

end profit_scenario_theorem_l1718_171889


namespace claires_calculation_l1718_171831

theorem claires_calculation (a b c d f : ℚ) : 
  a = 2 ∧ b = 3 ∧ c = 4 ∧ d = 5 →
  (a + b - c + d - f = a + (b - (c * (d - f)))) →
  f = 21/5 := by sorry

end claires_calculation_l1718_171831


namespace piece_exits_at_A2_l1718_171828

/-- Represents the directions a piece can move on the grid -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a cell on the 4x4 grid -/
structure Cell :=
  (row : Fin 4)
  (col : Fin 4)

/-- Represents the state of the grid -/
structure GridState :=
  (currentCell : Cell)
  (arrows : Cell → Direction)

/-- Defines a single move on the grid -/
def move (state : GridState) : GridState :=
  sorry

/-- Checks if a cell is on the boundary of the grid -/
def isOnBoundary (cell : Cell) : Bool :=
  sorry

/-- Simulates the movement of the piece until it reaches the boundary -/
def simulateUntilExit (initialState : GridState) : Cell :=
  sorry

/-- The main theorem to prove -/
theorem piece_exits_at_A2 :
  let initialState : GridState := {
    currentCell := { row := 2, col := 1 },  -- C2 in 0-indexed
    arrows := sorry  -- Initial arrow configuration
  }
  let exitCell := simulateUntilExit initialState
  exitCell = { row := 0, col := 1 }  -- A2 in 0-indexed
  :=
sorry

end piece_exits_at_A2_l1718_171828


namespace sum_reciprocals_inequality_l1718_171881

theorem sum_reciprocals_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hsum : a + b + c ≤ 3) : 
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) ≥ 3 / 2 := by
  sorry

end sum_reciprocals_inequality_l1718_171881


namespace solution_equivalence_l1718_171837

-- Define the set of points satisfying the original equation
def S : Set (ℝ × ℝ) := {p | |p.1| + |p.2| = p.1^2}

-- Define the set of points as described in the solution
def T : Set (ℝ × ℝ) := 
  {(0, 0)} ∪ 
  {p | p.1 ≥ 1 ∧ (p.2 = p.1^2 - p.1 ∨ p.2 = -(p.1^2 - p.1))} ∪
  {p | p.1 ≤ -1 ∧ (p.2 = p.1^2 + p.1 ∨ p.2 = -(p.1^2 + p.1))}

-- Theorem statement
theorem solution_equivalence : S = T := by sorry

end solution_equivalence_l1718_171837


namespace substitution_result_l1718_171816

theorem substitution_result (x y : ℝ) :
  y = x - 1 ∧ x + 2*y = 7 → x + 2*x - 2 = 7 := by
  sorry

end substitution_result_l1718_171816


namespace largest_choir_size_l1718_171805

theorem largest_choir_size :
  ∃ (x r m : ℕ),
    (r * x + 3 = m) ∧
    ((r - 3) * (x + 2) = m) ∧
    (m < 150) ∧
    (∀ (x' r' m' : ℕ),
      (r' * x' + 3 = m') ∧
      ((r' - 3) * (x' + 2) = m') ∧
      (m' < 150) →
      m' ≤ m) ∧
    m = 759 :=
by sorry

end largest_choir_size_l1718_171805


namespace max_k_for_f_geq_kx_l1718_171802

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem max_k_for_f_geq_kx :
  ∃ (k : ℝ), k = 1 ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ k * x) ∧
  (∀ k' : ℝ, k' > k → ∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x < k' * x) :=
sorry

end max_k_for_f_geq_kx_l1718_171802


namespace puppy_kibble_percentage_proof_l1718_171855

/-- The percentage of vets recommending Puppy Kibble -/
def puppy_kibble_percentage : ℝ := 20

/-- The percentage of vets recommending Yummy Dog Kibble -/
def yummy_kibble_percentage : ℝ := 30

/-- The total number of vets in the state -/
def total_vets : ℕ := 1000

/-- The difference in number of vets recommending Yummy Dog Kibble vs Puppy Kibble -/
def vet_difference : ℕ := 100

theorem puppy_kibble_percentage_proof :
  puppy_kibble_percentage = 20 ∧
  yummy_kibble_percentage = 30 ∧
  total_vets = 1000 ∧
  vet_difference = 100 →
  puppy_kibble_percentage * (total_vets : ℝ) / 100 + vet_difference = 
  yummy_kibble_percentage * (total_vets : ℝ) / 100 :=
by sorry

end puppy_kibble_percentage_proof_l1718_171855


namespace cylinder_from_constant_radius_l1718_171896

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying r = c in cylindrical coordinates -/
def CylindricalSet (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.r = c}

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) : Prop :=
  ∃ c : ℝ, c > 0 ∧ S = CylindricalSet c

theorem cylinder_from_constant_radius (c : ℝ) (h : c > 0) :
  IsCylinder (CylindricalSet c) := by
  sorry

#check cylinder_from_constant_radius

end cylinder_from_constant_radius_l1718_171896


namespace all_routes_have_eight_stations_l1718_171874

/-- Represents a bus route in the city -/
structure BusRoute where
  stations : Set Nat
  station_count : Nat

/-- Represents the city's bus network -/
structure BusNetwork where
  routes : Finset BusRoute
  route_count : Nat

/-- Conditions for the bus network -/
def valid_network (n : BusNetwork) : Prop :=
  -- There are 57 bus routes
  n.route_count = 57 ∧
  -- Any two routes share exactly one station
  ∀ r1 r2 : BusRoute, r1 ∈ n.routes ∧ r2 ∈ n.routes ∧ r1 ≠ r2 →
    ∃! s : Nat, s ∈ r1.stations ∧ s ∈ r2.stations ∧
  -- Each route has at least 3 stations
  ∀ r : BusRoute, r ∈ n.routes → r.station_count ≥ 3 ∧
  -- From any station, it's possible to reach any other station without changing buses
  ∀ s1 s2 : Nat, ∃ r : BusRoute, r ∈ n.routes ∧ s1 ∈ r.stations ∧ s2 ∈ r.stations

/-- The main theorem to prove -/
theorem all_routes_have_eight_stations (n : BusNetwork) (h : valid_network n) :
  ∀ r : BusRoute, r ∈ n.routes → r.station_count = 8 := by
  sorry

end all_routes_have_eight_stations_l1718_171874


namespace hostel_problem_solution_l1718_171865

/-- Represents the hostel problem with given initial conditions -/
structure HostelProblem where
  initial_students : ℕ
  budget_decrease : ℕ
  expenditure_increase : ℕ
  new_total_expenditure : ℕ

/-- Calculates the number of new students given a HostelProblem -/
def new_students (problem : HostelProblem) : ℕ :=
  -- Implementation not provided, as per instructions
  sorry

/-- Theorem stating that for the given problem, 35 new students joined -/
theorem hostel_problem_solution :
  let problem : HostelProblem := {
    initial_students := 100,
    budget_decrease := 10,
    expenditure_increase := 400,
    new_total_expenditure := 5400
  }
  new_students problem = 35 := by
  sorry

end hostel_problem_solution_l1718_171865


namespace f_derivative_and_tangent_lines_l1718_171891

noncomputable def f (x : ℝ) : ℝ := (x - 1) * (x^2 + 1) + 1

theorem f_derivative_and_tangent_lines :
  (∃ f' : ℝ → ℝ, ∀ x, deriv f x = f' x ∧ f' x = 3 * x^2 - 2 * x + 1) ∧
  (∃ t₁ t₂ : ℝ → ℝ,
    (∀ x, t₁ x = x) ∧
    (∀ x, t₂ x = 2 * x - 1) ∧
    (t₁ 1 = f 1 ∧ t₂ 1 = f 1) ∧
    (∃ x₀, deriv f x₀ = deriv t₁ x₀ ∧ f x₀ = t₁ x₀) ∧
    (∃ x₁, deriv f x₁ = deriv t₂ x₁ ∧ f x₁ = t₂ x₁)) :=
by sorry

end f_derivative_and_tangent_lines_l1718_171891


namespace amusement_park_ticket_price_l1718_171894

/-- Given the following conditions for an amusement park admission:
  * The total cost for admission tickets is $720
  * The price of an adult ticket is $15
  * There are 15 children in the group
  * There are 25 more adults than children
  Prove that the price of a child ticket is $8 -/
theorem amusement_park_ticket_price 
  (total_cost : ℕ) 
  (adult_price : ℕ) 
  (num_children : ℕ) 
  (adult_child_diff : ℕ) 
  (h1 : total_cost = 720)
  (h2 : adult_price = 15)
  (h3 : num_children = 15)
  (h4 : adult_child_diff = 25) :
  ∃ (child_price : ℕ), 
    child_price = 8 ∧ 
    total_cost = adult_price * (num_children + adult_child_diff) + child_price * num_children :=
by sorry

end amusement_park_ticket_price_l1718_171894


namespace calculator_probability_l1718_171863

/-- Represents a 7-segment calculator display --/
def SegmentDisplay := Fin 7 → Bool

/-- The probability of a segment being illuminated --/
def segmentProbability : ℚ := 1/2

/-- The total number of possible displays --/
def totalDisplays : ℕ := 2^7

/-- The number of valid digit displays (0-9) --/
def validDigitDisplays : ℕ := 10

/-- The probability of displaying a valid digit --/
def validDigitProbability : ℚ := validDigitDisplays / totalDisplays

theorem calculator_probability (a b : ℕ) (h : validDigitProbability = a / b) :
  9 * a + 2 * b = 173 := by
  sorry

end calculator_probability_l1718_171863


namespace min_value_implies_a_value_l1718_171850

/-- The function f(x) = x^2 + ax - 1 has a minimum value of -2 on the interval [0, 3] -/
def has_min_value_neg_two (a : ℝ) : Prop :=
  ∃ x₀ ∈ Set.Icc 0 3, ∀ x ∈ Set.Icc 0 3, x^2 + a*x - 1 ≥ x₀^2 + a*x₀ - 1 ∧ x₀^2 + a*x₀ - 1 = -2

/-- If f(x) = x^2 + ax - 1 has a minimum value of -2 on [0, 3], then a = -10/3 -/
theorem min_value_implies_a_value (a : ℝ) :
  has_min_value_neg_two a → a = -10/3 := by
  sorry

end min_value_implies_a_value_l1718_171850


namespace quadratic_factorization_l1718_171862

theorem quadratic_factorization (c d : ℕ) (hc : c > d) :
  (∀ x, x^2 - 18*x + 72 = (x - c)*(x - d)) →
  4*d - c = 12 := by
  sorry

end quadratic_factorization_l1718_171862


namespace tangent_circles_bound_l1718_171871

/-- The maximum number of pairs of tangent circles for n circles -/
def l (n : ℕ) : ℕ :=
  match n with
  | 3 => 3
  | 4 => 5
  | 5 => 7
  | 7 => 12
  | 8 => 14
  | 9 => 16
  | 10 => 19
  | _ => 3 * n - 11

/-- Theorem: For n ≥ 9, the maximum number of pairs of tangent circles is at most 3n - 11 -/
theorem tangent_circles_bound (n : ℕ) (h : n ≥ 9) : l n ≤ 3 * n - 11 := by
  sorry

end tangent_circles_bound_l1718_171871


namespace room_entry_exit_ways_l1718_171825

/-- The number of doors in the room -/
def num_doors : ℕ := 4

/-- The number of times the person enters the room -/
def num_entries : ℕ := 1

/-- The number of times the person exits the room -/
def num_exits : ℕ := 1

/-- The total number of ways to enter and exit the room -/
def total_ways : ℕ := num_doors ^ (num_entries + num_exits)

theorem room_entry_exit_ways :
  total_ways = 16 := by sorry

end room_entry_exit_ways_l1718_171825


namespace train_speed_l1718_171842

/-- Calculates the speed of a train given its composition and the time it takes to cross a bridge. -/
theorem train_speed (num_carriages : ℕ) (carriage_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  num_carriages = 24 →
  carriage_length = 60 →
  bridge_length = 3.5 →
  crossing_time = 5 / 60 →
  let total_train_length := (num_carriages + 1 : ℝ) * carriage_length
  let total_distance := total_train_length / 1000 + bridge_length
  let speed := total_distance / crossing_time
  speed = 60 := by
    sorry


end train_speed_l1718_171842


namespace simplify_expression_1_simplify_expression_2_l1718_171824

-- First expression
theorem simplify_expression_1 (x y : ℝ) :
  2 - x + 3*y + 8*x - 5*y - 6 = 7*x - 2*y - 4 := by sorry

-- Second expression
theorem simplify_expression_2 (a b : ℝ) :
  15*a^2*b - 12*a*b^2 + 12 - 4*a^2*b - 18 + 8*a*b^2 = 11*a^2*b - 4*a*b^2 - 6 := by sorry

end simplify_expression_1_simplify_expression_2_l1718_171824


namespace total_weight_of_diamonds_and_jades_l1718_171887

/-- Given that 5 diamonds weigh 100 g and a jade is 10 g heavier than a diamond,
    prove that the total weight of 4 diamonds and 2 jades is 140 g. -/
theorem total_weight_of_diamonds_and_jades :
  let diamond_weight : ℚ := 100 / 5
  let jade_weight : ℚ := diamond_weight + 10
  4 * diamond_weight + 2 * jade_weight = 140 := by
  sorry

end total_weight_of_diamonds_and_jades_l1718_171887


namespace polynomial_equality_implies_sum_l1718_171801

theorem polynomial_equality_implies_sum (m n : ℝ) : 
  (∀ x : ℝ, (x + 8) * (x - 1) = x^2 + m*x + n) → m + n = -1 :=
by sorry

end polynomial_equality_implies_sum_l1718_171801


namespace intersection_of_A_and_B_l1718_171819

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {x | x ≤ 0 ∨ x ≥ 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | x ≤ 0} := by
  sorry

end intersection_of_A_and_B_l1718_171819


namespace pythagorean_proof_l1718_171869

theorem pythagorean_proof (a b : ℝ) (h1 : 0 < a) (h2 : a < b) : 
  b^2 = 13 * (b - a)^2 → a / b = 2 / 3 := by
sorry

end pythagorean_proof_l1718_171869


namespace circle_properties_l1718_171893

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 14*y + 45 = 0

-- Define point Q
def Q : ℝ × ℝ := (-2, 3)

-- Define point P
def P : ℝ × ℝ := (4, 5)

-- Theorem statement
theorem circle_properties :
  -- P is on circle C
  C P.1 P.2 ∧
  -- Distance PQ
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * Real.sqrt 10 ∧
  -- Slope of PQ
  (P.2 - Q.2) / (P.1 - Q.1) = 1/3 ∧
  -- Maximum and minimum distances from Q to any point on C
  (∀ M : ℝ × ℝ, C M.1 M.2 → 
    Real.sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2) ≤ 6 * Real.sqrt 2 ∧
    Real.sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2) ≥ 2 * Real.sqrt 2) :=
by
  sorry

end circle_properties_l1718_171893


namespace gas_usage_difference_l1718_171880

theorem gas_usage_difference (felicity_gas adhira_gas : ℕ) : 
  felicity_gas = 23 →
  felicity_gas + adhira_gas = 30 →
  4 * adhira_gas - felicity_gas = 5 :=
by
  sorry

end gas_usage_difference_l1718_171880


namespace substitution_result_l1718_171858

theorem substitution_result (x y : ℝ) :
  (4 * x + 5 * y = 7) ∧ (y = 2 * x - 1) →
  (4 * x + 10 * x - 5 = 7) :=
by sorry

end substitution_result_l1718_171858


namespace cos_2alpha_plus_pi_third_l1718_171832

theorem cos_2alpha_plus_pi_third (α : ℝ) (h : Real.sin (α - π/3) = 2/3) :
  Real.cos (2*α + π/3) = -1/9 := by
  sorry

end cos_2alpha_plus_pi_third_l1718_171832


namespace units_digit_characteristic_l1718_171879

/-- Given a positive even integer p, if the units digit of p^3 minus the units digit of p^2 is 0
    and the units digit of p + 1 is 7, then the units digit of p is 6. -/
theorem units_digit_characteristic (p : ℕ) : 
  p > 0 → 
  Even p → 
  (p^3 % 10 - p^2 % 10) % 10 = 0 → 
  (p + 1) % 10 = 7 → 
  p % 10 = 6 := by
  sorry

end units_digit_characteristic_l1718_171879


namespace calculation_proof_l1718_171818

theorem calculation_proof :
  (Real.sqrt 8 - Real.sqrt 2 - Real.sqrt (1/3) * Real.sqrt 6 = 0) ∧
  (Real.sqrt 15 / Real.sqrt 3 + (Real.sqrt 5 - 1)^2 = 6 - Real.sqrt 5) :=
by sorry

end calculation_proof_l1718_171818


namespace product_of_positive_real_solutions_l1718_171857

def solutions (x : ℂ) : Prop := x^8 = -256

def positive_real_part (z : ℂ) : Prop := z.re > 0

theorem product_of_positive_real_solutions :
  ∃ (S : Finset ℂ), 
    (∀ z ∈ S, solutions z ∧ positive_real_part z) ∧ 
    (∀ z, solutions z ∧ positive_real_part z → z ∈ S) ∧
    S.prod id = 8 :=
sorry

end product_of_positive_real_solutions_l1718_171857


namespace expression_is_perfect_square_l1718_171875

theorem expression_is_perfect_square (x y z : ℤ) (A : ℤ) :
  A = x * y + y * z + z * x →
  A = (x + 1) * (y - 2) + (y - 2) * (z - 2) + (z - 2) * (x + 1) →
  ∃ k : ℤ, (-1) * A = k^2 := by
  sorry

end expression_is_perfect_square_l1718_171875
