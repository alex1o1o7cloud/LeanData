import Mathlib

namespace NUMINAMATH_CALUDE_series_sum_equals_one_l15_1580

/-- The sum of the series ∑(k=0 to ∞) 2^(2^k) / (4^(2^k) - 1) is equal to 1 -/
theorem series_sum_equals_one : 
  ∑' (k : ℕ), (2^(2^k)) / ((4^(2^k)) - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_l15_1580


namespace NUMINAMATH_CALUDE_polynomial_B_value_l15_1598

def polynomial (z A B C D : ℝ) : ℝ := z^6 - 12*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 36

theorem polynomial_B_value (A B C D : ℝ) :
  (∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (∀ z : ℝ, polynomial z A B C D = (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) * (z - r₅) * (z - r₆)) ∧
    r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 12) →
  B = -122 := by
sorry

end NUMINAMATH_CALUDE_polynomial_B_value_l15_1598


namespace NUMINAMATH_CALUDE_chocolate_division_l15_1569

theorem chocolate_division (total : ℚ) (piles : ℕ) (keep_fraction : ℚ) :
  total = 72 / 7 →
  piles = 6 →
  keep_fraction = 1 / 3 →
  (total / piles) * (1 - keep_fraction) = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l15_1569


namespace NUMINAMATH_CALUDE_sector_radius_range_l15_1530

theorem sector_radius_range (a : ℝ) (m : ℝ) (h1 : a > 0) (h2 : 0 < m) (h3 : m < 360) :
  ∃ R : ℝ, a / (2 * (1 + π)) < R ∧ R < a / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_range_l15_1530


namespace NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l15_1509

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k > 24 ∧ ¬(k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
  ∀ m : ℤ, m ≤ 24 → (m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l15_1509


namespace NUMINAMATH_CALUDE_no_universal_divisibility_l15_1525

/-- Represents a nonzero digit (1-9) -/
def NonzeroDigit := {d : Nat // d ≥ 1 ∧ d ≤ 9}

/-- Concatenates three numbers to form a new number -/
def concat3 (a : NonzeroDigit) (n : Nat) (b : NonzeroDigit) : Nat :=
  100 * a.val + 10 * n + b.val

/-- Concatenates two numbers to form a new number -/
def concat2 (a b : NonzeroDigit) : Nat :=
  10 * a.val + b.val

/-- Statement: There does not exist a natural number n such that
    for all nonzero digits a and b, concat3 a n b is divisible by concat2 a b -/
theorem no_universal_divisibility :
  ¬ ∃ n : Nat, ∀ (a b : NonzeroDigit), (concat3 a n b) % (concat2 a b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_universal_divisibility_l15_1525


namespace NUMINAMATH_CALUDE_cistern_filling_time_l15_1578

theorem cistern_filling_time (T : ℝ) : 
  T > 0 →  -- T must be positive
  (1 / 4 : ℝ) - (1 / T) = (3 / 28 : ℝ) → 
  T = 7 :=
by sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l15_1578


namespace NUMINAMATH_CALUDE_chairs_per_table_l15_1537

theorem chairs_per_table (indoor_tables outdoor_tables total_chairs : ℕ) 
  (h1 : indoor_tables = 8)
  (h2 : outdoor_tables = 12)
  (h3 : total_chairs = 60) :
  ∃ (chairs_per_table : ℕ), 
    chairs_per_table * (indoor_tables + outdoor_tables) = total_chairs ∧ 
    chairs_per_table = 3 := by
  sorry

end NUMINAMATH_CALUDE_chairs_per_table_l15_1537


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_sum_l15_1555

theorem rectangle_area_perimeter_sum (a b : ℕ+) : 
  let A := (a : ℝ) * (b : ℝ)
  let P := 2 * (a : ℝ) + 2 * (b : ℝ) + 2
  A + P ≠ 114 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_sum_l15_1555


namespace NUMINAMATH_CALUDE_first_worker_time_l15_1538

/-- Given two workers loading a truck, prove that the first worker's time is 5 hours. -/
theorem first_worker_time (T : ℝ) : 
  T > 0 →  -- The time must be positive
  (1 / T + 1 / 4 : ℝ) = 1 / 2.2222222222222223 → 
  T = 5 := by 
sorry

end NUMINAMATH_CALUDE_first_worker_time_l15_1538


namespace NUMINAMATH_CALUDE_identify_brothers_l15_1546

-- Define the brothers
inductive Brother
| trulya
| tralya

-- Define a function to represent whether a brother tells the truth
def tellsTruth : Brother → Prop
| Brother.trulya => true
| Brother.tralya => false

-- Define the statements made by the brothers
def firstBrotherStatement (first second : Brother) : Prop :=
  first = Brother.trulya

def secondBrotherStatement (first second : Brother) : Prop :=
  second = Brother.tralya

def cardSuitStatement : Prop := false  -- Cards are not of the same suit

-- The main theorem
theorem identify_brothers :
  ∃ (first second : Brother),
    first ≠ second ∧
    (tellsTruth first → firstBrotherStatement first second) ∧
    (tellsTruth second → secondBrotherStatement first second) ∧
    (tellsTruth first → cardSuitStatement) ∧
    first = Brother.tralya ∧
    second = Brother.trulya :=
  sorry

end NUMINAMATH_CALUDE_identify_brothers_l15_1546


namespace NUMINAMATH_CALUDE_sum_square_bound_l15_1531

/-- The sum of integers from 1 to n -/
def sum_to (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Predicate for a natural number being a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem sum_square_bound :
  ∀ K : ℕ, K > 0 →
    (is_perfect_square (sum_to K) ∧
     ∃ N : ℕ, sum_to K = N * N ∧ N + K < 120) ↔
    (K = 1 ∨ K = 8 ∨ K = 49) :=
sorry

end NUMINAMATH_CALUDE_sum_square_bound_l15_1531


namespace NUMINAMATH_CALUDE_balls_triangle_to_square_l15_1528

theorem balls_triangle_to_square (n : ℕ) (h1 : n * (n + 1) / 2 = 1176) :
  let square_side := n - 8
  square_side * square_side - n * (n + 1) / 2 = 424 := by
  sorry

end NUMINAMATH_CALUDE_balls_triangle_to_square_l15_1528


namespace NUMINAMATH_CALUDE_x_values_l15_1596

theorem x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 12) (h2 : y + 1 / x = 7 / 18) :
  x = 6 + Real.sqrt 5 ∨ x = 6 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_x_values_l15_1596


namespace NUMINAMATH_CALUDE_solve_for_m_l15_1588

theorem solve_for_m : ∃ m : ℕ, (2022^2 - 4) * (2021^2 - 4) = 2024 * 2020 * 2019 * m ∧ m = 2023 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l15_1588


namespace NUMINAMATH_CALUDE_largest_rational_l15_1565

theorem largest_rational (a b c d : ℚ) : 
  a = -1 → b = 0 → c = -3 → d = (8 : ℚ) / 100 → 
  max a (max b (max c d)) = d := by
  sorry

end NUMINAMATH_CALUDE_largest_rational_l15_1565


namespace NUMINAMATH_CALUDE_circle_pentagon_visibility_l15_1581

noncomputable def radius_of_circle (side_length : ℝ) (probability : ℝ) : ℝ :=
  (side_length * Real.sqrt ((5 - 2 * Real.sqrt 5) / 5)) / (2 * 0.9511)

theorem circle_pentagon_visibility 
  (r : ℝ) 
  (side_length : ℝ) 
  (probability : ℝ) 
  (h1 : side_length = 3) 
  (h2 : probability = 1/2) :
  r = radius_of_circle side_length probability :=
by sorry

end NUMINAMATH_CALUDE_circle_pentagon_visibility_l15_1581


namespace NUMINAMATH_CALUDE_phi_value_l15_1529

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.cos (Real.sqrt 3 * x + φ)

noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := f x φ + (deriv (f · φ)) x

theorem phi_value (φ : ℝ) 
  (h1 : -π < φ ∧ φ < 0) 
  (h2 : ∀ x, g x φ = g (-x) φ) : 
  φ = -π/3 := by
sorry

end NUMINAMATH_CALUDE_phi_value_l15_1529


namespace NUMINAMATH_CALUDE_student_divisor_problem_l15_1592

theorem student_divisor_problem (dividend : ℕ) (student_answer : ℕ) (correct_answer : ℕ) (correct_divisor : ℕ) : 
  student_answer = 24 →
  correct_answer = 32 →
  correct_divisor = 36 →
  dividend / correct_divisor = correct_answer →
  ∃ (student_divisor : ℕ), 
    dividend / student_divisor = student_answer ∧ 
    student_divisor = 48 :=
by sorry

end NUMINAMATH_CALUDE_student_divisor_problem_l15_1592


namespace NUMINAMATH_CALUDE_rachel_piggy_bank_l15_1583

/-- The amount of money originally in Rachel's piggy bank -/
def original_amount : ℕ := 5

/-- The amount of money Rachel took from her piggy bank -/
def amount_taken : ℕ := 2

/-- The amount of money left in Rachel's piggy bank -/
def amount_left : ℕ := original_amount - amount_taken

theorem rachel_piggy_bank : amount_left = 3 := by
  sorry

end NUMINAMATH_CALUDE_rachel_piggy_bank_l15_1583


namespace NUMINAMATH_CALUDE_inscribed_circle_angle_theorem_l15_1521

/-- A triangle with an inscribed circle --/
structure InscribedCircleTriangle where
  /-- The angle at the tangent point on side BC --/
  angle_bc : ℝ
  /-- The angle at the tangent point on side CA --/
  angle_ca : ℝ
  /-- The angle at the tangent point on side AB --/
  angle_ab : ℝ
  /-- The sum of angles at tangent points is 360° --/
  sum_angles : angle_bc + angle_ca + angle_ab = 360

/-- Theorem: If the angles at tangent points are 120°, 130°, and θ°, then θ = 110° --/
theorem inscribed_circle_angle_theorem (t : InscribedCircleTriangle) 
    (h1 : t.angle_bc = 120) (h2 : t.angle_ca = 130) : t.angle_ab = 110 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_angle_theorem_l15_1521


namespace NUMINAMATH_CALUDE_nancy_crayon_packs_l15_1526

theorem nancy_crayon_packs (total_crayons : ℕ) (crayons_per_pack : ℕ) 
  (h1 : total_crayons = 615) 
  (h2 : crayons_per_pack = 15) : 
  total_crayons / crayons_per_pack = 41 := by
  sorry

end NUMINAMATH_CALUDE_nancy_crayon_packs_l15_1526


namespace NUMINAMATH_CALUDE_garage_spokes_count_l15_1552

/-- Represents a bicycle with two wheels -/
structure Bicycle where
  front_spokes : ℕ
  back_spokes : ℕ

/-- Represents a tricycle with three wheels -/
structure Tricycle where
  front_spokes : ℕ
  middle_spokes : ℕ
  back_spokes : ℕ

/-- The total number of spokes in all bicycles and the tricycle -/
def total_spokes (bikes : List Bicycle) (trike : Tricycle) : ℕ :=
  (bikes.map (fun b => b.front_spokes + b.back_spokes)).sum +
  (trike.front_spokes + trike.middle_spokes + trike.back_spokes)

theorem garage_spokes_count :
  let bikes : List Bicycle := [
    { front_spokes := 16, back_spokes := 18 },
    { front_spokes := 20, back_spokes := 22 },
    { front_spokes := 24, back_spokes := 26 },
    { front_spokes := 28, back_spokes := 30 }
  ]
  let trike : Tricycle := { front_spokes := 32, middle_spokes := 34, back_spokes := 36 }
  total_spokes bikes trike = 286 := by
  sorry


end NUMINAMATH_CALUDE_garage_spokes_count_l15_1552


namespace NUMINAMATH_CALUDE_square_equation_solution_l15_1544

theorem square_equation_solution :
  ∃! x : ℚ, (2015 + x)^2 = x^2 ∧ x = -2015/2 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l15_1544


namespace NUMINAMATH_CALUDE_inequality_proof_l15_1570

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b^5 + b * c^5 + c * a^5 ≥ a * b * c * (a^2 * b + b^2 * c + c^2 * a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l15_1570


namespace NUMINAMATH_CALUDE_factorial_sum_division_l15_1566

theorem factorial_sum_division (n : ℕ) : (Nat.factorial 8 + Nat.factorial 9) / Nat.factorial 6 = 560 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_division_l15_1566


namespace NUMINAMATH_CALUDE_ratio_independence_l15_1591

/-- Two infinite increasing arithmetic progressions of positive numbers -/
def ArithmeticProgression (a : ℕ → ℚ) : Prop :=
  ∃ (first d : ℚ), first > 0 ∧ d > 0 ∧ ∀ k, a k = first + k * d

/-- The theorem statement -/
theorem ratio_independence
  (a b : ℕ → ℚ)
  (ha : ArithmeticProgression a)
  (hb : ArithmeticProgression b)
  (h_int_ratio : ∀ k, ∃ m : ℤ, a k = m * b k) :
  ∃ c : ℚ, ∀ k, a k = c * b k :=
sorry

end NUMINAMATH_CALUDE_ratio_independence_l15_1591


namespace NUMINAMATH_CALUDE_equal_distribution_of_drawings_l15_1573

/-- Given 54 animal drawings distributed equally among 6 neighbors, prove that each neighbor receives 9 drawings. -/
theorem equal_distribution_of_drawings (total_drawings : ℕ) (num_neighbors : ℕ) (drawings_per_neighbor : ℕ) : 
  total_drawings = 54 → 
  num_neighbors = 6 → 
  total_drawings = num_neighbors * drawings_per_neighbor →
  drawings_per_neighbor = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_of_drawings_l15_1573


namespace NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l15_1567

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem smallest_prime_12_less_than_square : 
  ∃ (n : ℕ) (k : ℕ), 
    n > 0 ∧ 
    is_prime n ∧ 
    n = k^2 - 12 ∧ 
    ∀ (m : ℕ) (j : ℕ), m > 0 → is_prime m → m = j^2 - 12 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l15_1567


namespace NUMINAMATH_CALUDE_largest_circle_tangent_to_line_l15_1514

/-- The largest circle with center (0,2) that is tangent to the line mx - y - 3m - 1 = 0 -/
theorem largest_circle_tangent_to_line (m : ℝ) :
  ∃! (r : ℝ), r > 0 ∧
    (∀ (x y : ℝ), x^2 + (y - 2)^2 = r^2 →
      ∃ (x₀ y₀ : ℝ), x₀^2 + (y₀ - 2)^2 = r^2 ∧
        m * x₀ - y₀ - 3 * m - 1 = 0) ∧
    (∀ (r' : ℝ), r' > r →
      ¬∃ (x y : ℝ), x^2 + (y - 2)^2 = r'^2 ∧
        m * x - y - 3 * m - 1 = 0) ∧
    r^2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_largest_circle_tangent_to_line_l15_1514


namespace NUMINAMATH_CALUDE_garden_ratio_maintenance_l15_1556

/-- Represents a garden with tulips and daisies -/
structure Garden where
  tulips : ℕ
  daisies : ℕ

/-- Calculates the number of tulips needed to maintain a 3:7 ratio with the given number of daisies -/
def tulipsForRatio (daisies : ℕ) : ℕ :=
  (3 * daisies + 6) / 7

theorem garden_ratio_maintenance (initial : Garden) (added_daisies : ℕ) :
  initial.daisies = 35 →
  added_daisies = 30 →
  (3 : ℚ) / 7 = initial.tulips / initial.daisies →
  tulipsForRatio (initial.daisies + added_daisies) = 28 := by
  sorry

end NUMINAMATH_CALUDE_garden_ratio_maintenance_l15_1556


namespace NUMINAMATH_CALUDE_sum_seven_terms_l15_1548

/-- An arithmetic sequence with specific terms. -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  second_term : a 2 = 5 / 3
  sixth_term : a 6 = -7 / 3

/-- The sum of the first n terms of an arithmetic sequence. -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The theorem stating that the sum of the first 7 terms is -7/3. -/
theorem sum_seven_terms (seq : ArithmeticSequence) : sum_n_terms seq 7 = -7 / 3 := by
  sorry


end NUMINAMATH_CALUDE_sum_seven_terms_l15_1548


namespace NUMINAMATH_CALUDE_john_twice_frank_age_l15_1512

/-- Given that Frank is 15 years younger than John and Frank will be 16 in 4 years,
    prove that John will be twice as old as Frank in 3 years. -/
theorem john_twice_frank_age (frank_age john_age x : ℕ) : 
  john_age = frank_age + 15 →
  frank_age + 4 = 16 →
  john_age + x = 2 * (frank_age + x) →
  x = 3 := by sorry

end NUMINAMATH_CALUDE_john_twice_frank_age_l15_1512


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l15_1589

/-- The line kx - y + 1 = 3k passes through the point (3, 1) for all real k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k * 3 : ℝ) - 1 + 1 = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l15_1589


namespace NUMINAMATH_CALUDE_union_complement_equal_set_l15_1535

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_equal_set : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equal_set_l15_1535


namespace NUMINAMATH_CALUDE_sequence_inequality_l15_1593

theorem sequence_inequality (a : ℕ → ℝ) (h_nonneg : ∀ n, a n ≥ 0) 
  (h_ineq : ∀ m n, a (m + n) ≤ a m + a n) (m n : ℕ) (h_ge : n ≥ m) :
  a n ≤ m * a 1 + (n / m - 1) * a m :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l15_1593


namespace NUMINAMATH_CALUDE_factor_expression_l15_1582

theorem factor_expression (y : ℝ) : 
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 6 * y^4 + 9) = 6 * (2 * y^6 + 7 * y^4 - 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l15_1582


namespace NUMINAMATH_CALUDE_factorization_3x2_minus_12y2_l15_1586

theorem factorization_3x2_minus_12y2 (x y : ℝ) :
  3 * x^2 - 12 * y^2 = 3 * (x - 2*y) * (x + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3x2_minus_12y2_l15_1586


namespace NUMINAMATH_CALUDE_visible_sum_range_l15_1547

/-- Represents a die with 6 faces -/
structure Die :=
  (faces : Fin 6 → Nat)
  (opposite_sum : ∀ i : Fin 6, faces i + faces (5 - i) = 7)
  (face_range : ∀ i : Fin 6, 1 ≤ faces i ∧ faces i ≤ 6)

/-- Represents the larger 3x3x3 cube made of 27 dice -/
def LargeCube := Fin 3 → Fin 3 → Fin 3 → Die

/-- Calculates the sum of visible face values on the larger cube -/
def visible_sum (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating the range of possible sums of visible face values -/
theorem visible_sum_range (cube : LargeCube) :
  90 ≤ visible_sum cube ∧ visible_sum cube ≤ 288 :=
sorry

end NUMINAMATH_CALUDE_visible_sum_range_l15_1547


namespace NUMINAMATH_CALUDE_new_city_buildings_count_l15_1571

/-- Calculates the total number of buildings for the new city project --/
def new_city_buildings (pittsburgh_stores : ℕ) (pittsburgh_hospitals : ℕ) (pittsburgh_schools : ℕ) (pittsburgh_police : ℕ) : ℕ :=
  (pittsburgh_stores / 2) + (pittsburgh_hospitals * 2) + (pittsburgh_schools - 50) + (pittsburgh_police + 5)

/-- Theorem stating that the total number of buildings for the new city is 2175 --/
theorem new_city_buildings_count : 
  new_city_buildings 2000 500 200 20 = 2175 := by
  sorry

end NUMINAMATH_CALUDE_new_city_buildings_count_l15_1571


namespace NUMINAMATH_CALUDE_infinitely_many_squares_2012_2013_divisibility_condition_l15_1518

-- Part (a)
theorem infinitely_many_squares_2012_2013 :
  ∀ k : ℕ, ∃ t > k, ∃ a b : ℕ,
    2012 * t + 1 = a^2 ∧ 2013 * t + 1 = b^2 :=
sorry

-- Part (b)
theorem divisibility_condition (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (∃ x y : ℕ, m * n + 1 = x^2 ∧ m * n + n + 1 = y^2) →
  8 * (2 * m + 1) ∣ n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_squares_2012_2013_divisibility_condition_l15_1518


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l15_1524

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point (x, y) lies on a line -/
def lies_on (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- The problem statement -/
theorem line_through_point_parallel_to_line :
  ∃ (l : Line),
    lies_on (-1) 2 l ∧
    parallel l { a := 2, b := -3, c := 4 } ∧
    l = { a := 2, b := -3, c := 8 } := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l15_1524


namespace NUMINAMATH_CALUDE_octal_7421_to_decimal_l15_1576

def octal_to_decimal (octal : ℕ) : ℕ :=
  let digits := [7, 4, 2, 1]
  (List.zipWith (λ (d : ℕ) (p : ℕ) => d * (8 ^ p)) digits (List.range 4)).sum

theorem octal_7421_to_decimal :
  octal_to_decimal 7421 = 1937 := by
  sorry

end NUMINAMATH_CALUDE_octal_7421_to_decimal_l15_1576


namespace NUMINAMATH_CALUDE_sine_cosine_sum_equals_root_two_over_two_l15_1510

theorem sine_cosine_sum_equals_root_two_over_two :
  Real.sin (30 * π / 180) * Real.cos (15 * π / 180) + 
  Real.cos (30 * π / 180) * Real.sin (15 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_equals_root_two_over_two_l15_1510


namespace NUMINAMATH_CALUDE_hot_dog_purchase_l15_1553

theorem hot_dog_purchase (cost_per_hot_dog : ℕ) (total_paid : ℕ) (h1 : cost_per_hot_dog = 50) (h2 : total_paid = 300) :
  total_paid / cost_per_hot_dog = 6 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_purchase_l15_1553


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l15_1502

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence condition
  (a 1 + a 2 + a 3 = 8) →                    -- first condition
  (a 4 + a 5 + a 6 = -4) →                   -- second condition
  (a 7 + a 8 + a 9 = 2) :=                   -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l15_1502


namespace NUMINAMATH_CALUDE_last_digit_of_seven_to_seventh_l15_1597

theorem last_digit_of_seven_to_seventh : 7^7 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_seven_to_seventh_l15_1597


namespace NUMINAMATH_CALUDE_parabola_vertex_condition_l15_1519

/-- A parabola with equation y = x^2 + 2x + a -/
structure Parabola where
  a : ℝ

/-- The vertex of a parabola y = x^2 + 2x + a is below the x-axis -/
def vertex_below_x_axis (p : Parabola) : Prop :=
  let x := -1  -- x-coordinate of the vertex
  let y := x^2 + 2*x + p.a  -- y-coordinate of the vertex
  y < 0

/-- If the vertex of the parabola y = x^2 + 2x + a is below the x-axis, then a < 1 -/
theorem parabola_vertex_condition (p : Parabola) : vertex_below_x_axis p → p.a < 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_condition_l15_1519


namespace NUMINAMATH_CALUDE_greater_a_than_c_l15_1539

theorem greater_a_than_c (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : (a - b) * (b - c) * (c - a) > 0) : 
  a > c := by
  sorry

end NUMINAMATH_CALUDE_greater_a_than_c_l15_1539


namespace NUMINAMATH_CALUDE_polynomial_factorization_l15_1506

theorem polynomial_factorization (x : ℝ) : 
  x^2 + 4*x + 4 - 81*x^4 = (-9*x^2 + x + 2) * (9*x^2 + x + 2) := by
  sorry

#check polynomial_factorization

end NUMINAMATH_CALUDE_polynomial_factorization_l15_1506


namespace NUMINAMATH_CALUDE_symmetry_about_origin_l15_1532

/-- A point on the graph of y = 3^x -/
structure PointOn3x where
  x : ℝ
  y : ℝ
  h : y = 3^x

/-- A point on the graph of y = -3^(-x) -/
structure PointOnNeg3NegX where
  x : ℝ
  y : ℝ
  h : y = -3^(-x)

/-- The condition given in the problem -/
axiom symmetry_condition {p : PointOn3x} :
  ∃ (q : PointOnNeg3NegX), q.x = -p.x ∧ q.y = -p.y

/-- The theorem to be proved -/
theorem symmetry_about_origin :
  ∀ (p : PointOn3x), ∃ (q : PointOnNeg3NegX), q.x = -p.x ∧ q.y = -p.y :=
sorry

end NUMINAMATH_CALUDE_symmetry_about_origin_l15_1532


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l15_1561

/-- The line 5x + 12y + a = 0 is tangent to the circle (x-1)^2 + y^2 = 1 if and only if a = 8 or a = -18 -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∀ x y : ℝ, (5 * x + 12 * y + a = 0) → ((x - 1)^2 + y^2 = 1)) ↔ (a = 8 ∨ a = -18) := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l15_1561


namespace NUMINAMATH_CALUDE_regular_hexagon_area_l15_1504

/-- The area of a regular hexagon with vertices A at (0,0) and C at (8,2) is 34√3 -/
theorem regular_hexagon_area : 
  let A : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (8, 2)
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * AC^2
  let hexagon_area : ℝ := 2 * triangle_area
  hexagon_area = 34 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_regular_hexagon_area_l15_1504


namespace NUMINAMATH_CALUDE_expanded_garden_perimeter_l15_1574

/-- Given a square garden with an area of 49 square meters, if each side is expanded by 4 meters
    to form a new square garden, the perimeter of the new garden is 44 meters. -/
theorem expanded_garden_perimeter : ∀ (original_side : ℝ),
  original_side^2 = 49 →
  (4 * (original_side + 4) = 44) :=
by
  sorry

end NUMINAMATH_CALUDE_expanded_garden_perimeter_l15_1574


namespace NUMINAMATH_CALUDE_middle_number_proof_l15_1527

theorem middle_number_proof (A B C : ℝ) (h1 : A < B) (h2 : B < C) 
  (h3 : B - C = A - B) (h4 : A * B = 85) (h5 : B * C = 115) : B = 10 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l15_1527


namespace NUMINAMATH_CALUDE_keith_missed_games_l15_1545

theorem keith_missed_games (total_games : ℕ) (attended_games : ℕ) 
  (h1 : total_games = 8)
  (h2 : attended_games = 4) :
  total_games - attended_games = 4 := by
  sorry

end NUMINAMATH_CALUDE_keith_missed_games_l15_1545


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l15_1579

theorem complex_fraction_evaluation : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l15_1579


namespace NUMINAMATH_CALUDE_not_proportional_l15_1513

-- Define the properties of direct and inverse proportionality
def is_directly_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

def is_inversely_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x * x = k

-- Define the function representing y = 3x + 2
def f (x : ℝ) : ℝ := 3 * x + 2

-- Theorem statement
theorem not_proportional :
  ¬(is_directly_proportional f) ∧ ¬(is_inversely_proportional f) :=
by sorry

end NUMINAMATH_CALUDE_not_proportional_l15_1513


namespace NUMINAMATH_CALUDE_graph_intersection_symmetry_l15_1557

/-- Given real numbers a, b, c, and d, if the graphs of 
    y = 2a + 1/(x-b) and y = 2c + 1/(x-d) have exactly one common point, 
    then the graphs of y = 2b + 1/(x-a) and y = 2d + 1/(x-c) 
    also have exactly one common point. -/
theorem graph_intersection_symmetry (a b c d : ℝ) :
  (∃! x : ℝ, 2*a + 1/(x-b) = 2*c + 1/(x-d)) →
  (∃! x : ℝ, 2*b + 1/(x-a) = 2*d + 1/(x-c)) :=
by sorry

end NUMINAMATH_CALUDE_graph_intersection_symmetry_l15_1557


namespace NUMINAMATH_CALUDE_work_completion_time_l15_1559

/-- Given that two workers A and B can complete a task together in a certain time,
    and B can complete the task alone in a known time,
    this theorem proves how long it takes A to complete the task alone. -/
theorem work_completion_time
  (joint_time : ℝ)
  (b_time : ℝ)
  (h_joint : joint_time = 8.571428571428571)
  (h_b : b_time = 20)
  : ∃ (a_time : ℝ), a_time = 15 ∧ 1 / a_time + 1 / b_time = 1 / joint_time :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l15_1559


namespace NUMINAMATH_CALUDE_tournament_games_theorem_l15_1503

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_teams : ℕ
  single_elimination : Bool
  no_ties : Bool

/-- Calculates the number of games needed to determine a winner in a tournament. -/
def games_to_winner (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- The theorem stating that a single-elimination tournament with 23 teams and no ties requires 22 games to determine a winner. -/
theorem tournament_games_theorem (t : Tournament) 
  (h1 : t.num_teams = 23) 
  (h2 : t.single_elimination = true) 
  (h3 : t.no_ties = true) : 
  games_to_winner t = 22 := by
  sorry


end NUMINAMATH_CALUDE_tournament_games_theorem_l15_1503


namespace NUMINAMATH_CALUDE_tangent_line_angle_l15_1587

open Real

theorem tangent_line_angle (n : ℤ) : 
  let M : ℝ × ℝ := (7, 1)
  let O : ℝ × ℝ := (4, 4)
  let r : ℝ := 2
  let MO : ℝ × ℝ := (O.1 - M.1, O.2 - M.2)
  let MO_length : ℝ := Real.sqrt ((MO.1)^2 + (MO.2)^2)
  let MO_angle : ℝ := Real.arctan (MO.2 / MO.1) + π
  let φ : ℝ := Real.arcsin (r / MO_length)
  ∃ (a : ℝ), a = MO_angle - φ + n * π ∨ a = MO_angle + φ + n * π := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_angle_l15_1587


namespace NUMINAMATH_CALUDE_albaszu_machine_productivity_l15_1584

-- Define the number of trees cut daily before improvement
def trees_before : ℕ := 16

-- Define the productivity increase factor
def productivity_increase : ℚ := 3/2

-- Define the number of trees cut daily after improvement
def trees_after : ℕ := 25

-- Theorem statement
theorem albaszu_machine_productivity : 
  ↑trees_after = ↑trees_before * productivity_increase :=
by sorry

end NUMINAMATH_CALUDE_albaszu_machine_productivity_l15_1584


namespace NUMINAMATH_CALUDE_largest_gcd_sum_780_l15_1599

theorem largest_gcd_sum_780 :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y = 780 ∧
  (∀ (a b : ℕ), a > 0 → b > 0 → a + b = 780 → Nat.gcd a b ≤ Nat.gcd x y) ∧
  Nat.gcd x y = 390 := by
sorry

end NUMINAMATH_CALUDE_largest_gcd_sum_780_l15_1599


namespace NUMINAMATH_CALUDE_base_prime_representation_of_540_l15_1516

/-- Base prime representation of a natural number -/
def BasePrimeRepresentation (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a list represents a valid base prime representation -/
def IsValidBasePrimeRepresentation (l : List ℕ) : Prop :=
  sorry

theorem base_prime_representation_of_540 :
  let representation := [1, 3, 1]
  540 = 2^1 * 3^3 * 5^1 →
  IsValidBasePrimeRepresentation representation ∧
  BasePrimeRepresentation 540 = representation :=
by sorry

end NUMINAMATH_CALUDE_base_prime_representation_of_540_l15_1516


namespace NUMINAMATH_CALUDE_min_sum_of_ten_numbers_l15_1568

theorem min_sum_of_ten_numbers (S : Finset ℕ) : 
  S.card = 10 → 
  (∀ T ⊆ S, T.card = 5 → (T.prod id) % 2 = 0) → 
  (S.sum id) % 2 = 1 → 
  ∃ min_sum : ℕ, 
    (S.sum id = min_sum) ∧ 
    (∀ S' : Finset ℕ, S'.card = 10 → 
      (∀ T' ⊆ S', T'.card = 5 → (T'.prod id) % 2 = 0) → 
      (S'.sum id) % 2 = 1 → 
      S'.sum id ≥ min_sum) ∧
    min_sum = 51 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_ten_numbers_l15_1568


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l15_1594

/-- Parabola defined by y = x^2 -/
def parabola (x y : ℝ) : Prop := y = x^2

/-- Point Q -/
def Q : ℝ × ℝ := (10, 25)

/-- Line passing through Q with slope m -/
def line (m x y : ℝ) : Prop := y - Q.2 = m * (x - Q.1)

/-- Line does not intersect parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x y : ℝ, ¬(parabola x y ∧ line m x y)

/-- Theorem statement -/
theorem parabola_line_intersection :
  ∃ r s : ℝ, (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) ∧ r + s = 40 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l15_1594


namespace NUMINAMATH_CALUDE_square_approximation_l15_1554

theorem square_approximation (x : ℝ) (h : x ≥ 1/2) :
  ∃ n : ℤ, |x - (n : ℝ)^2| ≤ Real.sqrt (x - 1/4) := by
  sorry

end NUMINAMATH_CALUDE_square_approximation_l15_1554


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l15_1585

theorem sum_of_coefficients (A B : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → A / (x - 2) + B * (x + 3) = (-5 * x^2 + 20 * x + 34) / (x - 2)) →
  A + B = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l15_1585


namespace NUMINAMATH_CALUDE_inequality_proof_l15_1536

theorem inequality_proof (a b c : ℝ) : a^2 + 4*b^2 + 9*c^2 ≥ 2*a*b + 3*a*c + 6*b*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l15_1536


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l15_1563

/-- Proves that adding 750 mL of 30% alcohol solution to 250 mL of 10% alcohol solution
    results in a 25% alcohol solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 250
  let x_concentration : ℝ := 0.10
  let y_volume : ℝ := 750
  let y_concentration : ℝ := 0.30
  let target_concentration : ℝ := 0.25
  
  let total_volume := x_volume + y_volume
  let total_alcohol := x_volume * x_concentration + y_volume * y_concentration
  
  total_alcohol / total_volume = target_concentration := by sorry

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l15_1563


namespace NUMINAMATH_CALUDE_inequalities_hold_l15_1507

theorem inequalities_hold (a b : ℝ) (h : a ≠ b) : 
  (a^2 - 4*a + 5 > 0) ∧ (a^2 + b^2 ≥ 2*(a - b - 1)) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l15_1507


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l15_1560

theorem constant_term_binomial_expansion (x : ℝ) : 
  let binomial := (x - 1 / (2 * Real.sqrt x)) ^ 9
  ∃ c : ℝ, c = 21/16 ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → |binomial - c| < ε) :=
by
  sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l15_1560


namespace NUMINAMATH_CALUDE_temperature_decrease_l15_1542

/-- The temperature that is 6°C lower than -3°C is -9°C. -/
theorem temperature_decrease : ((-3 : ℤ) - 6) = -9 := by
  sorry

end NUMINAMATH_CALUDE_temperature_decrease_l15_1542


namespace NUMINAMATH_CALUDE_fraction_ordering_l15_1575

theorem fraction_ordering : (8 : ℚ) / 25 < 6 / 17 ∧ 6 / 17 < 11 / 29 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l15_1575


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l15_1549

theorem quadratic_coefficient (a b c : ℤ) :
  (∀ x : ℝ, a * (x - 1)^2 + 3 = a * x^2 + b * x + c) →
  a * 0^2 + b * 0 + c = 1 →
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l15_1549


namespace NUMINAMATH_CALUDE_fish_count_l15_1550

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 11

/-- The total number of fish Lilly and Rosy have -/
def total_fish : ℕ := lilly_fish + rosy_fish

theorem fish_count : total_fish = 21 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l15_1550


namespace NUMINAMATH_CALUDE_car_trip_speed_l15_1577

/-- Proves that given the conditions of the car trip, the return speed must be 37.5 mph -/
theorem car_trip_speed (distance : ℝ) (speed_ab : ℝ) (avg_speed : ℝ) :
  distance = 150 →
  speed_ab = 75 →
  avg_speed = 50 →
  ∃ speed_ba : ℝ,
    speed_ba = 37.5 ∧
    avg_speed = (2 * distance) / (distance / speed_ab + distance / speed_ba) :=
by sorry

end NUMINAMATH_CALUDE_car_trip_speed_l15_1577


namespace NUMINAMATH_CALUDE_quentavious_gum_pieces_l15_1541

/-- Given the initial number of nickels, the number of nickels left, and the number of gum pieces per nickel,
    calculate the total number of gum pieces received. -/
def gumPiecesReceived (initialNickels : ℕ) (nickelsLeft : ℕ) (gumPiecesPerNickel : ℕ) : ℕ :=
  (initialNickels - nickelsLeft) * gumPiecesPerNickel

/-- Theorem: The number of gum pieces Quentavious received is 6, given the problem conditions. -/
theorem quentavious_gum_pieces :
  gumPiecesReceived 5 2 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_quentavious_gum_pieces_l15_1541


namespace NUMINAMATH_CALUDE_complement_A_B_when_a_is_one_A_intersection_B_equals_A_l15_1500

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < a * x + 1 ∧ a * x + 1 ≤ 3}
def B : Set ℝ := {x | -1/2 < x ∧ x < 2}

-- Theorem for part (1)
theorem complement_A_B_when_a_is_one :
  (Set.univ \ A 1) ∩ B = {x | -1 < x ∧ x ≤ -1/2} ∪ {2} := by sorry

-- Theorem for part (2)
theorem A_intersection_B_equals_A (a : ℝ) :
  A a ∩ B = A a ↔ a < -4 ∨ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_complement_A_B_when_a_is_one_A_intersection_B_equals_A_l15_1500


namespace NUMINAMATH_CALUDE_abs_increasing_on_unit_interval_l15_1515

-- Define the function f(x) = |x|
def f (x : ℝ) : ℝ := |x|

-- State the theorem
theorem abs_increasing_on_unit_interval : 
  ∀ x y : ℝ, 0 < x → x < y → y < 1 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_abs_increasing_on_unit_interval_l15_1515


namespace NUMINAMATH_CALUDE_correct_answer_calculation_l15_1508

theorem correct_answer_calculation (incorrect_answer : ℝ) (h : incorrect_answer = 115.15) :
  let original_value := incorrect_answer / 7
  let correct_answer := original_value / 7
  correct_answer = 2.35 := by
sorry

end NUMINAMATH_CALUDE_correct_answer_calculation_l15_1508


namespace NUMINAMATH_CALUDE_count_valid_voucher_codes_l15_1522

/-- Represents a voucher code -/
structure VoucherCode where
  first : Char
  second : Nat
  third : Nat
  fourth : Nat

/-- Checks if a character is a valid first character -/
def isValidFirstChar (c : Char) : Bool :=
  c = 'V' || c = 'X' || c = 'P'

/-- Checks if a voucher code is valid -/
def isValidVoucherCode (code : VoucherCode) : Bool :=
  isValidFirstChar code.first &&
  code.second < 10 &&
  code.third < 10 &&
  code.second ≠ code.third &&
  code.fourth = (code.second + code.third) % 10

/-- The set of all valid voucher codes -/
def validVoucherCodes : Finset VoucherCode :=
  sorry

/-- The number of valid voucher codes is 270 -/
theorem count_valid_voucher_codes :
  Finset.card validVoucherCodes = 270 :=
sorry

end NUMINAMATH_CALUDE_count_valid_voucher_codes_l15_1522


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l15_1551

-- Define the universal set U
def U : Finset Nat := {0, 1, 2, 3}

-- Define set A
def A : Finset Nat := {1, 3}

-- Theorem statement
theorem complement_of_A_in_U : 
  (U \ A) = {0, 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l15_1551


namespace NUMINAMATH_CALUDE_pastor_prayer_theorem_l15_1564

/-- Represents the number of times Pastor Paul prays per day (except on Sundays) -/
def paul_prayers : ℕ := sorry

/-- Represents the number of times Pastor Bruce prays per day (except on Sundays) -/
def bruce_prayers : ℕ := sorry

/-- The total number of times Pastor Paul prays in a week -/
def paul_weekly_prayers : ℕ := 6 * paul_prayers + 2 * paul_prayers

/-- The total number of times Pastor Bruce prays in a week -/
def bruce_weekly_prayers : ℕ := 6 * (paul_prayers / 2) + 4 * paul_prayers

theorem pastor_prayer_theorem :
  paul_prayers = 20 ∧
  bruce_prayers = paul_prayers / 2 ∧
  paul_weekly_prayers = bruce_weekly_prayers + 20 := by
sorry

end NUMINAMATH_CALUDE_pastor_prayer_theorem_l15_1564


namespace NUMINAMATH_CALUDE_sparrow_percentage_among_non_eagles_l15_1523

theorem sparrow_percentage_among_non_eagles (total percentage : ℝ)
  (robins eagles falcons sparrows : ℝ)
  (h1 : total = 100)
  (h2 : robins = 20)
  (h3 : eagles = 30)
  (h4 : falcons = 15)
  (h5 : sparrows = total - (robins + eagles + falcons))
  (h6 : percentage = (sparrows / (total - eagles)) * 100) :
  percentage = 50 := by
sorry

end NUMINAMATH_CALUDE_sparrow_percentage_among_non_eagles_l15_1523


namespace NUMINAMATH_CALUDE_sam_pages_sam_read_100_pages_l15_1501

def minimum_assigned : ℕ := 25

def harrison_extra : ℕ := 10

def pam_extra : ℕ := 15

def sam_multiplier : ℕ := 2

theorem sam_pages : ℕ :=
  let harrison_pages := minimum_assigned + harrison_extra
  let pam_pages := harrison_pages + pam_extra
  sam_multiplier * pam_pages

theorem sam_read_100_pages : sam_pages = 100 := by
  sorry

end NUMINAMATH_CALUDE_sam_pages_sam_read_100_pages_l15_1501


namespace NUMINAMATH_CALUDE_complex_multiplication_l15_1534

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (1 - i)^2 * (1 + i) = 2 - 2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l15_1534


namespace NUMINAMATH_CALUDE_fraction_equality_l15_1595

theorem fraction_equality (w x y : ℝ) (hw_x : w / x = 1 / 3) (hw_y : w / y = 3 / 4) 
  (hx : x ≠ 0) (hy : y ≠ 0) : (x + y) / y = 13 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l15_1595


namespace NUMINAMATH_CALUDE_sum_of_two_squares_equivalence_l15_1511

theorem sum_of_two_squares_equivalence (n : ℕ) (hn : n > 0) :
  (∃ a b : ℤ, n = a^2 + b^2) ↔ (∃ A B : ℤ, 2 * n = A^2 + B^2) :=
sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_equivalence_l15_1511


namespace NUMINAMATH_CALUDE_grandpa_lou_movie_time_l15_1520

theorem grandpa_lou_movie_time :
  ∀ (tuesday_movies : ℕ),
    (tuesday_movies + 2 * tuesday_movies ≤ 9) →
    (tuesday_movies * 90 = 270) :=
by
  sorry

end NUMINAMATH_CALUDE_grandpa_lou_movie_time_l15_1520


namespace NUMINAMATH_CALUDE_no_x_squared_term_l15_1543

theorem no_x_squared_term (a : ℝ) : 
  (∀ x : ℝ, (x^2 + a*x + 2) * (2*x - 4) = 2*x^3 + (2*a - 4)*x^2 + (4 - 4*a)*x - 8) →
  (2*a - 4 = 0 ↔ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l15_1543


namespace NUMINAMATH_CALUDE_boat_distance_against_stream_l15_1558

/-- Calculates the distance a boat travels against a stream in one hour, given its speed in still water and its distance traveled along the stream in one hour. -/
def distance_against_stream (speed_still_water : ℝ) (distance_along_stream : ℝ) : ℝ :=
  speed_still_water - (distance_along_stream - speed_still_water)

/-- Theorem stating that a boat with a speed of 8 km/hr in still water, which travels 11 km along a stream in one hour, will travel 5 km against the stream in one hour. -/
theorem boat_distance_against_stream :
  distance_against_stream 8 11 = 5 := by
  sorry

#eval distance_against_stream 8 11

end NUMINAMATH_CALUDE_boat_distance_against_stream_l15_1558


namespace NUMINAMATH_CALUDE_bicycle_sale_price_l15_1562

/-- Given a cost price and two consecutive percentage markups, 
    calculate the final selling price. -/
def final_price (cost_price : ℚ) (markup_percent : ℚ) : ℚ :=
  let first_sale := cost_price * (1 + markup_percent / 100)
  first_sale * (1 + markup_percent / 100)

/-- Theorem: The final selling price of a bicycle with an initial cost of 144,
    after two consecutive 25% markups, is 225. -/
theorem bicycle_sale_price : final_price 144 25 = 225 := by
  sorry

#eval final_price 144 25

end NUMINAMATH_CALUDE_bicycle_sale_price_l15_1562


namespace NUMINAMATH_CALUDE_square_ratio_proof_l15_1590

theorem square_ratio_proof (a b : ℝ) (h : a > 0 ∧ b > 0) (h_ratio : a^2 / b^2 = 75 / 98) :
  ∃ (x y z : ℕ), 
    (Real.sqrt (a / b) = x * Real.sqrt 6 / (y : ℝ)) ∧ 
    (x + 6 + y = z) ∧
    x = 5 ∧ y = 14 ∧ z = 25 := by
  sorry


end NUMINAMATH_CALUDE_square_ratio_proof_l15_1590


namespace NUMINAMATH_CALUDE_trapezoid_xy_length_l15_1540

/-- Represents a trapezoid WXYZ with specific properties -/
structure Trapezoid where
  -- Points W, X, Y, Z
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  -- WX is parallel to ZY
  parallel_WX_ZY : (X.1 - W.1) * (Y.2 - Z.2) = (X.2 - W.2) * (Y.1 - Z.1)
  -- WY is perpendicular to ZY
  perpendicular_WY_ZY : (Y.1 - W.1) * (Y.1 - Z.1) + (Y.2 - W.2) * (Y.2 - Z.2) = 0
  -- YZ = 20
  yz_length : Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) = 20
  -- tan Z = 2
  tan_Z : (Y.2 - Z.2) / (Y.1 - Z.1) = 2
  -- tan X = 2.5
  tan_X : (Y.2 - X.2) / (X.1 - Y.1) = 2.5

/-- The length of XY in the trapezoid is 4√116 -/
theorem trapezoid_xy_length (t : Trapezoid) : 
  Real.sqrt ((t.X.1 - t.Y.1)^2 + (t.X.2 - t.Y.2)^2) = 4 * Real.sqrt 116 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_xy_length_l15_1540


namespace NUMINAMATH_CALUDE_inequality_proof_l15_1572

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (a^3 + b^3 + a*b*c) + 1 / (b^3 + c^3 + a*b*c) + 1 / (c^3 + a^3 + a*b*c) ≤ 1 / (a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l15_1572


namespace NUMINAMATH_CALUDE_clock_strike_time_l15_1505

/-- If a clock strikes 12 in 33 seconds, it will strike 6 in 15 seconds -/
theorem clock_strike_time (strike_12_time : ℕ) (strike_6_time : ℕ) : 
  strike_12_time = 33 → strike_6_time = 15 := by
  sorry

#check clock_strike_time

end NUMINAMATH_CALUDE_clock_strike_time_l15_1505


namespace NUMINAMATH_CALUDE_camera_rental_theorem_l15_1517

def camera_rental_problem (camera_value : ℝ) (rental_weeks : ℕ) 
  (base_fee_rate : ℝ) (high_demand_rate : ℝ) (low_demand_rate : ℝ)
  (insurance_rate : ℝ) (sales_tax_rate : ℝ)
  (mike_contribution_rate : ℝ) (sarah_contribution_rate : ℝ) (sarah_contribution_cap : ℝ)
  (alex_contribution_rate : ℝ) (alex_contribution_cap : ℝ) : Prop :=
  let base_fee := camera_value * base_fee_rate
  let high_demand_fee := base_fee + (camera_value * high_demand_rate)
  let low_demand_fee := base_fee - (camera_value * low_demand_rate)
  let total_rental_fee := 2 * high_demand_fee + 2 * low_demand_fee
  let insurance_fee := camera_value * insurance_rate
  let subtotal := total_rental_fee + insurance_fee
  let total_cost := subtotal + (subtotal * sales_tax_rate)
  let mike_contribution := total_cost * mike_contribution_rate
  let sarah_contribution := min (total_cost * sarah_contribution_rate) sarah_contribution_cap
  let alex_contribution := min (total_cost * alex_contribution_rate) alex_contribution_cap
  let total_contribution := mike_contribution + sarah_contribution + alex_contribution
  let john_payment := total_cost - total_contribution
  john_payment = 1015.20

theorem camera_rental_theorem : 
  camera_rental_problem 5000 4 0.10 0.03 0.02 0.05 0.08 0.20 0.30 1000 0.10 700 := by
  sorry

#check camera_rental_theorem

end NUMINAMATH_CALUDE_camera_rental_theorem_l15_1517


namespace NUMINAMATH_CALUDE_f_extrema_max_k_bound_l15_1533

noncomputable section

def f (x : ℝ) : ℝ := x + x * Real.log x

theorem f_extrema :
  (∃ (x_min : ℝ), x_min = Real.exp (-2) ∧
    (∀ x > 0, f x ≥ f x_min) ∧
    f x_min = -Real.exp (-2)) ∧
  (∀ M : ℝ, ∃ x > 0, f x > M) :=
sorry

theorem max_k_bound :
  (∀ k : ℤ, (∀ x > 1, f x > k * (x - 1)) → k ≤ 3) ∧
  (∃ x > 1, f x > 3 * (x - 1)) :=
sorry

end NUMINAMATH_CALUDE_f_extrema_max_k_bound_l15_1533
