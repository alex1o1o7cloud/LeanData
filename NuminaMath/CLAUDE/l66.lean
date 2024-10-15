import Mathlib

namespace NUMINAMATH_CALUDE_two_blue_probability_l66_6658

def total_balls : ℕ := 15
def blue_balls : ℕ := 5
def red_balls : ℕ := 10
def drawn_balls : ℕ := 6
def target_blue : ℕ := 2

def probability_two_blue : ℚ := 2100 / 5005

theorem two_blue_probability :
  (Nat.choose blue_balls target_blue * Nat.choose red_balls (drawn_balls - target_blue)) /
  Nat.choose total_balls drawn_balls = probability_two_blue := by
  sorry

end NUMINAMATH_CALUDE_two_blue_probability_l66_6658


namespace NUMINAMATH_CALUDE_high_school_math_club_payment_l66_6678

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem high_school_math_club_payment :
  ∀ B : ℕ, 
    B < 10 →
    is_divisible_by (2000 + 100 * B + 40) 15 →
    B = 7 := by
  sorry

end NUMINAMATH_CALUDE_high_school_math_club_payment_l66_6678


namespace NUMINAMATH_CALUDE_zoo_animal_count_l66_6625

/-- Represents the zoo layout and animal counts -/
structure Zoo where
  tigerEnclosures : Nat
  zebraEnclosuresPerTiger : Nat
  giraffeEnclosureMultiplier : Nat
  tigersPerEnclosure : Nat
  zebrasPerEnclosure : Nat
  giraffesPerEnclosure : Nat

/-- Calculates the total number of animals in the zoo -/
def totalAnimals (zoo : Zoo) : Nat :=
  let zebraEnclosures := zoo.tigerEnclosures * zoo.zebraEnclosuresPerTiger
  let giraffeEnclosures := zebraEnclosures * zoo.giraffeEnclosureMultiplier
  let tigers := zoo.tigerEnclosures * zoo.tigersPerEnclosure
  let zebras := zebraEnclosures * zoo.zebrasPerEnclosure
  let giraffes := giraffeEnclosures * zoo.giraffesPerEnclosure
  tigers + zebras + giraffes

/-- Theorem stating that the total number of animals in the zoo is 144 -/
theorem zoo_animal_count :
  ∀ (zoo : Zoo),
    zoo.tigerEnclosures = 4 →
    zoo.zebraEnclosuresPerTiger = 2 →
    zoo.giraffeEnclosureMultiplier = 3 →
    zoo.tigersPerEnclosure = 4 →
    zoo.zebrasPerEnclosure = 10 →
    zoo.giraffesPerEnclosure = 2 →
    totalAnimals zoo = 144 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_count_l66_6625


namespace NUMINAMATH_CALUDE_worker_productivity_increase_l66_6641

theorem worker_productivity_increase 
  (initial_value : ℝ) 
  (final_value : ℝ) 
  (increase_percentage : ℝ) 
  (h1 : initial_value = 2500)
  (h2 : final_value = 2809)
  (h3 : final_value = initial_value * (1 + increase_percentage / 100)^2) :
  increase_percentage = 6 := by
sorry

end NUMINAMATH_CALUDE_worker_productivity_increase_l66_6641


namespace NUMINAMATH_CALUDE_detached_calculations_l66_6662

theorem detached_calculations : 
  (78 * 12 - 531 = 405) ∧ 
  (32 * (69 - 54) = 480) ∧ 
  (58 / 2 * 16 = 464) ∧ 
  (352 / 8 / 4 = 11) := by
  sorry

end NUMINAMATH_CALUDE_detached_calculations_l66_6662


namespace NUMINAMATH_CALUDE_missy_dog_yells_l66_6694

/-- Represents the number of times Missy yells at her dogs -/
structure DogYells where
  obedient : ℕ
  stubborn : ℕ
  total : ℕ

/-- Theorem: If Missy yells at the obedient dog 12 times and yells at both dogs combined 60 times,
    then she yells at the stubborn dog 4 times for every one time she yells at the obedient dog -/
theorem missy_dog_yells (d : DogYells) 
    (h1 : d.obedient = 12)
    (h2 : d.total = 60)
    (h3 : d.total = d.obedient + d.stubborn) :
    d.stubborn = 4 * d.obedient := by
  sorry

#check missy_dog_yells

end NUMINAMATH_CALUDE_missy_dog_yells_l66_6694


namespace NUMINAMATH_CALUDE_barrel_capacity_l66_6600

theorem barrel_capacity (total_capacity : ℝ) (increase : ℝ) (decrease : ℝ)
  (h1 : total_capacity = 7000)
  (h2 : increase = 1000)
  (h3 : decrease = 4000) :
  ∃ (x y : ℝ),
    x + y = total_capacity ∧
    x = 6400 ∧
    y = 600 ∧
    x / (total_capacity + increase) + y / (total_capacity - decrease) = 1 :=
by sorry

end NUMINAMATH_CALUDE_barrel_capacity_l66_6600


namespace NUMINAMATH_CALUDE_sum_is_linear_l66_6622

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Applies the described transformations to a parabola -/
def transform (p : Parabola) : ℝ → ℝ := 
  fun x => p.a * (x - 4)^2 + p.b * (x - 4) + p.c + 2

/-- Applies the described transformations to the reflection of a parabola -/
def transform_reflection (p : Parabola) : ℝ → ℝ := 
  fun x => -p.a * (x + 6)^2 - p.b * (x + 6) - p.c + 2

/-- The sum of the transformed parabola and its reflection -/
def sum_of_transformations (p : Parabola) : ℝ → ℝ :=
  fun x => transform p x + transform_reflection p x

theorem sum_is_linear (p : Parabola) : 
  ∀ x, sum_of_transformations p x = -20 * p.a * x + 52 * p.a - 10 * p.b + 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_is_linear_l66_6622


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l66_6602

theorem negation_of_existence (p : ℕ → Prop) :
  (¬ ∃ n, p n) ↔ ∀ n, ¬ p n := by sorry

theorem negation_of_proposition :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l66_6602


namespace NUMINAMATH_CALUDE_solution_difference_l66_6653

-- Define the equation
def equation (x : ℝ) : Prop :=
  (6 * x - 18) / (x^2 + x - 20) = x + 3

-- Define the theorem
theorem solution_difference :
  ∃ (p q : ℝ), 
    p ≠ q ∧
    equation p ∧
    equation q ∧
    p > q ∧
    p - q = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_difference_l66_6653


namespace NUMINAMATH_CALUDE_solution_fraction_proof_l66_6699

def initial_amount : ℚ := 2

def first_day_usage (amount : ℚ) : ℚ := (1 / 4) * amount

def second_day_usage (amount : ℚ) : ℚ := (1 / 2) * amount

def remaining_after_two_days (initial : ℚ) : ℚ :=
  initial - first_day_usage initial - second_day_usage (initial - first_day_usage initial)

theorem solution_fraction_proof :
  remaining_after_two_days initial_amount / initial_amount = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_solution_fraction_proof_l66_6699


namespace NUMINAMATH_CALUDE_happy_street_weekend_traffic_l66_6682

/-- Number of cars passing Happy Street each day of the week -/
structure WeekTraffic where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  weekend_day : ℕ

/-- Conditions for the Happy Street traffic problem -/
def happy_street_conditions (w : WeekTraffic) : Prop :=
  w.tuesday = 25 ∧
  w.monday = w.tuesday - (w.tuesday / 5) ∧
  w.wednesday = w.monday + 2 ∧
  w.thursday = 10 ∧
  w.friday = 10 ∧
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + 2 * w.weekend_day = 97

theorem happy_street_weekend_traffic (w : WeekTraffic) 
  (h : happy_street_conditions w) : w.weekend_day = 5 := by
  sorry


end NUMINAMATH_CALUDE_happy_street_weekend_traffic_l66_6682


namespace NUMINAMATH_CALUDE_intersection_count_theorem_m_value_theorem_l66_6670

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 2

-- Define the line l
def l (x y : ℝ) : Prop := y = x ∧ x ≥ 0

-- Define the number of intersection points
def intersection_count : ℕ := 1

-- Define the equation for C₂ when θ = π/4
def C₂_equation (ρ m : ℝ) : Prop := ρ^2 - 3 * Real.sqrt 2 * ρ + 2 * m = 0

-- Theorem for the number of intersection points
theorem intersection_count_theorem :
  ∃! (x y : ℝ), C₁ x y ∧ l x y :=
sorry

-- Theorem for the value of m
theorem m_value_theorem (ρ₁ ρ₂ m : ℝ) :
  C₂_equation ρ₁ m ∧ C₂_equation ρ₂ m ∧ ρ₂ = 2 * ρ₁ → m = 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_count_theorem_m_value_theorem_l66_6670


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l66_6688

/-- Given vectors a and b, if a is perpendicular to (t*a + b), then t = -1 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (t : ℝ) :
  a = (1, -1) →
  b = (6, -4) →
  (a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) →
  t = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l66_6688


namespace NUMINAMATH_CALUDE_train_length_is_100m_l66_6654

-- Define the given constants
def train_speed : Real := 60  -- km/h
def bridge_length : Real := 80  -- meters
def crossing_time : Real := 10.799136069114471  -- seconds

-- Theorem to prove
theorem train_length_is_100m :
  let speed_ms : Real := train_speed * 1000 / 3600  -- Convert km/h to m/s
  let total_distance : Real := speed_ms * crossing_time
  let train_length : Real := total_distance - bridge_length
  train_length = 100 := by sorry

end NUMINAMATH_CALUDE_train_length_is_100m_l66_6654


namespace NUMINAMATH_CALUDE_determinant_solution_set_implies_a_value_l66_6639

-- Define the determinant function
def det (x a : ℝ) : ℝ := a * x + 2

-- Define the inequality
def inequality (x a : ℝ) : Prop := det x a < 6

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | inequality x a}

-- Theorem statement
theorem determinant_solution_set_implies_a_value :
  (∀ x : ℝ, x > -1 ↔ x ∈ solution_set a) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_determinant_solution_set_implies_a_value_l66_6639


namespace NUMINAMATH_CALUDE_matrix_inverse_l66_6657

theorem matrix_inverse (x : ℝ) (h : x ≠ -12) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, x; -2, 6]
  let A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![6 / (24 + 2*x), -x / (24 + 2*x); 2 / (24 + 2*x), 4 / (24 + 2*x)]
  A * A_inv = 1 ∧ A_inv * A = 1 :=
by sorry


end NUMINAMATH_CALUDE_matrix_inverse_l66_6657


namespace NUMINAMATH_CALUDE_unique_integer_solution_inequality_proof_l66_6674

-- Part 1
theorem unique_integer_solution (m : ℤ) 
  (h : ∃! (x : ℤ), |2*x - m| < 1 ∧ x = 2) : m = 4 := by
  sorry

-- Part 2
theorem inequality_proof (a b : ℝ) 
  (h1 : a * b = 4)
  (h2 : a > b)
  (h3 : b > 0) : 
  (a^2 + b^2) / (a - b) ≥ 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_inequality_proof_l66_6674


namespace NUMINAMATH_CALUDE_triangle_inequality_l66_6629

/-- For any triangle with sides a, b, c, semi-perimeter p, inradius r, and area S,
    where S = √(p(p-a)(p-b)(p-c)) and r = S/p, the following inequality holds:
    1/(p-a)² + 1/(p-b)² + 1/(p-c)² ≥ 1/r² -/
theorem triangle_inequality (a b c p r S : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : p = (a + b + c) / 2)
  (h5 : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (h6 : r = S / p) :
  1 / (p - a)^2 + 1 / (p - b)^2 + 1 / (p - c)^2 ≥ 1 / r^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l66_6629


namespace NUMINAMATH_CALUDE_sequence_partition_sequence_partition_general_l66_6691

-- Define the type for our sequence
def Sequence := ℕ → Set ℝ

-- Define what it means for a sequence to be in [0, 1)
def InUnitInterval (s : Sequence) : Prop :=
  ∀ n, ∀ x ∈ s n, 0 ≤ x ∧ x < 1

-- Define what it means for a set to contain infinitely many elements of a sequence
def ContainsInfinitelyMany (A : Set ℝ) (s : Sequence) : Prop :=
  ∀ N, ∃ n ≥ N, ∃ x ∈ s n, x ∈ A

theorem sequence_partition (s : Sequence) (h : InUnitInterval s) :
  ContainsInfinitelyMany (Set.Icc 0 (1/2)) s ∨ ContainsInfinitelyMany (Set.Ico (1/2) 1) s :=
sorry

theorem sequence_partition_general (s : Sequence) (h : InUnitInterval s) :
  ∀ n : ℕ, n ≥ 1 →
    ∃ k : ℕ, k < 2^n ∧
      ContainsInfinitelyMany (Set.Ico (k / 2^n) ((k + 1) / 2^n)) s :=
sorry

end NUMINAMATH_CALUDE_sequence_partition_sequence_partition_general_l66_6691


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l66_6679

theorem smallest_solution_of_equation :
  ∃ (x : ℝ), x = 4 - Real.sqrt 2 ∧
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ (y : ℝ), (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → y ≥ x := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l66_6679


namespace NUMINAMATH_CALUDE_liquor_and_beer_cost_l66_6636

/-- The price of one bottle of beer in yuan -/
def beer_price : ℚ := 2

/-- The price of one bottle of liquor in yuan -/
def liquor_price : ℚ := 16

/-- The total cost of 2 bottles of liquor and 12 bottles of beer in yuan -/
def total_cost : ℚ := 56

/-- The number of bottles of beer equivalent in price to one bottle of liquor -/
def liquor_to_beer_ratio : ℕ := 8

theorem liquor_and_beer_cost :
  (2 * liquor_price + 12 * beer_price = total_cost) →
  (liquor_price = liquor_to_beer_ratio * beer_price) →
  (liquor_price + beer_price = 18) := by
    sorry

end NUMINAMATH_CALUDE_liquor_and_beer_cost_l66_6636


namespace NUMINAMATH_CALUDE_otimes_equation_solution_l66_6680

/-- Custom binary operator ⊗ -/
def otimes (a b : ℝ) : ℝ := -2 * a + b

/-- Theorem stating that if x ⊗ (-5) = 3, then x = -4 -/
theorem otimes_equation_solution (x : ℝ) (h : otimes x (-5) = 3) : x = -4 := by
  sorry

end NUMINAMATH_CALUDE_otimes_equation_solution_l66_6680


namespace NUMINAMATH_CALUDE_equation_D_is_quadratic_l66_6647

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 1 = 0 -/
def equation_D (x : ℝ) : ℝ :=
  x^2 - 1

/-- Theorem: equation_D is a quadratic equation -/
theorem equation_D_is_quadratic : is_quadratic_equation equation_D :=
  sorry

end NUMINAMATH_CALUDE_equation_D_is_quadratic_l66_6647


namespace NUMINAMATH_CALUDE_max_remainder_of_division_by_11_l66_6669

theorem max_remainder_of_division_by_11 (A B C : ℕ) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A = 11 * B + C →
  C ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_remainder_of_division_by_11_l66_6669


namespace NUMINAMATH_CALUDE_david_scott_age_difference_l66_6608

/-- Represents the ages of three brothers -/
structure BrotherAges where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- Defines the conditions given in the problem -/
def satisfiesConditions (ages : BrotherAges) : Prop :=
  ages.richard = ages.david + 6 ∧
  ages.richard + 8 = 2 * (ages.scott + 8) ∧
  ages.david = 14

/-- Theorem stating that David is 8 years older than Scott -/
theorem david_scott_age_difference (ages : BrotherAges) :
  satisfiesConditions ages → ages.david - ages.scott = 8 := by
  sorry

end NUMINAMATH_CALUDE_david_scott_age_difference_l66_6608


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_range_l66_6627

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - a| ≤ 1}
def B : Set ℝ := {x : ℝ | x^2 - 5*x + 4 ≥ 0}

-- State the theorem
theorem intersection_empty_implies_a_range (a : ℝ) : A a ∩ B = ∅ → 2 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_range_l66_6627


namespace NUMINAMATH_CALUDE_min_stamps_for_35_cents_l66_6661

/-- Represents the number of ways to make a certain amount of cents using 5-cent and 7-cent stamps -/
def stamp_combinations (cents : ℕ) : Set (ℕ × ℕ) :=
  {(x, y) | 5 * x + 7 * y = cents}

/-- The total number of stamps used in a combination -/
def total_stamps (combo : ℕ × ℕ) : ℕ :=
  combo.1 + combo.2

theorem min_stamps_for_35_cents :
  ∃ (combo : ℕ × ℕ),
    combo ∈ stamp_combinations 35 ∧
    ∀ (other : ℕ × ℕ), other ∈ stamp_combinations 35 →
      total_stamps combo ≤ total_stamps other ∧
      total_stamps combo = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_stamps_for_35_cents_l66_6661


namespace NUMINAMATH_CALUDE_inequality_proof_l66_6656

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : 1/a + 1/b + 1/c = 1) :
  a^(a*b*c) + b^(b*c*a) + c^(c*a*b) ≥ 27*b*c + 27*c*a + 27*a*b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l66_6656


namespace NUMINAMATH_CALUDE_range_of_m_l66_6659

-- Define the propositions p and q
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 < 0

-- Define the condition that ¬p is a necessary but not sufficient condition for ¬q
def not_p_necessary_not_sufficient_for_not_q (m : ℝ) : Prop :=
  ∀ x, ¬(q x m) → ¬(p x) ∧ ∃ x, ¬(p x) ∧ (q x m)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (∀ x, p x → q x m) ∧ not_p_necessary_not_sufficient_for_not_q m
  ↔ m ≥ 9 ∨ m ≤ -9 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l66_6659


namespace NUMINAMATH_CALUDE_m_range_l66_6685

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, x ≤ -1 → (m^2 - m) * 2^x - (1/2)^x < 1) → 
  -2 < m ∧ m < 3 := by
sorry

end NUMINAMATH_CALUDE_m_range_l66_6685


namespace NUMINAMATH_CALUDE_complex_square_root_l66_6697

theorem complex_square_root (a b : ℕ+) (h : (a - b * Complex.I) ^ 2 = 8 - 6 * Complex.I) :
  a - b * Complex.I = 3 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_l66_6697


namespace NUMINAMATH_CALUDE_cube_opposite_face_l66_6634

-- Define a cube type
structure Cube :=
  (faces : Fin 6 → Char)

-- Define adjacency relation
def adjacent (c : Cube) (x y : Char) : Prop :=
  ∃ (i j : Fin 6), i ≠ j ∧ c.faces i = x ∧ c.faces j = y

-- Define opposite relation
def opposite (c : Cube) (x y : Char) : Prop :=
  ∃ (i j : Fin 6), i ≠ j ∧ c.faces i = x ∧ c.faces j = y ∧
  ∀ (k : Fin 6), k ≠ i → k ≠ j → ¬(adjacent c (c.faces i) (c.faces k) ∧ adjacent c (c.faces j) (c.faces k))

theorem cube_opposite_face (c : Cube) :
  (c.faces = λ i => ['А', 'Б', 'В', 'Г', 'Д', 'Е'][i]) →
  (adjacent c 'А' 'Б') →
  (adjacent c 'А' 'Г') →
  (adjacent c 'Г' 'Д') →
  (adjacent c 'Г' 'Е') →
  (adjacent c 'В' 'Д') →
  (adjacent c 'В' 'Б') →
  opposite c 'Д' 'Б' :=
by sorry

end NUMINAMATH_CALUDE_cube_opposite_face_l66_6634


namespace NUMINAMATH_CALUDE_alphametic_puzzle_solution_l66_6628

theorem alphametic_puzzle_solution :
  ∃! (T H E B G M A : ℕ),
    T ≠ H ∧ T ≠ E ∧ T ≠ B ∧ T ≠ G ∧ T ≠ M ∧ T ≠ A ∧
    H ≠ E ∧ H ≠ B ∧ H ≠ G ∧ H ≠ M ∧ H ≠ A ∧
    E ≠ B ∧ E ≠ G ∧ E ≠ M ∧ E ≠ A ∧
    B ≠ G ∧ B ≠ M ∧ B ≠ A ∧
    G ≠ M ∧ G ≠ A ∧
    M ≠ A ∧
    T < 10 ∧ H < 10 ∧ E < 10 ∧ B < 10 ∧ G < 10 ∧ M < 10 ∧ A < 10 ∧
    1000 * T + 100 * H + 10 * E + T + 1000 * B + 100 * E + 10 * T + A =
    10000 * G + 1000 * A + 100 * M + 10 * M + A ∧
    T = 4 ∧ H = 9 ∧ E = 4 ∧ B = 5 ∧ G = 1 ∧ M = 8 ∧ A = 0 :=
by sorry

end NUMINAMATH_CALUDE_alphametic_puzzle_solution_l66_6628


namespace NUMINAMATH_CALUDE_inequalities_given_ordered_reals_l66_6692

theorem inequalities_given_ordered_reals (a b c : ℝ) 
  (h1 : a > 0) (h2 : 0 > b) (h3 : b > c) : 
  (a / c > a / b) ∧ 
  ((a - b) / (a - c) > b / c) ∧ 
  (a - c ≥ 2 * Real.sqrt ((a - b) * (b - c))) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_given_ordered_reals_l66_6692


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l66_6614

theorem quadratic_inequality_solution_set (a b c : ℝ) :
  a > 0 → (∀ x, a * x^2 + b * x + c > 0) ↔ b^2 - 4*a*c < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l66_6614


namespace NUMINAMATH_CALUDE_sequence_formula_l66_6638

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, n > 0 → a (n + 1) - a n = r * (a n - a (n - 1))

theorem sequence_formula (a : ℕ → ℝ) :
  geometric_sequence (λ n => a (n + 1) - a n) ∧
  (a 2 - a 1 = 1) ∧
  (∀ n : ℕ, n > 0 → a (n + 1) - a n = (1 / 3) * (a n - a (n - 1))) →
  ∀ n : ℕ, n > 0 → a n = (3 / 2) * (1 - (1 / 3) ^ n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formula_l66_6638


namespace NUMINAMATH_CALUDE_power_four_inequality_l66_6664

theorem power_four_inequality (x y : ℝ) : x^4 + y^4 ≥ x*y*(x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_power_four_inequality_l66_6664


namespace NUMINAMATH_CALUDE_f_bounded_g_bounded_l66_6606

-- Define the functions f and g
def f (x : ℝ) := 3 * x - 4 * x^3
def g (x : ℝ) := 3 * x - 4 * x^3

-- Theorem for function f
theorem f_bounded : ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |f x| ≤ 1 := by
  sorry

-- Theorem for function g
theorem g_bounded : ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |g x| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_f_bounded_g_bounded_l66_6606


namespace NUMINAMATH_CALUDE_chessboard_selections_theorem_l66_6667

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (is_valid : size = 4 ∨ size = 8)

/-- Represents a selection of squares on a chessboard -/
structure Selection (board : Chessboard) :=
  (count : Nat)
  (row_count : Nat)
  (col_count : Nat)
  (is_valid : count = row_count * board.size ∧ row_count = col_count)

/-- Counts the number of valid selections on a 4x4 board -/
def count_4x4_selections (board : Chessboard) (sel : Selection board) : Nat :=
  24

/-- Counts the number of valid selections on an 8x8 board with all black squares chosen -/
def count_8x8_selections (board : Chessboard) (sel : Selection board) : Nat :=
  576

/-- The main theorem to prove -/
theorem chessboard_selections_theorem (board4 : Chessboard) (board8 : Chessboard) 
  (sel4 : Selection board4) (sel8 : Selection board8) :
  board4.size = 4 ∧ 
  board8.size = 8 ∧ 
  sel4.count = 12 ∧ 
  sel4.row_count = 3 ∧
  sel8.count = 56 ∧
  sel8.row_count = 7 →
  count_8x8_selections board8 sel8 = (count_4x4_selections board4 sel4) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_chessboard_selections_theorem_l66_6667


namespace NUMINAMATH_CALUDE_bag_weight_l66_6689

theorem bag_weight (w : ℝ) (h : w = 16 / (w / 4)) : w = 16 := by
  sorry

end NUMINAMATH_CALUDE_bag_weight_l66_6689


namespace NUMINAMATH_CALUDE_max_trig_fraction_l66_6618

theorem max_trig_fraction (x : ℝ) : 
  (Real.sin x)^4 + (Real.cos x)^4 ≤ (Real.sin x)^2 + (Real.cos x)^2 + 2*(Real.sin x)^2*(Real.cos x)^2 := by
  sorry

#check max_trig_fraction

end NUMINAMATH_CALUDE_max_trig_fraction_l66_6618


namespace NUMINAMATH_CALUDE_inequality_proof_l66_6615

theorem inequality_proof (a b c d : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d)
  (h5 : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c)) ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l66_6615


namespace NUMINAMATH_CALUDE_mrs_excellent_class_size_l66_6651

/-- Represents the number of students in Mrs. Excellent's class -/
def total_students : ℕ := 29

/-- Represents the number of girls in the class -/
def girls : ℕ := 13

/-- Represents the number of boys in the class -/
def boys : ℕ := girls + 3

/-- Represents the total number of jellybeans Mrs. Excellent has -/
def total_jellybeans : ℕ := 450

/-- Represents the number of jellybeans left after distribution -/
def leftover_jellybeans : ℕ := 10

theorem mrs_excellent_class_size :
  (girls * girls + boys * boys + leftover_jellybeans = total_jellybeans) ∧
  (girls + boys = total_students) := by
  sorry

#check mrs_excellent_class_size

end NUMINAMATH_CALUDE_mrs_excellent_class_size_l66_6651


namespace NUMINAMATH_CALUDE_min_sum_box_dimensions_l66_6601

theorem min_sum_box_dimensions (a b c : ℕ+) : 
  a * b * c = 3003 → 
  ∀ x y z : ℕ+, x * y * z = 3003 → a + b + c ≤ x + y + z → 
  a + b + c = 45 :=
sorry

end NUMINAMATH_CALUDE_min_sum_box_dimensions_l66_6601


namespace NUMINAMATH_CALUDE_larger_number_proof_l66_6619

/-- Given two positive integers with the specified HCF and LCM factors, 
    prove that the larger of the two numbers is 3289 -/
theorem larger_number_proof (a b : ℕ+) 
  (hcf_condition : Nat.gcd a b = 23)
  (lcm_condition : ∃ k : ℕ+, Nat.lcm a b = 23 * 11 * 13 * 15^2 * k) :
  max a b = 3289 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l66_6619


namespace NUMINAMATH_CALUDE_max_roses_325_l66_6611

/-- Represents the pricing options for roses -/
structure RosePricing where
  individual_price : ℚ
  dozen_price : ℚ
  two_dozen_price : ℚ

/-- Calculates the maximum number of roses that can be purchased given a budget and pricing options -/
def max_roses (budget : ℚ) (pricing : RosePricing) : ℕ :=
  sorry

/-- The theorem stating that given the specific pricing and budget, 325 roses is the maximum that can be purchased -/
theorem max_roses_325 :
  let pricing : RosePricing := {
    individual_price := 23/10,
    dozen_price := 36,
    two_dozen_price := 50
  }
  max_roses 680 pricing = 325 := by sorry

end NUMINAMATH_CALUDE_max_roses_325_l66_6611


namespace NUMINAMATH_CALUDE_menu_restriction_l66_6623

theorem menu_restriction (total_dishes : ℕ) (sugar_free_ratio : ℚ) (shellfish_free_ratio : ℚ)
  (h1 : sugar_free_ratio = 1 / 10)
  (h2 : shellfish_free_ratio = 3 / 4) :
  (sugar_free_ratio * shellfish_free_ratio : ℚ) = 3 / 40 := by
  sorry

end NUMINAMATH_CALUDE_menu_restriction_l66_6623


namespace NUMINAMATH_CALUDE_f_properties_l66_6617

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 4
  else if x ≤ 4 then x^2 - 2*x
  else -x + 2

theorem f_properties :
  (f 0 = 4) ∧
  (f 5 = -3) ∧
  (f (f (f 5)) = -1) ∧
  (∃! a, f a = 8 ∧ a = 4) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l66_6617


namespace NUMINAMATH_CALUDE_difference_half_and_sixth_l66_6603

theorem difference_half_and_sixth (x : ℝ) (hx : x = 1/2 - 1/6) : x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_difference_half_and_sixth_l66_6603


namespace NUMINAMATH_CALUDE_function_minimum_value_equality_condition_l66_6687

theorem function_minimum_value (x : ℝ) (h : x > 0) : x^2 + 2/x ≥ 3 := by sorry

theorem equality_condition (x : ℝ) (h : x > 0) : x^2 + 2/x = 3 ↔ x = 1 := by sorry

end NUMINAMATH_CALUDE_function_minimum_value_equality_condition_l66_6687


namespace NUMINAMATH_CALUDE_hyperbola_equation_l66_6613

/-- Given a parabola and a hyperbola with specific properties, 
    prove that the standard equation of the hyperbola is x² - y²/2 = 1 -/
theorem hyperbola_equation 
  (parabola : ℝ → ℝ → Prop) 
  (hyperbola : ℝ → ℝ → ℝ → ℝ → Prop)
  (a b : ℝ)
  (A B F : ℝ × ℝ)
  (h_parabola : ∀ x y, parabola x y ↔ y^2 = 4 * Real.sqrt 3 * x)
  (h_hyperbola : ∀ x y, hyperbola a b x y ↔ x^2 / a^2 - y^2 / b^2 = 1)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_intersect : parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
                 hyperbola a b A.1 A.2 ∧ hyperbola a b B.1 B.2)
  (h_A_above_B : A.2 > B.2)
  (h_asymptote : ∀ x, b * x / a = Real.sqrt 2 * x)
  (h_F_focus : F = (Real.sqrt 3, 0))
  (h_equilateral : 
    (A.1 - F.1)^2 + (A.2 - F.2)^2 = 
    (B.1 - F.1)^2 + (B.2 - F.2)^2 ∧
    (A.1 - F.1)^2 + (A.2 - F.2)^2 = 
    (A.1 - B.1)^2 + (A.2 - B.2)^2) :
  ∀ x y, hyperbola a b x y ↔ x^2 - y^2 / 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l66_6613


namespace NUMINAMATH_CALUDE_projection_vector_equals_result_l66_6645

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-1, 2]

theorem projection_vector_equals_result :
  let proj := (a • b) / (b • b) • b
  proj 0 = -3/5 ∧ proj 1 = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_projection_vector_equals_result_l66_6645


namespace NUMINAMATH_CALUDE_vectors_not_collinear_l66_6646

/-- Given two vectors in ℝ³, construct two new vectors and prove they are not collinear -/
theorem vectors_not_collinear :
  let a : Fin 3 → ℝ := ![1, 2, -3]
  let b : Fin 3 → ℝ := ![2, -1, -1]
  let c₁ : Fin 3 → ℝ := fun i => 4 * a i + 3 * b i
  let c₂ : Fin 3 → ℝ := fun i => 8 * a i - b i
  ¬ ∃ (k : ℝ), c₁ = fun i => k * c₂ i :=
by
  sorry


end NUMINAMATH_CALUDE_vectors_not_collinear_l66_6646


namespace NUMINAMATH_CALUDE_remainder_1234567891011_div_210_l66_6666

theorem remainder_1234567891011_div_210 : 1234567891011 % 210 = 31 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1234567891011_div_210_l66_6666


namespace NUMINAMATH_CALUDE_exists_special_six_digit_number_l66_6644

/-- A six-digit number is between 100000 and 999999 -/
def SixDigitNumber (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

/-- The last six digits of a number -/
def LastSixDigits (n : ℕ) : ℕ := n % 1000000

theorem exists_special_six_digit_number :
  ∃ A : ℕ, SixDigitNumber A ∧
    ∀ k m : ℕ, 1 ≤ k → k < m → m ≤ 500000 →
      LastSixDigits (k * A) ≠ LastSixDigits (m * A) := by
  sorry

end NUMINAMATH_CALUDE_exists_special_six_digit_number_l66_6644


namespace NUMINAMATH_CALUDE_remaining_files_correct_l66_6665

/-- Calculates the number of remaining files on a flash drive -/
def remaining_files (music_files video_files deleted_files : ℕ) : ℕ :=
  music_files + video_files - deleted_files

/-- Theorem: The number of remaining files is correct given the initial conditions -/
theorem remaining_files_correct (music_files video_files deleted_files : ℕ) :
  remaining_files music_files video_files deleted_files =
  music_files + video_files - deleted_files :=
by sorry

end NUMINAMATH_CALUDE_remaining_files_correct_l66_6665


namespace NUMINAMATH_CALUDE_projection_problem_l66_6621

/-- Given a projection that takes (2, -3) to (1, -3/2), 
    prove that the projection of (3, -2) is (24/13, -36/13) -/
theorem projection_problem (proj : ℝ × ℝ → ℝ × ℝ) 
  (h : proj (2, -3) = (1, -3/2)) :
  proj (3, -2) = (24/13, -36/13) := by
  sorry

end NUMINAMATH_CALUDE_projection_problem_l66_6621


namespace NUMINAMATH_CALUDE_product_equality_l66_6695

theorem product_equality : 500 * 2019 * 0.02019 * 5 = 0.25 * 2019^2 := by sorry

end NUMINAMATH_CALUDE_product_equality_l66_6695


namespace NUMINAMATH_CALUDE_probability_at_least_one_from_A_l66_6686

/-- Represents the number of factories in each district -/
structure DistrictFactories where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Represents the number of factories sampled from each district -/
structure SampledFactories where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Calculates the probability of selecting at least one factory from district A 
    when randomly choosing 2 out of 7 stratified sampled factories -/
def probabilityAtLeastOneFromA (df : DistrictFactories) (sf : SampledFactories) : ℚ :=
  sorry

/-- Theorem stating the probability of selecting at least one factory from district A 
    is 11/21 given the specific conditions -/
theorem probability_at_least_one_from_A : 
  let df : DistrictFactories := { A := 18, B := 27, C := 18 }
  let sf : SampledFactories := { A := 2, B := 3, C := 2 }
  probabilityAtLeastOneFromA df sf = 11/21 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_from_A_l66_6686


namespace NUMINAMATH_CALUDE_equation_solvable_l66_6607

/-- For a given real number b, this function represents the equation x - b = ∑_{k=0}^∞ x^k -/
def equation (b x : ℝ) : Prop :=
  x - b = (∑' k, x^k)

/-- This theorem states the conditions on b for which the equation has solutions -/
theorem equation_solvable (b : ℝ) : 
  (∃ x : ℝ, equation b x) ↔ (b ≤ -1 ∨ (-3/2 < b ∧ b ≤ -1)) :=
sorry

end NUMINAMATH_CALUDE_equation_solvable_l66_6607


namespace NUMINAMATH_CALUDE_total_marbles_is_240_l66_6610

/-- The number of marbles in a dozen -/
def dozen : ℕ := 12

/-- The number of red marbles Jessica has -/
def jessica_marbles : ℕ := 3 * dozen

/-- The number of red marbles Sandy has -/
def sandy_marbles : ℕ := 4 * jessica_marbles

/-- The number of red marbles Alex has -/
def alex_marbles : ℕ := jessica_marbles + 2 * dozen

/-- The total number of red marbles Jessica, Sandy, and Alex have -/
def total_marbles : ℕ := jessica_marbles + sandy_marbles + alex_marbles

theorem total_marbles_is_240 : total_marbles = 240 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_is_240_l66_6610


namespace NUMINAMATH_CALUDE_correct_security_response_l66_6643

/-- Represents an email with potentially suspicious characteristics -/
structure Email :=
  (sender : String)
  (content : String)
  (links : List String)

/-- Represents a website with potentially suspicious characteristics -/
structure Website :=
  (url : String)
  (content : String)
  (requestedInfo : List String)

/-- Represents an offer that may be unrealistic -/
structure Offer :=
  (description : String)
  (price : Nat)
  (originalPrice : Nat)

/-- Represents security measures to be followed -/
inductive SecurityMeasure
  | UseSecureNetwork
  | UseAntivirus
  | UpdateApplications
  | CheckHTTPS
  | UseComplexPasswords
  | Use2FA
  | RecognizeBankProtocols

/-- Represents the correct security response -/
structure SecurityResponse :=
  (trustSource : Bool)
  (enterInformation : Bool)
  (measures : List SecurityMeasure)

/-- Main theorem: Given suspicious conditions, prove the correct security response -/
theorem correct_security_response 
  (email : Email) 
  (website : Website) 
  (offer : Offer) : 
  (email.sender ≠ "official@aliexpress.com" ∧ 
   website.url ≠ "https://www.aliexpress.com" ∧ 
   offer.price < offer.originalPrice / 10) → 
  ∃ (response : SecurityResponse), 
    response.trustSource = false ∧ 
    response.enterInformation = false ∧ 
    response.measures.length ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_correct_security_response_l66_6643


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l66_6631

theorem opposite_of_negative_three :
  ∀ x : ℤ, x = -3 → -x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l66_6631


namespace NUMINAMATH_CALUDE_smallest_b_for_real_root_l66_6612

theorem smallest_b_for_real_root : 
  ∀ b : ℕ, (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_real_root_l66_6612


namespace NUMINAMATH_CALUDE_cello_viola_pairs_l66_6652

/-- The number of cellos in stock -/
def num_cellos : ℕ := 800

/-- The number of violas in stock -/
def num_violas : ℕ := 600

/-- The probability of randomly choosing a cello and a viola made from the same tree -/
def same_tree_prob : ℚ := 1 / 4800

/-- The number of cello-viola pairs made with wood from the same tree -/
def num_pairs : ℕ := 100

theorem cello_viola_pairs :
  num_pairs = (same_tree_prob * (num_cellos * num_violas : ℚ)).num := by
  sorry

end NUMINAMATH_CALUDE_cello_viola_pairs_l66_6652


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l66_6672

/-- Proves that the cost price of one meter of cloth is 85 rupees -/
theorem cost_price_per_meter
  (total_length : ℕ)
  (total_selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : total_length = 85)
  (h2 : total_selling_price = 8500)
  (h3 : profit_per_meter = 15) :
  (total_selling_price - total_length * profit_per_meter) / total_length = 85 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_l66_6672


namespace NUMINAMATH_CALUDE_sum_of_self_opposite_and_self_reciprocal_l66_6698

theorem sum_of_self_opposite_and_self_reciprocal (a b : ℝ) : 
  ((-a) = a) → ((1 / b) = b) → (a + b = 1 ∨ a + b = -1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_self_opposite_and_self_reciprocal_l66_6698


namespace NUMINAMATH_CALUDE_subset_implies_a_value_l66_6637

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 = 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

-- State the theorem
theorem subset_implies_a_value (a : ℝ) : B a ⊆ A → a ∈ ({0, 1, -1} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_value_l66_6637


namespace NUMINAMATH_CALUDE_extracurricular_materials_selection_l66_6673

theorem extracurricular_materials_selection : 
  let total_materials : ℕ := 6
  let materials_per_student : ℕ := 2
  let common_materials : ℕ := 1
  
  (total_materials.choose common_materials) * 
  ((total_materials - common_materials).choose (materials_per_student - common_materials)) = 60 :=
by sorry

end NUMINAMATH_CALUDE_extracurricular_materials_selection_l66_6673


namespace NUMINAMATH_CALUDE_sara_initial_quarters_l66_6632

/-- The number of quarters Sara had initially -/
def initial_quarters : ℕ := sorry

/-- The number of quarters Sara's dad gave her -/
def dads_gift : ℕ := 49

/-- The total number of quarters Sara has after her dad's gift -/
def total_quarters : ℕ := 70

/-- Theorem stating that Sara's initial number of quarters was 21 -/
theorem sara_initial_quarters : initial_quarters = 21 := by
  sorry

end NUMINAMATH_CALUDE_sara_initial_quarters_l66_6632


namespace NUMINAMATH_CALUDE_applicant_a_wins_l66_6690

/-- Represents an applicant with their test scores -/
structure Applicant where
  education : ℝ
  experience : ℝ
  work_attitude : ℝ

/-- Calculates the final score of an applicant given the weights -/
def final_score (a : Applicant) (w_edu w_exp w_att : ℝ) : ℝ :=
  a.education * w_edu + a.experience * w_exp + a.work_attitude * w_att

/-- Theorem stating that Applicant A's final score is higher than Applicant B's -/
theorem applicant_a_wins (applicant_a applicant_b : Applicant)
    (h_a_edu : applicant_a.education = 7)
    (h_a_exp : applicant_a.experience = 8)
    (h_a_att : applicant_a.work_attitude = 9)
    (h_b_edu : applicant_b.education = 10)
    (h_b_exp : applicant_b.experience = 7)
    (h_b_att : applicant_b.work_attitude = 8) :
    final_score applicant_a (1/6) (1/3) (1/2) > final_score applicant_b (1/6) (1/3) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_applicant_a_wins_l66_6690


namespace NUMINAMATH_CALUDE_cupcake_frosting_l66_6676

theorem cupcake_frosting (cagney_rate lacey_rate lacey_rest total_time : ℕ) :
  cagney_rate = 15 →
  lacey_rate = 25 →
  lacey_rest = 10 →
  total_time = 480 →
  (total_time : ℚ) / ((1 : ℚ) / cagney_rate + (1 : ℚ) / (lacey_rate + lacey_rest)) = 45 :=
by sorry

end NUMINAMATH_CALUDE_cupcake_frosting_l66_6676


namespace NUMINAMATH_CALUDE_collinear_points_reciprocal_sum_l66_6648

theorem collinear_points_reciprocal_sum (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃ (t : ℝ), (2 + t * (a - 2), 2 + t * (-2)) = (0, b)) →
  1 / a + 1 / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_reciprocal_sum_l66_6648


namespace NUMINAMATH_CALUDE_log_function_range_l66_6633

theorem log_function_range (a : ℝ) : 
  (∀ x : ℝ, x ≥ 2 → |Real.log x / Real.log a| > 1) ↔ 
  (a > 1/2 ∧ a < 1) ∨ (a > 1 ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_log_function_range_l66_6633


namespace NUMINAMATH_CALUDE_equidistant_point_existence_l66_6624

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane represented by y = mx + b -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Distance between a point and a circle -/
def distanceToCircle (p : Point) (c : Circle) : ℝ :=
  sorry

/-- Distance between a point and a line -/
def distanceToLine (p : Point) (l : Line) : ℝ :=
  sorry

/-- The main theorem -/
theorem equidistant_point_existence (c : Circle) (upper_tangent lower_tangent : Line) :
  c.radius = 5 →
  distanceToLine (0, c.center.2 + c.radius) upper_tangent = 3 →
  distanceToLine (0, c.center.2 - c.radius) lower_tangent = 7 →
  ∃! p : Point, 
    distanceToCircle p c = distanceToLine p upper_tangent ∧ 
    distanceToCircle p c = distanceToLine p lower_tangent :=
  sorry

end NUMINAMATH_CALUDE_equidistant_point_existence_l66_6624


namespace NUMINAMATH_CALUDE_negative_roots_condition_l66_6663

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - (a + 1) * x + a + 4

-- Define the condition for both roots being negative
def both_roots_negative (a : ℝ) : Prop :=
  ∀ x : ℝ, quadratic a x = 0 → x < 0

-- Theorem statement
theorem negative_roots_condition :
  ∀ a : ℝ, both_roots_negative a ↔ -4 < a ∧ a ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_negative_roots_condition_l66_6663


namespace NUMINAMATH_CALUDE_extra_food_is_zero_point_four_l66_6626

/-- The amount of cat food needed for one cat per day -/
def food_for_one_cat : ℝ := 0.5

/-- The total amount of cat food needed for two cats per day -/
def total_food_for_two_cats : ℝ := 0.9

/-- The extra amount of cat food needed for the second cat per day -/
def extra_food_for_second_cat : ℝ := total_food_for_two_cats - food_for_one_cat

theorem extra_food_is_zero_point_four :
  extra_food_for_second_cat = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_extra_food_is_zero_point_four_l66_6626


namespace NUMINAMATH_CALUDE_passengers_boarding_other_stops_eq_five_l66_6642

/-- Calculates the number of passengers who got on the bus at other stops -/
def passengers_boarding_other_stops (initial : ℕ) (first_stop : ℕ) (getting_off : ℕ) (final : ℕ) : ℕ :=
  final - (initial + first_stop - getting_off)

/-- Theorem: Given the initial, first stop, getting off, and final passenger counts, 
    prove that 5 passengers got on at other stops -/
theorem passengers_boarding_other_stops_eq_five :
  passengers_boarding_other_stops 50 16 22 49 = 5 := by
  sorry

end NUMINAMATH_CALUDE_passengers_boarding_other_stops_eq_five_l66_6642


namespace NUMINAMATH_CALUDE_triangle_c_coordinates_l66_6684

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a given line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Defines the Euler line of a triangle -/
def Triangle.eulerLine (t : Triangle) : Line :=
  { a := 1, b := -1, c := 2 }

/-- Theorem: If triangle ABC has vertices A(2,0) and B(0,4), and its Euler line 
    is x-y+2=0, then the coordinates of C must be (-4,0) -/
theorem triangle_c_coordinates (t : Triangle) : 
  t.A = { x := 2, y := 0 } →
  t.B = { x := 0, y := 4 } →
  (t.eulerLine = { a := 1, b := -1, c := 2 }) →
  t.C = { x := -4, y := 0 } :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_c_coordinates_l66_6684


namespace NUMINAMATH_CALUDE_exponential_inequality_l66_6681

theorem exponential_inequality (x y a : ℝ) (hx : x > y) (hy : y > 1) (ha : 0 < a) (ha1 : a < 1) :
  a^x < a^y := by sorry

end NUMINAMATH_CALUDE_exponential_inequality_l66_6681


namespace NUMINAMATH_CALUDE_hyperbola_equation_l66_6693

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * Real.sqrt 5 = 2 * Real.sqrt (a^2 + b^2)) →
  (b / a = 1 / 2) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 4 - y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l66_6693


namespace NUMINAMATH_CALUDE_unique_increasing_function_l66_6668

def f (x : ℕ) : ℤ := x^3 - 1

theorem unique_increasing_function :
  (∀ x y : ℕ, x < y → f x < f y) ∧
  f 2 = 7 ∧
  (∀ m n : ℕ, f (m * n) = f m + f n + f m * f n) ∧
  (∀ g : ℕ → ℤ, 
    (∀ x y : ℕ, x < y → g x < g y) →
    g 2 = 7 →
    (∀ m n : ℕ, g (m * n) = g m + g n + g m * g n) →
    ∀ x : ℕ, g x = f x) :=
by sorry

end NUMINAMATH_CALUDE_unique_increasing_function_l66_6668


namespace NUMINAMATH_CALUDE_tangent_line_equation_l66_6605

/-- The function f(x) = x^3 + 2x -/
def f (x : ℝ) : ℝ := x^3 + 2*x

/-- The derivative of f(x) -/
def f_derivative (x : ℝ) : ℝ := 3*x^2 + 2

/-- The point of tangency -/
def point_of_tangency : ℝ × ℝ := (1, f 1)

/-- Theorem: The equation of the tangent line to y = f(x) at (1, f(1)) is 5x - y - 2 = 0 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, (x, y) ∈ {(x, y) | 5*x - y - 2 = 0} ↔ 
  y - (f 1) = (f_derivative 1) * (x - 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l66_6605


namespace NUMINAMATH_CALUDE_range_of_a_l66_6604

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 7)
  (h_f2 : f 2 > 1)
  (h_f2014 : f 2014 = (a + 3) / (a - 3)) :
  0 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l66_6604


namespace NUMINAMATH_CALUDE_digit_multiplication_problem_l66_6630

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Checks if all elements in a list are different -/
def all_different (list : List Digit) : Prop :=
  ∀ i j, i ≠ j → list.get i ≠ list.get j

/-- Converts a three-digit number represented by digits to a natural number -/
def to_nat_3digit (a b c : Digit) : ℕ :=
  100 * a.val + 10 * b.val + c.val

/-- Converts a two-digit number represented by digits to a natural number -/
def to_nat_2digit (d e : Digit) : ℕ :=
  10 * d.val + e.val

/-- Converts a four-digit number represented by digits to a natural number -/
def to_nat_4digit (d1 d2 e1 e2 : Digit) : ℕ :=
  1000 * d1.val + 100 * d2.val + 10 * e1.val + e2.val

theorem digit_multiplication_problem (A B C D E : Digit) :
  all_different [A, B, C, D, E] →
  to_nat_3digit A B C * to_nat_2digit D E = to_nat_4digit D D E E →
  A.val + B.val = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_multiplication_problem_l66_6630


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_180_l66_6616

def sum_of_divisors (n : ℕ) : ℕ := sorry

def largest_prime_factor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_180 :
  largest_prime_factor (sum_of_divisors 180) = 13 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_180_l66_6616


namespace NUMINAMATH_CALUDE_cos_alpha_value_l66_6675

theorem cos_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (α / 2) + Real.cos (α / 2) = Real.sqrt 6 / 2) : 
  Real.cos α = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l66_6675


namespace NUMINAMATH_CALUDE_great_wall_scientific_notation_l66_6683

/-- Represents the length of the Great Wall in meters -/
def great_wall_length : ℝ := 6700010

/-- Converts a number to scientific notation with two significant figures -/
def to_scientific_notation (x : ℝ) : ℝ × ℤ :=
  sorry

/-- Theorem stating that the scientific notation of the Great Wall's length
    is equal to 6.7 × 10^6 when rounded to two significant figures -/
theorem great_wall_scientific_notation :
  to_scientific_notation great_wall_length = (6.7, 6) :=
sorry

end NUMINAMATH_CALUDE_great_wall_scientific_notation_l66_6683


namespace NUMINAMATH_CALUDE_twelve_rolls_in_case_l66_6650

/-- Calculates the number of rolls in a case of paper towels given the case price, individual roll price, and savings percentage. -/
def rolls_in_case (case_price : ℚ) (roll_price : ℚ) (savings_percent : ℚ) : ℚ :=
  case_price / (roll_price * (1 - savings_percent / 100))

/-- Theorem stating that there are 12 rolls in the case under the given conditions. -/
theorem twelve_rolls_in_case :
  rolls_in_case 9 1 25 = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelve_rolls_in_case_l66_6650


namespace NUMINAMATH_CALUDE_plane_perp_necessary_not_sufficient_l66_6660

/-- Two planes are perpendicular -/
def planes_perpendicular (α β : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perpendicular_to_plane (m : Line) (β : Plane) : Prop := sorry

/-- A line lies in a plane -/
def line_in_plane (m : Line) (α : Plane) : Prop := sorry

theorem plane_perp_necessary_not_sufficient
  (α β : Plane) (m : Line)
  (h_different : α ≠ β)
  (h_m_in_α : line_in_plane m α) :
  (planes_perpendicular α β → line_perpendicular_to_plane m β) ∧
  ¬(line_perpendicular_to_plane m β → planes_perpendicular α β) :=
sorry

end NUMINAMATH_CALUDE_plane_perp_necessary_not_sufficient_l66_6660


namespace NUMINAMATH_CALUDE_circle_area_ratio_l66_6620

theorem circle_area_ratio (s r : ℝ) (hs : s > 0) (hr : r > 0) 
  (h : r = 0.8 * s) : 
  (π * (r / 2)^2) / (π * (s / 2)^2) = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l66_6620


namespace NUMINAMATH_CALUDE_min_value_implies_a_geq_two_l66_6640

/-- The function f(x) defined as x^2 - 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

/-- Theorem: If the minimum value of f(x) in the interval [-1, 2] is f(2), then a ≥ 2 -/
theorem min_value_implies_a_geq_two (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 2, f a x ≥ f a 2) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_geq_two_l66_6640


namespace NUMINAMATH_CALUDE_certain_number_calculation_l66_6677

theorem certain_number_calculation : ∀ (x y : ℕ),
  x + y = 36 →
  x = 19 →
  8 * x + 3 * y = 203 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_calculation_l66_6677


namespace NUMINAMATH_CALUDE_beakers_with_no_metal_ions_l66_6649

theorem beakers_with_no_metal_ions (total_beakers : Nat) (copper_beakers : Nat) (silver_beakers : Nat)
  (drops_a_per_beaker : Nat) (drops_b_per_beaker : Nat) (total_drops_a : Nat) (total_drops_b : Nat) :
  total_beakers = 50 →
  copper_beakers = 10 →
  silver_beakers = 5 →
  drops_a_per_beaker = 3 →
  drops_b_per_beaker = 4 →
  total_drops_a = 106 →
  total_drops_b = 80 →
  total_beakers - copper_beakers - silver_beakers = 15 :=
by sorry

end NUMINAMATH_CALUDE_beakers_with_no_metal_ions_l66_6649


namespace NUMINAMATH_CALUDE_line_point_distance_condition_l66_6609

theorem line_point_distance_condition (a : ℝ) : 
  (∃ x y : ℝ, a * x + y + 2 = 0 ∧ 
    ((x + 3)^2 + y^2)^(1/2) = 2 * (x^2 + y^2)^(1/2)) → 
  a ≤ 0 ∨ a ≥ 4/3 := by
sorry

end NUMINAMATH_CALUDE_line_point_distance_condition_l66_6609


namespace NUMINAMATH_CALUDE_tetrahedron_medians_intersect_l66_6696

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A₁ : Point3D
  A₂ : Point3D
  A₃ : Point3D
  A₄ : Point3D

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Median of a tetrahedron -/
def median (t : Tetrahedron) (v : Fin 4) : Line3D :=
  sorry  -- Definition of median based on tetrahedron and vertex index

/-- Intersection point of two lines -/
def intersectionPoint (l1 l2 : Line3D) : Option Point3D :=
  sorry  -- Definition of intersection point of two lines

/-- Theorem: All medians of a tetrahedron intersect at a single point -/
theorem tetrahedron_medians_intersect (t : Tetrahedron) :
  ∃ (c : Point3D), ∀ (i j : Fin 4),
    intersectionPoint (median t i) (median t j) = some c :=
  sorry

end NUMINAMATH_CALUDE_tetrahedron_medians_intersect_l66_6696


namespace NUMINAMATH_CALUDE_parabola_distance_theorem_l66_6671

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def B : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem parabola_distance_theorem (A : ℝ × ℝ) :
  parabola A.1 A.2 →  -- A lies on the parabola
  (A.1 - focus.1)^2 + (A.2 - focus.2)^2 = (B.1 - focus.1)^2 + (B.2 - focus.2)^2 →  -- |AF| = |BF|
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8 :=  -- |AB| = 2√2
by sorry

end NUMINAMATH_CALUDE_parabola_distance_theorem_l66_6671


namespace NUMINAMATH_CALUDE_max_sum_is_1446_l66_6655

/-- Represents a cube with six faces --/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- The set of numbers on each cube --/
def cube_numbers : Finset ℕ := {1, 3, 9, 27, 81, 243}

/-- A valid cube has all numbers from cube_numbers --/
def is_valid_cube (c : Cube) : Prop :=
  ∀ n ∈ cube_numbers, ∃ i : Fin 6, c.faces i = n

/-- The sum of visible faces when cubes are stacked --/
def visible_sum (cubes : Fin 4 → Cube) : ℕ :=
  sorry

/-- The maximum possible sum of visible faces --/
def max_visible_sum : ℕ :=
  sorry

/-- The main theorem to prove --/
theorem max_sum_is_1446 :
  ∀ cubes : Fin 4 → Cube,
  (∀ i : Fin 4, is_valid_cube (cubes i)) →
  visible_sum cubes ≤ 1446 ∧
  ∃ optimal_cubes : Fin 4 → Cube,
    (∀ i : Fin 4, is_valid_cube (optimal_cubes i)) ∧
    visible_sum optimal_cubes = 1446 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_is_1446_l66_6655


namespace NUMINAMATH_CALUDE_decreasing_f_implies_a_range_l66_6635

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - a*x + 5 else a/x

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

theorem decreasing_f_implies_a_range (a : ℝ) :
  is_decreasing (f a) → 2 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_f_implies_a_range_l66_6635
