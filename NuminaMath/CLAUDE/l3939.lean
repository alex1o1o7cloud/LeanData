import Mathlib

namespace NUMINAMATH_CALUDE_prob_A_value_l3939_393942

/-- The probability of producing a grade B product -/
def prob_B : ℝ := 0.05

/-- The probability of producing a grade C product -/
def prob_C : ℝ := 0.03

/-- The probability of a randomly inspected product being grade A (non-defective) -/
def prob_A : ℝ := 1 - prob_B - prob_C

theorem prob_A_value : prob_A = 0.92 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_value_l3939_393942


namespace NUMINAMATH_CALUDE_unique_prime_pair_with_prime_root_l3939_393976

theorem unique_prime_pair_with_prime_root :
  ∃! (m n : ℕ), Prime m ∧ Prime n ∧
  (∃ x : ℕ, Prime x ∧ x^2 - m*x - n = 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_with_prime_root_l3939_393976


namespace NUMINAMATH_CALUDE_increasing_digits_mod_1000_l3939_393987

/-- The number of 8-digit positive integers with digits in increasing order -/
def count_increasing_digits : ℕ := (Nat.choose 17 8)

/-- The theorem stating that the count of such integers is congruent to 310 modulo 1000 -/
theorem increasing_digits_mod_1000 :
  count_increasing_digits % 1000 = 310 := by sorry

end NUMINAMATH_CALUDE_increasing_digits_mod_1000_l3939_393987


namespace NUMINAMATH_CALUDE_exponent_negative_product_squared_l3939_393917

theorem exponent_negative_product_squared (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_negative_product_squared_l3939_393917


namespace NUMINAMATH_CALUDE_checkerboard_coverage_l3939_393986

/-- Represents a checkerboard --/
structure Checkerboard where
  rows : ℕ
  cols : ℕ
  removed_squares : ℕ

/-- Checks if a checkerboard can be covered by dominoes --/
def can_be_covered (board : Checkerboard) : Prop :=
  (board.rows * board.cols - board.removed_squares) % 2 = 0

/-- Theorem stating which boards can be covered --/
theorem checkerboard_coverage (board : Checkerboard) :
  can_be_covered board ↔ 
  (board ≠ ⟨4, 4, 1⟩ ∧ board ≠ ⟨3, 7, 0⟩ ∧ board ≠ ⟨7, 3, 0⟩) :=
sorry

end NUMINAMATH_CALUDE_checkerboard_coverage_l3939_393986


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_16_l3939_393974

theorem sqrt_of_sqrt_16 : ∃ (x : ℝ), x^2 = 16 ∧ (∀ y : ℝ, y^2 = x → y = 2 ∨ y = -2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_16_l3939_393974


namespace NUMINAMATH_CALUDE_salad_bar_olive_count_l3939_393921

theorem salad_bar_olive_count (lettuce_types : Nat) (tomato_types : Nat) (soup_types : Nat) (total_options : Nat) (olive_types : Nat) : 
  lettuce_types = 2 →
  tomato_types = 3 →
  soup_types = 2 →
  total_options = 48 →
  total_options = lettuce_types * tomato_types * olive_types * soup_types →
  olive_types = 4 := by
sorry

end NUMINAMATH_CALUDE_salad_bar_olive_count_l3939_393921


namespace NUMINAMATH_CALUDE_escalator_speed_l3939_393950

/-- Proves that the escalator speed is 12 feet per second given the conditions -/
theorem escalator_speed (escalator_length : ℝ) (person_speed : ℝ) (time_taken : ℝ) :
  escalator_length = 210 →
  person_speed = 2 →
  time_taken = 15 →
  (person_speed + (escalator_length / time_taken)) * time_taken = escalator_length →
  escalator_length / time_taken = 12 := by
  sorry


end NUMINAMATH_CALUDE_escalator_speed_l3939_393950


namespace NUMINAMATH_CALUDE_gloria_purchase_l3939_393966

/-- The cost of items at a store -/
structure StorePrices where
  pencil : ℕ
  notebook : ℕ
  eraser : ℕ

/-- The conditions from the problem -/
def store_conditions (p : StorePrices) : Prop :=
  p.pencil + p.notebook = 80 ∧
  p.pencil + p.eraser = 45 ∧
  3 * p.pencil + 3 * p.notebook + 3 * p.eraser = 315

/-- The theorem to prove -/
theorem gloria_purchase (p : StorePrices) : 
  store_conditions p → p.notebook + p.eraser = 85 := by
  sorry

end NUMINAMATH_CALUDE_gloria_purchase_l3939_393966


namespace NUMINAMATH_CALUDE_first_day_exceeding_150_l3939_393964

def paperclips : ℕ → ℕ
  | 0 => 5  -- Monday (day 1)
  | n + 1 => 2 * paperclips n + 2

theorem first_day_exceeding_150 :
  ∃ n : ℕ, paperclips n > 150 ∧ ∀ m : ℕ, m < n → paperclips m ≤ 150 ∧ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_first_day_exceeding_150_l3939_393964


namespace NUMINAMATH_CALUDE_phillip_test_results_l3939_393940

/-- Represents the number of questions Phillip gets right on a test -/
def correct_answers (total : ℕ) (percentage : ℚ) : ℚ :=
  (total : ℚ) * percentage

/-- Represents the total number of correct answers across all tests -/
def total_correct_answers (x : ℕ) : ℚ :=
  correct_answers 40 (75/100) + correct_answers 50 (98/100) + (x : ℚ) * ((100 - x : ℚ)/100)

theorem phillip_test_results (x : ℕ) (h : 1 ≤ x ∧ x ≤ 100) :
  total_correct_answers x = 79 + (x : ℚ) * ((100 - x : ℚ)/100) :=
by sorry

end NUMINAMATH_CALUDE_phillip_test_results_l3939_393940


namespace NUMINAMATH_CALUDE_maggie_bouncy_balls_l3939_393926

/-- The number of packs of red bouncy balls -/
def red_packs : ℕ := 4

/-- The number of balls in each pack of red bouncy balls -/
def red_per_pack : ℕ := 12

/-- The number of packs of yellow bouncy balls -/
def yellow_packs : ℕ := 8

/-- The number of balls in each pack of yellow bouncy balls -/
def yellow_per_pack : ℕ := 10

/-- The number of packs of green bouncy balls -/
def green_packs : ℕ := 4

/-- The number of balls in each pack of green bouncy balls -/
def green_per_pack : ℕ := 14

/-- The number of packs of blue bouncy balls -/
def blue_packs : ℕ := 6

/-- The number of balls in each pack of blue bouncy balls -/
def blue_per_pack : ℕ := 8

/-- The total number of bouncy balls Maggie bought -/
def total_balls : ℕ := red_packs * red_per_pack + yellow_packs * yellow_per_pack + 
                        green_packs * green_per_pack + blue_packs * blue_per_pack

theorem maggie_bouncy_balls : total_balls = 232 := by
  sorry

end NUMINAMATH_CALUDE_maggie_bouncy_balls_l3939_393926


namespace NUMINAMATH_CALUDE_race_distance_proof_l3939_393906

/-- The distance of a race where:
  * A covers the distance in 36 seconds
  * B covers the distance in 45 seconds
  * A beats B by 22 meters
-/
def race_distance : ℝ := 110

theorem race_distance_proof (A_time B_time : ℝ) (beat_distance : ℝ) 
  (h1 : A_time = 36)
  (h2 : B_time = 45)
  (h3 : beat_distance = 22)
  (h4 : A_time * (race_distance / B_time) + beat_distance = race_distance) :
  race_distance = 110 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_proof_l3939_393906


namespace NUMINAMATH_CALUDE_sine_cosine_relation_l3939_393937

theorem sine_cosine_relation (θ : ℝ) (h : Real.cos (3 * Real.pi / 14 - θ) = 1 / 3) :
  Real.sin (2 * Real.pi / 7 + θ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_relation_l3939_393937


namespace NUMINAMATH_CALUDE_trigonometric_problem_l3939_393965

theorem trigonometric_problem (α β : Real) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2)
  (h_distance : Real.sqrt ((Real.cos α - Real.cos β)^2 + (Real.sin α - Real.sin β)^2) = Real.sqrt 10 / 5)
  (h_tan : Real.tan (α/2) = 1/2) :
  Real.cos (α - β) = 4/5 ∧ Real.cos α = 3/5 ∧ Real.cos β = 24/25 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l3939_393965


namespace NUMINAMATH_CALUDE_solve_potatoes_problem_l3939_393992

def potatoes_problem (initial : ℕ) (gina : ℕ) : Prop :=
  let tom := 2 * gina
  let anne := tom / 3
  let remaining := initial - (gina + tom + anne)
  remaining = 47

theorem solve_potatoes_problem :
  potatoes_problem 300 69 := by
  sorry

end NUMINAMATH_CALUDE_solve_potatoes_problem_l3939_393992


namespace NUMINAMATH_CALUDE_organization_member_count_organization_has_ten_members_l3939_393978

/-- Represents an organization with committees and members -/
structure Organization :=
  (num_committees : ℕ)
  (num_members : ℕ)
  (member_committee_count : ℕ)
  (shared_member_count : ℕ)

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem stating the required number of members in the organization -/
theorem organization_member_count (org : Organization) 
  (h1 : org.num_committees = 5)
  (h2 : org.member_committee_count = 2)
  (h3 : org.shared_member_count = 1) :
  org.num_members = choose_two org.num_committees :=
by sorry

/-- The main theorem proving the organization must have 10 members -/
theorem organization_has_ten_members (org : Organization) 
  (h1 : org.num_committees = 5)
  (h2 : org.member_committee_count = 2)
  (h3 : org.shared_member_count = 1) :
  org.num_members = 10 :=
by sorry

end NUMINAMATH_CALUDE_organization_member_count_organization_has_ten_members_l3939_393978


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3939_393929

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) →
  a + b = -14 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3939_393929


namespace NUMINAMATH_CALUDE_num_distinct_arrangements_l3939_393910

-- Define a cube with six faces
inductive Face
| F | E | A | B | H | J

-- Define an arrangement as a function from Face to Int
def Arrangement := Face → Fin 6

-- Define adjacency for cube faces
def adjacent : Face → Face → Prop :=
  sorry

-- Define if two numbers are consecutive (including 6 and 1)
def consecutive (a b : Fin 6) : Prop :=
  (a + 1 = b) ∨ (a = 5 ∧ b = 0)

-- Define a valid arrangement
def valid_arrangement (arr : Arrangement) : Prop :=
  ∀ (f1 f2 : Face), adjacent f1 f2 → consecutive (arr f1) (arr f2)

-- Define equivalence of arrangements under cube symmetry and cyclic permutation
def equivalent_arrangements (arr1 arr2 : Arrangement) : Prop :=
  sorry

-- The main theorem
theorem num_distinct_arrangements :
  ∃ (arr1 arr2 : Arrangement),
    valid_arrangement arr1 ∧
    valid_arrangement arr2 ∧
    ¬equivalent_arrangements arr1 arr2 ∧
    ∀ (arr : Arrangement), valid_arrangement arr →
      equivalent_arrangements arr arr1 ∨ equivalent_arrangements arr arr2 :=
  sorry

end NUMINAMATH_CALUDE_num_distinct_arrangements_l3939_393910


namespace NUMINAMATH_CALUDE_find_c_l3939_393944

theorem find_c (a b c : ℝ) : 
  (∀ x, (x + 3) * (x + b) = x^2 + c*x + 15) → c = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_c_l3939_393944


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l3939_393924

theorem consecutive_integers_product_sum (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l3939_393924


namespace NUMINAMATH_CALUDE_det_trig_matrix_zero_l3939_393963

theorem det_trig_matrix_zero (α β : Real) : 
  let M : Matrix (Fin 3) (Fin 3) Real := ![
    ![0, Real.cos α, Real.sin α],
    ![-Real.cos α, 0, Real.cos β],
    ![-Real.sin α, -Real.cos β, 0]
  ]
  Matrix.det M = 0 := by
sorry

end NUMINAMATH_CALUDE_det_trig_matrix_zero_l3939_393963


namespace NUMINAMATH_CALUDE_intersection_point_equality_l3939_393922

theorem intersection_point_equality (a b c d : ℝ) : 
  (1 = 1^2 + a * 1 + b) → 
  (1 = 1^2 + c * 1 + d) → 
  a^5 + d^6 = c^6 - b^5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_equality_l3939_393922


namespace NUMINAMATH_CALUDE_dhoni_leftover_percentage_l3939_393925

/-- The percentage of earnings Dhoni had left over after spending on rent and a dishwasher -/
theorem dhoni_leftover_percentage : ℝ := by
  -- Define the percentage spent on rent
  let rent_percentage : ℝ := 20
  -- Define the percentage spent on dishwasher (5% less than rent)
  let dishwasher_percentage : ℝ := rent_percentage - 5
  -- Define the total percentage spent
  let total_spent_percentage : ℝ := rent_percentage + dishwasher_percentage
  -- Define the leftover percentage
  let leftover_percentage : ℝ := 100 - total_spent_percentage
  -- Prove that the leftover percentage is 65%
  have : leftover_percentage = 65 := by sorry
  -- Return the result
  exact leftover_percentage

end NUMINAMATH_CALUDE_dhoni_leftover_percentage_l3939_393925


namespace NUMINAMATH_CALUDE_sqrt_pattern_l3939_393918

theorem sqrt_pattern (n : ℕ+) : 
  Real.sqrt (1 + 1 / n^2 + 1 / (n + 1)^2) = 1 + 1 / (n * (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_pattern_l3939_393918


namespace NUMINAMATH_CALUDE_max_thursday_hours_l3939_393949

def max_video_game_hours (wednesday : ℝ) (friday : ℝ) (average : ℝ) : Prop :=
  ∃ thursday : ℝ,
    wednesday = 2 ∧
    friday > wednesday + 3 ∧
    average = 3 ∧
    (wednesday + thursday + friday) / 3 = average ∧
    thursday = 2

theorem max_thursday_hours :
  max_video_game_hours 2 5 3 :=
sorry

end NUMINAMATH_CALUDE_max_thursday_hours_l3939_393949


namespace NUMINAMATH_CALUDE_workshop_average_salary_l3939_393907

theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (avg_salary_technicians : ℕ)
  (avg_salary_rest : ℕ)
  (h1 : total_workers = 42)
  (h2 : technicians = 7)
  (h3 : avg_salary_technicians = 18000)
  (h4 : avg_salary_rest = 6000) :
  (technicians * avg_salary_technicians + (total_workers - technicians) * avg_salary_rest) / total_workers = 8000 := by
  sorry

#check workshop_average_salary

end NUMINAMATH_CALUDE_workshop_average_salary_l3939_393907


namespace NUMINAMATH_CALUDE_stacy_pages_per_day_l3939_393923

/-- Given a paper with a certain number of pages due in a certain number of days,
    calculate the number of pages that need to be written per day to finish on time. -/
def pages_per_day (total_pages : ℕ) (total_days : ℕ) : ℚ :=
  total_pages / total_days

/-- Theorem: Stacy needs to write 1 page per day to finish her paper on time. -/
theorem stacy_pages_per_day :
  pages_per_day 12 12 = 1 := by
  sorry

#eval pages_per_day 12 12

end NUMINAMATH_CALUDE_stacy_pages_per_day_l3939_393923


namespace NUMINAMATH_CALUDE_distinct_sets_count_l3939_393962

def A : Finset ℕ := {1, 2, 3, 4}
def B : Finset ℕ := {5, 6, 7}
def C : Finset ℕ := {8, 9}

def form_sets (X Y : Finset ℕ) : Finset (Finset ℕ) :=
  (X.product Y).image (λ (x, y) => {x, y})

theorem distinct_sets_count :
  (form_sets A B ∪ form_sets A C ∪ form_sets B C).card = 26 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sets_count_l3939_393962


namespace NUMINAMATH_CALUDE_converse_A_false_others_true_l3939_393953

-- Define the basic geometric concepts
structure Triangle where
  angles : Fin 3 → ℝ
  sides : Fin 3 → ℝ

def is_congruent (t1 t2 : Triangle) : Prop := sorry

def is_right_triangle (t : Triangle) : Prop := sorry

def is_equilateral (t : Triangle) : Prop := sorry

def are_complementary (a b : ℝ) : Prop := sorry

-- Define the statements and their converses
def statement_A (t1 t2 : Triangle) : Prop :=
  is_congruent t1 t2 → ∀ i : Fin 3, t1.angles i = t2.angles i

def converse_A (t1 t2 : Triangle) : Prop :=
  (∀ i : Fin 3, t1.angles i = t2.angles i) → is_congruent t1 t2

def statement_B (t : Triangle) : Prop :=
  (∀ i j : Fin 3, t.angles i = t.angles j) → (∀ i j : Fin 3, t.sides i = t.sides j)

def converse_B (t : Triangle) : Prop :=
  (∀ i j : Fin 3, t.sides i = t.sides j) → (∀ i j : Fin 3, t.angles i = t.angles j)

def statement_C (t : Triangle) : Prop :=
  is_right_triangle t → are_complementary (t.angles 0) (t.angles 1)

def converse_C (t : Triangle) : Prop :=
  are_complementary (t.angles 0) (t.angles 1) → is_right_triangle t

def statement_D (t : Triangle) : Prop :=
  is_equilateral t → (∀ i j : Fin 3, t.angles i = t.angles j)

def converse_D (t : Triangle) : Prop :=
  (∀ i j : Fin 3, t.angles i = t.angles j) → is_equilateral t

-- Main theorem
theorem converse_A_false_others_true :
  (∃ t1 t2 : Triangle, converse_A t1 t2 = false) ∧
  (∀ t : Triangle, converse_B t = true) ∧
  (∀ t : Triangle, converse_C t = true) ∧
  (∀ t : Triangle, converse_D t = true) := by sorry

end NUMINAMATH_CALUDE_converse_A_false_others_true_l3939_393953


namespace NUMINAMATH_CALUDE_zero_subset_integers_negation_squared_positive_l3939_393955

-- Define the set containing only 0
def zero_set : Set ℤ := {0}

-- Statement 1: {0} is a subset of ℤ
theorem zero_subset_integers : zero_set ⊆ Set.univ := by sorry

-- Statement 2: Negation of "for all x in ℤ, x² > 0" is "there exists x in ℤ such that x² ≤ 0"
theorem negation_squared_positive :
  (¬ ∀ x : ℤ, x^2 > 0) ↔ (∃ x : ℤ, x^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_zero_subset_integers_negation_squared_positive_l3939_393955


namespace NUMINAMATH_CALUDE_nh3_formation_l3939_393972

-- Define the chemical reaction
structure Reaction where
  nh4no3 : ℕ
  naoh : ℕ
  nh3 : ℕ

-- Define the stoichiometric relationship
def stoichiometric (r : Reaction) : Prop :=
  r.nh4no3 = r.naoh ∧ r.nh3 = r.nh4no3

-- Theorem statement
theorem nh3_formation (r : Reaction) (h : stoichiometric r) :
  r.nh3 = r.nh4no3 := by
  sorry

end NUMINAMATH_CALUDE_nh3_formation_l3939_393972


namespace NUMINAMATH_CALUDE_girls_together_arrangements_boy_A_not_end_two_girls_together_arrangements_l3939_393930

-- Define the number of boys and girls
def num_boys : ℕ := 3
def num_girls : ℕ := 3
def total_students : ℕ := num_boys + num_girls

-- Define the function for the number of arrangements when three girls must stand together
def arrangements_girls_together : ℕ := sorry

-- Define the function for the number of arrangements when boy A cannot stand at either end and exactly two girls stand together
def arrangements_boy_A_not_end_two_girls_together : ℕ := sorry

-- Theorem for the first question
theorem girls_together_arrangements :
  arrangements_girls_together = 144 := by sorry

-- Theorem for the second question
theorem boy_A_not_end_two_girls_together_arrangements :
  arrangements_boy_A_not_end_two_girls_together = 288 := by sorry

end NUMINAMATH_CALUDE_girls_together_arrangements_boy_A_not_end_two_girls_together_arrangements_l3939_393930


namespace NUMINAMATH_CALUDE_cucumber_water_percentage_l3939_393939

theorem cucumber_water_percentage (initial_weight initial_water_percentage new_weight : ℝ) :
  initial_weight = 100 →
  initial_water_percentage = 99 →
  new_weight = 50 →
  let initial_water := initial_weight * (initial_water_percentage / 100)
  let initial_solid := initial_weight - initial_water
  let new_water := new_weight - initial_solid
  new_water / new_weight * 100 = 98 := by
sorry

end NUMINAMATH_CALUDE_cucumber_water_percentage_l3939_393939


namespace NUMINAMATH_CALUDE_expression_value_l3939_393985

theorem expression_value (x y : ℝ) : 
  x^4 + 6*x^2*y + 9*y^2 + 2*x^2 + 6*y + 4 = 7 → 
  x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = -2 ∨ 
  x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = 14 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l3939_393985


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3939_393914

theorem least_addition_for_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < 4 → ¬(5 ∣ (2496 + m))) ∧ (5 ∣ (2496 + 4)) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3939_393914


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l3939_393984

/-- The volume of a rectangular solid with given face areas -/
theorem rectangular_solid_volume
  (side_face_area front_face_area bottom_face_area : ℝ)
  (h_side : side_face_area = 18)
  (h_front : front_face_area = 15)
  (h_bottom : bottom_face_area = 10) :
  ∃ (x y z : ℝ),
    x * y = side_face_area ∧
    y * z = front_face_area ∧
    z * x = bottom_face_area ∧
    x * y * z = 30 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l3939_393984


namespace NUMINAMATH_CALUDE_arithmetic_equality_l3939_393993

theorem arithmetic_equality : 253 - 47 + 29 + 18 = 253 := by sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l3939_393993


namespace NUMINAMATH_CALUDE_no_perfect_squares_l3939_393927

def digit_repeat (d : ℕ) (n : ℕ) : ℕ :=
  d * (10^100 - 1) / 9

def two_digit_repeat (d₁ d₂ : ℕ) (n : ℕ) : ℕ :=
  (10 * d₁ + d₂) * (10^99 - 1) / 99 + d₁ * 10^99

def N₁ : ℕ := digit_repeat 3 100
def N₂ : ℕ := digit_repeat 6 100
def N₃ : ℕ := two_digit_repeat 1 5 100
def N₄ : ℕ := two_digit_repeat 2 1 100
def N₅ : ℕ := two_digit_repeat 2 7 100

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem no_perfect_squares :
  ¬(is_perfect_square N₁ ∨ is_perfect_square N₂ ∨ is_perfect_square N₃ ∨ is_perfect_square N₄ ∨ is_perfect_square N₅) :=
by sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l3939_393927


namespace NUMINAMATH_CALUDE_valid_arrangement_probability_l3939_393959

-- Define the number of teachers and days
def num_teachers : ℕ := 6
def num_days : ℕ := 3
def teachers_per_day : ℕ := 2

-- Define the teachers who have restrictions
structure RestrictedTeacher where
  name : String
  restricted_day : ℕ

-- Define the specific restrictions
def wang : RestrictedTeacher := ⟨"Wang", 2⟩
def li : RestrictedTeacher := ⟨"Li", 3⟩

-- Define the probability function
def probability_of_valid_arrangement (t : ℕ) (d : ℕ) (tpd : ℕ) 
  (r1 r2 : RestrictedTeacher) : ℚ :=
  7/15

-- State the theorem
theorem valid_arrangement_probability :
  probability_of_valid_arrangement num_teachers num_days teachers_per_day wang li = 7/15 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangement_probability_l3939_393959


namespace NUMINAMATH_CALUDE_radio_selling_price_l3939_393957

/-- Calculates the selling price of a radio given the purchase price, overhead expenses, and profit percentage. -/
def calculate_selling_price (purchase_price : ℚ) (overhead : ℚ) (profit_percent : ℚ) : ℚ :=
  let total_cost := purchase_price + overhead
  let profit := (profit_percent / 100) * total_cost
  total_cost + profit

/-- Theorem stating that the selling price of the radio is 300 given the specified conditions. -/
theorem radio_selling_price :
  calculate_selling_price 225 28 (18577075098814234 / 1000000000) = 300 := by
  sorry

end NUMINAMATH_CALUDE_radio_selling_price_l3939_393957


namespace NUMINAMATH_CALUDE_value_of_expression_l3939_393908

theorem value_of_expression (m n : ℝ) 
  (h1 : m^2 + 2*m*n = 3) 
  (h2 : m*n + n^2 = 4) : 
  m^2 + 3*m*n + n^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3939_393908


namespace NUMINAMATH_CALUDE_silk_order_total_l3939_393956

/-- The number of yards of green silk dyed by the factory -/
def green_silk : ℕ := 61921

/-- The number of yards of pink silk dyed by the factory -/
def pink_silk : ℕ := 49500

/-- The total number of yards of silk dyed by the factory -/
def total_silk : ℕ := green_silk + pink_silk

theorem silk_order_total :
  total_silk = 111421 :=
by sorry

end NUMINAMATH_CALUDE_silk_order_total_l3939_393956


namespace NUMINAMATH_CALUDE_book_discount_l3939_393928

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens ≥ 1 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- The price reduction percentage -/
def reduction_percentage : Rat := 62.5

/-- Calculates the value of a two-digit number -/
def value (n : TwoDigitNumber) : Nat := 10 * n.tens + n.ones

/-- Checks if two TwoDigitNumbers have the same digits in different order -/
def same_digits (n1 n2 : TwoDigitNumber) : Prop :=
  (n1.tens = n2.ones) ∧ (n1.ones = n2.tens)

theorem book_discount (original reduced : TwoDigitNumber)
  (h_reduction : value reduced = (100 - reduction_percentage) / 100 * value original)
  (h_same_digits : same_digits original reduced) :
  value original - value reduced = 45 := by
  sorry

end NUMINAMATH_CALUDE_book_discount_l3939_393928


namespace NUMINAMATH_CALUDE_only_two_is_possible_l3939_393975

/-- Represents a triangular grid with 9 cells -/
def TriangularGrid := Fin 9 → ℤ

/-- Represents a move on the triangular grid -/
inductive Move
| add (i j : Fin 9) : Move
| subtract (i j : Fin 9) : Move

/-- Applies a move to the grid -/
def applyMove (grid : TriangularGrid) (move : Move) : TriangularGrid :=
  match move with
  | Move.add i j => 
      fun k => if k = i ∨ k = j then grid k + 1 else grid k
  | Move.subtract i j => 
      fun k => if k = i ∨ k = j then grid k - 1 else grid k

/-- Checks if two cells are adjacent in the triangular grid -/
def isAdjacent (i j : Fin 9) : Prop := sorry

/-- Checks if a grid contains consecutive natural numbers from n to n+8 -/
def containsConsecutiveNumbers (grid : TriangularGrid) (n : ℕ) : Prop :=
  ∃ (perm : Fin 9 → Fin 9), ∀ i : Fin 9, grid (perm i) = n + i

/-- The main theorem stating that n = 2 is the only solution -/
theorem only_two_is_possible :
  ∀ (n : ℕ),
    (∃ (grid : TriangularGrid) (moves : List Move),
      (∀ i : Fin 9, grid i = 0) ∧
      (∀ move ∈ moves, ∃ i j, move = Move.add i j ∨ move = Move.subtract i j) ∧
      (∀ move ∈ moves, ∃ i j, isAdjacent i j) ∧
      (containsConsecutiveNumbers (moves.foldl applyMove grid) n)) ↔
    n = 2 := by
  sorry


end NUMINAMATH_CALUDE_only_two_is_possible_l3939_393975


namespace NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_24_l3939_393969

theorem prime_square_minus_one_divisible_by_24 (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ k : ℕ, p^2 - 1 = 24 * k := by
  sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_24_l3939_393969


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3939_393911

theorem sum_of_coefficients (f : ℝ → ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, f x = (x - 5)^7 + (x - 8)^5) →
  (∀ x, f x = a₀ + a₁*(x - 6) + a₂*(x - 6)^2 + a₃*(x - 6)^3 + a₄*(x - 6)^4 + 
           a₅*(x - 6)^5 + a₆*(x - 6)^6 + a₇*(x - 6)^7) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 127 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3939_393911


namespace NUMINAMATH_CALUDE_author_earnings_proof_l3939_393999

/-- Calculates the author's earnings from book sales -/
def author_earnings (paper_copies : ℕ) (paper_price : ℚ) (paper_percentage : ℚ)
                    (hard_copies : ℕ) (hard_price : ℚ) (hard_percentage : ℚ) : ℚ :=
  let paper_total := paper_copies * paper_price
  let hard_total := hard_copies * hard_price
  paper_total * paper_percentage + hard_total * hard_percentage

/-- Proves that the author's earnings are $1,104 given the specified conditions -/
theorem author_earnings_proof :
  author_earnings 32000 0.20 0.06 15000 0.40 0.12 = 1104 := by
  sorry

#eval author_earnings 32000 (20 / 100) (6 / 100) 15000 (40 / 100) (12 / 100)

end NUMINAMATH_CALUDE_author_earnings_proof_l3939_393999


namespace NUMINAMATH_CALUDE_midpoint_movement_l3939_393916

/-- Given two points A and B on a Cartesian plane, their midpoint, and their new positions after moving,
    prove that the new midpoint and its distance from the original midpoint are as calculated. -/
theorem midpoint_movement (a b c d m n : ℝ) :
  let A : ℝ × ℝ := (a, b)
  let B : ℝ × ℝ := (c, d)
  let M : ℝ × ℝ := (m, n)
  let A' : ℝ × ℝ := (a + 3, b + 5)
  let B' : ℝ × ℝ := (c - 4, d - 6)
  M = ((a + c) / 2, (b + d) / 2) →
  let M' : ℝ × ℝ := ((A'.1 + B'.1) / 2, (A'.2 + B'.2) / 2)
  M' = (m - 0.5, n - 0.5) ∧
  Real.sqrt ((M'.1 - M.1)^2 + (M'.2 - M.2)^2) = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_movement_l3939_393916


namespace NUMINAMATH_CALUDE_system_solution_l3939_393947

theorem system_solution :
  let solutions : List (ℤ × ℤ × ℤ) := [(15, -1, -2), (-1, 15, -2), (3, -5, 14), (-5, 3, 14)]
  ∀ (x y z : ℤ), (x, y, z) ∈ solutions →
    (x + y + z = 12 ∧ x^2 + y^2 + z^2 = 230 ∧ x * y = -15) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3939_393947


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_17_l3939_393995

theorem three_digit_divisible_by_17 : 
  (Finset.filter (fun k => 100 ≤ 17 * k ∧ 17 * k ≤ 999) (Finset.range 1000)).card = 53 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_17_l3939_393995


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3939_393941

def M : Set Nat := {1, 3, 5, 7}
def N : Set Nat := {2, 5, 8}

theorem intersection_of_M_and_N : M ∩ N = {5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3939_393941


namespace NUMINAMATH_CALUDE_ceiling_floor_product_range_l3939_393991

theorem ceiling_floor_product_range (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 120 → -11 < y ∧ y < -10 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_range_l3939_393991


namespace NUMINAMATH_CALUDE_nonreal_cube_root_of_unity_sum_l3939_393931

theorem nonreal_cube_root_of_unity_sum (ω : ℂ) : 
  ω^3 = 1 ∧ ω ≠ 1 → (1 - ω + ω^2)^6 + (1 + ω - ω^2)^6 = 128 := by sorry

end NUMINAMATH_CALUDE_nonreal_cube_root_of_unity_sum_l3939_393931


namespace NUMINAMATH_CALUDE_bank_coins_l3939_393977

/-- Given a total of 11 coins, including 2 dimes and 2 nickels, prove that the number of quarters is 7. -/
theorem bank_coins (total : ℕ) (dimes : ℕ) (nickels : ℕ) (quarters : ℕ)
  (h_total : total = 11)
  (h_dimes : dimes = 2)
  (h_nickels : nickels = 2)
  (h_sum : total = dimes + nickels + quarters) :
  quarters = 7 := by
  sorry

end NUMINAMATH_CALUDE_bank_coins_l3939_393977


namespace NUMINAMATH_CALUDE_omega_sum_simplification_l3939_393994

theorem omega_sum_simplification (ω : ℂ) (h1 : ω^8 = 1) (h2 : ω ≠ 1) :
  ω^17 + ω^21 + ω^25 + ω^29 + ω^33 + ω^37 + ω^41 + ω^45 + ω^49 + ω^53 + ω^57 + ω^61 + ω^65 = ω :=
by sorry

end NUMINAMATH_CALUDE_omega_sum_simplification_l3939_393994


namespace NUMINAMATH_CALUDE_calculate_expression_l3939_393980

theorem calculate_expression : 
  3 * 995 + 4 * 996 + 5 * 997 + 6 * 998 + 7 * 999 - 4985 * 3 = 9980 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3939_393980


namespace NUMINAMATH_CALUDE_trevors_age_when_brother_is_three_times_older_l3939_393998

theorem trevors_age_when_brother_is_three_times_older (trevor_current_age : ℕ) (brother_current_age : ℕ) :
  trevor_current_age = 11 →
  brother_current_age = 20 →
  ∃ (future_age : ℕ), future_age = 24 ∧ brother_current_age + future_age - trevor_current_age = 3 * trevor_current_age :=
by
  sorry

end NUMINAMATH_CALUDE_trevors_age_when_brother_is_three_times_older_l3939_393998


namespace NUMINAMATH_CALUDE_root_zero_implies_m_six_l3939_393933

theorem root_zero_implies_m_six (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x + m - 6 = 0 ∧ x = 0) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_root_zero_implies_m_six_l3939_393933


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3939_393961

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perp : (2 : ℝ) * b - (b - 3) * a = 0) : 
  ∀ x y : ℝ, 2 * a + 3 * b ≥ 25 := by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3939_393961


namespace NUMINAMATH_CALUDE_trip_cost_equalization_l3939_393973

/-- Given three people who shared expenses on a trip, this theorem calculates
    the amount two of them must pay the third to equalize the costs. -/
theorem trip_cost_equalization
  (A B C : ℝ)  -- Amounts paid by LeRoy, Bernardo, and Carlos respectively
  (h1 : A < B) (h2 : B < C)  -- Ordering of the amounts
  : (2 * C - A - B) / 3 = 
    ((A + B + C) / 3 - A) + ((A + B + C) / 3 - B) :=
by sorry


end NUMINAMATH_CALUDE_trip_cost_equalization_l3939_393973


namespace NUMINAMATH_CALUDE_fiftieth_term_is_ten_l3939_393990

def sequence_term (n : ℕ) : ℕ := 
  Nat.sqrt (2 * n + 1/4 : ℚ).ceil.toNat + 1

theorem fiftieth_term_is_ten : sequence_term 50 = 10 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_is_ten_l3939_393990


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3939_393967

theorem problem_1 (x y : ℝ) : 3 * (x - y)^2 - 6 * (x - y)^2 + 2 * (x - y)^2 = -(x - y)^2 := by
  sorry

theorem problem_2 (a b : ℝ) (h : a^2 - 2*b = 2) : 4*a^2 - 8*b - 9 = -1 := by
  sorry

theorem problem_3 (a b c d : ℝ) (h1 : a - 2*b = 4) (h2 : b - c = -5) (h3 : 3*c + d = 10) :
  (a + 3*c) - (2*b + c) + (b + d) = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3939_393967


namespace NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l3939_393951

-- Define the hyperbola
def is_hyperbola (x y m : ℝ) : Prop := x^2 - y^2 / m^2 = 1

-- Define the condition that m is positive
def m_positive (m : ℝ) : Prop := m > 0

-- Define the distance from focus to asymptote
def focus_asymptote_distance (m : ℝ) : ℝ := m

-- Theorem statement
theorem hyperbola_focus_asymptote_distance (m : ℝ) :
  m_positive m →
  (∃ x y, is_hyperbola x y m) →
  focus_asymptote_distance m = 4 →
  m = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l3939_393951


namespace NUMINAMATH_CALUDE_determinant_transformation_l3939_393932

theorem determinant_transformation (x y z w : ℝ) :
  (x * w - y * z = 3) →
  (x * (9 * z + 4 * w) - z * (9 * x + 4 * y) = 12) := by
  sorry

end NUMINAMATH_CALUDE_determinant_transformation_l3939_393932


namespace NUMINAMATH_CALUDE_cube_sum_ge_mixed_product_l3939_393958

theorem cube_sum_ge_mixed_product {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 ≥ a^2 * b + a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_ge_mixed_product_l3939_393958


namespace NUMINAMATH_CALUDE_sets_problem_l3939_393952

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 9*x + 18 ≥ 0}
def B : Set ℝ := {x | -2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem statement
theorem sets_problem :
  (A ∪ B = Set.univ) ∧
  ((Set.univ \ A) ∩ B = {x | 3 < x ∧ x < 6}) ∧
  (∀ a : ℝ, C a ⊆ B → -2 ≤ a ∧ a ≤ 8) := by
  sorry


end NUMINAMATH_CALUDE_sets_problem_l3939_393952


namespace NUMINAMATH_CALUDE_problem_solution_l3939_393903

theorem problem_solution (x y : ℚ) (hx : x = 3) (hy : y = 5) : 
  (x^5 + 2*y^2 - 15) / 7 = 39 + 5/7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3939_393903


namespace NUMINAMATH_CALUDE_class_gpa_theorem_l3939_393979

/-- The grade point average (GPA) of a class, given the GPAs of two subgroups -/
def classGPA (fraction1 : ℚ) (gpa1 : ℚ) (fraction2 : ℚ) (gpa2 : ℚ) : ℚ :=
  fraction1 * gpa1 + fraction2 * gpa2

/-- Theorem: The GPA of a class where one-third has GPA 60 and two-thirds has GPA 66 is 64 -/
theorem class_gpa_theorem :
  classGPA (1/3) 60 (2/3) 66 = 64 := by
  sorry

end NUMINAMATH_CALUDE_class_gpa_theorem_l3939_393979


namespace NUMINAMATH_CALUDE_shooter_probability_l3939_393901

theorem shooter_probability (p10 p9 p8 : ℝ) 
  (h1 : p10 = 0.2) 
  (h2 : p9 = 0.3) 
  (h3 : p8 = 0.1) : 
  1 - (p10 + p9) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_shooter_probability_l3939_393901


namespace NUMINAMATH_CALUDE_f_2a_equals_7_l3939_393915

def f (x : ℝ) : ℝ := 2 * x + 2 - x

theorem f_2a_equals_7 (a : ℝ) (h : f a = 3) : f (2 * a) = 7 := by
  sorry

end NUMINAMATH_CALUDE_f_2a_equals_7_l3939_393915


namespace NUMINAMATH_CALUDE_total_ways_is_81_l3939_393945

/-- The number of base options available to each student -/
def num_bases : ℕ := 3

/-- The number of students choosing bases -/
def num_students : ℕ := 4

/-- The total number of ways for students to choose bases -/
def total_ways : ℕ := num_bases ^ num_students

/-- Theorem stating that the total number of ways is 81 -/
theorem total_ways_is_81 : total_ways = 81 := by
  sorry

end NUMINAMATH_CALUDE_total_ways_is_81_l3939_393945


namespace NUMINAMATH_CALUDE_triangle_area_l3939_393938

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  a = Real.sqrt 2 →
  A = π / 4 →
  B = π / 3 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  (1 / 2) * a * b * Real.sin C = (3 + Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3939_393938


namespace NUMINAMATH_CALUDE_temperature_conversion_l3939_393919

theorem temperature_conversion (t k r : ℝ) 
  (eq1 : t = 5/9 * (k - 32))
  (eq2 : r = 3*t)
  (eq3 : r = 150) : 
  k = 122 := by
sorry

end NUMINAMATH_CALUDE_temperature_conversion_l3939_393919


namespace NUMINAMATH_CALUDE_cubic_factorization_l3939_393997

theorem cubic_factorization (x : ℝ) : x^3 - 4*x^2 + 4*x = x*(x-2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3939_393997


namespace NUMINAMATH_CALUDE_fourth_vertex_of_complex_rectangle_l3939_393920

/-- A rectangle in the complex plane --/
structure ComplexRectangle where
  a : ℂ
  b : ℂ
  c : ℂ
  d : ℂ
  is_rectangle : (b - a).arg.cos * (c - b).arg.cos + (b - a).arg.sin * (c - b).arg.sin = 0

/-- The theorem stating that given three vertices of a rectangle in the complex plane,
    we can determine the fourth vertex --/
theorem fourth_vertex_of_complex_rectangle (r : ComplexRectangle)
  (h1 : r.a = 3 + 2*I)
  (h2 : r.b = 1 + I)
  (h3 : r.c = -1 - 2*I) :
  r.d = -3 - 3*I := by
  sorry

#check fourth_vertex_of_complex_rectangle

end NUMINAMATH_CALUDE_fourth_vertex_of_complex_rectangle_l3939_393920


namespace NUMINAMATH_CALUDE_parabola_directrix_l3939_393982

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the directrix
def directrix (y : ℝ) : Prop := y = -2

-- Theorem statement
theorem parabola_directrix : 
  ∀ (x y : ℝ), parabola x y → directrix y :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3939_393982


namespace NUMINAMATH_CALUDE_counterexample_to_goldbach_like_conjecture_l3939_393996

theorem counterexample_to_goldbach_like_conjecture :
  (∃ n : ℕ, n > 5 ∧ Odd n ∧ ¬∃ (a b c : ℕ), Odd a ∧ Odd b ∧ Odd c ∧ n = a + b + c) ∨
  (∃ n : ℕ, n > 5 ∧ Even n ∧ ¬∃ (a b c : ℕ), Odd a ∧ Odd b ∧ Odd c ∧ n = a + b + c) →
  ¬(∀ n : ℕ, n > 5 → ∃ (a b c : ℕ), Odd a ∧ Odd b ∧ Odd c ∧ n = a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_counterexample_to_goldbach_like_conjecture_l3939_393996


namespace NUMINAMATH_CALUDE_matthews_cracker_distribution_l3939_393981

theorem matthews_cracker_distribution (total_crackers : ℕ) (crackers_per_person : ℕ) (num_friends : ℕ) : 
  total_crackers = 36 → 
  crackers_per_person = 2 → 
  total_crackers = num_friends * crackers_per_person → 
  num_friends = 18 := by
sorry

end NUMINAMATH_CALUDE_matthews_cracker_distribution_l3939_393981


namespace NUMINAMATH_CALUDE_smallest_divisor_power_l3939_393968

def polynomial (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_divisor_power : ∃! k : ℕ+, 
  (∀ z : ℂ, polynomial z ∣ (z^k.val - 1)) ∧ 
  (∀ m : ℕ+, m < k → ∃ z : ℂ, ¬(polynomial z ∣ (z^m.val - 1))) ∧
  k = 84 := by sorry

end NUMINAMATH_CALUDE_smallest_divisor_power_l3939_393968


namespace NUMINAMATH_CALUDE_boys_from_clay_middle_school_l3939_393904

theorem boys_from_clay_middle_school 
  (total_students : ℕ)
  (total_boys : ℕ)
  (total_girls : ℕ)
  (jonas_students : ℕ)
  (clay_students : ℕ)
  (jonas_girls : ℕ)
  (h1 : total_students = 100)
  (h2 : total_boys = 52)
  (h3 : total_girls = 48)
  (h4 : jonas_students = 40)
  (h5 : clay_students = 60)
  (h6 : jonas_girls = 20)
  (h7 : total_students = total_boys + total_girls)
  (h8 : total_students = jonas_students + clay_students)
  : ∃ (clay_boys : ℕ), clay_boys = 32 ∧ 
    clay_boys + (total_boys - clay_boys) = total_boys ∧
    clay_boys + (clay_students - clay_boys) = clay_students :=
by sorry

end NUMINAMATH_CALUDE_boys_from_clay_middle_school_l3939_393904


namespace NUMINAMATH_CALUDE_max_value_expression_l3939_393935

theorem max_value_expression (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (1 / ((2 - x) * (2 - y) * (2 - z)) + 1 / ((2 + x) * (2 + y) * (2 + z))) ≤ 12 / 27 ∧
  (1 / ((2 - 1) * (2 - 1) * (2 - 1)) + 1 / ((2 + 1) * (2 + 1) * (2 + 1))) = 12 / 27 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3939_393935


namespace NUMINAMATH_CALUDE_unique_parallel_line_l3939_393988

-- Define the concept of a plane
variable (Plane : Type)

-- Define the concept of a point
variable (Point : Type)

-- Define the concept of a line
variable (Line : Type)

-- Define the relation of a point lying on a plane
variable (lies_on : Point → Plane → Prop)

-- Define the relation of two planes intersecting
variable (intersect : Plane → Plane → Prop)

-- Define the relation of a line being parallel to a plane
variable (parallel_to_plane : Line → Plane → Prop)

-- Define the relation of a line passing through a point
variable (passes_through : Line → Point → Prop)

-- Theorem statement
theorem unique_parallel_line 
  (α β : Plane) (A : Point) 
  (h_intersect : intersect α β)
  (h_not_on_α : ¬ lies_on A α)
  (h_not_on_β : ¬ lies_on A β) :
  ∃! l : Line, passes_through l A ∧ parallel_to_plane l α ∧ parallel_to_plane l β :=
sorry

end NUMINAMATH_CALUDE_unique_parallel_line_l3939_393988


namespace NUMINAMATH_CALUDE_k_value_proof_l3939_393900

theorem k_value_proof (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) →
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_k_value_proof_l3939_393900


namespace NUMINAMATH_CALUDE_building_height_ratio_l3939_393934

/-- Given three buildings with specific height relationships, prove the ratio of the second to the first building's height. -/
theorem building_height_ratio :
  ∀ (h₁ h₂ h₃ : ℝ),
  h₁ = 600 →
  h₃ = 3 * (h₁ + h₂) →
  h₁ + h₂ + h₃ = 7200 →
  h₂ / h₁ = 2 := by
sorry

end NUMINAMATH_CALUDE_building_height_ratio_l3939_393934


namespace NUMINAMATH_CALUDE_smallest_a_divisibility_l3939_393902

theorem smallest_a_divisibility : 
  ∃ (n : ℕ), 
    n % 2 = 1 ∧ 
    (55^n + 2000 * 32^n) % 2001 = 0 ∧ 
    ∀ (a : ℕ), a > 0 → a < 2000 → 
      ∀ (m : ℕ), m % 2 = 1 → (55^m + a * 32^m) % 2001 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_divisibility_l3939_393902


namespace NUMINAMATH_CALUDE_real_part_of_complex_number_l3939_393905

theorem real_part_of_complex_number (z : ℂ) (h : z - Complex.abs z = -8 + 12*I) : 
  Complex.re z = 5 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_number_l3939_393905


namespace NUMINAMATH_CALUDE_inequality_generalization_l3939_393954

theorem inequality_generalization (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 0) :
  x + n^n / x^n ≥ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_generalization_l3939_393954


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3939_393948

theorem fraction_multiplication (x : ℚ) : (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5020 = 753 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3939_393948


namespace NUMINAMATH_CALUDE_book_reading_time_l3939_393960

theorem book_reading_time (chapters : ℕ) (total_pages : ℕ) (pages_per_day : ℕ) : 
  chapters = 41 → total_pages = 450 → pages_per_day = 15 → 
  (total_pages / pages_per_day : ℕ) = 30 := by
sorry

end NUMINAMATH_CALUDE_book_reading_time_l3939_393960


namespace NUMINAMATH_CALUDE_odd_gon_symmetry_axis_through_vertex_l3939_393912

/-- A (2k+1)-gon is a polygon with 2k+1 vertices, where k is a positive integer. -/
structure OddGon where
  k : ℕ+
  vertices : Fin (2 * k + 1) → ℝ × ℝ

/-- An axis of symmetry for a polygon -/
structure SymmetryAxis (P : OddGon) where
  line : ℝ × ℝ → Prop

/-- A vertex lies on a line -/
def vertex_on_line (P : OddGon) (axis : SymmetryAxis P) (v : Fin (2 * P.k + 1)) : Prop :=
  axis.line (P.vertices v)

/-- The theorem stating that the axis of symmetry of a (2k+1)-gon passes through one of its vertices -/
theorem odd_gon_symmetry_axis_through_vertex (P : OddGon) (axis : SymmetryAxis P) :
  ∃ v : Fin (2 * P.k + 1), vertex_on_line P axis v := by
  sorry

end NUMINAMATH_CALUDE_odd_gon_symmetry_axis_through_vertex_l3939_393912


namespace NUMINAMATH_CALUDE_radio_survey_l3939_393946

theorem radio_survey (total_listeners : ℕ) (total_non_listeners : ℕ) 
  (male_listeners : ℕ) (female_non_listeners : ℕ) 
  (h1 : total_listeners = 200)
  (h2 : total_non_listeners = 180)
  (h3 : male_listeners = 75)
  (h4 : female_non_listeners = 120) :
  total_listeners + total_non_listeners - male_listeners - female_non_listeners = 185 :=
by sorry

end NUMINAMATH_CALUDE_radio_survey_l3939_393946


namespace NUMINAMATH_CALUDE_tan_2alpha_values_l3939_393971

theorem tan_2alpha_values (α : ℝ) (h : 2 * Real.sin (2 * α) = 1 + Real.cos (2 * α)) :
  Real.tan (2 * α) = 4/3 ∨ Real.tan (2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_2alpha_values_l3939_393971


namespace NUMINAMATH_CALUDE_last_two_digits_of_expression_l3939_393983

theorem last_two_digits_of_expression : 
  (1941^3846 + 1961^4181 - 1981^4556 * 2141^4917) % 100 = 81 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_expression_l3939_393983


namespace NUMINAMATH_CALUDE_square_difference_equality_l3939_393936

theorem square_difference_equality : 535^2 - 465^2 = 70000 := by sorry

end NUMINAMATH_CALUDE_square_difference_equality_l3939_393936


namespace NUMINAMATH_CALUDE_smallest_c_for_three_in_range_l3939_393909

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

-- Theorem statement
theorem smallest_c_for_three_in_range :
  ∀ c : ℝ, (∃ x : ℝ, f c x = 3) ↔ c ≥ 12 := by sorry

end NUMINAMATH_CALUDE_smallest_c_for_three_in_range_l3939_393909


namespace NUMINAMATH_CALUDE_sin_cos_equation_l3939_393943

theorem sin_cos_equation (x : Real) (p q : Nat) 
  (h1 : (1 + Real.sin x) * (1 + Real.cos x) = 9/4)
  (h2 : (1 - Real.sin x) * (1 - Real.cos x) = p - Real.sqrt q)
  (h3 : 0 < p) (h4 : 0 < q) : p + q = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_equation_l3939_393943


namespace NUMINAMATH_CALUDE_routes_on_3x2_grid_l3939_393970

/-- The number of routes on a grid with more right moves than down moves -/
def num_routes (right down : ℕ) : ℕ :=
  Nat.choose (right + down) down

/-- The grid dimensions -/
def grid_width : ℕ := 3
def grid_height : ℕ := 2

theorem routes_on_3x2_grid :
  num_routes grid_width grid_height = 21 :=
sorry

end NUMINAMATH_CALUDE_routes_on_3x2_grid_l3939_393970


namespace NUMINAMATH_CALUDE_incorrect_inequality_transformation_l3939_393989

theorem incorrect_inequality_transformation (a b : ℝ) (h : a > b) :
  ¬(1 - a > 1 - b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_transformation_l3939_393989


namespace NUMINAMATH_CALUDE_partner_contribution_correct_l3939_393913

/-- Calculates the partner's contribution given the investment details and profit ratio -/
def calculate_partner_contribution (a_investment : ℚ) (a_months : ℚ) (b_months : ℚ) (a_ratio : ℚ) (b_ratio : ℚ) : ℚ :=
  (a_investment * a_months * b_ratio) / (a_ratio * b_months)

theorem partner_contribution_correct :
  let a_investment : ℚ := 3500
  let a_months : ℚ := 12
  let b_months : ℚ := 3
  let a_ratio : ℚ := 2
  let b_ratio : ℚ := 3
  calculate_partner_contribution a_investment a_months b_months a_ratio b_ratio = 21000 := by
  sorry

end NUMINAMATH_CALUDE_partner_contribution_correct_l3939_393913
