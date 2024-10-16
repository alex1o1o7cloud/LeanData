import Mathlib

namespace NUMINAMATH_CALUDE_jenny_ate_65_chocolates_l1014_101445

/-- The number of chocolates Mike ate -/
def mike_chocolates : ℕ := 20

/-- The number of chocolates John ate -/
def john_chocolates : ℕ := mike_chocolates / 2

/-- The combined number of chocolates Mike and John ate -/
def combined_chocolates : ℕ := mike_chocolates + john_chocolates

/-- The number of chocolates Jenny ate -/
def jenny_chocolates : ℕ := 2 * combined_chocolates + 5

/-- Theorem stating that Jenny ate 65 chocolates -/
theorem jenny_ate_65_chocolates : jenny_chocolates = 65 := by
  sorry

end NUMINAMATH_CALUDE_jenny_ate_65_chocolates_l1014_101445


namespace NUMINAMATH_CALUDE_not_prime_sum_product_l1014_101415

theorem not_prime_sum_product (a b c d : ℕ) 
  (h_pos : 0 < d ∧ 0 < c ∧ 0 < b ∧ 0 < a)
  (h_order : d < c ∧ c < b ∧ b < a)
  (h_eq : a * c + b * d = (b + d + a - c) * (b + d - a + c)) :
  ¬ Nat.Prime (a * b + c * d) :=
by sorry

end NUMINAMATH_CALUDE_not_prime_sum_product_l1014_101415


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1014_101482

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 6}
def B : Set Nat := {1, 2}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1014_101482


namespace NUMINAMATH_CALUDE_power_of_two_pairs_l1014_101439

theorem power_of_two_pairs (m n : ℕ+) :
  (∃ k : ℕ, m + n = 2^(k+1)) ∧ 
  (∃ l : ℕ, m * n + 1 = 2^l) →
  ∃ a : ℕ, (m = 2^(a+1) - 1 ∧ n = 1) ∨ 
           (m = 2^a + 1 ∧ n = 2^a - 1) ∨
           (m = 2^a - 1 ∧ n = 2^a + 1) ∨
           (m = 1 ∧ n = 2^(a+1) - 1) :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_pairs_l1014_101439


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1014_101414

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2^x - 1 > 0) ↔ (∃ x : ℝ, x^2 + 2^x - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1014_101414


namespace NUMINAMATH_CALUDE_base_conversion_sum_l1014_101498

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b ^ i) 0

/-- Represents the digit C in base 13 -/
def C : Nat := 12

theorem base_conversion_sum : 
  let base_5_num := to_base_10 [2, 4, 3] 5
  let base_13_num := to_base_10 [9, C, 2] 13
  base_5_num + base_13_num = 600 := by sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l1014_101498


namespace NUMINAMATH_CALUDE_exists_a_greater_than_bound_l1014_101419

def a : ℕ → ℚ
  | 0 => 1
  | 1 => 1/3
  | (n+2) => (2 * a (n+1)) / 3 - a n

theorem exists_a_greater_than_bound : ∃ n : ℕ, a n > 999/1000 := by
  sorry

end NUMINAMATH_CALUDE_exists_a_greater_than_bound_l1014_101419


namespace NUMINAMATH_CALUDE_subjectB_least_hours_subjectB_total_hours_l1014_101435

/-- Represents the study hours for each subject over a 15-week semester. -/
structure StudyHours where
  subjectA : ℕ
  subjectB : ℕ
  subjectC : ℕ
  subjectD : ℕ

/-- Calculates the total study hours for Subject A over 15 weeks. -/
def calculateSubjectA : ℕ := 3 * 5 * 15

/-- Calculates the total study hours for Subject B over 15 weeks. -/
def calculateSubjectB : ℕ := 2 * 3 * 15

/-- Calculates the total study hours for Subject C over 15 weeks. -/
def calculateSubjectC : ℕ := (4 + 3 + 3) * 15

/-- Calculates the total study hours for Subject D over 15 weeks. -/
def calculateSubjectD : ℕ := (1 * 5 + 5) * 15

/-- Creates a StudyHours structure with the calculated hours for each subject. -/
def parisStudyHours : StudyHours :=
  { subjectA := calculateSubjectA
  , subjectB := calculateSubjectB
  , subjectC := calculateSubjectC
  , subjectD := calculateSubjectD }

/-- Theorem: Subject B has the least study hours among all subjects. -/
theorem subjectB_least_hours (h : StudyHours) (h_eq : h = parisStudyHours) :
  h.subjectB ≤ h.subjectA ∧ h.subjectB ≤ h.subjectC ∧ h.subjectB ≤ h.subjectD :=
by sorry

/-- Theorem: The total study hours for Subject B is 90. -/
theorem subjectB_total_hours : parisStudyHours.subjectB = 90 :=
by sorry

end NUMINAMATH_CALUDE_subjectB_least_hours_subjectB_total_hours_l1014_101435


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1014_101441

def M : Set Int := {-1, 1}
def N : Set Int := {-1, 0, 2}

theorem intersection_of_M_and_N : M ∩ N = {-1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1014_101441


namespace NUMINAMATH_CALUDE_tree_cutting_theorem_l1014_101421

/-- The number of trees James cuts per day -/
def james_trees_per_day : ℕ := 20

/-- The number of days James works alone -/
def james_solo_days : ℕ := 2

/-- The number of days the brothers help -/
def brother_help_days : ℕ := 3

/-- The number of brothers helping -/
def num_brothers : ℕ := 2

/-- The percentage reduction in trees cut by brothers compared to James -/
def brother_reduction_percent : ℚ := 20 / 100

/-- The total number of trees cut down -/
def total_trees_cut : ℕ := 136

theorem tree_cutting_theorem :
  james_trees_per_day * james_solo_days + 
  (james_trees_per_day * (1 - brother_reduction_percent) * num_brothers * brother_help_days).floor = 
  total_trees_cut :=
sorry

end NUMINAMATH_CALUDE_tree_cutting_theorem_l1014_101421


namespace NUMINAMATH_CALUDE_rhombus_area_l1014_101485

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 13) (h2 : d2 = 20) :
  (d1 * d2) / 2 = 130 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l1014_101485


namespace NUMINAMATH_CALUDE_mortezas_wish_impossible_l1014_101443

theorem mortezas_wish_impossible :
  ¬ ∃ (x₁ x₂ x₃ x₄ x₅ x₆ S P : ℝ),
    (x₁ ≠ x₂) ∧ (x₁ ≠ x₃) ∧ (x₁ ≠ x₄) ∧ (x₁ ≠ x₅) ∧ (x₁ ≠ x₆) ∧
    (x₂ ≠ x₃) ∧ (x₂ ≠ x₄) ∧ (x₂ ≠ x₅) ∧ (x₂ ≠ x₆) ∧
    (x₃ ≠ x₄) ∧ (x₃ ≠ x₅) ∧ (x₃ ≠ x₆) ∧
    (x₄ ≠ x₅) ∧ (x₄ ≠ x₆) ∧
    (x₅ ≠ x₆) ∧
    ((x₁ + x₂ + x₃ = S) ∨ (x₁ * x₂ * x₃ = P)) ∧
    ((x₂ + x₃ + x₄ = S) ∨ (x₂ * x₃ * x₄ = P)) ∧
    ((x₃ + x₄ + x₅ = S) ∨ (x₃ * x₄ * x₅ = P)) ∧
    ((x₄ + x₅ + x₆ = S) ∨ (x₄ * x₅ * x₆ = P)) ∧
    ((x₅ + x₆ + x₁ = S) ∨ (x₅ * x₆ * x₁ = P)) ∧
    ((x₆ + x₁ + x₂ = S) ∨ (x₆ * x₁ * x₂ = P)) :=
by sorry

end NUMINAMATH_CALUDE_mortezas_wish_impossible_l1014_101443


namespace NUMINAMATH_CALUDE_triangle_sides_from_heights_l1014_101481

theorem triangle_sides_from_heights (h_a h_b h_c A : ℝ) (h_positive : h_a > 0 ∧ h_b > 0 ∧ h_c > 0) (h_area : A > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    A = (1/2) * a * h_a ∧
    A = (1/2) * b * h_b ∧
    A = (1/2) * c * h_c :=
sorry


end NUMINAMATH_CALUDE_triangle_sides_from_heights_l1014_101481


namespace NUMINAMATH_CALUDE_d₂₀₁₇_equidistant_points_l1014_101487

/-- The set S of integer coordinates (x, y) where 0 ≤ x, y ≤ 2016 -/
def S : Set (ℤ × ℤ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 2016 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2016}

/-- The distance function d₂₀₁₇ -/
def d₂₀₁₇ (a b : ℤ × ℤ) : ℤ :=
  ((a.1 - b.1)^2 + (a.2 - b.2)^2) % 2017

/-- The theorem to be proved -/
theorem d₂₀₁₇_equidistant_points :
  ∃ O ∈ S,
  d₂₀₁₇ O (5, 5) = d₂₀₁₇ O (2, 6) ∧
  d₂₀₁₇ O (5, 5) = d₂₀₁₇ O (7, 11) →
  d₂₀₁₇ O (5, 5) = 1021 := by
  sorry

end NUMINAMATH_CALUDE_d₂₀₁₇_equidistant_points_l1014_101487


namespace NUMINAMATH_CALUDE_correct_number_of_choices_l1014_101471

/-- Represents a team in the club -/
inductive Team
| A
| B

/-- Represents the gender of a club member -/
inductive Gender
| Boy
| Girl

/-- Represents the composition of a team -/
structure TeamComposition :=
  (boys : ℕ)
  (girls : ℕ)

/-- The total number of members in the club -/
def totalMembers : ℕ := 24

/-- The number of boys in the club -/
def totalBoys : ℕ := 14

/-- The number of girls in the club -/
def totalGirls : ℕ := 10

/-- The composition of Team A -/
def teamA : TeamComposition := ⟨8, 6⟩

/-- The composition of Team B -/
def teamB : TeamComposition := ⟨6, 4⟩

/-- Returns the number of ways to choose a president and vice-president -/
def chooseLeaders : ℕ := sorry

/-- Theorem stating that the number of ways to choose a president and vice-president
    of different genders and from different teams is 136 -/
theorem correct_number_of_choices :
  chooseLeaders = 136 := by sorry

end NUMINAMATH_CALUDE_correct_number_of_choices_l1014_101471


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l1014_101476

/-- Given a quadratic equation x^2 - 6x + 5 = 0, when rewritten in the form (x + b)^2 = c
    where b and c are integers, prove that b + c = 11 -/
theorem quadratic_complete_square (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x + b)^2 = c) → b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l1014_101476


namespace NUMINAMATH_CALUDE_triangle_side_length_l1014_101446

theorem triangle_side_length (a c : ℝ) (B : ℝ) (h1 : a = 2) (h2 : c = 4) (h3 : B = π / 3) :
  let b := Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B)
  b = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1014_101446


namespace NUMINAMATH_CALUDE_correct_calculation_l1014_101428

theorem correct_calculation (x : ℝ) : 5 * x + 4 = 104 → (x + 5) / 4 = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1014_101428


namespace NUMINAMATH_CALUDE_number_problem_l1014_101477

theorem number_problem : ∃! x : ℝ, (x / 3) + 12 = 20 ∧ x = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1014_101477


namespace NUMINAMATH_CALUDE_ball_hits_middle_pocket_l1014_101470

/-- Represents a rectangular billiard table -/
structure BilliardTable where
  p : ℕ
  q : ℕ
  p_odd : Odd p
  q_odd : Odd q

/-- Represents the trajectory of a ball on the billiard table -/
def ball_trajectory (table : BilliardTable) : ℕ → ℕ → Prop :=
  fun x y => y = x

/-- Represents a middle pocket on the long side of the table -/
def middle_pocket (table : BilliardTable) : ℕ → ℕ → Prop :=
  fun x y => (x = table.p / 2 ∧ (y = 0 ∨ y = 2 * table.q)) ∨ 
             (y = table.q ∧ (x = 0 ∨ x = table.p))

/-- The main theorem stating that the ball will hit a middle pocket -/
theorem ball_hits_middle_pocket (table : BilliardTable) :
  ∃ (x y : ℕ), ball_trajectory table x y ∧ middle_pocket table x y :=
sorry

end NUMINAMATH_CALUDE_ball_hits_middle_pocket_l1014_101470


namespace NUMINAMATH_CALUDE_player_a_winning_strategy_l1014_101447

/-- The game board represented as a 3x3 matrix -/
def GameBoard : Matrix (Fin 3) (Fin 3) ℕ :=
  !![7, 8, 9;
     4, 5, 6;
     1, 2, 3]

/-- Checks if two numbers are in the same row or column on the game board -/
def inSameRowOrCol (a b : ℕ) : Prop :=
  ∃ i j k : Fin 3, (GameBoard i j = a ∧ GameBoard i k = b) ∨
                   (GameBoard j i = a ∧ GameBoard k i = b)

/-- Represents a valid move in the game -/
structure Move where
  number : ℕ
  valid : number ≥ 1 ∧ number ≤ 9

/-- Represents the game state -/
structure GameState where
  moves : List Move
  total : ℕ
  lastMove : Move

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  move.number ≠ state.lastMove.number ∧
  inSameRowOrCol move.number state.lastMove.number ∧
  state.total + move.number ≤ 30

/-- Represents a winning strategy for Player A -/
def WinningStrategy :=
  ∃ (firstMove : Move),
    ∀ (b1 : Move),
      ∃ (a2 : Move),
        ∀ (b2 : Move),
          ∃ (a3 : Move),
            (isValidMove ⟨[firstMove], firstMove.number, firstMove⟩ b1 →
             isValidMove ⟨[b1, firstMove], firstMove.number + b1.number, b1⟩ a2 →
             isValidMove ⟨[a2, b1, firstMove], firstMove.number + b1.number + a2.number, a2⟩ b2 →
             isValidMove ⟨[b2, a2, b1, firstMove], firstMove.number + b1.number + a2.number + b2.number, b2⟩ a3) ∧
            (firstMove.number + b1.number + a2.number + b2.number + a3.number = 30)

theorem player_a_winning_strategy : WinningStrategy := sorry

end NUMINAMATH_CALUDE_player_a_winning_strategy_l1014_101447


namespace NUMINAMATH_CALUDE_stock_price_decrease_l1014_101464

/-- The percentage decrease required for a stock to return to its original price after a 40% increase -/
theorem stock_price_decrease (initial_price : ℝ) (h : initial_price > 0) :
  let increased_price := 1.4 * initial_price
  let decrease_percent := (increased_price - initial_price) / increased_price
  decrease_percent = 0.2857142857142857 := by
sorry

end NUMINAMATH_CALUDE_stock_price_decrease_l1014_101464


namespace NUMINAMATH_CALUDE_crosswalk_parallelogram_l1014_101431

/-- A parallelogram with given dimensions -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ
  height1 : ℝ
  height2 : ℝ

/-- The theorem about the parallelogram representing the crosswalk -/
theorem crosswalk_parallelogram (p : Parallelogram) 
  (h1 : p.side1 = 18)
  (h2 : p.height1 = 60)
  (h3 : p.side2 = 60) :
  p.height2 = 18 := by
  sorry

#check crosswalk_parallelogram

end NUMINAMATH_CALUDE_crosswalk_parallelogram_l1014_101431


namespace NUMINAMATH_CALUDE_seeds_per_flower_bed_l1014_101466

theorem seeds_per_flower_bed 
  (total_seeds : ℕ) 
  (num_flower_beds : ℕ) 
  (h1 : total_seeds = 54) 
  (h2 : num_flower_beds = 9) 
  (h3 : num_flower_beds ≠ 0) : 
  total_seeds / num_flower_beds = 6 := by
sorry

end NUMINAMATH_CALUDE_seeds_per_flower_bed_l1014_101466


namespace NUMINAMATH_CALUDE_smallest_x_equals_f_2001_l1014_101408

def f (x : ℝ) : ℝ := sorry

axiom f_triple (x : ℝ) (h : 0 < x) : f (3 * x) = 3 * f x

axiom f_definition (x : ℝ) (h : 1 ≤ x ∧ x ≤ 3) : f x = 1 - |x - 2|

theorem smallest_x_equals_f_2001 :
  ∃ (x : ℝ), x > 0 ∧ f x = f 2001 ∧ ∀ (y : ℝ), y > 0 ∧ f y = f 2001 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_x_equals_f_2001_l1014_101408


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1014_101494

theorem product_of_three_numbers (a b c : ℝ) : 
  a + b + c = 300 ∧ 
  5 * a = c - 14 ∧ 
  5 * a = b + 14 → 
  a * b * c = 664500 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1014_101494


namespace NUMINAMATH_CALUDE_distance_between_points_l1014_101438

/-- The distance between points (1, 16) and (9, 3) is √233 -/
theorem distance_between_points : Real.sqrt 233 = Real.sqrt ((9 - 1)^2 + (3 - 16)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1014_101438


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1014_101450

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_property : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Main theorem about the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : seq.S 1023 - seq.S 1000 = 23) : 
  seq.a 1012 = 1 ∧ seq.S 2023 = 2023 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1014_101450


namespace NUMINAMATH_CALUDE_arithmetic_operation_proof_l1014_101426

theorem arithmetic_operation_proof : 65 + 5 * 12 / (180 / 3) = 66 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operation_proof_l1014_101426


namespace NUMINAMATH_CALUDE_basketball_free_throws_l1014_101401

theorem basketball_free_throws (two_point_shots three_point_shots free_throws : ℕ) : 
  (2 * two_point_shots = 3 * three_point_shots) →
  (free_throws = two_point_shots + 1) →
  (2 * two_point_shots + 3 * three_point_shots + free_throws = 61) →
  free_throws = 13 := by
  sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l1014_101401


namespace NUMINAMATH_CALUDE_mary_saw_256_snakes_l1014_101461

/-- The number of breeding balls -/
def num_breeding_balls : Nat := 7

/-- The number of snakes in each breeding ball -/
def snakes_in_balls : List Nat := [15, 20, 25, 30, 35, 40, 45]

/-- The number of extra pairs of snakes -/
def extra_pairs : Nat := 23

/-- The total number of snakes Mary saw -/
def total_snakes : Nat := (List.sum snakes_in_balls) + (2 * extra_pairs)

theorem mary_saw_256_snakes :
  total_snakes = 256 := by sorry

end NUMINAMATH_CALUDE_mary_saw_256_snakes_l1014_101461


namespace NUMINAMATH_CALUDE_sum_of_ages_after_20_years_l1014_101417

/-- Given the ages of Ann and her siblings and cousin, calculate the sum of their ages after 20 years -/
theorem sum_of_ages_after_20_years 
  (ann_age : ℕ)
  (tom_age : ℕ)
  (bill_age : ℕ)
  (cathy_age : ℕ)
  (emily_age : ℕ)
  (h1 : ann_age = 6)
  (h2 : tom_age = 2 * ann_age)
  (h3 : bill_age = tom_age - 3)
  (h4 : cathy_age = 2 * tom_age)
  (h5 : emily_age = cathy_age / 2)
  : ann_age + tom_age + bill_age + cathy_age + emily_age + 20 * 5 = 163 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_after_20_years_l1014_101417


namespace NUMINAMATH_CALUDE_factory_days_worked_l1014_101499

/-- A factory produces refrigerators and coolers. -/
structure Factory where
  refrigerators_per_hour : ℕ
  coolers_per_hour : ℕ
  hours_per_day : ℕ
  total_products : ℕ

/-- Calculate the number of days worked by the factory. -/
def days_worked (f : Factory) : ℕ :=
  f.total_products / (f.hours_per_day * (f.refrigerators_per_hour + f.coolers_per_hour))

/-- Theorem stating the number of days worked by the factory. -/
theorem factory_days_worked :
  let f : Factory := {
    refrigerators_per_hour := 90,
    coolers_per_hour := 90 + 70,
    hours_per_day := 9,
    total_products := 11250
  }
  days_worked f = 5 := by sorry

end NUMINAMATH_CALUDE_factory_days_worked_l1014_101499


namespace NUMINAMATH_CALUDE_det_cyclic_matrix_cubic_roots_l1014_101442

/-- Given a cubic equation x³ - 2x² + px + q = 0 with roots a, b, and c,
    the determinant of the matrix [[a,b,c],[b,c,a],[c,a,b]] is -p - 8 -/
theorem det_cyclic_matrix_cubic_roots (p q : ℝ) (a b c : ℝ) 
    (h₁ : a^3 - 2*a^2 + p*a + q = 0)
    (h₂ : b^3 - 2*b^2 + p*b + q = 0)
    (h₃ : c^3 - 2*c^2 + p*c + q = 0) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![a,b,c; b,c,a; c,a,b]
  Matrix.det M = -p - 8 := by
sorry

end NUMINAMATH_CALUDE_det_cyclic_matrix_cubic_roots_l1014_101442


namespace NUMINAMATH_CALUDE_range_of_a_l1014_101455

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + 4 > 0) → -2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1014_101455


namespace NUMINAMATH_CALUDE_james_age_when_thomas_grows_l1014_101495

/-- Given the ages and relationships of Thomas, Shay, and James, prove James' age when Thomas reaches his current age. -/
theorem james_age_when_thomas_grows (thomas_age : ℕ) (shay_thomas_diff : ℕ) (james_shay_diff : ℕ) : 
  thomas_age = 6 →
  shay_thomas_diff = 13 →
  james_shay_diff = 5 →
  thomas_age + shay_thomas_diff + james_shay_diff + shay_thomas_diff = 37 :=
by sorry

end NUMINAMATH_CALUDE_james_age_when_thomas_grows_l1014_101495


namespace NUMINAMATH_CALUDE_basketball_game_theorem_l1014_101458

/-- Represents the scores of a team in a four-quarter basketball game -/
structure GameScores where
  q1 : ℕ
  q2 : ℕ
  q3 : ℕ
  q4 : ℕ

/-- Checks if the given scores form an arithmetic sequence -/
def is_arithmetic (s : GameScores) : Prop :=
  ∃ (a d : ℕ), s.q1 = a ∧ s.q2 = a + d ∧ s.q3 = a + 2*d ∧ s.q4 = a + 3*d

/-- Checks if the given scores form a geometric sequence -/
def is_geometric (s : GameScores) : Prop :=
  ∃ (b r : ℕ), r > 1 ∧ s.q1 = b ∧ s.q2 = b * r ∧ s.q3 = b * r^2 ∧ s.q4 = b * r^3

/-- Calculates the total score for a team -/
def total_score (s : GameScores) : ℕ := s.q1 + s.q2 + s.q3 + s.q4

/-- Calculates the first half score for a team -/
def first_half_score (s : GameScores) : ℕ := s.q1 + s.q2

/-- The main theorem stating the conditions and the result to be proved -/
theorem basketball_game_theorem (team1 team2 : GameScores) : 
  is_arithmetic team1 →
  is_geometric team2 →
  total_score team1 = total_score team2 + 2 →
  total_score team1 ≤ 100 →
  total_score team2 ≤ 100 →
  first_half_score team1 + first_half_score team2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_theorem_l1014_101458


namespace NUMINAMATH_CALUDE_four_digit_cube_square_sum_multiple_of_seven_l1014_101409

theorem four_digit_cube_square_sum_multiple_of_seven :
  ∃ (x y : ℕ), 
    1000 ≤ x ∧ x < 10000 ∧ 
    7 ∣ x ∧
    x = (y^3 + y^2) / 7 ∧
    (x = 1386 ∨ x = 1200) :=
sorry

end NUMINAMATH_CALUDE_four_digit_cube_square_sum_multiple_of_seven_l1014_101409


namespace NUMINAMATH_CALUDE_addition_sequence_terms_l1014_101475

/-- Represents the nth term of the first sequence in the addition pattern -/
def a (n : ℕ) : ℕ := 2 * n + 1

/-- Represents the nth term of the second sequence in the addition pattern -/
def b (n : ℕ) : ℕ := 5 * n - 1

/-- Proves the correctness of the 10th and 80th terms in the addition sequence -/
theorem addition_sequence_terms :
  (a 10 = 21 ∧ b 10 = 49) ∧ (a 80 = 161 ∧ b 80 = 399) := by
  sorry

#eval a 10  -- Expected: 21
#eval b 10  -- Expected: 49
#eval a 80  -- Expected: 161
#eval b 80  -- Expected: 399

end NUMINAMATH_CALUDE_addition_sequence_terms_l1014_101475


namespace NUMINAMATH_CALUDE_max_value_cubic_quartic_sum_l1014_101451

theorem max_value_cubic_quartic_sum (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_eq_one : x + y + z = 1) :
  x + y^3 + z^4 ≤ 1 ∧ ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧ a + b^3 + c^4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_cubic_quartic_sum_l1014_101451


namespace NUMINAMATH_CALUDE_sum_of_powers_eight_l1014_101413

theorem sum_of_powers_eight (x : ℕ) : 
  x^8 + x^8 + x^8 + x^8 + x^5 = 4 * x^8 + x^5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_eight_l1014_101413


namespace NUMINAMATH_CALUDE_makoto_trip_cost_l1014_101418

/-- Calculates the cost of a round trip given odometer readings, fuel consumption rate, and gas price -/
def round_trip_cost (initial_reading final_reading : ℕ) (fuel_rate : ℚ) (gas_price : ℚ) : ℚ :=
  let distance := final_reading - initial_reading
  let gas_used := (distance : ℚ) / fuel_rate
  gas_used * gas_price

/-- Proves that the cost of Makoto's round trip is $6.00 -/
theorem makoto_trip_cost :
  round_trip_cost 82435 82475 25 (375/100) = 6 := by
  sorry

end NUMINAMATH_CALUDE_makoto_trip_cost_l1014_101418


namespace NUMINAMATH_CALUDE_cube_of_prime_condition_l1014_101492

theorem cube_of_prime_condition (n : ℕ) : 
  (∃ p : ℕ, Nat.Prime p ∧ 2^n + n^2 + 25 = p^3) ↔ n = 6 :=
sorry

end NUMINAMATH_CALUDE_cube_of_prime_condition_l1014_101492


namespace NUMINAMATH_CALUDE_angle_problem_l1014_101423

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define complementary angles
def complementary (a b : Angle) : Prop :=
  a.degrees * 60 + a.minutes + b.degrees * 60 + b.minutes = 90 * 60

-- Define supplementary angles
def supplementary (a b : Angle) : Prop :=
  a.degrees * 60 + a.minutes + b.degrees * 60 + b.minutes = 180 * 60

-- State the theorem
theorem angle_problem (angle1 angle2 angle3 : Angle) :
  complementary angle1 angle2 →
  supplementary angle2 angle3 →
  angle1 = Angle.mk 67 12 →
  angle3 = Angle.mk 157 12 :=
by sorry

end NUMINAMATH_CALUDE_angle_problem_l1014_101423


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1014_101424

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / x) ↔ x ≠ 0 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1014_101424


namespace NUMINAMATH_CALUDE_triangle_cosine_inequality_l1014_101416

theorem triangle_cosine_inequality (A B C : Real) : 
  A > 0 → B > 0 → C > 0 → A + B + C = Real.pi → 
  Real.cos (A/2) + Real.cos (B/2) + Real.cos (C/2) ≤ 3 * Real.sqrt 3 / 2 ∧
  (Real.cos (A/2) + Real.cos (B/2) + Real.cos (C/2) = 3 * Real.sqrt 3 / 2 ↔ 
   A = Real.pi/3 ∧ B = Real.pi/3 ∧ C = Real.pi/3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_cosine_inequality_l1014_101416


namespace NUMINAMATH_CALUDE_divisibility_by_2016_l1014_101410

theorem divisibility_by_2016 (n : ℕ) : 
  2016 ∣ ((n^2 + n)^2 - (n^2 - n)^2) * (n^6 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_2016_l1014_101410


namespace NUMINAMATH_CALUDE_product_repeating_decimal_nine_l1014_101402

theorem product_repeating_decimal_nine (x : ℚ) : x = 1/3 → x * 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_repeating_decimal_nine_l1014_101402


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1014_101459

theorem quadratic_root_difference (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a = 1 ∧ b = -6 ∧ c = 8 → |r₁ - r₂| = 2 :=
by
  sorry

#check quadratic_root_difference

end NUMINAMATH_CALUDE_quadratic_root_difference_l1014_101459


namespace NUMINAMATH_CALUDE_car_distance_proof_l1014_101432

theorem car_distance_proof (initial_time : ℝ) (speed : ℝ) (time_factor : ℝ) : 
  initial_time = 6 →
  speed = 32 →
  time_factor = 3 / 2 →
  speed * (time_factor * initial_time) = 288 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l1014_101432


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1014_101480

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|x - 3| = 5 - x) ↔ (x = 4) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1014_101480


namespace NUMINAMATH_CALUDE_distance_to_origin_of_complex_number_l1014_101448

theorem distance_to_origin_of_complex_number : 
  let i : ℂ := Complex.I
  let z : ℂ := i / (i + 1)
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_distance_to_origin_of_complex_number_l1014_101448


namespace NUMINAMATH_CALUDE_mother_double_age_in_18_years_l1014_101440

/-- Represents the number of years until Xiaoming's mother's age is twice Xiaoming's age -/
def years_until_double_age (xiaoming_age : ℕ) (mother_age : ℕ) : ℕ :=
  mother_age - 2 * xiaoming_age

theorem mother_double_age_in_18_years :
  let xiaoming_current_age : ℕ := 6
  let mother_current_age : ℕ := 30
  years_until_double_age xiaoming_current_age mother_current_age = 18 :=
by
  sorry

#check mother_double_age_in_18_years

end NUMINAMATH_CALUDE_mother_double_age_in_18_years_l1014_101440


namespace NUMINAMATH_CALUDE_inequality_proof_l1014_101474

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^4 + y^4 + z^2 ≥ x*y*z*Real.sqrt 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1014_101474


namespace NUMINAMATH_CALUDE_ln_inequality_implies_p_range_l1014_101400

theorem ln_inequality_implies_p_range (p : ℝ) : 
  (∀ x : ℝ, x > 0 → Real.log x ≤ p * x - 1) → p ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ln_inequality_implies_p_range_l1014_101400


namespace NUMINAMATH_CALUDE_celia_video_streaming_budget_l1014_101490

/-- Represents Celia's monthly budget --/
structure Budget where
  food_per_week : ℕ
  rent : ℕ
  cell_phone : ℕ
  savings : ℕ
  weeks : ℕ
  savings_rate : ℚ

/-- Calculates the total known expenses --/
def total_known_expenses (b : Budget) : ℕ :=
  b.food_per_week * b.weeks + b.rent + b.cell_phone

/-- Calculates the total spending including savings --/
def total_spending (b : Budget) : ℚ :=
  b.savings / b.savings_rate

/-- Calculates the amount set aside for video streaming services --/
def video_streaming_budget (b : Budget) : ℚ :=
  total_spending b - total_known_expenses b

/-- Theorem stating that Celia's video streaming budget is $30 --/
theorem celia_video_streaming_budget :
  ∃ (b : Budget),
    b.food_per_week ≤ 100 ∧
    b.rent = 1500 ∧
    b.cell_phone = 50 ∧
    b.savings = 198 ∧
    b.weeks = 4 ∧
    b.savings_rate = 1/10 ∧
    video_streaming_budget b = 30 :=
  sorry

end NUMINAMATH_CALUDE_celia_video_streaming_budget_l1014_101490


namespace NUMINAMATH_CALUDE_simplify_expression_l1014_101483

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x + y)⁻¹ * (x⁻¹ + y⁻¹) = x⁻¹ * y⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1014_101483


namespace NUMINAMATH_CALUDE_logan_watch_hours_l1014_101469

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The number of minutes Logan watched television -/
def logan_watch_time : ℕ := 300

/-- Theorem: Logan watched television for 5 hours -/
theorem logan_watch_hours : logan_watch_time / minutes_per_hour = 5 := by
  sorry

end NUMINAMATH_CALUDE_logan_watch_hours_l1014_101469


namespace NUMINAMATH_CALUDE_solution_set_equals_union_l1014_101468

-- Define the set of solutions
def solution_set : Set ℝ := {x : ℝ | |x^2 - 2| < 2}

-- State the theorem
theorem solution_set_equals_union : 
  solution_set = Set.union (Set.Ioo (-2) 0) (Set.Ioo 0 2) := by sorry

end NUMINAMATH_CALUDE_solution_set_equals_union_l1014_101468


namespace NUMINAMATH_CALUDE_sum_of_digits_of_even_numbers_up_to_12000_l1014_101493

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Check if a natural number is even -/
def isEven (n : ℕ) : Prop := sorry

/-- Sum of digits of all even numbers in a sequence from 1 to n -/
def sumOfDigitsOfEvenNumbers (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of digits of all even numbers from 1 to 12000 is 129348 -/
theorem sum_of_digits_of_even_numbers_up_to_12000 :
  sumOfDigitsOfEvenNumbers 12000 = 129348 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_even_numbers_up_to_12000_l1014_101493


namespace NUMINAMATH_CALUDE_solution_range_for_a_l1014_101472

/-- The system of equations has a solution with distinct real x, y, z if and only if a is in (23/27, 1) -/
theorem solution_range_for_a (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 + y^2 + z^2 = a ∧
    x^2 + y^3 + z^2 = a ∧
    x^2 + y^2 + z^3 = a) ↔
  (23/27 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_solution_range_for_a_l1014_101472


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l1014_101460

/-- A quadratic function with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The derivative of the function -/
def HasDerivative (f : ℝ → ℝ) : Prop :=
  ∀ x, deriv f x = 2 * x + 2

/-- The function has two equal real roots -/
def HasEqualRoots (f : ℝ → ℝ) : Prop :=
  ∃ r : ℝ, (∀ x, f x = 0 ↔ x = r) ∧ (deriv f r = 0)

/-- The main theorem -/
theorem quadratic_function_theorem (f : ℝ → ℝ) 
  (h1 : QuadraticFunction f) 
  (h2 : HasDerivative f) 
  (h3 : HasEqualRoots f) : 
  ∀ x, f x = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l1014_101460


namespace NUMINAMATH_CALUDE_sum_of_f_values_l1014_101465

noncomputable def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem sum_of_f_values : 
  f 1 + f 2 + f (1/2) + f 3 + f (1/3) + f 4 + f (1/4) = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_values_l1014_101465


namespace NUMINAMATH_CALUDE_sum_of_row_and_column_for_2023_l1014_101489

/-- Represents the value in the table at a given row and column -/
def tableValue (row : ℕ) (col : ℕ) : ℕ :=
  if row % 2 = 1 then
    (row - 1) * 20 + (col - 1) * 2 + 1
  else
    row * 20 - (col - 1) * 2 - 1

/-- The row where 2023 is located -/
def m : ℕ := 253

/-- The column where 2023 is located -/
def n : ℕ := 5

theorem sum_of_row_and_column_for_2023 :
  tableValue m n = 2023 → m + n = 258 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_row_and_column_for_2023_l1014_101489


namespace NUMINAMATH_CALUDE_random_triangle_probability_l1014_101491

/-- The number of ways to choose 3 different numbers from 1 to 179 -/
def total_combinations : ℕ := 939929

/-- The number of valid angle triples that form a triangle -/
def valid_triples : ℕ := 2611

/-- A function that determines if three numbers form valid angles of a triangle -/
def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b + c = 180 ∧ a > 0 ∧ b > 0 ∧ c > 0

/-- The probability of randomly selecting three different numbers from 1 to 179
    that form valid angles of a triangle -/
def triangle_probability : ℚ := valid_triples / total_combinations

/-- Theorem stating the probability of randomly selecting three different numbers
    from 1 to 179 that form valid angles of a triangle -/
theorem random_triangle_probability :
  triangle_probability = 2611 / 939929 := by sorry

end NUMINAMATH_CALUDE_random_triangle_probability_l1014_101491


namespace NUMINAMATH_CALUDE_average_time_per_km_l1014_101420

-- Define the race distance in kilometers
def race_distance : ℝ := 10

-- Define the time for the first half of the race in minutes
def first_half_time : ℝ := 20

-- Define the time for the second half of the race in minutes
def second_half_time : ℝ := 30

-- Theorem statement
theorem average_time_per_km (total_time : ℝ) (avg_time_per_km : ℝ) :
  total_time = first_half_time + second_half_time →
  avg_time_per_km = total_time / race_distance →
  avg_time_per_km = 5 := by
  sorry


end NUMINAMATH_CALUDE_average_time_per_km_l1014_101420


namespace NUMINAMATH_CALUDE_tate_additional_tickets_l1014_101473

/-- The number of additional tickets Tate bought -/
def additional_tickets : ℕ := sorry

theorem tate_additional_tickets : 
  let initial_tickets : ℕ := 32
  let total_tickets : ℕ := initial_tickets + additional_tickets
  let peyton_tickets : ℕ := total_tickets / 2
  51 = total_tickets + peyton_tickets →
  additional_tickets = 2 := by sorry

end NUMINAMATH_CALUDE_tate_additional_tickets_l1014_101473


namespace NUMINAMATH_CALUDE_ellipse_cos_angle_l1014_101433

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  let a := 3
  let b := 2
  let c := Real.sqrt (a^2 - b^2)
  F₁ = (-c, 0) ∧ F₂ = (c, 0)

-- Define a point on the ellipse
def point_on_ellipse (M : ℝ × ℝ) : Prop :=
  ellipse M.1 M.2

-- Define perpendicularity condition
def perpendicular_condition (M F₁ F₂ : ℝ × ℝ) : Prop :=
  (M.1 - F₁.1) * (F₂.1 - F₁.1) + (M.2 - F₁.2) * (F₂.2 - F₁.2) = 0

-- Theorem statement
theorem ellipse_cos_angle (M F₁ F₂ : ℝ × ℝ) :
  foci F₁ F₂ →
  point_on_ellipse M →
  perpendicular_condition M F₁ F₂ →
  let MF₁ := Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2)
  let MF₂ := Real.sqrt ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2)
  MF₁ / MF₂ = 2/7 :=
sorry

end NUMINAMATH_CALUDE_ellipse_cos_angle_l1014_101433


namespace NUMINAMATH_CALUDE_cos_difference_of_complex_exponentials_l1014_101478

theorem cos_difference_of_complex_exponentials 
  (θ φ : ℝ) 
  (h1 : Complex.exp (Complex.I * θ) = 4/5 + 3/5 * Complex.I)
  (h2 : Complex.exp (Complex.I * φ) = 5/13 + 12/13 * Complex.I) : 
  Real.cos (θ - φ) = -16/65 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_of_complex_exponentials_l1014_101478


namespace NUMINAMATH_CALUDE_line_equation_l1014_101479

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y + 21 = 0

-- Define point A
def point_A : ℝ × ℝ := (-6, 7)

-- Define the property of being tangent to the circle
def is_tangent_to_circle (a b c : ℝ) : Prop :=
  let center := (4, -3)
  let radius := 2
  abs (a * center.1 + b * center.2 + c) / Real.sqrt (a^2 + b^2) = radius

-- Theorem statement
theorem line_equation :
  ∃ (a b c : ℝ), 
    (a * point_A.1 + b * point_A.2 + c = 0) ∧
    is_tangent_to_circle a b c ∧
    ((a = 3 ∧ b = 4 ∧ c = -10) ∨ (a = 4 ∧ b = 3 ∧ c = 3)) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l1014_101479


namespace NUMINAMATH_CALUDE_bob_ken_situp_difference_l1014_101412

-- Define the number of sit-ups each person can do
def ken_situps : ℕ := 20
def nathan_situps : ℕ := 2 * ken_situps
def bob_situps : ℕ := (ken_situps + nathan_situps) / 2

-- Theorem statement
theorem bob_ken_situp_difference :
  bob_situps - ken_situps = 10 := by
  sorry

end NUMINAMATH_CALUDE_bob_ken_situp_difference_l1014_101412


namespace NUMINAMATH_CALUDE_aaron_scarves_count_l1014_101437

/-- The number of scarves Aaron made -/
def aaronScarves : ℕ := 10

/-- The number of sweaters Aaron made -/
def aaronSweaters : ℕ := 5

/-- The number of sweaters Enid made -/
def enidSweaters : ℕ := 8

/-- The number of balls of wool used for one scarf -/
def woolPerScarf : ℕ := 3

/-- The number of balls of wool used for one sweater -/
def woolPerSweater : ℕ := 4

/-- The total number of balls of wool used -/
def totalWool : ℕ := 82

theorem aaron_scarves_count : 
  woolPerScarf * aaronScarves + 
  woolPerSweater * (aaronSweaters + enidSweaters) = 
  totalWool := by sorry

end NUMINAMATH_CALUDE_aaron_scarves_count_l1014_101437


namespace NUMINAMATH_CALUDE_custom_operation_equation_l1014_101457

-- Define the custom operation *
def star (a b : ℝ) : ℝ := 4 * a - b

-- State the theorem
theorem custom_operation_equation :
  ∃ x : ℝ, (star 4 (star 3 x) = 2) ∧ (x = -2) := by sorry

end NUMINAMATH_CALUDE_custom_operation_equation_l1014_101457


namespace NUMINAMATH_CALUDE_power_function_through_point_l1014_101467

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Theorem statement
theorem power_function_through_point :
  ∀ f : ℝ → ℝ,
  isPowerFunction f →
  f 2 = 32 →
  ∀ x : ℝ, f x = x^5 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1014_101467


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1014_101436

theorem trigonometric_identity (α : ℝ) : 
  1 - Real.cos (3 * Real.pi / 2 - 3 * α) - Real.sin (3 * α / 2) ^ 2 + Real.cos (3 * α / 2) ^ 2 = 
  2 * Real.sqrt 2 * Real.cos (3 * α / 2) * Real.sin (3 * α / 2 + Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1014_101436


namespace NUMINAMATH_CALUDE_sin_alpha_value_l1014_101430

theorem sin_alpha_value (α : Real) (h : Real.sin (Real.pi - α) = -1/3) :
  Real.sin α = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l1014_101430


namespace NUMINAMATH_CALUDE_bus_seat_difference_l1014_101449

theorem bus_seat_difference :
  let left_seats : ℕ := 15
  let seat_capacity : ℕ := 3
  let back_seat_capacity : ℕ := 9
  let total_capacity : ℕ := 90
  let right_seats : ℕ := (total_capacity - (left_seats * seat_capacity + back_seat_capacity)) / seat_capacity
  left_seats - right_seats = 3 := by
  sorry

end NUMINAMATH_CALUDE_bus_seat_difference_l1014_101449


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1014_101429

theorem diophantine_equation_solutions (n : ℕ) :
  let solutions := {(a, b, c, d) : ℕ × ℕ × ℕ × ℕ | a^2 + b^2 + c^2 + d^2 = 7 * 4^n}
  solutions = {(5 * 2^(n-1), 2^(n-1), 2^(n-1), 2^(n-1)),
               (2^(n+1), 2^n, 2^n, 2^n),
               (3 * 2^(n-1), 3 * 2^(n-1), 3 * 2^(n-1), 2^(n-1))} :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1014_101429


namespace NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l1014_101486

theorem sin_sum_of_complex_exponentials (γ δ : ℝ) :
  Complex.exp (Complex.I * γ) = 4/5 + Complex.I * 3/5 →
  Complex.exp (Complex.I * δ) = -5/13 + Complex.I * 12/13 →
  Real.sin (γ + δ) = 33/65 := by
sorry

end NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l1014_101486


namespace NUMINAMATH_CALUDE_sqrt_calculation_and_algebraic_expression_l1014_101453

theorem sqrt_calculation_and_algebraic_expression :
  (∃ x : ℝ, x^2 = 18) ∧ 
  (∃ y : ℝ, y^2 = 8) ∧ 
  (∃ z : ℝ, z^2 = 1/2) ∧
  (∃ a : ℝ, a^2 = 5) ∧
  (∃ b : ℝ, b^2 = 3) ∧
  (∃ c : ℝ, c^2 = 12) ∧
  (∃ d : ℝ, d^2 = 27) →
  (∃ x y z : ℝ, x^2 = 18 ∧ y^2 = 8 ∧ z^2 = 1/2 ∧ x - y + z = 3 * Real.sqrt 2 / 2) ∧
  (∃ a b c d : ℝ, a^2 = 5 ∧ b^2 = 3 ∧ c^2 = 12 ∧ d^2 = 27 ∧
    (2*a - 1) * (1 + 2*a) + b * (c - d) = 16) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_and_algebraic_expression_l1014_101453


namespace NUMINAMATH_CALUDE_water_sip_calculation_l1014_101403

/-- Proves that given a 2-liter bottle of water consumed in 250 minutes with sips taken every 5 minutes, each sip is 40 ml. -/
theorem water_sip_calculation (bottle_volume : ℕ) (total_time : ℕ) (sip_interval : ℕ) :
  bottle_volume = 2000 →
  total_time = 250 →
  sip_interval = 5 →
  (bottle_volume / (total_time / sip_interval) : ℚ) = 40 := by
  sorry

#check water_sip_calculation

end NUMINAMATH_CALUDE_water_sip_calculation_l1014_101403


namespace NUMINAMATH_CALUDE_reciprocal_roots_implies_p_zero_l1014_101463

-- Define the quadratic equation
def quadratic (p : ℝ) (x : ℝ) : ℝ := 2 * x^2 + p * x + 4

-- Define the condition for reciprocal roots
def has_reciprocal_roots (p : ℝ) : Prop :=
  ∃ (r s : ℝ), r ≠ 0 ∧ s ≠ 0 ∧ r * s = 1 ∧
  quadratic p r = 0 ∧ quadratic p s = 0

-- Theorem statement
theorem reciprocal_roots_implies_p_zero :
  has_reciprocal_roots p → p = 0 := by sorry

end NUMINAMATH_CALUDE_reciprocal_roots_implies_p_zero_l1014_101463


namespace NUMINAMATH_CALUDE_division_problem_l1014_101454

theorem division_problem : (8900 / 6) / 4 = 1483 + 1/3 := by sorry

end NUMINAMATH_CALUDE_division_problem_l1014_101454


namespace NUMINAMATH_CALUDE_square_root_problem_l1014_101404

theorem square_root_problem (m : ℝ) : (Real.sqrt (m - 1) = 2) → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l1014_101404


namespace NUMINAMATH_CALUDE_integral_one_plus_sin_over_pi_halves_l1014_101462

open Real MeasureTheory

theorem integral_one_plus_sin_over_pi_halves : 
  ∫ x in (-π/2)..(π/2), (1 + Real.sin x) = π := by sorry

end NUMINAMATH_CALUDE_integral_one_plus_sin_over_pi_halves_l1014_101462


namespace NUMINAMATH_CALUDE_right_triangle_area_l1014_101456

theorem right_triangle_area (h : ℝ) (θ : ℝ) (area : ℝ) : 
  h = 20 →  -- hypotenuse is 20 inches
  θ = π / 6 →  -- one angle is 30° (π/6 radians)
  area = 50 * Real.sqrt 3 →  -- area is 50√3 square inches
  ∃ (a b : ℝ), 
    a^2 + b^2 = h^2 ∧  -- Pythagorean theorem
    a * b / 2 = area ∧  -- area formula for a triangle
    Real.sin θ = a / h  -- trigonometric relation
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1014_101456


namespace NUMINAMATH_CALUDE_sequence_properties_l1014_101427

-- Define a sequence as a function from natural numbers to real numbers
def Sequence := ℕ → ℝ

-- Statement 1: Sequences appear as isolated points when graphed
def isolated_points (s : Sequence) : Prop :=
  ∀ n : ℕ, ∃ ε > 0, ∀ m : ℕ, m ≠ n → |s m - s n| ≥ ε

-- Statement 2: All sequences have infinite terms
def infinite_terms (s : Sequence) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, m > n

-- Statement 3: The general term formula of a sequence is unique
def unique_formula (s : Sequence) : Prop :=
  ∀ f g : Sequence, (∀ n : ℕ, f n = s n) → (∀ n : ℕ, g n = s n) → f = g

-- Theorem stating that only the first statement is correct
theorem sequence_properties :
  (∀ s : Sequence, isolated_points s) ∧
  (∃ s : Sequence, ¬infinite_terms s) ∧
  (∃ s : Sequence, ¬unique_formula s) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1014_101427


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1014_101452

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a → a 4 = 4 → a 2 * a 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1014_101452


namespace NUMINAMATH_CALUDE_students_behind_in_line_l1014_101407

/-- Given a line of students waiting for a bus, this theorem proves
    the number of students behind a specific student. -/
theorem students_behind_in_line
  (total_students : ℕ)
  (students_in_front : ℕ)
  (h1 : total_students = 30)
  (h2 : students_in_front = 20) :
  total_students - (students_in_front + 1) = 9 := by
  sorry

end NUMINAMATH_CALUDE_students_behind_in_line_l1014_101407


namespace NUMINAMATH_CALUDE_fraction_woodwind_brass_this_year_l1014_101422

-- Define the fractions of students for each instrument last year
def woodwind_last_year : ℚ := 1/2
def brass_last_year : ℚ := 2/5
def percussion_last_year : ℚ := 1 - (woodwind_last_year + brass_last_year)

-- Define the fractions of students who left for each instrument
def woodwind_left : ℚ := 1/2
def brass_left : ℚ := 1/4
def percussion_left : ℚ := 0

-- Calculate the fractions of students for each instrument this year
def woodwind_this_year : ℚ := woodwind_last_year * (1 - woodwind_left)
def brass_this_year : ℚ := brass_last_year * (1 - brass_left)
def percussion_this_year : ℚ := percussion_last_year * (1 - percussion_left)

-- Theorem to prove
theorem fraction_woodwind_brass_this_year :
  woodwind_this_year + brass_this_year = 11/20 := by sorry

end NUMINAMATH_CALUDE_fraction_woodwind_brass_this_year_l1014_101422


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l1014_101405

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 - 8 * x - 16 * y^2 + 32 * y - 12 = 0

/-- The distance between the vertices of the hyperbola -/
def vertex_distance : ℝ := 2

/-- Theorem: The distance between the vertices of the hyperbola is 2 -/
theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, hyperbola_equation x y → vertex_distance = 2 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l1014_101405


namespace NUMINAMATH_CALUDE_triangle_division_regions_l1014_101411

/-- Given a triangle ABC and a positive integer n, with the sides divided into 2^n equal parts
    and cevians drawn as described, the number of regions into which the triangle is divided
    is equal to 3 · 2^(2n) - 6 · 2^n + 6. -/
theorem triangle_division_regions (n : ℕ+) : ℕ := by
  sorry

end NUMINAMATH_CALUDE_triangle_division_regions_l1014_101411


namespace NUMINAMATH_CALUDE_sum_difference_equals_result_l1014_101488

theorem sum_difference_equals_result : 12.1212 + 17.0005 - 9.1103 = 20.0114 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_equals_result_l1014_101488


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1014_101434

def A : Set ℝ := {0, 2, 4, 6}
def B : Set ℝ := {x | 3 < x ∧ x < 7}

theorem intersection_of_A_and_B : A ∩ B = {4, 6} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1014_101434


namespace NUMINAMATH_CALUDE_product_not_equal_48_l1014_101406

theorem product_not_equal_48 : ∃! (a b : ℚ), (a, b) ∈ ({(-4, -12), (-3, -16), (1/2, -96), (1, 48), (4/3, 36)} : Set (ℚ × ℚ)) ∧ a * b ≠ 48 := by
  sorry

end NUMINAMATH_CALUDE_product_not_equal_48_l1014_101406


namespace NUMINAMATH_CALUDE_bus_tour_tickets_l1014_101496

/-- Represents the total number of tickets sold in a local bus tour. -/
def total_tickets (senior_tickets : ℕ) (regular_tickets : ℕ) : ℕ :=
  senior_tickets + regular_tickets

/-- Represents the total sales from tickets. -/
def total_sales (senior_tickets : ℕ) (regular_tickets : ℕ) : ℕ :=
  10 * senior_tickets + 15 * regular_tickets

theorem bus_tour_tickets :
  ∃ (senior_tickets : ℕ),
    total_tickets senior_tickets 41 = 65 ∧
    total_sales senior_tickets 41 = 855 :=
by sorry

end NUMINAMATH_CALUDE_bus_tour_tickets_l1014_101496


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1014_101497

theorem complex_equation_solution (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) :
  z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1014_101497


namespace NUMINAMATH_CALUDE_tangent_line_problem_l1014_101444

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := x^3 + (5/2) * x^2 + 3 * Real.log x + b

theorem tangent_line_problem (b : ℝ) :
  (∃ (m : ℝ), (g b 1 = m * 1 - 5) ∧ 
              (∀ (x : ℝ), x ≠ 1 → (g b x - g b 1) / (x - 1) < m)) →
  b = 5/2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l1014_101444


namespace NUMINAMATH_CALUDE_sum_first_11_odd_numbers_l1014_101484

theorem sum_first_11_odd_numbers : 
  (Finset.range 11).sum (fun i => 2 * i + 1) = 121 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_11_odd_numbers_l1014_101484


namespace NUMINAMATH_CALUDE_fifth_element_row_20_l1014_101425

/-- Definition of binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Pascal's triangle element at row n, position k -/
def pascal_triangle_element (n k : ℕ) : ℕ := binomial n (k - 1)

/-- The fifth element in Row 20 of Pascal's triangle is 4845 -/
theorem fifth_element_row_20 : pascal_triangle_element 20 5 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_fifth_element_row_20_l1014_101425
