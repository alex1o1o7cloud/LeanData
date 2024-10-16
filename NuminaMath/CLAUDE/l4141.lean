import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l4141_414119

theorem arithmetic_square_root_of_16 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l4141_414119


namespace NUMINAMATH_CALUDE_meeting_percentage_is_42_percent_l4141_414133

def work_day_hours : ℕ := 10
def lunch_break_minutes : ℕ := 30
def first_meeting_minutes : ℕ := 60

def work_day_minutes : ℕ := work_day_hours * 60
def effective_work_minutes : ℕ := work_day_minutes - lunch_break_minutes
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

def meeting_percentage : ℚ := (total_meeting_minutes : ℚ) / (effective_work_minutes : ℚ) * 100

theorem meeting_percentage_is_42_percent : 
  ⌊meeting_percentage⌋ = 42 :=
sorry

end NUMINAMATH_CALUDE_meeting_percentage_is_42_percent_l4141_414133


namespace NUMINAMATH_CALUDE_circle_max_cube_root_sum_l4141_414188

theorem circle_max_cube_root_sum (x y : ℝ) : 
  x^2 + y^2 = 1 → 
  ∀ a b : ℝ, a^2 + b^2 = 1 → 
  Real.sqrt (|x|^3 + |y|^3) ≤ Real.sqrt (2 * Real.sqrt 2 + 1) / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_max_cube_root_sum_l4141_414188


namespace NUMINAMATH_CALUDE_probability_x_less_than_y_l4141_414160

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the condition x < y
def condition (p : ℝ × ℝ) : Prop := p.1 < p.2

-- Define the probability measure on the rectangle
noncomputable def prob : MeasureTheory.ProbabilityMeasure (ℝ × ℝ) :=
  sorry

-- State the theorem
theorem probability_x_less_than_y :
  prob {p ∈ rectangle | condition p} = 1/8 := by sorry

end NUMINAMATH_CALUDE_probability_x_less_than_y_l4141_414160


namespace NUMINAMATH_CALUDE_divisibility_condition_l4141_414192

theorem divisibility_condition (a b : ℕ+) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔
  ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l4141_414192


namespace NUMINAMATH_CALUDE_two_face_painted_count_l4141_414196

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents a painted cube -/
structure PaintedCube (n : ℕ) extends Cube n where
  painted_faces : ℕ := 6

/-- Represents a painted cube cut into unit cubes -/
structure CutPaintedCube (n : ℕ) extends PaintedCube n

/-- The number of unit cubes with at least two painted faces in a cut painted cube -/
def num_two_face_painted (c : CutPaintedCube 4) : ℕ := 32

theorem two_face_painted_count (c : CutPaintedCube 4) : 
  num_two_face_painted c = 32 := by sorry

end NUMINAMATH_CALUDE_two_face_painted_count_l4141_414196


namespace NUMINAMATH_CALUDE_sons_age_l4141_414125

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 26 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 24 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l4141_414125


namespace NUMINAMATH_CALUDE_common_solution_l4141_414154

def m_values : List ℤ := [-5, -4, -3, -1, 0, 1, 3, 23, 124, 1000]

def equation (m x y : ℤ) : Prop :=
  (2 * m + 1) * x + (2 - 3 * m) * y + 1 - 5 * m = 0

theorem common_solution :
  ∀ m ∈ m_values, equation m 1 (-1) :=
by sorry

end NUMINAMATH_CALUDE_common_solution_l4141_414154


namespace NUMINAMATH_CALUDE_final_passengers_count_l4141_414150

/-- The number of people on the bus after all stops -/
def final_passengers : ℕ :=
  let initial := 110
  let stop1 := initial - 20 + 15
  let stop2 := stop1 - 34 + 17
  let stop3 := stop2 - 18 + 7
  let stop4 := stop3 - 29 + 19
  let stop5 := stop4 - 11 + 13
  let stop6 := stop5 - 15 + 8
  let stop7 := stop6 - 13 + 5
  let stop8 := stop7 - 6 + 0
  stop8

/-- Theorem stating that the final number of passengers is 48 -/
theorem final_passengers_count : final_passengers = 48 := by
  sorry

end NUMINAMATH_CALUDE_final_passengers_count_l4141_414150


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_16_l4141_414113

theorem sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 ∨ Real.sqrt (Real.sqrt 16) = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_16_l4141_414113


namespace NUMINAMATH_CALUDE_same_gender_probability_l4141_414159

/-- The probability of selecting two students of the same gender -/
theorem same_gender_probability (n_male n_female : ℕ) (h_male : n_male = 2) (h_female : n_female = 8) :
  let total := n_male + n_female
  let same_gender_ways := Nat.choose n_male 2 + Nat.choose n_female 2
  let total_ways := Nat.choose total 2
  (same_gender_ways : ℚ) / total_ways = 29 / 45 := by sorry

end NUMINAMATH_CALUDE_same_gender_probability_l4141_414159


namespace NUMINAMATH_CALUDE_area_fraction_above_line_l4141_414153

/-- The fraction of the area of a square above a line -/
def fraction_above_line (square_vertices : Fin 4 → ℝ × ℝ) (line_point1 line_point2 : ℝ × ℝ) : ℚ :=
  sorry

/-- The theorem statement -/
theorem area_fraction_above_line :
  let square_vertices : Fin 4 → ℝ × ℝ := ![
    (2, 1), (5, 1), (5, 4), (2, 4)
  ]
  let line_point1 : ℝ × ℝ := (2, 3)
  let line_point2 : ℝ × ℝ := (5, 1)
  fraction_above_line square_vertices line_point1 line_point2 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_area_fraction_above_line_l4141_414153


namespace NUMINAMATH_CALUDE_degree_of_q_l4141_414168

-- Define polynomials p, q, and i
variable (p q i : Polynomial ℝ)

-- Define the relationship between i, p, and q
def poly_relation (p q i : Polynomial ℝ) : Prop :=
  i = p.comp q ^ 2 - q ^ 3

-- State the theorem
theorem degree_of_q (hp : Polynomial.degree p = 4)
                    (hi : Polynomial.degree i = 12)
                    (h_rel : poly_relation p q i) :
  Polynomial.degree q = 4 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_q_l4141_414168


namespace NUMINAMATH_CALUDE_base8_square_unique_l4141_414138

/-- Converts a base-10 number to base-8 --/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 8) :: aux (m / 8)
  aux n |>.reverse

/-- Checks if a list contains each number from 0 to 7 exactly once --/
def containsEachDigitOnce (l : List ℕ) : Prop :=
  ∀ d, d ∈ Finset.range 8 → (l.count d = 1)

/-- The main theorem --/
theorem base8_square_unique : 
  ∃! n : ℕ, 
    (toBase8 n).length = 3 ∧ 
    containsEachDigitOnce (toBase8 n) ∧
    containsEachDigitOnce (toBase8 (n * n)) ∧
    n = 256 := by sorry

end NUMINAMATH_CALUDE_base8_square_unique_l4141_414138


namespace NUMINAMATH_CALUDE_no_double_application_successor_function_l4141_414146

theorem no_double_application_successor_function :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_successor_function_l4141_414146


namespace NUMINAMATH_CALUDE_sum_of_integers_l4141_414158

theorem sum_of_integers (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a - 1) * (b - 1) * (c - 1) * (d - 1) = 25 →
  a + b + c + d = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l4141_414158


namespace NUMINAMATH_CALUDE_handshakes_at_reunion_l4141_414156

/-- Represents a family reunion with married couples -/
structure FamilyReunion where
  couples : ℕ
  people_per_couple : ℕ := 2

/-- Calculates the total number of handshakes at a family reunion -/
def total_handshakes (reunion : FamilyReunion) : ℕ :=
  let total_people := reunion.couples * reunion.people_per_couple
  let handshakes_per_person := total_people - 1 - 1 - (3 * reunion.people_per_couple)
  (total_people * handshakes_per_person) / 2

/-- Theorem: The total number of handshakes at a specific family reunion is 64 -/
theorem handshakes_at_reunion :
  let reunion : FamilyReunion := { couples := 8 }
  total_handshakes reunion = 64 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_at_reunion_l4141_414156


namespace NUMINAMATH_CALUDE_train_speed_calculation_l4141_414155

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (time : ℝ) :
  train_length = 250 ∧ bridge_length = 120 ∧ time = 20 →
  (train_length + bridge_length) / time = 18.5 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l4141_414155


namespace NUMINAMATH_CALUDE_lipschitz_arithmetic_is_translation_l4141_414131

/-- A function f : ℝ → ℝ satisfying the given conditions -/
def LipschitzArithmeticFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, |f x - f y| ≤ |x - y|) ∧
  (∀ x : ℝ, ∃ d : ℝ, ∀ n : ℕ, (f^[n]) x = x + n • d)

/-- The main theorem -/
theorem lipschitz_arithmetic_is_translation
  (f : ℝ → ℝ) (h : LipschitzArithmeticFunction f) :
  ∃ a : ℝ, ∀ x : ℝ, f x = x + a := by
  sorry

end NUMINAMATH_CALUDE_lipschitz_arithmetic_is_translation_l4141_414131


namespace NUMINAMATH_CALUDE_proposition_equivalences_l4141_414124

theorem proposition_equivalences (a b c : ℝ) :
  (((c < 0 ∧ a * c > b * c) → a < b) ∧
   ((c < 0 ∧ a < b) → a * c > b * c) ∧
   ((c < 0 ∧ a * c ≤ b * c) → a ≥ b) ∧
   ((c < 0 ∧ a ≥ b) → a * c ≤ b * c) ∧
   ((a * b = 0) → (a = 0 ∨ b = 0)) ∧
   ((a = 0 ∨ b = 0) → a * b = 0) ∧
   ((a * b ≠ 0) → (a ≠ 0 ∧ b ≠ 0)) ∧
   ((a ≠ 0 ∧ b ≠ 0) → a * b ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_equivalences_l4141_414124


namespace NUMINAMATH_CALUDE_unique_remainder_mod_10_l4141_414101

theorem unique_remainder_mod_10 : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_remainder_mod_10_l4141_414101


namespace NUMINAMATH_CALUDE_ln_inequality_l4141_414114

theorem ln_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < Real.exp 1) :
  a * Real.log b > b * Real.log a := by
  sorry

end NUMINAMATH_CALUDE_ln_inequality_l4141_414114


namespace NUMINAMATH_CALUDE_unique_quadrilateral_perimeter_unique_perimeter_value_l4141_414121

/-- Represents a quadrilateral with integer side lengths -/
structure Quadrilateral where
  AB : ℕ+
  BC : ℕ+
  CD : ℕ+
  AD : ℕ+

/-- The perimeter of a quadrilateral -/
def perimeter (q : Quadrilateral) : ℕ :=
  q.AB.val + q.BC.val + q.CD.val + q.AD.val

/-- Theorem stating that there is a unique quadrilateral satisfying the given conditions -/
theorem unique_quadrilateral_perimeter :
  ∃! (q : Quadrilateral),
    q.AB = 3 ∧
    q.BC = q.AD - 1 ∧
    q.BC = q.CD - 1 ∧
    (q.AB ^ 2 + q.BC ^ 2 : ℕ) = q.AD ^ 2 ∧
    (q.CD ^ 2 + q.BC ^ 2 : ℕ) = q.AD ^ 2 ∧
    perimeter q = 17 :=
  sorry

/-- Corollary: The perimeter of the unique quadrilateral is 17 -/
theorem unique_perimeter_value (p : ℕ) :
  (∃ (q : Quadrilateral),
    q.AB = 3 ∧
    q.BC = q.AD - 1 ∧
    q.BC = q.CD - 1 ∧
    (q.AB ^ 2 + q.BC ^ 2 : ℕ) = q.AD ^ 2 ∧
    (q.CD ^ 2 + q.BC ^ 2 : ℕ) = q.AD ^ 2 ∧
    perimeter q = p) →
  p = 17 :=
  sorry

end NUMINAMATH_CALUDE_unique_quadrilateral_perimeter_unique_perimeter_value_l4141_414121


namespace NUMINAMATH_CALUDE_expected_steps_is_five_l4141_414143

/-- The coloring process on the unit interval [0,1] --/
structure ColoringProcess where
  /-- The random selection of x in [0,1] --/
  select_x : Unit → Real
  /-- The coloring rule for x ≤ 1/2 --/
  color_left (x : Real) : Set Real := { y | x ≤ y ∧ y ≤ x + 1/2 }
  /-- The coloring rule for x > 1/2 --/
  color_right (x : Real) : Set Real := { y | x ≤ y ∧ y ≤ 1 } ∪ { y | 0 ≤ y ∧ y ≤ x - 1/2 }

/-- The expected number of steps to color the entire interval --/
def expected_steps (process : ColoringProcess) : Real :=
  5  -- The actual value we want to prove

/-- The theorem stating that the expected number of steps is 5 --/
theorem expected_steps_is_five (process : ColoringProcess) :
  expected_steps process = 5 := by sorry

end NUMINAMATH_CALUDE_expected_steps_is_five_l4141_414143


namespace NUMINAMATH_CALUDE_smallest_b_value_l4141_414165

theorem smallest_b_value (a b c : ℕ) : 
  (a * b * c = 360) → 
  (1 < a) → (a < b) → (b < c) → 
  (∀ b' : ℕ, (∃ a' c' : ℕ, a' * b' * c' = 360 ∧ 1 < a' ∧ a' < b' ∧ b' < c') → b ≤ b') → 
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l4141_414165


namespace NUMINAMATH_CALUDE_employee_payment_percentage_l4141_414100

theorem employee_payment_percentage (total payment_B : ℝ) 
  (h1 : total = 570)
  (h2 : payment_B = 228) : 
  (total - payment_B) / payment_B * 100 = 150 := by
  sorry

end NUMINAMATH_CALUDE_employee_payment_percentage_l4141_414100


namespace NUMINAMATH_CALUDE_initial_players_count_video_game_players_l4141_414180

theorem initial_players_count (players_quit : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  let remaining_players := total_lives / lives_per_player
  remaining_players + players_quit

theorem video_game_players : initial_players_count 7 8 24 = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_players_count_video_game_players_l4141_414180


namespace NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l4141_414102

theorem tan_seventeen_pi_fourths : Real.tan (17 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l4141_414102


namespace NUMINAMATH_CALUDE_perpendicular_vector_scalar_l4141_414107

theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (x : ℝ) :
  a = (3, 4) →
  b = (2, -1) →
  (a.1 + x * b.1, a.2 + x * b.2) • b = 0 →
  x = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vector_scalar_l4141_414107


namespace NUMINAMATH_CALUDE_ratio_problem_l4141_414136

theorem ratio_problem (w x y z : ℝ) (hw : w ≠ 0) 
  (h1 : w / x = 2 / 3) 
  (h2 : w / y = 6 / 15) 
  (h3 : w / z = 4 / 5) : 
  (x + y) / z = 16 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l4141_414136


namespace NUMINAMATH_CALUDE_tv_clients_count_l4141_414197

def total_clients : ℕ := 180
def radio_clients : ℕ := 110
def magazine_clients : ℕ := 130
def tv_and_magazine_clients : ℕ := 85
def tv_and_radio_clients : ℕ := 75
def radio_and_magazine_clients : ℕ := 95
def all_three_clients : ℕ := 80

theorem tv_clients_count :
  ∃ (tv_clients : ℕ),
    tv_clients = total_clients + all_three_clients - radio_clients - magazine_clients + 
                 tv_and_magazine_clients + tv_and_radio_clients + radio_and_magazine_clients ∧
    tv_clients = 130 := by
  sorry

end NUMINAMATH_CALUDE_tv_clients_count_l4141_414197


namespace NUMINAMATH_CALUDE_not_always_swap_cities_l4141_414167

/-- A graph representing cities and their railroad connections. -/
structure CityGraph where
  V : Type
  E : V → V → Prop

/-- A bijective function representing a renaming of cities. -/
def IsRenaming (G : CityGraph) (f : G.V → G.V) : Prop :=
  Function.Bijective f

/-- A renaming that preserves the graph structure (i.e., a graph isomorphism). -/
def IsValidRenaming (G : CityGraph) (f : G.V → G.V) : Prop :=
  IsRenaming G f ∧ ∀ u v : G.V, G.E u v ↔ G.E (f u) (f v)

/-- For any two cities, there exists a valid renaming that maps one to the other. -/
axiom any_city_can_be_renamed (G : CityGraph) :
  ∀ u v : G.V, ∃ f : G.V → G.V, IsValidRenaming G f ∧ f u = v

/-- The theorem to be proved. -/
theorem not_always_swap_cities (G : CityGraph) :
  ¬(∀ x y : G.V, ∃ f : G.V → G.V, IsValidRenaming G f ∧ f x = y ∧ f y = x) :=
sorry

end NUMINAMATH_CALUDE_not_always_swap_cities_l4141_414167


namespace NUMINAMATH_CALUDE_forty_coins_impossible_l4141_414172

/-- Represents the contents of Bethany's purse -/
structure Purse where
  pound_coins : ℕ
  twenty_pence : ℕ
  fifty_pence : ℕ

/-- Calculates the total value of coins in pence -/
def total_value (p : Purse) : ℕ :=
  100 * p.pound_coins + 20 * p.twenty_pence + 50 * p.fifty_pence

/-- Calculates the total number of coins -/
def total_coins (p : Purse) : ℕ :=
  p.pound_coins + p.twenty_pence + p.fifty_pence

/-- Represents Bethany's purse with the given conditions -/
def bethany_purse : Purse :=
  { pound_coins := 11
  , twenty_pence := 0  -- placeholder, actual value unknown
  , fifty_pence := 0 } -- placeholder, actual value unknown

/-- The mean value of coins in pence -/
def mean_value : ℚ := 52

theorem forty_coins_impossible :
  ∀ p : Purse,
    p.pound_coins = 11 →
    (total_value p : ℚ) / (total_coins p : ℚ) = mean_value →
    total_coins p ≠ 40 :=
by sorry

end NUMINAMATH_CALUDE_forty_coins_impossible_l4141_414172


namespace NUMINAMATH_CALUDE_negative_three_times_two_l4141_414177

theorem negative_three_times_two : (-3) * 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_times_two_l4141_414177


namespace NUMINAMATH_CALUDE_trisomy21_caused_by_sperm_l4141_414120

/-- Represents a genotype for the STR marker on chromosome 21 -/
inductive Genotype
  | Negative
  | Positive
  | DoublePositive

/-- Represents a person with their genotype -/
structure Person where
  genotype : Genotype

/-- Represents a family with a child, father, and mother -/
structure Family where
  child : Person
  father : Person
  mother : Person

/-- Defines Trisomy 21 syndrome -/
def hasTrisomy21 (p : Person) : Prop := p.genotype = Genotype.DoublePositive

/-- Defines the condition of sperm having 2 chromosome 21s -/
def spermHasTwoChromosome21 (f : Family) : Prop :=
  f.father.genotype = Genotype.Positive ∧
  f.mother.genotype = Genotype.Negative ∧
  f.child.genotype = Genotype.DoublePositive

/-- Theorem stating that given the family's genotypes, the child's Trisomy 21 is caused by sperm with 2 chromosome 21s -/
theorem trisomy21_caused_by_sperm (f : Family)
  (h_child : f.child.genotype = Genotype.DoublePositive)
  (h_father : f.father.genotype = Genotype.Positive)
  (h_mother : f.mother.genotype = Genotype.Negative) :
  hasTrisomy21 f.child ∧ spermHasTwoChromosome21 f := by
  sorry


end NUMINAMATH_CALUDE_trisomy21_caused_by_sperm_l4141_414120


namespace NUMINAMATH_CALUDE_violet_buddy_hiking_time_l4141_414166

/-- Represents the hiking scenario of Violet and Buddy -/
structure HikingScenario where
  violet_water_rate : Real  -- ml per hour
  buddy_water_rate : Real   -- ml per hour
  violet_capacity : Real    -- L
  buddy_capacity : Real     -- L
  hiking_speed : Real       -- km/h
  break_interval : Real     -- hours
  break_duration : Real     -- hours

/-- Calculates the total time Violet and Buddy can spend on the trail before running out of water -/
def total_trail_time (scenario : HikingScenario) : Real :=
  sorry

/-- Theorem stating that Violet and Buddy can spend 6.25 hours on the trail before running out of water -/
theorem violet_buddy_hiking_time :
  let scenario : HikingScenario := {
    violet_water_rate := 800,
    buddy_water_rate := 400,
    violet_capacity := 4.8,
    buddy_capacity := 1.5,
    hiking_speed := 4,
    break_interval := 2,
    break_duration := 0.5
  }
  total_trail_time scenario = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_violet_buddy_hiking_time_l4141_414166


namespace NUMINAMATH_CALUDE_snail_reaches_tree_on_day_37_l4141_414108

/-- The number of days it takes for a snail to reach another tree -/
def days_to_reach_tree (s l₁ l₂ : ℕ) : ℕ :=
  let daily_progress := l₁ - l₂
  let days_to_final_stretch := (s - l₁) / daily_progress
  days_to_final_stretch + 1

/-- Theorem stating that the snail reaches the tree on the 37th day -/
theorem snail_reaches_tree_on_day_37 :
  days_to_reach_tree 40 4 3 = 37 := by
  sorry

#eval days_to_reach_tree 40 4 3

end NUMINAMATH_CALUDE_snail_reaches_tree_on_day_37_l4141_414108


namespace NUMINAMATH_CALUDE_number_division_and_addition_l4141_414127

theorem number_division_and_addition (x : ℝ) : x / 9 = 8 → x + 11 = 83 := by
  sorry

end NUMINAMATH_CALUDE_number_division_and_addition_l4141_414127


namespace NUMINAMATH_CALUDE_stone_pile_division_l4141_414144

/-- Two natural numbers are similar if they differ by no more than twice -/
def similar (a b : ℕ) : Prop := a ≤ b ∧ b ≤ 2 * a

/-- A sequence of operations to combine piles -/
inductive CombineSeq : ℕ → ℕ → Type
  | single : (n : ℕ) → CombineSeq n 1
  | combine : {n m k : ℕ} → (s : CombineSeq n m) → (t : CombineSeq n k) → 
              similar m k → CombineSeq n (m + k)

/-- Any pile of stones can be divided into piles of single stones -/
theorem stone_pile_division (n : ℕ) : CombineSeq n n := by sorry

end NUMINAMATH_CALUDE_stone_pile_division_l4141_414144


namespace NUMINAMATH_CALUDE_unique_intersection_implies_line_equation_l4141_414147

/-- Given a line y = mx + b passing through (2, 3), prove that if there exists exactly one k
    where x = k intersects y = x^2 - 4x + 4 and y = mx + b at points 6 units apart,
    then m = -6 and b = 15 -/
theorem unique_intersection_implies_line_equation 
  (m b : ℝ) 
  (passes_through : 3 = 2 * m + b) 
  (h : ∃! k : ℝ, ∃ y₁ y₂ : ℝ, 
    y₁ = k^2 - 4*k + 4 ∧ 
    y₂ = m*k + b ∧ 
    (y₁ - y₂)^2 = 36) : 
  m = -6 ∧ b = 15 := by
sorry

end NUMINAMATH_CALUDE_unique_intersection_implies_line_equation_l4141_414147


namespace NUMINAMATH_CALUDE_sets_intersection_and_complement_l4141_414134

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 * x^2 + a * x + 2 = 0}
def B (b : ℝ) : Set ℝ := {x | x^2 + 3 * x - b = 0}

-- State the theorem
theorem sets_intersection_and_complement (a b : ℝ) :
  (A a ∩ B b = {2}) →
  ∃ (U : Set ℝ),
    a = -5 ∧
    b = 10 ∧
    U = A a ∪ B b ∧
    (Uᶜ ∩ A a) ∪ (Uᶜ ∩ B b) = {-5, 1/2} := by
  sorry

end NUMINAMATH_CALUDE_sets_intersection_and_complement_l4141_414134


namespace NUMINAMATH_CALUDE_sum_of_digits_l4141_414105

def is_valid_arrangement (digits : Finset ℕ) (vertical horizontal : Finset ℕ) : Prop :=
  digits.card = 7 ∧ 
  digits ⊆ Finset.range 9 ∧ 
  vertical.card = 4 ∧ 
  horizontal.card = 4 ∧ 
  (vertical ∩ horizontal).card = 1 ∧
  vertical ⊆ digits ∧ 
  horizontal ⊆ digits

theorem sum_of_digits 
  (digits : Finset ℕ) 
  (vertical horizontal : Finset ℕ) 
  (h_valid : is_valid_arrangement digits vertical horizontal)
  (h_vertical_sum : vertical.sum id = 26)
  (h_horizontal_sum : horizontal.sum id = 20) :
  digits.sum id = 32 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l4141_414105


namespace NUMINAMATH_CALUDE_complex_sum_of_powers_l4141_414170

theorem complex_sum_of_powers : 
  ((-1 + Complex.I * Real.sqrt 3) / 2) ^ 12 + ((-1 - Complex.I * Real.sqrt 3) / 2) ^ 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_powers_l4141_414170


namespace NUMINAMATH_CALUDE_max_q_minus_r_for_852_l4141_414161

theorem max_q_minus_r_for_852 :
  ∃ (q r : ℕ), 
    q > 0 ∧ r > 0 ∧ 
    852 = 21 * q + r ∧
    ∀ (q' r' : ℕ), q' > 0 → r' > 0 → 852 = 21 * q' + r' → q' - r' ≤ q - r ∧
    q - r = 28 :=
sorry

end NUMINAMATH_CALUDE_max_q_minus_r_for_852_l4141_414161


namespace NUMINAMATH_CALUDE_product_of_numbers_l4141_414185

theorem product_of_numbers (x y : ℝ) 
  (h1 : (x - y)^2 / (x + y)^3 = 4 / 27)
  (h2 : x + y = 5 * (x - y) + 3) : 
  x * y = 15.75 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l4141_414185


namespace NUMINAMATH_CALUDE_horner_method_example_l4141_414171

def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 3*x + 2

theorem horner_method_example : f (-2) = 320 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_example_l4141_414171


namespace NUMINAMATH_CALUDE_race_heartbeats_l4141_414106

/-- Calculates the total number of heartbeats during a race given the race distance, cycling pace, and heart rate. -/
def total_heartbeats (race_distance : ℕ) (cycling_pace : ℕ) (heart_rate : ℕ) : ℕ :=
  race_distance * cycling_pace * heart_rate

/-- Theorem stating that for a 100-mile race, with a cycling pace of 4 minutes per mile and a heart rate of 120 beats per minute, the total number of heartbeats is 48000. -/
theorem race_heartbeats :
  total_heartbeats 100 4 120 = 48000 := by
  sorry

end NUMINAMATH_CALUDE_race_heartbeats_l4141_414106


namespace NUMINAMATH_CALUDE_lcm_18_30_l4141_414122

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_30_l4141_414122


namespace NUMINAMATH_CALUDE_unique_square_with_property_l4141_414140

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def all_digits_less_than_7 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ (n.digits 10) → d < 7

def add_3_to_digits (n : ℕ) : ℕ :=
  n.digits 10
   |> List.map (· + 3)
   |> List.foldl (λ acc d => acc * 10 + d) 0

theorem unique_square_with_property :
  ∃! N : ℕ,
    1000 ≤ N ∧ N < 10000 ∧
    is_perfect_square N ∧
    all_digits_less_than_7 N ∧
    is_perfect_square (add_3_to_digits N) ∧
    N = 1156 :=
sorry

end NUMINAMATH_CALUDE_unique_square_with_property_l4141_414140


namespace NUMINAMATH_CALUDE_ln_sufficient_not_necessary_l4141_414116

-- Define the statement that ln a > ln b implies e^a > e^b
def ln_implies_exp (a b : ℝ) : Prop :=
  (∀ a b : ℝ, Real.log a > Real.log b → Real.exp a > Real.exp b)

-- Define the statement that e^a > e^b does not always imply ln a > ln b
def exp_not_always_implies_ln (a b : ℝ) : Prop :=
  (∃ a b : ℝ, Real.exp a > Real.exp b ∧ ¬(Real.log a > Real.log b))

-- Theorem stating that ln a > ln b is sufficient but not necessary for e^a > e^b
theorem ln_sufficient_not_necessary :
  (∀ a b : ℝ, ln_implies_exp a b) ∧ (∃ a b : ℝ, exp_not_always_implies_ln a b) :=
sorry

end NUMINAMATH_CALUDE_ln_sufficient_not_necessary_l4141_414116


namespace NUMINAMATH_CALUDE_second_year_students_l4141_414183

/-- Represents the number of students in each year and the total number of students. -/
structure SchoolPopulation where
  firstYear : ℕ
  secondYear : ℕ
  thirdYear : ℕ
  total : ℕ

/-- Represents the sample size and the number of first-year students in the sample. -/
structure Sample where
  size : ℕ
  firstYearSample : ℕ

/-- 
Proves that given the conditions of the problem, the number of second-year students is 300.
-/
theorem second_year_students 
  (school : SchoolPopulation)
  (sample : Sample)
  (h1 : school.firstYear = 450)
  (h2 : school.thirdYear = 250)
  (h3 : sample.size = 60)
  (h4 : sample.firstYearSample = 27)
  (h5 : (school.firstYear : ℚ) / school.total = sample.firstYearSample / sample.size) :
  school.secondYear = 300 := by
  sorry

#check second_year_students

end NUMINAMATH_CALUDE_second_year_students_l4141_414183


namespace NUMINAMATH_CALUDE_combined_average_age_l4141_414126

theorem combined_average_age (people_c : ℕ) (avg_c : ℚ) (people_d : ℕ) (avg_d : ℚ) :
  people_c = 8 →
  avg_c = 35 →
  people_d = 6 →
  avg_d = 30 →
  (people_c * avg_c + people_d * avg_d) / (people_c + people_d : ℚ) = 33 := by
  sorry

end NUMINAMATH_CALUDE_combined_average_age_l4141_414126


namespace NUMINAMATH_CALUDE_traffic_light_change_probability_l4141_414193

/-- Represents the duration of a traffic light cycle in seconds -/
def cycle_duration : ℕ := 90

/-- Represents the duration of the green light in seconds -/
def green_duration : ℕ := 45

/-- Represents the duration of the yellow light in seconds -/
def yellow_duration : ℕ := 5

/-- Represents the duration of the red light in seconds -/
def red_duration : ℕ := 40

/-- Represents the duration of Sam's observation in seconds -/
def observation_duration : ℕ := 5

/-- Theorem: The probability of observing a color change during a random 5-second interval
    in a 90-second traffic light cycle (with 45s green, 5s yellow, 40s red) is 1/6 -/
theorem traffic_light_change_probability :
  (3 * observation_duration : ℚ) / cycle_duration = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_change_probability_l4141_414193


namespace NUMINAMATH_CALUDE_locus_of_points_l4141_414169

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  a : ℝ
  b : ℝ
  center : Point

/-- Represents an octagon -/
structure Octagon where
  vertices : Fin 8 → Point

/-- Calculate the absolute distance from a point to a line segment -/
def distToSegment (p : Point) (s1 s2 : Point) : ℝ :=
  sorry

/-- Calculate the sum of distances from a point to the sides of a rectangle -/
def sumDistToSides (p : Point) (r : Rectangle) : ℝ :=
  sorry

/-- Check if a point is inside or on the boundary of an octagon -/
def isInOctagon (p : Point) (o : Octagon) : Prop :=
  sorry

/-- Construct the octagon based on the rectangle and c value -/
def constructOctagon (r : Rectangle) (c : ℝ) : Octagon :=
  sorry

/-- The main theorem statement -/
theorem locus_of_points (r : Rectangle) (c : ℝ) :
  ∀ p : Point, sumDistToSides p r = r.a + r.b + c ↔ isInOctagon p (constructOctagon r c) :=
  sorry

end NUMINAMATH_CALUDE_locus_of_points_l4141_414169


namespace NUMINAMATH_CALUDE_spice_combinations_l4141_414181

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem spice_combinations : choose 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_spice_combinations_l4141_414181


namespace NUMINAMATH_CALUDE_tadpole_fish_ratio_l4141_414173

/-- The ratio of initial tadpoles to initial fish in a pond -/
theorem tadpole_fish_ratio :
  ∀ (initial_tadpoles : ℕ) (initial_fish : ℕ),
  initial_fish = 50 →
  ∃ (remaining_fish : ℕ) (remaining_tadpoles : ℕ),
  remaining_fish = initial_fish - 7 ∧
  remaining_tadpoles = initial_tadpoles / 2 ∧
  remaining_tadpoles = remaining_fish + 32 →
  (initial_tadpoles : ℚ) / initial_fish = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_tadpole_fish_ratio_l4141_414173


namespace NUMINAMATH_CALUDE_candy_distribution_l4141_414179

theorem candy_distribution (total_candies : ℕ) (num_friends : ℕ) (candies_per_friend : ℕ) :
  total_candies = 36 →
  num_friends = 9 →
  candies_per_friend = 4 →
  total_candies = num_friends * candies_per_friend :=
by
  sorry

#check candy_distribution

end NUMINAMATH_CALUDE_candy_distribution_l4141_414179


namespace NUMINAMATH_CALUDE_ellipse_m_value_l4141_414174

/-- An ellipse with equation x²/(10-m) + y²/(m-2) = 1, major axis along y-axis, and focal length 4 -/
structure Ellipse (m : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / (10 - m) + y^2 / (m - 2) = 1
  major_axis_y : m - 2 > 10 - m
  focal_length : ∃ (a b : ℝ), a^2 - b^2 = 16 ∧ a^2 = m - 2 ∧ b^2 = 10 - m

theorem ellipse_m_value (m : ℝ) (e : Ellipse m) : m = 8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l4141_414174


namespace NUMINAMATH_CALUDE_no_solution_condition_l4141_414189

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → (x / (x - 3) + (3 * a) / (3 - x) ≠ 2 * a)) ↔ (a = 1 ∨ a = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l4141_414189


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l4141_414141

theorem cylinder_volume_change (r h V : ℝ) : 
  V = π * r^2 * h → (π * (3*r)^2 * (4*h) = 36 * V) := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l4141_414141


namespace NUMINAMATH_CALUDE_family_heights_l4141_414118

/-- Given the heights of a family, prove the calculated heights of specific members -/
theorem family_heights (cary bill jan tim sara : ℝ) : 
  cary = 72 →
  bill = 0.8 * cary →
  jan = bill + 5 →
  tim = (bill + jan) / 2 - 4 →
  sara = 1.2 * ((cary + bill + jan + tim) / 4) →
  (bill = 57.6 ∧ jan = 62.6 ∧ tim = 56.1 ∧ sara = 74.49) := by
  sorry

end NUMINAMATH_CALUDE_family_heights_l4141_414118


namespace NUMINAMATH_CALUDE_quadratic_minimum_positive_l4141_414129

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 5

-- Theorem statement
theorem quadratic_minimum_positive :
  ∃ (x_min : ℝ), x_min > 0 ∧ 
  (∀ (x : ℝ), f x ≥ f x_min) ∧
  (∀ (ε : ℝ), ε > 0 → ∃ (x : ℝ), x ≠ x_min ∧ |x - x_min| < ε ∧ f x > f x_min) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_positive_l4141_414129


namespace NUMINAMATH_CALUDE_complex_fraction_opposite_parts_l4141_414164

theorem complex_fraction_opposite_parts (b : ℝ) : 
  let z₁ : ℂ := 1 + b * I
  let z₂ : ℂ := -2 + I
  (((z₁ / z₂).re = -(z₁ / z₂).im) → b = -1/3) ∧ 
  (b = -1/3 → (z₁ / z₂).re = -(z₁ / z₂).im) := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_opposite_parts_l4141_414164


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l4141_414190

/-- Given two concentric circles where a chord of length 120 units is tangent to the smaller circle
    with radius 40 units, the area between the circles is 3600π square units. -/
theorem area_between_concentric_circles :
  ∀ (r R : ℝ) (chord_length : ℝ),
  r = 40 →
  chord_length = 120 →
  chord_length^2 = 4 * R * r →
  (R^2 - r^2) * π = 3600 * π :=
by sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l4141_414190


namespace NUMINAMATH_CALUDE_number_ratio_problem_l4141_414175

theorem number_ratio_problem (x y z : ℝ) : 
  x + y + z = 110 →
  y = 30 →
  z = (1/3) * x →
  x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_problem_l4141_414175


namespace NUMINAMATH_CALUDE_louis_fabric_purchase_l4141_414187

theorem louis_fabric_purchase (fabric_cost_per_yard : ℝ) (pattern_cost : ℝ) (thread_cost_per_spool : ℝ) (total_spent : ℝ) : 
  fabric_cost_per_yard = 24 →
  pattern_cost = 15 →
  thread_cost_per_spool = 3 →
  total_spent = 141 →
  (total_spent - pattern_cost - 2 * thread_cost_per_spool) / fabric_cost_per_yard = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_louis_fabric_purchase_l4141_414187


namespace NUMINAMATH_CALUDE_johns_playing_days_l4141_414128

def beats_per_minute : ℕ := 200
def hours_per_day : ℕ := 2
def total_beats : ℕ := 72000

def minutes_per_hour : ℕ := 60
def minutes_per_day : ℕ := hours_per_day * minutes_per_hour
def beats_per_day : ℕ := beats_per_minute * minutes_per_day

theorem johns_playing_days :
  total_beats / beats_per_day = 3 :=
by sorry

end NUMINAMATH_CALUDE_johns_playing_days_l4141_414128


namespace NUMINAMATH_CALUDE_grid_sum_example_unique_transformed_grid_sum_constant_grid_sum_difference_count_grids_with_sum_104_l4141_414178

/-- Definition of a 2x2 grid of positive digits -/
structure Grid :=
  (a b c d : ℕ)
  (ha : 0 < a ∧ a < 10)
  (hb : 0 < b ∧ b < 10)
  (hc : 0 < c ∧ c < 10)
  (hd : 0 < d ∧ d < 10)

/-- The grid sum operation -/
def gridSum (g : Grid) : ℕ := 10*g.a + g.b + 10*g.c + g.d + 10*g.a + g.c + 10*g.b + g.d

/-- Theorem for part (a) -/
theorem grid_sum_example : 
  ∃ g : Grid, g.a = 7 ∧ g.b = 3 ∧ g.c = 2 ∧ g.d = 7 ∧ gridSum g = 209 := sorry

/-- Theorem for part (b) -/
theorem unique_transformed_grid_sum :
  ∃! x y : ℕ, ∀ b c : ℕ, 0 < b ∧ b < 9 ∧ 0 < c ∧ c < 10 →
    ∃ g1 g2 : Grid,
      g1.a = 5 ∧ g1.b = b ∧ g1.c = c ∧ g1.d = 7 ∧
      g2.a = x ∧ g2.b = b+1 ∧ g2.c = c-3 ∧ g2.d = y ∧
      gridSum g1 = gridSum g2 := sorry

/-- Theorem for part (c) -/
theorem constant_grid_sum_difference :
  ∃ k : ℤ, ∀ g : Grid,
    gridSum g - gridSum ⟨g.a+1, g.b-2, g.c-1, g.d+1, sorry, sorry, sorry, sorry⟩ = k := sorry

/-- Theorem for part (d) -/
theorem count_grids_with_sum_104 :
  ∃! ls : List Grid, (∀ g ∈ ls, gridSum g = 104) ∧ ls.length = 5 := sorry

end NUMINAMATH_CALUDE_grid_sum_example_unique_transformed_grid_sum_constant_grid_sum_difference_count_grids_with_sum_104_l4141_414178


namespace NUMINAMATH_CALUDE_alla_boris_meeting_l4141_414109

/-- Represents the meeting point of Alla and Boris on a street with streetlights. -/
def meeting_point (total_lights : ℕ) (alla_position : ℕ) (boris_position : ℕ) : ℕ :=
  let gaps_covered := (alla_position - 1) + (total_lights - boris_position)
  let total_gaps := total_lights - 1
  let alla_total_gaps := 3 * (alla_position - 1)
  1 + alla_total_gaps

/-- Theorem stating that Alla and Boris meet at the 163rd streetlight under given conditions. -/
theorem alla_boris_meeting :
  meeting_point 400 55 321 = 163 :=
sorry

end NUMINAMATH_CALUDE_alla_boris_meeting_l4141_414109


namespace NUMINAMATH_CALUDE_sqrt_twelve_equals_two_sqrt_three_l4141_414123

theorem sqrt_twelve_equals_two_sqrt_three : Real.sqrt 12 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_equals_two_sqrt_three_l4141_414123


namespace NUMINAMATH_CALUDE_problem_hexagon_area_l4141_414104

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- A hexagon defined by six points -/
structure Hexagon where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point
  p5 : Point
  p6 : Point

/-- The area of a hexagon -/
def hexagonArea (h : Hexagon) : ℝ := sorry

/-- The specific hexagon in the problem -/
def problemHexagon : Hexagon := {
  p1 := { x := 0, y := 0 },
  p2 := { x := 2, y := 4 },
  p3 := { x := 6, y := 4 },
  p4 := { x := 8, y := 0 },
  p5 := { x := 6, y := -4 },
  p6 := { x := 2, y := -4 }
}

/-- Theorem stating that the area of the problem hexagon is 16 square units -/
theorem problem_hexagon_area : hexagonArea problemHexagon = 16 := by sorry

end NUMINAMATH_CALUDE_problem_hexagon_area_l4141_414104


namespace NUMINAMATH_CALUDE_add_preserves_inequality_l4141_414135

theorem add_preserves_inequality (a b : ℝ) (h : a > b) : a + 2 > b + 2 := by
  sorry

end NUMINAMATH_CALUDE_add_preserves_inequality_l4141_414135


namespace NUMINAMATH_CALUDE_lcm_gcf_product_24_60_l4141_414115

theorem lcm_gcf_product_24_60 : Nat.lcm 24 60 * Nat.gcd 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_product_24_60_l4141_414115


namespace NUMINAMATH_CALUDE_sphere_radius_range_l4141_414157

/-- Represents the parabola x^2 = 2y where 0 ≤ y ≤ 20 -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 2*p.2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 20}

/-- A sphere touching the bottom of the parabola with its center on the y-axis -/
structure Sphere :=
  (center : ℝ)
  (radius : ℝ)
  (touches_bottom : radius = center)
  (inside_parabola : ∀ x y, (x, y) ∈ Parabola → x^2 + (y - center)^2 ≥ radius^2)

/-- The theorem stating the range of the sphere's radius -/
theorem sphere_radius_range (s : Sphere) : 0 < s.radius ∧ s.radius ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_sphere_radius_range_l4141_414157


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l4141_414191

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (4, x) (-4, 4) → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l4141_414191


namespace NUMINAMATH_CALUDE_train_speed_l4141_414186

/-- The speed of a train given its length and time to cross a post -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 250.02) (h2 : time = 22.5) :
  (length * 3600) / (time * 1000) = 40.0032 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l4141_414186


namespace NUMINAMATH_CALUDE_intersection_M_and_naturals_l4141_414103

def M : Set ℝ := {x | (x + 2) / (x - 1) ≤ 0}

theorem intersection_M_and_naturals :
  M ∩ Set.range (Nat.cast : ℕ → ℝ) = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_and_naturals_l4141_414103


namespace NUMINAMATH_CALUDE_range_of_a_l4141_414152

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x, x^2 - (a-1)*x + 1 > 0

def q (a : ℝ) : Prop := ∀ x y, x < y → (a+1)^x < (a+1)^y

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) →
  ((-1 < a ∧ a ≤ 0) ∨ (a ≥ 3)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4141_414152


namespace NUMINAMATH_CALUDE_parabola_through_point_l4141_414117

-- Define a parabola type
structure Parabola where
  -- A parabola is defined by its equation
  equation : ℝ → ℝ → Prop

-- Define the condition that a parabola passes through a point
def passes_through (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

-- Define the two possible standard forms of a parabola
def vertical_parabola (a : ℝ) : Parabola :=
  ⟨λ x y => y^2 = 4*a*x⟩

def horizontal_parabola (b : ℝ) : Parabola :=
  ⟨λ x y => x^2 = 4*b*y⟩

-- The theorem to be proved
theorem parabola_through_point :
  ∃ (p : Parabola), passes_through p (-2) 4 ∧
    ((∃ a : ℝ, p = vertical_parabola a ∧ a = -2) ∨
     (∃ b : ℝ, p = horizontal_parabola b ∧ b = 1/4)) :=
sorry

end NUMINAMATH_CALUDE_parabola_through_point_l4141_414117


namespace NUMINAMATH_CALUDE_subset_sum_exists_l4141_414145

theorem subset_sum_exists (nums : List ℕ) : 
  nums.length = 100 ∧ 
  (∀ n ∈ nums, n < 100) ∧ 
  nums.sum = 200 → 
  ∃ subset : List ℕ, subset ⊆ nums ∧ subset.sum = 100 := by
sorry

end NUMINAMATH_CALUDE_subset_sum_exists_l4141_414145


namespace NUMINAMATH_CALUDE_distance_between_points_l4141_414195

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3, 4)
  let p2 : ℝ × ℝ := (-6, -1)
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = Real.sqrt 106 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l4141_414195


namespace NUMINAMATH_CALUDE_guys_age_proof_l4141_414194

theorem guys_age_proof :
  ∃ (age : ℕ), 
    (((age + 8) * 8 - (age - 8) * 8) / 2 = age) ∧ 
    (age = 64) := by
  sorry

end NUMINAMATH_CALUDE_guys_age_proof_l4141_414194


namespace NUMINAMATH_CALUDE_six_number_list_product_l4141_414199

theorem six_number_list_product (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (h_order : a₁ ≤ a₂ ∧ a₂ ≤ a₃ ∧ a₃ ≤ a₄ ∧ a₄ ≤ a₅ ∧ a₅ ≤ a₆)
  (h_remove_largest : (a₁ + a₂ + a₃ + a₄ + a₅) / 5 = (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 - 1)
  (h_remove_smallest : (a₂ + a₃ + a₄ + a₅ + a₆) / 5 = (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 + 1)
  (h_remove_both : (a₂ + a₃ + a₄ + a₅) / 4 = 20) :
  a₁ * a₆ = 375 := by
sorry

end NUMINAMATH_CALUDE_six_number_list_product_l4141_414199


namespace NUMINAMATH_CALUDE_x_squared_minus_4x_geq_m_l4141_414162

theorem x_squared_minus_4x_geq_m (m : ℝ) : 
  (∀ x ∈ Set.Icc 3 4, x^2 - 4*x ≥ m) → m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_4x_geq_m_l4141_414162


namespace NUMINAMATH_CALUDE_friends_contribution_impossibility_l4141_414163

theorem friends_contribution_impossibility (a b c d e : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → e ≥ 0 →
  a + b + c + d + e > 0 →
  a + b < (a + b + c + d + e) / 3 →
  b + c < (a + b + c + d + e) / 3 →
  c + d < (a + b + c + d + e) / 3 →
  d + e < (a + b + c + d + e) / 3 →
  e + a < (a + b + c + d + e) / 3 →
  False :=
by sorry

end NUMINAMATH_CALUDE_friends_contribution_impossibility_l4141_414163


namespace NUMINAMATH_CALUDE_four_thirds_of_nine_halves_l4141_414110

theorem four_thirds_of_nine_halves : (4 / 3 : ℚ) * (9 / 2 : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_four_thirds_of_nine_halves_l4141_414110


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_l4141_414148

def point_symmetry_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

theorem symmetric_point_y_axis :
  let P : ℝ × ℝ := (2, 3)
  point_symmetry_y_axis P = (-2, 3) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_l4141_414148


namespace NUMINAMATH_CALUDE_chicken_price_chicken_price_is_8_l4141_414176

/-- Calculates the price of a chicken given the conditions of the farmer's sales. -/
theorem chicken_price : ℝ → Prop :=
  fun price =>
    let duck_price := 10
    let num_ducks := 2
    let num_chickens := 5
    let total_earnings := duck_price * num_ducks + price * num_chickens
    let wheelbarrow_cost := total_earnings / 2
    let wheelbarrow_sale := wheelbarrow_cost * 2
    let additional_earnings := 60
    wheelbarrow_sale - wheelbarrow_cost = additional_earnings →
    price = 8

/-- The price of a chicken is $8. -/
theorem chicken_price_is_8 : chicken_price 8 := by
  sorry

end NUMINAMATH_CALUDE_chicken_price_chicken_price_is_8_l4141_414176


namespace NUMINAMATH_CALUDE_sara_picked_24_more_peaches_l4141_414182

/-- The number of additional peaches Sara picked at the orchard -/
def additional_peaches (initial_peaches total_peaches : ℝ) : ℝ :=
  total_peaches - initial_peaches

/-- Theorem: Sara picked 24 additional peaches at the orchard -/
theorem sara_picked_24_more_peaches (initial_peaches total_peaches : ℝ)
  (h1 : initial_peaches = 61.0)
  (h2 : total_peaches = 85.0) :
  additional_peaches initial_peaches total_peaches = 24 := by
  sorry

end NUMINAMATH_CALUDE_sara_picked_24_more_peaches_l4141_414182


namespace NUMINAMATH_CALUDE_sarah_test_performance_l4141_414151

theorem sarah_test_performance :
  let test1_questions : ℕ := 30
  let test2_questions : ℕ := 20
  let test3_questions : ℕ := 50
  let test1_correct_rate : ℚ := 85 / 100
  let test2_correct_rate : ℚ := 75 / 100
  let test3_correct_rate : ℚ := 90 / 100
  let calculation_mistakes : ℕ := 3
  let total_questions := test1_questions + test2_questions + test3_questions
  let correct_before_mistakes := 
    (test1_correct_rate * test1_questions).ceil +
    (test2_correct_rate * test2_questions).floor +
    (test3_correct_rate * test3_questions).floor
  let correct_after_mistakes := correct_before_mistakes - calculation_mistakes
  (correct_after_mistakes : ℚ) / total_questions = 83 / 100 :=
by sorry

end NUMINAMATH_CALUDE_sarah_test_performance_l4141_414151


namespace NUMINAMATH_CALUDE_intersection_implies_b_range_l4141_414142

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m*p.1 + b}

-- State the theorem
theorem intersection_implies_b_range :
  (∀ m : ℝ, ∃ p : ℝ × ℝ, p ∈ M ∩ N m b) →
  b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_b_range_l4141_414142


namespace NUMINAMATH_CALUDE_picnic_men_count_l4141_414198

/-- Represents the number of people at a picnic -/
structure PicnicAttendance where
  total : ℕ
  men : ℕ
  women : ℕ
  adults : ℕ
  children : ℕ

/-- Conditions for the picnic attendance -/
def picnicConditions (p : PicnicAttendance) : Prop :=
  p.total = 200 ∧
  p.men = p.women + 20 ∧
  p.adults = p.children + 20 ∧
  p.adults = p.men + p.women ∧
  p.total = p.men + p.women + p.children

/-- Theorem: Given the conditions, the number of men at the picnic is 65 -/
theorem picnic_men_count (p : PicnicAttendance) :
  picnicConditions p → p.men = 65 := by
  sorry

end NUMINAMATH_CALUDE_picnic_men_count_l4141_414198


namespace NUMINAMATH_CALUDE_max_xy_value_l4141_414132

theorem max_xy_value (a b c x y : ℝ) :
  a * x + b * y + 2 * c = 0 →
  c ≠ 0 →
  a * b - c^2 ≥ 0 →
  x * y ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_value_l4141_414132


namespace NUMINAMATH_CALUDE_monotonicity_of_f_l4141_414130

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x / (x - 1)

theorem monotonicity_of_f (a : ℝ) (h_a : a ≠ 0) :
  (∀ x₁ x₂ : ℝ, -1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → 
    (a > 0 → f a x₁ > f a x₂) ∧ 
    (a < 0 → f a x₁ < f a x₂)) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_of_f_l4141_414130


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l4141_414149

-- Define a right triangle with two acute angles
structure RightTriangle where
  angle1 : ℝ
  angle2 : ℝ
  is_right_triangle : angle1 + angle2 = 90

-- Define the condition that the ratio of the two acute angles is 3:1
def angle_ratio (t : RightTriangle) : Prop :=
  t.angle1 / t.angle2 = 3

-- Theorem statement
theorem right_triangle_acute_angles 
  (t : RightTriangle) 
  (h : angle_ratio t) : 
  (t.angle1 = 67.5 ∧ t.angle2 = 22.5) ∨ (t.angle1 = 22.5 ∧ t.angle2 = 67.5) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l4141_414149


namespace NUMINAMATH_CALUDE_tinas_career_win_loss_difference_l4141_414139

/-- Represents Tina's boxing career -/
structure BoxingCareer where
  initial_wins : ℕ
  additional_wins_before_first_loss : ℕ
  wins_doubled : Bool

/-- Calculates the total number of wins in Tina's career -/
def total_wins (career : BoxingCareer) : ℕ :=
  let wins_before_doubling := career.initial_wins + career.additional_wins_before_first_loss
  if career.wins_doubled then
    2 * wins_before_doubling
  else
    wins_before_doubling

/-- Calculates the total number of losses in Tina's career -/
def total_losses (career : BoxingCareer) : ℕ :=
  if career.wins_doubled then 2 else 1

/-- Theorem stating the difference between wins and losses in Tina's career -/
theorem tinas_career_win_loss_difference :
  ∀ (career : BoxingCareer),
    career.initial_wins = 10 →
    career.additional_wins_before_first_loss = 5 →
    career.wins_doubled = true →
    total_wins career - total_losses career = 28 := by
  sorry

end NUMINAMATH_CALUDE_tinas_career_win_loss_difference_l4141_414139


namespace NUMINAMATH_CALUDE_left_handed_rock_lovers_l4141_414111

theorem left_handed_rock_lovers (total : ℕ) (left_handed : ℕ) (rock_lovers : ℕ) (right_handed_rock_dislikers : ℕ) :
  total = 30 →
  left_handed = 14 →
  rock_lovers = 20 →
  right_handed_rock_dislikers = 5 →
  ∃ (x : ℕ),
    x = left_handed + rock_lovers - total + right_handed_rock_dislikers ∧
    x = 9 :=
by sorry

end NUMINAMATH_CALUDE_left_handed_rock_lovers_l4141_414111


namespace NUMINAMATH_CALUDE_unique_solution_value_l4141_414137

theorem unique_solution_value (p : ℝ) : 
  (∃! x : ℝ, x ≠ 0 ∧ (1 : ℝ) / (3 * x) = (p - x) / 4) ↔ p = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_value_l4141_414137


namespace NUMINAMATH_CALUDE_directrix_of_given_parabola_l4141_414184

/-- A parabola in the xy-plane -/
structure Parabola where
  /-- The equation of the parabola in the form y = ax^2 + bx + c -/
  equation : ℝ → ℝ → Prop

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ → ℝ → Prop :=
  sorry

/-- The given parabola y = -3x^2 + 6x - 5 -/
def given_parabola : Parabola :=
  { equation := λ x y => y = -3 * x^2 + 6 * x - 5 }

theorem directrix_of_given_parabola :
  directrix given_parabola = λ x y => y = -35/18 :=
sorry

end NUMINAMATH_CALUDE_directrix_of_given_parabola_l4141_414184


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l4141_414112

theorem correct_quadratic_equation :
  ∀ (b c : ℝ),
  (∃ (b' c' : ℝ), (5 : ℝ) * (1 : ℝ) = c' ∧ 5 + 1 = -b) →
  (∃ (b'' : ℝ), (-7 : ℝ) * (-2 : ℝ) = c) →
  (x^2 + b*x + c = 0) = (x^2 - 6*x + 14 = 0) :=
by sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l4141_414112
