import Mathlib

namespace NUMINAMATH_CALUDE_club_members_after_four_years_l4088_408810

def club_members (n : ℕ) : ℕ :=
  match n with
  | 0 => 20
  | k + 1 => 3 * club_members k - 10

theorem club_members_after_four_years :
  club_members 4 = 1220 := by
  sorry

end NUMINAMATH_CALUDE_club_members_after_four_years_l4088_408810


namespace NUMINAMATH_CALUDE_injective_impl_neq_injective_impl_unique_preimage_l4088_408806

variable {A B : Type*} (f : A → B)

/-- Definition of injective function -/
def Injective (f : A → B) : Prop :=
  ∀ x₁ x₂ : A, f x₁ = f x₂ → x₁ = x₂

theorem injective_impl_neq (hf : Injective f) :
    ∀ x₁ x₂ : A, x₁ ≠ x₂ → f x₁ ≠ f x₂ := by sorry

theorem injective_impl_unique_preimage (hf : Injective f) :
    ∀ b : B, ∃! a : A, f a = b := by sorry

end NUMINAMATH_CALUDE_injective_impl_neq_injective_impl_unique_preimage_l4088_408806


namespace NUMINAMATH_CALUDE_tree_height_increase_l4088_408833

/-- Proves that the annual increase in tree height is 2 feet given the initial conditions --/
theorem tree_height_increase (initial_height : ℝ) (annual_increase : ℝ) : 
  initial_height = 4 →
  initial_height + 6 * annual_increase = (initial_height + 4 * annual_increase) * (4/3) →
  annual_increase = 2 := by
sorry

end NUMINAMATH_CALUDE_tree_height_increase_l4088_408833


namespace NUMINAMATH_CALUDE_arson_sentence_calculation_l4088_408819

/-- Calculates the sentence for each arson count given the total sentence and other crime details. -/
theorem arson_sentence_calculation (total_sentence : ℕ) (arson_counts : ℕ) (burglary_charges : ℕ) 
  (burglary_sentence : ℕ) (petty_larceny_ratio : ℕ) (petty_larceny_sentence_fraction : ℚ) :
  total_sentence = 216 →
  arson_counts = 3 →
  burglary_charges = 2 →
  burglary_sentence = 18 →
  petty_larceny_ratio = 6 →
  petty_larceny_sentence_fraction = 1/3 →
  ∃ (arson_sentence : ℕ),
    arson_sentence = 36 ∧
    total_sentence = arson_counts * arson_sentence + 
                     burglary_charges * burglary_sentence +
                     petty_larceny_ratio * burglary_charges * (petty_larceny_sentence_fraction * burglary_sentence) :=
by
  sorry


end NUMINAMATH_CALUDE_arson_sentence_calculation_l4088_408819


namespace NUMINAMATH_CALUDE_original_bikes_count_l4088_408808

/-- Represents the number of bikes added per week -/
def bikes_added_per_week : ℕ := 3

/-- Represents the number of weeks in a month -/
def weeks_in_month : ℕ := 4

/-- Represents the number of bikes sold in a month -/
def bikes_sold : ℕ := 18

/-- Represents the number of bikes in stock after a month -/
def bikes_in_stock : ℕ := 45

/-- Theorem stating that the original number of bikes is 51 -/
theorem original_bikes_count : 
  ∃ (original : ℕ), 
    original + (bikes_added_per_week * weeks_in_month) - bikes_sold = bikes_in_stock ∧ 
    original = 51 := by
  sorry

end NUMINAMATH_CALUDE_original_bikes_count_l4088_408808


namespace NUMINAMATH_CALUDE_room_length_proof_l4088_408823

/-- Given the cost of carpeting, carpet width, cost per meter, and room breadth, 
    prove the length of the room. -/
theorem room_length_proof 
  (total_cost : ℝ) 
  (carpet_width : ℝ) 
  (cost_per_meter : ℝ) 
  (room_breadth : ℝ) 
  (h1 : total_cost = 36)
  (h2 : carpet_width = 0.75)
  (h3 : cost_per_meter = 0.30)
  (h4 : room_breadth = 6) :
  ∃ (room_length : ℝ), room_length = 15 := by
  sorry

end NUMINAMATH_CALUDE_room_length_proof_l4088_408823


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l4088_408865

/-- Represents a teacher with their name and number of created questions -/
structure Teacher where
  name : String
  questions : ℕ

/-- Represents the result of stratified sampling -/
structure SamplingResult where
  wu : ℕ
  wang : ℕ
  zhang : ℕ

/-- Calculates the number of questions selected for each teacher in stratified sampling -/
def stratifiedSampling (teachers : List Teacher) (totalSamples : ℕ) : SamplingResult :=
  sorry

/-- Calculates the probability of selecting at least one question from a specific teacher -/
def probabilityAtLeastOne (samplingResult : SamplingResult) (teacherQuestions : ℕ) (selectionSize : ℕ) : ℚ :=
  sorry

theorem stratified_sampling_theorem (wu wang zhang : Teacher) (h1 : wu.questions = 350) (h2 : wang.questions = 700) (h3 : zhang.questions = 1050) :
  let teachers := [wu, wang, zhang]
  let result := stratifiedSampling teachers 6
  result.wu = 1 ∧ result.wang = 2 ∧ result.zhang = 3 ∧
  probabilityAtLeastOne result result.wang 2 = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l4088_408865


namespace NUMINAMATH_CALUDE_doctor_lawyer_ratio_l4088_408801

theorem doctor_lawyer_ratio (total : ℕ) (avg_all avg_doc avg_law : ℚ) 
  (h_total : total = 50)
  (h_avg_all : avg_all = 50)
  (h_avg_doc : avg_doc = 45)
  (h_avg_law : avg_law = 60) :
  ∃ (num_doc num_law : ℕ),
    num_doc + num_law = total ∧
    (avg_doc * num_doc + avg_law * num_law : ℚ) / total = avg_all ∧
    2 * num_law = num_doc :=
by sorry

end NUMINAMATH_CALUDE_doctor_lawyer_ratio_l4088_408801


namespace NUMINAMATH_CALUDE_alfred_incurred_loss_no_gain_percent_l4088_408869

/-- Represents the financial transaction of buying and selling a scooter --/
structure ScooterTransaction where
  purchase_price : ℝ
  repair_cost : ℝ
  taxes_and_fees : ℝ
  accessories_cost : ℝ
  selling_price : ℝ

/-- Calculates the total cost of the scooter transaction --/
def total_cost (t : ScooterTransaction) : ℝ :=
  t.purchase_price + t.repair_cost + t.taxes_and_fees + t.accessories_cost

/-- Theorem stating that Alfred incurred a loss on the scooter transaction --/
theorem alfred_incurred_loss (t : ScooterTransaction) 
  (h1 : t.purchase_price = 4700)
  (h2 : t.repair_cost = 800)
  (h3 : t.taxes_and_fees = 300)
  (h4 : t.accessories_cost = 250)
  (h5 : t.selling_price = 6000) :
  total_cost t > t.selling_price := by
  sorry

/-- Corollary stating that there is no gain percent as Alfred incurred a loss --/
theorem no_gain_percent (t : ScooterTransaction) 
  (h1 : t.purchase_price = 4700)
  (h2 : t.repair_cost = 800)
  (h3 : t.taxes_and_fees = 300)
  (h4 : t.accessories_cost = 250)
  (h5 : t.selling_price = 6000) :
  ¬∃ (gain_percent : ℝ), gain_percent > 0 ∧ t.selling_price = total_cost t * (1 + gain_percent / 100) := by
  sorry

end NUMINAMATH_CALUDE_alfred_incurred_loss_no_gain_percent_l4088_408869


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l4088_408853

theorem sum_of_solutions_quadratic (x : ℝ) : 
  let a : ℝ := -48
  let b : ℝ := 96
  let c : ℝ := -72
  let sum_of_roots := -b / a
  (a * x^2 + b * x + c = 0) → sum_of_roots = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l4088_408853


namespace NUMINAMATH_CALUDE_lattice_point_in_triangle_l4088_408884

/-- A point in a 2D integer lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A convex quadrilateral in a 2D integer lattice -/
structure ConvexLatticeQuadrilateral where
  P : LatticePoint
  Q : LatticePoint
  R : LatticePoint
  S : LatticePoint
  is_convex : Bool  -- Assume this is true for a convex quadrilateral

/-- The angle between two vectors -/
def angle (v1 v2 : LatticePoint → LatticePoint) : ℝ := sorry

/-- Check if a point is inside or on the boundary of a triangle -/
def is_in_triangle (X P Q E : LatticePoint) : Prop := sorry

theorem lattice_point_in_triangle
  (PQRS : ConvexLatticeQuadrilateral)
  (E : LatticePoint)
  (h_diagonals_intersect : E = sorry)  -- E is the intersection of diagonals
  (h_angle_sum : angle (λ p => PQRS.P) (λ p => PQRS.Q) < 180) :
  ∃ X : LatticePoint, X ≠ PQRS.P ∧ X ≠ PQRS.Q ∧ is_in_triangle X PQRS.P PQRS.Q E :=
sorry

end NUMINAMATH_CALUDE_lattice_point_in_triangle_l4088_408884


namespace NUMINAMATH_CALUDE_point_quadrant_relation_l4088_408848

/-- If P(a,b) is in the second quadrant, then Q(-b,a-3) is in the third quadrant -/
theorem point_quadrant_relation (a b : ℝ) : 
  (a < 0 ∧ b > 0) → (-b < 0 ∧ a - 3 < 0) := by
  sorry

end NUMINAMATH_CALUDE_point_quadrant_relation_l4088_408848


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l4088_408811

theorem algebraic_expression_equality (x y : ℝ) (h : x - 2*y = -2) : 
  9 - 2*x + 4*y = 13 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l4088_408811


namespace NUMINAMATH_CALUDE_tan_periodicity_l4088_408891

theorem tan_periodicity (m : ℤ) :
  -180 < m ∧ m < 180 ∧ Real.tan (m * π / 180) = Real.tan (1230 * π / 180) →
  m = -30 :=
by sorry

end NUMINAMATH_CALUDE_tan_periodicity_l4088_408891


namespace NUMINAMATH_CALUDE_marys_animal_count_l4088_408862

/-- The number of animals Mary thought were in the petting zoo -/
def marys_count (actual_count : ℕ) (double_counted : ℕ) (forgotten : ℕ) : ℕ :=
  actual_count + double_counted - forgotten

/-- Theorem stating that Mary thought there were 60 animals in the petting zoo -/
theorem marys_animal_count :
  marys_count 56 7 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_marys_animal_count_l4088_408862


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l4088_408879

/-- Represents a batsman's score data -/
structure BatsmanScore where
  initialAverage : ℝ
  inningsPlayed : ℕ
  newInningScore : ℝ
  averageIncrease : ℝ

/-- Theorem: If a batsman's average increases by 5 after scoring 100 in the 11th inning, 
    then his new average is 50 -/
theorem batsman_average_theorem (b : BatsmanScore) 
  (h1 : b.inningsPlayed = 10)
  (h2 : b.newInningScore = 100)
  (h3 : b.averageIncrease = 5)
  : b.initialAverage + b.averageIncrease = 50 := by
  sorry

#check batsman_average_theorem

end NUMINAMATH_CALUDE_batsman_average_theorem_l4088_408879


namespace NUMINAMATH_CALUDE_coronavirus_diameter_scientific_notation_l4088_408826

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem coronavirus_diameter_scientific_notation :
  toScientificNotation 0.00000012 = ScientificNotation.mk 1.2 (-7) (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_coronavirus_diameter_scientific_notation_l4088_408826


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l4088_408893

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (1 / (x + 3))
  else if x < -3 then Int.floor (1 / (x + 3))
  else 0  -- Arbitrary value for x = -3, as g is undefined there

theorem zero_not_in_range_of_g :
  ¬ ∃ (x : ℝ), g x = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l4088_408893


namespace NUMINAMATH_CALUDE_symmetric_point_simplification_l4088_408834

theorem symmetric_point_simplification (x : ℝ) :
  (∃ P : ℝ × ℝ, P = (x + 1, 2 * x - 1) ∧ 
   (∃ P' : ℝ × ℝ, P' = (-x - 1, -2 * x + 1) ∧ 
    P'.1 > 0 ∧ P'.2 > 0)) →
  |x - 3| - |1 - x| = 2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_point_simplification_l4088_408834


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l4088_408818

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) ∧
  (∃ x : ℝ, |x - 1| < 2 ∧ x * (x - 3) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l4088_408818


namespace NUMINAMATH_CALUDE_new_person_weight_l4088_408856

/-- Given a group of 8 persons where replacing one person weighing 35 kg
    with a new person increases the average weight by 5 kg,
    prove that the weight of the new person is 75 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 5 →
  replaced_weight = 35 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 75 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l4088_408856


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l4088_408851

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 4 / 3)
  (hy : y / z = 3 / 2)
  (hz : z / x = 1 / 6) :
  w / y = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l4088_408851


namespace NUMINAMATH_CALUDE_range_of_F_l4088_408815

-- Define the function F
def F (x : ℝ) : ℝ := |2*x + 2| - |2*x - 2|

-- State the theorem about the range of F
theorem range_of_F :
  ∀ y : ℝ, (∃ x : ℝ, F x = y) ↔ y ∈ Set.Icc (-4) 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_F_l4088_408815


namespace NUMINAMATH_CALUDE_polygon_sides_l4088_408805

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) → 
  (180 * (n - 2) = 5 * 360) → 
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l4088_408805


namespace NUMINAMATH_CALUDE_two_roots_condition_l4088_408857

-- Define the quadratic equation
def quadratic_equation (x a : ℝ) : Prop := x^2 - 2*x + a = 0

-- Define the condition for having two distinct real roots
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation x a ∧ quadratic_equation y a

-- Statement of the theorem
theorem two_roots_condition (a : ℝ) :
  has_two_distinct_roots a ↔ a < 1 :=
sorry

end NUMINAMATH_CALUDE_two_roots_condition_l4088_408857


namespace NUMINAMATH_CALUDE_dimes_in_jar_l4088_408898

/-- Represents the number of coins of each type in the jar -/
structure CoinCount where
  dimes : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in cents -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.dimes * 10 + coins.quarters * 25

/-- Theorem stating that given the conditions, there are 15 dimes in the jar -/
theorem dimes_in_jar : ∃ (coins : CoinCount),
  coins.dimes = 3 * coins.quarters / 2 ∧
  totalValue coins = 400 ∧
  coins.dimes = 15 := by
  sorry

end NUMINAMATH_CALUDE_dimes_in_jar_l4088_408898


namespace NUMINAMATH_CALUDE_max_performances_l4088_408847

/-- Represents a performance in the theater festival -/
structure Performance :=
  (students : Finset ℕ)
  (size_eq_six : students.card = 6)

/-- The theater festival -/
structure TheaterFestival :=
  (num_students : ℕ)
  (num_students_eq_twelve : num_students = 12)
  (performances : Finset Performance)
  (common_students : Performance → Performance → Finset ℕ)
  (common_students_le_two : ∀ p1 p2 : Performance, p1 ≠ p2 → (common_students p1 p2).card ≤ 2)

/-- The theorem stating the maximum number of performances -/
theorem max_performances (festival : TheaterFestival) : 
  festival.performances.card ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_performances_l4088_408847


namespace NUMINAMATH_CALUDE_smallest_x_sqrt_3x_eq_5x_l4088_408849

theorem smallest_x_sqrt_3x_eq_5x (x : ℝ) :
  x ≥ 0 ∧ Real.sqrt (3 * x) = 5 * x → x = 0 := by sorry

end NUMINAMATH_CALUDE_smallest_x_sqrt_3x_eq_5x_l4088_408849


namespace NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l4088_408807

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n * (a 1 + a n)) / 2

/-- The main theorem -/
theorem arithmetic_sequence_m_value (seq : ArithmeticSequence) (m : ℕ) 
    (h1 : seq.S (m - 1) = -2)
    (h2 : seq.S m = 0)
    (h3 : seq.S (m + 1) = 3) :
  m = 5 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l4088_408807


namespace NUMINAMATH_CALUDE_remy_water_usage_l4088_408871

theorem remy_water_usage (roman : ℕ) (remy : ℕ) : 
  remy = 3 * roman + 1 →
  roman + remy = 33 →
  remy = 25 := by
sorry

end NUMINAMATH_CALUDE_remy_water_usage_l4088_408871


namespace NUMINAMATH_CALUDE_derivative_at_zero_l4088_408845

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 
    Real.arcsin (x^2 * Real.cos (1 / (9 * x))) + (2/3) * x
  else 
    0

-- State the theorem
theorem derivative_at_zero (f : ℝ → ℝ) : 
  (deriv f) 0 = 2/3 := by sorry

end NUMINAMATH_CALUDE_derivative_at_zero_l4088_408845


namespace NUMINAMATH_CALUDE_budi_can_win_l4088_408816

/-- The set of numbers from which players choose -/
def S : Finset ℕ := Finset.range 30

/-- The total number of balls in the game -/
def totalBalls : ℕ := 2015

/-- Astri's chosen numbers -/
structure AstriChoice where
  a : ℕ
  b : ℕ
  a_in_S : a ∈ S
  b_in_S : b ∈ S
  a_ne_b : a ≠ b

/-- Budi's chosen numbers -/
structure BudiChoice (ac : AstriChoice) where
  c : ℕ
  d : ℕ
  c_in_S : c ∈ S
  d_in_S : d ∈ S
  c_ne_d : c ≠ d
  c_ne_a : c ≠ ac.a
  c_ne_b : c ≠ ac.b
  d_ne_a : d ≠ ac.a
  d_ne_b : d ≠ ac.b

/-- The game state -/
structure GameState where
  ballsLeft : ℕ
  astriTurn : Bool

/-- A winning strategy for Budi -/
def isWinningStrategy (ac : AstriChoice) (bc : BudiChoice ac) (strategy : GameState → ℕ) : Prop :=
  ∀ (gs : GameState), 
    (gs.astriTurn ∧ gs.ballsLeft < ac.a ∧ gs.ballsLeft < ac.b) ∨
    (¬gs.astriTurn ∧ 
      ((strategy gs = bc.c ∧ gs.ballsLeft ≥ bc.c) ∨ 
       (strategy gs = bc.d ∧ gs.ballsLeft ≥ bc.d)))

/-- The main theorem -/
theorem budi_can_win : 
  ∀ (ac : AstriChoice), ∃ (bc : BudiChoice ac) (strategy : GameState → ℕ), 
    isWinningStrategy ac bc strategy :=
sorry

end NUMINAMATH_CALUDE_budi_can_win_l4088_408816


namespace NUMINAMATH_CALUDE_relative_complement_M_N_l4088_408820

def M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℤ := {1, 2}

theorem relative_complement_M_N : (M \ N) = {-1, 0, 3} := by
  sorry

end NUMINAMATH_CALUDE_relative_complement_M_N_l4088_408820


namespace NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l4088_408836

theorem pythagorean_triple_divisibility (x y z : ℕ+) (h : x^2 + y^2 = z^2) :
  (3 ∣ x ∨ 3 ∣ y) ∧ (5 ∣ x ∨ 5 ∣ y ∨ 5 ∣ z) := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l4088_408836


namespace NUMINAMATH_CALUDE_chocolate_bars_l4088_408817

theorem chocolate_bars (cost_per_bar : ℕ) (remaining_bars : ℕ) (revenue : ℕ) :
  cost_per_bar = 4 →
  remaining_bars = 3 →
  revenue = 20 →
  ∃ total_bars : ℕ, total_bars = 8 ∧ cost_per_bar * (total_bars - remaining_bars) = revenue :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_l4088_408817


namespace NUMINAMATH_CALUDE_faster_train_speed_l4088_408877

/-- Proves that given two trains of specified lengths running in opposite directions,
    with a given crossing time and speed of the slower train, the speed of the faster train
    is as calculated. -/
theorem faster_train_speed
  (length_train1 : ℝ)
  (length_train2 : ℝ)
  (crossing_time : ℝ)
  (slower_train_speed : ℝ)
  (h1 : length_train1 = 180)
  (h2 : length_train2 = 360)
  (h3 : crossing_time = 21.598272138228943)
  (h4 : slower_train_speed = 30) :
  ∃ (faster_train_speed : ℝ),
    faster_train_speed = 60 ∧
    (length_train1 + length_train2) / crossing_time * 3.6 = slower_train_speed + faster_train_speed :=
by sorry

end NUMINAMATH_CALUDE_faster_train_speed_l4088_408877


namespace NUMINAMATH_CALUDE_quadratic_factorization_l4088_408876

theorem quadratic_factorization (C D : ℤ) :
  (∀ y : ℚ, 15 * y^2 - 56 * y + 48 = (C * y - 16) * (D * y - 3)) →
  C * D + C = 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l4088_408876


namespace NUMINAMATH_CALUDE_fraction_of_fraction_l4088_408896

theorem fraction_of_fraction : (5 / 12) / (3 / 4) = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_l4088_408896


namespace NUMINAMATH_CALUDE_angle_measure_problem_l4088_408828

theorem angle_measure_problem (x : ℝ) : 
  (180 - x = 3 * (90 - x) - 60) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_problem_l4088_408828


namespace NUMINAMATH_CALUDE_simple_interest_time_l4088_408861

/-- Proves that the time for simple interest is 3 years under given conditions -/
theorem simple_interest_time (P : ℝ) (r : ℝ) (compound_principal : ℝ) (compound_time : ℝ) : 
  P = 1400.0000000000014 →
  r = 0.10 →
  compound_principal = 4000 →
  compound_time = 2 →
  P * r * 3 = (compound_principal * ((1 + r) ^ compound_time - 1)) / 2 →
  3 = (((compound_principal * ((1 + r) ^ compound_time - 1)) / 2) / (P * r)) :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_time_l4088_408861


namespace NUMINAMATH_CALUDE_cuboid_height_theorem_l4088_408889

/-- Represents a cuboid (rectangular box) -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.length * c.width * c.height

/-- Theorem: A cuboid with volume 315, width 9, and length 7 has height 5 -/
theorem cuboid_height_theorem (c : Cuboid) 
  (h_volume : volume c = 315)
  (h_width : c.width = 9)
  (h_length : c.length = 7) : 
  c.height = 5 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_theorem_l4088_408889


namespace NUMINAMATH_CALUDE_notebook_difference_l4088_408852

theorem notebook_difference (price : ℚ) (mika_count leo_count : ℕ) : 
  price > (1 / 10 : ℚ) →
  price * mika_count = (12 / 5 : ℚ) →
  price * leo_count = (16 / 5 : ℚ) →
  leo_count - mika_count = 4 := by
  sorry

#check notebook_difference

end NUMINAMATH_CALUDE_notebook_difference_l4088_408852


namespace NUMINAMATH_CALUDE_daily_profit_function_l4088_408870

/-- The daily profit function for a product with given cost and sales quantity relation -/
theorem daily_profit_function (x : ℝ) : 
  let cost : ℝ := 8
  let sales_quantity : ℝ → ℝ := λ price => -price + 30
  let profit : ℝ → ℝ := λ price => (price - cost) * (sales_quantity price)
  profit x = -x^2 + 38*x - 240 := by
sorry

end NUMINAMATH_CALUDE_daily_profit_function_l4088_408870


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l4088_408860

theorem greatest_integer_fraction_inequality :
  ∀ y : ℤ, (8 : ℚ) / 11 > (y : ℚ) / 17 ↔ y ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l4088_408860


namespace NUMINAMATH_CALUDE_same_solution_k_value_l4088_408813

theorem same_solution_k_value (x : ℝ) (k : ℝ) : 
  (2 * x = 4) ∧ (3 * x + k = -2) → k = -8 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_k_value_l4088_408813


namespace NUMINAMATH_CALUDE_complex_absolute_value_product_l4088_408842

theorem complex_absolute_value_product : 
  Complex.abs ((3 * Real.sqrt 5 - 3 * Complex.I) * (2 * Real.sqrt 2 + 2 * Complex.I)) = 18 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_product_l4088_408842


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_under_1000_l4088_408837

theorem greatest_multiple_of_5_and_7_under_1000 : ∃ n : ℕ, 
  (n % 5 = 0) ∧ 
  (n % 7 = 0) ∧ 
  (n < 1000) ∧ 
  (∀ m : ℕ, (m % 5 = 0) ∧ (m % 7 = 0) ∧ (m < 1000) → m ≤ n) ∧
  (n = 980) := by
sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_under_1000_l4088_408837


namespace NUMINAMATH_CALUDE_solution_sum_of_squares_l4088_408846

-- Define the function f(t) = t^2 + sin(t)
noncomputable def f (t : ℝ) : ℝ := t^2 + Real.sin t

-- Define the equation
def equation (x y : ℝ) : Prop :=
  (4*x^2*y + 6*x^2 + 2*x*y - 4*x) / (3*x - y - 2) + 
  Real.sin ((3*x^2 + x*y + x - y - 2) / (3*x - y - 2)) = 
  2*x*y + y^2 + x^2/y^2 + 2*x/y + 
  (2*x*y*(x^2 + y^2)) / (3*x - y - 2)^2 + 
  (1 / (x + y)^2) * (x^2 * Real.sin ((x + y)^2 / x) + 
                     y^2 * Real.sin ((x + y)^2 / y^2) + 
                     2*x*y * Real.sin ((x + y)^2 / (3*x - y - 2)))

-- Theorem statement
theorem solution_sum_of_squares (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (heq : equation x y) :
  x^2 + y^2 = (85 + 13 * Real.sqrt 17) / 32 := by
  sorry

end NUMINAMATH_CALUDE_solution_sum_of_squares_l4088_408846


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l4088_408864

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 2, 2]
  A * B = !![23, -7; 24, -16] := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l4088_408864


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l4088_408825

theorem abs_sum_inequality (x y z : ℝ) : 
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l4088_408825


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_14_4_l4088_408802

theorem floor_plus_self_eq_14_4 :
  ∃! r : ℝ, ⌊r⌋ + r = 14.4 := by sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_14_4_l4088_408802


namespace NUMINAMATH_CALUDE_coefficient_x_cube_equals_neg_84_equal_coefficients_implies_n_7_l4088_408843

-- Part 1
def binomial_coefficient (n k : ℕ) : ℤ := sorry

def coefficient_x_cube (x : ℝ) : ℤ := 
  binomial_coefficient 9 3 * (-1)^3

theorem coefficient_x_cube_equals_neg_84 : 
  coefficient_x_cube = λ _ ↦ -84 := by sorry

-- Part 2
def nth_term_coefficient (n r : ℕ) : ℤ := 
  binomial_coefficient n r

theorem equal_coefficients_implies_n_7 (n : ℕ) : 
  nth_term_coefficient n 2 = nth_term_coefficient n 5 → n = 7 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_cube_equals_neg_84_equal_coefficients_implies_n_7_l4088_408843


namespace NUMINAMATH_CALUDE_fourth_player_win_probability_l4088_408885

/-- The probability of the fourth player winning in a coin-flipping game with four players -/
theorem fourth_player_win_probability :
  let p : ℕ → ℝ := λ n => (1 / 2) ^ (4 * n + 1)
  let total_prob := (∑' n, p n)
  total_prob = 1 / 30
  := by sorry

end NUMINAMATH_CALUDE_fourth_player_win_probability_l4088_408885


namespace NUMINAMATH_CALUDE_blue_flower_percentage_l4088_408881

theorem blue_flower_percentage (total_flowers : ℕ) (green_flowers : ℕ) (yellow_flowers : ℕ)
  (h1 : total_flowers = 96)
  (h2 : green_flowers = 9)
  (h3 : yellow_flowers = 12)
  : (total_flowers - (green_flowers + 3 * green_flowers + yellow_flowers)) / total_flowers * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_blue_flower_percentage_l4088_408881


namespace NUMINAMATH_CALUDE_second_sum_calculation_l4088_408872

theorem second_sum_calculation (total : ℚ) (x : ℚ) 
  (h1 : total = 2678)
  (h2 : x * (3 / 100) * 8 = (total - x) * (5 / 100) * 3) :
  total - x = 2401 := by
  sorry

end NUMINAMATH_CALUDE_second_sum_calculation_l4088_408872


namespace NUMINAMATH_CALUDE_polynomial_divisibility_by_five_l4088_408835

theorem polynomial_divisibility_by_five (a b c d : ℤ) :
  (∀ x : ℤ, (5 : ℤ) ∣ (a * x^3 + b * x^2 + c * x + d)) →
  (5 : ℤ) ∣ a ∧ (5 : ℤ) ∣ b ∧ (5 : ℤ) ∣ c ∧ (5 : ℤ) ∣ d := by
sorry


end NUMINAMATH_CALUDE_polynomial_divisibility_by_five_l4088_408835


namespace NUMINAMATH_CALUDE_sum_simplification_l4088_408888

theorem sum_simplification (n : ℕ) : 
  (Finset.range n).sum (λ i => (n - i) * 2^i) = 2^n + 1 - n - 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_simplification_l4088_408888


namespace NUMINAMATH_CALUDE_square_side_length_l4088_408855

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 225 → side * side = area → side = 15 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l4088_408855


namespace NUMINAMATH_CALUDE_number_satisfies_condition_l4088_408830

theorem number_satisfies_condition : ∃ n : ℕ, n = 250 ∧ (5 * n) / 8 = 156 ∧ (5 * n) % 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfies_condition_l4088_408830


namespace NUMINAMATH_CALUDE_pentagon_largest_angle_l4088_408887

/-- The sum of interior angles of a pentagon in degrees -/
def pentagon_angle_sum : ℕ := 540

/-- Represents the five consecutive integer angles of a pentagon -/
structure PentagonAngles where
  middle : ℕ
  valid : middle - 2 > 0 -- Ensures all angles are positive

/-- The sum of the five consecutive integer angles -/
def angle_sum (p : PentagonAngles) : ℕ :=
  (p.middle - 2) + (p.middle - 1) + p.middle + (p.middle + 1) + (p.middle + 2)

/-- The largest angle in the pentagon -/
def largest_angle (p : PentagonAngles) : ℕ := p.middle + 2

theorem pentagon_largest_angle :
  ∃ p : PentagonAngles, angle_sum p = pentagon_angle_sum ∧ largest_angle p = 110 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_largest_angle_l4088_408887


namespace NUMINAMATH_CALUDE_neither_directly_nor_inversely_proportional_l4088_408883

-- Define what it means for y to be directly proportional to x
def is_directly_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

-- Define what it means for y to be inversely proportional to x
def is_inversely_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

-- Define the two equations
def eq_A (x y : ℝ) : Prop := x^2 + x*y = 0
def eq_D (x y : ℝ) : Prop := 4*x + y^2 = 7

-- Theorem statement
theorem neither_directly_nor_inversely_proportional :
  (¬ ∃ f : ℝ → ℝ, (∀ x y : ℝ, eq_A x y ↔ y = f x) ∧ 
    (is_directly_proportional f ∨ is_inversely_proportional f)) ∧
  (¬ ∃ g : ℝ → ℝ, (∀ x y : ℝ, eq_D x y ↔ y = g x) ∧ 
    (is_directly_proportional g ∨ is_inversely_proportional g)) :=
by sorry

end NUMINAMATH_CALUDE_neither_directly_nor_inversely_proportional_l4088_408883


namespace NUMINAMATH_CALUDE_f_value_at_2_l4088_408844

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_value_at_2 (a b : ℝ) : f a b (-2) = 0 → f a b 2 = -16 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l4088_408844


namespace NUMINAMATH_CALUDE_oranges_for_juice_is_30_l4088_408831

/-- Given a number of bags of oranges, oranges per bag, rotten oranges, and oranges to be sold,
    calculate the number of oranges kept for juice. -/
def oranges_for_juice (bags : ℕ) (oranges_per_bag : ℕ) (rotten : ℕ) (to_sell : ℕ) : ℕ :=
  bags * oranges_per_bag - rotten - to_sell

/-- Theorem stating that under the given conditions, 30 oranges will be kept for juice. -/
theorem oranges_for_juice_is_30 :
  oranges_for_juice 10 30 50 220 = 30 := by
  sorry

end NUMINAMATH_CALUDE_oranges_for_juice_is_30_l4088_408831


namespace NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l4088_408886

theorem largest_x_sqrt_3x_eq_5x :
  (∃ x : ℝ, x > 0 ∧ Real.sqrt (3 * x) = 5 * x) →
  (∀ x : ℝ, Real.sqrt (3 * x) = 5 * x → x ≤ 3/25) ∧
  Real.sqrt (3 * (3/25)) = 5 * (3/25) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l4088_408886


namespace NUMINAMATH_CALUDE_xiaomas_calculation_l4088_408824

theorem xiaomas_calculation (square : ℤ) (h : 40 + square = 35) : 40 / square = -8 := by
  sorry

end NUMINAMATH_CALUDE_xiaomas_calculation_l4088_408824


namespace NUMINAMATH_CALUDE_intersection_solution_set_l4088_408882

def solution_set (f : ℝ → ℝ) : Set ℝ := {x | f x < 0}

def A : Set ℝ := solution_set (λ x => x^2 - 2*x - 3)
def B : Set ℝ := solution_set (λ x => x^2 + x - 6)

theorem intersection_solution_set (a b : ℝ) :
  solution_set (λ x => x^2 + a*x + b) = A ∩ B → a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_solution_set_l4088_408882


namespace NUMINAMATH_CALUDE_equation_solution_l4088_408822

theorem equation_solution :
  let f (x : ℝ) := Real.sqrt (7*x - 3) + Real.sqrt (2*x - 2)
  ∃ (x : ℝ), (f x = 3 ↔ (x = 2 ∨ x = 172/25)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4088_408822


namespace NUMINAMATH_CALUDE_right_triangle_angle_measure_l4088_408829

theorem right_triangle_angle_measure :
  ∀ (a b c : ℝ),
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = 90 →           -- One angle is a right angle (90°)
  b = 25 →           -- Another angle is 25°
  c = 65             -- The third angle is 65°
:= by sorry

end NUMINAMATH_CALUDE_right_triangle_angle_measure_l4088_408829


namespace NUMINAMATH_CALUDE_complex_fraction_cube_l4088_408841

theorem complex_fraction_cube (i : ℂ) (h : i^2 = -1) :
  ((1 + i) / (1 - i))^3 = -i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_cube_l4088_408841


namespace NUMINAMATH_CALUDE_probability_rain_given_wind_l4088_408867

theorem probability_rain_given_wind (P_rain P_wind P_rain_and_wind : ℝ) 
  (h1 : P_rain = 4/15)
  (h2 : P_wind = 2/5)
  (h3 : P_rain_and_wind = 1/10) :
  P_rain_and_wind / P_wind = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_rain_given_wind_l4088_408867


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_bounds_l4088_408894

-- Define the area function S
noncomputable def S (α : Real) : Real :=
  let β := α / 2
  let r := Real.sqrt (1 / (2 * Real.tan β))
  let a := r * (1 + Real.sin β) / Real.cos β
  let b := if β ≤ Real.pi / 4 then r * Real.sin (2 * β) else r
  (b / a) ^ 2

-- State the theorem
theorem isosceles_triangle_area_bounds :
  ∀ α : Real, Real.pi / 3 ≤ α ∧ α ≤ 2 * Real.pi / 3 →
    (1 / 4 : Real) ≥ S α ∧ S α ≥ 7 - 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_bounds_l4088_408894


namespace NUMINAMATH_CALUDE_polynomial_roots_l4088_408890

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

theorem polynomial_roots (P : ℝ → ℝ) (h_nonzero : P ≠ 0) 
  (h_form : ∀ x, P x = P 0 + P 1 * x + P 2 * x^2) :
  (∃ c ≠ 0, ∀ x, P x = c * (x^2 - x - 1)) ∧
  (∀ x, P x = 0 ↔ x = golden_ratio ∨ x = 1 - golden_ratio) :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l4088_408890


namespace NUMINAMATH_CALUDE_volunteer_selection_ways_l4088_408854

/-- The number of volunteers --/
def n : ℕ := 5

/-- The number of days for community service --/
def days : ℕ := 2

/-- The number of people selected each day --/
def selected_per_day : ℕ := 2

/-- Function to calculate the number of ways to select volunteers --/
def select_volunteers (n : ℕ) : ℕ :=
  (n) * (n - 1) * (n - 2)

theorem volunteer_selection_ways :
  select_volunteers n = 60 :=
sorry

end NUMINAMATH_CALUDE_volunteer_selection_ways_l4088_408854


namespace NUMINAMATH_CALUDE_y_equals_sixteen_l4088_408873

/-- The star operation defined as a ★ b = 4a - b -/
def star (a b : ℝ) : ℝ := 4 * a - b

/-- Theorem stating that y = 16 satisfies the equation 3 ★ (6 ★ y) = 4 -/
theorem y_equals_sixteen : ∃ y : ℝ, star 3 (star 6 y) = 4 ∧ y = 16 := by
  sorry

end NUMINAMATH_CALUDE_y_equals_sixteen_l4088_408873


namespace NUMINAMATH_CALUDE_A_subset_B_iff_l4088_408868

/-- The set A parameterized by a -/
def A (a : ℝ) : Set ℝ := {x | 1 < a * x ∧ a * x < 2}

/-- The set B -/
def B : Set ℝ := {x | |x| < 1}

/-- Theorem stating the condition for A to be a subset of B -/
theorem A_subset_B_iff (a : ℝ) : A a ⊆ B ↔ |a| ≥ 2 ∨ a = 0 := by sorry

end NUMINAMATH_CALUDE_A_subset_B_iff_l4088_408868


namespace NUMINAMATH_CALUDE_remainder_9876543210_mod_140_l4088_408838

theorem remainder_9876543210_mod_140 : 9876543210 % 140 = 70 := by
  sorry

end NUMINAMATH_CALUDE_remainder_9876543210_mod_140_l4088_408838


namespace NUMINAMATH_CALUDE_expression_simplification_l4088_408897

theorem expression_simplification (x y : ℝ) (h : x * y ≠ 0) :
  (x^2 + 2 / x^2) * (y^2 + 2 / y^2) + (x^2 - 2 / y^2) * (y^2 - 2 / x^2) = 2 + 8 / (x^2 * y^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4088_408897


namespace NUMINAMATH_CALUDE_max_value_quadratic_function_l4088_408821

/-- Given a quadratic function f(x) = ax^2 - 2x + c with x ∈ ℝ and range [0, +∞),
    the maximum value of 1/(c+1) + 4/(a+4) is 4/3 -/
theorem max_value_quadratic_function (a c : ℝ) : 
  (∀ x, a * x^2 - 2*x + c ≥ 0) →  -- Range is [0, +∞)
  (∃ x, a * x^2 - 2*x + c = 0) →  -- Minimum value is 0
  (∃ M, M = (1 / (c + 1) + 4 / (a + 4)) ∧ 
   ∀ a' c', (∀ x, a' * x^2 - 2*x + c' ≥ 0) → 
             (∃ x, a' * x^2 - 2*x + c' = 0) → 
             M ≥ (1 / (c' + 1) + 4 / (a' + 4))) →
  (1 / (c + 1) + 4 / (a + 4)) ≤ 4/3 := by
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_function_l4088_408821


namespace NUMINAMATH_CALUDE_alcohol_percentage_proof_l4088_408827

/-- Proves that given 16 liters of water added to 24 liters of an alcohol solution,
    resulting in a new mixture with 54% alcohol, the original solution contained 90% alcohol. -/
theorem alcohol_percentage_proof (original_volume : ℝ) (added_water : ℝ) (new_percentage : ℝ) :
  original_volume = 24 →
  added_water = 16 →
  new_percentage = 54 →
  let total_volume := original_volume + added_water
  let original_alcohol := (original_volume * new_percentage * total_volume) / (100 * original_volume)
  original_alcohol = 90 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_percentage_proof_l4088_408827


namespace NUMINAMATH_CALUDE_similar_right_triangles_l4088_408874

theorem similar_right_triangles (y : ℝ) : 
  (15 : ℝ) / 12 = y / 10 → y = 12.5 := by
sorry

end NUMINAMATH_CALUDE_similar_right_triangles_l4088_408874


namespace NUMINAMATH_CALUDE_rotational_homothety_similarity_l4088_408859

-- Define the rotational homothety transformation
def rotationalHomothety (k : ℝ) (θ : ℝ) (O : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

-- Define the fourth vertex of a parallelogram
def fourthVertex (O A A₁ : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define similarity of triangles
def trianglesSimilar (A B C A' B' C' : ℝ × ℝ) : Prop := sorry

theorem rotational_homothety_similarity 
  (A B C : ℝ × ℝ) -- Original triangle vertices
  (k : ℝ) (θ : ℝ) (O : ℝ × ℝ) -- Rotational homothety parameters
  (A₁ B₁ C₁ : ℝ × ℝ) -- Transformed triangle vertices
  (A₂ B₂ C₂ : ℝ × ℝ) -- Fourth vertices of parallelograms
  (h₁ : A₁ = rotationalHomothety k θ O A)
  (h₂ : B₁ = rotationalHomothety k θ O B)
  (h₃ : C₁ = rotationalHomothety k θ O C)
  (h₄ : A₂ = fourthVertex O A A₁)
  (h₅ : B₂ = fourthVertex O B B₁)
  (h₆ : C₂ = fourthVertex O C C₁) :
  trianglesSimilar A B C A₂ B₂ C₂ := by sorry

end NUMINAMATH_CALUDE_rotational_homothety_similarity_l4088_408859


namespace NUMINAMATH_CALUDE_power_function_special_case_l4088_408804

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x^a

-- State the theorem
theorem power_function_special_case (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2 / 2) : 
  f 4 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_power_function_special_case_l4088_408804


namespace NUMINAMATH_CALUDE_trapezoid_median_l4088_408858

/-- Given a triangle and a trapezoid with the same altitude, prove that the median of the trapezoid is 24 inches -/
theorem trapezoid_median (h : ℝ) : 
  let triangle_base : ℝ := 24
  let trapezoid_base1 : ℝ := 12
  let trapezoid_base2 : ℝ := 36
  let triangle_area : ℝ := (1/2) * triangle_base * h
  let trapezoid_area : ℝ := (1/2) * (trapezoid_base1 + trapezoid_base2) * h
  let trapezoid_median : ℝ := (1/2) * (trapezoid_base1 + trapezoid_base2)
  triangle_area = trapezoid_area → trapezoid_median = 24 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_median_l4088_408858


namespace NUMINAMATH_CALUDE_starWars_earnings_l4088_408840

/-- Represents movie financial data in millions of dollars -/
structure MovieData where
  cost : ℝ
  boxOffice : ℝ
  profit : ℝ

/-- The Lion King's financial data -/
def lionKing : MovieData := {
  cost := 10,
  boxOffice := 200,
  profit := 200 - 10
}

/-- Star Wars' financial data -/
def starWars : MovieData := {
  cost := 25,
  profit := 2 * lionKing.profit,
  boxOffice := 25 + 2 * lionKing.profit
}

/-- Theorem stating that Star Wars earned 405 million at the box office -/
theorem starWars_earnings : starWars.boxOffice = 405 := by
  sorry

#eval starWars.boxOffice

end NUMINAMATH_CALUDE_starWars_earnings_l4088_408840


namespace NUMINAMATH_CALUDE_garden_width_l4088_408895

theorem garden_width (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 768 →
  width = 16 := by
sorry

end NUMINAMATH_CALUDE_garden_width_l4088_408895


namespace NUMINAMATH_CALUDE_miss_graysons_class_fund_miss_graysons_class_fund_proof_l4088_408850

theorem miss_graysons_class_fund (initial_fund : ℕ) (student_contribution : ℕ) (num_students : ℕ) (trip_cost : ℕ) : ℕ :=
  let total_contribution := student_contribution * num_students
  let total_fund := initial_fund + total_contribution
  let total_trip_cost := trip_cost * num_students
  let remaining_fund := total_fund - total_trip_cost
  remaining_fund

theorem miss_graysons_class_fund_proof :
  miss_graysons_class_fund 50 5 20 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_miss_graysons_class_fund_miss_graysons_class_fund_proof_l4088_408850


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4088_408839

-- Define the sets M and N
def M : Set ℝ := {x | 1 - 2/x < 0}
def N : Set ℝ := {x | -1 ≤ x}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4088_408839


namespace NUMINAMATH_CALUDE_smallest_cube_for_pyramid_l4088_408878

/-- Represents a pyramid with a square base -/
structure Pyramid where
  height : ℝ
  baseLength : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Calculates the volume of a cube -/
def cubeVolume (c : Cube) : ℝ := c.sideLength ^ 3

/-- Determines if a cube can contain a pyramid standing upright -/
def canContainPyramid (c : Cube) (p : Pyramid) : Prop :=
  c.sideLength ≥ p.height ∧ c.sideLength ≥ p.baseLength

theorem smallest_cube_for_pyramid (p : Pyramid) (h1 : p.height = 12) (h2 : p.baseLength = 10) :
  ∃ (c : Cube), canContainPyramid c p ∧
    cubeVolume c = 1728 ∧
    ∀ (c' : Cube), canContainPyramid c' p → cubeVolume c' ≥ cubeVolume c :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_for_pyramid_l4088_408878


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l4088_408814

theorem least_addition_for_divisibility : 
  ∃ x : ℕ, x = 25 ∧ 
  (∀ y : ℕ, (27306 + y) % 151 = 0 → y ≥ x) ∧ 
  (27306 + x) % 151 = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l4088_408814


namespace NUMINAMATH_CALUDE_record_breaking_time_l4088_408800

/-- The number of jumps in the record -/
def record : ℕ := 54000

/-- The number of jumps Mark can do per second -/
def jumps_per_second : ℕ := 3

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The time required to break the record in hours -/
def time_to_break_record : ℚ :=
  (record / jumps_per_second) / seconds_per_hour

theorem record_breaking_time :
  time_to_break_record = 5 := by sorry

end NUMINAMATH_CALUDE_record_breaking_time_l4088_408800


namespace NUMINAMATH_CALUDE_race_winner_race_result_l4088_408812

theorem race_winner (race_length : ℝ) (speed_ratio_A : ℝ) (speed_ratio_B : ℝ) (head_start : ℝ) : ℝ :=
  let time_B_finish := race_length / speed_ratio_B
  let distance_A := speed_ratio_A * time_B_finish + head_start
  distance_A - race_length

theorem race_result :
  race_winner 500 3 4 140 = 15 := by sorry

end NUMINAMATH_CALUDE_race_winner_race_result_l4088_408812


namespace NUMINAMATH_CALUDE_bart_money_theorem_l4088_408875

theorem bart_money_theorem :
  ∃ m : ℕ, m > 0 ∧ ∀ n : ℕ, n ≥ m → ∃ a b : ℕ, n = 17 * a + 19 * b := by
  sorry

end NUMINAMATH_CALUDE_bart_money_theorem_l4088_408875


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l4088_408892

def i : ℂ := Complex.I

theorem z_in_third_quadrant (z : ℂ) (h : z * (1 + i) = -2 * i) :
  z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l4088_408892


namespace NUMINAMATH_CALUDE_right_triangle_arm_square_l4088_408832

theorem right_triangle_arm_square (a c : ℝ) (h1 : c = a + 2) :
  ∃ b : ℝ, a^2 + b^2 = c^2 ∧ b^2 = 4*a + 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arm_square_l4088_408832


namespace NUMINAMATH_CALUDE_new_cost_percentage_l4088_408803

variable (t b : ℝ)
variable (cost : ℝ → ℝ)

/-- The cost function is defined as tb^4 --/
def cost_function (t b : ℝ) : ℝ := t * b^4

/-- The original cost --/
def original_cost : ℝ := cost_function t b

/-- The new cost when b is doubled --/
def new_cost : ℝ := cost_function t (2*b)

/-- The theorem stating that the new cost is 1600% of the original cost --/
theorem new_cost_percentage : new_cost = 16 * original_cost := by sorry

end NUMINAMATH_CALUDE_new_cost_percentage_l4088_408803


namespace NUMINAMATH_CALUDE_divisibility_by_480_l4088_408863

theorem divisibility_by_480 (a : ℤ) 
  (h1 : ¬ (4 ∣ a)) 
  (h2 : a % 10 = 4) : 
  480 ∣ (a * (a^2 - 1) * (a^2 - 4)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_480_l4088_408863


namespace NUMINAMATH_CALUDE_kamal_age_double_son_l4088_408880

/-- The number of years after which Kamal will be twice as old as his son -/
def years_until_double_age (kamal_age : ℕ) (son_age : ℕ) : ℕ :=
  kamal_age + 8 - 2 * (son_age + 8)

/-- Kamal's current age -/
def kamal_current_age : ℕ := 40

theorem kamal_age_double_son :
  years_until_double_age kamal_current_age
    ((kamal_current_age - 8) / 4 + 8) = 8 := by
  sorry

end NUMINAMATH_CALUDE_kamal_age_double_son_l4088_408880


namespace NUMINAMATH_CALUDE_slices_in_large_pizza_l4088_408899

/-- Given that Mary orders 2 large pizzas, eats 7 slices, and has 9 slices remaining,
    prove that there are 8 slices in a large pizza. -/
theorem slices_in_large_pizza :
  ∀ (total_pizzas : ℕ) (slices_eaten : ℕ) (slices_remaining : ℕ),
    total_pizzas = 2 →
    slices_eaten = 7 →
    slices_remaining = 9 →
    (slices_remaining + slices_eaten) / total_pizzas = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_slices_in_large_pizza_l4088_408899


namespace NUMINAMATH_CALUDE_theta_max_ratio_l4088_408809

/-- Represents a participant's scores in the competition -/
structure Participant where
  day1_score : ℕ
  day1_total : ℕ
  day2_score : ℕ
  day2_total : ℕ
  day3_score : ℕ
  day3_total : ℕ

/-- The competition setup and conditions -/
def Competition (omega theta : Participant) : Prop :=
  omega.day1_score = 200 ∧
  omega.day1_total = 400 ∧
  omega.day2_score + omega.day3_score = 150 ∧
  omega.day2_total + omega.day3_total = 200 ∧
  omega.day1_total + omega.day2_total + omega.day3_total = 600 ∧
  theta.day1_total + theta.day2_total + theta.day3_total = 600 ∧
  theta.day1_score > 0 ∧ theta.day2_score > 0 ∧ theta.day3_score > 0 ∧
  (theta.day1_score : ℚ) / theta.day1_total < (omega.day1_score : ℚ) / omega.day1_total ∧
  (theta.day2_score : ℚ) / theta.day2_total < (omega.day2_score : ℚ) / omega.day2_total ∧
  (theta.day3_score : ℚ) / theta.day3_total < (omega.day3_score : ℚ) / omega.day3_total

/-- Theta's overall success ratio -/
def ThetaRatio (theta : Participant) : ℚ :=
  (theta.day1_score + theta.day2_score + theta.day3_score : ℚ) /
  (theta.day1_total + theta.day2_total + theta.day3_total)

/-- The main theorem stating Theta's maximum possible success ratio -/
theorem theta_max_ratio (omega theta : Participant) 
  (h : Competition omega theta) : ThetaRatio theta ≤ 56 / 75 := by
  sorry


end NUMINAMATH_CALUDE_theta_max_ratio_l4088_408809


namespace NUMINAMATH_CALUDE_symmetric_point_of_P_l4088_408866

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The origin point (0,0,0) -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- Given point P -/
def P : Point3D := ⟨3, 1, 5⟩

/-- Function to find the symmetric point about the origin -/
def symmetricPoint (p : Point3D) : Point3D :=
  ⟨-p.x, -p.y, -p.z⟩

/-- Theorem: The point symmetric to P(3,1,5) about the origin is (3,-1,-5) -/
theorem symmetric_point_of_P :
  symmetricPoint P = Point3D.mk 3 (-1) (-5) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_of_P_l4088_408866
