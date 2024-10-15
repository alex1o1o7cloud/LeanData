import Mathlib

namespace NUMINAMATH_CALUDE_f_extended_domain_l2509_250900

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_extended_domain (f : ℝ → ℝ) :
  is_even f →
  has_period f π →
  (∀ x ∈ Set.Icc 0 (π / 2), f x = 1 - Real.sin x) →
  ∀ x ∈ Set.Icc (5 * π / 2) (3 * π), f x = 1 - Real.sin x :=
by sorry

end NUMINAMATH_CALUDE_f_extended_domain_l2509_250900


namespace NUMINAMATH_CALUDE_product_multiple_of_five_probability_l2509_250993

def N : ℕ := 2020

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

def count_multiples_of_five : ℕ := N / 5

def prob_not_multiple_of_five : ℚ := (N - count_multiples_of_five) / N

theorem product_multiple_of_five_probability :
  let p := 1 - (prob_not_multiple_of_five * (prob_not_multiple_of_five - 1 / N) * (prob_not_multiple_of_five - 2 / N))
  ∃ ε > 0, |p - 0.485| < ε :=
sorry

end NUMINAMATH_CALUDE_product_multiple_of_five_probability_l2509_250993


namespace NUMINAMATH_CALUDE_square_root_of_four_l2509_250908

theorem square_root_of_four : ∃ x : ℝ, x^2 = 4 ∧ (x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_four_l2509_250908


namespace NUMINAMATH_CALUDE_immediate_boarding_probability_l2509_250909

/-- Represents the cycle time of a subway train in minutes -/
def cycletime : ℝ := 10

/-- Represents the time the train stops at the station in minutes -/
def stoptime : ℝ := 1

/-- Theorem: The probability of a passenger arriving at the platform 
    and immediately boarding the train is 1/10 -/
theorem immediate_boarding_probability : 
  stoptime / cycletime = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_immediate_boarding_probability_l2509_250909


namespace NUMINAMATH_CALUDE_sum_of_decimals_l2509_250956

theorem sum_of_decimals : 5.46 + 4.537 = 9.997 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l2509_250956


namespace NUMINAMATH_CALUDE_sports_club_tennis_players_l2509_250999

/-- Given a sports club with the following properties:
  * There are 30 members in total
  * 17 members play badminton
  * 2 members do not play either badminton or tennis
  * 10 members play both badminton and tennis
  Prove that 21 members play tennis -/
theorem sports_club_tennis_players :
  ∀ (total_members badminton_players neither_players both_players : ℕ),
    total_members = 30 →
    badminton_players = 17 →
    neither_players = 2 →
    both_players = 10 →
    ∃ (tennis_players : ℕ),
      tennis_players = 21 ∧
      tennis_players = total_members - neither_players - (badminton_players - both_players) :=
by sorry

end NUMINAMATH_CALUDE_sports_club_tennis_players_l2509_250999


namespace NUMINAMATH_CALUDE_ellipse_foci_range_ellipse_or_quadratic_range_l2509_250995

/-- Definition of an ellipse with semi-major axis 5 and semi-minor axis √a -/
def is_ellipse (a : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / 5 + y^2 / a = 1

/-- The foci of the ellipse are on the x-axis -/
def foci_on_x_axis (a : ℝ) : Prop :=
  is_ellipse a ∧ ∃ c : ℝ, c^2 = 5 - a ∧ c ≥ 0

/-- The quadratic inequality holds for all real x -/
def quadratic_inequality_holds (a : ℝ) : Prop :=
  ∀ x : ℝ, 3 * x^2 + 2 * a * x + 3 ≥ 0

theorem ellipse_foci_range (a : ℝ) :
  foci_on_x_axis a → 0 < a ∧ a < 5 :=
sorry

theorem ellipse_or_quadratic_range (a : ℝ) :
  (foci_on_x_axis a ∨ quadratic_inequality_holds a) ∧
  ¬(foci_on_x_axis a ∧ quadratic_inequality_holds a) →
  (3 < a ∧ a < 5) ∨ (-3 ≤ a ∧ a ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_range_ellipse_or_quadratic_range_l2509_250995


namespace NUMINAMATH_CALUDE_det_dilation_matrix_3d_det_dilation_matrix_3d_scale_5_l2509_250924

def dilation_matrix (scale : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.diagonal (λ _ => scale)

theorem det_dilation_matrix_3d (scale : ℝ) :
  Matrix.det (dilation_matrix scale) = scale ^ 3 := by
  sorry

theorem det_dilation_matrix_3d_scale_5 :
  Matrix.det (dilation_matrix 5) = 125 := by
  sorry

end NUMINAMATH_CALUDE_det_dilation_matrix_3d_det_dilation_matrix_3d_scale_5_l2509_250924


namespace NUMINAMATH_CALUDE_parabola_intersection_distance_l2509_250903

theorem parabola_intersection_distance : 
  ∀ (p q r s : ℝ), 
  (∃ x y : ℝ, y = 3*x^2 - 6*x + 3 ∧ y = -x^2 - 3*x + 3 ∧ ((x = p ∧ y = q) ∨ (x = r ∧ y = s))) → 
  r ≥ p → 
  r - p = 3/4 := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_distance_l2509_250903


namespace NUMINAMATH_CALUDE_negation_equivalence_l2509_250965

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2509_250965


namespace NUMINAMATH_CALUDE_andrea_skating_schedule_l2509_250926

/-- Represents Andrea's skating schedule and target average -/
structure SkatingSchedule where
  days_schedule1 : ℕ  -- Number of days for schedule 1
  minutes_per_day1 : ℕ  -- Minutes skated per day in schedule 1
  days_schedule2 : ℕ  -- Number of days for schedule 2
  minutes_per_day2 : ℕ  -- Minutes skated per day in schedule 2
  total_days : ℕ  -- Total number of days
  target_average : ℕ  -- Target average minutes per day

/-- Calculates the required skating time for the last day to achieve the target average -/
def required_last_day_minutes (schedule : SkatingSchedule) : ℕ :=
  schedule.target_average * schedule.total_days -
  (schedule.days_schedule1 * schedule.minutes_per_day1 +
   schedule.days_schedule2 * schedule.minutes_per_day2)

/-- Theorem stating that given Andrea's skating schedule, 
    she needs to skate 175 minutes on the ninth day to achieve the target average -/
theorem andrea_skating_schedule :
  let schedule : SkatingSchedule := {
    days_schedule1 := 6,
    minutes_per_day1 := 80,
    days_schedule2 := 2,
    minutes_per_day2 := 100,
    total_days := 9,
    target_average := 95
  }
  required_last_day_minutes schedule = 175 := by
  sorry

end NUMINAMATH_CALUDE_andrea_skating_schedule_l2509_250926


namespace NUMINAMATH_CALUDE_opposite_event_of_hit_at_least_once_l2509_250986

-- Define the sample space for two shots
inductive ShotOutcome
  | Hit
  | Miss

-- Define the event of hitting the target at least once
def hitAtLeastOnce (outcome1 outcome2 : ShotOutcome) : Prop :=
  outcome1 = ShotOutcome.Hit ∨ outcome2 = ShotOutcome.Hit

-- Define the event of both shots missing
def bothShotsMiss (outcome1 outcome2 : ShotOutcome) : Prop :=
  outcome1 = ShotOutcome.Miss ∧ outcome2 = ShotOutcome.Miss

-- Theorem statement
theorem opposite_event_of_hit_at_least_once 
  (outcome1 outcome2 : ShotOutcome) : 
  ¬(hitAtLeastOnce outcome1 outcome2) ↔ bothShotsMiss outcome1 outcome2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_event_of_hit_at_least_once_l2509_250986


namespace NUMINAMATH_CALUDE_complementary_event_l2509_250910

/-- The sample space of outcomes when two students purchase a beverage with a chance of winning a prize -/
inductive Outcome
  | BothWin
  | AWinsBLoses
  | ALosesBWins
  | BothLose

/-- The event where both students win a prize -/
def bothWin (o : Outcome) : Prop :=
  o = Outcome.BothWin

/-- The event where at most one student wins a prize -/
def atMostOneWins (o : Outcome) : Prop :=
  o = Outcome.AWinsBLoses ∨ o = Outcome.ALosesBWins ∨ o = Outcome.BothLose

/-- Theorem stating that the complementary event to "both win" is "at most one wins" -/
theorem complementary_event :
  ∀ o : Outcome, ¬(bothWin o) ↔ atMostOneWins o :=
sorry


end NUMINAMATH_CALUDE_complementary_event_l2509_250910


namespace NUMINAMATH_CALUDE_paint_mixture_ratio_l2509_250919

/-- Given a paint mixture with a ratio of 5:3:7 for red:yellow:white paint,
    if 21 quarts of white paint is used, then 9 quarts of yellow paint should be used. -/
theorem paint_mixture_ratio (red yellow white : ℚ) :
  red / yellow = 5 / 3 →
  yellow / white = 3 / 7 →
  white = 21 →
  yellow = 9 := by
  sorry

end NUMINAMATH_CALUDE_paint_mixture_ratio_l2509_250919


namespace NUMINAMATH_CALUDE_abs_sum_minimum_l2509_250940

theorem abs_sum_minimum (x : ℝ) : 
  |x + 3| + |x + 5| + |x + 6| + |x + 7| ≥ 5 ∧ 
  ∃ y : ℝ, |y + 3| + |y + 5| + |y + 6| + |y + 7| = 5 := by
  sorry

#check abs_sum_minimum

end NUMINAMATH_CALUDE_abs_sum_minimum_l2509_250940


namespace NUMINAMATH_CALUDE_largest_hope_number_proof_l2509_250963

/-- A Hope Number is a natural number with an odd number of divisors --/
def isHopeNumber (n : ℕ) : Prop := Odd (Nat.divisors n).card

/-- The largest Hope Number within 1000 --/
def largestHopeNumber : ℕ := 961

theorem largest_hope_number_proof :
  (∀ m : ℕ, m ≤ 1000 → isHopeNumber m → m ≤ largestHopeNumber) ∧
  isHopeNumber largestHopeNumber ∧
  largestHopeNumber ≤ 1000 :=
sorry

end NUMINAMATH_CALUDE_largest_hope_number_proof_l2509_250963


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2509_250911

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5/8) (h2 : x - y = 3/8) : x^2 - y^2 = 15/64 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2509_250911


namespace NUMINAMATH_CALUDE_sum_cubes_quartics_bounds_l2509_250927

theorem sum_cubes_quartics_bounds (p q r s : ℝ) 
  (sum_condition : p + q + r + s = 10)
  (sum_squares_condition : p^2 + q^2 + r^2 + s^2 = 20) :
  let expr := 3 * (p^3 + q^3 + r^3 + s^3) - 5 * (p^4 + q^4 + r^4 + s^4)
  ∃ (min max : ℝ), min = 132 ∧ max = -20 ∧ min ≤ expr ∧ expr ≤ max := by
  sorry

end NUMINAMATH_CALUDE_sum_cubes_quartics_bounds_l2509_250927


namespace NUMINAMATH_CALUDE_polymerization_of_tetrafluoroethylene_yields_teflon_l2509_250928

-- Define the monomer tetrafluoroethylene
structure Tetrafluoroethylene : Type :=
  (formula : String)

-- Define the polymer Teflon (PTFE)
structure Teflon : Type :=
  (formula : String)

-- Define the polymerization process
def polymerize (monomer : Tetrafluoroethylene) : Teflon :=
  sorry

-- Theorem statement
theorem polymerization_of_tetrafluoroethylene_yields_teflon 
  (monomer : Tetrafluoroethylene) 
  (h : monomer.formula = "CF2=CF2") :
  (polymerize monomer).formula = "(-CF2-CF2-)n" :=
sorry

end NUMINAMATH_CALUDE_polymerization_of_tetrafluoroethylene_yields_teflon_l2509_250928


namespace NUMINAMATH_CALUDE_cupcake_cost_l2509_250929

/-- Proves that the cost of a cupcake is 40 cents given initial amount, juice box cost, and remaining amount --/
theorem cupcake_cost (initial_amount : ℕ) (juice_cost : ℕ) (remaining : ℕ) :
  initial_amount = 75 →
  juice_cost = 27 →
  remaining = 8 →
  initial_amount - juice_cost - remaining = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_cupcake_cost_l2509_250929


namespace NUMINAMATH_CALUDE_chewbacca_gum_packs_l2509_250913

theorem chewbacca_gum_packs (y : ℚ) : 
  (∃ (orange_packs apple_packs : ℕ), 
    orange_packs * y + (25 : ℚ) % y = 25 ∧ 
    apple_packs * y + (35 : ℚ) % y = 35 ∧
    (25 - 2 * y) / 35 = 25 / (35 + 4 * y)) → 
  y = 15 / 4 := by
sorry

end NUMINAMATH_CALUDE_chewbacca_gum_packs_l2509_250913


namespace NUMINAMATH_CALUDE_f_1988_11_equals_169_l2509_250985

/-- Sum of digits of a positive integer -/
def sumOfDigits (k : ℕ+) : ℕ := sorry

/-- Square of sum of digits -/
def f₁ (k : ℕ+) : ℕ := (sumOfDigits k) ^ 2

/-- Recursive definition of fₙ -/
def f (n : ℕ) (k : ℕ+) : ℕ :=
  match n with
  | 0 => k.val
  | 1 => f₁ k
  | n + 1 => f₁ ⟨f n k, sorry⟩

/-- The main theorem to prove -/
theorem f_1988_11_equals_169 : f 1988 11 = 169 := by sorry

end NUMINAMATH_CALUDE_f_1988_11_equals_169_l2509_250985


namespace NUMINAMATH_CALUDE_geometric_progression_properties_l2509_250920

/-- A geometric progression with given second and fifth terms -/
structure GeometricProgression where
  b₂ : ℝ
  b₅ : ℝ
  h₁ : b₂ = 24.5
  h₂ : b₅ = 196

/-- The third term of the geometric progression -/
def thirdTerm (gp : GeometricProgression) : ℝ := 49

/-- The sum of the first four terms of the geometric progression -/
def sumFirstFour (gp : GeometricProgression) : ℝ := 183.75

theorem geometric_progression_properties (gp : GeometricProgression) :
  thirdTerm gp = 49 ∧ sumFirstFour gp = 183.75 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_properties_l2509_250920


namespace NUMINAMATH_CALUDE_number_of_women_at_tables_l2509_250925

/-- Proves that the number of women at the tables is 7.0 -/
theorem number_of_women_at_tables 
  (num_tables : Float) 
  (num_men : Float) 
  (avg_customers_per_table : Float) 
  (h1 : num_tables = 9.0)
  (h2 : num_men = 3.0)
  (h3 : avg_customers_per_table = 1.111111111) : 
  Float.round ((num_tables * avg_customers_per_table) - num_men) = 7.0 := by
  sorry

end NUMINAMATH_CALUDE_number_of_women_at_tables_l2509_250925


namespace NUMINAMATH_CALUDE_find_k_l2509_250989

/-- The sum of the first n terms of the sequence {a_n} -/
def S (n : ℕ) (k : ℝ) : ℝ := 5 * n^2 + k * n

/-- The nth term of the sequence {a_n} -/
def a (n : ℕ) (k : ℝ) : ℝ := S n k - S (n-1) k

theorem find_k : ∃ k : ℝ, (∀ n : ℕ, S n k = 5 * n^2 + k * n) ∧ a 2 k = 18 → k = 3 :=
sorry

end NUMINAMATH_CALUDE_find_k_l2509_250989


namespace NUMINAMATH_CALUDE_P_degree_P_terms_P_descending_y_l2509_250954

/-- The polynomial P(x,y) = 3x²y - xy² - 3xy³ + x⁵ - 1 -/
def P (x y : ℝ) : ℝ := 3*x^2*y - x*y^2 - 3*x*y^3 + x^5 - 1

/-- The degree of polynomial P is 5 -/
theorem P_degree : 
  ∃ (n : ℕ), n = 5 ∧ (∀ m : ℕ, (∃ (x y : ℝ), P x y ≠ 0 → m ≤ n)) ∧ 
  (∃ (x y : ℝ), P x y ≠ 0 ∧ n = 5) :=
sorry

/-- P has 5 terms -/
theorem P_terms : 
  ∃ (t : ℕ), t = 5 ∧ (∀ (x y : ℝ), P x y = 3*x^2*y - x*y^2 - 3*x*y^3 + x^5 - 1) :=
sorry

/-- When arranged in descending order of y, P = -3xy³ - xy² + 3x²y + x⁵ - 1 -/
theorem P_descending_y :
  ∀ (x y : ℝ), P x y = -3*x*y^3 - x*y^2 + 3*x^2*y + x^5 - 1 :=
sorry

end NUMINAMATH_CALUDE_P_degree_P_terms_P_descending_y_l2509_250954


namespace NUMINAMATH_CALUDE_wire_length_ratio_l2509_250922

/-- The ratio of wire lengths in cube frame construction -/
theorem wire_length_ratio : 
  ∀ (bonnie_wire_length roark_wire_length : ℕ) 
    (bonnie_cube_volume roark_total_volume : ℕ),
  bonnie_wire_length = 12 * 8 →
  bonnie_cube_volume = 8^3 →
  roark_total_volume = bonnie_cube_volume →
  (∃ (num_small_cubes : ℕ), 
    roark_total_volume = num_small_cubes * 2^3 ∧
    roark_wire_length = num_small_cubes * 12 * 2) →
  (bonnie_wire_length : ℚ) / roark_wire_length = 1 / 16 := by
sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l2509_250922


namespace NUMINAMATH_CALUDE_remaining_denomination_is_500_l2509_250917

/-- Represents the denomination problem with given conditions -/
def DenominationProblem (total_amount : ℕ) (total_notes : ℕ) (fifty_notes : ℕ) (fifty_value : ℕ) : Prop :=
  ∃ (other_denom : ℕ),
    total_amount = fifty_notes * fifty_value + (total_notes - fifty_notes) * other_denom ∧
    total_notes > fifty_notes ∧
    other_denom > 0

/-- Theorem stating that the denomination of remaining notes is 500 -/
theorem remaining_denomination_is_500 :
  DenominationProblem 10350 126 117 50 → ∃ (other_denom : ℕ), other_denom = 500 :=
by sorry

end NUMINAMATH_CALUDE_remaining_denomination_is_500_l2509_250917


namespace NUMINAMATH_CALUDE_positive_solution_x_l2509_250966

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y = 8 - 3 * x - 2 * y)
  (eq2 : y * z = 10 - 5 * y - 3 * z)
  (eq3 : x * z = 40 - 5 * x - 4 * z)
  (x_pos : x > 0) :
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_positive_solution_x_l2509_250966


namespace NUMINAMATH_CALUDE_multiples_of_15_between_21_and_205_l2509_250936

theorem multiples_of_15_between_21_and_205 : 
  (Finset.filter (fun n => n % 15 = 0 ∧ n > 21 ∧ n < 205) (Finset.range 205)).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_15_between_21_and_205_l2509_250936


namespace NUMINAMATH_CALUDE_smoking_lung_disease_relation_l2509_250991

/-- Represents the Chi-square statistic -/
def K_squared : ℝ := 5.231

/-- The probability that K^2 is greater than or equal to 3.841 -/
def P_3_841 : ℝ := 0.05

/-- The probability that K^2 is greater than or equal to 6.635 -/
def P_6_635 : ℝ := 0.01

/-- The confidence level for the relationship between smoking and lung disease -/
def confidence_level : ℝ := 1 - P_3_841

/-- Theorem stating that there is more than 95% confidence that smoking is related to lung disease -/
theorem smoking_lung_disease_relation :
  K_squared > 3.841 ∧ confidence_level > 0.95 := by sorry

end NUMINAMATH_CALUDE_smoking_lung_disease_relation_l2509_250991


namespace NUMINAMATH_CALUDE_impossibleConstruction_l2509_250918

-- Define the basic structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a triangle in 3D space
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

-- Define a function to check if a triangle is acute-angled
def isAcuteAngled (t : Triangle3D) : Prop := sorry

-- Define a function to check if a triangle is equilateral
def isEquilateral (t : Triangle3D) : Prop := sorry

-- Define a function to check if three lines intersect at a point
def linesIntersectAtPoint (A A' B B' C C' O : Point3D) : Prop := sorry

-- Main theorem
theorem impossibleConstruction (ABC : Triangle3D) (O : Point3D) :
  isAcuteAngled ABC →
  ¬∃ (A'B'C' : Triangle3D),
    isEquilateral A'B'C' ∧
    linesIntersectAtPoint ABC.A A'B'C'.A ABC.B A'B'C'.B ABC.C A'B'C'.C O :=
by sorry

end NUMINAMATH_CALUDE_impossibleConstruction_l2509_250918


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_q_gt_one_l2509_250942

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = q * a n)

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > a n

theorem geometric_sequence_increasing_iff_q_gt_one (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q → (IncreasingSequence a ↔ q > 1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_q_gt_one_l2509_250942


namespace NUMINAMATH_CALUDE_order_of_fractions_with_exponents_l2509_250932

theorem order_of_fractions_with_exponents :
  (1/5 : ℝ)^(2/3) < (1/2 : ℝ)^(2/3) ∧ (1/2 : ℝ)^(2/3) < (1/2 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_order_of_fractions_with_exponents_l2509_250932


namespace NUMINAMATH_CALUDE_boys_from_clay_middle_school_l2509_250951

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

end NUMINAMATH_CALUDE_boys_from_clay_middle_school_l2509_250951


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2509_250906

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- S_n is the sum of the first n terms

/-- Given an arithmetic sequence with S_10 = 20 and S_20 = 15, prove S_30 = -15 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence) 
  (h1 : a.S 10 = 20) 
  (h2 : a.S 20 = 15) : 
  a.S 30 = -15 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2509_250906


namespace NUMINAMATH_CALUDE_balboa_earned_180_l2509_250972

/-- Represents the earnings of students from three middle schools --/
structure SchoolEarnings where
  allen_students : Nat
  allen_days : Nat
  balboa_students : Nat
  balboa_days : Nat
  carver_students : Nat
  carver_days : Nat
  total_paid : Nat

/-- Calculates the total earnings for Balboa school students --/
def balboa_earnings (e : SchoolEarnings) : Nat :=
  let total_student_days := e.allen_students * e.allen_days + 
                            e.balboa_students * e.balboa_days + 
                            e.carver_students * e.carver_days
  let daily_wage := e.total_paid / total_student_days
  daily_wage * e.balboa_students * e.balboa_days

/-- Theorem stating that Balboa school students earned 180 dollars --/
theorem balboa_earned_180 (e : SchoolEarnings) 
  (h1 : e.allen_students = 7)
  (h2 : e.allen_days = 3)
  (h3 : e.balboa_students = 4)
  (h4 : e.balboa_days = 5)
  (h5 : e.carver_students = 5)
  (h6 : e.carver_days = 9)
  (h7 : e.total_paid = 744) :
  balboa_earnings e = 180 := by
  sorry

end NUMINAMATH_CALUDE_balboa_earned_180_l2509_250972


namespace NUMINAMATH_CALUDE_smallest_c_for_three_in_range_l2509_250981

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

-- Theorem statement
theorem smallest_c_for_three_in_range :
  ∀ c : ℝ, (∃ x : ℝ, f c x = 3) ↔ c ≥ 12 := by sorry

end NUMINAMATH_CALUDE_smallest_c_for_three_in_range_l2509_250981


namespace NUMINAMATH_CALUDE_nonreal_cube_root_of_unity_sum_l2509_250968

theorem nonreal_cube_root_of_unity_sum (ω : ℂ) : 
  ω^3 = 1 ∧ ω ≠ 1 → (1 - ω + ω^2)^6 + (1 + ω - ω^2)^6 = 128 := by sorry

end NUMINAMATH_CALUDE_nonreal_cube_root_of_unity_sum_l2509_250968


namespace NUMINAMATH_CALUDE_stacy_pages_per_day_l2509_250996

/-- Given a paper with a certain number of pages due in a certain number of days,
    calculate the number of pages that need to be written per day to finish on time. -/
def pages_per_day (total_pages : ℕ) (total_days : ℕ) : ℚ :=
  total_pages / total_days

/-- Theorem: Stacy needs to write 1 page per day to finish her paper on time. -/
theorem stacy_pages_per_day :
  pages_per_day 12 12 = 1 := by
  sorry

#eval pages_per_day 12 12

end NUMINAMATH_CALUDE_stacy_pages_per_day_l2509_250996


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2509_250994

theorem roots_of_polynomial (a b : ℝ) : 
  (a + 3 * Complex.I) * (b + 6 * Complex.I) = 52 + 105 * Complex.I ∧
  (a + 3 * Complex.I) + (b + 6 * Complex.I) = 12 + 15 * Complex.I →
  a = 23 ∧ b = -11 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2509_250994


namespace NUMINAMATH_CALUDE_city_visit_selection_schemes_l2509_250931

theorem city_visit_selection_schemes :
  let total_people : ℕ := 6
  let selected_people : ℕ := 4
  let total_cities : ℕ := 4
  let restricted_people : ℕ := 2
  let restricted_cities : ℕ := 1

  (total_people - restricted_people) *
  (total_people - 1) *
  (total_people - 2) *
  (total_people - 3) = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_city_visit_selection_schemes_l2509_250931


namespace NUMINAMATH_CALUDE_john_study_time_for_average_75_l2509_250921

/-- Represents the relationship between study time and test score -/
structure StudyScoreRelation where
  k : ℝ  -- Proportionality constant
  study_time : ℝ → ℝ  -- Function mapping score to study time
  score : ℝ → ℝ  -- Function mapping study time to score

/-- John's hypothesis about study time and test score -/
def john_hypothesis (r : StudyScoreRelation) : Prop :=
  ∀ t, r.score t = r.k * t

theorem john_study_time_for_average_75 
  (r : StudyScoreRelation)
  (h1 : john_hypothesis r)
  (h2 : r.score 3 = 60)  -- First exam result
  (h3 : r.k = 20)  -- Derived from first exam
  : r.study_time 90 = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_john_study_time_for_average_75_l2509_250921


namespace NUMINAMATH_CALUDE_unique_number_with_properties_l2509_250947

/-- Given a natural number n, returns the sum of its digits. -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Given a two-digit number n, returns the number formed by reversing its digits. -/
def reverse_digits (n : ℕ) : ℕ := sorry

/-- Predicate that checks if a number is two-digit. -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem unique_number_with_properties : 
  ∃! n : ℕ, is_two_digit n ∧ 
    n = 4 * (sum_of_digits n) + 3 ∧ 
    n + 18 = reverse_digits n := by sorry

end NUMINAMATH_CALUDE_unique_number_with_properties_l2509_250947


namespace NUMINAMATH_CALUDE_boat_distance_proof_l2509_250977

theorem boat_distance_proof (boat_speed : ℝ) (stream_speed : ℝ) (time_difference : ℝ) :
  boat_speed = 10 →
  stream_speed = 2 →
  time_difference = 1.5 →
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  ∃ distance : ℝ,
    distance / upstream_speed = distance / downstream_speed + time_difference ∧
    distance = 36 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_proof_l2509_250977


namespace NUMINAMATH_CALUDE_mabel_transactions_l2509_250916

theorem mabel_transactions :
  ∀ (mabel anthony cal jade : ℕ),
    anthony = mabel + mabel / 10 →
    cal = (2 * anthony) / 3 →
    jade = cal + 17 →
    jade = 83 →
    mabel = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_mabel_transactions_l2509_250916


namespace NUMINAMATH_CALUDE_pandemic_cut_fifty_percent_l2509_250960

/-- Represents a car factory with its production details -/
structure CarFactory where
  doorsPerCar : ℕ
  initialProduction : ℕ
  metalShortageDecrease : ℕ
  finalDoorProduction : ℕ

/-- Calculates the percentage of production cut due to a pandemic -/
def pandemicProductionCutPercentage (factory : CarFactory) : ℚ :=
  let productionAfterMetalShortage := factory.initialProduction - factory.metalShortageDecrease
  let finalCarProduction := factory.finalDoorProduction / factory.doorsPerCar
  let pandemicCut := productionAfterMetalShortage - finalCarProduction
  (pandemicCut / productionAfterMetalShortage) * 100

/-- Theorem stating that the pandemic production cut percentage is 50% for the given factory conditions -/
theorem pandemic_cut_fifty_percent (factory : CarFactory) 
  (h1 : factory.doorsPerCar = 5)
  (h2 : factory.initialProduction = 200)
  (h3 : factory.metalShortageDecrease = 50)
  (h4 : factory.finalDoorProduction = 375) :
  pandemicProductionCutPercentage factory = 50 := by
  sorry

#eval pandemicProductionCutPercentage ⟨5, 200, 50, 375⟩

end NUMINAMATH_CALUDE_pandemic_cut_fifty_percent_l2509_250960


namespace NUMINAMATH_CALUDE_rectangle_max_area_l2509_250979

/-- The maximum area of a rectangle with perimeter 40 meters is 100 square meters. -/
theorem rectangle_max_area (x : ℝ) :
  let perimeter := 40
  let width := x
  let length := (perimeter / 2) - x
  let area := width * length
  (∀ y, 0 < y ∧ y < perimeter / 2 → area ≥ y * (perimeter / 2 - y)) →
  area ≤ 100 ∧ ∃ z, 0 < z ∧ z < perimeter / 2 ∧ z * (perimeter / 2 - z) = 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l2509_250979


namespace NUMINAMATH_CALUDE_ellipse_set_is_ellipse_l2509_250935

-- Define the space
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

-- Define points A and B
variable (A B : E)

-- Define the set of points P satisfying the condition
def ellipse_set (A B : E) : Set E :=
  {P : E | dist P A + dist P B = 2 * dist A B}

-- Theorem statement
theorem ellipse_set_is_ellipse (A B : E) (h : A ≠ B) :
  ∃ (C : E) (a b : ℝ), a > b ∧ b > 0 ∧
    ellipse_set A B = {P : E | (dist P C)^2 / a^2 + (dist P (C + (B - A)))^2 / b^2 = 1} :=
sorry

end NUMINAMATH_CALUDE_ellipse_set_is_ellipse_l2509_250935


namespace NUMINAMATH_CALUDE_modular_inverse_14_mod_1001_l2509_250923

theorem modular_inverse_14_mod_1001 :
  ∃ x : ℕ, x ≤ 1000 ∧ (14 * x) % 1001 = 1 :=
by
  use 143
  sorry

end NUMINAMATH_CALUDE_modular_inverse_14_mod_1001_l2509_250923


namespace NUMINAMATH_CALUDE_gcd_count_for_product_360_l2509_250959

theorem gcd_count_for_product_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ), S.card = 8 ∧ ∀ d, d ∈ S ↔ ∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 360) :=
sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_360_l2509_250959


namespace NUMINAMATH_CALUDE_negative_sum_distribution_l2509_250941

theorem negative_sum_distribution (x y : ℝ) : -(x + y) = -x - y := by
  sorry

end NUMINAMATH_CALUDE_negative_sum_distribution_l2509_250941


namespace NUMINAMATH_CALUDE_intersects_three_points_iff_m_range_l2509_250975

/-- A quadratic function f(x) = x^2 + 2x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

/-- Predicate indicating if f intersects the coordinate axes at 3 points -/
def intersects_at_three_points (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f m x₁ = 0 ∧ f m x₂ = 0 ∧ f m 0 ≠ 0

/-- Theorem stating the range of m for which f intersects the coordinate axes at 3 points -/
theorem intersects_three_points_iff_m_range (m : ℝ) :
  intersects_at_three_points m ↔ m < 1 ∧ m ≠ 0 := by sorry

end NUMINAMATH_CALUDE_intersects_three_points_iff_m_range_l2509_250975


namespace NUMINAMATH_CALUDE_ohara_triple_36_25_l2509_250970

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b x : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ x > 0 ∧ Real.sqrt a + Real.sqrt b = x

/-- Theorem: If (36,25,x) is an O'Hara triple, then x = 11 -/
theorem ohara_triple_36_25 (x : ℕ) :
  is_ohara_triple 36 25 x → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_ohara_triple_36_25_l2509_250970


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2509_250937

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n : ℕ, S (n + 2) - S (n + 1) = S (n + 1) - S n

/-- Theorem: If S_10 = 12 and S_20 = 17, then S_30 = 22 for an arithmetic sequence -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h1 : seq.S 10 = 12) (h2 : seq.S 20 = 17) : seq.S 30 = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2509_250937


namespace NUMINAMATH_CALUDE_consecutive_squares_difference_l2509_250946

theorem consecutive_squares_difference (t : ℕ) : 
  (t + 1)^2 - t^2 = 191 → t^2 = 9025 :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_difference_l2509_250946


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l2509_250930

/-- Given a cube with face perimeter of 20 cm, prove its volume is 125 cubic centimeters -/
theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 20) : 
  let side_length := face_perimeter / 4
  let volume := side_length ^ 3
  volume = 125 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l2509_250930


namespace NUMINAMATH_CALUDE_summer_mowing_count_l2509_250950

/-- The number of times Ned mowed his lawn in the spring -/
def spring_mows : ℕ := 6

/-- The total number of times Ned mowed his lawn -/
def total_mows : ℕ := 11

/-- The number of times Ned mowed his lawn in the summer -/
def summer_mows : ℕ := total_mows - spring_mows

theorem summer_mowing_count : summer_mows = 5 := by
  sorry

end NUMINAMATH_CALUDE_summer_mowing_count_l2509_250950


namespace NUMINAMATH_CALUDE_problem_solution_l2509_250933

theorem problem_solution (a b A : ℝ) 
  (h1 : 3^a = A) 
  (h2 : 5^b = A) 
  (h3 : 1/a + 1/b = 2) : 
  A = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2509_250933


namespace NUMINAMATH_CALUDE_inequality_solution_l2509_250914

def solution_set (a : ℝ) : Set ℝ :=
  if a > 1 then {x | x ≤ 1 ∨ x ≥ a}
  else if a = 1 then Set.univ
  else {x | x ≤ a ∨ x ≥ 1}

theorem inequality_solution (a : ℝ) :
  {x : ℝ | x^2 - (a + 1)*x + a ≥ 0} = solution_set a := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2509_250914


namespace NUMINAMATH_CALUDE_count_specific_divisors_l2509_250978

theorem count_specific_divisors (p q : ℕ+) : 
  let n := 2^(p : ℕ) * 3^(q : ℕ)
  (∃ (s : Finset ℕ), s.card = p * q ∧ 
    (∀ d ∈ s, d ∣ n^2 ∧ d < n ∧ ¬(d ∣ n))) :=
by sorry

end NUMINAMATH_CALUDE_count_specific_divisors_l2509_250978


namespace NUMINAMATH_CALUDE_girls_together_arrangements_boy_A_not_end_two_girls_together_arrangements_l2509_250967

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

end NUMINAMATH_CALUDE_girls_together_arrangements_boy_A_not_end_two_girls_together_arrangements_l2509_250967


namespace NUMINAMATH_CALUDE_trig_identity_l2509_250944

theorem trig_identity (α : Real) (h : Real.sin (2 * Real.pi / 3 + α) = 1 / 3) :
  Real.cos (5 * Real.pi / 6 - α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2509_250944


namespace NUMINAMATH_CALUDE_solution_set_t_3_nonnegative_for_all_x_l2509_250969

-- Define the function f
def f (t x : ℝ) : ℝ := x^2 - (t + 1)*x + t

-- Theorem 1: Solution set when t = 3
theorem solution_set_t_3 :
  {x : ℝ | f 3 x > 0} = Set.Iio 1 ∪ Set.Ioi 3 :=
sorry

-- Theorem 2: Condition for f(x) ≥ 0 for all real x
theorem nonnegative_for_all_x :
  (∀ x : ℝ, f t x ≥ 0) ↔ t = 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_t_3_nonnegative_for_all_x_l2509_250969


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_property_l2509_250961

/-- Given a cubic polynomial x^3 + ax^2 + bx + 16a where a and b are nonzero integers,
    if two of its roots coincide and all three roots are integers,
    then |ab| = 2496 -/
theorem cubic_polynomial_root_property (a b : ℤ) : 
  a ≠ 0 → b ≠ 0 → 
  (∃ r s : ℤ, (X - r)^2 * (X - s) = X^3 + a*X^2 + b*X + 16*a) →
  |a * b| = 2496 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_property_l2509_250961


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2509_250943

theorem gcd_of_specific_numbers : Nat.gcd 123456789 987654321 = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2509_250943


namespace NUMINAMATH_CALUDE_three_primes_product_sum_l2509_250938

theorem three_primes_product_sum : 
  ∃! (p q r : ℕ), 
    Prime p ∧ Prime q ∧ Prime r ∧ 
    p < q ∧ q < r ∧
    p * q * r = 5 * (p + q + r) ∧
    p = 2 ∧ q = 5 ∧ r = 7 := by
  sorry

end NUMINAMATH_CALUDE_three_primes_product_sum_l2509_250938


namespace NUMINAMATH_CALUDE_root_in_interval_l2509_250982

-- Define the function f(x) = 2x - 3
def f (x : ℝ) : ℝ := 2 * x - 3

-- State the theorem
theorem root_in_interval :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l2509_250982


namespace NUMINAMATH_CALUDE_inequality_solution_l2509_250964

theorem inequality_solution (n : ℕ+) : 2*n - 5 < 5 - 2*n ↔ n = 1 ∨ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2509_250964


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l2509_250939

theorem no_real_roots_quadratic : ∀ x : ℝ, x^2 + 2*x + 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l2509_250939


namespace NUMINAMATH_CALUDE_problem_solution_l2509_250973

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def sum_probability (n : ℕ) : ℚ := (3 : ℚ) / choose n 2

def binomial_coefficient (n k : ℕ) : ℤ := (choose n k : ℤ)

def a (n k : ℕ) : ℤ := binomial_coefficient n k * (-2)^k

theorem problem_solution :
  ∃ (n : ℕ),
    (sum_probability n = 3/28) ∧
    (a 8 3 = -448) ∧
    (((choose 5 2 * choose 4 1 + choose 4 3) : ℚ) / choose 9 3 = 11/21) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2509_250973


namespace NUMINAMATH_CALUDE_couplet_distribution_ways_l2509_250902

def num_widows : ℕ := 4
def num_long_couplets : ℕ := 4
def num_short_couplets : ℕ := 7

def long_couplets_per_widow : ℕ := 1
def short_couplets_for_one_widow : ℕ := 1
def short_couplets_for_three_widows : ℕ := 2

theorem couplet_distribution_ways :
  (Nat.choose num_long_couplets long_couplets_per_widow) *
  (Nat.choose num_short_couplets short_couplets_for_three_widows) *
  (Nat.choose (num_long_couplets - long_couplets_per_widow) long_couplets_per_widow) *
  (Nat.choose (num_short_couplets - short_couplets_for_three_widows) short_couplets_for_one_widow) *
  (Nat.choose (num_long_couplets - 2 * long_couplets_per_widow) long_couplets_per_widow) *
  (Nat.choose (num_short_couplets - short_couplets_for_three_widows - short_couplets_for_one_widow) short_couplets_for_three_widows) *
  (Nat.choose (num_long_couplets - 3 * long_couplets_per_widow) long_couplets_per_widow) *
  (Nat.choose (num_short_couplets - 2 * short_couplets_for_three_widows - short_couplets_for_one_widow) short_couplets_for_three_widows) = 15120 := by
  sorry

end NUMINAMATH_CALUDE_couplet_distribution_ways_l2509_250902


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l2509_250962

theorem solve_cubic_equation (y : ℝ) : 
  5 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + y^(1/3) → y = 1000 := by
  sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l2509_250962


namespace NUMINAMATH_CALUDE_garrison_provisions_duration_l2509_250901

/-- The number of days provisions last for a garrison given reinforcements --/
theorem garrison_provisions_duration 
  (initial_men : ℕ) 
  (reinforcement_men : ℕ) 
  (days_before_reinforcement : ℕ) 
  (days_after_reinforcement : ℕ) 
  (h1 : initial_men = 2000)
  (h2 : reinforcement_men = 1900)
  (h3 : days_before_reinforcement = 15)
  (h4 : days_after_reinforcement = 20) :
  ∃ (initial_days : ℕ), 
    initial_days * initial_men = 
      (initial_men + reinforcement_men) * days_after_reinforcement + 
      initial_men * days_before_reinforcement ∧
    initial_days = 54 := by
  sorry

end NUMINAMATH_CALUDE_garrison_provisions_duration_l2509_250901


namespace NUMINAMATH_CALUDE_max_y_value_l2509_250988

theorem max_y_value (a b y : ℝ) (eq1 : a + b + y = 5) (eq2 : a * b + b * y + a * y = 3) :
  y ≤ 13/3 := by
sorry

end NUMINAMATH_CALUDE_max_y_value_l2509_250988


namespace NUMINAMATH_CALUDE_max_sum_constrained_integers_l2509_250987

theorem max_sum_constrained_integers (a b c d e f g : ℕ) 
  (eq1 : a + b + c = 2)
  (eq2 : b + c + d = 2)
  (eq3 : c + d + e = 2)
  (eq4 : d + e + f = 2)
  (eq5 : e + f + g = 2) :
  a + b + c + d + e + f + g ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_constrained_integers_l2509_250987


namespace NUMINAMATH_CALUDE_units_digit_sum_base8_l2509_250907

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def base10_to_base8 (n : ℕ) : ℕ := sorry

/-- Returns the units digit of a base-8 number --/
def units_digit_base8 (n : ℕ) : ℕ := sorry

theorem units_digit_sum_base8 : 
  units_digit_base8 (base10_to_base8 (base8_to_base10 53 + base8_to_base10 64)) = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_base8_l2509_250907


namespace NUMINAMATH_CALUDE_function_symmetry_l2509_250974

def is_symmetric_about_one (g : ℝ → ℝ) : Prop :=
  ∀ x, g (1 - x) = g (1 + x)

theorem function_symmetry 
  (f : ℝ → ℝ) 
  (h1 : f 0 = 0)
  (h2 : ∀ x, f (-x) = f x)
  (h3 : ∀ t, f (1 - t) - f (1 + t) + 4 * t = 0) :
  is_symmetric_about_one (λ x => f x - 2 * x) := by
sorry

end NUMINAMATH_CALUDE_function_symmetry_l2509_250974


namespace NUMINAMATH_CALUDE_max_area_rectangle_l2509_250905

/-- The maximum area of a rectangle with a perimeter of 40 inches is 100 square inches. -/
theorem max_area_rectangle (x y : ℝ) (h_perimeter : x + y = 20) :
  x * y ≤ 100 ∧ ∃ (a b : ℝ), a + b = 20 ∧ a * b = 100 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l2509_250905


namespace NUMINAMATH_CALUDE_largest_x_value_l2509_250953

theorem largest_x_value (x : ℝ) :
  (x / 3 + 1 / (7 * x) = 1 / 2) →
  x ≤ (21 + Real.sqrt 105) / 28 ∧
  ∃ y : ℝ, y / 3 + 1 / (7 * y) = 1 / 2 ∧ y = (21 + Real.sqrt 105) / 28 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l2509_250953


namespace NUMINAMATH_CALUDE_alyssa_pears_l2509_250904

theorem alyssa_pears (total_pears nancy_pears : ℕ) 
  (h1 : total_pears = 59) 
  (h2 : nancy_pears = 17) : 
  total_pears - nancy_pears = 42 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_pears_l2509_250904


namespace NUMINAMATH_CALUDE_variance_transformation_l2509_250955

/-- Given three real numbers with variance 1, prove that multiplying each by 3 and adding 2 results in a variance of 9. -/
theorem variance_transformation (a₁ a₂ a₃ : ℝ) (μ : ℝ) : 
  (1 / 3 : ℝ) * ((a₁ - μ)^2 + (a₂ - μ)^2 + (a₃ - μ)^2) = 1 →
  (1 / 3 : ℝ) * (((3 * a₁ + 2) - (3 * μ + 2))^2 + 
                 ((3 * a₂ + 2) - (3 * μ + 2))^2 + 
                 ((3 * a₃ + 2) - (3 * μ + 2))^2) = 9 :=
by sorry

end NUMINAMATH_CALUDE_variance_transformation_l2509_250955


namespace NUMINAMATH_CALUDE_train_length_l2509_250984

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) : 
  speed_kmh = 75.6 → time_sec = 21 → speed_kmh * (1000 / 3600) * time_sec = 441 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2509_250984


namespace NUMINAMATH_CALUDE_append_digit_square_difference_l2509_250983

theorem append_digit_square_difference (x y : ℕ) : 
  x > 0 → y ≤ 9 → (10 * x + y - x^2 = 8 * x) → 
  ((x = 2 ∧ y = 0) ∨ (x = 3 ∧ y = 3) ∨ (x = 4 ∧ y = 8)) := by
  sorry

end NUMINAMATH_CALUDE_append_digit_square_difference_l2509_250983


namespace NUMINAMATH_CALUDE_unique_root_condition_l2509_250949

theorem unique_root_condition (a : ℝ) : 
  (∃! x, Real.log (x - 2*a) - 3*(x - 2*a)^2 + 2*a = 0) ↔ 
  (a = (Real.log 6 + 1) / 4) := by sorry

end NUMINAMATH_CALUDE_unique_root_condition_l2509_250949


namespace NUMINAMATH_CALUDE_polynomial_non_negative_l2509_250997

theorem polynomial_non_negative (x : ℝ) : x^12 - x^7 - x^5 + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_non_negative_l2509_250997


namespace NUMINAMATH_CALUDE_coefficient_of_x_in_expansion_l2509_250912

/-- The coefficient of x in the expansion of (1 + √x)^6 * (1 + √x)^4 -/
def coefficient_of_x : ℕ := 45

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

theorem coefficient_of_x_in_expansion :
  coefficient_of_x = 
    binomial 4 2 + binomial 6 2 + binomial 6 1 * binomial 4 1 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_in_expansion_l2509_250912


namespace NUMINAMATH_CALUDE_trajectory_of_point_P_l2509_250958

/-- The trajectory of point P satisfying given conditions -/
theorem trajectory_of_point_P (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (x - a) * b = y * a ∧  -- P lies on line AB
    (x - 0)^2 + (y - b)^2 = 4 * ((a - x)^2 + y^2) ∧  -- BP = 2PA
    (-x) * (-a) + y * b = 1)  -- OQ · AB = 1
  → 3/2 * x^2 + 3 * y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_point_P_l2509_250958


namespace NUMINAMATH_CALUDE_jellybean_problem_l2509_250957

theorem jellybean_problem (initial_quantity : ℝ) : 
  (initial_quantity * (1 - 0.3)^4 = 48) → initial_quantity = 200 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l2509_250957


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_l2509_250971

theorem chinese_remainder_theorem (x : ℤ) :
  (2 + x) % (2^4) = 3^2 % (2^4) ∧
  (3 + x) % (3^4) = 2^3 % (3^4) ∧
  (5 + x) % (5^4) = 7^2 % (5^4) →
  x % 30 = 14 := by
sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_l2509_250971


namespace NUMINAMATH_CALUDE_seven_twentyfour_twentyfive_pythagorean_triple_l2509_250990

/-- A Pythagorean triple consists of three positive integers a, b, and c that satisfy a² + b² = c² -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- Prove that 7, 24, and 25 form a Pythagorean triple -/
theorem seven_twentyfour_twentyfive_pythagorean_triple :
  is_pythagorean_triple 7 24 25 := by
sorry

end NUMINAMATH_CALUDE_seven_twentyfour_twentyfive_pythagorean_triple_l2509_250990


namespace NUMINAMATH_CALUDE_probability_diamond_ace_face_card_l2509_250945

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (favorable_cards : ℕ)
  (h_total : total_cards = 54)
  (h_favorable : favorable_cards = 26)

/-- The probability of selecting at least one favorable card in two draws with replacement -/
def probability_favorable_card (d : Deck) : ℚ :=
  1 - (↑(d.total_cards - d.favorable_cards) / ↑d.total_cards) ^ 2

theorem probability_diamond_ace_face_card :
  ∃ d : Deck, probability_favorable_card d = 533 / 729 := by
  sorry

end NUMINAMATH_CALUDE_probability_diamond_ace_face_card_l2509_250945


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l2509_250976

/-- Proves the range of m for which x^2 - 3x - m = 0 has two unequal real roots -/
theorem quadratic_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 3*x - m = 0 ∧ y^2 - 3*y - m = 0) ↔ m > -9/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l2509_250976


namespace NUMINAMATH_CALUDE_solution_to_equation_l2509_250934

theorem solution_to_equation : ∃ x y : ℝ, x + 2 * y = 4 ∧ x = 0 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2509_250934


namespace NUMINAMATH_CALUDE_same_color_probability_l2509_250915

theorem same_color_probability (N : ℕ) : 
  (4 : ℚ) / 10 * 16 / (16 + N) + (6 : ℚ) / 10 * N / (16 + N) = 29 / 50 → N = 144 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2509_250915


namespace NUMINAMATH_CALUDE_real_part_of_complex_number_l2509_250952

theorem real_part_of_complex_number (z : ℂ) (h : z - Complex.abs z = -8 + 12*I) : 
  Complex.re z = 5 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_number_l2509_250952


namespace NUMINAMATH_CALUDE_tower_house_block_difference_l2509_250998

def blocks_for_tower : ℕ := 50
def blocks_for_house : ℕ := 20

theorem tower_house_block_difference :
  blocks_for_tower - blocks_for_house = 30 :=
by sorry

end NUMINAMATH_CALUDE_tower_house_block_difference_l2509_250998


namespace NUMINAMATH_CALUDE_circumscribed_quadrilateral_arc_angles_l2509_250992

theorem circumscribed_quadrilateral_arc_angles (a b c d : ℝ) :
  let x := (b + c + d) / 2
  let y := (a + c + d) / 2
  let z := (a + b + d) / 2
  let t := (a + b + c) / 2
  a + b + c + d = 360 →
  x + y + z + t = 540 := by
sorry

end NUMINAMATH_CALUDE_circumscribed_quadrilateral_arc_angles_l2509_250992


namespace NUMINAMATH_CALUDE_value_of_expression_l2509_250980

theorem value_of_expression (m n : ℝ) 
  (h1 : m^2 + 2*m*n = 3) 
  (h2 : m*n + n^2 = 4) : 
  m^2 + 3*m*n + n^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2509_250980


namespace NUMINAMATH_CALUDE_initial_average_age_proof_l2509_250948

/-- Proves that the initial average age of a group is 16 years, given the conditions of the problem -/
theorem initial_average_age_proof (initial_count : ℕ) (new_count : ℕ) (new_avg_age : ℚ) (final_avg_age : ℚ) :
  initial_count = 12 →
  new_count = 12 →
  new_avg_age = 15 →
  final_avg_age = 15.5 →
  (initial_count * (initial_count * final_avg_age - new_count * new_avg_age) / (initial_count * initial_count)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_age_proof_l2509_250948
