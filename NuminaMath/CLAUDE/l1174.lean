import Mathlib

namespace NUMINAMATH_CALUDE_no_solution_to_system_l1174_117465

theorem no_solution_to_system : ¬∃ x : ℝ, (Real.arccos (Real.cos x) = x / 3) ∧ (Real.sin x = Real.cos (x / 3)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_system_l1174_117465


namespace NUMINAMATH_CALUDE_triangle_base_length_l1174_117468

theorem triangle_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (h1 : area = 25) 
  (h2 : height = 5) : 
  area = (base * height) / 2 → base = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l1174_117468


namespace NUMINAMATH_CALUDE_wire_cut_ratio_l1174_117420

/-- Given a wire cut into two pieces of lengths a and b, where piece a forms a rectangle
    with length twice its width and piece b forms a circle, if the areas of the rectangle
    and circle are equal, then a/b = 3/√(2π). -/
theorem wire_cut_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (∃ x : ℝ, a = 6 * x ∧ 2 * x^2 = π * (b / (2 * π))^2) →
  a / b = 3 / Real.sqrt (2 * π) := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_ratio_l1174_117420


namespace NUMINAMATH_CALUDE_expression_bounds_l1174_117490

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) :
  4 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
    Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ∧
  Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
    Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l1174_117490


namespace NUMINAMATH_CALUDE_lcm_of_primes_l1174_117400

theorem lcm_of_primes (x y : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) 
  (hxy : x > y) (heq : 2 * x + y = 12) : 
  Nat.lcm x y = 10 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_primes_l1174_117400


namespace NUMINAMATH_CALUDE_negative_two_cubed_equality_l1174_117441

theorem negative_two_cubed_equality : -2^3 = (-2)^3 := by sorry

end NUMINAMATH_CALUDE_negative_two_cubed_equality_l1174_117441


namespace NUMINAMATH_CALUDE_not_p_and_q_implies_at_most_one_true_l1174_117456

theorem not_p_and_q_implies_at_most_one_true (p q : Prop) :
  ¬(p ∧ q) → (¬p ∨ ¬q) :=
by
  sorry

end NUMINAMATH_CALUDE_not_p_and_q_implies_at_most_one_true_l1174_117456


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l1174_117435

theorem smallest_n_for_candy_purchase : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → (∃ (m : ℕ), m > 0 ∧ 
    9 ∣ m ∧ 10 ∣ m ∧ 20 ∣ m ∧ m = 30 * k) → n ≤ k) ∧
  (∃ (m : ℕ), m > 0 ∧ 9 ∣ m ∧ 10 ∣ m ∧ 20 ∣ m ∧ m = 30 * n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l1174_117435


namespace NUMINAMATH_CALUDE_min_value_of_max_abs_min_value_achievable_l1174_117406

theorem min_value_of_max_abs (a b : ℝ) : 
  max (abs (a + b)) (max (abs (a - b)) (abs (1 - b))) ≥ (1 : ℝ) / 2 := by
  sorry

theorem min_value_achievable : 
  ∃ (a b : ℝ), max (abs (a + b)) (max (abs (a - b)) (abs (1 - b))) = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_max_abs_min_value_achievable_l1174_117406


namespace NUMINAMATH_CALUDE_problem_solution_l1174_117478

theorem problem_solution : ∀ A B Y : ℤ,
  A = 3009 / 3 →
  B = A / 3 →
  Y = A - B →
  Y = 669 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1174_117478


namespace NUMINAMATH_CALUDE_ababab_divisible_by_101_l1174_117495

/-- Represents a 6-digit number of the form ababab -/
def ababab_number (a b : Nat) : Nat :=
  100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b

/-- Theorem stating that 101 is a factor of any ababab number -/
theorem ababab_divisible_by_101 (a b : Nat) (h : 0 < a ∧ a ≤ 9 ∧ b ≤ 9) :
  101 ∣ ababab_number a b :=
sorry

end NUMINAMATH_CALUDE_ababab_divisible_by_101_l1174_117495


namespace NUMINAMATH_CALUDE_blast_distance_problem_l1174_117466

/-- The distance traveled by sound in a given time -/
def sound_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- The problem statement -/
theorem blast_distance_problem (man_speed : ℝ) (sound_speed : ℝ) (total_time : ℝ) (blast_interval : ℝ) :
  sound_speed = 330 →
  total_time = 30 * 60 + 12 →
  blast_interval = 30 * 60 →
  sound_distance sound_speed (total_time - blast_interval) = 3960 := by
  sorry

end NUMINAMATH_CALUDE_blast_distance_problem_l1174_117466


namespace NUMINAMATH_CALUDE_probability_of_specific_arrangement_l1174_117442

def total_tiles : ℕ := 6
def x_tiles : ℕ := 4
def o_tiles : ℕ := 2

theorem probability_of_specific_arrangement :
  (1 : ℚ) / (Nat.choose total_tiles x_tiles) = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_arrangement_l1174_117442


namespace NUMINAMATH_CALUDE_practice_time_proof_l1174_117488

/-- Calculates the required practice time for Friday given the practice times for Monday to Thursday and the total required practice time for the week. -/
def friday_practice_time (monday tuesday wednesday thursday total_time : ℕ) : ℕ :=
  total_time - (monday + tuesday + wednesday + thursday)

/-- Theorem stating that given the practice times for Monday to Thursday and the total required practice time, the remaining time for Friday is 60 minutes. -/
theorem practice_time_proof (total_time : ℕ) (h1 : total_time = 300) 
  (thursday : ℕ) (h2 : thursday = 50)
  (wednesday : ℕ) (h3 : wednesday = thursday + 5)
  (tuesday : ℕ) (h4 : tuesday = wednesday - 10)
  (monday : ℕ) (h5 : monday = 2 * tuesday) :
  friday_practice_time monday tuesday wednesday thursday total_time = 60 := by
  sorry

#eval friday_practice_time 90 45 55 50 300

end NUMINAMATH_CALUDE_practice_time_proof_l1174_117488


namespace NUMINAMATH_CALUDE_fraction_simplification_l1174_117464

theorem fraction_simplification : (4 : ℚ) / (2 - 4 / 5) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1174_117464


namespace NUMINAMATH_CALUDE_q_factor_change_l1174_117487

theorem q_factor_change (e x z : ℝ) (h : x ≠ 0 ∧ z ≠ 0) :
  let q := 5 * e / (4 * x * z^2)
  let q_new := 5 * (4 * e) / (4 * (2 * x) * (3 * z)^2)
  q_new = (4 / 9) * q :=
by
  sorry

end NUMINAMATH_CALUDE_q_factor_change_l1174_117487


namespace NUMINAMATH_CALUDE_min_value_theorem_l1174_117469

theorem min_value_theorem (m n : ℝ) (h1 : 2 * n + m = 4) (h2 : m > 0) (h3 : n > 0) :
  2 / m + 1 / n ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1174_117469


namespace NUMINAMATH_CALUDE_hydrogen_oxygen_reaction_certain_l1174_117421

/-- Represents the certainty of an event --/
inductive EventCertainty
  | Possible
  | Impossible
  | Certain

/-- Represents a chemical reaction --/
structure ChemicalReaction where
  reactants : List String
  products : List String

/-- The chemical reaction of hydrogen burning in oxygen to form water --/
def hydrogenOxygenReaction : ChemicalReaction :=
  { reactants := ["Hydrogen", "Oxygen"],
    products := ["Water"] }

/-- Theorem stating that the hydrogen-oxygen reaction is certain --/
theorem hydrogen_oxygen_reaction_certain :
  (hydrogenOxygenReaction.reactants = ["Hydrogen", "Oxygen"] ∧
   hydrogenOxygenReaction.products = ["Water"]) →
  EventCertainty.Certain = 
    match hydrogenOxygenReaction with
    | { reactants := ["Hydrogen", "Oxygen"], products := ["Water"] } => EventCertainty.Certain
    | _ => EventCertainty.Possible
  := by sorry

end NUMINAMATH_CALUDE_hydrogen_oxygen_reaction_certain_l1174_117421


namespace NUMINAMATH_CALUDE_mike_weekly_pullups_l1174_117407

/-- The number of pull-ups Mike does in a week -/
def weekly_pullups (pullups_per_visit : ℕ) (visits_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  pullups_per_visit * visits_per_day * days_per_week

/-- Theorem stating that Mike does 70 pull-ups in a week -/
theorem mike_weekly_pullups :
  weekly_pullups 2 5 7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_mike_weekly_pullups_l1174_117407


namespace NUMINAMATH_CALUDE_min_value_of_f_l1174_117403

/-- The function f(x, y) as defined in the problem -/
def f (x y : ℝ) : ℝ := 6 * (x^2 + y^2) * (x + y) - 4 * (x^2 + x*y + y^2) - 3 * (x + y) + 5

/-- The theorem statement -/
theorem min_value_of_f :
  (∀ x y : ℝ, x > 0 → y > 0 → f x y ≥ 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ f x y = 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1174_117403


namespace NUMINAMATH_CALUDE_seventh_twentyninth_150th_digit_l1174_117447

/-- The decimal expansion of 7/29 has a repeating block of length 28 -/
def decimal_period : ℕ := 28

/-- The repeating block in the decimal expansion of 7/29 -/
def repeating_block : List ℕ := [2, 4, 1, 3, 7, 9, 3, 1, 0, 3, 4, 4, 8, 2, 7, 5, 8, 6, 2, 0, 6, 8, 9, 6, 5, 5, 1, 7, 2]

/-- The 150th digit after the decimal point in the decimal expansion of 7/29 -/
def digit_150 : ℕ := repeating_block[(150 - 1) % decimal_period]

theorem seventh_twentyninth_150th_digit :
  digit_150 = 8 := by sorry

end NUMINAMATH_CALUDE_seventh_twentyninth_150th_digit_l1174_117447


namespace NUMINAMATH_CALUDE_larger_number_proof_l1174_117444

theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 40) →
  (Nat.lcm a b = 6600) →
  ((a = 40 * 11 ∧ b = 40 * 15) ∨ (a = 40 * 15 ∧ b = 40 * 11)) →
  max a b = 600 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1174_117444


namespace NUMINAMATH_CALUDE_division_theorem_specific_case_l1174_117450

theorem division_theorem_specific_case :
  ∀ (D d Q R : ℕ),
    D = d * Q + R →
    d * Q = 135 →
    R = 2 * d →
    R < d →
    Q > 0 →
    D = 165 ∧ d = 15 ∧ Q = 9 ∧ R = 30 :=
by sorry

end NUMINAMATH_CALUDE_division_theorem_specific_case_l1174_117450


namespace NUMINAMATH_CALUDE_initial_strawberry_plants_l1174_117460

/-- The number of strawberry plants after n months of doubling -/
def plants_after_months (initial : ℕ) (months : ℕ) : ℕ :=
  initial * (2 ^ months)

/-- The theorem stating the initial number of strawberry plants -/
theorem initial_strawberry_plants : ∃ (initial : ℕ), 
  plants_after_months initial 3 - 4 = 20 ∧ initial > 0 := by
  sorry

end NUMINAMATH_CALUDE_initial_strawberry_plants_l1174_117460


namespace NUMINAMATH_CALUDE_compound_interest_rate_interest_rate_problem_l1174_117414

/-- Compound interest calculation --/
theorem compound_interest_rate (P A : ℝ) (t n : ℕ) (h : A = P * (1 + 0.25)^(n * t)) :
  ∃ (r : ℝ), A = P * (1 + r)^(n * t) ∧ r = 0.25 :=
by sorry

/-- Problem-specific theorem --/
theorem interest_rate_problem (P A : ℝ) (t n : ℕ) 
  (h_P : P = 1200)
  (h_A : A = 2488.32)
  (h_t : t = 4)
  (h_n : n = 1) :
  ∃ (r : ℝ), A = P * (1 + r)^(n * t) ∧ r = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_rate_interest_rate_problem_l1174_117414


namespace NUMINAMATH_CALUDE_seventeen_students_earlier_l1174_117483

/-- The number of students who came earlier than Hoseok, given the total number of students and the number of students who came later. -/
def students_earlier (total : ℕ) (later : ℕ) : ℕ :=
  total - later - 1

/-- Theorem stating that 17 students came earlier than Hoseok. -/
theorem seventeen_students_earlier :
  students_earlier 30 12 = 17 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_students_earlier_l1174_117483


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1174_117485

def original_expression (y : ℝ) : ℝ := 3 * (y^3 - 2*y^2 + 3) - 5 * (y^2 - 4*y + 2)

def simplified_expression (y : ℝ) : ℝ := 3*y^3 - 11*y^2 + 20*y - 1

theorem sum_of_squared_coefficients :
  (3^2 + (-11)^2 + 20^2 + (-1)^2 = 531) ∧
  (∀ y : ℝ, original_expression y = simplified_expression y) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1174_117485


namespace NUMINAMATH_CALUDE_percentage_reduction_l1174_117437

theorem percentage_reduction (initial : ℝ) (increase_percent : ℝ) (final : ℝ) : 
  initial = 1500 →
  increase_percent = 20 →
  final = 1080 →
  let increased := initial * (1 + increase_percent / 100)
  (increased - final) / increased * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_reduction_l1174_117437


namespace NUMINAMATH_CALUDE_unique_a_value_l1174_117440

/-- A function to represent the exponent |a-2| --/
def abs_a_minus_2 (a : ℝ) : ℝ := |a - 2|

/-- The coefficient of x in the equation --/
def coeff_x (a : ℝ) : ℝ := a - 3

/-- Predicate to check if the equation is linear --/
def is_linear (a : ℝ) : Prop := abs_a_minus_2 a = 1

/-- Theorem stating that a = 1 is the only value satisfying the conditions --/
theorem unique_a_value : ∃! a : ℝ, is_linear a ∧ coeff_x a ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l1174_117440


namespace NUMINAMATH_CALUDE_angie_tax_payment_l1174_117433

/-- Represents Angie's monthly finances -/
structure AngieFinances where
  salary : ℕ
  necessities : ℕ
  leftOver : ℕ

/-- Calculates Angie's tax payment based on her finances -/
def taxPayment (finances : AngieFinances) : ℕ :=
  finances.salary - finances.necessities - finances.leftOver

/-- Theorem stating that Angie's tax payment is $20 given her financial situation -/
theorem angie_tax_payment :
  let finances : AngieFinances := { salary := 80, necessities := 42, leftOver := 18 }
  taxPayment finances = 20 := by
  sorry


end NUMINAMATH_CALUDE_angie_tax_payment_l1174_117433


namespace NUMINAMATH_CALUDE_max_acute_angles_is_four_l1174_117415

/-- An octagon is a polygon with 8 sides -/
def Octagon : Type := Unit

/-- A convex octagon is an octagon where all interior angles are less than 180° -/
def ConvexOctagon : Type := Octagon

/-- The sum of interior angles of a convex octagon -/
def sum_interior_angles (o : ConvexOctagon) : ℝ := 1080

/-- An angle is acute if it is less than 90° -/
def is_acute (angle : ℝ) : Prop := angle < 90

/-- The number of acute angles in a convex octagon -/
def num_acute_angles (o : ConvexOctagon) : ℕ := sorry

/-- The maximum number of acute angles in any convex octagon -/
def max_acute_angles : ℕ := sorry

theorem max_acute_angles_is_four :
  max_acute_angles = 4 := by sorry

end NUMINAMATH_CALUDE_max_acute_angles_is_four_l1174_117415


namespace NUMINAMATH_CALUDE_distribute_graduates_eq_90_l1174_117443

/-- The number of ways to evenly distribute 6 graduates to 3 schools -/
def distribute_graduates : ℕ :=
  Nat.choose 6 2 * Nat.choose 4 2 * Nat.choose 2 2

/-- Theorem stating that the number of ways to distribute graduates is 90 -/
theorem distribute_graduates_eq_90 : distribute_graduates = 90 := by
  sorry

end NUMINAMATH_CALUDE_distribute_graduates_eq_90_l1174_117443


namespace NUMINAMATH_CALUDE_cos_three_pi_fourth_plus_two_alpha_l1174_117425

theorem cos_three_pi_fourth_plus_two_alpha (α : ℝ) 
  (h : Real.cos (π / 8 - α) = 1 / 6) : 
  Real.cos (3 * π / 4 + 2 * α) = 17 / 18 := by
  sorry

end NUMINAMATH_CALUDE_cos_three_pi_fourth_plus_two_alpha_l1174_117425


namespace NUMINAMATH_CALUDE_mary_found_four_eggs_l1174_117484

/-- The number of eggs Mary started with -/
def initial_eggs : ℕ := 27

/-- The number of eggs Mary ended up with -/
def final_eggs : ℕ := 31

/-- The number of eggs Mary found -/
def found_eggs : ℕ := final_eggs - initial_eggs

theorem mary_found_four_eggs : found_eggs = 4 := by
  sorry

end NUMINAMATH_CALUDE_mary_found_four_eggs_l1174_117484


namespace NUMINAMATH_CALUDE_seating_arrangements_l1174_117429

/-- The number of ways to seat n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange a block of k people within a group of n people -/
def blockArrangements (n k : ℕ) : ℕ := (Nat.factorial n) * (Nat.factorial k)

/-- The number of valid seating arrangements for n people, 
    where k specific people cannot sit in k consecutive seats -/
def validArrangements (n k : ℕ) : ℕ := 
  totalArrangements n - blockArrangements (n - k + 1) k

theorem seating_arrangements : 
  validArrangements 10 4 = 3507840 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1174_117429


namespace NUMINAMATH_CALUDE_sum_of_consecutive_naturals_with_lcm_168_l1174_117451

def consecutive_naturals (n : ℕ) : Fin 3 → ℕ := λ i => n + i.val

theorem sum_of_consecutive_naturals_with_lcm_168 :
  ∃ n : ℕ, (Nat.lcm (consecutive_naturals n 0) (Nat.lcm (consecutive_naturals n 1) (consecutive_naturals n 2)) = 168) ∧
  (consecutive_naturals n 0 + consecutive_naturals n 1 + consecutive_naturals n 2 = 21) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_naturals_with_lcm_168_l1174_117451


namespace NUMINAMATH_CALUDE_savings_calculation_l1174_117408

def folder_price : ℝ := 2.50
def num_folders : ℕ := 5
def discount_rate : ℝ := 0.20

theorem savings_calculation :
  let original_total := folder_price * num_folders
  let discounted_total := original_total * (1 - discount_rate)
  original_total - discounted_total = 2.50 := by
sorry

end NUMINAMATH_CALUDE_savings_calculation_l1174_117408


namespace NUMINAMATH_CALUDE_cubic_root_sum_square_l1174_117455

theorem cubic_root_sum_square (p q r t : ℝ) : 
  (p^3 - 6*p^2 + 8*p - 1 = 0) →
  (q^3 - 6*q^2 + 8*q - 1 = 0) →
  (r^3 - 6*r^2 + 8*r - 1 = 0) →
  (t = Real.sqrt p + Real.sqrt q + Real.sqrt r) →
  (t^4 - 12*t^2 - 8*t = -4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_square_l1174_117455


namespace NUMINAMATH_CALUDE_smallest_n_for_cube_sum_inequality_l1174_117419

theorem smallest_n_for_cube_sum_inequality : 
  ∃ n : ℕ, (∀ x y z : ℝ, (x^3 + y^3 + z^3)^2 ≤ n * (x^6 + y^6 + z^6)) ∧ 
  (∀ m : ℕ, m < n → ∃ x y z : ℝ, (x^3 + y^3 + z^3)^2 > m * (x^6 + y^6 + z^6)) ∧
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_cube_sum_inequality_l1174_117419


namespace NUMINAMATH_CALUDE_median_in_60_64_interval_l1174_117439

/-- Represents a score interval with its frequency -/
structure ScoreInterval :=
  (lowerBound upperBound : ℕ)
  (frequency : ℕ)

/-- The problem setup -/
def testScores : List ScoreInterval :=
  [ ⟨45, 49, 8⟩
  , ⟨50, 54, 15⟩
  , ⟨55, 59, 20⟩
  , ⟨60, 64, 18⟩
  , ⟨65, 69, 17⟩
  , ⟨70, 74, 12⟩
  , ⟨75, 79, 9⟩
  , ⟨80, 84, 6⟩
  ]

def totalStudents : ℕ := 105

/-- The median interval is the one containing the (n+1)/2 th student -/
def medianPosition : ℕ := (totalStudents + 1) / 2

/-- Function to find the interval containing the median -/
def findMedianInterval (scores : List ScoreInterval) (medianPos : ℕ) : Option ScoreInterval :=
  let rec go (acc : ℕ) (remaining : List ScoreInterval) : Option ScoreInterval :=
    match remaining with
    | [] => none
    | interval :: rest =>
      let newAcc := acc + interval.frequency
      if newAcc ≥ medianPos then some interval
      else go newAcc rest
  go 0 scores

/-- Theorem stating that the median score is in the interval 60-64 -/
theorem median_in_60_64_interval :
  findMedianInterval testScores medianPosition = some ⟨60, 64, 18⟩ := by
  sorry


end NUMINAMATH_CALUDE_median_in_60_64_interval_l1174_117439


namespace NUMINAMATH_CALUDE_sin_50_sin_70_minus_cos_50_sin_20_l1174_117499

open Real

theorem sin_50_sin_70_minus_cos_50_sin_20 :
  sin (50 * π / 180) * sin (70 * π / 180) - cos (50 * π / 180) * sin (20 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_50_sin_70_minus_cos_50_sin_20_l1174_117499


namespace NUMINAMATH_CALUDE_positive_expression_l1174_117452

theorem positive_expression (x : ℝ) : x^2 * Real.sin x + x * Real.cos x + x^2 + (1/2 : ℝ) > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l1174_117452


namespace NUMINAMATH_CALUDE_final_eraser_count_l1174_117470

def initial_erasers : Float := 95.0
def bought_erasers : Float := 42.0

theorem final_eraser_count :
  initial_erasers + bought_erasers = 137.0 := by
  sorry

end NUMINAMATH_CALUDE_final_eraser_count_l1174_117470


namespace NUMINAMATH_CALUDE_symmetry_y_axis_l1174_117445

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetricYAxis (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := -p.z }

/-- The theorem stating that (-2, -1, -4) is symmetrical to (2, -1, 4) with respect to the y-axis -/
theorem symmetry_y_axis :
  let P : Point3D := { x := 2, y := -1, z := 4 }
  let Q : Point3D := { x := -2, y := -1, z := -4 }
  symmetricYAxis P = Q := by sorry

end NUMINAMATH_CALUDE_symmetry_y_axis_l1174_117445


namespace NUMINAMATH_CALUDE_unique_prime_product_perfect_power_l1174_117427

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The product of the first k prime numbers -/
def primeProduct (k : ℕ) : ℕ := sorry

/-- A number is a perfect power if it can be expressed as a^n where a > 1 and n > 1 -/
def isPerfectPower (m : ℕ) : Prop := sorry

theorem unique_prime_product_perfect_power :
  ∀ k : ℕ, (k ≠ 0 ∧ isPerfectPower (primeProduct k - 1)) ↔ k = 1 := by sorry

end NUMINAMATH_CALUDE_unique_prime_product_perfect_power_l1174_117427


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l1174_117411

theorem imaginary_part_of_complex_expression :
  let z : ℂ := (1 + I) / (1 - I) + (1 - I)
  Complex.im z = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l1174_117411


namespace NUMINAMATH_CALUDE_not_in_fourth_quadrant_l1174_117413

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the fourth quadrant
def fourth_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

-- Theorem statement
theorem not_in_fourth_quadrant (m : ℝ) :
  ¬(fourth_quadrant ⟨m - 2, m + 1⟩) := by
  sorry

end NUMINAMATH_CALUDE_not_in_fourth_quadrant_l1174_117413


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_perimeter_is_22_l1174_117459

/-- An isosceles triangle with two sides of length 9 and one side of length 4 has a perimeter of 22 -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∀ a b c : ℝ,
      a = 9 ∧ b = 9 ∧ c = 4 →
      a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
      a = b →  -- Isosceles condition
      perimeter = a + b + c

/-- The perimeter of the isosceles triangle is 22 -/
theorem perimeter_is_22 : isosceles_triangle_perimeter 22 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_perimeter_is_22_l1174_117459


namespace NUMINAMATH_CALUDE_factor_problem_l1174_117422

theorem factor_problem (n : ℤ) (f : ℚ) (h1 : n = 9) (h2 : (n + 2) * f = 24 + n) : f = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_problem_l1174_117422


namespace NUMINAMATH_CALUDE_dog_groupings_count_l1174_117432

/-- The number of ways to divide 12 dogs into three groups -/
def dog_groupings : ℕ :=
  let total_dogs : ℕ := 12
  let group1_size : ℕ := 4  -- Fluffy's group
  let group2_size : ℕ := 5  -- Nipper's group
  let group3_size : ℕ := 3
  let remaining_dogs : ℕ := total_dogs - 2  -- Excluding Fluffy and Nipper
  (Nat.choose remaining_dogs (group1_size - 1)) * (Nat.choose (remaining_dogs - (group1_size - 1)) (group2_size - 1))

/-- Theorem stating the number of ways to divide the dogs is 4200 -/
theorem dog_groupings_count : dog_groupings = 4200 := by
  sorry

end NUMINAMATH_CALUDE_dog_groupings_count_l1174_117432


namespace NUMINAMATH_CALUDE_circle_on_parabola_fixed_point_l1174_117453

/-- A circle with center on a parabola and tangent to a line passes through a fixed point -/
theorem circle_on_parabola_fixed_point (h k : ℝ) :
  k = (1/12) * h^2 →  -- Center (h, k) lies on the parabola y = (1/12)x^2
  (k + 3)^2 = h^2 + (k - 3)^2 →  -- Circle is tangent to the line y + 3 = 0
  (0 - h)^2 + (3 - k)^2 = (k + 3)^2 :=  -- Point (0, 3) lies on the circle
by sorry

end NUMINAMATH_CALUDE_circle_on_parabola_fixed_point_l1174_117453


namespace NUMINAMATH_CALUDE_unique_triple_l1174_117463

theorem unique_triple : 
  ∃! m p q : ℕ, 
    m > 0 ∧ 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    2^m * p^2 + 1 = q^5 ∧
    m = 1 ∧ p = 11 ∧ q = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_l1174_117463


namespace NUMINAMATH_CALUDE_village_population_equality_l1174_117428

/-- The number of years after which the populations are equal -/
def years : ℕ := 14

/-- The rate of population decrease per year for the first village -/
def decrease_rate : ℕ := 1200

/-- The rate of population increase per year for the second village -/
def increase_rate : ℕ := 800

/-- The initial population of the second village -/
def initial_population_second : ℕ := 42000

/-- The initial population of the first village -/
def initial_population_first : ℕ := 70000

theorem village_population_equality :
  initial_population_first - years * decrease_rate = 
  initial_population_second + years * increase_rate :=
by sorry

end NUMINAMATH_CALUDE_village_population_equality_l1174_117428


namespace NUMINAMATH_CALUDE_alpha_is_two_thirds_l1174_117472

theorem alpha_is_two_thirds (α : ℚ) 
  (h1 : 0 < α) 
  (h2 : α < 1) 
  (h3 : Real.cos (3 * Real.pi * α) + 2 * Real.cos (2 * Real.pi * α) = 0) : 
  α = 2/3 := by
sorry

end NUMINAMATH_CALUDE_alpha_is_two_thirds_l1174_117472


namespace NUMINAMATH_CALUDE_least_number_of_candles_l1174_117401

theorem least_number_of_candles (b : ℕ) : 
  b > 0 ∧ 
  b % 6 = 5 ∧ 
  b % 8 = 7 ∧ 
  b % 9 = 3 ∧ 
  (∀ c : ℕ, c > 0 ∧ c % 6 = 5 ∧ c % 8 = 7 ∧ c % 9 = 3 → c ≥ b) → 
  b = 119 :=
sorry

end NUMINAMATH_CALUDE_least_number_of_candles_l1174_117401


namespace NUMINAMATH_CALUDE_twelve_numbers_divisible_by_three_l1174_117457

theorem twelve_numbers_divisible_by_three (n : ℕ) : 
  (n ≥ 10) ∧ 
  (∃ (seq : List ℕ), seq.length = 12 ∧ 
    (∀ x ∈ seq, x ≥ 10 ∧ x ≤ n ∧ x % 3 = 0) ∧
    (∀ y, y ≥ 10 ∧ y ≤ n ∧ y % 3 = 0 → y ∈ seq)) →
  n = 45 :=
by sorry

end NUMINAMATH_CALUDE_twelve_numbers_divisible_by_three_l1174_117457


namespace NUMINAMATH_CALUDE_raghu_investment_l1174_117482

theorem raghu_investment (raghu trishul vishal : ℝ) 
  (h1 : vishal = 1.1 * trishul)
  (h2 : trishul = 0.9 * raghu)
  (h3 : raghu + trishul + vishal = 5780) : 
  raghu = 2000 := by
sorry

end NUMINAMATH_CALUDE_raghu_investment_l1174_117482


namespace NUMINAMATH_CALUDE_triangle_angle_property_l1174_117409

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A triangle in 3D space -/
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

/-- Check if a point is inside a triangle -/
def isInsideTriangle (P : Point3D) (T : Triangle3D) : Prop := sorry

/-- Check if a point is outside the plane of a triangle -/
def isOutsidePlane (D : Point3D) (T : Triangle3D) : Prop := sorry

/-- Angle between three points in 3D space -/
def angle (A B C : Point3D) : ℝ := sorry

/-- An angle is acute if it's less than 90 degrees -/
def isAcute (θ : ℝ) : Prop := θ < Real.pi / 2

/-- An angle is obtuse if it's greater than 90 degrees -/
def isObtuse (θ : ℝ) : Prop := θ > Real.pi / 2

theorem triangle_angle_property (T : Triangle3D) (P D : Point3D) :
  isInsideTriangle P T →
  isOutsidePlane D T →
  (isAcute (angle T.A P D) ∨ isAcute (angle T.B P D) ∨ isAcute (angle T.C P D)) →
  (isObtuse (angle T.A P D) ∨ isObtuse (angle T.B P D) ∨ isObtuse (angle T.C P D)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_property_l1174_117409


namespace NUMINAMATH_CALUDE_village_population_proof_l1174_117496

/-- Proves that given a 20% increase followed by a 20% decrease resulting in 9600,
    the initial population must have been 10000 -/
theorem village_population_proof (initial_population : ℝ) : 
  (initial_population * 1.2 * 0.8 = 9600) → initial_population = 10000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_proof_l1174_117496


namespace NUMINAMATH_CALUDE_fifth_month_sale_l1174_117479

-- Define the sales for the first four months
def first_four_sales : List Int := [5420, 5660, 6200, 6350]

-- Define the sale for the sixth month
def sixth_month_sale : Int := 7070

-- Define the average sale for six months
def average_sale : Int := 6200

-- Define the number of months
def num_months : Int := 6

-- Theorem to prove
theorem fifth_month_sale :
  let total_sales := average_sale * num_months
  let known_sales := first_four_sales.sum + sixth_month_sale
  total_sales - known_sales = 6500 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l1174_117479


namespace NUMINAMATH_CALUDE_intersection_orthogonality_l1174_117416

/-- The line equation -/
def line (x y : ℝ) : Prop := y = 2 * Real.sqrt 2 * (x - 1)

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

/-- Point A satisfies both line and parabola equations -/
def point_A (x y : ℝ) : Prop := line x y ∧ parabola x y

/-- Point B satisfies both line and parabola equations -/
def point_B (x y : ℝ) : Prop := line x y ∧ parabola x y

/-- Point M has coordinates (-1, m) -/
def point_M (m : ℝ) : ℝ × ℝ := (-1, m)

/-- The dot product of vectors MA and MB is zero -/
def orthogonal_condition (x_a y_a x_b y_b m : ℝ) : Prop :=
  (x_a + 1) * (x_b + 1) + (y_a - m) * (y_b - m) = 0

theorem intersection_orthogonality (x_a y_a x_b y_b m : ℝ) :
  point_A x_a y_a →
  point_B x_b y_b →
  orthogonal_condition x_a y_a x_b y_b m →
  m = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_intersection_orthogonality_l1174_117416


namespace NUMINAMATH_CALUDE_parabola_properties_l1174_117454

/-- Represents a parabola of the form y = ax^2 + 4ax + 3 -/
structure Parabola where
  a : ℝ
  h : a ≠ 0

/-- The x-coordinate of the axis of symmetry of the parabola -/
def Parabola.axisOfSymmetry (p : Parabola) : ℝ := -2

/-- Checks if a point (x, y) lies on the parabola -/
def Parabola.isOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + 4 * p.a * x + 3

theorem parabola_properties (p : Parabola) :
  (p.axisOfSymmetry = -2) ∧
  p.isOnParabola 0 3 := by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1174_117454


namespace NUMINAMATH_CALUDE_intersection_M_N_l1174_117471

def U : Set ℝ := Set.univ

def M : Set ℝ := {x : ℝ | x ^ 2 ≤ 4}

def N : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1174_117471


namespace NUMINAMATH_CALUDE_sock_probability_theorem_l1174_117486

/-- Represents the number of pairs of socks for each color -/
structure SockPairs :=
  (blue : ℕ)
  (red : ℕ)
  (green : ℕ)

/-- Calculates the probability of picking two socks of the same color -/
def probabilitySameColor (pairs : SockPairs) : ℚ :=
  let totalSocks := 2 * (pairs.blue + pairs.red + pairs.green)
  let blueProbability := (2 * pairs.blue * (2 * pairs.blue - 1)) / (totalSocks * (totalSocks - 1))
  let redProbability := (2 * pairs.red * (2 * pairs.red - 1)) / (totalSocks * (totalSocks - 1))
  let greenProbability := (2 * pairs.green * (2 * pairs.green - 1)) / (totalSocks * (totalSocks - 1))
  blueProbability + redProbability + greenProbability

/-- Theorem: The probability of picking two socks of the same color is 77/189 -/
theorem sock_probability_theorem (pairs : SockPairs) 
  (h1 : pairs.blue = 8) 
  (h2 : pairs.red = 4) 
  (h3 : pairs.green = 2) : 
  probabilitySameColor pairs = 77 / 189 := by
  sorry

#eval probabilitySameColor { blue := 8, red := 4, green := 2 }

end NUMINAMATH_CALUDE_sock_probability_theorem_l1174_117486


namespace NUMINAMATH_CALUDE_gcf_of_120_180_240_l1174_117446

theorem gcf_of_120_180_240 : Nat.gcd 120 (Nat.gcd 180 240) = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_120_180_240_l1174_117446


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l1174_117423

theorem min_value_geometric_sequence (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
    (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q)
    (h_condition : a 2 * a 3 * a 4 = a 2 + a 3 + a 4) : 
  a 3 ≥ Real.sqrt 3 ∧ ∃ a' : ℕ → ℝ, (∀ n, a' n > 0) ∧ 
    (∃ q' : ℝ, q' > 0 ∧ ∀ n, a' (n + 1) = a' n * q') ∧
    (a' 2 * a' 3 * a' 4 = a' 2 + a' 3 + a' 4) ∧
    (a' 3 = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l1174_117423


namespace NUMINAMATH_CALUDE_total_distance_proof_l1174_117430

/-- The total distance across the country in kilometers -/
def total_distance : ℕ := 8205

/-- The distance Amelia drove on Monday in kilometers -/
def monday_distance : ℕ := 907

/-- The distance Amelia drove on Tuesday in kilometers -/
def tuesday_distance : ℕ := 582

/-- The remaining distance Amelia has to drive in kilometers -/
def remaining_distance : ℕ := 6716

/-- Theorem stating that the total distance is the sum of the distances driven on Monday, Tuesday, and the remaining distance -/
theorem total_distance_proof : 
  total_distance = monday_distance + tuesday_distance + remaining_distance := by
  sorry

end NUMINAMATH_CALUDE_total_distance_proof_l1174_117430


namespace NUMINAMATH_CALUDE_correct_calculation_l1174_117412

theorem correct_calculation (a b : ℝ) (h : a ≠ 0) : (a^2 + a*b) / a = a + b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1174_117412


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l1174_117449

theorem longest_side_of_triangle (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧ (A + B + C = π) →
  -- Side lengths are positive
  (a > 0) ∧ (b > 0) ∧ (c > 0) →
  -- Given conditions
  (Real.tan A = 1/4) →
  (Real.tan B = 3/5) →
  (min a (min b c) = Real.sqrt 2) →
  -- Triangle inequality
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →
  -- Conclusion
  max a (max b c) = Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l1174_117449


namespace NUMINAMATH_CALUDE_factorial_difference_quotient_l1174_117467

theorem factorial_difference_quotient : (Nat.factorial 11 - Nat.factorial 10) / Nat.factorial 9 = 100 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_quotient_l1174_117467


namespace NUMINAMATH_CALUDE_probability_of_shaded_triangle_l1174_117448

/-- Given a diagram with triangles, this theorem proves the probability of selecting a shaded triangle -/
theorem probability_of_shaded_triangle (total_triangles : ℕ) (shaded_triangles : ℕ) : 
  total_triangles = 5 → shaded_triangles = 3 → (shaded_triangles : ℚ) / total_triangles = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_shaded_triangle_l1174_117448


namespace NUMINAMATH_CALUDE_system_solution_l1174_117417

theorem system_solution (x y m : ℝ) : 
  (2 * x + y = 4) → 
  (x + 2 * y = m) → 
  (x + y = 1) → 
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1174_117417


namespace NUMINAMATH_CALUDE_compute_fraction_expression_l1174_117489

theorem compute_fraction_expression : 8 * (1/3)^2 * (2/7) = 16/63 := by
  sorry

end NUMINAMATH_CALUDE_compute_fraction_expression_l1174_117489


namespace NUMINAMATH_CALUDE_sons_age_l1174_117476

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 35 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 33 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l1174_117476


namespace NUMINAMATH_CALUDE_glued_polyhedron_edge_length_l1174_117491

/-- A polyhedron formed by gluing a square-based pyramid to a regular tetrahedron -/
structure GluedPolyhedron where
  -- Square-based pyramid
  pyramid_edge_length : ℝ
  pyramid_edge_count : ℕ
  -- Regular tetrahedron
  tetrahedron_edge_length : ℝ
  tetrahedron_edge_count : ℕ
  -- Gluing properties
  glued_edges : ℕ
  merged_edges : ℕ
  -- Conditions
  pyramid_square_base : pyramid_edge_count = 8
  all_edges_length_2 : pyramid_edge_length = 2 ∧ tetrahedron_edge_length = 2
  tetrahedron_regular : tetrahedron_edge_count = 6
  glued_face_edges : glued_edges = 3
  merged_parallel_edges : merged_edges = 2

/-- The total edge length of the glued polyhedron -/
def totalEdgeLength (p : GluedPolyhedron) : ℝ :=
  (p.pyramid_edge_count + p.tetrahedron_edge_count - p.glued_edges - p.merged_edges) * p.pyramid_edge_length

/-- Theorem stating that the total edge length of the glued polyhedron is 18 -/
theorem glued_polyhedron_edge_length (p : GluedPolyhedron) : totalEdgeLength p = 18 := by
  sorry

end NUMINAMATH_CALUDE_glued_polyhedron_edge_length_l1174_117491


namespace NUMINAMATH_CALUDE_sequence_second_term_l1174_117480

/-- Given a sequence {aₙ} with sum of first n terms Sₙ, where Sₙ = 2aₙ - 1, prove a₂ = 4 -/
theorem sequence_second_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = 2 * a n - 1) : a 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sequence_second_term_l1174_117480


namespace NUMINAMATH_CALUDE_unique_sequence_coefficients_l1174_117497

/-- Given two distinct roots of a characteristic equation and two initial terms of a sequence,
    there exists a unique pair of coefficients that generates the entire sequence. -/
theorem unique_sequence_coefficients
  (x₁ x₂ : ℝ) (a₀ a₁ : ℝ) (h : x₁ ≠ x₂) :
  ∃! (c₁ c₂ : ℝ), ∀ (n : ℕ), c₁ * x₁^n + c₂ * x₂^n = 
    if n = 0 then a₀ else if n = 1 then a₁ else c₁ * x₁^n + c₂ * x₂^n :=
sorry

end NUMINAMATH_CALUDE_unique_sequence_coefficients_l1174_117497


namespace NUMINAMATH_CALUDE_only_sample_size_statement_correct_l1174_117410

/-- Represents a statistical study with a population and a sample. -/
structure StatisticalStudy where
  population_size : ℕ
  sample_size : ℕ

/-- Represents a statement about the statistical study. -/
inductive Statement
  | sample_is_population
  | sample_average_is_population_average
  | examinees_are_population
  | sample_size_is_1000

/-- Checks if a statement is correct for the given statistical study. -/
def is_correct_statement (study : StatisticalStudy) (stmt : Statement) : Prop :=
  match stmt with
  | Statement.sample_is_population => False
  | Statement.sample_average_is_population_average => False
  | Statement.examinees_are_population => False
  | Statement.sample_size_is_1000 => study.sample_size = 1000

/-- The main theorem stating that only the sample size statement is correct. -/
theorem only_sample_size_statement_correct (study : StatisticalStudy) 
    (h1 : study.population_size = 70000)
    (h2 : study.sample_size = 1000) :
    ∀ (stmt : Statement), is_correct_statement study stmt ↔ stmt = Statement.sample_size_is_1000 := by
  sorry

end NUMINAMATH_CALUDE_only_sample_size_statement_correct_l1174_117410


namespace NUMINAMATH_CALUDE_roots_difference_squared_l1174_117473

theorem roots_difference_squared (p q r s : ℝ) : 
  (r^2 - p*r + q = 0) → (s^2 - p*s + q = 0) → (r - s)^2 = p^2 - 4*q := by
  sorry

end NUMINAMATH_CALUDE_roots_difference_squared_l1174_117473


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1174_117405

theorem fraction_sum_equality : (1 : ℚ) / 5 * 3 / 7 + 1 / 2 = 41 / 70 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1174_117405


namespace NUMINAMATH_CALUDE_yvonne_probability_l1174_117458

theorem yvonne_probability (xavier_prob zelda_prob joint_prob : ℝ) :
  xavier_prob = 1/4 →
  zelda_prob = 5/8 →
  joint_prob = 0.0625 →
  ∃ yvonne_prob : ℝ,
    yvonne_prob = 1/16 ∧
    xavier_prob * yvonne_prob * (1 - zelda_prob) = joint_prob :=
by sorry

end NUMINAMATH_CALUDE_yvonne_probability_l1174_117458


namespace NUMINAMATH_CALUDE_angle_inequality_l1174_117481

theorem angle_inequality : 
  let a := (1/2) * Real.cos (7 * π / 180) - (Real.sqrt 3 / 2) * Real.sin (7 * π / 180)
  let b := (2 * Real.tan (12 * π / 180)) / (1 + Real.tan (12 * π / 180) ^ 2)
  let c := Real.sqrt ((1 - Real.cos (44 * π / 180)) / 2)
  c < a ∧ a < b :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_l1174_117481


namespace NUMINAMATH_CALUDE_parabola_vertex_l1174_117436

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- Theorem: The vertex of the parabola y = 3x^2 - 6x + 5 is at the point (1, 2) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1174_117436


namespace NUMINAMATH_CALUDE_product_mod_seven_l1174_117434

theorem product_mod_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l1174_117434


namespace NUMINAMATH_CALUDE_clock_angle_at_2_30_l1174_117426

/-- The number of hours on a standard analog clock -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full rotation -/
def full_rotation : ℕ := 360

/-- The time in hours (including fractional part) -/
def time : ℚ := 2.5

/-- Calculates the angle of the hour hand from the 12 o'clock position -/
def hour_hand_angle (t : ℚ) : ℚ := (t * full_rotation) / clock_hours

/-- Calculates the angle of the minute hand from the 12 o'clock position -/
def minute_hand_angle (t : ℚ) : ℚ := ((t - t.floor) * full_rotation)

/-- Calculates the absolute difference between two angles -/
def angle_difference (a b : ℚ) : ℚ := min (abs (a - b)) (full_rotation - abs (a - b))

theorem clock_angle_at_2_30 :
  angle_difference (hour_hand_angle time) (minute_hand_angle time) = 105 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_2_30_l1174_117426


namespace NUMINAMATH_CALUDE_isosceles_triangle_rational_trig_l1174_117492

/-- An isosceles triangle with integer base and height has rational sine and cosine of vertex angle -/
theorem isosceles_triangle_rational_trig (BC AD : ℤ) (h : BC > 0 ∧ AD > 0) : 
  ∃ (sinA cosA : ℚ), 
    sinA = Real.sin (Real.arccos ((BC * BC - 2 * AD * AD) / (BC * BC + 2 * AD * AD))) ∧
    cosA = (BC * BC - 2 * AD * AD) / (BC * BC + 2 * AD * AD) := by
  sorry

#check isosceles_triangle_rational_trig

end NUMINAMATH_CALUDE_isosceles_triangle_rational_trig_l1174_117492


namespace NUMINAMATH_CALUDE_system_solutions_l1174_117424

def has_solution (a : ℝ) (x y : ℝ) : Prop :=
  x > 0 ∧ y ≥ 0 ∧ 2*y - 2 = a*(x - 2) ∧ 4*y / (|x| + x) = Real.sqrt y

theorem system_solutions :
  ∀ a : ℝ,
    (a < 0 ∨ a > 1 → 
      has_solution a (2 - 2/a) 0 ∧ has_solution a 2 1) ∧
    (0 ≤ a ∧ a ≤ 1 → 
      has_solution a 2 1) ∧
    ((1 < a ∧ a < 2) ∨ a > 2 → 
      has_solution a (2 - 2/a) 0 ∧ 
      has_solution a 2 1 ∧ 
      has_solution a (2*a - 2) ((a-1)^2)) :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l1174_117424


namespace NUMINAMATH_CALUDE_unitsDigitOfSumOfSquares2023_l1174_117494

/-- The units digit of the sum of the squares of the first n odd, positive integers -/
def unitsDigitOfSumOfSquares (n : ℕ) : ℕ :=
  (n * 1 + n * 9 + (n / 2 + n % 2) * 5) % 10

/-- The theorem stating that the units digit of the sum of the squares 
    of the first 2023 odd, positive integers is 5 -/
theorem unitsDigitOfSumOfSquares2023 : 
  unitsDigitOfSumOfSquares 2023 = 5 := by
  sorry

#eval unitsDigitOfSumOfSquares 2023

end NUMINAMATH_CALUDE_unitsDigitOfSumOfSquares2023_l1174_117494


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1174_117477

theorem geometric_sequence_seventh_term
  (a₁ : ℝ)
  (a₁₀ : ℝ)
  (h₁ : a₁ = 12)
  (h₂ : a₁₀ = 78732)
  (h₃ : ∀ n : ℕ, 1 ≤ n → n ≤ 10 → ∃ r : ℝ, a₁ * r^(n-1) = a₁₀^((n-1)/9) * a₁^(1-(n-1)/9)) :
  ∃ a₇ : ℝ, a₇ = 8748 ∧ a₁ * (a₁₀ / a₁)^(6/9) = a₇ :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1174_117477


namespace NUMINAMATH_CALUDE_roosters_count_l1174_117431

theorem roosters_count (total_chickens egg_laying_hens non_egg_laying_hens : ℕ) 
  (h1 : total_chickens = 325)
  (h2 : egg_laying_hens = 277)
  (h3 : non_egg_laying_hens = 20) :
  total_chickens - (egg_laying_hens + non_egg_laying_hens) = 28 :=
by sorry

end NUMINAMATH_CALUDE_roosters_count_l1174_117431


namespace NUMINAMATH_CALUDE_series_sum_equals_four_implies_x_equals_half_l1174_117404

/-- The sum of the infinite series 1 + 2x + 3x^2 + ... -/
noncomputable def S (x : ℝ) : ℝ := ∑' n, (n + 1) * x^n

/-- The theorem stating that if S(x) = 4, then x = 1/2 -/
theorem series_sum_equals_four_implies_x_equals_half :
  ∀ x : ℝ, x < 1 → S x = 4 → x = 1/2 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_four_implies_x_equals_half_l1174_117404


namespace NUMINAMATH_CALUDE_parallel_planes_line_sufficiency_l1174_117418

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relations
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the subset relation for a line and a plane
variable (line_subset_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_line_sufficiency 
  (α β : Plane) (m : Line) 
  (h_subset : line_subset_plane m α) 
  (h_distinct : α ≠ β) :
  (∀ α β m, plane_parallel α β → line_parallel_plane m β) ∧ 
  (∃ α β m, line_parallel_plane m β ∧ ¬plane_parallel α β) :=
by sorry

end NUMINAMATH_CALUDE_parallel_planes_line_sufficiency_l1174_117418


namespace NUMINAMATH_CALUDE_time_difference_steve_jennifer_l1174_117438

/-- Represents the time in minutes for various running distances --/
structure RunningTimes where
  danny_to_steve : ℝ
  jennifer_to_danny : ℝ

/-- Theorem stating the difference in time between Steve and Jennifer reaching their respective halfway points --/
theorem time_difference_steve_jennifer (times : RunningTimes) 
  (h1 : times.danny_to_steve = 35)
  (h2 : times.jennifer_to_danny = 10)
  (h3 : times.jennifer_to_danny * 2 = times.danny_to_steve) : 
  (2 * times.danny_to_steve) / 2 - times.jennifer_to_danny / 2 = 30 := by
  sorry


end NUMINAMATH_CALUDE_time_difference_steve_jennifer_l1174_117438


namespace NUMINAMATH_CALUDE_systematic_sampling_proof_l1174_117474

/-- Represents a student with an ID number -/
structure Student where
  id : Nat
  deriving Repr

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Lottery
  deriving Repr

/-- Checks if a number is divisible by 5 -/
def isDivisibleByFive (n : Nat) : Bool :=
  n % 5 == 0

/-- Selects students whose IDs are divisible by 5 -/
def selectStudents (students : List Student) : List Student :=
  students.filter (fun s => isDivisibleByFive s.id)

/-- Theorem: Selecting students with IDs divisible by 5 from a group of 60 students
    numbered 1 to 60 is an example of systematic sampling -/
theorem systematic_sampling_proof (students : List Student) 
    (h1 : students.length = 60)
    (h2 : ∀ i, 1 ≤ i ∧ i ≤ 60 → ∃ s ∈ students, s.id = i)
    (h3 : ∀ s ∈ students, 1 ≤ s.id ∧ s.id ≤ 60) :
    (selectStudents students).length = 12 ∧ 
    SamplingMethod.Systematic = 
      (match (selectStudents students) with
       | [] => SamplingMethod.SimpleRandom  -- Default case, should not occur
       | _ => SamplingMethod.Systematic) := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_proof_l1174_117474


namespace NUMINAMATH_CALUDE_substitution_result_l1174_117402

theorem substitution_result (x y : ℝ) :
  y = 2 * x + 1 ∧ 5 * x - 2 * y = 7 →
  5 * x - 4 * x - 2 = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_substitution_result_l1174_117402


namespace NUMINAMATH_CALUDE_triangle_inequality_l1174_117462

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : 
  let s := (a + b + c) / 2
  (2*a*(2*a - s))/(b + c) + (2*b*(2*b - s))/(c + a) + (2*c*(2*c - s))/(a + b) ≥ s := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1174_117462


namespace NUMINAMATH_CALUDE_interval_representation_l1174_117498

def open_closed_interval (a b : ℝ) : Set ℝ := {x | a < x ∧ x ≤ b}

theorem interval_representation :
  open_closed_interval (-3) 2 = {x : ℝ | -3 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_interval_representation_l1174_117498


namespace NUMINAMATH_CALUDE_special_set_bounds_l1174_117493

/-- A set of points in 3D space satisfying the given conditions -/
def SpecialSet (n : ℕ) (S : Set (ℝ × ℝ × ℝ)) : Prop :=
  (n > 0) ∧ 
  (∀ (planes : Finset (Set (ℝ × ℝ × ℝ))), planes.card = n → 
    ∃ (p : ℝ × ℝ × ℝ), p ∈ S ∧ ∀ (plane : Set (ℝ × ℝ × ℝ)), plane ∈ planes → p ∉ plane) ∧
  (∀ (X : ℝ × ℝ × ℝ), X ∈ S → 
    ∃ (planes : Finset (Set (ℝ × ℝ × ℝ))), planes.card = n ∧ 
      ∀ (Y : ℝ × ℝ × ℝ), Y ∈ S \ {X} → ∃ (plane : Set (ℝ × ℝ × ℝ)), plane ∈ planes ∧ Y ∈ plane)

theorem special_set_bounds (n : ℕ) (S : Set (ℝ × ℝ × ℝ)) (h : SpecialSet n S) :
  (3 * n + 1 : ℕ) ≤ S.ncard ∧ S.ncard ≤ Nat.choose (n + 3) 3 := by
  sorry

end NUMINAMATH_CALUDE_special_set_bounds_l1174_117493


namespace NUMINAMATH_CALUDE_linear_inequality_solution_l1174_117475

theorem linear_inequality_solution (x : ℝ) : (2 * x - 1 ≥ 3) ↔ (x ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_linear_inequality_solution_l1174_117475


namespace NUMINAMATH_CALUDE_least_possible_area_of_square_l1174_117461

/-- Given a square with sides measured to the nearest centimeter as 7 cm,
    the least possible actual area of the square is 42.25 cm². -/
theorem least_possible_area_of_square (side_length : ℝ) : 
  (6.5 ≤ side_length) ∧ (side_length < 7.5) → side_length ^ 2 ≥ 42.25 := by
  sorry

end NUMINAMATH_CALUDE_least_possible_area_of_square_l1174_117461
