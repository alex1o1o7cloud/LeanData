import Mathlib

namespace sin_alpha_minus_pi_fourth_l2922_292282

theorem sin_alpha_minus_pi_fourth (α : Real) : 
  α ∈ Set.Icc (π) (3*π/2) →   -- α is in the third quadrant
  Real.tan (α + π/4) = -2 →   -- tan(α + π/4) = -2
  Real.sin (α - π/4) = -Real.sqrt 5 / 5 := by
sorry

end sin_alpha_minus_pi_fourth_l2922_292282


namespace total_students_l2922_292248

theorem total_students (group1_count : ℕ) (group1_avg : ℚ)
                       (group2_count : ℕ) (group2_avg : ℚ)
                       (total_avg : ℚ) :
  group1_count = 15 →
  group1_avg = 80 / 100 →
  group2_count = 10 →
  group2_avg = 90 / 100 →
  total_avg = 84 / 100 →
  group1_count + group2_count = 25 := by
sorry

end total_students_l2922_292248


namespace reciprocal_of_negative_fraction_l2922_292236

theorem reciprocal_of_negative_fraction (n : ℕ) (hn : n ≠ 0) :
  (-(1 : ℚ) / n)⁻¹ = -n := by
  sorry

end reciprocal_of_negative_fraction_l2922_292236


namespace group_size_proof_l2922_292270

theorem group_size_proof (total_rupees : ℚ) (h1 : total_rupees = 72.25) : ∃ n : ℕ, 
  (n : ℚ) * (n : ℚ) = total_rupees * 100 ∧ n = 85 := by
  sorry

end group_size_proof_l2922_292270


namespace exam_scores_l2922_292207

theorem exam_scores (x y : ℝ) (h1 : (x * y + 98) / (x + 1) = y + 1) 
  (h2 : (x * y + 98 + 70) / (x + 2) = y - 1) : 
  (x + 2 = 10) ∧ ((x * y + 98 + 70) / (x + 2) = 88) :=
by sorry

end exam_scores_l2922_292207


namespace inequality_proof_l2922_292296

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 := by
  sorry

end inequality_proof_l2922_292296


namespace intersection_of_M_and_N_l2922_292290

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x < 1}

theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end intersection_of_M_and_N_l2922_292290


namespace sqrt_meaningful_iff_x_geq_three_halves_l2922_292285

theorem sqrt_meaningful_iff_x_geq_three_halves (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = 2 * x - 3) ↔ x ≥ 3 / 2 :=
by sorry

end sqrt_meaningful_iff_x_geq_three_halves_l2922_292285


namespace complement_of_A_l2922_292275

-- Define the set A
def A : Set ℝ := {x : ℝ | (x + 1) / (x + 2) ≤ 0}

-- State the theorem
theorem complement_of_A : 
  Set.compl A = Set.Ici (-1) ∪ Set.Iic (-2) :=
sorry

end complement_of_A_l2922_292275


namespace distinct_grade_assignments_l2922_292273

/-- The number of students in the class -/
def num_students : ℕ := 12

/-- The number of possible grades -/
def num_grades : ℕ := 4

/-- Theorem: The number of distinct ways to assign grades to all students -/
theorem distinct_grade_assignments :
  (num_grades : ℕ) ^ num_students = 16777216 := by
  sorry

end distinct_grade_assignments_l2922_292273


namespace intersection_complement_equals_interval_l2922_292228

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 5*x ≥ 0}

-- Theorem statement
theorem intersection_complement_equals_interval :
  A ∩ (Set.univ \ B) = Set.Ioc 0 3 := by
  sorry

end intersection_complement_equals_interval_l2922_292228


namespace hypotenuse_length_from_quadratic_roots_l2922_292246

theorem hypotenuse_length_from_quadratic_roots :
  ∀ a b c : ℝ,
  (a^2 - 6*a + 4 = 0) →
  (b^2 - 6*b + 4 = 0) →
  (a ≠ b) →
  (c^2 = a^2 + b^2) →
  c = 2 * Real.sqrt 7 := by
sorry

end hypotenuse_length_from_quadratic_roots_l2922_292246


namespace abc_fraction_equality_l2922_292203

theorem abc_fraction_equality (a b c : ℝ) 
  (h1 : a * b + b * c + a * c = 1)
  (h2 : a ≠ 1 ∧ a ≠ -1)
  (h3 : b ≠ 1 ∧ b ≠ -1)
  (h4 : c ≠ 1 ∧ c ≠ -1) :
  a / (1 - a^2) + b / (1 - b^2) + c / (1 - c^2) = 
  4 * a * b * c / ((1 - a^2) * (1 - b^2) * (1 - c^2)) := by
sorry

end abc_fraction_equality_l2922_292203


namespace hyperbola_and_ellipse_condition_l2922_292215

/-- Represents a hyperbola equation -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (1 - m) + y^2 / (m + 2) = 1

/-- Represents an ellipse equation with foci on the x-axis -/
def is_ellipse_x_foci (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (2 * m) + y^2 / (2 - m) = 1

/-- Main theorem -/
theorem hyperbola_and_ellipse_condition (m : ℝ) 
  (h1 : is_hyperbola m) (h2 : is_ellipse_x_foci m) : 
  1 < m ∧ m < 2 := by
  sorry

end hyperbola_and_ellipse_condition_l2922_292215


namespace local_politics_coverage_l2922_292222

theorem local_politics_coverage (total_reporters : ℕ) 
  (h1 : total_reporters > 0) 
  (politics_coverage : ℝ) 
  (h2 : politics_coverage = 0.25) 
  (local_politics_non_coverage : ℝ) 
  (h3 : local_politics_non_coverage = 0.2) : 
  (politics_coverage * (1 - local_politics_non_coverage)) * total_reporters / total_reporters = 0.2 := by
  sorry

end local_politics_coverage_l2922_292222


namespace conference_handshakes_l2922_292298

/-- The number of handshakes in a conference with specified conditions -/
def max_handshakes (total_participants : ℕ) (committee_members : ℕ) : ℕ :=
  let non_committee := total_participants - committee_members
  (non_committee * (non_committee - 1)) / 2

/-- Theorem stating the maximum number of handshakes in the given conference scenario -/
theorem conference_handshakes :
  max_handshakes 50 10 = 780 :=
by sorry

end conference_handshakes_l2922_292298


namespace person_speed_l2922_292240

/-- Given a person crossing a street, calculate their speed in km/hr -/
theorem person_speed (distance : ℝ) (time : ℝ) (h1 : distance = 720) (h2 : time = 12) :
  distance / 1000 / (time / 60) = 3.6 := by
  sorry

end person_speed_l2922_292240


namespace tank_filling_time_l2922_292295

/-- The time taken to fill a tank with two pipes and a leak -/
theorem tank_filling_time (pipe1_time pipe2_time : ℝ) (leak_fraction : ℝ) : 
  pipe1_time = 20 → 
  pipe2_time = 30 → 
  leak_fraction = 1/3 → 
  (1 / ((1 / pipe1_time + 1 / pipe2_time) * (1 - leak_fraction))) = 18 := by
  sorry

end tank_filling_time_l2922_292295


namespace square_area_3_square_area_3_proof_l2922_292261

/-- The area of a square with side length 3 is 9 -/
theorem square_area_3 : Real → Prop :=
  fun area =>
    let side_length : Real := 3
    area = side_length ^ 2

#check square_area_3 9

/-- Proof of the theorem -/
theorem square_area_3_proof : square_area_3 9 := by
  sorry

end square_area_3_square_area_3_proof_l2922_292261


namespace sevenPointOneTwoThreeBar_eq_fraction_l2922_292220

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def RepeatingDecimal.toRational (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The repeating decimal 7.123̄ -/
def sevenPointOneTwoThreeBar : RepeatingDecimal :=
  { integerPart := 7, repeatingPart := 123 }

theorem sevenPointOneTwoThreeBar_eq_fraction :
  RepeatingDecimal.toRational sevenPointOneTwoThreeBar = 2372 / 333 := by
  sorry

end sevenPointOneTwoThreeBar_eq_fraction_l2922_292220


namespace boys_ratio_in_class_l2922_292241

theorem boys_ratio_in_class (p : ℝ) 
  (h1 : p ≥ 0 ∧ p ≤ 1) -- Probability is between 0 and 1
  (h2 : p = 3/4 * (1 - p)) -- Condition from the problem
  : p = 3/7 := by
  sorry

end boys_ratio_in_class_l2922_292241


namespace joan_change_l2922_292274

/-- The change Joan received after buying a cat toy and a cage -/
theorem joan_change (cat_toy_cost cage_cost payment : ℚ) : 
  cat_toy_cost = 877/100 →
  cage_cost = 1097/100 →
  payment = 20 →
  payment - (cat_toy_cost + cage_cost) = 26/100 := by
sorry

end joan_change_l2922_292274


namespace A_profit_share_l2922_292230

-- Define the investments and profit shares
def investment_A : ℕ := 6300
def investment_B : ℕ := 4200
def investment_C : ℕ := 10500
def profit_share_A : ℚ := 45 / 100
def profit_share_B : ℚ := 30 / 100
def profit_share_C : ℚ := 25 / 100
def total_profit : ℕ := 12200

-- Theorem to prove A's share of the profit
theorem A_profit_share :
  (profit_share_A * total_profit : ℚ) = 5490 := by sorry

end A_profit_share_l2922_292230


namespace pencil_cost_l2922_292287

theorem pencil_cost (total_students : Nat) (total_cost : Nat) 
  (h1 : total_students = 36)
  (h2 : total_cost = 1881)
  (s : Nat) (c : Nat) (n : Nat)
  (h3 : s > total_students / 2)
  (h4 : c > n)
  (h5 : n > 1)
  (h6 : s * c * n = total_cost) :
  c = 17 := by
  sorry

end pencil_cost_l2922_292287


namespace unique_solution_is_201_l2922_292254

theorem unique_solution_is_201 : ∃! (n : ℕ+), 
  (Finset.sum (Finset.range n) (λ k => 4*k + 1)) / (Finset.sum (Finset.range n) (λ k => 4*(k + 1))) = 100 / 101 :=
by
  -- The proof goes here
  sorry

end unique_solution_is_201_l2922_292254


namespace four_planes_max_parts_l2922_292223

/-- The maximum number of parts into which space can be divided by k planes -/
def max_parts (k : ℕ) : ℚ := (k^3 + 5*k + 6) / 6

/-- Theorem: The maximum number of parts into which space can be divided by four planes is 15 -/
theorem four_planes_max_parts : max_parts 4 = 15 := by
  sorry

end four_planes_max_parts_l2922_292223


namespace bonus_is_ten_dollars_l2922_292286

/-- Represents the payment structure for Brady's transcription job -/
structure TranscriptionJob where
  base_pay : ℚ  -- Base pay per card in dollars
  cards_for_bonus : ℕ  -- Number of cards needed for a bonus
  total_cards : ℕ  -- Total number of cards transcribed
  total_pay : ℚ  -- Total pay including bonuses in dollars

/-- Calculates the bonus amount per bonus interval -/
def bonus_amount (job : TranscriptionJob) : ℚ :=
  let base_total := job.base_pay * job.total_cards
  let bonus_count := job.total_cards / job.cards_for_bonus
  (job.total_pay - base_total) / bonus_count

/-- Theorem stating that the bonus amount is $10 for every 100 cards -/
theorem bonus_is_ten_dollars (job : TranscriptionJob) 
  (h1 : job.base_pay = 70 / 100)
  (h2 : job.cards_for_bonus = 100)
  (h3 : job.total_cards = 200)
  (h4 : job.total_pay = 160) :
  bonus_amount job = 10 := by
  sorry

end bonus_is_ten_dollars_l2922_292286


namespace vacuum_time_per_room_l2922_292288

theorem vacuum_time_per_room 
  (battery_life : ℕ) 
  (num_rooms : ℕ) 
  (additional_charges : ℕ) 
  (h1 : battery_life = 10)
  (h2 : num_rooms = 5)
  (h3 : additional_charges = 2) :
  (battery_life * (additional_charges + 1)) / num_rooms = 6 := by
  sorry

end vacuum_time_per_room_l2922_292288


namespace trigonometric_expression_equals_one_l2922_292279

theorem trigonometric_expression_equals_one :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
  sorry

end trigonometric_expression_equals_one_l2922_292279


namespace table_leg_problem_l2922_292292

theorem table_leg_problem :
  ∀ (x y : ℕ),
    x ≥ 2 →
    y ≥ 2 →
    3 * x + 4 * y = 23 →
    x = 5 ∧ y = 2 :=
by sorry

end table_leg_problem_l2922_292292


namespace high_school_population_change_l2922_292208

/-- Represents the number of students in a high school --/
structure SchoolPopulation where
  boys : ℕ
  girls : ℕ

/-- Represents the ratio of boys to girls --/
structure Ratio where
  boys : ℕ
  girls : ℕ

def SchoolPopulation.ratio (pop : SchoolPopulation) : Ratio :=
  { boys := pop.boys, girls := pop.girls }

theorem high_school_population_change 
  (initial_ratio : Ratio)
  (final_ratio : Ratio)
  (boys_left : ℕ)
  (girls_left : ℕ)
  (h1 : initial_ratio.boys = 3 ∧ initial_ratio.girls = 4)
  (h2 : final_ratio.boys = 4 ∧ final_ratio.girls = 5)
  (h3 : boys_left = 10)
  (h4 : girls_left = 20)
  (h5 : girls_left = 2 * boys_left) :
  ∃ (initial_pop : SchoolPopulation),
    initial_pop.ratio = initial_ratio ∧
    initial_pop.boys = 90 ∧
    let final_pop : SchoolPopulation :=
      { boys := initial_pop.boys - boys_left,
        girls := initial_pop.girls - girls_left }
    final_pop.ratio = final_ratio :=
  sorry


end high_school_population_change_l2922_292208


namespace han_xin_counting_l2922_292283

theorem han_xin_counting (n : ℕ) : n ≥ 53 ∧ n % 3 = 2 ∧ n % 5 = 3 ∧ n % 7 = 4 →
  ∀ m : ℕ, m < 53 → ¬(m % 3 = 2 ∧ m % 5 = 3 ∧ m % 7 = 4) := by
  sorry

end han_xin_counting_l2922_292283


namespace quadrilateral_diagonal_length_l2922_292242

/-- A convex quadrilateral in a plane -/
structure ConvexQuadrilateral where
  -- We don't need to define the specific properties of a convex quadrilateral here
  -- as they are not directly used in the problem statement

/-- The theorem stating the relation between the area, sum of sides and diagonals
    in a specific convex quadrilateral -/
theorem quadrilateral_diagonal_length 
  (Q : ConvexQuadrilateral) 
  (area : ℝ) 
  (sum_sides_and_diagonal : ℝ) 
  (h1 : area = 32) 
  (h2 : sum_sides_and_diagonal = 16) : 
  ∃ (other_diagonal : ℝ), other_diagonal = 8 * Real.sqrt 2 :=
sorry

end quadrilateral_diagonal_length_l2922_292242


namespace pam_miles_walked_l2922_292233

/-- Represents a pedometer with a maximum step count before resetting --/
structure Pedometer where
  max_count : ℕ
  resets : ℕ
  final_reading : ℕ

/-- Calculates the total steps walked given a pedometer --/
def total_steps (p : Pedometer) : ℕ :=
  p.max_count * p.resets + p.final_reading + p.resets

/-- Converts steps to miles given a steps-per-mile rate --/
def steps_to_miles (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

/-- Theorem stating the total miles walked by Pam --/
theorem pam_miles_walked :
  let p : Pedometer := { max_count := 49999, resets := 50, final_reading := 25000 }
  let steps_per_mile := 1500
  steps_to_miles (total_steps p) steps_per_mile = 1683 := by
  sorry


end pam_miles_walked_l2922_292233


namespace angle_value_proof_l2922_292235

theorem angle_value_proof (ABC : ℝ) (x : ℝ) 
  (h1 : ABC = 90)
  (h2 : ABC = 44 + x) : 
  x = 46 := by
  sorry

end angle_value_proof_l2922_292235


namespace prob_c_not_adjacent_to_ab_l2922_292238

/-- Represents the number of students in the group photo -/
def total_students : ℕ := 7

/-- Represents the probability that student C is not adjacent to student A or B,
    given that A and B stand together and C stands on the edge -/
def probability_not_adjacent : ℚ := 4/5

/-- Theorem stating the probability that student C is not adjacent to student A or B -/
theorem prob_c_not_adjacent_to_ab :
  probability_not_adjacent = 4/5 :=
sorry

end prob_c_not_adjacent_to_ab_l2922_292238


namespace sports_club_non_players_l2922_292231

theorem sports_club_non_players (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ)
  (h1 : total = 35)
  (h2 : badminton = 15)
  (h3 : tennis = 18)
  (h4 : both = 3) :
  total - (badminton + tennis - both) = 5 := by
  sorry

end sports_club_non_players_l2922_292231


namespace perpendicular_bisector_theorem_l2922_292260

/-- A structure representing a triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A function to construct the perpendicular bisector points A', B', and C' -/
def constructPerpendicularBisectorPoints (t : Triangle) : Triangle :=
  sorry

/-- A predicate to check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  sorry

/-- A predicate to check if a triangle has angles 30°, 30°, and 120° -/
def has30_30_120Angles (t : Triangle) : Prop :=
  sorry

/-- The main theorem -/
theorem perpendicular_bisector_theorem (t : Triangle) :
  let t' := constructPerpendicularBisectorPoints t
  isEquilateral t' ↔ (isEquilateral t ∨ has30_30_120Angles t) :=
sorry

end perpendicular_bisector_theorem_l2922_292260


namespace sequence_seventh_term_l2922_292249

theorem sequence_seventh_term (a : ℕ → ℤ) 
  (h : ∀ n : ℕ, a 1 + a (2 * n - 1) = 4 * n - 6) : 
  a 7 = 11 := by
sorry

end sequence_seventh_term_l2922_292249


namespace unique_number_l2922_292271

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def satisfies_conditions (n : ℕ) : Prop :=
  let h := n / 100
  let t := (n / 10) % 10
  let u := n % 10
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  h = 2 * t ∧           -- hundreds digit is twice the tens digit
  u = 2 * t^3 ∧         -- units digit is double the cube of tens digit
  is_prime (h + t + u)  -- sum of digits is prime

theorem unique_number : ∀ n : ℕ, satisfies_conditions n ↔ n = 212 :=
sorry

end unique_number_l2922_292271


namespace sum_of_roots_x4_minus_4x3_minus_1_l2922_292214

theorem sum_of_roots_x4_minus_4x3_minus_1 : 
  ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (∀ x : ℝ, x^4 - 4*x^3 - 1 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧
    r₁ + r₂ + r₃ + r₄ = 4 := by
  sorry

end sum_of_roots_x4_minus_4x3_minus_1_l2922_292214


namespace coin_ratio_is_equal_l2922_292255

/-- Represents the types of coins in the bag -/
inductive CoinType
  | OneRupee
  | FiftyPaise
  | TwentyFivePaise

/-- The value of a coin in rupees -/
def coinValue (c : CoinType) : Rat :=
  match c with
  | CoinType.OneRupee => 1
  | CoinType.FiftyPaise => 1/2
  | CoinType.TwentyFivePaise => 1/4

/-- The number of coins of each type -/
def numCoins : Nat := 80

/-- The total value of all coins in rupees -/
def totalValue : Rat := 140

/-- Theorem stating that the ratio of coin counts is 1:1:1 -/
theorem coin_ratio_is_equal :
  let oneRupeeCount := numCoins
  let fiftyPaiseCount := numCoins
  let twentyFivePaiseCount := numCoins
  let totalCalculatedValue := oneRupeeCount * coinValue CoinType.OneRupee +
                              fiftyPaiseCount * coinValue CoinType.FiftyPaise +
                              twentyFivePaiseCount * coinValue CoinType.TwentyFivePaise
  totalCalculatedValue = totalValue →
  oneRupeeCount = fiftyPaiseCount ∧ fiftyPaiseCount = twentyFivePaiseCount :=
by
  sorry

#check coin_ratio_is_equal

end coin_ratio_is_equal_l2922_292255


namespace extremum_and_max_min_of_f_l2922_292250

def f (x : ℝ) := x^3 + 4*x^2 - 11*x + 16

theorem extremum_and_max_min_of_f :
  (∃ (x : ℝ), f x = 10 ∧ ∀ y, |y - 1| < |x - 1| → f y ≠ 10) ∧
  (∀ x ∈ Set.Icc 0 2, f x ≤ 18) ∧
  (∃ x ∈ Set.Icc 0 2, f x = 18) ∧
  (∀ x ∈ Set.Icc 0 2, f x ≥ 10) ∧
  (∃ x ∈ Set.Icc 0 2, f x = 10) :=
sorry

end extremum_and_max_min_of_f_l2922_292250


namespace equal_intercept_line_equation_l2922_292212

/-- A line passing through point A(2, 1) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through point A(2, 1) -/
  passes_through_A : m * 2 + b = 1
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : b ≠ 0 → -b = b / m

/-- The equation of the line is either x - 2y = 0 or x + y - 3 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = 1/2 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = 3) :=
sorry

end equal_intercept_line_equation_l2922_292212


namespace rational_function_value_l2922_292293

/-- A rational function with specific properties -/
def rational_function (p q : ℝ → ℝ) : Prop :=
  ∃ k : ℝ,
    (∀ x, q x = (x + 5) * (x - 1)) ∧
    (∀ x, p x = k * x) ∧
    (p 0 / q 0 = 0) ∧
    (p 4 / q 4 = -1/2)

/-- The main theorem -/
theorem rational_function_value (p q : ℝ → ℝ) 
  (h : rational_function p q) : p (-1) / q (-1) = 27/64 := by
  sorry

end rational_function_value_l2922_292293


namespace regular_tetrahedron_sphere_ratio_l2922_292216

/-- A regular tetrahedron is a tetrahedron with four congruent equilateral triangles as faces -/
structure RegularTetrahedron where
  -- We don't need to define the structure explicitly for this problem

/-- The ratio of the radius of the circumscribed sphere to the inscribed sphere of a regular tetrahedron -/
def circumscribed_to_inscribed_ratio (t : RegularTetrahedron) : ℚ :=
  3 / 1

/-- Theorem: The ratio of the radius of the circumscribed sphere to the inscribed sphere of a regular tetrahedron is 3:1 -/
theorem regular_tetrahedron_sphere_ratio (t : RegularTetrahedron) :
  circumscribed_to_inscribed_ratio t = 3 / 1 := by
  sorry

end regular_tetrahedron_sphere_ratio_l2922_292216


namespace book_club_task_distribution_l2922_292218

theorem book_club_task_distribution (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  Nat.choose n k = 10 := by
  sorry

end book_club_task_distribution_l2922_292218


namespace larger_part_of_90_l2922_292204

theorem larger_part_of_90 (x : ℝ) : 
  x + (90 - x) = 90 ∧ 
  0.4 * x = 0.3 * (90 - x) + 15 → 
  max x (90 - x) = 60 := by
sorry

end larger_part_of_90_l2922_292204


namespace no_poly3_satisfies_conditions_l2922_292234

/-- A polynomial function of degree exactly 3 -/
structure Poly3 where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  degree_three : a ≠ 0

/-- Evaluation of a Poly3 at a point -/
def Poly3.eval (p : Poly3) (x : ℝ) : ℝ :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- The conditions that the polynomial must satisfy -/
def satisfies_conditions (p : Poly3) : Prop :=
  ∀ x, p.eval (x^2) = (p.eval x)^2 ∧
       p.eval (x^2) = p.eval (p.eval x) ∧
       p.eval 1 = 2

theorem no_poly3_satisfies_conditions :
  ¬∃ p : Poly3, satisfies_conditions p :=
sorry

end no_poly3_satisfies_conditions_l2922_292234


namespace f_increasing_implies_f_1_ge_25_l2922_292229

def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

theorem f_increasing_implies_f_1_ge_25 (m : ℝ) :
  (∀ x₁ x₂ : ℝ, -2 ≤ x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) →
  f m 1 ≥ 25 := by
  sorry

end f_increasing_implies_f_1_ge_25_l2922_292229


namespace no_x_term_iff_k_eq_two_l2922_292257

/-- The polynomial x^2 + (k-2)x - 3 does not contain the term with x if and only if k = 2 -/
theorem no_x_term_iff_k_eq_two (k : ℝ) : 
  (∀ x : ℝ, x^2 + (k-2)*x - 3 = x^2 - 3) ↔ k = 2 := by
sorry

end no_x_term_iff_k_eq_two_l2922_292257


namespace shopkeeper_loss_l2922_292205

/-- Represents the overall loss amount given stock worth and selling conditions --/
def overall_loss (stock_worth : ℝ) : ℝ :=
  let profit_part := 0.2 * stock_worth * 1.2
  let loss_part := 0.8 * stock_worth * 0.9
  stock_worth - (profit_part + loss_part)

/-- Theorem stating the overall loss for the given problem --/
theorem shopkeeper_loss : 
  overall_loss 12499.99 = 500 :=
by
  sorry

#eval overall_loss 12499.99

end shopkeeper_loss_l2922_292205


namespace cab_journey_time_l2922_292269

/-- Given a cab that arrives 12 minutes late when traveling at 5/6th of its usual speed,
    prove that its usual journey time is 60 minutes. -/
theorem cab_journey_time : ℝ := by
  -- Let S be the usual speed and T be the usual time
  let S : ℝ := 1  -- We can set S to any positive real number
  let T : ℝ := 60 -- This is what we want to prove

  -- Define the reduced speed
  let reduced_speed : ℝ := (5 / 6) * S

  -- Define the time taken at reduced speed
  let reduced_time : ℝ := T + 12

  -- Check if the speed-time relation holds
  have h : S * T = reduced_speed * reduced_time := by sorry

  -- Prove that T equals 60
  sorry

end cab_journey_time_l2922_292269


namespace negation_of_universal_proposition_l2922_292262

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l2922_292262


namespace card_area_problem_l2922_292226

theorem card_area_problem (l w : ℝ) (h1 : l = 8) (h2 : w = 3) 
  (h3 : (l - 2) * w = 15) : (l * (w - 2) = 8) := by
  sorry

end card_area_problem_l2922_292226


namespace sqrt_equation_solution_l2922_292224

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (2 + Real.sqrt x) = 3 → x = 49 := by
  sorry

end sqrt_equation_solution_l2922_292224


namespace x_minus_p_in_terms_of_p_l2922_292289

theorem x_minus_p_in_terms_of_p (x p : ℝ) (h1 : |x - 2| = p) (h2 : x < 2) : x - p = 2 - 2*p := by
  sorry

end x_minus_p_in_terms_of_p_l2922_292289


namespace inscribed_circle_theorem_l2922_292256

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop :=
  sorry

/-- Checks if a point lies on a line segment between two other points -/
def liesBetween (P Q R : Point) : Prop :=
  sorry

/-- Checks if a quadrilateral has an inscribed circle -/
def hasInscribedCircle (q : Quadrilateral) : Prop :=
  sorry

/-- The main theorem -/
theorem inscribed_circle_theorem (A B C D E F G H P : Point) 
  (q : Quadrilateral) (h1 : q = Quadrilateral.mk A B C D) 
  (h2 : isConvex q)
  (h3 : liesBetween A E B)
  (h4 : liesBetween B F C)
  (h5 : liesBetween C G D)
  (h6 : liesBetween D H A)
  (h7 : P = sorry) -- P is the intersection of EG and FH
  (h8 : hasInscribedCircle (Quadrilateral.mk H A E P))
  (h9 : hasInscribedCircle (Quadrilateral.mk E B F P))
  (h10 : hasInscribedCircle (Quadrilateral.mk F C G P))
  (h11 : hasInscribedCircle (Quadrilateral.mk G D H P)) :
  hasInscribedCircle q :=
sorry

end inscribed_circle_theorem_l2922_292256


namespace video_votes_total_l2922_292268

/-- Represents the voting system for a video -/
structure VideoVotes where
  totalVotes : ℕ
  likePercentage : ℚ
  finalScore : ℤ

/-- Theorem: Given the conditions, the total number of votes is 240 -/
theorem video_votes_total (v : VideoVotes) 
  (h1 : v.likePercentage = 3/4)
  (h2 : v.finalScore = 120) :
  v.totalVotes = 240 := by
  sorry


end video_votes_total_l2922_292268


namespace total_cards_l2922_292284

def sallys_cards (initial : ℕ) (dans_gift : ℕ) (purchased : ℕ) : ℕ :=
  initial + dans_gift + purchased

theorem total_cards : sallys_cards 27 41 20 = 88 := by
  sorry

end total_cards_l2922_292284


namespace arcsin_sum_equals_pi_over_four_l2922_292201

theorem arcsin_sum_equals_pi_over_four :
  Real.arcsin (1 / Real.sqrt 10) + Real.arcsin (1 / Real.sqrt 26) + 
  Real.arcsin (1 / Real.sqrt 50) + Real.arcsin (1 / Real.sqrt 65) = π / 4 := by
  sorry

end arcsin_sum_equals_pi_over_four_l2922_292201


namespace northern_village_conscription_l2922_292211

/-- The number of people to be conscripted from the northern village -/
def northern_conscription (total_population : ℕ) (northern_population : ℕ) (total_conscription : ℕ) : ℕ :=
  (northern_population * total_conscription) / total_population

theorem northern_village_conscription :
  northern_conscription 22500 8100 300 = 108 := by
sorry

end northern_village_conscription_l2922_292211


namespace function_has_period_two_l2922_292213

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_unit_shift (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = f (x - 1)

def matches_exp_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = Real.exp (Real.log 2 * x)

-- State the theorem
theorem function_has_period_two (f : ℝ → ℝ) 
  (h1 : is_even f) 
  (h2 : has_unit_shift f) 
  (h3 : matches_exp_on_unit_interval f) : 
  ∀ x, f (x + 2) = f x := by
  sorry

end function_has_period_two_l2922_292213


namespace bill_oranges_count_l2922_292251

/-- The number of oranges Betty picked -/
def betty_oranges : ℕ := 15

/-- The number of oranges Bill picked -/
def bill_oranges : ℕ := sorry

/-- The number of oranges Frank picked -/
def frank_oranges : ℕ := 3 * (betty_oranges + bill_oranges)

/-- The number of seeds Frank planted -/
def seeds_planted : ℕ := 2 * frank_oranges

/-- The number of oranges on each tree -/
def oranges_per_tree : ℕ := 5

/-- The total number of oranges Philip can pick -/
def philip_oranges : ℕ := 810

theorem bill_oranges_count : bill_oranges = 12 := by
  sorry

end bill_oranges_count_l2922_292251


namespace three_sixes_probability_l2922_292294

-- Define the probabilities for each die
def fair_die_prob : ℚ := 1 / 6
def biased_die_2_prob_six : ℚ := 1 / 2
def biased_die_2_prob_other : ℚ := 1 / 10
def biased_die_3_prob_six : ℚ := 3 / 4
def biased_die_3_prob_other : ℚ := 1 / 5  -- (1 - 3/4) / 5

-- Define the probability of choosing each die
def choose_die_prob : ℚ := 1 / 3

-- Define the event of rolling two sixes
def two_sixes_prob (die_prob : ℚ) : ℚ := die_prob * die_prob

-- Define the theorem
theorem three_sixes_probability :
  let total_two_sixes := 
    choose_die_prob * two_sixes_prob fair_die_prob +
    choose_die_prob * two_sixes_prob biased_die_2_prob_six +
    choose_die_prob * two_sixes_prob biased_die_3_prob_six
  let prob_fair_given_two_sixes := 
    (choose_die_prob * two_sixes_prob fair_die_prob) / total_two_sixes
  let prob_biased_2_given_two_sixes := 
    (choose_die_prob * two_sixes_prob biased_die_2_prob_six) / total_two_sixes
  let prob_biased_3_given_two_sixes := 
    (choose_die_prob * two_sixes_prob biased_die_3_prob_six) / total_two_sixes
  (prob_fair_given_two_sixes * fair_die_prob + 
   prob_biased_2_given_two_sixes * biased_die_2_prob_six + 
   prob_biased_3_given_two_sixes * biased_die_3_prob_six) = 109 / 148 := by
  sorry

end three_sixes_probability_l2922_292294


namespace max_product_of_three_numbers_l2922_292221

theorem max_product_of_three_numbers (n : ℕ+) :
  ∃ (a b c : ℕ), 
    a ∈ Finset.range (3*n + 2) ∧ 
    b ∈ Finset.range (3*n + 2) ∧ 
    c ∈ Finset.range (3*n + 2) ∧ 
    a + b + c = 3*n + 1 ∧
    a * b * c = n^3 + n^2 ∧
    ∀ (x y z : ℕ), 
      x ∈ Finset.range (3*n + 2) → 
      y ∈ Finset.range (3*n + 2) → 
      z ∈ Finset.range (3*n + 2) → 
      x + y + z = 3*n + 1 → 
      x * y * z ≤ n^3 + n^2 := by
  sorry

end max_product_of_three_numbers_l2922_292221


namespace negation_or_false_implies_and_false_l2922_292225

theorem negation_or_false_implies_and_false (p q : Prop) : 
  ¬(¬(p ∨ q)) → ¬(p ∧ q) := by
  sorry

end negation_or_false_implies_and_false_l2922_292225


namespace repayment_plan_earnings_l2922_292266

def hourly_rate (hour : ℕ) : ℕ :=
  if hour % 8 = 0 then 8 else hour % 8

def total_earnings (hours : ℕ) : ℕ :=
  (List.range hours).map hourly_rate |>.sum

theorem repayment_plan_earnings :
  total_earnings 50 = 219 :=
by sorry

end repayment_plan_earnings_l2922_292266


namespace smallest_enclosing_circle_radius_l2922_292243

theorem smallest_enclosing_circle_radius (r : ℝ) : 
  (∃ (A B C O : ℝ × ℝ),
    -- Three unit circles touching each other
    dist A B = 2 ∧ dist B C = 2 ∧ dist C A = 2 ∧
    -- O is the center of the enclosing circle
    dist O A = r ∧ dist O B = r ∧ dist O C = r ∧
    -- r is the smallest possible radius
    ∀ (r' : ℝ), (∃ (O' : ℝ × ℝ), dist O' A ≤ r' ∧ dist O' B ≤ r' ∧ dist O' C ≤ r') → r ≤ r') →
  r = 1 + 2 / Real.sqrt 3 :=
sorry

end smallest_enclosing_circle_radius_l2922_292243


namespace inscribed_circle_distance_relation_l2922_292202

/-- A right-angled triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Distance from the center of the inscribed circle to the right-angle vertex -/
  l : ℝ
  /-- Distance from the center of the inscribed circle to one of the other vertices -/
  m : ℝ
  /-- Distance from the center of the inscribed circle to the remaining vertex -/
  n : ℝ
  /-- l, m, and n are positive -/
  l_pos : l > 0
  m_pos : m > 0
  n_pos : n > 0

/-- The theorem relating the distances from the center of the inscribed circle to the vertices -/
theorem inscribed_circle_distance_relation (t : RightTriangleWithInscribedCircle) :
  1 / t.l^2 = 1 / t.m^2 + 1 / t.n^2 + Real.sqrt 2 / (t.m * t.n) := by
  sorry

end inscribed_circle_distance_relation_l2922_292202


namespace hyperbola_satisfies_equation_l2922_292253

/-- A hyperbola with given asymptotes and passing through a specific point -/
structure Hyperbola where
  -- The slope of the asymptotes
  asymptote_slope : ℝ
  -- The point through which the hyperbola passes
  point : ℝ × ℝ

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => x^2 / 14 - y^2 / 7 = 1

/-- Theorem stating that the given hyperbola satisfies the equation -/
theorem hyperbola_satisfies_equation (h : Hyperbola) 
  (h_asymptote : h.asymptote_slope = 1/2)
  (h_point : h.point = (4, Real.sqrt 2)) :
  hyperbola_equation h 4 (Real.sqrt 2) :=
sorry

end hyperbola_satisfies_equation_l2922_292253


namespace president_vice_president_selection_l2922_292206

/-- The number of people in the group -/
def groupSize : ℕ := 6

/-- The number of ways to choose a President and Vice-President when A is not President -/
def waysWithoutA : ℕ := groupSize * (groupSize - 1)

/-- The number of ways to choose a President and Vice-President when A is President -/
def waysWithA : ℕ := 1 * (groupSize - 2)

/-- The total number of ways to choose a President and Vice-President -/
def totalWays : ℕ := waysWithoutA + waysWithA

theorem president_vice_president_selection :
  totalWays = 34 := by
  sorry

end president_vice_president_selection_l2922_292206


namespace petya_bus_catch_l2922_292247

/-- Represents the maximum distance between bus stops that allows Petya to always catch the bus -/
def max_bus_stop_distance (v_p : ℝ) : ℝ :=
  0.12

/-- Theorem stating the maximum distance between bus stops for Petya to always catch the bus -/
theorem petya_bus_catch (v_p : ℝ) (h_v_p : v_p > 0) :
  let v_b := 5 * v_p
  let max_observation_distance := 0.6
  ∀ d : ℝ, d > 0 → d ≤ max_bus_stop_distance v_p →
    (∀ t : ℝ, t ≥ 0 → 
      (v_p * t ≤ d ∧ v_b * t ≤ max_observation_distance) ∨
      (v_p * t ≤ 2 * d ∧ v_b * t ≤ d + max_observation_distance)) :=
by
  sorry

end petya_bus_catch_l2922_292247


namespace centroid_property_l2922_292267

/-- The centroid of a triangle divides each median in the ratio 2:1 -/
def is_centroid (P Q R S : ℝ × ℝ) : Prop :=
  S.1 = (P.1 + Q.1 + R.1) / 3 ∧ S.2 = (P.2 + Q.2 + R.2) / 3

theorem centroid_property :
  let P : ℝ × ℝ := (2, 5)
  let Q : ℝ × ℝ := (9, 3)
  let R : ℝ × ℝ := (4, -4)
  let S : ℝ × ℝ := (x, y)
  is_centroid P Q R S → 9 * x + 4 * y = 151 / 3 := by
  sorry

end centroid_property_l2922_292267


namespace triangle_side_length_l2922_292252

theorem triangle_side_length 
  (A B C : Real) 
  (a b c : Real) 
  (h_area : (1/2) * b * c * Real.sin A = Real.sqrt 3)
  (h_angle : B = 60 * π / 180)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by sorry

end triangle_side_length_l2922_292252


namespace max_difference_m_n_l2922_292232

theorem max_difference_m_n (m n : ℤ) (hm : m > 0) (h : m^2 = 4*n^2 - 5*n + 16) :
  ∃ (m' n' : ℤ), m' > 0 ∧ m'^2 = 4*n'^2 - 5*n' + 16 ∧ |m' - n'| ≤ 33 ∧
  ∀ (m'' n'' : ℤ), m'' > 0 → m''^2 = 4*n''^2 - 5*n'' + 16 → |m'' - n''| ≤ |m' - n'| :=
sorry

end max_difference_m_n_l2922_292232


namespace equal_roots_quadratic_l2922_292217

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, mx^2 - mx + 2 = 0 ∧ (∀ y : ℝ, my^2 - my + 2 = 0 → y = x)) → m = 8 := by
  sorry

end equal_roots_quadratic_l2922_292217


namespace project_hours_difference_l2922_292297

theorem project_hours_difference (total_hours : ℕ) 
  (h_total : total_hours = 198) 
  (h_pat_kate : ∃ k : ℕ, pat_hours = 2 * k ∧ kate_hours = k)
  (h_pat_mark : ∃ m : ℕ, mark_hours = 3 * pat_hours ∧ pat_hours = m)
  (h_sum : pat_hours + kate_hours + mark_hours = total_hours) :
  mark_hours - kate_hours = 110 :=
sorry

end project_hours_difference_l2922_292297


namespace tetragon_diagonals_l2922_292227

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A tetragon is a polygon with 4 sides -/
def tetragon_sides : ℕ := 4

/-- Theorem: The number of diagonals in a tetragon is 2 -/
theorem tetragon_diagonals : num_diagonals tetragon_sides = 2 := by
  sorry

end tetragon_diagonals_l2922_292227


namespace quadratic_transformation_l2922_292219

theorem quadratic_transformation (p q r : ℤ) : 
  (∀ x, (p * x + q)^2 + r = 4 * x^2 - 16 * x + 15) → 
  p * q = -8 := by
  sorry

end quadratic_transformation_l2922_292219


namespace gwi_seed_count_l2922_292280

/-- The number of watermelon seeds Bom has -/
def bom_seeds : ℕ := 300

/-- The total number of watermelon seeds they have together -/
def total_seeds : ℕ := 1660

/-- The number of watermelon seeds Gwi has -/
def gwi_seeds : ℕ := 340

/-- The number of watermelon seeds Yeon has -/
def yeon_seeds : ℕ := 3 * gwi_seeds

theorem gwi_seed_count :
  bom_seeds < gwi_seeds ∧
  yeon_seeds = 3 * gwi_seeds ∧
  bom_seeds + gwi_seeds + yeon_seeds = total_seeds :=
by sorry

end gwi_seed_count_l2922_292280


namespace neighborhood_b_cookie_boxes_l2922_292209

/-- 
Proves that each home in Neighborhood B buys 5 boxes of cookies given the conditions of the problem.
-/
theorem neighborhood_b_cookie_boxes : 
  let neighborhood_a_homes : ℕ := 10
  let neighborhood_a_boxes_per_home : ℕ := 2
  let neighborhood_b_homes : ℕ := 5
  let price_per_box : ℕ := 2
  let better_neighborhood_revenue : ℕ := 50
  
  neighborhood_b_homes > 0 →
  (neighborhood_a_homes * neighborhood_a_boxes_per_home * price_per_box < better_neighborhood_revenue) →
  
  ∃ (boxes_per_home_b : ℕ),
    boxes_per_home_b * neighborhood_b_homes * price_per_box = better_neighborhood_revenue ∧
    boxes_per_home_b = 5 :=
by
  sorry

end neighborhood_b_cookie_boxes_l2922_292209


namespace olympiad_problem_l2922_292244

theorem olympiad_problem (a b c d : ℕ) 
  (h1 : (a * b - c * d) ∣ a)
  (h2 : (a * b - c * d) ∣ b)
  (h3 : (a * b - c * d) ∣ c)
  (h4 : (a * b - c * d) ∣ d) :
  a * b - c * d = 1 := by
sorry

end olympiad_problem_l2922_292244


namespace log_equality_implies_ratio_one_l2922_292291

theorem log_equality_implies_ratio_one (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (h : Real.log p / Real.log 8 = Real.log q / Real.log 18 ∧ 
       Real.log p / Real.log 8 = Real.log (p + 2*q) / Real.log 24) : 
  q / p = 1 := by
sorry

end log_equality_implies_ratio_one_l2922_292291


namespace december_sales_multiple_l2922_292200

theorem december_sales_multiple (A : ℝ) (M : ℝ) (h1 : M > 0) :
  M * A = 0.3125 * (11 * A + M * A) → M = 5 := by
sorry

end december_sales_multiple_l2922_292200


namespace arithmetic_sequence_150th_term_l2922_292276

/-- Arithmetic sequence with first term 3 and common difference 5 -/
def arithmeticSequence (n : ℕ) : ℕ := 3 + (n - 1) * 5

/-- The 150th term of the arithmetic sequence is 748 -/
theorem arithmetic_sequence_150th_term :
  arithmeticSequence 150 = 748 := by
  sorry

end arithmetic_sequence_150th_term_l2922_292276


namespace breaking_sequences_count_l2922_292237

/-- Represents the number of targets in each column -/
def targetDistribution : List Nat := [4, 3, 3]

/-- The total number of targets -/
def totalTargets : Nat := targetDistribution.sum

/-- Calculates the number of different sequences to break all targets -/
def breakingSequences (dist : List Nat) : Nat :=
  Nat.factorial totalTargets / (dist.map Nat.factorial).prod

theorem breaking_sequences_count : breakingSequences targetDistribution = 4200 := by
  sorry

end breaking_sequences_count_l2922_292237


namespace stratified_sample_theorem_l2922_292258

/-- Represents the number of fish in a sample, given the total population, 
    sample size, and the count of a specific type of fish in the population -/
def stratified_sample_count (population : ℕ) (sample_size : ℕ) (fish_count : ℕ) : ℕ :=
  (fish_count * sample_size) / population

/-- Proves that in a stratified sample of size 20 drawn from a population of 200 fish, 
    where silver carp make up 20 of the population and common carp make up 40 of the population, 
    the number of silver carp and common carp together in the sample is 6 -/
theorem stratified_sample_theorem (total_population : ℕ) (sample_size : ℕ) 
  (silver_carp_count : ℕ) (common_carp_count : ℕ) 
  (h1 : total_population = 200) 
  (h2 : sample_size = 20) 
  (h3 : silver_carp_count = 20) 
  (h4 : common_carp_count = 40) : 
  stratified_sample_count total_population sample_size silver_carp_count + 
  stratified_sample_count total_population sample_size common_carp_count = 6 := by
  sorry

end stratified_sample_theorem_l2922_292258


namespace simplify_fraction_l2922_292281

theorem simplify_fraction : 18 * (8 / 12) * (1 / 6) = 2 := by
  sorry

end simplify_fraction_l2922_292281


namespace problem_statement_l2922_292299

theorem problem_statement (x y : ℝ) (h1 : x * y = 4) (h2 : x - y = 5) :
  x^2 + 5*x*y + y^2 = 53 := by
sorry

end problem_statement_l2922_292299


namespace expected_product_1000_flips_l2922_292263

/-- The expected value of the product of heads and tails for n fair coin flips -/
def expected_product (n : ℕ) : ℚ := n * (n - 1) / 4

/-- Theorem: The expected value of the product of heads and tails for 1000 fair coin flips is 249750 -/
theorem expected_product_1000_flips : 
  expected_product 1000 = 249750 := by sorry

end expected_product_1000_flips_l2922_292263


namespace sum_of_roots_equals_one_l2922_292210

theorem sum_of_roots_equals_one :
  ∀ x₁ x₂ : ℝ, (x₁ + 3) * (x₁ - 4) = 22 ∧ (x₂ + 3) * (x₂ - 4) = 22 → x₁ + x₂ = 1 := by
  sorry

end sum_of_roots_equals_one_l2922_292210


namespace product_of_conjugates_l2922_292245

theorem product_of_conjugates (x p q : ℝ) :
  (x + p / 2 - Real.sqrt (p^2 / 4 - q)) * (x + p / 2 + Real.sqrt (p^2 / 4 - q)) = x^2 + p * x + q :=
by sorry

end product_of_conjugates_l2922_292245


namespace system_solution_l2922_292239

theorem system_solution :
  ∃! (x y : ℚ), (7 * x = -10 - 3 * y) ∧ (4 * x = 5 * y - 32) ∧
  (x = -219 / 88) ∧ (y = 97 / 22) := by
  sorry

end system_solution_l2922_292239


namespace gcd_factorial_eight_ten_l2922_292272

theorem gcd_factorial_eight_ten : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end gcd_factorial_eight_ten_l2922_292272


namespace arithmetic_sequence_2010th_term_l2922_292277

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_2010th_term 
  (p q : ℝ) 
  (h1 : 9 = arithmetic_sequence p (2 * q) 2)
  (h2 : 3 * p - q = arithmetic_sequence p (2 * q) 3)
  (h3 : 3 * p + q = arithmetic_sequence p (2 * q) 4) :
  arithmetic_sequence p (2 * q) 2010 = 8041 := by
sorry

end arithmetic_sequence_2010th_term_l2922_292277


namespace total_animals_legoland_animals_l2922_292278

/-- Given a ratio of kangaroos to koalas and the total number of kangaroos,
    calculate the total number of animals (koalas and kangaroos). -/
theorem total_animals (ratio : ℕ) (num_kangaroos : ℕ) : ℕ :=
  let num_koalas := num_kangaroos / ratio
  num_koalas + num_kangaroos

/-- Prove that given 5 kangaroos for each koala and 180 kangaroos in total,
    the total number of koalas and kangaroos is 216. -/
theorem legoland_animals : total_animals 5 180 = 216 := by
  sorry

end total_animals_legoland_animals_l2922_292278


namespace quadratic_is_square_of_binomial_l2922_292265

/-- A quadratic expression is a square of a binomial if and only if its coefficients satisfy certain conditions -/
theorem quadratic_is_square_of_binomial (b : ℝ) : 
  (∃ (t u : ℝ), ∀ x, b * x^2 + 8 * x + 4 = (t * x + u)^2) ↔ b = 4 := by
  sorry

end quadratic_is_square_of_binomial_l2922_292265


namespace unique_solution_sin_cos_equation_l2922_292264

theorem unique_solution_sin_cos_equation :
  ∃! (n : ℕ+), Real.sin (π / (2 * n.val)) * Real.cos (π / (2 * n.val)) = n.val / 8 := by
  sorry

end unique_solution_sin_cos_equation_l2922_292264


namespace particle_acceleration_l2922_292259

-- Define the displacement function
def s (t : ℝ) : ℝ := t^2 - t + 6

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 2 * t - 1

-- Define the acceleration function as the derivative of velocity
def a (t : ℝ) : ℝ := 2

-- Theorem statement
theorem particle_acceleration (t : ℝ) (h : t ∈ Set.Icc 1 4) :
  a t = 2 := by
  sorry

end particle_acceleration_l2922_292259
