import Mathlib

namespace NUMINAMATH_CALUDE_product_of_square_roots_l734_73435

theorem product_of_square_roots (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (48 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) = 60 * y * Real.sqrt (3 * y) :=
by sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l734_73435


namespace NUMINAMATH_CALUDE_range_of_a_l734_73430

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → a < x + 4 / x) → 
  a < 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l734_73430


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l734_73495

theorem solve_cubic_equation (y : ℝ) : 
  5 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + y^(1/3) → y = 1000 := by
  sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l734_73495


namespace NUMINAMATH_CALUDE_rice_mixture_cost_l734_73402

/-- The cost of a mixture of two rice varieties -/
def mixture_cost (c1 c2 r : ℚ) : ℚ :=
  (c1 * r + c2 * 1) / (r + 1)

theorem rice_mixture_cost :
  let c1 : ℚ := 5.5
  let c2 : ℚ := 8.75
  let r : ℚ := 5/8
  mixture_cost c1 c2 r = 7.5 := by
sorry

end NUMINAMATH_CALUDE_rice_mixture_cost_l734_73402


namespace NUMINAMATH_CALUDE_remainder_928927_div_6_l734_73418

theorem remainder_928927_div_6 : 928927 % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_928927_div_6_l734_73418


namespace NUMINAMATH_CALUDE_tablet_cash_savings_l734_73476

/-- Represents the savings when buying a tablet in cash versus installments -/
def tablet_savings (cash_price : ℕ) (down_payment : ℕ) 
  (first_4_months : ℕ) (next_4_months : ℕ) (last_4_months : ℕ) : ℕ :=
  (down_payment + 4 * first_4_months + 4 * next_4_months + 4 * last_4_months) - cash_price

/-- Theorem stating the savings when buying the tablet in cash -/
theorem tablet_cash_savings : 
  tablet_savings 450 100 40 35 30 = 70 := by
  sorry

end NUMINAMATH_CALUDE_tablet_cash_savings_l734_73476


namespace NUMINAMATH_CALUDE_num_tetrahedrons_in_cube_l734_73444

/-- The number of vertices in a cube. -/
def cube_vertices : ℕ := 8

/-- The number of vertices required to form a tetrahedron. -/
def tetrahedron_vertices : ℕ := 4

/-- The number of coplanar combinations in a cube (faces and diagonals). -/
def coplanar_combinations : ℕ := 12

/-- Calculates the number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of tetrahedrons that can be formed using the vertices of a cube. -/
theorem num_tetrahedrons_in_cube : 
  choose cube_vertices tetrahedron_vertices - coplanar_combinations = 58 := by
  sorry

end NUMINAMATH_CALUDE_num_tetrahedrons_in_cube_l734_73444


namespace NUMINAMATH_CALUDE_shekhar_shobha_age_ratio_l734_73413

/-- The ratio of Shekhar's age to Shobha's age -/
def age_ratio (shekhar_age shobha_age : ℕ) : ℚ :=
  shekhar_age / shobha_age

/-- Theorem stating the ratio of Shekhar's age to Shobha's age -/
theorem shekhar_shobha_age_ratio :
  ∃ (shekhar_age : ℕ),
    shekhar_age + 6 = 26 ∧
    age_ratio shekhar_age 15 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_shekhar_shobha_age_ratio_l734_73413


namespace NUMINAMATH_CALUDE_minimum_explorers_l734_73442

theorem minimum_explorers (large_capacity small_capacity : ℕ) 
  (h1 : large_capacity = 24)
  (h2 : small_capacity = 9)
  (explorers : ℕ) :
  (∃ k : ℕ, explorers = k * large_capacity - 4) ∧
  (∃ m : ℕ, explorers = m * small_capacity - 4) →
  explorers ≥ 68 :=
by sorry

end NUMINAMATH_CALUDE_minimum_explorers_l734_73442


namespace NUMINAMATH_CALUDE_inequality_proof_l734_73483

theorem inequality_proof (a b : ℝ) (h : a > b) : a^2 - a*b > b*a - b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l734_73483


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l734_73461

def num_men : ℕ := 10
def num_women : ℕ := 5
def num_chosen : ℕ := 4

theorem probability_at_least_one_woman :
  let total := num_men + num_women
  (1 - (Nat.choose num_men num_chosen : ℚ) / (Nat.choose total num_chosen : ℚ)) = 77 / 91 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l734_73461


namespace NUMINAMATH_CALUDE_hyperbola_equation_l734_73464

/-- The standard equation of a hyperbola that shares asymptotes with x²/3 - y² = 1
    and passes through (√3, 2) -/
theorem hyperbola_equation :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) ↔
    (y^2 / 3 - x^2 / 9 = 1)) ∧
  (3 / a^2 - 4 / b^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l734_73464


namespace NUMINAMATH_CALUDE_olivia_calculation_l734_73439

def round_to_nearest_ten (n : Int) : Int :=
  10 * ((n + 5) / 10)

theorem olivia_calculation : round_to_nearest_ten ((57 + 68) - 15) = 110 := by
  sorry

end NUMINAMATH_CALUDE_olivia_calculation_l734_73439


namespace NUMINAMATH_CALUDE_tobys_money_sharing_l734_73468

theorem tobys_money_sharing (initial_amount : ℚ) (brothers : ℕ) (share_fraction : ℚ) :
  initial_amount = 343 →
  brothers = 2 →
  share_fraction = 1/7 →
  initial_amount - (brothers * (share_fraction * initial_amount)) = 245 := by
  sorry

end NUMINAMATH_CALUDE_tobys_money_sharing_l734_73468


namespace NUMINAMATH_CALUDE_orange_harvest_after_six_days_l734_73462

/-- The number of sacks of oranges harvested after a given number of days. -/
def oranges_harvested (daily_rate : ℕ) (days : ℕ) : ℕ :=
  daily_rate * days

/-- Theorem stating that given a daily harvest rate of 83 sacks per day,
    the total number of sacks harvested after 6 days is equal to 498. -/
theorem orange_harvest_after_six_days :
  oranges_harvested 83 6 = 498 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_after_six_days_l734_73462


namespace NUMINAMATH_CALUDE_max_square_sum_max_square_sum_achievable_l734_73478

/-- The maximum value of x^2 + y^2 given 0 ≤ x ≤ 1 and 0 ≤ y ≤ 1 -/
theorem max_square_sum : 
  ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → x^2 + y^2 ≤ 2 := by
  sorry

/-- The maximum value 2 is achievable -/
theorem max_square_sum_achievable : 
  ∃ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x^2 + y^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_square_sum_max_square_sum_achievable_l734_73478


namespace NUMINAMATH_CALUDE_outlet_pipe_emptying_time_l734_73400

/-- Given an outlet pipe that empties 1/3 of a cistern in 8 minutes,
    prove that it takes 16 minutes to empty 2/3 of the cistern. -/
theorem outlet_pipe_emptying_time
  (emptying_rate : ℝ → ℝ)
  (h1 : emptying_rate 8 = 1/3)
  (h2 : ∀ t : ℝ, emptying_rate t = (t/8) * (1/3)) :
  ∃ t : ℝ, emptying_rate t = 2/3 ∧ t = 16 :=
sorry

end NUMINAMATH_CALUDE_outlet_pipe_emptying_time_l734_73400


namespace NUMINAMATH_CALUDE_seven_division_theorem_l734_73470

/-- Given a natural number n, returns the sum of its digits. -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Given a natural number n, returns the number of digits in n. -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Returns true if the given natural number consists only of the digit 7. -/
def all_sevens (n : ℕ) : Prop := sorry

theorem seven_division_theorem (N : ℕ) :
  digit_sum N = 2021 →
  ∃ q : ℕ, N = 7 * q ∧ all_sevens q →
  num_digits q = 503 := by sorry

end NUMINAMATH_CALUDE_seven_division_theorem_l734_73470


namespace NUMINAMATH_CALUDE_first_company_fixed_cost_l734_73486

/-- The fixed amount charged by the first rental company -/
def F : ℝ := 38.95

/-- The cost per mile for the first rental company -/
def cost_per_mile_first : ℝ := 0.31

/-- The fixed amount charged by Safety Rent A Truck -/
def fixed_cost_safety : ℝ := 41.95

/-- The cost per mile for Safety Rent A Truck -/
def cost_per_mile_safety : ℝ := 0.29

/-- The number of miles driven -/
def miles : ℝ := 150.0

theorem first_company_fixed_cost :
  F + cost_per_mile_first * miles = fixed_cost_safety + cost_per_mile_safety * miles :=
by sorry

end NUMINAMATH_CALUDE_first_company_fixed_cost_l734_73486


namespace NUMINAMATH_CALUDE_sum_of_xyz_l734_73482

theorem sum_of_xyz (x y z : ℝ) 
  (eq1 : x = y + z + 2)
  (eq2 : y = z + x + 1)
  (eq3 : z = x + y + 4) :
  x + y + z = -7 := by sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l734_73482


namespace NUMINAMATH_CALUDE_tissues_left_proof_l734_73490

/-- The number of tissues left after buying boxes and using some tissues. -/
def tissues_left (tissues_per_box : ℕ) (boxes_bought : ℕ) (tissues_used : ℕ) : ℕ :=
  tissues_per_box * boxes_bought - tissues_used

/-- Theorem: Given the conditions, prove that the number of tissues left is 270. -/
theorem tissues_left_proof :
  let tissues_per_box : ℕ := 160
  let boxes_bought : ℕ := 3
  let tissues_used : ℕ := 210
  tissues_left tissues_per_box boxes_bought tissues_used = 270 := by
  sorry

end NUMINAMATH_CALUDE_tissues_left_proof_l734_73490


namespace NUMINAMATH_CALUDE_ellipse_equation_1_ellipse_equation_2_l734_73414

-- Define the ellipse type
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the standard equation of an ellipse
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  y^2 / e.a^2 + x^2 / e.b^2 = 1

-- Define the focal length
def focal_length (e : Ellipse) : ℝ := 2 * e.c

-- Define the sum of distances from a point on the ellipse to the two focal points
def sum_of_distances (e : Ellipse) : ℝ := 2 * e.a

-- Theorem 1
theorem ellipse_equation_1 :
  ∃ (e : Ellipse),
    focal_length e = 4 ∧
    standard_equation e 3 2 ∧
    e.a = 4 ∧ e.b^2 = 12 :=
sorry

-- Theorem 2
theorem ellipse_equation_2 :
  ∃ (e : Ellipse),
    focal_length e = 10 ∧
    sum_of_distances e = 26 ∧
    ((e.a = 13 ∧ e.b = 12) ∨ (e.a = 12 ∧ e.b = 13)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_1_ellipse_equation_2_l734_73414


namespace NUMINAMATH_CALUDE_cube_root_of_110592_l734_73415

theorem cube_root_of_110592 :
  ∃ (n : ℕ), n^3 = 110592 ∧ n = 48 :=
by
  -- Define the number
  let number : ℕ := 110592

  -- Define the conditions
  have h1 : 10^3 = 1000 := by sorry
  have h2 : 100^3 = 1000000 := by sorry
  have h3 : 1000 < number ∧ number < 1000000 := by sorry
  have h4 : number % 10 = 2 := by sorry
  have h5 : ∀ (m : ℕ), m % 10 = 8 → (m^3) % 10 = 2 := by sorry
  have h6 : 4^3 = 64 := by sorry
  have h7 : 5^3 = 125 := by sorry
  have h8 : 64 < 110 ∧ 110 < 125 := by sorry

  -- Prove the theorem
  sorry

end NUMINAMATH_CALUDE_cube_root_of_110592_l734_73415


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_half_l734_73492

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem derivative_f_at_pi_half : 
  (deriv f) (π / 2) = 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_half_l734_73492


namespace NUMINAMATH_CALUDE_largest_hope_number_proof_l734_73496

/-- A Hope Number is a natural number with an odd number of divisors --/
def isHopeNumber (n : ℕ) : Prop := Odd (Nat.divisors n).card

/-- The largest Hope Number within 1000 --/
def largestHopeNumber : ℕ := 961

theorem largest_hope_number_proof :
  (∀ m : ℕ, m ≤ 1000 → isHopeNumber m → m ≤ largestHopeNumber) ∧
  isHopeNumber largestHopeNumber ∧
  largestHopeNumber ≤ 1000 :=
sorry

end NUMINAMATH_CALUDE_largest_hope_number_proof_l734_73496


namespace NUMINAMATH_CALUDE_aaron_can_lids_l734_73450

/-- The number of can lids Aaron is taking to the recycling center -/
def total_can_lids (num_boxes : ℕ) (existing_lids : ℕ) (lids_per_box : ℕ) : ℕ :=
  num_boxes * lids_per_box + existing_lids

/-- Proof that Aaron is taking 53 can lids to the recycling center -/
theorem aaron_can_lids : total_can_lids 3 14 13 = 53 := by
  sorry

end NUMINAMATH_CALUDE_aaron_can_lids_l734_73450


namespace NUMINAMATH_CALUDE_common_chord_equation_l734_73403

/-- Two circles in a 2D plane -/
structure TwoCircles where
  a : ℝ
  b : ℝ

/-- The equation of a line in 2D -/
structure LineEquation where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given two circles with a common chord of length 1, 
    prove that the equation of the common chord is 2ax + 2by - 3 = 0 -/
theorem common_chord_equation (circles : TwoCircles) : 
  ∃ (line : LineEquation), 
    line.A = 2 * circles.a ∧ 
    line.B = 2 * circles.b ∧ 
    line.C = -3 := by
  sorry

end NUMINAMATH_CALUDE_common_chord_equation_l734_73403


namespace NUMINAMATH_CALUDE_batsman_average_after_12_innings_l734_73463

/-- Represents a batsman's cricket statistics -/
structure BatsmanStats where
  innings : Nat
  totalRuns : Nat
  average : Rat

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : Nat) : Rat :=
  (stats.totalRuns + runsScored) / (stats.innings + 1)

/-- Theorem: The batsman's average after 12 innings is 47 -/
theorem batsman_average_after_12_innings
  (stats : BatsmanStats)
  (h1 : stats.innings = 11)
  (h2 : newAverage stats 80 = stats.average + 3)
  : newAverage stats 80 = 47 := by
  sorry

#eval newAverage ⟨11, 484, 44⟩ 80  -- Expected output: 47

end NUMINAMATH_CALUDE_batsman_average_after_12_innings_l734_73463


namespace NUMINAMATH_CALUDE_range_of_k_l734_73420

def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 2}
def B (k : ℝ) : Set ℝ := {x : ℝ | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1}

theorem range_of_k (k : ℝ) : A ⊇ B k ↔ -1 ≤ k ∧ k ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_l734_73420


namespace NUMINAMATH_CALUDE_gcd_problem_l734_73472

theorem gcd_problem (a b : ℕ+) (h : Nat.gcd a b = 12) :
  (∃ (x y : ℕ+), Nat.gcd x y = 12 ∧ Nat.gcd (12 * x) (18 * y) = 72) ∧
  (∀ (c d : ℕ+), Nat.gcd c d = 12 → Nat.gcd (12 * c) (18 * d) ≥ 72) :=
sorry

end NUMINAMATH_CALUDE_gcd_problem_l734_73472


namespace NUMINAMATH_CALUDE_room_length_proof_l734_73474

/-- Given a rectangular room with known width, total paving cost, and paving rate per square meter,
    prove that the length of the room is 5.5 meters. -/
theorem room_length_proof (width : ℝ) (total_cost : ℝ) (paving_rate : ℝ) :
  width = 3.75 ∧ total_cost = 20625 ∧ paving_rate = 1000 →
  total_cost / paving_rate / width = 5.5 :=
by sorry

end NUMINAMATH_CALUDE_room_length_proof_l734_73474


namespace NUMINAMATH_CALUDE_quarterback_sacks_l734_73436

theorem quarterback_sacks (total_attempts : ℕ) (no_throw_percentage : ℚ) (sack_ratio : ℚ) : 
  total_attempts = 80 → 
  no_throw_percentage = 30 / 100 → 
  sack_ratio = 1 / 2 → 
  ⌊(total_attempts : ℚ) * no_throw_percentage * sack_ratio⌋ = 12 := by
  sorry

end NUMINAMATH_CALUDE_quarterback_sacks_l734_73436


namespace NUMINAMATH_CALUDE_johns_donation_l734_73467

/-- Calculates the size of a donation that increases the average contribution by 75% to $100 when added to 10 existing contributions. -/
theorem johns_donation (initial_contributions : ℕ) (increase_percentage : ℚ) (new_average : ℚ) : 
  initial_contributions = 10 → 
  increase_percentage = 75 / 100 → 
  new_average = 100 → 
  (11 : ℚ) * new_average - initial_contributions * (new_average / (1 + increase_percentage)) = 3700 / 7 := by
  sorry

#eval (3700 : ℚ) / 7

end NUMINAMATH_CALUDE_johns_donation_l734_73467


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l734_73417

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2023 + 3) :
  (3 / (x + 3) - 1) / (x / (x^2 - 9)) = -Real.sqrt 2023 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l734_73417


namespace NUMINAMATH_CALUDE_cookies_left_l734_73499

theorem cookies_left (total_cookies : ℕ) (num_neighbors : ℕ) (intended_per_neighbor : ℕ) 
  (sarah_cookies : ℕ) (h1 : total_cookies = 150) (h2 : num_neighbors = 15) 
  (h3 : intended_per_neighbor = 10) (h4 : sarah_cookies = 12) : 
  total_cookies - (intended_per_neighbor * (num_neighbors - 1)) - sarah_cookies = 8 := by
  sorry

#check cookies_left

end NUMINAMATH_CALUDE_cookies_left_l734_73499


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_eq_one_l734_73493

/-- The function f(x) = ae^x - sin(x) has an extreme value at x = 0 -/
def has_extreme_value_at_zero (a : ℝ) : Prop :=
  let f := fun x => a * Real.exp x - Real.sin x
  ∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f x ≤ f 0 ∨ f x ≥ f 0

/-- If f(x) = ae^x - sin(x) has an extreme value at x = 0, then a = 1 -/
theorem extreme_value_implies_a_eq_one :
  ∀ a : ℝ, has_extreme_value_at_zero a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_eq_one_l734_73493


namespace NUMINAMATH_CALUDE_square_sum_inequality_l734_73484

theorem square_sum_inequality (a b : ℝ) 
  (h1 : a^3 - b^3 = 2) 
  (h2 : a^5 - b^5 ≥ 4) : 
  a^2 + b^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l734_73484


namespace NUMINAMATH_CALUDE_questions_per_exam_l734_73408

theorem questions_per_exam
  (num_classes : ℕ)
  (students_per_class : ℕ)
  (total_questions : ℕ)
  (h1 : num_classes = 5)
  (h2 : students_per_class = 35)
  (h3 : total_questions = 1750) :
  total_questions / (num_classes * students_per_class) = 10 := by
sorry

end NUMINAMATH_CALUDE_questions_per_exam_l734_73408


namespace NUMINAMATH_CALUDE_curve_and_point_properties_l734_73488

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = (p.1 + 1)^2}

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the property of the curve
def curve_property (p : ℝ × ℝ) : Prop :=
  p ∈ C → (p.1 - 1)^2 + p.2^2 = (p.1 + 1)^2

-- Define the equation of the curve
def curve_equation (p : ℝ × ℝ) : Prop :=
  p ∈ C → p.2^2 = 4 * p.1

-- Define the properties for point M
def M_properties (m : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  m ∈ C ∧ a ∈ C ∧ b ∈ C ∧
  ∃ (k : ℝ), k ≠ 0 ∧
    (a.2 - m.2) / (a.1 - m.1) = k ∧
    (b.2 - m.2) / (b.1 - m.1) = -k

-- Define the properties for points D and E
def DE_properties (d e : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  d ∈ C ∧ e ∈ C ∧
  (d.2 - e.2) / (d.1 - e.1) = -(b.1 - a.1) / (b.2 - a.2) ∧
  (d.1 - F.1) * (b.1 - a.1) + (d.2 - F.2) * (b.2 - a.2) = 0 ∧
  (e.1 - d.1)^2 + (e.2 - d.2)^2 = 64

-- State the theorem
theorem curve_and_point_properties :
  (∀ p, curve_property p) →
  (∀ p, curve_equation p) →
  ∀ m a b d e,
    M_properties m a b →
    DE_properties d e a b →
    m = (1, 2) ∨ m = (1, -2) := by sorry

end NUMINAMATH_CALUDE_curve_and_point_properties_l734_73488


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l734_73412

theorem greatest_divisor_four_consecutive_integers :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → 12 ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧
  (∀ (m : ℕ), m > 12 → ∃ (l : ℕ), l > 0 ∧ ¬(m ∣ (l * (l + 1) * (l + 2) * (l + 3)))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l734_73412


namespace NUMINAMATH_CALUDE_absolute_value_z_l734_73485

theorem absolute_value_z (w z : ℂ) : 
  w * z = 20 - 21 * I → Complex.abs w = Real.sqrt 29 → Complex.abs z = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_z_l734_73485


namespace NUMINAMATH_CALUDE_largest_good_set_size_l734_73421

/-- A set of positive integers is "good" if there exists a coloring with 2008 colors
    of all positive integers such that no number in the set is the sum of two
    different positive integers of the same color. -/
def isGoodSet (S : Set ℕ) : Prop :=
  ∃ (f : ℕ → Fin 2008), ∀ n ∈ S, ∀ x y : ℕ, x ≠ y → f x = f y → n ≠ x + y

/-- The set S(a, t) = {a+1, a+2, ..., a+t} for a positive integer a and natural number t. -/
def S (a t : ℕ) : Set ℕ := {n : ℕ | a + 1 ≤ n ∧ n ≤ a + t}

/-- The largest value of t for which S(a, t) is "good" for any positive integer a is 4014. -/
theorem largest_good_set_size :
  (∀ a : ℕ, a > 0 → isGoodSet (S a 4014)) ∧
  (∀ t : ℕ, t > 4014 → ∃ a : ℕ, a > 0 ∧ ¬isGoodSet (S a t)) :=
sorry

end NUMINAMATH_CALUDE_largest_good_set_size_l734_73421


namespace NUMINAMATH_CALUDE_remainder_97_103_mod_9_l734_73475

theorem remainder_97_103_mod_9 : (97 * 103) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_97_103_mod_9_l734_73475


namespace NUMINAMATH_CALUDE_reflection_theorem_l734_73404

noncomputable def C₁ (x : ℝ) : ℝ := Real.arccos (-x)

theorem reflection_theorem (x : ℝ) (h : 0 ≤ x ∧ x ≤ π) :
  ∃ y, C₁ y = x ∧ y = -Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_reflection_theorem_l734_73404


namespace NUMINAMATH_CALUDE_missing_sale_is_8562_l734_73487

/-- Calculates the missing sale amount given sales for 5 months and the average -/
def calculate_missing_sale (sale1 sale2 sale3 sale4 sale6 average : ℚ) : ℚ :=
  6 * average - (sale1 + sale2 + sale3 + sale4 + sale6)

theorem missing_sale_is_8562 :
  let sale1 : ℚ := 8435
  let sale2 : ℚ := 8927
  let sale3 : ℚ := 8855
  let sale4 : ℚ := 9230
  let sale6 : ℚ := 6991
  let average : ℚ := 8500
  calculate_missing_sale sale1 sale2 sale3 sale4 sale6 average = 8562 := by
  sorry

#eval calculate_missing_sale 8435 8927 8855 9230 6991 8500

end NUMINAMATH_CALUDE_missing_sale_is_8562_l734_73487


namespace NUMINAMATH_CALUDE_alex_has_largest_final_answer_l734_73416

def maria_operation (x : ℕ) : ℕ := ((x - 2) * 3) + 4

def alex_operation (x : ℕ) : ℕ := ((x * 3) - 3) + 4

def lee_operation (x : ℕ) : ℕ := ((x - 2) + 4) * 3

theorem alex_has_largest_final_answer :
  let maria_start := 12
  let alex_start := 15
  let lee_start := 13
  let maria_final := maria_operation maria_start
  let alex_final := alex_operation alex_start
  let lee_final := lee_operation lee_start
  alex_final > maria_final ∧ alex_final > lee_final :=
by sorry

end NUMINAMATH_CALUDE_alex_has_largest_final_answer_l734_73416


namespace NUMINAMATH_CALUDE_exchange_count_l734_73498

def number_of_people : ℕ := 10

def business_card_exchanges (n : ℕ) : ℕ := n.choose 2

theorem exchange_count : business_card_exchanges number_of_people = 45 := by
  sorry

end NUMINAMATH_CALUDE_exchange_count_l734_73498


namespace NUMINAMATH_CALUDE_utensil_pack_composition_l734_73440

/-- Represents a pack of utensils -/
structure UtensilPack where
  knives : ℕ
  forks : ℕ
  spoons : ℕ
  total : knives + forks + spoons = 30

/-- Theorem about the composition of utensil packs -/
theorem utensil_pack_composition 
  (pack : UtensilPack) 
  (h : 5 * pack.spoons = 50) : 
  pack.spoons = 10 ∧ pack.knives + pack.forks = 20 := by
  sorry


end NUMINAMATH_CALUDE_utensil_pack_composition_l734_73440


namespace NUMINAMATH_CALUDE_bird_migration_distance_l734_73437

/-- The combined distance traveled by a group of birds migrating between three lakes over two seasons. -/
def combined_distance (num_birds : ℕ) (distance1 : ℝ) (distance2 : ℝ) : ℝ :=
  num_birds * (distance1 + distance2)

/-- Theorem: The combined distance traveled by 20 birds over two seasons between three lakes is 2200 miles. -/
theorem bird_migration_distance :
  combined_distance 20 50 60 = 2200 := by
  sorry

end NUMINAMATH_CALUDE_bird_migration_distance_l734_73437


namespace NUMINAMATH_CALUDE_mall_audit_sampling_is_systematic_l734_73431

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Other

/-- Represents an invoice stub --/
structure InvoiceStub :=
  (number : ℕ)

/-- Represents a book of invoice stubs --/
def InvoiceBook := List InvoiceStub

/-- Represents a sampling process --/
structure SamplingProcess :=
  (book : InvoiceBook)
  (initialSelection : ℕ)
  (interval : ℕ)

/-- Determines if a sampling process is systematic --/
def isSystematicSampling (process : SamplingProcess) : Prop :=
  process.initialSelection ≤ 50 ∧ 
  process.interval = 50 ∧
  (∀ n : ℕ, (process.initialSelection + n * process.interval) ∈ (process.book.map InvoiceStub.number))

/-- The main theorem to prove --/
theorem mall_audit_sampling_is_systematic 
  (book : InvoiceBook)
  (initialStub : ℕ)
  (h1 : initialStub ≤ 50)
  (h2 : initialStub ∈ (book.map InvoiceStub.number))
  : isSystematicSampling ⟨book, initialStub, 50⟩ := by
  sorry

#check mall_audit_sampling_is_systematic

end NUMINAMATH_CALUDE_mall_audit_sampling_is_systematic_l734_73431


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l734_73455

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 4

/-- The x-intercept of the parabola -/
def x_intercept : ℝ := parabola 0

/-- The y-intercepts of the parabola -/
noncomputable def y_intercepts : Set ℝ := {y | parabola y = 0}

theorem parabola_intercepts_sum :
  ∃ (b c : ℝ), b ∈ y_intercepts ∧ c ∈ y_intercepts ∧ b ≠ c ∧ x_intercept + b + c = 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l734_73455


namespace NUMINAMATH_CALUDE_yellow_shirt_pairs_l734_73466

theorem yellow_shirt_pairs (blue_students : ℕ) (yellow_students : ℕ) (total_students : ℕ) (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  blue_students = 60 →
  yellow_students = 84 →
  total_students = 144 →
  total_pairs = 72 →
  blue_blue_pairs = 25 →
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 37 :=
by sorry

end NUMINAMATH_CALUDE_yellow_shirt_pairs_l734_73466


namespace NUMINAMATH_CALUDE_keegan_class_time_l734_73433

theorem keegan_class_time (total_hours : Real) (num_classes : Nat) (other_class_time : Real) :
  total_hours = 7.5 →
  num_classes = 7 →
  other_class_time = 72 / 60 →
  let other_classes_time := other_class_time * (num_classes - 2 : Real)
  let history_chem_time := total_hours - other_classes_time
  history_chem_time = 1.5 := by
sorry

end NUMINAMATH_CALUDE_keegan_class_time_l734_73433


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l734_73428

/-- Given a function f(x) = x^5 + ax^3 + bx - 8, if f(-2) = 10, then f(2) = -26 -/
theorem polynomial_symmetry (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^5 + a*x^3 + b*x - 8
  f (-2) = 10 → f 2 = -26 := by
sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l734_73428


namespace NUMINAMATH_CALUDE_root_sum_theorem_l734_73425

def polynomial (x : ℂ) : ℂ := x^5 + 2*x^4 + 3*x^3 + 4*x^2 + 5*x + 6

theorem root_sum_theorem (z₁ z₂ z₃ z₄ z₅ : ℂ) 
  (h₁ : polynomial z₁ = 0)
  (h₂ : polynomial z₂ = 0)
  (h₃ : polynomial z₃ = 0)
  (h₄ : polynomial z₄ = 0)
  (h₅ : polynomial z₅ = 0) :
  (z₁ / (z₁^2 + 1) + z₂ / (z₂^2 + 1) + z₃ / (z₃^2 + 1) + z₄ / (z₄^2 + 1) + z₅ / (z₅^2 + 1)) = 4/17 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l734_73425


namespace NUMINAMATH_CALUDE_total_hats_bought_l734_73429

theorem total_hats_bought (blue_cost green_cost total_price green_hats : ℕ) 
  (h1 : blue_cost = 6)
  (h2 : green_cost = 7)
  (h3 : total_price = 550)
  (h4 : green_hats = 40) :
  ∃ (blue_hats : ℕ), blue_cost * blue_hats + green_cost * green_hats = total_price ∧
                     blue_hats + green_hats = 85 := by
  sorry

end NUMINAMATH_CALUDE_total_hats_bought_l734_73429


namespace NUMINAMATH_CALUDE_no_simultaneous_squares_l734_73473

theorem no_simultaneous_squares : ¬∃ (x y : ℕ), ∃ (a b : ℕ), x^2 + y = a^2 ∧ x + y^2 = b^2 := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_squares_l734_73473


namespace NUMINAMATH_CALUDE_lines_one_unit_from_origin_l734_73432

theorem lines_one_unit_from_origin (x y : ℝ) (y' : ℝ → ℝ) :
  (∀ α : ℝ, x * Real.cos α + y * Real.sin α = 1) ↔
  y = x * y' x + Real.sqrt (1 + (y' x)^2) :=
sorry

end NUMINAMATH_CALUDE_lines_one_unit_from_origin_l734_73432


namespace NUMINAMATH_CALUDE_jules_blocks_to_walk_l734_73411

-- Define the given constants
def vacation_cost : ℚ := 1000
def family_members : ℕ := 5
def start_fee : ℚ := 2
def per_block_fee : ℚ := 1.25
def num_dogs : ℕ := 20

-- Define Jules' contribution
def jules_contribution : ℚ := vacation_cost / family_members

-- Define the function to calculate earnings based on number of blocks
def earnings (blocks : ℕ) : ℚ := num_dogs * (start_fee + per_block_fee * blocks)

-- Theorem statement
theorem jules_blocks_to_walk :
  ∃ (blocks : ℕ), earnings blocks ≥ jules_contribution ∧
    ∀ (b : ℕ), b < blocks → earnings b < jules_contribution :=
by sorry

end NUMINAMATH_CALUDE_jules_blocks_to_walk_l734_73411


namespace NUMINAMATH_CALUDE_right_triangle_with_60_degree_angle_l734_73446

theorem right_triangle_with_60_degree_angle (α β : ℝ) : 
  α = 60 → -- One acute angle is 60°
  α + β + 90 = 180 → -- Sum of angles in a triangle is 180°
  β = 30 := by -- The other acute angle is 30°
sorry

end NUMINAMATH_CALUDE_right_triangle_with_60_degree_angle_l734_73446


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l734_73465

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_lines 
  (m n : Line) (α β : Plane) :
  parallel m n →
  subset m α →
  perpendicular n β →
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l734_73465


namespace NUMINAMATH_CALUDE_exists_m_not_greater_l734_73489

theorem exists_m_not_greater (a b : ℝ) (h : a < b) : ∃ m : ℝ, m * a ≤ m * b := by
  sorry

end NUMINAMATH_CALUDE_exists_m_not_greater_l734_73489


namespace NUMINAMATH_CALUDE_solve_equation_l734_73453

theorem solve_equation (x y z a b c : ℤ) 
  (hx : x = -2272)
  (hy : y = 10^3 + 10^2 * c + 10 * b + a)
  (hz : z = 1)
  (heq : a * x + b * y + c * z = 1)
  (ha_pos : a > 0)
  (hb_pos : b > 0)
  (hc_pos : c > 0)
  (hab : a < b)
  (hbc : b < c) :
  y = 1987 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l734_73453


namespace NUMINAMATH_CALUDE_trains_crossing_time_l734_73406

/-- The time taken for two trains to cross each other -/
theorem trains_crossing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 200) 
  (h2 : length2 = 160) 
  (h3 : speed1 = 68 * 1000 / 3600) 
  (h4 : speed2 = 40 * 1000 / 3600) : 
  (length1 + length2) / (speed1 + speed2) = 12 :=
by sorry

end NUMINAMATH_CALUDE_trains_crossing_time_l734_73406


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l734_73427

theorem triangle_sine_inequality (A B C : Real) : 
  A > 0 → B > 0 → C > 0 → A + B + C = π →
  1 / Real.sin (A / 2) + 1 / Real.sin (B / 2) + 1 / Real.sin (C / 2) ≥ 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l734_73427


namespace NUMINAMATH_CALUDE_min_value_trigonometric_expression_l734_73407

theorem min_value_trigonometric_expression (x₁ x₂ x₃ x₄ : ℝ) 
  (h_positive : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) 
  (h_sum : x₁ + x₂ + x₃ + x₄ = π) : 
  (2 * Real.sin x₁ ^ 2 + 1 / Real.sin x₁ ^ 2) * 
  (2 * Real.sin x₂ ^ 2 + 1 / Real.sin x₂ ^ 2) * 
  (2 * Real.sin x₃ ^ 2 + 1 / Real.sin x₃ ^ 2) * 
  (2 * Real.sin x₄ ^ 2 + 1 / Real.sin x₄ ^ 2) ≥ 81 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_expression_l734_73407


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l734_73494

-- Define the quadratic function types
def QuadraticFunction (a b c : ℝ) := λ x : ℝ => a * x^2 + b * x + c

-- Define the solution set type
def SolutionSet := Set ℝ

-- State the theorem
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h1 : SolutionSet) 
  (h2 : h1 = {x : ℝ | x < (1/3) ∨ x > (1/2)}) 
  (h3 : h1 = {x : ℝ | QuadraticFunction a b c x < 0}) :
  {x : ℝ | QuadraticFunction c (-b) a x > 0} = Set.Ioo (-3) (-2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l734_73494


namespace NUMINAMATH_CALUDE_clock_centers_distance_l734_73405

/-- Two identically accurate clocks with hour hands -/
structure Clock where
  center : ℝ × ℝ
  hand_length : ℝ

/-- The configuration of two clocks -/
structure ClockPair where
  clock1 : Clock
  clock2 : Clock
  m : ℝ  -- Minimum distance between hour hand ends
  M : ℝ  -- Maximum distance between hour hand ends

/-- The theorem stating the distance between clock centers -/
theorem clock_centers_distance (cp : ClockPair) :
  let (x1, y1) := cp.clock1.center
  let (x2, y2) := cp.clock2.center
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = (cp.M + cp.m) / 2 := by
  sorry

end NUMINAMATH_CALUDE_clock_centers_distance_l734_73405


namespace NUMINAMATH_CALUDE_opposite_to_A_is_F_l734_73449

/-- Represents the labels of the squares --/
inductive Label
  | A | B | C | D | E | F

/-- Represents a cube formed by folding six connected squares --/
structure Cube where
  faces : Fin 6 → Label
  is_valid : ∀ (l : Label), ∃ (i : Fin 6), faces i = l

/-- Defines the opposite face relation on a cube --/
def opposite (c : Cube) (l1 l2 : Label) : Prop :=
  ∃ (i j : Fin 6), c.faces i = l1 ∧ c.faces j = l2 ∧ i ≠ j ∧
    ∀ (k : Fin 6), k ≠ i → k ≠ j → 
      ∃ (m : Fin 6), m ≠ i ∧ m ≠ j ∧ (c.faces k = c.faces m)

/-- Theorem stating that F is opposite to A in the cube --/
theorem opposite_to_A_is_F (c : Cube) : opposite c Label.A Label.F := by
  sorry

end NUMINAMATH_CALUDE_opposite_to_A_is_F_l734_73449


namespace NUMINAMATH_CALUDE_ten_machines_four_minutes_l734_73401

/-- The number of bottles produced by a given number of machines in a given time -/
def bottles_produced (machines : ℕ) (minutes : ℕ) : ℕ :=
  let bottles_per_minute := (270 * machines) / 5
  bottles_per_minute * minutes

/-- Theorem stating that 10 machines produce 2160 bottles in 4 minutes -/
theorem ten_machines_four_minutes :
  bottles_produced 10 4 = 2160 := by
  sorry

end NUMINAMATH_CALUDE_ten_machines_four_minutes_l734_73401


namespace NUMINAMATH_CALUDE_minimum_value_and_inequality_l734_73451

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 5|

-- State the theorem
theorem minimum_value_and_inequality :
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x₀ : ℝ, f x₀ = m) ∧ m = 6) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 6 → a^2 + b^2 + c^2 ≥ 12) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_and_inequality_l734_73451


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l734_73477

/-- Given vectors a and b in ℝ², if a is perpendicular to b, then the y-coordinate of b is -1 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a.1 * b.1 + a.2 * b.2 = 0) :
  a = (1, 3) → b.1 = 3 → b.2 = -1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l734_73477


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l734_73457

theorem imaginary_part_of_complex_fraction (a : ℝ) : 
  Complex.im ((1 + a * Complex.I) / Complex.I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l734_73457


namespace NUMINAMATH_CALUDE_subtracted_number_l734_73438

theorem subtracted_number (x n : ℚ) : 
  x / 4 - (x - n) / 6 = 1 → x = 6 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l734_73438


namespace NUMINAMATH_CALUDE_f_minimum_value_l734_73422

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- State the theorem
theorem f_minimum_value :
  (∀ x : ℝ, f x ≥ 3) ∧ (∃ x : ℝ, f x = 3) := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_value_l734_73422


namespace NUMINAMATH_CALUDE_right_triangle_area_l734_73423

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 8 * Real.sqrt 2 →
  angle = 45 * π / 180 →
  let area := (hypotenuse^2 / 4 : ℝ)
  area = 32 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l734_73423


namespace NUMINAMATH_CALUDE_perfect_square_condition_l734_73448

theorem perfect_square_condition (p : Nat) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  let n : Nat := ((p - 1) / 2) ^ 2
  ∃ (k : Nat), n * p + n^2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l734_73448


namespace NUMINAMATH_CALUDE_red_hair_count_example_l734_73426

/-- Given a class with a hair color ratio and total number of students,
    calculate the number of students with red hair. -/
def red_hair_count (red blonde black total : ℕ) : ℕ :=
  (red * total) / (red + blonde + black)

/-- Theorem: In a class of 48 students with a hair color ratio of 3 : 6 : 7
    (red : blonde : black), the number of students with red hair is 9. -/
theorem red_hair_count_example : red_hair_count 3 6 7 48 = 9 := by
  sorry

end NUMINAMATH_CALUDE_red_hair_count_example_l734_73426


namespace NUMINAMATH_CALUDE_root_product_fourth_power_l734_73481

theorem root_product_fourth_power (r s t : ℂ) : 
  (r^3 + 5*r + 4 = 0) → 
  (s^3 + 5*s + 4 = 0) → 
  (t^3 + 5*t + 4 = 0) → 
  (r+s)^4 * (s+t)^4 * (t+r)^4 = 256 := by
sorry

end NUMINAMATH_CALUDE_root_product_fourth_power_l734_73481


namespace NUMINAMATH_CALUDE_remainder_of_3_99_plus_5_mod_9_l734_73434

theorem remainder_of_3_99_plus_5_mod_9 : (3^99 + 5) % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_99_plus_5_mod_9_l734_73434


namespace NUMINAMATH_CALUDE_polynomial_roots_l734_73452

theorem polynomial_roots (r : ℝ) : 
  r^2 = r + 1 → r^5 = 5*r + 3 ∧ ∀ b c : ℤ, (∀ s : ℝ, s^2 = s + 1 → s^5 = b*s + c) → b = 5 ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l734_73452


namespace NUMINAMATH_CALUDE_intersection_equals_set_iff_complement_subset_l734_73480

universe u

theorem intersection_equals_set_iff_complement_subset {U : Type u} (A B : Set U) :
  A ∩ B = A ↔ (Bᶜ : Set U) ⊆ (Aᶜ : Set U) := by sorry

end NUMINAMATH_CALUDE_intersection_equals_set_iff_complement_subset_l734_73480


namespace NUMINAMATH_CALUDE_triangle_consecutive_numbers_l734_73424

/-- Represents the state of the triangle cells -/
def TriangleState := List Int

/-- Represents an operation on two adjacent cells -/
inductive Operation
| Add : Nat → Nat → Operation
| Subtract : Nat → Nat → Operation

/-- Checks if two cells are adjacent in the triangle -/
def are_adjacent (i j : Nat) : Bool := sorry

/-- Applies an operation to the triangle state -/
def apply_operation (state : TriangleState) (op : Operation) : TriangleState := sorry

/-- Checks if a list contains consecutive integers from n to n+8 -/
def is_consecutive_from_n (l : List Int) (n : Int) : Prop := sorry

/-- The main theorem -/
theorem triangle_consecutive_numbers :
  ∀ (initial_state : TriangleState),
  (initial_state.length = 9 ∧ initial_state.all (· = 0)) →
  ∃ (n : Int) (final_state : TriangleState),
  (∃ (ops : List Operation), 
    (∀ op ∈ ops, ∃ i j, are_adjacent i j ∧ (op = Operation.Add i j ∨ op = Operation.Subtract i j)) ∧
    (final_state = ops.foldl apply_operation initial_state)) ∧
  is_consecutive_from_n final_state n →
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_consecutive_numbers_l734_73424


namespace NUMINAMATH_CALUDE_power_product_equality_l734_73447

theorem power_product_equality : (-0.125)^2021 * 8^2022 = -8 := by sorry

end NUMINAMATH_CALUDE_power_product_equality_l734_73447


namespace NUMINAMATH_CALUDE_power_product_squared_l734_73491

theorem power_product_squared (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_squared_l734_73491


namespace NUMINAMATH_CALUDE_expected_imbalance_six_teams_l734_73445

/-- Represents a baseball league with n teams -/
structure BaseballLeague (n : ℕ) where
  teams : Fin n → Unit

/-- Represents the schedule of games in the league -/
def Schedule (n : ℕ) := Fin n → Fin n → Bool

/-- Calculates the imbalance (minimum number of undefeated teams) for a given schedule -/
def imbalance (n : ℕ) (schedule : Schedule n) : ℕ := sorry

/-- The expected value of the imbalance for a league with n teams -/
def expectedImbalance (n : ℕ) : ℚ := sorry

/-- Theorem: The expected imbalance for a 6-team league is 5055 / 2^15 -/
theorem expected_imbalance_six_teams :
  expectedImbalance 6 = 5055 / 2^15 := by sorry

end NUMINAMATH_CALUDE_expected_imbalance_six_teams_l734_73445


namespace NUMINAMATH_CALUDE_red_balls_per_box_l734_73459

theorem red_balls_per_box (total_balls : ℕ) (num_boxes : ℕ) (balls_per_box : ℕ) :
  total_balls = 10 →
  num_boxes = 5 →
  total_balls = num_boxes * balls_per_box →
  balls_per_box = 2 := by
sorry

end NUMINAMATH_CALUDE_red_balls_per_box_l734_73459


namespace NUMINAMATH_CALUDE_min_value_range_lower_bound_value_l734_73456

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + a * |x - 2|

-- Theorem for part I
theorem min_value_range (a : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m) → -1 ≤ a ∧ a ≤ 1 :=
by sorry

-- Theorem for part II
theorem lower_bound_value (a : ℝ) :
  (∀ (x : ℝ), f a x ≥ 1/2) → a = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_range_lower_bound_value_l734_73456


namespace NUMINAMATH_CALUDE_total_time_equals_sum_l734_73460

/-- The total time Porche initially had for homework -/
def total_time : ℕ := 180

/-- Time required for math homework -/
def math_time : ℕ := 45

/-- Time required for English homework -/
def english_time : ℕ := 30

/-- Time required for science homework -/
def science_time : ℕ := 50

/-- Time required for history homework -/
def history_time : ℕ := 25

/-- Time left for the special project -/
def project_time : ℕ := 30

/-- Theorem stating that the total time is the sum of all homework times -/
theorem total_time_equals_sum :
  total_time = math_time + english_time + science_time + history_time + project_time := by
  sorry

end NUMINAMATH_CALUDE_total_time_equals_sum_l734_73460


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_l734_73409

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular 
  (l m : Line) (α : Plane) :
  parallel m l → perpendicular m α → perpendicular l α :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_l734_73409


namespace NUMINAMATH_CALUDE_correct_proposition_l734_73410

-- Define proposition p
def p : Prop := ∀ a : ℝ, (∀ x : ℝ, a * x^2 + a * x + 1 > 0) → a ∈ Set.Ioo 0 4

-- Define proposition q
def q : Prop := (∀ x : ℝ, x > 5 → x^2 - 2*x - 8 > 0) ∧ (∃ x : ℝ, x^2 - 2*x - 8 > 0 ∧ x ≤ 5)

-- Theorem statement
theorem correct_proposition : (¬p) ∧ q := by sorry

end NUMINAMATH_CALUDE_correct_proposition_l734_73410


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l734_73469

theorem ratio_of_sum_and_difference (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (h : a + b = 7 * (a - b)) : a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l734_73469


namespace NUMINAMATH_CALUDE_sector_area_ratio_l734_73454

/-- Given a circular sector AOB with central angle α (in radians),
    and a line drawn through point B and the midpoint C of radius OA,
    the ratio of the area of triangle BCO to the area of figure ABC
    is sin(α) / (2α - sin(α)). -/
theorem sector_area_ratio (α : Real) :
  let R : Real := 1  -- Assume unit radius for simplicity
  let S : Real := (1/2) * R^2 * α  -- Area of sector AOB
  let S_BCO : Real := (1/4) * R^2 * Real.sin α  -- Area of triangle BCO
  let S_ABC : Real := S - S_BCO  -- Area of figure ABC
  S_BCO / S_ABC = Real.sin α / (2 * α - Real.sin α) := by
sorry

end NUMINAMATH_CALUDE_sector_area_ratio_l734_73454


namespace NUMINAMATH_CALUDE_pharmacy_tubs_l734_73479

def tubs_needed : ℕ := 100
def tubs_in_storage : ℕ := 20

def tubs_to_buy : ℕ := tubs_needed - tubs_in_storage

def tubs_from_new_vendor : ℕ := tubs_to_buy / 4

def tubs_from_usual_vendor : ℕ := tubs_needed - (tubs_in_storage + tubs_from_new_vendor)

theorem pharmacy_tubs :
  tubs_from_usual_vendor = 60 := by sorry

end NUMINAMATH_CALUDE_pharmacy_tubs_l734_73479


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_2pi_3_l734_73443

/-- Given an angle α in a Cartesian coordinate system with its vertex at the origin,
    its initial side on the non-negative x-axis, and its terminal side passing through (-1, 2),
    prove that sin(2α + 2π/3) = (4 - 3√3) / 10 -/
theorem sin_2alpha_plus_2pi_3 (α : Real) :
  (∃ (x y : Real), x = -1 ∧ y = 2 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin (2 * α + 2 * Real.pi / 3) = (4 - 3 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_2pi_3_l734_73443


namespace NUMINAMATH_CALUDE_green_tea_price_decrease_proof_l734_73471

/-- The percentage decrease in green tea price from June to July -/
def green_tea_price_decrease : ℝ := 90

/-- The cost per pound of green tea and coffee in June -/
def june_price : ℝ := 1

/-- The cost per pound of green tea in July -/
def july_green_tea_price : ℝ := 0.1

/-- The cost per pound of coffee in July -/
def july_coffee_price : ℝ := 2 * june_price

/-- The cost of 3 lbs of mixture containing equal quantities of green tea and coffee in July -/
def mixture_cost : ℝ := 3.15

theorem green_tea_price_decrease_proof :
  green_tea_price_decrease = (june_price - july_green_tea_price) / june_price * 100 ∧
  mixture_cost = 1.5 * july_green_tea_price + 1.5 * july_coffee_price :=
sorry

end NUMINAMATH_CALUDE_green_tea_price_decrease_proof_l734_73471


namespace NUMINAMATH_CALUDE_kiwifruit_problem_l734_73441

-- Define the structure for weight difference and box count
structure WeightDifference :=
  (difference : ℝ)
  (count : ℕ)

-- Define the problem parameters
def standard_weight : ℝ := 25
def total_boxes : ℕ := 20
def selling_price_per_kg : ℝ := 10.6

-- Define the weight differences
def weight_differences : List WeightDifference := [
  ⟨-3, 1⟩, ⟨-2, 4⟩, ⟨-1.5, 2⟩, ⟨0, 3⟩, ⟨1, 2⟩, ⟨2.5, 8⟩
]

-- Calculate the total overweight
def total_overweight : ℝ :=
  weight_differences.foldr (λ wd acc => acc + wd.difference * wd.count) 0

-- Calculate the total weight
def total_weight : ℝ :=
  standard_weight * total_boxes + total_overweight

-- Calculate the total selling price
def total_selling_price : ℝ :=
  total_weight * selling_price_per_kg

-- Theorem to prove
theorem kiwifruit_problem :
  total_overweight = 8 ∧ total_selling_price = 5384.8 := by
  sorry


end NUMINAMATH_CALUDE_kiwifruit_problem_l734_73441


namespace NUMINAMATH_CALUDE_inequality_solution_l734_73497

theorem inequality_solution (n : ℕ+) : 2*n - 5 < 5 - 2*n ↔ n = 1 ∨ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l734_73497


namespace NUMINAMATH_CALUDE_intersection_range_l734_73419

-- Define the semicircle
def semicircle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 4 ∧ y ≥ 2

-- Define the line
def line (x y k : ℝ) : Prop :=
  y = k * (x - 1) + 5

-- Define the condition for two distinct intersection points
def has_two_intersections (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    semicircle x₁ y₁ ∧ semicircle x₂ y₂ ∧
    line x₁ y₁ k ∧ line x₂ y₂ k

-- Theorem statement
theorem intersection_range :
  ∀ k : ℝ, has_two_intersections k ↔ 
    (k ∈ Set.Icc (-3/2) (-Real.sqrt 5/2) ∨ 
     k ∈ Set.Ioc (Real.sqrt 5/2) (3/2)) :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l734_73419


namespace NUMINAMATH_CALUDE_ellipse_focal_distance_l734_73458

-- Define the three given points
def p1 : ℝ × ℝ := (1, 5)
def p2 : ℝ × ℝ := (4, -3)
def p3 : ℝ × ℝ := (9, 5)

-- Define the ellipse type
structure Ellipse where
  endpoints : List (ℝ × ℝ)
  h_endpoints : endpoints.length = 3

-- Define the function to calculate the distance between foci
def focalDistance (e : Ellipse) : ℝ := sorry

-- Theorem statement
theorem ellipse_focal_distance :
  ∀ e : Ellipse, e.endpoints = [p1, p2, p3] → focalDistance e = 14 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focal_distance_l734_73458
