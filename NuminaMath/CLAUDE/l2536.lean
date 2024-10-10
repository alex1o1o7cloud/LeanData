import Mathlib

namespace divisibility_by_36_l2536_253654

theorem divisibility_by_36 : ∃! n : ℕ, n < 10 ∧ (6130 + n) % 36 = 0 :=
by
  -- The proof goes here
  sorry

end divisibility_by_36_l2536_253654


namespace percent_of_x_l2536_253696

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 5 + x / 25) / x = 24 / 100 := by
  sorry

end percent_of_x_l2536_253696


namespace average_increase_is_three_l2536_253601

/-- Represents a batsman's statistics -/
structure Batsman where
  total_runs : ℕ
  innings : ℕ
  average : ℚ

/-- Calculates the new average after an inning -/
def new_average (b : Batsman) (runs : ℕ) : ℚ :=
  (b.total_runs + runs) / (b.innings + 1)

/-- Theorem: The increase in average is 3 for the given conditions -/
theorem average_increase_is_three (b : Batsman) (h1 : b.innings = 16) 
    (h2 : new_average b 92 = 44) : 
    new_average b 92 - b.average = 3 := by
  sorry

#check average_increase_is_three

end average_increase_is_three_l2536_253601


namespace angle_ABH_measure_l2536_253629

/-- A regular octagon is a polygon with 8 equal sides and 8 equal angles. -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The measure of angle ABH in a regular octagon ABCDEFGH. -/
def angle_ABH (octagon : RegularOctagon) : ℝ := sorry

/-- Theorem: The measure of angle ABH in a regular octagon is 22.5 degrees. -/
theorem angle_ABH_measure (octagon : RegularOctagon) : 
  angle_ABH octagon = 22.5 := by sorry

end angle_ABH_measure_l2536_253629


namespace range_of_a_l2536_253612

-- Define the custom operation
def circleMultiply (x y : ℝ) : ℝ := x * (1 - y)

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, circleMultiply (x - a) (x + a) < 2) → 
  -1 < a ∧ a < 2 :=
by sorry

end range_of_a_l2536_253612


namespace triangle_perimeter_after_tripling_l2536_253694

theorem triangle_perimeter_after_tripling (a b c : ℝ) :
  a = 8 → b = 15 → c = 17 →
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) →
  (3 * a + 3 * b + 3 * c = 120) :=
by
  sorry

end triangle_perimeter_after_tripling_l2536_253694


namespace solution_count_l2536_253666

/-- The number of pairs of positive integers (x, y) that satisfy x^2 - y^2 = 45 -/
def count_solutions : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let (x, y) := p
    x > 0 ∧ y > 0 ∧ x^2 - y^2 = 45
  ) (Finset.product (Finset.range 46) (Finset.range 46))).card

theorem solution_count : count_solutions = 3 := by
  sorry

end solution_count_l2536_253666


namespace coin_value_calculation_l2536_253658

theorem coin_value_calculation (total_coins : ℕ) (dimes : ℕ) (nickels : ℕ) : 
  total_coins = 36 → 
  dimes = 26 → 
  nickels = total_coins - dimes → 
  (dimes * 10 + nickels * 5 : ℚ) / 100 = 3.1 := by
  sorry

end coin_value_calculation_l2536_253658


namespace exists_x0_sin_minus_tan_negative_l2536_253634

open Real

theorem exists_x0_sin_minus_tan_negative :
  ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < π/2 ∧ sin x₀ - tan x₀ < 0 := by
  sorry

end exists_x0_sin_minus_tan_negative_l2536_253634


namespace large_number_arithmetic_l2536_253664

theorem large_number_arithmetic : 
  999999999999 - 888888888888 + 111111111111 = 222222222222 := by
  sorry

end large_number_arithmetic_l2536_253664


namespace smallest_n_with_properties_l2536_253614

theorem smallest_n_with_properties : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (a : ℕ), 3 * n = a^2) ∧ 
  (∃ (b : ℕ), 2 * n = b^3) ∧ 
  (∃ (c : ℕ), 5 * n = c^5) ∧ 
  (∀ (m : ℕ), m > 0 → 
    ((∃ (x : ℕ), 3 * m = x^2) ∧ 
     (∃ (y : ℕ), 2 * m = y^3) ∧ 
     (∃ (z : ℕ), 5 * m = z^5)) → 
    m ≥ 7500) ∧
  n = 7500 :=
by sorry

end smallest_n_with_properties_l2536_253614


namespace quadratic_roots_relation_l2536_253699

theorem quadratic_roots_relation (m n p q : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0) :
  (∃ s₁ s₂ : ℝ, s₁ + s₂ = -p ∧ s₁ * s₂ = q ∧
               3 * s₁ + 3 * s₂ = -m ∧ 9 * s₁ * s₂ = n) →
  n / q = 9 := by
sorry

end quadratic_roots_relation_l2536_253699


namespace even_function_m_value_l2536_253608

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x in ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- The function f(x) = x^2 + (m + 2)x + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (m + 2) * x + 3

theorem even_function_m_value :
  ∀ m : ℝ, IsEven (f m) → m = -2 := by
  sorry

end even_function_m_value_l2536_253608


namespace intersection_A_B_l2536_253635

def A : Set ℝ := {-1, 0, 2, 3, 5}

def B : Set ℝ := {x | -1 < x ∧ x < 3}

theorem intersection_A_B : A ∩ B = {0, 2} := by
  sorry

end intersection_A_B_l2536_253635


namespace intersection_A_complement_B_l2536_253655

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x > 5}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | 2 < x ∧ x ≤ 5} := by sorry

end intersection_A_complement_B_l2536_253655


namespace no_four_distinct_real_roots_l2536_253602

theorem no_four_distinct_real_roots (a b : ℝ) : 
  ¬ (∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄) ∧
    (r₁^4 - 4*r₁^3 + 6*r₁^2 + a*r₁ + b = 0) ∧
    (r₂^4 - 4*r₂^3 + 6*r₂^2 + a*r₂ + b = 0) ∧
    (r₃^4 - 4*r₃^3 + 6*r₃^2 + a*r₃ + b = 0) ∧
    (r₄^4 - 4*r₄^3 + 6*r₄^2 + a*r₄ + b = 0)) :=
by sorry

end no_four_distinct_real_roots_l2536_253602


namespace regular_working_hours_is_eight_l2536_253657

/-- Represents the problem of finding regular working hours per day --/
def RegularWorkingHours :=
  {H : ℝ // 
    (20 * H * 2.40 + (175 - 20 * H) * 3.20 = 432) ∧ 
    (H > 0) ∧ 
    (H ≤ 24)}

/-- Theorem stating that the regular working hours per day is 8 --/
theorem regular_working_hours_is_eight : 
  ∃ (h : RegularWorkingHours), h.val = 8 := by
  sorry

end regular_working_hours_is_eight_l2536_253657


namespace equation_solution_l2536_253669

theorem equation_solution : ∃! x : ℝ, (1 / 6 + 6 / x = 10 / x + 1 / 15) ∧ x = 40 := by
  sorry

end equation_solution_l2536_253669


namespace sally_pen_distribution_l2536_253690

/-- Represents the problem of distributing pens to students --/
def pen_distribution (total_pens : ℕ) (num_students : ℕ) (pens_home : ℕ) : ℕ → Prop :=
  λ pens_per_student : ℕ =>
    let pens_given := pens_per_student * num_students
    let remainder := total_pens - pens_given
    let pens_in_locker := remainder / 2
    pens_in_locker + pens_home = remainder

theorem sally_pen_distribution :
  pen_distribution 342 44 17 7 := by
  sorry

#check sally_pen_distribution

end sally_pen_distribution_l2536_253690


namespace B_eq_A_pow2_l2536_253685

def A : ℕ → ℚ
  | 0 => 1
  | n + 1 => (A n + 2) / (A n + 1)

def B : ℕ → ℚ
  | 0 => 1
  | n + 1 => (B n^2 + 2) / (2 * B n)

theorem B_eq_A_pow2 (n : ℕ) : B (n + 1) = A (2^n) := by
  sorry

end B_eq_A_pow2_l2536_253685


namespace sets_intersection_union_and_subset_l2536_253656

def A (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 4}
def B : Set ℝ := {x | x < -5 ∨ x > 3}

theorem sets_intersection_union_and_subset :
  (∀ x, x ∈ A 1 ∩ B ↔ 3 < x ∧ x ≤ 5) ∧
  (∀ x, x ∈ A 1 ∪ B ↔ x < -5 ∨ x ≥ 1) ∧
  (∀ m, A m ⊆ B ↔ m < -9 ∨ m > 3) :=
sorry

end sets_intersection_union_and_subset_l2536_253656


namespace origin_outside_circle_l2536_253697

theorem origin_outside_circle (a : ℝ) (h : 0 < a ∧ a < 1) :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ x^2 + y^2 + 2*a*x + 2*y + (a-1)^2
  f (0, 0) > 0 := by sorry

end origin_outside_circle_l2536_253697


namespace product_with_9999_l2536_253600

theorem product_with_9999 (n : ℕ) : n * 9999 = 4691130840 → n = 469200 := by
  sorry

end product_with_9999_l2536_253600


namespace line_properties_l2536_253603

-- Define the line l₁
def l₁ (m : ℝ) (x y : ℝ) : Prop :=
  (m + 1) * x - (m - 3) * y - 8 = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (2, 2)

-- Define the line l₂
def l₂ (m : ℝ) (x y : ℝ) : Prop :=
  (m + 1) * x - (m - 3) * y = 0

-- Define the maximized distance line
def max_distance_line (x y : ℝ) : Prop :=
  x + y = 0

theorem line_properties :
  (∀ m : ℝ, l₁ m (fixed_point.1) (fixed_point.2)) ∧
  (∃ m : ℝ, ∀ x y : ℝ, l₂ m x y ↔ max_distance_line x y) :=
by sorry

end line_properties_l2536_253603


namespace ellipse_perpendicular_points_l2536_253618

/-- The ellipse on which points A and B lie -/
def Ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The condition that OA is perpendicular to OB -/
def Perpendicular (A B : ℝ × ℝ) : Prop := A.1 * B.1 + A.2 * B.2 = 0

/-- The condition that P is on segment AB -/
def OnSegment (P A B : ℝ × ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

/-- The condition that OP is perpendicular to AB -/
def Perpendicular_OP_AB (O P A B : ℝ × ℝ) : Prop := 
  (P.1 - O.1) * (B.1 - A.1) + (P.2 - O.2) * (B.2 - A.2) = 0

/-- The main theorem -/
theorem ellipse_perpendicular_points 
  (A B : ℝ × ℝ) 
  (hA : Ellipse A.1 A.2) 
  (hB : Ellipse B.1 B.2) 
  (hPerp : Perpendicular A B)
  (P : ℝ × ℝ)
  (hP : OnSegment P A B)
  (hPPerp : Perpendicular_OP_AB (0, 0) P A B) :
  (1 / (A.1^2 + A.2^2) + 1 / (B.1^2 + B.2^2) = 13/36) ∧
  (P.1^2 + P.2^2 = (6 * Real.sqrt 13 / 13)^2) := by
  sorry

end ellipse_perpendicular_points_l2536_253618


namespace pond_area_l2536_253662

/-- Given a square garden with a perimeter of 48 meters and an area not occupied by a pond of 124 square meters, the area of the pond is 20 square meters. -/
theorem pond_area (garden_perimeter : ℝ) (non_pond_area : ℝ) : 
  garden_perimeter = 48 →
  non_pond_area = 124 →
  (garden_perimeter / 4)^2 - non_pond_area = 20 := by
sorry

end pond_area_l2536_253662


namespace max_min_f_l2536_253671

-- Define the function f
def f (x : ℝ) : ℝ := 6 - 12 * x + x^3

-- Define the interval
def I : Set ℝ := {x | -1/3 ≤ x ∧ x ≤ 1}

-- Statement of the theorem
theorem max_min_f :
  ∃ (max min : ℝ),
    (∀ x ∈ I, f x ≤ max) ∧
    (∃ x ∈ I, f x = max) ∧
    (∀ x ∈ I, min ≤ f x) ∧
    (∃ x ∈ I, f x = min) ∧
    max = 27 ∧
    min = -5 := by
  sorry

end max_min_f_l2536_253671


namespace fair_coin_three_heads_probability_l2536_253639

theorem fair_coin_three_heads_probability :
  let n : ℕ := 7  -- number of coin tosses
  let k : ℕ := 3  -- number of heads we're looking for
  let total_outcomes : ℕ := 2^n  -- total number of possible outcomes
  let favorable_outcomes : ℕ := Nat.choose n k  -- number of ways to choose k heads from n tosses
  (favorable_outcomes : ℚ) / total_outcomes = 35 / 128 := by
  sorry

end fair_coin_three_heads_probability_l2536_253639


namespace school_count_correct_l2536_253651

/-- Represents the number of primary schools in a town. -/
def num_schools : ℕ := 4

/-- Represents the capacity of the first two schools. -/
def capacity_large : ℕ := 400

/-- Represents the capacity of the other two schools. -/
def capacity_small : ℕ := 340

/-- Represents the total capacity of all schools. -/
def total_capacity : ℕ := 1480

/-- Theorem stating that the number of schools is correct given the capacities. -/
theorem school_count_correct : 
  2 * capacity_large + 2 * capacity_small = total_capacity ∧
  num_schools = 2 + 2 := by sorry

end school_count_correct_l2536_253651


namespace sin_15_times_sin_75_l2536_253679

theorem sin_15_times_sin_75 : Real.sin (15 * π / 180) * Real.sin (75 * π / 180) = 1 / 4 := by
  sorry

end sin_15_times_sin_75_l2536_253679


namespace probability_not_same_intersection_is_two_thirds_l2536_253688

/-- Represents the number of officers -/
def num_officers : ℕ := 3

/-- Represents the number of intersections -/
def num_intersections : ℕ := 2

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := (num_officers.choose 2) * 2

/-- The number of arrangements where two specific officers are at the same intersection -/
def same_intersection_arrangements : ℕ := 2

/-- The probability that two specific officers are not at the same intersection -/
def probability_not_same_intersection : ℚ := 1 - (same_intersection_arrangements : ℚ) / total_arrangements

theorem probability_not_same_intersection_is_two_thirds :
  probability_not_same_intersection = 2/3 := by sorry

end probability_not_same_intersection_is_two_thirds_l2536_253688


namespace product_purely_imaginary_l2536_253646

theorem product_purely_imaginary (x : ℝ) : 
  (Complex.I : ℂ).im * ((x + Complex.I) * ((x^2 + 1 : ℝ) + Complex.I) * ((x^2 + 2 : ℝ) + Complex.I)).re = 0 ↔ 
  x^4 + x^3 + x^2 + 2*x - 2 = 0 :=
by sorry

#check product_purely_imaginary

end product_purely_imaginary_l2536_253646


namespace alternative_basis_l2536_253642

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (e₁ e₂ : V)

/-- Given that e₁ and e₂ form a basis for a plane, prove that e₁ + e₂ and e₁ - e₂ also form a basis for the same plane. -/
theorem alternative_basis (h : LinearIndependent ℝ ![e₁, e₂]) :
  LinearIndependent ℝ ![e₁ + e₂, e₁ - e₂] ∧ 
  Submodule.span ℝ {e₁, e₂} = Submodule.span ℝ {e₁ + e₂, e₁ - e₂} := by
  sorry

end alternative_basis_l2536_253642


namespace prob_three_even_out_of_six_l2536_253624

/-- The probability of rolling an even number on a fair 12-sided die -/
def prob_even : ℚ := 1 / 2

/-- The number of ways to choose 3 dice from 6 -/
def choose_3_from_6 : ℕ := 20

/-- The probability of a specific scenario where exactly 3 dice show even -/
def prob_specific_scenario : ℚ := (1 / 2) ^ 6

/-- The probability of exactly three out of six fair 12-sided dice showing an even number -/
theorem prob_three_even_out_of_six : 
  choose_3_from_6 * prob_specific_scenario = 5 / 16 := by sorry

end prob_three_even_out_of_six_l2536_253624


namespace max_floor_product_sum_l2536_253632

theorem max_floor_product_sum (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → x + y + z = 1399 →
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1399 →
  ⌊x⌋ * y + ⌊y⌋ * z + ⌊z⌋ * x ≤ ⌊a⌋ * b + ⌊b⌋ * c + ⌊c⌋ * a →
  ⌊a⌋ * b + ⌊b⌋ * c + ⌊c⌋ * a ≤ 652400 :=
by sorry

#check max_floor_product_sum

end max_floor_product_sum_l2536_253632


namespace original_ratio_proof_l2536_253692

/-- Represents the number of students in each category -/
structure StudentCount where
  boarders : ℕ
  dayStudents : ℕ

/-- The ratio of boarders to day students -/
def ratio (sc : StudentCount) : ℚ :=
  sc.boarders / sc.dayStudents

theorem original_ratio_proof (initial : StudentCount) (final : StudentCount) :
  initial.boarders = 330 →
  final.boarders = initial.boarders + 66 →
  ratio final = 1 / 2 →
  ratio initial = 5 / 12 := by
  sorry

#check original_ratio_proof

end original_ratio_proof_l2536_253692


namespace certain_number_problem_l2536_253689

theorem certain_number_problem (x : ℝ) : 0.75 * x = 0.5 * 900 → x = 600 := by
  sorry

end certain_number_problem_l2536_253689


namespace consecutive_zeros_count_is_3719_l2536_253637

/-- Sequence of numbers with no two consecutive zeros -/
def a : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n+2) => a (n+1) + a n

/-- The number of 12-digit positive integers with digits 0 or 1 
    that have at least two consecutive 0's -/
def consecutive_zeros_count : ℕ := 2^12 - a 12

theorem consecutive_zeros_count_is_3719 : consecutive_zeros_count = 3719 := by
  sorry

end consecutive_zeros_count_is_3719_l2536_253637


namespace division_remainder_problem_l2536_253661

theorem division_remainder_problem (larger smaller : ℕ) : 
  larger - smaller = 1515 →
  larger = 1600 →
  larger / smaller = 16 →
  larger % smaller = 240 := by
sorry

end division_remainder_problem_l2536_253661


namespace equation_solution_l2536_253678

theorem equation_solution : 
  ∀ x : ℝ, (Real.sqrt (5 * x - 2) + 12 / Real.sqrt (5 * x - 2) = 8) ↔ 
  (x = 38 / 5 ∨ x = 6 / 5) :=
by sorry

end equation_solution_l2536_253678


namespace overtime_rate_increase_l2536_253613

def regular_rate : ℚ := 16
def regular_hours : ℕ := 40
def total_compensation : ℚ := 1340
def total_hours : ℕ := 65

theorem overtime_rate_increase :
  let overtime_hours : ℕ := total_hours - regular_hours
  let regular_pay : ℚ := regular_rate * regular_hours
  let overtime_pay : ℚ := total_compensation - regular_pay
  let overtime_rate : ℚ := overtime_pay / overtime_hours
  let rate_increase : ℚ := (overtime_rate - regular_rate) / regular_rate
  rate_increase = 3/4 := by sorry

end overtime_rate_increase_l2536_253613


namespace jerry_throws_before_office_l2536_253670

def penalty_system (interrupt : ℕ) (insult : ℕ) (throw : ℕ) : ℕ :=
  5 * interrupt + 10 * insult + 25 * throw

def jerry_current_points : ℕ :=
  penalty_system 2 4 0

theorem jerry_throws_before_office : 
  ∃ (n : ℕ), 
    n = 2 ∧ 
    jerry_current_points + 25 * n < 100 ∧
    jerry_current_points + 25 * (n + 1) ≥ 100 :=
by sorry

end jerry_throws_before_office_l2536_253670


namespace dining_table_original_price_l2536_253620

theorem dining_table_original_price (discount_percentage : ℝ) (sale_price : ℝ) (original_price : ℝ) : 
  discount_percentage = 10 →
  sale_price = 450 →
  sale_price = original_price * (1 - discount_percentage / 100) →
  original_price = 500 := by
sorry

end dining_table_original_price_l2536_253620


namespace extracurricular_teams_problem_l2536_253649

theorem extracurricular_teams_problem (total_activities : ℕ) 
  (initial_ratio_tt : ℕ) (initial_ratio_bb : ℕ) 
  (new_ratio_tt : ℕ) (new_ratio_bb : ℕ) 
  (transfer : ℕ) :
  total_activities = 38 →
  initial_ratio_tt = 7 →
  initial_ratio_bb = 3 →
  new_ratio_tt = 3 →
  new_ratio_bb = 2 →
  transfer = 8 →
  ∃ (tt_original bb_original : ℕ),
    tt_original * initial_ratio_bb = bb_original * initial_ratio_tt ∧
    (tt_original - transfer) * new_ratio_bb = (bb_original + transfer) * new_ratio_tt ∧
    tt_original = 35 ∧
    bb_original = 15 := by
  sorry

end extracurricular_teams_problem_l2536_253649


namespace rope_cut_theorem_l2536_253653

theorem rope_cut_theorem (total_length : ℝ) (ratio_short : ℕ) (ratio_long : ℕ) 
  (h1 : total_length = 40)
  (h2 : ratio_short = 2)
  (h3 : ratio_long = 3) :
  (total_length * ratio_short) / (ratio_short + ratio_long) = 16 := by
  sorry

end rope_cut_theorem_l2536_253653


namespace triangle_inequality_l2536_253686

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) :
  (a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥
   (b + c - a) / a + (c + a - b) / b + (a + b - c) / c) ∧
  ((b + c - a) / a + (c + a - b) / b + (a + b - c) / c ≥ 3) := by
sorry

end triangle_inequality_l2536_253686


namespace diet_soda_bottles_l2536_253609

/-- Given a grocery store inventory, calculate the number of diet soda bottles. -/
theorem diet_soda_bottles (total_bottles regular_soda_bottles : ℕ) 
  (h1 : total_bottles = 38) 
  (h2 : regular_soda_bottles = 30) : 
  total_bottles - regular_soda_bottles = 8 := by
  sorry

end diet_soda_bottles_l2536_253609


namespace P_in_fourth_quadrant_iff_m_in_range_l2536_253648

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point being in the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The point P with coordinates dependent on m -/
def P (m : ℝ) : Point :=
  { x := m + 3, y := m - 2 }

/-- Theorem stating the range of m for P to be in the fourth quadrant -/
theorem P_in_fourth_quadrant_iff_m_in_range (m : ℝ) :
  in_fourth_quadrant (P m) ↔ -3 < m ∧ m < 2 := by sorry

end P_in_fourth_quadrant_iff_m_in_range_l2536_253648


namespace dvd_packs_theorem_l2536_253687

/-- The number of DVD packs that can be bought with a given amount of money -/
def dvd_packs (total_money : ℚ) (pack_cost : ℚ) : ℚ :=
  total_money / pack_cost

/-- Theorem: Given 110 dollars and a pack cost of 11 dollars, 10 DVD packs can be bought -/
theorem dvd_packs_theorem : dvd_packs 110 11 = 10 := by
  sorry

end dvd_packs_theorem_l2536_253687


namespace symmetry_condition_l2536_253680

def f (x a : ℝ) : ℝ := |x + 1| + |x - 1| + |x - a|

theorem symmetry_condition (a : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, f x a = f (2*k - x) a) ↔ a ∈ ({-3, 0, 3} : Set ℝ) := by
sorry

end symmetry_condition_l2536_253680


namespace inequality_proof_l2536_253619

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : y * z + z * x + x * y = 1) : 
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) ≤ (4/9) * Real.sqrt 3 := by
  sorry

end inequality_proof_l2536_253619


namespace larger_number_proof_l2536_253667

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1365) (h3 : L = 6 * S + 5) : L = 1637 := by
  sorry

end larger_number_proof_l2536_253667


namespace sheep_raising_profit_range_l2536_253693

/-- Represents the profit calculation for sheep raising with and without technical guidance. -/
theorem sheep_raising_profit_range (x : ℝ) : 
  x > 0 →
  (0.15 * (1 + 0.25*x) * (100000 - x) ≥ 0.15 * 100000) ↔ 
  (0 < x ∧ x ≤ 6) :=
by sorry

end sheep_raising_profit_range_l2536_253693


namespace existence_of_special_real_l2536_253645

theorem existence_of_special_real : ∃ A : ℝ, ∀ n : ℕ, ∃ m : ℕ, (⌊A^n⌋ : ℤ) + 2 = m^2 := by
  sorry

end existence_of_special_real_l2536_253645


namespace min_total_cost_l2536_253631

/-- Represents a dish with its price and quantity -/
structure Dish where
  price : ℕ
  quantity : ℕ

/-- Calculates the total price of a dish -/
def dishTotal (d : Dish) : ℕ := d.price * d.quantity

/-- Applies discount to an order based on its total -/
def applyDiscount (total : ℕ) : ℕ :=
  if total > 100 then total - 45
  else if total > 60 then total - 30
  else if total > 30 then total - 12
  else total

/-- Calculates the final cost of an order including delivery fee -/
def orderCost (total : ℕ) : ℕ := applyDiscount total + 3

/-- Theorem: The minimum total cost for Xiaoyu's order is 54 -/
theorem min_total_cost (dishes : List Dish) 
  (h1 : dishes = [
    ⟨30, 1⟩, -- Boiled Beef
    ⟨12, 1⟩, -- Vinegar Potatoes
    ⟨30, 1⟩, -- Spare Ribs in Black Bean Sauce
    ⟨12, 1⟩, -- Hand-Torn Cabbage
    ⟨3, 2⟩   -- Rice
  ]) :
  (dishes.map dishTotal).sum = 90 →
  ∃ (order1 order2 : ℕ), 
    order1 + order2 = 90 ∧ 
    orderCost order1 + orderCost order2 = 54 ∧
    ∀ (split1 split2 : ℕ), 
      split1 + split2 = 90 → 
      orderCost split1 + orderCost split2 ≥ 54 := by
  sorry

end min_total_cost_l2536_253631


namespace median_salary_proof_l2536_253615

/-- Represents a position in the company with its count and salary -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- Calculates the median salary given a list of positions -/
def medianSalary (positions : List Position) : Nat :=
  sorry

theorem median_salary_proof (positions : List Position) :
  positions = [
    ⟨"CEO", 1, 140000⟩,
    ⟨"Senior Vice-President", 4, 95000⟩,
    ⟨"Manager", 12, 80000⟩,
    ⟨"Team Leader", 8, 55000⟩,
    ⟨"Office Assistant", 38, 25000⟩
  ] →
  (positions.map (λ p => p.count)).sum = 63 →
  medianSalary positions = 25000 := by
  sorry

end median_salary_proof_l2536_253615


namespace simplify_inverse_product_l2536_253644

theorem simplify_inverse_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (((1 : ℝ) / a) * ((1 : ℝ) / (b + c)))⁻¹ = a * (b + c) := by sorry

end simplify_inverse_product_l2536_253644


namespace f_properties_l2536_253677

noncomputable def f (a b x : ℝ) : ℝ := Real.log x - a * x + b

theorem f_properties (a b : ℝ) (ha : a > 0) 
  (hf : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f a b x₁ = 0 ∧ f a b x₂ = 0) :
  (∃ (x : ℝ), ∀ (y : ℝ), f a b y ≤ f a b x) ∧
  (∀ (x₁ x₂ : ℝ), f a b x₁ = 0 → f a b x₂ = 0 → x₁ * x₂ < 1 / (a^2)) :=
by sorry

end f_properties_l2536_253677


namespace position_of_2010_l2536_253665

/-- The row number for a given positive integer in the arrangement -/
def row (n : ℕ) : ℕ := 
  (n.sqrt : ℕ) + (if n > (n.sqrt : ℕ)^2 then 1 else 0)

/-- The column number for a given positive integer in the arrangement -/
def column (n : ℕ) : ℕ := 
  n - (row n - 1)^2

/-- The theorem stating that 2010 appears in row 45 and column 74 -/
theorem position_of_2010 : row 2010 = 45 ∧ column 2010 = 74 := by
  sorry

end position_of_2010_l2536_253665


namespace cone_volume_l2536_253607

/-- Given a cone with slant height 15 cm and height 9 cm, its volume is 432π cubic centimeters -/
theorem cone_volume (s h r : ℝ) (hs : s = 15) (hh : h = 9) 
  (hr : r^2 = s^2 - h^2) : (1/3 : ℝ) * π * r^2 * h = 432 * π := by
  sorry

end cone_volume_l2536_253607


namespace series_sum_l2536_253605

-- Define the series
def series_term (n : ℕ) : ℚ := n / 5^n

-- State the theorem
theorem series_sum :
  (∑' n, series_term n) = 5/16 := by sorry

end series_sum_l2536_253605


namespace multiples_of_five_l2536_253636

theorem multiples_of_five (a b : ℤ) (ha : 5 ∣ a) (hb : 10 ∣ b) : 5 ∣ b ∧ 5 ∣ (a - b) := by
  sorry

end multiples_of_five_l2536_253636


namespace our_triangle_can_be_right_or_obtuse_l2536_253616

/-- A triangle with given perimeter and inradius -/
structure Triangle where
  perimeter : ℝ
  inradius : ℝ

/-- Definition of our specific triangle -/
def our_triangle : Triangle := { perimeter := 12, inradius := 1 }

/-- A function to determine if a triangle can be right-angled or obtuse-angled -/
def can_be_right_or_obtuse (t : Triangle) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = t.perimeter ∧
    (a * b * c) / (a + b + c) = 2 * t.inradius * t.perimeter ∧
    (a^2 + b^2 ≥ c^2 ∨ b^2 + c^2 ≥ a^2 ∨ c^2 + a^2 ≥ b^2)

/-- Theorem stating that our triangle can be right-angled or obtuse-angled -/
theorem our_triangle_can_be_right_or_obtuse :
  can_be_right_or_obtuse our_triangle := by
  sorry

end our_triangle_can_be_right_or_obtuse_l2536_253616


namespace floor_neg_seven_fourths_l2536_253604

theorem floor_neg_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end floor_neg_seven_fourths_l2536_253604


namespace largest_three_digit_product_l2536_253668

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_three_digit_product (n p q : ℕ) : 
  n ≥ 100 ∧ n < 1000 ∧
  is_prime p ∧ p < 10 ∧
  q < 10 ∧
  n = p * q * (10 * p + q) ∧
  p ≠ q ∧ p ≠ (10 * p + q) ∧ q ≠ (10 * p + q) →
  n ≤ 777 :=
sorry

end largest_three_digit_product_l2536_253668


namespace inequality_solution_set_l2536_253698

theorem inequality_solution_set (x : ℝ) :
  {x : ℝ | x^4 - 16*x^2 - 36*x > 0} = {x : ℝ | x < -4 ∨ (-4 < x ∧ x < -1) ∨ x > 9} :=
by sorry

end inequality_solution_set_l2536_253698


namespace reflection_across_x_axis_l2536_253633

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem reflection_across_x_axis :
  let A : Point := { x := 1, y := -2 }
  reflectAcrossXAxis A = { x := 1, y := 2 } := by
  sorry

end reflection_across_x_axis_l2536_253633


namespace range_start_divisible_by_eleven_l2536_253650

theorem range_start_divisible_by_eleven : ∃ (start : ℕ), 
  (start ≤ 79) ∧ 
  (∃ (a b c d : ℕ), 
    (start = 11 * a) ∧ 
    (start + 11 = 11 * b) ∧ 
    (start + 22 = 11 * c) ∧ 
    (start + 33 = 11 * d) ∧ 
    (start + 33 ≤ 79) ∧
    (start + 44 > 79)) ∧
  (start = 44) := by
sorry

end range_start_divisible_by_eleven_l2536_253650


namespace tan_315_degrees_l2536_253682

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end tan_315_degrees_l2536_253682


namespace other_root_is_one_l2536_253676

/-- Given a quadratic function f(x) = x^2 + 2x - a with a root of -3, 
    prove that the other root is 1. -/
theorem other_root_is_one (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^2 + 2*x - a) 
    (h2 : f (-3) = 0) : 
  ∃ x, x ≠ -3 ∧ f x = 0 ∧ x = 1 := by
sorry

end other_root_is_one_l2536_253676


namespace steves_book_earnings_l2536_253622

/-- The amount Steve gets for each copy of the book sold -/
def amount_per_copy : ℝ := 2

theorem steves_book_earnings :
  let total_copies : ℕ := 1000000
  let advance_copies : ℕ := 100000
  let agent_percentage : ℝ := 0.1
  let earnings_after_advance : ℝ := 1620000
  
  (total_copies - advance_copies : ℝ) * (1 - agent_percentage) * amount_per_copy = earnings_after_advance :=
by
  sorry

#check steves_book_earnings

end steves_book_earnings_l2536_253622


namespace equation_is_linear_l2536_253606

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants -/
def is_linear_equation_in_two_variables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y + c

/-- The equation x - 3y = -15 -/
def equation (x y : ℝ) : ℝ := x - 3 * y + 15

theorem equation_is_linear :
  is_linear_equation_in_two_variables equation := by
sorry

end equation_is_linear_l2536_253606


namespace penguin_fish_distribution_l2536_253641

theorem penguin_fish_distribution (days : ℕ) (fish_eaten_by_first_chick : ℕ) : 
  fish_eaten_by_first_chick = 44 →
  (days * 12 - fish_eaten_by_first_chick = 52) := by sorry

end penguin_fish_distribution_l2536_253641


namespace unique_solution_for_inequalities_l2536_253630

theorem unique_solution_for_inequalities :
  ∀ (x y z : ℝ),
    (1 + x^4 ≤ 2*(y - z)^2) ∧
    (1 + y^4 ≤ 2*(z - x)^2) ∧
    (1 + z^4 ≤ 2*(x - y)^2) →
    ((x = 1 ∧ y = 0 ∧ z = -1) ∨
     (x = 1 ∧ y = -1 ∧ z = 0) ∨
     (x = 0 ∧ y = 1 ∧ z = -1) ∨
     (x = 0 ∧ y = -1 ∧ z = 1) ∨
     (x = -1 ∧ y = 1 ∧ z = 0) ∨
     (x = -1 ∧ y = 0 ∧ z = 1)) :=
by sorry

end unique_solution_for_inequalities_l2536_253630


namespace expression_evaluation_l2536_253627

theorem expression_evaluation (a : ℝ) (h : a = Real.sqrt 5 + 1) :
  a / (a^2 - 2*a + 1) / (1 + 1/(a - 1)) = Real.sqrt 5 / 5 := by
  sorry

end expression_evaluation_l2536_253627


namespace cats_sold_during_sale_l2536_253659

/-- Represents the number of cats sold during a sale at a pet store. -/
def cats_sold (siamese_initial : ℕ) (house_initial : ℕ) (cats_left : ℕ) : ℕ :=
  siamese_initial + house_initial - cats_left

/-- Theorem stating that 19 cats were sold during the sale. -/
theorem cats_sold_during_sale :
  cats_sold 15 49 45 = 19 := by
  sorry

end cats_sold_during_sale_l2536_253659


namespace line_segment_length_l2536_253617

/-- Given points P, Q, R, and S arranged in order on a line segment,
    with PQ = 1, QR = 2PQ, and RS = 3QR, prove that the length of PS is 9. -/
theorem line_segment_length (P Q R S : ℝ) : 
  P < Q ∧ Q < R ∧ R < S →  -- Points are arranged in order
  Q - P = 1 →              -- PQ = 1
  R - Q = 2 * (Q - P) →    -- QR = 2PQ
  S - R = 3 * (R - Q) →    -- RS = 3QR
  S - P = 9 :=             -- PS = 9
by sorry

end line_segment_length_l2536_253617


namespace triangle_properties_l2536_253643

noncomputable section

-- Define the triangle ABC
def triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the theorem
theorem triangle_properties
  (a b c : ℝ)
  (h_triangle : triangle a b c)
  (h_angle_A : Real.cos (π/4) = b^2 + c^2 - a^2 / (2*b*c))
  (h_sides : b^2 - a^2 = (1/2) * c^2)
  (h_area : (1/2) * a * b * Real.sin (π/4) = 3) :
  Real.tan (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) = 2 ∧
  b = 3 ∧
  2 * π * ((a / (2 * Real.sin (π/4))) : ℝ) = Real.sqrt 10 * π :=
sorry

end

end triangle_properties_l2536_253643


namespace largest_base6_5digit_value_l2536_253691

def largest_base6_5digit : ℕ := 5 * 6^4 + 5 * 6^3 + 5 * 6^2 + 5 * 6^1 + 5 * 6^0

theorem largest_base6_5digit_value : largest_base6_5digit = 7775 := by
  sorry

end largest_base6_5digit_value_l2536_253691


namespace percentage_difference_l2536_253672

theorem percentage_difference (x y : ℝ) (P : ℝ) (h1 : x = y - (P / 100) * y) (h2 : y = 2 * x) :
  P = 50 := by
sorry

end percentage_difference_l2536_253672


namespace num_paths_through_F_and_H_l2536_253681

/-- A point on the grid --/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Calculate the number of paths between two points on a grid --/
def numPaths (start finish : GridPoint) : ℕ :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The grid layout --/
def E : GridPoint := ⟨0, 0⟩
def F : GridPoint := ⟨3, 2⟩
def H : GridPoint := ⟨5, 4⟩
def G : GridPoint := ⟨8, 4⟩

/-- The theorem to prove --/
theorem num_paths_through_F_and_H : 
  numPaths E F * numPaths F H * numPaths H G = 60 := by
  sorry

end num_paths_through_F_and_H_l2536_253681


namespace cheryl_material_problem_l2536_253684

theorem cheryl_material_problem (x : ℚ) :
  -- Cheryl buys x square yards of first material and 1/3 of second
  -- After project, 15/40 square yards left unused
  -- Total amount used is 1/3 square yards
  (x + 1/3 - 15/40 = 1/3) →
  -- The amount of first material needed is 3/8 square yards
  x = 3/8 := by
  sorry

end cheryl_material_problem_l2536_253684


namespace mean_equality_implies_x_value_l2536_253625

theorem mean_equality_implies_x_value : ∃ x : ℝ,
  (7 + 9 + 23) / 3 = (16 + x) / 2 → x = 10 := by sorry

end mean_equality_implies_x_value_l2536_253625


namespace arithmetic_sequence_middle_term_l2536_253623

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_middle_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a3 : a 3 = 4)
  (h_a5 : a 5 = m)
  (h_a7 : a 7 = 16) :
  m = 10 := by
  sorry

end arithmetic_sequence_middle_term_l2536_253623


namespace larger_number_given_hcf_lcm_factors_l2536_253663

theorem larger_number_given_hcf_lcm_factors (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  Nat.gcd a b = 63 ∧ 
  ∃ (k : ℕ), Nat.lcm a b = 63 * 11 * 17 * k →
  max a b = 1071 := by
  sorry

end larger_number_given_hcf_lcm_factors_l2536_253663


namespace cos_10_cos_20_minus_sin_10_sin_20_l2536_253660

theorem cos_10_cos_20_minus_sin_10_sin_20 :
  Real.cos (10 * π / 180) * Real.cos (20 * π / 180) -
  Real.sin (10 * π / 180) * Real.sin (20 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_10_cos_20_minus_sin_10_sin_20_l2536_253660


namespace angles_with_same_terminal_side_l2536_253652

def same_terminal_side (θ : ℝ) : Set ℝ :=
  {α | ∃ k : ℤ, α = k * 360 + θ}

def angle_range : Set ℝ :=
  {β | -360 ≤ β ∧ β < 720}

theorem angles_with_same_terminal_side :
  (same_terminal_side 60 ∩ angle_range) ∪ (same_terminal_side (-21) ∩ angle_range) =
  {-300, 60, 420, -21, 339, 699} := by
  sorry

end angles_with_same_terminal_side_l2536_253652


namespace visibility_time_correct_l2536_253640

/-- Represents a person walking along a straight path -/
structure Walker where
  speed : ℝ
  initial_position : ℝ × ℝ

/-- Represents the circular building -/
structure Building where
  center : ℝ × ℝ
  radius : ℝ

/-- The scenario of Jenny and Kenny walking -/
def walking_scenario : Building × Walker × Walker := 
  let building := { center := (0, 0), radius := 50 }
  let jenny := { speed := 2, initial_position := (-150, 100) }
  let kenny := { speed := 4, initial_position := (-150, -100) }
  (building, jenny, kenny)

/-- The time when Jenny and Kenny can see each other again -/
noncomputable def visibility_time (scenario : Building × Walker × Walker) : ℝ := 
  200  -- This is the value we want to prove

/-- The theorem stating that the visibility time is correct -/
theorem visibility_time_correct :
  let (building, jenny, kenny) := walking_scenario
  let t := visibility_time walking_scenario
  
  -- At time t, the line connecting Jenny and Kenny is tangent to the building
  ∃ (x y : ℝ),
    (x^2 + y^2 = building.radius^2) ∧
    ((jenny.initial_position.1 + jenny.speed * t - x) * (kenny.initial_position.2 - y) =
     (kenny.initial_position.1 + kenny.speed * t - x) * (jenny.initial_position.2 - y)) ∧
    (x * (jenny.initial_position.2 - y) + y * (x - jenny.initial_position.1 - jenny.speed * t) = 0) :=
by sorry

end visibility_time_correct_l2536_253640


namespace b6_b8_value_l2536_253647

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem b6_b8_value (a b : ℕ → ℝ) 
    (ha : arithmetic_sequence a)
    (hb : geometric_sequence b)
    (h1 : a 3 + a 11 = 8)
    (h2 : b 7 = a 7) :
  b 6 * b 8 = 16 := by
  sorry

end b6_b8_value_l2536_253647


namespace sphere_surface_area_l2536_253674

theorem sphere_surface_area (d : ℝ) (h : d = 2) : 
  4 * Real.pi * (d / 2)^2 = 4 * Real.pi := by
  sorry

end sphere_surface_area_l2536_253674


namespace real_y_condition_l2536_253611

theorem real_y_condition (x : ℝ) : 
  (∃ y : ℝ, 4 * y^2 + 6 * x * y + x + 10 = 0) ↔ (x ≤ -17/9 ∨ x ≥ 7/3) :=
by sorry

end real_y_condition_l2536_253611


namespace coordinates_wrt_y_axis_l2536_253638

/-- Given a point A(x,y) in a Cartesian coordinate system, 
    its coordinates with respect to the y-axis are (-x,y) -/
theorem coordinates_wrt_y_axis (x y : ℝ) : 
  let A : ℝ × ℝ := (x, y)
  let A_wrt_y_axis : ℝ × ℝ := (-x, y)
  A_wrt_y_axis = (- (A.1), A.2) :=
by sorry

end coordinates_wrt_y_axis_l2536_253638


namespace mixture_capacity_l2536_253621

/-- Represents the capacity and alcohol percentage of a vessel -/
structure Vessel where
  capacity : ℝ
  alcoholPercentage : ℝ

/-- Represents the mixture of two vessels -/
def Mixture (v1 v2 : Vessel) : ℝ × ℝ :=
  (v1.capacity + v2.capacity, v1.capacity * v1.alcoholPercentage + v2.capacity * v2.alcoholPercentage)

theorem mixture_capacity (v1 v2 : Vessel) (newConcentration : ℝ) :
  v1.capacity = 3 →
  v1.alcoholPercentage = 0.25 →
  v2.capacity = 5 →
  v2.alcoholPercentage = 0.40 →
  (Mixture v1 v2).1 = 8 →
  newConcentration = 0.275 →
  (Mixture v1 v2).2 / newConcentration = 10 := by
  sorry

#check mixture_capacity

end mixture_capacity_l2536_253621


namespace smallest_m_for_integral_solutions_l2536_253675

theorem smallest_m_for_integral_solutions : 
  (∃ m : ℕ, m > 0 ∧ 
   (∃ x y : ℤ, 10 * x^2 - m * x + 1980 = 0 ∧ 10 * y^2 - m * y + 1980 = 0 ∧ x ≠ y) ∧
   (∀ k : ℕ, k > 0 ∧ k < m → 
     ¬∃ x y : ℤ, 10 * x^2 - k * x + 1980 = 0 ∧ 10 * y^2 - k * y + 1980 = 0 ∧ x ≠ y)) ∧
  (∀ m : ℕ, m > 0 ∧ 
   (∃ x y : ℤ, 10 * x^2 - m * x + 1980 = 0 ∧ 10 * y^2 - m * y + 1980 = 0 ∧ x ≠ y) ∧
   (∀ k : ℕ, k > 0 ∧ k < m → 
     ¬∃ x y : ℤ, 10 * x^2 - k * x + 1980 = 0 ∧ 10 * y^2 - k * y + 1980 = 0 ∧ x ≠ y) →
   m = 290) :=
by sorry

end smallest_m_for_integral_solutions_l2536_253675


namespace solve_for_a_l2536_253673

theorem solve_for_a (a b d : ℤ) 
  (eq1 : a + b = d) 
  (eq2 : b + d = 7) 
  (eq3 : d = 4) : 
  a = 1 := by sorry

end solve_for_a_l2536_253673


namespace johns_breakfast_calories_l2536_253683

/-- Represents the number of calories in John's breakfast -/
def breakfast_calories : ℝ := 500

/-- Represents the number of calories in John's lunch -/
def lunch_calories : ℝ := 1.25 * breakfast_calories

/-- Represents the number of calories in John's dinner -/
def dinner_calories : ℝ := 2 * lunch_calories

/-- Represents the total number of calories from shakes -/
def shake_calories : ℝ := 3 * 300

/-- Represents the total number of calories John consumes in a day -/
def total_calories : ℝ := 3275

/-- Theorem stating that given the conditions, John's breakfast contains 500 calories -/
theorem johns_breakfast_calories :
  breakfast_calories + lunch_calories + dinner_calories + shake_calories = total_calories :=
by sorry

end johns_breakfast_calories_l2536_253683


namespace intersection_complement_equality_l2536_253610

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {2, 4, 5}
def B : Set Nat := {1, 3, 5, 7}

theorem intersection_complement_equality : A ∩ (U \ B) = {2, 4} := by
  sorry

end intersection_complement_equality_l2536_253610


namespace bells_toll_together_l2536_253626

theorem bells_toll_together (bell1 bell2 bell3 bell4 : ℕ) 
  (h1 : bell1 = 9) (h2 : bell2 = 10) (h3 : bell3 = 14) (h4 : bell4 = 18) :
  Nat.lcm bell1 (Nat.lcm bell2 (Nat.lcm bell3 bell4)) = 630 := by
  sorry

end bells_toll_together_l2536_253626


namespace cube_surface_area_increase_l2536_253695

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_area := 6 * L^2
  let new_length := 1.6 * L
  let new_area := 6 * new_length^2
  (new_area - original_area) / original_area * 100 = 156 := by
  sorry

end cube_surface_area_increase_l2536_253695


namespace fishing_ratio_l2536_253628

theorem fishing_ratio (sara_catch melanie_catch : ℕ) 
  (h1 : sara_catch = 5)
  (h2 : melanie_catch = 10) :
  (melanie_catch : ℚ) / sara_catch = 2 := by
  sorry

end fishing_ratio_l2536_253628
