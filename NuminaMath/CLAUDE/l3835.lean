import Mathlib

namespace NUMINAMATH_CALUDE_overestimation_proof_l3835_383584

theorem overestimation_proof (p q k d : ℤ) 
  (p_round q_round k_round d_round : ℚ)
  (hp : p = 150) (hq : q = 50) (hk : k = 2) (hd : d = 3)
  (hp_round : p_round = 160) (hq_round : q_round = 45) 
  (hk_round : k_round = 1) (hd_round : d_round = 4) :
  (p_round / q_round - k_round + d_round) > (p / q - k + d) := by
  sorry

#check overestimation_proof

end NUMINAMATH_CALUDE_overestimation_proof_l3835_383584


namespace NUMINAMATH_CALUDE_unique_solution_l3835_383540

-- Define the system of equations
def equation1 (x y : ℝ) : Prop :=
  x^2*y + x*y^2 + 3*x + 3*y + 24 = 0

def equation2 (x y : ℝ) : Prop :=
  x^3*y - x*y^3 + 3*x^2 - 3*y^2 - 48 = 0

-- Theorem stating that (-3, -1) is the unique solution
theorem unique_solution :
  (∃! p : ℝ × ℝ, equation1 p.1 p.2 ∧ equation2 p.1 p.2) ∧
  (equation1 (-3) (-1) ∧ equation2 (-3) (-1)) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3835_383540


namespace NUMINAMATH_CALUDE_hamburgers_served_l3835_383596

theorem hamburgers_served (total : Nat) (leftover : Nat) (served : Nat) :
  total = 9 → leftover = 6 → served = total - leftover → served = 3 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_served_l3835_383596


namespace NUMINAMATH_CALUDE_philips_banana_collection_l3835_383511

theorem philips_banana_collection (group_size : ℕ) (num_groups : ℕ) (total_bananas : ℕ) : 
  group_size = 18 → num_groups = 10 → total_bananas = group_size * num_groups → total_bananas = 180 := by
  sorry

end NUMINAMATH_CALUDE_philips_banana_collection_l3835_383511


namespace NUMINAMATH_CALUDE_water_one_fifth_after_three_pourings_l3835_383514

def water_remaining (n : ℕ) : ℚ :=
  1 / (2 * n - 1)

theorem water_one_fifth_after_three_pourings :
  water_remaining 3 = 1 / 5 := by
  sorry

#check water_one_fifth_after_three_pourings

end NUMINAMATH_CALUDE_water_one_fifth_after_three_pourings_l3835_383514


namespace NUMINAMATH_CALUDE_expand_product_l3835_383523

theorem expand_product (x : ℝ) : (x + 4) * (2 * x - 9) = 2 * x^2 - x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3835_383523


namespace NUMINAMATH_CALUDE_m2n2_equals_four_l3835_383510

/-- The constant term in the binomial expansion of (√mx + n/x^2)^6 -/
def constant_term (m n : ℝ) : ℝ := 15 * m^2 * n^2

/-- The theorem stating that if the constant term is 60, then m^2n^2 = 4 -/
theorem m2n2_equals_four (m n : ℝ) (h : constant_term m n = 60) : m^2 * n^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_m2n2_equals_four_l3835_383510


namespace NUMINAMATH_CALUDE_only_B_is_valid_assignment_l3835_383534

-- Define what constitutes a valid assignment statement
def is_valid_assignment (s : String) : Prop :=
  match s.split (· == '=') with
  | [lhs, rhs] => lhs.trim.all Char.isAlpha && (rhs.trim ≠ "")
  | _ => false

-- Theorem stating that option B is the only valid assignment statement
theorem only_B_is_valid_assignment :
  is_valid_assignment "M=-M" ∧
  ¬is_valid_assignment "3=A" ∧
  ¬is_valid_assignment "B=A=2" ∧
  ¬is_valid_assignment "x+y=0" :=
sorry

end NUMINAMATH_CALUDE_only_B_is_valid_assignment_l3835_383534


namespace NUMINAMATH_CALUDE_expression_evaluation_l3835_383560

theorem expression_evaluation : 
  let a : ℚ := 5
  let b : ℚ := a + 4
  let c : ℚ := b - 12
  (a + 2 ≠ 0) → (b - 3 ≠ 0) → (c + 7 ≠ 0) →
  ((a + 4) / (a + 2)) * ((b + 1) / (b - 3)) * ((c + 10) / (c + 7)) = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3835_383560


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3835_383537

theorem inequality_solution_set (c : ℝ) : 
  (c / 3 ≤ 2 + c ∧ 2 + c < -2 * (1 + c)) ↔ c ∈ Set.Icc (-3) (-4/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3835_383537


namespace NUMINAMATH_CALUDE_parallelogram_area_l3835_383507

/-- The area of a parallelogram with one angle of 135 degrees and two consecutive sides of lengths 10 and 17 is equal to 85√2. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 17) (h3 : θ = 135 * π / 180) :
  a * b * Real.sin θ = 85 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3835_383507


namespace NUMINAMATH_CALUDE_prob_odd_divisor_15_factorial_l3835_383592

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The number of odd positive integer divisors of n -/
def num_odd_divisors (n : ℕ) : ℕ := sorry

/-- The probability of a randomly chosen positive integer divisor of n being odd -/
def prob_odd_divisor (n : ℕ) : ℚ :=
  (num_odd_divisors n : ℚ) / (num_divisors n : ℚ)

theorem prob_odd_divisor_15_factorial :
  prob_odd_divisor (factorial 15) = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_prob_odd_divisor_15_factorial_l3835_383592


namespace NUMINAMATH_CALUDE_right_focus_of_hyperbola_l3835_383531

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (2, 0)

/-- Theorem: The right focus of the hyperbola x^2/3 - y^2 = 1 is (2, 0) -/
theorem right_focus_of_hyperbola :
  ∀ (x y : ℝ), hyperbola x y → right_focus = (2, 0) := by
  sorry

end NUMINAMATH_CALUDE_right_focus_of_hyperbola_l3835_383531


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_existence_l3835_383587

theorem geometric_arithmetic_progression_existence :
  ∃ (q : ℝ) (i j k : ℕ), 
    1 < q ∧ 
    i < j ∧ j < k ∧ 
    q^j - q^i = q^k - q^j ∧
    1.9 < q :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_existence_l3835_383587


namespace NUMINAMATH_CALUDE_investment_scientific_notation_l3835_383500

/-- Represents the total investment in yuan -/
def total_investment : ℝ := 82000000000

/-- The scientific notation representation of the total investment -/
def scientific_notation : ℝ := 8.2 * (10 ^ 10)

/-- Theorem stating that the total investment equals its scientific notation representation -/
theorem investment_scientific_notation : total_investment = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_investment_scientific_notation_l3835_383500


namespace NUMINAMATH_CALUDE_smallest_positive_difference_l3835_383552

/-- Vovochka's method of adding two three-digit numbers -/
def vovochka_sum (a b c d e f : Nat) : Nat :=
  (a + d) * 1000 + (b + e) * 100 + (c + f)

/-- The correct sum of two three-digit numbers -/
def correct_sum (a b c d e f : Nat) : Nat :=
  (a + d) * 100 + (b + e) * 10 + (c + f)

/-- The difference between Vovochka's sum and the correct sum -/
def sum_difference (a b c d e f : Nat) : Int :=
  (vovochka_sum a b c d e f) - (correct_sum a b c d e f)

theorem smallest_positive_difference :
  ∀ a b c d e f : Nat,
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 →
  sum_difference a b c d e f ≠ 0 →
  1800 ≤ |sum_difference a b c d e f| :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_difference_l3835_383552


namespace NUMINAMATH_CALUDE_fate_region_is_correct_l3835_383578

def f (x : ℝ) := x^2 + 3*x + 2
def g (x : ℝ) := 2*x + 3

def is_fate_function (f g : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x ∈ D, |f x - g x| ≤ 1

def fate_region (f g : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | |f x - g x| ≤ 1}

theorem fate_region_is_correct :
  fate_region f g = Set.union (Set.Icc (-2) (-1)) (Set.Icc 0 1) :=
by sorry

end NUMINAMATH_CALUDE_fate_region_is_correct_l3835_383578


namespace NUMINAMATH_CALUDE_circles_relation_l3835_383530

theorem circles_relation (a b c : ℝ) :
  (∃ x : ℝ, x^2 - 2*a*x + b^2 = c*(b - a) ∧ 
   ∀ y : ℝ, y^2 - 2*a*y + b^2 = c*(b - a) → y = x) →
  (a = b ∨ c = a + b) :=
by sorry

end NUMINAMATH_CALUDE_circles_relation_l3835_383530


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l3835_383598

/-- A circle inscribed in a rectangle --/
structure InscribedCircle where
  radius : ℝ
  rectangle_length : ℝ
  rectangle_width : ℝ
  inscribed : rectangle_width = 2 * radius
  ratio : rectangle_length = 3 * rectangle_width

/-- The area of a rectangle with an inscribed circle of radius 8 and length-to-width ratio of 3:1 is 768 --/
theorem inscribed_circle_rectangle_area (c : InscribedCircle) 
  (h1 : c.radius = 8) : c.rectangle_length * c.rectangle_width = 768 := by
  sorry

#check inscribed_circle_rectangle_area

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l3835_383598


namespace NUMINAMATH_CALUDE_missing_number_solution_l3835_383583

theorem missing_number_solution : ∃ x : ℤ, (476 + 424) * x - 4 * 476 * 424 = 2704 ∧ x = 904 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_solution_l3835_383583


namespace NUMINAMATH_CALUDE_percentage_relation_l3835_383556

/-- Given three real numbers A, B, and C, where A is 6% of C and 20% of B,
    prove that B is 30% of C. -/
theorem percentage_relation (A B C : ℝ) 
  (h1 : A = 0.06 * C) 
  (h2 : A = 0.20 * B) : 
  B = 0.30 * C := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l3835_383556


namespace NUMINAMATH_CALUDE_collection_for_37_members_l3835_383508

/-- Calculates the total collection amount in rupees for a group of students -/
def total_collection_rupees (num_members : ℕ) (paise_per_rupee : ℕ) : ℚ :=
  (num_members * num_members : ℚ) / paise_per_rupee

/-- Proves that the total collection for 37 members is 13.69 rupees -/
theorem collection_for_37_members :
  total_collection_rupees 37 100 = 13.69 := by
  sorry

#eval total_collection_rupees 37 100

end NUMINAMATH_CALUDE_collection_for_37_members_l3835_383508


namespace NUMINAMATH_CALUDE_arrangement_satisfies_conditions_l3835_383535

-- Define the set of solutions
inductive Solution
| CuSO4
| CuCl2
| BaCl2
| AgNO3
| NH4OH
| HNO3
| HCl
| H2SO4

-- Define the arrangement of solutions in test tubes
def arrangement : Fin 8 → Solution
| 0 => Solution.CuSO4
| 1 => Solution.CuCl2
| 2 => Solution.BaCl2
| 3 => Solution.AgNO3
| 4 => Solution.NH4OH
| 5 => Solution.HNO3
| 6 => Solution.HCl
| 7 => Solution.H2SO4

-- Define the precipitation reaction between two solutions
def precipitates : Solution → Solution → Bool := sorry

-- Define the solubility of a precipitate in excess of a solution
def soluble_in_excess : Solution → Solution → Solution → Bool := sorry

-- Theorem stating that the arrangement satisfies all conditions
theorem arrangement_satisfies_conditions :
  -- Condition 1
  (precipitates (arrangement 0) (arrangement 2)) ∧
  (precipitates (arrangement 0) (arrangement 4)) ∧
  (precipitates (arrangement 0) (arrangement 3)) ∧
  -- Condition 2
  (soluble_in_excess (arrangement 0) (arrangement 3) (arrangement 4)) ∧
  (soluble_in_excess (arrangement 0) (arrangement 4) (arrangement 4)) ∧
  (soluble_in_excess (arrangement 0) (arrangement 4) (arrangement 5)) ∧
  (soluble_in_excess (arrangement 0) (arrangement 4) (arrangement 6)) ∧
  (soluble_in_excess (arrangement 0) (arrangement 4) (arrangement 7)) ∧
  -- Condition 3
  (precipitates (arrangement 1) (arrangement 3)) ∧
  (precipitates (arrangement 1) (arrangement 4)) ∧
  (soluble_in_excess (arrangement 1) (arrangement 3) (arrangement 4)) ∧
  (soluble_in_excess (arrangement 1) (arrangement 4) (arrangement 4)) ∧
  (soluble_in_excess (arrangement 1) (arrangement 4) (arrangement 5)) ∧
  (soluble_in_excess (arrangement 1) (arrangement 4) (arrangement 6)) ∧
  (soluble_in_excess (arrangement 1) (arrangement 4) (arrangement 7)) ∧
  -- Condition 4
  (precipitates (arrangement 2) (arrangement 0)) ∧
  (precipitates (arrangement 2) (arrangement 3)) ∧
  (precipitates (arrangement 2) (arrangement 7)) ∧
  (soluble_in_excess (arrangement 2) (arrangement 3) (arrangement 4)) ∧
  -- Condition 5
  (precipitates (arrangement 3) (arrangement 1)) ∧
  (precipitates (arrangement 3) (arrangement 4)) ∧
  (precipitates (arrangement 3) (arrangement 6)) ∧
  (precipitates (arrangement 3) (arrangement 0)) ∧
  (precipitates (arrangement 3) (arrangement 7)) ∧
  (soluble_in_excess (arrangement 3) (arrangement 1) (arrangement 4)) ∧
  (soluble_in_excess (arrangement 3) (arrangement 4) (arrangement 4)) ∧
  (soluble_in_excess (arrangement 3) (arrangement 6) (arrangement 4)) ∧
  (soluble_in_excess (arrangement 3) (arrangement 0) (arrangement 4)) ∧
  (soluble_in_excess (arrangement 3) (arrangement 7) (arrangement 4)) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_satisfies_conditions_l3835_383535


namespace NUMINAMATH_CALUDE_equal_angle_measure_l3835_383515

-- Define the structure of our shape
structure RectangleTriangleConfig where
  -- Rectangle properties
  rect_length : ℝ
  rect_height : ℝ
  rect_height_lt_length : rect_height < rect_length

  -- Triangle properties
  triangle_base : ℝ
  triangle_leg : ℝ

  -- Shared side property
  shared_side_eq_rect_length : triangle_base = rect_length

  -- Isosceles triangle property
  isosceles_triangle : triangle_base = 2 * triangle_leg

-- Theorem statement
theorem equal_angle_measure (config : RectangleTriangleConfig) :
  let angle := Real.arccos (config.triangle_leg / config.triangle_base) * (180 / Real.pi)
  angle = 45 := by sorry

end NUMINAMATH_CALUDE_equal_angle_measure_l3835_383515


namespace NUMINAMATH_CALUDE_jaymee_is_22_l3835_383512

def shara_age : ℕ := 10

def jaymee_age : ℕ := 2 * shara_age + 2

theorem jaymee_is_22 : jaymee_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_jaymee_is_22_l3835_383512


namespace NUMINAMATH_CALUDE_sara_marbles_l3835_383550

theorem sara_marbles (initial_marbles additional_marbles : ℝ) 
  (h1 : initial_marbles = 4892.5)
  (h2 : additional_marbles = 2337.8) :
  initial_marbles + additional_marbles = 7230.3 := by
sorry

end NUMINAMATH_CALUDE_sara_marbles_l3835_383550


namespace NUMINAMATH_CALUDE_fraction_sum_l3835_383574

theorem fraction_sum : (3 : ℚ) / 9 + (6 : ℚ) / 12 = (5 : ℚ) / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l3835_383574


namespace NUMINAMATH_CALUDE_tangent_line_value_l3835_383599

/-- A function f: ℝ → ℝ is tangent to the line y = -x + 8 at x = 5 if:
    1. f(5) = -5 + 8
    2. f'(5) = -1
-/
def is_tangent_at_5 (f : ℝ → ℝ) : Prop :=
  f 5 = 3 ∧ deriv f 5 = -1

theorem tangent_line_value (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_tangent : is_tangent_at_5 f) : f 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_value_l3835_383599


namespace NUMINAMATH_CALUDE_fifteen_buses_needed_l3835_383568

/-- Given the number of students, bus capacity, and pre-reserved bus seats,
    calculate the total number of buses needed. -/
def total_buses_needed (total_students : ℕ) (bus_capacity : ℕ) (pre_reserved_seats : ℕ) : ℕ :=
  let remaining_students := total_students - pre_reserved_seats
  let new_buses := (remaining_students + bus_capacity - 1) / bus_capacity
  new_buses + 1

/-- Theorem stating that 15 buses are needed for the given conditions. -/
theorem fifteen_buses_needed :
  total_buses_needed 635 45 20 = 15 := by
  sorry

#eval total_buses_needed 635 45 20

end NUMINAMATH_CALUDE_fifteen_buses_needed_l3835_383568


namespace NUMINAMATH_CALUDE_two_common_tangents_l3835_383518

/-- The number of common tangents between two circles -/
def num_common_tangents (C₁ C₂ : ℝ → ℝ → Prop) : ℕ :=
  sorry

/-- Circle C₁ equation -/
def C₁ (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 8*y - 8 = 0

/-- Circle C₂ equation -/
def C₂ (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 4*y - 1 = 0

/-- Theorem stating that there are 2 common tangents between C₁ and C₂ -/
theorem two_common_tangents : num_common_tangents C₁ C₂ = 2 :=
  sorry

end NUMINAMATH_CALUDE_two_common_tangents_l3835_383518


namespace NUMINAMATH_CALUDE_kendra_shirts_theorem_l3835_383577

/-- The number of shirts Kendra needs for a three-week period -/
def shirts_needed : ℕ :=
  let school_shirts := 5  -- 5 weekdays
  let club_shirts := 3    -- 3 days a week
  let saturday_shirts := 3 -- 1 for workout, 1 for art class, 1 for rest of the day
  let sunday_shirts := 3   -- 1 for church, 1 for volunteer work, 1 for rest of the day
  let weekly_shirts := school_shirts + club_shirts + saturday_shirts + sunday_shirts
  let weeks := 3
  weekly_shirts * weeks

/-- Theorem stating that Kendra needs 42 shirts for a three-week period -/
theorem kendra_shirts_theorem : shirts_needed = 42 := by
  sorry

end NUMINAMATH_CALUDE_kendra_shirts_theorem_l3835_383577


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3835_383529

theorem reciprocal_of_negative_2023 :
  ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3835_383529


namespace NUMINAMATH_CALUDE_min_value_expression_l3835_383519

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let expr := (|2*a - b + 2*a*(b - a)| + |b + 2*a - a*(b + 4*a)|) / Real.sqrt (4*a^2 + b^2)
  ∃ (min_val : ℝ), (∀ (a' b' : ℝ), a' > 0 → b' > 0 → expr ≥ min_val) ∧ min_val = Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3835_383519


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3835_383509

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) → a ∈ Set.Icc (-1) 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3835_383509


namespace NUMINAMATH_CALUDE_intersection_line_circle_l3835_383542

/-- Given a line intersecting a circle, prove the value of parameter a -/
theorem intersection_line_circle (a : ℝ) : 
  ∃ (A B : ℝ × ℝ),
    (∀ (x y : ℝ), x - y + 2*a = 0 → x^2 + y^2 - 2*a*y - 2 = 0 → (x, y) = A ∨ (x, y) = B) ∧
    ‖A - B‖ = 4 * Real.sqrt 3 →
    a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_circle_l3835_383542


namespace NUMINAMATH_CALUDE_siblings_selection_probability_l3835_383580

/-- The probability of three siblings being selected simultaneously -/
theorem siblings_selection_probability (px py pz : ℚ) 
  (hx : px = 1/7) (hy : py = 2/9) (hz : pz = 3/11) : 
  px * py * pz = 1/115.5 := by
  sorry

end NUMINAMATH_CALUDE_siblings_selection_probability_l3835_383580


namespace NUMINAMATH_CALUDE_proposition_truth_l3835_383527

theorem proposition_truth (p q : Prop) (hp : ¬p) (hq : ¬q) :
  (p ∨ ¬q) ∧ ¬(p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l3835_383527


namespace NUMINAMATH_CALUDE_closed_path_count_l3835_383588

/-- The number of distinct closed paths on a grid with total length 2n -/
def num_closed_paths (n : ℕ) : ℕ := (Nat.choose (2 * n) n) ^ 2

/-- Theorem stating that the number of distinct closed paths on a grid
    with total length 2n is equal to (C_{2n}^n)^2 -/
theorem closed_path_count (n : ℕ) : 
  num_closed_paths n = (Nat.choose (2 * n) n) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_closed_path_count_l3835_383588


namespace NUMINAMATH_CALUDE_inequality_proof_l3835_383517

theorem inequality_proof (x a : ℝ) (h : x < a ∧ a < 0) : x^2 > a*x ∧ a*x > a^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3835_383517


namespace NUMINAMATH_CALUDE_distance_between_points_l3835_383567

theorem distance_between_points (speed_A speed_B speed_C : ℝ) (extra_time : ℝ) : 
  speed_A = 100 →
  speed_B = 90 →
  speed_C = 75 →
  extra_time = 3 →
  ∃ (distance : ℝ), 
    distance / (speed_A + speed_B) + extra_time = distance / speed_C ∧
    distance = 650 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3835_383567


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3835_383593

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if a point lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem perpendicular_line_through_point : 
  ∃ (l : Line), 
    perpendicular l { a := 3, b := -5, c := 6 } ∧ 
    point_on_line (-1) 2 l ∧
    l = { a := 5, b := 3, c := -1 } :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3835_383593


namespace NUMINAMATH_CALUDE_grid_puzzle_solution_l3835_383536

-- Define the type for our grid cells
def Cell := Fin 16

-- Define our grid
structure Grid :=
  (A B C D E F G H J K L M N P Q R : Cell)

-- Define the conditions
def conditions (g : Grid) : Prop :=
  (g.A.val + g.C.val + g.F.val = 10) ∧
  (g.B.val + g.H.val = g.R.val) ∧
  (g.D.val - g.C.val = 13) ∧
  (g.E.val * g.M.val = 126) ∧
  (g.F.val + g.G.val = 21) ∧
  (g.G.val / g.J.val = 2) ∧
  (g.H.val * g.M.val = 36) ∧
  (g.J.val * g.P.val = 80) ∧
  (g.K.val - g.N.val = g.Q.val) ∧
  (∀ i j : Fin 16, i ≠ j → 
    g.A.val ≠ g.B.val ∧ g.A.val ≠ g.C.val ∧ g.A.val ≠ g.D.val ∧
    g.A.val ≠ g.E.val ∧ g.A.val ≠ g.F.val ∧ g.A.val ≠ g.G.val ∧
    g.A.val ≠ g.H.val ∧ g.A.val ≠ g.J.val ∧ g.A.val ≠ g.K.val ∧
    g.A.val ≠ g.L.val ∧ g.A.val ≠ g.M.val ∧ g.A.val ≠ g.N.val ∧
    g.A.val ≠ g.P.val ∧ g.A.val ≠ g.Q.val ∧ g.A.val ≠ g.R.val ∧
    g.B.val ≠ g.C.val ∧ g.B.val ≠ g.D.val ∧ g.B.val ≠ g.E.val ∧
    g.B.val ≠ g.F.val ∧ g.B.val ≠ g.G.val ∧ g.B.val ≠ g.H.val ∧
    g.B.val ≠ g.J.val ∧ g.B.val ≠ g.K.val ∧ g.B.val ≠ g.L.val ∧
    g.B.val ≠ g.M.val ∧ g.B.val ≠ g.N.val ∧ g.B.val ≠ g.P.val ∧
    g.B.val ≠ g.Q.val ∧ g.B.val ≠ g.R.val ∧ g.C.val ≠ g.D.val ∧
    g.C.val ≠ g.E.val ∧ g.C.val ≠ g.F.val ∧ g.C.val ≠ g.G.val ∧
    g.C.val ≠ g.H.val ∧ g.C.val ≠ g.J.val ∧ g.C.val ≠ g.K.val ∧
    g.C.val ≠ g.L.val ∧ g.C.val ≠ g.M.val ∧ g.C.val ≠ g.N.val ∧
    g.C.val ≠ g.P.val ∧ g.C.val ≠ g.Q.val ∧ g.C.val ≠ g.R.val ∧
    g.D.val ≠ g.E.val ∧ g.D.val ≠ g.F.val ∧ g.D.val ≠ g.G.val ∧
    g.D.val ≠ g.H.val ∧ g.D.val ≠ g.J.val ∧ g.D.val ≠ g.K.val ∧
    g.D.val ≠ g.L.val ∧ g.D.val ≠ g.M.val ∧ g.D.val ≠ g.N.val ∧
    g.D.val ≠ g.P.val ∧ g.D.val ≠ g.Q.val ∧ g.D.val ≠ g.R.val ∧
    g.E.val ≠ g.F.val ∧ g.E.val ≠ g.G.val ∧ g.E.val ≠ g.H.val ∧
    g.E.val ≠ g.J.val ∧ g.E.val ≠ g.K.val ∧ g.E.val ≠ g.L.val ∧
    g.E.val ≠ g.M.val ∧ g.E.val ≠ g.N.val ∧ g.E.val ≠ g.P.val ∧
    g.E.val ≠ g.Q.val ∧ g.E.val ≠ g.R.val ∧ g.F.val ≠ g.G.val ∧
    g.F.val ≠ g.H.val ∧ g.F.val ≠ g.J.val ∧ g.F.val ≠ g.K.val ∧
    g.F.val ≠ g.L.val ∧ g.F.val ≠ g.M.val ∧ g.F.val ≠ g.N.val ∧
    g.F.val ≠ g.P.val ∧ g.F.val ≠ g.Q.val ∧ g.F.val ≠ g.R.val ∧
    g.G.val ≠ g.H.val ∧ g.G.val ≠ g.J.val ∧ g.G.val ≠ g.K.val ∧
    g.G.val ≠ g.L.val ∧ g.G.val ≠ g.M.val ∧ g.G.val ≠ g.N.val ∧
    g.G.val ≠ g.P.val ∧ g.G.val ≠ g.Q.val ∧ g.G.val ≠ g.R.val ∧
    g.H.val ≠ g.J.val ∧ g.H.val ≠ g.K.val ∧ g.H.val ≠ g.L.val ∧
    g.H.val ≠ g.M.val ∧ g.H.val ≠ g.N.val ∧ g.H.val ≠ g.P.val ∧
    g.H.val ≠ g.Q.val ∧ g.H.val ≠ g.R.val ∧ g.J.val ≠ g.K.val ∧
    g.J.val ≠ g.L.val ∧ g.J.val ≠ g.M.val ∧ g.J.val ≠ g.N.val ∧
    g.J.val ≠ g.P.val ∧ g.J.val ≠ g.Q.val ∧ g.J.val ≠ g.R.val ∧
    g.K.val ≠ g.L.val ∧ g.K.val ≠ g.M.val ∧ g.K.val ≠ g.N.val ∧
    g.K.val ≠ g.P.val ∧ g.K.val ≠ g.Q.val ∧ g.K.val ≠ g.R.val ∧
    g.L.val ≠ g.M.val ∧ g.L.val ≠ g.N.val ∧ g.L.val ≠ g.P.val ∧
    g.L.val ≠ g.Q.val ∧ g.L.val ≠ g.R.val ∧ g.M.val ≠ g.N.val ∧
    g.M.val ≠ g.P.val ∧ g.M.val ≠ g.Q.val ∧ g.M.val ≠ g.R.val ∧
    g.N.val ≠ g.P.val ∧ g.N.val ≠ g.Q.val ∧ g.N.val ≠ g.R.val ∧
    g.P.val ≠ g.Q.val ∧ g.P.val ≠ g.R.val ∧ g.Q.val ≠ g.R.val)

-- State the theorem
theorem grid_puzzle_solution (g : Grid) (h : conditions g) : g.L.val = 6 := by
  sorry

end NUMINAMATH_CALUDE_grid_puzzle_solution_l3835_383536


namespace NUMINAMATH_CALUDE_least_value_quadratic_equation_l3835_383561

theorem least_value_quadratic_equation :
  let f : ℝ → ℝ := λ y => 3 * y^2 + 5 * y + 2
  ∃ y_min : ℝ, (f y_min = 4) ∧ (∀ y : ℝ, f y = 4 → y ≥ y_min) ∧ y_min = -2 := by
  sorry

end NUMINAMATH_CALUDE_least_value_quadratic_equation_l3835_383561


namespace NUMINAMATH_CALUDE_sum_first_ten_even_numbers_l3835_383594

-- Define the first 10 even numbers
def firstTenEvenNumbers : List Nat := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

-- Theorem: The sum of the first 10 even numbers is 110
theorem sum_first_ten_even_numbers :
  firstTenEvenNumbers.sum = 110 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_ten_even_numbers_l3835_383594


namespace NUMINAMATH_CALUDE_min_height_for_box_l3835_383565

/-- Represents the dimensions of a rectangular box with square base --/
structure BoxDimensions where
  base : ℕ  -- side length of the square base
  height : ℕ -- height of the box

/-- Calculates the surface area of the box --/
def surfaceArea (d : BoxDimensions) : ℕ :=
  2 * d.base^2 + 4 * d.base * d.height

/-- Checks if the box dimensions satisfy the height condition --/
def satisfiesHeightCondition (d : BoxDimensions) : Prop :=
  d.height = 2 * d.base + 1

/-- Checks if the box dimensions satisfy the surface area condition --/
def satisfiesSurfaceAreaCondition (d : BoxDimensions) : Prop :=
  surfaceArea d ≥ 130

/-- The main theorem stating the minimum height that satisfies all conditions --/
theorem min_height_for_box : 
  ∃ (d : BoxDimensions), 
    satisfiesHeightCondition d ∧ 
    satisfiesSurfaceAreaCondition d ∧ 
    (∀ (d' : BoxDimensions), 
      satisfiesHeightCondition d' ∧ 
      satisfiesSurfaceAreaCondition d' → 
      d.height ≤ d'.height) ∧
    d.height = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_height_for_box_l3835_383565


namespace NUMINAMATH_CALUDE_petya_can_buy_ice_cream_l3835_383564

theorem petya_can_buy_ice_cream (total : ℕ) (kolya vasya petya : ℕ) : 
  total = 2200 →
  kolya * 18 = vasya →
  total = kolya + vasya + petya →
  petya ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_petya_can_buy_ice_cream_l3835_383564


namespace NUMINAMATH_CALUDE_min_value_on_circle_l3835_383558

theorem min_value_on_circle (x y : ℝ) (h : (x - 2)^2 + (y - 2)^2 = 1) :
  ∃ (m : ℝ), (∀ (a b : ℝ), (a - 2)^2 + (b - 2)^2 = 1 → a^2 + b^2 ≥ m) ∧
  (m = 9 - 4 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l3835_383558


namespace NUMINAMATH_CALUDE_complex_condition_l3835_383543

theorem complex_condition (a b : ℝ) (hb : b ≠ 0) :
  let z : ℂ := Complex.mk a b
  (z^2 - 4*b*z).im = 0 → a = 2*b := by
  sorry

end NUMINAMATH_CALUDE_complex_condition_l3835_383543


namespace NUMINAMATH_CALUDE_johnson_family_reunion_l3835_383532

theorem johnson_family_reunion (num_children : ℕ) (num_adults : ℕ) (num_blue_adults : ℕ) : 
  num_children = 45 →
  num_adults = num_children / 3 →
  num_blue_adults = num_adults / 3 →
  num_adults - num_blue_adults = 10 := by
sorry

end NUMINAMATH_CALUDE_johnson_family_reunion_l3835_383532


namespace NUMINAMATH_CALUDE_min_time_35_minutes_l3835_383502

/-- Represents a rectangular parallelepiped -/
structure Brick where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a point moving on the surface of a brick -/
structure MovingPoint where
  v_x : ℝ → ℝ
  v_y : ℝ → ℝ
  velocity_constraint : ∀ t, (v_x t)^2 + 4*(v_y t)^2 = 1

/-- The minimum time for a point to travel from one vertex of the lower base
    to the opposite vertex of the upper base of a brick -/
def min_travel_time (b : Brick) (p : MovingPoint) : ℝ := sorry

/-- The theorem stating the minimum travel time for the given problem -/
theorem min_time_35_minutes (b : Brick) (p : MovingPoint)
    (h1 : b.length = 28)
    (h2 : b.width = 9)
    (h3 : b.height = 6) :
  min_travel_time b p = 35 := by sorry

end NUMINAMATH_CALUDE_min_time_35_minutes_l3835_383502


namespace NUMINAMATH_CALUDE_tea_mixture_ratio_l3835_383570

/-- Proves that the ratio of tea at Rs. 64 per kg to tea at Rs. 74 per kg is 1:1 in a mixture worth Rs. 69 per kg -/
theorem tea_mixture_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  64 * x + 74 * y = 69 * (x + y) → x = y := by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_ratio_l3835_383570


namespace NUMINAMATH_CALUDE_second_number_value_l3835_383520

theorem second_number_value (A B C : ℝ) 
  (sum_eq : A + B + C = 98)
  (ratio_AB : A / B = 2 / 3)
  (ratio_BC : B / C = 5 / 8)
  (pos_A : A > 0)
  (pos_B : B > 0)
  (pos_C : C > 0) : 
  B = 30 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l3835_383520


namespace NUMINAMATH_CALUDE_percentage_relationship_l3835_383546

theorem percentage_relationship (x y : ℝ) (h : x = y * (1 - 28.57142857142857 / 100)) :
  y = x * (1 + 28.57142857142857 / 100) :=
by sorry

end NUMINAMATH_CALUDE_percentage_relationship_l3835_383546


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l3835_383506

theorem least_k_for_inequality (k : ℤ) : 
  (∀ n : ℤ, n < k → (0.0010101 : ℝ) * (10 : ℝ) ^ (n : ℝ) ≤ 100) ∧ 
  (0.0010101 : ℝ) * (10 : ℝ) ^ (k : ℝ) > 100 → 
  k = 6 :=
sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l3835_383506


namespace NUMINAMATH_CALUDE_reseating_ways_l3835_383571

/-- Represents the number of ways n women can be reseated under the given rules -/
def S : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 3
  | n + 3 => S (n + 2) + S (n + 1) + S n

/-- The number of women -/
def num_women : ℕ := 12

/-- The theorem stating that the number of ways 12 women can be reseated is 927 -/
theorem reseating_ways : S num_women = 927 := by
  sorry

end NUMINAMATH_CALUDE_reseating_ways_l3835_383571


namespace NUMINAMATH_CALUDE_flower_planting_cost_l3835_383573

theorem flower_planting_cost (flower_cost soil_cost clay_pot_cost total_cost : ℕ) : 
  flower_cost = 9 →
  clay_pot_cost = flower_cost + 20 →
  soil_cost < flower_cost →
  total_cost = 45 →
  total_cost = flower_cost + clay_pot_cost + soil_cost →
  flower_cost - soil_cost = 2 := by
sorry

end NUMINAMATH_CALUDE_flower_planting_cost_l3835_383573


namespace NUMINAMATH_CALUDE_most_accurate_estimate_l3835_383576

-- Define the temperature range
def lower_bound : Float := 98.6
def upper_bound : Float := 99.1

-- Define a type for temperature readings
structure TemperatureReading where
  value : Float
  is_within_range : lower_bound ≤ value ∧ value ≤ upper_bound

-- Define a function to determine if a reading is closer to the upper bound
def closer_to_upper_bound (reading : TemperatureReading) : Prop :=
  reading.value > (lower_bound + upper_bound) / 2

-- Theorem statement
theorem most_accurate_estimate (reading : TemperatureReading) 
  (h : closer_to_upper_bound reading) : 
  upper_bound = 99.1 ∧ upper_bound - reading.value < reading.value - lower_bound :=
by
  sorry

end NUMINAMATH_CALUDE_most_accurate_estimate_l3835_383576


namespace NUMINAMATH_CALUDE_work_completion_time_l3835_383579

/-- The time it takes to complete a work given two workers with different rates and a specific work schedule. -/
theorem work_completion_time 
  (total_work : ℝ) 
  (p_rate : ℝ) 
  (q_rate : ℝ) 
  (p_solo_days : ℝ) 
  (hp : p_rate = total_work / 10) 
  (hq : q_rate = total_work / 6) 
  (hp_solo : p_solo_days = 2) : 
  p_solo_days + (total_work - p_solo_days * p_rate) / (p_rate + q_rate) = 5 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3835_383579


namespace NUMINAMATH_CALUDE_cone_water_volume_ratio_l3835_383562

theorem cone_water_volume_ratio (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let water_height := (2 / 3) * h
  let water_radius := (2 / 3) * r
  let cone_volume := (1 / 3) * Real.pi * r^2 * h
  let water_volume := (1 / 3) * Real.pi * water_radius^2 * water_height
  water_volume / cone_volume = 8 / 27 := by sorry

end NUMINAMATH_CALUDE_cone_water_volume_ratio_l3835_383562


namespace NUMINAMATH_CALUDE_larry_lost_stickers_l3835_383544

/-- Given that Larry starts with 93 stickers and ends up with 87 stickers,
    prove that he lost 6 stickers. -/
theorem larry_lost_stickers (initial : ℕ) (final : ℕ) (h1 : initial = 93) (h2 : final = 87) :
  initial - final = 6 := by
  sorry

end NUMINAMATH_CALUDE_larry_lost_stickers_l3835_383544


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l3835_383553

theorem at_least_one_not_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1 / b ≥ 2) ∨ (b + 1 / a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l3835_383553


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3835_383525

/-- The perimeter of a rectangle with longer sides 28cm and shorter sides 22cm is 100cm -/
theorem rectangle_perimeter : ℕ → ℕ → ℕ
  | 28, 22 => 100
  | _, _ => 0

#check rectangle_perimeter

end NUMINAMATH_CALUDE_rectangle_perimeter_l3835_383525


namespace NUMINAMATH_CALUDE_relay_race_total_time_l3835_383563

/-- The time taken by the relay team to finish the race -/
def relay_race_time (mary_time susan_time jen_time tiffany_time : ℕ) : ℕ :=
  mary_time + susan_time + jen_time + tiffany_time

/-- Theorem stating the total time for the relay race -/
theorem relay_race_total_time : ∃ (mary_time susan_time jen_time tiffany_time : ℕ),
  mary_time = 2 * susan_time ∧
  susan_time = jen_time + 10 ∧
  jen_time = 30 ∧
  tiffany_time = mary_time - 7 ∧
  relay_race_time mary_time susan_time jen_time tiffany_time = 223 := by
  sorry


end NUMINAMATH_CALUDE_relay_race_total_time_l3835_383563


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3835_383521

theorem complex_equation_sum (a b : ℝ) :
  (a + 4 * Complex.I) * Complex.I = b + Complex.I →
  a + b = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3835_383521


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3835_383586

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (15 * Complex.I) / (3 + 4 * Complex.I)
  Complex.im z = 9/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3835_383586


namespace NUMINAMATH_CALUDE_room_length_calculation_l3835_383516

theorem room_length_calculation (breadth height pole_length : ℝ) 
  (h1 : breadth = 8)
  (h2 : height = 9)
  (h3 : pole_length = 17) : 
  ∃ length : ℝ, length^2 + breadth^2 + height^2 = pole_length^2 ∧ length = 12 := by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l3835_383516


namespace NUMINAMATH_CALUDE_function_properties_l3835_383539

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x + (1/2) * m * x^2 - 2

theorem function_properties (m : ℝ) :
  (((Real.exp 1)⁻¹ + m = -(1/2)) → m = -(3/2)) ∧
  (∀ x > 0, f m x + 2 ≤ m * x^2 + (m - 1) * x - 1) →
  m ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_function_properties_l3835_383539


namespace NUMINAMATH_CALUDE_ajays_income_l3835_383538

/-- Ajay's monthly income in Rupees -/
def monthly_income : ℝ := 90000

/-- Percentage of income spent on household items -/
def household_percentage : ℝ := 0.50

/-- Percentage of income spent on clothes -/
def clothes_percentage : ℝ := 0.25

/-- Percentage of income spent on medicines -/
def medicines_percentage : ℝ := 0.15

/-- Amount saved in Rupees -/
def savings : ℝ := 9000

theorem ajays_income :
  monthly_income * household_percentage +
  monthly_income * clothes_percentage +
  monthly_income * medicines_percentage +
  savings = monthly_income :=
by sorry

end NUMINAMATH_CALUDE_ajays_income_l3835_383538


namespace NUMINAMATH_CALUDE_field_area_calculation_l3835_383513

theorem field_area_calculation (smaller_area larger_area : ℝ) : 
  smaller_area = 315 →
  larger_area - smaller_area = (1 / 5) * ((smaller_area + larger_area) / 2) →
  smaller_area + larger_area = 700 := by
  sorry

end NUMINAMATH_CALUDE_field_area_calculation_l3835_383513


namespace NUMINAMATH_CALUDE_salary_change_percentage_l3835_383559

theorem salary_change_percentage (S : ℝ) (x : ℝ) (h : S > 0) :
  S * (1 - x / 100) * (1 + x / 100) = 0.75 * S → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l3835_383559


namespace NUMINAMATH_CALUDE_square_side_length_l3835_383569

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 361 → side * side = area → side = 19 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3835_383569


namespace NUMINAMATH_CALUDE_cubic_trinomial_degree_l3835_383528

theorem cubic_trinomial_degree (n : ℕ) : 
  (∃ (p : Polynomial ℝ), p = X^n - 5*X + 4 ∧ Polynomial.degree p = 3) → n = 3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_trinomial_degree_l3835_383528


namespace NUMINAMATH_CALUDE_years_of_writing_comics_l3835_383545

/-- Represents the number of comics written in a year -/
def comics_per_year : ℕ := 182

/-- Represents the total number of comics written -/
def total_comics : ℕ := 730

/-- Theorem: Given the conditions, the number of years of writing comics is 4 -/
theorem years_of_writing_comics : 
  (total_comics / comics_per_year : ℕ) = 4 := by sorry

end NUMINAMATH_CALUDE_years_of_writing_comics_l3835_383545


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3835_383551

theorem quadratic_inequality (x : ℝ) : x^2 - 6*x + 5 > 0 ↔ x < 1 ∨ x > 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3835_383551


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_pi_12_l3835_383501

theorem cos_2alpha_plus_pi_12 (α : Real) (h1 : 0 < α) (h2 : α < π/2) 
  (h3 : Real.sin (α - π/8) = Real.sqrt 3 / 3) : 
  Real.cos (2*α + π/12) = (1 - 2*Real.sqrt 6) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_pi_12_l3835_383501


namespace NUMINAMATH_CALUDE_jack_sent_three_bestbuy_cards_l3835_383590

def total_requested : ℕ := 6 * 500 + 9 * 200

def walmart_sent : ℕ := 2

def walmart_value : ℕ := 200

def bestbuy_value : ℕ := 500

def remaining_value : ℕ := 3900

def bestbuy_sent : ℕ := 3

theorem jack_sent_three_bestbuy_cards :
  total_requested - remaining_value = walmart_sent * walmart_value + bestbuy_sent * bestbuy_value :=
by sorry

end NUMINAMATH_CALUDE_jack_sent_three_bestbuy_cards_l3835_383590


namespace NUMINAMATH_CALUDE_g_composition_of_three_l3835_383581

def g (n : ℤ) : ℤ :=
  if n < 5 then 2 * n^2 + 3 else 4 * n + 1

theorem g_composition_of_three : g (g (g 3)) = 341 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l3835_383581


namespace NUMINAMATH_CALUDE_probability_marked_vertex_half_l3835_383522

/-- Represents a shape with triangles and a marked vertex -/
structure TriangleShape where
  totalTriangles : ℕ
  trianglesWithMarkedVertex : ℕ
  hasProp : trianglesWithMarkedVertex ≤ totalTriangles

/-- The probability of selecting a triangle with the marked vertex -/
def probabilityMarkedVertex (shape : TriangleShape) : ℚ :=
  shape.trianglesWithMarkedVertex / shape.totalTriangles

theorem probability_marked_vertex_half (shape : TriangleShape) 
  (h1 : shape.totalTriangles = 6)
  (h2 : shape.trianglesWithMarkedVertex = 3) :
  probabilityMarkedVertex shape = 1/2 := by
  sorry

#check probability_marked_vertex_half

end NUMINAMATH_CALUDE_probability_marked_vertex_half_l3835_383522


namespace NUMINAMATH_CALUDE_weaving_increase_l3835_383526

/-- Represents the daily increase in weaving -/
def daily_increase : ℚ := 16 / 29

/-- Represents the number of days in a month -/
def days_in_month : ℕ := 30

/-- Represents the amount woven on the first day -/
def first_day_weaving : ℚ := 10

/-- Represents the total amount woven in a month -/
def total_weaving : ℚ := 6

theorem weaving_increase :
  first_day_weaving + (days_in_month - 1) * daily_increase / 2 * days_in_month = total_weaving :=
sorry

end NUMINAMATH_CALUDE_weaving_increase_l3835_383526


namespace NUMINAMATH_CALUDE_sam_initial_yellow_marbles_l3835_383585

/-- The number of yellow marbles Sam had initially -/
def initial_yellow_marbles : ℕ := sorry

/-- The number of yellow marbles Joan took -/
def marbles_taken : ℕ := 25

/-- The number of yellow marbles Sam has now -/
def current_yellow_marbles : ℕ := 61

theorem sam_initial_yellow_marbles :
  initial_yellow_marbles = current_yellow_marbles + marbles_taken :=
by sorry

end NUMINAMATH_CALUDE_sam_initial_yellow_marbles_l3835_383585


namespace NUMINAMATH_CALUDE_signboard_white_area_l3835_383541

/-- Represents the dimensions of a letter stroke -/
structure StrokeDimensions where
  width : ℝ
  height : ℝ

/-- Represents a letter on the signboard -/
inductive Letter
| L
| A
| S
| T

/-- Calculates the area of a letter based on its strokes -/
def letterArea (letter : Letter) : ℝ :=
  match letter with
  | Letter.L => 9
  | Letter.A => 7.5
  | Letter.S => 13
  | Letter.T => 9

/-- Represents the signboard -/
structure Signboard where
  width : ℝ
  height : ℝ
  word : List Letter
  strokeWidth : ℝ

def signboard : Signboard :=
  { width := 6
  , height := 18
  , word := [Letter.L, Letter.A, Letter.S, Letter.T]
  , strokeWidth := 1 }

/-- Calculates the total area of the signboard -/
def totalArea (s : Signboard) : ℝ :=
  s.width * s.height

/-- Calculates the area covered by the letters -/
def coveredArea (s : Signboard) : ℝ :=
  s.word.map letterArea |> List.sum

/-- Calculates the white area remaining on the signboard -/
def whiteArea (s : Signboard) : ℝ :=
  totalArea s - coveredArea s

/-- Theorem stating that the white area of the given signboard is 69.5 square units -/
theorem signboard_white_area :
  whiteArea signboard = 69.5 := by
  sorry

end NUMINAMATH_CALUDE_signboard_white_area_l3835_383541


namespace NUMINAMATH_CALUDE_solution_set_inequality_inequality_for_positive_mn_l3835_383566

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x + 1|

-- Theorem for part 1
theorem solution_set_inequality (x : ℝ) :
  f x ≤ 10 - |x - 3| ↔ x ∈ Set.Icc (-8/3) 4 := by sorry

-- Theorem for part 2
theorem inequality_for_positive_mn (m n : ℝ) 
  (hm : m > 0) (hn : n > 0) (h_mn : m + 2 * n = m * n) :
  f m + f (-2 * n) ≥ 16 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_inequality_for_positive_mn_l3835_383566


namespace NUMINAMATH_CALUDE_sequence_increasing_l3835_383547

def a (n : ℕ) : ℚ := (n - 1) / (n + 1)

theorem sequence_increasing : ∀ n : ℕ, n ≥ 1 → a n < a (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_increasing_l3835_383547


namespace NUMINAMATH_CALUDE_product_of_roots_l3835_383524

theorem product_of_roots : (32 : ℝ) ^ (1/5 : ℝ) * (128 : ℝ) ^ (1/7 : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3835_383524


namespace NUMINAMATH_CALUDE_inequalities_proof_l3835_383582

theorem inequalities_proof (a b : ℝ) (h : a + b > 0) :
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a+2)*(b+2) > a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l3835_383582


namespace NUMINAMATH_CALUDE_systematic_sampling_seventh_group_l3835_383591

/-- Represents the systematic sampling method for a population --/
structure SystematicSampling where
  populationSize : Nat
  numGroups : Nat
  sampleSize : Nat
  firstDrawn : Nat

/-- Calculates the number drawn for a given group --/
def SystematicSampling.numberDrawn (s : SystematicSampling) (group : Nat) : Nat :=
  let offset := (s.firstDrawn + 33 * group) % 100
  let baseNumber := (group - 1) * (s.populationSize / s.numGroups)
  baseNumber + offset

/-- The main theorem to prove --/
theorem systematic_sampling_seventh_group 
  (s : SystematicSampling)
  (h1 : s.populationSize = 1000)
  (h2 : s.numGroups = 10)
  (h3 : s.sampleSize = 10)
  (h4 : s.firstDrawn = 57) :
  s.numberDrawn 7 = 688 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_seventh_group_l3835_383591


namespace NUMINAMATH_CALUDE_tan_equality_proof_l3835_383503

theorem tan_equality_proof (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (225 * π / 180) → n = 45 := by
sorry

end NUMINAMATH_CALUDE_tan_equality_proof_l3835_383503


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3835_383557

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 5/8
  let b : ℚ := 7/8
  let c : ℚ := 3/4
  c = (a + b) / 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3835_383557


namespace NUMINAMATH_CALUDE_solve_for_m_l3835_383597

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + m
def g (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + 9*m

-- State the theorem
theorem solve_for_m : ∃ m : ℝ, f m 2 = 2 * g m 2 ∧ m = 0 := by sorry

end NUMINAMATH_CALUDE_solve_for_m_l3835_383597


namespace NUMINAMATH_CALUDE_parallel_line_slope_l3835_383595

/-- Given a line with equation 5x - 3y = 21, prove that the slope of any parallel line is 5/3 -/
theorem parallel_line_slope (x y : ℝ) :
  (5 * x - 3 * y = 21) → 
  (∃ (m : ℝ), m = 5 / 3 ∧ ∀ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ → 
    (5 * x₁ - 3 * y₁ = 21 ∧ 5 * x₂ - 3 * y₂ = 21) → 
    (y₂ - y₁) / (x₂ - x₁) = m) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l3835_383595


namespace NUMINAMATH_CALUDE_factor_expression_l3835_383555

theorem factor_expression (y : ℝ) : 64 - 16 * y^2 = 16 * (2 - y) * (2 + y) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3835_383555


namespace NUMINAMATH_CALUDE_tysons_ocean_speed_l3835_383505

/-- Tyson's swimming speed problem -/
theorem tysons_ocean_speed (lake_speed : ℝ) (total_races : ℕ) (race_distance : ℝ) (total_time : ℝ) :
  lake_speed = 3 →
  total_races = 10 →
  race_distance = 3 →
  total_time = 11 →
  ∃ (ocean_speed : ℝ),
    ocean_speed = 2.5 ∧
    (lake_speed * (total_races / 2 * race_distance) + ocean_speed * (total_races / 2 * race_distance)) / total_races = race_distance / (total_time / total_races) :=
by sorry

end NUMINAMATH_CALUDE_tysons_ocean_speed_l3835_383505


namespace NUMINAMATH_CALUDE_tree_growth_rate_l3835_383549

/-- Proves that the annual increase in tree height is 1 foot -/
theorem tree_growth_rate (h : ℝ) : 
  (4 : ℝ) + 6 * h = ((4 : ℝ) + 4 * h) * (5/4) → h = 1 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_rate_l3835_383549


namespace NUMINAMATH_CALUDE_min_value_trig_function_min_value_trig_function_achievable_l3835_383548

theorem min_value_trig_function (α : Real) (h : α ∈ Set.Ioo 0 (π / 2)) :
  1 / (Real.sin α)^2 + 3 / (Real.cos α)^2 ≥ 4 + 2 * Real.sqrt 3 := by
  sorry

theorem min_value_trig_function_achievable :
  ∃ α : Real, α ∈ Set.Ioo 0 (π / 2) ∧
  1 / (Real.sin α)^2 + 3 / (Real.cos α)^2 = 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_function_min_value_trig_function_achievable_l3835_383548


namespace NUMINAMATH_CALUDE_dodecagon_enclosure_l3835_383504

theorem dodecagon_enclosure (m : ℕ) (n : ℕ) : 
  m = 12 →
  (360 : ℝ) / n = (180 : ℝ) - (m - 2 : ℝ) * 180 / m →
  n = 6 :=
sorry

end NUMINAMATH_CALUDE_dodecagon_enclosure_l3835_383504


namespace NUMINAMATH_CALUDE_general_term_of_sequence_l3835_383533

/-- Given a sequence {a_n} where the sum of the first n terms is S_n = n^2 + 3n,
    prove that the general term a_n = 2n + 2 for all positive integers n. -/
theorem general_term_of_sequence (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
    (h : ∀ n : ℕ+, S n = n^2 + 3*n) : 
    ∀ n : ℕ+, a n = 2*n + 2 := by
  sorry

end NUMINAMATH_CALUDE_general_term_of_sequence_l3835_383533


namespace NUMINAMATH_CALUDE_diagonal_length_of_courtyard_l3835_383589

/-- Represents a rectangular courtyard with sides in ratio 4:3 -/
structure Courtyard where
  length : ℝ
  width : ℝ
  ratio_constraint : length = (4/3) * width

/-- The cost of paving in Rupees per square meter -/
def paving_cost_per_sqm : ℝ := 0.5

/-- The total cost of paving the courtyard in Rupees -/
def total_paving_cost : ℝ := 600

theorem diagonal_length_of_courtyard (c : Courtyard) : 
  c.length * c.width * paving_cost_per_sqm = total_paving_cost →
  Real.sqrt (c.length^2 + c.width^2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_of_courtyard_l3835_383589


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3835_383575

theorem quadratic_root_problem (v : ℚ) : 
  (3 * ((-12 - Real.sqrt 400) / 15)^2 + 12 * ((-12 - Real.sqrt 400) / 15) + v = 0) → 
  v = 704/75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3835_383575


namespace NUMINAMATH_CALUDE_middle_number_theorem_l3835_383554

theorem middle_number_theorem (x y z : ℝ) 
  (h1 : x + y = 15) 
  (h2 : x + z = 18) 
  (h3 : y + z = 21) 
  (h4 : x < y ∧ y < z) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_theorem_l3835_383554


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l3835_383572

/-- Simple interest calculation -/
def simpleInterest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem loan_principal_calculation (rate time interest : ℚ) 
  (h_rate : rate = 12)
  (h_time : time = 3)
  (h_interest : interest = 7200) :
  ∃ principal : ℚ, simpleInterest principal rate time = interest ∧ principal = 20000 := by
  sorry

#check loan_principal_calculation

end NUMINAMATH_CALUDE_loan_principal_calculation_l3835_383572
