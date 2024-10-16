import Mathlib

namespace NUMINAMATH_CALUDE_base7_246_equals_base10_132_l2887_288732

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (hundreds : ℕ) (tens : ℕ) (ones : ℕ) : ℕ :=
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- Proves that 246 in base 7 is equal to 132 in base 10 -/
theorem base7_246_equals_base10_132 : base7ToBase10 2 4 6 = 132 := by
  sorry

end NUMINAMATH_CALUDE_base7_246_equals_base10_132_l2887_288732


namespace NUMINAMATH_CALUDE_some_number_value_l2887_288786

theorem some_number_value (some_number : ℝ) : 
  (some_number * 10) / 100 = 0.032420000000000004 → 
  some_number = 0.32420000000000004 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l2887_288786


namespace NUMINAMATH_CALUDE_annie_figurines_count_l2887_288756

def number_of_tvs : ℕ := 5
def cost_per_tv : ℕ := 50
def total_spent : ℕ := 260
def cost_per_figurine : ℕ := 1

theorem annie_figurines_count :
  (total_spent - number_of_tvs * cost_per_tv) / cost_per_figurine = 10 := by
  sorry

end NUMINAMATH_CALUDE_annie_figurines_count_l2887_288756


namespace NUMINAMATH_CALUDE_range_of_a_l2887_288790

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | x < 2*a ∨ x > -a}

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

-- State the theorem
theorem range_of_a (a : ℝ) : 
  a < 0 → 
  (∀ x, ¬(p x) → ¬(q a x)) ∧ 
  (∃ x, ¬(p x) ∧ q a x) → 
  a ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2887_288790


namespace NUMINAMATH_CALUDE_pat_calculation_l2887_288700

theorem pat_calculation (x : ℝ) : (x / 6) - 14 = 16 → (x * 6) + 14 > 1000 := by
  sorry

end NUMINAMATH_CALUDE_pat_calculation_l2887_288700


namespace NUMINAMATH_CALUDE_book_box_ratio_l2887_288726

theorem book_box_ratio (total : ℕ) (chris_percent : ℚ) (h1 : chris_percent = 60 / 100) :
  (total - total * chris_percent) / (total * chris_percent) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_book_box_ratio_l2887_288726


namespace NUMINAMATH_CALUDE_simons_flower_purchase_l2887_288765

def flower_purchase (pansy_price petunia_price hydrangea_price : ℝ)
                    (pansy_count petunia_count : ℕ)
                    (discount_rate : ℝ)
                    (change_received : ℝ) : Prop :=
  let total_before_discount := pansy_price * (pansy_count : ℝ) +
                               petunia_price * (petunia_count : ℝ) +
                               hydrangea_price
  let discount := discount_rate * total_before_discount
  let total_after_discount := total_before_discount - discount
  let amount_paid := total_after_discount + change_received
  amount_paid = 50

theorem simons_flower_purchase :
  flower_purchase 2.5 1 12.5 5 5 0.1 23 := by
  sorry

end NUMINAMATH_CALUDE_simons_flower_purchase_l2887_288765


namespace NUMINAMATH_CALUDE_clothing_sale_profit_l2887_288716

def initial_cost : ℕ := 400
def num_sets : ℕ := 8
def sale_price : ℕ := 55
def adjustments : List ℤ := [2, -3, 2, 1, -2, -1, 0, -2]

theorem clothing_sale_profit :
  (num_sets * sale_price : ℤ) + (adjustments.sum) - initial_cost = 37 := by
  sorry

end NUMINAMATH_CALUDE_clothing_sale_profit_l2887_288716


namespace NUMINAMATH_CALUDE_constant_value_l2887_288749

theorem constant_value (t : ℝ) (constant : ℝ) :
  let x := constant - 2 * t
  let y := 2 * t - 2
  (t = 0.75 → x = y) →
  constant = 1 := by sorry

end NUMINAMATH_CALUDE_constant_value_l2887_288749


namespace NUMINAMATH_CALUDE_range_of_m_min_value_sum_squares_equality_condition_l2887_288776

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 4|

-- Theorem 1: Range of m
theorem range_of_m (m : ℝ) :
  (∀ x, f x ≤ -m^2 + 6*m) → 1 ≤ m ∧ m ≤ 5 :=
sorry

-- Theorem 2: Minimum value of a^2 + b^2 + c^2
theorem min_value_sum_squares (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 → 3*a + 4*b + 5*c = 5 →
  a^2 + b^2 + c^2 ≥ 1/2 :=
sorry

-- Theorem 3: Equality condition
theorem equality_condition (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 → 3*a + 4*b + 5*c = 5 →
  a^2 + b^2 + c^2 = 1/2 ↔ a = 3/10 ∧ b = 4/10 ∧ c = 5/10 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_min_value_sum_squares_equality_condition_l2887_288776


namespace NUMINAMATH_CALUDE_tenth_grader_max_points_l2887_288782

/-- Represents the grade of a student --/
inductive Grade
  | tenth
  | eleventh

/-- Represents the result of a chess game --/
inductive GameResult
  | win
  | draw
  | loss

/-- Calculate points for a game result --/
def pointsForResult (result : GameResult) : Real :=
  match result with
  | GameResult.win => 1
  | GameResult.draw => 0.5
  | GameResult.loss => 0

/-- Structure representing a chess tournament --/
structure ChessTournament where
  tenthGraders : Nat
  eleventhGraders : Nat
  tenthGraderPoints : Real
  eleventhGraderPoints : Real

/-- Theorem stating the maximum points a 10th grader can score --/
theorem tenth_grader_max_points (tournament : ChessTournament) 
  (h1 : tournament.eleventhGraders = 10 * tournament.tenthGraders)
  (h2 : tournament.eleventhGraderPoints = 4.5 * tournament.tenthGraderPoints)
  (h3 : tournament.tenthGraders > 0) :
  ∃ (maxPoints : Real), 
    maxPoints = 10 ∧ 
    ∀ (points : Real), 
      (∃ (player : Nat), player ≤ tournament.tenthGraders ∧ points = tournament.tenthGraderPoints / tournament.tenthGraders) →
      points ≤ maxPoints :=
by sorry

end NUMINAMATH_CALUDE_tenth_grader_max_points_l2887_288782


namespace NUMINAMATH_CALUDE_sum_of_roots_l2887_288772

theorem sum_of_roots (a β : ℝ) (ha : a^2 - 2*a = 1) (hβ : β^2 - 2*β - 1 = 0) (hneq : a ≠ β) :
  a + β = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2887_288772


namespace NUMINAMATH_CALUDE_watermelon_seeds_theorem_l2887_288760

/-- Given a number of watermelons and seeds per watermelon, 
    calculate the total number of seeds -/
def total_seeds (watermelons : ℕ) (seeds_per_watermelon : ℕ) : ℕ :=
  watermelons * seeds_per_watermelon

/-- Theorem: If we have 4 watermelons with 100 seeds each, 
    the total number of seeds is 400 -/
theorem watermelon_seeds_theorem :
  total_seeds 4 100 = 400 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_seeds_theorem_l2887_288760


namespace NUMINAMATH_CALUDE_clock_equivalent_hours_l2887_288794

theorem clock_equivalent_hours : ∃ (n : ℕ), n > 3 ∧ 
  (∀ k : ℕ, k > 3 ∧ k < n → ¬(12 ∣ (k^2 - k))) ∧ 
  (12 ∣ (n^2 - n)) := by
  sorry

end NUMINAMATH_CALUDE_clock_equivalent_hours_l2887_288794


namespace NUMINAMATH_CALUDE_unique_assignment_l2887_288724

/-- Represents a valid assignment of digits to letters -/
structure Assignment where
  a : Fin 5
  m : Fin 5
  e : Fin 5
  h : Fin 5
  z : Fin 5
  different : a ≠ m ∧ a ≠ e ∧ a ≠ h ∧ a ≠ z ∧ m ≠ e ∧ m ≠ h ∧ m ≠ z ∧ e ≠ h ∧ e ≠ z ∧ h ≠ z

/-- The inequalities that must be satisfied -/
def satisfies_inequalities (assign : Assignment) : Prop :=
  3 > assign.a.val + 1 ∧
  assign.a.val + 1 > assign.m.val + 1 ∧
  assign.m.val + 1 < assign.e.val + 1 ∧
  assign.e.val + 1 < assign.h.val + 1 ∧
  assign.h.val + 1 < assign.a.val + 1

/-- The theorem stating that the only valid assignment results in ZAMENA = 541234 -/
theorem unique_assignment :
  ∀ (assign : Assignment),
    satisfies_inequalities assign →
    assign.z.val = 4 ∧
    assign.a.val = 3 ∧
    assign.m.val = 0 ∧
    assign.e.val = 1 ∧
    assign.h.val = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_assignment_l2887_288724


namespace NUMINAMATH_CALUDE_fathers_full_time_jobs_l2887_288798

theorem fathers_full_time_jobs (total_parents : ℝ) (h1 : total_parents > 0) : 
  let mothers := 0.4 * total_parents
  let fathers := 0.6 * total_parents
  let mothers_full_time := 0.9 * mothers
  let total_full_time := 0.81 * total_parents
  let fathers_full_time := total_full_time - mothers_full_time
  fathers_full_time / fathers = 3/4 := by sorry

end NUMINAMATH_CALUDE_fathers_full_time_jobs_l2887_288798


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2887_288723

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : a 4 = 8) (h2 : a 5 = 12) (h3 : a 6 = 16) :
  a 1 + a 2 + a 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2887_288723


namespace NUMINAMATH_CALUDE_T_divisibility_l2887_288706

-- Define the set T
def T : Set ℕ := {x | ∃ n : ℕ, x = (2*n - 2)^2 + (2*n)^2 + (2*n + 2)^2}

-- Theorem statement
theorem T_divisibility :
  (∀ x ∈ T, 4 ∣ x) ∧ (∃ x ∈ T, 5 ∣ x) := by
  sorry

end NUMINAMATH_CALUDE_T_divisibility_l2887_288706


namespace NUMINAMATH_CALUDE_f_bounds_and_solution_set_l2887_288743

noncomputable def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

theorem f_bounds_and_solution_set :
  (∀ x : ℝ, -3 ≤ f x ∧ f x ≤ 3) ∧
  {x : ℝ | f x ≥ x^2 - 8*x + 14} = {x : ℝ | 3 ≤ x ∧ x ≤ 4 + Real.sqrt 5} :=
by sorry

end NUMINAMATH_CALUDE_f_bounds_and_solution_set_l2887_288743


namespace NUMINAMATH_CALUDE_apple_cost_calculation_l2887_288711

/-- The cost of three dozen apples in dollars -/
def cost_three_dozen : ℚ := 25.20

/-- The number of dozens we want to calculate the cost for -/
def target_dozens : ℕ := 4

/-- The cost of the target number of dozens of apples -/
def cost_target_dozens : ℚ := 33.60

/-- Theorem stating that the cost of the target number of dozens of apples is correct -/
theorem apple_cost_calculation : 
  (cost_three_dozen / 3) * target_dozens = cost_target_dozens := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_calculation_l2887_288711


namespace NUMINAMATH_CALUDE_cube_sum_l2887_288762

theorem cube_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_l2887_288762


namespace NUMINAMATH_CALUDE_P_not_subset_Q_l2887_288747

-- Define the sets P and Q
def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | |x| > 0}

-- Statement to prove
theorem P_not_subset_Q : ¬(P ⊆ Q) := by
  sorry

end NUMINAMATH_CALUDE_P_not_subset_Q_l2887_288747


namespace NUMINAMATH_CALUDE_expression_simplification_l2887_288735

theorem expression_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -2) (h3 : x ≠ 2) :
  ((x^2 + 4) / x - 4) / ((x^2 - 4) / (x^2 + 2*x)) = x - 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2887_288735


namespace NUMINAMATH_CALUDE_square_tiles_l2887_288712

theorem square_tiles (n : ℕ) (h : n * n = 81) :
  n * n - n = 72 :=
sorry

end NUMINAMATH_CALUDE_square_tiles_l2887_288712


namespace NUMINAMATH_CALUDE_largest_difference_l2887_288740

def A : ℕ := 3 * 2005^2006
def B : ℕ := 2005^2006
def C : ℕ := 2004 * 2005^2005
def D : ℕ := 3 * 2005^2005
def E : ℕ := 2005^2005
def F : ℕ := 2005^2004

theorem largest_difference : A - B > max (B - C) (max (C - D) (max (D - E) (E - F))) := by
  sorry

end NUMINAMATH_CALUDE_largest_difference_l2887_288740


namespace NUMINAMATH_CALUDE_volume_of_specific_pyramid_l2887_288709

/-- A regular triangular pyramid with specific properties -/
structure RegularTriangularPyramid where
  /-- Distance from the midpoint of the height to the lateral face -/
  midpoint_to_face : ℝ
  /-- Distance from the midpoint of the height to the lateral edge -/
  midpoint_to_edge : ℝ

/-- The volume of a regular triangular pyramid -/
noncomputable def volume (p : RegularTriangularPyramid) : ℝ := sorry

/-- Theorem stating the volume of the specific regular triangular pyramid -/
theorem volume_of_specific_pyramid :
  ∀ (p : RegularTriangularPyramid),
    p.midpoint_to_face = 2 →
    p.midpoint_to_edge = Real.sqrt 12 →
    volume p = 216 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_pyramid_l2887_288709


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2887_288710

def M : Set ℝ := {x | x + 2 ≥ 0}
def N : Set ℝ := {x | x - 1 < 0}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -2 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2887_288710


namespace NUMINAMATH_CALUDE_arithmetic_sequence_before_negative_seventeen_l2887_288757

/-- 
Given an arithmetic sequence with first term 88 and common difference -3,
prove that the number of terms that appear before -17 is 35.
-/
theorem arithmetic_sequence_before_negative_seventeen :
  let a : ℕ → ℤ := λ n => 88 - 3 * (n - 1)
  ∃ k : ℕ, a k = -17 ∧ k - 1 = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_before_negative_seventeen_l2887_288757


namespace NUMINAMATH_CALUDE_marcus_savings_l2887_288705

-- Define the given values
def max_budget : ℚ := 250
def shoe_price : ℚ := 120
def shoe_discount : ℚ := 0.3
def shoe_cashback : ℚ := 10
def shoe_tax : ℚ := 0.08
def sock_price : ℚ := 25
def sock_tax : ℚ := 0.06
def shirt_price : ℚ := 55
def shirt_discount : ℚ := 0.1
def shirt_tax : ℚ := 0.07

-- Define the calculation functions
def calculate_shoe_cost : ℚ := 
  (shoe_price * (1 - shoe_discount) - shoe_cashback) * (1 + shoe_tax)

def calculate_sock_cost : ℚ := 
  sock_price * (1 + sock_tax) / 2

def calculate_shirt_cost : ℚ := 
  shirt_price * (1 - shirt_discount) * (1 + shirt_tax)

def total_cost : ℚ := 
  calculate_shoe_cost + calculate_sock_cost + calculate_shirt_cost

-- Theorem statement
theorem marcus_savings : 
  max_budget - total_cost = 103.86 := by sorry

end NUMINAMATH_CALUDE_marcus_savings_l2887_288705


namespace NUMINAMATH_CALUDE_semicircle_chord_product_l2887_288793

/-- The radius of the semicircle -/
def radius : ℝ := 3

/-- The number of equal parts the semicircle is divided into -/
def num_parts : ℕ := 8

/-- The number of chords -/
def num_chords : ℕ := 14

/-- The product of the lengths of the chords in a semicircle -/
def chord_product (r : ℝ) (n : ℕ) : ℝ :=
  (2 * r ^ (n - 1)) * (2 ^ n)

theorem semicircle_chord_product :
  chord_product radius num_chords = 196608 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_chord_product_l2887_288793


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2887_288780

theorem inequality_equivalence (x : ℝ) : 
  |x - 2| + |x + 3| < 7 ↔ -4 < x ∧ x < 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2887_288780


namespace NUMINAMATH_CALUDE_cos_pi_sixth_minus_alpha_l2887_288741

theorem cos_pi_sixth_minus_alpha (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π/6)) 
  (h2 : Real.sin (α + π/3) = 12/13) : Real.cos (π/6 - α) = 12/13 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_sixth_minus_alpha_l2887_288741


namespace NUMINAMATH_CALUDE_vision_data_median_l2887_288729

/-- Represents the vision data for a class of students -/
def VisionData : List (Float × Nat) := [
  (4.0, 1), (4.1, 2), (4.2, 6), (4.3, 3), (4.4, 3),
  (4.5, 4), (4.6, 1), (4.7, 2), (4.8, 5), (4.9, 7), (5.0, 5)
]

/-- The total number of students -/
def totalStudents : Nat := 39

/-- Calculates the median of the vision data -/
def median (data : List (Float × Nat)) (total : Nat) : Float :=
  sorry

/-- Theorem stating that the median of the given vision data is 4.6 -/
theorem vision_data_median : median VisionData totalStudents = 4.6 := by
  sorry

end NUMINAMATH_CALUDE_vision_data_median_l2887_288729


namespace NUMINAMATH_CALUDE_license_plate_count_l2887_288708

/-- The number of consonants in the English alphabet (including Y) -/
def num_consonants : ℕ := 21

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of digits (0 through 9) -/
def num_digits : ℕ := 10

/-- The total number of possible license plates -/
def total_plates : ℕ := num_consonants * num_consonants * num_vowels * num_vowels * num_digits

theorem license_plate_count : total_plates = 110250 := by sorry

end NUMINAMATH_CALUDE_license_plate_count_l2887_288708


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2887_288771

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem sum_of_coefficients :
  (∀ x, f (x + 2) = 2 * x^3 + 5 * x^2 + 3 * x + 6) →
  (∃ a b c d : ℝ, ∀ x, f x = a * x^3 + b * x^2 + c * x + d) →
  (∃ a b c d : ℝ, (∀ x, f x = a * x^3 + b * x^2 + c * x + d) ∧ a + b + c + d = 6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2887_288771


namespace NUMINAMATH_CALUDE_greatest_distance_C_D_l2887_288704

def C : Set ℂ := {z : ℂ | z^3 = 1}

def D : Set ℂ := {z : ℂ | z^3 - 27*z^2 + 27*z - 1 = 0}

theorem greatest_distance_C_D : 
  ∃ (c : ℂ) (d : ℂ), c ∈ C ∧ d ∈ D ∧ 
    ∀ (c' : ℂ) (d' : ℂ), c' ∈ C → d' ∈ D → 
      Complex.abs (c - d) ≥ Complex.abs (c' - d') ∧
      Complex.abs (c - d) = Real.sqrt (184.5 + 60 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_greatest_distance_C_D_l2887_288704


namespace NUMINAMATH_CALUDE_power_four_difference_divisibility_l2887_288725

theorem power_four_difference_divisibility (m n k : ℕ) :
  (4^m - 4^n) % 3^(k+1) = 0 ↔ (m - n) % 3^k = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_four_difference_divisibility_l2887_288725


namespace NUMINAMATH_CALUDE_cut_to_square_l2887_288796

/-- Represents a shape on a checkered paper --/
structure Shape :=
  (area : ℕ)
  (has_hole : Bool)

/-- Represents a square --/
def is_square (s : Shape) : Prop :=
  ∃ (side : ℕ), s.area = side * side ∧ s.has_hole = false

/-- Represents the ability to cut a shape into two parts --/
def can_cut (s : Shape) : Prop :=
  ∃ (part1 part2 : Shape), part1.area + part2.area = s.area

/-- Represents the ability to form a square from two parts --/
def can_form_square (part1 part2 : Shape) : Prop :=
  is_square (Shape.mk (part1.area + part2.area) false)

/-- The main theorem: given a shape with a hole, it can be cut into two parts
    that can form a square --/
theorem cut_to_square (s : Shape) (h : s.has_hole = true) :
  ∃ (part1 part2 : Shape),
    can_cut s ∧
    can_form_square part1 part2 :=
sorry

end NUMINAMATH_CALUDE_cut_to_square_l2887_288796


namespace NUMINAMATH_CALUDE_cloth_sale_calculation_l2887_288758

/-- Proves that the number of metres of cloth sold is 500 given the conditions -/
theorem cloth_sale_calculation (total_selling_price : ℕ) (loss_per_metre : ℕ) (cost_price_per_metre : ℕ)
  (h1 : total_selling_price = 18000)
  (h2 : loss_per_metre = 5)
  (h3 : cost_price_per_metre = 41) :
  total_selling_price / (cost_price_per_metre - loss_per_metre) = 500 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_calculation_l2887_288758


namespace NUMINAMATH_CALUDE_equation_solution_l2887_288728

theorem equation_solution : 
  let f (x : ℝ) := (x^2 - 3*x + 2) * (x^2 + 3*x - 2)
  let g (x : ℝ) := x^2 * (x + 3) * (x - 3)
  f (1/3) = g (1/3) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2887_288728


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2887_288797

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x = 1 → x^3 = x) ∧ ¬(x^3 = x → x = 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2887_288797


namespace NUMINAMATH_CALUDE_A_profit_share_l2887_288767

def investment_A : ℕ := 6300
def investment_B : ℕ := 4200
def investment_C : ℕ := 10500

def profit_share_A : ℚ := 45 / 100
def profit_share_B : ℚ := 30 / 100
def profit_share_C : ℚ := 25 / 100

def total_profit : ℕ := 12200

theorem A_profit_share :
  (profit_share_A * total_profit : ℚ) = 5490 := by sorry

end NUMINAMATH_CALUDE_A_profit_share_l2887_288767


namespace NUMINAMATH_CALUDE_maries_school_students_maries_school_students_proof_l2887_288703

theorem maries_school_students : ℕ → ℕ → Prop :=
  fun m c =>
    m = 4 * c ∧ m + c = 2500 → m = 2000

-- The proof is omitted
theorem maries_school_students_proof : maries_school_students 2000 500 := by
  sorry

end NUMINAMATH_CALUDE_maries_school_students_maries_school_students_proof_l2887_288703


namespace NUMINAMATH_CALUDE_photos_per_album_l2887_288789

theorem photos_per_album (total_photos : ℕ) (num_albums : ℕ) (h1 : total_photos = 180) (h2 : num_albums = 9) :
  total_photos / num_albums = 20 := by
sorry

end NUMINAMATH_CALUDE_photos_per_album_l2887_288789


namespace NUMINAMATH_CALUDE_hypotenuse_ratio_l2887_288746

/-- Represents a right-angled triangle with a 30° angle -/
structure Triangle30 where
  hypotenuse : ℝ
  shared_side : ℝ
  hypotenuse_gt_shared : hypotenuse > shared_side

/-- The three triangles in our problem -/
def three_triangles (a b c : Triangle30) : Prop :=
  a.shared_side = b.shared_side ∧ 
  b.shared_side = c.shared_side ∧ 
  a.hypotenuse ≠ b.hypotenuse ∧ 
  b.hypotenuse ≠ c.hypotenuse ∧ 
  a.hypotenuse ≠ c.hypotenuse

theorem hypotenuse_ratio (a b c : Triangle30) :
  three_triangles a b c →
  (∃ (k : ℝ), k > 0 ∧ 
    (max a.hypotenuse (max b.hypotenuse c.hypotenuse) = 2 * k) ∧
    (max (min a.hypotenuse b.hypotenuse) c.hypotenuse = 2 * k / Real.sqrt 3) ∧
    (min a.hypotenuse (min b.hypotenuse c.hypotenuse) = k)) :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_ratio_l2887_288746


namespace NUMINAMATH_CALUDE_leah_chocolates_l2887_288744

theorem leah_chocolates (leah_chocolates max_chocolates : ℕ) : 
  leah_chocolates = max_chocolates + 8 →
  max_chocolates = leah_chocolates / 3 →
  leah_chocolates = 12 := by
sorry

end NUMINAMATH_CALUDE_leah_chocolates_l2887_288744


namespace NUMINAMATH_CALUDE_factorization_3m_squared_minus_12_l2887_288763

theorem factorization_3m_squared_minus_12 (m : ℝ) : 3 * m^2 - 12 = 3 * (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3m_squared_minus_12_l2887_288763


namespace NUMINAMATH_CALUDE_percentage_to_pass_l2887_288721

/-- The percentage needed to pass an exam, given the achieved score, shortfall, and maximum possible marks. -/
theorem percentage_to_pass 
  (achieved_score : ℕ) 
  (shortfall : ℕ) 
  (max_marks : ℕ) 
  (h1 : achieved_score = 212)
  (h2 : shortfall = 28)
  (h3 : max_marks = 800) : 
  (achieved_score + shortfall) / max_marks * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_to_pass_l2887_288721


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l2887_288759

theorem unique_solution_quadratic_inequality (a : ℝ) : 
  (∃! x : ℝ, |x^2 + a*x + 4*a| ≤ 3) ↔ (a = 8 + 2*Real.sqrt 13 ∨ a = 8 - 2*Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l2887_288759


namespace NUMINAMATH_CALUDE_power_of_power_of_three_l2887_288717

theorem power_of_power_of_three : (3^3)^(3^3) = 7625597484987 := by sorry

end NUMINAMATH_CALUDE_power_of_power_of_three_l2887_288717


namespace NUMINAMATH_CALUDE_lcm_gcd_ratio_540_360_l2887_288755

theorem lcm_gcd_ratio_540_360 : Nat.lcm 540 360 / Nat.gcd 540 360 = 6 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_ratio_540_360_l2887_288755


namespace NUMINAMATH_CALUDE_fraction_simplification_l2887_288736

theorem fraction_simplification (x : ℝ) (hx : x ≠ 1 ∧ x ≠ -1) :
  (2 / (x + 1)) / ((2 / (x^2 - 1)) + (1 / (x + 1))) = (2*x - 2) / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2887_288736


namespace NUMINAMATH_CALUDE_tangent_circles_sum_l2887_288730

-- Define the circles w1 and w2
def w1 (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 20*y - 75 = 0
def w2 (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 20*y + 115 = 0

-- Define the condition for a circle to be externally tangent to w1
def externally_tangent_w1 (cx cy r : ℝ) : Prop :=
  (cx + 4)^2 + (cy - 10)^2 = (r + 11)^2

-- Define the condition for a circle to be internally tangent to w2
def internally_tangent_w2 (cx cy r : ℝ) : Prop :=
  (cx - 6)^2 + (cy - 10)^2 = (7 - r)^2

-- Define the theorem
theorem tangent_circles_sum (p q : ℕ) (h_coprime : Nat.Coprime p q) :
  (∃ (m : ℝ), m > 0 ∧ m^2 = p / q ∧
    (∃ (cx cy r : ℝ), cy = m * cx ∧
      externally_tangent_w1 cx cy r ∧
      internally_tangent_w2 cx cy r) ∧
    (∀ (a : ℝ), a > 0 → a < m →
      ¬∃ (cx cy r : ℝ), cy = a * cx ∧
        externally_tangent_w1 cx cy r ∧
        internally_tangent_w2 cx cy r)) →
  p + q = 181 := by sorry

end NUMINAMATH_CALUDE_tangent_circles_sum_l2887_288730


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2887_288727

def i : ℂ := Complex.I

theorem complex_fraction_equality : (2 * i) / (1 + i) = 1 + i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2887_288727


namespace NUMINAMATH_CALUDE_pat_to_mark_ratio_l2887_288715

/-- Represents the hours charged by each person --/
structure ProjectHours where
  kate : ℕ
  pat : ℕ
  mark : ℕ

/-- Conditions of the problem --/
def project_conditions (h : ProjectHours) : Prop :=
  h.pat + h.kate + h.mark = 180 ∧
  h.pat = 2 * h.kate ∧
  h.mark = h.kate + 100

/-- Theorem stating the ratio of Pat's hours to Mark's hours --/
theorem pat_to_mark_ratio (h : ProjectHours) :
  project_conditions h → h.pat * 3 = h.mark * 1 := by
  sorry

#check pat_to_mark_ratio

end NUMINAMATH_CALUDE_pat_to_mark_ratio_l2887_288715


namespace NUMINAMATH_CALUDE_alpha_value_l2887_288702

theorem alpha_value (α β : ℂ) 
  (h1 : (α + 2*β).re > 0)
  (h2 : (Complex.I * (α - 3*β)).re > 0)
  (h3 : β = 2 + 3*Complex.I) : 
  α = 6 - 6*Complex.I := by sorry

end NUMINAMATH_CALUDE_alpha_value_l2887_288702


namespace NUMINAMATH_CALUDE_promotion_savings_difference_l2887_288714

/-- Represents a promotion for sweater purchases -/
structure Promotion where
  first_sweater_price : ℝ
  second_sweater_discount : ℝ

/-- Calculates the total cost of two sweaters under a given promotion -/
def total_cost (p : Promotion) (original_price : ℝ) : ℝ :=
  p.first_sweater_price + (original_price - p.second_sweater_discount)

theorem promotion_savings_difference :
  let original_price : ℝ := 50
  let promotion_x : Promotion := { first_sweater_price := original_price, second_sweater_discount := 0.4 * original_price }
  let promotion_y : Promotion := { first_sweater_price := original_price, second_sweater_discount := 15 }
  total_cost promotion_y original_price - total_cost promotion_x original_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_promotion_savings_difference_l2887_288714


namespace NUMINAMATH_CALUDE_goods_train_speed_l2887_288713

/-- The speed of a goods train crossing a platform -/
theorem goods_train_speed (platform_length : ℝ) (crossing_time : ℝ) (train_length : ℝ)
  (h1 : platform_length = 250)
  (h2 : crossing_time = 26)
  (h3 : train_length = 270.0416) :
  ∃ (speed : ℝ), abs (speed - 20) < 0.01 ∧ 
  speed = (platform_length + train_length) / crossing_time :=
sorry

end NUMINAMATH_CALUDE_goods_train_speed_l2887_288713


namespace NUMINAMATH_CALUDE_smallest_positive_linear_combination_l2887_288787

theorem smallest_positive_linear_combination : 
  ∃ (k : ℕ), k > 0 ∧ (∃ (m n : ℤ), k = 1205 * m + 27090 * n) ∧ 
  (∀ (j : ℕ), j > 0 → (∃ (x y : ℤ), j = 1205 * x + 27090 * y) → j ≥ k) ∧
  k = 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_linear_combination_l2887_288787


namespace NUMINAMATH_CALUDE_a_range_theorem_l2887_288739

/-- Proposition p: (a-2)x^2 + 2(a-2)x - 4 < 0 for all x ∈ ℝ -/
def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

/-- Proposition q: One root of x^2 + (a-1)x + 1 = 0 is in (0,1), and the other is in (1,2) -/
def prop_q (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + (a - 1) * x + 1 = 0 ∧ y^2 + (a - 1) * y + 1 = 0 ∧
    0 < x ∧ x < 1 ∧ 1 < y ∧ y < 2

/-- The range of values for a -/
def a_range (a : ℝ) : Prop :=
  (a > -2 ∧ a ≤ -3/2) ∨ (a ≥ -1 ∧ a ≤ 2)

theorem a_range_theorem (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬(prop_p a ∧ prop_q a) → a_range a := by
  sorry

end NUMINAMATH_CALUDE_a_range_theorem_l2887_288739


namespace NUMINAMATH_CALUDE_arg_ratio_of_unit_complex_l2887_288773

theorem arg_ratio_of_unit_complex (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 1)
  (h₂ : Complex.abs z₂ = 1)
  (h₃ : z₂ - z₁ = -1) :
  Complex.arg (z₁ / z₂) = π / 3 ∨ Complex.arg (z₁ / z₂) = 5 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arg_ratio_of_unit_complex_l2887_288773


namespace NUMINAMATH_CALUDE_exists_divisible_by_1988_l2887_288783

def f (x : ℤ) : ℤ := 3 * x + 2

def f_iter (k : ℕ) : ℤ → ℤ :=
  match k with
  | 0 => id
  | n + 1 => f ∘ (f_iter n)

theorem exists_divisible_by_1988 :
  ∃ m : ℕ+, (1988 : ℤ) ∣ (f_iter 100 m.val) := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_1988_l2887_288783


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l2887_288770

def sequence_sum (n : ℕ) : ℤ := n^2 - 1

def sequence_term (n : ℕ) : ℤ :=
  if n = 1 then 0
  else 2 * n - 2

theorem sum_of_specific_terms : 
  sequence_term 1 + sequence_term 3 + sequence_term 5 + sequence_term 7 + sequence_term 9 = 44 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l2887_288770


namespace NUMINAMATH_CALUDE_alternating_work_completion_work_fully_completed_l2887_288745

/-- Represents the number of days it takes to complete the work when A and B work on alternate days, starting with B. -/
def alternating_work_days (a_days b_days : ℕ) : ℕ :=
  2 * (9 * b_days * a_days) / (b_days + 3 * a_days)

/-- Theorem stating that if A can complete the work in 12 days and B in 36 days,
    working on alternate days starting with B will complete the work in 18 days. -/
theorem alternating_work_completion :
  alternating_work_days 12 36 = 18 := by
  sorry

/-- Proof that the work is fully completed after 18 days. -/
theorem work_fully_completed (a_days b_days : ℕ) 
  (ha : a_days = 12) (hb : b_days = 36) :
  (9 : ℚ) * (1 / b_days + 1 / a_days) = 1 := by
  sorry

end NUMINAMATH_CALUDE_alternating_work_completion_work_fully_completed_l2887_288745


namespace NUMINAMATH_CALUDE_odd_coefficients_equals_two_pow_binary_ones_l2887_288781

/-- The number of 1s in the binary representation of a natural number -/
def binaryOnes (n : ℕ) : ℕ := sorry

/-- The number of odd coefficients in the polynomial expansion of (1+x)^n -/
def oddCoefficients (n : ℕ) : ℕ := sorry

/-- Theorem: The number of odd coefficients in (1+x)^n is 2^d, where d is the number of 1s in n's binary representation -/
theorem odd_coefficients_equals_two_pow_binary_ones (n : ℕ) :
  oddCoefficients n = 2^(binaryOnes n) := by sorry

end NUMINAMATH_CALUDE_odd_coefficients_equals_two_pow_binary_ones_l2887_288781


namespace NUMINAMATH_CALUDE_liquid_x_percentage_in_mixed_solution_l2887_288768

/-- The percentage of liquid X in the resulting solution after mixing two solutions. -/
theorem liquid_x_percentage_in_mixed_solution
  (percent_x_in_a : ℝ)
  (percent_x_in_b : ℝ)
  (weight_a : ℝ)
  (weight_b : ℝ)
  (h1 : percent_x_in_a = 0.8)
  (h2 : percent_x_in_b = 1.8)
  (h3 : weight_a = 600)
  (h4 : weight_b = 700) :
  let weight_x_in_a := percent_x_in_a / 100 * weight_a
  let weight_x_in_b := percent_x_in_b / 100 * weight_b
  let total_weight_x := weight_x_in_a + weight_x_in_b
  let total_weight := weight_a + weight_b
  let percent_x_in_mixed := total_weight_x / total_weight * 100
  ∃ ε > 0, |percent_x_in_mixed - 1.34| < ε :=
sorry

end NUMINAMATH_CALUDE_liquid_x_percentage_in_mixed_solution_l2887_288768


namespace NUMINAMATH_CALUDE_total_pumpkin_pies_l2887_288753

theorem total_pumpkin_pies (pinky helen emily jake : ℕ)
  (h1 : pinky = 147)
  (h2 : helen = 56)
  (h3 : emily = 89)
  (h4 : jake = 122) :
  pinky + helen + emily + jake = 414 := by
  sorry

end NUMINAMATH_CALUDE_total_pumpkin_pies_l2887_288753


namespace NUMINAMATH_CALUDE_no_consecutive_even_fibonacci_l2887_288785

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem no_consecutive_even_fibonacci :
  ∀ n : ℕ, ¬(Even (fibonacci n) ∧ Even (fibonacci (n + 1))) := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_even_fibonacci_l2887_288785


namespace NUMINAMATH_CALUDE_percent_calculation_l2887_288764

theorem percent_calculation (x y : ℝ) (h : x = 120.5 ∧ y = 80.75) :
  (x / y) * 100 = 149.26 := by
  sorry

end NUMINAMATH_CALUDE_percent_calculation_l2887_288764


namespace NUMINAMATH_CALUDE_min_sum_first_two_terms_l2887_288775

def is_valid_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → b (n + 2) * (b (n + 1) + 1) = b n + 2210

theorem min_sum_first_two_terms (b : ℕ → ℕ) (h : is_valid_sequence b) :
  ∃ b₁ b₂ : ℕ, b 1 = b₁ ∧ b 2 = b₂ ∧ b₁ + b₂ = 147 ∧
  ∀ b₁' b₂' : ℕ, b 1 = b₁' ∧ b 2 = b₂' → b₁' + b₂' ≥ 147 :=
sorry

end NUMINAMATH_CALUDE_min_sum_first_two_terms_l2887_288775


namespace NUMINAMATH_CALUDE_fraction_calculation_l2887_288707

theorem fraction_calculation : (17/5) + (-23/8) - (-28/5) - (1/8) = 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2887_288707


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2887_288777

/-- If x^3 - 2x^2 + px + q is divisible by x + 2, then q = 16 + 2p -/
theorem polynomial_divisibility (p q : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^3 - 2*x^2 + p*x + q = (x + 2) * k) → 
  q = 16 + 2*p := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2887_288777


namespace NUMINAMATH_CALUDE_phyllis_marble_count_l2887_288754

/-- The number of groups of marbles in Phyllis's collection -/
def num_groups : ℕ := 32

/-- The number of marbles in each group -/
def marbles_per_group : ℕ := 2

/-- The total number of marbles in Phyllis's collection -/
def total_marbles : ℕ := num_groups * marbles_per_group

theorem phyllis_marble_count : total_marbles = 64 := by
  sorry

end NUMINAMATH_CALUDE_phyllis_marble_count_l2887_288754


namespace NUMINAMATH_CALUDE_elena_marco_sum_ratio_l2887_288761

def sum_odd_integers (n : ℕ) : ℕ := n * n

def sum_integers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem elena_marco_sum_ratio :
  (sum_odd_integers 250) / (sum_integers 250) = 2 := by
  sorry

end NUMINAMATH_CALUDE_elena_marco_sum_ratio_l2887_288761


namespace NUMINAMATH_CALUDE_new_weighted_average_age_l2887_288774

/-- The new weighted average age of a class after new students join -/
theorem new_weighted_average_age
  (n₁ : ℕ) (a₁ : ℝ)
  (n₂ : ℕ) (a₂ : ℝ)
  (n₃ : ℕ) (a₃ : ℝ)
  (n₄ : ℕ) (a₄ : ℝ)
  (n₅ : ℕ) (a₅ : ℝ)
  (h₁ : n₁ = 15) (h₂ : a₁ = 42)
  (h₃ : n₂ = 20) (h₄ : a₂ = 35)
  (h₅ : n₃ = 10) (h₆ : a₃ = 50)
  (h₇ : n₄ = 7)  (h₈ : a₄ = 30)
  (h₉ : n₅ = 11) (h₁₀ : a₅ = 45) :
  (n₁ * a₁ + n₂ * a₂ + n₃ * a₃ + n₄ * a₄ + n₅ * a₅) / (n₁ + n₂ + n₃ + n₄ + n₅) = 2535 / 63 := by
  sorry

#eval (2535 : Float) / 63

end NUMINAMATH_CALUDE_new_weighted_average_age_l2887_288774


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l2887_288719

def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 - 2 * x + 1

def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation m x = 0

theorem quadratic_real_roots_range (m : ℝ) :
  has_real_roots m ↔ m ≤ 2 ∧ m ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l2887_288719


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2887_288769

theorem simplify_trig_expression (α : ℝ) :
  (1 - Real.cos (2 * α) + Real.sin (2 * α)) / (1 + Real.cos (2 * α) + Real.sin (2 * α)) = Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2887_288769


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2887_288701

/-- The equation of a hyperbola with given properties -/
theorem hyperbola_equation (m n : ℝ) (h : m < 0) :
  (∀ x y : ℝ, x^2 / m + y^2 / n = 1) →  -- Given hyperbola equation
  (n = 1) →                            -- Derived from eccentricity = 2 and a = 1
  (m = -3) →                           -- Derived from b^2 = 3
  (∀ x y : ℝ, y^2 - x^2 / 3 = 1) :=    -- Equation to prove
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2887_288701


namespace NUMINAMATH_CALUDE_quadratic_necessary_not_sufficient_l2887_288792

theorem quadratic_necessary_not_sufficient :
  (∀ x : ℝ, (|x - 2| < 1) → (x^2 - 5*x + 4 < 0)) ∧
  (∃ x : ℝ, (x^2 - 5*x + 4 < 0) ∧ ¬(|x - 2| < 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_necessary_not_sufficient_l2887_288792


namespace NUMINAMATH_CALUDE_propositions_truth_l2887_288752

theorem propositions_truth :
  (∀ a b : ℝ, a > b ∧ 1/a > 1/b → a*b < 0) ∧
  (∃ a b : ℝ, a < b ∧ b < 0 ∧ ¬(a^2 < a*b ∧ a*b < b^2)) ∧
  (∃ c a b : ℝ, c > a ∧ a > b ∧ b > 0 ∧ ¬(a/(c-a) < b/(c-b))) ∧
  (∀ a b c : ℝ, a > b ∧ b > c ∧ c > 0 → a/b > (a+c)/(b+c)) :=
by
  sorry

end NUMINAMATH_CALUDE_propositions_truth_l2887_288752


namespace NUMINAMATH_CALUDE_equation_solution_l2887_288778

theorem equation_solution : ∃ x : ℝ, 
  (0.5^3 - x^3 / 0.5^2 + 0.05 + 0.1^2 = 0.4) ∧ 
  (abs (x + 0.378) < 0.001) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2887_288778


namespace NUMINAMATH_CALUDE_partition_M_theorem_l2887_288748

/-- The set M containing elements from 1 to 12 -/
def M : Finset ℕ := Finset.range 12

/-- Predicate to check if a set is a valid partition of M -/
def is_valid_partition (A B C : Finset ℕ) : Prop :=
  A ∪ B ∪ C = M ∧ A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧
  A.card = 4 ∧ B.card = 4 ∧ C.card = 4

/-- Predicate to check if C satisfies the ordering condition -/
def C_ordered (C : Finset ℕ) : Prop :=
  ∃ c₁ c₂ c₃ c₄, C = {c₁, c₂, c₃, c₄} ∧ c₁ < c₂ ∧ c₂ < c₃ ∧ c₃ < c₄

/-- Predicate to check if A, B, and C satisfy the sum condition -/
def sum_condition (A B C : Finset ℕ) : Prop :=
  ∃ a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ c₁ c₂ c₃ c₄,
    A = {a₁, a₂, a₃, a₄} ∧ B = {b₁, b₂, b₃, b₄} ∧ C = {c₁, c₂, c₃, c₄} ∧
    a₁ + b₁ = c₁ ∧ a₂ + b₂ = c₂ ∧ a₃ + b₃ = c₃ ∧ a₄ + b₄ = c₄

theorem partition_M_theorem :
  ∀ A B C : Finset ℕ,
    is_valid_partition A B C →
    C_ordered C →
    sum_condition A B C →
    C = {8, 9, 10, 12} ∨ C = {7, 9, 11, 12} ∨ C = {6, 10, 11, 12} :=
sorry

end NUMINAMATH_CALUDE_partition_M_theorem_l2887_288748


namespace NUMINAMATH_CALUDE_molecular_weight_ccl4_l2887_288795

/-- The atomic weight of carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of chlorine in g/mol -/
def chlorine_weight : ℝ := 35.45

/-- The number of carbon atoms in a molecule of carbon tetrachloride -/
def carbon_atoms : ℕ := 1

/-- The number of chlorine atoms in a molecule of carbon tetrachloride -/
def chlorine_atoms : ℕ := 4

/-- The number of moles of carbon tetrachloride -/
def moles : ℕ := 9

/-- The molecular weight of carbon tetrachloride in g/mol -/
def ccl4_weight : ℝ := carbon_weight * carbon_atoms + chlorine_weight * chlorine_atoms

/-- Theorem stating the molecular weight of 9 moles of carbon tetrachloride -/
theorem molecular_weight_ccl4 : 
  (ccl4_weight * moles : ℝ) = 1384.29 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_ccl4_l2887_288795


namespace NUMINAMATH_CALUDE_expand_product_l2887_288742

theorem expand_product (x : ℝ) : 3 * (x + 4) * (x + 5) = 3 * x^2 + 27 * x + 60 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2887_288742


namespace NUMINAMATH_CALUDE_exists_graph_chromatic_3_no_3clique_l2887_288734

/-- A graph G with vertex set V and edge relation E -/
structure Graph (V : Type) :=
  (E : V → V → Prop)

/-- The chromatic number of a graph -/
def chromaticNumber {V : Type} (G : Graph V) : ℕ := sorry

/-- A clique of size n in a graph -/
def hasClique {V : Type} (G : Graph V) (n : ℕ) : Prop := sorry

theorem exists_graph_chromatic_3_no_3clique :
  ∃ (V : Type) (G : Graph V), chromaticNumber G = 3 ∧ ¬ hasClique G 3 := by sorry

end NUMINAMATH_CALUDE_exists_graph_chromatic_3_no_3clique_l2887_288734


namespace NUMINAMATH_CALUDE_parallel_vectors_projection_magnitude_l2887_288733

theorem parallel_vectors_projection_magnitude (a b : ℝ × ℝ) :
  (∃ (k : ℝ), a = k • b) →
  ¬(∀ (a b : ℝ × ℝ), (∃ (k : ℝ), a = k • b) → 
    ‖(a • b / (b • b)) • b‖ = ‖a‖) :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_projection_magnitude_l2887_288733


namespace NUMINAMATH_CALUDE_white_sox_games_lost_l2887_288737

theorem white_sox_games_lost (total_games won_games : ℕ) 
  (h1 : total_games = 162)
  (h2 : won_games = 99)
  (h3 : won_games = lost_games + 36) : lost_games = 63 :=
by
  sorry

end NUMINAMATH_CALUDE_white_sox_games_lost_l2887_288737


namespace NUMINAMATH_CALUDE_system_solution_condition_l2887_288750

/-- The system of equations has at least one solution if and only if -|a| ≤ b ≤ √2|a| -/
theorem system_solution_condition (a b : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 = a^2 ∧ x + |y| = b) ↔ -|a| ≤ b ∧ b ≤ Real.sqrt 2 * |a| :=
by sorry

end NUMINAMATH_CALUDE_system_solution_condition_l2887_288750


namespace NUMINAMATH_CALUDE_radical_calculations_l2887_288731

theorem radical_calculations :
  (∃ x y : ℝ, x^2 = 3 ∧ y^2 = 2 ∧
    (Real.sqrt 48 + Real.sqrt 8 - Real.sqrt 18 - Real.sqrt 12 = 2*x - y)) ∧
  (∃ a b c : ℝ, a^2 = 2 ∧ b^2 = 3 ∧ c^2 = 6 ∧
    (2*(a + b) - (b - a)^2 = 2*a + 2*b + 2*c - 5)) :=
by sorry

end NUMINAMATH_CALUDE_radical_calculations_l2887_288731


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l2887_288751

/-- A geometric progression is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
structure GeometricProgression where
  firstTerm : ℚ
  commonRatio : ℚ

/-- The n-th term of a geometric progression. -/
def nthTerm (gp : GeometricProgression) (n : ℕ) : ℚ :=
  gp.firstTerm * gp.commonRatio ^ (n - 1)

theorem geometric_progression_solution :
  ∃ (gp : GeometricProgression),
    nthTerm gp 2 = 37 + 1/3 ∧
    nthTerm gp 6 = 2 + 1/3 ∧
    gp.firstTerm = 224/3 ∧
    gp.commonRatio = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l2887_288751


namespace NUMINAMATH_CALUDE_bus_wheel_radius_proof_l2887_288738

/-- The speed of the bus in km/h -/
def bus_speed : ℝ := 66

/-- The revolutions per minute of the wheel -/
def wheel_rpm : ℝ := 125.11373976342128

/-- The radius of the wheel in centimeters -/
def wheel_radius : ℝ := 140.007

/-- Theorem stating that given the bus speed and wheel rpm, the wheel radius is approximately 140.007 cm -/
theorem bus_wheel_radius_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |wheel_radius - (bus_speed * 100000 / (60 * wheel_rpm * 2 * Real.pi))| < ε :=
sorry

end NUMINAMATH_CALUDE_bus_wheel_radius_proof_l2887_288738


namespace NUMINAMATH_CALUDE_july_birth_percentage_l2887_288784

theorem july_birth_percentage (total_scientists : ℕ) (july_births : ℕ) : 
  total_scientists = 150 → july_births = 15 → 
  (july_births : ℚ) / (total_scientists : ℚ) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_july_birth_percentage_l2887_288784


namespace NUMINAMATH_CALUDE_vector_relationships_l2887_288722

/-- Given two vectors OA and OB in R², this theorem states the value of m in OB
    when OA is perpendicular to OB and when OA is parallel to OB. -/
theorem vector_relationships (OA OB : ℝ × ℝ) (m : ℝ) : 
  OA = (-1, 2) → OB = (3, m) →
  ((OA.1 * OB.1 + OA.2 * OB.2 = 0 → m = 3/2) ∧
   (∃ k : ℝ, OB = (k * OA.1, k * OA.2) → m = -6)) := by
  sorry

end NUMINAMATH_CALUDE_vector_relationships_l2887_288722


namespace NUMINAMATH_CALUDE_fraction_meaningful_condition_l2887_288779

theorem fraction_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y = (x + 2) / (x - 1)) ↔ x ≠ 1 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_condition_l2887_288779


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l2887_288720

def num_flavors : ℕ := 4
def num_scoops : ℕ := 5

def total_distributions : ℕ := (num_scoops + num_flavors - 1).choose (num_flavors - 1)
def non_mint_distributions : ℕ := (num_scoops + (num_flavors - 1) - 1).choose ((num_flavors - 1) - 1)

theorem ice_cream_flavors :
  total_distributions - non_mint_distributions = 35 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l2887_288720


namespace NUMINAMATH_CALUDE_game_result_l2887_288791

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 12
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [5, 4, 1, 2, 6]
def betty_rolls : List ℕ := [6, 3, 3, 2, 1]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_result :
  (total_points allie_rolls) * (total_points betty_rolls) = 702 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l2887_288791


namespace NUMINAMATH_CALUDE_train_passing_time_l2887_288718

/-- The time taken for a person to walk the length of a train, given the times it takes for the train to pass the person in opposite and same directions. -/
theorem train_passing_time (t₁ t₂ : ℝ) (h₁ : t₁ > 0) (h₂ : t₂ > 0) (h₃ : t₂ > t₁) : 
  let t₃ := (2 * t₁ * t₂) / (t₂ - t₁)
  t₁ = 1 ∧ t₂ = 2 → t₃ = 4 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l2887_288718


namespace NUMINAMATH_CALUDE_literature_class_b_count_l2887_288766

/-- In a literature class with the given grade distribution, prove the number of B grades. -/
theorem literature_class_b_count (total : ℕ) (p_a p_b p_c : ℝ) (b_count : ℕ) : 
  total = 25 →
  p_a = 0.8 * p_b →
  p_c = 1.2 * p_b →
  p_a + p_b + p_c = 1 →
  b_count = ⌊(total : ℝ) / 3⌋ →
  b_count = 8 := by
sorry

end NUMINAMATH_CALUDE_literature_class_b_count_l2887_288766


namespace NUMINAMATH_CALUDE_cubic_roots_cosine_relation_l2887_288799

theorem cubic_roots_cosine_relation (p q r : ℝ) :
  (∃ α β γ : ℝ, α > 0 ∧ β > 0 ∧ γ > 0 ∧
    (∀ x : ℝ, x^3 + p*x^2 + q*x + r = 0 ↔ x = α ∨ x = β ∨ x = γ) ∧
    (∃ θ₁ θ₂ θ₃ : ℝ, θ₁ > 0 ∧ θ₂ > 0 ∧ θ₃ > 0 ∧ θ₁ + θ₂ + θ₃ = π ∧
      α = Real.cos θ₁ ∧ β = Real.cos θ₂ ∧ γ = Real.cos θ₃)) →
  2*r + 1 = p^2 - 2*q :=
sorry

end NUMINAMATH_CALUDE_cubic_roots_cosine_relation_l2887_288799


namespace NUMINAMATH_CALUDE_syllogism_structure_l2887_288788

-- Define syllogism as a structure in deductive reasoning
structure Syllogism where
  major_premise : Prop
  minor_premise : Prop
  conclusion : Prop

-- Define deductive reasoning
def DeductiveReasoning : Type := Prop → Prop

-- Theorem stating that syllogism in deductive reasoning consists of major premise, minor premise, and conclusion
theorem syllogism_structure (dr : DeductiveReasoning) :
  ∃ (s : Syllogism), dr s.major_premise ∧ dr s.minor_premise ∧ dr s.conclusion :=
sorry

end NUMINAMATH_CALUDE_syllogism_structure_l2887_288788
