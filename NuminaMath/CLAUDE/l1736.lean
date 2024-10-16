import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l1736_173683

theorem inequality_proof (m n : ℕ) (h : m < n) :
  m^2 + Real.sqrt (m^2 + m) < n^2 - Real.sqrt (n^2 - n) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1736_173683


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1736_173686

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  (∀ θ : ℝ, θ = 160 ∧ θ * n = 180 * (n - 2)) → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1736_173686


namespace NUMINAMATH_CALUDE_last_segment_speed_l1736_173612

def total_distance : ℝ := 120
def total_time : ℝ := 1.5
def segment_time : ℝ := 0.5
def speed_segment1 : ℝ := 50
def speed_segment2 : ℝ := 70

theorem last_segment_speed :
  ∃ (speed_segment3 : ℝ),
    (speed_segment1 * segment_time + speed_segment2 * segment_time + speed_segment3 * segment_time) / total_time = total_distance / total_time ∧
    speed_segment3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_last_segment_speed_l1736_173612


namespace NUMINAMATH_CALUDE_two_digit_primes_with_digit_sum_10_l1736_173692

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ Nat.Prime n ∧ digit_sum n = 10

theorem two_digit_primes_with_digit_sum_10 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, satisfies_conditions n) ∧ S.card = 4 :=
sorry

end NUMINAMATH_CALUDE_two_digit_primes_with_digit_sum_10_l1736_173692


namespace NUMINAMATH_CALUDE_shorts_cost_l1736_173653

theorem shorts_cost (num_players : ℕ) (jersey_cost sock_cost total_cost : ℚ) :
  num_players = 16 ∧ 
  jersey_cost = 25 ∧ 
  sock_cost = 6.80 ∧ 
  total_cost = 752 →
  ∃ shorts_cost : ℚ, 
    shorts_cost = 15.20 ∧ 
    num_players * (jersey_cost + shorts_cost + sock_cost) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_shorts_cost_l1736_173653


namespace NUMINAMATH_CALUDE_ratio_equality_l1736_173673

theorem ratio_equality (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) :
  (a / 3) / (b / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1736_173673


namespace NUMINAMATH_CALUDE_difference_of_fractions_l1736_173687

theorem difference_of_fractions (n : ℕ) : 
  (n / 10 : ℚ) - (n / 1000 : ℚ) = 693 ↔ n = 7000 := by sorry

end NUMINAMATH_CALUDE_difference_of_fractions_l1736_173687


namespace NUMINAMATH_CALUDE_equiangular_parallelogram_iff_rectangle_l1736_173694

/-- A parallelogram is a quadrilateral with opposite sides parallel. -/
structure Parallelogram :=
  (is_parallel : Bool)

/-- An equiangular parallelogram is a parallelogram with all angles equal. -/
structure EquiangularParallelogram extends Parallelogram :=
  (all_angles_equal : Bool)

/-- A rectangle is a parallelogram with all right angles. -/
structure Rectangle extends Parallelogram :=
  (all_angles_right : Bool)

/-- Theorem: A parallelogram is equiangular if and only if it is a rectangle. -/
theorem equiangular_parallelogram_iff_rectangle :
  ∀ p : Parallelogram, (∃ ep : EquiangularParallelogram, ep.toParallelogram = p) ↔ (∃ r : Rectangle, r.toParallelogram = p) :=
sorry

end NUMINAMATH_CALUDE_equiangular_parallelogram_iff_rectangle_l1736_173694


namespace NUMINAMATH_CALUDE_compare_numbers_l1736_173603

-- Define the base conversion function
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun digit acc => acc * base + digit) 0

-- Define the numbers in their respective bases
def num1 : List Nat := [8, 5]
def base1 : Nat := 9

def num2 : List Nat := [2, 1, 0]
def base2 : Nat := 6

def num3 : List Nat := [1, 0, 0, 0]
def base3 : Nat := 4

def num4 : List Nat := [1, 1, 1, 1, 1, 1]
def base4 : Nat := 2

-- State the theorem
theorem compare_numbers :
  to_decimal num2 base2 > to_decimal num1 base1 ∧
  to_decimal num1 base1 > to_decimal num3 base3 ∧
  to_decimal num3 base3 > to_decimal num4 base4 := by
  sorry

end NUMINAMATH_CALUDE_compare_numbers_l1736_173603


namespace NUMINAMATH_CALUDE_sum_equals_rounded_sum_jo_equals_alex_sum_l1736_173651

def round_to_nearest_five (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def sum_to_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def sum_rounded_to_five (n : ℕ) : ℕ :=
  (List.range n).map round_to_nearest_five |>.sum

theorem sum_equals_rounded_sum (n : ℕ) : sum_to_n n = sum_rounded_to_five n := by
  sorry

-- The main theorem
theorem jo_equals_alex_sum : sum_to_n 200 = sum_rounded_to_five 200 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_rounded_sum_jo_equals_alex_sum_l1736_173651


namespace NUMINAMATH_CALUDE_additional_donation_amount_l1736_173622

/-- A proof that the additional donation was $20.00 given the conditions of the raffle ticket sale --/
theorem additional_donation_amount (num_tickets : ℕ) (ticket_price : ℚ) (num_fixed_donations : ℕ) (fixed_donation_amount : ℚ) (total_raised : ℚ) : 
  num_tickets = 25 →
  ticket_price = 2 →
  num_fixed_donations = 2 →
  fixed_donation_amount = 15 →
  total_raised = 100 →
  total_raised - (↑num_tickets * ticket_price + ↑num_fixed_donations * fixed_donation_amount) = 20 :=
by
  sorry

#check additional_donation_amount

end NUMINAMATH_CALUDE_additional_donation_amount_l1736_173622


namespace NUMINAMATH_CALUDE_nabla_problem_l1736_173674

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem nabla_problem : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end NUMINAMATH_CALUDE_nabla_problem_l1736_173674


namespace NUMINAMATH_CALUDE_lemonade_sales_profit_difference_l1736_173639

/-- Lemonade sales problem -/
theorem lemonade_sales_profit_difference : 
  let katya_glasses : ℕ := 8
  let katya_price : ℚ := 3/2
  let katya_cost : ℚ := 1/2
  let ricky_glasses : ℕ := 9
  let ricky_price : ℚ := 2
  let ricky_cost : ℚ := 3/4
  let tina_price : ℚ := 3
  let tina_cost : ℚ := 1
  
  let katya_revenue := katya_glasses * katya_price
  let ricky_revenue := ricky_glasses * ricky_price
  let combined_revenue := katya_revenue + ricky_revenue
  let tina_target := 2 * combined_revenue
  
  let katya_profit := katya_revenue - (katya_glasses : ℚ) * katya_cost
  let tina_glasses := tina_target / tina_price
  let tina_profit := tina_target - tina_glasses * tina_cost
  
  tina_profit - katya_profit = 32
  := by sorry

end NUMINAMATH_CALUDE_lemonade_sales_profit_difference_l1736_173639


namespace NUMINAMATH_CALUDE_children_attending_show_l1736_173655

/-- Proves that the number of children attending the show is 3 --/
theorem children_attending_show :
  let adult_ticket_price : ℕ := 12
  let child_ticket_price : ℕ := 10
  let num_adults : ℕ := 3
  let total_cost : ℕ := 66
  let num_children : ℕ := (total_cost - num_adults * adult_ticket_price) / child_ticket_price
  num_children = 3 := by
sorry


end NUMINAMATH_CALUDE_children_attending_show_l1736_173655


namespace NUMINAMATH_CALUDE_quadratic_one_root_l1736_173663

/-- A quadratic function f(x) = mx^2 - 4x + 1 has exactly one root if and only if m ≤ 4 -/
theorem quadratic_one_root (m : ℝ) :
  (∃! x, m * x^2 - 4 * x + 1 = 0) ↔ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l1736_173663


namespace NUMINAMATH_CALUDE_constant_term_proof_l1736_173685

theorem constant_term_proof (a k n : ℤ) (x : ℝ) :
  (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n →
  a - n + k = 7 →
  n = -6 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_proof_l1736_173685


namespace NUMINAMATH_CALUDE_work_completion_time_l1736_173689

theorem work_completion_time (x : ℕ) : 
  (50 * x = 25 * (x + 20)) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1736_173689


namespace NUMINAMATH_CALUDE_least_n_for_inequality_l1736_173646

theorem least_n_for_inequality : 
  ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, k > 0 → (1 / k - 1 / (k + 1) < 1 / 15) → k ≥ n) ∧ (1 / n - 1 / (n + 1) < 1 / 15) ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_least_n_for_inequality_l1736_173646


namespace NUMINAMATH_CALUDE_product_multiple_of_60_probability_l1736_173633

def is_multiple_of_60 (n : ℕ) : Prop := ∃ k : ℕ, n = 60 * k

def count_favorable_pairs : ℕ := 732

def total_pairs : ℕ := 60 * 60

theorem product_multiple_of_60_probability :
  (count_favorable_pairs : ℚ) / (total_pairs : ℚ) = 61 / 300 := by sorry

end NUMINAMATH_CALUDE_product_multiple_of_60_probability_l1736_173633


namespace NUMINAMATH_CALUDE_mn_m_plus_n_is_even_l1736_173636

theorem mn_m_plus_n_is_even (m n : ℤ) : 2 ∣ (m * n * (m + n)) := by
  sorry

end NUMINAMATH_CALUDE_mn_m_plus_n_is_even_l1736_173636


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1736_173662

theorem max_value_sqrt_sum (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  (∀ x y, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → 
    Real.sqrt (a * b) + Real.sqrt ((1 - a) * (1 - b)) ≥ Real.sqrt (x * y) + Real.sqrt ((1 - x) * (1 - y))) ∧
  Real.sqrt (a * b) + Real.sqrt ((1 - a) * (1 - b)) ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1736_173662


namespace NUMINAMATH_CALUDE_subset_ratio_theorem_l1736_173610

theorem subset_ratio_theorem (n k : ℕ) (h1 : n ≥ 2*k) (h2 : 2*k > 3) :
  (Nat.choose n k = (2*n - k) * Nat.choose n 2) ↔ (n = 27 ∧ k = 4) := by
  sorry

end NUMINAMATH_CALUDE_subset_ratio_theorem_l1736_173610


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1736_173616

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_sum : 1 / (a * b) + 1 / (b * c) + 1 / (c * d) + 1 / (d * a) = 1) :
  a * b * c * d + 16 ≥ 8 * Real.sqrt ((a + c) * (1 / a + 1 / c)) +
    8 * Real.sqrt ((b + d) * (1 / b + 1 / d)) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1736_173616


namespace NUMINAMATH_CALUDE_bridge_building_time_l1736_173672

/-- Represents the time taken to build a bridge given a number of workers -/
def build_time (workers : ℕ) : ℝ := sorry

/-- The constant representing the total work required -/
def total_work : ℝ := 18 * 6

theorem bridge_building_time :
  (build_time 18 = 6) →
  (∀ w₁ w₂ : ℕ, w₁ * build_time w₁ = w₂ * build_time w₂) →
  build_time 30 = 3.6 := by sorry

end NUMINAMATH_CALUDE_bridge_building_time_l1736_173672


namespace NUMINAMATH_CALUDE_ratio_to_percent_l1736_173611

theorem ratio_to_percent (a b : ℚ) (h : a / b = 2 / 10) : (a / b) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_percent_l1736_173611


namespace NUMINAMATH_CALUDE_chocolate_sales_l1736_173664

theorem chocolate_sales (cost_price selling_price : ℝ) (n : ℕ) : 
  44 * cost_price = n * selling_price →
  selling_price = cost_price * (1 + 5/6) →
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_chocolate_sales_l1736_173664


namespace NUMINAMATH_CALUDE_all_diagonals_bisect_l1736_173676

-- Define a type for quadrilaterals
inductive Quadrilateral
  | Parallelogram
  | Rectangle
  | Rhombus
  | Square

-- Define a property that diagonals bisect each other
def diagonals_bisect (q : Quadrilateral) : Prop :=
  match q with
  | Quadrilateral.Parallelogram => true
  | Quadrilateral.Rectangle => true
  | Quadrilateral.Rhombus => true
  | Quadrilateral.Square => true

-- Theorem stating that all these quadrilaterals have diagonals that bisect each other
theorem all_diagonals_bisect :
  ∀ q : Quadrilateral, diagonals_bisect q :=
by sorry

end NUMINAMATH_CALUDE_all_diagonals_bisect_l1736_173676


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1736_173626

theorem angle_between_vectors (a b : ℝ × ℝ) :
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 2) →
  (Real.sqrt (a.1^2 + a.2^2) = 1) →
  (Real.sqrt (b.1^2 + b.2^2) = 2) →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1736_173626


namespace NUMINAMATH_CALUDE_fourth_power_sum_l1736_173630

theorem fourth_power_sum (a b c : ℝ) 
  (sum_condition : a + b + c = 3)
  (sum_squares : a^2 + b^2 + c^2 = 5)
  (sum_cubes : a^3 + b^3 + c^3 = 15) :
  a^4 + b^4 + c^4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l1736_173630


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l1736_173656

def number_to_express : ℝ := 460000000

theorem scientific_notation_proof :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ number_to_express = a * (10 : ℝ) ^ n ∧ a = 4.6 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l1736_173656


namespace NUMINAMATH_CALUDE_limit_equals_derivative_l1736_173654

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem limit_equals_derivative
  (h1 : Differentiable ℝ f)  -- f is differentiable
  (h2 : deriv f 1 = 1) :     -- f'(1) = 1
  Filter.Tendsto
    (fun Δx => (f (1 - Δx) - f 1) / (-Δx))
    (nhds 0)
    (nhds 1) :=
by
  sorry

end NUMINAMATH_CALUDE_limit_equals_derivative_l1736_173654


namespace NUMINAMATH_CALUDE_roots_sum_of_cubes_reciprocal_l1736_173661

theorem roots_sum_of_cubes_reciprocal (a b c : ℝ) (r s : ℂ) 
  (hr : a * r^2 + b * r - c = 0) 
  (hs : a * s^2 + b * s - c = 0) 
  (ha : a ≠ 0) 
  (hc : c ≠ 0) : 
  1 / r^3 + 1 / s^3 = (b^3 + 3*a*b*c) / c^3 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_cubes_reciprocal_l1736_173661


namespace NUMINAMATH_CALUDE_log_50_between_consecutive_integers_l1736_173693

theorem log_50_between_consecutive_integers : 
  ∃ (m n : ℤ), m + 1 = n ∧ (m : ℝ) < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < n ∧ m + n = 3 := by
sorry

end NUMINAMATH_CALUDE_log_50_between_consecutive_integers_l1736_173693


namespace NUMINAMATH_CALUDE_white_pairs_coincide_l1736_173648

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_white : ℕ
  white_white : ℕ

def figure_problem (counts : TriangleCounts) (pairs : CoincidingPairs) : Prop :=
  counts.red = 4 ∧
  counts.blue = 6 ∧
  counts.white = 10 ∧
  pairs.red_red = 3 ∧
  pairs.blue_blue = 4 ∧
  pairs.red_white = 3 ∧
  pairs.white_white = 7

theorem white_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) :
  figure_problem counts pairs → pairs.white_white = 7 := by
  sorry

end NUMINAMATH_CALUDE_white_pairs_coincide_l1736_173648


namespace NUMINAMATH_CALUDE_floor_length_approximately_18_78_l1736_173668

/-- Represents the dimensions and painting cost of a rectangular floor. -/
structure Floor :=
  (breadth : ℝ)
  (paintCost : ℝ)
  (paintRate : ℝ)

/-- Calculates the length of the floor given its specifications. -/
def calculateFloorLength (floor : Floor) : ℝ :=
  let length := 3 * floor.breadth
  let area := floor.paintCost / floor.paintRate
  length

/-- Theorem stating that the calculated floor length is approximately 18.78 meters. -/
theorem floor_length_approximately_18_78 (floor : Floor) 
  (h1 : floor.paintCost = 529)
  (h2 : floor.paintRate = 3) :
  ∃ ε > 0, |calculateFloorLength floor - 18.78| < ε :=
sorry

end NUMINAMATH_CALUDE_floor_length_approximately_18_78_l1736_173668


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1736_173665

def A : Set ℝ := {x | x^2 - 2*x ≤ 0}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1736_173665


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1736_173637

theorem simplify_and_evaluate (x : ℝ) (h : x ≠ 3) :
  (x^2 - x - 6) / (x - 3) = x + 2 ∧ 
  (4^2 - 4 - 6) / (4 - 3) = 6 :=
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1736_173637


namespace NUMINAMATH_CALUDE_max_value_of_b_l1736_173604

/-- Given functions f and g with a common point and tangent, prove the maximum value of b -/
theorem max_value_of_b (a : ℝ) (h_a : a > 0) : 
  let f := fun x : ℝ => (1/2) * x^2 + 2 * a * x
  let g := fun x b : ℝ => 3 * a^2 * Real.log x + b
  ∃ (x₀ b₀ : ℝ), 
    (f x₀ = g x₀ b₀) ∧ 
    (deriv f x₀ = deriv (fun x => g x b₀) x₀) →
  (∀ b : ℝ, ∃ (x : ℝ), (f x = g x b) ∧ (deriv f x = deriv (fun x => g x b) x) → b ≤ (3/2) * Real.exp ((2/3) : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_b_l1736_173604


namespace NUMINAMATH_CALUDE_triangle_side_length_l1736_173606

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  S = (1/2) * b * c * Real.sin A ∧
  S = Real.sqrt 3 ∧
  b = 1 ∧
  A = π/3 →
  a = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1736_173606


namespace NUMINAMATH_CALUDE_fraction_equals_26_11_l1736_173642

def numerator : ℕ → ℚ
| 0 => 15
| n + 1 => (numerator n) * (1 + 14 / (n + 2))

def denominator : ℕ → ℚ
| 0 => 13
| n + 1 => (denominator n) * (1 + 12 / (n + 2))

def fraction : ℚ := (numerator 11) / (denominator 9)

theorem fraction_equals_26_11 : fraction = 26 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_26_11_l1736_173642


namespace NUMINAMATH_CALUDE_apple_banana_cost_l1736_173625

/-- The total cost of buying apples and bananas -/
def total_cost (a b : ℝ) : ℝ := 3 * a + 4 * b

/-- Theorem stating that the total cost of buying 3 kg of apples at 'a' yuan/kg
    and 4 kg of bananas at 'b' yuan/kg is (3a + 4b) yuan -/
theorem apple_banana_cost (a b : ℝ) :
  total_cost a b = 3 * a + 4 * b := by
  sorry

end NUMINAMATH_CALUDE_apple_banana_cost_l1736_173625


namespace NUMINAMATH_CALUDE_men_french_percentage_l1736_173647

/-- Represents the percentage of employees who are men -/
def percent_men : ℝ := 0.35

/-- Represents the percentage of employees who speak French -/
def percent_french : ℝ := 0.40

/-- Represents the percentage of women who do not speak French -/
def percent_women_not_french : ℝ := 0.7077

/-- Represents the percentage of men who speak French -/
def percent_men_french : ℝ := 0.60

theorem men_french_percentage :
  percent_men * percent_men_french + (1 - percent_men) * (1 - percent_women_not_french) = percent_french :=
sorry


end NUMINAMATH_CALUDE_men_french_percentage_l1736_173647


namespace NUMINAMATH_CALUDE_derek_age_is_20_l1736_173635

-- Define the ages as natural numbers
def aunt_beatrice_age : ℕ := 54

-- Define Emily's age in terms of Aunt Beatrice's age
def emily_age : ℕ := aunt_beatrice_age / 2

-- Define Derek's age in terms of Emily's age
def derek_age : ℕ := emily_age - 7

-- Theorem statement
theorem derek_age_is_20 : derek_age = 20 := by
  sorry

end NUMINAMATH_CALUDE_derek_age_is_20_l1736_173635


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1736_173632

/-- Proves that in a group of 9 players, if each player plays every other player
    the same number of times, and a total of 36 games are played, then each
    player must play every other player exactly once. -/
theorem chess_tournament_games (n : ℕ) (total_games : ℕ) 
    (h1 : n = 9)
    (h2 : total_games = 36)
    (h3 : ∀ i j : Fin n, i ≠ j → ∃ k : ℕ, k > 0) :
  ∀ i j : Fin n, i ≠ j → ∃ k : ℕ, k = 1 := by
  sorry

#check chess_tournament_games

end NUMINAMATH_CALUDE_chess_tournament_games_l1736_173632


namespace NUMINAMATH_CALUDE_factor_x9_minus_512_l1736_173671

theorem factor_x9_minus_512 (x : ℝ) : 
  x^9 - 512 = (x - 2) * (x^2 + 2*x + 4) * (x^6 + 8*x^3 + 64) := by
  sorry

end NUMINAMATH_CALUDE_factor_x9_minus_512_l1736_173671


namespace NUMINAMATH_CALUDE_circle_equation_l1736_173629

/-- A circle with center (1, -2) and radius 3 -/
structure Circle where
  center : ℝ × ℝ := (1, -2)
  radius : ℝ := 3

/-- A point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on the circle -/
def onCircle (c : Circle) (p : Point) : Prop :=
  (p.x - c.center.1)^2 + (p.y - c.center.2)^2 = c.radius^2

theorem circle_equation (c : Circle) (p : Point) :
  onCircle c p ↔ (p.x - 1)^2 + (p.y + 2)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l1736_173629


namespace NUMINAMATH_CALUDE_hyperbola_C_tangent_intersection_product_l1736_173634

/-- Hyperbola C -/
def hyperbola_C (x y : ℝ) : Prop := x^2 / 6 - y^2 / 3 = 1

/-- Point P on the line x = 2 -/
def point_P (t : ℝ) : ℝ × ℝ := (2, t)

/-- Function to calculate mn given t -/
noncomputable def mn (t : ℝ) : ℝ := 6 * Real.sqrt 6 - 15

theorem hyperbola_C_tangent_intersection_product :
  hyperbola_C (-3) (Real.sqrt 6 / 2) →
  ∀ t : ℝ, ∃ m n : ℝ,
    (∃ A B : ℝ × ℝ, 
      hyperbola_C A.1 A.2 ∧ 
      hyperbola_C B.1 B.2 ∧ 
      -- PA and PB are tangent to C
      -- M and N are defined as in the problem
      mn t = m * n) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_C_tangent_intersection_product_l1736_173634


namespace NUMINAMATH_CALUDE_tyrone_quarters_l1736_173608

/-- Represents the count of each type of coin or bill --/
structure CoinCount where
  dollars_1 : ℕ
  dollars_5 : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total value in dollars of a given coin count, excluding quarters --/
def value_without_quarters (c : CoinCount) : ℚ :=
  c.dollars_1 + 5 * c.dollars_5 + 0.1 * c.dimes + 0.05 * c.nickels + 0.01 * c.pennies

/-- The value of a quarter in dollars --/
def quarter_value : ℚ := 0.25

theorem tyrone_quarters : 
  ∀ (c : CoinCount) (total : ℚ),
    c.dollars_1 = 2 →
    c.dollars_5 = 1 →
    c.dimes = 20 →
    c.nickels = 8 →
    c.pennies = 35 →
    total = 13 →
    (total - value_without_quarters c) / quarter_value = 13 := by
  sorry

end NUMINAMATH_CALUDE_tyrone_quarters_l1736_173608


namespace NUMINAMATH_CALUDE_largest_two_digit_divisible_by_6_ending_in_4_l1736_173638

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∀ n : ℕ, is_two_digit n → n % 6 = 0 → ends_in_4 n → n ≤ 84 :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_divisible_by_6_ending_in_4_l1736_173638


namespace NUMINAMATH_CALUDE_box_with_balls_l1736_173669

theorem box_with_balls (total : ℕ) (white : ℕ) (blue : ℕ) (red : ℕ) : 
  total = 100 →
  blue = white + 12 →
  red = 2 * blue →
  total = white + blue + red →
  white = 16 := by
sorry

end NUMINAMATH_CALUDE_box_with_balls_l1736_173669


namespace NUMINAMATH_CALUDE_expression_value_l1736_173600

theorem expression_value (m n a b x : ℝ) 
  (h1 : m = -n)  -- m and n are opposites
  (h2 : a * b = -1)  -- a and b are negative reciprocals
  (h3 : |x| = 3)  -- absolute value of x equals 3
  : x^3 - (1 + m + n + a*b) * x^2 + (m + n) * x^2004 + (a*b)^2005 = 26 ∨ 
    x^3 - (1 + m + n + a*b) * x^2 + (m + n) * x^2004 + (a*b)^2005 = -28 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l1736_173600


namespace NUMINAMATH_CALUDE_gcd_problem_l1736_173699

/-- The operation * represents the greatest common divisor -/
def gcd_op (a b : ℕ) : ℕ := Nat.gcd a b

/-- Theorem stating that ((16 * 20) * (18 * 24)) = 2 using the gcd operation -/
theorem gcd_problem : gcd_op (gcd_op 16 20) (gcd_op 18 24) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1736_173699


namespace NUMINAMATH_CALUDE_shooting_probabilities_l1736_173645

/-- Represents the probabilities of hitting each ring in a shooting game. -/
structure RingProbabilities where
  ten : Real
  nine : Real
  eight : Real
  seven : Real
  sub_seven : Real

/-- The probabilities sum to 1. -/
axiom prob_sum_to_one (p : RingProbabilities) : 
  p.ten + p.nine + p.eight + p.seven + p.sub_seven = 1

/-- The given probabilities for each ring. -/
def given_probs : RingProbabilities := {
  ten := 0.24,
  nine := 0.28,
  eight := 0.19,
  seven := 0.16,
  sub_seven := 0.13
}

/-- The probability of hitting 10 or 9 rings. -/
def prob_ten_or_nine (p : RingProbabilities) : Real :=
  p.ten + p.nine

/-- The probability of hitting at least 7 ring. -/
def prob_at_least_seven (p : RingProbabilities) : Real :=
  p.ten + p.nine + p.eight + p.seven

/-- The probability of not hitting 8 ring. -/
def prob_not_eight (p : RingProbabilities) : Real :=
  1 - p.eight

theorem shooting_probabilities (p : RingProbabilities) 
  (h : p = given_probs) : 
  prob_ten_or_nine p = 0.52 ∧ 
  prob_at_least_seven p = 0.87 ∧ 
  prob_not_eight p = 0.81 := by
  sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l1736_173645


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1736_173644

theorem fraction_subtraction (d : ℝ) : (6 - 5 * d) / 9 - 3 = (-21 - 5 * d) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1736_173644


namespace NUMINAMATH_CALUDE_square_sum_xy_l1736_173621

theorem square_sum_xy (x y : ℝ) 
  (h1 : x * (x + y) = 40)
  (h2 : y * (x + y) = 90)
  (h3 : x = (2/3) * y) : 
  (x + y)^2 = 100 := by
sorry

end NUMINAMATH_CALUDE_square_sum_xy_l1736_173621


namespace NUMINAMATH_CALUDE_parabolic_arch_height_l1736_173658

/-- Represents a parabolic arch --/
structure ParabolicArch where
  width : ℝ
  area : ℝ

/-- Calculates the height of a parabolic arch given its width and area --/
def archHeight (arch : ParabolicArch) : ℝ :=
  sorry

/-- Theorem stating that a parabolic arch with width 8 and area 160 has height 30 --/
theorem parabolic_arch_height :
  let arch : ParabolicArch := { width := 8, area := 160 }
  archHeight arch = 30 := by sorry

end NUMINAMATH_CALUDE_parabolic_arch_height_l1736_173658


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l1736_173631

/-- Represents a parabola of the form y = x^2 + bx + c -/
structure Parabola where
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem stating the conditions and conclusion about the parabola -/
theorem parabola_intersection_theorem (p : Parabola) 
  (A B C : Point) :
  (A.x = 0) →  -- A is on y-axis
  (B.x > 0 ∧ C.x > 0) →  -- B and C are on positive x-axis
  (B.y = 0 ∧ C.y = 0) →  -- B and C are on x-axis
  (A.y = p.c) →  -- A is the y-intercept
  (C.x - B.x = 2) →  -- BC = 2
  (1/2 * A.y * (C.x - B.x) = 3) →  -- Area of triangle ABC is 3
  (p.b = -4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l1736_173631


namespace NUMINAMATH_CALUDE_expression_evaluation_l1736_173695

theorem expression_evaluation :
  let x : ℚ := 1/4
  let y : ℚ := 1/2
  let z : ℚ := 3
  4 * (x^3 * y^2 * z^2) = 9/64 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1736_173695


namespace NUMINAMATH_CALUDE_solve_equation_l1736_173675

theorem solve_equation (x : ℝ) : 3*x - 5*x + 4*x + 6 = 138 → x = 66 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1736_173675


namespace NUMINAMATH_CALUDE_product_xyz_equals_negative_one_l1736_173667

theorem product_xyz_equals_negative_one (x y z : ℝ) 
  (h1 : x + 1/y = 3) (h2 : y + 1/z = 3) : x * y * z = -1 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_equals_negative_one_l1736_173667


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l1736_173615

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (x ^ 2 - 1) / (x - 1) = 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l1736_173615


namespace NUMINAMATH_CALUDE_work_completion_time_l1736_173684

/-- Given two workers X and Y, where X can finish a job in 21 days and Y in 15 days,
    if Y works for 5 days and then leaves, prove that X needs 14 days to finish the remaining work. -/
theorem work_completion_time (x_rate y_rate : ℚ) (y_days : ℕ) :
  x_rate = 1 / 21 →
  y_rate = 1 / 15 →
  y_days = 5 →
  (1 - y_rate * y_days) / x_rate = 14 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1736_173684


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l1736_173682

/-- Ellipse C with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Line l with given properties -/
structure Line where
  m : ℝ

/-- Theorem stating the properties of the ellipse and line -/
theorem ellipse_and_line_properties
  (C : Ellipse)
  (e : ℝ)
  (min_dist : ℝ)
  (l : Line)
  (AB : ℝ)
  (h1 : e = Real.sqrt 3 / 3)
  (h2 : min_dist = Real.sqrt 3 - 1)
  (h3 : AB = 8 * Real.sqrt 3 / 5) :
  (∃ x y, x^2 / 3 + y^2 / 2 = 1) ∧
  (l.m = 1 ∨ l.m = -1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l1736_173682


namespace NUMINAMATH_CALUDE_small_cuboid_height_l1736_173640

/-- Proves that the height of small cuboids is 3 meters given the dimensions of a large cuboid
    and the dimensions of small cuboids that can be formed from it. -/
theorem small_cuboid_height
  (large_length large_width large_height : ℝ)
  (small_length small_width : ℝ)
  (num_small_cuboids : ℕ)
  (h_large_length : large_length = 18)
  (h_large_width : large_width = 15)
  (h_large_height : large_height = 2)
  (h_small_length : small_length = 5)
  (h_small_width : small_width = 2)
  (h_num_small_cuboids : num_small_cuboids = 18)
  : ∃ (small_height : ℝ),
    small_height = 3 ∧
    large_length * large_width * large_height =
    num_small_cuboids * small_length * small_width * small_height :=
by sorry

end NUMINAMATH_CALUDE_small_cuboid_height_l1736_173640


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l1736_173649

theorem min_value_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y^2 = 4) :
  x + 2*y ≥ 3 * (4 : ℝ)^(1/3) ∧ 
  ∃ (x₀ y₀ : ℝ), 0 < x₀ ∧ 0 < y₀ ∧ x₀ * y₀^2 = 4 ∧ x₀ + 2*y₀ = 3 * (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l1736_173649


namespace NUMINAMATH_CALUDE_max_value_fraction_l1736_173697

theorem max_value_fraction (x : ℝ) :
  x^4 / (x^8 + 2*x^6 - 4*x^4 + 8*x^2 + 16) ≤ 1/12 ∧
  ∃ y : ℝ, y^4 / (y^8 + 2*y^6 - 4*y^4 + 8*y^2 + 16) = 1/12 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1736_173697


namespace NUMINAMATH_CALUDE_stating_arrangements_count_l1736_173660

/-- 
Given a positive integer n, this function returns the number of arrangements
of integers 1 to n, where each number (except the leftmost) differs by 1
from some number to its left.
-/
def countArrangements (n : ℕ) : ℕ :=
  2^(n-1)

/-- 
Theorem stating that the number of arrangements of integers 1 to n,
where each number (except the leftmost) differs by 1 from some number to its left,
is equal to 2^(n-1).
-/
theorem arrangements_count (n : ℕ) (h : n > 0) :
  countArrangements n = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_stating_arrangements_count_l1736_173660


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1736_173602

-- First expression
theorem simplify_expression_1 (a b : ℝ) : (1 : ℝ) * (4 * a - 2 * b) - (5 * a - 3 * b) = -a + b := by
  sorry

-- Second expression
theorem simplify_expression_2 (x : ℝ) : 2 * (2 * x^2 + 3 * x - 1) - (4 * x^2 + 2 * x - 2) = 4 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1736_173602


namespace NUMINAMATH_CALUDE_max_knights_between_knights_theorem_l1736_173691

/-- Represents a seating arrangement of knights and samurais around a round table. -/
structure SeatingArrangement where
  total_knights : Nat
  total_samurais : Nat
  knights_with_samurai_right : Nat

/-- Calculates the maximum number of knights that could be seated next to two other knights. -/
def max_knights_between_knights (arrangement : SeatingArrangement) : Nat :=
  arrangement.total_knights - (arrangement.knights_with_samurai_right + 1)

/-- Theorem stating the maximum number of knights seated next to two other knights
    for the given arrangement. -/
theorem max_knights_between_knights_theorem (arrangement : SeatingArrangement) 
  (h1 : arrangement.total_knights = 40)
  (h2 : arrangement.total_samurais = 10)
  (h3 : arrangement.knights_with_samurai_right = 7) :
  max_knights_between_knights arrangement = 32 := by
  sorry

#eval max_knights_between_knights { total_knights := 40, total_samurais := 10, knights_with_samurai_right := 7 }

end NUMINAMATH_CALUDE_max_knights_between_knights_theorem_l1736_173691


namespace NUMINAMATH_CALUDE_garrison_reinforcement_reinforcement_size_l1736_173609

theorem garrison_reinforcement (initial_garrison : ℕ) (initial_days : ℕ) 
  (days_before_reinforcement : ℕ) (days_after_reinforcement : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_days
  let remaining_provisions := total_provisions - (initial_garrison * days_before_reinforcement)
  let reinforcement := (remaining_provisions / days_after_reinforcement) - initial_garrison
  reinforcement

theorem reinforcement_size :
  garrison_reinforcement 1000 60 15 20 = 1250 := by sorry

end NUMINAMATH_CALUDE_garrison_reinforcement_reinforcement_size_l1736_173609


namespace NUMINAMATH_CALUDE_square_equation_solution_l1736_173601

theorem square_equation_solution (x : ℝ) (h : x^2 - 100 = -75) : x = -5 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l1736_173601


namespace NUMINAMATH_CALUDE_gcd_84_36_l1736_173628

theorem gcd_84_36 : Nat.gcd 84 36 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_84_36_l1736_173628


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_m_when_A_subset_C_l1736_173690

-- Define sets A, B, and C
def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B : Set ℝ := {x | 6*x^2 - 5*x + 1 ≥ 0}
def C (m : ℝ) : Set ℝ := {x | (x - m)*(x - m - 9) < 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B :
  A ∩ B = {x | -1 < x ∧ x ≤ 1/3 ∨ 1/2 ≤ x ∧ x < 6} :=
sorry

-- Theorem for the range of m when A is a subset of C
theorem range_of_m_when_A_subset_C :
  (∀ m : ℝ, A ⊆ C m → -3 ≤ m ∧ m ≤ -1) ∧
  (∀ m : ℝ, -3 ≤ m ∧ m ≤ -1 → A ⊆ C m) :=
sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_m_when_A_subset_C_l1736_173690


namespace NUMINAMATH_CALUDE_wiener_age_theorem_l1736_173627

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

theorem wiener_age_theorem :
  ∃! a : ℕ, 
    is_four_digit (a^3) ∧ 
    is_six_digit (a^4) ∧ 
    (digits (a^3) ++ digits (a^4)).Nodup ∧
    (digits (a^3) ++ digits (a^4)).length = 10 ∧
    a = 18 := by
  sorry

end NUMINAMATH_CALUDE_wiener_age_theorem_l1736_173627


namespace NUMINAMATH_CALUDE_art_fair_sales_l1736_173688

theorem art_fair_sales (total_customers : ℕ) (two_painting_buyers : ℕ) 
  (one_painting_buyers : ℕ) (four_painting_buyers : ℕ) (total_paintings_sold : ℕ) :
  total_customers = 20 →
  one_painting_buyers = 12 →
  four_painting_buyers = 4 →
  total_paintings_sold = 36 →
  two_painting_buyers + one_painting_buyers + four_painting_buyers = total_customers →
  2 * two_painting_buyers + one_painting_buyers + 4 * four_painting_buyers = total_paintings_sold →
  two_painting_buyers = 4 := by
sorry

end NUMINAMATH_CALUDE_art_fair_sales_l1736_173688


namespace NUMINAMATH_CALUDE_girls_count_in_school_l1736_173643

/-- Represents the number of students in a school with a given boy-to-girl ratio. -/
structure School where
  total : ℕ
  ratio : ℚ
  boys : ℕ
  girls : ℕ
  ratio_def : ratio = boys / girls
  total_def : total = boys + girls

/-- Theorem: In a school with 90 students and a 1:2 boy-to-girl ratio, there are 60 girls. -/
theorem girls_count_in_school (s : School) 
    (h_total : s.total = 90)
    (h_ratio : s.ratio = 1/2) : 
    s.girls = 60 := by
  sorry

end NUMINAMATH_CALUDE_girls_count_in_school_l1736_173643


namespace NUMINAMATH_CALUDE_min_value_M_min_value_expression_min_value_equality_condition_l1736_173657

open Real

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Part I
theorem min_value_M : ∃ (M : ℝ), (∃ (x₀ : ℝ), f x₀ ≤ M) ∧ ∀ (m : ℝ), (∃ (x : ℝ), f x ≤ m) → M ≤ m :=
sorry

-- Part II
theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3 * a + b = 2) :
  1 / (2 * a) + 1 / (a + b) ≥ 2 :=
sorry

theorem min_value_equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3 * a + b = 2) :
  1 / (2 * a) + 1 / (a + b) = 2 ↔ a = 1/2 ∧ b = 1/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_M_min_value_expression_min_value_equality_condition_l1736_173657


namespace NUMINAMATH_CALUDE_total_cash_reward_l1736_173650

/-- Represents a subject with its grade, credit hours, and cash reward per grade point -/
structure Subject where
  name : String
  grade : Nat
  creditHours : Nat
  cashRewardPerPoint : Nat

/-- Calculates the total cash reward for a given subject -/
def subjectReward (s : Subject) : Nat :=
  s.grade * s.cashRewardPerPoint

/-- Represents the artwork reward -/
def artworkReward : Nat := 20

/-- List of subjects with their respective information -/
def subjects : List Subject := [
  ⟨"Mathematics", 2, 5, 5⟩,
  ⟨"English", 3, 4, 4⟩,
  ⟨"Spanish", 3, 4, 4⟩,
  ⟨"Physics", 3, 4, 3⟩,
  ⟨"Chemistry", 3, 3, 3⟩,
  ⟨"History", 4, 3, 5⟩
]

/-- Calculates the total cash reward for all subjects -/
def totalSubjectsReward : Nat :=
  (subjects.map subjectReward).sum

/-- Theorem: The total cash reward Milo gets is $92 -/
theorem total_cash_reward : totalSubjectsReward + artworkReward = 92 := by
  sorry

end NUMINAMATH_CALUDE_total_cash_reward_l1736_173650


namespace NUMINAMATH_CALUDE_range_of_a_l1736_173623

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y

-- Define the theorem
theorem range_of_a : 
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → -2 < a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1736_173623


namespace NUMINAMATH_CALUDE_positive_quadratic_expression_l1736_173618

theorem positive_quadratic_expression (x y : ℝ) : x^2 + 2*y^2 + 2*x*y + 6*y + 10 > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_quadratic_expression_l1736_173618


namespace NUMINAMATH_CALUDE_item_costs_l1736_173679

/-- The cost of items in yuan -/
structure ItemCosts where
  tableLamp : ℕ
  electricFan : ℕ
  bicycle : ℕ

/-- Theorem stating the total cost of all items and the cost of lamp and fan -/
theorem item_costs (c : ItemCosts) 
  (h1 : c.tableLamp = 86)
  (h2 : c.electricFan = 185)
  (h3 : c.bicycle = 445) :
  (c.tableLamp + c.electricFan + c.bicycle = 716) ∧
  (c.tableLamp + c.electricFan = 271) := by
  sorry

#check item_costs

end NUMINAMATH_CALUDE_item_costs_l1736_173679


namespace NUMINAMATH_CALUDE_spinner_probabilities_l1736_173620

theorem spinner_probabilities :
  ∀ (p_C : ℚ),
  (1 / 4 : ℚ) + (1 / 3 : ℚ) + p_C + p_C = 1 →
  p_C = (5 / 24 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_spinner_probabilities_l1736_173620


namespace NUMINAMATH_CALUDE_undefined_rational_expression_expression_undefined_l1736_173613

theorem undefined_rational_expression (x : ℝ) : 
  (x^2 - 12*x + 36 = 0) ↔ (x = 6) := by sorry

theorem expression_undefined (x : ℝ) :
  ¬∃y, y = (3*x^3 + 5) / (x^2 - 12*x + 36) ↔ x = 6 := by sorry

end NUMINAMATH_CALUDE_undefined_rational_expression_expression_undefined_l1736_173613


namespace NUMINAMATH_CALUDE_y_intercept_of_given_line_l1736_173619

/-- A line in the form y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis -/
def y_intercept (l : Line) : ℝ := l.b

/-- Given line with equation y = 3x + 2 -/
def given_line : Line := { m := 3, b := 2 }

/-- Theorem: The y-intercept of the given line is 2 -/
theorem y_intercept_of_given_line : 
  y_intercept given_line = 2 := by sorry

end NUMINAMATH_CALUDE_y_intercept_of_given_line_l1736_173619


namespace NUMINAMATH_CALUDE_endpoint_sum_coordinates_endpoint_sum_coordinates_proof_l1736_173652

/-- Given a line segment with one endpoint (6, 2) and midpoint (3, 7),
    the sum of coordinates of the other endpoint is 12. -/
theorem endpoint_sum_coordinates : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop :=
  fun endpoint1 midpoint endpoint2 =>
    endpoint1 = (6, 2) ∧
    midpoint = (3, 7) ∧
    midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) →
    endpoint2.1 + endpoint2.2 = 12
    
#check endpoint_sum_coordinates

theorem endpoint_sum_coordinates_proof : 
  ∃ (endpoint2 : ℝ × ℝ), endpoint_sum_coordinates (6, 2) (3, 7) endpoint2 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_sum_coordinates_endpoint_sum_coordinates_proof_l1736_173652


namespace NUMINAMATH_CALUDE_rs_length_l1736_173666

/-- Right-angled triangle PQR with perpendiculars to PQ at P and QR at R meeting at S -/
structure SpecialTriangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  pq_length : dist P Q = 6
  qr_length : dist Q R = 8
  right_angle : (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0
  perp_at_p : (S.1 - P.1) * (Q.1 - P.1) + (S.2 - P.2) * (Q.2 - P.2) = 0
  perp_at_r : (S.1 - R.1) * (Q.1 - R.1) + (S.2 - R.2) * (Q.2 - R.2) = 0

/-- The length of RS in the special triangle is 8 -/
theorem rs_length (t : SpecialTriangle) : dist t.R t.S = 8 := by
  sorry

end NUMINAMATH_CALUDE_rs_length_l1736_173666


namespace NUMINAMATH_CALUDE_grandpa_grandchildren_ages_l1736_173607

theorem grandpa_grandchildren_ages (grandpa_age : ℕ) (gc1_age gc2_age gc3_age : ℕ) (years : ℕ) :
  grandpa_age = 75 →
  gc1_age = 13 →
  gc2_age = 15 →
  gc3_age = 17 →
  years = 15 →
  grandpa_age + years = (gc1_age + years) + (gc2_age + years) + (gc3_age + years) :=
by sorry

end NUMINAMATH_CALUDE_grandpa_grandchildren_ages_l1736_173607


namespace NUMINAMATH_CALUDE_days_took_capsules_l1736_173617

/-- The number of days in July -/
def days_in_july : ℕ := 31

/-- The number of days Isla forgot to take capsules -/
def days_forgot : ℕ := 2

/-- Theorem: The number of days Isla took capsules in July is 29 -/
theorem days_took_capsules : days_in_july - days_forgot = 29 := by
  sorry

end NUMINAMATH_CALUDE_days_took_capsules_l1736_173617


namespace NUMINAMATH_CALUDE_subtracted_value_l1736_173614

theorem subtracted_value (chosen_number : ℕ) (final_result : ℕ) : 
  chosen_number = 63 → final_result = 110 → 
  (chosen_number * 4 - final_result) = 142 := by
sorry

end NUMINAMATH_CALUDE_subtracted_value_l1736_173614


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_k_nonpositive_l1736_173678

/-- A function f(x) = kx² + (3k-2)x - 5 is monotonically decreasing on [1, +∞) -/
def is_monotone_decreasing (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x < y → f y < f x

/-- The main theorem stating that if f(x) = kx² + (3k-2)x - 5 is monotonically
    decreasing on [1, +∞), then k ∈ (-∞, 0] -/
theorem monotone_decreasing_implies_k_nonpositive (k : ℝ) :
  is_monotone_decreasing (fun x => k*x^2 + (3*k-2)*x - 5) k →
  k ∈ Set.Iic 0 :=
by sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_k_nonpositive_l1736_173678


namespace NUMINAMATH_CALUDE_central_figure_diameter_bound_a_central_figure_diameter_bound_l1736_173680

/-- A convex figure with diameter 1 --/
class ConvexFigure (F : Type*) :=
  (diameter : ℝ)
  (is_one : diameter = 1)

/-- A planar convex figure --/
class PlanarConvexFigure (F : Type*) extends ConvexFigure F

/-- A spatial convex figure --/
class SpatialConvexFigure (F : Type*) extends ConvexFigure F

/-- The diameter of the central figure L(F) --/
def central_figure_diameter (F : Type*) [ConvexFigure F] : ℝ := sorry

/-- The diameter of the a-central figure II(a, F) --/
def a_central_figure_diameter (F : Type*) [ConvexFigure F] (a : ℝ) : ℝ := sorry

theorem central_figure_diameter_bound 
  (F : Type*) [ConvexFigure F] : 
  (PlanarConvexFigure F → central_figure_diameter F ≤ 1/2) ∧ 
  (SpatialConvexFigure F → central_figure_diameter F ≤ Real.sqrt 2 / 2) := by sorry

theorem a_central_figure_diameter_bound 
  (F : Type*) [ConvexFigure F] (a : ℝ) : 
  (PlanarConvexFigure F → a_central_figure_diameter F a ≤ 1 - a^2/2) ∧ 
  (SpatialConvexFigure F → a_central_figure_diameter F a ≤ Real.sqrt (1 - a^2/2)) := by sorry

end NUMINAMATH_CALUDE_central_figure_diameter_bound_a_central_figure_diameter_bound_l1736_173680


namespace NUMINAMATH_CALUDE_cows_husk_consumption_l1736_173681

/-- The number of bags of husk eaten by a group of cows in a given time period -/
def bags_eaten (num_cows : ℕ) (days : ℕ) (bags_per_cow : ℕ) : ℕ :=
  num_cows * bags_per_cow

/-- Theorem: 26 cows eat 26 bags of husk in 26 days -/
theorem cows_husk_consumption :
  bags_eaten 26 26 1 = 26 := by
  sorry

end NUMINAMATH_CALUDE_cows_husk_consumption_l1736_173681


namespace NUMINAMATH_CALUDE_box_volume_conversion_l1736_173605

/-- Converts cubic feet to cubic yards -/
def cubic_feet_to_cubic_yards (cubic_feet : ℚ) : ℚ :=
  cubic_feet / 27

theorem box_volume_conversion :
  let box_volume_cubic_feet : ℚ := 200
  let box_volume_cubic_yards : ℚ := cubic_feet_to_cubic_yards box_volume_cubic_feet
  box_volume_cubic_yards = 200 / 27 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_conversion_l1736_173605


namespace NUMINAMATH_CALUDE_line_b_production_l1736_173659

/-- Represents a production line in a factory -/
inductive ProductionLine
| A
| B
| C

/-- Represents the production of a factory with three production lines -/
structure FactoryProduction where
  total : ℕ
  lines : ProductionLine → ℕ
  sum_eq_total : lines ProductionLine.A + lines ProductionLine.B + lines ProductionLine.C = total
  arithmetic_seq : ∃ d : ℤ, 
    (lines ProductionLine.B : ℤ) - (lines ProductionLine.A : ℤ) = d ∧
    (lines ProductionLine.C : ℤ) - (lines ProductionLine.B : ℤ) = d

/-- The theorem stating the production of line B given the conditions -/
theorem line_b_production (fp : FactoryProduction) 
  (h_total : fp.total = 16800) : 
  fp.lines ProductionLine.B = 5600 := by
  sorry

end NUMINAMATH_CALUDE_line_b_production_l1736_173659


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1736_173696

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) : 
  a > 0 → b > 0 → 
  a^2 + b^2 = c^2 → 
  b^2 * c^2 = a^2 * (b^2 + c^2) → 
  c / a = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1736_173696


namespace NUMINAMATH_CALUDE_decimal_point_problem_l1736_173677

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) : 
  x = Real.sqrt 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l1736_173677


namespace NUMINAMATH_CALUDE_pencil_profit_problem_l1736_173670

theorem pencil_profit_problem (total_pencils : ℕ) (buy_price sell_price : ℚ) (profit : ℚ) :
  total_pencils = 2000 →
  buy_price = 8/100 →
  sell_price = 20/100 →
  profit = 160 →
  ∃ (sold_pencils : ℕ), sold_pencils = 1600 ∧ 
    sell_price * sold_pencils = buy_price * total_pencils + profit :=
by sorry

end NUMINAMATH_CALUDE_pencil_profit_problem_l1736_173670


namespace NUMINAMATH_CALUDE_carla_laundry_rate_l1736_173624

/-- Given a total number of laundry pieces and available hours, 
    calculate the number of pieces to be cleaned per hour. -/
def piecesPerHour (totalPieces : ℕ) (availableHours : ℕ) : ℕ :=
  totalPieces / availableHours

theorem carla_laundry_rate :
  piecesPerHour 80 4 = 20 := by
  sorry


end NUMINAMATH_CALUDE_carla_laundry_rate_l1736_173624


namespace NUMINAMATH_CALUDE_line_segment_lattice_points_l1736_173641

/-- The number of lattice points on a line segment --/
def latticePointCount (x1 y1 x2 y2 : Int) : Nat :=
  sorry

/-- Theorem: The line segment from (5, 5) to (65, 290) contains 16 lattice points --/
theorem line_segment_lattice_points :
  latticePointCount 5 5 65 290 = 16 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_lattice_points_l1736_173641


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1736_173698

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (x < 1 ∨ x > 5) → x^2 - 2*(a-2)*x + a > 0) ↔ 
  (1 < a ∧ a ≤ 5) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1736_173698
