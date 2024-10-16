import Mathlib

namespace NUMINAMATH_CALUDE_parabola_line_intersection_l668_66872

/-- Parabola represented by the equation y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Line represented by the equation y = kx - 1 -/
def line (k x y : ℝ) : Prop := y = k*x - 1

/-- Focus of the parabola y² = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- The line passes through the focus of the parabola -/
def line_passes_through_focus (k : ℝ) : Prop :=
  line k (focus.1) (focus.2)

/-- The line intersects the parabola at two points -/
def line_intersects_parabola (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧ 
    line k x₁ y₁ ∧ line k x₂ y₂

theorem parabola_line_intersection 
  (h1 : line_passes_through_focus k)
  (h2 : line_intersects_parabola k) :
  k = 1 ∧ ∃ (x₁ x₂ : ℝ), x₁ + x₂ + 2 = 8 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l668_66872


namespace NUMINAMATH_CALUDE_tims_children_treats_l668_66867

/-- Calculates the total number of treats Tim's children get while trick-or-treating --/
def total_treats (num_children : ℕ) (houses_per_hour : List ℕ) (treats_per_kid : List ℕ) : ℕ :=
  List.sum (List.zipWith (fun h t => h * t * num_children) houses_per_hour treats_per_kid)

/-- Theorem: Tim's children get 237 treats in total --/
theorem tims_children_treats :
  let num_children : ℕ := 3
  let houses_per_hour : List ℕ := [4, 6, 5, 7]
  let treats_per_kid : List ℕ := [3, 4, 3, 4]
  total_treats num_children houses_per_hour treats_per_kid = 237 := by
  sorry

#eval total_treats 3 [4, 6, 5, 7] [3, 4, 3, 4]

end NUMINAMATH_CALUDE_tims_children_treats_l668_66867


namespace NUMINAMATH_CALUDE_integer_quotient_problem_l668_66896

theorem integer_quotient_problem (x y : ℤ) :
  1996 * x + y / 96 = x + y →
  x / y = 1 / 2016 ∨ y / x = 2016 := by
sorry

end NUMINAMATH_CALUDE_integer_quotient_problem_l668_66896


namespace NUMINAMATH_CALUDE_smallest_positive_period_l668_66854

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem smallest_positive_period (f : ℝ → ℝ) 
  (h : ∀ x, f (3 * x) = f (3 * x - 3/2)) :
  ∃ T, T = 1/2 ∧ is_periodic f T ∧ ∀ T' > 0, is_periodic f T' → T ≤ T' :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_period_l668_66854


namespace NUMINAMATH_CALUDE_petty_cash_for_support_staff_bonus_l668_66889

/-- Represents the number of staff members in each category -/
structure StaffCount where
  total : Nat
  administrative : Nat
  junior : Nat
  support : Nat

/-- Represents the daily bonus amounts for each staff category -/
structure DailyBonus where
  administrative : Nat
  junior : Nat
  support : Nat

/-- Represents the financial details of the bonus distribution -/
structure BonusDistribution where
  staff : StaffCount
  daily_bonus : DailyBonus
  bonus_days : Nat
  accountant_amount : Nat
  petty_cash_budget : Nat

/-- Calculates the amount needed from petty cash for support staff bonuses -/
def petty_cash_needed (bd : BonusDistribution) : Nat :=
  let total_bonus := bd.staff.administrative * bd.daily_bonus.administrative * bd.bonus_days +
                     bd.staff.junior * bd.daily_bonus.junior * bd.bonus_days +
                     bd.staff.support * bd.daily_bonus.support * bd.bonus_days
  total_bonus - bd.accountant_amount

/-- Theorem stating the amount needed from petty cash for support staff bonuses -/
theorem petty_cash_for_support_staff_bonus 
  (bd : BonusDistribution) 
  (h1 : bd.staff.total = 30)
  (h2 : bd.staff.administrative = 10)
  (h3 : bd.staff.junior = 10)
  (h4 : bd.staff.support = 10)
  (h5 : bd.daily_bonus.administrative = 100)
  (h6 : bd.daily_bonus.junior = 120)
  (h7 : bd.daily_bonus.support = 80)
  (h8 : bd.bonus_days = 30)
  (h9 : bd.accountant_amount = 85000)
  (h10 : bd.petty_cash_budget = 25000) :
  petty_cash_needed bd = 5000 := by
  sorry

end NUMINAMATH_CALUDE_petty_cash_for_support_staff_bonus_l668_66889


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l668_66802

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 60 → b = 80 → c^2 = a^2 + b^2 → c = 100 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l668_66802


namespace NUMINAMATH_CALUDE_subtraction_sum_l668_66818

/-- Given a subtraction problem with digits K, L, M, and N, prove that their sum is 20 -/
theorem subtraction_sum (K L M N : Nat) : 
  (K < 10) → (L < 10) → (M < 10) → (N < 10) →
  (5000 + 100 * K + 30 + L) - (1000 * M + 400 + 10 * N + 1) = 4451 →
  K + L + M + N = 20 := by
sorry

end NUMINAMATH_CALUDE_subtraction_sum_l668_66818


namespace NUMINAMATH_CALUDE_total_water_flow_l668_66862

def water_flow_rate : ℚ := 2 + 2/3
def time_period : ℕ := 9

theorem total_water_flow (rate : ℚ) (time : ℕ) (h1 : rate = water_flow_rate) (h2 : time = time_period) :
  rate * time = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_water_flow_l668_66862


namespace NUMINAMATH_CALUDE_hyperbola_intersection_l668_66870

/-- Hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_real_axis : a = 1
  h_focus : a^2 + b^2 = 5

/-- Line intersecting the hyperbola -/
def intersecting_line (x : ℝ) : ℝ := x + 2

/-- Theorem about the hyperbola and its intersecting line -/
theorem hyperbola_intersection (C : Hyperbola) :
  (∀ x y, x^2 / C.a^2 - y^2 / C.b^2 = 1 ↔ x^2 - y^2 / 4 = 1) ∧
  (∃ A B : ℝ × ℝ,
    A ≠ B ∧
    (A.1^2 - A.2^2 / 4 = 1) ∧
    (B.1^2 - B.2^2 / 4 = 1) ∧
    (A.2 = intersecting_line A.1) ∧
    (B.2 = intersecting_line B.1) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 32 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_l668_66870


namespace NUMINAMATH_CALUDE_triangle_inequality_expression_l668_66803

theorem triangle_inequality_expression (a b c : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) → 
  (a + b > c) → (b + c > a) → (c + a > b) →
  (a^2 + b^2 - c^2 - 2*a*b < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_expression_l668_66803


namespace NUMINAMATH_CALUDE_first_term_of_ap_l668_66846

/-- 
Given an arithmetic progression where:
- The 10th term is 26
- The common difference is 2

Prove that the first term is 8
-/
theorem first_term_of_ap (a : ℝ) : 
  (∃ (d : ℝ), d = 2 ∧ a + 9 * d = 26) → a = 8 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_ap_l668_66846


namespace NUMINAMATH_CALUDE_max_abs_z_on_line_segment_l668_66857

theorem max_abs_z_on_line_segment (z : ℂ) :
  Complex.abs (z - 6*I) + Complex.abs (z - 5) = Real.sqrt 61 →
  Complex.abs z ≤ 6 ∧ ∃ w : ℂ, Complex.abs (w - 6*I) + Complex.abs (w - 5) = Real.sqrt 61 ∧ Complex.abs w = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_z_on_line_segment_l668_66857


namespace NUMINAMATH_CALUDE_max_marble_diff_is_six_l668_66835

/-- Represents a basket of marbles -/
structure Basket where
  color1 : String
  count1 : Nat
  color2 : String
  count2 : Nat

/-- Calculates the absolute difference between two numbers -/
def absDiff (a b : Nat) : Nat :=
  if a ≥ b then a - b else b - a

/-- Theorem: The maximum difference between marble counts in any basket is 6 -/
theorem max_marble_diff_is_six (basketA basketB basketC : Basket)
  (hA : basketA = { color1 := "red", count1 := 4, color2 := "yellow", count2 := 2 })
  (hB : basketB = { color1 := "green", count1 := 6, color2 := "yellow", count2 := 1 })
  (hC : basketC = { color1 := "white", count1 := 3, color2 := "yellow", count2 := 9 }) :
  max (absDiff basketA.count1 basketA.count2)
      (max (absDiff basketB.count1 basketB.count2)
           (absDiff basketC.count1 basketC.count2)) = 6 := by
  sorry


end NUMINAMATH_CALUDE_max_marble_diff_is_six_l668_66835


namespace NUMINAMATH_CALUDE_final_price_approx_l668_66894

/-- The final price after applying two successive discounts to a list price. -/
def final_price (list_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  list_price * (1 - discount1) * (1 - discount2)

/-- Theorem stating that the final price after discounts is approximately 57.33 -/
theorem final_price_approx :
  let list_price : ℝ := 65
  let discount1 : ℝ := 0.1  -- 10%
  let discount2 : ℝ := 0.020000000000000027  -- 2.0000000000000027%
  abs (final_price list_price discount1 discount2 - 57.33) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_final_price_approx_l668_66894


namespace NUMINAMATH_CALUDE_m_range_l668_66899

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x - 3) * (x + 1) > 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 > 0

-- Define the set of x values satisfying p
def A : Set ℝ := {x | p x}

-- Define the set of x values satisfying q
def B (m : ℝ) : Set ℝ := {x | q x m}

-- State the theorem
theorem m_range :
  (∀ m : ℝ, m > 0 → (A ⊂ B m) ∧ (A ≠ B m)) →
  {m : ℝ | 0 < m ∧ m ≤ 2} = {m : ℝ | ∃ x, q x m ∧ ¬p x} :=
sorry

end NUMINAMATH_CALUDE_m_range_l668_66899


namespace NUMINAMATH_CALUDE_three_men_five_jobs_earnings_l668_66822

/-- Calculates the total earnings for a group of workers completing multiple jobs -/
def totalEarnings (numWorkers : ℕ) (numJobs : ℕ) (hourlyRate : ℕ) (hoursPerJob : ℕ) : ℕ :=
  numWorkers * numJobs * hourlyRate * hoursPerJob

/-- Proves that 3 men working on 5 jobs at $10 per hour, with each job taking 1 hour, earn $150 in total -/
theorem three_men_five_jobs_earnings :
  totalEarnings 3 5 10 1 = 150 := by
  sorry

end NUMINAMATH_CALUDE_three_men_five_jobs_earnings_l668_66822


namespace NUMINAMATH_CALUDE_product_ratio_theorem_l668_66856

theorem product_ratio_theorem (a b c d e f : ℝ) 
  (h1 : a * b * c = 65)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250)
  : (a * f) / (c * d) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_product_ratio_theorem_l668_66856


namespace NUMINAMATH_CALUDE_tip_amount_is_36_dollars_l668_66837

/-- The cost of a woman's haircut in dollars -/
def womens_haircut_cost : ℝ := 48

/-- The cost of a child's haircut in dollars -/
def childs_haircut_cost : ℝ := 36

/-- The cost of a teenager's haircut in dollars -/
def teens_haircut_cost : ℝ := 40

/-- The cost of Tayzia's hair treatment in dollars -/
def hair_treatment_cost : ℝ := 20

/-- The tip percentage as a decimal -/
def tip_percentage : ℝ := 0.20

/-- The total cost of haircuts and treatment before tip -/
def total_cost : ℝ :=
  womens_haircut_cost + 2 * childs_haircut_cost + teens_haircut_cost + hair_treatment_cost

/-- The theorem stating that the 20% tip is $36 -/
theorem tip_amount_is_36_dollars : tip_percentage * total_cost = 36 := by
  sorry

end NUMINAMATH_CALUDE_tip_amount_is_36_dollars_l668_66837


namespace NUMINAMATH_CALUDE_firefighters_total_fires_l668_66825

/-- The number of fires put out by three firefighters -/
def total_fires (doug_fires : ℕ) (kai_multiplier : ℕ) (eli_divisor : ℕ) : ℕ :=
  doug_fires + (doug_fires * kai_multiplier) + (doug_fires * kai_multiplier / eli_divisor)

/-- Theorem stating the total number of fires put out by Doug, Kai, and Eli -/
theorem firefighters_total_fires :
  total_fires 20 3 2 = 110 := by
  sorry

#eval total_fires 20 3 2

end NUMINAMATH_CALUDE_firefighters_total_fires_l668_66825


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_plane_l668_66821

/-- Two lines are parallel -/
def parallel (m n : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular (l : Line) (p : Plane) : Prop := sorry

/-- Two geometric objects are different -/
def different (a b : α) : Prop := a ≠ b

theorem line_parallel_perpendicular_plane 
  (m n : Line) (α : Plane) 
  (h1 : different m n) 
  (h2 : parallel m n) 
  (h3 : perpendicular n α) : 
  perpendicular m α := by sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_plane_l668_66821


namespace NUMINAMATH_CALUDE_raisin_count_l668_66836

theorem raisin_count (total_raisins : ℕ) (total_boxes : ℕ) (second_box : ℕ) (other_boxes : ℕ) (other_box_count : ℕ) :
  total_raisins = 437 →
  total_boxes = 5 →
  second_box = 74 →
  other_boxes = 97 →
  other_box_count = 3 →
  ∃ (first_box : ℕ), first_box = total_raisins - (second_box + other_box_count * other_boxes) ∧ first_box = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_raisin_count_l668_66836


namespace NUMINAMATH_CALUDE_factorization_2m_squared_minus_18_l668_66841

theorem factorization_2m_squared_minus_18 (m : ℝ) : 2 * m^2 - 18 = 2 * (m + 3) * (m - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_2m_squared_minus_18_l668_66841


namespace NUMINAMATH_CALUDE_perpendicular_parallel_theorem_l668_66801

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_theorem 
  (m n l : Line) (α β γ : Plane) 
  (h1 : m ≠ n ∧ m ≠ l ∧ n ≠ l) 
  (h2 : α ≠ β ∧ α ≠ γ ∧ β ≠ γ) :
  perpendicularLP m α → parallelLP n β → parallelPP α β → perpendicular m n :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_theorem_l668_66801


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l668_66853

def f (x a : ℝ) : ℝ := |2 * x - a| + a

theorem solution_set_implies_a_value :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 ↔ f x 1 ≤ 6) →
  (∀ x : ℝ, f x 1 ≤ 6 → -2 ≤ x ∧ x ≤ 3) →
  (∃ a : ℝ, ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 ↔ f x a ≤ 6) →
  (∃! a : ℝ, ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 ↔ f x a ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l668_66853


namespace NUMINAMATH_CALUDE_trapezoid_isosceles_and_diagonal_l668_66882

-- Define the trapezoid
structure Trapezoid :=
  (AB CD AD BC : ℝ)
  (parallel : AB ≠ CD)
  (ab_length : AB = 25)
  (cd_length : CD = 13)
  (ad_length : AD = 15)
  (bc_length : BC = 17)

-- Define an isosceles trapezoid
def IsoscelesTrapezoid (t : Trapezoid) : Prop :=
  ∃ (h : ℝ), h > 0 ∧ 
  (t.AD)^2 = h^2 + ((t.AB - t.CD) / 2)^2 ∧
  (t.BC)^2 = h^2 + ((t.AB - t.CD) / 2)^2

-- State the theorem
theorem trapezoid_isosceles_and_diagonal (t : Trapezoid) : 
  IsoscelesTrapezoid t ∧ ∃ (AC : ℝ), AC = Real.sqrt 524 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_isosceles_and_diagonal_l668_66882


namespace NUMINAMATH_CALUDE_min_value_of_expression_l668_66849

theorem min_value_of_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (4 - n, 2)
  (∃ (k : ℝ), a = k • b) → 
  (∀ (x y : ℝ), x > 0 → y > 0 → (1 / x + 8 / y ≥ 9 / 2) ∧ 
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1 / x₀ + 8 / y₀ = 9 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l668_66849


namespace NUMINAMATH_CALUDE_children_getting_on_bus_l668_66827

theorem children_getting_on_bus (initial : ℝ) (got_off : ℝ) (final : ℝ) 
  (h1 : initial = 42.5)
  (h2 : got_off = 21.3)
  (h3 : final = 35.8) :
  final - (initial - got_off) = 14.6 := by
  sorry

end NUMINAMATH_CALUDE_children_getting_on_bus_l668_66827


namespace NUMINAMATH_CALUDE_section_area_is_28_sqrt_34_l668_66819

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  edge_length : ℝ
  origin : Point3D

/-- Calculates the area of the section cut by a plane in a cube -/
noncomputable def sectionArea (cube : Cube) (plane : Plane) : ℝ :=
  sorry

/-- Theorem: The area of the section cut by plane α in the given cube is 28√34 -/
theorem section_area_is_28_sqrt_34 :
  let cube : Cube := { edge_length := 12, origin := { x := 0, y := 0, z := 0 } }
  let A : Point3D := cube.origin
  let E : Point3D := { x := 12, y := 0, z := 9 }
  let F : Point3D := { x := 0, y := 12, z := 9 }
  let plane : Plane := { a := 1, b := 1, c := -3/4, d := 0 }
  sectionArea cube plane = 28 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_section_area_is_28_sqrt_34_l668_66819


namespace NUMINAMATH_CALUDE_speeding_ticket_percentage_l668_66830

/-- Proves that 20% of motorists who exceed the speed limit do not receive tickets -/
theorem speeding_ticket_percentage (M : ℝ) (h1 : M > 0) : 
  let exceed_limit := 0.125 * M
  let receive_ticket := 0.1 * M
  (exceed_limit - receive_ticket) / exceed_limit = 0.2 := by
sorry

end NUMINAMATH_CALUDE_speeding_ticket_percentage_l668_66830


namespace NUMINAMATH_CALUDE_smallest_weight_set_has_11_weights_l668_66839

/-- A set of weights that can be divided into equal piles -/
structure WeightSet where
  weights : List ℕ
  divisible_by_4 : ∃ (n : ℕ), 4 * n = weights.sum
  divisible_by_5 : ∃ (n : ℕ), 5 * n = weights.sum
  divisible_by_6 : ∃ (n : ℕ), 6 * n = weights.sum

/-- The property of being the smallest set of weights divisible by 4, 5, and 6 -/
def is_smallest_weight_set (ws : WeightSet) : Prop :=
  ∀ (other : WeightSet), other.weights.length ≥ ws.weights.length

/-- The theorem stating that 11 is the smallest number of weights divisible by 4, 5, and 6 -/
theorem smallest_weight_set_has_11_weights :
  ∃ (ws : WeightSet), ws.weights.length = 11 ∧ is_smallest_weight_set ws :=
sorry

end NUMINAMATH_CALUDE_smallest_weight_set_has_11_weights_l668_66839


namespace NUMINAMATH_CALUDE_quadratic_points_order_l668_66845

/-- Given a quadratic function f(x) = x² + x - 1, prove that y₂ < y₁ < y₃ 
    where y₁, y₂, and y₃ are the y-coordinates of points on the graph of f 
    with x-coordinates -2, 0, and 2 respectively. -/
theorem quadratic_points_order : 
  let f : ℝ → ℝ := λ x ↦ x^2 + x - 1
  let y₁ : ℝ := f (-2)
  let y₂ : ℝ := f 0
  let y₃ : ℝ := f 2
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_points_order_l668_66845


namespace NUMINAMATH_CALUDE_davids_trip_money_l668_66885

theorem davids_trip_money (initial_amount spent_amount remaining_amount : ℕ) :
  remaining_amount = 500 →
  remaining_amount = spent_amount - 500 →
  initial_amount = spent_amount + remaining_amount →
  initial_amount = 1500 := by
sorry

end NUMINAMATH_CALUDE_davids_trip_money_l668_66885


namespace NUMINAMATH_CALUDE_max_sequence_sum_l668_66866

def arithmetic_sequence (n : ℕ) : ℚ := 5 - (5/7) * (n - 1)

def sequence_sum (n : ℕ) : ℚ := n * (2 * 5 + (n - 1) * (-5/7)) / 2

theorem max_sequence_sum :
  (∃ n : ℕ, sequence_sum n = 20) ∧
  (∀ m : ℕ, sequence_sum m ≤ 20) ∧
  (∀ n : ℕ, sequence_sum n = 20 → (n = 7 ∨ n = 8)) :=
sorry

end NUMINAMATH_CALUDE_max_sequence_sum_l668_66866


namespace NUMINAMATH_CALUDE_smallest_base_not_divisible_by_five_l668_66863

theorem smallest_base_not_divisible_by_five : 
  ∃ (b : ℕ), b > 2 ∧ b = 6 ∧ ¬(5 ∣ (2 * b^3 - 1)) ∧
  ∀ (k : ℕ), 2 < k ∧ k < b → (5 ∣ (2 * k^3 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_not_divisible_by_five_l668_66863


namespace NUMINAMATH_CALUDE_inequality_solution_set_l668_66888

theorem inequality_solution_set : 
  ∀ x : ℝ, -x^2 - 3*x + 4 > 0 ↔ x ∈ Set.Ioo (-4) 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l668_66888


namespace NUMINAMATH_CALUDE_reena_loan_interest_l668_66852

/-- Calculate simple interest for a loan where the loan period in years equals the interest rate -/
def simple_interest (principal : ℚ) (rate : ℚ) : ℚ :=
  principal * rate * rate / 100

theorem reena_loan_interest :
  let principal : ℚ := 1200
  let rate : ℚ := 4
  simple_interest principal rate = 192 := by
sorry

end NUMINAMATH_CALUDE_reena_loan_interest_l668_66852


namespace NUMINAMATH_CALUDE_max_product_of_functions_l668_66893

/-- Given functions f and g on ℝ with specified ranges, prove that the maximum value of their product is 10 -/
theorem max_product_of_functions (f g : ℝ → ℝ) 
  (hf : ∀ x, f x ∈ Set.Icc (-5) 3) 
  (hg : ∀ x, g x ∈ Set.Icc (-2) 1) : 
  (∃ x, f x * g x = 10) ∧ (∀ x, f x * g x ≤ 10) := by
  sorry

#check max_product_of_functions

end NUMINAMATH_CALUDE_max_product_of_functions_l668_66893


namespace NUMINAMATH_CALUDE_variance_best_stability_measure_l668_66884

/-- A performance measure is a function that takes a list of real numbers (representing performance data) and returns a real number. -/
def PerformanceMeasure := (List ℝ) → ℝ

/-- Average of a list of real numbers -/
def average : PerformanceMeasure := sorry

/-- Median of a list of real numbers -/
def median : PerformanceMeasure := sorry

/-- Mode of a list of real numbers -/
def mode : PerformanceMeasure := sorry

/-- Variance of a list of real numbers -/
def variance : PerformanceMeasure := sorry

/-- A measure is considered stable if it reflects the spread of the data -/
def reflectsSpread (m : PerformanceMeasure) : Prop := sorry

/-- Theorem stating that variance is the measure that best reflects the stability of performance -/
theorem variance_best_stability_measure :
  reflectsSpread variance ∧
  ¬reflectsSpread average ∧
  ¬reflectsSpread median ∧
  ¬reflectsSpread mode :=
sorry

end NUMINAMATH_CALUDE_variance_best_stability_measure_l668_66884


namespace NUMINAMATH_CALUDE_most_likely_top_quality_count_l668_66864

/-- The proportion of top-quality products -/
def p : ℝ := 0.31

/-- The number of products in the batch -/
def n : ℕ := 75

/-- The most likely number of top-quality products in the batch -/
def most_likely_count : ℕ := 23

/-- Theorem stating that the most likely number of top-quality products in the batch is 23 -/
theorem most_likely_top_quality_count :
  ⌊n * p⌋ = most_likely_count ∧
  (n * p - (1 - p) ≤ most_likely_count) ∧
  (most_likely_count ≤ n * p + p) :=
sorry

end NUMINAMATH_CALUDE_most_likely_top_quality_count_l668_66864


namespace NUMINAMATH_CALUDE_lowest_cost_option_c_l668_66865

/-- Represents a shipping option with a flat fee and per-pound rate -/
structure ShippingOption where
  flatFee : ℝ
  perPoundRate : ℝ

/-- Calculates the total cost for a given shipping option and weight -/
def totalCost (option : ShippingOption) (weight : ℝ) : ℝ :=
  option.flatFee + option.perPoundRate * weight

/-- The three shipping options available -/
def optionA : ShippingOption := ⟨5.00, 0.80⟩
def optionB : ShippingOption := ⟨4.50, 0.85⟩
def optionC : ShippingOption := ⟨3.00, 0.95⟩

/-- The weight of the package in pounds -/
def packageWeight : ℝ := 5

theorem lowest_cost_option_c :
  let costA := totalCost optionA packageWeight
  let costB := totalCost optionB packageWeight
  let costC := totalCost optionC packageWeight
  (costC < costA ∧ costC < costB) ∧ costC = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_lowest_cost_option_c_l668_66865


namespace NUMINAMATH_CALUDE_no_odd_solution_l668_66892

theorem no_odd_solution :
  ¬∃ (a b c d e f : ℕ), 
    Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ Odd e ∧ Odd f ∧
    (1 / a + 1 / b + 1 / c + 1 / d + 1 / e + 1 / f : ℚ) = 1 :=
by sorry

end NUMINAMATH_CALUDE_no_odd_solution_l668_66892


namespace NUMINAMATH_CALUDE_intersection_has_one_element_l668_66844

/-- The set A in ℝ² defined by the equation x^2 - 3xy + 4y^2 = 7/2 -/
def A : Set (ℝ × ℝ) := {p | p.1^2 - 3*p.1*p.2 + 4*p.2^2 = 7/2}

/-- The set B in ℝ² defined by the equation kx + y = 2, where k > 0 -/
def B (k : ℝ) : Set (ℝ × ℝ) := {p | k*p.1 + p.2 = 2}

/-- The theorem stating that when k = 1/4, the intersection of A and B has exactly one element -/
theorem intersection_has_one_element :
  ∃! p : ℝ × ℝ, p ∈ A ∩ B (1/4) :=
sorry

end NUMINAMATH_CALUDE_intersection_has_one_element_l668_66844


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l668_66816

theorem arctan_equation_solution :
  ∃ x : ℝ, 2 * Real.arctan (1/3) + Real.arctan (1/5) + Real.arctan (1/x) = π/4 ∧ x = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l668_66816


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l668_66817

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l668_66817


namespace NUMINAMATH_CALUDE_final_number_theorem_l668_66897

/-- Represents the state of the number on the board -/
structure BoardState where
  digits : List Nat
  deriving Repr

/-- Applies Operation 1 to the board state -/
def applyOperation1 (state : BoardState) : BoardState :=
  sorry

/-- Applies Operation 2 to the board state -/
def applyOperation2 (state : BoardState) : BoardState :=
  sorry

/-- Checks if a number is a valid final state (two digits) -/
def isValidFinalState (state : BoardState) : Bool :=
  sorry

/-- Generates the initial state with 100 fives -/
def initialState : BoardState :=
  { digits := List.replicate 100 5 }

/-- Theorem stating the final result of the operations -/
theorem final_number_theorem :
  ∃ (finalState : BoardState),
    (isValidFinalState finalState) ∧
    (finalState.digits = [8, 0] ∨ finalState.digits = [6, 6]) ∧
    (∃ (operations : List (BoardState → BoardState)),
      operations.foldl (λ state op => op state) initialState = finalState) :=
sorry

end NUMINAMATH_CALUDE_final_number_theorem_l668_66897


namespace NUMINAMATH_CALUDE_remainder_problem_l668_66876

theorem remainder_problem (n : ℕ) : n % 13 = 11 → n = 349 → n % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l668_66876


namespace NUMINAMATH_CALUDE_area_of_triangle_MEF_l668_66800

-- Define the circle P
def circle_P : Real := 10

-- Define the chord EF
def chord_EF : Real := 12

-- Define the segment MQ
def segment_MQ : Real := 20

-- Define the perpendicular distance from P to EF
def perpendicular_distance : Real := 8

-- Theorem statement
theorem area_of_triangle_MEF :
  let radius : Real := circle_P
  let chord_length : Real := chord_EF
  let height : Real := perpendicular_distance
  (1/2 : Real) * chord_length * height = 48 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_MEF_l668_66800


namespace NUMINAMATH_CALUDE_fraction_sum_difference_l668_66883

theorem fraction_sum_difference : 2/5 + 3/8 - 1/10 = 27/40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_difference_l668_66883


namespace NUMINAMATH_CALUDE_sphere_center_reciprocal_sum_l668_66833

/-- Given a sphere with center (p,q,r) passing through the origin and three points on the axes,
    prove that the sum of reciprocals of its center coordinates equals 49/72 -/
theorem sphere_center_reciprocal_sum :
  ∀ (p q r : ℝ),
  (p^2 + q^2 + r^2 = p^2 + q^2 + r^2) ∧  -- Distance from center to origin
  (p^2 + q^2 + r^2 = (p-2)^2 + q^2 + r^2) ∧  -- Distance from center to (2,0,0)
  (p^2 + q^2 + r^2 = p^2 + (q-4)^2 + r^2) ∧  -- Distance from center to (0,4,0)
  (p^2 + q^2 + r^2 = p^2 + q^2 + (r-6)^2) →  -- Distance from center to (0,0,6)
  1/p + 1/q + 1/r = 49/72 := by
sorry

end NUMINAMATH_CALUDE_sphere_center_reciprocal_sum_l668_66833


namespace NUMINAMATH_CALUDE_problem_solid_surface_area_l668_66880

/-- Represents a 3D solid composed of unit cubes -/
structure CubeSolid where
  cubes : ℕ
  top_layer : ℕ
  bottom_layer : ℕ
  height : ℕ
  length : ℕ
  depth : ℕ

/-- Calculates the surface area of a CubeSolid -/
def surface_area (s : CubeSolid) : ℕ := sorry

/-- The specific solid described in the problem -/
def problem_solid : CubeSolid :=
  { cubes := 15
  , top_layer := 5
  , bottom_layer := 5
  , height := 3
  , length := 5
  , depth := 3 }

/-- Theorem stating that the surface area of the problem_solid is 26 square units -/
theorem problem_solid_surface_area :
  surface_area problem_solid = 26 := by sorry

end NUMINAMATH_CALUDE_problem_solid_surface_area_l668_66880


namespace NUMINAMATH_CALUDE_converse_of_proposition_l668_66895

theorem converse_of_proposition (a b : ℝ) : 
  (∀ x y : ℝ, x ≥ y → x^3 ≥ y^3) → 
  (∀ x y : ℝ, x^3 ≥ y^3 → x ≥ y) :=
sorry

end NUMINAMATH_CALUDE_converse_of_proposition_l668_66895


namespace NUMINAMATH_CALUDE_doughnuts_distribution_l668_66848

theorem doughnuts_distribution (total_doughnuts : ℕ) (total_boxes : ℕ) (first_two_boxes : ℕ) (doughnuts_per_first_two : ℕ) :
  total_doughnuts = 72 →
  total_boxes = 6 →
  first_two_boxes = 2 →
  doughnuts_per_first_two = 12 →
  (total_doughnuts - first_two_boxes * doughnuts_per_first_two) % (total_boxes - first_two_boxes) = 0 →
  (total_doughnuts - first_two_boxes * doughnuts_per_first_two) / (total_boxes - first_two_boxes) = 12 :=
by sorry

end NUMINAMATH_CALUDE_doughnuts_distribution_l668_66848


namespace NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l668_66869

theorem unique_solution_ceiling_equation :
  ∃! c : ℝ, c + ⌈c⌉ = 23.2 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l668_66869


namespace NUMINAMATH_CALUDE_expression_evaluation_l668_66814

theorem expression_evaluation (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(2*y) * y^(3*x)) / (y^(2*y) * x^(3*x)) = (x/y)^(2*y - 3*x) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l668_66814


namespace NUMINAMATH_CALUDE_parabola_line_intersections_l668_66850

theorem parabola_line_intersections (a b c : ℝ) (ha : a ≠ 0) :
  (∀ (x y : ℝ), (y = a * x^2 + b * x + c ∧ y = a * x + b) → 
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y)) ∧
  (∀ (x y : ℝ), (y = a * x^2 + b * x + c ∧ y = b * x + c) → 
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y)) ∧
  (∀ (x y : ℝ), (y = a * x^2 + b * x + c ∧ y = c * x + a) → 
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y)) ∧
  (∀ (x y : ℝ), (y = a * x^2 + b * x + c ∧ y = b * x + a) → 
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y)) ∧
  (∀ (x y : ℝ), (y = a * x^2 + b * x + c ∧ y = c * x + b) → 
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y)) ∧
  (∀ (x y : ℝ), (y = a * x^2 + b * x + c ∧ y = a * x + c) → 
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y)) →
  1 ≤ c / a ∧ c / a ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_parabola_line_intersections_l668_66850


namespace NUMINAMATH_CALUDE_number_equals_scientific_notation_l668_66826

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be represented in scientific notation -/
def number : ℕ := 858000000

/-- The scientific notation representation of the number -/
def scientific_representation : ScientificNotation := {
  coefficient := 8.58
  exponent := 8
  is_valid := by sorry
}

/-- Theorem stating that the number is equal to its scientific notation representation -/
theorem number_equals_scientific_notation : 
  (scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent) = number := by
  sorry

end NUMINAMATH_CALUDE_number_equals_scientific_notation_l668_66826


namespace NUMINAMATH_CALUDE_tan_alpha_minus_pi_sixth_l668_66871

theorem tan_alpha_minus_pi_sixth (α : Real) : 
  (∃ (x y : Real), x = -Real.sqrt 3 ∧ y = 2 ∧ 
   Real.tan α = y / x) →
  Real.tan (α - π/6) = -3 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_pi_sixth_l668_66871


namespace NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l668_66847

/-- A quadratic function f(x) = ax^2 + bx + c with roots at -2 and 4, and maximum value 54 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum_of_coefficients 
  (a b c : ℝ) 
  (h1 : QuadraticFunction a b c (-2) = 0)
  (h2 : QuadraticFunction a b c 4 = 0)
  (h3 : ∀ x, QuadraticFunction a b c x ≤ 54)
  (h4 : ∃ x, QuadraticFunction a b c x = 54) :
  a + b + c = 54 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l668_66847


namespace NUMINAMATH_CALUDE_counting_game_result_l668_66815

/-- Represents the counting game with students in a circle. -/
def CountingGame (n : ℕ) (start : ℕ) (last : ℕ) : Prop :=
  ∃ (process : ℕ → ℕ → ℕ), 
    (process 0 start = last) ∧ 
    (∀ k, k > 0 → process k start ≠ last → 
      process (k+1) start = process k (((process k start + 2) % n) + 1))

/-- The main theorem stating that if student 37 is the last remaining
    in a circle of 40 students, then the initial student was number 5. -/
theorem counting_game_result : CountingGame 40 5 37 := by
  sorry


end NUMINAMATH_CALUDE_counting_game_result_l668_66815


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l668_66807

/-- Given a, b, c form a geometric sequence, the quadratic function f(x) = ax^2 + bx + c has no real roots -/
theorem quadratic_no_real_roots (a b c : ℝ) (h_geo : b^2 = a*c) (h_pos : a*c > 0) :
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l668_66807


namespace NUMINAMATH_CALUDE_rational_function_value_l668_66890

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  linear_p : ∃ (a b : ℝ), ∀ x, p x = a * x + b
  quadratic_q : ∃ (a b c : ℝ), ∀ x, q x = a * x^2 + b * x + c
  asymptote_neg4 : q (-4) = 0
  asymptote_1 : q 1 = 0
  point_0 : p 0 / q 0 = -1
  point_1 : p 1 / q 1 = -2

/-- The main theorem -/
theorem rational_function_value (f : RationalFunction) : f.p 0 / f.q 0 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l668_66890


namespace NUMINAMATH_CALUDE_probability_sum_10_four_dice_l668_66879

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice thrown -/
def numDice : ℕ := 4

/-- The target sum we're looking for -/
def targetSum : ℕ := 10

/-- The total number of possible outcomes when throwing four dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (combinations that sum to 10) -/
def favorableOutcomes : ℕ := 46

/-- The probability of getting a sum of 10 when throwing four 6-sided dice -/
theorem probability_sum_10_four_dice : 
  (favorableOutcomes : ℚ) / totalOutcomes = 23 / 648 := by sorry

end NUMINAMATH_CALUDE_probability_sum_10_four_dice_l668_66879


namespace NUMINAMATH_CALUDE_range_of_x_minus_2y_range_of_2a_plus_3b_l668_66831

-- Problem 1
theorem range_of_x_minus_2y (x y : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) (hy : 0 ≤ y ∧ y ≤ 1) :
  ∃ (z : ℝ), -3 ≤ z ∧ z ≤ 2 ∧ ∃ (x' y' : ℝ), 
    (-1 ≤ x' ∧ x' ≤ 2) ∧ (0 ≤ y' ∧ y' ≤ 1) ∧ z = x' - 2 * y' :=
sorry

-- Problem 2
theorem range_of_2a_plus_3b (a b : ℝ) (hab1 : -1 < a + b ∧ a + b < 3) (hab2 : 2 < a - b ∧ a - b < 4) :
  ∃ (z : ℝ), -9/2 < z ∧ z < 13/2 ∧ ∃ (a' b' : ℝ), 
    (-1 < a' + b' ∧ a' + b' < 3) ∧ (2 < a' - b' ∧ a' - b' < 4) ∧ z = 2 * a' + 3 * b' :=
sorry

end NUMINAMATH_CALUDE_range_of_x_minus_2y_range_of_2a_plus_3b_l668_66831


namespace NUMINAMATH_CALUDE_sequence_not_square_rational_l668_66810

def sequence_a : ℕ → ℚ
  | 0 => 2016
  | (n + 1) => sequence_a n + 2 / sequence_a n

theorem sequence_not_square_rational : ∀ n : ℕ, ∀ q : ℚ, (sequence_a n) ≠ q^2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_not_square_rational_l668_66810


namespace NUMINAMATH_CALUDE_y_worked_days_proof_l668_66813

/-- The number of days x needs to finish the entire work -/
def x_total_days : ℝ := 24

/-- The number of days y needs to finish the entire work -/
def y_total_days : ℝ := 16

/-- The number of days x needs to finish the remaining work after y leaves -/
def x_remaining_days : ℝ := 9

/-- The number of days y worked before leaving the job -/
def y_worked_days : ℝ := 10

theorem y_worked_days_proof :
  y_worked_days * (1 / y_total_days) + x_remaining_days * (1 / x_total_days) = 1 := by
  sorry

end NUMINAMATH_CALUDE_y_worked_days_proof_l668_66813


namespace NUMINAMATH_CALUDE_product_of_powers_plus_one_l668_66842

theorem product_of_powers_plus_one : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) * (3^8 + 1^8) = 21527360 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_plus_one_l668_66842


namespace NUMINAMATH_CALUDE_product_sum_geq_geometric_mean_sum_l668_66804

theorem product_sum_geq_geometric_mean_sum {a b c : ℝ} (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a * b + b * c + c * a ≥ a * Real.sqrt (b * c) + b * Real.sqrt (a * c) + c * Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_product_sum_geq_geometric_mean_sum_l668_66804


namespace NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_neg_seven_l668_66874

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 / b1 = a2 / b2 ∧ a1 / b1 ≠ c1 / c2

/-- Definition of line l1 -/
def l1 (m : ℝ) (x y : ℝ) : Prop :=
  (3 + m) * x + 4 * y = 5 - 3 * m

/-- Definition of line l2 -/
def l2 (m : ℝ) (x y : ℝ) : Prop :=
  2 * x + (5 + m) * y = 8

/-- Theorem: Lines l1 and l2 are parallel if and only if m = -7 -/
theorem lines_parallel_iff_m_eq_neg_seven :
  ∀ m : ℝ, parallel_lines (3 + m) 4 (5 - 3 * m) 2 (5 + m) 8 ↔ m = -7 := by sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_neg_seven_l668_66874


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l668_66829

theorem divisibility_equivalence (m n : ℕ+) : 
  (19 ∣ (11 * m.val + 2 * n.val)) ↔ (19 ∣ (18 * m.val + 5 * n.val)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l668_66829


namespace NUMINAMATH_CALUDE_west_60_meters_representation_l668_66851

/-- Represents the direction of movement --/
inductive Direction
  | East
  | West

/-- Represents a movement with direction and distance --/
structure Movement where
  direction : Direction
  distance : ℝ

/-- Converts a movement to its numerical representation --/
def Movement.toNumber (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => -m.distance
  | Direction.West => m.distance

theorem west_60_meters_representation :
  ∀ (m : Movement),
    m.direction = Direction.West ∧
    m.distance = 60 →
    m.toNumber = 60 :=
by sorry

end NUMINAMATH_CALUDE_west_60_meters_representation_l668_66851


namespace NUMINAMATH_CALUDE_triangles_in_ten_point_config_l668_66858

/-- Represents a configuration of points on a circle with chords --/
structure CircleConfiguration where
  numPoints : ℕ
  numChords : ℕ
  numIntersections : ℕ

/-- Calculates the number of triangles formed by chord intersections --/
def numTriangles (config : CircleConfiguration) : ℕ :=
  sorry

/-- The specific configuration for our problem --/
def tenPointConfig : CircleConfiguration :=
  { numPoints := 10
  , numChords := 45
  , numIntersections := 210 }

/-- Theorem stating that the number of triangles in the given configuration is 120 --/
theorem triangles_in_ten_point_config :
  numTriangles tenPointConfig = 120 :=
sorry

end NUMINAMATH_CALUDE_triangles_in_ten_point_config_l668_66858


namespace NUMINAMATH_CALUDE_net_amount_is_2550_l668_66860

/-- Calculates the net amount received from selling puppies given the specified conditions -/
def calculate_net_amount (first_litter : ℕ) (second_litter : ℕ) 
  (first_price : ℕ) (second_price : ℕ) (raising_cost : ℕ) : ℕ :=
  let sold_first := (first_litter * 3) / 4
  let sold_second := (second_litter * 3) / 4
  let revenue := sold_first * first_price + sold_second * second_price
  let expenses := (first_litter + second_litter) * raising_cost
  revenue - expenses

/-- The net amount received from selling puppies under the given conditions is $2550 -/
theorem net_amount_is_2550 : 
  calculate_net_amount 10 12 200 250 50 = 2550 := by
  sorry

end NUMINAMATH_CALUDE_net_amount_is_2550_l668_66860


namespace NUMINAMATH_CALUDE_quadratic_function_passes_through_points_l668_66808

/-- The quadratic function f(x) = x² + 2x - 3 passes through the points (0, -3), (1, 0), and (-3, 0). -/
theorem quadratic_function_passes_through_points :
  let f : ℝ → ℝ := λ x ↦ x^2 + 2*x - 3
  f 0 = -3 ∧ f 1 = 0 ∧ f (-3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_passes_through_points_l668_66808


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l668_66820

theorem green_shirt_pairs (total_students : ℕ) (red_students : ℕ) (green_students : ℕ) 
  (total_pairs : ℕ) (red_pairs : ℕ) :
  total_students = 148 →
  red_students = 65 →
  green_students = 83 →
  total_pairs = 74 →
  red_pairs = 27 →
  red_students + green_students = total_students →
  2 * total_pairs = total_students →
  ∃ (green_pairs : ℕ), green_pairs = 36 ∧ 
    red_pairs + green_pairs + (total_students - 2 * (red_pairs + green_pairs)) / 2 = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l668_66820


namespace NUMINAMATH_CALUDE_function_value_theorem_l668_66878

theorem function_value_theorem (f : ℝ → ℝ) (h : ∀ x, f ((1/2) * x - 1) = 2 * x + 3) :
  f (-3/4) = 4 :=
by sorry

end NUMINAMATH_CALUDE_function_value_theorem_l668_66878


namespace NUMINAMATH_CALUDE_product_of_squares_l668_66812

theorem product_of_squares (a b : ℝ) (h1 : a + b = 21) (h2 : a^2 - b^2 = 45) :
  a^2 * b^2 = 28606956 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_product_of_squares_l668_66812


namespace NUMINAMATH_CALUDE_trig_sum_zero_l668_66898

theorem trig_sum_zero (α β γ : ℝ) : 
  (Real.sin α / (Real.sin (α - β) * Real.sin (α - γ))) +
  (Real.sin β / (Real.sin (β - α) * Real.sin (β - γ))) +
  (Real.sin γ / (Real.sin (γ - α) * Real.sin (γ - β))) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_zero_l668_66898


namespace NUMINAMATH_CALUDE_final_share_is_132_75_l668_66828

/-- Calculates the final amount each person receives after combining and sharing their updated amounts equally. -/
def final_share (emani_initial : ℚ) (howard_difference : ℚ) (jamal_initial : ℚ) 
                (emani_increase : ℚ) (howard_increase : ℚ) (jamal_increase : ℚ) : ℚ :=
  let howard_initial := emani_initial - howard_difference
  let emani_updated := emani_initial * (1 + emani_increase)
  let howard_updated := howard_initial * (1 + howard_increase)
  let jamal_updated := jamal_initial * (1 + jamal_increase)
  let total_updated := emani_updated + howard_updated + jamal_updated
  total_updated / 3

/-- Theorem stating that each person receives $132.75 after combining and sharing their updated amounts equally. -/
theorem final_share_is_132_75 :
  final_share 150 30 75 (20/100) (10/100) (15/100) = 132.75 := by
  sorry

end NUMINAMATH_CALUDE_final_share_is_132_75_l668_66828


namespace NUMINAMATH_CALUDE_favorite_song_not_heard_probability_l668_66838

-- Define the number of songs
def num_songs : ℕ := 10

-- Define the duration of the shortest song (in seconds)
def shortest_song : ℕ := 40

-- Define the increment in duration for each subsequent song (in seconds)
def duration_increment : ℕ := 40

-- Define the duration of the favorite song (in seconds)
def favorite_song_duration : ℕ := 240

-- Define the total playtime we're considering (in seconds)
def total_playtime : ℕ := 300

-- Function to calculate the duration of the nth song
def song_duration (n : ℕ) : ℕ := shortest_song + (n - 1) * duration_increment

-- Theorem stating the probability of not hearing the favorite song in its entirety
theorem favorite_song_not_heard_probability :
  let favorite_song_index : ℕ := (favorite_song_duration - shortest_song) / duration_increment + 1
  (favorite_song_index ≤ num_songs) →
  (∀ n : ℕ, n < favorite_song_index → song_duration n + favorite_song_duration > total_playtime) →
  (num_songs - 1 : ℚ) / num_songs = 9 / 10 :=
by sorry

end NUMINAMATH_CALUDE_favorite_song_not_heard_probability_l668_66838


namespace NUMINAMATH_CALUDE_f_is_odd_and_increasing_l668_66811

-- Define the function
def f (x : ℝ) : ℝ := 3 * x

-- State the theorem
theorem f_is_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ a b, a < b → f a < f b) :=
sorry

end NUMINAMATH_CALUDE_f_is_odd_and_increasing_l668_66811


namespace NUMINAMATH_CALUDE_newborn_count_l668_66875

theorem newborn_count (total_children : ℕ) (toddlers : ℕ) : 
  total_children = 40 →
  toddlers = 6 →
  total_children = 5 * toddlers + toddlers + (total_children - 5 * toddlers - toddlers) →
  (total_children - 5 * toddlers - toddlers) = 4 :=
by sorry

end NUMINAMATH_CALUDE_newborn_count_l668_66875


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l668_66881

theorem exponential_equation_solution :
  ∃ x : ℝ, (9 : ℝ)^x * (9 : ℝ)^x * (9 : ℝ)^x * (9 : ℝ)^x = (81 : ℝ)^6 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l668_66881


namespace NUMINAMATH_CALUDE_prime_equation_unique_solution_l668_66805

theorem prime_equation_unique_solution (p q : ℕ) :
  Prime p ∧ Prime q ∧ p^3 - q^5 = (p + q)^2 ↔ p = 7 ∧ q = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_equation_unique_solution_l668_66805


namespace NUMINAMATH_CALUDE_vector_problem_l668_66873

def a : ℝ × ℝ := (-3, 1)
def b : ℝ × ℝ := (1, -2)
def c : ℝ × ℝ := (1, -1)

def m (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

theorem vector_problem :
  (∃ k : ℝ, (m k).1 * (2 * a.1 - b.1) + (m k).2 * (2 * a.2 - b.2) = 0 ∧ k = 5/3) ∧
  (∃ k : ℝ, ∃ t : ℝ, t ≠ 0 ∧ (m k).1 = t * (k * b.1 + c.1) ∧ (m k).2 = t * (k * b.2 + c.2) ∧ k = -1/3) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l668_66873


namespace NUMINAMATH_CALUDE_marbles_given_to_eric_l668_66861

def marble_redistribution (tyrone_initial : ℕ) (eric_initial : ℕ) (x : ℕ) : Prop :=
  (tyrone_initial - x) = 3 * (eric_initial + x)

theorem marbles_given_to_eric 
  (tyrone_initial : ℕ) (eric_initial : ℕ) (x : ℕ)
  (h1 : tyrone_initial = 120)
  (h2 : eric_initial = 20)
  (h3 : marble_redistribution tyrone_initial eric_initial x) :
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_marbles_given_to_eric_l668_66861


namespace NUMINAMATH_CALUDE_complex_equation_solution_l668_66834

theorem complex_equation_solution (c d : ℂ) (x : ℝ) 
  (h1 : Complex.abs c = 3)
  (h2 : Complex.abs d = 5)
  (h3 : c * d = x - 6 * Complex.I) :
  x = 3 * Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l668_66834


namespace NUMINAMATH_CALUDE_intersection_distance_sum_l668_66809

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 2)^2 + (y - 2)^2 = 2

-- Define point A
def point_A : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem intersection_distance_sum :
  ∃ (P Q : ℝ × ℝ),
    line_l P.1 P.2 ∧ circle_C P.1 P.2 ∧
    line_l Q.1 Q.2 ∧ circle_C Q.1 Q.2 ∧
    P ≠ Q ∧
    Real.sqrt ((P.1 - point_A.1)^2 + (P.2 - point_A.2)^2) +
    Real.sqrt ((Q.1 - point_A.1)^2 + (Q.2 - point_A.2)^2) =
    Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_sum_l668_66809


namespace NUMINAMATH_CALUDE_pentagon_sum_problem_l668_66824

theorem pentagon_sum_problem : ∃ (a b c d e : ℝ),
  a + b = 1 ∧
  b + c = 2 ∧
  c + d = 3 ∧
  d + e = 4 ∧
  e + a = 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_sum_problem_l668_66824


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l668_66859

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ d ∣ n

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def has_tens_digit_2 (n : ℕ) : Prop := n ≥ 20 ∧ n < 30

theorem smallest_two_digit_prime_with_composite_reverse :
  ∃ n : ℕ, is_prime n ∧ 
           is_composite (reverse_digits n) ∧ 
           has_tens_digit_2 n ∧
           (∀ m : ℕ, m < n → ¬(is_prime m ∧ is_composite (reverse_digits m) ∧ has_tens_digit_2 m)) ∧
           n = 23 :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l668_66859


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l668_66886

theorem quadratic_solution_property (a b : ℝ) : 
  (3 * a^2 - 9 * a + 21 = 0) ∧ 
  (3 * b^2 - 9 * b + 21 = 0) →
  (3 * a - 4) * (6 * b - 8) = 50 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l668_66886


namespace NUMINAMATH_CALUDE_segment_division_sum_l668_66877

/-- Given a line segment AB with A = (1, 1) and B = (x, y), and a point C = (2, 4) that divides AB in the ratio 2:1, prove that x + y = 8 -/
theorem segment_division_sum (x y : ℝ) : 
  let A : ℝ × ℝ := (1, 1)
  let B : ℝ × ℝ := (x, y)
  let C : ℝ × ℝ := (2, 4)
  (C.1 - A.1) / (B.1 - C.1) = 2 ∧ 
  (C.2 - A.2) / (B.2 - C.2) = 2 →
  x + y = 8 := by
sorry

end NUMINAMATH_CALUDE_segment_division_sum_l668_66877


namespace NUMINAMATH_CALUDE_unique_solution_l668_66823

-- Define the equation
def equation (x : ℝ) : Prop := Real.rpow (5 - x) (1/3) + Real.sqrt (x + 2) = 3

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, equation x ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l668_66823


namespace NUMINAMATH_CALUDE_fraction_inequality_l668_66832

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l668_66832


namespace NUMINAMATH_CALUDE_calculation_proofs_l668_66840

theorem calculation_proofs :
  (1 - 2^2 / (1/5) * 5 - (-10)^2 - |(-3)| = -123) ∧
  ((-1)^2023 + (-5) * ((-2)^3 + 2) - (-4)^2 / (-1/2) = 61) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l668_66840


namespace NUMINAMATH_CALUDE_geometric_sum_7_terms_l668_66887

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_7_terms :
  let a : ℚ := 1/2
  let r : ℚ := -1/2
  let n : ℕ := 7
  geometric_sum a r n = 129/384 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_7_terms_l668_66887


namespace NUMINAMATH_CALUDE_teacher_worksheets_proof_l668_66843

def calculate_total_worksheets (initial : ℕ) (graded : ℕ) (additional : ℕ) : ℕ :=
  let remaining := initial - graded
  let after_additional := remaining + additional
  after_additional + 2 * after_additional

theorem teacher_worksheets_proof : 
  let initial := 6
  let graded := 4
  let additional := 18
  calculate_total_worksheets initial graded additional = 60 := by
  sorry

end NUMINAMATH_CALUDE_teacher_worksheets_proof_l668_66843


namespace NUMINAMATH_CALUDE_single_root_condition_l668_66891

theorem single_root_condition (n : ℕ) (a : ℝ) (h : n > 1) :
  (∃! x : ℝ, (1 + x)^(1/n : ℝ) + (1 - x)^(1/n : ℝ) = a) ↔ a = 2 := by sorry

end NUMINAMATH_CALUDE_single_root_condition_l668_66891


namespace NUMINAMATH_CALUDE_paper_cutting_l668_66868

def can_obtain (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 1 + 7 * a + 11 * b

theorem paper_cutting :
  (¬ can_obtain 60) ∧
  (∀ n : ℕ, n > 60 → can_obtain n) :=
sorry

end NUMINAMATH_CALUDE_paper_cutting_l668_66868


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l668_66855

def f (x : ℝ) := x^3 + 3*x - 1

theorem root_sum_reciprocal (a b c : ℝ) (m n : ℕ) :
  f a = 0 → f b = 0 → f c = 0 →
  (1 / (a^3 + b^3) + 1 / (b^3 + c^3) + 1 / (c^3 + a^3) : ℝ) = m / n →
  m > 0 → n > 0 →
  Nat.gcd m n = 1 →
  100 * m + n = 3989 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l668_66855


namespace NUMINAMATH_CALUDE_gum_cost_theorem_l668_66806

/-- The price of one piece of gum in cents -/
def price_per_piece : ℕ := 2

/-- The number of pieces of gum being purchased -/
def num_pieces : ℕ := 5000

/-- The discount rate as a decimal -/
def discount_rate : ℚ := 1/20

/-- The minimum number of pieces required for the discount to apply -/
def discount_threshold : ℕ := 4000

/-- Calculates the final cost in dollars after applying the discount if applicable -/
def final_cost : ℚ :=
  let total_cents := price_per_piece * num_pieces
  let discounted_cents := if num_pieces > discount_threshold
                          then total_cents * (1 - discount_rate)
                          else total_cents
  discounted_cents / 100

theorem gum_cost_theorem :
  final_cost = 95 := by sorry

end NUMINAMATH_CALUDE_gum_cost_theorem_l668_66806
