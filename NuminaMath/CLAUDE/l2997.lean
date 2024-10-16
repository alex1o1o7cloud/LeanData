import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_stacks_with_green_3_and_4_l2997_299760

/-- Represents a card with a color and a number -/
structure Card where
  color : String
  number : Nat

/-- Represents a stack of cards -/
structure Stack where
  green : Card
  orange : Option Card

/-- Checks if a stack is valid according to the problem rules -/
def isValidStack (s : Stack) : Bool :=
  match s.orange with
  | none => true
  | some o => s.green.number ≤ o.number

/-- The set of all green cards -/
def greenCards : List Card :=
  [1, 2, 3, 4, 5].map (λ n => ⟨"green", n⟩)

/-- The set of all orange cards -/
def orangeCards : List Card :=
  [2, 3, 4, 5].map (λ n => ⟨"orange", n⟩)

/-- Calculates the sum of numbers in a stack -/
def stackSum (s : Stack) : Nat :=
  s.green.number + match s.orange with
  | none => 0
  | some o => o.number

/-- The main theorem to prove -/
theorem sum_of_stacks_with_green_3_and_4 :
  ∃ (s₁ s₂ : Stack),
    s₁.green.number = 3 ∧
    s₂.green.number = 4 ∧
    s₁.green ∈ greenCards ∧
    s₂.green ∈ greenCards ∧
    (∀ o₁ ∈ s₁.orange, o₁ ∈ orangeCards) ∧
    (∀ o₂ ∈ s₂.orange, o₂ ∈ orangeCards) ∧
    isValidStack s₁ ∧
    isValidStack s₂ ∧
    stackSum s₁ + stackSum s₂ = 14 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_stacks_with_green_3_and_4_l2997_299760


namespace NUMINAMATH_CALUDE_noah_grammy_calls_cost_l2997_299730

/-- Calculates the total cost of Noah's calls to Grammy for a year -/
theorem noah_grammy_calls_cost 
  (weeks_per_year : ℕ) 
  (call_duration_minutes : ℕ) 
  (cost_per_minute : ℚ) 
  (calls_per_week : ℕ) : 
  weeks_per_year = 52 → 
  call_duration_minutes = 30 → 
  cost_per_minute = 5 / 100 → 
  calls_per_week = 1 → 
  (weeks_per_year * calls_per_week * call_duration_minutes * cost_per_minute : ℚ) = 78 := by
  sorry

end NUMINAMATH_CALUDE_noah_grammy_calls_cost_l2997_299730


namespace NUMINAMATH_CALUDE_trivia_team_score_l2997_299766

theorem trivia_team_score (total_members : ℕ) (absent_members : ℕ) (total_score : ℕ) :
  total_members = 5 →
  absent_members = 2 →
  total_score = 18 →
  ∃ (points_per_member : ℕ),
    points_per_member * (total_members - absent_members) = total_score ∧
    points_per_member = 6 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_score_l2997_299766


namespace NUMINAMATH_CALUDE_circle_reflection_translation_l2997_299732

/-- Reflects a point about the line y = x -/
def reflect_about_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- Translates a point vertically by a given amount -/
def translate_vertical (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + dy)

/-- The main theorem -/
theorem circle_reflection_translation (center : ℝ × ℝ) :
  center = (3, -7) →
  (translate_vertical (reflect_about_y_eq_x center) 4) = (-7, 7) := by
  sorry


end NUMINAMATH_CALUDE_circle_reflection_translation_l2997_299732


namespace NUMINAMATH_CALUDE_work_to_pump_oil_horizontal_cylinder_l2997_299739

/-- Work required to pump oil from a horizontal cylindrical tank -/
theorem work_to_pump_oil_horizontal_cylinder 
  (δ : ℝ) -- specific weight of oil
  (H : ℝ) -- length of the cylinder
  (R : ℝ) -- radius of the cylinder
  (h : R > 0) -- assumption that radius is positive
  (h' : H > 0) -- assumption that length is positive
  (h'' : δ > 0) -- assumption that specific weight is positive
  : ∃ (Q : ℝ), Q = π * δ * H * R^3 :=
sorry

end NUMINAMATH_CALUDE_work_to_pump_oil_horizontal_cylinder_l2997_299739


namespace NUMINAMATH_CALUDE_A9_coordinates_l2997_299773

/-- Define a sequence of points in a Cartesian coordinate system -/
def A (n : ℕ) : ℝ × ℝ := (n, n^2)

/-- Theorem: The 9th point in the sequence has coordinates (9, 81) -/
theorem A9_coordinates : A 9 = (9, 81) := by
  sorry

end NUMINAMATH_CALUDE_A9_coordinates_l2997_299773


namespace NUMINAMATH_CALUDE_reposition_convergence_l2997_299797

/-- Reposition transformation function -/
def reposition (n : ℕ) : ℕ :=
  sorry

/-- Theorem: Repeated reposition of a 4-digit number always results in 312 -/
theorem reposition_convergence (n : ℕ) (h : 1000 ≤ n ∧ n ≤ 9999) :
  ∃ k : ℕ, ∀ m : ℕ, m ≥ k → (reposition^[m] n) = 312 :=
sorry

end NUMINAMATH_CALUDE_reposition_convergence_l2997_299797


namespace NUMINAMATH_CALUDE_pie_shop_earnings_l2997_299704

def price_per_slice : ℕ := 3
def slices_per_pie : ℕ := 10
def number_of_pies : ℕ := 6

theorem pie_shop_earnings : 
  price_per_slice * slices_per_pie * number_of_pies = 180 := by
  sorry

end NUMINAMATH_CALUDE_pie_shop_earnings_l2997_299704


namespace NUMINAMATH_CALUDE_parade_probability_l2997_299796

/-- The number of possible permutations of n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The probability of an event occurring, given the number of favorable outcomes and total outcomes -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := 
  if total = 0 then 0 else (favorable : ℚ) / (total : ℚ)

/-- The number of formations in the parade -/
def num_formations : ℕ := 3

/-- The number of favorable outcomes (B passes before both A and C) -/
def favorable_outcomes : ℕ := 2

theorem parade_probability :
  probability favorable_outcomes (factorial num_formations) = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_parade_probability_l2997_299796


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2997_299779

/-- Given an ellipse with equation x^2 / (3-k) + y^2 / (1+k) = 1 and foci on the x-axis,
    the range of k values is (-1, 1) -/
theorem ellipse_k_range (k : ℝ) :
  (∀ x y : ℝ, x^2 / (3-k) + y^2 / (1+k) = 1) →
  (∃ c : ℝ, c > 0 ∧ c^2 = (3-k) - (1+k)) →
  -1 < k ∧ k < 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2997_299779


namespace NUMINAMATH_CALUDE_complement_M_equals_closed_interval_l2997_299751

def U : Set ℝ := Set.univ

def M : Set ℝ := {x : ℝ | x^2 - 2*x > 0}

theorem complement_M_equals_closed_interval :
  (Set.univ \ M) = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_complement_M_equals_closed_interval_l2997_299751


namespace NUMINAMATH_CALUDE_smallest_balanced_number_l2997_299738

/-- A function that returns true if a number is a three-digit number with distinct non-zero digits -/
def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100 ≠ (n / 10) % 10) ∧
  (n / 100 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10) ∧
  (n / 100 ≠ 0) ∧
  ((n / 10) % 10 ≠ 0) ∧
  (n % 10 ≠ 0)

/-- A function that calculates the sum of all two-digit numbers formed from the digits of a three-digit number -/
def sum_of_two_digit_numbers (n : ℕ) : ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  (10 * a + b) + (10 * b + a) + (10 * a + c) + (10 * c + a) + (10 * b + c) + (10 * c + b)

/-- The main theorem stating that 132 is the smallest balanced number -/
theorem smallest_balanced_number :
  is_valid_number 132 ∧
  132 = sum_of_two_digit_numbers 132 ∧
  ∀ n < 132, is_valid_number n → n ≠ sum_of_two_digit_numbers n :=
sorry

end NUMINAMATH_CALUDE_smallest_balanced_number_l2997_299738


namespace NUMINAMATH_CALUDE_square_diagonal_l2997_299708

theorem square_diagonal (A : ℝ) (h : A = 200) : 
  ∃ d : ℝ, d^2 = 2 * A ∧ d = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_l2997_299708


namespace NUMINAMATH_CALUDE_coin_stack_count_l2997_299798

/-- Thickness of a 2p coin in millimeters -/
def thickness_2p : ℚ := 205/100

/-- Thickness of a 10p coin in millimeters -/
def thickness_10p : ℚ := 195/100

/-- Total height of the stack in millimeters -/
def stack_height : ℚ := 19

/-- The number of coins in the stack -/
def total_coins : ℕ := 10

/-- Theorem stating that the total number of coins in a stack of 19 mm height,
    consisting only of 2p and 10p coins, is 10 -/
theorem coin_stack_count :
  ∃ (x y : ℕ), x + y = total_coins ∧
  x * thickness_2p + y * thickness_10p = stack_height :=
sorry

end NUMINAMATH_CALUDE_coin_stack_count_l2997_299798


namespace NUMINAMATH_CALUDE_monic_quartic_value_at_zero_l2997_299759

def is_monic_quartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_value_at_zero 
  (f : ℝ → ℝ) 
  (h_monic : is_monic_quartic f)
  (h_m2 : f (-2) = -4)
  (h_1 : f 1 = -1)
  (h_3 : f 3 = -9)
  (h_5 : f 5 = -25) :
  f 0 = 30 := by
sorry

end NUMINAMATH_CALUDE_monic_quartic_value_at_zero_l2997_299759


namespace NUMINAMATH_CALUDE_zero_not_in_empty_set_l2997_299724

theorem zero_not_in_empty_set : 0 ∉ (∅ : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_zero_not_in_empty_set_l2997_299724


namespace NUMINAMATH_CALUDE_bills_double_pay_threshold_l2997_299740

/-- Proves that Bill starts getting paid double after 40 hours -/
theorem bills_double_pay_threshold (base_rate : ℝ) (double_rate : ℝ) (total_hours : ℝ) (total_pay : ℝ)
  (h1 : base_rate = 20)
  (h2 : double_rate = 2 * base_rate)
  (h3 : total_hours = 50)
  (h4 : total_pay = 1200) :
  ∃ x : ℝ, x = 40 ∧ base_rate * x + double_rate * (total_hours - x) = total_pay :=
by
  sorry

end NUMINAMATH_CALUDE_bills_double_pay_threshold_l2997_299740


namespace NUMINAMATH_CALUDE_not_all_tangents_equal_l2997_299709

/-- A convex quadrilateral where the tangent of one angle is m -/
structure ConvexQuadrilateral (m : ℝ) where
  angles : Fin 4 → ℝ
  sum_360 : angles 0 + angles 1 + angles 2 + angles 3 = 360
  all_positive : ∀ i, 0 < angles i
  all_less_180 : ∀ i, angles i < 180
  one_tangent_m : ∃ i, Real.tan (angles i) = m

/-- Theorem stating that it's impossible for all angles to have tangent m -/
theorem not_all_tangents_equal (m : ℝ) (q : ConvexQuadrilateral m) :
  ¬(∀ i, Real.tan (q.angles i) = m) :=
sorry

end NUMINAMATH_CALUDE_not_all_tangents_equal_l2997_299709


namespace NUMINAMATH_CALUDE_equation_solutions_l2997_299707

theorem equation_solutions :
  (∀ x : ℝ, 25 * x^2 = 36 ↔ x = 6/5 ∨ x = -6/5) ∧
  (∀ x : ℝ, (1/3) * (x + 2)^3 - 9 = 0 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2997_299707


namespace NUMINAMATH_CALUDE_distance_sum_bounded_l2997_299761

/-- The sum of squared distances from a point on an ellipse to four fixed points is bounded -/
theorem distance_sum_bounded (x y : ℝ) :
  (x / 2)^2 + (y / 3)^2 = 1 →
  32 ≤ (x - 1)^2 + (y - Real.sqrt 3)^2 +
       (x + Real.sqrt 3)^2 + (y - 1)^2 +
       (x + 1)^2 + (y + Real.sqrt 3)^2 +
       (x - Real.sqrt 3)^2 + (y + 1)^2 ∧
  (x - 1)^2 + (y - Real.sqrt 3)^2 +
  (x + Real.sqrt 3)^2 + (y - 1)^2 +
  (x + 1)^2 + (y + Real.sqrt 3)^2 +
  (x - Real.sqrt 3)^2 + (y + 1)^2 ≤ 52 := by
  sorry


end NUMINAMATH_CALUDE_distance_sum_bounded_l2997_299761


namespace NUMINAMATH_CALUDE_sqrt_fourteen_times_sqrt_seven_minus_sqrt_two_l2997_299747

theorem sqrt_fourteen_times_sqrt_seven_minus_sqrt_two : 
  Real.sqrt 14 * Real.sqrt 7 - Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fourteen_times_sqrt_seven_minus_sqrt_two_l2997_299747


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l2997_299718

theorem cubic_roots_relation (p q r : ℝ) (u v w : ℝ) : 
  (∀ x : ℝ, x^3 - 6*x^2 + 11*x + 10 = (x - p) * (x - q) * (x - r)) →
  (∀ x : ℝ, x^3 + u*x^2 + v*x + w = (x - (p + q)) * (x - (q + r)) * (x - (r + p))) →
  w = 80 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l2997_299718


namespace NUMINAMATH_CALUDE_intersection_A_B_l2997_299762

-- Define the sets A and B
def A : Set ℝ := {x | Real.log (x - 1) ≤ 0}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem intersection_A_B : A ∩ B = Set.Ioc 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2997_299762


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2997_299705

noncomputable def i : ℂ := Complex.I

theorem modulus_of_complex_fraction :
  Complex.abs (2 * i / (1 + i)) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2997_299705


namespace NUMINAMATH_CALUDE_purification_cost_is_one_l2997_299713

/-- The cost to purify a gallon of fresh water -/
def purification_cost (water_per_person : ℚ) (family_size : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / (water_per_person * family_size)

/-- Theorem: The cost to purify a gallon of fresh water is $1 -/
theorem purification_cost_is_one :
  purification_cost (1/2) 6 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_purification_cost_is_one_l2997_299713


namespace NUMINAMATH_CALUDE_division_result_l2997_299737

theorem division_result : (-0.91) / (-0.13) = 7 := by sorry

end NUMINAMATH_CALUDE_division_result_l2997_299737


namespace NUMINAMATH_CALUDE_base_five_product_theorem_l2997_299746

def base_five_to_decimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (5 ^ i)) 0

def decimal_to_base_five (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec convert (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else convert (m / 5) ((m % 5) :: acc)
  convert n []

def base_five_multiply (a b : List Nat) : List Nat :=
  decimal_to_base_five ((base_five_to_decimal a) * (base_five_to_decimal b))

theorem base_five_product_theorem :
  base_five_multiply [1, 3, 1] [2, 1] = [2, 2, 1, 2] := by sorry

end NUMINAMATH_CALUDE_base_five_product_theorem_l2997_299746


namespace NUMINAMATH_CALUDE_missing_number_equation_l2997_299794

theorem missing_number_equation (x : ℝ) : 11 + Real.sqrt (-4 + 6 * x / 3) = 13 ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_equation_l2997_299794


namespace NUMINAMATH_CALUDE_xy_value_l2997_299725

theorem xy_value (x y : ℝ) (h : (1 + Complex.I) * x + (1 - Complex.I) * y = 2) : x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2997_299725


namespace NUMINAMATH_CALUDE_f_range_l2997_299720

-- Define the function
def f (x : ℝ) := x^2 - 6*x + 7

-- State the theorem
theorem f_range :
  {y : ℝ | ∃ x ≥ 4, f x = y} = {y : ℝ | y ≥ -1} :=
by sorry

end NUMINAMATH_CALUDE_f_range_l2997_299720


namespace NUMINAMATH_CALUDE_three_lines_determine_plane_l2997_299792

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane where
  -- Define properties of a plane

/-- Represents the intersection of two lines -/
def intersect (l1 l2 : Line3D) : Prop :=
  sorry

/-- Represents that three lines have no common point -/
def no_common_point (l1 l2 l3 : Line3D) : Prop :=
  sorry

/-- Represents that a plane contains a line -/
def plane_contains_line (p : Plane) (l : Line3D) : Prop :=
  sorry

/-- Three lines intersecting in pairs without a common point determine a unique plane -/
theorem three_lines_determine_plane (l1 l2 l3 : Line3D) :
  intersect l1 l2 ∧ intersect l2 l3 ∧ intersect l3 l1 ∧ no_common_point l1 l2 l3 →
  ∃! p : Plane, plane_contains_line p l1 ∧ plane_contains_line p l2 ∧ plane_contains_line p l3 :=
sorry

end NUMINAMATH_CALUDE_three_lines_determine_plane_l2997_299792


namespace NUMINAMATH_CALUDE_sam_exchange_probability_l2997_299703

/-- Represents the vending machine and Sam's purchasing scenario -/
structure VendingMachine where
  num_toys : Nat
  toy_prices : List Rat
  favorite_toy_price : Rat
  sam_quarters : Nat
  sam_bill : Nat

/-- Calculates the probability of Sam needing to exchange his bill -/
def probability_need_exchange (vm : VendingMachine) : Rat :=
  1 - (Nat.factorial 7 : Rat) / (Nat.factorial vm.num_toys : Rat)

/-- Theorem stating the probability of Sam needing to exchange his bill -/
theorem sam_exchange_probability (vm : VendingMachine) :
  vm.num_toys = 10 ∧
  vm.toy_prices = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5] ∧
  vm.favorite_toy_price = 4 ∧
  vm.sam_quarters = 12 ∧
  vm.sam_bill = 20 →
  probability_need_exchange vm = 719 / 720 := by
  sorry

#eval probability_need_exchange {
  num_toys := 10,
  toy_prices := [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
  favorite_toy_price := 4,
  sam_quarters := 12,
  sam_bill := 20
}

end NUMINAMATH_CALUDE_sam_exchange_probability_l2997_299703


namespace NUMINAMATH_CALUDE_water_tank_capacity_l2997_299726

theorem water_tank_capacity (initial_fraction : ℚ) (final_fraction : ℚ) (added_amount : ℚ) (capacity : ℚ) : 
  initial_fraction = 1/7 →
  final_fraction = 1/5 →
  added_amount = 5 →
  initial_fraction * capacity + added_amount = final_fraction * capacity →
  capacity = 87.5 := by
sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l2997_299726


namespace NUMINAMATH_CALUDE_monthly_salary_is_6250_l2997_299715

/-- Calculates the monthly salary given savings rate, expense increase, and new savings amount -/
def calculate_salary (savings_rate : ℚ) (expense_increase : ℚ) (new_savings : ℚ) : ℚ :=
  new_savings / (savings_rate - (1 - savings_rate) * expense_increase)

/-- Theorem stating that under the given conditions, the monthly salary is 6250 -/
theorem monthly_salary_is_6250 :
  let savings_rate : ℚ := 1/5
  let expense_increase : ℚ := 1/5
  let new_savings : ℚ := 250
  calculate_salary savings_rate expense_increase new_savings = 6250 := by
sorry

#eval calculate_salary (1/5) (1/5) 250

end NUMINAMATH_CALUDE_monthly_salary_is_6250_l2997_299715


namespace NUMINAMATH_CALUDE_gcd_360_504_l2997_299786

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by
  sorry

end NUMINAMATH_CALUDE_gcd_360_504_l2997_299786


namespace NUMINAMATH_CALUDE_min_sum_squares_l2997_299700

-- Define the set of possible values
def S : Finset Int := {-6, -4, -1, 0, 3, 5, 7, 10}

-- Define the theorem
theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 98 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2997_299700


namespace NUMINAMATH_CALUDE_parabola_x_intercepts_distance_l2997_299791

-- Define the parabola function
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Theorem statement
theorem parabola_x_intercepts_distance : 
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - x₁ = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_x_intercepts_distance_l2997_299791


namespace NUMINAMATH_CALUDE_committee_formation_l2997_299788

theorem committee_formation (n : ℕ) (k : ℕ) (h : n = 8 ∧ k = 4) : 
  (Nat.choose (n - 1) (k - 1)) * (Nat.choose (n - 1) k) = 1225 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_l2997_299788


namespace NUMINAMATH_CALUDE_colored_triangle_existence_l2997_299710

-- Define the number of colors
def num_colors : ℕ := 1992

-- Define a type for colors
def Color := Fin num_colors

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a type for triangles
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Define a function to check if a point is on a side of a triangle (excluding vertices)
def on_side (p : Point) (t : Triangle) : Prop := sorry

-- Define a function to check if a side of a triangle contains a point of a given color
def side_has_color (t : Triangle) (c : Color) : Prop := sorry

-- State the theorem
theorem colored_triangle_existence :
  (∀ c : Color, ∃ p : Point, coloring p = c) →
  ∀ T : Triangle, ∃ T' : Triangle,
    congruent T T' ∧
    ∃ c1 c2 c3 : Color,
      side_has_color T' c1 ∧
      side_has_color T' c2 ∧
      side_has_color T' c3 :=
by sorry

end NUMINAMATH_CALUDE_colored_triangle_existence_l2997_299710


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_one_two_l2997_299706

theorem sqrt_equality_implies_one_two :
  ∀ a b : ℕ+,
  a < b →
  (Real.sqrt (1 + Real.sqrt (18 + 8 * Real.sqrt 2)) = Real.sqrt a + Real.sqrt b) →
  (a = 1 ∧ b = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_one_two_l2997_299706


namespace NUMINAMATH_CALUDE_remainder_three_pow_244_mod_5_l2997_299741

theorem remainder_three_pow_244_mod_5 : 3^244 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_pow_244_mod_5_l2997_299741


namespace NUMINAMATH_CALUDE_largest_term_binomial_expansion_largest_term_specific_case_l2997_299754

theorem largest_term_binomial_expansion (n : ℕ) (x : ℝ) (h : x > 0) :
  let A : ℕ → ℝ := λ k => (n.choose k) * x^k
  ∃ k : ℕ, k ≤ n ∧ ∀ j : ℕ, j ≤ n → A k ≥ A j :=
by
  sorry

theorem largest_term_specific_case :
  let n : ℕ := 500
  let x : ℝ := 0.3
  let A : ℕ → ℝ := λ k => (n.choose k) * x^k
  ∃ k : ℕ, k = 125 ∧ ∀ j : ℕ, j ≤ n → A k ≥ A j :=
by
  sorry

end NUMINAMATH_CALUDE_largest_term_binomial_expansion_largest_term_specific_case_l2997_299754


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2997_299784

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (((a > 0 ∧ b > 0) → (a * b > 0)) ∧
   (∃ a b : ℝ, (a * b > 0) ∧ ¬(a > 0 ∧ b > 0))) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2997_299784


namespace NUMINAMATH_CALUDE_cat_arrangements_l2997_299752

def number_of_arrangements (n : ℕ) : ℕ := Nat.factorial n

theorem cat_arrangements : number_of_arrangements 3 = 6 := by sorry

end NUMINAMATH_CALUDE_cat_arrangements_l2997_299752


namespace NUMINAMATH_CALUDE_concertHallSeats_l2997_299755

/-- Represents a concert hall with a specific seating arrangement. -/
structure ConcertHall where
  rows : ℕ
  middleRowSeats : ℕ
  middleRowIndex : ℕ
  increaseFactor : ℕ

/-- Calculates the total number of seats in the concert hall. -/
def totalSeats (hall : ConcertHall) : ℕ :=
  let firstRowSeats := hall.middleRowSeats - 2 * (hall.middleRowIndex - 1)
  let lastRowSeats := hall.middleRowSeats + 2 * (hall.rows - hall.middleRowIndex)
  hall.rows * (firstRowSeats + lastRowSeats) / 2

/-- Theorem stating that a concert hall with the given properties has 1984 seats. -/
theorem concertHallSeats (hall : ConcertHall) 
    (h1 : hall.rows = 31)
    (h2 : hall.middleRowSeats = 64)
    (h3 : hall.middleRowIndex = 16)
    (h4 : hall.increaseFactor = 2) : 
  totalSeats hall = 1984 := by
  sorry


end NUMINAMATH_CALUDE_concertHallSeats_l2997_299755


namespace NUMINAMATH_CALUDE_range_of_f_l2997_299749

-- Define the function f
def f (x : ℝ) : ℝ := 3 * (x - 4)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range (fun (x : ℝ) => f x) = {y : ℝ | y < -27 ∨ y > -27} :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l2997_299749


namespace NUMINAMATH_CALUDE_total_pencils_is_fifty_l2997_299756

/-- The number of pencils Sabrina has -/
def sabrina_pencils : ℕ := 14

/-- The number of pencils Justin has -/
def justin_pencils : ℕ := 2 * sabrina_pencils + 8

/-- The total number of pencils Justin and Sabrina have combined -/
def total_pencils : ℕ := justin_pencils + sabrina_pencils

/-- Theorem stating that the total number of pencils is 50 -/
theorem total_pencils_is_fifty : total_pencils = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_is_fifty_l2997_299756


namespace NUMINAMATH_CALUDE_xyz_sum_reciprocal_l2997_299717

theorem xyz_sum_reciprocal (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_sum_x : x + 1 / z = 4)
  (h_sum_y : y + 1 / x = 30) :
  z + 1 / y = 36 / 119 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_reciprocal_l2997_299717


namespace NUMINAMATH_CALUDE_total_hours_worked_l2997_299742

/-- 
Given a person works 8 hours per day for 4 days, 
prove that the total number of hours worked is 32.
-/
theorem total_hours_worked (hours_per_day : ℕ) (days_worked : ℕ) : 
  hours_per_day = 8 → days_worked = 4 → hours_per_day * days_worked = 32 := by
sorry

end NUMINAMATH_CALUDE_total_hours_worked_l2997_299742


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l2997_299795

theorem sum_of_squares_and_products (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → 
  x^2 + y^2 + z^2 = 75 → 
  x*y + y*z + z*x = 32 → 
  x + y + z = Real.sqrt 139 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l2997_299795


namespace NUMINAMATH_CALUDE_distribute_six_among_four_l2997_299734

/-- The number of ways to distribute n indistinguishable objects among k distinct containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The theorem stating that there are 84 ways to distribute 6 objects among 4 containers -/
theorem distribute_six_among_four : distribute 6 4 = 84 := by sorry

end NUMINAMATH_CALUDE_distribute_six_among_four_l2997_299734


namespace NUMINAMATH_CALUDE_translation_of_A_l2997_299723

-- Define the points A, B, and C
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (5, 2)
def C : ℝ × ℝ := (3, -1)

-- Define the translation function
def translate (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + (C.1 - B.1), p.2 + (C.2 - B.2))

-- Theorem statement
theorem translation_of_A :
  translate A = (0, 1) := by sorry

end NUMINAMATH_CALUDE_translation_of_A_l2997_299723


namespace NUMINAMATH_CALUDE_timeDifference_div_by_40_l2997_299781

/-- Represents time in days, hours, minutes, and seconds -/
structure Time where
  days : ℕ
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Converts Time to its numerical representation (ignoring punctuation) -/
def Time.toNumerical (t : Time) : ℕ :=
  10^6 * t.days + 10^4 * t.hours + 100 * t.minutes + t.seconds

/-- Converts Time to total seconds -/
def Time.toSeconds (t : Time) : ℕ :=
  86400 * t.days + 3600 * t.hours + 60 * t.minutes + t.seconds

/-- The difference between numerical representation and total seconds -/
def timeDifference (t : Time) : ℤ :=
  (t.toNumerical : ℤ) - (t.toSeconds : ℤ)

/-- Theorem: 40 always divides the time difference -/
theorem timeDifference_div_by_40 (t : Time) : 
  (40 : ℤ) ∣ timeDifference t := by
  sorry

end NUMINAMATH_CALUDE_timeDifference_div_by_40_l2997_299781


namespace NUMINAMATH_CALUDE_set_formation_criterion_l2997_299748

-- Define a type for objects
variable {α : Type}

-- Define a predicate for definiteness and distinctness
def is_definite_and_distinct (S : Set α) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x = y ∨ x ≠ y)

-- Define a predicate for forming a set
def can_form_set (S : Set α) : Prop :=
  is_definite_and_distinct S

-- Theorem statement
theorem set_formation_criterion (S : Set α) :
  can_form_set S ↔ is_definite_and_distinct S :=
by
  sorry


end NUMINAMATH_CALUDE_set_formation_criterion_l2997_299748


namespace NUMINAMATH_CALUDE_find_x_l2997_299778

def binary_op (n : ℤ) (x : ℚ) : ℚ := n - (n * x)

theorem find_x : 
  (∀ n : ℤ, n > 3 → binary_op n x ≥ 14) ∧
  (binary_op 3 x < 14) →
  x = -3 := by sorry

end NUMINAMATH_CALUDE_find_x_l2997_299778


namespace NUMINAMATH_CALUDE_gcd_459_357_l2997_299777

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l2997_299777


namespace NUMINAMATH_CALUDE_line_through_two_points_line_with_special_intercepts_l2997_299767

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def pointOnLine (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Part 1
theorem line_through_two_points :
  ∃ (l : Line), pointOnLine l ⟨4, 1⟩ ∧ pointOnLine l ⟨-1, 6⟩ →
  l.slope * 1 + l.intercept = 5 :=
sorry

-- Part 2
theorem line_with_special_intercepts :
  ∃ (l : Line), pointOnLine l ⟨4, 1⟩ ∧ 
  (l.intercept = 2 * (- l.intercept / l.slope)) →
  (l.slope = 1/4 ∧ l.intercept = 0) ∨ (l.slope = -2 ∧ l.intercept = 9) :=
sorry

end NUMINAMATH_CALUDE_line_through_two_points_line_with_special_intercepts_l2997_299767


namespace NUMINAMATH_CALUDE_correct_equation_transformation_l2997_299744

theorem correct_equation_transformation (y : ℝ) :
  (5 * y = -4 * y + 2) ↔ (5 * y + 4 * y = 2) :=
by sorry

end NUMINAMATH_CALUDE_correct_equation_transformation_l2997_299744


namespace NUMINAMATH_CALUDE_sheep_per_herd_l2997_299763

theorem sheep_per_herd (total_sheep : ℕ) (num_herds : ℕ) (h1 : total_sheep = 60) (h2 : num_herds = 3) :
  total_sheep / num_herds = 20 := by
  sorry

end NUMINAMATH_CALUDE_sheep_per_herd_l2997_299763


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_l2997_299750

theorem quadratic_always_nonnegative (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) ↔ a ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_l2997_299750


namespace NUMINAMATH_CALUDE_consecutive_substring_perfect_square_l2997_299770

/-- A type representing a 16-digit positive integer -/
def SixteenDigitInteger := { n : ℕ // 10^15 ≤ n ∧ n < 10^16 }

/-- A function that checks if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- A function that returns the product of digits in a substring of a number -/
def substring_product (n : ℕ) (start finish : ℕ) : ℕ := sorry

/-- The main theorem: For any 16-digit positive integer, there exists a consecutive
    substring of digits whose product is a perfect square -/
theorem consecutive_substring_perfect_square (A : SixteenDigitInteger) :
  ∃ start finish : ℕ, start ≤ finish ∧ finish ≤ 16 ∧
    is_perfect_square (substring_product A.val start finish) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_substring_perfect_square_l2997_299770


namespace NUMINAMATH_CALUDE_zoo_ticket_price_l2997_299729

def ticket_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (service_fee : ℝ) : ℝ :=
  let price_after_discount1 := initial_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  price_after_discount2 + service_fee

theorem zoo_ticket_price :
  ticket_price 15 0.4 0.1 2 = 10.1 := by
  sorry

end NUMINAMATH_CALUDE_zoo_ticket_price_l2997_299729


namespace NUMINAMATH_CALUDE_ratio_problem_l2997_299769

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 4) (h2 : c/b = 2) :
  (a + b + c) / (a + b) = 13/5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2997_299769


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_of_100_l2997_299716

def is_factor (n m : ℕ) : Prop := m % n = 0

theorem smallest_non_factor_product_of_100 :
  ∃ (a b : ℕ),
    a ≠ b ∧
    a > 0 ∧
    b > 0 ∧
    is_factor a 100 ∧
    is_factor b 100 ∧
    ¬(is_factor (a * b) 100) ∧
    a * b = 8 ∧
    ∀ (c d : ℕ),
      c ≠ d →
      c > 0 →
      d > 0 →
      is_factor c 100 →
      is_factor d 100 →
      ¬(is_factor (c * d) 100) →
      c * d ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_of_100_l2997_299716


namespace NUMINAMATH_CALUDE_water_in_mixture_l2997_299722

theorem water_in_mixture (water_parts syrup_parts total_volume : ℚ) 
  (h1 : water_parts = 5)
  (h2 : syrup_parts = 2)
  (h3 : total_volume = 3) : 
  (water_parts * total_volume) / (water_parts + syrup_parts) = 15 / 7 := by
  sorry

end NUMINAMATH_CALUDE_water_in_mixture_l2997_299722


namespace NUMINAMATH_CALUDE_multiple_of_a_l2997_299733

theorem multiple_of_a (a : ℤ) : 
  (∃ k : ℤ, 97 * a^2 + 84 * a - 55 = k * a) ↔ 
  (a = 1 ∨ a = 5 ∨ a = 11 ∨ a = 55 ∨ a = -1 ∨ a = -5 ∨ a = -11 ∨ a = -55) :=
by sorry

end NUMINAMATH_CALUDE_multiple_of_a_l2997_299733


namespace NUMINAMATH_CALUDE_total_fish_count_l2997_299701

theorem total_fish_count (micah kenneth matthias gabrielle : ℕ) : 
  micah = 7 →
  kenneth = 3 * micah →
  matthias = kenneth - 15 →
  gabrielle = 2 * (micah + kenneth + matthias) →
  micah + kenneth + matthias + gabrielle = 102 := by
sorry

end NUMINAMATH_CALUDE_total_fish_count_l2997_299701


namespace NUMINAMATH_CALUDE_fashion_show_models_l2997_299775

/-- The number of bathing suit sets each model wears -/
def bathing_suit_sets : ℕ := 2

/-- The number of evening wear sets each model wears -/
def evening_wear_sets : ℕ := 3

/-- The time in minutes for one runway walk -/
def runway_walk_time : ℕ := 2

/-- The total runway time for the show in minutes -/
def total_runway_time : ℕ := 60

/-- The number of models in the fashion show -/
def number_of_models : ℕ := 6

theorem fashion_show_models :
  (bathing_suit_sets + evening_wear_sets) * runway_walk_time * number_of_models = total_runway_time :=
by sorry

end NUMINAMATH_CALUDE_fashion_show_models_l2997_299775


namespace NUMINAMATH_CALUDE_sophie_germain_characterization_l2997_299702

/-- A prime number p is a Sophie Germain prime if 2p + 1 is also prime. -/
def SophieGermainPrime (p : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime (2 * p + 1)

/-- The product of all possible units digits of Sophie Germain primes greater than 6 is 189. -/
axiom units_digit_product : ∃ (S : Finset ℕ), (∀ n ∈ S, n > 6 ∧ SophieGermainPrime n) ∧
  (Finset.prod S (λ n => n % 10) = 189)

theorem sophie_germain_characterization (p : ℕ) (h_prime : Nat.Prime p) (h_greater : p > 6) :
  SophieGermainPrime p ↔ Nat.Prime (2 * p + 1) :=
sorry

end NUMINAMATH_CALUDE_sophie_germain_characterization_l2997_299702


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l2997_299768

/-- The focal length of an ellipse with given properties -/
theorem ellipse_focal_length (k : ℝ) : 
  let ellipse := {(x, y) : ℝ × ℝ | x^2 - y^2 / k = 1}
  let focus_on_x_axis := true  -- This is a simplification, as we can't directly represent this in Lean
  let eccentricity := (1 : ℝ) / 2
  let focal_length := 1
  (∀ (x y : ℝ), (x, y) ∈ ellipse → x^2 - y^2 / k = 1) ∧ 
  focus_on_x_axis ∧ 
  eccentricity = 1 / 2 →
  focal_length = 1 := by
sorry


end NUMINAMATH_CALUDE_ellipse_focal_length_l2997_299768


namespace NUMINAMATH_CALUDE_existence_of_xy_l2997_299774

theorem existence_of_xy (a b c : ℝ) 
  (h1 : |a| > 2) 
  (h2 : a^2 + b^2 + c^2 = a*b*c + 4) : 
  ∃ x y : ℝ, 
    a = x + 1/x ∧ 
    b = y + 1/y ∧ 
    c = x*y + 1/(x*y) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_xy_l2997_299774


namespace NUMINAMATH_CALUDE_number_of_piglets_born_l2997_299783

def sellPrice : ℕ := 300
def feedCost : ℕ := 10
def profitEarned : ℕ := 960

def pigsSoldAt12Months : ℕ := 3
def pigsSoldAt16Months : ℕ := 3

def totalPigsSold : ℕ := pigsSoldAt12Months + pigsSoldAt16Months

theorem number_of_piglets_born (sellPrice feedCost profitEarned 
  pigsSoldAt12Months pigsSoldAt16Months totalPigsSold : ℕ) :
  sellPrice = 300 →
  feedCost = 10 →
  profitEarned = 960 →
  pigsSoldAt12Months = 3 →
  pigsSoldAt16Months = 3 →
  totalPigsSold = pigsSoldAt12Months + pigsSoldAt16Months →
  totalPigsSold = 6 :=
by sorry

end NUMINAMATH_CALUDE_number_of_piglets_born_l2997_299783


namespace NUMINAMATH_CALUDE_probability_different_topics_l2997_299764

/-- The number of topics in the essay competition -/
def num_topics : ℕ := 6

/-- The probability that two students select different topics -/
def prob_different_topics : ℚ := 5/6

/-- Theorem stating the probability of two students selecting different topics -/
theorem probability_different_topics :
  (num_topics : ℚ) * (num_topics - 1) / (num_topics * num_topics) = prob_different_topics :=
sorry

end NUMINAMATH_CALUDE_probability_different_topics_l2997_299764


namespace NUMINAMATH_CALUDE_complement_union_sets_l2997_299712

def I : Finset ℕ := {0,1,2,3,4,5,6,7,8}
def M : Finset ℕ := {1,2,4,5}
def N : Finset ℕ := {0,3,5,7}

theorem complement_union_sets : (I \ (M ∪ N)) = {6,8} := by sorry

end NUMINAMATH_CALUDE_complement_union_sets_l2997_299712


namespace NUMINAMATH_CALUDE_sin_2beta_value_l2997_299745

theorem sin_2beta_value (α β : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : Real.cos (2 * α + β) - 2 * Real.cos (α + β) * Real.cos α = 3/5) :
  Real.sin (2 * β) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2beta_value_l2997_299745


namespace NUMINAMATH_CALUDE_min_stool_height_l2997_299790

/-- The minimum height of the stool for Alice to reach the ceiling fan switch -/
theorem min_stool_height (ceiling_height : ℝ) (switch_below_ceiling : ℝ) 
  (alice_height : ℝ) (alice_reach : ℝ) (books_height : ℝ) 
  (h1 : ceiling_height = 300) 
  (h2 : switch_below_ceiling = 15)
  (h3 : alice_height = 160)
  (h4 : alice_reach = 50)
  (h5 : books_height = 12) : 
  ∃ (s : ℝ), s ≥ 63 ∧ 
  ∀ (x : ℝ), x < 63 → alice_height + alice_reach + books_height + x < ceiling_height - switch_below_ceiling :=
sorry

end NUMINAMATH_CALUDE_min_stool_height_l2997_299790


namespace NUMINAMATH_CALUDE_elevator_floors_l2997_299782

/-- The number of floors the elevator needs to move down. -/
def total_floors : ℕ := sorry

/-- The time taken for the first half of the floors (in minutes). -/
def first_half_time : ℕ := 15

/-- The time taken per floor for the next 5 floors (in minutes). -/
def middle_time_per_floor : ℕ := 5

/-- The number of floors in the middle section. -/
def middle_floors : ℕ := 5

/-- The time taken per floor for the final 5 floors (in minutes). -/
def final_time_per_floor : ℕ := 16

/-- The number of floors in the final section. -/
def final_floors : ℕ := 5

/-- The total time taken to reach the bottom (in minutes). -/
def total_time : ℕ := 120

theorem elevator_floors :
  first_half_time + 
  (middle_time_per_floor * middle_floors) + 
  (final_time_per_floor * final_floors) = total_time ∧
  total_floors = (total_floors / 2) + middle_floors + final_floors ∧
  total_floors = 20 := by sorry

end NUMINAMATH_CALUDE_elevator_floors_l2997_299782


namespace NUMINAMATH_CALUDE_existence_of_polynomials_l2997_299780

theorem existence_of_polynomials : ∃ (p q : Polynomial ℤ),
  (∃ (i j : ℕ), (abs (p.coeff i) > 2015) ∧ (abs (q.coeff j) > 2015)) ∧
  (∀ k : ℕ, abs ((p * q).coeff k) ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_polynomials_l2997_299780


namespace NUMINAMATH_CALUDE_point_coordinates_l2997_299772

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: A point in the second quadrant with distance 2 to the x-axis
    and distance 3 to the y-axis has coordinates (-3, 2) -/
theorem point_coordinates (p : Point) 
    (h1 : SecondQuadrant p) 
    (h2 : DistanceToXAxis p = 2) 
    (h3 : DistanceToYAxis p = 3) : 
    p = Point.mk (-3) 2 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2997_299772


namespace NUMINAMATH_CALUDE_rectangle_area_sum_l2997_299753

theorem rectangle_area_sum : 
  let rect1 := 7 * 8
  let rect2 := 5 * 3
  let rect3 := 2 * 8
  let rect4 := 2 * 7
  let rect5 := 4 * 4
  rect1 + rect2 + rect3 + rect4 + rect5 = 117 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_sum_l2997_299753


namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l2997_299719

theorem larger_solution_of_quadratic (x : ℝ) :
  x^2 - 9*x - 22 = 0 →
  (∃ y : ℝ, y ≠ x ∧ y^2 - 9*y - 22 = 0) →
  (x = 11 ∨ x < 11) :=
sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l2997_299719


namespace NUMINAMATH_CALUDE_apple_count_correct_l2997_299757

/-- The number of apples in a box containing apples and oranges -/
def num_apples : ℕ := 14

/-- The initial number of oranges in the box -/
def initial_oranges : ℕ := 20

/-- The number of oranges removed from the box -/
def removed_oranges : ℕ := 14

/-- The percentage of apples after removing oranges -/
def apple_percentage : ℝ := 0.7

theorem apple_count_correct :
  num_apples = 14 ∧
  initial_oranges = 20 ∧
  removed_oranges = 14 ∧
  apple_percentage = 0.7 ∧
  (num_apples : ℝ) / ((num_apples : ℝ) + (initial_oranges - removed_oranges : ℝ)) = apple_percentage :=
by sorry

end NUMINAMATH_CALUDE_apple_count_correct_l2997_299757


namespace NUMINAMATH_CALUDE_min_p_plus_q_min_p_plus_q_value_l2997_299728

theorem min_p_plus_q (p q : ℕ+) (h : 90 * p = q^3) : 
  ∀ (p' q' : ℕ+), 90 * p' = q'^3 → p + q ≤ p' + q' :=
by sorry

theorem min_p_plus_q_value (p q : ℕ+) (h : 90 * p = q^3) 
  (h_min : ∀ (p' q' : ℕ+), 90 * p' = q'^3 → p + q ≤ p' + q') : 
  p + q = 330 :=
by sorry

end NUMINAMATH_CALUDE_min_p_plus_q_min_p_plus_q_value_l2997_299728


namespace NUMINAMATH_CALUDE_local_road_speed_l2997_299735

/-- Proves that the speed of a car on local roads is 20 mph given the specified conditions -/
theorem local_road_speed (local_distance : ℝ) (highway_distance : ℝ) (highway_speed : ℝ) (average_speed : ℝ) :
  local_distance = 60 →
  highway_distance = 120 →
  highway_speed = 60 →
  average_speed = 36 →
  (local_distance + highway_distance) / (local_distance / (local_distance / (local_distance / average_speed - highway_distance / highway_speed)) + highway_distance / highway_speed) = average_speed →
  local_distance / (local_distance / average_speed - highway_distance / highway_speed) = 20 :=
by sorry

end NUMINAMATH_CALUDE_local_road_speed_l2997_299735


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2997_299758

/-- A quadratic function with a specific extremum and tangent line property -/
def QuadraticFunction (a b k : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + k

theorem quadratic_function_property (a b k : ℝ) (h1 : k > 0) :
  (∃ (y : ℝ → ℝ), y = QuadraticFunction a b k ∧ 
    (∀ x, (deriv y) x = 2 * a * x + b) ∧  -- Definition of derivative
    ((deriv y) 0 = 0) ∧  -- Extremum at x = 0
    ((deriv y) 1 = -1 / 2)) →  -- Tangent line perpendicular to x + 2y + 1 = 0
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2997_299758


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l2997_299727

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l2997_299727


namespace NUMINAMATH_CALUDE_arithmetic_mean_function_constant_l2997_299714

/-- A function from ℤ² to ℤ⁺ satisfying the arithmetic mean property -/
def ArithmeticMeanFunction (f : ℤ × ℤ → ℕ+) : Prop :=
  ∀ x y : ℤ, (f (x, y) : ℚ) = (f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)) / 4

/-- If a function satisfies the arithmetic mean property, then it is constant -/
theorem arithmetic_mean_function_constant (f : ℤ × ℤ → ℕ+) 
  (h : ArithmeticMeanFunction f) : 
  ∀ p q : ℤ × ℤ, f p = f q :=
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_function_constant_l2997_299714


namespace NUMINAMATH_CALUDE_sum_of_digits_product_of_nines_l2997_299771

/-- 
Given a natural number n, define a function that calculates the product:
9 × 99 × 9999 × ⋯ × (99...99) where the number of nines doubles in each factor
and the last factor has 2^n nines.
-/
def productOfNines (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * (10^(2^i) - 1)) 9

/-- 
Sum of digits function
-/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- 
Theorem: The sum of the digits of the product of nines is equal to 9 * 2^n
-/
theorem sum_of_digits_product_of_nines (n : ℕ) :
  sumOfDigits (productOfNines n) = 9 * 2^n := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_product_of_nines_l2997_299771


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_distance_line_equations_l2997_299721

-- Define the lines
def l₁ (x y : ℝ) : Prop := y = 2 * x
def l₂ (x y : ℝ) : Prop := x + y = 6
def l₀ (x y : ℝ) : Prop := x - 2 * y = 0

-- Define the intersection point P
def P : ℝ × ℝ := (2, 4)

-- Theorem for part (1)
theorem perpendicular_line_equation :
  ∀ x y : ℝ, (x - P.1) = -2 * (y - P.2) ↔ 2 * x + y - 8 = 0 :=
sorry

-- Theorem for part (2)
theorem distance_line_equations :
  ∀ x y : ℝ, 
    (x = P.1 ∨ 3 * x - 4 * y + 10 = 0) ↔
    (∃ k : ℝ, y - P.2 = k * (x - P.1) ∧ 
      |k * P.1 - P.2| / Real.sqrt (k^2 + 1) = 2) ∨
    (x = P.1 ∧ |x| = 2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_distance_line_equations_l2997_299721


namespace NUMINAMATH_CALUDE_hyperbola_center_l2997_299743

/-- The center of a hyperbola given by the equation ((3y+3)^2)/(7^2) - ((4x-8)^2)/(6^2) = 1 -/
theorem hyperbola_center (x y : ℝ) : 
  (((3 * y + 3)^2) / 7^2) - (((4 * x - 8)^2) / 6^2) = 1 → 
  (∃ (h k : ℝ), h = 2 ∧ k = -1 ∧ 
    ((y - k)^2) / ((7/3)^2) - ((x - h)^2) / ((3/2)^2) = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_center_l2997_299743


namespace NUMINAMATH_CALUDE_scaling_transformation_cosine_curve_l2997_299787

/-- The scaling transformation applied to the curve y = cos 6x results in y' = 2cos 2x' -/
theorem scaling_transformation_cosine_curve :
  ∀ (x y x' y' : ℝ),
  y = Real.cos (6 * x) →
  x' = 3 * x →
  y' = 2 * y →
  y' = 2 * Real.cos (2 * x') := by
sorry

end NUMINAMATH_CALUDE_scaling_transformation_cosine_curve_l2997_299787


namespace NUMINAMATH_CALUDE_fib_100_mod_9_l2997_299765

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The 100th Fibonacci number is congruent to 3 modulo 9 -/
theorem fib_100_mod_9 : fib 100 % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fib_100_mod_9_l2997_299765


namespace NUMINAMATH_CALUDE_cubic_polynomial_proof_l2997_299799

theorem cubic_polynomial_proof : 
  let p : ℝ → ℝ := λ x => -5/6 * x^3 + 5 * x^2 - 85/6 * x - 5
  (p 1 = -10) ∧ (p 2 = -20) ∧ (p 3 = -30) ∧ (p 5 = -70) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_proof_l2997_299799


namespace NUMINAMATH_CALUDE_max_good_word_length_l2997_299736

/-- An alphabet is a finite set of letters. -/
def Alphabet (n : ℕ) := Fin n

/-- A word is a finite sequence of letters where consecutive letters are different. -/
def Word (α : Type) := List α

/-- A good word is one where it's impossible to delete all but four letters to obtain aabb. -/
def isGoodWord {α : Type} (w : Word α) : Prop :=
  ∀ (a b : α), a ≠ b → ¬∃ (i j k l : ℕ), i < j ∧ j < k ∧ k < l ∧
    w.get? i = some a ∧ w.get? j = some a ∧ w.get? k = some b ∧ w.get? l = some b

/-- The maximum length of a good word in an alphabet with n > 1 letters is 2n + 1. -/
theorem max_good_word_length {n : ℕ} (h : n > 1) :
  ∃ (w : Word (Alphabet n)), isGoodWord w ∧ w.length = 2 * n + 1 ∧
  ∀ (w' : Word (Alphabet n)), isGoodWord w' → w'.length ≤ 2 * n + 1 :=
sorry

end NUMINAMATH_CALUDE_max_good_word_length_l2997_299736


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2997_299731

def M : Set ℝ := {x | |x - 1| > 1}
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

theorem intersection_of_M_and_N : M ∩ N = {x | 2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2997_299731


namespace NUMINAMATH_CALUDE_barn_paint_area_l2997_299789

/-- Represents the dimensions of a rectangular barn -/
structure BarnDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the total area to be painted for a barn with given dimensions -/
def totalPaintArea (d : BarnDimensions) : ℝ :=
  let wallArea1 := 2 * d.width * d.height
  let wallArea2 := 2 * d.length * d.height
  let ceilingArea := d.width * d.length
  let roofArea := d.width * d.length
  2 * wallArea1 + 2 * wallArea2 + ceilingArea + roofArea

/-- The theorem stating that the total paint area for the given barn dimensions is 1116 sq yd -/
theorem barn_paint_area :
  let barn := BarnDimensions.mk 12 15 7
  totalPaintArea barn = 1116 := by sorry

end NUMINAMATH_CALUDE_barn_paint_area_l2997_299789


namespace NUMINAMATH_CALUDE_project_completion_proof_l2997_299793

/-- Represents the number of days to complete the project -/
def project_completion_time : ℕ := 11

/-- Represents A's completion rate per day -/
def rate_A : ℚ := 1 / 20

/-- Represents B's initial completion rate per day -/
def rate_B : ℚ := 1 / 30

/-- Represents C's completion rate per day -/
def rate_C : ℚ := 1 / 40

/-- Represents B's doubled completion rate -/
def rate_B_doubled : ℚ := 2 * rate_B

/-- Represents the time A quits before project completion -/
def time_A_quits_before : ℕ := 10

/-- Theorem stating that the project will be completed in 11 days -/
theorem project_completion_proof :
  let total_work : ℚ := 1
  let combined_rate : ℚ := rate_A + rate_B + rate_C
  let final_rate : ℚ := rate_B_doubled + rate_C
  (project_completion_time - time_A_quits_before) * combined_rate +
  time_A_quits_before * final_rate = total_work :=
by sorry


end NUMINAMATH_CALUDE_project_completion_proof_l2997_299793


namespace NUMINAMATH_CALUDE_simplified_root_sum_l2997_299776

theorem simplified_root_sum (a b : ℕ+) :
  (2^11 * 5^5 : ℝ)^(1/4) = a * b^(1/4) → a + b = 30 := by
  sorry

end NUMINAMATH_CALUDE_simplified_root_sum_l2997_299776


namespace NUMINAMATH_CALUDE_quadratic_solution_existence_l2997_299711

theorem quadratic_solution_existence (a : ℝ) :
  (∃ x : ℝ, x^2 - 2*x + a = 0) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_existence_l2997_299711


namespace NUMINAMATH_CALUDE_triangle_intersection_invariance_l2997_299785

/-- Represents a right triangle in a plane -/
structure RightTriangle where
  leg1 : Real
  leg2 : Real

/-- Represents a line in a plane -/
structure Line where
  slope : Real
  intercept : Real

/-- Represents the configuration of three right triangles relative to a line -/
structure TriangleConfiguration where
  triangles : Fin 3 → RightTriangle
  base_line : Line
  intersecting_line : Line

/-- Checks if a line intersects three triangles into equal segments -/
def intersects_equally (config : TriangleConfiguration) : Prop :=
  sorry

/-- The main theorem -/
theorem triangle_intersection_invariance 
  (initial_config : TriangleConfiguration)
  (rotated_config : TriangleConfiguration)
  (h1 : intersects_equally initial_config)
  (h2 : ∀ i : Fin 3, 
    (initial_config.triangles i).leg1 = (rotated_config.triangles i).leg2 ∧
    (initial_config.triangles i).leg2 = (rotated_config.triangles i).leg1)
  (h3 : initial_config.base_line = rotated_config.base_line) :
  ∃ new_line : Line, 
    new_line.slope = initial_config.intersecting_line.slope ∧
    intersects_equally { triangles := rotated_config.triangles,
                         base_line := rotated_config.base_line,
                         intersecting_line := new_line } :=
sorry

end NUMINAMATH_CALUDE_triangle_intersection_invariance_l2997_299785
