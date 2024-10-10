import Mathlib

namespace intersection_A_B_l2287_228795

def U : Set Nat := {0, 1, 3, 7, 9}
def C_UA : Set Nat := {0, 5, 9}
def B : Set Nat := {3, 5, 7}
def A : Set Nat := U \ C_UA

theorem intersection_A_B : A ∩ B = {3, 7} := by sorry

end intersection_A_B_l2287_228795


namespace prob_different_colors_l2287_228768

/-- Probability of drawing two different colored chips -/
theorem prob_different_colors (blue yellow red : ℕ) 
  (h_blue : blue = 6)
  (h_yellow : yellow = 4)
  (h_red : red = 2) :
  let total := blue + yellow + red
  (blue * yellow + blue * red + yellow * red) * 2 / (total * total) = 11 / 18 := by
  sorry

end prob_different_colors_l2287_228768


namespace parabola_equation_l2287_228797

/-- A parabola is a set of points equidistant from a fixed point (focus) and a fixed line (directrix) -/
def Parabola (focus : ℝ × ℝ) (directrix : ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p focus = |p.2 - directrix p.1|}

theorem parabola_equation (p : ℝ × ℝ) :
  p ∈ Parabola (0, -1) (fun _ ↦ 1) ↔ p.1^2 = -4 * p.2 := by
  sorry

#check parabola_equation

end parabola_equation_l2287_228797


namespace cubic_roots_sum_l2287_228728

theorem cubic_roots_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 10*p - 3 = 0 ∧ 
  q^3 - 8*q^2 + 10*q - 3 = 0 ∧ 
  r^3 - 8*r^2 + 10*r - 3 = 0 → 
  (p / (q*r + 2)) + (q / (p*r + 2)) + (r / (p*q + 2)) = 367/183 := by
sorry

end cubic_roots_sum_l2287_228728


namespace triangle_formation_condition_l2287_228793

theorem triangle_formation_condition (a b : ℝ) : 
  (∃ (c : ℝ), c = 1 ∧ a + b + c = 2) →
  (a + b > c ∧ a + c > b ∧ b + c > a) ↔ (a + b = 1 ∧ a ≥ 0 ∧ b ≥ 0) :=
sorry

end triangle_formation_condition_l2287_228793


namespace smallest_n_for_inequality_l2287_228737

theorem smallest_n_for_inequality : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (a : Fin n → ℝ), (∀ i, 1 < a i ∧ a i < 1000) → (∀ i j, i ≠ j → a i ≠ a j) → 
    ∃ i j, i ≠ j ∧ 0 < a i - a j ∧ a i - a j < 1 + 3 * Real.rpow (a i * a j) (1/3)) ∧
  (∀ m : ℕ, m < n → 
    ∃ (a : Fin m → ℝ), (∀ i, 1 < a i ∧ a i < 1000) ∧ (∀ i j, i ≠ j → a i ≠ a j) ∧
      ∀ i j, i ≠ j → ¬(0 < a i - a j ∧ a i - a j < 1 + 3 * Real.rpow (a i * a j) (1/3))) ∧
  n = 11 :=
sorry

end smallest_n_for_inequality_l2287_228737


namespace parallelogram_smaller_angle_l2287_228774

theorem parallelogram_smaller_angle (smaller_angle larger_angle : ℝ) : 
  larger_angle = smaller_angle + 90 →
  smaller_angle + larger_angle = 180 →
  smaller_angle = 45 := by
  sorry

end parallelogram_smaller_angle_l2287_228774


namespace solve_equation_l2287_228739

theorem solve_equation (x : ℝ) :
  Real.sqrt (3 / x + 3) = 5 / 3 → x = -27 / 2 := by
  sorry

end solve_equation_l2287_228739


namespace ellipse_focus_distance_range_l2287_228727

/-- An ellipse with equation x²/4 + y²/t = 1 -/
structure Ellipse (t : ℝ) where
  x : ℝ
  y : ℝ
  eq : x^2/4 + y^2/t = 1

/-- The distance from a point on the ellipse to one of its foci -/
noncomputable def distance_to_focus (t : ℝ) (e : Ellipse t) : ℝ :=
  sorry  -- Definition omitted as it's not directly given in the problem

/-- The theorem stating the range of t for which the distance to a focus is always greater than 1 -/
theorem ellipse_focus_distance_range :
  ∀ t : ℝ, (∀ e : Ellipse t, distance_to_focus t e > 1) →
    t ∈ Set.union (Set.Ioo 3 4) (Set.Ioo 4 (25/4)) :=
sorry

end ellipse_focus_distance_range_l2287_228727


namespace business_profit_calculation_l2287_228785

-- Define the partners
inductive Partner
| Mary
| Mike
| Anna
| Ben

-- Define the investment amounts
def investment (p : Partner) : ℕ :=
  match p with
  | Partner.Mary => 800
  | Partner.Mike => 200
  | Partner.Anna => 600
  | Partner.Ben => 400

-- Define the profit sharing ratios for the last part
def profit_ratio (p : Partner) : ℕ :=
  match p with
  | Partner.Mary => 2
  | Partner.Mike => 1
  | Partner.Anna => 3
  | Partner.Ben => 4

-- Define the total investment
def total_investment : ℕ := 
  investment Partner.Mary + investment Partner.Mike + 
  investment Partner.Anna + investment Partner.Ben

-- Define the theorem
theorem business_profit_calculation (P : ℚ) : 
  (3 * P / 10 - 3 * P / 20 = 900) ∧ 
  (17 * P / 60 - 13 * P / 60 = 600) → 
  P = 6000 := by
sorry

end business_profit_calculation_l2287_228785


namespace hotel_profit_equation_l2287_228787

/-- Represents a hotel's pricing and occupancy model -/
structure Hotel where
  totalRooms : ℕ
  basePrice : ℕ
  priceStep : ℕ
  costPerRoom : ℕ

/-- Calculates the number of occupied rooms based on the current price -/
def occupiedRooms (h : Hotel) (price : ℕ) : ℕ :=
  h.totalRooms - (price - h.basePrice) / h.priceStep

/-- Calculates the profit for a given price -/
def profit (h : Hotel) (price : ℕ) : ℕ :=
  (price - h.costPerRoom) * occupiedRooms h price

/-- Theorem stating that the given equation correctly represents the hotel's profit -/
theorem hotel_profit_equation (desiredProfit : ℕ) :
  let h : Hotel := {
    totalRooms := 50,
    basePrice := 180,
    priceStep := 10,
    costPerRoom := 20
  }
  ∀ x : ℕ, profit h x = desiredProfit ↔ (x - 20) * (50 - (x - 180) / 10) = desiredProfit :=
by sorry

end hotel_profit_equation_l2287_228787


namespace unique_perfect_square_solution_l2287_228751

/-- A number is a perfect square if it's equal to some integer squared. -/
def IsPerfectSquare (x : ℤ) : Prop := ∃ k : ℤ, x = k^2

/-- The theorem states that 125 is the only positive integer n such that
    both 20n and 5n + 275 are perfect squares. -/
theorem unique_perfect_square_solution :
  ∀ n : ℕ+, (IsPerfectSquare (20 * n.val)) ∧ (IsPerfectSquare (5 * n.val + 275)) ↔ n = 125 := by
  sorry

end unique_perfect_square_solution_l2287_228751


namespace land_reaping_l2287_228714

/-- Given that 4 men can reap 40 acres in 15 days, prove that 16 men can reap 320 acres in 30 days. -/
theorem land_reaping (men_initial : ℕ) (acres_initial : ℕ) (days_initial : ℕ)
                     (men_final : ℕ) (days_final : ℕ) :
  men_initial = 4 →
  acres_initial = 40 →
  days_initial = 15 →
  men_final = 16 →
  days_final = 30 →
  (men_final * days_final * acres_initial) / (men_initial * days_initial) = 320 := by
  sorry

#check land_reaping

end land_reaping_l2287_228714


namespace meaningful_expression_l2287_228753

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 1)) ↔ x > 1 := by
  sorry

end meaningful_expression_l2287_228753


namespace solve_for_c_l2287_228724

theorem solve_for_c (y : ℝ) (c : ℝ) (h1 : y > 0) (h2 : (6 * y) / 20 + (c * y) / 10 = 0.6 * y) : c = 3 := by
  sorry

end solve_for_c_l2287_228724


namespace tan_theta_solution_l2287_228757

theorem tan_theta_solution (θ : Real) (h1 : 0 < θ * (180 / Real.pi)) 
  (h2 : θ * (180 / Real.pi) < 30) 
  (h3 : Real.tan θ + Real.tan (4 * θ) + Real.tan (6 * θ) = 0) : 
  Real.tan θ = 1 / Real.sqrt 3 := by
sorry

end tan_theta_solution_l2287_228757


namespace canoe_kayak_difference_l2287_228715

/-- Represents the rental information for canoes and kayaks --/
structure RentalInfo where
  canoe_cost : ℕ  -- Cost of renting a canoe per day
  kayak_cost : ℕ  -- Cost of renting a kayak per day
  canoe_kayak_ratio : Rat  -- Ratio of canoes to kayaks rented
  total_revenue : ℕ  -- Total revenue from rentals

/-- 
Given rental information, proves that the difference between 
the number of canoes and kayaks rented is 4
--/
theorem canoe_kayak_difference (info : RentalInfo) 
  (h1 : info.canoe_cost = 14)
  (h2 : info.kayak_cost = 15)
  (h3 : info.canoe_kayak_ratio = 3/2)
  (h4 : info.total_revenue = 288) :
  ∃ (c k : ℕ), c = k + 4 ∧ 
    c * info.canoe_cost + k * info.kayak_cost = info.total_revenue ∧
    (c : Rat) / k = info.canoe_kayak_ratio := by
  sorry

end canoe_kayak_difference_l2287_228715


namespace solve_diamond_equation_l2287_228729

-- Define the binary operation ◊
noncomputable def diamond (a b : ℝ) : ℝ := a / b

-- State the properties of the operation
axiom diamond_prop1 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  diamond a (diamond b c) = diamond a b * c

axiom diamond_prop2 (a : ℝ) (ha : a ≠ 0) :
  diamond a a = 1

-- State the theorem to be proved
theorem solve_diamond_equation :
  ∃ x : ℝ, x ≠ 0 ∧ diamond 2016 (diamond 6 x) = 100 ∧ x = 25 / 84 := by
  sorry

end solve_diamond_equation_l2287_228729


namespace square_fence_perimeter_l2287_228798

theorem square_fence_perimeter 
  (num_posts : ℕ) 
  (post_width_inches : ℝ) 
  (gap_between_posts_feet : ℝ) : 
  num_posts = 36 →
  post_width_inches = 6 →
  gap_between_posts_feet = 8 →
  let posts_per_side : ℕ := num_posts / 4
  let gaps_per_side : ℕ := posts_per_side - 1
  let total_gap_length : ℝ := (gaps_per_side : ℝ) * gap_between_posts_feet
  let post_width_feet : ℝ := post_width_inches / 12
  let total_post_width : ℝ := (posts_per_side : ℝ) * post_width_feet
  let side_length : ℝ := total_gap_length + total_post_width
  let perimeter : ℝ := 4 * side_length
  perimeter = 242 := by
sorry

end square_fence_perimeter_l2287_228798


namespace solve_system_l2287_228732

theorem solve_system (x y : ℝ) 
  (eq1 : 3 * x - y = 7)
  (eq2 : x + 3 * y = 7) :
  x = 2.8 := by
sorry

end solve_system_l2287_228732


namespace absolute_value_equation_l2287_228704

theorem absolute_value_equation (x z : ℝ) (h : |2*x - Real.sqrt z| = 2*x + Real.sqrt z) :
  x ≥ 0 ∧ z = 0 := by
  sorry

end absolute_value_equation_l2287_228704


namespace dry_mixed_fruits_weight_l2287_228775

/-- Calculates the weight of dry mixed fruits after dehydration -/
def weight_dry_mixed_fruits (fresh_grapes fresh_apples : ℝ) 
  (fresh_grapes_water fresh_apples_water : ℝ) : ℝ :=
  (1 - fresh_grapes_water) * fresh_grapes + (1 - fresh_apples_water) * fresh_apples

/-- Theorem: The weight of dry mixed fruits is 188 kg -/
theorem dry_mixed_fruits_weight :
  weight_dry_mixed_fruits 400 300 0.65 0.84 = 188 := by
  sorry

#eval weight_dry_mixed_fruits 400 300 0.65 0.84

end dry_mixed_fruits_weight_l2287_228775


namespace borya_segments_imply_isosceles_l2287_228764

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- A segment represented by its length. -/
structure Segment where
  length : ℝ

/-- The set of nine segments drawn by Borya. -/
def BoryaSegments : Set Segment := sorry

/-- The three altitudes of the triangle. -/
def altitudes (t : Triangle) : Set Segment := sorry

/-- The three angle bisectors of the triangle. -/
def angleBisectors (t : Triangle) : Set Segment := sorry

/-- The three medians of the triangle. -/
def medians (t : Triangle) : Set Segment := sorry

/-- Predicate to check if a triangle is isosceles. -/
def isIsosceles (t : Triangle) : Prop := sorry

theorem borya_segments_imply_isosceles (t : Triangle) 
  (h1 : ∃ s1 s2 s3 : Segment, s1 ∈ BoryaSegments ∧ s2 ∈ BoryaSegments ∧ s3 ∈ BoryaSegments ∧ 
                              {s1, s2, s3} = altitudes t)
  (h2 : ∃ s1 s2 s3 : Segment, s1 ∈ BoryaSegments ∧ s2 ∈ BoryaSegments ∧ s3 ∈ BoryaSegments ∧ 
                              {s1, s2, s3} = angleBisectors t)
  (h3 : ∃ s1 s2 s3 : Segment, s1 ∈ BoryaSegments ∧ s2 ∈ BoryaSegments ∧ s3 ∈ BoryaSegments ∧ 
                              {s1, s2, s3} = medians t)
  (h4 : ∀ s ∈ BoryaSegments, ∃ s' ∈ BoryaSegments, s ≠ s' ∧ s.length = s'.length) :
  isIsosceles t :=
sorry

end borya_segments_imply_isosceles_l2287_228764


namespace system_two_solutions_l2287_228713

-- Define the system of equations
def system (a x y : ℝ) : Prop :=
  (x - a)^2 + y^2 = 64 ∧ (|x| - 8)^2 + (|y| - 15)^2 = 289

-- Define the set of values for parameter a
def valid_a_set : Set ℝ :=
  {-28} ∪ Set.Ioc (-24) (-8) ∪ Set.Ico 8 24 ∪ {28}

-- Theorem statement
theorem system_two_solutions (a : ℝ) :
  (∃! x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∨ y₁ ≠ y₂ ∧ system a x₁ y₁ ∧ system a x₂ y₂) ↔
  a ∈ valid_a_set :=
sorry

end system_two_solutions_l2287_228713


namespace volleyball_team_selection_l2287_228703

def total_players : ℕ := 16
def triplets : ℕ := 3
def team_size : ℕ := 7

def choose_with_triplets (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem volleyball_team_selection :
  (choose_with_triplets 3 1 * choose_with_triplets 13 6) +
  (choose_with_triplets 3 2 * choose_with_triplets 13 5) +
  (choose_with_triplets 3 3 * choose_with_triplets 13 4) = 9724 :=
by
  sorry

#check volleyball_team_selection

end volleyball_team_selection_l2287_228703


namespace cube_sum_greater_than_product_sum_l2287_228745

theorem cube_sum_greater_than_product_sum {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 := by
sorry

end cube_sum_greater_than_product_sum_l2287_228745


namespace a_less_than_abs_a_implies_negative_l2287_228708

theorem a_less_than_abs_a_implies_negative (a : ℝ) : a < |a| → a < 0 := by
  sorry

end a_less_than_abs_a_implies_negative_l2287_228708


namespace sum_of_roots_l2287_228744

theorem sum_of_roots (α β : ℝ) 
  (hα : α^3 - 3*α^2 + 5*α - 4 = 0)
  (hβ : β^3 - 3*β^2 + 5*β - 2 = 0) : 
  α + β = 2 := by
sorry

end sum_of_roots_l2287_228744


namespace dividend_calculation_l2287_228730

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 14)
  (h2 : quotient = 9)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 131 := by
  sorry

end dividend_calculation_l2287_228730


namespace quadratic_factorization_l2287_228718

theorem quadratic_factorization (c d : ℤ) :
  (∀ x : ℝ, (5*x + c) * (5*x + d) = 25*x^2 - 135*x - 150) →
  c + 2*d = -59 := by
sorry

end quadratic_factorization_l2287_228718


namespace min_distance_between_curves_l2287_228767

/-- The minimum distance between two points on different curves -/
theorem min_distance_between_curves (m : ℝ) : 
  let A := {x : ℝ | ∃ y, y = m ∧ y = 2 * (x + 1)}
  let B := {x : ℝ | ∃ y, y = m ∧ y = x + Real.log x}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ A ∧ x₂ ∈ B ∧ 
    (∀ (a b : ℝ), a ∈ A → b ∈ B → |x₂ - x₁| ≤ |b - a|) ∧
    |x₂ - x₁| = 3/2 :=
by sorry

end min_distance_between_curves_l2287_228767


namespace composite_divides_factorial_l2287_228701

theorem composite_divides_factorial (n : ℕ) (h1 : n > 4) (h2 : ¬ Nat.Prime n) :
  n ∣ Nat.factorial (n - 1) := by
  sorry

end composite_divides_factorial_l2287_228701


namespace division_problem_l2287_228759

theorem division_problem (n t : ℝ) (hn : n > 0) (ht : t > 0) 
  (h : n / t = (n + 2) / (t + 7)) : 
  ∃ z, n / t = (n + 3) / (t + z) ∧ z = 21 / 2 := by
  sorry

end division_problem_l2287_228759


namespace fixed_point_on_line_l2287_228762

theorem fixed_point_on_line (m : ℝ) : (2 * m + 1) * 3 + (m + 1) * 1 - 7 * m - 4 = 0 := by
  sorry

#check fixed_point_on_line

end fixed_point_on_line_l2287_228762


namespace max_handshakes_l2287_228746

/-- In a group of N people (N > 5), if at least two people have not shaken hands with everyone else,
    then the maximum number of people who could have shaken hands with every other person is N-2. -/
theorem max_handshakes (N : ℕ) (h1 : N > 5) :
  ∃ (max : ℕ), max = N - 2 ∧
  ∀ (shaken : Fin N → Fin N → Bool),
    (∃ (p1 p2 : Fin N), p1 ≠ p2 ∧
      (∃ (q : Fin N), shaken p1 q = false ∧ shaken p2 q = false)) →
    (∀ (p : Fin N), (∀ (q : Fin N), p ≠ q → shaken p q = true) → p.val < max) :=
sorry

end max_handshakes_l2287_228746


namespace octal_734_eq_476_l2287_228765

-- Define the octal number as a list of digits
def octal_number : List Nat := [7, 3, 4]

-- Define the function to convert octal to decimal
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Theorem statement
theorem octal_734_eq_476 :
  octal_to_decimal octal_number = 476 := by
  sorry

end octal_734_eq_476_l2287_228765


namespace min_correct_answers_l2287_228784

theorem min_correct_answers (total : Nat) (a b c d : Nat)
  (h_total : total = 15)
  (h_a : a = 11)
  (h_b : b = 12)
  (h_c : c = 13)
  (h_d : d = 14)
  (h_a_le : a ≤ total)
  (h_b_le : b ≤ total)
  (h_c_le : c ≤ total)
  (h_d_le : d ≤ total) :
  ∃ (x : Nat), x = min a (min b (min c d)) ∧ x ≥ 5 :=
sorry

end min_correct_answers_l2287_228784


namespace safe_locks_theorem_l2287_228778

/-- The number of people in the commission -/
def n : ℕ := 9

/-- The minimum number of people required to access the safe -/
def k : ℕ := 6

/-- The number of keys per lock -/
def keys_per_lock : ℕ := n - k + 1

/-- The number of locks required -/
def num_locks : ℕ := Nat.choose n keys_per_lock

theorem safe_locks_theorem : 
  num_locks = Nat.choose n keys_per_lock :=
by sorry

end safe_locks_theorem_l2287_228778


namespace smallest_n_divisible_sixty_satisfies_conditions_sixty_is_smallest_smallest_n_is_sixty_l2287_228755

theorem smallest_n_divisible (n : ℕ) : n > 0 ∧ 45 ∣ n^2 ∧ 1152 ∣ n^4 → n ≥ 60 :=
by sorry

theorem sixty_satisfies_conditions : 45 ∣ 60^2 ∧ 1152 ∣ 60^4 :=
by sorry

theorem sixty_is_smallest : ∀ m : ℕ, m > 0 ∧ 45 ∣ m^2 ∧ 1152 ∣ m^4 → m ≥ 60 :=
by sorry

theorem smallest_n_is_sixty : ∃! n : ℕ, n > 0 ∧ 45 ∣ n^2 ∧ 1152 ∣ n^4 ∧ ∀ m : ℕ, (m > 0 ∧ 45 ∣ m^2 ∧ 1152 ∣ m^4 → m ≥ n) :=
by sorry

end smallest_n_divisible_sixty_satisfies_conditions_sixty_is_smallest_smallest_n_is_sixty_l2287_228755


namespace triangle_angle_inequalities_l2287_228717

theorem triangle_angle_inequalities (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) : 
  (Real.sin α * Real.sin β * Real.sin γ ≤ 3 * Real.sqrt 3 / 8) ∧
  (Real.cos (α/2) * Real.cos (β/2) * Real.cos (γ/2) ≤ 3 * Real.sqrt 3 / 8) := by
  sorry

end triangle_angle_inequalities_l2287_228717


namespace puzzle_solution_l2287_228725

theorem puzzle_solution (x y z w : ℕ+) 
  (h1 : x^3 = y^2) 
  (h2 : z^4 = w^3) 
  (h3 : z - x = 17) : 
  w - y = 229 := by
  sorry

end puzzle_solution_l2287_228725


namespace polynomial_characterization_l2287_228735

-- Define a real polynomial
def RealPolynomial := ℝ → ℝ

-- Define the condition for a, b, c
def SatisfiesCondition (a b c : ℝ) : Prop := a * b + b * c + c * a = 0

-- Define the property that P must satisfy
def SatisfiesProperty (P : RealPolynomial) : Prop :=
  ∀ a b c : ℝ, SatisfiesCondition a b c →
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)

-- Define the form of the polynomial we want to prove
def IsQuarticQuadratic (P : RealPolynomial) : Prop :=
  ∃ α β : ℝ, ∀ x : ℝ, P x = α * x^4 + β * x^2

-- The main theorem
theorem polynomial_characterization (P : RealPolynomial) :
  SatisfiesProperty P → IsQuarticQuadratic P :=
sorry

end polynomial_characterization_l2287_228735


namespace average_of_multiples_of_10_l2287_228789

def multiples_of_10 : List ℕ := [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

theorem average_of_multiples_of_10 : 
  (List.sum multiples_of_10) / (List.length multiples_of_10) = 55 := by
  sorry

end average_of_multiples_of_10_l2287_228789


namespace fencing_cost_calculation_l2287_228707

-- Define the ratio of the sides
def side_ratio : ℚ := 3 / 4

-- Define the area of the field in square meters
def field_area : ℝ := 7500

-- Define the cost of fencing in paise per meter
def fencing_cost_paise : ℝ := 25

-- Theorem statement
theorem fencing_cost_calculation :
  let length : ℝ := Real.sqrt (field_area * side_ratio / (side_ratio + 1))
  let width : ℝ := length / side_ratio
  let perimeter : ℝ := 2 * (length + width)
  let total_cost : ℝ := perimeter * fencing_cost_paise / 100
  total_cost = 87.5 := by sorry

end fencing_cost_calculation_l2287_228707


namespace inequality_proof_l2287_228712

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  (a * b / c) + (b * c / a) + (c * a / b) ≥ Real.sqrt 3 := by
  sorry

end inequality_proof_l2287_228712


namespace range_of_f_l2287_228763

def f (x : ℕ) : ℤ := x^2 - 2*x

def domain : Set ℕ := {0, 1, 2, 3}

theorem range_of_f : 
  {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end range_of_f_l2287_228763


namespace absolute_value_inequality_l2287_228779

theorem absolute_value_inequality (x : ℝ) : |x - 2| > 2 - x ↔ x > 2 := by
  sorry

end absolute_value_inequality_l2287_228779


namespace problem_solution_l2287_228758

theorem problem_solution (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = 5) : 
  (2 / x + 2 / y = 12 / 5) ∧ ((x - y)^2 = 16) ∧ (x^2 + y^2 = 26) := by
sorry

end problem_solution_l2287_228758


namespace volume_not_occupied_by_cones_l2287_228752

/-- The volume of a cylinder not occupied by two identical cones -/
theorem volume_not_occupied_by_cones (r h_cyl h_cone : ℝ) 
  (hr : r = 10)
  (h_cyl_height : h_cyl = 30)
  (h_cone_height : h_cone = 15) :
  π * r^2 * h_cyl - 2 * (1/3 * π * r^2 * h_cone) = 2000 * π := by
  sorry

end volume_not_occupied_by_cones_l2287_228752


namespace point_coordinates_wrt_origin_l2287_228772

/-- In a Cartesian coordinate system, if a point P has coordinates (3, -5),
    then its coordinates with respect to the origin are also (3, -5). -/
theorem point_coordinates_wrt_origin :
  ∀ (P : ℝ × ℝ), P = (3, -5) → P = (3, -5) := by
  sorry

end point_coordinates_wrt_origin_l2287_228772


namespace numbered_cube_sum_l2287_228747

/-- Represents a cube with numbered faces -/
structure NumberedCube where
  numbers : Fin 6 → ℕ
  consecutive_even : ∀ i : Fin 5, numbers i.succ = numbers i + 2
  smallest_is_12 : numbers 0 = 12
  opposite_faces_sum_equal : 
    numbers 0 + numbers 5 = numbers 1 + numbers 4 ∧ 
    numbers 1 + numbers 4 = numbers 2 + numbers 3

/-- The sum of all numbers on the cube is 102 -/
theorem numbered_cube_sum (cube : NumberedCube) : 
  (Finset.univ.sum cube.numbers) = 102 := by
  sorry

end numbered_cube_sum_l2287_228747


namespace solution_and_parabola_equivalence_l2287_228760

-- Define the set of solutions for x - 3 > 0
def solution_set : Set ℝ := {x | x - 3 > 0}

-- Define the set of points on the parabola y = x^2 - 1
def parabola_points : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2 - 1}

theorem solution_and_parabola_equivalence :
  (solution_set = {x : ℝ | x > 3}) ∧
  (parabola_points = {p : ℝ × ℝ | p.2 = p.1^2 - 1}) := by
  sorry

end solution_and_parabola_equivalence_l2287_228760


namespace connors_date_cost_is_36_l2287_228740

/-- The cost of Connor's movie date --/
def connors_date_cost : ℝ :=
  let ticket_price : ℝ := 10
  let ticket_quantity : ℕ := 2
  let combo_meal_price : ℝ := 11
  let candy_price : ℝ := 2.5
  let candy_quantity : ℕ := 2
  ticket_price * ticket_quantity + combo_meal_price + candy_price * candy_quantity

/-- Theorem stating the total cost of Connor's date --/
theorem connors_date_cost_is_36 : connors_date_cost = 36 := by
  sorry

end connors_date_cost_is_36_l2287_228740


namespace mango_rate_per_kg_mango_rate_proof_l2287_228769

/-- The rate of mangoes per kilogram given the purchase conditions --/
theorem mango_rate_per_kg : ℝ → Prop :=
  fun rate =>
    let grape_quantity : ℝ := 8
    let grape_rate : ℝ := 70
    let mango_quantity : ℝ := 9
    let total_paid : ℝ := 1145
    grape_quantity * grape_rate + mango_quantity * rate = total_paid →
    rate = 65

/-- Proof of the mango rate per kilogram --/
theorem mango_rate_proof : mango_rate_per_kg 65 := by
  sorry

end mango_rate_per_kg_mango_rate_proof_l2287_228769


namespace equal_integers_from_gcd_l2287_228790

theorem equal_integers_from_gcd (a b : ℤ) 
  (h : ∀ (n : ℤ), n ≥ 1 → Nat.gcd (Int.natAbs (a + n)) (Int.natAbs (b + n)) > 1) : 
  a = b := by
  sorry

end equal_integers_from_gcd_l2287_228790


namespace smaller_is_999_l2287_228788

/-- Two 3-digit positive integers whose average equals their decimal concatenation -/
structure SpecialIntegerPair where
  m : ℕ
  n : ℕ
  m_three_digit : 100 ≤ m ∧ m ≤ 999
  n_three_digit : 100 ≤ n ∧ n ≤ 999
  avg_eq_concat : (m + n) / 2 = m + n / 1000

/-- The smaller of the two integers in a SpecialIntegerPair is 999 -/
theorem smaller_is_999 (pair : SpecialIntegerPair) : min pair.m pair.n = 999 := by
  sorry

end smaller_is_999_l2287_228788


namespace share_price_increase_l2287_228723

theorem share_price_increase (initial_price : ℝ) (q1_increase : ℝ) (q2_increase : ℝ) :
  q1_increase = 0.25 →
  q2_increase = 0.44 →
  ((initial_price * (1 + q1_increase) * (1 + q2_increase) - initial_price) / initial_price) = 0.80 :=
by sorry

end share_price_increase_l2287_228723


namespace continuous_function_property_l2287_228756

open Real Set

theorem continuous_function_property (d : ℝ) (h_d : d ∈ Ioc 0 1) :
  (∀ f : ℝ → ℝ, ContinuousOn f (Icc 0 1) → f 0 = f 1 →
    ∃ x₀ ∈ Icc 0 (1 - d), f x₀ = f (x₀ + d)) ↔
  ∃ k : ℕ, d = 1 / k :=
by sorry


end continuous_function_property_l2287_228756


namespace factor_expression_l2287_228700

theorem factor_expression (x : ℝ) : 84 * x^7 - 270 * x^13 = 6 * x^7 * (14 - 45 * x^6) := by
  sorry

end factor_expression_l2287_228700


namespace largest_house_number_l2287_228780

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if all digits in a natural number are distinct -/
def has_distinct_digits (n : ℕ) : Prop := sorry

/-- The largest 3-digit number with distinct digits whose sum equals the sum of digits in 5039821 -/
theorem largest_house_number : 
  ∃ (n : ℕ), 
    100 ≤ n ∧ n < 1000 ∧ 
    has_distinct_digits n ∧
    digit_sum n = digit_sum 5039821 ∧
    ∀ (m : ℕ), 100 ≤ m ∧ m < 1000 ∧ 
      has_distinct_digits m ∧ 
      digit_sum m = digit_sum 5039821 → 
      m ≤ n ∧
    n = 981 := by sorry

end largest_house_number_l2287_228780


namespace angle_minus_510_in_third_quadrant_l2287_228776

-- Define the function to convert an angle to its equivalent within 0° to 360°
def convertAngle (angle : Int) : Int :=
  angle % 360

-- Define the function to determine the quadrant of an angle
def getQuadrant (angle : Int) : Nat :=
  let convertedAngle := convertAngle angle
  if 0 ≤ convertedAngle ∧ convertedAngle < 90 then 1
  else if 90 ≤ convertedAngle ∧ convertedAngle < 180 then 2
  else if 180 ≤ convertedAngle ∧ convertedAngle < 270 then 3
  else 4

-- Theorem statement
theorem angle_minus_510_in_third_quadrant :
  getQuadrant (-510) = 3 := by
  sorry

end angle_minus_510_in_third_quadrant_l2287_228776


namespace total_angles_count_l2287_228754

/-- The number of 90° angles in a rectangle -/
def rectangleAngles : ℕ := 4

/-- The number of 90° angles in a square -/
def squareAngles : ℕ := 4

/-- The number of rectangular flower beds in the park -/
def flowerBeds : ℕ := 3

/-- The number of square goal areas in the football field -/
def goalAreas : ℕ := 4

/-- The total number of 90° angles in the park and football field -/
def totalAngles : ℕ := 
  rectangleAngles + flowerBeds * rectangleAngles + 
  squareAngles + goalAreas * squareAngles

theorem total_angles_count : totalAngles = 36 := by
  sorry

end total_angles_count_l2287_228754


namespace plot_size_in_acres_l2287_228738

-- Define the scale of the map
def map_scale : ℝ := 1

-- Define the dimensions of the plot on the map
def map_length : ℝ := 20
def map_width : ℝ := 25

-- Define the conversion from square miles to acres
def acres_per_square_mile : ℝ := 640

-- State the theorem
theorem plot_size_in_acres :
  let real_area : ℝ := map_length * map_width * map_scale^2
  real_area * acres_per_square_mile = 320000 := by
  sorry

end plot_size_in_acres_l2287_228738


namespace ferry_tourist_count_l2287_228761

/-- The number of trips the ferry makes -/
def num_trips : ℕ := 7

/-- The number of tourists on the first trip -/
def initial_tourists : ℕ := 100

/-- The decrease in tourists per trip -/
def tourist_decrease : ℕ := 2

/-- The total number of tourists transported -/
def total_tourists : ℕ := 
  (num_trips * (2 * initial_tourists - (num_trips - 1) * tourist_decrease)) / 2

theorem ferry_tourist_count : total_tourists = 658 := by
  sorry

end ferry_tourist_count_l2287_228761


namespace hyperbola_dot_product_theorem_l2287_228773

/-- The hyperbola in the Cartesian coordinate system -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := (-2, 0)

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := (2, 0)

/-- A point on the hyperbola -/
structure HyperbolaPoint where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola x y

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The vector from one point to another -/
def vector (a b : ℝ × ℝ) : ℝ × ℝ := (b.1 - a.1, b.2 - a.2)

/-- The theorem to be proved -/
theorem hyperbola_dot_product_theorem 
  (P Q : HyperbolaPoint) 
  (h_line : ∃ (m b : ℝ), P.y = m * P.x + b ∧ Q.y = m * Q.x + b ∧ F₁.2 = m * F₁.1 + b) 
  (h_dot_product : dot_product (vector F₁ F₂) (vector F₁ (P.x, P.y)) = 16) :
  dot_product (vector F₂ (P.x, P.y)) (vector F₂ (Q.x, Q.y)) = 27 / 13 := by
  sorry


end hyperbola_dot_product_theorem_l2287_228773


namespace two_valid_configurations_l2287_228786

/-- Represents a square piece of the figure -/
inductive Square
| Base
| A
| B
| C
| D
| E
| F
| G

/-- Represents the L-shaped figure -/
def LShape := List Square

/-- Represents a configuration of squares -/
def Configuration := List Square

/-- Checks if a configuration can form a topless cubical box -/
def is_valid_box (config : Configuration) : Prop :=
  sorry

/-- The set of all possible configurations -/
def all_configurations : Set Configuration :=
  sorry

/-- The number of valid configurations that form a topless cubical box -/
def num_valid_configurations : ℕ :=
  sorry

/-- Theorem stating that there are exactly two valid configurations -/
theorem two_valid_configurations :
  num_valid_configurations = 2 :=
sorry

end two_valid_configurations_l2287_228786


namespace repeating_decimal_sum_l2287_228799

theorem repeating_decimal_sum (a b c d : ℕ) : 
  (a ≤ 9) → (b ≤ 9) → (c ≤ 9) → (d ≤ 9) →
  ((10 * a + c) / 99 + (1000 * a + 100 * b + 10 * c + d) / 9999 = 17 / 37) →
  (a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 5) := by
sorry

end repeating_decimal_sum_l2287_228799


namespace min_m_is_one_l2287_228702

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem min_m_is_one (f g h : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = Real.exp x) →
  (∀ x, f x = g x - h x) →
  IsEven g →
  IsOdd h →
  (∀ x ∈ Set.Icc (-1) 1, m * g x + h x ≥ 0) →
  (∀ m' : ℝ, (∀ x ∈ Set.Icc (-1) 1, m' * g x + h x ≥ 0) → m' ≥ m) →
  m = 1 := by sorry

end min_m_is_one_l2287_228702


namespace zeroth_power_of_nonzero_rational_l2287_228716

theorem zeroth_power_of_nonzero_rational (r : ℚ) (h : r ≠ 0) : r^0 = 1 := by
  sorry

end zeroth_power_of_nonzero_rational_l2287_228716


namespace hyperbola_slope_product_l2287_228771

/-- Hyperbola theorem -/
theorem hyperbola_slope_product (a b x₀ y₀ : ℝ) (ha : a > 0) (hb : b > 0) 
  (hp : x₀^2 / a^2 - y₀^2 / b^2 = 1) (hx : x₀ ≠ a ∧ x₀ ≠ -a) : 
  (y₀ / (x₀ + a)) * (y₀ / (x₀ - a)) = b^2 / a^2 := by
sorry

end hyperbola_slope_product_l2287_228771


namespace arithmetic_sequence_sum_nine_l2287_228722

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence with a_5 = 2, S_9 = 18 -/
theorem arithmetic_sequence_sum_nine 
  (seq : ArithmeticSequence) 
  (h : seq.a 5 = 2) : 
  seq.S 9 = 18 := by
  sorry


end arithmetic_sequence_sum_nine_l2287_228722


namespace problem_solution_l2287_228710

theorem problem_solution (x : ℝ) (n : ℝ) (h1 : x > 0) 
  (h2 : x / n + x / 25 = 0.24000000000000004 * x) : n = 5 := by
  sorry

end problem_solution_l2287_228710


namespace gcd_upper_bound_from_lcm_lower_bound_l2287_228766

theorem gcd_upper_bound_from_lcm_lower_bound 
  (a b : ℕ) 
  (ha : a < 10^7) 
  (hb : b < 10^7) 
  (hlcm : 10^11 ≤ Nat.lcm a b) : 
  Nat.gcd a b < 1000 := by
sorry

end gcd_upper_bound_from_lcm_lower_bound_l2287_228766


namespace binomial_12_10_l2287_228743

theorem binomial_12_10 : Nat.choose 12 10 = 66 := by
  sorry

end binomial_12_10_l2287_228743


namespace repeating_decimal_47_l2287_228734

/-- The repeating decimal 0.474747... is equal to 47/99 -/
theorem repeating_decimal_47 : ∀ (x : ℚ), (∃ (n : ℕ), x * 10^n = ⌊x * 10^n⌋ + 0.47) → x = 47 / 99 := by
  sorry

end repeating_decimal_47_l2287_228734


namespace forest_ecosystem_l2287_228792

/-- Given a forest ecosystem where:
    - Each bird eats 12 beetles per day
    - Each snake eats 3 birds per day
    - Each jaguar eats 5 snakes per day
    - The jaguars in the forest eat 1080 beetles each day
    This theorem proves that there are 6 jaguars in the forest. -/
theorem forest_ecosystem (beetles_per_bird : ℕ) (birds_per_snake : ℕ) (snakes_per_jaguar : ℕ) 
                         (total_beetles_eaten : ℕ) : ℕ :=
  sorry

end forest_ecosystem_l2287_228792


namespace certain_number_divisor_l2287_228750

theorem certain_number_divisor : ∃ n : ℕ, 
  n > 1 ∧ 
  n < 509 - 5 ∧ 
  (509 - 5) % n = 0 ∧ 
  ∀ m : ℕ, m > n → m < 509 - 5 → (509 - 5) % m ≠ 0 ∧
  ∀ k : ℕ, k < 5 → (509 - k) % n ≠ 0 :=
by sorry

end certain_number_divisor_l2287_228750


namespace inverse_of_5_mod_33_l2287_228791

theorem inverse_of_5_mod_33 : ∃ x : ℕ, x < 33 ∧ (5 * x) % 33 = 1 := by
  use 20
  sorry

end inverse_of_5_mod_33_l2287_228791


namespace base_conversion_theorem_l2287_228782

/-- Converts a list of digits in a given base to its decimal representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

/-- Checks if a list of digits represents 2017 in the given base -/
def is_2017 (digits : List Nat) (base : Nat) : Prop :=
  to_decimal digits base = 2017

/-- Checks if a list of digits can have one digit removed to represent 2017 in another base -/
def can_remove_digit_for_2017 (digits : List Nat) (new_base : Nat) : Prop :=
  ∃ (new_digits : List Nat), new_digits.length + 1 = digits.length ∧ 
    (∃ (i : Nat), i < digits.length ∧ new_digits = (digits.take i ++ digits.drop (i+1))) ∧
    is_2017 new_digits new_base

theorem base_conversion_theorem :
  ∃ (a b c : Nat),
    is_2017 [1, 3, 3, 2, 0, 1] a ∧
    can_remove_digit_for_2017 [1, 3, 3, 2, 0, 1] b ∧
    (∃ (digits : List Nat), digits.length = 5 ∧ 
      can_remove_digit_for_2017 digits c ∧
      is_2017 digits b) ∧
    a + b + c = 22 := by
  sorry

end base_conversion_theorem_l2287_228782


namespace regular_polygon_perimeters_l2287_228720

/-- Regular polygon perimeters for a unit circle -/
noncomputable def RegularPolygonPerimeters (n : ℕ) : ℝ × ℝ :=
  sorry

/-- Circumscribed polygon perimeter -/
noncomputable def P (n : ℕ) : ℝ := (RegularPolygonPerimeters n).1

/-- Inscribed polygon perimeter -/
noncomputable def p (n : ℕ) : ℝ := (RegularPolygonPerimeters n).2

theorem regular_polygon_perimeters :
  (P 4 = 8 ∧ p 4 = 4 * Real.sqrt 2 ∧ P 6 = 4 * Real.sqrt 3 ∧ p 6 = 6) ∧
  (∀ n ≥ 3, P (2 * n) = (2 * P n * p n) / (P n + p n) ∧
            p (2 * n) = Real.sqrt (p n * P (2 * n))) ∧
  (3^10 / 71 < Real.pi ∧ Real.pi < 22 / 7) :=
sorry

end regular_polygon_perimeters_l2287_228720


namespace trinomial_cube_l2287_228783

theorem trinomial_cube (x : ℝ) :
  (x^2 - 2*x + 1)^3 = x^6 - 6*x^5 + 15*x^4 - 20*x^3 + 15*x^2 - 6*x + 1 :=
by sorry

end trinomial_cube_l2287_228783


namespace sector_perimeter_l2287_228749

theorem sector_perimeter (θ : Real) (r : Real) (h1 : θ = 54) (h2 : r = 20) : 
  let α := θ * (π / 180)
  let arc_length := α * r
  let perimeter := arc_length + 2 * r
  perimeter = 6 * π + 40 := by sorry

end sector_perimeter_l2287_228749


namespace aquarium_fish_count_l2287_228726

/-- Given an initial number of fish and a number of fish added to an aquarium,
    the total number of fish is equal to the sum of the initial number and the number added. -/
theorem aquarium_fish_count (initial : ℕ) (added : ℕ) :
  initial + added = initial + added :=
by sorry

end aquarium_fish_count_l2287_228726


namespace equation_roots_reciprocal_l2287_228711

theorem equation_roots_reciprocal (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    (a^2 - 1) * x^2 - (a + 1) * x + 1 = 0 ∧ 
    (a^2 - 1) * y^2 - (a + 1) * y + 1 = 0 ∧ 
    x * y = 1) → 
  a = Real.sqrt 2 := by
sorry

end equation_roots_reciprocal_l2287_228711


namespace congruence_solution_l2287_228719

theorem congruence_solution (m : ℕ) : m ∈ Finset.range 47 → (13 * m ≡ 9 [ZMOD 47]) ↔ m = 29 := by
  sorry

end congruence_solution_l2287_228719


namespace triangle_area_l2287_228709

/-- Given a triangle with one side of length 2, a median to this side of length 1,
    and the sum of the other two sides equal to 1 + √3,
    prove that the area of the triangle is √3/2. -/
theorem triangle_area (a b c : ℝ) (h1 : c = 2) (h2 : a + b = 1 + Real.sqrt 3)
  (h3 : ∃ (m : ℝ), m = 1 ∧ m^2 = (a^2 + b^2) / 4 + c^2 / 16) :
  (a * b) / 2 = Real.sqrt 3 / 2 := by
  sorry

end triangle_area_l2287_228709


namespace monochromatic_triangle_exists_l2287_228733

/-- A coloring of the edges of a complete graph using three colors. -/
def ThreeColoring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A complete graph with n vertices. -/
def CompleteGraph (n : ℕ) := Fin n

/-- A triangle in a graph is a set of three distinct vertices. -/
def Triangle (n : ℕ) := { t : Fin n × Fin n × Fin n // t.1 ≠ t.2.1 ∧ t.1 ≠ t.2.2 ∧ t.2.1 ≠ t.2.2 }

/-- A triangle is monochromatic if all its edges have the same color. -/
def IsMonochromatic (n : ℕ) (coloring : ThreeColoring n) (t : Triangle n) : Prop :=
  coloring t.val.1 t.val.2.1 = coloring t.val.1 t.val.2.2 ∧
  coloring t.val.1 t.val.2.1 = coloring t.val.2.1 t.val.2.2

/-- The main theorem: any 3-coloring of K₁₇ contains a monochromatic triangle. -/
theorem monochromatic_triangle_exists :
  ∀ (coloring : ThreeColoring 17),
  ∃ (t : Triangle 17), IsMonochromatic 17 coloring t :=
sorry


end monochromatic_triangle_exists_l2287_228733


namespace two_roots_k_range_l2287_228742

theorem two_roots_k_range (f : ℝ → ℝ) (k : ℝ) :
  (∀ x, f x = x * Real.exp (-2 * x) + k) →
  (∃! x₁ x₂, x₁ ∈ Set.Ioo (-2 : ℝ) 2 ∧ x₂ ∈ Set.Ioo (-2 : ℝ) 2 ∧ x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) →
  k ∈ Set.Ioo (-(1 / (2 * Real.exp 1))) (-(2 / Real.exp 4)) :=
by sorry

end two_roots_k_range_l2287_228742


namespace departmental_store_average_salary_l2287_228748

def average_salary (num_managers : ℕ) (num_associates : ℕ) (avg_salary_managers : ℚ) (avg_salary_associates : ℚ) : ℚ :=
  let total_salary := num_managers * avg_salary_managers + num_associates * avg_salary_associates
  let total_employees := num_managers + num_associates
  total_salary / total_employees

theorem departmental_store_average_salary :
  average_salary 9 18 1300 12000 = 8433.33 := by
  sorry

end departmental_store_average_salary_l2287_228748


namespace smallest_n_with_common_factor_l2287_228794

theorem smallest_n_with_common_factor : 
  ∀ n : ℕ, n > 0 → n < 38 → ¬(∃ k : ℕ, k > 1 ∧ k ∣ (11*n - 3) ∧ k ∣ (8*n + 4)) ∧
  ∃ k : ℕ, k > 1 ∧ k ∣ (11*38 - 3) ∧ k ∣ (8*38 + 4) :=
by sorry

end smallest_n_with_common_factor_l2287_228794


namespace sum_of_roots_cubic_l2287_228796

theorem sum_of_roots_cubic (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - 3*x^2 + 2*x
  (∃ a b c : ℝ, f x = (x - a) * (x - b) * (x - c)) → 
  (a + b + c = 3) :=
sorry

end sum_of_roots_cubic_l2287_228796


namespace quadratic_roots_property_l2287_228736

theorem quadratic_roots_property : 
  ∀ x₁ x₂ : ℝ, 
  x₁^2 - 3*x₁ + 1 = 0 → 
  x₂^2 - 3*x₂ + 1 = 0 → 
  x₁ ≠ x₂ → 
  x₁^2 + 3*x₂ + x₁*x₂ - 2 = 7 := by
sorry

end quadratic_roots_property_l2287_228736


namespace optimal_distribution_minimizes_cost_l2287_228770

noncomputable section

/-- Represents the distribution of potatoes among three farms -/
structure PotatoDistribution where
  farm1 : ℝ
  farm2 : ℝ
  farm3 : ℝ

/-- The cost function for potato distribution -/
def cost (d : PotatoDistribution) : ℝ :=
  4 * d.farm1 + 3 * d.farm2 + d.farm3

/-- Checks if a distribution satisfies all constraints -/
def isValid (d : PotatoDistribution) : Prop :=
  d.farm1 ≥ 0 ∧ d.farm2 ≥ 0 ∧ d.farm3 ≥ 0 ∧
  d.farm1 + d.farm2 + d.farm3 = 12 ∧
  d.farm1 + 4 * d.farm2 + 3 * d.farm3 ≤ 40 ∧
  d.farm1 ≤ 10 ∧ d.farm2 ≤ 8 ∧ d.farm3 ≤ 6

/-- The optimal distribution of potatoes -/
def optimalDistribution : PotatoDistribution :=
  { farm1 := 2/3, farm2 := 16/3, farm3 := 6 }

/-- Theorem stating that the optimal distribution minimizes the cost -/
theorem optimal_distribution_minimizes_cost :
  isValid optimalDistribution ∧
  ∀ d : PotatoDistribution, isValid d → cost optimalDistribution ≤ cost d :=
sorry

end

end optimal_distribution_minimizes_cost_l2287_228770


namespace expand_expression_l2287_228741

theorem expand_expression (a : ℝ) : 4 * a^2 * (3*a - 1) = 12*a^3 - 4*a^2 := by
  sorry

end expand_expression_l2287_228741


namespace age_difference_proof_l2287_228721

def elder_age : ℕ := 30

theorem age_difference_proof (younger_age : ℕ) 
  (h1 : elder_age - 6 = 3 * (younger_age - 6)) : 
  elder_age - younger_age = 16 := by
  sorry

end age_difference_proof_l2287_228721


namespace max_cookies_bound_l2287_228777

def num_jars : Nat := 2023

/-- Represents the state of cookie jars -/
def JarState := Fin num_jars → Nat

/-- Elmo's action of adding cookies to two distinct jars -/
def elmo_action (state : JarState) : JarState := sorry

/-- Cookie Monster's action of eating cookies from the jar with the most cookies -/
def monster_action (state : JarState) : JarState := sorry

/-- One complete cycle of Elmo's and Cookie Monster's actions -/
def cycle (state : JarState) : JarState := monster_action (elmo_action state)

/-- The maximum number of cookies in any jar -/
def max_cookies (state : JarState) : Nat :=
  Finset.sup (Finset.univ : Finset (Fin num_jars)) (fun i => state i)

theorem max_cookies_bound (initial_state : JarState) :
  ∀ n : Nat, max_cookies ((cycle^[n]) initial_state) ≤ 12 := by sorry

end max_cookies_bound_l2287_228777


namespace greatest_divisor_four_consecutive_integers_l2287_228705

theorem greatest_divisor_four_consecutive_integers :
  ∃ (d : ℕ), d > 0 ∧
  (∀ (n : ℕ), n > 0 → (n * (n + 1) * (n + 2) * (n + 3)) % d = 0) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ (m * (m + 1) * (m + 2) * (m + 3)) % k ≠ 0) ∧
  d = 12 :=
by
  sorry

end greatest_divisor_four_consecutive_integers_l2287_228705


namespace target_parabola_satisfies_conditions_l2287_228731

/-- Represents a parabola with specific properties -/
structure Parabola where
  -- Equation coefficients
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  -- Conditions
  passes_through : a * 2^2 + b * 2 * 8 + c * 8^2 + d * 2 + e * 8 + f = 0
  focus_y : ℤ := 4
  vertex_on_y_axis : a * 0^2 + b * 0 * 4 + c * 4^2 + d * 0 + e * 4 + f = 0
  c_positive : c > 0
  gcd_one : Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Int.natAbs c)) (Int.natAbs d)) (Int.natAbs e)) (Int.natAbs f) = 1

/-- The specific parabola we want to prove -/
def target_parabola : Parabola :=
  { a := 0,
    b := 0,
    c := 1,
    d := -8,
    e := -8,
    f := 16,
    passes_through := sorry,
    focus_y := 4,
    vertex_on_y_axis := sorry,
    c_positive := sorry,
    gcd_one := sorry }

/-- Theorem stating that the target parabola satisfies all conditions -/
theorem target_parabola_satisfies_conditions : 
  ∃ (p : Parabola), p = target_parabola := by sorry

end target_parabola_satisfies_conditions_l2287_228731


namespace number_of_grandchildren_excluding_shelby_l2287_228706

/-- Proves the number of grandchildren excluding Shelby, given the inheritance details --/
theorem number_of_grandchildren_excluding_shelby
  (total_inheritance : ℕ)
  (shelby_share : ℕ)
  (remaining_share : ℕ)
  (one_grandchild_share : ℕ)
  (h1 : total_inheritance = 124600)
  (h2 : shelby_share = total_inheritance / 2)
  (h3 : remaining_share = total_inheritance - shelby_share)
  (h4 : one_grandchild_share = 6230)
  (h5 : remaining_share % one_grandchild_share = 0) :
  remaining_share / one_grandchild_share = 10 := by
  sorry

#check number_of_grandchildren_excluding_shelby

end number_of_grandchildren_excluding_shelby_l2287_228706


namespace smallest_number_l2287_228781

theorem smallest_number : ∀ (a b c d : ℝ), 
  a = -2023 → b = 0 → c = 0.999 → d = 1 →
  a < b ∧ a < c ∧ a < d :=
by
  sorry

end smallest_number_l2287_228781
