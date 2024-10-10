import Mathlib

namespace quadratic_equation_solution_l3466_346641

theorem quadratic_equation_solution (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - 4*x₁ - 2*m + 5 = 0 ∧ 
    x₂^2 - 4*x₂ - 2*m + 5 = 0 ∧
    x₁*x₂ + x₁ + x₂ = m^2 + 6) →
  m = 1 := by
sorry

end quadratic_equation_solution_l3466_346641


namespace lucy_current_age_l3466_346603

/-- Lucy's current age -/
def lucy_age : ℕ := sorry

/-- Lovely's current age -/
def lovely_age : ℕ := sorry

/-- Lucy's age was three times Lovely's age 5 years ago -/
axiom past_age_relation : lucy_age - 5 = 3 * (lovely_age - 5)

/-- Lucy's age will be twice Lovely's age 10 years from now -/
axiom future_age_relation : lucy_age + 10 = 2 * (lovely_age + 10)

/-- Lucy's current age is 50 -/
theorem lucy_current_age : lucy_age = 50 := by sorry

end lucy_current_age_l3466_346603


namespace max_square_cookies_l3466_346621

theorem max_square_cookies (length width : ℕ) (h1 : length = 24) (h2 : width = 18) :
  let cookie_size := Nat.gcd length width
  (length / cookie_size) * (width / cookie_size) = 12 :=
by sorry

end max_square_cookies_l3466_346621


namespace range_of_a_l3466_346607

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

theorem range_of_a (a : ℝ) :
  (∀ x y, x < y → x < -1 → y < -1 → f a y < f a x) →
  (∀ x y, x < y → 1 < x → 1 < y → f a x < f a y) →
  a ∈ Set.Icc (-1) 1 :=
by sorry

end range_of_a_l3466_346607


namespace total_baking_time_l3466_346601

def bread_time_1 : ℕ := 375
def bread_time_2 : ℕ := 160
def bread_time_3 : ℕ := 320

theorem total_baking_time :
  max (max bread_time_1 bread_time_2) bread_time_3 = 375 := by
  sorry

end total_baking_time_l3466_346601


namespace stationery_sales_equation_l3466_346680

/-- Represents the sales equation for a stationery store during a promotional event. -/
theorem stationery_sales_equation (x : ℝ) : 
  (1.2 * 0.8 * x + 2 * 0.9 * (60 - x) = 87) ↔ 
  (x ≥ 0 ∧ x ≤ 60 ∧ 
   1.2 * (1 - 0.2) * x + 2 * (1 - 0.1) * (60 - x) = 87) := by
  sorry

#check stationery_sales_equation

end stationery_sales_equation_l3466_346680


namespace car_arrival_delay_l3466_346639

/-- Proves that a car traveling 225 km at 50 kmph instead of 60 kmph arrives 45 minutes later -/
theorem car_arrival_delay (distance : ℝ) (speed1 speed2 : ℝ) :
  distance = 225 →
  speed1 = 60 →
  speed2 = 50 →
  (distance / speed2 - distance / speed1) * 60 = 45 := by
sorry

end car_arrival_delay_l3466_346639


namespace asymptote_sum_l3466_346695

/-- Given an equation y = x / (x^3 + Dx^2 + Ex + F) where D, E, F are integers,
    if the graph has vertical asymptotes at x = -3, 0, and 3,
    then D + E + F = -9 -/
theorem asymptote_sum (D E F : ℤ) : 
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 → 
    ∃ y : ℝ, y = x / (x^3 + D*x^2 + E*x + F)) →
  D + E + F = -9 := by
  sorry

end asymptote_sum_l3466_346695


namespace min_distance_point_to_line_l3466_346606

/-- The minimum distance between a point in the feasible region and a line -/
theorem min_distance_point_to_line :
  ∀ (x y : ℝ),
  (2 * x + y - 4 ≥ 0) →
  (x - y - 2 ≤ 0) →
  (y - 3 ≤ 0) →
  ∃ (x' y' : ℝ),
  (y' = -2 * x' + 2) →
  ∀ (x'' y'' : ℝ),
  (y'' = -2 * x'' + 2) →
  Real.sqrt ((x - x')^2 + (y - y')^2) ≥ (2 * Real.sqrt 5) / 5 :=
by sorry

end min_distance_point_to_line_l3466_346606


namespace star_operation_result_l3466_346664

-- Define the sets M and N
def M : Set ℝ := {y | y ≥ 0}
def N : Set ℝ := {y | -3 ≤ y ∧ y ≤ 3}

-- Define the set difference operation
def setDifference (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- Define the * operation
def starOperation (A B : Set ℝ) : Set ℝ := (setDifference A B) ∪ (setDifference B A)

-- State the theorem
theorem star_operation_result :
  starOperation M N = {x : ℝ | -3 ≤ x ∧ x < 0 ∨ x > 3} := by
  sorry

end star_operation_result_l3466_346664


namespace equation_solution_l3466_346661

theorem equation_solution :
  ∃ x : ℚ, x = -62/29 ∧ (Real.sqrt (7*x + 1) / Real.sqrt (4*(x + 2) - 1) = 3) := by
  sorry

end equation_solution_l3466_346661


namespace abcdef_hex_bits_proof_l3466_346614

/-- The number of bits required to represent ABCDEF₁₆ in binary -/
def abcdef_hex_to_bits : ℕ := 24

/-- The decimal value of ABCDEF₁₆ -/
def abcdef_hex_to_decimal : ℕ := 11293375

theorem abcdef_hex_bits_proof :
  abcdef_hex_to_bits = 24 ∧
  2^23 < abcdef_hex_to_decimal ∧
  abcdef_hex_to_decimal < 2^24 := by
  sorry

#eval abcdef_hex_to_bits
#eval abcdef_hex_to_decimal

end abcdef_hex_bits_proof_l3466_346614


namespace class_b_more_consistent_l3466_346699

/-- Represents the variance of a class's test scores -/
structure ClassVariance where
  value : ℝ
  is_nonneg : value ≥ 0

/-- Determines if one class has more consistent scores than another based on their variances -/
def has_more_consistent_scores (class_a class_b : ClassVariance) : Prop :=
  class_a.value > class_b.value

theorem class_b_more_consistent :
  let class_a : ClassVariance := ⟨2.56, by norm_num⟩
  let class_b : ClassVariance := ⟨1.92, by norm_num⟩
  has_more_consistent_scores class_b class_a := by
  sorry

end class_b_more_consistent_l3466_346699


namespace game_draw_probability_l3466_346626

theorem game_draw_probability (amy_win lily_win eve_win draw : ℚ) : 
  amy_win = 2/5 → lily_win = 1/5 → eve_win = 1/10 → 
  amy_win + lily_win + eve_win + draw = 1 →
  draw = 3/10 := by
sorry

end game_draw_probability_l3466_346626


namespace inscribed_circles_area_ratio_l3466_346678

theorem inscribed_circles_area_ratio : 
  ∀ (R r : ℝ), R > 0 → r > 0 →
  (∃ (s : ℝ), s > 0 ∧ R = (s * Real.sqrt 2) / 2 ∧ r = s / 2) →
  (π * r^2) / (π * R^2) = 1 / 2 := by
sorry

end inscribed_circles_area_ratio_l3466_346678


namespace factorization_coefficient_sum_l3466_346648

theorem factorization_coefficient_sum : 
  ∃ (A B C D E F G H J K : ℤ),
    (125 : ℤ) * X^9 - 216 * Y^9 = 
      (A * X + B * Y) * 
      (C * X^3 + D * X * Y^2 + E * Y^3) * 
      (F * X + G * Y) * 
      (H * X^3 + J * X * Y^2 + K * Y^3) ∧
    A + B + C + D + E + F + G + H + J + K = 24 :=
by sorry

end factorization_coefficient_sum_l3466_346648


namespace alyssas_spending_l3466_346613

/-- Calculates the total spending given an amount paid and a refund. -/
def totalSpending (amountPaid refund : ℚ) : ℚ :=
  amountPaid - refund

/-- Proves that Alyssa's total spending is $2.23 given the conditions. -/
theorem alyssas_spending :
  let grapesPayment : ℚ := 12.08
  let cherriesRefund : ℚ := 9.85
  totalSpending grapesPayment cherriesRefund = 2.23 := by
  sorry

end alyssas_spending_l3466_346613


namespace total_coins_always_odd_never_equal_coins_l3466_346685

/-- Represents the state of Laura's coins -/
structure CoinState where
  red : Nat
  green : Nat

/-- Represents the slot machine operation -/
def slotMachine (state : CoinState) (insertRed : Bool) : CoinState :=
  if insertRed then
    { red := state.red - 1, green := state.green + 5 }
  else
    { red := state.red + 5, green := state.green - 1 }

/-- The initial state of Laura's coins -/
def initialState : CoinState := { red := 0, green := 1 }

/-- Theorem stating that the total number of coins is always odd -/
theorem total_coins_always_odd (state : CoinState) (n : Nat) :
  (state.red + state.green) % 2 = 1 := by
  sorry

/-- Theorem stating that Laura can never have an equal number of red and green coins -/
theorem never_equal_coins (state : CoinState) :
  state.red ≠ state.green := by
  sorry

end total_coins_always_odd_never_equal_coins_l3466_346685


namespace gcd_45736_123456_l3466_346654

theorem gcd_45736_123456 : Nat.gcd 45736 123456 = 352 := by
  sorry

end gcd_45736_123456_l3466_346654


namespace smallest_x_squared_l3466_346677

/-- Represents a trapezoid ABCD with a circle tangent to its sides --/
structure TrapezoidWithCircle where
  AB : ℝ
  CD : ℝ
  x : ℝ
  circle_center_distance : ℝ

/-- The smallest possible value of x in the trapezoid configuration --/
def smallest_x (t : TrapezoidWithCircle) : ℝ := sorry

/-- Main theorem: The square of the smallest possible x is 256 --/
theorem smallest_x_squared (t : TrapezoidWithCircle) 
  (h1 : t.AB = 70)
  (h2 : t.CD = 25)
  (h3 : t.circle_center_distance = 10) :
  (smallest_x t)^2 = 256 := by sorry

end smallest_x_squared_l3466_346677


namespace project_hours_theorem_l3466_346637

theorem project_hours_theorem (kate mark pat : ℕ) 
  (h1 : pat = 2 * kate)
  (h2 : 3 * pat = mark)
  (h3 : mark = kate + 120) :
  kate + mark + pat = 216 := by
  sorry

end project_hours_theorem_l3466_346637


namespace volume_conversion_m_to_dm_volume_conversion_mL_to_L_volume_conversion_cm_to_dm_l3466_346628

-- Define conversion factors
def m_to_dm : ℝ := 10
def L_to_mL : ℝ := 1000
def dm_to_cm : ℝ := 10

-- Theorem statements
theorem volume_conversion_m_to_dm : 
  20 * (m_to_dm ^ 3) = 20000 := by sorry

theorem volume_conversion_mL_to_L : 
  15 / L_to_mL = 0.015 := by sorry

theorem volume_conversion_cm_to_dm : 
  1200 / (dm_to_cm ^ 3) = 1.2 := by sorry

end volume_conversion_m_to_dm_volume_conversion_mL_to_L_volume_conversion_cm_to_dm_l3466_346628


namespace square_overlap_area_l3466_346631

theorem square_overlap_area (β : Real) (h1 : 0 < β) (h2 : β < Real.pi / 2) (h3 : Real.cos β = 3/5) :
  let square_side : Real := 2
  let overlap_area : Real := 
    2 * (square_side * (1 - Real.tan (β/2)) / (1 + Real.tan (β/2))) * square_side / 2
  overlap_area = 4/3 := by sorry

end square_overlap_area_l3466_346631


namespace polynomial_expansion_problem_l3466_346622

theorem polynomial_expansion_problem (p q : ℝ) : 
  p > 0 → q > 0 → p + q = 1 → 
  55 * p^9 * q^2 = 165 * p^8 * q^3 → 
  p = 3/4 := by
sorry

end polynomial_expansion_problem_l3466_346622


namespace transformations_result_l3466_346618

/-- Rotates a point (x, y) by 180° counterclockwise around (2, 3) -/
def rotate180 (x y : ℝ) : ℝ × ℝ :=
  (4 - x, 6 - y)

/-- Reflects a point (x, y) about the line y = x -/
def reflectAboutYEqX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

/-- Applies both transformations to a point (x, y) -/
def applyTransformations (x y : ℝ) : ℝ × ℝ :=
  let (x', y') := rotate180 x y
  reflectAboutYEqX x' y'

theorem transformations_result (c d : ℝ) :
  applyTransformations c d = (1, -4) → d - c = 7 := by
  sorry

end transformations_result_l3466_346618


namespace reader_one_hour_ago_page_l3466_346690

/-- A reader who reads at a constant rate -/
structure Reader where
  rate : ℕ  -- pages per hour
  total_pages : ℕ
  current_page : ℕ
  remaining_hours : ℕ

/-- Calculates the page a reader was on one hour ago -/
def page_one_hour_ago (r : Reader) : ℕ :=
  r.current_page - r.rate

/-- Theorem: Given the specified conditions, the reader was on page 60 one hour ago -/
theorem reader_one_hour_ago_page :
  ∀ (r : Reader),
  r.total_pages = 210 →
  r.current_page = 90 →
  r.remaining_hours = 4 →
  (r.total_pages - r.current_page) = (r.rate * r.remaining_hours) →
  page_one_hour_ago r = 60 := by
  sorry


end reader_one_hour_ago_page_l3466_346690


namespace sum_of_squared_differences_l3466_346657

theorem sum_of_squared_differences : (302^2 - 298^2) + (152^2 - 148^2) = 3600 := by
  sorry

end sum_of_squared_differences_l3466_346657


namespace lottery_is_systematic_sampling_l3466_346681

-- Define the lottery range
def lottery_range : Set ℕ := {n | 0 ≤ n ∧ n < 100000}

-- Define the winning number criteria
def is_winning_number (n : ℕ) : Prop :=
  n ∈ lottery_range ∧ (n % 100 = 88 ∨ n % 100 = 68)

-- Define systematic sampling
def systematic_sampling (S : Set ℕ) (f : ℕ → Prop) : Prop :=
  ∃ k : ℕ, k > 0 ∧ ∀ n ∈ S, f n ↔ ∃ m : ℕ, n = m * k

-- Theorem statement
theorem lottery_is_systematic_sampling :
  systematic_sampling lottery_range is_winning_number := by
  sorry


end lottery_is_systematic_sampling_l3466_346681


namespace victor_remaining_lives_l3466_346665

def calculate_lives_remaining (initial_lives : ℕ) 
                               (first_level_loss : ℕ) 
                               (second_level_gain_rate : ℕ) 
                               (second_level_duration : ℕ) 
                               (third_level_loss_rate : ℕ) 
                               (third_level_duration : ℕ) : ℕ :=
  let lives_after_first := initial_lives - first_level_loss
  let second_level_intervals := second_level_duration / 45
  let lives_after_second := lives_after_first + second_level_gain_rate * second_level_intervals
  let third_level_intervals := third_level_duration / 20
  lives_after_second - third_level_loss_rate * third_level_intervals

theorem victor_remaining_lives : 
  calculate_lives_remaining 246 14 3 135 4 80 = 225 := by
  sorry

end victor_remaining_lives_l3466_346665


namespace union_complement_equal_l3466_346623

def U : Set ℕ := {x | x < 4}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem union_complement_equal : B ∪ (U \ A) = {2, 3} := by
  sorry

end union_complement_equal_l3466_346623


namespace floor_plus_self_unique_solution_l3466_346642

theorem floor_plus_self_unique_solution :
  ∃! r : ℝ, ⌊r⌋ + r = 20.7 :=
by sorry

end floor_plus_self_unique_solution_l3466_346642


namespace bob_always_wins_l3466_346609

/-- The game described in the problem -/
def Game (n : ℕ) : Prop :=
  ∀ (A : Fin (n + 1) → Finset (Fin (2^n))),
    (∀ i, (A i).card = 2^(n-1)) →
    ∃ (a : Fin (n + 1) → Fin (2^n)),
      ∀ t : Fin (2^n),
        ∃ i s, s ∈ A i ∧ (s + a i : Fin (2^n)) = t

/-- Bob always has a winning strategy for any positive n -/
theorem bob_always_wins :
  ∀ n : ℕ, n > 0 → Game n :=
sorry

end bob_always_wins_l3466_346609


namespace inequality_system_solution_set_l3466_346696

theorem inequality_system_solution_set : 
  ∀ x : ℝ, (abs x < 1 ∧ x * (x + 2) > 0) ↔ (0 < x ∧ x < 1) :=
by sorry

end inequality_system_solution_set_l3466_346696


namespace johns_allowance_l3466_346624

/-- John's weekly allowance problem -/
theorem johns_allowance (A : ℚ) : 
  (A > 0) →                                           -- Allowance is positive
  (3 / 5 * A + 1 / 3 * (2 / 5 * A) + 96 / 100 = A) →  -- Total spending equals allowance
  (A = 36 / 10) :=                                    -- Allowance is $3.60
by sorry

end johns_allowance_l3466_346624


namespace pens_distribution_eq_six_l3466_346668

/-- The number of ways to distribute n identical items among k distinct groups,
    where each group gets at least m items. -/
def distribute_with_minimum (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- The number of ways to distribute 8 pens among 3 friends,
    where each friend gets at least 2 pens. -/
def pens_distribution : ℕ :=
  distribute_with_minimum 8 3 2

theorem pens_distribution_eq_six :
  pens_distribution = 6 := by
  sorry

end pens_distribution_eq_six_l3466_346668


namespace smallest_number_minus_three_divisible_by_fifteen_l3466_346647

theorem smallest_number_minus_three_divisible_by_fifteen : 
  ∃ N : ℕ, (N ≥ 18) ∧ (N - 3) % 15 = 0 ∧ ∀ M : ℕ, M < N → (M - 3) % 15 ≠ 0 := by
  sorry

end smallest_number_minus_three_divisible_by_fifteen_l3466_346647


namespace sum_of_base8_digits_of_888_l3466_346611

/-- Given a natural number n and a base b, returns the list of digits of n in base b -/
def toDigits (n : ℕ) (b : ℕ) : List ℕ := sorry

/-- The sum of a list of natural numbers -/
def sum (l : List ℕ) : ℕ := sorry

theorem sum_of_base8_digits_of_888 :
  sum (toDigits 888 8) = 13 := by sorry

end sum_of_base8_digits_of_888_l3466_346611


namespace puppy_sleeps_16_hours_l3466_346625

def connor_sleep_time : ℕ := 6

def luke_sleep_time (connor_sleep_time : ℕ) : ℕ := connor_sleep_time + 2

def puppy_sleep_time (luke_sleep_time : ℕ) : ℕ := 2 * luke_sleep_time

theorem puppy_sleeps_16_hours :
  puppy_sleep_time (luke_sleep_time connor_sleep_time) = 16 := by
  sorry

end puppy_sleeps_16_hours_l3466_346625


namespace circumscribed_sphere_area_l3466_346608

theorem circumscribed_sphere_area (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 2 * Real.sqrt 6) :
  let diagonal_squared := a^2 + b^2 + c^2
  let sphere_radius := Real.sqrt (diagonal_squared / 4)
  let sphere_surface_area := 4 * Real.pi * sphere_radius^2
  sphere_surface_area = 49 * Real.pi := by
sorry

end circumscribed_sphere_area_l3466_346608


namespace solution_when_a_is_one_solution_when_a_greater_than_two_solution_when_a_equals_two_solution_when_a_less_than_two_l3466_346616

-- Define the inequality
def inequality (a x : ℝ) : Prop := x^2 - (a+2)*x + 2*a < 0

-- Theorem for a = 1
theorem solution_when_a_is_one :
  ∀ x : ℝ, inequality 1 x ↔ 1 < x ∧ x < 2 :=
sorry

-- Theorem for a > 2
theorem solution_when_a_greater_than_two :
  ∀ a x : ℝ, a > 2 → (inequality a x ↔ 2 < x ∧ x < a) :=
sorry

-- Theorem for a = 2
theorem solution_when_a_equals_two :
  ∀ x : ℝ, ¬(inequality 2 x) :=
sorry

-- Theorem for a < 2
theorem solution_when_a_less_than_two :
  ∀ a x : ℝ, a < 2 → (inequality a x ↔ a < x ∧ x < 2) :=
sorry

end solution_when_a_is_one_solution_when_a_greater_than_two_solution_when_a_equals_two_solution_when_a_less_than_two_l3466_346616


namespace sum_of_A_and_B_l3466_346676

/-- The number of five-digit odd numbers -/
def A : ℕ := 9 * 10 * 10 * 10 * 5

/-- The number of five-digit multiples of 5 that are also odd -/
def B : ℕ := 9 * 10 * 10 * 10 * 1

/-- The sum of A and B is equal to 45,000 -/
theorem sum_of_A_and_B : A + B = 45000 := by
  sorry

end sum_of_A_and_B_l3466_346676


namespace normal_dist_symmetry_normal_dist_property_l3466_346605

/-- A normal distribution with mean 0 and standard deviation σ -/
def normal_dist (σ : ℝ) : Type := ℝ

/-- Probability measure for the normal distribution -/
def P (σ : ℝ) : Set ℝ → ℝ := sorry

theorem normal_dist_symmetry 
  (σ : ℝ) (a : ℝ) : 
  P σ {x | -a ≤ x ∧ x ≤ 0} = P σ {x | 0 ≤ x ∧ x ≤ a} :=
sorry

theorem normal_dist_property 
  (σ : ℝ) (h : P σ {x | -2 ≤ x ∧ x ≤ 0} = 0.3) : 
  P σ {x | x > 2} = 0.2 :=
sorry

end normal_dist_symmetry_normal_dist_property_l3466_346605


namespace compound_interest_problem_l3466_346662

/-- Proves that the principal amount is 20000 given the specified conditions --/
theorem compound_interest_problem (P : ℝ) : 
  P * (1 + 0.2 / 2)^4 - P * (1 + 0.2)^2 = 482 → P = 20000 := by
  sorry

end compound_interest_problem_l3466_346662


namespace exists_dividable_polyhedron_l3466_346663

/-- A face of a polyhedron -/
structure Face where
  -- Add necessary properties of a face

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  faces : Set Face
  -- Add necessary properties to ensure convexity

/-- A function that checks if a set of faces can form a convex polyhedron -/
def can_form_convex_polyhedron (faces : Set Face) : Prop :=
  ∃ (p : ConvexPolyhedron), p.faces = faces

/-- Theorem: There exists a convex polyhedron whose faces can be divided into two sets,
    each of which can form a convex polyhedron -/
theorem exists_dividable_polyhedron :
  ∃ (p : ConvexPolyhedron) (s₁ s₂ : Set Face),
    s₁ ∪ s₂ = p.faces ∧
    s₁ ∩ s₂ = ∅ ∧
    can_form_convex_polyhedron s₁ ∧
    can_form_convex_polyhedron s₂ :=
sorry

end exists_dividable_polyhedron_l3466_346663


namespace union_of_sets_l3466_346600

/-- Given sets M and N, prove that their union is equal to the set of all x between -1 and 5 inclusive -/
theorem union_of_sets (M N : Set ℝ) (hM : M = {x : ℝ | -1 ≤ x ∧ x < 3}) (hN : N = {x : ℝ | 2 < x ∧ x ≤ 5}) :
  M ∪ N = {x : ℝ | -1 ≤ x ∧ x ≤ 5} := by
  sorry

end union_of_sets_l3466_346600


namespace purchase_plan_monthly_payment_l3466_346653

theorem purchase_plan_monthly_payment 
  (purchase_price : ℝ) 
  (down_payment : ℝ) 
  (num_payments : ℕ) 
  (interest_rate : ℝ) 
  (h1 : purchase_price = 118)
  (h2 : down_payment = 18)
  (h3 : num_payments = 12)
  (h4 : interest_rate = 0.15254237288135593) :
  let total_interest : ℝ := purchase_price * interest_rate
  let total_paid : ℝ := purchase_price + total_interest
  let monthly_payment : ℝ := (total_paid - down_payment) / num_payments
  monthly_payment = 9.833333333333334 := by sorry

end purchase_plan_monthly_payment_l3466_346653


namespace sin_sixty_minus_third_power_zero_l3466_346629

theorem sin_sixty_minus_third_power_zero :
  2 * Real.sin (60 * π / 180) - (1/3)^0 = Real.sqrt 3 - 1 := by
sorry

end sin_sixty_minus_third_power_zero_l3466_346629


namespace rosas_phone_calls_l3466_346658

/-- Rosa's phone book calling problem -/
theorem rosas_phone_calls (pages_last_week pages_this_week : ℝ) 
  (h1 : pages_last_week = 10.2)
  (h2 : pages_this_week = 8.6) : 
  pages_last_week + pages_this_week = 18.8 := by
  sorry

end rosas_phone_calls_l3466_346658


namespace perpendicular_tangents_sum_l3466_346646

/-- The problem statement -/
theorem perpendicular_tangents_sum (a b : ℝ) : 
  (∃ x₀ y₀ : ℝ, 
    -- Point (x₀, y₀) is on both curves
    (y₀ = x₀^2 - 2*x₀ + 2 ∧ y₀ = -x₀^2 + a*x₀ + b) ∧ 
    -- Tangents are perpendicular
    (2*x₀ - 2) * (-2*x₀ + a) = -1) → 
  a + b = 5/2 := by
sorry

end perpendicular_tangents_sum_l3466_346646


namespace square_of_integer_proof_l3466_346684

theorem square_of_integer_proof (n : ℕ+) (h : ∃ (k : ℤ), k^2 = 1 + 12 * (n : ℤ)^2) :
  ∃ (m : ℤ), (2 : ℤ) + 2 * Int.sqrt (1 + 12 * (n : ℤ)^2) = m^2 := by
  sorry

end square_of_integer_proof_l3466_346684


namespace larger_cross_section_distance_l3466_346651

/-- Represents a right octagonal pyramid -/
structure RightOctagonalPyramid where
  /-- Height of the pyramid -/
  height : ℝ
  /-- Side length of the base octagon -/
  base_side : ℝ

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  /-- Distance from the apex of the pyramid -/
  distance_from_apex : ℝ
  /-- Area of the cross section -/
  area : ℝ

/-- Main theorem about the distance of the larger cross section from the apex -/
theorem larger_cross_section_distance
  (pyramid : RightOctagonalPyramid)
  (cs1 cs2 : CrossSection)
  (h_areas : cs1.area = 300 * Real.sqrt 2 ∧ cs2.area = 675 * Real.sqrt 2)
  (h_distance : |cs1.distance_from_apex - cs2.distance_from_apex| = 10)
  (h_order : cs1.area < cs2.area) :
  cs2.distance_from_apex = 30 := by
sorry

end larger_cross_section_distance_l3466_346651


namespace quadratic_expression_value_l3466_346673

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 2*x + y = 5) 
  (eq2 : x + 2*y = 6) : 
  7*x^2 + 10*x*y + 7*y^2 = 85 := by
sorry

end quadratic_expression_value_l3466_346673


namespace maurice_rides_before_is_10_l3466_346652

/-- The number of times Maurice had been horseback riding before visiting Matt -/
def maurice_rides_before : ℕ := 10

/-- The number of different horses Maurice rode before his visit -/
def maurice_horses_before : ℕ := 2

/-- The number of different horses Matt has ridden -/
def matt_horses : ℕ := 4

/-- The number of times Maurice rode during his visit -/
def maurice_rides_visit : ℕ := 8

/-- The number of additional times Matt rode on his other horses -/
def matt_additional_rides : ℕ := 16

/-- The number of horses Matt rode each time with Maurice -/
def matt_horses_per_ride : ℕ := 2

theorem maurice_rides_before_is_10 :
  maurice_rides_before = 10 ∧
  maurice_horses_before = 2 ∧
  matt_horses = 4 ∧
  maurice_rides_visit = 8 ∧
  matt_additional_rides = 16 ∧
  matt_horses_per_ride = 2 ∧
  maurice_rides_visit = maurice_rides_before ∧
  (maurice_rides_visit * matt_horses_per_ride + matt_additional_rides) = 3 * maurice_rides_before :=
by sorry

end maurice_rides_before_is_10_l3466_346652


namespace degrees_to_radians_conversion_l3466_346689

theorem degrees_to_radians_conversion :
  ∀ (degrees : ℝ) (radians : ℝ),
  degrees * (π / 180) = radians →
  -630 * (π / 180) = -7 * π / 2 :=
by sorry

end degrees_to_radians_conversion_l3466_346689


namespace gcd_conditions_and_sum_of_digits_l3466_346620

/-- The least positive integer greater than 1000 satisfying the given GCD conditions -/
def n : ℕ := sorry

/-- Sum of digits function -/
def sum_of_digits (m : ℕ) : ℕ := sorry

theorem gcd_conditions_and_sum_of_digits :
  n > 1000 ∧
  Nat.gcd 75 (n + 150) = 25 ∧
  Nat.gcd (n + 75) 150 = 75 ∧
  (∀ k, k > 1000 → Nat.gcd 75 (k + 150) = 25 → Nat.gcd (k + 75) 150 = 75 → k ≥ n) ∧
  sum_of_digits n = 9 := by sorry

end gcd_conditions_and_sum_of_digits_l3466_346620


namespace gina_money_to_mom_l3466_346687

theorem gina_money_to_mom (total : ℝ) (clothes_fraction : ℝ) (charity_fraction : ℝ) (kept : ℝ) :
  total = 400 →
  clothes_fraction = 1/8 →
  charity_fraction = 1/5 →
  kept = 170 →
  ∃ (mom_fraction : ℝ), 
    mom_fraction * total + clothes_fraction * total + charity_fraction * total + kept = total ∧
    mom_fraction = 1/4 :=
by sorry

end gina_money_to_mom_l3466_346687


namespace sixth_term_value_l3466_346635

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the roots of the quadratic equation
def roots_of_equation (a : ℕ → ℝ) : Prop :=
  3 * (a 3)^2 - 11 * (a 3) + 9 = 0 ∧ 3 * (a 9)^2 - 11 * (a 9) + 9 = 0

-- Theorem statement
theorem sixth_term_value (a : ℕ → ℝ) :
  geometric_sequence a → roots_of_equation a → (a 6)^2 = 3 :=
by sorry

end sixth_term_value_l3466_346635


namespace inequality_system_solution_l3466_346640

-- Define the inequality system
def inequality_system (a b x : ℝ) : Prop :=
  x - a > 2 ∧ x + 1 < b

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  -1 < x ∧ x < 1

-- Theorem statement
theorem inequality_system_solution (a b : ℝ) :
  (∀ x, inequality_system a b x ↔ solution_set x) →
  (a + b)^2023 = -1 :=
by sorry

end inequality_system_solution_l3466_346640


namespace field_fencing_l3466_346633

/-- Proves that a rectangular field with area 80 sq. feet and one side 20 feet requires 28 feet of fencing for the other three sides. -/
theorem field_fencing (length width : ℝ) : 
  length * width = 80 → 
  length = 20 → 
  length + 2 * width = 28 := by sorry

end field_fencing_l3466_346633


namespace set_equality_through_double_complement_l3466_346659

universe u

theorem set_equality_through_double_complement 
  {U : Type u} [Nonempty U] (M N P : Set U) 
  (h1 : M = (Nᶜ : Set U)) 
  (h2 : N = (Pᶜ : Set U)) : 
  M = P := by
  sorry

end set_equality_through_double_complement_l3466_346659


namespace goldfish_graph_is_finite_distinct_points_l3466_346692

def cost (n : ℕ) : ℕ := 20 * n + 500

def goldfish_points : Set (ℕ × ℕ) :=
  {p | ∃ n : ℕ, 1 ≤ n ∧ n ≤ 20 ∧ p = (n, cost n)}

theorem goldfish_graph_is_finite_distinct_points :
  Finite goldfish_points ∧ 
  (∀ p q : ℕ × ℕ, p ∈ goldfish_points → q ∈ goldfish_points → p ≠ q → p.2 ≠ q.2) :=
sorry

end goldfish_graph_is_finite_distinct_points_l3466_346692


namespace remaining_item_is_bead_l3466_346638

/-- Represents the three types of items --/
inductive Item
  | GoldBar
  | Pearl
  | Bead

/-- Represents the state of the tribe's possessions --/
structure TribeState where
  goldBars : Nat
  pearls : Nat
  beads : Nat

/-- Represents the possible exchanges --/
inductive Exchange
  | Cortes    -- 1 gold bar + 1 pearl → 1 bead
  | Montezuma -- 1 gold bar + 1 bead → 1 pearl
  | Totonacs  -- 1 pearl + 1 bead → 1 gold bar

def initialState : TribeState :=
  { goldBars := 24, pearls := 26, beads := 25 }

def applyExchange (state : TribeState) (exchange : Exchange) : TribeState :=
  match exchange with
  | Exchange.Cortes =>
      { goldBars := state.goldBars - 1, pearls := state.pearls - 1, beads := state.beads + 1 }
  | Exchange.Montezuma =>
      { goldBars := state.goldBars - 1, pearls := state.pearls + 1, beads := state.beads - 1 }
  | Exchange.Totonacs =>
      { goldBars := state.goldBars + 1, pearls := state.pearls - 1, beads := state.beads - 1 }

def remainingItem (state : TribeState) : Option Item :=
  if state.goldBars > 0 && state.pearls = 0 && state.beads = 0 then some Item.GoldBar
  else if state.goldBars = 0 && state.pearls > 0 && state.beads = 0 then some Item.Pearl
  else if state.goldBars = 0 && state.pearls = 0 && state.beads > 0 then some Item.Bead
  else none

/-- Theorem stating that if only one item type remains after any number of exchanges, it must be beads --/
theorem remaining_item_is_bead (exchanges : List Exchange) :
  let finalState := exchanges.foldl applyExchange initialState
  remainingItem finalState = some Item.Bead ∨ remainingItem finalState = none := by
  sorry

end remaining_item_is_bead_l3466_346638


namespace polynomial_division_remainder_l3466_346683

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^4 + 8 * X^3 - 35 * X^2 - 45 * X + 52 = 
  (X^2 + 5 * X - 3) * q + (-21 * X + 79) :=
by sorry

end polynomial_division_remainder_l3466_346683


namespace rem_neg_five_sixths_three_fourths_l3466_346688

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem rem_neg_five_sixths_three_fourths :
  rem (-5/6) (3/4) = 2/3 := by sorry

end rem_neg_five_sixths_three_fourths_l3466_346688


namespace games_played_calculation_l3466_346666

/-- Represents the gambler's poker game statistics -/
structure GamblerStats where
  gamesPlayed : ℝ
  initialWinRate : ℝ
  newWinRate : ℝ
  targetWinRate : ℝ
  additionalGames : ℝ

/-- Theorem stating the number of games played given the conditions -/
theorem games_played_calculation (stats : GamblerStats)
  (h1 : stats.initialWinRate = 0.4)
  (h2 : stats.newWinRate = 0.8)
  (h3 : stats.targetWinRate = 0.6)
  (h4 : stats.additionalGames = 19.999999999999993)
  (h5 : stats.initialWinRate * stats.gamesPlayed + stats.newWinRate * stats.additionalGames = 
        stats.targetWinRate * (stats.gamesPlayed + stats.additionalGames)) :
  stats.gamesPlayed = 20 := by
  sorry

end games_played_calculation_l3466_346666


namespace f_10_equals_144_l3466_346615

def f : ℕ → ℕ
  | 0 => 0  -- define f(0) as 0 for completeness
  | 1 => 2
  | 2 => 3
  | (n + 3) => f (n + 2) + f (n + 1)

theorem f_10_equals_144 : f 10 = 144 := by sorry

end f_10_equals_144_l3466_346615


namespace profit_for_450_pieces_l3466_346674

/-- The price function for the clothing factory -/
def price (x : ℕ) : ℚ :=
  if x ≤ 100 then 60
  else 62 - x / 50

/-- The profit function for the clothing factory -/
def profit (x : ℕ) : ℚ :=
  (price x - 40) * x

/-- The theorem stating the profit for an order of 450 pieces -/
theorem profit_for_450_pieces :
  0 < 450 ∧ 450 ≤ 500 → profit 450 = 5850 := by sorry

end profit_for_450_pieces_l3466_346674


namespace smallest_four_digit_mod_8_5_l3466_346682

theorem smallest_four_digit_mod_8_5 : ∃ (n : ℕ), 
  (n ≥ 1000) ∧ 
  (n < 10000) ∧ 
  (n % 8 = 5) ∧ 
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 8 = 5) → m ≥ n) ∧
  (n = 1005) := by
sorry

end smallest_four_digit_mod_8_5_l3466_346682


namespace initial_coins_l3466_346617

/-- Given a box of coins, prove that the initial number of coins is 21 when 8 coins are added and the total becomes 29. -/
theorem initial_coins (initial_coins added_coins total_coins : ℕ) 
  (h1 : added_coins = 8)
  (h2 : total_coins = 29)
  (h3 : initial_coins + added_coins = total_coins) : 
  initial_coins = 21 := by
sorry

end initial_coins_l3466_346617


namespace negative_five_times_three_l3466_346675

theorem negative_five_times_three : (-5 : ℤ) * 3 = -15 := by
  sorry

end negative_five_times_three_l3466_346675


namespace solution_of_quadratic_equations_l3466_346667

theorem solution_of_quadratic_equations :
  let eq1 : ℝ → Prop := λ x ↦ 2 * x^2 = 3 * (2 * x + 1)
  let eq2 : ℝ → Prop := λ x ↦ 3 * x * (x + 2) = 4 * x + 8
  let sol1 : Set ℝ := {(3 + Real.sqrt 15) / 2, (3 - Real.sqrt 15) / 2}
  let sol2 : Set ℝ := {-2, 4/3}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ x, eq1 x → x ∈ sol1) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ x, eq2 x → x ∈ sol2) := by
  sorry

end solution_of_quadratic_equations_l3466_346667


namespace unique_student_count_l3466_346610

theorem unique_student_count : ∃! n : ℕ, n < 600 ∧ n % 25 = 24 ∧ n % 19 = 18 ∧ n = 424 := by
  sorry

end unique_student_count_l3466_346610


namespace arithmetic_square_root_of_sqrt_16_l3466_346602

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end arithmetic_square_root_of_sqrt_16_l3466_346602


namespace wolf_does_not_catch_hare_l3466_346644

/-- Prove that the wolf does not catch the hare given the initial conditions -/
theorem wolf_does_not_catch_hare (initial_distance : ℝ) (distance_to_refuge : ℝ) 
  (wolf_speed : ℝ) (hare_speed : ℝ) 
  (h1 : initial_distance = 30) 
  (h2 : distance_to_refuge = 250) 
  (h3 : wolf_speed = 600) 
  (h4 : hare_speed = 550) : 
  (distance_to_refuge / hare_speed) < ((initial_distance + distance_to_refuge) / wolf_speed) :=
by
  sorry

#check wolf_does_not_catch_hare

end wolf_does_not_catch_hare_l3466_346644


namespace gcd_property_l3466_346670

theorem gcd_property (a b c : ℤ) (h : Nat.gcd a.natAbs b.natAbs = 1) :
  Nat.gcd a.natAbs (b * c).natAbs = Nat.gcd a.natAbs c.natAbs := by
  sorry

end gcd_property_l3466_346670


namespace cubic_root_sum_cubes_l3466_346694

theorem cubic_root_sum_cubes (r s t : ℂ) : 
  (8 * r^3 + 2010 * r + 4016 = 0) →
  (8 * s^3 + 2010 * s + 4016 = 0) →
  (8 * t^3 + 2010 * t + 4016 = 0) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 1506 := by
  sorry

end cubic_root_sum_cubes_l3466_346694


namespace annieka_made_14_throws_l3466_346634

/-- The number of free-throws made by DeShawn -/
def deshawn_throws : ℕ := 12

/-- The number of free-throws made by Kayla -/
def kayla_throws : ℕ := (deshawn_throws * 3) / 2

/-- The number of free-throws made by Annieka -/
def annieka_throws : ℕ := kayla_throws - 4

/-- Theorem: Annieka made 14 free-throws -/
theorem annieka_made_14_throws : annieka_throws = 14 := by
  sorry

end annieka_made_14_throws_l3466_346634


namespace equation_solution_l3466_346693

theorem equation_solution :
  let f (x : ℂ) := (x^2 + x + 1) / (x + 1)
  let g (x : ℂ) := x^2 + 2*x + 3
  ∀ x : ℂ, f x = g x ↔ x = -2 ∨ x = Complex.I * Real.sqrt 2 ∨ x = -Complex.I * Real.sqrt 2 :=
by sorry

end equation_solution_l3466_346693


namespace grid_d4_is_5_l3466_346636

/-- Represents a 5x5 grid of numbers -/
def Grid := Fin 5 → Fin 5 → Fin 5

/-- Checks if a row contains all different numbers -/
def row_all_different (g : Grid) (r : Fin 5) : Prop :=
  ∀ i j : Fin 5, i ≠ j → g r i ≠ g r j

/-- Checks if a column contains all different numbers -/
def col_all_different (g : Grid) (c : Fin 5) : Prop :=
  ∀ i j : Fin 5, i ≠ j → g i c ≠ g j c

/-- Checks if all rows and columns contain different numbers -/
def all_different (g : Grid) : Prop :=
  (∀ r : Fin 5, row_all_different g r) ∧ (∀ c : Fin 5, col_all_different g c)

/-- Checks if the sum of numbers in the 4th column is 9 -/
def fourth_column_sum_9 (g : Grid) : Prop :=
  (g 1 3).val + (g 3 3).val = 9

/-- Checks if the sum of numbers in white cells of row C is 7 -/
def row_c_white_sum_7 (g : Grid) : Prop :=
  (g 2 0).val + (g 2 2).val + (g 2 4).val = 7

/-- Checks if the sum of numbers in white cells of 2nd column is 8 -/
def second_column_white_sum_8 (g : Grid) : Prop :=
  (g 0 1).val + (g 2 1).val + (g 4 1).val = 8

/-- Checks if the sum of numbers in white cells of row B is less than row D -/
def row_b_less_than_row_d (g : Grid) : Prop :=
  (g 1 1).val + (g 1 3).val < (g 3 1).val + (g 3 3).val

theorem grid_d4_is_5 (g : Grid) 
  (h1 : all_different g)
  (h2 : fourth_column_sum_9 g)
  (h3 : row_c_white_sum_7 g)
  (h4 : second_column_white_sum_8 g)
  (h5 : row_b_less_than_row_d g) :
  g 3 3 = 5 := by
  sorry

end grid_d4_is_5_l3466_346636


namespace average_of_multiples_10_to_300_l3466_346619

def multiples_of_10 (n : ℕ) : List ℕ :=
  List.filter (fun x => x % 10 = 0) (List.range (n + 1))

theorem average_of_multiples_10_to_300 :
  let sequence := multiples_of_10 300
  (sequence.sum / sequence.length : ℚ) = 155 := by
sorry

end average_of_multiples_10_to_300_l3466_346619


namespace cos_seven_pi_fourth_l3466_346643

theorem cos_seven_pi_fourth : Real.cos (7 * π / 4) = 1 / Real.sqrt 2 := by
  sorry

end cos_seven_pi_fourth_l3466_346643


namespace composite_numbers_l3466_346691

theorem composite_numbers (n : ℕ) (h : n = 3^2001) : 
  (∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 2^n + 1 = a * b) ∧ 
  (∃ (c d : ℕ), c > 1 ∧ d > 1 ∧ 2^n - 1 = c * d) := by
sorry


end composite_numbers_l3466_346691


namespace no_quadratic_trinomial_always_power_of_two_l3466_346632

theorem no_quadratic_trinomial_always_power_of_two : 
  ¬ ∃ (a b c : ℤ), ∀ (x : ℕ), ∃ (n : ℕ), a * x^2 + b * x + c = 2^n := by
  sorry

end no_quadratic_trinomial_always_power_of_two_l3466_346632


namespace decreasing_quadratic_implies_a_le_two_l3466_346655

/-- A quadratic function f(x) = -x² - 2(a-1)x + 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 - 2*(a-1)*x + 5

/-- The theorem states that if f(x) is decreasing on [-1, +∞), then a ≤ 2 -/
theorem decreasing_quadratic_implies_a_le_two (a : ℝ) :
  (∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ < x₂ → f a x₂ < f a x₁) →
  a ≤ 2 :=
sorry

end decreasing_quadratic_implies_a_le_two_l3466_346655


namespace second_discount_percentage_prove_discount_percentage_l3466_346669

/-- Calculates the second discount percentage given the original price, first discount percentage, and final price --/
theorem second_discount_percentage 
  (original_price : ℝ) 
  (first_discount_percent : ℝ) 
  (final_price : ℝ) : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount_percent / 100)
  let second_discount_decimal := (price_after_first_discount - final_price) / price_after_first_discount
  second_discount_decimal * 100

/-- Proves that the second discount percentage is approximately 2% given the problem conditions --/
theorem prove_discount_percentage : 
  let original_price : ℝ := 65
  let first_discount_percent : ℝ := 10
  let final_price : ℝ := 57.33
  let result := second_discount_percentage original_price first_discount_percent final_price
  abs (result - 2) < 0.01 := by
  sorry

end second_discount_percentage_prove_discount_percentage_l3466_346669


namespace eggs_per_basket_l3466_346698

theorem eggs_per_basket (yellow_eggs : Nat) (pink_eggs : Nat) (min_eggs : Nat) : 
  yellow_eggs = 30 → pink_eggs = 45 → min_eggs = 5 → 
  ∃ (eggs_per_basket : Nat), 
    eggs_per_basket ∣ yellow_eggs ∧ 
    eggs_per_basket ∣ pink_eggs ∧ 
    eggs_per_basket ≥ min_eggs ∧
    ∀ (n : Nat), n ∣ yellow_eggs → n ∣ pink_eggs → n ≥ min_eggs → n ≤ eggs_per_basket :=
by
  sorry

end eggs_per_basket_l3466_346698


namespace original_deck_size_l3466_346649

/-- Represents a deck of cards with red and black cards only -/
structure Deck where
  red : ℕ
  black : ℕ

/-- The probability of selecting a red card from the deck -/
def redProbability (d : Deck) : ℚ :=
  d.red / (d.red + d.black)

theorem original_deck_size :
  ∀ d : Deck,
  redProbability d = 1/4 →
  redProbability {red := d.red, black := d.black + 6} = 1/5 →
  d.red + d.black = 24 := by
sorry

end original_deck_size_l3466_346649


namespace monotone_increasing_implies_a_geq_5_l3466_346650

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

-- State the theorem
theorem monotone_increasing_implies_a_geq_5 :
  ∀ a : ℝ, (∀ x y : ℝ, -5 ≤ x ∧ x < y ∧ y ≤ 5 → f a x < f a y) → a ≥ 5 := by
  sorry

end monotone_increasing_implies_a_geq_5_l3466_346650


namespace cafeteria_apples_l3466_346627

theorem cafeteria_apples (apples_to_students : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) :
  apples_to_students = 42 →
  num_pies = 9 →
  apples_per_pie = 6 →
  apples_to_students + num_pies * apples_per_pie = 96 :=
by
  sorry

end cafeteria_apples_l3466_346627


namespace vector_sum_diff_magnitude_bounds_l3466_346697

theorem vector_sum_diff_magnitude_bounds (a b : ℝ × ℝ) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) : 
  (∃ x : ℝ × ℝ, ‖x‖ = 1 ∧ ∃ y : ℝ × ℝ, ‖y‖ = 2 ∧ ‖x + y‖ + ‖x - y‖ = 4) ∧
  (∃ x : ℝ × ℝ, ‖x‖ = 1 ∧ ∃ y : ℝ × ℝ, ‖y‖ = 2 ∧ ‖x + y‖ + ‖x - y‖ = 2 * Real.sqrt 5) ∧
  (∀ x y : ℝ × ℝ, ‖x‖ = 1 → ‖y‖ = 2 → 4 ≤ ‖x + y‖ + ‖x - y‖ ∧ ‖x + y‖ + ‖x - y‖ ≤ 2 * Real.sqrt 5) :=
by sorry

end vector_sum_diff_magnitude_bounds_l3466_346697


namespace special_number_property_l3466_346672

/-- The greatest integer less than 100 for which the greatest common factor with 18 is 3 -/
def special_number : ℕ := 93

/-- Theorem stating that special_number satisfies the required conditions -/
theorem special_number_property : 
  special_number < 100 ∧ 
  Nat.gcd special_number 18 = 3 ∧ 
  ∀ n : ℕ, n < 100 → Nat.gcd n 18 = 3 → n ≤ special_number := by
  sorry

end special_number_property_l3466_346672


namespace x_power_five_minus_twenty_seven_x_squared_l3466_346604

theorem x_power_five_minus_twenty_seven_x_squared (x : ℝ) (h : x^3 - 3*x = 5) :
  x^5 - 27*x^2 = -22*x^2 + 9*x + 15 := by
  sorry

end x_power_five_minus_twenty_seven_x_squared_l3466_346604


namespace max_value_expression_l3466_346660

theorem max_value_expression (a b c d : ℝ) 
  (ha : -6 ≤ a ∧ a ≤ 6)
  (hb : -6 ≤ b ∧ b ≤ 6)
  (hc : -6 ≤ c ∧ c ≤ 6)
  (hd : -6 ≤ d ∧ d ≤ 6) :
  (∀ x y z w, -6 ≤ x ∧ x ≤ 6 → -6 ≤ y ∧ y ≤ 6 → -6 ≤ z ∧ z ≤ 6 → -6 ≤ w ∧ w ≤ 6 →
    x + 2*y + z + 2*w - x*y - y*z - z*w - w*x ≤ a + 2*b + c + 2*d - a*b - b*c - c*d - d*a) →
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a = 156 :=
by sorry

end max_value_expression_l3466_346660


namespace quadruple_equation_solutions_l3466_346679

theorem quadruple_equation_solutions :
  {q : ℕ × ℕ × ℕ × ℕ | let (x, y, z, n) := q; x^2 + y^2 + z^2 + 1 = 2^n} =
  {(0,0,0,0), (1,0,0,1), (0,1,0,1), (0,0,1,1), (1,1,1,2)} := by
  sorry

end quadruple_equation_solutions_l3466_346679


namespace range_of_a_l3466_346671

/-- Custom multiplication operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the range of 'a' given the condition -/
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) → -1/2 < a ∧ a < 3/2 := by
  sorry

end range_of_a_l3466_346671


namespace function_comparison_l3466_346645

theorem function_comparison (a : ℝ) (h_a : a > 1/2) :
  ∀ x₁ ∈ Set.Ioc 0 2, ∃ x₂ ∈ Set.Ioc 0 2,
    (1/2 * a * x₁^2 - (2*a + 1) * x₁ + 21) < (x₂^2 - 2*x₂ + Real.exp x₂) := by
  sorry

end function_comparison_l3466_346645


namespace probability_less_than_four_l3466_346630

/-- A square in the 2D plane -/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- The probability that a randomly chosen point in the square satisfies a given condition -/
def probability (s : Square) (condition : ℝ × ℝ → Prop) : ℝ :=
  sorry

/-- The square with vertices at (0,0), (0,3), (3,0), and (3,3) -/
def givenSquare : Square :=
  { bottomLeft := (0, 0),
    topRight := (3, 3) }

/-- The condition x + y < 4 -/
def condition (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 < 4

theorem probability_less_than_four :
  probability givenSquare condition = 7/9 := by
  sorry

end probability_less_than_four_l3466_346630


namespace binomial_coefficient_equality_l3466_346612

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose 8 n = Nat.choose 8 2) → (n = 2 ∨ n = 6) := by
  sorry

end binomial_coefficient_equality_l3466_346612


namespace returning_players_count_l3466_346686

/-- The number of returning players in a baseball team -/
def returning_players (new_players : ℕ) (group_size : ℕ) (num_groups : ℕ) : ℕ :=
  group_size * num_groups - new_players

/-- Theorem stating the number of returning players in the given scenario -/
theorem returning_players_count : returning_players 4 5 2 = 6 := by
  sorry

end returning_players_count_l3466_346686


namespace simplify_expression_l3466_346656

theorem simplify_expression (x : ℝ) : (3*x - 4)*(2*x + 6) - (2*x + 7)*(3*x - 2) = -7*x - 10 := by
  sorry

end simplify_expression_l3466_346656
