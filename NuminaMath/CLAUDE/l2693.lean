import Mathlib

namespace NUMINAMATH_CALUDE_farm_entrance_fee_l2693_269367

theorem farm_entrance_fee (num_students : ℕ) (num_adults : ℕ) (student_fee : ℕ) (total_cost : ℕ) :
  num_students = 35 →
  num_adults = 4 →
  student_fee = 5 →
  total_cost = 199 →
  ∃ (adult_fee : ℕ), 
    adult_fee = 6 ∧ 
    num_students * student_fee + num_adults * adult_fee = total_cost :=
by sorry

end NUMINAMATH_CALUDE_farm_entrance_fee_l2693_269367


namespace NUMINAMATH_CALUDE_max_value_of_f_l2693_269372

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + 3 * x^2 + 5 * x + 2

theorem max_value_of_f :
  ∃ (M : ℝ), M = 31/3 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2693_269372


namespace NUMINAMATH_CALUDE_four_digit_sum_problem_l2693_269347

theorem four_digit_sum_problem :
  ∃ (a b c d : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    0 ≤ d ∧ d ≤ 9 ∧
    a > b ∧ b > c ∧ c > d ∧
    1000 * a + 100 * b + 10 * c + d + 1000 * d + 100 * c + 10 * b + a = 10477 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_sum_problem_l2693_269347


namespace NUMINAMATH_CALUDE_minimum_score_raises_average_l2693_269335

def scores : List ℕ := [92, 88, 74, 65, 80]

def current_average : ℚ := (scores.sum : ℚ) / scores.length

def target_average : ℚ := current_average + 5

def minimum_score : ℕ := 110

theorem minimum_score_raises_average : 
  (((scores.sum + minimum_score) : ℚ) / (scores.length + 1)) = target_average := by sorry

end NUMINAMATH_CALUDE_minimum_score_raises_average_l2693_269335


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2693_269359

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- A sequence is geometric if the ratio between consecutive terms is constant. -/
def IsGeometric (a : Sequence) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem to be proved. -/
theorem geometric_sequence_fourth_term
  (a : Sequence)
  (h1 : a 1 = 2)
  (h2 : IsGeometric (fun n => 1 + a n) 3) :
  a 4 = 80 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2693_269359


namespace NUMINAMATH_CALUDE_parabola_intersection_l2693_269304

theorem parabola_intersection :
  let f (x : ℝ) := 4 * x^2 + 3 * x - 4
  let g (x : ℝ) := 2 * x^2 + 15
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧
    f x₁ = g x₁ ∧ f x₂ = g x₂ ∧
    x₁ = -19/2 ∧ x₂ = 5/2 ∧
    f x₁ = 195.5 ∧ f x₂ = 27.5 ∧
    ∀ (x : ℝ), f x = g x → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2693_269304


namespace NUMINAMATH_CALUDE_greatest_valid_number_l2693_269397

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length > 2 ∧
  ∀ i, 1 < i ∧ i < digits.length - 1 →
    digits[i]! < (digits[i-1]! + digits[i+1]!) / 2

theorem greatest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 986421 :=
sorry

end NUMINAMATH_CALUDE_greatest_valid_number_l2693_269397


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_6_14_22_30_l2693_269395

theorem smallest_perfect_square_divisible_by_6_14_22_30 :
  ∃ (n : ℕ), n > 0 ∧ n = 5336100 ∧ 
  (∃ (k : ℕ), n = k^2) ∧
  6 ∣ n ∧ 14 ∣ n ∧ 22 ∣ n ∧ 30 ∣ n ∧
  (∀ (m : ℕ), m > 0 → (∃ (j : ℕ), m = j^2) → 
    6 ∣ m → 14 ∣ m → 22 ∣ m → 30 ∣ m → m ≥ n) :=
by sorry


end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_6_14_22_30_l2693_269395


namespace NUMINAMATH_CALUDE_m_range_l2693_269306

theorem m_range (x m : ℝ) :
  (∀ x, (1/3 < x ∧ x < 1/2) ↔ |x - m| < 1) →
  -1/2 ≤ m ∧ m ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l2693_269306


namespace NUMINAMATH_CALUDE_repeating_decimal_47_l2693_269349

theorem repeating_decimal_47 :
  ∃ (x : ℚ), (∀ (n : ℕ), (x * 10^(2*n+2) - ⌊x * 10^(2*n+2)⌋ = 47 / 100)) ∧ x = 47 / 99 :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_47_l2693_269349


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2005_l2693_269330

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_2005 :
  ∃ n : ℕ, arithmetic_sequence 1 3 n = 2005 ∧ n = 669 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2005_l2693_269330


namespace NUMINAMATH_CALUDE_prob_red_tile_value_l2693_269366

/-- The number of integers from 1 to 100 that are congruent to 3 mod 7 -/
def red_tiles : ℕ := (Finset.filter (fun n => n % 7 = 3) (Finset.range 100)).card

/-- The total number of tiles -/
def total_tiles : ℕ := 100

/-- The probability of selecting a red tile -/
def prob_red_tile : ℚ := red_tiles / total_tiles

theorem prob_red_tile_value :
  prob_red_tile = 7 / 50 := by sorry

end NUMINAMATH_CALUDE_prob_red_tile_value_l2693_269366


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l2693_269338

-- Define the function f and its properties
def f (x : ℝ) : ℝ := sorry

-- Assume |f'(x)| < 1 for all x in the domain of f
axiom f_deriv_bound (x : ℝ) : |deriv f x| < 1

-- Part 1
theorem part_one (a : ℝ) (h : ∀ x ∈ Set.Icc 1 2, f x = a * x + Real.log x) :
  a ∈ Set.Ioo (-3/2) 0 := sorry

-- Part 2
theorem part_two : ∃! x, f x = x := sorry

-- Part 3
def is_periodic (f : ℝ → ℝ) (p : ℝ) :=
  ∀ x, f (x + p) = f x

theorem part_three (h : is_periodic f 2) :
  ∀ x₁ x₂ : ℝ, |f x₁ - f x₂| < 1 := sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l2693_269338


namespace NUMINAMATH_CALUDE_square_EFGH_product_l2693_269365

/-- A point on a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A square on the grid -/
structure GridSquare where
  E : GridPoint
  F : GridPoint
  G : GridPoint
  H : GridPoint

/-- The side length of a square given two of its corners -/
def sideLength (p1 p2 : GridPoint) : ℤ :=
  max (abs (p1.x - p2.x)) (abs (p1.y - p2.y))

/-- The area of a square -/
def area (s : GridSquare) : ℤ :=
  (sideLength s.E s.F) ^ 2

/-- The perimeter of a square -/
def perimeter (s : GridSquare) : ℤ :=
  4 * (sideLength s.E s.F)

theorem square_EFGH_product :
  ∃ (s : GridSquare),
    s.E = ⟨1, 5⟩ ∧
    s.F = ⟨5, 5⟩ ∧
    s.G = ⟨5, 1⟩ ∧
    s.H = ⟨1, 1⟩ ∧
    (area s * perimeter s = 256) := by
  sorry

end NUMINAMATH_CALUDE_square_EFGH_product_l2693_269365


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l2693_269314

theorem quadratic_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + (2*m - 3)*x + m - 150 = 0 ∧
               y^2 + (2*m - 3)*y + m - 150 = 0 ∧
               x > 2 ∧ y < 2) ↔
  m > 5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l2693_269314


namespace NUMINAMATH_CALUDE_friends_bill_split_l2693_269345

-- Define the problem parameters
def num_friends : ℕ := 5
def original_bill : ℚ := 100
def discount_percentage : ℚ := 6

-- Define the theorem
theorem friends_bill_split :
  let discount := discount_percentage / 100 * original_bill
  let discounted_bill := original_bill - discount
  let individual_payment := discounted_bill / num_friends
  individual_payment = 18.8 := by sorry

end NUMINAMATH_CALUDE_friends_bill_split_l2693_269345


namespace NUMINAMATH_CALUDE_distance_ratio_theorem_l2693_269331

theorem distance_ratio_theorem (x : ℝ) (h1 : x^2 + (-4)^2 = 8^2) :
  |(-4)| / 8 = 1/2 := by sorry

end NUMINAMATH_CALUDE_distance_ratio_theorem_l2693_269331


namespace NUMINAMATH_CALUDE_toy_box_paths_l2693_269301

/-- Represents a rectangular grid --/
structure Grid :=
  (length : ℕ)
  (width : ℕ)

/-- Calculates the number of paths in a grid from one corner to the opposite corner,
    moving only right and up, covering a specific total distance --/
def numPaths (g : Grid) (totalDistance : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that for a 50x40 grid with total distance 90,
    there are 12 possible paths --/
theorem toy_box_paths :
  let g : Grid := { length := 50, width := 40 }
  numPaths g 90 = 12 := by
  sorry

end NUMINAMATH_CALUDE_toy_box_paths_l2693_269301


namespace NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_gt_zero_l2693_269307

theorem x_eq_one_sufficient_not_necessary_for_x_gt_zero :
  (∃ x : ℝ, x = 1 → x > 0) ∧
  (∃ x : ℝ, x > 0 ∧ x ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_gt_zero_l2693_269307


namespace NUMINAMATH_CALUDE_tangent_line_triangle_area_l2693_269387

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 3

-- State the theorem
theorem tangent_line_triangle_area (a : ℝ) : 
  (f' a 1 = -6) →  -- Condition for perpendicularity
  (∃ b c : ℝ, 
    (∀ x : ℝ, -6 * x + b = c * x + f a 1) ∧  -- Equation of tangent line
    (b = 6) ∧  -- y-intercept of tangent line
    (c = -6)) →  -- Slope of tangent line
  (1/2 * 1 * 6 = 3) :=  -- Area of triangle
by sorry

end NUMINAMATH_CALUDE_tangent_line_triangle_area_l2693_269387


namespace NUMINAMATH_CALUDE_smallest_product_l2693_269315

def digits : List Nat := [1, 2, 3, 4]

def is_valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat,
    is_valid_arrangement a b c d →
    product a b c d ≥ 312 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_l2693_269315


namespace NUMINAMATH_CALUDE_jessie_dimes_l2693_269381

/-- Represents the contents of Jessie's piggy bank -/
structure PiggyBank where
  dimes : ℕ
  quarters : ℕ
  total_cents : ℕ
  dime_quarter_difference : ℕ

/-- The piggy bank satisfies the given conditions -/
def valid_piggy_bank (pb : PiggyBank) : Prop :=
  pb.dimes = pb.quarters + pb.dime_quarter_difference ∧
  pb.total_cents = 10 * pb.dimes + 25 * pb.quarters ∧
  pb.dime_quarter_difference = 10 ∧
  pb.total_cents = 580

/-- The theorem stating that Jessie has 23 dimes -/
theorem jessie_dimes (pb : PiggyBank) (h : valid_piggy_bank pb) : pb.dimes = 23 := by
  sorry

end NUMINAMATH_CALUDE_jessie_dimes_l2693_269381


namespace NUMINAMATH_CALUDE_biff_break_even_hours_l2693_269341

/-- Calculates the number of hours needed to break even given expenses and income rates -/
def hours_to_break_even (ticket_cost snacks_cost headphones_cost hourly_rate wifi_cost : ℚ) : ℚ :=
  let total_expenses := ticket_cost + snacks_cost + headphones_cost
  let net_hourly_rate := hourly_rate - wifi_cost
  total_expenses / net_hourly_rate

/-- Proves that Biff needs 3 hours to break even given his expenses and income rates -/
theorem biff_break_even_hours :
  hours_to_break_even 11 3 16 12 2 = 3 := by
  sorry

#eval hours_to_break_even 11 3 16 12 2

end NUMINAMATH_CALUDE_biff_break_even_hours_l2693_269341


namespace NUMINAMATH_CALUDE_dividend_calculation_l2693_269346

/-- Calculates the dividend received from an investment in shares -/
theorem dividend_calculation (investment : ℝ) (face_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ)
  (h1 : investment = 14400)
  (h2 : face_value = 100)
  (h3 : premium_rate = 0.2)
  (h4 : dividend_rate = 0.07) :
  let share_cost := face_value * (1 + premium_rate)
  let num_shares := investment / share_cost
  let dividend_per_share := face_value * dividend_rate
  num_shares * dividend_per_share = 840 := by sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2693_269346


namespace NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l2693_269357

/-- A geometric sequence with the given properties -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  (a 0 = 2048) ∧ 
  (a 1 = 512) ∧ 
  (a 2 = 128) ∧ 
  (a 5 = 2) ∧ 
  ∀ n, a (n + 1) = a n * (a 1 / a 0)

/-- The sum of the fourth and fifth terms in the sequence is 40 -/
theorem sum_of_fourth_and_fifth_terms (a : ℕ → ℚ) 
  (h : geometric_sequence a) : a 3 + a 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l2693_269357


namespace NUMINAMATH_CALUDE_bus_ride_difference_l2693_269376

/-- Proves that 15 more children got on the bus than got off during the entire ride -/
theorem bus_ride_difference (initial : ℕ) (final : ℕ) 
  (got_off_first : ℕ) (got_off_second : ℕ) (got_off_third : ℕ) 
  (h1 : initial = 20)
  (h2 : final = 35)
  (h3 : got_off_first = 54)
  (h4 : got_off_second = 30)
  (h5 : got_off_third = 15) :
  ∃ (got_on_total : ℕ),
    got_on_total = final - initial + got_off_first + got_off_second + got_off_third ∧
    got_on_total - (got_off_first + got_off_second + got_off_third) = 15 := by
  sorry


end NUMINAMATH_CALUDE_bus_ride_difference_l2693_269376


namespace NUMINAMATH_CALUDE_impossible_belief_l2693_269340

-- Define the characters
inductive Character : Type
| King : Character
| Queen : Character

-- Define the state of mind
inductive MindState : Type
| Sane : MindState
| NotSane : MindState

-- Define a belief
structure Belief where
  subject : Character
  object : Character
  state : MindState

-- Define a nested belief
structure NestedBelief where
  level1 : Character
  level2 : Character
  level3 : Character
  finalBelief : Belief

-- Define logical consistency
def logicallyConsistent (c : Character) : Prop :=
  ∀ (b : Belief), b.subject = c → (b.state = MindState.Sane ↔ c = b.object)

-- Define the problematic belief
def problematicBelief : NestedBelief :=
  { level1 := Character.King
  , level2 := Character.Queen
  , level3 := Character.King
  , finalBelief := { subject := Character.King
                   , object := Character.Queen
                   , state := MindState.NotSane } }

-- Theorem statement
theorem impossible_belief
  (h1 : logicallyConsistent Character.King)
  (h2 : logicallyConsistent Character.Queen) :
  ¬ (∃ (b : NestedBelief), b = problematicBelief) :=
sorry

end NUMINAMATH_CALUDE_impossible_belief_l2693_269340


namespace NUMINAMATH_CALUDE_orchid_to_rose_ratio_l2693_269394

/-- Proves that the ratio of orchids to roses in each centerpiece is 2:1 given the specified conditions. -/
theorem orchid_to_rose_ratio 
  (num_centerpieces : ℕ) 
  (roses_per_centerpiece : ℕ) 
  (lilies_per_centerpiece : ℕ) 
  (total_budget : ℕ) 
  (cost_per_flower : ℕ) 
  (h1 : num_centerpieces = 6)
  (h2 : roses_per_centerpiece = 8)
  (h3 : lilies_per_centerpiece = 6)
  (h4 : total_budget = 2700)
  (h5 : cost_per_flower = 15) : 
  ∃ (orchids_per_centerpiece : ℕ), 
    orchids_per_centerpiece = 2 * roses_per_centerpiece :=
by sorry

end NUMINAMATH_CALUDE_orchid_to_rose_ratio_l2693_269394


namespace NUMINAMATH_CALUDE_spike_cricket_count_l2693_269371

/-- The number of crickets Spike hunts in the morning -/
def morning_crickets : ℕ := 5

/-- The number of crickets Spike hunts in the afternoon and evening -/
def afternoon_evening_crickets : ℕ := 3 * morning_crickets

/-- The total number of crickets Spike hunts per day -/
def total_crickets : ℕ := morning_crickets + afternoon_evening_crickets

theorem spike_cricket_count : total_crickets = 20 := by
  sorry

end NUMINAMATH_CALUDE_spike_cricket_count_l2693_269371


namespace NUMINAMATH_CALUDE_midpoint_chain_l2693_269352

/-- Given a line segment XY with midpoints as described, prove that XY = 80 when XJ = 5 -/
theorem midpoint_chain (X Y G H I J : ℝ) : 
  (G = (X + Y) / 2) →  -- G is midpoint of XY
  (H = (X + G) / 2) →  -- H is midpoint of XG
  (I = (X + H) / 2) →  -- I is midpoint of XH
  (J = (X + I) / 2) →  -- J is midpoint of XI
  (J - X = 5) →        -- XJ = 5
  (Y - X = 80) :=      -- XY = 80
by sorry

end NUMINAMATH_CALUDE_midpoint_chain_l2693_269352


namespace NUMINAMATH_CALUDE_trig_range_equivalence_l2693_269327

theorem trig_range_equivalence (α : Real) :
  (0 < α ∧ α < 2 * Real.pi) →
  (Real.sin α < Real.sqrt 3 / 2 ∧ Real.cos α > 1 / 2) ↔
  ((0 < α ∧ α < Real.pi / 3) ∨ (5 * Real.pi / 3 < α ∧ α < 2 * Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_trig_range_equivalence_l2693_269327


namespace NUMINAMATH_CALUDE_difference_given_sum_and_difference_of_squares_l2693_269316

theorem difference_given_sum_and_difference_of_squares (x y : ℝ) 
  (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end NUMINAMATH_CALUDE_difference_given_sum_and_difference_of_squares_l2693_269316


namespace NUMINAMATH_CALUDE_selection_problem_l2693_269396

theorem selection_problem (n : ℕ) (r : ℕ) (h1 : n = 10) (h2 : r = 4) :
  Nat.choose n r = 210 := by
  sorry

end NUMINAMATH_CALUDE_selection_problem_l2693_269396


namespace NUMINAMATH_CALUDE_salt_solution_problem_l2693_269373

theorem salt_solution_problem (initial_volume : ℝ) (added_water : ℝ) (final_salt_percentage : ℝ) :
  initial_volume = 80 →
  added_water = 20 →
  final_salt_percentage = 8 →
  let final_volume := initial_volume + added_water
  let initial_salt_amount := (initial_volume * final_salt_percentage) / final_volume
  let initial_salt_percentage := (initial_salt_amount / initial_volume) * 100
  initial_salt_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_salt_solution_problem_l2693_269373


namespace NUMINAMATH_CALUDE_president_and_committee_count_l2693_269322

/-- The number of ways to choose a president and a 2-person committee -/
def choose_president_and_committee (total_people : ℕ) (people_over_30 : ℕ) : ℕ :=
  total_people * (people_over_30 * (people_over_30 - 1) / 2 + 
  (total_people - people_over_30) * people_over_30 * (people_over_30 - 1) / 2)

/-- Theorem stating the number of ways to choose a president and committee -/
theorem president_and_committee_count :
  choose_president_and_committee 10 6 = 120 := by sorry

end NUMINAMATH_CALUDE_president_and_committee_count_l2693_269322


namespace NUMINAMATH_CALUDE_not_divisible_by_169_l2693_269393

theorem not_divisible_by_169 (n : ℕ) : ¬(169 ∣ (n^2 + 5*n + 16)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_169_l2693_269393


namespace NUMINAMATH_CALUDE_sum_of_coordinates_D_l2693_269360

/-- Given points C and M, where M is the midpoint of segment CD, 
    prove that the sum of coordinates of point D is 0 -/
theorem sum_of_coordinates_D (C M : ℝ × ℝ) (h1 : C = (-1, 5)) (h2 : M = (4, -2)) : 
  let D := (2 * M.1 - C.1, 2 * M.2 - C.2)
  D.1 + D.2 = 0 := by
sorry


end NUMINAMATH_CALUDE_sum_of_coordinates_D_l2693_269360


namespace NUMINAMATH_CALUDE_x9_plus_y9_not_eq_neg_one_l2693_269362

theorem x9_plus_y9_not_eq_neg_one :
  ∀ (x y : ℂ),
  x = (-1 + Complex.I * Real.sqrt 3) / 2 →
  y = (-1 - Complex.I * Real.sqrt 3) / 2 →
  x^9 + y^9 ≠ -1 := by
sorry

end NUMINAMATH_CALUDE_x9_plus_y9_not_eq_neg_one_l2693_269362


namespace NUMINAMATH_CALUDE_max_d_value_l2693_269358

def a (n : ℕ+) : ℕ := 80 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  (∃ n : ℕ+, d n = 5) ∧ (∀ n : ℕ+, d n ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l2693_269358


namespace NUMINAMATH_CALUDE_perimeter_implies_equilateral_l2693_269383

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Define an equilateral triangle
def is_equilateral (t : Triangle) : Prop := t.a = t.b ∧ t.b = t.c

-- Theorem statement
theorem perimeter_implies_equilateral (t : Triangle) :
  perimeter t = 3 + 2 * Real.sqrt 3 → is_equilateral t := by
  sorry

end NUMINAMATH_CALUDE_perimeter_implies_equilateral_l2693_269383


namespace NUMINAMATH_CALUDE_power_inequality_l2693_269389

theorem power_inequality : 2^(1/5) > 0.4^(1/5) ∧ 0.4^(1/5) > 0.4^(3/5) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2693_269389


namespace NUMINAMATH_CALUDE_handshakes_count_l2693_269368

/-- Represents the social event with given conditions -/
structure SocialEvent where
  total_people : ℕ
  group_a_size : ℕ
  group_b_size : ℕ
  group_a_knows_all : group_a_size = 25
  group_b_knows_one : group_b_size = 15
  total_is_sum : total_people = group_a_size + group_b_size

/-- Calculates the number of handshakes in the social event -/
def count_handshakes (event : SocialEvent) : ℕ :=
  let group_b_internal_handshakes := (event.group_b_size * (event.group_b_size - 1)) / 2
  let group_a_b_handshakes := event.group_b_size * (event.group_a_size - 1)
  group_b_internal_handshakes + group_a_b_handshakes

/-- Theorem stating that the number of handshakes in the given social event is 465 -/
theorem handshakes_count (event : SocialEvent) : count_handshakes event = 465 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_count_l2693_269368


namespace NUMINAMATH_CALUDE_divisibility_by_29_fourth_power_l2693_269399

theorem divisibility_by_29_fourth_power (x y z : ℤ) (S : ℤ) 
  (h1 : S = x^4 + y^4 + z^4) 
  (h2 : 29 ∣ S) : 
  29^4 ∣ S := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_29_fourth_power_l2693_269399


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2693_269391

theorem inequality_solution_set (x : ℝ) : 3 * x + 2 > 5 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2693_269391


namespace NUMINAMATH_CALUDE_line_obtuse_angle_a_range_l2693_269308

/-- Given a line passing through points K(1-a, 1+a) and Q(3, 2a),
    if the line forms an obtuse angle, then a is in the open interval (-2, 1). -/
theorem line_obtuse_angle_a_range (a : ℝ) :
  let K : ℝ × ℝ := (1 - a, 1 + a)
  let Q : ℝ × ℝ := (3, 2 * a)
  let m : ℝ := (Q.2 - K.2) / (Q.1 - K.1)
  (m < 0) → a ∈ Set.Ioo (-2 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_line_obtuse_angle_a_range_l2693_269308


namespace NUMINAMATH_CALUDE_ages_sum_l2693_269324

theorem ages_sum (kiana_age twin_age : ℕ) : 
  kiana_age > twin_age →
  kiana_age * twin_age * twin_age = 72 →
  kiana_age + twin_age + twin_age = 14 :=
by sorry

end NUMINAMATH_CALUDE_ages_sum_l2693_269324


namespace NUMINAMATH_CALUDE_roots_properties_l2693_269313

def i : ℂ := Complex.I

def quadratic_equation (z : ℂ) : Prop :=
  z^2 + 2*z = -4 + 8*i

def roots (z₁ z₂ : ℂ) : Prop :=
  quadratic_equation z₁ ∧ quadratic_equation z₂ ∧ z₁ ≠ z₂

theorem roots_properties :
  ∃ z₁ z₂ : ℂ, roots z₁ z₂ ∧
  (z₁.re * z₂.re = -7) ∧
  (z₁.im + z₂.im = 0) := by sorry

end NUMINAMATH_CALUDE_roots_properties_l2693_269313


namespace NUMINAMATH_CALUDE_abs_and_recip_of_neg_one_point_two_l2693_269334

theorem abs_and_recip_of_neg_one_point_two :
  let x : ℝ := -1.2
  abs x = 1.2 ∧ x⁻¹ = -5/6 := by
  sorry

end NUMINAMATH_CALUDE_abs_and_recip_of_neg_one_point_two_l2693_269334


namespace NUMINAMATH_CALUDE_f_lower_bound_g_inequality_min_a_l2693_269328

noncomputable section

variables (x : ℝ) (a : ℝ)

def f (x : ℝ) (a : ℝ) : ℝ := a * Real.exp (2*x - 1) - x^2 * (Real.log x + 1/2)

def g (x : ℝ) (a : ℝ) : ℝ := x * f x a + x^2 / Real.exp x

theorem f_lower_bound (h : x > 0) : f x 0 ≥ x^2/2 - x^3 := by sorry

theorem g_inequality_min_a :
  (∀ x > 1, x * g (Real.log x / (x - 1)) a < g (x * Real.log x / (x - 1)) a) ↔ a ≥ 1 / Real.exp 1 := by sorry

end

end NUMINAMATH_CALUDE_f_lower_bound_g_inequality_min_a_l2693_269328


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l2693_269380

theorem final_sum_after_operations (a b S : ℝ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := by sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l2693_269380


namespace NUMINAMATH_CALUDE_bakers_purchase_cost_l2693_269339

/-- Calculate the total cost in dollars after discount for a baker's purchase -/
theorem bakers_purchase_cost (flour_price : ℝ) (egg_price : ℝ) (milk_price : ℝ) (soda_price : ℝ)
  (discount_rate : ℝ) (exchange_rate : ℝ) :
  flour_price = 6 →
  egg_price = 12 →
  milk_price = 3 →
  soda_price = 1.5 →
  discount_rate = 0.15 →
  exchange_rate = 1.2 →
  let total_euro := 5 * flour_price + 6 * egg_price + 8 * milk_price + 4 * soda_price
  let discounted_euro := total_euro * (1 - discount_rate)
  let total_dollar := discounted_euro * exchange_rate
  total_dollar = 134.64 := by
sorry

end NUMINAMATH_CALUDE_bakers_purchase_cost_l2693_269339


namespace NUMINAMATH_CALUDE_birds_on_fence_l2693_269337

theorem birds_on_fence (initial_birds joining_birds : ℕ) :
  initial_birds + joining_birds = initial_birds + joining_birds :=
by sorry

end NUMINAMATH_CALUDE_birds_on_fence_l2693_269337


namespace NUMINAMATH_CALUDE_remainder_sum_mod_seven_l2693_269319

theorem remainder_sum_mod_seven (a b c : ℕ) : 
  a < 7 → b < 7 → c < 7 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 7 = 2 →
  (3 * c) % 7 = 1 →
  (4 * b) % 7 = (2 + b) % 7 →
  (a + b + c) % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_seven_l2693_269319


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l2693_269311

theorem trigonometric_expression_equality : 
  (Real.sin (10 * π / 180) * Real.sin (80 * π / 180)) / 
  (Real.cos (35 * π / 180)^2 - Real.sin (35 * π / 180)^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l2693_269311


namespace NUMINAMATH_CALUDE_sum_of_digits_in_special_addition_formula_l2693_269388

/-- Represents a four-digit number ABCD -/
def fourDigitNumber (a b c d : Nat) : Nat :=
  1000 * a + 100 * b + 10 * c + d

/-- The main theorem -/
theorem sum_of_digits_in_special_addition_formula 
  (A B C D : Nat) 
  (h_different : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h_single_digit : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10)
  (h_formula : fourDigitNumber D C B A + fourDigitNumber A B C D = 10 * fourDigitNumber A B C D) :
  A + B + C + D = 18 := by
    sorry

end NUMINAMATH_CALUDE_sum_of_digits_in_special_addition_formula_l2693_269388


namespace NUMINAMATH_CALUDE_equal_distribution_of_drawings_l2693_269386

theorem equal_distribution_of_drawings (total_drawings : ℕ) (num_neighbors : ℕ) 
  (h1 : total_drawings = 54) (h2 : num_neighbors = 6) :
  total_drawings / num_neighbors = 9 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_of_drawings_l2693_269386


namespace NUMINAMATH_CALUDE_polynomial_equality_l2693_269309

theorem polynomial_equality (a k n : ℚ) :
  (∀ x : ℚ, (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) →
  a - n + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2693_269309


namespace NUMINAMATH_CALUDE_range_of_expression_l2693_269326

theorem range_of_expression (x y : ℝ) (h1 : x + 2*y - 6 = 0) (h2 : 0 < x) (h3 : x < 3) :
  1 < (x + 2) / (y - 1) ∧ (x + 2) / (y - 1) < 10 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l2693_269326


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2693_269310

theorem complex_modulus_problem (Z : ℂ) (a : ℝ) :
  Z = 3 + a * I ∧ Complex.abs Z = 5 → a = 4 ∨ a = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2693_269310


namespace NUMINAMATH_CALUDE_article_price_after_decrease_l2693_269361

theorem article_price_after_decrease (decreased_price : ℝ) (decrease_percentage : ℝ) (original_price : ℝ) : 
  decreased_price = 532 → 
  decrease_percentage = 24 → 
  decreased_price = original_price * (1 - decrease_percentage / 100) → 
  original_price = 700 := by
sorry

end NUMINAMATH_CALUDE_article_price_after_decrease_l2693_269361


namespace NUMINAMATH_CALUDE_mystery_book_shelves_l2693_269300

theorem mystery_book_shelves (total_books : ℕ) (books_per_shelf : ℕ) (picture_shelves : ℕ) :
  total_books = 72 →
  books_per_shelf = 9 →
  picture_shelves = 5 →
  (total_books - picture_shelves * books_per_shelf) / books_per_shelf = 3 :=
by sorry

end NUMINAMATH_CALUDE_mystery_book_shelves_l2693_269300


namespace NUMINAMATH_CALUDE_captains_age_and_crew_size_l2693_269343

theorem captains_age_and_crew_size (l k : ℕ) : 
  l * (l - 1) = k * (l - 2) + 15 → 
  ((l = 1 ∧ k = 15) ∨ (l = 15 ∧ k = 15)) := by
sorry

end NUMINAMATH_CALUDE_captains_age_and_crew_size_l2693_269343


namespace NUMINAMATH_CALUDE_base_3_of_121_l2693_269323

def base_3_representation (n : ℕ) : List ℕ :=
  sorry

theorem base_3_of_121 :
  base_3_representation 121 = [1, 1, 1, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_base_3_of_121_l2693_269323


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l2693_269385

theorem sqrt_product_simplification (x : ℝ) :
  Real.sqrt (96 * x^2) * Real.sqrt (50 * x) * Real.sqrt (28 * x^3) = 1260 * x^3 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l2693_269385


namespace NUMINAMATH_CALUDE_football_team_numbers_l2693_269303

theorem football_team_numbers (x : ℕ) (n : ℕ) : 
  (n * (n + 1)) / 2 - x = 100 → x = 5 ∧ n = 14 := by
  sorry

end NUMINAMATH_CALUDE_football_team_numbers_l2693_269303


namespace NUMINAMATH_CALUDE_total_earrings_l2693_269369

/-- Proves that the total number of earrings for Bella, Monica, and Rachel is 70 -/
theorem total_earrings (bella_earrings : ℕ) (monica_earrings : ℕ) (rachel_earrings : ℕ)
  (h1 : bella_earrings = 10)
  (h2 : bella_earrings = monica_earrings / 4)
  (h3 : monica_earrings = 2 * rachel_earrings) :
  bella_earrings + monica_earrings + rachel_earrings = 70 := by
  sorry

#check total_earrings

end NUMINAMATH_CALUDE_total_earrings_l2693_269369


namespace NUMINAMATH_CALUDE_order_of_magnitude_l2693_269320

theorem order_of_magnitude (a b : ℝ) (ha : a > 0) (hb : b < 0) (hab : |a| < |b|) :
  -b > a ∧ a > -a ∧ -a > b := by
  sorry

end NUMINAMATH_CALUDE_order_of_magnitude_l2693_269320


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_in_circle_l2693_269329

/-- Given an isosceles right triangle inscribed in a circle with radius √2,
    where the side lengths are in the ratio 1:1:√2,
    prove that the area of the triangle is 2 and the circumference of the circle is 2π√2. -/
theorem isosceles_right_triangle_in_circle 
  (r : ℝ) 
  (h_r : r = Real.sqrt 2) 
  (a b c : ℝ) 
  (h_abc : a = b ∧ c = a * Real.sqrt 2) 
  (h_inscribed : c = 2 * r) : 
  (1/2 * a * b = 2) ∧ (2 * Real.pi * r = 2 * Real.pi * Real.sqrt 2) := by
  sorry

#check isosceles_right_triangle_in_circle

end NUMINAMATH_CALUDE_isosceles_right_triangle_in_circle_l2693_269329


namespace NUMINAMATH_CALUDE_rad_divides_theorem_l2693_269348

-- Define the rad function
def rad : ℕ → ℕ
| 0 => 1
| 1 => 1
| n+2 => (Finset.prod (Nat.factors (n+2)).toFinset id)

-- Define a polynomial with nonnegative integer coefficients
def NonnegIntPoly := {f : Polynomial ℕ // ∀ i, 0 ≤ f.coeff i}

theorem rad_divides_theorem (f : NonnegIntPoly) :
  (∀ n : ℕ, rad (f.val.eval n) ∣ rad (f.val.eval (n^(rad n)))) →
  ∃ a m : ℕ, f.val = Polynomial.monomial m a :=
sorry

end NUMINAMATH_CALUDE_rad_divides_theorem_l2693_269348


namespace NUMINAMATH_CALUDE_birds_can_gather_l2693_269374

/-- Represents a configuration of birds on trees -/
structure BirdConfiguration (n : ℕ) where
  positions : Fin n → Fin n

/-- The sum of bird positions in a configuration -/
def sum_positions (n : ℕ) (config : BirdConfiguration n) : ℕ :=
  (Finset.univ.sum fun i => (config.positions i).val) + n

/-- A bird movement that preserves the sum of positions -/
def valid_movement (n : ℕ) (config1 config2 : BirdConfiguration n) : Prop :=
  sum_positions n config1 = sum_positions n config2

/-- All birds are on the same tree -/
def all_gathered (n : ℕ) (config : BirdConfiguration n) : Prop :=
  ∃ k : Fin n, ∀ i : Fin n, config.positions i = k

/-- Initial configuration with one bird on each tree -/
def initial_config (n : ℕ) : BirdConfiguration n :=
  ⟨id⟩

/-- Theorem: Birds can gather on one tree iff n is odd and greater than 1 -/
theorem birds_can_gather (n : ℕ) :
  (∃ (config : BirdConfiguration n), valid_movement n (initial_config n) config ∧ all_gathered n config) ↔
  n % 2 = 1 ∧ n > 1 :=
sorry

end NUMINAMATH_CALUDE_birds_can_gather_l2693_269374


namespace NUMINAMATH_CALUDE_competition_tables_l2693_269351

theorem competition_tables (total_legs : ℕ) : total_legs = 816 →
  ∃ (num_tables : ℕ),
    num_tables * (3 * 8 + 6 * 2 + 4) = total_legs ∧
    num_tables = 20 := by
  sorry

end NUMINAMATH_CALUDE_competition_tables_l2693_269351


namespace NUMINAMATH_CALUDE_string_average_length_l2693_269344

theorem string_average_length :
  let string1 : ℚ := 4
  let string2 : ℚ := 5
  let string3 : ℚ := 7
  let num_strings : ℕ := 3
  (string1 + string2 + string3) / num_strings = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_string_average_length_l2693_269344


namespace NUMINAMATH_CALUDE_complex_number_calculation_l2693_269317

theorem complex_number_calculation : 
  let z : ℂ := 1 + Complex.I * Real.sqrt 2
  z^2 - 2*z = -3 := by sorry

end NUMINAMATH_CALUDE_complex_number_calculation_l2693_269317


namespace NUMINAMATH_CALUDE_research_budget_allocation_l2693_269336

theorem research_budget_allocation (microphotonics : ℝ) (home_electronics : ℝ)
  (genetically_modified_microorganisms : ℝ) (industrial_lubricants : ℝ)
  (basic_astrophysics_degrees : ℝ) :
  microphotonics = 14 →
  home_electronics = 24 →
  genetically_modified_microorganisms = 29 →
  industrial_lubricants = 8 →
  basic_astrophysics_degrees = 18 →
  ∃ (food_additives : ℝ),
    food_additives = 20 ∧
    microphotonics + home_electronics + genetically_modified_microorganisms +
    industrial_lubricants + (basic_astrophysics_degrees / 360 * 100) + food_additives = 100 :=
by sorry

end NUMINAMATH_CALUDE_research_budget_allocation_l2693_269336


namespace NUMINAMATH_CALUDE_line_through_intersection_with_equal_intercepts_l2693_269312

/-- The equation of a line passing through the intersection of two given lines and having equal intercepts on the coordinate axes -/
theorem line_through_intersection_with_equal_intercepts :
  ∃ (a b c : ℝ),
    (∀ x y : ℝ, x + 2*y - 6 = 0 ∧ x - 2*y + 2 = 0 → a*x + b*y + c = 0) ∧
    (∃ x₁ x₂ : ℝ, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ a*x₁ + c = 0 ∧ b*x₂ + c = 0 ∧ x₁ = x₂) →
    (a = 1 ∧ b = -1 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -4) := by
  sorry

end NUMINAMATH_CALUDE_line_through_intersection_with_equal_intercepts_l2693_269312


namespace NUMINAMATH_CALUDE_triangle_c_range_and_perimeter_l2693_269382

-- Define the triangle sides and conditions
def triangle_sides (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_c_range_and_perimeter
  (a b c : ℝ)
  (h_sides : triangle_sides a b c)
  (h_sum : a + b = 3 * c - 2)
  (h_diff : a - b = 2 * c - 6) :
  (1 < c ∧ c < 6) ∧
  (a + b + c = 18 → c = 5) := by
sorry


end NUMINAMATH_CALUDE_triangle_c_range_and_perimeter_l2693_269382


namespace NUMINAMATH_CALUDE_f_symmetric_l2693_269390

/-- The number of integer sequences of length n with sum of absolute values not exceeding m -/
def f (n m : ℕ) : ℕ := sorry

/-- Theorem stating that f(a, b) = f(b, a) for positive integers a and b -/
theorem f_symmetric {a b : ℕ} (ha : 0 < a) (hb : 0 < b) : f a b = f b a := by sorry

end NUMINAMATH_CALUDE_f_symmetric_l2693_269390


namespace NUMINAMATH_CALUDE_fourth_episode_length_l2693_269364

theorem fourth_episode_length 
  (episode1 : ℕ) 
  (episode2 : ℕ) 
  (episode3 : ℕ) 
  (total_duration : ℕ) 
  (h1 : episode1 = 58)
  (h2 : episode2 = 62)
  (h3 : episode3 = 65)
  (h4 : total_duration = 240) :
  total_duration - (episode1 + episode2 + episode3) = 55 :=
by sorry

end NUMINAMATH_CALUDE_fourth_episode_length_l2693_269364


namespace NUMINAMATH_CALUDE_mrs_hilt_pie_arrangement_l2693_269305

/-- Given the number of pecan pies, apple pies, and rows, 
    calculate the number of pies in each row -/
def piesPerRow (pecanPies applePies rows : ℕ) : ℕ :=
  (pecanPies + applePies) / rows

/-- Theorem: Given 16 pecan pies, 14 apple pies, and 30 rows,
    the number of pies in each row is 1 -/
theorem mrs_hilt_pie_arrangement :
  piesPerRow 16 14 30 = 1 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_pie_arrangement_l2693_269305


namespace NUMINAMATH_CALUDE_sticker_distribution_l2693_269354

theorem sticker_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 5) :
  (Nat.choose (n + k - 1) (k - 1)) = 1001 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2693_269354


namespace NUMINAMATH_CALUDE_max_sum_of_cubes_l2693_269333

/-- Given real numbers a, b, c, d satisfying the condition,
    the sum of their cubes is bounded above by 4√10 -/
theorem max_sum_of_cubes (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d = 10) : 
  a^3 + b^3 + c^3 + d^3 ≤ 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_cubes_l2693_269333


namespace NUMINAMATH_CALUDE_ant_final_position_l2693_269392

/-- Represents a point in 2D space -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents the state of the ant -/
structure AntState where
  position : Point
  direction : Direction
  moveCount : Nat
  moveDistance : Nat

/-- Function to update the ant's state after a move -/
def move (state : AntState) : AntState :=
  match state.direction with
  | Direction.North => { state with position := ⟨state.position.x, state.position.y + state.moveDistance⟩, direction := Direction.East }
  | Direction.East => { state with position := ⟨state.position.x + state.moveDistance, state.position.y⟩, direction := Direction.South }
  | Direction.South => { state with position := ⟨state.position.x, state.position.y - state.moveDistance⟩, direction := Direction.West }
  | Direction.West => { state with position := ⟨state.position.x - state.moveDistance, state.position.y⟩, direction := Direction.North }

/-- Function to perform multiple moves -/
def multiMove (initialState : AntState) (n : Nat) : AntState :=
  match n with
  | 0 => initialState
  | m + 1 => 
    let newState := move initialState
    multiMove { newState with moveCount := newState.moveCount + 1, moveDistance := newState.moveDistance + 2 } m

/-- Theorem stating the final position of the ant -/
theorem ant_final_position :
  let initialState : AntState := {
    position := ⟨10, -10⟩,
    direction := Direction.North,
    moveCount := 0,
    moveDistance := 2
  }
  let finalState := multiMove initialState 10
  finalState.position = ⟨22, 0⟩ := by
  sorry


end NUMINAMATH_CALUDE_ant_final_position_l2693_269392


namespace NUMINAMATH_CALUDE_sample_size_is_ten_l2693_269378

/-- A structure representing a quality inspection scenario -/
structure QualityInspection where
  total_products : ℕ
  selected_products : ℕ

/-- Definition of sample size for a quality inspection -/
def sample_size (qi : QualityInspection) : ℕ := qi.selected_products

/-- Theorem stating that for the given scenario, the sample size is 10 -/
theorem sample_size_is_ten (qi : QualityInspection) 
  (h1 : qi.total_products = 80) 
  (h2 : qi.selected_products = 10) : 
  sample_size qi = 10 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_ten_l2693_269378


namespace NUMINAMATH_CALUDE_chocolate_candy_difference_l2693_269325

/-- The difference in cost between chocolate and candy bar -/
def cost_difference (chocolate_cost candy_cost : ℕ) : ℕ :=
  chocolate_cost - candy_cost

/-- Theorem stating the difference in cost between chocolate and candy bar -/
theorem chocolate_candy_difference :
  cost_difference 7 2 = 5 := by sorry

end NUMINAMATH_CALUDE_chocolate_candy_difference_l2693_269325


namespace NUMINAMATH_CALUDE_min_container_cost_l2693_269353

def container_cost (a b : ℝ) : ℝ := 20 * (a * b) + 10 * 2 * (a + b)

theorem min_container_cost :
  ∀ a b : ℝ,
  a > 0 → b > 0 →
  a * b = 4 →
  container_cost a b ≥ 160 :=
by
  sorry

end NUMINAMATH_CALUDE_min_container_cost_l2693_269353


namespace NUMINAMATH_CALUDE_shooting_stars_count_difference_l2693_269302

/-- The number of shooting stars counted by Bridget -/
def bridget_count : ℕ := 14

/-- The number of shooting stars counted by Reginald -/
def reginald_count : ℕ := 12

/-- The number of shooting stars counted by Sam -/
def sam_count : ℕ := reginald_count + 4

/-- The average number of shooting stars counted by all three -/
def average_count : ℚ := (bridget_count + reginald_count + sam_count) / 3

theorem shooting_stars_count_difference :
  sam_count = average_count + 2 →
  bridget_count - reginald_count = 2 := by
  sorry

#eval bridget_count - reginald_count

end NUMINAMATH_CALUDE_shooting_stars_count_difference_l2693_269302


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l2693_269321

/-- Given that i is the imaginary unit and zi = 2i - z, prove that z is in the first quadrant -/
theorem z_in_first_quadrant (i : ℂ) (z : ℂ) 
  (h_i : i * i = -1) 
  (h_z : z * i = 2 * i - z) : 
  Real.sqrt 2 / 2 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l2693_269321


namespace NUMINAMATH_CALUDE_inequality_proof_l2693_269379

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2693_269379


namespace NUMINAMATH_CALUDE_rational_number_ordering_l2693_269342

theorem rational_number_ordering (a b c : ℚ) 
  (h1 : a - b > 0) (h2 : b - c > 0) : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_rational_number_ordering_l2693_269342


namespace NUMINAMATH_CALUDE_number_ratio_l2693_269370

theorem number_ratio (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y = 8 * (x - y)) : x / y = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l2693_269370


namespace NUMINAMATH_CALUDE_initial_typists_count_initial_typists_count_proof_l2693_269356

/-- Given that some typists can type 38 letters in 20 minutes and 30 typists working at the same rate can complete 171 letters in 1 hour, prove that the number of typists in the initial group is 20. -/
theorem initial_typists_count : ℕ :=
  let initial_letters : ℕ := 38
  let initial_time : ℕ := 20
  let second_typists : ℕ := 30
  let second_letters : ℕ := 171
  let second_time : ℕ := 60
  20

/-- Proof of the theorem -/
theorem initial_typists_count_proof : initial_typists_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_initial_typists_count_initial_typists_count_proof_l2693_269356


namespace NUMINAMATH_CALUDE_anna_overall_score_l2693_269398

/-- Represents a test with a number of problems and a score percentage -/
structure Test where
  problems : ℕ
  score : ℚ
  h_score_range : 0 ≤ score ∧ score ≤ 1

/-- Calculates the number of problems answered correctly in a test -/
def correctProblems (t : Test) : ℚ :=
  t.problems * t.score

/-- Theorem stating that Anna's overall score across three tests is 78% -/
theorem anna_overall_score (test1 test2 test3 : Test)
  (h1 : test1.problems = 30 ∧ test1.score = 3/4)
  (h2 : test2.problems = 50 ∧ test2.score = 17/20)
  (h3 : test3.problems = 20 ∧ test3.score = 13/20) :
  (correctProblems test1 + correctProblems test2 + correctProblems test3) /
  (test1.problems + test2.problems + test3.problems) = 39/50 := by
  sorry

end NUMINAMATH_CALUDE_anna_overall_score_l2693_269398


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l2693_269332

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (triangle_inequality : a < b + c ∧ b < c + a ∧ c < a + b)

-- Define the angle bisector points
structure AngleBisectorPoints (t : Triangle) :=
  (A₁ B₁ C₁ : ℝ × ℝ)

-- Define the property of points being concyclic
def are_concyclic (p₁ p₂ p₃ p₄ : ℝ × ℝ) : Prop := sorry

-- Define the theorem
theorem triangle_inequality_theorem (t : Triangle) (abp : AngleBisectorPoints t) :
  are_concyclic abp.A₁ abp.B₁ abp.C₁ (0, t.b) →
  (t.a / (t.b + t.c)) + (t.b / (t.c + t.a)) + (t.c / (t.a + t.b)) ≥ (Real.sqrt 17 - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l2693_269332


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2693_269355

theorem complex_magnitude_problem (z : ℂ) (h : (z - Complex.I) * (1 + Complex.I) = 2 - Complex.I) :
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2693_269355


namespace NUMINAMATH_CALUDE_min_value_expression_l2693_269377

theorem min_value_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x + y + z = 1) (h5 : x = 2 * y) :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1 → a = 2 * b →
    (x + 2 * y) / (x * y * z) ≤ (a + 2 * b) / (a * b * c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2693_269377


namespace NUMINAMATH_CALUDE_games_missed_l2693_269318

theorem games_missed (total_games attended_games : ℕ) 
  (h1 : total_games = 89) 
  (h2 : attended_games = 47) : 
  total_games - attended_games = 42 := by
  sorry

end NUMINAMATH_CALUDE_games_missed_l2693_269318


namespace NUMINAMATH_CALUDE_factorization_proof_l2693_269350

theorem factorization_proof (x y a b : ℝ) : 
  (x * y^2 - 2 * x * y = x * y * (y - 2)) ∧ 
  (6 * a * (x + y) - 5 * b * (x + y) = (x + y) * (6 * a - 5 * b)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_proof_l2693_269350


namespace NUMINAMATH_CALUDE_camp_gender_difference_l2693_269384

theorem camp_gender_difference (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 133 →
  girls = 50 →
  boys > girls →
  total = boys + girls →
  boys - girls = 33 := by
sorry

end NUMINAMATH_CALUDE_camp_gender_difference_l2693_269384


namespace NUMINAMATH_CALUDE_unplaced_unbroken_bottles_l2693_269363

-- Define the parameters
def total_bottles : ℕ := 250
def total_crates : ℕ := 15
def small_crate_capacity : ℕ := 8
def medium_crate_capacity : ℕ := 12
def large_crate_capacity : ℕ := 20
def available_small_crates : ℕ := 5
def available_medium_crates : ℕ := 5
def available_large_crates : ℕ := 5
def max_usable_small_crates : ℕ := 3
def max_usable_medium_crates : ℕ := 4
def max_usable_large_crates : ℕ := 5
def broken_bottles : ℕ := 11

-- Theorem statement
theorem unplaced_unbroken_bottles : 
  total_bottles - broken_bottles - 
  (max_usable_small_crates * small_crate_capacity + 
   max_usable_medium_crates * medium_crate_capacity + 
   max_usable_large_crates * large_crate_capacity) = 67 := by
  sorry

end NUMINAMATH_CALUDE_unplaced_unbroken_bottles_l2693_269363


namespace NUMINAMATH_CALUDE_correct_calculation_l2693_269375

/-- Represents a cricketer's bowling statistics --/
structure BowlingStats where
  initialAverage : ℝ
  initialStrikeRate : ℝ
  initialWickets : ℕ
  currentMatchWickets : ℕ
  currentMatchRuns : ℕ
  newAverage : ℝ
  newStrikeRate : ℝ

/-- Calculates the total wickets and balls bowled before the current match --/
def calculateStats (stats : BowlingStats) : ℕ × ℕ :=
  sorry

/-- Theorem stating the correct calculation of wickets and balls bowled --/
theorem correct_calculation (stats : BowlingStats) 
  (h1 : stats.initialAverage = 12.4)
  (h2 : stats.initialStrikeRate = 30)
  (h3 : stats.initialWickets ≥ 50)
  (h4 : stats.currentMatchWickets = 5)
  (h5 : stats.currentMatchRuns = 26)
  (h6 : stats.newAverage = stats.initialAverage - 0.4)
  (h7 : stats.newStrikeRate = 28) :
  calculateStats stats = (85, 2550) :=
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2693_269375
