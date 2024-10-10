import Mathlib

namespace parallelepiped_properties_l3602_360296

/-- Represents an oblique parallelepiped with given properties -/
structure ObliqueParallelepiped where
  lateral_edge_projection : ℝ
  height : ℝ
  rhombus_area : ℝ
  rhombus_diagonal : ℝ

/-- Calculates the lateral surface area of the parallelepiped -/
def lateral_surface_area (p : ObliqueParallelepiped) : ℝ := sorry

/-- Calculates the volume of the parallelepiped -/
def volume (p : ObliqueParallelepiped) : ℝ := sorry

/-- Theorem stating the lateral surface area and volume of the given parallelepiped -/
theorem parallelepiped_properties :
  let p : ObliqueParallelepiped := {
    lateral_edge_projection := 5,
    height := 12,
    rhombus_area := 24,
    rhombus_diagonal := 8
  }
  lateral_surface_area p = 260 ∧ volume p = 312 := by sorry

end parallelepiped_properties_l3602_360296


namespace cafeteria_green_apples_l3602_360210

/-- Prove that the number of green apples ordered by the cafeteria is 23 -/
theorem cafeteria_green_apples :
  let red_apples : ℕ := 33
  let students_wanting_fruit : ℕ := 21
  let extra_apples : ℕ := 35
  let green_apples : ℕ := 23
  (red_apples + green_apples - students_wanting_fruit = extra_apples) →
  green_apples = 23 := by
sorry

end cafeteria_green_apples_l3602_360210


namespace officer_selection_theorem_l3602_360288

/-- Represents the Chemistry Club and its officer selection process -/
structure ChemistryClub where
  totalMembers : Nat
  aliceAndBobCondition : Bool
  ronaldCondition : Bool

/-- Calculates the number of ways to select officers -/
def selectOfficers (club : ChemistryClub) : Nat :=
  let withoutAliceBob := (club.totalMembers - 3) * (club.totalMembers - 4) * (club.totalMembers - 5)
  let withAliceBob := 6
  withoutAliceBob + withAliceBob

/-- The main theorem stating the number of ways to select officers -/
theorem officer_selection_theorem (club : ChemistryClub) 
  (h1 : club.totalMembers = 25)
  (h2 : club.aliceAndBobCondition = true)
  (h3 : club.ronaldCondition = true) :
  selectOfficers club = 9246 := by
  sorry

#eval selectOfficers { totalMembers := 25, aliceAndBobCondition := true, ronaldCondition := true }

end officer_selection_theorem_l3602_360288


namespace exactly_one_correct_l3602_360212

/-- The probability that exactly one of three independent events occurs, given their individual probabilities -/
theorem exactly_one_correct (pA pB pC : ℝ) 
  (hA : 0 ≤ pA ∧ pA ≤ 1) 
  (hB : 0 ≤ pB ∧ pB ≤ 1) 
  (hC : 0 ≤ pC ∧ pC ≤ 1) 
  (hpA : pA = 3/4) 
  (hpB : pB = 2/3) 
  (hpC : pC = 2/3) : 
  pA * (1 - pB) * (1 - pC) + (1 - pA) * pB * (1 - pC) + (1 - pA) * (1 - pB) * pC = 7/36 := by
  sorry

#check exactly_one_correct

end exactly_one_correct_l3602_360212


namespace function_divisibility_l3602_360297

def is_divisible (a b : ℤ) : Prop := ∃ k : ℤ, b = k * a

theorem function_divisibility 
  (f : ℤ → ℕ+) 
  (h : ∀ (m n : ℤ), is_divisible (f (m - n)) (f m - f n)) :
  ∀ (m n : ℤ), f m ≤ f n → is_divisible (f m) (f n) :=
sorry

end function_divisibility_l3602_360297


namespace power_of_256_l3602_360263

theorem power_of_256 : (256 : ℝ) ^ (4/5 : ℝ) = 64 := by
  have h1 : 256 = 2^8 := by sorry
  sorry

end power_of_256_l3602_360263


namespace power_eight_mod_eleven_l3602_360231

theorem power_eight_mod_eleven : 8^2030 % 11 = 1 := by
  sorry

end power_eight_mod_eleven_l3602_360231


namespace special_square_divisions_l3602_360276

/-- Represents a 5x5 square with a 3x3 center and 1x3 rectangles on each side -/
structure SpecialSquare :=
  (size : Nat)
  (center_size : Nat)
  (side_rectangle_size : Nat)
  (h_size : size = 5)
  (h_center : center_size = 3)
  (h_side : side_rectangle_size = 3)

/-- Counts the number of ways to divide the SpecialSquare into 1x3 rectangles -/
def count_divisions (square : SpecialSquare) : Nat :=
  2

/-- Theorem stating that the number of ways to divide the SpecialSquare into 1x3 rectangles is 2 -/
theorem special_square_divisions (square : SpecialSquare) :
  count_divisions square = 2 := by
  sorry

end special_square_divisions_l3602_360276


namespace A_not_necessary_for_B_A_not_sufficient_for_B_A_neither_necessary_nor_sufficient_for_B_l3602_360234

-- Define condition A
def condition_A (x y : ℝ) : Prop := x ≠ 1 ∧ y ≠ 2

-- Define condition B
def condition_B (x y : ℝ) : Prop := x + y ≠ 3

-- Theorem stating that A is not necessary for B
theorem A_not_necessary_for_B : ¬∀ x y : ℝ, condition_B x y → condition_A x y := by
  sorry

-- Theorem stating that A is not sufficient for B
theorem A_not_sufficient_for_B : ¬∀ x y : ℝ, condition_A x y → condition_B x y := by
  sorry

-- Main theorem combining the above results
theorem A_neither_necessary_nor_sufficient_for_B :
  (¬∀ x y : ℝ, condition_B x y → condition_A x y) ∧
  (¬∀ x y : ℝ, condition_A x y → condition_B x y) := by
  sorry

end A_not_necessary_for_B_A_not_sufficient_for_B_A_neither_necessary_nor_sufficient_for_B_l3602_360234


namespace king_ace_probability_l3602_360248

/-- A standard deck of cards. -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (card_count : cards.card = 52)
  (suit_count : ∀ s, (cards.filter (λ c => c.1 = s)).card = 13)
  (rank_count : ∀ r, (cards.filter (λ c => c.2 = r)).card = 4)

/-- The probability of drawing a King first and an Ace second from a standard deck. -/
def king_ace_prob (d : Deck) : ℚ :=
  4 / 663

/-- Theorem stating that the probability of drawing a King first and an Ace second
    from a standard deck is 4/663. -/
theorem king_ace_probability (d : Deck) :
  king_ace_prob d = 4 / 663 := by
  sorry

end king_ace_probability_l3602_360248


namespace tangent_line_at_A_l3602_360298

/-- The function f(x) = -x^3 + 3x --/
def f (x : ℝ) := -x^3 + 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) := -3*x^2 + 3

/-- Point A --/
def A : ℝ × ℝ := (2, -2)

/-- Equation of a line passing through A with slope m --/
def line_eq (m : ℝ) (x : ℝ) : ℝ := m*(x - A.1) + A.2

/-- Theorem: The tangent line to f(x) at A is either y = -2 or 9x + y - 16 = 0 --/
theorem tangent_line_at_A : 
  (∃ x y, line_eq (f' A.1) x = y ∧ 9*x + y - 16 = 0) ∨
  (∀ x, line_eq (f' A.1) x = -2) :=
sorry

end tangent_line_at_A_l3602_360298


namespace quadratic_equation_solution_l3602_360229

theorem quadratic_equation_solution :
  let a : ℝ := 2
  let b : ℝ := -6
  let c : ℝ := 1
  let x₁ : ℝ := (3 + Real.sqrt 7) / 2
  let x₂ : ℝ := (3 - Real.sqrt 7) / 2
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end quadratic_equation_solution_l3602_360229


namespace average_of_five_integers_l3602_360245

theorem average_of_five_integers (k m r s t : ℕ) : 
  k < m → m < r → r < s → s < t → 
  t = 42 → 
  r ≤ 17 → 
  (k + m + r + s + t : ℚ) / 5 = 266 / 10 := by
sorry

end average_of_five_integers_l3602_360245


namespace cosine_inequality_l3602_360240

theorem cosine_inequality (x y : Real) : 
  x ∈ Set.Icc 0 (Real.pi / 2) →
  y ∈ Set.Icc 0 (Real.pi / 2) →
  Real.cos (x - y) ≥ Real.cos x - Real.cos y := by
sorry

end cosine_inequality_l3602_360240


namespace set_equality_l3602_360250

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end set_equality_l3602_360250


namespace geometric_sequence_fifth_term_l3602_360220

theorem geometric_sequence_fifth_term 
  (a : ℕ) (r : ℕ) (h1 : a = 4) (h2 : a * r^3 = 324) : a * r^4 = 324 :=
by sorry

end geometric_sequence_fifth_term_l3602_360220


namespace fill_water_tank_days_l3602_360246

/-- Represents the number of days needed to fill a water tank -/
def days_to_fill_tank (tank_capacity : ℕ) (daily_collection : ℕ) : ℕ :=
  (tank_capacity * 1000 + daily_collection - 1) / daily_collection

/-- Theorem stating that it takes 206 days to fill the water tank -/
theorem fill_water_tank_days : days_to_fill_tank 350 1700 = 206 := by
  sorry

end fill_water_tank_days_l3602_360246


namespace discount_saves_money_savings_amount_l3602_360290

/-- Represents the ticket pricing strategy for a park -/
structure TicketStrategy where
  regular_price : ℕ  -- Regular price per ticket
  discount_rate : ℚ  -- Discount rate for group tickets
  discount_threshold : ℕ  -- Minimum number of people for group discount

/-- Calculates the total cost for a given number of tickets -/
def total_cost (strategy : TicketStrategy) (num_tickets : ℕ) : ℚ :=
  if num_tickets ≥ strategy.discount_threshold
  then (strategy.regular_price * num_tickets * (1 - strategy.discount_rate))
  else (strategy.regular_price * num_tickets)

/-- Theorem: Purchasing 25 tickets with discount is cheaper than 23 without discount -/
theorem discount_saves_money (strategy : TicketStrategy) 
  (h1 : strategy.regular_price = 10)
  (h2 : strategy.discount_rate = 1/5)
  (h3 : strategy.discount_threshold = 25) :
  total_cost strategy 25 < total_cost strategy 23 ∧ 
  total_cost strategy 23 - total_cost strategy 25 = 30 :=
by sorry

/-- Corollary: The savings amount to exactly 30 yuan -/
theorem savings_amount (strategy : TicketStrategy)
  (h1 : strategy.regular_price = 10)
  (h2 : strategy.discount_rate = 1/5)
  (h3 : strategy.discount_threshold = 25) :
  total_cost strategy 23 - total_cost strategy 25 = 30 :=
by sorry

end discount_saves_money_savings_amount_l3602_360290


namespace calculation_proof_l3602_360265

theorem calculation_proof :
  2 / (-1/4) - |(-Real.sqrt 18)| + (1/5)⁻¹ = -3 - 3 * Real.sqrt 2 := by
  sorry

end calculation_proof_l3602_360265


namespace f_property_l3602_360227

def f (x : ℝ) : ℝ := x * |x|

theorem f_property : ∀ x : ℝ, f (Real.sqrt 2 * x) = 2 * f x := by
  sorry

end f_property_l3602_360227


namespace intersection_implies_a_values_l3602_360254

def A (a : ℝ) : Set ℝ := {a^2, a+1, 3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_values :
  ∀ a : ℝ, A a ∩ B a = {3} → a = 6 ∨ a = Real.sqrt 2 ∨ a = -Real.sqrt 2 := by
  sorry

end intersection_implies_a_values_l3602_360254


namespace robert_reading_capacity_l3602_360283

/-- The number of complete books Robert can read in a given time -/
def books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (available_hours : ℕ) : ℕ :=
  (pages_per_hour * available_hours) / pages_per_book

/-- Theorem: Robert can read 2 complete 360-page books in 8 hours at a rate of 120 pages per hour -/
theorem robert_reading_capacity :
  books_read 120 360 8 = 2 := by
  sorry

end robert_reading_capacity_l3602_360283


namespace no_numbers_satisfying_conditions_l3602_360286

def is_in_range (n : ℕ) : Prop := 7 ≤ n ∧ n ≤ 49

def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def remainder_3_mod_5 (n : ℕ) : Prop := n % 5 = 3

-- We don't need to define primality as it's already in Mathlib

theorem no_numbers_satisfying_conditions :
  ¬∃ n : ℕ, is_in_range n ∧ divisible_by_6 n ∧ remainder_3_mod_5 n ∧ Nat.Prime n :=
sorry

end no_numbers_satisfying_conditions_l3602_360286


namespace alex_has_sixty_shells_l3602_360275

/-- The number of seashells in a dozen -/
def dozen : ℕ := 12

/-- The number of seashells Mimi picked up -/
def mimi_shells : ℕ := 2 * dozen

/-- The number of seashells Kyle found -/
def kyle_shells : ℕ := 2 * mimi_shells

/-- The number of seashells Leigh grabbed -/
def leigh_shells : ℕ := kyle_shells / 3

/-- The number of seashells Alex unearthed -/
def alex_shells : ℕ := 3 * leigh_shells + mimi_shells / 2

/-- Theorem stating that Alex had 60 seashells -/
theorem alex_has_sixty_shells : alex_shells = 60 := by
  sorry

end alex_has_sixty_shells_l3602_360275


namespace solve_exponential_equation_l3602_360228

theorem solve_exponential_equation :
  ∃ x : ℝ, (2 : ℝ) ^ (x - 3) = 4 ^ (x + 1) ∧ x = -5 := by
  sorry

end solve_exponential_equation_l3602_360228


namespace B_power_difference_l3602_360244

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_difference : 
  B^30 - B^29 = !![2, 4; 0, 1] := by sorry

end B_power_difference_l3602_360244


namespace hyperbola_equilateral_triangle_l3602_360269

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define an equilateral triangle
def is_equilateral_triangle (P Q R : ℝ × ℝ) : Prop :=
  let d₁ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2
  let d₂ := (Q.1 - R.1)^2 + (Q.2 - R.2)^2
  let d₃ := (R.1 - P.1)^2 + (R.2 - P.2)^2
  d₁ = d₂ ∧ d₂ = d₃

-- Define being on the same branch of the hyperbola
def same_branch (P Q R : ℝ × ℝ) : Prop :=
  (P.1 > 0 ∧ Q.1 > 0 ∧ R.1 > 0) ∨ (P.1 < 0 ∧ Q.1 < 0 ∧ R.1 < 0)

-- Main theorem
theorem hyperbola_equilateral_triangle :
  ∀ P Q R : ℝ × ℝ,
  hyperbola P.1 P.2 →
  hyperbola Q.1 Q.2 →
  hyperbola R.1 R.2 →
  is_equilateral_triangle P Q R →
  (¬ same_branch P Q R) ∧
  (P = (-1, -1) →
   Q.1 > 0 →
   R.1 > 0 →
   Q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧
   R = (2 + Real.sqrt 3, 2 - Real.sqrt 3)) :=
by sorry

end hyperbola_equilateral_triangle_l3602_360269


namespace parabola_translation_up_2_l3602_360237

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Translates a parabola vertically -/
def translate_vertical (p : Parabola) (k : ℝ) : Parabola where
  f := λ x => p.f x + k

/-- The standard parabola y = x^2 -/
def standard_parabola : Parabola where
  f := λ x => x^2

theorem parabola_translation_up_2 :
  (translate_vertical standard_parabola 2).f = λ x => x^2 + 2 := by
  sorry

end parabola_translation_up_2_l3602_360237


namespace alice_twice_bob_age_l3602_360268

theorem alice_twice_bob_age (alice_age bob_age : ℕ) : 
  alice_age = bob_age + 10 →
  alice_age + 5 = 19 →
  ∃ (years : ℕ), (alice_age + years = 2 * (bob_age + years)) ∧ years = 6 :=
by sorry

end alice_twice_bob_age_l3602_360268


namespace cubic_roots_arithmetic_progression_b_value_l3602_360292

/-- A cubic polynomial with coefficient b -/
def cubic (x b : ℂ) : ℂ := x^3 - 9*x^2 + 33*x + b

/-- Predicate to check if three complex numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℂ) : Prop := b - a = c - b

/-- Theorem stating that if the roots of the cubic form an arithmetic progression
    and at least one root is non-real, then b = -15 -/
theorem cubic_roots_arithmetic_progression_b_value (b : ℝ) :
  (∃ (r₁ r₂ r₃ : ℂ), 
    (∀ x, cubic x b = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ 
    isArithmeticProgression r₁ r₂ r₃ ∧
    (r₁.im ≠ 0 ∨ r₂.im ≠ 0 ∨ r₃.im ≠ 0)) →
  b = -15 := by sorry

end cubic_roots_arithmetic_progression_b_value_l3602_360292


namespace point_in_fourth_quadrant_l3602_360289

/-- A point in the Cartesian plane lies in the fourth quadrant if and only if
    its x-coordinate is positive and its y-coordinate is negative. -/
def is_in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The point (8, -3) lies in the fourth quadrant of the Cartesian coordinate system. -/
theorem point_in_fourth_quadrant :
  is_in_fourth_quadrant 8 (-3) := by
  sorry

end point_in_fourth_quadrant_l3602_360289


namespace quadratic_roots_range_l3602_360249

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   (a - 1) * x^2 - 2 * x + 1 = 0 ∧ 
   (a - 1) * y^2 - 2 * y + 1 = 0) → 
  (a < 2 ∧ a ≠ 1) :=
by sorry

end quadratic_roots_range_l3602_360249


namespace girls_in_class_l3602_360235

theorem girls_in_class (num_boys : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) :
  num_boys = 16 →
  ratio_boys = 4 →
  ratio_girls = 5 →
  (num_boys * ratio_girls) / ratio_boys = 20 :=
by
  sorry

end girls_in_class_l3602_360235


namespace impossible_arrangement_l3602_360215

/-- Represents the type of student: Knight (always tells the truth) or Liar (always lies) -/
inductive StudentType
| Knight
| Liar

/-- Represents a desk with two students -/
structure Desk where
  student1 : StudentType
  student2 : StudentType

/-- The initial arrangement of students -/
def initial_arrangement (desks : List Desk) : Prop :=
  desks.length = 13 ∧ 
  ∀ d ∈ desks, (d.student1 = StudentType.Knight ∧ d.student2 = StudentType.Liar) ∨
                (d.student1 = StudentType.Liar ∧ d.student2 = StudentType.Knight)

/-- The final arrangement of students -/
def final_arrangement (desks : List Desk) : Prop :=
  desks.length = 13 ∧
  ∀ d ∈ desks, d.student1 = d.student2

/-- Theorem stating the impossibility of the final arrangement -/
theorem impossible_arrangement :
  ∀ (initial_desks final_desks : List Desk),
    initial_arrangement initial_desks →
    ¬(final_arrangement final_desks) :=
by sorry


end impossible_arrangement_l3602_360215


namespace remaining_typing_orders_l3602_360280

/-- The number of letters in total -/
def totalLetters : ℕ := 10

/-- The label of the letter that has been typed by midday -/
def typedLetter : ℕ := 9

/-- The number of different orders for typing the remaining letters -/
def typingOrders : ℕ := 1280

/-- 
Theorem: Given 10 letters labeled from 1 to 10, where letter 9 has been typed by midday,
the number of different orders for typing the remaining letters is 1280.
-/
theorem remaining_typing_orders :
  (totalLetters = 10) →
  (typedLetter = 9) →
  (typingOrders = 1280) :=
by sorry

end remaining_typing_orders_l3602_360280


namespace rivertown_marching_band_max_members_l3602_360203

theorem rivertown_marching_band_max_members :
  ∀ n : ℕ, 
    (20 * n ≡ 11 [MOD 31]) → 
    (20 * n < 1200) → 
    (∀ m : ℕ, (20 * m ≡ 11 [MOD 31]) → (20 * m < 1200) → (20 * m ≤ 20 * n)) →
    20 * n = 1100 :=
by sorry

end rivertown_marching_band_max_members_l3602_360203


namespace joyce_apples_l3602_360272

/-- Proves that if Joyce starts with 75 apples and gives 52 to Larry, she ends up with 23 apples -/
theorem joyce_apples : ∀ (initial_apples given_apples remaining_apples : ℕ),
  initial_apples = 75 →
  given_apples = 52 →
  remaining_apples = initial_apples - given_apples →
  remaining_apples = 23 := by
  sorry


end joyce_apples_l3602_360272


namespace arithmetic_sequence_sum_l3602_360252

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 - a 5 + a 9 - a 13 + a 17 = 117 →
  a 3 + a 15 = 234 := by
sorry

end arithmetic_sequence_sum_l3602_360252


namespace abs_neg_one_third_eq_one_third_l3602_360226

theorem abs_neg_one_third_eq_one_third : |(-1/3 : ℚ)| = 1/3 := by sorry

end abs_neg_one_third_eq_one_third_l3602_360226


namespace intersection_point_l3602_360255

/-- The quadratic function f(x) = x^2 - 4x + 4 -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 4

/-- Theorem: The point (2,0) is the only intersection point of y = x^2 - 4x + 4 with the x-axis -/
theorem intersection_point : 
  (∃! x : ℝ, f x = 0) ∧ (f 2 = 0) := by sorry

end intersection_point_l3602_360255


namespace no_valid_schedule_for_100_l3602_360264

/-- Represents a duty schedule for militia members -/
structure DutySchedule (n : ℕ) where
  nights : Set (Fin n × Fin n × Fin n)
  all_pairs_once : ∀ i j, i < j → ∃! k, (i, j, k) ∈ nights ∨ (i, k, j) ∈ nights ∨ (j, i, k) ∈ nights ∨ (j, k, i) ∈ nights ∨ (k, i, j) ∈ nights ∨ (k, j, i) ∈ nights

/-- Theorem stating the impossibility of creating a valid duty schedule for 100 militia members -/
theorem no_valid_schedule_for_100 : ¬∃ (schedule : DutySchedule 100), True := by
  sorry

end no_valid_schedule_for_100_l3602_360264


namespace soccer_team_size_l3602_360260

theorem soccer_team_size (total_goals : ℕ) (games_played : ℕ) (goals_other_players : ℕ) :
  total_goals = 150 →
  games_played = 15 →
  goals_other_players = 30 →
  ∃ (team_size : ℕ),
    team_size > 0 ∧
    (team_size / 3 : ℚ) * games_played + goals_other_players = total_goals ∧
    team_size = 24 :=
by
  sorry

end soccer_team_size_l3602_360260


namespace infiniteSeriesSum_l3602_360200

/-- The sum of the infinite series Σ(k/3^k) for k from 1 to ∞ -/
noncomputable def infiniteSeries : ℝ := ∑' k, k / (3 ^ k)

/-- Theorem: The sum of the infinite series Σ(k/3^k) for k from 1 to ∞ is equal to 3/4 -/
theorem infiniteSeriesSum : infiniteSeries = 3/4 := by sorry

end infiniteSeriesSum_l3602_360200


namespace berts_profit_l3602_360273

/-- Calculates the profit for a single item --/
def itemProfit (salesPrice : ℚ) (taxRate : ℚ) : ℚ :=
  salesPrice - (salesPrice * taxRate) - (salesPrice - 10)

/-- Calculates the total profit from the sale --/
def totalProfit (barrelPrice : ℚ) (toolsPrice : ℚ) (fertilizerPrice : ℚ) 
  (barrelTaxRate : ℚ) (toolsTaxRate : ℚ) (fertilizerTaxRate : ℚ) : ℚ :=
  itemProfit barrelPrice barrelTaxRate + 
  itemProfit toolsPrice toolsTaxRate + 
  itemProfit fertilizerPrice fertilizerTaxRate

/-- Theorem stating that Bert's total profit is $14.90 --/
theorem berts_profit : 
  totalProfit 90 50 30 (10/100) (5/100) (12/100) = 149/10 :=
by sorry

end berts_profit_l3602_360273


namespace geometric_sequence_properties_l3602_360217

theorem geometric_sequence_properties (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_geom : ∃ q : ℝ, q > 0 ∧ b = a * q ∧ c = a * q^2) :
  ∃ r : ℝ, r > 0 ∧
    (a + b + c) = (Real.sqrt (3 * (a * b + b * c + c * a))) * r ∧
    (Real.sqrt (3 * (a * b + b * c + c * a))) = (27 * a * b * c)^(1/3) * r :=
by sorry

end geometric_sequence_properties_l3602_360217


namespace balloon_difference_is_one_l3602_360221

/-- The number of balloons Jake has more than Allan -/
def balloon_difference (allan_balloons jake_initial_balloons jake_bought_balloons : ℕ) : ℕ :=
  (jake_initial_balloons + jake_bought_balloons) - allan_balloons

/-- Theorem stating the difference in balloons between Jake and Allan -/
theorem balloon_difference_is_one :
  balloon_difference 6 3 4 = 1 := by
  sorry

end balloon_difference_is_one_l3602_360221


namespace square_circle_union_area_l3602_360291

theorem square_circle_union_area (s : Real) (r : Real) :
  s = 12 ∧ r = 12 →
  (s^2) + (π * r^2) - (π * r^2 / 4) = 144 + 108 * π := by
  sorry

end square_circle_union_area_l3602_360291


namespace notebook_savings_proof_l3602_360238

/-- Calculates the savings on a notebook purchase with discounts -/
def calculateSavings (originalPrice : ℝ) (quantity : ℕ) (saleDiscount : ℝ) (volumeDiscount : ℝ) : ℝ :=
  let discountedPrice := originalPrice * (1 - saleDiscount)
  let finalPrice := discountedPrice * (1 - volumeDiscount)
  quantity * (originalPrice - finalPrice)

/-- Proves that the savings on the notebook purchase is $7.84 -/
theorem notebook_savings_proof :
  calculateSavings 3 8 0.25 0.1 = 7.84 := by
  sorry

#eval calculateSavings 3 8 0.25 0.1

end notebook_savings_proof_l3602_360238


namespace crayon_box_problem_l3602_360299

theorem crayon_box_problem (C R B G Y P U : ℝ) : 
  R + B + G + Y + P + U = C →
  R = 12 →
  B = 8 →
  G = (3/4) * B →
  Y = 0.15 * C →
  P = U →
  P = 0.425 * C - 13 := by
sorry

end crayon_box_problem_l3602_360299


namespace saturday_visitors_200_l3602_360241

/-- Calculates the number of visitors on Saturday given the ticket price, 
    weekday visitors, Sunday visitors, and total revenue -/
def visitors_on_saturday (ticket_price : ℕ) (weekday_visitors : ℕ) 
  (sunday_visitors : ℕ) (total_revenue : ℕ) : ℕ :=
  (total_revenue / ticket_price) - (5 * weekday_visitors) - sunday_visitors

/-- Proves that the number of visitors on Saturday is 200 given the specified conditions -/
theorem saturday_visitors_200 : 
  visitors_on_saturday 3 100 300 3000 = 200 := by
  sorry

end saturday_visitors_200_l3602_360241


namespace jia_incorrect_questions_l3602_360274

-- Define the type for questions
inductive Question
| Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7

-- Define a person's answers
def Answers := Question → Bool

-- Define the correct answers
def correct_answers : Answers := sorry

-- Define Jia's answers
def jia_answers : Answers := sorry

-- Define Yi's answers
def yi_answers : Answers := sorry

-- Define Bing's answers
def bing_answers : Answers := sorry

-- Function to count correct answers
def count_correct (answers : Answers) : Nat := sorry

-- Theorem stating the problem conditions and the conclusion to be proved
theorem jia_incorrect_questions :
  (count_correct jia_answers = 5) →
  (count_correct yi_answers = 5) →
  (count_correct bing_answers = 5) →
  (jia_answers Question.Q1 ≠ correct_answers Question.Q1) ∧
  (jia_answers Question.Q3 ≠ correct_answers Question.Q3) :=
by sorry

end jia_incorrect_questions_l3602_360274


namespace proposition_relationship_l3602_360284

theorem proposition_relationship (p q : Prop) : 
  (¬p ∨ ¬q → ¬p ∧ ¬q) ∧ 
  ∃ (p q : Prop), (¬p ∧ ¬q) ∧ ¬(¬p ∨ ¬q) :=
by sorry

end proposition_relationship_l3602_360284


namespace lucy_had_twenty_l3602_360213

/-- The amount of money Lucy originally had -/
def lucy_original : ℕ := sorry

/-- The amount of money Linda originally had -/
def linda_original : ℕ := 10

/-- Proposition that if Lucy gives Linda $5, they would have the same amount of money -/
def equal_after_transfer : Prop :=
  lucy_original - 5 = linda_original + 5

theorem lucy_had_twenty :
  lucy_original = 20 :=
by sorry

end lucy_had_twenty_l3602_360213


namespace parallel_lines_distance_l3602_360247

/-- Two parallel lines with a specified distance between them -/
structure ParallelLines where
  -- First line equation: 3x - y + 3 = 0
  l₁ : ℝ → ℝ → Prop
  l₁_def : l₁ = fun x y ↦ 3 * x - y + 3 = 0
  -- Second line equation: 3x - y + C = 0
  l₂ : ℝ → ℝ → Prop
  C : ℝ
  l₂_def : l₂ = fun x y ↦ 3 * x - y + C = 0
  -- Distance between the lines is √10
  distance : ℝ
  distance_def : distance = Real.sqrt 10

/-- The main theorem stating the possible values of C -/
theorem parallel_lines_distance (pl : ParallelLines) : pl.C = 13 ∨ pl.C = -7 := by
  sorry


end parallel_lines_distance_l3602_360247


namespace library_book_redistribution_l3602_360225

theorem library_book_redistribution (total_books : Nat) (initial_stack : Nat) (new_stack : Nat)
    (h1 : total_books = 1452)
    (h2 : initial_stack = 42)
    (h3 : new_stack = 43) :
  total_books % new_stack = 33 := by
  sorry

end library_book_redistribution_l3602_360225


namespace f_monotone_decreasing_iff_a_in_range_l3602_360261

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x^2 - 8 * a * x + 3 else Real.log x / Real.log a

def monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f y ≤ f x

theorem f_monotone_decreasing_iff_a_in_range (a : ℝ) :
  (monotone_decreasing (f a)) ↔ (1/2 ≤ a ∧ a ≤ 5/8) :=
sorry

end f_monotone_decreasing_iff_a_in_range_l3602_360261


namespace jerry_speed_is_30_l3602_360201

/-- Jerry's average speed in miles per hour -/
def jerry_speed : ℝ := 30

/-- Carla's average speed in miles per hour -/
def carla_speed : ℝ := 35

/-- Time difference between Jerry and Carla's departure in hours -/
def time_difference : ℝ := 0.5

/-- Time it takes Carla to catch up to Jerry in hours -/
def catch_up_time : ℝ := 3

/-- Theorem stating that Jerry's speed is 30 miles per hour -/
theorem jerry_speed_is_30 :
  jerry_speed = 30 ∧
  carla_speed * catch_up_time = jerry_speed * (catch_up_time + time_difference) :=
by sorry

end jerry_speed_is_30_l3602_360201


namespace average_student_height_l3602_360209

/-- Calculates the average height of all students given the average heights of males and females and the ratio of males to females. -/
theorem average_student_height
  (avg_female_height : ℝ)
  (avg_male_height : ℝ)
  (male_to_female_ratio : ℝ)
  (h1 : avg_female_height = 170)
  (h2 : avg_male_height = 185)
  (h3 : male_to_female_ratio = 2) :
  (male_to_female_ratio * avg_male_height + avg_female_height) / (male_to_female_ratio + 1) = 180 :=
by sorry

end average_student_height_l3602_360209


namespace complex_power_sum_l3602_360293

theorem complex_power_sum (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^99 + z^100 + z^101 + z^102 + z^103 = 1 + z := by
  sorry

end complex_power_sum_l3602_360293


namespace thirty_minus_twelve_base5_l3602_360211

/-- Converts a natural number to its base 5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Theorem: 30 in base 10 minus 12 in base 10 equals 33 in base 5 --/
theorem thirty_minus_twelve_base5 : toBase5 (30 - 12) = [3, 3] := by
  sorry

end thirty_minus_twelve_base5_l3602_360211


namespace angle_bisector_c_value_l3602_360251

/-- Triangle with vertices A, B, C in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Angle bisector of a triangle -/
def angleBisector (t : Triangle) (v : ℝ × ℝ) (l : LineEquation) : Prop :=
  -- This is a placeholder for the actual definition of an angle bisector
  True

theorem angle_bisector_c_value (t : Triangle) (l : LineEquation) :
  t.A = (-2, 3) →
  t.B = (-6, -8) →
  t.C = (4, -1) →
  l.a = 5 →
  l.b = 4 →
  angleBisector t t.B l →
  l.c + 5 = -155/7 := by
  sorry

end angle_bisector_c_value_l3602_360251


namespace exists_tricolor_right_triangle_l3602_360266

/-- A color type with three possible values -/
inductive Color
  | One
  | Two
  | Three

/-- A point on the integer plane -/
structure Point where
  x : Int
  y : Int

/-- A coloring of the integer plane -/
def Coloring := Point → Color

/-- Predicate for a right triangle -/
def is_right_triangle (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

/-- Main theorem -/
theorem exists_tricolor_right_triangle (c : Coloring) 
  (h1 : ∃ p : Point, c p = Color.One)
  (h2 : ∃ p : Point, c p = Color.Two)
  (h3 : ∃ p : Point, c p = Color.Three) :
  ∃ p1 p2 p3 : Point, 
    is_right_triangle p1 p2 p3 ∧ 
    c p1 ≠ c p2 ∧ c p2 ≠ c p3 ∧ c p3 ≠ c p1 :=
sorry

end exists_tricolor_right_triangle_l3602_360266


namespace system_solution_l3602_360239

theorem system_solution :
  ∃ (x y : ℚ), 4 * x - 3 * y = 2 ∧ 5 * x + y = (3 / 2) ∧ x = (13 / 38) ∧ y = (-4 / 19) := by
  sorry

end system_solution_l3602_360239


namespace square_perimeters_sum_l3602_360216

theorem square_perimeters_sum (a b : ℝ) (h1 : a ^ 2 + b ^ 2 = 145) (h2 : a ^ 2 - b ^ 2 = 25) :
  4 * Real.sqrt a ^ 2 + 4 * Real.sqrt b ^ 2 = 4 * Real.sqrt 85 + 4 * Real.sqrt 60 := by
  sorry

end square_perimeters_sum_l3602_360216


namespace collinear_points_k_value_l3602_360287

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a b : V)

-- Define vectors A, B, and D
def A (k : ℝ) := 2 • a + k • b
def B := a + b
def D := a - 2 • b

-- Define collinearity
def collinear (x y z : V) : Prop := ∃ (t : ℝ), y - x = t • (z - x)

-- Theorem statement
theorem collinear_points_k_value
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hnc : ¬ ∃ (r : ℝ), a = r • b)
  (hcol : collinear V (A V a b k) (B V a b) (D V a b)) :
  k = -1 :=
sorry

end collinear_points_k_value_l3602_360287


namespace johns_running_speed_l3602_360259

/-- John's running problem -/
theorem johns_running_speed
  (speed_with_dog : ℝ)
  (time_with_dog : ℝ)
  (time_alone : ℝ)
  (total_distance : ℝ)
  (h1 : speed_with_dog = 6)
  (h2 : time_with_dog = 0.5)
  (h3 : time_alone = 0.5)
  (h4 : total_distance = 5)
  (h5 : speed_with_dog * time_with_dog + speed_alone * time_alone = total_distance) :
  speed_alone = 4 := by
  sorry


end johns_running_speed_l3602_360259


namespace lg_2_plus_lg_5_equals_1_l3602_360223

-- Define lg as the logarithm with base 10
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem statement
theorem lg_2_plus_lg_5_equals_1 : lg 2 + lg 5 = 1 := by
  sorry

end lg_2_plus_lg_5_equals_1_l3602_360223


namespace ten_row_triangle_pieces_l3602_360256

/-- Calculates the sum of the first n natural numbers -/
def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculates the number of rods in an n-row triangle -/
def num_rods (n : ℕ) : ℕ := 3 * sum_of_naturals n

/-- Calculates the number of connectors in an n-row triangle -/
def num_connectors (n : ℕ) : ℕ := sum_of_naturals (n + 1)

/-- Calculates the total number of pieces in an n-row triangle -/
def total_pieces (n : ℕ) : ℕ := num_rods n + num_connectors n

theorem ten_row_triangle_pieces :
  total_pieces 10 = 231 := by
  sorry

end ten_row_triangle_pieces_l3602_360256


namespace car_bike_speed_ratio_l3602_360243

/-- Proves that the ratio of the average speed of a car to the average speed of a bike is 1.8 -/
theorem car_bike_speed_ratio :
  let tractor_distance : ℝ := 575
  let tractor_time : ℝ := 25
  let car_distance : ℝ := 331.2
  let car_time : ℝ := 4
  let tractor_speed : ℝ := tractor_distance / tractor_time
  let bike_speed : ℝ := 2 * tractor_speed
  let car_speed : ℝ := car_distance / car_time
  car_speed / bike_speed = 1.8 := by
sorry


end car_bike_speed_ratio_l3602_360243


namespace new_light_wattage_l3602_360262

/-- Given a light with a rating of 60 watts, a new light with 12% higher wattage will have 67.2 watts. -/
theorem new_light_wattage :
  let original_wattage : ℝ := 60
  let increase_percentage : ℝ := 12
  let new_wattage : ℝ := original_wattage * (1 + increase_percentage / 100)
  new_wattage = 67.2 := by
  sorry

end new_light_wattage_l3602_360262


namespace min_value_reciprocal_sum_l3602_360224

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : 4 * x + y = 1) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 4 * x' + y' = 1 → 1 / x' + 4 / y' ≥ 1 / x + 4 / y) →
  1 / x + 4 / y = 16 :=
by sorry

end min_value_reciprocal_sum_l3602_360224


namespace division_problem_l3602_360294

theorem division_problem (divisor quotient remainder : ℕ) : 
  divisor = 10 * quotient →
  divisor = 5 * remainder →
  remainder = 46 →
  divisor * quotient + remainder = 5336 :=
by sorry

end division_problem_l3602_360294


namespace multiply_monomials_l3602_360277

theorem multiply_monomials (x : ℝ) : 2*x * 5*x^2 = 10*x^3 := by
  sorry

end multiply_monomials_l3602_360277


namespace count_eight_in_product_l3602_360257

/-- The number of occurrences of a digit in a natural number -/
def countDigit (n : ℕ) (d : ℕ) : ℕ := sorry

/-- The product of 987654321 and 9 -/
def product : ℕ := 987654321 * 9

/-- Theorem: The number of occurrences of the digit 8 in the product of 987654321 and 9 is 9 -/
theorem count_eight_in_product : countDigit product 8 = 9 := by sorry

end count_eight_in_product_l3602_360257


namespace common_tangent_sum_l3602_360219

-- Define the parabolas
def P₁ (x y : ℚ) : Prop := y = x^2 + 51/50
def P₂ (x y : ℚ) : Prop := x = y^2 + 19/2

-- Define the tangent line
def TangentLine (a b c : ℕ) (x y : ℚ) : Prop := a * x + b * y = c

-- Define the property of being a common tangent to both parabolas
def CommonTangent (a b c : ℕ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℚ), 
    P₁ x₁ y₁ ∧ P₂ x₂ y₂ ∧ 
    TangentLine a b c x₁ y₁ ∧ 
    TangentLine a b c x₂ y₂

-- The main theorem
theorem common_tangent_sum :
  ∀ (a b c : ℕ), 
    a > 0 → b > 0 → c > 0 →
    Nat.gcd a (Nat.gcd b c) = 1 →
    CommonTangent a b c →
    (a : ℤ) + b + c = 37 := by sorry

end common_tangent_sum_l3602_360219


namespace fraction_simplification_l3602_360230

theorem fraction_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hsum : y + 1/x ≠ 0) :
  (x + 1/y) / (y + 1/x) = x / y := by
  sorry

end fraction_simplification_l3602_360230


namespace inequality_proof_l3602_360279

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^3 + b^3 + c^3) / (a + b + c) + (b^3 + c^3 + d^3) / (b + c + d) +
  (c^3 + d^3 + a^3) / (c + d + a) + (d^3 + a^3 + b^3) / (d + a + b) ≥
  a^2 + b^2 + c^2 + d^2 := by
  sorry

end inequality_proof_l3602_360279


namespace triangle_sum_equality_l3602_360202

theorem triangle_sum_equality 
  (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + x*y + y^2 = a^2)
  (h2 : y^2 + y*z + z^2 = b^2)
  (h3 : x^2 + x*z + z^2 = c^2) :
  let p := (a + b + c) / 2
  x*y + y*z + x*z = 4 * Real.sqrt ((p*(p-a)*(p-b)*(p-c))/3) := by
  sorry

end triangle_sum_equality_l3602_360202


namespace divisors_of_x_15_minus_1_l3602_360218

theorem divisors_of_x_15_minus_1 :
  ∀ k : ℕ, k ≤ 14 →
    ∃ p : Polynomial ℤ, (Polynomial.degree p = k) ∧ (p ∣ (X ^ 15 - 1)) :=
by sorry

end divisors_of_x_15_minus_1_l3602_360218


namespace trigonometric_equation_solution_l3602_360281

theorem trigonometric_equation_solution (x : ℝ) : 
  Real.cos (7 * x) + Real.sin (8 * x) = Real.cos (3 * x) - Real.sin (2 * x) → 
  (∃ n : ℤ, x = n * Real.pi / 5) ∨ 
  (∃ k : ℤ, x = Real.pi / 2 * (4 * k - 1)) ∨ 
  (∃ l : ℤ, x = Real.pi / 10 * (4 * l + 1)) := by
  sorry

end trigonometric_equation_solution_l3602_360281


namespace sector_central_angle_l3602_360206

/-- Given a sector of a circle with arc length and area both equal to 5,
    prove that its central angle is 2.5 radians. -/
theorem sector_central_angle (r : ℝ) (θ : ℝ) : 
  r > 0 → 
  r * θ = 5 →  -- arc length formula
  1/2 * r^2 * θ = 5 →  -- sector area formula
  θ = 2.5 := by
sorry

end sector_central_angle_l3602_360206


namespace infinitely_many_solutions_l3602_360253

theorem infinitely_many_solutions (d : ℝ) : 
  (∀ x : ℝ, 3 * (5 + d * x) = 15 * x + 15) ↔ d = 5 :=
sorry

end infinitely_many_solutions_l3602_360253


namespace rectangle_length_from_square_wire_l3602_360270

/-- Given a square with side length 12 and a rectangle with the same perimeter and width 6,
    prove that the length of the rectangle is 18. -/
theorem rectangle_length_from_square_wire (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) :
  square_side = 12 →
  rect_width = 6 →
  4 * square_side = 2 * (rect_width + rect_length) →
  rect_length = 18 := by
sorry

end rectangle_length_from_square_wire_l3602_360270


namespace product_equality_implies_n_equals_six_l3602_360267

theorem product_equality_implies_n_equals_six (n : ℕ) : 
  2 * 2 * 3 * 3 * 5 * 6 = 5 * 6 * n * n → n = 6 := by
  sorry

end product_equality_implies_n_equals_six_l3602_360267


namespace smallest_prime_after_six_nonprimes_l3602_360295

/-- A function that returns true if a natural number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- A function that returns true if there are at least six consecutive nonprime numbers before n, false otherwise -/
def hasSixConsecutiveNonprimes (n : ℕ) : Prop := sorry

theorem smallest_prime_after_six_nonprimes : 
  ∃ (k : ℕ), 
    isPrime k ∧ 
    hasSixConsecutiveNonprimes k ∧ 
    (∀ (m : ℕ), m < k → ¬(isPrime m ∧ hasSixConsecutiveNonprimes m)) ∧
    k = 97 := by sorry

end smallest_prime_after_six_nonprimes_l3602_360295


namespace complex_square_root_of_18i_l3602_360232

theorem complex_square_root_of_18i :
  ∀ (z : ℂ), (∃ (x y : ℝ), z = x + y * I ∧ x > 0 ∧ z^2 = 18 * I) → z = 3 + 3 * I :=
by
  sorry

end complex_square_root_of_18i_l3602_360232


namespace exactly_two_integers_l3602_360214

/-- Define the function that we want to check for integrality --/
def f (n : ℕ) : ℚ :=
  (Nat.factorial (n^3 - 1)) / ((Nat.factorial n)^(n + 2))

/-- Predicate to check if a number is in the range [1, 50] --/
def in_range (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 50

/-- Predicate to check if f(n) is an integer --/
def is_integer (n : ℕ) : Prop :=
  ∃ k : ℤ, f n = k

/-- Main theorem statement --/
theorem exactly_two_integers :
  (∃ (S : Finset ℕ), S.card = 2 ∧ 
    (∀ n, n ∈ S ↔ (in_range n ∧ is_integer n)) ∧
    (∀ n, in_range n → is_integer n → n ∈ S)) :=
sorry

end exactly_two_integers_l3602_360214


namespace store_purchase_combinations_l3602_360205

theorem store_purchase_combinations (headphones : ℕ) (mice : ℕ) (keyboards : ℕ) 
  (keyboard_mouse_sets : ℕ) (headphones_mouse_sets : ℕ) : 
  headphones = 9 → mice = 13 → keyboards = 5 → 
  keyboard_mouse_sets = 4 → headphones_mouse_sets = 5 → 
  keyboard_mouse_sets * headphones + 
  headphones_mouse_sets * keyboards + 
  headphones * mice * keyboards = 646 := by
  sorry

end store_purchase_combinations_l3602_360205


namespace max_median_is_four_point_five_l3602_360242

/-- Represents the soda shop scenario -/
structure SodaShop where
  total_cans : ℕ
  total_customers : ℕ
  min_cans_per_customer : ℕ
  h_total_cans : total_cans = 310
  h_total_customers : total_customers = 120
  h_min_cans : min_cans_per_customer = 1

/-- Calculates the maximum possible median number of cans bought per customer -/
def max_median_cans (shop : SodaShop) : ℚ :=
  sorry

/-- Theorem stating that the maximum possible median is 4.5 -/
theorem max_median_is_four_point_five (shop : SodaShop) :
  max_median_cans shop = 4.5 := by
  sorry

end max_median_is_four_point_five_l3602_360242


namespace angle_at_point_l3602_360258

theorem angle_at_point (x : ℝ) : 
  (170 : ℝ) + 3 * x = 360 → x = 190 / 3 := by
sorry

end angle_at_point_l3602_360258


namespace expected_sum_of_marbles_l3602_360236

/-- The set of marble numbers -/
def marbleNumbers : Finset ℕ := {2, 3, 4, 5, 6, 7}

/-- The sum of two different elements from the set -/
def pairSum (a b : ℕ) : ℕ := a + b

/-- The set of all possible pairs of different marbles -/
def marblePairs : Finset (ℕ × ℕ) :=
  (marbleNumbers.product marbleNumbers).filter (fun p => p.1 < p.2)

/-- The expected value of the sum of two randomly drawn marbles -/
def expectedSum : ℚ :=
  (marblePairs.sum (fun p => pairSum p.1 p.2)) / marblePairs.card

theorem expected_sum_of_marbles :
  expectedSum = 145 / 15 := by sorry

end expected_sum_of_marbles_l3602_360236


namespace ratio_of_Δy_to_Δx_l3602_360282

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the two points on the curve
def point1 : ℝ × ℝ := (1, 2)
def point2 (Δx : ℝ) : ℝ × ℝ := (1 + Δx, f (1 + Δx))

-- Define Δy
def Δy (Δx : ℝ) : ℝ := (point2 Δx).2 - point1.2

-- Theorem statement
theorem ratio_of_Δy_to_Δx (Δx : ℝ) (h : Δx ≠ 0) :
  Δy Δx / Δx = Δx + 2 :=
by sorry

end ratio_of_Δy_to_Δx_l3602_360282


namespace not_power_function_l3602_360208

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x^α

-- Define the specific function
def f (x : ℝ) : ℝ := 2 * x^(1/2)

-- Theorem statement
theorem not_power_function : ¬ isPowerFunction f := by
  sorry

end not_power_function_l3602_360208


namespace number_of_polynomials_l3602_360204

/-- A function to determine if an expression is a polynomial -/
def isPolynomial (expr : String) : Bool :=
  match expr with
  | "x^2+2" => true
  | "1/a+4" => false
  | "3ab^2/7" => true
  | "ab/c" => false
  | "-5x" => true
  | _ => false

/-- The list of expressions to check -/
def expressions : List String :=
  ["x^2+2", "1/a+4", "3ab^2/7", "ab/c", "-5x"]

/-- Theorem stating that the number of polynomials in the given list is 3 -/
theorem number_of_polynomials :
  (expressions.filter isPolynomial).length = 3 := by
  sorry

end number_of_polynomials_l3602_360204


namespace polynomial_product_l3602_360233

theorem polynomial_product (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 - x)^2 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a + a₂ + a₄) * (a₁ + a₃ + a₅) = -256 := by
sorry

end polynomial_product_l3602_360233


namespace daughters_and_granddaughters_without_children_l3602_360285

/-- Represents Bertha's family structure -/
structure BerthaFamily where
  daughters : ℕ
  granddaughters : ℕ
  total_descendants : ℕ
  daughters_with_children : ℕ

/-- The conditions of Bertha's family -/
def bertha_conditions : BerthaFamily where
  daughters := 8
  granddaughters := 32
  total_descendants := 40
  daughters_with_children := 8

/-- Theorem stating the number of daughters and granddaughters without children -/
theorem daughters_and_granddaughters_without_children 
  (family : BerthaFamily) 
  (h1 : family.daughters = bertha_conditions.daughters)
  (h2 : family.total_descendants = bertha_conditions.total_descendants)
  (h3 : family.granddaughters = family.total_descendants - family.daughters)
  (h4 : family.daughters_with_children * 4 = family.granddaughters)
  (h5 : family.daughters_with_children ≤ family.daughters) :
  family.total_descendants - family.daughters_with_children = 32 := by
  sorry

end daughters_and_granddaughters_without_children_l3602_360285


namespace shooter_probabilities_l3602_360278

-- Define the probability of hitting the target on a single shot
def p_hit : ℝ := 0.9

-- Define the number of shots
def n_shots : ℕ := 4

-- Statement 1: Probability of hitting the target on the third shot
def statement1 : Prop := p_hit = 0.9

-- Statement 2: Probability of hitting the target exactly three times
def statement2 : Prop := Nat.choose n_shots 3 * p_hit^3 * (1 - p_hit) = p_hit^3 * (1 - p_hit)

-- Statement 3: Probability of hitting the target at least once
def statement3 : Prop := 1 - (1 - p_hit)^n_shots = 1 - (1 - 0.9)^4

theorem shooter_probabilities :
  statement1 ∧ ¬statement2 ∧ statement3 :=
sorry

end shooter_probabilities_l3602_360278


namespace line_passes_through_points_l3602_360271

/-- Given a line y = (1/2)x + c passing through points (b+4, 5) and (-2, 2),
    prove that c = 3 -/
theorem line_passes_through_points (b : ℝ) :
  ∃ c : ℝ, (5 : ℝ) = (1/2 : ℝ) * (b + 4) + c ∧ (2 : ℝ) = (1/2 : ℝ) * (-2) + c ∧ c = 3 := by
  sorry

end line_passes_through_points_l3602_360271


namespace asymptote_sum_l3602_360222

/-- 
Given a rational function y = x / (x³ + Ax² + Bx + C) where A, B, C are integers,
if the graph has vertical asymptotes at x = -3, 0, and 2,
then A + B + C = -5
-/
theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 2 → 
    ∃ y : ℝ, y = x / (x^3 + A*x^2 + B*x + C)) →
  A + B + C = -5 := by
  sorry

end asymptote_sum_l3602_360222


namespace cafeteria_pies_l3602_360207

def initial_apples : ℕ := 372
def handed_out : ℕ := 135
def apples_per_pie : ℕ := 15

theorem cafeteria_pies : 
  (initial_apples - handed_out) / apples_per_pie = 15 := by
  sorry

end cafeteria_pies_l3602_360207
