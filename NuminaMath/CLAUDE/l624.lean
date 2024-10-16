import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l624_62403

theorem equation_solution : ∃! x : ℝ, (27 : ℝ) ^ (x - 2) / (9 : ℝ) ^ (x - 1) = (81 : ℝ) ^ (3 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l624_62403


namespace NUMINAMATH_CALUDE_same_color_marble_probability_l624_62411

/-- The probability of drawing three marbles of the same color from a bag containing
    red, white, and blue marbles, without replacement. -/
theorem same_color_marble_probability
  (red : ℕ) (white : ℕ) (blue : ℕ)
  (h_red : red = 5)
  (h_white : white = 7)
  (h_blue : blue = 8) :
  let total := red + white + blue
  let p_red := (red * (red - 1) * (red - 2)) / (total * (total - 1) * (total - 2))
  let p_white := (white * (white - 1) * (white - 2)) / (total * (total - 1) * (total - 2))
  let p_blue := (blue * (blue - 1) * (blue - 2)) / (total * (total - 1) * (total - 2))
  p_red + p_white + p_blue = 101 / 1140 :=
by sorry


end NUMINAMATH_CALUDE_same_color_marble_probability_l624_62411


namespace NUMINAMATH_CALUDE_specific_theater_seats_l624_62468

/-- Represents a theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row : ℕ
  seat_increase : ℕ
  last_row : ℕ

/-- Calculates the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let num_rows := (t.last_row - t.first_row) / t.seat_increase + 1
  num_rows * (t.first_row + t.last_row) / 2

/-- Theorem stating that a theater with specific parameters has 570 seats -/
theorem specific_theater_seats :
  let t : Theater := { first_row := 12, seat_increase := 2, last_row := 48 }
  total_seats t = 570 := by sorry

end NUMINAMATH_CALUDE_specific_theater_seats_l624_62468


namespace NUMINAMATH_CALUDE_sqrt_450_equals_15_sqrt_2_l624_62491

theorem sqrt_450_equals_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_equals_15_sqrt_2_l624_62491


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l624_62494

/-- Given an arithmetic sequence of 25 terms with first term 5 and last term 77,
    prove that the 10th term is 32. -/
theorem arithmetic_sequence_10th_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
    (a 1 = 5) →                          -- first term is 5
    (a 25 = 77) →                        -- last term is 77
    (a 10 = 32) :=                       -- 10th term is 32
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l624_62494


namespace NUMINAMATH_CALUDE_opposite_signs_and_greater_absolute_value_l624_62427

theorem opposite_signs_and_greater_absolute_value (a b : ℝ) 
  (h1 : a * b < 0) (h2 : a + b > 0) : 
  (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0) ∧ 
  (a > 0 → |a| > |b|) ∧ 
  (b > 0 → |b| > |a|) := by
  sorry

end NUMINAMATH_CALUDE_opposite_signs_and_greater_absolute_value_l624_62427


namespace NUMINAMATH_CALUDE_traffic_light_probability_l624_62465

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  total_duration : ℕ
  red_duration : ℕ
  yellow_duration : ℕ
  green_duration : ℕ

/-- Calculates the probability of waiting no more than a given time -/
def probability_of_waiting (cycle : TrafficLightCycle) (max_wait : ℕ) : ℚ :=
  let proceed_duration := cycle.yellow_duration + cycle.green_duration
  let favorable_duration := min max_wait cycle.red_duration + proceed_duration
  favorable_duration / cycle.total_duration

/-- The main theorem to be proved -/
theorem traffic_light_probability (cycle : TrafficLightCycle) 
  (h1 : cycle.total_duration = 80)
  (h2 : cycle.red_duration = 40)
  (h3 : cycle.yellow_duration = 10)
  (h4 : cycle.green_duration = 30) :
  probability_of_waiting cycle 10 = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_probability_l624_62465


namespace NUMINAMATH_CALUDE_find_c_l624_62442

theorem find_c (a b c : ℝ) (x : ℝ) 
  (eq : (x + a) * (x + b) = x^2 + c*x + 12)
  (h1 : b = 4)
  (h2 : a + b = 6) : 
  c = 6 := by
sorry

end NUMINAMATH_CALUDE_find_c_l624_62442


namespace NUMINAMATH_CALUDE_largest_three_digit_sum_15_l624_62415

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_sum_15 :
  ∀ n : ℕ, is_three_digit n → digit_sum n = 15 → n ≤ 960 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_sum_15_l624_62415


namespace NUMINAMATH_CALUDE_village_population_equality_l624_62437

theorem village_population_equality (t : ℝ) (G : ℝ) : ¬(t > 0 ∧ 
  78000 - 1200 * t = 42000 + 800 * t ∧
  78000 - 1200 * t = 65000 + G * t ∧
  42000 + 800 * t = 65000 + G * t) :=
sorry

end NUMINAMATH_CALUDE_village_population_equality_l624_62437


namespace NUMINAMATH_CALUDE_power_equation_solution_l624_62454

theorem power_equation_solution (m : ℕ) : 8^2 = 4^2 * 2^m → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l624_62454


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_400_l624_62407

theorem largest_multiple_of_15_under_400 : ∃ (n : ℕ), n * 15 = 390 ∧ 
  390 < 400 ∧ 
  ∀ (m : ℕ), m * 15 < 400 → m * 15 ≤ 390 := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_400_l624_62407


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l624_62401

theorem simultaneous_equations_solution (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 3 ∧ y = (3 * m - 2) * x^2 + 5) ↔ 
  (m ≤ 12 - 8 * Real.sqrt 2 ∨ m ≥ 12 + 8 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l624_62401


namespace NUMINAMATH_CALUDE_sum_congruence_and_parity_l624_62418

def sum : ℕ := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999

theorem sum_congruence_and_parity :
  (sum % 9 = 6) ∧ Even 6 := by sorry

end NUMINAMATH_CALUDE_sum_congruence_and_parity_l624_62418


namespace NUMINAMATH_CALUDE_buratino_apples_theorem_l624_62476

theorem buratino_apples_theorem (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (h_distinct : a₁ > a₂ ∧ a₂ > a₃ ∧ a₃ > a₄ ∧ a₄ > a₅ ∧ a₅ > a₆) :
  ∃ x₁ x₂ x₃ x₄ : ℝ, 
    x₁ + x₂ = a₁ ∧ 
    x₁ + x₃ = a₂ ∧ 
    x₂ + x₃ = a₃ ∧ 
    x₃ + x₄ = a₄ ∧ 
    x₁ + x₄ ≥ a₅ ∧ 
    x₂ + x₄ ≥ a₆ ∧
    ∀ y₁ y₂ y₃ y₄ : ℝ, 
      (y₁ + y₂ = a₁ → y₁ + y₃ = a₂ → y₂ + y₃ = a₃ → y₃ + y₄ = a₄ → 
       y₁ + y₄ = a₅ → y₂ + y₄ = a₆) → False :=
by sorry

end NUMINAMATH_CALUDE_buratino_apples_theorem_l624_62476


namespace NUMINAMATH_CALUDE_power_of_five_mod_eight_l624_62495

theorem power_of_five_mod_eight : 5^1082 % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_mod_eight_l624_62495


namespace NUMINAMATH_CALUDE_angle_equality_l624_62458

-- Define angles A, B, and C
variable (A B C : ℝ)

-- Define the conditions
axiom angle_sum_1 : A + B = 180
axiom angle_sum_2 : B + C = 180

-- State the theorem
theorem angle_equality : A = C := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l624_62458


namespace NUMINAMATH_CALUDE_total_pizza_slices_l624_62423

theorem total_pizza_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_pizzas = 35) (h2 : slices_per_pizza = 12) : 
  num_pizzas * slices_per_pizza = 420 := by
  sorry

end NUMINAMATH_CALUDE_total_pizza_slices_l624_62423


namespace NUMINAMATH_CALUDE_function_inequality_l624_62482

open Set
open Function
open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x, (x - 1) * (deriv f x) ≤ 0)
variable (h2 : ∀ x, f (-x) = f (2 + x))

-- Define the theorem
theorem function_inequality (x₁ x₂ : ℝ) 
  (h3 : |x₁ - 1| < |x₂ - 1|) : 
  f (2 - x₁) > f (2 - x₂) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l624_62482


namespace NUMINAMATH_CALUDE_dante_balloon_sharing_l624_62405

theorem dante_balloon_sharing :
  ∀ (num_friends : ℕ),
    num_friends > 0 →
    250 / num_friends - 11 = 39 →
    num_friends = 5 := by
  sorry

end NUMINAMATH_CALUDE_dante_balloon_sharing_l624_62405


namespace NUMINAMATH_CALUDE_triangle_area_heron_l624_62410

theorem triangle_area_heron (a b c : ℝ) (h_a : a = 6) (h_b : b = 8) (h_c : c = 10) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_heron_l624_62410


namespace NUMINAMATH_CALUDE_students_taking_music_l624_62469

theorem students_taking_music (total : ℕ) (art : ℕ) (both : ℕ) (neither : ℕ) :
  total = 500 →
  art = 20 →
  both = 10 →
  neither = 470 →
  total - neither - art + both = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_students_taking_music_l624_62469


namespace NUMINAMATH_CALUDE_sqrt_two_minus_one_power_l624_62490

theorem sqrt_two_minus_one_power (n : ℕ+) :
  ∃ (a b : ℤ) (m : ℕ),
    (Real.sqrt 2 - 1) ^ (n : ℝ) = b * Real.sqrt 2 - a ∧
    m = a ^ 2 * b ^ 2 + 1 ∧
    m > 1 ∧
    (Real.sqrt 2 - 1) ^ (n : ℝ) = Real.sqrt m - Real.sqrt (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_one_power_l624_62490


namespace NUMINAMATH_CALUDE_factor_calculation_l624_62466

theorem factor_calculation (x : ℝ) : 60 * x - 138 = 102 ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_factor_calculation_l624_62466


namespace NUMINAMATH_CALUDE_sum_of_digits_equals_four_l624_62414

/-- Converts a base-5 number to decimal --/
def base5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (5^i)) 0

/-- Converts a decimal number to base-6 --/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The base-5 representation of 2014₅ --/
def base5Number : List Nat := [4, 1, 0, 2]

theorem sum_of_digits_equals_four :
  let decimal := base5ToDecimal base5Number
  let base6 := decimalToBase6 decimal
  base6.sum = 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_equals_four_l624_62414


namespace NUMINAMATH_CALUDE_triangle_with_altitudes_is_obtuse_l624_62400

/-- A triangle with given altitudes is obtuse -/
theorem triangle_with_altitudes_is_obtuse (h_a h_b h_c : ℝ) 
  (h_alt_a : h_a = 1/14)
  (h_alt_b : h_b = 1/10)
  (h_alt_c : h_c = 1/5)
  (h_positive_a : h_a > 0)
  (h_positive_b : h_b > 0)
  (h_positive_c : h_c > 0) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b > c ∧ b + c > a ∧ a + c > b ∧
    a * h_a = b * h_b ∧ b * h_b = c * h_c ∧
    (b^2 + c^2 - a^2) / (2 * b * c) < 0 :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_altitudes_is_obtuse_l624_62400


namespace NUMINAMATH_CALUDE_selection_theorem_l624_62493

theorem selection_theorem (n_volunteers : ℕ) (n_bokchoys : ℕ) : 
  n_volunteers = 4 → n_bokchoys = 3 → 
  (Nat.choose (n_volunteers + n_bokchoys) 4 - Nat.choose n_volunteers 4) = 34 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l624_62493


namespace NUMINAMATH_CALUDE_pair_probability_after_removal_l624_62408

def initial_deck_size : ℕ := 52
def cards_per_value : ℕ := 4
def values_count : ℕ := 13
def removed_pairs : ℕ := 2

def remaining_deck_size : ℕ := initial_deck_size - removed_pairs * cards_per_value / 2

def total_ways_to_select_two : ℕ := remaining_deck_size.choose 2

def full_value_count : ℕ := values_count - removed_pairs
def pair_forming_ways_full : ℕ := full_value_count * (cards_per_value.choose 2)
def pair_forming_ways_reduced : ℕ := removed_pairs * ((cards_per_value - 2).choose 2)
def total_pair_forming_ways : ℕ := pair_forming_ways_full + pair_forming_ways_reduced

theorem pair_probability_after_removal : 
  (total_pair_forming_ways : ℚ) / total_ways_to_select_two = 17 / 282 := by sorry

end NUMINAMATH_CALUDE_pair_probability_after_removal_l624_62408


namespace NUMINAMATH_CALUDE_student_allowance_l624_62434

theorem student_allowance (initial_allowance : ℚ) : 
  let remaining_after_clothes := initial_allowance * (4/7)
  let remaining_after_games := remaining_after_clothes * (3/5)
  let remaining_after_books := remaining_after_games * (5/9)
  let remaining_after_donation := remaining_after_books * (1/2)
  remaining_after_donation = 3.75 → initial_allowance = 39.375 := by
  sorry

end NUMINAMATH_CALUDE_student_allowance_l624_62434


namespace NUMINAMATH_CALUDE_odd_k_triple_f_35_l624_62452

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n - 2

theorem odd_k_triple_f_35 (k : ℤ) (h1 : k % 2 = 1) (h2 : f (f (f k)) = 35) : k = 29 := by
  sorry

end NUMINAMATH_CALUDE_odd_k_triple_f_35_l624_62452


namespace NUMINAMATH_CALUDE_dot_product_problem_l624_62447

theorem dot_product_problem (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  (a.1 + b.1, a.2 + b.2) = (-1, 1) →
  a.1 * b.1 + a.2 * b.2 = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_dot_product_problem_l624_62447


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l624_62438

/-- An isosceles triangle with two interior angles summing to 100° has a vertex angle of either 20° or 80°. -/
theorem isosceles_triangle_vertex_angle (a b c : ℝ) :
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  (a = b ∨ b = c ∨ a = c) →  -- Triangle is isosceles
  ((a + b = 100 ∧ c ≠ b) ∨ (b + c = 100 ∧ a ≠ b) ∨ (a + c = 100 ∧ a ≠ b)) →  -- Two angles sum to 100°
  (c = 20 ∨ c = 80) ∨ (a = 20 ∨ a = 80) ∨ (b = 20 ∨ b = 80) :=  -- Vertex angle is 20° or 80°
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l624_62438


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l624_62446

-- Define the sets A, B, and M
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}
def M (k : ℝ) : Set ℝ := {x | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1}

-- Theorem statement
theorem set_operations_and_subset :
  (A ∩ B = {x | 1 < x ∧ x ≤ 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x ≤ 1 ∨ x > 3}) ∧
  (∀ k, M k ⊆ A ↔ k < -5/2 ∨ k > 1) := by sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l624_62446


namespace NUMINAMATH_CALUDE_range_of_a_l624_62420

/-- The range of values for a given the conditions -/
theorem range_of_a (f g : ℝ → ℝ) (a : ℝ) : 
  (∀ x > 0, f x = x * Real.log x) →
  (∀ x, g x = x^3 + a*x - x + 2) →
  (∀ x > 0, 2 * f x ≤ deriv g x + 2) →
  a ≥ -2 ∧ ∀ b ≥ -2, ∃ x > 0, 2 * f x ≤ deriv g x + 2 :=
by sorry


end NUMINAMATH_CALUDE_range_of_a_l624_62420


namespace NUMINAMATH_CALUDE_unique_seq_largest_gt_100_l624_62483

/-- A sequence of 9 positive integers with unique sums property -/
def UniqueSeq (a : Fin 9 → ℕ) : Prop :=
  (∀ i j, i < j → a i < a j) ∧
  (∀ s₁ s₂ : Finset (Fin 9), s₁ ≠ s₂ → s₁.sum a ≠ s₂.sum a)

/-- Theorem: In a sequence with unique sums property, the largest element is greater than 100 -/
theorem unique_seq_largest_gt_100 (a : Fin 9 → ℕ) (h : UniqueSeq a) : a 8 > 100 := by
  sorry

end NUMINAMATH_CALUDE_unique_seq_largest_gt_100_l624_62483


namespace NUMINAMATH_CALUDE_hyperbola_equation_l624_62421

/-- Given a hyperbola and an ellipse with shared foci and related eccentricities,
    prove that the hyperbola has the equation x²/4 - y²/3 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) ∧ 
  (∃ (x y : ℝ), x^2/16 + y^2/9 = 1) ∧
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c^2 = 16 - 9) ∧
  (∃ (e_h e_e : ℝ), e_h = c/a ∧ e_e = c/4 ∧ e_h = 2*e_e) →
  a^2 = 4 ∧ b^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l624_62421


namespace NUMINAMATH_CALUDE_horner_method_result_l624_62486

def f (x : ℝ) : ℝ := 9 + 15*x - 8*x^2 - 20*x^3 + 6*x^4 + 3*x^5

theorem horner_method_result : f 4 = 3269 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_result_l624_62486


namespace NUMINAMATH_CALUDE_heart_then_face_prob_l624_62460

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- The suit of a card -/
inductive Suit
  | Hearts | Diamonds | Clubs | Spades

/-- The rank of a card -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A playing card -/
structure Card :=
  (suit : Suit)
  (rank : Rank)

/-- Definition of a face card -/
def isFaceCard (c : Card) : Prop :=
  c.rank = Rank.Jack ∨ c.rank = Rank.Queen ∨ c.rank = Rank.King ∨ c.rank = Rank.Ace

/-- The probability of drawing a heart as the first card and a face card as the second -/
def heartThenFaceProbability (d : Deck) : ℚ :=
  5 / 86

/-- Theorem stating the probability of drawing a heart then a face card -/
theorem heart_then_face_prob (d : Deck) :
  heartThenFaceProbability d = 5 / 86 := by
  sorry


end NUMINAMATH_CALUDE_heart_then_face_prob_l624_62460


namespace NUMINAMATH_CALUDE_equation_solution_l624_62443

theorem equation_solution : ∃ k : ℝ, (5/9 * (k^2 - 32))^3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l624_62443


namespace NUMINAMATH_CALUDE_jeds_speed_jeds_speed_is_89_l624_62441

def speed_limit : ℕ := 50
def speeding_fine_per_mph : ℕ := 16
def red_light_fine : ℕ := 75
def cellphone_fine : ℕ := 120
def parking_fine : ℕ := 50
def total_fine : ℕ := 1046
def red_light_violations : ℕ := 2
def parking_violations : ℕ := 3

theorem jeds_speed : ℕ :=
  let non_speeding_fines := red_light_fine * red_light_violations + 
                            cellphone_fine + 
                            parking_fine * parking_violations
  let speeding_fine := total_fine - non_speeding_fines
  let mph_over_limit := speeding_fine / speeding_fine_per_mph
  speed_limit + mph_over_limit

#check jeds_speed

theorem jeds_speed_is_89 : jeds_speed = 89 := by
  sorry

end NUMINAMATH_CALUDE_jeds_speed_jeds_speed_is_89_l624_62441


namespace NUMINAMATH_CALUDE_inverse_g_84_l624_62498

def g (x : ℝ) : ℝ := 3 * x^3 + 3

theorem inverse_g_84 : g⁻¹ 84 = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_g_84_l624_62498


namespace NUMINAMATH_CALUDE_update_year_is_ninth_l624_62497

def maintenance_cost (n : ℕ) : ℚ :=
  if n ≤ 7 then 2 * n + 2 else 16 * (5/4)^(n-7)

def maintenance_sum (n : ℕ) : ℚ :=
  if n ≤ 7 then n^2 + 3*n else 80 * (5/4)^(n-7) - 10

def average_maintenance_cost (n : ℕ) : ℚ :=
  maintenance_sum n / n

theorem update_year_is_ninth :
  ∀ k, k < 9 → average_maintenance_cost k ≤ 12 ∧
  average_maintenance_cost 9 > 12 :=
sorry

end NUMINAMATH_CALUDE_update_year_is_ninth_l624_62497


namespace NUMINAMATH_CALUDE_sport_drink_water_amount_l624_62462

/-- Represents the composition of a sport drink -/
structure SportDrink where
  flavoringRatio : ℚ
  cornSyrupRatio : ℚ
  waterRatio : ℚ
  cornSyrupOunces : ℚ

/-- Calculates the amount of water in a sport drink -/
def waterAmount (drink : SportDrink) : ℚ :=
  (drink.waterRatio / drink.cornSyrupRatio) * drink.cornSyrupOunces

/-- Theorem stating the amount of water in the sport drink -/
theorem sport_drink_water_amount 
  (drink : SportDrink)
  (h1 : drink.flavoringRatio = 1)
  (h2 : drink.cornSyrupRatio = 4)
  (h3 : drink.waterRatio = 60)
  (h4 : drink.cornSyrupOunces = 7) :
  waterAmount drink = 105 := by
  sorry

#check sport_drink_water_amount

end NUMINAMATH_CALUDE_sport_drink_water_amount_l624_62462


namespace NUMINAMATH_CALUDE_initial_bunnies_l624_62488

theorem initial_bunnies (initial : ℕ) : 
  (3 / 5 : ℚ) * initial + 2 * ((3 / 5 : ℚ) * initial) = 54 → initial = 30 := by
  sorry

end NUMINAMATH_CALUDE_initial_bunnies_l624_62488


namespace NUMINAMATH_CALUDE_money_sharing_l624_62419

theorem money_sharing (ken_share tony_share total : ℕ) : 
  ken_share = 1750 →
  tony_share = 2 * ken_share →
  total = ken_share + tony_share →
  total = 5250 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_l624_62419


namespace NUMINAMATH_CALUDE_kirill_height_l624_62485

/-- Represents the heights of three siblings -/
structure SiblingHeights where
  kirill : ℝ
  brother : ℝ
  sister : ℝ

/-- The conditions of the problem -/
def height_conditions (h : SiblingHeights) : Prop :=
  h.brother = h.kirill + 14 ∧
  h.sister = 2 * h.kirill ∧
  h.kirill + h.brother + h.sister = 264

/-- Theorem stating Kirill's height given the conditions -/
theorem kirill_height (h : SiblingHeights) 
  (hc : height_conditions h) : h.kirill = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_kirill_height_l624_62485


namespace NUMINAMATH_CALUDE_original_price_calculation_l624_62499

theorem original_price_calculation (discount_percentage : ℝ) (discounted_price : ℝ) : 
  discount_percentage = 20 ∧ discounted_price = 96 → 
  ∃ (original_price : ℝ), original_price = 120 ∧ discounted_price = original_price * (1 - discount_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_original_price_calculation_l624_62499


namespace NUMINAMATH_CALUDE_existence_of_prime_and_sequence_l624_62433

theorem existence_of_prime_and_sequence (k : ℕ+) :
  ∃ (p : ℕ) (a : Fin (k+3) → ℕ), 
    Prime p ∧ 
    (∀ i : Fin (k+3), 1 ≤ a i ∧ a i < p) ∧
    (∀ i j : Fin (k+3), i ≠ j → a i ≠ a j) ∧
    (∀ i : Fin k, p ∣ (a i * a (i+1) * a (i+2) * a (i+3) - i)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_prime_and_sequence_l624_62433


namespace NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l624_62430

theorem no_integer_satisfies_conditions : 
  ¬∃ (n : ℤ), ∃ (k : ℤ), 
    n / (25 - n) = k^2 ∧ 
    ∃ (m : ℤ), n = 3 * m :=
by sorry

end NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l624_62430


namespace NUMINAMATH_CALUDE_collinear_vectors_m_value_l624_62445

def a (m : ℝ) : Fin 2 → ℝ := ![2*m, 3]
def b (m : ℝ) : Fin 2 → ℝ := ![m-1, 1]

theorem collinear_vectors_m_value (m : ℝ) :
  (∃ (k : ℝ), a m = k • b m) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_value_l624_62445


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l624_62425

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ (p : ℚ) / q < 5 / 8 →
  q ≥ 13 ∧ (q = 13 → p = 8) :=
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l624_62425


namespace NUMINAMATH_CALUDE_smallest_cookie_packages_l624_62479

theorem smallest_cookie_packages (cookie_per_package : Nat) (milk_per_package : Nat) 
  (h1 : cookie_per_package = 5) (h2 : milk_per_package = 7) :
  ∃ n : Nat, n > 0 ∧ (cookie_per_package * n) % milk_per_package = 0 ∧
  ∀ m : Nat, m > 0 ∧ (cookie_per_package * m) % milk_per_package = 0 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_cookie_packages_l624_62479


namespace NUMINAMATH_CALUDE_ellipse_properties_l624_62404

/-- Ellipse structure -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line with slope 1 -/
structure Line where
  c : ℝ

/-- Theorem about ellipse properties -/
theorem ellipse_properties (E : Ellipse) (F₁ F₂ A B P : Point) (l : Line) :
  -- Line l passes through F₁ and has slope 1
  F₁.x = -E.a.sqrt^2 - E.b^2 ∧ F₁.y = 0 ∧ l.c = F₁.x →
  -- A and B are intersection points of l and E
  (A.x^2 / E.a^2 + A.y^2 / E.b^2 = 1) ∧ (B.x^2 / E.a^2 + B.y^2 / E.b^2 = 1) ∧
  A.x = A.y - l.c ∧ B.x = B.y - l.c →
  -- |AF₂|, |AB|, |BF₂| form arithmetic sequence
  2 * ((A.x - B.x)^2 + (A.y - B.y)^2) = 
    ((A.x - F₂.x)^2 + (A.y - F₂.y)^2) + ((B.x - F₂.x)^2 + (B.y - F₂.y)^2) →
  -- P(0, -1) satisfies |PA| = |PB|
  P.x = 0 ∧ P.y = -1 ∧
  (P.x - A.x)^2 + (P.y - A.y)^2 = (P.x - B.x)^2 + (P.y - B.y)^2 →
  -- Eccentricity is √2/2 and equation is x^2/18 + y^2/9 = 1
  (E.a^2 - E.b^2) / E.a^2 = 1/2 ∧ E.a^2 = 18 ∧ E.b^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l624_62404


namespace NUMINAMATH_CALUDE_sqrt_domain_sqrt_nonneg_sqrt_undefined_neg_l624_62402

-- Define the square root function for non-negative real numbers
noncomputable def sqrt (a : ℝ) : ℝ := Real.sqrt a

-- Theorem stating that the domain of the square root function is non-negative real numbers
theorem sqrt_domain (a : ℝ) : ∃ (x : ℝ), x ^ 2 = a → a ≥ 0 := by
  sorry

-- Theorem stating that the square root of a non-negative number is non-negative
theorem sqrt_nonneg (a : ℝ) (h : a ≥ 0) : sqrt a ≥ 0 := by
  sorry

-- Theorem stating that the square root function is undefined for negative numbers
theorem sqrt_undefined_neg (a : ℝ) : a < 0 → ¬∃ (x : ℝ), x ^ 2 = a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_domain_sqrt_nonneg_sqrt_undefined_neg_l624_62402


namespace NUMINAMATH_CALUDE_james_total_earnings_l624_62450

def january_earnings : ℕ := 4000

def february_earnings (jan : ℕ) : ℕ := 2 * jan

def march_earnings (feb : ℕ) : ℕ := feb - 2000

def total_earnings (jan feb mar : ℕ) : ℕ := jan + feb + mar

theorem james_total_earnings :
  total_earnings january_earnings (february_earnings january_earnings) (march_earnings (february_earnings january_earnings)) = 18000 := by
  sorry

end NUMINAMATH_CALUDE_james_total_earnings_l624_62450


namespace NUMINAMATH_CALUDE_unit_square_fits_in_parallelogram_l624_62481

/-- A parallelogram with heights greater than 1 -/
structure Parallelogram where
  heights : ℝ → ℝ
  height_gt_one : ∀ h, heights h > 1

/-- A unit square -/
structure UnitSquare where
  side_length : ℝ
  is_unit : side_length = 1

/-- A placement of a shape inside a parallelogram -/
structure Placement (P : Parallelogram) (S : Type) where
  is_inside : S → Bool

/-- Theorem: For any parallelogram with heights greater than 1, 
    there exists a placement of a unit square inside it -/
theorem unit_square_fits_in_parallelogram (P : Parallelogram) :
  ∃ (U : UnitSquare) (place : Placement P UnitSquare), place.is_inside U = true := by
  sorry

end NUMINAMATH_CALUDE_unit_square_fits_in_parallelogram_l624_62481


namespace NUMINAMATH_CALUDE_special_number_exists_l624_62448

/-- Number of digits in a natural number -/
def number_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For every natural number a, there exists a natural number b and a non-negative integer k
    such that a * 10^k + b = a * (b * 10^(number_of_digits a) + a) -/
theorem special_number_exists (a : ℕ) : ∃ (b : ℕ) (k : ℕ), 
  a * 10^k + b = a * (b * 10^(number_of_digits a) + a) := by sorry

end NUMINAMATH_CALUDE_special_number_exists_l624_62448


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l624_62492

theorem partial_fraction_decomposition_sum (x A B C D E : ℝ) : 
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x - 4)) = 
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x - 4) →
  A + B + C + D + E = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l624_62492


namespace NUMINAMATH_CALUDE_calculation_proof_l624_62480

theorem calculation_proof :
  (4.8 * (3.5 - 2.1) / 7 = 0.96) ∧
  (18.75 - 0.23 * 2 - 4.54 = 13.75) ∧
  (0.9 + 99 * 0.9 = 90) ∧
  (4 / 0.8 - 0.8 / 4 = 4.8) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l624_62480


namespace NUMINAMATH_CALUDE_average_is_three_l624_62416

/-- Given four real numbers A, B, C, and D satisfying certain conditions,
    prove that their average is 3. -/
theorem average_is_three (A B C D : ℝ) 
    (eq1 : 501 * C - 2004 * A = 3006)
    (eq2 : 2502 * B + 6006 * A = 10010)
    (eq3 : D = A + 2) :
    (A + B + C + D) / 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_is_three_l624_62416


namespace NUMINAMATH_CALUDE_triangle_area_change_l624_62459

/-- Theorem: Effect on triangle area when height is decreased by 40% and base is increased by 40% -/
theorem triangle_area_change (base height : ℝ) (base_new height_new area area_new : ℝ) 
  (h1 : base_new = base * 1.4)
  (h2 : height_new = height * 0.6)
  (h3 : area = (base * height) / 2)
  (h4 : area_new = (base_new * height_new) / 2) :
  area_new = area * 0.84 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_change_l624_62459


namespace NUMINAMATH_CALUDE_fraction_identity_l624_62470

theorem fraction_identity (n : ℕ) : 
  2 / ((2 * n - 1) * (2 * n + 1)) = 1 / (2 * n - 1) - 1 / (2 * n + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_identity_l624_62470


namespace NUMINAMATH_CALUDE_perpendicular_tangents_range_l624_62473

open Real

theorem perpendicular_tangents_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    let f : ℝ → ℝ := λ x => a * x + sin x + cos x
    let f' : ℝ → ℝ := λ x => a + cos x - sin x
    (f' x₁) * (f' x₂) = -1) →
  -1 ≤ a ∧ a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_range_l624_62473


namespace NUMINAMATH_CALUDE_victors_flower_stickers_l624_62457

theorem victors_flower_stickers :
  ∀ (flower_stickers animal_stickers : ℕ),
    animal_stickers = flower_stickers - 2 →
    flower_stickers + animal_stickers = 14 →
    flower_stickers = 8 := by
  sorry

end NUMINAMATH_CALUDE_victors_flower_stickers_l624_62457


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l624_62412

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) := a * x^2 + 5*x - 2

-- Define the solution set of the original inequality
def solution_set (a : ℝ) := {x : ℝ | 1/2 < x ∧ x < 2}

-- Define the second quadratic function
def g (a : ℝ) (x : ℝ) := a * x^2 - 5*x + a^2 - 1

-- Theorem statement
theorem quadratic_inequality_problem 
  (a : ℝ) 
  (h : ∀ x, f a x > 0 ↔ x ∈ solution_set a) :
  a = -2 ∧ 
  (∀ x, g a x > 0 ↔ -3 < x ∧ x < 1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l624_62412


namespace NUMINAMATH_CALUDE_cubes_volume_percentage_l624_62422

def box_length : ℕ := 8
def box_width : ℕ := 5
def box_height : ℕ := 12
def cube_side : ℕ := 4

def cubes_per_dimension (box_dim : ℕ) (cube_dim : ℕ) : ℕ :=
  box_dim / cube_dim

def total_cubes : ℕ :=
  (cubes_per_dimension box_length cube_side) *
  (cubes_per_dimension box_width cube_side) *
  (cubes_per_dimension box_height cube_side)

def cube_volume : ℕ := cube_side ^ 3
def total_cubes_volume : ℕ := total_cubes * cube_volume
def box_volume : ℕ := box_length * box_width * box_height

theorem cubes_volume_percentage :
  (total_cubes_volume : ℚ) / (box_volume : ℚ) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cubes_volume_percentage_l624_62422


namespace NUMINAMATH_CALUDE_inequality_solution_l624_62439

theorem inequality_solution (p q : ℝ) :
  q > 0 →
  (3 * (p * q^2 + p^2 * q + 3 * q^2 + 3 * p * q)) / (p + q) > 2 * p^2 * q ↔
  p ≥ 0 ∧ p < 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l624_62439


namespace NUMINAMATH_CALUDE_lattice_points_on_segment_l624_62428

/-- The number of lattice points on a line segment with given endpoints -/
def latticePointCount (x₁ y₁ x₂ y₂ : ℤ) : ℕ :=
  sorry

/-- Theorem: The number of lattice points on the line segment from (7,23) to (61,353) is 7 -/
theorem lattice_points_on_segment : latticePointCount 7 23 61 353 = 7 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_on_segment_l624_62428


namespace NUMINAMATH_CALUDE_no_roots_geq_two_l624_62456

theorem no_roots_geq_two : ∀ x : ℝ, x ≥ 2 → 4 * x^3 - 5 * x^2 - 6 * x + 3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_no_roots_geq_two_l624_62456


namespace NUMINAMATH_CALUDE_repeated_digit_sum_2_power_2004_l624_62496

/-- The digit sum function -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Repeated application of digit_sum until a single digit is reached -/
def repeated_digit_sum (n : ℕ) : ℕ := sorry

/-- 2^2004 mod 9 ≡ 1 -/
lemma power_two_2004_mod_9 : 2^2004 % 9 = 1 := sorry

/-- For any natural number n, n ≡ digit_sum(n) (mod 9) -/
lemma digit_sum_congruence (n : ℕ) : n % 9 = digit_sum n % 9 := sorry

/-- The main theorem -/
theorem repeated_digit_sum_2_power_2004 : 
  repeated_digit_sum (2^2004) = 1 := sorry

end NUMINAMATH_CALUDE_repeated_digit_sum_2_power_2004_l624_62496


namespace NUMINAMATH_CALUDE_constant_term_of_f_composition_l624_62429

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (x - 1/x)^8 else -Real.sqrt x

theorem constant_term_of_f_composition (x : ℝ) (h : x > 0) :
  ∃ (expansion : ℝ → ℝ),
    (∀ y, y > 0 → f (f y) = expansion y) ∧
    (∃ c, ∀ ε > 0, |expansion x - c| < ε) ∧
    (∀ c, (∃ ε > 0, |expansion x - c| < ε) → c = 70) :=
sorry

end NUMINAMATH_CALUDE_constant_term_of_f_composition_l624_62429


namespace NUMINAMATH_CALUDE_complex_fraction_squared_l624_62436

theorem complex_fraction_squared (i : ℂ) : i * i = -1 → ((1 - i) / (1 + i))^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_squared_l624_62436


namespace NUMINAMATH_CALUDE_trip_distance_is_3_6_miles_l624_62409

/-- Calculates the trip distance given the taxi fare parameters -/
def calculate_trip_distance (initial_fee : ℚ) (additional_charge : ℚ) (charge_distance : ℚ) (total_charge : ℚ) : ℚ :=
  let distance_charge := total_charge - initial_fee
  let segments := distance_charge / additional_charge
  segments * charge_distance

/-- Proves that the trip distance is 3.6 miles given the specified taxi fare parameters -/
theorem trip_distance_is_3_6_miles :
  let initial_fee : ℚ := 9/4  -- $2.25
  let additional_charge : ℚ := 3/10  -- $0.3
  let charge_distance : ℚ := 2/5  -- 2/5 mile
  let total_charge : ℚ := 99/20  -- $4.95
  calculate_trip_distance initial_fee additional_charge charge_distance total_charge = 18/5  -- 3.6 miles
  := by sorry

end NUMINAMATH_CALUDE_trip_distance_is_3_6_miles_l624_62409


namespace NUMINAMATH_CALUDE_frog_count_l624_62463

theorem frog_count : ∀ (N : ℕ), 
  (∃ (T : ℝ), T > 0 ∧
    50 * (0.3 * T / 50) ≤ 0.43 * T / (N - 94) ∧
    0.43 * T / (N - 94) ≤ 44 * (0.27 * T / 44) ∧
    N > 94)
  → N = 165 := by
sorry

end NUMINAMATH_CALUDE_frog_count_l624_62463


namespace NUMINAMATH_CALUDE_coffee_maker_price_l624_62487

def original_price (sale_price : ℝ) (discount : ℝ) : ℝ :=
  sale_price + discount

theorem coffee_maker_price :
  let sale_price : ℝ := 70
  let discount : ℝ := 20
  original_price sale_price discount = 90 :=
by sorry

end NUMINAMATH_CALUDE_coffee_maker_price_l624_62487


namespace NUMINAMATH_CALUDE_hot_chocolate_consumption_l624_62472

/-- The number of cups of hot chocolate Tom can drink in 5 hours -/
def cups_in_five_hours : ℕ := 15

/-- The time interval between each cup of hot chocolate in minutes -/
def interval_minutes : ℕ := 20

/-- The total time in hours -/
def total_time_hours : ℕ := 5

/-- Theorem stating the number of cups Tom can drink in 5 hours -/
theorem hot_chocolate_consumption :
  cups_in_five_hours = (total_time_hours * 60) / interval_minutes :=
by sorry

end NUMINAMATH_CALUDE_hot_chocolate_consumption_l624_62472


namespace NUMINAMATH_CALUDE_percentage_decrease_l624_62475

theorem percentage_decrease (original : ℝ) (increase_percent : ℝ) (difference : ℝ) : 
  original = 80 →
  increase_percent = 12.5 →
  difference = 30 →
  let increased_value := original * (1 + increase_percent / 100)
  let decreased_value := original * (1 - (25 : ℝ) / 100)
  increased_value - decreased_value = difference :=
by
  sorry

#check percentage_decrease

end NUMINAMATH_CALUDE_percentage_decrease_l624_62475


namespace NUMINAMATH_CALUDE_bills_naps_l624_62424

theorem bills_naps (total_hours : ℕ) (work_hours : ℕ) (nap_duration : ℕ) : 
  total_hours = 96 → work_hours = 54 → nap_duration = 7 → 
  (total_hours - work_hours) / nap_duration = 6 := by
sorry

end NUMINAMATH_CALUDE_bills_naps_l624_62424


namespace NUMINAMATH_CALUDE_existence_of_symmetric_axis_l624_62451

/-- Represents the color of a stone -/
inductive Color
| Black
| White

/-- Represents a regular 13-gon with colored stones at each vertex -/
def Regular13Gon := Fin 13 → Color

/-- Counts the number of symmetric pairs with the same color for a given axis -/
def symmetricPairsCount (polygon : Regular13Gon) (axis : Fin 13) : ℕ :=
  sorry

/-- Main theorem: There exists an axis with at least 4 symmetric pairs of the same color -/
theorem existence_of_symmetric_axis (polygon : Regular13Gon) :
  ∃ axis : Fin 13, symmetricPairsCount polygon axis ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_symmetric_axis_l624_62451


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l624_62461

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ a) → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l624_62461


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l624_62474

theorem quadratic_equation_roots (a b c : ℝ) (h1 : a ≠ 0) :
  let r1 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r2 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (a = 1 ∧ b = -6 ∧ c = -7) →
  (r1 + r2 = 6 ∧ r1 - r2 = 8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l624_62474


namespace NUMINAMATH_CALUDE_coefficient_sum_l624_62467

theorem coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_coefficient_sum_l624_62467


namespace NUMINAMATH_CALUDE_division_remainder_l624_62455

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 271 →
  divisor = 30 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 1 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l624_62455


namespace NUMINAMATH_CALUDE_q_polynomial_form_l624_62489

theorem q_polynomial_form (q : ℝ → ℝ) 
  (h : ∀ x, q x + (2 * x^5 + 5 * x^4 + 4 * x^3 + 12 * x) = 3 * x^4 + 14 * x^3 + 32 * x^2 + 17 * x + 3) :
  ∀ x, q x = -2 * x^5 - 2 * x^4 + 10 * x^3 + 32 * x^2 + 5 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_q_polynomial_form_l624_62489


namespace NUMINAMATH_CALUDE_radius_greater_than_distance_to_center_l624_62453

/-- A circle with center O and a point P inside it -/
structure CircleWithInnerPoint where
  O : ℝ × ℝ  -- Center of the circle
  P : ℝ × ℝ  -- Point inside the circle
  r : ℝ      -- Radius of the circle
  h_inside : dist P O < r  -- P is inside the circle

/-- The theorem stating that if P is inside circle O and distance from P to O is 5,
    then the radius of circle O must be greater than 5 -/
theorem radius_greater_than_distance_to_center 
  (c : CircleWithInnerPoint) (h : dist c.P c.O = 5) : c.r > 5 := by
  sorry

end NUMINAMATH_CALUDE_radius_greater_than_distance_to_center_l624_62453


namespace NUMINAMATH_CALUDE_triangle_area_l624_62426

-- Define the triangle
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  cos_angle : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.side1 = 5 ∧ 
  t.side2 = 3 ∧ 
  5 * t.cos_angle^2 - 7 * t.cos_angle - 6 = 0

-- Theorem statement
theorem triangle_area (t : Triangle) 
  (h : triangle_conditions t) : 
  (1/2) * t.side1 * t.side2 * Real.sqrt (1 - t.cos_angle^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l624_62426


namespace NUMINAMATH_CALUDE_grape_juice_theorem_l624_62406

/-- Represents a fruit drink composition -/
structure FruitDrink where
  total : ℝ
  orange_percent : ℝ
  watermelon_percent : ℝ

/-- Calculates the amount of grape juice in the drink -/
def grape_juice_amount (drink : FruitDrink) : ℝ :=
  drink.total - (drink.orange_percent * drink.total + drink.watermelon_percent * drink.total)

/-- Theorem: The amount of grape juice in the specified drink is 70 ounces -/
theorem grape_juice_theorem (drink : FruitDrink) 
    (h1 : drink.total = 200)
    (h2 : drink.orange_percent = 0.25)
    (h3 : drink.watermelon_percent = 0.40) : 
  grape_juice_amount drink = 70 := by
  sorry

#eval grape_juice_amount { total := 200, orange_percent := 0.25, watermelon_percent := 0.40 }

end NUMINAMATH_CALUDE_grape_juice_theorem_l624_62406


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l624_62477

def f (x : ℝ) : ℝ := -3 * x^2 + 2

theorem vertex_of_quadratic (x : ℝ) :
  (∀ x, f x ≤ f 0) ∧ f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l624_62477


namespace NUMINAMATH_CALUDE_operation_problem_l624_62449

-- Define the set of operations
inductive Operation
| Add
| Sub
| Mul
| Div

-- Define the function that applies the operation
def apply_op (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- State the theorem
theorem operation_problem (star mul : Operation) (h : apply_op star 16 4 / apply_op mul 8 2 = 4) :
  apply_op star 9 3 / apply_op mul 18 6 = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_operation_problem_l624_62449


namespace NUMINAMATH_CALUDE_inequality_solution_set_l624_62431

theorem inequality_solution_set (x : ℝ) : 
  (4 * x + 2 < (x - 1)^2 ∧ (x - 1)^2 < 6 * x + 4) ↔ 
  (4 - 2 * Real.sqrt 19 < x ∧ x < 3 + 2 * Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l624_62431


namespace NUMINAMATH_CALUDE_fraction_well_defined_l624_62435

theorem fraction_well_defined (x : ℝ) (h : x ≠ 2) : 2 * x - 4 ≠ 0 := by
  sorry

#check fraction_well_defined

end NUMINAMATH_CALUDE_fraction_well_defined_l624_62435


namespace NUMINAMATH_CALUDE_average_equation_l624_62464

theorem average_equation (y : ℝ) : 
  (55 + 48 + 507 + 2 + 684 + y) / 6 = 223 → y = 42 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_l624_62464


namespace NUMINAMATH_CALUDE_correlation_difference_l624_62478

/-- Represents a relationship between two variables -/
structure Relationship where
  var1 : String
  var2 : String
  description : String

/-- Determines if a relationship represents a positive correlation -/
def is_positive_correlation (r : Relationship) : Bool :=
  sorry  -- The actual implementation would depend on how we define positive correlation

/-- Given set of relationships -/
def relationships : List Relationship := [
  { var1 := "teacher quality", var2 := "student performance", description := "A great teacher produces outstanding students" },
  { var1 := "tide level", var2 := "boat height", description := "A rising tide lifts all boats" },
  { var1 := "moon brightness", var2 := "visible stars", description := "The brighter the moon, the fewer the stars" },
  { var1 := "climbing height", var2 := "viewing distance", description := "Climbing high to see far" }
]

theorem correlation_difference :
  ∃ (i : Fin 4), ¬(is_positive_correlation (relationships.get i)) ∧
    (∀ (j : Fin 4), j ≠ i → is_positive_correlation (relationships.get j)) :=
  sorry

end NUMINAMATH_CALUDE_correlation_difference_l624_62478


namespace NUMINAMATH_CALUDE_twelve_divisor_number_is_1989_l624_62413

/-- The type of natural numbers with exactly 12 positive divisors. -/
def TwelveDivisorNumber (N : ℕ) : Prop :=
  (∃ (d : Fin 12 → ℕ), 
    (∀ i j, i < j → d i < d j) ∧
    (∀ i, d i ∣ N) ∧
    (∀ m, m ∣ N → ∃ i, d i = m) ∧
    (d 0 = 1) ∧
    (d 11 = N))

/-- The property that the divisor with index d₄ - 1 is equal to (d₁ + d₂ + d₄) · d₈ -/
def SpecialDivisorProperty (N : ℕ) (d : Fin 12 → ℕ) : Prop :=
  d ((d 3 : ℕ) - 1) = (d 0 + d 1 + d 3) * d 7

theorem twelve_divisor_number_is_1989 :
  ∃ N : ℕ, TwelveDivisorNumber N ∧ 
    (∃ d : Fin 12 → ℕ, SpecialDivisorProperty N d) ∧
    N = 1989 := by
  sorry

end NUMINAMATH_CALUDE_twelve_divisor_number_is_1989_l624_62413


namespace NUMINAMATH_CALUDE_inequality_relationships_l624_62484

theorem inequality_relationships (a b : ℝ) (h : a < b ∧ b < 0) :
  (1 / a > 1 / b) ∧
  (1 / (a - b) < 1 / a) ∧
  (|a| > |b|) ∧
  (a^4 > b^4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_relationships_l624_62484


namespace NUMINAMATH_CALUDE_classroom_average_score_l624_62444

theorem classroom_average_score (n : ℕ) (h1 : n > 15) :
  let total_average : ℚ := 10
  let subset_average : ℚ := 17
  let subset_size : ℕ := 15
  let remaining_average := (total_average * n - subset_average * subset_size) / (n - subset_size)
  remaining_average = (10 * n - 255 : ℚ) / (n - 15 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_classroom_average_score_l624_62444


namespace NUMINAMATH_CALUDE_power_sum_fifth_l624_62440

theorem power_sum_fifth (a b x y : ℝ) 
  (h1 : a*x + b*y = 1)
  (h2 : a*x^2 + b*y^2 = 9)
  (h3 : a*x^3 + b*y^3 = 28)
  (h4 : a*x^4 + b*y^4 = 96) :
  a*x^5 + b*y^5 = 28616 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_fifth_l624_62440


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l624_62432

theorem smallest_prime_divisor_of_sum (p : ℕ) : 
  (p.Prime ∧ p ∣ (7^15 + 9^7)) → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l624_62432


namespace NUMINAMATH_CALUDE_hidden_cave_inventory_sum_l624_62471

/-- Converts a number from base 5 to base 10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The problem statement --/
theorem hidden_cave_inventory_sum : 
  let artifact := base5ToBase10 [3, 1, 2, 4]
  let sculpture := base5ToBase10 [1, 3, 4, 2]
  let coins := base5ToBase10 [3, 1, 2]
  artifact + sculpture + coins = 982 := by
sorry

end NUMINAMATH_CALUDE_hidden_cave_inventory_sum_l624_62471


namespace NUMINAMATH_CALUDE_power_of_product_equality_l624_62417

theorem power_of_product_equality (a b : ℝ) : (-2 * a * b^2)^3 = -8 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_equality_l624_62417
