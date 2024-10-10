import Mathlib

namespace money_distribution_l439_43919

/-- Proves that B and C together have Rs. 450 given the conditions of the problem -/
theorem money_distribution (total : ℕ) (ac_sum : ℕ) (c_amount : ℕ) 
  (h1 : total = 600)
  (h2 : ac_sum = 250)
  (h3 : c_amount = 100) : 
  total - (ac_sum - c_amount) + c_amount = 450 := by
  sorry

#check money_distribution

end money_distribution_l439_43919


namespace expression_evaluation_l439_43902

theorem expression_evaluation : 6^2 - 4*5 + 2^2 = 20 := by
  sorry

end expression_evaluation_l439_43902


namespace standard_deck_three_card_selections_l439_43930

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cardsPerSuit : Nat)
  (redSuits : Nat)
  (blackSuits : Nat)

/-- A standard deck of 52 cards -/
def standardDeck : Deck :=
  { cards := 52
  , suits := 4
  , cardsPerSuit := 13
  , redSuits := 2
  , blackSuits := 2 }

/-- The number of ways to select three different cards from a deck, where order matters -/
def threeCardSelections (d : Deck) : Nat :=
  d.cards * (d.cards - 1) * (d.cards - 2)

/-- Theorem stating the number of ways to select three different cards from a standard deck -/
theorem standard_deck_three_card_selections :
  threeCardSelections standardDeck = 132600 := by
  sorry

end standard_deck_three_card_selections_l439_43930


namespace sum_of_numbers_l439_43958

theorem sum_of_numbers (x y : ℝ) : y = 2 * x - 3 ∧ y = 33 → x + y = 51 := by
  sorry

end sum_of_numbers_l439_43958


namespace sum_of_angles_F_and_C_l439_43953

-- Define the circle and points
variable (circle : Circle ℝ)
variable (A B C D E : circle.sphere)

-- Define the arcs and their measures
variable (arc_AB arc_DE : circle.sphere)
variable (measure_AB measure_DE : ℝ)

-- Define point F as intersection of chords
variable (F : circle.sphere)

-- Hypotheses
variable (h1 : measure_AB = 60)
variable (h2 : measure_DE = 72)
variable (h3 : F ∈ (circle.chord A C) ∩ (circle.chord B D))

-- Theorem statement
theorem sum_of_angles_F_and_C :
  ∃ (angle_F angle_C : ℝ),
    angle_F + angle_C = 42 ∧
    angle_F = abs ((measure circle.arc A C - measure circle.arc B D) / 2) ∧
    angle_C = measure_DE / 2 :=
sorry

end sum_of_angles_F_and_C_l439_43953


namespace adams_change_l439_43962

def adams_money : ℝ := 5.00
def airplane_cost : ℝ := 4.28

theorem adams_change :
  adams_money - airplane_cost = 0.72 := by sorry

end adams_change_l439_43962


namespace phone_not_answered_probability_l439_43903

theorem phone_not_answered_probability 
  (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.1) 
  (h2 : p2 = 0.25) 
  (h3 : p3 = 0.45) : 
  1 - (p1 + p2 + p3) = 0.2 := by
  sorry

end phone_not_answered_probability_l439_43903


namespace complex_calculation_l439_43998

def a : ℂ := 3 + 2*Complex.I
def b : ℂ := 1 - 2*Complex.I

theorem complex_calculation : 3*a - 4*b = 5 + 14*Complex.I := by
  sorry

end complex_calculation_l439_43998


namespace select_four_from_seven_l439_43973

theorem select_four_from_seven (n : ℕ) (k : ℕ) : n = 7 ∧ k = 4 → Nat.choose n k = 35 := by
  sorry

end select_four_from_seven_l439_43973


namespace carrie_tshirt_purchase_l439_43938

def tshirt_cost : ℚ := 965 / 100  -- $9.65 represented as a rational number
def number_of_tshirts : ℕ := 12

def total_cost : ℚ := tshirt_cost * number_of_tshirts

theorem carrie_tshirt_purchase :
  total_cost = 11580 / 100 := by sorry

end carrie_tshirt_purchase_l439_43938


namespace first_year_interest_rate_l439_43911

/-- Proves that the first-year interest rate is 4% given the problem conditions --/
theorem first_year_interest_rate 
  (initial_amount : ℝ) 
  (final_amount : ℝ) 
  (second_year_rate : ℝ) 
  (h1 : initial_amount = 4000)
  (h2 : final_amount = 4368)
  (h3 : second_year_rate = 0.05)
  : ∃ (R : ℝ), 
    initial_amount * (1 + R) * (1 + second_year_rate) = final_amount ∧ 
    R = 0.04 := by
  sorry

end first_year_interest_rate_l439_43911


namespace contrapositive_equivalence_l439_43932

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end contrapositive_equivalence_l439_43932


namespace matrix_equation_l439_43931

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]

def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]

theorem matrix_equation (x y z w : ℝ) 
  (h1 : A * B x y z w = B x y z w * A) 
  (h2 : 4 * y ≠ z) : 
  (x - w) / (z - 4 * y) = 1 := by
  sorry

end matrix_equation_l439_43931


namespace total_distinct_students_l439_43959

/-- Represents the number of distinct students in the mathematics competition --/
def distinct_students (germain newton young germain_newton_overlap germain_young_overlap : ℕ) : ℕ :=
  germain + newton + young - germain_newton_overlap - germain_young_overlap

/-- Theorem stating that the total number of distinct students is 32 --/
theorem total_distinct_students :
  distinct_students 13 10 12 2 1 = 32 := by
  sorry

#eval distinct_students 13 10 12 2 1

end total_distinct_students_l439_43959


namespace ellipse_major_axis_length_l439_43955

/-- The length of the major axis of an ellipse with given foci and tangent to x-axis -/
theorem ellipse_major_axis_length : 
  ∀ (E : Set (ℝ × ℝ)) (F₁ F₂ : ℝ × ℝ),
  F₁ = (2, 10) →
  F₂ = (26, 35) →
  (∃ (X : ℝ), (X, 0) ∈ E) →
  (∀ (P : ℝ × ℝ), P ∈ E ↔ 
    ∃ (k : ℝ), dist P F₁ + dist P F₂ = k ∧ 
    ∀ (Q : ℝ × ℝ), Q ∈ E → dist Q F₁ + dist Q F₂ = k) →
  ∃ (A B : ℝ × ℝ), A ∈ E ∧ B ∈ E ∧ dist A B = 102 ∧
    ∀ (P Q : ℝ × ℝ), P ∈ E → Q ∈ E → dist P Q ≤ 102 :=
by sorry


end ellipse_major_axis_length_l439_43955


namespace smallest_three_digit_multiple_plus_one_l439_43929

theorem smallest_three_digit_multiple_plus_one : ∃! n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  (∃ k : ℕ, n = 3 * k + 1) ∧
  (∃ k : ℕ, n = 4 * k + 1) ∧
  (∃ k : ℕ, n = 5 * k + 1) ∧
  (∃ k : ℕ, n = 7 * k + 1) ∧
  (∃ k : ℕ, n = 8 * k + 1) ∧
  (∀ m : ℕ, m < n →
    ¬(100 ≤ m ∧ m < 1000 ∧
      (∃ k : ℕ, m = 3 * k + 1) ∧
      (∃ k : ℕ, m = 4 * k + 1) ∧
      (∃ k : ℕ, m = 5 * k + 1) ∧
      (∃ k : ℕ, m = 7 * k + 1) ∧
      (∃ k : ℕ, m = 8 * k + 1))) :=
by sorry

end smallest_three_digit_multiple_plus_one_l439_43929


namespace elephant_giraffe_jade_ratio_l439_43971

/-- The amount of jade (in grams) needed for a giraffe statue -/
def giraffe_jade : ℝ := 120

/-- The selling price of a giraffe statue -/
def giraffe_price : ℝ := 150

/-- The selling price of an elephant statue -/
def elephant_price : ℝ := 350

/-- The total amount of jade Nancy has -/
def total_jade : ℝ := 1920

/-- The additional revenue from making all elephants instead of giraffes -/
def additional_revenue : ℝ := 400

/-- The ratio of jade used for an elephant statue to a giraffe statue -/
def jade_ratio : ℝ := 2

theorem elephant_giraffe_jade_ratio :
  let elephant_jade := giraffe_jade * jade_ratio
  let giraffe_count := total_jade / giraffe_jade
  let elephant_count := total_jade / elephant_jade
  giraffe_count * giraffe_price + additional_revenue = elephant_count * elephant_price :=
sorry

end elephant_giraffe_jade_ratio_l439_43971


namespace dogwood_trees_planted_today_l439_43999

/-- The number of dogwood trees planted today in the park. -/
def trees_planted_today : ℕ := sorry

/-- The current number of dogwood trees in the park. -/
def current_trees : ℕ := 7

/-- The number of dogwood trees to be planted tomorrow. -/
def trees_planted_tomorrow : ℕ := 2

/-- The total number of dogwood trees after planting is finished. -/
def total_trees : ℕ := 12

theorem dogwood_trees_planted_today :
  trees_planted_today = 3 :=
by
  have h : current_trees + trees_planted_today + trees_planted_tomorrow = total_trees := sorry
  sorry

end dogwood_trees_planted_today_l439_43999


namespace smallest_quotient_is_seven_l439_43942

/-- A type representing a division of numbers 1 to 10 into two groups -/
def Division := (Finset Nat) × (Finset Nat)

/-- Checks if a division is valid (contains all numbers from 1 to 10 exactly once) -/
def is_valid_division (d : Division) : Prop :=
  d.1 ∪ d.2 = Finset.range 10 ∧ d.1 ∩ d.2 = ∅

/-- Calculates the product of numbers in a Finset -/
def product (s : Finset Nat) : Nat :=
  s.prod id

/-- Checks if the division satisfies the divisibility condition -/
def satisfies_condition (d : Division) : Prop :=
  (product d.1) % (product d.2) = 0

/-- The main theorem stating the smallest possible quotient is 7 -/
theorem smallest_quotient_is_seven :
  ∀ d : Division, 
    is_valid_division d → 
    satisfies_condition d → 
    (product d.1) / (product d.2) ≥ 7 :=
sorry

end smallest_quotient_is_seven_l439_43942


namespace scavenger_hunting_students_l439_43985

theorem scavenger_hunting_students (total : ℕ) (skiing : ℕ → ℕ) (scavenger : ℕ) :
  total = 12000 →
  skiing scavenger = 2 * scavenger →
  total = skiing scavenger + scavenger →
  scavenger = 4000 := by
sorry

end scavenger_hunting_students_l439_43985


namespace total_sacks_needed_l439_43914

/-- The number of sacks of strawberries needed for the first bakery per week -/
def bakery1_weekly_need : ℕ := 2

/-- The number of sacks of strawberries needed for the second bakery per week -/
def bakery2_weekly_need : ℕ := 4

/-- The number of sacks of strawberries needed for the third bakery per week -/
def bakery3_weekly_need : ℕ := 12

/-- The number of weeks for which the supply is calculated -/
def supply_period : ℕ := 4

/-- Theorem stating that the total number of sacks needed for all bakeries in 4 weeks is 72 -/
theorem total_sacks_needed :
  (bakery1_weekly_need + bakery2_weekly_need + bakery3_weekly_need) * supply_period = 72 := by
  sorry

end total_sacks_needed_l439_43914


namespace all_propositions_true_l439_43978

-- Proposition 1
def expanded_terms (a b c d p q r m n : ℕ) : ℕ := 24

-- Proposition 2
def five_digit_numbers : ℕ := 36

-- Proposition 3
def seating_arrangements : ℕ := 24

-- Proposition 4
def odd_coefficients (x : ℝ) : ℕ := 2

theorem all_propositions_true :
  (∀ a b c d p q r m n, expanded_terms a b c d p q r m n = 24) ∧
  (five_digit_numbers = 36) ∧
  (seating_arrangements = 24) ∧
  (∀ x, odd_coefficients x = 2) :=
by sorry

end all_propositions_true_l439_43978


namespace fractions_product_one_l439_43987

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def irreducible (n d : ℕ) : Prop := Nat.gcd n d = 1

def valid_fraction (n d : ℕ) : Prop :=
  n ∈ S ∧ d ∈ S ∧ n ≠ d ∧ irreducible n d

theorem fractions_product_one :
  ∃ (n₁ d₁ n₂ d₂ n₃ d₃ : ℕ),
    valid_fraction n₁ d₁ ∧
    valid_fraction n₂ d₂ ∧
    valid_fraction n₃ d₃ ∧
    n₁ ≠ n₂ ∧ n₁ ≠ n₃ ∧ n₁ ≠ d₂ ∧ n₁ ≠ d₃ ∧
    n₂ ≠ n₃ ∧ n₂ ≠ d₁ ∧ n₂ ≠ d₃ ∧
    n₃ ≠ d₁ ∧ n₃ ≠ d₂ ∧
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧
    d₂ ≠ d₃ ∧
    (n₁ : ℚ) / d₁ * (n₂ : ℚ) / d₂ * (n₃ : ℚ) / d₃ = 1 := by
  sorry

end fractions_product_one_l439_43987


namespace mathematics_partition_ways_l439_43989

/-- Represents the word "MATHEMATICS" -/
def word : String := "MATHEMATICS"

/-- The positions of vowels in the word -/
def vowel_positions : List Nat := [2, 5, 7, 9]

/-- The number of vowels in the word -/
def num_vowels : Nat := vowel_positions.length

/-- A function to calculate the number of partition ways -/
def num_partition_ways : Nat := 4 * 3 * 3

/-- Theorem stating that the number of ways to partition the word "MATHEMATICS" 
    such that each part contains at least one vowel is 36 -/
theorem mathematics_partition_ways :
  num_partition_ways = 36 := by sorry

end mathematics_partition_ways_l439_43989


namespace age_sum_problem_l439_43922

theorem age_sum_problem (leonard_age nina_age jerome_age : ℕ) : 
  leonard_age = 6 →
  nina_age = leonard_age + 4 →
  jerome_age = 2 * nina_age →
  leonard_age + nina_age + jerome_age = 36 := by
sorry

end age_sum_problem_l439_43922


namespace p_sufficient_not_necessary_for_q_l439_43915

-- Define the function f
variable {f : ℝ → ℝ}

-- Define what it means for f to have an extreme value at x
def has_extreme_value (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y : ℝ, f y ≤ f x ∨ f y ≥ f x

-- Define what it means for f to be differentiable
def is_differentiable (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, DifferentiableAt ℝ f x

-- Define the proposition p
def p (f : ℝ → ℝ) (x : ℝ) : Prop :=
  has_extreme_value f x

-- Define the proposition q
def q (f : ℝ → ℝ) (x : ℝ) : Prop :=
  is_differentiable f ∧ deriv f x = 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ f : ℝ → ℝ, ∀ x : ℝ, p f x → q f x) ∧
  (∃ f : ℝ → ℝ, ∃ x : ℝ, q f x ∧ ¬p f x) :=
sorry

end p_sufficient_not_necessary_for_q_l439_43915


namespace weighted_average_girl_scouts_permission_l439_43949

-- Define the structure for each trip
structure Trip where
  total_scouts : ℕ
  boy_scout_percentage : ℚ
  girl_scout_percentage : ℚ
  boy_scout_permission_percentage : ℚ
  girl_scout_participation_percentage : ℚ
  girl_scout_permission_percentage : ℚ

-- Define the three trips
def trip1 : Trip := {
  total_scouts := 100,
  boy_scout_percentage := 60/100,
  girl_scout_percentage := 40/100,
  boy_scout_permission_percentage := 75/100,
  girl_scout_participation_percentage := 50/100,
  girl_scout_permission_percentage := 50/100
}

def trip2 : Trip := {
  total_scouts := 150,
  boy_scout_percentage := 50/100,
  girl_scout_percentage := 50/100,
  boy_scout_permission_percentage := 80/100,
  girl_scout_participation_percentage := 70/100,
  girl_scout_permission_percentage := 60/100
}

def trip3 : Trip := {
  total_scouts := 200,
  boy_scout_percentage := 40/100,
  girl_scout_percentage := 60/100,
  boy_scout_permission_percentage := 85/100,
  girl_scout_participation_percentage := 100/100,
  girl_scout_permission_percentage := 75/100
}

-- Function to calculate the number of Girl Scouts with permission slips for a trip
def girl_scouts_with_permission (trip : Trip) : ℚ :=
  trip.total_scouts * trip.girl_scout_percentage * trip.girl_scout_participation_percentage * trip.girl_scout_permission_percentage

-- Function to calculate the total number of participating Girl Scouts for a trip
def participating_girl_scouts (trip : Trip) : ℚ :=
  trip.total_scouts * trip.girl_scout_percentage * trip.girl_scout_participation_percentage

-- Theorem statement
theorem weighted_average_girl_scouts_permission (ε : ℚ) (h : ε > 0) :
  let total_with_permission := girl_scouts_with_permission trip1 + girl_scouts_with_permission trip2 + girl_scouts_with_permission trip3
  let total_participating := participating_girl_scouts trip1 + participating_girl_scouts trip2 + participating_girl_scouts trip3
  let weighted_average := total_with_permission / total_participating * 100
  |weighted_average - 68| < ε :=
by sorry

end weighted_average_girl_scouts_permission_l439_43949


namespace f_properties_l439_43957

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a) / (sin x + 2)

theorem f_properties (a : ℝ) (h : a ≥ -2) :
  (∀ x ∈ Set.Icc 0 (π/2), Monotone (f π)) ∧
  (∀ x ∈ Set.Icc 0 (π/2), f a x ≤ π/6 - a/3) := by
  sorry

end f_properties_l439_43957


namespace more_solutions_for_first_eq_l439_43979

/-- The upper bound for x, y, z, and t -/
def upperBound : ℕ := 10^6

/-- The number of integral solutions of x² - y² = z³ - t³ -/
def N : ℕ := sorry

/-- The number of integral solutions of x² - y² = z³ - t³ + 1 -/
def M : ℕ := sorry

/-- Predicate for the first equation -/
def firstEq (x y z t : ℕ) : Prop :=
  x^2 - y^2 = z^3 - t^3 ∧ x ≤ upperBound ∧ y ≤ upperBound ∧ z ≤ upperBound ∧ t ≤ upperBound

/-- Predicate for the second equation -/
def secondEq (x y z t : ℕ) : Prop :=
  x^2 - y^2 = z^3 - t^3 + 1 ∧ x ≤ upperBound ∧ y ≤ upperBound ∧ z ≤ upperBound ∧ t ≤ upperBound

/-- Theorem stating that N > M -/
theorem more_solutions_for_first_eq : N > M := by
  sorry

end more_solutions_for_first_eq_l439_43979


namespace exists_solution_set_exists_a_range_l439_43925

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x + 1| + |2*x - a| + a

-- Theorem for part (Ⅰ)
theorem exists_solution_set (a : ℝ) (h : a = 3) :
  ∃ S : Set ℝ, ∀ x ∈ S, f x a > 7 :=
sorry

-- Theorem for part (Ⅱ)
theorem exists_a_range :
  ∃ a_min a_max : ℝ, ∀ a : ℝ, a_min ≤ a ∧ a ≤ a_max →
    ∀ x : ℝ, f x a ≥ 3 :=
sorry

end exists_solution_set_exists_a_range_l439_43925


namespace vector_AB_and_magnitude_l439_43975

def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, 3)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

theorem vector_AB_and_magnitude :
  vector_AB = (1, 1) ∧ 
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = Real.sqrt 2 := by
  sorry

end vector_AB_and_magnitude_l439_43975


namespace smallest_m_dividing_power_minus_one_l439_43951

theorem smallest_m_dividing_power_minus_one :
  ∃ (m : ℕ+), (2^1990 : ℕ) ∣ (1989^(m : ℕ) - 1) ∧
    ∀ (k : ℕ+), (2^1990 : ℕ) ∣ (1989^(k : ℕ) - 1) → m ≤ k :=
by
  use 2^1988
  sorry

end smallest_m_dividing_power_minus_one_l439_43951


namespace find_multiplier_l439_43966

theorem find_multiplier : ∃ (m : ℕ), 
  220050 = m * (555 - 445) * (555 + 445) + 50 ∧ 
  m * (555 - 445) = 220050 / (555 + 445) :=
by sorry

end find_multiplier_l439_43966


namespace angle_T_measure_l439_43950

/-- Represents a heptagon with specific angle properties -/
structure Heptagon :=
  (G E O M Y J R T : ℝ)
  (sum_angles : G + E + O + M + Y + J + R + T = 900)
  (equal_angles : G = E ∧ E = T ∧ T = R)
  (supplementary_M_Y : M + Y = 180)
  (supplementary_J_O : J + O = 180)

/-- The measure of angle T in the specified heptagon is 135° -/
theorem angle_T_measure (h : Heptagon) : h.T = 135 := by
  sorry

end angle_T_measure_l439_43950


namespace parabola_vertex_l439_43968

-- Define the parabola function
def f (x : ℝ) : ℝ := 3 * (x + 4)^2 - 9

-- State the theorem
theorem parabola_vertex :
  ∃ (x y : ℝ), (∀ t : ℝ, f t ≥ f x) ∧ f x = y ∧ x = -4 ∧ y = -9 := by
  sorry

end parabola_vertex_l439_43968


namespace mary_cut_ten_roses_l439_43970

/-- The number of roses Mary cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Proof that Mary cut 10 roses from her garden -/
theorem mary_cut_ten_roses (initial_roses final_roses : ℕ)
  (h1 : initial_roses = 6)
  (h2 : final_roses = 16) :
  roses_cut initial_roses final_roses = 10 := by
  sorry

end mary_cut_ten_roses_l439_43970


namespace john_racecar_earnings_l439_43933

/-- The amount of money John made from his racecar after one race -/
def money_made (initial_cost maintenance_cost : ℝ) (discount prize_percent : ℝ) (prize : ℝ) : ℝ :=
  prize * prize_percent - initial_cost * (1 - discount) - maintenance_cost

/-- Theorem stating the amount John made from his racecar -/
theorem john_racecar_earnings (x : ℝ) :
  money_made 20000 x 0.2 0.9 70000 = 47000 - x := by
  sorry

end john_racecar_earnings_l439_43933


namespace angle_FAG_measure_l439_43916

-- Define the structure of the geometric configuration
structure GeometricConfiguration where
  -- Triangle ABC is equilateral
  triangle_ABC_equilateral : Bool
  -- BCDFG is a regular pentagon
  pentagon_BCDFG_regular : Bool
  -- Triangle ABC and pentagon BCDFG share side BC
  shared_side_BC : Bool

-- Define the theorem
theorem angle_FAG_measure (config : GeometricConfiguration) 
  (h1 : config.triangle_ABC_equilateral = true)
  (h2 : config.pentagon_BCDFG_regular = true)
  (h3 : config.shared_side_BC = true) :
  ∃ (angle_FAG : ℝ), angle_FAG = 36 := by
  sorry

end angle_FAG_measure_l439_43916


namespace max_rectangle_area_l439_43940

def is_valid_rectangle (l w : ℕ) : Prop :=
  l + w = 20 ∧ l ≥ w + 3

def rectangle_area (l w : ℕ) : ℕ :=
  l * w

theorem max_rectangle_area :
  ∃ (l w : ℕ), is_valid_rectangle l w ∧
    rectangle_area l w = 91 ∧
    ∀ (l' w' : ℕ), is_valid_rectangle l' w' →
      rectangle_area l' w' ≤ 91 := by
  sorry

end max_rectangle_area_l439_43940


namespace min_value_of_function_equality_condition_l439_43934

theorem min_value_of_function (x : ℝ) (h : x > 2) : x + 4 / (x - 2) ≥ 6 := by
  sorry

theorem equality_condition : ∃ x > 2, x + 4 / (x - 2) = 6 := by
  sorry

end min_value_of_function_equality_condition_l439_43934


namespace scooter_gain_percent_l439_43926

/-- Calculate the gain percent on a scooter sale given the purchase price, repair costs, and selling price. -/
theorem scooter_gain_percent 
  (purchase_price : ℝ)
  (repair_costs : ℝ)
  (selling_price : ℝ)
  (h1 : purchase_price = 800)
  (h2 : repair_costs = 200)
  (h3 : selling_price = 1400) :
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 40 := by
sorry


end scooter_gain_percent_l439_43926


namespace balloon_arrangements_count_l439_43928

/-- The number of distinct arrangements of letters in "balloon" -/
def balloon_arrangements : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating the number of distinct arrangements of letters in "balloon" -/
theorem balloon_arrangements_count : balloon_arrangements = 1260 := by
  sorry

end balloon_arrangements_count_l439_43928


namespace hollow_square_students_l439_43988

/-- Represents a hollow square formation of students -/
structure HollowSquare where
  outer_layer : Nat
  inner_layer : Nat

/-- Calculates the total number of students in a hollow square formation -/
def total_students (hs : HollowSquare) : Nat :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that a hollow square with 52 in the outer layer and 28 in the inner layer has 160 students total -/
theorem hollow_square_students :
  let hs : HollowSquare := { outer_layer := 52, inner_layer := 28 }
  total_students hs = 160 := by
  sorry

end hollow_square_students_l439_43988


namespace binary_110101_equals_53_l439_43912

def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldr (λ (i, b) acc => acc + if b then 2^i else 0) 0

theorem binary_110101_equals_53 :
  binary_to_decimal [true, false, true, false, true, true] = 53 := by
  sorry

end binary_110101_equals_53_l439_43912


namespace memory_card_capacity_l439_43986

/-- Proves that a memory card with capacity for 3,000 pictures of 8 megabytes
    can hold 4,000 pictures of 6 megabytes -/
theorem memory_card_capacity 
  (initial_count : Nat) 
  (initial_size : Nat) 
  (new_size : Nat) 
  (h1 : initial_count = 3000)
  (h2 : initial_size = 8)
  (h3 : new_size = 6) :
  (initial_count * initial_size) / new_size = 4000 :=
by
  sorry

end memory_card_capacity_l439_43986


namespace boys_in_class_l439_43992

theorem boys_in_class (total : ℕ) (girls_fraction : ℚ) (boys : ℕ) : 
  total = 160 → girls_fraction = 1 / 4 → boys = 120 → 
  boys = total * (1 - girls_fraction) := by
sorry

end boys_in_class_l439_43992


namespace solve_equation_l439_43909

theorem solve_equation : ∃ y : ℚ, 2 * y + 3 * y = 500 - (4 * y + 6 * y) → y = 100 / 3 := by
  sorry

end solve_equation_l439_43909


namespace lcm_product_hcf_l439_43924

theorem lcm_product_hcf (x y : ℕ+) : 
  Nat.lcm x y = 560 → x * y = 42000 → Nat.gcd x y = 75 := by
  sorry

end lcm_product_hcf_l439_43924


namespace gmat_question_percentages_l439_43917

/-- Given percentages of test takers answering questions correctly or incorrectly, 
    prove the percentage that answered both questions correctly. -/
theorem gmat_question_percentages 
  (first_correct : ℝ) 
  (second_correct : ℝ) 
  (neither_correct : ℝ) 
  (h1 : first_correct = 85) 
  (h2 : second_correct = 70) 
  (h3 : neither_correct = 5) : 
  first_correct + second_correct - (100 - neither_correct) = 60 := by
  sorry


end gmat_question_percentages_l439_43917


namespace jelly_beans_solution_l439_43941

/-- The number of jelly beans in jar Y -/
def jelly_beans_Y : ℕ := sorry

/-- The number of jelly beans in jar X -/
def jelly_beans_X : ℕ := 3 * jelly_beans_Y - 400

/-- The total number of jelly beans -/
def total_jelly_beans : ℕ := 1200

theorem jelly_beans_solution :
  jelly_beans_X + jelly_beans_Y = total_jelly_beans ∧ jelly_beans_Y = 400 := by sorry

end jelly_beans_solution_l439_43941


namespace smallest_sum_a_b_l439_43952

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem smallest_sum_a_b (a b : ℕ) 
  (h1 : a^a % b^b = 0)
  (h2 : ¬(a % b = 0))
  (h3 : is_coprime b 210) :
  (∀ (x y : ℕ), x^x % y^y = 0 → ¬(x % y = 0) → is_coprime y 210 → a + b ≤ x + y) →
  a + b = 374 := by
sorry

end smallest_sum_a_b_l439_43952


namespace subtract_square_equals_two_square_l439_43910

theorem subtract_square_equals_two_square (x : ℝ) : 3 * x^2 - x^2 = 2 * x^2 := by
  sorry

end subtract_square_equals_two_square_l439_43910


namespace cube_construction_count_l439_43990

/-- Represents the group of rotations for a 3x3x3 cube -/
def CubeRotations : Type := Unit

/-- The number of elements in the group of rotations for a 3x3x3 cube -/
def rotationGroupSize : ℕ := 27

/-- The total number of ways to arrange 13 white cubes in a 3x3x3 cube -/
def totalArrangements : ℕ := 10400600

/-- The estimated number of fixed points for non-identity rotations -/
def fixedPointsNonIdentity : ℕ := 1000

/-- The total number of fixed points across all rotations -/
def totalFixedPoints : ℕ := totalArrangements + fixedPointsNonIdentity

/-- The number of distinct ways to construct the 3x3x3 cube -/
def distinctConstructions : ℕ := totalFixedPoints / rotationGroupSize

theorem cube_construction_count :
  distinctConstructions = 385244 := by sorry

end cube_construction_count_l439_43990


namespace sum_of_fractions_l439_43996

theorem sum_of_fractions (A B : ℕ) (h : (A : ℚ) / 11 + (B : ℚ) / 3 = 17 / 33) : A + B = 3 := by
  sorry

end sum_of_fractions_l439_43996


namespace min_w_for_max_sin_l439_43943

theorem min_w_for_max_sin (y : ℝ → ℝ) (w : ℝ) : 
  (∀ x, y x = Real.sin (w * x)) →  -- Condition 1
  w > 0 →  -- Condition 2
  (∃ n : ℕ, n ≥ 50 ∧ ∀ i : ℕ, i < n → ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ y x = 1) →  -- Condition 3
  w ≥ Real.pi * 100 :=  -- Conclusion
by sorry

end min_w_for_max_sin_l439_43943


namespace range_of_absolute_value_sum_l439_43995

theorem range_of_absolute_value_sum (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  ∀ x : ℝ, |x - a| + |x - b| = a - b ↔ b ≤ x ∧ x ≤ a :=
by sorry

end range_of_absolute_value_sum_l439_43995


namespace problem_solution_l439_43936

theorem problem_solution (P Q R : ℚ) : 
  (5 / 8 = P / 56) → 
  (5 / 8 = 80 / Q) → 
  (R = P - 4) → 
  (Q + R = 159) := by
sorry

end problem_solution_l439_43936


namespace floor_abs_negative_real_l439_43918

theorem floor_abs_negative_real : ⌊|(-56.7 : ℝ)|⌋ = 56 := by sorry

end floor_abs_negative_real_l439_43918


namespace min_value_trig_expression_min_value_achievable_l439_43913

theorem min_value_trig_expression (θ : Real) :
  (1 / (2 - Real.cos θ ^ 2)) + (1 / (2 - Real.sin θ ^ 2)) ≥ 4 / 3 :=
sorry

theorem min_value_achievable :
  ∃ θ : Real, (1 / (2 - Real.cos θ ^ 2)) + (1 / (2 - Real.sin θ ^ 2)) = 4 / 3 :=
sorry

end min_value_trig_expression_min_value_achievable_l439_43913


namespace jason_grew_37_watermelons_l439_43980

/-- The number of watermelons Jason grew -/
def jason_watermelons : ℕ := 37

/-- The number of watermelons Sandy grew -/
def sandy_watermelons : ℕ := 11

/-- The total number of watermelons grown by Jason and Sandy -/
def total_watermelons : ℕ := 48

/-- Theorem stating that Jason grew 37 watermelons -/
theorem jason_grew_37_watermelons : jason_watermelons = total_watermelons - sandy_watermelons := by
  sorry

end jason_grew_37_watermelons_l439_43980


namespace arithmetic_equality_l439_43948

theorem arithmetic_equality : 2021 - 2223 + 2425 = 2223 := by
  sorry

end arithmetic_equality_l439_43948


namespace odometer_sum_squares_l439_43964

/-- Represents the odometer reading as a three-digit number -/
structure OdometerReading where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≠ 0 ∧ hundreds + tens + ones = 7

/-- Represents a car journey -/
structure CarJourney where
  duration : Nat
  average_speed : Nat
  initial_reading : OdometerReading
  final_reading : OdometerReading
  speed_constraint : average_speed = 60
  odometer_constraint : final_reading.hundreds = initial_reading.ones ∧
                        final_reading.tens = initial_reading.tens ∧
                        final_reading.ones = initial_reading.hundreds

theorem odometer_sum_squares (journey : CarJourney) :
  journey.initial_reading.hundreds ^ 2 +
  journey.initial_reading.tens ^ 2 +
  journey.initial_reading.ones ^ 2 = 37 := by
  sorry

end odometer_sum_squares_l439_43964


namespace parallel_lines_a_value_l439_43976

/-- Two lines are parallel if their normal vectors are proportional -/
def parallel (a b c d e f : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ a = k * d ∧ b = k * e

/-- Two lines coincide if their coefficients are proportional -/
def coincide (a b c d e f : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ a = k * d ∧ b = k * e ∧ c = k * f

theorem parallel_lines_a_value (a : ℝ) :
  parallel a (a + 2) 2 1 a 1 ∧ ¬ coincide a (a + 2) 2 1 a 1 → a = -1 := by
  sorry

end parallel_lines_a_value_l439_43976


namespace all_propositions_true_l439_43904

-- Define the original proposition
def original_proposition (x : ℝ) : Prop :=
  (x = 2 ∨ x = -3) → (x - 2) * (x + 3) = 0

-- Define the converse
def converse (x : ℝ) : Prop :=
  (x - 2) * (x + 3) = 0 → (x = 2 ∨ x = -3)

-- Define the inverse
def inverse (x : ℝ) : Prop :=
  (x ≠ 2 ∧ x ≠ -3) → (x - 2) * (x + 3) ≠ 0

-- Define the contrapositive
def contrapositive (x : ℝ) : Prop :=
  (x - 2) * (x + 3) ≠ 0 → (x ≠ 2 ∧ x ≠ -3)

-- Theorem stating that all propositions are true for all real numbers
theorem all_propositions_true :
  ∀ x : ℝ, original_proposition x ∧ converse x ∧ inverse x ∧ contrapositive x :=
by sorry


end all_propositions_true_l439_43904


namespace sum_of_squares_even_2_to_14_l439_43920

def evenSquareSum : ℕ → ℕ
| 0 => 0
| n + 1 => if n + 1 ≤ 7 ∧ 2 * (n + 1) ≤ 14 then (2 * (n + 1))^2 + evenSquareSum n else evenSquareSum n

theorem sum_of_squares_even_2_to_14 : evenSquareSum 7 = 560 := by
  sorry

end sum_of_squares_even_2_to_14_l439_43920


namespace jane_hiking_distance_l439_43969

/-- The distance between two points given a specific path --/
theorem jane_hiking_distance (A B D : ℝ × ℝ) : 
  (A.1 = B.1 ∧ A.2 + 3 = B.2) →  -- AB is 3 units northward
  (Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = 8) →  -- BD is 8 units long
  (D.1 - B.1 = D.2 - B.2) →  -- 45 degree angle (isosceles right triangle)
  Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = Real.sqrt (73 + 24 * Real.sqrt 2) :=
by sorry

end jane_hiking_distance_l439_43969


namespace set_operations_l439_43961

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | (x - 2) * (x + 5) < 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {x | -5 < x ∧ x ≤ -1}) ∧
  (A ∪ (U \ B) = {x | -5 < x ∧ x < 3}) := by
  sorry

end set_operations_l439_43961


namespace uno_card_discount_l439_43927

theorem uno_card_discount (original_price : ℝ) (num_cards : ℕ) (total_paid : ℝ) : 
  original_price = 12 → num_cards = 10 → total_paid = 100 → 
  (original_price * num_cards - total_paid) / num_cards = 2 := by
  sorry

end uno_card_discount_l439_43927


namespace fifth_power_sum_l439_43983

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := by
sorry

end fifth_power_sum_l439_43983


namespace potato_slab_length_l439_43981

/-- The length of the original uncut potato slab given the lengths of its two pieces -/
theorem potato_slab_length 
  (piece1 : ℕ) 
  (piece2 : ℕ) 
  (h1 : piece1 = 275)
  (h2 : piece2 = piece1 + 50) : 
  piece1 + piece2 = 600 := by
  sorry

end potato_slab_length_l439_43981


namespace bridge_length_is_954_l439_43935

/-- The length of a bridge given train parameters -/
def bridge_length (train_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) : ℝ :=
  train_speed * crossing_time - train_length

/-- Theorem: The length of the bridge is 954 meters -/
theorem bridge_length_is_954 :
  bridge_length 90 36 29 = 954 := by
  sorry

end bridge_length_is_954_l439_43935


namespace g_of_3_l439_43937

theorem g_of_3 (g : ℝ → ℝ) :
  (∀ x, g x = (x^2 + 1) / (4*x - 5)) →
  g 3 = 10/7 := by
sorry

end g_of_3_l439_43937


namespace angle_inequality_l439_43972

theorem angle_inequality (x y z : Real) 
  (h1 : 0 < x ∧ x < π/2)
  (h2 : 0 < y ∧ y < π/2)
  (h3 : 0 < z ∧ z < π/2)
  (h4 : (Real.sin x + Real.cos x) * (Real.sin y + 2 * Real.cos y) * (Real.sin z + 3 * Real.cos z) = 10) :
  x = π/4 ∧ x > y ∧ y > z := by sorry

end angle_inequality_l439_43972


namespace union_contains_1980_l439_43963

/-- An arithmetic progression of integers -/
def ArithmeticProgression (a₀ d : ℤ) : Set ℤ :=
  {n : ℤ | ∃ k : ℕ, n = a₀ + k * d}

theorem union_contains_1980
  (A B C : Set ℤ)
  (hA : ∃ a₀ d : ℤ, A = ArithmeticProgression a₀ d)
  (hB : ∃ a₀ d : ℤ, B = ArithmeticProgression a₀ d)
  (hC : ∃ a₀ d : ℤ, C = ArithmeticProgression a₀ d)
  (h_union : {1, 2, 3, 4, 5, 6, 7, 8} ⊆ A ∪ B ∪ C) :
  1980 ∈ A ∪ B ∪ C :=
sorry

end union_contains_1980_l439_43963


namespace barry_sotter_magic_l439_43956

theorem barry_sotter_magic (n : ℕ) : (n + 3 : ℚ) / 3 = 50 ↔ n = 147 := by sorry

end barry_sotter_magic_l439_43956


namespace specific_cube_stack_surface_area_l439_43994

/-- Represents a three-dimensional shape formed by stacking cubes -/
structure CubeStack where
  num_cubes : ℕ
  edge_length : ℝ
  num_layers : ℕ

/-- Calculates the surface area of a cube stack -/
def surface_area (stack : CubeStack) : ℝ :=
  sorry

/-- Theorem stating that a specific cube stack has a surface area of 72 square meters -/
theorem specific_cube_stack_surface_area :
  let stack : CubeStack := {
    num_cubes := 30,
    edge_length := 1,
    num_layers := 4
  }
  surface_area stack = 72 := by
  sorry

end specific_cube_stack_surface_area_l439_43994


namespace square_root_problem_l439_43900

theorem square_root_problem (a b : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ (a + 3)^2 = x ∧ (2*a - 6)^2 = x) →
  ((-2)^3 = b) →
  (∃ (y : ℝ), y^2 = a - b ∧ (y = 3 ∨ y = -3)) :=
by sorry

end square_root_problem_l439_43900


namespace candy_distribution_l439_43997

theorem candy_distribution (a b d : ℕ) : 
  (4 * b = 3 * a) →  -- While Andrey eats 4 candies, Boris eats 3
  (6 * d = 7 * a) →  -- While Andrey eats 6 candies, Denis eats 7
  (a + b + d = 70) → -- Total candies eaten
  (a = 24 ∧ b = 18 ∧ d = 28) := by sorry

end candy_distribution_l439_43997


namespace reflection_over_y_axis_l439_43967

theorem reflection_over_y_axis :
  let reflect_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![-1, 0], ![0, 1]]
  ∀ (x y : ℝ), 
    reflect_matrix.mulVec ![x, y] = ![-x, y] := by sorry

end reflection_over_y_axis_l439_43967


namespace solve_for_y_l439_43944

theorem solve_for_y (x y n : ℝ) (h : x ≠ y) (h_n : n = (3 * x * y) / (x - y)) :
  y = (n * x) / (3 * x + n) := by
  sorry

end solve_for_y_l439_43944


namespace goods_train_length_l439_43977

/-- The length of a goods train passing a man in another train --/
theorem goods_train_length (man_train_speed goods_train_speed : ℝ) 
  (passing_time : ℝ) : 
  man_train_speed = 36 →
  goods_train_speed = 50.4 →
  passing_time = 10 →
  (man_train_speed + goods_train_speed) * (1000 / 3600) * passing_time = 1200 :=
by
  sorry

end goods_train_length_l439_43977


namespace perpendicular_travel_time_l439_43982

theorem perpendicular_travel_time 
  (adam_speed : ℝ) 
  (simon_speed : ℝ) 
  (distance : ℝ) 
  (h1 : adam_speed = 10)
  (h2 : simon_speed = 5)
  (h3 : distance = 75) :
  ∃ (time : ℝ), 
    time = 3 * Real.sqrt 5 ∧ 
    distance^2 = (adam_speed * time)^2 + (simon_speed * time)^2 := by
  sorry

end perpendicular_travel_time_l439_43982


namespace ski_and_snowboard_intersection_l439_43923

theorem ski_and_snowboard_intersection (total : ℕ) (ski : ℕ) (snowboard : ℕ) (neither : ℕ)
  (h_total : total = 20)
  (h_ski : ski = 11)
  (h_snowboard : snowboard = 13)
  (h_neither : neither = 3) :
  ski + snowboard - (total - neither) = 7 :=
by sorry

end ski_and_snowboard_intersection_l439_43923


namespace range_of_g_l439_43965

noncomputable def g (x : ℝ) : ℝ := 1 / x^2 + 3

theorem range_of_g :
  Set.range g = Set.Ici 3 :=
sorry

end range_of_g_l439_43965


namespace min_value_reciprocal_sum_l439_43960

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end min_value_reciprocal_sum_l439_43960


namespace rook_placement_count_l439_43991

theorem rook_placement_count (n k : ℕ) (h1 : n = 8) (h2 : k = 6) :
  (Nat.choose n k)^2 * Nat.factorial k = 564480 := by
  sorry

end rook_placement_count_l439_43991


namespace cube_volume_proof_l439_43993

theorem cube_volume_proof (n : ℕ) (m : ℕ) : 
  (n^3 = 98 + m^3) ∧ 
  (m ≠ 1) ∧ 
  (∃ (k : ℕ), n^3 = 99 * k) →
  n^3 = 125 := by
sorry

end cube_volume_proof_l439_43993


namespace lemonade_pitchers_l439_43906

/-- Represents the number of glasses a pitcher can serve -/
def glasses_per_pitcher : ℕ := 5

/-- Represents the total number of glasses served -/
def total_glasses_served : ℕ := 30

/-- Calculates the number of pitchers needed -/
def pitchers_needed : ℕ := total_glasses_served / glasses_per_pitcher

theorem lemonade_pitchers : pitchers_needed = 6 := by
  sorry

end lemonade_pitchers_l439_43906


namespace quadratic_equation_solution_l439_43907

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 + 6*x₁ - 7 = 0) ∧ 
  (x₂^2 + 6*x₂ - 7 = 0) ∧ 
  x₁ = -7 ∧ 
  x₂ = 1 :=
by
  sorry

#check quadratic_equation_solution

end quadratic_equation_solution_l439_43907


namespace intersection_A_B_l439_43921

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (4 - x)}

-- Define set B
def B : Set ℝ := {x | x - 1 > 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x ≤ 4} := by sorry

end intersection_A_B_l439_43921


namespace min_value_theorem_l439_43974

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 2 / (3 * a + b) + 1 / (a + 2 * b) = 4) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 / (3 * x + y) + 1 / (x + 2 * y) = 4 → 7 * x + 4 * y ≥ 9/4 :=
by sorry

end min_value_theorem_l439_43974


namespace prime_power_equation_solutions_l439_43946

theorem prime_power_equation_solutions :
  ∀ (p x y : ℕ),
    Prime p →
    x > 0 →
    y > 0 →
    p^x = y^3 + 1 →
    ((p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2)) :=
by sorry

end prime_power_equation_solutions_l439_43946


namespace trig_identity_l439_43908

theorem trig_identity (α : Real) (h : Real.tan α = 2) :
  7 * Real.sin α ^ 2 + 3 * Real.cos α ^ 2 = 31 / 5 := by
  sorry

end trig_identity_l439_43908


namespace ant_meeting_point_l439_43901

/-- Triangle XYZ with given side lengths -/
structure Triangle where
  xy : ℝ
  yz : ℝ
  xz : ℝ

/-- Point P where ants meet -/
def MeetingPoint (t : Triangle) : ℝ := sorry

/-- Theorem stating that YP = 5 in the given triangle -/
theorem ant_meeting_point (t : Triangle) 
  (h_xy : t.xy = 5) 
  (h_yz : t.yz = 7) 
  (h_xz : t.xz = 8) : 
  MeetingPoint t = 5 := by sorry

end ant_meeting_point_l439_43901


namespace not_right_triangle_l439_43947

theorem not_right_triangle (A B C : ℝ) (h1 : A = B) (h2 : A = 3 * C) 
  (h3 : A + B + C = 180) : A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

end not_right_triangle_l439_43947


namespace triangle_area_l439_43954

/-- Given a triangle with perimeter 28 and inradius 2.5, prove that its area is 35 -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
    (h1 : perimeter = 28) 
    (h2 : inradius = 2.5) 
    (h3 : area = inradius * (perimeter / 2)) : area = 35 := by
  sorry

end triangle_area_l439_43954


namespace third_term_expansion_l439_43984

-- Define i as the imaginary unit
axiom i : ℂ
axiom i_squared : i * i = -1

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem third_term_expansion :
  let n : ℕ := 6
  let r : ℕ := 2
  (binomial n r : ℂ) * (1 : ℂ)^(n - r) * i^r = -15 := by sorry

end third_term_expansion_l439_43984


namespace triangle_properties_l439_43939

theorem triangle_properties (a b c : ℝ) (h_ratio : (a, b, c) = (3, 4, 5)) 
  (h_perimeter : a + b + c = 36) : 
  (a^2 + b^2 = c^2) ∧ (a * b / 2 = 54) := by
  sorry

end triangle_properties_l439_43939


namespace resort_tips_fraction_l439_43945

theorem resort_tips_fraction (average_tips : ℝ) (h : average_tips > 0) :
  let other_months_total := 6 * average_tips
  let august_tips := 6 * average_tips
  let total_tips := other_months_total + august_tips
  august_tips / total_tips = 1 / 2 := by
sorry

end resort_tips_fraction_l439_43945


namespace johns_initial_marbles_l439_43905

/-- Given that:
    - Ben had 18 marbles initially
    - John had an unknown number of marbles initially
    - Ben gave half of his marbles to John
    - After the transfer, John had 17 more marbles than Ben
    Prove that John had 17 marbles initially -/
theorem johns_initial_marbles :
  ∀ (john_initial : ℕ),
  let ben_initial : ℕ := 18
  let ben_gave : ℕ := ben_initial / 2
  let ben_final : ℕ := ben_initial - ben_gave
  let john_final : ℕ := john_initial + ben_gave
  john_final = ben_final + 17 →
  john_initial = 17 := by
sorry

end johns_initial_marbles_l439_43905
