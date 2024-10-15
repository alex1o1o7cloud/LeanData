import Mathlib

namespace NUMINAMATH_CALUDE_new_person_weight_l4060_406020

/-- Given 4 people, with one weighing 95 kg, if the average weight increases by 8.5 kg
    when a new person replaces the 95 kg person, then the new person weighs 129 kg. -/
theorem new_person_weight (initial_count : Nat) (replaced_weight : Real) (avg_increase : Real) :
  initial_count = 4 →
  replaced_weight = 95 →
  avg_increase = 8.5 →
  (initial_count : Real) * avg_increase + replaced_weight = 129 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l4060_406020


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l4060_406080

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 2*a - 3 = 0) → (b^2 - 2*b - 3 = 0) → (a ≠ b) → a^2 + b^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l4060_406080


namespace NUMINAMATH_CALUDE_james_barbell_cost_l4060_406078

/-- The final cost of James' new barbell purchase -/
def final_barbell_cost (old_barbell_cost : ℝ) (price_increase_rate : ℝ) 
  (sales_tax_rate : ℝ) (trade_in_value : ℝ) : ℝ :=
  let new_barbell_cost := old_barbell_cost * (1 + price_increase_rate)
  let total_cost_with_tax := new_barbell_cost * (1 + sales_tax_rate)
  total_cost_with_tax - trade_in_value

/-- Theorem stating the final cost of James' new barbell -/
theorem james_barbell_cost : 
  final_barbell_cost 250 0.30 0.10 100 = 257.50 := by
  sorry

end NUMINAMATH_CALUDE_james_barbell_cost_l4060_406078


namespace NUMINAMATH_CALUDE_base7_to_base10_54231_l4060_406098

/-- Converts a base 7 number to base 10 -/
def base7_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- The base 7 representation of the number -/
def base7_num : List Nat := [1, 3, 2, 4, 5]

theorem base7_to_base10_54231 :
  base7_to_base10 base7_num = 13497 := by sorry

end NUMINAMATH_CALUDE_base7_to_base10_54231_l4060_406098


namespace NUMINAMATH_CALUDE_divisor_proof_l4060_406042

theorem divisor_proof (dividend quotient remainder divisor : ℤ) : 
  dividend = 474232 →
  quotient = 594 →
  remainder = -968 →
  dividend = divisor * quotient + remainder →
  divisor = 800 := by
sorry

end NUMINAMATH_CALUDE_divisor_proof_l4060_406042


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_3913_l4060_406073

theorem largest_prime_factor_of_3913 :
  ∃ (p : ℕ), p.Prime ∧ p ∣ 3913 ∧ ∀ (q : ℕ), q.Prime → q ∣ 3913 → q ≤ p ∧ p = 23 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_3913_l4060_406073


namespace NUMINAMATH_CALUDE_binary_subtraction_result_l4060_406025

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 111111111₂ -/
def binary_111111111 : List Bool := [true, true, true, true, true, true, true, true, true]

/-- The binary representation of 111111₂ -/
def binary_111111 : List Bool := [true, true, true, true, true, true]

/-- The theorem stating that the difference between the decimal representations
    of 111111111₂ and 111111₂ is equal to 448 -/
theorem binary_subtraction_result :
  binary_to_decimal binary_111111111 - binary_to_decimal binary_111111 = 448 := by
  sorry

end NUMINAMATH_CALUDE_binary_subtraction_result_l4060_406025


namespace NUMINAMATH_CALUDE_simple_interest_problem_l4060_406060

/-- Proves that given a sum P at simple interest for 10 years, 
    if increasing the interest rate by 3% results in $300 more interest, 
    then P = $1000. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 3) * 10 / 100 - P * R * 10 / 100 = 300) → P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l4060_406060


namespace NUMINAMATH_CALUDE_limit_fraction_powers_three_five_l4060_406089

/-- The limit of (3^n + 5^n) / (3^(n-1) + 5^(n-1)) as n approaches infinity is 5 -/
theorem limit_fraction_powers_three_five :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |((3 : ℝ)^n + 5^n) / ((3 : ℝ)^(n-1) + 5^(n-1)) - 5| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_fraction_powers_three_five_l4060_406089


namespace NUMINAMATH_CALUDE_ivy_room_spiders_l4060_406028

/-- Given the total number of spider legs in a room, calculate the number of spiders. -/
def spiders_in_room (total_legs : ℕ) : ℕ :=
  total_legs / 8

/-- Theorem: There are 4 spiders in Ivy's room given 32 total spider legs. -/
theorem ivy_room_spiders : spiders_in_room 32 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ivy_room_spiders_l4060_406028


namespace NUMINAMATH_CALUDE_two_black_cards_selection_l4060_406045

/-- The number of cards in each suit of a standard deck -/
def cards_per_suit : ℕ := 13

/-- The number of black suits in a standard deck -/
def black_suits : ℕ := 2

/-- The total number of black cards in a standard deck -/
def total_black_cards : ℕ := black_suits * cards_per_suit

/-- The number of ways to select two different black cards from a standard deck, where order matters -/
def ways_to_select_two_black_cards : ℕ := total_black_cards * (total_black_cards - 1)

theorem two_black_cards_selection :
  ways_to_select_two_black_cards = 650 := by
  sorry

end NUMINAMATH_CALUDE_two_black_cards_selection_l4060_406045


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l4060_406014

theorem cubic_equation_solution (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  (∃ a b : ℝ, ∀ x : ℝ, x = a * m + b * n → (x + m)^3 - (x + n)^3 = (m - n)^3) ↔
  (∀ x : ℝ, (x + m)^3 - (x + n)^3 = (m - n)^3 ↔ x = -m + n) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l4060_406014


namespace NUMINAMATH_CALUDE_roshesmina_pennies_theorem_l4060_406057

/-- Represents a piggy bank with a given number of compartments and pennies per compartment -/
structure PiggyBank where
  compartments : Nat
  penniesPerCompartment : Nat

/-- Calculates the total number of pennies in a piggy bank -/
def totalPennies (pb : PiggyBank) : Nat :=
  pb.compartments * pb.penniesPerCompartment

/-- Represents Roshesmina's piggy bank -/
def roshesminaBank : PiggyBank :=
  { compartments := 12,
    penniesPerCompartment := 2 }

/-- Adds a specified number of pennies to each compartment of a piggy bank -/
def addPennies (pb : PiggyBank) (amount : Nat) : PiggyBank :=
  { compartments := pb.compartments,
    penniesPerCompartment := pb.penniesPerCompartment + amount }

/-- Theorem stating that after adding 6 pennies to each compartment of Roshesmina's piggy bank, 
    the total number of pennies is 96 -/
theorem roshesmina_pennies_theorem :
  totalPennies (addPennies roshesminaBank 6) = 96 := by
  sorry

end NUMINAMATH_CALUDE_roshesmina_pennies_theorem_l4060_406057


namespace NUMINAMATH_CALUDE_exam_results_l4060_406039

theorem exam_results (total : ℕ) (failed_hindi : ℕ) (failed_english : ℕ) (failed_both : ℕ)
  (h1 : failed_hindi = total / 4)
  (h2 : failed_english = total / 2)
  (h3 : failed_both = total / 4)
  : (total - (failed_hindi + failed_english - failed_both)) = total / 2 := by
  sorry

end NUMINAMATH_CALUDE_exam_results_l4060_406039


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4060_406081

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the line
def line (x y : ℝ) : Prop :=
  y = 2*x - 4

-- Define the right focus
def right_focus (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 = a^2 + b^2 ∧ x > 0 ∧ y = 0

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ (x y : ℝ), right_focus a b x y ∧ line x y) →
  (∃! (p : ℝ × ℝ), hyperbola a b p.1 p.2 ∧ line p.1 p.2) →
  (∀ (x y : ℝ), hyperbola a b x y ↔ 5*x^2/4 - 5*y^2/16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4060_406081


namespace NUMINAMATH_CALUDE_comic_books_count_l4060_406088

theorem comic_books_count (total : ℕ) 
  (h1 : (30 : ℚ) / 100 * total = (total - (70 : ℚ) / 100 * total))
  (h2 : (70 : ℚ) / 100 * total ≥ 120)
  (h3 : ∀ n : ℕ, n < total → (70 : ℚ) / 100 * n < 120) : 
  total = 172 := by
sorry

end NUMINAMATH_CALUDE_comic_books_count_l4060_406088


namespace NUMINAMATH_CALUDE_min_value_theorem_l4060_406084

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3) :
  (4 / x + 1 / (y + 1) ≥ 9 / 4) ∧
  (4 / x + 1 / (y + 1) = 9 / 4 ↔ x = 2 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4060_406084


namespace NUMINAMATH_CALUDE_vertex_locus_is_parabola_l4060_406019

/-- The locus of vertices of a family of parabolas forms another parabola -/
theorem vertex_locus_is_parabola (a c : ℝ) (ha : a > 0) (hc : c > 0) :
  ∃ (A B C : ℝ), A ≠ 0 ∧
    ∀ (x y : ℝ), (∃ t : ℝ, x = -t / (2 * a) ∧ y = c - t^2 / (4 * a)) ↔
      y = A * x^2 + B * x + C :=
by sorry

end NUMINAMATH_CALUDE_vertex_locus_is_parabola_l4060_406019


namespace NUMINAMATH_CALUDE_simplify_logarithmic_expression_l4060_406087

theorem simplify_logarithmic_expression (x : Real) (h : 0 < x ∧ x < Real.pi / 2) :
  Real.log (Real.cos x * Real.tan x + 1 - 2 * Real.sin (x / 2) ^ 2) +
  Real.log (Real.sqrt 2 * Real.cos (x - Real.pi / 4)) -
  Real.log (1 + Real.sin (2 * x)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_logarithmic_expression_l4060_406087


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l4060_406038

def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)
def a : ℝ × ℝ := (2 * i.1 + 0 * i.2, 0 * j.1 + 3 * j.2)
def b (k : ℝ) : ℝ × ℝ := (k * i.1 + 0 * i.2, 0 * j.1 - 4 * j.2)

theorem perpendicular_vectors_k_value :
  ∀ k : ℝ, (a.1 * (b k).1 + a.2 * (b k).2 = 0) → k = 6 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l4060_406038


namespace NUMINAMATH_CALUDE_x_value_is_36_l4060_406050

theorem x_value_is_36 (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 110) : x = 36 := by
  sorry

end NUMINAMATH_CALUDE_x_value_is_36_l4060_406050


namespace NUMINAMATH_CALUDE_solution_set_absolute_value_inequality_l4060_406053

theorem solution_set_absolute_value_inequality :
  {x : ℝ | |2 - x| ≥ 1} = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_absolute_value_inequality_l4060_406053


namespace NUMINAMATH_CALUDE_constant_difference_expressions_l4060_406031

theorem constant_difference_expressions (x : ℤ) : 
  (∃ k : ℤ, (x^2 - 4*x + 5) - (2*x - 6) = k ∧ 
             (4*x - 8) - (x^2 - 4*x + 5) = k ∧ 
             (3*x^2 - 12*x + 11) - (4*x - 8) = k) ↔ 
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_constant_difference_expressions_l4060_406031


namespace NUMINAMATH_CALUDE_yellow_light_probability_is_one_twelfth_l4060_406036

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDuration where
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the probability of seeing the yellow light -/
def yellowLightProbability (d : TrafficLightDuration) : ℚ :=
  d.yellow / (d.red + d.green + d.yellow)

/-- Theorem stating the probability of seeing the yellow light is 1/12 -/
theorem yellow_light_probability_is_one_twelfth :
  let d : TrafficLightDuration := ⟨30, 25, 5⟩
  yellowLightProbability d = 1 / 12 := by
  sorry

#check yellow_light_probability_is_one_twelfth

end NUMINAMATH_CALUDE_yellow_light_probability_is_one_twelfth_l4060_406036


namespace NUMINAMATH_CALUDE_football_team_handedness_l4060_406058

theorem football_team_handedness (total_players : ℕ) (throwers : ℕ) (right_handed : ℕ)
  (h1 : total_players = 70)
  (h2 : throwers = 28)
  (h3 : right_handed = 56)
  (h4 : throwers ≤ right_handed) :
  (total_players - throwers - (right_handed - throwers)) / (total_players - throwers) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_football_team_handedness_l4060_406058


namespace NUMINAMATH_CALUDE_solutions_count_l4060_406076

theorem solutions_count : 
  (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ 3 * p.1 + p.2 = 100) 
    (Finset.product (Finset.range 101) (Finset.range 101))).card = 33 := by
  sorry

end NUMINAMATH_CALUDE_solutions_count_l4060_406076


namespace NUMINAMATH_CALUDE_businessmen_drinks_l4060_406009

theorem businessmen_drinks (total : ℕ) (coffee : ℕ) (tea : ℕ) (both : ℕ) :
  total = 30 →
  coffee = 15 →
  tea = 13 →
  both = 6 →
  total - (coffee + tea - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_drinks_l4060_406009


namespace NUMINAMATH_CALUDE_complex_modulus_one_l4060_406006

theorem complex_modulus_one (z : ℂ) (h : (1 + z) / (1 - z) = Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l4060_406006


namespace NUMINAMATH_CALUDE_custom_op_two_five_l4060_406070

/-- Custom binary operation on real numbers -/
def custom_op (a b : ℝ) : ℝ := 4 * a + 3 * b

/-- Theorem stating that 2 ⊗ 5 = 23 under the custom operation -/
theorem custom_op_two_five : custom_op 2 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_two_five_l4060_406070


namespace NUMINAMATH_CALUDE_sum_quotient_reciprocal_l4060_406065

theorem sum_quotient_reciprocal (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x + y = 45) (h4 : x * y = 500) : 
  (x / y) + (1 / x) + (1 / y) = 1.34 := by
  sorry

end NUMINAMATH_CALUDE_sum_quotient_reciprocal_l4060_406065


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_point_five_l4060_406083

theorem reciprocal_of_negative_one_point_five :
  let x : ℚ := -3/2  -- -1.5 as a rational number
  let y : ℚ := -2/3  -- The proposed reciprocal
  (∀ z : ℚ, z ≠ 0 → ∃ w : ℚ, z * w = 1) →  -- Definition of reciprocal
  x * y = 1 ∧ y * x = 1 :=  -- Proving y is the reciprocal of x
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_point_five_l4060_406083


namespace NUMINAMATH_CALUDE_age_problem_l4060_406026

/-- The problem of finding when B was half the age A will be in 10 years -/
theorem age_problem (B_age : ℕ) (A_age : ℕ) (x : ℕ) : 
  B_age = 37 →
  A_age = B_age + 7 →
  B_age - x = (A_age + 10) / 2 →
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l4060_406026


namespace NUMINAMATH_CALUDE_social_event_handshakes_l4060_406085

/-- Represents the social event setup -/
structure SocialEvent where
  total_people : Nat
  group_a_size : Nat
  group_b_size : Nat
  group_b_connections : Nat

/-- Calculate the number of handshakes in the social event -/
def calculate_handshakes (event : SocialEvent) : Nat :=
  let group_b_internal := event.group_b_size.choose 2
  let group_ab_handshakes := event.group_b_size * event.group_b_connections
  group_b_internal + group_ab_handshakes

/-- Theorem stating the number of handshakes in the given social event -/
theorem social_event_handshakes :
  ∃ (event : SocialEvent),
    event.total_people = 40 ∧
    event.group_a_size = 25 ∧
    event.group_b_size = 15 ∧
    event.group_b_connections = 5 ∧
    calculate_handshakes event = 180 := by
  sorry

end NUMINAMATH_CALUDE_social_event_handshakes_l4060_406085


namespace NUMINAMATH_CALUDE_odot_commutative_odot_no_identity_odot_associativity_undetermined_l4060_406008

-- Define the binary operation
def odot (x y : ℝ) : ℝ := 2 * (x + 2) * (y + 2) - 3

-- Theorem for commutativity
theorem odot_commutative : ∀ x y : ℝ, odot x y = odot y x := by sorry

-- Theorem for non-existence of identity element
theorem odot_no_identity : ¬ ∃ e : ℝ, ∀ x : ℝ, odot x e = x ∧ odot e x = x := by sorry

-- Theorem for undetermined associativity
theorem odot_associativity_undetermined : 
  ¬ (∀ x y z : ℝ, odot (odot x y) z = odot x (odot y z)) ∧ 
  ¬ (∃ x y z : ℝ, odot (odot x y) z ≠ odot x (odot y z)) := by sorry

end NUMINAMATH_CALUDE_odot_commutative_odot_no_identity_odot_associativity_undetermined_l4060_406008


namespace NUMINAMATH_CALUDE_bulb_probability_l4060_406043

/-- The probability that a bulb from factory X works for over 4000 hours -/
def prob_x : ℝ := 0.59

/-- The probability that a bulb from factory Y works for over 4000 hours -/
def prob_y : ℝ := 0.65

/-- The probability that a bulb from factory Z works for over 4000 hours -/
def prob_z : ℝ := 0.70

/-- The proportion of bulbs supplied by factory X -/
def supply_x : ℝ := 0.5

/-- The proportion of bulbs supplied by factory Y -/
def supply_y : ℝ := 0.3

/-- The proportion of bulbs supplied by factory Z -/
def supply_z : ℝ := 0.2

/-- The overall probability that a randomly selected bulb will work for over 4000 hours -/
def overall_prob : ℝ := supply_x * prob_x + supply_y * prob_y + supply_z * prob_z

theorem bulb_probability : overall_prob = 0.63 := by sorry

end NUMINAMATH_CALUDE_bulb_probability_l4060_406043


namespace NUMINAMATH_CALUDE_min_vertical_distance_l4060_406047

/-- The vertical distance between |x| and -x^2-4x-3 -/
def verticalDistance (x : ℝ) : ℝ := |x| - (-x^2 - 4*x - 3)

/-- The minimum vertical distance between |x| and -x^2-4x-3 is 3/4 -/
theorem min_vertical_distance :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), verticalDistance x₀ ≤ verticalDistance x ∧ verticalDistance x₀ = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_min_vertical_distance_l4060_406047


namespace NUMINAMATH_CALUDE_square_condition_l4060_406018

theorem square_condition (n : ℕ) : 
  (∃ k : ℕ, (n^3 + 39*n - 2)*n.factorial + 17*21^n + 5 = k^2) ↔ n = 1 :=
sorry

end NUMINAMATH_CALUDE_square_condition_l4060_406018


namespace NUMINAMATH_CALUDE_church_seating_capacity_l4060_406094

theorem church_seating_capacity (chairs_per_row : ℕ) (num_rows : ℕ) (total_people : ℕ) :
  chairs_per_row = 6 →
  num_rows = 20 →
  total_people = 600 →
  total_people / (chairs_per_row * num_rows) = 5 :=
by sorry

end NUMINAMATH_CALUDE_church_seating_capacity_l4060_406094


namespace NUMINAMATH_CALUDE_billys_bicycles_l4060_406049

/-- The number of spokes per wheel -/
def spokes_per_wheel : ℕ := 10

/-- The total number of spokes in the garage -/
def total_spokes : ℕ := 80

/-- The number of wheels per bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- The number of bicycles owned by Billy's family -/
def number_of_bicycles : ℕ := total_spokes / (spokes_per_wheel * wheels_per_bicycle)

theorem billys_bicycles : number_of_bicycles = 4 := by
  sorry

end NUMINAMATH_CALUDE_billys_bicycles_l4060_406049


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l4060_406024

theorem fraction_to_decimal : (15 : ℚ) / 625 = (24 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l4060_406024


namespace NUMINAMATH_CALUDE_total_cases_giving_one_card_l4060_406075

def blue_cards : ℕ := 3
def yellow_cards : ℕ := 5

theorem total_cases_giving_one_card : blue_cards + yellow_cards = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_cases_giving_one_card_l4060_406075


namespace NUMINAMATH_CALUDE_ruby_apples_remaining_l4060_406072

theorem ruby_apples_remaining (initial : ℕ) (taken : ℕ) (remaining : ℕ) : 
  initial = 6357912 → taken = 2581435 → remaining = 3776477 → initial - taken = remaining := by
  sorry

end NUMINAMATH_CALUDE_ruby_apples_remaining_l4060_406072


namespace NUMINAMATH_CALUDE_length_AC_l4060_406003

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 49}

structure PointsOnCircle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_A : A ∈ Circle
  h_B : B ∈ Circle
  h_AB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64
  h_C : C ∈ Circle
  h_C_midpoint : C.1 = (A.1 + B.1) / 2 ∧ C.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem length_AC (points : PointsOnCircle) :
  (points.A.1 - points.C.1)^2 + (points.A.2 - points.C.2)^2 = 98 - 14 * Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_length_AC_l4060_406003


namespace NUMINAMATH_CALUDE_power_function_coefficient_l4060_406034

/-- A function f is a power function if it has the form f(x) = ax^n, where a and n are constants and n ≠ 0 -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), n ≠ 0 ∧ ∀ x, f x = a * x ^ n

/-- If f(x) = (2m-1)x^3 is a power function, then m = 1 -/
theorem power_function_coefficient (m : ℝ) :
  IsPowerFunction (fun x => (2 * m - 1) * x ^ 3) → m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_power_function_coefficient_l4060_406034


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l4060_406054

/-- Given two digits X and Y in base d > 8, if XY_d + XX_d = 234_d, then X_d - Y_d = -2_d. -/
theorem digit_difference_in_base_d (d : ℕ) (X Y : ℕ) (h_d : d > 8) 
  (h_digits : X < d ∧ Y < d) 
  (h_sum : X * d + Y + X * d + X = 2 * d * d + 3 * d + 4) :
  X - Y = d - 2 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l4060_406054


namespace NUMINAMATH_CALUDE_sin_150_degrees_l4060_406051

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l4060_406051


namespace NUMINAMATH_CALUDE_opposite_numbers_example_l4060_406097

theorem opposite_numbers_example : -(-(5 : ℤ)) = -(-|5|) → -(-(5 : ℤ)) + (-|5|) = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_example_l4060_406097


namespace NUMINAMATH_CALUDE_factorization_equality_l4060_406033

theorem factorization_equality (a b : ℝ) : 
  a^2 - 4*b^2 - 2*a + 4*b = (a + 2*b - 2) * (a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l4060_406033


namespace NUMINAMATH_CALUDE_minus_510_in_third_quadrant_l4060_406099

-- Define a function to normalize an angle to the range [0, 360)
def normalizeAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360

-- Define a function to determine the quadrant of an angle
def getQuadrant (angle : Int) : Nat :=
  let normalizedAngle := normalizeAngle angle
  if 0 < normalizedAngle && normalizedAngle < 90 then 1
  else if 90 ≤ normalizedAngle && normalizedAngle < 180 then 2
  else if 180 ≤ normalizedAngle && normalizedAngle < 270 then 3
  else 4

-- Theorem statement
theorem minus_510_in_third_quadrant :
  getQuadrant (-510) = 3 :=
sorry

end NUMINAMATH_CALUDE_minus_510_in_third_quadrant_l4060_406099


namespace NUMINAMATH_CALUDE_line_circle_separate_trajectory_of_P_l4060_406011

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define the line l
def line_l (a b α t : ℝ) (x y : ℝ) : Prop :=
  x = a + t * Real.cos α ∧ y = b + t * Real.sin α

-- Part 1: Line and circle are separate
theorem line_circle_separate :
  ∀ x y t : ℝ,
  line_l 8 0 (π/3) t x y →
  ¬ circle_C x y :=
sorry

-- Part 2: Trajectory of point P
theorem trajectory_of_P :
  ∀ a b x y : ℝ,
  (∃ α t₁ t₂ : ℝ,
    circle_C (a + t₁ * Real.cos α) (b + t₁ * Real.sin α) ∧
    circle_C (a + t₂ * Real.cos α) (b + t₂ * Real.sin α) ∧
    t₁ ≠ t₂ ∧
    (a^2 + b^2) * ((a + t₁ * Real.cos α)^2 + (b + t₁ * Real.sin α)^2) =
    ((a + t₂ * Real.cos α)^2 + (b + t₂ * Real.sin α)^2) * a^2 + b^2) →
  x^2 + y^2 = 8 :=
sorry

end NUMINAMATH_CALUDE_line_circle_separate_trajectory_of_P_l4060_406011


namespace NUMINAMATH_CALUDE_range_of_a_l4060_406061

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 5 * x + 6 = 0}

-- Theorem statement
theorem range_of_a (a : ℝ) : Set.Nonempty (A a) → a ∈ Set.Iic (25/24) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4060_406061


namespace NUMINAMATH_CALUDE_second_next_perfect_square_l4060_406032

theorem second_next_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ m : ℕ, m^2 = x + 4 * (x : ℝ).sqrt + 4 :=
sorry

end NUMINAMATH_CALUDE_second_next_perfect_square_l4060_406032


namespace NUMINAMATH_CALUDE_f_less_than_g_max_l4060_406010

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (2*a + 1) * x + 2 * log x

def g (x : ℝ) : ℝ := x^2 - 2*x

theorem f_less_than_g_max (a : ℝ) :
  (∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Ioo 0 2, f a x₁ < g x₂) →
  a > log 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_f_less_than_g_max_l4060_406010


namespace NUMINAMATH_CALUDE_remainder_theorem_l4060_406013

/-- The polynomial to be divided -/
def f (x : ℝ) : ℝ := x^5 - 2*x^4 - x^3 + 2*x^2 + x

/-- The divisor polynomial -/
def g (x : ℝ) : ℝ := (x^2 - 9) * (x - 1)

/-- The proposed remainder -/
def r (x : ℝ) : ℝ := 9*x^2 + 73*x - 81

/-- Theorem stating that r is the remainder when f is divided by g -/
theorem remainder_theorem : ∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4060_406013


namespace NUMINAMATH_CALUDE_men_per_table_l4060_406056

/-- Given a restaurant scenario, prove the number of men at each table. -/
theorem men_per_table (num_tables : ℕ) (women_per_table : ℕ) (total_customers : ℕ) : 
  num_tables = 6 → women_per_table = 3 → total_customers = 48 → 
  (total_customers - num_tables * women_per_table) / num_tables = 5 := by
  sorry

end NUMINAMATH_CALUDE_men_per_table_l4060_406056


namespace NUMINAMATH_CALUDE_class_gender_ratio_l4060_406012

/-- Proves that given the boys' average score of 90, girls' average score of 96,
    and overall class average of 94, the ratio of boys to girls in the class is 1:2. -/
theorem class_gender_ratio (B G : ℕ) (B_pos : B > 0) (G_pos : G > 0) : 
  (90 * B + 96 * G) / (B + G) = 94 → B = G / 2 := by
  sorry

end NUMINAMATH_CALUDE_class_gender_ratio_l4060_406012


namespace NUMINAMATH_CALUDE_water_gun_game_theorem_l4060_406052

/-- Represents a student with a position -/
structure Student where
  position : ℝ × ℝ

/-- The environment of the water gun game -/
structure WaterGunGame where
  n : ℕ
  students : Fin (2*n+1) → Student
  distinct_distances : ∀ i j k l, i ≠ j → k ≠ l → 
    (students i).position ≠ (students j).position → 
    (students k).position ≠ (students l).position →
    dist (students i).position (students j).position ≠ 
    dist (students k).position (students l).position

/-- A student squirts another student -/
def squirts (game : WaterGunGame) (i j : Fin (2*game.n+1)) : Prop :=
  ∀ k, k ≠ j → 
    dist (game.students i).position (game.students j).position < 
    dist (game.students i).position (game.students k).position

theorem water_gun_game_theorem (game : WaterGunGame) : 
  (∃ i j, i ≠ j ∧ squirts game i j ∧ squirts game j i) ∧ 
  (∃ i, ∀ j, ¬squirts game j i) :=
sorry

end NUMINAMATH_CALUDE_water_gun_game_theorem_l4060_406052


namespace NUMINAMATH_CALUDE_pizza_distribution_l4060_406082

theorem pizza_distribution (total_pizzas : ℕ) (slices_per_pizza : ℕ) (num_students : ℕ)
  (leftover_cheese : ℕ) (leftover_onion : ℕ) (onion_per_student : ℕ)
  (h1 : total_pizzas = 6)
  (h2 : slices_per_pizza = 18)
  (h3 : num_students = 32)
  (h4 : leftover_cheese = 8)
  (h5 : leftover_onion = 4)
  (h6 : onion_per_student = 1) :
  (total_pizzas * slices_per_pizza - leftover_cheese - leftover_onion - num_students * onion_per_student) / num_students = 2 := by
  sorry

#check pizza_distribution

end NUMINAMATH_CALUDE_pizza_distribution_l4060_406082


namespace NUMINAMATH_CALUDE_word_arrangements_l4060_406002

/-- The number of distinct letters in the word -/
def n : ℕ := 6

/-- The number of units to be arranged after combining the T's -/
def k : ℕ := 5

/-- The number of ways to arrange the T's within their unit -/
def t : ℕ := 2

/-- The total number of arrangements -/
def total_arrangements : ℕ := k.factorial * t.factorial

theorem word_arrangements : total_arrangements = 240 := by
  sorry

end NUMINAMATH_CALUDE_word_arrangements_l4060_406002


namespace NUMINAMATH_CALUDE_complex_expression_value_l4060_406015

theorem complex_expression_value : 
  (10 - (10.5 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)))) * 20 = 192.6 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_value_l4060_406015


namespace NUMINAMATH_CALUDE_problem_solution_l4060_406022

theorem problem_solution (x y : ℚ) : 
  x = 51 → x^3 * y - 3 * x^2 * y + 2 * x * y = 122650 → y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4060_406022


namespace NUMINAMATH_CALUDE_smallest_number_between_10_and_11_l4060_406074

theorem smallest_number_between_10_and_11 (x y z : ℝ) 
  (sum_eq : x + y + z = 150)
  (y_eq : y = 3 * x + 10)
  (z_eq : z = x^2 - 5) :
  ∃ w, w = min x (min y z) ∧ 10 < w ∧ w < 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_between_10_and_11_l4060_406074


namespace NUMINAMATH_CALUDE_eighth_term_of_specific_sequence_l4060_406079

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  firstTerm : ℚ
  lastTerm : ℚ
  numTerms : ℕ

/-- Calculates the nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  let commonDiff := (seq.lastTerm - seq.firstTerm) / (seq.numTerms - 1)
  seq.firstTerm + (n - 1) * commonDiff

/-- Theorem: The 8th term of the specified arithmetic sequence is 731/29 -/
theorem eighth_term_of_specific_sequence :
  let seq : ArithmeticSequence := ⟨3, 95, 30⟩
  nthTerm seq 8 = 731 / 29 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_of_specific_sequence_l4060_406079


namespace NUMINAMATH_CALUDE_consecutive_integers_product_336_sum_21_l4060_406077

theorem consecutive_integers_product_336_sum_21 :
  ∃ (x : ℤ), (x * (x + 1) * (x + 2) = 336) ∧ (x + (x + 1) + (x + 2) = 21) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_336_sum_21_l4060_406077


namespace NUMINAMATH_CALUDE_nine_zeros_in_binary_representation_l4060_406021

/-- The number of zeros in the binary representation of a natural number -/
def countZeros (n : ℕ) : ℕ := sorry

/-- An unknown non-negative integer -/
def someNumber : ℕ := sorry

/-- The main expression: 6 * 1024 + 4 * 64 + someNumber -/
def mainExpression : ℕ := 6 * 1024 + 4 * 64 + someNumber

theorem nine_zeros_in_binary_representation :
  countZeros mainExpression = 9 := by sorry

end NUMINAMATH_CALUDE_nine_zeros_in_binary_representation_l4060_406021


namespace NUMINAMATH_CALUDE_gcd_problem_l4060_406091

theorem gcd_problem (b : ℤ) (h : 1039 ∣ b) : Int.gcd (b^2 + 7*b + 18) (b + 6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l4060_406091


namespace NUMINAMATH_CALUDE_jane_exercise_hours_per_day_l4060_406037

/-- Given Jane's exercise routine, prove the number of hours she exercises per day --/
theorem jane_exercise_hours_per_day 
  (days_per_week : ℕ) 
  (total_weeks : ℕ) 
  (total_hours : ℕ) 
  (h1 : days_per_week = 5)
  (h2 : total_weeks = 8)
  (h3 : total_hours = 40) :
  total_hours / (total_weeks * days_per_week) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_jane_exercise_hours_per_day_l4060_406037


namespace NUMINAMATH_CALUDE_yearly_salary_calculation_l4060_406027

/-- Proves that the yearly salary excluding turban is 160 rupees given the problem conditions --/
theorem yearly_salary_calculation (partial_payment : ℕ) (turban_value : ℕ) (months_worked : ℕ) (total_months : ℕ) :
  partial_payment = 50 →
  turban_value = 70 →
  months_worked = 9 →
  total_months = 12 →
  (partial_payment + turban_value : ℚ) / (months_worked : ℚ) * (total_months : ℚ) = 160 := by
sorry

end NUMINAMATH_CALUDE_yearly_salary_calculation_l4060_406027


namespace NUMINAMATH_CALUDE_symmetry_implies_sum_power_l4060_406029

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other. -/
def symmetric_x_axis (P Q : ℝ × ℝ) : Prop :=
  P.1 = Q.1 ∧ P.2 = -Q.2

theorem symmetry_implies_sum_power (a b : ℝ) :
  symmetric_x_axis (a, 3) (4, b) → (a + b)^2021 = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_sum_power_l4060_406029


namespace NUMINAMATH_CALUDE_root_product_equals_twenty_l4060_406086

theorem root_product_equals_twenty :
  (32 : ℝ) ^ (1/5) * (16 : ℝ) ^ (1/4) * (25 : ℝ) ^ (1/2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equals_twenty_l4060_406086


namespace NUMINAMATH_CALUDE_anderson_pet_food_weight_l4060_406067

/-- Calculates the total weight of pet food in ounces -/
def total_pet_food_ounces (cat_food_bags : ℕ) (cat_food_weight : ℕ) 
                          (dog_food_bags : ℕ) (dog_food_extra_weight : ℕ) 
                          (ounces_per_pound : ℕ) : ℕ :=
  let total_cat_food := cat_food_bags * cat_food_weight
  let dog_food_weight := cat_food_weight + dog_food_extra_weight
  let total_dog_food := dog_food_bags * dog_food_weight
  let total_weight := total_cat_food + total_dog_food
  total_weight * ounces_per_pound

/-- Theorem: The total weight of pet food Mrs. Anderson bought is 256 ounces -/
theorem anderson_pet_food_weight : 
  total_pet_food_ounces 2 3 2 2 16 = 256 := by
  sorry

end NUMINAMATH_CALUDE_anderson_pet_food_weight_l4060_406067


namespace NUMINAMATH_CALUDE_squirrel_acorns_l4060_406063

theorem squirrel_acorns (num_squirrels : ℕ) (acorns_collected : ℕ) (acorns_needed_per_squirrel : ℕ) :
  num_squirrels = 5 →
  acorns_collected = 575 →
  acorns_needed_per_squirrel = 130 →
  (num_squirrels * acorns_needed_per_squirrel - acorns_collected) / num_squirrels = 15 :=
by sorry

end NUMINAMATH_CALUDE_squirrel_acorns_l4060_406063


namespace NUMINAMATH_CALUDE_remaining_distance_to_hotel_l4060_406064

/-- Calculates the remaining distance to the hotel given Samuel's journey conditions --/
theorem remaining_distance_to_hotel : 
  let total_distance : ℕ := 600
  let speed1 : ℕ := 50
  let time1 : ℕ := 3
  let speed2 : ℕ := 80
  let time2 : ℕ := 4
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let traveled_distance := distance1 + distance2
  total_distance - traveled_distance = 130 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_to_hotel_l4060_406064


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l4060_406017

theorem smallest_positive_integer_with_remainders : 
  ∃ n : ℕ, n > 1 ∧ 
    n % 5 = 1 ∧ 
    n % 7 = 1 ∧ 
    n % 8 = 1 ∧ 
    (∀ m : ℕ, m > 1 → m % 5 = 1 → m % 7 = 1 → m % 8 = 1 → n ≤ m) ∧
    80 < n ∧ 
    n < 299 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l4060_406017


namespace NUMINAMATH_CALUDE_cubic_derivative_root_existence_l4060_406096

/-- A cubic polynomial with real coefficients -/
structure CubicPolynomial where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The roots of a cubic polynomial -/
structure CubicRoots where
  a : ℝ
  b : ℝ
  c : ℝ
  h_order : a ≤ b ∧ b ≤ c

/-- Theorem: The derivative of a cubic polynomial has a root in the specified interval -/
theorem cubic_derivative_root_existence (f : CubicPolynomial) (roots : CubicRoots) :
  ∃ x : ℝ, x ∈ Set.Icc ((roots.b + roots.c) / 2) ((roots.b + 2 * roots.c) / 3) ∧
    (3 * x^2 + 2 * f.p * x + f.q) = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_derivative_root_existence_l4060_406096


namespace NUMINAMATH_CALUDE_master_zhang_apple_sales_l4060_406007

/-- The number of apples Master Zhang must sell to make a profit of 15 yuan -/
def apples_to_sell : ℕ := 100

/-- The buying price in yuan per apple -/
def buying_price : ℚ := 1 / 4

/-- The selling price in yuan per apple -/
def selling_price : ℚ := 2 / 5

/-- The desired profit in yuan -/
def desired_profit : ℕ := 15

theorem master_zhang_apple_sales :
  apples_to_sell = (desired_profit : ℚ) / (selling_price - buying_price) := by sorry

end NUMINAMATH_CALUDE_master_zhang_apple_sales_l4060_406007


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l4060_406066

/-- Given a circle with area π/4 square units, its diameter is 1 unit. -/
theorem circle_diameter_from_area :
  ∀ (r : ℝ), π * r^2 = π / 4 → 2 * r = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l4060_406066


namespace NUMINAMATH_CALUDE_horner_method_v2_l4060_406004

def f (x : ℝ) : ℝ := 2*x^5 - 5*x^4 - 4*x^3 + 3*x^2 - 6*x + 7

def horner_v2 (a b c d e f x : ℝ) : ℝ :=
  ((a * x + b) * x + c) * x + d

theorem horner_method_v2 :
  horner_v2 2 (-5) (-4) 3 (-6) 7 5 = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_horner_method_v2_l4060_406004


namespace NUMINAMATH_CALUDE_cylinder_cross_section_area_l4060_406069

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents the cross-section area of a sliced cylinder -/
def crossSectionArea (c : Cylinder) (arcAngle : ℝ) : ℝ :=
  sorry

theorem cylinder_cross_section_area :
  let c : Cylinder := { radius := 8, height := 5 }
  let arcAngle : ℝ := 90 * π / 180  -- 90 degrees in radians
  crossSectionArea c arcAngle = 16 * π * Real.sqrt 2 + 32 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_cross_section_area_l4060_406069


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l4060_406040

theorem right_triangle_side_length 
  (Q R S : ℝ × ℝ) 
  (right_angle_Q : (R.1 - Q.1) * (S.1 - Q.1) + (R.2 - Q.2) * (S.2 - Q.2) = 0) 
  (cos_R : (R.1 - Q.1) / Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) = 4/9)
  (RS_length : Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 9) :
  Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) = Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l4060_406040


namespace NUMINAMATH_CALUDE_sqrt_sum_squared_l4060_406046

theorem sqrt_sum_squared (x : ℝ) :
  (Real.sqrt (10 + x) + Real.sqrt (30 - x) = 8) →
  ((10 + x) * (30 - x) = 144) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_squared_l4060_406046


namespace NUMINAMATH_CALUDE_quadratic_inequality_l4060_406068

theorem quadratic_inequality (z : ℝ) : z^2 - 40*z + 400 ≤ 36 ↔ 14 ≤ z ∧ z ≤ 26 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l4060_406068


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4060_406071

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 - x - 2 = 0}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4060_406071


namespace NUMINAMATH_CALUDE_only_paintable_integer_l4060_406023

/-- Represents a painting pattern for the fence. -/
structure PaintingPattern where
  start : ℕ
  interval : ℕ

/-- Checks if a given triple (h, t, u) results in a valid painting pattern. -/
def isValidPainting (h t u : ℕ) : Prop :=
  let harold := PaintingPattern.mk 4 h
  let tanya := PaintingPattern.mk 5 (2 * t)
  let ulysses := PaintingPattern.mk 6 (3 * u)
  ∀ n : ℕ, n ≥ 1 →
    (∃! painter, painter ∈ [harold, tanya, ulysses] ∧
      ∃ k, n = painter.start + painter.interval * k)

/-- Calculates the paintable integer for a given triple (h, t, u). -/
def paintableInteger (h t u : ℕ) : ℕ :=
  100 * h + 20 * t + 2 * u

/-- The main theorem stating that 390 is the only paintable integer. -/
theorem only_paintable_integer :
  ∀ h t u : ℕ, h > 0 ∧ t > 0 ∧ u > 0 →
    isValidPainting h t u ↔ paintableInteger h t u = 390 :=
sorry

end NUMINAMATH_CALUDE_only_paintable_integer_l4060_406023


namespace NUMINAMATH_CALUDE_min_type1_figures_l4060_406000

/-- The side length of the equilateral triangle T -/
def side_length : ℕ := 2022

/-- The total number of unit triangles in T -/
def total_triangles : ℕ := side_length * (side_length + 1) / 2

/-- The number of upward-pointing unit triangles in T -/
def upward_triangles : ℕ := (total_triangles + side_length) / 2

/-- The number of downward-pointing unit triangles in T -/
def downward_triangles : ℕ := (total_triangles - side_length) / 2

/-- The excess of upward-pointing unit triangles -/
def excess_upward : ℕ := upward_triangles - downward_triangles

/-- A figure consisting of 4 equilateral unit triangles -/
inductive Figure
| Type1 : Figure  -- Has an excess of ±2 upward-pointing unit triangles
| Type2 : Figure  -- Has equal number of upward and downward-pointing unit triangles
| Type3 : Figure  -- Has equal number of upward and downward-pointing unit triangles
| Type4 : Figure  -- Has equal number of upward and downward-pointing unit triangles

/-- A covering of the triangle T with figures -/
def Covering := List Figure

/-- Predicate to check if a covering is valid -/
def is_valid_covering (c : Covering) : Prop := sorry

/-- The number of Type1 figures in a covering -/
def count_type1 (c : Covering) : ℕ := sorry

theorem min_type1_figures :
  ∃ (c : Covering), is_valid_covering c ∧
  count_type1 c = 1011 ∧
  ∀ (c' : Covering), is_valid_covering c' → count_type1 c' ≥ 1011 := by sorry

end NUMINAMATH_CALUDE_min_type1_figures_l4060_406000


namespace NUMINAMATH_CALUDE_chess_tournament_players_l4060_406092

/-- A chess tournament with specific point distribution rules -/
structure ChessTournament where
  n : ℕ  -- Number of players not in the lowest-scoring group
  total_players : ℕ := n + 15
  
  -- Each player plays exactly one game against every other player
  games_played : ℕ := total_players * (total_players - 1) / 2
  
  -- Points from games between non-lowest scoring players
  points_upper : ℕ := n * (n - 1) / 2
  
  -- Points from games within lowest scoring group
  points_lower : ℕ := 105
  
  -- Conditions on point distribution
  point_distribution : Prop := 
    2 * points_upper + 2 * points_lower = games_played

/-- The theorem stating that the total number of players is 36 -/
theorem chess_tournament_players : 
  ∀ t : ChessTournament, t.total_players = 36 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l4060_406092


namespace NUMINAMATH_CALUDE_min_value_sin_cos_cubic_min_value_achievable_l4060_406035

theorem min_value_sin_cos_cubic (x : ℝ) : 
  Real.sin x ^ 3 + 2 * Real.cos x ^ 3 ≥ -4 * Real.sqrt 2 / 3 :=
sorry

theorem min_value_achievable : 
  ∃ x : ℝ, Real.sin x ^ 3 + 2 * Real.cos x ^ 3 = -4 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_sin_cos_cubic_min_value_achievable_l4060_406035


namespace NUMINAMATH_CALUDE_original_number_proof_l4060_406055

theorem original_number_proof (x : ℝ) : ((x - 3) / 6) * 12 = 8 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l4060_406055


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l4060_406093

/-- The focus of the parabola y = 3x^2 has coordinates (0, 1/12) -/
theorem parabola_focus_coordinates :
  ∀ (x y : ℝ), y = 3 * x^2 → ∃ (p : ℝ), p > 0 ∧ x^2 = (1/(4*p)) * y ∧ (0, p) = (0, 1/12) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l4060_406093


namespace NUMINAMATH_CALUDE_total_time_remaining_wallpaper_l4060_406095

/-- Represents the time in hours to remove wallpaper from one wall -/
def time_per_wall : ℕ := 2

/-- Represents the number of walls in the dining room -/
def dining_room_walls : ℕ := 4

/-- Represents the number of walls in the living room -/
def living_room_walls : ℕ := 4

/-- Represents the number of walls already completed in the dining room -/
def completed_walls : ℕ := 1

/-- Theorem stating the total time to remove wallpaper from the remaining walls -/
theorem total_time_remaining_wallpaper :
  time_per_wall * (dining_room_walls + living_room_walls - completed_walls) = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_time_remaining_wallpaper_l4060_406095


namespace NUMINAMATH_CALUDE_rectangle_area_l4060_406005

theorem rectangle_area (length width diagonal : ℝ) : 
  length = 16 →
  length / diagonal = 4 / 5 →
  length ^ 2 + width ^ 2 = diagonal ^ 2 →
  length * width = 192 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l4060_406005


namespace NUMINAMATH_CALUDE_sum_of_fractions_l4060_406062

theorem sum_of_fractions : (3 : ℚ) / 7 + 9 / 12 = 33 / 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l4060_406062


namespace NUMINAMATH_CALUDE_fraction_problem_l4060_406030

theorem fraction_problem (p q : ℚ) : 
  q = 5 → 
  1/7 + (2*q - p)/(2*q + p) = 4/7 → 
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l4060_406030


namespace NUMINAMATH_CALUDE_tourist_count_l4060_406044

theorem tourist_count : 
  ∃ (n : ℕ), 
    (1/2 : ℚ) * n + (1/3 : ℚ) * n + (1/4 : ℚ) * n = 39 ∧ 
    n = 36 := by
  sorry

end NUMINAMATH_CALUDE_tourist_count_l4060_406044


namespace NUMINAMATH_CALUDE_right_triangle_has_one_right_angle_l4060_406059

-- Define a right triangle
structure RightTriangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  has_right_angle : ∃ i, angles i = 90

-- Theorem: A right triangle has exactly one right angle
theorem right_triangle_has_one_right_angle (t : RightTriangle) : 
  (∃! i, t.angles i = 90) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_has_one_right_angle_l4060_406059


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l4060_406001

-- Define propositions p and q
def p (x : ℝ) : Prop := |x + 1| ≤ 4
def q (x : ℝ) : Prop := x^2 < 5*x - 6

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem stating that ¬p is a sufficient but not necessary condition for ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x : ℝ, not_p x → not_q x) ∧ 
  (∃ x : ℝ, not_q x ∧ ¬(not_p x)) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l4060_406001


namespace NUMINAMATH_CALUDE_marshmallow_challenge_l4060_406041

/-- The marshmallow challenge problem -/
theorem marshmallow_challenge 
  (haley michael brandon : ℕ) 
  (haley_marshmallows : haley = 8)
  (brandon_half_michael : brandon = michael / 2)
  (total_marshmallows : haley + michael + brandon = 44) :
  michael / haley = 3 := by
  sorry

end NUMINAMATH_CALUDE_marshmallow_challenge_l4060_406041


namespace NUMINAMATH_CALUDE_parabola_translation_l4060_406016

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk 3 0 0
  let translated := translate original 2 (-3)
  y = 3 * x^2 → y = 3 * (x - 2)^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l4060_406016


namespace NUMINAMATH_CALUDE_race_distance_l4060_406090

/-- Prove that the total distance of a race is 240 meters given the specified conditions -/
theorem race_distance (D : ℝ) 
  (h1 : D / 60 * 100 = D + 160) : D = 240 := by
  sorry

#check race_distance

end NUMINAMATH_CALUDE_race_distance_l4060_406090


namespace NUMINAMATH_CALUDE_money_sum_existence_l4060_406048

theorem money_sum_existence : ∃ (k n : ℕ), 
  1 ≤ k ∧ k ≤ 9 ∧ n ≥ 1 ∧
  (k * (100 * n + 10 + 1) = 10666612) ∧
  (k * (n + 2) = (1 + 0 + 6 + 6 + 6 + 6 + 1 + 2)) :=
sorry

end NUMINAMATH_CALUDE_money_sum_existence_l4060_406048
