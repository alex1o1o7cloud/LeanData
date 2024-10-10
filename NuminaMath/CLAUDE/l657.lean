import Mathlib

namespace equation_holds_l657_65762

theorem equation_holds (x : ℝ) (h : x = 12) : ((17.28 / x) / (3.6 * 0.2)) = 2 := by
  sorry

end equation_holds_l657_65762


namespace point_in_fourth_quadrant_l657_65769

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define what it means for a point to be in the fourth quadrant
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- The point we want to prove is in the fourth quadrant
def point_to_check : Point := (1, -2)

-- Theorem statement
theorem point_in_fourth_quadrant : 
  is_in_fourth_quadrant point_to_check := by
  sorry

end point_in_fourth_quadrant_l657_65769


namespace math_olympiad_reform_l657_65797

-- Define the probability of achieving a top-20 ranking in a single competition
def top20_prob : ℚ := 1/4

-- Define the maximum number of competitions
def max_competitions : ℕ := 5

-- Define the number of top-20 rankings needed to join the provincial team
def required_top20 : ℕ := 2

-- Define the function to calculate the probability of joining the provincial team
def prob_join_team : ℚ := sorry

-- Define the random variable ξ representing the number of competitions participated
def ξ : ℕ → ℚ
| 2 => 1/16
| 3 => 3/32
| 4 => 27/64
| 5 => 27/64
| _ => 0

-- Define the expected value of ξ
def expected_ξ : ℚ := sorry

-- Theorem statement
theorem math_olympiad_reform :
  (prob_join_team = 67/256) ∧ (expected_ξ = 356/256) := by sorry

end math_olympiad_reform_l657_65797


namespace total_jellybeans_l657_65735

/-- The number of jellybeans in a bag with black, green, and orange beans. -/
def jellybean_count (black green orange : ℕ) : ℕ :=
  black + green + orange

/-- Theorem stating the total number of jellybeans in the bag -/
theorem total_jellybeans :
  ∃ (black green orange : ℕ),
    black = 8 ∧
    green = black + 2 ∧
    orange = green - 1 ∧
    jellybean_count black green orange = 27 := by
  sorry

end total_jellybeans_l657_65735


namespace max_value_of_e_l657_65754

theorem max_value_of_e (a b c d e : ℝ) 
  (sum_condition : a + b + c + d + e = 10)
  (product_condition : a*b + a*c + a*d + a*e + b*c + b*d + b*e + c*d + c*e + d*e = 20) :
  e ≤ 8 := by
sorry

end max_value_of_e_l657_65754


namespace harkamal_mangoes_l657_65778

/-- Calculates the amount of mangoes purchased given the total cost, grape quantity, grape price, and mango price -/
def mangoes_purchased (total_cost : ℕ) (grape_quantity : ℕ) (grape_price : ℕ) (mango_price : ℕ) : ℕ :=
  (total_cost - grape_quantity * grape_price) / mango_price

theorem harkamal_mangoes :
  mangoes_purchased 1145 8 70 65 = 9 := by
sorry

end harkamal_mangoes_l657_65778


namespace residue_modulo_17_l657_65722

theorem residue_modulo_17 : (101 * 15 - 7 * 9 + 5) % 17 = 7 := by
  sorry

end residue_modulo_17_l657_65722


namespace train_speed_l657_65740

/-- Calculates the speed of a train passing over a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) :
  train_length = 320 →
  bridge_length = 140 →
  time = 36.8 →
  (((train_length + bridge_length) / time) * 3.6) = 45 := by
  sorry

end train_speed_l657_65740


namespace plan2_better_l657_65711

/-- The number of optional questions -/
def total_questions : ℕ := 5

/-- The number of questions Student A can solve -/
def solvable_questions : ℕ := 3

/-- Probability of participating under Plan 1 -/
def prob_plan1 : ℚ := solvable_questions / total_questions

/-- Probability of participating under Plan 2 -/
def prob_plan2 : ℚ := (Nat.choose solvable_questions 2 * Nat.choose (total_questions - solvable_questions) 1 + 
                       Nat.choose solvable_questions 3) / 
                      Nat.choose total_questions 3

/-- Theorem stating that Plan 2 gives a higher probability for Student A -/
theorem plan2_better : prob_plan2 > prob_plan1 := by
  sorry

end plan2_better_l657_65711


namespace root_of_polynomial_l657_65723

theorem root_of_polynomial : ∃ (x : ℝ), x^3 = 5 ∧ x^6 - 6*x^4 - 10*x^3 - 60*x + 7 = 0 := by
  sorry

end root_of_polynomial_l657_65723


namespace unique_tangent_circle_l657_65748

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are tangent to each other -/
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Theorem: There exists exactly one circle of radius 4 that is tangent to two circles of radius 2
    which are tangent to each other, at their point of tangency -/
theorem unique_tangent_circle (c1 c2 : Circle) : 
  c1.radius = 2 → 
  c2.radius = 2 → 
  are_tangent c1 c2 → 
  ∃! c : Circle, c.radius = 4 ∧ are_tangent c c1 ∧ are_tangent c c2 :=
sorry

end unique_tangent_circle_l657_65748


namespace smallest_divisor_with_remainder_fifteen_satisfies_condition_fifteen_is_smallest_l657_65720

theorem smallest_divisor_with_remainder (d : ℕ) : d > 0 ∧ 2021 % d = 11 → d ≥ 15 := by
  sorry

theorem fifteen_satisfies_condition : 2021 % 15 = 11 := by
  sorry

theorem fifteen_is_smallest : ∀ d : ℕ, d > 0 ∧ 2021 % d = 11 → d ≥ 15 := by
  sorry

end smallest_divisor_with_remainder_fifteen_satisfies_condition_fifteen_is_smallest_l657_65720


namespace product_of_integers_l657_65789

theorem product_of_integers (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0)
  (sum_eq : p + q + r = 24)
  (frac_eq : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 240 / (p * q * r) = 1) :
  p * q * r = 384 := by
  sorry

end product_of_integers_l657_65789


namespace min_value_of_function_l657_65739

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  ∃ (y : ℝ), y = x + 1 / (x + 1) ∧
  ∀ (z : ℝ), z > -1 → z + 1 / (z + 1) ≥ x + 1 / (x + 1) ↔ x = 0 :=
by sorry

end min_value_of_function_l657_65739


namespace quadratic_no_roots_l657_65709

/-- A quadratic polynomial function -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The derivative of a quadratic polynomial function -/
def quadratic_derivative (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

/-- Theorem: If the graph of a quadratic polynomial and its derivative
    divide the coordinate plane into four parts, then the polynomial has no real roots -/
theorem quadratic_no_roots (a b c : ℝ) (ha : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    quadratic a b c x₁ = quadratic_derivative a b x₁ ∧
    quadratic a b c x₂ = quadratic_derivative a b x₂) →
  (∀ x : ℝ, quadratic a b c x ≠ 0) :=
by sorry

end quadratic_no_roots_l657_65709


namespace correct_systematic_sampling_l657_65753

def total_missiles : ℕ := 50
def selected_missiles : ℕ := 5

def systematic_sampling (total : ℕ) (selected : ℕ) : ℕ := total / selected

def generate_sequence (start : ℕ) (interval : ℕ) (count : ℕ) : List ℕ :=
  List.range count |>.map (fun i => start + i * interval)

theorem correct_systematic_sampling :
  let interval := systematic_sampling total_missiles selected_missiles
  let sequence := generate_sequence 3 interval selected_missiles
  interval = 10 ∧ sequence = [3, 13, 23, 33, 43] := by sorry

end correct_systematic_sampling_l657_65753


namespace two_true_propositions_l657_65781

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x = 3 → x^2 = 9

-- Define the converse proposition
def converse_prop (x : ℝ) : Prop := x^2 = 9 → x = 3

-- Define the inverse proposition
def inverse_prop (x : ℝ) : Prop := x ≠ 3 → x^2 ≠ 9

-- Define the contrapositive proposition
def contrapositive_prop (x : ℝ) : Prop := x^2 ≠ 9 → x ≠ 3

-- Define the negation proposition
def negation_prop (x : ℝ) : Prop := ¬(x = 3 → x^2 = 9)

-- Theorem statement
theorem two_true_propositions :
  ∃ (A B : (ℝ → Prop)), 
    (A = original_prop ∨ A = converse_prop ∨ A = inverse_prop ∨ A = contrapositive_prop ∨ A = negation_prop) ∧
    (B = original_prop ∨ B = converse_prop ∨ B = inverse_prop ∨ B = contrapositive_prop ∨ B = negation_prop) ∧
    A ≠ B ∧
    (∀ x, A x) ∧ 
    (∀ x, B x) ∧
    (∀ C, (C = original_prop ∨ C = converse_prop ∨ C = inverse_prop ∨ C = contrapositive_prop ∨ C = negation_prop) →
      C ≠ A → C ≠ B → ∃ x, ¬(C x)) :=
by sorry

end two_true_propositions_l657_65781


namespace max_consecutive_interesting_l657_65730

/-- A positive integer is interesting if it is a product of two prime numbers -/
def IsInteresting (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p * q

/-- The maximum number of consecutive interesting positive integers -/
theorem max_consecutive_interesting : 
  (∃ k : ℕ, k > 0 ∧ ∀ i : ℕ, i < k → IsInteresting (i + 1)) ∧ 
  (∀ k : ℕ, k > 3 → ∃ i : ℕ, i < k ∧ ¬IsInteresting (i + 1)) :=
sorry

end max_consecutive_interesting_l657_65730


namespace tan_beta_value_l657_65784

theorem tan_beta_value (α β : Real) 
  (h1 : (Real.sin α * Real.cos α) / (1 - Real.cos (2 * α)) = 1)
  (h2 : Real.tan (α - β) = 1/3) : 
  Real.tan β = 1/7 := by
sorry

end tan_beta_value_l657_65784


namespace base_prime_rep_441_l657_65746

def base_prime_representation (n : ℕ) (primes : List ℕ) : List ℕ :=
  sorry

/-- The base prime representation of 441 using primes 2, 3, 5, and 7 is 0202 -/
theorem base_prime_rep_441 : 
  base_prime_representation 441 [2, 3, 5, 7] = [0, 2, 0, 2] := by
  sorry

end base_prime_rep_441_l657_65746


namespace hypotenuse_angle_is_45_degrees_l657_65761

/-- A right triangle with perimeter 2 units -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  perimeter_eq_two : a + b + c = 2
  pythagorean : a^2 + b^2 = c^2

/-- Point on the internal angle bisector of the right angle -/
structure BisectorPoint (t : RightTriangle) where
  distance_sqrt_two : ℝ
  is_sqrt_two : distance_sqrt_two = Real.sqrt 2

/-- The angle subtended by the hypotenuse from the bisector point -/
def hypotenuse_angle (t : RightTriangle) (p : BisectorPoint t) : ℝ := sorry

theorem hypotenuse_angle_is_45_degrees (t : RightTriangle) (p : BisectorPoint t) :
  hypotenuse_angle t p = 45 * π / 180 := by sorry

end hypotenuse_angle_is_45_degrees_l657_65761


namespace andrew_age_is_five_l657_65756

/-- Andrew's age in years -/
def andrew_age : ℕ := 5

/-- Andrew's grandfather's age in years -/
def grandfather_age : ℕ := 10 * andrew_age

/-- The age difference between Andrew's grandfather and Andrew when Andrew was born -/
def age_difference_at_birth : ℕ := 45

theorem andrew_age_is_five :
  andrew_age = 5 ∧
  grandfather_age = 10 * andrew_age ∧
  grandfather_age - andrew_age = age_difference_at_birth :=
by sorry

end andrew_age_is_five_l657_65756


namespace special_triangle_AB_length_l657_65798

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Point K on BC -/
  K : ℝ × ℝ
  /-- Point M on AB -/
  M : ℝ × ℝ
  /-- Point N on AC -/
  N : ℝ × ℝ
  /-- AC length is 18 -/
  h_AC : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 18
  /-- BC length is 21 -/
  h_BC : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 21
  /-- K is midpoint of BC -/
  h_K_midpoint : K = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  /-- M is midpoint of AB -/
  h_M_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  /-- AN length is 6 -/
  h_AN : Real.sqrt ((N.1 - A.1)^2 + (N.2 - A.2)^2) = 6
  /-- MN = KN -/
  h_MN_eq_KN : Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) = Real.sqrt ((N.1 - K.1)^2 + (N.2 - K.2)^2)

/-- The length of AB in the special triangle is 15 -/
theorem special_triangle_AB_length (t : SpecialTriangle) : 
  Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2) = 15 := by
  sorry

end special_triangle_AB_length_l657_65798


namespace sum_of_sixth_powers_mod_seven_l657_65743

theorem sum_of_sixth_powers_mod_seven :
  (1^6 + 2^6 + 3^6 + 4^6 + 5^6 + 6^6) % 7 = 6 := by
  sorry

end sum_of_sixth_powers_mod_seven_l657_65743


namespace seeds_in_bag_l657_65736

-- Define the problem parameters
def seeds_per_ear : ℕ := 4
def price_per_ear : ℚ := 1 / 10
def cost_per_bag : ℚ := 1 / 2
def profit : ℚ := 40
def ears_sold : ℕ := 500

-- Define the theorem
theorem seeds_in_bag : 
  ∃ (seeds_per_bag : ℕ), 
    (ears_sold : ℚ) * price_per_ear - profit = 
    (ears_sold * seeds_per_ear : ℚ) / seeds_per_bag * cost_per_bag ∧ 
    seeds_per_bag = 100 :=
sorry

end seeds_in_bag_l657_65736


namespace C_share_approx_l657_65734

-- Define the total rent
def total_rent : ℚ := 225

-- Define the number of oxen and months for each person
def oxen_A : ℕ := 10
def months_A : ℕ := 7
def oxen_B : ℕ := 12
def months_B : ℕ := 5
def oxen_C : ℕ := 15
def months_C : ℕ := 3
def oxen_D : ℕ := 20
def months_D : ℕ := 6

-- Calculate oxen-months for each person
def oxen_months_A : ℕ := oxen_A * months_A
def oxen_months_B : ℕ := oxen_B * months_B
def oxen_months_C : ℕ := oxen_C * months_C
def oxen_months_D : ℕ := oxen_D * months_D

-- Calculate total oxen-months
def total_oxen_months : ℕ := oxen_months_A + oxen_months_B + oxen_months_C + oxen_months_D

-- Calculate C's share of the rent
def C_share : ℚ := total_rent * (oxen_months_C : ℚ) / (total_oxen_months : ℚ)

-- Theorem to prove
theorem C_share_approx : ∃ ε > 0, abs (C_share - 34.32) < ε :=
sorry

end C_share_approx_l657_65734


namespace complex_number_solution_l657_65760

theorem complex_number_solution (z : ℂ) : 
  Complex.abs z = Real.sqrt 13 ∧ 
  ∃ (k : ℝ), (2 + 3*I)*z*I = k*I → 
  z = 3 + 2*I ∨ z = -3 - 2*I :=
by sorry

end complex_number_solution_l657_65760


namespace land_plot_side_length_l657_65706

theorem land_plot_side_length (area : ℝ) (side : ℝ) : 
  area = 1600 → side * side = area → side = 40 := by
  sorry

end land_plot_side_length_l657_65706


namespace sequence_problem_l657_65715

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a b : ℕ → ℝ) 
  (ha : arithmetic_sequence a) 
  (hb : geometric_sequence b)
  (h_non_zero : ∀ n, a n ≠ 0)
  (h_eq : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)
  (h_b7 : b 7 = a 7) :
  b 5 * b 9 = 16 := by sorry

end sequence_problem_l657_65715


namespace reflection_sequence_exists_l657_65750

/-- Definition of a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Definition of a triangle using three points -/
structure Triangle :=
  (p1 : Point)
  (p2 : Point)
  (p3 : Point)

/-- Definition of a reflection line -/
inductive ReflectionLine
  | AB
  | BC
  | CA

/-- A sequence of reflections -/
def ReflectionSequence := List ReflectionLine

/-- Apply a single reflection to a point -/
def reflect (p : Point) (line : ReflectionLine) : Point :=
  match line with
  | ReflectionLine.AB => ⟨p.x, -p.y⟩
  | ReflectionLine.BC => ⟨3 - p.y, 3 - p.x⟩
  | ReflectionLine.CA => ⟨-p.x, p.y⟩

/-- Apply a sequence of reflections to a point -/
def applyReflections (p : Point) (seq : ReflectionSequence) : Point :=
  seq.foldl reflect p

/-- Apply a sequence of reflections to a triangle -/
def reflectTriangle (t : Triangle) (seq : ReflectionSequence) : Triangle :=
  ⟨applyReflections t.p1 seq, applyReflections t.p2 seq, applyReflections t.p3 seq⟩

/-- The original triangle -/
def originalTriangle : Triangle :=
  ⟨⟨0, 0⟩, ⟨0, 1⟩, ⟨2, 0⟩⟩

/-- The target triangle -/
def targetTriangle : Triangle :=
  ⟨⟨24, 36⟩, ⟨24, 37⟩, ⟨26, 36⟩⟩

theorem reflection_sequence_exists : ∃ (seq : ReflectionSequence), reflectTriangle originalTriangle seq = targetTriangle := by
  sorry

end reflection_sequence_exists_l657_65750


namespace m_less_than_five_l657_65721

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being increasing
def Increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem m_less_than_five
  (h_increasing : Increasing f)
  (h_inequality : ∀ m : ℝ, f (2 * m + 1) > f (3 * m - 4)) :
  ∀ m : ℝ, m < 5 := by
  sorry

end m_less_than_five_l657_65721


namespace school_band_seats_l657_65766

/-- Calculates the total number of seats needed for a school band given the number of players for each instrument. -/
def total_seats (flute trumpet trombone drummer clarinet french_horn : ℕ) : ℕ :=
  flute + trumpet + trombone + drummer + clarinet + french_horn

/-- Proves that the total number of seats needed for the school band is 65. -/
theorem school_band_seats : ∃ (flute trumpet trombone drummer clarinet french_horn : ℕ),
  flute = 5 ∧
  trumpet = 3 * flute ∧
  trombone = trumpet - 8 ∧
  drummer = trombone + 11 ∧
  clarinet = 2 * flute ∧
  french_horn = trombone + 3 ∧
  total_seats flute trumpet trombone drummer clarinet french_horn = 65 := by
  sorry

#eval total_seats 5 15 7 18 10 10

end school_band_seats_l657_65766


namespace negation_existential_proposition_l657_65713

theorem negation_existential_proposition :
  (¬ ∃ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∀ x : ℝ, x^2 + x + 1 < 0) := by
  sorry

end negation_existential_proposition_l657_65713


namespace same_solution_equations_l657_65788

theorem same_solution_equations (x c : ℝ) : 
  (3 * x + 9 = 0) ∧ (c * x - 5 = -11) → c = 2 := by
  sorry

end same_solution_equations_l657_65788


namespace mean_of_remaining_numbers_l657_65749

def numbers : List ℕ := [1877, 1999, 2039, 2045, 2119, 2131]

theorem mean_of_remaining_numbers :
  ∀ (subset : List ℕ),
    subset.length = 4 ∧
    subset ⊆ numbers ∧
    (subset.sum : ℚ) / 4 = 2015 →
    let remaining := numbers.filter (λ x => x ∉ subset)
    (remaining.sum : ℚ) / 2 = 2075 := by
  sorry

end mean_of_remaining_numbers_l657_65749


namespace rectangular_solid_properties_l657_65705

theorem rectangular_solid_properties (a b c : ℝ) 
  (h1 : a * b = Real.sqrt 6)
  (h2 : a * c = Real.sqrt 3)
  (h3 : b * c = Real.sqrt 2) :
  (a * b * c = 6) ∧ 
  (Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 6) := by
sorry

end rectangular_solid_properties_l657_65705


namespace sufficient_not_necessary_condition_l657_65733

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (b > a ∧ a > 0 → (1 : ℝ) / a^2 > (1 : ℝ) / b^2) ∧
  ∃ a b : ℝ, (1 : ℝ) / a^2 > (1 : ℝ) / b^2 ∧ ¬(b > a ∧ a > 0) :=
by sorry

end sufficient_not_necessary_condition_l657_65733


namespace largest_divisor_of_three_consecutive_integers_l657_65700

theorem largest_divisor_of_three_consecutive_integers :
  ∃ (d : ℕ), d > 0 ∧
  (∀ (n : ℤ), (n * (n + 1) * (n + 2)) % d = 0) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℤ), (m * (m + 1) * (m + 2)) % k ≠ 0) ∧
  d = 6 := by
  sorry

end largest_divisor_of_three_consecutive_integers_l657_65700


namespace percentage_increase_l657_65787

theorem percentage_increase (x : ℝ) (h : x = 89.6) :
  ((x - 80) / 80) * 100 = 12 := by sorry

end percentage_increase_l657_65787


namespace movie_length_after_cuts_l657_65755

def original_length : ℝ := 97
def cut_scene1 : ℝ := 4.5
def cut_scene2 : ℝ := 2.75
def cut_scene3 : ℝ := 6.25

theorem movie_length_after_cuts :
  original_length - (cut_scene1 + cut_scene2 + cut_scene3) = 83.5 := by
  sorry

end movie_length_after_cuts_l657_65755


namespace right_triangle_area_l657_65782

theorem right_triangle_area (a b : ℝ) (h1 : a + b = 4) (h2 : a^2 + b^2 = 14) :
  (1/2) * a * b = (1/2) := by
sorry

end right_triangle_area_l657_65782


namespace sum_of_squares_with_means_l657_65783

/-- Given three positive real numbers with specific arithmetic, geometric, and harmonic means, 
    prove that the sum of their squares equals 385.5 -/
theorem sum_of_squares_with_means (x y z : ℝ) 
    (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
    (h_arithmetic : (x + y + z) / 3 = 10)
    (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 7)
    (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 385.5 := by
  sorry

end sum_of_squares_with_means_l657_65783


namespace cube_root_of_eight_l657_65742

theorem cube_root_of_eight (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end cube_root_of_eight_l657_65742


namespace leadership_diagram_is_organizational_structure_l657_65741

/-- Represents types of diagrams --/
inductive Diagram
  | ProgramFlowchart
  | ProcessFlowchart
  | KnowledgeStructureDiagram
  | OrganizationalStructureDiagram

/-- Represents a leadership relationship diagram --/
structure LeadershipDiagram where
  represents_leadership : Bool
  represents_structure : Bool

/-- Definition of an organizational structure diagram --/
def is_organizational_structure_diagram (d : LeadershipDiagram) : Prop :=
  d.represents_leadership ∧ d.represents_structure

/-- Theorem stating that a leadership relationship diagram in a governance group 
    is an organizational structure diagram --/
theorem leadership_diagram_is_organizational_structure :
  ∀ (d : LeadershipDiagram),
  d.represents_leadership ∧ d.represents_structure →
  is_organizational_structure_diagram d :=
by
  sorry

#check leadership_diagram_is_organizational_structure

end leadership_diagram_is_organizational_structure_l657_65741


namespace gem_bonus_percentage_l657_65793

theorem gem_bonus_percentage (purchase : ℝ) (rate : ℝ) (final_gems : ℝ) : 
  purchase = 250 → 
  rate = 100 → 
  final_gems = 30000 → 
  (final_gems - purchase * rate) / (purchase * rate) * 100 = 20 := by
sorry

end gem_bonus_percentage_l657_65793


namespace student_average_age_l657_65758

theorem student_average_age (n : ℕ) (teacher_age : ℕ) (avg_increase : ℚ) :
  n = 19 →
  teacher_age = 40 →
  avg_increase = 1 →
  ∃ (student_avg : ℚ),
    (n : ℚ) * student_avg + teacher_age = (n + 1 : ℚ) * (student_avg + avg_increase) ∧
    student_avg = 20 :=
by sorry

end student_average_age_l657_65758


namespace max_m_inequality_l657_65763

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ m : ℝ, (2/a + 1/b ≥ m/(2*a + b)) → m ≤ 9) ∧ 
  (∃ m : ℝ, m = 9 ∧ 2/a + 1/b ≥ m/(2*a + b)) :=
sorry

end max_m_inequality_l657_65763


namespace apple_slices_l657_65726

theorem apple_slices (S : ℕ) : 
  S > 0 ∧ 
  (S / 16 : ℚ) * S = 5 → 
  S = 16 := by
sorry

end apple_slices_l657_65726


namespace soccer_penalty_kicks_l657_65708

theorem soccer_penalty_kicks (total_players : ℕ) (goalies : ℕ) (shots : ℕ) :
  total_players = 22 →
  goalies = 4 →
  shots = goalies * (total_players - 1) →
  shots = 84 :=
by sorry

end soccer_penalty_kicks_l657_65708


namespace largest_side_is_sixty_l657_65780

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  perimeter_eq : length * 2 + width * 2 = 180
  area_eq : length * width = 10 * 180

/-- The largest side of a SpecialRectangle is 60 feet -/
theorem largest_side_is_sixty (r : SpecialRectangle) : 
  max r.length r.width = 60 := by
  sorry

#check largest_side_is_sixty

end largest_side_is_sixty_l657_65780


namespace simplify_and_evaluate_evaluate_at_one_l657_65732

theorem simplify_and_evaluate (m : ℝ) (h1 : m ≠ -3) (h2 : m ≠ 3) (h3 : m ≠ 0) :
  (m / (m + 3) - 2 * m / (m - 3)) / (m / (m^2 - 9)) = -m - 9 :=
by sorry

-- Evaluation at m = 1
theorem evaluate_at_one :
  (1 / (1 + 3) - 2 * 1 / (1 - 3)) / (1 / (1^2 - 9)) = -10 :=
by sorry

end simplify_and_evaluate_evaluate_at_one_l657_65732


namespace circle_area_problem_l657_65771

/-- The area of the region outside a circle of radius 1.5 and inside two circles of radius 2
    that are internally tangent to the smaller circle at opposite ends of its diameter -/
theorem circle_area_problem : ∃ (area : ℝ),
  let r₁ : ℝ := 1.5 -- radius of smaller circle
  let r₂ : ℝ := 2   -- radius of larger circles
  area = (13/4 : ℝ) * Real.pi - 3 * Real.sqrt 1.75 ∧
  area = 2 * (
    -- Area of sector in larger circle
    (1/3 : ℝ) * Real.pi * r₂^2 -
    -- Area of triangle
    (1/2 : ℝ) * r₁ * Real.sqrt (r₂^2 - r₁^2) -
    -- Area of quarter of smaller circle
    (1/4 : ℝ) * Real.pi * r₁^2
  ) := by sorry


end circle_area_problem_l657_65771


namespace cube_volume_problem_l657_65765

theorem cube_volume_problem (a : ℝ) : 
  (a + 3) * (a - 2) * a - a^3 = 6 → a = 3 + Real.sqrt 15 := by
  sorry

end cube_volume_problem_l657_65765


namespace percent_less_than_l657_65744

theorem percent_less_than (x y z : ℝ) 
  (h1 : x = 1.3 * y) 
  (h2 : x = 0.78 * z) : 
  y = 0.6 * z := by
sorry

end percent_less_than_l657_65744


namespace f_inequality_range_l657_65716

def f (x : ℝ) := -x^3 + 3*x + 2

theorem f_inequality_range (m : ℝ) :
  (∀ θ : ℝ, f (3 + 2 * Real.sin θ) < m) → m > 4 := by
  sorry

end f_inequality_range_l657_65716


namespace a_work_days_l657_65791

/-- The number of days B takes to finish the work alone -/
def b_days : ℝ := 15

/-- The number of days A and B work together -/
def together_days : ℝ := 2

/-- The number of days B works alone after A leaves -/
def b_alone_days : ℝ := 7

/-- The total amount of work to be done -/
def total_work : ℝ := 1

-- The theorem to prove
theorem a_work_days : 
  ∃ (x : ℝ), 
    x > 0 ∧ 
    together_days * (1/x + 1/b_days) + b_alone_days * (1/b_days) = total_work ∧ 
    x = 5 := by
  sorry

end a_work_days_l657_65791


namespace equation_solutions_l657_65794

def equation (x y : ℤ) : ℤ := 
  4*x^3 + 4*x^2*y - 15*x*y^2 - 18*y^3 - 12*x^2 + 6*x*y + 36*y^2 + 5*x - 10*y

def solution_set : Set (ℤ × ℤ) :=
  {p | p.1 = 1 ∧ p.2 = 1} ∪ {p | ∃ (y : ℕ), p.1 = 2*y ∧ p.2 = y}

theorem equation_solutions :
  ∀ (x y : ℤ), x > 0 ∧ y > 0 →
    (equation x y = 0 ↔ (x, y) ∈ solution_set) := by
  sorry

end equation_solutions_l657_65794


namespace max_toads_in_two_ponds_l657_65767

/-- Represents a pond with frogs and toads -/
structure Pond where
  frogRatio : ℕ
  toadRatio : ℕ

/-- The maximum number of toads given two ponds and a total number of frogs -/
def maxToads (pond1 pond2 : Pond) (totalFrogs : ℕ) : ℕ :=
  sorry

/-- Theorem stating the maximum number of toads in the given scenario -/
theorem max_toads_in_two_ponds :
  let pond1 : Pond := { frogRatio := 3, toadRatio := 4 }
  let pond2 : Pond := { frogRatio := 5, toadRatio := 6 }
  let totalFrogs : ℕ := 36
  maxToads pond1 pond2 totalFrogs = 46 := by
  sorry

end max_toads_in_two_ponds_l657_65767


namespace jason_omelet_eggs_l657_65779

/-- The number of eggs Jason consumes in two weeks -/
def total_eggs : ℕ := 42

/-- The number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- The number of eggs Jason uses for his omelet each morning -/
def eggs_per_day : ℚ := total_eggs / days_in_two_weeks

theorem jason_omelet_eggs : eggs_per_day = 3 := by
  sorry

end jason_omelet_eggs_l657_65779


namespace perpendicular_planes_parallel_line_not_perpendicular_no_perpendicular_line_perpendicular_intersection_perpendicular_l657_65710

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the basic relations
variable (on_plane : Point → Plane → Prop)
variable (on_line : Point → Line → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (intersection : Plane → Plane → Line → Prop)

-- Define the given conditions
variable (l : Line) (α β γ : Plane)
variable (h_diff : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)

-- Statement 1
theorem perpendicular_planes_parallel_line :
  perpendicular α β → ∃ m : Line, line_in_plane m α ∧ parallel m β := by sorry

-- Statement 2
theorem not_perpendicular_no_perpendicular_line :
  ¬perpendicular α β → ¬∃ m : Line, line_in_plane m α ∧ line_perpendicular_plane m β := by sorry

-- Statement 3
theorem perpendicular_intersection_perpendicular :
  perpendicular α γ → perpendicular β γ → intersection α β l → line_perpendicular_plane l γ := by sorry

end perpendicular_planes_parallel_line_not_perpendicular_no_perpendicular_line_perpendicular_intersection_perpendicular_l657_65710


namespace max_single_game_schedules_max_n_value_l657_65776

/-- Represents a chess tournament between two teams -/
structure ChessTournament where
  team_size : ℕ
  total_games : ℕ
  games_played : ℕ

/-- Creates a chess tournament with the given parameters -/
def create_tournament (size : ℕ) : ChessTournament :=
  { team_size := size
  , total_games := size * size
  , games_played := 0 }

/-- Theorem stating the maximum number of ways to schedule a single game -/
theorem max_single_game_schedules (t : ChessTournament) (h1 : t.team_size = 15) :
  (t.total_games - t.games_played) ≤ 120 := by
  sorry

/-- Main theorem proving the maximum value of N -/
theorem max_n_value :
  ∃ (t : ChessTournament), t.team_size = 15 ∧ (t.total_games - t.games_played) = 120 := by
  sorry

end max_single_game_schedules_max_n_value_l657_65776


namespace simplify_and_evaluate_l657_65773

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 3 / 3) :
  (a + 1)^2 + a * (1 - a) = Real.sqrt 3 + 1 := by
  sorry

end simplify_and_evaluate_l657_65773


namespace kim_total_points_l657_65777

/-- Represents the points structure of a math contest --/
structure ContestPoints where
  easy : Nat
  average : Nat
  hard : Nat
  expert : Nat
  bonusPerComplex : Nat

/-- Represents a contestant's performance in the math contest --/
structure ContestPerformance where
  points : ContestPoints
  easyCorrect : Nat
  averageCorrect : Nat
  hardCorrect : Nat
  expertCorrect : Nat
  complexSolved : Nat

/-- Calculates the total points for a contestant --/
def calculateTotalPoints (performance : ContestPerformance) : Nat :=
  performance.easyCorrect * performance.points.easy +
  performance.averageCorrect * performance.points.average +
  performance.hardCorrect * performance.points.hard +
  performance.expertCorrect * performance.points.expert +
  performance.complexSolved * performance.points.bonusPerComplex

/-- Theorem stating that Kim's total points in the contest equal 61 --/
theorem kim_total_points :
  let contestPoints : ContestPoints := {
    easy := 2,
    average := 3,
    hard := 5,
    expert := 7,
    bonusPerComplex := 1
  }
  let kimPerformance : ContestPerformance := {
    points := contestPoints,
    easyCorrect := 6,
    averageCorrect := 2,
    hardCorrect := 4,
    expertCorrect := 3,
    complexSolved := 2
  }
  calculateTotalPoints kimPerformance = 61 := by
  sorry


end kim_total_points_l657_65777


namespace intersection_complement_equality_l657_65747

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem intersection_complement_equality : A ∩ (U \ B) = {1} := by sorry

end intersection_complement_equality_l657_65747


namespace sum_remainder_zero_l657_65725

theorem sum_remainder_zero (n : ℤ) : (10 - 2*n + 4*n + 2) % 6 = 0 := by
  sorry

end sum_remainder_zero_l657_65725


namespace max_value_quadratic_l657_65703

theorem max_value_quadratic (p q : ℝ) : 
  q = p - 2 → 
  ∃ (max : ℝ), max = 26 + 2/3 ∧ 
  ∀ (p : ℝ), -3 * p^2 + 24 * p - 50 + 10 * q ≤ max :=
by sorry

end max_value_quadratic_l657_65703


namespace complement_intersection_theorem_l657_65752

def I : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2, 3, 4}
def B : Set Nat := {3, 4, 5, 6}

theorem complement_intersection_theorem :
  (A ∩ B)ᶜ = {1, 2, 5, 6} :=
by sorry

end complement_intersection_theorem_l657_65752


namespace inequality_proof_l657_65712

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 := by
  sorry

end inequality_proof_l657_65712


namespace sin_product_equals_one_eighth_l657_65785

theorem sin_product_equals_one_eighth :
  Real.sin (π / 14) * Real.sin (3 * π / 14) * Real.sin (5 * π / 14) = 1 / 8 := by
  sorry

end sin_product_equals_one_eighth_l657_65785


namespace company_bonus_problem_l657_65768

/-- Represents the company bonus distribution problem -/
theorem company_bonus_problem (n : ℕ) 
  (h1 : 60 * n - 15 = 45 * n + 135) : 
  60 * n - 15 = 585 := by
  sorry

#check company_bonus_problem

end company_bonus_problem_l657_65768


namespace cafeteria_red_apples_l657_65772

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 42

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 7

/-- The number of students who wanted fruit -/
def students_wanting_fruit : ℕ := 9

/-- The number of extra apples -/
def extra_apples : ℕ := 40

/-- Theorem: The cafeteria ordered 42 red apples -/
theorem cafeteria_red_apples :
  red_apples = 42 ∧
  red_apples + green_apples = students_wanting_fruit + extra_apples :=
sorry

end cafeteria_red_apples_l657_65772


namespace real_y_condition_l657_65728

theorem real_y_condition (x y : ℝ) : 
  (4 * y^2 + 6 * x * y + x + 8 = 0) → 
  (∃ y : ℝ, 4 * y^2 + 6 * x * y + x + 8 = 0) ↔ (x ≤ -8/9 ∨ x ≥ 4) :=
by sorry

end real_y_condition_l657_65728


namespace sum_of_squares_l657_65786

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 6*y = -17)
  (eq2 : y^2 + 4*z = 1)
  (eq3 : z^2 + 2*x = 2) :
  x^2 + y^2 + z^2 = 14 := by
  sorry

end sum_of_squares_l657_65786


namespace intersection_when_m_is_one_union_equals_B_l657_65704

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | 0 < x - m ∧ x - m < 3}
def B : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

-- Theorem 1: Intersection of A and B when m = 1
theorem intersection_when_m_is_one :
  A 1 ∩ B = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem 2: Condition for A ∪ B = B
theorem union_equals_B (m : ℝ) :
  A m ∪ B = B ↔ m ≥ 3 ∨ m ≤ -3 := by sorry

end intersection_when_m_is_one_union_equals_B_l657_65704


namespace predicted_weight_approx_l657_65714

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 0.849 * x - 85.712

-- Define the height of the student
def student_height : ℝ := 172

-- Define the tolerance for "approximately" (e.g., within 0.001)
def tolerance : ℝ := 0.001

-- Theorem statement
theorem predicted_weight_approx :
  ∃ (predicted_weight : ℝ), 
    regression_equation student_height = predicted_weight ∧ 
    abs (predicted_weight - 60.316) < tolerance := by
  sorry

end predicted_weight_approx_l657_65714


namespace hyperbola_eccentricity_l657_65724

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if 2x - √3y = 0 is one of its asymptotes, then its eccentricity is √21/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, x^2/a^2 - y^2/b^2 = 1 → 2*x - Real.sqrt 3*y = 0) →
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 21 / 3 := by
  sorry

end hyperbola_eccentricity_l657_65724


namespace min_value_shifted_function_l657_65718

def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + 5 - c

theorem min_value_shifted_function (c : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f c x ≥ m ∧ ∃ (x₀ : ℝ), f c x₀ = m) ∧
  (∀ (x : ℝ), f c x ≥ 2) ∧
  (∃ (x₁ : ℝ), f c x₁ = 2) →
  (∃ (m : ℝ), ∀ (x : ℝ), f c (x - 3) ≥ m ∧ ∃ (x₀ : ℝ), f c (x₀ - 3) = m) ∧
  (∀ (x : ℝ), f c (x - 3) ≥ 2) ∧
  (∃ (x₁ : ℝ), f c (x₁ - 3) = 2) :=
by sorry

end min_value_shifted_function_l657_65718


namespace problem_solution_l657_65799

theorem problem_solution (a b : ℝ) (h : (a - 1)^2 + |b + 2| = 0) :
  2 * (5 * a^2 - 7 * a * b + 9 * b^2) - 3 * (14 * a^2 - 2 * a * b + 3 * b^2) = 20 := by
  sorry

end problem_solution_l657_65799


namespace inverse_image_of_three_l657_65759

-- Define the mapping f: A → B
def f (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement
theorem inverse_image_of_three (h : f 1 = 3) : ∃ x, f x = 3 ∧ x = 1 := by
  sorry


end inverse_image_of_three_l657_65759


namespace equation_general_form_l657_65796

theorem equation_general_form :
  ∀ x : ℝ, (x - 1) * (2 * x + 1) = 2 ↔ 2 * x^2 - x - 3 = 0 :=
by sorry

end equation_general_form_l657_65796


namespace jellybean_ratio_l657_65727

/-- Prove that the ratio of Matilda's jellybeans to Matt's jellybeans is 1:2 -/
theorem jellybean_ratio :
  let steve_jellybeans : ℕ := 84
  let matt_jellybeans : ℕ := 10 * steve_jellybeans
  let matilda_jellybeans : ℕ := 420
  (matilda_jellybeans : ℚ) / (matt_jellybeans : ℚ) = 1 / 2 :=
by sorry

end jellybean_ratio_l657_65727


namespace sum_of_repeating_decimals_l657_65775

/-- Converts a repeating decimal with a single repeating digit to a rational number -/
def repeating_decimal_to_rational (n : ℕ) : ℚ := n / 9

theorem sum_of_repeating_decimals :
  repeating_decimal_to_rational 6 + 
  repeating_decimal_to_rational 2 - 
  repeating_decimal_to_rational 4 + 
  repeating_decimal_to_rational 9 = 13 / 9 := by
  sorry

end sum_of_repeating_decimals_l657_65775


namespace flower_pot_price_difference_l657_65737

theorem flower_pot_price_difference 
  (n : ℕ) 
  (total_cost : ℚ) 
  (largest_pot_cost : ℚ) 
  (h1 : n = 6) 
  (h2 : total_cost = 39/5) 
  (h3 : largest_pot_cost = 77/40) : 
  ∃ (d : ℚ), d = 1/4 ∧ 
  ∃ (x : ℚ), 
    (x + (n - 1) * d = largest_pot_cost) ∧ 
    (n * x + (n * (n - 1) / 2) * d = total_cost) :=
by sorry

end flower_pot_price_difference_l657_65737


namespace divisibility_by_six_l657_65764

theorem divisibility_by_six (a b c : ℤ) (h : 18 ∣ (a^3 + b^3 + c^3)) : 6 ∣ (a * b * c) := by
  sorry

end divisibility_by_six_l657_65764


namespace speed_ratio_l657_65707

/-- The speed of runner A in meters per hour -/
def speed_A : ℝ := sorry

/-- The speed of runner B in meters per hour -/
def speed_B : ℝ := sorry

/-- The length of the track in meters -/
def track_length : ℝ := sorry

/-- When running in the same direction, A catches up with B after 3 hours -/
axiom same_direction : 3 * (speed_A - speed_B) = track_length

/-- When running in opposite directions, A and B meet after 2 hours -/
axiom opposite_direction : 2 * (speed_A + speed_B) = track_length

/-- The ratio of A's speed to B's speed is 5:1 -/
theorem speed_ratio : speed_A / speed_B = 5 := by sorry

end speed_ratio_l657_65707


namespace car_speed_calculation_l657_65795

/-- Calculates the car speed given train and car travel information -/
theorem car_speed_calculation (train_speed : ℝ) (train_time : ℝ) (remaining_distance : ℝ) (car_time : ℝ) :
  train_speed = 120 →
  train_time = 2 →
  remaining_distance = 2.4 →
  car_time = 3 →
  (train_speed * train_time + remaining_distance) / car_time = 80.8 := by
  sorry


end car_speed_calculation_l657_65795


namespace polynomial_multiplication_l657_65757

theorem polynomial_multiplication (x : ℝ) : 
  (x^4 + 50*x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end polynomial_multiplication_l657_65757


namespace square_circle_union_area_l657_65751

/-- The area of the union of a square with side length 8 and a circle with radius 8
    centered at one of the square's vertices is equal to 64 + 48π. -/
theorem square_circle_union_area :
  let square_side : ℝ := 8
  let circle_radius : ℝ := 8
  let square_area := square_side ^ 2
  let circle_area := π * circle_radius ^ 2
  let overlap_area := (1 / 4) * circle_area
  square_area + circle_area - overlap_area = 64 + 48 * π := by
sorry

end square_circle_union_area_l657_65751


namespace cube_sum_divisibility_l657_65731

theorem cube_sum_divisibility (a b c : ℤ) : 
  (∃ k : ℤ, a + b + c = 6 * k) → (∃ m : ℤ, a^3 + b^3 + c^3 = 6 * m) := by
  sorry

end cube_sum_divisibility_l657_65731


namespace determinant_inequality_l657_65717

def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_inequality (x : ℝ) :
  det2x2 7 (x^2) 2 1 > det2x2 3 (-2) 1 x ↔ -5/2 < x ∧ x < 1 := by sorry

end determinant_inequality_l657_65717


namespace initial_number_of_persons_l657_65790

theorem initial_number_of_persons
  (average_weight_increase : ℝ)
  (weight_of_leaving_person : ℝ)
  (weight_of_new_person : ℝ)
  (h1 : average_weight_increase = 5.5)
  (h2 : weight_of_leaving_person = 68)
  (h3 : weight_of_new_person = 95.5) :
  ∃ N : ℕ, N = 5 ∧ 
  N * average_weight_increase = weight_of_new_person - weight_of_leaving_person :=
by sorry

end initial_number_of_persons_l657_65790


namespace alcohol_percentage_after_dilution_l657_65792

theorem alcohol_percentage_after_dilution :
  let initial_volume : ℝ := 15
  let initial_alcohol_percentage : ℝ := 20
  let added_water : ℝ := 2
  let initial_alcohol_volume : ℝ := initial_volume * (initial_alcohol_percentage / 100)
  let new_total_volume : ℝ := initial_volume + added_water
  let new_alcohol_percentage : ℝ := (initial_alcohol_volume / new_total_volume) * 100
  ∀ ε > 0, |new_alcohol_percentage - 17.65| < ε :=
by
  sorry

end alcohol_percentage_after_dilution_l657_65792


namespace exactly_two_base_pairs_l657_65702

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 4 else (a * x^2 + a * x - 1) / x

-- Define what it means for two points to be symmetric about the origin
def symmetricAboutOrigin (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2

-- Define a base pair
def basePair (a : ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  symmetricAboutOrigin p1 p2 ∧ p1.2 = f a p1.1 ∧ p2.2 = f a p2.1

-- The main theorem
theorem exactly_two_base_pairs (a : ℝ) : 
  (∃ p1 p2 p3 p4 : ℝ × ℝ, 
    basePair a p1 p2 ∧ basePair a p3 p4 ∧ 
    p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧
    (∀ p5 p6 : ℝ × ℝ, basePair a p5 p6 → (p5 = p1 ∧ p6 = p2) ∨ (p5 = p3 ∧ p6 = p4) ∨ 
                                         (p5 = p2 ∧ p6 = p1) ∨ (p5 = p4 ∧ p6 = p3))) ↔ 
  a > -6 + 2 * Real.sqrt 6 ∧ a < 1 :=
sorry

end exactly_two_base_pairs_l657_65702


namespace line_intersects_unit_circle_l657_65719

theorem line_intersects_unit_circle 
  (a b : ℝ) (θ : ℝ) (h_neq : a ≠ b) 
  (h_a : a^2 * Real.sin θ + a * Real.cos θ - Real.pi/4 = 0)
  (h_b : b^2 * Real.sin θ + b * Real.cos θ - Real.pi/4 = 0) : 
  ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ (b + a) * x - y - a * b = 0 :=
by sorry

end line_intersects_unit_circle_l657_65719


namespace evies_age_l657_65770

theorem evies_age (x : ℕ) : x + 4 = 3 * (x - 2) → x + 1 = 6 := by
  sorry

end evies_age_l657_65770


namespace average_of_combined_results_l657_65701

theorem average_of_combined_results (n₁ n₂ : ℕ) (avg₁ avg₂ : ℚ) 
  (h₁ : n₁ = 55) (h₂ : n₂ = 28) (h₃ : avg₁ = 28) (h₄ : avg₂ = 55) :
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂ : ℚ) = (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) :=
by sorry

end average_of_combined_results_l657_65701


namespace value_of_a_l657_65745

theorem value_of_a (a b c : ℤ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 8) 
  (eq3 : c = 4) : 
  a = 0 := by sorry

end value_of_a_l657_65745


namespace profit_share_b_is_1800_l657_65774

/-- Represents the profit share calculation for a business partnership --/
def ProfitShare (investment_a investment_b investment_c : ℕ) (profit_diff_ac : ℕ) : ℕ :=
  let ratio_sum := (investment_a / 2000) + (investment_b / 2000) + (investment_c / 2000)
  let part_value := profit_diff_ac / ((investment_c / 2000) - (investment_a / 2000))
  (investment_b / 2000) * part_value

/-- Theorem stating that given the investments and profit difference, 
    the profit share of b is 1800 --/
theorem profit_share_b_is_1800 :
  ProfitShare 8000 10000 12000 720 = 1800 := by
  sorry

end profit_share_b_is_1800_l657_65774


namespace min_value_of_a_l657_65729

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 2*x
def g (a : ℝ) (x : ℝ) : ℝ := a*x + 2

-- Define the theorem
theorem min_value_of_a (h_a : ℝ) (h_a_pos : h_a > 0) : 
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₂ ∈ Set.Icc (-1 : ℝ) 2, f x₁ = g h_a x₂) → 
  h_a ≥ 3 := by
  sorry

-- Note: Set.Icc a b represents the closed interval [a, b]

end min_value_of_a_l657_65729


namespace sin_cos_15_ratio_l657_65738

theorem sin_cos_15_ratio : 
  (Real.sin (15 * π / 180) + Real.cos (15 * π / 180)) / 
  (Real.sin (15 * π / 180) - Real.cos (15 * π / 180)) = -Real.sqrt 3 := by
  sorry

end sin_cos_15_ratio_l657_65738
