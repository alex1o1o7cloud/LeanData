import Mathlib

namespace NUMINAMATH_CALUDE_carpet_area_proof_l408_40858

/-- Calculates the total carpet area required for three rooms -/
def totalCarpetArea (w1 l1 w2 l2 w3 l3 : ℝ) : ℝ :=
  w1 * l1 + w2 * l2 + w3 * l3

/-- Proves that the total carpet area for the given room dimensions is 353 square feet -/
theorem carpet_area_proof :
  totalCarpetArea 12 15 7 9 10 11 = 353 := by
  sorry

#eval totalCarpetArea 12 15 7 9 10 11

end NUMINAMATH_CALUDE_carpet_area_proof_l408_40858


namespace NUMINAMATH_CALUDE_positive_difference_problem_l408_40890

theorem positive_difference_problem : 
  ∀ x : ℝ, (33 + x) / 2 = 37 → |x - 33| = 8 := by
sorry

end NUMINAMATH_CALUDE_positive_difference_problem_l408_40890


namespace NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l408_40836

/-- The volume of space in a cylinder not occupied by three cones -/
theorem cylinder_minus_cones_volume (h_cyl : ℝ) (r_cyl : ℝ) (h_cone : ℝ) (r_cone : ℝ) :
  h_cyl = 36 →
  r_cyl = 10 →
  h_cone = 18 →
  r_cone = 10 →
  (π * r_cyl^2 * h_cyl) - 3 * (1/3 * π * r_cone^2 * h_cone) = 1800 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l408_40836


namespace NUMINAMATH_CALUDE_transform_f1_to_f2_l408_40865

/-- Represents a quadratic function of the form y = a(x - h)^2 + k -/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Applies a horizontal and vertical translation to a quadratic function -/
def translate (f : QuadraticFunction) (dx dy : ℝ) : QuadraticFunction :=
  { a := f.a
  , h := f.h - dx
  , k := f.k - dy }

/-- The original quadratic function y = -2(x - 1)^2 + 3 -/
def f1 : QuadraticFunction :=
  { a := -2
  , h := 1
  , k := 3 }

/-- The target quadratic function y = -2x^2 -/
def f2 : QuadraticFunction :=
  { a := -2
  , h := 0
  , k := 0 }

/-- Theorem stating that translating f1 by 1 unit left and 3 units down results in f2 -/
theorem transform_f1_to_f2 : translate f1 1 3 = f2 := by sorry

end NUMINAMATH_CALUDE_transform_f1_to_f2_l408_40865


namespace NUMINAMATH_CALUDE_probability_sum_twenty_l408_40827

/-- A dodecahedral die with faces labeled 1 through 12 -/
def DodecahedralDie : Finset ℕ := Finset.range 12 

/-- The sample space of rolling two dodecahedral dice -/
def TwoDiceRolls : Finset (ℕ × ℕ) :=
  DodecahedralDie.product DodecahedralDie

/-- The event of rolling a sum of 20 with two dodecahedral dice -/
def SumTwenty : Finset (ℕ × ℕ) :=
  TwoDiceRolls.filter (fun p => p.1 + p.2 = 20)

/-- The probability of an event in a finite sample space -/
def probability (event : Finset α) (sampleSpace : Finset α) : ℚ :=
  event.card / sampleSpace.card

theorem probability_sum_twenty :
  probability SumTwenty TwoDiceRolls = 5 / 144 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_twenty_l408_40827


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l408_40856

/-- A line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

theorem parallel_lines_m_value :
  ∀ m : ℝ,
  let l1 : Line := ⟨6, m, -1⟩
  let l2 : Line := ⟨2, -1, 1⟩
  parallel l1 l2 → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l408_40856


namespace NUMINAMATH_CALUDE_product_of_numbers_l408_40820

theorem product_of_numbers (a b : ℝ) (h1 : a + b = 70) (h2 : a - b = 10) : a * b = 1200 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l408_40820


namespace NUMINAMATH_CALUDE_cable_length_equals_scientific_notation_l408_40816

/-- The total length of fiber optic cable routes in kilometers -/
def cable_length : ℝ := 59580000

/-- The scientific notation representation of the cable length -/
def cable_length_scientific : ℝ := 5.958 * (10 ^ 7)

/-- Theorem stating that the cable length is equal to its scientific notation representation -/
theorem cable_length_equals_scientific_notation : cable_length = cable_length_scientific := by
  sorry

end NUMINAMATH_CALUDE_cable_length_equals_scientific_notation_l408_40816


namespace NUMINAMATH_CALUDE_cl_ab_ratio_l408_40826

/-- A regular pentagon with specific points and angle conditions -/
structure RegularPentagonWithPoints where
  /-- The side length of the regular pentagon -/
  s : ℝ
  /-- Point K on side AE -/
  k : ℝ
  /-- Point L on side CD -/
  l : ℝ
  /-- The sum of angles LAE and KCD is 108° -/
  angle_sum : k + l = 108
  /-- The ratio of AK to KE is 3:7 -/
  length_ratio : k / (s - k) = 3 / 7
  /-- The side length is positive -/
  s_pos : s > 0
  /-- K is between A and E -/
  k_between : 0 < k ∧ k < s
  /-- L is between C and D -/
  l_between : 0 < l ∧ l < s

/-- The theorem stating the ratio of CL to AB in the given pentagon -/
theorem cl_ab_ratio (p : RegularPentagonWithPoints) : (p.s - p.l) / p.s = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_cl_ab_ratio_l408_40826


namespace NUMINAMATH_CALUDE_three_to_negative_x_is_exponential_l408_40859

/-- Definition of an exponential function -/
def is_exponential_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x, f x = a^x

/-- The function y = 3^(-x) is an exponential function -/
theorem three_to_negative_x_is_exponential :
  is_exponential_function (fun x => 3^(-x)) :=
sorry

end NUMINAMATH_CALUDE_three_to_negative_x_is_exponential_l408_40859


namespace NUMINAMATH_CALUDE_spade_ace_probability_l408_40888

/-- Represents a standard deck of 52 cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suit of a card -/
inductive Suit
  | Spades | Hearts | Diamonds | Clubs

/-- Represents the rank of a card -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A function to determine if a card is a spade -/
def is_spade : Fin 52 → Bool := sorry

/-- A function to determine if a card is an ace -/
def is_ace : Fin 52 → Bool := sorry

/-- The number of spades in a standard deck -/
def spade_count : Nat := 13

/-- The number of aces in a standard deck -/
def ace_count : Nat := 4

/-- Theorem: The probability of drawing a spade as the first card
    and an ace as the second card from a standard 52-card deck is 1/52 -/
theorem spade_ace_probability (d : Deck) :
  (Finset.filter (λ c₁ => is_spade c₁) d.cards).card * 
  (Finset.filter (λ c₂ => is_ace c₂) d.cards).card / 
  (d.cards.card * (d.cards.card - 1)) = 1 / 52 := by
  sorry

end NUMINAMATH_CALUDE_spade_ace_probability_l408_40888


namespace NUMINAMATH_CALUDE_A_intersect_B_l408_40883

-- Define set A
def A : Set ℝ := {x : ℝ | |x - 2| ≤ 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- Theorem stating the intersection of A and B
theorem A_intersect_B : A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l408_40883


namespace NUMINAMATH_CALUDE_employee_salary_l408_40805

theorem employee_salary (total_salary : ℝ) (m_percentage : ℝ) (n_salary : ℝ) : 
  total_salary = 616 →
  m_percentage = 1.20 →
  n_salary + m_percentage * n_salary = total_salary →
  n_salary = 280 := by
sorry

end NUMINAMATH_CALUDE_employee_salary_l408_40805


namespace NUMINAMATH_CALUDE_initial_girls_count_l408_40867

theorem initial_girls_count (b g : ℕ) : 
  (3 * (g - 20) = b) →
  (7 * (b - 54) = g - 20) →
  g = 39 := by
sorry

end NUMINAMATH_CALUDE_initial_girls_count_l408_40867


namespace NUMINAMATH_CALUDE_nested_subtraction_simplification_l408_40813

theorem nested_subtraction_simplification (y : ℝ) : 2 - (2 - (2 - (2 - (2 - y)))) = 4 - y := by
  sorry

end NUMINAMATH_CALUDE_nested_subtraction_simplification_l408_40813


namespace NUMINAMATH_CALUDE_player_positions_satisfy_distances_l408_40884

/-- Represents the positions of four soccer players on a number line -/
def PlayerPositions : Fin 4 → ℝ
| 0 => 0
| 1 => 1
| 2 => 4
| 3 => 6

/-- Calculates the distance between two players -/
def distance (i j : Fin 4) : ℝ :=
  |PlayerPositions i - PlayerPositions j|

/-- The set of required pairwise distances -/
def RequiredDistances : Set ℝ := {1, 2, 3, 4, 5, 6}

/-- Theorem stating that the player positions satisfy the required distances -/
theorem player_positions_satisfy_distances :
  ∀ i j : Fin 4, i ≠ j → distance i j ∈ RequiredDistances :=
sorry

end NUMINAMATH_CALUDE_player_positions_satisfy_distances_l408_40884


namespace NUMINAMATH_CALUDE_algae_coverage_day_l408_40841

/-- Represents the coverage of algae in the lake on a given day -/
def algaeCoverage (day : ℕ) : ℚ :=
  1 / 2^(30 - day)

/-- The problem statement -/
theorem algae_coverage_day : ∃ d : ℕ, d ≤ 30 ∧ algaeCoverage d < (1/10) ∧ (1/10) ≤ algaeCoverage (d+1) :=
  sorry

end NUMINAMATH_CALUDE_algae_coverage_day_l408_40841


namespace NUMINAMATH_CALUDE_jake_planting_charge_l408_40861

/-- The hourly rate Jake wants to make -/
def desired_hourly_rate : ℝ := 20

/-- The time it takes to plant flowers in hours -/
def planting_time : ℝ := 2

/-- The amount Jake should charge for planting flowers -/
def planting_charge : ℝ := desired_hourly_rate * planting_time

theorem jake_planting_charge : planting_charge = 40 := by
  sorry

end NUMINAMATH_CALUDE_jake_planting_charge_l408_40861


namespace NUMINAMATH_CALUDE_domino_covering_implies_divisibility_by_three_l408_40846

/-- Represents a domino covering of a square grid -/
structure Covering (n : ℕ) where
  red : Fin (2*n) → Fin (2*n) → Bool
  blue : Fin (2*n) → Fin (2*n) → Bool

/-- Checks if a covering is valid -/
def is_valid_covering (n : ℕ) (c : Covering n) : Prop :=
  ∀ i j, ∃! k l, (c.red i j ∧ c.red k l) ∨ (c.blue i j ∧ c.blue k l)

/-- Represents an integer assignment to each square -/
def Assignment (n : ℕ) := Fin (2*n) → Fin (2*n) → ℤ

/-- Checks if an assignment satisfies the neighbor difference condition -/
def satisfies_difference_condition (n : ℕ) (c : Covering n) (a : Assignment n) : Prop :=
  ∀ i j, ∃ k₁ l₁ k₂ l₂, 
    (c.red i j ∧ c.red k₁ l₁ ∧ c.blue i j ∧ c.blue k₂ l₂) →
    (a i j ≠ 0 ∧ a i j = a k₁ l₁ - a k₂ l₂)

theorem domino_covering_implies_divisibility_by_three (n : ℕ) 
  (h₁ : n > 0)
  (c : Covering n)
  (h₂ : is_valid_covering n c)
  (a : Assignment n)
  (h₃ : satisfies_difference_condition n c a) :
  3 ∣ n :=
sorry

end NUMINAMATH_CALUDE_domino_covering_implies_divisibility_by_three_l408_40846


namespace NUMINAMATH_CALUDE_simplify_expression_l408_40880

theorem simplify_expression : 5 * (18 / 7) * (49 / -54) = -(245 / 9) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l408_40880


namespace NUMINAMATH_CALUDE_chemistry_class_grades_l408_40821

theorem chemistry_class_grades (total_students : ℕ) 
  (prob_A prob_B prob_C prob_F : ℝ) : 
  total_students = 50 →
  prob_A = 0.6 * prob_B →
  prob_C = 1.5 * prob_B →
  prob_F = 0.4 * prob_B →
  prob_A + prob_B + prob_C + prob_F = 1 →
  ⌊total_students * prob_B⌋ = 14 := by
sorry

end NUMINAMATH_CALUDE_chemistry_class_grades_l408_40821


namespace NUMINAMATH_CALUDE_ticket_price_is_six_l408_40807

/-- The price of a concert ticket, given the following conditions:
  * Lana bought 8 tickets for herself and friends
  * Lana bought 2 extra tickets
  * Lana spent $60 in total
-/
def ticket_price : ℚ := by
  -- Define the number of tickets for Lana and friends
  let lana_friends_tickets : ℕ := 8
  -- Define the number of extra tickets
  let extra_tickets : ℕ := 2
  -- Define the total amount spent
  let total_spent : ℚ := 60
  -- Calculate the total number of tickets
  let total_tickets : ℕ := lana_friends_tickets + extra_tickets
  -- Calculate the price per ticket
  exact total_spent / total_tickets
  
-- Prove that the ticket price is $6
theorem ticket_price_is_six : ticket_price = 6 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_is_six_l408_40807


namespace NUMINAMATH_CALUDE_parabola_directrix_l408_40819

/-- Represents a parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ := sorry

/-- Theorem: For a parabola with equation y = -1/8 x^2, its directrix has the equation y = 2 -/
theorem parabola_directrix :
  let p : Parabola := { a := -1/8, b := 0, c := 0 }
  directrix p = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l408_40819


namespace NUMINAMATH_CALUDE_max_crates_on_trip_l408_40830

theorem max_crates_on_trip (crate_weight : ℝ) (max_weight : ℝ) (h1 : crate_weight ≥ 1250) (h2 : max_weight = 6250) :
  ⌊max_weight / crate_weight⌋ = 5 :=
sorry

end NUMINAMATH_CALUDE_max_crates_on_trip_l408_40830


namespace NUMINAMATH_CALUDE_ice_cream_survey_l408_40886

theorem ice_cream_survey (total_people : ℕ) (ice_cream_angle : ℕ) :
  total_people = 620 →
  ice_cream_angle = 198 →
  ⌊(total_people : ℝ) * (ice_cream_angle : ℝ) / 360⌋ = 341 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_survey_l408_40886


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l408_40822

theorem quadratic_roots_relation (s t : ℝ) 
  (hs : 19 * s^2 + 99 * s + 1 = 0)
  (ht : t^2 + 99 * t + 19 = 0)
  (hst : s * t ≠ 1) :
  (s * t + 4 * s + 1) / t = -5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l408_40822


namespace NUMINAMATH_CALUDE_min_value_theorem_l408_40823

theorem min_value_theorem (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_prod : x * y * z = 27) : 
  ∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 27 → 3 * a + 2 * b + c ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l408_40823


namespace NUMINAMATH_CALUDE_yellow_highlighters_count_l408_40870

/-- The number of yellow highlighters in Kaya's teacher's desk -/
def yellow_highlighters : ℕ := 11 - (4 + 5)

/-- The total number of highlighters -/
def total_highlighters : ℕ := 11

/-- The number of pink highlighters -/
def pink_highlighters : ℕ := 4

/-- The number of blue highlighters -/
def blue_highlighters : ℕ := 5

theorem yellow_highlighters_count :
  yellow_highlighters = 2 :=
by sorry

end NUMINAMATH_CALUDE_yellow_highlighters_count_l408_40870


namespace NUMINAMATH_CALUDE_polynomial_product_sum_l408_40863

/-- Given two polynomials in d with coefficients g and h, prove their sum equals 15.5 -/
theorem polynomial_product_sum (g h : ℚ) : 
  (∀ d : ℚ, (8*d^2 - 4*d + g) * (5*d^2 + h*d - 10) = 40*d^4 - 75*d^3 - 90*d^2 + 5*d + 20) →
  g + h = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_sum_l408_40863


namespace NUMINAMATH_CALUDE_boat_speed_l408_40839

/-- The speed of a boat in still water, given its speeds with and against a stream -/
theorem boat_speed (along_stream : ℝ) (against_stream : ℝ) 
  (h1 : along_stream = 11) 
  (h2 : against_stream = 3) : ℝ :=
by
  -- The speed of the boat in still water is 7 km/hr
  sorry

#check boat_speed

end NUMINAMATH_CALUDE_boat_speed_l408_40839


namespace NUMINAMATH_CALUDE_first_rewind_time_l408_40844

theorem first_rewind_time (total_time second_rewind_time first_segment second_segment third_segment : ℕ) 
  (h1 : total_time = 120)
  (h2 : second_rewind_time = 15)
  (h3 : first_segment = 35)
  (h4 : second_segment = 45)
  (h5 : third_segment = 20) :
  total_time - (first_segment + second_segment + third_segment) - second_rewind_time = 5 := by
sorry

end NUMINAMATH_CALUDE_first_rewind_time_l408_40844


namespace NUMINAMATH_CALUDE_coefficient_x_squared_is_70_l408_40872

/-- The coefficient of x^2 in the expansion of (2+x)(1-2x)^5 -/
def coefficient_x_squared : ℤ :=
  2 * (Nat.choose 5 2) * (-2)^2 + (Nat.choose 5 1) * (-2)

/-- Theorem stating that the coefficient of x^2 in the expansion of (2+x)(1-2x)^5 is 70 -/
theorem coefficient_x_squared_is_70 : coefficient_x_squared = 70 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_is_70_l408_40872


namespace NUMINAMATH_CALUDE_smallest_multiple_of_30_and_40_not_16_l408_40879

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_multiple_of_30_and_40_not_16 : 
  (∀ n : ℕ, n > 0 ∧ is_multiple n 30 ∧ is_multiple n 40 ∧ ¬is_multiple n 16 → n ≥ 120) ∧ 
  (is_multiple 120 30 ∧ is_multiple 120 40 ∧ ¬is_multiple 120 16) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_30_and_40_not_16_l408_40879


namespace NUMINAMATH_CALUDE_garden_area_l408_40873

/-- The area of a rectangular garden given specific walking conditions -/
theorem garden_area (length width : ℝ) : 
  length * 30 = 1500 →
  2 * (length + width) * 12 = 1500 →
  length * width = 625 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l408_40873


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_a_l408_40887

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 2| + |x - 3|

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set_f (x : ℝ) : 
  f x ≤ 5 ↔ -4/3 ≤ x ∧ x ≤ 0 := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |a^2 - 3*a| ≤ f x) ↔ -1 ≤ a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_a_l408_40887


namespace NUMINAMATH_CALUDE_max_total_score_is_four_l408_40853

/-- Represents an instructor's scoring for a set of problems -/
structure InstructorScoring :=
  (scores : List ℕ)
  (one_count : ℕ)
  (h_scores : ∀ s ∈ scores, s = 0 ∨ s = 1)
  (h_one_count : one_count = 3)
  (h_one_count_correct : scores.count 1 = one_count)

/-- Calculates the rounded mean of three scores -/
def roundedMean (a b c : ℕ) : ℕ :=
  (a + b + c + 1) / 3

/-- Calculates the total score based on three instructors' scorings -/
def totalScore (i1 i2 i3 : InstructorScoring) : ℕ :=
  List.sum (List.zipWith3 roundedMean i1.scores i2.scores i3.scores)

/-- The main theorem stating that the maximum possible total score is 4 -/
theorem max_total_score_is_four (i1 i2 i3 : InstructorScoring) :
  totalScore i1 i2 i3 ≤ 4 :=
sorry

#check max_total_score_is_four

end NUMINAMATH_CALUDE_max_total_score_is_four_l408_40853


namespace NUMINAMATH_CALUDE_system_solvability_l408_40893

/-- The first equation of the system -/
def equation1 (x y : ℝ) : Prop :=
  (x - 2)^2 + (|y - 1| - 1)^2 = 4

/-- The second equation of the system -/
def equation2 (x y a b : ℝ) : Prop :=
  y = b * |x - 1| + a

/-- The system has a solution for given a and b -/
def has_solution (a b : ℝ) : Prop :=
  ∃ x y, equation1 x y ∧ equation2 x y a b

theorem system_solvability (a : ℝ) :
  (∀ b, has_solution a b) ↔ -Real.sqrt 3 ≤ a ∧ a ≤ 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_system_solvability_l408_40893


namespace NUMINAMATH_CALUDE_absolute_value_expression_l408_40874

theorem absolute_value_expression (x : ℝ) (E : ℝ) :
  x = 10 ∧ 30 - |E| = 26 → E = 4 ∨ E = -4 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l408_40874


namespace NUMINAMATH_CALUDE_conditional_probability_rain_given_east_wind_l408_40845

def east_wind_prob : ℚ := 9/30
def rain_prob : ℚ := 11/30
def both_prob : ℚ := 8/30

theorem conditional_probability_rain_given_east_wind :
  (both_prob / east_wind_prob : ℚ) = 8/9 := by sorry

end NUMINAMATH_CALUDE_conditional_probability_rain_given_east_wind_l408_40845


namespace NUMINAMATH_CALUDE_finns_purchase_theorem_l408_40825

/-- The cost of Finn's purchase given the conditions of the problem -/
def finns_purchase_cost (paper_clip_cost index_card_cost : ℚ) : ℚ :=
  12 * paper_clip_cost + 10 * index_card_cost

/-- The theorem stating the cost of Finn's purchase -/
theorem finns_purchase_theorem :
  ∃ (index_card_cost : ℚ),
    15 * (1.85 : ℚ) + 7 * index_card_cost = 55.40 ∧
    finns_purchase_cost 1.85 index_card_cost = 61.70 := by
  sorry

#eval finns_purchase_cost (1.85 : ℚ) (3.95 : ℚ)

end NUMINAMATH_CALUDE_finns_purchase_theorem_l408_40825


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_less_than_zero_is_true_l408_40891

theorem negation_of_absolute_value_less_than_zero_is_true : 
  ¬(∃ x : ℝ, |x - 1| < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_less_than_zero_is_true_l408_40891


namespace NUMINAMATH_CALUDE_intersection_angle_cosine_l408_40812

/-- The cosine of the angle formed by the foci and an intersection point of an ellipse and hyperbola with common foci -/
theorem intersection_angle_cosine 
  (x y : ℝ) 
  (ellipse_eq : x^2/6 + y^2/2 = 1) 
  (hyperbola_eq : x^2/3 - y^2 = 1) 
  (is_intersection : x^2/6 + y^2/2 = 1 ∧ x^2/3 - y^2 = 1) : 
  ∃ (f₁_x f₁_y f₂_x f₂_y : ℝ), 
    let f₁ := (f₁_x, f₁_y)
    let f₂ := (f₂_x, f₂_y)
    let p := (x, y)
    let v₁ := (x - f₁_x, y - f₁_y)
    let v₂ := (x - f₂_x, y - f₂_y)
    (f₁.1^2/6 + f₁.2^2/2 < 1 ∧ f₂.1^2/6 + f₂.2^2/2 < 1) ∧  -- f₁ and f₂ are inside the ellipse
    (f₁.1^2/3 - f₁.2^2 > 1 ∧ f₂.1^2/3 - f₂.2^2 > 1) ∧      -- f₁ and f₂ are outside the hyperbola
    (v₁.1 * v₂.1 + v₁.2 * v₂.2) / 
    (Real.sqrt (v₁.1^2 + v₁.2^2) * Real.sqrt (v₂.1^2 + v₂.2^2)) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_angle_cosine_l408_40812


namespace NUMINAMATH_CALUDE_lines_are_parallel_l408_40854

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

theorem lines_are_parallel : 
  let line1 : Line := { a := 2, b := -1, c := 7 }
  let line2 : Line := { a := 2, b := -1, c := 1 }
  parallel line1 line2 := by
  sorry

end NUMINAMATH_CALUDE_lines_are_parallel_l408_40854


namespace NUMINAMATH_CALUDE_cos_two_beta_l408_40815

theorem cos_two_beta (α β : Real) 
  (h1 : Real.sin α = Real.cos β) 
  (h2 : Real.sin α * Real.cos β - 2 * Real.cos α * Real.sin β = 1/2) : 
  Real.cos (2 * β) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_cos_two_beta_l408_40815


namespace NUMINAMATH_CALUDE_chessboard_tiling_impossible_l408_40810

/-- Represents a square on the chessboard -/
inductive Square
| White
| Black

/-- Represents the chessboard after removal of two squares -/
def ModifiedChessboard : Type := Fin 62 → Square

/-- A function to check if a tiling with dominoes is valid -/
def IsValidTiling (board : ModifiedChessboard) (tiling : List (Fin 62 × Fin 62)) : Prop :=
  ∀ (pair : Fin 62 × Fin 62), pair ∈ tiling →
    (board pair.1 ≠ board pair.2) ∧ 
    (∀ (i : Fin 62), i ∉ [pair.1, pair.2] → 
      ∀ (other_pair : Fin 62 × Fin 62), other_pair ∈ tiling → i ∉ [other_pair.1, other_pair.2])

theorem chessboard_tiling_impossible :
  ∀ (board : ModifiedChessboard),
    (∃ (white_count black_count : Nat), 
      (white_count + black_count = 62) ∧
      (white_count = 30) ∧ (black_count = 32) ∧
      (∀ (i : Fin 62), (board i = Square.White ↔ i.val < white_count))) →
    ¬∃ (tiling : List (Fin 62 × Fin 62)), IsValidTiling board tiling ∧ tiling.length = 31 :=
by sorry

end NUMINAMATH_CALUDE_chessboard_tiling_impossible_l408_40810


namespace NUMINAMATH_CALUDE_max_pieces_on_board_l408_40803

/-- Represents a piece on the grid -/
inductive Piece
| Red
| Blue

/-- Represents a cell on the grid -/
structure Cell :=
(row : Nat)
(col : Nat)
(piece : Option Piece)

/-- Represents the game board -/
structure Board :=
(cells : List Cell)
(rowCount : Nat)
(colCount : Nat)

/-- Checks if a cell contains a piece -/
def Cell.hasPiece (cell : Cell) : Bool :=
  cell.piece.isSome

/-- Counts the number of pieces on the board -/
def Board.pieceCount (board : Board) : Nat :=
  board.cells.filter Cell.hasPiece |>.length

/-- Checks if a piece sees exactly five pieces of the other color in its row and column -/
def Board.validPiecePlacement (board : Board) (cell : Cell) : Bool :=
  sorry

/-- Checks if all pieces on the board satisfy the placement rule -/
def Board.validBoard (board : Board) : Bool :=
  board.cells.all (Board.validPiecePlacement board)

theorem max_pieces_on_board (board : Board) :
  board.rowCount = 200 ∧ board.colCount = 200 ∧ board.validBoard →
  board.pieceCount ≤ 3800 :=
sorry

end NUMINAMATH_CALUDE_max_pieces_on_board_l408_40803


namespace NUMINAMATH_CALUDE_cos_45_sin_30_product_equation_equivalence_l408_40897

-- Problem 1
theorem cos_45_sin_30_product : 4 * Real.cos (π / 4) * Real.sin (π / 6) = Real.sqrt 2 := by
  sorry

-- Problem 2
theorem equation_equivalence (x : ℝ) : (x + 2) * (x - 3) = 2 * x - 6 ↔ x^2 - 3 * x = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_45_sin_30_product_equation_equivalence_l408_40897


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_thirds_l408_40855

theorem reciprocal_of_negative_two_thirds :
  let x : ℚ := -2/3
  let y : ℚ := -3/2
  (x * y = 1) → y = x⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_thirds_l408_40855


namespace NUMINAMATH_CALUDE_det_2x2_matrix_l408_40871

/-- The determinant of a 2x2 matrix [[5, x], [-3, 4]] is 20 + 3x -/
theorem det_2x2_matrix (x : ℝ) : 
  Matrix.det !![5, x; -3, 4] = 20 + 3 * x := by
  sorry

end NUMINAMATH_CALUDE_det_2x2_matrix_l408_40871


namespace NUMINAMATH_CALUDE_fraction_3x_3x_minus_2_simplest_form_l408_40876

/-- A fraction is in simplest form if its numerator and denominator have no common factors other than 1 -/
def IsSimplestForm (n d : ℤ) : Prop :=
  ∀ k : ℤ, k ∣ n ∧ k ∣ d → k = 1 ∨ k = -1

/-- The fraction 3x/(3x-2) -/
def f (x : ℤ) : ℚ := (3 * x) / (3 * x - 2)

/-- Theorem: The fraction 3x/(3x-2) is in its simplest form -/
theorem fraction_3x_3x_minus_2_simplest_form (x : ℤ) :
  IsSimplestForm (3 * x) (3 * x - 2) :=
sorry

end NUMINAMATH_CALUDE_fraction_3x_3x_minus_2_simplest_form_l408_40876


namespace NUMINAMATH_CALUDE_range_of_a_l408_40862

-- Define the inequality and its solution set
def inequality (a x : ℝ) : Prop := (a - 1) * x > 2
def solution_set (a x : ℝ) : Prop := x < 2 / (a - 1)

-- Theorem stating the range of a
theorem range_of_a (a : ℝ) :
  (∀ x, inequality a x ↔ solution_set a x) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l408_40862


namespace NUMINAMATH_CALUDE_no_real_roots_of_quadratic_l408_40832

theorem no_real_roots_of_quadratic : 
  ¬∃ (x : ℝ), x^2 - 4*x + 8 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_of_quadratic_l408_40832


namespace NUMINAMATH_CALUDE_combined_shape_is_pentahedron_l408_40828

/-- A regular square pyramid -/
structure RegularSquarePyramid :=
  (edge_length : ℝ)
  (edge_length_pos : edge_length > 0)

/-- A regular tetrahedron -/
structure RegularTetrahedron :=
  (edge_length : ℝ)
  (edge_length_pos : edge_length > 0)

/-- The result of combining a regular square pyramid and a regular tetrahedron -/
def CombinedShape (pyramid : RegularSquarePyramid) (tetrahedron : RegularTetrahedron) :=
  pyramid.edge_length = tetrahedron.edge_length

/-- The number of faces in the resulting shape -/
def num_faces (pyramid : RegularSquarePyramid) (tetrahedron : RegularTetrahedron) 
  (h : CombinedShape pyramid tetrahedron) : ℕ := 5

theorem combined_shape_is_pentahedron 
  (pyramid : RegularSquarePyramid) (tetrahedron : RegularTetrahedron) 
  (h : CombinedShape pyramid tetrahedron) : 
  num_faces pyramid tetrahedron h = 5 := by sorry

end NUMINAMATH_CALUDE_combined_shape_is_pentahedron_l408_40828


namespace NUMINAMATH_CALUDE_calculation_proof_l408_40849

theorem calculation_proof (a b : ℝ) (h1 : a = 7) (h2 : b = 3) : 
  ((a^3 + b^3) / (a^2 - a*b + b^2) = 10) ∧ ((a^2 + b^2) / (a + b) = 5.8) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l408_40849


namespace NUMINAMATH_CALUDE_inequality_problem_l408_40877

theorem inequality_problem (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 4*x + 3 < 0 ∧ x^2 - 6*x + 8 < 0) → 2*x^2 - 9*x + m < 0) → 
  m ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l408_40877


namespace NUMINAMATH_CALUDE_sheila_picnic_probability_l408_40809

/-- The probability of Sheila attending the picnic -/
def probability_attend : ℝ := 0.55

/-- The probability of rain -/
def p_rain : ℝ := 0.30

/-- The probability of sunny weather -/
def p_sunny : ℝ := 0.50

/-- The probability of partly cloudy weather -/
def p_partly_cloudy : ℝ := 0.20

/-- The probability Sheila attends if it rains -/
def p_attend_rain : ℝ := 0.15

/-- The probability Sheila attends if it's sunny -/
def p_attend_sunny : ℝ := 0.85

/-- The probability Sheila attends if it's partly cloudy -/
def p_attend_partly_cloudy : ℝ := 0.40

/-- Theorem stating that the probability of Sheila attending the picnic is correct -/
theorem sheila_picnic_probability : 
  probability_attend = p_rain * p_attend_rain + p_sunny * p_attend_sunny + p_partly_cloudy * p_attend_partly_cloudy :=
by sorry

end NUMINAMATH_CALUDE_sheila_picnic_probability_l408_40809


namespace NUMINAMATH_CALUDE_chocolate_bars_per_box_l408_40835

theorem chocolate_bars_per_box 
  (total_bars : ℕ) 
  (total_boxes : ℕ) 
  (h1 : total_bars = 710) 
  (h2 : total_boxes = 142) : 
  total_bars / total_boxes = 5 := by
sorry

end NUMINAMATH_CALUDE_chocolate_bars_per_box_l408_40835


namespace NUMINAMATH_CALUDE_det_invariant_under_row_operation_l408_40833

/-- Given a 2x2 matrix with determinant 7, prove that modifying the first row
    by adding twice the second row doesn't change the determinant. -/
theorem det_invariant_under_row_operation {a b c d : ℝ} 
  (h : a * d - b * c = 7) :
  (a + 2*c) * d - (b + 2*d) * c = 7 := by
  sorry

#check det_invariant_under_row_operation

end NUMINAMATH_CALUDE_det_invariant_under_row_operation_l408_40833


namespace NUMINAMATH_CALUDE_fraction_equality_l408_40860

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 5 / 4) 
  (h2 : s / u = 7 / 8) : 
  (2 * p * s - 3 * q * u) / (5 * q * u - 4 * p * s) = -13 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l408_40860


namespace NUMINAMATH_CALUDE_reciprocal_sum_l408_40838

theorem reciprocal_sum : (1 / (1/4 + 1/5) : ℚ) = 20/9 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_l408_40838


namespace NUMINAMATH_CALUDE_lucy_aquarium_cleaning_l408_40847

/-- The number of aquariums Lucy can clean in a given time period. -/
def aquariums_cleaned (aquariums_per_period : ℚ) (hours : ℚ) : ℚ :=
  aquariums_per_period * hours

/-- Theorem stating how many aquariums Lucy can clean in 24 hours. -/
theorem lucy_aquarium_cleaning :
  let aquariums_per_3hours : ℚ := 2
  let cleaning_period : ℚ := 3
  let working_hours : ℚ := 24
  aquariums_cleaned (aquariums_per_3hours / cleaning_period) working_hours = 16 := by
  sorry

end NUMINAMATH_CALUDE_lucy_aquarium_cleaning_l408_40847


namespace NUMINAMATH_CALUDE_lorelai_ate_180_jellybeans_l408_40837

-- Define the number of jellybeans each person has
def gigi_jellybeans : ℕ := 15
def rory_jellybeans : ℕ := gigi_jellybeans + 30

-- Define the total number of jellybeans both girls have
def total_girls_jellybeans : ℕ := gigi_jellybeans + rory_jellybeans

-- Define the number of jellybeans Lorelai has eaten
def lorelai_jellybeans : ℕ := 3 * total_girls_jellybeans

-- Theorem to prove
theorem lorelai_ate_180_jellybeans : lorelai_jellybeans = 180 := by
  sorry

end NUMINAMATH_CALUDE_lorelai_ate_180_jellybeans_l408_40837


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l408_40878

theorem least_n_satisfying_inequality : 
  (∀ k : ℕ, k > 0 → k < 4 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ 1 / 12) ∧ 
  ((1 : ℚ) / 4 - (1 : ℚ) / 5 < 1 / 12) := by
  sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l408_40878


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l408_40882

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (2 - 3 * i) / (4 + 5 * i + 3 * i^2) = (-1/2 : ℂ) - (1/2 : ℂ) * i :=
by
  -- The proof would go here, but we're skipping it as per instructions
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l408_40882


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_fifty_l408_40801

theorem at_least_one_greater_than_fifty (a₁ a₂ : ℝ) (h : a₁ + a₂ > 100) :
  a₁ > 50 ∨ a₂ > 50 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_fifty_l408_40801


namespace NUMINAMATH_CALUDE_chimpanzee_arrangements_l408_40842

theorem chimpanzee_arrangements : 
  let word := "chimpanzee"
  let total_letters := word.length
  let unique_letters := word.toList.eraseDups.length
  let repeat_letter := 'e'
  let repeat_count := word.toList.filter (· == repeat_letter) |>.length
  (total_letters.factorial / repeat_count.factorial : ℕ) = 1814400 := by
  sorry

end NUMINAMATH_CALUDE_chimpanzee_arrangements_l408_40842


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l408_40892

/-- Given a geometric sequence of positive integers where the first term is 3
    and the fifth term is 243, the seventh term is 2187. -/
theorem geometric_sequence_seventh_term :
  ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 3 →                            -- first term is 3
  a 5 = 243 →                          -- fifth term is 243
  a 7 = 2187 :=                        -- seventh term is 2187
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l408_40892


namespace NUMINAMATH_CALUDE_roots_sum_angle_l408_40840

theorem roots_sum_angle (a : ℝ) (α β : ℝ) : 
  a > 2 → 
  α ∈ Set.Ioo (-π/2) (π/2) →
  β ∈ Set.Ioo (-π/2) (π/2) →
  (Real.tan α)^2 + 3*a*(Real.tan α) + 3*a + 1 = 0 →
  (Real.tan β)^2 + 3*a*(Real.tan β) + 3*a + 1 = 0 →
  α + β = π/4 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_angle_l408_40840


namespace NUMINAMATH_CALUDE_part_one_part_two_l408_40804

/-- Definition of proposition p -/
def p (x a : ℝ) : Prop := (x - 3*a) * (x - a) < 0

/-- Definition of proposition q -/
def q (x : ℝ) : Prop := |x - 3| < 1

/-- Part 1 of the theorem -/
theorem part_one : 
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x < 3) :=
sorry

/-- Part 2 of the theorem -/
theorem part_two :
  ∀ a : ℝ, a > 0 → 
  ((∀ x : ℝ, ¬(p x a) → ¬(q x)) ∧ (∃ x : ℝ, ¬(q x) ∧ p x a)) →
  (4/3 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l408_40804


namespace NUMINAMATH_CALUDE_program_output_correct_l408_40885

/-- The output function of Xiao Wang's program -/
def program_output (n : ℕ+) : ℚ :=
  n / (n^2 + 1)

/-- The theorem stating the correctness of the program output -/
theorem program_output_correct (n : ℕ+) :
  program_output n = n / (n^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_program_output_correct_l408_40885


namespace NUMINAMATH_CALUDE_election_vote_difference_l408_40814

theorem election_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 7500 → 
  candidate_percentage = 35/100 → 
  (total_votes : ℚ) * candidate_percentage - (total_votes : ℚ) * (1 - candidate_percentage) = -2250 := by
  sorry

end NUMINAMATH_CALUDE_election_vote_difference_l408_40814


namespace NUMINAMATH_CALUDE_tyrone_total_money_l408_40899

-- Define the currency values
def one_dollar : ℚ := 1
def ten_dollar : ℚ := 10
def five_dollar : ℚ := 5
def quarter : ℚ := 0.25
def half_dollar : ℚ := 0.5
def dime : ℚ := 0.1
def nickel : ℚ := 0.05
def penny : ℚ := 0.01
def two_dollar : ℚ := 2
def fifty_cent : ℚ := 0.5

-- Define Tyrone's currency counts
def one_dollar_bills : ℕ := 3
def ten_dollar_bills : ℕ := 1
def five_dollar_bills : ℕ := 2
def quarters : ℕ := 26
def half_dollar_coins : ℕ := 5
def dimes : ℕ := 45
def nickels : ℕ := 8
def one_dollar_coins : ℕ := 3
def pennies : ℕ := 56
def two_dollar_bills : ℕ := 2
def fifty_cent_coins : ℕ := 4

-- Define the total amount function
def total_amount : ℚ :=
  (one_dollar_bills : ℚ) * one_dollar +
  (ten_dollar_bills : ℚ) * ten_dollar +
  (five_dollar_bills : ℚ) * five_dollar +
  (quarters : ℚ) * quarter +
  (half_dollar_coins : ℚ) * half_dollar +
  (dimes : ℚ) * dime +
  (nickels : ℚ) * nickel +
  (one_dollar_coins : ℚ) * one_dollar +
  (pennies : ℚ) * penny +
  (two_dollar_bills : ℚ) * two_dollar +
  (fifty_cent_coins : ℚ) * fifty_cent

-- Theorem stating that the total amount is $46.46
theorem tyrone_total_money : total_amount = 46.46 := by
  sorry

end NUMINAMATH_CALUDE_tyrone_total_money_l408_40899


namespace NUMINAMATH_CALUDE_probability_rain_three_days_l408_40818

theorem probability_rain_three_days
  (prob_friday : ℝ)
  (prob_saturday : ℝ)
  (prob_sunday : ℝ)
  (prob_sunday_given_saturday : ℝ)
  (h1 : prob_friday = 0.3)
  (h2 : prob_saturday = 0.5)
  (h3 : prob_sunday = 0.4)
  (h4 : prob_sunday_given_saturday = 0.7)
  : prob_friday * prob_saturday * prob_sunday_given_saturday = 0.105 := by
  sorry

end NUMINAMATH_CALUDE_probability_rain_three_days_l408_40818


namespace NUMINAMATH_CALUDE_goldfish_equality_l408_40896

theorem goldfish_equality (n : ℕ) : (∀ k : ℕ, k < n → 4^(k+1) ≠ 128 * 2^k) ∧ 4^(n+1) = 128 * 2^n ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_equality_l408_40896


namespace NUMINAMATH_CALUDE_math_competition_results_l408_40895

/-- Represents the number of joint math competitions --/
def num_competitions : ℕ := 5

/-- Represents the probability of ranking in the top 20 in each competition --/
def prob_top20 : ℚ := 1/4

/-- Represents the number of top 20 rankings needed to qualify for provincial training --/
def qualify_threshold : ℕ := 2

/-- Models the outcome of a student's participation in the math competitions --/
structure StudentOutcome where
  num_participated : ℕ
  num_top20 : ℕ
  qualified : Bool

/-- Calculates the probability of a specific outcome --/
noncomputable def prob_outcome (outcome : StudentOutcome) : ℚ :=
  sorry

/-- Calculates the probability of qualifying for provincial training --/
noncomputable def prob_qualify : ℚ :=
  sorry

/-- Calculates the expected number of competitions participated in, given qualification or completion --/
noncomputable def expected_num_competitions : ℚ :=
  sorry

/-- Main theorem stating the probabilities and expected value --/
theorem math_competition_results :
  prob_qualify = 67/256 ∧ expected_num_competitions = 65/16 :=
by sorry

end NUMINAMATH_CALUDE_math_competition_results_l408_40895


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_4_mod_7_l408_40852

theorem largest_integer_less_than_100_with_remainder_4_mod_7 : ∃ n : ℕ, n = 95 ∧ 
  (∀ m : ℕ, m < 100 → m % 7 = 4 → m ≤ n) ∧ n < 100 ∧ n % 7 = 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_4_mod_7_l408_40852


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l408_40851

theorem pizza_toppings_combinations : Nat.choose 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l408_40851


namespace NUMINAMATH_CALUDE_distance_from_point_to_line_l408_40848

def point : ℝ × ℝ × ℝ := (2, 3, 4)
def line_point : ℝ × ℝ × ℝ := (4, 6, 5)
def line_direction : ℝ × ℝ × ℝ := (1, 3, -1)

def distance_to_line (p : ℝ × ℝ × ℝ) (a : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_from_point_to_line :
  distance_to_line point line_point line_direction = Real.sqrt 62 / 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_point_to_line_l408_40848


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l408_40811

theorem ratio_of_sum_and_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l408_40811


namespace NUMINAMATH_CALUDE_canMeasureFourLiters_l408_40857

/-- Represents a container with a certain capacity -/
structure Container where
  capacity : ℕ
  current : ℕ
  h : current ≤ capacity

/-- Represents the state of the water measuring system -/
structure WaterSystem where
  small : Container
  large : Container

/-- Checks if the given state has 4 liters in the large container -/
def hasFourLiters (state : WaterSystem) : Prop :=
  state.large.current = 4

/-- Defines the possible operations on the water system -/
inductive Operation
  | FillSmall
  | FillLarge
  | EmptySmall
  | EmptyLarge
  | PourSmallToLarge
  | PourLargeToSmall

/-- Applies an operation to the water system -/
def applyOperation (op : Operation) (state : WaterSystem) : WaterSystem :=
  sorry

/-- Theorem stating that it's possible to measure 4 liters -/
theorem canMeasureFourLiters :
  ∃ (ops : List Operation),
    let initialState : WaterSystem := {
      small := { capacity := 3, current := 0, h := by simp },
      large := { capacity := 5, current := 0, h := by simp }
    }
    let finalState := ops.foldl (fun state op => applyOperation op state) initialState
    hasFourLiters finalState :=
  sorry

end NUMINAMATH_CALUDE_canMeasureFourLiters_l408_40857


namespace NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_is_five_thirds_l408_40802

/-- Represents a square subdivided into smaller squares --/
structure SubdividedSquare where
  -- The side length of the largest square
  side_length : ℝ
  -- The number of subdivisions (levels of recursion)
  subdivisions : ℕ

/-- Calculates the ratio of shaded area to unshaded area in a subdivided square --/
def shaded_to_unshaded_ratio (square : SubdividedSquare) : ℚ :=
  5 / 3

/-- Theorem stating that the ratio of shaded to unshaded area is 5/3 --/
theorem shaded_to_unshaded_ratio_is_five_thirds (square : SubdividedSquare) :
  shaded_to_unshaded_ratio square = 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_is_five_thirds_l408_40802


namespace NUMINAMATH_CALUDE_race_distance_is_17_l408_40817

/-- Represents the relay race with given conditions -/
structure RelayRace where
  totalTime : Real
  sadieTime : Real
  sadieSpeed : Real
  arianaTime : Real
  arianaSpeed : Real
  sarahSpeed : Real

/-- Calculates the total distance of the relay race -/
def totalDistance (race : RelayRace) : Real :=
  let sadieDistance := race.sadieTime * race.sadieSpeed
  let arianaDistance := race.arianaTime * race.arianaSpeed
  let sarahTime := race.totalTime - race.sadieTime - race.arianaTime
  let sarahDistance := sarahTime * race.sarahSpeed
  sadieDistance + arianaDistance + sarahDistance

/-- Theorem stating that the total distance of the given race is 17 miles -/
theorem race_distance_is_17 (race : RelayRace) 
  (h1 : race.totalTime = 4.5)
  (h2 : race.sadieTime = 2)
  (h3 : race.sadieSpeed = 3)
  (h4 : race.arianaTime = 0.5)
  (h5 : race.arianaSpeed = 6)
  (h6 : race.sarahSpeed = 4) :
  totalDistance race = 17 := by
  sorry

#eval totalDistance { totalTime := 4.5, sadieTime := 2, sadieSpeed := 3, arianaTime := 0.5, arianaSpeed := 6, sarahSpeed := 4 }

end NUMINAMATH_CALUDE_race_distance_is_17_l408_40817


namespace NUMINAMATH_CALUDE_lcm_18_24_l408_40881

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l408_40881


namespace NUMINAMATH_CALUDE_negation_of_or_implies_both_false_l408_40898

theorem negation_of_or_implies_both_false (p q : Prop) :
  ¬(p ∨ q) → ¬p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_negation_of_or_implies_both_false_l408_40898


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l408_40868

theorem diophantine_equation_solutions : 
  ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (x y : ℕ), (x, y) ∈ S ↔ 
      (0 < x ∧ 0 < y ∧ x < y ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 2007)) ∧
    S.card = 7 :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l408_40868


namespace NUMINAMATH_CALUDE_bobby_has_two_pizzas_l408_40824

-- Define the number of slices per pizza
def slices_per_pizza : ℕ := 6

-- Define Mrs. Kaplan's number of slices
def kaplan_slices : ℕ := 3

-- Define the ratio of Mrs. Kaplan's slices to Bobby's slices
def kaplan_to_bobby_ratio : ℚ := 1 / 4

-- Define Bobby's number of pizzas
def bobby_pizzas : ℕ := 2

-- Theorem to prove
theorem bobby_has_two_pizzas :
  kaplan_slices = kaplan_to_bobby_ratio * (bobby_pizzas * slices_per_pizza) :=
by sorry

end NUMINAMATH_CALUDE_bobby_has_two_pizzas_l408_40824


namespace NUMINAMATH_CALUDE_range_of_a_l408_40808

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ Real.exp x * (x + a) < 1) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l408_40808


namespace NUMINAMATH_CALUDE_a_10_value_l408_40831

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_10_value (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 = 4 →
  a 6 = 6 →
  a 10 = 9 := by
sorry

end NUMINAMATH_CALUDE_a_10_value_l408_40831


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_range_l408_40864

def integer_range : List ℤ := List.range 12 |>.map (λ i => i - 5)

theorem arithmetic_mean_of_range : (integer_range.sum : ℚ) / integer_range.length = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_range_l408_40864


namespace NUMINAMATH_CALUDE_square_area_of_fourth_side_l408_40875

theorem square_area_of_fourth_side (EF FG GH : ℝ) (h1 : EF^2 = 25) (h2 : FG^2 = 49) (h3 : GH^2 = 64) : 
  ∃ EG EH : ℝ, EG^2 = EF^2 + FG^2 ∧ EH^2 = EG^2 + GH^2 ∧ EH^2 = 138 := by
  sorry

end NUMINAMATH_CALUDE_square_area_of_fourth_side_l408_40875


namespace NUMINAMATH_CALUDE_sector_area_l408_40869

theorem sector_area (r : Real) (θ : Real) (h1 : r = Real.pi) (h2 : θ = 2 * Real.pi / 3) :
  (1 / 2) * r * r * θ = Real.pi^3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l408_40869


namespace NUMINAMATH_CALUDE_map_distance_calculation_l408_40866

/-- Given a map scale and a measured distance on the map, calculate the actual distance in kilometers. -/
theorem map_distance_calculation (scale : ℚ) (map_distance : ℚ) (actual_distance : ℚ) :
  scale = 1 / 1000000 →
  map_distance = 12 →
  actual_distance = map_distance / scale / 100000 →
  actual_distance = 120 := by
  sorry

end NUMINAMATH_CALUDE_map_distance_calculation_l408_40866


namespace NUMINAMATH_CALUDE_unique_solution_system_l408_40850

theorem unique_solution_system :
  ∃! (x y : ℚ), (3 * x - 2 * y = (6 - 2 * x) + (6 - 2 * y)) ∧
                 (x + 3 * y = (2 * x + 1) - (2 * y + 1)) ∧
                 x = 12 / 5 ∧ y = 12 / 25 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l408_40850


namespace NUMINAMATH_CALUDE_bakery_problem_l408_40843

/-- Calculates the number of cookies remaining in the last bag -/
def cookies_in_last_bag (total_cookies : ℕ) (bag_capacity : ℕ) : ℕ :=
  total_cookies % bag_capacity

theorem bakery_problem (total_cookies : ℕ) (choc_chip : ℕ) (oatmeal : ℕ) (sugar : ℕ) 
  (bag_capacity : ℕ) (h1 : total_cookies = choc_chip + oatmeal + sugar) 
  (h2 : choc_chip = 154) (h3 : oatmeal = 86) (h4 : sugar = 52) (h5 : bag_capacity = 16) :
  (cookies_in_last_bag choc_chip bag_capacity = 10) ∧ 
  (cookies_in_last_bag oatmeal bag_capacity = 6) ∧ 
  (cookies_in_last_bag sugar bag_capacity = 4) := by
  sorry

#eval cookies_in_last_bag 154 16  -- Should output 10
#eval cookies_in_last_bag 86 16   -- Should output 6
#eval cookies_in_last_bag 52 16   -- Should output 4

end NUMINAMATH_CALUDE_bakery_problem_l408_40843


namespace NUMINAMATH_CALUDE_existence_of_special_multiple_l408_40800

theorem existence_of_special_multiple (p : ℕ) (hp : p > 1) (hgcd : Nat.gcd p 10 = 1) :
  ∃ n : ℕ, 
    (Nat.digits 10 n).length = p - 2 ∧ 
    (∀ d ∈ Nat.digits 10 n, d = 1 ∨ d = 3) ∧
    p ∣ n :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_multiple_l408_40800


namespace NUMINAMATH_CALUDE_modular_congruence_unique_solution_l408_40806

theorem modular_congruence_unique_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 15893 [ZMOD 16] := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_unique_solution_l408_40806


namespace NUMINAMATH_CALUDE_union_of_sets_l408_40894

theorem union_of_sets (A B : Set ℕ) (h1 : A = {0, 1}) (h2 : B = {2}) :
  A ∪ B = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l408_40894


namespace NUMINAMATH_CALUDE_isabel_cupcakes_l408_40834

/-- The number of cupcakes Todd ate -/
def todd_ate : ℕ := 21

/-- The number of packages Isabel could make after Todd ate some cupcakes -/
def packages : ℕ := 6

/-- The number of cupcakes in each package -/
def cupcakes_per_package : ℕ := 3

/-- The initial number of cupcakes Isabel baked -/
def initial_cupcakes : ℕ := todd_ate + packages * cupcakes_per_package

theorem isabel_cupcakes : initial_cupcakes = 39 := by
  sorry

end NUMINAMATH_CALUDE_isabel_cupcakes_l408_40834


namespace NUMINAMATH_CALUDE_certain_number_value_l408_40829

/-- Represents the number system in the certain country -/
structure CountryNumber where
  value : ℕ

/-- Multiplication operation in the country's number system -/
def country_mul (a b : CountryNumber) : CountryNumber :=
  ⟨a.value * b.value⟩

/-- Division operation in the country's number system -/
def country_div (a b : CountryNumber) : CountryNumber :=
  ⟨a.value / b.value⟩

/-- Equality in the country's number system -/
def country_eq (a b : CountryNumber) : Prop :=
  a.value = b.value

theorem certain_number_value :
  ∀ (eight seven five : CountryNumber),
    country_eq (country_div eight seven) five →
    ∀ (x : CountryNumber),
      country_eq (country_div x ⟨5⟩) ⟨35⟩ →
      country_eq x ⟨175⟩ :=
by sorry

end NUMINAMATH_CALUDE_certain_number_value_l408_40829


namespace NUMINAMATH_CALUDE_chocolate_chip_cookie_price_l408_40889

/-- The price of a box of chocolate chip cookies given the following conditions:
  * Total boxes sold: 1,585
  * Combined value of all boxes: $1,586.75
  * Plain cookies price: $0.75 each
  * Number of plain cookie boxes sold: 793.375
-/
theorem chocolate_chip_cookie_price :
  let total_boxes : ℝ := 1585
  let total_value : ℝ := 1586.75
  let plain_cookie_price : ℝ := 0.75
  let plain_cookie_boxes : ℝ := 793.375
  let chocolate_chip_boxes : ℝ := total_boxes - plain_cookie_boxes
  let chocolate_chip_price : ℝ := (total_value - (plain_cookie_price * plain_cookie_boxes)) / chocolate_chip_boxes
  chocolate_chip_price = 1.2525 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_chip_cookie_price_l408_40889
