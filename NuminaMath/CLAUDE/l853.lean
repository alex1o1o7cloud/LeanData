import Mathlib

namespace NUMINAMATH_CALUDE_square_area_error_l853_85394

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.04)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 8.16 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l853_85394


namespace NUMINAMATH_CALUDE_road_trip_distance_l853_85340

/-- Proves that given the conditions of the road trip, the first day's distance is 200 miles -/
theorem road_trip_distance (total_distance : ℝ) (day1 : ℝ) :
  total_distance = 525 →
  total_distance = day1 + (3/4 * day1) + (1/2 * (day1 + (3/4 * day1))) →
  day1 = 200 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_distance_l853_85340


namespace NUMINAMATH_CALUDE_peter_initial_erasers_l853_85377

-- Define the variables
def initial_erasers : ℕ := sorry
def received_erasers : ℕ := 3
def final_erasers : ℕ := 11

-- State the theorem
theorem peter_initial_erasers : 
  initial_erasers + received_erasers = final_erasers → initial_erasers = 8 := by
  sorry

end NUMINAMATH_CALUDE_peter_initial_erasers_l853_85377


namespace NUMINAMATH_CALUDE_volumes_equal_l853_85367

-- Define the region for V₁
def region_V1 (x y : ℝ) : Prop :=
  (x^2 = 4*y ∨ x^2 = -4*y) ∧ x ≥ -4 ∧ x ≤ 4

-- Define the region for V₂
def region_V2 (x y : ℝ) : Prop :=
  x^2 * y^2 ≤ 16 ∧ x^2 + (y-2)^2 ≥ 4 ∧ x^2 + (y+2)^2 ≥ 4

-- Define the volume of revolution around y-axis
noncomputable def volume_of_revolution (region : ℝ → ℝ → Prop) : ℝ :=
  sorry

-- State the theorem
theorem volumes_equal :
  volume_of_revolution region_V1 = volume_of_revolution region_V2 :=
sorry

end NUMINAMATH_CALUDE_volumes_equal_l853_85367


namespace NUMINAMATH_CALUDE_inequality_solution_count_l853_85352

theorem inequality_solution_count : ∃! (n : ℤ), (n - 2) * (n + 4) * (n - 3) < 0 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_count_l853_85352


namespace NUMINAMATH_CALUDE_yellow_block_weight_l853_85368

theorem yellow_block_weight (green_weight : ℝ) (weight_difference : ℝ) 
  (h1 : green_weight = 0.4)
  (h2 : weight_difference = 0.2) : 
  green_weight + weight_difference = 0.6 :=
by sorry

end NUMINAMATH_CALUDE_yellow_block_weight_l853_85368


namespace NUMINAMATH_CALUDE_min_sequence_length_is_eight_l853_85316

/-- The set S containing elements 1, 2, 3, and 4 -/
def S : Finset ℕ := {1, 2, 3, 4}

/-- A sequence of natural numbers -/
def Sequence := List ℕ

/-- Check if a list contains exactly the elements of a given set -/
def containsExactly (l : List ℕ) (s : Finset ℕ) : Prop :=
  l.toFinset = s

/-- Check if a sequence satisfies the property for all non-empty subsets of S -/
def satisfiesProperty (seq : Sequence) : Prop :=
  ∀ B : Finset ℕ, B ⊆ S → B.Nonempty → 
    ∃ subseq : List ℕ, subseq.length = B.card ∧ 
      seq.Sublist subseq ∧ containsExactly subseq B

/-- The minimum length of a sequence satisfying the property -/
def minSequenceLength : ℕ := 8

/-- Theorem stating that the minimum length of a sequence satisfying the property is 8 -/
theorem min_sequence_length_is_eight :
  (∃ seq : Sequence, seq.length = minSequenceLength ∧ satisfiesProperty seq) ∧
  (∀ seq : Sequence, seq.length < minSequenceLength → ¬satisfiesProperty seq) := by
  sorry


end NUMINAMATH_CALUDE_min_sequence_length_is_eight_l853_85316


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l853_85395

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 2| + |x + 1| ≤ 5} = Set.Icc (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l853_85395


namespace NUMINAMATH_CALUDE_dragon_eye_centering_l853_85362

-- Define a circle with a figure drawn on it
structure FiguredCircle where
  center : ℝ × ℝ
  radius : ℝ
  figure : Set (ℝ × ℝ)

-- Define a point that represents the dragon's eye
def dragonEye (fc : FiguredCircle) : ℝ × ℝ := 
  sorry

-- State the theorem
theorem dragon_eye_centering 
  (c1 c2 : FiguredCircle) 
  (h_congruent : c1.radius = c2.radius) 
  (h_identical_figures : c1.figure = c2.figure) 
  (h_c1_centered : dragonEye c1 = c1.center) 
  (h_c2_not_centered : dragonEye c2 ≠ c2.center) : 
  ∃ (part1 part2 : Set (ℝ × ℝ)), 
    (∃ (c3 : FiguredCircle), 
      c3.radius = c1.radius ∧ 
      c3.figure = c1.figure ∧ 
      dragonEye c3 = c3.center ∧ 
      c3.figure = part1 ∪ part2 ∧ 
      part1 ∩ part2 = ∅ ∧ 
      part1 ∪ part2 = c2.figure) :=
sorry

end NUMINAMATH_CALUDE_dragon_eye_centering_l853_85362


namespace NUMINAMATH_CALUDE_swimming_contest_outcomes_l853_85317

/-- The number of permutations of k elements chosen from a set of n elements -/
def permutations (n k : ℕ) : ℕ := sorry

/-- The number of participants in the swimming contest -/
def num_participants : ℕ := 6

/-- The number of places we're interested in (1st, 2nd, 3rd) -/
def num_places : ℕ := 3

theorem swimming_contest_outcomes :
  permutations num_participants num_places = 120 := by sorry

end NUMINAMATH_CALUDE_swimming_contest_outcomes_l853_85317


namespace NUMINAMATH_CALUDE_total_cost_calculation_l853_85376

def land_acres : ℕ := 30
def land_cost_per_acre : ℕ := 20
def house_cost : ℕ := 120000
def cow_count : ℕ := 20
def cow_cost_per_unit : ℕ := 1000
def chicken_count : ℕ := 100
def chicken_cost_per_unit : ℕ := 5
def solar_installation_hours : ℕ := 6
def solar_installation_cost_per_hour : ℕ := 100
def solar_equipment_cost : ℕ := 6000

theorem total_cost_calculation :
  land_acres * land_cost_per_acre +
  house_cost +
  cow_count * cow_cost_per_unit +
  chicken_count * chicken_cost_per_unit +
  solar_installation_hours * solar_installation_cost_per_hour +
  solar_equipment_cost = 147700 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l853_85376


namespace NUMINAMATH_CALUDE_cos_210_degrees_l853_85301

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l853_85301


namespace NUMINAMATH_CALUDE_circle_center_coordinates_sum_l853_85319

/-- Given a circle with equation x² + y² = -4x + 6y - 12, 
    the sum of the x and y coordinates of its center is 1. -/
theorem circle_center_coordinates_sum : 
  ∀ (x y : ℝ), x^2 + y^2 = -4*x + 6*y - 12 → 
  ∃ (h k : ℝ), (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = 1) ∧ h + k = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_sum_l853_85319


namespace NUMINAMATH_CALUDE_drew_marbles_difference_l853_85360

theorem drew_marbles_difference (drew_initial : ℕ) (marcus_initial : ℕ) (john_initial : ℕ) 
  (h1 : marcus_initial = 45)
  (h2 : john_initial = 70)
  (h3 : ∃ x : ℕ, drew_initial / 4 + marcus_initial = x ∧ drew_initial / 8 + john_initial = x) :
  drew_initial - marcus_initial = 155 :=
by sorry

end NUMINAMATH_CALUDE_drew_marbles_difference_l853_85360


namespace NUMINAMATH_CALUDE_power_of_seven_mod_thousand_l853_85397

theorem power_of_seven_mod_thousand : 7^2023 % 1000 = 343 := by sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_thousand_l853_85397


namespace NUMINAMATH_CALUDE_diamonds_count_l853_85324

/-- Represents the number of gems in a treasure chest. -/
def total_gems : ℕ := 5155

/-- Represents the number of rubies in the treasure chest. -/
def rubies : ℕ := 5110

/-- Theorem stating that the number of diamonds in the treasure chest is 45. -/
theorem diamonds_count : total_gems - rubies = 45 := by
  sorry

end NUMINAMATH_CALUDE_diamonds_count_l853_85324


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l853_85329

-- Define the triangle
def triangle_side_1 : ℝ := 9
def triangle_side_2 : ℝ := 12
def triangle_hypotenuse : ℝ := 15

-- Define the rectangle
def rectangle_length : ℝ := 6

-- Theorem statement
theorem rectangle_perimeter : 
  -- Right triangle condition
  triangle_side_1^2 + triangle_side_2^2 = triangle_hypotenuse^2 →
  -- Rectangle area equals triangle area
  (1/2 * triangle_side_1 * triangle_side_2) = (rectangle_length * (1/2 * triangle_side_1 * triangle_side_2 / rectangle_length)) →
  -- Perimeter of the rectangle is 30
  2 * (rectangle_length + (1/2 * triangle_side_1 * triangle_side_2 / rectangle_length)) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l853_85329


namespace NUMINAMATH_CALUDE_circles_intersection_condition_l853_85322

/-- Two circles in the xy-plane -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y + 1 = 0

def circle2 (x y m : ℝ) : Prop := x^2 + y^2 - 2*x + 2*y - m = 0

/-- The circles intersect -/
def circles_intersect (m : ℝ) : Prop :=
  ∃ x y : ℝ, circle1 x y ∧ circle2 x y m

/-- Theorem stating the condition for the circles to intersect -/
theorem circles_intersection_condition :
  ∀ m : ℝ, circles_intersect m ↔ -1 < m ∧ m < 79 :=
sorry

end NUMINAMATH_CALUDE_circles_intersection_condition_l853_85322


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l853_85336

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron -/
def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 3 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 3,
    PR := 4,
    PS := 5,
    QR := 5,
    QS := Real.sqrt 34,
    RS := Real.sqrt 41
  }
  volume t = 3 := by sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l853_85336


namespace NUMINAMATH_CALUDE_triangle_area_l853_85371

/-- A triangle with sides 8, 15, and 17 has an area of 60 -/
theorem triangle_area : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun a b c area =>
    a = 8 ∧ b = 15 ∧ c = 17 →
    area = 60

/-- The proof of the theorem -/
lemma prove_triangle_area : triangle_area 8 15 17 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l853_85371


namespace NUMINAMATH_CALUDE_mango_cost_theorem_l853_85385

/-- The cost of mangoes in dollars per pound -/
def cost_per_pound (total_cost : ℚ) (total_pounds : ℚ) : ℚ :=
  total_cost / total_pounds

/-- The cost of a given weight of mangoes in dollars -/
def cost_of_weight (cost_per_pound : ℚ) (weight : ℚ) : ℚ :=
  cost_per_pound * weight

theorem mango_cost_theorem (total_cost : ℚ) (total_pounds : ℚ) 
  (h : total_cost = 12 ∧ total_pounds = 10) : 
  cost_of_weight (cost_per_pound total_cost total_pounds) (1/2) = 0.6 := by
  sorry

#eval cost_of_weight (cost_per_pound 12 10) (1/2)

end NUMINAMATH_CALUDE_mango_cost_theorem_l853_85385


namespace NUMINAMATH_CALUDE_national_park_trees_l853_85379

theorem national_park_trees (num_pines : ℕ) (num_redwoods : ℕ) : 
  num_pines = 600 →
  num_redwoods = num_pines + (num_pines * 20 / 100) →
  num_pines + num_redwoods = 1320 := by
sorry

end NUMINAMATH_CALUDE_national_park_trees_l853_85379


namespace NUMINAMATH_CALUDE_dog_spots_l853_85348

/-- The number of spots on dogs problem -/
theorem dog_spots (rover_spots : ℕ) (cisco_spots : ℕ) (granger_spots : ℕ)
  (h1 : rover_spots = 46)
  (h2 : cisco_spots = rover_spots / 2 - 5)
  (h3 : granger_spots = 5 * cisco_spots) :
  granger_spots + cisco_spots = 108 := by
  sorry

end NUMINAMATH_CALUDE_dog_spots_l853_85348


namespace NUMINAMATH_CALUDE_binomial_12_4_l853_85342

theorem binomial_12_4 : Nat.choose 12 4 = 495 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_4_l853_85342


namespace NUMINAMATH_CALUDE_prob_heart_or_king_two_draws_l853_85325

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards that are either hearts or kings -/
def heart_or_king : ℕ := 16

/-- The probability of drawing a card that is neither a heart nor a king -/
def prob_not_heart_or_king : ℚ := (deck_size - heart_or_king) / deck_size

/-- The probability of drawing at least one heart or king in two draws with replacement -/
def prob_at_least_one_heart_or_king : ℚ := 1 - prob_not_heart_or_king ^ 2

theorem prob_heart_or_king_two_draws :
  prob_at_least_one_heart_or_king = 88 / 169 := by
  sorry

end NUMINAMATH_CALUDE_prob_heart_or_king_two_draws_l853_85325


namespace NUMINAMATH_CALUDE_fruit_remaining_l853_85311

-- Define the quantities of fruits picked and eaten
def mike_apples : ℝ := 7.0
def nancy_apples : ℝ := 3.0
def john_apples : ℝ := 5.0
def keith_apples : ℝ := 6.0
def lisa_apples : ℝ := 2.0
def oranges_picked_and_eaten : ℝ := 8.0
def cherries_picked_and_eaten : ℝ := 4.0

-- Define the total apples picked and eaten
def total_apples_picked : ℝ := mike_apples + nancy_apples + john_apples
def total_apples_eaten : ℝ := keith_apples + lisa_apples

-- Theorem statement
theorem fruit_remaining :
  (total_apples_picked - total_apples_eaten = 7.0) ∧
  (oranges_picked_and_eaten - oranges_picked_and_eaten = 0) ∧
  (cherries_picked_and_eaten - cherries_picked_and_eaten = 0) :=
by sorry

end NUMINAMATH_CALUDE_fruit_remaining_l853_85311


namespace NUMINAMATH_CALUDE_cube_operations_impossibility_l853_85358

structure Cube :=
  (vertices : Fin 8 → ℕ)

def initial_state : Cube :=
  { vertices := λ i => if i = 0 then 1 else 0 }

def operation (c : Cube) (e : Fin 8 × Fin 8) : Cube :=
  { vertices := λ i => if i = e.1 ∨ i = e.2 then c.vertices i + 1 else c.vertices i }

def all_equal (c : Cube) : Prop :=
  ∀ i j, c.vertices i = c.vertices j

def all_divisible_by_three (c : Cube) : Prop :=
  ∀ i, c.vertices i % 3 = 0

theorem cube_operations_impossibility :
  (¬ ∃ (ops : List (Fin 8 × Fin 8)), all_equal (ops.foldl operation initial_state)) ∧
  (¬ ∃ (ops : List (Fin 8 × Fin 8)), all_divisible_by_three (ops.foldl operation initial_state)) :=
sorry

end NUMINAMATH_CALUDE_cube_operations_impossibility_l853_85358


namespace NUMINAMATH_CALUDE_chef_cooks_25_wings_l853_85361

/-- The number of additional chicken wings cooked by the chef for a group of friends -/
def additional_chicken_wings (num_friends : ℕ) (pre_cooked : ℕ) (wings_per_person : ℕ) : ℕ :=
  num_friends * wings_per_person - pre_cooked

/-- Theorem stating that for 9 friends, 2 pre-cooked wings, and 3 wings per person, 
    the chef needs to cook 25 additional wings -/
theorem chef_cooks_25_wings : additional_chicken_wings 9 2 3 = 25 := by
  sorry

end NUMINAMATH_CALUDE_chef_cooks_25_wings_l853_85361


namespace NUMINAMATH_CALUDE_smallest_rational_number_l853_85346

theorem smallest_rational_number : ∀ (a b c d : ℚ), 
  a = 0 → b = -1/2 → c = -1/3 → d = 4 →
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_rational_number_l853_85346


namespace NUMINAMATH_CALUDE_no_solution_for_equation_expression_simplifies_to_half_l853_85399

-- Define the domain for x
def X := {x : ℤ | -3 < x ∧ x ≤ 0}

-- Problem 1
theorem no_solution_for_equation :
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 → (2 / (x - 2) - 4 / (x^2 - 4) ≠ 1 / (x + 2)) :=
sorry

-- Problem 2
theorem expression_simplifies_to_half :
  ∀ x ∈ X, x = 0 →
  (x^2 / (x + 1) - x + 1) / ((x + 2) / (x^2 + 2*x + 1)) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_expression_simplifies_to_half_l853_85399


namespace NUMINAMATH_CALUDE_power_five_mod_six_l853_85388

theorem power_five_mod_six : 5^2023 % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_five_mod_six_l853_85388


namespace NUMINAMATH_CALUDE_system_solution_l853_85337

theorem system_solution (x y : ℝ) :
  (2 / (x^2 + y^2) + x^2 * y^2 = 2) ∧
  (x^4 + y^4 + 3 * x^2 * y^2 = 5) ↔
  ((x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l853_85337


namespace NUMINAMATH_CALUDE_football_competition_kicks_l853_85386

/-- Calculates the number of penalty kicks required for a football competition --/
def penalty_kicks (total_players : ℕ) (goalkeepers : ℕ) : ℕ :=
  goalkeepers * (total_players - 1)

/-- Theorem: Given 24 players with 4 goalkeepers, 92 penalty kicks are required --/
theorem football_competition_kicks : penalty_kicks 24 4 = 92 := by
  sorry

end NUMINAMATH_CALUDE_football_competition_kicks_l853_85386


namespace NUMINAMATH_CALUDE_joan_sofa_cost_l853_85389

theorem joan_sofa_cost (joan karl : ℕ) 
  (h1 : joan + karl = 600)
  (h2 : 2 * joan = karl + 90) : 
  joan = 230 := by
sorry

end NUMINAMATH_CALUDE_joan_sofa_cost_l853_85389


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l853_85338

theorem square_perimeter_ratio (s₁ s₂ : ℝ) (h : s₁ > 0 ∧ s₂ > 0) :
  s₂ = 2.5 * s₁ * Real.sqrt 2 / Real.sqrt 2 →
  (4 * s₂) / (4 * s₁) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l853_85338


namespace NUMINAMATH_CALUDE_rationalize_and_simplify_l853_85321

theorem rationalize_and_simplify :
  ∃ (A B C D E : ℤ),
    (B < D) ∧
    (3 : ℝ) / (4 * Real.sqrt 5 + 3 * Real.sqrt 7) = 
      (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    A = 12 ∧ B = 5 ∧ C = -9 ∧ D = 7 ∧ E = 17 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_and_simplify_l853_85321


namespace NUMINAMATH_CALUDE_smallest_n_for_f_greater_than_15_l853_85382

def digit_sum (x : ℚ) : ℕ :=
  sorry

def f (n : ℕ+) : ℕ :=
  digit_sum ((1 : ℚ) / (7 ^ (n : ℕ)))

theorem smallest_n_for_f_greater_than_15 :
  ∀ k : ℕ+, k < 7 → f k ≤ 15 ∧ f 7 > 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_f_greater_than_15_l853_85382


namespace NUMINAMATH_CALUDE_problem_statement_l853_85320

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define the problem statement
theorem problem_statement (a b : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_equation : 2 * (Real.sqrt (log10 a) + Real.sqrt (log10 b)) + log10 (Real.sqrt a) + log10 (Real.sqrt b) = 108)
  (h_int_sqrt_log_a : ∃ m : ℕ, Real.sqrt (log10 a) = m)
  (h_int_sqrt_log_b : ∃ n : ℕ, Real.sqrt (log10 b) = n)
  (h_int_log_sqrt_a : ∃ k : ℕ, log10 (Real.sqrt a) = k)
  (h_int_log_sqrt_b : ∃ l : ℕ, log10 (Real.sqrt b) = l) :
  a * b = 10^116 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l853_85320


namespace NUMINAMATH_CALUDE_least_possible_difference_l853_85356

theorem least_possible_difference (x y z N : ℤ) (h1 : x < y) (h2 : y < z) 
  (h3 : y - x > 5) (h4 : Even x) (h5 : Odd y) (h6 : Odd z) (h7 : ∃ k : ℤ, x = 5 * k) 
  (h8 : y^2 + z^2 = N) (h9 : N > 0) : 
  (∀ w : ℤ, w ≥ 0 → z - x ≥ w + 9) ∧ (z - x = 9) :=
sorry

end NUMINAMATH_CALUDE_least_possible_difference_l853_85356


namespace NUMINAMATH_CALUDE_power_multiplication_equals_512_l853_85314

theorem power_multiplication_equals_512 : 2^3 * 2^6 = 512 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_equals_512_l853_85314


namespace NUMINAMATH_CALUDE_smallest_portion_l853_85359

def bread_distribution (a : ℚ) (d : ℚ) : Prop :=
  -- Total sum is 100
  5 * a + 10 * d = 100 ∧
  -- Sum of largest three portions is 1/7 of sum of smaller two
  (3 * a + 6 * d) = (1/7) * (2 * a + d)

theorem smallest_portion : 
  ∃ (a d : ℚ), bread_distribution a d ∧ a = 5/3 :=
sorry

end NUMINAMATH_CALUDE_smallest_portion_l853_85359


namespace NUMINAMATH_CALUDE_alkaline_probability_l853_85363

/-- Represents the total number of solutions -/
def total_solutions : ℕ := 5

/-- Represents the number of alkaline solutions -/
def alkaline_solutions : ℕ := 2

/-- Represents the probability of selecting an alkaline solution -/
def probability : ℚ := alkaline_solutions / total_solutions

/-- Theorem stating that the probability of selecting an alkaline solution is 2/5 -/
theorem alkaline_probability : probability = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_alkaline_probability_l853_85363


namespace NUMINAMATH_CALUDE_farm_area_calculation_l853_85327

/-- Calculates the area of a rectangular farm given its length and width ratio. -/
def farm_area (length : ℝ) (width_ratio : ℝ) : ℝ :=
  length * (width_ratio * length)

/-- Theorem stating that a rectangular farm with length 0.6 km and width three times its length has an area of 1.08 km². -/
theorem farm_area_calculation :
  farm_area 0.6 3 = 1.08 := by
  sorry

end NUMINAMATH_CALUDE_farm_area_calculation_l853_85327


namespace NUMINAMATH_CALUDE_train_crossing_time_l853_85387

/-- Proves that a train 130 m long, moving at 144 km/hr, takes 3.25 seconds to cross an electric pole -/
theorem train_crossing_time : 
  let train_length : ℝ := 130 -- meters
  let train_speed_kmh : ℝ := 144 -- km/hr
  let train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600 -- Convert km/hr to m/s
  let crossing_time : ℝ := train_length / train_speed_ms
  crossing_time = 3.25 := by
sorry


end NUMINAMATH_CALUDE_train_crossing_time_l853_85387


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l853_85392

theorem arctan_tan_difference (θ : Real) :
  0 ≤ θ ∧ θ ≤ π →
  Real.arctan (Real.tan (5 * π / 12) - 3 * Real.tan (π / 12)) = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arctan_tan_difference_l853_85392


namespace NUMINAMATH_CALUDE_dance_class_girls_l853_85304

theorem dance_class_girls (total : ℕ) (g b : ℚ) : 
  total = 28 →
  g / b = 3 / 4 →
  g + b = total →
  g = 12 := by sorry

end NUMINAMATH_CALUDE_dance_class_girls_l853_85304


namespace NUMINAMATH_CALUDE_total_trips_is_seven_l853_85375

/-- Calculates the number of trips needed to carry a given number of trays -/
def trips_needed (trays_per_trip : ℕ) (num_trays : ℕ) : ℕ :=
  (num_trays + trays_per_trip - 1) / trays_per_trip

/-- Proves that the total number of trips needed is 7 -/
theorem total_trips_is_seven (trays_per_trip : ℕ) (table1_trays : ℕ) (table2_trays : ℕ)
    (h1 : trays_per_trip = 3)
    (h2 : table1_trays = 15)
    (h3 : table2_trays = 5) :
    trips_needed trays_per_trip table1_trays + trips_needed trays_per_trip table2_trays = 7 := by
  sorry

#eval trips_needed 3 15 + trips_needed 3 5

end NUMINAMATH_CALUDE_total_trips_is_seven_l853_85375


namespace NUMINAMATH_CALUDE_unique_k_solution_l853_85354

/-- The function f(x) = x^2 - 2x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- The theorem stating that k = 2 is the only solution -/
theorem unique_k_solution :
  ∃! k : ℝ, ∀ x : ℝ, f (x + k) = x^2 + 2*x + 1 ∧ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_solution_l853_85354


namespace NUMINAMATH_CALUDE_roses_sold_l853_85306

/-- Proves that the number of roses sold is 2, given the initial, picked, and final numbers of roses. -/
theorem roses_sold (initial : ℕ) (picked : ℕ) (final : ℕ) 
  (h1 : initial = 11) 
  (h2 : picked = 32) 
  (h3 : final = 41) : 
  initial - (final - picked) = 2 := by
  sorry

end NUMINAMATH_CALUDE_roses_sold_l853_85306


namespace NUMINAMATH_CALUDE_min_odd_integers_l853_85334

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum1 : a + b = 34)
  (sum2 : a + b + c + d = 51)
  (sum3 : a + b + c + d + e + f = 72) :
  ∃ (odds : Finset ℤ), odds ⊆ {a, b, c, d, e, f} ∧ 
    (∀ x ∈ odds, Odd x) ∧ 
    odds.card = 2 ∧
    (∀ (odds' : Finset ℤ), odds' ⊆ {a, b, c, d, e, f} ∧ 
      (∀ x ∈ odds', Odd x) → odds'.card ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_odd_integers_l853_85334


namespace NUMINAMATH_CALUDE_odot_problem_l853_85369

/-- Definition of the ⊙ operation -/
def odot (x y : ℝ) : ℝ := 2 * x + y

/-- Theorem statement -/
theorem odot_problem (a b : ℝ) (h : odot a (-6 * b) = 4) :
  odot (a - 5 * b) (a + b) = 6 := by
  sorry

end NUMINAMATH_CALUDE_odot_problem_l853_85369


namespace NUMINAMATH_CALUDE_K_bounds_l853_85349

/-- The number of triples in a given system for a natural number n -/
noncomputable def K (n : ℕ) : ℝ := sorry

/-- Theorem stating the bounds for K(n) -/
theorem K_bounds (n : ℕ) : n / 6 - 1 < K n ∧ K n < 2 * n / 9 := by sorry

end NUMINAMATH_CALUDE_K_bounds_l853_85349


namespace NUMINAMATH_CALUDE_alice_profit_l853_85331

/-- Calculates the profit from selling friendship bracelets -/
def calculate_profit (total_bracelets : ℕ) (material_cost : ℚ) (given_away : ℕ) (price_per_bracelet : ℚ) : ℚ :=
  let bracelets_sold := total_bracelets - given_away
  let revenue := (bracelets_sold : ℚ) * price_per_bracelet
  revenue - material_cost

/-- Theorem: Alice's profit from selling friendship bracelets is $8.00 -/
theorem alice_profit :
  calculate_profit 52 3 8 (1/4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_alice_profit_l853_85331


namespace NUMINAMATH_CALUDE_B_is_smallest_l853_85350

def A : ℤ := 32 + 7
def B : ℤ := 3 * 10 + 3
def C : ℤ := 50 - 9

theorem B_is_smallest : B ≤ A ∧ B ≤ C := by
  sorry

end NUMINAMATH_CALUDE_B_is_smallest_l853_85350


namespace NUMINAMATH_CALUDE_marias_cookies_l853_85364

theorem marias_cookies (x : ℕ) : 
  x ≥ 5 →
  (x - 5) % 2 = 0 →
  ((x - 5) / 2 - 2 = 5) →
  x = 19 := by
sorry

end NUMINAMATH_CALUDE_marias_cookies_l853_85364


namespace NUMINAMATH_CALUDE_roots_cubic_sum_l853_85345

theorem roots_cubic_sum (p q : ℝ) : 
  (p^2 - 5*p + 3 = 0) → (q^2 - 5*q + 3 = 0) → (p + q)^3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_roots_cubic_sum_l853_85345


namespace NUMINAMATH_CALUDE_square_area_decrease_l853_85333

theorem square_area_decrease (initial_area : ℝ) (side_decrease_percent : ℝ) 
  (h1 : initial_area = 50) 
  (h2 : side_decrease_percent = 20) : 
  let new_area := initial_area * (1 - side_decrease_percent / 100)^2
  (initial_area - new_area) / initial_area * 100 = 36 := by
sorry

end NUMINAMATH_CALUDE_square_area_decrease_l853_85333


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l853_85312

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a coloring function type
def Coloring := ℤ × ℤ → Color

-- Define a rectangle type
structure Rectangle where
  x1 : ℤ
  y1 : ℤ
  x2 : ℤ
  y2 : ℤ
  h_x : x1 < x2
  h_y : y1 < y2

-- State the theorem
theorem monochromatic_rectangle_exists (c : Coloring) :
  ∃ (r : Rectangle), 
    c (r.x1, r.y1) = c (r.x1, r.y2) ∧
    c (r.x1, r.y1) = c (r.x2, r.y1) ∧
    c (r.x1, r.y1) = c (r.x2, r.y2) :=
by sorry

end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l853_85312


namespace NUMINAMATH_CALUDE_cubic_increasing_iff_a_positive_l853_85308

/-- A cubic function f(x) = ax³ + x is increasing on ℝ if and only if a > 0 -/
theorem cubic_increasing_iff_a_positive (a : ℝ) :
  (∀ x : ℝ, StrictMono (fun x => a * x^3 + x)) ↔ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_increasing_iff_a_positive_l853_85308


namespace NUMINAMATH_CALUDE_cards_distribution_l853_85341

theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 72) 
  (h2 : num_people = 10) : 
  (num_people - (total_cards % num_people)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l853_85341


namespace NUMINAMATH_CALUDE_complex_cube_sum_l853_85339

theorem complex_cube_sum (w z : ℂ) (h1 : Complex.abs (w + z) = 2) (h2 : Complex.abs (w^2 + z^2) = 8) :
  Complex.abs (w^3 + z^3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_sum_l853_85339


namespace NUMINAMATH_CALUDE_smallest_winning_N_for_berta_l853_85373

/-- A game where two players take turns removing marbles from a table. -/
structure MarbleGame where
  initialMarbles : ℕ
  currentMarbles : ℕ
  playerTurn : Bool  -- True for Anna, False for Berta

/-- The rules for removing marbles in a turn -/
def validMove (game : MarbleGame) (k : ℕ) : Prop :=
  k ≥ 1 ∧
  ((k % 2 = 0 ∧ k ≤ game.currentMarbles / 2) ∨
   (k % 2 = 1 ∧ game.currentMarbles / 2 ≤ k ∧ k ≤ game.currentMarbles))

/-- The condition for a winning position -/
def isWinningPosition (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≥ 2 ∧ n = 2^m - 2

/-- The theorem to prove -/
theorem smallest_winning_N_for_berta :
  ∃ N : ℕ,
    N ≥ 100000 ∧
    isWinningPosition N ∧
    (∀ M : ℕ, M ≥ 100000 ∧ M < N → ¬isWinningPosition M) :=
  sorry

end NUMINAMATH_CALUDE_smallest_winning_N_for_berta_l853_85373


namespace NUMINAMATH_CALUDE_fourth_bell_interval_l853_85335

theorem fourth_bell_interval 
  (bell1 bell2 bell3 : ℕ) 
  (h1 : bell1 = 5)
  (h2 : bell2 = 8)
  (h3 : bell3 = 11)
  (h4 : ∃ bell4 : ℕ, Nat.lcm (Nat.lcm (Nat.lcm bell1 bell2) bell3) bell4 = 1320) :
  ∃ bell4 : ℕ, bell4 = 1320 ∧ 
    Nat.lcm (Nat.lcm (Nat.lcm bell1 bell2) bell3) bell4 = 1320 :=
by sorry

end NUMINAMATH_CALUDE_fourth_bell_interval_l853_85335


namespace NUMINAMATH_CALUDE_f_composition_equals_pi_plus_one_l853_85378

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

-- State the theorem
theorem f_composition_equals_pi_plus_one :
  f (f (f (-1))) = Real.pi + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_pi_plus_one_l853_85378


namespace NUMINAMATH_CALUDE_expression_factorization_l853_85398

theorem expression_factorization (x : ℝ) : 3 * x^2 + 12 * x + 12 = 3 * (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l853_85398


namespace NUMINAMATH_CALUDE_smallest_a_for_integer_sqrt_8a_l853_85315

theorem smallest_a_for_integer_sqrt_8a : 
  (∃ (a : ℕ), a > 0 ∧ ∃ (n : ℕ), n^2 = 8*a) → 
  (∀ (a : ℕ), a > 0 → (∃ (n : ℕ), n^2 = 8*a) → a ≥ 2) ∧
  (∃ (n : ℕ), n^2 = 8*2) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_integer_sqrt_8a_l853_85315


namespace NUMINAMATH_CALUDE_siamese_twins_case_l853_85396

/-- Represents a person on trial --/
structure Defendant where
  guilty : Bool

/-- Represents a pair of defendants --/
structure DefendantPair where
  defendant1 : Defendant
  defendant2 : Defendant
  areConjoined : Bool

/-- Represents the judge's decision --/
def judgeDecision (pair : DefendantPair) : Bool :=
  pair.defendant1.guilty ≠ pair.defendant2.guilty → 
  (pair.defendant1.guilty ∨ pair.defendant2.guilty) → 
  pair.areConjoined

theorem siamese_twins_case (pair : DefendantPair) :
  pair.defendant1.guilty ≠ pair.defendant2.guilty →
  (pair.defendant1.guilty ∨ pair.defendant2.guilty) →
  judgeDecision pair →
  pair.areConjoined := by
  sorry


end NUMINAMATH_CALUDE_siamese_twins_case_l853_85396


namespace NUMINAMATH_CALUDE_difference_of_squares_l853_85344

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 10) :
  x^2 - y^2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l853_85344


namespace NUMINAMATH_CALUDE_boxes_per_day_calculation_l853_85318

/-- The number of apples packed in a box -/
def apples_per_box : ℕ := 40

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The total number of apples packed in two weeks -/
def total_apples : ℕ := 24500

/-- The number of fewer apples packed per day in the second week -/
def fewer_apples_per_day : ℕ := 500

/-- The number of full boxes produced per day -/
def boxes_per_day : ℕ := 50

theorem boxes_per_day_calculation :
  boxes_per_day * apples_per_box * days_per_week +
  (boxes_per_day * apples_per_box - fewer_apples_per_day) * days_per_week = total_apples :=
by sorry

end NUMINAMATH_CALUDE_boxes_per_day_calculation_l853_85318


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l853_85307

theorem imaginary_part_of_one_minus_i_squared (i : ℂ) : Complex.im ((1 - i)^2) = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l853_85307


namespace NUMINAMATH_CALUDE_league_games_l853_85366

theorem league_games (n : ℕ) (k : ℕ) (m : ℕ) (h1 : n = 30) (h2 : k = 2) (h3 : m = 6) :
  (n.choose k) * m = 2610 := by
  sorry

end NUMINAMATH_CALUDE_league_games_l853_85366


namespace NUMINAMATH_CALUDE_product_difference_squared_l853_85328

theorem product_difference_squared : 2012 * 2016 - 2014^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_squared_l853_85328


namespace NUMINAMATH_CALUDE_sons_age_is_eighteen_l853_85326

/-- Proves that the son's age is 18 years given the conditions in the problem -/
theorem sons_age_is_eighteen (son_age man_age : ℕ) : 
  man_age = son_age + 20 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 18 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_is_eighteen_l853_85326


namespace NUMINAMATH_CALUDE_magnified_tissue_diameter_l853_85391

/-- Calculates the diameter of a magnified image given the actual diameter and magnification factor. -/
def magnifiedDiameter (actualDiameter : ℝ) (magnificationFactor : ℝ) : ℝ :=
  actualDiameter * magnificationFactor

/-- Proves that for a tissue with actual diameter 0.0003 cm and a microscope with 1000x magnification,
    the magnified image diameter is 0.3 cm. -/
theorem magnified_tissue_diameter :
  magnifiedDiameter 0.0003 1000 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_magnified_tissue_diameter_l853_85391


namespace NUMINAMATH_CALUDE_course_size_l853_85381

theorem course_size (total : ℕ) 
  (grade_A : ℕ → Prop) (grade_B : ℕ → Prop) (grade_C : ℕ → Prop) (grade_D : ℕ → Prop)
  (h1 : ∀ n, grade_A n ↔ n = total / 5)
  (h2 : ∀ n, grade_B n ↔ n = total / 4)
  (h3 : ∀ n, grade_C n ↔ n = total / 2)
  (h4 : ∀ n, grade_D n ↔ n = 25)
  (h5 : ∀ n, n ≤ total → (grade_A n ∨ grade_B n ∨ grade_C n ∨ grade_D n))
  (h6 : ∀ n, (grade_A n → ¬grade_B n ∧ ¬grade_C n ∧ ¬grade_D n) ∧
             (grade_B n → ¬grade_A n ∧ ¬grade_C n ∧ ¬grade_D n) ∧
             (grade_C n → ¬grade_A n ∧ ¬grade_B n ∧ ¬grade_D n) ∧
             (grade_D n → ¬grade_A n ∧ ¬grade_B n ∧ ¬grade_C n)) :
  total = 500 := by
sorry

end NUMINAMATH_CALUDE_course_size_l853_85381


namespace NUMINAMATH_CALUDE_pascal_cycling_trip_l853_85303

theorem pascal_cycling_trip (current_speed : ℝ) (speed_reduction : ℝ) (time_increase : ℝ) 
  (h1 : current_speed = 8)
  (h2 : speed_reduction = 4)
  (h3 : time_increase = 16)
  (h4 : current_speed * (time_increase + t) = (current_speed - speed_reduction) * (time_increase + t + time_increase))
  (h5 : current_speed * t = (current_speed + current_speed / 2) * (time_increase + t - time_increase)) :
  current_speed * t = 256 := by
  sorry

end NUMINAMATH_CALUDE_pascal_cycling_trip_l853_85303


namespace NUMINAMATH_CALUDE_parking_lot_problem_l853_85374

theorem parking_lot_problem (total_vehicles : ℕ) (total_wheels : ℕ) 
  (h1 : total_vehicles = 24) 
  (h2 : total_wheels = 86) : 
  ∃ (cars motorcycles : ℕ), 
    cars + motorcycles = total_vehicles ∧ 
    4 * cars + 3 * motorcycles = total_wheels ∧ 
    motorcycles = 10 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_problem_l853_85374


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_planes_l853_85309

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- Theorem statement
theorem line_parallel_perpendicular_planes 
  (m : Line) (α β : Plane) :
  parallel m α → perpendicular m β → perpendicularPlanes α β :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_planes_l853_85309


namespace NUMINAMATH_CALUDE_integer_list_mean_l853_85357

theorem integer_list_mean (m : ℤ) : 
  let ones := m + 1
  let twos := m + 2
  let threes := m + 3
  let fours := m + 4
  let fives := m + 5
  let total_count := ones + twos + threes + fours + fives
  let sum := ones * 1 + twos * 2 + threes * 3 + fours * 4 + fives * 5
  (sum : ℚ) / total_count = 19 / 6 → m = 9 := by
sorry

end NUMINAMATH_CALUDE_integer_list_mean_l853_85357


namespace NUMINAMATH_CALUDE_fourth_root_fifth_root_approx_l853_85323

theorem fourth_root_fifth_root_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |((32 : ℝ) / 100000)^((1/5 : ℝ) * (1/4 : ℝ)) - 0.6687| < ε := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_fifth_root_approx_l853_85323


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l853_85330

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l853_85330


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l853_85351

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 3) = 7 → x = 46 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l853_85351


namespace NUMINAMATH_CALUDE_roulette_probability_l853_85332

/-- Represents a roulette wheel with sections A, B, and C. -/
structure RouletteWheel where
  probA : ℚ
  probB : ℚ
  probC : ℚ

/-- The sum of probabilities for all sections in a roulette wheel is 1. -/
def validWheel (wheel : RouletteWheel) : Prop :=
  wheel.probA + wheel.probB + wheel.probC = 1

/-- Theorem: Given a valid roulette wheel with probA = 1/4 and probB = 1/2, probC must be 1/4. -/
theorem roulette_probability (wheel : RouletteWheel) 
  (h_valid : validWheel wheel) 
  (h_probA : wheel.probA = 1/4) 
  (h_probB : wheel.probB = 1/2) : 
  wheel.probC = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_roulette_probability_l853_85332


namespace NUMINAMATH_CALUDE_planting_probabilities_l853_85347

structure CropPlanting where
  transition_A : Fin 2 → ℚ
  transition_B : Fin 2 → ℚ
  transition_C : Fin 2 → ℚ

def planting : CropPlanting :=
  { transition_A := ![1/3, 2/3],
    transition_B := ![1/4, 3/4],
    transition_C := ![2/5, 3/5] }

def probability_A_third_given_B_first (p : CropPlanting) : ℚ :=
  p.transition_B 1 * p.transition_C 0

def distribution_X_given_A_first (p : CropPlanting) : Fin 2 → ℚ
  | 0 => p.transition_A 1 * p.transition_C 1 + p.transition_A 0 * p.transition_B 1
  | 1 => p.transition_A 1 * p.transition_C 0 + p.transition_A 0 * p.transition_B 0

def expectation_X_given_A_first (p : CropPlanting) : ℚ :=
  1 * distribution_X_given_A_first p 0 + 2 * distribution_X_given_A_first p 1

theorem planting_probabilities :
  probability_A_third_given_B_first planting = 3/10 ∧
  distribution_X_given_A_first planting 0 = 13/20 ∧
  distribution_X_given_A_first planting 1 = 7/20 ∧
  expectation_X_given_A_first planting = 27/20 := by
  sorry

end NUMINAMATH_CALUDE_planting_probabilities_l853_85347


namespace NUMINAMATH_CALUDE_angle_conversion_l853_85380

theorem angle_conversion (angle : Real) : ∃ (k : Int) (α : Real),
  angle * Real.pi / 180 = 2 * k * Real.pi + α ∧ 0 < α ∧ α < 2 * Real.pi :=
by
  -- The angle -1485° in radians is equal to -1485 * π / 180
  -- We need to prove that this is equal to -10π + 7π/4
  -- and that 7π/4 satisfies the conditions for α
  sorry

#check angle_conversion (-1485)

end NUMINAMATH_CALUDE_angle_conversion_l853_85380


namespace NUMINAMATH_CALUDE_chord_length_is_2_sqrt_2_l853_85365

/-- The circle with center (1, 1) and radius 2 -/
def C : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 = 4}

/-- The line x - y + 2 = 0 -/
def l : Set (ℝ × ℝ) := {p | p.1 - p.2 + 2 = 0}

/-- The length of the chord intercepted by line l on circle C -/
def chord_length : ℝ := sorry

/-- Theorem stating that the chord length is 2√2 -/
theorem chord_length_is_2_sqrt_2 : chord_length = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_chord_length_is_2_sqrt_2_l853_85365


namespace NUMINAMATH_CALUDE_value_of_expression_l853_85370

theorem value_of_expression (x : ℝ) (h : 5 * x - 3 = 7) : 3 * x^2 + 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l853_85370


namespace NUMINAMATH_CALUDE_curve_range_l853_85300

/-- The curve y^2 - xy + 2x + k = 0 passes through the point (a, -a) -/
def passes_through (k a : ℝ) : Prop :=
  (-a)^2 - a * (-a) + 2 * a + k = 0

/-- The range of k values for which the curve passes through (a, -a) for some real a -/
def k_range (k : ℝ) : Prop :=
  ∃ a : ℝ, passes_through k a

theorem curve_range :
  ∀ k : ℝ, k_range k → k ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_curve_range_l853_85300


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_two_equation_l853_85305

theorem unique_solution_sqrt_two_equation (m n : ℤ) :
  (5 + 3 * Real.sqrt 2) ^ m = (3 + 5 * Real.sqrt 2) ^ n ↔ m = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_two_equation_l853_85305


namespace NUMINAMATH_CALUDE_sin_cos_15_deg_l853_85302

theorem sin_cos_15_deg : 
  Real.sin (15 * π / 180) ^ 4 - Real.cos (15 * π / 180) ^ 4 = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_15_deg_l853_85302


namespace NUMINAMATH_CALUDE_dinitrogen_monoxide_weight_is_44_02_l853_85353

/-- The atomic weight of nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in dinitrogen monoxide -/
def nitrogen_count : ℕ := 2

/-- The number of oxygen atoms in dinitrogen monoxide -/
def oxygen_count : ℕ := 1

/-- The molecular weight of dinitrogen monoxide (N2O) in g/mol -/
def dinitrogen_monoxide_weight : ℝ :=
  nitrogen_count * nitrogen_weight + oxygen_count * oxygen_weight

/-- Theorem stating that the molecular weight of dinitrogen monoxide is 44.02 g/mol -/
theorem dinitrogen_monoxide_weight_is_44_02 :
  dinitrogen_monoxide_weight = 44.02 := by
  sorry

end NUMINAMATH_CALUDE_dinitrogen_monoxide_weight_is_44_02_l853_85353


namespace NUMINAMATH_CALUDE_triangle_properties_l853_85310

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  t.c * Real.cos t.B = (2 * t.a - t.b) * Real.cos t.C ∧
  t.c = 2 ∧
  t.a + t.b + t.c = 2 * Real.sqrt 3 + 2

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : TriangleProperties t) :
  t.C = π / 3 ∧ 
  (1/2 * t.a * t.b * Real.sin t.C : ℝ) = 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l853_85310


namespace NUMINAMATH_CALUDE_equation_implications_l853_85390

theorem equation_implications (x y : ℝ) (h : x^2 - 2*x*y + 2*y^2 = 1) :
  (abs x ≤ Real.sqrt 2) ∧ (x^2 + 2*y^2 > 1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_implications_l853_85390


namespace NUMINAMATH_CALUDE_no_integer_solutions_l853_85372

theorem no_integer_solutions : ¬∃ (x y z : ℤ), 
  (x^6 + x^3 + x^3*y + y = 147^157) ∧ 
  (x^3 + x^3*y + y^2 + y + z^9 = 157^147) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l853_85372


namespace NUMINAMATH_CALUDE_davids_physics_marks_l853_85393

/-- Given David's marks in various subjects and his average, prove his marks in Physics --/
theorem davids_physics_marks
  (english : ℕ)
  (mathematics : ℕ)
  (chemistry : ℕ)
  (biology : ℕ)
  (average : ℕ)
  (total_subjects : ℕ)
  (h_english : english = 86)
  (h_mathematics : mathematics = 85)
  (h_chemistry : chemistry = 87)
  (h_biology : biology = 95)
  (h_average : average = 89)
  (h_subjects : total_subjects = 5) :
  ∃ (physics : ℕ), physics = 92 ∧
    average * total_subjects = english + mathematics + physics + chemistry + biology :=
by sorry

end NUMINAMATH_CALUDE_davids_physics_marks_l853_85393


namespace NUMINAMATH_CALUDE_hypotenuse_of_special_triangle_l853_85355

-- Define a right triangle
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  angle_opposite_leg1 : ℝ
  is_right_triangle : leg1^2 + leg2^2 = hypotenuse^2

-- Theorem statement
theorem hypotenuse_of_special_triangle 
  (triangle : RightTriangle)
  (h1 : triangle.leg1 = 15)
  (h2 : triangle.angle_opposite_leg1 = 30 * π / 180) :
  triangle.hypotenuse = 30 :=
sorry

end NUMINAMATH_CALUDE_hypotenuse_of_special_triangle_l853_85355


namespace NUMINAMATH_CALUDE_flower_prices_l853_85313

theorem flower_prices (x y z : ℚ) 
  (eq1 : 3 * x + 7 * y + z = 14)
  (eq2 : 4 * x + 10 * y + z = 16) :
  3 * (x + y + z) = 30 := by
sorry

end NUMINAMATH_CALUDE_flower_prices_l853_85313


namespace NUMINAMATH_CALUDE_oranges_thrown_away_l853_85383

theorem oranges_thrown_away (initial_oranges new_oranges final_oranges : ℕ) : 
  initial_oranges = 31 → new_oranges = 38 → final_oranges = 60 → 
  ∃ thrown_away : ℕ, initial_oranges - thrown_away + new_oranges = final_oranges ∧ thrown_away = 9 :=
by sorry

end NUMINAMATH_CALUDE_oranges_thrown_away_l853_85383


namespace NUMINAMATH_CALUDE_set_operations_l853_85343

def A : Set ℝ := {x | -4 ≤ x ∧ x ≤ -2}
def B : Set ℝ := {x | x + 3 ≥ 0}
def U : Set ℝ := {x | x ≤ -1}

theorem set_operations :
  (A ∩ B = {x | -3 ≤ x ∧ x ≤ -2}) ∧
  (A ∪ B = {x | x ≥ -4}) ∧
  (U \ (A ∩ B) = {x | x < -3 ∨ (-2 < x ∧ x ≤ -1)}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l853_85343


namespace NUMINAMATH_CALUDE_divisibility_problem_l853_85384

theorem divisibility_problem (n : ℕ) (h1 : n > 0) (h2 : 100 % n = 3) (h3 : 197 % n = 3) : n = 97 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l853_85384
