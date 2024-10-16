import Mathlib

namespace NUMINAMATH_CALUDE_cards_distribution_l2239_223942

theorem cards_distribution (total_cards : ℕ) (num_people : ℕ) (h1 : total_cards = 60) (h2 : num_people = 9) :
  let cards_per_person : ℕ := total_cards / num_people
  let remaining_cards : ℕ := total_cards % num_people
  let people_with_extra : ℕ := remaining_cards
  (num_people - people_with_extra) = 3 :=
by sorry

end NUMINAMATH_CALUDE_cards_distribution_l2239_223942


namespace NUMINAMATH_CALUDE_problem_statement_l2239_223931

theorem problem_statement (a b : ℝ) :
  (∀ x y : ℝ, (2 * x^2 + a * x - y + 6) - (b * x^2 - 3 * x + 5 * y - 1) = -6 * y + 7) →
  a^2 + b^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2239_223931


namespace NUMINAMATH_CALUDE_evaluate_expression_l2239_223935

theorem evaluate_expression : (2^2010 * 3^2012) / 6^2011 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2239_223935


namespace NUMINAMATH_CALUDE_div_fraction_equality_sum_fraction_equality_l2239_223918

-- Define variables
variable (a b : ℝ)

-- Assume a ≠ b and a ≠ 0 to avoid division by zero
variable (h1 : a ≠ b) (h2 : a ≠ 0)

-- Theorem 1
theorem div_fraction_equality : (4 * b^3 / a) / (2 * b / a^2) = 2 * a * b^2 := by sorry

-- Theorem 2
theorem sum_fraction_equality : a^2 / (a - b) + b^2 / (a - b) - 2 * a * b / (a - b) = a - b := by sorry

end NUMINAMATH_CALUDE_div_fraction_equality_sum_fraction_equality_l2239_223918


namespace NUMINAMATH_CALUDE_afternoon_bags_count_l2239_223943

def morning_bags : ℕ := 29
def bag_weight : ℕ := 7
def total_weight : ℕ := 322

def afternoon_bags : ℕ := (total_weight - morning_bags * bag_weight) / bag_weight

theorem afternoon_bags_count : afternoon_bags = 17 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_bags_count_l2239_223943


namespace NUMINAMATH_CALUDE_dans_age_proof_l2239_223937

/-- Dan's present age in years -/
def dans_present_age : ℕ := 16

/-- Theorem stating that Dan's present age satisfies the given condition -/
theorem dans_age_proof :
  dans_present_age + 16 = 4 * (dans_present_age - 8) :=
by sorry

end NUMINAMATH_CALUDE_dans_age_proof_l2239_223937


namespace NUMINAMATH_CALUDE_bob_spending_theorem_l2239_223996

def monday_spending (initial_amount : ℚ) : ℚ := initial_amount / 2

def tuesday_spending (monday_remainder : ℚ) : ℚ := monday_remainder / 5

def wednesday_spending (tuesday_remainder : ℚ) : ℚ := tuesday_remainder * 3 / 8

def final_amount (initial_amount : ℚ) : ℚ :=
  let monday_remainder := initial_amount - monday_spending initial_amount
  let tuesday_remainder := monday_remainder - tuesday_spending monday_remainder
  tuesday_remainder - wednesday_spending tuesday_remainder

theorem bob_spending_theorem :
  final_amount 80 = 20 := by
  sorry

end NUMINAMATH_CALUDE_bob_spending_theorem_l2239_223996


namespace NUMINAMATH_CALUDE_weight_gain_difference_l2239_223947

def weight_gain_problem (orlando_gain jose_gain fernando_gain : ℕ) : Prop :=
  orlando_gain = 5 ∧
  jose_gain = 2 * orlando_gain + 2 ∧
  fernando_gain < jose_gain / 2 ∧
  orlando_gain + jose_gain + fernando_gain = 20

theorem weight_gain_difference (orlando_gain jose_gain fernando_gain : ℕ) 
  (h : weight_gain_problem orlando_gain jose_gain fernando_gain) :
  jose_gain / 2 - fernando_gain = 3 := by
  sorry

end NUMINAMATH_CALUDE_weight_gain_difference_l2239_223947


namespace NUMINAMATH_CALUDE_floor_sum_eval_l2239_223913

theorem floor_sum_eval : ⌊(-7/4 : ℚ)⌋ + ⌊(-3/2 : ℚ)⌋ - ⌊(-5/3 : ℚ)⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_eval_l2239_223913


namespace NUMINAMATH_CALUDE_handshake_arrangements_l2239_223945

/-- The number of ways to arrange 10 people into two rings of 5, where each person in a ring is connected to 3 others -/
def M : ℕ := sorry

/-- The number of ways to select 5 people from 10 -/
def choose_five_from_ten : ℕ := sorry

/-- The number of arrangements within a ring of 5 -/
def ring_arrangements : ℕ := sorry

theorem handshake_arrangements :
  M = choose_five_from_ten * ring_arrangements * ring_arrangements ∧
  M % 1000 = 288 := by sorry

end NUMINAMATH_CALUDE_handshake_arrangements_l2239_223945


namespace NUMINAMATH_CALUDE_equal_distances_l2239_223953

def circular_distance (n : ℕ) (a b : ℕ) : ℕ :=
  (b - a + n) % n

theorem equal_distances : ∃ n : ℕ, 
  n > 0 ∧ 
  circular_distance n 31 7 = circular_distance n 31 14 ∧ 
  n = 41 := by
  sorry

end NUMINAMATH_CALUDE_equal_distances_l2239_223953


namespace NUMINAMATH_CALUDE_sum_is_composite_l2239_223949

theorem sum_is_composite (a b c d : ℕ+) 
  (h : a^2 + b^2 + a*b = c^2 + d^2 + c*d) : 
  ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ (a : ℕ) + b + c + d = k * m :=
sorry

end NUMINAMATH_CALUDE_sum_is_composite_l2239_223949


namespace NUMINAMATH_CALUDE_combined_shoe_size_l2239_223973

-- Define Jasmine's shoe size
def jasmine_size : ℕ := 7

-- Define the relationship between Alexa's and Jasmine's shoe sizes
def alexa_size : ℕ := 2 * jasmine_size

-- Define the combined shoe size
def combined_size : ℕ := jasmine_size + alexa_size

-- Theorem to prove
theorem combined_shoe_size : combined_size = 21 := by
  sorry

end NUMINAMATH_CALUDE_combined_shoe_size_l2239_223973


namespace NUMINAMATH_CALUDE_tan_alpha_for_point_l2239_223959

/-- If the terminal side of angle α passes through the point (-4, -3), then tan α = 3/4 -/
theorem tan_alpha_for_point (α : Real) : 
  (∃ (t : Real), t > 0 ∧ t * Real.cos α = -4 ∧ t * Real.sin α = -3) → 
  Real.tan α = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_for_point_l2239_223959


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l2239_223999

-- Define the circles and line
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 2*y + 7 = 0
def l (x : ℝ) : Prop := x = 1

-- Define the intersection points
def intersection_points (x y : ℝ) : Prop := C₁ x y ∧ C₂ x y

-- Define the line y = x
def y_eq_x (x y : ℝ) : Prop := y = x

-- Define circle C₃
def C₃ (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

theorem circle_intersection_theorem :
  (∀ x y, intersection_points x y → l x) ∧
  (∃ x₀ y₀, C₃ x₀ y₀ ∧ y_eq_x x₀ y₀ ∧ (∀ x y, intersection_points x y → C₃ x y)) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l2239_223999


namespace NUMINAMATH_CALUDE_raven_current_age_l2239_223921

/-- Represents a person's age -/
structure Person where
  age : ℕ

/-- The current ages of Raven and Phoebe -/
def raven_phoebe_ages : Person × Person → Prop
  | (raven, phoebe) => 
    -- In 5 years, Raven will be 4 times as old as Phoebe
    raven.age + 5 = 4 * (phoebe.age + 5) ∧
    -- Phoebe is currently 10 years old
    phoebe.age = 10

/-- Theorem stating Raven's current age -/
theorem raven_current_age : 
  ∀ (raven phoebe : Person), raven_phoebe_ages (raven, phoebe) → raven.age = 55 := by
  sorry

end NUMINAMATH_CALUDE_raven_current_age_l2239_223921


namespace NUMINAMATH_CALUDE_number_of_pupils_in_class_number_of_pupils_in_class_is_correct_l2239_223915

/-- The number of pupils in a class, given an error in mark entry and its effect on the class average. -/
theorem number_of_pupils_in_class : ℕ :=
  let incorrect_mark : ℕ := 73
  let correct_mark : ℕ := 63
  let average_increase : ℚ := 1/2
  20

/-- Proof that the number of pupils in the class is correct. -/
theorem number_of_pupils_in_class_is_correct (n : ℕ) 
  (h1 : n = number_of_pupils_in_class)
  (h2 : (incorrect_mark - correct_mark : ℚ) / n = average_increase) : 
  n = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pupils_in_class_number_of_pupils_in_class_is_correct_l2239_223915


namespace NUMINAMATH_CALUDE_gcd_7163_209_l2239_223961

theorem gcd_7163_209 : Nat.gcd 7163 209 = 19 := by
  have h1 : 7163 = 209 * 34 + 57 := by sorry
  have h2 : 209 = 57 * 3 + 38 := by sorry
  have h3 : 57 = 38 * 1 + 19 := by sorry
  have h4 : 38 = 19 * 2 := by sorry
  sorry

#check gcd_7163_209

end NUMINAMATH_CALUDE_gcd_7163_209_l2239_223961


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l2239_223927

theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) :
  area = 300 ∧ d2 = 20 ∧ area = (d1 * d2) / 2 → d1 = 30 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l2239_223927


namespace NUMINAMATH_CALUDE_average_equation_solution_l2239_223907

theorem average_equation_solution (x : ℝ) : 
  ((2 * x + 12 + 5 * x^2 + 3 * x + 1 + 3 * x + 14) / 3 = 6 * x^2 + x - 21) ↔ 
  (x = (5 + Real.sqrt 4705) / 26 ∨ x = (5 - Real.sqrt 4705) / 26) :=
by sorry

end NUMINAMATH_CALUDE_average_equation_solution_l2239_223907


namespace NUMINAMATH_CALUDE_original_average_age_of_class_l2239_223920

theorem original_average_age_of_class (A : ℝ) : 
  (12 : ℝ) * A + (12 : ℝ) * 32 = (24 : ℝ) * (A - 4) → A = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_average_age_of_class_l2239_223920


namespace NUMINAMATH_CALUDE_unique_divisor_sum_function_l2239_223962

theorem unique_divisor_sum_function :
  ∃! f : ℕ → ℕ, ∀ m n : ℕ, (f m + f n) ∣ (m + n) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_sum_function_l2239_223962


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l2239_223905

/-- Given a circle and a moving chord, prove the trajectory of the chord's midpoint -/
theorem midpoint_trajectory (x y : ℝ) :
  (∀ a b : ℝ, a^2 + b^2 = 25 → (x - a)^2 + (y - b)^2 ≤ 9) →
  x^2 + y^2 = 16 :=
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l2239_223905


namespace NUMINAMATH_CALUDE_smallest_with_twelve_divisors_l2239_223981

/-- A function that returns the number of positive integer divisors of a given positive integer -/
def numberOfDivisors (n : ℕ+) : ℕ := sorry

/-- A function that checks if a given positive integer has exactly 12 positive integer divisors -/
def hasTwelveDivisors (n : ℕ+) : Prop :=
  numberOfDivisors n = 12

/-- Theorem stating that 108 is the smallest positive integer with exactly 12 positive integer divisors -/
theorem smallest_with_twelve_divisors :
  (∀ m : ℕ+, m < 108 → ¬(hasTwelveDivisors m)) ∧ hasTwelveDivisors 108 := by
  sorry

end NUMINAMATH_CALUDE_smallest_with_twelve_divisors_l2239_223981


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2239_223986

/-- A line y = kx is tangent to the circle x^2 + y^2 - 6x + 8 = 0 at a point in the fourth quadrant -/
def is_tangent (k : ℝ) : Prop :=
  ∃ (x y : ℝ),
    y = k * x ∧
    x^2 + y^2 - 6*x + 8 = 0 ∧
    x > 0 ∧ y < 0 ∧
    ∀ (x' y' : ℝ), y' = k * x' → (x' - x)^2 + (y' - y)^2 > 0

theorem tangent_line_to_circle (k : ℝ) :
  is_tangent k → k = -Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2239_223986


namespace NUMINAMATH_CALUDE_product_of_integers_l2239_223987

theorem product_of_integers (a b : ℕ+) : 
  a + b = 30 → 3 * (a * b) + 4 * a = 5 * b + 318 → a * b = 56 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l2239_223987


namespace NUMINAMATH_CALUDE_pet_store_cages_l2239_223914

def cages_used (total_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) : ℕ :=
  let remaining_puppies := total_puppies - sold_puppies
  (remaining_puppies / puppies_per_cage) + if remaining_puppies % puppies_per_cage > 0 then 1 else 0

theorem pet_store_cages :
  cages_used 1700 621 26 = 42 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l2239_223914


namespace NUMINAMATH_CALUDE_cyclic_triples_count_l2239_223909

/-- Represents a round-robin tournament -/
structure Tournament where
  n : ℕ  -- number of teams
  wins : Fin n → ℕ  -- number of wins for each team
  losses : Fin n → ℕ  -- number of losses for each team

/-- The number of sets of three teams with a cyclic winning relationship -/
def cyclic_triples (t : Tournament) : ℕ := sorry

/-- Main theorem about the number of cyclic triples in the specific tournament -/
theorem cyclic_triples_count (t : Tournament) 
  (h1 : t.n > 0)
  (h2 : ∀ i : Fin t.n, t.wins i = 9)
  (h3 : ∀ i : Fin t.n, t.losses i = 9)
  (h4 : ∀ i j : Fin t.n, i ≠ j → (t.wins i + t.losses i = t.wins j + t.losses j)) :
  cyclic_triples t = 969 := by sorry

end NUMINAMATH_CALUDE_cyclic_triples_count_l2239_223909


namespace NUMINAMATH_CALUDE_isosceles_points_count_l2239_223994

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the property of being acute
def isAcute (t : Triangle) : Prop := sorry

-- Define the ordering of side lengths
def sideOrdering (t : Triangle) : Prop := sorry

-- Define the property of a point P making isosceles triangles with AB and BC
def makesIsosceles (P : ℝ × ℝ) (t : Triangle) : Prop := sorry

-- The main theorem
theorem isosceles_points_count (t : Triangle) : 
  isAcute t → sideOrdering t → ∃! (points : Finset (ℝ × ℝ)), 
    Finset.card points = 15 ∧ 
    ∀ P ∈ points, makesIsosceles P t := by
  sorry

end NUMINAMATH_CALUDE_isosceles_points_count_l2239_223994


namespace NUMINAMATH_CALUDE_complement_of_A_when_a_is_one_range_of_a_given_subset_l2239_223995

-- Define set A
def A (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x - 8*a^2 ≤ 0}

-- Part I
theorem complement_of_A_when_a_is_one :
  Set.compl (A 1) = {x | x < -2 ∨ x > 4} := by sorry

-- Part II
theorem range_of_a_given_subset (a : ℝ) (h1 : a > 0) (h2 : Set.Ioo (-1 : ℝ) 1 ⊆ A a) :
  a ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_when_a_is_one_range_of_a_given_subset_l2239_223995


namespace NUMINAMATH_CALUDE_certain_event_three_people_two_groups_l2239_223969

theorem certain_event_three_people_two_groups : 
  ∀ (group1 group2 : Finset Nat), 
  (group1 ∪ group2).card = 3 → 
  group1 ∩ group2 = ∅ → 
  group1 ≠ ∅ → 
  group2 ≠ ∅ → 
  (group1.card = 2 ∨ group2.card = 2) :=
sorry

end NUMINAMATH_CALUDE_certain_event_three_people_two_groups_l2239_223969


namespace NUMINAMATH_CALUDE_cookie_jar_problem_l2239_223977

theorem cookie_jar_problem (x : ℕ) : 
  (x - 1 = (x + 5) / 2) → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_cookie_jar_problem_l2239_223977


namespace NUMINAMATH_CALUDE_square_of_complex_fraction_l2239_223952

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem square_of_complex_fraction : (2 * i / (1 - i)) ^ 2 = -2 * i := by
  sorry

end NUMINAMATH_CALUDE_square_of_complex_fraction_l2239_223952


namespace NUMINAMATH_CALUDE_investment_time_calculation_l2239_223912

/-- Investment time calculation for partners P and Q -/
theorem investment_time_calculation 
  (investment_ratio_p : ℝ) 
  (investment_ratio_q : ℝ)
  (profit_ratio_p : ℝ) 
  (profit_ratio_q : ℝ)
  (time_q : ℝ) :
  investment_ratio_p = 7 →
  investment_ratio_q = 5.00001 →
  profit_ratio_p = 7.00001 →
  profit_ratio_q = 10 →
  time_q = 9.999965714374696 →
  ∃ (time_p : ℝ), abs (time_p - 50) < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_investment_time_calculation_l2239_223912


namespace NUMINAMATH_CALUDE_min_box_value_l2239_223900

/-- Given the equation (cx+d)(dx+c) = 15x^2 + ◻x + 15, where c, d, and ◻ are distinct integers,
    the minimum possible value of ◻ is 34. -/
theorem min_box_value (c d box : ℤ) : 
  (c * d + c = 15) →
  (c + d = box) →
  (c ≠ d) ∧ (c ≠ box) ∧ (d ≠ box) →
  (∀ (c' d' box' : ℤ), (c' * d' + c' = 15) → (c' + d' = box') → 
    (c' ≠ d') ∧ (c' ≠ box') ∧ (d' ≠ box') → box ≤ box') →
  box = 34 :=
by sorry

end NUMINAMATH_CALUDE_min_box_value_l2239_223900


namespace NUMINAMATH_CALUDE_special_cubes_in_4x5x6_prism_l2239_223976

/-- Represents a rectangular prism with integer dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of unit cubes in a rectangular prism that have
    either exactly one face on the surface or no faces on the surface -/
def count_special_cubes (prism : RectangularPrism) : ℕ :=
  let interior_cubes := (prism.length - 2) * (prism.width - 2) * (prism.height - 2)
  let one_face_cubes := 2 * ((prism.width - 2) * (prism.height - 2) +
                             (prism.length - 2) * (prism.height - 2) +
                             (prism.length - 2) * (prism.width - 2))
  interior_cubes + one_face_cubes

/-- The main theorem stating that a 4x5x6 prism has 76 special cubes -/
theorem special_cubes_in_4x5x6_prism :
  count_special_cubes ⟨4, 5, 6⟩ = 76 := by
  sorry

end NUMINAMATH_CALUDE_special_cubes_in_4x5x6_prism_l2239_223976


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l2239_223960

/-- The number of ways to partition n indistinguishable objects into at most k parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 11 ways to partition 7 indistinguishable objects into at most 4 parts -/
theorem seven_balls_four_boxes : partition_count 7 4 = 11 := by sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l2239_223960


namespace NUMINAMATH_CALUDE_factor_implies_b_value_l2239_223946

theorem factor_implies_b_value (a b : ℤ) : 
  (∃ (c : ℤ), (X^2 - 2*X - 1) * (a*X - c) = a*X^3 + b*X^2 + 2) → b = -6 :=
by sorry

end NUMINAMATH_CALUDE_factor_implies_b_value_l2239_223946


namespace NUMINAMATH_CALUDE_opposite_of_sqrt_difference_l2239_223993

theorem opposite_of_sqrt_difference : -(Real.sqrt 2 - Real.sqrt 3) = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt_difference_l2239_223993


namespace NUMINAMATH_CALUDE_director_selection_probability_l2239_223934

def total_actors : ℕ := 5
def golden_rooster_winners : ℕ := 2
def hundred_flowers_winners : ℕ := 3

def probability_select_2_golden_1_hundred : ℚ := 3 / 10

theorem director_selection_probability :
  (golden_rooster_winners.choose 2 * hundred_flowers_winners) / 
  (total_actors.choose 3) = probability_select_2_golden_1_hundred := by
  sorry

end NUMINAMATH_CALUDE_director_selection_probability_l2239_223934


namespace NUMINAMATH_CALUDE_aubriella_poured_gallons_l2239_223904

/-- Proves that Aubriella has poured 18 gallons into the fish tank -/
theorem aubriella_poured_gallons
  (tank_capacity : ℕ)
  (remaining_gallons : ℕ)
  (seconds_per_gallon : ℕ)
  (pouring_time_minutes : ℕ)
  (h1 : tank_capacity = 50)
  (h2 : remaining_gallons = 32)
  (h3 : seconds_per_gallon = 20)
  (h4 : pouring_time_minutes = 6) :
  tank_capacity - remaining_gallons = 18 :=
by sorry

end NUMINAMATH_CALUDE_aubriella_poured_gallons_l2239_223904


namespace NUMINAMATH_CALUDE_triangle_inequality_altitudes_l2239_223910

/-- Triangle inequality for side lengths and altitudes -/
theorem triangle_inequality_altitudes (a b c h_a h_b h_c Δ : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_ha : 0 < h_a) (h_pos_hb : 0 < h_b) (h_pos_hc : 0 < h_c)
  (h_area : 0 < Δ)
  (h_area_a : Δ = (a * h_a) / 2)
  (h_area_b : Δ = (b * h_b) / 2)
  (h_area_c : Δ = (c * h_c) / 2) :
  a * h_b + b * h_c + c * h_a ≥ 6 * Δ := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_altitudes_l2239_223910


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2239_223928

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : 6 * (2 * x - 1) - 3 * (5 + 2 * x) = 6 * x - 21 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (a : ℝ) : (4 * a^2 - 8 * a - 9) + 3 * (2 * a^2 - 2 * a - 5) = 10 * a^2 - 14 * a - 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2239_223928


namespace NUMINAMATH_CALUDE_probability_closer_to_five_than_one_l2239_223925

noncomputable def probability_closer_to_five (a b c : ℝ) : ℝ :=
  let midpoint := (a + c) / 2
  let favorable_length := b - midpoint
  let total_length := b - 0
  favorable_length / total_length

theorem probability_closer_to_five_than_one :
  probability_closer_to_five 1 6 5 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_probability_closer_to_five_than_one_l2239_223925


namespace NUMINAMATH_CALUDE_green_turtles_count_l2239_223968

theorem green_turtles_count (total : ℕ) (h : total = 3200) :
  ∃ (green hawksbill : ℕ),
    green + hawksbill = total ∧
    hawksbill = 2 * green ∧
    green = 1066 :=
by
  sorry

end NUMINAMATH_CALUDE_green_turtles_count_l2239_223968


namespace NUMINAMATH_CALUDE_cube_space_diagonal_probability_l2239_223902

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The number of space diagonals in a cube -/
def space_diagonals : ℕ := 4

/-- The number of ways to choose 2 vertices from a cube -/
def total_choices : ℕ := Nat.choose cube_vertices 2

/-- The probability of selecting two vertices that are endpoints of a space diagonal -/
def space_diagonal_probability : ℚ := space_diagonals / total_choices

theorem cube_space_diagonal_probability :
  space_diagonal_probability = 1/7 := by sorry

end NUMINAMATH_CALUDE_cube_space_diagonal_probability_l2239_223902


namespace NUMINAMATH_CALUDE_rectangle_length_equals_square_side_l2239_223929

/-- The length of a rectangle with width 3 cm and area equal to a square with side length 3 cm is 3 cm. -/
theorem rectangle_length_equals_square_side : 
  ∀ (length : ℝ),
  length > 0 →
  3 * length = 3 * 3 →
  length = 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_equals_square_side_l2239_223929


namespace NUMINAMATH_CALUDE_solution_in_quadrant_I_l2239_223979

/-- The solution to the system of equations x + 2y = 6 and kx - y = 2 is in Quadrant I iff k > 1/3 -/
theorem solution_in_quadrant_I (k : ℝ) :
  (∃ x y : ℝ, x + 2*y = 6 ∧ k*x - y = 2 ∧ x > 0 ∧ y > 0) ↔ k > 1/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_quadrant_I_l2239_223979


namespace NUMINAMATH_CALUDE_remainder_theorem_l2239_223954

theorem remainder_theorem (n : ℤ) (h : ∃ k : ℤ, n = 60 * k + 1) :
  (n^2 + 2*n + 3) % 60 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2239_223954


namespace NUMINAMATH_CALUDE_function_properties_l2239_223951

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem function_properties :
  (∀ x y, x < y → x < -1 → y < -1 → f x < f y) ∧
  (∀ x y, x < y → x > 3 → y > 3 → f x < f y) ∧
  (∀ x ∈ interval, f x ≤ 5) ∧
  (∃ x ∈ interval, f x = 5) ∧
  (∀ x ∈ interval, f x ≥ -22) ∧
  (∃ x ∈ interval, f x = -22) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2239_223951


namespace NUMINAMATH_CALUDE_quadratic_discriminant_theorem_l2239_223911

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The discriminant of a quadratic polynomial -/
def discriminant (p : QuadraticPolynomial) : ℝ := p.b^2 - 4*p.a*p.c

/-- A function to check if a quadratic equation has exactly one root -/
def has_one_root (a b c : ℝ) : Prop := (b^2 - 4*a*c = 0)

theorem quadratic_discriminant_theorem (p : QuadraticPolynomial) 
  (h1 : has_one_root p.a p.b (p.c + 2))
  (h2 : has_one_root p.a (p.b + 1/2) (p.c - 1)) :
  discriminant p = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_theorem_l2239_223911


namespace NUMINAMATH_CALUDE_trapezoid_upper_side_length_l2239_223901

/-- Theorem: For a trapezoid with a base of 25 cm, a height of 13 cm, and an area of 286 cm²,
    the length of the upper side is 19 cm. -/
theorem trapezoid_upper_side_length 
  (base : ℝ) (height : ℝ) (area : ℝ) (upper_side : ℝ) 
  (h1 : base = 25) 
  (h2 : height = 13) 
  (h3 : area = 286) 
  (h4 : area = (1/2) * (base + upper_side) * height) : 
  upper_side = 19 := by
  sorry

#check trapezoid_upper_side_length

end NUMINAMATH_CALUDE_trapezoid_upper_side_length_l2239_223901


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2239_223944

theorem simplify_fraction_product : 8 * (15 / 4) * (-40 / 45) = -80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2239_223944


namespace NUMINAMATH_CALUDE_sum_of_alpha_beta_l2239_223982

/-- Given constants α and β satisfying the rational equation, prove their sum is 176 -/
theorem sum_of_alpha_beta (α β : ℝ) 
  (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 102*x + 2021) / (x^2 + 89*x - 3960)) : 
  α + β = 176 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_alpha_beta_l2239_223982


namespace NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l2239_223967

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (acute_θ : 0 < θ ∧ θ < π/2) 
  (x_gt_1 : x > 1) 
  (h : Real.sin (θ/2) = Real.sqrt ((x + 1)/(2*x))) : 
  Real.tan θ = Real.sqrt (2*x - 1) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l2239_223967


namespace NUMINAMATH_CALUDE_rulers_added_l2239_223938

theorem rulers_added (initial_rulers : ℕ) (final_rulers : ℕ) (added_rulers : ℕ) : 
  initial_rulers = 46 → final_rulers = 71 → added_rulers = final_rulers - initial_rulers → 
  added_rulers = 25 := by
  sorry

end NUMINAMATH_CALUDE_rulers_added_l2239_223938


namespace NUMINAMATH_CALUDE_largest_area_quadrilateral_in_sector_l2239_223965

/-- The largest area of a right-angled quadrilateral inscribed in a circular sector -/
theorem largest_area_quadrilateral_in_sector (r : ℝ) (h : r > 0) :
  let max_area (α : ℝ) := 
    (2 * r^2 * Real.sin (α/2) * Real.sin (α/2)) / Real.sin α
  (max_area (2*π/3) = (r^2 * Real.sqrt 3) / 3) ∧ 
  (max_area (4*π/3) = r^2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_largest_area_quadrilateral_in_sector_l2239_223965


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2239_223972

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r

theorem geometric_sequence_product (a b c : ℝ) :
  is_geometric_sequence (-1) a b c (-2) →
  a * b * c = -2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2239_223972


namespace NUMINAMATH_CALUDE_paths_H_to_J_through_I_l2239_223930

/-- The number of paths from H to I -/
def paths_H_to_I : ℕ := Nat.choose 6 1

/-- The number of paths from I to J -/
def paths_I_to_J : ℕ := Nat.choose 5 2

/-- The total number of steps from H to J -/
def total_steps : ℕ := 11

/-- Theorem stating the number of paths from H to J passing through I -/
theorem paths_H_to_J_through_I : paths_H_to_I * paths_I_to_J = 60 :=
sorry

end NUMINAMATH_CALUDE_paths_H_to_J_through_I_l2239_223930


namespace NUMINAMATH_CALUDE_unique_three_prime_product_l2239_223963

def isPrime (n : ℕ) : Prop := Nat.Prime n

def primeFactors (n : ℕ) : List ℕ := sorry

theorem unique_three_prime_product : 
  ∃! n : ℕ, 
    ∃ p1 p2 p3 : ℕ, 
      isPrime p1 ∧ isPrime p2 ∧ isPrime p3 ∧
      p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
      n = p1 * p2 * p3 ∧
      p1 + p2 + p3 = (primeFactors 9271).sum := by sorry

end NUMINAMATH_CALUDE_unique_three_prime_product_l2239_223963


namespace NUMINAMATH_CALUDE_slope_of_line_l2239_223970

/-- The slope of a line given by the equation 4y = 5x - 20 is 5/4 -/
theorem slope_of_line (x y : ℝ) : 4 * y = 5 * x - 20 → (∃ b : ℝ, y = (5/4) * x + b) := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l2239_223970


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l2239_223985

/-- Given a rectangle with breadth b and length l, if its perimeter is 5 times its breadth
    and its area is 216 sq. cm, then its diagonal is 6√13 cm. -/
theorem rectangle_diagonal (b l : ℝ) (h1 : 2 * (l + b) = 5 * b) (h2 : l * b = 216) :
  Real.sqrt (l^2 + b^2) = 6 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l2239_223985


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l2239_223997

theorem unique_solution_power_equation : 
  ∃! (x : ℝ), x ≠ 0 ∧ (9 * x)^18 = (18 * x)^9 :=
by
  use 2/9
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l2239_223997


namespace NUMINAMATH_CALUDE_probability_of_observing_change_l2239_223908

/-- Represents the duration of the traffic light cycle in seconds -/
def cycle_duration : ℕ := 63

/-- Represents the points in the cycle where color changes occur -/
def change_points : List ℕ := [30, 33, 63]

/-- Represents the duration of the observation interval in seconds -/
def observation_duration : ℕ := 4

/-- Calculates the total duration of intervals where a change can be observed -/
def total_change_duration (cycle : ℕ) (changes : List ℕ) (obs : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the probability of observing a color change -/
theorem probability_of_observing_change :
  (total_change_duration cycle_duration change_points observation_duration : ℚ) / cycle_duration = 5 / 21 :=
sorry

end NUMINAMATH_CALUDE_probability_of_observing_change_l2239_223908


namespace NUMINAMATH_CALUDE_rectangle_width_l2239_223964

/-- Given a rectangle with length 5.4 cm and area 48.6 cm², prove its width is 9 cm -/
theorem rectangle_width (length : ℝ) (area : ℝ) (h1 : length = 5.4) (h2 : area = 48.6) :
  area / length = 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l2239_223964


namespace NUMINAMATH_CALUDE_intersection_of_M_and_P_l2239_223924

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def P : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_P : M ∩ P = {(3, -1)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_P_l2239_223924


namespace NUMINAMATH_CALUDE_cost_price_per_metre_l2239_223983

/-- Given a cloth sale with a loss, calculate the cost price per metre. -/
theorem cost_price_per_metre
  (total_metres : ℕ)
  (total_selling_price : ℕ)
  (loss_per_metre : ℕ)
  (h1 : total_metres = 600)
  (h2 : total_selling_price = 18000)
  (h3 : loss_per_metre = 5) :
  (total_selling_price + total_metres * loss_per_metre) / total_metres = 35 := by
  sorry

#check cost_price_per_metre

end NUMINAMATH_CALUDE_cost_price_per_metre_l2239_223983


namespace NUMINAMATH_CALUDE_mixed_doubles_groupings_l2239_223971

theorem mixed_doubles_groupings (male_players : Nat) (female_players : Nat) :
  male_players = 5 → female_players = 3 →
  (Nat.choose male_players 2) * (Nat.choose female_players 2) * (Nat.factorial 2) = 60 :=
by sorry

end NUMINAMATH_CALUDE_mixed_doubles_groupings_l2239_223971


namespace NUMINAMATH_CALUDE_hyperbola_condition_l2239_223957

-- Define the equation
def hyperbola_equation (x y k : ℝ) : Prop :=
  x^2 / (3 - k) + y^2 / (k - 2) = 1

-- Define the condition for a hyperbola
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y, hyperbola_equation x y k ∧ (3 - k) * (k - 2) < 0

-- Theorem statement
theorem hyperbola_condition (k : ℝ) :
  is_hyperbola k ↔ k < 2 ∨ k > 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l2239_223957


namespace NUMINAMATH_CALUDE_cookie_solution_l2239_223988

def cookie_problem (initial_cookies : ℕ) : Prop :=
  let andy_ate : ℕ := 3
  let brother_ate : ℕ := 5
  let team_size : ℕ := 8
  let team_sequence : List ℕ := List.range team_size |>.map (λ n => 2*n + 1)
  let team_ate : ℕ := team_sequence.sum
  initial_cookies = andy_ate + brother_ate + team_ate

theorem cookie_solution : 
  ∃ (initial_cookies : ℕ), cookie_problem initial_cookies ∧ initial_cookies = 72 := by
  sorry

end NUMINAMATH_CALUDE_cookie_solution_l2239_223988


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2239_223932

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the theorem
theorem complex_equation_sum (a b : ℝ) : 
  (a + 2 * i) / i = b + i → a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2239_223932


namespace NUMINAMATH_CALUDE_abs_ac_plus_bd_le_one_l2239_223941

theorem abs_ac_plus_bd_le_one (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) (h2 : c^2 + d^2 = 1) : 
  |a*c + b*d| ≤ 1 := by sorry

end NUMINAMATH_CALUDE_abs_ac_plus_bd_le_one_l2239_223941


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2239_223991

theorem square_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 7) (h2 : x * y = 5) : x^2 + y^2 = 39 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2239_223991


namespace NUMINAMATH_CALUDE_find_number_l2239_223955

theorem find_number : ∃! x : ℝ, 0.8 * x + 20 = x := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2239_223955


namespace NUMINAMATH_CALUDE_siblings_total_weight_siblings_total_weight_is_88_l2239_223984

/-- The total weight of two siblings, where one weighs 50 kg and the other weighs 12 kg less. -/
theorem siblings_total_weight : ℝ :=
  let antonio_weight : ℝ := 50
  let sister_weight : ℝ := antonio_weight - 12
  antonio_weight + sister_weight
  
/-- Prove that the total weight of the siblings is 88 kg. -/
theorem siblings_total_weight_is_88 : siblings_total_weight = 88 := by
  sorry

end NUMINAMATH_CALUDE_siblings_total_weight_siblings_total_weight_is_88_l2239_223984


namespace NUMINAMATH_CALUDE_power_equality_l2239_223992

theorem power_equality (n m : ℕ) (h1 : 4^n = 3) (h2 : 8^m = 5) : 2^(2*n + 3*m) = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2239_223992


namespace NUMINAMATH_CALUDE_bicyclists_meet_at_1030_l2239_223926

-- Define the problem parameters
def alicia_start_time : ℝ := 7.75  -- 7:45 AM in decimal hours
def david_start_time : ℝ := 8.25   -- 8:15 AM in decimal hours
def alicia_speed : ℝ := 15         -- miles per hour
def david_speed : ℝ := 18          -- miles per hour
def total_distance : ℝ := 84       -- miles

-- Define the meeting time in decimal hours (10:30 AM = 10.5)
def meeting_time : ℝ := 10.5

-- Theorem statement
theorem bicyclists_meet_at_1030 :
  let t := meeting_time - alicia_start_time
  alicia_speed * t + david_speed * (t - (david_start_time - alicia_start_time)) = total_distance :=
by sorry

end NUMINAMATH_CALUDE_bicyclists_meet_at_1030_l2239_223926


namespace NUMINAMATH_CALUDE_blue_balls_count_l2239_223939

theorem blue_balls_count (total : ℕ) (yellow : ℕ) (blue : ℕ) : 
  total = 35 →
  yellow + blue = total →
  4 * blue = 3 * yellow →
  blue = 15 := by
sorry

end NUMINAMATH_CALUDE_blue_balls_count_l2239_223939


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2239_223940

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^3 - (x + 2)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₂ + a₄ = -54 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2239_223940


namespace NUMINAMATH_CALUDE_range_of_m_l2239_223903

-- Define proposition p
def p (m : ℝ) : Prop := ∀ x y : ℝ, x^2 / (m + 1) + y^2 / (m - 1) = 1 → 
  (m + 1 > 0 ∧ m - 1 < 0)

-- Define proposition q
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 2*m + 3 ≠ 0

-- Theorem statement
theorem range_of_m (m : ℝ) : 
  (¬(p m ∧ q m) ∧ (p m ∨ q m)) → m ∈ Set.Icc 1 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2239_223903


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2239_223998

/-- A geometric sequence with first term 2 and fifth term 8 has its third term equal to 4 -/
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)  -- a is a sequence of real numbers indexed by natural numbers
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1))  -- a is a geometric sequence
  (h_a1 : a 1 = 2)  -- first term is 2
  (h_a5 : a 5 = 8)  -- fifth term is 8
  : a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2239_223998


namespace NUMINAMATH_CALUDE_product_of_roots_l2239_223989

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 5) = 24 → ∃ y : ℝ, (x + 3) * (x - 5) = 24 ∧ (x * y = -39) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l2239_223989


namespace NUMINAMATH_CALUDE_proposition_p_false_and_q_true_l2239_223933

theorem proposition_p_false_and_q_true :
  (∃ x : ℝ, 2^x ≤ x^2) ∧
  ((∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) ∧
   (∃ a b : ℝ, a * b > 1 ∧ (a ≤ 1 ∨ b ≤ 1))) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_false_and_q_true_l2239_223933


namespace NUMINAMATH_CALUDE_books_loaned_out_l2239_223948

/-- Proves the number of books loaned out given initial and final book counts and return rate -/
theorem books_loaned_out 
  (initial_books : ℕ) 
  (final_books : ℕ) 
  (return_rate : ℚ) 
  (h1 : initial_books = 150)
  (h2 : final_books = 122)
  (h3 : return_rate = 65 / 100) :
  ∃ (loaned_books : ℕ), 
    (initial_books : ℚ) - (loaned_books : ℚ) * (1 - return_rate) = final_books ∧ 
    loaned_books = 80 := by
  sorry

end NUMINAMATH_CALUDE_books_loaned_out_l2239_223948


namespace NUMINAMATH_CALUDE_difference_of_squares_l2239_223936

theorem difference_of_squares (a b : ℝ) :
  ∃ (p q : ℝ), (a - 2*b) * (a + 2*b) = (p + q) * (p - q) ∧
                (-a + b) * (-a - b) = (p + q) * (p - q) ∧
                (-a - 1) * (1 - a) = (p + q) * (p - q) ∧
                ¬(∃ (r s : ℝ), (-x + y) * (x - y) = (r + s) * (r - s)) :=
by sorry


end NUMINAMATH_CALUDE_difference_of_squares_l2239_223936


namespace NUMINAMATH_CALUDE_common_chord_length_l2239_223906

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 25
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 20 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_O A.1 A.2 ∧ circle_C A.1 A.2 ∧
  circle_O B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem common_chord_length (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 95 := by
  sorry

end NUMINAMATH_CALUDE_common_chord_length_l2239_223906


namespace NUMINAMATH_CALUDE_test_total_points_l2239_223974

/-- Given a test with the following properties:
  * Total number of questions is 30
  * Questions are either worth 5 or 10 points
  * There are 20 questions worth 5 points each
  Prove that the total point value of the test is 200 points -/
theorem test_total_points :
  ∀ (total_questions five_point_questions : ℕ)
    (point_values : Finset ℕ),
  total_questions = 30 →
  five_point_questions = 20 →
  point_values = {5, 10} →
  (total_questions - five_point_questions) * 10 + five_point_questions * 5 = 200 :=
by sorry

end NUMINAMATH_CALUDE_test_total_points_l2239_223974


namespace NUMINAMATH_CALUDE_largest_circle_area_l2239_223922

theorem largest_circle_area (playground_area : Real) (π : Real) : 
  playground_area = 400 → π = 3.1 → 
  (π * (Real.sqrt playground_area / 2)^2 : Real) = 310 := by
  sorry

end NUMINAMATH_CALUDE_largest_circle_area_l2239_223922


namespace NUMINAMATH_CALUDE_final_paycheck_amount_l2239_223980

def biweekly_gross_pay : ℝ := 1120
def retirement_rate : ℝ := 0.25
def tax_deduction : ℝ := 100

theorem final_paycheck_amount :
  biweekly_gross_pay * (1 - retirement_rate) - tax_deduction = 740 := by
  sorry

end NUMINAMATH_CALUDE_final_paycheck_amount_l2239_223980


namespace NUMINAMATH_CALUDE_y_never_perfect_square_l2239_223975

theorem y_never_perfect_square (x : ℕ) : ∃ (n : ℕ), (x^4 + 2*x^3 + 2*x^2 + 2*x + 1) ≠ n^2 := by
  sorry

end NUMINAMATH_CALUDE_y_never_perfect_square_l2239_223975


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2239_223917

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 - 3 * x^3 + 5 * x^2 - 8 * x + 15) + (-x^4 + 4 * x^3 - 2 * x^2 + 8 * x - 7) =
  x^4 + x^3 + 3 * x^2 + 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2239_223917


namespace NUMINAMATH_CALUDE_max_value_cube_root_sum_max_value_achievable_l2239_223923

theorem max_value_cube_root_sum (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  (a * b * c) ^ (1/3 : ℝ) + ((2 - a) * (2 - b) * (2 - c)) ^ (1/3 : ℝ) ≤ 2 :=
sorry

theorem max_value_achievable :
  ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧
  (a * b * c) ^ (1/3 : ℝ) + ((2 - a) * (2 - b) * (2 - c)) ^ (1/3 : ℝ) = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_cube_root_sum_max_value_achievable_l2239_223923


namespace NUMINAMATH_CALUDE_babylonian_square_58_l2239_223919

/-- Represents the Babylonian method of expressing squares --/
def babylonian_square (n : ℕ) : ℕ × ℕ :=
  let square := n * n
  let quotient := square / 60
  let remainder := square % 60
  if remainder = 0 then (quotient, 60) else (quotient, remainder)

/-- The theorem to be proved --/
theorem babylonian_square_58 : babylonian_square 58 = (56, 4) := by
  sorry

#eval babylonian_square 58  -- To check the result

end NUMINAMATH_CALUDE_babylonian_square_58_l2239_223919


namespace NUMINAMATH_CALUDE_librarian_books_taken_solve_librarian_books_l2239_223978

theorem librarian_books_taken (total_books : ℕ) (books_per_shelf : ℕ) (shelves_needed : ℕ) : ℕ :=
  total_books - (books_per_shelf * shelves_needed)

theorem solve_librarian_books (total_books : ℕ) (books_per_shelf : ℕ) (shelves_needed : ℕ)
    (h1 : total_books = 46)
    (h2 : books_per_shelf = 4)
    (h3 : shelves_needed = 9) :
    librarian_books_taken total_books books_per_shelf shelves_needed = 10 := by
  sorry

end NUMINAMATH_CALUDE_librarian_books_taken_solve_librarian_books_l2239_223978


namespace NUMINAMATH_CALUDE_k_value_l2239_223916

theorem k_value (a b k : ℝ) 
  (h1 : 2 * a = k) 
  (h2 : 3 * b = k) 
  (h3 : k ≠ 1) 
  (h4 : 2 * a + b = a * b) : 
  k = 8 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l2239_223916


namespace NUMINAMATH_CALUDE_inequality_proof_l2239_223950

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2239_223950


namespace NUMINAMATH_CALUDE_petya_vasya_equal_numbers_possible_l2239_223966

theorem petya_vasya_equal_numbers_possible :
  ∃ (n : ℤ) (k : ℕ), n ≠ 0 ∧ 
  (n + 10 * k) * 2014 = (n - 10 * k) / 2014 := by
  sorry

end NUMINAMATH_CALUDE_petya_vasya_equal_numbers_possible_l2239_223966


namespace NUMINAMATH_CALUDE_solve_probability_problem_l2239_223956

def probability_problem (p_man : ℚ) (p_wife : ℚ) : Prop :=
  p_man = 1/4 ∧ p_wife = 1/3 →
  (1 - p_man) * (1 - p_wife) = 1/2

theorem solve_probability_problem : probability_problem (1/4) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_solve_probability_problem_l2239_223956


namespace NUMINAMATH_CALUDE_tony_age_is_twelve_l2239_223958

/-- Represents Tony's work and payment information -/
structure TonyWork where
  hoursPerDay : ℕ
  payPerHourPerYear : ℚ
  workDays : ℕ
  totalEarnings : ℚ

/-- Calculates Tony's age based on his work information -/
def calculateAge (work : TonyWork) : ℕ :=
  sorry

/-- Theorem stating that Tony's age at the end of the five-month period was 12 -/
theorem tony_age_is_twelve (work : TonyWork) 
  (h1 : work.hoursPerDay = 2)
  (h2 : work.payPerHourPerYear = 1)
  (h3 : work.workDays = 60)
  (h4 : work.totalEarnings = 1140) :
  calculateAge work = 12 :=
sorry

end NUMINAMATH_CALUDE_tony_age_is_twelve_l2239_223958


namespace NUMINAMATH_CALUDE_root_product_theorem_l2239_223990

theorem root_product_theorem (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) → 
  q = 13/3 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l2239_223990
