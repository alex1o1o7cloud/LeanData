import Mathlib

namespace NUMINAMATH_CALUDE_infinitely_many_expressible_l768_76855

def is_strictly_increasing (a : ℕ → ℕ) : Prop :=
  ∀ k, a k < a (k + 1)

def expressible (a : ℕ → ℕ) (m : ℕ) : Prop :=
  ∃ (x y p q : ℕ), x > 0 ∧ y > 0 ∧ p ≠ q ∧ a m = x * a p + y * a q

theorem infinitely_many_expressible (a : ℕ → ℕ) 
  (h : is_strictly_increasing a) : 
  Set.Infinite {m : ℕ | expressible a m} :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_expressible_l768_76855


namespace NUMINAMATH_CALUDE_batsman_new_average_l768_76821

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  initialAverage : ℝ
  inningsPlayed : ℕ
  newInningScore : ℝ
  averageIncrease : ℝ

/-- Theorem: Given a batsman's stats, prove that his new average is 40 runs -/
theorem batsman_new_average (stats : BatsmanStats)
  (h1 : stats.inningsPlayed = 10)
  (h2 : stats.newInningScore = 90)
  (h3 : stats.averageIncrease = 5)
  : stats.initialAverage + stats.averageIncrease = 40 := by
  sorry

#check batsman_new_average

end NUMINAMATH_CALUDE_batsman_new_average_l768_76821


namespace NUMINAMATH_CALUDE_camp_III_sample_size_l768_76846

def systematic_sample (total : ℕ) (sample_size : ℕ) (start : ℕ) (range_start : ℕ) (range_end : ℕ) : ℕ :=
  sorry

theorem camp_III_sample_size :
  systematic_sample 600 50 3 496 600 = 8 :=
sorry

end NUMINAMATH_CALUDE_camp_III_sample_size_l768_76846


namespace NUMINAMATH_CALUDE_quadratic_equation_implication_l768_76894

theorem quadratic_equation_implication (x : ℝ) : 
  x^2 + 3*x + 5 = 7 → 3*x^2 + 9*x - 11 = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_implication_l768_76894


namespace NUMINAMATH_CALUDE_kyunghoon_descent_time_l768_76888

/-- Proves that given the conditions of Kyunghoon's mountain hike, the time it took him to go down is 2 hours. -/
theorem kyunghoon_descent_time :
  ∀ (d : ℝ), -- distance up the mountain
  d > 0 →
  d / 3 + (d + 2) / 4 = 4 → -- total time equation
  (d + 2) / 4 = 2 -- time to go down
  := by sorry

end NUMINAMATH_CALUDE_kyunghoon_descent_time_l768_76888


namespace NUMINAMATH_CALUDE_airplane_speed_proof_l768_76820

/-- Proves that the speed of one airplane is 400 mph given the conditions of the problem -/
theorem airplane_speed_proof (v : ℝ) : 
  v > 0 →  -- Assuming positive speed for the first airplane
  (2.5 * v + 2.5 * 250 = 1625) →  -- Condition from the problem
  v = 400 := by
sorry

end NUMINAMATH_CALUDE_airplane_speed_proof_l768_76820


namespace NUMINAMATH_CALUDE_sphere_plane_intersection_l768_76817

/-- A sphere intersecting a plane creates a circular intersection. -/
theorem sphere_plane_intersection
  (r : ℝ) -- radius of the sphere
  (h : ℝ) -- depth of the intersection
  (w : ℝ) -- radius of the circular intersection
  (hr : r = 16.25)
  (hh : h = 10)
  (hw : w = 15) :
  r^2 = h * (2 * r - h) + w^2 :=
sorry

end NUMINAMATH_CALUDE_sphere_plane_intersection_l768_76817


namespace NUMINAMATH_CALUDE_trajectory_of_symmetric_point_l768_76886

/-- The equation of the trajectory of point N, which is symmetric to a point M on the circle x^2 + y^2 = 4 with respect to the point A(1,1) -/
theorem trajectory_of_symmetric_point (x y : ℝ) :
  (∃ (mx my : ℝ), mx^2 + my^2 = 4 ∧ x = 2 - mx ∧ y = 2 - my) →
  (x - 2)^2 + (y - 2)^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_trajectory_of_symmetric_point_l768_76886


namespace NUMINAMATH_CALUDE_staircase_steps_l768_76878

/-- Represents a staircase with a given number of steps. -/
structure Staircase :=
  (steps : ℕ)

/-- Calculates the total number of toothpicks used in a staircase. -/
def toothpicks (s : Staircase) : ℕ :=
  3 * (s.steps * (s.steps + 1)) / 2

/-- Theorem stating that a staircase with 270 toothpicks has 12 steps. -/
theorem staircase_steps : ∃ s : Staircase, toothpicks s = 270 ∧ s.steps = 12 := by
  sorry


end NUMINAMATH_CALUDE_staircase_steps_l768_76878


namespace NUMINAMATH_CALUDE_investment_interest_rate_l768_76837

/-- Calculates the total interest rate for a two-share investment --/
def total_interest_rate (total_investment : ℚ) (rate1 : ℚ) (rate2 : ℚ) (amount2 : ℚ) : ℚ :=
  let amount1 := total_investment - amount2
  let interest1 := amount1 * rate1
  let interest2 := amount2 * rate2
  let total_interest := interest1 + interest2
  (total_interest / total_investment) * 100

/-- Theorem stating the total interest rate for the given investment scenario --/
theorem investment_interest_rate :
  total_interest_rate 10000 (9/100) (11/100) 3750 = (975/10000) * 100 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l768_76837


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l768_76833

theorem min_value_quadratic_sum (a b c : ℝ) (h : 2*a + 2*b + c = 8) :
  (a - 1)^2 + (b + 2)^2 + (c - 3)^2 ≥ 49/9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l768_76833


namespace NUMINAMATH_CALUDE_fraction_simplification_l768_76803

theorem fraction_simplification : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l768_76803


namespace NUMINAMATH_CALUDE_no_solution_condition_l768_76844

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, 8 * |x - 4*a| + |x - a^2| + 7*x - 2*a ≠ 0) ↔ (a < -22 ∨ a > 0) :=
sorry

end NUMINAMATH_CALUDE_no_solution_condition_l768_76844


namespace NUMINAMATH_CALUDE_rice_container_problem_l768_76852

theorem rice_container_problem (total_weight : ℚ) (container_weight : ℕ) 
  (h1 : total_weight = 33 / 4)
  (h2 : container_weight = 33)
  (h3 : (1 : ℚ) = 16 / 16) : 
  (total_weight * 16) / container_weight = 4 := by
  sorry

end NUMINAMATH_CALUDE_rice_container_problem_l768_76852


namespace NUMINAMATH_CALUDE_pirate_treasure_division_l768_76816

def pirate_share (n : ℕ) (k : ℕ) (remaining : ℚ) : ℚ :=
  (k : ℚ) / (n : ℚ) * remaining

def remaining_coins (n : ℕ) (k : ℕ) (initial : ℚ) : ℚ :=
  if k = 0 then initial
  else
    (1 - (k : ℚ) / (n : ℚ)) * remaining_coins n (k - 1) initial

def is_valid_distribution (n : ℕ) (initial : ℕ) : Prop :=
  ∀ k : ℕ, k ≤ n → ∃ (m : ℕ), pirate_share n k (remaining_coins n (k - 1) initial) = m

theorem pirate_treasure_division (n : ℕ) (h : n = 15) :
  ∃ (initial : ℕ),
    (∀ smaller : ℕ, smaller < initial → ¬is_valid_distribution n smaller) ∧
    is_valid_distribution n initial ∧
    pirate_share n n (remaining_coins n (n - 1) initial) = 1536 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_division_l768_76816


namespace NUMINAMATH_CALUDE_problem_solution_l768_76854

theorem problem_solution (a b : ℝ) (h1 : a + b = 8) (h2 : a^2 * b^2 = 4) :
  (a^2 + b^2)/2 - a*b = 28 ∨ (a^2 + b^2)/2 - a*b = 36 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l768_76854


namespace NUMINAMATH_CALUDE_isosceles_triangle_locus_l768_76897

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def isIsosceles (t : Triangle) : Prop :=
  (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 = (t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2

def satisfiesLocus (C : ℝ × ℝ) : Prop :=
  C.1^2 + C.2^2 - 6*C.1 + 4*C.2 - 5 = 0

theorem isosceles_triangle_locus :
  ∀ t : Triangle,
    t.A = (3, -2) →
    t.B = (0, 1) →
    isIsosceles t →
    t.C ≠ (0, 1) →
    t.C ≠ (6, -5) →
    satisfiesLocus t.C :=
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_locus_l768_76897


namespace NUMINAMATH_CALUDE_increasing_geometric_sequence_formula_l768_76842

/-- An increasing geometric sequence with specific properties -/
def IncreasingGeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧
  (a 5)^2 = a 10 ∧
  (∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1))

/-- The general term formula for the sequence -/
def GeneralTermFormula (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = 2^n

/-- Theorem stating that an increasing geometric sequence with the given properties
    has the general term formula a_n = 2^n -/
theorem increasing_geometric_sequence_formula (a : ℕ → ℝ) :
  IncreasingGeometricSequence a → GeneralTermFormula a := by
  sorry

end NUMINAMATH_CALUDE_increasing_geometric_sequence_formula_l768_76842


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l768_76840

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given the man is 27 years older than his son and the son's present age is 25 years. -/
theorem man_son_age_ratio :
  let son_age : ℕ := 25
  let man_age : ℕ := son_age + 27
  let son_age_in_two_years : ℕ := son_age + 2
  let man_age_in_two_years : ℕ := man_age + 2
  (man_age_in_two_years : ℚ) / (son_age_in_two_years : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l768_76840


namespace NUMINAMATH_CALUDE_scout_troop_profit_l768_76801

-- Define the parameters
def num_bars : ℕ := 1500
def purchase_rate : ℚ := 1 / 3
def selling_rate : ℚ := 3 / 4
def fixed_cost : ℚ := 50

-- Define the profit calculation
def profit : ℚ :=
  num_bars * selling_rate - (num_bars * purchase_rate + fixed_cost)

-- Theorem statement
theorem scout_troop_profit : profit = 575 := by
  sorry

end NUMINAMATH_CALUDE_scout_troop_profit_l768_76801


namespace NUMINAMATH_CALUDE_coefficient_of_x_fifth_l768_76850

theorem coefficient_of_x_fifth (a : ℝ) : 
  (Nat.choose 8 5) * a^5 = 56 → a = 1 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_fifth_l768_76850


namespace NUMINAMATH_CALUDE_score_order_l768_76848

theorem score_order (a b c d : ℝ) 
  (h1 : b + d = a + c)
  (h2 : a + c > b + d)
  (h3 : d > b + c)
  (ha : a ≥ 0)
  (hb : b ≥ 0)
  (hc : c ≥ 0)
  (hd : d ≥ 0) :
  a > d ∧ d > b ∧ b > c :=
by sorry

end NUMINAMATH_CALUDE_score_order_l768_76848


namespace NUMINAMATH_CALUDE_student_ticket_price_l768_76871

/-- The price of a senior citizen ticket -/
def senior_price : ℝ := sorry

/-- The price of a student ticket -/
def student_price : ℝ := sorry

/-- First day sales equation -/
axiom first_day_sales : 4 * senior_price + 3 * student_price = 79

/-- Second day sales equation -/
axiom second_day_sales : 12 * senior_price + 10 * student_price = 246

/-- Theorem stating that the student ticket price is 9 dollars -/
theorem student_ticket_price : student_price = 9 := by sorry

end NUMINAMATH_CALUDE_student_ticket_price_l768_76871


namespace NUMINAMATH_CALUDE_alberts_to_angelas_marbles_ratio_l768_76802

/-- Proves that the ratio of Albert's marbles to Angela's marbles is 3:1 -/
theorem alberts_to_angelas_marbles_ratio (allison_marbles : ℕ) (angela_more_than_allison : ℕ) 
  (albert_and_allison_total : ℕ) 
  (h1 : allison_marbles = 28)
  (h2 : angela_more_than_allison = 8)
  (h3 : albert_and_allison_total = 136) : 
  (albert_and_allison_total - allison_marbles) / (allison_marbles + angela_more_than_allison) = 3 := by
  sorry

#check alberts_to_angelas_marbles_ratio

end NUMINAMATH_CALUDE_alberts_to_angelas_marbles_ratio_l768_76802


namespace NUMINAMATH_CALUDE_boat_distance_against_stream_l768_76860

/-- Calculates the distance a boat travels against the stream in one hour. -/
def distance_against_stream (boat_speed : ℝ) (distance_with_stream : ℝ) : ℝ :=
  boat_speed - (distance_with_stream - boat_speed)

/-- Theorem: Given a boat with speed 10 km/hr in still water that travels 15 km along the stream in one hour,
    the distance it travels against the stream in one hour is 5 km. -/
theorem boat_distance_against_stream :
  distance_against_stream 10 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_against_stream_l768_76860


namespace NUMINAMATH_CALUDE_second_alignment_l768_76810

/-- Represents the number of Heavenly Stems -/
def heavenly_stems : ℕ := 10

/-- Represents the number of Earthly Branches -/
def earthly_branches : ℕ := 12

/-- Represents the cycle length of the combined Heavenly Stems and Earthly Branches -/
def cycle_length : ℕ := lcm heavenly_stems earthly_branches

/-- 
Theorem: The second occurrence of the first Heavenly Stem aligning with 
the first Earthly Branch happens at column 61.
-/
theorem second_alignment : 
  cycle_length + 1 = 61 := by sorry

end NUMINAMATH_CALUDE_second_alignment_l768_76810


namespace NUMINAMATH_CALUDE_fraction_zero_solution_l768_76872

theorem fraction_zero_solution (x : ℝ) : 
  (x^2 + x - 2) / (x - 1) = 0 ∧ x ≠ 1 → x = -2 :=
by
  sorry

#check fraction_zero_solution

end NUMINAMATH_CALUDE_fraction_zero_solution_l768_76872


namespace NUMINAMATH_CALUDE_division_result_l768_76861

theorem division_result : (3486 : ℚ) / 189 = 18.444444444444443 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l768_76861


namespace NUMINAMATH_CALUDE_tom_family_members_l768_76827

/-- The number of family members Tom invited, excluding siblings -/
def family_members : ℕ := 2

/-- The number of Tom's siblings -/
def siblings : ℕ := 3

/-- The number of meals per day -/
def meals_per_day : ℕ := 3

/-- The number of plates used per meal -/
def plates_per_meal : ℕ := 2

/-- The duration of the stay in days -/
def stay_duration : ℕ := 4

/-- The total number of plates used -/
def total_plates : ℕ := 144

theorem tom_family_members :
  family_members = 2 :=
by sorry

end NUMINAMATH_CALUDE_tom_family_members_l768_76827


namespace NUMINAMATH_CALUDE_prize_distribution_l768_76880

theorem prize_distribution (total_winners : ℕ) (min_award : ℚ) (max_award : ℚ) :
  total_winners = 20 →
  min_award = 20 →
  max_award = 160 →
  ∃ (total_prize : ℚ),
    total_prize > 0 ∧
    (2 / 5 : ℚ) * total_prize = max_award ∧
    (∀ (winner : ℕ), winner ≤ total_winners → ∃ (award : ℚ), min_award ≤ award ∧ award ≤ max_award) ∧
    total_prize = 1000 :=
by sorry

end NUMINAMATH_CALUDE_prize_distribution_l768_76880


namespace NUMINAMATH_CALUDE_reeta_pencils_l768_76884

theorem reeta_pencils (reeta_pencils : ℕ) 
  (h1 : reeta_pencils + (2 * reeta_pencils + 4) = 64) : 
  reeta_pencils = 20 := by
  sorry

end NUMINAMATH_CALUDE_reeta_pencils_l768_76884


namespace NUMINAMATH_CALUDE_box_volume_perimeter_triples_l768_76831

def is_valid_triple (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ a * b * c = 4 * (a + b + c)

theorem box_volume_perimeter_triples :
  ∃! (n : ℕ), ∃ (S : Finset (ℕ × ℕ × ℕ)),
    S.card = n ∧
    (∀ (t : ℕ × ℕ × ℕ), t ∈ S ↔ is_valid_triple t.1 t.2.1 t.2.2) ∧
    n = 5 :=
sorry

end NUMINAMATH_CALUDE_box_volume_perimeter_triples_l768_76831


namespace NUMINAMATH_CALUDE_product_sum_8670_l768_76862

theorem product_sum_8670 : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 8670 ∧ 
  a + b = 187 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_8670_l768_76862


namespace NUMINAMATH_CALUDE_coupon_discount_percentage_l768_76873

theorem coupon_discount_percentage 
  (total_bill : ℝ) 
  (num_friends : ℕ) 
  (individual_payment : ℝ) 
  (h1 : total_bill = 100) 
  (h2 : num_friends = 5) 
  (h3 : individual_payment = 18.8) : 
  (total_bill - num_friends * individual_payment) / total_bill * 100 = 6 := by
sorry

end NUMINAMATH_CALUDE_coupon_discount_percentage_l768_76873


namespace NUMINAMATH_CALUDE_remainder_theorem_example_l768_76813

theorem remainder_theorem_example (x : ℤ) :
  (Polynomial.X ^ 9 + 3 : Polynomial ℤ).eval 2 = 515 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_example_l768_76813


namespace NUMINAMATH_CALUDE_no_formula_fits_all_pairs_l768_76832

-- Define the pairs of x and y values
def xy_pairs : List (ℕ × ℕ) := [(1, 5), (2, 15), (3, 35), (4, 69), (5, 119)]

-- Define the formulas
def formula_A (x : ℕ) : ℕ := x^3 + x^2 + x + 2
def formula_B (x : ℕ) : ℕ := 3*x^2 + 2*x + 1
def formula_C (x : ℕ) : ℕ := 2*x^3 - x + 4
def formula_D (x : ℕ) : ℕ := 3*x^3 + 2*x^2 + x + 1

-- Theorem statement
theorem no_formula_fits_all_pairs :
  ∀ (pair : ℕ × ℕ), pair ∈ xy_pairs →
    (formula_A pair.1 ≠ pair.2) ∧
    (formula_B pair.1 ≠ pair.2) ∧
    (formula_C pair.1 ≠ pair.2) ∧
    (formula_D pair.1 ≠ pair.2) :=
by sorry

end NUMINAMATH_CALUDE_no_formula_fits_all_pairs_l768_76832


namespace NUMINAMATH_CALUDE_smallest_coconut_pile_l768_76818

def process (n : ℕ) : ℕ := (n - 1) * 4 / 5

def iterate_process (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | m + 1 => process (iterate_process n m)

theorem smallest_coconut_pile :
  ∃ (n : ℕ), n > 0 ∧ 
    (iterate_process n 5) % 5 = 0 ∧
    n ≥ (iterate_process n 0) - (iterate_process n 1) +
        (iterate_process n 1) - (iterate_process n 2) +
        (iterate_process n 2) - (iterate_process n 3) +
        (iterate_process n 3) - (iterate_process n 4) +
        (iterate_process n 4) - (iterate_process n 5) + 5 ∧
    (∀ (m : ℕ), m > 0 ∧ m < n →
      (iterate_process m 5) % 5 ≠ 0 ∨
      m < (iterate_process m 0) - (iterate_process m 1) +
          (iterate_process m 1) - (iterate_process m 2) +
          (iterate_process m 2) - (iterate_process m 3) +
          (iterate_process m 3) - (iterate_process m 4) +
          (iterate_process m 4) - (iterate_process m 5) + 5) ∧
    n = 3121 := by
  sorry

#check smallest_coconut_pile

end NUMINAMATH_CALUDE_smallest_coconut_pile_l768_76818


namespace NUMINAMATH_CALUDE_equation_solution_l768_76835

theorem equation_solution : ∃ x : ℚ, x + 2/5 = 7/10 + 1/2 ∧ x = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l768_76835


namespace NUMINAMATH_CALUDE_regions_in_circle_l768_76829

/-- The number of regions created by radii and concentric circles within a larger circle -/
def num_regions (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

/-- Theorem: 16 radii and 10 concentric circles create 176 regions -/
theorem regions_in_circle (r : ℕ) (c : ℕ) 
    (h1 : r = 16) (h2 : c = 10) : 
    num_regions r c = 176 := by
  sorry

#eval num_regions 16 10

end NUMINAMATH_CALUDE_regions_in_circle_l768_76829


namespace NUMINAMATH_CALUDE_cube_split_with_39_l768_76814

/-- Given a natural number m > 1, if m³ can be split into a sum of consecutive odd numbers 
    starting from (m+1)² and one of these odd numbers is 39, then m = 6 -/
theorem cube_split_with_39 (m : ℕ) (h1 : m > 1) :
  (∃ k : ℕ, (m + 1)^2 + 2*k = 39) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_with_39_l768_76814


namespace NUMINAMATH_CALUDE_students_not_participating_l768_76882

-- Define the sets and their cardinalities
def totalStudents : ℕ := 45
def volleyballParticipants : ℕ := 12
def trackFieldParticipants : ℕ := 20
def bothParticipants : ℕ := 6

-- Define the theorem
theorem students_not_participating : 
  totalStudents - volleyballParticipants - trackFieldParticipants + bothParticipants = 19 :=
by sorry

end NUMINAMATH_CALUDE_students_not_participating_l768_76882


namespace NUMINAMATH_CALUDE_min_value_theorem_l768_76811

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  ∃ (m : ℝ), m = 3 ∧ ∀ x y, x > 0 → y > 0 → x + y = 4 → (y / x + 4 / y) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l768_76811


namespace NUMINAMATH_CALUDE_monomial_sum_implies_m_pow_n_eq_nine_l768_76800

/-- If the sum of a^(m-1)b^2 and (1/2)a^2b^n is a monomial, then m^n = 9 -/
theorem monomial_sum_implies_m_pow_n_eq_nine 
  (a b : ℝ) (m n : ℕ) 
  (h : ∃ (k : ℝ) (p q : ℕ), a^(m-1) * b^2 + (1/2) * a^2 * b^n = k * a^p * b^q) :
  m^n = 9 := by
sorry

end NUMINAMATH_CALUDE_monomial_sum_implies_m_pow_n_eq_nine_l768_76800


namespace NUMINAMATH_CALUDE_max_b_value_l768_76853

/-- The maximum value of b given the conditions -/
theorem max_b_value (a : ℝ) (f g : ℝ → ℝ) (h₁ : a > 0)
  (h₂ : ∀ x, f x = 6 * a^2 * Real.log x)
  (h₃ : ∀ x, g x = x^2 - 4*a*x - b)
  (h₄ : ∃ x₀, x₀ > 0 ∧ (deriv f x₀ = deriv g x₀) ∧ (f x₀ = g x₀)) :
  (∃ b : ℝ, ∀ b' : ℝ, b' ≤ b) ∧ (∀ b : ℝ, (∃ b' : ℝ, ∀ b'' : ℝ, b'' ≤ b') → b ≤ 1 / (3 * Real.exp 2)) :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l768_76853


namespace NUMINAMATH_CALUDE_function_monotonicity_implies_c_zero_and_b_positive_l768_76896

-- Define the function f(x)
def f (b c x : ℝ) : ℝ := -x^3 - b*x^2 - 5*c*x

-- State the theorem
theorem function_monotonicity_implies_c_zero_and_b_positive
  (b c : ℝ)
  (h1 : ∀ x ≤ 0, Monotone (fun x => f b c x))
  (h2 : ∀ x ∈ Set.Icc 0 6, StrictMono (fun x => f b c x)) :
  c = 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_monotonicity_implies_c_zero_and_b_positive_l768_76896


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l768_76830

theorem sqrt_equation_solution (x y : ℝ) : 
  Real.sqrt (4 - 5*x + y) = 9 → y = 77 + 5*x := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l768_76830


namespace NUMINAMATH_CALUDE_john_bought_36_rolls_l768_76849

/-- The number of rolls John bought given the cost per dozen and the amount spent -/
def rolls_bought (cost_per_dozen : ℚ) (amount_spent : ℚ) : ℚ :=
  (amount_spent / cost_per_dozen) * 12

/-- Theorem stating that John bought 36 rolls -/
theorem john_bought_36_rolls :
  let cost_per_dozen : ℚ := 5
  let amount_spent : ℚ := 15
  rolls_bought cost_per_dozen amount_spent = 36 := by
  sorry

end NUMINAMATH_CALUDE_john_bought_36_rolls_l768_76849


namespace NUMINAMATH_CALUDE_min_value_of_sequence_l768_76890

/-- Given a sequence {a_n} where a₂ = 102 and aₙ₊₁ - aₙ = 4n for n ∈ ℕ*, 
    the minimum value of {aₙ/n} is 26. -/
theorem min_value_of_sequence (a : ℕ → ℝ) : 
  (a 2 = 102) → 
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 4 * n) → 
  (∃ n₀ : ℕ, n₀ ≥ 1 ∧ a n₀ / n₀ = 26) ∧ 
  (∀ n : ℕ, n ≥ 1 → a n / n ≥ 26) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sequence_l768_76890


namespace NUMINAMATH_CALUDE_visitor_growth_rate_l768_76859

theorem visitor_growth_rate (initial_visitors : ℝ) (final_visitors : ℝ) (x : ℝ) : 
  initial_visitors = 42 → 
  final_visitors = 133.91 → 
  initial_visitors * (1 + x)^2 = final_visitors :=
by sorry

end NUMINAMATH_CALUDE_visitor_growth_rate_l768_76859


namespace NUMINAMATH_CALUDE_all_statements_true_l768_76885

def A : Set ℝ := {-1, 2, 3}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}

theorem all_statements_true :
  (A ∩ B ≠ A) ∧
  (A ∪ B ≠ B) ∧
  (3 ∉ {x : ℝ | x < -1 ∨ x ≥ 3}) ∧
  (A ∩ {x : ℝ | x < -1 ∨ x ≥ 3} ≠ ∅) := by
  sorry

end NUMINAMATH_CALUDE_all_statements_true_l768_76885


namespace NUMINAMATH_CALUDE_repeating_decimal_28_l768_76807

/-- The repeating decimal 0.2828... is equal to 28/99 -/
theorem repeating_decimal_28 : ∃ (x : ℚ), x = 28 / 99 ∧ x = 0 + (28 / 100) * (1 / (1 - 1 / 100)) :=
by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_28_l768_76807


namespace NUMINAMATH_CALUDE_intersection_M_N_l768_76843

def M : Set ℤ := {-2, 0, 2}
def N : Set ℤ := {x | x^2 = x}

theorem intersection_M_N : M ∩ N = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l768_76843


namespace NUMINAMATH_CALUDE_vector_dot_product_properties_l768_76838

/-- Given vectors a and b in R², prove properties about their dot product. -/
theorem vector_dot_product_properties (α β : ℝ) (k : ℝ) 
  (h_k_pos : k > 0)
  (a : ℝ × ℝ := (Real.cos α, Real.sin α))
  (b : ℝ × ℝ := (Real.cos β, Real.sin β))
  (h_norm : ‖k • a + b‖ = Real.sqrt 3 * ‖a - k • b‖) :
  let dot := a.1 * b.1 + a.2 * b.2
  ∃ θ : ℝ,
    (dot = Real.cos (α - β)) ∧ 
    (dot = (k^2 + 1) / (4 * k)) ∧
    (0 ≤ θ ∧ θ ≤ π) ∧
    (dot ≥ 1/2) ∧
    (dot = 1/2 ↔ θ = π/3) :=
sorry

end NUMINAMATH_CALUDE_vector_dot_product_properties_l768_76838


namespace NUMINAMATH_CALUDE_always_two_real_roots_one_nonnegative_root_iff_l768_76865

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + (4-m)*x + (3-m)

-- Theorem 1: The equation always has two real roots
theorem always_two_real_roots (m : ℝ) :
  ∃ (x1 x2 : ℝ), quadratic m x1 = 0 ∧ quadratic m x2 = 0 :=
sorry

-- Theorem 2: The equation has exactly one non-negative real root iff m ≥ 3
theorem one_nonnegative_root_iff (m : ℝ) :
  (∃! (x : ℝ), x ≥ 0 ∧ quadratic m x = 0) ↔ m ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_always_two_real_roots_one_nonnegative_root_iff_l768_76865


namespace NUMINAMATH_CALUDE_sandys_book_purchase_l768_76841

theorem sandys_book_purchase (cost_shop1 : ℕ) (books_shop2 : ℕ) (cost_shop2 : ℕ) (avg_price : ℕ) : 
  cost_shop1 = 1480 →
  books_shop2 = 55 →
  cost_shop2 = 920 →
  avg_price = 20 →
  ∃ (books_shop1 : ℕ), 
    books_shop1 = 65 ∧ 
    (cost_shop1 + cost_shop2) / (books_shop1 + books_shop2) = avg_price :=
by sorry

end NUMINAMATH_CALUDE_sandys_book_purchase_l768_76841


namespace NUMINAMATH_CALUDE_rhombus_diagonals_l768_76858

/-- A rhombus with side length 1 and one angle of 120° has diagonals of length 1 and √3. -/
theorem rhombus_diagonals (s : ℝ) (α : ℝ) (d₁ d₂ : ℝ) 
  (h_side : s = 1)
  (h_angle : α = 120 * π / 180) :
  d₁ = 1 ∧ d₂ = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_rhombus_diagonals_l768_76858


namespace NUMINAMATH_CALUDE_time_to_get_ahead_l768_76815

/-- Proves that the time for a faster traveler to get 1/3 mile ahead of a slower traveler is 2 minutes -/
theorem time_to_get_ahead (man_speed woman_speed : ℝ) (catch_up_time : ℝ) : 
  man_speed = 5 →
  woman_speed = 15 →
  catch_up_time = 4 →
  (woman_speed - man_speed) * 2 / 60 = 1 / 3 :=
by
  sorry

#check time_to_get_ahead

end NUMINAMATH_CALUDE_time_to_get_ahead_l768_76815


namespace NUMINAMATH_CALUDE_div_mul_calculation_l768_76819

theorem div_mul_calculation : (120 / 5) / 3 * 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_div_mul_calculation_l768_76819


namespace NUMINAMATH_CALUDE_work_completion_theorem_l768_76870

def work_completion_time (rate_A rate_B rate_C : ℚ) (initial_days : ℕ) : ℚ :=
  let combined_rate_AB := rate_A + rate_B
  let work_done_AB := combined_rate_AB * initial_days
  let remaining_work := 1 - work_done_AB
  let combined_rate_AC := rate_A + rate_C
  initial_days + remaining_work / combined_rate_AC

theorem work_completion_theorem :
  let rate_A : ℚ := 1 / 30
  let rate_B : ℚ := 1 / 15
  let rate_C : ℚ := 1 / 20
  let initial_days : ℕ := 5
  work_completion_time rate_A rate_B rate_C initial_days = 11 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l768_76870


namespace NUMINAMATH_CALUDE_domain_of_composition_l768_76899

-- Define the function f with domain [1,5]
def f : Set ℝ := Set.Icc 1 5

-- State the theorem
theorem domain_of_composition (f : Set ℝ) (h : f = Set.Icc 1 5) :
  {x : ℝ | ∃ y ∈ f, y = 2*x - 1} = Set.Icc 1 3 := by sorry

end NUMINAMATH_CALUDE_domain_of_composition_l768_76899


namespace NUMINAMATH_CALUDE_jasper_kite_raising_time_l768_76822

/-- Given Omar's kite-raising rate and Jasper's rate being three times Omar's,
    prove that Jasper takes 10 minutes to raise his kite 600 feet. -/
theorem jasper_kite_raising_time 
  (omar_height : ℝ) 
  (omar_time : ℝ) 
  (jasper_height : ℝ) 
  (omar_height_val : omar_height = 240) 
  (omar_time_val : omar_time = 12) 
  (jasper_height_val : jasper_height = 600) 
  (jasper_rate_mul : ℝ) 
  (jasper_rate_rel : jasper_rate_mul = 3) :
  (jasper_height / (jasper_rate_mul * omar_height / omar_time)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_jasper_kite_raising_time_l768_76822


namespace NUMINAMATH_CALUDE_equation_solutions_l768_76866

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (x₁^2 + 2*x₁ = 0 ∧ x₂^2 + 2*x₂ = 0) ∧ x₁ = 0 ∧ x₂ = -2) ∧
  (∃ x₁ x₂ : ℝ, (2*x₁^2 - 6*x₁ = 3 ∧ 2*x₂^2 - 6*x₂ = 3) ∧ 
    x₁ = (3 + Real.sqrt 15) / 2 ∧ x₂ = (3 - Real.sqrt 15) / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l768_76866


namespace NUMINAMATH_CALUDE_vacation_duration_l768_76825

/-- The number of emails received on the first day -/
def first_day_emails : ℕ := 16

/-- The ratio of emails received on each subsequent day compared to the previous day -/
def email_ratio : ℚ := 1/2

/-- The total number of emails received during the vacation -/
def total_emails : ℕ := 30

/-- Calculate the sum of a geometric series -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The number of days in the vacation -/
def vacation_days : ℕ := 4

theorem vacation_duration :
  geometric_sum first_day_emails email_ratio vacation_days = total_emails := by
  sorry

end NUMINAMATH_CALUDE_vacation_duration_l768_76825


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l768_76883

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ k : ℝ, m^2 + m - 2 + (m^2 - 1) * Complex.I = k * Complex.I) ↔ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l768_76883


namespace NUMINAMATH_CALUDE_train_speed_l768_76893

/-- Calculate the speed of a train given its length and time to cross a point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 500) (h2 : time = 20) :
  length / time = 25 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l768_76893


namespace NUMINAMATH_CALUDE_ninety_six_times_one_hundred_four_l768_76857

theorem ninety_six_times_one_hundred_four : 96 * 104 = 9984 := by
  sorry

end NUMINAMATH_CALUDE_ninety_six_times_one_hundred_four_l768_76857


namespace NUMINAMATH_CALUDE_right_triangles_common_hypotenuse_l768_76805

-- Define the triangles and their properties
def triangle_ABC (a : ℝ) := {BC : ℝ // BC = 2 ∧ ∃ (AC : ℝ), AC = a}
def triangle_ABD := {AD : ℝ // AD = 3}

-- Define the theorem
theorem right_triangles_common_hypotenuse (a : ℝ) 
  (h : a ≥ Real.sqrt 5) -- Ensure BD is real
  (ABC : triangle_ABC a) (ABD : triangle_ABD) :
  ∃ (BD : ℝ), BD = Real.sqrt (a^2 - 5) :=
sorry

end NUMINAMATH_CALUDE_right_triangles_common_hypotenuse_l768_76805


namespace NUMINAMATH_CALUDE_two_int_points_probability_l768_76887

/-- Square S with diagonal endpoints (1/2, 3/2) and (-1/2, -3/2) -/
def S : Set (ℝ × ℝ) := sorry

/-- Random point v = (x,y) where 0 ≤ x ≤ 1006 and 0 ≤ y ≤ 1006 -/
def v : ℝ × ℝ := sorry

/-- T(v) is a translated copy of S centered at v -/
def T (v : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- The probability that T(v) contains exactly two integer points in its interior -/
def prob_two_int_points : ℝ := sorry

theorem two_int_points_probability :
  prob_two_int_points = 2 / 25 := by sorry

end NUMINAMATH_CALUDE_two_int_points_probability_l768_76887


namespace NUMINAMATH_CALUDE_principal_is_720_l768_76877

/-- Calculates the principal amount given simple interest, time, and rate -/
def calculate_principal (simple_interest : ℚ) (time : ℚ) (rate : ℚ) : ℚ :=
  simple_interest * 100 / (rate * time)

/-- Theorem stating that the principal amount is 720 given the problem conditions -/
theorem principal_is_720 :
  let simple_interest : ℚ := 180
  let time : ℚ := 4
  let rate : ℚ := 6.25
  calculate_principal simple_interest time rate = 720 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_720_l768_76877


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_bound_l768_76868

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying certain conditions, its 12th term is less than or equal to 7. -/
theorem arithmetic_sequence_12th_term_bound
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_8 : a 8 ≥ 15)
  (h_9 : a 9 ≤ 13) :
  a 12 ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_bound_l768_76868


namespace NUMINAMATH_CALUDE_linear_equation_equivalence_l768_76875

theorem linear_equation_equivalence (x y : ℝ) :
  (3 * x - 2 * y = 6) ↔ (y = (3 / 2) * x - 3) := by sorry

end NUMINAMATH_CALUDE_linear_equation_equivalence_l768_76875


namespace NUMINAMATH_CALUDE_interest_rate_difference_l768_76892

theorem interest_rate_difference 
  (principal : ℝ) 
  (time : ℝ) 
  (interest_diff : ℝ) 
  (h1 : principal = 2100) 
  (h2 : time = 3) 
  (h3 : interest_diff = 63) : 
  ∃ (rate1 rate2 : ℝ), rate2 - rate1 = 0.01 ∧ 
    principal * rate2 * time - principal * rate1 * time = interest_diff :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l768_76892


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l768_76826

theorem tan_double_angle_special_case (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) : 
  Real.tan (2 * α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l768_76826


namespace NUMINAMATH_CALUDE_min_distance_to_i_l768_76806

-- Define the complex number z
variable (z : ℂ)

-- State the theorem
theorem min_distance_to_i (h : Complex.abs (z^2 - 1) = Complex.abs (z * (z - Complex.I))) :
  Complex.abs (z - Complex.I) ≥ (3 * Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_i_l768_76806


namespace NUMINAMATH_CALUDE_arrange_sticks_into_triangles_l768_76856

/-- Represents a stick with a positive length -/
structure Stick where
  length : ℝ
  positive : length > 0

/-- Represents a triangle formed by three sticks -/
structure Triangle where
  side1 : Stick
  side2 : Stick
  side3 : Stick

/-- Checks if three sticks can form a valid triangle -/
def isValidTriangle (s1 s2 s3 : Stick) : Prop :=
  s1.length + s2.length > s3.length ∧
  s1.length + s3.length > s2.length ∧
  s2.length + s3.length > s1.length

/-- Theorem stating that it's always possible to arrange six sticks into two triangles
    with one triangle having sides of one, two, and three sticks -/
theorem arrange_sticks_into_triangles
  (s1 s2 s3 s4 s5 s6 : Stick)
  (h_pairwise_different : s1.length < s2.length ∧ s2.length < s3.length ∧
                          s3.length < s4.length ∧ s4.length < s5.length ∧
                          s5.length < s6.length) :
  ∃ (t1 t2 : Triangle),
    (isValidTriangle t1.side1 t1.side2 t1.side3) ∧
    (isValidTriangle t2.side1 t2.side2 t2.side3) ∧
    ((t1.side1.length = s1.length ∧ t1.side2.length = s3.length + s5.length ∧ t1.side3.length = s2.length + s4.length + s6.length) ∨
     (t2.side1.length = s1.length ∧ t2.side2.length = s3.length + s5.length ∧ t2.side3.length = s2.length + s4.length + s6.length)) :=
by sorry

end NUMINAMATH_CALUDE_arrange_sticks_into_triangles_l768_76856


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l768_76889

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a^2 < b → -Real.sqrt b < a ∧ a < Real.sqrt b)) ↔
  ((a ≥ Real.sqrt b ∨ a ≤ -Real.sqrt b) → a^2 ≥ b) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l768_76889


namespace NUMINAMATH_CALUDE_permutations_of_111222_l768_76845

/-- The number of permutations of a multiset with 6 elements, where 3 elements are of one type
    and 3 elements are of another type. -/
def permutations_of_multiset : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 3)

/-- The theorem states that the number of permutations of the multiset {1, 1, 1, 2, 2, 2}
    is equal to 20. -/
theorem permutations_of_111222 : permutations_of_multiset = 20 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_111222_l768_76845


namespace NUMINAMATH_CALUDE_closest_fraction_to_two_thirds_l768_76851

theorem closest_fraction_to_two_thirds :
  let fractions : List ℚ := [4/7, 9/14, 20/31, 61/95, 73/110]
  let target : ℚ := 2/3
  let differences := fractions.map (fun x => |x - target|)
  differences.minimum? = some |73/110 - 2/3| :=
by sorry

end NUMINAMATH_CALUDE_closest_fraction_to_two_thirds_l768_76851


namespace NUMINAMATH_CALUDE_jeff_fills_ten_boxes_l768_76881

/-- The number of boxes Jeff can fill with his donuts -/
def boxes_filled (donuts_per_day : ℕ) (days : ℕ) (jeff_eats_per_day : ℕ) (chris_eats : ℕ) (donuts_per_box : ℕ) : ℕ :=
  ((donuts_per_day * days) - (jeff_eats_per_day * days) - chris_eats) / donuts_per_box

/-- Theorem stating that Jeff can fill 10 boxes with his donuts -/
theorem jeff_fills_ten_boxes :
  boxes_filled 10 12 1 8 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_jeff_fills_ten_boxes_l768_76881


namespace NUMINAMATH_CALUDE_brad_profit_l768_76874

/-- Represents the sizes of lemonade glasses -/
inductive Size
| Small
| Medium
| Large

/-- Represents the lemonade stand data -/
structure LemonadeStand where
  yield_per_gallon : Size → ℕ
  cost_per_gallon : Size → ℚ
  price_per_glass : Size → ℚ
  gallons_made : Size → ℕ
  small_drunk : ℕ
  medium_bought : ℕ
  medium_spilled : ℕ
  large_unsold : ℕ

def brad_stand : LemonadeStand :=
  { yield_per_gallon := λ s => match s with
      | Size.Small => 16
      | Size.Medium => 10
      | Size.Large => 6
    cost_per_gallon := λ s => match s with
      | Size.Small => 2
      | Size.Medium => 7/2
      | Size.Large => 5
    price_per_glass := λ s => match s with
      | Size.Small => 1
      | Size.Medium => 7/4
      | Size.Large => 5/2
    gallons_made := λ _ => 2
    small_drunk := 4
    medium_bought := 3
    medium_spilled := 1
    large_unsold := 2 }

def total_cost (stand : LemonadeStand) : ℚ :=
  (stand.cost_per_gallon Size.Small * stand.gallons_made Size.Small) +
  (stand.cost_per_gallon Size.Medium * stand.gallons_made Size.Medium) +
  (stand.cost_per_gallon Size.Large * stand.gallons_made Size.Large)

def total_revenue (stand : LemonadeStand) : ℚ :=
  (stand.price_per_glass Size.Small * (stand.yield_per_gallon Size.Small * stand.gallons_made Size.Small - stand.small_drunk)) +
  (stand.price_per_glass Size.Medium * (stand.yield_per_gallon Size.Medium * stand.gallons_made Size.Medium - stand.medium_bought)) +
  (stand.price_per_glass Size.Large * (stand.yield_per_gallon Size.Large * stand.gallons_made Size.Large - stand.large_unsold))

def net_profit (stand : LemonadeStand) : ℚ :=
  total_revenue stand - total_cost stand

theorem brad_profit :
  net_profit brad_stand = 247/4 := by
  sorry

end NUMINAMATH_CALUDE_brad_profit_l768_76874


namespace NUMINAMATH_CALUDE_square_field_area_l768_76812

/-- The area of a square field with a diagonal of 20 meters is 200 square meters. -/
theorem square_field_area (diagonal : Real) (area : Real) :
  diagonal = 20 →
  area = diagonal^2 / 2 →
  area = 200 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l768_76812


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l768_76895

/-- The number of bottle caps Danny has after throwing some away and finding new ones -/
def final_bottle_caps (initial : ℕ) (thrown_away : ℕ) (found : ℕ) : ℕ :=
  initial - thrown_away + found

/-- Theorem stating that Danny's final bottle cap count is 67 -/
theorem danny_bottle_caps :
  final_bottle_caps 69 60 58 = 67 := by
  sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l768_76895


namespace NUMINAMATH_CALUDE_parallel_line_y_intercept_l768_76876

/-- A line in the 2D plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line. -/
def yIntercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

theorem parallel_line_y_intercept :
  ∀ (b : Line),
    b.slope = 3 →                -- b is parallel to y = 3x - 6
    b.point = (3, 4) →           -- b passes through (3, 4)
    yIntercept b = -5            -- the y-intercept of b is -5
  := by sorry

end NUMINAMATH_CALUDE_parallel_line_y_intercept_l768_76876


namespace NUMINAMATH_CALUDE_davids_pushups_l768_76809

/-- Given that Zachary did 19 push-ups and David did 39 more push-ups than Zachary,
    prove that David did 58 push-ups. -/
theorem davids_pushups (zachary_pushups : ℕ) (david_extra_pushups : ℕ) 
    (h1 : zachary_pushups = 19)
    (h2 : david_extra_pushups = 39) : 
    zachary_pushups + david_extra_pushups = 58 := by
  sorry

end NUMINAMATH_CALUDE_davids_pushups_l768_76809


namespace NUMINAMATH_CALUDE_systematic_sampling_problem_l768_76864

/-- Represents the systematic sampling problem --/
theorem systematic_sampling_problem 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (num_groups : ℕ) 
  (group_size : ℕ) 
  (sixteenth_group_num : ℕ) :
  total_students = 160 →
  sample_size = 20 →
  num_groups = 20 →
  group_size = total_students / num_groups →
  sixteenth_group_num = 126 →
  ∃ (first_group_num : ℕ), first_group_num = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_problem_l768_76864


namespace NUMINAMATH_CALUDE_length_of_PQ_l768_76834

/-- The problem setup -/
structure ProblemSetup where
  /-- Point R with coordinates (10, 15) -/
  R : ℝ × ℝ
  hR : R = (10, 15)
  
  /-- Line 1 with equation 7y = 24x -/
  line1 : ℝ → ℝ
  hline1 : ∀ x y, line1 y = 24 * x ∧ 7 * y = 24 * x
  
  /-- Line 2 with equation 15y = 4x -/
  line2 : ℝ → ℝ
  hline2 : ∀ x y, line2 y = 4/15 * x ∧ 15 * y = 4 * x
  
  /-- Point P on Line 1 -/
  P : ℝ × ℝ
  hP : line1 P.2 = 24 * P.1 ∧ 7 * P.2 = 24 * P.1
  
  /-- Point Q on Line 2 -/
  Q : ℝ × ℝ
  hQ : line2 Q.2 = 4/15 * Q.1 ∧ 15 * Q.2 = 4 * Q.1
  
  /-- R is the midpoint of PQ -/
  hMidpoint : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

/-- The main theorem -/
theorem length_of_PQ (setup : ProblemSetup) : 
  Real.sqrt ((setup.P.1 - setup.Q.1)^2 + (setup.P.2 - setup.Q.2)^2) = 3460 / 83 := by
  sorry

end NUMINAMATH_CALUDE_length_of_PQ_l768_76834


namespace NUMINAMATH_CALUDE_unique_intersection_l768_76804

/-- The quadratic function f(x) = bx^2 + bx + 2 -/
def f (b : ℝ) (x : ℝ) : ℝ := b * x^2 + b * x + 2

/-- The linear function g(x) = 2x + 4 -/
def g (x : ℝ) : ℝ := 2 * x + 4

/-- The discriminant of the quadratic equation resulting from equating f and g -/
def discriminant (b : ℝ) : ℝ := (b - 2)^2 + 8 * b

theorem unique_intersection (b : ℝ) : 
  (∃! x, f b x = g x) ↔ b = -2 := by sorry

end NUMINAMATH_CALUDE_unique_intersection_l768_76804


namespace NUMINAMATH_CALUDE_log_problem_l768_76808

theorem log_problem (p q r x : ℝ) (d : ℝ) 
  (hp : Real.log x / Real.log p = 2)
  (hq : Real.log x / Real.log q = 3)
  (hr : Real.log x / Real.log r = 6)
  (hd : Real.log x / Real.log (p * q * r) = d)
  (h_pos : p > 0 ∧ q > 0 ∧ r > 0 ∧ x > 0) : d = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l768_76808


namespace NUMINAMATH_CALUDE_equation_solution_l768_76828

theorem equation_solution : 
  ∀ x : ℤ, x * (x + 1) = 2014 * 2015 ↔ x = 2014 ∨ x = -2015 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l768_76828


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l768_76867

theorem cyclic_sum_inequality (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 1 →
  (a^6 / ((a - b) * (a - c))) + (b^6 / ((b - c) * (b - a))) + (c^6 / ((c - a) * (c - b))) > 15 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l768_76867


namespace NUMINAMATH_CALUDE_profit_division_l768_76869

theorem profit_division (profit_x profit_y total_profit : ℚ) : 
  profit_x / profit_y = 1/2 / (1/3) →
  profit_x - profit_y = 100 →
  profit_x + profit_y = total_profit →
  total_profit = 500 := by
sorry

end NUMINAMATH_CALUDE_profit_division_l768_76869


namespace NUMINAMATH_CALUDE_question_ratio_l768_76836

/-- Represents the number of questions submitted by each person -/
structure QuestionSubmission where
  rajat : ℕ
  vikas : ℕ
  abhishek : ℕ

/-- The total number of questions submitted -/
def total_questions : ℕ := 24

/-- Theorem stating the ratio of questions submitted -/
theorem question_ratio (qs : QuestionSubmission) 
  (h1 : qs.rajat + qs.vikas + qs.abhishek = total_questions)
  (h2 : qs.vikas = 6) :
  ∃ (r a : ℕ), r = qs.rajat ∧ a = qs.abhishek ∧ r + a = 18 :=
by sorry

end NUMINAMATH_CALUDE_question_ratio_l768_76836


namespace NUMINAMATH_CALUDE_content_paths_count_l768_76839

/-- Represents the grid structure of the "CONTENT" word pattern --/
def ContentGrid : Type := Unit  -- Placeholder for the grid structure

/-- Represents a valid path in the ContentGrid --/
def ValidPath (grid : ContentGrid) : Type := Unit  -- Placeholder for path representation

/-- Counts the number of valid paths in the ContentGrid --/
def countValidPaths (grid : ContentGrid) : ℕ := sorry

/-- The main theorem stating that the number of valid paths is 127 --/
theorem content_paths_count (grid : ContentGrid) : countValidPaths grid = 127 := by
  sorry

end NUMINAMATH_CALUDE_content_paths_count_l768_76839


namespace NUMINAMATH_CALUDE_min_product_of_three_l768_76847

def S : Finset Int := {-9, -7, -5, 0, 4, 6, 8}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z → 
    a * b * c ≤ x * y * z) → 
  a * b * c = -336 :=
sorry

end NUMINAMATH_CALUDE_min_product_of_three_l768_76847


namespace NUMINAMATH_CALUDE_add_million_minutes_to_start_date_l768_76863

/-- Represents a date and time -/
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The start date and time -/
def startDateTime : DateTime :=
  { year := 2007, month := 4, day := 15, hour := 12, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : Nat := 1000000

/-- The expected end date and time -/
def expectedEndDateTime : DateTime :=
  { year := 2009, month := 3, day := 10, hour := 10, minute := 40 }

theorem add_million_minutes_to_start_date :
  addMinutes startDateTime minutesToAdd = expectedEndDateTime :=
sorry

end NUMINAMATH_CALUDE_add_million_minutes_to_start_date_l768_76863


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l768_76824

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l768_76824


namespace NUMINAMATH_CALUDE_right_triangle_tangent_circles_area_sum_l768_76891

theorem right_triangle_tangent_circles_area_sum :
  ∀ (r s t : ℝ),
  r > 0 → s > 0 → t > 0 →
  r + s = 6 →
  r + t = 8 →
  s + t = 10 →
  (6 : ℝ)^2 + 8^2 = 10^2 →
  π * (r^2 + s^2 + t^2) = 36 * π := by
sorry

end NUMINAMATH_CALUDE_right_triangle_tangent_circles_area_sum_l768_76891


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l768_76823

/-- Definition of the diamond operation -/
def diamond (a b : ℝ) : ℝ := 3 * a - b^2

/-- Theorem stating that if a ◇ 6 = 15, then a = 17 -/
theorem diamond_equation_solution (a : ℝ) : diamond a 6 = 15 → a = 17 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l768_76823


namespace NUMINAMATH_CALUDE_least_number_divisible_by_multiple_l768_76898

theorem least_number_divisible_by_multiple (n : ℕ) : n = 856 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 8) = 24 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 8) = 32 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 8) = 36 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 8) = 54 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ : ℕ, (n + 8) = 24 * k₁ ∧ (n + 8) = 32 * k₂ ∧ (n + 8) = 36 * k₃ ∧ (n + 8) = 54 * k₄) :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_multiple_l768_76898


namespace NUMINAMATH_CALUDE_polygon_sides_l768_76879

/-- A convex polygon with the sum of all angles except one equal to 2790° has 18 sides -/
theorem polygon_sides (n : ℕ) (angle_sum : ℝ) : 
  n > 2 →
  angle_sum = 2790 →
  (n - 2) * 180 > angle_sum →
  (n - 1) * 180 ≥ angle_sum →
  n = 18 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l768_76879
