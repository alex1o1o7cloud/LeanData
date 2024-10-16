import Mathlib

namespace NUMINAMATH_CALUDE_prob_even_after_removal_l3088_308847

/-- Probability of selecting a dot from a face with n dots -/
def probSelectDot (n : ℕ) : ℚ := n / 21

/-- Probability that a face with n dots remains even after removing two dots -/
def probRemainsEven (n : ℕ) : ℚ :=
  if n % 2 = 0
  then 1 - probSelectDot n * ((n - 1) / 20)
  else probSelectDot n * ((n - 1) / 20)

/-- The probability of rolling an even number of dots after removing two random dots -/
def probEvenAfterRemoval : ℚ :=
  (1 / 6) * (probRemainsEven 1 + probRemainsEven 2 + probRemainsEven 3 +
             probRemainsEven 4 + probRemainsEven 5 + probRemainsEven 6)

theorem prob_even_after_removal :
  probEvenAfterRemoval = 167 / 630 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_after_removal_l3088_308847


namespace NUMINAMATH_CALUDE_parallel_condition_l3088_308829

/-- Two lines are parallel if and only if their slopes are equal and they have different y-intercepts -/
def are_parallel (m : ℝ) : Prop :=
  ((-1 : ℝ) / (1 + m) = -m / 2) ∧ 
  ((2 - m) / (1 + m) ≠ -4)

/-- Line l₁: x + (1+m)y = 2-m -/
def line_l1 (m : ℝ) (x y : ℝ) : Prop :=
  x + (1 + m) * y = 2 - m

/-- Line l₂: 2mx + 4y = -16 -/
def line_l2 (m : ℝ) (x y : ℝ) : Prop :=
  2 * m * x + 4 * y = -16

theorem parallel_condition :
  ∀ m : ℝ, (m = 1) ↔ are_parallel m :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l3088_308829


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l3088_308887

theorem sum_of_solutions_quadratic (a b c d e : ℝ) :
  (∀ x, a * x^2 + b * x + c = d * x + e) →
  (a ≠ 0) →
  (∃ x y, a * x^2 + b * x + c = d * x + e ∧ 
          a * y^2 + b * y + c = d * y + e ∧ 
          x ≠ y) →
  (x + y = -(b - d) / a) :=
by sorry

theorem sum_of_solutions_specific_quadratic :
  let a : ℝ := 2
  let b : ℝ := -8
  let c : ℝ := 5
  let d : ℝ := 4
  let e : ℝ := -20
  (∀ x, a * x^2 + b * x + c = d * x + e) →
  (∃ x y, a * x^2 + b * x + c = d * x + e ∧ 
          a * y^2 + b * y + c = d * y + e ∧ 
          x ≠ y) →
  (x + y = 6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l3088_308887


namespace NUMINAMATH_CALUDE_abs_neg_two_equals_two_l3088_308854

theorem abs_neg_two_equals_two : abs (-2 : ℤ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_equals_two_l3088_308854


namespace NUMINAMATH_CALUDE_art_students_l3088_308890

/-- Proves that the number of students taking art is 20 -/
theorem art_students (total : ℕ) (music : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 500)
  (h2 : music = 30)
  (h3 : both = 10)
  (h4 : neither = 460) :
  total - neither - (music - both) = 20 := by
  sorry

end NUMINAMATH_CALUDE_art_students_l3088_308890


namespace NUMINAMATH_CALUDE_network_connections_l3088_308888

/-- Given a network of switches where each switch is connected to exactly
    four others, this function calculates the total number of connections. -/
def calculate_connections (num_switches : ℕ) : ℕ :=
  (num_switches * 4) / 2

/-- Theorem stating that in a network of 30 switches, where each switch
    is directly connected to exactly 4 other switches, the total number
    of connections is 60. -/
theorem network_connections :
  calculate_connections 30 = 60 := by
  sorry

#eval calculate_connections 30

end NUMINAMATH_CALUDE_network_connections_l3088_308888


namespace NUMINAMATH_CALUDE_smallest_class_size_l3088_308822

theorem smallest_class_size (n : ℕ) (scores : Fin n → ℕ) : 
  (∀ i, scores i ≥ 50) →  -- Each student scored at least 50
  (∃ a b c d, scores a = 80 ∧ scores b = 80 ∧ scores c = 80 ∧ scores d = 80) →  -- Four students achieved the maximum score
  (Finset.sum Finset.univ scores / n = 65) →  -- The average score was 65
  (∀ i, scores i ≤ 80) →  -- Maximum possible score is 80
  n ≥ 8  -- The smallest possible number of students is at least 8
:= by sorry

#check smallest_class_size

end NUMINAMATH_CALUDE_smallest_class_size_l3088_308822


namespace NUMINAMATH_CALUDE_jake_peaches_l3088_308860

/-- Given information about peaches owned by Steven, Jill, and Jake -/
theorem jake_peaches (steven_peaches : ℕ) (jill_peaches : ℕ) (jake_peaches : ℕ)
  (h1 : steven_peaches = 15)
  (h2 : steven_peaches = jill_peaches + 14)
  (h3 : jake_peaches = steven_peaches - 7) :
  jake_peaches = 8 := by
  sorry

end NUMINAMATH_CALUDE_jake_peaches_l3088_308860


namespace NUMINAMATH_CALUDE_solve_system_for_x_l3088_308864

theorem solve_system_for_x (x y : ℚ) 
  (eq1 : 3 * x - 4 * y = 8) 
  (eq2 : 2 * x + 3 * y = 1) : 
  x = 28 / 17 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_for_x_l3088_308864


namespace NUMINAMATH_CALUDE_beau_age_proof_l3088_308863

theorem beau_age_proof (sons_age_today : ℕ) (sons_are_triplets : Bool) : 
  sons_age_today = 16 ∧ sons_are_triplets = true → 42 = (sons_age_today - 3) * 3 + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_beau_age_proof_l3088_308863


namespace NUMINAMATH_CALUDE_estimated_probability_is_0_30_l3088_308870

/-- Represents a single shot result -/
inductive ShotResult
| Hit
| Miss

/-- Represents the result of three shots -/
structure ThreeShotResult :=
  (shot1 shot2 shot3 : ShotResult)

/-- Checks if a ThreeShotResult has exactly two hits -/
def hasTwoHits (result : ThreeShotResult) : Bool :=
  match result with
  | ⟨ShotResult.Hit, ShotResult.Hit, ShotResult.Miss⟩ => true
  | ⟨ShotResult.Hit, ShotResult.Miss, ShotResult.Hit⟩ => true
  | ⟨ShotResult.Miss, ShotResult.Hit, ShotResult.Hit⟩ => true
  | _ => false

/-- Converts a digit to a ShotResult -/
def digitToShotResult (d : Nat) : ShotResult :=
  if d ≤ 3 then ShotResult.Hit else ShotResult.Miss

/-- Converts a three-digit number to a ThreeShotResult -/
def numberToThreeShotResult (n : Nat) : ThreeShotResult :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  ⟨digitToShotResult d1, digitToShotResult d2, digitToShotResult d3⟩

/-- The list of simulation results -/
def simulationResults : List Nat :=
  [321, 421, 191, 925, 271, 932, 800, 478, 589, 663,
   531, 297, 396, 021, 546, 388, 230, 113, 507, 965]

/-- Counts the number of ThreeShotResults with exactly two hits -/
def countTwoHits (results : List Nat) : Nat :=
  results.filter (fun n => hasTwoHits (numberToThreeShotResult n)) |>.length

/-- Theorem: The estimated probability of hitting the bullseye exactly twice in three shots is 0.30 -/
theorem estimated_probability_is_0_30 :
  (countTwoHits simulationResults : Rat) / simulationResults.length = 0.30 := by
  sorry


end NUMINAMATH_CALUDE_estimated_probability_is_0_30_l3088_308870


namespace NUMINAMATH_CALUDE_liquid_X_percentage_l3088_308805

/-- The percentage of liquid X in solution P -/
def percentage_X_in_P : ℝ := sorry

/-- The percentage of liquid X in solution Q -/
def percentage_X_in_Q : ℝ := 0.015

/-- The weight of solution P in grams -/
def weight_P : ℝ := 200

/-- The weight of solution Q in grams -/
def weight_Q : ℝ := 800

/-- The percentage of liquid X in the resulting mixture -/
def percentage_X_in_mixture : ℝ := 0.013

theorem liquid_X_percentage :
  percentage_X_in_P * weight_P + percentage_X_in_Q * weight_Q =
  percentage_X_in_mixture * (weight_P + weight_Q) ∧
  percentage_X_in_P = 0.005 := by sorry

end NUMINAMATH_CALUDE_liquid_X_percentage_l3088_308805


namespace NUMINAMATH_CALUDE_stating_count_valid_starters_l3088_308858

/-- 
Represents the number of boys who can start the game to ensure it goes for at least a full turn 
in a circular arrangement of m boys and n girls.
-/
def valid_starters (m n : ℕ) : ℕ :=
  m - n

/-- 
Theorem stating that the number of valid starters is m - n, 
given that there are more boys than girls.
-/
theorem count_valid_starters (m n : ℕ) (h : m > n) : 
  valid_starters m n = m - n := by
  sorry

end NUMINAMATH_CALUDE_stating_count_valid_starters_l3088_308858


namespace NUMINAMATH_CALUDE_average_speed_on_time_l3088_308899

/-- The average speed needed to reach the destination on time given the conditions -/
theorem average_speed_on_time (total_distance : ℝ) (late_speed : ℝ) (late_time : ℝ) :
  total_distance = 70 →
  late_speed = 35 →
  late_time = 0.25 →
  (total_distance / late_speed) - late_time = total_distance / 40 :=
by sorry

end NUMINAMATH_CALUDE_average_speed_on_time_l3088_308899


namespace NUMINAMATH_CALUDE_expression_value_at_negative_two_l3088_308861

theorem expression_value_at_negative_two :
  let x : ℤ := -2
  (3 * x + 4)^2 - 2 * x = 8 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_negative_two_l3088_308861


namespace NUMINAMATH_CALUDE_chess_team_arrangement_l3088_308884

/-- The number of boys on the chess team -/
def num_boys : ℕ := 3

/-- The number of girls on the chess team -/
def num_girls : ℕ := 2

/-- The total number of students on the chess team -/
def total_students : ℕ := num_boys + num_girls

/-- The number of ways to arrange the team with girls at the ends and boys in the middle -/
def num_arrangements : ℕ := (Nat.factorial num_girls) * (Nat.factorial num_boys)

theorem chess_team_arrangement :
  num_arrangements = 12 :=
sorry

end NUMINAMATH_CALUDE_chess_team_arrangement_l3088_308884


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3088_308843

/-- 
Given a quadratic equation (m-1)x^2 + 3x - 1 = 0,
prove that for the equation to have real roots,
m must satisfy: m ≥ -5/4 and m ≠ 1
-/
theorem quadratic_real_roots_condition (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + 3 * x - 1 = 0) ↔ 
  (m ≥ -5/4 ∧ m ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3088_308843


namespace NUMINAMATH_CALUDE_max_a_value_max_a_value_achieved_l3088_308880

theorem max_a_value (a : ℝ) : 
  (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → 4 * a * x ≤ Real.exp (x + y - 2) + Real.exp (x - y - 2) + 2) →
  a ≤ (1 : ℝ) / 2 :=
by sorry

theorem max_a_value_achieved : 
  ∃ a : ℝ, a = (1 : ℝ) / 2 ∧ 
  (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → 4 * a * x ≤ Real.exp (x + y - 2) + Real.exp (x - y - 2) + 2) :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_max_a_value_achieved_l3088_308880


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3088_308814

theorem arithmetic_sequence_ratio : 
  let n1 := (60 - 4) / 4 + 1
  let n2 := (75 - 5) / 5 + 1
  let sum1 := n1 * (4 + 60) / 2
  let sum2 := n2 * (5 + 75) / 2
  sum1 / sum2 = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3088_308814


namespace NUMINAMATH_CALUDE_seating_theorem_l3088_308832

/-- Represents a seating arrangement --/
structure SeatingArrangement where
  total_people : ℕ
  rows_with_seven : ℕ
  rows_with_six : ℕ

/-- Checks if a seating arrangement is valid --/
def is_valid_arrangement (s : SeatingArrangement) : Prop :=
  s.total_people = 53 ∧
  s.rows_with_seven * 7 + s.rows_with_six * 6 = s.total_people

/-- The theorem to be proved --/
theorem seating_theorem :
  ∃ (s : SeatingArrangement), is_valid_arrangement s ∧ s.rows_with_seven = 5 :=
sorry

end NUMINAMATH_CALUDE_seating_theorem_l3088_308832


namespace NUMINAMATH_CALUDE_conic_section_types_l3088_308878

/-- The equation y^4 - 6x^4 = 3y^2 - 4 represents the union of a hyperbola and an ellipse -/
theorem conic_section_types (x y : ℝ) : 
  y^4 - 6*x^4 = 3*y^2 - 4 → 
  (∃ (a b : ℝ), y^2 - a*x^2 = b ∧ a > 0 ∧ b > 0) ∧ 
  (∃ (c d : ℝ), y^2 + c*x^2 = d ∧ c > 0 ∧ d > 0) := by
sorry

end NUMINAMATH_CALUDE_conic_section_types_l3088_308878


namespace NUMINAMATH_CALUDE_proportion_equality_l3088_308813

theorem proportion_equality (x : ℚ) : 
  (3 : ℚ) / 5 = 12 / 20 ∧ x / 10 = 16 / 40 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l3088_308813


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sum_l3088_308885

theorem arithmetic_geometric_sum (a₁ d : ℚ) (g₁ r : ℚ) (n : ℕ) 
  (h₁ : a₁ = 15)
  (h₂ : d = 0.2)
  (h₃ : g₁ = 15)
  (h₄ : r = 2)
  (h₅ : n = 101) :
  (n : ℚ) * (a₁ + (a₁ + (n - 1) * d)) / 2 + g₁ * (r^n - 1) / (r - 1) = 15 * (2^101 - 1) + 2525 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sum_l3088_308885


namespace NUMINAMATH_CALUDE_circle_hexagon_area_difference_l3088_308812

theorem circle_hexagon_area_difference (r : ℝ) (s : ℝ) : 
  r = (Real.sqrt 2) / 2 →
  s = 1 →
  (π * r^2) - (3 * Real.sqrt 3 / 2 * s^2) = π / 2 - 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_hexagon_area_difference_l3088_308812


namespace NUMINAMATH_CALUDE_prism_18_edges_has_8_faces_l3088_308802

/-- Represents a prism -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism -/
def num_faces (p : Prism) : ℕ :=
  let base_sides := p.edges / 3
  base_sides + 2

/-- Theorem: A prism with 18 edges has 8 faces -/
theorem prism_18_edges_has_8_faces :
  ∀ p : Prism, p.edges = 18 → num_faces p = 8 := by
  sorry

end NUMINAMATH_CALUDE_prism_18_edges_has_8_faces_l3088_308802


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l3088_308897

theorem sum_of_reciprocals_of_roots (p q : ℝ) : 
  p^2 - 20*p + 9 = 0 → q^2 - 20*q + 9 = 0 → p ≠ q → (1/p + 1/q) = 20/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l3088_308897


namespace NUMINAMATH_CALUDE_inequality_proof_l3088_308820

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ (3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3088_308820


namespace NUMINAMATH_CALUDE_maxwell_walking_speed_l3088_308886

-- Define the given constants
def total_distance : ℝ := 36
def brad_speed : ℝ := 4
def maxwell_distance : ℝ := 12

-- Define Maxwell's speed as a variable
def maxwell_speed : ℝ := sorry

-- Theorem statement
theorem maxwell_walking_speed :
  maxwell_speed = 8 :=
by
  -- The proof would go here, but we're using sorry to skip it
  sorry

end NUMINAMATH_CALUDE_maxwell_walking_speed_l3088_308886


namespace NUMINAMATH_CALUDE_composition_inverse_implies_value_l3088_308828

-- Define the functions
def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 6
def h (a b x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem composition_inverse_implies_value (a b : ℝ) :
  (∀ x, h a b x = x - 9) →
  2 * a - 3 * b = 22 := by
  sorry

end NUMINAMATH_CALUDE_composition_inverse_implies_value_l3088_308828


namespace NUMINAMATH_CALUDE_cube_difference_of_squares_l3088_308896

theorem cube_difference_of_squares (a : ℕ+) :
  ∃ (x y : ℤ), x^2 - y^2 = (a : ℤ)^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_of_squares_l3088_308896


namespace NUMINAMATH_CALUDE_prime_triplets_theorem_l3088_308891

/-- A prime triplet (a, b, c) satisfying the given conditions -/
structure PrimeTriplet where
  a : Nat
  b : Nat
  c : Nat
  h1 : a < b
  h2 : b < c
  h3 : c < 100
  h4 : Nat.Prime a
  h5 : Nat.Prime b
  h6 : Nat.Prime c
  h7 : (b + 1 - (a + 1)) * (c + 1 - (b + 1)) = (b + 1) * (b + 1 - (a + 1))

/-- The set of all valid prime triplets -/
def validTriplets : Set PrimeTriplet := {
  ⟨2, 5, 11, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩,
  ⟨5, 11, 23, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩,
  ⟨7, 11, 23, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩,
  ⟨11, 23, 47, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩
}

/-- The main theorem -/
theorem prime_triplets_theorem :
  ∀ t : PrimeTriplet, t ∈ validTriplets := by
  sorry

end NUMINAMATH_CALUDE_prime_triplets_theorem_l3088_308891


namespace NUMINAMATH_CALUDE_intersection_point_l3088_308856

/-- The first curve equation -/
def curve1 (x : ℝ) : ℝ := x^3 + 3*x^2 + 4*x - 5

/-- The second curve equation -/
def curve2 (x : ℝ) : ℝ := 2*x^2 + 11

/-- Theorem stating that (2, 19) is the only intersection point of the two curves -/
theorem intersection_point : 
  (∃! p : ℝ × ℝ, curve1 p.1 = curve2 p.1 ∧ p.2 = curve1 p.1) ∧ 
  (∀ p : ℝ × ℝ, curve1 p.1 = curve2 p.1 → p = (2, 19)) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3088_308856


namespace NUMINAMATH_CALUDE_square_land_side_length_l3088_308824

theorem square_land_side_length (area : ℝ) (side : ℝ) :
  area = 400 →
  side * side = area →
  side = 20 := by
sorry

end NUMINAMATH_CALUDE_square_land_side_length_l3088_308824


namespace NUMINAMATH_CALUDE_total_animals_is_100_l3088_308804

/-- The number of rabbits -/
def num_rabbits : ℕ := 4

/-- The number of ducks -/
def num_ducks : ℕ := num_rabbits + 12

/-- The number of chickens -/
def num_chickens : ℕ := 5 * num_ducks

/-- The total number of animals -/
def total_animals : ℕ := num_chickens + num_ducks + num_rabbits

theorem total_animals_is_100 : total_animals = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_is_100_l3088_308804


namespace NUMINAMATH_CALUDE_charms_per_necklace_is_10_l3088_308848

/-- The number of charms used to make each necklace -/
def charms_per_necklace : ℕ := sorry

/-- The cost of each charm in dollars -/
def charm_cost : ℕ := 15

/-- The selling price of each necklace in dollars -/
def necklace_price : ℕ := 200

/-- The number of necklaces sold -/
def necklaces_sold : ℕ := 30

/-- The total profit in dollars -/
def total_profit : ℕ := 1500

theorem charms_per_necklace_is_10 :
  charms_per_necklace = 10 ∧
  charm_cost = 15 ∧
  necklace_price = 200 ∧
  necklaces_sold = 30 ∧
  total_profit = 1500 ∧
  necklaces_sold * (necklace_price - charms_per_necklace * charm_cost) = total_profit :=
sorry

end NUMINAMATH_CALUDE_charms_per_necklace_is_10_l3088_308848


namespace NUMINAMATH_CALUDE_monday_pages_proof_l3088_308846

def total_pages : ℕ := 158
def tuesday_pages : ℕ := 38
def wednesday_pages : ℕ := 61
def thursday_pages : ℕ := 12
def friday_pages : ℕ := 2 * thursday_pages

theorem monday_pages_proof :
  total_pages - (tuesday_pages + wednesday_pages + thursday_pages + friday_pages) = 23 := by
  sorry

end NUMINAMATH_CALUDE_monday_pages_proof_l3088_308846


namespace NUMINAMATH_CALUDE_no_multiples_of_five_end_in_two_l3088_308871

theorem no_multiples_of_five_end_in_two :
  {n : ℕ | n > 0 ∧ n < 500 ∧ n % 5 = 0 ∧ n % 10 = 2} = ∅ := by
  sorry

end NUMINAMATH_CALUDE_no_multiples_of_five_end_in_two_l3088_308871


namespace NUMINAMATH_CALUDE_school_trip_photos_l3088_308809

theorem school_trip_photos (claire_photos : ℕ) (lisa_photos : ℕ) (robert_photos : ℕ) :
  claire_photos = 10 →
  lisa_photos = 3 * claire_photos →
  robert_photos = claire_photos + 20 →
  lisa_photos + robert_photos = 60 := by
  sorry

end NUMINAMATH_CALUDE_school_trip_photos_l3088_308809


namespace NUMINAMATH_CALUDE_max_product_l3088_308806

def digits : Finset Nat := {3, 5, 6, 8, 9}

def is_valid_pair (a b c d e : Nat) : Prop :=
  {a, b, c, d, e} = digits ∧ 
  100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c < 1000 ∧
  10 ≤ 10 * d + e ∧ 10 * d + e < 100

def product (a b c d e : Nat) : Nat :=
  (100 * a + 10 * b + c) * (10 * d + e)

theorem max_product :
  ∀ a b c d e : Nat, is_valid_pair a b c d e →
    product a b c d e ≤ product 9 5 3 8 6 :=
by sorry

end NUMINAMATH_CALUDE_max_product_l3088_308806


namespace NUMINAMATH_CALUDE_odometer_sum_squares_l3088_308807

/-- Represents the odometer reading as a triple of natural numbers -/
structure OdometerReading where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : b ≥ 1
  h2 : a + b + c ≤ 9

/-- Represents Liam's car trip -/
structure CarTrip where
  speed : ℕ
  hours : ℕ
  initial : OdometerReading
  final : OdometerReading
  h1 : speed = 60
  h2 : final.a = initial.b
  h3 : final.b = initial.c
  h4 : final.c = initial.a
  h5 : 100 * final.b + 10 * final.c + final.a - (100 * initial.a + 10 * initial.b + initial.c) = speed * hours

theorem odometer_sum_squares (trip : CarTrip) : 
  trip.initial.a^2 + trip.initial.b^2 + trip.initial.c^2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_odometer_sum_squares_l3088_308807


namespace NUMINAMATH_CALUDE_units_digit_of_19_times_37_l3088_308874

theorem units_digit_of_19_times_37 : (19 * 37) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_19_times_37_l3088_308874


namespace NUMINAMATH_CALUDE_sin_6theta_l3088_308857

theorem sin_6theta (θ : ℝ) :
  Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 4 →
  Real.sin (6 * θ) = -855 * Real.sqrt 2 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_sin_6theta_l3088_308857


namespace NUMINAMATH_CALUDE_intersection_point_implies_m_equals_six_l3088_308816

theorem intersection_point_implies_m_equals_six (m : ℕ+) 
  (h : ∃ (x y : ℤ), 13 * x + 11 * y = 700 ∧ y = m * x - 1) : m = 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_implies_m_equals_six_l3088_308816


namespace NUMINAMATH_CALUDE_domain_implies_k_range_inequality_solution_set_l3088_308882

-- Problem I
theorem domain_implies_k_range (f : ℝ → ℝ) (h : ∀ x, ∃ y, f x = y) :
  (∀ x, f x = Real.sqrt (x^2 - x * k - k)) → k ∈ Set.Icc (-4) 0 := by sorry

-- Problem II
theorem inequality_solution_set (a : ℝ) :
  {x : ℝ | (x - a) * (x + a - 1) > 0} =
    if a = 1/2 then
      {x : ℝ | x ≠ 1/2}
    else if a < 1/2 then
      {x : ℝ | x > 1 - a ∨ x < a}
    else
      {x : ℝ | x > a ∨ x < 1 - a} := by sorry

end NUMINAMATH_CALUDE_domain_implies_k_range_inequality_solution_set_l3088_308882


namespace NUMINAMATH_CALUDE_complement_of_union_A_B_l3088_308819

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x ≤ 1}

-- Define set B
def B : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_of_union_A_B : 
  (A ∪ B)ᶜ = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_A_B_l3088_308819


namespace NUMINAMATH_CALUDE_no_two_distinct_roots_ellipse_slope_product_constant_l3088_308867

-- Statement for ①
theorem no_two_distinct_roots (f : ℝ → ℝ) (h : Monotone f) :
  ¬∃ k : ℝ, ∃ x y : ℝ, x ≠ y ∧ f x + k = 0 ∧ f y + k = 0 :=
sorry

-- Statement for ④
theorem ellipse_slope_product_constant (a b : ℝ) (h : a > b) (h' : b > 0) :
  ∃ c : ℝ, ∀ m n : ℝ, 
    (b^2 * m^2 + a^2 * n^2 = a^2 * b^2) →
    (n / (m + a)) * (n / (m - a)) = c :=
sorry

end NUMINAMATH_CALUDE_no_two_distinct_roots_ellipse_slope_product_constant_l3088_308867


namespace NUMINAMATH_CALUDE_outer_circle_diameter_l3088_308836

/-- Proves that given an outer circle with diameter D and an inner circle with diameter 24,
    if 0.36 of the outer circle's surface is not covered by the inner circle,
    then the diameter of the outer circle is 30. -/
theorem outer_circle_diameter
  (D : ℝ) -- Diameter of the outer circle
  (h1 : D > 0) -- Diameter is positive
  (h2 : π * (D / 2)^2 - π * 12^2 = 0.36 * π * (D / 2)^2) -- Condition about uncovered area
  : D = 30 := by
  sorry


end NUMINAMATH_CALUDE_outer_circle_diameter_l3088_308836


namespace NUMINAMATH_CALUDE_centroid_positions_count_l3088_308872

/-- A point on the perimeter of the square -/
structure PerimeterPoint where
  x : Fin 21
  y : Fin 21
  on_perimeter : (x = 0 ∨ x = 20) ∨ (y = 0 ∨ y = 20)

/-- The centroid of a triangle -/
def centroid (p q r : PerimeterPoint) : ℚ × ℚ :=
  ((p.x + q.x + r.x : ℚ) / 3, (p.y + q.y + r.y : ℚ) / 3)

/-- Predicate for valid centroid positions -/
def is_valid_centroid (c : ℚ × ℚ) : Prop :=
  0 < c.1 ∧ c.1 < 20 ∧ 0 < c.2 ∧ c.2 < 20

/-- The main theorem -/
theorem centroid_positions_count :
  ∃ (valid_centroids : Finset (ℚ × ℚ)),
    (∀ c ∈ valid_centroids, is_valid_centroid c) ∧
    (∀ p q r : PerimeterPoint, p ≠ q ∧ q ≠ r ∧ p ≠ r →
      centroid p q r ∈ valid_centroids) ∧
    valid_centroids.card = 3481 :=
  sorry

end NUMINAMATH_CALUDE_centroid_positions_count_l3088_308872


namespace NUMINAMATH_CALUDE_cosine_sum_theorem_l3088_308875

theorem cosine_sum_theorem : 
  Real.cos 0 ^ 4 + Real.cos (π / 6) ^ 4 + Real.cos (π / 3) ^ 4 + Real.cos (π / 2) ^ 4 + 
  Real.cos (2 * π / 3) ^ 4 + Real.cos (5 * π / 6) ^ 4 + Real.cos π ^ 4 = 13 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_theorem_l3088_308875


namespace NUMINAMATH_CALUDE_solution_sets_l3088_308826

-- Define the solution sets
def S (c b : ℝ) : Set ℝ := {x | c * x^2 + x + b < 0}
def M (b c : ℝ) : Set ℝ := {x | b * x^2 + x + c > 0}
def N (a : ℝ) : Set ℝ := {x | x^2 + x < a^2 - a}

-- State the theorem
theorem solution_sets :
  ∃ (c b : ℝ),
    (S c b = {x | -1 < x ∧ x < 1/2}) ∧
    (∃ (a : ℝ), M b c ∪ (Set.univ \ N a) = Set.univ) →
    (M b c = {x | -1 < x ∧ x < 2}) ∧
    {a : ℝ | 0 ≤ a ∧ a ≤ 1} = {a : ℝ | ∃ (b c : ℝ), M b c ∪ (Set.univ \ N a) = Set.univ} :=
by sorry

end NUMINAMATH_CALUDE_solution_sets_l3088_308826


namespace NUMINAMATH_CALUDE_unique_a_value_l3088_308833

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- State the theorem
theorem unique_a_value : ∃! a : ℝ, (A a ∩ B).Nonempty ∧ (A a ∩ C = ∅) ∧ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l3088_308833


namespace NUMINAMATH_CALUDE_probability_five_blue_marbles_l3088_308898

def total_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 4
def total_draws : ℕ := 8
def blue_draws : ℕ := 5

def probability_blue : ℚ := blue_marbles / total_marbles
def probability_red : ℚ := red_marbles / total_marbles

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem probability_five_blue_marbles :
  (binomial_coefficient total_draws blue_draws : ℚ) * 
  (probability_blue ^ blue_draws) * 
  (probability_red ^ (total_draws - blue_draws)) = 1792 / 6561 := by
sorry

end NUMINAMATH_CALUDE_probability_five_blue_marbles_l3088_308898


namespace NUMINAMATH_CALUDE_two_intersection_points_l3088_308841

/-- Define the first curve -/
def curve1 (x y : ℝ) : Prop :=
  (x + 2*y - 6) * (2*x - y + 4) = 0

/-- Define the second curve -/
def curve2 (x y : ℝ) : Prop :=
  (x - 3*y + 2) * (4*x + y - 14) = 0

/-- Define an intersection point -/
def is_intersection (x y : ℝ) : Prop :=
  curve1 x y ∧ curve2 x y

/-- The theorem stating that there are exactly two distinct intersection points -/
theorem two_intersection_points :
  ∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧
    is_intersection p1.1 p1.2 ∧
    is_intersection p2.1 p2.2 ∧
    ∀ (x y : ℝ), is_intersection x y → (x, y) = p1 ∨ (x, y) = p2 :=
by
  sorry

end NUMINAMATH_CALUDE_two_intersection_points_l3088_308841


namespace NUMINAMATH_CALUDE_valid_pairs_eq_expected_l3088_308889

def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))

def valid_pairs : Finset (ℕ × ℕ) :=
  (divisors 660).product (divisors 72) |>.filter (λ (a, b) => a - b = 4)

theorem valid_pairs_eq_expected : valid_pairs = {(6, 2), (10, 6), (12, 8), (22, 18)} := by
  sorry

end NUMINAMATH_CALUDE_valid_pairs_eq_expected_l3088_308889


namespace NUMINAMATH_CALUDE_carol_birthday_invitations_l3088_308894

/-- The number of friends Carol wants to invite -/
def num_friends : ℕ := sorry

/-- The number of invitations in each package -/
def invitations_per_package : ℕ := 3

/-- The number of packages Carol bought -/
def packages_bought : ℕ := 2

/-- The number of extra invitations Carol needs to buy -/
def extra_invitations : ℕ := 3

/-- Theorem stating that the number of friends Carol wants to invite
    is equal to the sum of invitations in bought packs and extra invitations -/
theorem carol_birthday_invitations :
  num_friends = packages_bought * invitations_per_package + extra_invitations := by
  sorry

end NUMINAMATH_CALUDE_carol_birthday_invitations_l3088_308894


namespace NUMINAMATH_CALUDE_model_parameters_l3088_308851

/-- Given a model y = c * e^(k * x) where c > 0, and its logarithmic transformation
    z = ln y resulting in the linear regression equation z = 2x - 1,
    prove that k = 2 and c = 1/e. -/
theorem model_parameters (c : ℝ) (k : ℝ) :
  c > 0 →
  (∀ x y z : ℝ, y = c * Real.exp (k * x) → z = Real.log y → z = 2 * x - 1) →
  k = 2 ∧ c = 1 / Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_model_parameters_l3088_308851


namespace NUMINAMATH_CALUDE_cabbage_production_l3088_308892

theorem cabbage_production (last_year_side : ℕ) (this_year_side : ℕ) : 
  (this_year_side * this_year_side = last_year_side * last_year_side + 197) →
  (this_year_side = last_year_side + 1) →
  (this_year_side * this_year_side = 9801) := by
  sorry

#check cabbage_production

end NUMINAMATH_CALUDE_cabbage_production_l3088_308892


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l3088_308837

theorem sqrt_difference_inequality (x : ℝ) (h : x ≥ 4) :
  Real.sqrt x - Real.sqrt (x - 1) ≥ 1 / x := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l3088_308837


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l3088_308831

/-- The surface area of a sphere circumscribing a rectangular solid with edge lengths 2, 3, and 4 is 29π. -/
theorem circumscribed_sphere_surface_area (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 4) :
  let diagonal_squared := a^2 + b^2 + c^2
  let radius := Real.sqrt (diagonal_squared / 4)
  4 * Real.pi * radius^2 = 29 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l3088_308831


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l3088_308839

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (r1 r2 d : ℝ) : Prop := d = r1 + r2

/-- Given two circles with radii 2 and 3, whose centers are 5 units apart,
    prove that they are externally tangent -/
theorem circles_externally_tangent :
  let r1 : ℝ := 2
  let r2 : ℝ := 3
  let d : ℝ := 5
  externally_tangent r1 r2 d := by
sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l3088_308839


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l3088_308881

theorem unique_solution_for_exponential_equation :
  ∀ x y : ℕ, x ≥ 1 → y ≥ 1 → (3^x : ℤ) - 2^y = 7 → x = 2 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l3088_308881


namespace NUMINAMATH_CALUDE_average_listening_time_is_44_l3088_308810

/-- Represents the distribution of audience members and their listening durations -/
structure AudienceDistribution where
  total_audience : ℕ
  lecture_duration : ℕ
  full_listeners_percent : ℚ
  non_listeners_percent : ℚ
  half_listeners_percent : ℚ

/-- Calculates the average listening time given an audience distribution -/
def average_listening_time (dist : AudienceDistribution) : ℚ :=
  sorry

/-- The theorem stating that the average listening time is 44 minutes -/
theorem average_listening_time_is_44 (dist : AudienceDistribution) : 
  dist.lecture_duration = 90 ∧ 
  dist.full_listeners_percent = 30/100 ∧ 
  dist.non_listeners_percent = 15/100 ∧
  dist.half_listeners_percent = 40/100 * (1 - dist.full_listeners_percent - dist.non_listeners_percent) →
  average_listening_time dist = 44 :=
sorry

end NUMINAMATH_CALUDE_average_listening_time_is_44_l3088_308810


namespace NUMINAMATH_CALUDE_stock_ratio_proof_l3088_308830

def stock_problem (expensive_shares : ℕ) (other_shares : ℕ) (total_value : ℕ) (expensive_price : ℕ) : Prop :=
  ∃ (other_price : ℕ),
    expensive_shares * expensive_price + other_shares * other_price = total_value ∧
    expensive_price / other_price = 2

theorem stock_ratio_proof :
  stock_problem 14 26 2106 78 := by
  sorry

end NUMINAMATH_CALUDE_stock_ratio_proof_l3088_308830


namespace NUMINAMATH_CALUDE_shm_first_return_time_l3088_308865

/-- Time for a particle in Simple Harmonic Motion to first return to origin -/
theorem shm_first_return_time (m k : ℝ) (hm : m > 0) (hk : k > 0) :
  ∃ (t : ℝ), t = π * Real.sqrt (m / k) ∧ t > 0 := by
  sorry

end NUMINAMATH_CALUDE_shm_first_return_time_l3088_308865


namespace NUMINAMATH_CALUDE_conference_handshakes_l3088_308845

theorem conference_handshakes (n : ℕ) (h : n = 30) :
  (n * (n - 1)) / 2 = 435 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l3088_308845


namespace NUMINAMATH_CALUDE_sum_remainder_by_eight_l3088_308838

theorem sum_remainder_by_eight (n : ℤ) : ((8 - n) + (n + 4)) % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_by_eight_l3088_308838


namespace NUMINAMATH_CALUDE_masking_tape_wall_width_l3088_308835

theorem masking_tape_wall_width (total_tape : ℝ) (known_wall_width : ℝ) (known_wall_count : ℕ) (unknown_wall_count : ℕ) :
  total_tape = 20 →
  known_wall_width = 6 →
  known_wall_count = 2 →
  unknown_wall_count = 2 →
  (unknown_wall_count : ℝ) * (total_tape - known_wall_count * known_wall_width) / unknown_wall_count = 4 := by
sorry

end NUMINAMATH_CALUDE_masking_tape_wall_width_l3088_308835


namespace NUMINAMATH_CALUDE_complex_square_root_l3088_308817

theorem complex_square_root (z : ℂ) : z^2 = 3 - 4*I → z = 1 - 2*I ∨ z = -1 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_l3088_308817


namespace NUMINAMATH_CALUDE_length_of_CE_l3088_308842

/-- Given a plot ABCD with specific measurements, prove the length of CE -/
theorem length_of_CE (AF ED AE : ℝ) (area_ABCD : ℝ) :
  AF = 30 ∧ ED = 50 ∧ AE = 120 ∧ area_ABCD = 7200 →
  ∃ CE : ℝ, CE = 138 ∧
    area_ABCD = (1/2 * AE * ED) + (1/2 * (AF + CE) * ED) := by
  sorry

end NUMINAMATH_CALUDE_length_of_CE_l3088_308842


namespace NUMINAMATH_CALUDE_plot_length_is_56_l3088_308815

/-- Proves that the length of a rectangular plot is 56 meters given the specified conditions -/
theorem plot_length_is_56 (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 12 →
  perimeter = 2 * (length + breadth) →
  cost_per_meter = 26.5 →
  total_cost = 5300 →
  total_cost = cost_per_meter * perimeter →
  length = 56 := by
  sorry

end NUMINAMATH_CALUDE_plot_length_is_56_l3088_308815


namespace NUMINAMATH_CALUDE_solution_opposite_implies_a_l3088_308818

theorem solution_opposite_implies_a (a : ℝ) : 
  (∃ x : ℝ, 5 * x - 1 = 2 * x + a) ∧ 
  (∃ y : ℝ, 4 * y + 3 = 7) ∧
  (∀ x y : ℝ, (5 * x - 1 = 2 * x + a ∧ 4 * y + 3 = 7) → x = -y) →
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_solution_opposite_implies_a_l3088_308818


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3088_308852

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  sum_first_two : a 1 + a 2 = 324
  sum_third_fourth : a 3 + a 4 = 36

/-- The theorem to be proved -/
theorem geometric_sequence_sum (seq : GeometricSequence) : seq.a 5 + seq.a 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3088_308852


namespace NUMINAMATH_CALUDE_basketball_count_l3088_308895

theorem basketball_count :
  ∀ (basketballs volleyballs soccerballs : ℕ),
    basketballs + volleyballs + soccerballs = 100 →
    basketballs = 2 * volleyballs →
    volleyballs = soccerballs + 8 →
    basketballs = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_count_l3088_308895


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_implies_x_geq_one_l3088_308825

theorem sqrt_x_minus_one_real_implies_x_geq_one (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_implies_x_geq_one_l3088_308825


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3088_308834

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a + b + c = 3) : 
  ∀ x y z, x > 0 → y > 0 → z > 0 → x + y + z = 3 → 
  1 / (a + b) + 1 / c ≤ 1 / (x + y) + 1 / z :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3088_308834


namespace NUMINAMATH_CALUDE_y_derivative_l3088_308850

noncomputable def y (x : ℝ) : ℝ := Real.sin x + Real.exp x * Real.cos x

theorem y_derivative (x : ℝ) : 
  deriv y x = (1 + Real.exp x) * Real.cos x - Real.exp x * Real.sin x :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l3088_308850


namespace NUMINAMATH_CALUDE_expansion_coefficient_l3088_308883

theorem expansion_coefficient (m : ℤ) : 
  (Nat.choose 6 3 : ℤ) * m^3 = -160 → m = -2 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l3088_308883


namespace NUMINAMATH_CALUDE_third_derivative_of_y_l3088_308821

noncomputable def y (x : ℝ) : ℝ := (1 / x) * Real.sin (2 * x)

theorem third_derivative_of_y (x : ℝ) (hx : x ≠ 0) :
  (deriv^[3] y) x = ((-6 / x^4 + 12 / x^2) * Real.sin (2 * x) + 
                     (12 / x^3 - 8 / x) * Real.cos (2 * x)) :=
by sorry

end NUMINAMATH_CALUDE_third_derivative_of_y_l3088_308821


namespace NUMINAMATH_CALUDE_remaining_average_l3088_308844

theorem remaining_average (total : ℝ) (group1 : ℝ) (group2 : ℝ) :
  total = 6 * 2.8 ∧ group1 = 2 * 2.4 ∧ group2 = 2 * 2.3 →
  (total - group1 - group2) / 2 = 3.7 := by
sorry

end NUMINAMATH_CALUDE_remaining_average_l3088_308844


namespace NUMINAMATH_CALUDE_faith_change_is_ten_l3088_308893

/-- The change Faith receives from her purchase at the baking shop. -/
def faith_change : ℕ :=
  let flour_cost : ℕ := 5
  let cake_stand_cost : ℕ := 28
  let total_cost : ℕ := flour_cost + cake_stand_cost
  let bill_payment : ℕ := 2 * 20
  let coin_payment : ℕ := 3
  let total_payment : ℕ := bill_payment + coin_payment
  total_payment - total_cost

/-- Theorem stating that Faith receives $10 in change. -/
theorem faith_change_is_ten : faith_change = 10 := by
  sorry

end NUMINAMATH_CALUDE_faith_change_is_ten_l3088_308893


namespace NUMINAMATH_CALUDE_inequality_proof_l3088_308853

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b * c / a + c * a / b + a * b / c ≥ a + b + c) ∧
  (a + b + c = 1 → (1 - a) / a + (1 - b) / b + (1 - c) / c ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3088_308853


namespace NUMINAMATH_CALUDE_remainder_r_15_minus_1_l3088_308866

theorem remainder_r_15_minus_1 (r : ℝ) : (r^15 - 1) % (r - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_r_15_minus_1_l3088_308866


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3088_308827

/-- Given an arithmetic sequence {a_n} with positive terms, sum S_n, and common difference d,
    if {√S_n} is also arithmetic with the same difference d, then a_n = (2n - 1) / 4 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, S n = n * a 1 + n * (n - 1) / 2 * d) →
  (∀ n, a (n + 1) = a n + d) →
  (∀ n, Real.sqrt (S (n + 1)) = Real.sqrt (S n) + d) →
  ∀ n, a n = (2 * n - 1) / 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3088_308827


namespace NUMINAMATH_CALUDE_factorization_equality_l3088_308800

theorem factorization_equality (m n : ℝ) : 4 * m^3 * n - 16 * m * n^3 = 4 * m * n * (m + 2*n) * (m - 2*n) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3088_308800


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3088_308869

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 - 2*x - 3 = 0) ↔ ((x - 1)^2 = 4) := by
sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3088_308869


namespace NUMINAMATH_CALUDE_pet_store_bird_count_l3088_308801

/-- Calculates the total number of birds in a pet store with specific cage arrangements. -/
theorem pet_store_bird_count : 
  let total_cages : ℕ := 9
  let parrots_per_mixed_cage : ℕ := 2
  let parakeets_per_mixed_cage : ℕ := 3
  let cockatiels_per_mixed_cage : ℕ := 1
  let parakeets_per_special_cage : ℕ := 5
  let special_cage_frequency : ℕ := 3

  let special_cages : ℕ := total_cages / special_cage_frequency
  let mixed_cages : ℕ := total_cages - special_cages

  let total_parrots : ℕ := mixed_cages * parrots_per_mixed_cage
  let total_parakeets : ℕ := (mixed_cages * parakeets_per_mixed_cage) + (special_cages * parakeets_per_special_cage)
  let total_cockatiels : ℕ := mixed_cages * cockatiels_per_mixed_cage

  let total_birds : ℕ := total_parrots + total_parakeets + total_cockatiels
  
  total_birds = 51 := by sorry

end NUMINAMATH_CALUDE_pet_store_bird_count_l3088_308801


namespace NUMINAMATH_CALUDE_prob_double_is_one_seventh_l3088_308868

/-- Represents a domino set with integers from 0 to 12 -/
def DominoSet : Type := Unit

/-- The number of integers in the domino set -/
def num_integers : ℕ := 13

/-- The total number of domino tiles in the set -/
def total_tiles (ds : DominoSet) : ℕ := (num_integers * (num_integers + 1)) / 2

/-- The number of double tiles in the set -/
def num_doubles (ds : DominoSet) : ℕ := num_integers

/-- The probability of randomly selecting a double from the domino set -/
def prob_double (ds : DominoSet) : ℚ := (num_doubles ds : ℚ) / (total_tiles ds : ℚ)

theorem prob_double_is_one_seventh (ds : DominoSet) :
  prob_double ds = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_double_is_one_seventh_l3088_308868


namespace NUMINAMATH_CALUDE_volume_Q4_l3088_308840

/-- Represents the volume of the i-th polyhedron in the sequence --/
def Q (i : ℕ) : ℝ :=
  sorry

/-- The volume difference between consecutive polyhedra --/
def ΔQ (i : ℕ) : ℝ :=
  sorry

theorem volume_Q4 :
  Q 0 = 8 →
  (∀ i : ℕ, ΔQ (i + 1) = (1 / 2) * ΔQ i) →
  ΔQ 1 = 4 →
  Q 4 = 15.5 :=
by
  sorry

end NUMINAMATH_CALUDE_volume_Q4_l3088_308840


namespace NUMINAMATH_CALUDE_restaurant_location_l3088_308862

theorem restaurant_location (A B C : ℝ × ℝ) : 
  let road_y : ℝ := 0
  let A_x : ℝ := 0
  let A_y : ℝ := 300
  let B_y : ℝ := road_y
  let dist_AB : ℝ := 500
  A = (A_x, A_y) →
  B.2 = road_y →
  Real.sqrt ((B.1 - A_x)^2 + (B.2 - A_y)^2) = dist_AB →
  C.2 = road_y →
  Real.sqrt ((C.1 - A_x)^2 + (C.2 - A_y)^2) = Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) →
  C.1 = 200 := by
sorry

end NUMINAMATH_CALUDE_restaurant_location_l3088_308862


namespace NUMINAMATH_CALUDE_goods_train_length_l3088_308873

/-- The length of a goods train given relative speeds and passing time -/
theorem goods_train_length (v_passenger : ℝ) (v_goods : ℝ) (t_pass : ℝ) : 
  v_passenger = 80 → 
  v_goods = 32 → 
  t_pass = 9 →
  ∃ (length : ℝ), abs (length - 280) < 1 ∧ 
    length = (v_passenger + v_goods) * 1000 / 3600 * t_pass :=
by sorry

end NUMINAMATH_CALUDE_goods_train_length_l3088_308873


namespace NUMINAMATH_CALUDE_intersection_M_N_l3088_308808

def M : Set ℝ := {0, 2}
def N : Set ℝ := {x | 0 ≤ x ∧ x < 2}

theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3088_308808


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l3088_308849

theorem unknown_blanket_rate (price1 price2 avg_price : ℚ) 
  (count1 count2 count_unknown : ℕ) : 
  price1 = 100 → 
  price2 = 150 → 
  avg_price = 150 → 
  count1 = 4 → 
  count2 = 5 → 
  count_unknown = 2 → 
  (count1 * price1 + count2 * price2 + count_unknown * 
    ((count1 + count2 + count_unknown) * avg_price - count1 * price1 - count2 * price2) / count_unknown) / 
    (count1 + count2 + count_unknown) = avg_price → 
  ((count1 + count2 + count_unknown) * avg_price - count1 * price1 - count2 * price2) / count_unknown = 250 :=
by sorry

end NUMINAMATH_CALUDE_unknown_blanket_rate_l3088_308849


namespace NUMINAMATH_CALUDE_solution_inequality1_no_solution_system_l3088_308803

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := (2*x - 2)/3 ≤ 2 - (2*x + 2)/2

def inequality2 (x : ℝ) : Prop := 3*(x - 2) - 1 ≥ -4 - 2*(x - 2)

def inequality3 (x : ℝ) : Prop := (1/3)*(1 - 2*x) > (3*(2*x - 1))/2

-- Theorem for the first inequality
theorem solution_inequality1 : 
  ∀ x : ℝ, inequality1 x ↔ x ≤ 1 := by sorry

-- Theorem for the system of inequalities
theorem no_solution_system : 
  ¬∃ x : ℝ, inequality2 x ∧ inequality3 x := by sorry

end NUMINAMATH_CALUDE_solution_inequality1_no_solution_system_l3088_308803


namespace NUMINAMATH_CALUDE_power_product_l3088_308855

theorem power_product (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 8) : a^m * a^n = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_product_l3088_308855


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3088_308876

-- Define the ellipse Γ
def Γ : Set (ℝ × ℝ) := sorry

-- Define points A, B, and C
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry

-- State the theorem
theorem ellipse_foci_distance :
  -- AB is the major axis of ellipse Γ
  (∀ p ∈ Γ, (p.1 - A.1)^2 + (p.2 - A.2)^2 ≤ (B.1 - A.1)^2 + (B.2 - A.2)^2) →
  -- Point C is on Γ
  C ∈ Γ →
  -- Angle CBA = π/4
  Real.arccos ((C.1 - B.1) / Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)) = π/4 →
  -- AB = 4
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 4 →
  -- BC = √2
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = Real.sqrt 2 →
  -- The distance between the two foci is 4√6/3
  ∃ F₁ F₂ : ℝ × ℝ, F₁ ∈ Γ ∧ F₂ ∈ Γ ∧
    Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2) = 4 * Real.sqrt 6 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3088_308876


namespace NUMINAMATH_CALUDE_not_divisible_by_three_and_four_l3088_308879

theorem not_divisible_by_three_and_four (n : ℤ) : 
  ¬(∃ k : ℤ, n^2 + 1 = 3 * k) ∧ ¬(∃ m : ℤ, n^2 + 1 = 4 * m) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_three_and_four_l3088_308879


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l3088_308811

/-- 
Given a quadratic equation kx^2 - 2x - 1 = 0 with two distinct real roots,
prove that the range of values for k is k > -1 and k ≠ 0.
-/
theorem quadratic_equation_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2 * x - 1 = 0 ∧ k * y^2 - 2 * y - 1 = 0) →
  (k > -1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l3088_308811


namespace NUMINAMATH_CALUDE_number_triangle_problem_l3088_308823

theorem number_triangle_problem (x y : ℕ+) (h : x * y = 2022) : 
  (∃ (n : ℕ+), ∀ (m : ℕ+), (m * m ∣ x) ∧ (m * m ∣ y) → m ≤ n) ∧
  (∀ (n : ℕ+), (n * n ∣ x) ∧ (n * n ∣ y) → n = 1) :=
sorry

end NUMINAMATH_CALUDE_number_triangle_problem_l3088_308823


namespace NUMINAMATH_CALUDE_area_of_intersection_is_point_eight_l3088_308859

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space with slope-intercept form y = mx + b -/
structure Line2D where
  m : ℝ
  b : ℝ

/-- Calculates the area of intersection between two triangles -/
noncomputable def areaOfIntersection (a b c d e : Point2D) (lineAC lineDE : Line2D) : ℝ :=
  sorry

/-- Theorem: The area of intersection between two specific triangles is 0.8 square units -/
theorem area_of_intersection_is_point_eight :
  let a : Point2D := ⟨1, 4⟩
  let b : Point2D := ⟨0, 0⟩
  let c : Point2D := ⟨2, 0⟩
  let d : Point2D := ⟨0, 1⟩
  let e : Point2D := ⟨4, 0⟩
  let lineAC : Line2D := ⟨-4, 8⟩
  let lineDE : Line2D := ⟨-1/4, 1⟩
  areaOfIntersection a b c d e lineAC lineDE = 0.8 :=
by
  sorry

end NUMINAMATH_CALUDE_area_of_intersection_is_point_eight_l3088_308859


namespace NUMINAMATH_CALUDE_stratified_selection_count_l3088_308877

def female_students : ℕ := 8
def male_students : ℕ := 4
def total_selected : ℕ := 3

theorem stratified_selection_count :
  (Nat.choose female_students 2 * Nat.choose male_students 1) +
  (Nat.choose female_students 1 * Nat.choose male_students 2) = 112 :=
by sorry

end NUMINAMATH_CALUDE_stratified_selection_count_l3088_308877
