import Mathlib

namespace NUMINAMATH_CALUDE_complex_expression_equality_l906_90634

theorem complex_expression_equality : 
  (2 + 7/9)^(1/2 : ℝ) + (1/10)^(-2 : ℝ) + (2 + 10/27)^(-(2/3) : ℝ) - Real.pi^(0 : ℝ) + 37/48 = 807/8 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l906_90634


namespace NUMINAMATH_CALUDE_intersection_condition_subset_condition_l906_90626

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 8 ≤ 0}

-- Define set B as a function of m
def B (m : ℝ) : Set ℝ := {x | x^2 - (2*m-3)*x + m^2 - 3*m ≤ 0}

-- Part 1: Intersection condition
theorem intersection_condition (m : ℝ) : A ∩ B m = Set.Icc 2 4 → m = 5 := by sorry

-- Part 2: Subset condition
theorem subset_condition (m : ℝ) : A ⊆ (Set.univ \ B m) → m < -2 ∨ m > 7 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_subset_condition_l906_90626


namespace NUMINAMATH_CALUDE_remainder_101_37_mod_100_l906_90616

theorem remainder_101_37_mod_100 : 101^37 ≡ 1 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_remainder_101_37_mod_100_l906_90616


namespace NUMINAMATH_CALUDE_strip_to_upper_half_plane_l906_90615

-- Define the complex exponential function
noncomputable def complex_exp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- Define the mapping function
noncomputable def w (z : ℂ) (h : ℝ) : ℂ := complex_exp ((Real.pi * z) / h)

-- State the theorem
theorem strip_to_upper_half_plane (z : ℂ) (h : ℝ) (h_pos : h > 0) (z_in_strip : 0 < z.im ∧ z.im < h) :
  (w z h).im > 0 := by sorry

end NUMINAMATH_CALUDE_strip_to_upper_half_plane_l906_90615


namespace NUMINAMATH_CALUDE_A_time_to_complete_l906_90680

-- Define the rates of work for A, B, and C
variable (rA rB rC : ℝ)

-- Define the conditions
axiom AB_time : rA + rB = 1 / 2
axiom BC_time : rB + rC = 1 / 4
axiom AC_time : rA + rC = 5 / 12

-- Define the theorem
theorem A_time_to_complete : 1 / rA = 3 := by
  sorry

end NUMINAMATH_CALUDE_A_time_to_complete_l906_90680


namespace NUMINAMATH_CALUDE_largest_possible_a_l906_90618

theorem largest_possible_a (a b c d : ℕ+) 
  (h1 : a < 3 * b)
  (h2 : b < 2 * c + 1)
  (h3 : c < 5 * d - 2)
  (h4 : d ≤ 50)
  (h5 : ∃ k : ℕ, d = 5 * k) :
  a ≤ 1481 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 1481 ∧
    a' < 3 * b' ∧
    b' < 2 * c' + 1 ∧
    c' < 5 * d' - 2 ∧
    d' ≤ 50 ∧
    ∃ k : ℕ, d' = 5 * k :=
by sorry

end NUMINAMATH_CALUDE_largest_possible_a_l906_90618


namespace NUMINAMATH_CALUDE_quadratic_roots_transformation_l906_90601

theorem quadratic_roots_transformation (K : ℝ) (α β : ℝ) : 
  (3 * α^2 + 7 * α + K = 0) →
  (3 * β^2 + 7 * β + K = 0) →
  (∃ m : ℝ, (α^2 - α)^2 + p * (α^2 - α) + m = 0 ∧ (β^2 - β)^2 + p * (β^2 - β) + m = 0) →
  p = -70/9 + 2*K/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_transformation_l906_90601


namespace NUMINAMATH_CALUDE_mean_of_remaining_numbers_l906_90622

theorem mean_of_remaining_numbers : 
  let numbers : List ℕ := [1867, 1993, 2019, 2025, 2109, 2121]
  let total_sum : ℕ := numbers.sum
  let mean_of_four : ℕ := 2008
  let sum_of_four : ℕ := 4 * mean_of_four
  let sum_of_two : ℕ := total_sum - sum_of_four
  sum_of_two / 2 = 2051 := by sorry

end NUMINAMATH_CALUDE_mean_of_remaining_numbers_l906_90622


namespace NUMINAMATH_CALUDE_train_passengers_l906_90633

theorem train_passengers (initial_passengers : ℕ) (stops : ℕ) : 
  initial_passengers = 64 → stops = 4 → 
  (initial_passengers : ℚ) * ((2 : ℚ) / 3) ^ stops = 1024 / 81 := by
  sorry

end NUMINAMATH_CALUDE_train_passengers_l906_90633


namespace NUMINAMATH_CALUDE_rectangle_tiles_l906_90640

theorem rectangle_tiles (length width : ℕ) : 
  width = 2 * length →
  (length * length + width * width : ℚ).sqrt = 45 →
  length * width = 810 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_tiles_l906_90640


namespace NUMINAMATH_CALUDE_hyperbola_equation_l906_90623

/-- Given a hyperbola with equation x²/a² - y²/4 = 1 and an asymptote y = x/2,
    prove that the equation of the hyperbola is x²/16 - y²/4 = 1 -/
theorem hyperbola_equation (a : ℝ) :
  (∃ x y, x^2 / a^2 - y^2 / 4 = 1) →
  (∃ x, x / 2 = x / 2) →
  (∃ x y, x^2 / 16 - y^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l906_90623


namespace NUMINAMATH_CALUDE_roger_bike_distance_l906_90650

/-- Calculates the total distance Roger rode his bike over three sessions -/
theorem roger_bike_distance (morning_distance : ℝ) (evening_multiplier : ℝ) (km_per_mile : ℝ) : 
  morning_distance = 2 →
  evening_multiplier = 5 →
  km_per_mile = 1.6 →
  morning_distance + (evening_multiplier * morning_distance) + 
    (2 * morning_distance * km_per_mile / km_per_mile) = 16 := by
  sorry


end NUMINAMATH_CALUDE_roger_bike_distance_l906_90650


namespace NUMINAMATH_CALUDE_division_relation_l906_90653

theorem division_relation (h : 2994 / 14.5 = 171) : 29.94 / 1.75 = 17.1 := by
  sorry

end NUMINAMATH_CALUDE_division_relation_l906_90653


namespace NUMINAMATH_CALUDE_stem_and_leaf_update_l906_90691

/-- Represents a stem-and-leaf diagram --/
structure StemAndLeaf :=
  (stem : List ℕ)
  (leaf : List (List ℕ))

/-- The initial stem-and-leaf diagram --/
def initial_diagram : StemAndLeaf := {
  stem := [0, 1, 2, 3, 4],
  leaf := [[], [0, 0, 1, 2, 2, 3], [1, 5, 6], [0, 2, 4, 6], [1, 6]]
}

/-- Function to update ages in the diagram --/
def update_ages (d : StemAndLeaf) (years : ℕ) : StemAndLeaf :=
  sorry

/-- Theorem stating the time passed and the reconstruction of the new diagram --/
theorem stem_and_leaf_update :
  ∃ (years : ℕ) (new_diagram : StemAndLeaf),
    years = 6 ∧
    new_diagram = update_ages initial_diagram years ∧
    new_diagram.stem = [0, 1, 2, 3, 4] ∧
    new_diagram.leaf = [[],
                        [5, 5],
                        [1, 5, 6],
                        [0, 2, 4, 6],
                        [1, 6]] :=
  sorry

end NUMINAMATH_CALUDE_stem_and_leaf_update_l906_90691


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l906_90676

def p (m : ℝ) (x : ℝ) : ℝ := 4 * x^3 - 16 * x^2 + m * x - 20

theorem polynomial_divisibility (m : ℝ) :
  (∃ q : ℝ → ℝ, ∀ x, p m x = (x - 4) * q x) →
  (m = 5 ∧ ¬∃ r : ℝ → ℝ, ∀ x, p 5 x = (x - 5) * r x) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l906_90676


namespace NUMINAMATH_CALUDE_inequality_proof_l906_90665

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : 
  Real.sqrt (b^2 - a*c) > Real.sqrt 3 * a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l906_90665


namespace NUMINAMATH_CALUDE_num_true_propositions_l906_90608

-- Define the original proposition
def original_proposition (x y : ℝ) : Prop := x^2 > y^2 → x > y

-- Define the converse
def converse (x y : ℝ) : Prop := x > y → x^2 > y^2

-- Define the inverse
def inverse (x y : ℝ) : Prop := ¬(x^2 > y^2) → ¬(x > y)

-- Define the contrapositive
def contrapositive (x y : ℝ) : Prop := ¬(x > y) → ¬(x^2 > y^2)

-- Theorem statement
theorem num_true_propositions : 
  (∃ x y : ℝ, ¬(original_proposition x y)) ∧ 
  (∃ x y : ℝ, ¬(converse x y)) ∧ 
  (∃ x y : ℝ, ¬(inverse x y)) ∧ 
  (∃ x y : ℝ, ¬(contrapositive x y)) := by
  sorry

end NUMINAMATH_CALUDE_num_true_propositions_l906_90608


namespace NUMINAMATH_CALUDE_jellybean_box_capacity_l906_90607

/-- Represents a rectangular box with a certain capacity of jellybeans -/
structure JellyBean_Box where
  height : ℝ
  width : ℝ
  length : ℝ
  capacity : ℕ

/-- Calculates the volume of a JellyBean_Box -/
def box_volume (box : JellyBean_Box) : ℝ :=
  box.height * box.width * box.length

/-- Theorem stating the relationship between box sizes and jellybean capacities -/
theorem jellybean_box_capacity 
  (box_b box_c : JellyBean_Box)
  (h_capacity_b : box_b.capacity = 125)
  (h_height : box_c.height = 2 * box_b.height)
  (h_width : box_c.width = 2 * box_b.width)
  (h_length : box_c.length = 2 * box_b.length) :
  box_c.capacity = 1000 :=
by sorry


end NUMINAMATH_CALUDE_jellybean_box_capacity_l906_90607


namespace NUMINAMATH_CALUDE_smallest_n_without_quadratic_number_l906_90619

def isQuadraticNumber (x : ℝ) : Prop :=
  ∃ (a b c : ℤ), a ≠ 0 ∧ 
  (|a| ≤ 10 ∧ |a| ≥ 1) ∧ 
  (|b| ≤ 10 ∧ |b| ≥ 1) ∧ 
  (|c| ≤ 10 ∧ |c| ≥ 1) ∧ 
  a * x^2 + b * x + c = 0

def hasQuadraticNumber (l r : ℝ) : Prop :=
  ∃ x, l < x ∧ x < r ∧ isQuadraticNumber x

def noQuadraticNumber (n : ℕ) : Prop :=
  ¬(hasQuadraticNumber (n - 1/3) n) ∨ ¬(hasQuadraticNumber n (n + 1/3))

theorem smallest_n_without_quadratic_number :
  (∀ m : ℕ, m < 11 → ¬(noQuadraticNumber m)) ∧ noQuadraticNumber 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_without_quadratic_number_l906_90619


namespace NUMINAMATH_CALUDE_value_of_2a_plus_b_l906_90606

-- Define the functions f, g, and h
def f (a b : ℝ) (x : ℝ) : ℝ := a * x - b
def g (x : ℝ) : ℝ := -4 * x + 6
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem value_of_2a_plus_b (a b : ℝ) :
  (∀ x, h a b x = x - 9) →
  2 * a + b = 7 := by
sorry

end NUMINAMATH_CALUDE_value_of_2a_plus_b_l906_90606


namespace NUMINAMATH_CALUDE_shortest_path_equals_two_R_l906_90629

/-- A truncated cone with a specific angle between generatrix and larger base -/
structure TruncatedCone where
  R : ℝ  -- Radius of the larger base
  r : ℝ  -- Radius of the smaller base
  h : ℝ  -- Height of the truncated cone
  angle : ℝ  -- Angle between generatrix and larger base in radians

/-- The shortest path on the surface of a truncated cone -/
def shortestPath (cone : TruncatedCone) : ℝ := sorry

/-- Theorem stating that the shortest path is twice the radius of the larger base -/
theorem shortest_path_equals_two_R (cone : TruncatedCone) 
  (h₁ : cone.angle = π / 3)  -- 60 degrees in radians
  : shortestPath cone = 2 * cone.R := by
  sorry

end NUMINAMATH_CALUDE_shortest_path_equals_two_R_l906_90629


namespace NUMINAMATH_CALUDE_chemistry_students_l906_90641

def basketball_team : ℕ := 18
def math_students : ℕ := 10
def physics_students : ℕ := 6
def math_and_physics : ℕ := 3
def all_three : ℕ := 2

theorem chemistry_students : ℕ := by
  -- The number of students studying chemistry is 7
  have h : basketball_team = math_students + physics_students - math_and_physics + (basketball_team - (math_students + physics_students - math_and_physics)) := by sorry
  -- Proof goes here
  sorry

#check chemistry_students -- Should evaluate to 7

end NUMINAMATH_CALUDE_chemistry_students_l906_90641


namespace NUMINAMATH_CALUDE_second_train_length_l906_90669

-- Define constants
def train1_speed : Real := 60  -- km/hr
def train2_speed : Real := 40  -- km/hr
def crossing_time : Real := 11.159107271418288  -- seconds
def train1_length : Real := 140  -- meters

-- Define the theorem
theorem second_train_length :
  let relative_speed := (train1_speed + train2_speed) * (5/18)  -- Convert km/hr to m/s
  let total_distance := relative_speed * crossing_time
  let train2_length := total_distance - train1_length
  train2_length = 170 := by
  sorry

end NUMINAMATH_CALUDE_second_train_length_l906_90669


namespace NUMINAMATH_CALUDE_happy_snakes_not_purple_l906_90613

structure Snake where
  happy : Bool
  purple : Bool
  canAdd : Bool
  canSubtract : Bool

def TomSnakes : Set Snake := sorry

theorem happy_snakes_not_purple :
  ∀ s ∈ TomSnakes,
  (s.happy → s.canAdd) ∧
  (s.purple → ¬s.canSubtract) ∧
  (¬s.canSubtract → ¬s.canAdd) →
  (s.happy → ¬s.purple) := by
  sorry

#check happy_snakes_not_purple

end NUMINAMATH_CALUDE_happy_snakes_not_purple_l906_90613


namespace NUMINAMATH_CALUDE_remainder_theorem_l906_90679

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 75 * k - 2) :
  (n^2 + 2*n + 3) % 75 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l906_90679


namespace NUMINAMATH_CALUDE_cos_increasing_interval_l906_90645

theorem cos_increasing_interval (a : Real) : 
  (∀ x₁ x₂, -Real.pi ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ a → Real.cos x₁ < Real.cos x₂) → 
  a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_increasing_interval_l906_90645


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l906_90611

-- Define the function g
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (1 / (x + 3))
  else if x < -3 then Int.floor (1 / (x + 3))
  else 0  -- This value doesn't matter as g is undefined at x = -3

-- Theorem statement
theorem zero_not_in_range_of_g :
  ¬ ∃ (x : ℝ), g x = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l906_90611


namespace NUMINAMATH_CALUDE_election_votes_l906_90647

theorem election_votes (V : ℝ) 
  (h1 : V > 0) -- Ensure total votes is positive
  (h2 : ∃ (x : ℝ), x = 0.25 * V ∧ x + 4000 = V - x) : V = 8000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l906_90647


namespace NUMINAMATH_CALUDE_fish_catch_problem_l906_90609

theorem fish_catch_problem (total_fish : ℕ) 
  (first_fisherman_carp_ratio : ℚ) (second_fisherman_perch_ratio : ℚ) :
  total_fish = 70 ∧ 
  first_fisherman_carp_ratio = 5 / 9 ∧ 
  second_fisherman_perch_ratio = 7 / 17 →
  ∃ (first_catch second_catch : ℕ),
    first_catch + second_catch = total_fish ∧
    first_catch * first_fisherman_carp_ratio = 
      second_catch * second_fisherman_perch_ratio ∧
    first_catch = 36 ∧ 
    second_catch = 34 := by
  sorry

#check fish_catch_problem

end NUMINAMATH_CALUDE_fish_catch_problem_l906_90609


namespace NUMINAMATH_CALUDE_largest_non_expressible_l906_90672

def is_composite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_not_multiple_of_four (n : ℕ) : Prop :=
  ¬(∃ k, n = 4 * k)

def is_expressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ is_composite b ∧ is_not_multiple_of_four b ∧ n = 36 * a + b

theorem largest_non_expressible : 
  (∀ n > 147, is_expressible n) ∧ ¬(is_expressible 147) :=
sorry

end NUMINAMATH_CALUDE_largest_non_expressible_l906_90672


namespace NUMINAMATH_CALUDE_ratio_equality_l906_90646

theorem ratio_equality (x y z : ℝ) (h : x / 3 = y / 4 ∧ y / 4 = z / 5) :
  (x + y - z) / (2 * x - y + z) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l906_90646


namespace NUMINAMATH_CALUDE_cinnamon_swirl_sharing_l906_90664

theorem cinnamon_swirl_sharing (total_swirls : ℕ) (jane_pieces : ℕ) (people : ℕ) : 
  total_swirls = 12 →
  jane_pieces = 4 →
  total_swirls % jane_pieces = 0 →
  total_swirls / jane_pieces = people →
  people = 3 := by
  sorry

end NUMINAMATH_CALUDE_cinnamon_swirl_sharing_l906_90664


namespace NUMINAMATH_CALUDE_second_round_score_l906_90692

/-- Represents the points scored in a round of darts --/
structure DartScore :=
  (points : ℕ)

/-- Represents the scores for three rounds of darts --/
structure ThreeRoundScores :=
  (round1 : DartScore)
  (round2 : DartScore)
  (round3 : DartScore)

/-- Defines the relationship between scores in three rounds --/
def validScores (scores : ThreeRoundScores) : Prop :=
  scores.round2.points = 2 * scores.round1.points ∧
  scores.round3.points = (3 * scores.round1.points : ℕ)

/-- Theorem: Given the conditions, the score in the second round is 48 --/
theorem second_round_score (scores : ThreeRoundScores) 
  (h : validScores scores) : scores.round2.points = 48 := by
  sorry

#check second_round_score

end NUMINAMATH_CALUDE_second_round_score_l906_90692


namespace NUMINAMATH_CALUDE_complex_modulus_of_three_plus_i_squared_l906_90673

theorem complex_modulus_of_three_plus_i_squared :
  let z : ℂ := (3 + Complex.I) ^ 2
  ‖z‖ = 10 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_of_three_plus_i_squared_l906_90673


namespace NUMINAMATH_CALUDE_largest_n_binomial_equality_l906_90660

theorem largest_n_binomial_equality : ∃ n : ℕ, n = 6 ∧ 
  (∀ m : ℕ, (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m) → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_binomial_equality_l906_90660


namespace NUMINAMATH_CALUDE_bug_path_tiles_l906_90689

def floor_width : ℕ := 10
def floor_length : ℕ := 17

theorem bug_path_tiles : 
  floor_width + floor_length - Nat.gcd floor_width floor_length = 26 := by
  sorry

end NUMINAMATH_CALUDE_bug_path_tiles_l906_90689


namespace NUMINAMATH_CALUDE_derivative_y_wrt_x_at_zero_l906_90602

noncomputable def x (t : ℝ) : ℝ := Real.exp t * Real.cos t

noncomputable def y (t : ℝ) : ℝ := Real.exp t * Real.sin t

theorem derivative_y_wrt_x_at_zero :
  deriv (fun t => y t) 0 / deriv (fun t => x t) 0 = 1 := by sorry

end NUMINAMATH_CALUDE_derivative_y_wrt_x_at_zero_l906_90602


namespace NUMINAMATH_CALUDE_expected_scurries_eq_37_div_7_l906_90638

/-- Represents the number of people and horses -/
def n : ℕ := 8

/-- The probability that the i-th person scurries home -/
def scurry_prob (i : ℕ) : ℚ :=
  if i ≤ 1 then 0 else (i - 1 : ℚ) / i

/-- The expected number of people who scurry home -/
def expected_scurries : ℚ :=
  (Finset.range n).sum (λ i => scurry_prob (i + 1))

/-- Theorem stating that the expected number of people who scurry home is 37/7 -/
theorem expected_scurries_eq_37_div_7 :
  expected_scurries = 37 / 7 := by sorry

end NUMINAMATH_CALUDE_expected_scurries_eq_37_div_7_l906_90638


namespace NUMINAMATH_CALUDE_apple_weight_difference_l906_90677

/-- Given two baskets of apples with a total weight and the weight of one basket,
    prove the difference in weight between the baskets. -/
theorem apple_weight_difference (total_weight weight_a : ℕ) 
  (h1 : total_weight = 72)
  (h2 : weight_a = 42) :
  weight_a - (total_weight - weight_a) = 12 := by
  sorry

#check apple_weight_difference

end NUMINAMATH_CALUDE_apple_weight_difference_l906_90677


namespace NUMINAMATH_CALUDE_ice_cream_cost_l906_90667

/-- Given the following conditions:
    - Alok ordered 16 chapatis, 5 plates of rice, 7 plates of mixed vegetable, and 6 ice-cream cups
    - Cost of each chapati is Rs. 6
    - Cost of each plate of rice is Rs. 45
    - Cost of each plate of mixed vegetable is Rs. 70
    - Alok paid the cashier Rs. 985
    Prove that the cost of each ice-cream cup is Rs. 29 -/
theorem ice_cream_cost (chapati_count : ℕ) (rice_count : ℕ) (vegetable_count : ℕ) (ice_cream_count : ℕ)
                       (chapati_cost : ℕ) (rice_cost : ℕ) (vegetable_cost : ℕ) (total_paid : ℕ) :
  chapati_count = 16 →
  rice_count = 5 →
  vegetable_count = 7 →
  ice_cream_count = 6 →
  chapati_cost = 6 →
  rice_cost = 45 →
  vegetable_cost = 70 →
  total_paid = 985 →
  (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost)) / ice_cream_count = 29 := by
  sorry


end NUMINAMATH_CALUDE_ice_cream_cost_l906_90667


namespace NUMINAMATH_CALUDE_shipping_cost_formula_l906_90681

/-- The shipping cost function for a parcel and flat-rate envelope -/
def shippingCost (P : ℝ) : ℝ :=
  let firstPoundFee : ℝ := 12
  let additionalPoundFee : ℝ := 5
  let flatRateEnvelopeFee : ℝ := 20
  firstPoundFee + additionalPoundFee * (P - 1) + flatRateEnvelopeFee

theorem shipping_cost_formula (P : ℝ) :
  shippingCost P = 5 * P + 27 := by
  sorry

end NUMINAMATH_CALUDE_shipping_cost_formula_l906_90681


namespace NUMINAMATH_CALUDE_linear_equations_l906_90624

-- Define what a linear equation is
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

-- Define the equations
def eq1 : ℝ → ℝ := λ _ => 12
def eq2 : ℝ → ℝ := λ x => 5 * x + 3
def eq3 : ℝ → ℝ → ℝ := λ x y => 2 * x + 3 * y
def eq4 : ℝ → ℝ := λ a => 2 * a - 1
def eq5 : ℝ → ℝ := λ x => 2 * x^2 + x

-- Theorem statement
theorem linear_equations :
  (¬ is_linear_equation eq1) ∧
  (is_linear_equation eq2) ∧
  (¬ is_linear_equation (λ x => eq3 x 0)) ∧
  (is_linear_equation eq4) ∧
  (¬ is_linear_equation eq5) :=
sorry

end NUMINAMATH_CALUDE_linear_equations_l906_90624


namespace NUMINAMATH_CALUDE_new_boarders_count_l906_90637

/-- Represents the number of boarders and day students at a school -/
structure SchoolPopulation where
  boarders : ℕ
  dayStudents : ℕ

/-- Represents the ratio of boarders to day students -/
structure Ratio where
  boarders : ℕ
  dayStudents : ℕ

def initialPopulation : SchoolPopulation :=
  { boarders := 150, dayStudents := 360 }

def initialRatio : Ratio :=
  { boarders := 5, dayStudents := 12 }

def finalRatio : Ratio :=
  { boarders := 1, dayStudents := 2 }

/-- The theorem to be proved -/
theorem new_boarders_count (newBoarders : ℕ) :
  (initialPopulation.boarders + newBoarders) / initialPopulation.dayStudents = 
    finalRatio.boarders / finalRatio.dayStudents ∧
  initialPopulation.boarders / initialPopulation.dayStudents = 
    initialRatio.boarders / initialRatio.dayStudents →
  newBoarders = 30 := by
  sorry

end NUMINAMATH_CALUDE_new_boarders_count_l906_90637


namespace NUMINAMATH_CALUDE_right_triangle_circumscribed_circle_radius_l906_90663

theorem right_triangle_circumscribed_circle_radius 
  (a b c R : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_a : a = 5) 
  (h_b : b = 12) 
  (h_R : R = c / 2) : R = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_circumscribed_circle_radius_l906_90663


namespace NUMINAMATH_CALUDE_class_average_weight_l906_90658

theorem class_average_weight (students_A : ℕ) (students_B : ℕ) (avg_weight_A : ℝ) (avg_weight_B : ℝ)
  (h1 : students_A = 24)
  (h2 : students_B = 16)
  (h3 : avg_weight_A = 40)
  (h4 : avg_weight_B = 35) :
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B : ℝ) = 38 := by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l906_90658


namespace NUMINAMATH_CALUDE_special_sequence_first_term_l906_90699

/-- An arithmetic sequence with common difference 2 where a₁, a₂, and a₄ form a geometric sequence -/
def special_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n + 2) ∧  -- arithmetic sequence with difference 2
  ∃ r, a 2 = a 1 * r ∧ a 4 = a 2 * r  -- a₁, a₂, a₄ form geometric sequence

/-- The first term of the special sequence is 2 -/
theorem special_sequence_first_term (a : ℕ → ℝ) (h : special_sequence a) : a 1 = 2 :=
sorry

end NUMINAMATH_CALUDE_special_sequence_first_term_l906_90699


namespace NUMINAMATH_CALUDE_heat_engine_efficiency_l906_90636

theorem heat_engine_efficiency
  (η₀ η₁ η₂ α : ℝ)
  (h1 : η₁ < η₀)
  (h2 : η₂ < η₀)
  (h3 : η₀ < 1)
  (h4 : η₁ < 1)
  (h5 : η₂ = (η₀ - η₁) / (1 - η₁))
  (h6 : η₁ = (1 - 0.01 * α) * η₀) :
  η₂ = α / (100 - (100 - α) * η₀) := by
sorry

end NUMINAMATH_CALUDE_heat_engine_efficiency_l906_90636


namespace NUMINAMATH_CALUDE_motel_rental_rate_l906_90682

theorem motel_rental_rate (lower_rate higher_rate total_rent : ℚ) 
  (h1 : lower_rate = 40)
  (h2 : total_rent = 400)
  (h3 : total_rent / 2 = total_rent - 10 * (higher_rate - lower_rate)) :
  higher_rate = 60 := by
  sorry

end NUMINAMATH_CALUDE_motel_rental_rate_l906_90682


namespace NUMINAMATH_CALUDE_probability_different_colors_is_83_128_l906_90610

/-- Represents the number of chips of each color in the bag -/
structure ChipCounts where
  blue : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the probability of drawing two chips of different colors -/
def probabilityDifferentColors (counts : ChipCounts) : ℚ :=
  let total := counts.blue + counts.yellow + counts.red
  let pBlue := counts.blue / total
  let pYellow := counts.yellow / total
  let pRed := counts.red / total
  pBlue * (pYellow + pRed) + pYellow * (pBlue + pRed) + pRed * (pBlue + pYellow)

/-- The main theorem stating the probability of drawing two chips of different colors -/
theorem probability_different_colors_is_83_128 :
  probabilityDifferentColors ⟨7, 5, 4⟩ = 83 / 128 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_colors_is_83_128_l906_90610


namespace NUMINAMATH_CALUDE_money_redistribution_l906_90649

def initial_amount (i : Nat) : Nat :=
  2^(i-1) - 1

def final_amount (n : Nat) : Nat :=
  8 * (List.sum (List.map initial_amount (List.range n)))

theorem money_redistribution (n : Nat) :
  n = 9 → final_amount n = 512 := by sorry

end NUMINAMATH_CALUDE_money_redistribution_l906_90649


namespace NUMINAMATH_CALUDE_chloe_age_sum_of_digits_l906_90652

/-- Represents a person's age -/
structure Age :=
  (value : ℕ)

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- Represents the family's ages and their properties -/
structure FamilyAges :=
  (joey : Age)
  (chloe : Age)
  (max : Age)
  (joey_chloe_diff : joey.value = chloe.value + 2)
  (max_age : max.value = 2)
  (joey_multiple_of_max : ∃ k : ℕ, joey.value = k * max.value)
  (future_multiples : ∃ n₁ n₂ n₃ n₄ n₅ : ℕ, 
    (joey.value + n₁) % (max.value + n₁) = 0 ∧
    (joey.value + n₂) % (max.value + n₂) = 0 ∧
    (joey.value + n₃) % (max.value + n₃) = 0 ∧
    (joey.value + n₄) % (max.value + n₄) = 0 ∧
    (joey.value + n₅) % (max.value + n₅) = 0)

theorem chloe_age_sum_of_digits (family : FamilyAges) :
  ∃ n : ℕ, n > 0 ∧ 
    (family.chloe.value + n) % (family.max.value + n) = 0 ∧
    sumOfDigits (family.chloe.value + n) = 10 :=
  sorry

end NUMINAMATH_CALUDE_chloe_age_sum_of_digits_l906_90652


namespace NUMINAMATH_CALUDE_equation_solution_l906_90697

theorem equation_solution : ∃ x : ℚ, (3 / 7 + 7 / x = 10 / x + 1 / 10) ∧ x = 210 / 23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l906_90697


namespace NUMINAMATH_CALUDE_correct_journey_equation_l906_90666

/-- Represents the journey of a ship between two ports -/
def ship_journey (distance : ℝ) (flow_speed : ℝ) (ship_speed : ℝ) : Prop :=
  distance / (ship_speed + flow_speed) + distance / (ship_speed - flow_speed) = 8

/-- Theorem stating that the given equation correctly represents the ship's journey -/
theorem correct_journey_equation :
  ∀ x : ℝ, x > 4 → ship_journey 50 4 x :=
by
  sorry

end NUMINAMATH_CALUDE_correct_journey_equation_l906_90666


namespace NUMINAMATH_CALUDE_total_scoops_needed_l906_90661

def flour_cups : ℚ := 3
def sugar_cups : ℚ := 2
def scoop_size : ℚ := 1/3

theorem total_scoops_needed : 
  (flour_cups / scoop_size + sugar_cups / scoop_size : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_scoops_needed_l906_90661


namespace NUMINAMATH_CALUDE_angle_equality_l906_90600

/-- Given a straight line split into two angles and a triangle with specific properties,
    prove that one of the angles equals 60 degrees. -/
theorem angle_equality (angle1 angle2 angle3 angle4 : ℝ) : 
  angle1 + angle2 = 180 →  -- Straight line condition
  angle1 + angle3 + 60 = 180 →  -- Triangle angle sum
  angle3 = angle4 →  -- Given equality
  angle4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l906_90600


namespace NUMINAMATH_CALUDE_equivalent_representations_l906_90644

theorem equivalent_representations : 
  (16 : ℚ) / 20 = 24 / 30 ∧ 
  (16 : ℚ) / 20 = 80 / 100 ∧ 
  (16 : ℚ) / 20 = 4 / 5 ∧ 
  (16 : ℚ) / 20 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_equivalent_representations_l906_90644


namespace NUMINAMATH_CALUDE_pet_store_cats_count_l906_90657

theorem pet_store_cats_count (siamese : Float) (house : Float) (added : Float) :
  siamese = 13.0 → house = 5.0 → added = 10.0 →
  siamese + house + added = 28.0 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cats_count_l906_90657


namespace NUMINAMATH_CALUDE_four_weighings_sufficient_three_weighings_insufficient_l906_90695

/-- Represents the result of a weighing: lighter, equal, or heavier -/
inductive WeighingResult
  | Lighter
  | Equal
  | Heavier

/-- Represents a sequence of weighing results -/
def WeighingSequence := List WeighingResult

/-- The number of cans in the problem -/
def numCans : Nat := 80

/-- A function that simulates a weighing, returning a WeighingResult -/
def weighing (a b : Nat) : WeighingResult :=
  sorry

theorem four_weighings_sufficient :
  ∃ (f : Fin numCans → WeighingSequence),
    (∀ (i j : Fin numCans), i ≠ j → f i ≠ f j) ∧
    (∀ (s : WeighingSequence), s.length = 4) :=
  sorry

theorem three_weighings_insufficient :
  ¬∃ (f : Fin numCans → WeighingSequence),
    (∀ (i j : Fin numCans), i ≠ j → f i ≠ f j) ∧
    (∀ (s : WeighingSequence), s.length = 3) :=
  sorry

end NUMINAMATH_CALUDE_four_weighings_sufficient_three_weighings_insufficient_l906_90695


namespace NUMINAMATH_CALUDE_faster_car_distance_l906_90614

/-- Two cars driving towards each other, with one twice as fast as the other and initial distance of 4 miles -/
structure TwoCars where
  slow_speed : ℝ
  fast_speed : ℝ
  initial_distance : ℝ
  slow_distance : ℝ
  fast_distance : ℝ
  meeting_condition : slow_distance + fast_distance = initial_distance
  speed_relation : fast_speed = 2 * slow_speed
  distance_relation : fast_distance = 2 * slow_distance

/-- The theorem stating that the faster car travels 8/3 miles when they meet -/
theorem faster_car_distance (cars : TwoCars) (h : cars.initial_distance = 4) :
  cars.fast_distance = 8/3 := by
  sorry

#check faster_car_distance

end NUMINAMATH_CALUDE_faster_car_distance_l906_90614


namespace NUMINAMATH_CALUDE_g_value_at_one_l906_90693

def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (g (x - y)) = g x * g y - g x + g y - 2 * x * y

theorem g_value_at_one (g : ℝ → ℝ) (h : g_property g) : g 1 = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_one_l906_90693


namespace NUMINAMATH_CALUDE_childrens_tickets_sold_l906_90684

theorem childrens_tickets_sold 
  (adult_price : ℚ) 
  (child_price : ℚ) 
  (total_tickets : ℕ) 
  (total_revenue : ℚ) 
  (h1 : adult_price = 6)
  (h2 : child_price = 9/2)
  (h3 : total_tickets = 400)
  (h4 : total_revenue = 2100) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_revenue ∧
    child_tickets = 200 := by
  sorry

end NUMINAMATH_CALUDE_childrens_tickets_sold_l906_90684


namespace NUMINAMATH_CALUDE_book_sales_ratio_l906_90642

theorem book_sales_ratio : 
  ∀ (wednesday thursday friday : ℕ),
  wednesday = 15 →
  thursday = 3 * wednesday →
  wednesday + thursday + friday = 69 →
  friday * 5 = thursday :=
λ wednesday thursday friday hw ht htot =>
  sorry

end NUMINAMATH_CALUDE_book_sales_ratio_l906_90642


namespace NUMINAMATH_CALUDE_green_packs_count_l906_90685

-- Define the number of balls per pack
def balls_per_pack : ℕ := 10

-- Define the number of packs for red and yellow balls
def red_packs : ℕ := 4
def yellow_packs : ℕ := 8

-- Define the total number of balls
def total_balls : ℕ := 160

-- Define the number of packs of green balls
def green_packs : ℕ := (total_balls - (red_packs + yellow_packs) * balls_per_pack) / balls_per_pack

-- Theorem statement
theorem green_packs_count : green_packs = 4 := by
  sorry

end NUMINAMATH_CALUDE_green_packs_count_l906_90685


namespace NUMINAMATH_CALUDE_octagon_area_l906_90605

/-- The area of a regular octagon formed by cutting corners from a square --/
theorem octagon_area (m : ℝ) : 
  let square_side : ℝ := 2 * m
  let octagon_area : ℝ := 4 * (Real.sqrt 2 - 1) * m^2
  octagon_area = 
    square_side^2 - 4 * (1/2 * (m * (2 - Real.sqrt 2))^2) :=
by sorry

end NUMINAMATH_CALUDE_octagon_area_l906_90605


namespace NUMINAMATH_CALUDE_water_ratio_horse_to_pig_l906_90688

/-- Proves that the ratio of water needed by a horse to a pig is 2:1 given the specified conditions -/
theorem water_ratio_horse_to_pig :
  let num_pigs : ℕ := 8
  let num_horses : ℕ := 10
  let water_per_pig : ℕ := 3
  let water_for_chickens : ℕ := 30
  let total_water : ℕ := 114
  let water_for_horses : ℕ := total_water - (num_pigs * water_per_pig) - water_for_chickens
  let water_per_horse : ℚ := water_for_horses / num_horses
  water_per_horse / water_per_pig = 2 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_water_ratio_horse_to_pig_l906_90688


namespace NUMINAMATH_CALUDE_unique_perfect_square_sum_l906_90643

theorem unique_perfect_square_sum (p : Nat) (hp : p.Prime ∧ p > 2) :
  ∃! n : Nat, n > 0 ∧ ∃ k : Nat, n^2 + n*p = k^2 :=
by
  use ((p - 1)^2) / 4
  sorry

end NUMINAMATH_CALUDE_unique_perfect_square_sum_l906_90643


namespace NUMINAMATH_CALUDE_no_intersection_in_S_l906_90612

-- Define the set S inductively
inductive S : (Real → Real) → Prop
  | base : S (fun x ↦ x)
  | sub {f} : S f → S (fun x ↦ x - f x)
  | add {f} : S f → S (fun x ↦ x + (1 - x) * f x)

-- Define the theorem
theorem no_intersection_in_S :
  ∀ (f g : Real → Real), S f → S g → f ≠ g →
  ∀ x, 0 < x → x < 1 → f x ≠ g x :=
by sorry

end NUMINAMATH_CALUDE_no_intersection_in_S_l906_90612


namespace NUMINAMATH_CALUDE_circle_parallel_lines_distance_l906_90662

-- Define the circle
variable (r : ℝ) -- radius of the circle

-- Define the chords
def chord1 : ℝ := 45
def chord2 : ℝ := 49
def chord3 : ℝ := 49
def chord4 : ℝ := 45

-- Define the distance between adjacent parallel lines
def d : ℝ := 2.8

-- State the theorem
theorem circle_parallel_lines_distance :
  ∃ (r : ℝ), 
    r > 0 ∧
    chord1 = 45 ∧
    chord2 = 49 ∧
    chord3 = 49 ∧
    chord4 = 45 ∧
    d = 2.8 ∧
    r^2 = 506.25 + (1/4) * d^2 ∧
    r^2 = 600.25 + (49/4) * d^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_parallel_lines_distance_l906_90662


namespace NUMINAMATH_CALUDE_sphere_polyhedra_radii_ratio_l906_90604

/-- The ratio of radii for a sequence of spheres inscribed in and circumscribed around
    regular polyhedra (octahedron, icosahedron, dodecahedron, tetrahedron, hexahedron) -/
theorem sphere_polyhedra_radii_ratio :
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ),
    r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧ r₄ > 0 ∧ r₅ > 0 ∧ r₆ > 0 ∧
    (r₂ / r₁ = Real.sqrt (9 + 4 * Real.sqrt 5)) ∧
    (r₃ / r₂ = Real.sqrt (27 + 12 * Real.sqrt 5)) ∧
    (r₄ / r₃ = 3 * Real.sqrt (5 + 2 * Real.sqrt 5)) ∧
    (r₅ / r₄ = 3 * Real.sqrt 15) ∧
    (r₆ / r₅ = 3 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_sphere_polyhedra_radii_ratio_l906_90604


namespace NUMINAMATH_CALUDE_smallest_of_four_consecutive_even_numbers_l906_90694

theorem smallest_of_four_consecutive_even_numbers (x : ℤ) : 
  (∃ y z w : ℤ, y = x + 2 ∧ z = x + 4 ∧ w = x + 6 ∧ 
   x % 2 = 0 ∧ x + y + z + w = 140) → x = 32 := by
sorry

end NUMINAMATH_CALUDE_smallest_of_four_consecutive_even_numbers_l906_90694


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l906_90621

theorem product_mod_seventeen : (2024 * 2025 * 2026 * 2027 * 2028) % 17 = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l906_90621


namespace NUMINAMATH_CALUDE_smallest_interesting_number_l906_90603

/-- A natural number is interesting if 2n is a perfect square and 15n is a perfect cube. -/
def IsInteresting (n : ℕ) : Prop :=
  ∃ k m : ℕ, 2 * n = k^2 ∧ 15 * n = m^3

/-- 1800 is the smallest interesting natural number. -/
theorem smallest_interesting_number : 
  IsInteresting 1800 ∧ ∀ n < 1800, ¬IsInteresting n := by
  sorry

#check smallest_interesting_number

end NUMINAMATH_CALUDE_smallest_interesting_number_l906_90603


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l906_90690

theorem inverse_variation_problem (a b : ℝ) (k : ℝ) (h1 : a * b^3 = k) (h2 : 8 * 2^3 = k) :
  a * 4^3 = k → a = 1 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l906_90690


namespace NUMINAMATH_CALUDE_gcd_90_270_l906_90698

theorem gcd_90_270 : Nat.gcd 90 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_90_270_l906_90698


namespace NUMINAMATH_CALUDE_paint_used_after_four_weeks_l906_90683

/-- Calculates the amount of paint used over 4 weeks given an initial amount and usage fractions --/
def paint_used (initial : ℝ) (w1_frac w2_frac w3_frac w4_frac : ℝ) : ℝ :=
  let w1_used := w1_frac * initial
  let w1_remaining := initial - w1_used
  let w2_used := w2_frac * w1_remaining
  let w2_remaining := w1_remaining - w2_used
  let w3_used := w3_frac * w2_remaining
  let w3_remaining := w2_remaining - w3_used
  let w4_used := w4_frac * w3_remaining
  w1_used + w2_used + w3_used + w4_used

/-- The theorem stating the amount of paint used after 4 weeks --/
theorem paint_used_after_four_weeks :
  let initial_paint := 360
  let week1_fraction := 1/4
  let week2_fraction := 1/3
  let week3_fraction := 2/5
  let week4_fraction := 3/7
  abs (paint_used initial_paint week1_fraction week2_fraction week3_fraction week4_fraction - 298.2857) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_paint_used_after_four_weeks_l906_90683


namespace NUMINAMATH_CALUDE_square_difference_division_equals_318_l906_90620

theorem square_difference_division_equals_318 : (165^2 - 153^2) / 12 = 318 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_division_equals_318_l906_90620


namespace NUMINAMATH_CALUDE_sin_2x_derivative_at_pi_6_l906_90696

theorem sin_2x_derivative_at_pi_6 (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin (2 * x)) :
  deriv f (π / 6) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_derivative_at_pi_6_l906_90696


namespace NUMINAMATH_CALUDE_halfway_point_between_fractions_l906_90687

theorem halfway_point_between_fractions :
  (1 / 12 + 1 / 14) / 2 = 13 / 168 := by
  sorry

end NUMINAMATH_CALUDE_halfway_point_between_fractions_l906_90687


namespace NUMINAMATH_CALUDE_no_double_application_function_l906_90674

theorem no_double_application_function : ¬∃ (f : ℕ → ℕ), ∀ n, f (f n) = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_function_l906_90674


namespace NUMINAMATH_CALUDE_percentage_of_male_students_l906_90686

theorem percentage_of_male_students (M F : ℝ) : 
  M + F = 100 →
  0.60 * M + 0.70 * F = 66 →
  M = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_male_students_l906_90686


namespace NUMINAMATH_CALUDE_girls_not_adjacent_arrangements_l906_90671

def num_boys : ℕ := 3
def num_girls : ℕ := 2

theorem girls_not_adjacent_arrangements :
  (num_boys.factorial * (num_boys + 1).choose num_girls) = 72 :=
by sorry

end NUMINAMATH_CALUDE_girls_not_adjacent_arrangements_l906_90671


namespace NUMINAMATH_CALUDE_somu_father_age_ratio_l906_90654

/-- Proves the ratio of Somu's age to his father's age -/
theorem somu_father_age_ratio :
  ∀ (S F : ℕ),
  S = 12 →
  S - 6 = (F - 6) / 5 →
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ S * b = F * a ∧ a = 1 ∧ b = 3 :=
by sorry

end NUMINAMATH_CALUDE_somu_father_age_ratio_l906_90654


namespace NUMINAMATH_CALUDE_max_intersection_points_l906_90670

/-- The number of points on the positive x-axis -/
def num_x_points : ℕ := 12

/-- The number of points on the positive y-axis -/
def num_y_points : ℕ := 6

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections : ℕ := 990

/-- Theorem stating the maximum number of intersection points -/
theorem max_intersection_points :
  (num_x_points.choose 2) * (num_y_points.choose 2) = max_intersections := by
  sorry

end NUMINAMATH_CALUDE_max_intersection_points_l906_90670


namespace NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l906_90656

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_15th_term 
  (a : ℕ → ℕ) 
  (h_arith : is_arithmetic_sequence a)
  (h_first : a 1 = 3)
  (h_second : a 2 = 12)
  (h_third : a 3 = 21) :
  a 15 = 129 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l906_90656


namespace NUMINAMATH_CALUDE_fraction_simplification_l906_90639

theorem fraction_simplification (m n : ℚ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (24 * m^3 * n^4) / (32 * m^4 * n^2) = (3 * n^2) / (4 * m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l906_90639


namespace NUMINAMATH_CALUDE_vacation_pictures_remaining_l906_90678

-- Define the number of pictures taken at each location
def zoo_pictures : ℕ := 49
def museum_pictures : ℕ := 8

-- Define the number of deleted pictures
def deleted_pictures : ℕ := 38

-- Theorem to prove
theorem vacation_pictures_remaining :
  zoo_pictures + museum_pictures - deleted_pictures = 19 := by
  sorry

end NUMINAMATH_CALUDE_vacation_pictures_remaining_l906_90678


namespace NUMINAMATH_CALUDE_jons_laundry_capacity_l906_90648

/-- Given information about Jon's laundry and machine capacity -/
structure LaundryInfo where
  shirts_per_pound : ℕ  -- Number of shirts that weigh 1 pound
  pants_per_pound : ℕ   -- Number of pairs of pants that weigh 1 pound
  total_shirts : ℕ      -- Total number of shirts to wash
  total_pants : ℕ       -- Total number of pants to wash
  loads : ℕ             -- Number of loads Jon has to do

/-- Calculate the machine capacity given laundry information -/
def machine_capacity (info : LaundryInfo) : ℚ :=
  let shirt_weight := info.total_shirts / info.shirts_per_pound
  let pants_weight := info.total_pants / info.pants_per_pound
  let total_weight := shirt_weight + pants_weight
  total_weight / info.loads

/-- Theorem stating Jon's laundry machine capacity -/
theorem jons_laundry_capacity :
  let info : LaundryInfo := {
    shirts_per_pound := 4,
    pants_per_pound := 2,
    total_shirts := 20,
    total_pants := 20,
    loads := 3
  }
  machine_capacity info = 5 := by sorry

end NUMINAMATH_CALUDE_jons_laundry_capacity_l906_90648


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l906_90659

/-- Given a circle C with equation x^2 + 6x + 36 = -y^2 - 8y + 45,
    prove that its center coordinates (a, b) and radius r satisfy a + b + r = -7 + √34 -/
theorem circle_center_radius_sum (x y a b r : ℝ) : 
  (∀ x y, x^2 + 6*x + 36 = -y^2 - 8*y + 45) →
  (∀ x y, (x + 3)^2 + (y + 4)^2 = 34) →
  (a = -3 ∧ b = -4) →
  r = Real.sqrt 34 →
  a + b + r = -7 + Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l906_90659


namespace NUMINAMATH_CALUDE_min_score_for_higher_average_l906_90628

/-- Represents the scores of a student in four tests -/
structure Scores :=
  (test1 : ℕ) (test2 : ℕ) (test3 : ℕ) (test4 : ℕ)

/-- A-Long's scores -/
def aLong : Scores :=
  { test1 := 81, test2 := 81, test3 := 81, test4 := 81 }

/-- A-Hai's scores -/
def aHai : Scores :=
  { test1 := aLong.test1 + 1,
    test2 := aLong.test2 + 2,
    test3 := aLong.test3 + 3,
    test4 := 99 }

/-- The average score of a student -/
def average (s : Scores) : ℚ :=
  (s.test1 + s.test2 + s.test3 + s.test4) / 4

theorem min_score_for_higher_average :
  average aHai ≥ average aLong + 4 :=
by sorry

end NUMINAMATH_CALUDE_min_score_for_higher_average_l906_90628


namespace NUMINAMATH_CALUDE_residue_sum_mod_19_l906_90627

theorem residue_sum_mod_19 : (8^1356 + 7^1200) % 19 = 10 := by
  sorry

end NUMINAMATH_CALUDE_residue_sum_mod_19_l906_90627


namespace NUMINAMATH_CALUDE_extreme_value_and_inequality_l906_90617

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (x + 1)

theorem extreme_value_and_inequality (a : ℝ) :
  (∃ x, f x = -1 ∧ ∀ y, f y ≥ f x) ∧
  (∀ x > 0, f x ≥ x + Real.log x + a + 1) ↔ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_and_inequality_l906_90617


namespace NUMINAMATH_CALUDE_possible_values_of_a_l906_90668

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x + 1 = 0}

theorem possible_values_of_a (a : ℝ) : A ∪ B a = A → a ∈ ({0, 1/3, -1/2} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l906_90668


namespace NUMINAMATH_CALUDE_largest_c_for_negative_three_in_range_l906_90675

-- Define the function f
def f (x c : ℝ) : ℝ := x^2 + 5*x + c

-- State the theorem
theorem largest_c_for_negative_three_in_range :
  (∃ (c : ℝ), ∀ (d : ℝ), 
    (∃ (x : ℝ), f x c = -3) → 
    (∃ (y : ℝ), f y d = -3) → 
    d ≤ c) ∧
  (∃ (x : ℝ), f x (13/4) = -3) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_negative_three_in_range_l906_90675


namespace NUMINAMATH_CALUDE_cost_of_type_B_books_cost_equals_formula_l906_90631

/-- Given a total of 100 books to be purchased, with x books of type A,
    prove that the cost of purchasing type B books is 8(100-x) yuan,
    where the unit price of type B book is 8. -/
theorem cost_of_type_B_books (x : ℕ) : ℕ :=
  let total_books : ℕ := 100
  let unit_price_B : ℕ := 8
  let num_type_B : ℕ := total_books - x
  unit_price_B * num_type_B

#check cost_of_type_B_books

/-- Proof that the cost of type B books is 8(100-x) -/
theorem cost_equals_formula (x : ℕ) :
  cost_of_type_B_books x = 8 * (100 - x) :=
by sorry

#check cost_equals_formula

end NUMINAMATH_CALUDE_cost_of_type_B_books_cost_equals_formula_l906_90631


namespace NUMINAMATH_CALUDE_equation_root_range_l906_90632

theorem equation_root_range (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x^2 - a*x + a^2 - 3 = 0) → 
  -Real.sqrt 3 < a ∧ a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_equation_root_range_l906_90632


namespace NUMINAMATH_CALUDE_ellipse_line_slope_l906_90625

/-- Given an ellipse and a line intersecting it, prove the slope of the line --/
theorem ellipse_line_slope (A B : ℝ × ℝ) : 
  (∀ (x y : ℝ), x^2 / 4 + y^2 / 2 = 1 → (x, y) = A ∨ (x, y) = B) →  -- A and B are on the ellipse
  (1, 1) = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →                    -- (1, 1) is the midpoint of AB
  (B.2 - A.2) / (B.1 - A.1) = -1/2 :=                              -- The slope of AB is -1/2
by sorry

end NUMINAMATH_CALUDE_ellipse_line_slope_l906_90625


namespace NUMINAMATH_CALUDE_prob_different_numbers_is_three_fourths_l906_90651

/-- Men's team has 3 players -/
def num_men : ℕ := 3

/-- Women's team has 4 players -/
def num_women : ℕ := 4

/-- Total number of possible outcomes when selecting one player from each team -/
def total_outcomes : ℕ := num_men * num_women

/-- Number of outcomes where players have the same number -/
def same_number_outcomes : ℕ := min num_men num_women

/-- Probability of selecting players with different numbers -/
def prob_different_numbers : ℚ := 1 - (same_number_outcomes : ℚ) / total_outcomes

theorem prob_different_numbers_is_three_fourths : 
  prob_different_numbers = 3/4 := by sorry

end NUMINAMATH_CALUDE_prob_different_numbers_is_three_fourths_l906_90651


namespace NUMINAMATH_CALUDE_tram_route_difference_l906_90655

/-- Represents a point on the circular tram line -/
inductive TramStop
| Circus
| Park
| Zoo

/-- Represents the distance between two points on the tram line -/
def distance (a b : TramStop) : ℝ := sorry

/-- The total circumference of the tram line -/
def circumference : ℝ := sorry

theorem tram_route_difference :
  let park_to_zoo := distance TramStop.Park TramStop.Zoo
  let park_to_circus_via_zoo := distance TramStop.Park TramStop.Zoo + distance TramStop.Zoo TramStop.Circus
  let park_to_circus_direct := distance TramStop.Park TramStop.Circus
  
  -- The distance from Park to Zoo via Circus is three times longer than the direct route
  distance TramStop.Park TramStop.Zoo + distance TramStop.Zoo TramStop.Circus + distance TramStop.Circus TramStop.Park = 3 * park_to_zoo →
  
  -- The distance from Circus to Zoo via Park is half as long as the direct route
  distance TramStop.Circus TramStop.Park + park_to_zoo = (1/2) * distance TramStop.Circus TramStop.Zoo →
  
  -- The difference between the longer and shorter routes from Park to Circus is 1/12 of the total circumference
  park_to_circus_via_zoo - park_to_circus_direct = (1/12) * circumference :=
by sorry

end NUMINAMATH_CALUDE_tram_route_difference_l906_90655


namespace NUMINAMATH_CALUDE_hash_sum_plus_five_l906_90635

-- Define the operation #
def hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

-- Theorem statement
theorem hash_sum_plus_five (a b : ℕ) : hash a b = 100 → (a + b) + 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_hash_sum_plus_five_l906_90635


namespace NUMINAMATH_CALUDE_complex_multiplication_l906_90630

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (3 + 2*i)*i = -2 + 3*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l906_90630
