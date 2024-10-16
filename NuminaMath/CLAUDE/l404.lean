import Mathlib

namespace NUMINAMATH_CALUDE_inverse_f_93_l404_40452

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 + 9

-- State the theorem
theorem inverse_f_93 : f⁻¹ 93 = (28 : ℝ)^(1/3) := by sorry

end NUMINAMATH_CALUDE_inverse_f_93_l404_40452


namespace NUMINAMATH_CALUDE_convention_handshakes_specific_l404_40442

/-- The number of handshakes in a convention with specified conditions -/
def convention_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_participants := num_companies * reps_per_company
  let handshakes_per_person := total_participants - reps_per_company
  (total_participants * handshakes_per_person) / 2

theorem convention_handshakes_specific : convention_handshakes 5 5 = 250 := by
  sorry

#eval convention_handshakes 5 5

end NUMINAMATH_CALUDE_convention_handshakes_specific_l404_40442


namespace NUMINAMATH_CALUDE_students_using_red_color_l404_40450

theorem students_using_red_color 
  (total_students : ℕ) 
  (green_users : ℕ) 
  (both_colors : ℕ) 
  (h1 : total_students = 70) 
  (h2 : green_users = 52) 
  (h3 : both_colors = 38) : 
  total_students + both_colors - green_users = 56 := by
  sorry

end NUMINAMATH_CALUDE_students_using_red_color_l404_40450


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l404_40487

def A : Set ℤ := {-2, -1, 0, 1, 2}

def B : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l404_40487


namespace NUMINAMATH_CALUDE_prob_three_spades_two_hearts_correct_l404_40457

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Fin 52)

/-- Represents the suit of a card -/
inductive Suit
| spades | hearts | diamonds | clubs

/-- Represents the rank of a card -/
def Rank := Fin 13

/-- The probability of drawing three spades followed by two hearts from a standard deck -/
def prob_three_spades_two_hearts : ℚ :=
  432 / 6497400

theorem prob_three_spades_two_hearts_correct (d : Deck) :
  prob_three_spades_two_hearts = 
    (13 * 12 * 11 * 13 * 12) / (52 * 51 * 50 * 49 * 48) :=
by sorry

end NUMINAMATH_CALUDE_prob_three_spades_two_hearts_correct_l404_40457


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l404_40478

theorem fraction_sum_equals_decimal : 2/10 + 4/100 + 6/1000 = 0.246 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l404_40478


namespace NUMINAMATH_CALUDE_special_triangle_sum_l404_40429

/-- A triangle with an incircle that evenly trisects a median -/
structure SpecialTriangle where
  -- The side length BC
  a : ℝ
  -- The area of the triangle
  area : ℝ
  -- k and p, where area = k√p
  k : ℕ
  p : ℕ
  -- Conditions
  side_length : a = 24
  area_form : area = k * Real.sqrt p
  p_not_square_divisible : ∀ (q : ℕ), Prime q → ¬(q^2 ∣ p)
  incircle_trisects_median : True  -- This condition is implicit in the structure

/-- The sum of k and p for the special triangle is 51 -/
theorem special_triangle_sum (t : SpecialTriangle) : t.k + t.p = 51 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_sum_l404_40429


namespace NUMINAMATH_CALUDE_x_equals_y_at_half_l404_40403

theorem x_equals_y_at_half (t : ℝ) : 
  let x := 1 - 4 * t
  let y := 2 * t - 2
  t = 0.5 → x = y := by sorry

end NUMINAMATH_CALUDE_x_equals_y_at_half_l404_40403


namespace NUMINAMATH_CALUDE_equation_real_root_implies_m_value_l404_40439

theorem equation_real_root_implies_m_value (x m : ℝ) (i : ℂ) :
  (∃ x : ℝ, x^2 + (1 - 2*i)*x + 3*m - i = 0) →
  m = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_equation_real_root_implies_m_value_l404_40439


namespace NUMINAMATH_CALUDE_square_perimeter_from_rectangle_perimeter_l404_40458

/-- Given a square divided into six congruent rectangles, if each rectangle has a perimeter of 30 inches, then the perimeter of the square is 360/7 inches. -/
theorem square_perimeter_from_rectangle_perimeter (s : ℝ) : 
  s > 0 → 
  (2 * s + 2 * (s / 6) = 30) → 
  (4 * s = 360 / 7) := by
  sorry

#check square_perimeter_from_rectangle_perimeter

end NUMINAMATH_CALUDE_square_perimeter_from_rectangle_perimeter_l404_40458


namespace NUMINAMATH_CALUDE_ellipse_equation_l404_40453

theorem ellipse_equation (a b c : ℝ) (h1 : 2 * a = 10) (h2 : c / a = 3 / 5) (h3 : b^2 = a^2 - c^2) :
  ∀ (x y : ℝ), (x^2 / 16 + y^2 / 25 = 1) ↔ (x^2 / b^2 + y^2 / a^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l404_40453


namespace NUMINAMATH_CALUDE_sum_u_v_l404_40406

theorem sum_u_v (u v : ℚ) (h1 : 5 * u - 6 * v = 19) (h2 : 3 * u + 5 * v = -1) : 
  u + v = 27 / 43 := by
  sorry

end NUMINAMATH_CALUDE_sum_u_v_l404_40406


namespace NUMINAMATH_CALUDE_trigonometric_identity_l404_40420

theorem trigonometric_identity (α φ : ℝ) : 
  Real.cos α ^ 2 + Real.cos φ ^ 2 + Real.cos (α + φ) ^ 2 - 
  2 * Real.cos α * Real.cos φ * Real.cos (α + φ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l404_40420


namespace NUMINAMATH_CALUDE_octadecagon_relation_l404_40451

/-- Given a regular octadecagon inscribed in a circle with side length a and radius r,
    prove that a³ + r³ = 3r²a. -/
theorem octadecagon_relation (a r : ℝ) (h : a > 0) (k : r > 0) :
  a^3 + r^3 = 3 * r^2 * a := by
  sorry

end NUMINAMATH_CALUDE_octadecagon_relation_l404_40451


namespace NUMINAMATH_CALUDE_milk_water_ratio_l404_40496

theorem milk_water_ratio (initial_volume : ℚ) (initial_milk_ratio : ℚ) (initial_water_ratio : ℚ) (added_water : ℚ) :
  initial_volume = 45 ∧
  initial_milk_ratio = 4 ∧
  initial_water_ratio = 1 ∧
  added_water = 3 →
  let total_parts := initial_milk_ratio + initial_water_ratio
  let initial_milk_volume := (initial_milk_ratio / total_parts) * initial_volume
  let initial_water_volume := (initial_water_ratio / total_parts) * initial_volume
  let new_water_volume := initial_water_volume + added_water
  let new_milk_ratio := initial_milk_volume
  let new_water_ratio := new_water_volume
  (new_milk_ratio : ℚ) / new_water_ratio = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_milk_water_ratio_l404_40496


namespace NUMINAMATH_CALUDE_trajectory_and_line_theorem_l404_40462

-- Define the circle P
def circle_P (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 36

-- Define point B
def point_B : ℝ × ℝ := (-2, 0)

-- Define the condition that P is on line segment AB
def P_on_AB (A P : ℝ × ℝ) : Prop := 
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * point_B.1, t * A.2 + (1 - t) * point_B.2)

-- Define the ratio condition
def ratio_condition (A P : ℝ × ℝ) : Prop :=
  (P.1 - point_B.1)^2 + (P.2 - point_B.2)^2 = 1/4 * ((A.1 - P.1)^2 + (A.2 - P.2)^2)

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop := x^2 + y^2 = 4 ∧ x ≠ -2

-- Define line l
def line_l (x y : ℝ) : Prop := 4*x + 3*y - 5 = 0 ∨ x = -1

-- Define the intersection condition
def intersection_condition (M N : ℝ × ℝ) : Prop :=
  trajectory_C M.1 M.2 ∧ trajectory_C N.1 N.2 ∧
  line_l M.1 M.2 ∧ line_l N.1 N.2 ∧
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 12

-- Main theorem
theorem trajectory_and_line_theorem 
  (A P : ℝ × ℝ) 
  (h1 : circle_P A.1 A.2)
  (h2 : P_on_AB A P)
  (h3 : ratio_condition A P)
  (h4 : ∃ M N : ℝ × ℝ, line_l (-1) 3 ∧ intersection_condition M N) :
  trajectory_C P.1 P.2 ∧ line_l (-1) 3 :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_line_theorem_l404_40462


namespace NUMINAMATH_CALUDE_max_draws_at_23_l404_40476

/-- Represents a lottery draw as a list of distinct integers -/
def LotteryDraw := List Nat

/-- The number of numbers drawn in each lottery draw -/
def drawSize : Nat := 5

/-- The maximum number that can be drawn -/
def maxNumber : Nat := 90

/-- Function to calculate the number of possible draws for a given second smallest number -/
def countDraws (secondSmallest : Nat) : Nat :=
  (secondSmallest - 1) * (maxNumber - secondSmallest) * (maxNumber - secondSmallest - 1) * (maxNumber - secondSmallest - 2)

theorem max_draws_at_23 :
  ∀ m, m ≠ 23 → countDraws 23 ≥ countDraws m :=
sorry

end NUMINAMATH_CALUDE_max_draws_at_23_l404_40476


namespace NUMINAMATH_CALUDE_largest_number_l404_40482

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

def a : Nat := base_to_decimal [5, 8] 9
def b : Nat := base_to_decimal [1, 0, 3] 5
def c : Nat := base_to_decimal [1, 0, 0, 1] 2

theorem largest_number : a > b ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l404_40482


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l404_40404

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * (a^3 + b^3 + c^3 + 3) ≥ 3 * (a + 1) * (b + 1) * (c + 1) :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * (a^3 + b^3 + c^3 + 3) = 3 * (a + 1) * (b + 1) * (c + 1) ↔ a = 1 ∧ b = 1 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l404_40404


namespace NUMINAMATH_CALUDE_smallest_b_value_l404_40492

/-- Given real numbers a and b satisfying certain conditions, 
    the smallest possible value of b is 2. -/
theorem smallest_b_value (a b : ℝ) 
  (h1 : 2 < a) 
  (h2 : a < b) 
  (h3 : ¬ (2 + a > b ∧ 2 + b > a ∧ a + b > 2)) 
  (h4 : ¬ (1/b + 1/a > 2 ∧ 1/b + 2 > 1/a ∧ 1/a + 2 > 1/b)) : 
  ∀ ε > 0, b ≥ 2 - ε := by
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l404_40492


namespace NUMINAMATH_CALUDE_guise_hot_dog_consumption_l404_40448

/-- Proves that given the conditions of Guise's hot dog consumption, the daily increase was 2 hot dogs. -/
theorem guise_hot_dog_consumption (monday_consumption : ℕ) (total_by_wednesday : ℕ) (daily_increase : ℕ) : 
  monday_consumption = 10 →
  total_by_wednesday = 36 →
  total_by_wednesday = monday_consumption + (monday_consumption + daily_increase) + (monday_consumption + 2 * daily_increase) →
  daily_increase = 2 := by
  sorry

end NUMINAMATH_CALUDE_guise_hot_dog_consumption_l404_40448


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l404_40431

theorem fraction_zero_implies_x_equals_one (x : ℝ) : 
  (x - 1) / (2 - x) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l404_40431


namespace NUMINAMATH_CALUDE_distance_PQ_l404_40416

def P : ℝ × ℝ := (-1, 2)
def Q : ℝ × ℝ := (3, 0)

theorem distance_PQ : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_PQ_l404_40416


namespace NUMINAMATH_CALUDE_rectangular_box_dimensions_l404_40428

theorem rectangular_box_dimensions :
  ∃! (a b c : ℕ),
    2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧
    Even a ∧ Even b ∧ Even c ∧
    2 * (a * b + a * c + b * c) = 4 * (a + b + c) ∧
    a = 2 ∧ b = 2 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_dimensions_l404_40428


namespace NUMINAMATH_CALUDE_min_sum_of_constrained_integers_l404_40421

theorem min_sum_of_constrained_integers (x y : ℕ) 
  (h1 : x - y < 1)
  (h2 : 2 * x - y > 2)
  (h3 : x < 5) :
  ∃ (a b : ℕ), a + b = 6 ∧ 
    (∀ (x' y' : ℕ), x' - y' < 1 → 2 * x' - y' > 2 → x' < 5 → x' + y' ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_constrained_integers_l404_40421


namespace NUMINAMATH_CALUDE_magic_square_sum_l404_40483

/-- Represents a 3x3 magic square -/
def MagicSquare := Fin 3 → Fin 3 → ℕ

/-- The magic sum of a magic square -/
def magicSum (s : MagicSquare) : ℕ := s 0 0 + s 0 1 + s 0 2

/-- Predicate to check if a square is magic -/
def isMagic (s : MagicSquare) : Prop :=
  let sum := magicSum s
  (∀ i, s i 0 + s i 1 + s i 2 = sum) ∧
  (∀ j, s 0 j + s 1 j + s 2 j = sum) ∧
  (s 0 0 + s 1 1 + s 2 2 = sum) ∧
  (s 0 2 + s 1 1 + s 2 0 = sum)

theorem magic_square_sum (s : MagicSquare) (x y : ℕ) 
  (h1 : s 0 0 = x)
  (h2 : s 0 1 = 6)
  (h3 : s 0 2 = 20)
  (h4 : s 1 0 = 22)
  (h5 : s 1 1 = y)
  (h6 : isMagic s) :
  x + y = 12 := by
  sorry


end NUMINAMATH_CALUDE_magic_square_sum_l404_40483


namespace NUMINAMATH_CALUDE_diagonal_difference_bound_l404_40469

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral :=
  (a b c d e f : ℝ)
  (cyclic : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0)
  (ptolemy : a * c + b * d = e * f)

-- State the theorem
theorem diagonal_difference_bound (q : CyclicQuadrilateral) :
  |q.e - q.f| ≤ |q.b - q.d| := by sorry

end NUMINAMATH_CALUDE_diagonal_difference_bound_l404_40469


namespace NUMINAMATH_CALUDE_yard_area_l404_40443

theorem yard_area (fence_length : ℝ) (unfenced_side : ℝ) (h1 : fence_length = 64) (h2 : unfenced_side = 40) :
  ∃ (width : ℝ), 
    unfenced_side + 2 * width = fence_length ∧ 
    unfenced_side * width = 480 :=
by sorry

end NUMINAMATH_CALUDE_yard_area_l404_40443


namespace NUMINAMATH_CALUDE_population_is_all_scores_l404_40438

/-- Represents a participant in the math test. -/
structure Participant where
  id : Nat
  score : ℝ

/-- Represents the entire set of participants in the math test. -/
def AllParticipants : Set Participant :=
  { p : Participant | p.id ≤ 1000 }

/-- Represents the sample of participants whose scores are analyzed. -/
def SampleParticipants : Set Participant :=
  { p : Participant | p.id ≤ 100 }

/-- The population in the context of this statistical analysis. -/
def Population : Set ℝ :=
  { score | ∃ p ∈ AllParticipants, p.score = score }

/-- Theorem stating that the population refers to the math scores of all 1000 participants. -/
theorem population_is_all_scores :
  Population = { score | ∃ p ∈ AllParticipants, p.score = score } :=
by sorry

end NUMINAMATH_CALUDE_population_is_all_scores_l404_40438


namespace NUMINAMATH_CALUDE_extra_bananas_l404_40405

theorem extra_bananas (total_children : ℕ) (original_bananas_per_child : ℕ) (absent_children : ℕ) : 
  total_children = 740 →
  original_bananas_per_child = 2 →
  absent_children = 370 →
  (total_children * original_bananas_per_child) / (total_children - absent_children) - original_bananas_per_child = 2 := by
sorry

end NUMINAMATH_CALUDE_extra_bananas_l404_40405


namespace NUMINAMATH_CALUDE_function_passes_through_point_l404_40445

/-- The function f(x) = 3x + 1 passes through the point (2,7) -/
theorem function_passes_through_point :
  let f : ℝ → ℝ := λ x ↦ 3 * x + 1
  f 2 = 7 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l404_40445


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l404_40434

theorem geometric_arithmetic_sequence_ratio (x y z : ℝ) 
  (h1 : (4*y)^2 = (3*x)*(5*z))  -- Geometric sequence condition
  (h2 : 2/y = 1/x + 1/z)        -- Arithmetic sequence condition
  : x/z + z/x = 34/15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l404_40434


namespace NUMINAMATH_CALUDE_x_plus_y_value_l404_40423

theorem x_plus_y_value (x y : ℝ) (hx : |x| = 2) (hy : |y| = 5) (hxy : x < y) :
  x + y = 7 ∨ x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l404_40423


namespace NUMINAMATH_CALUDE_meet_time_l404_40485

/-- Represents the scenario of Petya and Vasya's journey --/
structure Journey where
  distance : ℝ  -- Total distance between Petya and Vasya
  speed_dirt : ℝ  -- Speed on dirt road
  speed_paved : ℝ  -- Speed on paved road
  time_to_bridge : ℝ  -- Time for Petya to reach the bridge

/-- The conditions of the journey --/
def journey_conditions (j : Journey) : Prop :=
  j.speed_paved = 3 * j.speed_dirt ∧
  j.time_to_bridge = 1 ∧
  j.distance / 2 = j.speed_paved * j.time_to_bridge

/-- The theorem to be proved --/
theorem meet_time (j : Journey) (h : journey_conditions j) : 
  ∃ (t : ℝ), t = 2 ∧ t = j.time_to_bridge + (j.distance / 2 - j.speed_dirt * j.time_to_bridge) / (2 * j.speed_dirt) :=
sorry

end NUMINAMATH_CALUDE_meet_time_l404_40485


namespace NUMINAMATH_CALUDE_lastDigitOf2Power2023_l404_40472

-- Define the pattern of last digits for powers of 2
def lastDigitPattern : Fin 4 → Nat
  | 0 => 2
  | 1 => 4
  | 2 => 8
  | 3 => 6

-- Define the function to get the last digit of 2^n
def lastDigitOfPowerOf2 (n : Nat) : Nat :=
  lastDigitPattern ((n - 1) % 4)

-- Theorem statement
theorem lastDigitOf2Power2023 : lastDigitOfPowerOf2 2023 = 8 := by
  sorry

end NUMINAMATH_CALUDE_lastDigitOf2Power2023_l404_40472


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l404_40498

/-- Given a cone with base radius 6 and volume 30π, its lateral surface area is 39π -/
theorem cone_lateral_surface_area (r h l : ℝ) : 
  r = 6 → 
  (1 / 3) * π * r^2 * h = 30 * π → 
  l^2 = r^2 + h^2 → 
  π * r * l = 39 * π := by
sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l404_40498


namespace NUMINAMATH_CALUDE_banana_weights_l404_40484

/-- A scale with a constant displacement --/
structure DisplacedScale where
  displacement : ℝ

/-- Measurements of banana bunches on a displaced scale --/
structure BananaMeasurements where
  small_bunch : ℝ
  large_bunch : ℝ
  combined_bunches : ℝ

/-- The actual weights of the banana bunches --/
def actual_weights (s : DisplacedScale) (m : BananaMeasurements) : Prop :=
  ∃ (small large : ℝ),
    small = m.small_bunch - s.displacement ∧
    large = m.large_bunch - s.displacement ∧
    small + large = m.combined_bunches - s.displacement ∧
    small = 1 ∧ large = 2

/-- Theorem stating that given the measurements, the actual weights are 1 kg and 2 kg --/
theorem banana_weights (s : DisplacedScale) (m : BananaMeasurements) 
  (h1 : m.small_bunch = 1.5)
  (h2 : m.large_bunch = 2.5)
  (h3 : m.combined_bunches = 3.5) :
  actual_weights s m :=
by sorry

end NUMINAMATH_CALUDE_banana_weights_l404_40484


namespace NUMINAMATH_CALUDE_max_profit_l404_40435

-- Define the types of products
inductive Product
| A
| B

-- Define the profit function
def profit (x y : ℕ) : ℕ := 300 * x + 400 * y

-- Define the material constraints
def material_constraint (x y : ℕ) : Prop :=
  x + 2 * y ≤ 12 ∧ 2 * x + y ≤ 12

-- State the theorem
theorem max_profit :
  ∃ x y : ℕ,
    material_constraint x y ∧
    profit x y = 2800 ∧
    ∀ a b : ℕ, material_constraint a b → profit a b ≤ 2800 :=
sorry

end NUMINAMATH_CALUDE_max_profit_l404_40435


namespace NUMINAMATH_CALUDE_donalds_apples_l404_40477

theorem donalds_apples (marin_apples total_apples : ℕ) 
  (h1 : marin_apples = 9)
  (h2 : total_apples = 11) :
  total_apples - marin_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_donalds_apples_l404_40477


namespace NUMINAMATH_CALUDE_special_right_triangle_sides_l404_40479

/-- A right triangle with a special inscribed circle -/
structure SpecialRightTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The first leg of the triangle -/
  x : ℝ
  /-- The second leg of the triangle -/
  y : ℝ
  /-- The hypotenuse of the triangle -/
  z : ℝ
  /-- The area of the triangle is 2r^2/3 -/
  area_eq : x * y / 2 = 2 * r^2 / 3
  /-- The triangle is right-angled -/
  pythagoras : x^2 + y^2 = z^2
  /-- The circle touches one leg, the extension of the other leg, and the hypotenuse -/
  circle_property : z = 2*r + x - y

/-- The sides of a special right triangle are r, 4r/3, and 5r/3 -/
theorem special_right_triangle_sides (t : SpecialRightTriangle) : 
  t.x = t.r ∧ t.y = 4 * t.r / 3 ∧ t.z = 5 * t.r / 3 := by
  sorry

end NUMINAMATH_CALUDE_special_right_triangle_sides_l404_40479


namespace NUMINAMATH_CALUDE_total_tax_percentage_calculation_l404_40488

/-- Calculates the total tax percentage given spending percentages and tax rates -/
def totalTaxPercentage (clothingSpendPercentage : ℝ) (foodSpendPercentage : ℝ) 
  (electronicsSpendPercentage : ℝ) (otherSpendPercentage : ℝ)
  (clothingTaxRate : ℝ) (foodTaxRate : ℝ) (electronicsTaxRate : ℝ) (otherTaxRate : ℝ) : ℝ :=
  clothingSpendPercentage * clothingTaxRate + 
  foodSpendPercentage * foodTaxRate + 
  electronicsSpendPercentage * electronicsTaxRate + 
  otherSpendPercentage * otherTaxRate

theorem total_tax_percentage_calculation :
  totalTaxPercentage 0.585 0.12 0.225 0.07 0.052 0 0.073 0.095 = 0.053495 := by
  sorry

end NUMINAMATH_CALUDE_total_tax_percentage_calculation_l404_40488


namespace NUMINAMATH_CALUDE_henry_games_count_l404_40470

theorem henry_games_count :
  ∀ (h n l : ℕ),
    h = 3 * n →                 -- Henry had 3 times as many games as Neil initially
    h = 2 * l →                 -- Henry had 2 times as many games as Linda initially
    n = 7 →                     -- Neil had 7 games initially
    l = 7 →                     -- Linda had 7 games initially
    h - 10 = 4 * (n + 6) →      -- After giving games, Henry has 4 times more games than Neil
    h = 62                      -- Henry originally had 62 games
  := by sorry

end NUMINAMATH_CALUDE_henry_games_count_l404_40470


namespace NUMINAMATH_CALUDE_line_length_after_erasing_l404_40432

-- Define the original length in meters
def original_length_m : ℝ := 1.5

-- Define the erased length in centimeters
def erased_length_cm : ℝ := 37.5

-- Define the conversion factor from meters to centimeters
def m_to_cm : ℝ := 100

-- Theorem statement
theorem line_length_after_erasing :
  (original_length_m * m_to_cm - erased_length_cm) = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_line_length_after_erasing_l404_40432


namespace NUMINAMATH_CALUDE_min_board_size_is_77_l404_40426

/-- A domino placement on a square board. -/
structure DominoPlacement where
  n : ℕ  -- Size of the square board
  dominoes : ℕ  -- Number of dominoes placed

/-- Checks if the domino placement is valid. -/
def is_valid_placement (p : DominoPlacement) : Prop :=
  p.dominoes * 2 = 2008 ∧  -- Total area covered by dominoes
  (p.n + 1)^2 ≥ p.dominoes * 6  -- Extended board can fit dominoes with shadows

/-- The minimum board size for a valid domino placement. -/
def min_board_size : ℕ := 77

/-- Theorem stating that 77 is the minimum board size for a valid domino placement. -/
theorem min_board_size_is_77 :
  ∀ p : DominoPlacement, is_valid_placement p → p.n ≥ min_board_size :=
by sorry

end NUMINAMATH_CALUDE_min_board_size_is_77_l404_40426


namespace NUMINAMATH_CALUDE_solve_for_k_l404_40460

theorem solve_for_k : ∃ k : ℚ, 
  (let x : ℚ := -3
   k * (x - 2) - 4 = k - 2 * x) ∧ 
  k = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l404_40460


namespace NUMINAMATH_CALUDE_geometric_sequence_l404_40468

theorem geometric_sequence (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n) →
  a 1 + a 3 = 10 →
  a 2 + a 4 = 5 →
  a 8 = 1 / 16 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_l404_40468


namespace NUMINAMATH_CALUDE_hotel_problem_l404_40481

/-- The number of men who spent Rs. 3 each on meals -/
def num_standard_spenders : ℕ := 8

/-- The amount spent by each of the standard spenders in Rs. -/
def standard_spend : ℚ := 3

/-- The total amount spent by all men in Rs. -/
def total_spend : ℚ := 29.25

/-- The extra amount spent by one man above the average in Rs. -/
def extra_spend : ℚ := 2

/-- The number of men who went to the hotel -/
def num_men : ℕ := 9

theorem hotel_problem :
  ∃ (avg_spend : ℚ),
    (num_standard_spenders : ℚ) * standard_spend + (avg_spend + extra_spend) = total_spend ∧
    avg_spend = total_spend / (num_men : ℚ) ∧
    num_men > num_standard_spenders :=
by sorry

end NUMINAMATH_CALUDE_hotel_problem_l404_40481


namespace NUMINAMATH_CALUDE_cookie_problem_indeterminate_l404_40417

/-- Represents the number of cookies Paco had and ate --/
structure CookieCount where
  initialSweet : ℕ
  initialSalty : ℕ
  eatenSweet : ℕ
  eatenSalty : ℕ

/-- Represents the conditions of the cookie problem --/
def CookieProblem (c : CookieCount) : Prop :=
  c.initialSalty = 6 ∧
  c.eatenSweet = 20 ∧
  c.eatenSalty = 34 ∧
  c.eatenSalty = c.eatenSweet + 14

theorem cookie_problem_indeterminate :
  ∀ (c : CookieCount), CookieProblem c →
    (c.initialSweet ≥ 20 ∧
     ∀ (n : ℕ), n ≥ 20 → ∃ (c' : CookieCount), CookieProblem c' ∧ c'.initialSweet = n) :=
by sorry

#check cookie_problem_indeterminate

end NUMINAMATH_CALUDE_cookie_problem_indeterminate_l404_40417


namespace NUMINAMATH_CALUDE_count_triangles_l404_40449

/-- A point in the plane with coordinates that are multiples of 3 -/
structure Point :=
  (x : ℤ)
  (y : ℤ)
  (x_multiple : 3 ∣ x)
  (y_multiple : 3 ∣ y)

/-- The equation 47x + y = 2353 -/
def satisfies_equation (p : Point) : Prop :=
  47 * p.x + p.y = 2353

/-- The area of triangle OPQ where O is the origin -/
def triangle_area (p q : Point) : ℚ :=
  (p.x * q.y - q.x * p.y : ℚ) / 2

/-- The main theorem -/
theorem count_triangles :
  ∃ (triangle_set : Finset (Point × Point)),
    (∀ (p q : Point), (p, q) ∈ triangle_set →
      p ≠ q ∧
      satisfies_equation p ∧
      satisfies_equation q ∧
      (triangle_area p q).num ≠ 0 ∧
      (triangle_area p q).den = 1) ∧
    triangle_set.card = 64 ∧
    ∀ (p q : Point),
      p ≠ q →
      satisfies_equation p →
      satisfies_equation q →
      (triangle_area p q).num ≠ 0 →
      (triangle_area p q).den = 1 →
      (p, q) ∈ triangle_set :=
sorry

end NUMINAMATH_CALUDE_count_triangles_l404_40449


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_irrational_l404_40418

/-- Given two consecutive even integers and their sum, the square root of the sum of their squares is irrational -/
theorem sqrt_sum_squares_irrational (x : ℤ) : 
  let a : ℤ := 2 * x
  let b : ℤ := 2 * x + 2
  let c : ℤ := a + b
  let D : ℤ := a^2 + b^2 + c^2
  Irrational (Real.sqrt (D : ℝ)) := by
sorry


end NUMINAMATH_CALUDE_sqrt_sum_squares_irrational_l404_40418


namespace NUMINAMATH_CALUDE_nested_sqrt_value_l404_40446

theorem nested_sqrt_value : ∃ x : ℝ, x > 0 ∧ x = Real.sqrt (2 - x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_value_l404_40446


namespace NUMINAMATH_CALUDE_largest_integer_solution_l404_40474

theorem largest_integer_solution (x : ℤ) : (3 - 2 * x > 0) → x ≤ 1 ∧ (∀ y : ℤ, 3 - 2 * y > 0 → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_solution_l404_40474


namespace NUMINAMATH_CALUDE_gcd_8_factorial_6_factorial_squared_l404_40430

theorem gcd_8_factorial_6_factorial_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8_factorial_6_factorial_squared_l404_40430


namespace NUMINAMATH_CALUDE_positive_interval_l404_40412

theorem positive_interval (x : ℝ) : (x + 2) * (x - 3) > 0 ↔ x < -2 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_interval_l404_40412


namespace NUMINAMATH_CALUDE_height_problem_l404_40466

theorem height_problem (h₁ h₂ h₃ h₄ : ℝ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- heights in increasing order
  h₂ = h₁ + 2 →                  -- difference between 1st and 2nd
  h₃ = h₂ + 2 →                  -- difference between 2nd and 3rd
  h₄ = h₃ + 6 →                  -- difference between 3rd and 4th
  (h₁ + h₂ + h₃ + h₄) / 4 = 76 → -- average height
  h₄ = 82 :=                     -- height of the fourth person
by sorry

end NUMINAMATH_CALUDE_height_problem_l404_40466


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l404_40440

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y / z + y * z / x + z * x / y) > 2 * (x^3 + y^3 + z^3)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l404_40440


namespace NUMINAMATH_CALUDE_total_share_l404_40437

theorem total_share (z y x : ℝ) : 
  z = 300 →
  y = 1.2 * z →
  x = 1.25 * y →
  x + y + z = 1110 := by
sorry

end NUMINAMATH_CALUDE_total_share_l404_40437


namespace NUMINAMATH_CALUDE_quadratic_function_with_specific_properties_l404_40424

theorem quadratic_function_with_specific_properties :
  ∀ (a b x₁ x₂ : ℝ),
    a < 0 →
    b > 0 →
    x₁ ≠ x₂ →
    x₁^2 + a*x₁ + b = 0 →
    x₂^2 + a*x₂ + b = 0 →
    ((x₁ - (-2) = x₂ - x₁) ∨ (x₁ / (-2) = x₂ / x₁)) →
    (∀ x, x^2 + a*x + b = x^2 - 5*x + 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_with_specific_properties_l404_40424


namespace NUMINAMATH_CALUDE_max_value_of_expression_l404_40480

/-- An arithmetic sequence with positive first term and common difference -/
structure ArithmeticSequence where
  a₁ : ℝ
  d : ℝ
  a₁_pos : 0 < a₁
  d_pos : 0 < d

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a₁ + (n - 1) * seq.d

theorem max_value_of_expression (seq : ArithmeticSequence)
    (h1 : seq.nthTerm 1 + seq.nthTerm 2 ≤ 60)
    (h2 : seq.nthTerm 2 + seq.nthTerm 3 ≤ 100) :
    5 * seq.nthTerm 1 + seq.nthTerm 5 ≤ 200 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l404_40480


namespace NUMINAMATH_CALUDE_curve_is_hyperbola_l404_40407

/-- The curve defined by r = 1 / (1 - sin θ) is a hyperbola -/
theorem curve_is_hyperbola (θ : ℝ) (r : ℝ) :
  r = 1 / (1 - Real.sin θ) → ∃ (a b c d e f : ℝ), 
    a ≠ 0 ∧ c ≠ 0 ∧ a * c < 0 ∧
    ∀ (x y : ℝ), a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0 := by
  sorry

end NUMINAMATH_CALUDE_curve_is_hyperbola_l404_40407


namespace NUMINAMATH_CALUDE_rectangle_triangle_altitude_l404_40497

theorem rectangle_triangle_altitude (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≥ b) :
  let rectangle_area := a * b
  let triangle_leg := 2 * b
  let triangle_hypotenuse := Real.sqrt (a^2 + triangle_leg^2)
  let triangle_area := (1/2) * a * triangle_leg
  triangle_area = rectangle_area →
  (2 * rectangle_area) / triangle_hypotenuse = (2 * a * b) / Real.sqrt (a^2 + 4 * b^2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_triangle_altitude_l404_40497


namespace NUMINAMATH_CALUDE_exact_time_l404_40489

/-- Represents the time in minutes after 4:00 --/
def t : ℝ := by sorry

/-- The angle of the minute hand at time t --/
def minute_hand (t : ℝ) : ℝ := 6 * t

/-- The angle of the hour hand at time t --/
def hour_hand (t : ℝ) : ℝ := 120 + 0.5 * t

/-- The condition that the time is between 4:00 and 5:00 --/
axiom time_range : 0 ≤ t ∧ t < 60

/-- The condition that the minute hand is opposite to where the hour hand was 5 minutes ago --/
axiom opposite_hands : 
  |minute_hand (t + 10) - hour_hand (t - 5)| = 180 ∨ 
  |minute_hand (t + 10) - hour_hand (t - 5)| = 540

theorem exact_time : t = 25 := by sorry

end NUMINAMATH_CALUDE_exact_time_l404_40489


namespace NUMINAMATH_CALUDE_prism_volume_l404_40414

/-- The volume of a right rectangular prism with face areas 30, 50, and 75 square centimeters is 335 cubic centimeters. -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 50) (h3 : b * c = 75) :
  a * b * c = 335 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l404_40414


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l404_40455

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def B : Set ℝ := {x | 0 < x ∧ x < 10}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l404_40455


namespace NUMINAMATH_CALUDE_abc_product_l404_40415

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 24 * Real.sqrt 3)
  (hac : a * c = 30 * Real.sqrt 3)
  (hbc : b * c = 40 * Real.sqrt 3) :
  a * b * c = 120 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l404_40415


namespace NUMINAMATH_CALUDE_binomial_133_133_l404_40467

theorem binomial_133_133 : Nat.choose 133 133 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_133_133_l404_40467


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l404_40401

theorem binomial_expansion_properties (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ = 1 ∧ a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -1) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l404_40401


namespace NUMINAMATH_CALUDE_isosceles_triangle_sides_l404_40402

/-- Given a rope of length 18 cm forming an isosceles triangle with one side of 5 cm,
    the length of the other two sides can be either 5 cm or 6.5 cm. -/
theorem isosceles_triangle_sides (rope_length : ℝ) (given_side : ℝ) : 
  rope_length = 18 → given_side = 5 → 
  ∃ (other_side : ℝ), (other_side = 5 ∨ other_side = 6.5) ∧ 
  ((2 * other_side + given_side = rope_length) ∨ 
   (2 * given_side + other_side = rope_length)) := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_sides_l404_40402


namespace NUMINAMATH_CALUDE_absolute_difference_inequality_l404_40493

theorem absolute_difference_inequality (x y : ℝ) 
  (hx : |x| < 1) (hy : |y| < 1) : 
  |x - y| < |1 - x*y| := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_inequality_l404_40493


namespace NUMINAMATH_CALUDE_swimming_pool_count_l404_40427

theorem swimming_pool_count (total : ℕ) (garage : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 70 → garage = 50 → both = 35 → neither = 15 → 
  ∃ pool : ℕ, pool = 40 ∧ total = garage + pool - both + neither :=
by sorry

end NUMINAMATH_CALUDE_swimming_pool_count_l404_40427


namespace NUMINAMATH_CALUDE_function_properties_quadratic_inequality_solution_set_maximum_value_of_fraction_l404_40413

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b - 8) * x - a - a * b

theorem function_properties :
  ∀ a b : ℝ,
  (∀ x ∈ Set.Ioo (-3) 2, f a b x > 0) ∧
  (∀ x ∈ Set.Iic (-3) ∪ Set.Ici 2, f a b x < 0) →
  ∀ x, f a b x = -3 * x^2 - 3 * x + 15 :=
sorry

theorem quadratic_inequality_solution_set :
  ∀ a b c : ℝ,
  (∀ x : ℝ, a * x^2 + b * x + c ≤ 0) →
  c ≤ -25/12 :=
sorry

theorem maximum_value_of_fraction :
  ∀ x : ℝ,
  x > -1 →
  (f (-3) 5 x - 21) / (x + 1) ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_function_properties_quadratic_inequality_solution_set_maximum_value_of_fraction_l404_40413


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l404_40441

theorem simplify_fraction_product : 
  (4 : ℚ) * (18 / 5) * (35 / -63) * (8 / 14) = -32 / 7 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l404_40441


namespace NUMINAMATH_CALUDE_smaller_number_is_42_l404_40409

theorem smaller_number_is_42 (x y : ℕ) (h1 : x + y = 96) (h2 : y = x + 12) : x = 42 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_is_42_l404_40409


namespace NUMINAMATH_CALUDE_tennis_game_wins_l404_40454

theorem tennis_game_wins (total_games : ℕ) (player_a_wins player_b_wins player_c_wins : ℕ) :
  total_games = 6 →
  player_a_wins = 5 →
  player_b_wins = 2 →
  player_c_wins = 1 →
  ∃ player_d_wins : ℕ, player_d_wins = 4 ∧ player_a_wins + player_b_wins + player_c_wins + player_d_wins = 2 * total_games :=
by sorry

end NUMINAMATH_CALUDE_tennis_game_wins_l404_40454


namespace NUMINAMATH_CALUDE_product_inequality_l404_40419

theorem product_inequality (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0) (h : A * B * C = 1) :
  (A - 1 + 1/B) * (B - 1 + 1/C) * (C - 1 + 1/A) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l404_40419


namespace NUMINAMATH_CALUDE_road_trip_duration_l404_40408

theorem road_trip_duration (family_size : ℕ) (water_per_person_per_hour : ℚ) 
  (total_water_bottles : ℕ) (h : ℕ) : 
  family_size = 4 → 
  water_per_person_per_hour = 1/2 → 
  total_water_bottles = 32 → 
  (2 * h : ℚ) * (family_size : ℚ) * water_per_person_per_hour = total_water_bottles → 
  h = 8 := by
sorry

end NUMINAMATH_CALUDE_road_trip_duration_l404_40408


namespace NUMINAMATH_CALUDE_richmond_tigers_ticket_sales_l404_40459

theorem richmond_tigers_ticket_sales (first_half_sales second_half_sales : ℕ) 
  (h1 : first_half_sales = 3867)
  (h2 : second_half_sales = 5703) :
  first_half_sales + second_half_sales = 9570 := by
  sorry

end NUMINAMATH_CALUDE_richmond_tigers_ticket_sales_l404_40459


namespace NUMINAMATH_CALUDE_puppy_feeding_schedule_l404_40494

-- Define the feeding schedule and amounts
def total_days : ℕ := 28 -- 4 weeks
def today_food : ℚ := 1/2
def last_two_weeks_daily : ℚ := 1
def total_food : ℚ := 25

-- Define the unknown amount for the first two weeks
def first_two_weeks_per_meal : ℚ := 1/4

-- Theorem statement
theorem puppy_feeding_schedule :
  let first_two_weeks_total := 14 * 3 * first_two_weeks_per_meal
  let last_two_weeks_total := 14 * last_two_weeks_daily
  today_food + first_two_weeks_total + last_two_weeks_total = total_food :=
by sorry

end NUMINAMATH_CALUDE_puppy_feeding_schedule_l404_40494


namespace NUMINAMATH_CALUDE_mean_equality_problem_l404_40436

theorem mean_equality_problem : ∃ z : ℚ, (7 + 12 + 21) / 3 = (15 + z) / 2 ∧ z = 35 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_problem_l404_40436


namespace NUMINAMATH_CALUDE_fractional_equation_one_l404_40473

theorem fractional_equation_one (x : ℝ) : 
  x ≠ 0 ∧ x ≠ -1 → (2 / x = 3 / (x + 1) ↔ x = 2) := by sorry

end NUMINAMATH_CALUDE_fractional_equation_one_l404_40473


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l404_40425

def geometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem fourth_term_of_geometric_sequence (a : ℕ → ℝ) :
  geometricSequence a → a 1 = 2 → a 2 = 4 → a 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l404_40425


namespace NUMINAMATH_CALUDE_lcm_sum_inequality_l404_40456

theorem lcm_sum_inequality (a b c d e : ℕ) (h1 : 1 ≤ a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < e) :
  (1 : ℚ) / Nat.lcm a b + 1 / Nat.lcm b c + 1 / Nat.lcm c d + 1 / Nat.lcm d e ≤ 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_lcm_sum_inequality_l404_40456


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_l404_40463

/-- The sum of the first n terms of an arithmetic sequence -/
def S (a d : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * d) / 2

/-- Theorem: If the ratio of S_{4n} to S_n is constant for an arithmetic sequence 
    with common difference 5, then the first term is 5/2 -/
theorem arithmetic_sequence_constant_ratio (a : ℚ) :
  (∀ n : ℕ, n > 0 → ∃ c : ℚ, S a 5 (4 * n) / S a 5 n = c) →
  a = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_l404_40463


namespace NUMINAMATH_CALUDE_sum_of_triangle_operations_l404_40491

def triangle_operation (a b c : ℤ) : ℤ := 2*a + b - c

theorem sum_of_triangle_operations : 
  triangle_operation 1 2 3 + triangle_operation 4 6 5 + triangle_operation 2 7 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_triangle_operations_l404_40491


namespace NUMINAMATH_CALUDE_divisor_problem_l404_40499

theorem divisor_problem : ∃ (N D : ℕ), 
  (N % D = 6) ∧ 
  (N % 19 = 7) ∧ 
  (D = 39) := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l404_40499


namespace NUMINAMATH_CALUDE_bruce_lost_eggs_main_theorem_l404_40495

/-- Proof that Bruce lost 70 eggs -/
theorem bruce_lost_eggs : ℕ → ℕ → ℕ → Prop :=
  fun initial_eggs remaining_eggs lost_eggs =>
    initial_eggs = 75 →
    remaining_eggs = 5 →
    lost_eggs = initial_eggs - remaining_eggs →
    lost_eggs = 70

/-- Main theorem statement -/
theorem main_theorem : ∃ lost_eggs : ℕ, bruce_lost_eggs 75 5 lost_eggs := by
  sorry

end NUMINAMATH_CALUDE_bruce_lost_eggs_main_theorem_l404_40495


namespace NUMINAMATH_CALUDE_door_challenge_sequences_l404_40411

/-- Represents the number of doors and family members -/
def n : ℕ := 7

/-- Represents the number of binary choices made after the first person -/
def m : ℕ := n - 1

/-- The number of possible sequences given n doors and m binary choices -/
def num_sequences (n m : ℕ) : ℕ := 2^m

theorem door_challenge_sequences :
  n = 7 → m = 6 → num_sequences n m = 64 := by
  sorry

end NUMINAMATH_CALUDE_door_challenge_sequences_l404_40411


namespace NUMINAMATH_CALUDE_bus_stop_distance_l404_40465

theorem bus_stop_distance (n : ℕ) (d : ℝ) (h1 : n = 12) (h2 : d > 0) 
  (h3 : 3 * d = 3300) : (n - 1) * d / 1000 = 12.1 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_distance_l404_40465


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l404_40461

-- Define the circle's equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 5

-- Define the line's equation
def line_equation (x y : ℝ) : Prop := 2*x - y - 1 = 0

-- State the theorem
theorem circle_tangent_to_line :
  -- The circle has center (-1, 2)
  ∃ (x₀ y₀ : ℝ), x₀ = -1 ∧ y₀ = 2 ∧
  -- The circle is tangent to the line
  ∃ (x y : ℝ), circle_equation x y ∧ line_equation x y ∧
  -- Any point satisfying both equations is unique (tangency condition)
  ∀ (x' y' : ℝ), circle_equation x' y' ∧ line_equation x' y' → x' = x ∧ y' = y :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l404_40461


namespace NUMINAMATH_CALUDE_min_cos_C_in_triangle_l404_40410

theorem min_cos_C_in_triangle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_eq : a^2 + 2*b^2 = 3*c^2) :
  let cos_C := (a^2 + b^2 - c^2) / (2*a*b)
  cos_C ≥ Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_min_cos_C_in_triangle_l404_40410


namespace NUMINAMATH_CALUDE_no_integer_distances_point_l404_40422

theorem no_integer_distances_point (a b : ℕ) (ha : Odd a) (hb : Odd b) :
  ¬ ∃ (x y : ℚ), 0 < x ∧ x < b ∧ 0 < y ∧ y < a ∧
    (∀ (i j : ℕ), i ≤ 1 ∧ j ≤ 1 →
      ∃ (n : ℕ), (x - i * b)^2 + (y - j * a)^2 = n^2) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_distances_point_l404_40422


namespace NUMINAMATH_CALUDE_quadratic_sequence_l404_40486

theorem quadratic_sequence (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 1)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 12)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 123) :
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ = 334 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sequence_l404_40486


namespace NUMINAMATH_CALUDE_four_digit_number_l404_40447

/-- Represents a 6x6 grid of numbers -/
def Grid := Matrix (Fin 6) (Fin 6) Nat

/-- Check if a number is within the range 1 to 6 -/
def inRange (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 6

/-- Check if a list of numbers contains no duplicates -/
def noDuplicates (l : List Nat) : Prop := l.Nodup

/-- Check if a number is prime -/
def isPrime (n : Nat) : Prop := Nat.Prime n

/-- Check if a number is composite -/
def isComposite (n : Nat) : Prop := ¬(isPrime n) ∧ n > 1

/-- Theorem: Under the given conditions, the four-digit number is 4123 -/
theorem four_digit_number (g : Grid) 
  (range_check : ∀ i j, inRange (g i j))
  (row_unique : ∀ i, noDuplicates (List.ofFn (λ j => g i j)))
  (col_unique : ∀ j, noDuplicates (List.ofFn (λ i => g i j)))
  (rect_unique : ∀ i j, noDuplicates [g i j, g i (j+1), g i (j+2), g (i+1) j, g (i+1) (j+1), g (i+1) (j+2)])
  (circle_sum : ∀ i j, isComposite (g i j + g (i+1) j) → ∀ k l, (k, l) ≠ (i, j) → isPrime (g k l + g (k+1) l))
  : ∃ i j k l, g i j = 4 ∧ g k j = 1 ∧ g k l = 2 ∧ g i l = 3 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_l404_40447


namespace NUMINAMATH_CALUDE_fraction_relation_l404_40444

theorem fraction_relation (n d : ℚ) (k : ℚ) : 
  d = k * (2 * n) →
  (n + 1) / (d + 1) = 3 / 5 →
  n / d = 5 / 9 →
  k = 9 / 10 := by
sorry

end NUMINAMATH_CALUDE_fraction_relation_l404_40444


namespace NUMINAMATH_CALUDE_negative_division_result_l404_40464

theorem negative_division_result : (-150) / (-25) = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_division_result_l404_40464


namespace NUMINAMATH_CALUDE_sons_age_is_24_l404_40433

/-- Proves that the son's age is 24 given the conditions of the problem -/
theorem sons_age_is_24 (son_age father_age : ℕ) : 
  father_age = son_age + 26 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 24 := by
sorry

end NUMINAMATH_CALUDE_sons_age_is_24_l404_40433


namespace NUMINAMATH_CALUDE_correct_num_bedrooms_l404_40490

/-- The number of bedrooms to clean -/
def num_bedrooms : ℕ := sorry

/-- Time in minutes to clean one bedroom -/
def bedroom_time : ℕ := 20

/-- Time in minutes to clean the living room -/
def living_room_time : ℕ := num_bedrooms * bedroom_time

/-- Time in minutes to clean one bathroom -/
def bathroom_time : ℕ := 2 * living_room_time

/-- Time in minutes to clean the house (bedrooms, living room, and bathrooms) -/
def house_time : ℕ := num_bedrooms * bedroom_time + living_room_time + 2 * bathroom_time

/-- Time in minutes to clean outside -/
def outside_time : ℕ := 2 * house_time

/-- Total time in minutes for all three siblings to work -/
def total_work_time : ℕ := 3 * 4 * 60

theorem correct_num_bedrooms : num_bedrooms = 3 := by sorry

end NUMINAMATH_CALUDE_correct_num_bedrooms_l404_40490


namespace NUMINAMATH_CALUDE_min_weighings_to_identify_defective_l404_40475

/-- Represents a piece that can be either standard or defective -/
inductive Piece
| Standard
| Defective

/-- Represents the result of a weighing -/
inductive WeighingResult
| Equal
| LeftHeavier
| RightHeavier

/-- A function that simulates a weighing on a balance scale -/
def weigh (left right : List Piece) : WeighingResult := sorry

/-- The set of all possible pieces -/
def allPieces : Finset Piece := sorry

/-- The number of pieces -/
def numPieces : Nat := 5

/-- The number of standard pieces -/
def numStandard : Nat := 4

/-- The number of defective pieces -/
def numDefective : Nat := 1

/-- A strategy for identifying the defective piece -/
def identifyDefective : Nat → Option Piece := sorry

theorem min_weighings_to_identify_defective :
  ∃ (strategy : Nat → Option Piece),
    (∀ defective : Piece, 
      defective ∈ allPieces → 
      ∃ n : Nat, n ≤ 3 ∧ strategy n = some defective) ∧
    (∀ m : Nat, 
      (∀ defective : Piece, 
        defective ∈ allPieces → 
        (∃ n : Nat, n ≤ m ∧ strategy n = some defective)) → 
      m ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_min_weighings_to_identify_defective_l404_40475


namespace NUMINAMATH_CALUDE_sin_390_degrees_l404_40471

theorem sin_390_degrees : Real.sin (390 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_390_degrees_l404_40471


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l404_40400

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 - 11x - 18 -/
def a : ℝ := 5
def b : ℝ := -11
def c : ℝ := -18

theorem quadratic_discriminant : discriminant a b c = 481 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l404_40400
