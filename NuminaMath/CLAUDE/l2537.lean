import Mathlib

namespace NUMINAMATH_CALUDE_bbq_ice_per_person_l2537_253759

/-- Given the conditions of Chad's BBQ, prove that the amount of ice needed per person is 2 pounds. -/
theorem bbq_ice_per_person (people : ℕ) (pack_price : ℚ) (pack_size : ℕ) (total_spent : ℚ) :
  people = 15 →
  pack_price = 3 →
  pack_size = 10 →
  total_spent = 9 →
  (total_spent / pack_price * pack_size) / people = 2 := by
  sorry

#check bbq_ice_per_person

end NUMINAMATH_CALUDE_bbq_ice_per_person_l2537_253759


namespace NUMINAMATH_CALUDE_exam_score_problem_l2537_253704

theorem exam_score_problem (scores : List ℝ) (avg : ℝ) : 
  scores.length = 4 →
  scores = [80, 90, 100, 110] →
  avg = 96 →
  (scores.sum + (5 * avg - scores.sum)) / 5 = avg →
  5 * avg - scores.sum = 100 := by
sorry

end NUMINAMATH_CALUDE_exam_score_problem_l2537_253704


namespace NUMINAMATH_CALUDE_fraction_equality_l2537_253730

theorem fraction_equality (x y : ℝ) (h : x ≠ -y) : (-x + y) / (-x - y) = (x - y) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2537_253730


namespace NUMINAMATH_CALUDE_danai_decorations_l2537_253764

/-- The number of decorations Danai will put up in total -/
def total_decorations (skulls broomsticks spiderwebs pumpkins cauldron budget_left left_to_put_up : ℕ) : ℕ :=
  skulls + broomsticks + spiderwebs + pumpkins + cauldron + budget_left + left_to_put_up

/-- Theorem stating the total number of decorations Danai will put up -/
theorem danai_decorations :
  ∀ (skulls broomsticks spiderwebs pumpkins cauldron budget_left left_to_put_up : ℕ),
    skulls = 12 →
    broomsticks = 4 →
    spiderwebs = 12 →
    pumpkins = 2 * spiderwebs →
    cauldron = 1 →
    budget_left = 20 →
    left_to_put_up = 10 →
    total_decorations skulls broomsticks spiderwebs pumpkins cauldron budget_left left_to_put_up = 83 :=
by sorry

end NUMINAMATH_CALUDE_danai_decorations_l2537_253764


namespace NUMINAMATH_CALUDE_greatest_real_part_of_sixth_power_l2537_253717

theorem greatest_real_part_of_sixth_power : 
  let z₁ : ℂ := -3
  let z₂ : ℂ := -Real.sqrt 6 + Complex.I
  let z₃ : ℂ := -Real.sqrt 3 + (Real.sqrt 3 : ℝ) * Complex.I
  let z₄ : ℂ := -1 + (Real.sqrt 6 : ℝ) * Complex.I
  let z₅ : ℂ := 2 * Complex.I
  Complex.re (z₁^6) > Complex.re (z₂^6) ∧
  Complex.re (z₁^6) > Complex.re (z₃^6) ∧
  Complex.re (z₁^6) > Complex.re (z₄^6) ∧
  Complex.re (z₁^6) > Complex.re (z₅^6) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_real_part_of_sixth_power_l2537_253717


namespace NUMINAMATH_CALUDE_P_in_fourth_quadrant_iff_m_in_range_l2537_253772

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point being in the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The point P with coordinates dependent on m -/
def P (m : ℝ) : Point :=
  { x := m + 3, y := m - 2 }

/-- Theorem stating the range of m for P to be in the fourth quadrant -/
theorem P_in_fourth_quadrant_iff_m_in_range (m : ℝ) :
  in_fourth_quadrant (P m) ↔ -3 < m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_P_in_fourth_quadrant_iff_m_in_range_l2537_253772


namespace NUMINAMATH_CALUDE_consecutive_integers_equation_l2537_253733

theorem consecutive_integers_equation (x y z n : ℤ) : 
  x = y + 1 → 
  y = z + 1 → 
  x > y → 
  y > z → 
  z = 3 → 
  2*x + 3*y + 3*z = 5*y + n → 
  n = 11 := by sorry

end NUMINAMATH_CALUDE_consecutive_integers_equation_l2537_253733


namespace NUMINAMATH_CALUDE_min_velocity_increase_is_6_l2537_253736

/-- Represents a car with its velocity -/
structure Car where
  velocity : ℝ

/-- Represents the road scenario -/
structure RoadScenario where
  carA : Car
  carB : Car
  carC : Car
  initialDistanceAB : ℝ
  initialDistanceAC : ℝ

/-- Calculates the minimum velocity increase needed for car A -/
def minVelocityIncrease (scenario : RoadScenario) : ℝ :=
  sorry

/-- Theorem stating the minimum velocity increase for the given scenario -/
theorem min_velocity_increase_is_6 (scenario : RoadScenario) 
  (h1 : scenario.carA.velocity > scenario.carB.velocity)
  (h2 : scenario.initialDistanceAB = 50)
  (h3 : scenario.initialDistanceAC = 300)
  (h4 : scenario.carB.velocity = 50)
  (h5 : scenario.carC.velocity = 70)
  (h6 : scenario.carA.velocity = 68) :
  minVelocityIncrease scenario = 6 :=
sorry

end NUMINAMATH_CALUDE_min_velocity_increase_is_6_l2537_253736


namespace NUMINAMATH_CALUDE_square_difference_of_integers_l2537_253722

theorem square_difference_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 50) 
  (diff_eq : x - y = 12) : 
  x^2 - y^2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_integers_l2537_253722


namespace NUMINAMATH_CALUDE_prob_three_even_out_of_six_l2537_253788

/-- The probability of rolling an even number on a fair 12-sided die -/
def prob_even : ℚ := 1 / 2

/-- The number of ways to choose 3 dice from 6 -/
def choose_3_from_6 : ℕ := 20

/-- The probability of a specific scenario where exactly 3 dice show even -/
def prob_specific_scenario : ℚ := (1 / 2) ^ 6

/-- The probability of exactly three out of six fair 12-sided dice showing an even number -/
theorem prob_three_even_out_of_six : 
  choose_3_from_6 * prob_specific_scenario = 5 / 16 := by sorry

end NUMINAMATH_CALUDE_prob_three_even_out_of_six_l2537_253788


namespace NUMINAMATH_CALUDE_complex_norm_problem_l2537_253749

theorem complex_norm_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 12)
  (h2 : Complex.abs (z + 3 * w) = 9)
  (h3 : Complex.abs (z - w) = 7) :
  Complex.abs w = Real.sqrt 36.75 :=
sorry

end NUMINAMATH_CALUDE_complex_norm_problem_l2537_253749


namespace NUMINAMATH_CALUDE_range_of_a_l2537_253786

-- Define the custom operation
def circleMultiply (x y : ℝ) : ℝ := x * (1 - y)

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, circleMultiply (x - a) (x + a) < 2) → 
  -1 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2537_253786


namespace NUMINAMATH_CALUDE_track_circumference_l2537_253725

/-- The circumference of a circular track given two people walking in opposite directions -/
theorem track_circumference (v1 v2 t : ℝ) (h1 : v1 = 4.5) (h2 : v2 = 3.75) (h3 : t = 5.28 / 60) :
  v1 * t + v2 * t = 0.726 := by
  sorry

end NUMINAMATH_CALUDE_track_circumference_l2537_253725


namespace NUMINAMATH_CALUDE_sandy_comic_books_l2537_253756

/-- Proves that Sandy bought 6 comic books given the initial conditions -/
theorem sandy_comic_books :
  let initial_books : ℕ := 14
  let sold_books : ℕ := initial_books / 2
  let current_books : ℕ := 13
  let bought_books : ℕ := current_books - (initial_books - sold_books)
  bought_books = 6 := by
  sorry

end NUMINAMATH_CALUDE_sandy_comic_books_l2537_253756


namespace NUMINAMATH_CALUDE_fishing_ratio_l2537_253777

theorem fishing_ratio (sara_catch melanie_catch : ℕ) 
  (h1 : sara_catch = 5)
  (h2 : melanie_catch = 10) :
  (melanie_catch : ℚ) / sara_catch = 2 := by
  sorry

end NUMINAMATH_CALUDE_fishing_ratio_l2537_253777


namespace NUMINAMATH_CALUDE_school_count_correct_l2537_253796

/-- Represents the number of primary schools in a town. -/
def num_schools : ℕ := 4

/-- Represents the capacity of the first two schools. -/
def capacity_large : ℕ := 400

/-- Represents the capacity of the other two schools. -/
def capacity_small : ℕ := 340

/-- Represents the total capacity of all schools. -/
def total_capacity : ℕ := 1480

/-- Theorem stating that the number of schools is correct given the capacities. -/
theorem school_count_correct : 
  2 * capacity_large + 2 * capacity_small = total_capacity ∧
  num_schools = 2 + 2 := by sorry

end NUMINAMATH_CALUDE_school_count_correct_l2537_253796


namespace NUMINAMATH_CALUDE_sine_symmetry_axis_symmetric_angles_sqrt_cos_minus_one_even_l2537_253712

open Real

-- Statement 2
theorem sine_symmetry_axis (k : ℤ) :
  ∀ x : ℝ, sin x = sin (π - x + (k * 2 * π)) := by sorry

-- Statement 3
theorem symmetric_angles (α β : ℝ) (k : ℤ) :
  (∀ x : ℝ, sin (α + x) = sin (β - x)) →
  α + β = (2 * k - 1) * π := by sorry

-- Statement 5
theorem sqrt_cos_minus_one_even :
  ∀ x : ℝ, sqrt (cos x - 1) = sqrt (cos (-x) - 1) := by sorry

end NUMINAMATH_CALUDE_sine_symmetry_axis_symmetric_angles_sqrt_cos_minus_one_even_l2537_253712


namespace NUMINAMATH_CALUDE_hidden_dots_sum_l2537_253754

/-- Represents a standard six-sided die -/
def StandardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The sum of all numbers on a standard die -/
def DieSum : ℕ := Finset.sum StandardDie id

/-- The number of dice in the stack -/
def NumDice : ℕ := 4

/-- The visible numbers on the stack -/
def VisibleNumbers : Finset ℕ := {1, 2, 3, 5, 6}

/-- The sum of visible numbers -/
def VisibleSum : ℕ := Finset.sum VisibleNumbers id

theorem hidden_dots_sum :
  NumDice * DieSum - VisibleSum = 67 := by sorry

end NUMINAMATH_CALUDE_hidden_dots_sum_l2537_253754


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2537_253794

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_middle_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a3 : a 3 = 4)
  (h_a5 : a 5 = m)
  (h_a7 : a 7 = 16) :
  m = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2537_253794


namespace NUMINAMATH_CALUDE_tan_315_degrees_l2537_253782

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l2537_253782


namespace NUMINAMATH_CALUDE_function_is_constant_l2537_253720

/-- Given a function f: ℝ → ℝ satisfying certain conditions, prove that f is constant. -/
theorem function_is_constant (f : ℝ → ℝ) (a : ℝ) (ha : a > 0)
  (h1 : ∀ x, 0 < f x ∧ f x ≤ a)
  (h2 : ∀ x y, Real.sqrt (f x * f y) ≥ f ((x + y) / 2)) :
  ∃ c, ∀ x, f x = c :=
sorry

end NUMINAMATH_CALUDE_function_is_constant_l2537_253720


namespace NUMINAMATH_CALUDE_real_y_condition_l2537_253776

theorem real_y_condition (x : ℝ) : 
  (∃ y : ℝ, 4 * y^2 + 6 * x * y + x + 10 = 0) ↔ (x ≤ -17/9 ∨ x ≥ 7/3) :=
by sorry

end NUMINAMATH_CALUDE_real_y_condition_l2537_253776


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2537_253724

theorem geometric_sequence_sum (a : ℝ) : 
  (a + 2*a + 4*a + 8*a = 1) →  -- Sum of first 4 terms equals 1
  (a + 2*a + 4*a + 8*a + 16*a + 32*a + 64*a + 128*a = 17) :=  -- Sum of first 8 terms equals 17
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2537_253724


namespace NUMINAMATH_CALUDE_snow_probability_first_week_l2537_253727

def probability_of_snow (days : ℕ) (daily_prob : ℚ) : ℚ :=
  1 - (1 - daily_prob) ^ days

theorem snow_probability_first_week :
  let prob_first_four := probability_of_snow 4 (1/4)
  let prob_next_three := probability_of_snow 3 (1/3)
  1 - (1 - prob_first_four) * (1 - prob_next_three) = 29/32 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_first_week_l2537_253727


namespace NUMINAMATH_CALUDE_intersected_cubes_count_l2537_253713

/-- Represents a 3D coordinate --/
structure Coord :=
  (x y z : ℕ)

/-- Represents a cube --/
structure Cube :=
  (side_length : ℕ)

/-- Represents a plane perpendicular to the main diagonal of a cube --/
structure DiagonalPlane :=
  (cube : Cube)
  (passes_through_center : Bool)

/-- Counts the number of unit cubes intersected by a diagonal plane in a larger cube --/
def count_intersected_cubes (c : Cube) (p : DiagonalPlane) : ℕ :=
  sorry

/-- The main theorem to be proved --/
theorem intersected_cubes_count (c : Cube) (p : DiagonalPlane) :
  c.side_length = 5 →
  p.cube = c →
  p.passes_through_center = true →
  count_intersected_cubes c p = 55 :=
sorry

end NUMINAMATH_CALUDE_intersected_cubes_count_l2537_253713


namespace NUMINAMATH_CALUDE_otimes_example_l2537_253718

-- Define the ⊗ operation
def otimes (a b : ℤ) : ℤ := (a + b) * (a - b)

-- State the theorem
theorem otimes_example : otimes 4 (otimes 2 (-1)) = 7 := by sorry

end NUMINAMATH_CALUDE_otimes_example_l2537_253718


namespace NUMINAMATH_CALUDE_average_of_two_numbers_l2537_253734

theorem average_of_two_numbers (a b c : ℝ) : 
  (a + b + c) / 3 = 48 → c = 32 → (a + b) / 2 = 56 := by
sorry

end NUMINAMATH_CALUDE_average_of_two_numbers_l2537_253734


namespace NUMINAMATH_CALUDE_parabola_perpendicular_chords_locus_l2537_253700

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a chord of the parabola -/
structure Chord where
  slope : ℝ

/-- The locus of point M -/
def locus (p : ℝ) (x y : ℝ) : Prop :=
  (x - 2*p)^2 + y^2 = 4*p^2

theorem parabola_perpendicular_chords_locus 
  (para : Parabola) 
  (chord1 chord2 : Chord) 
  (O M : Point) :
  O.x = 0 ∧ O.y = 0 ∧  -- Vertex O at origin
  (chord1.slope * chord2.slope = -1) →  -- Perpendicular chords
  locus para.p M.x M.y  -- Locus of projection M
  := by sorry

end NUMINAMATH_CALUDE_parabola_perpendicular_chords_locus_l2537_253700


namespace NUMINAMATH_CALUDE_equation_solution_l2537_253762

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), (21 / (x₁^2 - 9) - 3 / (x₁ - 3) = 2) ∧ 
                 (21 / (x₂^2 - 9) - 3 / (x₂ - 3) = 2) ∧ 
                 (abs (x₁ - 4.695) < 0.001) ∧ 
                 (abs (x₂ + 3.195) < 0.001) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2537_253762


namespace NUMINAMATH_CALUDE_amelia_wins_probability_l2537_253750

/-- Probability of Amelia's coin landing heads -/
def p_amelia : ℚ := 1/4

/-- Probability of Blaine's coin landing heads -/
def p_blaine : ℚ := 1/6

/-- Probability of Amelia winning -/
noncomputable def p_amelia_wins : ℚ := 2/3

/-- Theorem stating that the probability of Amelia winning is 2/3 -/
theorem amelia_wins_probability :
  p_amelia_wins = p_amelia * (1 - p_blaine) + p_amelia * p_blaine + 
  (1 - p_amelia) * (1 - p_blaine) * p_amelia_wins :=
by sorry

end NUMINAMATH_CALUDE_amelia_wins_probability_l2537_253750


namespace NUMINAMATH_CALUDE_negation_of_forall_squared_plus_one_nonnegative_l2537_253714

theorem negation_of_forall_squared_plus_one_nonnegative :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + 1 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_squared_plus_one_nonnegative_l2537_253714


namespace NUMINAMATH_CALUDE_num_paths_through_F_and_H_l2537_253781

/-- A point on the grid --/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Calculate the number of paths between two points on a grid --/
def numPaths (start finish : GridPoint) : ℕ :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The grid layout --/
def E : GridPoint := ⟨0, 0⟩
def F : GridPoint := ⟨3, 2⟩
def H : GridPoint := ⟨5, 4⟩
def G : GridPoint := ⟨8, 4⟩

/-- The theorem to prove --/
theorem num_paths_through_F_and_H : 
  numPaths E F * numPaths F H * numPaths H G = 60 := by
  sorry

end NUMINAMATH_CALUDE_num_paths_through_F_and_H_l2537_253781


namespace NUMINAMATH_CALUDE_sales_function_properties_l2537_253702

def f (x : ℝ) : ℝ := x^2 - 7*x + 14

theorem sales_function_properties :
  (∃ (a b : ℝ), a < b ∧ 
    (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y) ∧
    (∀ x y, b ≤ x ∧ x < y → f x ≤ f y)) ∧
  f 1 = 8 ∧
  f 3 = 2 := by sorry

end NUMINAMATH_CALUDE_sales_function_properties_l2537_253702


namespace NUMINAMATH_CALUDE_eggs_per_box_l2537_253765

/-- Given that Maria has 3 boxes of eggs and a total of 21 eggs, 
    prove that each box contains 7 eggs. -/
theorem eggs_per_box (total_eggs : ℕ) (num_boxes : ℕ) 
  (h1 : total_eggs = 21) (h2 : num_boxes = 3) : 
  total_eggs / num_boxes = 7 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_box_l2537_253765


namespace NUMINAMATH_CALUDE_base_nine_solution_l2537_253766

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem base_nine_solution :
  ∃! b : Nat, b > 0 ∧ 
    to_decimal [1, 7, 2] b + to_decimal [1, 4, 5] b = to_decimal [3, 2, 7] b :=
by sorry

end NUMINAMATH_CALUDE_base_nine_solution_l2537_253766


namespace NUMINAMATH_CALUDE_group_size_problem_l2537_253726

theorem group_size_problem (total_collection : ℕ) 
  (h1 : total_collection = 2916) : ∃ n : ℕ, n * n = total_collection ∧ n = 54 := by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l2537_253726


namespace NUMINAMATH_CALUDE_inequality_proof_l2537_253783

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : y * z + z * x + x * y = 1) : 
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) ≤ (4/9) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2537_253783


namespace NUMINAMATH_CALUDE_bells_toll_together_l2537_253771

theorem bells_toll_together (bell1 bell2 bell3 bell4 : ℕ) 
  (h1 : bell1 = 9) (h2 : bell2 = 10) (h3 : bell3 = 14) (h4 : bell4 = 18) :
  Nat.lcm bell1 (Nat.lcm bell2 (Nat.lcm bell3 bell4)) = 630 := by
  sorry

end NUMINAMATH_CALUDE_bells_toll_together_l2537_253771


namespace NUMINAMATH_CALUDE_dvd_packs_theorem_l2537_253787

/-- The number of DVD packs that can be bought with a given amount of money -/
def dvd_packs (total_money : ℚ) (pack_cost : ℚ) : ℚ :=
  total_money / pack_cost

/-- Theorem: Given 110 dollars and a pack cost of 11 dollars, 10 DVD packs can be bought -/
theorem dvd_packs_theorem : dvd_packs 110 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_dvd_packs_theorem_l2537_253787


namespace NUMINAMATH_CALUDE_function_composition_l2537_253703

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := 3 * x - 5

-- State the theorem
theorem function_composition :
  ∀ x : ℝ, f (g x) = 6 * x - 7 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l2537_253703


namespace NUMINAMATH_CALUDE_max_floors_theorem_l2537_253732

/-- Represents a building with elevators and floors -/
structure Building where
  num_elevators : ℕ
  num_floors : ℕ
  stops_per_elevator : ℕ
  all_pairs_connected : Bool

/-- The maximum number of floors possible for a building with given constraints -/
def max_floors (b : Building) : ℕ :=
  sorry

/-- Theorem stating that for a building with 7 elevators, each stopping on 6 floors,
    and all pairs of floors connected, the maximum number of floors is 14 -/
theorem max_floors_theorem (b : Building) 
  (h1 : b.num_elevators = 7)
  (h2 : b.stops_per_elevator = 6)
  (h3 : b.all_pairs_connected = true) :
  max_floors b = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_floors_theorem_l2537_253732


namespace NUMINAMATH_CALUDE_total_swim_distance_l2537_253746

/-- The total distance Molly swam on Saturday in meters -/
def saturday_distance : ℕ := 400

/-- The total distance Molly swam on Sunday in meters -/
def sunday_distance : ℕ := 300

/-- The theorem states that the total distance Molly swam in all four pools
    is equal to the sum of the distances she swam on Saturday and Sunday -/
theorem total_swim_distance :
  saturday_distance + sunday_distance = 700 := by
  sorry

end NUMINAMATH_CALUDE_total_swim_distance_l2537_253746


namespace NUMINAMATH_CALUDE_rachels_father_age_at_25_is_60_l2537_253741

/-- Calculates the age of Rachel's father when Rachel is 25 years old -/
def rachels_father_age_at_25 (rachel_current_age : ℕ) (grandfather_age_multiplier : ℕ) (father_age_difference : ℕ) : ℕ :=
  let grandfather_age := rachel_current_age * grandfather_age_multiplier
  let mother_age := grandfather_age / 2
  let father_current_age := mother_age + father_age_difference
  let years_until_25 := 25 - rachel_current_age
  father_current_age + years_until_25

/-- Theorem stating that Rachel's father will be 60 years old when Rachel is 25 -/
theorem rachels_father_age_at_25_is_60 :
  rachels_father_age_at_25 12 7 5 = 60 := by
  sorry

#eval rachels_father_age_at_25 12 7 5

end NUMINAMATH_CALUDE_rachels_father_age_at_25_is_60_l2537_253741


namespace NUMINAMATH_CALUDE_simplify_inverse_product_l2537_253785

theorem simplify_inverse_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (((1 : ℝ) / a) * ((1 : ℝ) / (b + c)))⁻¹ = a * (b + c) := by sorry

end NUMINAMATH_CALUDE_simplify_inverse_product_l2537_253785


namespace NUMINAMATH_CALUDE_water_height_in_aquarium_l2537_253760

/-- Proves that the height of water in an aquarium with given dimensions and volume of water is 10 cm. -/
theorem water_height_in_aquarium :
  let aquarium_length : ℝ := 50
  let aquarium_breadth : ℝ := 20
  let aquarium_height : ℝ := 40
  let water_volume : ℝ := 10000  -- 10 litres * 1000 cm³/litre
  let water_height : ℝ := water_volume / (aquarium_length * aquarium_breadth)
  water_height = 10 := by sorry

end NUMINAMATH_CALUDE_water_height_in_aquarium_l2537_253760


namespace NUMINAMATH_CALUDE_teddy_has_seven_dogs_l2537_253731

/-- Represents the number of dogs Teddy has -/
def teddy_dogs : ℕ := sorry

/-- Represents the number of cats Teddy has -/
def teddy_cats : ℕ := 8

/-- Represents the number of dogs Ben has -/
def ben_dogs : ℕ := teddy_dogs + 9

/-- Represents the number of cats Dave has -/
def dave_cats : ℕ := teddy_cats + 13

/-- Represents the number of dogs Dave has -/
def dave_dogs : ℕ := teddy_dogs - 5

/-- The total number of pets -/
def total_pets : ℕ := 54

theorem teddy_has_seven_dogs : 
  teddy_dogs = 7 ∧ 
  teddy_dogs + teddy_cats + ben_dogs + dave_cats + dave_dogs = total_pets :=
sorry

end NUMINAMATH_CALUDE_teddy_has_seven_dogs_l2537_253731


namespace NUMINAMATH_CALUDE_symmetry_condition_l2537_253798

def f (x a : ℝ) : ℝ := |x + 1| + |x - 1| + |x - a|

theorem symmetry_condition (a : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, f x a = f (2*k - x) a) ↔ a ∈ ({-3, 0, 3} : Set ℝ) := by
sorry

end NUMINAMATH_CALUDE_symmetry_condition_l2537_253798


namespace NUMINAMATH_CALUDE_square_perimeter_diagonal_ratio_l2537_253709

theorem square_perimeter_diagonal_ratio (P₁ P₂ d₁ d₂ : ℝ) :
  P₁ > 0 ∧ P₂ > 0 ∧ d₁ > 0 ∧ d₂ > 0 ∧ 
  (P₂ / P₁ = 11) ∧
  (P₁ = 4 * (d₁ / Real.sqrt 2)) ∧
  (P₂ = 4 * (d₂ / Real.sqrt 2)) →
  d₂ / d₁ = 11 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_diagonal_ratio_l2537_253709


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2537_253761

theorem fraction_sum_equality : 
  (3 : ℚ) / 5 + (2 : ℚ) / 3 + (1 + (1 : ℚ) / 15) = 2 + (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2537_253761


namespace NUMINAMATH_CALUDE_diophantine_equation_implication_l2537_253755

theorem diophantine_equation_implication 
  (a b : ℤ) 
  (ha : ¬∃ (n : ℤ), a = n^2) 
  (hb : ¬∃ (n : ℤ), b = n^2) 
  (h : ∃ (x y z w : ℤ), x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∨ w ≠ 0 ∧ x^2 - a*y^2 - b*z^2 + a*b*w^2 = 0) :
  ∃ (X Y Z : ℤ), X ≠ 0 ∨ Y ≠ 0 ∨ Z ≠ 0 ∧ X^2 - a*Y^2 - b*Z^2 = 0 :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_implication_l2537_253755


namespace NUMINAMATH_CALUDE_initial_amount_proof_l2537_253723

/-- 
Theorem: If an amount increases by 1/8th of itself each year for two years 
and results in 82265.625, then the initial amount was 65000.
-/
theorem initial_amount_proof (initial_amount : ℚ) : 
  (initial_amount * (9/8)^2 = 82265.625) → initial_amount = 65000 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l2537_253723


namespace NUMINAMATH_CALUDE_base3_to_base10_equiv_l2537_253705

/-- Converts a base 3 number to base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number -/
def base3Number : List Nat := [1, 2, 2, 0, 1]

theorem base3_to_base10_equiv : base3ToBase10 base3Number = 106 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_equiv_l2537_253705


namespace NUMINAMATH_CALUDE_problem_solution_l2537_253757

theorem problem_solution (x : ℝ) : (0.25 * x = 0.15 * 1500 - 30) → x = 780 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2537_253757


namespace NUMINAMATH_CALUDE_solve_for_m_l2537_253710

theorem solve_for_m (x m : ℝ) (h1 : 3 * x - 2 * m = 4) (h2 : x = 6) : m = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l2537_253710


namespace NUMINAMATH_CALUDE_white_to_black_stone_ratio_l2537_253799

theorem white_to_black_stone_ratio :
  ∀ (total_stones white_stones black_stones : ℕ),
    total_stones = 100 →
    white_stones = 60 →
    black_stones = total_stones - white_stones →
    white_stones > black_stones →
    (white_stones : ℚ) / (black_stones : ℚ) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_white_to_black_stone_ratio_l2537_253799


namespace NUMINAMATH_CALUDE_floor_of_3_999_l2537_253779

theorem floor_of_3_999 : ⌊(3.999 : ℝ)⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_floor_of_3_999_l2537_253779


namespace NUMINAMATH_CALUDE_angle_ABH_measure_l2537_253778

/-- A regular octagon is a polygon with 8 equal sides and 8 equal angles. -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The measure of angle ABH in a regular octagon ABCDEFGH. -/
def angle_ABH (octagon : RegularOctagon) : ℝ := sorry

/-- Theorem: The measure of angle ABH in a regular octagon is 22.5 degrees. -/
theorem angle_ABH_measure (octagon : RegularOctagon) : 
  angle_ABH octagon = 22.5 := by sorry

end NUMINAMATH_CALUDE_angle_ABH_measure_l2537_253778


namespace NUMINAMATH_CALUDE_equal_chicken_wing_distribution_l2537_253792

theorem equal_chicken_wing_distribution 
  (num_friends : ℕ)
  (pre_cooked_wings : ℕ)
  (additional_wings : ℕ)
  (h1 : num_friends = 4)
  (h2 : pre_cooked_wings = 9)
  (h3 : additional_wings = 7) :
  (pre_cooked_wings + additional_wings) / num_friends = 4 :=
by sorry

end NUMINAMATH_CALUDE_equal_chicken_wing_distribution_l2537_253792


namespace NUMINAMATH_CALUDE_water_left_over_l2537_253719

theorem water_left_over (players : ℕ) (initial_water : ℕ) (water_per_player : ℕ) (spilled_water : ℕ) :
  players = 30 →
  initial_water = 8000 →
  water_per_player = 200 →
  spilled_water = 250 →
  initial_water - (players * water_per_player + spilled_water) = 1750 :=
by
  sorry

end NUMINAMATH_CALUDE_water_left_over_l2537_253719


namespace NUMINAMATH_CALUDE_no_real_roots_for_polynomial_l2537_253763

theorem no_real_roots_for_polynomial (a : ℝ) : 
  ¬∃ x : ℝ, x^4 + a^2*x^3 - 2*x^2 + a*x + 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_for_polynomial_l2537_253763


namespace NUMINAMATH_CALUDE_midpoint_ratio_range_l2537_253747

-- Define the lines and points
def line1 (x y : ℝ) : Prop := x + 3 * y - 2 = 0
def line2 (x y : ℝ) : Prop := x + 3 * y + 6 = 0

-- Define the midpoint condition
def is_midpoint (x₀ y₀ x_p y_p x_q y_q : ℝ) : Prop :=
  x₀ = (x_p + x_q) / 2 ∧ y₀ = (y_p + y_q) / 2

-- State the theorem
theorem midpoint_ratio_range (x₀ y₀ x_p y_p x_q y_q : ℝ) :
  line1 x_p y_p →
  line2 x_q y_q →
  is_midpoint x₀ y₀ x_p y_p x_q y_q →
  y₀ < x₀ + 2 →
  (y₀ / x₀ < -1/3 ∨ y₀ / x₀ > 0) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_ratio_range_l2537_253747


namespace NUMINAMATH_CALUDE_income_growth_equation_correct_l2537_253758

/-- Represents the growth of per capita disposable income in China from 2020 to 2022 -/
def income_growth (x : ℝ) : Prop :=
  let income_2020 : ℝ := 3.2  -- in ten thousand yuan
  let income_2022 : ℝ := 3.7  -- in ten thousand yuan
  let years : ℕ := 2
  income_2020 * (1 + x) ^ years = income_2022

/-- Theorem stating that the equation correctly represents the income growth -/
theorem income_growth_equation_correct :
  ∃ x : ℝ, income_growth x := by
  sorry

end NUMINAMATH_CALUDE_income_growth_equation_correct_l2537_253758


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2537_253728

theorem cubic_root_sum_cubes (p q r : ℝ) : 
  (p^3 - 2*p^2 + 3*p - 1 = 0) ∧ 
  (q^3 - 2*q^2 + 3*q - 1 = 0) ∧ 
  (r^3 - 2*r^2 + 3*r - 1 = 0) →
  p^3 + q^3 + r^3 = -7 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2537_253728


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l2537_253791

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem reflection_across_x_axis :
  let A : Point := { x := 1, y := -2 }
  reflectAcrossXAxis A = { x := 1, y := 2 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l2537_253791


namespace NUMINAMATH_CALUDE_economics_test_absentees_l2537_253748

theorem economics_test_absentees (total_students : ℕ) (q1_correct : ℕ) (q2_correct : ℕ) (both_correct : ℕ) 
  (h1 : total_students = 29)
  (h2 : q1_correct = 19)
  (h3 : q2_correct = 24)
  (h4 : both_correct = 19) :
  total_students - (q1_correct + q2_correct - both_correct) = 5 := by
  sorry


end NUMINAMATH_CALUDE_economics_test_absentees_l2537_253748


namespace NUMINAMATH_CALUDE_sin_15_times_sin_75_l2537_253797

theorem sin_15_times_sin_75 : Real.sin (15 * π / 180) * Real.sin (75 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_times_sin_75_l2537_253797


namespace NUMINAMATH_CALUDE_always_quadratic_in_x_l2537_253740

/-- A quadratic equation in x is of the form ax² + bx + c = 0 where a ≠ 0 -/
def is_quadratic_in_x (a b c : ℝ) : Prop := a ≠ 0

/-- The equation (m²+1)x² - mx - 3 = 0 is quadratic in x for all real m -/
theorem always_quadratic_in_x (m : ℝ) : 
  is_quadratic_in_x (m^2 + 1) (-m) (-3) := by sorry

end NUMINAMATH_CALUDE_always_quadratic_in_x_l2537_253740


namespace NUMINAMATH_CALUDE_longest_pole_in_room_l2537_253743

theorem longest_pole_in_room (length width height : ℝ) 
  (h_length : length = 12)
  (h_width : width = 8)
  (h_height : height = 9) :
  Real.sqrt (length^2 + width^2 + height^2) = 17 := by
  sorry

end NUMINAMATH_CALUDE_longest_pole_in_room_l2537_253743


namespace NUMINAMATH_CALUDE_polynomial_roots_l2537_253737

/-- The polynomial x^3 - 7x^2 + 11x + 13 -/
def f (x : ℝ) := x^3 - 7*x^2 + 11*x + 13

/-- The set of roots of the polynomial -/
def roots : Set ℝ := {2, 6, -1}

theorem polynomial_roots :
  (∀ x ∈ roots, f x = 0) ∧
  (∀ x : ℝ, f x = 0 → x ∈ roots) :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2537_253737


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2011_l2537_253711

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

/-- The problem statement -/
theorem arithmetic_sequence_2011 :
  arithmeticSequenceTerm 1 3 671 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2011_l2537_253711


namespace NUMINAMATH_CALUDE_nods_per_kilometer_l2537_253707

/-- Given the relationships between winks, nods, leaps, and kilometers,
    prove that the number of nods in one kilometer is equal to qts / (pru) -/
theorem nods_per_kilometer
  (p q r s t u : ℚ)
  (h1 : p * 1 = q)  -- p winks equal q nods
  (h2 : r * 1 = s)  -- r leaps equal s winks
  (h3 : t * 1 = u)  -- t leaps are equivalent to u kilometers
  : 1 = q * t * s / (p * r * u) :=
sorry

end NUMINAMATH_CALUDE_nods_per_kilometer_l2537_253707


namespace NUMINAMATH_CALUDE_real_solutions_quadratic_l2537_253738

theorem real_solutions_quadratic (x : ℝ) :
  (∃ y : ℝ, 9 * y^2 - 3 * x * y + x + 8 = 0) ↔ x ≤ -4 ∨ x ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_real_solutions_quadratic_l2537_253738


namespace NUMINAMATH_CALUDE_main_theorem_l2537_253706

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The sequence a_n -/
noncomputable def a : Sequence := sorry

/-- The sequence b_n -/
noncomputable def b (n : ℕ) : ℝ := a n + n

/-- The sum of the first n terms of b_n -/
noncomputable def S (n : ℕ) : ℝ := sorry

/-- Main theorem -/
theorem main_theorem :
  (∀ n : ℕ, a n < 0) ∧
  (∀ n : ℕ, a (n + 1) = 2/3 * a n) ∧
  (a 2 * a 5 = 8/27) →
  (∀ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = 2/3) ∧
  (∀ n : ℕ, a n = -(2/3)^(n-1)) ∧
  (∀ n : ℕ, S n = (n^2 + n + 6)/2 - 3 * (2/3)^n) :=
by sorry

end NUMINAMATH_CALUDE_main_theorem_l2537_253706


namespace NUMINAMATH_CALUDE_triangle_inequality_l2537_253770

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) :
  (a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥
   (b + c - a) / a + (c + a - b) / b + (a + b - c) / c) ∧
  ((b + c - a) / a + (c + a - b) / b + (a + b - c) / c ≥ 3) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2537_253770


namespace NUMINAMATH_CALUDE_irrational_and_rational_numbers_l2537_253701

theorem irrational_and_rational_numbers : ∃ (x : ℝ), 
  (Irrational (-Real.sqrt 5)) ∧ 
  (¬ Irrational (Real.sqrt 4)) ∧ 
  (¬ Irrational (2 / 3)) ∧ 
  (¬ Irrational 0) := by
  sorry

end NUMINAMATH_CALUDE_irrational_and_rational_numbers_l2537_253701


namespace NUMINAMATH_CALUDE_max_value_of_linear_combination_l2537_253752

theorem max_value_of_linear_combination (x y : ℝ) :
  x^2 + y^2 = 18*x + 8*y + 10 →
  4*x + 3*y ≤ 74 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_linear_combination_l2537_253752


namespace NUMINAMATH_CALUDE_angles_with_same_terminal_side_l2537_253780

def same_terminal_side (θ : ℝ) : Set ℝ :=
  {α | ∃ k : ℤ, α = k * 360 + θ}

def angle_range : Set ℝ :=
  {β | -360 ≤ β ∧ β < 720}

theorem angles_with_same_terminal_side :
  (same_terminal_side 60 ∩ angle_range) ∪ (same_terminal_side (-21) ∩ angle_range) =
  {-300, 60, 420, -21, 339, 699} := by
  sorry

end NUMINAMATH_CALUDE_angles_with_same_terminal_side_l2537_253780


namespace NUMINAMATH_CALUDE_rectangle_cutting_l2537_253742

theorem rectangle_cutting (large_width large_height small_width small_height : ℝ) 
  (hw : large_width = 50)
  (hh : large_height = 90)
  (hsw : small_width = 1)
  (hsh : small_height = 10 * Real.sqrt 2) :
  ⌊(large_width * large_height) / (small_width * small_height)⌋ = 318 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_cutting_l2537_253742


namespace NUMINAMATH_CALUDE_range_start_divisible_by_eleven_l2537_253795

theorem range_start_divisible_by_eleven : ∃ (start : ℕ), 
  (start ≤ 79) ∧ 
  (∃ (a b c d : ℕ), 
    (start = 11 * a) ∧ 
    (start + 11 = 11 * b) ∧ 
    (start + 22 = 11 * c) ∧ 
    (start + 33 = 11 * d) ∧ 
    (start + 33 ≤ 79) ∧
    (start + 44 > 79)) ∧
  (start = 44) := by
sorry

end NUMINAMATH_CALUDE_range_start_divisible_by_eleven_l2537_253795


namespace NUMINAMATH_CALUDE_ice_cream_difference_l2537_253744

-- Define the number of scoops for Oli and Victoria
def oli_scoops : ℕ := 4
def victoria_scoops : ℕ := 2 * oli_scoops

-- Theorem statement
theorem ice_cream_difference : victoria_scoops - oli_scoops = 4 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_difference_l2537_253744


namespace NUMINAMATH_CALUDE_largest_three_digit_product_l2537_253774

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_three_digit_product (n p q : ℕ) : 
  n ≥ 100 ∧ n < 1000 ∧
  is_prime p ∧ p < 10 ∧
  q < 10 ∧
  n = p * q * (10 * p + q) ∧
  p ≠ q ∧ p ≠ (10 * p + q) ∧ q ≠ (10 * p + q) →
  n ≤ 777 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_product_l2537_253774


namespace NUMINAMATH_CALUDE_friends_at_reception_l2537_253729

/-- Calculates the number of friends attending a wedding reception --/
theorem friends_at_reception (total_guests : ℕ) (bride_couples : ℕ) (groom_couples : ℕ) : 
  total_guests - 2 * (bride_couples + groom_couples) = 100 :=
by
  sorry

#check friends_at_reception 180 20 20

end NUMINAMATH_CALUDE_friends_at_reception_l2537_253729


namespace NUMINAMATH_CALUDE_probability_of_even_product_l2537_253715

-- Define the set of chips in each box
def chips : Set ℕ := {1, 2, 4}

-- Define the function to check if a number is even
def isEven (n : ℕ) : Prop := n % 2 = 0

-- Define the total number of possible outcomes
def totalOutcomes : ℕ := 27

-- Define the number of favorable outcomes (even products)
def favorableOutcomes : ℕ := 26

-- Theorem statement
theorem probability_of_even_product :
  (favorableOutcomes : ℚ) / totalOutcomes = 26 / 27 := by sorry

end NUMINAMATH_CALUDE_probability_of_even_product_l2537_253715


namespace NUMINAMATH_CALUDE_my_matrix_is_projection_l2537_253708

def projection_matrix (A : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  A * A = A

def my_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![9/25, 18/45],
    ![12/25, 27/45]]

theorem my_matrix_is_projection : projection_matrix my_matrix := by
  sorry

end NUMINAMATH_CALUDE_my_matrix_is_projection_l2537_253708


namespace NUMINAMATH_CALUDE_triangle_properties_l2537_253784

noncomputable section

-- Define the triangle ABC
def triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the theorem
theorem triangle_properties
  (a b c : ℝ)
  (h_triangle : triangle a b c)
  (h_angle_A : Real.cos (π/4) = b^2 + c^2 - a^2 / (2*b*c))
  (h_sides : b^2 - a^2 = (1/2) * c^2)
  (h_area : (1/2) * a * b * Real.sin (π/4) = 3) :
  Real.tan (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) = 2 ∧
  b = 3 ∧
  2 * π * ((a / (2 * Real.sin (π/4))) : ℝ) = Real.sqrt 10 * π :=
sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l2537_253784


namespace NUMINAMATH_CALUDE_inequality_solutions_l2537_253753

theorem inequality_solutions :
  (∀ x : ℝ, x^2 - 5*x + 5 > 0 ↔ (x > (5 + Real.sqrt 5) / 2 ∨ x < (5 - Real.sqrt 5) / 2)) ∧
  (∀ x : ℝ, -2*x^2 + x - 3 < 0) := by
sorry

end NUMINAMATH_CALUDE_inequality_solutions_l2537_253753


namespace NUMINAMATH_CALUDE_no_four_distinct_real_roots_l2537_253768

theorem no_four_distinct_real_roots (a b : ℝ) : 
  ¬ (∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄) ∧
    (r₁^4 - 4*r₁^3 + 6*r₁^2 + a*r₁ + b = 0) ∧
    (r₂^4 - 4*r₂^3 + 6*r₂^2 + a*r₂ + b = 0) ∧
    (r₃^4 - 4*r₃^3 + 6*r₃^2 + a*r₃ + b = 0) ∧
    (r₄^4 - 4*r₄^3 + 6*r₄^2 + a*r₄ + b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_no_four_distinct_real_roots_l2537_253768


namespace NUMINAMATH_CALUDE_van_distance_proof_l2537_253790

theorem van_distance_proof (initial_time : ℝ) (new_speed : ℝ) :
  initial_time = 6 →
  new_speed = 32 →
  (initial_time * 3 / 2) * new_speed = 288 := by
  sorry

end NUMINAMATH_CALUDE_van_distance_proof_l2537_253790


namespace NUMINAMATH_CALUDE_extracurricular_teams_problem_l2537_253773

theorem extracurricular_teams_problem (total_activities : ℕ) 
  (initial_ratio_tt : ℕ) (initial_ratio_bb : ℕ) 
  (new_ratio_tt : ℕ) (new_ratio_bb : ℕ) 
  (transfer : ℕ) :
  total_activities = 38 →
  initial_ratio_tt = 7 →
  initial_ratio_bb = 3 →
  new_ratio_tt = 3 →
  new_ratio_bb = 2 →
  transfer = 8 →
  ∃ (tt_original bb_original : ℕ),
    tt_original * initial_ratio_bb = bb_original * initial_ratio_tt ∧
    (tt_original - transfer) * new_ratio_bb = (bb_original + transfer) * new_ratio_tt ∧
    tt_original = 35 ∧
    bb_original = 15 := by
  sorry

end NUMINAMATH_CALUDE_extracurricular_teams_problem_l2537_253773


namespace NUMINAMATH_CALUDE_average_increase_is_three_l2537_253767

/-- Represents a batsman's statistics -/
structure Batsman where
  total_runs : ℕ
  innings : ℕ
  average : ℚ

/-- Calculates the new average after an inning -/
def new_average (b : Batsman) (runs : ℕ) : ℚ :=
  (b.total_runs + runs) / (b.innings + 1)

/-- Theorem: The increase in average is 3 for the given conditions -/
theorem average_increase_is_three (b : Batsman) (h1 : b.innings = 16) 
    (h2 : new_average b 92 = 44) : 
    new_average b 92 - b.average = 3 := by
  sorry

#check average_increase_is_three

end NUMINAMATH_CALUDE_average_increase_is_three_l2537_253767


namespace NUMINAMATH_CALUDE_cake_and_muffin_buyers_l2537_253721

theorem cake_and_muffin_buyers (total : ℕ) (cake : ℕ) (muffin : ℕ) (neither_prob : ℚ) 
  (h_total : total = 100)
  (h_cake : cake = 50)
  (h_muffin : muffin = 40)
  (h_neither_prob : neither_prob = 1/4) :
  ∃ both : ℕ, 
    both = cake + muffin - (total * (1 - neither_prob)) ∧
    both = 15 := by
  sorry

end NUMINAMATH_CALUDE_cake_and_muffin_buyers_l2537_253721


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l2537_253745

theorem binary_to_octal_conversion : 
  (1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 
  (1 * 8^2 + 1 * 8^1 + 5 * 8^0) :=
by sorry

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l2537_253745


namespace NUMINAMATH_CALUDE_B_eq_A_pow2_l2537_253769

def A : ℕ → ℚ
  | 0 => 1
  | n + 1 => (A n + 2) / (A n + 1)

def B : ℕ → ℚ
  | 0 => 1
  | n + 1 => (B n^2 + 2) / (2 * B n)

theorem B_eq_A_pow2 (n : ℕ) : B (n + 1) = A (2^n) := by
  sorry

end NUMINAMATH_CALUDE_B_eq_A_pow2_l2537_253769


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2537_253775

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {2, 4, 5}
def B : Set Nat := {1, 3, 5, 7}

theorem intersection_complement_equality : A ∩ (U \ B) = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2537_253775


namespace NUMINAMATH_CALUDE_smallest_k_for_product_sign_change_l2537_253716

def sequence_a (n : ℕ) : ℚ :=
  15 - 2/3 * (n - 1)

theorem smallest_k_for_product_sign_change :
  let a := sequence_a
  (∀ n : ℕ, n ≥ 1 → 3 * a (n + 1) = 3 * a n - 2) →
  (∃ k : ℕ, k > 0 ∧ a k * a (k + 1) < 0) →
  (∀ j : ℕ, 0 < j → j < 23 → a j * a (j + 1) ≥ 0) →
  a 23 * a 24 < 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_product_sign_change_l2537_253716


namespace NUMINAMATH_CALUDE_a_86_in_geometric_subsequence_l2537_253751

/-- Represents an arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence :=
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_nonzero : d ≠ 0)
  (h_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d)

/-- Represents a subsequence of an arithmetic sequence that forms a geometric sequence -/
structure GeometricSubsequence (as : ArithmeticSequence) :=
  (k : ℕ → ℕ)
  (h_geometric : ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, as.a (k (n + 1)) = r * as.a (k n))
  (h_k1 : k 1 = 1)
  (h_k2 : k 2 = 2)
  (h_k3 : k 3 = 6)

/-- The main theorem stating that a_86 is in the geometric subsequence -/
theorem a_86_in_geometric_subsequence (as : ArithmeticSequence) (gs : GeometricSubsequence as) :
  ∃ n : ℕ, gs.k n = 86 :=
sorry

end NUMINAMATH_CALUDE_a_86_in_geometric_subsequence_l2537_253751


namespace NUMINAMATH_CALUDE_container_volume_ratio_l2537_253739

theorem container_volume_ratio : 
  ∀ (A B : ℝ),  -- A and B are the volumes of the first and second containers
  A > 0 → B > 0 →  -- Both volumes are positive
  (4/5 * A - 1/5 * A) = 2/3 * B →  -- Amount poured equals 2/3 of second container
  A / B = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l2537_253739


namespace NUMINAMATH_CALUDE_star_difference_l2537_253735

-- Define the ⭐ operation
def star (x y : ℝ) : ℝ := x^2 * y - 3 * x + y

-- Theorem statement
theorem star_difference : star 3 5 - star 5 3 = -22 := by
  sorry

end NUMINAMATH_CALUDE_star_difference_l2537_253735


namespace NUMINAMATH_CALUDE_max_min_f_l2537_253789

-- Define the function f
def f (x : ℝ) : ℝ := 6 - 12 * x + x^3

-- Define the interval
def I : Set ℝ := {x | -1/3 ≤ x ∧ x ≤ 1}

-- Statement of the theorem
theorem max_min_f :
  ∃ (max min : ℝ),
    (∀ x ∈ I, f x ≤ max) ∧
    (∃ x ∈ I, f x = max) ∧
    (∀ x ∈ I, min ≤ f x) ∧
    (∃ x ∈ I, f x = min) ∧
    max = 27 ∧
    min = -5 := by
  sorry

end NUMINAMATH_CALUDE_max_min_f_l2537_253789


namespace NUMINAMATH_CALUDE_women_fraction_in_room_l2537_253793

theorem women_fraction_in_room (total_people : ℕ) (married_fraction : ℚ) 
  (max_unmarried_women : ℕ) (h1 : total_people = 80) (h2 : married_fraction = 1/2) 
  (h3 : max_unmarried_women = 32) : 
  (max_unmarried_women + (married_fraction * total_people / 2)) / total_people = 1/2 :=
sorry

end NUMINAMATH_CALUDE_women_fraction_in_room_l2537_253793
