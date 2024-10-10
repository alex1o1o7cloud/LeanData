import Mathlib

namespace initial_money_calculation_l2384_238414

/-- Calculates the initial amount of money given the cost of bread, peanut butter, and the amount left over --/
theorem initial_money_calculation (bread_cost : ℝ) (bread_quantity : ℕ) (peanut_butter_cost : ℝ) (money_left : ℝ) : 
  bread_cost = 2.25 →
  bread_quantity = 3 →
  peanut_butter_cost = 2 →
  money_left = 5.25 →
  bread_cost * (bread_quantity : ℝ) + peanut_butter_cost + money_left = 14 :=
by sorry

end initial_money_calculation_l2384_238414


namespace game_tie_fraction_l2384_238473

theorem game_tie_fraction (mark_wins jane_wins : ℚ) 
  (h1 : mark_wins = 5 / 12)
  (h2 : jane_wins = 1 / 4) : 
  1 - (mark_wins + jane_wins) = 1 / 3 := by
sorry

end game_tie_fraction_l2384_238473


namespace units_digit_sum_base8_l2384_238493

/-- The units digit of a number in base 8 -/
def units_digit_base8 (n : ℕ) : ℕ := n % 8

/-- Addition in base 8 -/
def add_base8 (a b : ℕ) : ℕ := (a + b) % 8

theorem units_digit_sum_base8 :
  units_digit_base8 (add_base8 67 54) = 3 := by
  sorry

end units_digit_sum_base8_l2384_238493


namespace prism_volume_l2384_238446

/-- The volume of a right rectangular prism with face areas 40, 50, and 100 square centimeters -/
theorem prism_volume (x y z : ℝ) (hxy : x * y = 40) (hxz : x * z = 50) (hyz : y * z = 100) :
  x * y * z = 100 * Real.sqrt 2 := by
  sorry

end prism_volume_l2384_238446


namespace grandpa_water_distribution_l2384_238424

/-- The number of water bottles Grandpa has -/
def num_bottles : ℕ := 12

/-- The volume of each water bottle in liters -/
def bottle_volume : ℚ := 3

/-- The volume of water to be distributed to each student in liters -/
def student_share : ℚ := 3/4

/-- The number of students Grandpa can share water with -/
def num_students : ℕ := 48

theorem grandpa_water_distribution :
  (↑num_bottles * bottle_volume) / student_share = num_students := by
  sorry

end grandpa_water_distribution_l2384_238424


namespace comparison_inequality_l2384_238427

theorem comparison_inequality (a b : ℝ) (h1 : a ≠ b) (h2 : a < 0) :
  a < 2 * b - b^2 / a := by
sorry

end comparison_inequality_l2384_238427


namespace point_division_l2384_238436

/-- Given a line segment AB and a point P on AB such that AP:PB = 3:5,
    prove that P can be expressed as a linear combination of A and B with coefficients 5/8 and 3/8 respectively. -/
theorem point_division (A B P : ℝ × ℝ) : 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B) →  -- P is on line segment AB
  (dist A P) / (dist P B) = 3 / 5 →                     -- AP:PB = 3:5
  P = (5/8) • A + (3/8) • B :=                          -- P = (5/8)A + (3/8)B
by sorry

end point_division_l2384_238436


namespace triangle_third_side_prime_l2384_238412

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def valid_third_side (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_prime (a b : ℕ) (ha : a = 7) (hb : b = 11) :
  ∃ (c : ℕ), is_prime c ∧ valid_third_side a b c ↔ 
  c = 5 ∨ c = 7 ∨ c = 11 ∨ c = 13 ∨ c = 17 :=
sorry

end triangle_third_side_prime_l2384_238412


namespace lcm_of_ratio_numbers_l2384_238456

theorem lcm_of_ratio_numbers (a b : ℕ) (h1 : a = 21) (h2 : 4 * a = 3 * b) : 
  Nat.lcm a b = 84 := by sorry

end lcm_of_ratio_numbers_l2384_238456


namespace cosine_sine_relation_l2384_238419

theorem cosine_sine_relation (x : ℝ) :
  2 * Real.cos x + 3 * Real.sin x = 4 →
  Real.cos x = 8 / 13 ∧ Real.sin x = 12 / 13 →
  3 * Real.cos x - 2 * Real.sin x = 0 := by
sorry

end cosine_sine_relation_l2384_238419


namespace garden_flowers_l2384_238477

theorem garden_flowers (red_flowers : ℕ) (additional_red : ℕ) (white_flowers : ℕ) :
  red_flowers = 347 →
  additional_red = 208 →
  white_flowers = red_flowers + additional_red →
  white_flowers = 555 := by
  sorry

end garden_flowers_l2384_238477


namespace intersection_area_theorem_l2384_238466

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  center : Point
  width : ℝ
  height : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a right triangle in 2D space -/
structure RightTriangle where
  vertex : Point
  leg1 : ℝ
  leg2 : ℝ

/-- Calculate the area of intersection between a rectangle, circle, and right triangle -/
def areaOfIntersection (rect : Rectangle) (circ : Circle) (tri : RightTriangle) : ℝ :=
  sorry

theorem intersection_area_theorem (rect : Rectangle) (circ : Circle) (tri : RightTriangle) :
  rect.width = 10 →
  rect.height = 4 →
  circ.radius = 4 →
  rect.center = circ.center →
  tri.leg1 = 3 →
  tri.leg2 = 3 →
  -- Assuming the triangle is positioned correctly
  areaOfIntersection rect circ tri = 4.5 := by
  sorry

end intersection_area_theorem_l2384_238466


namespace difference_of_squares_75_35_l2384_238497

theorem difference_of_squares_75_35 : 75^2 - 35^2 = 4400 := by
  sorry

end difference_of_squares_75_35_l2384_238497


namespace symmetric_periodic_function_max_period_l2384_238471

/-- A function with symmetry around x=1 and x=8, and a periodic property -/
def SymmetricPeriodicFunction (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T > 0 ∧
  (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ x : ℝ, f (1 + x) = f (1 - x)) ∧
  (∀ x : ℝ, f (8 + x) = f (8 - x)) ∧
  ∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T'

theorem symmetric_periodic_function_max_period :
  ∀ f : ℝ → ℝ, SymmetricPeriodicFunction f →
  ∃ T : ℝ, T > 0 ∧ SymmetricPeriodicFunction f ∧ T = 14 ∧
  ∀ T' : ℝ, T' > 0 → SymmetricPeriodicFunction f → T' ≤ T :=
sorry

end symmetric_periodic_function_max_period_l2384_238471


namespace problem_statement_l2384_238496

theorem problem_statement (a b c : ℝ) (h : a + b = ab ∧ ab = c) :
  (c ≠ 0 → (2*a - 3*a*b + 2*b) / (5*a + 7*a*b + 5*b) = -1/12) ∧
  (a = 3 → b + c = 6) ∧
  (c ≠ 0 → (1-a)*(1-b) = 1/a + 1/b) ∧
  (c = 4 → a^2 + b^2 = 8) := by
sorry

end problem_statement_l2384_238496


namespace playlist_duration_l2384_238499

/-- Given a playlist with three songs of durations 3, 2, and 3 minutes respectively,
    prove that listening to this playlist 5 times takes 40 minutes. -/
theorem playlist_duration (song1 song2 song3 : ℕ) (repetitions : ℕ) :
  song1 = 3 ∧ song2 = 2 ∧ song3 = 3 ∧ repetitions = 5 →
  (song1 + song2 + song3) * repetitions = 40 :=
by sorry

end playlist_duration_l2384_238499


namespace expression_value_l2384_238415

theorem expression_value (x : ℝ) : x = 2 → 3 * x^2 - 4 * x + 2 = 6 := by
  sorry

end expression_value_l2384_238415


namespace thief_catch_time_l2384_238465

/-- The time it takes for the passenger to catch the thief -/
def catchUpTime (thief_speed : ℝ) (passenger_speed : ℝ) (bus_speed : ℝ) (stop_time : ℝ) : ℝ :=
  stop_time

theorem thief_catch_time :
  ∀ (thief_speed : ℝ),
    thief_speed > 0 →
    let passenger_speed := 2 * thief_speed
    let bus_speed := 10 * thief_speed
    let stop_time := 40
    catchUpTime thief_speed passenger_speed bus_speed stop_time = 40 :=
by
  sorry

#check thief_catch_time

end thief_catch_time_l2384_238465


namespace probability_of_losing_is_one_third_l2384_238428

/-- A game where a single standard die is rolled once -/
structure DieGame where
  /-- The set of all possible outcomes when rolling a standard die -/
  outcomes : Finset Nat
  /-- The set of losing outcomes -/
  losing_outcomes : Finset Nat
  /-- Assumption that outcomes are the numbers 1 to 6 -/
  outcomes_def : outcomes = Finset.range 6
  /-- Assumption that losing outcomes are 5 and 6 -/
  losing_def : losing_outcomes = {5, 6}

/-- The probability of losing in the die game -/
def probability_of_losing (game : DieGame) : ℚ :=
  (game.losing_outcomes.card : ℚ) / (game.outcomes.card : ℚ)

/-- Theorem stating that the probability of losing is 1/3 -/
theorem probability_of_losing_is_one_third (game : DieGame) :
    probability_of_losing game = 1 / 3 := by
  sorry

end probability_of_losing_is_one_third_l2384_238428


namespace sqrt_eight_combinable_with_sqrt_two_l2384_238480

theorem sqrt_eight_combinable_with_sqrt_two :
  ∃ (n : ℤ), Real.sqrt 8 = n * Real.sqrt 2 :=
sorry

end sqrt_eight_combinable_with_sqrt_two_l2384_238480


namespace moon_permutations_l2384_238445

-- Define the word as a list of characters
def moon : List Char := ['M', 'O', 'O', 'N']

-- Define the number of unique permutations
def uniquePermutations (word : List Char) : ℕ :=
  Nat.factorial word.length / (Nat.factorial (word.count 'O'))

-- Theorem statement
theorem moon_permutations :
  uniquePermutations moon = 12 := by
  sorry

end moon_permutations_l2384_238445


namespace trajectory_of_G_no_perpendicular_bisector_l2384_238454

-- Define the circle M
def circle_M (m n r : ℝ) (x y : ℝ) : Prop :=
  (x - m)^2 + (y - n)^2 = r^2

-- Define point N
def point_N : ℝ × ℝ := (1, 0)

-- Define the conditions for points P, Q, and G
def point_conditions (m n r : ℝ) (P Q G : ℝ × ℝ) : Prop :=
  circle_M m n r P.1 P.2 ∧
  (∃ t : ℝ, Q = point_N + t • (P - point_N)) ∧
  (∃ s : ℝ, G = (m, n) + s • (P - (m, n))) ∧
  P - point_N = 2 • (Q - point_N) ∧
  (G - Q) • (P - point_N) = 0

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Theorem 1: Trajectory of G is the ellipse C
theorem trajectory_of_G (P Q G : ℝ × ℝ) :
  point_conditions (-1) 0 4 P Q G →
  trajectory_C G.1 G.2 :=
sorry

-- Theorem 2: No positive m, n, r exist such that MN perpendicularly bisects AB
theorem no_perpendicular_bisector :
  ¬ ∃ (m n r : ℝ) (A B : ℝ × ℝ),
    m > 0 ∧ n > 0 ∧ r > 0 ∧
    circle_M m n r A.1 A.2 ∧
    circle_M m n r B.1 B.2 ∧
    trajectory_C A.1 A.2 ∧
    trajectory_C B.1 B.2 ∧
    A ≠ B ∧
    (∃ (t : ℝ), (A + B) / 2 = point_N + t • ((m, n) - point_N)) ∧
    ((m, n) - point_N) • (B - A) = 0 :=
sorry

end trajectory_of_G_no_perpendicular_bisector_l2384_238454


namespace reading_homework_pages_isabel_homework_l2384_238491

theorem reading_homework_pages (math_pages : ℕ) (problems_per_page : ℕ) (total_problems : ℕ) : ℕ :=
  let reading_pages := (total_problems - math_pages * problems_per_page) / problems_per_page
  reading_pages

theorem isabel_homework :
  reading_homework_pages 2 5 30 = 4 := by
  sorry

end reading_homework_pages_isabel_homework_l2384_238491


namespace cos_three_pi_four_plus_two_alpha_l2384_238400

theorem cos_three_pi_four_plus_two_alpha (α : ℝ) 
  (h : Real.cos (π / 8 - α) = 1 / 6) : 
  Real.cos (3 * π / 4 + 2 * α) = 17 / 18 := by
  sorry

end cos_three_pi_four_plus_two_alpha_l2384_238400


namespace fraction_not_on_time_l2384_238417

/-- Represents the fraction of attendees who are male -/
def male_fraction : ℚ := 3/5

/-- Represents the fraction of male attendees who arrived on time -/
def male_on_time : ℚ := 7/8

/-- Represents the fraction of female attendees who arrived on time -/
def female_on_time : ℚ := 4/5

/-- Theorem stating that the fraction of attendees who did not arrive on time is 3/20 -/
theorem fraction_not_on_time : 
  1 - (male_fraction * male_on_time + (1 - male_fraction) * female_on_time) = 3/20 := by
  sorry

end fraction_not_on_time_l2384_238417


namespace problem_part_1_problem_part_2_l2384_238438

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a
def g (x : ℝ) : ℝ := |x + 1| + |x - 2|

theorem problem_part_1 :
  {x : ℝ | f (-4) x ≥ g x} = {x : ℝ | x ≤ -1 - Real.sqrt 6 ∨ x ≥ 3} := by sorry

theorem problem_part_2 (a : ℝ) :
  (∀ x ∈ Set.Icc 0 3, f a x ≤ g x) → a ≤ -4 := by sorry

end problem_part_1_problem_part_2_l2384_238438


namespace volleyball_starters_count_l2384_238485

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

/-- The number of ways to choose 6 starters from 15 players, with at least one of 3 specific players -/
def volleyball_starters : ℕ :=
  binomial 15 6 - binomial 12 6

theorem volleyball_starters_count : volleyball_starters = 4081 := by
  sorry

end volleyball_starters_count_l2384_238485


namespace min_value_expression_l2384_238431

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 48) :
  x^2 + 6*x*y + 9*y^2 + 4*z^2 ≥ 128 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 48 ∧ x₀^2 + 6*x₀*y₀ + 9*y₀^2 + 4*z₀^2 = 128 :=
by sorry

end min_value_expression_l2384_238431


namespace initial_number_proof_l2384_238476

theorem initial_number_proof : 
  ∃ x : ℝ, (3 * (2 * x + 9) = 81) ∧ (x = 9) := by
  sorry

end initial_number_proof_l2384_238476


namespace xy_value_l2384_238492

theorem xy_value (x y : ℝ) 
  (h1 : (8 : ℝ)^x / (4 : ℝ)^(x + y) = 16)
  (h2 : (16 : ℝ)^(x + y) / (4 : ℝ)^(7 * y) = 1024) : 
  x * y = 30 := by sorry

end xy_value_l2384_238492


namespace f_properties_l2384_238450

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 11

-- State the theorem
theorem f_properties :
  -- Part 1: Tangent line at x = 1 is y = 5
  (∀ y, (y - f 1 = 0 * (x - 1)) ↔ y = 5) ∧
  -- Part 2: Monotonicity intervals
  (∀ x, x < -1 → (deriv f) x > 0) ∧
  (∀ x, x > 1 → (deriv f) x > 0) ∧
  (∀ x, -1 < x ∧ x < 1 → (deriv f) x < 0) ∧
  -- Part 3: Maximum value on [-1, 1] is 17
  (∀ x, -1 ≤ x ∧ x ≤ 1 → f x ≤ 17) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 1 ∧ f x = 17) :=
by sorry

end f_properties_l2384_238450


namespace total_dogs_l2384_238470

theorem total_dogs (num_boxes : ℕ) (dogs_per_box : ℕ) (h1 : num_boxes = 7) (h2 : dogs_per_box = 4) :
  num_boxes * dogs_per_box = 28 := by
  sorry

end total_dogs_l2384_238470


namespace processing_400_parts_l2384_238420

/-- Linear regression function for processing time -/
def processingTime (x : ℝ) : ℝ := 0.2 * x + 3

/-- Theorem: Processing 400 parts takes 83 hours -/
theorem processing_400_parts : processingTime 400 = 83 := by
  sorry

end processing_400_parts_l2384_238420


namespace orange_buckets_problem_l2384_238416

/-- The problem of calculating the number of oranges and their total weight -/
theorem orange_buckets_problem :
  let bucket1 : ℝ := 22.5
  let bucket2 : ℝ := 2 * bucket1 + 3
  let bucket3 : ℝ := bucket2 - 11.5
  let bucket4 : ℝ := 1.5 * (bucket1 + bucket3)
  let weight1 : ℝ := 0.3
  let weight3 : ℝ := 0.4
  let weight4 : ℝ := 0.35
  let total_oranges : ℝ := bucket1 + bucket2 + bucket3 + bucket4
  let total_weight : ℝ := weight1 * bucket1 + weight3 * bucket3 + weight4 * bucket4
  total_oranges = 195.5 ∧ total_weight = 52.325 := by
  sorry


end orange_buckets_problem_l2384_238416


namespace unique_perfect_square_p_l2384_238447

/-- The polynomial p(x) = x^4 + 6x^3 + 11x^2 + 3x + 31 -/
def p (x : ℤ) : ℤ := x^4 + 6*x^3 + 11*x^2 + 3*x + 31

/-- A function that checks if a given integer is a perfect square -/
def is_perfect_square (n : ℤ) : Prop :=
  ∃ m : ℤ, n = m^2

/-- Theorem stating that there exists exactly one integer x for which p(x) is a perfect square -/
theorem unique_perfect_square_p :
  ∃! x : ℤ, is_perfect_square (p x) :=
sorry

end unique_perfect_square_p_l2384_238447


namespace exponent_calculations_l2384_238407

theorem exponent_calculations (x m : ℝ) (hx : x ≠ 0) (hm : m ≠ 0) :
  (x^7 / x^3 * x^4 = x^8) ∧ (m * m^3 + (-m^2)^3 / m^2 = 0) := by sorry

end exponent_calculations_l2384_238407


namespace negation_of_universal_proposition_l2384_238402

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ∈ [1, 2] → x^2 < 4) ↔ (∃ x : ℝ, x ∈ [1, 2] ∧ x^2 ≥ 4) :=
by sorry

end negation_of_universal_proposition_l2384_238402


namespace max_sin_angle_ellipse_l2384_238432

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

def is_focus (F : ℝ × ℝ) (a b : ℝ) : Prop :=
  F.1^2 + F.2^2 = a^2 - b^2 ∧ a > b ∧ a > 0 ∧ b > 0

def angle_sin (A B C : ℝ × ℝ) : ℝ := sorry

theorem max_sin_angle_ellipse :
  ∃ (a b : ℝ) (F₁ F₂ : ℝ × ℝ),
    a = 3 ∧ b = Real.sqrt 5 ∧
    is_focus F₁ a b ∧ is_focus F₂ a b ∧
    (∀ (P : ℝ × ℝ), ellipse P.1 P.2 →
      angle_sin F₁ P F₂ ≤ 4 * Real.sqrt 5 / 9) ∧
    (∃ (P : ℝ × ℝ), ellipse P.1 P.2 ∧
      angle_sin F₁ P F₂ = 4 * Real.sqrt 5 / 9) :=
sorry

end max_sin_angle_ellipse_l2384_238432


namespace contrapositive_equivalence_l2384_238474

-- Define the original proposition
def original_proposition (m : ℝ) : Prop :=
  m > 0 → ∃ x : ℝ, x^2 + x - m = 0

-- Define the contrapositive
def contrapositive (m : ℝ) : Prop :=
  (¬∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0

-- Theorem stating the equivalence of the contrapositive to the original proposition
theorem contrapositive_equivalence :
  ∀ m : ℝ, (¬original_proposition m) ↔ contrapositive m :=
by
  sorry


end contrapositive_equivalence_l2384_238474


namespace same_tangent_line_implies_b_value_l2384_238433

def f (x : ℝ) : ℝ := 2 * x^3 + 1
def g (b : ℝ) (x : ℝ) : ℝ := 3 * x^2 - b

theorem same_tangent_line_implies_b_value :
  ∀ b : ℝ, (∃ x₀ : ℝ, (deriv f x₀ = deriv (g b) x₀) ∧ 
    (f x₀ = g b x₀)) → (b = 0 ∨ b = -1) :=
by sorry

end same_tangent_line_implies_b_value_l2384_238433


namespace exponent_calculation_l2384_238481

theorem exponent_calculation : (64 : ℝ)^(1/4) * (16 : ℝ)^(3/8) = 8 := by
  have h1 : (64 : ℝ) = 2^6 := by sorry
  have h2 : (16 : ℝ) = 2^4 := by sorry
  sorry

end exponent_calculation_l2384_238481


namespace ladder_problem_l2384_238489

theorem ladder_problem (h1 h2 l : Real) 
  (hyp1 : h1 = 12)
  (hyp2 : h2 = 9)
  (hyp3 : l = 15) :
  Real.sqrt (l^2 - h1^2) + Real.sqrt (l^2 - h2^2) = 21 := by
  sorry

end ladder_problem_l2384_238489


namespace fourth_berry_count_l2384_238451

/-- A sequence of berry counts where the difference between consecutive terms increases by 2 -/
def BerrySequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) = (a (n + 1) - a n) + 2

theorem fourth_berry_count
  (a : ℕ → ℕ)
  (seq : BerrySequence a)
  (first : a 0 = 3)
  (second : a 1 = 4)
  (third : a 2 = 7)
  (fifth : a 4 = 19) :
  a 3 = 12 := by
  sorry

end fourth_berry_count_l2384_238451


namespace repeating_six_equals_two_thirds_l2384_238462

/-- The decimal representation of a real number with a single repeating digit. -/
def repeatingDecimal (digit : ℕ) : ℚ :=
  (digit : ℚ) / 9

/-- Prove that the repeating decimal 0.666... is equal to 2/3 -/
theorem repeating_six_equals_two_thirds :
  repeatingDecimal 6 = 2 / 3 := by
  sorry

end repeating_six_equals_two_thirds_l2384_238462


namespace coeff_x_cubed_in_product_l2384_238403

def p (x : ℝ) : ℝ := x^5 - 4*x^3 + 3*x^2 - 2*x + 1
def q (x : ℝ) : ℝ := 3*x^3 - 2*x^2 + x + 5

theorem coeff_x_cubed_in_product (x : ℝ) :
  ∃ (a b c d e : ℝ), p x * q x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + (p 0 * q 0) ∧ c = -10 :=
sorry

end coeff_x_cubed_in_product_l2384_238403


namespace ellipse_equation_l2384_238405

/-- An ellipse with the given properties has the equation x²/2 + 3y²/2 = 1 or 3x²/2 + y²/2 = 1 -/
theorem ellipse_equation (E : Set (ℝ × ℝ)) (P Q : ℝ × ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ E ↔ ∃ (m n : ℝ), m * x^2 + n * y^2 = 1) →  -- E is an ellipse
  (0, 0) ∈ E →  -- center at origin
  (∃ (a : ℝ), (a, 0) ∈ E ∧ (-a, 0) ∈ E) ∨ (∃ (b : ℝ), (0, b) ∈ E ∧ (0, -b) ∈ E) →  -- foci on coordinate axis
  (∃ (x : ℝ), P = (x, x + 1) ∧ Q = (x, x + 1) ∧ P ∈ E ∧ Q ∈ E) →  -- P and Q on y = x + 1 and on E
  P.1 * Q.1 + P.2 * Q.2 = 0 →  -- OP · OQ = 0
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 5/2 →  -- |PQ|² = (√10/2)² = 5/2
  (∀ (x y : ℝ), (x, y) ∈ E ↔ x^2/2 + 3*y^2/2 = 1) ∨
  (∀ (x y : ℝ), (x, y) ∈ E ↔ 3*x^2/2 + y^2/2 = 1) :=
by sorry

end ellipse_equation_l2384_238405


namespace common_external_tangent_intercept_l2384_238460

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The common external tangent problem --/
theorem common_external_tangent_intercept 
  (c1 : Circle) 
  (c2 : Circle) 
  (h1 : c1.center = (3, 2)) 
  (h2 : c1.radius = 5) 
  (h3 : c2.center = (12, 10)) 
  (h4 : c2.radius = 7) :
  ∃ (m b : ℝ), m > 0 ∧ 
    (∀ (x y : ℝ), y = m * x + b → 
      ((x - c1.center.1)^2 + (y - c1.center.2)^2 = c1.radius^2 ∨
       (x - c2.center.1)^2 + (y - c2.center.2)^2 = c2.radius^2)) ∧
    b = -313/17 := by
  sorry

end common_external_tangent_intercept_l2384_238460


namespace P_equals_set_l2384_238467

def P : Set ℝ := {x | x^2 = 1}

theorem P_equals_set : P = {-1, 1} := by
  sorry

end P_equals_set_l2384_238467


namespace complement_of_A_wrt_U_l2384_238490

def U : Set Int := {-1, 2, 4}
def A : Set Int := {-1, 4}

theorem complement_of_A_wrt_U :
  (U \ A) = {2} := by sorry

end complement_of_A_wrt_U_l2384_238490


namespace extra_bananas_l2384_238455

theorem extra_bananas (total_children absent_children original_bananas : ℕ) 
  (h1 : total_children = 640)
  (h2 : absent_children = 320)
  (h3 : original_bananas = 2) : 
  (total_children * original_bananas) / (total_children - absent_children) - original_bananas = 2 := by
  sorry

end extra_bananas_l2384_238455


namespace sequence_sum_l2384_238463

theorem sequence_sum (a b : ℕ) : 
  let seq : List ℕ := [a, b, a + b, a + 2*b, 2*a + 3*b, a + 2*b + 7, a + 2*b + 14, 2*a + 4*b + 21, 3*a + 6*b + 35]
  2*a + 3*b = 7 → 
  3*a + 6*b + 35 = 47 → 
  seq.sum = 122 := by
  sorry

end sequence_sum_l2384_238463


namespace sqrt_40_between_6_and_7_l2384_238479

theorem sqrt_40_between_6_and_7 :
  ∃ (x : ℝ), x = Real.sqrt 40 ∧ 6 < x ∧ x < 7 :=
by
  have h1 : Real.sqrt 36 < Real.sqrt 40 ∧ Real.sqrt 40 < Real.sqrt 49 := by sorry
  sorry

end sqrt_40_between_6_and_7_l2384_238479


namespace rectangle_count_in_5x5_grid_l2384_238418

/-- The number of ways to select a rectangle in a 5x5 grid -/
def rectangleCount : ℕ := 225

/-- The number of horizontal or vertical lines in a 5x5 grid, including boundaries -/
def lineCount : ℕ := 6

theorem rectangle_count_in_5x5_grid :
  rectangleCount = (lineCount.choose 2) * (lineCount.choose 2) :=
sorry

end rectangle_count_in_5x5_grid_l2384_238418


namespace no_valid_operation_l2384_238453

-- Define the type for standard arithmetic operations
inductive ArithOp
  | Add
  | Sub
  | Mul
  | Div

def applyOp (op : ArithOp) (a b : Int) : Int :=
  match op with
  | ArithOp.Add => a + b
  | ArithOp.Sub => a - b
  | ArithOp.Mul => a * b
  | ArithOp.Div => a / b

theorem no_valid_operation :
  ∀ (op : ArithOp), (applyOp op 8 4) + 5 - (3 - 2) ≠ 4 := by
  sorry

#check no_valid_operation

end no_valid_operation_l2384_238453


namespace engineer_number_theorem_l2384_238448

def proper_divisors (n : ℕ) : Set ℕ :=
  {d | d ∣ n ∧ d ≠ 1 ∧ d ≠ n}

def increased_divisors (n : ℕ) : Set ℕ :=
  {d + 1 | d ∈ proper_divisors n}

theorem engineer_number_theorem :
  {n : ℕ | ∃ m : ℕ, increased_divisors n = proper_divisors m} = {4, 8} := by
sorry

end engineer_number_theorem_l2384_238448


namespace rationalize_denominator_l2384_238487

theorem rationalize_denominator :
  1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by
  sorry

end rationalize_denominator_l2384_238487


namespace root_product_equals_27_l2384_238425

theorem root_product_equals_27 : 
  (81 : ℝ) ^ (1/4) * (27 : ℝ) ^ (1/3) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end root_product_equals_27_l2384_238425


namespace sasha_floor_problem_l2384_238457

theorem sasha_floor_problem (total_floors : ℕ) :
  (∃ (floors_descended : ℕ),
    floors_descended = total_floors / 3 ∧
    floors_descended + 1 = total_floors - (total_floors / 2)) →
  total_floors + 1 = 7 :=
by sorry

end sasha_floor_problem_l2384_238457


namespace ratio_problem_l2384_238444

theorem ratio_problem (a b : ℝ) (h1 : a / b = 5) (h2 : a = 65) : b = 13 := by
  sorry

end ratio_problem_l2384_238444


namespace marcel_potatoes_l2384_238449

theorem marcel_potatoes (marcel_corn : ℕ) (dale_potatoes : ℕ) (total_vegetables : ℕ)
  (h1 : marcel_corn = 10)
  (h2 : total_vegetables = 27)
  (h3 : dale_potatoes = 8) :
  total_vegetables - (marcel_corn + marcel_corn / 2 + dale_potatoes) = 4 := by
sorry

end marcel_potatoes_l2384_238449


namespace solution_set_absolute_value_inequality_l2384_238459

-- Define the set of real numbers less than 2
def lessThanTwo : Set ℝ := {x | x < 2}

-- State the theorem
theorem solution_set_absolute_value_inequality :
  {x : ℝ | |x - 2| > x - 2} = lessThanTwo := by
  sorry

end solution_set_absolute_value_inequality_l2384_238459


namespace scientific_notation_159600_l2384_238406

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_159600 :
  toScientificNotation 159600 = ScientificNotation.mk 1.596 5 (by norm_num) :=
sorry

end scientific_notation_159600_l2384_238406


namespace pump_water_in_half_hour_l2384_238484

/-- Given a pump that moves 560 gallons of water per hour, 
    prove that it will move 280 gallons in 30 minutes. -/
theorem pump_water_in_half_hour (pump_rate : ℝ) (time : ℝ) : 
  pump_rate = 560 → time = 0.5 → pump_rate * time = 280 := by
  sorry

end pump_water_in_half_hour_l2384_238484


namespace jacksons_entertainment_spending_l2384_238488

/-- The total amount Jackson spent on entertainment -/
def total_spent (computer_game_price movie_ticket_price number_of_tickets : ℕ) : ℕ :=
  computer_game_price + movie_ticket_price * number_of_tickets

/-- Theorem stating that Jackson's total entertainment spending is $102 -/
theorem jacksons_entertainment_spending :
  total_spent 66 12 3 = 102 := by
  sorry

end jacksons_entertainment_spending_l2384_238488


namespace product_bounds_l2384_238468

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  x + y = a ∧ x^2 + y^2 = -a^2 + 2

-- Define the product function
def product (x y : ℝ) : ℝ := x * y

-- Theorem statement
theorem product_bounds :
  ∀ x y a : ℝ, system x y a → 
    (∃ x' y' a' : ℝ, system x' y' a' ∧ product x' y' = 1/3) ∧
    (∃ x'' y'' a'' : ℝ, system x'' y'' a'' ∧ product x'' y'' = -1) ∧
    (∀ x''' y''' a''' : ℝ, system x''' y''' a''' → 
      -1 ≤ product x''' y''' ∧ product x''' y''' ≤ 1/3) :=
sorry

end product_bounds_l2384_238468


namespace sum_of_four_numbers_l2384_238440

theorem sum_of_four_numbers : 1432 + 3214 + 2143 + 4321 = 11110 := by
  sorry

end sum_of_four_numbers_l2384_238440


namespace fifteen_percent_greater_l2384_238429

theorem fifteen_percent_greater : ∃ (x : ℝ), (15 / 100 * 40 = 25 / 100 * x + 2) ∧ (x = 16) := by
  sorry

end fifteen_percent_greater_l2384_238429


namespace hulk_jump_exceeds_1000_l2384_238441

def hulk_jump (n : ℕ) : ℕ := 2^(n-1)

theorem hulk_jump_exceeds_1000 :
  ∃ n : ℕ, n > 0 ∧ hulk_jump n > 1000 ∧ ∀ m : ℕ, m > 0 ∧ m < n → hulk_jump m ≤ 1000 :=
by
  use 11
  sorry

end hulk_jump_exceeds_1000_l2384_238441


namespace paint_mixture_ratio_l2384_238478

/-- Given a paint mixture with a ratio of red:blue:white as 5:3:7,
    if 21 quarts of white paint are used, then 15 quarts of red paint should be used. -/
theorem paint_mixture_ratio (red blue white : ℚ) (h1 : red / white = 5 / 7) (h2 : white = 21) :
  red = 15 := by
  sorry

end paint_mixture_ratio_l2384_238478


namespace product_equals_zero_l2384_238442

theorem product_equals_zero (b : ℤ) (h : b = 3) : 
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b * (b + 1) * (b + 2) = 0 := by
sorry

end product_equals_zero_l2384_238442


namespace parabola_transformation_l2384_238413

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a
    b := p.b - 2 * p.a * h
    c := p.a * h^2 - p.b * h + p.c }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a
    b := p.b
    c := p.c + v }

/-- The original parabola y = x^2 + 2 -/
def original_parabola : Parabola :=
  { a := 1
    b := 0
    c := 2 }

theorem parabola_transformation :
  let p1 := shift_horizontal original_parabola (-1)
  let p2 := shift_vertical p1 (-1)
  p2 = { a := 1, b := 2, c := 1 } :=
by sorry

end parabola_transformation_l2384_238413


namespace sum_largest_smallest_prime_factors_1365_l2384_238434

theorem sum_largest_smallest_prime_factors_1365 : ∃ (p q : ℕ), 
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  p ∣ 1365 ∧ 
  q ∣ 1365 ∧ 
  (∀ r : ℕ, Nat.Prime r → r ∣ 1365 → p ≤ r ∧ r ≤ q) ∧ 
  p + q = 16 := by
  sorry

end sum_largest_smallest_prime_factors_1365_l2384_238434


namespace cody_payment_proof_l2384_238486

def initial_purchase : ℝ := 40
def tax_rate : ℝ := 0.05
def discount : ℝ := 8
def cody_payment : ℝ := 17

theorem cody_payment_proof :
  cody_payment = (initial_purchase * (1 + tax_rate) - discount) / 2 := by
  sorry

end cody_payment_proof_l2384_238486


namespace unique_square_number_l2384_238411

/-- A function to convert a two-digit number to its decimal representation -/
def twoDigitToNumber (a b : ℕ) : ℕ := 10 * a + b

/-- A function to convert a three-digit number to its decimal representation -/
def threeDigitToNumber (c₁ c₂ b : ℕ) : ℕ := 100 * c₁ + 10 * c₂ + b

/-- Theorem stating that under given conditions, ccb must be 441 -/
theorem unique_square_number (a b c : ℕ) : 
  a ≠ b → b ≠ c → a ≠ c →
  b = 1 →
  0 < a → a < 10 →
  0 ≤ c → c < 10 →
  (twoDigitToNumber a b)^2 = threeDigitToNumber c c b →
  threeDigitToNumber c c b > 300 →
  threeDigitToNumber c c b = 441 := by
sorry

end unique_square_number_l2384_238411


namespace greene_family_spending_l2384_238483

theorem greene_family_spending (admission_cost food_cost total_cost : ℕ) : 
  admission_cost = 45 →
  food_cost = admission_cost - 13 →
  total_cost = admission_cost + food_cost →
  total_cost = 77 := by
sorry

end greene_family_spending_l2384_238483


namespace total_hours_is_fifty_l2384_238426

/-- Represents the worker's pay structure and work week --/
structure WorkWeek where
  ordinary_rate : ℚ  -- Rate for ordinary time in dollars per hour
  overtime_rate : ℚ  -- Rate for overtime in dollars per hour
  total_pay : ℚ      -- Total pay for the week in dollars
  overtime_hours : ℕ  -- Number of overtime hours worked

/-- Calculates the total hours worked given a WorkWeek --/
def total_hours (w : WorkWeek) : ℚ :=
  let ordinary_hours := (w.total_pay - w.overtime_rate * w.overtime_hours) / w.ordinary_rate
  ordinary_hours + w.overtime_hours

/-- Theorem stating that given the specific conditions, the total hours worked is 50 --/
theorem total_hours_is_fifty : 
  ∀ (w : WorkWeek), 
    w.ordinary_rate = 0.60 ∧ 
    w.overtime_rate = 0.90 ∧ 
    w.total_pay = 32.40 ∧ 
    w.overtime_hours = 8 → 
    total_hours w = 50 :=
by
  sorry


end total_hours_is_fifty_l2384_238426


namespace twentieth_stage_toothpicks_l2384_238498

/-- Number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ := 3 + 3 * (n - 1)

/-- The 20th stage of the toothpick pattern has 60 toothpicks -/
theorem twentieth_stage_toothpicks : toothpicks 20 = 60 := by
  sorry

end twentieth_stage_toothpicks_l2384_238498


namespace katie_mp3_songs_l2384_238421

theorem katie_mp3_songs (initial : ℕ) (deleted : ℕ) (added : ℕ) (final : ℕ) : 
  initial = 11 → deleted = 7 → added = 24 → final = initial - deleted + added → final = 28 := by
  sorry

end katie_mp3_songs_l2384_238421


namespace pentadecagon_diagonals_l2384_238469

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A pentadecagon is a 15-sided polygon -/
def pentadecagon_sides : ℕ := 15

/-- Theorem: The number of diagonals in a pentadecagon is 90 -/
theorem pentadecagon_diagonals : num_diagonals pentadecagon_sides = 90 := by
  sorry

end pentadecagon_diagonals_l2384_238469


namespace sqrt_y_squared_range_l2384_238495

theorem sqrt_y_squared_range (y : ℝ) (h : (y + 16) ^ (1/3) - (y - 16) ^ (1/3) = 4) :
  15 < Real.sqrt (y^2) ∧ Real.sqrt (y^2) < 16 := by
  sorry

end sqrt_y_squared_range_l2384_238495


namespace intersection_point_satisfies_equations_unique_intersection_point_l2384_238404

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (50/17, 24/17)

/-- First line equation: 2y = 3x - 6 -/
def line1 (x y : ℚ) : Prop := 2 * y = 3 * x - 6

/-- Second line equation: x + 5y = 10 -/
def line2 (x y : ℚ) : Prop := x + 5 * y = 10

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations : 
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end intersection_point_satisfies_equations_unique_intersection_point_l2384_238404


namespace solution_set_is_open_interval_l2384_238475

/-- A function with specific symmetry and derivative properties -/
class SpecialFunction (f : ℝ → ℝ) where
  symmetric : ∀ x, f x = f (-2 - x)
  derivative_property : ∀ x, x < -1 → (x + 1) * (f x + (x + 1) * (deriv f) x) < 0

/-- The solution set of the inequality xf(x-1) > f(0) -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | x * f (x - 1) > f 0}

/-- The theorem stating the solution set of the inequality -/
theorem solution_set_is_open_interval
  (f : ℝ → ℝ) [SpecialFunction f] :
  SolutionSet f = Set.Ioo (-1) 1 := by
  sorry

end solution_set_is_open_interval_l2384_238475


namespace maria_workday_end_l2384_238408

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat

def Time.add (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.minutes + d.minutes
  let newHours := (t.hours + d.hours + totalMinutes / 60) % 24
  let newMinutes := totalMinutes % 60
  ⟨newHours, newMinutes, by sorry, by sorry⟩

def Duration.add (d1 d2 : Duration) : Duration :=
  let totalMinutes := d1.minutes + d2.minutes + (d1.hours + d2.hours) * 60
  ⟨totalMinutes / 60, totalMinutes % 60⟩

def workDay (start : Time) (workDuration : Duration) (lunchStart : Time) (lunchDuration : Duration) (breakStart : Time) (breakDuration : Duration) : Time :=
  let lunchEnd := lunchStart.add lunchDuration
  let breakEnd := breakStart.add breakDuration
  let totalBreakDuration := Duration.add lunchDuration breakDuration
  start.add (Duration.add workDuration totalBreakDuration)

theorem maria_workday_end :
  let start : Time := ⟨8, 0, by sorry, by sorry⟩
  let workDuration : Duration := ⟨8, 0⟩
  let lunchStart : Time := ⟨13, 0, by sorry, by sorry⟩
  let lunchDuration : Duration := ⟨1, 0⟩
  let breakStart : Time := ⟨15, 30, by sorry, by sorry⟩
  let breakDuration : Duration := ⟨0, 15⟩
  let endTime : Time := workDay start workDuration lunchStart lunchDuration breakStart breakDuration
  endTime = ⟨18, 0, by sorry, by sorry⟩ := by
  sorry


end maria_workday_end_l2384_238408


namespace pizza_slices_left_l2384_238443

theorem pizza_slices_left (total_slices : ℕ) (people : ℕ) (slices_per_person : ℕ) :
  total_slices = 16 →
  people = 6 →
  slices_per_person = 2 →
  total_slices - (people * slices_per_person) = 4 :=
by
  sorry

end pizza_slices_left_l2384_238443


namespace commute_days_l2384_238437

theorem commute_days (x : ℕ) 
  (h1 : x > 0)
  (h2 : 2 * x = 9 + 8 + 15) : 
  x = 16 := by
sorry

end commute_days_l2384_238437


namespace basketball_evaluation_theorem_l2384_238452

/-- The number of rounds in the basketball evaluation -/
def num_rounds : ℕ := 3

/-- The number of shots per round -/
def shots_per_round : ℕ := 2

/-- The probability of player A making a shot -/
def prob_make_shot : ℚ := 2/3

/-- The probability of passing a single round -/
def prob_pass_round : ℚ := 1 - (1 - prob_make_shot) ^ shots_per_round

/-- The expected number of rounds player A will pass -/
def expected_passed_rounds : ℚ := num_rounds * prob_pass_round

theorem basketball_evaluation_theorem :
  expected_passed_rounds = 8/3 := by sorry

end basketball_evaluation_theorem_l2384_238452


namespace soccer_substitution_remainder_l2384_238458

/-- Represents the number of ways to make substitutions in a soccer game -/
def substitution_ways (total_players : ℕ) (starting_players : ℕ) (max_substitutions : ℕ) : ℕ :=
  let substitute_players := total_players - starting_players
  let rec ways_for_n (n : ℕ) : ℕ :=
    if n = 0 then 1
    else starting_players * (substitute_players - n + 1) * ways_for_n (n - 1)
  (List.range (max_substitutions + 1)).map ways_for_n |> List.sum

/-- The main theorem stating the remainder of substitution ways when divided by 1000 -/
theorem soccer_substitution_remainder :
  substitution_ways 22 11 4 % 1000 = 25 := by
  sorry


end soccer_substitution_remainder_l2384_238458


namespace f_monotonicity_and_zeros_l2384_238439

def f (k : ℝ) (x : ℝ) : ℝ := x^3 - k*x + k^2

theorem f_monotonicity_and_zeros (k : ℝ) :
  (∀ x y, x < y → k ≤ 0 → f k x < f k y) ∧
  (k > 0 → ∀ x y, (x < y ∧ y < -Real.sqrt (k/3)) ∨ (x < y ∧ x > Real.sqrt (k/3)) → f k x < f k y) ∧
  (k > 0 → ∀ x y, -Real.sqrt (k/3) < x ∧ x < y ∧ y < Real.sqrt (k/3) → f k x > f k y) ∧
  (∃ x y z, x < y ∧ y < z ∧ f k x = 0 ∧ f k y = 0 ∧ f k z = 0 ↔ 0 < k ∧ k < 4/27) :=
by sorry

end f_monotonicity_and_zeros_l2384_238439


namespace vasya_toy_choices_l2384_238423

/-- The number of different remote-controlled cars available -/
def num_cars : ℕ := 7

/-- The number of different construction sets available -/
def num_sets : ℕ := 5

/-- The total number of toys available -/
def total_toys : ℕ := num_cars + num_sets

/-- The number of toys Vasya can choose -/
def toys_to_choose : ℕ := 2

theorem vasya_toy_choices :
  Nat.choose total_toys toys_to_choose = 66 :=
sorry

end vasya_toy_choices_l2384_238423


namespace fraction_product_proof_l2384_238461

theorem fraction_product_proof :
  (8 / 4) * (10 / 25) * (27 / 18) * (16 / 24) * (35 / 21) * (30 / 50) * (14 / 7) * (20 / 40) = 4 / 5 := by
  sorry

end fraction_product_proof_l2384_238461


namespace total_vegetables_bought_l2384_238494

/-- The number of vegetables bought by Marcel and Dale -/
def total_vegetables (marcel_corn : ℕ) (dale_corn : ℕ) (marcel_potatoes : ℕ) (dale_potatoes : ℕ) : ℕ :=
  marcel_corn + dale_corn + marcel_potatoes + dale_potatoes

/-- Theorem stating the total number of vegetables bought by Marcel and Dale -/
theorem total_vegetables_bought :
  ∃ (marcel_corn marcel_potatoes dale_potatoes : ℕ),
    marcel_corn = 10 ∧
    marcel_potatoes = 4 ∧
    dale_potatoes = 8 ∧
    total_vegetables marcel_corn (marcel_corn / 2) marcel_potatoes dale_potatoes = 27 := by
  sorry

end total_vegetables_bought_l2384_238494


namespace w_share_is_375_l2384_238401

/-- A structure representing the distribution of money among four individuals -/
structure MoneyDistribution where
  total : ℝ
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ
  proportion_w : ℝ := 1
  proportion_x : ℝ := 6
  proportion_y : ℝ := 2
  proportion_z : ℝ := 4
  sum_proportions : ℝ := proportion_w + proportion_x + proportion_y + proportion_z
  proportional_distribution :
    w / proportion_w = x / proportion_x ∧
    x / proportion_x = y / proportion_y ∧
    y / proportion_y = z / proportion_z ∧
    w + x + y + z = total
  x_exceeds_y : x = y + 1500

theorem w_share_is_375 (d : MoneyDistribution) : d.w = 375 := by
  sorry

end w_share_is_375_l2384_238401


namespace largest_four_digit_multiple_of_48_l2384_238464

theorem largest_four_digit_multiple_of_48 : 
  (∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 48 = 0 → n ≤ 9984) ∧ 
  9984 % 48 = 0 ∧ 
  9984 ≤ 9999 ∧ 
  9984 ≥ 1000 := by
sorry

end largest_four_digit_multiple_of_48_l2384_238464


namespace persimmon_count_l2384_238472

theorem persimmon_count (total : ℕ) (difference : ℕ) (persimmons : ℕ) (tangerines : ℕ) : 
  total = 129 →
  difference = 43 →
  total = persimmons + tangerines →
  persimmons + difference = tangerines →
  persimmons = 43 := by
sorry

end persimmon_count_l2384_238472


namespace arithmetic_simplification_l2384_238482

theorem arithmetic_simplification : 
  (427 / 2.68) * 16 * 26.8 / 42.7 * 16 = 25600 := by
  sorry

end arithmetic_simplification_l2384_238482


namespace sum_reciprocal_product_bound_sum_product_bound_l2384_238422

-- Part (a)
theorem sum_reciprocal_product_bound (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hsum : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := by sorry

-- Part (b)
theorem sum_product_bound (u v : ℝ) (hu : 0 < u ∧ u < 1) (hv : 0 < v ∧ v < 1) :
  0 < u + v - u*v ∧ u + v - u*v < 1 := by sorry

end sum_reciprocal_product_bound_sum_product_bound_l2384_238422


namespace product_of_one_plus_tans_l2384_238435

theorem product_of_one_plus_tans (α β : Real) (h : α + β = π / 4) :
  (1 + Real.tan α) * (1 + Real.tan β) = 2 := by
  sorry

end product_of_one_plus_tans_l2384_238435


namespace quadratic_equation_and_inequality_l2384_238409

theorem quadratic_equation_and_inequality 
  (a b : ℝ) 
  (h1 : (a:ℝ) * (-1/2)^2 + b * (-1/2) + 2 = 0)
  (h2 : (a:ℝ) * 2^2 + b * 2 + 2 = 0) :
  (a = -2 ∧ b = 3) ∧ 
  (∀ x : ℝ, a * x^2 + b * x - 1 > 0 ↔ 1/2 < x ∧ x < 1) := by
sorry

end quadratic_equation_and_inequality_l2384_238409


namespace min_distance_parabola_circle_l2384_238430

/-- The minimum distance between a point on the parabola y^2 = 6x and a point on the circle (x-4)^2 + y^2 = 1 is √15 - 1 -/
theorem min_distance_parabola_circle :
  ∃ (d : ℝ), d = Real.sqrt 15 - 1 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    y₁^2 = 6*x₁ →
    (x₂ - 4)^2 + y₂^2 = 1 →
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 : ℝ) ≥ d^2 :=
by sorry

end min_distance_parabola_circle_l2384_238430


namespace M_intersect_N_l2384_238410

def M : Set Int := {m | -3 < m ∧ m < 2}
def N : Set Int := {n | -1 ≤ n ∧ n ≤ 3}

theorem M_intersect_N : M ∩ N = {-1, 0, 1} := by sorry

end M_intersect_N_l2384_238410
