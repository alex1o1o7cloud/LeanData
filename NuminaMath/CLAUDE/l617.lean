import Mathlib

namespace NUMINAMATH_CALUDE_wood_amount_correct_l617_61744

/-- The amount of wood (in cubic meters) that two workers need to saw and chop in one day -/
def wood_amount : ℚ := 40 / 13

/-- The amount of wood (in cubic meters) that two workers can saw in one day -/
def saw_capacity : ℚ := 5

/-- The amount of wood (in cubic meters) that two workers can chop in one day -/
def chop_capacity : ℚ := 8

/-- Theorem stating that the wood_amount is the correct amount of wood that two workers 
    need to saw in order to have enough time to chop it for the remainder of the day -/
theorem wood_amount_correct : 
  wood_amount / saw_capacity + wood_amount / chop_capacity = 1 := by
  sorry


end NUMINAMATH_CALUDE_wood_amount_correct_l617_61744


namespace NUMINAMATH_CALUDE_trig_identity_l617_61790

theorem trig_identity (θ φ : ℝ) 
  (h : (Real.sin θ)^6 / (Real.sin φ)^3 + (Real.cos θ)^6 / (Real.cos φ)^3 = 1) :
  (Real.sin φ)^6 / (Real.sin θ)^3 + (Real.cos φ)^6 / (Real.cos θ)^3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trig_identity_l617_61790


namespace NUMINAMATH_CALUDE_is_circle_center_l617_61700

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8*y + 9 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (3, -4)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center :
  ∀ x y : ℝ, circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_is_circle_center_l617_61700


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l617_61776

-- Define the triangle
structure Triangle :=
  (a b c : ℝ)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a + t.b + t.c = 180

-- Define the specific conditions of our triangle
def our_triangle (t : Triangle) : Prop :=
  is_valid_triangle t ∧
  t.a = 70 ∧
  t.b = 40 ∧
  t.c = 70

-- Theorem statement
theorem triangle_angle_sum (t : Triangle) :
  our_triangle t → t.c = 40 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l617_61776


namespace NUMINAMATH_CALUDE_smallest_odd_digit_multiple_of_11_proof_l617_61754

def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

def smallest_odd_digit_multiple_of_11 : ℕ := 11341

theorem smallest_odd_digit_multiple_of_11_proof :
  (smallest_odd_digit_multiple_of_11 > 10000) ∧
  (has_only_odd_digits smallest_odd_digit_multiple_of_11) ∧
  (smallest_odd_digit_multiple_of_11 % 11 = 0) ∧
  (∀ n : ℕ, n > 10000 → has_only_odd_digits n → n % 11 = 0 → n ≥ smallest_odd_digit_multiple_of_11) :=
by sorry

end NUMINAMATH_CALUDE_smallest_odd_digit_multiple_of_11_proof_l617_61754


namespace NUMINAMATH_CALUDE_family_travel_distance_l617_61761

/-- Proves that the total distance travelled is 448 km given the specified conditions --/
theorem family_travel_distance : 
  ∀ (total_distance : ℝ),
  (total_distance / (2 * 35) + total_distance / (2 * 40) = 12) →
  total_distance = 448 := by
sorry

end NUMINAMATH_CALUDE_family_travel_distance_l617_61761


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l617_61773

theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) :
  b₁ = 2 →
  (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) →
  ∀ b₂' b₃' : ℝ, (∃ r' : ℝ, b₂' = 2 * r' ∧ b₃' = b₂' * r') →
  3 * b₂ + 4 * b₃ ≥ 3 * b₂' + 4 * b₃' →
  3 * b₂ + 4 * b₃ ≥ -9/8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l617_61773


namespace NUMINAMATH_CALUDE_quadratic_sum_equals_27_l617_61777

theorem quadratic_sum_equals_27 (m n : ℝ) (h : m + n = 4) : 
  2 * m^2 + 4 * m * n + 2 * n^2 - 5 = 27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_equals_27_l617_61777


namespace NUMINAMATH_CALUDE_tom_roses_count_l617_61763

/-- The number of roses in a dozen -/
def roses_per_dozen : ℕ := 12

/-- The number of dozens Tom sends per day -/
def dozens_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of roses Tom sent in a week -/
def total_roses : ℕ := days_in_week * dozens_per_day * roses_per_dozen

theorem tom_roses_count : total_roses = 168 := by
  sorry

end NUMINAMATH_CALUDE_tom_roses_count_l617_61763


namespace NUMINAMATH_CALUDE_triangle_inequality_l617_61769

/-- Theorem: For any triangle with side lengths a, b, c, and area S,
    the inequality a² + b² + c² ≥ 4√3 S holds, with equality if and only if
    the triangle is equilateral. -/
theorem triangle_inequality (a b c S : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : S > 0)
    (h_area_def : S = Real.sqrt (s * (s - a) * (s - b) * (s - c)))
    (h_s_def : s = (a + b + c) / 2) :
    a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S ∧
    (a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * S ↔ a = b ∧ b = c) :=
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l617_61769


namespace NUMINAMATH_CALUDE_seventh_observation_value_l617_61764

theorem seventh_observation_value 
  (n : Nat) 
  (initial_avg : ℝ) 
  (new_avg : ℝ) 
  (h1 : n = 6) 
  (h2 : initial_avg = 11) 
  (h3 : new_avg = initial_avg - 1) : 
  (n : ℝ) * initial_avg - ((n + 1) : ℝ) * new_avg = -4 := by
  sorry

end NUMINAMATH_CALUDE_seventh_observation_value_l617_61764


namespace NUMINAMATH_CALUDE_infinite_occurrences_l617_61755

-- Define the sequence
def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 
    let prev := a n
    let god := (n + 1).factorization.prod (λ p k => p)  -- greatest odd divisor
    if god % 4 = 1 then prev + 1 else prev - 1

-- State the theorem
theorem infinite_occurrences :
  (∀ k : ℕ+, Set.Infinite {n : ℕ | a n = k}) ∧
  Set.Infinite {n : ℕ | a n = 1} := by
  sorry

end NUMINAMATH_CALUDE_infinite_occurrences_l617_61755


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l617_61780

/-- Given a rhombus with one diagonal of length 60 meters and an area of 1950 square meters,
    prove that the length of the other diagonal is 65 meters. -/
theorem rhombus_diagonal (d₁ : ℝ) (d₂ : ℝ) (area : ℝ) 
    (h₁ : d₁ = 60)
    (h₂ : area = 1950)
    (h₃ : area = (d₁ * d₂) / 2) : 
  d₂ = 65 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l617_61780


namespace NUMINAMATH_CALUDE_alligators_not_hiding_l617_61718

/-- The number of alligators not hiding in a zoo cage -/
theorem alligators_not_hiding (total_alligators hiding_alligators : ℕ) :
  total_alligators = 75 → hiding_alligators = 19 →
  total_alligators - hiding_alligators = 56 := by
  sorry

#check alligators_not_hiding

end NUMINAMATH_CALUDE_alligators_not_hiding_l617_61718


namespace NUMINAMATH_CALUDE_smallest_lucky_number_unique_lucky_number_divisible_by_seven_l617_61791

/-- Definition of a lucky number -/
def is_lucky_number (M : ℕ) : Prop :=
  ∃ (A B : ℕ), 
    M % 10 ≠ 0 ∧
    M = A * B ∧
    A ≥ B ∧
    10 ≤ A ∧ A < 100 ∧
    10 ≤ B ∧ B < 100 ∧
    (A / 10) = (B / 10) ∧
    (A % 10) + (B % 10) = 6

/-- The smallest lucky number is 165 -/
theorem smallest_lucky_number : 
  is_lucky_number 165 ∧ ∀ M, is_lucky_number M → M ≥ 165 := by sorry

/-- There exists a unique lucky number M such that (A + B) / (A - B) is divisible by 7, and it equals 3968 -/
theorem unique_lucky_number_divisible_by_seven :
  ∃! M, is_lucky_number M ∧ 
    (∃ A B, M = A * B ∧ ((A + B) / (A - B)) % 7 = 0) ∧
    M = 3968 := by sorry

end NUMINAMATH_CALUDE_smallest_lucky_number_unique_lucky_number_divisible_by_seven_l617_61791


namespace NUMINAMATH_CALUDE_max_profit_is_270000_l617_61752

/-- Represents the production quantities of products A and B -/
structure Production where
  a : ℝ
  b : ℝ

/-- Represents the constraints and profit calculation for the company -/
def Company :=
  { p : Production //
    p.a ≥ 0 ∧
    p.b ≥ 0 ∧
    3 * p.a + p.b ≤ 13 ∧
    2 * p.a + 3 * p.b ≤ 18 }

/-- Calculates the profit for a given production -/
def profit (p : Production) : ℝ := 50000 * p.a + 30000 * p.b

/-- Theorem stating that the maximum profit is 270,000 yuan -/
theorem max_profit_is_270000 :
  ∃ (p : Company), ∀ (q : Company), profit p.val ≥ profit q.val ∧ profit p.val = 270000 := by
  sorry


end NUMINAMATH_CALUDE_max_profit_is_270000_l617_61752


namespace NUMINAMATH_CALUDE_missing_number_proof_l617_61781

theorem missing_number_proof (known_numbers : List ℕ) (mean : ℚ) : 
  known_numbers = [1, 23, 24, 25, 26, 27, 2] ∧ 
  mean = 20 ∧ 
  (List.sum known_numbers + 32) / 8 = mean →
  32 = 8 * mean - List.sum known_numbers :=
by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l617_61781


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l617_61740

theorem complex_fraction_simplification (a b : ℝ) (ha : a = 4.91) (hb : b = 0.09) :
  (((a^2 - b^2) * (a^2 + b^(2/3) + a * b^(1/3))) / (a * b^(1/3) + a * a^(1/2) - b * b^(1/3) - (a * b^2)^(1/2))) /
  ((a^3 - b) / (a * b^(1/3) - (a^3 * b^2)^(1/6) - b^(2/3) + a * a^(1/2))) = a + b :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l617_61740


namespace NUMINAMATH_CALUDE_james_toys_l617_61760

theorem james_toys (toy_cars : ℕ) (toy_soldiers : ℕ) : 
  toy_cars = 20 → 
  toy_soldiers = 2 * toy_cars → 
  toy_cars + toy_soldiers = 60 := by
  sorry

end NUMINAMATH_CALUDE_james_toys_l617_61760


namespace NUMINAMATH_CALUDE_community_population_l617_61733

/-- Represents the number of people in each category of a community --/
structure Community where
  babies : ℝ
  seniors : ℝ
  children : ℝ
  teenagers : ℝ
  women : ℝ
  men : ℝ

/-- The total number of people in the community --/
def totalPeople (c : Community) : ℝ :=
  c.babies + c.seniors + c.children + c.teenagers + c.women + c.men

/-- Theorem stating the relationship between the number of babies and the total population --/
theorem community_population (c : Community) 
  (h1 : c.men = 1.5 * c.women)
  (h2 : c.women = 3 * c.teenagers)
  (h3 : c.teenagers = 2.5 * c.children)
  (h4 : c.children = 4 * c.seniors)
  (h5 : c.seniors = 3.5 * c.babies) :
  totalPeople c = 316 * c.babies := by
  sorry


end NUMINAMATH_CALUDE_community_population_l617_61733


namespace NUMINAMATH_CALUDE_tangent_line_problem_range_problem_l617_61723

noncomputable section

-- Define the function f(x) = x - ln x
def f (x : ℝ) : ℝ := x - Real.log x

-- Define the function g(x) = (e-1)x
def g (x : ℝ) : ℝ := (Real.exp 1 - 1) * x

-- Define the piecewise function F(x)
def F (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ a then f x else g x

-- Theorem for the tangent line problem
theorem tangent_line_problem (x₀ : ℝ) :
  (∃ k : ℝ, ∀ x : ℝ, k * x = f x + (x - x₀) * (1 - 1 / x₀)) →
  (x₀ = Real.exp 1 ∧ ∃ k : ℝ, k = 1 - 1 / Real.exp 1) :=
sorry

-- Theorem for the range problem
theorem range_problem (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, F a x = y) →
  a ≥ 1 / (Real.exp 1 - 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_problem_range_problem_l617_61723


namespace NUMINAMATH_CALUDE_festival_worker_assignment_l617_61725

def number_of_workers : ℕ := 6

def number_of_desks : ℕ := 2

def min_workers_per_desk : ℕ := 2

def ways_to_assign_workers (n : ℕ) (k : ℕ) (min_per_group : ℕ) : ℕ :=
  sorry

theorem festival_worker_assignment :
  ways_to_assign_workers number_of_workers number_of_desks min_workers_per_desk = 28 :=
sorry

end NUMINAMATH_CALUDE_festival_worker_assignment_l617_61725


namespace NUMINAMATH_CALUDE_cube_rotation_theorem_l617_61796

/-- Represents a cube with natural numbers on each face -/
structure Cube where
  front : ℕ
  right : ℕ
  back : ℕ
  left : ℕ
  top : ℕ
  bottom : ℕ

/-- Theorem stating the properties of the cube and the results to be proved -/
theorem cube_rotation_theorem (c : Cube) 
  (h1 : c.front + c.right + c.top = 42)
  (h2 : c.right + c.top + c.back = 34)
  (h3 : c.top + c.back + c.left = 53)
  (h4 : c.bottom = 6) :
  (c.left + c.front + c.top = 61) ∧ 
  (c.front + c.right + c.back + c.left + c.top + c.bottom ≤ 100) := by
  sorry


end NUMINAMATH_CALUDE_cube_rotation_theorem_l617_61796


namespace NUMINAMATH_CALUDE_area_outside_inscribed_square_l617_61747

def square_side_length : ℝ := 2

theorem area_outside_inscribed_square (square_side : ℝ) (h : square_side = square_side_length) :
  let circle_radius : ℝ := square_side * Real.sqrt 2 / 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let square_area : ℝ := square_side ^ 2
  circle_area - square_area = 2 * π - 4 := by
sorry

end NUMINAMATH_CALUDE_area_outside_inscribed_square_l617_61747


namespace NUMINAMATH_CALUDE_negation_of_proposition_l617_61766

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 + 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l617_61766


namespace NUMINAMATH_CALUDE_total_pups_is_91_l617_61710

/-- Represents the number of pups for each dog breed --/
structure DogBreedPups where
  huskies : Nat
  pitbulls : Nat
  goldenRetrievers : Nat
  germanShepherds : Nat
  bulldogs : Nat
  poodles : Nat

/-- Calculates the total number of pups from all dog breeds --/
def totalPups (d : DogBreedPups) : Nat :=
  d.huskies + d.pitbulls + d.goldenRetrievers + d.germanShepherds + d.bulldogs + d.poodles

/-- Theorem stating that the total number of pups is 91 --/
theorem total_pups_is_91 :
  let numHuskies := 5
  let numPitbulls := 2
  let numGoldenRetrievers := 4
  let numGermanShepherds := 3
  let numBulldogs := 2
  let numPoodles := 3
  let huskiePups := 4
  let pitbullPups := 3
  let goldenRetrieverPups := huskiePups + 2
  let germanShepherdPups := pitbullPups + 3
  let bulldogPups := 4
  let poodlePups := bulldogPups + 1
  let d := DogBreedPups.mk
    (numHuskies * huskiePups)
    (numPitbulls * pitbullPups)
    (numGoldenRetrievers * goldenRetrieverPups)
    (numGermanShepherds * germanShepherdPups)
    (numBulldogs * bulldogPups)
    (numPoodles * poodlePups)
  totalPups d = 91 := by
  sorry

end NUMINAMATH_CALUDE_total_pups_is_91_l617_61710


namespace NUMINAMATH_CALUDE_four_team_win_structure_exists_l617_61756

/-- Represents the result of a match between two teams -/
inductive MatchResult
  | Win
  | Loss

/-- Represents a volleyball tournament -/
structure Tournament where
  teams : Finset Nat
  results : Nat → Nat → MatchResult
  round_robin : ∀ i j, i ≠ j → (results i j = MatchResult.Win ↔ results j i = MatchResult.Loss)

/-- The main theorem to be proved -/
theorem four_team_win_structure_exists (t : Tournament) 
  (h_eight_teams : t.teams.card = 8) :
  ∃ A B C D, A ∈ t.teams ∧ B ∈ t.teams ∧ C ∈ t.teams ∧ D ∈ t.teams ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    t.results A B = MatchResult.Win ∧
    t.results A C = MatchResult.Win ∧
    t.results A D = MatchResult.Win ∧
    t.results B C = MatchResult.Win ∧
    t.results B D = MatchResult.Win ∧
    t.results C D = MatchResult.Win :=
  sorry

end NUMINAMATH_CALUDE_four_team_win_structure_exists_l617_61756


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l617_61734

/-- Given a point M and its reflection N across the y-axis, and a point P on the y-axis,
    this theorem states that the line passing through P and N has the equation x - y + 1 = 0. -/
theorem reflected_ray_equation (M P N : ℝ × ℝ) : 
  M.1 = 3 ∧ M.2 = -2 ∧   -- M(3, -2)
  P.1 = 0 ∧ P.2 = 1 ∧    -- P(0, 1) on y-axis
  N.1 = -M.1 ∧ N.2 = M.2 -- N is reflection of M across y-axis
  → (∀ x y : ℝ, (x - y + 1 = 0) ↔ (∃ t : ℝ, x = N.1 * t + P.1 * (1 - t) ∧ y = N.2 * t + P.2 * (1 - t))) :=
by sorry


end NUMINAMATH_CALUDE_reflected_ray_equation_l617_61734


namespace NUMINAMATH_CALUDE_solution_count_l617_61787

/-- The number of positive integer solutions for a system of equations involving a prime number -/
def num_solutions (p : ℕ) : ℕ :=
  if p = 2 then 5
  else if p % 4 = 1 then 11
  else 3

/-- The main theorem stating the number of solutions for the given system of equations -/
theorem solution_count (p : ℕ) (hp : Nat.Prime p) :
  (∃ (n : ℕ), n = (Finset.filter (fun (quad : ℕ × ℕ × ℕ × ℕ) =>
    let (a, b, c, d) := quad
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a * c + b * d = p * (a + c) ∧
    b * c - a * d = p * (b - d))
    (Finset.product (Finset.range (p^3 + 1)) (Finset.product (Finset.range (p^3 + 1))
      (Finset.product (Finset.range (p^3 + 1)) (Finset.range (p^3 + 1)))))).card) ∧
  n = num_solutions p :=
sorry

end NUMINAMATH_CALUDE_solution_count_l617_61787


namespace NUMINAMATH_CALUDE_prob_one_sunny_day_l617_61709

/-- The probability of exactly one sunny day in a three-day festival --/
theorem prob_one_sunny_day (p_sunny : ℝ) (p_not_sunny : ℝ) :
  p_sunny = 0.1 →
  p_not_sunny = 0.9 →
  3 * (p_sunny * p_not_sunny * p_not_sunny) = 0.243 :=
by sorry

end NUMINAMATH_CALUDE_prob_one_sunny_day_l617_61709


namespace NUMINAMATH_CALUDE_scientific_notation_79000_l617_61778

theorem scientific_notation_79000 : 79000 = 7.9 * (10 ^ 4) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_79000_l617_61778


namespace NUMINAMATH_CALUDE_smallest_n_fourth_root_l617_61782

theorem smallest_n_fourth_root (n : ℕ) : n = 4097 ↔ 
  (n > 0 ∧ 
   ∀ m : ℕ, m > 0 → m < n → 
   ¬(0 < (m : ℝ)^(1/4) - ⌊(m : ℝ)^(1/4)⌋ ∧ (m : ℝ)^(1/4) - ⌊(m : ℝ)^(1/4)⌋ < 1/2015)) ∧
  (0 < (n : ℝ)^(1/4) - ⌊(n : ℝ)^(1/4)⌋ ∧ (n : ℝ)^(1/4) - ⌊(n : ℝ)^(1/4)⌋ < 1/2015) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_fourth_root_l617_61782


namespace NUMINAMATH_CALUDE_binary_remainder_by_8_l617_61793

/-- The remainder when 101110100101₂ is divided by 8 is 5. -/
theorem binary_remainder_by_8 : (101110100101 : Nat) % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_remainder_by_8_l617_61793


namespace NUMINAMATH_CALUDE_min_passengers_for_no_loss_l617_61784

/-- Represents the monthly expenditure of the bus in yuan -/
def monthly_expenditure : ℕ := 6000

/-- Represents the fare per person in yuan -/
def fare_per_person : ℕ := 2

/-- Represents the relationship between the number of passengers (x) and the difference between income and expenditure (y) -/
def income_expenditure_difference (x : ℕ) : ℤ :=
  (fare_per_person * x : ℤ) - monthly_expenditure

/-- Represents the condition for the bus to operate without a loss -/
def no_loss (x : ℕ) : Prop :=
  income_expenditure_difference x ≥ 0

/-- States that the minimum number of passengers required for the bus to operate without a loss is 3000 -/
theorem min_passengers_for_no_loss :
  ∀ x : ℕ, no_loss x ↔ x ≥ 3000 :=
by sorry

end NUMINAMATH_CALUDE_min_passengers_for_no_loss_l617_61784


namespace NUMINAMATH_CALUDE_integer_decimal_parts_theorem_l617_61735

theorem integer_decimal_parts_theorem :
  ∀ (a b : ℝ),
  (a = ⌊7 - Real.sqrt 13⌋) →
  (b = 7 - Real.sqrt 13 - a) →
  (2 * a - b = 2 + Real.sqrt 13) := by
sorry

end NUMINAMATH_CALUDE_integer_decimal_parts_theorem_l617_61735


namespace NUMINAMATH_CALUDE_equation_satisfied_l617_61785

theorem equation_satisfied (a b c : ℤ) (h1 : a = c) (h2 : b - 1 = a) : 
  a * (a - b) + b * (b - c) + c * (c - a) = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_l617_61785


namespace NUMINAMATH_CALUDE_certain_number_problem_l617_61705

theorem certain_number_problem : ∃ x : ℕ, 
  220040 = (x + 445) * (2 * (x - 445)) + 40 ∧ x = 555 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l617_61705


namespace NUMINAMATH_CALUDE_sum_of_multiples_l617_61714

theorem sum_of_multiples (p q : ℤ) : 
  (∃ m : ℤ, p = 5 * m) → (∃ n : ℤ, q = 10 * n) → (∃ k : ℤ, p + q = 5 * k) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l617_61714


namespace NUMINAMATH_CALUDE_parabola_translation_l617_61749

-- Define the original parabola function
def original_parabola (x : ℝ) : ℝ := -x^2

-- Define the translated parabola function
def translated_parabola (x : ℝ) : ℝ := -(x + 2)^2 + 3

-- Theorem statement
theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = original_parabola (x + 2) + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l617_61749


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l617_61774

theorem simplify_trig_expression (a : Real) (h : 0 < a ∧ a < π / 2) :
  Real.sqrt (1 + Real.sin a) + Real.sqrt (1 - Real.sin a) - Real.sqrt (2 + 2 * Real.cos a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l617_61774


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l617_61798

theorem right_triangle_side_length (A B C : ℝ × ℝ) (AB AC BC : ℝ) :
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = AB^2 →
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = AC^2 →
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = BC^2 →
  AB^2 + AC^2 = BC^2 →
  Real.cos (30 * π / 180) = (BC^2 + AC^2 - AB^2) / (2 * BC * AC) →
  AC = 18 →
  AB = 18 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l617_61798


namespace NUMINAMATH_CALUDE_right_triangle_sets_l617_61799

-- Define a function to check if three numbers can form a right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Theorem stating that the given sets of numbers satisfy or don't satisfy the right triangle condition
theorem right_triangle_sets :
  (is_right_triangle 6 8 10) ∧
  (is_right_triangle (6/5) 2 (8/5)) ∧
  (is_right_triangle 5 12 13) ∧
  ¬(is_right_triangle (Real.sqrt 8) 2 (Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l617_61799


namespace NUMINAMATH_CALUDE_trig_identity_l617_61717

theorem trig_identity (α : Real) 
  (h : Real.sin α + Real.cos α = 1/5) : 
  ((Real.sin α - Real.cos α)^2 = 49/25) ∧ 
  (Real.sin α^3 + Real.cos α^3 = 37/125) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l617_61717


namespace NUMINAMATH_CALUDE_other_cat_weight_l617_61762

/-- Represents the weights of animals in a household -/
structure HouseholdWeights where
  cat1 : ℝ
  cat2 : ℝ
  dog : ℝ

/-- Theorem stating the weight of the second cat given the conditions -/
theorem other_cat_weight (h : HouseholdWeights) 
    (h1 : h.cat1 = 10)
    (h2 : h.dog = 34)
    (h3 : h.dog = 2 * (h.cat1 + h.cat2)) : 
  h.cat2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_other_cat_weight_l617_61762


namespace NUMINAMATH_CALUDE_plane_line_perpendicular_l617_61770

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem plane_line_perpendicular 
  (m : Line) (α β γ : Plane) :
  parallel α β → parallel β γ → perpendicular m α → perpendicular m γ :=
sorry

end NUMINAMATH_CALUDE_plane_line_perpendicular_l617_61770


namespace NUMINAMATH_CALUDE_express_x_in_terms_of_y_l617_61753

theorem express_x_in_terms_of_y (x y : ℝ) (h : 2 * x - 3 * y = 7) : 
  x = 7 / 2 + 3 / 2 * y := by
  sorry

end NUMINAMATH_CALUDE_express_x_in_terms_of_y_l617_61753


namespace NUMINAMATH_CALUDE_expression_evaluation_l617_61743

theorem expression_evaluation :
  let x : ℝ := 4
  let y : ℝ := -1/2
  2 * x^2 * y - (5 * x * y^2 + 2 * (x^2 * y - 3 * x * y^2 + 1)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l617_61743


namespace NUMINAMATH_CALUDE_product_remainder_l617_61789

theorem product_remainder (a b c : ℕ) (h : a = 1125 ∧ b = 1127 ∧ c = 1129) : 
  (a * b * c) % 12 = 3 := by
sorry

end NUMINAMATH_CALUDE_product_remainder_l617_61789


namespace NUMINAMATH_CALUDE_melanie_balloons_l617_61727

theorem melanie_balloons (joan_balloons total_balloons : ℕ) 
  (h1 : joan_balloons = 40)
  (h2 : total_balloons = 81) :
  total_balloons - joan_balloons = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_melanie_balloons_l617_61727


namespace NUMINAMATH_CALUDE_total_people_in_park_l617_61711

/-- The number of lines formed by people in the park -/
def num_lines : ℕ := 4

/-- The number of people in each line -/
def people_per_line : ℕ := 8

/-- The total number of people doing gymnastics in the park -/
def total_people : ℕ := num_lines * people_per_line

theorem total_people_in_park : total_people = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_park_l617_61711


namespace NUMINAMATH_CALUDE_multiple_calculation_l617_61772

theorem multiple_calculation (n a : ℕ) (m : ℚ) 
  (h1 : n = 16) 
  (h2 : a = 12) 
  (h3 : m * n - a = 20) : 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_calculation_l617_61772


namespace NUMINAMATH_CALUDE_adam_total_figurines_l617_61732

/-- The number of figurines that can be made from one block of basswood -/
def basswood_figurines : ℕ := 3

/-- The number of figurines that can be made from one block of butternut wood -/
def butternut_figurines : ℕ := 4

/-- The number of figurines that can be made from one block of Aspen wood -/
def aspen_figurines : ℕ := 2 * basswood_figurines

/-- The number of basswood blocks Adam owns -/
def basswood_blocks : ℕ := 15

/-- The number of butternut wood blocks Adam owns -/
def butternut_blocks : ℕ := 20

/-- The number of Aspen wood blocks Adam owns -/
def aspen_blocks : ℕ := 20

/-- Theorem: Adam can make 245 figurines in total -/
theorem adam_total_figurines : 
  basswood_blocks * basswood_figurines + 
  butternut_blocks * butternut_figurines + 
  aspen_blocks * aspen_figurines = 245 := by
  sorry

end NUMINAMATH_CALUDE_adam_total_figurines_l617_61732


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l617_61741

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def binary_110011 : List Bool := [true, true, false, false, true, true]

theorem binary_110011_equals_51 : binary_to_decimal binary_110011 = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l617_61741


namespace NUMINAMATH_CALUDE_smallest_n_for_P_less_than_threshold_l617_61720

/-- Represents the number of boxes in the game --/
def total_boxes : ℕ := 2023

/-- Represents the number of boxes with 2 red marbles --/
def boxes_with_two_red : ℕ := 1012

/-- Calculates the probability of drawing a red marble from a box --/
def prob_red (box_number : ℕ) : ℚ :=
  if box_number ≤ boxes_with_two_red then
    2 / (box_number + 2)
  else
    1 / (box_number + 1)

/-- Calculates the probability of drawing a white marble from a box --/
def prob_white (box_number : ℕ) : ℚ :=
  1 - prob_red box_number

/-- Represents the probability of Isabella stopping after drawing exactly n marbles --/
noncomputable def P (n : ℕ) : ℚ :=
  sorry -- Definition of P(n) based on the game rules

/-- Theorem stating that 51 is the smallest n for which P(n) < 1/2023 --/
theorem smallest_n_for_P_less_than_threshold :
  (∀ k < 51, P k ≥ 1 / total_boxes) ∧
  P 51 < 1 / total_boxes :=
sorry

#check smallest_n_for_P_less_than_threshold

end NUMINAMATH_CALUDE_smallest_n_for_P_less_than_threshold_l617_61720


namespace NUMINAMATH_CALUDE_one_real_root_condition_l617_61708

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (k : ℝ) (x : ℝ) : Prop := lg (k * x) = 2 * lg (x + 1)

-- Theorem statement
theorem one_real_root_condition (k : ℝ) : 
  (∃! x : ℝ, equation k x) ↔ (k = 4 ∨ k < 0) :=
sorry

end NUMINAMATH_CALUDE_one_real_root_condition_l617_61708


namespace NUMINAMATH_CALUDE_total_birds_l617_61792

theorem total_birds (cardinals : ℕ) (robins : ℕ) (blue_jays : ℕ) (sparrows : ℕ) (pigeons : ℕ) (finches : ℕ) : 
  cardinals = 3 →
  robins = 4 * cardinals →
  blue_jays = 2 * cardinals →
  sparrows = 3 * cardinals + 1 →
  pigeons = 3 * blue_jays →
  finches = robins / 2 →
  cardinals + robins + blue_jays + sparrows + pigeons + finches = 55 := by
sorry

end NUMINAMATH_CALUDE_total_birds_l617_61792


namespace NUMINAMATH_CALUDE_fraction_calculation_l617_61779

theorem fraction_calculation : 
  (1/4 + 1/5) / (3/7 - 1/8) = 42/25 := by sorry

end NUMINAMATH_CALUDE_fraction_calculation_l617_61779


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l617_61748

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 1| > 1} = {x : ℝ | x < -2 ∨ x > 0} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l617_61748


namespace NUMINAMATH_CALUDE_last_three_digits_of_2_power_10000_l617_61716

theorem last_three_digits_of_2_power_10000 (h : 2^500 ≡ 1 [ZMOD 1250]) :
  2^10000 ≡ 1 [ZMOD 1000] := by
sorry

end NUMINAMATH_CALUDE_last_three_digits_of_2_power_10000_l617_61716


namespace NUMINAMATH_CALUDE_complement_of_P_in_U_l617_61707

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set P
def P : Set ℝ := {x : ℝ | x^2 - 5*x - 6 ≥ 0}

-- State the theorem
theorem complement_of_P_in_U : 
  U \ P = Set.Ioo (-1) 6 := by sorry

end NUMINAMATH_CALUDE_complement_of_P_in_U_l617_61707


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l617_61758

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l617_61758


namespace NUMINAMATH_CALUDE_sam_bank_total_l617_61702

def initial_dimes : ℕ := 9
def initial_quarters : ℕ := 5
def initial_nickels : ℕ := 3

def dad_dimes : ℕ := 7
def dad_quarters : ℕ := 2

def mom_nickels : ℕ := 1
def mom_dimes : ℕ := 2

def grandma_dollars : ℕ := 3

def sister_quarters : ℕ := 4
def sister_nickels : ℕ := 2

def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def nickel_value : ℕ := 5
def dollar_value : ℕ := 100

theorem sam_bank_total :
  (initial_dimes * dime_value +
   initial_quarters * quarter_value +
   initial_nickels * nickel_value +
   dad_dimes * dime_value +
   dad_quarters * quarter_value -
   mom_nickels * nickel_value -
   mom_dimes * dime_value +
   grandma_dollars * dollar_value +
   sister_quarters * quarter_value +
   sister_nickels * nickel_value) = 735 := by
  sorry

end NUMINAMATH_CALUDE_sam_bank_total_l617_61702


namespace NUMINAMATH_CALUDE_division_result_l617_61737

theorem division_result : (0.05 : ℚ) / (0.002 : ℚ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l617_61737


namespace NUMINAMATH_CALUDE_inequality_equivalence_l617_61797

theorem inequality_equivalence (m : ℝ) : (3 * m - 4 < 6) ↔ (m < 6) := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l617_61797


namespace NUMINAMATH_CALUDE_sum_of_p_x_coordinates_l617_61771

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (a b c : Point) : ℝ := sorry

/-- Theorem: Sum of possible x-coordinates of P -/
theorem sum_of_p_x_coordinates : ∃ (P₁ P₂ P₃ P₄ : Point),
  let Q : Point := ⟨0, 0⟩
  let R : Point := ⟨368, 0⟩
  let S₁ : Point := ⟨901, 501⟩
  let S₂ : Point := ⟨912, 514⟩
  triangleArea P₁ Q R = 4128 ∧
  triangleArea P₂ Q R = 4128 ∧
  triangleArea P₃ Q R = 4128 ∧
  triangleArea P₄ Q R = 4128 ∧
  (triangleArea P₁ R S₁ = 12384 ∨ triangleArea P₁ R S₂ = 12384) ∧
  (triangleArea P₂ R S₁ = 12384 ∨ triangleArea P₂ R S₂ = 12384) ∧
  (triangleArea P₃ R S₁ = 12384 ∨ triangleArea P₃ R S₂ = 12384) ∧
  (triangleArea P₄ R S₁ = 12384 ∨ triangleArea P₄ R S₂ = 12384) ∧
  P₁.x + P₂.x + P₃.x + P₄.x = 4000 :=
sorry

end NUMINAMATH_CALUDE_sum_of_p_x_coordinates_l617_61771


namespace NUMINAMATH_CALUDE_circle1_correct_circle2_correct_l617_61713

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 8)^2 + (y + 3)^2 = 25
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 20 = 0

-- Define the points
def point_A1 : ℝ × ℝ := (5, 1)
def point_A2 : ℝ × ℝ := (-1, 5)
def point_B2 : ℝ × ℝ := (5, 5)
def point_C2 : ℝ × ℝ := (6, -2)

-- Theorem for circle 1
theorem circle1_correct :
  circle1 (point_A1.1) (point_A1.2) ∧
  ∀ (x y : ℝ), circle1 x y → (x - 8)^2 + (y + 3)^2 = 25 := by sorry

-- Theorem for circle 2
theorem circle2_correct :
  circle2 (point_A2.1) (point_A2.2) ∧
  circle2 (point_B2.1) (point_B2.2) ∧
  circle2 (point_C2.1) (point_C2.2) ∧
  ∀ (x y : ℝ), circle2 x y → x^2 + y^2 - 4*x - 2*y - 20 = 0 := by sorry

end NUMINAMATH_CALUDE_circle1_correct_circle2_correct_l617_61713


namespace NUMINAMATH_CALUDE_solve_for_T_l617_61703

theorem solve_for_T : ∃ T : ℚ, (3/4 : ℚ) * (1/6 : ℚ) * T = (2/5 : ℚ) * (1/4 : ℚ) * 200 ∧ T = 80 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_T_l617_61703


namespace NUMINAMATH_CALUDE_income_comparison_l617_61739

/-- Given the income relationships between Juan, Tim, Mary, and Alex, prove that the sum of Mary's and Alex's incomes is 209% of Juan's income. -/
theorem income_comparison (Juan Tim Mary Alex : ℝ) 
  (h1 : Mary = 1.4 * Tim)
  (h2 : Tim = 0.6 * Juan)
  (h3 : Alex = 1.25 * Juan)
  (h4 : Alex = 0.8 * Mary) : 
  (Mary + Alex) / Juan = 2.09 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l617_61739


namespace NUMINAMATH_CALUDE_coffee_decaf_percentage_l617_61765

/-- Given initial coffee stock, percentages, and additional purchase,
    calculate the percentage of decaffeinated coffee in the new batch. -/
theorem coffee_decaf_percentage
  (initial_stock : ℝ)
  (initial_decaf_percent : ℝ)
  (additional_purchase : ℝ)
  (final_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 0.3)
  (h3 : additional_purchase = 100)
  (h4 : final_decaf_percent = 0.36)
  (h5 : initial_stock > 0)
  (h6 : additional_purchase > 0) :
  let total_stock := initial_stock + additional_purchase
  let initial_decaf := initial_stock * initial_decaf_percent
  let total_decaf := total_stock * final_decaf_percent
  let new_decaf := total_decaf - initial_decaf
  new_decaf / additional_purchase = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_coffee_decaf_percentage_l617_61765


namespace NUMINAMATH_CALUDE_fib_150_mod_5_l617_61745

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The 150th Fibonacci number modulo 5 is 0 -/
theorem fib_150_mod_5 : fib 149 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_150_mod_5_l617_61745


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l617_61706

theorem max_sum_of_squares (m n : ℕ) : 
  m ∈ Finset.range 1982 →
  n ∈ Finset.range 1982 →
  (n^2 - m*n - m^2)^2 = 1 →
  m^2 + n^2 ≤ 3524578 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l617_61706


namespace NUMINAMATH_CALUDE_ellipse_intersection_length_l617_61786

-- Define the ellipse (C)
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

-- Define the line passing through (0, 2) with slope 1
def line_l (x y : ℝ) : Prop :=
  y = x + 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem ellipse_intersection_length :
  -- Given conditions
  let F₁ : ℝ × ℝ := (-2 * Real.sqrt 2, 0)
  let F₂ : ℝ × ℝ := (2 * Real.sqrt 2, 0)
  let major_axis_length : ℝ := 6
  
  -- Prove the following
  ∀ A B : ℝ × ℝ, intersection_points A B →
    -- 1. The standard equation of the ellipse is correct
    (∀ x y : ℝ, (x^2 / 9 + y^2 = 1) ↔ ellipse_C x y) ∧
    -- 2. The length of AB is 6√3/5
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6 * Real.sqrt 3 / 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ellipse_intersection_length_l617_61786


namespace NUMINAMATH_CALUDE_dima_grade_and_instrument_l617_61795

-- Define the students
inductive Student : Type
| Vasya : Student
| Dima : Student
| Kolya : Student
| Sergey : Student

-- Define the grades
inductive Grade : Type
| Fifth : Grade
| Sixth : Grade
| Seventh : Grade
| Eighth : Grade

-- Define the instruments
inductive Instrument : Type
| Saxophone : Instrument
| Keyboard : Instrument
| Drums : Instrument
| Guitar : Instrument

-- Define the assignment of grades and instruments to students
def grade_assignment : Student → Grade := sorry
def instrument_assignment : Student → Instrument := sorry

-- State the theorem
theorem dima_grade_and_instrument :
  (instrument_assignment Student.Vasya = Instrument.Saxophone) ∧
  (grade_assignment Student.Vasya ≠ Grade.Eighth) ∧
  (∃ s, grade_assignment s = Grade.Sixth ∧ instrument_assignment s = Instrument.Keyboard) ∧
  (∀ s, instrument_assignment s = Instrument.Drums → s ≠ Student.Dima) ∧
  (instrument_assignment Student.Sergey ≠ Instrument.Keyboard) ∧
  (grade_assignment Student.Sergey ≠ Grade.Fifth) ∧
  (grade_assignment Student.Dima ≠ Grade.Sixth) ∧
  (∀ s, instrument_assignment s = Instrument.Drums → grade_assignment s ≠ Grade.Eighth) →
  (grade_assignment Student.Dima = Grade.Eighth ∧ instrument_assignment Student.Dima = Instrument.Guitar) :=
by sorry


end NUMINAMATH_CALUDE_dima_grade_and_instrument_l617_61795


namespace NUMINAMATH_CALUDE_batsman_average_l617_61746

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat

/-- Calculates the average score of a batsman -/
def average (b : Batsman) : Nat :=
  b.totalRuns / b.innings

/-- Theorem: Given the conditions, prove that the batsman's average after 10 innings is 33 -/
theorem batsman_average (b : Batsman) 
  (h1 : b.innings = 10)
  (h2 : b.totalRuns = (average b * 9) + 60)
  (h3 : average { innings := b.innings, totalRuns := b.totalRuns, averageIncrease := b.averageIncrease } = 
        average { innings := b.innings - 1, totalRuns := b.totalRuns - 60, averageIncrease := b.averageIncrease } + 3) :
  average b = 33 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_l617_61746


namespace NUMINAMATH_CALUDE_triangle_inequality_l617_61721

/-- Given a triangle with side lengths a, b, c, and semiperimeter p, 
    prove that 2√((p-b)(p-c)) ≤ a. -/
theorem triangle_inequality (a b c p : ℝ) 
    (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
    (h_semiperimeter : p = (a + b + c) / 2) : 
  2 * Real.sqrt ((p - b) * (p - c)) ≤ a := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l617_61721


namespace NUMINAMATH_CALUDE_money_distribution_l617_61724

theorem money_distribution (total : ℝ) (p q r : ℝ) : 
  p + q + r = total →
  p / q = 3 / 7 →
  q / r = 7 / 12 →
  q - p = 2400 →
  r - q = 3000 :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l617_61724


namespace NUMINAMATH_CALUDE_parabola_constant_term_l617_61751

theorem parabola_constant_term (p q : ℝ) : 
  (∀ x y : ℝ, y = x^2 + p*x + q → 
    ((x = 3 ∧ y = 4) ∨ (x = 5 ∧ y = 4))) → 
  q = 19 := by
sorry

end NUMINAMATH_CALUDE_parabola_constant_term_l617_61751


namespace NUMINAMATH_CALUDE_triangle_problem_l617_61788

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and vectors m and n, prove that B = π/3 and the maximum area is √3 --/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  let m : ℝ × ℝ := (2 * Real.sin B, -Real.sqrt 3)
  let n : ℝ × ℝ := (Real.cos (2 * B), 2 * (Real.cos (B / 2))^2 - 1)
  b = 2 →
  (∃ (k : ℝ), m.1 = k * n.1 ∧ m.2 = k * n.2) →
  B = π / 3 ∧
  (∀ (S : ℝ), S = 1/2 * a * c * Real.sin B → S ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l617_61788


namespace NUMINAMATH_CALUDE_average_bull_weight_l617_61704

/-- Represents a section of the farm with a ratio of cows to bulls -/
structure FarmSection where
  cows : ℕ
  bulls : ℕ

/-- Represents the farm with its sections and total cattle -/
structure Farm where
  sectionA : FarmSection
  sectionB : FarmSection
  sectionC : FarmSection
  totalCattle : ℕ
  totalBullWeight : ℕ

def farm : Farm := {
  sectionA := { cows := 7, bulls := 21 },
  sectionB := { cows := 5, bulls := 15 },
  sectionC := { cows := 3, bulls := 9 },
  totalCattle := 1220,
  totalBullWeight := 200000
}

theorem average_bull_weight (f : Farm) :
  f = farm →
  (f.totalBullWeight : ℚ) / (((f.sectionA.bulls + f.sectionB.bulls + f.sectionC.bulls) * f.totalCattle) / (f.sectionA.cows + f.sectionA.bulls + f.sectionB.cows + f.sectionB.bulls + f.sectionC.cows + f.sectionC.bulls)) = 218579 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_average_bull_weight_l617_61704


namespace NUMINAMATH_CALUDE_no_real_solutions_l617_61738

theorem no_real_solutions :
  ¬∃ y : ℝ, (3 * y - 4)^2 + 4 = -(y + 3) := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l617_61738


namespace NUMINAMATH_CALUDE_certain_number_proof_l617_61783

theorem certain_number_proof (x : ℝ) : 144 / x = 14.4 / 0.0144 → x = 0.144 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l617_61783


namespace NUMINAMATH_CALUDE_number_problem_l617_61742

theorem number_problem (n : ℝ) : (0.6 * (3/5) * n = 36) → n = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l617_61742


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l617_61775

/-- 
Given an arithmetic sequence {a_n} with S_n as the sum of its first n terms,
prove that if S_6 = 24 and S_9 = 63, then a_4 = 5.
-/
theorem arithmetic_sequence_fourth_term 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) 
  (h_S6 : S 6 = 24) 
  (h_S9 : S 9 = 63) : 
  a 4 = 5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l617_61775


namespace NUMINAMATH_CALUDE_not_all_countries_have_complete_systems_l617_61759

/-- Represents a country's internet regulation system -/
structure InternetRegulation where
  country : String
  hasCompleteSystem : Bool

/-- Information about internet regulation systems in different countries -/
def countryRegulations : List InternetRegulation := [
  { country := "United States", hasCompleteSystem := false },
  { country := "United Kingdom", hasCompleteSystem := false },
  { country := "Russia", hasCompleteSystem := true }
]

/-- Theorem stating that not all countries (US, UK, and Russia) have complete internet regulation systems -/
theorem not_all_countries_have_complete_systems : 
  ¬ (∀ c ∈ countryRegulations, c.hasCompleteSystem = true) := by
  sorry

end NUMINAMATH_CALUDE_not_all_countries_have_complete_systems_l617_61759


namespace NUMINAMATH_CALUDE_product_real_implies_a_value_l617_61712

theorem product_real_implies_a_value (z₁ z₂ : ℂ) (a : ℝ) :
  z₁ = 2 + I →
  z₂ = 1 + a * I →
  (z₁ * z₂).im = 0 →
  a = -1/2 := by sorry

end NUMINAMATH_CALUDE_product_real_implies_a_value_l617_61712


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l617_61730

/-- The surface area of a rectangular solid. -/
def surface_area (length width depth : ℝ) : ℝ :=
  2 * (length * width + length * depth + width * depth)

/-- Theorem: The surface area of a rectangular solid with length 10 meters, width 9 meters, 
    and depth 6 meters is 408 square meters. -/
theorem rectangular_solid_surface_area :
  surface_area 10 9 6 = 408 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l617_61730


namespace NUMINAMATH_CALUDE_possible_sets_B_l617_61719

theorem possible_sets_B (A B : Set Int) : 
  A = {-1} → A ∪ B = {-1, 3} → (B = {3} ∨ B = {-1, 3}) := by
  sorry

end NUMINAMATH_CALUDE_possible_sets_B_l617_61719


namespace NUMINAMATH_CALUDE_divisible_by_seventeen_l617_61715

theorem divisible_by_seventeen (n : ℕ) : 
  (6^(2*n) + 2^(n+2) + 12 * 2^n) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_seventeen_l617_61715


namespace NUMINAMATH_CALUDE_church_full_capacity_l617_61731

/-- Calculates the total number of people that can be seated in a church with three sections -/
def church_capacity (section1_rows section1_chairs_per_row section2_rows section2_chairs_per_row section3_rows section3_chairs_per_row : ℕ) : ℕ :=
  section1_rows * section1_chairs_per_row +
  section2_rows * section2_chairs_per_row +
  section3_rows * section3_chairs_per_row

/-- Theorem stating that the church capacity is 490 given the specified section configurations -/
theorem church_full_capacity :
  church_capacity 15 8 20 6 25 10 = 490 := by
  sorry

end NUMINAMATH_CALUDE_church_full_capacity_l617_61731


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l617_61722

theorem smallest_n_congruence :
  ∃ (n : ℕ), n > 0 ∧ (3 * n) % 26 = 1367 % 26 ∧
  ∀ (m : ℕ), m > 0 ∧ (3 * m) % 26 = 1367 % 26 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l617_61722


namespace NUMINAMATH_CALUDE_sum_of_simplified_fraction_l617_61794

-- Define the repeating decimal 0.̅4̅5̅
def repeating_decimal : ℚ := 45 / 99

-- Define the function to simplify a fraction
def simplify (q : ℚ) : ℚ := q

-- Define the function to sum numerator and denominator
def sum_num_denom (q : ℚ) : ℕ := q.num.natAbs + q.den

-- Theorem statement
theorem sum_of_simplified_fraction :
  sum_num_denom (simplify repeating_decimal) = 16 := by sorry

end NUMINAMATH_CALUDE_sum_of_simplified_fraction_l617_61794


namespace NUMINAMATH_CALUDE_pizza_special_pricing_l617_61768

/-- Represents the cost calculation for pizzas with special pricing --/
def pizza_cost (standard_price : ℕ) (triple_cheese_count : ℕ) (meat_lovers_count : ℕ) : ℕ :=
  let triple_cheese_cost := (triple_cheese_count / 2) * standard_price
  let meat_lovers_cost := ((meat_lovers_count + 2) / 3 * 2) * standard_price
  triple_cheese_cost + meat_lovers_cost

/-- Theorem stating the total cost of pizzas under special pricing --/
theorem pizza_special_pricing :
  pizza_cost 5 10 9 = 55 := by
  sorry


end NUMINAMATH_CALUDE_pizza_special_pricing_l617_61768


namespace NUMINAMATH_CALUDE_white_coincide_pairs_l617_61729

-- Define the structure of our figure
structure Figure where
  red_triangles : ℕ
  blue_triangles : ℕ
  white_triangles : ℕ
  red_coincide : ℕ
  blue_coincide : ℕ
  red_white_pairs : ℕ

-- Define our specific figure
def our_figure : Figure :=
  { red_triangles := 4
  , blue_triangles := 6
  , white_triangles := 10
  , red_coincide := 3
  , blue_coincide := 4
  , red_white_pairs := 3 }

-- Theorem statement
theorem white_coincide_pairs (f : Figure) (h : f = our_figure) : 
  ∃ (white_coincide : ℕ), white_coincide = 3 := by
  sorry

end NUMINAMATH_CALUDE_white_coincide_pairs_l617_61729


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_reciprocals_of_first_four_primes_l617_61757

-- Define the first four prime numbers
def first_four_primes : List ℕ := [2, 3, 5, 7]

-- Define a function to calculate the reciprocal of a natural number
def reciprocal (n : ℕ) : ℚ := 1 / n

-- Define the arithmetic mean of a list of rational numbers
def arithmetic_mean (list : List ℚ) : ℚ := (list.sum) / list.length

-- Theorem statement
theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean (first_four_primes.map reciprocal) = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_reciprocals_of_first_four_primes_l617_61757


namespace NUMINAMATH_CALUDE_hiking_duration_is_six_hours_l617_61736

/-- Represents the hiking scenario with given initial weights and consumption rates. --/
structure HikingScenario where
  initialWater : ℝ
  initialFood : ℝ
  initialGear : ℝ
  waterConsumptionRate : ℝ
  foodConsumptionRate : ℝ

/-- Calculates the remaining weight after a given number of hours. --/
def remainingWeight (scenario : HikingScenario) (hours : ℝ) : ℝ :=
  scenario.initialWater + scenario.initialFood + scenario.initialGear -
  scenario.waterConsumptionRate * hours -
  scenario.foodConsumptionRate * hours

/-- Theorem stating that under the given conditions, the hiking duration is 6 hours. --/
theorem hiking_duration_is_six_hours (scenario : HikingScenario)
  (h1 : scenario.initialWater = 20)
  (h2 : scenario.initialFood = 10)
  (h3 : scenario.initialGear = 20)
  (h4 : scenario.waterConsumptionRate = 2)
  (h5 : scenario.foodConsumptionRate = 2/3)
  (h6 : remainingWeight scenario 6 = 34) :
  ∃ (h : ℝ), h = 6 ∧ remainingWeight scenario h = 34 := by
  sorry


end NUMINAMATH_CALUDE_hiking_duration_is_six_hours_l617_61736


namespace NUMINAMATH_CALUDE_cost_of_one_each_l617_61767

theorem cost_of_one_each (x y z : ℝ) 
  (eq1 : 3 * x + 7 * y + z = 325)
  (eq2 : 4 * x + 10 * y + z = 410) :
  x + y + z = 155 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_one_each_l617_61767


namespace NUMINAMATH_CALUDE_tech_ownership_1995_l617_61726

/-- The percentage of families owning computers, tablets, and smartphones in City X in 1995 -/
def tech_ownership (pc_1992 : ℝ) (pc_increase_1993 : ℝ) (family_increase_1993 : ℝ)
                   (tablet_adoption_1994 : ℝ) (smartphone_adoption_1995 : ℝ) : ℝ :=
  let pc_1993 := pc_1992 * (1 + pc_increase_1993)
  let pc_tablet_1994 := pc_1993 * tablet_adoption_1994
  pc_tablet_1994 * smartphone_adoption_1995

theorem tech_ownership_1995 :
  tech_ownership 0.6 0.5 0.03 0.4 0.3 = 0.108 := by
  sorry

end NUMINAMATH_CALUDE_tech_ownership_1995_l617_61726


namespace NUMINAMATH_CALUDE_total_monthly_time_is_200_l617_61750

/-- Represents the days of the week -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Returns true if the given day is a weekday -/
def is_weekday (d : Day) : Bool :=
  match d with
  | Day.Saturday | Day.Sunday => false
  | _ => true

/-- Returns the amount of TV time for a given day -/
def tv_time (d : Day) : Nat :=
  match d with
  | Day.Monday | Day.Wednesday | Day.Friday => 4
  | Day.Tuesday | Day.Thursday => 3
  | Day.Saturday | Day.Sunday => 5

/-- Returns the amount of piano practice time for a given day -/
def piano_time (d : Day) : Nat :=
  if is_weekday d then 2 else 3

/-- Calculates the total weekly TV time -/
def total_weekly_tv_time : Nat :=
  (tv_time Day.Monday) + (tv_time Day.Tuesday) + (tv_time Day.Wednesday) +
  (tv_time Day.Thursday) + (tv_time Day.Friday) + (tv_time Day.Saturday) +
  (tv_time Day.Sunday)

/-- Calculates the average daily TV time -/
def avg_daily_tv_time : Nat :=
  total_weekly_tv_time / 7

/-- Calculates the total weekly video game time -/
def total_weekly_video_game_time : Nat :=
  (avg_daily_tv_time / 2) * 3

/-- Calculates the total weekly piano time -/
def total_weekly_piano_time : Nat :=
  (piano_time Day.Monday) + (piano_time Day.Tuesday) + (piano_time Day.Wednesday) +
  (piano_time Day.Thursday) + (piano_time Day.Friday) + (piano_time Day.Saturday) +
  (piano_time Day.Sunday)

/-- Calculates the total weekly time for all activities -/
def total_weekly_time : Nat :=
  total_weekly_tv_time + total_weekly_video_game_time + total_weekly_piano_time

/-- The main theorem stating that the total monthly time is 200 hours -/
theorem total_monthly_time_is_200 :
  total_weekly_time * 4 = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_monthly_time_is_200_l617_61750


namespace NUMINAMATH_CALUDE_convention_handshakes_l617_61728

/-- Represents the convention of twins and triplets --/
structure Convention where
  twin_sets : ℕ
  triplet_sets : ℕ

/-- Calculates the total number of handshakes in the convention --/
def total_handshakes (c : Convention) : ℕ :=
  let twin_count := c.twin_sets * 2
  let triplet_count := c.triplet_sets * 3
  let twin_handshakes := (twin_count * (twin_count - 2)) / 2
  let triplet_handshakes := (triplet_count * (triplet_count - 3)) / 2
  let twin_to_triplet := twin_count * (triplet_count / 2)
  twin_handshakes + triplet_handshakes + twin_to_triplet

/-- The theorem stating that the total number of handshakes in the given convention is 354 --/
theorem convention_handshakes :
  total_handshakes ⟨10, 4⟩ = 354 := by sorry

end NUMINAMATH_CALUDE_convention_handshakes_l617_61728


namespace NUMINAMATH_CALUDE_remainder_problem_l617_61701

theorem remainder_problem (k : Nat) (h : k > 0) :
  (90 % (k^2) = 6) → (130 % k = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l617_61701
