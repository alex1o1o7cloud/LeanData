import Mathlib

namespace wheel_of_fortune_probability_l3074_307421

theorem wheel_of_fortune_probability (p_D p_E p_F p_G : ℚ) : 
  p_D = 3/8 → p_E = 1/4 → p_G = 1/8 → p_D + p_E + p_F + p_G = 1 → p_F = 1/4 := by
  sorry

end wheel_of_fortune_probability_l3074_307421


namespace root_in_interval_l3074_307437

theorem root_in_interval (a : ℤ) : 
  (∃ x : ℝ, x > a ∧ x < a + 1 ∧ Real.log x + x - 4 = 0) → a = 2 := by
sorry

end root_in_interval_l3074_307437


namespace distinct_permutations_eq_twelve_l3074_307423

/-- The number of distinct permutations of the multiset {2, 3, 3, 9} -/
def distinct_permutations : ℕ :=
  Nat.factorial 4 / Nat.factorial 2

/-- Theorem stating that the number of distinct permutations of the multiset {2, 3, 3, 9} is 12 -/
theorem distinct_permutations_eq_twelve : distinct_permutations = 12 := by
  sorry

end distinct_permutations_eq_twelve_l3074_307423


namespace existence_of_pair_l3074_307436

theorem existence_of_pair (x : Fin 670 → ℝ)
  (h_positive : ∀ i, 0 < x i)
  (h_less_than_one : ∀ i, x i < 1)
  (h_distinct : ∀ i j, i ≠ j → x i ≠ x j) :
  ∃ i j, i ≠ j ∧ 0 < x i * x j * (x j - x i) ∧ x i * x j * (x j - x i) < 1 / 2007 := by
  sorry

end existence_of_pair_l3074_307436


namespace undefined_expression_expression_undefined_iff_x_eq_12_l3074_307417

theorem undefined_expression (x : ℝ) : 
  (x^2 - 24*x + 144 = 0) ↔ (x = 12) := by sorry

theorem expression_undefined_iff_x_eq_12 :
  ∀ x : ℝ, (∃ y : ℝ, (3*x^3 - 5*x + 2) / (x^2 - 24*x + 144) = y) ↔ (x ≠ 12) := by sorry

end undefined_expression_expression_undefined_iff_x_eq_12_l3074_307417


namespace smallest_prime_with_reverse_composite_l3074_307408

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, 1 < m ∧ m < n ∧ m ∣ n

theorem smallest_prime_with_reverse_composite : 
  ∀ n : ℕ, 30 ≤ n ∧ n < 41 →
    ¬(is_prime n ∧ 
      is_composite (reverse_digits n) ∧ 
      ∃ m : ℕ, m ≠ 7 ∧ m > 1 ∧ m ∣ reverse_digits n) →
  is_prime 41 ∧ 
  is_composite (reverse_digits 41) ∧ 
  ∃ m : ℕ, m ≠ 7 ∧ m > 1 ∧ m ∣ reverse_digits 41 :=
sorry

end smallest_prime_with_reverse_composite_l3074_307408


namespace trajectory_of_point_M_l3074_307418

/-- The trajectory of point M given the specified conditions -/
theorem trajectory_of_point_M :
  ∀ (M : ℝ × ℝ),
  let A : ℝ × ℝ := (0, -1)
  let B : ℝ × ℝ := (M.1, -3)
  let O : ℝ × ℝ := (0, 0)
  -- MB parallel to OA
  (∃ k : ℝ, B.1 - M.1 = k * A.1 ∧ B.2 - M.2 = k * A.2) →
  -- MA • AB = MB • BA
  ((A.1 - M.1) * (B.1 - A.1) + (A.2 - M.2) * (B.2 - A.2) =
   (B.1 - M.1) * (A.1 - B.1) + (B.2 - M.2) * (A.2 - B.2)) →
  -- Trajectory equation
  M.2 = (1/4) * M.1^2 - 2 := by
sorry

end trajectory_of_point_M_l3074_307418


namespace function_inequality_l3074_307496

/-- Given functions f and g, prove that g(x) > f(x) + kx - 1 for all x > 0 and a ∈ (0, e^2/2] -/
theorem function_inequality (k : ℝ) :
  ∀ (x a : ℝ), x > 0 → 0 < a → a ≤ Real.exp 2 / 2 →
  (Real.exp x) / (a * x) > Real.log x - k * x + 1 + k * x - 1 := by
  sorry


end function_inequality_l3074_307496


namespace afternoon_eggs_calculation_l3074_307467

theorem afternoon_eggs_calculation (total_eggs day_eggs morning_eggs : ℕ) 
  (h1 : total_eggs = 1339)
  (h2 : morning_eggs = 816)
  (h3 : day_eggs = total_eggs - morning_eggs) : 
  day_eggs = 523 := by
  sorry

end afternoon_eggs_calculation_l3074_307467


namespace shortest_side_length_l3074_307483

/-- Triangle with specific properties -/
structure SpecialTriangle where
  -- Base of the triangle
  base : ℝ
  -- One base angle in radians
  baseAngle : ℝ
  -- Sum of the other two sides
  sumOtherSides : ℝ
  -- Conditions
  base_positive : base > 0
  baseAngle_in_range : 0 < baseAngle ∧ baseAngle < π
  sumOtherSides_positive : sumOtherSides > 0

/-- The length of the shortest side in the special triangle -/
def shortestSide (t : SpecialTriangle) : ℝ := sorry

/-- Theorem stating the length of the shortest side in the specific triangle -/
theorem shortest_side_length (t : SpecialTriangle) 
  (h1 : t.base = 80)
  (h2 : t.baseAngle = π / 3)  -- 60° in radians
  (h3 : t.sumOtherSides = 90) :
  shortestSide t = 40 := by sorry

end shortest_side_length_l3074_307483


namespace log_base_is_two_range_of_m_l3074_307401

noncomputable section

-- Define the logarithm function with base a
def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := log_base a x

-- Theorem 1: If f(x) = log_a(x), a > 0, a ≠ 1, and f(2) = 1, then f(x) = log_2(x)
theorem log_base_is_two (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 1) :
  ∀ x > 0, f a x = log_base 2 x :=
sorry

-- Theorem 2: For f(x) = log_2(x), the set of real numbers m satisfying f(m^2 - m) < 1 is (-1,0) ∪ (1,2)
theorem range_of_m (m : ℝ) :
  log_base 2 (m^2 - m) < 1 ↔ (m > -1 ∧ m < 0) ∨ (m > 1 ∧ m < 2) :=
sorry

end

end log_base_is_two_range_of_m_l3074_307401


namespace inequality_solution_l3074_307415

theorem inequality_solution (x : ℝ) : 
  (x^3 - 3*x^2 + 2*x) / (x^2 - 2*x + 1) ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio 1 ∪ Set.Ici 2 :=
by sorry

end inequality_solution_l3074_307415


namespace solve_movie_problem_l3074_307407

def movie_problem (ticket_count : ℕ) (rental_cost bought_cost total_cost : ℚ) : Prop :=
  ticket_count = 2 ∧
  rental_cost = 1.59 ∧
  bought_cost = 13.95 ∧
  total_cost = 36.78 ∧
  ∃ (ticket_price : ℚ),
    ticket_price * ticket_count + rental_cost + bought_cost = total_cost ∧
    ticket_price = 10.62

theorem solve_movie_problem :
  ∃ (ticket_count : ℕ) (rental_cost bought_cost total_cost : ℚ),
    movie_problem ticket_count rental_cost bought_cost total_cost :=
  sorry

end solve_movie_problem_l3074_307407


namespace ratio_evaluation_l3074_307487

theorem ratio_evaluation : 
  (5^3003 * 2^3005) / 10^3004 = 2/5 := by
sorry

end ratio_evaluation_l3074_307487


namespace simplify_and_ratio_l3074_307469

theorem simplify_and_ratio : 
  ∀ (k : ℝ), 
  (6 * k + 18) / 6 = k + 3 ∧ 
  ∃ (a b : ℤ), k + 3 = a * k + b ∧ (a : ℝ) / (b : ℝ) = 1 / 3 := by
  sorry

end simplify_and_ratio_l3074_307469


namespace intersecting_lines_equality_l3074_307498

/-- Given two linear functions y = ax + b and y = cx + d that intersect at (1, 0),
    prove that a^3 + c^2 = d^2 - b^3 -/
theorem intersecting_lines_equality (a b c d : ℝ) 
  (h1 : a * 1 + b = 0)  -- y = ax + b passes through (1, 0)
  (h2 : c * 1 + d = 0)  -- y = cx + d passes through (1, 0)
  : a^3 + c^2 = d^2 - b^3 := by
  sorry

end intersecting_lines_equality_l3074_307498


namespace division_of_fractions_l3074_307412

theorem division_of_fractions :
  (-4 / 5) / (8 / 25) = -5 / 2 := by
  sorry

end division_of_fractions_l3074_307412


namespace number_problem_l3074_307405

theorem number_problem :
  ∃ (x : ℝ), ∃ (y : ℝ), 0.5 * x = y + 20 ∧ x - 2 * y = 40 ∧ x = 40 := by
  sorry

end number_problem_l3074_307405


namespace smallest_reducible_n_is_correct_l3074_307430

/-- The smallest positive integer n for which (n-17)/(6n+8) is non-zero and reducible -/
def smallest_reducible_n : ℕ := 127

/-- A fraction is reducible if the GCD of its numerator and denominator is greater than 1 -/
def is_reducible (n : ℕ) : Prop :=
  Nat.gcd (n - 17) (6 * n + 8) > 1

theorem smallest_reducible_n_is_correct :
  (∀ k : ℕ, k > 0 ∧ k < smallest_reducible_n → ¬(is_reducible k)) ∧
  (smallest_reducible_n > 0) ∧
  (is_reducible smallest_reducible_n) :=
sorry

end smallest_reducible_n_is_correct_l3074_307430


namespace unreachable_y_value_l3074_307449

theorem unreachable_y_value (x : ℝ) (h : x ≠ -5/4) :
  ¬∃y : ℝ, y = -3/4 ∧ y = (2 - 3*x) / (4*x + 5) :=
by sorry

end unreachable_y_value_l3074_307449


namespace largest_prime_divisor_of_2102012_base7_l3074_307443

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- Gets the largest prime divisor of a number -/
def largestPrimeDivisor (n : ℕ) : ℕ := sorry

theorem largest_prime_divisor_of_2102012_base7 :
  largestPrimeDivisor (base7ToBase10 2102012) = 79 := by sorry

end largest_prime_divisor_of_2102012_base7_l3074_307443


namespace pets_remaining_l3074_307413

theorem pets_remaining (initial_puppies initial_kittens puppies_sold kittens_sold : ℕ) :
  initial_puppies = 7 →
  initial_kittens = 6 →
  puppies_sold = 2 →
  kittens_sold = 3 →
  initial_puppies + initial_kittens - (puppies_sold + kittens_sold) = 8 :=
by
  sorry

end pets_remaining_l3074_307413


namespace product_evaluation_l3074_307471

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end product_evaluation_l3074_307471


namespace quiz_average_after_drop_l3074_307425

theorem quiz_average_after_drop (n : ℕ) (initial_avg : ℚ) (dropped_score : ℕ) :
  n = 16 →
  initial_avg = 60.5 →
  dropped_score = 8 →
  let total_score := n * initial_avg
  let remaining_score := total_score - dropped_score
  let new_avg := remaining_score / (n - 1)
  new_avg = 64 := by sorry

end quiz_average_after_drop_l3074_307425


namespace angle_A_is_pi_over_four_angle_C_is_5pi_over_12_l3074_307472

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  -- Ensure all sides and angles are positive
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  hA : 0 < A ∧ A < π
  hB : 0 < B ∧ B < π
  hC : 0 < C ∧ C < π
  -- Ensure the sum of angles is π
  angle_sum : A + B + C = π
  -- Area formula
  area_formula : S = (1/2) * b * c * Real.sin A

-- Theorem 1
theorem angle_A_is_pi_over_four (t : Triangle) (h : t.a^2 + 4*t.S = t.b^2 + t.c^2) :
  t.A = π/4 := by sorry

-- Theorem 2
theorem angle_C_is_5pi_over_12 (t : Triangle) 
  (h1 : t.a^2 + 4*t.S = t.b^2 + t.c^2) (h2 : t.a = Real.sqrt 2) (h3 : t.b = Real.sqrt 3) :
  t.C = 5*π/12 := by sorry

end angle_A_is_pi_over_four_angle_C_is_5pi_over_12_l3074_307472


namespace proposition_count_l3074_307462

theorem proposition_count : 
  let prop1 := ∀ x : ℝ, x^2 + x + 1 ≥ 0
  let prop2 := ∀ x y : ℝ, (x + y ≠ 3) → (x ≠ 2 ∨ y ≠ 1)
  let prop3 := 
    let slope : ℝ := 1.23
    let center : ℝ × ℝ := (4, 5)
    (5 : ℝ) = slope * (4 : ℝ) + 0.08
  let prop4 := 
    ∀ m : ℝ, (m = 3 ↔ 
      ∀ x y : ℝ, ((m + 3) * x + m * y - 2 = 0 → m * x - 6 * y + 5 = 0) ∧
                 (m * x - 6 * y + 5 = 0 → (m + 3) * x + m * y - 2 = 0))
  (prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ ¬prop4) ∨
  (¬prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4) ∨
  (¬prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4) ∨
  (¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4) ∨
  (prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4) ∨
  (prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4) ∨
  (prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4) ∨
  (¬prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4) ∨
  (¬prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4) ∨
  (¬prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4) := by
  sorry

end proposition_count_l3074_307462


namespace container_volume_tripled_l3074_307402

theorem container_volume_tripled (original_volume : ℝ) (h : original_volume = 2) :
  let new_volume := original_volume * 3 * 3 * 3
  new_volume = 54 := by
sorry

end container_volume_tripled_l3074_307402


namespace only_event1_is_random_l3074_307457

/-- Represents an event in a probability space -/
structure Event where
  description : String

/-- Defines what it means for an event to be random -/
def isRandomEvent (e : Event) : Prop :=
  sorry  -- Definition of random event

/-- Event 1: Tossing a coin twice in a row and getting heads both times -/
def event1 : Event := ⟨"Tossing a coin twice in a row and getting heads both times"⟩

/-- Event 2: Opposite charges attract each other -/
def event2 : Event := ⟨"Opposite charges attract each other"⟩

/-- Event 3: Water freezes at 1°C under standard atmospheric pressure -/
def event3 : Event := ⟨"Water freezes at 1°C under standard atmospheric pressure"⟩

/-- Theorem: Only event1 is a random event among the given events -/
theorem only_event1_is_random :
  isRandomEvent event1 ∧ ¬isRandomEvent event2 ∧ ¬isRandomEvent event3 :=
by
  sorry


end only_event1_is_random_l3074_307457


namespace total_bottles_l3074_307434

theorem total_bottles (juice : ℕ) (water : ℕ) : 
  juice = 34 → 
  water = (3 * juice) / 2 + 3 → 
  juice + water = 88 := by
sorry

end total_bottles_l3074_307434


namespace conference_children_count_l3074_307435

theorem conference_children_count :
  let total_men : ℕ := 700
  let total_women : ℕ := 500
  let indian_men_percentage : ℚ := 20 / 100
  let indian_women_percentage : ℚ := 40 / 100
  let indian_children_percentage : ℚ := 10 / 100
  let non_indian_percentage : ℚ := 79 / 100
  ∃ (total_children : ℕ),
    (indian_men_percentage * total_men +
     indian_women_percentage * total_women +
     indian_children_percentage * total_children : ℚ) =
    ((1 - non_indian_percentage) * (total_men + total_women + total_children) : ℚ) ∧
    total_children = 800 :=
by sorry

end conference_children_count_l3074_307435


namespace alyosha_age_claim_possible_l3074_307429

-- Define a structure for dates
structure Date :=
  (year : ℕ)
  (month : ℕ)
  (day : ℕ)

-- Define a structure for a person's age and birthday
structure Person :=
  (birthday : Date)
  (current_date : Date)

def age (p : Person) : ℕ :=
  p.current_date.year - p.birthday.year

def is_birthday (p : Person) : Prop :=
  p.birthday.month = p.current_date.month ∧ p.birthday.day = p.current_date.day

-- Define Alyosha
def alyosha (birthday : Date) : Person :=
  { birthday := birthday,
    current_date := ⟨2024, 1, 1⟩ }  -- Assuming current year is 2024

-- Theorem statement
theorem alyosha_age_claim_possible :
  ∃ (birthday : Date),
    age (alyosha birthday) = 11 ∧
    age { birthday := birthday, current_date := ⟨2023, 12, 30⟩ } = 9 ∧
    age { birthday := birthday, current_date := ⟨2025, 1, 1⟩ } = 12 ↔
    birthday = ⟨2013, 12, 31⟩ :=
sorry

end alyosha_age_claim_possible_l3074_307429


namespace line_point_x_coordinate_l3074_307480

/-- Theorem: For a line passing through points (x₁, -4) and (5, 0.8) with slope 0.8, x₁ = -1 -/
theorem line_point_x_coordinate (x₁ : ℝ) : 
  let y₁ : ℝ := -4
  let x₂ : ℝ := 5
  let y₂ : ℝ := 0.8
  let k : ℝ := 0.8
  (y₂ - y₁) / (x₂ - x₁) = k → x₁ = -1 := by
  sorry

end line_point_x_coordinate_l3074_307480


namespace quadratic_abs_equivalence_l3074_307438

theorem quadratic_abs_equivalence (a : ℝ) : a^2 + 4*a - 5 > 0 ↔ |a + 2| > 3 := by
  sorry

end quadratic_abs_equivalence_l3074_307438


namespace inequality_implies_sum_nonnegative_l3074_307494

theorem inequality_implies_sum_nonnegative (a b : ℝ) :
  Real.exp a + Real.pi ^ b ≥ Real.exp (-b) + Real.pi ^ (-a) → a + b ≥ 0 := by
  sorry

end inequality_implies_sum_nonnegative_l3074_307494


namespace probability_theorem_l3074_307490

def total_shoes : ℕ := 28
def black_pairs : ℕ := 7
def brown_pairs : ℕ := 4
def gray_pairs : ℕ := 2
def white_pairs : ℕ := 1

def probability_same_color_left_right : ℚ :=
  (black_pairs * 2 / total_shoes) * (black_pairs / (total_shoes - 1)) +
  (brown_pairs * 2 / total_shoes) * (brown_pairs / (total_shoes - 1)) +
  (gray_pairs * 2 / total_shoes) * (gray_pairs / (total_shoes - 1)) +
  (white_pairs * 2 / total_shoes) * (white_pairs / (total_shoes - 1))

theorem probability_theorem :
  probability_same_color_left_right = 35 / 189 := by
  sorry

end probability_theorem_l3074_307490


namespace supplement_of_complement_of_sixty_degrees_l3074_307414

/-- The degree measure of the supplement of the complement of a 60-degree angle is 150°. -/
theorem supplement_of_complement_of_sixty_degrees : 
  let original_angle : ℝ := 60
  let complement := 90 - original_angle
  let supplement := 180 - complement
  supplement = 150 := by sorry

end supplement_of_complement_of_sixty_degrees_l3074_307414


namespace computation_proof_l3074_307454

theorem computation_proof : 24 * ((150 / 3) - (36 / 6) + (7.2 / 0.4) + 2) = 1536 := by
  sorry

end computation_proof_l3074_307454


namespace cone_angle_l3074_307466

/-- Given a cone where the ratio of its lateral surface area to the area of the section through its axis
    is 2√3π/3, the angle between its generatrix and axis is π/6. -/
theorem cone_angle (r h l : ℝ) (θ : ℝ) : 
  r > 0 → h > 0 → l > 0 →
  (π * r * l) / (r * h) = 2 * Real.sqrt 3 * π / 3 →
  l = Real.sqrt ((r^2) + (h^2)) →
  θ = Real.arccos (h / l) →
  θ = π / 6 := by
  sorry

#check cone_angle

end cone_angle_l3074_307466


namespace no_real_roots_l3074_307404

theorem no_real_roots : ¬ ∃ x : ℝ, Real.sqrt (x + 9) - Real.sqrt (x - 6) + 2 = 0 := by
  sorry

end no_real_roots_l3074_307404


namespace alice_winning_strategy_l3074_307445

theorem alice_winning_strategy :
  ∀ (n : ℕ), n < 10000000 ∧ n ≥ 1000000 →
  (n % 10 = 1 ∨ n % 10 = 3 ∨ n % 10 = 5 ∨ n % 10 = 7 ∨ n % 10 = 9) →
  ∃ (k : ℕ), k^7 % 10000000 = n := by
sorry

end alice_winning_strategy_l3074_307445


namespace min_value_in_geometric_sequence_l3074_307491

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r > 0, ∀ k, a (k + 1) = r * a k

-- Define the theorem
theorem min_value_in_geometric_sequence (a : ℕ → ℝ) 
  (h1 : is_positive_geometric_sequence a) 
  (h2 : a 4 * a 14 = 8) : 
  (∀ x y, x > 0 ∧ y > 0 ∧ x * y = 8 → 2*x + y ≥ 8) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x * y = 8 ∧ 2*x + y = 8) :=
sorry

end min_value_in_geometric_sequence_l3074_307491


namespace joan_football_games_l3074_307479

/-- Given that Joan went to 4 football games this year and 13 games in total,
    prove that she went to 9 games last year. -/
theorem joan_football_games (games_this_year games_total : ℕ)
  (h1 : games_this_year = 4)
  (h2 : games_total = 13) :
  games_total - games_this_year = 9 := by
  sorry

end joan_football_games_l3074_307479


namespace cookies_packages_bought_l3074_307427

def num_children : ℕ := 5
def cookies_per_package : ℕ := 25
def cookies_per_child : ℕ := 15

theorem cookies_packages_bought : 
  (num_children * cookies_per_child) / cookies_per_package = 3 := by
  sorry

end cookies_packages_bought_l3074_307427


namespace complex_equation_solution_l3074_307478

theorem complex_equation_solution (z : ℂ) :
  (1 - Complex.I) * z = 2 * Complex.I → z = -1 + Complex.I := by
  sorry

end complex_equation_solution_l3074_307478


namespace total_cost_is_10_79_l3074_307406

/-- The total cost of peppers purchased by Dale's Vegetarian Restaurant -/
def total_cost_peppers : ℝ :=
  let green_peppers := 2.8333333333333335
  let red_peppers := 3.254
  let yellow_peppers := 1.375
  let orange_peppers := 0.567
  let green_price := 1.20
  let red_price := 1.35
  let yellow_price := 1.50
  let orange_price := 1.65
  green_peppers * green_price +
  red_peppers * red_price +
  yellow_peppers * yellow_price +
  orange_peppers * orange_price

/-- Theorem stating that the total cost of peppers is $10.79 -/
theorem total_cost_is_10_79 : total_cost_peppers = 10.79 := by
  sorry

end total_cost_is_10_79_l3074_307406


namespace max_c_value_l3074_307465

theorem max_c_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : 2 * (a + b) = a * b) (h2 : a + b + c = a * b * c) :
  c ≤ 8 / 15 := by
  sorry

end max_c_value_l3074_307465


namespace fifteen_times_number_equals_three_hundred_l3074_307482

theorem fifteen_times_number_equals_three_hundred :
  ∃ x : ℝ, 15 * x = 300 ∧ x = 20 :=
by
  sorry

end fifteen_times_number_equals_three_hundred_l3074_307482


namespace escalator_step_count_l3074_307463

/-- Represents the number of steps a person counts while descending an escalator -/
def count_steps (escalator_length : ℕ) (walking_count : ℕ) (speed_multiplier : ℕ) : ℕ :=
  let escalator_speed := escalator_length - walking_count
  let speed_ratio := escalator_speed / walking_count
  let new_ratio := speed_ratio / speed_multiplier
  escalator_length / (new_ratio + 1)

/-- Theorem stating that given an escalator of 200 steps, where a person counts 50 steps while 
    walking down, the same person will count 80 steps when running twice as fast -/
theorem escalator_step_count :
  count_steps 200 50 2 = 80 := by
  sorry

end escalator_step_count_l3074_307463


namespace inequality_property_l3074_307419

theorem inequality_property (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end inequality_property_l3074_307419


namespace unique_p_for_three_natural_roots_l3074_307411

/-- The cubic equation with parameter p -/
def cubic_equation (p : ℝ) (x : ℝ) : ℝ :=
  5 * x^3 - 5 * (p + 1) * x^2 + (71 * p - 1) * x + 1 - 66 * p

/-- Predicate to check if a number is a natural number -/
def is_natural (x : ℝ) : Prop := ∃ n : ℕ, x = n

/-- The theorem to be proved -/
theorem unique_p_for_three_natural_roots :
  ∃! p : ℝ, ∃ x y z : ℝ,
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    is_natural x ∧ is_natural y ∧ is_natural z ∧
    cubic_equation p x = 0 ∧ cubic_equation p y = 0 ∧ cubic_equation p z = 0 ∧
    p = 76 :=
sorry

end unique_p_for_three_natural_roots_l3074_307411


namespace two_digit_primes_with_units_digit_9_l3074_307450

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def has_units_digit_9 (n : ℕ) : Prop := n % 10 = 9

theorem two_digit_primes_with_units_digit_9 :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, is_two_digit n ∧ has_units_digit_9 n ∧ Nat.Prime n) ∧ 
    (∀ n, is_two_digit n → has_units_digit_9 n → Nat.Prime n → n ∈ s) ∧
    s.card = 5 := by
  sorry

end two_digit_primes_with_units_digit_9_l3074_307450


namespace tetrahedron_PQRS_volume_l3074_307485

def tetrahedron_volume (PQ PR PS QR QS RS : ℝ) : ℝ := sorry

theorem tetrahedron_PQRS_volume :
  let PQ : ℝ := 3
  let PR : ℝ := Real.sqrt 10
  let PS : ℝ := Real.sqrt 17
  let QR : ℝ := 5
  let QS : ℝ := 3 * Real.sqrt 2
  let RS : ℝ := 6
  let z : ℝ := Real.sqrt (17 - (4/3)^2 - (1/(2*Real.sqrt 10))^2)
  tetrahedron_volume PQ PR PS QR QS RS = (Real.sqrt 10 / 2) * z := by sorry

end tetrahedron_PQRS_volume_l3074_307485


namespace sum_of_multiples_of_6_and_9_is_multiple_of_3_l3074_307441

theorem sum_of_multiples_of_6_and_9_is_multiple_of_3 (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 6 * k) (hb : ∃ m : ℤ, b = 9 * m) : 
  ∃ n : ℤ, a + b = 3 * n := by
sorry

end sum_of_multiples_of_6_and_9_is_multiple_of_3_l3074_307441


namespace oil_leak_calculation_l3074_307446

theorem oil_leak_calculation (total_leaked : ℕ) (leaked_while_fixing : ℕ) 
  (h1 : total_leaked = 6206)
  (h2 : leaked_while_fixing = 3731) :
  total_leaked - leaked_while_fixing = 2475 := by
sorry

end oil_leak_calculation_l3074_307446


namespace solution_existence_implies_a_bound_l3074_307475

theorem solution_existence_implies_a_bound (a : ℝ) :
  (∃ x : ℝ, |x + a| + |x - 2| + a < 2010) → a < 1006 := by
  sorry

end solution_existence_implies_a_bound_l3074_307475


namespace successive_discounts_equivalence_l3074_307464

/-- The equivalent single discount rate after applying successive discounts -/
def equivalent_discount (d1 d2 : ℝ) : ℝ :=
  1 - (1 - d1) * (1 - d2)

/-- Theorem stating that the equivalent single discount rate after applying
    successive discounts of 15% and 25% is 36.25% -/
theorem successive_discounts_equivalence :
  equivalent_discount 0.15 0.25 = 0.3625 := by
  sorry

end successive_discounts_equivalence_l3074_307464


namespace necessary_not_sufficient_condition_l3074_307424

theorem necessary_not_sufficient_condition (a : ℝ) : 
  (∀ a, (1 / a > 1 → a < 1)) ∧ 
  (∃ a, a < 1 ∧ ¬(1 / a > 1)) :=
by sorry

end necessary_not_sufficient_condition_l3074_307424


namespace davids_windows_l3074_307442

/-- Represents the time taken to wash windows -/
def wash_time : ℕ := 160

/-- Represents the number of windows washed in a single set -/
def windows_per_set : ℕ := 4

/-- Represents the time taken to wash one set of windows -/
def time_per_set : ℕ := 10

/-- Theorem stating the number of windows in David's house -/
theorem davids_windows : 
  (wash_time / time_per_set) * windows_per_set = 64 := by
  sorry

end davids_windows_l3074_307442


namespace train_passing_platform_l3074_307468

/-- Calculates the time for a train to pass a platform -/
theorem train_passing_platform (train_length : ℝ) (tree_passing_time : ℝ) (platform_length : ℝ) :
  train_length = 1200 →
  tree_passing_time = 120 →
  platform_length = 1100 →
  (train_length + platform_length) / (train_length / tree_passing_time) = 230 := by
sorry

end train_passing_platform_l3074_307468


namespace max_min_on_interval_l3074_307460

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_on_interval :
  ∃ (a b : ℝ), a ∈ Set.Icc 0 3 ∧ b ∈ Set.Icc 0 3 ∧
  (∀ x, x ∈ Set.Icc 0 3 → f x ≤ f a) ∧
  (∀ x, x ∈ Set.Icc 0 3 → f x ≥ f b) ∧
  f a = 5 ∧ f b = -15 :=
sorry

end max_min_on_interval_l3074_307460


namespace max_value_inequality_max_value_achieved_l3074_307447

theorem max_value_inequality (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + 2*c^2 = 1) : 
  a*b*Real.sqrt 3 + 3*b*c ≤ Real.sqrt 7 :=
sorry

theorem max_value_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ a b c : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 
  a^2 + b^2 + 2*c^2 = 1 ∧ 
  Real.sqrt 7 - ε < a*b*Real.sqrt 3 + 3*b*c :=
sorry

end max_value_inequality_max_value_achieved_l3074_307447


namespace wendy_albums_l3074_307484

/-- Given a total number of pictures, the number of pictures in the first album,
    and the number of pictures per album in the remaining albums,
    calculate the number of albums created for the remaining pictures. -/
def calculate_remaining_albums (total_pictures : ℕ) (first_album_pictures : ℕ) (pictures_per_album : ℕ) : ℕ :=
  (total_pictures - first_album_pictures) / pictures_per_album

theorem wendy_albums :
  calculate_remaining_albums 79 44 7 = 5 := by
  sorry

end wendy_albums_l3074_307484


namespace divisible_by_nine_l3074_307433

theorem divisible_by_nine : ∃ (n : ℕ), 5742 = 9 * n := by
  sorry

end divisible_by_nine_l3074_307433


namespace intersection_M_N_l3074_307420

def M : Set ℕ := {1, 2, 4, 8}

def N : Set ℕ := {x | ∃ k, x = 2 * k}

theorem intersection_M_N : M ∩ N = {2, 4, 8} := by
  sorry

end intersection_M_N_l3074_307420


namespace independence_test_distribution_X_expected_value_Y_variance_Y_l3074_307440

-- Define the contingency table
def male_noodles : ℕ := 30
def male_rice : ℕ := 25
def female_noodles : ℕ := 20
def female_rice : ℕ := 25
def total_students : ℕ := 100

-- Define the chi-square formula
def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value at α = 0.05
def critical_value : ℚ := 3841 / 1000

-- Theorem for independence test
theorem independence_test :
  chi_square male_noodles male_rice female_noodles female_rice < critical_value :=
sorry

-- Define the distribution of X
def prob_X (x : ℕ) : ℚ :=
  match x with
  | 0 => 3 / 10
  | 1 => 3 / 5
  | 2 => 1 / 10
  | _ => 0

-- Theorem for the distribution of X
theorem distribution_X :
  (prob_X 0 + prob_X 1 + prob_X 2 = 1) ∧
  (∀ x, x > 2 → prob_X x = 0) :=
sorry

-- Define Y as a binomial distribution
def p_Y : ℚ := 3 / 5
def n_Y : ℕ := 3

-- Theorems for expected value and variance of Y
theorem expected_value_Y :
  (n_Y : ℚ) * p_Y = 9 / 5 :=
sorry

theorem variance_Y :
  (n_Y : ℚ) * p_Y * (1 - p_Y) = 18 / 25 :=
sorry

end independence_test_distribution_X_expected_value_Y_variance_Y_l3074_307440


namespace two_inscribed_cube_lengths_l3074_307451

/-- A regular tetrahedron with unit edge length -/
structure RegularTetrahedron where
  edge_length : ℝ
  is_unit : edge_length = 1

/-- A cube inscribed in a tetrahedron such that each vertex lies on a face of the tetrahedron -/
structure InscribedCube where
  edge_length : ℝ
  vertices_on_faces : True  -- This is a placeholder for the geometric condition

/-- The set of all possible edge lengths for inscribed cubes in a unit regular tetrahedron -/
def inscribed_cube_edge_lengths (t : RegularTetrahedron) : Set ℝ :=
  {l | ∃ c : InscribedCube, c.edge_length = l}

/-- Theorem stating that there are exactly two distinct edge lengths for inscribed cubes -/
theorem two_inscribed_cube_lengths (t : RegularTetrahedron) :
  ∃ l₁ l₂ : ℝ, l₁ ≠ l₂ ∧ inscribed_cube_edge_lengths t = {l₁, l₂} :=
sorry

end two_inscribed_cube_lengths_l3074_307451


namespace mark_bread_making_time_l3074_307473

/-- The time it takes Mark to finish making bread -/
def bread_making_time (rise_time : ℕ) (rise_count : ℕ) (knead_time : ℕ) (bake_time : ℕ) : ℕ :=
  rise_time * rise_count + knead_time + bake_time

/-- Theorem stating the total time Mark takes to finish making the bread -/
theorem mark_bread_making_time :
  bread_making_time 120 2 10 30 = 280 := by
  sorry

end mark_bread_making_time_l3074_307473


namespace womens_average_age_l3074_307476

theorem womens_average_age 
  (n : ℕ) 
  (initial_men : ℕ) 
  (replaced_men_ages : ℕ × ℕ) 
  (age_increase : ℚ) :
  initial_men = 8 →
  replaced_men_ages = (20, 10) →
  age_increase = 2 →
  ∃ (total_age : ℚ),
    (total_age / initial_men + age_increase) * initial_men = 
      total_age - (replaced_men_ages.1 + replaced_men_ages.2) + 46 →
    46 / 2 = 23 :=
by sorry

end womens_average_age_l3074_307476


namespace division_problem_l3074_307497

theorem division_problem (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 52)
  (h2 : quotient = 16)
  (h3 : remainder = 4)
  (h4 : dividend = divisor * quotient + remainder) :
  divisor = 3 := by sorry

end division_problem_l3074_307497


namespace inequality_proof_l3074_307459

theorem inequality_proof (a b c : ℝ) : a^4 + b^4 + c^4 ≥ a*b*c^2 + b*c*a^2 + c*a*b^2 := by
  sorry

end inequality_proof_l3074_307459


namespace triangle_abc_is_right_angled_l3074_307444

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A parabola in the 2D plane -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Theorem: Given point A(1, 2) and a line passing through (5, -2) that intersects
    the parabola y^2 = 4x at points B and C, triangle ABC is right-angled -/
theorem triangle_abc_is_right_angled 
  (A : Point)
  (l : Line)
  (p : Parabola)
  (B C : Point)
  (h1 : A.x = 1 ∧ A.y = 2)
  (h2 : l.slope * (-2) + l.intercept = 5)
  (h3 : p.a = 4 ∧ p.h = 0 ∧ p.k = 0)
  (h4 : B.y^2 = 4 * B.x ∧ B.y = l.slope * B.x + l.intercept)
  (h5 : C.y^2 = 4 * C.x ∧ C.y = l.slope * C.x + l.intercept)
  (h6 : B ≠ C) :
  (B.x - A.x) * (C.x - A.x) + (B.y - A.y) * (C.y - A.y) = 0 := by
  sorry


end triangle_abc_is_right_angled_l3074_307444


namespace problem_statement_l3074_307488

theorem problem_statement (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  (a + b + c = 0 → a*b + b*c + c*a = -1/2) ∧
  ((a + b + c)^2 ≤ 3 ∧ ∃ (x y z : ℝ), x^2 + y^2 + z^2 = 1 ∧ (x + y + z)^2 = 3) :=
by sorry

end problem_statement_l3074_307488


namespace quadratic_points_relationship_l3074_307448

theorem quadratic_points_relationship (m : ℝ) (y₁ y₂ y₃ : ℝ) : 
  (2^2 - 4*2 - m = y₁) →
  (3^2 - 4*3 - m = y₂) →
  ((-1)^2 - 4*(-1) - m = y₃) →
  (y₃ > y₂ ∧ y₂ > y₁) :=
by sorry

end quadratic_points_relationship_l3074_307448


namespace solution_set_f_range_of_m_l3074_307495

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3| - 5
def g (x : ℝ) : ℝ := |x + 2| - 2

-- Theorem for the solution set of f(x) ≤ 2
theorem solution_set_f (x : ℝ) : f x ≤ 2 ↔ -4 ≤ x ∧ x ≤ 10 := by sorry

-- Theorem for the range of m
theorem range_of_m : 
  ∀ m : ℝ, (∃ x : ℝ, f x - g x ≥ m - 3) ↔ m ≤ 5 := by sorry

end solution_set_f_range_of_m_l3074_307495


namespace two_valid_plans_l3074_307400

/-- The number of valid purchasing plans for notebooks and pens -/
def valid_purchasing_plans : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ 3 * p.1 + 5 * p.2 = 35) 
    (Finset.product (Finset.range 36) (Finset.range 36))).card

/-- Theorem stating that there are exactly 2 valid purchasing plans -/
theorem two_valid_plans : valid_purchasing_plans = 2 := by
  sorry

end two_valid_plans_l3074_307400


namespace pet_store_combinations_l3074_307416

def num_puppies : ℕ := 20
def num_kittens : ℕ := 4
def num_hamsters : ℕ := 6
def num_rabbits : ℕ := 10
def num_people : ℕ := 4

theorem pet_store_combinations : 
  num_puppies * num_kittens * num_hamsters * num_rabbits * (Nat.factorial num_people) = 115200 := by
  sorry

end pet_store_combinations_l3074_307416


namespace cube_volume_from_surface_area_l3074_307431

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 864 → s^3 = 1728 := by
  sorry

end cube_volume_from_surface_area_l3074_307431


namespace tank_insulation_cost_l3074_307492

/-- Calculates the surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Calculates the total cost of insulation for a rectangular tank -/
def insulation_cost (length width height cost_per_sqft : ℝ) : ℝ :=
  surface_area length width height * cost_per_sqft

/-- Theorem: The cost of insulating a 4x5x2 feet tank at $20 per square foot is $1520 -/
theorem tank_insulation_cost :
  insulation_cost 4 5 2 20 = 1520 := by
  sorry

#eval insulation_cost 4 5 2 20

end tank_insulation_cost_l3074_307492


namespace finance_charge_rate_example_l3074_307477

/-- Given an original balance and a total payment, calculate the finance charge rate. -/
def finance_charge_rate (original_balance total_payment : ℚ) : ℚ :=
  (total_payment - original_balance) / original_balance * 100

/-- Theorem: The finance charge rate is 2% when the original balance is $150 and the total payment is $153. -/
theorem finance_charge_rate_example :
  finance_charge_rate 150 153 = 2 := by
  sorry

end finance_charge_rate_example_l3074_307477


namespace product_scaling_l3074_307493

theorem product_scaling (a b c : ℝ) (h : (a * 100) * (b * 100) = c) : 
  a * b = c / 10000 := by
  sorry

end product_scaling_l3074_307493


namespace population_decrease_proof_l3074_307428

/-- The annual rate of population decrease -/
def annual_decrease_rate : ℝ := 0.1

/-- The population after 2 years -/
def population_after_2_years : ℕ := 6480

/-- The initial population of the town -/
def initial_population : ℕ := 8000

theorem population_decrease_proof :
  (1 - annual_decrease_rate)^2 * initial_population = population_after_2_years :=
by sorry

end population_decrease_proof_l3074_307428


namespace log_xyz_t_equals_three_l3074_307453

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_xyz_t_equals_three 
  (t x y z : ℝ) 
  (h1 : log x t = 6)
  (h2 : log y t = 10)
  (h3 : log z t = 15) :
  log (x * y * z) t = 3 :=
by sorry

end log_xyz_t_equals_three_l3074_307453


namespace swimmer_distance_l3074_307461

/-- Calculates the distance swum against a current given the swimmer's speed in still water,
    the speed of the current, and the time taken. -/
def distance_against_current (swimmer_speed : ℝ) (current_speed : ℝ) (time : ℝ) : ℝ :=
  (swimmer_speed - current_speed) * time

/-- Proves that given the specified conditions, the swimmer travels 6 km against the current. -/
theorem swimmer_distance :
  let swimmer_speed := 4
  let current_speed := 2
  let time := 3
  distance_against_current swimmer_speed current_speed time = 6 := by
sorry

end swimmer_distance_l3074_307461


namespace books_combination_l3074_307426

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem books_combination : choose 15 3 = 455 := by sorry

end books_combination_l3074_307426


namespace lemon_pie_degrees_l3074_307439

/-- The number of degrees in a circle -/
def circle_degrees : ℕ := 360

/-- The total number of students -/
def total_students : ℕ := 45

/-- The number of students preferring chocolate pie -/
def chocolate_preference : ℕ := 15

/-- The number of students preferring apple pie -/
def apple_preference : ℕ := 10

/-- The number of students preferring blueberry pie -/
def blueberry_preference : ℕ := 7

/-- The number of students preferring lemon pie -/
def lemon_preference : ℕ := (total_students - (chocolate_preference + apple_preference + blueberry_preference)) / 2

theorem lemon_pie_degrees :
  (lemon_preference : ℚ) / total_students * circle_degrees = 56 := by
  sorry

end lemon_pie_degrees_l3074_307439


namespace student_count_l3074_307456

theorem student_count (total_average : ℝ) (group1_count : ℕ) (group1_average : ℝ)
                      (group2_count : ℕ) (group2_average : ℝ) (last_student_age : ℕ) :
  total_average = 15 →
  group1_count = 8 →
  group1_average = 14 →
  group2_count = 6 →
  group2_average = 16 →
  last_student_age = 17 →
  (group1_count : ℝ) * group1_average + (group2_count : ℝ) * group2_average + last_student_age = 15 * 15 :=
by sorry

end student_count_l3074_307456


namespace vector_operations_l3074_307403

def a : Fin 2 → ℝ := ![3, 4]
def b : Fin 2 → ℝ := ![2, -6]
def c : Fin 2 → ℝ := ![4, 1]

theorem vector_operations :
  (a • (a + b) = 7) ∧
  (c = a + (1/2 : ℝ) • b) := by sorry

end vector_operations_l3074_307403


namespace inscribed_rectangle_area_l3074_307422

-- Define the right triangle
def rightTriangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- Define the inscribed rectangle
def inscribedRectangle (x : ℝ) (a b c : ℝ) : Prop :=
  rightTriangle a b c ∧ x > 0 ∧ 2*x > 0 ∧ x ≤ a ∧ 2*x ≤ b

-- Theorem statement
theorem inscribed_rectangle_area (x : ℝ) :
  rightTriangle 24 (60 - 24) 60 →
  inscribedRectangle x 24 (60 - 24) 60 →
  x * (2*x) = 1440 :=
by sorry

end inscribed_rectangle_area_l3074_307422


namespace roots_count_lower_bound_l3074_307499

def count_roots (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  sorry

theorem roots_count_lower_bound
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (3 + x) = f (3 - x))
  (h2 : ∀ x, f (9 + x) = f (9 - x))
  (h3 : f 1 = 0) :
  count_roots f (-1000) 1000 ≥ 334 := by
  sorry

end roots_count_lower_bound_l3074_307499


namespace bus_remaining_distance_l3074_307410

def distance_between_points (z : ℝ) : Prop :=
  ∃ (x : ℝ), x > 0 ∧
  (z / 2) / (z - 19.2) = x ∧
  (z - 12) / (z / 2) = x

theorem bus_remaining_distance (z : ℝ) (h : distance_between_points z) :
  z - z * (4/5) = 6.4 :=
sorry

end bus_remaining_distance_l3074_307410


namespace ratio_equations_solution_l3074_307432

theorem ratio_equations_solution (x y z a : ℤ) : 
  (∃ k : ℤ, x = k ∧ y = 4*k ∧ z = 5*k) →
  y = 9*a^2 - 2*a - 8 →
  z = 10*a + 2 →
  a = 5 :=
by sorry

end ratio_equations_solution_l3074_307432


namespace teachers_per_grade_l3074_307474

theorem teachers_per_grade (fifth_graders sixth_graders seventh_graders : ℕ)
  (parents_per_grade : ℕ) (num_buses seat_per_bus : ℕ) (num_grades : ℕ) :
  fifth_graders = 109 →
  sixth_graders = 115 →
  seventh_graders = 118 →
  parents_per_grade = 2 →
  num_buses = 5 →
  seat_per_bus = 72 →
  num_grades = 3 →
  (num_buses * seat_per_bus - (fifth_graders + sixth_graders + seventh_graders + parents_per_grade * num_grades)) / num_grades = 4 := by
  sorry

end teachers_per_grade_l3074_307474


namespace equation_solution_l3074_307470

theorem equation_solution :
  ∃! x : ℚ, 7 * (4 * x + 3) - 5 = -3 * (2 - 5 * x) ∧ x = -22 / 13 := by
sorry

end equation_solution_l3074_307470


namespace factorization_proof_l3074_307458

theorem factorization_proof (x : ℝ) : 
  (3 * x^2 - 12 = 3 * (x + 2) * (x - 2)) ∧ 
  (x^2 - 2*x - 8 = (x - 4) * (x + 2)) := by
sorry

end factorization_proof_l3074_307458


namespace final_class_size_l3074_307455

def fourth_grade_class_size (initial_students : ℕ) 
  (first_semester_left : ℕ) (first_semester_joined : ℕ)
  (second_semester_joined : ℕ) (second_semester_transferred : ℕ) (second_semester_switched : ℕ) : ℕ :=
  initial_students - first_semester_left + first_semester_joined + 
  second_semester_joined - second_semester_transferred - second_semester_switched

theorem final_class_size : 
  fourth_grade_class_size 11 6 25 15 3 2 = 40 := by
  sorry

end final_class_size_l3074_307455


namespace arithmetic_sequence_common_difference_l3074_307481

/-- Proves that the common difference of an arithmetic sequence is 5,
    given the specified conditions. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ) -- First term
  (a_n : ℕ) -- Last term
  (S : ℕ) -- Sum of all terms
  (h_a : a = 5)
  (h_a_n : a_n = 50)
  (h_S : S = 275)
  : ∃ (n : ℕ) (d : ℕ), n > 1 ∧ d = 5 ∧ 
    a_n = a + (n - 1) * d ∧
    S = n * (a + a_n) / 2 :=
sorry

end arithmetic_sequence_common_difference_l3074_307481


namespace diana_hourly_wage_l3074_307452

/-- Diana's work schedule and earnings --/
structure DianaWork where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculate Diana's hourly wage --/
def hourly_wage (d : DianaWork) : ℚ :=
  d.weekly_earnings / (d.monday_hours + d.tuesday_hours + d.wednesday_hours + d.thursday_hours + d.friday_hours)

/-- Theorem: Diana's hourly wage is $30 --/
theorem diana_hourly_wage :
  let d : DianaWork := {
    monday_hours := 10,
    tuesday_hours := 15,
    wednesday_hours := 10,
    thursday_hours := 15,
    friday_hours := 10,
    weekly_earnings := 1800
  }
  hourly_wage d = 30 := by sorry

end diana_hourly_wage_l3074_307452


namespace min_k_value_l3074_307409

theorem min_k_value (x y k : ℝ) : 
  (x - y + 5 ≥ 0) → 
  (x ≤ 3) → 
  (x + y + k ≥ 0) → 
  (∃ z : ℝ, z = 2*x + 4*y ∧ z ≥ -6 ∧ ∀ w : ℝ, w = 2*x + 4*y → w ≥ z) →
  k ≥ 0 :=
by sorry

end min_k_value_l3074_307409


namespace amicable_iff_ge_seven_l3074_307486

/-- An integer n ≥ 2 is amicable if there exist subsets A₁, A₂, ..., Aₙ of {1, 2, ..., n} satisfying:
    (i) i ∉ Aᵢ for any i = 1, 2, ..., n
    (ii) i ∈ Aⱼ for any j ∉ Aᵢ, for any i ≠ j
    (iii) Aᵢ ∩ Aⱼ ≠ ∅ for any i, j ∈ {1, 2, ..., n} -/
def IsAmicable (n : ℕ) : Prop :=
  n ≥ 2 ∧
  ∃ A : Fin n → Set (Fin n),
    (∀ i, i ∉ A i) ∧
    (∀ i j, i ≠ j → (j ∉ A i ↔ i ∈ A j)) ∧
    (∀ i j, (A i ∩ A j).Nonempty)

/-- For any integer n ≥ 2, n is amicable if and only if n ≥ 7 -/
theorem amicable_iff_ge_seven (n : ℕ) : IsAmicable n ↔ n ≥ 7 := by
  sorry

end amicable_iff_ge_seven_l3074_307486


namespace circle_quadratic_intersection_l3074_307489

/-- Given a circle and a quadratic equation, prove the center coordinates and condition --/
theorem circle_quadratic_intersection (p q b c : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*p*x - 2*q*y + 2*q - 1 = 0 ↔ 
   (y = 0 → x^2 + b*x + c = 0)) →
  (p = -b/2 ∧ q = (1+c)/2 ∧ b^2 - 4*c ≥ 0) := by
  sorry

end circle_quadratic_intersection_l3074_307489
