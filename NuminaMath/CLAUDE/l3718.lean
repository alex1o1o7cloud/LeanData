import Mathlib

namespace NUMINAMATH_CALUDE_problem_1_l3718_371828

theorem problem_1 (a : ℚ) (h : a = 4/5) :
  -24.7 * a + 1.3 * a - (33/5) * a = -24 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3718_371828


namespace NUMINAMATH_CALUDE_tan_value_problem_l3718_371801

theorem tan_value_problem (θ : Real) 
  (h1 : 0 < θ) (h2 : θ < π) 
  (h3 : Real.sin θ + Real.cos θ = 1/5) : 
  Real.tan θ = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_problem_l3718_371801


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l3718_371805

theorem smallest_m_for_integral_solutions : 
  ∃ (m : ℕ), m > 0 ∧ 
  (∃ (x : ℤ), 18 * x^2 - m * x + 252 = 0) ∧ 
  (∀ (k : ℕ), k > 0 ∧ k < m → ¬∃ (y : ℤ), 18 * y^2 - k * y + 252 = 0) ∧ 
  m = 162 := by
sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l3718_371805


namespace NUMINAMATH_CALUDE_trig_identity_l3718_371843

theorem trig_identity (α : Real) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3718_371843


namespace NUMINAMATH_CALUDE_g_6_equals_666_l3718_371800

def g (x : ℝ) : ℝ := 3*x^4 - 19*x^3 + 31*x^2 - 27*x - 72

theorem g_6_equals_666 : g 6 = 666 := by
  sorry

end NUMINAMATH_CALUDE_g_6_equals_666_l3718_371800


namespace NUMINAMATH_CALUDE_fraction_chain_l3718_371871

theorem fraction_chain (a b c d : ℚ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7) :
  d / a = 4 / 35 := by
sorry

end NUMINAMATH_CALUDE_fraction_chain_l3718_371871


namespace NUMINAMATH_CALUDE_divide_decimals_l3718_371879

theorem divide_decimals : (0.08 : ℚ) / (0.002 : ℚ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_divide_decimals_l3718_371879


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l3718_371862

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a line passes through a point
def passesThroughPoint (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on both axes
def hasEqualIntercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ (-l.c / l.a = -l.c / l.b)

-- Theorem statement
theorem line_through_point_with_equal_intercepts :
  ∃ (l : Line), passesThroughPoint l ⟨-3, -2⟩ ∧ hasEqualIntercepts l ∧
  ((l.a = 2 ∧ l.b = -3 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = 5)) := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l3718_371862


namespace NUMINAMATH_CALUDE_function_bound_implies_parameter_range_l3718_371812

-- Define the function f
def f (a d x : ℝ) : ℝ := a * x^3 + x^2 + x + d

-- State the theorem
theorem function_bound_implies_parameter_range :
  ∀ (a d : ℝ),
  (∀ x : ℝ, |x| ≤ 1 → |f a d x| ≤ 1) →
  (a ∈ Set.Icc (-2) 0 ∧ d ∈ Set.Icc (-2) 0) :=
by sorry

end NUMINAMATH_CALUDE_function_bound_implies_parameter_range_l3718_371812


namespace NUMINAMATH_CALUDE_cornelia_asian_countries_l3718_371816

theorem cornelia_asian_countries (total : ℕ) (europe : ℕ) (south_america : ℕ) 
  (h1 : total = 42)
  (h2 : europe = 20)
  (h3 : south_america = 10)
  (h4 : (total - europe - south_america) % 2 = 0) :
  (total - europe - south_america) / 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_cornelia_asian_countries_l3718_371816


namespace NUMINAMATH_CALUDE_square_root_problem_l3718_371820

theorem square_root_problem (a b : ℝ) 
  (h1 : (2 * a + 1) = 9)
  (h2 : (5 * a + 2 * b - 2) = 16) :
  (3 * a - 4 * b) = 16 := by
sorry

end NUMINAMATH_CALUDE_square_root_problem_l3718_371820


namespace NUMINAMATH_CALUDE_matthew_cooks_30_hotdogs_l3718_371860

/-- The number of hotdogs Matthew needs to cook for his family dinner -/
def total_hotdogs : ℝ :=
  let ella_emma := 2.5 * 2
  let luke := 2 * ella_emma
  let michael := 7
  let hunter := 1.5 * ella_emma
  let zoe := 0.5
  ella_emma + luke + michael + hunter + zoe

/-- Theorem stating that Matthew needs to cook 30 hotdogs -/
theorem matthew_cooks_30_hotdogs : total_hotdogs = 30 := by
  sorry

end NUMINAMATH_CALUDE_matthew_cooks_30_hotdogs_l3718_371860


namespace NUMINAMATH_CALUDE_clothes_transport_expenditure_l3718_371822

/-- Calculates the monthly amount spent on clothes and transport given the yearly savings --/
def monthly_clothes_transport (yearly_savings : ℕ) : ℕ :=
  let monthly_savings := yearly_savings / 12
  let monthly_salary := monthly_savings * 5
  monthly_salary / 5

/-- Theorem stating that given the conditions in the problem, 
    the monthly amount spent on clothes and transport is 4038 --/
theorem clothes_transport_expenditure :
  monthly_clothes_transport 48456 = 4038 := by
  sorry

#eval monthly_clothes_transport 48456

end NUMINAMATH_CALUDE_clothes_transport_expenditure_l3718_371822


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_expressions_l3718_371815

theorem consecutive_odd_integers_expressions (p q : ℤ) 
  (h1 : ∃ k : ℤ, p = 2*k + 1 ∧ q = 2*k + 3) : 
  Odd (2*p + 5*q) ∧ Odd (5*p - 2*q) ∧ Odd (2*p*q + 5) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_expressions_l3718_371815


namespace NUMINAMATH_CALUDE_urn_probability_l3718_371873

/-- Represents the content of the urn -/
structure UrnContent where
  red : ℕ
  blue : ℕ

/-- Represents a single draw operation -/
inductive DrawResult
| Red
| Blue

/-- Represents a sequence of draw results -/
def DrawSequence := List DrawResult

/-- The initial content of the urn -/
def initial_urn : UrnContent := ⟨2, 1⟩

/-- The number of draw operations performed -/
def num_operations : ℕ := 5

/-- The final number of balls in the urn -/
def final_total_balls : ℕ := 8

/-- The target final content of the urn -/
def target_final_urn : UrnContent := ⟨3, 5⟩

/-- Calculates the probability of drawing a red ball from the urn -/
def prob_draw_red (urn : UrnContent) : ℚ :=
  urn.red / (urn.red + urn.blue)

/-- Updates the urn content after a draw -/
def update_urn (urn : UrnContent) (draw : DrawResult) : UrnContent :=
  match draw with
  | DrawResult.Red => ⟨urn.red + 1, urn.blue⟩
  | DrawResult.Blue => ⟨urn.red, urn.blue + 1⟩

/-- Calculates the probability of a specific draw sequence -/
def sequence_probability (seq : DrawSequence) : ℚ :=
  sorry

/-- Calculates the number of valid sequences leading to the target urn content -/
def num_valid_sequences : ℕ :=
  sorry

/-- The main theorem stating the probability of ending with the target urn content -/
theorem urn_probability : 
  sequence_probability (List.replicate num_operations DrawResult.Red) * num_valid_sequences = 4/21 :=
sorry

end NUMINAMATH_CALUDE_urn_probability_l3718_371873


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3718_371846

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 47) 
  (eq2 : 4 * a + 3 * b = 39) : 
  a + b = 82 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3718_371846


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l3718_371802

/-- A regular nonagon is a 9-sided polygon with all sides and angles equal. -/
def RegularNonagon : Type := Unit

/-- A diagonal of a regular nonagon is a line segment connecting two non-adjacent vertices. -/
def Diagonal (n : RegularNonagon) : Type := Unit

/-- The set of all diagonals in a regular nonagon. -/
def AllDiagonals (n : RegularNonagon) : Set (Diagonal n) := sorry

/-- Two diagonals intersect if they cross each other inside the nonagon. -/
def Intersect (n : RegularNonagon) (d1 d2 : Diagonal n) : Prop := sorry

/-- The probability of an event is the number of favorable outcomes divided by the total number of possible outcomes. -/
def Probability (n : RegularNonagon) (event : Set (Diagonal n × Diagonal n)) : ℚ := sorry

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  Probability n {p : Diagonal n × Diagonal n | Intersect n p.1 p.2} = 14/39 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l3718_371802


namespace NUMINAMATH_CALUDE_binomial_sum_equality_l3718_371894

theorem binomial_sum_equality (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) : 
  (∑' d, Nat.choose (n - r + 1) d * Nat.choose (r - 1) (d - 1)) = Nat.choose n r :=
by sorry

end NUMINAMATH_CALUDE_binomial_sum_equality_l3718_371894


namespace NUMINAMATH_CALUDE_prob_red_then_black_54_card_deck_l3718_371875

/-- A deck of cards with red and black cards, including jokers -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)

/-- The probability of drawing a red card first and a black card second -/
def prob_red_then_black (d : Deck) : ℚ :=
  (d.red_cards : ℚ) * d.black_cards / (d.total_cards * (d.total_cards - 1))

/-- The theorem stating the probability for the specific deck -/
theorem prob_red_then_black_54_card_deck :
  prob_red_then_black ⟨54, 26, 28⟩ = 364 / 1431 := by sorry

end NUMINAMATH_CALUDE_prob_red_then_black_54_card_deck_l3718_371875


namespace NUMINAMATH_CALUDE_shortest_distance_is_four_l3718_371886

/-- Represents the distances between three points A, B, and C. -/
structure TriangleDistances where
  ab : ℝ  -- Distance between A and B
  bc : ℝ  -- Distance between B and C
  ac : ℝ  -- Distance between A and C

/-- Given conditions for the problem -/
def problem_conditions (d : TriangleDistances) : Prop :=
  d.ab + d.bc = 10 ∧
  d.bc + d.ac = 13 ∧
  d.ac + d.ab = 11

/-- The theorem to be proved -/
theorem shortest_distance_is_four (d : TriangleDistances) 
  (h : problem_conditions d) : 
  min d.ab (min d.bc d.ac) = 4 := by
  sorry


end NUMINAMATH_CALUDE_shortest_distance_is_four_l3718_371886


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_l3718_371845

theorem quadratic_root_ratio (c : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (4 * x₁^2 - 5 * x₁ + c = 0) ∧ 
    (4 * x₂^2 - 5 * x₂ + c = 0) ∧ 
    (x₁ / x₂ = -3/4)) →
  c = -75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_l3718_371845


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3718_371898

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = p.1^2}
def N : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {(1, 1), (-1, 1)} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3718_371898


namespace NUMINAMATH_CALUDE_equation_solution_l3718_371853

theorem equation_solution :
  let f : ℝ → ℝ := λ x => (3*x + 7)*(x - 2) - (7*x - 4)
  ∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 39 / 3 ∧
              x₂ = 1 - Real.sqrt 39 / 3 ∧
              f x₁ = 0 ∧
              f x₂ = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3718_371853


namespace NUMINAMATH_CALUDE_fourth_power_equation_l3718_371889

theorem fourth_power_equation : 10^4 + 15^4 + 8^4 + 2*3^4 = 16^4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_equation_l3718_371889


namespace NUMINAMATH_CALUDE_ab_gt_1_sufficient_not_necessary_for_a_plus_b_gt_2_l3718_371895

theorem ab_gt_1_sufficient_not_necessary_for_a_plus_b_gt_2 :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x * y > 1 → x + y > 2) ∧
  (∃ (c d : ℝ), c > 0 ∧ d > 0 ∧ c + d > 2 ∧ c * d ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ab_gt_1_sufficient_not_necessary_for_a_plus_b_gt_2_l3718_371895


namespace NUMINAMATH_CALUDE_systematic_sample_valid_l3718_371887

def is_valid_systematic_sample (sample : List Nat) (population_size : Nat) (sample_size : Nat) : Prop :=
  sample.length = sample_size ∧
  ∀ i j, i < j → i < sample.length → j < sample.length →
    sample[i]! < sample[j]! ∧
    (sample[j]! - sample[i]!) % (population_size / sample_size) = 0 ∧
    sample[sample.length - 1]! ≤ population_size

theorem systematic_sample_valid :
  is_valid_systematic_sample [3, 13, 23, 33, 43, 53] 60 6 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_valid_l3718_371887


namespace NUMINAMATH_CALUDE_tetrahedron_edges_form_triangles_l3718_371883

/-- Represents a tetrahedron with edge lengths a, b, c, d, e, f -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d
  e_pos : 0 < e
  f_pos : 0 < f
  vertex_sum_equal : a + b + c = b + d + f ∧ a + b + c = c + d + e ∧ a + b + c = a + e + f

theorem tetrahedron_edges_form_triangles (t : Tetrahedron) :
  (t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b) ∧
  (t.b + t.d > t.f ∧ t.d + t.f > t.b ∧ t.f + t.b > t.d) ∧
  (t.c + t.d > t.e ∧ t.d + t.e > t.c ∧ t.e + t.c > t.d) ∧
  (t.a + t.e > t.f ∧ t.e + t.f > t.a ∧ t.f + t.a > t.e) := by
  sorry


end NUMINAMATH_CALUDE_tetrahedron_edges_form_triangles_l3718_371883


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3718_371804

def M : Set ℝ := {x | 4 ≤ x ∧ x ≤ 7}
def N : Set ℝ := {3, 5, 8}

theorem intersection_of_M_and_N : M ∩ N = {5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3718_371804


namespace NUMINAMATH_CALUDE_rectangles_not_necessarily_similar_l3718_371835

-- Define a rectangle
structure Rectangle where
  length : ℝ
  width : ℝ
  length_positive : length > 0
  width_positive : width > 0

-- Define similarity for rectangles
def are_similar (r1 r2 : Rectangle) : Prop :=
  r1.length / r1.width = r2.length / r2.width

-- Theorem stating that rectangles are not necessarily similar
theorem rectangles_not_necessarily_similar :
  ∃ (r1 r2 : Rectangle), ¬(are_similar r1 r2) :=
sorry

end NUMINAMATH_CALUDE_rectangles_not_necessarily_similar_l3718_371835


namespace NUMINAMATH_CALUDE_division_problem_l3718_371842

theorem division_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 190 →
  divisor = 21 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3718_371842


namespace NUMINAMATH_CALUDE_charlie_running_steps_l3718_371837

/-- Given that Charlie makes 5350 steps on a 3-kilometer running field,
    prove that running 2 1/2 times around the field results in 13375 steps. -/
theorem charlie_running_steps (steps_per_field : ℕ) (field_length : ℝ) (laps : ℝ) :
  steps_per_field = 5350 →
  field_length = 3 →
  laps = 2.5 →
  (steps_per_field : ℝ) * laps = 13375 := by
  sorry

end NUMINAMATH_CALUDE_charlie_running_steps_l3718_371837


namespace NUMINAMATH_CALUDE_b_completion_time_l3718_371855

/-- The time it takes for person A to complete the work alone -/
def time_A : ℝ := 15

/-- The time A and B work together -/
def time_together : ℝ := 5

/-- The fraction of work left after A and B work together -/
def work_left : ℝ := 0.41666666666666663

/-- The time it takes for person B to complete the work alone -/
def time_B : ℝ := 20

/-- Theorem stating that given the conditions, B takes 20 days to complete the work alone -/
theorem b_completion_time :
  (time_together * (1 / time_A + 1 / time_B) = 1 - work_left) →
  time_B = 20 := by
  sorry

end NUMINAMATH_CALUDE_b_completion_time_l3718_371855


namespace NUMINAMATH_CALUDE_negation_of_exists_proposition_l3718_371818

theorem negation_of_exists_proposition :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_exists_proposition_l3718_371818


namespace NUMINAMATH_CALUDE_not_divisible_1985_1987_divisible_1987_1989_l3718_371867

/-- Represents an L-shaped piece consisting of 3 unit squares -/
structure LShape :=
  (width : ℕ)
  (height : ℕ)
  (area_eq_3 : width * height = 3)

/-- Checks if a rectangle can be divided into L-shapes -/
def can_divide_into_l_shapes (m n : ℕ) : Prop :=
  (m * n) % 3 = 0 ∨ 
  ∃ (a b c d : ℕ), m = 2 * a + 7 * b ∧ n = 3 * c + 9 * d

/-- Theorem stating the divisibility condition for 1985 × 1987 rectangle -/
theorem not_divisible_1985_1987 : ¬(can_divide_into_l_shapes 1985 1987) :=
sorry

/-- Theorem stating the divisibility condition for 1987 × 1989 rectangle -/
theorem divisible_1987_1989 : can_divide_into_l_shapes 1987 1989 :=
sorry

end NUMINAMATH_CALUDE_not_divisible_1985_1987_divisible_1987_1989_l3718_371867


namespace NUMINAMATH_CALUDE_spinner_prime_sum_probability_l3718_371865

/-- Represents a spinner with numbered sectors -/
structure Spinner :=
  (sectors : List Nat)

/-- Checks if a number is prime -/
def isPrime (n : Nat) : Bool :=
  n > 1 && (List.range (n - 1)).all (fun d => d <= 1 || n % (d + 2) ≠ 0)

/-- Calculates all possible sums from two spinners -/
def allSums (s1 s2 : Spinner) : List Nat :=
  List.join (s1.sectors.map (fun x => s2.sectors.map (fun y => x + y)))

/-- Counts the number of prime sums -/
def countPrimeSums (sums : List Nat) : Nat :=
  sums.filter isPrime |>.length

theorem spinner_prime_sum_probability :
  let spinner1 : Spinner := ⟨[1, 2, 3]⟩
  let spinner2 : Spinner := ⟨[3, 4, 5]⟩
  let allPossibleSums := allSums spinner1 spinner2
  let totalSums := allPossibleSums.length
  let primeSums := countPrimeSums allPossibleSums
  (primeSums : Rat) / totalSums = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_spinner_prime_sum_probability_l3718_371865


namespace NUMINAMATH_CALUDE_isabel_homework_problem_l3718_371861

theorem isabel_homework_problem (math_pages : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) :
  math_pages = 2 →
  reading_pages = 4 →
  problems_per_page = 5 →
  (math_pages + reading_pages) * problems_per_page = 30 := by
  sorry

end NUMINAMATH_CALUDE_isabel_homework_problem_l3718_371861


namespace NUMINAMATH_CALUDE_amc_12_scoring_problem_l3718_371863

/-- The minimum number of correctly solved problems to achieve the target score -/
def min_correct_problems (total_problems : ℕ) (attempted_problems : ℕ) (points_correct : ℕ) 
  (points_unanswered : ℕ) (target_score : ℕ) : ℕ :=
  let unanswered := total_problems - attempted_problems
  let points_from_unanswered := unanswered * points_unanswered
  let required_points := target_score - points_from_unanswered
  (required_points + points_correct - 1) / points_correct

theorem amc_12_scoring_problem :
  min_correct_problems 30 25 7 2 120 = 16 := by
  sorry

end NUMINAMATH_CALUDE_amc_12_scoring_problem_l3718_371863


namespace NUMINAMATH_CALUDE_blue_marbles_count_l3718_371825

theorem blue_marbles_count (yellow green black : ℕ) (total : ℕ) (prob_black : ℚ) :
  yellow = 12 →
  green = 5 →
  black = 1 →
  prob_black = 1 / 28 →
  total = yellow + green + black + (total - yellow - green - black) →
  prob_black = black / total →
  (total - yellow - green - black) = 10 := by
  sorry

end NUMINAMATH_CALUDE_blue_marbles_count_l3718_371825


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_implies_a_range_l3718_371811

-- Define the complex number z
def z (a : ℝ) : ℂ := (2 + a * Complex.I) * (a - Complex.I)

-- Define the condition for z to be in the third quadrant
def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

-- State the theorem
theorem z_in_third_quadrant_implies_a_range (a : ℝ) :
  in_third_quadrant (z a) → -Real.sqrt 2 < a ∧ a < 0 := by sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_implies_a_range_l3718_371811


namespace NUMINAMATH_CALUDE_infinite_product_value_l3718_371857

/-- The nth term of the sequence in the infinite product -/
def a (n : ℕ) : ℝ := (2^n)^(1 / 3^n)

/-- The sum of the exponents in the infinite product -/
noncomputable def S : ℝ := ∑' n, n / 3^n

/-- The infinite product -/
noncomputable def infiniteProduct : ℝ := 2^S

theorem infinite_product_value :
  infiniteProduct = 2^(3/4) := by sorry

end NUMINAMATH_CALUDE_infinite_product_value_l3718_371857


namespace NUMINAMATH_CALUDE_tile_side_length_proof_l3718_371854

/-- Represents a rectangular room with length and width in centimeters -/
structure Room where
  length : ℕ
  width : ℕ

/-- Represents a square tile with side length in centimeters -/
structure Tile where
  side_length : ℕ

/-- Calculates the area of a room in square centimeters -/
def room_area (r : Room) : ℕ := r.length * r.width

/-- Calculates the area of a tile in square centimeters -/
def tile_area (t : Tile) : ℕ := t.side_length * t.side_length

theorem tile_side_length_proof (r : Room) (num_tiles : ℕ) (h1 : r.length = 5000) (h2 : r.width = 1125) (h3 : num_tiles = 9000) :
  ∃ t : Tile, tile_area t * num_tiles = room_area r ∧ t.side_length = 25 := by
  sorry

end NUMINAMATH_CALUDE_tile_side_length_proof_l3718_371854


namespace NUMINAMATH_CALUDE_tangent_line_p_values_l3718_371870

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 8 = 0

/-- The equation of the tangent line -/
def tangent_line (p x : ℝ) : Prop := x = -p/2

/-- The line is tangent to the circle -/
def is_tangent (p : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_equation x y ∧ tangent_line p x

/-- Theorem: If the line x = -p/2 is tangent to the circle x^2 + y^2 + 6x + 8 = 0, then p = 4 or p = 8 -/
theorem tangent_line_p_values (p : ℝ) : is_tangent p → p = 4 ∨ p = 8 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_p_values_l3718_371870


namespace NUMINAMATH_CALUDE_cube_root_over_sixth_root_of_eight_l3718_371840

theorem cube_root_over_sixth_root_of_eight (x : ℝ) :
  (8 ^ (1/3)) / (8 ^ (1/6)) = 8 ^ (1/6) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_over_sixth_root_of_eight_l3718_371840


namespace NUMINAMATH_CALUDE_all_propositions_false_l3718_371899

-- Define the type for lines in space
def Line : Type := ℝ → ℝ → ℝ → Prop

-- Define the relations between lines
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def skew (l1 l2 : Line) : Prop := sorry
def intersect (l1 l2 : Line) : Prop := sorry
def coplanar (l1 l2 : Line) : Prop := sorry

-- Define the propositions
def proposition1 (a b c : Line) : Prop :=
  (perpendicular a b ∧ perpendicular b c) → parallel a c

def proposition2 (a b c : Line) : Prop :=
  (skew a b ∧ skew b c) → skew a c

def proposition3 (a b c : Line) : Prop :=
  (intersect a b ∧ intersect b c) → intersect a c

def proposition4 (a b c : Line) : Prop :=
  (coplanar a b ∧ coplanar b c) → coplanar a c

-- Theorem stating that all propositions are false
theorem all_propositions_false (a b c : Line) :
  ¬ proposition1 a b c ∧
  ¬ proposition2 a b c ∧
  ¬ proposition3 a b c ∧
  ¬ proposition4 a b c :=
sorry

end NUMINAMATH_CALUDE_all_propositions_false_l3718_371899


namespace NUMINAMATH_CALUDE_alla_boris_meeting_point_l3718_371831

/-- The number of lanterns along the alley -/
def total_lanterns : ℕ := 400

/-- Alla's position when the first observation is made -/
def alla_position : ℕ := 55

/-- Boris's position when the first observation is made -/
def boris_position : ℕ := 321

/-- The meeting point of Alla and Boris -/
def meeting_point : ℕ := 163

/-- Theorem stating that Alla and Boris will meet at the calculated meeting point -/
theorem alla_boris_meeting_point :
  ∀ (alla_start boris_start : ℕ),
  alla_start = 1 ∧ boris_start = total_lanterns ∧
  alla_position > alla_start ∧ boris_position < boris_start ∧
  (alla_position - alla_start) / (total_lanterns - alla_position - (boris_start - boris_position)) =
  (meeting_point - alla_start) / (boris_start - meeting_point) :=
by sorry

end NUMINAMATH_CALUDE_alla_boris_meeting_point_l3718_371831


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3718_371892

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a where a₄ = 7 and a₆ = 21, prove that a₈ = 63. -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) 
  (h_a4 : a 4 = 7) 
  (h_a6 : a 6 = 21) : 
  a 8 = 63 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3718_371892


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3718_371844

/-- The repeating decimal 0.3535... expressed as a real number -/
def repeating_decimal : ℚ := 35 / 99

theorem repeating_decimal_equals_fraction : repeating_decimal = 35 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3718_371844


namespace NUMINAMATH_CALUDE_anika_age_l3718_371880

theorem anika_age :
  ∀ (anika_age maddie_age : ℕ),
  anika_age = (4 * maddie_age) / 3 →
  (anika_age + 15 + maddie_age + 15) / 2 = 50 →
  anika_age = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_anika_age_l3718_371880


namespace NUMINAMATH_CALUDE_alice_needs_1615_stamps_l3718_371893

def stamps_problem (alice ernie peggy danny bert : ℕ) : Prop :=
  alice = 65 ∧
  danny = alice + 5 ∧
  peggy = 2 * danny ∧
  ernie = 3 * peggy ∧
  bert = 4 * ernie

theorem alice_needs_1615_stamps 
  (alice ernie peggy danny bert : ℕ) 
  (h : stamps_problem alice ernie peggy danny bert) : 
  bert - alice = 1615 := by
sorry

end NUMINAMATH_CALUDE_alice_needs_1615_stamps_l3718_371893


namespace NUMINAMATH_CALUDE_assignment_schemes_l3718_371824

/-- Represents the number of teachers and schools -/
def n : ℕ := 4

/-- The total number of assignment schemes -/
def total_schemes : ℕ := n^n

/-- The number of schemes where exactly one school is not assigned any teachers -/
def one_school_empty : ℕ := n * (n - 1)^(n - 1)

/-- The number of schemes where a certain school is assigned 2 teachers -/
def two_teachers_one_school : ℕ := Nat.choose n 2 * (n - 1)^(n - 2)

/-- The number of schemes where exactly two schools are not assigned any teachers -/
def two_schools_empty : ℕ := Nat.choose n 2 * (Nat.choose n 2 / 2 + n) * 2

theorem assignment_schemes :
  total_schemes = 256 ∧
  one_school_empty = 144 ∧
  two_teachers_one_school = 54 ∧
  two_schools_empty = 84 := by
  sorry

end NUMINAMATH_CALUDE_assignment_schemes_l3718_371824


namespace NUMINAMATH_CALUDE_car_selection_problem_l3718_371882

theorem car_selection_problem (cars : ℕ) (selections_per_client : ℕ) (selections_per_car : ℕ) 
  (h1 : cars = 12)
  (h2 : selections_per_client = 4)
  (h3 : selections_per_car = 3) :
  (cars * selections_per_car) / selections_per_client = 9 := by
  sorry

end NUMINAMATH_CALUDE_car_selection_problem_l3718_371882


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3718_371829

theorem arithmetic_calculation : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3718_371829


namespace NUMINAMATH_CALUDE_inequality_upper_bound_upper_bound_tight_l3718_371877

theorem inequality_upper_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1) ≤ 2 + Real.sqrt 5 := by
  sorry

theorem upper_bound_tight : 
  ∀ ε > 0, ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧ 
  (2 + Real.sqrt 5) - (Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1)) < ε := by
  sorry

end NUMINAMATH_CALUDE_inequality_upper_bound_upper_bound_tight_l3718_371877


namespace NUMINAMATH_CALUDE_imaginary_part_of_minus_one_plus_i_squared_l3718_371858

theorem imaginary_part_of_minus_one_plus_i_squared :
  Complex.im ((-1 + Complex.I) ^ 2) = -2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_minus_one_plus_i_squared_l3718_371858


namespace NUMINAMATH_CALUDE_range_of_a_for_always_negative_l3718_371876

/-- The quadratic function f(x) = ax^2 + ax - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x - 4

/-- The predicate that f(x) < 0 for all real x -/
def always_negative (a : ℝ) : Prop := ∀ x, f a x < 0

/-- The theorem stating the range of a for which f(x) < 0 holds for all real x -/
theorem range_of_a_for_always_negative :
  ∀ a, always_negative a ↔ -16 < a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_always_negative_l3718_371876


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3718_371807

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (1 - Complex.I) / (1 + Complex.I)
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3718_371807


namespace NUMINAMATH_CALUDE_antons_number_l3718_371806

/-- Checks if two numbers match in exactly one digit place -/
def matchesOneDigit (a b : Nat) : Prop :=
  (a % 10 = b % 10 ∧ a / 10 ≠ b / 10) ∨
  (a / 10 % 10 = b / 10 % 10 ∧ a % 10 ≠ b % 10 ∧ a / 100 ≠ b / 100) ∨
  (a / 100 = b / 100 ∧ a % 100 ≠ b % 100)

theorem antons_number (x : Nat) :
  x ≥ 100 ∧ x < 1000 ∧
  matchesOneDigit x 109 ∧
  matchesOneDigit x 704 ∧
  matchesOneDigit x 124 →
  x = 729 := by
sorry

end NUMINAMATH_CALUDE_antons_number_l3718_371806


namespace NUMINAMATH_CALUDE_eight_digit_even_increasing_numbers_l3718_371821

theorem eight_digit_even_increasing_numbers (n : ℕ) (k : ℕ) : 
  n = 8 ∧ k = 4 → (n + k - 1).choose (k - 1) = 165 := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_even_increasing_numbers_l3718_371821


namespace NUMINAMATH_CALUDE_jake_present_weight_l3718_371866

/-- Jake's present weight in pounds -/
def jake_weight : ℕ := 194

/-- Kendra's weight in pounds -/
def kendra_weight : ℕ := 287 - jake_weight

/-- The amount of weight Jake needs to lose to weigh twice as much as Kendra -/
def weight_loss : ℕ := jake_weight - 2 * kendra_weight

theorem jake_present_weight : jake_weight = 194 := by
  sorry

end NUMINAMATH_CALUDE_jake_present_weight_l3718_371866


namespace NUMINAMATH_CALUDE_point_order_l3718_371859

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on the line y = -7x + 14 -/
def lies_on_line (p : Point) : Prop :=
  p.y = -7 * p.x + 14

theorem point_order (A B C : Point) 
  (hA : lies_on_line A) 
  (hB : lies_on_line B) 
  (hC : lies_on_line C) 
  (hx : A.x > C.x ∧ C.x > B.x) : 
  A.y < C.y ∧ C.y < B.y := by
  sorry

end NUMINAMATH_CALUDE_point_order_l3718_371859


namespace NUMINAMATH_CALUDE_y_bound_l3718_371836

theorem y_bound (x y : ℝ) (hx : x = 7) (heq : (x - 2*y)^y = 0.001) : 
  0 < y ∧ y < 7/2 := by
  sorry

end NUMINAMATH_CALUDE_y_bound_l3718_371836


namespace NUMINAMATH_CALUDE_triangle_special_condition_l3718_371851

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and area S, if 4√3S = (a+b)² - c², then the measure of angle C is π/3 -/
theorem triangle_special_condition (a b c S : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (area_eq : 4 * Real.sqrt 3 * S = (a + b)^2 - c^2) :
  ∃ (A B C : ℝ), 
    0 < A ∧ 0 < B ∧ 0 < C ∧ 
    A + B + C = π ∧
    S = 1/2 * a * b * Real.sin C ∧
    c^2 = a^2 + b^2 - 2*a*b*Real.cos C ∧
    C = π/3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_special_condition_l3718_371851


namespace NUMINAMATH_CALUDE_product_digit_sum_equals_800_l3718_371823

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Represents a number with n repeated digits of 7 -/
def repeated_sevens (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem product_digit_sum_equals_800 :
  sum_of_digits (8 * repeated_sevens 788) = 800 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_equals_800_l3718_371823


namespace NUMINAMATH_CALUDE_exam_score_l3718_371874

/-- Calculates the total marks in an examination based on given parameters. -/
def totalMarks (totalQuestions : ℕ) (correctMarks : ℤ) (wrongMarks : ℤ) (correctAnswers : ℕ) : ℤ :=
  (correctAnswers : ℤ) * correctMarks + (totalQuestions - correctAnswers : ℤ) * wrongMarks

/-- Theorem stating that under the given conditions, the student secures 130 marks. -/
theorem exam_score :
  totalMarks 80 4 (-1) 42 = 130 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_l3718_371874


namespace NUMINAMATH_CALUDE_f_monotonicity_and_zeros_l3718_371841

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - x - a

theorem f_monotonicity_and_zeros (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ∨
  (∃ z : ℝ, ∀ x y : ℝ, x < y → x < z → f a x > f a y) ∧
    (∀ x y : ℝ, x < y → z < x → f a x < f a y) ∧
  (∃! x₁ x₂ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧ x₁ ≠ x₂) ↔ a ∈ Set.Ioo 0 1 ∪ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_zeros_l3718_371841


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3718_371809

def arun_future_age : ℕ := 26
def years_to_future : ℕ := 6
def deepak_current_age : ℕ := 15

theorem age_ratio_proof :
  let arun_current_age := arun_future_age - years_to_future
  (arun_current_age : ℚ) / deepak_current_age = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3718_371809


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3718_371896

def arithmeticSequence (a₁ a₂ a₃ : ℕ) : ℕ → ℕ :=
  fun n => a₁ + (n - 1) * (a₂ - a₁)

theorem fifteenth_term_of_sequence (h : arithmeticSequence 3 14 25 3 = 25) :
  arithmeticSequence 3 14 25 15 = 157 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3718_371896


namespace NUMINAMATH_CALUDE_sams_correct_percentage_l3718_371839

theorem sams_correct_percentage (y : ℕ) : 
  let total_problems := 8 * y
  let missed_problems := 3 * y
  let correct_problems := total_problems - missed_problems
  (correct_problems : ℚ) / (total_problems : ℚ) * 100 = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_sams_correct_percentage_l3718_371839


namespace NUMINAMATH_CALUDE_range_of_a_l3718_371852

theorem range_of_a (a : ℝ) 
  (p : ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x)
  (q : ∃ x₀ : ℝ, x₀^2 + 4*x₀ + a = 0) :
  e ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3718_371852


namespace NUMINAMATH_CALUDE_car_average_speed_l3718_371885

/-- Calculates the average speed of a car given specific conditions during a 4-hour trip -/
theorem car_average_speed : 
  let first_hour_speed : ℝ := 145
  let second_hour_speed : ℝ := 60
  let stop_duration : ℝ := 1/3
  let fourth_hour_min_speed : ℝ := 45
  let fourth_hour_max_speed : ℝ := 100
  let total_time : ℝ := 4 + stop_duration
  let fourth_hour_avg_speed : ℝ := (fourth_hour_min_speed + fourth_hour_max_speed) / 2
  let total_distance : ℝ := first_hour_speed + second_hour_speed + fourth_hour_avg_speed
  let average_speed : ℝ := total_distance / total_time
  ∃ ε > 0, |average_speed - 64.06| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_car_average_speed_l3718_371885


namespace NUMINAMATH_CALUDE_min_k_for_sqrt_inequality_l3718_371891

theorem min_k_for_sqrt_inequality : 
  ∃ k : ℝ, k = Real.sqrt 2 ∧ 
  (∀ x y : ℝ, Real.sqrt x + Real.sqrt y ≤ k * Real.sqrt (x + y)) ∧
  (∀ k' : ℝ, k' < k → 
    ∃ x y : ℝ, Real.sqrt x + Real.sqrt y > k' * Real.sqrt (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_min_k_for_sqrt_inequality_l3718_371891


namespace NUMINAMATH_CALUDE_yearly_subscription_cost_proof_l3718_371834

/-- The yearly subscription cost to professional magazines, given that a 50% reduction
    in the budget results in spending $470 less. -/
def yearly_subscription_cost : ℝ := 940

theorem yearly_subscription_cost_proof :
  yearly_subscription_cost - yearly_subscription_cost / 2 = 470 := by
  sorry

end NUMINAMATH_CALUDE_yearly_subscription_cost_proof_l3718_371834


namespace NUMINAMATH_CALUDE_empire_state_building_height_l3718_371826

/-- The height of the Empire State Building to the top floor -/
def height_to_top_floor : ℝ := 1454 - 204

/-- The total height of the Empire State Building -/
def total_height : ℝ := 1454

/-- The height of the antenna spire -/
def antenna_height : ℝ := 204

theorem empire_state_building_height : height_to_top_floor = 1250 := by
  sorry

end NUMINAMATH_CALUDE_empire_state_building_height_l3718_371826


namespace NUMINAMATH_CALUDE_john_annual_oil_change_cost_l3718_371830

/-- Calculates the annual cost of oil changes for a driver named John. -/
theorem john_annual_oil_change_cost :
  ∀ (miles_per_month : ℕ) 
    (miles_per_oil_change : ℕ) 
    (free_oil_changes_per_year : ℕ) 
    (cost_per_oil_change : ℕ),
  miles_per_month = 1000 →
  miles_per_oil_change = 3000 →
  free_oil_changes_per_year = 1 →
  cost_per_oil_change = 50 →
  (12 * miles_per_month / miles_per_oil_change - free_oil_changes_per_year) * cost_per_oil_change = 150 :=
by
  sorry

#check john_annual_oil_change_cost

end NUMINAMATH_CALUDE_john_annual_oil_change_cost_l3718_371830


namespace NUMINAMATH_CALUDE_unique_integer_solution_l3718_371872

theorem unique_integer_solution : ∃! (x : ℕ+), (4 * x.val)^2 - 2 * x.val = 8066 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l3718_371872


namespace NUMINAMATH_CALUDE_right_triangle_geometric_mean_l3718_371813

theorem right_triangle_geometric_mean (a c : ℝ) (h₁ : 0 < a) (h₂ : 0 < c) :
  (c * c = a * c) → (a = (c * (Real.sqrt 5 - 1)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_geometric_mean_l3718_371813


namespace NUMINAMATH_CALUDE_f_iterative_application_l3718_371868

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 - 3*x + 2 else x + 10

theorem f_iterative_application : f (f (f (f (f 2)))) = -8 := by
  sorry

end NUMINAMATH_CALUDE_f_iterative_application_l3718_371868


namespace NUMINAMATH_CALUDE_train_length_l3718_371832

/-- Calculates the length of a train given its speed, the speed of a vehicle it overtakes, and the time it takes to overtake. -/
theorem train_length (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) : 
  train_speed = 100 →
  motorbike_speed = 64 →
  overtake_time = 40 →
  (train_speed - motorbike_speed) * overtake_time * (1000 / 3600) = 400 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3718_371832


namespace NUMINAMATH_CALUDE_copying_result_correct_l3718_371827

/-- Represents the copying cost and discount structure -/
structure CopyingCost where
  cost_per_5_pages : ℚ  -- Cost in cents for 5 pages
  budget : ℚ           -- Budget in dollars
  discount_rate : ℚ    -- Discount rate after 1000 pages
  discount_threshold : ℕ -- Number of pages after which discount applies

/-- Calculates the total number of pages that can be copied and the total cost with discount -/
def calculate_copying_result (c : CopyingCost) : ℕ × ℚ :=
  sorry

/-- Theorem stating the correctness of the calculation -/
theorem copying_result_correct (c : CopyingCost) :
  c.cost_per_5_pages = 10 ∧ 
  c.budget = 50 ∧ 
  c.discount_rate = 0.1 ∧
  c.discount_threshold = 1000 →
  calculate_copying_result c = (2500, 47) :=
sorry

end NUMINAMATH_CALUDE_copying_result_correct_l3718_371827


namespace NUMINAMATH_CALUDE_juniper_has_six_bones_l3718_371803

/-- Calculates the number of bones Juniper has remaining after her master doubles 
    her initial number of bones and the neighbor's dog steals two bones. -/
def junipersBones (initialBones : ℕ) : ℕ :=
  2 * initialBones - 2

/-- Theorem stating that Juniper has 6 bones remaining after the events. -/
theorem juniper_has_six_bones : junipersBones 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_juniper_has_six_bones_l3718_371803


namespace NUMINAMATH_CALUDE_root_conditions_imply_m_range_l3718_371890

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 3

-- State the theorem
theorem root_conditions_imply_m_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f m x = 0 ∧ f m y = 0 ∧ x > 1 ∧ y < 1) →
  m > 4 :=
sorry

end NUMINAMATH_CALUDE_root_conditions_imply_m_range_l3718_371890


namespace NUMINAMATH_CALUDE_ad_rate_per_square_foot_l3718_371808

-- Define the problem parameters
def num_companies : ℕ := 3
def ads_per_company : ℕ := 10
def ad_length : ℕ := 12
def ad_width : ℕ := 5
def total_paid : ℕ := 108000

-- Define the theorem
theorem ad_rate_per_square_foot :
  let total_area : ℕ := num_companies * ads_per_company * ad_length * ad_width
  let rate_per_square_foot : ℚ := total_paid / total_area
  rate_per_square_foot = 60 := by
  sorry

end NUMINAMATH_CALUDE_ad_rate_per_square_foot_l3718_371808


namespace NUMINAMATH_CALUDE_search_plans_count_l3718_371848

/-- Represents the number of children in the group -/
def total_children : ℕ := 6

/-- Represents the number of food drop locations -/
def num_locations : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the number of search plans when Grace doesn't participate -/
def plans_without_grace : ℕ := 
  (choose (total_children - 1) 1) * ((choose (total_children - 2) 2) / 2) * (Nat.factorial num_locations)

/-- Calculates the number of search plans when Grace participates -/
def plans_with_grace : ℕ := choose (total_children - 1) 2

/-- The total number of different search plans -/
def total_plans : ℕ := plans_without_grace + plans_with_grace

/-- Theorem stating that the total number of different search plans is 40 -/
theorem search_plans_count : total_plans = 40 := by
  sorry


end NUMINAMATH_CALUDE_search_plans_count_l3718_371848


namespace NUMINAMATH_CALUDE_sara_gave_dan_28_pears_l3718_371817

/-- The number of pears Sara initially picked -/
def initial_pears : ℕ := 35

/-- The number of pears Sara has left -/
def remaining_pears : ℕ := 7

/-- The number of pears Sara gave to Dan -/
def pears_given_to_dan : ℕ := initial_pears - remaining_pears

theorem sara_gave_dan_28_pears : pears_given_to_dan = 28 := by
  sorry

end NUMINAMATH_CALUDE_sara_gave_dan_28_pears_l3718_371817


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3718_371881

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 2) % 12 = 0 ∧
  (n - 2) % 16 = 0 ∧
  (n - 2) % 18 = 0 ∧
  (n - 2) % 21 = 0 ∧
  (n - 2) % 28 = 0 ∧
  (n - 2) % 32 = 0 ∧
  (n - 2) % 45 = 0

def is_sum_of_consecutive_primes (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ Nat.Prime (p + 1) ∧ n = p + (p + 1)

theorem smallest_number_satisfying_conditions :
  (is_divisible_by_all 10090 ∧ is_sum_of_consecutive_primes 10090) ∧
  ∀ m : ℕ, m < 10090 → ¬(is_divisible_by_all m ∧ is_sum_of_consecutive_primes m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3718_371881


namespace NUMINAMATH_CALUDE_original_ratio_proof_l3718_371897

theorem original_ratio_proof (initial_boarders : ℕ) (new_boarders : ℕ) :
  initial_boarders = 220 →
  new_boarders = 44 →
  (initial_boarders + new_boarders) * 2 = (initial_boarders + new_boarders + (initial_boarders + new_boarders) * 2) →
  (5 : ℚ) / 12 = initial_boarders / ((initial_boarders + new_boarders) * 2 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_original_ratio_proof_l3718_371897


namespace NUMINAMATH_CALUDE_converse_not_always_true_l3718_371864

-- Define the types for points, lines, and planes in space
variable (Point Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)  -- plane contains line
variable (perp : Line → Plane → Prop)      -- line perpendicular to plane
variable (perp_planes : Plane → Plane → Prop)  -- plane perpendicular to plane

-- State the theorem
theorem converse_not_always_true 
  (b : Line) (α β : Plane) : 
  ¬(∀ b α β, (contains α b ∧ perp b β → perp_planes α β) → 
             (perp_planes α β → contains α b ∧ perp b β)) :=
sorry

end NUMINAMATH_CALUDE_converse_not_always_true_l3718_371864


namespace NUMINAMATH_CALUDE_sum_of_integers_l3718_371810

theorem sum_of_integers (x y : ℕ+) (h1 : x - y = 10) (h2 : x * y = 56) : x + y = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3718_371810


namespace NUMINAMATH_CALUDE_same_solution_implies_a_equals_four_l3718_371878

theorem same_solution_implies_a_equals_four (a : ℝ) : 
  (∃ x : ℝ, 2 * x + 1 = 3 ∧ 2 - (a - x) / 3 = 1) → a = 4 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_equals_four_l3718_371878


namespace NUMINAMATH_CALUDE_number_of_rats_l3718_371814

/-- Given a total of 70 animals where the number of rats is 6 times the number of chihuahuas,
    prove that the number of rats is 60. -/
theorem number_of_rats (total : ℕ) (chihuahuas : ℕ) (rats : ℕ) 
    (h1 : total = 70)
    (h2 : total = chihuahuas + rats)
    (h3 : rats = 6 * chihuahuas) : 
  rats = 60 := by
  sorry

end NUMINAMATH_CALUDE_number_of_rats_l3718_371814


namespace NUMINAMATH_CALUDE_units_digit_of_n_l3718_371847

/-- Given two natural numbers a and b, returns true if a has units digit 9 -/
def hasUnitsDigit9 (a : ℕ) : Prop :=
  a % 10 = 9

/-- Given a natural number n, returns its units digit -/
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_n (a b n : ℕ) 
  (h1 : a * b = 34^8) 
  (h2 : hasUnitsDigit9 a) 
  (h3 : n = b) : 
  unitsDigit n = 4 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l3718_371847


namespace NUMINAMATH_CALUDE_share_division_l3718_371838

theorem share_division (total : ℝ) (a b c : ℝ) 
  (h_total : total = 400)
  (h_sum : a + b + c = total)
  (h_a : a = (2/3) * (b + c))
  (h_b : b = (6/9) * (a + c)) :
  a = 160 := by sorry

end NUMINAMATH_CALUDE_share_division_l3718_371838


namespace NUMINAMATH_CALUDE_exists_unique_affine_transformation_basis_exists_unique_affine_transformation_triangle_exists_unique_affine_transformation_parallelogram_l3718_371856

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

-- Define points and vectors
variable (O O' : V) (e1 e2 e1' e2' : V)
variable (A B C A1 B1 C1 : V)

-- Define affine transformation
def AffineTransformation (f : V → V) :=
  ∃ (T : V →L[ℝ] V) (b : V), ∀ x, f x = T x + b

-- Statement for part (a)
theorem exists_unique_affine_transformation_basis :
  ∃! f : V → V, AffineTransformation f ∧
  f O = O' ∧ f (O + e1) = O' + e1' ∧ f (O + e2) = O' + e2' :=
sorry

-- Statement for part (b)
theorem exists_unique_affine_transformation_triangle :
  ∃! f : V → V, AffineTransformation f ∧
  f A = A1 ∧ f B = B1 ∧ f C = C1 :=
sorry

-- Define parallelogram
def IsParallelogram (P Q R S : V) :=
  P - Q = S - R ∧ P - S = Q - R

-- Statement for part (c)
theorem exists_unique_affine_transformation_parallelogram
  (P Q R S P' Q' R' S' : V)
  (h1 : IsParallelogram P Q R S)
  (h2 : IsParallelogram P' Q' R' S') :
  ∃! f : V → V, AffineTransformation f ∧
  f P = P' ∧ f Q = Q' ∧ f R = R' ∧ f S = S' :=
sorry

end NUMINAMATH_CALUDE_exists_unique_affine_transformation_basis_exists_unique_affine_transformation_triangle_exists_unique_affine_transformation_parallelogram_l3718_371856


namespace NUMINAMATH_CALUDE_surface_classification_l3718_371888

/-- A surface in 3D space -/
inductive Surface
  | CircularCone
  | OneSheetHyperboloid
  | TwoSheetHyperboloid
  | EllipticParaboloid

/-- Determine the type of surface given its equation -/
def determine_surface_type (equation : ℝ → ℝ → ℝ → Prop) : Surface :=
  sorry

theorem surface_classification :
  (determine_surface_type (fun x y z => x^2 - y^2 = z^2) = Surface.CircularCone) ∧
  (determine_surface_type (fun x y z => -2*x^2 + 2*y^2 + z^2 = 4) = Surface.OneSheetHyperboloid) ∧
  (determine_surface_type (fun x y z => 2*x^2 - y^2 + z^2 + 2 = 0) = Surface.TwoSheetHyperboloid) ∧
  (determine_surface_type (fun x y z => 3*y^2 + 2*z^2 = 6*x) = Surface.EllipticParaboloid) :=
by
  sorry

end NUMINAMATH_CALUDE_surface_classification_l3718_371888


namespace NUMINAMATH_CALUDE_video_subscription_cost_l3718_371819

def monthly_cost : ℚ := 14
def num_people : ℕ := 2
def months_in_year : ℕ := 12

theorem video_subscription_cost :
  (monthly_cost / num_people) * months_in_year = 84 := by
sorry

end NUMINAMATH_CALUDE_video_subscription_cost_l3718_371819


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_smallest_primes_l3718_371833

/-- The five smallest prime numbers -/
def smallest_primes : List Nat := [2, 3, 5, 7, 11]

/-- A number is five-digit if it's between 10000 and 99999 inclusive -/
def is_five_digit (n : Nat) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem smallest_five_digit_divisible_by_smallest_primes :
  ∃ (n : Nat), is_five_digit n ∧ 
    (∀ p ∈ smallest_primes, n % p = 0) ∧
    (∀ m : Nat, is_five_digit m ∧ (∀ p ∈ smallest_primes, m % p = 0) → n ≤ m) ∧
    n = 11550 := by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_smallest_primes_l3718_371833


namespace NUMINAMATH_CALUDE_odd_number_induction_l3718_371850

theorem odd_number_induction (P : ℕ → Prop) 
  (base : P 1) 
  (step : ∀ k : ℕ, k ≥ 1 → P k → P (k + 2)) : 
  ∀ n : ℕ, n % 2 = 1 → P n := by
  sorry

end NUMINAMATH_CALUDE_odd_number_induction_l3718_371850


namespace NUMINAMATH_CALUDE_fred_total_games_l3718_371849

/-- The total number of basketball games Fred attended over two years -/
def total_games (games_this_year games_last_year : ℕ) : ℕ :=
  games_this_year + games_last_year

/-- Theorem stating that Fred attended 85 games in total -/
theorem fred_total_games : 
  total_games 60 25 = 85 := by
  sorry

end NUMINAMATH_CALUDE_fred_total_games_l3718_371849


namespace NUMINAMATH_CALUDE_min_cost_2009_l3718_371884

/-- Represents the available coin denominations in rubles -/
inductive Coin : Type
  | one : Coin
  | two : Coin
  | five : Coin
  | ten : Coin

/-- The value of a coin in rubles -/
def coinValue : Coin → Nat
  | Coin.one => 1
  | Coin.two => 2
  | Coin.five => 5
  | Coin.ten => 10

/-- An arithmetic expression using coins and operations -/
inductive Expr : Type
  | coin : Coin → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluates an expression to its numerical value -/
def eval : Expr → Int
  | Expr.coin c => coinValue c
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Calculates the cost of an expression in rubles -/
def cost : Expr → Nat
  | Expr.coin c => coinValue c
  | Expr.add e1 e2 => cost e1 + cost e2
  | Expr.sub e1 e2 => cost e1 + cost e2
  | Expr.mul e1 e2 => cost e1 + cost e2
  | Expr.div e1 e2 => cost e1 + cost e2

/-- Theorem: The minimum cost to represent 2009 is 23 rubles -/
theorem min_cost_2009 : 
  (∃ e : Expr, eval e = 2009 ∧ cost e = 23) ∧
  (∀ e : Expr, eval e = 2009 → cost e ≥ 23) := by sorry

end NUMINAMATH_CALUDE_min_cost_2009_l3718_371884


namespace NUMINAMATH_CALUDE_sheep_buying_equation_l3718_371869

/-- Represents the price of the sheep -/
def sheep_price (x : ℤ) : ℤ := 5 * x + 45

/-- Represents the total contribution when each person gives 7 coins -/
def contribution_7 (x : ℤ) : ℤ := 7 * x

theorem sheep_buying_equation (x : ℤ) : 
  sheep_price x = contribution_7 x - 3 := by sorry

end NUMINAMATH_CALUDE_sheep_buying_equation_l3718_371869
