import Mathlib

namespace min_m_value_x_range_l3085_308512

-- Define the conditions
def conditions (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 1

-- Part 1: Minimum value of m
theorem min_m_value (a b : ℝ) (h : conditions a b) :
  ∀ m : ℝ, (∀ a b : ℝ, conditions a b → a * b ≤ m) → m ≥ 1/4 :=
sorry

-- Part 2: Range of x
theorem x_range (a b : ℝ) (h : conditions a b) :
  ∀ x : ℝ, (4/a + 1/b ≥ |2*x - 1| - |x + 2|) ↔ -6 ≤ x ∧ x ≤ 12 :=
sorry

end min_m_value_x_range_l3085_308512


namespace walking_competition_analysis_l3085_308508

/-- The Chi-square statistic for a 2x2 contingency table -/
def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The critical value for 90% confidence in a Chi-square test with 1 degree of freedom -/
def critical_value : ℚ := 2706 / 1000

/-- The probability of selecting a Female Walking Star -/
def p_female_walking_star : ℚ := 14 / 70

/-- The number of trials in the binomial distribution -/
def num_trials : ℕ := 3

/-- The expected value of X (number of Female Walking Stars in a sample of 3) -/
def expected_value : ℚ := num_trials * p_female_walking_star

theorem walking_competition_analysis :
  let k_squared := chi_square 24 16 16 14
  k_squared < critical_value ∧ expected_value = 3/5 := by sorry

end walking_competition_analysis_l3085_308508


namespace expression_equality_l3085_308572

theorem expression_equality : 
  (-2^3 = (-2)^3) ∧ 
  (2^3 ≠ 2*3) ∧ 
  (-((-2)^2) ≠ (-2)^2) ∧ 
  (-3^2 ≠ 3^2) := by
  sorry

end expression_equality_l3085_308572


namespace scientific_notation_of_9600000_l3085_308540

/-- Proves that 9600000 is equal to 9.6 × 10^6 -/
theorem scientific_notation_of_9600000 : 9600000 = 9.6 * (10 ^ 6) := by
  sorry

end scientific_notation_of_9600000_l3085_308540


namespace price_reduction_percentage_l3085_308553

/-- Proves that given a price reduction x%, if the sale increases by 80% and the net effect on the sale is 53%, then x = 15. -/
theorem price_reduction_percentage (x : ℝ) : 
  (1 - x / 100) * 1.80 = 1.53 → x = 15 := by
  sorry

end price_reduction_percentage_l3085_308553


namespace partner_c_investment_l3085_308593

/-- Calculates the investment of partner C in a partnership business --/
theorem partner_c_investment 
  (a_investment : ℕ) 
  (b_investment : ℕ) 
  (total_profit : ℕ) 
  (a_profit_share : ℕ) 
  (h1 : a_investment = 6300)
  (h2 : b_investment = 4200)
  (h3 : total_profit = 12300)
  (h4 : a_profit_share = 3690) :
  ∃ c_investment : ℕ, 
    c_investment = 10500 ∧ 
    (a_investment : ℚ) / (a_investment + b_investment + c_investment : ℚ) = 
    (a_profit_share : ℚ) / (total_profit : ℚ) :=
by sorry

end partner_c_investment_l3085_308593


namespace ben_homework_theorem_l3085_308555

/-- The time in minutes Ben has to work on homework -/
def total_time : ℕ := 60

/-- The time taken to solve the i-th problem -/
def problem_time (i : ℕ) : ℕ := i

/-- The sum of time taken to solve the first n problems -/
def total_problem_time (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

/-- The maximum number of problems Ben can solve -/
def max_problems : ℕ := 10

theorem ben_homework_theorem :
  (∀ n : ℕ, n > max_problems → total_problem_time n > total_time) ∧
  total_problem_time max_problems ≤ total_time :=
sorry

end ben_homework_theorem_l3085_308555


namespace words_per_page_l3085_308501

theorem words_per_page (total_pages : Nat) (words_mod : Nat) (mod_value : Nat) :
  total_pages = 150 →
  words_mod = 210 →
  mod_value = 221 →
  ∃ (words_per_page : Nat),
    words_per_page ≤ 120 ∧
    (total_pages * words_per_page) % mod_value = words_mod ∧
    words_per_page = 195 := by
  sorry

end words_per_page_l3085_308501


namespace conjugate_complex_magnitude_l3085_308538

theorem conjugate_complex_magnitude (α β : ℂ) : 
  (∃ (x y : ℝ), α = x + y * Complex.I ∧ β = x - y * Complex.I) →  -- conjugate complex numbers
  (∃ (r : ℝ), α / β^3 = r) →  -- α/β³ is real
  Complex.abs (α - β) = 4 →  -- |α - β| = 4
  Complex.abs α = 4 * Real.sqrt 3 / 3 :=  -- |α| = 4√3/3
by sorry

end conjugate_complex_magnitude_l3085_308538


namespace min_value_reciprocal_sum_l3085_308537

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end min_value_reciprocal_sum_l3085_308537


namespace complex_fraction_simplification_l3085_308582

theorem complex_fraction_simplification (z : ℂ) (h : z = -1 + I) :
  (z + 2) / (z^2 + z) = -1 := by
  sorry

end complex_fraction_simplification_l3085_308582


namespace worker_a_completion_time_l3085_308571

/-- The time it takes for Worker A and Worker B to complete a job together and independently -/
def combined_time : ℝ := 2.857142857142857

/-- The time it takes for Worker B to complete the job alone -/
def worker_b_time : ℝ := 10

/-- The time it takes for Worker A to complete the job alone -/
def worker_a_time : ℝ := 4

/-- Theorem stating that Worker A takes 4 hours to complete the job alone -/
theorem worker_a_completion_time :
  (1 / worker_a_time + 1 / worker_b_time) * combined_time = 1 := by
  sorry

end worker_a_completion_time_l3085_308571


namespace max_dot_product_l3085_308599

/-- An ellipse with focus on the x-axis -/
structure Ellipse where
  /-- The b parameter in the ellipse equation x^2/4 + y^2/b^2 = 1 -/
  b : ℝ
  /-- The eccentricity of the ellipse -/
  e : ℝ
  /-- Condition that the eccentricity is 1/2 -/
  h_e : e = 1/2

/-- A point on the ellipse -/
structure PointOnEllipse (ε : Ellipse) where
  x : ℝ
  y : ℝ
  /-- The point satisfies the ellipse equation -/
  h_on_ellipse : x^2/4 + y^2/ε.b^2 = 1

/-- The left focus of the ellipse -/
def leftFocus (ε : Ellipse) : ℝ × ℝ := sorry

/-- The right vertex of the ellipse -/
def rightVertex (ε : Ellipse) : ℝ × ℝ := sorry

/-- The dot product of vectors PF and PA -/
def dotProduct (ε : Ellipse) (p : PointOnEllipse ε) : ℝ := sorry

/-- Theorem: The maximum value of the dot product of PF and PA is 4 -/
theorem max_dot_product (ε : Ellipse) :
  ∃ (max : ℝ), max = 4 ∧ ∀ (p : PointOnEllipse ε), dotProduct ε p ≤ max :=
sorry

end max_dot_product_l3085_308599


namespace only_one_solution_l3085_308596

def sum_of_squares (K : ℕ) : ℕ := K * (K + 1) * (2 * K + 1) / 6

theorem only_one_solution (K : ℕ) (M : ℕ) :
  sum_of_squares K = M^3 →
  M < 50 →
  K = 1 :=
by sorry

end only_one_solution_l3085_308596


namespace intersection_theorem_l3085_308510

-- Define the four lines
def line1 (x y : ℚ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℚ) : Prop := x + 3 * y = 3
def line3 (x y : ℚ) : Prop := 6 * x - 4 * y = 2
def line4 (x y : ℚ) : Prop := 5 * x - 15 * y = 15

-- Define the set of intersection points
def intersection_points : Set (ℚ × ℚ) :=
  {(18/11, 13/11), (21/11, 8/11)}

-- Define a function to check if a point lies on at least two lines
def on_at_least_two_lines (p : ℚ × ℚ) : Prop :=
  let (x, y) := p
  (line1 x y ∧ line2 x y) ∨ (line1 x y ∧ line3 x y) ∨ (line1 x y ∧ line4 x y) ∨
  (line2 x y ∧ line3 x y) ∨ (line2 x y ∧ line4 x y) ∨ (line3 x y ∧ line4 x y)

-- Theorem statement
theorem intersection_theorem :
  {p : ℚ × ℚ | on_at_least_two_lines p} = intersection_points := by sorry

end intersection_theorem_l3085_308510


namespace analysis_method_seeks_sufficient_condition_l3085_308588

/-- The analysis method for proving inequalities -/
def analysis_method : Type := Unit

/-- A condition that makes an inequality hold -/
def condition : Type := Unit

/-- Predicate indicating if a condition is sufficient -/
def is_sufficient (c : condition) : Prop := sorry

/-- The condition sought by the analysis method -/
def sought_condition (m : analysis_method) : condition := sorry

/-- Theorem stating that the analysis method seeks a sufficient condition -/
theorem analysis_method_seeks_sufficient_condition :
  ∀ (m : analysis_method), is_sufficient (sought_condition m) := by
  sorry

end analysis_method_seeks_sufficient_condition_l3085_308588


namespace table_rotation_l3085_308575

theorem table_rotation (table_length table_width : ℝ) (S : ℕ) : 
  table_length = 9 →
  table_width = 12 →
  S = ⌈(table_length^2 + table_width^2).sqrt⌉ →
  S = 15 :=
by
  sorry

end table_rotation_l3085_308575


namespace envelope_addressing_machines_l3085_308535

theorem envelope_addressing_machines (machine1_time machine2_time combined_time : ℚ) :
  machine1_time = 10 →
  combined_time = 4 →
  (1 / machine1_time + 1 / machine2_time = 1 / combined_time) →
  machine2_time = 20 / 3 := by
  sorry

end envelope_addressing_machines_l3085_308535


namespace original_paint_intensity_l3085_308580

/-- Proves that the intensity of the original red paint was 45% given the specified conditions. -/
theorem original_paint_intensity
  (replace_fraction : Real)
  (replacement_solution_intensity : Real)
  (final_intensity : Real)
  (h1 : replace_fraction = 0.25)
  (h2 : replacement_solution_intensity = 0.25)
  (h3 : final_intensity = 0.40) :
  ∃ (original_intensity : Real),
    original_intensity = 0.45 ∧
    (1 - replace_fraction) * original_intensity +
    replace_fraction * replacement_solution_intensity = final_intensity :=
by
  sorry

end original_paint_intensity_l3085_308580


namespace meaningful_expression_l3085_308573

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x + 1)) ↔ x > -1 :=
by sorry

end meaningful_expression_l3085_308573


namespace integer_fraction_count_l3085_308576

theorem integer_fraction_count : 
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0 ∧ ∃ k : ℕ, k > 0 ∧ 1722 = k * (m^2 - 3)) ∧ 
    S.card = 3 := by
  sorry

end integer_fraction_count_l3085_308576


namespace lunks_needed_correct_l3085_308532

/-- Exchange rate between lunks and kunks -/
def lunk_to_kunk_rate : ℚ := 1/2

/-- Exchange rate between kunks and apples -/
def kunk_to_apple_rate : ℚ := 5/3

/-- Number of apples to purchase -/
def apples_to_buy : ℕ := 20

/-- The number of lunks needed to purchase the given number of apples -/
def lunks_needed : ℕ := 24

theorem lunks_needed_correct : 
  ↑lunks_needed = ↑apples_to_buy / (kunk_to_apple_rate * lunk_to_kunk_rate) := by
  sorry

end lunks_needed_correct_l3085_308532


namespace solution_values_l3085_308544

/-- A quadratic function f(x) = ax^2 - 2(a+1)x + b where a and b are real numbers. -/
def f (a b x : ℝ) : ℝ := a * x^2 - 2 * (a + 1) * x + b

/-- The property that the solution set of f(x) < 0 is (1,2) -/
def solution_set_property (a b : ℝ) : Prop :=
  ∀ x, f a b x < 0 ↔ 1 < x ∧ x < 2

/-- Theorem stating that if the solution set property holds, then a = 2 and b = 4 -/
theorem solution_values (a b : ℝ) (h : solution_set_property a b) : a = 2 ∧ b = 4 := by
  sorry


end solution_values_l3085_308544


namespace cost_doubling_l3085_308557

theorem cost_doubling (t b : ℝ) (t_pos : t > 0) (b_pos : b > 0) :
  let original_cost := t * b^4
  let new_cost := t * (2*b)^4
  (new_cost / original_cost) * 100 = 1600 := by
  sorry

end cost_doubling_l3085_308557


namespace stadium_perimeter_stadium_breadth_l3085_308578

/-- Represents a rectangular stadium -/
structure Stadium where
  perimeter : ℝ
  length : ℝ
  breadth : ℝ

/-- The perimeter of a rectangle is twice the sum of its length and breadth -/
theorem stadium_perimeter (s : Stadium) : s.perimeter = 2 * (s.length + s.breadth) := by sorry

/-- Given a stadium with perimeter 800 and length 100, its breadth is 300 -/
theorem stadium_breadth : 
  ∀ (s : Stadium), s.perimeter = 800 ∧ s.length = 100 → s.breadth = 300 := by sorry

end stadium_perimeter_stadium_breadth_l3085_308578


namespace average_of_pqrs_l3085_308581

theorem average_of_pqrs (p q r s : ℝ) (h : (5 / 4) * (p + q + r + s) = 20) :
  (p + q + r + s) / 4 = 4 := by
  sorry

end average_of_pqrs_l3085_308581


namespace remainder_of_least_number_l3085_308542

theorem remainder_of_least_number (n : ℕ) (h1 : n = 261) (h2 : ∀ m < n, m % 37 ≠ n % 37 ∨ m % 7 ≠ n % 7) : n % 7 = 2 := by
  sorry

end remainder_of_least_number_l3085_308542


namespace marias_coffee_order_l3085_308556

/-- Maria's daily coffee order calculation -/
theorem marias_coffee_order (visits_per_day : ℕ) (cups_per_visit : ℕ)
  (h1 : visits_per_day = 2)
  (h2 : cups_per_visit = 3) :
  visits_per_day * cups_per_visit = 6 := by
  sorry

end marias_coffee_order_l3085_308556


namespace inequality_range_l3085_308565

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + m * x - 4 < 2 * x^2 + 2 * x - 1) ↔ 
  (m > -10 ∧ m ≤ 2) := by
sorry

end inequality_range_l3085_308565


namespace marble_statue_weight_l3085_308569

/-- The weight of a marble statue after three successive reductions -/
def final_weight (original : ℝ) : ℝ :=
  original * (1 - 0.28) * (1 - 0.18) * (1 - 0.20)

/-- Theorem stating the relationship between the original and final weights -/
theorem marble_statue_weight (original : ℝ) :
  final_weight original = 85.0176 → original = 144 := by
  sorry

#eval final_weight 144

end marble_statue_weight_l3085_308569


namespace derivative_at_two_l3085_308520

open Real

theorem derivative_at_two (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x > 0, f x = x^2 + 3 * x * (deriv f 2) - log x) : 
  deriv f 2 = -7/4 := by
  sorry

end derivative_at_two_l3085_308520


namespace quadratic_root_zero_l3085_308507

/-- The quadratic equation (a-1)x^2 + x + a^2 - 1 = 0 has 0 as one of its roots if and only if a = -1 -/
theorem quadratic_root_zero (a : ℝ) : 
  (∃ x : ℝ, (a - 1) * x^2 + x + a^2 - 1 = 0 ∧ x = 0) ↔ a = -1 :=
by sorry

end quadratic_root_zero_l3085_308507


namespace sum_digits_base7_of_777_l3085_308570

-- Define a function to convert a number from base 10 to base 7
def toBase7 (n : ℕ) : List ℕ := sorry

-- Define a function to sum the digits of a number represented as a list
def sumDigits (digits : List ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_digits_base7_of_777 : sumDigits (toBase7 777) = 9 := by sorry

end sum_digits_base7_of_777_l3085_308570


namespace shortest_altitude_of_triangle_l3085_308516

/-- Given a triangle with sides 9, 12, and 15, its shortest altitude has length 7.2 -/
theorem shortest_altitude_of_triangle (a b c h : ℝ) : 
  a = 9 ∧ b = 12 ∧ c = 15 ∧ 
  a^2 + b^2 = c^2 ∧
  h * c = 2 * (1/2 * a * b) →
  h = 7.2 := by
  sorry

end shortest_altitude_of_triangle_l3085_308516


namespace simplify_expression_l3085_308505

theorem simplify_expression : 
  (Real.sqrt 308 / Real.sqrt 77) - (Real.sqrt 245 / Real.sqrt 49) = 2 - Real.sqrt 5 := by
  sorry

end simplify_expression_l3085_308505


namespace factory_output_increase_l3085_308514

theorem factory_output_increase (P : ℝ) : 
  (1 + P / 100) * (1 + 20 / 100) * (1 - 24.242424242424242 / 100) = 1 → P = 10 := by
sorry

end factory_output_increase_l3085_308514


namespace solve_average_weight_problem_l3085_308563

def average_weight_problem (initial_average : ℝ) (new_man_weight : ℝ) (weight_increase : ℝ) (crew_size : ℕ) : Prop :=
  let replaced_weight := new_man_weight - (crew_size : ℝ) * weight_increase
  replaced_weight = initial_average * (crew_size : ℝ) + weight_increase * (crew_size : ℝ) - new_man_weight

theorem solve_average_weight_problem :
  average_weight_problem 0 71 1.8 10 = true :=
sorry

end solve_average_weight_problem_l3085_308563


namespace election_theorem_l3085_308543

theorem election_theorem (winner_percentage : ℝ) (winner_margin : ℕ) (winner_votes : ℕ) :
  winner_percentage = 0.62 →
  winner_votes = 992 →
  winner_margin = 384 →
  ∃ (total_votes : ℕ) (runner_up_votes : ℕ),
    total_votes = 1600 ∧
    runner_up_votes = 608 ∧
    winner_votes = winner_percentage * total_votes ∧
    winner_votes = runner_up_votes + winner_margin :=
by
  sorry

#check election_theorem

end election_theorem_l3085_308543


namespace base_edges_same_color_l3085_308523

/-- A color type representing red or green -/
inductive Color
| Red
| Green

/-- A vertex of the prism -/
structure Vertex where
  base : Bool  -- True for top base, False for bottom base
  index : Fin 5

/-- An edge of the prism -/
structure Edge where
  v1 : Vertex
  v2 : Vertex

/-- A prism with pentagonal bases -/
structure Prism where
  /-- The color of each edge -/
  edge_color : Edge → Color
  /-- Ensure that any triangle has edges of different colors -/
  triangle_property : ∀ (v1 v2 v3 : Vertex),
    v1 ≠ v2 ∧ v2 ≠ v3 ∧ v3 ≠ v1 →
    (edge_color ⟨v1, v2⟩ ≠ edge_color ⟨v2, v3⟩ ∨
     edge_color ⟨v2, v3⟩ ≠ edge_color ⟨v3, v1⟩ ∨
     edge_color ⟨v3, v1⟩ ≠ edge_color ⟨v1, v2⟩)

/-- The main theorem -/
theorem base_edges_same_color (p : Prism) :
  (∀ (i j : Fin 5), p.edge_color ⟨⟨true, i⟩, ⟨true, j⟩⟩ = p.edge_color ⟨⟨false, i⟩, ⟨false, j⟩⟩) :=
sorry

end base_edges_same_color_l3085_308523


namespace triangle_angle_values_l3085_308517

theorem triangle_angle_values (a b c A B C : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given conditions
  b * Real.sin A = Real.sqrt 3 * a * Real.cos B →
  Real.cos A * Real.sin C = (Real.sqrt 3 - 1) / 4 →
  -- Conclusions
  B = π / 3 ∧ A = 5 * π / 12 := by
  sorry

end triangle_angle_values_l3085_308517


namespace sector_central_angle_l3085_308559

theorem sector_central_angle (radius : ℝ) (area : ℝ) (angle : ℝ) :
  radius = 1 →
  area = 1 →
  area = (1 / 2) * angle * radius^2 →
  angle = 2 := by
sorry

end sector_central_angle_l3085_308559


namespace abs_diff_eq_diff_abs_implies_product_nonneg_but_not_conversely_l3085_308591

theorem abs_diff_eq_diff_abs_implies_product_nonneg_but_not_conversely (a b : ℝ) :
  (∀ a b : ℝ, |a - b| = |a| - |b| → a * b ≥ 0) ∧
  (∃ a b : ℝ, a * b ≥ 0 ∧ |a - b| ≠ |a| - |b|) :=
by sorry

end abs_diff_eq_diff_abs_implies_product_nonneg_but_not_conversely_l3085_308591


namespace bus_seat_capacity_l3085_308527

/-- Represents the seating arrangement and capacity of a bus -/
structure BusSeating where
  left_seats : ℕ
  right_seats : ℕ
  back_seat_capacity : ℕ
  total_capacity : ℕ

/-- Calculates the number of people each regular seat can hold -/
def seats_capacity (bus : BusSeating) : ℚ :=
  (bus.total_capacity - bus.back_seat_capacity) / (bus.left_seats + bus.right_seats)

/-- Theorem stating that for the given bus configuration, each seat can hold 3 people -/
theorem bus_seat_capacity :
  let bus : BusSeating := {
    left_seats := 15,
    right_seats := 12,
    back_seat_capacity := 7,
    total_capacity := 88
  }
  seats_capacity bus = 3 := by sorry

end bus_seat_capacity_l3085_308527


namespace geometric_sequence_sixth_term_l3085_308518

/-- A geometric sequence with first term 512 and 8th term 2 has 6th term equal to 16 -/
theorem geometric_sequence_sixth_term : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) = a n * (a 8 / a 7)) →  -- Geometric sequence property
  a 1 = 512 →                            -- First term is 512
  a 8 = 2 →                              -- 8th term is 2
  a 6 = 16 :=                            -- 6th term is 16
by
  sorry


end geometric_sequence_sixth_term_l3085_308518


namespace jimin_candies_count_l3085_308529

/-- The number of candies Jimin gave to Yuna -/
def candies_to_yuna : ℕ := 25

/-- The number of candies Jimin gave to her sister -/
def candies_to_sister : ℕ := 13

/-- The total number of candies Jimin had at first -/
def total_candies : ℕ := candies_to_yuna + candies_to_sister

theorem jimin_candies_count : total_candies = 38 := by
  sorry

end jimin_candies_count_l3085_308529


namespace cubic_sum_minus_product_l3085_308549

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 12) 
  (sum_products_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 648 := by
  sorry

end cubic_sum_minus_product_l3085_308549


namespace space_probe_distance_l3085_308577

theorem space_probe_distance (total_distance : ℕ) (distance_after_refuel : ℕ) 
  (h1 : total_distance = 5555555555555)
  (h2 : distance_after_refuel = 3333333333333) :
  total_distance - distance_after_refuel = 2222222222222 := by
  sorry

end space_probe_distance_l3085_308577


namespace susan_chair_count_l3085_308583

/-- The number of chairs in Susan's house -/
def total_chairs (red : ℕ) (yellow : ℕ) (blue : ℕ) : ℕ := red + yellow + blue

/-- Susan's chair collection -/
structure SusanChairs where
  red : ℕ
  yellow : ℕ
  blue : ℕ
  red_count : red = 5
  yellow_count : yellow = 4 * red
  blue_count : blue = yellow - 2

theorem susan_chair_count (s : SusanChairs) : total_chairs s.red s.yellow s.blue = 43 := by
  sorry

end susan_chair_count_l3085_308583


namespace sarah_hardback_count_l3085_308590

/-- The number of hardback books Sarah bought -/
def sarah_hardback : ℕ := sorry

/-- The number of paperback books Sarah bought -/
def sarah_paperback : ℕ := 6

/-- The number of paperback books Sarah's brother bought -/
def brother_paperback : ℕ := sarah_paperback / 3

/-- The number of hardback books Sarah's brother bought -/
def brother_hardback : ℕ := 2 * sarah_hardback

/-- The total number of books Sarah's brother bought -/
def brother_total : ℕ := 10

theorem sarah_hardback_count : sarah_hardback = 4 := by
  sorry

end sarah_hardback_count_l3085_308590


namespace nonagon_diagonal_intersection_probability_l3085_308546

-- Define the number of sides in a nonagon
def nonagon_sides : ℕ := 9

-- Define the number of diagonals in a nonagon
def nonagon_diagonals : ℕ := 27

-- Define a function to calculate combinations
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem nonagon_diagonal_intersection_probability :
  let total_diagonal_pairs := choose nonagon_diagonals 2
  let intersecting_diagonal_pairs := choose nonagon_sides 4
  (intersecting_diagonal_pairs : ℚ) / total_diagonal_pairs = 14 / 39 := by
  sorry

end nonagon_diagonal_intersection_probability_l3085_308546


namespace no_valid_labeling_l3085_308539

/-- A labeling function that assigns one of four labels to each point in the integer lattice. -/
def Labeling := ℤ × ℤ → Fin 4

/-- Predicate that checks if a given labeling satisfies the constraints on a unit square. -/
def valid_square (f : Labeling) (x y : ℤ) : Prop :=
  f (x, y) ≠ f (x + 1, y) ∧
  f (x, y) ≠ f (x, y + 1) ∧
  f (x, y) ≠ f (x + 1, y + 1) ∧
  f (x + 1, y) ≠ f (x, y + 1) ∧
  f (x + 1, y) ≠ f (x + 1, y + 1) ∧
  f (x, y + 1) ≠ f (x + 1, y + 1)

/-- Predicate that checks if a given labeling satisfies the constraints on a row. -/
def valid_row (f : Labeling) (y : ℤ) : Prop :=
  ∀ x : ℤ, ∃ i j k l : ℤ, i < j ∧ j < k ∧ k < l ∧
    f (i, y) ≠ f (j, y) ∧ f (j, y) ≠ f (k, y) ∧ f (k, y) ≠ f (l, y) ∧
    f (i, y) ≠ f (k, y) ∧ f (i, y) ≠ f (l, y) ∧ f (j, y) ≠ f (l, y)

/-- Predicate that checks if a given labeling satisfies the constraints on a column. -/
def valid_column (f : Labeling) (x : ℤ) : Prop :=
  ∀ y : ℤ, ∃ i j k l : ℤ, i < j ∧ j < k ∧ k < l ∧
    f (x, i) ≠ f (x, j) ∧ f (x, j) ≠ f (x, k) ∧ f (x, k) ≠ f (x, l) ∧
    f (x, i) ≠ f (x, k) ∧ f (x, i) ≠ f (x, l) ∧ f (x, j) ≠ f (x, l)

/-- Theorem stating that no labeling can satisfy all the given constraints. -/
theorem no_valid_labeling : ¬∃ f : Labeling, 
  (∀ x y : ℤ, valid_square f x y) ∧ 
  (∀ y : ℤ, valid_row f y) ∧ 
  (∀ x : ℤ, valid_column f x) := by
  sorry

end no_valid_labeling_l3085_308539


namespace absolute_value_zero_implies_negative_three_l3085_308564

theorem absolute_value_zero_implies_negative_three (a : ℝ) :
  |a + 3| = 0 → a = -3 := by
sorry

end absolute_value_zero_implies_negative_three_l3085_308564


namespace quadratic_equation_properties_l3085_308560

theorem quadratic_equation_properties (k : ℝ) :
  (∃ x y : ℝ, x^2 - k*x + k - 1 = 0 ∧ y^2 - k*y + k - 1 = 0 ∧ (x = y ∨ x ≠ y)) ∧
  (∃ x : ℝ, x^2 - k*x + k - 1 = 0 ∧ x < 0) → k < 1 :=
by sorry

end quadratic_equation_properties_l3085_308560


namespace p_necessary_not_sufficient_for_q_l3085_308552

theorem p_necessary_not_sufficient_for_q :
  (∃ x, x < 2 ∧ ¬(-2 < x ∧ x < 2)) ∧
  (∀ x, -2 < x ∧ x < 2 → x < 2) :=
by sorry

end p_necessary_not_sufficient_for_q_l3085_308552


namespace tangent_slope_at_half_l3085_308554

-- Define the function f(x) = x^3 - 2
def f (x : ℝ) : ℝ := x^3 - 2

-- State the theorem
theorem tangent_slope_at_half :
  (deriv f) (1/2) = 3/4 := by
  sorry

end tangent_slope_at_half_l3085_308554


namespace cab_journey_time_l3085_308589

/-- Given a cab traveling at 5/6 of its usual speed is 8 minutes late, 
    prove that its usual time to cover the journey is 48 minutes. -/
theorem cab_journey_time (usual_time : ℝ) : 
  (5 / 6 : ℝ) * usual_time + 8 = usual_time → usual_time = 48 :=
by sorry

end cab_journey_time_l3085_308589


namespace technician_round_trip_l3085_308541

theorem technician_round_trip 
  (D : ℝ) 
  (P : ℝ) 
  (h1 : D > 0) -- Ensure distance is positive
  (h2 : 0 ≤ P ∧ P ≤ 100) -- Ensure percentage is between 0 and 100
  (h3 : D + (P / 100) * D = 0.7 * (2 * D)) -- Total distance traveled equals 70% of round-trip
  : P = 40 := by
sorry

end technician_round_trip_l3085_308541


namespace xy_xz_yz_bounds_l3085_308519

theorem xy_xz_yz_bounds (x y z : ℝ) (h : 3 * (x + y + z) = x^2 + y^2 + z^2) :
  (∃ (a b c : ℝ), a + b + c = x + y + z ∧ a * b + b * c + c * a = 27) ∧
  (∃ (d e f : ℝ), d + e + f = x + y + z ∧ d * e + e * f + f * d = 0) ∧
  (∀ (u v w : ℝ), u + v + w = x + y + z → u * v + v * w + w * u ≤ 27) ∧
  (∀ (u v w : ℝ), u + v + w = x + y + z → u * v + v * w + w * u ≥ 0) :=
by sorry

end xy_xz_yz_bounds_l3085_308519


namespace complex_number_powers_l3085_308506

theorem complex_number_powers (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^97 + z^98 + z^99 + z^100 + z^101 = -1 := by
  sorry

end complex_number_powers_l3085_308506


namespace toms_initial_books_l3085_308561

/-- Given that Tom sold 4 books, bought 38 new books, and now has 39 books,
    prove that he initially had 5 books. -/
theorem toms_initial_books :
  ∀ (initial_books : ℕ),
    initial_books - 4 + 38 = 39 →
    initial_books = 5 := by
  sorry

end toms_initial_books_l3085_308561


namespace proposition_b_is_true_l3085_308511

theorem proposition_b_is_true : 3 > 4 ∨ 3 < 4 := by
  sorry

end proposition_b_is_true_l3085_308511


namespace smallest_undefined_inverse_l3085_308502

theorem smallest_undefined_inverse (a : ℕ) : 
  (∀ b < a, (Nat.gcd b 70 = 1 ∨ Nat.gcd b 84 = 1)) → 
  (Nat.gcd a 70 > 1 ∧ Nat.gcd a 84 > 1) → 
  a = 14 := by sorry

end smallest_undefined_inverse_l3085_308502


namespace xyz_mod_8_l3085_308526

theorem xyz_mod_8 (x y z : ℕ) : 
  x < 8 → y < 8 → z < 8 → x > 0 → y > 0 → z > 0 →
  (x * y * z) % 8 = 1 → 
  (3 * z) % 8 = 5 → 
  (7 * y) % 8 = (4 + y) % 8 → 
  (x + y + z) % 8 = 1 := by
  sorry

end xyz_mod_8_l3085_308526


namespace same_sign_product_and_quotient_abs_l3085_308598

theorem same_sign_product_and_quotient_abs (a b : ℚ) (hb : b ≠ 0) :
  (a * b > 0 ↔ |a| / |b| > 0) ∧ (a * b < 0 ↔ |a| / |b| < 0) ∧ (a * b = 0 ↔ |a| / |b| = 0) :=
sorry

end same_sign_product_and_quotient_abs_l3085_308598


namespace total_people_is_36_l3085_308574

/-- A circular arrangement of people shaking hands -/
structure HandshakeCircle where
  people : ℕ
  handshakes : ℕ
  smallest_set : ℕ

/-- The number of people in the circle equals the number of handshakes -/
def handshakes_equal_people (circle : HandshakeCircle) : Prop :=
  circle.people = circle.handshakes

/-- The smallest set size plus the remaining people equals the total people -/
def smallest_set_property (circle : HandshakeCircle) : Prop :=
  circle.smallest_set + (circle.people - circle.smallest_set) = circle.people

/-- The main theorem: given the conditions, prove the total number of people is 36 -/
theorem total_people_is_36 (circle : HandshakeCircle) 
    (h1 : circle.handshakes = 36)
    (h2 : circle.smallest_set = 12)
    (h3 : handshakes_equal_people circle)
    (h4 : smallest_set_property circle) : 
  circle.people = 36 := by
  sorry

#check total_people_is_36

end total_people_is_36_l3085_308574


namespace soccer_league_games_l3085_308524

theorem soccer_league_games (n : ℕ) (h : n = 11) : 
  (n * (n - 1)) / 2 = 55 := by
  sorry

end soccer_league_games_l3085_308524


namespace line_perp_plane_implies_planes_perp_l3085_308515

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular and subset relations
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_planes_perp
  (α β : Plane) (l : Line)
  (h1 : α ≠ β)
  (h2 : subset l α)
  (h3 : perpendicular_line_plane l β) :
  perpendicular α β :=
sorry

end line_perp_plane_implies_planes_perp_l3085_308515


namespace point_on_line_l3085_308522

/-- Given two points (m, n) and (m + a, n + 1.5) on the line x = 2y + 5, prove that a = 3 -/
theorem point_on_line (m n a : ℝ) : 
  (m = 2 * n + 5) → 
  (m + a = 2 * (n + 1.5) + 5) → 
  a = 3 := by sorry

end point_on_line_l3085_308522


namespace restaurant_bill_division_l3085_308579

theorem restaurant_bill_division (total_bill : ℕ) (individual_payment : ℕ) (num_friends : ℕ) :
  total_bill = 135 →
  individual_payment = 45 →
  total_bill = individual_payment * num_friends →
  num_friends = 3 := by
  sorry

end restaurant_bill_division_l3085_308579


namespace units_digit_of_expression_l3085_308594

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of the given expression -/
def expression : ℕ := 8 * 19 * 1981 - 8^3

theorem units_digit_of_expression :
  unitsDigit expression = 0 := by
  sorry

end units_digit_of_expression_l3085_308594


namespace max_polygon_length_8x8_grid_l3085_308558

/-- Represents a square grid -/
structure SquareGrid where
  size : ℕ

/-- Represents a polygon on a grid -/
structure GridPolygon where
  grid : SquareGrid
  length : ℕ
  closed : Bool
  self_avoiding : Bool

/-- The maximum length of a closed self-avoiding polygon on an 8x8 grid is 80 -/
theorem max_polygon_length_8x8_grid :
  ∃ (p : GridPolygon), p.grid.size = 8 ∧ p.closed ∧ p.self_avoiding ∧
    p.length = 80 ∧
    ∀ (q : GridPolygon), q.grid.size = 8 → q.closed → q.self_avoiding →
      q.length ≤ p.length := by
  sorry

end max_polygon_length_8x8_grid_l3085_308558


namespace max_earnings_theorem_l3085_308567

/-- Represents the exchange rates for a given day -/
structure ExchangeRates where
  gbp_to_usd : ℝ
  jpy_to_usd : ℝ
  eur_to_usd : ℝ

/-- Calculates the maximum total earnings in USD -/
def max_total_earnings (usd_hours : ℝ) (gbp_hours : ℝ) (jpy_hours : ℝ) (eur_hours : ℝ)
  (usd_rate : ℝ) (gbp_rate : ℝ) (jpy_rate : ℝ) (eur_rate : ℝ)
  (day1 : ExchangeRates) (day2 : ExchangeRates) (day3 : ExchangeRates) : ℝ :=
  sorry

/-- Theorem stating that the maximum total earnings is $32.61 -/
theorem max_earnings_theorem :
  let day1 : ExchangeRates := { gbp_to_usd := 1.35, jpy_to_usd := 0.009, eur_to_usd := 1.18 }
  let day2 : ExchangeRates := { gbp_to_usd := 1.38, jpy_to_usd := 0.0085, eur_to_usd := 1.20 }
  let day3 : ExchangeRates := { gbp_to_usd := 1.33, jpy_to_usd := 0.0095, eur_to_usd := 1.21 }
  max_total_earnings 4 0.5 1.5 1 5 3 400 4 day1 day2 day3 = 32.61 := by
  sorry

end max_earnings_theorem_l3085_308567


namespace player_one_wins_l3085_308592

/-- A cubic polynomial with integer coefficients -/
def CubicPolynomial (a b c : ℤ) : ℤ → ℤ := fun x ↦ x^3 + a*x^2 + b*x + c

/-- A proposition stating that a cubic polynomial has three integer roots -/
def HasThreeIntegerRoots (p : ℤ → ℤ) : Prop :=
  ∃ x y z : ℤ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ p x = 0 ∧ p y = 0 ∧ p z = 0

theorem player_one_wins :
  ∀ a b : ℤ, ∃ c : ℤ, HasThreeIntegerRoots (CubicPolynomial a b c) :=
by sorry

end player_one_wins_l3085_308592


namespace divide_number_80_l3085_308551

theorem divide_number_80 (smaller larger : ℝ) : 
  smaller + larger = 80 ∧ 
  larger / 2 = smaller + 10 → 
  smaller = 20 ∧ larger = 60 := by
sorry

end divide_number_80_l3085_308551


namespace catch_up_equation_correct_l3085_308587

/-- Represents the problem of two horses racing, where one starts earlier than the other. -/
structure HorseRace where
  fast_speed : ℕ  -- Speed of the faster horse in miles per day
  slow_speed : ℕ  -- Speed of the slower horse in miles per day
  head_start : ℕ  -- Number of days the slower horse starts earlier

/-- The equation representing when the faster horse catches up to the slower horse -/
def catch_up_equation (race : HorseRace) (x : ℝ) : Prop :=
  (race.fast_speed : ℝ) * x = (race.slow_speed : ℝ) * (x + race.head_start)

/-- The specific race described in the problem -/
def zhu_shijie_race : HorseRace :=
  { fast_speed := 240
  , slow_speed := 150
  , head_start := 12 }

/-- Theorem stating that the given equation correctly represents the race situation -/
theorem catch_up_equation_correct :
  catch_up_equation zhu_shijie_race = fun x => 240 * x = 150 * (x + 12) :=
by sorry


end catch_up_equation_correct_l3085_308587


namespace coloring_books_sold_l3085_308568

theorem coloring_books_sold (initial_stock : ℕ) (shelves : ℕ) (books_per_shelf : ℕ) : initial_stock = 87 → shelves = 9 → books_per_shelf = 6 → initial_stock - (shelves * books_per_shelf) = 33 := by
  sorry

end coloring_books_sold_l3085_308568


namespace ratio_of_percentages_l3085_308530

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.25 * P) 
  (hN : N = 0.6 * P) 
  (hP : P ≠ 0) : 
  M / N = 1 / 6 := by
  sorry

end ratio_of_percentages_l3085_308530


namespace abs_h_value_l3085_308528

theorem abs_h_value (h : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    (x₁^4 + 4*h*x₁^2 = 2) ∧ 
    (x₂^4 + 4*h*x₂^2 = 2) ∧ 
    (x₃^4 + 4*h*x₃^2 = 2) ∧ 
    (x₄^4 + 4*h*x₄^2 = 2) ∧ 
    (x₁^2 + x₂^2 + x₃^2 + x₄^2 = 34)) → 
  |h| = 17/4 := by
sorry

end abs_h_value_l3085_308528


namespace rectangle_segment_length_l3085_308547

/-- Given a rectangle ABCD with side lengths AB = 6 and BC = 5,
    and a segment GH through B perpendicular to DB,
    with A on DG and C on DH, prove that GH = 11√61/6 -/
theorem rectangle_segment_length (A B C D G H : ℝ × ℝ) :
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let DB := Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2)
  let GH := Real.sqrt ((G.1 - H.1)^2 + (G.2 - H.2)^2)
  AB = 6 →
  BC = 5 →
  (G.1 - H.1) * (D.1 - B.1) + (G.2 - H.2) * (D.2 - B.2) = 0 →  -- GH ⟂ DB
  ∃ t₁ : ℝ, A = t₁ • (G - D) + D →  -- A lies on DG
  ∃ t₂ : ℝ, C = t₂ • (H - D) + D →  -- C lies on DH
  GH = 11 * Real.sqrt 61 / 6 := by
  sorry


end rectangle_segment_length_l3085_308547


namespace tree_calculation_l3085_308503

theorem tree_calculation (T P R : ℝ) (h1 : T = 400) (h2 : P = 0.20) (h3 : R = 5) :
  T - (P * T) + (P * T * R) = 720 :=
by sorry

end tree_calculation_l3085_308503


namespace student_distribution_theorem_l3085_308513

/-- The number of ways to distribute students among attractions -/
def distribute_students (n m k : ℕ) : ℕ :=
  Nat.choose n k * (m - 1)^(n - k)

/-- Theorem: The number of ways to distribute 6 students among 6 attractions,
    where exactly 2 students visit a specific attraction, is C₆² × 5⁴ -/
theorem student_distribution_theorem :
  distribute_students 6 6 2 = Nat.choose 6 2 * 5^4 := by
  sorry

end student_distribution_theorem_l3085_308513


namespace return_probability_limit_l3085_308531

/-- Represents a player in the money exchange game --/
inductive Player : Type
| Alan : Player
| Beth : Player
| Charlie : Player
| Dana : Player

/-- The state of the game is represented by a function from Player to ℕ (amount of money) --/
def GameState : Type := Player → ℕ

/-- The initial state of the game where each player has $1 --/
def initialState : GameState :=
  fun p => 1

/-- A single round of the game where players randomly exchange money --/
def playRound (state : GameState) : GameState :=
  sorry

/-- The probability of returning to the initial state after many rounds --/
def returnProbability (numRounds : ℕ) : ℚ :=
  sorry

/-- The main theorem stating that the probability approaches 1/9 as the number of rounds increases --/
theorem return_probability_limit :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |returnProbability n - 1/9| < ε :=
sorry

end return_probability_limit_l3085_308531


namespace geraldo_tea_consumption_l3085_308586

/-- Proves that given 20 gallons of tea poured into 80 containers, if Geraldo drinks 3.5 containers, he consumes 7 pints of tea. -/
theorem geraldo_tea_consumption 
  (total_gallons : ℝ) 
  (num_containers : ℝ) 
  (geraldo_containers : ℝ) 
  (gallons_to_pints : ℝ → ℝ) :
  total_gallons = 20 ∧ 
  num_containers = 80 ∧ 
  geraldo_containers = 3.5 ∧ 
  (∀ x, gallons_to_pints x = 8 * x) →
  geraldo_containers * (gallons_to_pints total_gallons / num_containers) = 7 :=
by sorry

end geraldo_tea_consumption_l3085_308586


namespace solution_of_functional_equation_l3085_308533

def f (x : ℝ) := x^2 + 2*x - 5

theorem solution_of_functional_equation :
  let s1 := (-1 + Real.sqrt 21) / 2
  let s2 := (-1 - Real.sqrt 21) / 2
  let s3 := (-3 + Real.sqrt 17) / 2
  let s4 := (-3 - Real.sqrt 17) / 2
  (∀ x : ℝ, f (f x) = x ↔ x = s1 ∨ x = s2 ∨ x = s3 ∨ x = s4) :=
by sorry

end solution_of_functional_equation_l3085_308533


namespace inverse_of_A_squared_l3085_308509

theorem inverse_of_A_squared (A : Matrix (Fin 2) (Fin 2) ℝ) : 
  A⁻¹ = !![3, 8; -2, -5] → (A^2)⁻¹ = !![(-7), (-16); 4, 9] := by
  sorry

end inverse_of_A_squared_l3085_308509


namespace smallest_base_is_five_l3085_308504

/-- Representation of a number in base b -/
def BaseRepresentation (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Condition: In base b, 12_b squared equals 144_b -/
def SquareCondition (b : Nat) : Prop :=
  (BaseRepresentation [1, 2] b) ^ 2 = BaseRepresentation [1, 4, 4] b

/-- The smallest base b greater than 4 for which 12_b squared equals 144_b is 5 -/
theorem smallest_base_is_five :
  ∃ (b : Nat), b > 4 ∧ SquareCondition b ∧ ∀ (k : Nat), k > 4 ∧ k < b → ¬SquareCondition k :=
by sorry

end smallest_base_is_five_l3085_308504


namespace three_good_sets_l3085_308521

-- Define the "good set" property
def is_good_set (C : Set (ℝ × ℝ)) : Prop :=
  ∀ p₁ ∈ C, ∃ p₂ ∈ C, p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

-- Define the sets
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}
def C₂ : Set (ℝ × ℝ) := {p | p.1^2 - p.2^2 = 9}
def C₃ : Set (ℝ × ℝ) := {p | 2*p.1^2 + p.2^2 = 9}
def C₄ : Set (ℝ × ℝ) := {p | p.1^2 + p.2 = 9}

-- Theorem statement
theorem three_good_sets : 
  (is_good_set C₁ ∧ is_good_set C₃ ∧ is_good_set C₄ ∧ ¬is_good_set C₂) := by
  sorry

end three_good_sets_l3085_308521


namespace largest_special_square_l3085_308597

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def remove_last_two_digits (n : ℕ) : ℕ := n / 100

theorem largest_special_square : 
  (is_perfect_square 1681) ∧ 
  (1681 % 10 ≠ 0) ∧ 
  (is_perfect_square (remove_last_two_digits 1681)) ∧ 
  (∀ m : ℕ, m > 1681 → 
    ¬(is_perfect_square m ∧ 
      m % 10 ≠ 0 ∧ 
      is_perfect_square (remove_last_two_digits m))) :=
by sorry

end largest_special_square_l3085_308597


namespace min_value_on_common_chord_l3085_308584

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x-2)^2 + (y-2)^2 = 4

-- Define the common chord
def common_chord (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

-- Theorem statement
theorem min_value_on_common_chord :
  ∀ a b : ℝ, a > 0 → b > 0 → common_chord a b →
  (∀ x y : ℝ, x > 0 → y > 0 → common_chord x y → 1/a + 9/b ≤ 1/x + 9/y) →
  1/a + 9/b = 8 :=
by sorry

end min_value_on_common_chord_l3085_308584


namespace evaluate_expression_l3085_308525

theorem evaluate_expression (b : ℝ) : 
  let x : ℝ := b + 9
  (x - b + 5) = 14 := by
  sorry

end evaluate_expression_l3085_308525


namespace xiao_ming_banknote_combinations_l3085_308500

def is_valid_combination (x y z : ℕ) : Prop :=
  x + 2*y + 5*z = 18 ∧ x + y + z ≤ 10 ∧ (x > 0 ∧ y > 0) ∨ (x > 0 ∧ z > 0) ∨ (y > 0 ∧ z > 0)

def count_valid_combinations : ℕ := sorry

theorem xiao_ming_banknote_combinations : count_valid_combinations = 9 := by sorry

end xiao_ming_banknote_combinations_l3085_308500


namespace state_returns_sold_l3085_308545

-- Define the prices and quantities
def federal_price : ℕ := 50
def state_price : ℕ := 30
def quarterly_price : ℕ := 80
def federal_quantity : ℕ := 60
def quarterly_quantity : ℕ := 10
def total_revenue : ℕ := 4400

-- Define the function to calculate total revenue
def calculate_revenue (state_quantity : ℕ) : ℕ :=
  federal_price * federal_quantity +
  state_price * state_quantity +
  quarterly_price * quarterly_quantity

-- Theorem statement
theorem state_returns_sold : 
  ∃ (state_quantity : ℕ), calculate_revenue state_quantity = total_revenue ∧ state_quantity = 20 := by
  sorry

end state_returns_sold_l3085_308545


namespace certain_number_proof_l3085_308548

theorem certain_number_proof : 
  ∃ x : ℚ, (x + 720) / 125 = 7392 / 462 ∧ x = 1280 := by
sorry

end certain_number_proof_l3085_308548


namespace arithmetic_computation_l3085_308536

theorem arithmetic_computation : 2 + 3^2 * 4 - 5 + 6 / 2 = 36 := by
  sorry

end arithmetic_computation_l3085_308536


namespace consecutive_even_integers_l3085_308566

theorem consecutive_even_integers (a b c d : ℤ) : 
  (∀ n : ℤ, a = n - 2 ∧ b = n ∧ c = n + 2 ∧ d = n + 4) →
  (a + c = 92) →
  d = 50 := by
sorry

end consecutive_even_integers_l3085_308566


namespace kylie_picked_558_apples_l3085_308534

/-- Represents the number of apples picked in each hour -/
structure ApplesPicked where
  first_hour : ℕ
  second_hour : ℕ
  third_hour : ℕ

/-- Calculates the total number of apples picked -/
def total_apples (ap : ApplesPicked) : ℕ :=
  ap.first_hour + ap.second_hour + ap.third_hour

/-- Represents the first three Fibonacci numbers -/
def first_three_fibonacci : List ℕ := [1, 1, 2]

/-- Represents the first three terms of the arithmetic progression -/
def arithmetic_progression (a₁ d : ℕ) : List ℕ :=
  [a₁, a₁ + d, a₁ + 2*d]

/-- Kylie's apple picking scenario -/
def kylie_apples : ApplesPicked where
  first_hour := 66
  second_hour := (List.sum first_three_fibonacci) * 66
  third_hour := List.sum (arithmetic_progression 66 10)

/-- Theorem stating that Kylie picked 558 apples in total -/
theorem kylie_picked_558_apples :
  total_apples kylie_apples = 558 := by
  sorry


end kylie_picked_558_apples_l3085_308534


namespace min_solution_value_l3085_308550

def system (x y : ℝ) : Prop :=
  3^(-x) * y^4 - 2*y^2 + 3^x ≤ 0 ∧ 27^x + y^4 - 3^x - 1 = 0

def solution_value (x y : ℝ) : ℝ := x^3 + y^3

theorem min_solution_value :
  ∃ (min : ℝ), min = -1 ∧
  (∀ x y : ℝ, system x y → solution_value x y ≥ min) ∧
  (∃ x y : ℝ, system x y ∧ solution_value x y = min) :=
sorry

end min_solution_value_l3085_308550


namespace x_coordinate_of_first_point_l3085_308562

/-- Given a line with equation x = 2y + 3 and two points (m, n) and (m + 2, n + 1) on this line,
    prove that the x-coordinate of the first point, m, is equal to 2n + 3. -/
theorem x_coordinate_of_first_point
  (m n : ℝ)
  (h1 : m = 2 * n + 3)
  (h2 : m + 2 = 2 * (n + 1) + 3) :
  m = 2 * n + 3 := by
  sorry

end x_coordinate_of_first_point_l3085_308562


namespace sum_of_cubes_l3085_308595

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end sum_of_cubes_l3085_308595


namespace cone_volume_from_half_sector_l3085_308585

theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) :
  let base_radius := r / 2
  let cone_height := Real.sqrt (r^2 - base_radius^2)
  (1/3) * π * base_radius^2 * cone_height = 9 * π * Real.sqrt 3 := by
  sorry

end cone_volume_from_half_sector_l3085_308585
