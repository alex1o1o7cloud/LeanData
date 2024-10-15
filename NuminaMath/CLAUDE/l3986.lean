import Mathlib

namespace NUMINAMATH_CALUDE_perfect_cube_in_range_l3986_398659

theorem perfect_cube_in_range (Y J : ℤ) : 
  (150 < Y) → (Y < 300) → (Y = J^5) → (∃ n : ℤ, Y = n^3) → J = 3 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cube_in_range_l3986_398659


namespace NUMINAMATH_CALUDE_factory_production_theorem_l3986_398653

/-- Represents a production line with its output and sample size -/
structure ProductionLine where
  output : ℕ
  sample : ℕ

/-- Represents the factory's production data -/
structure FactoryProduction where
  total_output : ℕ
  line_a : ProductionLine
  line_b : ProductionLine
  line_c : ProductionLine

/-- Checks if three numbers form an arithmetic sequence -/
def isArithmeticSequence (a b c : ℕ) : Prop :=
  b - a = c - b

/-- The main theorem about the factory's production -/
theorem factory_production_theorem (f : FactoryProduction) :
  f.total_output = 16800 ∧
  isArithmeticSequence f.line_a.sample f.line_b.sample f.line_c.sample ∧
  f.line_a.output + f.line_b.output + f.line_c.output = f.total_output →
  f.line_b.output = 5600 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_theorem_l3986_398653


namespace NUMINAMATH_CALUDE_building_cleaning_earnings_l3986_398674

/-- Calculates the total earnings for cleaning a building -/
def total_earnings (floors : ℕ) (rooms_per_floor : ℕ) (hours_per_room : ℕ) (hourly_rate : ℕ) : ℕ :=
  floors * rooms_per_floor * hours_per_room * hourly_rate

/-- Proves that the total earnings for cleaning the specified building is $3600 -/
theorem building_cleaning_earnings :
  total_earnings 4 10 6 15 = 3600 := by
  sorry

#eval total_earnings 4 10 6 15

end NUMINAMATH_CALUDE_building_cleaning_earnings_l3986_398674


namespace NUMINAMATH_CALUDE_fraction_equality_l3986_398695

theorem fraction_equality (x y : ℚ) (hx : x = 4/7) (hy : y = 5/11) :
  (7*x + 11*y) / (63*x*y) = 11/20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3986_398695


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_iff_abs_a_gt_one_l3986_398665

-- Define the equation
def equation (a x y : ℤ) : Prop := x^2 + a*x*y + y^2 = 1

-- Define the property of having infinitely many integer solutions
def has_infinitely_many_solutions (a : ℤ) : Prop :=
  ∀ n : ℕ, ∃ (x y : ℤ), equation a x y ∧ x.natAbs + y.natAbs > n

-- Theorem statement
theorem infinitely_many_solutions_iff_abs_a_gt_one (a : ℤ) :
  has_infinitely_many_solutions a ↔ a.natAbs > 1 := by sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_iff_abs_a_gt_one_l3986_398665


namespace NUMINAMATH_CALUDE_log_expression_equality_complex_expression_equality_l3986_398611

-- Part 1
theorem log_expression_equality : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

-- Part 2
theorem complex_expression_equality : (8 / 125) ^ (-(1 / 3)) - (-3 / 5) ^ 0 + 16 ^ 0.75 = 19 / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_complex_expression_equality_l3986_398611


namespace NUMINAMATH_CALUDE_exam_pass_probability_l3986_398678

/-- The probability of answering a single question correctly -/
def p : ℝ := 0.4

/-- The number of questions in the exam -/
def n : ℕ := 4

/-- The minimum number of correct answers required to pass -/
def k : ℕ := 3

/-- The probability of passing the exam -/
def pass_probability : ℝ := 
  (Nat.choose n k * p^k * (1-p)^(n-k)) + (p^n)

theorem exam_pass_probability : pass_probability = 112/625 := by
  sorry

end NUMINAMATH_CALUDE_exam_pass_probability_l3986_398678


namespace NUMINAMATH_CALUDE_production_improvement_l3986_398668

/-- Represents the production efficiency of a team --/
structure ProductionTeam where
  initial_time : ℕ  -- Initial completion time in hours
  ab_swap_reduction : ℕ  -- Time reduction when swapping A and B
  cd_swap_reduction : ℕ  -- Time reduction when swapping C and D

/-- Calculates the time reduction when swapping both A with B and C with D --/
def time_reduction (team : ProductionTeam) : ℕ :=
  -- Definition to be proved
  108

theorem production_improvement (team : ProductionTeam) 
  (h1 : team.initial_time = 9)
  (h2 : team.ab_swap_reduction = 1)
  (h3 : team.cd_swap_reduction = 1) :
  time_reduction team = 108 := by
  sorry


end NUMINAMATH_CALUDE_production_improvement_l3986_398668


namespace NUMINAMATH_CALUDE_max_absolute_sum_l3986_398643

theorem max_absolute_sum (x y z : ℝ) :
  (|x + 2*y - 3*z| ≤ 6) →
  (|x - 2*y + 3*z| ≤ 6) →
  (|x - 2*y - 3*z| ≤ 6) →
  (|x + 2*y + 3*z| ≤ 6) →
  |x| + |y| + |z| ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_max_absolute_sum_l3986_398643


namespace NUMINAMATH_CALUDE_equation_solution_l3986_398632

theorem equation_solution : 
  {x : ℝ | -x^2 = (2*x + 4)/(x + 2)} = {-2, -1} :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3986_398632


namespace NUMINAMATH_CALUDE_concert_ticket_prices_l3986_398657

theorem concert_ticket_prices (x : ℕ) : 
  (∃ a b : ℕ, a * x = 80 ∧ b * x = 100) → 
  (Finset.filter (fun d => d ∣ 80 ∧ d ∣ 100) (Finset.range 101)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_prices_l3986_398657


namespace NUMINAMATH_CALUDE_system_solutions_correct_l3986_398663

theorem system_solutions_correct :
  -- System 1
  (∃ x y : ℝ, 2*x + y = 3 ∧ 3*x - 5*y = 11 ∧ x = 2 ∧ y = -1) ∧
  -- System 2
  (∃ a b c : ℝ, a + b + c = 0 ∧ a - b + c = -4 ∧ 4*a + 2*b + c = 5 ∧
                a = 1 ∧ b = 2 ∧ c = -3) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_correct_l3986_398663


namespace NUMINAMATH_CALUDE_correct_total_spent_l3986_398690

/-- The total amount Mike spent at the music store after applying the discount -/
def total_spent (trumpet_price songbook_price accessories_price discount_rate : ℝ) : ℝ :=
  let total_before_discount := trumpet_price + songbook_price + accessories_price
  let discount_amount := discount_rate * total_before_discount
  total_before_discount - discount_amount

/-- Theorem stating the correct total amount spent -/
theorem correct_total_spent :
  total_spent 145.16 5.84 18.50 0.12 = 149.16 := by
  sorry

end NUMINAMATH_CALUDE_correct_total_spent_l3986_398690


namespace NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l3986_398603

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_of_digit_factorials (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.map factorial |> List.sum

theorem unique_three_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n = sum_of_digit_factorials n :=
by
  use 145
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l3986_398603


namespace NUMINAMATH_CALUDE_range_of_m_l3986_398637

def p (x : ℝ) : Prop := |x - 4| ≤ 6

def q (x m : ℝ) : Prop := x^2 - m^2 - 2*x + 1 ≤ 0

theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, q x m → p x) →
  (∃ x, p x ∧ ¬q x m) →
  m ∈ Set.Ici 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3986_398637


namespace NUMINAMATH_CALUDE_arrangement_count_l3986_398677

theorem arrangement_count (n m : ℕ) (hn : n = 6) (hm : m = 4) :
  (Nat.choose n m) * (Nat.factorial m) = 360 :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_l3986_398677


namespace NUMINAMATH_CALUDE_problem_solution_l3986_398617

/-- The distance between two points A and B, given the conditions of the problem -/
def distance_AB : ℝ := 1656

/-- Jun Jun's speed -/
def v_jun : ℝ := 14

/-- Ping's speed -/
def v_ping : ℝ := 9

/-- Distance from C to the point where Jun Jun turns back -/
def d_turn : ℝ := 100

/-- Distance from C to the point where Jun Jun catches up with Ping -/
def d_catchup : ℝ := 360

theorem problem_solution :
  ∃ (d_AC d_BC : ℝ),
    d_AC + d_BC = distance_AB ∧
    d_AC / d_BC = v_jun / v_ping ∧
    d_AC - d_catchup = d_turn + d_catchup ∧
    (d_AC - d_catchup) / (d_BC + d_catchup) = v_ping / v_jun :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3986_398617


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l3986_398642

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 61 ways to distribute 5 distinguishable balls into 4 indistinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 61 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l3986_398642


namespace NUMINAMATH_CALUDE_unique_triple_l3986_398679

def is_infinite_repeating_decimal (a b c : ℕ+) : Prop :=
  (a + b / 9 : ℚ)^2 = c + 7/9

def fraction_is_integer (c a : ℕ+) : Prop :=
  ∃ k : ℤ, (c + a : ℚ) / (c - a) = k

theorem unique_triple : 
  ∃! (a b c : ℕ+), 
    b < 10 ∧ 
    is_infinite_repeating_decimal a b c ∧ 
    fraction_is_integer c a ∧
    a = 1 ∧ b = 6 ∧ c = 2 := by sorry

end NUMINAMATH_CALUDE_unique_triple_l3986_398679


namespace NUMINAMATH_CALUDE_limit_equals_negative_six_l3986_398630

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x

-- State the theorem
theorem limit_equals_negative_six :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0,
    |Δx| < δ → |((f (1 - 2*Δx) - f 1) / Δx) + 6| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_equals_negative_six_l3986_398630


namespace NUMINAMATH_CALUDE_series_convergence_implies_k_value_l3986_398600

/-- Given a real number k > 1 such that the infinite series
    Σ(n=1 to ∞) (7n-3)/k^n converges to 5, prove that k = 1.2 + 0.2√46. -/
theorem series_convergence_implies_k_value (k : ℝ) 
  (h1 : k > 1) 
  (h2 : ∑' n, (7 * n - 3) / k^n = 5) : 
  k = 1.2 + 0.2 * Real.sqrt 46 := by
  sorry

end NUMINAMATH_CALUDE_series_convergence_implies_k_value_l3986_398600


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3986_398660

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 2 - Complex.I) → z = -1 - 2*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3986_398660


namespace NUMINAMATH_CALUDE_select_students_problem_l3986_398602

/-- The number of ways to select students for a meeting -/
def select_students (num_boys num_girls : ℕ) (total_selected : ℕ) (min_boys min_girls : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of ways to select students for the given conditions -/
theorem select_students_problem : select_students 5 4 4 2 1 = 100 := by
  sorry

end NUMINAMATH_CALUDE_select_students_problem_l3986_398602


namespace NUMINAMATH_CALUDE_fifteen_solutions_l3986_398639

/-- The system of equations has exactly 15 distinct real solutions -/
theorem fifteen_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ × ℝ)),
    (∀ (u v s t : ℝ), (u, v, s, t) ∈ solutions ↔ 
      (u = s + t + s*u*t ∧
       v = t + u + t*u*v ∧
       s = u + v + u*v*s ∧
       t = v + s + v*s*t)) ∧
    solutions.card = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_solutions_l3986_398639


namespace NUMINAMATH_CALUDE_limit_of_r_as_m_approaches_zero_l3986_398673

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 + 2*x - 6

-- Define L(m) as the smaller root of x^2 + 2x - (m + 6) = 0
noncomputable def L (m : ℝ) : ℝ := -1 - Real.sqrt (m + 7)

-- Define r as a function of m
noncomputable def r (m : ℝ) : ℝ := (L (-m) - L m) / m

-- Theorem statement
theorem limit_of_r_as_m_approaches_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 0 < |m| ∧ |m| < δ → |r m - 1 / Real.sqrt 7| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_r_as_m_approaches_zero_l3986_398673


namespace NUMINAMATH_CALUDE_expected_groups_formula_l3986_398627

/-- A sequence of k zeros and m ones arranged in random order -/
structure BinarySequence where
  k : ℕ
  m : ℕ

/-- The expected number of alternating groups in a BinarySequence -/
noncomputable def expectedGroups (seq : BinarySequence) : ℝ :=
  1 + (2 * seq.k * seq.m : ℝ) / (seq.k + seq.m)

/-- Theorem stating the expected number of alternating groups -/
theorem expected_groups_formula (seq : BinarySequence) :
    expectedGroups seq = 1 + (2 * seq.k * seq.m : ℝ) / (seq.k + seq.m) := by
  sorry

end NUMINAMATH_CALUDE_expected_groups_formula_l3986_398627


namespace NUMINAMATH_CALUDE_problem_solution_l3986_398699

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = 3)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = -4) :
  a / (a + c) + b / (b + a) + c / (c + b) = -2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3986_398699


namespace NUMINAMATH_CALUDE_circle_properties_l3986_398622

/-- A circle in the xy-plane is defined by the equation x^2 + y^2 - 6x = 0. -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

/-- The center of the circle is the point (h, k) in ℝ² -/
def circle_center : ℝ × ℝ := (3, 0)

/-- The radius of the circle is r -/
def circle_radius : ℝ := 3

/-- Theorem stating that the given equation describes a circle with center (3, 0) and radius 3 -/
theorem circle_properties :
  ∀ x y : ℝ, circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3986_398622


namespace NUMINAMATH_CALUDE_smallest_positive_integer_solution_l3986_398620

theorem smallest_positive_integer_solution : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (|5 * (x : ℤ) - 8| = 47) ∧ 
  (∀ (y : ℕ), y > 0 ∧ |5 * (y : ℤ) - 8| = 47 → x ≤ y) ∧
  (x = 11) := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_solution_l3986_398620


namespace NUMINAMATH_CALUDE_tournament_divisibility_l3986_398684

theorem tournament_divisibility (n : ℕ) : 
  let tournament_year := fun i => 1978 + i
  (tournament_year 43 = 2021) →
  (∃! k, k = 3 ∧ 
    (∀ i ∈ Finset.range k, 
      ∃ m > 43, tournament_year m % m = 0 ∧
      ∀ j ∈ Finset.Icc 44 (m - 1), tournament_year j % j ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_tournament_divisibility_l3986_398684


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3986_398689

theorem pure_imaginary_condition (x y : ℝ) : 
  (∀ z : ℂ, z.re = x ∧ z.im = y → (z.re = 0 ↔ z.im ≠ 0)) ↔
  (x = 0 → ∃ y : ℝ, y ≠ 0) ∧ (∃ x y : ℝ, x = 0 ∧ y = 0) :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3986_398689


namespace NUMINAMATH_CALUDE_traced_path_is_asterisk_l3986_398662

/-- A regular n-gon in the plane -/
structure RegularNGon where
  n : ℕ
  center : ℝ × ℝ
  vertices : Fin n → ℝ × ℝ

/-- Triangle formed by two adjacent vertices and the center of a regular n-gon -/
structure TriangleABO (ngon : RegularNGon) where
  A : Fin ngon.n
  B : Fin ngon.n
  hAdjacent : (A.val + 1) % ngon.n = B.val

/-- The path traced by point O when triangle ABO glides around the n-gon -/
def tracedPath (ngon : RegularNGon) : Set (ℝ × ℝ) := sorry

/-- An asterisk consisting of n segments emanating from the center -/
def asterisk (center : ℝ × ℝ) (n : ℕ) (length : ℝ) : Set (ℝ × ℝ) := sorry

/-- Main theorem: The path traced by O forms an asterisk -/
theorem traced_path_is_asterisk (ngon : RegularNGon) :
  ∃ (length : ℝ), tracedPath ngon = asterisk ngon.center ngon.n length := by sorry

end NUMINAMATH_CALUDE_traced_path_is_asterisk_l3986_398662


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l3986_398605

-- Define an isosceles triangle with a vertex angle of 40°
structure IsoscelesTriangle where
  vertex_angle : ℝ
  is_isosceles : Bool
  vertex_angle_value : vertex_angle = 40

-- Define the property we want to prove
def base_angle_is_70 (triangle : IsoscelesTriangle) : Prop :=
  ∃ (base_angle : ℝ), base_angle = 70 ∧ 
    triangle.vertex_angle + 2 * base_angle = 180

-- State the theorem
theorem isosceles_triangle_base_angle 
  (triangle : IsoscelesTriangle) : 
  base_angle_is_70 triangle :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l3986_398605


namespace NUMINAMATH_CALUDE_cyclic_fraction_inequality_l3986_398646

theorem cyclic_fraction_inequality (a b x y z : ℝ) (ha : a > 0) (hb : b > 0) :
  x / (a * y + b * z) + y / (a * z + b * x) + z / (a * x + b * y) ≥ 3 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_fraction_inequality_l3986_398646


namespace NUMINAMATH_CALUDE_exists_non_regular_triangle_with_similar_median_triangle_l3986_398636

/-- Represents a triangle with sides a, b, c and medians s_a, s_b, s_c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  s_a : ℝ
  s_b : ℝ
  s_c : ℝ
  h_order : a ≤ b ∧ b ≤ c
  h_median_a : 4 * s_a^2 = 2 * b^2 + 2 * c^2 - a^2
  h_median_b : 4 * s_b^2 = 2 * c^2 + 2 * a^2 - b^2
  h_median_c : 4 * s_c^2 = 2 * a^2 + 2 * b^2 - c^2

/-- Two triangles are similar if the ratios of their corresponding sides are equal -/
def similar (t1 t2 : Triangle) : Prop :=
  (t1.a / t2.a)^2 = (t1.b / t2.b)^2 ∧ (t1.b / t2.b)^2 = (t1.c / t2.c)^2

/-- A triangle is regular if all its sides are equal -/
def regular (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

theorem exists_non_regular_triangle_with_similar_median_triangle :
  ∃ t : Triangle, ¬regular t ∧ similar t ⟨t.s_a, t.s_b, t.s_c, 0, 0, 0, sorry, sorry, sorry, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_exists_non_regular_triangle_with_similar_median_triangle_l3986_398636


namespace NUMINAMATH_CALUDE_income_tax_problem_l3986_398676

theorem income_tax_problem (q : ℝ) :
  let tax_rate_low := q / 100
  let tax_rate_high := (q + 3) / 100
  let total_tax_rate := (q + 0.5) / 100
  let income := 36000
  let tax_low := tax_rate_low * 30000
  let tax_high := tax_rate_high * (income - 30000)
  tax_low + tax_high = total_tax_rate * income := by sorry

end NUMINAMATH_CALUDE_income_tax_problem_l3986_398676


namespace NUMINAMATH_CALUDE_triangle_segment_equality_l3986_398691

theorem triangle_segment_equality (AB AC : ℝ) (n : ℕ) :
  AB = 33 →
  AC = 21 →
  (∃ (D E : ℝ), 0 ≤ D ∧ D ≤ AB ∧ 0 ≤ E ∧ E ≤ AC ∧ D = n ∧ AB - D = n ∧ E = n ∧ AC - E = n) →
  (∃ (BC : ℕ), BC = 30) :=
by sorry

end NUMINAMATH_CALUDE_triangle_segment_equality_l3986_398691


namespace NUMINAMATH_CALUDE_decimal_fraction_equality_l3986_398669

theorem decimal_fraction_equality : (0.5^4) / (0.05^3) = 500 := by
  sorry

end NUMINAMATH_CALUDE_decimal_fraction_equality_l3986_398669


namespace NUMINAMATH_CALUDE_alpha_value_l3986_398607

/-- A structure representing the relationship between α, β, and γ -/
structure Relationship where
  α : ℝ
  β : ℝ
  γ : ℝ
  k : ℝ
  h1 : α = k * γ / β

/-- The theorem stating the relationship between α, β, and γ -/
theorem alpha_value (r : Relationship) (h2 : r.α = 4) (h3 : r.β = 27) (h4 : r.γ = 3) :
  ∃ (r' : Relationship), r'.β = -81 ∧ r'.γ = 9 ∧ r'.α = -4 :=
sorry

end NUMINAMATH_CALUDE_alpha_value_l3986_398607


namespace NUMINAMATH_CALUDE_jason_gave_nine_cards_l3986_398609

/-- The number of Pokemon cards Jason started with -/
def initial_cards : ℕ := 13

/-- The number of Pokemon cards Jason has left -/
def remaining_cards : ℕ := 4

/-- The number of Pokemon cards Jason gave to his friends -/
def cards_given : ℕ := initial_cards - remaining_cards

theorem jason_gave_nine_cards : cards_given = 9 := by
  sorry

end NUMINAMATH_CALUDE_jason_gave_nine_cards_l3986_398609


namespace NUMINAMATH_CALUDE_triple_composition_even_l3986_398645

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem triple_composition_even
  (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (fun x ↦ g (g (g x))) :=
by
  sorry

end NUMINAMATH_CALUDE_triple_composition_even_l3986_398645


namespace NUMINAMATH_CALUDE_symmetry_probability_l3986_398615

/-- Represents a point on the grid --/
structure GridPoint where
  x : Fin 11
  y : Fin 11

/-- The center point of the grid --/
def centerPoint : GridPoint :=
  ⟨5, 5⟩

/-- The set of all points on the grid --/
def allPoints : Finset GridPoint :=
  sorry

/-- The set of all points except the center point --/
def nonCenterPoints : Finset GridPoint :=
  sorry

/-- Predicate to check if a line through two points is a line of symmetry --/
def isSymmetryLine (p q : GridPoint) : Prop :=
  sorry

/-- The set of points that form symmetry lines with the center point --/
def symmetryPoints : Finset GridPoint :=
  sorry

theorem symmetry_probability :
    (symmetryPoints.card : ℚ) / (nonCenterPoints.card : ℚ) = 1 / 3 :=
  sorry

end NUMINAMATH_CALUDE_symmetry_probability_l3986_398615


namespace NUMINAMATH_CALUDE_bronze_medals_count_l3986_398629

theorem bronze_medals_count (total : ℕ) (gold : ℕ) (silver : ℕ) (bronze : ℕ)
  (h_total : total = 67)
  (h_gold : gold = 19)
  (h_silver : silver = 32)
  (h_sum : total = gold + silver + bronze) :
  bronze = 16 :=
by sorry

end NUMINAMATH_CALUDE_bronze_medals_count_l3986_398629


namespace NUMINAMATH_CALUDE_area_between_tangent_circles_l3986_398656

/-- The area of the region between two tangent circles -/
theorem area_between_tangent_circles 
  (r₁ : ℝ) -- radius of the inner circle
  (d : ℝ)  -- distance between the centers of the circles
  (h₁ : r₁ = 5) -- given radius of inner circle
  (h₂ : d = 3)  -- given distance between centers
  : (π * ((r₁ + d)^2 - r₁^2) : ℝ) = 39 * π :=
sorry

end NUMINAMATH_CALUDE_area_between_tangent_circles_l3986_398656


namespace NUMINAMATH_CALUDE_best_fitting_model_l3986_398631

structure Model where
  id : Nat
  r_squared : Real

def models : List Model := [
  { id := 1, r_squared := 0.98 },
  { id := 2, r_squared := 0.80 },
  { id := 3, r_squared := 0.54 },
  { id := 4, r_squared := 0.35 }
]

theorem best_fitting_model :
  ∃ m ∈ models, ∀ m' ∈ models, m.r_squared ≥ m'.r_squared ∧ m.id = 1 :=
by sorry

end NUMINAMATH_CALUDE_best_fitting_model_l3986_398631


namespace NUMINAMATH_CALUDE_sum_seventeen_terms_l3986_398658

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference
  sum_first_fifth : a + (a + 4 * d) = 5 / 3
  product_third_fourth : (a + 2 * d) * (a + 3 * d) = 65 / 72

/-- Sum of the first n terms of an arithmetic progression -/
def sum_n_terms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  n / 2 * (2 * ap.a + (n - 1) * ap.d)

/-- The main theorem to prove -/
theorem sum_seventeen_terms (ap : ArithmeticProgression) :
  sum_n_terms ap 17 = 119 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_seventeen_terms_l3986_398658


namespace NUMINAMATH_CALUDE_system_solution_l3986_398616

theorem system_solution (x y z b : ℝ) : 
  (3 * x * y * z - x^3 - y^3 - z^3 = b^3) ∧ 
  (x + y + z = 2 * b) ∧ 
  (x^2 + y^2 - z^2 = b^2) →
  ((b = 0 ∧ ((x = 0 ∧ z = -y) ∨ (y = 0 ∧ z = -x))) ∨
   (b ≠ 0 ∧ z = 0 ∧ 
    ((x = (1 + Real.sqrt (-1/2)) * b ∧ y = (1 - Real.sqrt (-1/2)) * b) ∨
     (x = (1 - Real.sqrt (-1/2)) * b ∧ y = (1 + Real.sqrt (-1/2)) * b)))) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3986_398616


namespace NUMINAMATH_CALUDE_sum_of_decimals_l3986_398697

theorem sum_of_decimals :
  5.46 + 2.793 + 3.1 = 11.353 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l3986_398697


namespace NUMINAMATH_CALUDE_mary_earnings_l3986_398692

/-- Mary's earnings from cleaning homes -/
theorem mary_earnings (total_earnings : ℕ) (homes_cleaned : ℕ) 
  (h1 : total_earnings = 276)
  (h2 : homes_cleaned = 6) :
  total_earnings / homes_cleaned = 46 := by
  sorry

end NUMINAMATH_CALUDE_mary_earnings_l3986_398692


namespace NUMINAMATH_CALUDE_quadratic_roots_distance_l3986_398652

theorem quadratic_roots_distance (m : ℝ) : 
  (∃ α β : ℂ, (α^2 - 2 * Real.sqrt 2 * α + m = 0) ∧ 
              (β^2 - 2 * Real.sqrt 2 * β + m = 0) ∧ 
              (Complex.abs (α - β) = 3)) → 
  m = 17/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_distance_l3986_398652


namespace NUMINAMATH_CALUDE_factorization_of_a_squared_minus_4ab_l3986_398638

theorem factorization_of_a_squared_minus_4ab (a b : ℝ) :
  a^2 - 4*a*b = a*(a - 4*b) := by sorry

end NUMINAMATH_CALUDE_factorization_of_a_squared_minus_4ab_l3986_398638


namespace NUMINAMATH_CALUDE_tangent_circles_radius_l3986_398644

theorem tangent_circles_radius (r : ℝ) (R : ℝ) : 
  r > 0 → 
  (∃ (A B C : ℝ × ℝ) (O : ℝ × ℝ),
    -- Three circles with centers A, B, C and radius r are externally tangent to each other
    dist A B = 2 * r ∧ 
    dist B C = 2 * r ∧ 
    dist C A = 2 * r ∧
    -- These three circles are internally tangent to a larger circle with center O and radius R
    dist O A = R - r ∧
    dist O B = R - r ∧
    dist O C = R - r) →
  -- Then the radius of the large circle is 2(√3 + 1) when r = 2
  r = 2 → R = 2 * (Real.sqrt 3 + 1) := by
sorry


end NUMINAMATH_CALUDE_tangent_circles_radius_l3986_398644


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3986_398683

theorem sphere_surface_area (r : ℝ) (h : r = 2) : 4 * Real.pi * r^2 = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3986_398683


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l3986_398670

theorem polynomial_division_quotient :
  ∀ x : ℝ, x ≠ 1 →
  (x^6 + 6) = (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) + 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l3986_398670


namespace NUMINAMATH_CALUDE_candy_cost_l3986_398661

/-- 
Given Chris's babysitting earnings and expenses, prove the cost of the candy assortment.
-/
theorem candy_cost 
  (video_game_cost : ℕ) 
  (hourly_rate : ℕ) 
  (hours_worked : ℕ) 
  (money_left : ℕ) 
  (h1 : video_game_cost = 60)
  (h2 : hourly_rate = 8)
  (h3 : hours_worked = 9)
  (h4 : money_left = 7) :
  video_game_cost + money_left + 5 = hourly_rate * hours_worked :=
by sorry

end NUMINAMATH_CALUDE_candy_cost_l3986_398661


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l3986_398610

theorem max_value_of_sum_products (a b c : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) 
  (h4 : a^2 + b^2 + c^2 = 3) : 
  ∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → x^2 + y^2 + z^2 = 3 → 
  a * b + b * c + c * a ≥ x * y + y * z + z * x ∧
  a * b + b * c + c * a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l3986_398610


namespace NUMINAMATH_CALUDE_pigs_joined_l3986_398634

theorem pigs_joined (initial_pigs final_pigs : ℕ) (h : initial_pigs ≤ final_pigs) :
  final_pigs - initial_pigs = final_pigs - initial_pigs :=
by sorry

end NUMINAMATH_CALUDE_pigs_joined_l3986_398634


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_difference_l3986_398664

theorem repeating_decimal_sum_difference (x y z : ℚ) :
  x = 5/9 ∧ y = 1/9 ∧ z = 3/9 → x + y - z = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_difference_l3986_398664


namespace NUMINAMATH_CALUDE_meeting_problem_solution_l3986_398621

/-- Represents the problem of two people moving towards each other --/
structure MeetingProblem where
  total_distance : ℝ
  meeting_time : ℝ
  distance_difference : ℝ
  time_to_b_after_meeting : ℝ

/-- The solution to the meeting problem --/
structure MeetingSolution where
  speed_xiaogang : ℝ
  speed_xiaoqiang : ℝ
  time_to_a_after_meeting : ℝ

/-- Theorem stating the solution to the meeting problem --/
theorem meeting_problem_solution (p : MeetingProblem) 
  (h1 : p.meeting_time = 2)
  (h2 : p.distance_difference = 24)
  (h3 : p.time_to_b_after_meeting = 0.5) :
  ∃ (s : MeetingSolution),
    s.speed_xiaogang = 16 ∧
    s.speed_xiaoqiang = 4 ∧
    s.time_to_a_after_meeting = 8 := by
  sorry

end NUMINAMATH_CALUDE_meeting_problem_solution_l3986_398621


namespace NUMINAMATH_CALUDE_sin_sum_alpha_beta_l3986_398672

theorem sin_sum_alpha_beta (α β : Real) 
  (h1 : 13 * Real.sin α + 5 * Real.cos β = 9)
  (h2 : 13 * Real.cos α + 5 * Real.sin β = 15) : 
  Real.sin (α + β) = 56 / 65 := by
sorry

end NUMINAMATH_CALUDE_sin_sum_alpha_beta_l3986_398672


namespace NUMINAMATH_CALUDE_unique_solution_value_l3986_398671

/-- For a quadratic equation ax^2 + bx + c = 0 to have exactly one solution,
    its discriminant (b^2 - 4ac) must be zero -/
def has_unique_solution (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

/-- The quadratic equation 3x^2 - 7x + k = 0 -/
def quadratic_equation (k : ℝ) (x : ℝ) : Prop :=
  3*x^2 - 7*x + k = 0

theorem unique_solution_value (k : ℝ) :
  (∃! x, quadratic_equation k x) ↔ k = 49/12 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_value_l3986_398671


namespace NUMINAMATH_CALUDE_marys_income_percentage_l3986_398618

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = 0.4 * juan) 
  (h2 : mary = 1.6 * tim) : 
  mary = 0.64 * juan := by
sorry

end NUMINAMATH_CALUDE_marys_income_percentage_l3986_398618


namespace NUMINAMATH_CALUDE_lucas_payment_l3986_398649

/-- Calculates the payment for window cleaning based on given conditions --/
def calculate_payment (windows_per_floor : ℕ) (num_floors : ℕ) (payment_per_window : ℕ) 
  (penalty_per_period : ℕ) (days_per_period : ℕ) (total_days : ℕ) : ℕ :=
  let total_windows := windows_per_floor * num_floors
  let total_earned := total_windows * payment_per_window
  let num_periods := total_days / days_per_period
  let total_penalty := num_periods * penalty_per_period
  total_earned - total_penalty

theorem lucas_payment :
  calculate_payment 5 4 3 2 4 12 = 54 :=
sorry

end NUMINAMATH_CALUDE_lucas_payment_l3986_398649


namespace NUMINAMATH_CALUDE_problem_statement_l3986_398608

theorem problem_statement (a b : ℝ) 
  (h1 : a^2 * b^3 / 5 = 1000) 
  (h2 : a * b = 2) : 
  a^3 * b^2 / 3 = 2 / 705 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3986_398608


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l3986_398640

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x > Real.sin x) ↔ (∀ x : ℝ, x ≤ Real.sin x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l3986_398640


namespace NUMINAMATH_CALUDE_xy_range_l3986_398628

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y + x*y = 30) :
  12 < x*y ∧ x*y < 870 := by
  sorry

end NUMINAMATH_CALUDE_xy_range_l3986_398628


namespace NUMINAMATH_CALUDE_cosine_inequality_solution_l3986_398655

theorem cosine_inequality_solution (y : Real) : 
  (y ∈ Set.Icc 0 Real.pi) → 
  ((∀ x ∈ Set.Icc 0 Real.pi, Real.cos (x + y) ≥ Real.cos x * Real.cos y) ↔ 
   (y = 0 ∨ y = Real.pi)) := by
  sorry

end NUMINAMATH_CALUDE_cosine_inequality_solution_l3986_398655


namespace NUMINAMATH_CALUDE_sea_glass_collection_l3986_398675

/-- Sea glass collection problem -/
theorem sea_glass_collection (blanche_green blanche_red rose_blue dorothy_total : ℕ) 
  (h1 : blanche_green = 12)
  (h2 : blanche_red = 3)
  (h3 : rose_blue = 11)
  (h4 : dorothy_total = 57) :
  ∃ (rose_red : ℕ),
    dorothy_total = 2 * (blanche_red + rose_red) + 3 * rose_blue ∧ 
    rose_red = 9 := by
  sorry

end NUMINAMATH_CALUDE_sea_glass_collection_l3986_398675


namespace NUMINAMATH_CALUDE_sum_of_squares_l3986_398682

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3986_398682


namespace NUMINAMATH_CALUDE_inverse_not_always_true_l3986_398604

theorem inverse_not_always_true :
  ¬(∀ (a b m : ℝ), (a < b → a * m^2 < b * m^2)) :=
sorry

end NUMINAMATH_CALUDE_inverse_not_always_true_l3986_398604


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3986_398687

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 10 = 0) : 
  x^3 - 3*x^2 - 10*x + 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3986_398687


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l3986_398614

theorem min_value_of_sum_of_roots (x : ℝ) : 
  Real.sqrt (x^2 - 4*x + 13) + Real.sqrt (x^2 - 10*x + 26) ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l3986_398614


namespace NUMINAMATH_CALUDE_odd_cube_plus_one_not_square_l3986_398681

theorem odd_cube_plus_one_not_square (n : ℤ) (h : Odd n) :
  ¬ ∃ x : ℤ, n^3 + 1 = x^2 := by
sorry

end NUMINAMATH_CALUDE_odd_cube_plus_one_not_square_l3986_398681


namespace NUMINAMATH_CALUDE_zoom_video_glitch_duration_l3986_398696

theorem zoom_video_glitch_duration :
  let mac_download_time : ℕ := 10
  let windows_download_time : ℕ := 3 * mac_download_time
  let total_download_time : ℕ := mac_download_time + windows_download_time
  let total_time : ℕ := 82
  let call_time : ℕ := total_time - total_download_time
  let audio_glitch_time : ℕ := 2 * 4
  let video_glitch_time : ℕ := call_time - (audio_glitch_time + 2 * (audio_glitch_time + video_glitch_time))
  video_glitch_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_zoom_video_glitch_duration_l3986_398696


namespace NUMINAMATH_CALUDE_smallest_page_number_l3986_398601

theorem smallest_page_number : ∃ n : ℕ, n > 0 ∧ 4 ∣ n ∧ 13 ∣ n ∧ ∀ m : ℕ, (m > 0 ∧ 4 ∣ m ∧ 13 ∣ m) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_page_number_l3986_398601


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3986_398624

theorem perfect_square_condition (m : ℤ) : 
  (∃ n : ℤ, m^2 + 6*m + 28 = n^2) ↔ (m = 6 ∨ m = -12) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3986_398624


namespace NUMINAMATH_CALUDE_constant_remainder_iff_b_eq_neg_five_halves_l3986_398606

/-- The dividend polynomial -/
def dividend (b x : ℝ) : ℝ := 12 * x^4 - 5 * x^3 + b * x^2 - 4 * x + 8

/-- The divisor polynomial -/
def divisor (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1

/-- Theorem stating that the remainder is constant iff b = -5/2 -/
theorem constant_remainder_iff_b_eq_neg_five_halves :
  ∃ (q : ℝ → ℝ) (r : ℝ), ∀ x, dividend (-5/2) x = q x * divisor x + r ↔ 
  ∀ b, (∃ (q : ℝ → ℝ) (r : ℝ), ∀ x, dividend b x = q x * divisor x + r) → b = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_constant_remainder_iff_b_eq_neg_five_halves_l3986_398606


namespace NUMINAMATH_CALUDE_average_chocolate_pieces_per_cookie_l3986_398667

theorem average_chocolate_pieces_per_cookie 
  (total_cookies : ℕ) 
  (chocolate_chips : ℕ) 
  (mms_ratio : ℚ) :
  total_cookies = 48 →
  chocolate_chips = 108 →
  mms_ratio = 1/3 →
  (chocolate_chips + (mms_ratio * chocolate_chips)) / total_cookies = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_chocolate_pieces_per_cookie_l3986_398667


namespace NUMINAMATH_CALUDE_triangle_inequality_and_side_length_relations_l3986_398666

/-- Given a triangle with side lengths a, b, c, prove the existence of a triangle
    with side lengths √a, √b, √c and the inequality involving these lengths. -/
theorem triangle_inequality_and_side_length_relations
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hab : a + b > c)
  (hbc : b + c > a)
  (hca : c + a > b) :
  (∃ (x y z : ℝ), x = Real.sqrt a ∧ y = Real.sqrt b ∧ z = Real.sqrt c ∧
    x + y > z ∧ y + z > x ∧ z + x > y) ∧
  Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≤ a + b + c ∧
  a + b + c < 2 * (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_and_side_length_relations_l3986_398666


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_one_neg_two_l3986_398625

/-- Given that the terminal side of angle α passes through point P(1,-2),
    prove that sin α = -2√5/5 -/
theorem sin_alpha_for_point_one_neg_two (α : Real) :
  (∃ (P : ℝ × ℝ), P = (1, -2) ∧ P.1 = Real.cos α ∧ P.2 = Real.sin α) →
  Real.sin α = -2 * Real.sqrt 5 / 5 := by
sorry


end NUMINAMATH_CALUDE_sin_alpha_for_point_one_neg_two_l3986_398625


namespace NUMINAMATH_CALUDE_complex_magnitude_thirteen_l3986_398654

theorem complex_magnitude_thirteen (x : ℝ) : 
  x > 0 → (Complex.abs (3 + x * Complex.I) = 13 ↔ x = 8 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_complex_magnitude_thirteen_l3986_398654


namespace NUMINAMATH_CALUDE_mikes_second_job_hours_l3986_398647

/-- Given Mike's total wages, wages from his first job, and hourly rate at his second job,
    calculate the number of hours he worked at his second job. -/
theorem mikes_second_job_hours
  (total_wages : ℕ)
  (first_job_wages : ℕ)
  (second_job_hourly_rate : ℕ)
  (h1 : total_wages = 160)
  (h2 : first_job_wages = 52)
  (h3 : second_job_hourly_rate = 9) :
  (total_wages - first_job_wages) / second_job_hourly_rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_mikes_second_job_hours_l3986_398647


namespace NUMINAMATH_CALUDE_two_numbers_product_l3986_398612

theorem two_numbers_product (x y : ℕ) : 
  x ∈ Finset.range 33 ∧ 
  y ∈ Finset.range 33 ∧ 
  x ≠ y ∧
  (Finset.sum (Finset.range 33) id - x - y = x * y) →
  x * y = 484 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_product_l3986_398612


namespace NUMINAMATH_CALUDE_ellipse_focus_distance_l3986_398680

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_a_ge_b : a ≥ b

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The theorem stating the property of the ellipse -/
theorem ellipse_focus_distance (e : Ellipse) (p : PointOnEllipse e) 
  (h_focus1 : ℝ) (h_on_ellipse : e.a = 5 ∧ e.b = 4) :
  h_focus1 = 8 → ∃ h_focus2 : ℝ, h_focus2 = 2 ∧ h_focus1 + h_focus2 = 2 * e.a := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focus_distance_l3986_398680


namespace NUMINAMATH_CALUDE_carla_bob_payment_difference_l3986_398686

/-- Represents the pizza and its properties -/
structure Pizza :=
  (total_slices : ℕ)
  (vegetarian_slices : ℕ)
  (plain_cost : ℚ)
  (vegetarian_extra_cost : ℚ)

/-- Calculates the cost per slice of the pizza -/
def cost_per_slice (p : Pizza) : ℚ :=
  (p.plain_cost + p.vegetarian_extra_cost) / p.total_slices

/-- Calculates the cost for a given number of slices -/
def cost_for_slices (p : Pizza) (slices : ℕ) : ℚ :=
  (cost_per_slice p) * slices

/-- The main theorem to prove -/
theorem carla_bob_payment_difference
  (p : Pizza)
  (carla_slices bob_slices : ℕ)
  : p.total_slices = 12 →
    p.vegetarian_slices = 6 →
    p.plain_cost = 10 →
    p.vegetarian_extra_cost = 3 →
    carla_slices = 8 →
    bob_slices = 3 →
    (cost_for_slices p carla_slices) - (cost_for_slices p bob_slices) = 5.42 := by
  sorry

end NUMINAMATH_CALUDE_carla_bob_payment_difference_l3986_398686


namespace NUMINAMATH_CALUDE_four_integer_average_l3986_398685

theorem four_integer_average (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different integers
  d = 90 ∧                 -- Largest integer is 90
  a ≥ 13 →                 -- Smallest integer is at least 13
  (a + b + c + d) / 4 = 33 -- Average is 33
  := by sorry

end NUMINAMATH_CALUDE_four_integer_average_l3986_398685


namespace NUMINAMATH_CALUDE_parking_solution_l3986_398623

def parking_problem (first_level second_level third_level fourth_level : ℕ) : Prop :=
  first_level = 4 ∧
  second_level = first_level + 7 ∧
  third_level > second_level ∧
  fourth_level = 14 ∧
  first_level + second_level + third_level + fourth_level = 46

theorem parking_solution :
  ∀ first_level second_level third_level fourth_level : ℕ,
  parking_problem first_level second_level third_level fourth_level →
  third_level - second_level = 6 :=
by
  sorry

#check parking_solution

end NUMINAMATH_CALUDE_parking_solution_l3986_398623


namespace NUMINAMATH_CALUDE_intersection_range_l3986_398626

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 15 = 0

-- Define the line
def line (k x : ℝ) (y : ℝ) : Prop := y = k*x - 2

-- Define the condition for intersection
def intersection_condition (k : ℝ) : Prop :=
  ∃ (x y : ℝ), line k x y ∧
    ∃ (x' y' : ℝ), circle_C x' y' ∧
      (x - x')^2 + (y - y')^2 ≤ 4

-- Theorem statement
theorem intersection_range :
  ∀ k : ℝ, intersection_condition k ↔ 0 ≤ k ∧ k ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l3986_398626


namespace NUMINAMATH_CALUDE_previous_day_visitor_count_l3986_398693

/-- The number of visitors to Buckingham Palace on the current day -/
def current_day_visitors : ℕ := 661

/-- The difference in visitors between the current day and the previous day -/
def visitor_difference : ℕ := 61

/-- The number of visitors on the previous day -/
def previous_day_visitors : ℕ := current_day_visitors - visitor_difference

theorem previous_day_visitor_count : previous_day_visitors = 600 := by
  sorry

end NUMINAMATH_CALUDE_previous_day_visitor_count_l3986_398693


namespace NUMINAMATH_CALUDE_max_roads_removal_l3986_398651

/-- A graph representing the Empire of Westeros --/
structure WesterosGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  is_connected : Bool
  vertex_count : vertices.card = 1000
  edge_count : edges.card = 2017
  initial_connectivity : is_connected = true

/-- The result of removing roads from the graph --/
structure KingdomFormation where
  removed_roads : Nat
  kingdom_count : Nat

/-- The maximum number of roads that can be removed to form exactly 7 kingdoms --/
def max_removable_roads (g : WesterosGraph) : Nat :=
  993

/-- Theorem stating the maximum number of removable roads --/
theorem max_roads_removal (g : WesterosGraph) :
  ∃ (kf : KingdomFormation),
    kf.removed_roads = max_removable_roads g ∧
    kf.kingdom_count = 7 ∧
    ∀ (kf' : KingdomFormation),
      kf'.kingdom_count = 7 → kf'.removed_roads ≤ kf.removed_roads :=
sorry


end NUMINAMATH_CALUDE_max_roads_removal_l3986_398651


namespace NUMINAMATH_CALUDE_marbles_given_correct_l3986_398698

/-- The number of marbles given to the brother -/
def marbles_given : ℕ := 2

/-- The initial number of marbles you have -/
def initial_marbles : ℕ := 16

/-- The total number of marbles among all three people -/
def total_marbles : ℕ := 63

theorem marbles_given_correct :
  -- After giving marbles, you have double your brother's marbles
  2 * ((initial_marbles - marbles_given) / 2) = initial_marbles - marbles_given ∧
  -- Your friend has triple your marbles after giving
  3 * (initial_marbles - marbles_given) = 
    total_marbles - (initial_marbles - marbles_given) - ((initial_marbles - marbles_given) / 2) :=
by sorry

end NUMINAMATH_CALUDE_marbles_given_correct_l3986_398698


namespace NUMINAMATH_CALUDE_same_color_probability_l3986_398688

/-- The number of green balls in the bag -/
def green_balls : ℕ := 8

/-- The number of red balls in the bag -/
def red_balls : ℕ := 7

/-- The total number of balls in the bag -/
def total_balls : ℕ := green_balls + red_balls

/-- The probability of drawing two balls of the same color with replacement -/
theorem same_color_probability : 
  (green_balls / total_balls) ^ 2 + (red_balls / total_balls) ^ 2 = 113 / 225 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l3986_398688


namespace NUMINAMATH_CALUDE_product_of_fractions_l3986_398635

theorem product_of_fractions :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3986_398635


namespace NUMINAMATH_CALUDE_amy_local_calls_l3986_398650

/-- Proves that Amy made 15 local calls given the conditions of the problem -/
theorem amy_local_calls :
  ∀ (L I : ℕ),
  (L : ℚ) / I = 5 / 2 →
  L / (I + 3) = 5 / 3 →
  L = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_amy_local_calls_l3986_398650


namespace NUMINAMATH_CALUDE_overall_pass_rate_l3986_398633

theorem overall_pass_rate (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  (1 - a) * (1 - b) = ab - a - b + 1 := by
  sorry

end NUMINAMATH_CALUDE_overall_pass_rate_l3986_398633


namespace NUMINAMATH_CALUDE_circumradius_inradius_inequality_l3986_398648

/-- A triangle with its circumradius and inradius -/
structure Triangle where
  R : ℝ  -- Circumradius
  r : ℝ  -- Inradius

/-- Predicate to check if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop :=
  t.R = 2 * t.r

theorem circumradius_inradius_inequality (t : Triangle) :
  t.R ≥ 2 * t.r ∧ (t.R = 2 * t.r ↔ is_equilateral t) :=
sorry

end NUMINAMATH_CALUDE_circumradius_inradius_inequality_l3986_398648


namespace NUMINAMATH_CALUDE_proportion_estimate_correct_l3986_398694

/-- Proportion of households with 3+ housing sets -/
def proportion_with_3plus_housing (total_households : ℕ) 
  (ordinary_households : ℕ) (high_income_households : ℕ)
  (sampled_ordinary : ℕ) (sampled_high_income : ℕ)
  (sampled_ordinary_with_3plus : ℕ) (sampled_high_income_with_3plus : ℕ) : ℚ :=
  let estimated_ordinary_with_3plus := (sampled_ordinary_with_3plus : ℚ) * ordinary_households / sampled_ordinary
  let estimated_high_income_with_3plus := (sampled_high_income_with_3plus : ℚ) * high_income_households / sampled_high_income
  (estimated_ordinary_with_3plus + estimated_high_income_with_3plus) / total_households

theorem proportion_estimate_correct : 
  proportion_with_3plus_housing 100000 99000 1000 990 100 40 80 = 48 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_proportion_estimate_correct_l3986_398694


namespace NUMINAMATH_CALUDE_rice_mixture_price_l3986_398613

theorem rice_mixture_price (price1 price2 proportion1 proportion2 : ℚ) 
  (h1 : price1 = 31/10)
  (h2 : price2 = 36/10)
  (h3 : proportion1 = 7)
  (h4 : proportion2 = 3)
  : (price1 * proportion1 + price2 * proportion2) / (proportion1 + proportion2) = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_rice_mixture_price_l3986_398613


namespace NUMINAMATH_CALUDE_cubic_equation_product_l3986_398619

theorem cubic_equation_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2010) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2009)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2010) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2009)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2010) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2009) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_product_l3986_398619


namespace NUMINAMATH_CALUDE_incorrect_operation_correction_l3986_398641

theorem incorrect_operation_correction (x : ℝ) : 
  x - 4.3 = 8.8 → x + 4.3 = 17.4 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_operation_correction_l3986_398641
