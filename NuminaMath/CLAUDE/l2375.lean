import Mathlib

namespace investment_problem_l2375_237533

/-- Given two investment projects with specific conditions, 
    prove the minimum distance between them and the profitability of the deal. -/
theorem investment_problem 
  (p₁ x₁ p₂ x₂ : ℝ) 
  (h₁ : 4 * x₁ - 3 * p₁ - 44 = 0) 
  (h₂ : p₂^2 - 12 * p₂ + x₂^2 - 8 * x₂ + 43 = 0) 
  (h₃ : p₁ > 0) 
  (h₄ : p₂ > 0) : 
  let d := Real.sqrt ((x₁ - x₂)^2 + (p₁ - p₂)^2)
  ∃ (min_d : ℝ), 
    (∀ p₁' x₁' p₂' x₂', 
      4 * x₁' - 3 * p₁' - 44 = 0 → 
      p₂'^2 - 12 * p₂' + x₂'^2 - 8 * x₂' + 43 = 0 → 
      p₁' > 0 → 
      p₂' > 0 → 
      Real.sqrt ((x₁' - x₂')^2 + (p₁' - p₂')^2) ≥ min_d) ∧ 
    d = min_d ∧ 
    min_d = 6.2 ∧ 
    x₁ + x₂ - p₁ - p₂ > 0 := by
  sorry

end investment_problem_l2375_237533


namespace angle_measure_l2375_237568

-- Define the angle
def angle : ℝ := sorry

-- Define the complement of the angle
def complement : ℝ := 90 - angle

-- Define the supplement of the angle
def supplement : ℝ := 180 - angle

-- State the theorem
theorem angle_measure : 
  supplement = 4 * complement + 15 → angle = 65 := by sorry

end angle_measure_l2375_237568


namespace rug_on_floor_l2375_237580

theorem rug_on_floor (rug_length : ℝ) (rug_width : ℝ) (floor_area : ℝ) : 
  rug_length = 2 →
  rug_width = 7 →
  floor_area = 64 →
  rug_length * rug_width ≤ floor_area →
  (floor_area - rug_length * rug_width) / floor_area = 25 / 32 := by
  sorry

end rug_on_floor_l2375_237580


namespace solve_equations_l2375_237514

theorem solve_equations :
  (∃ x : ℝ, 2 * (x + 8) = 3 * (x - 1) ∧ x = 19) ∧
  (∃ y : ℝ, (3 * y - 1) / 4 - 1 = (5 * y - 7) / 6 ∧ y = -1) :=
by sorry

end solve_equations_l2375_237514


namespace statements_proof_l2375_237560

theorem statements_proof :
  (∀ a b c : ℝ, a > b → c < 0 → a^3 * c < b^3 * c) ∧
  (∀ a b c : ℝ, c > a → a > b → b > 0 → a / (c - a) > b / (c - b)) ∧
  (∀ a b : ℝ, a > b → (1 : ℝ) / a > (1 : ℝ) / b → a > 0 ∧ b < 0) := by
  sorry

end statements_proof_l2375_237560


namespace amount_spent_on_books_l2375_237543

/-- Calculates the amount spent on books given the total allowance and percentages spent on other items --/
theorem amount_spent_on_books
  (total_allowance : ℚ)
  (games_percentage : ℚ)
  (clothes_percentage : ℚ)
  (snacks_percentage : ℚ)
  (h1 : total_allowance = 50)
  (h2 : games_percentage = 1/4)
  (h3 : clothes_percentage = 2/5)
  (h4 : snacks_percentage = 3/20) :
  total_allowance - (games_percentage + clothes_percentage + snacks_percentage) * total_allowance = 10 :=
by sorry

end amount_spent_on_books_l2375_237543


namespace birdhouse_earnings_l2375_237508

/-- The price of a large birdhouse in dollars -/
def large_price : ℕ := 22

/-- The price of a medium birdhouse in dollars -/
def medium_price : ℕ := 16

/-- The price of a small birdhouse in dollars -/
def small_price : ℕ := 7

/-- The number of large birdhouses sold -/
def large_sold : ℕ := 2

/-- The number of medium birdhouses sold -/
def medium_sold : ℕ := 2

/-- The number of small birdhouses sold -/
def small_sold : ℕ := 3

/-- The total money earned from selling birdhouses -/
def total_earned : ℕ := large_price * large_sold + medium_price * medium_sold + small_price * small_sold

theorem birdhouse_earnings : total_earned = 97 := by
  sorry

end birdhouse_earnings_l2375_237508


namespace inequality_solution_set_l2375_237525

theorem inequality_solution_set (a b c : ℝ) : 
  a > 0 → 
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 2 ↔ 0 ≤ a * x^2 + b * x + c ∧ a * x^2 + b * x + c ≤ 1) →
  4 * a + 5 * b + c = -1/4 ∨ 4 * a + 5 * b + c = Real.sqrt 3 / 2 := by
  sorry

end inequality_solution_set_l2375_237525


namespace not_power_of_two_l2375_237530

theorem not_power_of_two (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ¬ ∃ k : ℕ, (36 * a + b) * (a + 36 * b) = 2^k :=
by sorry

end not_power_of_two_l2375_237530


namespace special_function_properties_l2375_237573

/-- A function satisfying the given properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y - 3) ∧
  (∀ x : ℝ, x > 0 → f x < 3)

theorem special_function_properties (f : ℝ → ℝ) (hf : SpecialFunction f) :
  (f 0 = 3) ∧
  (∀ x y : ℝ, x < y → f y < f x) ∧
  (∀ x : ℝ, (∀ t : ℝ, t ∈ Set.Ioo 2 4 → 
    f ((t - 2) * |x - 4|) + 3 > f (t^2 + 8) + f (5 - 4*t)) →
    x ∈ Set.Icc (-5/2) (21/2)) :=
by sorry

end special_function_properties_l2375_237573


namespace correct_observation_value_l2375_237529

theorem correct_observation_value (n : ℕ) (initial_mean corrected_mean wrong_value : ℝ) 
  (h1 : n = 50)
  (h2 : initial_mean = 40)
  (h3 : corrected_mean = 40.66)
  (h4 : wrong_value = 15) :
  let total_sum := n * initial_mean
  let corrected_sum := n * corrected_mean
  let difference := corrected_sum - total_sum
  let actual_value := wrong_value + difference
  actual_value = 48 := by sorry

end correct_observation_value_l2375_237529


namespace square_difference_equality_l2375_237581

theorem square_difference_equality : 1007^2 - 993^2 - 1005^2 + 995^2 = 8000 := by
  sorry

end square_difference_equality_l2375_237581


namespace triangle_area_l2375_237520

theorem triangle_area (a c : Real) (B : Real) 
  (h1 : a = Real.sqrt 2)
  (h2 : c = 2 * Real.sqrt 2)
  (h3 : B = 30 * π / 180) :
  (1/2) * a * c * Real.sin B = 1 := by sorry

end triangle_area_l2375_237520


namespace cody_marbles_l2375_237552

/-- The number of marbles Cody gave to his brother -/
def marbles_given : ℕ := 5

/-- The number of marbles Cody has now -/
def marbles_now : ℕ := 7

/-- The initial number of marbles Cody had -/
def initial_marbles : ℕ := marbles_now + marbles_given

theorem cody_marbles : initial_marbles = 12 := by
  sorry

end cody_marbles_l2375_237552


namespace solution_set_equivalence_l2375_237599

/-- Given that the solution set of (ax+1)/(x+b) > 1 is (-∞, -1) ∪ (3, +∞),
    prove that the solution set of x^2 + ax - 2b < 0 is (-3, -2) -/
theorem solution_set_equivalence (a b : ℝ) :
  ({x : ℝ | (a * x + 1) / (x + b) > 1} = {x : ℝ | x < -1 ∨ x > 3}) →
  {x : ℝ | x^2 + a*x - 2*b < 0} = {x : ℝ | -3 < x ∧ x < -2} :=
by sorry

end solution_set_equivalence_l2375_237599


namespace karlsson_candies_l2375_237564

/-- The number of ones initially written on the board -/
def initial_ones : ℕ := 28

/-- The number of minutes the process continues -/
def total_minutes : ℕ := 28

/-- The number of edges in a complete graph with n vertices -/
def complete_graph_edges (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The maximum number of candies Karlsson could eat -/
def max_candies : ℕ := complete_graph_edges initial_ones

theorem karlsson_candies :
  max_candies = 378 :=
sorry

end karlsson_candies_l2375_237564


namespace sandals_sold_example_l2375_237565

/-- Given a ratio of shoes to sandals and the number of shoes sold, 
    calculate the number of sandals sold. -/
def sandals_sold (shoe_ratio : ℕ) (sandal_ratio : ℕ) (shoes : ℕ) : ℕ :=
  (shoes / shoe_ratio) * sandal_ratio

/-- Theorem stating that given the specific ratio and number of shoes sold,
    the number of sandals sold is 40. -/
theorem sandals_sold_example : sandals_sold 9 5 72 = 40 := by
  sorry

end sandals_sold_example_l2375_237565


namespace odd_m_triple_g_16_l2375_237544

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5
  else if n % 3 = 0 ∧ n % 2 ≠ 0 then n / 3
  else n / 2

theorem odd_m_triple_g_16 (m : ℤ) (h1 : m % 2 = 1) (h2 : g (g (g m)) = 16) :
  m = 59 ∨ m = 91 := by
  sorry

end odd_m_triple_g_16_l2375_237544


namespace pet_store_cages_l2375_237554

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 56)
  (h2 : sold_puppies = 24)
  (h3 : puppies_per_cage = 4) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 8 :=
by sorry

end pet_store_cages_l2375_237554


namespace triangle_with_small_angle_l2375_237592

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a set of n points
def PointSet (n : ℕ) := Fin n → Point

-- Define a function to calculate the angle between three points
noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

-- Theorem statement
theorem triangle_with_small_angle (n : ℕ) (points : PointSet n) :
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    ∃ (θ : ℝ), θ ≤ 180 / n ∧
      (θ = angle (points i) (points j) (points k) ∨
       θ = angle (points j) (points k) (points i) ∨
       θ = angle (points k) (points i) (points j)) :=
sorry

end triangle_with_small_angle_l2375_237592


namespace cos_2_sum_of_tan_roots_l2375_237519

theorem cos_2_sum_of_tan_roots (α β : ℝ) : 
  (∃ x y : ℝ, x^2 + 5*x - 6 = 0 ∧ y^2 + 5*y - 6 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  Real.cos (2 * (α + β)) = 12 / 37 := by
sorry

end cos_2_sum_of_tan_roots_l2375_237519


namespace nonagon_angle_measure_l2375_237590

theorem nonagon_angle_measure :
  ∀ (small_angle large_angle : ℝ),
  (9 : ℝ) * small_angle + (9 : ℝ) * large_angle = (7 : ℝ) * 180 →
  6 * small_angle + 3 * large_angle = (7 : ℝ) * 180 →
  large_angle = 3 * small_angle →
  large_angle = 252 := by
  sorry

end nonagon_angle_measure_l2375_237590


namespace homework_decrease_iff_thirty_percent_l2375_237575

/-- Represents the decrease in homework duration over two reforms -/
def homework_decrease (a : ℝ) (x : ℝ) : Prop :=
  a * (1 - x)^2 = 0.3 * a

/-- Theorem stating that the homework decrease equation holds if and only if
    the final duration is 30% of the initial duration -/
theorem homework_decrease_iff_thirty_percent (a : ℝ) (x : ℝ) (h_a : a > 0) :
  homework_decrease a x ↔ a * (1 - x)^2 = 0.3 * a :=
sorry

end homework_decrease_iff_thirty_percent_l2375_237575


namespace product_of_three_numbers_l2375_237594

theorem product_of_three_numbers (a b c : ℝ) 
  (sum_eq : a + b + c = 30)
  (first_eq : a = 5 * (b + c))
  (second_eq : b = 9 * c) :
  a * b * c = 56.25 := by
  sorry

end product_of_three_numbers_l2375_237594


namespace factor_210_into_four_l2375_237518

def prime_factors : Multiset ℕ := {2, 3, 5, 7}

/-- The number of ways to partition a multiset of 4 distinct elements into 4 non-empty subsets -/
def partition_count (m : Multiset ℕ) : ℕ := sorry

theorem factor_210_into_four : partition_count prime_factors = 15 := by sorry

end factor_210_into_four_l2375_237518


namespace S_min_at_5_l2375_237589

/-- An arithmetic sequence with first term -9 and S_3 = S_7 -/
def ArithSeq : ℕ → ℤ := fun n => 2*n - 11

/-- Sum of first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℤ := n * (ArithSeq 1 + ArithSeq n) / 2

/-- The condition that S_3 = S_7 -/
axiom S3_eq_S7 : S 3 = S 7

/-- The theorem stating that S_n is minimized when n = 5 -/
theorem S_min_at_5 : ∀ n : ℕ, n ≠ 0 → S 5 ≤ S n :=
sorry

end S_min_at_5_l2375_237589


namespace ellipse_equation_l2375_237515

/-- Given an ellipse with equation x²/a² + 25y²/(9a²) = 1, prove that the equation
    of the ellipse is x² + 25/9 * y² = 1 under the following conditions:
    - Points A and B are on the ellipse
    - F₂ is the right focus of the ellipse
    - |AF₂| + |BF₂| = 8/5 * a
    - Distance from midpoint of AB to left directrix is 3/2 -/
theorem ellipse_equation (a : ℝ) (A B F₂ : ℝ × ℝ) :
  (∀ x y, x^2/a^2 + 25*y^2/(9*a^2) = 1 → (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)) →
  (F₂.1 > 0) →
  (Real.sqrt ((A.1 - F₂.1)^2 + (A.2 - F₂.2)^2) + Real.sqrt ((B.1 - F₂.1)^2 + (B.2 - F₂.2)^2) = 8/5 * a) →
  (((A.1 + B.1)/2 + 5/4*a) = 3/2) →
  (∀ x y, x^2 + 25/9 * y^2 = 1) :=
by sorry

end ellipse_equation_l2375_237515


namespace average_speed_is_27_point_5_l2375_237597

-- Define the initial and final odometer readings
def initial_reading : ℕ := 1551
def final_reading : ℕ := 1881

-- Define the total riding time in hours
def total_time : ℕ := 12

-- Define the average speed
def average_speed : ℚ := (final_reading - initial_reading : ℚ) / total_time

-- Theorem statement
theorem average_speed_is_27_point_5 : average_speed = 27.5 := by
  sorry

end average_speed_is_27_point_5_l2375_237597


namespace probability_ratio_l2375_237517

def total_cards : ℕ := 50
def numbers_range : ℕ := 10
def cards_per_number : ℕ := 5
def cards_drawn : ℕ := 5

def probability_all_same (total : ℕ) (range : ℕ) (per_num : ℕ) (drawn : ℕ) : ℚ :=
  (range : ℚ) / (total.choose drawn)

def probability_four_and_one (total : ℕ) (range : ℕ) (per_num : ℕ) (drawn : ℕ) : ℚ :=
  ((range * (range - 1)) * (per_num.choose (drawn - 1)) * (per_num.choose 1) : ℚ) / (total.choose drawn)

theorem probability_ratio :
  (probability_four_and_one total_cards numbers_range cards_per_number cards_drawn) /
  (probability_all_same total_cards numbers_range cards_per_number cards_drawn) = 225 := by
  sorry

end probability_ratio_l2375_237517


namespace evaluate_expression_l2375_237559

theorem evaluate_expression : (32 / (7 + 3 - 5)) * 8 = 51.2 := by
  sorry

end evaluate_expression_l2375_237559


namespace minimal_hexahedron_volume_l2375_237563

/-- A trihedral angle -/
structure TrihedralAngle where
  planarAngle : ℝ

/-- The configuration of two trihedral angles -/
structure TrihedralAngleConfiguration where
  angle1 : TrihedralAngle
  angle2 : TrihedralAngle
  vertexDistance : ℝ
  isEquidistant : Bool

/-- The volume of the hexahedron bounded by the faces of two trihedral angles -/
def hexahedronVolume (config : TrihedralAngleConfiguration) : ℝ := sorry

/-- The theorem stating the minimal volume of the hexahedron -/
theorem minimal_hexahedron_volume 
  (config : TrihedralAngleConfiguration) 
  (h1 : config.angle1.planarAngle = π/3) 
  (h2 : config.angle2.planarAngle = π/2)
  (h3 : config.isEquidistant = true) :
  hexahedronVolume config = (config.vertexDistance^3 * Real.sqrt 3) / 20 := by
  sorry


end minimal_hexahedron_volume_l2375_237563


namespace graph_single_point_implies_d_value_l2375_237579

/-- The equation of the graph -/
def graph_equation (x y : ℝ) (d : ℝ) : Prop :=
  x^2 + 3*y^2 + 6*x - 18*y + d = 0

/-- The graph consists of a single point -/
def single_point (d : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, graph_equation p.1 p.2 d

/-- If the graph of x^2 + 3y^2 + 6x - 18y + d = 0 consists of a single point, then d = -27 -/
theorem graph_single_point_implies_d_value :
  ∀ d : ℝ, single_point d → d = -27 := by
  sorry

end graph_single_point_implies_d_value_l2375_237579


namespace sum_of_special_function_l2375_237541

theorem sum_of_special_function (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (1/2 + x) + f (1/2 - x) = 2) : 
  f (1/8) + f (2/8) + f (3/8) + f (4/8) + f (5/8) + f (6/8) + f (7/8) = 7 := by
  sorry

end sum_of_special_function_l2375_237541


namespace rectangle_roots_l2375_237538

/-- The polynomial whose roots we are considering -/
def f (a : ℝ) (z : ℂ) : ℂ := z^4 - 8*z^3 + 13*a*z^2 - 3*(3*a^2 + 2*a - 4)*z + 1

/-- Predicate to check if four complex numbers form vertices of a rectangle -/
def isRectangle (z₁ z₂ z₃ z₄ : ℂ) : Prop := sorry

/-- The theorem stating that a = 3 is the only real value satisfying the condition -/
theorem rectangle_roots (a : ℝ) : 
  (∃ z₁ z₂ z₃ z₄ : ℂ, f a z₁ = 0 ∧ f a z₂ = 0 ∧ f a z₃ = 0 ∧ f a z₄ = 0 ∧ 
    isRectangle z₁ z₂ z₃ z₄) ↔ a = 3 := by sorry

end rectangle_roots_l2375_237538


namespace line_slope_l2375_237578

/-- The slope of the line given by the equation x/4 + y/5 = 1 is -5/4 -/
theorem line_slope (x y : ℝ) : 
  (x / 4 + y / 5 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -5/4) :=
by sorry

end line_slope_l2375_237578


namespace extremum_point_implies_a_eq_3_f_increasing_when_a_le_2_max_m_value_l2375_237556

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x^2 - a*x

theorem extremum_point_implies_a_eq_3 :
  ∀ a : ℝ, (∀ h : ℝ, h ≠ 0 → (f a (1 + h) - f a 1) / h = 0) → a = 3 :=
sorry

theorem f_increasing_when_a_le_2 :
  ∀ a : ℝ, 0 < a → a ≤ 2 → StrictMono (f a) :=
sorry

theorem max_m_value :
  ∃ m : ℝ, m = -(Real.log 2)⁻¹ ∧
  (∀ a x₀ : ℝ, 1 < a → a < 2 → 1 ≤ x₀ → x₀ ≤ 2 → f a x₀ > m * Real.log a) ∧
  (∀ m' : ℝ, m' > m → ∃ a x₀ : ℝ, 1 < a ∧ a < 2 ∧ 1 ≤ x₀ ∧ x₀ ≤ 2 ∧ f a x₀ ≤ m' * Real.log a) :=
sorry

end extremum_point_implies_a_eq_3_f_increasing_when_a_le_2_max_m_value_l2375_237556


namespace angle_a_is_sixty_degrees_l2375_237526

/-- In a triangle ABC, if the sum of angles B and C is twice angle A, then angle A is 60 degrees. -/
theorem angle_a_is_sixty_degrees (A B C : ℝ) (h1 : A + B + C = 180) (h2 : B + C = 2 * A) : A = 60 := by
  sorry

end angle_a_is_sixty_degrees_l2375_237526


namespace algebraic_expression_value_l2375_237549

theorem algebraic_expression_value (a b : ℝ) (h : a + b - 2 = 0) :
  a^2 - b^2 + 4*b = 4 := by sorry

end algebraic_expression_value_l2375_237549


namespace arithmetic_sequence_common_difference_l2375_237504

/-- Given an arithmetic sequence {aₙ} with S₃ = 6 and a₃ = 4, prove that the common difference d = 2 -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sequence of partial sums
  (h1 : S 3 = 6)  -- Given S₃ = 6
  (h2 : a 3 = 4)  -- Given a₃ = 4
  (h3 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)  -- Sum formula for arithmetic sequence
  (h4 : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)  -- Definition of arithmetic sequence
  : ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 2 :=
sorry

end arithmetic_sequence_common_difference_l2375_237504


namespace average_of_first_group_l2375_237546

theorem average_of_first_group (n₁ : ℕ) (n₂ : ℕ) (avg₂ : ℝ) (avg_total : ℝ) :
  n₁ = 40 →
  n₂ = 30 →
  avg₂ = 40 →
  avg_total = 34.285714285714285 →
  (n₁ * (n₁ + n₂) * avg_total - n₂ * avg₂ * (n₁ + n₂)) / (n₁ * (n₁ + n₂)) = 30 :=
by
  sorry

end average_of_first_group_l2375_237546


namespace exam_score_proof_l2375_237532

/-- Proves that the average score of students who took the exam on the assigned day is 60% -/
theorem exam_score_proof (total_students : ℕ) (assigned_day_percentage : ℝ) 
  (makeup_score : ℝ) (class_average : ℝ) : 
  total_students = 100 →
  assigned_day_percentage = 0.7 →
  makeup_score = 90 →
  class_average = 69 →
  let assigned_students := total_students * assigned_day_percentage
  let makeup_students := total_students - assigned_students
  let assigned_score := (class_average * total_students - makeup_score * makeup_students) / assigned_students
  assigned_score = 60 := by
sorry


end exam_score_proof_l2375_237532


namespace vans_needed_l2375_237500

def van_capacity : ℕ := 5
def num_students : ℕ := 12
def num_adults : ℕ := 3

theorem vans_needed : 
  (num_students + num_adults + van_capacity - 1) / van_capacity = 3 := by
sorry

end vans_needed_l2375_237500


namespace intersection_M_N_l2375_237516

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end intersection_M_N_l2375_237516


namespace expression_simplification_l2375_237548

theorem expression_simplification (a : ℝ) (h : a^2 + 2*a - 8 = 0) :
  ((a^2 - 4) / (a^2 - 4*a + 4) - a / (a - 2)) / ((a^2 + 2*a) / (a - 2)) = 1/4 := by
  sorry

end expression_simplification_l2375_237548


namespace existence_of_integers_l2375_237537

theorem existence_of_integers (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃ (x y k : ℤ), 0 < 2 * k ∧ 2 * k < p ∧ k * p + 3 = x^2 + y^2 := by
  sorry

end existence_of_integers_l2375_237537


namespace sum_of_numbers_l2375_237522

theorem sum_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 16) 
  (h4 : 1 / x = 3 * (1 / y)) : x + y = (16 * Real.sqrt 3) / 3 := by
  sorry

end sum_of_numbers_l2375_237522


namespace perfect_square_condition_l2375_237531

theorem perfect_square_condition (m n : ℕ) (hm : m > 1) (hn : n > 1) :
  (∃ k : ℕ, 2^m + 3^n = k^2) ↔ 
  (∃ a b : ℕ, m = 2*a ∧ n = 2*b ∧ a ≥ 1 ∧ b ≥ 1) :=
sorry

end perfect_square_condition_l2375_237531


namespace exam_score_problem_l2375_237545

theorem exam_score_problem (correct_score : ℕ) (wrong_score : ℕ) 
  (total_score : ℕ) (num_correct : ℕ) :
  correct_score = 3 →
  wrong_score = 1 →
  total_score = 180 →
  num_correct = 75 →
  ∃ (num_wrong : ℕ), 
    total_score = correct_score * num_correct - wrong_score * num_wrong ∧
    num_correct + num_wrong = 120 :=
by sorry

end exam_score_problem_l2375_237545


namespace conic_is_ellipse_l2375_237539

/-- Represents a conic section --/
inductive ConicSection
| Parabola
| Circle
| Ellipse
| Hyperbola
| Point
| Line
| TwoLines
| Empty

/-- Determines if the given equation represents an ellipse --/
def is_ellipse (a b h k : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ≠ b

/-- The equation of the conic section --/
def conic_equation (x y : ℝ) : Prop :=
  x^2 + 6*x + 9*y^2 - 36 = 0

/-- Theorem stating that the given equation represents an ellipse --/
theorem conic_is_ellipse : 
  ∃ (a b h k : ℝ), 
    (∀ (x y : ℝ), conic_equation x y ↔ ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1)) ∧
    is_ellipse a b h k :=
sorry


end conic_is_ellipse_l2375_237539


namespace curve_tangent_problem_l2375_237550

theorem curve_tangent_problem (a b : ℝ) : 
  (2 * a + b / 2 = -5) →  -- Curve passes through (2, -5)
  (4 * a - b / 4 = -7/2) →  -- Tangent slope at (2, -5) is -7/2
  a + b = -3 := by
  sorry

end curve_tangent_problem_l2375_237550


namespace abs_sum_equals_sum_abs_necessary_not_sufficient_l2375_237507

theorem abs_sum_equals_sum_abs_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a * b > 0 → |a + b| = |a| + |b|) ∧
  (∃ a b : ℝ, |a + b| = |a| + |b| ∧ a * b ≤ 0) :=
by sorry

end abs_sum_equals_sum_abs_necessary_not_sufficient_l2375_237507


namespace equation_equivalence_l2375_237583

theorem equation_equivalence (y : ℝ) (Q : ℝ) (h : 5 * (3 * y + 7 * Real.pi) = Q) :
  10 * (6 * y + 14 * Real.pi) = 4 * Q :=
by sorry

end equation_equivalence_l2375_237583


namespace average_salary_proof_l2375_237593

def workshop_problem (total_workers : ℕ) (technicians : ℕ) (avg_salary_technicians : ℚ) (avg_salary_others : ℚ) : Prop :=
  let non_technicians : ℕ := total_workers - technicians
  let total_salary_technicians : ℚ := technicians * avg_salary_technicians
  let total_salary_others : ℚ := non_technicians * avg_salary_others
  let total_salary : ℚ := total_salary_technicians + total_salary_others
  let avg_salary_all : ℚ := total_salary / total_workers
  avg_salary_all = 8000

theorem average_salary_proof :
  workshop_problem 28 7 14000 6000 := by
  sorry

end average_salary_proof_l2375_237593


namespace cos_difference_value_l2375_237558

theorem cos_difference_value (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by
sorry

end cos_difference_value_l2375_237558


namespace percent_relation_l2375_237506

theorem percent_relation (x y : ℝ) (h : (1/2) * (x - y) = (1/5) * (x + y)) : 
  y = (3/7) * x := by
sorry

end percent_relation_l2375_237506


namespace jack_final_apples_l2375_237540

def initial_apples : ℕ := 150
def sold_to_jill_percent : ℚ := 30 / 100
def sold_to_june_percent : ℚ := 20 / 100
def apples_eaten : ℕ := 2
def apples_given_to_teacher : ℕ := 1

theorem jack_final_apples :
  let after_jill := initial_apples - (initial_apples * sold_to_jill_percent).floor
  let after_june := after_jill - (after_jill * sold_to_june_percent).floor
  let after_eating := after_june - apples_eaten
  let final_apples := after_eating - apples_given_to_teacher
  final_apples = 81 := by sorry

end jack_final_apples_l2375_237540


namespace star_not_associative_l2375_237523

-- Define the set T as non-zero real numbers
def T := {x : ℝ | x ≠ 0}

-- Define the binary operation ★
def star (x y : ℝ) : ℝ := 3 * x * y + x + y

-- Theorem stating that ★ is not associative over T
theorem star_not_associative :
  ∃ (x y z : T), star (star x y) z ≠ star x (star y z) := by
  sorry

end star_not_associative_l2375_237523


namespace andreys_stamps_l2375_237511

theorem andreys_stamps :
  ∃ (x : ℕ), x > 0 ∧ x % 3 = 1 ∧ x % 5 = 3 ∧ x % 7 = 5 ∧ x = 208 := by
  sorry

end andreys_stamps_l2375_237511


namespace value_equals_scientific_notation_l2375_237596

/-- Represents the value in billion yuan -/
def value : ℝ := 24953

/-- Represents the scientific notation coefficient -/
def coefficient : ℝ := 2.4953

/-- Represents the scientific notation exponent -/
def exponent : ℕ := 13

/-- Theorem stating that the given value in billion yuan is equal to its scientific notation representation -/
theorem value_equals_scientific_notation : value * 10^9 = coefficient * 10^exponent := by
  sorry

end value_equals_scientific_notation_l2375_237596


namespace only_origin_satisfies_l2375_237535

def satisfies_inequality (x y : ℝ) : Prop := x + y - 1 < 0

theorem only_origin_satisfies : 
  satisfies_inequality 0 0 ∧ 
  ¬satisfies_inequality 2 4 ∧ 
  ¬satisfies_inequality (-1) 4 ∧ 
  ¬satisfies_inequality 1 8 :=
by sorry

end only_origin_satisfies_l2375_237535


namespace ten_passengers_five_stops_l2375_237595

/-- The number of ways for passengers to get off a bus -/
def bus_stop_combinations (num_passengers : ℕ) (num_stops : ℕ) : ℕ :=
  num_stops ^ num_passengers

/-- Theorem: 10 passengers and 5 stops result in 5^10 combinations -/
theorem ten_passengers_five_stops :
  bus_stop_combinations 10 5 = 5^10 := by
  sorry

end ten_passengers_five_stops_l2375_237595


namespace sum_of_coefficients_l2375_237513

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 1)^8 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3 + 
    a₄*(x-2)^4 + a₅*(x-2)^5 + a₆*(x-2)^6 + a₇*(x-2)^7 + a₈*(x-2)^8 + a₉*(x-2)^9 + a₁₀*(x-2)^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 2555 := by
sorry

end sum_of_coefficients_l2375_237513


namespace parallelogram_height_l2375_237503

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area base height : ℝ) (h1 : area = 704) (h2 : base = 32) 
  (h3 : area = base * height) : height = 22 := by
  sorry

end parallelogram_height_l2375_237503


namespace smallest_fourth_number_l2375_237586

def sum_of_digits (n : ℕ) : ℕ := n % 10 + n / 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem smallest_fourth_number :
  let known_numbers := [34, 56, 45]
  let sum_known := known_numbers.sum
  let sum_digits_known := (known_numbers.map sum_of_digits).sum
  ∃ x : ℕ,
    is_two_digit x ∧
    (∀ y : ℕ, is_two_digit y →
      sum_digits_known + sum_of_digits x + sum_digits_known + sum_of_digits y = (sum_known + x + sum_known + y) / 3
      → x ≤ y) ∧
    x = 35 := by
  sorry

end smallest_fourth_number_l2375_237586


namespace expected_value_is_six_point_five_l2375_237524

/-- A fair 12-sided die with faces numbered from 1 to 12 -/
def twelve_sided_die : Finset ℕ := Finset.range 12

/-- The expected value of rolling a fair 12-sided die with faces numbered from 1 to 12 -/
def expected_value : ℚ :=
  (Finset.sum twelve_sided_die (λ i => i + 1)) / 12

/-- Theorem: The expected value of rolling a fair 12-sided die with faces numbered from 1 to 12 is 6.5 -/
theorem expected_value_is_six_point_five :
  expected_value = 13/2 := by sorry

end expected_value_is_six_point_five_l2375_237524


namespace expression_simplification_l2375_237582

theorem expression_simplification (x y : ℚ) (hx : x = 1/9) (hy : y = 5) :
  -1/5 * x * y^2 - 3 * x^2 * y + x * y^2 + 2 * x^2 * y + 3 * x * y^2 + x^2 * y - 2 * x * y^2 = 20/9 := by
  sorry

end expression_simplification_l2375_237582


namespace isosceles_triangle_same_color_l2375_237555

-- Define a circle
def Circle : Type := Unit

-- Define a color type
inductive Color
| C1
| C2

-- Define a point on the circle
structure Point (c : Circle) where
  color : Color

-- Define an isosceles triangle
structure IsoscelesTriangle (c : Circle) where
  p1 : Point c
  p2 : Point c
  p3 : Point c
  isIsosceles : True  -- We assume this property without proving it

-- State the theorem
theorem isosceles_triangle_same_color (c : Circle) 
  (coloring : Point c → Color) :
  ∃ (t : IsoscelesTriangle c), 
    t.p1.color = t.p2.color ∧ 
    t.p2.color = t.p3.color :=
sorry

end isosceles_triangle_same_color_l2375_237555


namespace combined_sale_price_l2375_237505

/-- Calculate the sale price given the purchase cost and profit percentage -/
def calculateSalePrice (purchaseCost : ℚ) (profitPercentage : ℚ) : ℚ :=
  purchaseCost * (1 + profitPercentage)

/-- The problem statement -/
theorem combined_sale_price :
  let itemA_cost : ℚ := 650
  let itemB_cost : ℚ := 350
  let itemC_cost : ℚ := 400
  let itemA_profit : ℚ := 0.40
  let itemB_profit : ℚ := 0.25
  let itemC_profit : ℚ := 0.30
  let itemA_sale := calculateSalePrice itemA_cost itemA_profit
  let itemB_sale := calculateSalePrice itemB_cost itemB_profit
  let itemC_sale := calculateSalePrice itemC_cost itemC_profit
  itemA_sale + itemB_sale + itemC_sale = 1867.50 := by
  sorry

end combined_sale_price_l2375_237505


namespace firm_employs_80_looms_l2375_237527

/-- Represents a textile manufacturing firm with looms -/
structure TextileFirm where
  totalSales : ℕ
  manufacturingExpenses : ℕ
  establishmentCharges : ℕ
  profitDecreaseOnBreakdown : ℕ

/-- Calculates the number of looms employed by the firm -/
def calculateLooms (firm : TextileFirm) : ℕ :=
  (firm.totalSales - firm.manufacturingExpenses) / firm.profitDecreaseOnBreakdown

/-- Theorem stating that the firm employs 80 looms -/
theorem firm_employs_80_looms (firm : TextileFirm) 
  (h1 : firm.totalSales = 500000)
  (h2 : firm.manufacturingExpenses = 150000)
  (h3 : firm.establishmentCharges = 75000)
  (h4 : firm.profitDecreaseOnBreakdown = 4375) :
  calculateLooms firm = 80 := by
  sorry

#eval calculateLooms { totalSales := 500000, 
                       manufacturingExpenses := 150000, 
                       establishmentCharges := 75000, 
                       profitDecreaseOnBreakdown := 4375 }

end firm_employs_80_looms_l2375_237527


namespace smallest_positive_angle_with_same_terminal_side_l2375_237542

theorem smallest_positive_angle_with_same_terminal_side (angle : Real) : 
  angle = -660 * Real.pi / 180 → 
  ∃ (k : ℤ), (angle + 2 * Real.pi * k) % (2 * Real.pi) = Real.pi / 3 ∧ 
  ∀ (x : Real), 0 < x ∧ x < Real.pi / 3 → 
  ¬∃ (m : ℤ), (angle + 2 * Real.pi * m) % (2 * Real.pi) = x :=
by sorry

end smallest_positive_angle_with_same_terminal_side_l2375_237542


namespace count_valid_numbers_valid_numbers_are_l2375_237551

def digits : List Nat := [2, 3, 0]

def is_valid_number (n : Nat) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧
  d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits

def valid_numbers : List Nat :=
  (List.range 1000).filter is_valid_number

theorem count_valid_numbers :
  valid_numbers.length = 4 := by sorry

theorem valid_numbers_are :
  valid_numbers = [230, 203, 302, 320] := by sorry

end count_valid_numbers_valid_numbers_are_l2375_237551


namespace equation_solution_l2375_237572

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  -2 * x^2 = (4 * x + 2) / (x - 2) ↔ x = 1 :=
by sorry

end equation_solution_l2375_237572


namespace M_inter_N_eq_M_l2375_237534

def M : Set ℝ := {x | x^2 - x < 0}
def N : Set ℝ := {x | |x| < 2}

theorem M_inter_N_eq_M : M ∩ N = M := by
  sorry

end M_inter_N_eq_M_l2375_237534


namespace problem1_l2375_237591

theorem problem1 (x : ℝ) : (12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1) = 2 * x - 4 * x^2 := by
  sorry

end problem1_l2375_237591


namespace perfect_square_proof_l2375_237528

theorem perfect_square_proof (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_odd_b : Odd b) (h_int : ∃ k : ℤ, (a + b)^2 + 4*a = k * a * b) : 
  ∃ u : ℕ, a = u^2 := by
  sorry

end perfect_square_proof_l2375_237528


namespace sin_cos_sum_21_39_l2375_237502

theorem sin_cos_sum_21_39 : 
  Real.sin (21 * π / 180) * Real.cos (39 * π / 180) + 
  Real.cos (21 * π / 180) * Real.sin (39 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end sin_cos_sum_21_39_l2375_237502


namespace unique_square_of_divisors_l2375_237585

/-- The number of positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- n is a positive integer that equals the square of its number of positive divisors -/
def is_square_of_divisors (n : ℕ) : Prop :=
  n > 0 ∧ n = (num_divisors n) ^ 2

theorem unique_square_of_divisors :
  ∃! n : ℕ, is_square_of_divisors n ∧ n = 9 := by sorry

end unique_square_of_divisors_l2375_237585


namespace intersection_complement_theorem_l2375_237562

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def P : Set Nat := {1, 2, 3, 4}
def Q : Set Nat := {3, 4, 5}

theorem intersection_complement_theorem :
  P ∩ (U \ Q) = {1, 2} := by
  sorry

end intersection_complement_theorem_l2375_237562


namespace intersection_points_form_line_l2375_237561

theorem intersection_points_form_line (s : ℝ) :
  let x := s + 15
  let y := 2*s - 8
  (2*x + 3*y = 8*s + 6) ∧ (x + 2*y = 5*s - 1) →
  y = 2*x - 38 := by
sorry

end intersection_points_form_line_l2375_237561


namespace gcd_of_three_numbers_l2375_237588

theorem gcd_of_three_numbers :
  Nat.gcd 188094 (Nat.gcd 244122 395646) = 6 := by
  sorry

end gcd_of_three_numbers_l2375_237588


namespace sum_of_smallest_solutions_l2375_237569

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the equation
def equation (x : ℝ) : Prop := x - floor x = 2 / (floor x : ℝ)

-- Define the set of positive solutions
def positive_solutions : Set ℝ := {x : ℝ | x > 0 ∧ equation x}

-- State the theorem
theorem sum_of_smallest_solutions :
  ∃ (s₁ s₂ s₃ : ℝ),
    s₁ ∈ positive_solutions ∧
    s₂ ∈ positive_solutions ∧
    s₃ ∈ positive_solutions ∧
    (∀ x ∈ positive_solutions, x ≤ s₁ ∨ x ≤ s₂ ∨ x ≤ s₃) ∧
    s₁ + s₂ + s₃ = 13 + 17 / 30 :=
sorry

end sum_of_smallest_solutions_l2375_237569


namespace simplify_expression_l2375_237567

theorem simplify_expression (w : ℝ) : -2*w + 3 - 4*w + 7 + 6*w - 5 - 8*w + 8 = -8*w + 13 := by
  sorry

end simplify_expression_l2375_237567


namespace mass_percentage_not_sufficient_for_unique_compound_l2375_237536

/-- Represents a chemical compound -/
structure Compound where
  name : String
  mass_percentage_O : Float

/-- The mass percentage of O in the compound -/
def given_mass_percentage : Float := 36.36

/-- Theorem stating that the given mass percentage of O is not sufficient to uniquely determine a compound -/
theorem mass_percentage_not_sufficient_for_unique_compound :
  ∃ (c1 c2 : Compound), c1.mass_percentage_O = given_mass_percentage ∧ 
                        c2.mass_percentage_O = given_mass_percentage ∧ 
                        c1.name ≠ c2.name :=
sorry

end mass_percentage_not_sufficient_for_unique_compound_l2375_237536


namespace rational_fraction_value_l2375_237547

theorem rational_fraction_value (x y : ℝ) : 
  (x - y) / (x + y) = 4 → 
  ∃ (q : ℚ), x / y = ↑q →
  x / y = -5/3 := by
sorry

end rational_fraction_value_l2375_237547


namespace inequality_proof_l2375_237577

theorem inequality_proof (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) 
  (h_sum : x + 2*y + 3*z = 12) : x^2 + 2*y^3 + 3*z^2 > 24 := by
  sorry

end inequality_proof_l2375_237577


namespace rose_group_size_l2375_237570

theorem rose_group_size (n : ℕ+) (h : Nat.lcm n 19 = 171) : n = 9 := by
  sorry

end rose_group_size_l2375_237570


namespace expression_evaluation_l2375_237512

theorem expression_evaluation (x z : ℝ) (hz : z ≠ 0) (hx : x = 1 / z^2) :
  (x + 1/x) * (z^2 - 1/z^2) = z^4 - 1/z^4 := by
  sorry

end expression_evaluation_l2375_237512


namespace eraser_price_is_75_cents_l2375_237566

/-- The price of each eraser sold by the student council -/
def price_per_eraser (num_boxes : ℕ) (erasers_per_box : ℕ) (total_revenue : ℚ) : ℚ :=
  total_revenue / (num_boxes * erasers_per_box)

/-- Theorem: The price of each eraser is $0.75 -/
theorem eraser_price_is_75_cents :
  price_per_eraser 48 24 864 = 3/4 := by
  sorry

end eraser_price_is_75_cents_l2375_237566


namespace system_solution_l2375_237553

theorem system_solution : ∃! (x y z : ℝ), 
  x * y / (x + y) = 1 / 3 ∧
  y * z / (y + z) = 1 / 4 ∧
  z * x / (z + x) = 1 / 5 ∧
  x = 1 / 2 ∧ y = 1 ∧ z = 1 / 3 := by
sorry

end system_solution_l2375_237553


namespace teacher_number_game_l2375_237521

theorem teacher_number_game (x : ℝ) : 
  x = 5 → 3 * ((2 * (2 * x + 3)) + 2) = 84 := by
  sorry

end teacher_number_game_l2375_237521


namespace product_65_35_l2375_237557

theorem product_65_35 : 65 * 35 = 2275 := by
  sorry

end product_65_35_l2375_237557


namespace school_gender_ratio_l2375_237501

/-- The number of boys in the school -/
def num_boys : ℕ := 50

/-- The number of girls in the school -/
def num_girls : ℕ := num_boys + 80

/-- The ratio of boys to girls as a pair of natural numbers -/
def boys_to_girls_ratio : ℕ × ℕ := (5, 13)

theorem school_gender_ratio :
  (num_boys, num_girls) = (boys_to_girls_ratio.1 * 10, boys_to_girls_ratio.2 * 10) := by
  sorry

end school_gender_ratio_l2375_237501


namespace largest_sum_of_digits_l2375_237576

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the decimal 0.abc -/
def DecimalABC (a b c : Digit) : ℚ :=
  (a.val * 100 + b.val * 10 + c.val : ℕ) / 1000

theorem largest_sum_of_digits (a b c : Digit) (y : ℕ) 
  (h1 : DecimalABC a b c = 1 / y)
  (h2 : 0 < y) (h3 : y ≤ 16) :
  a.val + b.val + c.val ≤ 13 :=
sorry

end largest_sum_of_digits_l2375_237576


namespace rectangle_area_l2375_237587

theorem rectangle_area (width : ℝ) (h1 : width > 0) : 
  let length := 2 * width
  let diagonal := 10
  width ^ 2 + length ^ 2 = diagonal ^ 2 →
  width * length = 40 := by
  sorry

end rectangle_area_l2375_237587


namespace integral_sqrt_plus_x_equals_pi_over_two_l2375_237510

open Set
open MeasureTheory
open Interval

/-- The definite integral of √(1-x²) + x from -1 to 1 equals π/2 -/
theorem integral_sqrt_plus_x_equals_pi_over_two :
  ∫ x in (-1)..1, (Real.sqrt (1 - x^2) + x) = π / 2 := by sorry

end integral_sqrt_plus_x_equals_pi_over_two_l2375_237510


namespace solution_part1_solution_part2_l2375_237574

def f (x a : ℝ) := |2*x - 1| + |x - a|

theorem solution_part1 : 
  {x : ℝ | f x 3 ≤ 4} = Set.Icc 0 2 := by sorry

theorem solution_part2 (a : ℝ) :
  (∀ x, f x a = |x - 1 + a|) → 
  (a < 1/2 → {x : ℝ | f x a = |x - 1 + a|} = Set.Icc a (1/2)) ∧
  (a = 1/2 → {x : ℝ | f x a = |x - 1 + a|} = {1/2}) ∧
  (a > 1/2 → {x : ℝ | f x a = |x - 1 + a|} = Set.Icc (1/2) a) := by sorry

end solution_part1_solution_part2_l2375_237574


namespace probability_first_greater_than_second_l2375_237598

def card_set : Finset ℕ := {1, 2, 3, 4, 5}

def favorable_outcomes : Finset (ℕ × ℕ) :=
  {(2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3), (5, 1), (5, 2), (5, 3), (5, 4)}

theorem probability_first_greater_than_second :
  (Finset.card favorable_outcomes : ℚ) / (Finset.card card_set ^ 2 : ℚ) = 2 / 5 := by
  sorry

end probability_first_greater_than_second_l2375_237598


namespace unique_digit_multiplication_l2375_237509

theorem unique_digit_multiplication :
  ∃! (A B C D E : Nat),
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧
    A ≠ 0 ∧
    (A * 10000 + B * 1000 + C * 100 + D * 10 + E) * 4 =
    E * 10000 + D * 1000 + C * 100 + B * 10 + A ∧
    A = 2 ∧ B = 1 ∧ C = 9 ∧ D = 7 ∧ E = 8 :=
by sorry

end unique_digit_multiplication_l2375_237509


namespace committee_formation_count_l2375_237571

def total_students : ℕ := 8
def committee_size : ℕ := 4
def required_students : ℕ := 2
def remaining_students : ℕ := total_students - required_students

theorem committee_formation_count : 
  Nat.choose remaining_students (committee_size - required_students) = 15 := by
  sorry

end committee_formation_count_l2375_237571


namespace equation_solution_l2375_237584

theorem equation_solution : 
  ∀ x : ℝ, (2 / ((x - 1) * (x - 2)) + 2 / ((x - 2) * (x - 3)) + 2 / ((x - 3) * (x - 4)) = 1 / 3) ↔ 
  (x = 8 ∨ x = -5/2) := by
sorry

end equation_solution_l2375_237584
