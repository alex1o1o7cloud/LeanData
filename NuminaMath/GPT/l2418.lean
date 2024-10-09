import Mathlib

namespace unique_solution_exists_l2418_241816

theorem unique_solution_exists (k : ℚ) (h : k ≠ 0) : 
  (∀ x : ℚ, (x + 3) / (kx - 2) = x → x = -2) ↔ k = -3 / 4 := 
by
  sorry

end unique_solution_exists_l2418_241816


namespace evaluate_x_squared_plus_y_squared_l2418_241846

theorem evaluate_x_squared_plus_y_squared (x y : ℚ) (h1 : x + 2 * y = 20) (h2 : 3 * x + y = 19) : x^2 + y^2 = 401 / 5 :=
sorry

end evaluate_x_squared_plus_y_squared_l2418_241846


namespace smallest_N_l2418_241875

-- Definitions for conditions
variable (a b c : ℕ) (N : ℕ)

-- Define the conditions for the given problem
def valid_block (a b c : ℕ) : Prop :=
  (a - 1) * (b - 1) * (c - 1) = 252

def block_volume (a b c : ℕ) : ℕ := a * b * c

-- The target theorem to be proved
theorem smallest_N (h : valid_block a b c) : N = 224 :=
  sorry

end smallest_N_l2418_241875


namespace minimum_value_expression_l2418_241840

noncomputable def minimum_expression (a b c : ℝ) : ℝ :=
  a^2 + 6 * a * b + 9 * b^2 + 3 * c^2

theorem minimum_value_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  minimum_expression a b c ≥ 126 :=
by
  sorry

end minimum_value_expression_l2418_241840


namespace arithmetic_sequence_geometric_subsequence_l2418_241857

theorem arithmetic_sequence_geometric_subsequence :
  ∀ (a : ℕ → ℝ), (∀ n, a (n + 1) = a n + 2) ∧ (a 1 * a 3 = a 2 ^ 2) → a 2 = 4 :=
by
  intros a h
  sorry

end arithmetic_sequence_geometric_subsequence_l2418_241857


namespace CNY_share_correct_l2418_241850

noncomputable def total_NWF : ℝ := 1388.01
noncomputable def deductions_method1 : List ℝ := [41.89, 2.77, 478.48, 554.91, 0.24]
noncomputable def previous_year_share_CNY : ℝ := 17.77
noncomputable def deductions_method2 : List (ℝ × String) := [(3.02, "EUR"), (0.2, "USD"), (34.47, "GBP"), (39.98, "others"), (0.02, "other")]

theorem CNY_share_correct :
  let CNY22 := total_NWF - (deductions_method1.foldl (λ a b => a + b) 0)
  let alpha22_CNY := (CNY22 / total_NWF) * 100
  let method2_result := 100 - (deductions_method2.foldl (λ a b => a + b.1) 0)
  alpha22_CNY = 22.31 ∧ method2_result = 22.31 := 
sorry

end CNY_share_correct_l2418_241850


namespace part_1_part_2_l2418_241863

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part_1 (x : ℝ) : f x ≤ 4 ↔ x ∈ Set.Icc (-2 : ℝ) 2 :=
by sorry

theorem part_2 (b : ℝ) (h₁ : b ≠ 0) (x : ℝ) (h₂ : f x ≥ (|2 * b + 1| + |1 - b|) / |b|) : x ≤ -1.5 :=
by sorry

end part_1_part_2_l2418_241863


namespace proof_case_a_proof_case_b1_proof_case_b2_proof_case_c1_proof_case_c2_l2418_241880

structure CubeSymmetry where
  planes : Nat
  axes : Nat
  has_center : Bool

def general_cube_symmetry : CubeSymmetry :=
  { planes := 9, axes := 9, has_center := true }

def case_a : CubeSymmetry :=
  { planes := 4, axes := 1, has_center := false }

def case_b1 : CubeSymmetry :=
  { planes := 5, axes := 3, has_center := true }

def case_b2 : CubeSymmetry :=
  { planes := 2, axes := 1, has_center := false }

def case_c1 : CubeSymmetry :=
  { planes := 3, axes := 0, has_center := false }

def case_c2 : CubeSymmetry :=
  { planes := 2, axes := 1, has_center := false }

theorem proof_case_a : case_a = { planes := 4, axes := 1, has_center := false } := by
  sorry

theorem proof_case_b1 : case_b1 = { planes := 5, axes := 3, has_center := true } := by
  sorry

theorem proof_case_b2 : case_b2 = { planes := 2, axes := 1, has_center := false } := by
  sorry

theorem proof_case_c1 : case_c1 = { planes := 3, axes := 0, has_center := false } := by
  sorry

theorem proof_case_c2 : case_c2 = { planes := 2, axes := 1, has_center := false } := by
  sorry

end proof_case_a_proof_case_b1_proof_case_b2_proof_case_c1_proof_case_c2_l2418_241880


namespace num_valid_constants_m_l2418_241879

theorem num_valid_constants_m : 
  ∃ (m1 m2 : ℝ), 
  m1 ≠ m2 ∧ 
  (∃ (a b c d : ℝ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    (1 / 2) * abs (2 * c) * abs (2 * d) = 12 ∧ 
    (c / (2 * d) = 2 ∧ 8 = m1 ∨ 2 * c / d = 8) ∧ 
    (c / (2 * d) = (1 / 2) ∧ (1 / 2) = m2 ∨ 2 * c / d = 2)) ∧
  (∀ (m : ℝ), 
    (m = m1 ∨ m = m2) →
    ∃ (a b c d : ℝ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    (1 / 2) * abs (2 * c) * abs (2 * d) = 12 ∧ 
    (c / (2 * d) = 2 ∨ 2 * c / d = 8) ∧ 
    (c / (2 * d) = (1 / 2) ∨ 2 * c / d = 2)) :=
sorry

end num_valid_constants_m_l2418_241879


namespace not_all_less_than_two_l2418_241817

theorem not_all_less_than_two {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ¬(a + 1/b < 2 ∧ b + 1/c < 2 ∧ c + 1/a < 2) :=
sorry

end not_all_less_than_two_l2418_241817


namespace mul_example_l2418_241870

theorem mul_example : (3.6 * 0.5 = 1.8) := by
  sorry

end mul_example_l2418_241870


namespace order_numbers_l2418_241807

theorem order_numbers (a b c : ℕ) (h1 : a = 8^10) (h2 : b = 4^15) (h3 : c = 2^31) : b = a ∧ a < c :=
by {
  sorry
}

end order_numbers_l2418_241807


namespace maximum_and_minimum_values_l2418_241800

noncomputable def f (p q x : ℝ) : ℝ := x^3 - p * x^2 - q * x

theorem maximum_and_minimum_values
  (p q : ℝ)
  (h1 : f p q 1 = 0)
  (h2 : (deriv (f p q)) 1 = 0) :
  ∃ (max_val min_val : ℝ), max_val = 4 / 27 ∧ min_val = 0 := 
by {
  sorry
}

end maximum_and_minimum_values_l2418_241800


namespace inequality_solution_real_roots_range_l2418_241866

noncomputable def f (x : ℝ) : ℝ :=
|2 * x - 4| - |x - 3|

theorem inequality_solution :
  ∀ x, f x ≤ 2 → x ∈ Set.Icc (-1 : ℝ) 3 :=
sorry

theorem real_roots_range (k : ℝ) :
  (∃ x, f x = 0) → k ∈ Set.Icc (-1 : ℝ) 3 :=
sorry

end inequality_solution_real_roots_range_l2418_241866


namespace rectangle_area_l2418_241899

theorem rectangle_area (a : ℕ) (w l : ℕ) (h_square_area : a = 36) (h_square_side : w * w = a) (h_rectangle_length : l = 3 * w) : w * l = 108 :=
by
  -- Placeholder for proof
  sorry

end rectangle_area_l2418_241899


namespace side_length_uncovered_l2418_241835

theorem side_length_uncovered (L W : ℝ) (h₁ : L * W = 50) (h₂ : 2 * W + L = 25) : L = 20 :=
by {
  sorry
}

end side_length_uncovered_l2418_241835


namespace pet_purchase_ways_l2418_241808

-- Define the conditions
def number_of_puppies : Nat := 20
def number_of_kittens : Nat := 6
def number_of_hamsters : Nat := 8

def alice_choices : Nat := number_of_puppies

-- Define the problem statement in Lean
theorem pet_purchase_ways : 
  (number_of_puppies = 20) ∧ 
  (number_of_kittens = 6) ∧ 
  (number_of_hamsters = 8) → 
  (alice_choices * 2 * number_of_kittens * number_of_hamsters) = 1920 := 
by
  intros h
  sorry

end pet_purchase_ways_l2418_241808


namespace income_expenditure_ratio_l2418_241819

theorem income_expenditure_ratio (I E S : ℝ) (hI : I = 10000) (hS : S = 2000) (hEq : S = I - E) : I / E = 5 / 4 :=
by {
  sorry
}

end income_expenditure_ratio_l2418_241819


namespace percentage_in_quarters_l2418_241826

theorem percentage_in_quarters:
  let dimes : ℕ := 40
  let quarters : ℕ := 30
  let value_dimes : ℕ := dimes * 10
  let value_quarters : ℕ := quarters * 25
  let total_value : ℕ := value_dimes + value_quarters
  let percentage_quarters : ℚ := (value_quarters : ℚ) / total_value * 100
  percentage_quarters = 65.22 := sorry

end percentage_in_quarters_l2418_241826


namespace derivative_at_1_l2418_241814

def f (x : ℝ) : ℝ := x^3 + x^2 - 2 * x - 2

def f_derivative (x : ℝ) : ℝ := 3*x^2 + 2*x - 2

theorem derivative_at_1 : f_derivative 1 = 3 := by
  sorry

end derivative_at_1_l2418_241814


namespace perpendicular_condition_l2418_241821

theorem perpendicular_condition (a : ℝ) :
  (2 * a * x + (a - 1) * y + 2 = 0) ∧ ((a + 1) * x + 3 * a * y + 3 = 0) →
  (a = 1/5 ↔ ∃ x y: ℝ, ((- (2 * a / (a - 1))) * (-(a + 1) / (3 * a)) = -1)) :=
by
  sorry

end perpendicular_condition_l2418_241821


namespace original_price_of_saree_is_400_l2418_241843

-- Define the original price of the saree
variable (P : ℝ)

-- Define the sale price after successive discounts
def sale_price (P : ℝ) : ℝ := 0.80 * P * 0.95

-- We want to prove that the original price P is 400 given that the sale price is 304
theorem original_price_of_saree_is_400 (h : sale_price P = 304) : P = 400 :=
sorry

end original_price_of_saree_is_400_l2418_241843


namespace evaluate_expression_l2418_241824

theorem evaluate_expression : 
  ∀ (x y z : ℝ), 
  x = 2 → 
  y = -3 → 
  z = 1 → 
  x^2 + y^2 + z^2 + 2 * x * y - z^3 = 1 := by
  intros x y z hx hy hz
  rw [hx, hy, hz]
  sorry

end evaluate_expression_l2418_241824


namespace rounding_no_order_l2418_241894

theorem rounding_no_order (x : ℝ) (hx : x > 0) :
  let a := round (x * 100) / 100
  let b := round (x * 1000) / 1000
  let c := round (x * 10000) / 10000
  (¬((a ≥ b ∧ b ≥ c) ∨ (a ≤ b ∧ b ≤ c))) :=
sorry

end rounding_no_order_l2418_241894


namespace slope_of_tangent_line_at_x_2_l2418_241896

noncomputable def curve (x : ℝ) : ℝ := x^2 + 3*x

theorem slope_of_tangent_line_at_x_2 : (deriv curve 2) = 7 := by
  sorry

end slope_of_tangent_line_at_x_2_l2418_241896


namespace abc_eq_1_l2418_241855

theorem abc_eq_1 (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
(h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a)
(h7 : a + 1 / b^2 = b + 1 / c^2) (h8 : b + 1 / c^2 = c + 1 / a^2) :
  |a * b * c| = 1 :=
sorry

end abc_eq_1_l2418_241855


namespace rogers_coaches_l2418_241878

-- Define the structure for the problem conditions
structure snacks_problem :=
  (team_members : ℕ)
  (helpers : ℕ)
  (packs_purchased : ℕ)
  (pouches_per_pack : ℕ)

-- Create an instance of the problem with given conditions
def rogers_problem : snacks_problem :=
  { team_members := 13,
    helpers := 2,
    packs_purchased := 3,
    pouches_per_pack := 6 }

-- Define the theorem to state that given the conditions, the number of coaches is 3
theorem rogers_coaches (p : snacks_problem) : p.packs_purchased * p.pouches_per_pack - p.team_members - p.helpers = 3 :=
by
  sorry

end rogers_coaches_l2418_241878


namespace John_can_lift_now_l2418_241885

def originalWeight : ℕ := 135
def trainingIncrease : ℕ := 265
def bracerIncreaseFactor : ℕ := 6

def newWeight : ℕ := originalWeight + trainingIncrease
def bracerIncrease : ℕ := newWeight * bracerIncreaseFactor
def totalWeight : ℕ := newWeight + bracerIncrease

theorem John_can_lift_now :
  totalWeight = 2800 :=
by
  -- proof steps go here
  sorry

end John_can_lift_now_l2418_241885


namespace fraction_of_problems_solved_by_Andrey_l2418_241839

theorem fraction_of_problems_solved_by_Andrey (N x : ℕ) 
  (h1 : 0 < N) 
  (h2 : x = N / 2)
  (Boris_solves : ∀ y : ℕ, y = N - x → y / 3 = (N - x) / 3)
  (remaining_problems : ∀ y : ℕ, y = (N - x) - (N - x) / 3 → y = 2 * (N - x) / 3) 
  (Viktor_solves : (2 * (N - x) / 3 = N / 3)) :
  x / N = 1 / 2 := 
by {
  sorry
}

end fraction_of_problems_solved_by_Andrey_l2418_241839


namespace rectangular_prism_edge_properties_l2418_241844

-- Define a rectangular prism and the concept of parallel and perpendicular pairs of edges.
structure RectangularPrism :=
  (vertices : Fin 8 → Fin 3 → ℝ)
  -- Additional necessary conditions on the structure could be added here.

-- Define the number of parallel edges in a rectangular prism
def number_of_parallel_edge_pairs (rp : RectangularPrism) : ℕ :=
  -- Formula or logic to count parallel edge pairs.
  8 -- Placeholder for actual logic computation, based on problem conditions.

-- Define the number of perpendicular edges in a rectangular prism
def number_of_perpendicular_edge_pairs (rp : RectangularPrism) : ℕ :=
  -- Formula or logic to count perpendicular edge pairs.
  20 -- Placeholder for actual logic computation, based on problem conditions.

-- Theorem that asserts the requirement based on conditions
theorem rectangular_prism_edge_properties (rp : RectangularPrism) :
  number_of_parallel_edge_pairs rp = 8 ∧ number_of_perpendicular_edge_pairs rp = 20 :=
  by
    -- Placeholder proof that establishes the theorem
    sorry

end rectangular_prism_edge_properties_l2418_241844


namespace calculate_total_difference_in_miles_l2418_241860

def miles_bus_a : ℝ := 1.25
def miles_walk_1 : ℝ := 0.35
def miles_bus_b : ℝ := 2.68
def miles_walk_2 : ℝ := 0.47
def miles_bus_c : ℝ := 3.27
def miles_walk_3 : ℝ := 0.21

def total_miles_on_buses : ℝ := miles_bus_a + miles_bus_b + miles_bus_c
def total_miles_walked : ℝ := miles_walk_1 + miles_walk_2 + miles_walk_3
def total_difference_in_miles : ℝ := total_miles_on_buses - total_miles_walked

theorem calculate_total_difference_in_miles :
  total_difference_in_miles = 6.17 := by
  sorry

end calculate_total_difference_in_miles_l2418_241860


namespace sandy_bought_6_books_l2418_241871

variable (initialBooks soldBooks boughtBooks remainingBooks : ℕ)

def half (n : ℕ) : ℕ := n / 2

theorem sandy_bought_6_books :
  initialBooks = 14 →
  soldBooks = half initialBooks →
  remainingBooks = initialBooks - soldBooks →
  remainingBooks + boughtBooks = 13 →
  boughtBooks = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end sandy_bought_6_books_l2418_241871


namespace point_transformations_l2418_241813

theorem point_transformations (a b : ℝ) (h : (a ≠ 2 ∨ b ≠ 3))
  (H1 : ∃ x y : ℝ, (x, y) = (2 - (b - 3), 3 + (a - 2)) ∧ (y, x) = (-4, 2)) :
  b - a = -6 :=
by
  sorry

end point_transformations_l2418_241813


namespace negation_of_exists_x_lt_0_l2418_241872

theorem negation_of_exists_x_lt_0 :
  (¬ ∃ x : ℝ, x + |x| < 0) ↔ (∀ x : ℝ, x + |x| ≥ 0) :=
by {
  sorry
}

end negation_of_exists_x_lt_0_l2418_241872


namespace arithmetic_sum_eight_terms_l2418_241831

theorem arithmetic_sum_eight_terms :
  ∀ (a d : ℤ) (n : ℕ), a = -3 → d = 6 → n = 8 → 
  (last_term = a + (n - 1) * d) →
  (last_term = 39) →
  (sum = (n * (a + last_term)) / 2) →
  sum = 144 :=
by
  intros a d n ha hd hn hlast_term hlast_term_value hsum
  sorry

end arithmetic_sum_eight_terms_l2418_241831


namespace solve_for_x_l2418_241898

theorem solve_for_x (x : ℝ) : 64 = 4 * (16:ℝ)^(x - 2) → x = 3 :=
by 
  intro h
  sorry

end solve_for_x_l2418_241898


namespace watched_commercials_eq_100_l2418_241832

variable (x : ℕ) -- number of people who watched commercials
variable (s : ℕ := 27) -- number of subscribers
variable (rev_comm : ℝ := 0.50) -- revenue per commercial
variable (rev_sub : ℝ := 1.00) -- revenue per subscriber
variable (total_rev : ℝ := 77.00) -- total revenue

theorem watched_commercials_eq_100 (h : rev_comm * (x : ℝ) + rev_sub * (s : ℝ) = total_rev) : x = 100 := by
  sorry

end watched_commercials_eq_100_l2418_241832


namespace correlational_relationships_l2418_241858

-- Definitions of relationships
def learning_attitude_and_academic_performance := "The relationship between a student's learning attitude and their academic performance"
def teacher_quality_and_student_performance := "The relationship between a teacher's teaching quality and students' academic performance"
def student_height_and_academic_performance := "The relationship between a student's height and their academic performance"
def family_economic_conditions_and_performance := "The relationship between family economic conditions and students' academic performance"

-- Definition of a correlational relationship
def correlational_relationship (relation : String) : Prop :=
  relation = learning_attitude_and_academic_performance ∨
  relation = teacher_quality_and_student_performance

-- Problem statement to prove
theorem correlational_relationships :
  correlational_relationship learning_attitude_and_academic_performance ∧ 
  correlational_relationship teacher_quality_and_student_performance :=
by
  -- Placeholder to indicate the proof is omitted
  sorry

end correlational_relationships_l2418_241858


namespace range_of_m_l2418_241883

noncomputable def f (x : ℝ) : ℝ := |x - 3| - 2
noncomputable def g (x : ℝ) : ℝ := -|x + 1| + 4

theorem range_of_m :
  (∀ x : ℝ, f x - g x ≥ m + 1) ↔ m ≤ -3 :=
by sorry

end range_of_m_l2418_241883


namespace exist_indices_with_non_decreasing_subsequences_l2418_241852

theorem exist_indices_with_non_decreasing_subsequences
  (a b c : ℕ → ℕ) :
  (∀ n m : ℕ, n < m → ∃ p q : ℕ, q < p ∧ 
    a p ≥ a q ∧ 
    b p ≥ b q ∧ 
    c p ≥ c q) :=
  sorry

end exist_indices_with_non_decreasing_subsequences_l2418_241852


namespace can_place_more_domino_domino_placement_possible_l2418_241874

theorem can_place_more_domino (total_squares : ℕ := 36) (uncovered_squares : ℕ := 14) : Prop :=
∃ (n : ℕ), (n * 2 + uncovered_squares ≤ total_squares) ∧ (n ≥ 1)

/-- Proof that on a 6x6 chessboard with some 1x2 dominoes placed, if there are 14 uncovered
squares, then at least one more domino can be placed on the board. -/
theorem domino_placement_possible :
  can_place_more_domino := by
  sorry

end can_place_more_domino_domino_placement_possible_l2418_241874


namespace parabola_y_range_l2418_241869

theorem parabola_y_range
  (x y : ℝ)
  (M_on_C : x^2 = 8 * y)
  (F : ℝ × ℝ)
  (F_focus : F = (0, 2))
  (circle_intersects_directrix : F.2 + y > 4) :
  y > 2 :=
by
  sorry

end parabola_y_range_l2418_241869


namespace original_students_l2418_241811

theorem original_students (a b : ℕ) : 
  a + b = 92 ∧ a - 5 = 3 * (b + 5 - 32) → a = 45 ∧ b = 47 :=
by sorry

end original_students_l2418_241811


namespace class_average_correct_l2418_241810

def class_average_test_A : ℝ :=
  0.30 * 97 + 0.25 * 85 + 0.20 * 78 + 0.15 * 65 + 0.10 * 55

def class_average_test_B : ℝ :=
  0.30 * 93 + 0.25 * 80 + 0.20 * 75 + 0.15 * 70 + 0.10 * 60

theorem class_average_correct :
  round class_average_test_A = 81 ∧
  round class_average_test_B = 79 := 
by 
  sorry

end class_average_correct_l2418_241810


namespace min_value_proof_l2418_241887

theorem min_value_proof (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x - 2 * y + 3 * z = 0) : 3 = 3 :=
by
  sorry

end min_value_proof_l2418_241887


namespace sqrt_neg_squared_eq_two_l2418_241891

theorem sqrt_neg_squared_eq_two : (-Real.sqrt 2) ^ 2 = 2 := by
  sorry

end sqrt_neg_squared_eq_two_l2418_241891


namespace find_a_l2418_241818

theorem find_a (k a : ℚ) (hk : 4 * k = 60) (ha : 15 * a - 5 = 60) : a = 13 / 3 :=
by
  sorry

end find_a_l2418_241818


namespace count_decorations_l2418_241812

/--
Define a function T(n) that determines the number of ways to decorate the window 
with n stripes according to the given conditions.
--/
def T : ℕ → ℕ
| 0       => 1 -- optional case for completeness
| 1       => 2
| 2       => 2
| (n + 1) => T n + T (n - 1)

theorem count_decorations : T 10 = 110 := by
  sorry

end count_decorations_l2418_241812


namespace problem_equiv_math_problem_l2418_241842
-- Lean Statement for the proof problem

variable {x y z : ℝ}

theorem problem_equiv_math_problem (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (eq1 : x^2 + x * y + y^2 / 3 = 25) 
  (eq2 : y^2 / 3 + z^2 = 9) 
  (eq3 : z^2 + z * x + x^2 = 16) :
  x * y + 2 * y * z + 3 * z * x = 24 * Real.sqrt 3 :=
by
  sorry

end problem_equiv_math_problem_l2418_241842


namespace line_through_point_parallel_l2418_241849

theorem line_through_point_parallel (x y : ℝ) : 
  (∃ c : ℝ, x - 2 * y + c = 0 ∧ ∃ p : ℝ × ℝ, p = (1, 0) ∧ x - 2 * p.2 + c = 0) → (x - 2 * y - 1 = 0) :=
by
  sorry

end line_through_point_parallel_l2418_241849


namespace abs_diff_squares_110_108_l2418_241881

theorem abs_diff_squares_110_108 : abs ((110 : ℤ)^2 - (108 : ℤ)^2) = 436 := by
  sorry

end abs_diff_squares_110_108_l2418_241881


namespace factor_expression_l2418_241877

theorem factor_expression (x : ℝ) : 
  x^2 * (x + 3) + 2 * x * (x + 3) + (x + 3) = (x + 1)^2 * (x + 3) := by
  sorry

end factor_expression_l2418_241877


namespace prism_volume_eq_400_l2418_241854

noncomputable def prism_volume (a b c : ℝ) : ℝ := a * b * c

theorem prism_volume_eq_400 
  (a b c : ℝ)
  (h1 : a * b = 40)
  (h2 : a * c = 50)
  (h3 : b * c = 80) :
  prism_volume a b c = 400 :=
by
  sorry

end prism_volume_eq_400_l2418_241854


namespace num_ordered_pairs_xy_eq_2200_l2418_241851

/-- There are 24 ordered pairs (x, y) such that xy = 2200. -/
theorem num_ordered_pairs_xy_eq_2200 : 
  ∃ (n : ℕ), n = 24 ∧ (∃ divisors : Finset ℕ, 
    (∀ d ∈ divisors, 2200 % d = 0) ∧ 
    (divisors.card = 24)) := 
sorry

end num_ordered_pairs_xy_eq_2200_l2418_241851


namespace rahim_sequence_final_value_l2418_241828

theorem rahim_sequence_final_value :
  ∃ (a : ℕ) (b : ℕ), a ^ b = 5 ^ 16 :=
sorry

end rahim_sequence_final_value_l2418_241828


namespace men_took_dip_l2418_241897

theorem men_took_dip 
  (tank_length : ℝ) (tank_breadth : ℝ) (water_rise_cm : ℝ) (man_displacement : ℝ)
  (H1 : tank_length = 40) (H2 : tank_breadth = 20) (H3 : water_rise_cm = 25) (H4 : man_displacement = 4) :
  let water_rise_m := water_rise_cm / 100
  let total_volume_displaced := tank_length * tank_breadth * water_rise_m
  let number_of_men := total_volume_displaced / man_displacement
  number_of_men = 50 :=
by
  sorry

end men_took_dip_l2418_241897


namespace cross_square_side_length_l2418_241809

theorem cross_square_side_length (A : ℝ) (s : ℝ) (h1 : A = 810) 
(h2 : (2 * (s / 2)^2 + 2 * (s / 4)^2) = A) : s = 36 := by
  sorry

end cross_square_side_length_l2418_241809


namespace total_lunch_cost_l2418_241865

theorem total_lunch_cost
  (children chaperones herself additional_lunches cost_per_lunch : ℕ)
  (h1 : children = 35)
  (h2 : chaperones = 5)
  (h3 : herself = 1)
  (h4 : additional_lunches = 3)
  (h5 : cost_per_lunch = 7) :
  (children + chaperones + herself + additional_lunches) * cost_per_lunch = 308 :=
by
  sorry

end total_lunch_cost_l2418_241865


namespace power_of_point_l2418_241859

namespace ChordsIntersect

variables (A B C D P : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited P]

def AP := 4
def CP := 9

theorem power_of_point (BP DP : ℕ) :
  AP * BP = CP * DP -> (BP / DP) = 9 / 4 :=
by
  sorry

end ChordsIntersect

end power_of_point_l2418_241859


namespace grandpa_movie_time_l2418_241864

theorem grandpa_movie_time
  (each_movie_time : ℕ := 90)
  (max_movies_2_days : ℕ := 9)
  (x_movies_tuesday : ℕ)
  (movies_wednesday := 2 * x_movies_tuesday)
  (total_movies := x_movies_tuesday + movies_wednesday)
  (h : total_movies = max_movies_2_days) :
  90 * x_movies_tuesday = 270 :=
by
  sorry

end grandpa_movie_time_l2418_241864


namespace ordinate_of_point_A_l2418_241837

noncomputable def p : ℝ := 1 / 4
noncomputable def distance_to_focus (y₀ : ℝ) : ℝ := y₀ + p / 2

theorem ordinate_of_point_A :
  ∃ y₀ : ℝ, (distance_to_focus y₀ = 9 / 8) → y₀ = 1 :=
by
  -- Assume solution steps here
  sorry

end ordinate_of_point_A_l2418_241837


namespace find_k_l2418_241829

def system_of_equations (x y k : ℝ) : Prop :=
  x - y = k - 3 ∧
  3 * x + 5 * y = 2 * k + 8 ∧
  x + y = 2

theorem find_k (x y k : ℝ) (h : system_of_equations x y k) : k = 1 := 
sorry

end find_k_l2418_241829


namespace smallest_sum_B_c_l2418_241890

theorem smallest_sum_B_c (B : ℕ) (c : ℕ) (hB : B < 5) (hc : c > 6) :
  31 * B = 4 * c + 4 → (B + c) = 34 :=
by
  sorry

end smallest_sum_B_c_l2418_241890


namespace range_of_a_l2418_241827

theorem range_of_a (x a : ℝ) : (∃ x : ℝ,  |x + 2| + |x - 3| ≤ |a - 1| ) ↔ (a ≤ -4 ∨ a ≥ 6) :=
by
  sorry

end range_of_a_l2418_241827


namespace Juan_birth_year_proof_l2418_241801

-- Let BTC_year(n) be the year of the nth BTC competition.
def BTC_year (n : ℕ) : ℕ :=
  1990 + (n - 1) * 2

-- Juan's birth year given his age and the BTC he participated in.
def Juan_birth_year (current_year : ℕ) (age : ℕ) : ℕ :=
  current_year - age

-- Main proof problem statement.
theorem Juan_birth_year_proof :
  (BTC_year 5 = 1998) →
  (Juan_birth_year 1998 14 = 1984) :=
by
  intros
  sorry

end Juan_birth_year_proof_l2418_241801


namespace num_integers_satisfy_l2418_241820

theorem num_integers_satisfy : 
  ∃ n : ℕ, (n = 7 ∧ ∀ k : ℤ, (k > -5 ∧ k < 3) → (k = -4 ∨ k = -3 ∨ k = -2 ∨ k = -1 ∨ k = 0 ∨ k = 1 ∨ k = 2)) := 
sorry

end num_integers_satisfy_l2418_241820


namespace arrangement_possible_32_arrangement_possible_100_l2418_241823

-- Problem (1)
theorem arrangement_possible_32 : 
  ∃ (f : Fin 32 → Fin 32), ∀ (a b : Fin 32), ∀ (i : Fin 32), 
    a < b → i < b → f i = (a + b) / 2 → False := 
sorry

-- Problem (2)
theorem arrangement_possible_100 : 
  ∃ (f : Fin 100 → Fin 100), ∀ (a b : Fin 100), ∀ (i : Fin 100),
    a < b → i < b → f i = (a + b) / 2 → False := 
sorry


end arrangement_possible_32_arrangement_possible_100_l2418_241823


namespace triangle_sine_inequality_l2418_241893

theorem triangle_sine_inequality (A B C : Real) (h : A + B + C = Real.pi) :
  Real.sin (A / 2) + Real.sin (B / 2) + Real.sin (C / 2) ≤
  1 + (1 / 2) * Real.cos ((A - B) / 4) ^ 2 :=
by
  sorry

end triangle_sine_inequality_l2418_241893


namespace hyperbola_focal_length_l2418_241834

-- Define the constants a^2 and b^2 based on the given hyperbola equation.
def a_squared : ℝ := 16
def b_squared : ℝ := 25

-- Define the constants a and b as the square roots of a^2 and b^2.
noncomputable def a : ℝ := Real.sqrt a_squared
noncomputable def b : ℝ := Real.sqrt b_squared

-- Define the constant c based on the relation c^2 = a^2 + b^2.
noncomputable def c : ℝ := Real.sqrt (a_squared + b_squared)

-- The focal length of the hyperbola is 2c.
noncomputable def focal_length : ℝ := 2 * c

-- The theorem that captures the statement of the problem.
theorem hyperbola_focal_length : focal_length = 2 * Real.sqrt 41 := by
  -- Proof omitted.
  sorry

end hyperbola_focal_length_l2418_241834


namespace number_of_paths_l2418_241895

-- Definition of vertices
inductive Vertex
| A | B | C | D | E | F | G

-- Edges based on the description
def edges : List (Vertex × Vertex) := [
  (Vertex.A, Vertex.G), (Vertex.G, Vertex.C), (Vertex.G, Vertex.D), (Vertex.C, Vertex.B),
  (Vertex.D, Vertex.C), (Vertex.D, Vertex.F), (Vertex.D, Vertex.E), (Vertex.E, Vertex.F),
  (Vertex.F, Vertex.B), (Vertex.C, Vertex.F), (Vertex.A, Vertex.C), (Vertex.A, Vertex.D)
]

-- Function to count paths from A to B without revisiting any vertex
def countPaths (start : Vertex) (goal : Vertex) (adj : List (Vertex × Vertex)) : Nat :=
sorry

-- The theorem statement
theorem number_of_paths : countPaths Vertex.A Vertex.B edges = 10 :=
sorry

end number_of_paths_l2418_241895


namespace geometric_series_sum_eq_4_div_3_l2418_241806

theorem geometric_series_sum_eq_4_div_3 (a : ℝ) (r : ℝ) (h₀ : a = 1) (h₁ : r = 1 / 4) :
  ∑' n : ℕ, a * r^n = 4 / 3 := by
  sorry

end geometric_series_sum_eq_4_div_3_l2418_241806


namespace organization_members_count_l2418_241892

theorem organization_members_count (num_committees : ℕ) (pair_membership : ℕ → ℕ → ℕ) :
  num_committees = 5 →
  (∀ i j k l : ℕ, i ≠ j → k ≠ l → pair_membership i j = pair_membership k l → i = k ∧ j = l ∨ i = l ∧ j = k) →
  ∃ (num_members : ℕ), num_members = 10 :=
by
  sorry

end organization_members_count_l2418_241892


namespace Carson_skipped_times_l2418_241802

variable (length width total_circles actual_distance perimeter distance_skipped : ℕ)
variable (total_distance : ℕ)

def perimeter_calculation (length width : ℕ) : ℕ := 2 * (length + width)

def total_distance_calculation (total_circles perimeter : ℕ) : ℕ := total_circles * perimeter

def distance_skipped_calculation (total_distance actual_distance : ℕ) : ℕ := total_distance - actual_distance

def times_skipped_calculation (distance_skipped perimeter : ℕ) : ℕ := distance_skipped / perimeter

theorem Carson_skipped_times (h_length : length = 600) 
                             (h_width : width = 400) 
                             (h_total_circles : total_circles = 10) 
                             (h_actual_distance : actual_distance = 16000) 
                             (h_perimeter : perimeter = perimeter_calculation length width) 
                             (h_total_distance : total_distance = total_distance_calculation total_circles perimeter) 
                             (h_distance_skipped : distance_skipped = distance_skipped_calculation total_distance actual_distance) :
                             times_skipped_calculation distance_skipped perimeter = 2 := 
by
  simp [perimeter_calculation, total_distance_calculation, distance_skipped_calculation, times_skipped_calculation]
  sorry

end Carson_skipped_times_l2418_241802


namespace probability_of_winning_fifth_game_championship_correct_overall_probability_of_winning_championship_correct_l2418_241836

noncomputable def binomial (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ)

noncomputable def probability_of_winning_fifth_game_championship : ℝ :=
  binomial 4 3 * 0.6^4 * 0.4

noncomputable def overall_probability_of_winning_championship : ℝ :=
  0.6^4 +
  binomial 4 3 * 0.6^4 * 0.4 +
  binomial 5 3 * 0.6^4 * 0.4^2 +
  binomial 6 3 * 0.6^4 * 0.4^3

theorem probability_of_winning_fifth_game_championship_correct :
  probability_of_winning_fifth_game_championship = 0.20736 := by
  sorry

theorem overall_probability_of_winning_championship_correct :
  overall_probability_of_winning_championship = 0.710208 := by
  sorry

end probability_of_winning_fifth_game_championship_correct_overall_probability_of_winning_championship_correct_l2418_241836


namespace distance_between_bakery_and_butcher_shop_l2418_241805

variables (v1 v2 : ℝ) -- speeds of the butcher's and baker's son respectively
variables (x : ℝ) -- distance covered by the baker's son by the time they meet
variable (distance : ℝ) -- distance between the bakery and the butcher shop

-- Given conditions
def butcher_walks_500_more := x + 0.5
def butcher_time_left := 10 / 60
def baker_time_left := 22.5 / 60

-- Equivalent relationships
def v1_def := v1 = 6 * x
def v2_def := v2 = (8/3) * (x + 0.5)

-- Final proof problem
theorem distance_between_bakery_and_butcher_shop :
  (x + 0.5 + x) = 2.5 :=
sorry

end distance_between_bakery_and_butcher_shop_l2418_241805


namespace find_second_remainder_l2418_241838

theorem find_second_remainder (k m n r : ℕ) 
  (h1 : n = 12 * k + 56) 
  (h2 : n = 34 * m + r) 
  (h3 : (22 + r) % 12 = 10) : 
  r = 10 :=
sorry

end find_second_remainder_l2418_241838


namespace series_sum_l2418_241825

noncomputable def series_term (n : ℕ) : ℝ :=
  (4 * n + 2) / ((6 * n - 5)^2 * (6 * n + 1)^2)

theorem series_sum :
  (∑' n : ℕ, series_term (n + 1)) = 1 / 6 :=
by
  sorry

end series_sum_l2418_241825


namespace value_of_a_l2418_241886

theorem value_of_a (a : ℝ) (h_neg : a < 0) (h_f : ∀ (x : ℝ), (0 < x ∧ x ≤ 1) → 
  (x + 4 * a / x - a < 0)) : a ≤ -1 / 3 := 
sorry

end value_of_a_l2418_241886


namespace scale_reading_l2418_241847

theorem scale_reading (x : ℝ) (h₁ : 3.25 < x) (h₂ : x < 3.5) : x = 3.3 :=
sorry

end scale_reading_l2418_241847


namespace circle_equation_l2418_241861

theorem circle_equation {a b c : ℝ} (hc : c ≠ 0) :
  ∃ D E F : ℝ, 
    (D = -(a + b)) ∧
    (E = - (c + ab / c)) ∧ 
    (F = ab) ∧
    ∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 :=
sorry

end circle_equation_l2418_241861


namespace measure_of_angle_l2418_241848

theorem measure_of_angle (x : ℝ) (h1 : 90 = x + (3 * x + 10)) : x = 20 :=
by
  sorry

end measure_of_angle_l2418_241848


namespace smoking_lung_cancer_problem_l2418_241830

-- Defining the confidence relationship
def smoking_related_to_lung_cancer (confidence: ℝ) := confidence > 0.99

-- Statement 4: Among 100 smokers, it is possible that not a single person has lung cancer.
def statement_4 (N: ℕ) (p: ℝ) := N = 100 ∧ 0 ≤ p ∧ p ≤ 1 ∧ p ^ 100 > 0

-- The main theorem statement in Lean 4
theorem smoking_lung_cancer_problem (confidence: ℝ) (N: ℕ) (p: ℝ) 
  (h1: smoking_related_to_lung_cancer confidence): 
  statement_4 N p :=
by
  sorry -- Proof goes here

end smoking_lung_cancer_problem_l2418_241830


namespace santana_brothers_birthday_l2418_241841

theorem santana_brothers_birthday (b : ℕ) (oct : ℕ) (nov : ℕ) (dec : ℕ) (c_presents_diff : ℕ) :
  b = 7 → oct = 1 → nov = 1 → dec = 2 → c_presents_diff = 8 → (∃ M : ℕ, M = 3) :=
by
  sorry

end santana_brothers_birthday_l2418_241841


namespace min_value_expr_l2418_241815

theorem min_value_expr (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h_xyz : x * y * z = 1) : 
  x^2 + 4 * x * y + 9 * y^2 + 8 * y * z + 3 * z^2 ≥ 9^(10/9) :=
sorry

end min_value_expr_l2418_241815


namespace balls_left_correct_l2418_241884

def initial_balls : ℕ := 10
def balls_removed : ℕ := 3
def balls_left : ℕ := initial_balls - balls_removed

theorem balls_left_correct : balls_left = 7 := 
by
  -- Proof omitted
  sorry

end balls_left_correct_l2418_241884


namespace sprinted_further_than_jogged_l2418_241822

def sprint_distance1 := 0.8932
def sprint_distance2 := 0.7773
def sprint_distance3 := 0.9539
def sprint_distance4 := 0.5417
def sprint_distance5 := 0.6843

def jog_distance1 := 0.7683
def jog_distance2 := 0.4231
def jog_distance3 := 0.5733
def jog_distance4 := 0.625
def jog_distance5 := 0.6549

def total_sprint_distance := sprint_distance1 + sprint_distance2 + sprint_distance3 + sprint_distance4 + sprint_distance5
def total_jog_distance := jog_distance1 + jog_distance2 + jog_distance3 + jog_distance4 + jog_distance5

theorem sprinted_further_than_jogged :
  total_sprint_distance - total_jog_distance = 0.8058 :=
by
  sorry

end sprinted_further_than_jogged_l2418_241822


namespace faye_gave_away_books_l2418_241888

theorem faye_gave_away_books (x : ℕ) (H1 : 34 - x + 48 = 79) : x = 3 :=
by {
  sorry
}

end faye_gave_away_books_l2418_241888


namespace probability_sunflower_seed_l2418_241833

theorem probability_sunflower_seed :
  ∀ (sunflower_seeds green_bean_seeds pumpkin_seeds : ℕ),
  sunflower_seeds = 2 →
  green_bean_seeds = 3 →
  pumpkin_seeds = 4 →
  (sunflower_seeds + green_bean_seeds + pumpkin_seeds = 9) →
  (sunflower_seeds : ℚ) / (sunflower_seeds + green_bean_seeds + pumpkin_seeds) = 2 / 9 := 
by 
  intros sunflower_seeds green_bean_seeds pumpkin_seeds h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  rw [h1, h2, h3]
  sorry -- Proof omitted as per instructions.

end probability_sunflower_seed_l2418_241833


namespace unique_fraction_satisfying_condition_l2418_241876

theorem unique_fraction_satisfying_condition : ∃! (x y : ℕ), Nat.gcd x y = 1 ∧ y ≠ 0 ∧ (x + 1) * 5 * y = (y + 1) * 6 * x :=
by
  sorry

end unique_fraction_satisfying_condition_l2418_241876


namespace problem1_problem2_l2418_241845

theorem problem1 (x : ℝ) : (x + 4) ^ 2 - 5 * (x + 4) = 0 → x = -4 ∨ x = 1 :=
by
  sorry

theorem problem2 (x : ℝ) : x ^ 2 - 2 * x - 15 = 0 → x = -3 ∨ x = 5 :=
by
  sorry

end problem1_problem2_l2418_241845


namespace compute_b_l2418_241868

noncomputable def rational_coefficients (a b : ℚ) :=
∃ x : ℚ, (x^3 + a * x^2 + b * x + 15 = 0)

theorem compute_b (a b : ℚ) (h1 : (3 + Real.sqrt 5)∈{root : ℝ | root^3 + a * root^2 + b * root + 15 = 0}) 
(h2 : rational_coefficients a b) : b = -18.5 :=
by
  sorry

end compute_b_l2418_241868


namespace karl_total_miles_l2418_241889

def car_mileage_per_gallon : ℕ := 30
def full_tank_gallons : ℕ := 14
def initial_drive_miles : ℕ := 300
def gas_bought_gallons : ℕ := 10
def final_tank_fraction : ℚ := 1 / 3

theorem karl_total_miles (initial_fuel : ℕ) :
  initial_fuel = full_tank_gallons →
  (initial_drive_miles / car_mileage_per_gallon + gas_bought_gallons) = initial_fuel - (initial_fuel * final_tank_fraction) / car_mileage_per_gallon + (580 - initial_drive_miles) / car_mileage_per_gallon →
  initial_drive_miles + (initial_fuel - initial_drive_miles / car_mileage_per_gallon + gas_bought_gallons - initial_fuel * final_tank_fraction / car_mileage_per_gallon) * car_mileage_per_gallon = 580 := 
sorry

end karl_total_miles_l2418_241889


namespace rented_movie_cost_l2418_241853

def cost_of_tickets (c_ticket : ℝ) (n_tickets : ℕ) := c_ticket * n_tickets
def total_cost (cost_tickets cost_bought : ℝ) := cost_tickets + cost_bought
def remaining_cost (total_spent cost_so_far : ℝ) := total_spent - cost_so_far

theorem rented_movie_cost
  (c_ticket : ℝ)
  (n_tickets : ℕ)
  (c_bought : ℝ)
  (c_total : ℝ)
  (h1 : c_ticket = 10.62)
  (h2 : n_tickets = 2)
  (h3 : c_bought = 13.95)
  (h4 : c_total = 36.78) :
  remaining_cost c_total (total_cost (cost_of_tickets c_ticket n_tickets) c_bought) = 1.59 :=
by 
  sorry

end rented_movie_cost_l2418_241853


namespace hundred_days_from_friday_is_sunday_l2418_241803

/-- Given that today is Friday, determine that 100 days from now is Sunday. -/
theorem hundred_days_from_friday_is_sunday (today : ℕ) (days_in_week : ℕ := 7) 
(friday : ℕ := 0) (sunday : ℕ := 2) : (((today + 100) % days_in_week) = sunday) :=
sorry

end hundred_days_from_friday_is_sunday_l2418_241803


namespace quiz_answer_key_count_l2418_241856

theorem quiz_answer_key_count :
  ∃ n : ℕ, n = 480 ∧
  (∃ tf_count : ℕ, tf_count = 30 ∧
   (∃ mc_count : ℕ, mc_count = 16 ∧ 
    n = tf_count * mc_count)) :=
    sorry

end quiz_answer_key_count_l2418_241856


namespace water_distribution_scheme_l2418_241867

theorem water_distribution_scheme (a b c : ℚ) : 
  a + b + c = 1 ∧ 
  (∀ x : ℂ, ∃ n : ℕ, x^n = 1 → x = 1) ∧
  (∀ (x : ℂ), (1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 + x^11 + x^12 + x^13 + x^14 + x^15 + x^16 + x^17 + x^18 + x^19 + x^20 + x^21 + x^22 = 0) → false) → 
  a = 0 ∧ b = 0 ∧ c = 1 :=
by
  sorry

end water_distribution_scheme_l2418_241867


namespace solution_set_of_inequality_l2418_241873

theorem solution_set_of_inequality :
  {x : ℝ | x * (x - 1) * (x - 2) > 0} = {x | (0 < x ∧ x < 1) ∨ x > 2} :=
by sorry

end solution_set_of_inequality_l2418_241873


namespace petya_friends_l2418_241882

variable (x : ℕ) -- Define x to be a natural number (number of friends)
variable (S : ℕ) -- Define S to be a natural number (total number of stickers Petya has)

-- Conditions from the problem
axiom condition1 : S = 5 * x + 8 -- If Petya gives 5 stickers to each friend, 8 stickers are left
axiom condition2 : S = 6 * x - 11 -- If Petya gives 6 stickers to each friend, he is short 11 stickers

theorem petya_friends : x = 19 :=
by
  -- Proof goes here
  sorry

end petya_friends_l2418_241882


namespace absolute_value_inequality_range_of_xyz_l2418_241862

-- Question 1 restated
theorem absolute_value_inequality (x : ℝ) :
  (|x + 2| + |x + 3| ≤ 2) ↔ -7/2 ≤ x ∧ x ≤ -3/2 :=
sorry

-- Question 2 restated
theorem range_of_xyz (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  -1/2 ≤ x * y + y * z + z * x ∧ x * y + y * z + z * x ≤ 1 :=
sorry

end absolute_value_inequality_range_of_xyz_l2418_241862


namespace largest_expr_is_a_squared_plus_b_squared_l2418_241804

noncomputable def largest_expression (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) (h₃ : a ≠ b) : Prop :=
  (a^2 + b^2 > a - b) ∧ (a^2 + b^2 > a + b) ∧ (a^2 + b^2 > 2 * a * b)

theorem largest_expr_is_a_squared_plus_b_squared (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) (h₃ : a ≠ b) : 
  largest_expression a b h₁ h₂ h₃ :=
by
  sorry

end largest_expr_is_a_squared_plus_b_squared_l2418_241804
