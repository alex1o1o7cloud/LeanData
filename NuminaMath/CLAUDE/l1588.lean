import Mathlib

namespace N_eq_P_l1588_158846

def N : Set ℚ := {x | ∃ n : ℤ, x = n / 2 - 1 / 3}
def P : Set ℚ := {x | ∃ p : ℤ, x = p / 2 + 1 / 6}

theorem N_eq_P : N = P := by sorry

end N_eq_P_l1588_158846


namespace quadratic_roots_l1588_158802

theorem quadratic_roots : 
  let f : ℝ → ℝ := λ x => x^2 - 2*x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end quadratic_roots_l1588_158802


namespace geometric_sequence_sum_l1588_158822

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- a_n is a geometric sequence with common ratio q
  a 1 + a 2 = 3 →               -- a_1 + a_2 = 3
  a 3 + a 4 = 6 →               -- a_3 + a_4 = 6
  a 7 + a 8 = 24 :=             -- a_7 + a_8 = 24
by
  sorry

end geometric_sequence_sum_l1588_158822


namespace isosceles_triangle_on_cube_l1588_158810

-- Define a cube
def Cube : Type := Unit

-- Define a function to count the number of ways to choose 3 vertices from 8
def choose_3_from_8 : ℕ := 56

-- Define the number of isosceles triangles that can be formed on the cube
def isosceles_triangles_count : ℕ := 32

-- Define the probability of forming an isosceles triangle
def isosceles_triangle_probability : ℚ := 4/7

-- Theorem statement
theorem isosceles_triangle_on_cube :
  (isosceles_triangles_count : ℚ) / choose_3_from_8 = isosceles_triangle_probability :=
sorry

end isosceles_triangle_on_cube_l1588_158810


namespace min_n_with_three_same_color_l1588_158896

/-- A coloring of an n × n grid using three colors. -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin 3

/-- Checks if a coloring satisfies the condition of having at least three squares
    of the same color in a row or column. -/
def satisfiesCondition (n : ℕ) (c : Coloring n) : Prop :=
  ∃ (i : Fin n) (color : Fin 3),
    (∃ (j₁ j₂ j₃ : Fin n), j₁ ≠ j₂ ∧ j₁ ≠ j₃ ∧ j₂ ≠ j₃ ∧
      c i j₁ = color ∧ c i j₂ = color ∧ c i j₃ = color) ∨
    (∃ (i₁ i₂ i₃ : Fin n), i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₂ ≠ i₃ ∧
      c i₁ i = color ∧ c i₂ i = color ∧ c i₃ i = color)

/-- The main theorem stating that 7 is the smallest n that satisfies the condition. -/
theorem min_n_with_three_same_color :
  (∀ (c : Coloring 7), satisfiesCondition 7 c) ∧
  (∀ (n : ℕ), n < 7 → ∃ (c : Coloring n), ¬satisfiesCondition n c) :=
sorry

end min_n_with_three_same_color_l1588_158896


namespace binomial_expansion_coeff_l1588_158830

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of x^3 in the expansion of (x^2 - m/x)^6 -/
def coeff_x3 (m : ℝ) : ℝ := (-1)^3 * binomial 6 3 * m^3

theorem binomial_expansion_coeff (m : ℝ) :
  coeff_x3 m = -160 → m = 2 := by sorry

end binomial_expansion_coeff_l1588_158830


namespace simplify_expression_l1588_158828

theorem simplify_expression :
  ∀ x : ℝ, x > 0 →
  (3 * Real.sqrt 10) / (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7) =
  Real.sqrt 3 + Real.sqrt 4 - Real.sqrt 7 :=
by sorry

end simplify_expression_l1588_158828


namespace correct_second_number_l1588_158848

/-- Proves that the correct value of the second wrongly copied number is 27 --/
theorem correct_second_number (n : ℕ) (original_avg correct_avg : ℚ) 
  (first_error second_error : ℚ) (h1 : n = 10) (h2 : original_avg = 40.2) 
  (h3 : correct_avg = 40) (h4 : first_error = 16) (h5 : second_error = 13) : 
  ∃ (x : ℚ), n * correct_avg = n * original_avg - first_error - second_error + x ∧ x = 27 := by
  sorry

end correct_second_number_l1588_158848


namespace equidistant_points_l1588_158870

/-- Two points are equidistant if the larger of their distances to the x and y axes are equal -/
def equidistant (p q : ℝ × ℝ) : Prop :=
  max (|p.1|) (|p.2|) = max (|q.1|) (|q.2|)

theorem equidistant_points :
  (equidistant (-3, 7) (3, -7) ∧ equidistant (-3, 7) (7, 4)) ∧
  (equidistant (-4, 2) (-4, -3) ∧ equidistant (-4, 2) (3, 4)) ∧
  (equidistant (3, 4 + 2) (2 * 2 - 5, 6) ∧ equidistant (3, 4 + 9) (2 * 9 - 5, 6)) :=
by sorry

end equidistant_points_l1588_158870


namespace hyperbola_vertices_distance_l1588_158824

theorem hyperbola_vertices_distance (x y : ℝ) :
  (((x - 1)^2 / 16) - (y^2 / 25) = 1) →
  (∃ v₁ v₂ : ℝ, v₁ ≠ v₂ ∧ 
    (((v₁ - 1)^2 / 16) - (0^2 / 25) = 1) ∧
    (((v₂ - 1)^2 / 16) - (0^2 / 25) = 1) ∧
    |v₁ - v₂| = 8) :=
by
  sorry

end hyperbola_vertices_distance_l1588_158824


namespace four_dice_same_face_probability_l1588_158878

/-- The number of sides on a standard die -/
def numSides : ℕ := 6

/-- The number of dice being tossed -/
def numDice : ℕ := 4

/-- The probability of a specific outcome on a single die -/
def singleDieProbability : ℚ := 1 / numSides

/-- The probability of all dice showing the same number -/
def allSameProbability : ℚ := singleDieProbability ^ (numDice - 1)

theorem four_dice_same_face_probability :
  allSameProbability = 1 / 216 := by
  sorry

end four_dice_same_face_probability_l1588_158878


namespace urn_gold_coin_percentage_l1588_158889

/-- Represents the composition of objects in an urn --/
structure UrnComposition where
  beadPercentage : ℝ
  silverCoinPercentage : ℝ
  goldCoinPercentage : ℝ
  bronzeCoinPercentage : ℝ

/-- Calculates the percentage of gold coins in the urn --/
def goldCoinPercentage (u : UrnComposition) : ℝ :=
  (1 - u.beadPercentage) * u.goldCoinPercentage

/-- The theorem states that given the specified urn composition,
    the percentage of gold coins in the urn is 35% --/
theorem urn_gold_coin_percentage :
  ∀ (u : UrnComposition),
    u.beadPercentage = 0.3 ∧
    u.silverCoinPercentage = 0.25 * (1 - u.beadPercentage) ∧
    u.goldCoinPercentage = 0.5 * (1 - u.beadPercentage) ∧
    u.bronzeCoinPercentage = (1 - u.beadPercentage) * (1 - 0.25 - 0.5) →
    goldCoinPercentage u = 0.35 := by
  sorry

end urn_gold_coin_percentage_l1588_158889


namespace group_size_proof_l1588_158807

/-- The number of men in a group where:
    1) The average age increases by 1 year
    2) Two men aged 21 and 23 are replaced by two men with an average age of 32 -/
def number_of_men : ℕ := 20

theorem group_size_proof :
  let original_average : ℝ := number_of_men
  let new_average : ℝ := original_average + 1
  let replaced_sum : ℝ := 21 + 23
  let new_sum : ℝ := 2 * 32
  number_of_men * original_average + new_sum - replaced_sum = number_of_men * new_average :=
by sorry

end group_size_proof_l1588_158807


namespace height_distribution_study_l1588_158895

-- Define the type for students
def Student : Type := Unit

-- Define the school population
def schoolPopulation : Finset Student := sorry

-- Define the sample of measured students
def measuredSample : Finset Student := sorry

-- State the theorem
theorem height_distribution_study :
  (Finset.card schoolPopulation = 240) ∧
  (∀ s : Student, s ∈ schoolPopulation) ∧
  (measuredSample ⊆ schoolPopulation) ∧
  (Finset.card measuredSample = 40) →
  (Finset.card schoolPopulation = 240) ∧
  (∀ s : Student, s ∈ schoolPopulation → s = s) ∧
  (measuredSample = measuredSample) ∧
  (Finset.card measuredSample = 40) := by
  sorry

end height_distribution_study_l1588_158895


namespace unique_solution_characterization_l1588_158833

-- Define the function representing the equation
def f (a : ℝ) (x : ℝ) : Prop :=
  2 * Real.log (x + 3) = Real.log (a * x)

-- Define the set of a values for which the equation has a unique solution
def uniqueSolutionSet : Set ℝ :=
  {a : ℝ | a < 0 ∨ a = 12}

-- Theorem statement
theorem unique_solution_characterization (a : ℝ) :
  (∃! x : ℝ, f a x) ↔ a ∈ uniqueSolutionSet :=
sorry

end unique_solution_characterization_l1588_158833


namespace credit_card_balance_l1588_158854

/-- Represents the initial balance on a credit card -/
def initial_balance : ℝ := 170

/-- Represents the payment made on the credit card -/
def payment : ℝ := 50

/-- Represents the new balance after the payment -/
def new_balance : ℝ := 120

/-- Theorem stating that the initial balance minus the payment equals the new balance -/
theorem credit_card_balance :
  initial_balance - payment = new_balance := by sorry

end credit_card_balance_l1588_158854


namespace coprime_divisibility_theorem_l1588_158859

theorem coprime_divisibility_theorem (a b : ℕ+) :
  (Nat.gcd (2 * a.val - 1) (2 * b.val + 1) = 1) →
  (a.val + b.val ∣ 4 * a.val * b.val + 1) →
  ∃ n : ℕ+, a.val = n.val ∧ b.val = n.val + 1 :=
by sorry

end coprime_divisibility_theorem_l1588_158859


namespace distance_from_origin_to_point_l1588_158819

theorem distance_from_origin_to_point : Real.sqrt (12^2 + (-16)^2) = 20 := by
  sorry

end distance_from_origin_to_point_l1588_158819


namespace line_canonical_form_l1588_158886

theorem line_canonical_form (x y z : ℝ) :
  (2 * x - 3 * y - 3 * z - 9 = 0 ∧ x - 2 * y + z + 3 = 0) →
  ∃ (t : ℝ), x = 9 * t ∧ y = 5 * t ∧ z = t - 3 :=
by sorry

end line_canonical_form_l1588_158886


namespace smallest_prime_angle_in_special_right_triangle_l1588_158872

-- Define a structure for a right triangle with two acute angles
structure RightTriangle where
  angle1 : ℝ
  angle2 : ℝ
  sum_less_than_45 : angle1 + angle2 < 45
  angles_positive : 0 < angle1 ∧ 0 < angle2

-- Define a predicate for primality (approximate for real numbers)
def is_prime_approx (x : ℝ) : Prop := sorry

-- Define the theorem
theorem smallest_prime_angle_in_special_right_triangle :
  ∀ (t : RightTriangle),
    is_prime_approx t.angle1 →
    is_prime_approx t.angle2 →
    ∃ (smaller_angle : ℝ),
      smaller_angle = min t.angle1 t.angle2 ∧
      smaller_angle ≥ 2.3 :=
sorry

end smallest_prime_angle_in_special_right_triangle_l1588_158872


namespace normal_distribution_problem_l1588_158809

theorem normal_distribution_problem (σ μ : ℝ) (h1 : σ = 2) (h2 : μ = 55) :
  ∃ k : ℕ, k = 3 ∧ μ - k * σ > 48 ∧ ∀ m : ℕ, m > k → μ - m * σ ≤ 48 :=
sorry

end normal_distribution_problem_l1588_158809


namespace sin_three_pi_halves_l1588_158894

theorem sin_three_pi_halves : Real.sin (3 * π / 2) = -1 := by
  sorry

end sin_three_pi_halves_l1588_158894


namespace variance_best_for_stability_l1588_158893

-- Define a type for math test scores
def MathScore := ℝ

-- Define a type for a set of consecutive math test scores
def ConsecutiveScores := List MathScore

-- Define a function to calculate variance
noncomputable def variance (scores : ConsecutiveScores) : ℝ := sorry

-- Define a function to calculate other statistical measures
noncomputable def otherMeasure (scores : ConsecutiveScores) : ℝ := sorry

-- Define a function to measure stability
noncomputable def stability (scores : ConsecutiveScores) : ℝ := sorry

-- Theorem stating that variance is the most appropriate measure for stability
theorem variance_best_for_stability (scores : ConsecutiveScores) :
  ∀ (other : ConsecutiveScores → ℝ), other ≠ variance →
  |stability scores - variance scores| < |stability scores - other scores| :=
sorry

end variance_best_for_stability_l1588_158893


namespace solve_equation_l1588_158865

theorem solve_equation : 42 / (7 - 3/7) = 147/23 := by
  sorry

end solve_equation_l1588_158865


namespace train_catch_up_time_l1588_158813

/-- The problem of finding the time difference between two trains --/
theorem train_catch_up_time (goods_speed express_speed catch_up_time : ℝ) 
  (h1 : goods_speed = 36)
  (h2 : express_speed = 90)
  (h3 : catch_up_time = 4) :
  ∃ t : ℝ, t > 0 ∧ goods_speed * (t + catch_up_time) = express_speed * catch_up_time ∧ t = 6 := by
  sorry


end train_catch_up_time_l1588_158813


namespace section_B_students_l1588_158829

def section_A_students : ℕ := 50
def section_A_avg_weight : ℝ := 50
def section_B_avg_weight : ℝ := 70
def total_avg_weight : ℝ := 58.89

theorem section_B_students :
  ∃ x : ℕ, 
    (section_A_students * section_A_avg_weight + x * section_B_avg_weight) / (section_A_students + x) = total_avg_weight ∧
    x = 40 :=
by sorry

end section_B_students_l1588_158829


namespace postage_for_5_25_ounces_l1588_158891

/-- Calculates the postage cost for a letter given its weight and postage rates. -/
def calculate_postage (weight : ℚ) (base_rate : ℕ) (additional_rate : ℕ) : ℚ :=
  let additional_weight := max (weight - 1) 0
  let additional_charges := ⌈additional_weight⌉
  (base_rate + additional_charges * additional_rate) / 100

/-- Theorem stating that the postage for a 5.25 ounce letter is $1.60 under the given rates. -/
theorem postage_for_5_25_ounces :
  calculate_postage (5.25 : ℚ) 35 25 = (1.60 : ℚ) := by
  sorry

#eval calculate_postage (5.25 : ℚ) 35 25

end postage_for_5_25_ounces_l1588_158891


namespace solve_for_y_l1588_158844

theorem solve_for_y (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : x = 4) : y = 3 := by
  sorry

end solve_for_y_l1588_158844


namespace periodic_function_l1588_158877

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) + f (x - 1) = Real.sqrt 3 * f x

/-- The period of a function -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem periodic_function (f : ℝ → ℝ) (h : SatisfiesEquation f) :
    HasPeriod f 12 := by
  sorry

end periodic_function_l1588_158877


namespace specific_window_height_l1588_158898

/-- Represents a rectangular window with glass panes. -/
structure Window where
  num_panes : ℕ
  rows : ℕ
  columns : ℕ
  pane_height_ratio : ℚ
  pane_width_ratio : ℚ
  border_width : ℕ

/-- Calculates the height of a window given its specifications. -/
def window_height (w : Window) : ℕ :=
  let pane_width := 4 * w.border_width
  let pane_height := 3 * w.border_width
  pane_height * w.rows + w.border_width * (w.rows + 1)

/-- The theorem stating the height of the specific window. -/
theorem specific_window_height :
  let w : Window := {
    num_panes := 8,
    rows := 4,
    columns := 2,
    pane_height_ratio := 3/4,
    pane_width_ratio := 4/3,
    border_width := 3
  }
  window_height w = 51 := by sorry

end specific_window_height_l1588_158898


namespace fixed_monthly_costs_l1588_158880

/-- A problem about calculating fixed monthly costs for a computer manufacturer. -/
theorem fixed_monthly_costs (production_cost shipping_cost monthly_units lowest_price : ℕ) :
  production_cost = 80 →
  shipping_cost = 2 →
  monthly_units = 150 →
  lowest_price = 190 →
  (production_cost + shipping_cost) * monthly_units + 16200 = lowest_price * monthly_units :=
by sorry

end fixed_monthly_costs_l1588_158880


namespace solve_for_b_l1588_158840

theorem solve_for_b (m a b c k : ℝ) (h : m = (c^2 * a * b) / (a - k * b)) : 
  b = m * a / (c^2 * a + m * k) := by
  sorry

end solve_for_b_l1588_158840


namespace train_length_calculation_l1588_158815

theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (passing_time : ℝ) : 
  train_speed = 90 ∧ bridge_length = 140 ∧ passing_time = 20 → 
  (train_speed * 1000 / 3600) * passing_time - bridge_length = 360 := by
  sorry

#check train_length_calculation

end train_length_calculation_l1588_158815


namespace bread_cost_is_30_cents_l1588_158882

/-- The cost of a sandwich in dollars -/
def sandwich_price : ℚ := 1.5

/-- The cost of a slice of ham in dollars -/
def ham_cost : ℚ := 0.25

/-- The cost of a slice of cheese in dollars -/
def cheese_cost : ℚ := 0.35

/-- The total cost to make a sandwich in dollars -/
def total_cost : ℚ := 0.9

/-- The number of slices of bread in a sandwich -/
def bread_slices : ℕ := 2

/-- Theorem: The cost of a slice of bread is $0.30 -/
theorem bread_cost_is_30_cents :
  (total_cost - ham_cost - cheese_cost) / bread_slices = 0.3 := by
  sorry

end bread_cost_is_30_cents_l1588_158882


namespace no_divisible_by_19_l1588_158831

def a (n : ℕ) : ℤ := 9 * 10^n + 11

theorem no_divisible_by_19 : ∀ k : ℕ, k < 3050 → ¬(19 ∣ a k) := by
  sorry

end no_divisible_by_19_l1588_158831


namespace domain_of_f_l1588_158821

def f (x : ℝ) : ℝ := (x - 3) ^ (1/3) + (5 - x) ^ (1/3) + (x + 1) ^ (1/2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ -1} :=
sorry

end domain_of_f_l1588_158821


namespace necessary_not_sufficient_l1588_158857

theorem necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a ≤ 1 ∧ b ≤ 1 → a + b ≤ 2) ∧
  (∃ a b : ℝ, a + b ≤ 2 ∧ ¬(a ≤ 1 ∧ b ≤ 1)) := by
sorry

end necessary_not_sufficient_l1588_158857


namespace regular_pentagon_diagonal_angle_l1588_158860

/-- A regular pentagon is a polygon with 5 equal sides and 5 equal angles -/
structure RegularPentagon where
  vertices : Fin 5 → ℝ × ℝ
  is_regular : sorry

/-- The measure of an angle in degrees -/
def angle_measure (a b c : ℝ × ℝ) : ℝ := sorry

theorem regular_pentagon_diagonal_angle 
  (ABCDE : RegularPentagon) 
  (h_interior : ∀ (i : Fin 5), angle_measure (ABCDE.vertices i) (ABCDE.vertices (i + 1)) (ABCDE.vertices (i + 2)) = 108) :
  angle_measure (ABCDE.vertices 0) (ABCDE.vertices 2) (ABCDE.vertices 1) = 36 := by
  sorry

end regular_pentagon_diagonal_angle_l1588_158860


namespace gcd_360_504_l1588_158842

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by sorry

end gcd_360_504_l1588_158842


namespace meena_baked_five_dozens_l1588_158841

/-- The number of cookies in a dozen -/
def cookies_per_dozen : ℕ := 12

/-- The number of dozens of cookies sold to Mr. Stone -/
def dozens_sold_to_stone : ℕ := 2

/-- The number of cookies bought by Brock -/
def cookies_bought_by_brock : ℕ := 7

/-- The number of cookies Meena has left -/
def cookies_left : ℕ := 15

/-- Theorem: Meena baked 5 dozens of cookies initially -/
theorem meena_baked_five_dozens :
  let cookies_sold_to_stone := dozens_sold_to_stone * cookies_per_dozen
  let cookies_bought_by_katy := 2 * cookies_bought_by_brock
  let total_cookies_sold := cookies_sold_to_stone + cookies_bought_by_brock + cookies_bought_by_katy
  let total_cookies := total_cookies_sold + cookies_left
  total_cookies / cookies_per_dozen = 5 := by
  sorry

end meena_baked_five_dozens_l1588_158841


namespace trig_identity_l1588_158806

theorem trig_identity : Real.sin (47 * π / 180) * Real.cos (17 * π / 180) + 
                        Real.cos (47 * π / 180) * Real.cos (107 * π / 180) = 1/2 := by
  sorry

end trig_identity_l1588_158806


namespace train_speed_l1588_158811

/-- The speed of a train passing a jogger --/
theorem train_speed (jogger_speed : ℝ) (initial_lead : ℝ) (train_length : ℝ) (passing_time : ℝ) : 
  jogger_speed = 9 →
  initial_lead = 240 →
  train_length = 110 →
  passing_time = 35 →
  ∃ (train_speed : ℝ), train_speed = 45 := by
  sorry

#check train_speed

end train_speed_l1588_158811


namespace inequality_proof_l1588_158881

theorem inequality_proof (a b u v k : ℝ) 
  (ha : a > 0) (hb : b > 0) (huv : u < v) (hk : k > 0) :
  (a^u + b^u) / (a^v + b^v) ≥ (a^(u+k) + b^(u+k)) / (a^(v+k) + b^(v+k)) := by
  sorry

end inequality_proof_l1588_158881


namespace jason_remaining_seashells_l1588_158855

def initial_seashells : ℕ := 49
def seashells_given_away : ℕ := 13

theorem jason_remaining_seashells :
  initial_seashells - seashells_given_away = 36 :=
by sorry

end jason_remaining_seashells_l1588_158855


namespace multiply_add_distribute_l1588_158826

theorem multiply_add_distribute : 3.5 * 2.5 + 6.5 * 2.5 = 25 := by
  sorry

end multiply_add_distribute_l1588_158826


namespace solve_system_for_x_l1588_158875

theorem solve_system_for_x (x y : ℚ) 
  (eq1 : 3 * x - 4 * y = 8) 
  (eq2 : 2 * x + 3 * y = 1) : 
  x = 28 / 17 := by
  sorry

end solve_system_for_x_l1588_158875


namespace simplest_common_denominator_l1588_158861

variable (x : ℝ)

theorem simplest_common_denominator :
  ∃ (d : ℝ), d = x * (x + 1) * (x - 1) ∧
  (∃ (a b : ℝ), a / (x^2 - 1) + b / (x^2 + x) = (a * (x^2 + x) + b * (x^2 - 1)) / d) ∧
  (∀ (d' : ℝ), (∃ (a' b' : ℝ), a' / (x^2 - 1) + b' / (x^2 + x) = (a' * (x^2 + x) + b' * (x^2 - 1)) / d') →
    d ∣ d') :=
sorry

end simplest_common_denominator_l1588_158861


namespace equal_probability_implies_g_equals_5_l1588_158834

-- Define the number of marbles in each bag
def redMarbles1 : ℕ := 2
def blueMarbles1 : ℕ := 2
def redMarbles2 : ℕ := 2
def blueMarbles2 : ℕ := 2

-- Define the probability function for bag 1
def prob1 : ℚ := (redMarbles1 * (redMarbles1 - 1) + blueMarbles1 * (blueMarbles1 - 1)) / 
              ((redMarbles1 + blueMarbles1) * (redMarbles1 + blueMarbles1 - 1))

-- Define the probability function for bag 2
def prob2 (g : ℕ) : ℚ := (redMarbles2 * (redMarbles2 - 1) + blueMarbles2 * (blueMarbles2 - 1) + g * (g - 1)) / 
                       ((redMarbles2 + blueMarbles2 + g) * (redMarbles2 + blueMarbles2 + g - 1))

-- Theorem statement
theorem equal_probability_implies_g_equals_5 :
  ∃ (g : ℕ), g > 0 ∧ prob1 = prob2 g → g = 5 :=
sorry

end equal_probability_implies_g_equals_5_l1588_158834


namespace certain_number_multiplication_l1588_158856

theorem certain_number_multiplication (x : ℝ) : 37 - x = 24 → x * 24 = 312 := by
  sorry

end certain_number_multiplication_l1588_158856


namespace algebraic_expression_value_l1588_158897

theorem algebraic_expression_value (a : ℝ) (h : a^2 + 2*a - 1 = 5) :
  -2*a^2 - 4*a + 5 = -7 := by
  sorry

end algebraic_expression_value_l1588_158897


namespace subset_implies_m_values_l1588_158853

def A (m : ℝ) : Set ℝ := {-1, 3, 2*m-1}
def B (m : ℝ) : Set ℝ := {3, m}

theorem subset_implies_m_values (m : ℝ) :
  B m ⊆ A m → m = 1 ∨ m = -1 := by
  sorry

end subset_implies_m_values_l1588_158853


namespace system_solution_existence_l1588_158851

theorem system_solution_existence (a b : ℤ) :
  (∃ x y : ℝ, ⌊x⌋ + 2 * y = a ∧ ⌊y⌋ + 2 * x = b) ↔
  (a + b) % 3 = 0 ∨ (a + b) % 3 = 1 := by
  sorry

end system_solution_existence_l1588_158851


namespace bullet_problem_l1588_158839

theorem bullet_problem :
  ∀ (initial_bullets : ℕ),
    (5 * (initial_bullets - 4) = initial_bullets) →
    initial_bullets = 5 := by
  sorry

end bullet_problem_l1588_158839


namespace triangle_problem_l1588_158879

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  let vec_a : ℝ × ℝ := (Real.cos A, Real.cos B)
  let vec_b : ℝ × ℝ := (a, 2*c - b)
  (∃ k : ℝ, vec_a = k • vec_b) →  -- vectors are parallel
  b = 3 →
  (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 →  -- area condition
  A = π/3 ∧ a = Real.sqrt 13 := by
  sorry

end triangle_problem_l1588_158879


namespace geometric_progression_equality_l1588_158801

theorem geometric_progression_equality (a r : ℝ) (n : ℕ) (hr : r ≠ 1) :
  let S : ℕ → ℝ := λ m ↦ a * (r^m - 1) / (r - 1)
  (S n) / (S (2*n) - S n) = (S (2*n) - S n) / (S (3*n) - S (2*n)) := by
  sorry

end geometric_progression_equality_l1588_158801


namespace equation_solution_l1588_158852

theorem equation_solution : ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 54 ∧ x = 15 := by
  sorry

end equation_solution_l1588_158852


namespace point_on_positive_x_axis_l1588_158874

theorem point_on_positive_x_axis (m : ℝ) : 
  let x := m^2 + Real.pi
  let y := 0
  x > 0 ∧ y = 0 :=
by sorry

end point_on_positive_x_axis_l1588_158874


namespace right_handed_players_count_l1588_158850

theorem right_handed_players_count (total_players throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 37)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) :
  throwers + 2 * ((total_players - throwers) / 3) = 59 := by
sorry

end right_handed_players_count_l1588_158850


namespace infinite_logarithm_equation_l1588_158836

theorem infinite_logarithm_equation : ∃! x : ℝ, x > 0 ∧ 2^x = x + 64 := by
  sorry

end infinite_logarithm_equation_l1588_158836


namespace circle_equation_radius_l1588_158843

/-- Given a circle with equation x^2 - 8x + y^2 + 10y + d = 0 and radius 5, prove that d = 16 -/
theorem circle_equation_radius (d : ℝ) : 
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + d = 0 → (x - 4)^2 + (y + 5)^2 = 5^2) → 
  d = 16 := by
sorry

end circle_equation_radius_l1588_158843


namespace max_servings_is_twelve_l1588_158825

/-- Represents the number of servings that can be made from a given ingredient --/
def ServingsFromIngredient (available : ℕ) (required : ℕ) : ℕ :=
  (available * 4) / required

/-- Represents the recipe and available ingredients --/
structure SmoothieRecipe where
  bananas_required : ℕ
  yogurt_required : ℕ
  strawberries_required : ℕ
  bananas_available : ℕ
  yogurt_available : ℕ
  strawberries_available : ℕ

/-- Calculates the maximum number of servings that can be made --/
def MaxServings (recipe : SmoothieRecipe) : ℕ :=
  min (ServingsFromIngredient recipe.bananas_available recipe.bananas_required)
    (min (ServingsFromIngredient recipe.yogurt_available recipe.yogurt_required)
      (ServingsFromIngredient recipe.strawberries_available recipe.strawberries_required))

theorem max_servings_is_twelve :
  ∀ (recipe : SmoothieRecipe),
    recipe.bananas_required = 3 →
    recipe.yogurt_required = 2 →
    recipe.strawberries_required = 1 →
    recipe.bananas_available = 9 →
    recipe.yogurt_available = 10 →
    recipe.strawberries_available = 3 →
    MaxServings recipe = 12 := by
  sorry

end max_servings_is_twelve_l1588_158825


namespace spinner_probability_l1588_158818

theorem spinner_probability (pA pB pC pD pE : ℚ) : 
  pA = 3/8 →
  pB = 1/8 →
  pC = pD →
  pC = pE →
  pA + pB + pC + pD + pE = 1 →
  pC = 1/6 := by
sorry

end spinner_probability_l1588_158818


namespace square_diagonal_l1588_158817

theorem square_diagonal (perimeter : ℝ) (h : perimeter = 28) :
  let side := perimeter / 4
  let diagonal := Real.sqrt (2 * side ^ 2)
  diagonal = 7 * Real.sqrt 2 := by
sorry

end square_diagonal_l1588_158817


namespace cubic_equation_has_real_root_l1588_158805

theorem cubic_equation_has_real_root (a b : ℝ) : ∃ x : ℝ, x^3 + a*x - b = 0 := by
  sorry

end cubic_equation_has_real_root_l1588_158805


namespace sum_reciprocals_equals_two_l1588_158835

theorem sum_reciprocals_equals_two
  (a b c d : ℝ)
  (ω : ℂ)
  (ha : a ≠ -1)
  (hb : b ≠ -1)
  (hc : c ≠ -1)
  (hd : d ≠ -1)
  (hω1 : ω^4 = 1)
  (hω2 : ω ≠ 1)
  (h : 1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 2 / ω^2) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2 :=
by sorry

end sum_reciprocals_equals_two_l1588_158835


namespace three_digit_number_operation_l1588_158838

theorem three_digit_number_operation : ∀ a b c : ℕ,
  a ≥ 1 → a ≤ 9 →
  b ≥ 0 → b ≤ 9 →
  c ≥ 0 → c ≤ 9 →
  a = c + 3 →
  (100 * a + 10 * b + c) - ((100 * c + 10 * b + a) + 50) ≡ 7 [MOD 10] := by
  sorry

end three_digit_number_operation_l1588_158838


namespace candy_distribution_l1588_158867

theorem candy_distribution (total : Nat) (sisters : Nat) (take_away : Nat) : 
  total = 24 →
  sisters = 5 →
  take_away = 4 →
  (total - take_away) % sisters = 0 →
  ∀ x : Nat, x < take_away → (total - x) % sisters ≠ 0 := by
  sorry

end candy_distribution_l1588_158867


namespace find_extreme_stone_l1588_158873

/-- A stone with a specific weight -/
structure Stone where
  weight : ℝ

/-- A two-tiered balance scale that can compare two pairs of stones -/
def TwoTieredScale (stones : Finset Stone) : Prop :=
  ∀ (a b c d : Stone), a ∈ stones → b ∈ stones → c ∈ stones → d ∈ stones →
    ((a.weight + b.weight) > (c.weight + d.weight)) ∨
    ((a.weight + b.weight) < (c.weight + d.weight))

/-- The theorem stating that we can find either the heaviest or the lightest stone -/
theorem find_extreme_stone
  (stones : Finset Stone)
  (h_count : stones.card = 10)
  (h_distinct_weights : ∀ (a b : Stone), a ∈ stones → b ∈ stones → a ≠ b → a.weight ≠ b.weight)
  (h_distinct_sums : ∀ (a b c d : Stone), a ∈ stones → b ∈ stones → c ∈ stones → d ∈ stones →
    (a ≠ b ∧ c ≠ d ∧ (a, b) ≠ (c, d) ∧ (a, b) ≠ (d, c)) →
    a.weight + b.weight ≠ c.weight + d.weight)
  (h_scale : TwoTieredScale stones) :
  (∃ (s : Stone), s ∈ stones ∧ ∀ (t : Stone), t ∈ stones → s.weight ≥ t.weight) ∨
  (∃ (s : Stone), s ∈ stones ∧ ∀ (t : Stone), t ∈ stones → s.weight ≤ t.weight) :=
sorry

end find_extreme_stone_l1588_158873


namespace simplified_root_sum_l1588_158869

theorem simplified_root_sum (a b : ℕ+) :
  (2^11 * 5^5 : ℝ)^(1/4) = a * b^(1/4) → a + b = 30 := by
  sorry

end simplified_root_sum_l1588_158869


namespace composite_sum_l1588_158863

theorem composite_sum (a b : ℕ+) (h : 34 * a = 43 * b) : 
  ∃ (k m : ℕ) (hk : k > 1) (hm : m > 1), a + b = k * m := by
sorry

end composite_sum_l1588_158863


namespace reposition_convergence_l1588_158876

/-- Reposition transformation function -/
def reposition (n : ℕ) : ℕ :=
  sorry

/-- Theorem: Repeated reposition of a 4-digit number always results in 312 -/
theorem reposition_convergence (n : ℕ) (h : 1000 ≤ n ∧ n ≤ 9999) :
  ∃ k : ℕ, ∀ m : ℕ, m ≥ k → (reposition^[m] n) = 312 :=
sorry

end reposition_convergence_l1588_158876


namespace missing_number_implies_next_prime_l1588_158864

theorem missing_number_implies_next_prime (n : ℕ) : n > 3 →
  (∀ r s : ℕ, r ≥ 3 ∧ s ≥ 3 → n ≠ r * s - (r + s)) →
  Nat.Prime (n + 1) := by
  sorry

end missing_number_implies_next_prime_l1588_158864


namespace simplify_trig_expression_l1588_158871

theorem simplify_trig_expression : 
  Real.sqrt (1 - Real.sin (3 * Real.pi / 5) ^ 2) = -Real.cos (3 * Real.pi / 5) := by
  sorry

end simplify_trig_expression_l1588_158871


namespace no_solution_iff_m_equals_five_l1588_158858

theorem no_solution_iff_m_equals_five :
  ∀ m : ℝ, (∀ x : ℝ, x ≠ 5 ∧ x ≠ 8 → (x - 2) / (x - 5) ≠ (x - m) / (x - 8)) ↔ m = 5 := by
  sorry

end no_solution_iff_m_equals_five_l1588_158858


namespace problem_statement_l1588_158899

theorem problem_statement (x y : ℝ) 
  (h1 : 3 * x + y = 5) 
  (h2 : x + 3 * y = 6) : 
  5 * x^2 + 8 * x * y + 5 * y^2 = 61 := by
  sorry

end problem_statement_l1588_158899


namespace secant_minimum_value_l1588_158827

/-- The secant function -/
noncomputable def sec (x : ℝ) : ℝ := 1 / Real.cos x

/-- The function y = a * sec(bx + c) -/
noncomputable def f (a b c x : ℝ) : ℝ := a * sec (b * x + c)

theorem secant_minimum_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x : ℝ, f a b c x > 0 → f a b c x ≥ 3) →
  (∃ x : ℝ, f a b c x = 3) →
  a = 3 :=
sorry

end secant_minimum_value_l1588_158827


namespace max_value_of_fraction_l1588_158847

theorem max_value_of_fraction (x : ℝ) : (4*x^2 + 8*x + 19) / (4*x^2 + 8*x + 5) ≤ 15 := by
  sorry

#check max_value_of_fraction

end max_value_of_fraction_l1588_158847


namespace initial_sets_count_l1588_158890

/-- The number of letters available for initials -/
def num_letters : ℕ := 10

/-- The length of each set of initials -/
def set_length : ℕ := 4

/-- The number of different four-letter sets of initials possible using letters A through J -/
def num_initial_sets : ℕ := num_letters ^ set_length

theorem initial_sets_count : num_initial_sets = 10000 := by
  sorry

end initial_sets_count_l1588_158890


namespace pairing_count_l1588_158816

/-- The number of bowls -/
def num_bowls : ℕ := 4

/-- The number of glasses -/
def num_glasses : ℕ := 5

/-- The total number of possible pairings -/
def total_pairings : ℕ := num_bowls * num_glasses

theorem pairing_count : total_pairings = 20 := by sorry

end pairing_count_l1588_158816


namespace perfect_square_values_l1588_158845

theorem perfect_square_values (p : ℤ) (n : ℚ) : 
  n = 16 * (10 : ℚ)^(-p) →
  -4 < p →
  p < 2 →
  (∃ (a b c : ℤ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (16 * (10 : ℚ)^(-a) = (m : ℚ)^2 ∧
     16 * (10 : ℚ)^(-b) = (k : ℚ)^2 ∧
     16 * (10 : ℚ)^(-c) = (l : ℚ)^2) ∧
    (∀ (x : ℤ), x ≠ a ∧ x ≠ b ∧ x ≠ c →
      ¬∃ (y : ℚ), 16 * (10 : ℚ)^(-x) = y^2)) :=
by
  sorry

end perfect_square_values_l1588_158845


namespace tim_weekly_fluid_intake_l1588_158808

/-- Calculates Tim's weekly fluid intake in ounces -/
def weekly_fluid_intake : ℝ :=
  let water_bottles_per_day : ℝ := 2
  let water_quarts_per_bottle : ℝ := 1.5
  let orange_juice_oz_per_day : ℝ := 20
  let soda_liters_per_other_day : ℝ := 1.5
  let coffee_cups_per_week : ℝ := 4
  let quart_to_oz : ℝ := 32
  let liter_to_oz : ℝ := 33.814
  let cup_to_oz : ℝ := 8
  let days_per_week : ℝ := 7
  let soda_days_per_week : ℝ := 4

  let water_oz : ℝ := water_bottles_per_day * water_quarts_per_bottle * quart_to_oz * days_per_week
  let orange_juice_oz : ℝ := orange_juice_oz_per_day * days_per_week
  let soda_oz : ℝ := soda_liters_per_other_day * liter_to_oz * soda_days_per_week
  let coffee_oz : ℝ := coffee_cups_per_week * cup_to_oz

  water_oz + orange_juice_oz + soda_oz + coffee_oz

/-- Theorem stating Tim's weekly fluid intake -/
theorem tim_weekly_fluid_intake : weekly_fluid_intake = 1046.884 := by
  sorry

end tim_weekly_fluid_intake_l1588_158808


namespace sphere_ratios_l1588_158888

/-- Given two spheres with radii in the ratio 2:3, prove that the ratio of their surface areas is 4:9 and the ratio of their volumes is 8:27 -/
theorem sphere_ratios (r₁ r₂ : ℝ) (h : r₁ / r₂ = 2 / 3) :
  (4 * π * r₁^2) / (4 * π * r₂^2) = 4 / 9 ∧
  ((4 / 3) * π * r₁^3) / ((4 / 3) * π * r₂^3) = 8 / 27 := by
  sorry

end sphere_ratios_l1588_158888


namespace quiz_score_ratio_l1588_158868

/-- Given a quiz taken by three people with specific scoring conditions,
    prove that the ratio of Tatuya's score to Ivanna's score is 2:1 -/
theorem quiz_score_ratio (tatuya_score ivanna_score dorothy_score : ℚ) : 
  dorothy_score = 90 →
  ivanna_score = (3/5) * dorothy_score →
  (tatuya_score + ivanna_score + dorothy_score) / 3 = 84 →
  tatuya_score / ivanna_score = 2 := by
sorry

end quiz_score_ratio_l1588_158868


namespace product_evaluation_l1588_158800

theorem product_evaluation (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹ + d⁻¹) * (a*b + b*c + c*d + d*a + a*c + b*d)⁻¹ *
  ((a*b)⁻¹ + (b*c)⁻¹ + (c*d)⁻¹ + (d*a)⁻¹ + (a*c)⁻¹ + (b*d)⁻¹) = (a*a*b*b*c*c*d*d)⁻¹ :=
by sorry

end product_evaluation_l1588_158800


namespace perfect_square_arrangement_l1588_158832

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that represents a permutation of numbers from 1 to n -/
def permutation (n : ℕ) := Fin n → Fin n

/-- A property that checks if a permutation satisfies the perfect square sum condition -/
def valid_permutation (n : ℕ) (p : permutation n) : Prop :=
  ∀ i : Fin n, is_perfect_square (i.val + 1 + (p i).val + 1)

theorem perfect_square_arrangement :
  (∃ p : permutation 9, valid_permutation 9 p) ∧
  (¬ ∃ p : permutation 11, valid_permutation 11 p) ∧
  (∃ p : permutation 1996, valid_permutation 1996 p) :=
sorry

end perfect_square_arrangement_l1588_158832


namespace john_remaining_money_l1588_158849

def trip_finances (initial_amount spent_amount remaining_amount : ℕ) : Prop :=
  (initial_amount = 1600) ∧
  (remaining_amount = spent_amount - 600) ∧
  (remaining_amount = initial_amount - spent_amount)

theorem john_remaining_money :
  ∃ (spent_amount remaining_amount : ℕ),
    trip_finances 1600 spent_amount remaining_amount ∧
    remaining_amount = 500 :=
by
  sorry

end john_remaining_money_l1588_158849


namespace sum_of_powers_divisibility_l1588_158823

theorem sum_of_powers_divisibility (n : ℕ+) :
  (((1:ℤ)^n.val + 2^n.val + 3^n.val + 4^n.val) % 5 = 0) ↔ (n.val % 4 ≠ 0) := by
  sorry

end sum_of_powers_divisibility_l1588_158823


namespace profit_discount_rate_l1588_158812

/-- Proves that a 20% profit on a product with a purchase price of 200 yuan and a marked price of 300 yuan is achieved by selling at 80% of the marked price. -/
theorem profit_discount_rate (purchase_price marked_price : ℝ) 
  (h_purchase : purchase_price = 200)
  (h_marked : marked_price = 300)
  (profit_rate : ℝ) (h_profit : profit_rate = 0.2)
  (discount_rate : ℝ) :
  discount_rate * marked_price = purchase_price * (1 + profit_rate) →
  discount_rate = 0.8 := by
sorry

end profit_discount_rate_l1588_158812


namespace angle_C_is_two_pi_third_l1588_158885

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define vectors p and q
def p (t : Triangle) : ℝ × ℝ := (t.a + t.c, t.b)
def q (t : Triangle) : ℝ × ℝ := (t.b + t.a, t.c - t.a)

-- Define parallelism of vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem statement
theorem angle_C_is_two_pi_third (t : Triangle) 
  (h : parallel (p t) (q t)) : t.C = 2 * π / 3 := by
  sorry

end angle_C_is_two_pi_third_l1588_158885


namespace pool_time_ratio_l1588_158866

/-- The ratio of George's time to Elaine's time in the pool --/
def time_ratio (jerry_time elaine_time george_time : ℚ) : ℚ × ℚ :=
  (george_time, elaine_time)

theorem pool_time_ratio :
  ∀ (jerry_time elaine_time george_time total_time : ℚ),
    jerry_time = 3 →
    elaine_time = 2 * jerry_time →
    total_time = 11 →
    total_time = jerry_time + elaine_time + george_time →
    time_ratio jerry_time elaine_time george_time = (1, 3) := by
  sorry

#check pool_time_ratio

end pool_time_ratio_l1588_158866


namespace product_inequality_l1588_158837

theorem product_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_prod : a * b * c * d = 1) :
  (a^4 + b^4) / (a^2 + b^2) + (b^4 + c^4) / (b^2 + c^2) + 
  (c^4 + d^4) / (c^2 + d^2) + (d^4 + a^4) / (d^2 + a^2) ≥ 4 := by
  sorry

end product_inequality_l1588_158837


namespace equality_of_sqrt_five_terms_l1588_158884

theorem equality_of_sqrt_five_terms 
  (a b c d : ℚ) 
  (h : a + b * Real.sqrt 5 = c + d * Real.sqrt 5) : 
  a = c ∧ b = d := by
sorry

end equality_of_sqrt_five_terms_l1588_158884


namespace min_value_theorem_l1588_158892

theorem min_value_theorem (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 8) 
  (h2 : e * f * g * h = 16) : 
  (2 * a * e)^2 + (2 * b * f)^2 + (2 * c * g)^2 + (2 * d * h)^2 ≥ 512 := by
  sorry

end min_value_theorem_l1588_158892


namespace exists_number_not_in_progressions_l1588_158862

/-- Represents a geometric progression of natural numbers -/
structure GeometricProgression where
  first_term : ℕ
  common_ratio : ℕ
  h_positive : common_ratio > 1

/-- Checks if a natural number is in a geometric progression -/
def isInProgression (n : ℕ) (gp : GeometricProgression) : Prop :=
  ∃ k : ℕ, n = gp.first_term * gp.common_ratio ^ k

/-- The main theorem -/
theorem exists_number_not_in_progressions (progressions : Fin 100 → GeometricProgression) :
  ∃ n : ℕ, ∀ i : Fin 100, ¬ isInProgression n (progressions i) := by
  sorry


end exists_number_not_in_progressions_l1588_158862


namespace field_division_l1588_158883

theorem field_division (total_area smaller_area : ℝ) (h1 : total_area = 700) (h2 : smaller_area = 315) :
  ∃ (larger_area X : ℝ),
    larger_area + smaller_area = total_area ∧
    larger_area - smaller_area = (1 / 5) * X ∧
    X = 350 := by
  sorry

end field_division_l1588_158883


namespace rectangle_area_diagonal_l1588_158803

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  ∃ k : ℝ, l * w = k * d^2 ∧ k = 10 / 29 := by
sorry

end rectangle_area_diagonal_l1588_158803


namespace difference_of_squares_division_l1588_158804

theorem difference_of_squares_division : (324^2 - 300^2) / 24 = 624 := by
  sorry

end difference_of_squares_division_l1588_158804


namespace least_possible_difference_l1588_158887

theorem least_possible_difference (x y z : ℤ) : 
  x < y → y < z → 
  y - x > 5 → 
  Even x → 
  Odd y → Odd z → 
  (∀ w, w = z - x → w ≥ 9) ∧ (∃ w, w = z - x ∧ w = 9) :=
by sorry

end least_possible_difference_l1588_158887


namespace expected_rolls_in_year_l1588_158814

/-- Represents the possible outcomes of rolling an 8-sided die -/
inductive DieOutcome
  | Prime
  | Composite
  | OddNonPrime
  | Reroll

/-- The probability of each outcome when rolling a fair 8-sided die -/
def outcomeProb (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.Prime => 1/2
  | DieOutcome.Composite => 1/4
  | DieOutcome.OddNonPrime => 1/8
  | DieOutcome.Reroll => 1/8

/-- The expected number of rolls on a single day -/
noncomputable def expectedRollsPerDay : ℝ :=
  1

/-- The number of days in a non-leap year -/
def daysInNonLeapYear : ℕ := 365

/-- Theorem: The expected number of die rolls in a non-leap year
    is equal to the number of days in the year -/
theorem expected_rolls_in_year :
  (expectedRollsPerDay * daysInNonLeapYear : ℝ) = daysInNonLeapYear := by
  sorry

end expected_rolls_in_year_l1588_158814


namespace same_perimeter_l1588_158820

-- Define the rectangle
def rectangle_length : ℝ := 10
def rectangle_width : ℝ := 8

-- Define the square side length
def square_side : ℝ := 9

-- Define perimeter functions
def rectangle_perimeter (l w : ℝ) : ℝ := 2 * (l + w)
def square_perimeter (s : ℝ) : ℝ := 4 * s

-- Theorem statement
theorem same_perimeter :
  rectangle_perimeter rectangle_length rectangle_width = square_perimeter square_side :=
by sorry

end same_perimeter_l1588_158820
