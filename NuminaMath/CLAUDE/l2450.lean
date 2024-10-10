import Mathlib

namespace student_correct_answers_l2450_245067

/-- Represents a test score calculation system -/
structure TestScore where
  totalQuestions : ℕ
  score : ℤ
  correctAnswers : ℕ
  incorrectAnswers : ℕ

/-- Theorem: Given the conditions, prove that the student answered 91 questions correctly -/
theorem student_correct_answers
  (test : TestScore)
  (h1 : test.totalQuestions = 100)
  (h2 : test.score = test.correctAnswers - 2 * test.incorrectAnswers)
  (h3 : test.score = 73)
  (h4 : test.correctAnswers + test.incorrectAnswers = test.totalQuestions) :
  test.correctAnswers = 91 := by
  sorry


end student_correct_answers_l2450_245067


namespace fourth_fifth_sum_l2450_245029

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n
  sum_first_two : a 1 + a 2 = 1
  sum_third_fourth : a 3 + a 4 = 9

/-- The sum of the fourth and fifth terms is either 27 or -27 -/
theorem fourth_fifth_sum (seq : GeometricSequence) : 
  seq.a 4 + seq.a 5 = 27 ∨ seq.a 4 + seq.a 5 = -27 := by
  sorry


end fourth_fifth_sum_l2450_245029


namespace darryl_break_even_point_l2450_245036

/-- Calculates the break-even point for Darryl's machine sales -/
theorem darryl_break_even_point 
  (parts_cost : ℕ) 
  (patent_cost : ℕ) 
  (selling_price : ℕ) 
  (h1 : parts_cost = 3600)
  (h2 : patent_cost = 4500)
  (h3 : selling_price = 180) :
  (parts_cost + patent_cost) / selling_price = 45 :=
by sorry

end darryl_break_even_point_l2450_245036


namespace word_probabilities_l2450_245070

def word : String := "дифференцициал"

def is_vowel (c : Char) : Bool :=
  c ∈ ['а', 'е', 'и']

def is_consonant (c : Char) : Bool :=
  c ∈ ['д', 'ф', 'р', 'н', 'ц', 'л']

theorem word_probabilities :
  let total_letters := word.length
  let vowels := (word.toList.filter is_vowel).length
  let consonants := (word.toList.filter is_consonant).length
  (vowels : ℚ) / total_letters = 5 / 12 ∧
  (consonants : ℚ) / total_letters = 7 / 12 ∧
  ((word.toList.filter (· = 'ч')).length : ℚ) / total_letters = 0 := by
  sorry


end word_probabilities_l2450_245070


namespace hyperbola_equation_correct_l2450_245099

/-- A hyperbola is defined by its equation, asymptotes, and a point it passes through. -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form (x²/a²) - (y²/b²) = 1 -/
  equation : ℝ → ℝ → Prop
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- A point that the hyperbola passes through -/
  point : ℝ × ℝ

/-- The hyperbola satisfies its equation at the given point -/
def satisfies_equation (h : Hyperbola) : Prop :=
  h.equation h.point.1 h.point.2

/-- The asymptotes of the hyperbola have the correct slope -/
def has_correct_asymptotes (h : Hyperbola) : Prop :=
  h.asymptote_slope = 1 / 2

/-- Theorem stating that the given hyperbola equation is correct -/
theorem hyperbola_equation_correct (h : Hyperbola)
  (heq : h.equation = fun x y => x^2 / 8 - y^2 / 2 = 1)
  (hpoint : h.point = (4, Real.sqrt 2))
  (hslope : h.asymptote_slope = 1 / 2)
  : satisfies_equation h ∧ has_correct_asymptotes h := by
  sorry


end hyperbola_equation_correct_l2450_245099


namespace production_growth_l2450_245047

theorem production_growth (a : ℝ) (x : ℕ) (y : ℝ) (h : x > 0) :
  y = a * (1 + 0.05) ^ x ↔ 
  (∀ n : ℕ, n ≤ x → 
    (n = 0 → y = a) ∧ 
    (n > 0 → y = a * (1 + 0.05) ^ n)) :=
sorry

end production_growth_l2450_245047


namespace marys_remaining_money_equals_50_minus_12p_l2450_245017

/-- The amount of money Mary has left after purchasing pizzas and drinks -/
def marys_remaining_money (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 2 * p
  let large_pizza_cost := 3 * p
  let total_cost := 5 * drink_cost + 2 * medium_pizza_cost + large_pizza_cost
  50 - total_cost

/-- Theorem stating that Mary's remaining money is equal to 50 - 12p -/
theorem marys_remaining_money_equals_50_minus_12p (p : ℝ) :
  marys_remaining_money p = 50 - 12 * p := by
  sorry

end marys_remaining_money_equals_50_minus_12p_l2450_245017


namespace sum_of_roots_quadratic_l2450_245022

/-- Given a quadratic equation x^2 - bx + 20 = 0 where the product of roots is 20,
    prove that the sum of roots is b. -/
theorem sum_of_roots_quadratic (b : ℝ) : 
  (∃ x y : ℝ, x^2 - b*x + 20 = 0 ∧ y^2 - b*y + 20 = 0 ∧ x*y = 20) → 
  (∃ x y : ℝ, x^2 - b*x + 20 = 0 ∧ y^2 - b*y + 20 = 0 ∧ x + y = b) :=
by sorry

end sum_of_roots_quadratic_l2450_245022


namespace arithmetic_geometric_mean_difference_bounds_l2450_245035

theorem arithmetic_geometric_mean_difference_bounds (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧
  (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := by
  sorry

end arithmetic_geometric_mean_difference_bounds_l2450_245035


namespace hall_mat_expenditure_l2450_245095

/-- Calculates the total expenditure for covering the interior of a rectangular hall with mat -/
def total_expenditure (length width height cost_per_sqm : ℝ) : ℝ :=
  let floor_area := length * width
  let wall_area := 2 * (length * height + width * height)
  let total_area := floor_area + wall_area
  total_area * cost_per_sqm

/-- Theorem stating that the total expenditure for the given hall dimensions and mat cost is 19500 -/
theorem hall_mat_expenditure :
  total_expenditure 20 15 5 30 = 19500 := by
  sorry

#eval total_expenditure 20 15 5 30

end hall_mat_expenditure_l2450_245095


namespace cosine_amplitude_l2450_245005

theorem cosine_amplitude (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, a * Real.cos (b * x) ≤ 3) ∧
  (∃ x, a * Real.cos (b * x) = 3) ∧
  (∀ x, a * Real.cos (b * x) = a * Real.cos (b * (x + 2 * Real.pi))) →
  a = 3 := by
sorry

end cosine_amplitude_l2450_245005


namespace community_center_chairs_l2450_245088

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : Nat) : Nat :=
  (n / 100) * 25 + ((n % 100) / 10) * 5 + (n % 10)

/-- Calculates the number of chairs needed given a capacity in base 5 and people per chair -/
def chairsNeeded (capacityBase5 : Nat) (peoplePerChair : Nat) : Nat :=
  (base5ToBase10 capacityBase5) / peoplePerChair

theorem community_center_chairs :
  chairsNeeded 310 3 = 26 := by
  sorry

end community_center_chairs_l2450_245088


namespace right_triangle_angle_split_l2450_245010

theorem right_triangle_angle_split (BC AC : ℝ) (h_right : BC = 5 ∧ AC = 12) :
  let AB := Real.sqrt (BC^2 + AC^2)
  let angle_ratio := (1 : ℝ) / 3
  let smaller_segment := AB * (Real.sqrt 3 / 2)
  smaller_segment = 13 * Real.sqrt 3 / 2 := by
  sorry

end right_triangle_angle_split_l2450_245010


namespace joke_spread_after_one_minute_l2450_245048

def joke_spread (base : ℕ) (intervals : ℕ) : ℕ :=
  (base^(intervals + 1) - 1) / (base - 1)

theorem joke_spread_after_one_minute :
  joke_spread 6 6 = 55987 :=
by sorry

end joke_spread_after_one_minute_l2450_245048


namespace nine_books_arrangement_l2450_245055

/-- Represents a collection of books with specific adjacency requirements -/
structure BookArrangement where
  total_books : Nat
  adjacent_pairs : Nat
  single_books : Nat

/-- Calculates the number of ways to arrange books with adjacency requirements -/
def arrange_books (ba : BookArrangement) : Nat :=
  (2 ^ ba.adjacent_pairs) * Nat.factorial (ba.single_books + ba.adjacent_pairs)

/-- Theorem stating the number of ways to arrange 9 books with 2 adjacent pairs -/
theorem nine_books_arrangement :
  arrange_books ⟨9, 2, 5⟩ = 4 * Nat.factorial 7 := by
  sorry

end nine_books_arrangement_l2450_245055


namespace dagger_five_eighths_three_fourths_l2450_245033

-- Define the operation †
def dagger (m n p q : ℚ) : ℚ := 2 * m * p * (q / n)

-- Theorem statement
theorem dagger_five_eighths_three_fourths :
  dagger (5/8) (8/8) (3/4) (4/4) = 15 := by
  sorry

end dagger_five_eighths_three_fourths_l2450_245033


namespace marys_double_counted_sheep_l2450_245076

/-- Given Mary's animal counting problem, prove that she double-counted 7 sheep. -/
theorem marys_double_counted_sheep :
  let marys_count : ℕ := 60
  let actual_animals : ℕ := 56
  let forgotten_pigs : ℕ := 3
  let double_counted_sheep : ℕ := marys_count - actual_animals + forgotten_pigs
  double_counted_sheep = 7 := by sorry

end marys_double_counted_sheep_l2450_245076


namespace coefficient_of_y_l2450_245061

theorem coefficient_of_y (x y a : ℝ) : 
  7 * x + y = 19 → 
  x + a * y = 1 → 
  2 * x + y = 5 → 
  a = 3 := by
sorry

end coefficient_of_y_l2450_245061


namespace striped_shirt_ratio_l2450_245064

theorem striped_shirt_ratio (total : ℕ) (checkered shorts striped : ℕ) : 
  total = 81 →
  total = checkered + striped →
  shorts = checkered + 19 →
  striped = shorts + 8 →
  striped * 3 = total * 2 := by
sorry

end striped_shirt_ratio_l2450_245064


namespace power_sum_problem_l2450_245062

theorem power_sum_problem (a b : ℝ) 
  (h1 : a^5 + b^5 = 3) 
  (h2 : a^15 + b^15 = 9) : 
  a^10 + b^10 = 5 := by
sorry

end power_sum_problem_l2450_245062


namespace set_operations_and_intersection_l2450_245019

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | a < x}

theorem set_operations_and_intersection :
  (A ∪ B = {x | 1 < x ∧ x ≤ 8}) ∧
  ((Aᶜ) ∩ B = {x | 1 < x ∧ x < 2}) ∧
  (∀ a : ℝ, (A ∩ C a).Nonempty ↔ a < 8) := by sorry

end set_operations_and_intersection_l2450_245019


namespace translation_result_l2450_245013

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translates a point vertically -/
def translateVertical (p : Point2D) (dy : ℝ) : Point2D :=
  { x := p.x, y := p.y + dy }

/-- Translates a point horizontally -/
def translateHorizontal (p : Point2D) (dx : ℝ) : Point2D :=
  { x := p.x - dx, y := p.y }

/-- The theorem to be proved -/
theorem translation_result :
  let initial_point : Point2D := { x := 3, y := -2 }
  let after_vertical := translateVertical initial_point 3
  let final_point := translateHorizontal after_vertical 2
  final_point = { x := 1, y := 1 } := by
  sorry

end translation_result_l2450_245013


namespace number_division_l2450_245073

theorem number_division : ∃ x : ℝ, x / 0.04 = 400.90000000000003 ∧ x = 16.036 := by
  sorry

end number_division_l2450_245073


namespace units_digit_of_product_l2450_245051

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def unitsDigit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_product :
  unitsDigit (2 * (factorial 1 + factorial 2 + factorial 3 + factorial 4)) = 6 := by
  sorry

end units_digit_of_product_l2450_245051


namespace custom_op_example_l2450_245094

/-- Custom binary operation ⊗ defined as a ⊗ b = a - |b| -/
def custom_op (a b : ℝ) : ℝ := a - abs b

/-- Theorem stating that 2 ⊗ (-3) = -1 -/
theorem custom_op_example : custom_op 2 (-3) = -1 := by
  sorry

end custom_op_example_l2450_245094


namespace straight_line_distance_l2450_245020

/-- The straight-line distance between two points, where one point is 20 yards south
    and 50 yards east of the other, is 10√29 yards. -/
theorem straight_line_distance (south east : ℝ) (h1 : south = 20) (h2 : east = 50) :
  Real.sqrt (south^2 + east^2) = 10 * Real.sqrt 29 := by
  sorry

end straight_line_distance_l2450_245020


namespace solutions_equation1_solutions_equation2_l2450_245058

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := x^2 + 3*x - 4 = 0
def equation2 (x : ℝ) : Prop := 2*x^2 - 4*x - 1 = 0

-- Theorem for the solutions of the first equation
theorem solutions_equation1 : 
  (∃ x : ℝ, equation1 x) ↔ (equation1 1 ∧ equation1 (-4)) :=
sorry

-- Theorem for the solutions of the second equation
theorem solutions_equation2 : 
  (∃ x : ℝ, equation2 x) ↔ (equation2 (1 + Real.sqrt 6 / 2) ∧ equation2 (1 - Real.sqrt 6 / 2)) :=
sorry

end solutions_equation1_solutions_equation2_l2450_245058


namespace polynomial_remainder_l2450_245060

/-- Given a polynomial q(x) = Dx^4 + Ex^2 + Fx + 9, 
    if the remainder when divided by x - 2 is 17, 
    then the remainder when divided by x + 2 is 33. -/
theorem polynomial_remainder (D E F : ℝ) : 
  let q : ℝ → ℝ := λ x => D*x^4 + E*x^2 + F*x + 9
  (q 2 = 17) → (q (-2) = 33) := by
  sorry

end polynomial_remainder_l2450_245060


namespace joey_age_is_12_l2450_245087

def ages : List ℕ := [4, 6, 8, 10, 12, 14]

def went_to_movies (a b : ℕ) : Prop := a + b = 18 ∧ a ∈ ages ∧ b ∈ ages ∧ a ≠ b

def went_to_soccer (a b : ℕ) : Prop := a < 12 ∧ b < 12 ∧ a ∈ ages ∧ b ∈ ages ∧ a ≠ b

def stayed_home (joey_age : ℕ) : Prop := joey_age ∈ ages ∧ 6 ∈ ages

theorem joey_age_is_12 :
  ∃! (joey_age : ℕ),
    (∃ (a b c d : ℕ),
      went_to_movies a b ∧
      went_to_soccer c d ∧
      stayed_home joey_age ∧
      {a, b, c, d, joey_age, 6} = ages.toFinset) ∧
    joey_age = 12 :=
by sorry

end joey_age_is_12_l2450_245087


namespace cristina_croissants_l2450_245097

/-- The number of croissants Cristina baked -/
def total_croissants (num_guests : ℕ) (croissants_per_guest : ℕ) : ℕ :=
  num_guests * croissants_per_guest

/-- Proof that Cristina baked 14 croissants -/
theorem cristina_croissants :
  total_croissants 7 2 = 14 := by
  sorry

end cristina_croissants_l2450_245097


namespace problem_solution_l2450_245090

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.tan x = 1

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2

-- Theorem to prove
theorem problem_solution :
  (p ∧ q) ∧
  ¬(p ∧ ¬q) ∧
  (¬p ∨ q) ∧
  ¬(¬p ∨ ¬q) := by
  sorry

end problem_solution_l2450_245090


namespace probability_theorem_l2450_245089

-- Define the probabilities of events A1, A2, A3
def P_A1 : ℚ := 1/2
def P_A2 : ℚ := 1/5
def P_A3 : ℚ := 3/10

-- Define the conditional probabilities
def P_B_given_A1 : ℚ := 5/11
def P_B_given_A2 : ℚ := 4/11
def P_B_given_A3 : ℚ := 4/11

-- Define the probability of event B
def P_B : ℚ := 9/22

-- Define the theorem to be proved
theorem probability_theorem :
  (P_B_given_A1 = 5/11) ∧
  (P_B = 9/22) ∧
  (P_A1 + P_A2 + P_A3 = 1) := by
  sorry

#check probability_theorem

end probability_theorem_l2450_245089


namespace rhombus_longer_diagonal_l2450_245003

theorem rhombus_longer_diagonal (side_length : ℝ) (shorter_diagonal : ℝ) (longer_diagonal : ℝ) : 
  side_length = 40 →
  shorter_diagonal = 30 →
  longer_diagonal = 10 * Real.sqrt 55 :=
by sorry

end rhombus_longer_diagonal_l2450_245003


namespace reflection_matrix_correct_l2450_245054

/-- Reflection matrix over the line y = x -/
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 1],
    ![1, 0]]

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflection over the line y = x -/
def reflect (p : Point2D) : Point2D :=
  ⟨p.y, p.x⟩

theorem reflection_matrix_correct :
  ∀ (p : Point2D),
  let reflected := reflect p
  let matrix_result := reflection_matrix.mulVec ![p.x, p.y]
  matrix_result = ![reflected.x, reflected.y] := by
  sorry

end reflection_matrix_correct_l2450_245054


namespace b_21_mod_12_l2450_245077

/-- Definition of b_n as the integer obtained by writing all integers from 1 to n from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- Theorem stating that b_21 mod 12 = 9 -/
theorem b_21_mod_12 : b 21 % 12 = 9 := by sorry

end b_21_mod_12_l2450_245077


namespace quadratic_root_implication_l2450_245082

theorem quadratic_root_implication (a b : ℝ) : 
  (∃ x : ℝ, a * x^2 + b * x + 6 = 0 ∧ x = -2) → 
  6 * a - 3 * b + 6 = -3 := by
  sorry

end quadratic_root_implication_l2450_245082


namespace cost_of_pens_l2450_245014

theorem cost_of_pens (box_size : ℕ) (box_cost : ℚ) (total_pens : ℕ) : 
  box_size = 150 → box_cost = 45 → total_pens = 4500 → 
  (total_pens : ℚ) * (box_cost / box_size) = 1350 := by
sorry

end cost_of_pens_l2450_245014


namespace mixture_proportion_l2450_245004

/-- Represents a solution with a given percentage of chemical a -/
structure Solution :=
  (percent_a : ℝ)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (sol_x : Solution)
  (sol_y : Solution)
  (percent_x : ℝ)
  (percent_mixture_a : ℝ)

/-- The theorem stating the proportion of solution x in the mixture -/
theorem mixture_proportion
  (mix : Mixture)
  (hx : mix.sol_x.percent_a = 0.3)
  (hy : mix.sol_y.percent_a = 0.4)
  (hm : mix.percent_mixture_a = 0.32)
  : mix.percent_x = 0.8 := by
  sorry

end mixture_proportion_l2450_245004


namespace five_line_triangle_bounds_l2450_245040

/-- A line in a plane --/
structure Line where
  -- Add necessary fields here
  
/-- A region in a plane --/
structure Region where
  -- Add necessary fields here

/-- Represents a configuration of lines in a plane --/
structure PlaneConfiguration where
  lines : List Line
  regions : List Region

/-- Checks if lines are in general position --/
def is_general_position (config : PlaneConfiguration) : Prop :=
  sorry

/-- Counts the number of triangular regions --/
def count_triangles (config : PlaneConfiguration) : Nat :=
  sorry

/-- Main theorem about triangles in a plane divided by five lines --/
theorem five_line_triangle_bounds 
  (config : PlaneConfiguration) 
  (h1 : config.lines.length = 5)
  (h2 : config.regions.length = 16)
  (h3 : is_general_position config) :
  3 ≤ count_triangles config ∧ count_triangles config ≤ 5 :=
sorry

end five_line_triangle_bounds_l2450_245040


namespace principal_is_300_l2450_245015

/-- Given a principal amount P and an interest rate R, 
    if increasing the rate by 6% for 5 years results in 90 more interest,
    then P must be 300. -/
theorem principal_is_300 (P R : ℝ) : 
  (P * (R + 6) * 5) / 100 = (P * R * 5) / 100 + 90 → P = 300 := by
  sorry

end principal_is_300_l2450_245015


namespace y_in_terms_of_x_l2450_245023

theorem y_in_terms_of_x (x y : ℝ) : 2 * x + y = 5 → y = 5 - 2 * x := by
  sorry

end y_in_terms_of_x_l2450_245023


namespace length_PR_l2450_245032

-- Define the circle and points
def Circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def O : ℝ × ℝ := (0, 0)  -- Center of the circle
def radius : ℝ := 10

-- Define points P, Q, and R
variable (P Q R : ℝ × ℝ)

-- State the conditions
variable (h1 : P ∈ Circle O radius)
variable (h2 : Q ∈ Circle O radius)
variable (h3 : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 12^2)
variable (h4 : R ∈ Circle O radius)
variable (h5 : R.1 = (P.1 + Q.1) / 2 ∧ R.2 = (P.2 + Q.2) / 2)

-- State the theorem
theorem length_PR : (P.1 - R.1)^2 + (P.2 - R.2)^2 = 40 := by sorry

end length_PR_l2450_245032


namespace swords_per_orc_l2450_245056

theorem swords_per_orc (total_swords : ℕ) (num_squads : ℕ) (orcs_per_squad : ℕ) :
  total_swords = 1200 →
  num_squads = 10 →
  orcs_per_squad = 8 →
  total_swords / (num_squads * orcs_per_squad) = 15 := by
  sorry

end swords_per_orc_l2450_245056


namespace common_factor_proof_l2450_245079

theorem common_factor_proof (x y : ℝ) : ∃ (k : ℝ), 5*x^2 - 25*x^2*y = 5*x^2 * k :=
sorry

end common_factor_proof_l2450_245079


namespace prob_two_queens_or_two_jacks_standard_deck_l2450_245052

/-- A standard deck of cards. -/
structure Deck :=
  (total_cards : ℕ)
  (queens : ℕ)
  (jacks : ℕ)

/-- The probability of drawing either two queens or at least two jacks
    when selecting 3 cards randomly from a standard deck. -/
def prob_two_queens_or_two_jacks (d : Deck) : ℚ :=
  -- Definition to be proved
  74 / 850

/-- Theorem stating the probability of drawing either two queens or at least two jacks
    when selecting 3 cards randomly from a standard 52-card deck. -/
theorem prob_two_queens_or_two_jacks_standard_deck :
  prob_two_queens_or_two_jacks ⟨52, 4, 4⟩ = 74 / 850 := by
  sorry

end prob_two_queens_or_two_jacks_standard_deck_l2450_245052


namespace sandwich_ratio_l2450_245043

/-- The number of sandwiches Samson ate at lunch on Monday -/
def lunch_sandwiches : ℕ := 3

/-- The number of sandwiches Samson ate at dinner on Monday -/
def dinner_sandwiches : ℕ := 6

/-- The number of sandwiches Samson ate for breakfast on Tuesday -/
def tuesday_sandwiches : ℕ := 1

/-- The difference in total sandwiches eaten between Monday and Tuesday -/
def sandwich_difference : ℕ := 8

theorem sandwich_ratio :
  (dinner_sandwiches : ℚ) / lunch_sandwiches = 2 ∧
  lunch_sandwiches + dinner_sandwiches = tuesday_sandwiches + sandwich_difference :=
sorry

end sandwich_ratio_l2450_245043


namespace alice_prob_three_turns_correct_l2450_245049

/-- Represents the player who has the ball -/
inductive Player : Type
| Alice : Player
| Bob : Player

/-- The probability of keeping the ball for each player -/
def keep_prob (p : Player) : ℚ :=
  match p with
  | Player.Alice => 2/3
  | Player.Bob => 3/4

/-- The probability of tossing the ball for each player -/
def toss_prob (p : Player) : ℚ :=
  match p with
  | Player.Alice => 1/3
  | Player.Bob => 1/4

/-- The probability of Alice having the ball after three turns -/
def alice_prob_after_three_turns : ℚ := 203/432

theorem alice_prob_three_turns_correct :
  alice_prob_after_three_turns =
    keep_prob Player.Alice * keep_prob Player.Alice * keep_prob Player.Alice +
    toss_prob Player.Alice * toss_prob Player.Bob * keep_prob Player.Alice +
    keep_prob Player.Alice * toss_prob Player.Alice * toss_prob Player.Bob +
    toss_prob Player.Alice * keep_prob Player.Bob * toss_prob Player.Bob :=
by sorry

end alice_prob_three_turns_correct_l2450_245049


namespace distance_traveled_l2450_245071

/-- Given a person traveling at 65 km/hr for 3 hours, prove that the distance traveled is 195 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed = 65)
  (h2 : time = 3)
  (h3 : distance = speed * time) :
  distance = 195 := by
  sorry

end distance_traveled_l2450_245071


namespace room_tiling_l2450_245098

theorem room_tiling (room_length room_width : ℕ) 
  (border_tile_size inner_tile_size : ℕ) : 
  room_length = 16 → 
  room_width = 12 → 
  border_tile_size = 1 → 
  inner_tile_size = 2 → 
  (2 * (room_length - 2 + room_width - 2) + 4) + 
  ((room_length - 2) * (room_width - 2)) / (inner_tile_size ^ 2) = 87 :=
by
  sorry

end room_tiling_l2450_245098


namespace cube_volume_problem_l2450_245093

theorem cube_volume_problem (s : ℝ) : 
  s > 0 →
  (s + 2) * (s + 2) * (s - 2) = s^3 - 10 →
  s^3 = 27 := by
sorry

end cube_volume_problem_l2450_245093


namespace marble_probability_l2450_245096

/-- The probability of drawing either a green or purple marble from a bag -/
theorem marble_probability (green purple orange : ℕ) 
  (h_green : green = 5)
  (h_purple : purple = 4)
  (h_orange : orange = 6) :
  (green + purple : ℚ) / (green + purple + orange) = 3 / 5 := by
  sorry

end marble_probability_l2450_245096


namespace range_and_minimum_value_l2450_245083

def f (a x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + (a^2 - 2 * a + 2)

theorem range_and_minimum_value (a : ℝ) :
  (a = 1 → Set.range (fun x => f 1 x) ∩ Set.Icc 0 2 = Set.Icc 0 9) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 2 ∧ ∀ (y : ℝ), y ∈ Set.Icc 0 2 → f a x ≤ f a y) →
  (∀ (x : ℝ), x ∈ Set.Icc 0 2 → f a x ≥ 3) →
  (a = 1 - Real.sqrt 2 ∨ a = 5 + Real.sqrt 10) :=
sorry

end range_and_minimum_value_l2450_245083


namespace ratio_x_to_y_l2450_245068

theorem ratio_x_to_y (x y : ℝ) (h : y = x * (1 - 0.8333333333333334)) :
  x / y = 6 := by
  sorry

end ratio_x_to_y_l2450_245068


namespace machine_worked_three_minutes_l2450_245086

/-- An industrial machine that makes shirts -/
structure ShirtMachine where
  shirts_per_minute : ℕ
  shirts_made_yesterday : ℕ

/-- The number of minutes the machine worked yesterday -/
def minutes_worked_yesterday (machine : ShirtMachine) : ℕ :=
  machine.shirts_made_yesterday / machine.shirts_per_minute

/-- Theorem stating that the machine worked for 3 minutes yesterday -/
theorem machine_worked_three_minutes (machine : ShirtMachine) 
    (h1 : machine.shirts_per_minute = 3)
    (h2 : machine.shirts_made_yesterday = 9) : 
  minutes_worked_yesterday machine = 3 := by
  sorry

end machine_worked_three_minutes_l2450_245086


namespace not_value_preserving_g_value_preserving_f_condition_l2450_245072

/-- Definition of a value-preserving function on an interval [m, n] -/
def is_value_preserving (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  m < n ∧ 
  Monotone (fun x => f x) ∧
  Set.range (fun x => f x) = Set.Icc m n

/-- The function g(x) = x^2 - 2x -/
def g (x : ℝ) : ℝ := x^2 - 2*x

/-- The function f(x) = 2 + 1/a - 1/(a^2x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 + 1/a - 1/(a^2*x)

theorem not_value_preserving_g : ¬ is_value_preserving g 0 1 := by sorry

theorem value_preserving_f_condition (a : ℝ) :
  (∃ m n, is_value_preserving (f a) m n) ↔ (a > 1/2 ∨ a < -3/2) := by sorry

end not_value_preserving_g_value_preserving_f_condition_l2450_245072


namespace probability_of_successful_meeting_l2450_245024

/-- Friend's train arrival time in minutes after 1:00 -/
def FriendArrivalTime : Type := {t : ℝ // 0 ≤ t ∧ t ≤ 60}

/-- Alex's arrival time in minutes after 1:00 -/
def AlexArrivalTime : Type := {t : ℝ // 0 ≤ t ∧ t ≤ 120}

/-- The waiting time of the friend's train in minutes -/
def WaitingTime : ℝ := 10

/-- The event that Alex arrives while the friend's train is still at the station -/
def SuccessfulMeeting (f : FriendArrivalTime) (a : AlexArrivalTime) : Prop :=
  f.val ≤ a.val ∧ a.val ≤ f.val + WaitingTime

/-- The probability measure for the problem -/
noncomputable def P : Set (FriendArrivalTime × AlexArrivalTime) → ℝ := sorry

/-- The theorem stating the probability of a successful meeting -/
theorem probability_of_successful_meeting :
  P {p : FriendArrivalTime × AlexArrivalTime | SuccessfulMeeting p.1 p.2} = 1/4 := by sorry

end probability_of_successful_meeting_l2450_245024


namespace problem_statement_l2450_245084

def f (x : ℝ) := |2*x - 1|

theorem problem_statement (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : f a > f c) (h4 : f c > f b) : 
  2 - a < 2*c := by
sorry

end problem_statement_l2450_245084


namespace thirty_sixth_bead_is_white_l2450_245044

/-- Represents the color of a bead -/
inductive BeadColor
| Black
| White

/-- Defines the sequence of bead colors -/
def beadSequence : ℕ → BeadColor
| 0 => BeadColor.White
| n + 1 => match (n + 1) % 5 with
  | 1 => BeadColor.White
  | 2 => BeadColor.Black
  | 3 => BeadColor.White
  | 4 => BeadColor.Black
  | _ => BeadColor.White

/-- Theorem: The 36th bead in the sequence is white -/
theorem thirty_sixth_bead_is_white : beadSequence 35 = BeadColor.White := by
  sorry

end thirty_sixth_bead_is_white_l2450_245044


namespace quadratic_equation_value_l2450_245063

theorem quadratic_equation_value (y : ℝ) (h : y = 4) : 3 * y^2 + 4 * y + 2 = 66 := by
  sorry

end quadratic_equation_value_l2450_245063


namespace foil_covered_prism_width_l2450_245002

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The inner core of the prism not touching tin foil -/
def inner : PrismDimensions :=
  { length := 2^(5/3),
    width := 2^(8/3),
    height := 2^(5/3) }

/-- The outer prism covered in tin foil -/
def outer : PrismDimensions :=
  { length := inner.length + 2,
    width := inner.width + 2,
    height := inner.height + 2 }

theorem foil_covered_prism_width :
  (inner.length * inner.width * inner.height = 128) →
  (inner.width = 2 * inner.length) →
  (inner.width = 2 * inner.height) →
  (outer.width = 10) := by
  sorry

end foil_covered_prism_width_l2450_245002


namespace sin_2alpha_value_l2450_245069

theorem sin_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.tan (α + π/4) = 3 * Real.cos (2*α)) : 
  Real.sin (2*α) = 2/3 := by
sorry

end sin_2alpha_value_l2450_245069


namespace leadership_selection_ways_l2450_245042

/-- The number of ways to choose a president, vice-president, and committee from a group. -/
def choose_leadership (n : ℕ) : ℕ :=
  n * (n - 1) * (Nat.choose (n - 2) 3)

/-- The problem statement as a theorem. -/
theorem leadership_selection_ways :
  choose_leadership 10 = 5040 := by
  sorry

end leadership_selection_ways_l2450_245042


namespace farm_cows_l2450_245021

/-- Given a farm with cows and horses, prove the number of cows -/
theorem farm_cows (total_horses : ℕ) (ratio_cows : ℕ) (ratio_horses : ℕ) 
  (h1 : total_horses = 6)
  (h2 : ratio_cows = 7)
  (h3 : ratio_horses = 2) :
  (ratio_cows : ℚ) / ratio_horses * total_horses = 21 := by
  sorry

end farm_cows_l2450_245021


namespace bagel_savings_theorem_l2450_245008

/-- The cost of a single bagel in dollars -/
def single_bagel_cost : ℚ := 2.25

/-- The cost of a dozen bagels in dollars -/
def dozen_bagels_cost : ℚ := 24

/-- The number of bagels in a dozen -/
def dozen : ℕ := 12

/-- The savings per bagel in cents when buying a dozen -/
def savings_per_bagel : ℚ :=
  ((single_bagel_cost * dozen - dozen_bagels_cost) / dozen) * 100

theorem bagel_savings_theorem :
  savings_per_bagel = 25 := by sorry

end bagel_savings_theorem_l2450_245008


namespace vector_problem_l2450_245009

/-- Given three vectors a, b, c in ℝ² -/
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 3)
def c : ℝ → ℝ × ℝ := λ m ↦ (-2, m)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

/-- Main theorem combining both parts of the problem -/
theorem vector_problem :
  (∃ m : ℝ, perpendicular a (b + c m) → m = -1) ∧
  (∃ k : ℝ, collinear (k • a + b) (2 • a - b) → k = -2) := by
  sorry

end vector_problem_l2450_245009


namespace wheat_mixture_profit_percentage_l2450_245038

/-- Calculates the profit percentage for a wheat mixture sale --/
theorem wheat_mixture_profit_percentage
  (wheat1_weight : ℝ)
  (wheat1_price : ℝ)
  (wheat2_weight : ℝ)
  (wheat2_price : ℝ)
  (selling_price : ℝ)
  (h1 : wheat1_weight = 30)
  (h2 : wheat1_price = 11.5)
  (h3 : wheat2_weight = 20)
  (h4 : wheat2_price = 14.25)
  (h5 : selling_price = 15.75) :
  let total_cost := wheat1_weight * wheat1_price + wheat2_weight * wheat2_price
  let total_weight := wheat1_weight + wheat2_weight
  let total_selling_price := total_weight * selling_price
  let profit := total_selling_price - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage = 25 := by
    sorry

end wheat_mixture_profit_percentage_l2450_245038


namespace inequality_solution_set_l2450_245059

theorem inequality_solution_set (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (a - x) * (x - 1/a) > 0} = {x : ℝ | a < x ∧ x < 1/a} := by
  sorry

end inequality_solution_set_l2450_245059


namespace competition_theorem_l2450_245075

/-- Represents a team in the competition -/
inductive Team
| A
| B
| E

/-- Represents an event in the competition -/
inductive Event
| Vaulting
| GrenadeThrowingv
| Other1
| Other2
| Other3

/-- Represents a place in an event -/
inductive Place
| First
| Second
| Third

/-- The scoring system for the competition -/
structure ScoringSystem where
  first : ℕ
  second : ℕ
  third : ℕ
  first_gt_second : first > second
  second_gt_third : second > third
  third_pos : third > 0

/-- The result of a single event -/
structure EventResult where
  event : Event
  first : Team
  second : Team
  third : Team

/-- The final scores of the teams -/
structure FinalScores where
  team_A : ℕ
  team_B : ℕ
  team_E : ℕ

/-- The competition results -/
structure CompetitionResults where
  scoring : ScoringSystem
  events : List EventResult
  scores : FinalScores

/-- The main theorem to prove -/
theorem competition_theorem (r : CompetitionResults) : 
  r.scores.team_A = 22 ∧ 
  r.scores.team_B = 9 ∧ 
  r.scores.team_E = 9 ∧
  (∃ e : EventResult, e ∈ r.events ∧ e.event = Event.Vaulting ∧ e.first = Team.B) →
  r.events.length = 5 ∧
  (∃ e : EventResult, e ∈ r.events ∧ e.event = Event.GrenadeThrowingv ∧ e.second = Team.B) :=
by sorry

end competition_theorem_l2450_245075


namespace robbers_river_crossing_impossibility_l2450_245045

theorem robbers_river_crossing_impossibility :
  ∀ (n : ℕ) (trips : ℕ → ℕ → Prop),
    n = 40 →
    (∀ i j, i < n → j < n → i ≠ j → (trips i j ∨ trips j i)) →
    (∀ i j k, i < n → j < n → k < n → i ≠ j → j ≠ k → i ≠ k →
      ¬(trips i j ∧ trips i k)) →
    False :=
by
  sorry

end robbers_river_crossing_impossibility_l2450_245045


namespace nonagon_diagonal_intersections_nonagon_intersections_eq_choose_four_l2450_245012

/-- The number of intersection points of diagonals in a regular nonagon -/
theorem nonagon_diagonal_intersections : ℕ := by
  -- Define a regular nonagon
  sorry

/-- The number of ways to choose 4 vertices from 9 vertices -/
def choose_four_from_nine : ℕ := Nat.choose 9 4

/-- Theorem: The number of distinct interior points where two or more diagonals
    intersect in a regular nonagon is equal to choose_four_from_nine -/
theorem nonagon_intersections_eq_choose_four :
  nonagon_diagonal_intersections = choose_four_from_nine := by
  sorry

end nonagon_diagonal_intersections_nonagon_intersections_eq_choose_four_l2450_245012


namespace aaron_gave_five_sweets_l2450_245085

/-- Represents the number of sweets given to a friend -/
def sweets_given_to_friend (initial_cherry initial_strawberry initial_pineapple : ℕ) 
  (remaining : ℕ) : ℕ :=
  initial_cherry / 2 + initial_strawberry / 2 + initial_pineapple / 2 - remaining

/-- Proves that Aaron gave 5 cherry sweets to his friend -/
theorem aaron_gave_five_sweets : 
  sweets_given_to_friend 30 40 50 55 = 5 := by
  sorry

#eval sweets_given_to_friend 30 40 50 55

end aaron_gave_five_sweets_l2450_245085


namespace problem_solution_l2450_245065

theorem problem_solution : 
  (2008^2 - 2007 * 2009 = 1) ∧ 
  ((-0.125)^2011 * 8^2010 = -0.125) := by
sorry

end problem_solution_l2450_245065


namespace quadratic_root_k_value_l2450_245001

theorem quadratic_root_k_value (k : ℝ) :
  (∃ x : ℝ, 2 * x^2 + 3 * x - k = 0) ∧ (2 * 4^2 + 3 * 4 - k = 0) → k = 44 := by
  sorry

end quadratic_root_k_value_l2450_245001


namespace parabola_point_coordinates_l2450_245028

/-- Theorem: For a point P(x, y) on the parabola y² = 4x, if its distance from the focus is 4, then x = 3 and y = ±2√3 -/
theorem parabola_point_coordinates (x y : ℝ) :
  y^2 = 4*x →                           -- P is on the parabola y² = 4x
  (x - 1)^2 + y^2 = 16 →                -- Distance from P to focus (1, 0) is 4
  (x = 3 ∧ y = 2*Real.sqrt 3 ∨ y = -2*Real.sqrt 3) :=
by sorry


end parabola_point_coordinates_l2450_245028


namespace subset_implies_m_range_l2450_245050

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem subset_implies_m_range (m : ℝ) : B m ⊆ A → m ≤ 3 := by
  sorry

end subset_implies_m_range_l2450_245050


namespace original_equals_scientific_l2450_245081

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 1300000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { coefficient := 1.3
    exponent := 6
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end original_equals_scientific_l2450_245081


namespace expression_one_equality_l2450_245092

theorem expression_one_equality : 2 - (-4) + 8 / (-2) + (-3) = -1 := by sorry

end expression_one_equality_l2450_245092


namespace boxes_theorem_l2450_245080

def boxes_problem (total : ℕ) (markers : ℕ) (erasers : ℕ) (both : ℕ) : Prop :=
  let neither := total - (markers + erasers - both)
  neither = 3

theorem boxes_theorem : boxes_problem 12 7 5 3 := by
  sorry

end boxes_theorem_l2450_245080


namespace rational_equation_result_l2450_245078

theorem rational_equation_result (x y : ℚ) 
  (h : |2*x - 3*y + 1| + (x + 3*y + 5)^2 = 0) : 
  (-2*x*y)^2 * (-y^2) * 6*x*y^2 = 192 := by
sorry

end rational_equation_result_l2450_245078


namespace perfect_square_pairs_l2450_245041

theorem perfect_square_pairs (m n : ℤ) :
  (∃ a : ℤ, m^2 + n = a^2) ∧ (∃ b : ℤ, n^2 + m = b^2) →
  (m = 0 ∧ ∃ k : ℤ, n = k^2) ∨
  (n = 0 ∧ ∃ k : ℤ, m = k^2) ∨
  (m = 1 ∧ n = -1) ∨
  (m = -1 ∧ n = 1) := by
sorry

end perfect_square_pairs_l2450_245041


namespace factor_between_l2450_245006

theorem factor_between (n a b : ℕ) (hn : n > 10) (ha : a > 0) (hb : b > 0) 
  (hab : a ≠ b) (hdiv_a : a ∣ n) (hdiv_b : b ∣ n) (heq : n = a^2 + b) : 
  ∃ k : ℕ, k ∣ n ∧ a < k ∧ k < b := by
  sorry

end factor_between_l2450_245006


namespace tic_tac_toe_probability_l2450_245034

/-- Represents a tic-tac-toe board -/
def TicTacToeBoard := Fin 3 → Fin 3 → Bool

/-- The number of cells in a tic-tac-toe board -/
def boardSize : Nat := 9

/-- The number of noughts on the board -/
def noughtsCount : Nat := 3

/-- The number of crosses on the board -/
def crossesCount : Nat := 6

/-- The number of ways to choose noughts positions -/
def totalPositions : Nat := Nat.choose boardSize noughtsCount

/-- The number of winning positions for noughts -/
def winningPositions : Nat := 8

/-- Theorem: The probability of 3 noughts being in a winning position is 2/21 -/
theorem tic_tac_toe_probability : 
  (winningPositions : ℚ) / totalPositions = 2 / 21 := by
  sorry

end tic_tac_toe_probability_l2450_245034


namespace max_value_inequality_equality_case_l2450_245025

theorem max_value_inequality (a : ℝ) : (∀ x > 1, x + 1 / (x - 1) ≥ a) → a ≤ 3 := by sorry

theorem equality_case : ∃ x > 1, x + 1 / (x - 1) = 3 := by sorry

end max_value_inequality_equality_case_l2450_245025


namespace ratio_xyz_l2450_245039

theorem ratio_xyz (x y z : ℚ) 
  (h1 : (3/4) * y = (1/2) * x) 
  (h2 : (3/10) * x = (1/5) * z) : 
  ∃ (k : ℚ), k > 0 ∧ x = 6*k ∧ y = 4*k ∧ z = 9*k := by
sorry

end ratio_xyz_l2450_245039


namespace alex_grocery_delivery_l2450_245007

/-- Alex's grocery delivery problem -/
theorem alex_grocery_delivery 
  (savings : ℝ) 
  (car_cost : ℝ) 
  (trip_charge : ℝ) 
  (grocery_percentage : ℝ) 
  (num_trips : ℕ) 
  (h1 : savings = 14500)
  (h2 : car_cost = 14600)
  (h3 : trip_charge = 1.5)
  (h4 : grocery_percentage = 0.05)
  (h5 : num_trips = 40)
  : ∃ (grocery_worth : ℝ), 
    trip_charge * num_trips + grocery_percentage * grocery_worth = car_cost - savings ∧ 
    grocery_worth = 800 := by
  sorry

end alex_grocery_delivery_l2450_245007


namespace range_of_f_l2450_245066

def f (x : ℤ) : ℤ := x^2 - 1

def domain : Set ℤ := {-1, 0, 1, 2}

theorem range_of_f :
  {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end range_of_f_l2450_245066


namespace common_chord_length_l2450_245011

/-- Given two circles with radius 12 and centers 16 units apart,
    the length of their common chord is 8√5. -/
theorem common_chord_length (r : ℝ) (d : ℝ) (h1 : r = 12) (h2 : d = 16) :
  let chord_length := 2 * Real.sqrt (r^2 - (d/2)^2)
  chord_length = 8 * Real.sqrt 5 := by
sorry

end common_chord_length_l2450_245011


namespace circle_intersection_range_l2450_245027

theorem circle_intersection_range (r : ℝ) : 
  (∃ (x y : ℝ), x^2 + (y - 1)^2 = r^2 ∧ r > 0 ∧
   ∃ (x' y' : ℝ), (x' - 2)^2 + (y' - 1)^2 = 1 ∧
   x' = y ∧ y' = x) →
  r ∈ Set.Icc (Real.sqrt 2 - 1) (Real.sqrt 2 + 1) :=
by sorry

end circle_intersection_range_l2450_245027


namespace matrix_power_zero_l2450_245000

open Matrix Complex

theorem matrix_power_zero (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℂ) 
  (h1 : A * B = B * A)
  (h2 : B.det ≠ 0)
  (h3 : ∀ z : ℂ, Complex.abs z = 1 → Complex.abs ((A + z • B).det) = 1) :
  A ^ n = 0 := by
  sorry

end matrix_power_zero_l2450_245000


namespace interest_equivalence_l2450_245030

/-- Simple interest calculation function -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- The problem statement -/
theorem interest_equivalence (P : ℝ) : 
  simple_interest 100 0.05 8 = simple_interest P 0.10 2 → P = 200 := by
  sorry

end interest_equivalence_l2450_245030


namespace g_of_3_l2450_245037

def g (x : ℝ) : ℝ := 5 * x^3 + 7 * x^2 - 3 * x - 6

theorem g_of_3 : g 3 = 183 := by
  sorry

end g_of_3_l2450_245037


namespace modified_chessboard_no_tiling_l2450_245026

/-- Represents a chessboard cell --/
inductive Cell
| White
| Black

/-- Represents a 2x1 tile --/
structure Tile :=
  (first : Cell)
  (second : Cell)

/-- Represents the modified chessboard --/
def ModifiedChessboard : Type :=
  Fin 8 → Fin 8 → Option Cell

/-- A valid 2x1 tile covers one white and one black cell --/
def isValidTile (t : Tile) : Prop :=
  (t.first = Cell.White ∧ t.second = Cell.Black) ∨
  (t.first = Cell.Black ∧ t.second = Cell.White)

/-- A tiling of the modified chessboard --/
def Tiling : Type :=
  List Tile

/-- Checks if a tiling is valid for the modified chessboard --/
def isValidTiling (t : Tiling) (mb : ModifiedChessboard) : Prop :=
  sorry

theorem modified_chessboard_no_tiling :
  ∀ (mb : ModifiedChessboard),
    (mb 0 0 = none) →  -- Bottom-left square removed
    (mb 7 7 = none) →  -- Top-right square removed
    (∀ i j, i ≠ 0 ∨ j ≠ 0 → i ≠ 7 ∨ j ≠ 7 → mb i j ≠ none) →  -- All other squares present
    (∀ i j, (i + j) % 2 = 0 → mb i j = some Cell.White) →  -- White cells
    (∀ i j, (i + j) % 2 = 1 → mb i j = some Cell.Black) →  -- Black cells
    ¬∃ (t : Tiling), isValidTiling t mb :=
by
  sorry

end modified_chessboard_no_tiling_l2450_245026


namespace geometric_sequence_sum_l2450_245057

/-- Given a sequence {a_n} where the sum of the first n terms is S_n = 3 * 2^n + k,
    prove that if {a_n} is a geometric sequence, then k = -3. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℝ) :
  (∀ n, S n = 3 * 2^n + k) →
  (∀ n, a n = S n - S (n-1)) →
  (∀ n, n ≥ 2 → a n * a (n-2) = (a (n-1))^2) →
  k = -3 := by
sorry

end geometric_sequence_sum_l2450_245057


namespace smallest_nonprime_with_large_factors_l2450_245053

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def has_no_prime_factor_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < 20 → ¬(n % p = 0)

theorem smallest_nonprime_with_large_factors :
  ∃ n : ℕ, n > 1 ∧ ¬(is_prime n) ∧ has_no_prime_factor_less_than_20 n ∧
  (∀ m : ℕ, m > 1 → ¬(is_prime m) → has_no_prime_factor_less_than_20 m → m ≥ n) ∧
  n = 529 :=
sorry

end smallest_nonprime_with_large_factors_l2450_245053


namespace unique_prime_p_l2450_245016

theorem unique_prime_p : ∃! p : ℕ, 
  Nat.Prime p ∧ Nat.Prime (3 * p^2 + 1) :=
by sorry

end unique_prime_p_l2450_245016


namespace arithmetic_sequence_property_l2450_245046

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_property 
  (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 5 + a 6 = 20) : 
  (a 4 + a 7) / 2 = 10 := by
  sorry

end arithmetic_sequence_property_l2450_245046


namespace thirty_percent_less_than_hundred_is_one_fourth_more_than_fiftysix_l2450_245031

theorem thirty_percent_less_than_hundred_is_one_fourth_more_than_fiftysix : ∃ x : ℝ, 
  (100 - 0.3 * 100 = x + 0.25 * x) ∧ x = 56 := by
  sorry

end thirty_percent_less_than_hundred_is_one_fourth_more_than_fiftysix_l2450_245031


namespace line_equation_through_point_with_inclination_l2450_245018

/-- Proves that the equation of a line passing through point (2, -1) with an inclination angle of π/4 is x - y - 3 = 0 -/
theorem line_equation_through_point_with_inclination (x y : ℝ) :
  let point : ℝ × ℝ := (2, -1)
  let inclination : ℝ := π / 4
  let slope : ℝ := Real.tan inclination
  (y - point.2 = slope * (x - point.1)) → (x - y - 3 = 0) :=
by
  sorry

end line_equation_through_point_with_inclination_l2450_245018


namespace x_can_be_any_real_value_l2450_245074

theorem x_can_be_any_real_value
  (x y z w : ℝ)
  (h1 : x / y > z / w)
  (h2 : y ≠ 0 ∧ w ≠ 0)
  (h3 : y * w > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b < 0 ∧ c = 0 ∧
    (x = a ∨ x = b ∨ x = c) :=
sorry

end x_can_be_any_real_value_l2450_245074


namespace E_equals_three_iff_x_equals_y_infinite_solutions_exist_l2450_245091

def E (x y : ℕ) : ℚ :=
  x / y + (x + 1) / (y + 1) + (x + 2) / (y + 2)

theorem E_equals_three_iff_x_equals_y (x y : ℕ) :
  E x y = 3 ↔ x = y :=
sorry

theorem infinite_solutions_exist (k : ℕ) :
  ∃ x y : ℕ, E x y = 11 * k + 3 :=
sorry

end E_equals_three_iff_x_equals_y_infinite_solutions_exist_l2450_245091
