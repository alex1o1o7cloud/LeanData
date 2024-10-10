import Mathlib

namespace school_play_seating_l2367_236793

theorem school_play_seating (rows : ℕ) (chairs_per_row : ℕ) (unoccupied : ℕ) : 
  rows = 40 → chairs_per_row = 20 → unoccupied = 10 → 
  rows * chairs_per_row - unoccupied = 790 := by
  sorry

end school_play_seating_l2367_236793


namespace unique_solution_sqrt_equation_l2367_236796

theorem unique_solution_sqrt_equation :
  ∀ x y : ℕ,
    x ≥ 1 →
    y ≥ 1 →
    y ≥ x →
    (Real.sqrt (2 * x) - 1) * (Real.sqrt (2 * y) - 1) = 1 →
    x = 2 ∧ y = 2 := by
  sorry

end unique_solution_sqrt_equation_l2367_236796


namespace sphere_surface_area_l2367_236705

theorem sphere_surface_area (V : ℝ) (r : ℝ) (A : ℝ) : 
  V = 72 * Real.pi → 
  V = (4/3) * Real.pi * r^3 → 
  A = 4 * Real.pi * r^2 → 
  A = 36 * Real.pi * 2^(2/3) := by
sorry

end sphere_surface_area_l2367_236705


namespace trapezoid_bc_length_l2367_236788

/-- Represents a trapezoid ABCD with given properties -/
structure Trapezoid where
  area : ℝ
  altitude : ℝ
  ab : ℝ
  cd : ℝ

/-- Theorem stating the length of BC in the trapezoid -/
theorem trapezoid_bc_length (t : Trapezoid) 
  (h_area : t.area = 180)
  (h_altitude : t.altitude = 8)
  (h_ab : t.ab = 14)
  (h_cd : t.cd = 20) :
  ∃ (bc : ℝ), bc = 22.5 - Real.sqrt 33 - 2 * Real.sqrt 21 := by
  sorry

end trapezoid_bc_length_l2367_236788


namespace rope_folding_l2367_236752

theorem rope_folding (n : ℕ) (original_length : ℝ) (h : n = 3) :
  let num_parts := 2^n
  let part_length := original_length / num_parts
  part_length = (1 / 8) * original_length := by
  sorry

end rope_folding_l2367_236752


namespace left_handed_fraction_conference_l2367_236740

/-- Represents the fraction of left-handed participants for each country type -/
structure LeftHandedFractions where
  red : ℚ
  blue : ℚ
  green : ℚ
  yellow : ℚ

/-- Represents the ratio of participants from each country type -/
structure ParticipantRatio where
  red : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the fraction of left-handed participants given the ratio of participants
    and the fractions of left-handed participants for each country type -/
def leftHandedFraction (ratio : ParticipantRatio) (fractions : LeftHandedFractions) : ℚ :=
  (ratio.red * fractions.red + ratio.blue * fractions.blue +
   ratio.green * fractions.green + ratio.yellow * fractions.yellow) /
  (ratio.red + ratio.blue + ratio.green + ratio.yellow)

theorem left_handed_fraction_conference :
  let ratio : ParticipantRatio := ⟨10, 5, 3, 2⟩
  let fractions : LeftHandedFractions := ⟨37/100, 61/100, 26/100, 48/100⟩
  leftHandedFraction ratio fractions = 849/2000 := by
  sorry

end left_handed_fraction_conference_l2367_236740


namespace max_cables_cut_theorem_l2367_236730

/-- Represents a computer network with computers and cables -/
structure ComputerNetwork where
  num_computers : Nat
  num_cables : Nat
  num_clusters : Nat

/-- The initial state of the computer network -/
def initial_network : ComputerNetwork :=
  { num_computers := 200
  , num_cables := 345
  , num_clusters := 1 }

/-- The final state of the computer network after cable cutting -/
def final_network : ComputerNetwork :=
  { num_computers := 200
  , num_cables := initial_network.num_cables - 153
  , num_clusters := 8 }

/-- The maximum number of cables that can be cut -/
def max_cables_cut : Nat := 153

/-- Theorem stating the maximum number of cables that can be cut -/
theorem max_cables_cut_theorem :
  max_cables_cut = initial_network.num_cables - final_network.num_cables ∧
  final_network.num_clusters = 8 ∧
  final_network.num_cables ≥ final_network.num_computers - final_network.num_clusters :=
by sorry


end max_cables_cut_theorem_l2367_236730


namespace l_shaped_figure_perimeter_l2367_236718

/-- Represents an L-shaped figure formed by a 3x3 square with a 2x2 square attached to one side -/
structure LShapedFigure :=
  (base_side : ℕ)
  (extension_side : ℕ)
  (unit_length : ℝ)
  (h_base : base_side = 3)
  (h_extension : extension_side = 2)
  (h_unit : unit_length = 1)

/-- Calculates the perimeter of the L-shaped figure -/
def perimeter (figure : LShapedFigure) : ℝ :=
  2 * (figure.base_side + figure.extension_side + figure.base_side) * figure.unit_length

/-- Theorem stating that the perimeter of the L-shaped figure is 15 units -/
theorem l_shaped_figure_perimeter :
  ∀ (figure : LShapedFigure), perimeter figure = 15 := by
  sorry

end l_shaped_figure_perimeter_l2367_236718


namespace incorrect_to_correct_calculation_l2367_236744

theorem incorrect_to_correct_calculation (x : ℝ) : x * 3 - 5 = 103 → (x / 3) - 5 = 7 := by
  sorry

end incorrect_to_correct_calculation_l2367_236744


namespace parallel_vectors_difference_magnitude_l2367_236766

theorem parallel_vectors_difference_magnitude :
  ∀ x : ℝ,
  let a : Fin 2 → ℝ := ![1, x]
  let b : Fin 2 → ℝ := ![2*x + 3, -x]
  (∃ (k : ℝ), a = k • b) →
  ‖a - b‖ = 2 ∨ ‖a - b‖ = 2 * Real.sqrt 5 :=
by sorry

end parallel_vectors_difference_magnitude_l2367_236766


namespace units_digit_of_k_squared_plus_two_to_k_l2367_236771

/-- Given k = 2012² + 2^2014, prove that (k² + 2^k) mod 10 = 5 -/
theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ) : k = 2012^2 + 2^2014 → (k^2 + 2^k) % 10 = 5 := by
  sorry

end units_digit_of_k_squared_plus_two_to_k_l2367_236771


namespace distance_to_focus_l2367_236716

-- Define the parabola C: y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define point P
def P : ℝ × ℝ := (-2, 0)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define points A and B as intersection points
axiom A : ℝ × ℝ
axiom B : ℝ × ℝ

-- A and B are on the parabola
axiom A_on_parabola : parabola A.1 A.2
axiom B_on_parabola : parabola B.1 B.2

-- A, B, and P are collinear
axiom collinear : ∃ (t : ℝ), B.1 - P.1 = t * (A.1 - P.1) ∧ B.2 - P.2 = t * (A.2 - P.2)

-- |PA| = 1/2 |AB|
axiom distance_relation : Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) = 1/2 * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Theorem to prove
theorem distance_to_focus :
  Real.sqrt ((A.1 - focus.1)^2 + (A.2 - focus.2)^2) = 5/3 := by sorry

end distance_to_focus_l2367_236716


namespace luke_bought_twelve_stickers_l2367_236757

/-- The number of stickers Luke bought from the store -/
def stickers_bought (initial : ℕ) (birthday : ℕ) (given_away : ℕ) (used : ℕ) (remaining : ℕ) : ℕ :=
  remaining + given_away + used - initial - birthday

/-- Theorem stating that Luke bought 12 stickers from the store -/
theorem luke_bought_twelve_stickers :
  stickers_bought 20 20 5 8 39 = 12 := by
  sorry

end luke_bought_twelve_stickers_l2367_236757


namespace arrangement_theorem_l2367_236729

/-- The number of permutations of n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k objects from n distinct objects. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to arrange 7 students with Student A not at either end. -/
def arrangement_count_1 : ℕ := 5 * permutations 6

/-- The number of ways to arrange 7 students with Student A not on the left end
    and Student B not on the right end. -/
def arrangement_count_2 : ℕ := permutations 6 + choose 5 1 * choose 5 1 * permutations 5

theorem arrangement_theorem :
  (arrangement_count_1 = 5 * permutations 6) ∧
  (arrangement_count_2 = permutations 6 + choose 5 1 * choose 5 1 * permutations 5) :=
by sorry

end arrangement_theorem_l2367_236729


namespace polynomial_expansion_l2367_236703

/-- Proves the equality of the expanded polynomial expression -/
theorem polynomial_expansion (y : ℝ) : 
  (3 * y + 2) * (5 * y^12 - y^11 + 3 * y^10 + 2) = 
  15 * y^13 + 7 * y^12 + 7 * y^11 + 6 * y^10 + 6 * y + 4 := by
  sorry

end polynomial_expansion_l2367_236703


namespace sam_and_dan_balloons_l2367_236727

/-- The number of red balloons Sam and Dan have in total -/
def total_balloons (sam_initial : ℝ) (sam_given : ℝ) (dan : ℝ) : ℝ :=
  (sam_initial - sam_given) + dan

/-- Theorem stating the total number of red balloons Sam and Dan have -/
theorem sam_and_dan_balloons :
  total_balloons 46.0 10.0 16.0 = 52.0 := by
  sorry

end sam_and_dan_balloons_l2367_236727


namespace water_needed_l2367_236765

/-- Represents the recipe for lemonade tea --/
structure LemonadeTea where
  lemonJuice : ℝ
  sugar : ℝ
  water : ℝ
  tea : ℝ

/-- Checks if the recipe satisfies the given conditions --/
def isValidRecipe (recipe : LemonadeTea) : Prop :=
  recipe.water = 3 * recipe.sugar ∧
  recipe.sugar = 1.5 * recipe.lemonJuice ∧
  recipe.tea = (recipe.water + recipe.sugar + recipe.lemonJuice) / 6 ∧
  recipe.lemonJuice = 4

/-- Theorem stating that a valid recipe requires 18 cups of water --/
theorem water_needed (recipe : LemonadeTea) 
  (h : isValidRecipe recipe) : recipe.water = 18 := by
  sorry


end water_needed_l2367_236765


namespace max_x5_value_l2367_236747

theorem max_x5_value (x₁ x₂ x₃ x₄ x₅ : ℕ+) 
  (h : x₁ + x₂ + x₃ + x₄ + x₅ = x₁ * x₂ * x₃ * x₄ * x₅) : 
  x₅ ≤ 5 ∧ ∃ (a b c d : ℕ+), a + b + c + d + 5 = a * b * c * d * 5 := by
  sorry

end max_x5_value_l2367_236747


namespace museum_admission_difference_l2367_236753

theorem museum_admission_difference (men women free_admission : ℕ) 
  (h1 : men = 194)
  (h2 : women = 235)
  (h3 : free_admission = 68) :
  (men + women) - free_admission - free_admission = 293 := by
  sorry

end museum_admission_difference_l2367_236753


namespace triangle_area_l2367_236701

theorem triangle_area (a b c : ℝ) (h1 : a = 17) (h2 : b = 144) (h3 : c = 145) :
  (1/2) * a * b = 1224 :=
by sorry

end triangle_area_l2367_236701


namespace class_average_problem_l2367_236778

theorem class_average_problem (total_students : Nat) (high_scorers : Nat) (zero_scorers : Nat)
  (high_score : Nat) (class_average : Rat) :
  total_students = 27 →
  high_scorers = 5 →
  zero_scorers = 3 →
  high_score = 95 →
  class_average = 49.25925925925926 →
  let remaining_students := total_students - high_scorers - zero_scorers
  let total_score := class_average * total_students
  let high_scorers_total := high_scorers * high_score
  (total_score - high_scorers_total) / remaining_students = 45 := by
  sorry

end class_average_problem_l2367_236778


namespace bamboo_volume_proof_l2367_236720

/-- An arithmetic sequence of 9 terms -/
def ArithmeticSequence (a : Fin 9 → ℚ) : Prop :=
  ∃ d : ℚ, ∀ i j : Fin 9, a j - a i = (j - i : ℤ) • d

theorem bamboo_volume_proof (a : Fin 9 → ℚ) 
  (h_arith : ArithmeticSequence a)
  (h_bottom : a 0 + a 1 + a 2 = 4)
  (h_top : a 5 + a 6 + a 7 + a 8 = 3) :
  a 3 + a 4 = 2 + 3/22 := by
  sorry

end bamboo_volume_proof_l2367_236720


namespace maria_score_is_15_l2367_236770

/-- Represents a quiz result -/
structure QuizResult where
  total_questions : Nat
  correct_answers : Nat
  incorrect_answers : Nat
  unanswered_questions : Nat
  deriving Repr

/-- Calculates the score for a quiz result -/
def calculate_score (result : QuizResult) : Nat :=
  result.correct_answers

/-- Maria's quiz result -/
def maria_result : QuizResult :=
  { total_questions := 20
  , correct_answers := 15
  , incorrect_answers := 3
  , unanswered_questions := 2
  }

theorem maria_score_is_15 :
  calculate_score maria_result = 15 ∧
  maria_result.total_questions = maria_result.correct_answers + maria_result.incorrect_answers + maria_result.unanswered_questions :=
by sorry

end maria_score_is_15_l2367_236770


namespace system_of_equations_solution_l2367_236762

theorem system_of_equations_solution :
  ∃! (x y : ℝ), 4*x + 3*y = 6.4 ∧ 5*x - 6*y = -1.5 ∧ x = 11.3/13 ∧ y = 2.9232/3 := by
  sorry

end system_of_equations_solution_l2367_236762


namespace total_crayons_l2367_236773

/-- The number of crayons each person has -/
structure CrayonCounts where
  wanda : ℕ
  dina : ℕ
  jacob : ℕ
  emma : ℕ
  xavier : ℕ
  hannah : ℕ

/-- The conditions of the problem -/
def crayon_problem (c : CrayonCounts) : Prop :=
  c.wanda = 62 ∧
  c.dina = 28 ∧
  c.jacob = c.dina - 2 ∧
  c.emma = 2 * c.wanda - 3 ∧
  c.xavier = ((c.jacob + c.dina) / 2) ^ 3 - 7 ∧
  c.hannah = (c.wanda + c.dina + c.jacob + c.emma + c.xavier) / 5

/-- The theorem to be proved -/
theorem total_crayons (c : CrayonCounts) : 
  crayon_problem c → c.wanda + c.dina + c.jacob + c.emma + c.xavier + c.hannah = 23895 := by
  sorry


end total_crayons_l2367_236773


namespace proposition_implications_l2367_236722

theorem proposition_implications (p q : Prop) 
  (h : ¬(¬p ∨ ¬q)) : (p ∧ q) ∧ (p ∨ q) := by
  sorry

end proposition_implications_l2367_236722


namespace proposition_analysis_l2367_236706

theorem proposition_analysis (m n : ℝ) : 
  (¬ (((m ≤ 0) ∨ (n ≤ 0)) → (m + n ≤ 0))) ∧ 
  ((m + n ≤ 0) → ((m ≤ 0) ∨ (n ≤ 0))) ∧
  (((m > 0) ∧ (n > 0)) → (m + n > 0)) ∧
  (¬ ((m + n > 0) → ((m > 0) ∧ (n > 0)))) ∧
  (((m + n ≤ 0) → ((m ≤ 0) ∨ (n ≤ 0))) ∧ ¬(((m ≤ 0) ∨ (n ≤ 0)) → (m + n ≤ 0))) :=
by sorry

end proposition_analysis_l2367_236706


namespace function_set_property_l2367_236783

/-- A set of functions from ℝ to ℝ satisfying a specific property -/
def FunctionSet : Type := {A : Set (ℝ → ℝ) // 
  ∀ (f₁ f₂ : ℝ → ℝ), f₁ ∈ A → f₂ ∈ A → 
    ∃ (f₃ : ℝ → ℝ), f₃ ∈ A ∧ 
      ∀ (x y : ℝ), f₁ (f₂ y - x) + 2 * x = f₃ (x + y)}

/-- The main theorem -/
theorem function_set_property (A : FunctionSet) :
  ∀ (f : ℝ → ℝ), f ∈ A.val → ∀ (x : ℝ), f (x - f x) = 0 := by
  sorry

end function_set_property_l2367_236783


namespace quadratic_one_solution_l2367_236710

theorem quadratic_one_solution (b : ℝ) (hb : b ≠ 0) :
  (∃! x, 3 * x^2 + b * x + 12 = 0) →
  ((b = 12 ∧ ∃ x, 3 * x^2 + b * x + 12 = 0 ∧ x = -2) ∨
   (b = -12 ∧ ∃ x, 3 * x^2 + b * x + 12 = 0 ∧ x = 2)) :=
by sorry

end quadratic_one_solution_l2367_236710


namespace remaining_amount_after_buying_folders_l2367_236786

def initial_amount : ℕ := 19
def folder_cost : ℕ := 2

theorem remaining_amount_after_buying_folders :
  initial_amount - (initial_amount / folder_cost * folder_cost) = 1 := by
sorry

end remaining_amount_after_buying_folders_l2367_236786


namespace hyperbola_condition_l2367_236777

/-- Defines the equation of a conic section -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m - 1) + y^2 / (m - 2) = 1 ∧ (m - 1) * (m - 2) < 0

/-- Theorem stating the necessary and sufficient condition for the equation to represent a hyperbola -/
theorem hyperbola_condition (m : ℝ) :
  is_hyperbola m ↔ 1 < m ∧ m < 2 :=
sorry

end hyperbola_condition_l2367_236777


namespace conic_is_ellipse_l2367_236711

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y+2)^2) + Real.sqrt ((x-6)^2 + (y-4)^2) = 14

-- Define what it means for a point to be on the conic
def point_on_conic (x y : ℝ) : Prop :=
  conic_equation x y

-- Define the foci of the conic
def focus1 : ℝ × ℝ := (0, -2)
def focus2 : ℝ × ℝ := (6, 4)

-- Theorem stating that the conic is an ellipse
theorem conic_is_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧
  ∀ (x y : ℝ), point_on_conic x y ↔
    (x - (focus1.1 + focus2.1) / 2)^2 / a^2 +
    (y - (focus1.2 + focus2.2) / 2)^2 / b^2 = 1 :=
sorry

end conic_is_ellipse_l2367_236711


namespace jake_earnings_l2367_236702

/-- Jake's earnings calculation -/
theorem jake_earnings (jacob_hourly_rate : ℝ) (jake_daily_hours : ℝ) (days : ℝ) :
  jacob_hourly_rate = 6 →
  jake_daily_hours = 8 →
  days = 5 →
  (3 * jacob_hourly_rate * jake_daily_hours * days : ℝ) = 720 := by
  sorry

end jake_earnings_l2367_236702


namespace triangle_problem_l2367_236739

theorem triangle_problem (a b c A B C : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B →
  b = 2 →
  (1/2) * a * c * Real.sin B = Real.sqrt 3 →
  B = π/3 ∧ a = 2 ∧ c = 2 := by sorry

end triangle_problem_l2367_236739


namespace pauls_money_duration_l2367_236708

/-- Given Paul's earnings and weekly spending, prove how long the money will last. -/
theorem pauls_money_duration (lawn_earnings weed_earnings weekly_spending : ℕ) 
  (h1 : lawn_earnings = 68)
  (h2 : weed_earnings = 13)
  (h3 : weekly_spending = 9) :
  (lawn_earnings + weed_earnings) / weekly_spending = 9 := by
  sorry

end pauls_money_duration_l2367_236708


namespace fraction_zero_implies_x_one_l2367_236787

theorem fraction_zero_implies_x_one :
  ∀ x : ℝ, (x - 1) / (x - 5) = 0 → x = 1 := by
  sorry

end fraction_zero_implies_x_one_l2367_236787


namespace melinda_original_cost_l2367_236725

/-- Represents the original cost of clothing items before tax and discounts -/
def original_cost (jeans_price shirt_price jacket_price : ℝ) : ℝ :=
  jeans_price + shirt_price + jacket_price

/-- The theorem stating the original cost of Melinda's purchase -/
theorem melinda_original_cost :
  original_cost 14.50 9.50 21.00 = 45.00 := by
  sorry

end melinda_original_cost_l2367_236725


namespace cut_square_corners_l2367_236790

/-- Given a square with side length 24 units, if each corner is cut to form an isoscelos right
    triangle resulting in a smaller rectangle, then the total area of the four removed triangles
    is 288 square units. -/
theorem cut_square_corners (r s : ℝ) : 
  (r + s)^2 + (r - s)^2 = 24^2 → r^2 + s^2 = 288 := by sorry

end cut_square_corners_l2367_236790


namespace number_of_paths_l2367_236785

theorem number_of_paths (paths_A_to_B paths_B_to_D paths_D_to_C : ℕ) 
  (direct_path_A_to_C : ℕ) :
  paths_A_to_B = 2 →
  paths_B_to_D = 3 →
  paths_D_to_C = 3 →
  direct_path_A_to_C = 1 →
  paths_A_to_B * paths_B_to_D * paths_D_to_C + direct_path_A_to_C = 19 :=
by sorry

end number_of_paths_l2367_236785


namespace angle_value_l2367_236774

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem angle_value (a : ℕ → ℝ) (α : ℝ) :
  geometric_sequence a →
  (a 1 * a 1 - 2 * a 1 * Real.sin α - Real.sqrt 3 * Real.sin α = 0) →
  (a 8 * a 8 - 2 * a 8 * Real.sin α - Real.sqrt 3 * Real.sin α = 0) →
  ((a 1 + a 8) ^ 2 = 2 * a 3 * a 6 + 6) →
  (0 < α ∧ α < Real.pi / 2) →
  α = Real.pi / 3 := by
  sorry

end angle_value_l2367_236774


namespace imaginary_part_of_z_l2367_236782

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) : 
  let z := Complex.abs (3 + 4 * i) / (1 - 2 * i)
  Complex.im z = 2 := by
sorry

end imaginary_part_of_z_l2367_236782


namespace article_count_proof_l2367_236775

/-- The number of articles we are considering -/
def X : ℕ := 50

/-- The number of articles sold at selling price -/
def sold_articles : ℕ := 35

/-- The gain percentage -/
def gain_percentage : ℚ := 42857142857142854 / 100000000000000000

theorem article_count_proof :
  (∃ (C S : ℚ), C > 0 ∧ S > 0 ∧
    X * C = sold_articles * S ∧
    (S - C) / C = gain_percentage) →
  X = 50 :=
by sorry

end article_count_proof_l2367_236775


namespace vasya_no_purchase_days_l2367_236751

theorem vasya_no_purchase_days :
  ∀ (x y z w : ℕ),
  x + y + z + w = 15 →
  9 * x + 4 * z = 30 →
  2 * y + z = 9 →
  w = 7 :=
by
  sorry

end vasya_no_purchase_days_l2367_236751


namespace sum_consecutive_odd_integers_to_25_l2367_236715

/-- Sum of consecutive odd integers from 1 to n -/
def sumConsecutiveOddIntegers (n : ℕ) : ℕ :=
  let k := (n + 1) / 2
  k * k

/-- Theorem: The sum of consecutive odd integers from 1 to 25 is 169 -/
theorem sum_consecutive_odd_integers_to_25 :
  sumConsecutiveOddIntegers 25 = 169 := by
  sorry

#eval sumConsecutiveOddIntegers 25

end sum_consecutive_odd_integers_to_25_l2367_236715


namespace sequence_increasing_iff_a_in_range_l2367_236761

def sequence_a (a : ℝ) (n : ℕ) : ℝ :=
  if n ≤ 7 then (3 - a) * n - 3 else a^(n - 6)

theorem sequence_increasing_iff_a_in_range (a : ℝ) :
  (∀ n : ℕ, sequence_a a n ≤ sequence_a a (n + 1)) ↔ (9/4 < a ∧ a < 3) :=
sorry

end sequence_increasing_iff_a_in_range_l2367_236761


namespace subtracted_value_proof_l2367_236767

theorem subtracted_value_proof (N : ℕ) (h : N = 2976) : ∃ V : ℚ, (N / 12 : ℚ) - V = 8 ∧ V = 240 := by
  sorry

end subtracted_value_proof_l2367_236767


namespace small_tub_cost_l2367_236735

def total_cost : ℕ := 48
def num_large_tubs : ℕ := 3
def num_small_tubs : ℕ := 6
def cost_large_tub : ℕ := 6

theorem small_tub_cost : 
  ∃ (cost_small_tub : ℕ), 
    cost_small_tub * num_small_tubs + cost_large_tub * num_large_tubs = total_cost ∧
    cost_small_tub = 5 :=
by sorry

end small_tub_cost_l2367_236735


namespace intersection_A_B_l2367_236731

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 < 0}

-- Define set B
def B : Set ℝ := {x | x > 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 0 2 := by
  sorry

end intersection_A_B_l2367_236731


namespace prime_between_30_and_50_l2367_236733

theorem prime_between_30_and_50 (n : ℕ) :
  Prime n →
  30 < n →
  n < 50 →
  n % 6 = 1 →
  n % 5 ≠ 0 →
  n = 31 ∨ n = 37 ∨ n = 43 := by
sorry

end prime_between_30_and_50_l2367_236733


namespace proposition_truth_l2367_236748

theorem proposition_truth : 
  (∀ x : ℝ, x > 0 → (3 : ℝ) ^ x > (2 : ℝ) ^ x) ∧ 
  (∀ x : ℝ, x < 0 → (3 : ℝ) * x ≤ (2 : ℝ) * x) := by
  sorry

end proposition_truth_l2367_236748


namespace ab_leq_one_l2367_236780

theorem ab_leq_one (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a + b = 2) : a * b ≤ 1 := by
  sorry

end ab_leq_one_l2367_236780


namespace highest_elevation_l2367_236764

/-- The elevation function of a particle projected vertically -/
def s (t : ℝ) : ℝ := 100 * t - 5 * t^2

/-- The initial velocity of the particle in meters per second -/
def initial_velocity : ℝ := 100

theorem highest_elevation :
  ∃ (t_max : ℝ), ∀ (t : ℝ), s t ≤ s t_max ∧ s t_max = 500 := by
  sorry

end highest_elevation_l2367_236764


namespace modulus_of_Z_l2367_236750

-- Define the operation
def matrix_op (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem modulus_of_Z : ∃ (Z : ℂ), 
  (matrix_op Z i 1 i = 1 + i) ∧ (Complex.abs Z = Real.sqrt 5) := by
  sorry

end modulus_of_Z_l2367_236750


namespace shaded_square_area_l2367_236779

/-- A configuration of four unit squares arranged in a 2x2 grid, each containing an inscribed equilateral triangle sharing an edge with the square. -/
structure SquareTriangleConfig where
  /-- The side length of each unit square -/
  unit_square_side : ℝ
  /-- The side length of each equilateral triangle -/
  triangle_side : ℝ
  /-- The side length of the larger square formed by the four unit squares -/
  large_square_side : ℝ
  /-- The side length of the shaded square formed by connecting triangle vertices -/
  shaded_square_side : ℝ
  /-- Condition: Each unit square has side length 1 -/
  unit_square_cond : unit_square_side = 1
  /-- Condition: The triangle side is equal to the unit square side -/
  triangle_side_cond : triangle_side = unit_square_side
  /-- Condition: The larger square has side length 2 -/
  large_square_cond : large_square_side = 2 * unit_square_side
  /-- Condition: The diagonal of the shaded square equals the side of the larger square -/
  shaded_square_diag_cond : shaded_square_side * Real.sqrt 2 = large_square_side

/-- The theorem stating that the area of the shaded square is 2 square units -/
theorem shaded_square_area (config : SquareTriangleConfig) : 
  config.shaded_square_side ^ 2 = 2 := by
  sorry

end shaded_square_area_l2367_236779


namespace janice_earnings_l2367_236738

/-- Calculates the total earnings for a week given specific working conditions --/
def calculate_earnings (weekday_hours : ℕ) (weekend_hours : ℕ) (holiday_hours : ℕ) : ℕ :=
  let weekday_rate := 10
  let weekend_rate := 12
  let holiday_rate := 2 * weekend_rate
  let weekday_earnings := weekday_hours * weekday_rate
  let weekend_earnings := weekend_hours * weekend_rate
  let holiday_earnings := holiday_hours * holiday_rate
  weekday_earnings + weekend_earnings + holiday_earnings

/-- Theorem stating that Janice's earnings for the given week are $720 --/
theorem janice_earnings : calculate_earnings 30 25 5 = 720 := by
  sorry

end janice_earnings_l2367_236738


namespace abc_inequality_l2367_236704

theorem abc_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  a * b > a * c := by
  sorry

end abc_inequality_l2367_236704


namespace trig_equation_solution_l2367_236794

theorem trig_equation_solution (t : ℝ) : 
  (2 * Real.cos (2 * t) + 5) * Real.cos t ^ 4 - (2 * Real.cos (2 * t) + 5) * Real.sin t ^ 4 = 3 ↔ 
  ∃ k : ℤ, t = π / 6 * (6 * ↑k + 1) ∨ t = π / 6 * (6 * ↑k - 1) :=
by sorry

end trig_equation_solution_l2367_236794


namespace exists_overlap_at_least_one_fifth_l2367_236719

/-- Represents a patch on the coat -/
structure Patch where
  area : ℝ
  area_nonneg : area ≥ 0

/-- Represents a coat with patches -/
structure Coat where
  total_area : ℝ
  patches : Finset Patch
  total_area_is_one : total_area = 1
  five_patches : patches.card = 5
  patch_area_at_least_half : ∀ p ∈ patches, p.area ≥ 1/2

/-- The theorem to be proved -/
theorem exists_overlap_at_least_one_fifth (coat : Coat) : 
  ∃ p1 p2 : Patch, p1 ∈ coat.patches ∧ p2 ∈ coat.patches ∧ p1 ≠ p2 ∧ 
    ∃ overlap_area : ℝ, overlap_area ≥ 1/5 ∧ 
      overlap_area ≤ min p1.area p2.area := by
  sorry

end exists_overlap_at_least_one_fifth_l2367_236719


namespace geometric_sequence_ratio_sum_l2367_236795

theorem geometric_sequence_ratio_sum (k a₂ a₃ b₂ b₃ p r : ℝ) 
  (h1 : p ≠ 1)
  (h2 : r ≠ 1)
  (h3 : p ≠ r)
  (h4 : a₂ = k * p)
  (h5 : a₃ = k * p^2)
  (h6 : b₂ = k * r)
  (h7 : b₃ = k * r^2)
  (h8 : 3 * a₃ - 4 * b₃ = 5 * (3 * a₂ - 4 * b₂)) :
  p + r = 5 := by
sorry

end geometric_sequence_ratio_sum_l2367_236795


namespace compute_expression_l2367_236736

theorem compute_expression : 
  20 * (240 / 3 + 40 / 5 + 16 / 25 + 2) = 1772.8 := by
  sorry

end compute_expression_l2367_236736


namespace laura_charge_account_theorem_l2367_236754

/-- Calculates the total amount owed after one year with simple interest -/
def total_amount_owed (principal : ℝ) (interest_rate : ℝ) (time : ℝ) : ℝ :=
  principal + principal * interest_rate * time

/-- Proves that the total amount owed after one year is $37.45 -/
theorem laura_charge_account_theorem :
  let principal : ℝ := 35
  let interest_rate : ℝ := 0.07
  let time : ℝ := 1
  total_amount_owed principal interest_rate time = 37.45 := by
sorry

end laura_charge_account_theorem_l2367_236754


namespace p_min_value_l2367_236746

/-- The quadratic function p(x) = x^2 + 6x + 5 -/
def p (x : ℝ) : ℝ := x^2 + 6*x + 5

/-- The minimum value of p(x) is -4 -/
theorem p_min_value : ∀ x : ℝ, p x ≥ -4 := by sorry

end p_min_value_l2367_236746


namespace action_movies_rented_l2367_236713

theorem action_movies_rented (a : ℝ) : 
  let total_movies := 10 * a / 0.64
  let comedy_movies := 10 * a
  let non_comedy_movies := total_movies - comedy_movies
  let drama_movies := 5 * (non_comedy_movies / 6)
  let action_movies := non_comedy_movies / 6
  action_movies = 0.9375 * a := by
sorry

end action_movies_rented_l2367_236713


namespace set_operations_l2367_236756

def U : Set ℕ := {1,2,3,4,5,6,7,8}

def A : Set ℕ := {x | x^2 - 3*x + 2 = 0}

def B : Set ℕ := {x ∈ U | 1 ≤ x ∧ x ≤ 5}

def C : Set ℕ := {x ∈ U | 2 < x ∧ x < 9}

theorem set_operations :
  (A ∪ (B ∩ C) = {1,2,3,4,5}) ∧
  ((U \ B) ∪ (U \ C) = {1,2,6,7,8}) := by
  sorry

end set_operations_l2367_236756


namespace final_fish_count_l2367_236737

def fish_count (day : ℕ) : ℕ :=
  match day with
  | 0 => 10  -- Initial number of fish
  | 1 => 30  -- Day 1: 10 * 3
  | 2 => 90  -- Day 2: 30 * 3
  | 3 => 270 -- Day 3: 90 * 3
  | 4 => 162 -- Day 4: (270 * 3) - (270 * 3 * 2 / 5)
  | 5 => 486 -- Day 5: 162 * 3
  | 6 => 834 -- Day 6: (486 * 3) - (486 * 3 * 3 / 7)
  | 7 => 2502 -- Day 7: 834 * 3
  | 8 => 7531 -- Day 8: (2502 * 3) + 25
  | 9 => 22593 -- Day 9: 7531 * 3
  | 10 => 33890 -- Day 10: (22593 * 3) - (22593 * 3 / 2)
  | 11 => 101670 -- Day 11: 33890 * 3
  | _ => 305010 -- Day 12: 101670 * 3

theorem final_fish_count :
  fish_count 12 + (3 * fish_count 12 + 5) = 1220045 := by
  sorry

#eval fish_count 12 + (3 * fish_count 12 + 5)

end final_fish_count_l2367_236737


namespace line_k_value_l2367_236769

/-- A line passes through the points (0, 3), (7, k), and (21, 2) -/
def line_passes_through (k : ℚ) : Prop :=
  ∃ m b : ℚ, 
    (3 = m * 0 + b) ∧ 
    (k = m * 7 + b) ∧ 
    (2 = m * 21 + b)

/-- Theorem: If a line passes through (0, 3), (7, k), and (21, 2), then k = 8/3 -/
theorem line_k_value : 
  ∀ k : ℚ, line_passes_through k → k = 8/3 := by
  sorry

end line_k_value_l2367_236769


namespace square_minus_one_divisible_by_three_l2367_236791

theorem square_minus_one_divisible_by_three (x : ℤ) (h : ¬ 3 ∣ x) : 3 ∣ (x^2 - 1) := by
  sorry

end square_minus_one_divisible_by_three_l2367_236791


namespace height_difference_l2367_236741

/-- Given heights of Jana, Jess, and Kelly, prove the height difference between Jess and Kelly. -/
theorem height_difference (jana_height jess_height : ℕ) : 
  jana_height = 74 →
  jess_height = 72 →
  ∃ kelly_height : ℕ, 
    jana_height = kelly_height + 5 ∧ 
    jess_height - kelly_height = 3 := by
  sorry

end height_difference_l2367_236741


namespace parabola_and_circle_problem_l2367_236749

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point K
def K : ℝ × ℝ := (-1, 0)

-- Define the line l passing through K
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m*y - 1

-- Define the condition for points A and B on the parabola and line l
def point_on_parabola_and_line (x y m : ℝ) : Prop :=
  parabola x y ∧ line_l m x y

-- Define the symmetry condition for points A and D
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = x₂ ∧ y₁ = -y₂

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - 1) * (x₂ - 1) + y₁ * y₂ = 8/9

-- Main theorem
theorem parabola_and_circle_problem
  (x₁ y₁ x₂ y₂ xd yd m : ℝ)
  (h₁ : point_on_parabola_and_line x₁ y₁ m)
  (h₂ : point_on_parabola_and_line x₂ y₂ m)
  (h₃ : symmetric_points x₁ y₁ xd yd)
  (h₄ : dot_product_condition x₁ y₁ x₂ y₂) :
  (∃ (k : ℝ), focus.1 = k * (x₂ - xd) + xd ∧ focus.2 = k * (y₂ + yd)) ∧
  (∃ (c : ℝ × ℝ) (r : ℝ), c = (1/9, 0) ∧ r = 2/3 ∧
    ∀ (x y : ℝ), (x - c.1)^2 + (y - c.2)^2 = r^2 ↔
      (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
        x = t * x₂ + (1-t) * K.1 ∧
        y = t * y₂ + (1-t) * K.2) ∨
      (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
        x = t * xd + (1-t) * K.1 ∧
        y = t * yd + (1-t) * K.2) ∨
      (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
        x = t * x₂ + (1-t) * xd ∧
        y = t * y₂ + (1-t) * yd)) :=
sorry

end parabola_and_circle_problem_l2367_236749


namespace max_d_value_l2367_236709

def a (n : ℕ) : ℕ := n^3 + 4

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value : ∃ (k : ℕ), d k = 433 ∧ ∀ (n : ℕ), d n ≤ 433 :=
sorry

end max_d_value_l2367_236709


namespace sum_of_fractions_l2367_236789

theorem sum_of_fractions : (48 : ℚ) / 72 + (30 : ℚ) / 45 = 4 / 3 := by
  sorry

end sum_of_fractions_l2367_236789


namespace solve_for_a_l2367_236759

theorem solve_for_a (a b d : ℤ) 
  (h1 : a + b = d) 
  (h2 : b + d = 7) 
  (h3 : d = 4) : 
  a = 1 := by
sorry

end solve_for_a_l2367_236759


namespace rhombus_area_fraction_l2367_236732

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rhombus defined by four vertices -/
structure Rhombus where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- The grid size -/
def gridSize : ℕ := 6

/-- The rhombus in question -/
def specialRhombus : Rhombus := {
  v1 := ⟨2, 2⟩,
  v2 := ⟨4, 2⟩,
  v3 := ⟨3, 3⟩,
  v4 := ⟨3, 1⟩
}

/-- Calculate the area of a rhombus -/
def rhombusArea (r : Rhombus) : ℝ := sorry

/-- Calculate the area of the grid -/
def gridArea : ℝ := gridSize ^ 2

/-- The main theorem to prove -/
theorem rhombus_area_fraction :
  rhombusArea specialRhombus / gridArea = 1 / 18 := by sorry

end rhombus_area_fraction_l2367_236732


namespace square_difference_l2367_236724

theorem square_difference : (39 : ℤ)^2 = 40^2 - 79 := by
  sorry

end square_difference_l2367_236724


namespace elliptical_lines_l2367_236758

-- Define the points M and N
def M : ℝ × ℝ := (-1, 0)
def N : ℝ × ℝ := (-1, 0)

-- Define the ellipse equation
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the condition for a point to be on a line
def is_on_line (x y : ℝ) (a b c : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Define what it means for a line to be an "elliptical line"
def is_elliptical_line (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, is_on_line x y a b c ∧ is_on_ellipse x y

theorem elliptical_lines :
  is_elliptical_line 1 (-1) 0 ∧ 
  is_elliptical_line 2 (-1) 1 ∧ 
  ¬is_elliptical_line 1 (-2) 6 ∧ 
  ¬is_elliptical_line 1 1 (-3) :=
sorry

end elliptical_lines_l2367_236758


namespace points_collinear_l2367_236743

-- Define the points
variable (A B C K : Point)

-- Define the shapes
variable (square1 square2 : Square)
variable (triangle : Triangle)

-- Define the properties
variable (triangle_isosceles : IsIsosceles triangle)
variable (K_on_triangle_side : OnSide K triangle)

-- Define the theorem
theorem points_collinear (h1 : triangle_isosceles) (h2 : K_on_triangle_side) : 
  Collinear A B C := by sorry

end points_collinear_l2367_236743


namespace exist_three_fractions_product_one_l2367_236700

/-- The sequence of fractions from 1/2017 to 2017/1 -/
def fraction_sequence : Fin 2017 → Rat := λ i => (i + 1) / (2018 - (i + 1))

/-- Theorem: There exist three fractions in the sequence whose product is 1 -/
theorem exist_three_fractions_product_one :
  ∃ (i j k : Fin 2017), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    fraction_sequence i * fraction_sequence j * fraction_sequence k = 1 := by
  sorry

end exist_three_fractions_product_one_l2367_236700


namespace fraction_equation_solution_l2367_236726

theorem fraction_equation_solution (x y : ℝ) 
  (hx_nonzero : x ≠ 0) 
  (hx_not_one : x ≠ 1) 
  (hy_nonzero : y ≠ 0) 
  (hy_not_three : y ≠ 3) 
  (h_equation : (3 / x) + (2 / y) = 1 / 3) : 
  x = 9 * y / (y - 6) := by
sorry

end fraction_equation_solution_l2367_236726


namespace wellness_gym_ratio_l2367_236721

theorem wellness_gym_ratio (f m : ℕ) (hf : f > 0) (hm : m > 0) :
  (35 : ℝ) * f + 30 * m = 32 * (f + m) →
  (f : ℝ) / m = 2 / 3 := by
sorry

end wellness_gym_ratio_l2367_236721


namespace min_value_theorem_l2367_236745

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3*b = 1) :
  (1/a + 1/(3*b)) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3*b₀ = 1 ∧ 1/a₀ + 1/(3*b₀) = 4 :=
by sorry

end min_value_theorem_l2367_236745


namespace f_comp_three_roots_l2367_236707

/-- A quadratic function with a parameter c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 6*x + c

/-- The composition of f with itself -/
def f_comp (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- The theorem stating the condition for f(f(x)) to have exactly 3 distinct real roots -/
theorem f_comp_three_roots (c : ℝ) :
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f_comp c x = 0 ∧ f_comp c y = 0 ∧ f_comp c z = 0 ∧
    (∀ w : ℝ, f_comp c w = 0 → w = x ∨ w = y ∨ w = z)) ↔
  c = (11 - Real.sqrt 13) / 2 :=
sorry

end f_comp_three_roots_l2367_236707


namespace max_value_of_a_l2367_236797

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- State the theorem
theorem max_value_of_a :
  ∀ a : ℝ, (A a ∪ B a = Set.univ) → (∀ b : ℝ, (A b ∪ B b = Set.univ) → b ≤ a) → a = 2 := by
  sorry

-- Note: Set.univ represents the entire real number line (ℝ)

end max_value_of_a_l2367_236797


namespace min_difference_l2367_236734

noncomputable def f (x : ℝ) : ℝ := Real.exp (4 * x - 1)

noncomputable def g (x : ℝ) : ℝ := 1/2 + Real.log (2 * x)

theorem min_difference (m n : ℝ) (h : f m = g n) :
  ∃ (m₀ n₀ : ℝ), f m₀ = g n₀ ∧ ∀ m' n', f m' = g n' → n₀ - m₀ ≤ n' - m' ∧ n₀ - m₀ = (1 + Real.log 2) / 4 :=
sorry

end min_difference_l2367_236734


namespace two_numbers_difference_l2367_236798

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) :
  |x - y| = 4 := by sorry

end two_numbers_difference_l2367_236798


namespace sqrt_product_equality_l2367_236781

theorem sqrt_product_equality : Real.sqrt 54 * Real.sqrt 32 * Real.sqrt 6 = 72 * Real.sqrt 2 := by
  sorry

end sqrt_product_equality_l2367_236781


namespace numbers_with_2019_divisors_l2367_236742

theorem numbers_with_2019_divisors (n : ℕ) : 
  n < 128^97 → (Finset.card (Nat.divisors n) = 2019) → 
  (n = 2^672 * 3^2 ∨ n = 2^672 * 5^2 ∨ n = 2^672 * 7^2 ∨ n = 2^672 * 11^2) :=
by sorry

end numbers_with_2019_divisors_l2367_236742


namespace fraction_to_decimal_l2367_236714

theorem fraction_to_decimal : (17 : ℚ) / (2^2 * 5^4) = (68 : ℚ) / 10000 := by sorry

end fraction_to_decimal_l2367_236714


namespace solve_lawn_mowing_problem_l2367_236768

/-- Kaleb's lawn mowing business finances --/
def lawn_mowing_problem (spring_earnings summer_earnings final_amount : ℕ) : Prop :=
  let total_earnings := spring_earnings + summer_earnings
  let supplies_cost := total_earnings - final_amount
  supplies_cost = total_earnings - final_amount

theorem solve_lawn_mowing_problem :
  lawn_mowing_problem 4 50 50 = true :=
sorry

end solve_lawn_mowing_problem_l2367_236768


namespace dice_sum_product_l2367_236792

theorem dice_sum_product (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 → 
  1 ≤ b ∧ b ≤ 6 → 
  1 ≤ c ∧ c ≤ 6 → 
  1 ≤ d ∧ d ≤ 6 → 
  a * b * c * d = 216 → 
  a + b + c + d ≠ 19 := by
sorry

end dice_sum_product_l2367_236792


namespace line_no_dot_count_l2367_236760

/-- Represents the properties of an alphabet with dots and lines -/
structure Alphabet where
  total_letters : ℕ
  dot_and_line : ℕ
  dot_no_line : ℕ
  has_dot_or_line : Prop

/-- The number of letters with a straight line but no dot -/
def line_no_dot (α : Alphabet) : ℕ :=
  α.total_letters - (α.dot_and_line + α.dot_no_line)

/-- Theorem stating the number of letters with a line but no dot in the given alphabet -/
theorem line_no_dot_count (α : Alphabet) 
  (h1 : α.total_letters = 40)
  (h2 : α.dot_and_line = 11)
  (h3 : α.dot_no_line = 5)
  (h4 : α.has_dot_or_line) :
  line_no_dot α = 24 := by
  sorry

end line_no_dot_count_l2367_236760


namespace gcf_72_120_l2367_236799

theorem gcf_72_120 : Nat.gcd 72 120 = 24 := by
  sorry

end gcf_72_120_l2367_236799


namespace number_difference_l2367_236784

theorem number_difference (a b : ℕ) : 
  a + b = 30000 →
  b = 10 * a + 5 →
  b - a = 24548 := by
sorry

end number_difference_l2367_236784


namespace range_of_f_l2367_236728

def f (x : ℝ) : ℝ := |x + 8| - |x - 3|

theorem range_of_f :
  Set.range f = Set.Icc (-11) 11 := by sorry

end range_of_f_l2367_236728


namespace right_triangle_ratio_squared_l2367_236755

/-- Given a right triangle with legs a and b, and hypotenuse c, 
    where b > a, a/b = (1/2) * (b/c), and a + b + c = 12, 
    prove that (a/b)² = 1/2 -/
theorem right_triangle_ratio_squared (a b c : ℝ) 
  (h1 : b > a)
  (h2 : a / b = (1 / 2) * (b / c))
  (h3 : a + b + c = 12)
  (h4 : c^2 = a^2 + b^2) : 
  (a / b)^2 = 1 / 2 := by
  sorry

end right_triangle_ratio_squared_l2367_236755


namespace solve_equation_l2367_236772

theorem solve_equation (x : ℚ) : (2 * x + 3) / 5 = 11 → x = 26 := by
  sorry

end solve_equation_l2367_236772


namespace sum_of_squares_l2367_236712

theorem sum_of_squares (x y z : ℤ) 
  (sum_eq : x + y + z = 3)
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) :
  x^2 + y^2 + z^2 = 57 := by
  sorry

end sum_of_squares_l2367_236712


namespace combined_bus_capacity_l2367_236723

/-- The capacity of the train -/
def train_capacity : ℕ := 120

/-- The number of buses -/
def num_buses : ℕ := 2

/-- The capacity of each bus as a fraction of the train's capacity -/
def bus_capacity_fraction : ℚ := 1 / 6

/-- Theorem stating the combined capacity of the buses -/
theorem combined_bus_capacity :
  (↑train_capacity * bus_capacity_fraction * ↑num_buses : ℚ) = 40 := by
  sorry

end combined_bus_capacity_l2367_236723


namespace equilateral_triangle_max_area_l2367_236776

/-- The area of a triangle is maximum when it is equilateral, given a fixed perimeter -/
theorem equilateral_triangle_max_area 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let p := (a + b + c) / 2
  let S := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  ∀ a' b' c' : ℝ, 
    a' > 0 → b' > 0 → c' > 0 →
    a' + b' + c' = a + b + c →
    let p' := (a' + b' + c') / 2
    let S' := Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c'))
    S' ≤ S ∧ (S' = S → a' = b' ∧ b' = c') :=
by
  sorry


end equilateral_triangle_max_area_l2367_236776


namespace power_of_five_mod_eighteen_l2367_236717

theorem power_of_five_mod_eighteen (x : ℕ) : ∃ x, x > 0 ∧ (5^x : ℤ) % 18 = 13 ∧ ∀ y, 0 < y ∧ y < x → (5^y : ℤ) % 18 ≠ 13 := by
  sorry

end power_of_five_mod_eighteen_l2367_236717


namespace binomial_sum_identity_l2367_236763

theorem binomial_sum_identity (p q n : ℕ+) :
  (∑' k, (Nat.choose (p + k) p) * (Nat.choose (q + n - k) q)) = Nat.choose (p + q + n + 1) (p + q + 1) :=
sorry

end binomial_sum_identity_l2367_236763
