import Mathlib

namespace line_equation_from_slope_and_intercept_l1695_169554

/-- The equation of a line with slope 2 and y-intercept 4 is y = 2x + 4 -/
theorem line_equation_from_slope_and_intercept :
  ∀ (x y : ℝ), (∃ (m b : ℝ), m = 2 ∧ b = 4 ∧ y = m * x + b) → y = 2 * x + 4 :=
by sorry

end line_equation_from_slope_and_intercept_l1695_169554


namespace chess_tournament_games_l1695_169562

/-- The number of games in a chess tournament where each player plays twice with every other player -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 16 players, where each player plays twice with every other player, 
    the total number of games played is 480. -/
theorem chess_tournament_games : tournament_games 16 * 2 = 480 := by
  sorry

end chess_tournament_games_l1695_169562


namespace dessert_and_coffee_percentage_l1695_169566

theorem dessert_and_coffee_percentage :
  let dessert_percentage : ℝ := 100 - 25.00000000000001
  let dessert_and_coffee_ratio : ℝ := 1 - 0.2
  dessert_and_coffee_ratio * dessert_percentage = 59.999999999999992 :=
by sorry

end dessert_and_coffee_percentage_l1695_169566


namespace worksheets_graded_before_additional_l1695_169544

/-- The number of worksheets initially given to the teacher to grade. -/
def initial_worksheets : ℕ := 6

/-- The number of additional worksheets turned in later. -/
def additional_worksheets : ℕ := 18

/-- The total number of worksheets to grade after the additional ones were turned in. -/
def total_worksheets : ℕ := 20

/-- The number of worksheets graded before the additional ones were turned in. -/
def graded_worksheets : ℕ := 4

theorem worksheets_graded_before_additional :
  initial_worksheets - graded_worksheets + additional_worksheets = total_worksheets :=
sorry

end worksheets_graded_before_additional_l1695_169544


namespace missing_digit_divisible_by_9_l1695_169536

def is_divisible_by_9 (n : Nat) : Prop := n % 9 = 0

theorem missing_digit_divisible_by_9 :
  let n : Nat := 65304
  is_divisible_by_9 n ∧ 
  ∃ d : Nat, d < 10 ∧ n = 65000 + 300 + d * 10 + 4 :=
by sorry

end missing_digit_divisible_by_9_l1695_169536


namespace min_tokens_99x99_grid_l1695_169516

/-- Represents a grid of cells -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a square subgrid -/
structure Subgrid :=
  (size : ℕ)

/-- Calculates the minimum number of tokens required for a grid -/
def min_tokens (g : Grid) (sg : Subgrid) (tokens_per_subgrid : ℕ) : ℕ :=
  g.rows * g.cols - (g.rows / sg.size) * (g.cols / sg.size) * tokens_per_subgrid

/-- The main theorem stating the minimum number of tokens required -/
theorem min_tokens_99x99_grid : 
  let g : Grid := ⟨99, 99⟩
  let sg : Subgrid := ⟨4⟩
  let tokens_per_subgrid : ℕ := 8
  min_tokens g sg tokens_per_subgrid = 4801 := by
  sorry

#check min_tokens_99x99_grid

end min_tokens_99x99_grid_l1695_169516


namespace range_of_m_l1695_169551

/-- The range of m given specific conditions on the roots of a quadratic equation -/
theorem range_of_m (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ∧ 
  ¬(1 < m ∧ m < 3) ∧
  ¬¬(∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) →
  m ≥ 3 ∨ m < -2 :=
by sorry

end range_of_m_l1695_169551


namespace square_equation_solution_l1695_169538

theorem square_equation_solution (x : ℝ) (h : x^2 - 100 = -75) : x = -5 ∨ x = 5 := by
  sorry

end square_equation_solution_l1695_169538


namespace sum_of_four_consecutive_integers_can_be_prime_l1695_169561

theorem sum_of_four_consecutive_integers_can_be_prime : 
  ∃ n : ℤ, Prime (n + (n + 1) + (n + 2) + (n + 3)) :=
by sorry

end sum_of_four_consecutive_integers_can_be_prime_l1695_169561


namespace olympiads_spellings_l1695_169524

def word_length : Nat := 9

-- Function to calculate the number of valid spellings
def valid_spellings (n : Nat) : Nat :=
  if n = 0 then 1
  else if n = word_length then 2^(n-1)
  else 2 * valid_spellings (n-1)

theorem olympiads_spellings :
  valid_spellings word_length = 256 :=
by sorry

end olympiads_spellings_l1695_169524


namespace ellipse_eccentricity_l1695_169519

theorem ellipse_eccentricity (a b x₀ y₀ : ℝ) (h₁ : a > b) (h₂ : b > 0) 
  (h₃ : x₀^2 / a^2 + y₀^2 / b^2 = 1) 
  (h₄ : y₀^2 / ((x₀ + a) * (a - x₀)) = 1/3) : 
  Real.sqrt (1 - b^2 / a^2) = Real.sqrt 6 / 3 := by
  sorry

end ellipse_eccentricity_l1695_169519


namespace al_mass_percentage_in_mixture_l1695_169585

/-- The mass percentage of aluminum in a mixture of AlCl3, Al2(SO4)3, and Al(OH)3 --/
theorem al_mass_percentage_in_mixture (m_AlCl3 m_Al2SO4_3 m_AlOH3 : ℝ)
  (molar_mass_Al molar_mass_AlCl3 molar_mass_Al2SO4_3 molar_mass_AlOH3 : ℝ)
  (h1 : m_AlCl3 = 50)
  (h2 : m_Al2SO4_3 = 70)
  (h3 : m_AlOH3 = 40)
  (h4 : molar_mass_Al = 26.98)
  (h5 : molar_mass_AlCl3 = 133.33)
  (h6 : molar_mass_Al2SO4_3 = 342.17)
  (h7 : molar_mass_AlOH3 = 78.01) :
  let m_Al_AlCl3 := m_AlCl3 / molar_mass_AlCl3 * molar_mass_Al
  let m_Al_Al2SO4_3 := m_Al2SO4_3 / molar_mass_Al2SO4_3 * (2 * molar_mass_Al)
  let m_Al_AlOH3 := m_AlOH3 / molar_mass_AlOH3 * molar_mass_Al
  let total_m_Al := m_Al_AlCl3 + m_Al_Al2SO4_3 + m_Al_AlOH3
  let total_m_mixture := m_AlCl3 + m_Al2SO4_3 + m_AlOH3
  let mass_percentage := total_m_Al / total_m_mixture * 100
  ∃ ε > 0, |mass_percentage - 21.87| < ε :=
by sorry

end al_mass_percentage_in_mixture_l1695_169585


namespace circle_M_properties_l1695_169542

-- Define the circle M
def circle_M : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ (x + 3)^2 + (y - 4)^2 = 25}

-- Define the points
def point_A : ℝ × ℝ := (-3, -1)
def point_B : ℝ × ℝ := (-6, 8)
def point_C : ℝ × ℝ := (1, 1)
def point_P : ℝ × ℝ := (2, 3)

-- Define the tangent lines
def tangent_line_1 (x y : ℝ) : Prop := x = 2
def tangent_line_2 (x y : ℝ) : Prop := 12 * x - 5 * y - 9 = 0

theorem circle_M_properties :
  (point_A ∈ circle_M) ∧
  (point_B ∈ circle_M) ∧
  (point_C ∈ circle_M) ∧
  (∀ (x y : ℝ), tangent_line_1 x y → (x, y) ∈ circle_M → (x, y) = point_P) ∧
  (∀ (x y : ℝ), tangent_line_2 x y → (x, y) ∈ circle_M → (x, y) = point_P) :=
sorry

end circle_M_properties_l1695_169542


namespace only_tiger_leopard_valid_l1695_169584

-- Define the animals
inductive Animal : Type
| Lion : Animal
| Tiger : Animal
| Leopard : Animal
| Elephant : Animal

-- Define a pair of animals
def AnimalPair := (Animal × Animal)

-- Define the conditions
def validPair (pair : AnimalPair) : Prop :=
  -- Two different animals are sent
  pair.1 ≠ pair.2 ∧
  -- If lion is sent, tiger must be sent
  (pair.1 = Animal.Lion ∨ pair.2 = Animal.Lion) → 
    (pair.1 = Animal.Tiger ∨ pair.2 = Animal.Tiger) ∧
  -- If leopard is not sent, tiger cannot be sent
  (pair.1 ≠ Animal.Leopard ∧ pair.2 ≠ Animal.Leopard) → 
    (pair.1 ≠ Animal.Tiger ∧ pair.2 ≠ Animal.Tiger) ∧
  -- If leopard is sent, elephant cannot be sent
  (pair.1 = Animal.Leopard ∨ pair.2 = Animal.Leopard) → 
    (pair.1 ≠ Animal.Elephant ∧ pair.2 ≠ Animal.Elephant)

-- Theorem: The only valid pair is Tiger and Leopard
theorem only_tiger_leopard_valid :
  ∀ (pair : AnimalPair), validPair pair ↔ 
    ((pair.1 = Animal.Tiger ∧ pair.2 = Animal.Leopard) ∨
     (pair.1 = Animal.Leopard ∧ pair.2 = Animal.Tiger)) :=
by sorry

end only_tiger_leopard_valid_l1695_169584


namespace smallest_b_is_correct_l1695_169558

/-- N(b) is the number of natural numbers a for which x^2 + ax + b = 0 has integer roots -/
def N (b : ℕ) : ℕ := sorry

/-- The smallest value of b for which N(b) = 20 -/
def smallest_b : ℕ := 240

theorem smallest_b_is_correct :
  (N smallest_b = 20) ∧ (∀ b : ℕ, b < smallest_b → N b ≠ 20) := by sorry

end smallest_b_is_correct_l1695_169558


namespace geometric_sequence_third_term_l1695_169549

theorem geometric_sequence_third_term 
  (a : ℕ → ℝ) 
  (is_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (first_term : a 1 = 1) 
  (fifth_term : a 5 = 4) : 
  a 3 = 2 := by
sorry

end geometric_sequence_third_term_l1695_169549


namespace quadratic_intersection_and_equivalence_l1695_169572

-- Define the quadratic function p
def p (x : ℝ) : Prop := x^2 - 7*x + 10 < 0

-- Define the function q
def q (x m : ℝ) : Prop := (x - m) * (x - 3*m) < 0

theorem quadratic_intersection_and_equivalence :
  (∃ (a b : ℝ), a = 4 ∧ b = 5 ∧ 
    (∀ x : ℝ, (p x ∧ q x 4) ↔ (a < x ∧ x < b))) ∧
  (∃ (c d : ℝ), c = 5/3 ∧ d = 2 ∧ 
    (∀ m : ℝ, m > 0 → 
      ((∀ x : ℝ, ¬(q x m) ↔ ¬(p x)) ↔ (c ≤ m ∧ m ≤ d)))) :=
by sorry

end quadratic_intersection_and_equivalence_l1695_169572


namespace max_cakes_l1695_169540

/-- Represents the configuration of cuts on a rectangular cake -/
structure CakeCut where
  rows : Nat
  columns : Nat

/-- Calculates the total number of cake pieces after cutting -/
def totalPieces (cut : CakeCut) : Nat :=
  (cut.rows + 1) * (cut.columns + 1)

/-- Calculates the number of interior pieces -/
def interiorPieces (cut : CakeCut) : Nat :=
  (cut.rows - 1) * (cut.columns - 1)

/-- Calculates the number of perimeter pieces -/
def perimeterPieces (cut : CakeCut) : Nat :=
  2 * (cut.rows + cut.columns)

/-- Checks if the cutting configuration satisfies the given condition -/
def isValidCut (cut : CakeCut) : Prop :=
  interiorPieces cut = perimeterPieces cut + 1

/-- The main theorem stating the maximum number of cakes -/
theorem max_cakes : ∃ (cut : CakeCut), isValidCut cut ∧ 
  totalPieces cut = 65 ∧ 
  (∀ (other : CakeCut), isValidCut other → totalPieces other ≤ 65) :=
sorry

end max_cakes_l1695_169540


namespace root_condition_implies_m_range_l1695_169574

theorem root_condition_implies_m_range :
  ∀ (m : ℝ) (x₁ x₂ : ℝ),
    (m + 3) * x₁^2 - 4 * m * x₁ + 2 * m - 1 = 0 →
    (m + 3) * x₂^2 - 4 * m * x₂ + 2 * m - 1 = 0 →
    x₁ * x₂ < 0 →
    (x₁ < 0 ∧ x₂ > 0 → |x₁| > x₂) →
    (x₂ < 0 ∧ x₁ > 0 → |x₂| > x₁) →
    -3 < m ∧ m < 0 :=
by sorry

end root_condition_implies_m_range_l1695_169574


namespace remaining_sweets_theorem_l1695_169546

/-- The number of remaining sweets after Aaron's actions -/
def remaining_sweets (C S P R L : ℕ) : ℕ :=
  let eaten_C := (2 * C) / 5
  let eaten_S := S / 4
  let eaten_P := (3 * P) / 5
  let given_C := (C - P / 4) / 3
  let discarded_R := (3 * R) / 2
  let eaten_L := (eaten_S * 6) / 5
  (C - eaten_C - given_C) + (S - eaten_S) + (P - eaten_P) + (if R > discarded_R then R - discarded_R else 0) + (L - eaten_L)

theorem remaining_sweets_theorem :
  remaining_sweets 30 100 60 25 150 = 232 := by
  sorry

end remaining_sweets_theorem_l1695_169546


namespace gcd_of_nine_digit_numbers_l1695_169507

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000000 ∧ n ≤ 999999999) ∧
  ∃ (d₁ d₂ d₃ d₄ d₅ d₆ d₇ d₈ d₉ : ℕ),
    d₁ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₂ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₃ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₄ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₅ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₆ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₇ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₈ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₉ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₁ ≠ d₅ ∧ d₁ ≠ d₆ ∧ d₁ ≠ d₇ ∧ d₁ ≠ d₈ ∧ d₁ ≠ d₉ ∧
    d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₂ ≠ d₅ ∧ d₂ ≠ d₆ ∧ d₂ ≠ d₇ ∧ d₂ ≠ d₈ ∧ d₂ ≠ d₉ ∧
    d₃ ≠ d₄ ∧ d₃ ≠ d₅ ∧ d₃ ≠ d₆ ∧ d₃ ≠ d₇ ∧ d₃ ≠ d₈ ∧ d₃ ≠ d₉ ∧
    d₄ ≠ d₅ ∧ d₄ ≠ d₆ ∧ d₄ ≠ d₇ ∧ d₄ ≠ d₈ ∧ d₄ ≠ d₉ ∧
    d₅ ≠ d₆ ∧ d₅ ≠ d₇ ∧ d₅ ≠ d₈ ∧ d₅ ≠ d₉ ∧
    d₆ ≠ d₇ ∧ d₆ ≠ d₈ ∧ d₆ ≠ d₉ ∧
    d₇ ≠ d₈ ∧ d₇ ≠ d₉ ∧
    d₈ ≠ d₉ ∧
    n = d₁ * 100000000 + d₂ * 10000000 + d₃ * 1000000 + d₄ * 100000 + d₅ * 10000 + d₆ * 1000 + d₇ * 100 + d₈ * 10 + d₉

theorem gcd_of_nine_digit_numbers :
  ∃ (g : ℕ), g > 0 ∧ (∀ (n : ℕ), is_valid_number n → g ∣ n) ∧
  (∀ (d : ℕ), d > 0 → (∀ (n : ℕ), is_valid_number n → d ∣ n) → d ≤ g) ∧
  g = 9 := by
  sorry

end gcd_of_nine_digit_numbers_l1695_169507


namespace unique_number_divisible_by_792_l1695_169575

theorem unique_number_divisible_by_792 :
  ∀ (x y z : ℕ), x < 10 → y < 10 → z < 10 →
  (13 * 100000 + x * 10000 + y * 1000 + 45 * 10 + z) % 792 = 0 →
  (13 * 100000 + x * 10000 + y * 1000 + 45 * 10 + z) = 1380456 := by
sorry

end unique_number_divisible_by_792_l1695_169575


namespace initial_average_marks_l1695_169514

theorem initial_average_marks
  (n : ℕ)  -- number of students
  (correct_avg : ℚ)  -- correct average after fixing the error
  (wrong_mark : ℚ)  -- wrongly noted mark
  (right_mark : ℚ)  -- correct mark
  (h1 : n = 30)  -- there are 30 students
  (h2 : correct_avg = 98)  -- correct average is 98
  (h3 : wrong_mark = 70)  -- wrongly noted mark is 70
  (h4 : right_mark = 10)  -- correct mark is 10
  : (n * correct_avg + (right_mark - wrong_mark)) / n = 100 :=
by sorry

end initial_average_marks_l1695_169514


namespace quadratic_has_real_root_l1695_169560

theorem quadratic_has_real_root (a b : ℝ) : ∃ x : ℝ, x^2 + a*x + b = 0 := by
  sorry

end quadratic_has_real_root_l1695_169560


namespace melissa_games_played_l1695_169599

theorem melissa_games_played (points_per_game : ℕ) (total_points : ℕ) (h1 : points_per_game = 12) (h2 : total_points = 36) :
  total_points / points_per_game = 3 := by
  sorry

end melissa_games_played_l1695_169599


namespace least_x_value_l1695_169541

theorem least_x_value (x y : ℤ) (h : x * y + 6 * x + 8 * y = -4) :
  ∀ z : ℤ, z ≥ -52 ∨ ¬∃ w : ℤ, z * w + 6 * z + 8 * w = -4 :=
sorry

end least_x_value_l1695_169541


namespace complement_of_A_in_U_l1695_169563

-- Define the universal set U
def U : Set ℝ := {x | x < 5}

-- Define the set A
def A : Set ℝ := {x | x - 2 ≤ 0}

-- State the theorem
theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {x | 2 < x ∧ x < 5} :=
by sorry

end complement_of_A_in_U_l1695_169563


namespace cherry_difference_l1695_169580

theorem cherry_difference (initial_cherries left_cherries : ℕ) 
  (h1 : initial_cherries = 16)
  (h2 : left_cherries = 6) :
  initial_cherries - left_cherries = 10 := by
  sorry

end cherry_difference_l1695_169580


namespace division_by_negative_l1695_169592

theorem division_by_negative : 15 / (-3 : ℤ) = -5 := by sorry

end division_by_negative_l1695_169592


namespace owen_burger_spending_l1695_169501

/-- The number of days in June -/
def days_in_june : ℕ := 30

/-- The number of burgers Owen buys per day -/
def burgers_per_day : ℕ := 2

/-- The cost of each burger in dollars -/
def cost_per_burger : ℕ := 12

/-- Theorem: Owen's total spending on burgers in June is $720 -/
theorem owen_burger_spending :
  days_in_june * burgers_per_day * cost_per_burger = 720 := by
  sorry


end owen_burger_spending_l1695_169501


namespace undefined_values_expression_undefined_l1695_169502

theorem undefined_values (x : ℝ) : 
  (2 * x^2 - 8 * x - 42 = 0) ↔ (x = 7 ∨ x = -3) :=
by sorry

theorem expression_undefined (x : ℝ) :
  ¬ (∃ y : ℝ, y = (3 * x^2 - 1) / (2 * x^2 - 8 * x - 42)) ↔ (x = 7 ∨ x = -3) :=
by sorry

end undefined_values_expression_undefined_l1695_169502


namespace smallest_positive_multiple_of_45_l1695_169522

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 :=
by sorry

end smallest_positive_multiple_of_45_l1695_169522


namespace square_plus_one_geq_two_abs_l1695_169528

theorem square_plus_one_geq_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end square_plus_one_geq_two_abs_l1695_169528


namespace triple_base_exponent_l1695_169598

theorem triple_base_exponent (a b : ℤ) (x : ℚ) (h1 : b ≠ 0) :
  (3 * a) ^ (3 * b) = a ^ b * x ^ b → x = 27 * a ^ 2 := by
  sorry

end triple_base_exponent_l1695_169598


namespace evaluate_expression_l1695_169512

theorem evaluate_expression : (-3)^4 / 3^2 - 2^5 + 7^2 = 26 := by
  sorry

end evaluate_expression_l1695_169512


namespace shaded_region_angle_l1695_169589

/-- Given two concentric circles with radii 1 and 2, if the area of the shaded region
    between them is three times smaller than the area of the larger circle,
    then the angle subtending this shaded region at the center is 8π/9 radians. -/
theorem shaded_region_angle (r₁ r₂ : ℝ) (A_shaded A_large : ℝ) (θ : ℝ) :
  r₁ = 1 →
  r₂ = 2 →
  A_large = π * r₂^2 →
  A_shaded = (1/3) * A_large →
  A_shaded = (θ / (2 * π)) * (π * r₂^2 - π * r₁^2) →
  θ = (8 * π) / 9 := by
  sorry

end shaded_region_angle_l1695_169589


namespace isogonal_conjugate_is_conic_l1695_169555

/-- Trilinear coordinates -/
structure TrilinearCoord where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A triangle -/
structure Triangle where
  A : TrilinearCoord
  B : TrilinearCoord
  C : TrilinearCoord

/-- A line in trilinear coordinates -/
structure Line where
  p : ℝ
  q : ℝ
  r : ℝ

/-- Isogonal conjugation transformation -/
def isogonalConjugate (l : Line) : TrilinearCoord → Prop :=
  fun point => l.p * point.y * point.z + l.q * point.x * point.z + l.r * point.x * point.y = 0

/-- Definition of a conic section -/
def isConicSection (f : TrilinearCoord → Prop) : Prop := sorry

/-- The theorem to be proved -/
theorem isogonal_conjugate_is_conic (t : Triangle) (l : Line) 
  (h1 : l.p ≠ 0) (h2 : l.q ≠ 0) (h3 : l.r ≠ 0)
  (h4 : l.p * t.A.x + l.q * t.A.y + l.r * t.A.z ≠ 0)
  (h5 : l.p * t.B.x + l.q * t.B.y + l.r * t.B.z ≠ 0)
  (h6 : l.p * t.C.x + l.q * t.C.y + l.r * t.C.z ≠ 0) :
  isConicSection (isogonalConjugate l) ∧ 
  isogonalConjugate l t.A ∧ 
  isogonalConjugate l t.B ∧ 
  isogonalConjugate l t.C :=
sorry

end isogonal_conjugate_is_conic_l1695_169555


namespace vacation_cost_l1695_169511

theorem vacation_cost (C : ℝ) : 
  (C / 6 - C / 8 = 120) → C = 2880 := by
sorry

end vacation_cost_l1695_169511


namespace infinitely_many_coprime_linear_combination_l1695_169518

theorem infinitely_many_coprime_linear_combination (a b n : ℕ) 
  (ha : a > 0) (hb : b > 0) (hn : n > 0) (hab : Nat.gcd a b = 1) :
  Set.Infinite {k : ℕ | Nat.gcd (a * k + b) n = 1} := by
  sorry

end infinitely_many_coprime_linear_combination_l1695_169518


namespace shekar_social_studies_score_l1695_169543

theorem shekar_social_studies_score 
  (math_score : ℕ) 
  (science_score : ℕ) 
  (english_score : ℕ) 
  (biology_score : ℕ) 
  (average_score : ℕ) 
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : english_score = 67)
  (h4 : biology_score = 95)
  (h5 : average_score = 77)
  (h6 : (math_score + science_score + english_score + biology_score + social_studies_score) / 5 = average_score) :
  social_studies_score = 82 :=
by
  sorry

#check shekar_social_studies_score

end shekar_social_studies_score_l1695_169543


namespace max_sum_of_remaining_pairs_l1695_169548

/-- Given a set of four distinct real numbers, this function returns the list of their six pairwise sums. -/
def pairwiseSums (a b c d : ℝ) : List ℝ :=
  [a + b, a + c, a + d, b + c, b + d, c + d]

/-- This theorem states that given four distinct real numbers whose pairwise sums include 210, 360, 330, and 300,
    the maximum possible sum of the remaining two pairwise sums is 870. -/
theorem max_sum_of_remaining_pairs (a b c d : ℝ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∃ (l : List ℝ), l = pairwiseSums a b c d ∧ 
    (210 ∈ l) ∧ (360 ∈ l) ∧ (330 ∈ l) ∧ (300 ∈ l)) →
  (∃ (x y : ℝ), x ∈ pairwiseSums a b c d ∧ 
                y ∈ pairwiseSums a b c d ∧ 
                x ≠ 210 ∧ x ≠ 360 ∧ x ≠ 330 ∧ x ≠ 300 ∧
                y ≠ 210 ∧ y ≠ 360 ∧ y ≠ 330 ∧ y ≠ 300 ∧
                x + y ≤ 870) :=
by sorry


end max_sum_of_remaining_pairs_l1695_169548


namespace functions_equal_if_surjective_injective_and_greater_or_equal_l1695_169539

theorem functions_equal_if_surjective_injective_and_greater_or_equal
  (f g : ℕ → ℕ)
  (h_surj : Function.Surjective f)
  (h_inj : Function.Injective g)
  (h_ge : ∀ n : ℕ, f n ≥ g n) :
  ∀ n : ℕ, f n = g n := by
  sorry

end functions_equal_if_surjective_injective_and_greater_or_equal_l1695_169539


namespace last_four_average_l1695_169520

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  (list.sum / list.length) = 65 →
  ((list.take 3).sum / 3) = 60 →
  ((list.drop 3).sum / 4) = 68.75 := by
sorry

end last_four_average_l1695_169520


namespace perpendicular_parallel_implies_perpendicular_parallel_transitive_perpendicular_l1695_169597

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- Theorem 1
theorem perpendicular_parallel_implies_perpendicular
  (m n : Line) (α : Plane)
  (h1 : perpendicular m α)
  (h2 : parallel_line_plane n α)
  (h3 : m ≠ n) :
  perpendicular_lines m n :=
sorry

-- Theorem 2
theorem parallel_transitive_perpendicular
  (m : Line) (α β γ : Plane)
  (h1 : parallel_plane α β)
  (h2 : parallel_plane β γ)
  (h3 : perpendicular m α)
  (h4 : α ≠ β) (h5 : β ≠ γ) (h6 : α ≠ γ) :
  perpendicular m γ :=
sorry

end perpendicular_parallel_implies_perpendicular_parallel_transitive_perpendicular_l1695_169597


namespace f_is_quadratic_l1695_169576

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing 2x - x^2 -/
def f (x : ℝ) : ℝ := 2*x - x^2

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end f_is_quadratic_l1695_169576


namespace intersection_M_N_l1695_169591

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2 + 1}
def N : Set ℝ := {y | ∃ x, y = x + 1}

-- State the theorem
theorem intersection_M_N : M ∩ N = {y | y ≥ 1} := by sorry

end intersection_M_N_l1695_169591


namespace intersection_A_B_union_B_complement_A_C_subset_complement_B_l1695_169579

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x < 9}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}

-- State the theorems
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 5} := by sorry

theorem union_B_complement_A : B ∪ Aᶜ = {x : ℝ | x ≤ 5 ∨ x ≥ 9} := by sorry

theorem C_subset_complement_B (a : ℝ) :
  C a ⊆ Bᶜ ↔ a < -4 ∨ a > 5 := by sorry

end intersection_A_B_union_B_complement_A_C_subset_complement_B_l1695_169579


namespace largest_sum_proof_l1695_169565

theorem largest_sum_proof : 
  let sums : List ℚ := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/6, 1/3 + 1/9, 1/3 + 1/8]
  ∀ x ∈ sums, x ≤ 1/3 + 1/4 ∧ 1/3 + 1/4 = 7/12 := by
sorry

end largest_sum_proof_l1695_169565


namespace root_difference_l1695_169570

theorem root_difference (p q : ℝ) (h : p ≠ q) :
  let r := (p + q + Real.sqrt ((p - q)^2)) / 2
  let s := (p + q - Real.sqrt ((p - q)^2)) / 2
  r - s = |p - q| := by
sorry

end root_difference_l1695_169570


namespace unique_prime_solution_l1695_169577

theorem unique_prime_solution : ∃! (p : ℕ), 
  Prime p ∧ 
  ∃ (x y : ℕ), 
    x > 0 ∧ 
    y > 0 ∧ 
    p + 49 = 2 * x^2 ∧ 
    p^2 + 49 = 2 * y^2 ∧ 
    p = 23 := by
  sorry

end unique_prime_solution_l1695_169577


namespace total_cost_new_puppy_l1695_169595

def adoption_fee : ℝ := 20
def dog_food : ℝ := 20
def treat_bag_price : ℝ := 2.5
def num_treat_bags : ℕ := 2
def toys : ℝ := 15
def crate : ℝ := 20
def bed : ℝ := 20
def collar_leash : ℝ := 15
def discount_rate : ℝ := 0.2

theorem total_cost_new_puppy :
  let supplies_cost := dog_food + treat_bag_price * num_treat_bags + toys + crate + bed + collar_leash
  let discounted_supplies_cost := supplies_cost * (1 - discount_rate)
  adoption_fee + discounted_supplies_cost = 96 :=
by sorry

end total_cost_new_puppy_l1695_169595


namespace inequality_proof_l1695_169503

def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|1/3 * a + 1/6 * b| < 1/4) ∧ (|1 - 4*a*b| > 2 * |a - b|) := by
  sorry

end inequality_proof_l1695_169503


namespace average_marks_of_failed_boys_l1695_169582

theorem average_marks_of_failed_boys
  (total_boys : ℕ)
  (overall_average : ℚ)
  (passed_average : ℚ)
  (passed_boys : ℕ)
  (h1 : total_boys = 120)
  (h2 : overall_average = 38)
  (h3 : passed_average = 39)
  (h4 : passed_boys = 115) :
  (total_boys * overall_average - passed_boys * passed_average) / (total_boys - passed_boys) = 15 := by
sorry

end average_marks_of_failed_boys_l1695_169582


namespace frustum_volume_l1695_169513

/-- Represents a frustum of a cone -/
structure Frustum where
  upper_base_area : ℝ
  lower_base_area : ℝ
  lateral_area : ℝ

/-- Calculate the volume of a frustum -/
def volume (f : Frustum) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem frustum_volume (f : Frustum) 
  (h1 : f.upper_base_area = π)
  (h2 : f.lower_base_area = 4 * π)
  (h3 : f.lateral_area = 6 * π) : 
  volume f = 4 * π := by
  sorry

end frustum_volume_l1695_169513


namespace smallest_factor_for_perfect_cube_l1695_169573

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_factor_for_perfect_cube (x : ℕ) (hx : x = 3 * 40 * 75) :
  ∃ y : ℕ, y > 0 ∧ is_perfect_cube (x * y) ∧ ∀ z : ℕ, z > 0 → is_perfect_cube (x * z) → y ≤ z :=
by
  sorry

end smallest_factor_for_perfect_cube_l1695_169573


namespace inequality_proof_l1695_169547

theorem inequality_proof (x y m : ℝ) (h1 : x > y) (h2 : m > 0) : x - y > 0 := by
  sorry

end inequality_proof_l1695_169547


namespace bicycle_spoke_ratio_l1695_169534

theorem bicycle_spoke_ratio : 
  ∀ (front_spokes back_spokes : ℕ),
    front_spokes = 20 →
    front_spokes + back_spokes = 60 →
    (back_spokes : ℚ) / front_spokes = 2 := by
  sorry

end bicycle_spoke_ratio_l1695_169534


namespace data_set_properties_l1695_169510

def data_set : List ℕ := [67, 57, 37, 40, 46, 62, 31, 47, 31, 30]

def mode (l : List ℕ) : ℕ := sorry

def range (l : List ℕ) : ℕ := sorry

def quantile (l : List ℕ) (p : ℚ) : ℚ := sorry

theorem data_set_properties :
  (mode data_set = 31) ∧
  (range data_set = 37) ∧
  (quantile data_set (1/10) = 30.5) := by
  sorry

end data_set_properties_l1695_169510


namespace sum_a_plus_c_equals_four_l1695_169517

/-- Represents a three-digit number in the form abc -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem sum_a_plus_c_equals_four :
  ∀ (a c : Nat),
  let num1 := ThreeDigitNumber.mk 2 a 3 (by sorry)
  let num2 := ThreeDigitNumber.mk 6 c 9 (by sorry)
  (num1.toNat + 427 = num2.toNat) →
  (num2.toNat % 3 = 0) →
  a + c = 4 := by
  sorry

end sum_a_plus_c_equals_four_l1695_169517


namespace problem_equivalent_l1695_169556

theorem problem_equivalent : (16^1011) / 8 = 2^4033 := by
  sorry

end problem_equivalent_l1695_169556


namespace units_digit_sum_factorials_50_l1695_169525

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The sum of factorials from 1 to n -/
def sumFactorials (n : ℕ) : ℕ := 
  (List.range n).map (λ i => factorial (i + 1)) |> List.sum

/-- The units digit of the sum of factorials from 1! to 50! is 3 -/
theorem units_digit_sum_factorials_50 : 
  unitsDigit (sumFactorials 50) = 3 := by sorry

end units_digit_sum_factorials_50_l1695_169525


namespace mixed_oil_rate_l1695_169537

/-- The rate of mixed oil per litre given two different oils -/
theorem mixed_oil_rate (volume1 : ℝ) (price1 : ℝ) (volume2 : ℝ) (price2 : ℝ) :
  volume1 = 10 →
  price1 = 50 →
  volume2 = 5 →
  price2 = 66 →
  (volume1 * price1 + volume2 * price2) / (volume1 + volume2) = 55.33 := by
  sorry

end mixed_oil_rate_l1695_169537


namespace max_men_with_all_items_and_married_l1695_169567

theorem max_men_with_all_items_and_married 
  (total_men : ℕ) 
  (married_men : ℕ) 
  (men_with_tv : ℕ) 
  (men_with_radio : ℕ) 
  (men_with_ac : ℕ) 
  (h_total : total_men = 100)
  (h_married : married_men = 85)
  (h_tv : men_with_tv = 75)
  (h_radio : men_with_radio = 85)
  (h_ac : men_with_ac = 70)
  : ∃ (max_all_items_married : ℕ), 
    max_all_items_married ≤ 70 ∧ 
    max_all_items_married ≤ married_men ∧
    max_all_items_married ≤ men_with_tv ∧
    max_all_items_married ≤ men_with_radio ∧
    max_all_items_married ≤ men_with_ac :=
by sorry

end max_men_with_all_items_and_married_l1695_169567


namespace books_second_shop_correct_l1695_169559

/-- The number of books bought from the second shop -/
def books_second_shop : ℕ := 20

/-- The number of books bought from the first shop -/
def books_first_shop : ℕ := 27

/-- The cost of books from the first shop in rupees -/
def cost_first_shop : ℕ := 581

/-- The cost of books from the second shop in rupees -/
def cost_second_shop : ℕ := 594

/-- The average price per book in rupees -/
def average_price : ℕ := 25

theorem books_second_shop_correct : 
  books_second_shop = 20 ∧
  books_first_shop = 27 ∧
  cost_first_shop = 581 ∧
  cost_second_shop = 594 ∧
  average_price = 25 →
  (cost_first_shop + cost_second_shop : ℚ) / (books_first_shop + books_second_shop) = average_price := by
  sorry

end books_second_shop_correct_l1695_169559


namespace total_shuttlecocks_distributed_l1695_169527

-- Define the number of students
def num_students : ℕ := 24

-- Define the number of shuttlecocks per student
def shuttlecocks_per_student : ℕ := 19

-- Theorem to prove
theorem total_shuttlecocks_distributed :
  num_students * shuttlecocks_per_student = 456 := by
  sorry

end total_shuttlecocks_distributed_l1695_169527


namespace diophantine_equation_solutions_l1695_169509

theorem diophantine_equation_solutions :
  (∀ k : ℤ, (101 * (4 + 13 * k) - 13 * (31 + 101 * k) = 1)) ∧
  (∀ k : ℤ, (79 * (-6 + 19 * k) - 19 * (-25 + 79 * k) = 1)) :=
by sorry

end diophantine_equation_solutions_l1695_169509


namespace intersection_nonempty_implies_m_range_l1695_169550

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | m / 2 ≤ (p.1 - 2)^2 + p.2^2 ∧ (p.1 - 2)^2 + p.2^2 ≤ m^2}

def B (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * m ≤ p.1 + p.2 ∧ p.1 + p.2 ≤ 2 * m + 1}

-- State the theorem
theorem intersection_nonempty_implies_m_range (m : ℝ) :
  (A m ∩ B m).Nonempty → 1/2 ≤ m ∧ m ≤ 2 + Real.sqrt 2 :=
by sorry

end intersection_nonempty_implies_m_range_l1695_169550


namespace remainder_when_divided_by_nine_l1695_169500

/-- The smallest positive integer satisfying the given conditions -/
def smallest_n : ℕ := sorry

/-- The first condition: n mod 8 = 6 -/
axiom cond1 : smallest_n % 8 = 6

/-- The second condition: n mod 7 = 5 -/
axiom cond2 : smallest_n % 7 = 5

/-- The smallest_n is indeed the smallest positive integer satisfying both conditions -/
axiom smallest : ∀ m : ℕ, m > 0 → m % 8 = 6 → m % 7 = 5 → m ≥ smallest_n

/-- Theorem: The smallest n satisfying both conditions leaves a remainder of 1 when divided by 9 -/
theorem remainder_when_divided_by_nine : smallest_n % 9 = 1 := by sorry

end remainder_when_divided_by_nine_l1695_169500


namespace store_a_prices_store_b_original_price_l1695_169533

/-- Represents a store selling notebooks -/
structure Store where
  hardcover_price : ℕ
  softcover_price : ℕ
  hardcover_more_expensive : hardcover_price = softcover_price + 3

/-- Theorem for Store A's notebook prices -/
theorem store_a_prices (a : Store) 
  (h1 : 240 / a.hardcover_price = 195 / a.softcover_price) :
  a.hardcover_price = 16 := by
  sorry

/-- Represents Store B's discount policy -/
def discount_policy (price : ℕ) (quantity : ℕ) : ℕ :=
  if quantity ≥ 30 then price - 3 else price

/-- Theorem for Store B's original hardcover notebook price -/
theorem store_b_original_price (b : Store) (m : ℕ)
  (h1 : m < 30)
  (h2 : m + 5 ≥ 30)
  (h3 : m * b.hardcover_price = (m + 5) * (b.hardcover_price - 3)) :
  b.hardcover_price = 18 := by
  sorry

end store_a_prices_store_b_original_price_l1695_169533


namespace expected_value_of_sum_is_twelve_l1695_169521

def marbles : Finset ℕ := Finset.range 7

def choose_three (s : Finset ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ subset => subset.card = 3)

def sum_of_subset (subset : Finset ℕ) : ℕ :=
  subset.sum id

theorem expected_value_of_sum_is_twelve :
  let all_choices := choose_three marbles
  let sum_of_sums := all_choices.sum sum_of_subset
  let num_choices := all_choices.card
  (sum_of_sums : ℚ) / num_choices = 12 := by sorry

end expected_value_of_sum_is_twelve_l1695_169521


namespace range_of_m_for_inverse_proposition_l1695_169553

theorem range_of_m_for_inverse_proposition : 
  ∀ m : ℝ, 
  (∀ x : ℝ, (1 < x ∧ x < 3) → (m < x ∧ x < m + 3)) → 
  (0 ≤ m ∧ m ≤ 1) :=
by sorry

end range_of_m_for_inverse_proposition_l1695_169553


namespace function_symmetry_theorem_l1695_169586

-- Define the exponential function
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

-- Define the concept of symmetry about y-axis
def symmetric_about_y_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_symmetry_theorem :
  symmetric_about_y_axis (fun x ↦ f (x - 1)) exp →
  f = fun x ↦ exp (-x - 1) := by
  sorry

end function_symmetry_theorem_l1695_169586


namespace total_serving_time_l1695_169564

def total_patients : ℕ := 12
def standard_serving_time : ℕ := 5
def special_needs_ratio : ℚ := 1/3
def special_needs_time_increase : ℚ := 1/5

theorem total_serving_time :
  let special_patients := total_patients * special_needs_ratio
  let standard_patients := total_patients - special_patients
  let special_serving_time := standard_serving_time * (1 + special_needs_time_increase)
  let total_time := standard_patients * standard_serving_time + special_patients * special_serving_time
  total_time = 64 := by
  sorry

end total_serving_time_l1695_169564


namespace workers_wage_increase_l1695_169587

theorem workers_wage_increase (original_wage new_wage : ℝ) : 
  (new_wage = original_wage * 1.5) → (new_wage = 51) → (original_wage = 34) := by
  sorry

end workers_wage_increase_l1695_169587


namespace laticia_socks_l1695_169515

def sock_problem (nephew_socks week1_socks week2_extra week3_fraction week4_decrease : ℕ) : Prop :=
  let week2_socks := week1_socks + week2_extra
  let week3_socks := (week1_socks + week2_socks) / 2
  let week4_socks := week3_socks - week4_decrease
  nephew_socks + week1_socks + week2_socks + week3_socks + week4_socks = 57

theorem laticia_socks : 
  sock_problem 4 12 4 2 3 := by sorry

end laticia_socks_l1695_169515


namespace prime_ones_and_seven_l1695_169552

/-- Represents a number with n-1 digits 1 and one digit 7 -/
def A (n : ℕ) (k : ℕ) : ℕ := (10^n + 54 * 10^k - 1) / 9

/-- Checks if a number is prime -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- All numbers with n-1 digits 1 and one digit 7 are prime -/
def all_prime (n : ℕ) : Prop := ∀ k : ℕ, k < n → is_prime (A n k)

theorem prime_ones_and_seven :
  ∀ n : ℕ, (all_prime n ↔ n = 1 ∨ n = 2) :=
sorry

end prime_ones_and_seven_l1695_169552


namespace same_functions_l1695_169529

theorem same_functions (x : ℝ) (h : x ≠ 1) : (x - 1) ^ 0 = 1 / ((x - 1) ^ 0) := by
  sorry

end same_functions_l1695_169529


namespace quadratic_transformation_l1695_169557

theorem quadratic_transformation (a b c : ℝ) :
  (∃ (m q : ℝ), ∀ x, ax^2 + bx + c = 5*(x - 3)^2 + 15) →
  (∃ (m p q : ℝ), ∀ x, 4*ax^2 + 4*bx + 4*c = m*(x - p)^2 + q ∧ p = 3) :=
by sorry

end quadratic_transformation_l1695_169557


namespace circle_m_equation_and_common_chord_length_l1695_169530

/-- Circle M passes through points (0,-2) and (4,0), and its center lies on the line x-y=0 -/
def CircleM : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 - 2*p.2 - 8 = 0}

/-- Circle N with equation (x-3)^2 + y^2 = 25 -/
def CircleN : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 25}

/-- The common chord between CircleM and CircleN -/
def CommonChord : Set (ℝ × ℝ) :=
  CircleM ∩ CircleN

theorem circle_m_equation_and_common_chord_length :
  (∀ p : ℝ × ℝ, p ∈ CircleM ↔ p.1^2 + p.2^2 - 2*p.1 - 2*p.2 - 8 = 0) ∧
  (∃ a b : ℝ × ℝ, a ∈ CommonChord ∧ b ∈ CommonChord ∧ 
    Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 2 * Real.sqrt 5) :=
by sorry

end circle_m_equation_and_common_chord_length_l1695_169530


namespace min_value_x_sqrt_9_minus_x_squared_l1695_169508

theorem min_value_x_sqrt_9_minus_x_squared :
  ∃ (x : ℝ), -3 < x ∧ x < 0 ∧
  x * Real.sqrt (9 - x^2) = -9/2 ∧
  ∀ (y : ℝ), -3 < y ∧ y < 0 →
  y * Real.sqrt (9 - y^2) ≥ -9/2 := by
  sorry

end min_value_x_sqrt_9_minus_x_squared_l1695_169508


namespace macaroon_weight_l1695_169569

theorem macaroon_weight
  (total_macaroons : ℕ)
  (num_bags : ℕ)
  (remaining_weight : ℚ)
  (h1 : total_macaroons = 12)
  (h2 : num_bags = 4)
  (h3 : remaining_weight = 45)
  (h4 : total_macaroons % num_bags = 0)  -- Ensures equal distribution
  : ∃ (weight_per_macaroon : ℚ),
    weight_per_macaroon * (total_macaroons - total_macaroons / num_bags) = remaining_weight ∧
    weight_per_macaroon = 5 := by
  sorry

end macaroon_weight_l1695_169569


namespace excluded_students_average_mark_l1695_169545

/-- Proves that the average mark of excluded students is 40 given the conditions of the problem -/
theorem excluded_students_average_mark
  (total_students : ℕ)
  (total_average : ℚ)
  (remaining_average : ℚ)
  (excluded_count : ℕ)
  (h_total_students : total_students = 33)
  (h_total_average : total_average = 90)
  (h_remaining_average : remaining_average = 95)
  (h_excluded_count : excluded_count = 3) :
  let remaining_count := total_students - excluded_count
  let total_marks := total_students * total_average
  let remaining_marks := remaining_count * remaining_average
  let excluded_marks := total_marks - remaining_marks
  excluded_marks / excluded_count = 40 := by
  sorry

end excluded_students_average_mark_l1695_169545


namespace f_increasing_on_interval_l1695_169571

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^2 + 3

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < 4 → f x < f y := by sorry

end f_increasing_on_interval_l1695_169571


namespace smallest_integer_power_l1695_169532

theorem smallest_integer_power (x : ℕ) : (∀ y : ℕ, 27^y ≤ 3^24 → y < x) ∧ 27^x > 3^24 ↔ x = 9 := by
  sorry

end smallest_integer_power_l1695_169532


namespace binomial_expansion_problem_l1695_169523

theorem binomial_expansion_problem (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (Real.sqrt 5 * x - 1)^3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = -64 := by
  sorry

end binomial_expansion_problem_l1695_169523


namespace parabola_properties_l1695_169568

-- Define the parabola
def parabola (m n x : ℝ) : ℝ := m * x^2 - 2 * m^2 * x + n

-- Define the conditions and theorem
theorem parabola_properties
  (m n x₁ x₂ y₁ y₂ : ℝ)
  (h_m : m ≠ 0)
  (h_parabola₁ : parabola m n x₁ = y₁)
  (h_parabola₂ : parabola m n x₂ = y₂) :
  (x₁ = 1 ∧ x₂ = 3 ∧ y₁ = y₂ → 2 = (x₁ + x₂) / 2) ∧
  (x₁ + x₂ > 4 ∧ x₁ < x₂ ∧ y₁ < y₂ → 0 < m ∧ m ≤ 2) :=
by sorry

end parabola_properties_l1695_169568


namespace platform_length_l1695_169588

/-- The length of a platform given a train's speed, length, and crossing time -/
theorem platform_length (train_speed : ℝ) (train_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 →
  train_length = 230.0384 →
  crossing_time = 24 →
  (train_speed * 1000 / 3600) * crossing_time - train_length = 249.9616 := by
  sorry

end platform_length_l1695_169588


namespace secret_spread_theorem_l1695_169531

/-- The number of students who know the secret on a given day -/
def students_knowing_secret (day : ℕ) : ℕ :=
  (3^(day + 1) - 1) / 2

/-- The day of the week when 3280 students know the secret -/
def secret_spread_day : ℕ := 7

/-- Theorem stating that on the 7th day (Sunday), 3280 students know the secret -/
theorem secret_spread_theorem : 
  students_knowing_secret secret_spread_day = 3280 := by
  sorry

end secret_spread_theorem_l1695_169531


namespace intersection_points_polar_l1695_169535

/-- The intersection points of ρ = 2sin θ and ρ cos θ = -√3/2 in polar coordinates -/
theorem intersection_points_polar (θ : Real) (h : 0 ≤ θ ∧ θ < 2 * Real.pi) :
  ∃ (ρ₁ ρ₂ θ₁ θ₂ : Real),
    (ρ₁ = 2 * Real.sin θ₁ ∧ ρ₁ * Real.cos θ₁ = -Real.sqrt 3 / 2) ∧
    (ρ₂ = 2 * Real.sin θ₂ ∧ ρ₂ * Real.cos θ₂ = -Real.sqrt 3 / 2) ∧
    ((ρ₁ = 1 ∧ θ₁ = 5 * Real.pi / 6) ∨ (ρ₁ = Real.sqrt 3 ∧ θ₁ = 2 * Real.pi / 3)) ∧
    ((ρ₂ = 1 ∧ θ₂ = 5 * Real.pi / 6) ∨ (ρ₂ = Real.sqrt 3 ∧ θ₂ = 2 * Real.pi / 3)) ∧
    ρ₁ ≠ ρ₂ :=
by sorry

end intersection_points_polar_l1695_169535


namespace min_value_plus_argmin_l1695_169593

open Real

noncomputable def f (x : ℝ) : ℝ := 9 / (8 * cos (2 * x) + 16) - sin x ^ 2

theorem min_value_plus_argmin (m n : ℝ) 
  (hm : ∀ x, f x ≥ m)
  (hn : f n = m)
  (hp : ∀ x, 0 < x → x < n → f x > m) : 
  m + n = π / 3 := by
  sorry

end min_value_plus_argmin_l1695_169593


namespace intersection_probability_odd_polygon_l1695_169581

/-- 
Given a convex polygon with 2n + 1 vertices, this theorem states that 
the probability of two independently chosen diagonals intersecting
is n(2n - 1) / (3(2n^2 - n - 2)).
-/
theorem intersection_probability_odd_polygon (n : ℕ) : 
  let vertices := 2*n + 1
  let diagonals := vertices * (vertices - 3) / 2
  let intersecting_pairs := (vertices.choose 4)
  let total_pairs := diagonals.choose 2
  (intersecting_pairs : ℚ) / total_pairs = n * (2*n - 1) / (3 * (2*n^2 - n - 2)) :=
by sorry

end intersection_probability_odd_polygon_l1695_169581


namespace quadratic_roots_transformation_l1695_169526

theorem quadratic_roots_transformation (α β : ℝ) (p : ℝ) : 
  (3 * α^2 + 5 * α + 2 = 0) →
  (3 * β^2 + 5 * β + 2 = 0) →
  ((α^2 + 2) + (β^2 + 2) = -(p)) →
  (p = -49/9) := by
sorry

end quadratic_roots_transformation_l1695_169526


namespace repeating_decimal_division_l1695_169590

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ := (a * 10 + b : ℚ) / 99

/-- The fraction representation of 0.overline(63) -/
def frac63 : ℚ := RepeatingDecimal 6 3

/-- The fraction representation of 0.overline(21) -/
def frac21 : ℚ := RepeatingDecimal 2 1

/-- Proves that the division of 0.overline(63) by 0.overline(21) equals 3 -/
theorem repeating_decimal_division : frac63 / frac21 = 3 := by sorry

end repeating_decimal_division_l1695_169590


namespace combined_8th_grade_percentage_is_21_11_percent_l1695_169578

-- Define the schools and their properties
def parkwood_students : ℕ := 150
def maplewood_students : ℕ := 120
def parkwood_8th_grade_percentage : ℚ := 18 / 100
def maplewood_8th_grade_percentage : ℚ := 25 / 100

-- Define the combined percentage of 8th grade students
def combined_8th_grade_percentage : ℚ := 
  (parkwood_8th_grade_percentage * parkwood_students + maplewood_8th_grade_percentage * maplewood_students) / 
  (parkwood_students + maplewood_students)

-- Theorem statement
theorem combined_8th_grade_percentage_is_21_11_percent : 
  combined_8th_grade_percentage = 2111 / 10000 := by
  sorry

end combined_8th_grade_percentage_is_21_11_percent_l1695_169578


namespace leopard_arrangement_l1695_169504

theorem leopard_arrangement (n : ℕ) (h : n = 8) :
  (2 : ℕ) * Nat.factorial (n - 2) = 1440 := by
  sorry

end leopard_arrangement_l1695_169504


namespace min_value_of_function_l1695_169594

theorem min_value_of_function (x : ℝ) (h : x > 2) :
  x + 9 / (x - 2) ≥ 8 ∧ ∃ y > 2, y + 9 / (y - 2) = 8 :=
by sorry

end min_value_of_function_l1695_169594


namespace systematic_sampling_interval_2005_20_l1695_169505

/-- Given a population size and a desired sample size, calculate the sampling interval for systematic sampling -/
def systematicSamplingInterval (populationSize : ℕ) (sampleSize : ℕ) : ℕ :=
  (populationSize / sampleSize : ℕ)

/-- Theorem: For a population of 2005 numbers and a sample size of 20, the systematic sampling interval is 100 -/
theorem systematic_sampling_interval_2005_20 :
  systematicSamplingInterval 2005 20 = 100 := by
  sorry

end systematic_sampling_interval_2005_20_l1695_169505


namespace complex_fraction_value_l1695_169506

theorem complex_fraction_value (a : ℝ) (z : ℂ) : 
  z = (a^2 - 1) + (a - 1) * Complex.I → z.re = 0 → (a^2 + Complex.I) / (1 + a * Complex.I) = Complex.I :=
by sorry

end complex_fraction_value_l1695_169506


namespace five_thursdays_in_august_l1695_169583

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific date in July or August -/
structure Date where
  month : Nat
  day : Nat

/-- Function to get the day of the week for a given date -/
def dayOfWeek (d : Date) : DayOfWeek := sorry

/-- Function to check if a date is a Monday -/
def isMonday (d : Date) : Prop :=
  dayOfWeek d = DayOfWeek.Monday

/-- Function to check if a date is a Thursday -/
def isThursday (d : Date) : Prop :=
  dayOfWeek d = DayOfWeek.Thursday

/-- Theorem stating that if July has five Mondays, then August has five Thursdays -/
theorem five_thursdays_in_august
  (h1 : ∃ d1 d2 d3 d4 d5 : Date,
    d1.month = 7 ∧ d2.month = 7 ∧ d3.month = 7 ∧ d4.month = 7 ∧ d5.month = 7 ∧
    d1.day < d2.day ∧ d2.day < d3.day ∧ d3.day < d4.day ∧ d4.day < d5.day ∧
    isMonday d1 ∧ isMonday d2 ∧ isMonday d3 ∧ isMonday d4 ∧ isMonday d5)
  (h2 : ∀ d : Date, d.month = 7 → d.day ≤ 31)
  (h3 : ∀ d : Date, d.month = 8 → d.day ≤ 31) :
  ∃ d1 d2 d3 d4 d5 : Date,
    d1.month = 8 ∧ d2.month = 8 ∧ d3.month = 8 ∧ d4.month = 8 ∧ d5.month = 8 ∧
    d1.day < d2.day ∧ d2.day < d3.day ∧ d3.day < d4.day ∧ d4.day < d5.day ∧
    isThursday d1 ∧ isThursday d2 ∧ isThursday d3 ∧ isThursday d4 ∧ isThursday d5 :=
sorry

end five_thursdays_in_august_l1695_169583


namespace louise_picture_hanging_l1695_169596

/-- Given a total number of pictures, the number hung horizontally, and the number hung haphazardly,
    calculate the number of pictures hung vertically. -/
def verticalPictures (total horizontal haphazard : ℕ) : ℕ :=
  total - horizontal - haphazard

/-- Theorem stating that given 30 total pictures, with half hung horizontally and 5 haphazardly,
    the number of vertically hung pictures is 10. -/
theorem louise_picture_hanging :
  let total := 30
  let horizontal := total / 2
  let haphazard := 5
  verticalPictures total horizontal haphazard = 10 := by
  sorry

end louise_picture_hanging_l1695_169596
