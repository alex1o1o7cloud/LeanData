import Mathlib

namespace no_arithmetic_mean_l1653_165342

theorem no_arithmetic_mean (f1 f2 f3 : ℚ) : 
  f1 = 5/8 ∧ f2 = 3/4 ∧ f3 = 9/12 →
  (f1 ≠ (f2 + f3) / 2) ∧ (f2 ≠ (f1 + f3) / 2) ∧ (f3 ≠ (f1 + f2) / 2) :=
by sorry

#check no_arithmetic_mean

end no_arithmetic_mean_l1653_165342


namespace triangle_problem_l1653_165327

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Conditions
  a + b + c = 10 →
  Real.sin B + Real.sin C = 4 * Real.sin A →
  -- Part 1
  a = 2 ∧
  -- Additional condition for Part 2
  b * c = 16 →
  -- Part 2
  Real.cos A = 7/8 := by
sorry

end triangle_problem_l1653_165327


namespace toluene_moles_formed_l1653_165382

-- Define the molar mass of benzene
def benzene_molar_mass : ℝ := 78.11

-- Define the chemical reaction
def chemical_reaction (benzene methane toluene hydrogen : ℝ) : Prop :=
  benzene = methane ∧ benzene = toluene ∧ benzene = hydrogen

-- Define the given conditions
def given_conditions (benzene_mass methane_moles : ℝ) : Prop :=
  benzene_mass = 156 ∧ methane_moles = 2

-- Theorem statement
theorem toluene_moles_formed 
  (benzene_mass methane_moles toluene_moles : ℝ)
  (h1 : given_conditions benzene_mass methane_moles)
  (h2 : chemical_reaction (benzene_mass / benzene_molar_mass) methane_moles toluene_moles 2) :
  toluene_moles = 2 := by
  sorry

end toluene_moles_formed_l1653_165382


namespace gcd_372_684_l1653_165346

theorem gcd_372_684 : Nat.gcd 372 684 = 12 := by
  sorry

end gcd_372_684_l1653_165346


namespace max_value_theorem_l1653_165379

theorem max_value_theorem (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 3) :
  a^3 * b + b^3 * a ≤ 81/16 ∧ ∃ (a₀ b₀ : ℝ), 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ a₀ + b₀ = 3 ∧ a₀^3 * b₀ + b₀^3 * a₀ = 81/16 :=
by sorry

end max_value_theorem_l1653_165379


namespace polynomial_remainder_l1653_165343

def polynomial (x : ℝ) : ℝ := 3*x^8 - x^7 - 7*x^5 + 3*x^3 + 4*x^2 - 12*x - 1

def divisor (x : ℝ) : ℝ := 3*x - 9

theorem polynomial_remainder :
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ),
    polynomial x = (divisor x) * (q x) + 15951 := by
  sorry

end polynomial_remainder_l1653_165343


namespace linear_function_constant_point_l1653_165385

theorem linear_function_constant_point :
  ∀ (k : ℝ), (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by
  sorry

end linear_function_constant_point_l1653_165385


namespace waiter_customers_l1653_165376

/-- Calculates the total number of customers for a waiter given the number of tables and customers per table. -/
def total_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) : ℕ :=
  num_tables * (women_per_table + men_per_table)

/-- Theorem stating that a waiter with 5 tables, each having 5 women and 3 men, has a total of 40 customers. -/
theorem waiter_customers :
  total_customers 5 5 3 = 40 := by
  sorry

end waiter_customers_l1653_165376


namespace linear_function_composition_l1653_165371

def is_linear (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

theorem linear_function_composition (f : ℝ → ℝ) :
  is_linear f → (∀ x, f (f x) = 9 * x + 4) → 
  (∀ x, f x = 3 * x + 1) ∨ (∀ x, f x = -3 * x - 2) :=
by sorry

end linear_function_composition_l1653_165371


namespace absolute_value_inequality_solution_l1653_165339

theorem absolute_value_inequality_solution (x : ℝ) :
  (|x - 2| < 1) ↔ (1 < x ∧ x < 3) :=
sorry

end absolute_value_inequality_solution_l1653_165339


namespace rover_spots_l1653_165377

theorem rover_spots (granger cisco rover : ℕ) : 
  granger = 5 * cisco →
  cisco = rover / 2 - 5 →
  granger + cisco = 108 →
  rover = 46 := by
sorry

end rover_spots_l1653_165377


namespace josh_has_eight_riddles_l1653_165316

/-- The number of riddles each person has -/
structure Riddles where
  taso : ℕ
  ivory : ℕ
  josh : ℕ

/-- Given conditions about the riddles -/
def riddle_conditions (r : Riddles) : Prop :=
  r.taso = 24 ∧
  r.taso = 2 * r.ivory ∧
  r.ivory = r.josh + 4

/-- Theorem stating that Josh has 8 riddles -/
theorem josh_has_eight_riddles (r : Riddles) 
  (h : riddle_conditions r) : r.josh = 8 := by
  sorry

end josh_has_eight_riddles_l1653_165316


namespace unique_solution_l1653_165375

theorem unique_solution (a b c : ℝ) : 
  a > 2 ∧ b > 2 ∧ c > 2 →
  (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 48 →
  a = 7 ∧ b = 5 ∧ c = 3 :=
by sorry

end unique_solution_l1653_165375


namespace lines_perp_to_plane_are_parallel_l1653_165381

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- State the theorem
theorem lines_perp_to_plane_are_parallel
  (m n : Line) (α : Plane) 
  (h_diff : m ≠ n)
  (h_m_perp : perp m α)
  (h_n_perp : perp n α) :
  para m n :=
sorry

end lines_perp_to_plane_are_parallel_l1653_165381


namespace infinite_geometric_series_first_term_l1653_165357

theorem infinite_geometric_series_first_term
  (a r : ℝ)
  (h1 : 0 ≤ r ∧ r < 1)  -- Condition for convergence of infinite geometric series
  (h2 : a / (1 - r) = 15)  -- Sum of the series
  (h3 : a^2 / (1 - r^2) = 45)  -- Sum of the squares of the terms
  : a = 5 := by
  sorry

end infinite_geometric_series_first_term_l1653_165357


namespace eggs_to_buy_l1653_165341

def total_eggs_needed : ℕ := 222
def eggs_received : ℕ := 155

theorem eggs_to_buy : total_eggs_needed - eggs_received = 67 := by
  sorry

end eggs_to_buy_l1653_165341


namespace solution_set_a_1_no_a_for_all_reals_l1653_165384

-- Define the inequality function
def inequality (a : ℝ) (x : ℝ) : Prop :=
  |a*x - 1| + |a*x - a| ≥ 2

-- Part 1: Solution set when a = 1
theorem solution_set_a_1 :
  ∀ x : ℝ, inequality 1 x ↔ (x ≤ 0 ∨ x ≥ 2) :=
sorry

-- Part 2: No a > 0 makes the solution set ℝ
theorem no_a_for_all_reals :
  ¬ ∃ a : ℝ, a > 0 ∧ (∀ x : ℝ, inequality a x) :=
sorry

end solution_set_a_1_no_a_for_all_reals_l1653_165384


namespace min_values_xy_and_x_plus_y_l1653_165333

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : 2/x + 8/y = 1) : 
  (x * y ≥ 64 ∧ x + y ≥ 18) ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ * y₁ = 64 ∧ x₂ + y₂ = 18 ∧
   2/x₁ + 8/y₁ = 1 ∧ 2/x₂ + 8/y₂ = 1 ∧
   x₁ > 0 ∧ y₁ > 0 ∧ x₂ > 0 ∧ y₂ > 0) :=
by sorry

end min_values_xy_and_x_plus_y_l1653_165333


namespace units_digit_of_1389_pow_1247_l1653_165354

theorem units_digit_of_1389_pow_1247 (n : ℕ) :
  n = 1389^1247 → n % 10 = 9 := by
  sorry

end units_digit_of_1389_pow_1247_l1653_165354


namespace orange_apple_cost_l1653_165321

/-- The cost of oranges and apples given specific quantities and prices -/
theorem orange_apple_cost (orange_price apple_price : ℕ) 
  (h1 : 6 * orange_price + 5 * apple_price = 419)
  (h2 : orange_price = 29)
  (h3 : apple_price = 29) :
  5 * orange_price + 7 * apple_price = 348 := by
  sorry

#check orange_apple_cost

end orange_apple_cost_l1653_165321


namespace better_misspellings_l1653_165304

/-- The word to be considered -/
def word : String := "better"

/-- The number of distinct letters in the word -/
def distinct_letters : Nat := 4

/-- The total number of letters in the word -/
def total_letters : Nat := 6

/-- The number of repeated letters in the word -/
def repeated_letters : Nat := 2

/-- The number of repetitions for each repeated letter -/
def repetitions : Nat := 2

/-- The number of misspellings of the word "better" -/
def misspellings : Nat := 179

theorem better_misspellings :
  (Nat.factorial total_letters / (Nat.factorial repetitions ^ repeated_letters)) - 1 = misspellings :=
sorry

end better_misspellings_l1653_165304


namespace smallest_n_for_factorization_factorization_exists_for_31_l1653_165334

theorem smallest_n_for_factorization : 
  ∀ n : ℤ, n < 31 → 
  ¬∃ A B : ℤ, ∀ x : ℝ, 5 * x^2 + n * x + 48 = (5 * x + A) * (x + B) :=
by sorry

theorem factorization_exists_for_31 : 
  ∃ A B : ℤ, ∀ x : ℝ, 5 * x^2 + 31 * x + 48 = (5 * x + A) * (x + B) :=
by sorry

end smallest_n_for_factorization_factorization_exists_for_31_l1653_165334


namespace num_connecting_lines_correct_l1653_165313

/-- The number of straight lines connecting the intersection points of n intersecting lines -/
def num_connecting_lines (n : ℕ) : ℚ :=
  (n^2 * (n-1)^2 - 2*n * (n-1)) / 8

/-- Theorem stating that num_connecting_lines gives the correct number of lines -/
theorem num_connecting_lines_correct (n : ℕ) :
  num_connecting_lines n = (n^2 * (n-1)^2 - 2*n * (n-1)) / 8 :=
by sorry

end num_connecting_lines_correct_l1653_165313


namespace parallel_vectors_m_value_l1653_165393

/-- Given two parallel vectors a and b, prove that m = 1/2 --/
theorem parallel_vectors_m_value (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![m, -1]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → m = 1/2 := by
  sorry

end parallel_vectors_m_value_l1653_165393


namespace mixture_ratio_l1653_165372

/-- Given a mixture of liquids p and q, prove that the initial ratio is 3:2 -/
theorem mixture_ratio (p q : ℝ) : 
  p + q = 25 →                      -- Initial total volume
  p / (q + 2) = 5 / 4 →             -- Ratio after adding 2 liters of q
  p / q = 3 / 2 :=                  -- Initial ratio
by sorry

end mixture_ratio_l1653_165372


namespace mirror_image_properties_l1653_165305

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define mirror image operations
def mirrorY (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

def mirrorX (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

def mirrorOrigin (p : Point2D) : Point2D :=
  { x := -p.x, y := -p.y }

def mirrorYEqualsX (p : Point2D) : Point2D :=
  { x := p.y, y := p.x }

def mirrorYEqualsNegX (p : Point2D) : Point2D :=
  { x := -p.y, y := -p.x }

-- Theorem stating the mirror image properties
theorem mirror_image_properties (p : Point2D) :
  (mirrorY p = { x := -p.x, y := p.y }) ∧
  (mirrorX p = { x := p.x, y := -p.y }) ∧
  (mirrorOrigin p = { x := -p.x, y := -p.y }) ∧
  (mirrorYEqualsX p = { x := p.y, y := p.x }) ∧
  (mirrorYEqualsNegX p = { x := -p.y, y := -p.x }) :=
by sorry

end mirror_image_properties_l1653_165305


namespace decimal_to_binary_89_l1653_165395

theorem decimal_to_binary_89 : 
  (89 : ℕ).digits 2 = [1, 0, 0, 1, 1, 0, 1] :=
sorry

end decimal_to_binary_89_l1653_165395


namespace no_blue_in_red_triangle_l1653_165366

-- Define the color of a point
inductive Color
| Red
| Blue

-- Define a point in the plane with integer coordinates
structure Point where
  x : Int
  y : Int

-- Define the coloring function
def coloring : Point → Color := sorry

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define a predicate for a point being inside a triangle
def inside_triangle (p a b c : Point) : Prop := sorry

-- State the conditions
axiom condition1 : ∀ (p : Point), coloring p = Color.Red ∨ coloring p = Color.Blue

axiom condition2 : ∀ (p q : Point),
  coloring p = Color.Red → coloring q = Color.Red →
  ∀ (r : Point), inside_triangle r p q q → coloring r ≠ Color.Blue

axiom condition3 : ∀ (p q : Point),
  coloring p = Color.Blue → coloring q = Color.Blue →
  distance p q = 2 →
  coloring {x := (p.x + q.x) / 2, y := (p.y + q.y) / 2} = Color.Blue

-- State the theorem
theorem no_blue_in_red_triangle (a b c : Point) :
  coloring a = Color.Red → coloring b = Color.Red → coloring c = Color.Red →
  ∀ (p : Point), inside_triangle p a b c → coloring p ≠ Color.Blue :=
sorry

end no_blue_in_red_triangle_l1653_165366


namespace circle_chords_and_regions_l1653_165369

/-- The number of chords that can be drawn between n points on a circle's circumference -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- The number of regions formed inside a circle by chords connecting n points on its circumference -/
def num_regions (n : ℕ) : ℕ := 1 + n.choose 2 + n.choose 4

theorem circle_chords_and_regions (n : ℕ) (h : n = 10) :
  num_chords n = 45 ∧ num_regions n = 256 := by
  sorry

#eval num_chords 10
#eval num_regions 10

end circle_chords_and_regions_l1653_165369


namespace gcd_power_two_minus_one_l1653_165344

theorem gcd_power_two_minus_one :
  Nat.gcd (2^1510 - 1) (2^1500 - 1) = 2^10 - 1 := by
  sorry

end gcd_power_two_minus_one_l1653_165344


namespace car_expense_difference_l1653_165340

/-- The difference in car expenses between Alberto and Samara -/
def expense_difference (alberto_expense : ℕ) (samara_oil : ℕ) (samara_tires : ℕ) (samara_detailing : ℕ) : ℕ :=
  alberto_expense - (samara_oil + samara_tires + samara_detailing)

/-- Theorem stating the difference in car expenses between Alberto and Samara -/
theorem car_expense_difference :
  expense_difference 2457 25 467 79 = 1886 := by
  sorry

end car_expense_difference_l1653_165340


namespace science_club_enrollment_l1653_165362

theorem science_club_enrollment (total : ℕ) (chem : ℕ) (bio : ℕ) (both : ℕ) 
  (h1 : total = 75)
  (h2 : chem = 45)
  (h3 : bio = 30)
  (h4 : both = 18) :
  total - (chem + bio - both) = 18 := by
  sorry

end science_club_enrollment_l1653_165362


namespace triangle_problem_l1653_165359

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle
  A + B + C = π ∧
  a ≠ b ∧
  c = Real.sqrt 3 ∧
  Real.sqrt 3 * (Real.cos A)^2 - Real.sqrt 3 * (Real.cos B)^2 = Real.sin A * Real.cos A - Real.sin B * Real.cos B ∧
  Real.sin A = 4/5 →
  C = π/6 ∧
  1/2 * a * c * Real.sin B = (24 * Real.sqrt 3 + 18) / 25 :=
by sorry

end triangle_problem_l1653_165359


namespace min_omega_for_symmetry_axis_l1653_165383

/-- The minimum positive value of ω for which f(x) = sin(ωx + π/6) has a symmetry axis at x = π/12 -/
theorem min_omega_for_symmetry_axis : ∃ (ω_min : ℝ), 
  (∀ (ω : ℝ), ω > 0 → (∃ (k : ℤ), ω = 12 * k + 4)) → 
  (∀ (ω : ℝ), ω > 0 → ω ≥ ω_min) → 
  ω_min = 4 := by
sorry

end min_omega_for_symmetry_axis_l1653_165383


namespace negation_equivalence_l1653_165360

theorem negation_equivalence :
  (¬ ∃ x : ℝ, 2 * x^2 - 1 ≤ 0) ↔ (∀ x : ℝ, 2 * x^2 - 1 > 0) :=
by sorry

end negation_equivalence_l1653_165360


namespace set_operations_and_intersection_l1653_165347

def A : Set ℝ := {x | 4 ≤ x ∧ x < 8}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem set_operations_and_intersection :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((Aᶜ ∩ B) = {x | (8 ≤ x ∧ x < 10) ∨ (2 < x ∧ x < 4)}) ∧
  (∀ a : ℝ, (A ∩ C a).Nonempty ↔ a > 4) :=
sorry

end set_operations_and_intersection_l1653_165347


namespace planes_perp_to_line_are_parallel_lines_perp_to_plane_are_parallel_lines_perp_to_line_not_always_parallel_planes_perp_to_plane_not_always_parallel_l1653_165370

-- Define basic geometric objects
variable (Point Line Plane : Type)

-- Define perpendicular and parallel relations
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (perpendicular_line_line : Line → Line → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Theorem for proposition ②
theorem planes_perp_to_line_are_parallel 
  (l : Line) (p1 p2 : Plane) 
  (h1 : perpendicular_plane_line p1 l) 
  (h2 : perpendicular_plane_line p2 l) : 
  parallel_plane p1 p2 :=
sorry

-- Theorem for proposition ③
theorem lines_perp_to_plane_are_parallel 
  (p : Plane) (l1 l2 : Line) 
  (h1 : perpendicular_line_plane l1 p) 
  (h2 : perpendicular_line_plane l2 p) : 
  parallel_line l1 l2 :=
sorry

-- Theorem for proposition ① (to be proven false)
theorem lines_perp_to_line_not_always_parallel 
  (l : Line) (l1 l2 : Line) 
  (h1 : perpendicular_line_line l1 l) 
  (h2 : perpendicular_line_line l2 l) : 
  ¬(parallel_line l1 l2) :=
sorry

-- Theorem for proposition ④ (to be proven false)
theorem planes_perp_to_plane_not_always_parallel 
  (p : Plane) (p1 p2 : Plane) 
  (h1 : perpendicular_plane_plane p1 p) 
  (h2 : perpendicular_plane_plane p2 p) : 
  ¬(parallel_plane p1 p2) :=
sorry

end planes_perp_to_line_are_parallel_lines_perp_to_plane_are_parallel_lines_perp_to_line_not_always_parallel_planes_perp_to_plane_not_always_parallel_l1653_165370


namespace students_playing_neither_sport_l1653_165363

theorem students_playing_neither_sport
  (total : ℕ)
  (football : ℕ)
  (tennis : ℕ)
  (both : ℕ)
  (h1 : total = 50)
  (h2 : football = 32)
  (h3 : tennis = 28)
  (h4 : both = 24) :
  total - (football + tennis - both) = 14 :=
by sorry

end students_playing_neither_sport_l1653_165363


namespace joan_book_sale_l1653_165391

/-- Given that Joan initially gathered 33 books and found 26 more,
    prove that the total number of books she has for sale is 59. -/
theorem joan_book_sale (initial_books : ℕ) (additional_books : ℕ) 
  (h1 : initial_books = 33) (h2 : additional_books = 26) : 
  initial_books + additional_books = 59 := by
  sorry

end joan_book_sale_l1653_165391


namespace xiaohua_mother_age_ratio_l1653_165329

/-- The number of years ago when a mother's age was 5 times her child's age, given their current ages -/
def years_ago (child_age mother_age : ℕ) : ℕ :=
  child_age - (mother_age - child_age) / (5 - 1)

/-- Theorem stating that for Xiaohua (12 years old) and his mother (36 years old), 
    the mother's age was 5 times Xiaohua's age 6 years ago -/
theorem xiaohua_mother_age_ratio : years_ago 12 36 = 6 := by
  sorry

end xiaohua_mother_age_ratio_l1653_165329


namespace sample_size_eq_selected_cards_l1653_165318

/-- Represents a statistical study of student report cards -/
structure ReportCardStudy where
  totalStudents : ℕ
  selectedCards : ℕ
  h_total : totalStudents = 1000
  h_selected : selectedCards = 100
  h_selected_le_total : selectedCards ≤ totalStudents

/-- The sample size of a report card study is equal to the number of selected cards -/
theorem sample_size_eq_selected_cards (study : ReportCardStudy) : 
  study.selectedCards = 100 := by
  sorry

#check sample_size_eq_selected_cards

end sample_size_eq_selected_cards_l1653_165318


namespace identify_real_coins_l1653_165308

/-- Represents the result of weighing two coins -/
inductive WeighResult
| Equal : WeighResult
| LeftHeavier : WeighResult
| RightHeavier : WeighResult

/-- Represents a coin -/
structure Coin :=
  (id : Nat)
  (isReal : Bool)

/-- Represents the balance scale that always shows an incorrect result -/
def incorrectBalance (left right : Coin) : WeighResult :=
  sorry

/-- The main theorem to prove -/
theorem identify_real_coins 
  (coins : Finset Coin) 
  (h_count : coins.card = 100) 
  (h_real : ∃ (fake : Coin), fake ∈ coins ∧ 
    (∀ c ∈ coins, c ≠ fake → c.isReal) ∧ 
    (¬fake.isReal)) : 
  ∃ (realCoins : Finset Coin), realCoins ⊆ coins ∧ realCoins.card = 98 ∧ 
    (∀ c ∈ realCoins, c.isReal) :=
  sorry

end identify_real_coins_l1653_165308


namespace function_properties_l1653_165324

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2 * |x - a|

theorem function_properties (a : ℝ) :
  (∀ x, f a x = f a (-x)) ↔ a = 0 ∧
  (a = 1/2 → ∀ x, (x ≤ -1 ∨ (1/2 ≤ x ∧ x ≤ 1)) → 
    ∀ y, y < x → f (1/2) y < f (1/2) x) ∧
  (a > 0 → (∀ x : ℝ, x ≥ 0 → f a (x - 1) ≥ 2 * f a x) ↔ 
    (Real.sqrt 6 - 2 ≤ a ∧ a ≤ 1/2)) :=
by sorry

end function_properties_l1653_165324


namespace square_diff_product_l1653_165350

theorem square_diff_product (m n : ℝ) (h1 : m - n = 4) (h2 : m * n = -3) :
  (m^2 - 4) * (n^2 - 4) = -15 := by sorry

end square_diff_product_l1653_165350


namespace cloth_sale_problem_l1653_165361

/-- Proves that the number of meters of cloth sold is 400 -/
theorem cloth_sale_problem (total_selling_price : ℕ) (loss_per_meter : ℕ) (cost_price_per_meter : ℕ) :
  total_selling_price = 18000 →
  loss_per_meter = 5 →
  cost_price_per_meter = 50 →
  (total_selling_price / (cost_price_per_meter - loss_per_meter) : ℕ) = 400 := by
  sorry

end cloth_sale_problem_l1653_165361


namespace inequality_proof_l1653_165358

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_cond : |Real.sqrt (a * b) - Real.sqrt (c * d)| ≤ 2) : 
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) := by
  sorry

end inequality_proof_l1653_165358


namespace arithmetic_sum_l1653_165348

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 2 →
  a 2 + a 3 = 13 →
  a 4 + a 5 + a 6 = 42 :=
by
  sorry

end arithmetic_sum_l1653_165348


namespace cylinder_volume_change_l1653_165392

/-- Given a cylinder with original volume of 15 cubic feet, 
    prove that tripling its radius and quadrupling its height 
    results in a new volume of 540 cubic feet. -/
theorem cylinder_volume_change (r h : ℝ) : 
  r > 0 → h > 0 → π * r^2 * h = 15 → π * (3*r)^2 * (4*h) = 540 := by
  sorry

end cylinder_volume_change_l1653_165392


namespace decreasing_f_iff_a_in_range_l1653_165300

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem decreasing_f_iff_a_in_range (a : ℝ) :
  is_decreasing (f a) ↔ 0 < a ∧ a ≤ 1/4 :=
sorry

end decreasing_f_iff_a_in_range_l1653_165300


namespace complex_equation_sum_l1653_165337

theorem complex_equation_sum (a b : ℝ) : 
  (a + Complex.I) * Complex.I = b + (5 / (2 - Complex.I)) → a + b = -2 := by
  sorry

end complex_equation_sum_l1653_165337


namespace area_of_U_l1653_165309

/-- A regular octagon centered at the origin in the complex plane -/
def regularOctagon : Set ℂ :=
  sorry

/-- The distance between opposite sides of the octagon is 2 units -/
def oppositeDistanceIs2 : ℝ :=
  sorry

/-- One pair of sides of the octagon is parallel to the real axis -/
def sideParallelToRealAxis : Prop :=
  sorry

/-- The region outside the octagon -/
def T : Set ℂ :=
  {z : ℂ | z ∉ regularOctagon}

/-- The set of reciprocals of points in T -/
def U : Set ℂ :=
  {w : ℂ | ∃ z ∈ T, w = 1 / z}

/-- The area of a set in the complex plane -/
def area : Set ℂ → ℝ :=
  sorry

theorem area_of_U : area U = π / 2 :=
  sorry

end area_of_U_l1653_165309


namespace log_product_change_base_l1653_165349

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_product_change_base 
  (a b c : ℝ) (m n : ℝ) 
  (h1 : log b a = m) 
  (h2 : log c b = n) 
  (h3 : a > 0) (h4 : b > 1) (h5 : c > 1) :
  log (b * c) (a * b) = n * (m + 1) / (n + 1) := by
sorry

end log_product_change_base_l1653_165349


namespace journey_speeds_correct_l1653_165353

/-- Represents the speeds and meeting times of pedestrians and cyclists --/
structure JourneyData where
  distance : ℝ
  pedestrian_start : ℝ
  cyclist1_start : ℝ
  cyclist2_start : ℝ
  pedestrian_speed : ℝ
  cyclist_speed : ℝ

/-- Checks if the given speeds satisfy the journey conditions --/
def satisfies_conditions (data : JourneyData) : Prop :=
  let first_meeting_time := data.cyclist1_start + (data.distance / 2 - data.pedestrian_speed * (data.cyclist1_start - data.pedestrian_start)) / (data.cyclist_speed - data.pedestrian_speed)
  let second_meeting_time := first_meeting_time + 1
  let pedestrian_distance_at_second_meeting := data.pedestrian_speed * (second_meeting_time - data.pedestrian_start)
  let cyclist2_distance := data.cyclist_speed * (second_meeting_time - data.cyclist2_start)
  first_meeting_time - data.pedestrian_start > 0 ∧
  first_meeting_time - data.cyclist1_start > 0 ∧
  second_meeting_time - data.cyclist2_start > 0 ∧
  pedestrian_distance_at_second_meeting + cyclist2_distance = data.distance

/-- The main theorem stating that the given speeds satisfy the journey conditions --/
theorem journey_speeds_correct : ∃ (data : JourneyData),
  data.distance = 40 ∧
  data.pedestrian_start = 0 ∧
  data.cyclist1_start = 10/3 ∧
  data.cyclist2_start = 4.5 ∧
  data.pedestrian_speed = 5 ∧
  data.cyclist_speed = 30 ∧
  satisfies_conditions data := by
  sorry


end journey_speeds_correct_l1653_165353


namespace proposition_truth_count_l1653_165399

theorem proposition_truth_count (a b c : ℝ) : 
  (∃ (x y z : ℝ), x * z^2 > y * z^2 ∧ x ≤ y) ∨ 
  (∀ (x y z : ℝ), x > y → x * z^2 > y * z^2) ∨
  (∀ (x y z : ℝ), x ≤ y → x * z^2 ≤ y * z^2) :=
by sorry

end proposition_truth_count_l1653_165399


namespace square_side_length_l1653_165311

/-- Given a circle with area 100 and a square whose perimeter equals the circle's area,
    the length of one side of the square is 25. -/
theorem square_side_length (circle_area : ℝ) (square_perimeter : ℝ) :
  circle_area = 100 →
  square_perimeter = circle_area →
  square_perimeter = 4 * 25 :=
by
  sorry

#check square_side_length

end square_side_length_l1653_165311


namespace equation_solution_l1653_165312

theorem equation_solution : ∃ x₁ x₂ : ℝ, x₁ = -5 ∧ x₂ = 3 ∧ 
  (∀ x : ℝ, (x + 3) * (x - 1) = 12 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end equation_solution_l1653_165312


namespace no_equilateral_triangle_2D_exists_regular_tetrahedron_3D_l1653_165352

-- Define a 2D point with integer coordinates
structure Point2D where
  x : ℤ
  y : ℤ

-- Define a 3D point with integer coordinates
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

-- Function to calculate the square of the distance between two 2D points
def distanceSquared2D (p1 p2 : Point2D) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to calculate the square of the distance between two 3D points
def distanceSquared3D (p1 p2 : Point3D) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

-- Theorem: No equilateral triangle exists with vertices at integer coordinate points in 2D
theorem no_equilateral_triangle_2D :
  ¬∃ (a b c : Point2D), 
    distanceSquared2D a b = distanceSquared2D b c ∧
    distanceSquared2D b c = distanceSquared2D c a ∧
    distanceSquared2D c a = distanceSquared2D a b :=
sorry

-- Theorem: A regular tetrahedron exists with vertices at integer coordinate points in 3D
theorem exists_regular_tetrahedron_3D :
  ∃ (a b c d : Point3D),
    distanceSquared3D a b = distanceSquared3D b c ∧
    distanceSquared3D b c = distanceSquared3D c d ∧
    distanceSquared3D c d = distanceSquared3D d a ∧
    distanceSquared3D d a = distanceSquared3D a b ∧
    distanceSquared3D a c = distanceSquared3D b d :=
sorry

end no_equilateral_triangle_2D_exists_regular_tetrahedron_3D_l1653_165352


namespace calculation_proof_l1653_165394

theorem calculation_proof : (((20^10 / 20^9)^3 * 10^6) / 2^12) = 1953125 := by
  sorry

end calculation_proof_l1653_165394


namespace max_value_complex_fraction_l1653_165355

theorem max_value_complex_fraction (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (max_val : ℝ), max_val = (2 * Real.sqrt 5) / 3 ∧
  ∀ (w : ℂ), Complex.abs w = 1 →
    Complex.abs ((w + Complex.I) / (w + 2)) ≤ max_val :=
by sorry

end max_value_complex_fraction_l1653_165355


namespace major_axis_length_l1653_165314

/-- Represents a right circular cylinder. -/
structure RightCircularCylinder where
  radius : ℝ

/-- Represents an ellipse formed by the intersection of a plane and a cylinder. -/
structure Ellipse where
  minorAxis : ℝ
  majorAxis : ℝ

/-- The ellipse formed by the intersection of a plane and a right circular cylinder. -/
def intersectionEllipse (c : RightCircularCylinder) : Ellipse where
  minorAxis := 2 * c.radius
  majorAxis := 2 * c.radius * 1.5

theorem major_axis_length 
  (c : RightCircularCylinder) 
  (h : c.radius = 1) :
  (intersectionEllipse c).majorAxis = 3 := by
  sorry

end major_axis_length_l1653_165314


namespace geometric_series_equality_l1653_165365

def C (n : ℕ) : ℚ := 2048 * (1 - (1 / 2^n))

def D (n : ℕ) : ℚ := (6144 / 3) * (1 - (1 / (-2)^n))

theorem geometric_series_equality (n : ℕ) (h : n ≥ 1) : C n = D n ↔ n = 6 := by
  sorry

end geometric_series_equality_l1653_165365


namespace duck_price_is_correct_l1653_165335

/-- The price of a duck given the conditions of the problem -/
def duck_price : ℝ :=
  let chicken_price : ℝ := 8
  let num_chickens : ℕ := 5
  let num_ducks : ℕ := 2
  let additional_earnings : ℝ := 60
  10

theorem duck_price_is_correct :
  let chicken_price : ℝ := 8
  let num_chickens : ℕ := 5
  let num_ducks : ℕ := 2
  let additional_earnings : ℝ := 60
  let total_earnings := chicken_price * num_chickens + duck_price * num_ducks
  let wheelbarrow_cost := total_earnings / 2
  wheelbarrow_cost * 2 = additional_earnings ∧ duck_price = 10 := by
  sorry

end duck_price_is_correct_l1653_165335


namespace square_root_of_nine_l1653_165328

theorem square_root_of_nine : ∃ x : ℝ, x ^ 2 = 9 ∧ x = 3 := by sorry

end square_root_of_nine_l1653_165328


namespace three_distinct_roots_reciprocal_l1653_165374

theorem three_distinct_roots_reciprocal (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    a * x^5 + b * x^4 + c = 0 ∧
    a * y^5 + b * y^4 + c = 0 ∧
    a * z^5 + b * z^4 + c = 0) →
  (∃ u v w : ℝ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
    c * u^5 + b * u + a = 0 ∧
    c * v^5 + b * v + a = 0 ∧
    c * w^5 + b * w + a = 0) :=
by sorry

end three_distinct_roots_reciprocal_l1653_165374


namespace truck_distance_proof_l1653_165390

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The distance traveled by the truck -/
def truck_distance : ℕ := arithmetic_sum 5 7 30

theorem truck_distance_proof : truck_distance = 3195 := by
  sorry

end truck_distance_proof_l1653_165390


namespace sequence_properties_l1653_165364

theorem sequence_properties :
  (∀ n m : ℕ, (2 * n)^2 + 1 ≠ 3 * m^2) ∧
  (∀ p q : ℕ, p^2 + 1 ≠ 7 * q^2) := by
  sorry

end sequence_properties_l1653_165364


namespace f_symmetry_iff_a_eq_one_l1653_165319

/-- The function f(x) defined as -|x-a| -/
def f (a : ℝ) (x : ℝ) : ℝ := -|x - a|

/-- Theorem stating that f(1+x) = f(1-x) for all x is equivalent to a = 1 -/
theorem f_symmetry_iff_a_eq_one (a : ℝ) :
  (∀ x, f a (1 + x) = f a (1 - x)) ↔ a = 1 := by
  sorry

end f_symmetry_iff_a_eq_one_l1653_165319


namespace distance_between_trees_l1653_165378

theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) 
  (h1 : yard_length = 180)
  (h2 : num_trees = 11)
  (h3 : num_trees ≥ 2) :
  let distance := yard_length / (num_trees - 1)
  distance = 18 := by
  sorry

end distance_between_trees_l1653_165378


namespace systematic_sampling_5_from_100_correct_sequence_l1653_165320

/-- Systematic sampling function that returns the nth selected individual -/
def systematicSample (totalPopulation : ℕ) (sampleSize : ℕ) (n : ℕ) : ℕ :=
  n * (totalPopulation / sampleSize)

/-- Theorem stating that systematic sampling of 5 from 100 yields the correct sequence -/
theorem systematic_sampling_5_from_100 :
  let totalPopulation : ℕ := 100
  let sampleSize : ℕ := 5
  (systematicSample totalPopulation sampleSize 1 = 20) ∧
  (systematicSample totalPopulation sampleSize 2 = 40) ∧
  (systematicSample totalPopulation sampleSize 3 = 60) ∧
  (systematicSample totalPopulation sampleSize 4 = 80) ∧
  (systematicSample totalPopulation sampleSize 5 = 100) :=
by
  sorry

/-- Theorem stating that the correct sequence is 10, 30, 50, 70, 90 -/
theorem correct_sequence :
  let totalPopulation : ℕ := 100
  let sampleSize : ℕ := 5
  (systematicSample totalPopulation sampleSize 1 - 10 = 10) ∧
  (systematicSample totalPopulation sampleSize 2 - 10 = 30) ∧
  (systematicSample totalPopulation sampleSize 3 - 10 = 50) ∧
  (systematicSample totalPopulation sampleSize 4 - 10 = 70) ∧
  (systematicSample totalPopulation sampleSize 5 - 10 = 90) :=
by
  sorry

end systematic_sampling_5_from_100_correct_sequence_l1653_165320


namespace hyperbola_asymptotes_l1653_165310

/-- Given a hyperbola with equation x²/144 - y²/81 = 1, its asymptotes are y = ±(3/4)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ,
  x^2 / 144 - y^2 / 81 = 1 →
  ∃ m : ℝ, m > 0 ∧ (y = m * x ∨ y = -m * x) ∧ m = 3/4 := by
sorry

end hyperbola_asymptotes_l1653_165310


namespace quadratic_two_distinct_roots_l1653_165356

theorem quadratic_two_distinct_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a - 3) * x₁^2 - 4 * x₁ - 1 = 0 ∧ (a - 3) * x₂^2 - 4 * x₂ - 1 = 0) ↔
  (a > -1 ∧ a ≠ 3) :=
by sorry

end quadratic_two_distinct_roots_l1653_165356


namespace sqrt_a_squared_plus_one_is_quadratic_radical_l1653_165332

/-- A function is a quadratic radical if it can be expressed as the square root of a non-negative real-valued expression. -/
def is_quadratic_radical (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g x ≥ 0) ∧ (∀ x, f x = Real.sqrt (g x))

/-- The function f(a) = √(a² + 1) is a quadratic radical. -/
theorem sqrt_a_squared_plus_one_is_quadratic_radical :
  is_quadratic_radical (fun a => Real.sqrt (a^2 + 1)) :=
sorry

end sqrt_a_squared_plus_one_is_quadratic_radical_l1653_165332


namespace compound_interest_rate_l1653_165396

theorem compound_interest_rate (P : ℝ) (r : ℝ) : 
  P > 0 →
  r > 0 →
  P * (1 + r)^2 - P = 492 →
  P * (1 + r)^2 = 5292 →
  r = 0.05 := by
sorry

end compound_interest_rate_l1653_165396


namespace complex_power_difference_l1653_165380

theorem complex_power_difference (x : ℂ) : 
  x - 1 / x = 2 * Complex.I → x^729 - 1 / x^729 = 2 * Complex.I :=
by sorry

end complex_power_difference_l1653_165380


namespace inverse_f_at_120_l1653_165326

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 + 9

-- State the theorem
theorem inverse_f_at_120 :
  ∃ (y : ℝ), f y = 120 ∧ y = (37 : ℝ)^(1/3) :=
sorry

end inverse_f_at_120_l1653_165326


namespace f_decreasing_and_k_maximum_l1653_165325

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x + 1)) / x

theorem f_decreasing_and_k_maximum :
  (∀ x > 0, (deriv f) x < 0) ∧
  (∀ x > 0, f x > 3 / (x + 1)) ∧
  (¬ ∃ k : ℕ, k > 3 ∧ ∀ x > 0, f x > (k : ℝ) / (x + 1)) :=
by sorry

end f_decreasing_and_k_maximum_l1653_165325


namespace consecutive_product_not_power_l1653_165388

theorem consecutive_product_not_power (x a n : ℕ) : 
  a ≥ 2 → n ≥ 2 → (x - 1) * x * (x + 1) ≠ a^n := by
  sorry

end consecutive_product_not_power_l1653_165388


namespace lake_radius_l1653_165398

/-- Given a circular lake with a diameter of 26 meters, its radius is 13 meters. -/
theorem lake_radius (lake_diameter : ℝ) (h : lake_diameter = 26) : 
  lake_diameter / 2 = 13 := by sorry

end lake_radius_l1653_165398


namespace bake_sale_earnings_l1653_165397

theorem bake_sale_earnings (total : ℝ) (ingredients_cost shelter_donation : ℝ) 
  (h1 : ingredients_cost = 100)
  (h2 : shelter_donation = (total - ingredients_cost) / 2 + 10)
  (h3 : shelter_donation = 160) : 
  total = 400 := by
sorry

end bake_sale_earnings_l1653_165397


namespace least_integer_b_for_quadratic_range_l1653_165389

theorem least_integer_b_for_quadratic_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + b*x + 20 ≠ -10) ↔ b ≤ -10 :=
sorry

end least_integer_b_for_quadratic_range_l1653_165389


namespace gate_buyers_pay_more_l1653_165306

/-- Calculates the difference in total amount paid between gate buyers and pre-buyers --/
def ticketPriceDifference (preBuyerCount : ℕ) (gateBuyerCount : ℕ) (preBuyerPrice : ℕ) (gateBuyerPrice : ℕ) : ℕ :=
  gateBuyerCount * gateBuyerPrice - preBuyerCount * preBuyerPrice

theorem gate_buyers_pay_more :
  ticketPriceDifference 20 30 155 200 = 2900 := by
  sorry

end gate_buyers_pay_more_l1653_165306


namespace landscape_length_is_240_l1653_165331

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  breadth : ℝ
  length : ℝ
  playgroundArea : ℝ
  totalArea : ℝ

/-- The length of the landscape is 8 times its breadth -/
def lengthIsTotalRule (l : Landscape) : Prop :=
  l.length = 8 * l.breadth

/-- The playground occupies 1/6 of the total landscape area -/
def playgroundRule (l : Landscape) : Prop :=
  l.playgroundArea = l.totalArea / 6

/-- The playground has an area of 1200 square meters -/
def playgroundAreaRule (l : Landscape) : Prop :=
  l.playgroundArea = 1200

/-- The total area of the landscape is the product of its length and breadth -/
def totalAreaRule (l : Landscape) : Prop :=
  l.totalArea = l.length * l.breadth

/-- Theorem: Given the conditions, the length of the landscape is 240 meters -/
theorem landscape_length_is_240 (l : Landscape) 
  (h1 : lengthIsTotalRule l) 
  (h2 : playgroundRule l) 
  (h3 : playgroundAreaRule l) 
  (h4 : totalAreaRule l) : 
  l.length = 240 := by
  sorry

end landscape_length_is_240_l1653_165331


namespace permutation_expressions_l1653_165315

open Nat

-- Define the permutation function A
def A (n k : ℕ) : ℕ := factorial n / factorial (n - k)

-- Theorem statement
theorem permutation_expressions (n : ℕ) : 
  (A (n + 1) n ≠ factorial n) ∧ 
  ((1 / (n + 1 : ℚ)) * A (n + 1) (n + 1) = factorial n) ∧
  (A n n = factorial n) ∧
  (n * A (n - 1) (n - 1) = factorial n) :=
sorry

end permutation_expressions_l1653_165315


namespace calcium_iodide_weight_l1653_165386

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of moles of calcium iodide -/
def moles_CaI2 : ℝ := 5

/-- The molecular weight of calcium iodide (CaI2) in g/mol -/
def molecular_weight_CaI2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_I

/-- The total weight of calcium iodide in grams -/
def total_weight_CaI2 : ℝ := moles_CaI2 * molecular_weight_CaI2

theorem calcium_iodide_weight : total_weight_CaI2 = 1469.4 := by
  sorry

end calcium_iodide_weight_l1653_165386


namespace abs_eq_neg_implies_nonpositive_l1653_165307

theorem abs_eq_neg_implies_nonpositive (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end abs_eq_neg_implies_nonpositive_l1653_165307


namespace lily_typing_speed_l1653_165351

/-- Represents Lily's typing scenario -/
structure TypingScenario where
  totalTime : ℕ -- Total time including breaks
  totalWords : ℕ -- Total words typed
  breakInterval : ℕ -- Interval between breaks
  breakDuration : ℕ -- Duration of each break

/-- Calculates the words typed per minute -/
def wordsPerMinute (scenario : TypingScenario) : ℚ :=
  let effectiveTypingTime := scenario.totalTime - (scenario.totalTime / scenario.breakInterval) * scenario.breakDuration
  scenario.totalWords / effectiveTypingTime

/-- Theorem stating that Lily types 15 words per minute -/
theorem lily_typing_speed :
  let scenario : TypingScenario := {
    totalTime := 19
    totalWords := 255
    breakInterval := 10
    breakDuration := 2
  }
  wordsPerMinute scenario = 15 := by
  sorry


end lily_typing_speed_l1653_165351


namespace two_sided_icing_cubes_count_l1653_165373

/-- Represents a 3D coordinate --/
structure Coord where
  x : Nat
  y : Nat
  z : Nat

/-- Represents a cake with dimensions and icing information --/
structure Cake where
  dim : Nat
  hasIcingTop : Bool
  hasIcingBottom : Bool
  hasIcingSides : Bool

/-- Counts the number of unit cubes with icing on exactly two sides --/
def countTwoSidedIcingCubes (c : Cake) : Nat :=
  sorry

/-- The main theorem to prove --/
theorem two_sided_icing_cubes_count (c : Cake) : 
  c.dim = 4 ∧ c.hasIcingTop ∧ ¬c.hasIcingBottom ∧ c.hasIcingSides → 
  countTwoSidedIcingCubes c = 20 := by
  sorry

end two_sided_icing_cubes_count_l1653_165373


namespace squirrel_acorns_count_l1653_165301

/-- Represents the number of acorns hidden per hole by each animal -/
structure AcornsPerHole where
  chipmunk : ℕ
  squirrel : ℕ
  rabbit : ℕ

/-- Represents the number of holes dug by each animal -/
structure HolesCounts where
  chipmunk : ℕ
  squirrel : ℕ
  rabbit : ℕ

/-- The main theorem stating the number of acorns hidden by the squirrel -/
theorem squirrel_acorns_count 
  (aph : AcornsPerHole) 
  (hc : HolesCounts) 
  (h1 : aph.chipmunk = 4) 
  (h2 : aph.squirrel = 5) 
  (h3 : aph.rabbit = 2) 
  (h4 : aph.chipmunk * hc.chipmunk = aph.squirrel * hc.squirrel) 
  (h5 : hc.squirrel = hc.chipmunk - 5) 
  (h6 : aph.rabbit * hc.rabbit = aph.squirrel * hc.squirrel) 
  (h7 : hc.rabbit = hc.squirrel + 10) : 
  aph.squirrel * hc.squirrel = 100 := by
  sorry

end squirrel_acorns_count_l1653_165301


namespace valid_squares_count_l1653_165302

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  topLeft : Nat × Nat

/-- Checks if a square contains at least 6 black squares -/
def hasAtLeastSixBlackSquares (s : Square) : Bool :=
  sorry

/-- Counts the number of valid squares on the board -/
def countValidSquares (boardSize : Nat) : Nat :=
  sorry

theorem valid_squares_count :
  countValidSquares 10 = 140 :=
sorry

end valid_squares_count_l1653_165302


namespace simple_interest_problem_l1653_165368

/-- Proves that given a sum P put at simple interest for 4 years, 
    if increasing the interest rate by 2% results in $56 more interest, 
    then P = $700. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 2) * 4) / 100 - (P * R * 4) / 100 = 56 → P = 700 := by
  sorry

end simple_interest_problem_l1653_165368


namespace expression_simplification_l1653_165322

theorem expression_simplification (x y : ℝ) (h : x = -3) :
  x * (x - 4) * (x + 4) - (x + 3) * (x^2 - 6*x + 9) + 5*x^3*y^2 / (x^2*y^2) = -66 :=
by sorry

end expression_simplification_l1653_165322


namespace perpendicular_vectors_l1653_165303

def a : Fin 2 → ℝ := ![3, 4]
def b : Fin 2 → ℝ := ![2, -1]

theorem perpendicular_vectors (x : ℝ) : 
  (∀ i : Fin 2, (a + x • b) i * (-b i) = 0) → x = -2/5 := by
  sorry

end perpendicular_vectors_l1653_165303


namespace ratio_problem_l1653_165367

theorem ratio_problem (q r s t u : ℚ) 
  (h1 : q / r = 8)
  (h2 : s / r = 5)
  (h3 : s / t = 1 / 4)
  (h4 : u / t = 3)
  : u / q = 15 / 2 := by
  sorry

end ratio_problem_l1653_165367


namespace max_value_of_x_plus_inverse_l1653_165323

theorem max_value_of_x_plus_inverse (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ (max : ℝ), max = Real.sqrt 13 ∧ x + 1/x ≤ max :=
by sorry

end max_value_of_x_plus_inverse_l1653_165323


namespace total_charcoal_needed_l1653_165338

-- Define the ratios and water amounts for each batch
def batch1_ratio : ℚ := 2 / 30
def batch1_water : ℚ := 900

def batch2_ratio : ℚ := 3 / 50
def batch2_water : ℚ := 1150

def batch3_ratio : ℚ := 4 / 80
def batch3_water : ℚ := 1615

def batch4_ratio : ℚ := 2.3 / 25
def batch4_water : ℚ := 675

def batch5_ratio : ℚ := 5.5 / 115
def batch5_water : ℚ := 1930

-- Function to calculate charcoal needed for a batch
def charcoal_needed (ratio : ℚ) (water : ℚ) : ℚ :=
  ratio * water

-- Theorem stating the total charcoal needed is 363.28 grams
theorem total_charcoal_needed :
  (charcoal_needed batch1_ratio batch1_water +
   charcoal_needed batch2_ratio batch2_water +
   charcoal_needed batch3_ratio batch3_water +
   charcoal_needed batch4_ratio batch4_water +
   charcoal_needed batch5_ratio batch5_water) = 363.28 := by
  sorry

end total_charcoal_needed_l1653_165338


namespace speeding_ticket_problem_l1653_165345

theorem speeding_ticket_problem (total_motorists : ℝ) 
  (h1 : total_motorists > 0) 
  (h2 : total_motorists * 0.4 = total_motorists * 0.5 - (total_motorists * 0.5 - total_motorists * 0.4)) :
  (total_motorists * 0.5 - total_motorists * 0.4) / (total_motorists * 0.5) = 0.2 := by
  sorry

#check speeding_ticket_problem

end speeding_ticket_problem_l1653_165345


namespace c_neq_zero_necessary_not_sufficient_l1653_165317

/-- Represents a conic section of the form ax^2 + y^2 = c -/
structure ConicSection where
  a : ℝ
  c : ℝ

/-- Determines if a conic section is an ellipse or hyperbola -/
def is_ellipse_or_hyperbola (conic : ConicSection) : Prop :=
  -- We don't define this explicitly as it's not given in the problem conditions
  sorry

/-- Theorem stating that c ≠ 0 is necessary but not sufficient for
    ax^2 + y^2 = c to represent an ellipse or hyperbola -/
theorem c_neq_zero_necessary_not_sufficient :
  (∀ conic : ConicSection, is_ellipse_or_hyperbola conic → conic.c ≠ 0) ∧
  (∃ conic : ConicSection, conic.c ≠ 0 ∧ ¬is_ellipse_or_hyperbola conic) :=
by
  sorry

end c_neq_zero_necessary_not_sufficient_l1653_165317


namespace M_value_l1653_165387

def M : ℕ → ℕ
  | 0 => 0
  | 1 => 4
  | (n + 2) => (2*n + 2)^2 + (2*n + 4)^2 - M n

theorem M_value : M 75 = 22800 := by
  sorry

end M_value_l1653_165387


namespace max_value_abc_l1653_165330

theorem max_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b + 2 * b * c) / (a^2 + b^2 + c^2) ≤ Real.sqrt 5 / 2 := by
  sorry

end max_value_abc_l1653_165330


namespace existence_of_m_l1653_165336

def z : ℕ → ℚ
  | 0 => 3
  | n + 1 => (2 * (z n)^2 + 3 * (z n) + 6) / (z n + 8)

theorem existence_of_m :
  ∃ m : ℕ, m ∈ Finset.Icc 27 80 ∧
    z m ≤ 2 + 1 / 2^10 ∧
    ∀ k : ℕ, k > 0 ∧ k < 27 → z k > 2 + 1 / 2^10 :=
by sorry

end existence_of_m_l1653_165336
