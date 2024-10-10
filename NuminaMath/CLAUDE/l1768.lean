import Mathlib

namespace cars_meeting_time_l1768_176895

/-- Two cars traveling towards each other on a highway meet after a certain time -/
theorem cars_meeting_time (highway_length : ℝ) (speed1 speed2 : ℝ) (meeting_time : ℝ) :
  highway_length = 500 →
  speed1 = 40 →
  speed2 = 60 →
  meeting_time * (speed1 + speed2) = highway_length →
  meeting_time = 5 := by
  sorry

end cars_meeting_time_l1768_176895


namespace common_chord_equation_l1768_176861

theorem common_chord_equation (x y : ℝ) : 
  (x^2 + y^2 = 4) ∧ (x^2 + y^2 - 4*x + 4*y - 12 = 0) → 
  (x - y + 2 = 0) := by
sorry

end common_chord_equation_l1768_176861


namespace arithmetic_sequence_problem_l1768_176853

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a_n where a_3 + a_9 = 15 - a_6, prove that a_6 = 5 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_equation : a 3 + a 9 = 15 - a 6) : 
  a 6 = 5 := by
  sorry

end arithmetic_sequence_problem_l1768_176853


namespace natural_number_pairs_l1768_176821

theorem natural_number_pairs (x y a n m : ℕ) :
  x + y = a^n ∧ x^2 + y^2 = a^m →
  ∃ k : ℕ, x = 2^k ∧ y = 2^k :=
by sorry

end natural_number_pairs_l1768_176821


namespace remainder_meters_after_marathons_l1768_176813

/-- The length of a marathon in kilometers -/
def marathon_length : ℝ := 42.195

/-- The number of marathons run -/
def num_marathons : ℕ := 15

/-- The number of meters in a kilometer -/
def meters_per_km : ℕ := 1000

/-- The total distance in kilometers -/
def total_distance : ℝ := marathon_length * num_marathons

theorem remainder_meters_after_marathons :
  ∃ (k : ℕ) (m : ℕ), 
    total_distance = k + (m : ℝ) / meters_per_km ∧ 
    m < meters_per_km ∧ 
    m = 925 := by sorry

end remainder_meters_after_marathons_l1768_176813


namespace ducks_in_lake_l1768_176878

theorem ducks_in_lake (initial_ducks joining_ducks : ℕ) 
  (h1 : initial_ducks = 13)
  (h2 : joining_ducks = 20) : 
  initial_ducks + joining_ducks = 33 := by
sorry

end ducks_in_lake_l1768_176878


namespace real_world_length_l1768_176889

/-- Represents the scale factor of the model -/
def scale_factor : ℝ := 50

/-- Represents the length of the line segment in the model (in cm) -/
def model_length : ℝ := 7.5

/-- Theorem stating that the real-world length represented by the model line segment is 375 meters -/
theorem real_world_length : model_length * scale_factor = 375 := by
  sorry

end real_world_length_l1768_176889


namespace fixed_point_implies_sqrt_two_l1768_176842

noncomputable section

-- Define the logarithmic function
def log_func (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the power function
def power_func (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem fixed_point_implies_sqrt_two 
  (a : ℝ) 
  (h_a_pos : a > 0) 
  (h_a_neq_one : a ≠ 1) 
  (A : ℝ × ℝ) 
  (α : ℝ) 
  (h_log_point : log_func a (A.1 - 3) + 2 = A.2)
  (h_power_point : power_func α A.1 = A.2) :
  power_func α 2 = Real.sqrt 2 :=
sorry

end

end fixed_point_implies_sqrt_two_l1768_176842


namespace perfect_square_condition_l1768_176858

theorem perfect_square_condition (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 - 6*x + a^2 = y^2) → (a = 3 ∨ a = -3) := by
  sorry

end perfect_square_condition_l1768_176858


namespace complement_A_in_U_l1768_176843

def U : Set ℕ := {x | x ≥ 3}
def A : Set ℕ := {x | x^2 ≥ 10}

theorem complement_A_in_U : (U \ A) = {3} := by sorry

end complement_A_in_U_l1768_176843


namespace age_difference_approximation_l1768_176815

-- Define the age ratios and total age sum
def patrick_michael_monica_ratio : Rat := 3 / 5
def michael_monica_nola_ratio : Rat × Rat := (3 / 5, 5 / 7)
def monica_nola_olivia_ratio : Rat × Rat := (4 / 3, 3 / 2)
def total_age_sum : ℕ := 146

-- Define a function to calculate the age difference
def age_difference (patrick_michael_monica_ratio : Rat) 
                   (michael_monica_nola_ratio : Rat × Rat)
                   (monica_nola_olivia_ratio : Rat × Rat)
                   (total_age_sum : ℕ) : ℝ :=
  sorry

-- Theorem statement
theorem age_difference_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |age_difference patrick_michael_monica_ratio 
                  michael_monica_nola_ratio
                  monica_nola_olivia_ratio
                  total_age_sum - 6.412| < ε :=
sorry

end age_difference_approximation_l1768_176815


namespace average_running_distance_l1768_176869

def monday_distance : ℝ := 4.2
def tuesday_distance : ℝ := 3.8
def wednesday_distance : ℝ := 3.6
def thursday_distance : ℝ := 4.4
def number_of_days : ℕ := 4

theorem average_running_distance :
  (monday_distance + tuesday_distance + wednesday_distance + thursday_distance) / number_of_days = 4 := by
  sorry

end average_running_distance_l1768_176869


namespace min_rolls_for_repeat_sum_l1768_176822

/-- Represents an eight-sided die -/
def Die8 := Fin 8

/-- The sum of two dice rolls -/
def DiceSum := Fin 15

/-- The number of possible sums when rolling two eight-sided dice -/
def NumPossibleSums : ℕ := 15

/-- The minimum number of rolls to guarantee a repeated sum -/
def MinRollsForRepeat : ℕ := NumPossibleSums + 1

theorem min_rolls_for_repeat_sum : 
  ∀ (rolls : ℕ), rolls ≥ MinRollsForRepeat → 
  ∃ (sum : DiceSum), (∃ (i j : Fin rolls), i ≠ j ∧ 
    ∃ (d1 d2 d3 d4 : Die8), 
      sum = ⟨d1.val + d2.val - 1, by sorry⟩ ∧
      sum = ⟨d3.val + d4.val - 1, by sorry⟩) :=
by sorry

end min_rolls_for_repeat_sum_l1768_176822


namespace potato_cooking_time_l1768_176854

theorem potato_cooking_time 
  (total_potatoes : ℕ) 
  (cooked_potatoes : ℕ) 
  (remaining_time : ℕ) 
  (h1 : total_potatoes = 16) 
  (h2 : cooked_potatoes = 7) 
  (h3 : remaining_time = 45) :
  remaining_time / (total_potatoes - cooked_potatoes) = 5 := by
  sorry

end potato_cooking_time_l1768_176854


namespace crayon_theorem_l1768_176867

/-- The number of crayons the other friend has -/
def other_friend_crayons (lizzie_crayons : ℕ) : ℕ :=
  lizzie_crayons * 4 / 3

theorem crayon_theorem (lizzie_crayons : ℕ) 
  (h1 : lizzie_crayons = 27) : 
  other_friend_crayons lizzie_crayons = 18 :=
by
  sorry

#eval other_friend_crayons 27

end crayon_theorem_l1768_176867


namespace line_perp_to_plane_l1768_176870

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and between a line and a plane
variable (perp : Line → Line → Prop)
variable (perpToPlane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

-- Define the intersection operation for lines
variable (intersect : Line → Line → Set Point)

-- Theorem statement
theorem line_perp_to_plane 
  (a b c : Line) (α : Plane) (A : Point) :
  perp c a → 
  perp c b → 
  subset a α → 
  subset b α → 
  intersect a b = {A} → 
  perpToPlane c α :=
sorry

end line_perp_to_plane_l1768_176870


namespace simplify_expression_l1768_176831

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end simplify_expression_l1768_176831


namespace equation_solution_l1768_176828

theorem equation_solution (k : ℝ) : 
  (∀ x : ℝ, -x^2 - (k + 7)*x - 8 = -(x - 2)*(x - 4)) ↔ k = -13 := by
  sorry

end equation_solution_l1768_176828


namespace initial_mangoes_l1768_176802

/-- Given a bag of fruits with the following conditions:
    - Initially contains 7 apples, 8 oranges, and M mangoes
    - 2 apples are removed
    - 4 oranges are removed (twice the number of apples removed)
    - 2/3 of the mangoes are removed
    - 14 fruits remain in the bag
    Prove that the initial number of mangoes (M) is 15 -/
theorem initial_mangoes (M : ℕ) : 
  (7 - 2) + (8 - 4) + (M - (2 * M / 3)) = 14 → M = 15 := by
  sorry

end initial_mangoes_l1768_176802


namespace calculation_proof_l1768_176827

theorem calculation_proof : (1/2)⁻¹ - Real.sqrt 3 * Real.tan (30 * π / 180) + (π - 2023)^0 + |-2| = 4 := by
  sorry

end calculation_proof_l1768_176827


namespace fred_change_is_correct_l1768_176885

/-- The change Fred received after paying for movie tickets and borrowing a movie --/
def fred_change : ℝ :=
  let ticket_price : ℝ := 5.92
  let num_tickets : ℕ := 2
  let borrowed_movie_cost : ℝ := 6.79
  let payment : ℝ := 20
  let total_cost : ℝ := ticket_price * num_tickets + borrowed_movie_cost
  payment - total_cost

/-- Theorem stating that Fred's change is $1.37 --/
theorem fred_change_is_correct : fred_change = 1.37 := by
  sorry

end fred_change_is_correct_l1768_176885


namespace reciprocal_roots_identity_l1768_176880

theorem reciprocal_roots_identity (p q r s : ℝ) : 
  (∃ a : ℝ, a^2 + p*a + q = 0 ∧ (1/a)^2 + r*(1/a) + s = 0) →
  (p*s - r)*(q*r - p) = (q*s - 1)^2 := by
  sorry

end reciprocal_roots_identity_l1768_176880


namespace sphere_surface_area_ratio_l1768_176873

/-- Given a regular triangular prism with an inscribed sphere of radius r
    and a circumscribed sphere of radius R, prove that the ratio of their
    surface areas is 5:1 -/
theorem sphere_surface_area_ratio (r R : ℝ) :
  r > 0 →
  R = r * Real.sqrt 5 →
  (4 * Real.pi * R^2) / (4 * Real.pi * r^2) = 5 :=
by sorry

end sphere_surface_area_ratio_l1768_176873


namespace price_reduction_sales_increase_effect_l1768_176823

theorem price_reduction_sales_increase_effect 
  (original_price original_sales : ℝ) 
  (price_reduction_percent : ℝ) 
  (sales_increase_percent : ℝ) 
  (h1 : price_reduction_percent = 30)
  (h2 : sales_increase_percent = 80) :
  let new_price := original_price * (1 - price_reduction_percent / 100)
  let new_sales := original_sales * (1 + sales_increase_percent / 100)
  let original_revenue := original_price * original_sales
  let new_revenue := new_price * new_sales
  (new_revenue - original_revenue) / original_revenue = 0.26 := by
sorry

end price_reduction_sales_increase_effect_l1768_176823


namespace pie_rows_theorem_l1768_176882

def pecan_pies : ℕ := 16
def apple_pies : ℕ := 14
def pies_per_row : ℕ := 5

theorem pie_rows_theorem : 
  (pecan_pies + apple_pies) / pies_per_row = 6 := by
  sorry

end pie_rows_theorem_l1768_176882


namespace cos_75_degrees_l1768_176894

theorem cos_75_degrees :
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end cos_75_degrees_l1768_176894


namespace sqrt_inequalities_l1768_176898

theorem sqrt_inequalities (x : ℝ) :
  (∀ x, (Real.sqrt (x - 1) < 1) ↔ (1 ≤ x ∧ x < 2)) ∧
  (∀ x, (Real.sqrt (2*x - 3) ≤ Real.sqrt (x - 1)) ↔ ((3/2) ≤ x ∧ x ≤ 2)) := by
sorry

end sqrt_inequalities_l1768_176898


namespace median_mode_difference_l1768_176893

def data : List ℕ := [21, 23, 23, 24, 24, 33, 33, 33, 33, 42, 42, 47, 48, 51, 52, 53, 54, 62, 67, 68]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem median_mode_difference (h : data.length = 20) : 
  |median data - (mode data : ℚ)| = 0 := by sorry

end median_mode_difference_l1768_176893


namespace tree_height_after_two_years_l1768_176800

/-- The height of a tree that triples every year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

/-- Theorem: A tree that triples its height every year and reaches 243 feet after 5 years will be 9 feet tall after 2 years -/
theorem tree_height_after_two_years 
  (h : ∃ initial_height : ℝ, tree_height initial_height 5 = 243) :
  ∃ initial_height : ℝ, tree_height initial_height 2 = 9 := by
sorry

end tree_height_after_two_years_l1768_176800


namespace range_a_theorem_l1768_176809

open Set

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the range of a
def range_of_a : Set ℝ := Ioo (-2) (-1) ∪ Ici 1

-- Theorem statement
theorem range_a_theorem (h1 : ∀ a : ℝ, p a ∨ q a) (h2 : ¬ ∃ a : ℝ, p a ∧ q a) :
  ∀ a : ℝ, a ∈ range_of_a ↔ (p a ∨ q a) ∧ ¬(p a ∧ q a) :=
sorry

end range_a_theorem_l1768_176809


namespace odd_even_subsets_equal_l1768_176810

theorem odd_even_subsets_equal (n : ℕ) :
  let S := Fin (2 * n + 1)
  (Finset.filter (fun X : Finset S => X.card % 2 = 1) (Finset.powerset (Finset.univ))).card =
  (Finset.filter (fun X : Finset S => X.card % 2 = 0) (Finset.powerset (Finset.univ))).card :=
by sorry

end odd_even_subsets_equal_l1768_176810


namespace function_inequality_implies_a_bound_l1768_176816

theorem function_inequality_implies_a_bound 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ → ℝ) 
  (h : ∀ x₁ ∈ Set.Icc 0 1, ∃ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g a x₂)
  (hf : ∀ x, f x = x - 1 / (x + 1))
  (hg : ∀ a x, g a x = x^2 - 2*a*x + 4) :
  a ≥ 9/4 := by
  sorry

end function_inequality_implies_a_bound_l1768_176816


namespace equation_solution_l1768_176857

theorem equation_solution : ∃ x : ℝ, 3 * x - 6 = |(-20 + 5)| ∧ x = 7 := by
  sorry

end equation_solution_l1768_176857


namespace multiples_equality_l1768_176863

def c : ℕ := (Finset.filter (fun n => 12 ∣ n ∧ n < 60) (Finset.range 60)).card

def d : ℕ := (Finset.filter (fun n => 3 ∣ n ∧ 4 ∣ n ∧ n < 60) (Finset.range 60)).card

theorem multiples_equality : (c - d)^3 = 0 := by
  sorry

end multiples_equality_l1768_176863


namespace pattern_proof_l1768_176833

theorem pattern_proof (n : ℕ+) : n * (n + 2) + 1 = (n + 1)^2 := by
  sorry

end pattern_proof_l1768_176833


namespace inequality_proof_l1768_176899

theorem inequality_proof (a b : ℝ) (h1 : a + b > 0) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  a / b^2 + b / a^2 ≥ 1 / a + 1 / b := by
  sorry

end inequality_proof_l1768_176899


namespace tile_arrangement_exists_l1768_176807

/-- Represents a 2x1 tile with a diagonal -/
structure Tile :=
  (position : Fin 6 × Fin 6)  -- Top-left corner position in the 6x6 grid
  (orientation : Bool)        -- True for horizontal, False for vertical
  (diagonal : Bool)           -- True for one diagonal direction, False for the other

/-- Represents the 6x6 grid -/
def Grid := Fin 6 → Fin 6 → Option Tile

/-- Check if a tile placement is valid -/
def valid_placement (grid : Grid) (tile : Tile) : Prop :=
  -- Add conditions to check if the tile fits within the grid
  -- and doesn't overlap with other tiles
  sorry

/-- Check if diagonal endpoints don't coincide -/
def no_coinciding_diagonals (grid : Grid) : Prop :=
  -- Add conditions to check that no diagonal endpoints coincide
  sorry

theorem tile_arrangement_exists : ∃ (grid : Grid),
  (∃ (tiles : Finset Tile), tiles.card = 18 ∧ 
    (∀ t ∈ tiles, valid_placement grid t)) ∧
  no_coinciding_diagonals grid :=
sorry

end tile_arrangement_exists_l1768_176807


namespace inequality_solution_set_l1768_176859

theorem inequality_solution_set (x : ℝ) :
  (1 / (x^2 + 2) > 4 / x + 21 / 10) ↔ (-2 < x ∧ x < 0) :=
by sorry

end inequality_solution_set_l1768_176859


namespace inverse_sum_equals_root_difference_l1768_176819

-- Define the function g
def g (x : ℝ) : ℝ := x^3 * |x|

-- State the theorem
theorem inverse_sum_equals_root_difference :
  (∃ y₁ : ℝ, g y₁ = 8) ∧ (∃ y₂ : ℝ, g y₂ = -125) →
  (∃ y₁ y₂ : ℝ, g y₁ = 8 ∧ g y₂ = -125 ∧ y₁ + y₂ = 2^(1/2) - 5^(3/4)) :=
by sorry

end inverse_sum_equals_root_difference_l1768_176819


namespace three_primes_sum_to_86_l1768_176849

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem three_primes_sum_to_86 :
  ∃ (a b c : ℕ), isPrime a ∧ isPrime b ∧ isPrime c ∧ a + b + c = 86 ∧
  (∀ (x y z : ℕ), isPrime x ∧ isPrime y ∧ isPrime z ∧ x + y + z = 86 →
    (x = 2 ∧ y = 5 ∧ z = 79) ∨
    (x = 2 ∧ y = 11 ∧ z = 73) ∨
    (x = 2 ∧ y = 13 ∧ z = 71) ∨
    (x = 2 ∧ y = 17 ∧ z = 67) ∨
    (x = 2 ∧ y = 23 ∧ z = 61) ∨
    (x = 2 ∧ y = 31 ∧ z = 53) ∨
    (x = 2 ∧ y = 37 ∧ z = 47) ∨
    (x = 2 ∧ y = 41 ∧ z = 43) ∨
    (x = 5 ∧ y = 2 ∧ z = 79) ∨
    (x = 11 ∧ y = 2 ∧ z = 73) ∨
    (x = 13 ∧ y = 2 ∧ z = 71) ∨
    (x = 17 ∧ y = 2 ∧ z = 67) ∨
    (x = 23 ∧ y = 2 ∧ z = 61) ∨
    (x = 31 ∧ y = 2 ∧ z = 53) ∨
    (x = 37 ∧ y = 2 ∧ z = 47) ∨
    (x = 41 ∧ y = 2 ∧ z = 43) ∨
    (x = 79 ∧ y = 2 ∧ z = 5) ∨
    (x = 73 ∧ y = 2 ∧ z = 11) ∨
    (x = 71 ∧ y = 2 ∧ z = 13) ∨
    (x = 67 ∧ y = 2 ∧ z = 17) ∨
    (x = 61 ∧ y = 2 ∧ z = 23) ∨
    (x = 53 ∧ y = 2 ∧ z = 31) ∨
    (x = 47 ∧ y = 2 ∧ z = 37) ∨
    (x = 43 ∧ y = 2 ∧ z = 41)) :=
by sorry


end three_primes_sum_to_86_l1768_176849


namespace nested_expression_sum_l1768_176845

def nested_expression : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * (1 + nested_expression n)

theorem nested_expression_sum : nested_expression 8 = 1022 := by
  sorry

end nested_expression_sum_l1768_176845


namespace range_of_m_l1768_176865

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, (m + 1) * (x^2 + 1) ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 > 0

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, (¬(p m ∧ q m)) → (m ≤ -2 ∨ m > -1) :=
sorry

end range_of_m_l1768_176865


namespace marbles_given_to_brother_l1768_176852

def initial_marbles : ℕ := 12
def current_marbles : ℕ := 7

theorem marbles_given_to_brother :
  initial_marbles - current_marbles = 5 := by
  sorry

end marbles_given_to_brother_l1768_176852


namespace equation_solution_l1768_176841

theorem equation_solution : 
  {x : ℝ | x + 45 / (x - 4) = -10} = {-1, -5} := by sorry

end equation_solution_l1768_176841


namespace log_problem_l1768_176876

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_problem (x : ℝ) 
  (h1 : x < 1) 
  (h2 : (log10 x)^3 - 3 * log10 x = 522) : 
  (log10 x)^4 - log10 (x^4) = 6597 := by
  sorry

end log_problem_l1768_176876


namespace quadratic_roots_relation_l1768_176836

theorem quadratic_roots_relation (m n p q : ℤ) (r₁ r₂ : ℝ) : 
  (r₁^2 - m*r₁ + n = 0 ∧ r₂^2 - m*r₂ + n = 0) →
  (r₁^4 - p*r₁^2 + q = 0 ∧ r₂^4 - p*r₂^2 + q = 0) →
  p = m^2 - 2*n :=
by sorry

end quadratic_roots_relation_l1768_176836


namespace friday_price_calculation_l1768_176801

theorem friday_price_calculation (tuesday_price : ℝ) : 
  tuesday_price = 50 →
  let wednesday_price := tuesday_price * (1 + 0.2)
  let friday_price := wednesday_price * (1 - 0.15)
  friday_price = 51 := by
sorry

end friday_price_calculation_l1768_176801


namespace smallest_matching_end_digits_correct_l1768_176875

/-- The smallest positive integer M such that M and M^2 end in the same sequence of three non-zero digits in base 10 -/
def smallest_matching_end_digits : ℕ := 376

/-- Check if a number ends with the given three digits -/
def ends_with (n : ℕ) (xyz : ℕ) : Prop :=
  n % 1000 = xyz

/-- The property that M and M^2 end with the same three non-zero digits -/
def has_matching_end_digits (M : ℕ) : Prop :=
  ∃ (xyz : ℕ), xyz ≥ 100 ∧ xyz < 1000 ∧ ends_with M xyz ∧ ends_with (M^2) xyz

theorem smallest_matching_end_digits_correct :
  has_matching_end_digits smallest_matching_end_digits ∧
  ∀ M : ℕ, M < smallest_matching_end_digits → ¬(has_matching_end_digits M) :=
sorry

end smallest_matching_end_digits_correct_l1768_176875


namespace distance_from_origin_l1768_176814

theorem distance_from_origin (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by sorry

end distance_from_origin_l1768_176814


namespace quadratic_form_ratio_representation_l1768_176883

theorem quadratic_form_ratio_representation (x y u v : ℤ) :
  (∃ k : ℤ, (x^2 + 3*y^2) = k * (u^2 + 3*v^2)) →
  ∃ a b : ℤ, (x^2 + 3*y^2) / (u^2 + 3*v^2) = a^2 + 3*b^2 :=
by sorry

end quadratic_form_ratio_representation_l1768_176883


namespace carl_driving_hours_l1768_176812

/-- The number of hours Carl drives per day before promotion -/
def hours_per_day : ℝ :=
  2

/-- The number of additional hours Carl drives per week after promotion -/
def additional_hours_per_week : ℝ :=
  6

/-- The number of hours Carl drives in two weeks after promotion -/
def hours_in_two_weeks_after : ℝ :=
  40

/-- The number of days in two weeks -/
def days_in_two_weeks : ℝ :=
  14

theorem carl_driving_hours :
  hours_per_day * days_in_two_weeks + additional_hours_per_week * 2 = hours_in_two_weeks_after :=
by sorry

end carl_driving_hours_l1768_176812


namespace cookie_distribution_l1768_176872

/-- The number of ways to distribute n identical items into k distinct groups -/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of friends -/
def numFriends : ℕ := 4

/-- The total number of cookies -/
def totalCookies : ℕ := 10

/-- The minimum number of cookies each friend must have -/
def minCookies : ℕ := 2

/-- The number of ways to distribute the cookies -/
def numWays : ℕ := starsAndBars (totalCookies - minCookies * numFriends) numFriends

theorem cookie_distribution :
  numWays = 10 := by sorry

end cookie_distribution_l1768_176872


namespace total_spent_proof_l1768_176890

/-- The price of each flower in dollars -/
def flower_price : ℕ := 3

/-- The number of roses Zoe bought -/
def roses_bought : ℕ := 8

/-- The number of daisies Zoe bought -/
def daisies_bought : ℕ := 2

/-- Theorem: Given the price of each flower and the number of roses and daisies bought,
    prove that the total amount spent is 30 dollars -/
theorem total_spent_proof :
  flower_price * (roses_bought + daisies_bought) = 30 :=
by sorry

end total_spent_proof_l1768_176890


namespace divisibility_condition_l1768_176846

theorem divisibility_condition (m n : ℕ+) : (2*m^2 + n^2) ∣ (3*m*n + 3*m) ↔ (m = 1 ∧ n = 1) ∨ (m = 4 ∧ n = 2) ∨ (m = 4 ∧ n = 10) := by
  sorry

end divisibility_condition_l1768_176846


namespace sum_of_zeros_is_zero_l1768_176834

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define a function with exactly four zeros
def HasFourZeros (f : ℝ → ℝ) : Prop := ∃ x₁ x₂ x₃ x₄, 
  (f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0) ∧
  (∀ x, f x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

theorem sum_of_zeros_is_zero (f : ℝ → ℝ) 
  (heven : EvenFunction f) (hzeros : HasFourZeros f) : 
  ∃ x₁ x₂ x₃ x₄, f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ x₁ + x₂ + x₃ + x₄ = 0 :=
sorry

end sum_of_zeros_is_zero_l1768_176834


namespace gold_coins_per_hour_l1768_176881

def scuba_diving_hours : ℕ := 8
def treasure_chest_coins : ℕ := 100
def smaller_bags_count : ℕ := 2

def smaller_bag_coins : ℕ := treasure_chest_coins / 2

def total_coins : ℕ := treasure_chest_coins + smaller_bags_count * smaller_bag_coins

theorem gold_coins_per_hour :
  total_coins / scuba_diving_hours = 25 := by sorry

end gold_coins_per_hour_l1768_176881


namespace scores_mode_is_37_l1768_176848

def scores : List Nat := [35, 37, 39, 37, 38, 38, 37]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem scores_mode_is_37 : mode scores = 37 := by
  sorry

end scores_mode_is_37_l1768_176848


namespace distinct_primes_in_product_l1768_176868

theorem distinct_primes_in_product : ∃ (S : Finset Nat), 
  (∀ p ∈ S, Nat.Prime p) ∧ 
  (∀ p : Nat, Nat.Prime p → (p ∣ (85 * 87 * 90 * 92) ↔ p ∈ S)) ∧ 
  Finset.card S = 6 := by
  sorry

end distinct_primes_in_product_l1768_176868


namespace orange_bin_theorem_l1768_176891

/-- Calculates the final number of oranges in a bin after changes. -/
def final_oranges (initial : ℕ) (thrown_away : ℕ) (added : ℕ) : ℕ :=
  initial - thrown_away + added

/-- Proves that the final number of oranges is correct given the initial conditions. -/
theorem orange_bin_theorem (initial : ℕ) (thrown_away : ℕ) (added : ℕ) :
  final_oranges initial thrown_away added = initial - thrown_away + added :=
by sorry

end orange_bin_theorem_l1768_176891


namespace max_base8_digit_sum_l1768_176806

-- Define a function to convert a natural number to its base-8 representation
def toBase8 (n : ℕ) : List ℕ :=
  sorry

-- Define a function to sum the digits of a number in its base-8 representation
def sumBase8Digits (n : ℕ) : ℕ :=
  (toBase8 n).sum

-- Theorem statement
theorem max_base8_digit_sum :
  ∃ (m : ℕ), m < 5000 ∧ 
  (∀ (n : ℕ), n < 5000 → sumBase8Digits n ≤ sumBase8Digits m) ∧
  sumBase8Digits m = 28 :=
sorry

end max_base8_digit_sum_l1768_176806


namespace f_value_at_107_5_l1768_176877

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_value_at_107_5 (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_period : ∀ x, f (x + 3) = -1 / f x)
  (h_neg : ∀ x, x < 0 → f x = 4 * x) :
  f 107.5 = 1/10 := by
sorry

end f_value_at_107_5_l1768_176877


namespace total_cars_produced_l1768_176818

theorem total_cars_produced (north_america : ℕ) (europe : ℕ) 
  (h1 : north_america = 3884) (h2 : europe = 2871) : 
  north_america + europe = 6755 := by
  sorry

end total_cars_produced_l1768_176818


namespace evaluate_expression_l1768_176805

theorem evaluate_expression (a x : ℝ) (h : x = a + 10) : (x - a + 3) * (x - a - 2) = 104 := by
  sorry

end evaluate_expression_l1768_176805


namespace alcohol_percentage_in_mixture_l1768_176874

/-- Represents a solution with a specific ratio of alcohol to water -/
structure Solution :=
  (alcohol : ℚ)
  (water : ℚ)

/-- Calculates the percentage of alcohol in a solution -/
def alcoholPercentage (s : Solution) : ℚ :=
  s.alcohol / (s.alcohol + s.water)

/-- Represents the mixing of two solutions in a specific ratio -/
structure Mixture :=
  (s1 : Solution)
  (s2 : Solution)
  (ratio1 : ℚ)
  (ratio2 : ℚ)

/-- Calculates the percentage of alcohol in a mixture -/
def mixtureAlcoholPercentage (m : Mixture) : ℚ :=
  (alcoholPercentage m.s1 * m.ratio1 + alcoholPercentage m.s2 * m.ratio2) / (m.ratio1 + m.ratio2)

theorem alcohol_percentage_in_mixture :
  let solutionA : Solution := ⟨21, 4⟩
  let solutionB : Solution := ⟨2, 3⟩
  let mixture : Mixture := ⟨solutionA, solutionB, 5, 6⟩
  mixtureAlcoholPercentage mixture = 3/5 := by sorry

end alcohol_percentage_in_mixture_l1768_176874


namespace mans_speed_against_current_l1768_176856

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
theorem mans_speed_against_current
  (speed_with_current : ℝ)
  (current_speed : ℝ)
  (h1 : speed_with_current = 12)
  (h2 : current_speed = 2) :
  speed_with_current - 2 * current_speed = 8 := by
  sorry

end mans_speed_against_current_l1768_176856


namespace concentric_circles_ratio_l1768_176811

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents two concentric circles -/
structure ConcentricCircles where
  inner : Circle
  outer : Circle
  h : inner.radius < outer.radius

/-- Represents three circles tangent to two concentric circles and to each other -/
structure TangentCircles (cc : ConcentricCircles) where
  c1 : Circle
  c2 : Circle
  c3 : Circle
  tangent_to_concentric : 
    c1.radius = cc.outer.radius - cc.inner.radius ∧
    c2.radius = cc.outer.radius - cc.inner.radius ∧
    c3.radius = cc.outer.radius - cc.inner.radius
  tangent_to_each_other : True  -- This is a simplification, as we can't easily express tangency

/-- The main theorem: If three circles are tangent to two concentric circles and to each other,
    then the ratio of the radii of the concentric circles is 3 -/
theorem concentric_circles_ratio 
  (cc : ConcentricCircles) 
  (tc : TangentCircles cc) : 
  cc.outer.radius / cc.inner.radius = 3 := by
  sorry

end concentric_circles_ratio_l1768_176811


namespace remainder_17_pow_33_mod_7_l1768_176803

theorem remainder_17_pow_33_mod_7 : 17^33 % 7 = 6 := by sorry

end remainder_17_pow_33_mod_7_l1768_176803


namespace progression_product_exceeds_100000_l1768_176850

theorem progression_product_exceeds_100000 (n : ℕ) : 
  (n ≥ 11 ∧ ∀ k < 11, k > 0 → 10^((k * (k + 1)) / 22) ≤ 10^5) ↔ 
  (∀ k ≤ n, 10^((k * (k + 1)) / 22) > 10^5 ↔ k ≥ 11) := by
  sorry

end progression_product_exceeds_100000_l1768_176850


namespace special_polynomial_at_five_l1768_176855

/-- A cubic polynomial satisfying specific conditions -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, p x = a*x^3 + b*x^2 + c*x + d) ∧
  (∀ n ∈ ({1, 2, 3, 4, 6} : Set ℝ), p n = 1 / n^2) ∧
  p 0 = -1/25

/-- The main theorem -/
theorem special_polynomial_at_five 
  (p : ℝ → ℝ) 
  (h : special_polynomial p) : 
  p 5 = 20668/216000 := by
sorry

end special_polynomial_at_five_l1768_176855


namespace sum_and_count_equals_431_l1768_176892

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_equals_431 : 
  sum_of_integers 10 30 + count_even_integers 10 30 = 431 := by
  sorry

end sum_and_count_equals_431_l1768_176892


namespace cookie_difference_l1768_176840

/-- The number of chocolate chip cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 19

/-- The number of raisin cookies Helen baked this morning -/
def raisin_cookies : ℕ := 231

/-- The number of chocolate chip cookies Helen baked this morning -/
def cookies_today : ℕ := 237

/-- The total number of chocolate chip cookies Helen baked -/
def total_choc_cookies : ℕ := cookies_yesterday + cookies_today

/-- The difference between chocolate chip cookies and raisin cookies -/
theorem cookie_difference : total_choc_cookies - raisin_cookies = 25 := by
  sorry

end cookie_difference_l1768_176840


namespace coefficient_a4_value_l1768_176837

theorem coefficient_a4_value (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + a₅*(x+1)^5) →
  a₄ = -5 := by
sorry

end coefficient_a4_value_l1768_176837


namespace tabletop_coverage_fraction_l1768_176896

-- Define the radius of the circular mat
def mat_radius : ℝ := 10

-- Define the side length of the square tabletop
def table_side : ℝ := 24

-- Theorem to prove the fraction of the tabletop covered by the mat
theorem tabletop_coverage_fraction :
  (π * mat_radius^2) / (table_side^2) = 100 * π / 576 := by
  sorry

end tabletop_coverage_fraction_l1768_176896


namespace star_equation_solution_l1768_176808

/-- Definition of the star operation -/
def star (a b : ℝ) : ℝ := a * b + 3 * b - 2 * a

/-- Theorem stating that if 6 ★ x = 45, then x = 19/3 -/
theorem star_equation_solution :
  (star 6 x = 45) → x = 19/3 := by
  sorry

end star_equation_solution_l1768_176808


namespace halfway_point_between_fractions_l1768_176888

theorem halfway_point_between_fractions :
  let a := (1 : ℚ) / 7
  let b := (1 : ℚ) / 9
  let midpoint := (a + b) / 2
  midpoint = 8 / 63 := by sorry

end halfway_point_between_fractions_l1768_176888


namespace school_garden_flowers_l1768_176860

theorem school_garden_flowers :
  let total_flowers : ℕ := 96
  let green_flowers : ℕ := 9
  let red_flowers : ℕ := 3 * green_flowers
  let blue_flowers : ℕ := total_flowers / 2
  let yellow_flowers : ℕ := total_flowers - (green_flowers + red_flowers + blue_flowers)
  yellow_flowers = 12 := by
sorry

end school_garden_flowers_l1768_176860


namespace inequality_proof_l1768_176862

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + 4*c^2 = 3) :
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end inequality_proof_l1768_176862


namespace sin_2x_minus_pi_6_equals_cos_2x_minus_2pi_3_l1768_176817

theorem sin_2x_minus_pi_6_equals_cos_2x_minus_2pi_3 (x : ℝ) : 
  Real.sin (2 * x - π / 6) = Real.cos (2 * (x - π / 3)) := by
  sorry

end sin_2x_minus_pi_6_equals_cos_2x_minus_2pi_3_l1768_176817


namespace cost_of_3000_pencils_l1768_176835

/-- The cost of purchasing a given number of pencils with a bulk discount. -/
def cost_of_pencils (box_size : ℕ) (box_price : ℚ) (discount_threshold : ℕ) (discount_rate : ℚ) (num_pencils : ℕ) : ℚ :=
  let unit_price := box_price / box_size
  let total_price := unit_price * num_pencils
  if num_pencils > discount_threshold then
    total_price * (1 - discount_rate)
  else
    total_price

/-- Theorem stating that the cost of 3000 pencils is $675 given the problem conditions. -/
theorem cost_of_3000_pencils :
  cost_of_pencils 200 50 1000 (1/10) 3000 = 675 := by
  sorry

end cost_of_3000_pencils_l1768_176835


namespace problem_solution_l1768_176847

theorem problem_solution (a b c : ℝ) : 
  |a - 1| + Real.sqrt (b + 2) + (c - 3)^2 = 0 → (a + b)^c = -1 := by
  sorry

end problem_solution_l1768_176847


namespace marks_team_free_throws_marks_team_free_throws_correct_l1768_176830

theorem marks_team_free_throws (marks_two_pointers marks_three_pointers : ℕ) 
  (total_points : ℕ) (h1 : marks_two_pointers = 25) (h2 : marks_three_pointers = 8) 
  (h3 : total_points = 201) : ℕ :=
  let marks_points := 2 * marks_two_pointers + 3 * marks_three_pointers
  let opponents_two_pointers := 2 * marks_two_pointers
  let opponents_three_pointers := marks_three_pointers / 2
  let free_throws := total_points - (marks_points + 2 * opponents_two_pointers + 3 * opponents_three_pointers)
  10

theorem marks_team_free_throws_correct : marks_team_free_throws 25 8 201 rfl rfl rfl = 10 := by
  sorry

end marks_team_free_throws_marks_team_free_throws_correct_l1768_176830


namespace not_all_data_has_regression_equation_l1768_176844

-- Define the basic concepts
def DataSet : Type := Set (ℝ × ℝ)
def RegressionEquation : Type := ℝ → ℝ

-- Define the properties mentioned in the problem
def hasCorrelation (d : DataSet) : Prop := sorry
def hasCausalRelationship (d : DataSet) : Prop := sorry
def canBeRepresentedByScatterPlot (d : DataSet) : Prop := sorry
def hasLinearCorrelation (d : DataSet) : Prop := sorry
def hasRegressionEquation (d : DataSet) : Prop := sorry

-- Define the statements from the problem
axiom correlation_not_causation : 
  ∀ d : DataSet, hasCorrelation d → ¬ (hasCausalRelationship d)

axiom scatter_plot_reflects_correlation : 
  ∀ d : DataSet, hasCorrelation d → canBeRepresentedByScatterPlot d

axiom regression_line_best_represents : 
  ∀ d : DataSet, hasLinearCorrelation d → hasRegressionEquation d

-- The theorem to be proved
theorem not_all_data_has_regression_equation :
  ¬ (∀ d : DataSet, hasRegressionEquation d) := by
  sorry

end not_all_data_has_regression_equation_l1768_176844


namespace hyperbola_center_l1768_176839

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) (c : ℝ × ℝ) : 
  f1 = (6, -2) → f2 = (10, 6) → c = ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2) → c = (8, 2) := by
  sorry

#check hyperbola_center

end hyperbola_center_l1768_176839


namespace wedding_guests_fraction_l1768_176820

theorem wedding_guests_fraction (total_guests : ℚ) : 
  let children_fraction : ℚ := 1/8
  let adult_fraction : ℚ := 1 - children_fraction
  let men_fraction_of_adults : ℚ := 3/7
  let women_fraction_of_adults : ℚ := 1 - men_fraction_of_adults
  let adult_women_fraction : ℚ := adult_fraction * women_fraction_of_adults
  adult_women_fraction = 1/2 := by
sorry

end wedding_guests_fraction_l1768_176820


namespace dan_limes_remaining_l1768_176804

theorem dan_limes_remaining (initial_limes given_limes : ℕ) : 
  initial_limes = 9 → given_limes = 4 → initial_limes - given_limes = 5 := by
  sorry

end dan_limes_remaining_l1768_176804


namespace prime_form_and_infinitude_l1768_176879

theorem prime_form_and_infinitude (p : ℕ) :
  (Prime p ∧ p ≥ 3) →
  (∃! k : ℕ, k ≥ 1 ∧ (p = 4*k - 1 ∨ p = 4*k + 1)) ∧
  Set.Infinite {p : ℕ | Prime p ∧ ∃ k : ℕ, p = 4*k - 1} :=
by sorry

end prime_form_and_infinitude_l1768_176879


namespace odd_function_properties_l1768_176871

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + (a - 1)*x^2 + a*x + b

-- Define the property of f being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Main theorem
theorem odd_function_properties (a b : ℝ) 
  (h : is_odd_function (f a b)) : 
  (a + b = 1) ∧ 
  (∃ m c : ℝ, m = 4 ∧ c = -2 ∧ 
    ∀ x y : ℝ, y = f a b x → (y - f a b 1 = m * (x - 1) ↔ m*x - y + c = 0)) :=
by sorry

end odd_function_properties_l1768_176871


namespace endpoint_from_midpoint_and_other_endpoint_l1768_176864

theorem endpoint_from_midpoint_and_other_endpoint :
  ∀ (x y : ℝ),
  (3 : ℝ) = (7 + x) / 2 →
  (2 : ℝ) = (-4 + y) / 2 →
  (x, y) = (-1, 8) := by
sorry

end endpoint_from_midpoint_and_other_endpoint_l1768_176864


namespace arithmetic_geometric_sequence_property_l1768_176851

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence :=
  (a : ℕ → ℝ)
  (d : ℝ)
  (h : d ≠ 0)
  (h' : ∀ n, a (n + 1) = a n + d)

/-- A geometric sequence -/
structure GeometricSequence :=
  (b : ℕ → ℝ)
  (r : ℝ)
  (h : ∀ n, b (n + 1) = r * b n)

theorem arithmetic_geometric_sequence_property
  (as : ArithmeticSequence)
  (gs : GeometricSequence)
  (h1 : 2 * as.a 3 - (as.a 7)^2 + 2 * as.a 11 = 0)
  (h2 : gs.b 7 = as.a 7) :
  gs.b 6 * gs.b 8 = 16 :=
sorry

end arithmetic_geometric_sequence_property_l1768_176851


namespace biancas_books_l1768_176886

/-- The number of coloring books Bianca has after giving some away and buying more -/
def final_book_count (initial : ℕ) (given_away : ℕ) (bought : ℕ) : ℕ :=
  initial - given_away + bought

/-- Theorem stating that Bianca's final book count is 59 -/
theorem biancas_books : final_book_count 45 6 20 = 59 := by
  sorry

end biancas_books_l1768_176886


namespace perpendicular_segments_sum_maximum_l1768_176824

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define a point within the circle's disk
structure PointInDisk (c : Circle) where
  point : ℝ × ℝ
  in_disk : Real.sqrt ((point.1 - c.center.1)^2 + (point.2 - c.center.2)^2) ≤ c.radius

-- Define two perpendicular line segments from a point to the circle's boundary
structure PerpendicularSegments (c : Circle) (p : PointInDisk c) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ
  on_circle1 : Real.sqrt ((endpoint1.1 - c.center.1)^2 + (endpoint1.2 - c.center.2)^2) = c.radius
  on_circle2 : Real.sqrt ((endpoint2.1 - c.center.1)^2 + (endpoint2.2 - c.center.2)^2) = c.radius
  perpendicular : (endpoint1.1 - p.point.1) * (endpoint2.1 - p.point.1) + 
                  (endpoint1.2 - p.point.2) * (endpoint2.2 - p.point.2) = 0

-- Theorem statement
theorem perpendicular_segments_sum_maximum (c : Circle) (p : PointInDisk c) 
  (segments : PerpendicularSegments c p) :
  ∃ (max_segments : PerpendicularSegments c p),
    (Real.sqrt ((max_segments.endpoint1.1 - p.point.1)^2 + (max_segments.endpoint1.2 - p.point.2)^2) +
     Real.sqrt ((max_segments.endpoint2.1 - p.point.1)^2 + (max_segments.endpoint2.2 - p.point.2)^2)) ≥
    (Real.sqrt ((segments.endpoint1.1 - p.point.1)^2 + (segments.endpoint1.2 - p.point.2)^2) +
     Real.sqrt ((segments.endpoint2.1 - p.point.1)^2 + (segments.endpoint2.2 - p.point.2)^2)) ∧
    (Real.sqrt ((max_segments.endpoint1.1 - p.point.1)^2 + (max_segments.endpoint1.2 - p.point.2)^2) =
     Real.sqrt ((max_segments.endpoint2.1 - p.point.1)^2 + (max_segments.endpoint2.2 - p.point.2)^2)) ∧
    (Real.sqrt ((max_segments.endpoint1.1 - max_segments.endpoint2.1)^2 + 
                (max_segments.endpoint1.2 - max_segments.endpoint2.2)^2) = 2 * c.radius) :=
by
  sorry

end perpendicular_segments_sum_maximum_l1768_176824


namespace imaginary_part_of_fraction_l1768_176887

theorem imaginary_part_of_fraction (i : ℂ) : i * i = -1 → Complex.im ((1 - i) / (1 + i)) = -1 := by
  sorry

end imaginary_part_of_fraction_l1768_176887


namespace surface_is_one_sheet_hyperboloid_l1768_176825

/-- The equation of the surface -/
def surface_equation (x y z : ℝ) : Prop :=
  x^2 - 2*x - 3*y^2 + 12*y + 2*z^2 + 12*z - 11 = 0

/-- The standard form of a one-sheet hyperboloid -/
def one_sheet_hyperboloid (a b c : ℝ) (x y z : ℝ) : Prop :=
  (x - a)^2 / 18 - (y - b)^2 / 6 + (z - c)^2 / 9 = 1

/-- Theorem stating that the surface equation represents a one-sheet hyperboloid -/
theorem surface_is_one_sheet_hyperboloid :
  ∀ x y z : ℝ, surface_equation x y z ↔ one_sheet_hyperboloid 1 2 (-3) x y z :=
by sorry

end surface_is_one_sheet_hyperboloid_l1768_176825


namespace min_value_xy_l1768_176884

theorem min_value_xy (x y : ℝ) (h : x > 0 ∧ y > 0) (eq : 1/x + 2/y = Real.sqrt (x*y)) : 
  x * y ≥ 2 * Real.sqrt 2 :=
sorry

end min_value_xy_l1768_176884


namespace rectangle_area_l1768_176826

/-- The area of a rectangle given its perimeter and width -/
theorem rectangle_area (perimeter width : ℝ) (h1 : perimeter = 56) (h2 : width = 16) :
  let length := (perimeter - 2 * width) / 2
  width * length = 192 := by
sorry

end rectangle_area_l1768_176826


namespace muffins_for_sale_is_108_l1768_176829

/-- Calculate the number of muffins for sale given the following conditions:
  * 3 boys each make 12 muffins
  * 2 girls each make 20 muffins
  * 1 girl makes 15 muffins
  * 2 boys each make 18 muffins
  * 15% of all muffins will not make it to the sale
-/
def muffinsForSale : ℕ :=
  let boys_group1 := 3 * 12
  let boys_group2 := 2 * 18
  let girls_group1 := 2 * 20
  let girls_group2 := 1 * 15
  let total_muffins := boys_group1 + boys_group2 + girls_group1 + girls_group2
  let muffins_not_for_sale := (total_muffins : ℚ) * (15 : ℚ) / (100 : ℚ)
  ⌊(total_muffins : ℚ) - muffins_not_for_sale⌋.toNat

/-- Theorem stating that the number of muffins for sale is 108 -/
theorem muffins_for_sale_is_108 : muffinsForSale = 108 := by
  sorry

end muffins_for_sale_is_108_l1768_176829


namespace consecutive_numbers_lcm_660_l1768_176897

theorem consecutive_numbers_lcm_660 (x : ℕ) :
  (Nat.lcm x (Nat.lcm (x + 1) (x + 2)) = 660) →
  x = 10 ∧ (x + 1) = 11 ∧ (x + 2) = 12 := by
sorry

end consecutive_numbers_lcm_660_l1768_176897


namespace handshake_count_l1768_176838

theorem handshake_count (n : ℕ) (h : n = 8) : 
  (2 * n) * (2 * n - 2) / 2 = 112 := by sorry

end handshake_count_l1768_176838


namespace divisor_sum_equality_implies_prime_power_l1768_176866

/-- σ(N) is the sum of the positive integer divisors of N -/
def sigma (N : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem divisor_sum_equality_implies_prime_power (m n : ℕ) :
  m ≥ n → n ≥ 2 →
  (sigma m - 1) / (m - 1) = (sigma n - 1) / (n - 1) →
  (sigma m - 1) / (m - 1) = (sigma (m * n) - 1) / (m * n - 1) →
  ∃ (p : ℕ) (e f : ℕ), Prime p ∧ e ≥ f ∧ f ≥ 1 ∧ m = p^e ∧ n = p^f :=
sorry

end divisor_sum_equality_implies_prime_power_l1768_176866


namespace perfect_square_trinomial_k_l1768_176832

/-- A trinomial ax^2 + bx + c is a perfect square if there exist p and q such that
    ax^2 + bx + c = (px + q)^2 for all x -/
def IsPerfectSquareTrinomial (a b c : ℤ) : Prop :=
  ∃ p q : ℤ, ∀ x : ℤ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_k (k : ℤ) :
  IsPerfectSquareTrinomial 9 6 k → k = 1 := by
  sorry

#check perfect_square_trinomial_k

end perfect_square_trinomial_k_l1768_176832
