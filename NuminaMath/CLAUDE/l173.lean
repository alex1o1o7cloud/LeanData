import Mathlib

namespace chess_tournament_players_l173_17366

theorem chess_tournament_players (total_games : ℕ) (h : total_games = 306) :
  ∃ n : ℕ, n > 0 ∧ n * (n - 1) * 2 = total_games ∧ n = 18 := by
  sorry

end chess_tournament_players_l173_17366


namespace small_circle_radius_l173_17378

/-- Given a configuration of circles where:
    - There is a large circle with radius 10 meters
    - Six congruent smaller circles are arranged around it
    - Each smaller circle touches two others and the larger circle
    This theorem proves that the radius of each smaller circle is 5√3 meters -/
theorem small_circle_radius (R : ℝ) (r : ℝ) : 
  R = 10 → -- The radius of the larger circle is 10 meters
  R = (2 * r) / Real.sqrt 3 → -- Relationship between radii based on hexagon geometry
  r = 5 * Real.sqrt 3 := by sorry

end small_circle_radius_l173_17378


namespace f_comp_three_roots_l173_17306

/-- A quadratic function f(x) = x^2 + 6x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 6*x + c

/-- The composition of f with itself -/
def f_comp (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- Counts the number of distinct real roots of a function -/
noncomputable def count_distinct_roots (g : ℝ → ℝ) : ℕ := sorry

theorem f_comp_three_roots :
  ∃! c : ℝ, count_distinct_roots (f_comp c) = 3 ∧ c = (11 - Real.sqrt 13) / 2 := by sorry

end f_comp_three_roots_l173_17306


namespace product_of_consecutive_integers_l173_17362

theorem product_of_consecutive_integers : ∃ (a b c d e : ℤ),
  b = a + 1 ∧
  d = c + 1 ∧
  e = d + 1 ∧
  a * b = 300 ∧
  c * d * e = 300 ∧
  a + b + c + d + e = 49 :=
by sorry

end product_of_consecutive_integers_l173_17362


namespace function_growth_l173_17305

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State that f' is the derivative of f
variable (hf' : ∀ x, HasDerivAt f (f' x) x)

-- State the condition that f'(x) > f(x) for all x
variable (h : ∀ x, f' x > f x)

-- Theorem statement
theorem function_growth (f f' : ℝ → ℝ) (hf' : ∀ x, HasDerivAt f (f' x) x) (h : ∀ x, f' x > f x) :
  f 2012 > Real.exp 2012 * f 0 := by
  sorry

end function_growth_l173_17305


namespace logarithm_properties_l173_17376

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem logarithm_properties (a b x : ℝ) (ha : a > 0 ∧ a ≠ 1) (hb : b > 0 ∧ b ≠ 1) (hx : x > 0) :
  (log a x = (log b x) / (log b a)) ∧ (log a b = 1 / (log b a)) := by
  sorry

end logarithm_properties_l173_17376


namespace daily_savings_l173_17338

def original_coffees : ℕ := 4
def original_price : ℚ := 2
def price_increase_percentage : ℚ := 50
def new_coffees_ratio : ℚ := 1/2

def original_spending : ℚ := original_coffees * original_price

def new_price : ℚ := original_price * (1 + price_increase_percentage / 100)
def new_coffees : ℚ := original_coffees * new_coffees_ratio
def new_spending : ℚ := new_coffees * new_price

theorem daily_savings : original_spending - new_spending = 2 := by sorry

end daily_savings_l173_17338


namespace largest_n_for_factorization_l173_17348

/-- 
Given a quadratic expression of the form 6x^2 + nx + 72, 
this theorem states that the largest value of n for which 
the expression can be factored as the product of two linear 
factors with integer coefficients is 433.
-/
theorem largest_n_for_factorization : 
  (∀ n : ℤ, ∃ a b c d : ℤ, 
    (6 * x^2 + n * x + 72 = (a * x + b) * (c * x + d)) → n ≤ 433) ∧ 
  (∃ a b c d : ℤ, 6 * x^2 + 433 * x + 72 = (a * x + b) * (c * x + d)) := by
sorry


end largest_n_for_factorization_l173_17348


namespace hawks_score_l173_17334

/-- 
Given the total points scored and the winning margin in a basketball game,
this theorem proves the score of the losing team.
-/
theorem hawks_score (total_points winning_margin : ℕ) 
  (h1 : total_points = 42)
  (h2 : winning_margin = 6) : 
  (total_points - winning_margin) / 2 = 18 := by
  sorry

end hawks_score_l173_17334


namespace hyperbola_semi_focal_distance_l173_17381

/-- Given a hyperbola with equation x²/20 - y²/5 = 1, its semi-focal distance is 5 -/
theorem hyperbola_semi_focal_distance :
  ∀ (x y : ℝ), x^2 / 20 - y^2 / 5 = 1 → ∃ (c : ℝ), c = 5 ∧ c^2 = 20 + 5 := by
  sorry

end hyperbola_semi_focal_distance_l173_17381


namespace ferris_wheel_broken_seats_l173_17343

/-- The number of broken seats on a Ferris wheel -/
def broken_seats (total_seats : ℕ) (capacity_per_seat : ℕ) (current_capacity : ℕ) : ℕ :=
  total_seats - (current_capacity / capacity_per_seat)

/-- Theorem stating the number of broken seats on the Ferris wheel -/
theorem ferris_wheel_broken_seats :
  let total_seats : ℕ := 18
  let capacity_per_seat : ℕ := 15
  let current_capacity : ℕ := 120
  broken_seats total_seats capacity_per_seat current_capacity = 10 := by
  sorry

#eval broken_seats 18 15 120

end ferris_wheel_broken_seats_l173_17343


namespace cube_face_perimeter_l173_17399

-- Define the volume of the cube
def cube_volume : ℝ := 343

-- Theorem statement
theorem cube_face_perimeter :
  let side_length := (cube_volume ^ (1/3 : ℝ))
  (4 : ℝ) * side_length = 28 := by
  sorry

end cube_face_perimeter_l173_17399


namespace five_digit_multiple_of_three_l173_17396

def is_multiple_of_three (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

theorem five_digit_multiple_of_three :
  ∀ d : ℕ, d < 10 →
    (is_multiple_of_three (56780 + d) ↔ d = 1) :=
by sorry

end five_digit_multiple_of_three_l173_17396


namespace candy_packaging_remainder_l173_17323

theorem candy_packaging_remainder : 38759863 % 6 = 1 := by
  sorry

end candy_packaging_remainder_l173_17323


namespace de_moivres_formula_l173_17358

theorem de_moivres_formula (n : ℕ) (φ : ℝ) :
  (Complex.cos φ + Complex.I * Complex.sin φ) ^ n = Complex.cos (n * φ) + Complex.I * Complex.sin (n * φ) := by
  sorry

end de_moivres_formula_l173_17358


namespace total_copies_is_7050_l173_17317

/-- The total number of copies made by four copy machines in 30 minutes -/
def total_copies : ℕ :=
  let machine1 := 35 * 30
  let machine2 := 65 * 30
  let machine3 := 50 * 15 + 80 * 15
  let machine4 := 90 * 10 + 60 * 20
  machine1 + machine2 + machine3 + machine4

/-- Theorem stating that the total number of copies made by the four machines in 30 minutes is 7050 -/
theorem total_copies_is_7050 : total_copies = 7050 := by
  sorry

end total_copies_is_7050_l173_17317


namespace smallest_pretty_multiple_of_401_l173_17389

/-- A positive integer is pretty if for each of its proper divisors d,
    there exist two divisors whose difference is d. -/
def IsPretty (n : ℕ) : Prop :=
  n > 0 ∧ ∀ d : ℕ, d ∣ n → 1 < d → d < n →
    ∃ d₁ d₂ : ℕ, d₁ ∣ n ∧ d₂ ∣ n ∧ 1 ≤ d₁ ∧ d₁ ≤ n ∧ 1 ≤ d₂ ∧ d₂ ≤ n ∧ d₂ - d₁ = d

theorem smallest_pretty_multiple_of_401 :
  ∃ n : ℕ, n > 401 ∧ 401 ∣ n ∧ IsPretty n ∧
    ∀ m : ℕ, m > 401 → 401 ∣ m → IsPretty m → n ≤ m :=
by
  use 160400
  sorry

end smallest_pretty_multiple_of_401_l173_17389


namespace vance_family_stamp_cost_difference_l173_17371

theorem vance_family_stamp_cost_difference :
  let mr_rooster_count : ℕ := 3
  let mr_rooster_price : ℚ := 3/2
  let mr_daffodil_count : ℕ := 5
  let mr_daffodil_price : ℚ := 3/4
  let mrs_rooster_count : ℕ := 2
  let mrs_rooster_price : ℚ := 5/4
  let mrs_daffodil_count : ℕ := 7
  let mrs_daffodil_price : ℚ := 4/5
  let john_rooster_count : ℕ := 4
  let john_rooster_price : ℚ := 7/5
  let john_daffodil_count : ℕ := 3
  let john_daffodil_price : ℚ := 7/10

  let total_rooster_cost : ℚ := 
    mr_rooster_count * mr_rooster_price + 
    mrs_rooster_count * mrs_rooster_price + 
    john_rooster_count * john_rooster_price

  let total_daffodil_cost : ℚ := 
    mr_daffodil_count * mr_daffodil_price + 
    mrs_daffodil_count * mrs_daffodil_price + 
    john_daffodil_count * john_daffodil_price

  total_rooster_cost - total_daffodil_cost = 23/20
  := by sorry

end vance_family_stamp_cost_difference_l173_17371


namespace intersection_A_B_l173_17386

-- Define set A
def A : Set ℝ := {x : ℝ | 3 * x + 2 > 0}

-- Define set B
def B : Set ℝ := {x : ℝ | (x + 1) * (x - 3) > 0}

-- Theorem to prove
theorem intersection_A_B : A ∩ B = {x : ℝ | x > 3} := by
  sorry

end intersection_A_B_l173_17386


namespace negation_of_universal_proposition_l173_17380

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x > x^2) ↔ (∃ x : ℝ, Real.exp x ≤ x^2) := by
  sorry

end negation_of_universal_proposition_l173_17380


namespace custom_mult_theorem_l173_17340

/-- Custom multiplication operation -/
def custom_mult (a b : ℝ) : ℝ := (a - b) ^ 2

/-- Theorem stating that ((x-y)^2 + 1) * ((y-x)^2 + 1) = 0 for the custom multiplication -/
theorem custom_mult_theorem (x y : ℝ) : 
  custom_mult ((x - y) ^ 2 + 1) ((y - x) ^ 2 + 1) = 0 := by
  sorry


end custom_mult_theorem_l173_17340


namespace nh_not_equal_nk_l173_17331

/-- A structure representing a point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A structure representing a line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Given two points, return the line passing through them -/
def line_through_points (p1 p2 : Point) : Line :=
  sorry

/-- Given a point and a line, return the perpendicular line passing through the point -/
def perpendicular_line (p : Point) (l : Line) : Line :=
  sorry

/-- Given two lines, return the angle between them in radians -/
def angle_between_lines (l1 l2 : Line) : ℝ :=
  sorry

/-- Given two points, return the distance between them -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Given three points A, B, C, return a point that is 1/3 of the way from A to B -/
def one_third_point (a b : Point) : Point :=
  sorry

theorem nh_not_equal_nk (h k y z : Point) :
  let hk : Line := line_through_points h k
  let yz : Line := line_through_points y z
  let n : Point := one_third_point y z
  let yh : Line := perpendicular_line y hk
  let zk : Line := line_through_points z k
  angle_between_lines hk zk = π / 4 →
  distance n h ≠ distance n k :=
sorry

end nh_not_equal_nk_l173_17331


namespace percentage_of_indian_women_l173_17383

theorem percentage_of_indian_women (total_men : ℕ) (total_women : ℕ) (total_children : ℕ)
  (percent_indian_men : ℝ) (percent_indian_children : ℝ) (percent_not_indian : ℝ) :
  total_men = 500 →
  total_women = 300 →
  total_children = 500 →
  percent_indian_men = 10 →
  percent_indian_children = 70 →
  percent_not_indian = 55.38461538461539 →
  ∃ (percent_indian_women : ℝ),
    percent_indian_women = 60 ∧
    (percent_indian_men / 100 * total_men + percent_indian_women / 100 * total_women + percent_indian_children / 100 * total_children) /
    (total_men + total_women + total_children : ℝ) = 1 - percent_not_indian / 100 :=
by sorry

end percentage_of_indian_women_l173_17383


namespace starting_number_of_range_l173_17325

theorem starting_number_of_range (n : ℕ) (h1 : n ≤ 31) (h2 : n % 3 = 0) 
  (h3 : ∀ k, n - 18 ≤ k ∧ k ≤ n → k % 3 = 0) : n - 18 = 12 := by
  sorry

end starting_number_of_range_l173_17325


namespace harris_dog_vegetable_cost_l173_17368

/-- Represents the cost and quantity of a vegetable in a 1-pound bag -/
structure VegetableInfo where
  quantity : ℕ
  cost : ℚ

/-- Calculates the annual cost of vegetables for Harris's dog -/
def annual_vegetable_cost (carrot_info celery_info pepper_info : VegetableInfo) 
  (daily_carrot daily_celery daily_pepper : ℕ) : ℚ :=
  let daily_cost := 
    daily_carrot * (carrot_info.cost / carrot_info.quantity) +
    daily_celery * (celery_info.cost / celery_info.quantity) +
    daily_pepper * (pepper_info.cost / pepper_info.quantity)
  daily_cost * 365

/-- Theorem stating the annual cost of vegetables for Harris's dog -/
theorem harris_dog_vegetable_cost :
  let carrot_info : VegetableInfo := ⟨5, 2⟩
  let celery_info : VegetableInfo := ⟨10, 3/2⟩
  let pepper_info : VegetableInfo := ⟨3, 5/2⟩
  annual_vegetable_cost carrot_info celery_info pepper_info 1 2 1 = 11169/20 := by
  sorry

end harris_dog_vegetable_cost_l173_17368


namespace asterisk_replacement_l173_17304

theorem asterisk_replacement : (60 / 20) * (60 / 180) = 1 := by
  sorry

end asterisk_replacement_l173_17304


namespace isosceles_triangle_perimeter_l173_17395

/-- An isosceles triangle with two sides of length 12 and a third side of length 17 has a perimeter of 41. -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (a b c : ℝ),
      a = 12 ∧ b = 12 ∧ c = 17 ∧  -- Two sides are 12, third side is 17
      (a = b ∨ a = c ∨ b = c) ∧   -- Definition of isosceles triangle
      perimeter = a + b + c ∧     -- Definition of perimeter
      perimeter = 41              -- The perimeter we want to prove

/-- Proof of the theorem -/
lemma proof_isosceles_triangle_perimeter : isosceles_triangle_perimeter 41 := by
  sorry

#check proof_isosceles_triangle_perimeter

end isosceles_triangle_perimeter_l173_17395


namespace rectangle_circle_area_ratio_l173_17347

theorem rectangle_circle_area_ratio 
  (l w r : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : 2 * l + 2 * w = 2 * Real.pi * r) : 
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
  sorry

end rectangle_circle_area_ratio_l173_17347


namespace number_square_relationship_l173_17369

theorem number_square_relationship (n : ℝ) (h1 : n ≠ 0) (h2 : (n + n^2) / 2 = 5 * n) (h3 : n = 9) :
  (n + n^2) / 2 = 5 * n :=
by sorry

end number_square_relationship_l173_17369


namespace subset_transitive_and_complement_subset_l173_17330

variable {α : Type*}
variable (U : Set α)

theorem subset_transitive_and_complement_subset : 
  (∀ A B C : Set α, A ⊆ B → B ⊆ C → A ⊆ C) ∧ 
  (∀ A B : Set α, A ⊆ B → (U \ B) ⊆ (U \ A)) :=
sorry

end subset_transitive_and_complement_subset_l173_17330


namespace yellow_balls_count_l173_17313

theorem yellow_balls_count (total : ℕ) (white green red purple : ℕ) (prob : ℚ) :
  total = 100 →
  white = 50 →
  green = 30 →
  red = 9 →
  purple = 3 →
  prob = 88/100 →
  prob = (white + green + (total - white - green - red - purple)) / total →
  total - white - green - red - purple = 8 :=
by sorry

end yellow_balls_count_l173_17313


namespace b_months_is_five_l173_17398

/-- Represents the grazing arrangement for a pasture -/
structure GrazingArrangement where
  a_oxen : ℕ
  a_months : ℕ
  b_oxen : ℕ
  c_oxen : ℕ
  c_months : ℕ
  total_rent : ℚ
  c_share : ℚ

/-- Calculates the number of months B's oxen grazed given a GrazingArrangement -/
def calculate_b_months (g : GrazingArrangement) : ℚ :=
  ((g.total_rent - g.c_share) * g.c_oxen * g.c_months - g.a_oxen * g.a_months * g.c_share) /
  (g.b_oxen * g.c_share)

/-- Theorem stating that B's oxen grazed for 5 months under the given conditions -/
theorem b_months_is_five (g : GrazingArrangement) 
    (h1 : g.a_oxen = 10) 
    (h2 : g.a_months = 7) 
    (h3 : g.b_oxen = 12) 
    (h4 : g.c_oxen = 15) 
    (h5 : g.c_months = 3) 
    (h6 : g.total_rent = 175) 
    (h7 : g.c_share = 45) : 
  calculate_b_months g = 5 := by
  sorry

end b_months_is_five_l173_17398


namespace unique_solution_l173_17312

/-- Represents the number of children in each class -/
structure ClassSizes where
  judo : ℕ
  agriculture : ℕ
  math : ℕ

/-- Checks if the given class sizes satisfy all conditions -/
def satisfiesConditions (sizes : ClassSizes) : Prop :=
  sizes.judo + sizes.agriculture + sizes.math = 32 ∧
  sizes.judo > 0 ∧ sizes.agriculture > 0 ∧ sizes.math > 0 ∧
  sizes.judo / 2 + sizes.agriculture / 4 + sizes.math / 8 = 6

/-- The theorem stating that the unique solution satisfying all conditions is (4, 4, 24) -/
theorem unique_solution : 
  ∃! sizes : ClassSizes, satisfiesConditions sizes ∧ 
  sizes.judo = 4 ∧ sizes.agriculture = 4 ∧ sizes.math = 24 :=
sorry

end unique_solution_l173_17312


namespace intersection_nonempty_iff_a_geq_neg_eight_l173_17364

-- Define set A
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*x + a ≥ 0}

-- Theorem statement
theorem intersection_nonempty_iff_a_geq_neg_eight :
  ∀ a : ℝ, (A ∩ B a).Nonempty ↔ a ≥ -8 := by sorry

end intersection_nonempty_iff_a_geq_neg_eight_l173_17364


namespace min_value_is_11_l173_17321

-- Define the variables and constraints
def is_feasible (x y : ℝ) : Prop :=
  x * y - 3 ≥ 0 ∧ x - y ≥ 1 ∧ y ≥ 5

-- Define the objective function
def objective_function (x y : ℝ) : ℝ :=
  3 * x + 4 * y

-- Theorem statement
theorem min_value_is_11 :
  ∀ x y : ℝ, is_feasible x y →
  objective_function x y ≥ 11 ∧
  ∃ x₀ y₀ : ℝ, is_feasible x₀ y₀ ∧ objective_function x₀ y₀ = 11 :=
sorry

end min_value_is_11_l173_17321


namespace choose_three_cooks_from_twelve_l173_17332

theorem choose_three_cooks_from_twelve (n : Nat) (k : Nat) : Nat.choose 12 3 = 220 := by
  sorry

end choose_three_cooks_from_twelve_l173_17332


namespace stone_game_termination_and_uniqueness_l173_17337

/-- Represents the state of stones on the infinite strip --/
def StoneConfiguration := Int → ℕ

/-- Represents a move on the strip --/
inductive Move
  | typeA (n : Int) : Move
  | typeB (n : Int) : Move

/-- Applies a move to a configuration --/
def applyMove (config : StoneConfiguration) (move : Move) : StoneConfiguration :=
  match move with
  | Move.typeA n => fun i =>
      if i = n - 1 || i = n then config i - 1
      else if i = n + 1 then config i + 1
      else config i
  | Move.typeB n => fun i =>
      if i = n then config i - 2
      else if i = n + 1 || i = n - 2 then config i + 1
      else config i

/-- Checks if a move is valid for a given configuration --/
def isValidMove (config : StoneConfiguration) (move : Move) : Prop :=
  match move with
  | Move.typeA n => config (n - 1) > 0 ∧ config n > 0
  | Move.typeB n => config n ≥ 2

/-- Checks if any move is possible for a given configuration --/
def canMove (config : StoneConfiguration) : Prop :=
  ∃ (move : Move), isValidMove config move

/-- The theorem to be proved --/
theorem stone_game_termination_and_uniqueness 
  (initial : StoneConfiguration) : 
  ∃! (final : StoneConfiguration), 
    (∃ (moves : List Move), (moves.foldl applyMove initial = final)) ∧ 
    ¬(canMove final) := by
  sorry

end stone_game_termination_and_uniqueness_l173_17337


namespace expression_simplification_and_evaluation_l173_17375

theorem expression_simplification_and_evaluation :
  let x : ℝ := 2 * Real.sqrt 5 - 1
  (1 / (x^2 + 2*x + 1)) * (1 + 3 / (x - 1)) / ((x + 2) / (x^2 - 1)) = Real.sqrt 5 / 10 := by
  sorry

end expression_simplification_and_evaluation_l173_17375


namespace tree_planting_theorem_l173_17316

/-- Represents the tree planting activity -/
structure TreePlanting where
  totalVolunteers : ℕ
  poplars : ℕ
  seaBuckthorns : ℕ
  poplarTime : ℚ
  seaBuckthornsTime1 : ℚ
  seaBuckthornsTime2 : ℚ
  transferredVolunteers : ℕ

/-- Calculates the optimal allocation and durations for the tree planting activity -/
def optimalAllocation (tp : TreePlanting) : 
  (ℕ × ℕ) × ℚ × ℚ :=
  sorry

/-- The theorem stating the correctness of the optimal allocation and durations -/
theorem tree_planting_theorem (tp : TreePlanting) 
  (h1 : tp.totalVolunteers = 52)
  (h2 : tp.poplars = 150)
  (h3 : tp.seaBuckthorns = 200)
  (h4 : tp.poplarTime = 2/5)
  (h5 : tp.seaBuckthornsTime1 = 1/2)
  (h6 : tp.seaBuckthornsTime2 = 2/3)
  (h7 : tp.transferredVolunteers = 6) :
  let (allocation, initialDuration, finalDuration) := optimalAllocation tp
  allocation = (20, 32) ∧ 
  initialDuration = 25/8 ∧
  finalDuration = 27/7 :=
sorry

end tree_planting_theorem_l173_17316


namespace special_blend_probability_l173_17370

theorem special_blend_probability : 
  let n : ℕ := 6  -- Total number of visits
  let k : ℕ := 5  -- Number of times the special blend is served
  let p : ℚ := 3/4  -- Probability of serving the special blend each day
  Nat.choose n k * p^k * (1-p)^(n-k) = 1458/4096 := by
sorry

end special_blend_probability_l173_17370


namespace correct_operation_l173_17345

theorem correct_operation (a b : ℝ) : -a^2*b + 2*a^2*b = a^2*b := by
  sorry

end correct_operation_l173_17345


namespace total_questions_on_test_l173_17392

/-- Represents a student's test results -/
structure TestResult where
  score : Int
  correct : Nat
  total : Nat

/-- Calculates the score based on correct and incorrect responses -/
def calculateScore (correct : Nat) (incorrect : Nat) : Int :=
  correct - 2 * incorrect

/-- Theorem: Given the scoring system and Student A's results, prove the total number of questions -/
theorem total_questions_on_test (result : TestResult) 
  (h1 : result.score = calculateScore result.correct (result.total - result.correct))
  (h2 : result.score = 76)
  (h3 : result.correct = 92) :
  result.total = 100 := by
  sorry

#eval calculateScore 92 8

end total_questions_on_test_l173_17392


namespace solution_comparison_l173_17307

theorem solution_comparison (p p' q q' : ℝ) (hp : p ≠ 0) (hp' : p' ≠ 0) :
  (-q' / p > -q / p') ↔ (q' / p < q / p') :=
by sorry

end solution_comparison_l173_17307


namespace quadratic_root_sums_l173_17382

theorem quadratic_root_sums (a b c x₁ x₂ : ℝ) (h : a ≠ 0) : 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁^2 + x₂^2 = (b^2 - 2*a*c) / a^2 ∧ 
   x₁^3 + x₂^3 = (3*a*b*c - b^3) / a^3) := by
sorry

end quadratic_root_sums_l173_17382


namespace bus_interval_theorem_l173_17354

/-- The interval between buses on a circular route -/
def interval (num_buses : ℕ) (total_time : ℕ) : ℕ :=
  total_time / num_buses

/-- The theorem stating the relationship between intervals for 2 and 3 buses -/
theorem bus_interval_theorem (total_time : ℕ) :
  interval 2 total_time = 21 → interval 3 total_time = 14 :=
by
  sorry

#check bus_interval_theorem

end bus_interval_theorem_l173_17354


namespace wiener_age_theorem_l173_17391

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

theorem wiener_age_theorem :
  ∃! a : ℕ, 
    is_four_digit (a^3) ∧ 
    is_six_digit (a^4) ∧ 
    (digits (a^3) ++ digits (a^4)).Nodup ∧
    (digits (a^3) ++ digits (a^4)).length = 10 ∧
    a = 18 := by
  sorry

end wiener_age_theorem_l173_17391


namespace book_cd_price_difference_l173_17367

/-- Proves that the difference between book price and CD price is $4 -/
theorem book_cd_price_difference :
  let album_price : ℝ := 20
  let cd_price : ℝ := 0.7 * album_price
  let book_price : ℝ := 18
  book_price - cd_price = 4 := by
  sorry

end book_cd_price_difference_l173_17367


namespace complex_magnitude_l173_17342

theorem complex_magnitude (z : ℂ) : z = (2 + Complex.I) / Complex.I + Complex.I → Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_magnitude_l173_17342


namespace relationship_correctness_l173_17344

theorem relationship_correctness :
  (∃ a b c : ℝ, (a > b ↔ a * c^2 > b * c^2) → False) ∧
  (∃ a b : ℝ, (a > b → 1 / a < 1 / b) → False) ∧
  (∃ a b c d : ℝ, (a > b ∧ b > 0 ∧ c > d → a / d > b / c) → False) ∧
  (∀ a b c : ℝ, a > b ∧ b > 0 → a^c < b^c) :=
by sorry


end relationship_correctness_l173_17344


namespace exists_growth_rate_unique_growth_rate_l173_17314

/-- Represents the average annual growth rate of Fujian's regional GDP from 2020 to 2022 -/
def average_annual_growth_rate (x : ℝ) : Prop :=
  43903.89 * (1 + x)^2 = 53109.85

/-- The initial GDP of Fujian in 2020 (in billion yuan) -/
def initial_gdp : ℝ := 43903.89

/-- The GDP of Fujian in 2022 (in billion yuan) -/
def final_gdp : ℝ := 53109.85

/-- Theorem stating that there exists an average annual growth rate satisfying the equation -/
theorem exists_growth_rate : ∃ x : ℝ, average_annual_growth_rate x :=
  sorry

/-- Theorem stating that the average annual growth rate is unique -/
theorem unique_growth_rate : ∀ x y : ℝ, average_annual_growth_rate x → average_annual_growth_rate y → x = y :=
  sorry

end exists_growth_rate_unique_growth_rate_l173_17314


namespace money_ratio_l173_17333

/-- Proves that given the total money between three people is $68, one person (Doug) has $32, 
    and another person (Josh) has 3/4 as much as Doug, the ratio of Josh's money to the 
    third person's (Brad's) money is 2:1. -/
theorem money_ratio (total : ℚ) (doug : ℚ) (josh : ℚ) (brad : ℚ) 
  (h1 : total = 68)
  (h2 : doug = 32)
  (h3 : josh = (3/4) * doug)
  (h4 : total = josh + doug + brad) :
  josh / brad = 2 / 1 := by
  sorry

end money_ratio_l173_17333


namespace man_and_son_work_time_l173_17303

/-- Given a task that takes a man 5 days and his son 20 days to complete individually,
    prove that they can complete the task together in 4 days. -/
theorem man_and_son_work_time (task : ℝ) (man_rate son_rate combined_rate : ℝ) : 
  task > 0 ∧ 
  man_rate = task / 5 ∧ 
  son_rate = task / 20 ∧ 
  combined_rate = man_rate + son_rate →
  task / combined_rate = 4 := by
sorry

end man_and_son_work_time_l173_17303


namespace vector_perpendicular_condition_l173_17394

theorem vector_perpendicular_condition (a b : ℝ × ℝ) (m : ℝ) : 
  ‖a‖ = 3 →
  ‖b‖ = 2 →
  a • b = 3 →
  (a - m • b) • a = 0 →
  m = 3 := by
sorry

end vector_perpendicular_condition_l173_17394


namespace sahara_temperature_difference_l173_17352

/-- The maximum temperature difference in the Sahara Desert --/
theorem sahara_temperature_difference (highest_temp lowest_temp : ℤ) 
  (h_highest : highest_temp = 58)
  (h_lowest : lowest_temp = -34) :
  highest_temp - lowest_temp = 92 := by
  sorry

end sahara_temperature_difference_l173_17352


namespace negation_of_universal_proposition_l173_17346

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l173_17346


namespace megan_seashells_l173_17390

/-- Given that Megan has 19 seashells and wants to have 25 seashells in total,
    prove that she needs to find 6 more seashells. -/
theorem megan_seashells (current : ℕ) (target : ℕ) (h1 : current = 19) (h2 : target = 25) :
  target - current = 6 := by
  sorry

end megan_seashells_l173_17390


namespace zark_game_threshold_l173_17328

/-- The score for dropping n zarks -/
def drop_score (n : ℕ) : ℕ := n^2

/-- The score for eating n zarks -/
def eat_score (n : ℕ) : ℕ := 15 * n

/-- 16 is the smallest positive integer n for which dropping n zarks scores more than eating them -/
theorem zark_game_threshold : ∀ n : ℕ, n > 0 → (drop_score n > eat_score n ↔ n ≥ 16) :=
by sorry

end zark_game_threshold_l173_17328


namespace square_sum_theorem_l173_17320

theorem square_sum_theorem (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 3) :
  a^2 + b^2 = 31 := by
  sorry

end square_sum_theorem_l173_17320


namespace sum_of_cubes_of_roots_l173_17353

theorem sum_of_cubes_of_roots (x₁ x₂ x₃ : ℂ) : 
  x₁^3 + x₂^3 + x₃^3 = 0 → x₁ + x₂ + x₃ = -2 → x₁*x₂ + x₂*x₃ + x₃*x₁ = 1 → x₁*x₂*x₃ = 3 → 
  x₁^3 + x₂^3 + x₃^3 = 7 :=
by sorry

end sum_of_cubes_of_roots_l173_17353


namespace ball_distribution_theorem_l173_17339

/-- The number of ways to distribute indistinguishable balls into boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  Nat.choose (num_balls + num_boxes - 1) num_balls

/-- The number of ways to distribute balls of three colors into boxes -/
def distribute_three_colors (num_balls_per_color : ℕ) (num_boxes : ℕ) : ℕ :=
  (distribute_balls num_balls_per_color num_boxes) ^ 3

theorem ball_distribution_theorem (num_balls_per_color num_boxes : ℕ) 
  (h1 : num_balls_per_color = 4) 
  (h2 : num_boxes = 6) : 
  distribute_three_colors num_balls_per_color num_boxes = (Nat.choose 9 4) ^ 3 := by
  sorry

#eval distribute_three_colors 4 6

end ball_distribution_theorem_l173_17339


namespace opposite_of_two_thirds_l173_17350

theorem opposite_of_two_thirds :
  -(2 / 3 : ℚ) = -2 / 3 := by sorry

end opposite_of_two_thirds_l173_17350


namespace system_two_solutions_l173_17326

/-- The system of equations has exactly two solutions if and only if a = 1 or a = 25 -/
theorem system_two_solutions (a : ℝ) :
  (∃! (x₁ y₁ x₂ y₂ : ℝ), 
    (abs (y₁ - 3 - x₁) + abs (y₁ - 3 + x₁) = 6 ∧
     (abs x₁ - 4)^2 + (abs y₁ - 3)^2 = a) ∧
    (abs (y₂ - 3 - x₂) + abs (y₂ - 3 + x₂) = 6 ∧
     (abs x₂ - 4)^2 + (abs y₂ - 3)^2 = a) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) ↔ 
  (a = 1 ∨ a = 25) :=
by sorry

end system_two_solutions_l173_17326


namespace arithmetic_geometric_sequence_problem_l173_17387

/-- Arithmetic sequence term -/
def a_n (a b n : ℕ) : ℕ := a + (n - 1) * b

/-- Geometric sequence term -/
def b_n (a b n : ℕ) : ℕ := b * a^(n - 1)

/-- C_n sequence term -/
def C_n (a b n : ℕ) : ℕ := a_n a b (n + 1) + b_n a b n

theorem arithmetic_geometric_sequence_problem 
  (a b : ℕ) 
  (h_a_pos : a > 1) 
  (h_b_pos : b > 1) 
  (h_a1_lt_b1 : a < b) 
  (h_b2_lt_a3 : b * a < a + 2 * b) 
  (h_exists_m : ∀ n : ℕ, n > 0 → ∃ m : ℕ, m > 0 ∧ a_n a b m + 3 = b_n a b n) :
  (a = 2 ∧ b = 5) ∧ 
  (b = 4 → 
    ∃ n : ℕ, C_n a b n = 18 ∧ C_n a b (n + 1) = 30 ∧ C_n a b (n + 2) = 50 ∧
    ∀ k : ℕ, k ≠ n → ¬(C_n a b k * C_n a b (k + 2) = (C_n a b (k + 1))^2)) :=
sorry

end arithmetic_geometric_sequence_problem_l173_17387


namespace shmacks_in_shneids_l173_17318

-- Define the conversion rates
def shmacks_per_shick : ℚ := 5 / 2
def shicks_per_shure : ℚ := 3 / 5
def shures_per_shneid : ℚ := 2 / 9

-- Define the problem
def shneids_to_convert : ℚ := 6

-- Theorem to prove
theorem shmacks_in_shneids : 
  shneids_to_convert * shures_per_shneid * shicks_per_shure * shmacks_per_shick = 2 := by
  sorry

end shmacks_in_shneids_l173_17318


namespace veronica_ring_removal_ways_l173_17365

/-- Represents the number of rings on each finger --/
structure RingDistribution :=
  (little : Nat)
  (middle : Nat)
  (ring : Nat)

/-- Calculates the number of ways to remove rings given a distribution --/
def removalWays (dist : RingDistribution) (fixedOrderOnRingFinger : Bool) : Nat :=
  if fixedOrderOnRingFinger then
    dist.little * dist.middle
  else
    sorry

/-- The specific ring distribution in the problem --/
def veronicaRings : RingDistribution :=
  { little := 1, middle := 1, ring := 3 }

theorem veronica_ring_removal_ways :
  removalWays veronicaRings true = 20 := by sorry

end veronica_ring_removal_ways_l173_17365


namespace negation_of_statement_l173_17356

def S : Set Int := {1, -1, 0}

theorem negation_of_statement :
  (¬ ∀ x ∈ S, 2 * x + 1 > 0) ↔ (∃ x ∈ S, 2 * x + 1 ≤ 0) := by
  sorry

end negation_of_statement_l173_17356


namespace descending_order_exists_l173_17341

theorem descending_order_exists (x y z : ℤ) : ∃ (a b c : ℤ), 
  ({a, b, c} : Finset ℤ) = {x, y, z} ∧ a ≥ b ∧ b ≥ c := by sorry

end descending_order_exists_l173_17341


namespace max_value_of_f_l173_17300

noncomputable def f (x : ℝ) : ℝ := x^2 - 8*x + 6*Real.log x + 1

theorem max_value_of_f :
  ∃ (c : ℝ), c > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≤ f c ∧ f c = -6 :=
sorry

end max_value_of_f_l173_17300


namespace system_solution_existence_l173_17372

theorem system_solution_existence (b : ℝ) : 
  (∃ a : ℝ, ∃ x y : ℝ, x = |y - b| + 3/b ∧ x^2 + y^2 + 32 = a*(2*y - a) + 12*x) ↔ 
  (b < 0 ∨ b ≥ 3/8) :=
by sorry

end system_solution_existence_l173_17372


namespace complement_intersection_A_B_l173_17397

def U : Set Nat := {1,2,3,4,5,6,7,8}
def A : Set Nat := {1,2,3}
def B : Set Nat := {2,3,4,5}

theorem complement_intersection_A_B : 
  (U \ (A ∩ B)) = {1,4,5,6,7,8} := by sorry

end complement_intersection_A_B_l173_17397


namespace new_ratio_after_addition_l173_17384

theorem new_ratio_after_addition (x : ℤ) : 
  (x : ℚ) / (4 * x : ℚ) = 1 / 4 →
  4 * x = 24 →
  (x + 6 : ℚ) / (4 * x : ℚ) = 1 / 2 := by
  sorry

end new_ratio_after_addition_l173_17384


namespace option_2_cheaper_for_42_options_equal_at_45_unique_equality_at_45_l173_17310

-- Define the ticket price
def ticket_price : ℕ := 30

-- Define the discount rates
def discount_rate_1 : ℚ := 0.2
def discount_rate_2 : ℚ := 0.1

-- Define the number of free tickets in Option 2
def free_tickets : ℕ := 5

-- Function to calculate cost for Option 1
def cost_option_1 (students : ℕ) : ℚ :=
  (students : ℚ) * ticket_price * (1 - discount_rate_1)

-- Function to calculate cost for Option 2
def cost_option_2 (students : ℕ) : ℚ :=
  ((students - free_tickets) : ℚ) * ticket_price * (1 - discount_rate_2)

-- Theorem 1: For 42 students, Option 2 is cheaper
theorem option_2_cheaper_for_42 : cost_option_2 42 < cost_option_1 42 := by sorry

-- Theorem 2: Both options are equal when there are 45 students
theorem options_equal_at_45 : cost_option_1 45 = cost_option_2 45 := by sorry

-- Theorem 3: 45 is the only number of students (> 40) where both options are equal
theorem unique_equality_at_45 :
  ∀ n : ℕ, n > 40 → cost_option_1 n = cost_option_2 n → n = 45 := by sorry

end option_2_cheaper_for_42_options_equal_at_45_unique_equality_at_45_l173_17310


namespace brothers_multiple_l173_17322

/-- Given that Aaron has 4 brothers and Bennett has 6 brothers, 
    prove that the multiple relating their number of brothers is 2. -/
theorem brothers_multiple (aaron_brothers : ℕ) (bennett_brothers : ℕ) : 
  aaron_brothers = 4 → bennett_brothers = 6 → ∃ x : ℕ, x * aaron_brothers - 2 = bennett_brothers ∧ x = 2 := by
  sorry

end brothers_multiple_l173_17322


namespace goods_train_speed_l173_17302

/-- Calculates the speed of a goods train given the conditions of the problem. -/
theorem goods_train_speed
  (man_train_speed : ℝ)
  (passing_time : ℝ)
  (goods_train_length : ℝ)
  (h1 : man_train_speed = 20)
  (h2 : passing_time = 9)
  (h3 : goods_train_length = 280) :
  ∃ (goods_train_speed : ℝ),
    goods_train_speed = 92 ∧
    (man_train_speed + goods_train_speed) * (1 / 3.6) = goods_train_length / passing_time :=
by sorry

end goods_train_speed_l173_17302


namespace sports_equipment_problem_l173_17361

/-- Represents the purchase and selling prices of sports equipment -/
structure SportsPrices where
  tabletennis_purchase : ℝ
  badminton_purchase : ℝ
  tabletennis_sell : ℝ
  badminton_sell : ℝ

/-- Represents the number of sets and profit -/
structure SalesData where
  tabletennis_sets : ℝ
  profit : ℝ

/-- Theorem stating the conditions and results of the sports equipment problem -/
theorem sports_equipment_problem 
  (prices : SportsPrices)
  (sales : SalesData) :
  -- Conditions
  2 * prices.tabletennis_purchase + prices.badminton_purchase = 110 ∧
  4 * prices.tabletennis_purchase + 3 * prices.badminton_purchase = 260 ∧
  prices.tabletennis_sell = 50 ∧
  prices.badminton_sell = 60 ∧
  sales.tabletennis_sets ≤ 150 ∧
  sales.tabletennis_sets ≥ (300 - sales.tabletennis_sets) / 2 →
  -- Results
  prices.tabletennis_purchase = 35 ∧
  prices.badminton_purchase = 40 ∧
  sales.profit = -5 * sales.tabletennis_sets + 6000 ∧
  100 ≤ sales.tabletennis_sets ∧ sales.tabletennis_sets ≤ 150 ∧
  (∀ a : ℝ, 0 < a ∧ a < 10 →
    (a < 5 → sales.tabletennis_sets = 100) ∧
    (a > 5 → sales.tabletennis_sets = 150) ∧
    (a = 5 → sales.profit = 6000)) :=
by sorry

end sports_equipment_problem_l173_17361


namespace simplify_fraction_l173_17319

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  3 / (x - 1) + (x - 3) / (1 - x^2) = (2*x + 6) / (x^2 - 1) := by
  sorry

end simplify_fraction_l173_17319


namespace arithmetic_mean_of_set_l173_17360

def number_set : List ℝ := [16, 23, 38, 11.5]

theorem arithmetic_mean_of_set : 
  (number_set.sum / number_set.length : ℝ) = 22.125 := by
  sorry

end arithmetic_mean_of_set_l173_17360


namespace swimming_problem_solution_l173_17393

/-- Represents the amount paid by each person -/
structure Payment where
  adam : ℕ
  bill : ℕ
  chris : ℕ

/-- The problem setup -/
def swimming_problem : Prop :=
  ∃ (cost_per_session : ℕ) (final_payment : Payment),
    -- Total number of sessions
    15 * cost_per_session = final_payment.adam + final_payment.bill + final_payment.chris
    -- Adam paid 8 times
    ∧ 8 * cost_per_session = final_payment.adam + 18
    -- Bill paid 7 times
    ∧ 7 * cost_per_session = final_payment.bill + 12
    -- Chris owes £30
    ∧ final_payment.chris = 30
    -- All have paid the same amount after Chris's payment
    ∧ final_payment.adam = final_payment.bill
    ∧ final_payment.bill = final_payment.chris

theorem swimming_problem_solution : swimming_problem := by
  sorry

end swimming_problem_solution_l173_17393


namespace triangle_height_l173_17336

theorem triangle_height (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 3 ∧ BC = Real.sqrt 13 ∧ AC = 4 →
  ∃ D : ℝ × ℝ, 
    (D.1 - A.1) * (C.1 - A.1) + (D.2 - A.2) * (C.2 - A.2) = 0 ∧
    Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 3/2 * Real.sqrt 3 :=
by sorry

end triangle_height_l173_17336


namespace cucumber_water_percentage_l173_17335

/-- Given the initial and final conditions of cucumbers after water evaporation,
    prove that the initial water percentage was 99%. -/
theorem cucumber_water_percentage
  (initial_weight : ℝ)
  (final_water_percentage : ℝ)
  (final_weight : ℝ)
  (h_initial_weight : initial_weight = 100)
  (h_final_water_percentage : final_water_percentage = 96)
  (h_final_weight : final_weight = 25) :
  (initial_weight - (1 - final_water_percentage / 100) * final_weight) / initial_weight * 100 = 99 :=
by sorry

end cucumber_water_percentage_l173_17335


namespace additional_decorations_to_buy_l173_17349

def halloween_decorations (skulls broomsticks spiderwebs pumpkins cauldron left_to_put_up total : ℕ) : Prop :=
  skulls = 12 ∧
  broomsticks = 4 ∧
  spiderwebs = 12 ∧
  pumpkins = 2 * spiderwebs ∧
  cauldron = 1 ∧
  left_to_put_up = 10 ∧
  total = 83

theorem additional_decorations_to_buy 
  (skulls broomsticks spiderwebs pumpkins cauldron left_to_put_up total : ℕ)
  (h : halloween_decorations skulls broomsticks spiderwebs pumpkins cauldron left_to_put_up total) :
  total - (skulls + broomsticks + spiderwebs + pumpkins + cauldron) - left_to_put_up = 20 :=
sorry

end additional_decorations_to_buy_l173_17349


namespace chemical_solution_concentration_l173_17351

theorem chemical_solution_concentration 
  (initial_concentration : ℝ)
  (replacement_concentration : ℝ)
  (replaced_portion : ℝ)
  (h1 : initial_concentration = 0.85)
  (h2 : replacement_concentration = 0.3)
  (h3 : replaced_portion = 0.8181818181818182)
  (h4 : replaced_portion ≥ 0 ∧ replaced_portion ≤ 1) :
  let remaining_portion := 1 - replaced_portion
  let final_concentration := 
    (remaining_portion * initial_concentration + replaced_portion * replacement_concentration)
  final_concentration = 0.4 :=
by sorry

end chemical_solution_concentration_l173_17351


namespace option2_more_cost_effective_l173_17324

/-- The cost of a pair of badminton rackets in dollars -/
def racket_cost : ℕ := 100

/-- The cost of a box of shuttlecocks in dollars -/
def shuttlecock_cost : ℕ := 20

/-- The number of pairs of badminton rackets the school wants to buy -/
def racket_pairs : ℕ := 10

/-- The number of boxes of shuttlecocks the school wants to buy -/
def shuttlecock_boxes : ℕ := 60

/-- The cost of Option 1 in dollars -/
def option1_cost (x : ℕ) : ℕ := 20 * x + 800

/-- The cost of Option 2 in dollars -/
def option2_cost (x : ℕ) : ℕ := 18 * x + 900

/-- Theorem stating that Option 2 is more cost-effective when x = 60 -/
theorem option2_more_cost_effective :
  shuttlecock_boxes > 10 →
  option1_cost shuttlecock_boxes > option2_cost shuttlecock_boxes :=
by sorry

end option2_more_cost_effective_l173_17324


namespace inequality_solution_range_l173_17357

theorem inequality_solution_range (b : ℝ) : 
  (∀ x : ℤ, |3 * (x : ℝ) - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) →
  (5 < b ∧ b < 7) :=
by sorry

end inequality_solution_range_l173_17357


namespace board_piece_difference_l173_17355

def board_length : ℝ := 20
def shorter_piece : ℝ := 8

theorem board_piece_difference : 
  let longer_piece := board_length - shorter_piece
  2 * shorter_piece - longer_piece = 4 := by
  sorry

end board_piece_difference_l173_17355


namespace acquaintance_theorem_l173_17363

/-- A graph with 9 vertices where every subset of 3 vertices contains at least 2 connected vertices -/
def AcquaintanceGraph : Type :=
  { g : SimpleGraph (Fin 9) // ∀ (s : Finset (Fin 9)), s.card = 3 →
    ∃ (v w : Fin 9), v ∈ s ∧ w ∈ s ∧ v ≠ w ∧ g.Adj v w }

/-- The existence of a complete subgraph of 4 vertices in the AcquaintanceGraph -/
theorem acquaintance_theorem (g : AcquaintanceGraph) :
  ∃ (s : Finset (Fin 9)), s.card = 4 ∧ ∀ (v w : Fin 9), v ∈ s → w ∈ s → v ≠ w → g.val.Adj v w :=
sorry

end acquaintance_theorem_l173_17363


namespace x_value_proof_l173_17327

theorem x_value_proof : ∀ x : ℝ, x + Real.sqrt 25 = Real.sqrt 36 → x = 1 := by
  sorry

end x_value_proof_l173_17327


namespace dubblefud_red_ball_value_l173_17379

/-- The value of a red ball in the game of Dubblefud -/
def red_ball_value : ℕ := sorry

/-- The value of a blue ball in the game of Dubblefud -/
def blue_ball_value : ℕ := 4

/-- The value of a green ball in the game of Dubblefud -/
def green_ball_value : ℕ := 5

/-- The number of red balls in the selection -/
def num_red_balls : ℕ := 4

/-- The number of blue balls in the selection -/
def num_blue_balls : ℕ := sorry

/-- The number of green balls in the selection -/
def num_green_balls : ℕ := sorry

theorem dubblefud_red_ball_value :
  (red_ball_value ^ num_red_balls) * 
  (blue_ball_value ^ num_blue_balls) * 
  (green_ball_value ^ num_green_balls) = 16000 ∧
  num_blue_balls = num_green_balls →
  red_ball_value = 1 :=
sorry

end dubblefud_red_ball_value_l173_17379


namespace vertex_angle_is_45_degrees_l173_17374

/-- An isosceles triangle with specific properties -/
structure SpecialIsoscelesTriangle where
  a : ℝ  -- Length of congruent sides
  s : ℝ  -- Semi-perimeter
  h : ℝ  -- Height to the base
  b : ℝ  -- Length of the base
  a_pos : 0 < a  -- Side length is positive
  s_pos : 0 < s  -- Semi-perimeter is positive
  h_pos : 0 < h  -- Height is positive
  b_pos : 0 < b  -- Base length is positive
  isosceles : s = a + b/2  -- Definition of semi-perimeter for this triangle
  right_base_angle : h = a  -- One base angle is a right angle
  area_condition : b * (2 * h) = s^2  -- Given condition

/-- The vertex angle at the base of the special isosceles triangle is 45° -/
theorem vertex_angle_is_45_degrees (t : SpecialIsoscelesTriangle) : 
  Real.arccos ((t.b / 2) / t.a) * (180 / Real.pi) = 45 := by
  sorry

end vertex_angle_is_45_degrees_l173_17374


namespace irrational_sqrt_N_l173_17377

def N (n : ℕ) : ℚ :=
  (10^n - 1) / 9 * 10^(2*n) + 4 * (10^(2*n) - 1) / 9

theorem irrational_sqrt_N (n : ℕ) (h : n > 1) :
  Irrational (Real.sqrt (N n)) :=
sorry

end irrational_sqrt_N_l173_17377


namespace visitor_decrease_l173_17315

theorem visitor_decrease (P V : ℝ) (h1 : P > 0) (h2 : V > 0) : 
  let R := P * V
  let P' := 1.5 * P
  let R' := 1.2 * R
  ∃ V', R' = P' * V' ∧ V' = 0.8 * V :=
by sorry

end visitor_decrease_l173_17315


namespace couscous_shipment_l173_17309

theorem couscous_shipment (first_shipment second_shipment num_dishes couscous_per_dish : ℕ)
  (h1 : first_shipment = 7)
  (h2 : second_shipment = 13)
  (h3 : num_dishes = 13)
  (h4 : couscous_per_dish = 5) :
  let total_used := num_dishes * couscous_per_dish
  let first_two_shipments := first_shipment + second_shipment
  total_used - first_two_shipments = 45 := by
    sorry

end couscous_shipment_l173_17309


namespace cars_given_to_sister_l173_17301

/- Define the problem parameters -/
def initial_cars : ℕ := 14
def bought_cars : ℕ := 28
def birthday_cars : ℕ := 12
def cars_to_vinnie : ℕ := 3
def cars_left : ℕ := 43

/- Define the theorem -/
theorem cars_given_to_sister :
  ∃ (cars_to_sister : ℕ),
    initial_cars + bought_cars + birthday_cars
    = cars_to_sister + cars_to_vinnie + cars_left ∧
    cars_to_sister = 8 := by
  sorry

end cars_given_to_sister_l173_17301


namespace radical_product_simplification_l173_17329

theorem radical_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (14 * q) = 14 * q * Real.sqrt (21 * q) :=
by sorry

end radical_product_simplification_l173_17329


namespace absolute_value_not_positive_l173_17359

theorem absolute_value_not_positive (y : ℚ) : |5 * y - 3| ≤ 0 ↔ y = 3/5 := by
  sorry

end absolute_value_not_positive_l173_17359


namespace atMostOneHead_exactlyTwoHeads_mutually_exclusive_l173_17388

/-- Represents the outcome of a single coin toss -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the outcome of tossing two coins simultaneously -/
def TwoCoinsOutcome := (CoinOutcome × CoinOutcome)

/-- The event of getting at most one head when tossing two coins -/
def atMostOneHead (outcome : TwoCoinsOutcome) : Prop :=
  match outcome with
  | (CoinOutcome.Tails, CoinOutcome.Tails) => True
  | (CoinOutcome.Heads, CoinOutcome.Tails) => True
  | (CoinOutcome.Tails, CoinOutcome.Heads) => True
  | (CoinOutcome.Heads, CoinOutcome.Heads) => False

/-- The event of getting exactly two heads when tossing two coins -/
def exactlyTwoHeads (outcome : TwoCoinsOutcome) : Prop :=
  match outcome with
  | (CoinOutcome.Heads, CoinOutcome.Heads) => True
  | _ => False

/-- Theorem stating that "at most one head" and "exactly two heads" are mutually exclusive -/
theorem atMostOneHead_exactlyTwoHeads_mutually_exclusive :
  ∀ (outcome : TwoCoinsOutcome), ¬(atMostOneHead outcome ∧ exactlyTwoHeads outcome) :=
by
  sorry


end atMostOneHead_exactlyTwoHeads_mutually_exclusive_l173_17388


namespace equal_integers_from_equation_l173_17311

/-- The least prime divisor of a positive integer greater than 1 -/
def least_prime_divisor (m : ℕ) : ℕ :=
  Nat.minFac m

theorem equal_integers_from_equation (a b : ℕ) 
  (ha : a > 1) (hb : b > 1)
  (h : a^2 + b = least_prime_divisor a + (least_prime_divisor b)^2) :
  a = b :=
sorry

end equal_integers_from_equation_l173_17311


namespace square_sum_xy_l173_17373

theorem square_sum_xy (x y : ℝ) 
  (h1 : x * (x + y) = 40)
  (h2 : y * (x + y) = 90)
  (h3 : x = (2/3) * y) : 
  (x + y)^2 = 100 := by
sorry

end square_sum_xy_l173_17373


namespace fraction_equality_implies_x_value_l173_17385

theorem fraction_equality_implies_x_value :
  ∀ x : ℚ, (x + 6) / (x - 4) = (x - 7) / (x + 2) → x = 16 / 19 := by
  sorry

end fraction_equality_implies_x_value_l173_17385


namespace lucas_siblings_product_l173_17308

/-- A family with Lauren and Lucas as members -/
structure Family where
  lauren_sisters : ℕ
  lauren_brothers : ℕ
  lucas : Member

/-- A member of the family -/
inductive Member
  | Lauren
  | Lucas
  | OtherSister
  | OtherBrother

/-- The number of sisters Lucas has in the family -/
def lucas_sisters (f : Family) : ℕ :=
  f.lauren_sisters + 1

/-- The number of brothers Lucas has in the family -/
def lucas_brothers (f : Family) : ℕ :=
  f.lauren_brothers - 1

theorem lucas_siblings_product (f : Family) 
  (h1 : f.lauren_sisters = 4)
  (h2 : f.lauren_brothers = 7)
  (h3 : f.lucas = Member.Lucas) :
  lucas_sisters f * lucas_brothers f = 35 := by
  sorry

end lucas_siblings_product_l173_17308
