import Mathlib

namespace fraction_simplification_l4070_407011

theorem fraction_simplification (x y : ℝ) (h : x ≠ y) :
  2 / (x + y) - (x - 3*y) / (x^2 - y^2) = 1 / (x - y) := by sorry

end fraction_simplification_l4070_407011


namespace fraction_equality_implies_k_l4070_407084

theorem fraction_equality_implies_k (x y z k : ℝ) :
  (9 / (x + y) = k / (x + z)) ∧ (k / (x + z) = 15 / (z - y)) →
  k = 24 := by
sorry

end fraction_equality_implies_k_l4070_407084


namespace min_toothpicks_theorem_l4070_407041

/-- Represents a triangular grid made of toothpicks -/
structure ToothpickGrid where
  total_toothpicks : ℕ
  upward_triangles : ℕ
  downward_triangles : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (grid : ToothpickGrid) : ℕ := sorry

/-- Theorem stating the minimum number of toothpicks to remove -/
theorem min_toothpicks_theorem (grid : ToothpickGrid) 
  (h1 : grid.total_toothpicks = 40)
  (h2 : grid.upward_triangles = 10)
  (h3 : grid.downward_triangles = 15) : 
  min_toothpicks_to_remove grid = 10 := by sorry

end min_toothpicks_theorem_l4070_407041


namespace complex_multiplication_l4070_407045

theorem complex_multiplication (i : ℂ) : i * i = -1 → 2 * i * (1 + i) = -2 + 2 * i := by
  sorry

end complex_multiplication_l4070_407045


namespace raffle_probabilities_l4070_407027

/-- Represents the raffle ticket distribution -/
structure RaffleTickets where
  total : ℕ
  first_prize : ℕ
  second_prize : ℕ
  third_prize : ℕ
  h_total : total = first_prize + second_prize + third_prize

/-- The probability of drawing exactly k tickets of a specific type from n tickets in m draws -/
def prob_draw (n k m : ℕ) : ℚ :=
  (n.choose k * (n - k).choose (m - k)) / n.choose m

theorem raffle_probabilities (r : RaffleTickets)
    (h1 : r.total = 10)
    (h2 : r.first_prize = 2)
    (h3 : r.second_prize = 3)
    (h4 : r.third_prize = 5) :
  /- (I) Probability of drawing 2 first prize tickets -/
  (prob_draw r.first_prize 2 2 = 1 / 45) ∧
  /- (II) Probability of drawing at most 1 first prize ticket in 3 draws -/
  (prob_draw r.first_prize 0 3 + prob_draw r.first_prize 1 3 = 14 / 15) ∧
  /- (III) Mathematical expectation of second prize tickets in 3 draws -/
  (0 * prob_draw r.second_prize 0 3 +
   1 * prob_draw r.second_prize 1 3 +
   2 * prob_draw r.second_prize 2 3 +
   3 * prob_draw r.second_prize 3 3 = 9 / 10) :=
by sorry

end raffle_probabilities_l4070_407027


namespace tileD_in_rectangleII_l4070_407001

-- Define the structure for a tile
structure Tile where
  top : Nat
  right : Nat
  bottom : Nat
  left : Nat

-- Define the tiles
def tileA : Tile := ⟨3, 5, 2, 0⟩
def tileB : Tile := ⟨2, 0, 5, 3⟩
def tileC : Tile := ⟨5, 3, 1, 2⟩
def tileD : Tile := ⟨0, 1, 3, 5⟩

-- Define a function to check if two tiles match on their adjacent sides
def matchTiles (t1 t2 : Tile) (side : Nat) : Prop :=
  match side with
  | 0 => t1.right = t2.left   -- Right of t1 matches Left of t2
  | 1 => t1.bottom = t2.top   -- Bottom of t1 matches Top of t2
  | 2 => t1.left = t2.right   -- Left of t1 matches Right of t2
  | 3 => t1.top = t2.bottom   -- Top of t1 matches Bottom of t2
  | _ => False

-- Theorem stating that Tile D must be in Rectangle II
theorem tileD_in_rectangleII : ∃ (t1 t2 t3 : Tile), 
  (t1 = tileA ∨ t1 = tileB ∨ t1 = tileC) ∧
  (t2 = tileA ∨ t2 = tileB ∨ t2 = tileC) ∧
  (t3 = tileA ∨ t3 = tileB ∨ t3 = tileC) ∧
  t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧
  matchTiles t1 tileD 0 ∧
  matchTiles tileD t2 0 ∧
  matchTiles t3 tileD 3 :=
by sorry

end tileD_in_rectangleII_l4070_407001


namespace incorrect_product_calculation_l4070_407098

theorem incorrect_product_calculation (x : ℕ) : 
  (53 * x - 35 * x = 540) → (53 * x = 1590) := by
  sorry

end incorrect_product_calculation_l4070_407098


namespace comic_collection_equality_l4070_407065

/-- Kymbrea's initial comic book collection --/
def kymbrea_initial : ℕ := 50

/-- Kymbrea's monthly comic book addition rate --/
def kymbrea_rate : ℕ := 3

/-- LaShawn's initial comic book collection --/
def lashawn_initial : ℕ := 20

/-- LaShawn's monthly comic book addition rate --/
def lashawn_rate : ℕ := 7

/-- The number of months after which LaShawn's collection will be greater than or equal to Kymbrea's --/
def months_until_equal : ℕ := 8

theorem comic_collection_equality :
  ∀ m : ℕ, m < months_until_equal →
    (lashawn_initial + lashawn_rate * m < kymbrea_initial + kymbrea_rate * m) ∧
    (lashawn_initial + lashawn_rate * months_until_equal ≥ kymbrea_initial + kymbrea_rate * months_until_equal) :=
by sorry

end comic_collection_equality_l4070_407065


namespace flower_pots_total_cost_l4070_407046

/-- The number of flower pots -/
def num_pots : ℕ := 6

/-- The price difference between consecutive pots -/
def price_diff : ℚ := 25 / 100

/-- The price of the largest pot -/
def largest_pot_price : ℚ := 1925 / 1000

/-- Calculate the total cost of all flower pots -/
def total_cost : ℚ :=
  let smallest_pot_price := largest_pot_price - (num_pots - 1 : ℕ) * price_diff
  (num_pots : ℚ) * smallest_pot_price + (num_pots - 1 : ℕ) * (num_pots : ℚ) * price_diff / 2

theorem flower_pots_total_cost :
  total_cost = 780 / 100 := by sorry

end flower_pots_total_cost_l4070_407046


namespace multiple_of_six_between_twelve_and_thirty_l4070_407031

theorem multiple_of_six_between_twelve_and_thirty (x : ℕ) :
  (∃ k : ℕ, x = 6 * k) →
  x^2 > 144 →
  x < 30 →
  x = 18 ∨ x = 24 := by
sorry

end multiple_of_six_between_twelve_and_thirty_l4070_407031


namespace max_students_distribution_l4070_407020

theorem max_students_distribution (pens toys : ℕ) (h1 : pens = 451) (h2 : toys = 410) :
  Nat.gcd pens toys = 41 :=
sorry

end max_students_distribution_l4070_407020


namespace power_equality_l4070_407078

theorem power_equality (x : ℝ) : (1/4 : ℝ) * (2^32) = 4^x → x = 15 := by
  sorry

end power_equality_l4070_407078


namespace union_of_A_and_B_l4070_407040

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x < -1 ∨ (2 ≤ x ∧ x < 3)}
def B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x < 4} := by
  sorry

end union_of_A_and_B_l4070_407040


namespace solution_value_l4070_407083

-- Define the function E
def E (a b c : ℚ) : ℚ := a * b^2 + c

-- State the theorem
theorem solution_value :
  ∃ (a : ℚ), E a 3 10 = E a 5 (-2) ∧ a = 3/4 := by sorry

end solution_value_l4070_407083


namespace midpoint_of_intersection_l4070_407092

-- Define the curve
def curve (t : ℝ) : ℝ × ℝ := (t + 1, (t - 1)^2)

-- Define the ray at θ = π/4
def ray (x : ℝ) : ℝ × ℝ := (x, x)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, curve t = p ∧ ray p.1 = p}

-- Theorem statement
theorem midpoint_of_intersection :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧
  (A.1 + B.1) / 2 = 2.5 ∧ (A.2 + B.2) / 2 = 2.5 :=
sorry

end midpoint_of_intersection_l4070_407092


namespace problem_statement_l4070_407095

theorem problem_statement :
  let p := ∀ x : ℝ, (2 : ℝ)^x < (3 : ℝ)^x
  let q := ∃ x : ℝ, x^2 = 2 - x
  (¬p ∧ q) → (∃ x : ℝ, x^2 = 2 - x ∧ x = -2) := by
  sorry

end problem_statement_l4070_407095


namespace number_problem_l4070_407029

theorem number_problem (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 10) : 
  (40/100) * N = 120 := by
sorry

end number_problem_l4070_407029


namespace jessica_cut_forty_roses_l4070_407090

/-- The number of roses Jessica cut from her garden -/
def roses_cut (initial_vase : ℕ) (final_vase : ℕ) (returned_to_sarah : ℕ) (total_garden : ℕ) : ℕ :=
  (final_vase - initial_vase) + returned_to_sarah

/-- Theorem stating that Jessica cut 40 roses from her garden -/
theorem jessica_cut_forty_roses :
  roses_cut 7 37 10 84 = 40 := by
  sorry

end jessica_cut_forty_roses_l4070_407090


namespace box_volume_l4070_407082

theorem box_volume (l w h : ℝ) 
  (side1 : l * w = 120)
  (side2 : w * h = 72)
  (top : l * h = 60) :
  l * w * h = 720 := by
  sorry

end box_volume_l4070_407082


namespace lines_perpendicular_l4070_407072

-- Define the slopes of the lines
def slope_l1 : ℚ := -2
def slope_l2 : ℚ := 1/2

-- Define the perpendicularity condition
def perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem lines_perpendicular : perpendicular slope_l1 slope_l2 := by
  sorry

end lines_perpendicular_l4070_407072


namespace system_solution_l4070_407057

theorem system_solution (x y : ℝ) : 
  ((x = 0 ∧ y = 0) ∨ 
   (x = 1 ∧ y = 1) ∨ 
   (x = -(5/4)^(1/5) ∧ y = (-50)^(1/5))) → 
  (4 * x^2 - 3 * y = x * y^3 ∧ 
   x^2 + x^3 * y^2 = 2 * y) := by
sorry

end system_solution_l4070_407057


namespace bug_total_distance_l4070_407088

def bug_path : List ℤ := [-3, 0, -8, 10]

def total_distance (path : List ℤ) : ℕ :=
  (path.zip (path.tail!)).foldl (fun acc (a, b) => acc + (a - b).natAbs) 0

theorem bug_total_distance :
  total_distance bug_path = 29 := by
  sorry

end bug_total_distance_l4070_407088


namespace hyperbola_properties_l4070_407048

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 12 - x^2 / 24 = 1

-- Define the foci
def foci : Set (ℝ × ℝ) :=
  {(0, 6), (0, -6)}

-- Define the asymptotes of the reference hyperbola
def reference_asymptotes (x y : ℝ) : Prop :=
  y = (Real.sqrt 2 / 2) * x ∨ y = -(Real.sqrt 2 / 2) * x

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y, hyperbola_equation x y → 
    (∃ f ∈ foci, (x - f.1)^2 + (y - f.2)^2 = 36)) ∧
  (∀ x y, hyperbola_equation x y → reference_asymptotes x y) :=
sorry

end hyperbola_properties_l4070_407048


namespace remainder_problem_l4070_407006

theorem remainder_problem : 29 * 169^1990 ≡ 7 [MOD 11] := by
  sorry

end remainder_problem_l4070_407006


namespace minimum_games_for_90_percent_win_rate_min_additional_games_is_25_l4070_407059

theorem minimum_games_for_90_percent_win_rate : ℕ → Prop :=
  fun n =>
    let initial_games : ℕ := 5
    let initial_eagles_wins : ℕ := 2
    let total_games : ℕ := initial_games + n
    let total_eagles_wins : ℕ := initial_eagles_wins + n
    (total_eagles_wins : ℚ) / (total_games : ℚ) ≥ 9/10 ∧
    ∀ m : ℕ, m < n → (initial_eagles_wins + m : ℚ) / (initial_games + m : ℚ) < 9/10

theorem min_additional_games_is_25 : 
  minimum_games_for_90_percent_win_rate 25 := by sorry

end minimum_games_for_90_percent_win_rate_min_additional_games_is_25_l4070_407059


namespace shopkeeper_profit_l4070_407071

theorem shopkeeper_profit (total_apples : ℝ) (profit_rate1 profit_rate2 : ℝ) 
  (portion1 portion2 : ℝ) :
  total_apples = 280 ∧ 
  profit_rate1 = 0.1 ∧ 
  profit_rate2 = 0.3 ∧ 
  portion1 = 0.4 ∧ 
  portion2 = 0.6 ∧ 
  portion1 + portion2 = 1 →
  let selling_price1 := portion1 * total_apples * (1 + profit_rate1)
  let selling_price2 := portion2 * total_apples * (1 + profit_rate2)
  let total_selling_price := selling_price1 + selling_price2
  let total_profit := total_selling_price - total_apples
  let percentage_profit := (total_profit / total_apples) * 100
  percentage_profit = 22 := by
sorry

end shopkeeper_profit_l4070_407071


namespace fibonacci_geometric_sequence_l4070_407055

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the theorem
theorem fibonacci_geometric_sequence (a b d : ℕ) :
  (∃ r : ℚ, r > 1 ∧ fib b = r * fib a ∧ fib d = r * fib b) →  -- Fₐ, Fᵦ, Fᵈ form an increasing geometric sequence
  a + b + d = 3000 →  -- Sum of indices is 3000
  b = a + 2 →  -- b - a = 2
  d = b + 2 →  -- d = b + 2
  a = 998 := by  -- Conclusion: a = 998
sorry

end fibonacci_geometric_sequence_l4070_407055


namespace four_numbers_puzzle_l4070_407026

theorem four_numbers_puzzle (a b c d : ℝ) : 
  a + b + c + d = 45 ∧ 
  a + 2 = b - 2 ∧ 
  a + 2 = 2 * c ∧ 
  a + 2 = d / 2 → 
  a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20 := by
sorry

end four_numbers_puzzle_l4070_407026


namespace first_class_students_l4070_407038

theorem first_class_students (avg_first : ℝ) (students_second : ℕ) (avg_second : ℝ) (avg_all : ℝ)
  (h1 : avg_first = 30)
  (h2 : students_second = 50)
  (h3 : avg_second = 60)
  (h4 : avg_all = 48.75) :
  ∃ students_first : ℕ,
    students_first * avg_first + students_second * avg_second =
    (students_first + students_second) * avg_all ∧
    students_first = 30 :=
by sorry

end first_class_students_l4070_407038


namespace basketball_practice_time_ratio_l4070_407044

theorem basketball_practice_time_ratio :
  ∀ (total_practice_time shooting_time weightlifting_time running_time : ℕ),
  total_practice_time = 120 →
  shooting_time = total_practice_time / 2 →
  weightlifting_time = 20 →
  running_time = total_practice_time - shooting_time - weightlifting_time →
  running_time / weightlifting_time = 2 := by
  sorry

end basketball_practice_time_ratio_l4070_407044


namespace unique_integer_perfect_square_l4070_407073

theorem unique_integer_perfect_square : 
  ∃! x : ℤ, ∃ y : ℤ, x^4 + 8*x^3 + 18*x^2 + 8*x + 36 = y^2 :=
by sorry

end unique_integer_perfect_square_l4070_407073


namespace unique_solution_to_system_l4070_407010

/-- The number of integer solutions to the system of equations:
    x^2 - 4xy + 3y^2 + z^2 = 45
    x^2 + 5yz - z^2 = -52
    -2x^2 + xy - 7z^2 = -101 -/
theorem unique_solution_to_system : 
  ∃! (x y z : ℤ), 
    x^2 - 4*x*y + 3*y^2 + z^2 = 45 ∧ 
    x^2 + 5*y*z - z^2 = -52 ∧ 
    -2*x^2 + x*y - 7*z^2 = -101 := by
  sorry

end unique_solution_to_system_l4070_407010


namespace f_sum_equals_two_l4070_407047

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x ^ 2) - 3 * x) + 1

theorem f_sum_equals_two :
  f (Real.log 2 / Real.log 10) + f (Real.log (1 / 2) / Real.log 10) = 2 := by sorry

end f_sum_equals_two_l4070_407047


namespace max_bowls_proof_l4070_407032

/-- Represents the number of clusters in a spoonful for the nth bowl -/
def clusters_per_spoon (n : ℕ) : ℕ := 3 + n

/-- Represents the number of spoonfuls in the nth bowl -/
def spoonfuls_per_bowl (n : ℕ) : ℕ := 27 - 2 * n

/-- Calculates the total clusters used up to and including the nth bowl -/
def total_clusters (n : ℕ) : ℕ := 
  (List.range n).foldl (λ acc i => acc + clusters_per_spoon (i + 1) * spoonfuls_per_bowl (i + 1)) 0

/-- The maximum number of bowls that can be made from 500 clusters -/
def max_bowls : ℕ := 4

theorem max_bowls_proof : 
  total_clusters max_bowls ≤ 500 ∧ 
  total_clusters (max_bowls + 1) > 500 := by
  sorry

#eval max_bowls

end max_bowls_proof_l4070_407032


namespace certain_number_is_900_l4070_407068

theorem certain_number_is_900 :
  ∃ x : ℝ, (45 * 9 = 0.45 * x) ∧ (x = 900) :=
by
  sorry

end certain_number_is_900_l4070_407068


namespace square_plus_reciprocal_square_l4070_407009

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 2) : x^2 + 1/x^2 = 2 := by
  sorry

end square_plus_reciprocal_square_l4070_407009


namespace all_female_finalists_probability_l4070_407033

-- Define the total number of participants
def total_participants : ℕ := 6

-- Define the number of female participants
def female_participants : ℕ := 4

-- Define the number of male participants
def male_participants : ℕ := 2

-- Define the number of finalists to be chosen
def finalists : ℕ := 3

-- Define the probability of selecting all female finalists
def prob_all_female_finalists : ℚ := (female_participants.choose finalists) / (total_participants.choose finalists)

-- Theorem statement
theorem all_female_finalists_probability :
  prob_all_female_finalists = 1 / 5 := by sorry

end all_female_finalists_probability_l4070_407033


namespace linear_function_not_in_quadrant_I_l4070_407018

/-- A linear function defined by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  yIntercept : ℝ

/-- Defines the four quadrants of the coordinate plane -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Checks if a point (x, y) is in a given quadrant -/
def inQuadrant (x y : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.I => x > 0 ∧ y > 0
  | Quadrant.II => x < 0 ∧ y > 0
  | Quadrant.III => x < 0 ∧ y < 0
  | Quadrant.IV => x > 0 ∧ y < 0

/-- Theorem: The graph of y = -2x - 1 does not pass through Quadrant I -/
theorem linear_function_not_in_quadrant_I :
  let f : LinearFunction := { slope := -2, yIntercept := -1 }
  ∀ x y : ℝ, y = f.slope * x + f.yIntercept → ¬(inQuadrant x y Quadrant.I) :=
by
  sorry


end linear_function_not_in_quadrant_I_l4070_407018


namespace stone_slab_length_l4070_407086

theorem stone_slab_length (total_area : Real) (num_slabs : Nat) (slab_length : Real) : 
  total_area = 58.8 ∧ 
  num_slabs = 30 ∧ 
  slab_length * slab_length * num_slabs = total_area * 10000 →
  slab_length = 140 := by
sorry

end stone_slab_length_l4070_407086


namespace modular_inverse_30_mod_31_l4070_407005

theorem modular_inverse_30_mod_31 : ∃ x : ℕ, x ≤ 31 ∧ (30 * x) % 31 = 1 :=
by
  -- The proof goes here
  sorry

end modular_inverse_30_mod_31_l4070_407005


namespace inverse_sum_mod_23_l4070_407067

theorem inverse_sum_mod_23 : 
  (((13⁻¹ : ZMod 23) + (17⁻¹ : ZMod 23) + (19⁻¹ : ZMod 23))⁻¹ : ZMod 23) = 8 := by sorry

end inverse_sum_mod_23_l4070_407067


namespace train_length_l4070_407080

/-- The length of a train given its speed, time to cross a platform, and the platform's length -/
theorem train_length (speed : ℝ) (time : ℝ) (platform_length : ℝ) : 
  speed = 72 → time = 25 → platform_length = 250.04 → 
  speed * (5/18) * time - platform_length = 249.96 := by
  sorry

#check train_length

end train_length_l4070_407080


namespace sqrt_difference_of_squares_l4070_407060

theorem sqrt_difference_of_squares : 
  (Real.sqrt 2023 + Real.sqrt 23) * (Real.sqrt 2023 - Real.sqrt 23) = 2000 := by
  sorry

end sqrt_difference_of_squares_l4070_407060


namespace sum_of_seven_squares_not_perfect_square_l4070_407012

theorem sum_of_seven_squares_not_perfect_square (n : ℤ) : 
  ¬∃ (m : ℤ), 7 * (n ^ 2 + 4) = m ^ 2 := by
  sorry

end sum_of_seven_squares_not_perfect_square_l4070_407012


namespace dress_original_price_l4070_407014

/-- The original price of a dress given shopping conditions --/
theorem dress_original_price (shoe_discount : ℚ) (dress_discount : ℚ) 
  (shoe_original_price : ℚ) (shoe_quantity : ℕ) (total_spent : ℚ) :
  shoe_discount = 40 / 100 →
  dress_discount = 20 / 100 →
  shoe_original_price = 50 →
  shoe_quantity = 2 →
  total_spent = 140 →
  ∃ (dress_original_price : ℚ),
    dress_original_price = 100 ∧
    total_spent = shoe_quantity * (shoe_original_price * (1 - shoe_discount)) +
                  dress_original_price * (1 - dress_discount) :=
by sorry

end dress_original_price_l4070_407014


namespace function_value_difference_bound_l4070_407081

theorem function_value_difference_bound
  (f : Set.Icc 0 1 → ℝ)
  (h₁ : f ⟨0, by norm_num⟩ = f ⟨1, by norm_num⟩)
  (h₂ : ∀ (x₁ x₂ : Set.Icc 0 1), x₁ ≠ x₂ → |f x₂ - f x₁| < |x₂.val - x₁.val|) :
  ∀ (x₁ x₂ : Set.Icc 0 1), |f x₂ - f x₁| < (1 : ℝ) / 2 := by
  sorry

end function_value_difference_bound_l4070_407081


namespace min_cars_with_racing_stripes_l4070_407013

theorem min_cars_with_racing_stripes 
  (total_cars : ℕ) 
  (cars_without_ac : ℕ) 
  (max_ac_no_stripes : ℕ) 
  (h1 : total_cars = 100)
  (h2 : cars_without_ac = 47)
  (h3 : max_ac_no_stripes = 47)
  (h4 : cars_without_ac ≤ total_cars)
  (h5 : max_ac_no_stripes ≤ total_cars - cars_without_ac) :
  ∃ (min_cars_with_stripes : ℕ), 
    min_cars_with_stripes = total_cars - cars_without_ac - max_ac_no_stripes ∧ 
    min_cars_with_stripes = 6 :=
by
  sorry

#check min_cars_with_racing_stripes

end min_cars_with_racing_stripes_l4070_407013


namespace expression_evaluation_l4070_407077

theorem expression_evaluation (y : ℝ) (h : y = -3) : 
  (5 + y * (4 + y) - 4^2) / (y - 2 + y^2) = -3.5 := by
  sorry

end expression_evaluation_l4070_407077


namespace unique_real_solution_l4070_407074

theorem unique_real_solution (a : ℝ) :
  (∃! x : ℝ, x^3 - a*x^2 - 2*a*x + a^2 - 1 = 0) ↔ a < 3/4 := by
  sorry

end unique_real_solution_l4070_407074


namespace complex_equality_condition_l4070_407008

theorem complex_equality_condition :
  ∃ (x y : ℂ), x + y * Complex.I = 1 + Complex.I ∧ (x ≠ 1 ∨ y ≠ 1) :=
by sorry

end complex_equality_condition_l4070_407008


namespace g_4_equals_7_5_l4070_407000

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def g (x : ℝ) : ℝ := 1 / (f⁻¹ x) + 7

theorem g_4_equals_7_5 : g 4 = 7.5 := by sorry

end g_4_equals_7_5_l4070_407000


namespace weights_division_l4070_407036

theorem weights_division (n : ℕ) : 
  (∃ (a b c : ℕ), a + b + c = n * (n + 1) / 2 ∧ a = b ∧ b = c) ↔ 
  (n > 3 ∧ (n % 3 = 0 ∨ n % 3 = 2)) :=
sorry

end weights_division_l4070_407036


namespace meeting_participants_l4070_407024

theorem meeting_participants :
  ∀ (F M : ℕ),
  F > 0 →
  M > 0 →
  F / 2 = 130 →
  F / 2 + M / 4 = (F + M) / 3 →
  F + M = 780 :=
by
  sorry

end meeting_participants_l4070_407024


namespace dividend_calculation_l4070_407037

/-- Calculates the total dividend paid to a shareholder --/
def total_dividend (expected_earnings : ℚ) (actual_earnings : ℚ) (base_dividend_ratio : ℚ) 
  (additional_dividend_rate : ℚ) (additional_earnings_threshold : ℚ) (num_shares : ℕ) : ℚ :=
  let base_dividend := expected_earnings * base_dividend_ratio
  let earnings_difference := actual_earnings - expected_earnings
  let additional_dividend := 
    if earnings_difference > 0 
    then (earnings_difference / additional_earnings_threshold).floor * additional_dividend_rate
    else 0
  let total_dividend_per_share := base_dividend + additional_dividend
  total_dividend_per_share * num_shares

/-- Theorem stating the total dividend paid to a shareholder with given conditions --/
theorem dividend_calculation : 
  total_dividend 0.80 1.10 (1/2) 0.04 0.10 100 = 52 := by
  sorry

#eval total_dividend 0.80 1.10 (1/2) 0.04 0.10 100

end dividend_calculation_l4070_407037


namespace divide_by_eight_l4070_407019

theorem divide_by_eight (x y z : ℕ) (h1 : x > 0) (h2 : x = 11 * y + 4) 
  (h3 : 2 * x = 3 * y * z + 3) (h4 : 13 * y - x = 1) : z = 8 := by
  sorry

end divide_by_eight_l4070_407019


namespace repeating_decimal_56_l4070_407035

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

theorem repeating_decimal_56 :
  RepeatingDecimal 5 6 = 56 / 99 := by sorry

end repeating_decimal_56_l4070_407035


namespace right_triangle_ratio_l4070_407070

theorem right_triangle_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Right triangle condition
  a / b = 2 / 5 →    -- Given ratio of a to b
  r * c = a^2 →      -- Geometric mean theorem for r
  s * c = b^2 →      -- Geometric mean theorem for s
  r + s = c →        -- r and s are segments of c
  r / s = 4 / 25 :=  -- Conclusion to prove
by sorry

end right_triangle_ratio_l4070_407070


namespace quadratic_value_at_3_l4070_407053

/-- A quadratic function y = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  min_value : ℝ
  min_x : ℝ
  point_x : ℝ
  point_y : ℝ

/-- Properties of the quadratic function -/
def has_properties (f : QuadraticFunction) : Prop :=
  f.min_value = -8 ∧
  f.min_x = -2 ∧
  f.point_x = 1 ∧
  f.point_y = 5 ∧
  f.point_y = f.a * f.point_x^2 + f.b * f.point_x + f.c

/-- The value of y when x = 3 -/
def y_at_3 (f : QuadraticFunction) : ℝ :=
  f.a * 3^2 + f.b * 3 + f.c

/-- The main theorem -/
theorem quadratic_value_at_3 (f : QuadraticFunction) (h : has_properties f) :
  y_at_3 f = 253/9 :=
sorry

end quadratic_value_at_3_l4070_407053


namespace football_player_goals_l4070_407063

/-- Proves that a football player scored 2 goals in their fifth match -/
theorem football_player_goals (total_matches : ℕ) (total_goals : ℕ) (average_increase : ℚ) : 
  total_matches = 5 → 
  total_goals = 4 → 
  average_increase = 3/10 → 
  (total_goals : ℚ) / total_matches = 
    ((total_goals : ℚ) - (total_goals - goals_in_fifth_match)) / (total_matches - 1) + average_increase →
  goals_in_fifth_match = 2 :=
by
  sorry

#check football_player_goals

end football_player_goals_l4070_407063


namespace point_coordinates_l4070_407007

theorem point_coordinates (M N P : ℝ × ℝ) : 
  M = (3, -2) → 
  N = (-5, -1) → 
  P - M = (1/2 : ℝ) • (N - M) → 
  P = (-1, -3/2) := by
  sorry

end point_coordinates_l4070_407007


namespace complex_modulus_example_l4070_407002

theorem complex_modulus_example : Complex.abs (-3 + (9/4)*Complex.I) = 15/4 := by
  sorry

end complex_modulus_example_l4070_407002


namespace accident_calculation_highway_accidents_l4070_407064

/-- Given an accident rate and total number of vehicles, calculate the number of vehicles involved in accidents --/
theorem accident_calculation (accident_rate : ℕ) (vehicles_per_set : ℕ) (total_vehicles : ℕ) :
  accident_rate > 0 →
  vehicles_per_set > 0 →
  total_vehicles ≥ vehicles_per_set →
  (total_vehicles / vehicles_per_set) * accident_rate = 
    (total_vehicles * accident_rate) / vehicles_per_set :=
by
  sorry

/-- Calculate the number of vehicles involved in accidents on a highway --/
theorem highway_accidents :
  let accident_rate := 80  -- vehicles involved in accidents per set
  let vehicles_per_set := 100000000  -- vehicles per set (100 million)
  let total_vehicles := 4000000000  -- total vehicles (4 billion)
  (total_vehicles / vehicles_per_set) * accident_rate = 3200 :=
by
  sorry

end accident_calculation_highway_accidents_l4070_407064


namespace probability_red_or_white_l4070_407025

def total_marbles : ℕ := 20
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

theorem probability_red_or_white :
  (red_marbles + white_marbles : ℚ) / total_marbles = 3 / 4 := by
  sorry

end probability_red_or_white_l4070_407025


namespace standard_deviation_is_eight_l4070_407016

/-- Represents the age distribution of job applicants -/
structure AgeDistribution where
  average_age : ℕ
  num_different_ages : ℕ
  standard_deviation : ℕ

/-- Checks if the age distribution satisfies the given conditions -/
def is_valid_distribution (d : AgeDistribution) : Prop :=
  d.average_age = 30 ∧
  d.num_different_ages = 17 ∧
  d.num_different_ages = 2 * d.standard_deviation + 1

/-- Theorem stating that the standard deviation must be 8 given the conditions -/
theorem standard_deviation_is_eight (d : AgeDistribution) 
  (h : is_valid_distribution d) : d.standard_deviation = 8 := by
  sorry

#check standard_deviation_is_eight

end standard_deviation_is_eight_l4070_407016


namespace integer_triangle_properties_l4070_407089

/-- A triangle with positive integer side lengths and circumradius -/
structure IntegerTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  R : ℕ+

/-- Properties of an integer triangle -/
theorem integer_triangle_properties (T : IntegerTriangle) :
  ∃ (r : ℕ+) (P : ℕ),
    (∃ (k : ℕ), P = 4 * k) ∧
    (∃ (m n l : ℕ), T.a = 2 * m ∧ T.b = 2 * n ∧ T.c = 2 * l) := by
  sorry


end integer_triangle_properties_l4070_407089


namespace max_factors_of_b_power_n_l4070_407028

def is_prime (p : ℕ) : Prop := sorry

-- Function to count factors of a number
def count_factors (n : ℕ) : ℕ := sorry

-- Function to check if a number is the product of exactly two distinct primes less than 15
def is_product_of_two_primes_less_than_15 (b : ℕ) : Prop := sorry

theorem max_factors_of_b_power_n :
  ∃ (b n : ℕ),
    b ≤ 15 ∧
    n ≤ 15 ∧
    is_product_of_two_primes_less_than_15 b ∧
    count_factors (b^n) = 256 ∧
    ∀ (b' n' : ℕ),
      b' ≤ 15 →
      n' ≤ 15 →
      is_product_of_two_primes_less_than_15 b' →
      count_factors (b'^n') ≤ 256 := by
  sorry

end max_factors_of_b_power_n_l4070_407028


namespace envelope_length_l4070_407017

/-- Given a rectangular envelope with width 4 inches and area 16 square inches,
    prove that its length is 4 inches. -/
theorem envelope_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 4 → area = 16 → area = width * length → length = 4 := by
  sorry

end envelope_length_l4070_407017


namespace neither_sufficient_nor_necessary_l4070_407091

theorem neither_sufficient_nor_necessary :
  ¬(∀ x y : ℝ, (x > 1 ∧ y > 1) → (x + y > 3)) ∧
  ¬(∀ x y : ℝ, (x + y > 3) → (x > 1 ∧ y > 1)) := by
  sorry

end neither_sufficient_nor_necessary_l4070_407091


namespace dunk_a_clown_tickets_l4070_407075

/-- Proves the number of tickets spent at the 'dunk a clown' booth -/
theorem dunk_a_clown_tickets (total_tickets : ℕ) (num_rides : ℕ) (tickets_per_ride : ℕ) :
  total_tickets - (num_rides * tickets_per_ride) =
  total_tickets - num_rides * tickets_per_ride :=
by sorry

/-- Calculates the number of tickets spent at the 'dunk a clown' booth -/
def tickets_at_dunk_a_clown (total_tickets : ℕ) (num_rides : ℕ) (tickets_per_ride : ℕ) : ℕ :=
  total_tickets - (num_rides * tickets_per_ride)

#eval tickets_at_dunk_a_clown 79 8 7

end dunk_a_clown_tickets_l4070_407075


namespace stock_income_theorem_l4070_407061

/-- Calculates the income from a stock investment given the rate, market value, and investment amount. -/
def calculate_income (rate : ℚ) (market_value : ℚ) (investment : ℚ) : ℚ :=
  (rate / 100) * (investment / market_value) * 100

/-- Theorem stating that given the specific conditions, the income is 650. -/
theorem stock_income_theorem (rate market_value investment : ℚ) 
  (h_rate : rate = 10)
  (h_market_value : market_value = 96)
  (h_investment : investment = 6240) :
  calculate_income rate market_value investment = 650 :=
by
  sorry

#eval calculate_income 10 96 6240

end stock_income_theorem_l4070_407061


namespace diophantine_equation_solutions_l4070_407097

theorem diophantine_equation_solutions :
  {(x, y) : ℕ × ℕ | 3 * x + 2 * y = 21 ∧ x > 0 ∧ y > 0} =
  {(5, 3), (3, 6), (1, 9)} := by
sorry

end diophantine_equation_solutions_l4070_407097


namespace square_triangle_equal_area_l4070_407039

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (triangle_base : ℝ) :
  square_perimeter = 48 →
  triangle_height = 36 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_base * triangle_height →
  triangle_base = 8 := by
  sorry

end square_triangle_equal_area_l4070_407039


namespace sqrt_300_simplification_l4070_407066

theorem sqrt_300_simplification : Real.sqrt 300 = 10 * Real.sqrt 3 := by
  sorry

end sqrt_300_simplification_l4070_407066


namespace geometric_series_sum_l4070_407003

theorem geometric_series_sum (y : ℚ) : y = 23 / 13 ↔ 
  (∑' n, (1 / 3 : ℚ) ^ n) + (∑' n, (-1/4 : ℚ) ^ n) = ∑' n, (1 / y : ℚ) ^ n :=
by sorry

end geometric_series_sum_l4070_407003


namespace equation_solutions_l4070_407079

theorem equation_solutions :
  (∃ x : ℝ, 2 * (x + 3) = 5 * x ∧ x = 2) ∧
  (∃ x : ℝ, (x - 3) / 0.5 - (x + 4) / 0.2 = 1.6 ∧ x = -9.2) := by
  sorry

end equation_solutions_l4070_407079


namespace abc_fraction_theorem_l4070_407076

theorem abc_fraction_theorem (a b c : ℕ+) :
  ∃ (n : ℕ), n > 0 ∧ n = (a * b * c + a * b + a) / (a * b * c + b * c + c) → n = 1 ∨ n = 2 := by
  sorry

end abc_fraction_theorem_l4070_407076


namespace vehicle_y_speed_l4070_407052

/-- Proves that the average speed of vehicle Y is 45 miles per hour given the problem conditions -/
theorem vehicle_y_speed
  (initial_distance : ℝ)
  (vehicle_x_speed : ℝ)
  (overtake_time : ℝ)
  (final_lead : ℝ)
  (h1 : initial_distance = 22)
  (h2 : vehicle_x_speed = 36)
  (h3 : overtake_time = 5)
  (h4 : final_lead = 23) :
  (initial_distance + final_lead + vehicle_x_speed * overtake_time) / overtake_time = 45 :=
by
  sorry

end vehicle_y_speed_l4070_407052


namespace nearest_integer_to_3_plus_sqrt5_fourth_power_l4070_407043

theorem nearest_integer_to_3_plus_sqrt5_fourth_power :
  ∃ n : ℤ, n = 752 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 5)^4 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5)^4 - (m : ℝ)| :=
by sorry

end nearest_integer_to_3_plus_sqrt5_fourth_power_l4070_407043


namespace wage_ratio_is_two_to_one_l4070_407004

/-- The ratio of a man's daily wage to a woman's daily wage -/
def wage_ratio (men_wage women_wage : ℚ) : ℚ := men_wage / women_wage

/-- The total earnings of a group of workers over a period of time -/
def total_earnings (num_workers : ℕ) (days : ℕ) (daily_wage : ℚ) : ℚ :=
  (num_workers : ℚ) * (days : ℚ) * daily_wage

theorem wage_ratio_is_two_to_one 
  (men_wage women_wage : ℚ)
  (h1 : total_earnings 16 25 men_wage = 14400)
  (h2 : total_earnings 40 30 women_wage = 21600) :
  wage_ratio men_wage women_wage = 2 := by
  sorry

#eval wage_ratio 36 18  -- Expected output: 2

end wage_ratio_is_two_to_one_l4070_407004


namespace paula_karl_age_sum_l4070_407056

theorem paula_karl_age_sum : ∀ (P K : ℕ),
  (P - 5 = 3 * (K - 5)) →
  (P + 6 = 2 * (K + 6)) →
  P + K = 54 := by
  sorry

end paula_karl_age_sum_l4070_407056


namespace smallest_prime_factor_in_C_l4070_407034

def C : Set Nat := {65, 67, 68, 71, 74}

def has_smallest_prime_factor (n : Nat) (s : Set Nat) : Prop :=
  n ∈ s ∧ ∀ m ∈ s, ∀ p q : Nat, Prime p → Prime q → p ∣ n → q ∣ m → p ≤ q

theorem smallest_prime_factor_in_C :
  has_smallest_prime_factor 68 C := by sorry

end smallest_prime_factor_in_C_l4070_407034


namespace loan_amount_proof_l4070_407042

/-- Represents a simple interest loan -/
structure SimpleLoan where
  principal : ℝ
  rate : ℝ
  time : ℝ
  interest : ℝ

/-- The loan satisfies the given conditions -/
def loan_conditions (loan : SimpleLoan) : Prop :=
  loan.rate = 0.06 ∧
  loan.time = loan.rate ∧
  loan.interest = 432 ∧
  loan.interest = loan.principal * loan.rate * loan.time

theorem loan_amount_proof (loan : SimpleLoan) 
  (h : loan_conditions loan) : loan.principal = 1200 := by
  sorry

#check loan_amount_proof

end loan_amount_proof_l4070_407042


namespace circles_externally_tangent_m_value_l4070_407030

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

def circle_C2 (x y m : ℝ) : Prop := x^2 + y^2 - 8*x - 10*y + m + 6 = 0

-- Define external tangency
def externally_tangent (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), C1 x y ∧ C2 x y ∧
  ∀ (x' y' : ℝ), (C1 x' y' ∧ C2 x' y') → (x' = x ∧ y' = y)

-- Theorem statement
theorem circles_externally_tangent_m_value :
  externally_tangent circle_C1 (circle_C2 · · 26) :=
sorry

end circles_externally_tangent_m_value_l4070_407030


namespace nine_to_ten_div_eightyone_to_four_equals_eightyone_l4070_407093

theorem nine_to_ten_div_eightyone_to_four_equals_eightyone :
  9^10 / 81^4 = 81 := by
  sorry

end nine_to_ten_div_eightyone_to_four_equals_eightyone_l4070_407093


namespace vector_to_line_parallel_and_intersecting_l4070_407094

/-- A line parameterized by t -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

/-- A point lies on a parametric line if there exists a t satisfying both equations -/
def lies_on (p : ℝ × ℝ) (l : ParametricLine) : Prop :=
  ∃ t : ℝ, l.x t = p.1 ∧ l.y t = p.2

/-- The main theorem -/
theorem vector_to_line_parallel_and_intersecting :
  let l : ParametricLine := { x := λ t => 5 * t + 1, y := λ t => 2 * t + 1 }
  let v : ℝ × ℝ := (12.5, 5)
  let w : ℝ × ℝ := (5, 2)
  parallel v w ∧ lies_on v l := by sorry

end vector_to_line_parallel_and_intersecting_l4070_407094


namespace chord_intersects_diameter_l4070_407015

/-- In a circle with radius 6, a chord of length 10 intersects a diameter,
    dividing it into segments of lengths 6 - √11 and 6 + √11 -/
theorem chord_intersects_diameter (r : ℝ) (chord_length : ℝ) 
  (h1 : r = 6) (h2 : chord_length = 10) : 
  ∃ (s1 s2 : ℝ), s1 = 6 - Real.sqrt 11 ∧ s2 = 6 + Real.sqrt 11 ∧ s1 + s2 = 2 * r :=
sorry

end chord_intersects_diameter_l4070_407015


namespace hexagonal_grid_path_theorem_l4070_407049

/-- Represents a point in the hexagonal grid -/
structure HexPoint where
  x : ℤ
  y : ℤ

/-- Represents a direction in the hexagonal grid -/
inductive HexDirection
  | Right
  | UpRight
  | UpLeft
  | Left
  | DownLeft
  | DownRight

/-- Represents a path in the hexagonal grid -/
def HexPath := List (HexPoint × HexDirection)

/-- Function to calculate the length of a path -/
def pathLength (path : HexPath) : ℕ := path.length

/-- Function to check if a path is valid in the hexagonal grid -/
def isValidPath (path : HexPath) : Prop := sorry

/-- Function to find the longest continuous segment in the same direction -/
def longestContinuousSegment (path : HexPath) : ℕ := sorry

/-- Theorem: In a hexagonal grid, if the shortest path between two points is 20 units,
    then there exists a continuous segment of at least 10 units in the same direction -/
theorem hexagonal_grid_path_theorem (A B : HexPoint) (path : HexPath) :
  isValidPath path →
  pathLength path = 20 →
  (∀ p : HexPath, isValidPath p → pathLength p ≥ 20) →
  longestContinuousSegment path ≥ 10 := by
  sorry

end hexagonal_grid_path_theorem_l4070_407049


namespace oakwood_academy_walking_students_l4070_407050

theorem oakwood_academy_walking_students (total : ℚ) :
  let bus : ℚ := 1 / 3
  let car : ℚ := 1 / 5
  let cycle : ℚ := 1 / 8
  let walk : ℚ := total - (bus + car + cycle)
  walk = 41 / 120 := by
  sorry

end oakwood_academy_walking_students_l4070_407050


namespace square_area_diagonal_relation_l4070_407023

theorem square_area_diagonal_relation (d : ℝ) (h : d > 0) :
  ∃ (A : ℝ), A > 0 ∧ A = (1/2) * d^2 ∧ 
  (∃ (s : ℝ), s > 0 ∧ A = s^2 ∧ d^2 = 2 * s^2) := by
  sorry

end square_area_diagonal_relation_l4070_407023


namespace reflection_sum_l4070_407021

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line of reflection y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Checks if two points are reflections of each other across a given line -/
def areReflections (A B : Point) (L : Line) : Prop :=
  let midpoint : Point := ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩
  (midpoint.y = L.m * midpoint.x + L.b) ∧
  (L.m = -(B.x - A.x) / (B.y - A.y))

/-- The main theorem -/
theorem reflection_sum (A B : Point) (L : Line) :
  A = ⟨2, 3⟩ → B = ⟨10, 7⟩ → areReflections A B L → L.m + L.b = 15 := by
  sorry


end reflection_sum_l4070_407021


namespace collinear_points_right_triangle_l4070_407051

/-- Given that point O is the origin, this function defines vectors OA, OB, and OC -/
def vectors (m : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  ((3, -4), (6, -3), (5 - m, -3 - m))

/-- Theorem stating that if A, B, and C are collinear, then m = 1/2 -/
theorem collinear_points (m : ℝ) :
  let (oa, ob, oc) := vectors m
  (∃ (k : ℝ), (ob.1 - oa.1, ob.2 - oa.2) = k • (oc.1 - oa.1, oc.2 - oa.2)) →
  m = 1/2 := by sorry

/-- Theorem stating that if ABC is a right triangle with A as the right angle, then m = 7/4 -/
theorem right_triangle (m : ℝ) :
  let (oa, ob, oc) := vectors m
  let ab := (ob.1 - oa.1, ob.2 - oa.2)
  let ac := (oc.1 - oa.1, oc.2 - oa.2)
  (ab.1 * ac.1 + ab.2 * ac.2 = 0) →
  m = 7/4 := by sorry

end collinear_points_right_triangle_l4070_407051


namespace pizza_slices_l4070_407096

-- Define the number of slices in each pizza
def slices_per_pizza : ℕ := sorry

-- Define the total number of pizzas
def total_pizzas : ℕ := 2

-- Define the fractions eaten by each person
def bob_fraction : ℚ := 1/2
def tom_fraction : ℚ := 1/3
def sally_fraction : ℚ := 1/6
def jerry_fraction : ℚ := 1/4

-- Define the number of slices left over
def slices_left : ℕ := 9

theorem pizza_slices : 
  slices_per_pizza = 12 ∧
  (bob_fraction + tom_fraction + sally_fraction + jerry_fraction) * slices_per_pizza * total_pizzas = 
    slices_per_pizza * total_pizzas - slices_left :=
by sorry

end pizza_slices_l4070_407096


namespace find_number_l4070_407099

theorem find_number : ∃! x : ℝ, ((35 - x) * 2 + 12) / 8 = 9 := by sorry

end find_number_l4070_407099


namespace four_digit_sum_gcd_quotient_l4070_407087

theorem four_digit_sum_gcd_quotient
  (a b c d : Nat)
  (h1 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10)
  (h2 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h3 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) :
  let S := a + b + c + d
  let G := Nat.gcd a (Nat.gcd b (Nat.gcd c d))
  (33 * S - S * G) / S = 33 - G :=
by sorry

end four_digit_sum_gcd_quotient_l4070_407087


namespace quadratic_max_value_l4070_407058

def f (x : ℝ) : ℝ := -x^2 - 3*x + 4

theorem quadratic_max_value :
  ∃ (max : ℝ), max = 25/4 ∧ ∀ (x : ℝ), f x ≤ max :=
sorry

end quadratic_max_value_l4070_407058


namespace vector_subtraction_l4070_407062

/-- Given two vectors a and b in ℝ², prove that their difference is (1, 2). -/
theorem vector_subtraction (a b : ℝ × ℝ) (ha : a = (2, 3)) (hb : b = (1, 1)) :
  a - b = (1, 2) := by sorry

end vector_subtraction_l4070_407062


namespace total_canoes_by_april_l4070_407069

def canoes_per_month (month : Nat) : Nat :=
  match month with
  | 0 => 4  -- January (0-indexed)
  | n + 1 => 3 * canoes_per_month n

theorem total_canoes_by_april : 
  (canoes_per_month 0) + (canoes_per_month 1) + (canoes_per_month 2) + (canoes_per_month 3) = 160 := by
  sorry

end total_canoes_by_april_l4070_407069


namespace log_sum_of_zeros_gt_two_l4070_407022

open Real

/-- Given a function g(x) = ln x - bx, if it has two distinct positive zeros,
    then the sum of their natural logarithms is greater than 2. -/
theorem log_sum_of_zeros_gt_two (b : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : x₁ ≠ x₂)
  (hz₁ : log x₁ - b * x₁ = 0) (hz₂ : log x₂ - b * x₂ = 0) :
  log x₁ + log x₂ > 2 := by
sorry


end log_sum_of_zeros_gt_two_l4070_407022


namespace arithmetic_geometric_ratio_l4070_407085

-- Define the arithmetic sequence
def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := sorry

-- State the theorem
theorem arithmetic_geometric_ratio (d : ℝ) :
  d ≠ 0 →
  (∃ r : ℝ, arithmetic_sequence d 3 = arithmetic_sequence d 1 * r ∧ 
            arithmetic_sequence d 4 = arithmetic_sequence d 3 * r) →
  (arithmetic_sequence d 1 + arithmetic_sequence d 5 + arithmetic_sequence d 17) / 
  (arithmetic_sequence d 2 + arithmetic_sequence d 6 + arithmetic_sequence d 18) = 8 / 11 :=
by sorry

end arithmetic_geometric_ratio_l4070_407085


namespace parallel_slope_relation_l4070_407054

-- Define a structure for a line with a slope
structure Line where
  slope : ℝ

-- Define parallel relation for lines
def parallel (l₁ l₂ : Line) : Prop := sorry

theorem parallel_slope_relation :
  ∀ (l₁ l₂ : Line),
    (parallel l₁ l₂ → l₁.slope = l₂.slope) ∧
    ∃ (l₃ l₄ : Line), l₃.slope = l₄.slope ∧ ¬parallel l₃ l₄ := by
  sorry

end parallel_slope_relation_l4070_407054
