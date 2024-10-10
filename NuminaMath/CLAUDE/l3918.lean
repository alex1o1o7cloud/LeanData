import Mathlib

namespace length_of_AB_l3918_391822

-- Define the points
variable (A B C D E F G : ℝ)

-- Define the conditions
variable (h1 : C = (A + B) / 2)
variable (h2 : D = (A + C) / 2)
variable (h3 : E = (A + D) / 2)
variable (h4 : F = (A + E) / 2)
variable (h5 : G = (A + F) / 2)
variable (h6 : G - A = 1)

-- State the theorem
theorem length_of_AB : B - A = 32 := by sorry

end length_of_AB_l3918_391822


namespace graph_connectivity_probability_l3918_391830

/-- A complete graph with 20 vertices -/
def complete_graph : Nat := 20

/-- Number of edges removed -/
def removed_edges : Nat := 35

/-- Total number of edges in the complete graph -/
def total_edges : Nat := complete_graph * (complete_graph - 1) / 2

/-- Number of edges remaining after removal -/
def remaining_edges : Nat := total_edges - removed_edges

/-- Probability that the graph remains connected after edge removal -/
def connected_probability : ℚ :=
  1 - (complete_graph * (Nat.choose (total_edges - complete_graph + 1) (removed_edges - complete_graph + 1))) / 
      (Nat.choose total_edges removed_edges)

theorem graph_connectivity_probability :
  connected_probability = 1 - (20 * (Nat.choose 171 16)) / (Nat.choose 190 35) :=
sorry

end graph_connectivity_probability_l3918_391830


namespace remaining_distance_to_hotel_l3918_391807

/-- Calculates the remaining distance to the hotel given the initial conditions of Samuel's journey --/
theorem remaining_distance_to_hotel (total_distance : ℝ) (initial_speed : ℝ) (initial_time : ℝ) (second_speed : ℝ) (second_time : ℝ) :
  total_distance = 600 ∧
  initial_speed = 50 ∧
  initial_time = 3 ∧
  second_speed = 80 ∧
  second_time = 4 →
  total_distance - (initial_speed * initial_time + second_speed * second_time) = 130 := by
  sorry

#check remaining_distance_to_hotel

end remaining_distance_to_hotel_l3918_391807


namespace rectangle_longer_side_length_l3918_391823

/-- Given a rectangle formed from a rope of length 100 cm with shorter sides of 22 cm each,
    prove that the length of each longer side is 28 cm. -/
theorem rectangle_longer_side_length (total_length : ℝ) (short_side : ℝ) (long_side : ℝ) :
  total_length = 100 ∧ short_side = 22 →
  2 * short_side + 2 * long_side = total_length →
  long_side = 28 := by
  sorry

end rectangle_longer_side_length_l3918_391823


namespace AF_AT_ratio_l3918_391803

-- Define the triangle ABC and points D, E, F, T
variable (A B C D E F T : ℝ × ℝ)

-- Define the conditions
axiom on_AB : ∃ t : ℝ, D = (1 - t) • A + t • B ∧ 0 ≤ t ∧ t ≤ 1
axiom on_AC : ∃ s : ℝ, E = (1 - s) • A + s • C ∧ 0 ≤ s ∧ s ≤ 1
axiom on_DE : ∃ r : ℝ, F = (1 - r) • D + r • E ∧ 0 ≤ r ∧ r ≤ 1
axiom on_AT : ∃ q : ℝ, F = (1 - q) • A + q • T ∧ 0 ≤ q ∧ q ≤ 1

axiom AD_length : dist A D = 1
axiom DB_length : dist D B = 4
axiom AE_length : dist A E = 3
axiom EC_length : dist E C = 3

axiom angle_bisector : 
  dist B T / dist T C = dist A B / dist A C

-- Define the theorem to be proved
theorem AF_AT_ratio : 
  dist A F / dist A T = 11 / 40 :=
sorry

end AF_AT_ratio_l3918_391803


namespace shaded_region_perimeter_l3918_391834

/-- Given a circle with center O and radius r, where arc PQ is half the circle,
    the perimeter of the shaded region formed by OP, OQ, and arc PQ is 2r + πr. -/
theorem shaded_region_perimeter (r : ℝ) (h : r > 0) : 
  2 * r + π * r = 2 * r + (1 / 2) * (2 * π * r) := by sorry

end shaded_region_perimeter_l3918_391834


namespace frustum_cone_height_l3918_391899

theorem frustum_cone_height (h : ℝ) (a_lower a_upper : ℝ) 
  (h_positive : h > 0)
  (a_lower_positive : a_lower > 0)
  (a_upper_positive : a_upper > 0)
  (h_value : h = 30)
  (a_lower_value : a_lower = 400 * Real.pi)
  (a_upper_value : a_upper = 100 * Real.pi) :
  let r_lower := (a_lower / Real.pi).sqrt
  let r_upper := (a_upper / Real.pi).sqrt
  let h_total := h * r_lower / (r_lower - r_upper)
  h_total / 3 = 15 := by sorry

end frustum_cone_height_l3918_391899


namespace square25_on_top_l3918_391894

/-- Represents a position on the 5x5 grid -/
structure Position :=
  (row : Fin 5)
  (col : Fin 5)

/-- Represents the state of the grid after folding -/
structure FoldedGrid :=
  (top : Position)

/-- Fold operation 1: fold the top half over the bottom half -/
def fold1 (p : Position) : Position :=
  ⟨4 - p.row, p.col⟩

/-- Fold operation 2: fold the right half over the left half -/
def fold2 (p : Position) : Position :=
  ⟨p.row, 4 - p.col⟩

/-- Fold operation 3: fold along the diagonal from top-left to bottom-right -/
def fold3 (p : Position) : Position :=
  ⟨p.col, p.row⟩

/-- Fold operation 4: fold the bottom half over the top half -/
def fold4 (p : Position) : Position :=
  ⟨4 - p.row, p.col⟩

/-- Apply all fold operations in sequence -/
def applyAllFolds (p : Position) : Position :=
  fold4 (fold3 (fold2 (fold1 p)))

/-- The initial position of square 25 -/
def initialPos25 : Position :=
  ⟨4, 4⟩

/-- The theorem to be proved -/
theorem square25_on_top :
  applyAllFolds initialPos25 = ⟨0, 4⟩ :=
sorry


end square25_on_top_l3918_391894


namespace value_of_s_l3918_391896

-- Define the variables as natural numbers
variable (a b c p q s : ℕ)

-- Define the conditions
axiom distinct_nonzero : a ≠ b ∧ a ≠ c ∧ a ≠ p ∧ a ≠ q ∧ a ≠ s ∧
                         b ≠ c ∧ b ≠ p ∧ b ≠ q ∧ b ≠ s ∧
                         c ≠ p ∧ c ≠ q ∧ c ≠ s ∧
                         p ≠ q ∧ p ≠ s ∧
                         q ≠ s ∧
                         a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ p ≠ 0 ∧ q ≠ 0 ∧ s ≠ 0

axiom eq1 : a + b = p
axiom eq2 : p + c = s
axiom eq3 : s + a = q
axiom eq4 : b + c + q = 18

-- Theorem to prove
theorem value_of_s : s = 9 :=
sorry

end value_of_s_l3918_391896


namespace samantha_laundry_loads_l3918_391886

/-- The number of loads of laundry Samantha did in the wash -/
def laundry_loads : ℕ :=
  -- We'll define this later in the theorem
  sorry

/-- The cost of using a washer for one load -/
def washer_cost : ℚ := 4

/-- The cost of using a dryer for 10 minutes -/
def dryer_cost_per_10min : ℚ := (1 : ℚ) / 4

/-- The number of dryers Samantha uses -/
def num_dryers : ℕ := 3

/-- The number of minutes Samantha uses each dryer -/
def dryer_minutes : ℕ := 40

/-- The total amount Samantha spends -/
def total_spent : ℚ := 11

theorem samantha_laundry_loads :
  laundry_loads = 2 ∧
  laundry_loads * washer_cost +
    (num_dryers * (dryer_minutes / 10) * dryer_cost_per_10min) = total_spent :=
by sorry

end samantha_laundry_loads_l3918_391886


namespace sin_cos_difference_equals_neg_sqrt_two_over_two_l3918_391869

theorem sin_cos_difference_equals_neg_sqrt_two_over_two :
  Real.sin (18 * π / 180) * Real.cos (63 * π / 180) -
  Real.sin (72 * π / 180) * Real.sin (117 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_cos_difference_equals_neg_sqrt_two_over_two_l3918_391869


namespace years_before_aziz_birth_l3918_391865

def current_year : ℕ := 2021
def aziz_age : ℕ := 36
def parents_move_year : ℕ := 1982

theorem years_before_aziz_birth : 
  current_year - aziz_age - parents_move_year = 3 := by sorry

end years_before_aziz_birth_l3918_391865


namespace inequality_range_l3918_391887

theorem inequality_range (a : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  ∀ x : ℝ, (x^2 + a*x > 4*x + a - 3) ↔ (x > 3 ∨ x < -1) := by sorry

end inequality_range_l3918_391887


namespace problem_solution_l3918_391840

theorem problem_solution :
  let x : ℝ := 88 + (4/3) * 88
  let y : ℝ := x + (3/5) * x
  let z : ℝ := (1/2) * (x + y)
  z = 266.9325 := by
sorry

end problem_solution_l3918_391840


namespace sum_coordinates_reflection_l3918_391855

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define reflection over x-axis
def reflect_x (p : Point) : Point :=
  (p.1, -p.2)

theorem sum_coordinates_reflection (y : ℝ) :
  let C : Point := (3, y)
  let D : Point := reflect_x C
  C.1 + C.2 + D.1 + D.2 = 6 := by
sorry

end sum_coordinates_reflection_l3918_391855


namespace lemonade_solution_water_parts_l3918_391844

theorem lemonade_solution_water_parts (water_parts : ℝ) : 
  (7 : ℝ) / (water_parts + 7) > (1 : ℝ) / 10 ∧ 
  (7 : ℝ) / (water_parts + 7 - 2.1428571428571423 + 2.1428571428571423) = (1 : ℝ) / 10 → 
  water_parts = 63 := by
sorry

end lemonade_solution_water_parts_l3918_391844


namespace dog_paws_on_ground_l3918_391890

theorem dog_paws_on_ground (total_dogs : ℕ) (h1 : total_dogs = 12) : 
  (total_dogs / 2) * 2 + (total_dogs / 2) * 4 = 36 :=
by sorry

#check dog_paws_on_ground

end dog_paws_on_ground_l3918_391890


namespace geometric_sequence_ratio_main_theorem_l3918_391813

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : GeometricSequence a) :
  ∃ q : ℝ, q > 0 ∧ (a 2 = a 1 * q ∧ a 3 = a 2 * q ∧ a 4 = a 3 * q ∧ a 5 = a 4 * q) :=
sorry

/-- The second, half of the third, and twice the first term form an arithmetic sequence -/
def ArithmeticSubsequence (a : ℕ → ℝ) : Prop :=
  a 2 - (1/2 * a 3) = (1/2 * a 3) - (2 * a 1)

theorem main_theorem (a : ℕ → ℝ) (h1 : GeometricSequence a) (h2 : ArithmeticSubsequence a) :
  (a 3 + a 4) / (a 4 + a 5) = 1/2 :=
sorry

end geometric_sequence_ratio_main_theorem_l3918_391813


namespace expression_evaluation_l3918_391885

theorem expression_evaluation : (32 * 2 - 16) / (8 - (2 * 3)) = 24 := by
  sorry

end expression_evaluation_l3918_391885


namespace third_player_win_probability_is_one_fifteenth_l3918_391826

/-- Represents the probability of the third player winning in a four-player coin-flipping game -/
def third_player_win_probability : ℚ := 1 / 15

/-- The game has four players taking turns -/
def number_of_players : ℕ := 4

/-- The position of the player we're calculating the probability for -/
def target_player_position : ℕ := 3

/-- Theorem stating that the probability of the third player winning is 1/15 -/
theorem third_player_win_probability_is_one_fifteenth :
  third_player_win_probability = 1 / 15 := by sorry

end third_player_win_probability_is_one_fifteenth_l3918_391826


namespace courier_journey_l3918_391800

/-- The specified time for the courier's journey in minutes -/
def specified_time : ℝ := 40

/-- The total distance the courier traveled in kilometers -/
def total_distance : ℝ := 36

/-- The speed at which the courier arrives early in km/min -/
def early_speed : ℝ := 1.2

/-- The speed at which the courier arrives late in km/min -/
def late_speed : ℝ := 0.8

/-- The time by which the courier arrives early in minutes -/
def early_time : ℝ := 10

/-- The time by which the courier arrives late in minutes -/
def late_time : ℝ := 5

theorem courier_journey :
  early_speed * (specified_time - early_time) = late_speed * (specified_time + late_time) ∧
  total_distance = early_speed * (specified_time - early_time) :=
by sorry

end courier_journey_l3918_391800


namespace square_difference_305_295_l3918_391805

theorem square_difference_305_295 : 305^2 - 295^2 = 6000 := by
  sorry

end square_difference_305_295_l3918_391805


namespace nonnegative_difference_of_roots_l3918_391876

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  x^2 + 42*x + 336 = -48

-- Define the roots of the equation
def root1 : ℝ := -24
def root2 : ℝ := -16

-- Theorem statement
theorem nonnegative_difference_of_roots : 
  (quadratic_equation root1 ∧ quadratic_equation root2) → 
  |root1 - root2| = 8 := by
sorry

end nonnegative_difference_of_roots_l3918_391876


namespace joseph_card_distribution_l3918_391864

theorem joseph_card_distribution (initial_cards : ℕ) (cards_per_student : ℕ) (remaining_cards : ℕ) :
  initial_cards = 357 →
  cards_per_student = 23 →
  remaining_cards = 12 →
  ∃ (num_students : ℕ), num_students = 15 ∧ initial_cards = cards_per_student * num_students + remaining_cards :=
by sorry

end joseph_card_distribution_l3918_391864


namespace geometric_sequence_sum_l3918_391871

/-- Given a geometric sequence {a_n} where the sum of the first n terms is S_n = 2^n + r,
    prove that r = -1 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (r : ℝ) :
  (∀ n : ℕ, S n = 2^n + r) →
  (∀ n : ℕ, n ≥ 1 → a n = S n - S (n-1)) →
  (∀ n : ℕ, n ≥ 2 → a n = 2 * a (n-1)) →
  r = -1 := by
  sorry


end geometric_sequence_sum_l3918_391871


namespace triangle_tangent_range_l3918_391814

theorem triangle_tangent_range (a b c : ℝ) (A B C : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 0 < A ∧ A < π) (h5 : 0 < B ∧ B < π) (h6 : 0 < C ∧ C < π)
  (h7 : A + B + C = π) (h8 : a^2 + b^2 + Real.sqrt 2 * a * b = c^2) :
  0 < Real.tan A * Real.tan (2 * B) ∧ Real.tan A * Real.tan (2 * B) < 1/2 := by
sorry

end triangle_tangent_range_l3918_391814


namespace coin_problem_l3918_391833

theorem coin_problem (x y z : ℕ) : 
  x + y + z = 900 →
  x + 2*y + 5*z = 1950 →
  z = x / 2 →
  y = 450 := by
sorry

end coin_problem_l3918_391833


namespace largest_lcm_with_18_l3918_391874

theorem largest_lcm_with_18 (n : Fin 6 → ℕ) (h : n = ![3, 6, 9, 12, 15, 18]) :
  (Finset.range 6).sup (λ i => Nat.lcm 18 (n i)) = 90 := by
  sorry

end largest_lcm_with_18_l3918_391874


namespace constant_function_theorem_l3918_391884

/-- A function f: ℝ → ℝ is twice differentiable if it has a second derivative -/
def TwiceDifferentiable (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ Differentiable ℝ (deriv f)

/-- The given inequality condition for the function f -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (deriv^[2] f x) * Real.cos (f x) ≥ (deriv f x)^2 * Real.sin (f x)

/-- Main theorem: If f is twice differentiable and satisfies the inequality,
    then f is a constant function -/
theorem constant_function_theorem (f : ℝ → ℝ) 
    (h1 : TwiceDifferentiable f) (h2 : SatisfiesInequality f) :
    ∃ k : ℝ, ∀ x : ℝ, f x = k := by
  sorry

end constant_function_theorem_l3918_391884


namespace total_marbles_l3918_391893

theorem total_marbles (mary_marbles joan_marbles : ℕ) 
  (h1 : mary_marbles = 9) 
  (h2 : joan_marbles = 3) : 
  mary_marbles + joan_marbles = 12 := by
sorry

end total_marbles_l3918_391893


namespace equal_square_difference_subsequence_equal_square_difference_and_arithmetic_is_constant_l3918_391837

-- Define the property of being an "equal square difference sequence"
def is_equal_square_difference (a : ℕ → ℝ) : Prop :=
  ∃ p : ℝ, ∀ n : ℕ, n ≥ 2 → a n ^ 2 - a (n - 1) ^ 2 = p

-- Define the property of being an arithmetic sequence
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Theorem 1
theorem equal_square_difference_subsequence
  (a : ℕ → ℝ) (k : ℕ) (hk : k > 0) (ha : is_equal_square_difference a) :
  is_equal_square_difference (fun n ↦ a (k * n)) :=
sorry

-- Theorem 2
theorem equal_square_difference_and_arithmetic_is_constant
  (a : ℕ → ℝ) (ha1 : is_equal_square_difference a) (ha2 : is_arithmetic a) :
  ∃ c : ℝ, ∀ n : ℕ, a n = c :=
sorry

end equal_square_difference_subsequence_equal_square_difference_and_arithmetic_is_constant_l3918_391837


namespace remainder_problem_l3918_391843

theorem remainder_problem (x : ℤ) : x % 84 = 25 → x % 14 = 11 := by
  sorry

end remainder_problem_l3918_391843


namespace ellipse_foci_distance_l3918_391811

theorem ellipse_foci_distance (x y : ℝ) :
  (9 * x^2 + y^2 = 36) →
  (∃ (c : ℝ), c > 0 ∧ c^2 = 32 ∧ 2 * c = 8 * Real.sqrt 2) :=
by sorry

end ellipse_foci_distance_l3918_391811


namespace celia_video_streaming_budget_l3918_391815

/-- Represents Celia's monthly budget --/
structure Budget where
  food_per_week : ℕ
  rent : ℕ
  cell_phone : ℕ
  savings : ℕ
  weeks : ℕ
  savings_rate : ℚ

/-- Calculates the total known expenses --/
def total_known_expenses (b : Budget) : ℕ :=
  b.food_per_week * b.weeks + b.rent + b.cell_phone

/-- Calculates the total spending including savings --/
def total_spending (b : Budget) : ℚ :=
  b.savings / b.savings_rate

/-- Calculates the amount set aside for video streaming services --/
def video_streaming_budget (b : Budget) : ℚ :=
  total_spending b - total_known_expenses b

/-- Theorem stating that Celia's video streaming budget is $30 --/
theorem celia_video_streaming_budget :
  ∃ (b : Budget),
    b.food_per_week ≤ 100 ∧
    b.rent = 1500 ∧
    b.cell_phone = 50 ∧
    b.savings = 198 ∧
    b.weeks = 4 ∧
    b.savings_rate = 1/10 ∧
    video_streaming_budget b = 30 :=
  sorry

end celia_video_streaming_budget_l3918_391815


namespace negation_of_universal_proposition_l3918_391809

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) :=
by sorry

end negation_of_universal_proposition_l3918_391809


namespace function_always_positive_l3918_391863

theorem function_always_positive
  (f : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (h : ∀ x : ℝ, 2 * f x + x * (deriv f x) > x^2) :
  ∀ x : ℝ, f x > 0 :=
sorry

end function_always_positive_l3918_391863


namespace five_is_solution_l3918_391801

/-- The equation we're working with -/
def equation (x : ℝ) : Prop :=
  x^3 + 2*(x+1)^3 + 3*(x+2)^3 = 3*(x+3)^3

/-- Theorem stating that 5 is a solution to the equation -/
theorem five_is_solution : equation 5 := by
  sorry

end five_is_solution_l3918_391801


namespace total_viewing_time_is_900_hours_l3918_391820

/-- Calculates the total viewing time for two people watching multiple videos at different speeds -/
def totalViewingTime (videoLength : ℕ) (numVideos : ℕ) (lilaSpeed : ℕ) (rogerSpeed : ℕ) : ℕ :=
  (videoLength * numVideos / lilaSpeed) + (videoLength * numVideos / rogerSpeed)

/-- Theorem stating that the total viewing time for Lila and Roger is 900 hours -/
theorem total_viewing_time_is_900_hours :
  totalViewingTime 100 6 2 1 = 900 := by
  sorry

end total_viewing_time_is_900_hours_l3918_391820


namespace cos_alpha_for_point_P_l3918_391856

theorem cos_alpha_for_point_P (α : Real) :
  let P : ℝ × ℝ := (-3, 4)
  (∃ t : ℝ, t > 0 ∧ P = (t * Real.cos α, t * Real.sin α)) →
  Real.cos α = -3/5 := by
  sorry

end cos_alpha_for_point_P_l3918_391856


namespace like_terms_exponent_value_l3918_391846

theorem like_terms_exponent_value (x y : ℝ) (m n : ℕ) :
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * x^(m+3) * y^6 = b * x^5 * y^(2*n)) →
  m^n = 8 := by
sorry

end like_terms_exponent_value_l3918_391846


namespace equation_solution_l3918_391808

theorem equation_solution : ∃ X : ℝ, 
  (0.125 * X) / ((19/24 - 21/40) * 8*(7/16)) = 
  ((1 + 28/63 - 17/21) * 0.7) / (0.675 * 2.4 - 0.02) ∧ X = 5 := by
  sorry

end equation_solution_l3918_391808


namespace cube_of_prime_condition_l3918_391817

theorem cube_of_prime_condition (n : ℕ) : 
  (∃ p : ℕ, Nat.Prime p ∧ 2^n + n^2 + 25 = p^3) ↔ n = 6 :=
sorry

end cube_of_prime_condition_l3918_391817


namespace smallest_shift_is_sixty_l3918_391889

/-- A function with period 30 -/
def periodic_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (x + 30) = g x

/-- The smallest positive shift for g(2x) -/
def smallest_shift (g : ℝ → ℝ) (b : ℝ) : Prop :=
  (b > 0) ∧
  (∀ x : ℝ, g (2*x + b) = g (2*x)) ∧
  (∀ c : ℝ, c > 0 → (∀ x : ℝ, g (2*x + c) = g (2*x)) → b ≤ c)

theorem smallest_shift_is_sixty (g : ℝ → ℝ) :
  periodic_function g → smallest_shift g 60 := by
  sorry

end smallest_shift_is_sixty_l3918_391889


namespace max_d_value_l3918_391859

def a (n : ℕ) : ℕ := 100 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value : ∃ (n : ℕ), d n = 401 ∧ ∀ (m : ℕ), d m ≤ 401 :=
sorry

end max_d_value_l3918_391859


namespace prop_truth_values_l3918_391891

-- Define a structure for a line
structure Line where
  slope : ℝ
  x_intercept : ℝ
  y_intercept : ℝ

-- Define the original proposition
def original_prop (l : Line) : Prop :=
  l.slope = -1 → l.x_intercept = l.y_intercept

-- Define the converse
def converse_prop (l : Line) : Prop :=
  l.x_intercept = l.y_intercept → l.slope = -1

-- Define the inverse
def inverse_prop (l : Line) : Prop :=
  l.slope ≠ -1 → l.x_intercept ≠ l.y_intercept

-- Define the contrapositive
def contrapositive_prop (l : Line) : Prop :=
  l.x_intercept ≠ l.y_intercept → l.slope ≠ -1

-- Theorem stating the truth values of the propositions
theorem prop_truth_values :
  ∃ l : Line, original_prop l ∧
  ¬(∀ l : Line, converse_prop l) ∧
  ¬(∀ l : Line, inverse_prop l) ∧
  (∀ l : Line, contrapositive_prop l) :=
sorry

end prop_truth_values_l3918_391891


namespace dividend_calculation_l3918_391857

theorem dividend_calculation (divisor quotient remainder dividend : ℕ) : 
  divisor = 17 →
  quotient = 10 →
  remainder = 2 →
  dividend = divisor * quotient + remainder →
  dividend = 172 := by
sorry

end dividend_calculation_l3918_391857


namespace integer_solution_equation_l3918_391883

theorem integer_solution_equation (k x : ℤ) : 
  (Real.sqrt (39 - 6 * Real.sqrt 12) + Real.sqrt (k * x * (k * x + Real.sqrt 12) + 3) = 2 * k) → 
  (k = 3 ∨ k = 6) := by
sorry

end integer_solution_equation_l3918_391883


namespace monthly_order_is_168_l3918_391852

/-- The number of apples Chandler eats per week -/
def chandler_weekly : ℕ := 23

/-- The number of apples Lucy eats per week -/
def lucy_weekly : ℕ := 19

/-- The number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- The number of apples Chandler and Lucy need to order for a month -/
def monthly_order : ℕ := (chandler_weekly + lucy_weekly) * weeks_per_month

theorem monthly_order_is_168 : monthly_order = 168 := by
  sorry

end monthly_order_is_168_l3918_391852


namespace rectangle_z_value_l3918_391860

-- Define the rectangle
def rectangle (z : ℝ) : Set (ℝ × ℝ) :=
  {(-2, z), (6, z), (-2, 4), (6, 4)}

-- Define the area of the rectangle
def area (z : ℝ) : ℝ :=
  (6 - (-2)) * (z - 4)

-- Theorem statement
theorem rectangle_z_value (z : ℝ) :
  z > 0 ∧ area z = 64 → z = 12 := by
  sorry

end rectangle_z_value_l3918_391860


namespace cube_root_equation_sum_l3918_391872

theorem cube_root_equation_sum (x y z : ℕ+) :
  (4 : ℝ) * ((7 : ℝ)^(1/3) - (6 : ℝ)^(1/3))^(1/2) = (x : ℝ)^(1/3) + (y : ℝ)^(1/3) - (z : ℝ)^(1/3) →
  x + y + z = 75 := by
  sorry

end cube_root_equation_sum_l3918_391872


namespace sqrt_eight_and_nine_sixteenths_l3918_391880

theorem sqrt_eight_and_nine_sixteenths (x : ℝ) : 
  x = Real.sqrt (8 + 9/16) → x = Real.sqrt 137 / 4 := by
  sorry

end sqrt_eight_and_nine_sixteenths_l3918_391880


namespace reflected_tetrahedron_volume_l3918_391841

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  A : ℝ × ℝ × ℝ
  B : ℝ × ℝ × ℝ
  C : ℝ × ℝ × ℝ
  D : ℝ × ℝ × ℝ

/-- Calculates the volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- Reflects a point with respect to another point -/
def reflect (point center : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Calculates the centroid of a triangle -/
def centroid (a b c : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Creates a new tetrahedron by reflecting each vertex of the original tetrahedron
    with respect to the centroid of the opposite face -/
def reflectedTetrahedron (t : Tetrahedron) : Tetrahedron :=
  let A' := reflect t.A (centroid t.B t.C t.D)
  let B' := reflect t.B (centroid t.A t.C t.D)
  let C' := reflect t.C (centroid t.A t.B t.D)
  let D' := reflect t.D (centroid t.A t.B t.C)
  ⟨A', B', C', D'⟩

/-- Theorem: The volume of the reflected tetrahedron is 125/27 times the volume of the original tetrahedron -/
theorem reflected_tetrahedron_volume (t : Tetrahedron) :
  volume (reflectedTetrahedron t) = (125 / 27) * volume t := by sorry

end reflected_tetrahedron_volume_l3918_391841


namespace always_in_range_l3918_391836

theorem always_in_range (k : ℝ) : ∃ x : ℝ, x^2 + 2*k*x + 1 = 3 := by
  sorry

end always_in_range_l3918_391836


namespace paper_plates_and_cups_cost_l3918_391867

theorem paper_plates_and_cups_cost (plate_cost cup_cost : ℝ) : 
  100 * plate_cost + 200 * cup_cost = 7.5 → 
  20 * plate_cost + 40 * cup_cost = 1.5 := by
sorry

end paper_plates_and_cups_cost_l3918_391867


namespace orchids_sold_correct_l3918_391877

/-- The number of orchids sold by a plant supplier -/
def orchids_sold : ℕ := 20

/-- The price of each orchid -/
def orchid_price : ℕ := 50

/-- The number of potted Chinese money plants sold -/
def money_plants_sold : ℕ := 15

/-- The price of each potted Chinese money plant -/
def money_plant_price : ℕ := 25

/-- The number of workers -/
def workers : ℕ := 2

/-- The wage paid to each worker -/
def worker_wage : ℕ := 40

/-- The cost of new pots -/
def new_pots_cost : ℕ := 150

/-- The amount left after all transactions -/
def amount_left : ℕ := 1145

/-- Theorem stating that the number of orchids sold is correct given the problem conditions -/
theorem orchids_sold_correct :
  orchids_sold * orchid_price + 
  money_plants_sold * money_plant_price - 
  (workers * worker_wage + new_pots_cost) = 
  amount_left := by sorry

end orchids_sold_correct_l3918_391877


namespace equation_solution_l3918_391847

theorem equation_solution (m n : ℝ) (h : m ≠ n) :
  let f := fun x : ℝ => x^2 + (x + m)^2 - (x + n)^2 - 2*m*n
  (∀ x, f x = 0 ↔ x = -m + n + Real.sqrt (2*(n^2 - m*n + m^2)) ∨
                   x = -m + n - Real.sqrt (2*(n^2 - m*n + m^2))) :=
by sorry

end equation_solution_l3918_391847


namespace geometric_sequence_S3_lower_bound_l3918_391875

/-- A geometric sequence with positive terms where the second term is 1 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (a 2 = 1) ∧ (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q)

/-- The sum of the first three terms of a sequence -/
def S3 (a : ℕ → ℝ) : ℝ := a 1 + a 2 + a 3

theorem geometric_sequence_S3_lower_bound
  (a : ℕ → ℝ) (h : GeometricSequence a) : S3 a ≥ 3 :=
sorry

end geometric_sequence_S3_lower_bound_l3918_391875


namespace alice_bob_number_sum_l3918_391854

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem alice_bob_number_sum :
  ∀ (A B : ℕ),
  A ∈ Finset.range 50 →
  B ∈ Finset.range 50 →
  A ≠ B →
  A ≠ 1 →
  A ≠ 50 →
  is_prime B →
  (∃ k : ℕ, 120 * B + A = k * k) →
  A + B = 43 :=
by sorry

end alice_bob_number_sum_l3918_391854


namespace squares_below_line_eq_660_l3918_391806

/-- The number of squares below the line 7x + 221y = 1547 in the first quadrant -/
def squares_below_line : ℕ :=
  let x_intercept : ℕ := 221
  let y_intercept : ℕ := 7
  let total_squares : ℕ := x_intercept * y_intercept
  let diagonal_squares : ℕ := x_intercept + y_intercept - 1
  let non_diagonal_squares : ℕ := total_squares - diagonal_squares
  non_diagonal_squares / 2

/-- The number of squares below the line 7x + 221y = 1547 in the first quadrant is 660 -/
theorem squares_below_line_eq_660 : squares_below_line = 660 := by
  sorry

end squares_below_line_eq_660_l3918_391806


namespace money_ratio_problem_l3918_391850

/-- Given the ratios of money between Ram and Gopal (7:17) and between Gopal and Krishan (7:17),
    and that Ram has Rs. 588, prove that Krishan has Rs. 12,065. -/
theorem money_ratio_problem (ram gopal krishan : ℚ) : 
  ram / gopal = 7 / 17 →
  gopal / krishan = 7 / 17 →
  ram = 588 →
  krishan = 12065 := by
sorry

end money_ratio_problem_l3918_391850


namespace average_of_twenty_digits_l3918_391892

theorem average_of_twenty_digits :
  let total_count : ℕ := 20
  let group1_count : ℕ := 14
  let group2_count : ℕ := 6
  let group1_average : ℝ := 390
  let group2_average : ℝ := 756.67
  let total_average : ℝ := (group1_count * group1_average + group2_count * group2_average) / total_count
  total_average = 500.001 := by sorry

end average_of_twenty_digits_l3918_391892


namespace students_neither_outstanding_nor_pioneer_l3918_391849

theorem students_neither_outstanding_nor_pioneer (total : ℕ) (outstanding : ℕ) (pioneers : ℕ) (both : ℕ)
  (h_total : total = 87)
  (h_outstanding : outstanding = 58)
  (h_pioneers : pioneers = 63)
  (h_both : both = 49) :
  total - outstanding - pioneers + both = 15 :=
by sorry

end students_neither_outstanding_nor_pioneer_l3918_391849


namespace combined_average_score_l3918_391832

theorem combined_average_score (g₁ g₂ : ℕ) (avg₁ avg₂ : ℚ) :
  g₁ > 0 → g₂ > 0 →
  avg₁ = 88 →
  avg₂ = 76 →
  g₁ = (4 * g₂) / 5 →
  let total_score := g₁ * avg₁ + g₂ * avg₂
  let total_students := g₁ + g₂
  (total_score / total_students : ℚ) = 81 := by
  sorry

end combined_average_score_l3918_391832


namespace product_2022_sum_possibilities_l3918_391878

theorem product_2022_sum_possibilities (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧ 
  a * b * c * d * e = 2022 → 
  a + b + c + d + e = 342 ∨ 
  a + b + c + d + e = 338 ∨ 
  a + b + c + d + e = 336 ∨ 
  a + b + c + d + e = -332 :=
by sorry

end product_2022_sum_possibilities_l3918_391878


namespace inequality_and_equality_condition_l3918_391802

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_product : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + c*d + b*c + a*d + a*c + b*d ≥ 10 ∧ 
  (a^2 + b^2 + c^2 + d^2 + a*b + c*d + b*c + a*d + a*c + b*d = 10 ↔ a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) :=
by sorry

end inequality_and_equality_condition_l3918_391802


namespace smallest_gcd_multiple_l3918_391827

theorem smallest_gcd_multiple (a b : ℕ+) (h : Nat.gcd a b = 18) :
  (Nat.gcd (12 * a) (20 * b) ≥ 72) ∧ ∃ (a₀ b₀ : ℕ+), Nat.gcd a₀ b₀ = 18 ∧ Nat.gcd (12 * a₀) (20 * b₀) = 72 := by
  sorry

end smallest_gcd_multiple_l3918_391827


namespace sum_in_base8_l3918_391866

/-- Converts a base-8 number represented as a list of digits to a natural number. -/
def fromBase8 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a natural number to its base-8 representation as a list of digits. -/
def toBase8 (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: toBase8 (n / 8)

theorem sum_in_base8 :
  let a := fromBase8 [4, 7, 6, 5]
  let b := fromBase8 [5, 6, 3, 2]
  toBase8 (a + b) = [6, 2, 2, 0, 1] := by
  sorry

#eval fromBase8 [4, 7, 6, 5]
#eval fromBase8 [5, 6, 3, 2]
#eval toBase8 (fromBase8 [4, 7, 6, 5] + fromBase8 [5, 6, 3, 2])

end sum_in_base8_l3918_391866


namespace white_ball_probability_l3918_391858

/-- Represents the number of balls initially in the bag -/
def initial_balls : ℕ := 6

/-- Represents the total number of balls after adding the white ball -/
def total_balls : ℕ := initial_balls + 1

/-- Represents the number of white balls added -/
def white_balls : ℕ := 1

/-- The probability of extracting the white ball -/
def prob_white : ℚ := white_balls / total_balls

theorem white_ball_probability :
  prob_white = 1 / 7 := by sorry

end white_ball_probability_l3918_391858


namespace geometric_progression_proof_l3918_391842

theorem geometric_progression_proof (y : ℝ) : 
  (90 + y)^2 = (30 + y) * (180 + y) → 
  y = 90 ∧ (90 + y) / (30 + y) = 3/2 := by
  sorry

end geometric_progression_proof_l3918_391842


namespace sports_competition_results_l3918_391829

/-- Represents the outcome of a single event -/
inductive EventOutcome
| SchoolAWins
| SchoolBWins

/-- Represents the outcome of the entire championship -/
inductive ChampionshipOutcome
| SchoolAWins
| SchoolBWins

/-- The probability of School A winning each event -/
def probSchoolAWins : Fin 3 → ℝ
| 0 => 0.5
| 1 => 0.4
| 2 => 0.8

/-- The score awarded for winning an event -/
def winningScore : ℕ := 10

/-- Calculate the probability of School A winning the championship -/
def probSchoolAWinsChampionship : ℝ := sorry

/-- Calculate the expectation of School B's total score -/
def expectationSchoolBScore : ℝ := sorry

/-- Theorem stating the main results -/
theorem sports_competition_results :
  probSchoolAWinsChampionship = 0.6 ∧ expectationSchoolBScore = 13 := by sorry

end sports_competition_results_l3918_391829


namespace external_tangent_intersection_collinear_l3918_391895

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point type
abbrev Point := ℝ × ℝ

-- Define a function to check if three points are collinear
def collinear (p q r : Point) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

-- Define a function to get the intersection point of external tangents
def externalTangentIntersection (c1 c2 : Circle) : Point :=
  sorry  -- The actual implementation is not needed for the theorem statement

-- State the theorem
theorem external_tangent_intersection_collinear (γ₁ γ₂ γ₃ : Circle) :
  let X := externalTangentIntersection γ₁ γ₂
  let Y := externalTangentIntersection γ₂ γ₃
  let Z := externalTangentIntersection γ₃ γ₁
  collinear X Y Z :=
by sorry

end external_tangent_intersection_collinear_l3918_391895


namespace nth_prime_47_l3918_391818

def is_nth_prime (n : ℕ) (p : ℕ) : Prop :=
  p.Prime ∧ (Finset.filter Nat.Prime (Finset.range p)).card = n

theorem nth_prime_47 (n : ℕ) :
  is_nth_prime n 47 → n = 15 :=
by
  sorry

end nth_prime_47_l3918_391818


namespace isabellas_haircuts_l3918_391882

/-- The total length of hair cut off in two haircuts -/
def total_hair_cut (initial_length first_cut_length second_cut_length : ℝ) : ℝ :=
  (initial_length - first_cut_length) + (first_cut_length - second_cut_length)

/-- Theorem: The total length of hair cut off in Isabella's two haircuts is 9 inches -/
theorem isabellas_haircuts :
  total_hair_cut 18 14 9 = 9 := by
  sorry

end isabellas_haircuts_l3918_391882


namespace least_sum_with_conditions_l3918_391897

theorem least_sum_with_conditions (m n : ℕ+) 
  (h1 : Nat.gcd (m + n) 330 = 1)
  (h2 : ∃ k : ℕ, m ^ m.val = k * n ^ n.val)
  (h3 : ¬ ∃ k : ℕ, m = k * n) :
  (∀ m' n' : ℕ+, 
    Nat.gcd (m' + n') 330 = 1 → 
    (∃ k : ℕ, m' ^ m'.val = k * n' ^ n'.val) → 
    (¬ ∃ k : ℕ, m' = k * n') → 
    m' + n' ≥ m + n) → 
  m + n = 429 := by
sorry

end least_sum_with_conditions_l3918_391897


namespace school_transfer_percentage_l3918_391810

theorem school_transfer_percentage : 
  ∀ (total_students : ℕ) (school_A_percent school_C_percent : ℚ),
    school_A_percent = 60 / 100 →
    (30 / 100 * school_A_percent + 
     (school_C_percent - 30 / 100 * school_A_percent) / (1 - school_A_percent)) * total_students = 
    school_C_percent * total_students →
    school_C_percent = 34 / 100 →
    (school_C_percent - 30 / 100 * school_A_percent) / (1 - school_A_percent) = 40 / 100 :=
by sorry

end school_transfer_percentage_l3918_391810


namespace quadratic_range_solution_set_l3918_391868

/-- A quadratic function with real coefficients -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

/-- The theorem stating the value of c given the conditions -/
theorem quadratic_range_solution_set (a b m : ℝ) :
  (∀ x, f a b x ≥ 0) →  -- range of f is [0, +∞)
  (∃ c, ∀ x, f a b x < c ↔ m < x ∧ x < m + 6) →  -- solution set of f(x) < c is (m, m+6)
  ∃ c, c = 9 := by sorry

end quadratic_range_solution_set_l3918_391868


namespace ball_probability_relationship_l3918_391862

/-- Given a pocket with 7 balls, including 3 white and 4 black balls, 
    if x white balls and y black balls are added, and the probability 
    of drawing a white ball becomes 1/4, then y = 3x + 5 -/
theorem ball_probability_relationship (x y : ℤ) : 
  (((3 : ℚ) + x) / ((7 : ℚ) + x + y) = (1 : ℚ) / 4) → y = 3 * x + 5 := by
  sorry

end ball_probability_relationship_l3918_391862


namespace value_of_expression_l3918_391881

theorem value_of_expression (x : ℝ) (h : x = 5) : 4 * x - 2 = 18 := by
  sorry

end value_of_expression_l3918_391881


namespace intersection_when_a_zero_subset_condition_l3918_391861

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

-- Theorem 1: When a = 0, A ∩ B = {x | 0 < x < 1}
theorem intersection_when_a_zero :
  A 0 ∩ B = {x | 0 < x ∧ x < 1} := by sorry

-- Theorem 2: A ⊆ B if and only if 1 ≤ a ≤ 2
theorem subset_condition (a : ℝ) :
  A a ⊆ B ↔ 1 ≤ a ∧ a ≤ 2 := by sorry

end intersection_when_a_zero_subset_condition_l3918_391861


namespace no_divisible_by_five_l3918_391898

def g (x : ℤ) : ℤ := x^2 + 5*x + 3

def T : Set ℤ := {x | 0 ≤ x ∧ x ≤ 30}

theorem no_divisible_by_five : 
  ∀ t ∈ T, ¬(g t % 5 = 0) := by
sorry

end no_divisible_by_five_l3918_391898


namespace graduation_ceremony_attendance_l3918_391848

/-- Graduation ceremony attendance problem -/
theorem graduation_ceremony_attendance
  (graduates : ℕ)
  (chairs : ℕ)
  (parents_per_graduate : ℕ)
  (h_graduates : graduates = 50)
  (h_chairs : chairs = 180)
  (h_parents : parents_per_graduate = 2)
  (h_admins : administrators = teachers / 2) :
  teachers = 20 :=
by
  sorry

end graduation_ceremony_attendance_l3918_391848


namespace sprint_competition_races_l3918_391888

/-- Calculates the number of races required to determine a champion in a sprint competition. -/
def races_to_champion (total_sprinters : ℕ) (lanes : ℕ) (eliminated_per_race : ℕ) (advance_interval : ℕ) : ℕ :=
  let regular_races := 32
  let special_races := 16
  regular_races + special_races

/-- Theorem stating that 48 races are required for the given sprint competition setup. -/
theorem sprint_competition_races :
  races_to_champion 300 8 6 3 = 48 := by
  sorry

end sprint_competition_races_l3918_391888


namespace candy_bar_problem_l3918_391845

theorem candy_bar_problem (fred : ℕ) (bob : ℕ) (jacqueline : ℕ) : 
  fred = 12 →
  bob = fred + 6 →
  jacqueline = 10 * (fred + bob) →
  (40 : ℚ) / 100 * jacqueline = 120 :=
by
  sorry

end candy_bar_problem_l3918_391845


namespace triangle_area_l3918_391870

theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 7) (h2 : b = 11) (h3 : C = 60 * π / 180) :
  (1 / 2) * a * b * Real.sin C = (77 * Real.sqrt 3) / 4 := by
  sorry

end triangle_area_l3918_391870


namespace bus_variance_proof_l3918_391873

def bus_durations : List ℝ := [10, 11, 9, 9, 11]

theorem bus_variance_proof :
  let n : ℕ := bus_durations.length
  let mean : ℝ := (bus_durations.sum) / n
  let variance : ℝ := (bus_durations.map (fun x => (x - mean)^2)).sum / n
  (mean = 10 ∧ n = 5) → variance = 0.8 := by
  sorry

end bus_variance_proof_l3918_391873


namespace exists_five_digit_not_sum_of_beautiful_l3918_391831

/-- A beautiful number is a number consisting of identical digits. -/
def is_beautiful (n : ℕ) : Prop :=
  ∃ d : ℕ, d ≤ 9 ∧ ∃ k : ℕ, k > 0 ∧ n = d * (10^k - 1) / 9

/-- The sum of beautiful numbers with pairwise different lengths. -/
def sum_of_beautiful (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ), 
    n = a * 11111 + b * 1111 + c * 111 + d * 11 + e * 1 ∧
    is_beautiful (a * 11111) ∧ 
    is_beautiful (b * 1111) ∧ 
    is_beautiful (c * 111) ∧ 
    is_beautiful (d * 11) ∧ 
    is_beautiful e

/-- Theorem: There exists a five-digit number that cannot be represented as a sum of beautiful numbers with pairwise different lengths. -/
theorem exists_five_digit_not_sum_of_beautiful : 
  ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ ¬(sum_of_beautiful n) := by
  sorry

end exists_five_digit_not_sum_of_beautiful_l3918_391831


namespace sufficient_not_necessary_condition_l3918_391804

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x ≥ 1 → |x + 1| + |x - 1| = 2 * |x|) ∧
  ¬(|x + 1| + |x - 1| = 2 * |x| → x ≥ 1) :=
by sorry

end sufficient_not_necessary_condition_l3918_391804


namespace bike_distance_l3918_391819

/-- The distance traveled by a bike given its speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A bike traveling at 50 m/s for 7 seconds covers a distance of 350 meters -/
theorem bike_distance : distance_traveled 50 7 = 350 := by
  sorry

end bike_distance_l3918_391819


namespace last_four_digits_of_5_pow_2011_l3918_391838

/-- The last four digits of 5^n -/
def lastFourDigits (n : ℕ) : ℕ := 5^n % 10000

/-- Given conditions -/
axiom base_case_5 : lastFourDigits 5 = 3125
axiom base_case_6 : lastFourDigits 6 = 5625
axiom base_case_7 : lastFourDigits 7 = 8125

/-- Theorem statement -/
theorem last_four_digits_of_5_pow_2011 : lastFourDigits 2011 = 8125 := by
  sorry

end last_four_digits_of_5_pow_2011_l3918_391838


namespace train_crossing_time_l3918_391835

/-- Time for a train to cross another train moving in the same direction -/
theorem train_crossing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 420)
  (h2 : length2 = 640)
  (h3 : speed1 = 72)
  (h4 : speed2 = 36) :
  (length1 + length2) / ((speed1 - speed2) * (1000 / 3600)) = 106 := by
  sorry

end train_crossing_time_l3918_391835


namespace binomial_coefficient_n_minus_two_l3918_391824

theorem binomial_coefficient_n_minus_two (n : ℕ+) : 
  Nat.choose n (n - 2) = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_n_minus_two_l3918_391824


namespace legs_code_is_6189_l3918_391828

-- Define the type for our code
def Code := String

-- Define the mapping function
def digit_map (code : Code) (c : Char) : Nat :=
  match c with
  | 'N' => 0
  | 'E' => 1
  | 'W' => 2
  | 'C' => 3
  | 'H' => 4
  | 'A' => 5
  | 'L' => 6
  | 'G' => 8
  | 'S' => 9
  | _ => 0  -- Default case, should not occur in our problem

-- Define the function to convert a code word to a number
def code_to_number (code : Code) : Nat :=
  code.foldl (fun acc c => 10 * acc + digit_map code c) 0

-- The main theorem
theorem legs_code_is_6189 (code : Code) (h1 : code = "NEW CHALLENGES") :
  code_to_number "LEGS" = 6189 := by
  sorry


end legs_code_is_6189_l3918_391828


namespace emu_egg_production_l3918_391812

/-- The number of eggs laid by each female emu per day -/
def eggs_per_female_emu_per_day (num_pens : ℕ) (emus_per_pen : ℕ) (total_eggs_per_week : ℕ) : ℚ :=
  let total_emus := num_pens * emus_per_pen
  let female_emus := total_emus / 2
  (total_eggs_per_week : ℚ) / (female_emus : ℚ) / 7

theorem emu_egg_production :
  eggs_per_female_emu_per_day 4 6 84 = 1 := by
  sorry

#eval eggs_per_female_emu_per_day 4 6 84

end emu_egg_production_l3918_391812


namespace green_mandm_probability_l3918_391839

/-- Represents the count of M&Ms of each color -/
structure MandMCount where
  green : ℕ
  red : ℕ
  blue : ℕ
  orange : ℕ
  yellow : ℕ
  purple : ℕ
  brown : ℕ

/-- Calculates the total count of M&Ms -/
def totalCount (count : MandMCount) : ℕ :=
  count.green + count.red + count.blue + count.orange + count.yellow + count.purple + count.brown

/-- Represents the actions taken by Carter and others -/
def finalCount : MandMCount :=
  let initial := MandMCount.mk 35 25 10 15 0 0 0
  let afterCarter := MandMCount.mk (initial.green - 20) (initial.red - 8) initial.blue initial.orange 0 0 0
  let afterSister := MandMCount.mk afterCarter.green (afterCarter.red / 2) (afterCarter.blue - 5) afterCarter.orange 14 0 0
  let afterAlex := MandMCount.mk afterSister.green afterSister.red afterSister.blue (afterSister.orange - 7) (afterSister.yellow - 3) 8 0
  MandMCount.mk afterAlex.green afterAlex.red 0 afterAlex.orange afterAlex.yellow afterAlex.purple 10

/-- The main theorem to prove -/
theorem green_mandm_probability :
  (finalCount.green : ℚ) / (totalCount finalCount : ℚ) = 1/4 := by
  sorry

end green_mandm_probability_l3918_391839


namespace random_triangle_probability_l3918_391816

/-- The number of ways to choose 3 different numbers from 1 to 179 -/
def total_combinations : ℕ := 939929

/-- The number of valid angle triples that form a triangle -/
def valid_triples : ℕ := 2611

/-- A function that determines if three numbers form valid angles of a triangle -/
def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b + c = 180 ∧ a > 0 ∧ b > 0 ∧ c > 0

/-- The probability of randomly selecting three different numbers from 1 to 179
    that form valid angles of a triangle -/
def triangle_probability : ℚ := valid_triples / total_combinations

/-- Theorem stating the probability of randomly selecting three different numbers
    from 1 to 179 that form valid angles of a triangle -/
theorem random_triangle_probability :
  triangle_probability = 2611 / 939929 := by sorry

end random_triangle_probability_l3918_391816


namespace exists_angle_sum_with_adjacent_le_180_l3918_391821

/-- A convex quadrilateral is a quadrilateral where each interior angle is less than 180 degrees. -/
structure ConvexQuadrilateral where
  angles : Fin 4 → ℝ
  sum_angles : (angles 0) + (angles 1) + (angles 2) + (angles 3) = 360
  all_angles_less_than_180 : ∀ i, angles i < 180

/-- 
In any convex quadrilateral, there exists an angle such that the sum of 
this angle with each of its adjacent angles does not exceed 180°.
-/
theorem exists_angle_sum_with_adjacent_le_180 (q : ConvexQuadrilateral) : 
  ∃ i : Fin 4, (q.angles i + q.angles ((i + 1) % 4) ≤ 180) ∧ 
                (q.angles i + q.angles ((i + 3) % 4) ≤ 180) := by
  sorry


end exists_angle_sum_with_adjacent_le_180_l3918_391821


namespace product_divisible_by_sum_iff_not_odd_prime_l3918_391879

theorem product_divisible_by_sum_iff_not_odd_prime (n : ℕ) : 
  (∃ k : ℕ, n.factorial = k * (n * (n + 1) / 2)) ↔ ¬(Nat.Prime (n + 1) ∧ Odd (n + 1)) :=
sorry

end product_divisible_by_sum_iff_not_odd_prime_l3918_391879


namespace train_capacity_ratio_l3918_391825

def train_problem (red_boxcars blue_boxcars black_boxcars : ℕ)
  (black_capacity : ℕ) (red_multiplier : ℕ) (total_capacity : ℕ) : Prop :=
  red_boxcars = 3 ∧
  blue_boxcars = 4 ∧
  black_boxcars = 7 ∧
  black_capacity = 4000 ∧
  red_multiplier = 3 ∧
  total_capacity = 132000 ∧
  ∃ (blue_capacity : ℕ),
    red_boxcars * (red_multiplier * blue_capacity) +
    blue_boxcars * blue_capacity +
    black_boxcars * black_capacity = total_capacity ∧
    2 * black_capacity = blue_capacity

theorem train_capacity_ratio 
  (red_boxcars blue_boxcars black_boxcars : ℕ)
  (black_capacity : ℕ) (red_multiplier : ℕ) (total_capacity : ℕ) :
  train_problem red_boxcars blue_boxcars black_boxcars black_capacity red_multiplier total_capacity →
  ∃ (blue_capacity : ℕ), 2 * black_capacity = blue_capacity :=
by sorry

end train_capacity_ratio_l3918_391825


namespace prob_two_heads_is_one_fourth_l3918_391851

/-- The probability of getting heads on a single flip of a fair coin -/
def prob_heads : ℚ := 1/2

/-- The probability of getting heads on both of the first two flips of a fair coin -/
def prob_two_heads : ℚ := prob_heads * prob_heads

/-- Theorem stating that the probability of getting heads on both of the first two flips of a fair coin is 1/4 -/
theorem prob_two_heads_is_one_fourth : prob_two_heads = 1/4 := by
  sorry

end prob_two_heads_is_one_fourth_l3918_391851


namespace triangle_ax_length_l3918_391853

-- Define the triangle ABC and point X
structure Triangle :=
  (A B C X : ℝ × ℝ)

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  let d := (λ p q : ℝ × ℝ => ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt)
  d t.A t.B = 60 ∧ 
  d t.A t.C = 36 ∧
  -- C is on the angle bisector of ∠AXB
  (t.C.1 - t.X.1) / (t.A.1 - t.X.1) = (t.C.2 - t.X.2) / (t.A.2 - t.X.2) ∧
  (t.C.1 - t.X.1) / (t.B.1 - t.X.1) = (t.C.2 - t.X.2) / (t.B.2 - t.X.2)

-- Theorem statement
theorem triangle_ax_length (t : Triangle) (h : TriangleProperties t) : 
  let d := (λ p q : ℝ × ℝ => ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt)
  d t.A t.X = 20 := by
  sorry

end triangle_ax_length_l3918_391853
