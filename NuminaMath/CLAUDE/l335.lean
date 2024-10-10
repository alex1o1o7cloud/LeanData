import Mathlib

namespace sqrt_meaningful_range_l335_33584

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2 - x) ↔ x ≤ 2 := by sorry

end sqrt_meaningful_range_l335_33584


namespace daughters_age_is_twelve_l335_33583

/-- Proves that the daughter's age is 12 given the conditions about the father and daughter's ages -/
theorem daughters_age_is_twelve (D : ℕ) (F : ℕ) : 
  F = 3 * D →  -- Father's age is three times daughter's age this year
  F + 12 = 2 * (D + 12) →  -- After 12 years, father's age will be twice daughter's age
  D = 12 :=  -- Daughter's current age is 12
by
  sorry


end daughters_age_is_twelve_l335_33583


namespace system_of_equations_solutions_l335_33586

theorem system_of_equations_solutions :
  -- System 1
  (∃ x y : ℚ, 2 * x + 4 * y = 5 ∧ x = 1 - y) →
  (∃ x y : ℚ, 2 * x + 4 * y = 5 ∧ x = 1 - y ∧ x = -1/2 ∧ y = 3/2) ∧
  -- System 2
  (∃ x y : ℚ, 5 * x + 6 * y = 4 ∧ 3 * x - 4 * y = 10) →
  (∃ x y : ℚ, 5 * x + 6 * y = 4 ∧ 3 * x - 4 * y = 10 ∧ x = 2 ∧ y = -1) :=
by sorry

end system_of_equations_solutions_l335_33586


namespace quadratic_sum_l335_33559

/-- Given a quadratic function f(x) = -2x^2 + 16x - 72, prove that when expressed
    in the form a(x+b)^2 + c, the sum of a, b, and c is equal to -46. -/
theorem quadratic_sum (x : ℝ) :
  ∃ (a b c : ℝ), 
    (∀ x, -2 * x^2 + 16 * x - 72 = a * (x + b)^2 + c) ∧
    (a + b + c = -46) := by
  sorry

end quadratic_sum_l335_33559


namespace parabola_shift_left_one_l335_33579

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * shift + p.b,
    c := p.a * shift^2 - p.b * shift + p.c }

theorem parabola_shift_left_one :
  let original := Parabola.mk 1 0 2
  let shifted := shift_parabola original 1
  shifted = Parabola.mk 1 2 3 := by
  sorry

#check parabola_shift_left_one

end parabola_shift_left_one_l335_33579


namespace angelina_walking_speed_l335_33595

/-- Angelina's walking problem -/
theorem angelina_walking_speed 
  (home_to_grocery : ℝ) 
  (grocery_to_gym : ℝ) 
  (time_difference : ℝ) 
  (h1 : home_to_grocery = 100) 
  (h2 : grocery_to_gym = 180) 
  (h3 : time_difference = 40) :
  let v := home_to_grocery / ((grocery_to_gym / 2) / time_difference + home_to_grocery)
  2 * v = 1 / 2 := by sorry

end angelina_walking_speed_l335_33595


namespace probability_three_heads_five_coins_l335_33570

theorem probability_three_heads_five_coins :
  let n : ℕ := 5  -- number of coins
  let k : ℕ := 3  -- number of heads we want
  let p : ℚ := 1/2  -- probability of getting heads on a single coin toss
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k) = 5/16 :=
by sorry

end probability_three_heads_five_coins_l335_33570


namespace extreme_value_implies_a_and_b_l335_33506

/-- A function f(x) = ax³ + bx has an extreme value of -2 at x = 1 -/
def has_extreme_value (a b : ℝ) : Prop :=
  let f := fun x : ℝ => a * x^3 + b * x
  f 1 = -2 ∧ (deriv f) 1 = 0

/-- Theorem: If f(x) = ax³ + bx has an extreme value of -2 at x = 1, then a = 1 and b = -3 -/
theorem extreme_value_implies_a_and_b :
  ∀ a b : ℝ, has_extreme_value a b → a = 1 ∧ b = -3 := by
  sorry

end extreme_value_implies_a_and_b_l335_33506


namespace four_students_three_communities_l335_33522

/-- The number of ways to distribute n students among k communities,
    where each student goes to exactly one community and each community
    receives at least one student. -/
def distribute_students (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 4 students among 3 communities
    results in 36 different arrangements. -/
theorem four_students_three_communities :
  distribute_students 4 3 = 36 := by sorry

end four_students_three_communities_l335_33522


namespace neil_cookies_l335_33541

theorem neil_cookies (total : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  remaining = 12 ∧ given_away = (2 : ℕ) * total / 5 ∧ remaining = (3 : ℕ) * total / 5 → total = 20 :=
by
  sorry

end neil_cookies_l335_33541


namespace number_line_movement_1_number_line_movement_2_number_line_movement_general_absolute_value_equality_l335_33573

-- Problem 1
theorem number_line_movement_1 (A B : ℝ) :
  A = -2 → B = A + 5 → B = 3 ∧ |B - A| = 5 := by sorry

-- Problem 2
theorem number_line_movement_2 (A B : ℝ) :
  A = 5 → B = A - 4 + 7 → B = 8 ∧ |B - A| = 3 := by sorry

-- Problem 3
theorem number_line_movement_general (a b c A B : ℝ) :
  A = a → B = A + b - c → B = a + b - c ∧ |B - A| = |b - c| := by sorry

-- Problem 4
theorem absolute_value_equality (x : ℝ) :
  |x + 1| = |x - 2| ↔ x = 1/2 := by sorry

end number_line_movement_1_number_line_movement_2_number_line_movement_general_absolute_value_equality_l335_33573


namespace expression_evaluation_l335_33534

theorem expression_evaluation : -1^5 + (-3)^0 - (Real.sqrt 2)^2 + 4 * |-(1/4)| = -1 := by
  sorry

end expression_evaluation_l335_33534


namespace total_books_and_magazines_l335_33536

def books_per_shelf : ℕ := 23
def magazines_per_shelf : ℕ := 61
def number_of_shelves : ℕ := 29

theorem total_books_and_magazines :
  books_per_shelf * number_of_shelves + magazines_per_shelf * number_of_shelves = 2436 := by
  sorry

end total_books_and_magazines_l335_33536


namespace quadratic_form_sum_l335_33533

theorem quadratic_form_sum (b c : ℝ) : 
  (∀ x, x^2 - 12*x + 49 = (x + b)^2 + c) → b + c = 7 := by
sorry

end quadratic_form_sum_l335_33533


namespace janas_height_l335_33516

/-- Given the heights of several people and their relationships, prove Jana's height. -/
theorem janas_height
  (kelly_jess : ℝ) -- Height difference between Kelly and Jess
  (jana_kelly : ℝ) -- Height difference between Jana and Kelly
  (jess_height : ℝ) -- Jess's height
  (jess_alex : ℝ) -- Height difference between Jess and Alex
  (alex_sam : ℝ) -- Height difference between Alex and Sam
  (h1 : jana_kelly = 5.5)
  (h2 : kelly_jess = -3.75)
  (h3 : jess_height = 72)
  (h4 : jess_alex = -1.25)
  (h5 : alex_sam = 0.5)
  : jess_height - kelly_jess + jana_kelly = 73.75 := by
  sorry

#check janas_height

end janas_height_l335_33516


namespace cube_configurations_l335_33542

/-- The group of rotations around axes through midpoints of opposite edges of a 2×2×2 cube -/
def EdgeRotationGroup : Type := Unit

/-- The order of the EdgeRotationGroup -/
def EdgeRotationGroupOrder : ℕ := 7

/-- The number of fixed configurations under the identity rotation -/
def FixedConfigurationsIdentity : ℕ := 56

/-- The number of fixed configurations under each edge rotation -/
def FixedConfigurationsEdge : ℕ := 0

/-- The number of edge rotations -/
def NumEdgeRotations : ℕ := 6

/-- Burnside's Lemma applied to the cube configuration problem -/
theorem cube_configurations (g : EdgeRotationGroup) :
  (FixedConfigurationsIdentity + NumEdgeRotations * FixedConfigurationsEdge) / EdgeRotationGroupOrder = 8 := by
  sorry

end cube_configurations_l335_33542


namespace self_inverse_solutions_l335_33580

def is_self_inverse (a d : ℝ) : Prop :=
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![a, 4; -9, d]
  M * M = 1

theorem self_inverse_solutions :
  ∃! n : ℕ, ∃ S : Finset (ℝ × ℝ),
    S.card = n ∧
    (∀ p : ℝ × ℝ, p ∈ S ↔ is_self_inverse p.1 p.2) :=
by sorry

end self_inverse_solutions_l335_33580


namespace badArrangementsCount_l335_33538

-- Define a type for circular arrangements
def CircularArrangement := List ℕ

-- Define what it means for an arrangement to be valid
def isValidArrangement (arr : CircularArrangement) : Prop :=
  arr.length = 6 ∧ arr.toFinset = {1, 2, 3, 4, 5, 6}

-- Define consecutive subsets in a circular arrangement
def consecutiveSubsets (arr : CircularArrangement) : List (List ℕ) :=
  sorry

-- Define what it means for an arrangement to be "bad"
def isBadArrangement (arr : CircularArrangement) : Prop :=
  ∃ n : ℕ, n ≥ 1 ∧ n ≤ 20 ∧ ∀ subset ∈ consecutiveSubsets arr, (subset.sum ≠ n)

-- Define equivalence of arrangements under rotation and reflection
def areEquivalentArrangements (arr1 arr2 : CircularArrangement) : Prop :=
  sorry

-- The main theorem
theorem badArrangementsCount :
  ∃ badArrs : List CircularArrangement,
    badArrs.length = 3 ∧
    (∀ arr ∈ badArrs, isValidArrangement arr ∧ isBadArrangement arr) ∧
    (∀ arr, isValidArrangement arr → isBadArrangement arr →
      ∃ badArr ∈ badArrs, areEquivalentArrangements arr badArr) :=
  sorry

end badArrangementsCount_l335_33538


namespace special_op_nine_ten_l335_33547

-- Define the ⊕ operation
def special_op (A B : ℚ) : ℚ := 1 / (A * B) + 1 / ((A + 1) * (B + 2))

-- State the theorem
theorem special_op_nine_ten :
  special_op 9 10 = 7 / 360 :=
by
  -- The proof goes here
  sorry

-- Additional fact given in the problem
axiom special_op_one_two : special_op 1 2 = 5 / 8

end special_op_nine_ten_l335_33547


namespace average_of_multiples_10_to_400_l335_33564

def multiples_of_10 (n : ℕ) : List ℕ :=
  (List.range ((400 - 10) / 10 + 1)).map (λ i => 10 * (i + 1))

theorem average_of_multiples_10_to_400 :
  (List.sum (multiples_of_10 400)) / (List.length (multiples_of_10 400)) = 205 := by
  sorry

end average_of_multiples_10_to_400_l335_33564


namespace mike_car_payment_l335_33515

def car_price : ℝ := 35000
def loan_amount : ℝ := 20000
def interest_rate : ℝ := 0.15
def loan_period : ℝ := 1

def total_amount_to_pay : ℝ := car_price + loan_amount * interest_rate * loan_period

theorem mike_car_payment : total_amount_to_pay = 38000 :=
by sorry

end mike_car_payment_l335_33515


namespace prob_adjacent_vertices_decagon_l335_33501

/-- A decagon is a polygon with 10 vertices -/
def Decagon : Type := Unit

/-- The number of vertices in a decagon -/
def num_vertices (d : Decagon) : ℕ := 10

/-- The number of pairs of adjacent vertices in a decagon -/
def num_adjacent_pairs (d : Decagon) : ℕ := 20

/-- The total number of ways to choose 2 distinct vertices from a decagon -/
def total_vertex_pairs (d : Decagon) : ℕ := (num_vertices d).choose 2

/-- The probability of choosing two adjacent vertices in a decagon -/
def prob_adjacent_vertices (d : Decagon) : ℚ :=
  (num_adjacent_pairs d : ℚ) / (total_vertex_pairs d : ℚ)

theorem prob_adjacent_vertices_decagon :
  ∀ d : Decagon, prob_adjacent_vertices d = 4/9 := by sorry

end prob_adjacent_vertices_decagon_l335_33501


namespace min_sum_squares_l335_33513

def f (x : ℝ) := |x + 1| - |x - 4|

def m₀ : ℝ := 5

theorem min_sum_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 3 * a + 4 * b + 5 * c = m₀) :
  a^2 + b^2 + c^2 ≥ 1/2 :=
sorry

end min_sum_squares_l335_33513


namespace regular_polygon_sides_l335_33558

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (n - 2) * 180 = 3 * 360 → n = 8 := by sorry

end regular_polygon_sides_l335_33558


namespace total_area_of_fields_l335_33514

/-- The total area of three fields with given dimensions -/
theorem total_area_of_fields (d₁ : ℝ) (l₂ w₂ : ℝ) (b₃ h₃ : ℝ) : 
  d₁ = 12 → l₂ = 15 → w₂ = 8 → b₃ = 18 → h₃ = 10 → 
  (d₁^2 / 2) + (l₂ * w₂) + (b₃ * h₃) = 372 := by
  sorry

end total_area_of_fields_l335_33514


namespace volleyball_team_selection_l335_33531

theorem volleyball_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) 
  (quadruplets_in_lineup : ℕ) :
  total_players = 17 →
  quadruplets = 4 →
  starters = 6 →
  quadruplets_in_lineup = 2 →
  (Nat.choose quadruplets quadruplets_in_lineup) * 
  (Nat.choose (total_players - quadruplets) (starters - quadruplets_in_lineup)) = 4290 :=
by sorry

end volleyball_team_selection_l335_33531


namespace friend_walking_problem_l335_33521

/-- Two friends walking on a trail problem -/
theorem friend_walking_problem (trail_length : ℝ) (meeting_distance : ℝ) 
  (h1 : trail_length = 43)
  (h2 : meeting_distance = 23)
  (h3 : meeting_distance < trail_length) :
  let rate_ratio := meeting_distance / (trail_length - meeting_distance)
  (rate_ratio - 1) * 100 = 15 := by
  sorry

end friend_walking_problem_l335_33521


namespace tan_alpha_equals_two_implies_expression_equals_three_l335_33567

theorem tan_alpha_equals_two_implies_expression_equals_three (α : Real) 
  (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (2 * Real.sin α - 3 * Real.cos α) = 3 := by
  sorry

end tan_alpha_equals_two_implies_expression_equals_three_l335_33567


namespace no_double_application_function_l335_33548

theorem no_double_application_function :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n + 2019 := by
  sorry

end no_double_application_function_l335_33548


namespace point_on_x_axis_l335_33563

/-- 
A point P with coordinates (3+a, a-5) lies on the x-axis in a Cartesian coordinate system.
Prove that a = 5.
-/
theorem point_on_x_axis (a : ℝ) : (a - 5 = 0) → a = 5 := by
  sorry

end point_on_x_axis_l335_33563


namespace problem_solution_l335_33540

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 5

-- Theorem statement
theorem problem_solution :
  -- Condition 1: y-1 is directly proportional to x+2
  (∃ k : ℝ, ∀ x y : ℝ, y = f x → y - 1 = k * (x + 2)) ∧
  -- Condition 2: When x=1, y=7
  (f 1 = 7) ∧
  -- Solution 1: The function f satisfies the conditions
  (∀ x : ℝ, f x = 2 * x + 5) ∧
  -- Solution 2: The point (-7/2, -2) lies on the graph of f
  (f (-7/2) = -2) := by
  sorry

end problem_solution_l335_33540


namespace dan_found_two_dimes_l335_33588

/-- The number of dimes Dan found -/
def dimes_found (barry_dimes dan_initial_dimes dan_final_dimes : ℕ) : ℕ :=
  dan_final_dimes - dan_initial_dimes

theorem dan_found_two_dimes :
  ∀ (barry_dimes dan_initial_dimes dan_final_dimes : ℕ),
    barry_dimes = 100 →
    dan_initial_dimes = barry_dimes / 2 →
    dan_final_dimes = 52 →
    dimes_found barry_dimes dan_initial_dimes dan_final_dimes = 2 :=
by
  sorry

end dan_found_two_dimes_l335_33588


namespace total_cost_calculation_l335_33553

/-- The total cost of sandwiches and sodas -/
def total_cost (sandwich_price : ℚ) (soda_price : ℚ) (sandwich_quantity : ℕ) (soda_quantity : ℕ) : ℚ :=
  sandwich_price * sandwich_quantity + soda_price * soda_quantity

/-- Theorem: The total cost of 2 sandwiches at $2.49 each and 4 sodas at $1.87 each is $12.46 -/
theorem total_cost_calculation :
  total_cost (249/100) (187/100) 2 4 = 1246/100 := by
  sorry

end total_cost_calculation_l335_33553


namespace travel_distance_proof_l335_33566

theorem travel_distance_proof (total_distance : ℝ) (plane_fraction : ℝ) (train_to_bus_ratio : ℝ) 
  (h1 : total_distance = 1800)
  (h2 : plane_fraction = 1/3)
  (h3 : train_to_bus_ratio = 2/3) : 
  ∃ (bus_distance : ℝ), 
    bus_distance = 720 ∧ 
    plane_fraction * total_distance + train_to_bus_ratio * bus_distance + bus_distance = total_distance :=
by sorry

end travel_distance_proof_l335_33566


namespace diophantine_equation_solution_l335_33523

theorem diophantine_equation_solution :
  ∀ x y z : ℕ,
    x ≤ y →
    x^2 + y^2 = 3 * 2016^z + 77 →
    ((x = 4 ∧ y = 8 ∧ z = 0) ∨
     (x = 14 ∧ y = 49 ∧ z = 1) ∨
     (x = 35 ∧ y = 70 ∧ z = 1)) :=
by sorry

end diophantine_equation_solution_l335_33523


namespace expected_winnings_is_four_thirds_l335_33552

/-- Represents the faces of the coin -/
inductive Face
  | one
  | two
  | three
  | four

/-- The probability of each face appearing -/
def probability (f : Face) : ℚ :=
  match f with
  | Face.one => 5/12
  | Face.two => 1/3
  | Face.three => 1/6
  | Face.four => 1/12

/-- The winnings associated with each face -/
def winnings (f : Face) : ℤ :=
  match f with
  | Face.one => 2
  | Face.two => 0
  | Face.three => -2
  | Face.four => 10

/-- The expected winnings when tossing the coin -/
def expectedWinnings : ℚ :=
  (probability Face.one * winnings Face.one) +
  (probability Face.two * winnings Face.two) +
  (probability Face.three * winnings Face.three) +
  (probability Face.four * winnings Face.four)

theorem expected_winnings_is_four_thirds :
  expectedWinnings = 4/3 := by
  sorry

end expected_winnings_is_four_thirds_l335_33552


namespace inequality_proof_l335_33551

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end inequality_proof_l335_33551


namespace complement_intersection_theorem_l335_33560

def U : Finset ℕ := {0,1,2,3,4,5,6,7,8,9}
def A : Finset ℕ := {0,1,3,5,8}
def B : Finset ℕ := {2,4,5,6,8}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {7,9} := by sorry

end complement_intersection_theorem_l335_33560


namespace quadratic_trinomial_prime_square_solution_l335_33500

/-- A quadratic trinomial function -/
def f (x : ℤ) : ℤ := 2 * x^2 - x - 36

/-- Predicate to check if a number is prime -/
def is_prime (p : ℕ) : Prop := Nat.Prime p

/-- The main theorem statement -/
theorem quadratic_trinomial_prime_square_solution :
  ∃! x : ℤ, ∃ p : ℕ, is_prime p ∧ f x = p^2 ∧ x = 13 := by sorry

end quadratic_trinomial_prime_square_solution_l335_33500


namespace haley_seeds_l335_33520

/-- The number of seeds Haley planted in the big garden -/
def big_garden_seeds : ℕ := 35

/-- The number of small gardens Haley had -/
def small_gardens : ℕ := 7

/-- The number of seeds Haley planted in each small garden -/
def seeds_per_small_garden : ℕ := 3

/-- The total number of seeds Haley started with -/
def total_seeds : ℕ := big_garden_seeds + small_gardens * seeds_per_small_garden

theorem haley_seeds : total_seeds = 56 := by
  sorry

end haley_seeds_l335_33520


namespace calculate_expression_l335_33571

theorem calculate_expression : 6 * (1/3 - 1/2) - 3^2 / (-12) = -1/4 := by
  sorry

end calculate_expression_l335_33571


namespace reciprocal_sum_equals_five_l335_33549

theorem reciprocal_sum_equals_five (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x + y = 5 * x * y) (h2 : x = 2 * y) : 
  1 / x + 1 / y = 5 := by
  sorry

end reciprocal_sum_equals_five_l335_33549


namespace imaginary_unit_power_sum_l335_33578

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_unit_power_sum : i^2 + i^4 = 0 := by sorry

end imaginary_unit_power_sum_l335_33578


namespace num_boys_in_class_l335_33598

-- Define the number of girls in the class
def num_girls : ℕ := 10

-- Define the number of ways to select 1 girl and 2 boys
def num_selections : ℕ := 1050

-- Define the function to calculate the number of ways to select 1 girl and 2 boys
def selection_ways (n : ℕ) : ℕ := num_girls * (n * (n - 1) / 2)

-- Theorem statement
theorem num_boys_in_class : ∃ (n : ℕ), n > 0 ∧ selection_ways n = num_selections :=
sorry

end num_boys_in_class_l335_33598


namespace burger_distribution_theorem_l335_33525

/-- Represents the burger distribution problem --/
def burger_distribution (total_burgers : ℕ) (num_friends : ℕ) (slices_per_burger : ℕ) 
  (slices_friend3 : ℕ) (slices_friend4 : ℕ) (slices_era : ℕ) : Prop :=
  let total_slices := total_burgers * slices_per_burger
  let slices_for_friends12 := total_slices - (slices_friend3 + slices_friend4 + slices_era)
  slices_for_friends12 = 3

/-- Theorem stating that under the given conditions, the first and second friends get 3 slices combined --/
theorem burger_distribution_theorem : 
  burger_distribution 5 4 2 3 3 1 := by sorry

end burger_distribution_theorem_l335_33525


namespace garrison_size_l335_33561

/-- Given a garrison with provisions and reinforcements, calculate the initial number of men. -/
theorem garrison_size (initial_days : ℕ) (reinforcement_arrival : ℕ) (remaining_days : ℕ) (reinforcement_size : ℕ) : 
  initial_days = 54 →
  reinforcement_arrival = 15 →
  remaining_days = 20 →
  reinforcement_size = 1900 →
  ∃ (initial_men : ℕ), 
    initial_men * (initial_days - reinforcement_arrival) = 
    (initial_men + reinforcement_size) * remaining_days ∧
    initial_men = 2000 :=
by sorry

end garrison_size_l335_33561


namespace problem_solution_l335_33512

theorem problem_solution : (88 * 707 - 38 * 707) / 1414 = 25 := by
  have h : 1414 = 707 * 2 := by sorry
  sorry

end problem_solution_l335_33512


namespace complex_cube_root_l335_33529

theorem complex_cube_root (x y d : ℤ) (z : ℂ) : 
  x > 0 → y > 0 → z = x + y * Complex.I → z^3 = -54 + d * Complex.I → z = 3 + 3 * Complex.I := by
  sorry

end complex_cube_root_l335_33529


namespace f_properties_l335_33526

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) + (2 - a) * Real.exp x - a * x + a * Real.exp 1 / 2

theorem f_properties (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) ∨
  (a > 0 ∧ ∃ x_min, ∀ x, f a x ≥ f a x_min) ∧
  (∀ x, f a x ≥ 0 ↔ a ∈ Set.Icc 0 (2 * Real.exp 1)) :=
sorry

end f_properties_l335_33526


namespace larger_number_proof_l335_33502

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1390)
  (h2 : L = 6 * S + 15) : 
  L = 1665 := by
  sorry

end larger_number_proof_l335_33502


namespace inequality_proof_l335_33507

theorem inequality_proof (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  x^3 + x*y^2 + 2*x*y ≤ 2*x^2*y + x^2 + x + y :=
by sorry

end inequality_proof_l335_33507


namespace jason_music_store_spending_l335_33587

/-- The price of the flute Jason bought -/
def flute_price : ℚ := 142.46

/-- The price of the music stand Jason bought -/
def stand_price : ℚ := 8.89

/-- The price of the song book Jason bought -/
def book_price : ℚ := 7

/-- The total amount Jason spent at the music store -/
def total_spent : ℚ := flute_price + stand_price + book_price

/-- Theorem stating that the total amount Jason spent is $158.35 -/
theorem jason_music_store_spending :
  total_spent = 158.35 := by sorry

end jason_music_store_spending_l335_33587


namespace inequality_proof_l335_33596

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1 := by sorry

end inequality_proof_l335_33596


namespace moving_circle_trajectory_l335_33568

-- Define the two fixed circles
def circle_M (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

-- Define the trajectory hyperbola
def trajectory_hyperbola (x y : ℝ) : Prop := 4*(x + 2)^2 - y^2 = 1

-- Define the concept of a moving circle being externally tangent to two fixed circles
def externally_tangent (x y : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
  (∃ (x_m y_m : ℝ), circle_M x_m y_m ∧ (x - x_m)^2 + (y - y_m)^2 = (r + 1)^2) ∧
  (∃ (x_n y_n : ℝ), circle_N x_n y_n ∧ (x - x_n)^2 + (y - y_n)^2 = (r + 2)^2)

-- The main theorem
theorem moving_circle_trajectory :
  ∀ (x y : ℝ), externally_tangent x y → trajectory_hyperbola x y ∧ x < -2 :=
sorry

end moving_circle_trajectory_l335_33568


namespace combined_cost_theorem_l335_33572

def wallet_cost : ℕ := 22
def purse_cost : ℕ := 4 * wallet_cost - 3

theorem combined_cost_theorem : wallet_cost + purse_cost = 107 := by
  sorry

end combined_cost_theorem_l335_33572


namespace integer_property_l335_33537

theorem integer_property (k : ℕ) : k ≥ 3 → (
  (∃ m n : ℕ, 1 < m ∧ m < k ∧
              1 < n ∧ n < k ∧
              Nat.gcd m k = 1 ∧
              Nat.gcd n k = 1 ∧
              m + n > k ∧
              k ∣ (m - 1) * (n - 1))
  ↔ (k = 15 ∨ k = 30)
) := by
  sorry

end integer_property_l335_33537


namespace tangent_parallel_to_x_axis_l335_33597

open Real

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := (log x) / x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := (1 - log x) / (x^2)

theorem tangent_parallel_to_x_axis :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ f_derivative x₀ = 0 → f x₀ = 1/Real.exp 1 := by
  sorry

end tangent_parallel_to_x_axis_l335_33597


namespace identity_iff_annihilator_l335_33594

variable (R : Type) [Fintype R] [CommRing R]

def has_multiplicative_identity (R : Type) [Ring R] : Prop :=
  ∃ e : R, ∀ x : R, e * x = x ∧ x * e = x

def annihilator_is_zero (R : Type) [Ring R] : Prop :=
  ∀ a : R, (∀ x : R, a * x = 0) → a = 0

theorem identity_iff_annihilator (R : Type) [Fintype R] [CommRing R] :
  has_multiplicative_identity R ↔ annihilator_is_zero R :=
sorry

end identity_iff_annihilator_l335_33594


namespace initial_balloons_l335_33527

theorem initial_balloons (x : ℕ) : 
  Odd x ∧ 
  (x / 3 : ℚ) + 10 = 45 → 
  x = 105 := by
sorry

end initial_balloons_l335_33527


namespace divisors_of_720_l335_33599

theorem divisors_of_720 : Finset.card (Nat.divisors 720) = 30 := by
  sorry

end divisors_of_720_l335_33599


namespace fraction_simplification_l335_33544

theorem fraction_simplification (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) (h4 : x ≠ 5) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 4) * (x - 2)) := by
sorry

end fraction_simplification_l335_33544


namespace sum_abs_roots_quadratic_l335_33565

theorem sum_abs_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * r₁^2 + b * r₁ + c = 0 ∧ 
  a * r₂^2 + b * r₂ + c = 0 →
  |r₁| + |r₂| = 6 :=
by sorry

end sum_abs_roots_quadratic_l335_33565


namespace angle_A_measure_perimeter_range_l335_33508

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  t.a / (Real.sqrt 3 * Real.cos t.A) = t.c / Real.sin t.C

-- Theorem for angle A
theorem angle_A_measure (t : Triangle) (h : given_condition t) : t.A = π / 3 :=
sorry

-- Theorem for perimeter range
theorem perimeter_range (t : Triangle) (h1 : given_condition t) (h2 : t.a = 6) :
  12 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 18 :=
sorry

end angle_A_measure_perimeter_range_l335_33508


namespace amazon_tide_problem_l335_33528

theorem amazon_tide_problem (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin (2 * x + φ)) →
  (abs φ < π / 2) →
  (∀ x, f (x - π / 3) = -f (-x - π / 3)) →
  (φ = -π / 3) ∧
  (∀ x, f (5 * π / 12 + x) = f (5 * π / 12 - x)) ∧
  (∀ x ∈ Set.Icc (-π / 3) (-π / 6), ∀ y ∈ Set.Icc (-π / 3) (-π / 6), x < y → f x > f y) ∧
  (∃ x ∈ Set.Ioo 0 (π / 2), (deriv f) x = 0) :=
by sorry

end amazon_tide_problem_l335_33528


namespace tetrahedrons_count_l335_33557

/-- The number of tetrahedrons formed by choosing 4 vertices from a triangular prism -/
def tetrahedrons_from_prism : ℕ :=
  Nat.choose 6 4 - 3

/-- Theorem stating that the number of tetrahedrons is 12 -/
theorem tetrahedrons_count : tetrahedrons_from_prism = 12 := by
  sorry

end tetrahedrons_count_l335_33557


namespace reading_book_cost_is_12_l335_33556

/-- The cost of a reading book given the total amount available and number of students -/
def reading_book_cost (total_amount : ℕ) (num_students : ℕ) : ℚ :=
  (total_amount : ℚ) / (num_students : ℚ)

/-- Theorem: The cost of each reading book is $12 -/
theorem reading_book_cost_is_12 :
  reading_book_cost 360 30 = 12 := by
  sorry

end reading_book_cost_is_12_l335_33556


namespace rhombus_properties_l335_33591

-- Define a rhombus
structure Rhombus (V : Type*) [NormedAddCommGroup V] :=
  (A B C D : V)
  (is_rhombus : True)  -- This is a placeholder for the rhombus property

-- Define the theorem
theorem rhombus_properties {V : Type*} [NormedAddCommGroup V] (r : Rhombus V) :
  (‖r.A - r.B‖ = ‖r.B - r.C‖) ∧ 
  (‖r.A - r.B - (r.C - r.D)‖ = ‖r.A - r.D + (r.B - r.C)‖) ∧
  (‖r.A - r.C‖^2 + ‖r.B - r.D‖^2 = 4 * ‖r.A - r.B‖^2) :=
by sorry

end rhombus_properties_l335_33591


namespace negation_of_all_birds_can_fly_l335_33590

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (bird : U → Prop)
variable (can_fly : U → Prop)

-- State the theorem
theorem negation_of_all_birds_can_fly :
  (¬ ∀ (x : U), bird x → can_fly x) ↔ (∃ (x : U), bird x ∧ ¬ can_fly x) :=
by sorry

end negation_of_all_birds_can_fly_l335_33590


namespace divide_money_l335_33509

theorem divide_money (total : ℝ) (a b c : ℝ) : 
  total = 364 →
  a = (1/2) * b →
  b = (1/2) * c →
  a + b + c = total →
  c = 208 := by
sorry

end divide_money_l335_33509


namespace ellipse_major_minor_distance_l335_33577

/-- An ellipse with equation 4(x+2)^2 + 16y^2 = 64 -/
structure Ellipse where
  eq : ∀ x y : ℝ, 4 * (x + 2)^2 + 16 * y^2 = 64

/-- Point C is an endpoint of the major axis -/
def C (e : Ellipse) : ℝ × ℝ := sorry

/-- Point D is an endpoint of the minor axis -/
def D (e : Ellipse) : ℝ × ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_major_minor_distance (e : Ellipse) : 
  distance (C e) (D e) = 2 * Real.sqrt 5 := by sorry

end ellipse_major_minor_distance_l335_33577


namespace semicircle_perimeter_approx_l335_33510

/-- The perimeter of a semicircle with radius 6.83 cm is approximately 35.12 cm. -/
theorem semicircle_perimeter_approx : 
  let r : Real := 6.83
  let perimeter : Real := π * r + 2 * r
  ∃ ε > 0, abs (perimeter - 35.12) < ε :=
by sorry

end semicircle_perimeter_approx_l335_33510


namespace tamara_brownie_earnings_l335_33530

/-- Calculates the earnings from selling brownies --/
def brownie_earnings (pans : ℕ) (pieces_per_pan : ℕ) (price_per_piece : ℕ) : ℕ :=
  pans * pieces_per_pan * price_per_piece

/-- Proves that Tamara's earnings from brownies equal $32 --/
theorem tamara_brownie_earnings :
  brownie_earnings 2 8 2 = 32 := by
  sorry

end tamara_brownie_earnings_l335_33530


namespace sum_m_n_eq_67_l335_33589

-- Define the point R
def R : ℝ × ℝ := (8, 6)

-- Define the lines
def line1 (x y : ℝ) : Prop := 8 * y = 15 * x
def line2 (x y : ℝ) : Prop := 10 * y = 3 * x

-- Define points P and Q
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Define the conditions
axiom P_on_line1 : line1 P.1 P.2
axiom Q_on_line2 : line2 Q.1 Q.2
axiom R_is_midpoint : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the length of PQ
def PQ_length : ℝ := sorry

-- Define m and n as positive integers
def m : ℕ+ := sorry
def n : ℕ+ := sorry

-- PQ length is equal to m/n
axiom PQ_length_eq_m_div_n : PQ_length = m.val / n.val

-- m and n are coprime
axiom m_n_coprime : Nat.Coprime m.val n.val

-- Theorem to prove
theorem sum_m_n_eq_67 : m.val + n.val = 67 := sorry

end sum_m_n_eq_67_l335_33589


namespace remaining_three_digit_numbers_l335_33582

/-- The number of three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The number of three-digit numbers where the first and last digits are the same
    but the middle digit is different -/
def excluded_numbers : ℕ := 81

/-- The number of valid three-digit numbers after exclusion -/
def valid_numbers : ℕ := total_three_digit_numbers - excluded_numbers

theorem remaining_three_digit_numbers : valid_numbers = 819 := by
  sorry

end remaining_three_digit_numbers_l335_33582


namespace typing_orders_count_l335_33535

/-- The number of letters to be typed -/
def n : ℕ := 9

/-- The index of the letter that has already been typed -/
def typed_letter : ℕ := 8

/-- The set of possible remaining letters after the typed_letter has been removed -/
def remaining_letters : Finset ℕ := Finset.filter (λ x => x ≠ typed_letter ∧ x ≤ n) (Finset.range (n + 1))

/-- The number of possible typing orders for the remaining letters -/
def num_typing_orders : ℕ :=
  (Finset.range 8).sum (λ k => Nat.choose 7 k * (k + 2))

theorem typing_orders_count :
  num_typing_orders = 704 :=
sorry

end typing_orders_count_l335_33535


namespace library_bookshelf_selection_l335_33532

/-- Represents a bookshelf with three tiers -/
structure Bookshelf :=
  (tier1 : ℕ)
  (tier2 : ℕ)
  (tier3 : ℕ)

/-- The number of ways to select a book from a bookshelf -/
def selectBook (b : Bookshelf) : ℕ := b.tier1 + b.tier2 + b.tier3

/-- Theorem: The number of ways to select a book from the given bookshelf is 16 -/
theorem library_bookshelf_selection :
  ∃ (b : Bookshelf), b.tier1 = 3 ∧ b.tier2 = 5 ∧ b.tier3 = 8 ∧ selectBook b = 16 := by
  sorry


end library_bookshelf_selection_l335_33532


namespace subtract_negative_l335_33569

theorem subtract_negative : -2 - (-3) = 1 := by
  sorry

end subtract_negative_l335_33569


namespace hexagon_area_error_l335_33517

/-- If there's an 8% error in excess while measuring the sides of a hexagon, 
    the percentage of error in the estimated area is 16.64%. -/
theorem hexagon_area_error (s : ℝ) (h : s > 0) : 
  let true_area := (3 * Real.sqrt 3 / 2) * s^2
  let measured_side := 1.08 * s
  let estimated_area := (3 * Real.sqrt 3 / 2) * measured_side^2
  (estimated_area - true_area) / true_area * 100 = 16.64 := by
sorry

end hexagon_area_error_l335_33517


namespace perfect_squares_condition_l335_33562

theorem perfect_squares_condition (k : ℤ) : 
  (∃ n : ℤ, k + 1 = n^2) ∧ (∃ m : ℤ, 16*k + 1 = m^2) ↔ k = 0 ∨ k = 3 :=
by sorry

end perfect_squares_condition_l335_33562


namespace ef_fraction_of_gh_l335_33503

/-- Given a line segment GH with points E and F on it, prove that EF is 5/11 of GH -/
theorem ef_fraction_of_gh (G E F H : ℝ) : 
  (E ≤ F) → -- E is before or at F on the line
  (F ≤ H) → -- F is before or at H on the line
  (G ≤ E) → -- G is before or at E on the line
  (G - E = 5 * (H - E)) → -- GE = 5 * EH
  (G - F = 10 * (H - F)) → -- GF = 10 * FH
  F - E = 5 / 11 * (H - G) := by
sorry

end ef_fraction_of_gh_l335_33503


namespace share_a_correct_l335_33574

/-- Calculates the share of profit for partner A given the investment details and total profit -/
def calculate_share_a (initial_a initial_b : ℕ) (withdraw_a advance_b : ℕ) (total_months : ℕ) (change_month : ℕ) (total_profit : ℕ) : ℕ :=
  let investment_months_a := initial_a * change_month + (initial_a - withdraw_a) * (total_months - change_month)
  let investment_months_b := initial_b * change_month + (initial_b + advance_b) * (total_months - change_month)
  let total_investment_months := investment_months_a + investment_months_b
  (investment_months_a * total_profit) / total_investment_months

theorem share_a_correct (initial_a initial_b : ℕ) (withdraw_a advance_b : ℕ) (total_months : ℕ) (change_month : ℕ) (total_profit : ℕ) :
  initial_a = 3000 →
  initial_b = 4000 →
  withdraw_a = 1000 →
  advance_b = 1000 →
  total_months = 12 →
  change_month = 8 →
  total_profit = 840 →
  calculate_share_a initial_a initial_b withdraw_a advance_b total_months change_month total_profit = 320 :=
by sorry

end share_a_correct_l335_33574


namespace equation_solution_l335_33504

theorem equation_solution : 
  ∃! x : ℝ, |Real.sqrt ((x - 2)^2) - 1| = x :=
by
  -- The proof goes here
  sorry

end equation_solution_l335_33504


namespace area_greater_than_four_thirds_e_cubed_greater_than_twenty_l335_33505

-- Define the area function S(t)
noncomputable def S (t : ℝ) : ℝ :=
  ∫ x in (0)..(1/t), (Real.exp (t^2 * x))

-- State the theorem
theorem area_greater_than_four_thirds :
  ∀ t > 0, S t > 4/3 :=
by
  sorry

-- Additional fact that can be used in the proof
theorem e_cubed_greater_than_twenty : Real.exp 3 > 20 :=
by
  sorry

end area_greater_than_four_thirds_e_cubed_greater_than_twenty_l335_33505


namespace sum_of_roots_is_18_l335_33519

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the symmetry property of g
def is_symmetric_about_3 (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (3 + x) = g (3 - x)

-- Define the property of having exactly six distinct real roots
def has_six_distinct_roots (g : ℝ → ℝ) : Prop :=
  ∃ (a b c d e f : ℝ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
                        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
                        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
                        d ≠ e ∧ d ≠ f ∧
                        e ≠ f) ∧
  (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0 ∧ g e = 0 ∧ g f = 0) ∧
  (∀ x : ℝ, g x = 0 → (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e ∨ x = f))

-- Theorem statement
theorem sum_of_roots_is_18 (g : ℝ → ℝ) 
  (h1 : is_symmetric_about_3 g) 
  (h2 : has_six_distinct_roots g) :
  ∃ (a b c d e f : ℝ), 
    (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0 ∧ g e = 0 ∧ g f = 0) ∧
    (a + b + c + d + e + f = 18) :=
sorry

end sum_of_roots_is_18_l335_33519


namespace regular_octagon_interior_angle_l335_33539

/-- The measure of one interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle : ℝ := by
  sorry

end regular_octagon_interior_angle_l335_33539


namespace larger_cross_section_distance_l335_33581

/-- Right octagonal pyramid with two parallel cross sections -/
structure OctagonalPyramid where
  /-- Ratio of areas of two cross sections -/
  area_ratio : ℝ
  /-- Distance between the two cross sections -/
  cross_section_distance : ℝ

/-- Theorem about the distance of the larger cross section from the apex -/
theorem larger_cross_section_distance (pyramid : OctagonalPyramid) 
  (h_ratio : pyramid.area_ratio = 4 / 9)
  (h_distance : pyramid.cross_section_distance = 10) :
  ∃ (apex_distance : ℝ), apex_distance = 30 := by
  sorry

end larger_cross_section_distance_l335_33581


namespace max_value_of_f_l335_33518

/-- The function f(x) = x^3 - 3ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + 2

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

theorem max_value_of_f (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f a x ≥ f a 2) →
  (∃ x : ℝ, f a x = 18 ∧ ∀ y : ℝ, f a y ≤ f a x) :=
sorry

end max_value_of_f_l335_33518


namespace book_cost_problem_l335_33555

/-- Given two books with a total cost of 300 Rs, if one is sold at a 15% loss
    and the other at a 19% gain, and both are sold at the same price,
    then the cost of the book sold at a loss is 175 Rs. -/
theorem book_cost_problem (C₁ C₂ SP : ℝ) : 
  C₁ + C₂ = 300 →
  SP = 0.85 * C₁ →
  SP = 1.19 * C₂ →
  C₁ = 175 := by
sorry

end book_cost_problem_l335_33555


namespace complex_fourth_power_problem_l335_33575

theorem complex_fourth_power_problem : ∃ (d : ℤ), (1 + 3*I : ℂ)^4 = 82 + d*I := by sorry

end complex_fourth_power_problem_l335_33575


namespace line_plane_perpendicular_parallel_l335_33593

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicular_parallel 
  (l : Line) (m : Line) (α : Plane) (β : Plane)
  (h1 : perpendicular l α)
  (h2 : contained_in m β) :
  (parallel α β → perpendicular_lines l m) ∧
  (parallel_lines l m → perpendicular_planes α β) :=
sorry

end line_plane_perpendicular_parallel_l335_33593


namespace house_rent_fraction_l335_33550

def salary : ℚ := 140000

def food_fraction : ℚ := 1/5
def clothes_fraction : ℚ := 3/5
def remaining_amount : ℚ := 14000

theorem house_rent_fraction :
  ∃ (house_rent_fraction : ℚ),
    house_rent_fraction * salary + food_fraction * salary + clothes_fraction * salary + remaining_amount = salary ∧
    house_rent_fraction = 1/10 := by
  sorry

end house_rent_fraction_l335_33550


namespace committee_probability_l335_33592

def totalStudents : ℕ := 18
def numBoys : ℕ := 8
def numGirls : ℕ := 10
def committeeSize : ℕ := 4

theorem committee_probability : 
  let totalCommittees := Nat.choose totalStudents committeeSize
  let allBoysCommittees := Nat.choose numBoys committeeSize
  let allGirlsCommittees := Nat.choose numGirls committeeSize
  let probabilityAtLeastOneBoyOneGirl := 1 - (allBoysCommittees + allGirlsCommittees : ℚ) / totalCommittees
  probabilityAtLeastOneBoyOneGirl = 139 / 153 := by
  sorry

end committee_probability_l335_33592


namespace max_value_fraction_l335_33543

theorem max_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 3) :
  (∀ x' y', -5 ≤ x' ∧ x' ≤ -3 ∧ 1 ≤ y' ∧ y' ≤ 3 → (x' + 2*y') / x' ≤ (x + 2*y) / x) →
  (x + 2*y) / x = -1/5 :=
by sorry

end max_value_fraction_l335_33543


namespace village_news_spread_l335_33554

/-- Represents the village and its news spreading dynamics -/
structure Village where
  inhabitants : Finset Nat
  acquaintances : Nat → Finset Nat
  news_spreads : ∀ (n : Nat), n ∈ inhabitants → ∀ (m : Nat), m ∈ acquaintances n → m ∈ inhabitants

/-- A village satisfies the problem conditions -/
def ValidVillage (v : Village) : Prop :=
  v.inhabitants.card = 1000 ∧
  (∀ (news : Nat → Prop) (start : Nat → Prop),
    (∃ (d : Nat), ∀ (n : Nat), n ∈ v.inhabitants → news n))

/-- Represents the spread of news over time -/
def NewsSpread (v : Village) (informed : Finset Nat) (days : Nat) : Finset Nat :=
  sorry

/-- The main theorem to be proved -/
theorem village_news_spread (v : Village) (h : ValidVillage v) :
  ∃ (informed : Finset Nat),
    informed.card = 90 ∧
    ∀ (n : Nat), n ∈ v.inhabitants →
      n ∈ NewsSpread v informed 10 :=
sorry

end village_news_spread_l335_33554


namespace normal_vector_perpendicular_cosine_angle_between_lines_distance_point_to_line_l335_33546

/-- A line in 2D space represented by the equation Ax + By + C = 0 -/
structure Line2D where
  A : ℝ
  B : ℝ
  C : ℝ
  nonzero : A ≠ 0 ∨ B ≠ 0

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The vector perpendicular to a line is its normal vector -/
theorem normal_vector_perpendicular (l : Line2D) :
  let dir_vec := (-l.B, l.A)
  let normal_vec := (l.A, l.B)
  (dir_vec.1 * normal_vec.1 + dir_vec.2 * normal_vec.2 = 0) :=
sorry

/-- The cosine of the angle between two intersecting lines -/
theorem cosine_angle_between_lines (l₁ l₂ : Line2D) :
  let cos_theta := |(l₁.A * l₂.A + l₁.B * l₂.B) / (Real.sqrt (l₁.A^2 + l₁.B^2) * Real.sqrt (l₂.A^2 + l₂.B^2))|
  (0 ≤ cos_theta ∧ cos_theta ≤ 1) :=
sorry

/-- The distance from a point to a line -/
theorem distance_point_to_line (p : Point2D) (l : Line2D) :
  let d := |l.A * p.x + l.B * p.y + l.C| / Real.sqrt (l.A^2 + l.B^2)
  (d ≥ 0) :=
sorry

end normal_vector_perpendicular_cosine_angle_between_lines_distance_point_to_line_l335_33546


namespace unique_integer_divisible_by_15_with_sqrt_between_28_and_28_5_l335_33511

theorem unique_integer_divisible_by_15_with_sqrt_between_28_and_28_5 :
  ∃! n : ℕ+, (15 ∣ n) ∧ (28 < (n : ℝ).sqrt) ∧ ((n : ℝ).sqrt < 28.5) := by
  sorry

end unique_integer_divisible_by_15_with_sqrt_between_28_and_28_5_l335_33511


namespace sequence_term_l335_33524

theorem sequence_term (n : ℕ) (a : ℕ → ℝ) : 
  (∀ k, a k = Real.sqrt (2 * k - 1)) → 
  a 23 = 3 * Real.sqrt 5 :=
by
  sorry

end sequence_term_l335_33524


namespace problem_solution_l335_33545

def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {-a, a^2 + 3}

theorem problem_solution (a : ℝ) : A ∪ B a = {1, 2, 4} → a = -1 := by
  sorry

end problem_solution_l335_33545


namespace S_5_equals_31_l335_33576

-- Define the sequence and its partial sum
def S (n : ℕ) : ℕ := 2^n - 1

-- State the theorem
theorem S_5_equals_31 : S 5 = 31 := by
  sorry

end S_5_equals_31_l335_33576


namespace count_single_colored_face_for_given_cube_l335_33585

/-- Represents a cube cut in half and then into smaller cubes --/
structure CutCube where
  half_size : ℕ -- The number of small cubes along one edge of a half
  total_small_cubes : ℕ -- Total number of small cubes in each half

/-- Calculates the number of small cubes with only one colored face --/
def count_single_colored_face (c : CutCube) : ℕ :=
  4 * (c.half_size - 2) * (c.half_size - 2) * 2

/-- The theorem to be proved --/
theorem count_single_colored_face_for_given_cube :
  ∃ (c : CutCube), c.half_size = 4 ∧ c.total_small_cubes = 64 ∧ count_single_colored_face c = 32 :=
sorry

end count_single_colored_face_for_given_cube_l335_33585
