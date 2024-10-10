import Mathlib

namespace rectangle_area_proof_l3694_369432

/-- Calculates the area of a rectangular plot given its breadth and the fact that its length is thrice its breadth -/
def rectangle_area (breadth : ℝ) : ℝ :=
  3 * breadth * breadth

/-- Proves that the area of a rectangular plot with breadth 26 meters and length thrice its breadth is 2028 square meters -/
theorem rectangle_area_proof : rectangle_area 26 = 2028 := by
  sorry

end rectangle_area_proof_l3694_369432


namespace initial_marbles_l3694_369489

theorem initial_marbles (lost_marbles current_marbles : ℕ) 
  (h1 : lost_marbles = 7)
  (h2 : current_marbles = 9) :
  lost_marbles + current_marbles = 16 := by
  sorry

end initial_marbles_l3694_369489


namespace basketball_lineup_combinations_l3694_369476

/-- The number of ways to choose a starting lineup from a basketball team -/
def number_of_lineups (total_players : ℕ) (center_players : ℕ) (lineup_size : ℕ) : ℕ :=
  center_players * (total_players - 1) * (total_players - 2) * (total_players - 3)

/-- Theorem stating that for a team of 12 players with 4 centers, there are 3960 ways to choose a starting lineup of 4 players -/
theorem basketball_lineup_combinations :
  number_of_lineups 12 4 4 = 3960 := by
  sorry

end basketball_lineup_combinations_l3694_369476


namespace unique_occurrence_l3694_369431

-- Define the sequence type
def IntegerSequence := ℕ → ℤ

-- Define the property of having infinitely many positive and negative elements
def HasInfinitelyManyPositiveAndNegative (a : IntegerSequence) : Prop :=
  (∀ N : ℕ, ∃ n > N, a n > 0) ∧ (∀ N : ℕ, ∃ n > N, a n < 0)

-- Define the property of distinct remainders
def HasDistinctRemainders (a : IntegerSequence) : Prop :=
  ∀ n : ℕ, ∀ i j : ℕ, i < n → j < n → i ≠ j → a i % n ≠ a j % n

-- The main theorem
theorem unique_occurrence (a : IntegerSequence) 
  (h1 : HasInfinitelyManyPositiveAndNegative a)
  (h2 : HasDistinctRemainders a)
  (k : ℤ) : 
  ∃! n : ℕ, a n = k :=
sorry

end unique_occurrence_l3694_369431


namespace lisa_quiz_goal_impossible_l3694_369488

theorem lisa_quiz_goal_impossible (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (earned_as : ℕ) : 
  total_quizzes = 60 → 
  goal_percentage = 9/10 → 
  completed_quizzes = 40 → 
  earned_as = 30 → 
  ¬ ∃ (remaining_non_as : ℕ), 
    earned_as + (total_quizzes - completed_quizzes - remaining_non_as) ≥ 
    ⌈goal_percentage * total_quizzes⌉ := by
  sorry

#check lisa_quiz_goal_impossible

end lisa_quiz_goal_impossible_l3694_369488


namespace smallest_integer_with_two_cube_sum_representations_l3694_369430

def is_sum_of_three_cubes (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ n = a^3 + b^3 + c^3

def has_two_representations (n : ℕ) : Prop :=
  ∃ a₁ b₁ c₁ a₂ b₂ c₂ : ℕ,
    a₁ > 0 ∧ b₁ > 0 ∧ c₁ > 0 ∧
    a₂ > 0 ∧ b₂ > 0 ∧ c₂ > 0 ∧
    n = a₁^3 + b₁^3 + c₁^3 ∧
    n = a₂^3 + b₂^3 + c₂^3 ∧
    (a₁, b₁, c₁) ≠ (a₂, b₂, c₂)

theorem smallest_integer_with_two_cube_sum_representations :
  (has_two_representations 251) ∧
  (∀ m : ℕ, m < 251 → ¬(has_two_representations m)) :=
by sorry

end smallest_integer_with_two_cube_sum_representations_l3694_369430


namespace integral_exp_abs_plus_sqrt_l3694_369423

theorem integral_exp_abs_plus_sqrt : ∫ (x : ℝ) in (-1)..(1), (Real.exp (|x|) + Real.sqrt (1 - x^2)) = 2 * (Real.exp 1 - 1) + π / 2 := by
  sorry

end integral_exp_abs_plus_sqrt_l3694_369423


namespace parabola_focus_directrix_distance_l3694_369465

/-- Given a parabola x^2 = (1/4)y, the distance between its focus and directrix is 1/8 -/
theorem parabola_focus_directrix_distance (x y : ℝ) :
  x^2 = (1/4) * y → (distance_focus_directrix : ℝ) = 1/8 := by
  sorry

end parabola_focus_directrix_distance_l3694_369465


namespace trig_identity_l3694_369495

theorem trig_identity (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 + 2 * Real.sin x * Real.sin y * Real.sin (x + y) =
  2 - Real.cos x ^ 2 - Real.cos (x + y) ^ 2 := by
  sorry

end trig_identity_l3694_369495


namespace cloth_weaving_problem_l3694_369474

theorem cloth_weaving_problem (a₁ a₃₀ n : ℝ) (h1 : a₁ = 5) (h2 : a₃₀ = 1) (h3 : n = 30) :
  n / 2 * (a₁ + a₃₀) = 90 := by
  sorry

end cloth_weaving_problem_l3694_369474


namespace fundraising_goal_exceeded_l3694_369438

theorem fundraising_goal_exceeded (goal ken_amount : ℕ) 
  (h1 : ken_amount = 600)
  (h2 : goal = 4000) : 
  let mary_amount := 5 * ken_amount
  let scott_amount := mary_amount / 3
  ken_amount + mary_amount + scott_amount - goal = 600 := by
sorry

end fundraising_goal_exceeded_l3694_369438


namespace not_solution_and_solutions_l3694_369472

def is_solution (x y : ℤ) : Prop := 85 * x - 324 * y = 101

theorem not_solution_and_solutions :
  ¬(is_solution 978 256) ∧
  is_solution 5 1 ∧
  is_solution 329 86 ∧
  is_solution 653 171 ∧
  is_solution 1301 341 := by
  sorry

end not_solution_and_solutions_l3694_369472


namespace trigonometric_problem_l3694_369439

theorem trigonometric_problem (α β : Real) 
  (h1 : 3 * Real.sin α - Real.sin β = Real.sqrt 10)
  (h2 : α + β = Real.pi / 2) :
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ Real.cos (2 * β) = 4 / 5 := by
  sorry

end trigonometric_problem_l3694_369439


namespace twentieth_number_in_base8_l3694_369418

/-- Converts a decimal number to its base 8 representation -/
def toBase8 (n : ℕ) : ℕ := sorry

/-- Represents the sequence of numbers in base 8 -/
def base8Sequence : ℕ → ℕ := sorry

theorem twentieth_number_in_base8 :
  base8Sequence 20 = toBase8 24 := by sorry

end twentieth_number_in_base8_l3694_369418


namespace inequality_proof_l3694_369405

theorem inequality_proof (a b c d e : ℝ) 
  (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ e)
  (h6 : a + b + c + d + e = 1) : 
  a * d + d * c + c * b + b * e + e * a ≤ 1/5 := by
  sorry

end inequality_proof_l3694_369405


namespace nabla_computation_l3694_369484

-- Define the operation ∇
def nabla (x y : ℕ) : ℕ := x^3 - 2*y

-- State the theorem
theorem nabla_computation :
  (5^(nabla 7 4)) - 2*(2^(nabla 6 9)) = 5^1005 - 2^199 :=
by sorry

end nabla_computation_l3694_369484


namespace systematic_sampling_interval_l3694_369407

/-- Calculates the sampling interval for systematic sampling -/
def sampling_interval (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

theorem systematic_sampling_interval :
  sampling_interval 630 45 = 14 := by
  sorry

end systematic_sampling_interval_l3694_369407


namespace cookie_recipe_total_cups_l3694_369454

theorem cookie_recipe_total_cups (butter flour sugar : ℕ) (total : ℕ) : 
  (butter : ℚ) / flour = 2 / 5 →
  (sugar : ℚ) / flour = 3 / 5 →
  flour = 15 →
  total = butter + flour + sugar →
  total = 30 := by
sorry

end cookie_recipe_total_cups_l3694_369454


namespace max_container_weight_l3694_369463

def total_goods : ℕ := 1500
def num_platforms : ℕ := 25
def platform_capacity : ℕ := 80

def is_transportable (k : ℕ) : Prop :=
  ∀ (containers : List ℕ),
    (containers.sum = total_goods) →
    (∀ c ∈ containers, c ≤ k ∧ c > 0) →
    ∃ (loading : List (List ℕ)),
      loading.length ≤ num_platforms ∧
      (∀ platform ∈ loading, platform.sum ≤ platform_capacity) ∧
      loading.join.sum = total_goods

theorem max_container_weight :
  (∀ k ≤ 26, is_transportable k) ∧
  ¬(is_transportable 27) := by sorry

end max_container_weight_l3694_369463


namespace sin_sum_arcsin_arctan_l3694_369402

theorem sin_sum_arcsin_arctan :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = 11 * Real.sqrt 5 / 25 := by
  sorry

end sin_sum_arcsin_arctan_l3694_369402


namespace dentist_bill_ratio_l3694_369455

def cleaning_cost : ℕ := 70
def filling_cost : ℕ := 120
def extraction_cost : ℕ := 290

def total_bill : ℕ := cleaning_cost + 2 * filling_cost + extraction_cost

theorem dentist_bill_ratio :
  (total_bill : ℚ) / filling_cost = 5 := by sorry

end dentist_bill_ratio_l3694_369455


namespace line_passes_through_fixed_point_l3694_369410

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the parabola
def point_on_parabola (x₀ : ℝ) : Prop := parabola x₀ 4

-- Define a line passing through a point and intersecting the parabola at two other points
def intersecting_line (x₀ m t : ℝ) : Prop :=
  point_on_parabola x₀ ∧ ∃ y₁ y₂ : ℝ, 
    y₁ ≠ y₂ ∧ 
    parabola (m*y₁ + t) y₁ ∧ 
    parabola (m*y₂ + t) y₂

-- Define perpendicularity condition
def perpendicular_condition (x₀ m t : ℝ) : Prop :=
  ∃ y₁ y₂ : ℝ, 
    (m*y₁ + t - x₀) * (m*y₂ + t - x₀) + (y₁ - 4) * (y₂ - 4) = 0

-- Theorem statement
theorem line_passes_through_fixed_point (x₀ m t : ℝ) :
  intersecting_line x₀ m t ∧ perpendicular_condition x₀ m t →
  t = 4*m + 8 :=
sorry

end line_passes_through_fixed_point_l3694_369410


namespace fish_filets_count_l3694_369481

/-- The number of fish filets Ben and his family will have -/
def fish_filets : ℕ :=
  let ben_fish := 4
  let judy_fish := 1
  let billy_fish := 3
  let jim_fish := 2
  let susie_fish := 5
  let thrown_back := 3
  let filets_per_fish := 2
  let total_caught := ben_fish + judy_fish + billy_fish + jim_fish + susie_fish
  let fish_kept := total_caught - thrown_back
  fish_kept * filets_per_fish

theorem fish_filets_count : fish_filets = 24 := by
  sorry

end fish_filets_count_l3694_369481


namespace sum_of_cubes_inequality_l3694_369404

theorem sum_of_cubes_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1)^3 / b + (b + 1)^3 / c + (c + 1)^3 / a ≥ 81 / 4 := by
  sorry

end sum_of_cubes_inequality_l3694_369404


namespace min_value_inequality_l3694_369450

theorem min_value_inequality (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_eq : x + y + z = 9)
  (prod_sum_eq : x*y + y*z + z*x = 14) :
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 := by
sorry

end min_value_inequality_l3694_369450


namespace y_value_l3694_369428

theorem y_value (x y z : ℤ) 
  (eq1 : x + y + z = 25) 
  (eq2 : x + y = 19) 
  (eq3 : y + z = 18) : 
  y = 12 := by
sorry

end y_value_l3694_369428


namespace temperature_difference_l3694_369475

def highest_temp : ℝ := 10
def lowest_temp : ℝ := -1

theorem temperature_difference : highest_temp - lowest_temp = 11 := by
  sorry

end temperature_difference_l3694_369475


namespace subtracted_value_l3694_369482

theorem subtracted_value (n : ℕ) (x : ℕ) (h1 : n = 121) (h2 : 2 * n - x = 104) : x = 138 := by
  sorry

end subtracted_value_l3694_369482


namespace quadratic_inequality_relation_l3694_369436

theorem quadratic_inequality_relation :
  (∀ x : ℝ, x > 2 → x^2 + 5*x - 6 > 0) ∧
  (∃ x : ℝ, x^2 + 5*x - 6 > 0 ∧ x ≤ 2) :=
by sorry

end quadratic_inequality_relation_l3694_369436


namespace rectangle_height_from_square_perimeter_l3694_369494

theorem rectangle_height_from_square_perimeter (square_side : ℝ) (rect_width : ℝ) :
  square_side = 20 →
  rect_width = 14 →
  4 * square_side = 2 * (rect_width + (80 - 2 * rect_width) / 2) →
  (80 - 2 * rect_width) / 2 = 26 :=
by sorry

end rectangle_height_from_square_perimeter_l3694_369494


namespace consecutive_numbers_equation_l3694_369433

theorem consecutive_numbers_equation (x y z : ℤ) : 
  (x = y + 1) → 
  (z = y - 1) → 
  (x > y) → 
  (y > z) → 
  (z = 2) → 
  (2 * x + 3 * y + 3 * z = 8 * y - 1) :=
by sorry

end consecutive_numbers_equation_l3694_369433


namespace complex_number_properties_l3694_369409

/-- For a real number m and a complex number z = (m^2 - 5m + 6) + (m^2 - 3m)i, we define the following properties --/

def is_real (m : ℝ) : Prop := m^2 - 3*m = 0

def is_complex (m : ℝ) : Prop := m^2 - 3*m ≠ 0

def is_purely_imaginary (m : ℝ) : Prop := m^2 - 5*m + 6 = 0 ∧ m^2 - 3*m ≠ 0

def is_in_third_quadrant (m : ℝ) : Prop := m^2 - 5*m + 6 < 0 ∧ m^2 - 3*m < 0

/-- Main theorem stating the conditions for each case --/
theorem complex_number_properties (m : ℝ) :
  (is_real m ↔ (m = 0 ∨ m = 3)) ∧
  (is_complex m ↔ (m ≠ 0 ∧ m ≠ 3)) ∧
  (is_purely_imaginary m ↔ m = 2) ∧
  (is_in_third_quadrant m ↔ (2 < m ∧ m < 3)) :=
sorry

end complex_number_properties_l3694_369409


namespace cricketer_average_difference_is_13_l3694_369490

def cricketer_average_difference (runs_A runs_B : ℕ) (innings_A innings_B : ℕ) 
  (increase_A increase_B : ℚ) : ℚ :=
  let avg_A : ℚ := (runs_A : ℚ) / (innings_A : ℚ)
  let avg_B : ℚ := (runs_B : ℚ) / (innings_B : ℚ)
  (avg_B + increase_B) - (avg_A + increase_A)

theorem cricketer_average_difference_is_13 :
  cricketer_average_difference 125 145 20 18 5 6 = 13 := by
  sorry

end cricketer_average_difference_is_13_l3694_369490


namespace new_students_count_l3694_369406

theorem new_students_count (initial_students : ℕ) (left_students : ℕ) (final_students : ℕ) 
  (h1 : initial_students = 31)
  (h2 : left_students = 5)
  (h3 : final_students = 37) :
  final_students - (initial_students - left_students) = 11 :=
by sorry

end new_students_count_l3694_369406


namespace max_enclosure_area_l3694_369477

/-- The number of fence pieces --/
def num_pieces : ℕ := 15

/-- The length of each fence piece in meters --/
def piece_length : ℝ := 2

/-- The total length of fencing available in meters --/
def total_length : ℝ := num_pieces * piece_length

/-- The area of the rectangular enclosure as a function of its width --/
def area (w : ℝ) : ℝ := (total_length - 2 * w) * w

/-- The maximum area of the enclosure, rounded down to the nearest integer --/
def max_area : ℕ := 112

theorem max_enclosure_area :
  ∃ (w : ℝ), 0 < w ∧ w < total_length / 2 ∧
  (∀ (x : ℝ), 0 < x → x < total_length / 2 → area x ≤ area w) ∧
  ⌊area w⌋ = max_area :=
sorry

end max_enclosure_area_l3694_369477


namespace greatest_constant_right_triangle_l3694_369424

theorem greatest_constant_right_triangle (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_positive : c > 0) :
  ∀ N : ℝ, (∀ a b c : ℝ, c > 0 → a^2 + b^2 = c^2 → (a^2 + b^2 - c^2) / c^2 > N) → N ≤ -1 :=
by sorry

end greatest_constant_right_triangle_l3694_369424


namespace beacon_population_l3694_369422

/-- Given the populations of three cities with specific relationships, prove the population of Beacon. -/
theorem beacon_population
  (richmond victoria beacon : ℕ)
  (h1 : richmond = victoria + 1000)
  (h2 : victoria = 4 * beacon)
  (h3 : richmond = 3000) :
  beacon = 500 := by
  sorry

end beacon_population_l3694_369422


namespace fewer_bees_than_flowers_l3694_369442

theorem fewer_bees_than_flowers (flowers : ℕ) (bees : ℕ) 
  (h1 : flowers = 5) (h2 : bees = 3) : flowers - bees = 2 := by
  sorry

end fewer_bees_than_flowers_l3694_369442


namespace smallest_solution_congruence_l3694_369421

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (4 * x) % 31 = 17 % 31 ∧
  ∀ (y : ℕ), y > 0 ∧ (4 * y) % 31 = 17 % 31 → x ≤ y :=
by sorry

end smallest_solution_congruence_l3694_369421


namespace gambler_final_amount_l3694_369426

def bet_sequence := [true, false, true, false, false, true, false, true]

def apply_bet (current_amount : ℚ) (is_win : Bool) : ℚ :=
  if is_win then
    current_amount + (current_amount / 2)
  else
    current_amount / 2

def final_amount (initial_amount : ℚ) (bets : List Bool) : ℚ :=
  bets.foldl apply_bet initial_amount

theorem gambler_final_amount :
  final_amount 128 bet_sequence = 40.5 := by
  sorry

end gambler_final_amount_l3694_369426


namespace proper_subsets_count_l3694_369415

def S : Finset Nat := {2, 4, 6, 8}

theorem proper_subsets_count : (Finset.powerset S).card - 1 = 15 := by
  sorry

end proper_subsets_count_l3694_369415


namespace product_of_three_numbers_l3694_369461

theorem product_of_three_numbers (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 24 * (3 ^ (1/4)))
  (hac : a * c = 50 * (3 ^ (1/4)))
  (hbc : b * c = 18 * (3 ^ (1/4))) :
  a * b * c = 120 * (3 ^ (1/4)) := by
sorry

end product_of_three_numbers_l3694_369461


namespace circus_ticket_cost_l3694_369417

theorem circus_ticket_cost (total_cost : ℕ) (num_tickets : ℕ) (cost_per_ticket : ℕ) :
  total_cost = 308 →
  num_tickets = 7 →
  cost_per_ticket * num_tickets = total_cost →
  cost_per_ticket = 44 := by
sorry

end circus_ticket_cost_l3694_369417


namespace bakery_rolls_combinations_l3694_369466

theorem bakery_rolls_combinations :
  let total_rolls : ℕ := 8
  let num_kinds : ℕ := 4
  let rolls_to_distribute : ℕ := total_rolls - num_kinds
  (Nat.choose (rolls_to_distribute + num_kinds - 1) (num_kinds - 1)) = 35 := by
  sorry

end bakery_rolls_combinations_l3694_369466


namespace k_equals_p_l3694_369479

theorem k_equals_p (k p : ℕ) : 
  (∃ (nums_k : Finset ℕ) (nums_p : Finset ℕ), 
    (Finset.card nums_k = k) ∧ 
    (Finset.card nums_p = p) ∧
    (∀ x ∈ nums_k, x = 2*p + 3) ∧
    (∀ y ∈ nums_p, y = 5 - 2*k) ∧
    ((Finset.sum nums_k id + Finset.sum nums_p id) / (k + p : ℝ) = 4)) →
  k = p :=
sorry

end k_equals_p_l3694_369479


namespace second_draw_probability_l3694_369462

/-- Represents the total number of items -/
def total_items : ℕ := 10

/-- Represents the number of genuine items -/
def genuine_items : ℕ := 6

/-- Represents the number of defective items -/
def defective_items : ℕ := 4

/-- Represents the probability of drawing a genuine item on the second draw,
    given that the first item drawn is genuine -/
def prob_second_genuine : ℚ := 5 / 9

theorem second_draw_probability :
  total_items = genuine_items + defective_items →
  genuine_items > 0 →
  prob_second_genuine = (genuine_items - 1) / (total_items - 1) :=
by sorry

end second_draw_probability_l3694_369462


namespace largest_marble_count_l3694_369434

theorem largest_marble_count : ∃ n : ℕ, n < 400 ∧ 
  n % 3 = 1 ∧ n % 7 = 2 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, m < 400 → m % 3 = 1 → m % 7 = 2 → m % 5 = 0 → m ≤ n :=
by sorry

end largest_marble_count_l3694_369434


namespace uncle_dave_nieces_l3694_369459

theorem uncle_dave_nieces (total_sandwiches : ℕ) (sandwiches_per_niece : ℕ) (h1 : total_sandwiches = 143) (h2 : sandwiches_per_niece = 13) :
  total_sandwiches / sandwiches_per_niece = 11 := by
  sorry

end uncle_dave_nieces_l3694_369459


namespace period_of_sin_plus_cos_l3694_369427

/-- The period of the function y = 3sin(x) + 3cos(x) is 2π -/
theorem period_of_sin_plus_cos : 
  let f : ℝ → ℝ := λ x => 3 * Real.sin x + 3 * Real.cos x
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ ∀ q : ℝ, 0 < q ∧ q < p → ∃ x : ℝ, f (x + q) ≠ f x :=
by
  sorry

end period_of_sin_plus_cos_l3694_369427


namespace nuts_cost_to_age_ratio_l3694_369469

/-- The ratio of the cost of a pack of nuts to Betty's age -/
theorem nuts_cost_to_age_ratio : 
  ∀ (doug_age betty_age : ℕ) (num_packs total_cost : ℕ),
  doug_age = 40 →
  doug_age + betty_age = 90 →
  num_packs = 20 →
  total_cost = 2000 →
  (total_cost / num_packs : ℚ) / betty_age = 2 :=
by
  sorry

end nuts_cost_to_age_ratio_l3694_369469


namespace double_acute_angle_range_l3694_369498

theorem double_acute_angle_range (α : Real) (h : 0 < α ∧ α < π / 2) : 
  0 < 2 * α ∧ 2 * α < π := by
  sorry

end double_acute_angle_range_l3694_369498


namespace tangent_circle_center_l3694_369447

/-- A circle passes through (0,3) and is tangent to y = x^2 at (1,1) -/
structure TangentCircle where
  center : ℝ × ℝ
  passes_through : center.1^2 + (center.2 - 3)^2 = (center.1 - 0)^2 + (center.2 - 3)^2
  tangent_at : center.1^2 + (center.2 - 1)^2 = (center.1 - 1)^2 + (center.2 - 1)^2
  on_parabola : 1 = 1^2

/-- The center of the circle is (0, 3/2) -/
theorem tangent_circle_center : ∀ c : TangentCircle, c.center = (0, 3/2) := by
  sorry

end tangent_circle_center_l3694_369447


namespace equation_one_solutions_l3694_369456

theorem equation_one_solutions (x : ℝ) : 
  x^2 - 6*x - 1 = 0 ↔ x = 3 + Real.sqrt 10 ∨ x = 3 - Real.sqrt 10 := by
sorry

end equation_one_solutions_l3694_369456


namespace class_average_l3694_369416

theorem class_average (total_students : Nat) (perfect_score_students : Nat) (zero_score_students : Nat) (class_average : ℚ) : 
  total_students = 20 →
  perfect_score_students = 2 →
  zero_score_students = 3 →
  class_average = 40 →
  let remaining_students := total_students - perfect_score_students - zero_score_students
  let total_score := total_students * class_average
  let perfect_score_total := perfect_score_students * 100
  let remaining_score := total_score - perfect_score_total
  remaining_score / remaining_students = 40 := by
sorry

end class_average_l3694_369416


namespace rope_division_l3694_369496

theorem rope_division (rope_length : ℝ) (num_parts : ℕ) (part_length : ℝ) :
  rope_length = 5 →
  num_parts = 4 →
  rope_length = num_parts * part_length →
  part_length = 1.25 := by
sorry

end rope_division_l3694_369496


namespace problem_solution_l3694_369471

def A (x y : ℝ) : ℝ := 3 * x^2 + 2 * x * y - 2 * x - 1
def B (x y : ℝ) : ℝ := -x^2 + x * y - 1

theorem problem_solution (x y : ℝ) :
  (A x y + 3 * B x y = 5 * x * y - 2 * x - 4) ∧
  (∀ x, A x y + 3 * B x y = A 0 y + 3 * B 0 y → y = 2/5) := by
  sorry

end problem_solution_l3694_369471


namespace multiplication_puzzle_l3694_369400

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def are_distinct (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def matches_pattern (a b c d e : ℕ) : Prop :=
  let abba := a * 1000 + b * 100 + b * 10 + a
  let cdea := c * 1000 + d * 100 + e * 10 + a
  let product := abba * cdea
  ∃ (x y z : ℕ),
    product = z * 100000 + b * 1000 + b * 100 + e * 10 + e ∧
    z = x * 10000 + y * 1000 + c * 100 + e * 10 + e

theorem multiplication_puzzle :
  ∀ (a b c d e : ℕ),
    is_valid_digit a → is_valid_digit b → is_valid_digit c → is_valid_digit d → is_valid_digit e →
    are_distinct a b c d e →
    matches_pattern a b c d e →
    a = 3 ∧ b = 0 ∧ c = 7 ∧ d = 2 ∧ e = 9 :=
sorry

end multiplication_puzzle_l3694_369400


namespace log_cutting_l3694_369499

theorem log_cutting (fallen_pieces fixed_pieces : ℕ) 
  (h1 : fallen_pieces = 10)
  (h2 : fixed_pieces = 2) :
  fallen_pieces + fixed_pieces - 1 = 11 := by
sorry

end log_cutting_l3694_369499


namespace count_integers_with_three_digits_under_50000_l3694_369429

/-- A function that counts the number of positive integers less than n with at most k different digits. -/
def count_integers_with_limited_digits (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the count of positive integers less than 50,000 with at most three different digits is 7862. -/
theorem count_integers_with_three_digits_under_50000 :
  count_integers_with_limited_digits 50000 3 = 7862 :=
sorry

end count_integers_with_three_digits_under_50000_l3694_369429


namespace max_area_of_cut_triangle_l3694_369467

/-- Triangle ABC with side lengths -/
structure Triangle :=
  (AB : ℝ)
  (BC : ℝ)
  (CA : ℝ)

/-- The given triangle -/
def givenTriangle : Triangle :=
  { AB := 13, BC := 14, CA := 15 }

/-- A line cutting the triangle -/
structure CuttingLine :=
  (intersectsSide1 : ℝ)
  (intersectsSide2 : ℝ)

/-- The area of the triangle formed by the cutting line -/
def areaOfCutTriangle (t : Triangle) (l : CuttingLine) : ℝ :=
  sorry

/-- The perimeter of the triangle formed by the cutting line -/
def perimeterOfCutTriangle (t : Triangle) (l : CuttingLine) : ℝ :=
  sorry

/-- The perimeter of the quadrilateral formed by the cutting line -/
def perimeterOfCutQuadrilateral (t : Triangle) (l : CuttingLine) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem max_area_of_cut_triangle (t : Triangle) :
  t = givenTriangle →
  ∃ (l : CuttingLine),
    perimeterOfCutTriangle t l = perimeterOfCutQuadrilateral t l ∧
    ∀ (l' : CuttingLine),
      perimeterOfCutTriangle t l' = perimeterOfCutQuadrilateral t l' →
      areaOfCutTriangle t l' ≤ 1323 / 26 :=
sorry

end max_area_of_cut_triangle_l3694_369467


namespace tangent_line_equation_l3694_369493

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 2

-- Define the point P
def P : ℝ × ℝ := (2, -1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - 3*y + 3 = 0

-- Theorem statement
theorem tangent_line_equation :
  ∃ (A B : ℝ × ℝ),
    C A.1 A.2 ∧ C B.1 B.2 ∧
    (∀ (x y : ℝ), C x y → (x - P.1)*(y - P.2) = (A.1 - P.1)*(A.2 - P.2) ∨ (x - P.1)*(y - P.2) = (B.1 - P.1)*(B.2 - P.2)) →
    (∀ (x y : ℝ), line_equation x y ↔ (∃ t : ℝ, x = A.1 + t*(B.1 - A.1) ∧ y = A.2 + t*(B.2 - A.2))) :=
sorry


end tangent_line_equation_l3694_369493


namespace geometric_and_arithmetic_sequences_l3694_369460

-- Define the geometric sequence a_n
def a (n : ℕ) : ℝ := 3 * 2^(n - 1)

-- Define the arithmetic sequence b_n
def b (n : ℕ) : ℝ := 6 * n - 6

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℝ := 3 * n^2 - 3 * n

theorem geometric_and_arithmetic_sequences :
  (a 1 = 3) ∧ 
  (a 4 = 24) ∧ 
  (b 2 = a 2) ∧ 
  (b 9 = a 5) ∧ 
  (∀ n : ℕ, a n = 3 * 2^(n - 1)) ∧ 
  (∀ n : ℕ, S n = 3 * n^2 - 3 * n) :=
by sorry

end geometric_and_arithmetic_sequences_l3694_369460


namespace equation_solution_set_l3694_369451

theorem equation_solution_set (x : ℝ) : 
  (((9 : ℝ)^x + 32^x) / (15^x + 24^x) = 4/3) ↔ 
  (x = (Real.log (3/4)) / (Real.log (3/2)) ∨ x = (Real.log 4) / (Real.log 3)) :=
by sorry

end equation_solution_set_l3694_369451


namespace tutors_next_meeting_l3694_369419

def chris_schedule : ℕ := 5
def alex_schedule : ℕ := 6
def jordan_schedule : ℕ := 8
def taylor_schedule : ℕ := 9

theorem tutors_next_meeting :
  lcm (lcm (lcm chris_schedule alex_schedule) jordan_schedule) taylor_schedule = 360 := by
  sorry

end tutors_next_meeting_l3694_369419


namespace circle_equation_k_value_l3694_369420

theorem circle_equation_k_value (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 14*x + y^2 + 8*y - k = 0 ↔ (x + 7)^2 + (y + 4)^2 = 64) → 
  k = -1 := by
sorry

end circle_equation_k_value_l3694_369420


namespace complex_multiplication_l3694_369414

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (2 + 2*i) * (1 - 2*i) = 6 - 2*i := by
  sorry

end complex_multiplication_l3694_369414


namespace smallest_number_l3694_369425

theorem smallest_number (a b c : ℝ) (ha : a = 0.8) (hb : b = 1/2) (hc : c = 0.5) :
  min (min a b) c > 0.1 ∧ min (min a b) c = 0.5 := by
  sorry

end smallest_number_l3694_369425


namespace yasmin_has_two_children_l3694_369449

/-- The number of children Yasmin has -/
def yasmin_children : ℕ := 2

/-- The number of children John has -/
def john_children : ℕ := 2 * yasmin_children

/-- The total number of grandchildren -/
def total_grandchildren : ℕ := 6

theorem yasmin_has_two_children :
  yasmin_children = 2 ∧
  john_children = 2 * yasmin_children ∧
  yasmin_children + john_children = total_grandchildren :=
sorry

end yasmin_has_two_children_l3694_369449


namespace line_circle_intersection_a_eq_one_l3694_369440

/-- A line intersecting a circle forming a right triangle -/
structure LineCircleIntersection where
  a : ℝ
  -- Line equation: ax - y + 6 = 0
  line : ℝ → ℝ → Prop := fun x y ↦ a * x - y + 6 = 0
  -- Circle equation: (x + 1)^2 + (y - a)^2 = 16
  circle : ℝ → ℝ → Prop := fun x y ↦ (x + 1)^2 + (y - a)^2 = 16
  -- Circle center
  center : ℝ × ℝ := (-1, a)
  -- Existence of intersection points A and B
  A : ℝ × ℝ
  B : ℝ × ℝ
  hA : line A.1 A.2 ∧ circle A.1 A.2
  hB : line B.1 B.2 ∧ circle B.1 B.2
  -- Triangle ABC is a right triangle
  hRight : (A.1 - B.1) * (center.1 - B.1) + (A.2 - B.2) * (center.2 - B.2) = 0

/-- The positive value of a in the LineCircleIntersection is 1 -/
theorem line_circle_intersection_a_eq_one (lci : LineCircleIntersection) : 
  lci.a > 0 → lci.a = 1 := by sorry

end line_circle_intersection_a_eq_one_l3694_369440


namespace expand_expression_l3694_369401

theorem expand_expression (x : ℝ) : (x - 3) * (x + 6) = x^2 + 3*x - 18 := by
  sorry

end expand_expression_l3694_369401


namespace simplify_polynomial_l3694_369491

theorem simplify_polynomial (x : ℝ) : 
  3*x + 5 - 4*x^2 + 2*x - 7 + x^2 - 3*x + 8 = -3*x^2 + 2*x + 6 := by
sorry

end simplify_polynomial_l3694_369491


namespace functional_equation_solution_l3694_369453

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) + 2 * x = x * f y + 3 * f x

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → f (-1) = 7 → f (-1001) = -3493 := by
  sorry

end functional_equation_solution_l3694_369453


namespace probability_adjacent_points_l3694_369443

/-- Represents a point on the 3x3 square --/
inductive SquarePoint
| Corner
| MidSide
| Center

/-- The set of all points on the 3x3 square --/
def squarePoints : Finset SquarePoint := sorry

/-- Two points are considered adjacent if they are one unit apart --/
def adjacent : SquarePoint → SquarePoint → Prop := sorry

/-- The number of pairs of adjacent points --/
def adjacentPairsCount : ℕ := sorry

theorem probability_adjacent_points :
  (adjacentPairsCount : ℚ) / (Finset.card (squarePoints.powerset.filter (λ s => s.card = 2)) : ℚ) = 16/45 := by
  sorry

end probability_adjacent_points_l3694_369443


namespace unit_digit_of_23_power_100000_l3694_369412

theorem unit_digit_of_23_power_100000 : 23^100000 % 10 = 1 := by
  sorry

end unit_digit_of_23_power_100000_l3694_369412


namespace x_squared_plus_y_squared_equals_one_l3694_369441

theorem x_squared_plus_y_squared_equals_one
  (x y : ℝ)
  (h1 : (x^2 + y^2 + 1) * (x^2 + y^2 + 3) = 8)
  (h2 : x^2 + y^2 ≥ 0) :
  x^2 + y^2 = 1 :=
by sorry

end x_squared_plus_y_squared_equals_one_l3694_369441


namespace forty_percent_value_l3694_369437

theorem forty_percent_value (x : ℝ) (h : 0.5 * x = 200) : 0.4 * x = 160 := by
  sorry

end forty_percent_value_l3694_369437


namespace raviraj_cycled_20km_l3694_369468

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents Raviraj's cycling journey -/
def raviraj_journey (final_distance : ℝ) : Prop :=
  ∃ (home : Point) (last_turn : Point) (final : Point),
    -- Initial movements
    last_turn.x = home.x - 10 ∧
    last_turn.y = home.y ∧
    -- Final position
    final.x = last_turn.x - 20 ∧
    final.y = last_turn.y ∧
    -- Distance to home is 30 km
    (final.x - home.x)^2 + (final.y - home.y)^2 = final_distance^2

/-- The theorem stating that Raviraj cycled 20 km after the third turn -/
theorem raviraj_cycled_20km : raviraj_journey 30 → 20 = 20 := by
  sorry

end raviraj_cycled_20km_l3694_369468


namespace unique_x_for_all_y_l3694_369444

theorem unique_x_for_all_y : ∃! x : ℚ, ∀ y : ℚ, 10 * x * y - 15 * y + 2 * x - 3 = 0 := by
  sorry

end unique_x_for_all_y_l3694_369444


namespace vehicle_speeds_theorem_l3694_369435

/-- Represents the speeds of two vehicles traveling in opposite directions -/
structure VehicleSpeeds where
  slow : ℝ
  fast : ℝ
  speed_diff : fast = slow + 8

/-- Proves that given the conditions, the speeds of the vehicles are 44 and 52 mph -/
theorem vehicle_speeds_theorem (v : VehicleSpeeds) 
  (h : 4 * (v.slow + v.fast) = 384) : 
  v.slow = 44 ∧ v.fast = 52 := by
  sorry

#check vehicle_speeds_theorem

end vehicle_speeds_theorem_l3694_369435


namespace rectangle_diagonal_l3694_369403

/-- The diagonal of a rectangle with length 6 and width 8 is 10. -/
theorem rectangle_diagonal : ∀ (l w d : ℝ), 
  l = 6 → w = 8 → d^2 = l^2 + w^2 → d = 10 := by
  sorry

end rectangle_diagonal_l3694_369403


namespace shower_tasks_count_l3694_369492

/-- The number of tasks to clean the house -/
def clean_house_tasks : ℕ := 7

/-- The number of tasks to make dinner -/
def make_dinner_tasks : ℕ := 4

/-- The time each task takes in minutes -/
def time_per_task : ℕ := 10

/-- The total time to complete all tasks in hours -/
def total_time_hours : ℕ := 2

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

theorem shower_tasks_count : 
  (clean_house_tasks + make_dinner_tasks + 1) * time_per_task = total_time_hours * minutes_per_hour := by
  sorry

end shower_tasks_count_l3694_369492


namespace solve_yellow_balloons_problem_l3694_369448

def yellow_balloons_problem (sam_initial : Real) (sam_gives : Real) (total : Real) : Prop :=
  let sam_remaining : Real := sam_initial - sam_gives
  let mary_balloons : Real := total - sam_remaining
  mary_balloons = 7.0

theorem solve_yellow_balloons_problem :
  yellow_balloons_problem 6.0 5.0 8.0 := by
  sorry

end solve_yellow_balloons_problem_l3694_369448


namespace distance_between_4th_and_26th_red_lights_l3694_369470

/-- The distance in feet between two red lights in a repeating pattern -/
def distance_between_red_lights (n m : ℕ) : ℚ :=
  let inches_between_lights : ℕ := 4
  let pattern_length : ℕ := 5
  let inches_per_foot : ℕ := 12
  let position (k : ℕ) : ℕ := 1 + (k - 1) / 2 * pattern_length + 2 * ((k - 1) % 2)
  let gaps : ℕ := position m - position n
  (gaps * inches_between_lights : ℚ) / inches_per_foot

/-- The theorem stating the distance between the 4th and 26th red lights -/
theorem distance_between_4th_and_26th_red_lights :
  distance_between_red_lights 4 26 = 18.33 :=
sorry

end distance_between_4th_and_26th_red_lights_l3694_369470


namespace no_valid_a_exists_l3694_369458

theorem no_valid_a_exists : ¬ ∃ (a n : ℕ), 
  a > 1 ∧ 
  n > 0 ∧ 
  ∃ (k : ℕ), a * (10^n + 1) = k * a^2 := by
sorry

end no_valid_a_exists_l3694_369458


namespace equilateral_triangle_side_length_l3694_369457

theorem equilateral_triangle_side_length 
  (circumference : ℝ) 
  (h1 : circumference = 4 * 21) 
  (h2 : circumference > 0) : 
  ∃ (side_length : ℝ), side_length = 28 ∧ 3 * side_length = circumference :=
sorry

end equilateral_triangle_side_length_l3694_369457


namespace units_digit_of_product_sequence_l3694_369464

def product_sequence (n : ℕ) : ℕ :=
  (List.range 17).foldl (λ acc i => acc * (2^(2*i) + 1)) 3

theorem units_digit_of_product_sequence :
  (product_sequence 17 + 1) % 10 = 6 := by
  sorry

end units_digit_of_product_sequence_l3694_369464


namespace soccer_games_per_month_l3694_369497

/-- Given a total number of games and number of months in a season,
    calculate the number of games per month assuming equal distribution -/
def games_per_month (total_games : ℕ) (num_months : ℕ) : ℕ :=
  total_games / num_months

/-- Theorem: For 27 games over 3 months, there are 9 games per month -/
theorem soccer_games_per_month :
  games_per_month 27 3 = 9 := by
  sorry

end soccer_games_per_month_l3694_369497


namespace corner_sum_is_168_l3694_369411

def checkerboard_size : Nat := 9

def min_number : Nat := 2
def max_number : Nat := 82

def top_left : Nat := min_number
def top_right : Nat := min_number + checkerboard_size - 1
def bottom_left : Nat := max_number - checkerboard_size + 1
def bottom_right : Nat := max_number

theorem corner_sum_is_168 :
  top_left + top_right + bottom_left + bottom_right = 168 := by
  sorry

end corner_sum_is_168_l3694_369411


namespace total_fruits_is_236_l3694_369486

/-- The total number of fruits picked by Sara and Sally -/
def total_fruits (sara_pears sara_apples sara_plums sally_pears sally_apples sally_plums : ℕ) : ℕ :=
  (sara_pears + sally_pears) + (sara_apples + sally_apples) + (sara_plums + sally_plums)

/-- Theorem: The total number of fruits picked by Sara and Sally is 236 -/
theorem total_fruits_is_236 :
  total_fruits 45 22 64 11 38 56 = 236 := by
  sorry

end total_fruits_is_236_l3694_369486


namespace bookstore_purchase_equation_l3694_369445

theorem bookstore_purchase_equation (x : ℝ) : 
  (500 : ℝ) > 0 ∧ (700 : ℝ) > 0 ∧ x > 0 →
  (500 / x = 700 / (x + 4)) ↔ 
  (∃ (price_per_set : ℝ), 
    price_per_set > 0 ∧
    500 = price_per_set * x ∧
    700 = price_per_set * (x + 4)) :=
by sorry

end bookstore_purchase_equation_l3694_369445


namespace solution_range_l3694_369487

-- Define the new operation
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- Theorem statement
theorem solution_range (a : ℝ) :
  (∃ x : ℝ, otimes x (x - a) > 1) ↔ (a < -3 ∨ a > 1) :=
sorry

end solution_range_l3694_369487


namespace correct_regression_equation_l3694_369480

/-- Represents a linear regression equation -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Checks if a linear regression equation passes through a given point -/
def passes_through (eq : LinearRegression) (x y : ℝ) : Prop :=
  eq.slope * x + eq.intercept = y

/-- Represents the properties of the given data -/
structure DataProperties where
  x_mean : ℝ
  y_mean : ℝ
  positively_correlated : Prop

theorem correct_regression_equation 
  (data : DataProperties)
  (h_x_mean : data.x_mean = 2.4)
  (h_y_mean : data.y_mean = 3.2)
  (h_corr : data.positively_correlated) :
  ∃ (eq : LinearRegression), 
    eq.slope = 0.5 ∧ 
    eq.intercept = 2 ∧ 
    passes_through eq data.x_mean data.y_mean :=
sorry

end correct_regression_equation_l3694_369480


namespace box_content_theorem_l3694_369446

theorem box_content_theorem (total : ℕ) (pencil : ℕ) (pen : ℕ) (both : ℕ) :
  total = 12 →
  pencil = 7 →
  pen = 4 →
  both = 3 →
  total - (pencil + pen - both) = 4 := by
  sorry

end box_content_theorem_l3694_369446


namespace arithmetic_sequence_sum_l3694_369413

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 1 + a 2 + a 3 + a 4 = 30) →
  (a 2 + a 3 = 15) :=
by sorry

end arithmetic_sequence_sum_l3694_369413


namespace perpendicular_line_equation_l3694_369485

-- Define the given line
def given_line (x y : ℝ) : Prop := 3*x + y + 5 = 0

-- Define the point P
def point_P : ℝ × ℝ := (2, 1)

-- Define the perpendicular line l
def line_l (x y : ℝ) : Prop := x - 3*y + 1 = 0

-- Theorem statement
theorem perpendicular_line_equation :
  (∀ x y : ℝ, given_line x y → (line_l x y → ¬given_line x y)) ∧
  line_l point_P.1 point_P.2 :=
sorry

end perpendicular_line_equation_l3694_369485


namespace geometric_sequence_sum_property_l3694_369478

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The sum of two consecutive terms in a sequence -/
def ConsecutiveSum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a n + a (n + 1)

theorem geometric_sequence_sum_property
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_sum1 : ConsecutiveSum a 1 = 16)
  (h_sum2 : ConsecutiveSum a 3 = 24) :
  ConsecutiveSum a 7 = 54 :=
sorry

end geometric_sequence_sum_property_l3694_369478


namespace power_function_properties_l3694_369408

def f (m n : ℕ+) (x : ℝ) : ℝ := x ^ (m.val / n.val)

theorem power_function_properties (m n : ℕ+) (h_coprime : Nat.Coprime m.val n.val) :
  (∀ x, m.val % 2 = 1 ∧ n.val % 2 = 1 → f m n (-x) = -f m n x) ∧
  (∀ x, m.val % 2 = 0 ∧ n.val % 2 = 1 → f m n (-x) = f m n x) :=
sorry

end power_function_properties_l3694_369408


namespace min_reciprocal_sum_l3694_369483

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 2) :
  (1 / a + 1 / b) ≥ 2 + Real.sqrt 3 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 2 ∧ 1 / a₀ + 1 / b₀ = 2 + Real.sqrt 3 := by
  sorry

end min_reciprocal_sum_l3694_369483


namespace hyperbola_to_ellipse_l3694_369473

/-- Given a hyperbola with equation x²/4 - y²/12 = 1, prove that the ellipse with the foci of the
hyperbola as its vertices and the vertices of the hyperbola as its foci has the equation
x²/16 + y²/12 = 1 -/
theorem hyperbola_to_ellipse (x y : ℝ) :
  (x^2 / 4 - y^2 / 12 = 1) →
  ∃ (x' y' : ℝ), (x'^2 / 16 + y'^2 / 12 = 1 ∧
    (∀ (f_x f_y : ℝ), (f_x^2 / 4 - f_y^2 / 12 = 1 ∧ f_y = 0) →
      ((x' = f_x ∧ y' = 0) ∨ (x' = -f_x ∧ y' = 0))) ∧
    (∀ (v_x v_y : ℝ), (v_x^2 / 4 - v_y^2 / 12 = 1 ∧ v_y = 0) →
      (∃ (c : ℝ), x'^2 / 16 + y'^2 / 12 = 1 ∧ x'^2 - y'^2 = c^2 ∧ (v_x = c ∨ v_x = -c)))) :=
by sorry

end hyperbola_to_ellipse_l3694_369473


namespace intersection_and_complement_l3694_369452

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (1 - x)}

-- State the theorem
theorem intersection_and_complement :
  (M ∩ N = {x : ℝ | -2 ≤ x ∧ x ≤ 1}) ∧
  (Nᶜ = {x : ℝ | x > 1}) := by
  sorry

end intersection_and_complement_l3694_369452
