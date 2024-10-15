import Mathlib

namespace NUMINAMATH_CALUDE_f_sum_symmetric_l555_55532

def is_transformation (f g : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x : ℝ, f x = g (a * x + b) + c

theorem f_sum_symmetric (f : ℝ → ℝ) (hf : Differentiable ℝ f)
  (h1 : ∀ x : ℝ, f x + f (1 - x) = 10)
  (h2 : ∀ x : ℝ, f (1 + x) = 3 + f x)
  (h3 : ∃ g : ℝ → ℝ, is_transformation f g) :
  ∀ x : ℝ, f x + f (-x) = 7 :=
sorry

end NUMINAMATH_CALUDE_f_sum_symmetric_l555_55532


namespace NUMINAMATH_CALUDE_cos_sin_relation_l555_55508

theorem cos_sin_relation (α : ℝ) (h : Real.cos (α - π/5) = 5/13) :
  Real.sin (α - 7*π/10) = -5/13 := by sorry

end NUMINAMATH_CALUDE_cos_sin_relation_l555_55508


namespace NUMINAMATH_CALUDE_modular_inverse_of_3_mod_29_l555_55556

theorem modular_inverse_of_3_mod_29 : ∃ x : ℕ, x < 29 ∧ (3 * x) % 29 = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_3_mod_29_l555_55556


namespace NUMINAMATH_CALUDE_potato_count_l555_55564

/-- Given the initial number of potatoes and the number of new potatoes left after rabbits ate some,
    prove that the total number of potatoes is equal to the sum of the initial number and the number of new potatoes left. -/
theorem potato_count (initial : ℕ) (new_left : ℕ) : 
  initial + new_left = initial + new_left :=
by sorry

end NUMINAMATH_CALUDE_potato_count_l555_55564


namespace NUMINAMATH_CALUDE_diagonal_not_parallel_to_sides_l555_55574

theorem diagonal_not_parallel_to_sides (n : ℕ) (h : n > 0) :
  n * (2 * n - 3) > 2 * n * (n - 2) := by
  sorry

#check diagonal_not_parallel_to_sides

end NUMINAMATH_CALUDE_diagonal_not_parallel_to_sides_l555_55574


namespace NUMINAMATH_CALUDE_smallest_scalene_perimeter_l555_55524

-- Define a scalene triangle with integer side lengths
def ScaleneTriangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a + b > c ∧ b + c > a ∧ a + c > b

-- Theorem statement
theorem smallest_scalene_perimeter :
  ∀ a b c : ℕ, ScaleneTriangle a b c → a + b + c ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_scalene_perimeter_l555_55524


namespace NUMINAMATH_CALUDE_next_number_is_two_l555_55596

-- Define the sequence pattern
def sequence_pattern (n : ℕ) : ℕ → ℕ
| 0 => 1
| m + 1 => 
  let peak := n + 1
  let cycle_length := 2 * peak - 1
  let position := (m + 1) % cycle_length
  if position < peak then position + 1
  else 2 * peak - position - 1

-- Define the specific sequence from the problem
def given_sequence : List ℕ := [1, 1, 2, 1, 2, 3, 2, 1, 2, 3, 4, 3, 1, 2, 3, 4, 5, 4, 2, 1, 2, 3, 4, 5, 6, 5, 3, 1, 2, 3, 4, 5, 6, 7, 6, 4, 2, 1, 2, 3, 4, 5, 6, 7, 8, 7, 5, 3, 1]

-- Theorem to prove
theorem next_number_is_two : 
  ∃ (n : ℕ), sequence_pattern n (given_sequence.length) = 2 :=
by sorry

end NUMINAMATH_CALUDE_next_number_is_two_l555_55596


namespace NUMINAMATH_CALUDE_abc_inequality_l555_55566

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) :
  9 * a * b * c ≤ a * b + a * c + b * c ∧ a * b + a * c + b * c < 1/4 + 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l555_55566


namespace NUMINAMATH_CALUDE_min_a_value_l555_55569

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * Real.log x - 3 * x^2 - 11 * x

-- Define the inequality condition
def inequality_condition (a : ℝ) : Prop :=
  ∀ x > 0, f x ≤ (a - 3) * x^2 + (2 * a - 13) * x + 1

-- State the theorem
theorem min_a_value :
  ∃ a : ℤ, (inequality_condition a ∧ ∀ b : ℤ, b < a → ¬inequality_condition b) :=
sorry

end

end NUMINAMATH_CALUDE_min_a_value_l555_55569


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l555_55595

theorem geometric_sequence_problem (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  q > 0 ∧ 
  (∀ n, a (n + 1) = a n * q) ∧ 
  (∀ n, a n > 0) ∧
  (a 1 = 1 / q^2) ∧
  (S 5 = S 2 + 2) →
  q = (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l555_55595


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_implies_m_l555_55534

/-- Given a hyperbola (x²/m² - y² = 1) with m > 0, if one of its asymptotes
    is the line x + √3y = 0, then m = √3 -/
theorem hyperbola_asymptote_implies_m (m : ℝ) :
  m > 0 →
  (∃ x y : ℝ, x^2 / m^2 - y^2 = 1) →
  (∃ x y : ℝ, x + Real.sqrt 3 * y = 0) →
  m = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_implies_m_l555_55534


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l555_55504

theorem more_girls_than_boys (total_pupils : ℕ) (girls : ℕ) 
  (h1 : total_pupils = 926)
  (h2 : girls = 692)
  (h3 : girls > total_pupils - girls) :
  girls - (total_pupils - girls) = 458 := by
sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l555_55504


namespace NUMINAMATH_CALUDE_stones_can_be_combined_l555_55540

/-- Definition of similar sizes -/
def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

/-- A step in the combining process -/
inductive combine_step (stones : List ℕ) : List ℕ → Prop
  | combine (x y : ℕ) (rest : List ℕ) :
      x ∈ stones →
      y ∈ stones →
      similar_sizes x y →
      combine_step stones ((x + y) :: (stones.filter (λ z ↦ z ≠ x ∧ z ≠ y)))

/-- The transitive closure of combine_step -/
def can_combine := Relation.ReflTransGen combine_step

/-- The main theorem -/
theorem stones_can_be_combined (initial_stones : List ℕ) :
  ∃ (final_pile : ℕ), can_combine initial_stones [final_pile] :=
sorry

end NUMINAMATH_CALUDE_stones_can_be_combined_l555_55540


namespace NUMINAMATH_CALUDE_coin_problem_l555_55522

/-- Represents the number of different coin values that can be made -/
def different_values (x y : ℕ) : ℕ := 29 - (3 * x + 2 * y) / 2

/-- The coin problem -/
theorem coin_problem (total : ℕ) (values : ℕ) :
  total = 12 ∧ values = 21 →
  ∃ x y : ℕ, x + y = total ∧ different_values x y = values ∧ y = 7 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l555_55522


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l555_55501

theorem unique_four_digit_number : ∃! n : ℕ,
  1000 ≤ n ∧ n < 10000 ∧
  n % 131 = 112 ∧
  n % 132 = 98 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l555_55501


namespace NUMINAMATH_CALUDE_product_of_roots_l555_55572

theorem product_of_roots (x : ℂ) :
  2 * x^3 - 3 * x^2 - 10 * x + 14 = 0 →
  ∃ r₁ r₂ r₃ : ℂ, (x - r₁) * (x - r₂) * (x - r₃) = 2 * x^3 - 3 * x^2 - 10 * x + 14 ∧ r₁ * r₂ * r₃ = -7 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l555_55572


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l555_55519

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- Define the interval [0, 3]
def interval : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

-- Theorem statement
theorem f_max_min_on_interval :
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y) ∧
  (∀ x ∈ interval, f x ≤ 5) ∧
  (∀ x ∈ interval, -15 ≤ f x) ∧
  (∃ x ∈ interval, f x = 5) ∧
  (∃ x ∈ interval, f x = -15) :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l555_55519


namespace NUMINAMATH_CALUDE_katherines_fruit_ratio_l555_55573

/-- Katherine's fruit problem -/
theorem katherines_fruit_ratio : ∀ (pears apples bananas : ℕ),
  apples = 4 →
  bananas = 5 →
  pears + apples + bananas = 21 →
  pears / apples = 3 :=
by
  sorry

#check katherines_fruit_ratio

end NUMINAMATH_CALUDE_katherines_fruit_ratio_l555_55573


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l555_55511

theorem simplify_complex_expression (a : ℝ) (h : a > 0) :
  Real.sqrt ((2 * a) / ((1 + a) * (1 + a) ^ (1/3))) *
  ((4 + 8 / a + 4 / a^2) / Real.sqrt 2) ^ (1/3) =
  (2 * a^(5/6)) / a := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l555_55511


namespace NUMINAMATH_CALUDE_last_car_probability_2012_l555_55527

/-- Represents the parking procedure for a given number of spots. -/
def ParkingProcedure (n : ℕ) : Type :=
  Unit

/-- Calculates the probability of the last car parking in spot 1 given the parking procedure. -/
noncomputable def lastCarProbability (n : ℕ) (proc : ParkingProcedure n) : ℚ :=
  sorry

/-- The theorem stating the probability of the last car parking in spot 1 for 2012 spots. -/
theorem last_car_probability_2012 :
  ∃ (proc : ParkingProcedure 2012), lastCarProbability 2012 proc = 1 / 2062300 :=
by
  sorry

end NUMINAMATH_CALUDE_last_car_probability_2012_l555_55527


namespace NUMINAMATH_CALUDE_cube_root_floor_equality_l555_55516

theorem cube_root_floor_equality (n : ℕ) :
  ⌊(n : ℝ)^(1/3) + (n + 1 : ℝ)^(1/3)⌋ = ⌊(8*n + 3 : ℝ)^(1/3)⌋ :=
sorry

end NUMINAMATH_CALUDE_cube_root_floor_equality_l555_55516


namespace NUMINAMATH_CALUDE_jelly_bean_count_l555_55535

/-- The number of red jelly beans in one bag -/
def red_in_bag : ℕ := 24

/-- The number of white jelly beans in one bag -/
def white_in_bag : ℕ := 18

/-- The number of bags needed to fill the fishbowl -/
def bags_to_fill : ℕ := 3

/-- The total number of red and white jelly beans in the fishbowl -/
def total_red_white : ℕ := (red_in_bag + white_in_bag) * bags_to_fill

theorem jelly_bean_count : total_red_white = 126 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_count_l555_55535


namespace NUMINAMATH_CALUDE_bubble_gum_count_l555_55521

theorem bubble_gum_count (total_cost : ℕ) (cost_per_piece : ℕ) (h1 : total_cost = 2448) (h2 : cost_per_piece = 18) :
  total_cost / cost_per_piece = 136 := by
  sorry

end NUMINAMATH_CALUDE_bubble_gum_count_l555_55521


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l555_55541

/-- Given a hyperbola and a circle with specific properties, prove the eccentricity of the hyperbola -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + y^2 = 1}
  let asymptote := {(x, y) : ℝ × ℝ | y = (b / a) * x}
  let chord_length := Real.sqrt 3
  (∃ (p q : ℝ × ℝ), p ∈ asymptote ∧ q ∈ asymptote ∧ p ∈ circle ∧ q ∈ circle ∧
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord_length) →
  let c := Real.sqrt (a^2 + b^2)
  c / a = 2 / 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l555_55541


namespace NUMINAMATH_CALUDE_right_triangle_area_l555_55565

theorem right_triangle_area (a b : ℝ) (h1 : a = 25) (h2 : b = 20) :
  (1 / 2 : ℝ) * a * b = 250 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l555_55565


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_16_l555_55538

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_8_with_digit_sum_16 :
  ∀ n : ℕ, is_three_digit n → n % 8 = 0 → digit_sum n = 16 → n ≤ 880 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_16_l555_55538


namespace NUMINAMATH_CALUDE_integer_decimal_parts_sqrt10_l555_55544

theorem integer_decimal_parts_sqrt10 (a b : ℝ) : 
  (a = ⌊6 - Real.sqrt 10⌋) → 
  (b = 6 - Real.sqrt 10 - a) → 
  (2 * a + Real.sqrt 10) * b = 6 := by
sorry

end NUMINAMATH_CALUDE_integer_decimal_parts_sqrt10_l555_55544


namespace NUMINAMATH_CALUDE_binary_110101_is_53_l555_55531

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enum b).foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110101_is_53 :
  binary_to_decimal [true, false, true, false, true, true] = 53 := by
  sorry

end NUMINAMATH_CALUDE_binary_110101_is_53_l555_55531


namespace NUMINAMATH_CALUDE_quadratic_root_equivalence_l555_55546

theorem quadratic_root_equivalence (a b c : ℝ) (ha : a ≠ 0) :
  (a + b + c = 0) ↔ (a * 1^2 + b * 1 + c = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_equivalence_l555_55546


namespace NUMINAMATH_CALUDE_install_time_proof_l555_55559

/-- The time required to install the remaining windows -/
def time_to_install_remaining (total_windows : ℕ) (installed_windows : ℕ) (time_per_window : ℕ) : ℕ :=
  (total_windows - installed_windows) * time_per_window

/-- Proof that the time to install remaining windows is 36 hours -/
theorem install_time_proof (total_windows : ℕ) (installed_windows : ℕ) (time_per_window : ℕ)
  (h1 : total_windows = 14)
  (h2 : installed_windows = 5)
  (h3 : time_per_window = 4) :
  time_to_install_remaining total_windows installed_windows time_per_window = 36 :=
by sorry

end NUMINAMATH_CALUDE_install_time_proof_l555_55559


namespace NUMINAMATH_CALUDE_tournament_handshakes_eq_24_l555_55536

/-- The number of handshakes in a tournament with 4 teams of 2 players each -/
def tournament_handshakes : ℕ :=
  let num_teams : ℕ := 4
  let players_per_team : ℕ := 2
  let total_players : ℕ := num_teams * players_per_team
  let handshakes_per_player : ℕ := total_players - players_per_team
  (total_players * handshakes_per_player) / 2

theorem tournament_handshakes_eq_24 : tournament_handshakes = 24 := by
  sorry

end NUMINAMATH_CALUDE_tournament_handshakes_eq_24_l555_55536


namespace NUMINAMATH_CALUDE_book_arrangement_and_selection_l555_55542

/-- Given 3 math books, 4 physics books, and 2 chemistry books, prove:
    1. The number of arrangements keeping books of the same subject together
    2. The number of ways to select exactly 2 math books, 2 physics books, and 1 chemistry book
    3. The number of ways to select 5 books with at least 1 math book -/
theorem book_arrangement_and_selection 
  (math_books : ℕ) (physics_books : ℕ) (chemistry_books : ℕ) 
  (h_math : math_books = 3) 
  (h_physics : physics_books = 4) 
  (h_chemistry : chemistry_books = 2) :
  (-- 1. Number of arrangements
   (Nat.factorial math_books) * (Nat.factorial physics_books) * 
   (Nat.factorial chemistry_books) * (Nat.factorial 3) = 1728) ∧ 
  (-- 2. Number of ways to select 2 math, 2 physics, 1 chemistry
   (Nat.choose math_books 2) * (Nat.choose physics_books 2) * 
   (Nat.choose chemistry_books 1) = 36) ∧
  (-- 3. Number of ways to select 5 books with at least 1 math
   (Nat.choose (math_books + physics_books + chemistry_books) 5) - 
   (Nat.choose (physics_books + chemistry_books) 5) = 120) := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_and_selection_l555_55542


namespace NUMINAMATH_CALUDE_orangeade_water_ratio_l555_55561

/-- Represents the orangeade mixing and selling scenario over two days -/
structure OrangeadeScenario where
  orange_juice : ℝ  -- Amount of orange juice used (same for both days)
  water_day1 : ℝ    -- Amount of water used on day 1
  water_day2 : ℝ    -- Amount of water used on day 2
  price_day1 : ℝ    -- Price per glass on day 1
  price_day2 : ℝ    -- Price per glass on day 2
  glasses_day1 : ℝ  -- Number of glasses sold on day 1
  glasses_day2 : ℝ  -- Number of glasses sold on day 2

/-- The conditions of the orangeade scenario -/
def scenario_conditions (s : OrangeadeScenario) : Prop :=
  s.orange_juice > 0 ∧
  s.water_day1 = s.orange_juice ∧
  s.price_day1 = 0.48 ∧
  s.glasses_day1 * (s.orange_juice + s.water_day1) = s.glasses_day2 * (s.orange_juice + s.water_day2) ∧
  s.price_day1 * s.glasses_day1 = s.price_day2 * s.glasses_day2

/-- The main theorem: under the given conditions, the ratio of water used on day 2 to orange juice is 1:1 -/
theorem orangeade_water_ratio (s : OrangeadeScenario) 
  (h : scenario_conditions s) : s.water_day2 = s.orange_juice :=
sorry


end NUMINAMATH_CALUDE_orangeade_water_ratio_l555_55561


namespace NUMINAMATH_CALUDE_bakery_flour_usage_l555_55588

theorem bakery_flour_usage :
  0.2 + 0.1 + 0.15 + 0.05 + 0.1 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_bakery_flour_usage_l555_55588


namespace NUMINAMATH_CALUDE_complex_quadrant_l555_55517

theorem complex_quadrant (z : ℂ) (h : z * (1 + Complex.I) = -2 * Complex.I) :
  z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l555_55517


namespace NUMINAMATH_CALUDE_quadratic_factorization_l555_55571

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 6 * x - 8 = 2 * (x - 4) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l555_55571


namespace NUMINAMATH_CALUDE_farm_animals_l555_55502

theorem farm_animals (goats chickens ducks pigs : ℕ) : 
  goats = 66 →
  chickens = 2 * goats →
  ducks = (goats + chickens) / 2 →
  pigs = ducks / 3 →
  goats - pigs = 33 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l555_55502


namespace NUMINAMATH_CALUDE_square_fold_visible_area_l555_55575

theorem square_fold_visible_area (side_length : ℝ) (ao_length : ℝ) : 
  side_length = 1 → ao_length = 1/3 → 
  (visible_area : ℝ) = side_length * ao_length :=
by sorry

end NUMINAMATH_CALUDE_square_fold_visible_area_l555_55575


namespace NUMINAMATH_CALUDE_cost_of_three_l555_55560

/-- Represents the prices of fruits and vegetables -/
structure Prices where
  apples : ℝ
  bananas : ℝ
  cantaloupe : ℝ
  dates : ℝ
  eggplant : ℝ

/-- The total cost of all items is $30 -/
def total_cost (p : Prices) : Prop :=
  p.apples + p.bananas + p.cantaloupe + p.dates + p.eggplant = 30

/-- The carton of dates costs twice as much as the sack of apples -/
def dates_cost (p : Prices) : Prop :=
  p.dates = 2 * p.apples

/-- The price of cantaloupe equals price of apples minus price of bananas -/
def cantaloupe_cost (p : Prices) : Prop :=
  p.cantaloupe = p.apples - p.bananas

/-- The price of eggplant is the sum of apples and bananas prices -/
def eggplant_cost (p : Prices) : Prop :=
  p.eggplant = p.apples + p.bananas

/-- The main theorem: Given the conditions, the cost of bananas, cantaloupe, and eggplant is $12 -/
theorem cost_of_three (p : Prices) 
  (h1 : total_cost p) 
  (h2 : dates_cost p) 
  (h3 : cantaloupe_cost p) 
  (h4 : eggplant_cost p) : 
  p.bananas + p.cantaloupe + p.eggplant = 12 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_three_l555_55560


namespace NUMINAMATH_CALUDE_additional_distance_for_target_average_speed_l555_55586

/-- Proves that given an initial trip of 20 miles at 40 mph, an additional 90 miles
    driven at 60 mph will result in an average speed of 55 mph for the entire trip. -/
theorem additional_distance_for_target_average_speed
  (initial_distance : ℝ) (initial_speed : ℝ) (second_speed : ℝ) (target_avg_speed : ℝ)
  (additional_distance : ℝ) :
  initial_distance = 20 →
  initial_speed = 40 →
  second_speed = 60 →
  target_avg_speed = 55 →
  additional_distance = 90 →
  (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = target_avg_speed :=
by sorry

end NUMINAMATH_CALUDE_additional_distance_for_target_average_speed_l555_55586


namespace NUMINAMATH_CALUDE_remainder_divisibility_l555_55509

theorem remainder_divisibility (y : ℤ) : 
  ∃ k : ℤ, y = 288 * k + 45 → ∃ m : ℤ, y = 24 * m + 21 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l555_55509


namespace NUMINAMATH_CALUDE_canned_food_bins_l555_55563

theorem canned_food_bins (soup : ℝ) (vegetables : ℝ) (pasta : ℝ)
  (h1 : soup = 0.125)
  (h2 : vegetables = 0.125)
  (h3 : pasta = 0.5) :
  soup + vegetables + pasta = 0.75 := by
sorry

end NUMINAMATH_CALUDE_canned_food_bins_l555_55563


namespace NUMINAMATH_CALUDE_overlapping_triangle_is_equilateral_l555_55550

/-- Represents a right-angled triangle -/
structure RightTriangle where
  base : ℝ
  height : ℝ

/-- Represents the overlapping triangle formed by two identical right-angled triangles -/
structure OverlappingTriangle where
  original : RightTriangle
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- 
Given two identical right-angled triangles arranged such that the right angle vertex 
of one triangle lies on the side of the other, the resulting overlapping triangle is equilateral.
-/
theorem overlapping_triangle_is_equilateral (t : RightTriangle) 
  (ot : OverlappingTriangle) (h : ot.original = t) : 
  ot.side1 = ot.side2 ∧ ot.side2 = ot.side3 := by
  sorry


end NUMINAMATH_CALUDE_overlapping_triangle_is_equilateral_l555_55550


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l555_55512

theorem regular_polygon_sides (interior_angle : ℝ) (n : ℕ) : 
  interior_angle = 120 → (n : ℝ) * (180 - interior_angle) = 360 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l555_55512


namespace NUMINAMATH_CALUDE_flute_cost_calculation_l555_55552

/-- The cost of Jason's purchases at the music store -/
def total_spent : ℝ := 158.35

/-- The cost of the music tool -/
def music_tool_cost : ℝ := 8.89

/-- The cost of the song book -/
def song_book_cost : ℝ := 7

/-- The cost of the flute -/
def flute_cost : ℝ := total_spent - (music_tool_cost + song_book_cost)

theorem flute_cost_calculation : flute_cost = 142.46 := by
  sorry

end NUMINAMATH_CALUDE_flute_cost_calculation_l555_55552


namespace NUMINAMATH_CALUDE_digit_move_correction_l555_55505

theorem digit_move_correction : ∃ (a b c : ℕ), 
  (a = 101 ∧ b = 102 ∧ c = 1) ∧ 
  (a - b ≠ c) ∧
  (a - 10^2 = c) := by
  sorry

end NUMINAMATH_CALUDE_digit_move_correction_l555_55505


namespace NUMINAMATH_CALUDE_sara_letters_total_l555_55570

/-- The number of letters Sara sent in January -/
def january_letters : ℕ := 6

/-- The number of letters Sara sent in February -/
def february_letters : ℕ := 9

/-- The number of letters Sara sent in March -/
def march_letters : ℕ := 3 * january_letters

/-- The total number of letters Sara sent -/
def total_letters : ℕ := january_letters + february_letters + march_letters

theorem sara_letters_total : total_letters = 33 := by
  sorry

end NUMINAMATH_CALUDE_sara_letters_total_l555_55570


namespace NUMINAMATH_CALUDE_max_value_of_symmetric_f_l555_55577

/-- A function f(x) that is symmetric about the line x = -2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := (1 - x^2) * (x^2 + a*x + b)

/-- The symmetry condition for f(x) about x = -2 -/
def is_symmetric (a b : ℝ) : Prop :=
  ∀ x, f a b x = f a b (-4 - x)

/-- The maximum value of f(x) is 16 when it's symmetric about x = -2 -/
theorem max_value_of_symmetric_f (a b : ℝ) (h : is_symmetric a b) :
  ∃ x₀, ∀ x, f a b x ≤ f a b x₀ ∧ f a b x₀ = 16 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_symmetric_f_l555_55577


namespace NUMINAMATH_CALUDE_percentage_relation_l555_55576

theorem percentage_relation (x a b : ℝ) (h1 : a = 0.06 * x) (h2 : b = 0.3 * x) :
  a = 0.2 * b := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l555_55576


namespace NUMINAMATH_CALUDE_octal_subtraction_correct_l555_55583

/-- Represents a number in base 8 -/
def OctalNumber := List Nat

/-- Converts a list of digits in base 8 to a natural number -/
def octal_to_nat (x : OctalNumber) : Nat :=
  x.foldr (fun digit acc => acc * 8 + digit) 0

/-- Subtracts two octal numbers -/
def octal_subtract (x y : OctalNumber) : OctalNumber :=
  sorry -- Implementation of octal subtraction

theorem octal_subtraction_correct :
  octal_subtract [7, 3, 2, 4] [3, 6, 5, 7] = [4, 4, 4, 5] :=
by sorry

end NUMINAMATH_CALUDE_octal_subtraction_correct_l555_55583


namespace NUMINAMATH_CALUDE_juan_running_time_l555_55523

/-- Given that Juan ran at a speed of 10.0 miles per hour and covered a distance of 800 miles,
    prove that the time he ran equals 80 hours. -/
theorem juan_running_time (speed : ℝ) (distance : ℝ) (h1 : speed = 10.0) (h2 : distance = 800) :
  distance / speed = 80 :=
by sorry

end NUMINAMATH_CALUDE_juan_running_time_l555_55523


namespace NUMINAMATH_CALUDE_sin_sum_l555_55579

theorem sin_sum (α β : ℝ) : Real.sin (α + β) = Real.sin α * Real.cos β + Real.cos α * Real.sin β := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_l555_55579


namespace NUMINAMATH_CALUDE_unique_triple_l555_55513

/-- Least common multiple of two positive integers -/
def lcm (x y : ℕ+) : ℕ+ := sorry

/-- The number of ordered triples (a,b,c) satisfying the given LCM conditions -/
def count_triples : ℕ := sorry

theorem unique_triple : count_triples = 1 := by sorry

end NUMINAMATH_CALUDE_unique_triple_l555_55513


namespace NUMINAMATH_CALUDE_rohans_salary_l555_55520

/-- Rohan's monthly salary in Rupees -/
def monthly_salary : ℝ := 10000

/-- Percentage of salary spent on food -/
def food_percentage : ℝ := 40

/-- Percentage of salary spent on house rent -/
def rent_percentage : ℝ := 20

/-- Percentage of salary spent on entertainment -/
def entertainment_percentage : ℝ := 10

/-- Percentage of salary spent on conveyance -/
def conveyance_percentage : ℝ := 10

/-- Rohan's savings at the end of the month in Rupees -/
def savings : ℝ := 2000

theorem rohans_salary :
  monthly_salary * (1 - (food_percentage + rent_percentage + entertainment_percentage + conveyance_percentage) / 100) = savings := by
  sorry

#check rohans_salary

end NUMINAMATH_CALUDE_rohans_salary_l555_55520


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l555_55507

theorem partial_fraction_decomposition :
  ∃ (A B : ℚ), A = 7/3 ∧ B = 5/3 ∧
  ∀ (x : ℝ), x ≠ 6 ∧ x ≠ -3 →
    (4*x - 3) / (x^2 - 3*x - 18) = A / (x - 6) + B / (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l555_55507


namespace NUMINAMATH_CALUDE_cos_squared_half_diff_l555_55567

theorem cos_squared_half_diff (α β : Real) 
  (h1 : Real.sin α + Real.sin β = Real.sqrt 6 / 3)
  (h2 : Real.cos α + Real.cos β = Real.sqrt 3 / 3) : 
  (Real.cos ((α - β) / 2))^2 = 1/4 := by
sorry

end NUMINAMATH_CALUDE_cos_squared_half_diff_l555_55567


namespace NUMINAMATH_CALUDE_car_distance_theorem_l555_55500

/-- Calculates the total distance traveled by a car over a given number of hours,
    where the car's speed increases by a fixed amount each hour. -/
def total_distance (initial_speed : ℕ) (speed_increase : ℕ) (hours : ℕ) : ℕ :=
  (List.range hours).foldl (fun acc h => acc + (initial_speed + h * speed_increase)) 0

/-- Theorem stating that a car traveling 40 km in the first hour and increasing speed by 2 km/h
    every hour will travel 600 km in 12 hours. -/
theorem car_distance_theorem : total_distance 40 2 12 = 600 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l555_55500


namespace NUMINAMATH_CALUDE_polynomial_ratio_condition_l555_55590

/-- A polynomial f(x) = x^2 - α x + 1 can be expressed as a ratio of two polynomials
    with non-negative coefficients if and only if α < 2. -/
theorem polynomial_ratio_condition (α : ℝ) :
  (∃ (P Q : ℝ → ℝ), (∀ x, P x ≥ 0 ∧ Q x ≥ 0) ∧
    (∀ x, x^2 - α * x + 1 = P x / Q x)) ↔ α < 2 :=
sorry

end NUMINAMATH_CALUDE_polynomial_ratio_condition_l555_55590


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l555_55592

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, |x - 1| < 2 → (x + 2) * (x - 3) < 0) ∧
  (∃ x : ℝ, (x + 2) * (x - 3) < 0 ∧ |x - 1| ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l555_55592


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l555_55533

/-- 
Given a line ax + by + 1 = 0 where the distance from the origin to this line is 1/2,
prove that the circles (x - a)² + y² = 1 and x² + (y - b)² = 1 are externally tangent.
-/
theorem circles_externally_tangent (a b : ℝ) 
  (h : (a^2 + b^2)⁻¹ = 1/4) : 
  let d := Real.sqrt (a^2 + b^2)
  d = 2 := by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l555_55533


namespace NUMINAMATH_CALUDE_f_continuous_iff_a_eq_one_l555_55548

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 0 then Real.exp (3 * x) else a + 5 * x

theorem f_continuous_iff_a_eq_one (a : ℝ) :
  Continuous (f a) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_f_continuous_iff_a_eq_one_l555_55548


namespace NUMINAMATH_CALUDE_school_play_tickets_l555_55598

theorem school_play_tickets (student_price adult_price adult_count total : ℕ) 
  (h1 : student_price = 6)
  (h2 : adult_price = 8)
  (h3 : adult_count = 12)
  (h4 : total = 216) :
  ∃ student_count : ℕ, student_count * student_price + adult_count * adult_price = total ∧ student_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_school_play_tickets_l555_55598


namespace NUMINAMATH_CALUDE_vector_BA_complex_l555_55557

/-- Given two complex numbers representing vectors OA and OB, 
    prove that the complex number representing vector BA is their difference. -/
theorem vector_BA_complex (OA OB : ℂ) (h1 : OA = 2 - 3*I) (h2 : OB = -3 + 2*I) :
  OA - OB = 5 - 5*I := by
  sorry

end NUMINAMATH_CALUDE_vector_BA_complex_l555_55557


namespace NUMINAMATH_CALUDE_expressions_not_always_equal_l555_55568

theorem expressions_not_always_equal :
  ∃ (a b c : ℝ), a + b + c = 0 ∧ a + b * c ≠ (a + b) * (a + c) := by
  sorry

end NUMINAMATH_CALUDE_expressions_not_always_equal_l555_55568


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l555_55594

theorem simplify_complex_fraction :
  (1 / ((1 / (Real.sqrt 5 + 2)) - (2 / (Real.sqrt 7 - 3)))) =
  ((Real.sqrt 5 + Real.sqrt 7 - 1) / (11 + 2 * Real.sqrt 35)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l555_55594


namespace NUMINAMATH_CALUDE_correct_average_l555_55580

theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 →
  initial_avg = 16 →
  incorrect_num = 26 →
  correct_num = 46 →
  (n : ℚ) * initial_avg - incorrect_num + correct_num = n * 18 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l555_55580


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l555_55506

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ ∀ x > 0, ¬ P x := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x > 0, x^2 - 2*x + 1 > 0) ↔ (∀ x > 0, x^2 - 2*x + 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l555_55506


namespace NUMINAMATH_CALUDE_subtraction_equality_l555_55554

theorem subtraction_equality : 8888888888888 - 4444444444444 = 4444444444444 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_equality_l555_55554


namespace NUMINAMATH_CALUDE_dog_treat_expenditure_l555_55578

/-- Represents the cost and nutritional value of dog treats -/
structure DogTreat where
  cost : ℚ
  np : ℕ

/-- Calculates the discounted price based on quantity and discount rate -/
def discountedPrice (regularPrice : ℚ) (quantity : ℕ) (discountRate : ℚ) : ℚ :=
  regularPrice * (1 - discountRate)

/-- Theorem: The total expenditure on dog treats for the month is $11.70 -/
theorem dog_treat_expenditure :
  let treatA : DogTreat := { cost := 0.1, np := 1 }
  let treatB : DogTreat := { cost := 0.15, np := 2 }
  let quantityA : ℕ := 50
  let quantityB : ℕ := 60
  let discountRateA : ℚ := 0.1
  let discountRateB : ℚ := 0.2
  let totalNP : ℕ := quantityA * treatA.np + quantityB * treatB.np
  let regularPriceA : ℚ := treatA.cost * quantityA
  let regularPriceB : ℚ := treatB.cost * quantityB
  let discountedPriceA : ℚ := discountedPrice regularPriceA quantityA discountRateA
  let discountedPriceB : ℚ := discountedPrice regularPriceB quantityB discountRateB
  let totalExpenditure : ℚ := discountedPriceA + discountedPriceB
  totalNP ≥ 40 ∧ totalExpenditure = 11.7 := by
  sorry


end NUMINAMATH_CALUDE_dog_treat_expenditure_l555_55578


namespace NUMINAMATH_CALUDE_chimney_bricks_chimney_bricks_proof_l555_55555

theorem chimney_bricks : ℕ → Prop :=
  fun n =>
    let brenda_rate := n / 12
    let brandon_rate := n / 15
    let combined_rate := n / 12 + n / 15 - 15
    6 * combined_rate = n →
    n = 900

-- The proof is omitted
theorem chimney_bricks_proof : chimney_bricks 900 := by sorry

end NUMINAMATH_CALUDE_chimney_bricks_chimney_bricks_proof_l555_55555


namespace NUMINAMATH_CALUDE_smallest_value_of_complex_sum_l555_55528

theorem smallest_value_of_complex_sum (a b c d : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_omega_power : ω^4 = 1)
  (h_omega_neq_one : ω ≠ 1) :
  ∃ (m : ℝ), m = Real.sqrt (9/2) ∧
    ∀ (x : ℂ), x = a + b*ω + c*ω^2 + d*ω^3 → Complex.abs x ≥ m :=
  sorry

end NUMINAMATH_CALUDE_smallest_value_of_complex_sum_l555_55528


namespace NUMINAMATH_CALUDE_race_distance_l555_55551

/-- Represents a race between two participants A and B -/
structure Race where
  distance : ℝ
  timeA : ℝ
  timeB : ℝ
  speedA : ℝ
  speedB : ℝ

/-- The conditions of the race -/
def raceConditions (r : Race) : Prop :=
  r.timeA = 18 ∧
  r.timeB = r.timeA + 7 ∧
  r.distance = r.speedA * r.timeA ∧
  r.distance = r.speedB * r.timeB ∧
  r.distance - r.speedB * r.timeA = 56

theorem race_distance (r : Race) (h : raceConditions r) : r.distance = 200 := by
  sorry

#check race_distance

end NUMINAMATH_CALUDE_race_distance_l555_55551


namespace NUMINAMATH_CALUDE_units_digit_of_sum_l555_55587

-- Define the function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem units_digit_of_sum : unitsDigit ((56 ^ 78) + (87 ^ 65)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_l555_55587


namespace NUMINAMATH_CALUDE_triangle_inequality_on_side_l555_55518

/-- Given a triangle ABC and a point O on side AB (not coinciding with A or B),
    prove that OC · AB < OA · BC + OB · AC. -/
theorem triangle_inequality_on_side (A B C O : EuclideanSpace ℝ (Fin 2)) :
  O ≠ A →
  O ≠ B →
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ O = A + t • (B - A) →
  ‖C - O‖ * ‖B - A‖ < ‖O - A‖ * ‖C - B‖ + ‖O - B‖ * ‖C - A‖ := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_on_side_l555_55518


namespace NUMINAMATH_CALUDE_complex_power_difference_zero_l555_55558

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference_zero : (1 + i)^20 - (1 - i)^20 = 0 := by sorry

end NUMINAMATH_CALUDE_complex_power_difference_zero_l555_55558


namespace NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l555_55553

/-- The height of a right circular cylinder inscribed in a hemisphere --/
theorem cylinder_height_in_hemisphere (r_cylinder r_hemisphere : ℝ) 
  (h_cylinder : r_cylinder = 3)
  (h_hemisphere : r_hemisphere = 7)
  (h_inscribed : r_cylinder ≤ r_hemisphere) :
  Real.sqrt (r_hemisphere^2 - r_cylinder^2) = 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l555_55553


namespace NUMINAMATH_CALUDE_polyhedron_with_specific_projections_l555_55545

/-- A polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices. -/
def Polyhedron : Type := sorry

/-- A plane in three-dimensional space. -/
def Plane : Type := sorry

/-- A projection of a polyhedron onto a plane. -/
def projection (p : Polyhedron) (plane : Plane) : Set (ℝ × ℝ) := sorry

/-- A triangle is a polygon with three sides. -/
def isTriangle (s : Set (ℝ × ℝ)) : Prop := sorry

/-- A quadrilateral is a polygon with four sides. -/
def isQuadrilateral (s : Set (ℝ × ℝ)) : Prop := sorry

/-- A pentagon is a polygon with five sides. -/
def isPentagon (s : Set (ℝ × ℝ)) : Prop := sorry

/-- Two planes are perpendicular if they intersect at a right angle. -/
def arePerpendicular (p1 p2 : Plane) : Prop := sorry

theorem polyhedron_with_specific_projections :
  ∃ (p : Polyhedron) (p1 p2 p3 : Plane),
    arePerpendicular p1 p2 ∧
    arePerpendicular p2 p3 ∧
    arePerpendicular p3 p1 ∧
    isTriangle (projection p p1) ∧
    isQuadrilateral (projection p p2) ∧
    isPentagon (projection p p3) := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_with_specific_projections_l555_55545


namespace NUMINAMATH_CALUDE_francine_daily_drive_distance_l555_55593

/-- The number of days Francine doesn't go to work each week -/
def days_off_per_week : ℕ := 3

/-- The total distance Francine drives to work in 4 weeks (in km) -/
def total_distance_4_weeks : ℕ := 2240

/-- The number of weeks in the given period -/
def num_weeks : ℕ := 4

/-- The number of working days in a week -/
def work_days_per_week : ℕ := 7 - days_off_per_week

/-- The total number of working days in 4 weeks -/
def total_work_days : ℕ := work_days_per_week * num_weeks

/-- The distance Francine drives to work each day (in km) -/
def daily_distance : ℕ := total_distance_4_weeks / total_work_days

theorem francine_daily_drive_distance :
  daily_distance = 280 := by sorry

end NUMINAMATH_CALUDE_francine_daily_drive_distance_l555_55593


namespace NUMINAMATH_CALUDE_complex_magnitude_special_angle_l555_55599

theorem complex_magnitude_special_angle : 
  let z : ℂ := Complex.mk (Real.sin (π / 3)) (Real.cos (π / 6))
  ‖z‖ = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_special_angle_l555_55599


namespace NUMINAMATH_CALUDE_parabola_single_intersection_l555_55510

/-- A parabola with equation y = x^2 - x + k has only one intersection point with the x-axis if and only if k = 1/4 -/
theorem parabola_single_intersection (k : ℝ) : 
  (∃! x, x^2 - x + k = 0) ↔ k = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_single_intersection_l555_55510


namespace NUMINAMATH_CALUDE_cube_surface_area_l555_55543

theorem cube_surface_area (volume : ℝ) (surface_area : ℝ) : 
  volume = 343 → surface_area = 294 → 
  (∃ (side : ℝ), volume = side^3 ∧ surface_area = 6 * side^2) := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l555_55543


namespace NUMINAMATH_CALUDE_min_absolute_T_l555_55589

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

def T (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (a n) + (a (n+1)) + (a (n+2)) + (a (n+3)) + (a (n+4)) + (a (n+5))

theorem min_absolute_T (a : ℕ → ℤ) :
  arithmetic_sequence a →
  a 5 = 15 →
  a 10 = -10 →
  (∃ n : ℕ, ∀ m : ℕ, |T a n| ≤ |T a m|) →
  (∃ n : ℕ, n = 5 ∨ n = 6 ∧ ∀ m : ℕ, |T a n| ≤ |T a m|) :=
by sorry

end NUMINAMATH_CALUDE_min_absolute_T_l555_55589


namespace NUMINAMATH_CALUDE_triangle_perimeter_l555_55581

/-- Theorem: For a triangle with sides in the ratio 5:6:7 and the longest side measuring 280 cm, the perimeter is 720 cm. -/
theorem triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positive side lengths
  a / b = 5 / 6 →          -- Ratio of first two sides
  b / c = 6 / 7 →          -- Ratio of second two sides
  c = 280 →                -- Length of longest side
  a + b + c = 720 :=       -- Perimeter
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l555_55581


namespace NUMINAMATH_CALUDE_no_square_root_among_options_l555_55514

theorem no_square_root_among_options : ∃ (x : ℝ), x ^ 2 = 0 ∧
                                       ∃ (x : ℝ), x ^ 2 = (-2)^2 ∧
                                       ∃ (x : ℝ), x ^ 2 = |9| ∧
                                       ¬∃ (x : ℝ), x ^ 2 = -|(-5)| := by
  sorry

#check no_square_root_among_options

end NUMINAMATH_CALUDE_no_square_root_among_options_l555_55514


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l555_55597

theorem degree_to_radian_conversion (π : Real) :
  (180 : Real) * (π / 3) = 60 * π :=
by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l555_55597


namespace NUMINAMATH_CALUDE_sons_ages_l555_55515

def father_age : ℕ := 33
def youngest_son_age : ℕ := 2
def years_until_sum_equal : ℕ := 12

def is_valid_ages (middle_son_age oldest_son_age : ℕ) : Prop :=
  (father_age + years_until_sum_equal = 
   (youngest_son_age + years_until_sum_equal) + 
   (middle_son_age + years_until_sum_equal) + 
   (oldest_son_age + years_until_sum_equal)) ∧
  (middle_son_age > youngest_son_age) ∧
  (oldest_son_age > middle_son_age)

theorem sons_ages : 
  ∃ (middle_son_age oldest_son_age : ℕ),
    is_valid_ages middle_son_age oldest_son_age ∧
    middle_son_age = 3 ∧ oldest_son_age = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_sons_ages_l555_55515


namespace NUMINAMATH_CALUDE_total_hike_length_l555_55503

/-- Represents Ella's hike over three days -/
structure HikeData where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ

/-- Conditions of Ella's hike -/
def isValidHike (h : HikeData) : Prop :=
  h.day1 + h.day2 = 18 ∧
  (h.day1 + h.day3) / 2 = 12 ∧
  h.day2 + h.day3 = 24 ∧
  h.day2 + h.day3 = 20

/-- Theorem stating the total length of the trail -/
theorem total_hike_length (h : HikeData) (hValid : isValidHike h) :
  h.day1 + h.day2 + h.day3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_total_hike_length_l555_55503


namespace NUMINAMATH_CALUDE_system_solution_l555_55530

theorem system_solution (x y : ℝ) : 
  (1 / x + 1 / y = 2.25 ∧ x^2 / y + y^2 / x = 32.0625) ↔ 
  ((x = 4 ∧ y = 1/2) ∨ 
   (x = 1/12 * (-19 + Real.sqrt (1691/3)) ∧ 
    y = 1/12 * (-19 - Real.sqrt (1691/3)))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l555_55530


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_implies_product_l555_55537

theorem quadratic_inequality_solution_implies_product (a b : ℝ) :
  (∀ x : ℝ, ax^2 + bx + 1 > 0 ↔ -1 < x ∧ x < 1/3) →
  a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_implies_product_l555_55537


namespace NUMINAMATH_CALUDE_watch_dealer_profit_l555_55591

theorem watch_dealer_profit (n d : ℕ) (h1 : d > 0) : 
  (∃ m : ℕ, d = 3 * m) →
  (10 * n - 30 = 100) →
  (∀ k : ℕ, k < n → ¬(10 * k - 30 = 100)) →
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_watch_dealer_profit_l555_55591


namespace NUMINAMATH_CALUDE_equation_solution_l555_55525

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ (2 / x + (3 / x) / (6 / x) = 1.5) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l555_55525


namespace NUMINAMATH_CALUDE_chocolate_candy_cost_difference_l555_55549

/-- Calculates the cost difference between chocolates and candy bars --/
theorem chocolate_candy_cost_difference :
  let initial_money : ℚ := 50
  let candy_price : ℚ := 4
  let candy_discount_rate : ℚ := 0.2
  let candy_discount_threshold : ℕ := 3
  let candy_quantity : ℕ := 5
  let chocolate_price : ℚ := 6
  let chocolate_tax_rate : ℚ := 0.05
  let chocolate_quantity : ℕ := 4

  let candy_cost : ℚ := if candy_quantity ≥ candy_discount_threshold
    then candy_quantity * candy_price * (1 - candy_discount_rate)
    else candy_quantity * candy_price

  let chocolate_cost : ℚ := chocolate_quantity * chocolate_price * (1 + chocolate_tax_rate)

  chocolate_cost - candy_cost = 9.2 :=
by
  sorry


end NUMINAMATH_CALUDE_chocolate_candy_cost_difference_l555_55549


namespace NUMINAMATH_CALUDE_triangle_area_l555_55526

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop := sorry

-- Define the height of the triangle
def Height (A B C H : ℝ × ℝ) (h : ℝ) : Prop := sorry

-- Define the angles of the triangle
def Angle (A B C : ℝ × ℝ) (α : ℝ) : Prop := sorry

-- Define the area of a triangle
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

theorem triangle_area (A B C H : ℝ × ℝ) (h α γ : ℝ) :
  Triangle A B C →
  Height A B C H h →
  Angle B A C α →
  Angle B C A γ →
  TriangleArea A B C = (h^2 * Real.sin α) / (2 * Real.sin γ * Real.sin (α + γ)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l555_55526


namespace NUMINAMATH_CALUDE_inequality_solution_l555_55547

noncomputable def solution_set (a : ℝ) : Set ℝ :=
  {x | (a + 2) * x - 4 ≤ 2 * (x - 1)}

theorem inequality_solution (a : ℝ) (h : a > 0) :
  (0 < a ∧ a < 2 → solution_set a = {x | 1 < x ∧ x ≤ 2/a}) ∧
  (a = 2 → solution_set a = ∅) ∧
  (a > 2 → solution_set a = {x | 2/a ≤ x ∧ x < 1}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l555_55547


namespace NUMINAMATH_CALUDE_right_isosceles_triangle_median_area_l555_55582

theorem right_isosceles_triangle_median_area (h : ℝ) :
  h > 0 →
  let leg := h / Real.sqrt 2
  let area := (1 / 2) * leg * leg
  let median_area := area / 2
  (h = 16) → median_area = 32 := by sorry

end NUMINAMATH_CALUDE_right_isosceles_triangle_median_area_l555_55582


namespace NUMINAMATH_CALUDE_triangle_area_l555_55584

/-- The area of a triangle with base 4 and height 8 is 16 -/
theorem triangle_area : ℝ → ℝ → ℝ → Prop :=
  fun base height area =>
    base = 4 ∧ height = 8 →
    area = (base * height) / 2 →
    area = 16

/-- Proof of the theorem -/
lemma prove_triangle_area : triangle_area 4 8 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l555_55584


namespace NUMINAMATH_CALUDE_cosine_sum_simplification_l555_55539

theorem cosine_sum_simplification :
  Real.cos (2 * Real.pi / 15) + Real.cos (4 * Real.pi / 15) + 
  Real.cos (10 * Real.pi / 15) + Real.cos (14 * Real.pi / 15) = 
  (Real.sqrt 17 - 1) / 4 :=
by sorry

end NUMINAMATH_CALUDE_cosine_sum_simplification_l555_55539


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_T_l555_55585

def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_T_l555_55585


namespace NUMINAMATH_CALUDE_jana_travel_distance_l555_55529

/-- Calculates the total distance traveled by Jana given her walking and cycling rates and times. -/
theorem jana_travel_distance (walking_rate : ℝ) (walking_time : ℝ) (cycling_rate : ℝ) (cycling_time : ℝ) :
  walking_rate = 1 / 30 →
  walking_time = 45 →
  cycling_rate = 2 / 15 →
  cycling_time = 30 →
  walking_rate * walking_time + cycling_rate * cycling_time = 5.5 :=
by
  sorry

end NUMINAMATH_CALUDE_jana_travel_distance_l555_55529


namespace NUMINAMATH_CALUDE_initial_cookies_count_l555_55562

/-- The number of cookies Paul took out in 4 days -/
def cookies_taken_4_days : ℕ := 24

/-- The number of days Paul took cookies out -/
def days_taken : ℕ := 4

/-- The number of cookies remaining after a week -/
def cookies_remaining : ℕ := 28

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Proves that the initial number of cookies in the jar is 52 -/
theorem initial_cookies_count : ℕ := by
  sorry

end NUMINAMATH_CALUDE_initial_cookies_count_l555_55562
