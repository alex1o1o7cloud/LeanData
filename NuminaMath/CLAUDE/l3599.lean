import Mathlib

namespace NUMINAMATH_CALUDE_seokgi_paper_usage_l3599_359999

theorem seokgi_paper_usage (total : ℕ) (used : ℕ) (remaining : ℕ) : 
  total = 82 ∧ 
  remaining = total - used ∧ 
  remaining = used - 6 → 
  used = 44 := by sorry

end NUMINAMATH_CALUDE_seokgi_paper_usage_l3599_359999


namespace NUMINAMATH_CALUDE_average_z_squared_l3599_359914

theorem average_z_squared (z : ℝ) : 
  (0 + 2 * z^2 + 4 * z^2 + 8 * z^2 + 16 * z^2) / 5 = 6 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_average_z_squared_l3599_359914


namespace NUMINAMATH_CALUDE_bug_safe_probability_l3599_359916

theorem bug_safe_probability (r : ℝ) (h : r = 3) :
  let safe_radius := r - 1
  let total_volume := (4 / 3) * Real.pi * r^3
  let safe_volume := (4 / 3) * Real.pi * safe_radius^3
  safe_volume / total_volume = 8 / 27 := by
sorry

end NUMINAMATH_CALUDE_bug_safe_probability_l3599_359916


namespace NUMINAMATH_CALUDE_first_player_wins_l3599_359951

def Game := List Nat → List Nat

def validMove (n : Nat) (history : List Nat) : Prop :=
  n ∣ 328 ∧ n ∉ history ∧ ∀ m ∈ history, ¬(n ∣ m)

def gameOver (history : List Nat) : Prop :=
  328 ∈ history

def winningStrategy (strategy : Game) : Prop :=
  ∀ history : List Nat,
    ¬gameOver history →
    ∃ move,
      validMove move history ∧
      ∀ opponent_move,
        validMove opponent_move (move :: history) →
        gameOver (opponent_move :: move :: history)

theorem first_player_wins :
  ∃ strategy : Game, winningStrategy strategy :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l3599_359951


namespace NUMINAMATH_CALUDE_number_of_factors_7200_l3599_359940

theorem number_of_factors_7200 : Nat.card (Nat.divisors 7200) = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_7200_l3599_359940


namespace NUMINAMATH_CALUDE_optimization_problem_l3599_359926

theorem optimization_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  ((2 / x) + (1 / y) ≥ 9) ∧
  (4 * x^2 + y^2 ≥ 1/2) ∧
  (Real.sqrt (2 * x) + Real.sqrt y ≤ Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_optimization_problem_l3599_359926


namespace NUMINAMATH_CALUDE_min_distance_parallel_lines_l3599_359960

/-- Given two parallel lines and a point between them, prove the minimum distance from the point to a fixed point. -/
theorem min_distance_parallel_lines (x₀ y₀ : ℝ) :
  (∃ (xp yp xq yq : ℝ),
    (xp - 2*yp - 1 = 0) ∧
    (xq - 2*yq + 3 = 0) ∧
    (x₀ = (xp + xq) / 2) ∧
    (y₀ = (yp + yq) / 2) ∧
    (y₀ > -x₀ + 2)) →
  Real.sqrt ((x₀ - 4)^2 + y₀^2) ≥ Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_parallel_lines_l3599_359960


namespace NUMINAMATH_CALUDE_noah_doctor_visits_l3599_359991

/-- The number of holidays Noah took in a year -/
def total_holidays : ℕ := 36

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of times Noah visits the doctor each month -/
def doctor_visits_per_month : ℕ := total_holidays / months_in_year

theorem noah_doctor_visits :
  doctor_visits_per_month = 3 :=
sorry

end NUMINAMATH_CALUDE_noah_doctor_visits_l3599_359991


namespace NUMINAMATH_CALUDE_hexagon_triangle_area_l3599_359956

/-- The area of a regular hexagon with side length 2, topped by an equilateral triangle with side length 2, is 7√3 square units. -/
theorem hexagon_triangle_area : 
  let hexagon_side : ℝ := 2
  let triangle_side : ℝ := 2
  let hexagon_area : ℝ := (3 * Real.sqrt 3 / 2) * hexagon_side^2
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side^2
  hexagon_area + triangle_area = 7 * Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_hexagon_triangle_area_l3599_359956


namespace NUMINAMATH_CALUDE_problem_statement_l3599_359998

theorem problem_statement (a b : ℕ+) (h : 8 * (a : ℝ)^(a : ℝ) * (b : ℝ)^(b : ℝ) = 27 * (a : ℝ)^(b : ℝ) * (b : ℝ)^(a : ℝ)) : 
  (a : ℝ)^2 + (b : ℝ)^2 = 117 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3599_359998


namespace NUMINAMATH_CALUDE_number_problem_l3599_359906

theorem number_problem (x : ℝ) : 0.35 * x = 0.50 * x - 24 → x = 160 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3599_359906


namespace NUMINAMATH_CALUDE_flour_needed_l3599_359965

theorem flour_needed (total : ℕ) (added : ℕ) (needed : ℕ) : 
  total = 8 ∧ added = 2 → needed = 6 := by
  sorry

end NUMINAMATH_CALUDE_flour_needed_l3599_359965


namespace NUMINAMATH_CALUDE_gigi_mushrooms_l3599_359981

/-- The number of pieces each mushroom is cut into -/
def pieces_per_mushroom : ℕ := 4

/-- The number of mushroom pieces Kenny used -/
def kenny_pieces : ℕ := 38

/-- The number of mushroom pieces Karla used -/
def karla_pieces : ℕ := 42

/-- The number of mushroom pieces left on the cutting board -/
def remaining_pieces : ℕ := 8

/-- Theorem stating that the total number of whole mushrooms GiGi cut up is 22 -/
theorem gigi_mushrooms :
  (kenny_pieces + karla_pieces + remaining_pieces) / pieces_per_mushroom = 22 := by
  sorry

end NUMINAMATH_CALUDE_gigi_mushrooms_l3599_359981


namespace NUMINAMATH_CALUDE_equal_digit_probability_l3599_359911

def num_dice : ℕ := 5
def sides_per_die : ℕ := 20
def one_digit_sides : ℕ := 9
def two_digit_sides : ℕ := 11

theorem equal_digit_probability : 
  (num_dice.choose (num_dice / 2)) * 
  ((two_digit_sides : ℚ) / sides_per_die) ^ (num_dice / 2) * 
  ((one_digit_sides : ℚ) / sides_per_die) ^ (num_dice - num_dice / 2) = 
  1062889 / 128000000 := by sorry

end NUMINAMATH_CALUDE_equal_digit_probability_l3599_359911


namespace NUMINAMATH_CALUDE_forty_second_card_is_eight_of_spades_l3599_359910

-- Define the card suits
inductive Suit
| Hearts
| Spades

-- Define the card ranks
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

-- Define a card as a pair of rank and suit
structure Card where
  rank : Rank
  suit : Suit

-- Define the cycle of cards
def cardCycle : List Card := sorry

-- Define a function to get the nth card in the cycle
def nthCard (n : Nat) : Card := sorry

-- Theorem to prove
theorem forty_second_card_is_eight_of_spades :
  nthCard 42 = Card.mk Rank.Eight Suit.Spades := by sorry

end NUMINAMATH_CALUDE_forty_second_card_is_eight_of_spades_l3599_359910


namespace NUMINAMATH_CALUDE_largest_binomial_coefficient_sum_l3599_359976

theorem largest_binomial_coefficient_sum (n : ℕ) : 
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) → n ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_binomial_coefficient_sum_l3599_359976


namespace NUMINAMATH_CALUDE_factorial_200_less_than_100_pow_200_l3599_359978

-- Define factorial
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Theorem statement
theorem factorial_200_less_than_100_pow_200 :
  factorial 200 < 100^200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_200_less_than_100_pow_200_l3599_359978


namespace NUMINAMATH_CALUDE_birthday_number_l3599_359946

theorem birthday_number (T : ℕ) (x y : ℕ+) : 
  200 < T → T < 225 → T^2 = 4 * 10000 + x * 1000 + y * 100 + 29 → T = 223 := by
sorry

end NUMINAMATH_CALUDE_birthday_number_l3599_359946


namespace NUMINAMATH_CALUDE_son_age_proof_l3599_359920

theorem son_age_proof (father_age son_age : ℕ) : 
  father_age = 36 →
  4 * son_age = father_age →
  father_age - son_age = 27 →
  son_age = 9 := by
sorry

end NUMINAMATH_CALUDE_son_age_proof_l3599_359920


namespace NUMINAMATH_CALUDE_divides_n_l3599_359988

def n : ℕ := sorry

theorem divides_n : 1980 ∣ n := by sorry

end NUMINAMATH_CALUDE_divides_n_l3599_359988


namespace NUMINAMATH_CALUDE_gold_coin_percentage_is_49_percent_l3599_359966

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  bead_percentage : ℝ
  silver_coin_percentage : ℝ

/-- Calculates the percentage of gold coins in the urn -/
def gold_coin_percentage (urn : UrnComposition) : ℝ :=
  (1 - urn.bead_percentage) * (1 - urn.silver_coin_percentage)

/-- Theorem stating that for the given urn composition, 
    the percentage of gold coins is 49% -/
theorem gold_coin_percentage_is_49_percent 
  (urn : UrnComposition) 
  (h1 : urn.bead_percentage = 0.3) 
  (h2 : urn.silver_coin_percentage = 0.3) : 
  gold_coin_percentage urn = 0.49 := by
  sorry

#eval gold_coin_percentage ⟨0.3, 0.3⟩

end NUMINAMATH_CALUDE_gold_coin_percentage_is_49_percent_l3599_359966


namespace NUMINAMATH_CALUDE_count_true_propositions_l3599_359929

/-- Represents a line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The proposition p --/
def p (l1 l2 : Line) : Prop :=
  (∃ (k : ℝ), k ≠ 0 ∧ l1.a = k * l2.a ∧ l1.b = k * l2.b) → l1.a * l2.b - l2.a * l1.b = 0

/-- The converse of p --/
def p_converse (l1 l2 : Line) : Prop :=
  l1.a * l2.b - l2.a * l1.b = 0 → (∃ (k : ℝ), k ≠ 0 ∧ l1.a = k * l2.a ∧ l1.b = k * l2.b)

/-- Count of true propositions among p, its converse, negation, and contrapositive --/
def f_p : ℕ := 2

/-- The main theorem --/
theorem count_true_propositions :
  (∀ l1 l2 : Line, p l1 l2) ∧
  (∃ l1 l2 : Line, ¬(p_converse l1 l2)) ∧
  f_p = 2 := by sorry

end NUMINAMATH_CALUDE_count_true_propositions_l3599_359929


namespace NUMINAMATH_CALUDE_sqrt_product_equals_three_halves_l3599_359903

theorem sqrt_product_equals_three_halves : 
  Real.sqrt 5 * Real.sqrt (9 / 20) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_three_halves_l3599_359903


namespace NUMINAMATH_CALUDE_max_volume_at_eight_l3599_359959

/-- The side length of the original square plate in cm -/
def plate_side : ℝ := 48

/-- The volume of the container as a function of the cut square's side length -/
def volume (x : ℝ) : ℝ := (plate_side - 2*x)^2 * x

/-- The derivative of the volume function -/
def volume_derivative (x : ℝ) : ℝ := (plate_side - 2*x) * (plate_side - 6*x)

theorem max_volume_at_eight :
  ∃ (max_x : ℝ), max_x = 8 ∧
  ∀ (x : ℝ), 0 < x ∧ x < plate_side / 2 → volume x ≤ volume max_x :=
sorry

end NUMINAMATH_CALUDE_max_volume_at_eight_l3599_359959


namespace NUMINAMATH_CALUDE_num_valid_committees_l3599_359909

/-- Represents a community with speakers of different languages -/
structure Community where
  total : ℕ
  english : ℕ
  german : ℕ
  french : ℕ

/-- Defines a valid committee in the community -/
def ValidCommittee (c : Community) : Prop :=
  c.total = 20 ∧ c.english = 10 ∧ c.german = 10 ∧ c.french = 10

/-- Calculates the number of valid committees -/
noncomputable def NumValidCommittees (c : Community) : ℕ :=
  Nat.choose c.total 3 - Nat.choose (c.total - c.english) 3

/-- Theorem stating the number of valid committees -/
theorem num_valid_committees (c : Community) (h : ValidCommittee c) : 
  NumValidCommittees c = 1020 := by
  sorry


end NUMINAMATH_CALUDE_num_valid_committees_l3599_359909


namespace NUMINAMATH_CALUDE_willy_distance_theorem_l3599_359947

/-- Represents the distances from Willy to the corners of the square lot -/
structure Distances where
  d₁ : ℝ
  d₂ : ℝ
  d₃ : ℝ
  d₄ : ℝ

/-- The conditions given in the problem -/
def satisfies_conditions (d : Distances) : Prop :=
  d.d₁ < d.d₂ ∧ d.d₂ < d.d₄ ∧ d.d₄ < d.d₃ ∧
  d.d₂ = (d.d₁ + d.d₃) / 2 ∧
  d.d₄ ^ 2 = d.d₂ * d.d₃

/-- The theorem to be proved -/
theorem willy_distance_theorem (d : Distances) (h : satisfies_conditions d) :
  d.d₁ ^ 2 = (4 * d.d₁ * d.d₃ - d.d₃ ^ 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_willy_distance_theorem_l3599_359947


namespace NUMINAMATH_CALUDE_staircase_markups_l3599_359985

/-- Represents the number of different markups for a staircase with n cells -/
def L (n : ℕ) : ℕ := n + 1

/-- Theorem stating that the number of different markups for a staircase with n cells is n + 1 -/
theorem staircase_markups (n : ℕ) : L n = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_staircase_markups_l3599_359985


namespace NUMINAMATH_CALUDE_forty_nine_squared_equals_seven_to_zero_l3599_359984

theorem forty_nine_squared_equals_seven_to_zero : 49 * 49 = 7^0 := by
  sorry

end NUMINAMATH_CALUDE_forty_nine_squared_equals_seven_to_zero_l3599_359984


namespace NUMINAMATH_CALUDE_circle_area_theorem_l3599_359974

theorem circle_area_theorem (r : ℝ) (h : r > 0) :
  (2 * (1 / (2 * Real.pi * r)) = r / 2) → (Real.pi * r^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l3599_359974


namespace NUMINAMATH_CALUDE_trapezoid_to_square_l3599_359955

/-- Represents a trapezoid with bases a and b, and height h -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  area_eq : (a + b) * h / 2 = 5

/-- Represents a square with side length s -/
structure Square where
  s : ℝ
  area_eq : s^2 = 5

/-- Theorem stating that a trapezoid with area 5 can be cut into three parts to form a square -/
theorem trapezoid_to_square (t : Trapezoid) : ∃ (sq : Square), True := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_to_square_l3599_359955


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3599_359961

theorem simplify_fraction_product : (360 / 32) * (10 / 240) * (16 / 6) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3599_359961


namespace NUMINAMATH_CALUDE_vector_addition_l3599_359952

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![1, 5]

-- Define the operation 2a + b
def result : Fin 2 → ℝ := fun i => 2 * a i + b i

-- Theorem statement
theorem vector_addition : result = ![5, 7] := by sorry

end NUMINAMATH_CALUDE_vector_addition_l3599_359952


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l3599_359995

/-- Represents the number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 indistinguishable balls into 4 distinguishable boxes is 68 -/
theorem distribute_five_balls_four_boxes : distribute_balls 5 4 = 68 := by sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l3599_359995


namespace NUMINAMATH_CALUDE_sequence_odd_terms_l3599_359925

theorem sequence_odd_terms (a : ℕ → ℤ) 
  (h1 : a 1 = 2)
  (h2 : a 2 = 7)
  (h3 : ∀ n ≥ 2, -1/2 < (a (n+1) : ℚ) - (a n)^2 / (a (n-1))^2 ∧ 
                 (a (n+1) : ℚ) - (a n)^2 / (a (n-1))^2 ≤ 1/2) :
  ∀ n > 1, Odd (a n) := by
sorry

end NUMINAMATH_CALUDE_sequence_odd_terms_l3599_359925


namespace NUMINAMATH_CALUDE_ricky_sold_nine_l3599_359963

/-- Represents the number of glasses of lemonade sold by each person -/
structure LemonadeSales where
  katya : ℕ
  ricky : ℕ
  tina : ℕ

/-- The conditions of the lemonade sales problem -/
def lemonade_problem (sales : LemonadeSales) : Prop :=
  sales.katya = 8 ∧
  sales.tina = 2 * (sales.katya + sales.ricky) ∧
  sales.tina = sales.katya + 26

/-- Theorem stating that under the given conditions, Ricky sold 9 glasses of lemonade -/
theorem ricky_sold_nine (sales : LemonadeSales) 
  (h : lemonade_problem sales) : sales.ricky = 9 := by
  sorry

end NUMINAMATH_CALUDE_ricky_sold_nine_l3599_359963


namespace NUMINAMATH_CALUDE_ball_bearing_bulk_discount_percentage_l3599_359950

/-- Calculates the bulk discount percentage for John's ball bearing purchase --/
theorem ball_bearing_bulk_discount_percentage : 
  let num_machines : ℕ := 10
  let bearings_per_machine : ℕ := 30
  let normal_price : ℚ := 1
  let sale_price : ℚ := 3/4
  let total_savings : ℚ := 120
  let total_bearings := num_machines * bearings_per_machine
  let original_cost := total_bearings * normal_price
  let sale_cost := total_bearings * sale_price
  let discounted_cost := original_cost - total_savings
  let bulk_discount := sale_cost - discounted_cost
  let discount_percentage := (bulk_discount / sale_cost) * 100
  discount_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_ball_bearing_bulk_discount_percentage_l3599_359950


namespace NUMINAMATH_CALUDE_fraction_addition_l3599_359980

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3599_359980


namespace NUMINAMATH_CALUDE_fraction_equality_l3599_359948

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) : a / b = (a * b) / (b ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3599_359948


namespace NUMINAMATH_CALUDE_find_unknown_number_l3599_359908

theorem find_unknown_number (n : ℕ) : 
  (∀ m : ℕ, m < 3555 → ¬(711 ∣ m ∧ n ∣ m)) → 
  (711 ∣ 3555 ∧ n ∣ 3555) → 
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_find_unknown_number_l3599_359908


namespace NUMINAMATH_CALUDE_other_number_proof_l3599_359953

theorem other_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 16) (h2 : Nat.lcm a b = 396) (h3 : a = 176) : b = 36 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l3599_359953


namespace NUMINAMATH_CALUDE_vector_simplification_l3599_359919

variable {V : Type*} [AddCommGroup V]

theorem vector_simplification 
  (O P Q S : V) : 
  (O - P) - (Q - P) + (P - S) + (S - P) = O - Q := by sorry

end NUMINAMATH_CALUDE_vector_simplification_l3599_359919


namespace NUMINAMATH_CALUDE_ship_speed_l3599_359992

/-- The speed of a ship in still water, given specific conditions of its journey on a river -/
theorem ship_speed (total_time : ℝ) (total_distance : ℝ) (current_speed : ℝ) : 
  total_time = 6 ∧ 
  total_distance = 36 ∧ 
  current_speed = 3 → 
  ∃ (ship_speed : ℝ), 
    ship_speed = 3 + 3 * Real.sqrt 2 ∧
    total_time * (ship_speed^2 - current_speed^2) = 2 * total_distance * ship_speed :=
by sorry

end NUMINAMATH_CALUDE_ship_speed_l3599_359992


namespace NUMINAMATH_CALUDE_cosine_pi_third_derivative_l3599_359930

theorem cosine_pi_third_derivative :
  let y : ℝ → ℝ := λ _ => Real.cos (π / 3)
  ∀ x : ℝ, deriv y x = 0 := by
sorry

end NUMINAMATH_CALUDE_cosine_pi_third_derivative_l3599_359930


namespace NUMINAMATH_CALUDE_zoo_animal_ratio_l3599_359977

/-- Proves that the ratio of cheetahs to snakes is 7:10 given the zoo animal counts --/
theorem zoo_animal_ratio : 
  ∀ (snakes arctic_foxes leopards bee_eaters alligators cheetahs total : ℕ),
  snakes = 100 →
  arctic_foxes = 80 →
  leopards = 20 →
  bee_eaters = 10 * leopards →
  alligators = 2 * (arctic_foxes + leopards) →
  total = 670 →
  total = snakes + arctic_foxes + leopards + bee_eaters + alligators + cheetahs →
  (cheetahs : ℚ) / snakes = 7 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_ratio_l3599_359977


namespace NUMINAMATH_CALUDE_fraction_problem_l3599_359928

theorem fraction_problem (a b c : ℝ) 
  (h1 : a * b / (a + b) = 3)
  (h2 : b * c / (b + c) = 6)
  (h3 : a * c / (a + c) = 9) :
  c / (a * b) = -35 / 36 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3599_359928


namespace NUMINAMATH_CALUDE_distance_to_line_l3599_359993

/-- Represents a line in 2D space using parametric equations --/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Calculates the distance from a point to a line given in parametric form --/
def distanceToParametricLine (px py : ℝ) (line : ParametricLine) : ℝ :=
  sorry

/-- The problem statement --/
theorem distance_to_line : 
  let l : ParametricLine := { x := λ t => 1 + t, y := λ t => -1 + t }
  let p : (ℝ × ℝ) := (4, 0)
  distanceToParametricLine p.1 p.2 l = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_line_l3599_359993


namespace NUMINAMATH_CALUDE_ellipse_equation_equivalence_l3599_359983

theorem ellipse_equation_equivalence (x y : ℝ) :
  (Real.sqrt (x^2 + (y + 3)^2) + Real.sqrt (x^2 + (y - 3)^2) = 10) ↔
  (x^2 / 25 + y^2 / 16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_equivalence_l3599_359983


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l3599_359971

/-- The number of trailing zeros in the product of all multiples of 5 from 5 to 2015 -/
def trailingZeros : ℕ :=
  let n := 2015 / 5  -- number of terms in the product
  let factorsOf2 := (n / 2) + (n / 4) + (n / 8) + (n / 16) + (n / 32) + (n / 64) + (n / 128) + (n / 256)
  factorsOf2

theorem product_trailing_zeros : trailingZeros = 398 := by
  sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l3599_359971


namespace NUMINAMATH_CALUDE_probability_of_convex_quadrilateral_l3599_359939

-- Define the number of points on the circle
def num_points : ℕ := 8

-- Define the number of chords to be selected
def num_selected_chords : ℕ := 4

-- Define the total number of possible chords
def total_chords : ℕ := num_points.choose 2

-- Define the number of ways to select the chords
def ways_to_select_chords : ℕ := total_chords.choose num_selected_chords

-- Define the number of ways to form a convex quadrilateral
def convex_quadrilaterals : ℕ := num_points.choose 4

-- State the theorem
theorem probability_of_convex_quadrilateral :
  (convex_quadrilaterals : ℚ) / ways_to_select_chords = 2 / 585 :=
sorry

end NUMINAMATH_CALUDE_probability_of_convex_quadrilateral_l3599_359939


namespace NUMINAMATH_CALUDE_souvenir_spending_difference_l3599_359938

def total_spent : ℚ := 548
def keychain_bracelet_spent : ℚ := 347

theorem souvenir_spending_difference :
  keychain_bracelet_spent - (total_spent - keychain_bracelet_spent) = 146 := by
  sorry

end NUMINAMATH_CALUDE_souvenir_spending_difference_l3599_359938


namespace NUMINAMATH_CALUDE_perp_foot_curve_equation_l3599_359907

/-- The curve traced by the feet of perpendiculars from the origin to a moving unit segment -/
def PerpFootCurve (x y : ℝ) : Prop :=
  (x^2 + y^2)^3 = x^2 * y^2

/-- A point on the x-axis -/
def PointOnXAxis (p : ℝ × ℝ) : Prop :=
  p.2 = 0

/-- A point on the y-axis -/
def PointOnYAxis (p : ℝ × ℝ) : Prop :=
  p.1 = 0

/-- The distance between two points is 1 -/
def UnitDistance (p q : ℝ × ℝ) : Prop :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2 = 1

/-- The perpendicular foot from the origin to a line segment -/
def PerpFoot (p : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  (p.1 * (b.1 - a.1) + p.2 * (b.2 - a.2) = 0) ∧
  (∃ t : ℝ, p = (a.1 + t * (b.1 - a.1), a.2 + t * (b.2 - a.2)) ∧ 0 ≤ t ∧ t ≤ 1)

theorem perp_foot_curve_equation (x y : ℝ) :
  (∃ a b : ℝ × ℝ, PointOnXAxis a ∧ PointOnYAxis b ∧ UnitDistance a b ∧
    PerpFoot (x, y) a b) →
  PerpFootCurve x y :=
sorry

end NUMINAMATH_CALUDE_perp_foot_curve_equation_l3599_359907


namespace NUMINAMATH_CALUDE_amber_pieces_count_l3599_359937

theorem amber_pieces_count (green clear : ℕ) (h1 : green = 35) (h2 : clear = 85) 
  (h3 : green = (green + clear + amber) / 4) : amber = 20 := by
  sorry

end NUMINAMATH_CALUDE_amber_pieces_count_l3599_359937


namespace NUMINAMATH_CALUDE_card_sequence_periodicity_l3599_359913

def planet_value : ℕ := 2010
def hegemon_value (planets : ℕ) : ℕ := 4 * planets

def card_choice (n : ℕ) : ℕ := 
  if n ≤ 503 then 0 else (n - 503) % 2

theorem card_sequence_periodicity :
  ∃ (k : ℕ), k > 0 ∧ ∀ (n : ℕ), n ≥ 503 → card_choice (n + k) = card_choice n :=
sorry

end NUMINAMATH_CALUDE_card_sequence_periodicity_l3599_359913


namespace NUMINAMATH_CALUDE_fruit_display_total_l3599_359900

/-- Proves that the total number of fruits on a display is 35, given the specified conditions. -/
theorem fruit_display_total (bananas oranges apples : ℕ) : 
  bananas = 5 → 
  oranges = 2 * bananas → 
  apples = 2 * oranges → 
  bananas + oranges + apples = 35 := by
sorry

end NUMINAMATH_CALUDE_fruit_display_total_l3599_359900


namespace NUMINAMATH_CALUDE_number_of_sets_l3599_359967

/-- Represents a four-digit number in the game "Set" -/
def SetNumber := Fin 4 → Fin 3

/-- Checks if three numbers form a valid set in the game "Set" -/
def is_valid_set (a b c : SetNumber) : Prop :=
  ∀ i : Fin 4, (a i = b i ∧ b i = c i) ∨ (a i ≠ b i ∧ b i ≠ c i ∧ a i ≠ c i)

/-- The set of all possible four-digit numbers in the game "Set" -/
def all_set_numbers : Finset SetNumber :=
  sorry

/-- The set of all valid sets in the game "Set" -/
def all_valid_sets : Finset (Finset SetNumber) :=
  sorry

/-- The main theorem stating the number of valid sets in the game "Set" -/
theorem number_of_sets : Finset.card all_valid_sets = 1080 :=
  sorry

end NUMINAMATH_CALUDE_number_of_sets_l3599_359967


namespace NUMINAMATH_CALUDE_no_prime_solution_l3599_359989

def base_p_to_decimal (digits : List Nat) (p : Nat) : Nat :=
  digits.foldl (fun acc d => acc * p + d) 0

theorem no_prime_solution :
  ¬ ∃ (p : Nat), Prime p ∧ 
    (base_p_to_decimal [2,0,3,4] p + 
     base_p_to_decimal [4,0,5] p + 
     base_p_to_decimal [1,2] p + 
     base_p_to_decimal [2,1,2] p + 
     base_p_to_decimal [7] p = 
     base_p_to_decimal [1,3,1,5] p + 
     base_p_to_decimal [5,4,1] p + 
     base_p_to_decimal [2,2,2] p) :=
by
  sorry


end NUMINAMATH_CALUDE_no_prime_solution_l3599_359989


namespace NUMINAMATH_CALUDE_train_distance_difference_l3599_359936

/-- Proves that the difference in distance traveled by two trains is 70 km -/
theorem train_distance_difference (v1 v2 total_distance : ℝ) 
  (h1 : v1 = 20) 
  (h2 : v2 = 25) 
  (h3 : total_distance = 630) : ∃ (t : ℝ), v2 * t - v1 * t = 70 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_difference_l3599_359936


namespace NUMINAMATH_CALUDE_cats_not_eating_apples_or_chicken_l3599_359921

theorem cats_not_eating_apples_or_chicken
  (total_cats : ℕ)
  (cats_liking_apples : ℕ)
  (cats_liking_chicken : ℕ)
  (cats_liking_both : ℕ)
  (h1 : total_cats = 80)
  (h2 : cats_liking_apples = 15)
  (h3 : cats_liking_chicken = 60)
  (h4 : cats_liking_both = 10) :
  total_cats - (cats_liking_apples + cats_liking_chicken - cats_liking_both) = 15 := by
  sorry

end NUMINAMATH_CALUDE_cats_not_eating_apples_or_chicken_l3599_359921


namespace NUMINAMATH_CALUDE_roots_sum_cubes_fourth_powers_l3599_359931

theorem roots_sum_cubes_fourth_powers (α β : ℝ) : 
  α^2 - 3*α - 2 = 0 → β^2 - 3*β - 2 = 0 → 3*α^3 + 8*β^4 = 1229 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_cubes_fourth_powers_l3599_359931


namespace NUMINAMATH_CALUDE_johnny_practice_days_l3599_359972

/-- The number of days Johnny has been practicing up to now -/
def current_practice_days : ℕ := 40

/-- The number of days in the future when Johnny will have tripled his practice -/
def future_days : ℕ := 80

/-- Represents that Johnny practices the same amount each day -/
axiom consistent_practice : True

/-- In 80 days, Johnny will have 3 times as much practice as he does currently -/
axiom future_practice : current_practice_days + future_days = 3 * current_practice_days

/-- The number of days ago when Johnny had half as much practice -/
def half_practice_days : ℕ := current_practice_days / 2

theorem johnny_practice_days : half_practice_days = 20 := by sorry

end NUMINAMATH_CALUDE_johnny_practice_days_l3599_359972


namespace NUMINAMATH_CALUDE_rectangle_area_l3599_359901

/-- Given a rectangle where the length is thrice the breadth and the perimeter is 40,
    prove that its area is 75. -/
theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let p := 2 * (l + b)
  p = 40 →
  l * b = 75 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3599_359901


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_x_is_smallest_y_is_smallest_x_makes_square_y_makes_cube_l3599_359986

/-- The smallest positive integer x for which 420x is a square -/
def x : ℕ := 735

/-- The smallest positive integer y for which 420y is a cube -/
def y : ℕ := 22050

theorem sum_of_x_and_y : x + y = 22785 := by sorry

theorem x_is_smallest :
  ∀ n : ℕ, n > 0 → (∃ m : ℕ, 420 * n = m ^ 2) → n ≥ x := by sorry

theorem y_is_smallest :
  ∀ n : ℕ, n > 0 → (∃ m : ℕ, 420 * n = m ^ 3) → n ≥ y := by sorry

theorem x_makes_square : ∃ m : ℕ, 420 * x = m ^ 2 := by sorry

theorem y_makes_cube : ∃ m : ℕ, 420 * y = m ^ 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_x_is_smallest_y_is_smallest_x_makes_square_y_makes_cube_l3599_359986


namespace NUMINAMATH_CALUDE_stock_percentage_return_l3599_359932

def stock_yield : ℝ := 0.08
def market_value : ℝ := 137.5

theorem stock_percentage_return :
  (stock_yield * market_value) / market_value * 100 = stock_yield * 100 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_return_l3599_359932


namespace NUMINAMATH_CALUDE_chess_game_most_likely_outcome_l3599_359949

theorem chess_game_most_likely_outcome
  (prob_A_win : ℝ)
  (prob_A_not_lose : ℝ)
  (h1 : prob_A_win = 0.3)
  (h2 : prob_A_not_lose = 0.7)
  (h3 : 0 ≤ prob_A_win ∧ prob_A_win ≤ 1)
  (h4 : 0 ≤ prob_A_not_lose ∧ prob_A_not_lose ≤ 1) :
  let prob_draw := prob_A_not_lose - prob_A_win
  let prob_B_win := 1 - prob_A_not_lose
  prob_draw > prob_A_win ∧ prob_draw > prob_B_win :=
by sorry

end NUMINAMATH_CALUDE_chess_game_most_likely_outcome_l3599_359949


namespace NUMINAMATH_CALUDE_final_dog_count_l3599_359997

/-- Calculates the number of dogs remaining in the rescue center at the end of the month -/
def dogsRemaining (initial : ℕ) (arrivals : List ℕ) (adoptions : List ℕ) (returned : ℕ) : ℕ :=
  let weeklyChanges := List.zipWith (λ a b => a - b) arrivals adoptions
  initial + weeklyChanges.sum - returned

theorem final_dog_count :
  let initial : ℕ := 200
  let arrivals : List ℕ := [30, 40, 30]
  let adoptions : List ℕ := [40, 50, 30, 70]
  let returned : ℕ := 20
  dogsRemaining initial arrivals adoptions returned = 90 := by
  sorry

end NUMINAMATH_CALUDE_final_dog_count_l3599_359997


namespace NUMINAMATH_CALUDE_percentage_calculation_l3599_359935

theorem percentage_calculation (n : ℝ) (h : n = 4800) : n * 0.5 * 0.3 * 0.15 = 108 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3599_359935


namespace NUMINAMATH_CALUDE_composite_function_evaluation_l3599_359918

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 2
def g (x : ℝ) : ℝ := 5 * x + 2

-- State the theorem
theorem composite_function_evaluation : g (f (g 1)) = 82 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_evaluation_l3599_359918


namespace NUMINAMATH_CALUDE_parametric_to_cartesian_l3599_359943

theorem parametric_to_cartesian :
  ∀ (x y : ℝ), (∃ t : ℝ, 0 ≤ t ∧ t ≤ 2 * Real.pi ∧ x = 2 * Real.cos t ∧ y = 3 * Real.sin t) →
  x^2 / 4 + y^2 / 9 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_parametric_to_cartesian_l3599_359943


namespace NUMINAMATH_CALUDE_evaluate_expression_l3599_359942

theorem evaluate_expression : (36 - 6 * 3) / (6 / 3 * 2) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3599_359942


namespace NUMINAMATH_CALUDE_daxton_water_usage_l3599_359968

theorem daxton_water_usage 
  (tank_capacity : ℝ)
  (initial_fill_ratio : ℝ)
  (refill_ratio : ℝ)
  (final_volume : ℝ)
  (h1 : tank_capacity = 8000)
  (h2 : initial_fill_ratio = 3/4)
  (h3 : refill_ratio = 0.3)
  (h4 : final_volume = 4680) :
  let initial_volume := tank_capacity * initial_fill_ratio
  let usage_percentage := 
    (initial_volume - (final_volume - refill_ratio * (initial_volume - usage_volume))) / initial_volume
  let usage_volume := usage_percentage * initial_volume
  usage_percentage = 0.4 := by
sorry

end NUMINAMATH_CALUDE_daxton_water_usage_l3599_359968


namespace NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l3599_359905

theorem arithmetic_progression_common_difference 
  (n : ℕ) 
  (d : ℚ) 
  (sum_original : ℚ) 
  (sum_decrease_min : ℚ) 
  (sum_decrease_max : ℚ) :
  (n > 0) →
  (sum_original = 63) →
  (sum_original = (n / 2) * (3 * d + (n - 1) * d)) →
  (sum_decrease_min = 7) →
  (sum_decrease_max = 8) →
  (sum_original - (n / 2) * (2 * d + (n - 1) * d) ≥ sum_decrease_min) →
  (sum_original - (n / 2) * (2 * d + (n - 1) * d) ≤ sum_decrease_max) →
  (d = 21/8 ∨ d = 2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l3599_359905


namespace NUMINAMATH_CALUDE_part_one_part_two_l3599_359944

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < 2*m + 1}
def B : Set ℝ := {x | (x - 7) / (x - 2) < 0}

-- Part 1
theorem part_one : A 2 ∩ (Set.univ \ B) = Set.Ioc 1 2 := by sorry

-- Part 2
theorem part_two : ∀ m : ℝ, A m ∪ B = B ↔ m ∈ Set.Iic (-2) ∪ {3} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3599_359944


namespace NUMINAMATH_CALUDE_hcf_problem_l3599_359902

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 2560) (h2 : Nat.lcm a b = 160) :
  Nat.gcd a b = 16 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l3599_359902


namespace NUMINAMATH_CALUDE_equation_solution_l3599_359996

theorem equation_solution (x : ℝ) : (6 : ℝ) / (x + 1) = (3 : ℝ) / 2 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3599_359996


namespace NUMINAMATH_CALUDE_parabola_point_value_l3599_359973

/-- Given a parabola y = x^2 + (a+1)x + a that passes through the point (-1, m),
    prove that m = 0 -/
theorem parabola_point_value (a m : ℝ) : 
  ((-1)^2 + (a+1)*(-1) + a = m) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_value_l3599_359973


namespace NUMINAMATH_CALUDE_friend_walking_speed_difference_l3599_359979

theorem friend_walking_speed_difference 
  (total_distance : ℝ) 
  (p_distance : ℝ) 
  (hp : total_distance = 22) 
  (hpd : p_distance = 12) : 
  let q_distance := total_distance - p_distance
  let rate_ratio := p_distance / q_distance
  (rate_ratio - 1) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_friend_walking_speed_difference_l3599_359979


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_square_l3599_359927

/-- Given a square with side length a, there exists an isosceles triangle with the specified properties --/
theorem isosceles_triangle_from_square (a : ℝ) (h : a > 0) :
  ∃ (x y z : ℝ),
    -- The base of the triangle
    x = a * Real.sqrt 3 ∧
    -- The height of the triangle
    y = (2 * x) / 3 ∧
    -- The equal sides of the triangle
    z = (5 * a * Real.sqrt 3) / 6 ∧
    -- Area equality
    (1 / 2) * x * y = a^2 ∧
    -- Sum of base and height equals sum of equal sides
    x + y = 2 * z ∧
    -- Pythagorean theorem
    y^2 + (x / 2)^2 = z^2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_square_l3599_359927


namespace NUMINAMATH_CALUDE_intersection_sum_l3599_359933

/-- Two lines intersect at a point (3,1). -/
def intersection_point : ℝ × ℝ := (3, 1)

/-- The first line equation: x = (1/3)y + a -/
def line1 (a : ℝ) (x y : ℝ) : Prop := x = (1/3) * y + a

/-- The second line equation: y = (1/3)x + b -/
def line2 (b : ℝ) (x y : ℝ) : Prop := y = (1/3) * x + b

/-- The theorem states that if two lines intersect at (3,1), then a + b = 8/3 -/
theorem intersection_sum (a b : ℝ) : 
  line1 a (intersection_point.1) (intersection_point.2) ∧ 
  line2 b (intersection_point.1) (intersection_point.2) → 
  a + b = 8/3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_l3599_359933


namespace NUMINAMATH_CALUDE_relationship_abc_l3599_359912

theorem relationship_abc (a b c : ℝ) 
  (ha : a = (2/3)^(-(1/3 : ℝ))) 
  (hb : b = (5/3)^(-(2/3 : ℝ))) 
  (hc : c = (3/2)^(2/3 : ℝ)) : 
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3599_359912


namespace NUMINAMATH_CALUDE_triangle_properties_l3599_359904

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  2 * t.c = t.a + Real.cos t.A * t.b / Real.cos t.B ∧
  t.b = 4 ∧
  t.a + t.c = 3 * Real.sqrt 2

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.B = π / 3 ∧ 
  (1 / 2 * t.a * t.c * Real.sin t.B : ℝ) = Real.sqrt 3 / 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3599_359904


namespace NUMINAMATH_CALUDE_inequalities_for_positive_reals_l3599_359924

theorem inequalities_for_positive_reals (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  a * b ≤ 1 ∧ a^2 + b^2 ≥ 2 ∧ Real.sqrt a + Real.sqrt b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequalities_for_positive_reals_l3599_359924


namespace NUMINAMATH_CALUDE_watch_loss_percentage_l3599_359964

/-- Proves that the loss percentage is 5% when a watch is sold for Rs. 1140,
    given that selling it for Rs. 1260 would result in a 5% profit. -/
theorem watch_loss_percentage
  (loss_price : ℝ)
  (profit_price : ℝ)
  (profit_percentage : ℝ)
  (h1 : loss_price = 1140)
  (h2 : profit_price = 1260)
  (h3 : profit_percentage = 0.05)
  : (profit_price / (1 + profit_percentage) - loss_price) / (profit_price / (1 + profit_percentage)) = 0.05 := by
  sorry

#check watch_loss_percentage

end NUMINAMATH_CALUDE_watch_loss_percentage_l3599_359964


namespace NUMINAMATH_CALUDE_sum_in_range_l3599_359970

theorem sum_in_range : ∃ (x : ℚ), 
  (x = 3 + 3/8 + 4 + 1/3 + 6 + 1/21 - 2) ∧ 
  (11.5 < x) ∧ 
  (x < 12) := by
  sorry

end NUMINAMATH_CALUDE_sum_in_range_l3599_359970


namespace NUMINAMATH_CALUDE_tangent_perpendicular_line_l3599_359990

-- Define the curve
def C (x : ℝ) : ℝ := x^2 + x

-- Define the derivative of the curve
def C_derivative (x : ℝ) : ℝ := 2*x + 1

-- Define the slope of the tangent line at x = 1
def tangent_slope : ℝ := C_derivative 1

-- Define the condition for perpendicularity
def perpendicular_condition (a : ℝ) : Prop :=
  tangent_slope * a = -1

-- The theorem to prove
theorem tangent_perpendicular_line : 
  ∃ (a : ℝ), perpendicular_condition a ∧ a = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_line_l3599_359990


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l3599_359982

theorem quadratic_equation_real_roots (a b : ℝ) : 
  (∃ x : ℝ, x^2 + 2*(1+a)*x + (3*a^2 + 4*a*b + 4*b^2 + 2) = 0) → 
  (a = 1 ∧ b = -1/2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l3599_359982


namespace NUMINAMATH_CALUDE_square_difference_401_399_l3599_359923

theorem square_difference_401_399 : 401^2 - 399^2 = 1600 := by sorry

end NUMINAMATH_CALUDE_square_difference_401_399_l3599_359923


namespace NUMINAMATH_CALUDE_books_returned_percentage_l3599_359958

/-- Calculates the percentage of loaned books that were returned -/
def percentage_books_returned (initial_books : ℕ) (final_books : ℕ) (loaned_books : ℕ) : ℚ :=
  let returned_books := final_books - (initial_books - loaned_books)
  (returned_books : ℚ) / (loaned_books : ℚ) * 100

/-- Proves that the percentage of loaned books returned is 70% -/
theorem books_returned_percentage :
  percentage_books_returned 75 60 50 = 70 := by
  sorry

#eval percentage_books_returned 75 60 50

end NUMINAMATH_CALUDE_books_returned_percentage_l3599_359958


namespace NUMINAMATH_CALUDE_smaller_number_l3599_359954

theorem smaller_number (x y : ℝ) (sum_eq : x + y = 18) (diff_eq : x - y = 8) : 
  min x y = 5 := by sorry

end NUMINAMATH_CALUDE_smaller_number_l3599_359954


namespace NUMINAMATH_CALUDE_gcd_b_consecutive_is_one_l3599_359922

def b (n : ℕ) : ℤ := (7^n - 1) / 6

theorem gcd_b_consecutive_is_one (n : ℕ) : 
  Nat.gcd (Int.natAbs (b n)) (Int.natAbs (b (n + 1))) = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_b_consecutive_is_one_l3599_359922


namespace NUMINAMATH_CALUDE_tenth_term_value_l3599_359975

/-- An arithmetic sequence with 30 terms, first term 3, and last term 88 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  let d := (88 - 3) / 29
  3 + (n - 1) * d

/-- The 10th term of the arithmetic sequence is 852/29 -/
theorem tenth_term_value : arithmetic_sequence 10 = 852 / 29 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_value_l3599_359975


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3599_359962

/-- Given a line 2ax - by + 2 = 0 (where a > 0, b > 0) passing through the point (-1, 2),
    the minimum value of 1/a + 1/b is 4. -/
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_line : 2 * a * (-1) - b * 2 + 2 = 0) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x * (-1) - y * 2 + 2 = 0 → 1 / x + 1 / y ≥ 1 / a + 1 / b) ∧
  1 / a + 1 / b = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3599_359962


namespace NUMINAMATH_CALUDE_clown_balloon_count_l3599_359941

/-- The number of balloons a clown has after a series of events -/
def final_balloon_count (initial : ℕ) (additional : ℕ) (given_away : ℕ) (popped : ℕ) : ℕ :=
  initial + additional - given_away - popped

/-- Theorem stating the final number of balloons the clown has -/
theorem clown_balloon_count :
  final_balloon_count 47 13 20 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloon_count_l3599_359941


namespace NUMINAMATH_CALUDE_inequality_solution_equation_solution_range_l3599_359915

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 4| + |x + 1|

-- Theorem for the first part of the problem
theorem inequality_solution :
  {x : ℝ | f x ≤ 9} = {x : ℝ | -2 ≤ x ∧ x ≤ 4} :=
sorry

-- Theorem for the second part of the problem
theorem equation_solution_range :
  {a : ℝ | ∃ x ∈ Set.Icc 0 2, f x = -x^2 + a} = Set.Icc (19/4) 7 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_equation_solution_range_l3599_359915


namespace NUMINAMATH_CALUDE_polynomial_equality_sum_l3599_359987

theorem polynomial_equality_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_sum_l3599_359987


namespace NUMINAMATH_CALUDE_linear_functions_relation_l3599_359917

/-- Given two linear functions f and g, prove that A + B = 2A under certain conditions -/
theorem linear_functions_relation (A B : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = A * x + B + 1)
  (hg : ∀ x, g x = B * x + A - 1)
  (hAB : A ≠ -B)
  (h_comp : ∀ x, f (g x) - g (f x) = A - 2 * B) :
  A + B = 2 * A :=
sorry

end NUMINAMATH_CALUDE_linear_functions_relation_l3599_359917


namespace NUMINAMATH_CALUDE_parabola_symmetric_intersection_l3599_359957

/-- Represents a parabola of the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: For a parabola with symmetric axis x=1 and one x-axis intersection at (3,0),
    the other x-axis intersection is at (-1,0) --/
theorem parabola_symmetric_intersection
  (p : Parabola)
  (symmetric_axis : ℝ)
  (intersection : Point)
  (h1 : symmetric_axis = 1)
  (h2 : intersection = Point.mk 3 0)
  (h3 : p.a * intersection.x^2 + p.b * intersection.x + p.c = 0)
  : ∃ (other : Point), 
    other = Point.mk (-1) 0 ∧ 
    p.a * other.x^2 + p.b * other.x + p.c = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetric_intersection_l3599_359957


namespace NUMINAMATH_CALUDE_set_equality_from_union_intersection_equality_l3599_359994

theorem set_equality_from_union_intersection_equality {α : Type*} (A B : Set α) :
  A ∪ B = A ∩ B → A = B := by sorry

end NUMINAMATH_CALUDE_set_equality_from_union_intersection_equality_l3599_359994


namespace NUMINAMATH_CALUDE_ways_to_choose_all_suits_formula_l3599_359934

/-- The number of ways to choose 13 cards from a 52-card deck such that all four suits are represented -/
def waysToChooseAllSuits : ℕ :=
  Nat.choose 52 13 - 4 * Nat.choose 39 13 + 6 * Nat.choose 26 13 - 4 * Nat.choose 13 13

/-- Theorem stating that the number of ways to choose 13 cards from a 52-card deck
    such that all four suits are represented is equal to the given formula -/
theorem ways_to_choose_all_suits_formula :
  waysToChooseAllSuits =
    Nat.choose 52 13 - 4 * Nat.choose 39 13 + 6 * Nat.choose 26 13 - 4 * Nat.choose 13 13 := by
  sorry

#eval waysToChooseAllSuits

end NUMINAMATH_CALUDE_ways_to_choose_all_suits_formula_l3599_359934


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l3599_359945

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem min_value_geometric_sequence (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2015 + a 2017 = π →
  ∀ x : ℝ, a 2016 * (a 2014 + a 2018) ≥ π^2 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l3599_359945


namespace NUMINAMATH_CALUDE_roots_difference_squared_l3599_359969

theorem roots_difference_squared (α β : ℝ) : 
  α^2 - 3*α + 1 = 0 → β^2 - 3*β + 1 = 0 → (α - β)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_roots_difference_squared_l3599_359969
