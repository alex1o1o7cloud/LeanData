import Mathlib

namespace temperature_frequency_l3374_337442

def temperatures : List ℤ := [-2, 0, 3, -1, 1, 0, 4]

theorem temperature_frequency :
  (temperatures.filter (λ t => t > 0)).length = 3 := by
  sorry

end temperature_frequency_l3374_337442


namespace triangle_side_length_l3374_337474

/-- In a triangle ABC, given that angle C is four times angle A, 
    side a is 35, and side c is 64, prove that side b equals 140cos²A -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧          -- Sum of angles in a triangle
  C = 4 * A ∧              -- Angle C is four times angle A
  a = 35 ∧                 -- Side a is 35
  c = 64 →                 -- Side c is 64
  b = 140 * (Real.cos A)^2 := by
sorry

end triangle_side_length_l3374_337474


namespace bridget_apples_l3374_337431

theorem bridget_apples (x : ℕ) : 
  (x / 2 - (x / 2) / 3 = 5) → x = 15 := by
  sorry

end bridget_apples_l3374_337431


namespace interior_angle_of_17_sided_polygon_l3374_337469

theorem interior_angle_of_17_sided_polygon (S : ℝ) (x : ℝ) : 
  S = (17 - 2) * 180 ∧ S - x = 2570 → x = 130 := by
  sorry

end interior_angle_of_17_sided_polygon_l3374_337469


namespace lg_expression_equals_zero_l3374_337481

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_expression_equals_zero :
  lg 5 * lg 2 + lg (2^2) - lg 2 = 0 :=
by
  -- Properties of logarithms
  have h1 : ∀ m n : ℝ, lg (m^n) = n * lg m := sorry
  have h2 : ∀ a b : ℝ, lg (a * b) = lg a + lg b := sorry
  have h3 : lg 1 = 0 := sorry
  have h4 : lg 2 > 0 := sorry
  have h5 : lg 5 > 0 := sorry
  
  sorry -- Proof omitted

end lg_expression_equals_zero_l3374_337481


namespace solve_colored_copies_l3374_337444

def colored_copies_problem (colored_cost white_cost : ℚ) (total_copies : ℕ) (total_cost : ℚ) : Prop :=
  ∃ (colored_copies : ℕ),
    colored_copies ≤ total_copies ∧
    colored_cost * colored_copies + white_cost * (total_copies - colored_copies) = total_cost ∧
    colored_copies = 50

theorem solve_colored_copies :
  colored_copies_problem (10/100) (5/100) 400 (45/2) :=
sorry

end solve_colored_copies_l3374_337444


namespace pure_imaginary_condition_l3374_337494

theorem pure_imaginary_condition (b : ℝ) (i : ℂ) : 
  i * i = -1 →  -- i is the imaginary unit
  (∃ (k : ℝ), i * (b * i + 1) = k * i) →  -- i(bi+1) is a pure imaginary number
  b = 0 := by
sorry

end pure_imaginary_condition_l3374_337494


namespace sum_x_y_given_equations_l3374_337420

theorem sum_x_y_given_equations (x y : ℝ) 
  (eq1 : 2 * |x| + 3 * x + 3 * y = 30)
  (eq2 : 3 * x + 2 * |y| - 2 * y = 36) : 
  x + y = 8512 / 2513 := by
  sorry

end sum_x_y_given_equations_l3374_337420


namespace smallest_c_in_special_progression_l3374_337459

theorem smallest_c_in_special_progression (a b c : ℕ) : 
  a > b ∧ b > c ∧ c > 0 →  -- a, b, c are positive integers with a > b > c
  (b * b = a * c) →        -- a, b, c form a geometric progression
  (a + b = 2 * c) →        -- a, c, b form an arithmetic progression
  c ≥ 1 ∧                  -- c is at least 1
  (∀ k : ℕ, k > 0 ∧ k < c →
    ¬∃ x y : ℕ, x > y ∧ y > k ∧ 
    (y * y = x * k) ∧ 
    (x + y = 2 * k)) →     -- c is the smallest value satisfying the conditions
  c = 1                    -- The smallest possible value of c is 1
:= by sorry

end smallest_c_in_special_progression_l3374_337459


namespace annual_sales_profit_scientific_notation_l3374_337452

/-- Represents the annual sales profit in yuan -/
def annual_sales_profit : ℝ := 1.5e12

/-- Expresses the annual sales profit in scientific notation -/
def scientific_notation : ℝ := 1.5 * (10 ^ 12)

theorem annual_sales_profit_scientific_notation : 
  annual_sales_profit = scientific_notation := by sorry

end annual_sales_profit_scientific_notation_l3374_337452


namespace no_roots_for_equation_l3374_337421

theorem no_roots_for_equation : ∀ x : ℝ, ¬(Real.sqrt (7 - x) = x * Real.sqrt (7 - x) - 1) := by
  sorry

end no_roots_for_equation_l3374_337421


namespace g_twelve_equals_thirtysix_l3374_337457

/-- The area function of a rectangle with side lengths x and x+1 -/
def f (x : ℝ) : ℝ := x * (x + 1)

/-- The function g satisfying f(g(x)) = 9x^2 + 3x -/
noncomputable def g (x : ℝ) : ℝ := 
  (- 1 + Real.sqrt (36 * x^2 + 12 * x + 1)) / 2

theorem g_twelve_equals_thirtysix : g 12 = 36 := by sorry

end g_twelve_equals_thirtysix_l3374_337457


namespace intersection_distance_l3374_337411

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 36 = 1

-- Define the parabola (we don't know its exact equation, but we know it exists)
def parabola (x y : ℝ) : Prop := ∃ (a b c : ℝ), y = a * x^2 + b * x + c

-- Define the shared focus condition
def shared_focus (e p : ℝ → ℝ → Prop) : Prop :=
  ∃ (x₀ y₀ : ℝ), (∀ x y, e x y → (x - x₀)^2 + (y - y₀)^2 = 20) ∧
                 (∀ x y, p x y → (x - x₀)^2 + (y - y₀)^2 ≤ 20)

-- Define the directrix condition
def directrix_on_major_axis (p : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ x y, p x y → y = k * x^2

-- Theorem statement
theorem intersection_distance :
  ∀ (p : ℝ → ℝ → Prop),
  shared_focus ellipse p →
  directrix_on_major_axis p →
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ p x₁ y₁ ∧
    ellipse x₂ y₂ ∧ p x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (4 * Real.sqrt 5 / 3)^2 :=
sorry

end intersection_distance_l3374_337411


namespace irrational_equation_solution_l3374_337419

theorem irrational_equation_solution (a b : ℝ) : 
  Irrational a → (a * b + a - b = 1) → b = -1 := by
  sorry

end irrational_equation_solution_l3374_337419


namespace probability_in_standard_deck_l3374_337462

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Nat)
  (diamonds : Nat)
  (hearts : Nat)
  (spades : Nat)

/-- The probability of drawing a diamond, then a heart, then a spade from a standard 52 card deck -/
def probability_diamond_heart_spade (d : Deck) : ℚ :=
  (d.diamonds : ℚ) / d.cards *
  (d.hearts : ℚ) / (d.cards - 1) *
  (d.spades : ℚ) / (d.cards - 2)

/-- A standard 52 card deck -/
def standard_deck : Deck :=
  { cards := 52
  , diamonds := 13
  , hearts := 13
  , spades := 13 }

theorem probability_in_standard_deck :
  probability_diamond_heart_spade standard_deck = 2197 / 132600 :=
by sorry

end probability_in_standard_deck_l3374_337462


namespace solution_set_inequality_l3374_337434

theorem solution_set_inequality (x : ℝ) :
  (abs x - 2) * (x - 1) ≥ 0 ↔ -2 ≤ x ∧ x ≤ 1 ∨ x ≥ 2 :=
by sorry

end solution_set_inequality_l3374_337434


namespace distance_after_10_hours_l3374_337468

/-- The distance between two trains after a given time -/
def distance_between_trains (speed1 speed2 time : ℝ) : ℝ :=
  (speed2 - speed1) * time

/-- Theorem: The distance between two trains after 10 hours -/
theorem distance_after_10_hours :
  distance_between_trains 10 35 10 = 250 := by
  sorry

#eval distance_between_trains 10 35 10

end distance_after_10_hours_l3374_337468


namespace tanya_accompanied_twice_l3374_337497

/-- Represents a girl in the group --/
inductive Girl
| Anya
| Tanya
| Olya
| Katya

/-- The number of songs sung by each girl --/
def songsSung (g : Girl) : ℕ :=
  match g with
  | Girl.Anya => 8
  | Girl.Tanya => 6
  | Girl.Olya => 3
  | Girl.Katya => 7

/-- The total number of girls --/
def totalGirls : ℕ := 4

/-- The number of singers per song --/
def singersPerSong : ℕ := 3

/-- The total number of songs played --/
def totalSongs : ℕ := (songsSung Girl.Anya + songsSung Girl.Tanya + songsSung Girl.Olya + songsSung Girl.Katya) / singersPerSong

/-- The number of times Tanya accompanied --/
def tanyaAccompanied : ℕ := totalSongs - songsSung Girl.Tanya

theorem tanya_accompanied_twice :
  tanyaAccompanied = 2 :=
sorry

end tanya_accompanied_twice_l3374_337497


namespace room_carpet_cost_l3374_337496

/-- Calculates the total cost of carpeting a rectangular room -/
def carpet_cost (length width cost_per_sq_yard : ℚ) : ℚ :=
  let length_yards := length / 3
  let width_yards := width / 3
  let area_sq_yards := length_yards * width_yards
  area_sq_yards * cost_per_sq_yard

/-- Theorem stating the total cost of carpeting the given room -/
theorem room_carpet_cost :
  carpet_cost 15 12 10 = 200 := by
  sorry

end room_carpet_cost_l3374_337496


namespace polynomial_factorization_l3374_337430

theorem polynomial_factorization (x : ℝ) : 
  x^6 - 4*x^4 + 4*x^2 - 1 = (x - 1)^3 * (x + 1)^3 := by
  sorry

end polynomial_factorization_l3374_337430


namespace angle_subtraction_theorem_polynomial_simplification_theorem_l3374_337492

-- Define angle type
def Angle := ℕ × ℕ -- (degrees, minutes)

-- Define angle subtraction
def angle_sub (a b : Angle) : Angle := sorry

-- Define polynomial expression
def poly_expr (m : ℝ) := 5*m^2 - (m^2 - 6*m) - 2*(-m + 3*m^2)

theorem angle_subtraction_theorem :
  angle_sub (34, 26) (25, 33) = (8, 53) := by sorry

theorem polynomial_simplification_theorem (m : ℝ) :
  poly_expr m = -2*m^2 + 8*m := by sorry

end angle_subtraction_theorem_polynomial_simplification_theorem_l3374_337492


namespace perpendicular_and_parallel_properties_l3374_337400

-- Define the necessary structures
structure EuclideanPlane where
  -- Add necessary axioms for Euclidean plane

structure Line where
  -- Add necessary properties for a line

structure Point where
  -- Add necessary properties for a point

-- Define the relationships
def isOn (p : Point) (l : Line) : Prop := sorry

def isPerpendicular (l1 l2 : Line) : Prop := sorry

def isParallel (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem perpendicular_and_parallel_properties 
  (plane : EuclideanPlane) (l : Line) : 
  (∀ (p : Point), isOn p l → ∃ (perps : Set Line), 
    (∀ (l' : Line), l' ∈ perps ↔ isPerpendicular l' l ∧ isOn p l') ∧ 
    Set.Infinite perps) ∧
  (∀ (p : Point), ¬isOn p l → 
    ∃! (l' : Line), isPerpendicular l' l ∧ isOn p l') ∧
  (∀ (p : Point), ¬isOn p l → 
    ∃! (l' : Line), isParallel l' l ∧ isOn p l') := by
  sorry

end perpendicular_and_parallel_properties_l3374_337400


namespace number_puzzle_l3374_337466

theorem number_puzzle (N : ℝ) : 6 + (1/2) * (1/3) * (1/5) * N = (1/15) * N → N = 180 := by
  sorry

end number_puzzle_l3374_337466


namespace sticker_remainder_l3374_337490

theorem sticker_remainder (nina_stickers : Nat) (oliver_stickers : Nat) (patty_stickers : Nat) 
  (package_size : Nat) (h1 : nina_stickers = 53) (h2 : oliver_stickers = 68) 
  (h3 : patty_stickers = 29) (h4 : package_size = 18) : 
  (nina_stickers + oliver_stickers + patty_stickers) % package_size = 6 := by
  sorry

end sticker_remainder_l3374_337490


namespace vector_sum_magnitude_l3374_337448

theorem vector_sum_magnitude (x : ℝ) : 
  let a : ℝ × ℝ := (1, x)
  let b : ℝ × ℝ := (x + 2, -2)
  (a.1 * b.1 + a.2 * b.2 = 0) → 
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5 := by
sorry

end vector_sum_magnitude_l3374_337448


namespace perfect_square_trinomial_l3374_337422

theorem perfect_square_trinomial (x : ℝ) : x^2 - 6*x + 9 = (x - 3)^2 := by
  sorry

end perfect_square_trinomial_l3374_337422


namespace max_students_distribution_l3374_337486

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 2730) (h2 : pencils = 1890) :
  Nat.gcd pens pencils = 210 := by
  sorry

end max_students_distribution_l3374_337486


namespace adjacent_numbers_to_10000_l3374_337478

theorem adjacent_numbers_to_10000 :
  let adjacent_numbers (n : ℤ) := (n - 1, n + 1)
  adjacent_numbers 10000 = (9999, 10001) := by
  sorry

end adjacent_numbers_to_10000_l3374_337478


namespace ice_cream_scoop_permutations_l3374_337456

theorem ice_cream_scoop_permutations :
  (Finset.range 5).card.factorial = 120 := by sorry

end ice_cream_scoop_permutations_l3374_337456


namespace tile_arrangements_l3374_337404

/-- The number of distinguishable arrangements of tiles -/
def distinguishable_arrangements (red blue green yellow : ℕ) : ℕ :=
  Nat.factorial (red + blue + green + yellow) /
  (Nat.factorial red * Nat.factorial blue * Nat.factorial green * Nat.factorial yellow)

/-- Theorem stating that the number of distinguishable arrangements
    of 1 red tile, 2 blue tiles, 2 green tiles, and 4 yellow tiles is 3780 -/
theorem tile_arrangements :
  distinguishable_arrangements 1 2 2 4 = 3780 := by
  sorry

end tile_arrangements_l3374_337404


namespace a_range_l3374_337410

def A (a : ℝ) : Set ℝ := {x | -3 ≤ x ∧ x ≤ a}

def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ A a, y = 3*x + 10}

def C (a : ℝ) : Set ℝ := {z | ∃ x ∈ A a, z = 5 - x}

theorem a_range (a : ℝ) : B a ∩ C a = C a → a ∈ Set.Icc (-2/3) 4 := by
  sorry

end a_range_l3374_337410


namespace solution_set_part1_range_of_a_part2_l3374_337440

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end solution_set_part1_range_of_a_part2_l3374_337440


namespace twelve_balloons_floated_away_l3374_337493

/-- Calculates the number of balloons that floated away -/
def balloons_floated_away (initial_count : ℕ) (given_away : ℕ) (grabbed : ℕ) (final_count : ℕ) : ℕ :=
  initial_count - given_away + grabbed - final_count

/-- Proves that 12 balloons floated away given the problem conditions -/
theorem twelve_balloons_floated_away :
  balloons_floated_away 50 10 11 39 = 12 := by
  sorry

#eval balloons_floated_away 50 10 11 39

end twelve_balloons_floated_away_l3374_337493


namespace smallest_r_for_B_subset_C_l3374_337498

def A : Set ℝ := {t | 0 < t ∧ t < 2 * Real.pi}

def B : Set (ℝ × ℝ) := {p | ∃ t ∈ A, p.1 = Real.sin t ∧ p.2 = 2 * Real.sin t * Real.cos t}

def C (r : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ r^2 ∧ r > 0}

theorem smallest_r_for_B_subset_C :
  ∃! r : ℝ, (∀ s : ℝ, B ⊆ C s → r ≤ s) ∧ B ⊆ C r ∧ r = 5/4 := by sorry

end smallest_r_for_B_subset_C_l3374_337498


namespace painted_cubes_l3374_337467

theorem painted_cubes (n : ℕ) (interior_cubes : ℕ) : 
  n = 4 → 
  interior_cubes = 23 → 
  n^3 - interior_cubes = 41 :=
by sorry

end painted_cubes_l3374_337467


namespace second_caterer_cheaper_l3374_337406

/-- The pricing function for the first caterer -/
def first_caterer (x : ℕ) : ℚ := 150 + 18 * x

/-- The pricing function for the second caterer -/
def second_caterer (x : ℕ) : ℚ := 250 + 15 * x

/-- The least number of people for which the second caterer is cheaper -/
def least_people : ℕ := 34

theorem second_caterer_cheaper :
  (∀ n : ℕ, n ≥ least_people → second_caterer n < first_caterer n) ∧
  (∀ n : ℕ, n < least_people → second_caterer n ≥ first_caterer n) := by
  sorry

end second_caterer_cheaper_l3374_337406


namespace salary_difference_l3374_337428

def initial_salary : ℝ := 30000
def hansel_raise_percent : ℝ := 0.10
def gretel_raise_percent : ℝ := 0.15

def hansel_new_salary : ℝ := initial_salary * (1 + hansel_raise_percent)
def gretel_new_salary : ℝ := initial_salary * (1 + gretel_raise_percent)

theorem salary_difference :
  gretel_new_salary - hansel_new_salary = 1500 := by
  sorry

end salary_difference_l3374_337428


namespace system_solution_l3374_337407

theorem system_solution :
  let eq1 (x y z : ℝ) := x^3 + y^3 + z^3 = 8
  let eq2 (x y z : ℝ) := x^2 + y^2 + z^2 = 22
  let eq3 (x y z : ℝ) := 1/x + 1/y + 1/z = -z/(x*y)
  ∀ (x y z : ℝ),
    ((x = 3 ∧ y = 2 ∧ z = -3) ∨
     (x = -3 ∧ y = 2 ∧ z = 3) ∨
     (x = 2 ∧ y = 3 ∧ z = -3) ∨
     (x = 2 ∧ y = -3 ∧ z = 3)) →
    (eq1 x y z ∧ eq2 x y z ∧ eq3 x y z) :=
by sorry

end system_solution_l3374_337407


namespace quadratic_no_real_roots_l3374_337484

theorem quadratic_no_real_roots : ∀ x : ℝ, 3 * x^2 - 6 * x + 4 ≠ 0 := by
  sorry

end quadratic_no_real_roots_l3374_337484


namespace original_deck_size_l3374_337464

/-- Represents a deck of cards with blue and yellow cards -/
structure Deck where
  blue : ℕ
  yellow : ℕ

/-- The probability of drawing a blue card from the deck -/
def blueProbability (d : Deck) : ℚ :=
  d.blue / (d.blue + d.yellow)

/-- Adds yellow cards to the deck -/
def addYellow (d : Deck) (n : ℕ) : Deck :=
  { blue := d.blue, yellow := d.yellow + n }

theorem original_deck_size (d : Deck) :
  blueProbability d = 2/5 ∧ 
  blueProbability (addYellow d 6) = 5/14 →
  d.blue + d.yellow = 50 := by
  sorry

end original_deck_size_l3374_337464


namespace sum_digits_count_numeric_hex_below_2000_l3374_337435

/-- Converts a decimal number to hexadecimal --/
def decimalToHex (n : ℕ) : String := sorry

/-- Counts positive hexadecimal numbers below a given hexadecimal number
    that contain only numeric digits (0-9) --/
def countNumericHex (hex : String) : ℕ := sorry

/-- Sums the digits of a natural number --/
def sumDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of the digits of the count of positive hexadecimal numbers
    below the hexadecimal representation of 2000 that contain only numeric digits (0-9) is 25 --/
theorem sum_digits_count_numeric_hex_below_2000 :
  sumDigits (countNumericHex (decimalToHex 2000)) = 25 := by sorry

end sum_digits_count_numeric_hex_below_2000_l3374_337435


namespace certain_number_exists_and_unique_l3374_337441

theorem certain_number_exists_and_unique : 
  ∃! x : ℝ, (40 * 30 + (12 + x) * 3) / 5 = 1212 := by sorry

end certain_number_exists_and_unique_l3374_337441


namespace range_of_x_l3374_337461

def f (x : ℝ) : ℝ := 3 * x - 2

def assignment_process (x : ℝ) : ℕ → ℝ
| 0 => x
| n + 1 => f (assignment_process x n)

def process_stops (x : ℝ) (k : ℕ) : Prop :=
  assignment_process x (k - 1) ≤ 244 ∧ assignment_process x k > 244

theorem range_of_x (x : ℝ) (k : ℕ) (h : k > 0) (h_stop : process_stops x k) :
  x ∈ Set.Ioo (3^(5 - k) + 1) (3^(6 - k) + 1) :=
sorry

end range_of_x_l3374_337461


namespace twin_primes_difference_divisible_by_twelve_l3374_337446

/-- Twin primes are prime numbers that differ by 2 -/
def IsTwinPrime (p q : ℕ) : Prop :=
  Prime p ∧ Prime q ∧ (q = p + 2 ∨ p = q + 2)

/-- The main theorem statement -/
theorem twin_primes_difference_divisible_by_twelve 
  (p q r s : ℕ) 
  (hp : p > 3) 
  (hq : q > 3) 
  (hr : r > 3) 
  (hs : s > 3) 
  (hpq : IsTwinPrime p q) 
  (hrs : IsTwinPrime r s) : 
  12 ∣ (p * r - q * s) := by
  sorry

end twin_primes_difference_divisible_by_twelve_l3374_337446


namespace solve_exponential_equation_l3374_337479

theorem solve_exponential_equation :
  ∃ x : ℝ, (125 : ℝ) = 5 * (25 : ℝ)^(x - 2) → x = 3 :=
by
  sorry

end solve_exponential_equation_l3374_337479


namespace square_of_binomial_l3374_337472

theorem square_of_binomial (a : ℚ) : 
  (∃ r s : ℚ, ∀ x : ℚ, a * x^2 + 18 * x + 16 = (r * x + s)^2) → 
  a = 81 / 16 := by
sorry

end square_of_binomial_l3374_337472


namespace tennis_match_duration_l3374_337470

def minutes_per_hour : ℕ := 60

def hours : ℕ := 11
def additional_minutes : ℕ := 5

theorem tennis_match_duration : 
  hours * minutes_per_hour + additional_minutes = 665 := by
  sorry

end tennis_match_duration_l3374_337470


namespace pen_cost_l3374_337487

theorem pen_cost (pen pencil : ℚ) 
  (h1 : 3 * pen + 4 * pencil = 264/100)
  (h2 : 4 * pen + 2 * pencil = 230/100) : 
  pen = 392/1000 := by
sorry

end pen_cost_l3374_337487


namespace johns_number_l3374_337458

theorem johns_number : ∃ x : ℝ, (2 * (3 * x - 6) + 20 = 122) ∧ x = 19 := by
  sorry

end johns_number_l3374_337458


namespace cartesian_to_polar_l3374_337443

theorem cartesian_to_polar :
  let x : ℝ := -2
  let y : ℝ := 2 * Real.sqrt 3
  let ρ : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 2 * π / 3
  ρ = 4 ∧ Real.cos θ = x / ρ ∧ Real.sin θ = y / ρ := by sorry

end cartesian_to_polar_l3374_337443


namespace andrea_height_l3374_337495

/-- Given a tree's height and shadow length, and a person's shadow length,
    calculate the person's height assuming the same lighting conditions. -/
theorem andrea_height (tree_height shadow_tree shadow_andrea : ℝ) 
    (h_tree : tree_height = 70)
    (h_shadow_tree : shadow_tree = 14)
    (h_shadow_andrea : shadow_andrea = 3.5) :
  tree_height / shadow_tree * shadow_andrea = 17.5 := by
  sorry

end andrea_height_l3374_337495


namespace m_squared_plus_inverse_squared_plus_three_l3374_337477

theorem m_squared_plus_inverse_squared_plus_three (m : ℝ) (h : m + 1/m = 6) :
  m^2 + 1/m^2 + 3 = 37 := by sorry

end m_squared_plus_inverse_squared_plus_three_l3374_337477


namespace students_per_minibus_l3374_337471

theorem students_per_minibus (total_vehicles : Nat) (num_vans : Nat) (num_minibusses : Nat)
  (students_per_van : Nat) (total_students : Nat) :
  total_vehicles = num_vans + num_minibusses →
  num_vans = 6 →
  num_minibusses = 4 →
  students_per_van = 10 →
  total_students = 156 →
  (total_students - num_vans * students_per_van) / num_minibusses = 24 := by
  sorry

#check students_per_minibus

end students_per_minibus_l3374_337471


namespace octahedron_non_blue_probability_l3374_337455

structure Octahedron :=
  (total_faces : ℕ)
  (blue_faces : ℕ)
  (red_faces : ℕ)
  (green_faces : ℕ)

def non_blue_probability (o : Octahedron) : ℚ :=
  (o.red_faces + o.green_faces : ℚ) / o.total_faces

theorem octahedron_non_blue_probability :
  ∀ o : Octahedron,
  o.total_faces = 8 →
  o.blue_faces = 3 →
  o.red_faces = 3 →
  o.green_faces = 2 →
  non_blue_probability o = 5 / 8 := by
  sorry

end octahedron_non_blue_probability_l3374_337455


namespace shirts_count_l3374_337433

/-- Given a ratio of pants : shorts : shirts and the number of pants, 
    calculate the number of shirts -/
def calculate_shirts (pants_ratio : ℕ) (shorts_ratio : ℕ) (shirts_ratio : ℕ) 
                     (num_pants : ℕ) : ℕ :=
  (num_pants / pants_ratio) * shirts_ratio

/-- Prove that given the ratio 7 : 7 : 10 for pants : shorts : shirts, 
    and 14 pants, there are 20 shirts -/
theorem shirts_count : calculate_shirts 7 7 10 14 = 20 := by
  sorry

end shirts_count_l3374_337433


namespace omega_sum_l3374_337417

theorem omega_sum (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^16 + ω^18 + ω^20 + ω^22 + ω^24 + ω^26 + ω^28 + ω^30 + ω^32 + ω^34 + ω^36 + ω^38 + ω^40 + ω^42 + 
  ω^44 + ω^46 + ω^48 + ω^50 + ω^52 + ω^54 + ω^56 + ω^58 + ω^60 + ω^62 + ω^64 + ω^66 + ω^68 + ω^70 + ω^72 = -ω^7 :=
by sorry

end omega_sum_l3374_337417


namespace max_dot_product_l3374_337436

-- Define the hyperbola
def hyperbola (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 16) - (P.2^2 / 9) = 1

-- Define point A in terms of P and t
def point_A (P : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (t * P.1, t * P.2)

-- Define the dot product condition
def dot_product_condition (P : ℝ × ℝ) (t : ℝ) : Prop :=
  (point_A P t).1 * P.1 + (point_A P t).2 * P.2 = 64

-- Define point B
def B : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem max_dot_product 
  (P : ℝ × ℝ) 
  (t : ℝ) 
  (h1 : hyperbola P) 
  (h2 : dot_product_condition P t) :
  ∃ (M : ℝ), M = 24/5 ∧ ∀ (A : ℝ × ℝ), A = point_A P t → |B.1 * A.1 + B.2 * A.2| ≤ M :=
sorry

end max_dot_product_l3374_337436


namespace average_speed_calculation_l3374_337463

-- Define the variables
def distance_day1 : ℝ := 100
def distance_day2 : ℝ := 175
def time_difference : ℝ := 3

-- Define the theorem
theorem average_speed_calculation :
  ∃ (v : ℝ), v > 0 ∧
    distance_day2 / v - distance_day1 / v = time_difference ∧
    v = 25 := by
  sorry

end average_speed_calculation_l3374_337463


namespace arithmetic_geometric_sequence_sum_l3374_337499

/-- A positive arithmetic-geometric sequence -/
def ArithGeomSeq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r q : ℝ, r > 0 ∧ q > 1 ∧ ∀ k, a (n + k) = a n * r^k * q^(k*(k-1)/2)

theorem arithmetic_geometric_sequence_sum (a : ℕ → ℝ) 
  (h_seq : ArithGeomSeq a) 
  (h_eq : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25) : 
  a 3 + a 5 = 5 := by
sorry

end arithmetic_geometric_sequence_sum_l3374_337499


namespace constant_term_value_l3374_337405

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the expansion of (1+2x)^n
def expansion (n : ℕ) (x : ℝ) : ℝ := sorry

-- Define the coefficient of the r-th term in the expansion
def coefficient (n r : ℕ) : ℝ := sorry

-- Define the condition that only the fourth term has the maximum coefficient
def fourth_term_max (n : ℕ) : Prop :=
  ∀ r, r ≠ 3 → coefficient n r ≤ coefficient n 3

-- Main theorem
theorem constant_term_value (n : ℕ) :
  fourth_term_max n →
  (expansion n 0 + coefficient n 2) = 61 :=
by sorry

end constant_term_value_l3374_337405


namespace system_solution_l3374_337491

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(1, 1, 1), (-1, 1, -1), (1, -1, -1), (-1, -1, 1), (0, 0, 0)}

theorem system_solution (x y z : ℝ) :
  (x * y = z ∧ x * z = y ∧ y * z = x) ↔ (x, y, z) ∈ solution_set := by
  sorry

end system_solution_l3374_337491


namespace angle_in_first_quadrant_l3374_337401

theorem angle_in_first_quadrant (θ : Real) (h : θ = -5) : 
  ∃ n : ℤ, θ + 2 * π * n ∈ Set.Ioo 0 (π / 2) :=
by sorry

end angle_in_first_quadrant_l3374_337401


namespace base9_perfect_square_last_digit_l3374_337449

/-- Represents a number in base 9 of the form ab4d -/
structure Base9Number where
  a : Nat
  b : Nat
  d : Nat
  a_nonzero : a ≠ 0

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : Nat :=
  729 * n.a + 81 * n.b + 36 + n.d

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, n = m * m

theorem base9_perfect_square_last_digit 
  (n : Base9Number) 
  (h : isPerfectSquare (toDecimal n)) : 
  n.d = 0 ∨ n.d = 1 ∨ n.d = 4 ∨ n.d = 7 := by
  sorry

end base9_perfect_square_last_digit_l3374_337449


namespace no_linear_term_implies_sum_zero_l3374_337454

theorem no_linear_term_implies_sum_zero (a b : ℝ) :
  (∀ x : ℝ, (x + a) * (x + b) = x^2 + a*b) → a + b = 0 := by
  sorry

end no_linear_term_implies_sum_zero_l3374_337454


namespace show_length_ratio_l3374_337445

theorem show_length_ratio (first_show_length second_show_length total_time : ℕ) 
  (h1 : first_show_length = 30)
  (h2 : total_time = 150)
  (h3 : second_show_length = total_time - first_show_length) :
  second_show_length / first_show_length = 4 := by
  sorry

end show_length_ratio_l3374_337445


namespace simplify_expressions_l3374_337447

theorem simplify_expressions :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    (Real.sqrt (2 / 3) + 3 * Real.sqrt (1 / 6) - (1 / 2) * Real.sqrt 54 = -(2 * Real.sqrt 6) / 3) ∧
    (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1 / 2) * Real.sqrt 12 + Real.sqrt 24 = 4 + Real.sqrt 6)) :=
by sorry

end simplify_expressions_l3374_337447


namespace horner_v4_value_l3374_337412

def f (x : ℝ) : ℝ := 3 * x^6 - 2 * x^5 + x^3 + 1

def horner_step (v : ℝ) (a : ℝ) (x : ℝ) : ℝ := v * x + a

def horner_method (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (λ acc c => horner_step acc c x) 0

theorem horner_v4_value :
  let coeffs := [1, 0, 1, 0, 0, -2, 3]
  let x := 2
  let v0 := 3
  let v1 := horner_step v0 (-2) x
  let v2 := horner_step v1 0 x
  let v3 := horner_step v2 1 x
  let v4 := horner_step v3 0 x
  v4 = 34 ∧ horner_method coeffs x = f x := by sorry

end horner_v4_value_l3374_337412


namespace erica_earnings_l3374_337451

/-- The amount of money earned per kilogram of fish -/
def price_per_kg : ℝ := 20

/-- The amount of fish caught in the past four months in kilograms -/
def past_four_months_catch : ℝ := 80

/-- The amount of fish caught today in kilograms -/
def today_catch : ℝ := 2 * past_four_months_catch

/-- The total amount of fish caught in kilograms -/
def total_catch : ℝ := past_four_months_catch + today_catch

/-- Erica's total earnings for the past four months including today -/
def total_earnings : ℝ := total_catch * price_per_kg

theorem erica_earnings : total_earnings = 4800 := by
  sorry

end erica_earnings_l3374_337451


namespace medicine_price_reduction_l3374_337438

theorem medicine_price_reduction (x : ℝ) :
  (100 : ℝ) > 0 ∧ (81 : ℝ) > 0 →
  (∃ (initial_price final_price : ℝ),
    initial_price = 100 ∧
    final_price = 81 ∧
    final_price = initial_price * (1 - x) * (1 - x)) →
  100 * (1 - x)^2 = 81 :=
by sorry

end medicine_price_reduction_l3374_337438


namespace min_value_cubic_fraction_l3374_337429

theorem min_value_cubic_fraction (x : ℝ) (h : x > 9) :
  x^3 / (x - 9) ≥ 325 ∧ ∃ y > 9, y^3 / (y - 9) = 325 := by
  sorry

end min_value_cubic_fraction_l3374_337429


namespace max_value_product_l3374_337475

open Real

-- Define the function f(x) = ln(x+2) - x
noncomputable def f (x : ℝ) : ℝ := log (x + 2) - x

-- Define the derivative of f(x)
noncomputable def f' (x : ℝ) : ℝ := 1 / (x + 2) - 1

theorem max_value_product (a b : ℝ) : f' a = 0 → f a = b → a * b = -1 := by
  sorry

end max_value_product_l3374_337475


namespace plane_equation_l3374_337473

def point_on_plane (A B C D : ℤ) (x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

def perpendicular_planes (A1 B1 C1 D1 A2 B2 C2 D2 : ℤ) : Prop :=
  A1 * A2 + B1 * B2 + C1 * C2 = 0

theorem plane_equation : ∃ (A B C D : ℤ),
  (A > 0) ∧
  (Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1) ∧
  point_on_plane A B C D 0 0 0 ∧
  point_on_plane A B C D 2 (-2) 2 ∧
  perpendicular_planes A B C D 2 (-1) 3 4 ∧
  A = 2 ∧ B = -1 ∧ C = 1 ∧ D = 0 :=
sorry

end plane_equation_l3374_337473


namespace fertilizer_transport_l3374_337414

theorem fertilizer_transport (x y t : ℕ) : 
  (x * t = (x - 4) * (t + 10)) →
  (y * t = (y - 3) * (t + 10)) →
  (x * t - y * t = 60) →
  (x - 4 = 8) ∧ (y - 3 = 6) ∧ (t + 10 = 30) :=
by sorry

end fertilizer_transport_l3374_337414


namespace geometric_sequence_common_ratio_l3374_337413

/-- Given a geometric sequence {a_n} with a_1 = 8 and a_4 = 64, the common ratio q is 2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 8 →                     -- first term condition
  a 4 = 64 →                    -- fourth term condition
  q = 2 :=                      -- conclusion: common ratio is 2
by
  sorry


end geometric_sequence_common_ratio_l3374_337413


namespace jane_earnings_l3374_337402

def payment_per_bulb : ℚ := 0.50
def tulip_bulbs : ℕ := 20
def daffodil_bulbs : ℕ := 30

def iris_bulbs : ℕ := tulip_bulbs / 2
def crocus_bulbs : ℕ := daffodil_bulbs * 3

def total_bulbs : ℕ := tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs

def total_payment : ℚ := payment_per_bulb * total_bulbs

theorem jane_earnings : total_payment = 75 := by
  sorry

end jane_earnings_l3374_337402


namespace probability_third_ball_white_l3374_337476

-- Define the problem setup
theorem probability_third_ball_white (n : ℕ) (h : n > 2) :
  let bags := Finset.range n
  let balls_in_bag (k : ℕ) := k + (n - k)
  let prob_choose_bag := 1 / n
  let prob_white_third (k : ℕ) := (n - k) / n
  (bags.sum (λ k => prob_choose_bag * prob_white_third k)) = (n - 1) / (2 * n) :=
by sorry


end probability_third_ball_white_l3374_337476


namespace geometric_number_difference_l3374_337427

/-- A function that checks if a 3-digit number is geometric --/
def is_geometric (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a b c : ℕ) (r : ℚ),
    n = 100 * a + 10 * b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    r > 0 ∧
    b = (a : ℚ) * r ∧
    c = (b : ℚ) * r

/-- A function that checks if a number starts with an even digit --/
def starts_with_even (n : ℕ) : Prop :=
  ∃ (a : ℕ), n = 100 * a + n % 100 ∧ Even a

/-- The theorem to be proved --/
theorem geometric_number_difference :
  ∃ (max min : ℕ),
    is_geometric max ∧
    is_geometric min ∧
    starts_with_even max ∧
    starts_with_even min ∧
    (∀ n, is_geometric n ∧ starts_with_even n → n ≤ max) ∧
    (∀ n, is_geometric n ∧ starts_with_even n → n ≥ min) ∧
    max - min = 403 :=
sorry

end geometric_number_difference_l3374_337427


namespace georgia_carnation_cost_l3374_337408

-- Define the cost of a single carnation
def single_carnation_cost : ℚ := 1/2

-- Define the cost of a dozen carnations
def dozen_carnation_cost : ℚ := 4

-- Define the number of teachers
def num_teachers : ℕ := 5

-- Define the number of friends
def num_friends : ℕ := 14

-- Theorem statement
theorem georgia_carnation_cost : 
  (num_teachers : ℚ) * dozen_carnation_cost + (num_friends : ℚ) * single_carnation_cost = 27 := by
  sorry

end georgia_carnation_cost_l3374_337408


namespace complex_fraction_simplification_l3374_337482

theorem complex_fraction_simplification (z : ℂ) (h : z = 1 - I) :
  (z^2 - 2*z) / (z - 1) = -2*I :=
by sorry

end complex_fraction_simplification_l3374_337482


namespace expand_expression_l3374_337460

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end expand_expression_l3374_337460


namespace jumping_contest_l3374_337485

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump : ℕ) (frog_extra : ℕ) (mouse_extra : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : frog_extra = 10)
  (h3 : mouse_extra = 20) :
  (grasshopper_jump + frog_extra + mouse_extra) - grasshopper_jump = 30 := by
  sorry


end jumping_contest_l3374_337485


namespace arithmetic_sequence_with_geometric_subsequence_l3374_337415

/-- An arithmetic sequence with the property that removing one term results in a geometric sequence -/
def ArithmeticSequenceWithGeometricSubsequence (n : ℕ) (a : ℕ → ℝ) (d : ℝ) : Prop :=
  n ≥ 4 ∧ 
  d ≠ 0 ∧ 
  (∀ i, a i ≠ 0) ∧
  (∀ i, i < n → a (i + 1) = a i + d) ∧
  ∃ k, k < n ∧ 
    (∀ i j, i < j ∧ j < n ∧ i ≠ k ∧ j ≠ k → 
      (a j)^2 = a i * a (if j < k then j + 1 else j))

theorem arithmetic_sequence_with_geometric_subsequence 
  (n : ℕ) (a : ℕ → ℝ) (d : ℝ) : 
  ArithmeticSequenceWithGeometricSubsequence n a d → 
  n = 4 ∧ (a 1 / d = -4 ∨ a 1 / d = 1) :=
sorry

end arithmetic_sequence_with_geometric_subsequence_l3374_337415


namespace impossible_visit_all_squares_l3374_337465

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Fin 6)
  (y : Fin 6)

/-- Represents a jump on the chessboard -/
inductive Jump
  | One
  | Two

/-- Represents a sequence of jumps -/
def JumpSequence := List Jump

/-- Checks if a jump sequence is valid (alternating One and Two) -/
def isValidJumpSequence : JumpSequence → Bool
  | [] => true
  | [_] => true
  | Jump.One :: Jump.Two :: rest => isValidJumpSequence rest
  | Jump.Two :: Jump.One :: rest => isValidJumpSequence rest
  | _ => false

/-- Applies a jump to a position -/
def applyJump (pos : Position) (jump : Jump) (direction : Bool) : Position :=
  match jump, direction with
  | Jump.One, true => ⟨pos.x + 1, pos.y⟩
  | Jump.One, false => ⟨pos.x, pos.y + 1⟩
  | Jump.Two, true => ⟨pos.x + 2, pos.y⟩
  | Jump.Two, false => ⟨pos.x, pos.y + 2⟩

/-- Applies a sequence of jumps to a position -/
def applyJumpSequence (pos : Position) (jumps : JumpSequence) (directions : List Bool) : List Position :=
  match jumps, directions with
  | [], _ => [pos]
  | j :: js, d :: ds => pos :: applyJumpSequence (applyJump pos j d) js ds
  | _, _ => [pos]

/-- Theorem: It's impossible to visit all squares on a 6x6 chessboard
    with 35 jumps alternating between 1 and 2 squares -/
theorem impossible_visit_all_squares :
  ∀ (start : Position) (jumps : JumpSequence) (directions : List Bool),
    isValidJumpSequence jumps →
    jumps.length = 35 →
    directions.length = 35 →
    ¬(∀ (p : Position), p ∈ applyJumpSequence start jumps directions) :=
by
  sorry

end impossible_visit_all_squares_l3374_337465


namespace sum_ab_equals_four_l3374_337453

theorem sum_ab_equals_four (a b c d : ℤ) 
  (h1 : b + c = 7) 
  (h2 : c + d = 5) 
  (h3 : a + d = 2) : 
  a + b = 4 := by
sorry

end sum_ab_equals_four_l3374_337453


namespace sqrt_meaningful_range_l3374_337424

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 1) → x ≥ 1 / 2 := by
  sorry

end sqrt_meaningful_range_l3374_337424


namespace line_perp_parallel_implies_planes_perp_l3374_337416

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (m : Line) (α β : Plane) :
  perpendicular m β → parallel m α → perp_planes α β :=
sorry

end line_perp_parallel_implies_planes_perp_l3374_337416


namespace sum_of_digits_divisibility_l3374_337483

/-- Sum of digits function -/
def sum_of_digits (a : ℕ) : ℕ := sorry

/-- Theorem: If the sum of digits of a equals the sum of digits of 2a, then a is divisible by 9 -/
theorem sum_of_digits_divisibility (a : ℕ) : sum_of_digits a = sum_of_digits (2 * a) → 9 ∣ a := by sorry

end sum_of_digits_divisibility_l3374_337483


namespace complex_equation_solution_l3374_337425

theorem complex_equation_solution (z : ℂ) : z * Complex.I = -1 + (3/4) * Complex.I → z = 3/4 - Complex.I := by
  sorry

end complex_equation_solution_l3374_337425


namespace triangle_side_range_l3374_337418

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def validTriangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = 180

-- Theorem statement
theorem triangle_side_range (x : ℝ) :
  (∃ t1 t2 : Triangle, 
    validTriangle t1 ∧ validTriangle t2 ∧
    t1.b = 2 ∧ t2.b = 2 ∧
    t1.B = 60 ∧ t2.B = 60 ∧
    t1.a = x ∧ t2.a = x ∧
    t1 ≠ t2) →
  2 < x ∧ x < (4 * Real.sqrt 3) / 3 :=
sorry

end triangle_side_range_l3374_337418


namespace intersection_A_complement_B_l3374_337409

/-- The universal set U -/
def U : Set ℕ := {1, 2, 3, 4, 5}

/-- Set A -/
def A : Set ℕ := {1, 3, 4}

/-- Set B -/
def B : Set ℕ := {4, 5}

/-- Theorem stating that the intersection of A and the complement of B with respect to U is {1, 3} -/
theorem intersection_A_complement_B : A ∩ (U \ B) = {1, 3} := by sorry

end intersection_A_complement_B_l3374_337409


namespace cannot_equalize_sugar_l3374_337488

/-- Represents a jar with tea and sugar -/
structure Jar :=
  (volume : ℚ)
  (sugar : ℚ)

/-- Represents the state of all three jars -/
structure JarState :=
  (jar1 : Jar)
  (jar2 : Jar)
  (jar3 : Jar)

/-- Represents a single pouring operation -/
inductive PourOperation
  | pour12 : PourOperation  -- Pour from jar1 to jar2
  | pour13 : PourOperation  -- Pour from jar1 to jar3
  | pour21 : PourOperation  -- Pour from jar2 to jar1
  | pour23 : PourOperation  -- Pour from jar2 to jar3
  | pour31 : PourOperation  -- Pour from jar3 to jar1
  | pour32 : PourOperation  -- Pour from jar3 to jar2

def initialState : JarState :=
  { jar1 := { volume := 0, sugar := 0 },
    jar2 := { volume := 700/1000, sugar := 50 },
    jar3 := { volume := 800/1000, sugar := 60 } }

def measureCup : ℚ := 100/1000

/-- Applies a single pouring operation to the current state -/
def applyOperation (state : JarState) (op : PourOperation) : JarState :=
  sorry

/-- Checks if the sugar content is equal in jars 2 and 3, and jar 1 is empty -/
def isDesiredState (state : JarState) : Prop :=
  state.jar1.volume = 0 ∧ state.jar2.sugar = state.jar3.sugar

/-- The main theorem to prove -/
theorem cannot_equalize_sugar : ¬∃ (ops : List PourOperation),
  isDesiredState (ops.foldl applyOperation initialState) :=
sorry

end cannot_equalize_sugar_l3374_337488


namespace complex_point_location_l3374_337426

theorem complex_point_location (z : ℂ) (h : z = Complex.I * 2) : 
  z.re = 0 ∧ z.im > 0 :=
by sorry

end complex_point_location_l3374_337426


namespace factorization_of_5x_cubed_minus_125x_l3374_337403

theorem factorization_of_5x_cubed_minus_125x (x : ℝ) :
  5 * x^3 - 125 * x = 5 * x * (x + 5) * (x - 5) := by
  sorry

end factorization_of_5x_cubed_minus_125x_l3374_337403


namespace suzanna_bike_ride_l3374_337432

/-- Calculates the distance traveled given a constant speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem suzanna_bike_ride :
  let rate : ℝ := 0.75 / 5  -- miles per minute
  let time : ℝ := 45        -- minutes
  distance_traveled rate time = 6.75 := by
  sorry


end suzanna_bike_ride_l3374_337432


namespace clock_chimes_in_day_l3374_337439

/-- Calculates the number of chimes a clock makes in a day -/
def clock_chimes : ℕ :=
  let hours_in_day : ℕ := 24
  let half_hours_in_day : ℕ := hours_in_day * 2
  let sum_of_hour_strikes : ℕ := (12 * (1 + 12)) / 2
  let total_hour_strikes : ℕ := sum_of_hour_strikes * 2
  let total_half_hour_strikes : ℕ := half_hours_in_day
  total_hour_strikes + total_half_hour_strikes

/-- Theorem stating that a clock striking hours (1 to 12) and half-hours in a 24-hour day will chime 204 times -/
theorem clock_chimes_in_day : clock_chimes = 204 := by
  sorry

end clock_chimes_in_day_l3374_337439


namespace lesser_fraction_l3374_337450

theorem lesser_fraction (x y : ℚ) (sum_eq : x + y = 17/24) (prod_eq : x * y = 1/8) :
  min x y = 1/3 := by
  sorry

end lesser_fraction_l3374_337450


namespace min_value_sum_squares_l3374_337489

theorem min_value_sum_squares (x y z : ℝ) (h : 2*x + 3*y + 4*z = 1) :
  ∃ (min : ℝ), min = (1 : ℝ) / 29 ∧ x^2 + y^2 + z^2 ≥ min ∧
  (x^2 + y^2 + z^2 = min ↔ x/2 = y/3 ∧ y/3 = z/4) :=
sorry

end min_value_sum_squares_l3374_337489


namespace bugs_meeting_time_l3374_337437

/-- The time for two bugs to meet again at the starting point on two tangent circles -/
theorem bugs_meeting_time (r1 r2 v1 v2 : ℝ) (hr1 : r1 = 8) (hr2 : r2 = 4) 
  (hv1 : v1 = 3 * Real.pi) (hv2 : v2 = 4 * Real.pi) : 
  ∃ t : ℝ, t = 48 ∧ 
  (∃ n1 n2 : ℕ, t * v1 = n1 * (2 * Real.pi * r1) ∧ 
               t * v2 = n2 * (2 * Real.pi * r2)) := by
  sorry

#check bugs_meeting_time

end bugs_meeting_time_l3374_337437


namespace expand_difference_of_squares_l3374_337480

theorem expand_difference_of_squares (a : ℝ) : (a + 1) * (a - 1) = a^2 - 1 := by
  sorry

end expand_difference_of_squares_l3374_337480


namespace sams_remaining_pennies_l3374_337423

/-- Given an initial amount of pennies and an amount spent, calculate the remaining pennies. -/
def remaining_pennies (initial : ℕ) (spent : ℕ) : ℕ := initial - spent

/-- Theorem: Sam's remaining pennies -/
theorem sams_remaining_pennies :
  let initial := 98
  let spent := 93
  remaining_pennies initial spent = 5 := by
  sorry

end sams_remaining_pennies_l3374_337423
