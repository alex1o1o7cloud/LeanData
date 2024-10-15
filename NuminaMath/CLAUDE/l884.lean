import Mathlib

namespace NUMINAMATH_CALUDE_erasers_per_box_l884_88437

/-- Given that Jacqueline has 4 boxes of erasers and a total of 40 erasers,
    prove that there are 10 erasers in each box. -/
theorem erasers_per_box (total_erasers : ℕ) (num_boxes : ℕ) (h1 : total_erasers = 40) (h2 : num_boxes = 4) :
  total_erasers / num_boxes = 10 := by
  sorry

#check erasers_per_box

end NUMINAMATH_CALUDE_erasers_per_box_l884_88437


namespace NUMINAMATH_CALUDE_correct_statements_are_1_and_3_l884_88474

-- Define the proof methods
inductive ProofMethod
| Synthetic
| Analytic
| Contradiction

-- Define the properties of proof methods
def isCauseToEffect (m : ProofMethod) : Prop := m = ProofMethod.Synthetic
def isEffectToCause (m : ProofMethod) : Prop := m = ProofMethod.Analytic
def isDirectMethod (m : ProofMethod) : Prop := m = ProofMethod.Synthetic ∨ m = ProofMethod.Analytic
def isIndirectMethod (m : ProofMethod) : Prop := m = ProofMethod.Contradiction

-- Define the statements
def statement1 : Prop := isCauseToEffect ProofMethod.Synthetic
def statement2 : Prop := isIndirectMethod ProofMethod.Analytic
def statement3 : Prop := isEffectToCause ProofMethod.Analytic
def statement4 : Prop := isDirectMethod ProofMethod.Contradiction

-- Theorem to prove
theorem correct_statements_are_1_and_3 :
  (statement1 ∧ statement3) ∧ (¬statement2 ∧ ¬statement4) :=
sorry

end NUMINAMATH_CALUDE_correct_statements_are_1_and_3_l884_88474


namespace NUMINAMATH_CALUDE_triangle_base_length_l884_88447

/-- Proves that a triangle with area 54 square meters and height 6 meters has a base of 18 meters -/
theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 54 →
  height = 6 →
  area = (base * height) / 2 →
  base = 18 := by
sorry

end NUMINAMATH_CALUDE_triangle_base_length_l884_88447


namespace NUMINAMATH_CALUDE_prime_power_gcd_condition_l884_88455

theorem prime_power_gcd_condition (n : ℕ) (h : n > 1) :
  (∀ m : ℕ, 1 ≤ m ∧ m < n →
    Nat.gcd n ((n - m) / Nat.gcd n m) = 1) ↔
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ k > 0 ∧ n = p^k :=
by sorry

end NUMINAMATH_CALUDE_prime_power_gcd_condition_l884_88455


namespace NUMINAMATH_CALUDE_normal_dist_symmetry_l884_88433

-- Define a random variable with normal distribution
def normal_dist (μ σ : ℝ) : Type := ℝ

-- Define the probability function
noncomputable def P (ξ : normal_dist 0 σ) (event : Set ℝ) : ℝ := sorry

-- Theorem statement
theorem normal_dist_symmetry (σ : ℝ) (ξ : normal_dist 0 σ) :
  P ξ {x | x < 2} = 0.8 → P ξ {x | x < -2} = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_normal_dist_symmetry_l884_88433


namespace NUMINAMATH_CALUDE_complement_of_union_A_B_l884_88423

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x ≤ 1}

-- Define set B
def B : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_of_union_A_B : 
  (A ∪ B)ᶜ = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_A_B_l884_88423


namespace NUMINAMATH_CALUDE_room_width_calculation_l884_88483

theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (width : ℝ) : 
  length = 8 →
  cost_per_sqm = 900 →
  total_cost = 34200 →
  width = total_cost / cost_per_sqm / length →
  width = 4.75 :=
by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l884_88483


namespace NUMINAMATH_CALUDE_problem_solution_l884_88499

noncomputable def problem (a b c k x y z : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ k ≠ 0 ∧
  x * y / (x + y) = a ∧
  x * z / (x + z) = b ∧
  y * z / (y + z) = c ∧
  x * y * z / (x + y + z) = k

theorem problem_solution (a b c k x y z : ℝ) (h : problem a b c k x y z) :
  x = 2 * k * a * b / (a * b + b * c - a * c) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l884_88499


namespace NUMINAMATH_CALUDE_sum_of_square_roots_l884_88426

theorem sum_of_square_roots (x : ℝ) 
  (h1 : -Real.sqrt 15 ≤ x ∧ x ≤ Real.sqrt 15) 
  (h2 : Real.sqrt (25 - x^2) - Real.sqrt (15 - x^2) = 2) : 
  Real.sqrt (25 - x^2) + Real.sqrt (15 - x^2) = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_l884_88426


namespace NUMINAMATH_CALUDE_exists_line_intersecting_four_circles_l884_88402

/-- Represents a circle with a given diameter -/
structure Circle where
  diameter : ℝ

/-- Represents a configuration of circles in a unit square -/
structure CircleConfiguration where
  circles : List Circle
  sum_of_circumferences_eq_10 : (circles.map (fun c => c.diameter * Real.pi)).sum = 10

/-- Main theorem: If the sum of circumferences of circles in a unit square is 10,
    then there exists a line intersecting at least 4 of these circles -/
theorem exists_line_intersecting_four_circles (config : CircleConfiguration) :
  ∃ (line : ℝ → ℝ → Prop), (∃ (intersected_circles : List Circle),
    intersected_circles.length ≥ 4 ∧
    ∀ c ∈ intersected_circles, c ∈ config.circles ∧
    ∃ (x y : ℝ), x ∈ Set.Icc 0 1 ∧ y ∈ Set.Icc 0 1 ∧ line x y) :=
sorry

end NUMINAMATH_CALUDE_exists_line_intersecting_four_circles_l884_88402


namespace NUMINAMATH_CALUDE_intersection_complement_A_with_B_l884_88420

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 4}

theorem intersection_complement_A_with_B :
  (U \ A) ∩ B = {4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_A_with_B_l884_88420


namespace NUMINAMATH_CALUDE_family_income_theorem_l884_88430

theorem family_income_theorem (initial_members : ℕ) (new_average : ℝ) (deceased_income : ℝ) :
  initial_members = 4 →
  new_average = 650 →
  deceased_income = 990 →
  (initial_members - 1) * new_average + deceased_income = initial_members * 735 :=
by sorry

end NUMINAMATH_CALUDE_family_income_theorem_l884_88430


namespace NUMINAMATH_CALUDE_basketball_selection_probabilities_l884_88422

def shot_probability : ℚ := 2/3

def second_level_after_three_shots : ℚ := 8/27

def selected_probability : ℚ := 64/81

def selected_after_five_shots : ℚ := 16/81

theorem basketball_selection_probabilities :
  (2 * shot_probability * (1 - shot_probability) * shot_probability = second_level_after_three_shots) ∧
  (selected_after_five_shots / selected_probability = 1/4) := by
  sorry

end NUMINAMATH_CALUDE_basketball_selection_probabilities_l884_88422


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l884_88498

def total_area : ℝ := 700
def smaller_area : ℝ := 315

theorem area_ratio_theorem :
  let larger_area := total_area - smaller_area
  let difference := larger_area - smaller_area
  let average := (larger_area + smaller_area) / 2
  difference / average = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l884_88498


namespace NUMINAMATH_CALUDE_total_earnings_after_seven_days_l884_88489

/- Define the prices of books -/
def fantasy_price : ℕ := 6
def literature_price : ℕ := fantasy_price / 2
def mystery_price : ℕ := 4

/- Define the daily sales quantities -/
def fantasy_sales : ℕ := 5
def literature_sales : ℕ := 8
def mystery_sales : ℕ := 3

/- Define the number of days -/
def days : ℕ := 7

/- Calculate daily earnings -/
def daily_earnings : ℕ := 
  fantasy_sales * fantasy_price + 
  literature_sales * literature_price + 
  mystery_sales * mystery_price

/- Theorem to prove -/
theorem total_earnings_after_seven_days : 
  daily_earnings * days = 462 := by sorry

end NUMINAMATH_CALUDE_total_earnings_after_seven_days_l884_88489


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l884_88450

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem volleyball_team_selection :
  let total_players : ℕ := 14
  let triplets : ℕ := 3
  let starters : ℕ := 6
  let non_triplet_players : ℕ := total_players - triplets
  let lineups_without_triplets : ℕ := choose non_triplet_players starters
  let lineups_with_one_triplet : ℕ := triplets * (choose non_triplet_players (starters - 1))
  lineups_without_triplets + lineups_with_one_triplet = 1848 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l884_88450


namespace NUMINAMATH_CALUDE_smallest_covering_circle_l884_88470

-- Define the plane region
def PlaneRegion (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ x + 2*y - 4 ≤ 0

-- Define the circle equation
def Circle (a b r : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem smallest_covering_circle :
  ∃ (C : ℝ → ℝ → Prop),
    (∀ x y, PlaneRegion x y → C x y) ∧
    (∀ D : ℝ → ℝ → Prop, (∀ x y, PlaneRegion x y → D x y) → 
      ∃ a b r, C = Circle a b r ∧ 
      ∀ a' b' r', D = Circle a' b' r' → r ≤ r') ∧
    C = Circle 2 1 (Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_smallest_covering_circle_l884_88470


namespace NUMINAMATH_CALUDE_carly_dog_grooming_l884_88471

theorem carly_dog_grooming (total_nails : ℕ) (three_legged_dogs : ℕ) :
  total_nails = 164 →
  three_legged_dogs = 3 →
  ∃ (total_dogs : ℕ),
    total_dogs * 4 * 4 - three_legged_dogs * 4 = total_nails ∧
    total_dogs = 11 :=
by sorry

end NUMINAMATH_CALUDE_carly_dog_grooming_l884_88471


namespace NUMINAMATH_CALUDE_range_of_a_max_value_sum_of_roots_l884_88401

-- Define the function f
def f (x : ℝ) : ℝ := |x - 4| - |x + 2|

-- Part 1
theorem range_of_a (a : ℝ) :
  (∀ x, f x - a^2 + 5*a ≥ 0) → 2 ≤ a ∧ a ≤ 3 :=
sorry

-- Part 2
theorem max_value_sum_of_roots (M a b c : ℝ) :
  (∀ x, f x ≤ M) →
  a > 0 → b > 0 → c > 0 →
  a + b + c = M →
  (∃ (max_val : ℝ), ∀ a' b' c' : ℝ,
    a' > 0 → b' > 0 → c' > 0 →
    a' + b' + c' = M →
    Real.sqrt (a' + 1) + Real.sqrt (b' + 2) + Real.sqrt (c' + 3) ≤ max_val ∧
    max_val = 6) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_max_value_sum_of_roots_l884_88401


namespace NUMINAMATH_CALUDE_right_triangle_area_l884_88446

/-- The area of a right triangle with one leg of 12cm and a hypotenuse of 13cm is 30 cm². -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 12) (h2 : c = 13) 
    (h3 : a^2 + b^2 = c^2) : (1/2) * a * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l884_88446


namespace NUMINAMATH_CALUDE_squirrel_rainy_days_l884_88419

theorem squirrel_rainy_days 
  (sunny_nuts : ℕ) 
  (rainy_nuts : ℕ) 
  (total_nuts : ℕ) 
  (average_nuts : ℕ) 
  (h1 : sunny_nuts = 20)
  (h2 : rainy_nuts = 12)
  (h3 : total_nuts = 112)
  (h4 : average_nuts = 14)
  : ∃ (rainy_days : ℕ), rainy_days = 6 ∧ 
    ∃ (total_days : ℕ), 
      total_days * average_nuts = total_nuts ∧
      rainy_days * rainy_nuts + (total_days - rainy_days) * sunny_nuts = total_nuts :=
by sorry

end NUMINAMATH_CALUDE_squirrel_rainy_days_l884_88419


namespace NUMINAMATH_CALUDE_inequality_solution_set_l884_88452

-- Define the inequality
def inequality (x : ℝ) : Prop := (3 * x - 1) / (2 - x) ≥ 0

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | 1/3 ≤ x ∧ x < 2}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l884_88452


namespace NUMINAMATH_CALUDE_short_story_section_pages_l884_88467

/-- The number of pages in each booklet -/
def pages_per_booklet : ℕ := 9

/-- The number of booklets in the short story section -/
def number_of_booklets : ℕ := 49

/-- The total number of pages in the short story section -/
def total_pages : ℕ := pages_per_booklet * number_of_booklets

theorem short_story_section_pages :
  total_pages = 441 :=
by sorry

end NUMINAMATH_CALUDE_short_story_section_pages_l884_88467


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l884_88494

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧
  n % 2 = 1 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 10 = 9 ∧
  (∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 10 = 9 → n ≤ m) ∧
  n = 59 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l884_88494


namespace NUMINAMATH_CALUDE_apple_balance_theorem_l884_88469

variable {α : Type*} [LinearOrderedField α]

def balanced (s t : Finset (α)) : Prop :=
  s.sum id = t.sum id

theorem apple_balance_theorem
  (apples : Finset α)
  (h_count : apples.card = 6)
  (h_tanya : ∃ (s t : Finset α), s ⊆ apples ∧ t ⊆ apples ∧ s ∩ t = ∅ ∧ s ∪ t = apples ∧ s.card = 3 ∧ t.card = 3 ∧ balanced s t)
  (h_sasha : ∃ (u v : Finset α), u ⊆ apples ∧ v ⊆ apples ∧ u ∩ v = ∅ ∧ u ∪ v = apples ∧ u.card = 2 ∧ v.card = 4 ∧ balanced u v) :
  ∃ (x y : Finset α), x ⊆ apples ∧ y ⊆ apples ∧ x ∩ y = ∅ ∧ x ∪ y = apples ∧ x.card = 1 ∧ y.card = 2 ∧ balanced x y :=
by
  sorry

end NUMINAMATH_CALUDE_apple_balance_theorem_l884_88469


namespace NUMINAMATH_CALUDE_pencil_count_l884_88444

/-- The total number of pencils after adding more to an initial amount -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem: Given 41 initial pencils and 30 added pencils, the total is 71 -/
theorem pencil_count : total_pencils 41 30 = 71 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l884_88444


namespace NUMINAMATH_CALUDE_jessica_red_marbles_l884_88488

theorem jessica_red_marbles (sandy_marbles : ℕ) (sandy_times_more : ℕ) :
  sandy_marbles = 144 →
  sandy_times_more = 4 →
  (sandy_marbles / sandy_times_more) / 12 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_jessica_red_marbles_l884_88488


namespace NUMINAMATH_CALUDE_complex_inequality_l884_88486

theorem complex_inequality : ∀ (i : ℂ), i^2 = -1 → Complex.abs (2 - i) > 2 * (i^4).re :=
fun i h =>
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l884_88486


namespace NUMINAMATH_CALUDE_coffee_ratio_is_two_to_one_l884_88481

/-- Represents the amount of coffee used for different strengths -/
structure CoffeeAmount where
  weak : ℕ
  strong : ℕ

/-- Calculates the ratio of strong to weak coffee -/
def coffeeRatio (amount : CoffeeAmount) : ℚ :=
  amount.strong / amount.weak

/-- Theorem stating the ratio of strong to weak coffee is 2:1 -/
theorem coffee_ratio_is_two_to_one :
  ∃ (amount : CoffeeAmount),
    amount.weak + amount.strong = 36 ∧
    amount.weak = 12 ∧
    coffeeRatio amount = 2 := by
  sorry

end NUMINAMATH_CALUDE_coffee_ratio_is_two_to_one_l884_88481


namespace NUMINAMATH_CALUDE_election_total_votes_l884_88407

/-- Represents an election between two candidates -/
structure Election where
  total_votes : ℕ
  invalid_percent : ℚ
  b_votes : ℕ
  a_excess_percent : ℚ

/-- The election satisfies the given conditions -/
def valid_election (e : Election) : Prop :=
  e.invalid_percent = 1/5 ∧
  e.a_excess_percent = 3/20 ∧
  e.b_votes = 2184 ∧
  (e.total_votes : ℚ) * (1 - e.invalid_percent) = 
    (e.b_votes : ℚ) + (e.b_votes : ℚ) + e.total_votes * e.a_excess_percent

theorem election_total_votes (e : Election) (h : valid_election e) : 
  e.total_votes = 6720 := by
  sorry

#check election_total_votes

end NUMINAMATH_CALUDE_election_total_votes_l884_88407


namespace NUMINAMATH_CALUDE_sin_decreasing_interval_l884_88449

/-- The monotonic decreasing interval of sin(π/3 - 2x) -/
theorem sin_decreasing_interval (k : ℤ) :
  let f : ℝ → ℝ := λ x => Real.sin (π/3 - 2*x)
  ∀ x ∈ Set.Icc (k*π - π/12) (k*π + 5*π/12),
    ∀ y ∈ Set.Icc (k*π - π/12) (k*π + 5*π/12),
      x ≤ y → f y ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_sin_decreasing_interval_l884_88449


namespace NUMINAMATH_CALUDE_dice_sum_impossibility_l884_88493

theorem dice_sum_impossibility (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 →
  1 ≤ b ∧ b ≤ 6 →
  1 ≤ c ∧ c ≤ 6 →
  1 ≤ d ∧ d ≤ 6 →
  a * b * c * d = 216 →
  a + b + c + d ≠ 20 := by
sorry

end NUMINAMATH_CALUDE_dice_sum_impossibility_l884_88493


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l884_88491

theorem geometric_sequence_problem (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 280) (h₂ : a₂ > 0) (h₃ : a₃ = 90 / 56) 
  (h₄ : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) : a₂ = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l884_88491


namespace NUMINAMATH_CALUDE_toms_lawn_mowing_l884_88403

/-- Proves the number of lawns Tom mowed given his earnings and expenses -/
theorem toms_lawn_mowing (charge_per_lawn : ℕ) (gas_expense : ℕ) (weed_income : ℕ) (total_profit : ℕ) 
  (h1 : charge_per_lawn = 12)
  (h2 : gas_expense = 17)
  (h3 : weed_income = 10)
  (h4 : total_profit = 29) :
  ∃ (lawns_mowed : ℕ), 
    lawns_mowed * charge_per_lawn + weed_income - gas_expense = total_profit ∧ 
    lawns_mowed = 3 := by
  sorry


end NUMINAMATH_CALUDE_toms_lawn_mowing_l884_88403


namespace NUMINAMATH_CALUDE_probability_factor_less_than_eight_l884_88439

theorem probability_factor_less_than_eight (n : ℕ) (h : n = 90) :
  let factors := {d : ℕ | d > 0 ∧ n % d = 0}
  let factors_less_than_eight := {d ∈ factors | d < 8}
  Nat.card factors_less_than_eight / Nat.card factors = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_factor_less_than_eight_l884_88439


namespace NUMINAMATH_CALUDE_net_amount_theorem_l884_88435

def net_amount_spent (shorts_cost shirt_cost jacket_return : ℚ) : ℚ :=
  shorts_cost + shirt_cost - jacket_return

theorem net_amount_theorem (shorts_cost shirt_cost jacket_return : ℚ) :
  net_amount_spent shorts_cost shirt_cost jacket_return =
  shorts_cost + shirt_cost - jacket_return := by
  sorry

#eval net_amount_spent (13.99 : ℚ) (12.14 : ℚ) (7.43 : ℚ)

end NUMINAMATH_CALUDE_net_amount_theorem_l884_88435


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l884_88485

theorem least_number_for_divisibility : 
  ∃! x : ℕ, x < 577 ∧ (907223 + x) % 577 = 0 ∧ 
  ∀ y : ℕ, y < x → (907223 + y) % 577 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l884_88485


namespace NUMINAMATH_CALUDE_matrix_product_R_S_l884_88411

def R : Matrix (Fin 3) (Fin 3) ℝ := !![0, -1, 0; 1, 0, 0; 0, 0, 1]

def S (b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![d^2, d*c, d*b; d*c, c^2, c*b; d*b, c*b, b^2]

theorem matrix_product_R_S (b c d : ℝ) :
  R * S b c d = !![-(d*c), -(c^2), -(c*b); d^2, d*c, d*b; d*b, c*b, b^2] := by
  sorry

end NUMINAMATH_CALUDE_matrix_product_R_S_l884_88411


namespace NUMINAMATH_CALUDE_product_of_fractions_and_powers_of_three_l884_88410

theorem product_of_fractions_and_powers_of_three (x : ℚ) : 
  x = (1/3 : ℚ) * 9 * (1/27 : ℚ) * 81 * (1/243 : ℚ) * 729 * (1/2187 : ℚ) * 6561 * (1/19683 : ℚ) * 59049 → 
  x = 243 := by
sorry

end NUMINAMATH_CALUDE_product_of_fractions_and_powers_of_three_l884_88410


namespace NUMINAMATH_CALUDE_frog_population_difference_l884_88457

/-- Theorem: Given the conditions about frog populations in two lakes, prove that the percentage difference is 20%. -/
theorem frog_population_difference (lassie_frogs : ℕ) (total_frogs : ℕ) (P : ℝ) : 
  lassie_frogs = 45 →
  total_frogs = 81 →
  total_frogs = lassie_frogs + (lassie_frogs - P / 100 * lassie_frogs) →
  P = 20 := by
  sorry

end NUMINAMATH_CALUDE_frog_population_difference_l884_88457


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l884_88409

/-- Given a line with equation y - 6 = -3(x - 5), 
    the sum of its x-intercept and y-intercept is 28 -/
theorem line_intercepts_sum : 
  ∀ (x y : ℝ), y - 6 = -3 * (x - 5) → 
  ∃ (x_int y_int : ℝ),
    (y_int - 6 = -3 * (x_int - 5) ∧ y_int = 0) ∧
    (0 - 6 = -3 * (0 - 5) ∧ y = y_int) ∧
    x_int + y_int = 28 :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l884_88409


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l884_88415

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) = 60) → 
  ((n + 2) + (n + 3) + (n + 4) = 66) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l884_88415


namespace NUMINAMATH_CALUDE_expression_simplification_l884_88456

theorem expression_simplification (x y : ℝ) : 1 - (2 - (3 - (4 - (5 - x)))) - y = 3 - (x + y) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l884_88456


namespace NUMINAMATH_CALUDE_robin_final_candy_l884_88434

def initial_candy : ℕ := 23

def eaten_fraction : ℚ := 2/3

def sister_bonus_fraction : ℚ := 1/2

theorem robin_final_candy : 
  ∃ (eaten : ℕ) (leftover : ℕ) (bonus : ℕ),
    eaten = ⌊(eaten_fraction : ℚ) * initial_candy⌋ ∧
    leftover = initial_candy - eaten ∧
    bonus = ⌊(sister_bonus_fraction : ℚ) * initial_candy⌋ ∧
    leftover + bonus = 19 :=
by sorry

end NUMINAMATH_CALUDE_robin_final_candy_l884_88434


namespace NUMINAMATH_CALUDE_complement_M_inter_N_l884_88453

def M : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≥ 0}
def N : Set ℝ := {x : ℝ | |x - 2| ≤ 1}

theorem complement_M_inter_N : 
  (Set.compl M) ∩ N = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_complement_M_inter_N_l884_88453


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l884_88414

-- Define the given line
def given_line (x y : ℝ) : Prop := 2*x + y - 5 = 0

-- Define the point A
def point_A : ℝ × ℝ := (2, 3)

-- Define the new line
def new_line (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Theorem statement
theorem perpendicular_line_through_point :
  (∀ x y : ℝ, given_line x y → (new_line x y → ¬given_line x y)) ∧
  new_line point_A.1 point_A.2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l884_88414


namespace NUMINAMATH_CALUDE_parallel_transitivity_l884_88404

-- Define a type for lines
variable (Line : Type)

-- Define a relation for parallel lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem parallel_transitivity (l1 l2 l3 : Line) :
  parallel l1 l3 → parallel l2 l3 → parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l884_88404


namespace NUMINAMATH_CALUDE_divisibility_probability_l884_88443

/-- The number of positive divisors of 10^99 -/
def total_divisors : ℕ := 10000

/-- The number of positive divisors of 10^99 that are multiples of 10^88 -/
def favorable_divisors : ℕ := 144

/-- The probability of a randomly chosen positive divisor of 10^99 being an integer multiple of 10^88 -/
def probability : ℚ := favorable_divisors / total_divisors

theorem divisibility_probability :
  probability = 9 / 625 :=
sorry

end NUMINAMATH_CALUDE_divisibility_probability_l884_88443


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l884_88490

/-- Given a cone with base radius 6 and volume 30π, its lateral surface area is 39π -/
theorem cone_lateral_surface_area (r h l : ℝ) : 
  r = 6 → 
  (1 / 3) * π * r^2 * h = 30 * π → 
  l^2 = r^2 + h^2 → 
  π * r * l = 39 * π := by
sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l884_88490


namespace NUMINAMATH_CALUDE_cosine_triangle_condition_l884_88408

/-- A cubic equation with real coefficients -/
structure CubicEquation where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The roots of a cubic equation are all real and positive -/
def has_real_positive_roots (eq : CubicEquation) : Prop := sorry

/-- The roots of a cubic equation are the cosines of the angles of a triangle -/
def roots_are_triangle_cosines (eq : CubicEquation) : Prop := sorry

/-- The necessary and sufficient condition for the roots to be the cosines of the angles of a triangle -/
theorem cosine_triangle_condition (eq : CubicEquation) :
  roots_are_triangle_cosines eq ↔
    eq.p^2 - 2*eq.q - 2*eq.r - 1 = 0 ∧ eq.p < 0 ∧ eq.q > 0 ∧ eq.r < 0 :=
by sorry

end NUMINAMATH_CALUDE_cosine_triangle_condition_l884_88408


namespace NUMINAMATH_CALUDE_system_solution_exists_no_solution_when_m_eq_one_l884_88459

theorem system_solution_exists (m : ℝ) (h : m ≠ 1) :
  ∃ (x y : ℝ), y = m * x + 4 ∧ y = (3 * m - 2) * x + 5 := by
  sorry

theorem no_solution_when_m_eq_one :
  ¬ ∃ (x y : ℝ), y = 1 * x + 4 ∧ y = (3 * 1 - 2) * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_exists_no_solution_when_m_eq_one_l884_88459


namespace NUMINAMATH_CALUDE_four_equidistant_lines_l884_88460

-- Define the distance between two points
def distance (A B : ℝ × ℝ) : ℝ := sorry

-- Define a line in 2D space
def Line : Type := sorry

-- Define the distance from a point to a line
def point_to_line_distance (p : ℝ × ℝ) (l : Line) : ℝ := sorry

theorem four_equidistant_lines 
  (A B : ℝ × ℝ) 
  (h_distance : distance A B = 8) :
  ∃ (lines : Finset Line), 
    lines.card = 4 ∧ 
    (∀ l ∈ lines, point_to_line_distance A l = 3 ∧ point_to_line_distance B l = 4) ∧
    (∀ l : Line, point_to_line_distance A l = 3 ∧ point_to_line_distance B l = 4 → l ∈ lines) :=
sorry

end NUMINAMATH_CALUDE_four_equidistant_lines_l884_88460


namespace NUMINAMATH_CALUDE_batsman_average_runs_l884_88418

def average_runs (total_runs : ℕ) (num_matches : ℕ) : ℚ :=
  (total_runs : ℚ) / (num_matches : ℚ)

theorem batsman_average_runs :
  let first_20_matches := 20
  let next_10_matches := 10
  let total_matches := first_20_matches + next_10_matches
  let avg_first_20 := 40
  let avg_next_10 := 13
  let total_runs_first_20 := first_20_matches * avg_first_20
  let total_runs_next_10 := next_10_matches * avg_next_10
  let total_runs := total_runs_first_20 + total_runs_next_10
  average_runs total_runs total_matches = 31 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_runs_l884_88418


namespace NUMINAMATH_CALUDE_all_propositions_false_l884_88465

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_line_line : Line → Line → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (subset_line_plane : Line → Plane → Prop)

-- Define the theorem
theorem all_propositions_false 
  (a b : Line) (α β γ : Plane) : 
  ¬(∀ a b α, (parallel_line_plane a α ∧ parallel_line_plane b α) → parallel_line_line a b) ∧
  ¬(∀ α β γ, (perpendicular_plane α β ∧ perpendicular_plane β γ) → parallel_plane_plane α γ) ∧
  ¬(∀ a α β, (parallel_line_plane a α ∧ parallel_line_plane a β) → parallel_plane_plane α β) ∧
  ¬(∀ a b α, (parallel_line_line a b ∧ subset_line_plane b α) → parallel_line_plane a α) :=
sorry

end NUMINAMATH_CALUDE_all_propositions_false_l884_88465


namespace NUMINAMATH_CALUDE_nested_expression_sum_l884_88479

theorem nested_expression_sum : 
  4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4)))))))) = 1398100 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_sum_l884_88479


namespace NUMINAMATH_CALUDE_move_point_right_point_B_position_l884_88461

def point_on_number_line (x : ℤ) := x

theorem move_point_right (start : ℤ) (distance : ℕ) :
  point_on_number_line (start + distance) = point_on_number_line start + distance :=
by sorry

theorem point_B_position :
  let point_A := point_on_number_line (-3)
  let move_distance := 4
  let point_B := point_on_number_line (point_A + move_distance)
  point_B = 1 :=
by sorry

end NUMINAMATH_CALUDE_move_point_right_point_B_position_l884_88461


namespace NUMINAMATH_CALUDE_min_value_of_D_l884_88472

noncomputable def D (x a : ℝ) : ℝ := Real.sqrt ((x - a)^2 + (Real.exp x - 2 * Real.sqrt a)) + a + 2

theorem min_value_of_D :
  ∃ (min_D : ℝ), min_D = Real.sqrt 2 + 1 ∧
  ∀ (x a : ℝ), D x a ≥ min_D :=
sorry

end NUMINAMATH_CALUDE_min_value_of_D_l884_88472


namespace NUMINAMATH_CALUDE_tank_width_is_four_feet_l884_88482

/-- Proves that the width of a rectangular tank is 4 feet given specific conditions. -/
theorem tank_width_is_four_feet 
  (fill_rate : ℝ) 
  (length depth time_to_fill : ℝ) 
  (h_fill_rate : fill_rate = 4)
  (h_length : length = 6)
  (h_depth : depth = 3)
  (h_time_to_fill : time_to_fill = 18)
  : (fill_rate * time_to_fill) / (length * depth) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tank_width_is_four_feet_l884_88482


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_in_quarter_sector_inscribed_circle_radius_3cm_l884_88406

/-- The radius of an inscribed circle in a quarter circular sector --/
theorem inscribed_circle_radius_in_quarter_sector (R : ℝ) (h : R > 0) :
  let r := R * (Real.sqrt 2 - 1)
  r > 0 ∧ r * (1 + Real.sqrt 2) = R :=
by
  sorry

/-- The specific case where the outer radius is 3 cm --/
theorem inscribed_circle_radius_3cm :
  let r := 3 * (Real.sqrt 2 - 1)
  r > 0 ∧ r * (1 + Real.sqrt 2) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_in_quarter_sector_inscribed_circle_radius_3cm_l884_88406


namespace NUMINAMATH_CALUDE_tangent_line_and_hyperbola_l884_88475

/-- Given two functions f(x) = x + 4 and g(x) = k/x that are tangent to each other, 
    prove that k = -4 -/
theorem tangent_line_and_hyperbola (k : ℝ) :
  (∃ x : ℝ, x + 4 = k / x ∧ 
   ∀ y : ℝ, y ≠ x → (y + 4 - k / y) * (x - y) ≠ 0) → 
  k = -4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_hyperbola_l884_88475


namespace NUMINAMATH_CALUDE_fox_jeans_purchased_l884_88441

/-- Represents the problem of determining the number of Fox jeans purchased during a sale. -/
theorem fox_jeans_purchased (fox_price pony_price total_savings total_jeans pony_jeans sum_discount_rates pony_discount : ℝ) 
  (h1 : fox_price = 15)
  (h2 : pony_price = 20)
  (h3 : total_savings = 9)
  (h4 : total_jeans = 5)
  (h5 : pony_jeans = 2)
  (h6 : sum_discount_rates = 0.22)
  (h7 : pony_discount = 0.18000000000000014) :
  ∃ fox_jeans : ℝ, fox_jeans = 3 ∧ 
  fox_jeans + pony_jeans = total_jeans ∧
  fox_jeans * (fox_price * (sum_discount_rates - pony_discount)) + 
  pony_jeans * (pony_price * pony_discount) = total_savings :=
sorry

end NUMINAMATH_CALUDE_fox_jeans_purchased_l884_88441


namespace NUMINAMATH_CALUDE_distance_to_line_l884_88417

/-- The distance from a point in polar coordinates to a line in polar form -/
def distance_polar_to_line (m : ℝ) (line : ℝ → ℝ → Prop) : ℝ :=
  |m - 2|

/-- The theorem stating the distance from the point (m, π/3) to the line ρcos(θ - π/3) = 2 -/
theorem distance_to_line (m : ℝ) (h : m > 0) :
  distance_polar_to_line m (fun ρ θ ↦ ρ * Real.cos (θ - Real.pi / 3) = 2) = |m - 2| := by
  sorry

end NUMINAMATH_CALUDE_distance_to_line_l884_88417


namespace NUMINAMATH_CALUDE_more_students_than_rabbits_l884_88496

theorem more_students_than_rabbits : 
  let num_classrooms : ℕ := 5
  let students_per_classroom : ℕ := 22
  let rabbits_per_classroom : ℕ := 2
  let total_students : ℕ := num_classrooms * students_per_classroom
  let total_rabbits : ℕ := num_classrooms * rabbits_per_classroom
  total_students - total_rabbits = 100 := by
  sorry

end NUMINAMATH_CALUDE_more_students_than_rabbits_l884_88496


namespace NUMINAMATH_CALUDE_profit_loss_equality_l884_88462

/-- Given an article with cost price C, prove that if the profit at selling price S_p
    equals the loss when sold at $448, then the selling price for 30% profit is 1.30C. -/
theorem profit_loss_equality (C : ℝ) (S_p : ℝ) :
  S_p - C = C - 448 →
  ∃ (S_30 : ℝ), S_30 = 1.30 * C ∧ S_30 - C = 0.30 * C :=
by sorry

end NUMINAMATH_CALUDE_profit_loss_equality_l884_88462


namespace NUMINAMATH_CALUDE_pablo_puzzle_completion_time_l884_88464

/-- The number of days it takes Pablo to complete all his puzzles -/
def days_to_complete_puzzles (pieces_per_hour : ℕ) (max_hours_per_day : ℕ) 
  (num_puzzles_300 : ℕ) (num_puzzles_500 : ℕ) : ℕ :=
  let total_pieces := num_puzzles_300 * 300 + num_puzzles_500 * 500
  let pieces_per_day := pieces_per_hour * max_hours_per_day
  (total_pieces + pieces_per_day - 1) / pieces_per_day

/-- Theorem stating that it takes Pablo 7 days to complete all his puzzles -/
theorem pablo_puzzle_completion_time :
  days_to_complete_puzzles 100 7 8 5 = 7 := by
  sorry

#eval days_to_complete_puzzles 100 7 8 5

end NUMINAMATH_CALUDE_pablo_puzzle_completion_time_l884_88464


namespace NUMINAMATH_CALUDE_total_popsicles_l884_88458

/-- The number of grape popsicles in the freezer -/
def grape_popsicles : ℕ := 2

/-- The number of cherry popsicles in the freezer -/
def cherry_popsicles : ℕ := 13

/-- The number of banana popsicles in the freezer -/
def banana_popsicles : ℕ := 2

/-- Theorem stating the total number of popsicles in the freezer -/
theorem total_popsicles : grape_popsicles + cherry_popsicles + banana_popsicles = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_popsicles_l884_88458


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_ninety_five_satisfies_conditions_largest_integer_is_95_l884_88440

theorem largest_integer_with_remainder (n : ℕ) : n < 100 ∧ n % 7 = 4 → n ≤ 95 :=
by
  sorry

theorem ninety_five_satisfies_conditions : 95 < 100 ∧ 95 % 7 = 4 :=
by
  sorry

theorem largest_integer_is_95 : ∀ (n : ℕ), n < 100 ∧ n % 7 = 4 → n ≤ 95 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_ninety_five_satisfies_conditions_largest_integer_is_95_l884_88440


namespace NUMINAMATH_CALUDE_car_A_time_is_5_hours_l884_88445

/-- Represents the properties of a car's journey -/
structure CarJourney where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The problem setup -/
def problem : Prop :=
  ∃ (carA carB : CarJourney),
    carA.speed = 80 ∧
    carB.speed = 100 ∧
    carB.time = 2 ∧
    carA.distance = 2 * carB.distance ∧
    carA.distance = carA.speed * carA.time ∧
    carB.distance = carB.speed * carB.time

/-- The theorem to prove -/
theorem car_A_time_is_5_hours (h : problem) : 
  ∃ (carA : CarJourney), carA.time = 5 := by
  sorry


end NUMINAMATH_CALUDE_car_A_time_is_5_hours_l884_88445


namespace NUMINAMATH_CALUDE_cubic_equation_root_l884_88478

theorem cubic_equation_root (a b : ℚ) : 
  ((-2 - 3 * Real.sqrt 3) ^ 3 + a * (-2 - 3 * Real.sqrt 3) ^ 2 + b * (-2 - 3 * Real.sqrt 3) + 49 = 0) → 
  a = -3/23 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l884_88478


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l884_88454

/-- Given that point A(2, -5) is symmetric with respect to the x-axis to point (m, n), prove that m + n = 7. -/
theorem symmetric_points_sum (m n : ℝ) : 
  (2 = m ∧ -5 = -n) → m + n = 7 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l884_88454


namespace NUMINAMATH_CALUDE_missing_number_last_two_digits_l884_88436

def last_two_digits (n : ℕ) : ℕ := n % 100

def product_last_two_digits (nums : List ℕ) : ℕ :=
  last_two_digits (nums.foldl (λ acc x => last_two_digits (acc * last_two_digits x)) 1)

theorem missing_number_last_two_digits
  (h : product_last_two_digits [122, 123, 125, 129, x] = 50) :
  last_two_digits x = 1 :=
sorry

end NUMINAMATH_CALUDE_missing_number_last_two_digits_l884_88436


namespace NUMINAMATH_CALUDE_mass_percentage_H_in_C9H14N3O5_l884_88477

/-- Molar mass of carbon in g/mol -/
def molar_mass_C : ℝ := 12.01

/-- Molar mass of hydrogen in g/mol -/
def molar_mass_H : ℝ := 1.01

/-- Molar mass of nitrogen in g/mol -/
def molar_mass_N : ℝ := 14.01

/-- Molar mass of oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- Calculate the mass percentage of hydrogen in C9H14N3O5 -/
theorem mass_percentage_H_in_C9H14N3O5 :
  let total_mass := 9 * molar_mass_C + 14 * molar_mass_H + 3 * molar_mass_N + 5 * molar_mass_O
  let mass_H := 14 * molar_mass_H
  let percentage := (mass_H / total_mass) * 100
  ∃ ε > 0, |percentage - 5.79| < ε :=
sorry

end NUMINAMATH_CALUDE_mass_percentage_H_in_C9H14N3O5_l884_88477


namespace NUMINAMATH_CALUDE_f_properties_l884_88438

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.cos x)^2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3 * (Real.sin x)^2

theorem f_properties :
  (∀ x, f (x + π/12) = f (π/12 - x)) ∧
  (∀ x, f (π/3 + x) = -f (π/3 - x)) ∧
  (∃ x₁ x₂, |f x₁ - f x₂| ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l884_88438


namespace NUMINAMATH_CALUDE_ratio_fraction_equality_l884_88425

theorem ratio_fraction_equality (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ratio_fraction_equality_l884_88425


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l884_88497

/-- An ellipse with given properties -/
structure Ellipse where
  -- Point P on the ellipse
  p : ℝ × ℝ
  -- Focus F₁
  f1 : ℝ × ℝ
  -- Focus F₂
  f2 : ℝ × ℝ

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ :=
  sorry

/-- Theorem stating the eccentricity of the specific ellipse -/
theorem ellipse_eccentricity :
  let e : Ellipse := {
    p := (2, 3)
    f1 := (-2, 0)
    f2 := (2, 0)
  }
  eccentricity e = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l884_88497


namespace NUMINAMATH_CALUDE_zachary_bus_ride_length_l884_88405

theorem zachary_bus_ride_length : 
  let vince_ride : ℚ := 0.625
  let difference : ℚ := 0.125
  let zachary_ride : ℚ := vince_ride - difference
  zachary_ride = 0.500 := by sorry

end NUMINAMATH_CALUDE_zachary_bus_ride_length_l884_88405


namespace NUMINAMATH_CALUDE_x_value_proof_l884_88448

theorem x_value_proof (a b c x : ℝ) 
  (eq1 : a - b + c = 5)
  (eq2 : a^2 + b^2 + c^2 = 29)
  (eq3 : a*b + b*c + a*c = x^2) :
  x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l884_88448


namespace NUMINAMATH_CALUDE_fraction_equality_sum_l884_88476

theorem fraction_equality_sum (M N : ℚ) : 
  (4 : ℚ) / 7 = M / 63 ∧ (4 : ℚ) / 7 = 84 / N → M + N = 183 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_sum_l884_88476


namespace NUMINAMATH_CALUDE_area_relationship_l884_88416

/-- Represents a right triangle with a circumscribed circle -/
structure RightTriangleWithCircumcircle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  is_right_triangle : side1^2 + side2^2 = hypotenuse^2
  side1_positive : side1 > 0
  side2_positive : side2 > 0

/-- The areas of the non-triangular regions in the circumcircle -/
structure CircumcircleAreas where
  A : ℝ
  B : ℝ
  C : ℝ
  C_largest : C ≥ A ∧ C ≥ B

/-- Theorem stating the relationship between areas A, B, and C -/
theorem area_relationship (triangle : RightTriangleWithCircumcircle)
    (areas : CircumcircleAreas) (h : triangle.side1 = 15 ∧ triangle.side2 = 36 ∧ triangle.hypotenuse = 39) :
    areas.A + areas.B + 270 = areas.C := by
  sorry

end NUMINAMATH_CALUDE_area_relationship_l884_88416


namespace NUMINAMATH_CALUDE_inequality_proof_l884_88424

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ (3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l884_88424


namespace NUMINAMATH_CALUDE_geese_survival_theorem_l884_88412

/-- Represents the fraction of geese that did not survive the first year out of those that survived the first month -/
def fraction_not_survived_first_year (
  total_eggs : ℕ
  ) (
  hatch_rate : ℚ
  ) (
  first_month_survival_rate : ℚ
  ) (
  first_year_survivors : ℕ
  ) : ℚ :=
  1 - (first_year_survivors : ℚ) / (total_eggs * hatch_rate * first_month_survival_rate)

/-- Theorem stating that the fraction of geese that did not survive the first year is 0 -/
theorem geese_survival_theorem (
  total_eggs : ℕ
  ) (
  hatch_rate : ℚ
  ) (
  first_month_survival_rate : ℚ
  ) (
  first_year_survivors : ℕ
  ) (
  h1 : hatch_rate = 1/3
  ) (
  h2 : first_month_survival_rate = 4/5
  ) (
  h3 : first_year_survivors = 120
  ) (
  h4 : total_eggs * hatch_rate * first_month_survival_rate = first_year_survivors
  ) : fraction_not_survived_first_year total_eggs hatch_rate first_month_survival_rate first_year_survivors = 0 :=
by
  sorry

#check geese_survival_theorem

end NUMINAMATH_CALUDE_geese_survival_theorem_l884_88412


namespace NUMINAMATH_CALUDE_evaluate_expression_l884_88431

theorem evaluate_expression : (4 + 6 + 7) / 3 - 2 / 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l884_88431


namespace NUMINAMATH_CALUDE_total_animals_is_100_l884_88428

/-- The number of rabbits -/
def num_rabbits : ℕ := 4

/-- The number of ducks -/
def num_ducks : ℕ := num_rabbits + 12

/-- The number of chickens -/
def num_chickens : ℕ := 5 * num_ducks

/-- The total number of animals -/
def total_animals : ℕ := num_chickens + num_ducks + num_rabbits

theorem total_animals_is_100 : total_animals = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_is_100_l884_88428


namespace NUMINAMATH_CALUDE_solution_inequality1_no_solution_system_l884_88427

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := (2*x - 2)/3 ≤ 2 - (2*x + 2)/2

def inequality2 (x : ℝ) : Prop := 3*(x - 2) - 1 ≥ -4 - 2*(x - 2)

def inequality3 (x : ℝ) : Prop := (1/3)*(1 - 2*x) > (3*(2*x - 1))/2

-- Theorem for the first inequality
theorem solution_inequality1 : 
  ∀ x : ℝ, inequality1 x ↔ x ≤ 1 := by sorry

-- Theorem for the system of inequalities
theorem no_solution_system : 
  ¬∃ x : ℝ, inequality2 x ∧ inequality3 x := by sorry

end NUMINAMATH_CALUDE_solution_inequality1_no_solution_system_l884_88427


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l884_88468

theorem subtraction_of_decimals : 3.75 - 1.46 = 2.29 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l884_88468


namespace NUMINAMATH_CALUDE_simplify_expression_find_expression_value_evaluate_expression_l884_88495

-- Part 1
theorem simplify_expression (a b : ℝ) :
  10 * (a - b)^4 - 25 * (a - b)^4 + 5 * (a - b)^4 = -10 * (a - b)^4 := by sorry

-- Part 2
theorem find_expression_value (x y : ℝ) (h : 2 * x^2 - 3 * y = 8) :
  4 * x^2 - 6 * y - 32 = -16 := by sorry

-- Part 3
theorem evaluate_expression (a b : ℝ) 
  (h1 : a^2 + 2 * a * b = -5) (h2 : a * b - 2 * b^2 = -3) :
  3 * a^2 + 4 * a * b + 4 * b^2 = -9 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_find_expression_value_evaluate_expression_l884_88495


namespace NUMINAMATH_CALUDE_find_a9_l884_88487

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a9 (a : ℕ → ℝ) (h1 : arithmetic_sequence a) 
  (h2 : a 3 = 5) (h3 : a 4 + a 8 = 22) : 
  ∃ x : ℝ, a 9 = x :=
sorry

end NUMINAMATH_CALUDE_find_a9_l884_88487


namespace NUMINAMATH_CALUDE_books_together_l884_88484

/-- The number of books Keith and Jason have together -/
def total_books (keith_books jason_books : ℕ) : ℕ :=
  keith_books + jason_books

/-- Theorem: Keith and Jason have 41 books together -/
theorem books_together : total_books 20 21 = 41 := by
  sorry

end NUMINAMATH_CALUDE_books_together_l884_88484


namespace NUMINAMATH_CALUDE_rectangular_block_height_l884_88480

/-- The height of a rectangular block with given volume and base area -/
theorem rectangular_block_height (volume : ℝ) (base_area : ℝ) (height : ℝ) : 
  volume = 120 → base_area = 24 → volume = base_area * height → height = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_block_height_l884_88480


namespace NUMINAMATH_CALUDE_sphere_area_and_volume_l884_88432

theorem sphere_area_and_volume (d : ℝ) (h : d = 6) : 
  let r := d / 2
  (4 * Real.pi * r^2 = 36 * Real.pi) ∧ 
  ((4 / 3) * Real.pi * r^3 = 36 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_sphere_area_and_volume_l884_88432


namespace NUMINAMATH_CALUDE_tangent_curve_relation_l884_88442

noncomputable def tangent_line (a : ℝ) (x : ℝ) : ℝ := x + a

noncomputable def curve (b : ℝ) (x : ℝ) : ℝ := Real.exp (x - 1) - b + 1

theorem tangent_curve_relation (a b : ℝ) :
  (∃ x₀ : ℝ, tangent_line a x₀ = curve b x₀ ∧ 
    (deriv (tangent_line a)) x₀ = (deriv (curve b)) x₀) →
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_curve_relation_l884_88442


namespace NUMINAMATH_CALUDE_divisor_congruence_l884_88466

theorem divisor_congruence (d : ℕ) (x y : ℤ) : 
  d > 0 ∧ d ∣ (5 + 1998^1998) →
  (d = 2*x^2 + 2*x*y + 3*y^2 ↔ d % 20 = 3 ∨ d % 20 = 7) :=
by sorry

end NUMINAMATH_CALUDE_divisor_congruence_l884_88466


namespace NUMINAMATH_CALUDE_maria_bike_purchase_l884_88429

/-- The amount Maria needs to earn to buy a bike -/
def amount_to_earn (retail_price savings mother_contribution : ℕ) : ℕ :=
  retail_price - (savings + mother_contribution)

/-- Theorem: Maria needs to earn $230 to buy the bike -/
theorem maria_bike_purchase (retail_price savings mother_contribution : ℕ)
  (h1 : retail_price = 600)
  (h2 : savings = 120)
  (h3 : mother_contribution = 250) :
  amount_to_earn retail_price savings mother_contribution = 230 := by
  sorry

end NUMINAMATH_CALUDE_maria_bike_purchase_l884_88429


namespace NUMINAMATH_CALUDE_shelf_fill_relation_l884_88492

/-- Represents the number of books needed to fill a shelf. -/
structure ShelfFill :=
  (A H S M F : ℕ)
  (distinct : A ≠ H ∧ A ≠ S ∧ A ≠ M ∧ A ≠ F ∧
              H ≠ S ∧ H ≠ M ∧ H ≠ F ∧
              S ≠ M ∧ S ≠ F ∧
              M ≠ F)
  (positive : A > 0 ∧ H > 0 ∧ S > 0 ∧ M > 0 ∧ F > 0)
  (history_thicker : H < A ∧ M < S)

/-- Theorem stating the relation between the number of books needed to fill the shelf. -/
theorem shelf_fill_relation (sf : ShelfFill) : sf.F = (sf.A * sf.F - sf.S * sf.H) / (sf.M - sf.H) :=
  sorry

end NUMINAMATH_CALUDE_shelf_fill_relation_l884_88492


namespace NUMINAMATH_CALUDE_corresponding_angles_equal_l884_88451

-- Define the concept of a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to get the angles of a triangle
def angles (t : Triangle) : ℝ × ℝ × ℝ :=
  sorry

-- Define the property that corresponding angles are either equal or sum to 180°
def corresponding_angles_property (t1 t2 : Triangle) : Prop :=
  let (α1, β1, γ1) := angles t1
  let (α2, β2, γ2) := angles t2
  (α1 = α2 ∨ α1 + α2 = 180) ∧
  (β1 = β2 ∨ β1 + β2 = 180) ∧
  (γ1 = γ2 ∨ γ1 + γ2 = 180)

-- Theorem statement
theorem corresponding_angles_equal (t1 t2 : Triangle) 
  (h : corresponding_angles_property t1 t2) : 
  angles t1 = angles t2 :=
sorry

end NUMINAMATH_CALUDE_corresponding_angles_equal_l884_88451


namespace NUMINAMATH_CALUDE_circle_condition_l884_88473

theorem circle_condition (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*y + 2*a - 1 = 0 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 + 2*y' + 2*a - 1 = 0 → (x - x')^2 + (y - y')^2 = (x' - x)^2 + (y' - y)^2) 
  → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l884_88473


namespace NUMINAMATH_CALUDE_snow_depth_theorem_l884_88400

/-- Calculates the final snow depth after seven days given initial conditions and daily changes --/
def snow_depth_after_seven_days (initial_snow : Real) 
  (day2_snow : Real) (day2_compaction : Real)
  (daily_melt : Real) (day4_cleared : Real)
  (day5_multiplier : Real)
  (day6_melt : Real) (day6_accumulate : Real) : Real :=
  let day1 := initial_snow
  let day2 := day1 + day2_snow * (1 - day2_compaction)
  let day3 := day2 - daily_melt
  let day4 := day3 - daily_melt - day4_cleared
  let day5 := day4 - daily_melt + day5_multiplier * (day1 + day2_snow)
  let day6 := day5 - day6_melt + day6_accumulate
  day6

/-- The final snow depth after seven days is approximately 2.1667 feet --/
theorem snow_depth_theorem : 
  ∃ ε > 0, |snow_depth_after_seven_days 0.5 (8/12) 0.1 (1/12) (6/12) 1.5 (3/12) (4/12) - 2.1667| < ε :=
by sorry

end NUMINAMATH_CALUDE_snow_depth_theorem_l884_88400


namespace NUMINAMATH_CALUDE_school_population_l884_88413

theorem school_population (total_students : ℕ) : 
  (128 : ℕ) = (total_students / 2) →
  total_students = 256 := by
sorry

end NUMINAMATH_CALUDE_school_population_l884_88413


namespace NUMINAMATH_CALUDE_unique_student_count_l884_88421

theorem unique_student_count :
  ∃! n : ℕ, n < 400 ∧ n % 17 = 15 ∧ n % 19 = 10 ∧ n = 219 :=
by sorry

end NUMINAMATH_CALUDE_unique_student_count_l884_88421


namespace NUMINAMATH_CALUDE_line_passes_through_point_l884_88463

/-- Given a line y = 2x + b passing through the point (1, 2), prove that b = 0 -/
theorem line_passes_through_point (b : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + b) → 2 = 2 * 1 + b → b = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l884_88463
