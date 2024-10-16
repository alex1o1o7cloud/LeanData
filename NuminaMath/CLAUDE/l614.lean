import Mathlib

namespace NUMINAMATH_CALUDE_root_conditions_imply_m_range_l614_61452

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 3

-- State the theorem
theorem root_conditions_imply_m_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f m x = 0 ∧ f m y = 0 ∧ x > 1 ∧ y < 1) →
  m > 4 :=
sorry

end NUMINAMATH_CALUDE_root_conditions_imply_m_range_l614_61452


namespace NUMINAMATH_CALUDE_tile_side_length_proof_l614_61447

/-- Represents a rectangular room with length and width in centimeters -/
structure Room where
  length : ℕ
  width : ℕ

/-- Represents a square tile with side length in centimeters -/
structure Tile where
  side_length : ℕ

/-- Calculates the area of a room in square centimeters -/
def room_area (r : Room) : ℕ := r.length * r.width

/-- Calculates the area of a tile in square centimeters -/
def tile_area (t : Tile) : ℕ := t.side_length * t.side_length

theorem tile_side_length_proof (r : Room) (num_tiles : ℕ) (h1 : r.length = 5000) (h2 : r.width = 1125) (h3 : num_tiles = 9000) :
  ∃ t : Tile, tile_area t * num_tiles = room_area r ∧ t.side_length = 25 := by
  sorry

end NUMINAMATH_CALUDE_tile_side_length_proof_l614_61447


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_squares_l614_61400

theorem arithmetic_geometric_mean_squares (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 20)
  (h_geometric : Real.sqrt (a * b) = 10) : 
  a^2 + b^2 = 1400 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_squares_l614_61400


namespace NUMINAMATH_CALUDE_john_average_score_l614_61494

def john_scores : List ℝ := [88, 95, 90, 84, 91]

theorem john_average_score : (john_scores.sum / john_scores.length) = 89.6 := by
  sorry

end NUMINAMATH_CALUDE_john_average_score_l614_61494


namespace NUMINAMATH_CALUDE_probability_at_least_one_female_l614_61449

/-- The probability of selecting at least one female student when choosing 2 people
    from a group of 3 male and 2 female students is 0.7 -/
theorem probability_at_least_one_female (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (selected : ℕ) (h1 : total_students = male_students + female_students)
  (h2 : total_students = 5) (h3 : male_students = 3) (h4 : female_students = 2) (h5 : selected = 2) :
  (Nat.choose total_students selected - Nat.choose male_students selected : ℚ) /
  Nat.choose total_students selected = 7/10 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_female_l614_61449


namespace NUMINAMATH_CALUDE_polygon_labeling_exists_l614_61468

/-- A labeling of a polygon is a function that assigns a unique label to each vertex and midpoint -/
def Labeling (n : ℕ) := Fin (4*n+2) → Fin (4*n+2)

/-- The sum of labels for a side is the sum of the labels of its two vertices and midpoint -/
def sideSum (f : Labeling n) (i : Fin (2*n+1)) : ℕ :=
  f i + f (i+1) + f (i+2*n+1)

theorem polygon_labeling_exists (n : ℕ) :
  ∃ (f : Labeling n), Function.Injective f ∧
    ∀ (i j : Fin (2*n+1)), sideSum f i = sideSum f j :=
sorry

end NUMINAMATH_CALUDE_polygon_labeling_exists_l614_61468


namespace NUMINAMATH_CALUDE_twentiethTerm_eq_97_l614_61488

/-- The nth term of an arithmetic sequence -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 20th term of the specific arithmetic sequence -/
def twentiethTerm : ℝ :=
  arithmeticSequence 2 5 20

theorem twentiethTerm_eq_97 : twentiethTerm = 97 := by
  sorry

end NUMINAMATH_CALUDE_twentiethTerm_eq_97_l614_61488


namespace NUMINAMATH_CALUDE_choir_problem_l614_61444

/-- A choir problem involving singers joining in different verses -/
theorem choir_problem (total_singers : ℕ) (first_verse_singers : ℕ) 
  (second_verse_joiners : ℕ) (third_verse_joiners : ℕ) : 
  total_singers = 30 →
  first_verse_singers = total_singers / 2 →
  third_verse_joiners = 10 →
  first_verse_singers + second_verse_joiners + third_verse_joiners = total_singers →
  (second_verse_joiners : ℚ) / (total_singers - first_verse_singers : ℚ) = 1 / 3 := by
  sorry

#check choir_problem

end NUMINAMATH_CALUDE_choir_problem_l614_61444


namespace NUMINAMATH_CALUDE_inequality_transformation_l614_61410

theorem inequality_transformation (a : ℝ) : 
  (∀ x : ℝ, a * x > 2 ↔ x < 2 / a) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_transformation_l614_61410


namespace NUMINAMATH_CALUDE_distance_from_circle_center_to_line_l614_61432

/-- The distance from the center of a circle to a line --/
theorem distance_from_circle_center_to_line :
  let circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 = 0}
  let center : ℝ × ℝ := (2, 0)
  let line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 = 0}
  center ∈ circle →
  (∀ p ∈ circle, (p.1 - 2)^2 + p.2^2 = 4) →
  (∀ p ∈ line, p.1 = p.2) →
  ∃ d : ℝ, d = Real.sqrt 2 ∧ ∀ p ∈ line, (center.1 - p.1)^2 + (center.2 - p.2)^2 = d^2 :=
by sorry

end NUMINAMATH_CALUDE_distance_from_circle_center_to_line_l614_61432


namespace NUMINAMATH_CALUDE_inequality_statements_l614_61474

theorem inequality_statements :
  (∃ a b : ℝ, a > b ∧ a^2 ≤ b^2) ∧
  (∀ a b : ℝ, a > b ↔ a^3 > b^3) ∧
  (∃ a b : ℝ, a > b ∧ |a| ≤ |b|) ∧
  (∃ a b c : ℝ, a * c^2 ≤ b * c^2 ∧ a ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_statements_l614_61474


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_real_l614_61496

theorem arithmetic_geometric_mean_ratio_real (A B : ℂ) :
  (∃ r : ℝ, (A + B) / 2 = r * (A * B)^(1/2 : ℂ)) →
  (∃ r : ℝ, A = r * B) ∨ Complex.abs A = Complex.abs B :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_real_l614_61496


namespace NUMINAMATH_CALUDE_spinner_prime_sum_probability_l614_61485

/-- Represents a spinner with numbered sectors -/
structure Spinner :=
  (sectors : List Nat)

/-- Checks if a number is prime -/
def isPrime (n : Nat) : Bool :=
  n > 1 && (List.range (n - 1)).all (fun d => d <= 1 || n % (d + 2) ≠ 0)

/-- Calculates all possible sums from two spinners -/
def allSums (s1 s2 : Spinner) : List Nat :=
  List.join (s1.sectors.map (fun x => s2.sectors.map (fun y => x + y)))

/-- Counts the number of prime sums -/
def countPrimeSums (sums : List Nat) : Nat :=
  sums.filter isPrime |>.length

theorem spinner_prime_sum_probability :
  let spinner1 : Spinner := ⟨[1, 2, 3]⟩
  let spinner2 : Spinner := ⟨[3, 4, 5]⟩
  let allPossibleSums := allSums spinner1 spinner2
  let totalSums := allPossibleSums.length
  let primeSums := countPrimeSums allPossibleSums
  (primeSums : Rat) / totalSums = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_spinner_prime_sum_probability_l614_61485


namespace NUMINAMATH_CALUDE_sequence_bound_l614_61426

def sequence_rule (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, ∃ d : ℕ, d < 10 ∧ 
    (a (2*n) = a (2*n - 1) - d) ∧
    (a (2*n + 1) = a (2*n) + d))

theorem sequence_bound (a : ℕ → ℕ) (h : sequence_rule a) : 
  ∀ n : ℕ, a n ≤ 10 * a 1 :=
sorry

end NUMINAMATH_CALUDE_sequence_bound_l614_61426


namespace NUMINAMATH_CALUDE_jed_speeding_fine_l614_61499

/-- Calculates the speeding fine in Zeoland -/
def speeding_fine (speed_limit : ℕ) (actual_speed : ℕ) (fine_per_mph : ℕ) : ℕ :=
  if actual_speed > speed_limit
  then (actual_speed - speed_limit) * fine_per_mph
  else 0

/-- Proves that Jed's speeding fine is $256 -/
theorem jed_speeding_fine :
  let speed_limit : ℕ := 50
  let actual_speed : ℕ := 66
  let fine_per_mph : ℕ := 16
  speeding_fine speed_limit actual_speed fine_per_mph = 256 :=
by
  sorry

end NUMINAMATH_CALUDE_jed_speeding_fine_l614_61499


namespace NUMINAMATH_CALUDE_arithmetic_progression_includes_1999_l614_61411

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def IsArithmeticProgression (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_progression_includes_1999
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_positive : d > 0)
  (h_arithmetic : IsArithmeticProgression a d)
  (h_7 : ∃ n, a n = 7)
  (h_15 : ∃ n, a n = 15)
  (h_27 : ∃ n, a n = 27) :
  ∃ n, a n = 1999 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_includes_1999_l614_61411


namespace NUMINAMATH_CALUDE_integer_triple_problem_l614_61470

theorem integer_triple_problem (a b c : ℤ) :
  let N := ((a - b) * (b - c) * (c - a)) / 2 + 2
  (∃ m : ℕ, N = 1729^m ∧ N > 0) →
  ∃ k : ℤ, a = k + 2 ∧ b = k + 1 ∧ c = k :=
by sorry

end NUMINAMATH_CALUDE_integer_triple_problem_l614_61470


namespace NUMINAMATH_CALUDE_min_marked_cells_l614_61405

/-- Represents a board with dimensions m × n -/
structure Board (m n : ℕ) where
  cells : Fin m → Fin n → Bool

/-- Represents an L-shaped piece -/
inductive LPiece
  | mk : Fin 2 → Fin 2 → LPiece

/-- Checks if an L-piece touches a marked cell on the board -/
def touchesMarkedCell (b : Board m n) (l : LPiece) (i j : ℕ) : Prop :=
  ∃ (x y : Fin 2), b.cells ⟨i + x.val, sorry⟩ ⟨j + y.val, sorry⟩ = true

/-- A marking strategy for the board -/
def markingStrategy (b : Board m n) : Prop :=
  ∀ (i : Fin m) (j : Fin n), i.val % 2 = 0 → b.cells i j = true

/-- The main theorem stating that 50 is the smallest number of marked cells
    that ensures any L-shaped piece touches at least one marked cell on a 10 × 11 board -/
theorem min_marked_cells :
  ∀ (b : Board 10 11),
    (∃ (k : ℕ), k < 50 ∧
      (∀ (l : LPiece) (i j : ℕ), i < 9 ∧ j < 10 →
        touchesMarkedCell b l i j)) →
    (∃ (b' : Board 10 11),
      markingStrategy b' ∧
      (∀ (l : LPiece) (i j : ℕ), i < 9 ∧ j < 10 →
        touchesMarkedCell b' l i j) ∧
      (∃ (k : ℕ), k = 50 ∧
        k = (Finset.filter (fun i => b'.cells i.1 i.2) (Finset.product (Finset.range 10) (Finset.range 11))).card)) :=
by
  sorry


end NUMINAMATH_CALUDE_min_marked_cells_l614_61405


namespace NUMINAMATH_CALUDE_inequality_equivalence_l614_61419

theorem inequality_equivalence (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x / y + 1 / x + y ≥ y / x + 1 / y + x ↔ (x - y) * (x - 1) * (1 - y) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l614_61419


namespace NUMINAMATH_CALUDE_slope_of_line_l614_61462

theorem slope_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → (y - 4) / (-x) = -4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l614_61462


namespace NUMINAMATH_CALUDE_not_both_rising_left_l614_61478

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 2 * x^2
def parabola2 (x : ℝ) : ℝ := -2 * x^2

-- Define what it means for a function to be rising on an interval
def is_rising (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Theorem stating that it's not true that both parabolas are rising on the left side of the y-axis
theorem not_both_rising_left : ¬(∃ (a : ℝ), a < 0 ∧ 
  is_rising parabola1 a 0 ∧ is_rising parabola2 a 0) :=
sorry

end NUMINAMATH_CALUDE_not_both_rising_left_l614_61478


namespace NUMINAMATH_CALUDE_extremal_points_sum_bound_l614_61466

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / (a * x^2 + 1)

theorem extremal_points_sum_bound {a : ℝ} (ha : a > 0) 
  (x₁ x₂ : ℝ) (h_extremal : ∀ x, x ≠ x₁ → x ≠ x₂ → (deriv (f a)) x ≠ 0) :
  f a x₁ + f a x₂ < Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_extremal_points_sum_bound_l614_61466


namespace NUMINAMATH_CALUDE_log_difference_theorem_l614_61422

noncomputable def logBase (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

def satisfies_condition (a : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1 ∧
  (∀ x ∈ Set.Icc 1 3, logBase a x ≤ logBase a 3) ∧
  (∀ x ∈ Set.Icc 1 3, logBase a x ≥ logBase a 1) ∧
  logBase a 3 - logBase a 1 = 2

theorem log_difference_theorem :
  {a : ℝ | satisfies_condition a} = {Real.sqrt 3, Real.sqrt 3 / 3} :=
sorry

end NUMINAMATH_CALUDE_log_difference_theorem_l614_61422


namespace NUMINAMATH_CALUDE_colbert_treehouse_ratio_l614_61401

/-- Proves that the ratio of planks from Colbert's parents to the total number of planks is 1:2 -/
theorem colbert_treehouse_ratio :
  let total_planks : ℕ := 200
  let storage_planks : ℕ := total_planks / 4
  let friends_planks : ℕ := 20
  let store_planks : ℕ := 30
  let parents_planks : ℕ := total_planks - (storage_planks + friends_planks + store_planks)
  (parents_planks : ℚ) / total_planks = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_colbert_treehouse_ratio_l614_61401


namespace NUMINAMATH_CALUDE_zebra_stripes_l614_61498

theorem zebra_stripes (w n b : ℕ) : 
  w + n = b + 1 →  -- Total black stripes = white stripes + 1
  b = w + 7 →      -- White stripes = wide black stripes + 7
  n = 8            -- Number of narrow black stripes is 8
:= by sorry

end NUMINAMATH_CALUDE_zebra_stripes_l614_61498


namespace NUMINAMATH_CALUDE_monotonicity_indeterminate_l614_61457

-- Define the concept of an increasing function on an open interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- Define the theorem
theorem monotonicity_indeterminate
  (f : ℝ → ℝ) (a b c : ℝ) 
  (hab : a < b) (hbc : b < c)
  (h1 : IncreasingOn f a b)
  (h2 : IncreasingOn f b c) :
  ¬ (IncreasingOn f a c ∨ (∀ x y, a < x ∧ x < y ∧ y < c → f x > f y)) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_indeterminate_l614_61457


namespace NUMINAMATH_CALUDE_rectangle_area_l614_61437

theorem rectangle_area (width length : ℝ) : 
  length = 4 * width →
  2 * length + 2 * width = 200 →
  width * length = 1600 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l614_61437


namespace NUMINAMATH_CALUDE_no_distributive_laws_hold_l614_61415

-- Define the # operation
def hash (a b : ℝ) : ℝ := a + 2*b

-- Theorem stating that none of the distributive laws hold
theorem no_distributive_laws_hold :
  ¬(∀ (x y z : ℝ), hash x (y + z) = hash x y + hash x z) ∧
  ¬(∀ (x y z : ℝ), x + hash y z = hash (x + y) (x + z)) ∧
  ¬(∀ (x y z : ℝ), hash x (hash y z) = hash (hash x y) (hash x z)) :=
by sorry

end NUMINAMATH_CALUDE_no_distributive_laws_hold_l614_61415


namespace NUMINAMATH_CALUDE_time_to_earn_house_cost_l614_61445

/-- Represents the financial situation of a man buying a house -/
structure HouseBuying where
  /-- The cost of the house -/
  houseCost : ℝ
  /-- Annual household expenses -/
  annualExpenses : ℝ
  /-- Annual savings -/
  annualSavings : ℝ
  /-- The man spends the same on expenses in 8 years as on savings in 12 years -/
  expensesSavingsRelation : 8 * annualExpenses = 12 * annualSavings
  /-- It takes 24 years to buy the house with all earnings -/
  buyingTime : houseCost = 24 * (annualExpenses + annualSavings)

/-- Theorem stating the time needed to earn the house cost -/
theorem time_to_earn_house_cost (hb : HouseBuying) :
  hb.houseCost / hb.annualSavings = 60 := by
  sorry

end NUMINAMATH_CALUDE_time_to_earn_house_cost_l614_61445


namespace NUMINAMATH_CALUDE_store_a_cheaper_condition_store_b_cheaper_condition_store_a_cheaper_at_100_most_cost_effective_plan_is_best_l614_61431

/-- Represents the cost of purchasing from Store A or B -/
def store_cost (x : ℝ) (is_store_a : Bool) : ℝ :=
  if is_store_a then 20 * x + 2400 else 18 * x + 2700

/-- Theorem stating the conditions under which Store A is cheaper than Store B -/
theorem store_a_cheaper_condition (x : ℝ) (h1 : x > 30) :
  store_cost x true < store_cost x false ↔ x < 150 :=
sorry

/-- Theorem stating the conditions under which Store B is cheaper than Store A -/
theorem store_b_cheaper_condition (x : ℝ) (h1 : x > 30) :
  store_cost x false < store_cost x true ↔ x > 150 :=
sorry

/-- Theorem proving that for x = 100, Store A is cheaper -/
theorem store_a_cheaper_at_100 :
  store_cost 100 true < store_cost 100 false :=
sorry

/-- Definition of the cost for the most cost-effective plan when x = 100 -/
def most_cost_effective_plan : ℝ := 3000 + 20 * 70 * 0.9

/-- Theorem proving that the most cost-effective plan is cheaper than both Store A and B when x = 100 -/
theorem most_cost_effective_plan_is_best :
  most_cost_effective_plan < store_cost 100 true ∧
  most_cost_effective_plan < store_cost 100 false :=
sorry

end NUMINAMATH_CALUDE_store_a_cheaper_condition_store_b_cheaper_condition_store_a_cheaper_at_100_most_cost_effective_plan_is_best_l614_61431


namespace NUMINAMATH_CALUDE_inequality_bound_l614_61455

theorem inequality_bound (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  Real.sqrt (a / (b + c + d)) + Real.sqrt (b / (a + c + d)) + 
  Real.sqrt (c / (a + b + d)) + Real.sqrt (d / (a + b + c)) < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_bound_l614_61455


namespace NUMINAMATH_CALUDE_coefficient_x4y_value_l614_61418

/-- The coefficient of x^4y in the expansion of (x^2 + y + 3)^6 -/
def coefficient_x4y (x y : ℕ) : ℕ :=
  (Nat.choose 6 4) * (Nat.choose 4 3) * (3^3)

/-- Theorem stating that the coefficient of x^4y in (x^2 + y + 3)^6 is 1620 -/
theorem coefficient_x4y_value :
  ∀ x y, coefficient_x4y x y = 1620 := by
  sorry

#eval coefficient_x4y 0 0  -- To check the result

end NUMINAMATH_CALUDE_coefficient_x4y_value_l614_61418


namespace NUMINAMATH_CALUDE_roberto_skipping_rate_l614_61439

/-- Roberto's skipping rate problem -/
theorem roberto_skipping_rate 
  (valerie_rate : ℕ) 
  (total_skips : ℕ) 
  (duration : ℕ) 
  (h1 : valerie_rate = 80)
  (h2 : total_skips = 2250)
  (h3 : duration = 15) :
  ∃ (roberto_hourly_rate : ℕ), 
    roberto_hourly_rate = 4200 ∧ 
    roberto_hourly_rate * duration = (total_skips - valerie_rate * duration) * 4 :=
sorry

end NUMINAMATH_CALUDE_roberto_skipping_rate_l614_61439


namespace NUMINAMATH_CALUDE_workshop_sampling_theorem_l614_61438

-- Define the total number of workers in each group
def total_workers_A : ℕ := 10
def total_workers_B : ℕ := 10

-- Define the number of female workers in each group
def female_workers_A : ℕ := 4
def female_workers_B : ℕ := 6

-- Define the total number of workers selected for assessment
def total_selected : ℕ := 4

-- Define the number of workers selected from each group
def selected_from_A : ℕ := 2
def selected_from_B : ℕ := 2

-- Define the probability of selecting exactly 1 female worker from Group A
def prob_one_female_A : ℚ := (Nat.choose female_workers_A 1 * Nat.choose (total_workers_A - female_workers_A) (selected_from_A - 1)) / Nat.choose total_workers_A selected_from_A

-- Define the probability of selecting exactly 2 male workers from both groups
def prob_two_males : ℚ :=
  (Nat.choose (total_workers_A - female_workers_A) 0 * Nat.choose female_workers_A 2 *
   Nat.choose (total_workers_B - female_workers_B) 2 * Nat.choose female_workers_B 0 +
   Nat.choose (total_workers_A - female_workers_A) 1 * Nat.choose female_workers_A 1 *
   Nat.choose (total_workers_B - female_workers_B) 1 * Nat.choose female_workers_B 1 +
   Nat.choose (total_workers_A - female_workers_A) 2 * Nat.choose female_workers_A 0 *
   Nat.choose (total_workers_B - female_workers_B) 0 * Nat.choose female_workers_B 2) /
  (Nat.choose total_workers_A selected_from_A * Nat.choose total_workers_B selected_from_B)

theorem workshop_sampling_theorem :
  (selected_from_A + selected_from_B = total_selected) ∧
  (prob_one_female_A = (Nat.choose female_workers_A 1 * Nat.choose (total_workers_A - female_workers_A) (selected_from_A - 1)) / Nat.choose total_workers_A selected_from_A) ∧
  (prob_two_males = (Nat.choose (total_workers_A - female_workers_A) 0 * Nat.choose female_workers_A 2 *
                     Nat.choose (total_workers_B - female_workers_B) 2 * Nat.choose female_workers_B 0 +
                     Nat.choose (total_workers_A - female_workers_A) 1 * Nat.choose female_workers_A 1 *
                     Nat.choose (total_workers_B - female_workers_B) 1 * Nat.choose female_workers_B 1 +
                     Nat.choose (total_workers_A - female_workers_A) 2 * Nat.choose female_workers_A 0 *
                     Nat.choose (total_workers_B - female_workers_B) 0 * Nat.choose female_workers_B 2) /
                    (Nat.choose total_workers_A selected_from_A * Nat.choose total_workers_B selected_from_B)) := by
  sorry


end NUMINAMATH_CALUDE_workshop_sampling_theorem_l614_61438


namespace NUMINAMATH_CALUDE_fraction_equality_l614_61408

theorem fraction_equality (a b : ℝ) (h : 2/a - 1/b = 1/(a + 2*b)) :
  4/a^2 - 1/b^2 = 1/(a*b) := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l614_61408


namespace NUMINAMATH_CALUDE_min_sum_squared_distances_l614_61403

/-- Given five collinear points A, B, C, D, and E in that order, with specified distances between them,
    this function calculates the sum of squared distances from these points to any point P on the line. -/
def sum_of_squared_distances (x : ℝ) : ℝ :=
  x^2 + (x - 3)^2 + (x - 4)^2 + (x - 9)^2 + (x - 13)^2

/-- The theorem states that the minimum value of the sum of squared distances
    from five collinear points to any point on their line is 170.24,
    given specific distances between the points. -/
theorem min_sum_squared_distances :
  ∃ (min : ℝ), min = 170.24 ∧
  ∀ (x : ℝ), sum_of_squared_distances x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squared_distances_l614_61403


namespace NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_l614_61469

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

-- Define the hyperbola equation
def hyperbola (x y m : ℝ) : Prop := x^2 - m*(y+1)^2 = 1

-- Define the tangency condition
def are_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, ellipse x y ∧ hyperbola x y m ∧
  ∀ x' y' : ℝ, ellipse x' y' ∧ hyperbola x' y' m → (x', y') = (x, y)

-- State the theorem
theorem tangent_ellipse_hyperbola :
  ∀ m : ℝ, are_tangent m → m = 72 :=
by sorry

end NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_l614_61469


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l614_61414

theorem arithmetic_calculations : 
  ((1 : Int) * (-11) + 8 + (-14) = -17) ∧ 
  (13 - (-12) + (-21) = 4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l614_61414


namespace NUMINAMATH_CALUDE_vector_angle_cosine_l614_61458

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_angle_cosine (a b c : V) :
  a + b + c = 0 →
  ‖a‖ = 2 →
  ‖b‖ = 3 →
  ‖c‖ = 4 →
  ‖a‖ < ‖b‖ →
  inner a b / (‖a‖ * ‖b‖) = 1/4 := by sorry

end NUMINAMATH_CALUDE_vector_angle_cosine_l614_61458


namespace NUMINAMATH_CALUDE_gcd_2146_1813_l614_61412

theorem gcd_2146_1813 : Nat.gcd 2146 1813 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2146_1813_l614_61412


namespace NUMINAMATH_CALUDE_geometry_class_size_l614_61454

theorem geometry_class_size :
  ∀ (total_students : ℕ),
  (total_students / 2 : ℚ) = (total_students : ℚ) / 2 →
  ((total_students / 2) / 5 : ℚ) = (total_students : ℚ) / 10 →
  (total_students : ℚ) / 10 = 10 →
  total_students = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_geometry_class_size_l614_61454


namespace NUMINAMATH_CALUDE_prob_B_given_A_l614_61477

/-- Represents the number of male students -/
def num_male : ℕ := 3

/-- Represents the number of female students -/
def num_female : ℕ := 2

/-- Represents the total number of students -/
def total_students : ℕ := num_male + num_female

/-- Event A: drawing two students of the same gender -/
def prob_A : ℚ := (num_male.choose 2 + num_female.choose 2) / total_students.choose 2

/-- Event B: drawing two female students -/
def prob_B : ℚ := num_female.choose 2 / total_students.choose 2

/-- Event AB: drawing two female students given that two students of the same gender were drawn -/
def prob_AB : ℚ := num_female.choose 2 / total_students.choose 2

/-- Theorem stating that the probability of drawing two female students given that 
    two students of the same gender were drawn is 1/4 -/
theorem prob_B_given_A : prob_AB / prob_A = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_prob_B_given_A_l614_61477


namespace NUMINAMATH_CALUDE_trig_identity_l614_61413

theorem trig_identity (α : Real) (h : Real.tan (π + α) = 2) :
  4 * Real.sin α * Real.cos α + 3 * (Real.cos α)^2 = 11/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l614_61413


namespace NUMINAMATH_CALUDE_total_wheels_is_25_l614_61406

/-- The number of wheels in Zoe's garage --/
def total_wheels : ℕ :=
  let num_bicycles : ℕ := 3
  let num_tricycles : ℕ := 4
  let num_unicycles : ℕ := 7
  let wheels_per_bicycle : ℕ := 2
  let wheels_per_tricycle : ℕ := 3
  let wheels_per_unicycle : ℕ := 1
  num_bicycles * wheels_per_bicycle +
  num_tricycles * wheels_per_tricycle +
  num_unicycles * wheels_per_unicycle

theorem total_wheels_is_25 : total_wheels = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_is_25_l614_61406


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l614_61456

/-- An arithmetic sequence. -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h : ∀ n, a (n + 1) = a n + d

/-- The problem statement. -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 1 * seq.a 5 = 9)
  (h2 : seq.a 2 = 3) :
  seq.a 4 = 3 ∨ seq.a 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l614_61456


namespace NUMINAMATH_CALUDE_jade_cal_difference_l614_61483

-- Define the number of transactions for each person
def mabel_transactions : ℕ := 90
def anthony_transactions : ℕ := mabel_transactions + (mabel_transactions / 10)
def cal_transactions : ℕ := (2 * anthony_transactions) / 3
def jade_transactions : ℕ := 83

-- Theorem to prove
theorem jade_cal_difference : jade_transactions - cal_transactions = 17 := by
  sorry

end NUMINAMATH_CALUDE_jade_cal_difference_l614_61483


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l614_61493

theorem at_least_one_geq_two (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l614_61493


namespace NUMINAMATH_CALUDE_smallest_repeating_block_length_l614_61404

/-- The number of digits in the smallest repeating block of the decimal expansion of 3/11 -/
def repeating_block_length : ℕ := 2

/-- The fraction we are considering -/
def fraction : ℚ := 3 / 11

theorem smallest_repeating_block_length :
  repeating_block_length = 2 ∧
  ∀ n : ℕ, n < repeating_block_length →
    ¬∃ (a b : ℕ), fraction = (a : ℚ) / (10^n : ℚ) + (b : ℚ) / (10^n - 1 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_smallest_repeating_block_length_l614_61404


namespace NUMINAMATH_CALUDE_principal_is_800_l614_61442

/-- Calculates the principal amount given the simple interest rate, final amount, and time period. -/
def calculate_principal (rate : ℚ) (final_amount : ℚ) (time : ℕ) : ℚ :=
  (final_amount * 100) / (rate * time)

/-- Theorem stating that the principal amount is 800 given the specified conditions. -/
theorem principal_is_800 (rate : ℚ) (final_amount : ℚ) (time : ℕ) 
  (h_rate : rate = 25/400)  -- 6.25% as a rational number
  (h_final_amount : final_amount = 200)
  (h_time : time = 4) :
  calculate_principal rate final_amount time = 800 := by
  sorry

#eval calculate_principal (25/400) 200 4  -- This should evaluate to 800

end NUMINAMATH_CALUDE_principal_is_800_l614_61442


namespace NUMINAMATH_CALUDE_nap_time_calculation_l614_61480

/-- Calculates the remaining time for a nap given flight duration and time spent on activities -/
def remaining_nap_time (
  flight_hours : ℕ
) (
  flight_minutes : ℕ
) (
  reading_hours : ℕ
) (
  movie_hours : ℕ
) (
  dinner_minutes : ℕ
) (
  radio_minutes : ℕ
) (
  game_hours : ℕ
) (
  game_minutes : ℕ
) : ℕ :=
  let total_flight_minutes := flight_hours * 60 + flight_minutes
  let total_activity_minutes := 
    reading_hours * 60 + 
    movie_hours * 60 + 
    dinner_minutes + 
    radio_minutes + 
    game_hours * 60 + 
    game_minutes
  (total_flight_minutes - total_activity_minutes) / 60

theorem nap_time_calculation : 
  remaining_nap_time 11 20 2 4 30 40 1 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_nap_time_calculation_l614_61480


namespace NUMINAMATH_CALUDE_slope_product_is_four_l614_61427

-- Define the parabola and line
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0
def line (x y : ℝ) : Prop := y = 2*x - 4

-- Define points A, B, and C
def point_on_parabola_and_line (p : ℝ) (x y : ℝ) : Prop :=
  parabola p x y ∧ line x y

-- Define the vector relation
def vector_relation (p : ℝ) (xA yA xB yB xC yC : ℝ) : Prop :=
  xA + xB = (1/5) * xC ∧ yA + yB = (1/5) * yC

-- Define point M
def point_M (x y : ℝ) : Prop := x = 2 ∧ y = 2

-- Theorem statement
theorem slope_product_is_four (p : ℝ) (xA yA xB yB xC yC : ℝ) :
  point_on_parabola_and_line p xA yA →
  point_on_parabola_and_line p xB yB →
  parabola p xC yC →
  vector_relation p xA yA xB yB xC yC →
  point_M 2 2 →
  ((yA - 2) / (xA - 2)) * ((yB - 2) / (xB - 2)) = 4 :=
sorry

end NUMINAMATH_CALUDE_slope_product_is_four_l614_61427


namespace NUMINAMATH_CALUDE_largest_winning_start_is_correct_l614_61492

/-- The largest starting integer that guarantees a win for Bernardo in the number game. -/
def largest_winning_start : ℕ := 40

/-- Checks if a given number is a valid starting number for Bernardo to win. -/
def is_valid_start (m : ℕ) : Prop :=
  m ≥ 1 ∧ m ≤ 500 ∧
  3 * m < 1500 ∧
  3 * m + 30 < 1500 ∧
  9 * m + 90 < 1500 ∧
  9 * m + 120 < 1500 ∧
  27 * m + 360 < 1500 ∧
  27 * m + 390 < 1500

theorem largest_winning_start_is_correct :
  is_valid_start largest_winning_start ∧
  ∀ n : ℕ, n > largest_winning_start → ¬ is_valid_start n :=
by sorry

end NUMINAMATH_CALUDE_largest_winning_start_is_correct_l614_61492


namespace NUMINAMATH_CALUDE_quilt_transformation_l614_61465

theorem quilt_transformation (length width : ℝ) (h1 : length = 6) (h2 : width = 24) :
  ∃ (side : ℝ), side^2 = length * width ∧ side = 12 := by
  sorry

end NUMINAMATH_CALUDE_quilt_transformation_l614_61465


namespace NUMINAMATH_CALUDE_square_roots_equality_l614_61417

theorem square_roots_equality (m : ℝ) : 
  (∃ (k : ℝ), k > 0 ∧ (2*m - 4)^2 = k ∧ (3*m - 1)^2 = k) → (m = -3 ∨ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_square_roots_equality_l614_61417


namespace NUMINAMATH_CALUDE_symmetric_sine_cosine_l614_61473

theorem symmetric_sine_cosine (φ : ℝ) (h1 : |φ| < π/2) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (x + φ) - Real.sqrt 3 * Real.cos (x + φ)
  (∀ x, f (2*π - x) = f x) →
  Real.cos (2*φ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_sine_cosine_l614_61473


namespace NUMINAMATH_CALUDE_coordinates_of_C_l614_61491

-- Define the points
def A : ℝ × ℝ := (2, 8)
def M : ℝ × ℝ := (4, 11)
def L : ℝ × ℝ := (6, 6)

-- Define the properties
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

def on_angle_bisector (L B C : ℝ × ℝ) : Prop :=
  (L.1 - B.1) * (C.2 - B.2) = (L.2 - B.2) * (C.1 - B.1)

-- Theorem statement
theorem coordinates_of_C (B C : ℝ × ℝ) 
  (h1 : is_midpoint M A B)
  (h2 : on_angle_bisector L B C) :
  C = (14, 2) := by sorry

end NUMINAMATH_CALUDE_coordinates_of_C_l614_61491


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l614_61420

theorem cube_root_equation_solution (y : ℝ) :
  (5 - 2 / y) ^ (1/3 : ℝ) = -3 → y = 1/16 := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l614_61420


namespace NUMINAMATH_CALUDE_hyperbola_triangle_area_l614_61484

/-- The hyperbola with equation x^2/9 - y^2/16 = 1 -/
def hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 9) - (p.2^2 / 16) = 1}

/-- The right focus of the hyperbola -/
def F : ℝ × ℝ := (5, 0)

/-- The origin -/
def O : ℝ × ℝ := (0, 0)

/-- A point on the hyperbola where a line perpendicular to an asymptote intersects it -/
def P : ℝ × ℝ := sorry

/-- The area of triangle OPF -/
def area_OPF : ℝ := sorry

theorem hyperbola_triangle_area :
  area_OPF = 6 := by sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_area_l614_61484


namespace NUMINAMATH_CALUDE_cube_vertex_sum_difference_l614_61461

theorem cube_vertex_sum_difference (a b c d e f g h : ℝ) 
  (ha : 3 * a = b + e + d)
  (hb : 3 * b = c + f + a)
  (hc : 3 * c = d + g + b)
  (hd : 3 * d = a + h + c)
  (he : 3 * e = f + a + h)
  (hf : 3 * f = g + b + e)
  (hg : 3 * g = h + c + f)
  (hh : 3 * h = e + d + g) :
  (a + b + c + d) - (e + f + g + h) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_vertex_sum_difference_l614_61461


namespace NUMINAMATH_CALUDE_f_monotonicity_and_zeros_l614_61451

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - x - a

theorem f_monotonicity_and_zeros (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ∨
  (∃ z : ℝ, ∀ x y : ℝ, x < y → x < z → f a x > f a y) ∧
    (∀ x y : ℝ, x < y → z < x → f a x < f a y) ∧
  (∃! x₁ x₂ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧ x₁ ≠ x₂) ↔ a ∈ Set.Ioo 0 1 ∪ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_zeros_l614_61451


namespace NUMINAMATH_CALUDE_jack_sandwich_change_l614_61476

/-- Calculates the change Jack receives after buying sandwiches -/
theorem jack_sandwich_change :
  let sandwich1_price : ℚ := 5
  let sandwich2_price : ℚ := 5
  let sandwich3_price : ℚ := 6
  let sandwich4_price : ℚ := 7
  let discount1 : ℚ := 0.1
  let discount2 : ℚ := 0.1
  let discount3 : ℚ := 0.15
  let discount4 : ℚ := 0
  let tax_rate : ℚ := 0.05
  let service_fee : ℚ := 2
  let payment : ℚ := 34

  let total_cost : ℚ :=
    (sandwich1_price * (1 - discount1) +
     sandwich2_price * (1 - discount2) +
     sandwich3_price * (1 - discount3) +
     sandwich4_price * (1 - discount4)) *
    (1 + tax_rate) + service_fee

  payment - total_cost = 9.84
:= by sorry

end NUMINAMATH_CALUDE_jack_sandwich_change_l614_61476


namespace NUMINAMATH_CALUDE_expression_theorem_l614_61440

-- Define the expression E as a function of x
def E (x : ℝ) : ℝ := 6 * x + 45

-- State the theorem
theorem expression_theorem (x : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₂ - x₁ = 12) →
  (E x) / (2 * x + 15) = 3 →
  E x = 6 * x + 45 := by
sorry

end NUMINAMATH_CALUDE_expression_theorem_l614_61440


namespace NUMINAMATH_CALUDE_integer_list_mean_mode_l614_61423

theorem integer_list_mean_mode (y : ℕ) : 
  y > 0 ∧ y ≤ 150 →
  let l := [45, 76, 123, y, y, y]
  (l.sum / l.length : ℚ) = 2 * y →
  y = 27 := by
sorry

end NUMINAMATH_CALUDE_integer_list_mean_mode_l614_61423


namespace NUMINAMATH_CALUDE_parabola_point_x_coordinate_l614_61416

/-- The x-coordinate of a point on a parabola at a given distance from the directrix -/
theorem parabola_point_x_coordinate 
  (x y : ℝ) -- x and y coordinates of point M
  (h1 : y^2 = 4*x) -- point M is on the parabola y² = 4x
  (h2 : |x + 1| = 3) -- distance from M to the directrix x = -1 is 3
  : x = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_point_x_coordinate_l614_61416


namespace NUMINAMATH_CALUDE_jake_present_weight_l614_61486

/-- Jake's present weight in pounds -/
def jake_weight : ℕ := 194

/-- Kendra's weight in pounds -/
def kendra_weight : ℕ := 287 - jake_weight

/-- The amount of weight Jake needs to lose to weigh twice as much as Kendra -/
def weight_loss : ℕ := jake_weight - 2 * kendra_weight

theorem jake_present_weight : jake_weight = 194 := by
  sorry

end NUMINAMATH_CALUDE_jake_present_weight_l614_61486


namespace NUMINAMATH_CALUDE_search_plans_count_l614_61459

/-- Represents the number of children in the group -/
def total_children : ℕ := 6

/-- Represents the number of food drop locations -/
def num_locations : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the number of search plans when Grace doesn't participate -/
def plans_without_grace : ℕ := 
  (choose (total_children - 1) 1) * ((choose (total_children - 2) 2) / 2) * (Nat.factorial num_locations)

/-- Calculates the number of search plans when Grace participates -/
def plans_with_grace : ℕ := choose (total_children - 1) 2

/-- The total number of different search plans -/
def total_plans : ℕ := plans_without_grace + plans_with_grace

/-- Theorem stating that the total number of different search plans is 40 -/
theorem search_plans_count : total_plans = 40 := by
  sorry


end NUMINAMATH_CALUDE_search_plans_count_l614_61459


namespace NUMINAMATH_CALUDE_mei_oranges_l614_61436

theorem mei_oranges (peaches pears oranges baskets : ℕ) : 
  peaches = 9 →
  pears = 18 →
  baskets > 0 →
  peaches % baskets = 0 →
  pears % baskets = 0 →
  oranges % baskets = 0 →
  baskets = 3 →
  oranges = 9 :=
by sorry

end NUMINAMATH_CALUDE_mei_oranges_l614_61436


namespace NUMINAMATH_CALUDE_minimum_driving_age_l614_61489

/-- The minimum driving age problem -/
theorem minimum_driving_age 
  (kayla_age : ℕ) 
  (kimiko_age : ℕ) 
  (min_driving_age : ℕ) 
  (h1 : kayla_age * 2 = kimiko_age) 
  (h2 : kimiko_age = 26) 
  (h3 : min_driving_age = kayla_age + 5) : 
  min_driving_age = 18 := by
sorry

end NUMINAMATH_CALUDE_minimum_driving_age_l614_61489


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l614_61446

theorem mod_equivalence_unique_solution : 
  ∃! n : ℤ, 0 ≤ n ∧ n < 137 ∧ 12345 ≡ n [ZMOD 137] := by sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l614_61446


namespace NUMINAMATH_CALUDE_log_35_28_in_terms_of_a_and_b_l614_61497

theorem log_35_28_in_terms_of_a_and_b (a b : ℝ) 
  (h1 : Real.log 7 / Real.log 14 = a) 
  (h2 : Real.log 5 / Real.log 14 = b) : 
  Real.log 28 / Real.log 35 = (2 - a) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_log_35_28_in_terms_of_a_and_b_l614_61497


namespace NUMINAMATH_CALUDE_company_median_salary_l614_61409

/-- Represents a job position with its title, number of employees, and salary --/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- The list of positions in the company --/
def company_positions : List Position := [
  { title := "President", count := 1, salary := 140000 },
  { title := "Vice-President", count := 10, salary := 100000 },
  { title := "Director", count := 15, salary := 80000 },
  { title := "Manager", count := 5, salary := 55000 },
  { title := "Associate Director", count := 9, salary := 52000 },
  { title := "Administrative Specialist", count := 35, salary := 25000 }
]

/-- The total number of employees in the company --/
def total_employees : Nat := 75

/-- Calculates the median salary of the company --/
def median_salary (positions : List Position) (total : Nat) : Nat :=
  sorry

/-- Theorem stating that the median salary of the company is $52,000 --/
theorem company_median_salary :
  median_salary company_positions total_employees = 52000 := by
  sorry

end NUMINAMATH_CALUDE_company_median_salary_l614_61409


namespace NUMINAMATH_CALUDE_rectangle_diagonal_length_l614_61479

/-- A rectangle with given area and perimeter -/
structure Rectangle where
  area : ℝ
  perimeter : ℝ

/-- The length of the diagonal of a rectangle -/
def diagonal_length (r : Rectangle) : ℝ :=
  sorry

theorem rectangle_diagonal_length :
  ∀ r : Rectangle, r.area = 16 ∧ r.perimeter = 18 → diagonal_length r = 7 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_length_l614_61479


namespace NUMINAMATH_CALUDE_meeting_point_symmetry_l614_61450

theorem meeting_point_symmetry 
  (d : ℝ) (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b / (a + b) * (d - a / 2) - b / (a + b) * d = -2) →
  (a / (a + b) * (d - b / 2) - a / (a + b) * d = -2) :=
by sorry

end NUMINAMATH_CALUDE_meeting_point_symmetry_l614_61450


namespace NUMINAMATH_CALUDE_min_gumballs_for_five_colors_l614_61464

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat
  green : Nat

/-- The minimum number of gumballs needed to guarantee five of the same color -/
def minGumballsForFive (machine : GumballMachine) : Nat :=
  16

/-- Theorem stating that for a machine with 10 red, 10 white, 10 blue, and 6 green gumballs,
    the minimum number of gumballs needed to guarantee five of the same color is 16 -/
theorem min_gumballs_for_five_colors (machine : GumballMachine)
    (h_red : machine.red = 10)
    (h_white : machine.white = 10)
    (h_blue : machine.blue = 10)
    (h_green : machine.green = 6) :
    minGumballsForFive machine = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_gumballs_for_five_colors_l614_61464


namespace NUMINAMATH_CALUDE_set_containment_implies_a_bound_l614_61430

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2^(1 - x) + a ≤ 0}

-- State the theorem
theorem set_containment_implies_a_bound (a : ℝ) :
  A ⊆ B a → a ≤ -2 := by
  sorry

-- The range of a is implicitly (-∞, -2] because a ≤ -2

end NUMINAMATH_CALUDE_set_containment_implies_a_bound_l614_61430


namespace NUMINAMATH_CALUDE_divisibility_by_eight_l614_61467

theorem divisibility_by_eight : ∃ k : ℤ, 5^2001 + 7^2002 + 9^2003 + 11^2004 = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eight_l614_61467


namespace NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l614_61435

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: The sum of the interior angles of a pentagon is 540° -/
theorem sum_interior_angles_pentagon :
  sum_interior_angles 5 = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l614_61435


namespace NUMINAMATH_CALUDE_comparison_of_trigonometric_expressions_l614_61441

theorem comparison_of_trigonometric_expressions :
  let a := (1/2) * Real.cos (4 * π / 180) - (Real.sqrt 3 / 2) * Real.sin (4 * π / 180)
  let b := Real.cos (13 * π / 180)^2 - Real.sin (13 * π / 180)^2
  let c := (2 * Real.tan (23 * π / 180)) / (1 - Real.tan (23 * π / 180)^2)
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_comparison_of_trigonometric_expressions_l614_61441


namespace NUMINAMATH_CALUDE_b_completion_time_l614_61448

/-- The time it takes for person A to complete the work alone -/
def time_A : ℝ := 15

/-- The time A and B work together -/
def time_together : ℝ := 5

/-- The fraction of work left after A and B work together -/
def work_left : ℝ := 0.41666666666666663

/-- The time it takes for person B to complete the work alone -/
def time_B : ℝ := 20

/-- Theorem stating that given the conditions, B takes 20 days to complete the work alone -/
theorem b_completion_time :
  (time_together * (1 / time_A + 1 / time_B) = 1 - work_left) →
  time_B = 20 := by
  sorry

end NUMINAMATH_CALUDE_b_completion_time_l614_61448


namespace NUMINAMATH_CALUDE_largest_area_polygon_E_l614_61425

/-- Represents a polygon composed of unit squares and right triangles -/
structure Polygon where
  unitSquares : ℕ
  rightTriangles : ℕ

/-- Calculates the area of a polygon -/
def areaOfPolygon (p : Polygon) : ℚ :=
  p.unitSquares + p.rightTriangles / 2

/-- The given polygons -/
def polygonA : Polygon := ⟨3, 2⟩
def polygonB : Polygon := ⟨6, 0⟩
def polygonC : Polygon := ⟨4, 3⟩
def polygonD : Polygon := ⟨5, 1⟩
def polygonE : Polygon := ⟨7, 0⟩

theorem largest_area_polygon_E :
  ∀ p ∈ [polygonA, polygonB, polygonC, polygonD, polygonE],
    areaOfPolygon p ≤ areaOfPolygon polygonE :=
by sorry

end NUMINAMATH_CALUDE_largest_area_polygon_E_l614_61425


namespace NUMINAMATH_CALUDE_berry_picking_pattern_l614_61434

/-- A sequence of 5 numbers where the differences between consecutive terms
    form an arithmetic sequence with a common difference of 2 -/
def BerrySequence (a b c d e : ℕ) : Prop :=
  (c - b) - (b - a) = 2 ∧
  (d - c) - (c - b) = 2 ∧
  (e - d) - (d - c) = 2

theorem berry_picking_pattern (a b c d e : ℕ) :
  BerrySequence a b c d e →
  a = 3 →
  c = 7 →
  d = 12 →
  e = 19 →
  b = 6 := by
sorry

end NUMINAMATH_CALUDE_berry_picking_pattern_l614_61434


namespace NUMINAMATH_CALUDE_not_divisible_1985_1987_divisible_1987_1989_l614_61471

/-- Represents an L-shaped piece consisting of 3 unit squares -/
structure LShape :=
  (width : ℕ)
  (height : ℕ)
  (area_eq_3 : width * height = 3)

/-- Checks if a rectangle can be divided into L-shapes -/
def can_divide_into_l_shapes (m n : ℕ) : Prop :=
  (m * n) % 3 = 0 ∨ 
  ∃ (a b c d : ℕ), m = 2 * a + 7 * b ∧ n = 3 * c + 9 * d

/-- Theorem stating the divisibility condition for 1985 × 1987 rectangle -/
theorem not_divisible_1985_1987 : ¬(can_divide_into_l_shapes 1985 1987) :=
sorry

/-- Theorem stating the divisibility condition for 1987 × 1989 rectangle -/
theorem divisible_1987_1989 : can_divide_into_l_shapes 1987 1989 :=
sorry

end NUMINAMATH_CALUDE_not_divisible_1985_1987_divisible_1987_1989_l614_61471


namespace NUMINAMATH_CALUDE_eleanor_cookies_l614_61424

theorem eleanor_cookies (N : ℕ) : 
  N % 13 = 5 → N % 8 = 3 → N < 150 → N = 83 :=
by
  sorry

end NUMINAMATH_CALUDE_eleanor_cookies_l614_61424


namespace NUMINAMATH_CALUDE_part1_part2_l614_61421

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- angles
  (a b c : ℝ)  -- sides

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the specific conditions of our triangle
def ourTriangle (t : Triangle) : Prop :=
  isValidTriangle t ∧
  t.B = Real.pi / 3 ∧  -- 60 degrees
  t.c = 8

-- Define the midpoint condition
def isMidpoint (M : ℝ × ℝ) (B C : ℝ × ℝ) : Prop :=
  M.1 = (B.1 + C.1) / 2 ∧ M.2 = (B.2 + C.2) / 2

-- Theorem for part 1
theorem part1 (t : Triangle) (M : ℝ × ℝ) (B C : ℝ × ℝ) :
  ourTriangle t →
  isMidpoint M B C →
  (Real.sqrt 3) * (Real.sqrt ((M.1 - B.1)^2 + (M.2 - B.2)^2)) = 
    Real.sqrt ((t.A - M.1)^2 + (t.A - M.2)^2) →
  t.b = 8 :=
sorry

-- Theorem for part 2
theorem part2 (t : Triangle) :
  ourTriangle t →
  t.b = 12 →
  (1/2) * t.b * t.c * Real.sin t.A = 24 * Real.sqrt 2 + 8 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l614_61421


namespace NUMINAMATH_CALUDE_mutually_exclusive_pairs_l614_61481

-- Define a type for events
inductive Event
| SevenRing
| EightRing
| AtLeastOneHit
| AHitsBMisses
| AtLeastOneBlack
| BothRed
| NoBlack
| ExactlyOneRed

-- Define a function to check if two events are mutually exclusive
def mutuallyExclusive (e1 e2 : Event) : Prop :=
  ¬(∃ (outcome : Set Event), outcome.Nonempty ∧ e1 ∈ outcome ∧ e2 ∈ outcome)

-- Define the pairs of events
def pair1 : (Event × Event) := (Event.SevenRing, Event.EightRing)
def pair2 : (Event × Event) := (Event.AtLeastOneHit, Event.AHitsBMisses)
def pair3 : (Event × Event) := (Event.AtLeastOneBlack, Event.BothRed)
def pair4 : (Event × Event) := (Event.NoBlack, Event.ExactlyOneRed)

-- State the theorem
theorem mutually_exclusive_pairs :
  mutuallyExclusive pair1.1 pair1.2 ∧
  ¬(mutuallyExclusive pair2.1 pair2.2) ∧
  mutuallyExclusive pair3.1 pair3.2 ∧
  mutuallyExclusive pair4.1 pair4.2 := by
  sorry

end NUMINAMATH_CALUDE_mutually_exclusive_pairs_l614_61481


namespace NUMINAMATH_CALUDE_equation_solution_l614_61487

theorem equation_solution :
  let f : ℝ → ℝ := λ x => (3*x + 7)*(x - 2) - (7*x - 4)
  ∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 39 / 3 ∧
              x₂ = 1 - Real.sqrt 39 / 3 ∧
              f x₁ = 0 ∧
              f x₂ = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l614_61487


namespace NUMINAMATH_CALUDE_tesseract_hypervolume_l614_61475

/-- Given a tesseract with face volumes 72, 75, 48, and 50 cubic units,
    its hyper-volume is 3600 hyper-cubic units. -/
theorem tesseract_hypervolume (a b c d : ℝ) 
    (h1 : a * b * c = 72)
    (h2 : b * c * d = 75)
    (h3 : c * d * a = 48)
    (h4 : d * a * b = 50) : 
    a * b * c * d = 3600 := by
  sorry

#check tesseract_hypervolume

end NUMINAMATH_CALUDE_tesseract_hypervolume_l614_61475


namespace NUMINAMATH_CALUDE_water_balloon_ratio_l614_61463

/-- Prove that the ratio of Randy's water balloons to Janice's water balloons is 1:2 -/
theorem water_balloon_ratio 
  (cynthia_balloons : ℕ) 
  (janice_balloons : ℕ) 
  (h1 : cynthia_balloons = 12)
  (h2 : janice_balloons = 6)
  (h3 : cynthia_balloons = 4 * (cynthia_balloons / 4)) :
  (cynthia_balloons / 4) / janice_balloons = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_water_balloon_ratio_l614_61463


namespace NUMINAMATH_CALUDE_octagon_lines_l614_61443

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a pentagon -/
def pentagon_sides : ℕ := 5

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- The number of triangles Bill drew -/
def num_triangles : ℕ := 12

/-- The number of squares Bill drew -/
def num_squares : ℕ := 8

/-- The number of pentagons Bill drew -/
def num_pentagons : ℕ := 4

/-- The number of hexagons Bill drew -/
def num_hexagons : ℕ := 6

/-- The number of octagons Bill drew -/
def num_octagons : ℕ := 2

/-- The number of lines shared between triangles and squares -/
def shared_triangle_square : ℕ := 5

/-- The number of lines shared between pentagons and hexagons -/
def shared_pentagon_hexagon : ℕ := 3

/-- The number of lines shared between hexagons and octagons -/
def shared_hexagon_octagon : ℕ := 1

/-- Theorem: The number of lines drawn with the purple marker (for octagons) is 15 -/
theorem octagon_lines : 
  num_octagons * octagon_sides - shared_hexagon_octagon = 15 := by sorry

end NUMINAMATH_CALUDE_octagon_lines_l614_61443


namespace NUMINAMATH_CALUDE_product_equals_fraction_l614_61429

/-- The repeating decimal 0.456̄ as a rational number -/
def repeating_decimal : ℚ := 456 / 999

/-- The product of the repeating decimal 0.456̄ and 7 -/
def product : ℚ := repeating_decimal * 7

/-- Theorem stating that the product of 0.456̄ and 7 is equal to 1064/333 -/
theorem product_equals_fraction : product = 1064 / 333 := by sorry

end NUMINAMATH_CALUDE_product_equals_fraction_l614_61429


namespace NUMINAMATH_CALUDE_sequence_integer_count_l614_61428

def sequence_term (n : ℕ) : ℚ :=
  12150 / 3^n

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem sequence_integer_count :
  ∃ (k : ℕ), k = 5 ∧
  (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
  (∀ (n : ℕ), n ≥ k → ¬ is_integer (sequence_term n)) :=
sorry

end NUMINAMATH_CALUDE_sequence_integer_count_l614_61428


namespace NUMINAMATH_CALUDE_product_not_ending_1999_l614_61433

theorem product_not_ending_1999 (a b c d e : ℕ) : 
  a + b + c + d + e = 200 → 
  ∃ k : ℕ, a * b * c * d * e = 1000 * k ∨ a * b * c * d * e = 1000 * k + 1 ∨ 
          a * b * c * d * e = 1000 * k + 2 ∨ a * b * c * d * e = 1000 * k + 3 ∨ 
          a * b * c * d * e = 1000 * k + 4 ∨ a * b * c * d * e = 1000 * k + 5 ∨ 
          a * b * c * d * e = 1000 * k + 6 ∨ a * b * c * d * e = 1000 * k + 7 ∨ 
          a * b * c * d * e = 1000 * k + 8 ∨ a * b * c * d * e = 1000 * k + 9 := by
  sorry

end NUMINAMATH_CALUDE_product_not_ending_1999_l614_61433


namespace NUMINAMATH_CALUDE_not_all_two_equal_sides_congruent_l614_61407

-- Define a right triangle
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  is_right : leg1^2 + leg2^2 = hypotenuse^2

-- Define congruence for right triangles
def congruent (t1 t2 : RightTriangle) : Prop :=
  t1.leg1 = t2.leg1 ∧ t1.leg2 = t2.leg2 ∧ t1.hypotenuse = t2.hypotenuse

-- Statement to be proven false
theorem not_all_two_equal_sides_congruent :
  ¬ (∀ t1 t2 : RightTriangle,
    (t1.leg1 = t2.leg1 ∧ t1.leg2 = t2.leg2) ∨
    (t1.leg1 = t2.leg1 ∧ t1.hypotenuse = t2.hypotenuse) ∨
    (t1.leg2 = t2.leg2 ∧ t1.hypotenuse = t2.hypotenuse)
    → congruent t1 t2) :=
  sorry

end NUMINAMATH_CALUDE_not_all_two_equal_sides_congruent_l614_61407


namespace NUMINAMATH_CALUDE_min_value_w_l614_61402

theorem min_value_w (x y : ℝ) : 
  3 * x^2 + 3 * y^2 + 9 * x - 6 * y + 30 ≥ 20.25 ∧ 
  ∃ (a b : ℝ), 3 * a^2 + 3 * b^2 + 9 * a - 6 * b + 30 = 20.25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_w_l614_61402


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l614_61472

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a where a₄ = 7 and a₆ = 21, prove that a₈ = 63. -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) 
  (h_a4 : a 4 = 7) 
  (h_a6 : a 6 = 21) : 
  a 8 = 63 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l614_61472


namespace NUMINAMATH_CALUDE_area_difference_equals_target_l614_61460

/-- A right triangle with side lengths 3, 4, and 5 -/
structure RightTriangle where
  base : Real
  height : Real
  hypotenuse : Real
  is_right : base = 3 ∧ height = 4 ∧ hypotenuse = 5

/-- The set Xₙ as defined in the problem -/
def X (n : ℕ) (t : RightTriangle) : Set (Real × Real) :=
  sorry

/-- The area of the region outside X₂₀ but inside X₂₁ -/
def area_difference (t : RightTriangle) : Real :=
  sorry

/-- The main theorem to prove -/
theorem area_difference_equals_target (t : RightTriangle) :
  area_difference t = (41 * Real.pi / 2) + 12 := by
  sorry

end NUMINAMATH_CALUDE_area_difference_equals_target_l614_61460


namespace NUMINAMATH_CALUDE_luke_lawn_mowing_earnings_l614_61453

theorem luke_lawn_mowing_earnings :
  ∀ (L : ℝ),
  (∃ (total_earnings : ℝ),
    total_earnings = L + 18 ∧
    total_earnings = 3 * 9) →
  L = 9 := by
sorry

end NUMINAMATH_CALUDE_luke_lawn_mowing_earnings_l614_61453


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l614_61495

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃! x : ℕ, x < d ∧ (n - x) % d = 0 :=
by sorry

theorem problem_solution :
  let n := 42739
  let d := 15
  (least_subtraction_for_divisibility n d (by norm_num)).choose = 4 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l614_61495


namespace NUMINAMATH_CALUDE_element_value_l614_61490

theorem element_value (a : Nat) : 
  a ∈ ({0, 1, 2, 3} : Set Nat) → 
  a ∉ ({0, 1, 2} : Set Nat) → 
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_element_value_l614_61490


namespace NUMINAMATH_CALUDE_sector_angle_l614_61482

/-- Given a circular sector with arc length and area both equal to 6,
    prove that the central angle in radians is 3. -/
theorem sector_angle (r : ℝ) (α : ℝ) : 
  r * α = 6 →  -- arc length formula
  (1 / 2) * r * α = 6 →  -- area formula
  α = 3 := by sorry

end NUMINAMATH_CALUDE_sector_angle_l614_61482
