import Mathlib

namespace NUMINAMATH_CALUDE_nina_bought_two_card_packs_l4062_406287

def num_toys : ℕ := 3
def toy_price : ℕ := 10
def num_shirts : ℕ := 5
def shirt_price : ℕ := 6
def card_pack_price : ℕ := 5
def total_spent : ℕ := 70

theorem nina_bought_two_card_packs :
  (num_toys * toy_price + num_shirts * shirt_price + 2 * card_pack_price = total_spent) := by
  sorry

end NUMINAMATH_CALUDE_nina_bought_two_card_packs_l4062_406287


namespace NUMINAMATH_CALUDE_airplane_seats_l4062_406256

theorem airplane_seats (total_seats : ℕ) (first_class : ℕ) : 
  total_seats = 567 → 
  first_class + 3 * first_class + (7 * first_class + 5) = total_seats →
  7 * first_class + 5 = 362 := by
sorry

end NUMINAMATH_CALUDE_airplane_seats_l4062_406256


namespace NUMINAMATH_CALUDE_ratio_problem_l4062_406217

theorem ratio_problem (a b c d e f : ℝ) 
  (h1 : a * b * c / (d * e * f) = 1.875)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 1)
  (h4 : d / e = 3 / 2)
  (h5 : e / f = 4 / 3) :
  a / b = 1.40625 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l4062_406217


namespace NUMINAMATH_CALUDE_a_range_l4062_406288

def f (a : ℝ) (x : ℝ) : ℝ := x * |x - a|

theorem a_range (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ici 3 → x₂ ∈ Set.Ici 3 → x₁ ≠ x₂ → 
    (f a x₁ - f a x₂) / (x₁ - x₂) > 0) → 
  a ∈ Set.Iic 3 := by
sorry

end NUMINAMATH_CALUDE_a_range_l4062_406288


namespace NUMINAMATH_CALUDE_function_inequality_l4062_406208

open Set

theorem function_inequality (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f)
  (h_sym : ∀ x, f x = f (2 - x))
  (h_mono : ∀ x ∈ Iio 1, (x - 1) * deriv f x < 0) :
  f 3 < f 0 ∧ f 0 < f (1/2) := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l4062_406208


namespace NUMINAMATH_CALUDE_cupcakes_frosted_in_ten_minutes_l4062_406282

def mark_rate : ℚ := 1 / 15
def julia_rate : ℚ := 1 / 40
def total_time : ℚ := 10 * 60  -- 10 minutes in seconds

theorem cupcakes_frosted_in_ten_minutes : 
  ⌊(mark_rate + julia_rate) * total_time⌋ = 55 := by sorry

end NUMINAMATH_CALUDE_cupcakes_frosted_in_ten_minutes_l4062_406282


namespace NUMINAMATH_CALUDE_unique_white_bucket_count_l4062_406259

/-- Represents a bucket with its color and water content -/
structure Bucket :=
  (color : Bool)  -- true for red, false for white
  (water : ℕ)

/-- Represents a move where water is added to a pair of buckets -/
structure Move :=
  (red_bucket : ℕ)
  (white_bucket : ℕ)
  (water_added : ℕ)

/-- The main theorem statement -/
theorem unique_white_bucket_count
  (red_count : ℕ)
  (white_count : ℕ)
  (moves : List Move)
  (h_red_count : red_count = 100)
  (h_all_non_empty : ∀ b : Bucket, b.water > 0)
  (h_equal_water : ∀ m : Move, ∃ b1 b2 : Bucket,
    b1.color = true ∧ b2.color = false ∧
    b1.water = b2.water) :
  white_count = 100 := by
  sorry

end NUMINAMATH_CALUDE_unique_white_bucket_count_l4062_406259


namespace NUMINAMATH_CALUDE_linear_equation_equivalence_l4062_406260

theorem linear_equation_equivalence (x y : ℝ) :
  (2 * x - y = 3) ↔ (y = 2 * x - 3) := by sorry

end NUMINAMATH_CALUDE_linear_equation_equivalence_l4062_406260


namespace NUMINAMATH_CALUDE_curve_is_two_lines_l4062_406235

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop := x^2 - x*y - 2*y^2 = 0

/-- The curve represents two straight lines -/
theorem curve_is_two_lines : 
  ∃ (a b c d : ℝ), ∀ (x y : ℝ), 
    curve_equation x y ↔ (a*x + b*y = 0 ∧ c*x + d*y = 0) :=
sorry

end NUMINAMATH_CALUDE_curve_is_two_lines_l4062_406235


namespace NUMINAMATH_CALUDE_specific_quadratic_equation_l4062_406298

/-- A quadratic equation with given roots and leading coefficient -/
def quadratic_equation (root1 root2 : ℝ) (leading_coeff : ℝ) : ℝ → ℝ :=
  fun x => leading_coeff * (x - root1) * (x - root2)

/-- Theorem: The quadratic equation with roots -3 and 7 and leading coefficient 1 is x^2 - 4x - 21 = 0 -/
theorem specific_quadratic_equation :
  quadratic_equation (-3) 7 1 = fun x => x^2 - 4*x - 21 := by sorry

end NUMINAMATH_CALUDE_specific_quadratic_equation_l4062_406298


namespace NUMINAMATH_CALUDE_pairing_possibility_l4062_406239

/-- Represents a pairing of children -/
structure Pairing :=
  (boys : ℕ)   -- Number of boy-boy pairs
  (girls : ℕ)  -- Number of girl-girl pairs
  (mixed : ℕ)  -- Number of boy-girl pairs

/-- Represents a group of children that can be arranged in different pairings -/
structure ChildrenGroup :=
  (to_museum : Pairing)
  (from_museum : Pairing)
  (total_boys : ℕ)
  (total_girls : ℕ)

/-- The theorem to be proved -/
theorem pairing_possibility (group : ChildrenGroup) 
  (h1 : group.to_museum.boys = 3 * group.to_museum.girls)
  (h2 : group.from_museum.boys = 4 * group.from_museum.girls)
  (h3 : group.total_boys = 2 * group.to_museum.boys + group.to_museum.mixed)
  (h4 : group.total_girls = 2 * group.to_museum.girls + group.to_museum.mixed)
  (h5 : group.total_boys = 2 * group.from_museum.boys + group.from_museum.mixed)
  (h6 : group.total_girls = 2 * group.from_museum.girls + group.from_museum.mixed) :
  ∃ (new_pairing : Pairing), 
    new_pairing.boys = 7 * new_pairing.girls ∧ 
    2 * new_pairing.boys + 2 * new_pairing.girls + new_pairing.mixed = group.total_boys + group.total_girls :=
sorry

end NUMINAMATH_CALUDE_pairing_possibility_l4062_406239


namespace NUMINAMATH_CALUDE_john_share_l4062_406219

/-- Given a total amount and a ratio, calculates the share for a specific part -/
def calculateShare (totalAmount : ℚ) (ratio : List ℚ) (part : ℚ) : ℚ :=
  let totalParts := ratio.sum
  let valuePerPart := totalAmount / totalParts
  valuePerPart * part

/-- Proves that given a total amount of 4200 and a ratio of 2:4:6:8, 
    the amount received by the person with 2 parts is 420 -/
theorem john_share : 
  let totalAmount : ℚ := 4200
  let ratio : List ℚ := [2, 4, 6, 8]
  calculateShare totalAmount ratio 2 = 420 := by sorry

end NUMINAMATH_CALUDE_john_share_l4062_406219


namespace NUMINAMATH_CALUDE_bryden_receives_amount_l4062_406293

/-- The face value of a state quarter in dollars -/
def quarterValue : ℚ := 1 / 4

/-- The number of quarters Bryden has -/
def brydenQuarters : ℕ := 5

/-- The percentage multiplier offered by the collector -/
def collectorMultiplier : ℚ := 25

/-- The total amount Bryden will receive in dollars -/
def brydenReceives : ℚ := brydenQuarters * quarterValue * collectorMultiplier

theorem bryden_receives_amount : brydenReceives = 125 / 4 := by sorry

end NUMINAMATH_CALUDE_bryden_receives_amount_l4062_406293


namespace NUMINAMATH_CALUDE_tan_strictly_increasing_interval_l4062_406223

theorem tan_strictly_increasing_interval (k : ℤ) :
  StrictMonoOn (fun x ↦ Real.tan (2 * x - π / 3))
    (Set.Ioo (k * π / 2 - π / 12) (k * π / 2 + 5 * π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_tan_strictly_increasing_interval_l4062_406223


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l4062_406251

theorem smallest_solution_of_equation (x : ℝ) :
  x^4 - 54*x^2 + 441 = 0 → x ≥ -Real.sqrt 33 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l4062_406251


namespace NUMINAMATH_CALUDE_distance_to_reflection_over_y_axis_l4062_406273

/-- Given a point P at (3, 5), prove that the distance between P and its reflection over the y-axis is 6 -/
theorem distance_to_reflection_over_y_axis :
  let P : ℝ × ℝ := (3, 5)
  let P' : ℝ × ℝ := (-P.1, P.2)
  Real.sqrt ((P'.1 - P.1)^2 + (P'.2 - P.2)^2) = 6 := by sorry

end NUMINAMATH_CALUDE_distance_to_reflection_over_y_axis_l4062_406273


namespace NUMINAMATH_CALUDE_prime_triplets_equation_l4062_406221

theorem prime_triplets_equation (p q r : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  (p : ℚ) / q = 8 / (r - 1 : ℚ) + 1 ↔ 
  ((p = 3 ∧ q = 2 ∧ r = 17) ∨ 
   (p = 7 ∧ q = 3 ∧ r = 7) ∨ 
   (p = 5 ∧ q = 3 ∧ r = 13)) :=
by sorry

end NUMINAMATH_CALUDE_prime_triplets_equation_l4062_406221


namespace NUMINAMATH_CALUDE_average_growth_rate_correct_l4062_406268

/-- The average monthly growth rate of profit from March to May -/
def average_growth_rate : ℝ := 0.2

/-- The profit in March -/
def profit_march : ℝ := 5000

/-- The profit in May -/
def profit_may : ℝ := 7200

/-- The number of months between March and May -/
def months_between : ℕ := 2

/-- Theorem stating that the average monthly growth rate is correct -/
theorem average_growth_rate_correct : 
  profit_march * (1 + average_growth_rate) ^ months_between = profit_may := by
  sorry

end NUMINAMATH_CALUDE_average_growth_rate_correct_l4062_406268


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l4062_406281

def population_size : Nat := 1000
def num_groups : Nat := 10
def sample_size : Nat := 10

def systematic_sample (x : Nat) : List Nat :=
  List.range num_groups |>.map (fun k => (x + 33 * k) % 100)

def last_two_digits (n : Nat) : Nat := n % 100

theorem systematic_sampling_theorem :
  (systematic_sample 24 = [24, 57, 90, 23, 56, 89, 22, 55, 88, 21]) ∧
  (∀ x : Nat, x < population_size →
    (∃ n ∈ systematic_sample x, last_two_digits n = 87) →
    x ∈ [21, 22, 23, 54, 55, 56, 87, 88, 89, 90]) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l4062_406281


namespace NUMINAMATH_CALUDE_volume_of_solid_T_l4062_406290

/-- The solid T is defined as the set of all points (x, y, z) in ℝ³ that satisfy
    the given inequalities. -/
def solid_T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | 
    let (x, y, z) := p
    (|x| + |y| ≤ 1.5) ∧ (|x| + |z| ≤ 1) ∧ (|y| + |z| ≤ 1)}

/-- The volume of a set in ℝ³. -/
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the volume of solid T is 2/3. -/
theorem volume_of_solid_T : volume solid_T = 2/3 := by sorry

end NUMINAMATH_CALUDE_volume_of_solid_T_l4062_406290


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_18n_integer_l4062_406277

theorem smallest_n_for_sqrt_18n_integer (n : ℕ) : 
  (∀ k : ℕ, 0 < k → k < 2 → ¬ ∃ m : ℕ, m^2 = 18 * k) ∧ 
  (∃ m : ℕ, m^2 = 18 * 2) → 
  n = 2 → 
  (∃ m : ℕ, m^2 = 18 * n) ∧ 
  (∀ k : ℕ, 0 < k → k < n → ¬ ∃ m : ℕ, m^2 = 18 * k) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_18n_integer_l4062_406277


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l4062_406242

theorem other_root_of_quadratic (c : ℝ) : 
  (3 : ℝ) ∈ {x : ℝ | x^2 - 5*x + c = 0} → 
  ∃ x : ℝ, x ≠ 3 ∧ x^2 - 5*x + c = 0 ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l4062_406242


namespace NUMINAMATH_CALUDE_min_value_theorem_l4062_406250

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : x + 2 * y = 3) :
  (x^2 + 3 * y) / (x * y) ≥ 2 * Real.sqrt 2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4062_406250


namespace NUMINAMATH_CALUDE_root_of_multiplicity_l4062_406232

theorem root_of_multiplicity (k : ℝ) : 
  (∃ x : ℝ, (x - 1) / (x - 3) = k / (x - 3) ∧ 
   ∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, |y - x| < δ → 
   |((y - 1) / (y - 3) - k / (y - 3))| < ε * |y - x|) ↔ 
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_root_of_multiplicity_l4062_406232


namespace NUMINAMATH_CALUDE_lives_lost_l4062_406234

theorem lives_lost (starting_lives ending_lives : ℕ) 
  (h1 : starting_lives = 98)
  (h2 : ending_lives = 73) :
  starting_lives - ending_lives = 25 := by
  sorry

end NUMINAMATH_CALUDE_lives_lost_l4062_406234


namespace NUMINAMATH_CALUDE_car_wash_solution_l4062_406257

/-- Represents the car wash problem --/
structure CarWash where
  car_price : ℕ
  truck_price : ℕ
  suv_price : ℕ
  total_raised : ℕ
  num_suvs : ℕ
  num_cars : ℕ

/-- The solution to the car wash problem --/
def solve_car_wash (cw : CarWash) : ℕ :=
  (cw.total_raised - cw.car_price * cw.num_cars - cw.suv_price * cw.num_suvs) / cw.truck_price

/-- Theorem stating the solution to the specific problem --/
theorem car_wash_solution :
  let cw : CarWash := {
    car_price := 5,
    truck_price := 6,
    suv_price := 7,
    total_raised := 100,
    num_suvs := 5,
    num_cars := 7
  }
  solve_car_wash cw = 5 := by
  sorry

end NUMINAMATH_CALUDE_car_wash_solution_l4062_406257


namespace NUMINAMATH_CALUDE_largest_three_digit_geometric_l4062_406220

def is_geometric_sequence (a b c : ℕ) : Prop :=
  ∃ (r : ℚ), r ≠ 0 ∧ b = a * r ∧ c = b * r

def digits_are_distinct (n : ℕ) : Prop :=
  let d₁ := n / 100
  let d₂ := (n / 10) % 10
  let d₃ := n % 10
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃

theorem largest_three_digit_geometric : ∀ n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  digits_are_distinct n ∧
  is_geometric_sequence (n / 100) ((n / 10) % 10) (n % 10) ∧
  n / 100 ≤ 8 →
  n ≤ 842 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_geometric_l4062_406220


namespace NUMINAMATH_CALUDE_first_player_advantage_l4062_406262

/-- Represents a chocolate bar game state -/
structure ChocolateBar :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a player in the game -/
inductive Player
  | First
  | Second

/-- The result of a game -/
structure GameResult :=
  (first_player_pieces : ℕ)
  (second_player_pieces : ℕ)

/-- A strategy for playing the game -/
def Strategy := ChocolateBar → Player → ChocolateBar

/-- Play the game with a given strategy -/
def play_game (initial : ChocolateBar) (strategy : Strategy) : GameResult :=
  sorry

/-- The optimal strategy for the first player -/
def optimal_strategy : Strategy :=
  sorry

/-- Theorem stating that the first player can get at least 6 more pieces -/
theorem first_player_advantage (initial : ChocolateBar) :
  initial.rows = 9 ∧ initial.cols = 6 →
  let result := play_game initial optimal_strategy
  result.first_player_pieces ≥ result.second_player_pieces + 6 :=
by sorry

end NUMINAMATH_CALUDE_first_player_advantage_l4062_406262


namespace NUMINAMATH_CALUDE_inequality_proof_l4062_406238

theorem inequality_proof (x : ℝ) (h1 : 3/2 ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4062_406238


namespace NUMINAMATH_CALUDE_derivative_f_at_4_l4062_406226

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt x

theorem derivative_f_at_4 : 
  deriv f 4 = -1/16 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_4_l4062_406226


namespace NUMINAMATH_CALUDE_min_rice_purchase_l4062_406222

theorem min_rice_purchase (o r : ℝ) 
  (h1 : o ≥ 4 + 2 * r) 
  (h2 : o ≤ 3 * r) : 
  r ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_rice_purchase_l4062_406222


namespace NUMINAMATH_CALUDE_total_sightings_l4062_406202

def animal_sightings (january february march : ℕ) : Prop :=
  february = 3 * january ∧ march = february / 2

theorem total_sightings (january : ℕ) (h : animal_sightings january (3 * january) ((3 * january) / 2)) :
  january + (3 * january) + ((3 * january) / 2) = 143 :=
by
  sorry

#check total_sightings 26

end NUMINAMATH_CALUDE_total_sightings_l4062_406202


namespace NUMINAMATH_CALUDE_janet_flight_cost_l4062_406296

/-- The cost of flying between two cities -/
def flying_cost (distance : ℝ) (cost_per_km : ℝ) (booking_fee : ℝ) : ℝ :=
  distance * cost_per_km + booking_fee

/-- Theorem: The cost for Janet to fly from City D to City E is $720 -/
theorem janet_flight_cost : 
  flying_cost 4750 0.12 150 = 720 := by
  sorry

end NUMINAMATH_CALUDE_janet_flight_cost_l4062_406296


namespace NUMINAMATH_CALUDE_nested_custom_op_equals_two_l4062_406258

/-- Custom operation [a,b,c] defined as (a+b)/c where c ≠ 0 -/
def custom_op (a b c : ℚ) : ℚ := (a + b) / c

/-- Theorem stating that [[72,18,90],[4,2,6],[12,6,18]] = 2 -/
theorem nested_custom_op_equals_two :
  custom_op (custom_op 72 18 90) (custom_op 4 2 6) (custom_op 12 6 18) = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_custom_op_equals_two_l4062_406258


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4062_406244

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  ((4 + 7 * i) / (4 - 7 * i) + (4 - 7 * i) / (4 + 7 * i)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4062_406244


namespace NUMINAMATH_CALUDE_factorial_sum_equals_720_l4062_406209

def factorial : ℕ → ℕ
| 0 => 1
| n + 1 => (n + 1) * factorial n

theorem factorial_sum_equals_720 : 
  5 * factorial 5 + 4 * factorial 4 + factorial 4 = 720 := by
sorry

end NUMINAMATH_CALUDE_factorial_sum_equals_720_l4062_406209


namespace NUMINAMATH_CALUDE_max_additional_plates_l4062_406276

/-- Represents the number of elements in each set of characters for license plates -/
def initial_sets : Fin 3 → Nat
  | 0 => 4  -- {B, G, J, S}
  | 1 => 2  -- {E, U}
  | 2 => 3  -- {K, V, X}
  | _ => 0

/-- Calculates the total number of license plate combinations -/
def total_combinations (sets : Fin 3 → Nat) : Nat :=
  (sets 0) * (sets 1) * (sets 2)

/-- Represents the addition of two new letters to the sets -/
structure NewLetterAddition where
  set1 : Nat  -- Number of letters added to set 1
  set2 : Nat  -- Number of letters added to set 2
  set3 : Nat  -- Number of letters added to set 3

/-- The theorem to be proved -/
theorem max_additional_plates :
  ∃ (addition : NewLetterAddition),
    addition.set1 + addition.set2 + addition.set3 = 2 ∧
    ∀ (other : NewLetterAddition),
      other.set1 + other.set2 + other.set3 = 2 →
      total_combinations (λ i => initial_sets i + other.set1) -
      total_combinations initial_sets ≤
      total_combinations (λ i => initial_sets i + addition.set1) -
      total_combinations initial_sets ∧
      total_combinations (λ i => initial_sets i + addition.set1) -
      total_combinations initial_sets = 24 :=
sorry

end NUMINAMATH_CALUDE_max_additional_plates_l4062_406276


namespace NUMINAMATH_CALUDE_champion_is_c_l4062_406200

-- Define the athletes
inductive Athlete : Type
| a : Athlete
| b : Athlete
| c : Athlete

-- Define the students
inductive Student : Type
| A : Student
| B : Student
| C : Student

-- Define the correctness of a statement
inductive Correctness : Type
| Correct : Correctness
| HalfCorrect : Correctness
| Incorrect : Correctness

-- Define the champion
def champion : Athlete := Athlete.c

-- Define the statements made by each student
def statement (s : Student) : Athlete × Athlete :=
  match s with
  | Student.A => (Athlete.b, Athlete.c)
  | Student.B => (Athlete.b, Athlete.a)
  | Student.C => (Athlete.c, Athlete.b)

-- Define the correctness of each student's statement
def studentCorrectness (s : Student) : Correctness :=
  match s with
  | Student.A => Correctness.Correct
  | Student.B => Correctness.HalfCorrect
  | Student.C => Correctness.Incorrect

-- Theorem to prove
theorem champion_is_c :
  (∀ s : Student, (statement s).1 ≠ champion → (statement s).2 = champion ↔ studentCorrectness s = Correctness.Correct) ∧
  (∃! s : Student, studentCorrectness s = Correctness.Correct) ∧
  (∃! s : Student, studentCorrectness s = Correctness.HalfCorrect) ∧
  (∃! s : Student, studentCorrectness s = Correctness.Incorrect) →
  champion = Athlete.c := by
  sorry

end NUMINAMATH_CALUDE_champion_is_c_l4062_406200


namespace NUMINAMATH_CALUDE_initial_tank_capacity_initial_tank_capacity_solution_l4062_406207

theorem initial_tank_capacity 
  (initial_tanks : ℕ) 
  (additional_tanks : ℕ) 
  (fish_per_additional_tank : ℕ) 
  (total_fish : ℕ) : ℕ :=
  let fish_in_additional_tanks := additional_tanks * fish_per_additional_tank
  let remaining_fish := total_fish - fish_in_additional_tanks
  remaining_fish / initial_tanks

theorem initial_tank_capacity_solution 
  (h1 : initial_tanks = 3)
  (h2 : additional_tanks = 3)
  (h3 : fish_per_additional_tank = 10)
  (h4 : total_fish = 75) :
  initial_tank_capacity initial_tanks additional_tanks fish_per_additional_tank total_fish = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_tank_capacity_initial_tank_capacity_solution_l4062_406207


namespace NUMINAMATH_CALUDE_parabola_shift_l4062_406261

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shift amount
def shift : ℝ := 1

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := (x + shift)^2

-- Theorem statement
theorem parabola_shift :
  ∀ x y : ℝ, y = original_parabola (x + shift) ↔ y = shifted_parabola x :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l4062_406261


namespace NUMINAMATH_CALUDE_intersection_point_of_g_and_inverse_l4062_406266

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 4*x^2 + 10*x + 20

-- State the theorem
theorem intersection_point_of_g_and_inverse :
  ∃! p : ℝ × ℝ, p.1 = g p.2 ∧ p.2 = g p.1 ∧ p = (-2, -2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_g_and_inverse_l4062_406266


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_neg_one_range_of_a_for_p_sufficient_not_necessary_for_q_l4062_406265

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 + 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - 6*x - 72 ≤ 0 ∧ x^2 + x - 6 > 0

-- Part 1: Range of x when a = -1
theorem range_of_x_when_a_is_neg_one :
  ∀ x : ℝ, (p x (-1) ∨ q x) ↔ x ∈ Set.Ioc (-6) (-3) ∪ Set.Icc 1 12 :=
sorry

-- Part 2: Range of a when p is sufficient but not necessary for q
theorem range_of_a_for_p_sufficient_not_necessary_for_q :
  {a : ℝ | ∀ x : ℝ, p x a → q x} ∩ {a : ℝ | ∃ x : ℝ, q x ∧ ¬p x a} = Set.Icc (-4) (-2) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_neg_one_range_of_a_for_p_sufficient_not_necessary_for_q_l4062_406265


namespace NUMINAMATH_CALUDE_inequality_solution_abs_inequality_l4062_406205

def f (x : ℝ) := |x - 2|

theorem inequality_solution :
  ∀ x : ℝ, (f x + f (x + 1) ≥ 5) ↔ (x ≥ 4 ∨ x ≤ -1) :=
sorry

theorem abs_inequality :
  ∀ a b : ℝ, |a| > 1 → |a*b - 2| > |a| * |b/a - 2| → |b| > 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_abs_inequality_l4062_406205


namespace NUMINAMATH_CALUDE_complex_power_six_l4062_406203

theorem complex_power_six : (1 + 2 * Complex.I) ^ 6 = 117 + 44 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_six_l4062_406203


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_approximate_roots_l4062_406227

/-- The quadratic equation √3x² + √17x - √6 = 0 has two real roots -/
theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ),
  Real.sqrt 3 * x₁^2 + Real.sqrt 17 * x₁ - Real.sqrt 6 = 0 ∧
  Real.sqrt 3 * x₂^2 + Real.sqrt 17 * x₂ - Real.sqrt 6 = 0 ∧
  x₁ ≠ x₂ :=
by sorry

/-- The roots of the equation √3x² + √17x - √6 = 0 are approximately 0.492 and -2.873 -/
theorem approximate_roots : ∃ (x₁ x₂ : ℝ),
  Real.sqrt 3 * x₁^2 + Real.sqrt 17 * x₁ - Real.sqrt 6 = 0 ∧
  Real.sqrt 3 * x₂^2 + Real.sqrt 17 * x₂ - Real.sqrt 6 = 0 ∧
  abs (x₁ - 0.492) < 0.0005 ∧
  abs (x₂ + 2.873) < 0.0005 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_approximate_roots_l4062_406227


namespace NUMINAMATH_CALUDE_no_linear_factor_l4062_406255

/-- The polynomial p(x,y,z) = x^2-y^2+2yz-z^2+2x-y-3z has no linear factor with integer coefficients. -/
theorem no_linear_factor (x y z : ℤ) : 
  ¬ ∃ (a b c d : ℤ), (a*x + b*y + c*z + d) ∣ (x^2 - y^2 + 2*y*z - z^2 + 2*x - y - 3*z) :=
sorry

end NUMINAMATH_CALUDE_no_linear_factor_l4062_406255


namespace NUMINAMATH_CALUDE_pieces_after_n_divisions_no_2009_pieces_l4062_406237

/-- Represents the number of pieces after n divisions -/
def num_pieces (n : ℕ) : ℕ := 3 * n + 1

/-- Theorem stating the number of pieces after n divisions -/
theorem pieces_after_n_divisions (n : ℕ) :
  num_pieces n = 3 * n + 1 := by sorry

/-- Theorem stating that it's impossible to have 2009 pieces -/
theorem no_2009_pieces :
  ¬ ∃ (n : ℕ), num_pieces n = 2009 := by sorry

end NUMINAMATH_CALUDE_pieces_after_n_divisions_no_2009_pieces_l4062_406237


namespace NUMINAMATH_CALUDE_complex_equidistant_modulus_l4062_406204

theorem complex_equidistant_modulus (z : ℂ) : 
  Complex.abs z = Complex.abs (z - 1) ∧ 
  Complex.abs z = Complex.abs (z - Complex.I) → 
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equidistant_modulus_l4062_406204


namespace NUMINAMATH_CALUDE_age_difference_l4062_406246

/-- Given three people a, b, and c, prove that a is 2 years older than b -/
theorem age_difference (a b c : ℕ) : 
  b = 2 * c →
  a + b + c = 22 →
  b = 8 →
  a - b = 2 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l4062_406246


namespace NUMINAMATH_CALUDE_decimal_to_octal_conversion_l4062_406249

/-- Converts a natural number to its octal representation -/
def toOctal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: toOctal (n / 8)

/-- The decimal number to be converted -/
def decimalNumber : ℕ := 521

/-- The expected octal representation -/
def expectedOctal : List ℕ := [1, 1, 0, 1]

theorem decimal_to_octal_conversion :
  toOctal decimalNumber = expectedOctal := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_octal_conversion_l4062_406249


namespace NUMINAMATH_CALUDE_recover_sequence_l4062_406254

/-- A sequence of six positive integers in arithmetic progression. -/
def ArithmeticSequence : Type := Fin 6 → ℕ+

/-- The given sequence with one number omitted and one miscopied. -/
def GivenSequence : Fin 5 → ℕ+ := ![113, 137, 149, 155, 173]

/-- The correct sequence. -/
def CorrectSequence : ArithmeticSequence := ![113, 125, 137, 149, 161, 173]

/-- Checks if a sequence is in arithmetic progression. -/
def isArithmeticProgression (s : ArithmeticSequence) : Prop :=
  ∃ d : ℕ+, ∀ i : Fin 5, s (i + 1) = s i + d

/-- Checks if a sequence matches the given sequence except for one miscopied number. -/
def matchesGivenSequence (s : ArithmeticSequence) : Prop :=
  ∃ j : Fin 5, ∀ i : Fin 5, i ≠ j → s i = GivenSequence i

theorem recover_sequence :
  isArithmeticProgression CorrectSequence ∧
  matchesGivenSequence CorrectSequence :=
sorry

end NUMINAMATH_CALUDE_recover_sequence_l4062_406254


namespace NUMINAMATH_CALUDE_collinear_points_theorem_l4062_406274

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def is_collinear (v w : V) : Prop := ∃ (r : ℝ), v = r • w

theorem collinear_points_theorem 
  (e₁ e₂ : V) 
  (h_noncollinear : ¬ is_collinear e₁ e₂)
  (k : ℝ)
  (AB CB CD : V)
  (h_AB : AB = e₁ - k • e₂)
  (h_CB : CB = 2 • e₁ + e₂)
  (h_CD : CD = 3 • e₁ - e₂)
  (h_collinear : is_collinear AB (CD - CB)) :
  k = 2 := by sorry

end NUMINAMATH_CALUDE_collinear_points_theorem_l4062_406274


namespace NUMINAMATH_CALUDE_eric_required_bike_speed_l4062_406264

/-- Represents the triathlon components --/
structure Triathlon :=
  (swim_distance : ℚ)
  (swim_speed : ℚ)
  (run_distance : ℚ)
  (run_speed : ℚ)
  (bike_distance : ℚ)
  (total_time : ℚ)

/-- Calculates the required bike speed for a given triathlon --/
def required_bike_speed (t : Triathlon) : ℚ :=
  let swim_time := t.swim_distance / t.swim_speed
  let run_time := t.run_distance / t.run_speed
  let bike_time := t.total_time - (swim_time + run_time)
  t.bike_distance / bike_time

/-- The triathlon problem --/
def eric_triathlon : Triathlon :=
  { swim_distance := 1/4
  , swim_speed := 2
  , run_distance := 3
  , run_speed := 6
  , bike_distance := 15
  , total_time := 2 }

/-- Theorem stating that the required bike speed for Eric's triathlon is 120/11 --/
theorem eric_required_bike_speed :
  required_bike_speed eric_triathlon = 120/11 := by sorry


end NUMINAMATH_CALUDE_eric_required_bike_speed_l4062_406264


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l4062_406218

theorem simplify_sqrt_expression (y : ℝ) (h : y ≠ 0) : 
  Real.sqrt (4 + ((y^3 - 2) / (3 * y))^2) = (Real.sqrt (y^6 - 4*y^3 + 36*y^2 + 4)) / (3 * y) :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l4062_406218


namespace NUMINAMATH_CALUDE_train_platform_time_l4062_406252

/-- Given a train of length 1500 meters that crosses a tree in 100 seconds,
    calculate the time taken to pass a platform of length 500 meters. -/
theorem train_platform_time (train_length platform_length tree_crossing_time : ℝ)
    (h1 : train_length = 1500)
    (h2 : platform_length = 500)
    (h3 : tree_crossing_time = 100) :
    (train_length + platform_length) / (train_length / tree_crossing_time) = 400/3 := by
  sorry

#eval (1500 + 500) / (1500 / 100) -- Should output approximately 133.33333333

end NUMINAMATH_CALUDE_train_platform_time_l4062_406252


namespace NUMINAMATH_CALUDE_ryan_overall_score_l4062_406241

def first_test_questions : ℕ := 30
def first_test_score : ℚ := 85 / 100

def second_test_math_questions : ℕ := 20
def second_test_math_score : ℚ := 95 / 100
def second_test_science_questions : ℕ := 15
def second_test_science_score : ℚ := 80 / 100

def third_test_questions : ℕ := 15
def third_test_score : ℚ := 65 / 100

theorem ryan_overall_score :
  let total_questions := first_test_questions + second_test_math_questions + second_test_science_questions + third_test_questions
  let correct_answers := (first_test_questions : ℚ) * first_test_score +
                         (second_test_math_questions : ℚ) * second_test_math_score +
                         (second_test_science_questions : ℚ) * second_test_science_score +
                         (third_test_questions : ℚ) * third_test_score
  correct_answers / (total_questions : ℚ) = 8281 / 10000 :=
by sorry

end NUMINAMATH_CALUDE_ryan_overall_score_l4062_406241


namespace NUMINAMATH_CALUDE_fraction_unchanged_l4062_406247

theorem fraction_unchanged (x y : ℝ) : 
  (2*x) * (2*y) / ((2*x)^2 - (2*y)^2) = x * y / (x^2 - y^2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l4062_406247


namespace NUMINAMATH_CALUDE_sequence_sum_property_l4062_406284

/-- Given a positive sequence {a_n}, prove that a_n = 2n - 1 for all positive integers n,
    where S_n = (a_n + 1)^2 / 4 is the sum of the first n terms. -/
theorem sequence_sum_property (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ n, S n = (a n + 1)^2 / 4) →
  ∀ n, a n = 2 * n - 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_property_l4062_406284


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt5_l4062_406270

theorem rationalize_denominator_sqrt5 : 
  ∃ (A B C : ℤ), 
    (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = A + B * Real.sqrt C ∧ 
    A * B * C = 275 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt5_l4062_406270


namespace NUMINAMATH_CALUDE_trigonometric_identity_l4062_406248

theorem trigonometric_identity (α β : Real) 
  (h : (Real.sin α)^2 / (Real.cos β)^2 + (Real.cos α)^2 / (Real.sin β)^2 = 4) :
  (Real.cos β)^2 / (Real.sin α)^2 + (Real.sin β)^2 / (Real.cos α)^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l4062_406248


namespace NUMINAMATH_CALUDE_chocolate_distribution_l4062_406229

theorem chocolate_distribution (x y : ℕ) : 
  (y = x + 1) →  -- If each person is given 1 chocolate, then 1 chocolate is left
  (y = 2 * (x - 1)) →  -- If each person is given 2 chocolates, then 1 person will be left
  (x + y = 7) :=  -- The sum of persons and chocolates is 7
by
  sorry

#check chocolate_distribution

end NUMINAMATH_CALUDE_chocolate_distribution_l4062_406229


namespace NUMINAMATH_CALUDE_largest_N_is_120_l4062_406201

/-- A type representing a 6 × N table with entries from 1 to 6 -/
def Table (N : ℕ) := Fin 6 → Fin N → Fin 6

/-- Predicate to check if a column is a permutation of 1 to 6 -/
def IsPermutation (t : Table N) (col : Fin N) : Prop :=
  ∀ i : Fin 6, ∃ j : Fin 6, t j col = i

/-- Predicate to check if any two columns have a common entry in some row -/
def HasCommonEntry (t : Table N) : Prop :=
  ∀ i j : Fin N, i ≠ j → ∃ r : Fin 6, t r i = t r j

/-- Predicate to check if any two columns have a different entry in some row -/
def HasDifferentEntry (t : Table N) : Prop :=
  ∀ i j : Fin N, i ≠ j → ∃ s : Fin 6, t s i ≠ t s j

/-- The main theorem stating the largest possible N -/
theorem largest_N_is_120 :
  (∃ N : ℕ, N > 0 ∧ ∃ t : Table N,
    (∀ col, IsPermutation t col) ∧
    HasCommonEntry t ∧
    HasDifferentEntry t) ∧
  (∀ M : ℕ, M > 120 →
    ¬∃ t : Table M,
      (∀ col, IsPermutation t col) ∧
      HasCommonEntry t ∧
      HasDifferentEntry t) :=
sorry

end NUMINAMATH_CALUDE_largest_N_is_120_l4062_406201


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l4062_406286

/-- An arithmetic sequence of positive terms -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem min_value_of_reciprocal_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n, a n > 0) →
  a 1 + a 2015 = 2 →
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 2) →
  ∃ m : ℝ, m = 1/a 2 + 1/a 2014 ∧ m ≥ 2 ∧ ∀ z, z = 1/a 2 + 1/a 2014 → z ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l4062_406286


namespace NUMINAMATH_CALUDE_alligator_population_after_year_l4062_406295

/-- The number of alligators after a given number of 6-month periods -/
def alligator_population (initial_population : ℕ) (periods : ℕ) : ℕ :=
  initial_population * 2^periods

/-- Theorem stating that given 4 initial alligators and population doubling every 6 months, 
    there will be 16 alligators after 1 year -/
theorem alligator_population_after_year (initial_population : ℕ) 
  (h1 : initial_population = 4) : alligator_population initial_population 2 = 16 := by
  sorry

#check alligator_population_after_year

end NUMINAMATH_CALUDE_alligator_population_after_year_l4062_406295


namespace NUMINAMATH_CALUDE_correct_both_problems_l4062_406243

theorem correct_both_problems (total : ℕ) (sets_correct : ℕ) (functions_correct : ℕ) (both_wrong : ℕ) : 
  total = 50 ∧ sets_correct = 40 ∧ functions_correct = 31 ∧ both_wrong = 4 →
  ∃ (both_correct : ℕ), both_correct = 29 ∧ 
    total = sets_correct + functions_correct - both_correct + both_wrong :=
by
  sorry


end NUMINAMATH_CALUDE_correct_both_problems_l4062_406243


namespace NUMINAMATH_CALUDE_opposite_of_2023_l4062_406214

-- Define the concept of opposite for integers
def opposite (n : ℤ) : ℤ := -n

-- Theorem statement
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l4062_406214


namespace NUMINAMATH_CALUDE_solution_when_m_3_no_solution_conditions_l4062_406271

-- Define the fractional equation
def fractional_equation (x m : ℝ) : Prop :=
  (3 - 2*x) / (x - 2) - (m*x - 2) / (2 - x) = -1

-- Theorem 1: When m = 3, the solution is x = 1/2
theorem solution_when_m_3 :
  ∃ x : ℝ, fractional_equation x 3 ∧ x = 1/2 :=
sorry

-- Theorem 2: The equation has no solution when m = 1 or m = 3/2
theorem no_solution_conditions :
  (∀ x : ℝ, ¬ fractional_equation x 1) ∧
  (∀ x : ℝ, ¬ fractional_equation x (3/2)) :=
sorry

end NUMINAMATH_CALUDE_solution_when_m_3_no_solution_conditions_l4062_406271


namespace NUMINAMATH_CALUDE_truck_transport_time_l4062_406291

theorem truck_transport_time (total_time : ℝ) (first_truck_portion : ℝ) (actual_time : ℝ)
  (h1 : total_time = 6)
  (h2 : first_truck_portion = 3/5)
  (h3 : actual_time = 12) :
  ∃ (t1 t2 : ℝ),
    ((t1 = 10 ∧ t2 = 15) ∨ (t1 = 12 ∧ t2 = 12)) ∧
    (1 / t1 + 1 / t2 = 1 / total_time) ∧
    (first_truck_portion / t1 + (1 - first_truck_portion) / t2 = 1 / actual_time) := by
  sorry

end NUMINAMATH_CALUDE_truck_transport_time_l4062_406291


namespace NUMINAMATH_CALUDE_log_product_equals_two_l4062_406297

theorem log_product_equals_two (k x : ℝ) (h : k > 0) (h' : x > 0) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 10) = 2 →
  x = 100 := by
sorry

end NUMINAMATH_CALUDE_log_product_equals_two_l4062_406297


namespace NUMINAMATH_CALUDE_scientist_contemporary_probability_scientist_contemporary_probability_value_l4062_406294

/-- The probability that two scientists were contemporaries for any length of time -/
theorem scientist_contemporary_probability : ℝ :=
  let years_range : ℕ := 300
  let lifespan : ℕ := 80
  let total_possibility_area : ℕ := years_range * years_range
  let overlap_area : ℕ := (years_range - lifespan) * (years_range - lifespan) - 2 * (lifespan * lifespan / 2)
  (overlap_area : ℝ) / total_possibility_area

/-- The probability is equal to 7/15 -/
theorem scientist_contemporary_probability_value : scientist_contemporary_probability = 7 / 15 :=
sorry

end NUMINAMATH_CALUDE_scientist_contemporary_probability_scientist_contemporary_probability_value_l4062_406294


namespace NUMINAMATH_CALUDE_absolute_difference_100th_terms_l4062_406230

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem absolute_difference_100th_terms :
  let C := arithmetic_sequence 35 7
  let D := arithmetic_sequence 35 (-7)
  |C 100 - D 100| = 1386 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_100th_terms_l4062_406230


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l4062_406280

theorem boys_to_girls_ratio (total_students : ℕ) (boys : ℕ) 
  (h1 : total_students = 48) (h2 : boys = 42) : 
  (boys : ℚ) / (total_students - boys : ℚ) = 7 / 1 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l4062_406280


namespace NUMINAMATH_CALUDE_double_divide_four_equals_twelve_l4062_406216

theorem double_divide_four_equals_twelve (x : ℝ) : (2 * x) / 4 = 12 → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_double_divide_four_equals_twelve_l4062_406216


namespace NUMINAMATH_CALUDE_problem_solution_l4062_406267

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = -3) :
  x + x^3 / y^2 + y^3 / x^2 + y = 590 + 5/9 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4062_406267


namespace NUMINAMATH_CALUDE_area_is_nine_l4062_406289

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D plane defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Triangular region formed by two lines and x-axis -/
structure TriangularRegion where
  line1 : Line
  line2 : Line

def line1 : Line := { p1 := ⟨0, 3⟩, p2 := ⟨6, 0⟩ }
def line2 : Line := { p1 := ⟨1, 6⟩, p2 := ⟨7, 1⟩ }

def region : TriangularRegion := { line1 := line1, line2 := line2 }

/-- Calculate the area of the triangular region -/
def calculateArea (r : TriangularRegion) : ℝ :=
  sorry

theorem area_is_nine : calculateArea region = 9 := by
  sorry

end NUMINAMATH_CALUDE_area_is_nine_l4062_406289


namespace NUMINAMATH_CALUDE_lcm_12_35_l4062_406245

theorem lcm_12_35 : Nat.lcm 12 35 = 420 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_35_l4062_406245


namespace NUMINAMATH_CALUDE_trigonometric_identity_l4062_406275

theorem trigonometric_identity (α : ℝ) (h : (1 + Real.cos α) / Real.sin α = 1 / 2) :
  Real.cos α + 2 * Real.sin α = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l4062_406275


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_fourth_l4062_406206

theorem opposite_of_negative_one_fourth :
  -((-1 : ℚ) / 4) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_fourth_l4062_406206


namespace NUMINAMATH_CALUDE_no_common_divisor_l4062_406228

theorem no_common_divisor (a b n : ℕ) 
  (ha : a > 1) 
  (hb : b > 1)
  (hn : n > 0)
  (div_a : a ∣ (2^n - 1))
  (div_b : b ∣ (2^n + 1)) :
  ¬∃ k : ℕ, (a ∣ (2^k + 1)) ∧ (b ∣ (2^k - 1)) :=
sorry

end NUMINAMATH_CALUDE_no_common_divisor_l4062_406228


namespace NUMINAMATH_CALUDE_correct_remaining_money_l4062_406279

def remaining_money (olivia_initial : ℕ) (nigel_initial : ℕ) (num_passes : ℕ) (cost_per_pass : ℕ) : ℕ :=
  olivia_initial + nigel_initial - num_passes * cost_per_pass

theorem correct_remaining_money :
  remaining_money 112 139 6 28 = 83 := by
  sorry

end NUMINAMATH_CALUDE_correct_remaining_money_l4062_406279


namespace NUMINAMATH_CALUDE_curve_properties_l4062_406233

-- Define the curve equation
def curve_equation (x y t : ℝ) : Prop :=
  x^2 / (4 - t) + y^2 / (t - 1) = 1

-- Define what it means for the curve to be a hyperbola
def is_hyperbola (t : ℝ) : Prop :=
  t < 1 ∨ t > 4

-- Define what it means for the curve to be an ellipse with foci on the x-axis
def is_ellipse_x_foci (t : ℝ) : Prop :=
  1 < t ∧ t < 5/2

-- State the theorem
theorem curve_properties :
  ∀ t : ℝ,
    (∀ x y : ℝ, curve_equation x y t → is_hyperbola t) ∧
    (∀ x y : ℝ, curve_equation x y t → is_ellipse_x_foci t) :=
by sorry

end NUMINAMATH_CALUDE_curve_properties_l4062_406233


namespace NUMINAMATH_CALUDE_range_of_a_l4062_406285

-- Define the function f(x) = x^2 - 2x + a
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + a

-- State the theorem
theorem range_of_a (h : ∀ x ∈ Set.Icc 2 3, f a x > 0) : a > 0 := by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l4062_406285


namespace NUMINAMATH_CALUDE_cone_base_radius_l4062_406292

theorem cone_base_radius (S : ℝ) (r : ℝ) : 
  S = 9 * Real.pi → -- Surface area is 9π cm²
  S = 3 * Real.pi * r^2 → -- Surface area formula for a cone with semicircular lateral surface
  r = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cone_base_radius_l4062_406292


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4062_406278

/-- Proves that for an increasing geometric sequence with a_3 = 8 and S_3 = 14,
    the common ratio is 2. -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)  -- The geometric sequence
  (h_incr : ∀ n, a n < a (n + 1))  -- The sequence is increasing
  (h_a3 : a 3 = 8)  -- Third term is 8
  (h_S3 : (a 1) + (a 2) + (a 3) = 14)  -- Sum of first 3 terms is 14
  : ∃ q : ℝ, (∀ n, a (n + 1) = q * a n) ∧ q = 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4062_406278


namespace NUMINAMATH_CALUDE_comic_stacking_arrangements_l4062_406224

def batman_comics : ℕ := 8
def superman_comics : ℕ := 6
def wonder_woman_comics : ℕ := 3

theorem comic_stacking_arrangements :
  (batman_comics.factorial * superman_comics.factorial * wonder_woman_comics.factorial) *
  (batman_comics + superman_comics + wonder_woman_comics).choose 3 = 1040486400 :=
by sorry

end NUMINAMATH_CALUDE_comic_stacking_arrangements_l4062_406224


namespace NUMINAMATH_CALUDE_problem_solution_l4062_406283

theorem problem_solution :
  (12 / 60 = 0.2) ∧
  (0.2 = 4 / 20) ∧
  (0.2 = 20 / 100) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4062_406283


namespace NUMINAMATH_CALUDE_probability_under20_is_one_tenth_l4062_406213

/-- Represents a group of people with age distribution --/
structure AgeGroup where
  total : ℕ
  over30 : ℕ
  under20 : ℕ
  h1 : total = over30 + under20
  h2 : over30 < total

/-- Calculates the probability of selecting a person under 20 from the group --/
def probabilityUnder20 (group : AgeGroup) : ℚ :=
  group.under20 / group.total

/-- The main theorem to prove --/
theorem probability_under20_is_one_tenth
  (group : AgeGroup)
  (h3 : group.total = 100)
  (h4 : group.over30 = 90) :
  probabilityUnder20 group = 1/10 := by
  sorry


end NUMINAMATH_CALUDE_probability_under20_is_one_tenth_l4062_406213


namespace NUMINAMATH_CALUDE_two_different_expressions_equal_seven_l4062_406253

/-- An arithmetic expression using digits of 4 and basic operations -/
inductive Expr
  | four : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an arithmetic expression -/
def eval : Expr → ℚ
  | Expr.four => 4
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Count the number of 4's used in an expression -/
def count_fours : Expr → ℕ
  | Expr.four => 1
  | Expr.add e1 e2 => count_fours e1 + count_fours e2
  | Expr.sub e1 e2 => count_fours e1 + count_fours e2
  | Expr.mul e1 e2 => count_fours e1 + count_fours e2
  | Expr.div e1 e2 => count_fours e1 + count_fours e2

/-- Check if two expressions are equivalent under commutative and associative properties -/
def are_equivalent : Expr → Expr → Prop := sorry

theorem two_different_expressions_equal_seven :
  ∃ (e1 e2 : Expr),
    eval e1 = 7 ∧
    eval e2 = 7 ∧
    count_fours e1 = 4 ∧
    count_fours e2 = 4 ∧
    ¬(are_equivalent e1 e2) :=
  sorry

end NUMINAMATH_CALUDE_two_different_expressions_equal_seven_l4062_406253


namespace NUMINAMATH_CALUDE_angle_x_value_l4062_406211

theorem angle_x_value (equilateral_angle : ℝ) (isosceles_vertex : ℝ) (straight_line_sum : ℝ) :
  equilateral_angle = 60 →
  isosceles_vertex = 30 →
  straight_line_sum = 180 →
  ∃ x y : ℝ,
    y + y + isosceles_vertex = straight_line_sum ∧
    x + y + equilateral_angle = straight_line_sum ∧
    x = 45 :=
by sorry

end NUMINAMATH_CALUDE_angle_x_value_l4062_406211


namespace NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l4062_406240

/-- A function that returns the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 13 -/
def nthNumberWithDigitSum13 (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 11th number with digit sum 13 is 166 -/
theorem eleventh_number_with_digit_sum_13 : nthNumberWithDigitSum13 11 = 166 := by sorry

end NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l4062_406240


namespace NUMINAMATH_CALUDE_expression_evaluation_l4062_406210

theorem expression_evaluation :
  let x : ℝ := 3
  let y : ℝ := Real.sqrt 3
  (x - 2*y)^2 - (x + 2*y)*(x - 2*y) + 4*x*y = 24 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4062_406210


namespace NUMINAMATH_CALUDE_reflection_squared_is_identity_l4062_406231

open Matrix

-- Define a reflection matrix over a non-zero vector
def reflection_matrix (v : Fin 2 → ℝ) (h : v ≠ 0) : Matrix (Fin 2) (Fin 2) ℝ := sorry

-- Theorem: The square of a reflection matrix is the identity matrix
theorem reflection_squared_is_identity 
  (v : Fin 2 → ℝ) (h : v ≠ 0) :
  (reflection_matrix v h) ^ 2 = 1 := by sorry

end NUMINAMATH_CALUDE_reflection_squared_is_identity_l4062_406231


namespace NUMINAMATH_CALUDE_apples_left_proof_l4062_406236

/-- The number of apples left when the farmer's children got home -/
def apples_left (num_children : ℕ) (apples_per_child : ℕ) (children_who_ate : ℕ) 
  (apples_eaten_per_child : ℕ) (apples_sold : ℕ) : ℕ :=
  num_children * apples_per_child - (children_who_ate * apples_eaten_per_child + apples_sold)

/-- Theorem stating the number of apples left when the farmer's children got home -/
theorem apples_left_proof : 
  apples_left 5 15 2 4 7 = 60 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_proof_l4062_406236


namespace NUMINAMATH_CALUDE_certain_number_proof_l4062_406215

theorem certain_number_proof : ∃ x : ℝ, (0.60 * x = 0.50 * 600) ∧ (x = 500) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l4062_406215


namespace NUMINAMATH_CALUDE_milk_remainder_l4062_406272

theorem milk_remainder (initial_milk : ℚ) (given_away : ℚ) (remainder : ℚ) : 
  initial_milk = 4 → given_away = 7/3 → remainder = initial_milk - given_away → remainder = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_milk_remainder_l4062_406272


namespace NUMINAMATH_CALUDE_action_figure_cost_l4062_406299

theorem action_figure_cost (current : ℕ) (total : ℕ) (cost : ℕ) : 
  current = 7 → total = 16 → cost = 72 → 
  (cost : ℚ) / ((total : ℚ) - (current : ℚ)) = 8 := by sorry

end NUMINAMATH_CALUDE_action_figure_cost_l4062_406299


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l4062_406212

theorem arithmetic_calculations :
  (-1^2 + |(-3)| + 5 / (-5) = 1) ∧
  (2 * (-3)^2 + 24 * (1/4 - 3/8 - 1/12) = 4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l4062_406212


namespace NUMINAMATH_CALUDE_chinese_chess_pieces_sum_l4062_406225

theorem chinese_chess_pieces_sum :
  ∀ (Rook Knight Cannon : ℕ),
    Rook / Knight = 2 →
    Cannon / Rook = 4 →
    Cannon - Knight = 56 →
    Rook + Knight + Cannon = 88 :=
by
  sorry

end NUMINAMATH_CALUDE_chinese_chess_pieces_sum_l4062_406225


namespace NUMINAMATH_CALUDE_vanessa_chocolate_sales_l4062_406263

/-- Calculates the money made from selling chocolate bars -/
def money_made (total_bars : ℕ) (price_per_bar : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

/-- Proves that Vanessa made $16 from selling chocolate bars -/
theorem vanessa_chocolate_sales :
  money_made 11 4 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_chocolate_sales_l4062_406263


namespace NUMINAMATH_CALUDE_even_function_comparison_l4062_406269

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * m * x + 3

theorem even_function_comparison (m : ℝ) (h : ∀ x, f m x = f m (-x)) :
  f m (-Real.pi) < f m 3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_comparison_l4062_406269
