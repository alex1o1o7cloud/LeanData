import Mathlib

namespace part_one_part_two_l3814_381422

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem part_one :
  let f := f 2
  {x : ℝ | f x ≥ 3 - |x - 1|} = {x : ℝ | x ≤ 0 ∨ x ≥ 3} := by sorry

-- Part 2
theorem part_two :
  ∀ a m n : ℝ,
  m > 0 → n > 0 →
  m + 2*n = a →
  ({x : ℝ | f a x ≤ 1} = {x : ℝ | 2 ≤ x ∧ x ≤ 4}) →
  ∃ (min : ℝ), min = 9/2 ∧ ∀ m' n' : ℝ, m' > 0 → n' > 0 → m' + 2*n' = a → m'^2 + 4*n'^2 ≥ min := by sorry

end part_one_part_two_l3814_381422


namespace chord_length_l3814_381468

/-- The polar equation of line l is √3ρcosθ + ρsinθ - 1 = 0 -/
def line_l (ρ θ : ℝ) : Prop :=
  Real.sqrt 3 * ρ * Real.cos θ + ρ * Real.sin θ - 1 = 0

/-- The polar equation of curve C is ρ = 4 -/
def curve_C (ρ : ℝ) : Prop := ρ = 4

/-- The length of the chord formed by the intersection of l and C is 3√7 -/
theorem chord_length : 
  ∃ (A B : ℝ × ℝ), 
    (∃ (ρ_A θ_A ρ_B θ_B : ℝ), 
      line_l ρ_A θ_A ∧ line_l ρ_B θ_B ∧ 
      curve_C ρ_A ∧ curve_C ρ_B ∧
      A = (ρ_A * Real.cos θ_A, ρ_A * Real.sin θ_A) ∧
      B = (ρ_B * Real.cos θ_B, ρ_B * Real.sin θ_B)) →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 * Real.sqrt 7 :=
sorry

end chord_length_l3814_381468


namespace power_inequality_l3814_381405

theorem power_inequality : 0.1^0.8 < 0.2^0.8 := by
  sorry

end power_inequality_l3814_381405


namespace periodicity_theorem_l3814_381439

/-- A polynomial with rational coefficients -/
def RationalPolynomial := Polynomial ℚ

/-- A sequence of rational numbers -/
def RationalSequence := ℕ → ℚ

/-- The statement of the periodicity theorem -/
theorem periodicity_theorem
  (p : RationalPolynomial)
  (q : RationalSequence)
  (h1 : p.degree ≥ 2)
  (h2 : ∀ n : ℕ, n ≥ 1 → q n = p.eval (q (n + 1))) :
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, n ≥ 1 → q (n + k) = q n :=
sorry

end periodicity_theorem_l3814_381439


namespace units_digit_product_l3814_381447

theorem units_digit_product : (47 * 23 * 89) % 10 = 9 := by
  sorry

end units_digit_product_l3814_381447


namespace triangle_problem_l3814_381496

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The conditions given in the problem -/
def satisfiesConditions (t : Triangle) : Prop :=
  8 * t.a * t.b * Real.sin t.C = 3 * (t.b^2 + t.c^2 - t.a^2) ∧
  t.a = Real.sqrt 10 ∧
  t.c = 5

/-- The theorem to be proved -/
theorem triangle_problem (t : Triangle) (h : satisfiesConditions t) :
  Real.cos t.A = 4/5 ∧
  (t.a * t.b * Real.sin t.C / 2 = 15/2 ∨ t.a * t.b * Real.sin t.C / 2 = 9/2) :=
by sorry

end triangle_problem_l3814_381496


namespace range_of_g_l3814_381481

theorem range_of_g : ∀ x : ℝ, 
  (3/4 : ℝ) ≤ (Real.cos x)^4 + (Real.sin x)^2 ∧ 
  (Real.cos x)^4 + (Real.sin x)^2 ≤ 1 ∧
  ∃ y z : ℝ, (Real.cos y)^4 + (Real.sin y)^2 = (3/4 : ℝ) ∧
             (Real.cos z)^4 + (Real.sin z)^2 = 1 :=
by sorry

end range_of_g_l3814_381481


namespace custom_multiplication_l3814_381454

theorem custom_multiplication (a b : ℤ) : a * b = a^2 + a*b - b^2 → 5 * (-3) = 1 := by
  sorry

end custom_multiplication_l3814_381454


namespace triangular_coin_array_l3814_381462

theorem triangular_coin_array (N : ℕ) : (N * (N + 1)) / 2 = 3003 → N = 77 := by
  sorry

end triangular_coin_array_l3814_381462


namespace zeros_not_adjacent_probability_l3814_381470

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 3

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 3

/-- The total number of digits in the arrangement -/
def total_digits : ℕ := num_ones + num_zeros

/-- The probability that the zeros are not adjacent when randomly arranged -/
def prob_zeros_not_adjacent : ℚ := 1 / 5

/-- Theorem stating that the probability of zeros not being adjacent is 1/5 -/
theorem zeros_not_adjacent_probability :
  prob_zeros_not_adjacent = 1 / 5 := by
  sorry

end zeros_not_adjacent_probability_l3814_381470


namespace ellipse_condition_l3814_381415

def is_ellipse (m : ℝ) : Prop :=
  m + 3 > 0 ∧ m - 1 > 0

theorem ellipse_condition (m : ℝ) :
  (m > -3 → is_ellipse m) ∧ ¬(is_ellipse m → m > -3) :=
sorry

end ellipse_condition_l3814_381415


namespace trig_identities_l3814_381492

theorem trig_identities (α : Real) (h : Real.tan α = 3) : 
  ((Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) ∧
  (1 / (Real.sin α ^ 2 - Real.sin α * Real.cos α - 2 * Real.cos α ^ 2) = 2) := by
  sorry

end trig_identities_l3814_381492


namespace base_12_division_remainder_l3814_381410

def base_12_to_decimal (n : ℕ) : ℕ :=
  1 * 12^3 + 5 * 12^2 + 3 * 12 + 4

theorem base_12_division_remainder :
  (base_12_to_decimal 1534) % 9 = 2 := by
sorry

end base_12_division_remainder_l3814_381410


namespace tangent_line_sum_range_l3814_381416

noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem tangent_line_sum_range (x₀ : ℝ) (h : x₀ > 0) :
  let k := 1 / x₀
  let b := Real.log x₀ - 1
  ∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, x > 0 ∧ k + b = y :=
sorry

end tangent_line_sum_range_l3814_381416


namespace total_fruit_salads_l3814_381417

/-- The total number of fruit salads in three restaurants -/
theorem total_fruit_salads (alaya angel betty : ℕ) : 
  alaya = 200 →
  angel = 2 * alaya →
  betty = 3 * angel →
  alaya + angel + betty = 1800 :=
by
  sorry

end total_fruit_salads_l3814_381417


namespace age_difference_l3814_381493

/-- Given the age ratios and sum of ages, prove the difference between Patrick's and Monica's ages --/
theorem age_difference (patrick_age michael_age monica_age : ℕ) : 
  patrick_age * 5 = michael_age * 3 →
  michael_age * 5 = monica_age * 3 →
  patrick_age + michael_age + monica_age = 147 →
  monica_age - patrick_age = 48 :=
by
  sorry

end age_difference_l3814_381493


namespace train_passing_tree_l3814_381418

/-- Proves that a train of given length and speed takes the calculated time to pass a tree -/
theorem train_passing_tree (train_length : ℝ) (train_speed_km_hr : ℝ) (time : ℝ) :
  train_length = 175 →
  train_speed_km_hr = 63 →
  time = train_length / (train_speed_km_hr * (1000 / 3600)) →
  time = 10 := by
  sorry

end train_passing_tree_l3814_381418


namespace officers_on_duty_l3814_381409

theorem officers_on_duty (total_female_officers : ℕ) 
  (female_on_duty_ratio : ℚ) (female_ratio_on_duty : ℚ) :
  total_female_officers = 250 →
  female_on_duty_ratio = 1/5 →
  female_ratio_on_duty = 1/2 →
  (female_on_duty_ratio * total_female_officers : ℚ) / female_ratio_on_duty = 100 := by
  sorry

end officers_on_duty_l3814_381409


namespace unique_solution_implies_prime_l3814_381436

theorem unique_solution_implies_prime (n : ℕ) :
  (∃! (x y : ℕ), (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / n) →
  Nat.Prime n :=
by sorry

end unique_solution_implies_prime_l3814_381436


namespace martha_apples_problem_l3814_381499

/-- Given that Martha has 20 apples initially, gives 5 to Jane and 2 more than that to James,
    prove that she needs to give away 4 more apples to be left with exactly 4 apples. -/
theorem martha_apples_problem (initial_apples : ℕ) (jane_apples : ℕ) (james_extra_apples : ℕ) 
  (h1 : initial_apples = 20)
  (h2 : jane_apples = 5)
  (h3 : james_extra_apples = 2) :
  initial_apples - jane_apples - (jane_apples + james_extra_apples) - 4 = 4 := by
  sorry

#check martha_apples_problem

end martha_apples_problem_l3814_381499


namespace sharp_four_times_100_l3814_381498

-- Define the # function
def sharp (N : ℝ) : ℝ := 0.7 * N + 5

-- State the theorem
theorem sharp_four_times_100 : sharp (sharp (sharp (sharp 100))) = 36.675 := by
  sorry

end sharp_four_times_100_l3814_381498


namespace new_person_weight_l3814_381480

/-- Given a group of 8 persons where the average weight increases by 2.5 kg
    when a person weighing 50 kg is replaced, the weight of the new person is 70 kg. -/
theorem new_person_weight (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) :
  n = 8 →
  avg_increase = 2.5 →
  old_weight = 50 →
  n * avg_increase + old_weight = 70 :=
by sorry

end new_person_weight_l3814_381480


namespace power_sum_equality_l3814_381430

theorem power_sum_equality : 3 * 3^3 + 9^61 / 9^59 = 162 := by sorry

end power_sum_equality_l3814_381430


namespace second_crate_granola_weight_l3814_381406

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a crate given its dimensions -/
def crateVolume (d : CrateDimensions) : ℝ := d.height * d.width * d.length

/-- Represents the properties of the first crate -/
def firstCrate : CrateDimensions := {
  height := 4,
  width := 3,
  length := 6
}

/-- The weight of coffee the first crate can hold -/
def firstCrateWeight : ℝ := 72

/-- Represents the properties of the second crate -/
def secondCrate : CrateDimensions := {
  height := firstCrate.height * 1.5,
  width := firstCrate.width * 1.5,
  length := firstCrate.length
}

/-- Theorem stating that the second crate can hold 162 grams of granola -/
theorem second_crate_granola_weight :
  (crateVolume secondCrate / crateVolume firstCrate) * firstCrateWeight = 162 := by sorry

end second_crate_granola_weight_l3814_381406


namespace stones_combine_l3814_381402

/-- Two natural numbers are similar if the larger is at most twice the smaller -/
def similar (a b : ℕ) : Prop := max a b ≤ 2 * min a b

/-- A step in the combining process -/
inductive CombineStep (n : ℕ)
  | combine (a b : ℕ) (h : a + b ≤ n) (hsim : similar a b) : CombineStep n

/-- A sequence of combining steps -/
def CombineSeq (n : ℕ) := List (CombineStep n)

/-- The result of applying a sequence of combining steps -/
def applySeq (n : ℕ) (seq : CombineSeq n) : List ℕ :=
  sorry

/-- The theorem stating that any number of single-stone piles can be combined into one pile -/
theorem stones_combine (n : ℕ) : 
  ∃ (seq : CombineSeq n), applySeq n seq = [n] :=
sorry

end stones_combine_l3814_381402


namespace m_less_than_one_necessary_l3814_381491

/-- A function f(x) = x + mx + m has a root. -/
def has_root (m : ℝ) : Prop :=
  ∃ x : ℝ, x + m * x + m = 0

/-- "m < 1" is a necessary condition for f(x) = x + mx + m to have a root. -/
theorem m_less_than_one_necessary (m : ℝ) :
  has_root m → m < 1 := by sorry

end m_less_than_one_necessary_l3814_381491


namespace marilyn_bottle_caps_l3814_381453

/-- Calculates the remaining bottle caps after sharing. -/
def remaining_bottle_caps (start : ℕ) (shared : ℕ) : ℕ :=
  start - shared

/-- Proves that Marilyn ends up with 15 bottle caps. -/
theorem marilyn_bottle_caps : remaining_bottle_caps 51 36 = 15 := by
  sorry

end marilyn_bottle_caps_l3814_381453


namespace total_money_after_redistribution_l3814_381414

/-- Represents the money redistribution game with three players -/
structure MoneyGame where
  amy_initial : ℝ
  bob_initial : ℝ
  cal_initial : ℝ
  cal_final : ℝ

/-- The rules of the money redistribution game -/
def redistribute (game : MoneyGame) : Prop :=
  ∃ (amy_mid bob_mid cal_mid : ℝ),
    -- Amy's redistribution
    amy_mid + bob_mid + cal_mid = game.amy_initial + game.bob_initial + game.cal_initial ∧
    bob_mid = 2 * game.bob_initial ∧
    cal_mid = 2 * game.cal_initial ∧
    -- Bob's redistribution
    ∃ (amy_mid2 bob_mid2 cal_mid2 : ℝ),
      amy_mid2 + bob_mid2 + cal_mid2 = amy_mid + bob_mid + cal_mid ∧
      amy_mid2 = 2 * amy_mid ∧
      cal_mid2 = 2 * cal_mid ∧
      -- Cal's redistribution
      ∃ (amy_final bob_final : ℝ),
        amy_final + bob_final + game.cal_final = amy_mid2 + bob_mid2 + cal_mid2 ∧
        amy_final = 2 * amy_mid2 ∧
        bob_final = 2 * bob_mid2

/-- The theorem stating the total money after redistribution -/
theorem total_money_after_redistribution (game : MoneyGame)
    (h1 : game.cal_initial = 50)
    (h2 : game.cal_final = 100)
    (h3 : redistribute game) :
    game.amy_initial + game.bob_initial + game.cal_initial = 300 :=
  sorry

end total_money_after_redistribution_l3814_381414


namespace polygon_with_1080_degrees_has_8_sides_l3814_381411

/-- A polygon is a shape with a certain number of sides. -/
structure Polygon where
  sides : ℕ

/-- The sum of interior angles of a polygon. -/
def sumOfInteriorAngles (p : Polygon) : ℝ :=
  180 * (p.sides - 2)

/-- Theorem: A polygon with a sum of interior angles equal to 1080° has 8 sides. -/
theorem polygon_with_1080_degrees_has_8_sides :
  ∃ (p : Polygon), sumOfInteriorAngles p = 1080 → p.sides = 8 := by
  sorry

end polygon_with_1080_degrees_has_8_sides_l3814_381411


namespace B_max_at_181_l3814_381465

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The sequence B_k -/
def B (k : ℕ) : ℝ := (binomial 2000 k : ℝ) * (0.1 ^ k)

/-- The theorem stating that B_k is maximum when k = 181 -/
theorem B_max_at_181 : 
  ∀ k : ℕ, k ≤ 2000 → B k ≤ B 181 :=
sorry

end B_max_at_181_l3814_381465


namespace composition_value_l3814_381451

/-- Given two functions f and g, and a composition condition, prove that d equals 8 -/
theorem composition_value (c d : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = 5*x + c)
  (hg : ∀ x, g x = c*x + 1)
  (hcomp : ∀ x, f (g x) = 15*x + d) :
  d = 8 := by
  sorry

end composition_value_l3814_381451


namespace regular_polygon_20_sides_l3814_381429

-- Define a regular polygon with exterior angle of 18 degrees
structure RegularPolygon where
  sides : ℕ
  exteriorAngle : ℝ
  regular : exteriorAngle = 18

-- Theorem: A regular polygon with exterior angle of 18 degrees has 20 sides
theorem regular_polygon_20_sides (p : RegularPolygon) : p.sides = 20 := by
  sorry

end regular_polygon_20_sides_l3814_381429


namespace petyas_race_time_l3814_381403

/-- Proves that Petya's actual time is greater than the planned time -/
theorem petyas_race_time (a V : ℝ) (h1 : a > 0) (h2 : V > 0) :
  a / V < a / (2.5 * V) + a / (1.6 * V) :=
by sorry

end petyas_race_time_l3814_381403


namespace ricky_magic_box_friday_l3814_381475

/-- Calculates the number of pennies in the magic money box after a given number of days -/
def pennies_after_days (initial_pennies : ℕ) (days : ℕ) : ℕ :=
  initial_pennies * 2^days

/-- Theorem: Ricky's magic money box contains 48 pennies on Friday -/
theorem ricky_magic_box_friday : pennies_after_days 3 4 = 48 := by
  sorry

end ricky_magic_box_friday_l3814_381475


namespace larger_number_from_hcf_lcm_l3814_381408

theorem larger_number_from_hcf_lcm (a b : ℕ+) : 
  (Nat.gcd a b = 15) → 
  (Nat.lcm a b = 2475) → 
  (max a b = 225) :=
by sorry

end larger_number_from_hcf_lcm_l3814_381408


namespace sum_of_multiples_l3814_381479

def smallest_two_digit_multiple_of_5 : ℕ := 10

def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem sum_of_multiples : 
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 :=
by sorry

end sum_of_multiples_l3814_381479


namespace biker_journey_l3814_381495

/-- Proves that given a biker's journey conditions, the distance between towns is 140 km and initial speed is 28 km/hr -/
theorem biker_journey (total_distance : ℝ) (initial_speed : ℝ) : 
  (total_distance / 2 = initial_speed * 2.5) →
  (total_distance / 2 = (initial_speed + 2) * 2.333) →
  (total_distance = 140 ∧ initial_speed = 28) := by
  sorry

end biker_journey_l3814_381495


namespace base14_remainder_theorem_l3814_381488

-- Define a function to convert a base-14 integer to decimal
def base14ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (14 ^ i)) 0

-- Define our specific base-14 number
def ourNumber : List Nat := [1, 4, 6, 2]

-- Theorem statement
theorem base14_remainder_theorem :
  (base14ToDecimal ourNumber) % 10 = 1 := by
  sorry

end base14_remainder_theorem_l3814_381488


namespace multiply_24_99_l3814_381471

theorem multiply_24_99 : 24 * 99 = 2376 := by
  sorry

end multiply_24_99_l3814_381471


namespace goblet_sphere_max_radius_l3814_381452

theorem goblet_sphere_max_radius :
  let goblet_cross_section := fun (x : ℝ) => x^4
  let sphere_in_goblet := fun (r : ℝ) (x y : ℝ) => y ≥ goblet_cross_section x ∧ (y - r)^2 + x^2 = r^2
  ∃ (max_r : ℝ), max_r = 3 / Real.rpow 2 (1/3) ∧
    (∀ r, r > 0 → sphere_in_goblet r 0 0 → r ≤ max_r) ∧
    sphere_in_goblet max_r 0 0 :=
sorry

end goblet_sphere_max_radius_l3814_381452


namespace smallest_number_divisible_by_111_ending_2004_l3814_381450

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

def last_four_digits (n : ℕ) : ℕ := n % 10000

theorem smallest_number_divisible_by_111_ending_2004 :
  ∀ X : ℕ, 
    X > 0 ∧ 
    is_divisible_by X 111 ∧ 
    last_four_digits X = 2004 → 
    X ≥ 662004 :=
by sorry

end smallest_number_divisible_by_111_ending_2004_l3814_381450


namespace sasha_always_wins_l3814_381448

/-- Represents the state of the game board -/
structure GameState where
  digits : List Nat
  deriving Repr

/-- Represents a player's move -/
structure Move where
  appendedDigits : List Nat
  deriving Repr

/-- Checks if a number represented by a list of digits is divisible by 112 -/
def isDivisibleBy112 (digits : List Nat) : Bool :=
  sorry

/-- Generates all possible moves for Sasha (appending one digit) -/
def sashasMoves : List Move :=
  sorry

/-- Generates all possible moves for Andrey (appending two digits) -/
def andreysMoves : List Move :=
  sorry

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if Sasha wins in the current state -/
def sashaWins (state : GameState) : Bool :=
  sorry

/-- Checks if Andrey wins in the current state -/
def andreyWins (state : GameState) : Bool :=
  sorry

/-- Theorem: Sasha can always win the game -/
theorem sasha_always_wins :
  ∀ (state : GameState),
    (state.digits.length < 2018) →
    (∃ (move : Move), move ∈ sashasMoves ∧
      ∀ (andreyMove : Move), andreyMove ∈ andreysMoves →
        ¬(andreyWins (applyMove (applyMove state move) andreyMove))) ∨
    (sashaWins state) :=
  sorry

end sasha_always_wins_l3814_381448


namespace intersection_M_N_l3814_381483

def M : Set ℝ := {x | x^2 + 3*x + 2 > 0}

def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by sorry

end intersection_M_N_l3814_381483


namespace cone_height_equals_cube_volume_l3814_381446

/-- The height of a circular cone with base radius 5 units and volume equal to that of a cube with edge length 5 units is 15/π units. -/
theorem cone_height_equals_cube_volume (h : ℝ) : h = 15 / π := by
  -- Define the edge length of the cube
  let cube_edge : ℝ := 5

  -- Define the base radius of the cone
  let cone_radius : ℝ := 5

  -- Define the volume of the cube
  let cube_volume : ℝ := cube_edge ^ 3

  -- Define the volume of the cone
  let cone_volume : ℝ := (1 / 3) * π * cone_radius ^ 2 * h

  -- Assume the volumes are equal
  have volumes_equal : cube_volume = cone_volume := by sorry

  sorry

end cone_height_equals_cube_volume_l3814_381446


namespace inequality_solution_set_l3814_381458

theorem inequality_solution_set (c : ℝ) : 
  (c / 5 ≤ 4 + c ∧ 4 + c < -3 * (1 + 2 * c)) ↔ c ∈ Set.Icc (-5) (-1) := by
sorry

end inequality_solution_set_l3814_381458


namespace olivias_cookies_l3814_381401

/-- Proves the number of oatmeal cookies given the conditions of the problem -/
theorem olivias_cookies (cookies_per_baggie : ℕ) (total_baggies : ℕ) (chocolate_chip_cookies : ℕ)
  (h1 : cookies_per_baggie = 9)
  (h2 : total_baggies = 6)
  (h3 : chocolate_chip_cookies = 13) :
  cookies_per_baggie * total_baggies - chocolate_chip_cookies = 41 := by
  sorry

end olivias_cookies_l3814_381401


namespace tetrahedron_edge_length_l3814_381420

structure Tetrahedron where
  edges : Finset ℝ
  pq : ℝ
  rs : ℝ

def valid_tetrahedron (t : Tetrahedron) : Prop :=
  t.edges.card = 6 ∧
  t.edges = {9, 15, 22, 28, 34, 39} ∧
  t.pq ∈ t.edges ∧
  t.rs ∈ t.edges ∧
  t.pq = 39

theorem tetrahedron_edge_length (t : Tetrahedron) (h : valid_tetrahedron t) : t.rs = 9 := by
  sorry

end tetrahedron_edge_length_l3814_381420


namespace find_2a_plus_c_l3814_381469

theorem find_2a_plus_c (a b c : ℝ) 
  (eq1 : 3 * a + b + 2 * c = 3) 
  (eq2 : a + 3 * b + 2 * c = 1) : 
  2 * a + c = 2 := by sorry

end find_2a_plus_c_l3814_381469


namespace triangle_area_l3814_381445

/-- Given a triangle ABC where c² = (a-b)² + 6 and angle C = π/3, 
    prove that its area is 3√3/2 -/
theorem triangle_area (a b c : ℝ) (h1 : c^2 = (a-b)^2 + 6) (h2 : Real.pi / 3 = Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) : 
  (1/2) * a * b * Real.sin (Real.pi / 3) = (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_area_l3814_381445


namespace quadratic_inequality_sufficient_conditions_quadratic_inequality_not_necessary_l3814_381449

theorem quadratic_inequality_sufficient_conditions 
  (k : ℝ) (h : k = 0 ∨ (-3 < k ∧ k < 0) ∨ (-3 < k ∧ k < -1)) :
  ∀ x : ℝ, 2*k*x^2 + k*x - 3/8 < 0 :=
by sorry

theorem quadratic_inequality_not_necessary 
  (k : ℝ) (h : ∀ x : ℝ, 2*k*x^2 + k*x - 3/8 < 0) :
  ¬(k = 0 ∧ (-3 < k ∧ k < 0) ∧ (-3 < k ∧ k < -1)) :=
by sorry

end quadratic_inequality_sufficient_conditions_quadratic_inequality_not_necessary_l3814_381449


namespace remainder_1998_pow_10_mod_10000_l3814_381419

theorem remainder_1998_pow_10_mod_10000 : 1998^10 % 10000 = 1024 := by
  sorry

end remainder_1998_pow_10_mod_10000_l3814_381419


namespace root_existence_l3814_381438

theorem root_existence (a b c x₁ x₂ : ℝ) (ha : a ≠ 0)
  (h₁ : a * x₁^2 + b * x₁ + c = 0)
  (h₂ : -a * x₂^2 + b * x₂ + c = 0) :
  ∃ x₃, (a / 2) * x₃^2 + b * x₃ + c = 0 ∧
    ((x₁ ≤ x₃ ∧ x₃ ≤ x₂) ∨ (x₁ ≥ x₃ ∧ x₃ ≥ x₂)) :=
by sorry

end root_existence_l3814_381438


namespace abc_ordering_l3814_381404

theorem abc_ordering (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (heq : a^2 + c^2 = 2*b*c) :
  (b > a ∧ a > c) ∨ (c > b ∧ b > a) :=
sorry

end abc_ordering_l3814_381404


namespace sequence_existence_l3814_381425

theorem sequence_existence (n : ℕ) (hn : n ≥ 3) :
  (∃ a : ℕ → ℝ, 
    (a (n + 1) = a 1) ∧ 
    (a (n + 2) = a 2) ∧ 
    (∀ i ∈ Finset.range n, a i * a (i + 1) + 1 = a (i + 2)))
  ↔ 
  (∃ k : ℕ, n = 3 * k) :=
by sorry

end sequence_existence_l3814_381425


namespace equal_goals_moment_l3814_381460

/-- Represents the state of a football match at any given moment -/
structure MatchState where
  goalsWinner : ℕ
  goalsLoser : ℕ

/-- The final score of the match -/
def finalScore : MatchState := { goalsWinner := 9, goalsLoser := 5 }

/-- Theorem stating that there exists a point during the match where the number of goals
    the winning team still needs to score equals the number of goals the losing team has already scored -/
theorem equal_goals_moment :
  ∃ (state : MatchState), 
    state.goalsWinner ≤ finalScore.goalsWinner ∧ 
    state.goalsLoser ≤ finalScore.goalsLoser ∧
    (finalScore.goalsWinner - state.goalsWinner) = state.goalsLoser :=
sorry

end equal_goals_moment_l3814_381460


namespace complex_magnitude_example_l3814_381461

theorem complex_magnitude_example : Complex.abs (-5 - (8/3)*Complex.I) = 17/3 := by
  sorry

end complex_magnitude_example_l3814_381461


namespace problem_1_l3814_381427

theorem problem_1 (x : ℝ) : (3*x + 1)*(3*x - 1) - (3*x + 1)^2 = -6*x - 2 := by
  sorry

end problem_1_l3814_381427


namespace optimal_gasoline_percentage_l3814_381486

/-- Calculates the optimal gasoline percentage for a car's fuel mixture --/
theorem optimal_gasoline_percentage
  (initial_volume : ℝ)
  (initial_ethanol_percentage : ℝ)
  (initial_gasoline_percentage : ℝ)
  (added_ethanol : ℝ)
  (optimal_ethanol_percentage : ℝ)
  (h1 : initial_volume = 36)
  (h2 : initial_ethanol_percentage = 5)
  (h3 : initial_gasoline_percentage = 95)
  (h4 : added_ethanol = 2)
  (h5 : optimal_ethanol_percentage = 10)
  (h6 : initial_ethanol_percentage + initial_gasoline_percentage = 100) :
  let final_volume := initial_volume + added_ethanol
  let final_ethanol := initial_volume * (initial_ethanol_percentage / 100) + added_ethanol
  let final_ethanol_percentage := (final_ethanol / final_volume) * 100
  100 - optimal_ethanol_percentage = 90 ∧ final_ethanol_percentage = optimal_ethanol_percentage :=
by sorry

end optimal_gasoline_percentage_l3814_381486


namespace berries_and_coconut_cost_l3814_381466

/-- The cost of a bundle of berries and a coconut given the specified conditions -/
theorem berries_and_coconut_cost :
  ∀ (p b c d : ℚ),
  p + b + c + d = 30 →
  d = 3 * p →
  c = (p + b) / 2 →
  b + c = 65 / 9 := by
sorry

end berries_and_coconut_cost_l3814_381466


namespace certain_number_problem_l3814_381487

theorem certain_number_problem (x : ℝ) (n : ℝ) : 
  (9 - 4 / x = 7 + n / x) → (x = 6) → n = 8 := by
  sorry

end certain_number_problem_l3814_381487


namespace sum_of_roots_l3814_381421

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 12*a^2 + 9*a - 18 = 0)
  (hb : 9*b^3 - 135*b^2 + 450*b - 1650 = 0) :
  a + b = 6 ∨ a + b = 14 := by sorry

end sum_of_roots_l3814_381421


namespace remainder_theorem_l3814_381474

theorem remainder_theorem (n : ℤ) : (7 - 2*n + (n + 5)) % 5 = (-n + 2) % 5 := by
  sorry

end remainder_theorem_l3814_381474


namespace quadratic_solution_difference_l3814_381442

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  2 * x^2 - 5 * x + 18 = 3 * x + 55

-- Define the solutions of the quadratic equation
noncomputable def solution1 : ℝ := 2 + (3 * Real.sqrt 10) / 2
noncomputable def solution2 : ℝ := 2 - (3 * Real.sqrt 10) / 2

-- Theorem statement
theorem quadratic_solution_difference :
  quadratic_equation solution1 ∧ 
  quadratic_equation solution2 ∧ 
  |solution1 - solution2| = 3 * Real.sqrt 10 := by sorry

end quadratic_solution_difference_l3814_381442


namespace find_number_l3814_381497

theorem find_number : ∃! x : ℤ, (x + 12) / 4 = 12 ∧ (x + 12) % 4 = 3 := by
  sorry

end find_number_l3814_381497


namespace circle_tangents_and_chord_l3814_381437

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 5 = 0

-- Define the tangent lines
def tangent1 (x y : ℝ) : Prop := 4*x + 3*y - 23 = 0
def tangent2 (x : ℝ) : Prop := x = 5

-- Define the chord line
def chord_line (x y : ℝ) : Prop := x + y - 4 = 0

-- Theorem statement
theorem circle_tangents_and_chord :
  -- Part 1: Tangent lines
  (∀ x y, C x y → (tangent1 x y → x^2 + y^2 = 25)) ∧
  (∀ x y, C x y → (tangent2 x → x^2 + y^2 = 25)) ∧
  tangent1 5 1 ∧ tangent2 5 ∧
  -- Part 2: Chord line
  (∀ x₁ y₁ x₂ y₂, C x₁ y₁ ∧ C x₂ y₂ ∧ (x₁ + x₂)/2 = 3 ∧ (y₁ + y₂)/2 = 1 →
    chord_line x₁ y₁ ∧ chord_line x₂ y₂) :=
by sorry

end circle_tangents_and_chord_l3814_381437


namespace proof_by_contradiction_step_l3814_381431

theorem proof_by_contradiction_step (a : ℝ) (h : a > 1) :
  (∀ P : Prop, (¬P → False) → P) →
  (¬(a^2 > 1) ↔ a^2 ≤ 1) :=
sorry

end proof_by_contradiction_step_l3814_381431


namespace possible_values_of_a_l3814_381473

theorem possible_values_of_a : 
  ∀ (a b c : ℤ), 
  (∀ x : ℝ, (x - a) * (x - 8) + 4 = (x + b) * (x + c)) → 
  (a = 6 ∨ a = 10) := by
sorry

end possible_values_of_a_l3814_381473


namespace journey_distance_l3814_381412

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the final position after a series of movements -/
def finalPosition (initialDistance : ℝ) : Point :=
  let southWalk := Point.mk 0 (-initialDistance)
  let eastWalk := Point.mk initialDistance (-initialDistance)
  let northWalk := Point.mk initialDistance 0
  let finalEastWalk := Point.mk (initialDistance + 10) 0
  finalEastWalk

/-- Theorem stating that the initial distance must be 30 to end up 30 meters north -/
theorem journey_distance (initialDistance : ℝ) :
  (finalPosition initialDistance).y = 30 ↔ initialDistance = 30 :=
by sorry

end journey_distance_l3814_381412


namespace pet_store_cages_l3814_381464

def total_cages (initial_puppies initial_adult_dogs initial_kittens : ℕ)
                (sold_puppies sold_adult_dogs sold_kittens : ℕ)
                (puppies_per_cage adult_dogs_per_cage kittens_per_cage : ℕ) : ℕ :=
  let remaining_puppies := initial_puppies - sold_puppies
  let remaining_adult_dogs := initial_adult_dogs - sold_adult_dogs
  let remaining_kittens := initial_kittens - sold_kittens
  let puppy_cages := (remaining_puppies + puppies_per_cage - 1) / puppies_per_cage
  let adult_dog_cages := (remaining_adult_dogs + adult_dogs_per_cage - 1) / adult_dogs_per_cage
  let kitten_cages := (remaining_kittens + kittens_per_cage - 1) / kittens_per_cage
  puppy_cages + adult_dog_cages + kitten_cages

theorem pet_store_cages : 
  total_cages 45 30 25 39 15 10 3 2 2 = 18 := by
  sorry

end pet_store_cages_l3814_381464


namespace committee_selection_count_l3814_381484

/-- The number of ways to choose a committee of size k from n people -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The size of the club -/
def club_size : ℕ := 30

/-- The size of the committee -/
def committee_size : ℕ := 5

/-- Theorem: The number of ways to choose a 5-person committee from a 30-person club is 142506 -/
theorem committee_selection_count : choose club_size committee_size = 142506 := by
  sorry

end committee_selection_count_l3814_381484


namespace tv_selection_problem_l3814_381494

theorem tv_selection_problem (type_a : ℕ) (type_b : ℕ) (total_select : ℕ) : 
  type_a = 4 → type_b = 5 → total_select = 3 →
  (Nat.choose type_a 1 * Nat.choose type_b 2 + Nat.choose type_a 2 * Nat.choose type_b 1) = 70 :=
by sorry

end tv_selection_problem_l3814_381494


namespace work_completion_time_l3814_381432

theorem work_completion_time 
  (a b c : ℝ) 
  (h1 : a + b + c = 1/4)  -- a, b, and c together finish in 4 days
  (h2 : b = 1/18)         -- b alone finishes in 18 days
  (h3 : c = 1/6)          -- c alone finishes in 6 days
  : a = 1/36 :=           -- a alone finishes in 36 days
by sorry

end work_completion_time_l3814_381432


namespace function_passes_through_point_l3814_381489

-- Define the function
def f (x : ℝ) : ℝ := 3 * x - 2

-- State the theorem
theorem function_passes_through_point :
  f (-1) = -5 := by sorry

end function_passes_through_point_l3814_381489


namespace marble_count_proof_l3814_381434

/-- The smallest positive integer greater than 1 that leaves a remainder of 1 when divided by 6, 7, and 8 -/
def smallest_marble_count : ℕ := 169

/-- Proves that the smallest_marble_count satisfies the given conditions -/
theorem marble_count_proof :
  smallest_marble_count > 1 ∧
  smallest_marble_count % 6 = 1 ∧
  smallest_marble_count % 7 = 1 ∧
  smallest_marble_count % 8 = 1 ∧
  ∀ n : ℕ, n > 1 →
    (n % 6 = 1 ∧ n % 7 = 1 ∧ n % 8 = 1) →
    n ≥ smallest_marble_count :=
by sorry

end marble_count_proof_l3814_381434


namespace total_parts_calculation_l3814_381467

theorem total_parts_calculation (sample_size : ℕ) (probability : ℚ) (N : ℕ) : 
  sample_size = 30 → probability = 1/4 → N * probability = sample_size → N = 120 := by
  sorry

end total_parts_calculation_l3814_381467


namespace unique_solution_l3814_381457

theorem unique_solution (x y z : ℝ) 
  (hx : x > 2) (hy : y > 2) (hz : z > 2)
  (h : ((x + 3)^2) / (y + z - 3) + ((y + 5)^2) / (z + x - 5) + ((z + 7)^2) / (x + y - 7) = 45) :
  x = 13 ∧ y = 11 ∧ z = 6 := by
  sorry

end unique_solution_l3814_381457


namespace coin_difference_l3814_381433

def coin_values : List ℕ := [5, 10, 25, 50]

def target_amount : ℕ := 75

def is_valid_combination (combination : List ℕ) : Prop :=
  combination.all (λ x => x ∈ coin_values) ∧
  combination.sum = target_amount

def num_coins (combination : List ℕ) : ℕ := combination.length

theorem coin_difference :
  ∃ (max_combination min_combination : List ℕ),
    is_valid_combination max_combination ∧
    is_valid_combination min_combination ∧
    (∀ c, is_valid_combination c →
      num_coins c ≤ num_coins max_combination ∧
      num_coins c ≥ num_coins min_combination) ∧
    num_coins max_combination - num_coins min_combination = 13 :=
  sorry

end coin_difference_l3814_381433


namespace loan_period_is_three_l3814_381400

/-- The period of a loan (in years) where:
  - A lends Rs. 3500 to B at 10% per annum
  - B lends Rs. 3500 to C at 11.5% per annum
  - B gains Rs. 157.5 -/
def loanPeriod : ℝ → Prop := λ T =>
  let principal : ℝ := 3500
  let rateAtoB : ℝ := 10
  let rateBtoC : ℝ := 11.5
  let bGain : ℝ := 157.5
  let interestAtoB : ℝ := principal * rateAtoB * T / 100
  let interestBtoC : ℝ := principal * rateBtoC * T / 100
  interestBtoC - interestAtoB = bGain

theorem loan_period_is_three : loanPeriod 3 := by
  sorry

end loan_period_is_three_l3814_381400


namespace cuboid_inequality_l3814_381426

theorem cuboid_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_unit_diagonal : a^2 + b^2 + c^2 = 1) : 
  4*a + 4*b + 4*c + 4*a*b + 4*a*c + 4*b*c + 4*a*b*c < 12 := by
  sorry

end cuboid_inequality_l3814_381426


namespace amusement_park_distance_l3814_381440

-- Define the speeds and time
def speed_A : ℝ := 3
def speed_B : ℝ := 4
def total_time : ℝ := 4

-- Define the distance functions
def distance_A (t : ℝ) : ℝ := speed_A * t
def distance_B (t : ℝ) : ℝ := speed_B * t

-- Theorem statement
theorem amusement_park_distance :
  ∃ (t_A t_B : ℝ),
    t_A + t_B = total_time ∧
    distance_B t_B = distance_A t_A + 2 ∧
    distance_B t_B = 8 :=
by sorry

end amusement_park_distance_l3814_381440


namespace coffee_lasts_13_days_l3814_381478

def coffee_problem (coffee_weight : ℕ) (cups_per_pound : ℕ) 
  (angie_cups : ℕ) (bob_cups : ℕ) (carol_cups : ℕ) : ℕ :=
  let total_cups := coffee_weight * cups_per_pound
  let daily_consumption := angie_cups + bob_cups + carol_cups
  total_cups / daily_consumption

theorem coffee_lasts_13_days :
  coffee_problem 3 40 3 2 4 = 13 := by
  sorry

end coffee_lasts_13_days_l3814_381478


namespace flour_needed_for_doubled_recipe_l3814_381435

theorem flour_needed_for_doubled_recipe 
  (original_recipe : ℕ) 
  (already_added : ℕ) 
  (h1 : original_recipe = 7)
  (h2 : already_added = 3) : 
  (2 * original_recipe) - already_added = 11 :=
by sorry

end flour_needed_for_doubled_recipe_l3814_381435


namespace trapezoid_area_l3814_381459

/-- The area of a trapezoid given its median line and height -/
theorem trapezoid_area (median_line height : ℝ) (h1 : median_line = 8) (h2 : height = 12) :
  median_line * height = 96 :=
sorry

end trapezoid_area_l3814_381459


namespace angle_in_third_quadrant_l3814_381441

def is_in_third_quadrant (α : Real) : Prop :=
  180 < α % 360 ∧ α % 360 ≤ 270

theorem angle_in_third_quadrant (k : Int) (α : Real) 
  (h1 : (4*k + 1 : Real) * 180 < α) 
  (h2 : α < (4*k + 1 : Real) * 180 + 60) :
  is_in_third_quadrant α :=
sorry

end angle_in_third_quadrant_l3814_381441


namespace unique_base_number_exists_l3814_381413

/-- A number in base 2022 with digits 1 or 2 -/
def BaseNumber (n : ℕ) := {x : ℕ // ∀ d, d ∈ x.digits 2022 → d = 1 ∨ d = 2}

/-- The theorem statement -/
theorem unique_base_number_exists :
  ∃! (N : BaseNumber 1000), (N.val : ℤ) % (2^1000) = 0 := by
  sorry

end unique_base_number_exists_l3814_381413


namespace lcm_18_27_l3814_381476

theorem lcm_18_27 : Nat.lcm 18 27 = 54 := by
  sorry

end lcm_18_27_l3814_381476


namespace constant_value_l3814_381456

theorem constant_value (t : ℝ) (c : ℝ) : 
  let x := 1 - 3 * t
  let y := 2 * t - c
  (x = y ∧ t = 0.8) → c = 3 := by
sorry

end constant_value_l3814_381456


namespace inscribed_sphere_radius_eq_one_l3814_381424

/-- A right triangular pyramid with base edge length 6 and lateral edge length √21 -/
structure RightTriangularPyramid where
  base_edge_length : ℝ
  lateral_edge_length : ℝ
  base_edge_length_eq : base_edge_length = 6
  lateral_edge_length_eq : lateral_edge_length = Real.sqrt 21

/-- The radius of the inscribed sphere of a right triangular pyramid -/
def inscribed_sphere_radius (p : RightTriangularPyramid) : ℝ :=
  1 -- Definition, not proof

/-- Theorem: The radius of the inscribed sphere of a right triangular pyramid
    with base edge length 6 and lateral edge length √21 is equal to 1 -/
theorem inscribed_sphere_radius_eq_one (p : RightTriangularPyramid) :
  inscribed_sphere_radius p = 1 := by
  sorry

#check inscribed_sphere_radius_eq_one

end inscribed_sphere_radius_eq_one_l3814_381424


namespace no_article_before_word_l3814_381407

-- Define the sentence structure
def sentence_structure : String := "They sent us ______ word of the latest happenings."

-- Define the function to determine the correct article
def correct_article : String := ""

-- Theorem statement
theorem no_article_before_word :
  correct_article = "" := by sorry

end no_article_before_word_l3814_381407


namespace xiao_jun_pictures_xiao_jun_pictures_proof_l3814_381490

theorem xiao_jun_pictures : ℕ → Prop :=
  fun original : ℕ =>
    let half := original / 2
    let given_away := half - 1
    let remaining := original - given_away
    remaining = 25 → original = 48

-- The proof is omitted
theorem xiao_jun_pictures_proof : xiao_jun_pictures 48 := by
  sorry

end xiao_jun_pictures_xiao_jun_pictures_proof_l3814_381490


namespace complex_arithmetic_proof_l3814_381463

theorem complex_arithmetic_proof :
  let B : ℂ := 3 + 2*I
  let Q : ℂ := -5*I
  let R : ℂ := 1 + I
  let T : ℂ := 3 - 4*I
  B * R + Q + T = 4 + I :=
by sorry

end complex_arithmetic_proof_l3814_381463


namespace china_population_scientific_notation_l3814_381482

/-- Represents the population of China in millions -/
def china_population : ℝ := 1412.60

/-- The scientific notation representation of the population -/
def scientific_notation : ℝ := 1.4126 * (10 ^ 5)

/-- Theorem stating that the scientific notation representation is correct -/
theorem china_population_scientific_notation :
  china_population = scientific_notation := by
  sorry

end china_population_scientific_notation_l3814_381482


namespace quadratic_function_unique_l3814_381444

def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_function_unique 
  (f : ℝ → ℝ) 
  (h1 : ∃ a b c : ℝ, f = quadratic_function a b c)
  (h2 : f (-2) = 0)
  (h3 : f 4 = 0)
  (h4 : ∀ x : ℝ, f x ≤ 9)
  (h5 : ∃ x : ℝ, f x = 9) :
  f = quadratic_function (-1) 2 8 := by
sorry

end quadratic_function_unique_l3814_381444


namespace hyperbola_eccentricity_l3814_381472

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : b / a = 4 / 3) : 
  Real.sqrt (1 + (b / a)^2) = 5 / 3 := by
  sorry

#check hyperbola_eccentricity

end hyperbola_eccentricity_l3814_381472


namespace intersection_M_N_l3814_381477

-- Define set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x - |x|}

-- Define set N
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sqrt x}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end intersection_M_N_l3814_381477


namespace razorback_tshirt_sales_l3814_381428

/-- The Razorback t-shirt shop problem -/
theorem razorback_tshirt_sales (price : ℕ) (arkansas_sales : ℕ) (texas_tech_revenue : ℕ) :
  price = 78 →
  arkansas_sales = 172 →
  texas_tech_revenue = 1092 →
  arkansas_sales + (texas_tech_revenue / price) = 186 :=
by sorry

end razorback_tshirt_sales_l3814_381428


namespace sin_cos_relation_l3814_381455

theorem sin_cos_relation (α β : ℝ) (h : 2 * Real.sin α - Real.cos β = 2) :
  Real.sin α + 2 * Real.cos β = 1 ∨ Real.sin α + 2 * Real.cos β = -1 := by
  sorry

end sin_cos_relation_l3814_381455


namespace tax_reduction_theorem_l3814_381485

theorem tax_reduction_theorem (T C : ℝ) (x : ℝ) 
  (h1 : T > 0) (h2 : C > 0) 
  (h3 : (T * (1 - x / 100)) * (C * 1.1) = T * C * 0.88) : x = 20 := by
  sorry

end tax_reduction_theorem_l3814_381485


namespace max_sequence_length_l3814_381443

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The n-th term of the sequence -/
def a (n : ℕ) (x : ℕ) : ℤ :=
  if n % 2 = 1 then
    1000 * (fib (n - 2)) - x * (fib (n - 1))
  else
    x * (fib (n - 1)) - 1000 * (fib (n - 2))

/-- The condition for the first negative term -/
def first_negative (x : ℕ) : Prop :=
  ∃ n : ℕ, (∀ k < n, a k x ≥ 0) ∧ a n x < 0

/-- The maximum value of x that produces the longest sequence -/
def max_x : ℕ := 618

/-- The main theorem -/
theorem max_sequence_length :
  ∀ y : ℕ, y > max_x → first_negative y → 
  ∃ z : ℕ, z ≤ max_x ∧ first_negative z ∧ 
  ∀ w : ℕ, first_negative w → (∃ n : ℕ, a n z < 0) → (∃ m : ℕ, m ≤ n ∧ a m w < 0) :=
sorry

end max_sequence_length_l3814_381443


namespace average_monthly_balance_l3814_381423

def monthly_balances : List ℕ := [100, 200, 150, 150, 180]

theorem average_monthly_balance :
  (monthly_balances.sum / monthly_balances.length : ℚ) = 156 := by
  sorry

end average_monthly_balance_l3814_381423
