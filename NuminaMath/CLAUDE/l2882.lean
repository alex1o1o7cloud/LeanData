import Mathlib

namespace NUMINAMATH_CALUDE_robins_gum_problem_l2882_288212

theorem robins_gum_problem (initial_gum : ℕ) (total_gum : ℕ) (h1 : initial_gum = 18) (h2 : total_gum = 44) :
  total_gum - initial_gum = 26 := by
  sorry

end NUMINAMATH_CALUDE_robins_gum_problem_l2882_288212


namespace NUMINAMATH_CALUDE_penny_difference_l2882_288243

theorem penny_difference (kate_pennies john_pennies : ℕ) 
  (h1 : kate_pennies = 223) 
  (h2 : john_pennies = 388) : 
  john_pennies - kate_pennies = 165 := by
sorry

end NUMINAMATH_CALUDE_penny_difference_l2882_288243


namespace NUMINAMATH_CALUDE_lunch_calories_l2882_288266

/-- The total calories for a kid's lunch -/
def total_calories (burger_calories : ℕ) (carrot_stick_calories : ℕ) (cookie_calories : ℕ) : ℕ :=
  burger_calories + 5 * carrot_stick_calories + 5 * cookie_calories

/-- Theorem stating that the total calories for each kid's lunch is 750 -/
theorem lunch_calories :
  total_calories 400 20 50 = 750 := by
  sorry

end NUMINAMATH_CALUDE_lunch_calories_l2882_288266


namespace NUMINAMATH_CALUDE_log_equation_equivalence_l2882_288235

theorem log_equation_equivalence (x : ℝ) :
  x > 0 → ((Real.log x / Real.log 4) * (Real.log 5 / Real.log x) = Real.log 5 / Real.log 4 ↔ x ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_equivalence_l2882_288235


namespace NUMINAMATH_CALUDE_smallest_variance_l2882_288217

def minimumVariance (n : ℕ) (s : Finset ℝ) : Prop :=
  n ≥ 2 ∧
  s.card = n ∧
  (0 : ℝ) ∈ s ∧
  (1 : ℝ) ∈ s ∧
  (∀ x ∈ s, 0 ≤ x ∧ x ≤ 1) →
  ∀ ε > 0, ∃ (v : ℝ), v ≥ 1 / (2 * n) ∧
    v = (s.sum (λ x => (x - s.sum (λ y => y) / n) ^ 2)) / n

theorem smallest_variance (n : ℕ) (s : Finset ℝ) (h : minimumVariance n s) :
  ∃ (v : ℝ), v = 1 / (2 * n) ∧
    v = (s.sum (λ x => (x - s.sum (λ y => y) / n) ^ 2)) / n :=
sorry

end NUMINAMATH_CALUDE_smallest_variance_l2882_288217


namespace NUMINAMATH_CALUDE_son_age_problem_l2882_288273

theorem son_age_problem (father_age : ℕ) (son_age : ℕ) : 
  father_age = 40 ∧ 
  father_age = 4 * son_age ∧ 
  father_age + 20 = 2 * (son_age + 20) → 
  son_age = 10 :=
by sorry

end NUMINAMATH_CALUDE_son_age_problem_l2882_288273


namespace NUMINAMATH_CALUDE_derivative_x_squared_cos_l2882_288280

/-- The derivative of x^2 * cos(x) is 2x * cos(x) - x^2 * sin(x) -/
theorem derivative_x_squared_cos (x : ℝ) :
  deriv (λ x => x^2 * Real.cos x) x = 2 * x * Real.cos x - x^2 * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_squared_cos_l2882_288280


namespace NUMINAMATH_CALUDE_jasons_betta_fish_count_jasons_betta_fish_count_is_five_l2882_288291

/-- The number of betta fish Jason has, given:
  1. The moray eel eats 20 guppies per day.
  2. Each betta fish eats 7 guppies per day.
  3. The total number of guppies needed per day is 55. -/
theorem jasons_betta_fish_count : ℕ :=
  let moray_eel_guppies : ℕ := 20
  let betta_fish_guppies_per_day : ℕ := 7
  let total_guppies_per_day : ℕ := 55
  let betta_fish_count := (total_guppies_per_day - moray_eel_guppies) / betta_fish_guppies_per_day
  5

/-- Proof that Jason has 5 betta fish -/
theorem jasons_betta_fish_count_is_five : jasons_betta_fish_count = 5 := by
  sorry

end NUMINAMATH_CALUDE_jasons_betta_fish_count_jasons_betta_fish_count_is_five_l2882_288291


namespace NUMINAMATH_CALUDE_a_4_equals_4_l2882_288230

/-- Given a sequence {aₙ} defined by aₙ = (-1)ⁿ n, prove that a₄ = 4 -/
theorem a_4_equals_4 (a : ℕ → ℤ) (h : ∀ n, a n = (-1)^n * n) : a 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_a_4_equals_4_l2882_288230


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2882_288284

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- General term of the sequence
  S : ℕ → ℝ  -- Sum of the first n terms
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- Theorem about a specific arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
    (h1 : seq.a 1 + seq.a 3 = 16)
    (h2 : seq.S 4 = 28) :
  (∀ n : ℕ, seq.a n = 12 - 2 * n) ∧
  (∀ n : ℕ, n ≤ 5 → seq.S n ≤ seq.S 5) ∧
  seq.S 5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2882_288284


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l2882_288264

/-- The length of the pan in inches -/
def pan_length : ℕ := 24

/-- The width of the pan in inches -/
def pan_width : ℕ := 15

/-- The side length of a square brownie piece in inches -/
def piece_side : ℕ := 3

/-- The number of brownie pieces that can be cut from the pan -/
def num_pieces : ℕ := (pan_length * pan_width) / (piece_side * piece_side)

theorem brownie_pieces_count : num_pieces = 40 := by
  sorry

end NUMINAMATH_CALUDE_brownie_pieces_count_l2882_288264


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l2882_288238

theorem inequality_solution_sets (a : ℝ) :
  (∀ x, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) →
  (∀ x, ax^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l2882_288238


namespace NUMINAMATH_CALUDE_wynter_bicycle_count_l2882_288231

/-- The number of bicycles Wynter counted -/
def num_bicycles : ℕ := 50

/-- The number of tricycles Wynter counted -/
def num_tricycles : ℕ := 20

/-- The total number of wheels from all vehicles -/
def total_wheels : ℕ := 160

/-- The number of wheels on a bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- The number of wheels on a tricycle -/
def wheels_per_tricycle : ℕ := 3

/-- Theorem stating that the number of bicycles Wynter counted is 50 -/
theorem wynter_bicycle_count :
  num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle = total_wheels :=
by sorry

end NUMINAMATH_CALUDE_wynter_bicycle_count_l2882_288231


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2882_288279

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (b c d : ℝ) :=
  c * c = b * d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  a 4 = 10 →
  arithmetic_sequence a d →
  geometric_sequence (a 3) (a 6) (a 10) →
  ∀ n, a n = n + 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2882_288279


namespace NUMINAMATH_CALUDE_cassies_nail_trimming_l2882_288274

/-- The number of nails/claws Cassie needs to cut -/
def total_nails_to_cut (num_dogs : ℕ) (num_parrots : ℕ) (nails_per_dog_foot : ℕ) 
  (feet_per_dog : ℕ) (claws_per_parrot_leg : ℕ) (legs_per_parrot : ℕ) 
  (extra_claw : ℕ) : ℕ :=
  (num_dogs * nails_per_dog_foot * feet_per_dog) + 
  (num_parrots * claws_per_parrot_leg * legs_per_parrot) + 
  extra_claw

/-- Theorem stating the total number of nails/claws Cassie needs to cut -/
theorem cassies_nail_trimming :
  total_nails_to_cut 4 8 4 4 3 2 1 = 113 := by
  sorry

end NUMINAMATH_CALUDE_cassies_nail_trimming_l2882_288274


namespace NUMINAMATH_CALUDE_quadratic_root_implies_d_value_l2882_288215

theorem quadratic_root_implies_d_value 
  (d : ℝ) 
  (h : ∀ x : ℝ, 2 * x^2 + 14 * x + d = 0 ↔ x = (-14 + Real.sqrt 20) / 4 ∨ x = (-14 - Real.sqrt 20) / 4) :
  d = 22 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_d_value_l2882_288215


namespace NUMINAMATH_CALUDE_consecutive_numbers_theorem_l2882_288219

theorem consecutive_numbers_theorem (a b c d e : ℕ) : 
  (a > b) ∧ (b > c) ∧ (c > d) ∧ (d > e) ∧  -- Descending order
  (a - b = 1) ∧ (b - c = 1) ∧ (c - d = 1) ∧ (d - e = 1) ∧  -- Consecutive numbers
  ((a + b + c) / 3 = 45) ∧  -- Average of first three
  ((c + d + e) / 3 = 43) →  -- Average of last three
  c = 44 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_theorem_l2882_288219


namespace NUMINAMATH_CALUDE_cuboid_base_area_l2882_288241

/-- Theorem: For a cuboid with volume 28 cm³ and height 4 cm, the base area is 7 cm² -/
theorem cuboid_base_area (volume : ℝ) (height : ℝ) (base_area : ℝ) :
  volume = 28 →
  height = 4 →
  volume = base_area * height →
  base_area = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_cuboid_base_area_l2882_288241


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l2882_288222

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 17) 
  (h2 : c + a = 18) 
  (h3 : a + b = 19) : 
  Real.sqrt (a * b * c * (a + b + c)) = 54 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l2882_288222


namespace NUMINAMATH_CALUDE_abc_zero_necessary_not_sufficient_for_a_zero_l2882_288269

theorem abc_zero_necessary_not_sufficient_for_a_zero (a b c : ℝ) :
  (∀ a b c, a = 0 → a * b * c = 0) ∧
  (∃ a b c, a * b * c = 0 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_abc_zero_necessary_not_sufficient_for_a_zero_l2882_288269


namespace NUMINAMATH_CALUDE_kannon_fruit_consumption_l2882_288246

/-- Represents Kannon's fruit consumption over two days -/
structure FruitConsumption where
  apples_last_night : ℕ
  bananas_last_night : ℕ
  oranges_last_night : ℕ
  apples_increase : ℕ
  bananas_multiplier : ℕ

/-- Calculates the total number of fruits eaten over two days -/
def total_fruits (fc : FruitConsumption) : ℕ :=
  let apples_today := fc.apples_last_night + fc.apples_increase
  let bananas_today := fc.bananas_last_night * fc.bananas_multiplier
  let oranges_today := 2 * apples_today
  (fc.apples_last_night + apples_today) +
  (fc.bananas_last_night + bananas_today) +
  (fc.oranges_last_night + oranges_today)

/-- Theorem stating that Kannon's total fruit consumption is 39 -/
theorem kannon_fruit_consumption :
  ∃ (fc : FruitConsumption),
    fc.apples_last_night = 3 ∧
    fc.bananas_last_night = 1 ∧
    fc.oranges_last_night = 4 ∧
    fc.apples_increase = 4 ∧
    fc.bananas_multiplier = 10 ∧
    total_fruits fc = 39 := by
  sorry

end NUMINAMATH_CALUDE_kannon_fruit_consumption_l2882_288246


namespace NUMINAMATH_CALUDE_prob_from_third_farm_given_over_300kg_l2882_288211

/-- Represents the three farms supplying calves -/
inductive Farm : Type
  | first : Farm
  | second : Farm
  | third : Farm

/-- The proportion of calves from each farm -/
def farm_proportion : Farm → ℝ
  | Farm.first => 0.6
  | Farm.second => 0.3
  | Farm.third => 0.1

/-- The probability that a calf from a given farm weighs over 300 kg -/
def prob_over_300kg : Farm → ℝ
  | Farm.first => 0.15
  | Farm.second => 0.25
  | Farm.third => 0.35

/-- The probability that a randomly selected calf weighing over 300 kg came from the third farm -/
theorem prob_from_third_farm_given_over_300kg : 
  (farm_proportion Farm.third * prob_over_300kg Farm.third) / 
  (farm_proportion Farm.first * prob_over_300kg Farm.first + 
   farm_proportion Farm.second * prob_over_300kg Farm.second + 
   farm_proportion Farm.third * prob_over_300kg Farm.third) = 0.175 := by
  sorry

end NUMINAMATH_CALUDE_prob_from_third_farm_given_over_300kg_l2882_288211


namespace NUMINAMATH_CALUDE_carlsons_land_cost_l2882_288224

/-- Proves that the cost of Carlson's first land is $8000 --/
theorem carlsons_land_cost (initial_area : ℝ) (final_area : ℝ) (new_land_cost_per_sqm : ℝ) (additional_cost : ℝ) : ℝ :=
  by
  -- Define the given conditions
  have h1 : initial_area = 300 := by sorry
  have h2 : final_area = 900 := by sorry
  have h3 : new_land_cost_per_sqm = 20 := by sorry
  have h4 : additional_cost = 4000 := by sorry

  -- Define the first land cost
  let first_land_cost : ℝ := 8000

  -- Prove that the first land cost is $8000
  sorry

end NUMINAMATH_CALUDE_carlsons_land_cost_l2882_288224


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l2882_288216

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 - 4*p^3 + 7*p^2 - 3*p + 2 = 0) →
  (q^4 - 4*q^3 + 7*q^2 - 3*q + 2 = 0) →
  (r^4 - 4*r^3 + 7*r^2 - 3*r + 2 = 0) →
  (s^4 - 4*s^3 + 7*s^2 - 3*s + 2 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 7/2 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l2882_288216


namespace NUMINAMATH_CALUDE_probability_sphere_in_cube_l2882_288218

/-- The probability of a point (x, y, z) satisfying x^2 + y^2 + z^2 ≤ 4,
    given that -2 ≤ x ≤ 2, -2 ≤ y ≤ 2, and -2 ≤ z ≤ 2 -/
theorem probability_sphere_in_cube : 
  let cube_volume := (2 - (-2))^3
  let sphere_volume := (4/3) * Real.pi * 2^3
  sphere_volume / cube_volume = Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_probability_sphere_in_cube_l2882_288218


namespace NUMINAMATH_CALUDE_prob_even_sum_is_two_fifths_l2882_288202

/-- A card is represented by a natural number between 1 and 5 -/
def Card : Type := { n : ℕ // 1 ≤ n ∧ n ≤ 5 }

/-- The set of all cards -/
def allCards : Finset Card := sorry

/-- The function that determines if the sum of two cards is even -/
def isEvenSum (c1 c2 : Card) : Prop := Even (c1.val + c2.val)

/-- The set of all pairs of cards -/
def allPairs : Finset (Card × Card) := sorry

/-- The set of all pairs of cards with even sum -/
def evenSumPairs : Finset (Card × Card) := sorry

/-- The probability of drawing two cards with even sum -/
noncomputable def probEvenSum : ℚ := (Finset.card evenSumPairs : ℚ) / (Finset.card allPairs : ℚ)

theorem prob_even_sum_is_two_fifths : probEvenSum = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_prob_even_sum_is_two_fifths_l2882_288202


namespace NUMINAMATH_CALUDE_cone_volume_arithmetic_progression_l2882_288249

/-- The volume of a right circular cone with radius, slant height, and height in arithmetic progression -/
theorem cone_volume_arithmetic_progression (r s h d : ℝ) (π : ℝ) : 
  (s = r + d) → (h = r + 2*d) → (0 < r) → (0 < d) → (0 < π) →
  (1/3 : ℝ) * π * r^2 * h = (1/3 : ℝ) * π * (r^3 + 2*d*r^2) :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_arithmetic_progression_l2882_288249


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l2882_288204

theorem quadratic_inequality_solution_condition (d : ℝ) :
  d > 0 →
  (∃ x : ℝ, x^2 - 8*x + d < 0) ↔ d < 16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l2882_288204


namespace NUMINAMATH_CALUDE_problem_solution_l2882_288237

theorem problem_solution : 
  (Real.sqrt 8 - Real.sqrt 2 + 2 * Real.sqrt (1/2) = 2 * Real.sqrt 2) ∧
  (Real.sqrt 12 - 9 * Real.sqrt (1/3) + |2 - Real.sqrt 3| = 2 - 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2882_288237


namespace NUMINAMATH_CALUDE_total_pears_is_105_l2882_288289

/-- The number of pears picked by Jason -/
def jason_pears : ℕ := 46

/-- The number of pears picked by Keith -/
def keith_pears : ℕ := 47

/-- The number of pears picked by Mike -/
def mike_pears : ℕ := 12

/-- The total number of pears picked -/
def total_pears : ℕ := jason_pears + keith_pears + mike_pears

/-- Theorem stating that the total number of pears picked is 105 -/
theorem total_pears_is_105 : total_pears = 105 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_is_105_l2882_288289


namespace NUMINAMATH_CALUDE_jim_bought_three_pictures_l2882_288278

def total_pictures : ℕ := 10
def probability_not_bought : ℚ := 21/45

theorem jim_bought_three_pictures :
  ∀ x : ℕ,
  x ≤ total_pictures →
  (total_pictures - x : ℚ) * (total_pictures - 1 - x) / (total_pictures * (total_pictures - 1)) = probability_not_bought →
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_jim_bought_three_pictures_l2882_288278


namespace NUMINAMATH_CALUDE_vacant_seats_l2882_288214

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) (vacant_seats : ℕ) : 
  total_seats = 700 →
  filled_percentage = 75 / 100 →
  vacant_seats = total_seats - (filled_percentage * total_seats).floor →
  vacant_seats = 175 := by
  sorry

end NUMINAMATH_CALUDE_vacant_seats_l2882_288214


namespace NUMINAMATH_CALUDE_scientific_notation_of_388800_l2882_288260

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_388800 :
  toScientificNotation 388800 = ScientificNotation.mk 3.888 5 sorry := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_388800_l2882_288260


namespace NUMINAMATH_CALUDE_complex_expression_value_l2882_288268

theorem complex_expression_value : 
  (1 : ℝ) * (2 * 7 / 9) ^ (1 / 2 : ℝ) - (2 * Real.sqrt 3 - Real.pi) ^ (0 : ℝ) - 
  (2 * 10 / 27) ^ (-(2 / 3 : ℝ)) + (1 / 4 : ℝ) ^ (-(3 / 2 : ℝ)) = 389 / 48 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_value_l2882_288268


namespace NUMINAMATH_CALUDE_expand_product_l2882_288277

theorem expand_product (x : ℝ) : (x + 5) * (x + 9) = x^2 + 14*x + 45 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2882_288277


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2882_288226

theorem arithmetic_sequence_sum : 
  ∀ (a₁ aₙ d n : ℤ), 
  a₁ = -45 → 
  aₙ = -1 → 
  d = 2 → 
  n = (aₙ - a₁) / d + 1 → 
  n * (a₁ + aₙ) / 2 = -529 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2882_288226


namespace NUMINAMATH_CALUDE_sports_club_tennis_players_l2882_288205

/-- Given a sports club with the following properties:
  * There are 80 total members
  * 48 members play badminton
  * 7 members play neither badminton nor tennis
  * 21 members play both badminton and tennis
  Prove that 46 members play tennis -/
theorem sports_club_tennis_players (total : ℕ) (badminton : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 80)
  (h2 : badminton = 48)
  (h3 : neither = 7)
  (h4 : both = 21) :
  total - neither - (badminton - both) = 46 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_tennis_players_l2882_288205


namespace NUMINAMATH_CALUDE_distance_between_cities_l2882_288227

/-- The distance between two cities given two trains traveling towards each other -/
theorem distance_between_cities (t : ℝ) (v₁ v₂ : ℝ) (h₁ : t = 4) (h₂ : v₁ = 115) (h₃ : v₂ = 85) :
  (v₁ + v₂) * t = 800 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l2882_288227


namespace NUMINAMATH_CALUDE_man_rowing_speed_l2882_288261

/-- Given a man's speed in still water and downstream speed, calculate his upstream speed -/
theorem man_rowing_speed (v_still : ℝ) (v_downstream : ℝ) (v_upstream : ℝ) : 
  v_still = 31 → v_downstream = 37 → v_upstream = 25 := by
  sorry

#check man_rowing_speed

end NUMINAMATH_CALUDE_man_rowing_speed_l2882_288261


namespace NUMINAMATH_CALUDE_geometric_sequence_sufficient_not_necessary_l2882_288200

/-- Defines a geometric sequence of three real numbers -/
def is_geometric_sequence (a b c : ℝ) : Prop :=
  b ≠ 0 ∧ c / b = b / a

/-- Proves that "a, b, c form a geometric sequence" is a sufficient but not necessary condition for "b^2 = ac" -/
theorem geometric_sequence_sufficient_not_necessary :
  (∀ a b c : ℝ, is_geometric_sequence a b c → b^2 = a*c) ∧
  (∃ a b c : ℝ, b^2 = a*c ∧ ¬is_geometric_sequence a b c) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sufficient_not_necessary_l2882_288200


namespace NUMINAMATH_CALUDE_profit_distribution_l2882_288255

theorem profit_distribution (total_profit : ℕ) (a_prop b_prop c_prop : ℕ) 
  (h_total : total_profit = 20000)
  (h_prop : a_prop = 2 ∧ b_prop = 3 ∧ c_prop = 5) :
  let total_parts := a_prop + b_prop + c_prop
  let part_value := total_profit / total_parts
  let b_share := b_prop * part_value
  let c_share := c_prop * part_value
  c_share - b_share = 4000 := by
sorry

end NUMINAMATH_CALUDE_profit_distribution_l2882_288255


namespace NUMINAMATH_CALUDE_two_is_only_22_2_sum_of_squares_l2882_288223

/-- A number is of the form 22...2 if it consists of one or more 2's. -/
def is_form_22_2 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 + 10 * (2 * (10^k - 1) / 9)

/-- A number is a perfect square if it's equal to some integer squared. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^2

/-- The main theorem stating that 2 is the only number of the form 22...2
    that can be expressed as the sum of two perfect squares. -/
theorem two_is_only_22_2_sum_of_squares :
  ∀ n : ℕ, is_form_22_2 n ∧ (∃ a b : ℕ, n = a^2 + b^2) ↔ n = 2 := by
  sorry


end NUMINAMATH_CALUDE_two_is_only_22_2_sum_of_squares_l2882_288223


namespace NUMINAMATH_CALUDE_geometric_sequence_tan_property_l2882_288251

/-- Given a geometric sequence {a_n} where a₂a₆ + 2a₄² = π, prove that tan(a₃a₅) = √3 -/
theorem geometric_sequence_tan_property (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n)
  (h_condition : a 2 * a 6 + 2 * (a 4)^2 = Real.pi) :
  Real.tan (a 3 * a 5) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_tan_property_l2882_288251


namespace NUMINAMATH_CALUDE_trivia_team_tryouts_l2882_288267

theorem trivia_team_tryouts (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ) : 
  not_picked = 10 → groups = 8 → students_per_group = 6 → 
  not_picked + groups * students_per_group = 58 := by
sorry

end NUMINAMATH_CALUDE_trivia_team_tryouts_l2882_288267


namespace NUMINAMATH_CALUDE_f_max_and_inequality_l2882_288257

def f (x : ℝ) : ℝ := |x - 1| - 2 * |x + 1|

theorem f_max_and_inequality :
  (∃ (a : ℝ), ∀ x, f x ≤ a ∧ ∃ y, f y = a) ∧
  (∀ m n : ℝ, m > 0 → n > 0 → 1/m + 1/(2*n) = 2 → 2*m + n > 2) :=
sorry

end NUMINAMATH_CALUDE_f_max_and_inequality_l2882_288257


namespace NUMINAMATH_CALUDE_sin_300_degrees_l2882_288258

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l2882_288258


namespace NUMINAMATH_CALUDE_point_order_on_increasing_line_a_less_than_b_l2882_288285

/-- A line in 2D space defined by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a given line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

theorem point_order_on_increasing_line
  (l : Line)
  (p1 p2 : Point)
  (h_slope : l.slope > 0)
  (h_x : p1.x < p2.x)
  (h_on1 : p1.liesOn l)
  (h_on2 : p2.liesOn l) :
  p1.y < p2.y :=
sorry

theorem a_less_than_b :
  let l : Line := { slope := 2/3, intercept := -3 }
  let p1 : Point := { x := -1, y := a }
  let p2 : Point := { x := 1/2, y := b }
  p1.liesOn l → p2.liesOn l → a < b :=
sorry

end NUMINAMATH_CALUDE_point_order_on_increasing_line_a_less_than_b_l2882_288285


namespace NUMINAMATH_CALUDE_incorrect_number_calculation_l2882_288220

theorem incorrect_number_calculation (n : ℕ) (incorrect_avg correct_avg correct_num : ℝ) (X : ℝ) :
  n = 10 →
  incorrect_avg = 18 →
  correct_avg = 22 →
  correct_num = 66 →
  n * incorrect_avg = (n - 1) * correct_avg + X →
  n * correct_avg = (n - 1) * correct_avg + correct_num →
  X = 26 := by
    sorry

end NUMINAMATH_CALUDE_incorrect_number_calculation_l2882_288220


namespace NUMINAMATH_CALUDE_specific_device_works_prob_l2882_288210

/-- A device with two components, each having a probability of failure --/
structure Device where
  component_failure_prob : ℝ
  num_components : ℕ

/-- The probability that the device works --/
def device_works_prob (d : Device) : ℝ :=
  (1 - d.component_failure_prob) ^ d.num_components

/-- Theorem: The probability that a specific device works is 0.81 --/
theorem specific_device_works_prob :
  ∃ (d : Device), device_works_prob d = 0.81 := by
  sorry

end NUMINAMATH_CALUDE_specific_device_works_prob_l2882_288210


namespace NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_301_l2882_288288

theorem multiplicative_inverse_203_mod_301 : ∃ x : ℕ, x < 301 ∧ (203 * x) % 301 = 1 :=
by
  use 29
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_301_l2882_288288


namespace NUMINAMATH_CALUDE_fifi_hangers_l2882_288299

theorem fifi_hangers (total green blue yellow pink : ℕ) : 
  total = 16 →
  green = 4 →
  blue = green - 1 →
  yellow = blue - 1 →
  total = green + blue + yellow + pink →
  pink = 7 := by
sorry

end NUMINAMATH_CALUDE_fifi_hangers_l2882_288299


namespace NUMINAMATH_CALUDE_min_plates_for_seven_colors_l2882_288203

/-- The minimum number of plates needed to guarantee at least three matching pairs -/
def min_plates_for_three_pairs (num_colors : ℕ) : ℕ :=
  3 * num_colors + 3

/-- Theorem stating that given 7 different colors of plates, 
    the minimum number of plates needed to guarantee at least three matching pairs is 24 -/
theorem min_plates_for_seven_colors : 
  min_plates_for_three_pairs 7 = 24 := by
  sorry

#eval min_plates_for_three_pairs 7

end NUMINAMATH_CALUDE_min_plates_for_seven_colors_l2882_288203


namespace NUMINAMATH_CALUDE_order_parts_count_l2882_288296

-- Define the master's productivity per hour
def master_productivity : ℕ → Prop :=
  λ y => y > 5

-- Define the apprentice's productivity relative to the master
def apprentice_productivity (y : ℕ) : ℕ := y - 2

-- Define the total number of parts in the order
def total_parts (y : ℕ) : ℕ := 2 * y * (y - 2) / (y - 4)

-- Theorem statement
theorem order_parts_count :
  ∀ y : ℕ,
    master_productivity y →
    (∃ t : ℕ, t * y = total_parts y) →
    2 * (apprentice_productivity y) * (t - 1) = total_parts y →
    total_parts y = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_order_parts_count_l2882_288296


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l2882_288250

def senate_committee_size : ℕ := 18
def num_republicans : ℕ := 10
def num_democrats : ℕ := 8
def subcommittee_size : ℕ := 5
def min_republicans : ℕ := 2

theorem subcommittee_formation_count :
  (Finset.range (subcommittee_size - min_republicans + 1)).sum (λ k =>
    Nat.choose num_republicans (min_republicans + k) *
    Nat.choose num_democrats (subcommittee_size - (min_republicans + k))
  ) = 7812 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l2882_288250


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2882_288253

theorem quadratic_equation_solution (b : ℚ) : 
  ((-8 : ℚ)^2 + b * (-8) - 45 = 0) → b = 19/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2882_288253


namespace NUMINAMATH_CALUDE_inverse_of_A_cubed_l2882_288229

variable (A : Matrix (Fin 2) (Fin 2) ℝ)

theorem inverse_of_A_cubed 
  (h : A⁻¹ = ![![-3, 2], ![-1, 3]]) : 
  (A^3)⁻¹ = ![![-21, 14], ![-7, 21]] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_cubed_l2882_288229


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l2882_288276

theorem sum_of_squares_and_products (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x^2 + y^2 + z^2 = 75)
  (h5 : x*y + y*z + z*x = 28) :
  x + y + z = Real.sqrt 131 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l2882_288276


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2882_288256

/-- Two vectors are parallel if their corresponding components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a.1 * b.2 = k * a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (m, 2)
  parallel a b → m = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2882_288256


namespace NUMINAMATH_CALUDE_inverse_matrices_sum_l2882_288281

/-- Two 3x3 matrices that are inverses of each other -/
def A (x y z w : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x, 2, y],
    ![3, 3, 4],
    ![z, 6, w]]

def B (j k l m : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![-6, j, -12],
    ![k, -14, l],
    ![3, m, 5]]

/-- The theorem stating that the sum of all variables in the inverse matrices equals 52 -/
theorem inverse_matrices_sum (x y z w j k l m : ℝ) :
  (A x y z w) * (B j k l m) = 1 →
  x + y + z + w + j + k + l + m = 52 := by
  sorry

end NUMINAMATH_CALUDE_inverse_matrices_sum_l2882_288281


namespace NUMINAMATH_CALUDE_inverse_prop_percent_change_l2882_288298

/-- Theorem: Inverse Proportionality and Percentage Change

Given:
- x and y are inversely proportional and positive
- x decreases by q%

Prove:
y increases by (100q)/(100-q)%
-/
theorem inverse_prop_percent_change (x y k q : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_q : 0 < q) (h_q_lt_100 : q < 100)
  (h_inverse_prop : x * y = k) :
  let x' := x * (1 - q / 100)
  let y' := k / x'
  (y' - y) / y * 100 = 100 * q / (100 - q) :=
by sorry

end NUMINAMATH_CALUDE_inverse_prop_percent_change_l2882_288298


namespace NUMINAMATH_CALUDE_unique_integer_solution_l2882_288272

theorem unique_integer_solution (a : ℤ) : 
  (∃! x : ℤ, |a*x + a + 2| < 2) ↔ (a = 3 ∨ a = -3) :=
sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l2882_288272


namespace NUMINAMATH_CALUDE_part_one_part_two_l2882_288209

-- Define sets A and B
def A (a b : ℝ) : Set ℝ := {x | a - b < x ∧ x < a + b}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Part 1
theorem part_one (a : ℝ) : 
  (A a 1 ∩ B = A a 1) → (a ≤ -2 ∨ a ≥ 6) := by sorry

-- Part 2
theorem part_two (b : ℝ) :
  (A 1 b ∩ B = ∅) → (b ≤ 2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2882_288209


namespace NUMINAMATH_CALUDE_sum_f_negative_l2882_288295

/-- The function f(x) = -x^3 - x -/
def f (x : ℝ) : ℝ := -x^3 - x

/-- Theorem: For a, b, c ∈ ℝ satisfying a + b > 0, b + c > 0, and c + a > 0,
    it follows that f(a) + f(b) + f(c) < 0 -/
theorem sum_f_negative (a b c : ℝ) (hab : a + b > 0) (hbc : b + c > 0) (hca : c + a > 0) :
  f a + f b + f c < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_negative_l2882_288295


namespace NUMINAMATH_CALUDE_solve_system_l2882_288270

theorem solve_system (a b : ℚ) 
  (eq1 : 3 * a + 2 * b = 25)
  (eq2 : 5 * a + b = 20) :
  3 * a + 3 * b = 240 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2882_288270


namespace NUMINAMATH_CALUDE_cricket_runs_l2882_288290

theorem cricket_runs (a b c : ℕ) : 
  a + b + c = 95 →
  3 * a = b →
  5 * b = c →
  c = 75 := by
sorry

end NUMINAMATH_CALUDE_cricket_runs_l2882_288290


namespace NUMINAMATH_CALUDE_output_is_three_l2882_288208

def program_output (a b : ℕ) : ℕ := a + b

theorem output_is_three : program_output 1 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_output_is_three_l2882_288208


namespace NUMINAMATH_CALUDE_triangle_properties_l2882_288283

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Define the triangle
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  -- Given conditions
  a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A →
  b + c = Real.sqrt 10 →
  a = 2 →
  -- Prove
  Real.cos A = 1/2 ∧
  (1/2 : ℝ) * b * c * Real.sin A = (7 * Real.sqrt 3) / 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2882_288283


namespace NUMINAMATH_CALUDE_no_two_digit_sum_with_reverse_is_cube_l2882_288259

/-- Function to reverse the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Function to check if a number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

/-- Theorem: No two-digit positive integer N has the property that
    the sum of N and its digit-reversed number is a perfect cube -/
theorem no_two_digit_sum_with_reverse_is_cube :
  ¬∃ N : ℕ, 10 ≤ N ∧ N < 100 ∧ isPerfectCube (N + reverseDigits N) := by
  sorry

end NUMINAMATH_CALUDE_no_two_digit_sum_with_reverse_is_cube_l2882_288259


namespace NUMINAMATH_CALUDE_average_mark_proof_l2882_288236

/-- Given an examination with 50 candidates and a total of 2000 marks,
    prove that the average mark obtained by each candidate is 40. -/
theorem average_mark_proof (candidates : ℕ) (total_marks : ℕ) :
  candidates = 50 →
  total_marks = 2000 →
  (total_marks : ℚ) / (candidates : ℚ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_mark_proof_l2882_288236


namespace NUMINAMATH_CALUDE_infinite_solutions_exponential_equation_l2882_288228

theorem infinite_solutions_exponential_equation :
  ∀ x : ℝ, (2 : ℝ) ^ (6 * x + 3) * (4 : ℝ) ^ (3 * x + 6) = (8 : ℝ) ^ (4 * x + 5) := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_exponential_equation_l2882_288228


namespace NUMINAMATH_CALUDE_polynomial_characterization_l2882_288252

/-- A polynomial satisfying the given condition -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ (a b c : ℝ), a*b + b*c + c*a = 0 →
    P (a-b) + P (b-c) + P (c-a) = 2 * P (a+b+c)

/-- The form of the polynomial that satisfies the condition -/
def PolynomialForm (P : ℝ → ℝ) : Prop :=
  ∃ (r s : ℝ), ∀ x, P x = r * x^4 + s * x^2

theorem polynomial_characterization :
  ∀ P : ℝ → ℝ, SatisfyingPolynomial P ↔ PolynomialForm P :=
sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l2882_288252


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2882_288271

theorem contrapositive_equivalence (m : ℝ) :
  (¬(∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) ↔
  (m > 0 → ∃ x : ℝ, x^2 + x - m = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2882_288271


namespace NUMINAMATH_CALUDE_equation_solutions_l2882_288206

theorem equation_solutions :
  (∃ x₁ x₂, x₁ = -3/2 ∧ x₂ = 2 ∧ 2 * x₁^2 - x₁ - 6 = 0 ∧ 2 * x₂^2 - x₂ - 6 = 0) ∧
  (∃ y₁ y₂, y₁ = -1 ∧ y₂ = 1/2 ∧ (y₁ - 2)^2 = 9 * y₁^2 ∧ (y₂ - 2)^2 = 9 * y₂^2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2882_288206


namespace NUMINAMATH_CALUDE_sophia_pie_consumption_l2882_288201

theorem sophia_pie_consumption (pie_weight : ℝ) (fridge_weight : ℝ) : 
  fridge_weight = (5/6) * pie_weight ∧ fridge_weight = 1200 → 
  pie_weight - fridge_weight = 240 := by
sorry

end NUMINAMATH_CALUDE_sophia_pie_consumption_l2882_288201


namespace NUMINAMATH_CALUDE_samuel_journey_length_l2882_288262

/-- Represents a journey divided into three parts -/
structure Journey where
  first_part : ℚ  -- Fraction of the total journey
  middle_part : ℚ  -- Length in miles
  last_part : ℚ  -- Fraction of the total journey

/-- Calculates the total length of a journey -/
def journey_length (j : Journey) : ℚ :=
  j.middle_part / (1 - j.first_part - j.last_part)

theorem samuel_journey_length :
  let j : Journey := {
    first_part := 1/4,
    middle_part := 30,
    last_part := 1/6
  }
  journey_length j = 360/7 := by
  sorry

end NUMINAMATH_CALUDE_samuel_journey_length_l2882_288262


namespace NUMINAMATH_CALUDE_parabola_dot_product_zero_l2882_288233

/-- A point on the parabola y^2 = 4x -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x

/-- The line passing through two points intersects (4,0) -/
def line_through_four (A B : ParabolaPoint) : Prop :=
  ∃ t : ℝ, A.x + t * (B.x - A.x) = 4 ∧ A.y + t * (B.y - A.y) = 0

/-- The dot product of vectors OA and OB -/
def dot_product (A B : ParabolaPoint) : ℝ :=
  A.x * B.x + A.y * B.y

theorem parabola_dot_product_zero (A B : ParabolaPoint) 
  (h : line_through_four A B) : dot_product A B = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_dot_product_zero_l2882_288233


namespace NUMINAMATH_CALUDE_length_AB_trajectory_C_l2882_288232

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 5*y^2 = 5

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (ellipse_foci : A = (-2, 0) ∧ B = (2, 0))
  (angle_relation : ∀ (θA θB θC : ℝ), 
    Real.sin θB - Real.sin θA = Real.sin θC → 
    θA + θB + θC = π)

-- Statement 1: Length of AB is 4
theorem length_AB (t : Triangle) : 
  Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2) = 4 :=
sorry

-- Statement 2: Trajectory of C
theorem trajectory_C (t : Triangle) (x y : ℝ) :
  (∃ (C : ℝ × ℝ), t.C = C ∧ x > 1) →
  (x^2 - y^2/3 = 1) :=
sorry

end NUMINAMATH_CALUDE_length_AB_trajectory_C_l2882_288232


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2882_288239

theorem line_passes_through_fixed_point (p q : ℝ) (h : p + 2*q - 1 = 0) :
  ∃ (x y : ℝ), x = 1/2 ∧ y = -1/6 ∧ p*x + 3*y + q = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2882_288239


namespace NUMINAMATH_CALUDE_farm_animals_l2882_288275

theorem farm_animals (total_legs : ℕ) (total_animals : ℕ) (chicken_legs : ℕ) (sheep_legs : ℕ) :
  total_legs = 60 →
  total_animals = 20 →
  chicken_legs = 2 →
  sheep_legs = 4 →
  ∃ (num_chickens num_sheep : ℕ),
    num_chickens + num_sheep = total_animals ∧
    num_chickens * chicken_legs + num_sheep * sheep_legs = total_legs ∧
    num_sheep = 10 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l2882_288275


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2882_288244

/-- Represents the side lengths of an isosceles triangle -/
structure IsoscelesTriangle where
  equalSide : ℝ
  baseSide : ℝ

/-- Checks if the triangle satisfies the given conditions -/
def satisfiesConditions (t : IsoscelesTriangle) : Prop :=
  t.equalSide = 20 ∧ t.baseSide = (2/5) * t.equalSide

/-- Calculates the perimeter of the triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  2 * t.equalSide + t.baseSide

/-- Theorem stating that the perimeter of the triangle is 48 cm -/
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, satisfiesConditions t → perimeter t = 48 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2882_288244


namespace NUMINAMATH_CALUDE_triangle_area_l2882_288234

/-- Given a triangle ABC with angle A = 60°, side b = 4, and side a = 2√3, 
    prove that its area is 2√3 square units. -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  A = π / 3 →  -- 60° in radians
  b = 4 → 
  a = 2 * Real.sqrt 3 → 
  (1 / 2) * a * b * Real.sin C = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2882_288234


namespace NUMINAMATH_CALUDE_peanuts_in_box_l2882_288254

/-- The number of peanuts in a box after adding more -/
theorem peanuts_in_box (initial : ℕ) (added : ℕ) : 
  initial = 4 → added = 12 → initial + added = 16 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_box_l2882_288254


namespace NUMINAMATH_CALUDE_arrow_sequence_equivalence_l2882_288245

/-- Represents a point in the cycle -/
def CyclePoint := ℕ

/-- The length of the cycle -/
def cycleLength : ℕ := 5

/-- Returns the equivalent point within the cycle -/
def cycleEquivalent (n : ℕ) : CyclePoint :=
  n % cycleLength

/-- Theorem: The sequence of arrows from point 630 to point 633 is equivalent
    to the sequence from point 0 to point 3 in a cycle of length 5 -/
theorem arrow_sequence_equivalence :
  (cycleEquivalent 630 = cycleEquivalent 0) ∧
  (cycleEquivalent 631 = cycleEquivalent 1) ∧
  (cycleEquivalent 632 = cycleEquivalent 2) ∧
  (cycleEquivalent 633 = cycleEquivalent 3) := by
  sorry


end NUMINAMATH_CALUDE_arrow_sequence_equivalence_l2882_288245


namespace NUMINAMATH_CALUDE_trapezoid_area_is_80_l2882_288221

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid :=
  (long_base : ℝ)
  (base_angle : ℝ)
  (h : 0 < long_base)
  (angle_h : 0 < base_angle ∧ base_angle < π / 2)

/-- The area of an isosceles trapezoid -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

theorem trapezoid_area_is_80 (t : IsoscelesTrapezoid) 
  (h1 : t.long_base = 16)
  (h2 : t.base_angle = Real.arcsin 0.8) :
  trapezoid_area t = 80 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_80_l2882_288221


namespace NUMINAMATH_CALUDE_squirrel_cones_problem_l2882_288242

theorem squirrel_cones_problem :
  ∃ (x y : ℕ), 
    x + y < 25 ∧
    2 * x > y + 26 ∧
    2 * y > x - 4 ∧
    x = 17 ∧
    y = 7 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_cones_problem_l2882_288242


namespace NUMINAMATH_CALUDE_parabola_properties_l2882_288293

/-- Parabola represented by y = -3x^2 - 6x + 2 -/
def parabola (x : ℝ) : ℝ := -3 * x^2 - 6 * x + 2

theorem parabola_properties :
  (∃ (x_max y_max : ℝ),
    (∀ x, parabola x ≤ parabola x_max) ∧
    parabola x_max = y_max ∧
    x_max = 1 ∧
    y_max = -7) ∧
  (∀ x, parabola (2 - x) = parabola x) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2882_288293


namespace NUMINAMATH_CALUDE_books_selling_price_l2882_288265

/-- Calculates the total selling price of two books given their costs and profit/loss percentages -/
def total_selling_price (total_cost book1_cost loss_percent gain_percent : ℚ) : ℚ :=
  let book2_cost := total_cost - book1_cost
  let book1_sell := book1_cost * (1 - loss_percent / 100)
  let book2_sell := book2_cost * (1 + gain_percent / 100)
  book1_sell + book2_sell

/-- Theorem stating that the total selling price of two books is 297.50 Rs given the specified conditions -/
theorem books_selling_price :
  total_selling_price 300 175 15 19 = 297.50 := by
  sorry

end NUMINAMATH_CALUDE_books_selling_price_l2882_288265


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l2882_288297

theorem no_positive_integer_solutions : 
  ¬ ∃ (a b c : ℕ+), (a * b + b * c = 66) ∧ (a * c + b * c = 35) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l2882_288297


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2882_288225

theorem complex_equation_solution (a b : ℝ) (z : ℂ) :
  (∀ x : ℂ, x^2 + (4 + Complex.I) * x + 4 + a * Complex.I = 0 → x.im = 0) →
  z = a + b * Complex.I →
  (b : ℂ)^2 + (4 + Complex.I) * b + 4 + a * Complex.I = 0 →
  z = 2 - 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2882_288225


namespace NUMINAMATH_CALUDE_representation_of_2020_as_sum_of_five_cubes_l2882_288286

theorem representation_of_2020_as_sum_of_five_cubes :
  ∃ (n : ℤ), 2020 = (n + 2)^3 + n^3 + (-n - 1)^3 + (-n - 1)^3 + (-2)^3 :=
by
  use 337
  sorry

end NUMINAMATH_CALUDE_representation_of_2020_as_sum_of_five_cubes_l2882_288286


namespace NUMINAMATH_CALUDE_library_books_remaining_l2882_288287

/-- Calculates the number of remaining books after two days of borrowing. -/
def remaining_books (initial : ℕ) (day1_borrowed : ℕ) (day2_borrowed : ℕ) : ℕ :=
  initial - (day1_borrowed + day2_borrowed)

/-- Theorem stating the number of remaining books in the library scenario. -/
theorem library_books_remaining :
  remaining_books 100 10 20 = 70 := by
  sorry

end NUMINAMATH_CALUDE_library_books_remaining_l2882_288287


namespace NUMINAMATH_CALUDE_orange_removal_problem_l2882_288247

/-- Represents the number of oranges Mary must put back to achieve the desired average price -/
def oranges_to_remove : ℕ := sorry

/-- The price of an apple in cents -/
def apple_price : ℕ := 50

/-- The price of an orange in cents -/
def orange_price : ℕ := 60

/-- The total number of fruits initially selected -/
def total_fruits : ℕ := 10

/-- The initial average price of the fruits in cents -/
def initial_avg_price : ℕ := 56

/-- The desired average price after removing oranges in cents -/
def desired_avg_price : ℕ := 52

theorem orange_removal_problem :
  ∃ (apples oranges : ℕ),
    apples + oranges = total_fruits ∧
    (apple_price * apples + orange_price * oranges) / total_fruits = initial_avg_price ∧
    (apple_price * apples + orange_price * (oranges - oranges_to_remove)) / (total_fruits - oranges_to_remove) = desired_avg_price ∧
    oranges_to_remove = 5 := by sorry

end NUMINAMATH_CALUDE_orange_removal_problem_l2882_288247


namespace NUMINAMATH_CALUDE_complex_modulus_equation_solution_l2882_288263

theorem complex_modulus_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ Complex.abs (5 - 3 * Complex.I * x) = 7 ∧ x = Real.sqrt (8/3) := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_solution_l2882_288263


namespace NUMINAMATH_CALUDE_range_of_a_l2882_288292

/-- The function f(x) = x^2 - 2x --/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The function g(x) = ax + 2, where a > 0 --/
def g (a : ℝ) (x : ℝ) : ℝ := a*x + 2

/-- The theorem stating the range of a --/
theorem range_of_a (a : ℝ) : 
  (a > 0 ∧ 
   ∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₀ ∈ Set.Icc (-1 : ℝ) 2, g a x₁ = f x₀) → 
  (1/2 : ℝ) ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2882_288292


namespace NUMINAMATH_CALUDE_lake_distance_difference_l2882_288240

/-- The difference between the circumference of a circle with diameter 2 miles
    and its diameter, given π = 3.14 -/
theorem lake_distance_difference : 
  let π : ℝ := 3.14
  let diameter : ℝ := 2
  let circumference := π * diameter
  circumference - diameter = 4.28 := by sorry

end NUMINAMATH_CALUDE_lake_distance_difference_l2882_288240


namespace NUMINAMATH_CALUDE_max_prime_difference_l2882_288294

def is_prime (n : ℕ) : Prop := sorry

def are_distinct {α : Type*} (l : List α) : Prop := sorry

theorem max_prime_difference (a b c : ℕ) (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : is_prime a ∧ is_prime b ∧ is_prime c ∧ 
        is_prime (a+b-c) ∧ is_prime (a+c-b) ∧ is_prime (b+c-a) ∧ is_prime (a+b+c))
  (h3 : are_distinct [a, b, c, a+b-c, a+c-b, b+c-a, a+b+c])
  (h4 : (a + b = 800) ∨ (a + c = 800) ∨ (b + c = 800)) :
  ∃ d : ℕ, d ≤ 1594 ∧ 
  d = max (a+b+c) (max a (max b (max c (max (a+b-c) (max (a+c-b) (b+c-a)))))) -
      min (a+b+c) (min a (min b (min c (min (a+b-c) (min (a+c-b) (b+c-a)))))) ∧
  ∀ d' : ℕ, d' ≤ d := by sorry

end NUMINAMATH_CALUDE_max_prime_difference_l2882_288294


namespace NUMINAMATH_CALUDE_x_value_l2882_288248

theorem x_value : ∃ x : ℚ, (2 / 5 * x) - (1 / 3 * x) = 110 ∧ x = 1650 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2882_288248


namespace NUMINAMATH_CALUDE_boys_in_class_l2882_288213

theorem boys_in_class (total : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) (h1 : total = 49) (h2 : ratio_boys = 4) (h3 : ratio_girls = 3) : 
  (ratio_boys * total) / (ratio_boys + ratio_girls) = 28 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_class_l2882_288213


namespace NUMINAMATH_CALUDE_multiple_condition_l2882_288282

theorem multiple_condition (n : ℕ) : 
  n = 1475 → 0 < n → n < 2006 → ∃ k : ℕ, 2006 * n = k * (2006 + n) :=
sorry

end NUMINAMATH_CALUDE_multiple_condition_l2882_288282


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2882_288207

/-- Given a quadratic expression of the form 16x^2 - bx + 9, 
    prove that it is a perfect square trinomial if and only if b = ±24 -/
theorem perfect_square_condition (b : ℝ) : 
  (∃ (k : ℝ), ∀ (x : ℝ), 16 * x^2 - b * x + 9 = (k * x + 3)^2) ↔ (b = 24 ∨ b = -24) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2882_288207
