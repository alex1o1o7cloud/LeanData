import Mathlib

namespace NUMINAMATH_CALUDE_average_net_sales_proof_l869_86942

def monthly_sales : List ℕ := [120, 80, 50, 130, 90, 160]
def monthly_expense : ℕ := 30
def num_months : ℕ := 6

theorem average_net_sales_proof :
  (List.sum monthly_sales - monthly_expense * num_months) / num_months = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_net_sales_proof_l869_86942


namespace NUMINAMATH_CALUDE_wood_measurement_l869_86908

theorem wood_measurement (x y : ℝ) : 
  (y = x + 4.5 ∧ 0.5 * y = x - 1) ↔ 
  (∃ (wood_length rope_length : ℝ), 
    wood_length = x ∧ 
    rope_length = y ∧ 
    rope_length - wood_length = 4.5 ∧ 
    0.5 * rope_length - wood_length = -1) :=
by sorry

end NUMINAMATH_CALUDE_wood_measurement_l869_86908


namespace NUMINAMATH_CALUDE_fish_caught_l869_86975

theorem fish_caught (initial_fish : ℕ) (initial_tadpoles : ℕ) (fish_caught : ℕ) : 
  initial_fish = 50 →
  initial_tadpoles = 3 * initial_fish →
  initial_tadpoles / 2 = (initial_fish - fish_caught) + 32 →
  fish_caught = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_fish_caught_l869_86975


namespace NUMINAMATH_CALUDE_line_through_points_l869_86989

/-- Given a line y = ax + b passing through points (3, 2) and (7, 26), prove that a - b = 22 -/
theorem line_through_points (a b : ℝ) : 
  (2 : ℝ) = a * 3 + b ∧ (26 : ℝ) = a * 7 + b → a - b = 22 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l869_86989


namespace NUMINAMATH_CALUDE_grocery_store_costs_l869_86948

theorem grocery_store_costs (total_costs delivery_fraction orders_cost : ℚ)
  (h1 : total_costs = 4000)
  (h2 : orders_cost = 1800)
  (h3 : delivery_fraction = 1/4) :
  let remaining_after_orders := total_costs - orders_cost
  let delivery_cost := delivery_fraction * remaining_after_orders
  let salary_cost := remaining_after_orders - delivery_cost
  salary_cost / total_costs = 33/80 := by
sorry

end NUMINAMATH_CALUDE_grocery_store_costs_l869_86948


namespace NUMINAMATH_CALUDE_radical_axis_is_line_l869_86991

/-- The locus of points with equal power with respect to two non-concentric circles is a line -/
theorem radical_axis_is_line (R₁ R₂ a : ℝ) (ha : a ≠ 0) :
  ∃ k : ℝ, ∀ x y : ℝ, (x + a)^2 + y^2 - R₁^2 = (x - a)^2 + y^2 - R₂^2 ↔ x = k :=
sorry

end NUMINAMATH_CALUDE_radical_axis_is_line_l869_86991


namespace NUMINAMATH_CALUDE_expression_bound_l869_86905

theorem expression_bound (x : ℝ) (h : x^2 - 7*x + 12 ≤ 0) : 
  40 ≤ x^2 + 7*x + 10 ∧ x^2 + 7*x + 10 ≤ 54 := by
sorry

end NUMINAMATH_CALUDE_expression_bound_l869_86905


namespace NUMINAMATH_CALUDE_combined_experience_l869_86934

def james_experience : ℕ := 20

def john_experience (james_current : ℕ) : ℕ := 2 * (james_current - 8) + 8

def mike_experience (john_current : ℕ) : ℕ := john_current - 16

theorem combined_experience :
  james_experience + john_experience james_experience + mike_experience (john_experience james_experience) = 68 :=
by sorry

end NUMINAMATH_CALUDE_combined_experience_l869_86934


namespace NUMINAMATH_CALUDE_quadrupled_exponent_base_l869_86986

theorem quadrupled_exponent_base (c d y : ℝ) (hc : c > 0) (hd : d > 0) (hy : y > 0) 
  (h : (4 * c)^(4 * d) = c^d * y^d) : y = 256 * c^3 := by
  sorry

end NUMINAMATH_CALUDE_quadrupled_exponent_base_l869_86986


namespace NUMINAMATH_CALUDE_pine_percentage_correct_l869_86916

/-- Represents the number of trees of each type in the forest -/
structure ForestComposition where
  oak : ℕ
  pine : ℕ
  spruce : ℕ
  birch : ℕ

/-- The total number of trees in the forest -/
def total_trees : ℕ := 4000

/-- The actual composition of the forest -/
def forest : ForestComposition := {
  oak := 720,
  pine := 520,
  spruce := 400,
  birch := 2160
}

/-- The percentage of pine trees in the forest -/
def pine_percentage : ℚ := 13 / 100

theorem pine_percentage_correct :
  (forest.oak + forest.pine + forest.spruce + forest.birch = total_trees) ∧
  (forest.spruce = total_trees / 10) ∧
  (forest.oak = forest.spruce + forest.pine) ∧
  (forest.birch = 2160) →
  (forest.pine : ℚ) / total_trees = pine_percentage :=
by sorry

end NUMINAMATH_CALUDE_pine_percentage_correct_l869_86916


namespace NUMINAMATH_CALUDE_michael_passes_donovan_l869_86926

/-- The length of the circular track in meters -/
def track_length : ℝ := 500

/-- Donovan's lap time in seconds -/
def donovan_lap_time : ℝ := 45

/-- Michael's lap time in seconds -/
def michael_lap_time : ℝ := 40

/-- The number of laps Michael needs to complete to pass Donovan -/
def laps_to_pass : ℕ := 9

theorem michael_passes_donovan :
  (laps_to_pass : ℝ) * michael_lap_time = (laps_to_pass - 1 : ℝ) * donovan_lap_time :=
sorry

end NUMINAMATH_CALUDE_michael_passes_donovan_l869_86926


namespace NUMINAMATH_CALUDE_sand_pile_volume_l869_86962

/-- The volume of a cone with diameter 12 feet and height 60% of the diameter is 86.4π cubic feet -/
theorem sand_pile_volume : 
  let diameter : ℝ := 12
  let height : ℝ := 0.6 * diameter
  let radius : ℝ := diameter / 2
  let volume : ℝ := (1/3) * π * radius^2 * height
  volume = 86.4 * π := by sorry

end NUMINAMATH_CALUDE_sand_pile_volume_l869_86962


namespace NUMINAMATH_CALUDE_inequality_proof_l869_86985

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (b + c) * (c + a) * (a + b) ≥ 4 * ((a + b + c) * ((a + b + c) / 3) ^ (1/8) - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l869_86985


namespace NUMINAMATH_CALUDE_james_earnings_l869_86904

/-- James' earnings problem -/
theorem james_earnings (january : ℕ) (february : ℕ) (march : ℕ) 
  (h1 : february = 2 * january)
  (h2 : march = february - 2000)
  (h3 : january + february + march = 18000) :
  january = 4000 := by
  sorry

end NUMINAMATH_CALUDE_james_earnings_l869_86904


namespace NUMINAMATH_CALUDE_seventeen_sum_of_two_primes_l869_86965

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem seventeen_sum_of_two_primes :
  ∃! (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 17 :=
sorry

end NUMINAMATH_CALUDE_seventeen_sum_of_two_primes_l869_86965


namespace NUMINAMATH_CALUDE_one_minus_repeating_eight_eq_one_ninth_l869_86937

/-- The repeating decimal 0.overline{8} -/
def repeating_eight : ℚ := 8/9

/-- Theorem stating that 1 minus the repeating decimal 0.overline{8} equals 1/9 -/
theorem one_minus_repeating_eight_eq_one_ninth : 1 - repeating_eight = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_one_minus_repeating_eight_eq_one_ninth_l869_86937


namespace NUMINAMATH_CALUDE_min_value_expression_equality_achieved_l869_86919

theorem min_value_expression (x : ℝ) : 
  (x + 2) * (x + 3) * (x + 5) * (x + 6) + 2024 ≥ 2021.75 :=
sorry

theorem equality_achieved : 
  ∃ x : ℝ, (x + 2) * (x + 3) * (x + 5) * (x + 6) + 2024 = 2021.75 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_achieved_l869_86919


namespace NUMINAMATH_CALUDE_oneSeventhIncreaseAfterRemoval_l869_86987

/-- The decimal representation of 1/7 -/
def oneSeventhDecimal : ℚ := 1 / 7

/-- The position of the digit to be removed -/
def digitPosition : ℕ := 2021

/-- The function that removes the digit at the specified position and shifts subsequent digits -/
def removeDigitAndShift (q : ℚ) (pos : ℕ) : ℚ :=
  sorry -- Implementation details omitted

/-- Theorem stating that removing the 2021st digit after the decimal point in 1/7 increases the value -/
theorem oneSeventhIncreaseAfterRemoval :
  removeDigitAndShift oneSeventhDecimal digitPosition > oneSeventhDecimal :=
sorry

end NUMINAMATH_CALUDE_oneSeventhIncreaseAfterRemoval_l869_86987


namespace NUMINAMATH_CALUDE_smallest_perimeter_circle_circle_center_on_line_l869_86936

-- Define the points A and B
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (-1, 4)

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 2 * x - y - 4 = 0

-- Define the general equation of a circle
def circle_general_eq (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Define the standard equation of a circle
def circle_standard_eq (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Theorem for the circle with smallest perimeter
theorem smallest_perimeter_circle :
  ∃ (x y : ℝ), x^2 + y^2 - 2*y - 9 = 0 ∧
  circle_general_eq x y 0 1 (5 : ℝ) ∧
  (∀ (a b r : ℝ), circle_general_eq A.1 A.2 a b r → 
   circle_general_eq B.1 B.2 a b r → 
   r^2 ≥ 10) := by sorry

-- Theorem for the circle with center on the given line
theorem circle_center_on_line :
  ∃ (x y : ℝ), (x - 3)^2 + (y - 2)^2 = 20 ∧
  circle_standard_eq x y 3 2 (2 * Real.sqrt 5) ∧
  line_eq 3 2 := by sorry

end NUMINAMATH_CALUDE_smallest_perimeter_circle_circle_center_on_line_l869_86936


namespace NUMINAMATH_CALUDE_carla_cards_theorem_l869_86939

/-- A structure representing a card with two numbers -/
structure Card where
  visible : ℕ
  hidden : ℕ

/-- The setup of Carla's cards -/
def carla_cards : Card × Card :=
  ⟨⟨37, 0⟩, ⟨53, 0⟩⟩

/-- Predicate to check if a number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- Theorem stating the properties of Carla's card setup and the result -/
theorem carla_cards_theorem (cards : Card × Card) : 
  cards = carla_cards →
  (∃ p₁ p₂ : ℕ, 
    is_prime p₁ ∧ 
    is_prime p₂ ∧ 
    p₁ ≠ p₂ ∧
    cards.1.visible + p₁ = cards.2.visible + p₂ ∧
    (p₁ + p₂) / 2 = 11) := by
  sorry

#check carla_cards_theorem

end NUMINAMATH_CALUDE_carla_cards_theorem_l869_86939


namespace NUMINAMATH_CALUDE_total_material_calculation_l869_86972

/-- The amount of concrete ordered in tons -/
def concrete : ℝ := 0.16666666666666666

/-- The amount of bricks ordered in tons -/
def bricks : ℝ := 0.16666666666666666

/-- The amount of stone ordered in tons -/
def stone : ℝ := 0.5

/-- The total amount of material ordered in tons -/
def total_material : ℝ := concrete + bricks + stone

theorem total_material_calculation : total_material = 0.8333333333333332 := by
  sorry

end NUMINAMATH_CALUDE_total_material_calculation_l869_86972


namespace NUMINAMATH_CALUDE_natural_number_divisibility_l869_86995

theorem natural_number_divisibility (a b n : ℕ) 
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) : 
  a = b^n := by sorry

end NUMINAMATH_CALUDE_natural_number_divisibility_l869_86995


namespace NUMINAMATH_CALUDE_fliers_calculation_l869_86998

theorem fliers_calculation (initial_fliers : ℕ) : 
  (initial_fliers : ℚ) * (4/5) * (3/4) = 1800 → initial_fliers = 3000 := by
  sorry

end NUMINAMATH_CALUDE_fliers_calculation_l869_86998


namespace NUMINAMATH_CALUDE_calculation_proof_l869_86988

theorem calculation_proof : 3 * 8 * 9 + 18 / 3 - 2^3 = 214 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l869_86988


namespace NUMINAMATH_CALUDE_fraction_reduction_l869_86999

theorem fraction_reduction (a x : ℝ) :
  (Real.sqrt (a^2 + x^2) - (x^2 - a^2) / Real.sqrt (a^2 + x^2)) / (a^2 + x^2) = 
  2 * a^2 / (a^2 + x^2)^(3/2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_reduction_l869_86999


namespace NUMINAMATH_CALUDE_tan_neg_five_pi_fourth_l869_86915

theorem tan_neg_five_pi_fourth : Real.tan (-5 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_five_pi_fourth_l869_86915


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l869_86924

theorem quadratic_function_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : 0 < a) (h₂ : a < 3) 
  (h₃ : x₁ < x₂) (h₄ : x₁ + x₂ = 1 - a) : 
  let f := fun x => a * x^2 + 2 * a * x + 4
  f x₁ < f x₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l869_86924


namespace NUMINAMATH_CALUDE_papi_calot_plants_l869_86955

/-- The number of plants Papi Calot needs to buy -/
def total_plants (rows : ℕ) (plants_per_row : ℕ) (additional_plants : ℕ) : ℕ :=
  rows * plants_per_row + additional_plants

/-- Proof that Papi Calot needs to buy 141 plants -/
theorem papi_calot_plants : total_plants 7 18 15 = 141 := by
  sorry

end NUMINAMATH_CALUDE_papi_calot_plants_l869_86955


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l869_86980

-- Define the sets A and B
def A : Set ℕ := {4, 5, 6, 8}
def B : Set ℕ := {5, 7, 8, 9}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {5, 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l869_86980


namespace NUMINAMATH_CALUDE_T_5_value_l869_86967

/-- An arithmetic sequence with first term 1 and common difference 1 -/
def a (n : ℕ) : ℚ := n

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℚ := n * (n + 1) / 2

/-- Sum of the first n terms of the sequence {1/S_n} -/
def T (n : ℕ) : ℚ := 2 * n / (n + 1)

/-- Theorem: T_5 = 5/3 -/
theorem T_5_value : T 5 = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_T_5_value_l869_86967


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_sum_l869_86940

theorem rectangle_area_perimeter_sum (a b : ℕ+) : 
  let A := (a : ℕ) * b
  let P := 2 * ((a : ℕ) + b)
  A + P ≠ 102 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_sum_l869_86940


namespace NUMINAMATH_CALUDE_vegetable_load_weight_l869_86996

/-- Calculates the total weight of a load of vegetables given the weight of a crate, 
    the weight of a carton, and the number of crates and cartons. -/
def totalWeight (crateWeight cartonWeight : ℕ) (numCrates numCartons : ℕ) : ℕ :=
  crateWeight * numCrates + cartonWeight * numCartons

/-- Proves that the total weight of a specific load of vegetables is 96 kilograms. -/
theorem vegetable_load_weight :
  totalWeight 4 3 12 16 = 96 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_load_weight_l869_86996


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l869_86961

theorem meaningful_fraction_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l869_86961


namespace NUMINAMATH_CALUDE_white_animals_more_than_cats_l869_86929

theorem white_animals_more_than_cats (C W : ℕ) (h1 : C > 0) (h2 : W > 0) : W > C :=
  by
  -- Define the number of white cats (WC)
  have h3 : C / 3 = W / 6 :=
    -- Every third cat is white and every sixth white animal is a cat
    sorry
  
  -- Prove that W = 2C
  have h4 : W = 2 * C :=
    sorry

  -- Conclude that W > C
  sorry


end NUMINAMATH_CALUDE_white_animals_more_than_cats_l869_86929


namespace NUMINAMATH_CALUDE_most_sweets_l869_86943

/-- Given the distribution of sweets among three people, prove who received the most. -/
theorem most_sweets (total : ℕ) (minsu jaeyoung heesu : ℕ) 
  (h_total : total = 30)
  (h_minsu : minsu = 12)
  (h_jaeyoung : jaeyoung = 3)
  (h_heesu : heesu = 15)
  (h_sum : minsu + jaeyoung + heesu = total) :
  heesu > minsu ∧ heesu > jaeyoung := by
  sorry

end NUMINAMATH_CALUDE_most_sweets_l869_86943


namespace NUMINAMATH_CALUDE_shark_sightings_difference_l869_86930

theorem shark_sightings_difference (daytona_sightings cape_may_sightings : ℕ) 
  (h1 : daytona_sightings = 26)
  (h2 : cape_may_sightings = 7)
  (h3 : daytona_sightings > 3 * cape_may_sightings) :
  daytona_sightings - 3 * cape_may_sightings = 5 := by
sorry

end NUMINAMATH_CALUDE_shark_sightings_difference_l869_86930


namespace NUMINAMATH_CALUDE_billys_songbook_l869_86910

/-- The number of songs Billy can play -/
def songs_can_play : ℕ := 24

/-- The number of songs Billy still needs to learn -/
def songs_to_learn : ℕ := 28

/-- The total number of songs in Billy's music book -/
def total_songs : ℕ := songs_can_play + songs_to_learn

theorem billys_songbook :
  total_songs = 52 := by sorry

end NUMINAMATH_CALUDE_billys_songbook_l869_86910


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l869_86956

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

-- Define the solution set
def S : Set ℝ := {2} ∪ {x | x > 6}

-- Theorem statement
theorem solution_set_of_inequality :
  {x : ℝ | f x ≥ x^2 - 8*x + 15} = S :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l869_86956


namespace NUMINAMATH_CALUDE_pet_beds_per_pet_l869_86914

theorem pet_beds_per_pet (total_beds : ℕ) (num_pets : ℕ) (beds_per_pet : ℕ) : 
  total_beds = 20 → num_pets = 10 → beds_per_pet = total_beds / num_pets → beds_per_pet = 2 := by
  sorry

end NUMINAMATH_CALUDE_pet_beds_per_pet_l869_86914


namespace NUMINAMATH_CALUDE_manuscript_cost_l869_86984

/-- Represents the cost structure for typing and revising pages -/
structure TypingRates :=
  (initial : ℕ)
  (first_revision : ℕ)
  (second_revision : ℕ)
  (subsequent_revisions : ℕ)

/-- Represents the manuscript details -/
structure Manuscript :=
  (total_pages : ℕ)
  (revised_once : ℕ)
  (revised_twice : ℕ)
  (revised_thrice : ℕ)

/-- Calculates the total cost of typing and revising a manuscript -/
def total_cost (rates : TypingRates) (manuscript : Manuscript) : ℕ :=
  rates.initial * manuscript.total_pages +
  rates.first_revision * manuscript.revised_once +
  rates.second_revision * manuscript.revised_twice +
  rates.subsequent_revisions * manuscript.revised_thrice

/-- The typing service rates -/
def service_rates : TypingRates :=
  { initial := 10
  , first_revision := 5
  , second_revision := 7
  , subsequent_revisions := 10 }

/-- The manuscript details -/
def manuscript : Manuscript :=
  { total_pages := 150
  , revised_once := 20
  , revised_twice := 30
  , revised_thrice := 10 }

/-- Theorem stating that the total cost for the given manuscript is 1910 -/
theorem manuscript_cost : total_cost service_rates manuscript = 1910 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_l869_86984


namespace NUMINAMATH_CALUDE_special_integers_count_l869_86909

/-- The sum of all positive divisors of n including twice the greatest prime divisor of n -/
def g (n : ℕ) : ℕ := sorry

/-- The count of integers j such that 1 ≤ j ≤ 5000 and g(j) = j + 2√j + 1 -/
def count_special_integers : ℕ := sorry

theorem special_integers_count :
  count_special_integers = 19 := by sorry

end NUMINAMATH_CALUDE_special_integers_count_l869_86909


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_two_sqrt_two_l869_86969

theorem sqrt_sum_equals_two_sqrt_two :
  Real.sqrt (5 - 2 * Real.sqrt 6) + Real.sqrt (5 + 2 * Real.sqrt 6) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_two_sqrt_two_l869_86969


namespace NUMINAMATH_CALUDE_tourist_guide_groupings_l869_86941

/-- The number of ways to distribute n distinguishable objects into 2 non-empty groups -/
def distributionCount (n : ℕ) : ℕ :=
  2^n - 2

/-- The number of tourists -/
def numTourists : ℕ := 6

/-- The number of guides -/
def numGuides : ℕ := 2

theorem tourist_guide_groupings :
  distributionCount numTourists = 62 :=
sorry

end NUMINAMATH_CALUDE_tourist_guide_groupings_l869_86941


namespace NUMINAMATH_CALUDE_courtyard_paving_l869_86946

-- Define the courtyard dimensions in centimeters
def courtyard_length : ℕ := 2500
def courtyard_width : ℕ := 1800

-- Define the brick dimensions in centimeters
def brick_length : ℕ := 20
def brick_width : ℕ := 10

-- Define the function to calculate the number of bricks required
def bricks_required (cl cw bl bw : ℕ) : ℕ :=
  (cl * cw) / (bl * bw)

-- Theorem statement
theorem courtyard_paving :
  bricks_required courtyard_length courtyard_width brick_length brick_width = 22500 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_paving_l869_86946


namespace NUMINAMATH_CALUDE_increased_chickens_sum_l869_86918

/-- The number of increased chickens since the beginning -/
def increased_chickens (original : ℕ) (first_day : ℕ) (second_day : ℕ) : ℕ :=
  first_day + second_day

/-- Theorem stating that the number of increased chickens is the sum of chickens brought on the first and second day -/
theorem increased_chickens_sum (original : ℕ) (first_day : ℕ) (second_day : ℕ) :
  increased_chickens original first_day second_day = first_day + second_day :=
by sorry

#eval increased_chickens 45 18 12

end NUMINAMATH_CALUDE_increased_chickens_sum_l869_86918


namespace NUMINAMATH_CALUDE_tan_product_pi_ninths_l869_86977

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_pi_ninths_l869_86977


namespace NUMINAMATH_CALUDE_range_of_m_l869_86990

/-- Proposition p: the solution set for |x| + |x + 1| > m is ℝ -/
def p (m : ℝ) : Prop := ∀ x, |x| + |x + 1| > m

/-- Proposition q: the function f(x) = x^2 - 2mx + 1 is increasing on (2, +∞) -/
def q (m : ℝ) : Prop := ∀ x > 2, Monotone (fun x => x^2 - 2*m*x + 1)

/-- The range of real numbers m that satisfies the given conditions is [1, 2] -/
theorem range_of_m :
  ∀ m : ℝ, (∀ m', (p m' ∨ q m') ∧ ¬(p m' ∧ q m') → m' = m) → m ∈ Set.Icc 1 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l869_86990


namespace NUMINAMATH_CALUDE_leap_year_inequality_l869_86970

/-- Represents the dataset for a leap year as described in the problem -/
def leapYearData : List ℕ := sorry

/-- Calculates the median of modes for the leap year dataset -/
def medianOfModes (data : List ℕ) : ℚ := sorry

/-- Calculates the mean for the leap year dataset -/
def mean (data : List ℕ) : ℚ := sorry

/-- Calculates the median for the leap year dataset -/
def median (data : List ℕ) : ℚ := sorry

theorem leap_year_inequality :
  let d := medianOfModes leapYearData
  let μ := mean leapYearData
  let M := median leapYearData
  d < μ ∧ μ < M := by sorry

end NUMINAMATH_CALUDE_leap_year_inequality_l869_86970


namespace NUMINAMATH_CALUDE_expression_simplification_l869_86957

theorem expression_simplification (a b : ℝ) 
  (ha : a = 3 + Real.sqrt 5) 
  (hb : b = 3 - Real.sqrt 5) : 
  ((a^2 - 2*a*b + b^2) / (a^2 - b^2)) * (a*b / (a - b)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l869_86957


namespace NUMINAMATH_CALUDE_first_discount_percentage_l869_86983

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) 
  (second_discount : ℝ) (first_discount : ℝ) : 
  original_price = 200 →
  final_price = 152 →
  second_discount = 0.05 →
  final_price = original_price * (1 - first_discount) * (1 - second_discount) →
  first_discount = 0.20 := by
  sorry

#check first_discount_percentage

end NUMINAMATH_CALUDE_first_discount_percentage_l869_86983


namespace NUMINAMATH_CALUDE_proportional_function_k_value_l869_86960

/-- A proportional function passing through a specific point -/
def proportional_function (k : ℝ) (x : ℝ) : ℝ := k * x

theorem proportional_function_k_value :
  ∀ k : ℝ,
  k ≠ 0 →
  proportional_function k 3 = -6 →
  k = -2 := by
sorry

end NUMINAMATH_CALUDE_proportional_function_k_value_l869_86960


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l869_86911

/-- Given a quadratic function f(x) = x^2 + bx + c, 
    if its solution set for f(x) > 0 is (-1, 2), 
    then b + c = -3 -/
theorem quadratic_inequality_solution (b c : ℝ) : 
  (∀ x, x^2 + b*x + c > 0 ↔ -1 < x ∧ x < 2) → 
  b + c = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l869_86911


namespace NUMINAMATH_CALUDE_base_8_to_10_reverse_digits_l869_86953

theorem base_8_to_10_reverse_digits : ∃ (d e f : ℕ), 
  (0 ≤ d ∧ d ≤ 7) ∧ 
  (0 ≤ e ∧ e ≤ 7) ∧ 
  (0 ≤ f ∧ f ≤ 7) ∧ 
  e = 3 ∧
  (64 * d + 8 * e + f = 100 * f + 10 * e + d) := by
  sorry

end NUMINAMATH_CALUDE_base_8_to_10_reverse_digits_l869_86953


namespace NUMINAMATH_CALUDE_sin_480_plus_tan_300_l869_86917

/-- The sum of sine of 480 degrees and tangent of 300 degrees equals negative square root of 3 divided by 2. -/
theorem sin_480_plus_tan_300 : Real.sin (480 * π / 180) + Real.tan (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_480_plus_tan_300_l869_86917


namespace NUMINAMATH_CALUDE_tank_fill_time_l869_86979

-- Define the fill/drain rates for each pipe
def rate_A : ℚ := 1 / 10
def rate_B : ℚ := 1 / 20
def rate_C : ℚ := -(1 / 30)  -- Negative because it's draining

-- Define the combined rate
def combined_rate : ℚ := rate_A + rate_B + rate_C

-- Theorem to prove
theorem tank_fill_time :
  (1 : ℚ) / combined_rate = 60 / 7 := by sorry

end NUMINAMATH_CALUDE_tank_fill_time_l869_86979


namespace NUMINAMATH_CALUDE_order_of_values_l869_86921

/-- A function f is even if f(-x) = f(x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function f is monotonically increasing on [0, +∞) if
    for all a, b ≥ 0, a < b implies f(a) < f(b) -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ a b, 0 ≤ a → 0 ≤ b → a < b → f a < f b

theorem order_of_values (f : ℝ → ℝ) 
    (h_even : EvenFunction f) 
    (h_mono : MonoIncreasing f) :
    f (-π) > f 3 ∧ f 3 > f (-2) := by
  sorry

end NUMINAMATH_CALUDE_order_of_values_l869_86921


namespace NUMINAMATH_CALUDE_gcd_1230_990_l869_86925

theorem gcd_1230_990 : Nat.gcd 1230 990 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1230_990_l869_86925


namespace NUMINAMATH_CALUDE_complex_magnitude_equals_seven_l869_86963

theorem complex_magnitude_equals_seven (t : ℝ) (h1 : t > 0) :
  Complex.abs (3 + t * Complex.I) = 7 → t = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equals_seven_l869_86963


namespace NUMINAMATH_CALUDE_average_of_4_8_N_l869_86993

theorem average_of_4_8_N (N : ℝ) (h : 7 < N ∧ N < 15) : 
  let avg := (4 + 8 + N) / 3
  avg = 7 ∨ avg = 9 := by
sorry

end NUMINAMATH_CALUDE_average_of_4_8_N_l869_86993


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l869_86949

theorem product_from_lcm_gcd (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 60) 
  (h_gcd : Nat.gcd a b = 5) : 
  a * b = 300 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l869_86949


namespace NUMINAMATH_CALUDE_max_sum_with_square_diff_l869_86973

theorem max_sum_with_square_diff (a b : ℤ) (h : a^2 - b^2 = 144) :
  ∃ (d : ℤ), d = a + b ∧ d ≤ 72 ∧ ∃ (a' b' : ℤ), a'^2 - b'^2 = 144 ∧ a' + b' = 72 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_with_square_diff_l869_86973


namespace NUMINAMATH_CALUDE_miranda_pillow_stuffing_l869_86923

/-- 
Given:
- Two pounds of feathers are needed for each pillow
- A pound of goose feathers is approximately 300 feathers
- A pound of duck feathers is approximately 500 feathers
- Miranda's goose has approximately 3600 feathers
- Miranda's duck has approximately 4000 feathers

Prove that Miranda can stuff 10 pillows.
-/
theorem miranda_pillow_stuffing (
  feathers_per_pillow : ℕ)
  (goose_feathers_per_pound : ℕ)
  (duck_feathers_per_pound : ℕ)
  (goose_total_feathers : ℕ)
  (duck_total_feathers : ℕ)
  (h1 : feathers_per_pillow = 2)
  (h2 : goose_feathers_per_pound = 300)
  (h3 : duck_feathers_per_pound = 500)
  (h4 : goose_total_feathers = 3600)
  (h5 : duck_total_feathers = 4000) :
  (goose_total_feathers / goose_feathers_per_pound + 
   duck_total_feathers / duck_feathers_per_pound) / 
  feathers_per_pillow = 10 := by
  sorry

end NUMINAMATH_CALUDE_miranda_pillow_stuffing_l869_86923


namespace NUMINAMATH_CALUDE_tylers_dogs_l869_86928

theorem tylers_dogs (puppies_per_dog : ℕ) (total_puppies : ℕ) (initial_dogs : ℕ) : 
  puppies_per_dog = 5 → 
  total_puppies = 75 → 
  initial_dogs * puppies_per_dog = total_puppies → 
  initial_dogs = 15 := by
sorry

end NUMINAMATH_CALUDE_tylers_dogs_l869_86928


namespace NUMINAMATH_CALUDE_arithmetic_sequence_term_count_l869_86900

/-- 
Given an arithmetic sequence with:
- First term a = 5
- Last term l = 203
- Common difference d = 3

Prove that the number of terms in this sequence is 67.
-/
theorem arithmetic_sequence_term_count : 
  ∀ (a l d n : ℕ), 
  a = 5 → l = 203 → d = 3 → 
  l = a + (n - 1) * d → 
  n = 67 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_term_count_l869_86900


namespace NUMINAMATH_CALUDE_chloe_winter_clothing_l869_86901

/-- The number of boxes Chloe has -/
def num_boxes : ℕ := 4

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 2

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 6

/-- The total number of winter clothing pieces Chloe has -/
def total_pieces : ℕ := num_boxes * (scarves_per_box + mittens_per_box)

theorem chloe_winter_clothing :
  total_pieces = 32 :=
by sorry

end NUMINAMATH_CALUDE_chloe_winter_clothing_l869_86901


namespace NUMINAMATH_CALUDE_remainder_of_sum_l869_86938

theorem remainder_of_sum (x y u v : ℕ) (h1 : y > 0) (h2 : x = u * y + v) (h3 : v < y) :
  (x + 3 * u * y) % y = v :=
by sorry

end NUMINAMATH_CALUDE_remainder_of_sum_l869_86938


namespace NUMINAMATH_CALUDE_unique_integer_solution_l869_86971

theorem unique_integer_solution : ∃! x : ℤ, x + 12 > 14 ∧ -3*x > -9 := by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l869_86971


namespace NUMINAMATH_CALUDE_no_function_satisfies_conditions_l869_86913

theorem no_function_satisfies_conditions : ¬∃ (f : ℝ → ℝ), 
  (∃ (M : ℝ), M > 0 ∧ ∀ (x : ℝ), -M ≤ f x ∧ f x ≤ M) ∧ 
  (f 1 = 1) ∧ 
  (∀ (x : ℝ), x ≠ 0 → f (x + 1/x^2) = f x + (f (1/x))^2) := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_conditions_l869_86913


namespace NUMINAMATH_CALUDE_is_quadratic_equation_l869_86933

theorem is_quadratic_equation (x : ℝ) : ∃ (a b c : ℝ), a ≠ 0 ∧ (x - 1)^2 = 2*(3 - x)^2 ↔ a*x^2 + b*x + c = 0 :=
sorry

end NUMINAMATH_CALUDE_is_quadratic_equation_l869_86933


namespace NUMINAMATH_CALUDE_carols_weight_l869_86997

theorem carols_weight (alice_weight carol_weight : ℝ) 
  (h1 : alice_weight + carol_weight = 240)
  (h2 : carol_weight - alice_weight = 2/3 * carol_weight) : 
  carol_weight = 180 := by
sorry

end NUMINAMATH_CALUDE_carols_weight_l869_86997


namespace NUMINAMATH_CALUDE_three_numbers_sum_l869_86912

theorem three_numbers_sum (a b c : ℝ) : 
  b = 2 * a ∧ c = 3 * a ∧ a^2 + b^2 + c^2 = 2744 → a + b + c = 84 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l869_86912


namespace NUMINAMATH_CALUDE_unique_solution_l869_86945

-- Define Θ as a natural number
variable (Θ : ℕ)

-- Define the condition that Θ is a single digit
def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

-- Define the two-digit number 4Θ
def four_Θ (Θ : ℕ) : ℕ := 40 + Θ

-- State the theorem
theorem unique_solution :
  (630 / Θ = four_Θ Θ + 2 * Θ) ∧ 
  (is_single_digit Θ) →
  Θ = 9 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l869_86945


namespace NUMINAMATH_CALUDE_associate_professor_pencils_l869_86906

theorem associate_professor_pencils 
  (total_people : ℕ) 
  (total_pencils : ℕ) 
  (total_charts : ℕ) 
  (associate_profs : ℕ) 
  (assistant_profs : ℕ) 
  (associate_prof_charts : ℕ) 
  (assistant_prof_pencils : ℕ) 
  (assistant_prof_charts : ℕ) :
  total_people = 6 →
  total_pencils = 7 →
  total_charts = 11 →
  associate_profs + assistant_profs = total_people →
  associate_prof_charts = 1 →
  assistant_prof_pencils = 1 →
  assistant_prof_charts = 2 →
  ∃ (associate_prof_pencils : ℕ),
    associate_prof_pencils * associate_profs + assistant_prof_pencils * assistant_profs = total_pencils ∧
    associate_prof_charts * associate_profs + assistant_prof_charts * assistant_profs = total_charts ∧
    associate_prof_pencils = 2 :=
by sorry

end NUMINAMATH_CALUDE_associate_professor_pencils_l869_86906


namespace NUMINAMATH_CALUDE_f_minimum_l869_86974

-- Define the function f
def f (x a b : ℝ) : ℝ := |x + a| + |x - b|

-- State the theorem
theorem f_minimum (a b : ℝ) (h : 1 / (2 * a) + 2 / b = 1) :
  ∀ x : ℝ, f x a b ≥ 9/2 :=
by sorry

end NUMINAMATH_CALUDE_f_minimum_l869_86974


namespace NUMINAMATH_CALUDE_mother_age_proof_l869_86964

def id_number : ℕ := 6101131197410232923
def current_year : ℕ := 2014

def extract_birth_year (id : ℕ) : ℕ :=
  (id / 10^13) % 10000

def calculate_age (birth_year current_year : ℕ) : ℕ :=
  current_year - birth_year

theorem mother_age_proof :
  calculate_age (extract_birth_year id_number) current_year = 40 := by
  sorry

end NUMINAMATH_CALUDE_mother_age_proof_l869_86964


namespace NUMINAMATH_CALUDE_consecutive_squares_sum_181_l869_86981

theorem consecutive_squares_sum_181 :
  ∃ k : ℕ, k^2 + (k+1)^2 = 181 ∧ k = 9 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_sum_181_l869_86981


namespace NUMINAMATH_CALUDE_N_subset_M_l869_86902

-- Define set M
def M : Set ℝ := {x | ∃ k : ℤ, x = k / 2 + 1 / 3}

-- Define set N
def N : Set ℝ := {x | ∃ k : ℤ, x = k + 1 / 3}

-- Theorem statement
theorem N_subset_M : N ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_N_subset_M_l869_86902


namespace NUMINAMATH_CALUDE_club_members_after_four_years_l869_86958

/-- Represents the number of people in the club after k years -/
def club_members (k : ℕ) : ℕ :=
  if k = 0 then
    20
  else
    4 * club_members (k - 1) - 12

/-- The theorem stating the number of club members after 4 years -/
theorem club_members_after_four_years :
  club_members 4 = 4100 := by
  sorry

end NUMINAMATH_CALUDE_club_members_after_four_years_l869_86958


namespace NUMINAMATH_CALUDE_expand_and_simplify_l869_86966

theorem expand_and_simplify (x : ℝ) : 5 * (x + 6) * (x + 2) * (x + 7) = 5*x^3 + 75*x^2 + 340*x + 420 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l869_86966


namespace NUMINAMATH_CALUDE_trihedral_angle_existence_l869_86992

/-- A trihedral angle -/
structure TrihedralAngle where
  α : Real
  β : Real
  γ : Real

/-- Given three dihedral angles, there exists a trihedral angle with these angles -/
theorem trihedral_angle_existence (α β γ : Real) : 
  ∃ (T : TrihedralAngle), T.α = α ∧ T.β = β ∧ T.γ = γ := by
  sorry

end NUMINAMATH_CALUDE_trihedral_angle_existence_l869_86992


namespace NUMINAMATH_CALUDE_sum_of_coefficients_P_l869_86950

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 5 * (2 * x^9 - 3 * x^6 + 4) - 4 * (x^6 - 5 * x^3 + 6)

-- Theorem stating that the sum of coefficients of P(x) is 7
theorem sum_of_coefficients_P : (P 1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_P_l869_86950


namespace NUMINAMATH_CALUDE_car_speed_comparison_l869_86976

/-- Proves that given a car traveling at 80 km/hour takes 5 seconds longer to travel 1 km than at another speed, the other speed is 90 km/hour. -/
theorem car_speed_comparison (v : ℝ) : 
  v > 0 →  -- Ensure speed is positive
  (1 / (80 / 3600)) - (1 / (v / 3600)) = 5 → 
  v = 90 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_comparison_l869_86976


namespace NUMINAMATH_CALUDE_square_of_two_digit_is_68_l869_86935

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_digit (n : ℕ) : ℕ := n / 1000

def last_digit (n : ℕ) : ℕ := n % 10

def middle_digits_sum (n : ℕ) : ℕ := (n / 100 % 10) + (n / 10 % 10)

theorem square_of_two_digit_is_68 (n : ℕ) (h1 : is_four_digit n) 
  (h2 : first_digit n = last_digit n)
  (h3 : first_digit n + last_digit n = middle_digits_sum n)
  (h4 : ∃ m : ℕ, 10 ≤ m ∧ m ≤ 99 ∧ m * m = n) :
  ∃ m : ℕ, m = 68 ∧ m * m = n := by
sorry

end NUMINAMATH_CALUDE_square_of_two_digit_is_68_l869_86935


namespace NUMINAMATH_CALUDE_abs_neg_ten_eq_ten_l869_86959

theorem abs_neg_ten_eq_ten : |(-10 : ℤ)| = 10 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_ten_eq_ten_l869_86959


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l869_86968

/-- Given an algebraic expression mx^2 - 2x + n that equals 2 when x = 2,
    prove that it equals 10 when x = -2 -/
theorem algebraic_expression_value (m n : ℝ) 
  (h : m * 2^2 - 2 * 2 + n = 2) : 
  m * (-2)^2 - 2 * (-2) + n = 10 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l869_86968


namespace NUMINAMATH_CALUDE_population_change_proof_l869_86944

-- Define the initial population
def initial_population : ℕ := 4518

-- Define the sequence of population changes
def population_after_bombardment (p : ℕ) : ℕ := (p * 95) / 100
def population_after_migration (p : ℕ) : ℕ := (p * 80) / 100
def population_after_return (p : ℕ) : ℕ := (p * 115) / 100
def population_after_flood (p : ℕ) : ℕ := (p * 90) / 100

-- Define the final population
def final_population : ℕ := 3553

-- Theorem statement
theorem population_change_proof :
  population_after_flood
    (population_after_return
      (population_after_migration
        (population_after_bombardment initial_population)))
  = final_population := by sorry

end NUMINAMATH_CALUDE_population_change_proof_l869_86944


namespace NUMINAMATH_CALUDE_equation_solution_l869_86920

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (x - 1) / (x - 2) + 1 / (2 - x) = 3 ↔ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l869_86920


namespace NUMINAMATH_CALUDE_arctan_sum_three_seven_l869_86922

theorem arctan_sum_three_seven : Real.arctan (3/7) + Real.arctan (7/3) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_seven_l869_86922


namespace NUMINAMATH_CALUDE_complex_equation_solution_l869_86982

theorem complex_equation_solution :
  ∀ x : ℝ, (1 - 2*I) * (x + I) = 4 - 3*I → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l869_86982


namespace NUMINAMATH_CALUDE_monitor_student_ratio_l869_86932

/-- The ratio of monitors to students in a lunchroom --/
theorem monitor_student_ratio :
  ∀ (S : ℕ) (G B : ℝ),
    G = 0.4 * S →
    B = 0.6 * S →
    2 * G + B = 168 →
    (8 : ℝ) / S = 1 / 15 :=
by sorry

end NUMINAMATH_CALUDE_monitor_student_ratio_l869_86932


namespace NUMINAMATH_CALUDE_large_rectangle_ratio_l869_86947

/-- Represents the side length of a square in the arrangement -/
def square_side : ℝ := sorry

/-- Represents the length of the large rectangle -/
def large_rectangle_length : ℝ := 3 * square_side

/-- Represents the width of the large rectangle -/
def large_rectangle_width : ℝ := 3 * square_side

/-- Represents the length of the smaller rectangle -/
def small_rectangle_length : ℝ := 3 * square_side

/-- Represents the width of the smaller rectangle -/
def small_rectangle_width : ℝ := square_side

theorem large_rectangle_ratio :
  large_rectangle_length / large_rectangle_width = 3 := by sorry

end NUMINAMATH_CALUDE_large_rectangle_ratio_l869_86947


namespace NUMINAMATH_CALUDE_proposition_properties_l869_86907

theorem proposition_properties :
  -- 1. Negation of existential quantifier
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) ∧
  -- 2. Sufficient but not necessary condition
  (∃ x : ℝ, x = 1 → x^2 - 4*x + 3 = 0) ∧
  (∃ x : ℝ, x^2 - 4*x + 3 = 0 ∧ x ≠ 1) ∧
  -- 3. Converse of implication
  ((∀ x : ℝ, x^2 - 4*x + 3 = 0 → x = 1) →
   (∀ x : ℝ, x ≠ 1 → x^2 - 4*x + 3 ≠ 0)) ∧
  -- 4. Falsity of conjunction doesn't imply falsity of both propositions
  ∃ (p q : Prop), ¬(p ∧ q) ∧ (p ∨ q) :=
by sorry

end NUMINAMATH_CALUDE_proposition_properties_l869_86907


namespace NUMINAMATH_CALUDE_mandy_nutmeg_amount_l869_86952

/-- The amount of cinnamon Mandy used in tablespoons -/
def cinnamon : ℚ := 0.6666666666666666

/-- The difference between cinnamon and nutmeg in tablespoons -/
def difference : ℚ := 0.16666666666666666

/-- The amount of nutmeg Mandy used in tablespoons -/
def nutmeg : ℚ := cinnamon - difference

theorem mandy_nutmeg_amount : nutmeg = 0.5 := by sorry

end NUMINAMATH_CALUDE_mandy_nutmeg_amount_l869_86952


namespace NUMINAMATH_CALUDE_initial_men_count_l869_86951

/-- The number of men initially doing the work -/
def initial_men : ℕ := 36

/-- The time taken by the initial group of men to complete the work -/
def initial_time : ℕ := 25

/-- The number of men in the second group -/
def second_group : ℕ := 15

/-- The time taken by the second group to complete the work -/
def second_time : ℕ := 60

/-- Theorem stating that the initial number of men is 36 -/
theorem initial_men_count : initial_men = 36 := by
  sorry

#check initial_men_count

end NUMINAMATH_CALUDE_initial_men_count_l869_86951


namespace NUMINAMATH_CALUDE_tammys_climbing_speed_l869_86931

/-- Tammy's mountain climbing problem -/
theorem tammys_climbing_speed 
  (total_time : ℝ) 
  (total_distance : ℝ) 
  (speed_difference : ℝ) 
  (time_difference : ℝ) 
  (h1 : total_time = 14) 
  (h2 : total_distance = 52) 
  (h3 : speed_difference = 0.5) 
  (h4 : time_difference = 2) :
  ∃ (v : ℝ), 
    v * (total_time / 2 + time_difference) + 
    (v + speed_difference) * (total_time / 2 - time_difference) = total_distance ∧
    v + speed_difference = 4 := by
  sorry


end NUMINAMATH_CALUDE_tammys_climbing_speed_l869_86931


namespace NUMINAMATH_CALUDE_range_of_a_l869_86994

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → |x + 1/x| > |a - 5| + 1) → 
  a ∈ Set.Ioo 4 6 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l869_86994


namespace NUMINAMATH_CALUDE_emilias_blueberries_l869_86954

/-- The number of cartons of berries Emilia needs in total -/
def total_needed : ℕ := 42

/-- The number of cartons of strawberries Emilia has -/
def strawberries : ℕ := 2

/-- The number of cartons of berries Emilia buys at the supermarket -/
def bought : ℕ := 33

/-- The number of cartons of blueberries in Emilia's cupboard -/
def blueberries : ℕ := total_needed - (strawberries + bought)

theorem emilias_blueberries : blueberries = 7 := by
  sorry

end NUMINAMATH_CALUDE_emilias_blueberries_l869_86954


namespace NUMINAMATH_CALUDE_parallel_lines_angle_theorem_l869_86978

/-- Given a configuration of two parallel lines intersected by two other lines,
    if one angle is 70°, its adjacent angle is 40°, and the corresponding angle
    on the other parallel line is 110°, then the remaining angle is 40°. -/
theorem parallel_lines_angle_theorem (a b c d : Real) :
  a = 70 →
  b = 40 →
  c = 110 →
  a + b + c + d = 360 →
  d = 40 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_angle_theorem_l869_86978


namespace NUMINAMATH_CALUDE_same_terminal_side_same_quadrant_angle_2010_in_third_quadrant_l869_86927

-- Define the quadrants
inductive Quadrant
| first
| second
| third
| fourth

-- Define a function to determine the quadrant of an angle
def angle_quadrant (angle : ℝ) : Quadrant := sorry

-- Define the principle that angles with the same terminal side are in the same quadrant
theorem same_terminal_side_same_quadrant (angle1 angle2 : ℝ) :
  angle1 % 360 = angle2 % 360 → angle_quadrant angle1 = angle_quadrant angle2 := sorry

-- State the theorem
theorem angle_2010_in_third_quadrant :
  let angle_2010 : ℝ := 2010
  let angle_210 : ℝ := 210
  angle_2010 = 5 * 360 + angle_210 →
  angle_quadrant angle_210 = Quadrant.third →
  angle_quadrant angle_2010 = Quadrant.third := by
    sorry

end NUMINAMATH_CALUDE_same_terminal_side_same_quadrant_angle_2010_in_third_quadrant_l869_86927


namespace NUMINAMATH_CALUDE_sum_of_factors_of_30_l869_86903

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))

theorem sum_of_factors_of_30 : (factors 30).sum id = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_of_30_l869_86903
