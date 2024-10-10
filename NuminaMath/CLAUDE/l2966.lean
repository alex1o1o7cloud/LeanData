import Mathlib

namespace initial_goldfish_count_l2966_296617

/-- The number of goldfish Paige initially raised in the pond -/
def initial_goldfish : ℕ := 15

/-- The number of goldfish remaining in the pond -/
def remaining_goldfish : ℕ := 4

/-- The number of goldfish that disappeared -/
def disappeared_goldfish : ℕ := 11

/-- Theorem: The initial number of goldfish is equal to the sum of the remaining and disappeared goldfish -/
theorem initial_goldfish_count : initial_goldfish = remaining_goldfish + disappeared_goldfish := by
  sorry

end initial_goldfish_count_l2966_296617


namespace carla_candy_bags_l2966_296612

/-- Calculates the number of bags bought given the original price, discount percentage, and total amount spent -/
def bags_bought (original_price : ℚ) (discount_percentage : ℚ) (total_spent : ℚ) : ℚ :=
  total_spent / (original_price * (1 - discount_percentage))

/-- Proves that Carla bought 2 bags of candy -/
theorem carla_candy_bags : 
  let original_price : ℚ := 6
  let discount_percentage : ℚ := 3/4
  let total_spent : ℚ := 3
  bags_bought original_price discount_percentage total_spent = 2 := by
sorry

#eval bags_bought 6 (3/4) 3

end carla_candy_bags_l2966_296612


namespace christina_age_fraction_l2966_296672

/-- Christina's current age -/
def christina_age : ℕ := sorry

/-- Oscar's current age -/
def oscar_age : ℕ := 6

/-- The fraction of Christina's age in 5 years to 80 years -/
def christina_fraction : ℚ := (christina_age + 5) / 80

theorem christina_age_fraction :
  (oscar_age + 15 = 3 * christina_age / 5) →
  christina_fraction = 1 / 2 := by sorry

end christina_age_fraction_l2966_296672


namespace common_number_in_overlapping_sets_l2966_296662

theorem common_number_in_overlapping_sets (numbers : List ℝ) : 
  numbers.length = 9 →
  (numbers.take 5).sum / 5 = 7 →
  (numbers.drop 4).sum / 5 = 10 →
  numbers.sum / 9 = 74 / 9 →
  ∃ x ∈ numbers.take 5 ∩ numbers.drop 4, x = 11 := by
  sorry

end common_number_in_overlapping_sets_l2966_296662


namespace line_x_axis_intersection_l2966_296660

theorem line_x_axis_intersection (x y : ℝ) :
  (5 * y - 7 * x = 14) ∧ (y = 0) → (x = -2 ∧ y = 0) :=
by sorry

end line_x_axis_intersection_l2966_296660


namespace exactly_two_correct_propositions_l2966_296649

-- Define the concept of related curves
def related_curves (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (l : ℝ → ℝ → Prop), ∃ (x1 y1 x2 y2 : ℝ),
    C1 x1 y1 ∧ C2 x2 y2 ∧
    (∀ x y, l x y ↔ (y - y1) = (x - x1) * ((y2 - y1) / (x2 - x1))) ∧
    (∀ x y, l x y → (C1 x y ∨ C2 x y))

-- Define the curves
def C1_1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def C2_1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 4 = 0
def C1_2 (x y : ℝ) : Prop := 4*y^2 - x^2 = 1
def C2_2 (x y : ℝ) : Prop := x^2 - 4*y^2 = 1
def C1_3 (x y : ℝ) : Prop := y = Real.log x
def C2_3 (x y : ℝ) : Prop := y = x^2 - x

-- Define the propositions
def prop1 : Prop := ∃! (l1 l2 : ℝ → ℝ → Prop), 
  related_curves C1_1 C2_1 ∧ (∀ x y, l1 x y → (C1_1 x y ∨ C2_1 x y)) ∧
  (∀ x y, l2 x y → (C1_1 x y ∨ C2_1 x y)) ∧ l1 ≠ l2

def prop2 : Prop := related_curves C1_2 C2_2

def prop3 : Prop := related_curves C1_3 C2_3

-- The theorem to prove
theorem exactly_two_correct_propositions : 
  (prop1 ∧ ¬prop2 ∧ prop3) ∨ (prop1 ∧ prop2 ∧ ¬prop3) ∨ (¬prop1 ∧ prop2 ∧ prop3) :=
sorry

end exactly_two_correct_propositions_l2966_296649


namespace village_children_average_l2966_296688

/-- Given a village with families and children, calculates the average number of children in families with children -/
def average_children_in_families_with_children (total_families : ℕ) (total_average : ℚ) (childless_families : ℕ) : ℚ :=
  let total_children := total_families * total_average
  let families_with_children := total_families - childless_families
  total_children / families_with_children

/-- Proves that in a village with 12 families, an average of 3 children per family, and 3 childless families, 
    the average number of children in families with children is 4.0 -/
theorem village_children_average : average_children_in_families_with_children 12 3 3 = 4 := by
  sorry

end village_children_average_l2966_296688


namespace largest_n_polynomials_l2966_296640

/-- A type representing real polynomials -/
def RealPolynomial := ℝ → ℝ

/-- Predicate to check if a real polynomial has no real roots -/
def HasNoRealRoots (p : RealPolynomial) : Prop :=
  ∀ x : ℝ, p x ≠ 0

/-- Predicate to check if a real polynomial has at least one real root -/
def HasRealRoot (p : RealPolynomial) : Prop :=
  ∃ x : ℝ, p x = 0

/-- The main theorem statement -/
theorem largest_n_polynomials :
  (∃ (n : ℕ) (P : Fin n → RealPolynomial),
    (∀ (i j : Fin n) (hij : i ≠ j), HasNoRealRoots (fun x ↦ P i x + P j x)) ∧
    (∀ (i j k : Fin n) (hijk : i ≠ j ∧ j ≠ k ∧ i ≠ k), HasRealRoot (fun x ↦ P i x + P j x + P k x))) →
  (∃ (P : Fin 3 → RealPolynomial),
    (∀ (i j : Fin 3) (hij : i ≠ j), HasNoRealRoots (fun x ↦ P i x + P j x)) ∧
    (∀ (i j k : Fin 3) (hijk : i ≠ j ∧ j ≠ k ∧ i ≠ k), HasRealRoot (fun x ↦ P i x + P j x + P k x))) ∧
  (∀ (n : ℕ) (hn : n > 3),
    ¬∃ (P : Fin n → RealPolynomial),
      (∀ (i j : Fin n) (hij : i ≠ j), HasNoRealRoots (fun x ↦ P i x + P j x)) ∧
      (∀ (i j k : Fin n) (hijk : i ≠ j ∧ j ≠ k ∧ i ≠ k), HasRealRoot (fun x ↦ P i x + P j x + P k x))) :=
by
  sorry

end largest_n_polynomials_l2966_296640


namespace complex_arithmetic_evaluation_l2966_296631

theorem complex_arithmetic_evaluation :
  1234562 - ((12 * 3 * (2 + 7))^2 / 6) + 18 = 1217084 := by
  sorry

end complex_arithmetic_evaluation_l2966_296631


namespace favorite_number_is_27_l2966_296659

theorem favorite_number_is_27 : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  n^2 = (n / 10 + n % 10)^3 :=
by
  -- The proof would go here
  sorry

end favorite_number_is_27_l2966_296659


namespace average_of_numbers_l2966_296692

def numbers : List ℕ := [12, 13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers : (numbers.sum : ℚ) / numbers.length = 114391 := by
  sorry

end average_of_numbers_l2966_296692


namespace similar_triangle_perimeter_similar_triangle_perimeter_proof_l2966_296620

/-- Given an isosceles triangle with two sides of 15 inches and one side of 8 inches,
    a similar triangle with the longest side of 45 inches has a perimeter of 114 inches. -/
theorem similar_triangle_perimeter : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun original_long original_short similar_long perimeter =>
    original_long = 15 →
    original_short = 8 →
    similar_long = 45 →
    perimeter = similar_long + similar_long + (similar_long / original_long * original_short) →
    perimeter = 114

/-- Proof of the theorem -/
theorem similar_triangle_perimeter_proof :
  similar_triangle_perimeter 15 8 45 114 := by
  sorry

end similar_triangle_perimeter_similar_triangle_perimeter_proof_l2966_296620


namespace double_sum_equals_seven_fourths_l2966_296642

/-- The double sum of 1/(i^2j + 2ij + ij^2) over positive integers i and j from 1 to infinity equals 7/4 -/
theorem double_sum_equals_seven_fourths :
  (∑' i : ℕ+, ∑' j : ℕ+, (1 : ℝ) / ((i.val^2 * j.val) + (2 * i.val * j.val) + (i.val * j.val^2))) = 7/4 :=
by sorry

end double_sum_equals_seven_fourths_l2966_296642


namespace combined_yellow_ratio_approx_32_percent_l2966_296689

/-- Represents a bag of jelly beans -/
structure JellyBeanBag where
  total : ℕ
  yellow_ratio : ℚ

/-- Calculates the ratio of yellow jelly beans to all beans when multiple bags are combined -/
def combined_yellow_ratio (bags : List JellyBeanBag) : ℚ :=
  let total_beans := bags.map (λ bag => bag.total) |>.sum
  let total_yellow := bags.map (λ bag => (bag.total : ℚ) * bag.yellow_ratio) |>.sum
  total_yellow / total_beans

/-- The theorem to be proved -/
theorem combined_yellow_ratio_approx_32_percent : 
  let bags := [
    JellyBeanBag.mk 24 (2/5),
    JellyBeanBag.mk 32 (3/10),
    JellyBeanBag.mk 34 (1/4)
  ]
  abs (combined_yellow_ratio bags - 32222/100000) < 1/10000 := by
  sorry

end combined_yellow_ratio_approx_32_percent_l2966_296689


namespace negative_one_greater_than_negative_two_l2966_296600

theorem negative_one_greater_than_negative_two : -1 > -2 := by
  sorry

end negative_one_greater_than_negative_two_l2966_296600


namespace jakes_weight_l2966_296684

/-- Proves Jake's present weight given the conditions of the problem -/
theorem jakes_weight (jake kendra : ℕ) 
  (h1 : jake - 8 = 2 * kendra)
  (h2 : jake + kendra = 290) : 
  jake = 196 := by
  sorry

end jakes_weight_l2966_296684


namespace arithmetic_sequence_common_difference_l2966_296683

/-- 
Given an arithmetic sequence {a_n} with common difference d ≥ 0,
if a_2^2 is the arithmetic mean of a_1^2 and a_3^2 - 2, then d = 1.
-/
theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ) :
  d ≥ 0 →
  (∀ n, a (n + 1) = a n + d) →
  a 2^2 = (a 1^2 + (a 3^2 - 2)) / 2 →
  d = 1 := by
sorry

end arithmetic_sequence_common_difference_l2966_296683


namespace zeros_product_lower_bound_l2966_296607

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 + x

theorem zeros_product_lower_bound {a : ℝ} {x₁ x₂ : ℝ} 
  (h₁ : f a x₁ = 0)
  (h₂ : f a x₂ = 0)
  (h₃ : x₂ > 2 * x₁)
  (h₄ : x₁ > 0)
  (h₅ : x₂ > 0) :
  x₁ * x₂ > 8 / Real.exp 2 := by
  sorry

end zeros_product_lower_bound_l2966_296607


namespace john_feeds_twice_daily_l2966_296685

/-- Represents the scenario of John feeding his horses -/
structure HorseFeeding where
  num_horses : ℕ
  food_per_feeding : ℕ
  days : ℕ
  bags_bought : ℕ
  bag_weight : ℕ

/-- Calculates the number of feedings per horse per day -/
def feedings_per_horse_per_day (hf : HorseFeeding) : ℚ :=
  let total_food := hf.bags_bought * hf.bag_weight
  let food_per_day := total_food / hf.days
  let feedings_per_day := food_per_day / hf.food_per_feeding
  feedings_per_day / hf.num_horses

/-- Theorem stating that John feeds each horse twice a day -/
theorem john_feeds_twice_daily : 
  ∀ (hf : HorseFeeding), 
    hf.num_horses = 25 → 
    hf.food_per_feeding = 20 → 
    hf.days = 60 → 
    hf.bags_bought = 60 → 
    hf.bag_weight = 1000 → 
    feedings_per_horse_per_day hf = 2 := by
  sorry


end john_feeds_twice_daily_l2966_296685


namespace smallest_with_properties_l2966_296628

def is_smallest_with_properties (n : ℕ) : Prop :=
  (∃ (divisors : Finset ℕ), divisors.card = 144 ∧ (∀ d ∈ divisors, n % d = 0)) ∧
  (∃ (start : ℕ), ∀ i ∈ Finset.range 10, n % (start + i) = 0) ∧
  (∀ m < n, ¬(∃ (divisors : Finset ℕ), divisors.card = 144 ∧ (∀ d ∈ divisors, m % d = 0)) ∨
           ¬(∃ (start : ℕ), ∀ i ∈ Finset.range 10, m % (start + i) = 0))

theorem smallest_with_properties : is_smallest_with_properties 110880 := by
  sorry

end smallest_with_properties_l2966_296628


namespace constant_function_from_functional_equation_l2966_296616

/-- A continuous function f satisfying f(x) + f(x^2) = 2 for all real x is constant and equal to 1. -/
theorem constant_function_from_functional_equation (f : ℝ → ℝ) 
  (hf : Continuous f) 
  (h : ∀ x : ℝ, f x + f (x^2) = 2) : 
  ∀ x : ℝ, f x = 1 := by
  sorry

end constant_function_from_functional_equation_l2966_296616


namespace roger_coin_count_l2966_296636

/-- The total number of coins in Roger's collection -/
def total_coins (quarters : List Nat) (dimes : List Nat) (nickels : List Nat) (pennies : List Nat) : Nat :=
  quarters.sum + dimes.sum + nickels.sum + pennies.sum

/-- Theorem stating that Roger has 93 coins in total -/
theorem roger_coin_count :
  let quarters := [8, 6, 7, 5]
  let dimes := [7, 5, 9]
  let nickels := [4, 6]
  let pennies := [10, 3, 8, 2, 13]
  total_coins quarters dimes nickels pennies = 93 := by
  sorry

#eval total_coins [8, 6, 7, 5] [7, 5, 9] [4, 6] [10, 3, 8, 2, 13]

end roger_coin_count_l2966_296636


namespace factor_probability_l2966_296608

/-- The number of factors of m -/
def d (m : ℕ) : ℕ := (Nat.divisors m).card

/-- The probability of selecting a factor of m from 1 to m -/
def prob_factor (m : ℕ) : ℚ := (d m : ℚ) / m

theorem factor_probability (m : ℕ) (p : ℕ) (h : prob_factor m = p / 39) : p = 4 := by
  sorry

end factor_probability_l2966_296608


namespace continuous_stripe_probability_l2966_296677

/-- Represents a cube with stripes on its faces -/
structure StripedCube where
  faces : Fin 6 → Fin 3

/-- The probability of a continuous stripe encircling the cube -/
def probability_continuous_stripe : ℚ := 2 / 81

/-- The total number of possible stripe configurations -/
def total_configurations : ℕ := 3^6

/-- The number of configurations that result in a continuous stripe -/
def favorable_configurations : ℕ := 18

theorem continuous_stripe_probability :
  probability_continuous_stripe = favorable_configurations / total_configurations :=
sorry

end continuous_stripe_probability_l2966_296677


namespace standard_form_is_quadratic_expanded_form_is_quadratic_l2966_296622

/-- Definition of a quadratic equation -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation ax^2 + bx + c = 0 (where a ≠ 0) is quadratic -/
theorem standard_form_is_quadratic (a b c : ℝ) (ha : a ≠ 0) :
  is_quadratic (λ x => a * x^2 + b * x + c) :=
sorry

/-- The equation (x-2)^2 - 4 = 0 is quadratic -/
theorem expanded_form_is_quadratic :
  is_quadratic (λ x => (x - 2)^2 - 4) :=
sorry

end standard_form_is_quadratic_expanded_form_is_quadratic_l2966_296622


namespace positive_integer_problem_l2966_296648

theorem positive_integer_problem (n : ℕ+) (h : (12 : ℝ) * n.val = n.val ^ 2 + 36) : n = 6 := by
  sorry

end positive_integer_problem_l2966_296648


namespace principal_calculation_l2966_296633

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Problem statement -/
theorem principal_calculation (sum : ℝ) (rate : ℝ) (time : ℕ) 
  (h_sum : sum = 3969)
  (h_rate : rate = 0.05)
  (h_time : time = 2) :
  ∃ (principal : ℝ), 
    compound_interest principal rate time = sum ∧ 
    principal = 3600 := by
  sorry

end principal_calculation_l2966_296633


namespace simplified_expression_l2966_296613

theorem simplified_expression (a : ℤ) 
  (h1 : (a - 1) / 2 < 2)
  (h2 : (a + 1) / 2 ≥ (4 - a) / 3)
  (h3 : a ≠ 2)
  (h4 : a ≠ 4) :
  (16 - a^2) / (a^2 + 8*a + 16) / ((1 / 2) - (4 / (a + 4))) * (1 / (2*a - 4)) = -1 / (a - 2) :=
by sorry

end simplified_expression_l2966_296613


namespace milk_water_ratio_problem_l2966_296624

/-- Proves that the initial ratio of milk to water was 4:1 given the conditions of the mixture problem. -/
theorem milk_water_ratio_problem (initial_volume : ℝ) (added_water : ℝ) (final_ratio : ℝ) :
  initial_volume = 45 →
  added_water = 21 →
  final_ratio = 1.2 →
  ∃ (initial_milk initial_water : ℝ),
    initial_milk + initial_water = initial_volume ∧
    initial_milk / (initial_water + added_water) = final_ratio ∧
    initial_milk / initial_water = 4 / 1 :=
by
  sorry

#check milk_water_ratio_problem

end milk_water_ratio_problem_l2966_296624


namespace sum_of_distances_is_ten_l2966_296668

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the focus F
def focus : ℝ × ℝ := (3, 0)

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Define that P is on the line AB
def P_on_line_AB (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, P = (1 - t) • A + t • B

-- Define that P is the midpoint of AB
def P_is_midpoint (A B : ℝ × ℝ) : Prop :=
  P = (A + B) / 2

-- Define that A and B are on the parabola
def points_on_parabola (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2

-- State the theorem
theorem sum_of_distances_is_ten (A B : ℝ × ℝ) 
  (h1 : P_on_line_AB A B)
  (h2 : P_is_midpoint A B)
  (h3 : points_on_parabola A B) :
  dist A focus + dist B focus = 10 := by sorry

end sum_of_distances_is_ten_l2966_296668


namespace cricketer_average_score_l2966_296605

theorem cricketer_average_score (total_matches : ℕ) (overall_average : ℚ) 
  (last_matches : ℕ) (last_average : ℚ) (some_average : ℚ) :
  total_matches = 10 →
  overall_average = 389/10 →
  last_matches = 4 →
  last_average = 137/4 →
  some_average = 42 →
  ∃ (x : ℕ), x + last_matches = total_matches ∧ 
    (x : ℚ) * some_average + (last_matches : ℚ) * last_average = (total_matches : ℚ) * overall_average ∧
    x = 6 :=
by sorry

end cricketer_average_score_l2966_296605


namespace kids_difference_l2966_296674

theorem kids_difference (monday : ℕ) (tuesday : ℕ) 
  (h1 : monday = 6) (h2 : tuesday = 5) : monday - tuesday = 1 := by
  sorry

end kids_difference_l2966_296674


namespace chloe_carrots_initial_count_l2966_296681

/-- Proves that the initial number of carrots Chloe picked is 48, given the conditions of the problem. -/
theorem chloe_carrots_initial_count : ∃ x : ℕ, 
  (x - 45 + 42 = 45) ∧ 
  (x = 48) := by
  sorry

end chloe_carrots_initial_count_l2966_296681


namespace negation_of_existential_proposition_l2966_296665

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end negation_of_existential_proposition_l2966_296665


namespace tuesday_temperature_l2966_296629

def sunday_temp : ℝ := 40
def monday_temp : ℝ := 50
def wednesday_temp : ℝ := 36
def thursday_temp : ℝ := 82
def friday_temp : ℝ := 72
def saturday_temp : ℝ := 26
def average_temp : ℝ := 53
def days_in_week : ℕ := 7

theorem tuesday_temperature :
  ∃ tuesday_temp : ℝ,
    (sunday_temp + monday_temp + tuesday_temp + wednesday_temp +
     thursday_temp + friday_temp + saturday_temp) / days_in_week = average_temp ∧
    tuesday_temp = 65 := by
  sorry

end tuesday_temperature_l2966_296629


namespace hockey_league_face_count_l2966_296686

/-- The number of times each team faces other teams in a hockey league -/
def faceCount (n : ℕ) (total_games : ℕ) : ℕ :=
  total_games / (n * (n - 1) / 2)

/-- Theorem: In a hockey league with 19 teams and 1710 total games, each team faces others 5 times -/
theorem hockey_league_face_count :
  faceCount 19 1710 = 5 := by
  sorry

end hockey_league_face_count_l2966_296686


namespace shelter_cat_count_l2966_296664

/-- Represents the state of the animal shelter --/
structure AnimalShelter where
  initialCats : ℕ
  newCats : ℕ
  adoptedCats : ℕ
  bornKittens : ℕ
  claimedPets : ℕ

/-- Calculates the final number of cats in the shelter --/
def finalCatCount (shelter : AnimalShelter) : ℕ :=
  shelter.initialCats + shelter.newCats - shelter.adoptedCats + shelter.bornKittens - shelter.claimedPets

/-- Theorem stating the final number of cats in the shelter --/
theorem shelter_cat_count : ∃ (shelter : AnimalShelter),
  shelter.initialCats = 60 ∧
  shelter.newCats = 30 ∧
  shelter.adoptedCats = 20 ∧
  shelter.bornKittens = 15 ∧
  shelter.claimedPets = 2 ∧
  finalCatCount shelter = 83 := by
  sorry

#check shelter_cat_count

end shelter_cat_count_l2966_296664


namespace x_plus_y_equals_three_l2966_296614

theorem x_plus_y_equals_three (x y : ℝ) 
  (h1 : |x| + x + 5*y = 2) 
  (h2 : |y| - y + x = 7) : 
  x + y = 3 := by
sorry

end x_plus_y_equals_three_l2966_296614


namespace part_one_part_two_l2966_296693

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + |x - a^2|

-- Define the solution set for part (1)
def solution_set : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}

-- Theorem for part (1)
theorem part_one : 
  ∀ x, f x 1 ≥ 4 ↔ x ∈ solution_set :=
sorry

-- Theorem for part (2)
theorem part_two :
  (∀ x, ∃ a ∈ Set.Ioo (-1) 3, m < f x a) → m < 12 :=
sorry

end part_one_part_two_l2966_296693


namespace isosceles_right_triangle_area_l2966_296673

/-- The area of a right-angled isosceles triangle with hypotenuse length 1 is 1/4 -/
theorem isosceles_right_triangle_area (A B C : ℝ × ℝ) : 
  (A.1 = 0 ∧ A.2 = 0) →  -- A is at origin
  (B.1 = 1 ∧ B.2 = 0) →  -- B is at (1, 0)
  (C.1 = 0 ∧ C.2 = 1) →  -- C is at (0, 1)
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 →  -- AB = AC (isosceles)
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 →  -- right angle at A
  (1/2) * (B.1 - A.1) * (C.2 - A.2) = 1/4 :=
by sorry

end isosceles_right_triangle_area_l2966_296673


namespace line_and_symmetric_point_l2966_296694

/-- A line with inclination angle 135° passing through (1, 1) -/
structure Line :=
  (equation : ℝ → ℝ → Prop)
  (passes_through : equation 1 1)
  (inclination : Real.tan (135 * π / 180) = -1)

/-- The symmetric point of A with respect to a line -/
def symmetric_point (l : Line) (A : ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem line_and_symmetric_point (l : Line) :
  l.equation = fun x y ↦ x + y - 2 = 0 ∧
  symmetric_point l (3, 4) = (-2, -1) :=
sorry

end line_and_symmetric_point_l2966_296694


namespace second_number_proof_l2966_296615

theorem second_number_proof (x y z : ℚ) 
  (sum_eq : x + y + z = 150)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 5 / 7) :
  y = 1000 / 21 := by
  sorry

end second_number_proof_l2966_296615


namespace test_questions_count_l2966_296645

theorem test_questions_count (S I C : ℕ) : 
  S = C - 2 * I →
  S = 73 →
  C = 91 →
  C + I = 100 := by
sorry

end test_questions_count_l2966_296645


namespace f_derivative_at_zero_l2966_296638

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.arctan (x^2 * Real.sin (1 / (5 * x)))
  else 0

theorem f_derivative_at_zero :
  deriv f 0 = 0 := by sorry

end f_derivative_at_zero_l2966_296638


namespace carpet_design_problem_l2966_296696

/-- Represents the dimensions of a rectangular region in the carpet design. -/
structure RegionDimensions where
  length : ℝ
  width : ℝ

/-- Represents the area of a region in the carpet design. -/
def area (d : RegionDimensions) : ℝ := d.length * d.width

/-- Checks if three real numbers form an arithmetic sequence. -/
def isArithmeticSequence (a b c : ℝ) : Prop := b - a = c - b

/-- The carpet design problem. -/
theorem carpet_design_problem (inner middle outer : RegionDimensions) 
    (h1 : inner.width = 2)
    (h2 : middle.width = inner.width + 4)
    (h3 : middle.length = inner.length + 4)
    (h4 : outer.width = middle.width + 4)
    (h5 : outer.length = middle.length + 4)
    (h6 : isArithmeticSequence (area inner) (area middle) (area outer)) :
    inner.length = 4 := by
  sorry

end carpet_design_problem_l2966_296696


namespace find_a_value_l2966_296641

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem find_a_value :
  (∀ x, f (x + 3) = 3 * f x) →
  (∀ x ∈ Set.Ioo 0 3, f x = Real.log x - a * x) →
  (a > 1/3) →
  (Set.Ioo (-6) (-3)).image f ⊆ Set.Iic (-1/9) →
  (∃ x ∈ Set.Ioo (-6) (-3), f x = -1/9) →
  a = 1 := by sorry

end find_a_value_l2966_296641


namespace salary_calculation_l2966_296658

/-- The monthly salary of a man who saves 20% of his salary and can save Rs. 230 when expenses increase by 20% -/
def monthlySalary : ℝ := 1437.5

theorem salary_calculation (savings_rate : ℝ) (expense_increase : ℝ) (reduced_savings : ℝ)
    (h1 : savings_rate = 0.20)
    (h2 : expense_increase = 0.20)
    (h3 : reduced_savings = 230)
    (h4 : savings_rate * monthlySalary - expense_increase * (savings_rate * monthlySalary) = reduced_savings) :
  monthlySalary = 1437.5 := by
  sorry

end salary_calculation_l2966_296658


namespace function_is_identity_l2966_296601

-- Define the property that the function f must satisfy
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (2 * x + f y) = x + y + f x

-- Theorem statement
theorem function_is_identity 
  (f : ℝ → ℝ) 
  (h : satisfies_equation f) : 
  ∀ x : ℝ, f x = x := by
  sorry

end function_is_identity_l2966_296601


namespace sqrt_product_plus_one_equals_341_l2966_296656

theorem sqrt_product_plus_one_equals_341 : 
  Real.sqrt ((20 : ℝ) * 19 * 18 * 17 + 1) = 341 := by
  sorry

end sqrt_product_plus_one_equals_341_l2966_296656


namespace final_s_value_l2966_296655

/-- Represents the state of the variables in the loop -/
structure LoopState where
  i : ℕ
  s : ℤ

/-- Defines a single iteration of the loop -/
def loopIteration (state : LoopState) : LoopState :=
  { i := state.i + 1,
    s := 2 * state.s - 1 }

/-- Applies the loop iteration n times -/
def applyNTimes (n : ℕ) (state : LoopState) : LoopState :=
  match n with
  | 0 => state
  | n + 1 => loopIteration (applyNTimes n state)

/-- The main theorem to prove -/
theorem final_s_value :
  let initialState : LoopState := { i := 1, s := 0 }
  let finalState := applyNTimes 5 initialState
  finalState.s = -31 := by sorry

end final_s_value_l2966_296655


namespace no_solution_in_A_l2966_296604

-- Define the set A
def A : Set ℕ :=
  {n : ℕ | ∃ k : ℤ, |n * Real.sqrt 2022 - 1/3 - ↑k| ≤ 1/2022}

-- State the theorem
theorem no_solution_in_A :
  ∀ x y z : ℕ, x ∈ A → y ∈ A → z ∈ A → 20 * x + 21 * y ≠ 22 * z :=
by sorry

end no_solution_in_A_l2966_296604


namespace monotonic_decreasing_implies_a_bound_l2966_296661

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a - 1)*x + 2

-- State the theorem
theorem monotonic_decreasing_implies_a_bound (a : ℝ) :
  (∀ x y : ℝ, x < y → y ≤ 4 → f a y < f a x) → a ≤ -7 := by
  sorry

end monotonic_decreasing_implies_a_bound_l2966_296661


namespace speed_ratio_proof_l2966_296626

/-- The speed of A in yards per minute -/
def speed_A : ℝ := 333.33

/-- The speed of B in yards per minute -/
def speed_B : ℝ := 433.33

/-- The initial distance of B from point O in yards -/
def initial_distance_B : ℝ := 1000

/-- The time when A and B are first equidistant from O in minutes -/
def time_first_equidistant : ℝ := 3

/-- The time when A and B are second equidistant from O in minutes -/
def time_second_equidistant : ℝ := 10

theorem speed_ratio_proof :
  (∀ t : ℝ, t = time_first_equidistant → 
    (speed_A * t)^2 = (initial_distance_B - speed_B * t)^2) ∧
  (∀ t : ℝ, t = time_second_equidistant → 
    (speed_A * t)^2 = (initial_distance_B - speed_B * t)^2) →
  speed_A / speed_B = 333 / 433 := by
  sorry

end speed_ratio_proof_l2966_296626


namespace max_side_length_of_special_triangle_l2966_296679

theorem max_side_length_of_special_triangle (a b c : ℕ) : 
  a < b → b < c →                 -- Three different side lengths
  a + b + c = 24 →                -- Perimeter is 24
  a + b > c → b + c > a → c + a > b →  -- Triangle inequality
  c ≤ 11 := by
sorry

end max_side_length_of_special_triangle_l2966_296679


namespace restaurant_bill_proof_l2966_296690

theorem restaurant_bill_proof (total_friends : Nat) (paying_friends : Nat) (extra_payment : ℝ) :
  total_friends = 10 →
  paying_friends = 9 →
  extra_payment = 3 →
  ∃ (bill : ℝ), (paying_friends : ℝ) * ((bill / total_friends) + extra_payment) = bill ∧ bill = 270 :=
by
  sorry

end restaurant_bill_proof_l2966_296690


namespace circumcircle_equation_l2966_296603

/-- Given the vertices of triangle ABC: A(4,4), B(5,3), and C(1,1),
    prove that the equation of the circumcircle is x^2 + y^2 - 6x - 4y + 8 = 0 -/
theorem circumcircle_equation (A B C : ℝ × ℝ) :
  A = (4, 4) → B = (5, 3) → C = (1, 1) →
  ∃ D E F : ℝ, ∀ x y : ℝ,
    (x^2 + y^2 + D*x + E*y + F = 0 ↔
     ((x - 4)^2 + (y - 4)^2 = 0 ∨
      (x - 5)^2 + (y - 3)^2 = 0 ∨
      (x - 1)^2 + (y - 1)^2 = 0)) →
    D = -6 ∧ E = -4 ∧ F = 8 := by
  sorry


end circumcircle_equation_l2966_296603


namespace feline_sanctuary_tigers_l2966_296644

theorem feline_sanctuary_tigers (lions cougars tigers : ℕ) : 
  lions = 12 →
  cougars = (lions + tigers) / 2 →
  lions + tigers + cougars = 39 →
  tigers = 14 := by
sorry

end feline_sanctuary_tigers_l2966_296644


namespace gcd_7384_12873_l2966_296669

theorem gcd_7384_12873 : Nat.gcd 7384 12873 = 1 := by
  sorry

end gcd_7384_12873_l2966_296669


namespace triangle_side_range_l2966_296697

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the sine law
def sineLaw (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧
  t.b / Real.sin t.B = t.c / Real.sin t.C ∧
  t.c / Real.sin t.C = t.a / Real.sin t.A

-- Define the condition for two solutions
def hasTwoSolutions (t : Triangle) : Prop :=
  ∃ (t1 t2 : Triangle), t1 ≠ t2 ∧ sineLaw t1 ∧ sineLaw t2 ∧
    t1.b = t.b ∧ t1.B = t.B ∧ t2.b = t.b ∧ t2.B = t.B

-- State the theorem
theorem triangle_side_range (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : t.B = π/4)  -- 45° in radians
  (h3 : hasTwoSolutions t) :
  2 < t.a ∧ t.a < 2 * Real.sqrt 2 :=
sorry

end triangle_side_range_l2966_296697


namespace multiply_mixed_number_l2966_296623

theorem multiply_mixed_number : (7 : ℚ) * (9 + 2/5) = 65 + 4/5 := by
  sorry

end multiply_mixed_number_l2966_296623


namespace part1_part2_l2966_296611

-- Part 1
def f (m x : ℝ) : ℝ := x^2 - (m + 2) * x + 3

def has_max_min_in_range (m : ℝ) : Prop :=
  ∃ (M N : ℝ), (∀ x ∈ Set.Icc 1 2, f m x ≤ M ∧ f m x ≥ N) ∧ M - N ≤ 2

theorem part1 (m : ℝ) : has_max_min_in_range m → m ∈ Set.Icc (-1) 3 := by sorry

-- Part 2
def has_solution_in_range (m : ℝ) : Prop :=
  ∃ x ∈ Set.Icc 0 2, x^2 - (m + 2) * x + 3 = -(2 * m + 1) * x + 2

theorem part2 (m : ℝ) : has_solution_in_range m → m ∈ Set.Iic (-1) := by sorry

end part1_part2_l2966_296611


namespace g_of_5_equals_27_l2966_296657

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x + 2

-- Theorem statement
theorem g_of_5_equals_27 : g 5 = 27 := by
  sorry

end g_of_5_equals_27_l2966_296657


namespace sum_of_fractions_equals_one_l2966_296637

theorem sum_of_fractions_equals_one 
  {x y z : ℝ} (h : x * y * z = 1) : 
  1 / (1 + x + x * y) + 1 / (1 + y + y * z) + 1 / (1 + z + z * x) = 1 := by
sorry

end sum_of_fractions_equals_one_l2966_296637


namespace hiring_probability_l2966_296682

/-- The number of candidates -/
def numCandidates : ℕ := 4

/-- The number of people to be hired -/
def numHired : ℕ := 2

/-- The probability of hiring at least one of two specific candidates -/
def probAtLeastOne : ℚ := 5/6

theorem hiring_probability :
  (numCandidates : ℚ) > 0 ∧ numHired ≤ numCandidates →
  (1 : ℚ) - (Nat.choose (numCandidates - 2) numHired : ℚ) / (Nat.choose numCandidates numHired : ℚ) = probAtLeastOne :=
sorry

end hiring_probability_l2966_296682


namespace cottonwood_fiber_diameter_scientific_notation_l2966_296647

theorem cottonwood_fiber_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.0000108 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.08 ∧ n = -5 :=
by sorry

end cottonwood_fiber_diameter_scientific_notation_l2966_296647


namespace complex_fraction_simplification_l2966_296666

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  ((2 + i) * (3 - 4*i)) / (2 - i) = 5 := by sorry

end complex_fraction_simplification_l2966_296666


namespace min_operations_to_identify_controllers_l2966_296621

/-- The number of light bulbs and buttons -/
def n : ℕ := 64

/-- An operation consists of pressing a set of buttons and recording the on/off state of each light bulb -/
def Operation := Fin n → Bool

/-- A sequence of operations -/
def OperationSequence := List Operation

/-- The result of applying a sequence of operations to all light bulbs -/
def ApplyOperations (ops : OperationSequence) : Fin n → List Bool :=
  fun i => ops.map (fun op => op i)

/-- A mapping from light bulbs to their controlling buttons -/
def ControlMapping := Fin n → Fin n

theorem min_operations_to_identify_controllers :
  ∃ (k : ℕ), 
    (∃ (ops : OperationSequence), ops.length = k ∧
      (∀ (m : ControlMapping), Function.Injective m →
        Function.Injective (ApplyOperations ops ∘ m))) ∧
    (∀ (j : ℕ), j < k →
      ¬∃ (ops : OperationSequence), ops.length = j ∧
        (∀ (m : ControlMapping), Function.Injective m →
          Function.Injective (ApplyOperations ops ∘ m))) ∧
    k = 6 :=
  sorry

end min_operations_to_identify_controllers_l2966_296621


namespace gear_rotations_l2966_296687

/-- Represents a gear with a given number of teeth -/
structure Gear where
  teeth : ℕ

/-- Represents a system of two engaged gears -/
structure GearSystem where
  gearA : Gear
  gearB : Gear

/-- Checks if the rotations of two gears are valid (i.e., they mesh properly) -/
def validRotations (gs : GearSystem) (rotA : ℕ) (rotB : ℕ) : Prop :=
  rotA * gs.gearA.teeth = rotB * gs.gearB.teeth

/-- Checks if the given rotations are the smallest possible -/
def smallestRotations (gs : GearSystem) (rotA : ℕ) (rotB : ℕ) : Prop :=
  ∀ (a b : ℕ), validRotations gs a b → (rotA ≤ a ∧ rotB ≤ b)

/-- The main theorem to prove -/
theorem gear_rotations (gs : GearSystem) (h1 : gs.gearA.teeth = 12) (h2 : gs.gearB.teeth = 54) :
  smallestRotations gs 9 2 :=
sorry

end gear_rotations_l2966_296687


namespace red_balls_count_l2966_296676

theorem red_balls_count (total : ℕ) (white : ℕ) (green : ℕ) (yellow : ℕ) (purple : ℕ) (p : ℚ) :
  total = 100 →
  white = 20 →
  green = 30 →
  yellow = 10 →
  purple = 3 →
  p = 0.6 →
  p = (white + green + yellow : ℚ) / total →
  ∃ red : ℕ, red = 3 ∧ total = white + green + yellow + red + purple :=
by sorry

end red_balls_count_l2966_296676


namespace initial_average_calculation_l2966_296609

/-- Calculates the initial average daily production given the number of days,
    today's production, and the new average. -/
def initial_average (n : ℕ) (today_production : ℕ) (new_average : ℕ) : ℚ :=
  ((n + 1 : ℕ) * new_average - today_production) / n

theorem initial_average_calculation :
  initial_average 12 115 55 = 50 := by sorry

end initial_average_calculation_l2966_296609


namespace sin_15_mul_sin_75_eq_quarter_l2966_296630

theorem sin_15_mul_sin_75_eq_quarter : Real.sin (15 * π / 180) * Real.sin (75 * π / 180) = 1 / 4 := by
  sorry

end sin_15_mul_sin_75_eq_quarter_l2966_296630


namespace cosine_sum_equality_l2966_296698

theorem cosine_sum_equality : 
  Real.cos (43 * π / 180) * Real.cos (77 * π / 180) - 
  Real.sin (43 * π / 180) * Real.sin (77 * π / 180) = -1/2 := by
sorry

end cosine_sum_equality_l2966_296698


namespace square_field_area_l2966_296606

/-- The area of a square field with side length 6 meters is 36 square meters. -/
theorem square_field_area :
  let side_length : ℝ := 6
  let field_area : ℝ := side_length ^ 2
  field_area = 36 := by sorry

end square_field_area_l2966_296606


namespace percentage_of_b_l2966_296667

theorem percentage_of_b (a b c : ℝ) (h1 : 8 = 0.02 * a) (h2 : c = b / a) : 
  ∃ p : ℝ, p * b = 2 ∧ p = 0.005 := by sorry

end percentage_of_b_l2966_296667


namespace tree_growth_rate_l2966_296639

def initial_height : ℝ := 600
def final_height : ℝ := 720
def growth_period : ℝ := 240

theorem tree_growth_rate :
  (final_height - initial_height) / growth_period = 0.5 := by
  sorry

end tree_growth_rate_l2966_296639


namespace lcm_problem_l2966_296695

theorem lcm_problem (m n : ℕ) : 
  m > 0 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  Nat.lcm m n = 690 ∧ 
  ¬(3 ∣ n) ∧ 
  ¬(2 ∣ m) → 
  n = 230 := by
sorry

end lcm_problem_l2966_296695


namespace determine_sequence_from_final_state_l2966_296619

/-- Represents the state of the cards at each step -/
structure CardState where
  red : ℤ
  blue : ℤ

/-- Applies the transformation to the cards given k -/
def transform (state : CardState) (k : ℕ+) : CardState :=
  { red := k * state.red + state.blue
  , blue := state.red }

/-- Applies n transformations to the initial state using the sequence ks -/
def apply_transformations (initial : CardState) (ks : List ℕ+) : CardState :=
  ks.foldl transform initial

/-- States that it's possible to determine the sequence from the final state -/
theorem determine_sequence_from_final_state 
  (n : ℕ) 
  (ks : List ℕ+) 
  (h_length : ks.length = n) 
  (initial : CardState) 
  (h_initial : initial.red > initial.blue) :
  ∃ (f : CardState → List ℕ+), 
    f (apply_transformations initial ks) = ks :=
sorry

end determine_sequence_from_final_state_l2966_296619


namespace line_through_point_l2966_296675

/-- Given a line equation 2 - 3kx = -4y that passes through the point (3, -2),
    prove that k = -2/3 is the unique value that satisfies the equation. -/
theorem line_through_point (k : ℚ) : 
  (2 - 3 * k * 3 = -4 * (-2)) ↔ k = -2/3 := by sorry

end line_through_point_l2966_296675


namespace larger_root_of_equation_l2966_296663

theorem larger_root_of_equation (x : ℚ) : 
  (x - 2/3) * (x - 2/3) + 2 * (x - 2/3) * (x - 4/5) = 0 →
  (x = 2/3 ∨ x = 14/15) ∧ 
  (∀ y, (y - 2/3) * (y - 2/3) + 2 * (y - 2/3) * (y - 4/5) = 0 → y ≤ 14/15) :=
by sorry

#check larger_root_of_equation

end larger_root_of_equation_l2966_296663


namespace min_fold_length_l2966_296634

theorem min_fold_length (width height : ℝ) (hw : width = 8) (hh : height = 11) :
  let min_length := fun y : ℝ => Real.sqrt (width^2 + (y - height)^2)
  ∃ (y : ℝ), y ∈ Set.Icc 0 height ∧
    ∀ (z : ℝ), z ∈ Set.Icc 0 height → min_length y ≤ min_length z ∧
    min_length y = width :=
by sorry

end min_fold_length_l2966_296634


namespace octal_addition_l2966_296632

/-- Converts a base-10 integer to its octal representation -/
def to_octal (n : ℕ) : ℕ := sorry

/-- Converts an octal representation to base-10 integer -/
def from_octal (n : ℕ) : ℕ := sorry

theorem octal_addition : to_octal (from_octal 321 + from_octal 127) = 450 := by sorry

end octal_addition_l2966_296632


namespace age_ratio_theorem_l2966_296671

/-- Given a person's present age is 14 years, this theorem proves that the ratio of their age 
    16 years hence to their age 4 years ago is 3:1. -/
theorem age_ratio_theorem (present_age : ℕ) (h : present_age = 14) : 
  (present_age + 16) / (present_age - 4) = 3 := by
  sorry

#check age_ratio_theorem

end age_ratio_theorem_l2966_296671


namespace add_5_16_base8_l2966_296618

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def base10ToBase8 (n : ℕ) : ℕ := sorry

/-- Adds two base-8 numbers and returns the result in base-8 --/
def addBase8 (a b : ℕ) : ℕ :=
  base10ToBase8 (base8ToBase10 a + base8ToBase10 b)

theorem add_5_16_base8 :
  addBase8 5 16 = 23 := by sorry

end add_5_16_base8_l2966_296618


namespace grid_property_l2966_296650

-- Define a 4x4 grid of rational numbers
def Grid := Matrix (Fin 4) (Fin 4) ℚ

-- Define what it means for a row to be an arithmetic sequence
def is_arithmetic_row (g : Grid) (i : Fin 4) : Prop :=
  ∃ a d : ℚ, ∀ j : Fin 4, g i j = a + d * j

-- Define what it means for a column to be an arithmetic sequence
def is_arithmetic_col (g : Grid) (j : Fin 4) : Prop :=
  ∃ a d : ℚ, ∀ i : Fin 4, g i j = a + d * i

-- Main theorem
theorem grid_property (g : Grid) : 
  (∀ i : Fin 4, is_arithmetic_row g i) →
  (∀ j : Fin 4, is_arithmetic_col g j) →
  g 0 0 = 3 →
  g 0 3 = 18 →
  g 3 0 = 11 →
  g 3 3 = 50 →
  g 1 2 = 21 := by
  sorry

end grid_property_l2966_296650


namespace coin_distribution_problem_l2966_296643

theorem coin_distribution_problem :
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ 9 * x + 17 * y = 70 ∧ x = 4 ∧ y = 2 := by
  sorry

end coin_distribution_problem_l2966_296643


namespace fraction_evaluation_l2966_296602

theorem fraction_evaluation : (3^6 - 9 * 3^3 + 27) / (3^3 - 3) = 24.75 := by
  sorry

end fraction_evaluation_l2966_296602


namespace sum_xyz_equals_five_l2966_296610

theorem sum_xyz_equals_five (x y z : ℝ) 
  (eq1 : x + 2*y + 3*z = 10) 
  (eq2 : 4*x + 3*y + 2*z = 15) : 
  x + y + z = 5 := by
sorry

end sum_xyz_equals_five_l2966_296610


namespace sum_of_S_and_T_is_five_l2966_296651

theorem sum_of_S_and_T_is_five : 
  ∀ (S T : ℝ),
  let line_length : ℝ := 5
  let num_parts : ℕ := 20
  let part_length : ℝ := line_length / num_parts
  S = 5 * part_length →
  T = line_length - 5 * part_length →
  S + T = 5 := by
sorry

end sum_of_S_and_T_is_five_l2966_296651


namespace corresponding_angles_equality_incomplete_l2966_296627

-- Define the concept of corresponding angles
def corresponding_angles (α β : ℝ) : Prop := sorry

-- Define the concept of parallel lines
def parallel_lines (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem stating that the proposition "corresponding angles are equal" is false
-- when not explicitly specifying that the lines are parallel
theorem corresponding_angles_equality_incomplete :
  ¬ ∀ (α β : ℝ) (l₁ l₂ : Set (ℝ × ℝ)), corresponding_angles α β → α = β :=
sorry

end corresponding_angles_equality_incomplete_l2966_296627


namespace probability_JQKA_standard_deck_l2966_296691

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards of each rank (Jack, Queen, King, Ace) in a standard deck -/
def CardsPerRank : ℕ := 4

/-- Calculates the probability of drawing a specific sequence of four cards (Jack, Queen, King, Ace) from a standard deck without replacement -/
def probability_JQKA (deck : ℕ) (cards_per_rank : ℕ) : ℚ :=
  (cards_per_rank : ℚ) / deck *
  (cards_per_rank : ℚ) / (deck - 1) *
  (cards_per_rank : ℚ) / (deck - 2) *
  (cards_per_rank : ℚ) / (deck - 3)

/-- Theorem stating that the probability of drawing Jack, Queen, King, Ace in order from a standard deck without replacement is 64/1624350 -/
theorem probability_JQKA_standard_deck :
  probability_JQKA StandardDeck CardsPerRank = 64 / 1624350 := by
  sorry

end probability_JQKA_standard_deck_l2966_296691


namespace second_group_size_l2966_296652

/-- The number of men in the first group -/
def first_group : ℕ := 20

/-- The number of days taken by the first group -/
def first_days : ℕ := 30

/-- The number of days taken by the second group -/
def second_days : ℕ := 24

/-- The total amount of work in man-days -/
def total_work : ℕ := first_group * first_days

/-- The number of men in the second group -/
def second_group : ℕ := total_work / second_days

theorem second_group_size : second_group = 25 := by
  sorry

end second_group_size_l2966_296652


namespace birds_remaining_proof_l2966_296699

/-- Calculates the number of birds remaining on a fence after some fly away. -/
def birdsRemaining (initialBirds flownAway : ℝ) : ℝ :=
  initialBirds - flownAway

/-- Theorem stating that the number of birds remaining is the difference between
    the initial number and the number that flew away. -/
theorem birds_remaining_proof (initialBirds flownAway : ℝ) :
  birdsRemaining initialBirds flownAway = initialBirds - flownAway := by
  sorry

/-- Example calculation for the specific problem -/
example : birdsRemaining 15.3 6.5 = 8.8 := by
  sorry

end birds_remaining_proof_l2966_296699


namespace warehouse_loading_theorem_l2966_296654

/-- Represents the warehouse loading problem -/
def warehouseLoading (crateCapacity : ℕ) (numCrates : ℕ) 
                     (nailBags : ℕ) (nailWeight : ℕ) 
                     (hammerBags : ℕ) (hammerWeight : ℕ) 
                     (plankBags : ℕ) (plankWeight : ℕ) : Prop :=
  let totalWeight := nailBags * nailWeight + hammerBags * hammerWeight + plankBags * plankWeight
  let totalCapacity := crateCapacity * numCrates
  totalWeight - totalCapacity = 80

/-- Theorem stating the weight to be left out in the warehouse loading problem -/
theorem warehouse_loading_theorem : 
  warehouseLoading 20 15 4 5 12 5 10 30 := by
  sorry

end warehouse_loading_theorem_l2966_296654


namespace sqrt_three_simplification_l2966_296653

theorem sqrt_three_simplification : 3 * Real.sqrt 3 - 2 * Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end sqrt_three_simplification_l2966_296653


namespace inscribed_circle_radius_right_triangle_l2966_296678

theorem inscribed_circle_radius_right_triangle 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_leg : a = 15) 
  (h_proj : c - b = 16) : 
  (a + b - c) / 2 = 5 := by
sorry

end inscribed_circle_radius_right_triangle_l2966_296678


namespace complex_equation_solution_l2966_296646

theorem complex_equation_solution (i : ℂ) (h_i : i^2 = -1) :
  ∀ z : ℂ, i * z = 1 → z = -i := by
  sorry

end complex_equation_solution_l2966_296646


namespace jessica_age_l2966_296680

theorem jessica_age :
  (∀ (jessica_age claire_age : ℕ),
    jessica_age = claire_age + 6 →
    claire_age + 2 = 20 →
    jessica_age = 24) :=
by sorry

end jessica_age_l2966_296680


namespace equation_represents_two_lines_l2966_296635

/-- The equation x^2 - 72y^2 - 16x + 64 = 0 represents two lines in the xy-plane. -/
theorem equation_represents_two_lines :
  ∃ (a b c d : ℝ), ∀ (x y : ℝ),
    (x^2 - 72*y^2 - 16*x + 64 = 0) ↔ ((x = a*y + b) ∨ (x = c*y + d)) :=
by sorry

end equation_represents_two_lines_l2966_296635


namespace num_parallel_planes_zero_or_one_l2966_296670

/-- Two lines in 3D space are skew if they are not parallel and do not intersect. -/
def are_skew (a b : Line3D) : Prop := sorry

/-- A point is outside a line if it does not lie on the line. -/
def is_outside (P : Point3D) (l : Line3D) : Prop := sorry

/-- A plane is parallel to a line if they do not intersect. -/
def plane_parallel_to_line (π : Plane3D) (l : Line3D) : Prop := sorry

/-- The number of planes passing through a point and parallel to two lines. -/
def num_parallel_planes (P : Point3D) (a b : Line3D) : ℕ := sorry

theorem num_parallel_planes_zero_or_one 
  (P : Point3D) (a b : Line3D) 
  (h_skew : are_skew a b) 
  (h_outside_a : is_outside P a) 
  (h_outside_b : is_outside P b) : 
  num_parallel_planes P a b = 0 ∨ num_parallel_planes P a b = 1 := by
  sorry

end num_parallel_planes_zero_or_one_l2966_296670


namespace color_distance_existence_l2966_296625

-- Define the color type
inductive Color
| Red
| Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the coloring function
def coloring : Point → Color := sorry

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- The main theorem
theorem color_distance_existence (x : ℝ) (h : x > 0) :
  ∃ (c : Color), ∃ (p1 p2 : Point), coloring p1 = c ∧ coloring p2 = c ∧ distance p1 p2 = x := by
  sorry

end color_distance_existence_l2966_296625
