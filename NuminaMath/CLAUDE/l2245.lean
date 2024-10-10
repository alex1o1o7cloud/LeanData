import Mathlib

namespace series_equation_solutions_l2245_224594

def series_sum (x : ℝ) : ℝ := 1 + 3*x + 7*x^2 + 11*x^3 + 15*x^4 + 19*x^5 + 23*x^6 + 27*x^7 + 31*x^8 + 35*x^9 + 39*x^10

theorem series_equation_solutions :
  ∃ (x₁ x₂ : ℝ), 
    x₁ ≠ x₂ ∧
    -1 < x₁ ∧ x₁ < 1 ∧
    -1 < x₂ ∧ x₂ < 1 ∧
    series_sum x₁ = 50 ∧
    series_sum x₂ = 50 ∧
    abs (x₁ - 0.959) < 0.001 ∧
    abs (x₂ - 0.021) < 0.001 :=
by sorry

end series_equation_solutions_l2245_224594


namespace sine_cosine_function_minimum_l2245_224543

theorem sine_cosine_function_minimum (a ω : ℝ) : 
  a > 0 → ω > 0 → 
  (∃ (f : ℝ → ℝ), 
    (∀ x, f x = Real.sin (ω * x) + a * Real.cos (ω * x)) ∧ 
    f (π / 6) = -2 ∧ 
    (∀ x, f x ≥ -2)) → 
  ω = 7 := by sorry

end sine_cosine_function_minimum_l2245_224543


namespace equation_solution_l2245_224561

theorem equation_solution :
  ∃ (x : ℚ), x ≠ -3 ∧ (x^2 + 3*x + 4) / (x + 3) = x + 6 ↔ x = -7/3 := by
  sorry

end equation_solution_l2245_224561


namespace find_divisor_l2245_224540

theorem find_divisor : ∃ (d : ℕ), d > 1 ∧ 
  (1054 + 4 = 1058) ∧ 
  (1058 % d = 0) ∧
  (∀ k : ℕ, k < 4 → (1054 + k) % d ≠ 0) →
  d = 2 := by
  sorry

end find_divisor_l2245_224540


namespace range_of_a_l2245_224553

def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x | -a < x ∧ x ≤ a + 3}

theorem range_of_a (a : ℝ) : B a ⊆ (A ∩ B a) → a ≤ -1 ∧ a ∈ Set.Iic (-1 : ℝ) := by
  sorry

end range_of_a_l2245_224553


namespace smallest_product_l2245_224598

def S : Set Int := {-10, -3, 0, 2, 6}

theorem smallest_product (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x * y ≤ a * b ∧ x * y = -60 :=
sorry

end smallest_product_l2245_224598


namespace percentage_comparison_l2245_224581

theorem percentage_comparison (w x y z t : ℝ) 
  (hw : w = 0.6 * x)
  (hx : x = 0.6 * y)
  (hz : z = 0.54 * y)
  (ht : t = 0.48 * x) :
  (z - w) / w * 100 = 50 ∧ (w - t) / w * 100 = 20 :=
by sorry

end percentage_comparison_l2245_224581


namespace solve_abc_l2245_224510

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + a*x - 12 = 0}
def B (b c : ℝ) : Set ℝ := {x | x^2 + b*x + c = 0}

-- State the theorem
theorem solve_abc (a b c : ℝ) : 
  A a ≠ B b c ∧ 
  A a ∪ B b c = {-3, 4} ∧
  A a ∩ B b c = {-3} →
  a = -1 ∧ b = 6 ∧ c = 9 := by
sorry

end solve_abc_l2245_224510


namespace highest_power_of_two_dividing_difference_of_fifth_powers_l2245_224569

theorem highest_power_of_two_dividing_difference_of_fifth_powers :
  ∃ k : ℕ, k = 5 ∧ 2^k ∣ (17^5 - 15^5) ∧ ∀ m : ℕ, 2^m ∣ (17^5 - 15^5) → m ≤ k :=
by sorry

end highest_power_of_two_dividing_difference_of_fifth_powers_l2245_224569


namespace red_triangle_or_blue_quadrilateral_l2245_224549

/-- A type representing the color of an edge --/
inductive Color
| Red
| Blue

/-- A complete graph with 9 vertices --/
def Graph9 := Fin 9 → Fin 9 → Color

/-- A predicate that checks if a graph is complete --/
def is_complete (g : Graph9) : Prop :=
  ∀ i j : Fin 9, i ≠ j → (g i j = Color.Red ∨ g i j = Color.Blue)

/-- A predicate that checks if three vertices form a red triangle --/
def has_red_triangle (g : Graph9) : Prop :=
  ∃ i j k : Fin 9, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    g i j = Color.Red ∧ g j k = Color.Red ∧ g i k = Color.Red

/-- A predicate that checks if four vertices form a blue complete quadrilateral --/
def has_blue_quadrilateral (g : Graph9) : Prop :=
  ∃ i j k l : Fin 9, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    g i j = Color.Blue ∧ g i k = Color.Blue ∧ g i l = Color.Blue ∧
    g j k = Color.Blue ∧ g j l = Color.Blue ∧ g k l = Color.Blue

/-- The main theorem --/
theorem red_triangle_or_blue_quadrilateral (g : Graph9) 
  (h : is_complete g) : has_red_triangle g ∨ has_blue_quadrilateral g := by
  sorry

end red_triangle_or_blue_quadrilateral_l2245_224549


namespace twenty_percent_less_than_sixty_l2245_224573

theorem twenty_percent_less_than_sixty (x : ℝ) : x + (1/3) * x = 60 - 0.2 * 60 → x = 36 := by
  sorry

end twenty_percent_less_than_sixty_l2245_224573


namespace baby_births_theorem_l2245_224538

theorem baby_births_theorem (k : ℕ) (x : ℕ → ℕ) 
  (h1 : 1014 < k) (h2 : k ≤ 2014)
  (h3 : x 0 = 0) (h4 : x k = 2014)
  (h5 : ∀ i, i < k → x i < x (i + 1)) :
  ∃ i j, i < j ∧ j ≤ k ∧ x j - x i = 100 := by
sorry

end baby_births_theorem_l2245_224538


namespace divisibility_by_ten_l2245_224582

theorem divisibility_by_ten (x y : Nat) : 
  x < 10 → y < 10 → x + y = 2 → (65300 + 10 * x + y) % 10 = 0 → x = 2 ∧ y = 0 := by
  sorry

end divisibility_by_ten_l2245_224582


namespace prob_comparison_l2245_224568

/-- The probability of drawing two balls of the same color from two bags -/
def prob_same_color (m n : ℕ) : ℚ :=
  2 * m * n / ((m + n) * (m + n))

/-- The probability of drawing two balls of different colors from two bags -/
def prob_diff_color (m n : ℕ) : ℚ :=
  (m * m + n * n) / ((m + n) * (m + n))

theorem prob_comparison (m n : ℕ) :
  prob_same_color m n ≤ prob_diff_color m n ∧
  (prob_same_color m n = prob_diff_color m n ↔ m = n) :=
sorry

end prob_comparison_l2245_224568


namespace arithmetic_progression_divisibility_l2245_224596

theorem arithmetic_progression_divisibility 
  (a : ℕ → ℕ) 
  (h_ap : ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d) 
  (h_div : ∀ n : ℕ, (a n * a (n + 31)) % 2005 = 0) : 
  ∀ n : ℕ, a n % 2005 = 0 := by
sorry

end arithmetic_progression_divisibility_l2245_224596


namespace franks_remaining_money_l2245_224519

/-- Calculates the remaining money after Frank buys the most expensive lamp -/
def remaining_money (cheapest_lamp_cost : ℝ) (expensive_lamp_multiplier : ℝ) 
  (discount_rate : ℝ) (sales_tax_rate : ℝ) (initial_money : ℝ) : ℝ :=
  let expensive_lamp_cost := cheapest_lamp_cost * expensive_lamp_multiplier
  let discounted_price := expensive_lamp_cost * (1 - discount_rate)
  let final_price := discounted_price * (1 + sales_tax_rate)
  initial_money - final_price

/-- Theorem stating that Frank's remaining money is $31.68 -/
theorem franks_remaining_money :
  remaining_money 20 3 0.1 0.08 90 = 31.68 := by
  sorry

end franks_remaining_money_l2245_224519


namespace man_rowing_speed_l2245_224530

/-- Calculates the speed of a man rowing in still water given his downstream speed and current speed -/
theorem man_rowing_speed (current_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  current_speed = 6 →
  distance = 100 →
  time = 14.998800095992323 →
  (distance / time - current_speed * 1000 / 3600) * 3.6 = 18 := by
sorry

end man_rowing_speed_l2245_224530


namespace cubic_root_sum_product_l2245_224533

theorem cubic_root_sum_product (p q r : ℂ) : 
  (6 * p ^ 3 - 5 * p ^ 2 + 13 * p - 10 = 0) →
  (6 * q ^ 3 - 5 * q ^ 2 + 13 * q - 10 = 0) →
  (6 * r ^ 3 - 5 * r ^ 2 + 13 * r - 10 = 0) →
  p * q + q * r + r * p = 13 / 6 := by
sorry

end cubic_root_sum_product_l2245_224533


namespace max_unique_sums_l2245_224580

def coin_values : List ℕ := [1, 1, 1, 5, 10, 25]

def unique_sums (values : List ℕ) : Finset ℕ :=
  (values.map (λ x => values.map (λ y => x + y))).join.toFinset

theorem max_unique_sums :
  Finset.card (unique_sums coin_values) = 7 := by sorry

end max_unique_sums_l2245_224580


namespace inequality_holds_l2245_224562

theorem inequality_holds (x : ℝ) : 
  (4 * x^2) / (1 - Real.sqrt (1 + 2*x))^2 < 2*x + 9 ↔ 
  (x ≥ -1/2 ∧ x < 0) ∨ (x > 0 ∧ x < 45/8) := by sorry

end inequality_holds_l2245_224562


namespace jinas_mascots_l2245_224516

/-- The number of mascots Jina has -/
def total_mascots (teddies bunnies koalas additional_teddies : ℕ) : ℕ :=
  teddies + bunnies + koalas + additional_teddies

/-- Theorem stating the total number of Jina's mascots -/
theorem jinas_mascots :
  let teddies : ℕ := 5
  let bunnies : ℕ := 3 * teddies
  let koalas : ℕ := 1
  let additional_teddies : ℕ := 2 * bunnies
  total_mascots teddies bunnies koalas additional_teddies = 51 := by
  sorry

end jinas_mascots_l2245_224516


namespace x_minus_y_value_l2245_224571

theorem x_minus_y_value (x y : ℝ) (h1 : |x| = 2) (h2 : y^2 = 9) (h3 : x + y < 0) :
  x - y = 1 ∨ x - y = 5 := by
  sorry

end x_minus_y_value_l2245_224571


namespace prob_event_l2245_224566

/-- Represents a standard deck of cards -/
structure Deck :=
  (total : Nat)
  (queens : Nat)
  (jacks : Nat)
  (red : Nat)

/-- Calculates the probability of drawing two queens -/
def prob_two_queens (d : Deck) : Rat :=
  (d.queens * (d.queens - 1)) / (d.total * (d.total - 1))

/-- Calculates the probability of drawing at least one jack -/
def prob_at_least_one_jack (d : Deck) : Rat :=
  1 - (d.total - d.jacks) * (d.total - d.jacks - 1) / (d.total * (d.total - 1))

/-- Calculates the probability of drawing two red cards -/
def prob_two_red (d : Deck) : Rat :=
  (d.red * (d.red - 1)) / (d.total * (d.total - 1))

/-- Theorem stating the probability of the given event -/
theorem prob_event (d : Deck) (h1 : d.total = 52) (h2 : d.queens = 4) (h3 : d.jacks = 4) (h4 : d.red = 26) :
  prob_two_queens d + prob_at_least_one_jack d + prob_two_red d = 89 / 221 := by
  sorry

end prob_event_l2245_224566


namespace find_y_value_l2245_224546

theorem find_y_value (x y : ℚ) (h1 : x = 51) (h2 : x^3*y - 2*x^2*y + x*y = 63000) : y = 8/17 := by
  sorry

end find_y_value_l2245_224546


namespace kennedy_school_distance_l2245_224555

/-- Represents the fuel efficiency of Kennedy's car in miles per gallon -/
def fuel_efficiency : ℝ := 19

/-- Represents the initial amount of gas in Kennedy's car in gallons -/
def initial_gas : ℝ := 2

/-- Represents the distance to the softball park in miles -/
def distance_softball : ℝ := 6

/-- Represents the distance to the burger restaurant in miles -/
def distance_burger : ℝ := 2

/-- Represents the distance to her friend's house in miles -/
def distance_friend : ℝ := 4

/-- Represents the distance home in miles -/
def distance_home : ℝ := 11

/-- Theorem stating that Kennedy drove 15 miles to school -/
theorem kennedy_school_distance : 
  ∃ (distance_school : ℝ), 
    distance_school = fuel_efficiency * initial_gas - 
      (distance_softball + distance_burger + distance_friend + distance_home) ∧ 
    distance_school = 15 := by
  sorry

end kennedy_school_distance_l2245_224555


namespace stratified_sampling_most_reasonable_l2245_224537

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Systematic
  | Stratified
  | SimpleRandom

/-- Represents a population with plots -/
structure Population where
  totalPlots : ℕ
  sampleSize : ℕ
  highVariability : Bool

/-- Determines if a sampling method is reasonable given a population -/
def isReasonableSamplingMethod (p : Population) (m : SamplingMethod) : Prop :=
  p.highVariability → m = SamplingMethod.Stratified

/-- Theorem stating that stratified sampling is the most reasonable method
    for a population with high variability -/
theorem stratified_sampling_most_reasonable (p : Population) 
    (h1 : p.totalPlots = 200)
    (h2 : p.sampleSize = 20)
    (h3 : p.highVariability = true) :
    isReasonableSamplingMethod p SamplingMethod.Stratified :=
  sorry

#check stratified_sampling_most_reasonable

end stratified_sampling_most_reasonable_l2245_224537


namespace only_drug_effectiveness_suitable_l2245_224529

/-- Represents the suitability of an option for a sampling survey. -/
inductive Suitability
  | Suitable
  | NotSuitable

/-- Represents the different survey options. -/
inductive SurveyOption
  | DrugEffectiveness
  | ClassVision
  | EmployeeExamination
  | SatelliteInspection

/-- Determines the suitability of a survey option for sampling. -/
def suitabilityForSampling (option : SurveyOption) : Suitability :=
  match option with
  | SurveyOption.DrugEffectiveness => Suitability.Suitable
  | _ => Suitability.NotSuitable

/-- Theorem stating that only the drug effectiveness option is suitable for sampling. -/
theorem only_drug_effectiveness_suitable :
  ∀ (option : SurveyOption),
    suitabilityForSampling option = Suitability.Suitable ↔
    option = SurveyOption.DrugEffectiveness :=
by
  sorry

#check only_drug_effectiveness_suitable

end only_drug_effectiveness_suitable_l2245_224529


namespace allison_marbles_count_l2245_224564

theorem allison_marbles_count (albert angela allison : ℕ) 
  (h1 : albert = 3 * angela)
  (h2 : angela = allison + 8)
  (h3 : albert + allison = 136) :
  allison = 28 := by
sorry

end allison_marbles_count_l2245_224564


namespace final_concentration_l2245_224523

-- Define the volumes and concentrations
def volume1 : ℝ := 2
def concentration1 : ℝ := 0.4
def volume2 : ℝ := 3
def concentration2 : ℝ := 0.6

-- Define the total volume
def totalVolume : ℝ := volume1 + volume2

-- Define the total amount of acid
def totalAcid : ℝ := volume1 * concentration1 + volume2 * concentration2

-- Theorem: The final concentration is 52%
theorem final_concentration :
  totalAcid / totalVolume = 0.52 := by sorry

end final_concentration_l2245_224523


namespace P_in_second_quadrant_l2245_224525

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate. -/
def SecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

/-- The given point P with coordinates (-1, 2). -/
def P : ℝ × ℝ := (-1, 2)

/-- Theorem: The point P lies in the second quadrant. -/
theorem P_in_second_quadrant : SecondQuadrant P := by
  sorry

end P_in_second_quadrant_l2245_224525


namespace strawberry_vs_cabbage_l2245_224560

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Represents the result of cutting an isosceles right triangle -/
structure CutTriangle where
  original : Triangle
  cut1 : ℝ  -- Position of first cut (0 ≤ cut1 ≤ 1)
  cut2 : ℝ  -- Position of second cut (0 ≤ cut2 ≤ 1)

/-- Calculates the area of the rectangle formed by the cuts -/
def rectangleArea (ct : CutTriangle) : ℝ := sorry

/-- Calculates the sum of areas of the two smaller triangles formed by the cuts -/
def smallTrianglesArea (ct : CutTriangle) : ℝ := sorry

/-- Theorem: The area of the rectangle is always less than or equal to 
    the sum of the areas of the two smaller triangles -/
theorem strawberry_vs_cabbage (ct : CutTriangle) : 
  rectangleArea ct ≤ smallTrianglesArea ct := by
  sorry

end strawberry_vs_cabbage_l2245_224560


namespace airline_problem_l2245_224590

/-- The number of airplanes owned by an airline company -/
def num_airplanes (rows_per_plane : ℕ) (seats_per_row : ℕ) (flights_per_day : ℕ) (total_passengers : ℕ) : ℕ :=
  total_passengers / (rows_per_plane * seats_per_row * flights_per_day)

/-- Theorem stating the number of airplanes owned by the company -/
theorem airline_problem :
  num_airplanes 20 7 2 1400 = 5 := by
  sorry

#eval num_airplanes 20 7 2 1400

end airline_problem_l2245_224590


namespace nested_fraction_evaluation_l2245_224552

theorem nested_fraction_evaluation :
  2 + 1 / (2 + 1 / (2 + 1 / 3)) = 41 / 17 := by
  sorry

end nested_fraction_evaluation_l2245_224552


namespace best_of_three_max_value_l2245_224521

/-- The maximum value of 8q - 9p in a best-of-three table tennis match -/
theorem best_of_three_max_value (p : ℝ) (q : ℝ) 
  (h1 : 0 < p) (h2 : p < 1) (h3 : q = 3 * p^2 - 2 * p^3) : 
  ∃ (max_val : ℝ), ∀ (p' : ℝ) (q' : ℝ), 
    0 < p' → p' < 1 → q' = 3 * p'^2 - 2 * p'^3 → 
    8 * q' - 9 * p' ≤ max_val ∧ max_val = 0 := by
  sorry

end best_of_three_max_value_l2245_224521


namespace range_of_decreasing_function_l2245_224509

/-- A decreasing function on the real line. -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The range of a function. -/
def Range (f : ℝ → ℝ) : Set ℝ :=
  {y | ∃ x, f x = y}

/-- Theorem: For a decreasing function on the real line, 
    the range of values for a is (0,2]. -/
theorem range_of_decreasing_function (f : ℝ → ℝ) 
  (h : DecreasingFunction f) : 
  Range f = Set.Ioo 0 2 := by
  sorry

end range_of_decreasing_function_l2245_224509


namespace train_speed_l2245_224515

-- Define the length of the train in meters
def train_length : ℝ := 90

-- Define the time taken to cross the pole in seconds
def crossing_time : ℝ := 9

-- Define the conversion factor from m/s to km/hr
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem train_speed : 
  (train_length / crossing_time) * conversion_factor = 36 := by
  sorry

end train_speed_l2245_224515


namespace product_mod_thirteen_l2245_224522

theorem product_mod_thirteen : (1501 * 1502 * 1503 * 1504 * 1505) % 13 = 5 := by
  sorry

end product_mod_thirteen_l2245_224522


namespace closest_to_370_l2245_224570

def calculation : ℝ := 3.1 * 9.1 * (5.92 + 4.08) + 100

def options : List ℝ := [300, 350, 370, 400, 430]

theorem closest_to_370 : 
  ∀ x ∈ options, |calculation - 370| ≤ |calculation - x| :=
by sorry

end closest_to_370_l2245_224570


namespace max_consecutive_special_is_correct_l2245_224579

/-- A number is special if it's a 20-digit number that cannot be represented
    as a product of a 10-digit number and an 11-digit number. -/
def IsSpecial (n : ℕ) : Prop :=
  10^19 ≤ n ∧ n < 10^20 ∧
  ∀ a b : ℕ, 10^9 ≤ a ∧ a < 10^10 → 10^10 ≤ b ∧ b < 10^11 → n ≠ a * b

/-- The maximum quantity of consecutive special numbers -/
def MaxConsecutiveSpecial : ℕ := 10^9 - 1

/-- Theorem stating that MaxConsecutiveSpecial is indeed the maximum
    quantity of consecutive special numbers -/
theorem max_consecutive_special_is_correct :
  (∀ k : ℕ, k < MaxConsecutiveSpecial →
    ∀ i : ℕ, i < k → IsSpecial (10^19 + i + 1)) ∧
  (∀ k : ℕ, k > MaxConsecutiveSpecial →
    ∃ i j : ℕ, i < j ∧ j - i = k ∧ ¬IsSpecial j) :=
sorry

end max_consecutive_special_is_correct_l2245_224579


namespace solution_set_g_range_of_a_l2245_224563

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := |x| + 2 * |x + 2 - a|

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := g a (x - 2)

-- Part 1: Solution set for g(x) ≤ 4 when a = 3
theorem solution_set_g (x : ℝ) :
  g 3 x ≤ 4 ↔ -2/3 ≤ x ∧ x ≤ 2 := by sorry

-- Part 2: Range of a such that f(x) ≥ 1 for all x ∈ ℝ
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 1) ↔ a ≤ 1 ∨ a ≥ 3 := by sorry

end solution_set_g_range_of_a_l2245_224563


namespace event_A_is_certain_l2245_224527

/-- The set of card labels -/
def card_labels : Finset ℕ := {1, 2, 3, 4, 5}

/-- The event "The label is less than 6" -/
def event_A (n : ℕ) : Prop := n < 6

/-- Theorem: Event A is a certain event -/
theorem event_A_is_certain : ∀ n ∈ card_labels, event_A n := by
  sorry

end event_A_is_certain_l2245_224527


namespace certain_number_addition_l2245_224584

theorem certain_number_addition (x : ℤ) : x + 36 = 71 → x + 10 = 45 := by
  sorry

end certain_number_addition_l2245_224584


namespace two_players_game_count_l2245_224542

/-- Represents the number of players in the league -/
def totalPlayers : ℕ := 12

/-- Represents the number of players in each game -/
def playersPerGame : ℕ := 4

/-- Calculates the number of games two specific players play together -/
def gamesPlayedTogether : ℕ :=
  (totalPlayers.choose playersPerGame) / (totalPlayers - 1) * (playersPerGame - 1) / playersPerGame

/-- Theorem stating that two specific players play together in 45 games -/
theorem two_players_game_count :
  gamesPlayedTogether = 45 := by sorry

end two_players_game_count_l2245_224542


namespace max_goats_from_coconuts_l2245_224556

/-- Represents the trading rates and initial coconut count --/
structure TradingRates :=
  (coconuts_per_crab : ℝ)
  (crabs_per_fish : ℝ)
  (fish_per_goat : ℝ)
  (initial_coconuts : ℕ)

/-- Calculates the maximum number of whole goats obtainable --/
def max_goats (rates : TradingRates) : ℕ :=
  sorry

/-- The theorem stating that given the specific trading rates and 1000 coconuts, 
    Max can obtain 33 goats --/
theorem max_goats_from_coconuts :
  let rates := TradingRates.mk 3.5 (6.25 / 5.5) 7.5 1000
  max_goats rates = 33 := by sorry

end max_goats_from_coconuts_l2245_224556


namespace probability_of_at_least_three_successes_l2245_224504

def probability_of_success : ℚ := 4/5

def number_of_trials : ℕ := 4

def at_least_successes : ℕ := 3

theorem probability_of_at_least_three_successes :
  (Finset.sum (Finset.range (number_of_trials - at_least_successes + 1))
    (fun k => Nat.choose number_of_trials (number_of_trials - k) *
      probability_of_success ^ (number_of_trials - k) *
      (1 - probability_of_success) ^ k)) = 512/625 := by
  sorry

end probability_of_at_least_three_successes_l2245_224504


namespace base4_division_theorem_l2245_224592

/-- Converts a number from base 4 to base 10 -/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 -/
def base10ToBase4 (n : ℕ) : ℕ := sorry

theorem base4_division_theorem :
  let dividend := 2313
  let divisor := 13
  let quotient := 122
  base10ToBase4 (base4ToBase10 dividend / base4ToBase10 divisor) = quotient := by
  sorry

end base4_division_theorem_l2245_224592


namespace batting_average_is_62_l2245_224534

/-- Calculates the batting average given the total innings, highest score, score difference, and average excluding extremes. -/
def battingAverage (totalInnings : ℕ) (highestScore : ℕ) (scoreDifference : ℕ) (averageExcludingExtremes : ℕ) : ℚ :=
  let lowestScore := highestScore - scoreDifference
  let totalScoreExcludingExtremes := (totalInnings - 2) * averageExcludingExtremes
  let totalScore := totalScoreExcludingExtremes + highestScore + lowestScore
  totalScore / totalInnings

/-- Theorem stating that under the given conditions, the batting average is 62 runs. -/
theorem batting_average_is_62 :
  battingAverage 46 225 150 58 = 62 := by sorry

end batting_average_is_62_l2245_224534


namespace valid_numbers_l2245_224577

def is_valid_number (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : Fin 10) (m n : ℕ),
    n = m + 10^k * a.val + 10^(k+1) * n ∧
    m < 10^k ∧
    m + 10^k * n = (m + 10^k * a.val + 10^(k+1) * n) / 6 ∧
    n % 10 ≠ 0

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {12, 24, 36, 48, 108} :=
sorry

end valid_numbers_l2245_224577


namespace circle_radius_through_triangle_vertices_l2245_224513

theorem circle_radius_through_triangle_vertices (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) :
  let r := (max a (max b c)) / 2
  r = 17 / 2 := by
  sorry

end circle_radius_through_triangle_vertices_l2245_224513


namespace quadratic_equation_solution_l2245_224597

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 4*x + 3
  (f 1 = 0 ∧ f 3 = 0) ∧ 
  (∀ x : ℝ, f x = 0 → x = 1 ∨ x = 3) :=
by sorry

end quadratic_equation_solution_l2245_224597


namespace eighteenth_permutation_l2245_224583

def FourDigitPermutation : Type := Fin 4 → Fin 10

def isValidPermutation (p : FourDigitPermutation) : Prop :=
  (p 0 = 1 ∨ p 0 = 2 ∨ p 0 = 5 ∨ p 0 = 6) ∧
  (p 1 = 1 ∨ p 1 = 2 ∨ p 1 = 5 ∨ p 1 = 6) ∧
  (p 2 = 1 ∨ p 2 = 2 ∨ p 2 = 5 ∨ p 2 = 6) ∧
  (p 3 = 1 ∨ p 3 = 2 ∨ p 3 = 5 ∨ p 3 = 6) ∧
  (p 0 ≠ p 1) ∧ (p 0 ≠ p 2) ∧ (p 0 ≠ p 3) ∧
  (p 1 ≠ p 2) ∧ (p 1 ≠ p 3) ∧ (p 2 ≠ p 3)

def toInteger (p : FourDigitPermutation) : ℕ :=
  1000 * (p 0).val + 100 * (p 1).val + 10 * (p 2).val + (p 3).val

def isOrdered (p q : FourDigitPermutation) : Prop :=
  toInteger p ≤ toInteger q

theorem eighteenth_permutation :
  ∃ (perms : List FourDigitPermutation),
    (∀ p ∈ perms, isValidPermutation p) ∧
    (perms.length = 24) ∧
    (∀ i j, i < j → isOrdered (perms.get ⟨i, by sorry⟩) (perms.get ⟨j, by sorry⟩)) ∧
    (toInteger (perms.get ⟨17, by sorry⟩) = 5621) :=
  sorry

end eighteenth_permutation_l2245_224583


namespace quadratic_two_distinct_roots_l2245_224559

theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   (1 - m) * x₁^2 + 2 * x₁ - 2 = 0 ∧ 
   (1 - m) * x₂^2 + 2 * x₂ - 2 = 0) ↔ 
  (m < 3/2 ∧ m ≠ 1) :=
sorry

end quadratic_two_distinct_roots_l2245_224559


namespace apple_pear_worth_l2245_224576

-- Define the worth of apples in terms of pears
def apple_worth (x : ℚ) : Prop := (3/4 : ℚ) * 16 * x = 6

-- Theorem to prove
theorem apple_pear_worth (x : ℚ) (h : apple_worth x) : (1/3 : ℚ) * 9 * x = (3/2 : ℚ) :=
by sorry

end apple_pear_worth_l2245_224576


namespace calculation_proof_l2245_224536

theorem calculation_proof : (-1)^3 - 8 / (-2) + 4 * |(-5)| = 23 := by
  sorry

end calculation_proof_l2245_224536


namespace root_sum_cubes_l2245_224572

theorem root_sum_cubes (a b c d : ℝ) : 
  (3 * a^4 + 6 * a^3 + 1002 * a^2 + 2005 * a + 4010 = 0) →
  (3 * b^4 + 6 * b^3 + 1002 * b^2 + 2005 * b + 4010 = 0) →
  (3 * c^4 + 6 * c^3 + 1002 * c^2 + 2005 * c + 4010 = 0) →
  (3 * d^4 + 6 * d^3 + 1002 * d^2 + 2005 * d + 4010 = 0) →
  (a + b)^3 + (b + c)^3 + (c + d)^3 + (d + a)^3 = 9362 :=
by sorry

end root_sum_cubes_l2245_224572


namespace range_of_a_for_inequality_l2245_224526

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → Real.log x - a * (1 - 1/x) ≥ 0) ↔ a ≤ 1 := by
  sorry

end range_of_a_for_inequality_l2245_224526


namespace cat_and_mouse_positions_after_359_moves_l2245_224524

/-- Represents the possible positions of the cat -/
inductive CatPosition
  | TopLeft
  | TopRight
  | BottomRight
  | BottomLeft

/-- Represents the possible positions of the mouse -/
inductive MousePosition
  | TopMiddle
  | TopRight
  | RightMiddle
  | BottomRight
  | BottomMiddle
  | BottomLeft
  | LeftMiddle
  | TopLeft

/-- Calculate the position of the cat after a given number of moves -/
def catPositionAfterMoves (moves : ℕ) : CatPosition :=
  match moves % 4 with
  | 0 => CatPosition.BottomLeft
  | 1 => CatPosition.TopLeft
  | 2 => CatPosition.TopRight
  | _ => CatPosition.BottomRight

/-- Calculate the position of the mouse after a given number of moves -/
def mousePositionAfterMoves (moves : ℕ) : MousePosition :=
  match moves % 8 with
  | 0 => MousePosition.TopLeft
  | 1 => MousePosition.TopMiddle
  | 2 => MousePosition.TopRight
  | 3 => MousePosition.RightMiddle
  | 4 => MousePosition.BottomRight
  | 5 => MousePosition.BottomMiddle
  | 6 => MousePosition.BottomLeft
  | _ => MousePosition.LeftMiddle

theorem cat_and_mouse_positions_after_359_moves :
  (catPositionAfterMoves 359 = CatPosition.BottomRight) ∧
  (mousePositionAfterMoves 359 = MousePosition.LeftMiddle) := by
  sorry


end cat_and_mouse_positions_after_359_moves_l2245_224524


namespace inequality_and_equality_condition_l2245_224517

theorem inequality_and_equality_condition (x : ℝ) (hx : x ≥ 0) :
  1 + x^2 + x^6 + x^8 ≥ 4 * x^4 ∧
  (1 + x^2 + x^6 + x^8 = 4 * x^4 ↔ x = 1) :=
by sorry

end inequality_and_equality_condition_l2245_224517


namespace strawberry_picking_problem_l2245_224547

/-- Calculates the number of pounds of strawberries picked given the total paid, 
    number of pickers, entrance fee per person, and price per pound of strawberries -/
def strawberries_picked (total_paid : ℚ) (num_pickers : ℕ) (entrance_fee : ℚ) (price_per_pound : ℚ) : ℚ :=
  (total_paid + num_pickers * entrance_fee) / price_per_pound

/-- Theorem stating that under the given conditions, the number of pounds of strawberries picked is 7 -/
theorem strawberry_picking_problem :
  let total_paid : ℚ := 128
  let num_pickers : ℕ := 3
  let entrance_fee : ℚ := 4
  let price_per_pound : ℚ := 20
  strawberries_picked total_paid num_pickers entrance_fee price_per_pound = 7 := by
  sorry


end strawberry_picking_problem_l2245_224547


namespace midpoint_trajectory_l2245_224528

/-- The equation of the trajectory of the midpoint of a line segment between a point on a unit circle and a fixed point -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ (px py : ℝ), px^2 + py^2 = 1 ∧ x = (px + 3) / 2 ∧ y = py / 2) ↔ 
  (2*x - 3)^2 + 4*y^2 = 1 :=
by sorry

end midpoint_trajectory_l2245_224528


namespace intersection_M_N_l2245_224545

def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

theorem intersection_M_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by sorry

end intersection_M_N_l2245_224545


namespace total_clothes_washed_l2245_224585

/-- The total number of clothes washed by Cally, Danny, and Emily -/
theorem total_clothes_washed (
  cally_white_shirts cally_colored_shirts cally_shorts cally_pants cally_jackets : ℕ)
  (danny_white_shirts danny_colored_shirts danny_shorts danny_pants danny_jackets : ℕ)
  (emily_white_shirts emily_colored_shirts emily_shorts emily_pants emily_jackets : ℕ)
  (cally_danny_socks emily_danny_socks : ℕ)
  (h1 : cally_white_shirts = 10)
  (h2 : cally_colored_shirts = 5)
  (h3 : cally_shorts = 7)
  (h4 : cally_pants = 6)
  (h5 : cally_jackets = 3)
  (h6 : danny_white_shirts = 6)
  (h7 : danny_colored_shirts = 8)
  (h8 : danny_shorts = 10)
  (h9 : danny_pants = 6)
  (h10 : danny_jackets = 4)
  (h11 : emily_white_shirts = 8)
  (h12 : emily_colored_shirts = 6)
  (h13 : emily_shorts = 9)
  (h14 : emily_pants = 5)
  (h15 : emily_jackets = 2)
  (h16 : cally_danny_socks = 3)
  (h17 : emily_danny_socks = 2) :
  cally_white_shirts + cally_colored_shirts + cally_shorts + cally_pants + cally_jackets +
  danny_white_shirts + danny_colored_shirts + danny_shorts + danny_pants + danny_jackets +
  emily_white_shirts + emily_colored_shirts + emily_shorts + emily_pants + emily_jackets +
  cally_danny_socks + emily_danny_socks = 100 := by
  sorry

end total_clothes_washed_l2245_224585


namespace max_perimeter_of_rectangle_with_area_36_exists_rectangle_with_area_36_and_perimeter_74_l2245_224578

-- Define a rectangle with integer side lengths
structure Rectangle where
  length : ℕ
  width : ℕ

-- Define the area of a rectangle
def area (r : Rectangle) : ℕ := r.length * r.width

-- Define the perimeter of a rectangle
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

-- Theorem: The maximum perimeter of a rectangle with integer side lengths and area 36 is 74
theorem max_perimeter_of_rectangle_with_area_36 :
  ∀ r : Rectangle, area r = 36 → perimeter r ≤ 74 :=
by
  sorry

-- Theorem: There exists a rectangle with integer side lengths, area 36, and perimeter 74
theorem exists_rectangle_with_area_36_and_perimeter_74 :
  ∃ r : Rectangle, area r = 36 ∧ perimeter r = 74 :=
by
  sorry

end max_perimeter_of_rectangle_with_area_36_exists_rectangle_with_area_36_and_perimeter_74_l2245_224578


namespace insertion_possible_l2245_224589

/-- Represents a natural number with exactly 2007 digits -/
def Number2007 := { n : ℕ | 10^2006 ≤ n ∧ n < 10^2007 }

/-- Represents the operation of removing 7 digits from a number -/
def remove_seven_digits (n : Number2007) : ℕ := sorry

/-- Represents the operation of inserting 7 digits into a number -/
def insert_seven_digits (n : Number2007) : Number2007 := sorry

/-- The main theorem -/
theorem insertion_possible (a b : Number2007) :
  (∃ (c : ℕ), remove_seven_digits a = c ∧ remove_seven_digits b = c) →
  (∃ (d : Number2007), ∃ (f g : Number2007 → Number2007),
    f a = d ∧ g b = d ∧ 
    (∀ x : Number2007, ∃ y, insert_seven_digits x = f x ∧ insert_seven_digits y = g x)) :=
sorry

end insertion_possible_l2245_224589


namespace expression_equals_19_96_l2245_224508

theorem expression_equals_19_96 : 
  (7 * (19 / 2015) * (6 * (19 / 2016)) - 13 * (1996 / 2015) * (2 * (1997 / 2016)) - 9 * (19 / 2015)) = 19 / 96 := by
  sorry

end expression_equals_19_96_l2245_224508


namespace dihedral_angle_relation_l2245_224574

/-- Regular quadrilateral prism -/
structure RegularQuadPrism where
  -- We don't need to define the specific geometry, just the existence of the prism
  prism : Unit

/-- Dihedral angle between lateral face and base -/
def lateral_base_angle (p : RegularQuadPrism) : ℝ :=
  sorry

/-- Dihedral angle between adjacent lateral faces -/
def adjacent_lateral_angle (p : RegularQuadPrism) : ℝ :=
  sorry

/-- Theorem stating the relationship between dihedral angles in a regular quadrilateral prism -/
theorem dihedral_angle_relation (p : RegularQuadPrism) :
  Real.cos (adjacent_lateral_angle p) = -(Real.cos (lateral_base_angle p))^2 := by
  sorry

end dihedral_angle_relation_l2245_224574


namespace people_in_line_l2245_224535

theorem people_in_line (initial_people total_people : ℕ) 
  (h1 : initial_people = 61)
  (h2 : total_people = 83)
  (h3 : total_people > initial_people) :
  total_people - initial_people = 22 := by
sorry

end people_in_line_l2245_224535


namespace watch_cost_price_l2245_224595

theorem watch_cost_price (loss_percentage : ℝ) (gain_percentage : ℝ) (price_difference : ℝ) : 
  loss_percentage = 0.15 →
  gain_percentage = 0.10 →
  price_difference = 450 →
  ∃ (cost_price : ℝ),
    cost_price * (1 - loss_percentage) + price_difference = cost_price * (1 + gain_percentage) ∧
    cost_price = 1800 :=
by sorry

end watch_cost_price_l2245_224595


namespace base8_642_equals_base10_418_l2245_224550

/-- Converts a base-8 number to base-10 -/
def base8_to_base10 (x : ℕ) : ℕ :=
  let d₂ := x / 100
  let d₁ := (x / 10) % 10
  let d₀ := x % 10
  d₂ * 8^2 + d₁ * 8^1 + d₀ * 8^0

theorem base8_642_equals_base10_418 : base8_to_base10 642 = 418 := by
  sorry

end base8_642_equals_base10_418_l2245_224550


namespace cost_of_horse_cost_of_horse_proof_l2245_224551

/-- The cost of a horse given the conditions of Albert's purchase and sale -/
theorem cost_of_horse : ℝ :=
  let total_cost : ℝ := 13400
  let total_profit : ℝ := 1880
  let num_horses : ℕ := 4
  let num_cows : ℕ := 9
  let horse_profit_rate : ℝ := 0.1
  let cow_profit_rate : ℝ := 0.2

  2000

theorem cost_of_horse_proof (total_cost : ℝ) (total_profit : ℝ) 
  (num_horses num_cows : ℕ) (horse_profit_rate cow_profit_rate : ℝ) :
  total_cost = 13400 →
  total_profit = 1880 →
  num_horses = 4 →
  num_cows = 9 →
  horse_profit_rate = 0.1 →
  cow_profit_rate = 0.2 →
  ∃ (horse_cost cow_cost : ℝ),
    num_horses * horse_cost + num_cows * cow_cost = total_cost ∧
    num_horses * horse_cost * horse_profit_rate + num_cows * cow_cost * cow_profit_rate = total_profit ∧
    horse_cost = 2000 :=
by
  sorry

end cost_of_horse_cost_of_horse_proof_l2245_224551


namespace geometric_sequence_formula_l2245_224575

theorem geometric_sequence_formula (a : ℕ → ℝ) (q : ℝ) (h1 : q < 1) 
  (h2 : a 2 + a 4 = 5/8) (h3 : a 3 = 1/4) 
  (h4 : ∀ n : ℕ, a (n + 1) = q * a n) : 
  ∀ n : ℕ, a n = (1/2)^(n - 1) := by
  sorry

end geometric_sequence_formula_l2245_224575


namespace jessicas_allowance_l2245_224544

/-- Jessica's weekly allowance problem -/
theorem jessicas_allowance (allowance : ℝ) : 
  (allowance / 2 + 6 = 11) → allowance = 10 := by
  sorry

end jessicas_allowance_l2245_224544


namespace andy_max_cookies_l2245_224518

/-- The maximum number of cookies Andy can eat given the conditions -/
def max_cookies_andy (total : ℕ) (bella_ratio : ℕ) : ℕ :=
  total / (bella_ratio + 1)

/-- Proof that Andy's maximum cookie consumption is correct -/
theorem andy_max_cookies :
  let total := 36
  let bella_ratio := 2
  let andy_cookies := max_cookies_andy total bella_ratio
  andy_cookies = 12 ∧
  andy_cookies + bella_ratio * andy_cookies = total ∧
  ∀ x : ℕ, x > andy_cookies → x + bella_ratio * x > total :=
by sorry

#eval max_cookies_andy 36 2  -- Should output 12

end andy_max_cookies_l2245_224518


namespace sqrt_two_times_sqrt_six_l2245_224520

theorem sqrt_two_times_sqrt_six : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3 := by
  sorry

end sqrt_two_times_sqrt_six_l2245_224520


namespace unique_three_digit_number_divisibility_l2245_224567

theorem unique_three_digit_number_divisibility : ∃! a : ℕ, 
  100 ≤ a ∧ a < 1000 ∧ 
  (∃ k : ℕ, 504000 + a = 693 * k) :=
by
  sorry

end unique_three_digit_number_divisibility_l2245_224567


namespace equal_angle_locus_for_given_flagpoles_l2245_224588

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a flagpole -/
structure Flagpole where
  base : Point
  height : ℝ

/-- The locus of points with equal angles of elevation to two flagpoles -/
def equalAngleLocus (pole1 pole2 : Flagpole) : Set (Point) :=
  {p : Point | (p.x - 85/8)^2 + p.y^2 = (75/8)^2}

theorem equal_angle_locus_for_given_flagpoles :
  let pole1 : Flagpole := ⟨Point.mk (-5) 0, 5⟩
  let pole2 : Flagpole := ⟨Point.mk 5 0, 3⟩
  equalAngleLocus pole1 pole2 =
    {p : Point | (p.x - 85/8)^2 + p.y^2 = (75/8)^2} :=
by
  sorry

end equal_angle_locus_for_given_flagpoles_l2245_224588


namespace b_subscription_difference_l2245_224501

/-- Represents the subscription amounts and profit distribution for a business venture --/
structure BusinessVenture where
  total_subscription : ℕ
  total_profit : ℕ
  a_subscription : ℕ
  b_subscription : ℕ
  c_subscription : ℕ
  a_profit : ℕ

/-- The conditions of the business venture as described in the problem --/
def venture_conditions (v : BusinessVenture) : Prop :=
  v.total_subscription = 50000 ∧
  v.total_profit = 70000 ∧
  v.a_profit = 29400 ∧
  v.a_subscription = v.b_subscription + 4000 ∧
  v.b_subscription > v.c_subscription ∧
  v.a_subscription + v.b_subscription + v.c_subscription = v.total_subscription ∧
  v.a_profit * v.total_subscription = v.a_subscription * v.total_profit

/-- The theorem stating that B subscribed 5000 more than C --/
theorem b_subscription_difference (v : BusinessVenture) 
  (h : venture_conditions v) : v.b_subscription - v.c_subscription = 5000 := by
  sorry


end b_subscription_difference_l2245_224501


namespace hope_star_voting_l2245_224502

/-- The Hope Star finals voting problem -/
theorem hope_star_voting
  (total_votes : ℕ)
  (huanhuan_votes lele_votes yangyang_votes : ℕ)
  (h_total : total_votes = 200)
  (h_ratio1 : 3 * lele_votes = 2 * huanhuan_votes)
  (h_ratio2 : 6 * yangyang_votes = 5 * lele_votes)
  (h_sum : huanhuan_votes + lele_votes + yangyang_votes = total_votes) :
  huanhuan_votes = 90 ∧ lele_votes = 60 ∧ yangyang_votes = 50 := by
  sorry

#check hope_star_voting

end hope_star_voting_l2245_224502


namespace paint_can_display_space_l2245_224541

/-- Calculates the total number of cans in a triangular arrangement -/
def totalCans (n : ℕ) : ℕ := n * (n + 1) * 3 / 2

/-- Calculates the total space required for the cans -/
def totalSpace (n : ℕ) (spacePerCan : ℕ) : ℕ := 
  (n * (n + 1) * 3 / 2) * spacePerCan

theorem paint_can_display_space : 
  ∃ n : ℕ, totalCans n = 242 ∧ totalSpace n 50 = 3900 := by
  sorry

end paint_can_display_space_l2245_224541


namespace function_identity_l2245_224512

def is_positive_integer (n : ℕ) : Prop := n > 0

theorem function_identity 
  (f : ℕ → ℕ) 
  (h1 : ∀ n, is_positive_integer n → is_positive_integer (f n))
  (h2 : ∀ n, is_positive_integer n → f (n + 1) > f (f n)) :
  ∀ n, is_positive_integer n → f n = n :=
sorry

end function_identity_l2245_224512


namespace more_girls_than_boys_l2245_224511

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) 
  (h1 : total_students = 485)
  (h2 : boys = 208)
  (h3 : boys < total_students - boys) :
  total_students - boys - boys = 69 := by
  sorry

end more_girls_than_boys_l2245_224511


namespace parallel_condition_neither_sufficient_nor_necessary_l2245_224593

-- Define the types for lines and planes
variable (Line Plane : Type*)

-- Define the subset relation
variable (subset : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_condition_neither_sufficient_nor_necessary
  (l m : Line) (α : Plane) (h : subset m α) :
  ¬(∀ l m α, subset m α → (parallel_lines l m → parallel_line_plane l α)) ∧
  ¬(∀ l m α, subset m α → (parallel_line_plane l α → parallel_lines l m)) :=
sorry

end parallel_condition_neither_sufficient_nor_necessary_l2245_224593


namespace cubic_equation_natural_roots_l2245_224539

theorem cubic_equation_natural_roots :
  ∃! P : ℝ, ∀ x : ℕ,
    (5 * x^3 - 5 * (P + 1) * x^2 + (71 * P - 1) * x + 1 = 66 * P) →
    (∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      (5 * a^3 - 5 * (P + 1) * a^2 + (71 * P - 1) * a + 1 = 66 * P) ∧
      (5 * b^3 - 5 * (P + 1) * b^2 + (71 * P - 1) * b + 1 = 66 * P) ∧
      (5 * c^3 - 5 * (P + 1) * c^2 + (71 * P - 1) * c + 1 = 66 * P)) →
    P = 76 :=
by sorry

end cubic_equation_natural_roots_l2245_224539


namespace system_solution_ratio_l2245_224557

theorem system_solution_ratio (k : ℝ) (x y z : ℝ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
  x + k*y + 3*z = 0 →
  3*x + k*y - 2*z = 0 →
  2*x + 4*y - 3*z = 0 →
  x*z / (y^2) = 10 :=
by sorry

end system_solution_ratio_l2245_224557


namespace book_input_time_l2245_224591

/-- The original time to complete a book input task given certain conditions on computer count and time changes. -/
theorem book_input_time : ∃ (n : ℕ) (T : ℚ),
  T > 0 ∧
  n > 3 ∧
  (n : ℚ) * T = (n + 3 : ℚ) * (3/4 * T) ∧
  (n - 3 : ℚ) * (T + 5/6) = (n : ℚ) * T ∧
  T = 5/3 := by
  sorry

end book_input_time_l2245_224591


namespace no_solution_quadratic_inequality_l2245_224565

theorem no_solution_quadratic_inequality :
  ∀ x : ℝ, ¬(3 * x^2 + 9 * x ≤ -12) :=
by
  sorry

end no_solution_quadratic_inequality_l2245_224565


namespace area_ratio_of_squares_l2245_224506

/-- Given three square regions I, II, and III, where the perimeter of region I is 12 units,
    the perimeter of region II is 24 units, and the side length of region III is the sum of
    the side lengths of regions I and II, prove that the ratio of the area of region I to
    the area of region III is 1/9. -/
theorem area_ratio_of_squares (side_length_I side_length_II side_length_III : ℝ) :
  side_length_I * 4 = 12 →
  side_length_II * 4 = 24 →
  side_length_III = side_length_I + side_length_II →
  (side_length_I ^ 2) / (side_length_III ^ 2) = 1 / 9 := by
  sorry

end area_ratio_of_squares_l2245_224506


namespace complement_M_wrt_U_l2245_224554

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define the set M
def M : Set ℝ := {x | x > 1}

-- State the theorem
theorem complement_M_wrt_U : 
  {x ∈ U | x ∉ M} = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end complement_M_wrt_U_l2245_224554


namespace books_together_l2245_224558

/-- The number of books Tim and Sam have together -/
def total_books (tim_books sam_books : ℕ) : ℕ := tim_books + sam_books

/-- Theorem: Tim and Sam have 96 books together -/
theorem books_together : total_books 44 52 = 96 := by sorry

end books_together_l2245_224558


namespace perfect_square_binomial_l2245_224507

theorem perfect_square_binomial :
  ∃ a : ℝ, ∀ x : ℝ, x^2 + 120*x + 3600 = (x + a)^2 := by
sorry

end perfect_square_binomial_l2245_224507


namespace total_length_of_T_l2245_224586

-- Define the set T
def T : Set (ℝ × ℝ) := {p : ℝ × ℝ | ‖‖|p.1| - 3‖ - 2‖ + ‖‖|p.2| - 3‖ - 2‖ = 2}

-- Define the total length of lines in T
def total_length (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem total_length_of_T : total_length T = 128 * Real.sqrt 2 := by sorry

end total_length_of_T_l2245_224586


namespace vector_difference_sum_l2245_224503

theorem vector_difference_sum : 
  let v1 : Fin 2 → ℝ := ![5, -8]
  let v2 : Fin 2 → ℝ := ![2, 6]
  let v3 : Fin 2 → ℝ := ![-1, 4]
  let scalar : ℝ := 5
  v1 - scalar • v2 + v3 = ![-6, -34] := by
  sorry

end vector_difference_sum_l2245_224503


namespace angle_through_point_l2245_224500

theorem angle_through_point (θ : Real) :
  (∃ (k : ℤ), θ = 2 * k * Real.pi + 5 * Real.pi / 6) ↔
  (∃ (t : Real), t > 0 ∧ t * Real.cos θ = -Real.sqrt 3 / 2 ∧ t * Real.sin θ = 1 / 2) :=
by sorry

end angle_through_point_l2245_224500


namespace inscribed_square_area_l2245_224599

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 8 = 1

-- Define a square inscribed in the ellipse
def inscribed_square (s : ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse x y ∧ s = 2 * x ∧ s = 2 * y

-- Theorem statement
theorem inscribed_square_area :
  ∃ (s : ℝ), inscribed_square s ∧ s^2 = 32/3 :=
sorry

end inscribed_square_area_l2245_224599


namespace sum_of_four_numbers_l2245_224548

theorem sum_of_four_numbers : 2143 + 3412 + 4213 + 1324 = 11092 := by
  sorry

end sum_of_four_numbers_l2245_224548


namespace a_5_value_l2245_224531

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d

/-- The conditions of the problem -/
def problem_conditions (a : ℕ → ℚ) : Prop :=
  arithmetic_sequence a ∧ a 1 + a 5 - a 8 = 1 ∧ a 9 - a 2 = 5

theorem a_5_value (a : ℕ → ℚ) (h : problem_conditions a) : a 5 = 6 := by
  sorry

end a_5_value_l2245_224531


namespace largest_digit_sum_l2245_224587

theorem largest_digit_sum (a b c z : ℕ) : 
  (a < 10) → (b < 10) → (c < 10) → 
  (0 < z) → (z ≤ 12) → 
  (100 * a + 10 * b + c = 1000 / z) → 
  (a + b + c ≤ 8) :=
sorry

end largest_digit_sum_l2245_224587


namespace rainy_days_probability_exists_l2245_224514

theorem rainy_days_probability_exists :
  ∃ (n : ℕ), n > 0 ∧ 
    (Nat.choose n 3 : ℝ) * (1/2)^3 * (1/2)^(n-3) = 1/4 := by
  sorry

end rainy_days_probability_exists_l2245_224514


namespace min_value_m_plus_2n_l2245_224532

theorem min_value_m_plus_2n (m n : ℝ) (h : m - n^2 = 0) : 
  ∀ x y : ℝ, x - y^2 = 0 → m + 2*n ≤ x + 2*y :=
by sorry

end min_value_m_plus_2n_l2245_224532


namespace min_zeros_in_interval_l2245_224505

theorem min_zeros_in_interval (f : ℝ → ℝ) 
  (h0 : f 0 = 0)
  (h1 : ∀ x, f (9 + x) = f (9 - x))
  (h2 : ∀ x, f (x - 10) = f (-x - 10)) :
  ∃ n : ℕ, n = 107 ∧ 
    (∀ m : ℕ, (∃ S : Finset ℝ, S.card = m ∧ 
      (∀ x ∈ S, x ∈ Set.Icc 0 2014 ∧ f x = 0)) → m ≥ n) :=
sorry

end min_zeros_in_interval_l2245_224505
