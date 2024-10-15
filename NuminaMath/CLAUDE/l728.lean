import Mathlib

namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l728_72878

/-- Represents different sampling methods --/
inductive SamplingMethod
  | Lottery
  | RandomNumber
  | Systematic
  | Stratified

/-- Represents a population with two subgroups --/
structure Population where
  total_size : ℕ
  subgroup1_size : ℕ
  subgroup2_size : ℕ
  h_size_sum : subgroup1_size + subgroup2_size = total_size

/-- Represents the goal of the sampling --/
inductive SamplingGoal
  | UnderstandSubgroupDifferences

/-- The most appropriate sampling method given a population and a goal --/
def most_appropriate_sampling_method (pop : Population) (goal : SamplingGoal) : SamplingMethod :=
  sorry

/-- Theorem stating that stratified sampling is most appropriate for the given scenario --/
theorem stratified_sampling_most_appropriate 
  (pop : Population) 
  (h_equal_subgroups : pop.subgroup1_size = pop.subgroup2_size) 
  (goal : SamplingGoal) 
  (h_goal : goal = SamplingGoal.UnderstandSubgroupDifferences) :
  most_appropriate_sampling_method pop goal = SamplingMethod.Stratified :=
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l728_72878


namespace NUMINAMATH_CALUDE_promotion_difference_l728_72861

/-- Calculates the total cost of two pairs of shoes using Promotion A -/
def costPromotionA (price : ℝ) : ℝ :=
  price + price * 0.4

/-- Calculates the total cost of two pairs of shoes using Promotion B -/
def costPromotionB (price : ℝ) : ℝ :=
  price + (price - 15)

/-- Proves that the difference between Promotion B and Promotion A is $15 -/
theorem promotion_difference (shoe_price : ℝ) (h : shoe_price = 50) :
  costPromotionB shoe_price - costPromotionA shoe_price = 15 := by
  sorry

#eval costPromotionB 50 - costPromotionA 50

end NUMINAMATH_CALUDE_promotion_difference_l728_72861


namespace NUMINAMATH_CALUDE_sandy_bought_six_fish_l728_72871

/-- The number of fish Sandy bought -/
def fish_bought (initial : ℕ) (current : ℕ) : ℕ := current - initial

/-- Proof that Sandy bought 6 fish -/
theorem sandy_bought_six_fish :
  let initial_fish : ℕ := 26
  let current_fish : ℕ := 32
  fish_bought initial_fish current_fish = 6 := by
  sorry

end NUMINAMATH_CALUDE_sandy_bought_six_fish_l728_72871


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l728_72870

theorem modulo_eleven_residue : (305 + 7 * 44 + 9 * 176 + 6 * 18) % 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l728_72870


namespace NUMINAMATH_CALUDE_cone_height_l728_72838

/-- The height of a cone with volume 8192π cubic inches and a vertical cross-section vertex angle of 45 degrees is equal to the cube root of 24576 inches. -/
theorem cone_height (V : ℝ) (θ : ℝ) (h : V = 8192 * Real.pi) (angle : θ = 45) :
  ∃ (H : ℝ), H = (24576 : ℝ) ^ (1/3) ∧ V = (1/3) * Real.pi * H^3 := by
  sorry


end NUMINAMATH_CALUDE_cone_height_l728_72838


namespace NUMINAMATH_CALUDE_food_supply_duration_l728_72879

/-- Proves that given a food supply for 760 men that lasts for x days, 
    if after 2 days 1140 more men join and the food lasts for 8 more days, 
    then x = 20. -/
theorem food_supply_duration (x : ℝ) : 
  (760 * x = (760 + 1140) * 8) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_food_supply_duration_l728_72879


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l728_72889

theorem smallest_fraction_between (a₁ b₁ a₂ b₂ : ℕ) 
  (h₁ : a₁ < b₁) (h₂ : a₂ < b₂) 
  (h₃ : Nat.gcd a₁ b₁ = 1) (h₄ : Nat.gcd a₂ b₂ = 1)
  (h₅ : a₂ * b₁ - a₁ * b₂ = 1) :
  ∃ (n k : ℕ), 
    (∀ (n' k' : ℕ), a₁ * n' < b₁ * k' ∧ b₂ * k' < a₂ * n' → n ≤ n') ∧
    a₁ * n < b₁ * k ∧ b₂ * k < a₂ * n ∧
    n = b₁ + b₂ ∧ k = a₁ + a₂ := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l728_72889


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_3_and_6_l728_72849

theorem greatest_four_digit_divisible_by_3_and_6 : ∃ n : ℕ,
  n = 9996 ∧
  n ≥ 1000 ∧ n < 10000 ∧
  n % 3 = 0 ∧ n % 6 = 0 ∧
  ∀ m : ℕ, m > n → m < 10000 → (m % 3 ≠ 0 ∨ m % 6 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_3_and_6_l728_72849


namespace NUMINAMATH_CALUDE_twenty_first_term_of_ap_l728_72891

/-- The nth term of an arithmetic progression -/
def arithmeticProgressionTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem twenty_first_term_of_ap (a₁ d : ℝ) (h₁ : a₁ = 3) (h₂ : d = 5) :
  arithmeticProgressionTerm a₁ d 21 = 103 :=
by sorry

end NUMINAMATH_CALUDE_twenty_first_term_of_ap_l728_72891


namespace NUMINAMATH_CALUDE_card_collection_average_l728_72890

def card_count (k : ℕ) : ℕ := 2 * k - 1

def total_cards (n : ℕ) : ℕ := n^2

def sum_of_values (n : ℕ) : ℕ := (n * (n + 1) / 2)^2 - (n * (n + 1) * (2 * n + 1) / 6)

def average_value (n : ℕ) : ℚ := (sum_of_values n : ℚ) / (total_cards n : ℚ)

theorem card_collection_average (n : ℕ) :
  n > 0 ∧ average_value n = 100 → n = 10 :=
by sorry

end NUMINAMATH_CALUDE_card_collection_average_l728_72890


namespace NUMINAMATH_CALUDE_perfect_cubes_between_powers_of_three_l728_72869

theorem perfect_cubes_between_powers_of_three : 
  (Finset.filter (fun n : ℕ => 
    3^5 - 1 ≤ n^3 ∧ n^3 ≤ 3^15 + 1) 
    (Finset.range (Nat.floor (Real.rpow 3 5) + 1))).card = 18 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cubes_between_powers_of_three_l728_72869


namespace NUMINAMATH_CALUDE_triangle_area_l728_72853

theorem triangle_area : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (6, 1)
  let C : ℝ × ℝ := (10, 6)
  let v : ℝ × ℝ := (A.1 - C.1, A.2 - C.2)
  let w : ℝ × ℝ := (B.1 - C.1, B.2 - C.2)
  abs (v.1 * w.2 - v.2 * w.1) / 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l728_72853


namespace NUMINAMATH_CALUDE_book_distribution_theorem_l728_72847

def distribute_books (n : ℕ) : ℕ :=
  if n ≥ 2 then n - 1 else 0

theorem book_distribution_theorem :
  distribute_books 8 = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_book_distribution_theorem_l728_72847


namespace NUMINAMATH_CALUDE_bacteria_growth_l728_72800

theorem bacteria_growth (initial_count : ℕ) : 
  (initial_count * (4 ^ 15) = 4194304) → initial_count = 1 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_l728_72800


namespace NUMINAMATH_CALUDE_parabola_vertex_l728_72818

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 3 * (x - 1)^2 + 2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- Theorem: The vertex of the parabola y = 3(x-1)^2 + 2 is at the point (1, 2) -/
theorem parabola_vertex :
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧
  parabola (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l728_72818


namespace NUMINAMATH_CALUDE_train_length_l728_72855

/-- Given a train traveling at 45 km/hr, crossing a bridge of 240.03 meters in 30 seconds,
    the length of the train is 134.97 meters. -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 →
  bridge_length = 240.03 →
  crossing_time = 30 →
  (train_speed * 1000 / 3600 * crossing_time) - bridge_length = 134.97 := by
  sorry

#eval (45 * 1000 / 3600 * 30) - 240.03

end NUMINAMATH_CALUDE_train_length_l728_72855


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l728_72817

/-- A line with slope 1 passing through (0, a) is tangent to the circle x^2 + y^2 = 2 if and only if a = ±2 -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∃ (x y : ℝ), y = x + a ∧ x^2 + y^2 = 2 ∧ 
  ∀ (x' y' : ℝ), y' = x' + a → x'^2 + y'^2 ≥ 2) ↔ 
  (a = 2 ∨ a = -2) := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l728_72817


namespace NUMINAMATH_CALUDE_x_over_u_value_l728_72895

theorem x_over_u_value (u v w x : ℝ) 
  (h1 : u / v = 5)
  (h2 : w / v = 3)
  (h3 : w / x = 2 / 3) :
  x / u = 9 / 10 := by
sorry

end NUMINAMATH_CALUDE_x_over_u_value_l728_72895


namespace NUMINAMATH_CALUDE_transformation_eventually_repeats_l728_72805

/-- Represents a transformation step on a sequence of natural numbers -/
def transform (s : List ℕ) : List ℕ :=
  s.map (λ x => s.count x)

/-- Represents the sequence of transformations applied to an initial sequence -/
def transformation_sequence (initial : List ℕ) : ℕ → List ℕ
  | 0 => initial
  | n + 1 => transform (transformation_sequence initial n)

/-- The theorem stating that the transformation sequence will eventually repeat -/
theorem transformation_eventually_repeats (initial : List ℕ) :
  ∃ n : ℕ, transformation_sequence initial n = transformation_sequence initial (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_transformation_eventually_repeats_l728_72805


namespace NUMINAMATH_CALUDE_product_seven_consecutive_divisible_by_100_l728_72829

theorem product_seven_consecutive_divisible_by_100 (n : ℕ) : 
  100 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) :=
by sorry

end NUMINAMATH_CALUDE_product_seven_consecutive_divisible_by_100_l728_72829


namespace NUMINAMATH_CALUDE_seed_survival_rate_l728_72874

theorem seed_survival_rate 
  (germination_rate : ℝ) 
  (seedling_probability : ℝ) 
  (h1 : germination_rate = 0.9) 
  (h2 : seedling_probability = 0.81) : 
  ∃ p : ℝ, p = germination_rate ∧ p * germination_rate = seedling_probability :=
by
  sorry

end NUMINAMATH_CALUDE_seed_survival_rate_l728_72874


namespace NUMINAMATH_CALUDE_even_function_with_domain_l728_72893

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

theorem even_function_with_domain (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) →
  (∀ x, f a b x ≠ 0 → a - 1 ≤ x ∧ x ≤ 2 * a) →
  (∃ c d : ℝ, ∀ x, -2/3 ≤ x ∧ x ≤ 2/3 → f a b x = 1/3 * x^2 + 1) :=
sorry

end NUMINAMATH_CALUDE_even_function_with_domain_l728_72893


namespace NUMINAMATH_CALUDE_soda_survey_result_l728_72815

-- Define the total number of people surveyed
def total_surveyed : ℕ := 500

-- Define the central angle of the "Soda" sector in degrees
def soda_angle : ℕ := 198

-- Define the function to calculate the number of people who chose "Soda"
def soda_count : ℕ := (total_surveyed * soda_angle) / 360

-- Theorem statement
theorem soda_survey_result : soda_count = 275 := by
  sorry

end NUMINAMATH_CALUDE_soda_survey_result_l728_72815


namespace NUMINAMATH_CALUDE_basketball_score_proof_l728_72850

/-- Given two teams in a basketball game where:
  * The total points scored is 50
  * One team wins by a margin of 28 points
  Prove that the losing team scored 11 points -/
theorem basketball_score_proof (total_points winning_margin : ℕ) 
  (h1 : total_points = 50)
  (h2 : winning_margin = 28) :
  ∃ (winner_score loser_score : ℕ),
    winner_score + loser_score = total_points ∧
    winner_score - loser_score = winning_margin ∧
    loser_score = 11 := by
sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l728_72850


namespace NUMINAMATH_CALUDE_least_positive_linear_combination_l728_72810

theorem least_positive_linear_combination : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (∃ (x y : ℤ), 24 * x + 18 * y = m) → n ≤ m) ∧ 
  (∃ (x y : ℤ), 24 * x + 18 * y = n) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_linear_combination_l728_72810


namespace NUMINAMATH_CALUDE_sam_study_time_l728_72844

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The time Sam spends studying Science in minutes -/
def science_time : ℕ := 60

/-- The time Sam spends studying Math in minutes -/
def math_time : ℕ := 80

/-- The time Sam spends studying Literature in minutes -/
def literature_time : ℕ := 40

/-- The total time Sam spends studying in hours -/
def total_study_time : ℚ :=
  (science_time + math_time + literature_time : ℚ) / minutes_per_hour

theorem sam_study_time :
  total_study_time = 3 := by sorry

end NUMINAMATH_CALUDE_sam_study_time_l728_72844


namespace NUMINAMATH_CALUDE_octagon_diagonals_l728_72854

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l728_72854


namespace NUMINAMATH_CALUDE_no_solution_fractional_equation_l728_72863

theorem no_solution_fractional_equation :
  ∀ x : ℝ, (1 - x) / (x - 2) ≠ 1 / (2 - x) + 1 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_fractional_equation_l728_72863


namespace NUMINAMATH_CALUDE_tree_growth_theorem_l728_72839

/-- Represents the number of branches in Professor Fernando's tree after n weeks -/
def tree_branches : ℕ → ℕ
  | 0 => 0  -- No branches before the tree starts growing
  | 1 => 1  -- One branch in the first week
  | 2 => 1  -- Still one branch in the second week
  | n + 3 => tree_branches (n + 1) + tree_branches (n + 2)  -- Fibonacci recurrence for subsequent weeks

theorem tree_growth_theorem :
  (tree_branches 6 = 8) ∧ 
  (tree_branches 7 = 13) ∧ 
  (tree_branches 13 = 233) := by
sorry

#eval tree_branches 6  -- Expected: 8
#eval tree_branches 7  -- Expected: 13
#eval tree_branches 13  -- Expected: 233

end NUMINAMATH_CALUDE_tree_growth_theorem_l728_72839


namespace NUMINAMATH_CALUDE_weight_per_hour_is_correct_l728_72837

/-- Represents the types of coins Jim finds --/
inductive CoinType
| Gold
| Silver
| Bronze

/-- Represents a bag of coins --/
structure CoinBag where
  coinType : CoinType
  count : ℕ

def hours_spent : ℕ := 8

def coin_weight (ct : CoinType) : ℕ :=
  match ct with
  | CoinType.Gold => 10
  | CoinType.Silver => 5
  | CoinType.Bronze => 2

def treasure_chest : CoinBag := ⟨CoinType.Gold, 100⟩
def smaller_bags : List CoinBag := [⟨CoinType.Gold, 50⟩, ⟨CoinType.Gold, 50⟩]
def other_bags : List CoinBag := [⟨CoinType.Gold, 30⟩, ⟨CoinType.Gold, 20⟩, ⟨CoinType.Gold, 10⟩]
def silver_coins : CoinBag := ⟨CoinType.Silver, 30⟩
def bronze_coins : CoinBag := ⟨CoinType.Bronze, 50⟩

def all_bags : List CoinBag :=
  [treasure_chest] ++ smaller_bags ++ other_bags ++ [silver_coins, bronze_coins]

def total_weight (bags : List CoinBag) : ℕ :=
  bags.foldl (fun acc bag => acc + bag.count * coin_weight bag.coinType) 0

theorem weight_per_hour_is_correct :
  (total_weight all_bags : ℚ) / hours_spent = 356.25 := by sorry

end NUMINAMATH_CALUDE_weight_per_hour_is_correct_l728_72837


namespace NUMINAMATH_CALUDE_school_play_ticket_sales_l728_72896

/-- Calculates the total sales from school play tickets -/
def total_ticket_sales (student_price adult_price : ℕ) (student_tickets adult_tickets : ℕ) : ℕ :=
  student_price * student_tickets + adult_price * adult_tickets

/-- Theorem: The total sales from the school play tickets is $216 -/
theorem school_play_ticket_sales :
  total_ticket_sales 6 8 20 12 = 216 := by
  sorry

end NUMINAMATH_CALUDE_school_play_ticket_sales_l728_72896


namespace NUMINAMATH_CALUDE_small_boxes_packed_l728_72816

/-- Represents the number of feet of tape used for sealing each type of box --/
def seal_tape_large : ℕ := 4
def seal_tape_medium : ℕ := 2
def seal_tape_small : ℕ := 1

/-- Represents the number of feet of tape used for address label on each box --/
def label_tape : ℕ := 1

/-- Represents the number of large boxes packed --/
def num_large : ℕ := 2

/-- Represents the number of medium boxes packed --/
def num_medium : ℕ := 8

/-- Represents the total amount of tape used in feet --/
def total_tape : ℕ := 44

/-- Calculates the number of small boxes packed --/
def num_small : ℕ := 
  (total_tape - 
   (num_large * (seal_tape_large + label_tape) + 
    num_medium * (seal_tape_medium + label_tape))) / 
  (seal_tape_small + label_tape)

theorem small_boxes_packed : num_small = 5 := by
  sorry

end NUMINAMATH_CALUDE_small_boxes_packed_l728_72816


namespace NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l728_72877

/-- A triangle with consecutive even integer side lengths. -/
structure EvenTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  h_even : ∃ k : ℕ, a = 2*k ∧ b = 2*(k+1) ∧ c = 2*(k+2)
  h_triangle : a + b > c ∧ a + c > b ∧ b + c > a

/-- The perimeter of an EvenTriangle. -/
def perimeter (t : EvenTriangle) : ℕ := t.a + t.b + t.c

/-- The smallest possible perimeter of an EvenTriangle is 12. -/
theorem smallest_even_triangle_perimeter :
  ∃ t : EvenTriangle, perimeter t = 12 ∧ ∀ t' : EvenTriangle, perimeter t ≤ perimeter t' :=
sorry

end NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l728_72877


namespace NUMINAMATH_CALUDE_five_seventeenths_repetend_l728_72826

def decimal_repetend (n d : ℕ) (repetend : List ℕ) : Prop :=
  ∃ (k : ℕ), (n : ℚ) / d = (k : ℚ) / 10^(repetend.length) + 
    (List.sum (List.zipWith (λ (digit place) => (digit : ℚ) / 10^place) repetend 
    (List.range repetend.length))) / (10^(repetend.length) - 1)

theorem five_seventeenths_repetend :
  decimal_repetend 5 17 [2, 9, 4, 1, 1, 7, 6, 4, 7, 0, 5, 8, 8, 2, 3, 5] :=
sorry

end NUMINAMATH_CALUDE_five_seventeenths_repetend_l728_72826


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l728_72897

theorem parallelogram_base_length 
  (area : ℝ) 
  (altitude_base_ratio : ℝ) 
  (base_altitude_angle : ℝ) :
  area = 162 →
  altitude_base_ratio = 2 →
  base_altitude_angle = 60 * π / 180 →
  ∃ (base : ℝ), base = 9 ∧ area = base * (altitude_base_ratio * base) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l728_72897


namespace NUMINAMATH_CALUDE_revenue_decrease_l728_72809

theorem revenue_decrease (current_revenue : ℝ) (decrease_percentage : ℝ) (original_revenue : ℝ) : 
  current_revenue = 48.0 ∧ 
  decrease_percentage = 33.33333333333333 / 100 ∧
  current_revenue = original_revenue * (1 - decrease_percentage) →
  original_revenue = 72.0 := by
sorry

end NUMINAMATH_CALUDE_revenue_decrease_l728_72809


namespace NUMINAMATH_CALUDE_square_rectangle_overlap_ratio_l728_72866

theorem square_rectangle_overlap_ratio : 
  ∀ (s x y : ℝ),
  s > 0 → x > 0 → y > 0 →
  (0.25 * s^2 = 0.4 * x * y) →
  (y = s) →
  (x / y = 5 / 8) := by
sorry

end NUMINAMATH_CALUDE_square_rectangle_overlap_ratio_l728_72866


namespace NUMINAMATH_CALUDE_sally_orange_balloons_l728_72831

def initial_orange_balloons : ℕ := 9
def lost_orange_balloons : ℕ := 2

theorem sally_orange_balloons :
  initial_orange_balloons - lost_orange_balloons = 7 :=
by sorry

end NUMINAMATH_CALUDE_sally_orange_balloons_l728_72831


namespace NUMINAMATH_CALUDE_no_integer_solution_l728_72819

theorem no_integer_solution :
  ∀ (x y z : ℤ), x ≠ 0 → 2 * x^4 + 2 * x^2 * y^2 + y^4 ≠ z^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l728_72819


namespace NUMINAMATH_CALUDE_alternate_multiply_divide_result_l728_72833

def alternateMultiplyDivide (n : ℕ) (initial : ℕ) : ℚ :=
  match n with
  | 0 => initial
  | m + 1 => if m % 2 = 0
             then (alternateMultiplyDivide m initial) * 3
             else (alternateMultiplyDivide m initial) / 2

theorem alternate_multiply_divide_result :
  alternateMultiplyDivide 15 (9^6) = 3^20 / 2^7 := by
  sorry

end NUMINAMATH_CALUDE_alternate_multiply_divide_result_l728_72833


namespace NUMINAMATH_CALUDE_tip_percentage_is_twenty_percent_l728_72885

def lunch_cost : ℝ := 60.50
def total_spent : ℝ := 72.6

theorem tip_percentage_is_twenty_percent :
  (total_spent - lunch_cost) / lunch_cost * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tip_percentage_is_twenty_percent_l728_72885


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_cubes_l728_72865

theorem consecutive_integers_sum_of_cubes (n : ℕ) : 
  (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 = 8830 →
  (n - 1)^3 + n^3 + (n + 1)^3 + (n + 2)^3 = 52264 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_cubes_l728_72865


namespace NUMINAMATH_CALUDE_zoo_enclosure_claws_l728_72807

theorem zoo_enclosure_claws (num_wombats : ℕ) (num_rheas : ℕ) 
  (wombat_claws : ℕ) (rhea_claws : ℕ) : 
  num_wombats = 9 → 
  num_rheas = 3 → 
  wombat_claws = 4 → 
  rhea_claws = 1 → 
  num_wombats * wombat_claws + num_rheas * rhea_claws = 39 := by
  sorry

end NUMINAMATH_CALUDE_zoo_enclosure_claws_l728_72807


namespace NUMINAMATH_CALUDE_vector_norm_sum_l728_72862

theorem vector_norm_sum (a b c : EuclideanSpace ℝ (Fin 3)) 
  (h1 : ‖a‖ = 2) (h2 : ‖b‖ = 2) (h3 : ‖c‖ = 3) : 
  ‖a - 3 • b‖^2 + ‖b - 3 • c‖^2 + ‖c - 3 • a‖^2 = 170 := by
  sorry

end NUMINAMATH_CALUDE_vector_norm_sum_l728_72862


namespace NUMINAMATH_CALUDE_max_value_of_a_l728_72859

theorem max_value_of_a : ∃ (a_max : ℝ), a_max = 16175 ∧
  ∀ (a : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 →
    -2022 ≤ (a + 1) * x^2 - (a + 1) * x + 2022 ∧
    (a + 1) * x^2 - (a + 1) * x + 2022 ≤ 2022) →
  a ≤ a_max := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_a_l728_72859


namespace NUMINAMATH_CALUDE_fraction_power_five_l728_72828

theorem fraction_power_five : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_five_l728_72828


namespace NUMINAMATH_CALUDE_probability_ten_people_no_adjacent_standing_l728_72813

/-- Represents the number of valid arrangements for n people where no two adjacent people are standing --/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validArrangements (n + 1) + validArrangements n

/-- The probability of no two adjacent people standing in a circular arrangement of n people --/
def probabilityNoAdjacentStanding (n : ℕ) : ℚ :=
  validArrangements n / (2 ^ n : ℚ)

theorem probability_ten_people_no_adjacent_standing :
  probabilityNoAdjacentStanding 10 = 123 / 1024 := by
  sorry


end NUMINAMATH_CALUDE_probability_ten_people_no_adjacent_standing_l728_72813


namespace NUMINAMATH_CALUDE_monkey_count_l728_72882

/-- Given a group of monkeys that can eat 6 bananas in 6 minutes and 18 bananas in 18 minutes,
    prove that there are 6 monkeys in the group. -/
theorem monkey_count (eating_rate : ℕ → ℕ → ℕ) (monkey_count : ℕ) : 
  (eating_rate 6 6 = 6) →  -- 6 bananas in 6 minutes
  (eating_rate 18 18 = 18) →  -- 18 bananas in 18 minutes
  monkey_count = 6 :=
by sorry

end NUMINAMATH_CALUDE_monkey_count_l728_72882


namespace NUMINAMATH_CALUDE_least_eight_binary_digits_l728_72857

def binary_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

theorem least_eight_binary_digits : 
  ∀ k : ℕ, k > 0 → (binary_digits k ≥ 8 → k ≥ 128) ∧ binary_digits 128 = 8 :=
by sorry

end NUMINAMATH_CALUDE_least_eight_binary_digits_l728_72857


namespace NUMINAMATH_CALUDE_quadratic_root_value_l728_72827

theorem quadratic_root_value (c : ℚ) : 
  (∀ x : ℚ, (3/2 * x^2 + 13*x + c = 0) ↔ (x = (-13 + Real.sqrt 23)/3 ∨ x = (-13 - Real.sqrt 23)/3)) →
  c = 146/6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l728_72827


namespace NUMINAMATH_CALUDE_stamp_collection_difference_l728_72812

theorem stamp_collection_difference (kylie_stamps nelly_stamps : ℕ) : 
  kylie_stamps = 34 →
  nelly_stamps > kylie_stamps →
  kylie_stamps + nelly_stamps = 112 →
  nelly_stamps - kylie_stamps = 44 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_difference_l728_72812


namespace NUMINAMATH_CALUDE_parabola_c_value_l728_72894

/-- A parabola with equation x = ay² + by + c, vertex at (5, 3), and passing through (3, 5) has c = 1/2 -/
theorem parabola_c_value (a b c : ℝ) : 
  (∀ y : ℝ, 5 = a * 3^2 + b * 3 + c) →  -- vertex condition
  (∀ y : ℝ, 3 = a * 5^2 + b * 5 + c) →  -- point condition
  c = 1/2 := by sorry

end NUMINAMATH_CALUDE_parabola_c_value_l728_72894


namespace NUMINAMATH_CALUDE_exists_n_for_root_1000_l728_72820

theorem exists_n_for_root_1000 : ∃ n : ℕ, (1000 : ℝ) ^ (1 / n) < 1.001 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_for_root_1000_l728_72820


namespace NUMINAMATH_CALUDE_probability_of_shaded_triangle_l728_72811

/-- Given a diagram with 6 triangles, where 3 are shaded and all have equal selection probability, 
    the probability of selecting a shaded triangle is 1/2 -/
theorem probability_of_shaded_triangle (total_triangles : ℕ) (shaded_triangles : ℕ) :
  total_triangles = 6 →
  shaded_triangles = 3 →
  (shaded_triangles : ℚ) / (total_triangles : ℚ) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_shaded_triangle_l728_72811


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l728_72873

theorem cubic_roots_sum (a b c : ℝ) : 
  (10 * a^3 + 15 * a^2 + 2005 * a + 2010 = 0) →
  (10 * b^3 + 15 * b^2 + 2005 * b + 2010 = 0) →
  (10 * c^3 + 15 * c^2 + 2005 * c + 2010 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 907.125 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l728_72873


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l728_72880

/-- Represents a trapezoid -/
structure Trapezoid where
  long_base : ℝ
  short_base : ℝ
  midpoint_diagonal_length : ℝ

/-- 
Given a trapezoid where:
- The line joining the midpoints of the diagonals has length 5
- The longer base is 105
Then the shorter base must be 95
-/
theorem trapezoid_shorter_base 
  (t : Trapezoid) 
  (h1 : t.midpoint_diagonal_length = 5) 
  (h2 : t.long_base = 105) : 
  t.short_base = 95 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_shorter_base_l728_72880


namespace NUMINAMATH_CALUDE_root_product_plus_one_l728_72834

theorem root_product_plus_one (p q r : ℂ) : 
  p^3 - 15*p^2 + 10*p + 24 = 0 →
  q^3 - 15*q^2 + 10*q + 24 = 0 →
  r^3 - 15*r^2 + 10*r + 24 = 0 →
  (1+p)*(1+q)*(1+r) = 2 := by
sorry

end NUMINAMATH_CALUDE_root_product_plus_one_l728_72834


namespace NUMINAMATH_CALUDE_circle_radius_proof_l728_72822

theorem circle_radius_proof (r₁ r₂ : ℝ) : 
  r₂ = 2 →                             -- The smaller circle has a radius of 2 cm
  (π * r₁^2) = 4 * (π * r₂^2) →        -- The area of one circle is four times the area of the other
  r₁ = 4 :=                            -- The radius of the larger circle is 4 cm
by sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l728_72822


namespace NUMINAMATH_CALUDE_combined_salaries_l728_72840

/-- Given 5 individuals with an average salary of 8200 and one individual with a salary of 7000,
    prove that the sum of the other 4 individuals' salaries is 34000 -/
theorem combined_salaries (average_salary : ℕ) (num_individuals : ℕ) (d_salary : ℕ) :
  average_salary = 8200 →
  num_individuals = 5 →
  d_salary = 7000 →
  (average_salary * num_individuals) - d_salary = 34000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_l728_72840


namespace NUMINAMATH_CALUDE_complex_magnitude_l728_72867

theorem complex_magnitude (b : ℝ) : 
  let z : ℂ := (3 - b * Complex.I) / Complex.I
  (z.re = z.im) → Complex.abs z = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l728_72867


namespace NUMINAMATH_CALUDE_team_formation_proof_l728_72881

def number_of_teams (total_girls : ℕ) (total_boys : ℕ) (team_girls : ℕ) (team_boys : ℕ) (mandatory_girl : ℕ) : ℕ :=
  Nat.choose (total_girls - mandatory_girl) (team_girls - mandatory_girl) * Nat.choose total_boys team_boys

theorem team_formation_proof :
  let total_girls : ℕ := 5
  let total_boys : ℕ := 7
  let team_girls : ℕ := 2
  let team_boys : ℕ := 2
  let mandatory_girl : ℕ := 1
  number_of_teams total_girls total_boys team_girls team_boys mandatory_girl = 84 :=
by
  sorry

end NUMINAMATH_CALUDE_team_formation_proof_l728_72881


namespace NUMINAMATH_CALUDE_ninth_grade_students_l728_72835

theorem ninth_grade_students (S : ℕ) : 
  (S / 4 : ℚ) + (3 * S / 4 / 3 : ℚ) + 20 + 70 = S → S = 180 := by
  sorry

end NUMINAMATH_CALUDE_ninth_grade_students_l728_72835


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l728_72824

theorem quadratic_inequality_solution (a : ℝ) (x₁ x₂ : ℝ) : 
  a > 0 → 
  (∀ x, x^2 - 2*a*x - 8*a^2 < 0 ↔ x₁ < x ∧ x < x₂) → 
  x₂ + x₁ = 15 → 
  a = 15/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l728_72824


namespace NUMINAMATH_CALUDE_highest_numbered_street_l728_72806

/-- The length of Gretzky Street in meters -/
def street_length : ℕ := 5600

/-- The distance between intersecting streets in meters -/
def intersection_distance : ℕ := 350

/-- The number of non-numbered intersecting streets (Orr and Howe) -/
def non_numbered_streets : ℕ := 2

/-- Theorem stating the highest-numbered intersecting street -/
theorem highest_numbered_street :
  (street_length / intersection_distance) - non_numbered_streets = 14 := by
  sorry

end NUMINAMATH_CALUDE_highest_numbered_street_l728_72806


namespace NUMINAMATH_CALUDE_tan_fifteen_pi_fourths_l728_72845

theorem tan_fifteen_pi_fourths : Real.tan (15 * π / 4) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_fifteen_pi_fourths_l728_72845


namespace NUMINAMATH_CALUDE_ball_probability_l728_72864

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h_total : total = 100)
  (h_white : white = 50)
  (h_green : green = 30)
  (h_yellow : yellow = 10)
  (h_red : red = 7)
  (h_purple : purple = 3)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 9 / 10 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_l728_72864


namespace NUMINAMATH_CALUDE_radical_equation_solution_l728_72841

theorem radical_equation_solution :
  ∃! x : ℝ, x > 9 ∧ Real.sqrt (x - 6 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 9)) - 3 ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_radical_equation_solution_l728_72841


namespace NUMINAMATH_CALUDE_log_equation_solution_l728_72804

theorem log_equation_solution :
  ∃ x : ℝ, (Real.log x + 3 * Real.log 2 - 4 * Real.log 5 = 1) ∧ (x = 781.25) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l728_72804


namespace NUMINAMATH_CALUDE_percentage_difference_l728_72842

theorem percentage_difference : (0.9 * 40) - (0.8 * 30) = 12 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l728_72842


namespace NUMINAMATH_CALUDE_roots_less_than_one_l728_72887

theorem roots_less_than_one (a b : ℝ) (h : abs a + abs b < 1) :
  ∀ x, x^2 + a*x + b = 0 → abs x < 1 :=
sorry

end NUMINAMATH_CALUDE_roots_less_than_one_l728_72887


namespace NUMINAMATH_CALUDE_man_speed_man_speed_specific_case_l728_72814

/-- Calculates the speed of a man running opposite to a train, given the train's length, speed, and time to pass the man. -/
theorem man_speed (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / time_to_pass
  let man_speed_ms := relative_speed - train_speed_ms
  let man_speed_kmh := man_speed_ms * 3600 / 1000
  man_speed_kmh

/-- Proves that the speed of the man is approximately 6 km/hr given the specific conditions. -/
theorem man_speed_specific_case : 
  ∃ ε > 0, |man_speed 110 84 4.399648028157747 - 6| < ε :=
sorry

end NUMINAMATH_CALUDE_man_speed_man_speed_specific_case_l728_72814


namespace NUMINAMATH_CALUDE_complex_product_ab_l728_72884

theorem complex_product_ab (a b : ℝ) (i : ℂ) (h : i * i = -1) 
  (h1 : (1 - 2*i)*i = a + b*i) : a * b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_ab_l728_72884


namespace NUMINAMATH_CALUDE_solution_for_y_l728_72898

theorem solution_for_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 1 + 1/y) (eq2 : y = 2 + 1/x) :
  y = 1 + Real.sqrt 3 ∨ y = 1 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_for_y_l728_72898


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l728_72846

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose n 3 = 7 * Nat.choose n 1) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l728_72846


namespace NUMINAMATH_CALUDE_composite_shape_area_theorem_l728_72892

/-- The composite shape formed by a hexagon and an octagon attached to an equilateral triangle --/
structure CompositeShape where
  sideLength : ℝ
  hexagonArea : ℝ
  octagonArea : ℝ

/-- Calculate the area of the composite shape --/
def compositeShapeArea (shape : CompositeShape) : ℝ :=
  shape.hexagonArea + shape.octagonArea

/-- The theorem stating the area of the composite shape --/
theorem composite_shape_area_theorem (shape : CompositeShape) 
  (h1 : shape.sideLength = 2)
  (h2 : shape.hexagonArea = 4 * Real.sqrt 3 + 6)
  (h3 : shape.octagonArea = 8 * (1 + Real.sqrt 2) - 12) :
  compositeShapeArea shape = 4 * Real.sqrt 3 + 8 * Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_composite_shape_area_theorem_l728_72892


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l728_72808

theorem quadratic_form_sum (x : ℝ) : ∃ (a h k : ℝ), 
  (6 * x^2 - 24 * x + 10 = a * (x - h)^2 + k) ∧ (a + h + k = -6) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l728_72808


namespace NUMINAMATH_CALUDE_polynomial_simplification_l728_72856

theorem polynomial_simplification (x : ℝ) :
  (3*x - 2) * (5*x^12 + 3*x^11 + 7*x^9 + 3*x^8) =
  15*x^13 - x^12 - 6*x^11 + 21*x^10 - 5*x^9 - 6*x^8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l728_72856


namespace NUMINAMATH_CALUDE_condition_nature_l728_72852

-- Define the set M
def M (a : ℝ) : Set ℝ := {x | |2*x - a| < 2}

-- Theorem statement
theorem condition_nature (a : ℝ) :
  (∀ a, 1 ∈ M a → 0 ≤ a ∧ a ≤ 4) ∧
  (∃ a, 0 ≤ a ∧ a ≤ 4 ∧ 1 ∉ M a) := by
  sorry

end NUMINAMATH_CALUDE_condition_nature_l728_72852


namespace NUMINAMATH_CALUDE_x_squared_is_quadratic_l728_72883

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² = 0 -/
def f (x : ℝ) : ℝ := x^2

/-- Theorem stating that x² = 0 is a quadratic equation -/
theorem x_squared_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_is_quadratic_l728_72883


namespace NUMINAMATH_CALUDE_average_temperature_l728_72802

def temperatures : List ℝ := [52, 64, 59, 60, 47]

theorem average_temperature : 
  (temperatures.sum / temperatures.length : ℝ) = 56.4 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_l728_72802


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l728_72823

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 1) / Real.log 10}
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem intersection_complement_equals_set :
  N ∩ (Mᶜ) = {x | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l728_72823


namespace NUMINAMATH_CALUDE_straw_hat_value_is_four_l728_72801

/-- Represents the sheep problem scenario -/
structure SheepProblem where
  x : ℕ  -- number of sheep
  y : ℕ  -- number of times 10 yuan was taken
  z : ℕ  -- last amount taken by younger brother
  h1 : x^2 = x * x  -- price of each sheep equals number of sheep
  h2 : x^2 = 20 * y + 10 + z  -- total money distribution
  h3 : y ≥ 1  -- at least one round of 10 yuan taken
  h4 : z < 10  -- younger brother's last amount less than 10

/-- The value of the straw hat that equalizes the brothers' shares -/
def strawHatValue (p : SheepProblem) : ℕ := 10 - p.z

/-- Theorem stating the value of the straw hat is 4 yuan -/
theorem straw_hat_value_is_four (p : SheepProblem) : strawHatValue p = 4 := by
  sorry

#check straw_hat_value_is_four

end NUMINAMATH_CALUDE_straw_hat_value_is_four_l728_72801


namespace NUMINAMATH_CALUDE_sum_of_first_60_digits_l728_72872

/-- The decimal representation of 1/9999 -/
def decimal_rep : ℚ := 1 / 9999

/-- The sequence of digits in the decimal representation of 1/9999 -/
def digit_sequence : ℕ → ℕ
  | n => match n % 4 with
         | 0 => 0
         | 1 => 0
         | 2 => 0
         | 3 => 1
         | _ => 0  -- This case is technically unreachable

/-- The sum of the first n digits in the sequence -/
def digit_sum (n : ℕ) : ℕ := (List.range n).map digit_sequence |>.sum

theorem sum_of_first_60_digits :
  digit_sum 60 = 15 :=
sorry

end NUMINAMATH_CALUDE_sum_of_first_60_digits_l728_72872


namespace NUMINAMATH_CALUDE_inhabitable_earth_fraction_l728_72858

-- Define the fraction of Earth's surface that is land
def land_fraction : ℚ := 1 / 5

-- Define the fraction of land that is inhabitable
def inhabitable_land_fraction : ℚ := 1 / 3

-- Theorem: The fraction of Earth's surface that humans can live on is 1/15
theorem inhabitable_earth_fraction :
  land_fraction * inhabitable_land_fraction = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_inhabitable_earth_fraction_l728_72858


namespace NUMINAMATH_CALUDE_similar_triangles_leg_length_l728_72899

theorem similar_triangles_leg_length :
  ∀ (y : ℝ),
  (12 : ℝ) / y = 9 / 6 →
  y = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_length_l728_72899


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l728_72851

/-- Given that 2/3 of 10 bananas are worth as much as 8 oranges,
    prove that 1/2 of 5 bananas are worth as much as 3 oranges. -/
theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℚ),
    (2 / 3 : ℚ) * 10 * banana_value = 8 * orange_value →
    (1 / 2 : ℚ) * 5 * banana_value = 3 * orange_value := by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l728_72851


namespace NUMINAMATH_CALUDE_probability_cos_pi_x_geq_half_over_interval_probability_equals_one_third_l728_72836

/-- The probability that cos(πx) ≥ 1/2 for x uniformly distributed in [-1, 1] -/
theorem probability_cos_pi_x_geq_half_over_interval (x : ℝ) : 
  ℝ := by sorry

/-- The probability is equal to 1/3 -/
theorem probability_equals_one_third : 
  probability_cos_pi_x_geq_half_over_interval = 1/3 := by sorry

end NUMINAMATH_CALUDE_probability_cos_pi_x_geq_half_over_interval_probability_equals_one_third_l728_72836


namespace NUMINAMATH_CALUDE_bike_price_l728_72888

theorem bike_price (P : ℝ) : P + 0.1 * P = 82500 → P = 75000 := by
  sorry

end NUMINAMATH_CALUDE_bike_price_l728_72888


namespace NUMINAMATH_CALUDE_polynomial_simplification_l728_72832

theorem polynomial_simplification (w : ℝ) : 
  3*w + 4 - 2*w^2 - 5*w - 6 + w^2 + 7*w + 8 - 3*w^2 = 5*w - 4*w^2 + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l728_72832


namespace NUMINAMATH_CALUDE_calculate_expression_l728_72848

theorem calculate_expression : (18 / (3 + 9 - 6)) * 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l728_72848


namespace NUMINAMATH_CALUDE_complex_modulus_l728_72843

theorem complex_modulus (i : ℂ) (z : ℂ) : 
  i * i = -1 → z = 2 + 3 * i / (1 - i) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l728_72843


namespace NUMINAMATH_CALUDE_tangent_circles_locus_l728_72825

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency relation
def isTangent (c1 c2 : Circle) : Prop :=
  let d := Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2)
  d = c1.radius + c2.radius ∨ d = |c1.radius - c2.radius|

-- Define the locus of points
inductive Locus
  | Hyperbola
  | StraightLine

-- Theorem statement
theorem tangent_circles_locus 
  (O₁ O₂ P : Circle) 
  (h_separate : O₁.center ≠ O₂.center) 
  (h_tangent₁ : isTangent O₁ P) 
  (h_tangent₂ : isTangent O₂ P) :
  (∃ l₁ l₂ : Locus, l₁ = Locus.Hyperbola ∧ l₂ = Locus.Hyperbola) ∨
  (∃ l₁ l₂ : Locus, l₁ = Locus.Hyperbola ∧ l₂ = Locus.StraightLine) :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_locus_l728_72825


namespace NUMINAMATH_CALUDE_bakery_weekly_sales_l728_72876

/-- Represents the daily sales of cakes for a specific type -/
structure DailySales :=
  (monday : Nat)
  (tuesday : Nat)
  (wednesday : Nat)
  (thursday : Nat)
  (friday : Nat)
  (saturday : Nat)
  (sunday : Nat)

/-- Represents the weekly sales data for all cake types -/
structure WeeklySales :=
  (chocolate : DailySales)
  (vanilla : DailySales)
  (strawberry : DailySales)

def bakery_sales : WeeklySales :=
  { chocolate := { monday := 6, tuesday := 7, wednesday := 4, thursday := 8, friday := 9, saturday := 10, sunday := 5 },
    vanilla := { monday := 4, tuesday := 5, wednesday := 3, thursday := 7, friday := 6, saturday := 8, sunday := 4 },
    strawberry := { monday := 3, tuesday := 2, wednesday := 6, thursday := 4, friday := 5, saturday := 7, sunday := 4 } }

def total_sales (sales : DailySales) : Nat :=
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday + sales.friday + sales.saturday + sales.sunday

theorem bakery_weekly_sales :
  total_sales bakery_sales.chocolate = 49 ∧
  total_sales bakery_sales.vanilla = 37 ∧
  total_sales bakery_sales.strawberry = 31 := by
  sorry

end NUMINAMATH_CALUDE_bakery_weekly_sales_l728_72876


namespace NUMINAMATH_CALUDE_prep_school_cost_l728_72860

theorem prep_school_cost (cost_per_semester : ℕ) (semesters_per_year : ℕ) (years : ℕ) : 
  cost_per_semester = 20000 → semesters_per_year = 2 → years = 13 →
  cost_per_semester * semesters_per_year * years = 520000 := by
  sorry

end NUMINAMATH_CALUDE_prep_school_cost_l728_72860


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l728_72803

theorem cubic_equation_solution (a : ℝ) : a^3 = 21 * 25 * 35 * 63 → a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l728_72803


namespace NUMINAMATH_CALUDE_point_movement_to_x_axis_l728_72868

/-- Given a point P with coordinates (m+2, 2m+4) that is moved 2 units up to point Q which lies on the x-axis, prove that the coordinates of Q are (-1, 0) -/
theorem point_movement_to_x_axis (m : ℝ) :
  let P : ℝ × ℝ := (m + 2, 2*m + 4)
  let Q : ℝ × ℝ := (P.1, P.2 + 2)
  Q.2 = 0 → Q = (-1, 0) := by sorry

end NUMINAMATH_CALUDE_point_movement_to_x_axis_l728_72868


namespace NUMINAMATH_CALUDE_skitties_remainder_l728_72875

theorem skitties_remainder (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_skitties_remainder_l728_72875


namespace NUMINAMATH_CALUDE_rectangles_may_not_be_similar_squares_always_similar_equilateral_triangles_always_similar_isosceles_right_triangles_always_similar_l728_72886

-- Define the shapes
structure Square where
  side : ℝ
  side_positive : side > 0

structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

structure IsoscelesRightTriangle where
  leg : ℝ
  leg_positive : leg > 0

structure Rectangle where
  length : ℝ
  width : ℝ
  length_positive : length > 0
  width_positive : width > 0

-- Define similarity
def similar {α : Type*} (x y : α) : Prop := sorry

-- Theorem stating that rectangles may not always be similar
theorem rectangles_may_not_be_similar :
  ∃ (r1 r2 : Rectangle), ¬ similar r1 r2 :=
sorry

-- Theorems stating that other shapes are always similar
theorem squares_always_similar (s1 s2 : Square) :
  similar s1 s2 :=
sorry

theorem equilateral_triangles_always_similar (t1 t2 : EquilateralTriangle) :
  similar t1 t2 :=
sorry

theorem isosceles_right_triangles_always_similar (t1 t2 : IsoscelesRightTriangle) :
  similar t1 t2 :=
sorry

end NUMINAMATH_CALUDE_rectangles_may_not_be_similar_squares_always_similar_equilateral_triangles_always_similar_isosceles_right_triangles_always_similar_l728_72886


namespace NUMINAMATH_CALUDE_burger_calorie_content_l728_72830

/-- Represents the calorie content of a lunch meal -/
structure LunchMeal where
  burger_calories : ℕ
  carrot_stick_calories : ℕ
  cookie_calories : ℕ
  carrot_stick_count : ℕ
  cookie_count : ℕ
  total_calories : ℕ

/-- Theorem stating the calorie content of a burger in a specific lunch meal -/
theorem burger_calorie_content (meal : LunchMeal) 
  (h1 : meal.carrot_stick_calories = 20)
  (h2 : meal.cookie_calories = 50)
  (h3 : meal.carrot_stick_count = 5)
  (h4 : meal.cookie_count = 5)
  (h5 : meal.total_calories = 750) :
  meal.burger_calories = 400 := by
  sorry

end NUMINAMATH_CALUDE_burger_calorie_content_l728_72830


namespace NUMINAMATH_CALUDE_no_integer_square_root_Q_l728_72821

/-- The polynomial Q(x) = x^4 + 8x^3 + 18x^2 + 11x + 27 -/
def Q (x : ℤ) : ℤ := x^4 + 8*x^3 + 18*x^2 + 11*x + 27

/-- Theorem stating that there are no integer values of x for which Q(x) is a perfect square -/
theorem no_integer_square_root_Q :
  ∀ x : ℤ, ¬∃ y : ℤ, Q x = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_square_root_Q_l728_72821
