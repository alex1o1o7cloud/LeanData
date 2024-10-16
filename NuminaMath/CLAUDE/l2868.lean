import Mathlib

namespace NUMINAMATH_CALUDE_water_intake_calculation_l2868_286845

theorem water_intake_calculation (morning_intake : Real) 
  (h1 : morning_intake = 1.5)
  (h2 : afternoon_intake = 3 * morning_intake)
  (h3 : evening_intake = 0.5 * afternoon_intake) :
  morning_intake + afternoon_intake + evening_intake = 8.25 := by
  sorry

end NUMINAMATH_CALUDE_water_intake_calculation_l2868_286845


namespace NUMINAMATH_CALUDE_g_of_three_eq_seventeen_sixths_l2868_286897

/-- Given a function g satisfying the equation for all x ≠ 1/2, prove that g(3) = 17/6 -/
theorem g_of_three_eq_seventeen_sixths 
  (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, x ≠ 1/2 → g x + g ((x + 2) / (2 - 4*x)) = 2*x) : 
  g 3 = 17/6 := by
sorry

end NUMINAMATH_CALUDE_g_of_three_eq_seventeen_sixths_l2868_286897


namespace NUMINAMATH_CALUDE_cheapest_combination_is_12_apples_3_oranges_l2868_286804

/-- Represents the price of a fruit deal -/
structure FruitDeal where
  quantity : Nat
  price : Rat

/-- Represents the fruit options available -/
structure FruitOptions where
  apple_deals : List FruitDeal
  orange_deals : List FruitDeal

/-- Represents a combination of apples and oranges -/
structure FruitCombination where
  apples : Nat
  oranges : Nat

def total_fruits (combo : FruitCombination) : Nat :=
  combo.apples + combo.oranges

def is_valid_combination (combo : FruitCombination) : Prop :=
  total_fruits combo = 15 ∧
  (combo.apples % 2 = 0 ∨ combo.apples % 3 = 0) ∧
  (combo.oranges % 2 = 0 ∨ combo.oranges % 3 = 0)

def cost_of_combination (options : FruitOptions) (combo : FruitCombination) : Rat :=
  sorry

def cheapest_combination (options : FruitOptions) : FruitCombination :=
  sorry

theorem cheapest_combination_is_12_apples_3_oranges
  (options : FruitOptions)
  (h_apple_deals : options.apple_deals = [
    ⟨2, 48/100⟩, ⟨6, 126/100⟩, ⟨12, 224/100⟩
  ])
  (h_orange_deals : options.orange_deals = [
    ⟨2, 60/100⟩, ⟨6, 164/100⟩, ⟨12, 300/100⟩
  ]) :
  cheapest_combination options = ⟨12, 3⟩ ∧
  cost_of_combination options (cheapest_combination options) = 314/100 :=
sorry

end NUMINAMATH_CALUDE_cheapest_combination_is_12_apples_3_oranges_l2868_286804


namespace NUMINAMATH_CALUDE_johnny_work_days_l2868_286834

def daily_earnings : ℝ := 3 * 7 + 2 * 10 + 4 * 12

theorem johnny_work_days (x : ℝ) (h : x * daily_earnings = 445) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_johnny_work_days_l2868_286834


namespace NUMINAMATH_CALUDE_first_number_in_proportion_l2868_286869

/-- Given a proportion a : 1.65 :: 5 : 11, prove that a = 0.75 -/
theorem first_number_in_proportion (a : ℝ) : 
  (a / 1.65 = 5 / 11) → a = 0.75 := by
sorry

end NUMINAMATH_CALUDE_first_number_in_proportion_l2868_286869


namespace NUMINAMATH_CALUDE_james_lifting_ratio_l2868_286840

def initial_total : ℝ := 2200
def initial_weight : ℝ := 245
def total_gain_percentage : ℝ := 0.15
def weight_gain : ℝ := 8

def new_total : ℝ := initial_total * (1 + total_gain_percentage)
def new_weight : ℝ := initial_weight + weight_gain

theorem james_lifting_ratio :
  new_total / new_weight = 10 := by sorry

end NUMINAMATH_CALUDE_james_lifting_ratio_l2868_286840


namespace NUMINAMATH_CALUDE_transportation_puzzle_l2868_286849

def is_valid_assignment (T R A N S P O B K : ℕ) : Prop :=
  T > R ∧ R > A ∧ A > N ∧ N < S ∧ S < P ∧ P < O ∧ O < R ∧ R < T ∧
  T > R ∧ R > O ∧ O < A ∧ A > B ∧ B < K ∧ K < A ∧
  T ≠ R ∧ T ≠ A ∧ T ≠ N ∧ T ≠ S ∧ T ≠ P ∧ T ≠ O ∧ T ≠ B ∧ T ≠ K ∧
  R ≠ A ∧ R ≠ N ∧ R ≠ S ∧ R ≠ P ∧ R ≠ O ∧ R ≠ B ∧ R ≠ K ∧
  A ≠ N ∧ A ≠ S ∧ A ≠ P ∧ A ≠ O ∧ A ≠ B ∧ A ≠ K ∧
  N ≠ S ∧ N ≠ P ∧ N ≠ O ∧ N ≠ B ∧ N ≠ K ∧
  S ≠ P ∧ S ≠ O ∧ S ≠ B ∧ S ≠ K ∧
  P ≠ O ∧ P ≠ B ∧ P ≠ K ∧
  O ≠ B ∧ O ≠ K ∧
  B ≠ K

theorem transportation_puzzle :
  ∃! (T R A N S P O B K : ℕ), is_valid_assignment T R A N S P O B K :=
sorry

end NUMINAMATH_CALUDE_transportation_puzzle_l2868_286849


namespace NUMINAMATH_CALUDE_chernov_has_gray_hair_l2868_286883

-- Define the three people
inductive Person : Type
| Sedov : Person
| Chernov : Person
| Ryzhov : Person

-- Define the hair colors
inductive HairColor : Type
| Gray : HairColor
| Red : HairColor
| Black : HairColor

-- Define the sports ranks
inductive SportsRank : Type
| MasterOfSports : SportsRank
| CandidateMaster : SportsRank
| FirstRank : SportsRank

-- Define the function that assigns a hair color to each person
def hairColor : Person → HairColor := sorry

-- Define the function that assigns a sports rank to each person
def sportsRank : Person → SportsRank := sorry

-- State the theorem
theorem chernov_has_gray_hair :
  -- No person's hair color matches their surname
  (hairColor Person.Sedov ≠ HairColor.Gray) ∧
  (hairColor Person.Chernov ≠ HairColor.Black) ∧
  (hairColor Person.Ryzhov ≠ HairColor.Red) ∧
  -- One person is gray-haired, one is red-haired, and one is black-haired
  (∃! p : Person, hairColor p = HairColor.Gray) ∧
  (∃! p : Person, hairColor p = HairColor.Red) ∧
  (∃! p : Person, hairColor p = HairColor.Black) ∧
  -- The black-haired person made the statement
  (∃ p : Person, hairColor p = HairColor.Black ∧ p ≠ Person.Sedov ∧ p ≠ Person.Chernov) ∧
  -- The Master of Sports confirmed the statement
  (sportsRank Person.Sedov = SportsRank.MasterOfSports) ∧
  (sportsRank Person.Chernov = SportsRank.CandidateMaster) ∧
  (sportsRank Person.Ryzhov = SportsRank.FirstRank) →
  -- Conclusion: Chernov has gray hair
  hairColor Person.Chernov = HairColor.Gray :=
by
  sorry


end NUMINAMATH_CALUDE_chernov_has_gray_hair_l2868_286883


namespace NUMINAMATH_CALUDE_coefficient_of_x3_in_expansion_l2868_286838

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function to calculate the coefficient of x^3
def coefficientOfX3 (a b : ℝ) (n : ℕ) : ℝ :=
  (-b)^1 * binomial n 1 * a^3

-- Theorem statement
theorem coefficient_of_x3_in_expansion :
  coefficientOfX3 1 3 7 = -21 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x3_in_expansion_l2868_286838


namespace NUMINAMATH_CALUDE_plastic_rings_weight_sum_l2868_286896

theorem plastic_rings_weight_sum :
  let orange_ring : Float := 0.08333333333333333
  let purple_ring : Float := 0.3333333333333333
  let white_ring : Float := 0.4166666666666667
  orange_ring + purple_ring + white_ring = 0.8333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_plastic_rings_weight_sum_l2868_286896


namespace NUMINAMATH_CALUDE_fraction_integer_condition_l2868_286811

theorem fraction_integer_condition (n : ℤ) : 
  (↑(n + 1) / ↑(2 * n - 1) : ℚ).isInt ↔ n = 2 ∨ n = 1 ∨ n = 0 ∨ n = -1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_integer_condition_l2868_286811


namespace NUMINAMATH_CALUDE_sum_congruent_to_6_mod_9_l2868_286830

def sum : ℕ := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999

theorem sum_congruent_to_6_mod_9 : sum % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruent_to_6_mod_9_l2868_286830


namespace NUMINAMATH_CALUDE_problem_1_l2868_286858

theorem problem_1 : (-13/2 : ℚ) * (4/13 : ℚ) - 8 / |(-4 + 2)| = -6 := by sorry

end NUMINAMATH_CALUDE_problem_1_l2868_286858


namespace NUMINAMATH_CALUDE_abs_4y_minus_7_not_positive_l2868_286807

theorem abs_4y_minus_7_not_positive (y : ℚ) :
  (|4 * y - 7| ≤ 0) ↔ (y = 7 / 4) := by sorry

end NUMINAMATH_CALUDE_abs_4y_minus_7_not_positive_l2868_286807


namespace NUMINAMATH_CALUDE_max_sum_at_15_l2868_286812

/-- An arithmetic sequence with first term 29 and S_10 = S_20 -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_term : a 1 = 29
  sum_equal : (Finset.range 10).sum a = (Finset.range 20).sum a

/-- Sum of the first n terms of the arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (Finset.range n).sum seq.a

/-- The maximum value of S_n occurs when n = 15 -/
theorem max_sum_at_15 (seq : ArithmeticSequence) :
  ∀ n : ℕ, S seq n ≤ S seq 15 :=
sorry

end NUMINAMATH_CALUDE_max_sum_at_15_l2868_286812


namespace NUMINAMATH_CALUDE_sandy_comic_books_l2868_286870

theorem sandy_comic_books :
  ∃ (initial : ℕ), (initial / 2 + 6 : ℕ) = 13 ∧ initial = 14 :=
by sorry

end NUMINAMATH_CALUDE_sandy_comic_books_l2868_286870


namespace NUMINAMATH_CALUDE_one_meeting_l2868_286857

/-- Represents a boy moving on a circular track -/
structure Boy where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- The problem setup -/
def circularTrackProblem (circumference : ℝ) (boy1 boy2 : Boy) : Prop :=
  circumference > 0 ∧
  boy1.speed = 6 ∧
  boy2.speed = 10 ∧
  boy1.direction ≠ boy2.direction

/-- The number of meetings between the two boys -/
def numberOfMeetings (circumference : ℝ) (boy1 boy2 : Boy) : ℕ := sorry

/-- The theorem stating that the boys meet exactly once -/
theorem one_meeting (circumference : ℝ) (boy1 boy2 : Boy) 
  (h : circularTrackProblem circumference boy1 boy2) : 
  numberOfMeetings circumference boy1 boy2 = 1 := by sorry

end NUMINAMATH_CALUDE_one_meeting_l2868_286857


namespace NUMINAMATH_CALUDE_cafeteria_extra_apples_l2868_286844

/-- The number of extra apples in the cafeteria -/
def extra_apples (red_apples green_apples students_wanting_fruit : ℕ) : ℕ :=
  red_apples + green_apples - students_wanting_fruit

/-- Theorem: The cafeteria ends up with 32 extra apples -/
theorem cafeteria_extra_apples :
  extra_apples 25 17 10 = 32 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_extra_apples_l2868_286844


namespace NUMINAMATH_CALUDE_F_properties_l2868_286859

noncomputable def f (x : ℝ) : ℝ := 3 - 2 * abs x

noncomputable def g (x : ℝ) : ℝ := x^2 - 2*x

noncomputable def F (x : ℝ) : ℝ :=
  if f x ≥ g x then g x else f x

theorem F_properties :
  (∃ (M : ℝ), M = 7 - 2 * Real.sqrt 7 ∧ ∀ (x : ℝ), F x ≤ M) ∧
  (¬ ∃ (m : ℝ), ∀ (x : ℝ), F x ≥ m) :=
sorry

end NUMINAMATH_CALUDE_F_properties_l2868_286859


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l2868_286865

/-- Proves that the speed of a stream is 12.6 kmph given specific boat travel conditions -/
theorem stream_speed_calculation (boat_speed : ℝ) (downstream_time : ℝ) (upstream_time : ℝ) 
  (h1 : boat_speed = 15)
  (h2 : downstream_time = 1)
  (h3 : upstream_time = 11.5) :
  ∃ (stream_speed : ℝ), 
    stream_speed = 12.6 ∧ 
    (boat_speed + stream_speed) * downstream_time = (boat_speed - stream_speed) * upstream_time :=
by
  sorry


end NUMINAMATH_CALUDE_stream_speed_calculation_l2868_286865


namespace NUMINAMATH_CALUDE_equation_solution_l2868_286867

theorem equation_solution : ∃ (x₁ x₂ : ℝ), x₁ = -2 ∧ x₂ = 3 ∧
  ∀ x : ℝ, (x + 2)^2 - 5*(x + 2) = 0 ↔ x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2868_286867


namespace NUMINAMATH_CALUDE_multiplicative_inverse_480_mod_4799_l2868_286860

theorem multiplicative_inverse_480_mod_4799 : 
  (∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ a = 40 ∧ b = 399 ∧ c = 401) →
  (∃ (n : ℕ), n < 4799 ∧ (480 * n) % 4799 = 1) ∧
  (480 * 4789) % 4799 = 1 :=
by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_480_mod_4799_l2868_286860


namespace NUMINAMATH_CALUDE_even_digits_base7_512_l2868_286876

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Theorem: The number of even digits in the base-7 representation of 512₁₀ is 0 -/
theorem even_digits_base7_512 : countEvenDigits (toBase7 512) = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_digits_base7_512_l2868_286876


namespace NUMINAMATH_CALUDE_sqrt_nine_equals_three_l2868_286872

theorem sqrt_nine_equals_three : Real.sqrt 9 = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_nine_equals_three_l2868_286872


namespace NUMINAMATH_CALUDE_constant_ratio_problem_l2868_286882

theorem constant_ratio_problem (x y : ℝ) (k : ℝ) :
  (∀ x y, (4 * x - 5) / (2 * y + 20) = k) →
  (4 * 4 - 5) / (2 * 5 + 20) = k →
  (4 * 9 - 5) / (2 * (355 / 11) + 20) = k :=
by sorry

end NUMINAMATH_CALUDE_constant_ratio_problem_l2868_286882


namespace NUMINAMATH_CALUDE_product_calculation_l2868_286820

theorem product_calculation : 10 * 0.2 * 0.5 * 4 / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_calculation_l2868_286820


namespace NUMINAMATH_CALUDE_problem_solution_l2868_286816

def f (x : ℝ) : ℝ := |2*x + 3| + |x - 1|

theorem problem_solution :
  (∀ x : ℝ, f x > 4 ↔ x ∈ Set.Iio (-2) ∪ Set.Ioi 0) ∧
  (∀ m : ℝ, (∃ x₀ : ℝ, ∀ t : ℝ, f x₀ < |m + t| + |t - m|) ↔ 
    m ∈ Set.Iio (-5/4) ∪ Set.Ioi (5/4)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2868_286816


namespace NUMINAMATH_CALUDE_mother_daughter_ages_l2868_286888

/-- Proves the ages of a mother and daughter given certain conditions --/
theorem mother_daughter_ages :
  ∀ (daughter_age mother_age : ℕ),
  mother_age = daughter_age + 22 →
  2 * (2 * daughter_age) = 2 * daughter_age + 22 →
  daughter_age = 11 ∧ mother_age = 33 :=
by
  sorry

#check mother_daughter_ages

end NUMINAMATH_CALUDE_mother_daughter_ages_l2868_286888


namespace NUMINAMATH_CALUDE_unique_outfits_count_l2868_286829

def number_of_shirts : ℕ := 10
def number_of_ties : ℕ := 8
def shirts_per_outfit : ℕ := 5
def ties_per_outfit : ℕ := 4

theorem unique_outfits_count : 
  (Nat.choose number_of_shirts shirts_per_outfit) * 
  (Nat.choose number_of_ties ties_per_outfit) = 17640 := by
  sorry

end NUMINAMATH_CALUDE_unique_outfits_count_l2868_286829


namespace NUMINAMATH_CALUDE_range_of_a_l2868_286874

open Set Real

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Ioo 2 3, x^2 + 5 > a*x) = false → 
  a ∈ Ici (2 * sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2868_286874


namespace NUMINAMATH_CALUDE_sandwiches_left_for_others_l2868_286855

def total_sandwiches : ℕ := 20
def sandwiches_for_coworker : ℕ := 4
def sandwiches_for_self : ℕ := 2 * sandwiches_for_coworker

theorem sandwiches_left_for_others : 
  total_sandwiches - sandwiches_for_coworker - sandwiches_for_self = 8 := by
  sorry

end NUMINAMATH_CALUDE_sandwiches_left_for_others_l2868_286855


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_closed_open_interval_l2868_286853

def A : Set ℝ := {x : ℝ | |x| ≥ 2}
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

theorem A_intersect_B_equals_closed_open_interval :
  A ∩ B = Set.Icc 2 3 \ {3} :=
by sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_closed_open_interval_l2868_286853


namespace NUMINAMATH_CALUDE_distance_calculation_l2868_286878

def boat_speed : ℝ := 9
def stream_speed : ℝ := 6
def total_time : ℝ := 84

theorem distance_calculation (distance : ℝ) : 
  (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) = total_time) → 
  distance = 210 := by
sorry

end NUMINAMATH_CALUDE_distance_calculation_l2868_286878


namespace NUMINAMATH_CALUDE_table_height_is_five_l2868_286808

/-- Represents the configuration of blocks and table -/
structure Configuration where
  total_length : ℝ

/-- Represents the table and blocks setup -/
structure TableSetup where
  block_length : ℝ
  block_width : ℝ
  table_height : ℝ
  config1 : Configuration
  config2 : Configuration

/-- The theorem stating the height of the table given the configurations -/
theorem table_height_is_five (setup : TableSetup)
  (h1 : setup.config1.total_length = setup.block_length + setup.table_height + setup.block_width)
  (h2 : setup.config2.total_length = 2 * setup.block_width + setup.table_height)
  (h3 : setup.config1.total_length = 45)
  (h4 : setup.config2.total_length = 40) :
  setup.table_height = 5 := by
  sorry

#check table_height_is_five

end NUMINAMATH_CALUDE_table_height_is_five_l2868_286808


namespace NUMINAMATH_CALUDE_min_value_of_function_l2868_286821

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  (12 / x + 4 * x) ≥ 8 * Real.sqrt 3 ∧ ∃ y > 0, 12 / y + 4 * y = 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2868_286821


namespace NUMINAMATH_CALUDE_n_sided_polygon_interior_angles_l2868_286842

theorem n_sided_polygon_interior_angles (n : ℕ) : 
  (n - 2) * 180 = 720 → n = 6 := by sorry

end NUMINAMATH_CALUDE_n_sided_polygon_interior_angles_l2868_286842


namespace NUMINAMATH_CALUDE_plant_purchase_solution_l2868_286837

/-- Represents the prices and quantities of plants A and B -/
structure PlantPurchase where
  price_a : ℝ
  price_b : ℝ
  quantity_a : ℕ
  quantity_b : ℕ

/-- Calculates the total cost of a plant purchase -/
def total_cost (p : PlantPurchase) : ℝ :=
  p.price_a * p.quantity_a + p.price_b * p.quantity_b

/-- Represents the given conditions from the problem -/
structure ProblemConditions where
  first_phase : PlantPurchase
  second_phase : PlantPurchase
  total_cost_both_phases : ℝ

/-- The main theorem representing the problem and its solution -/
theorem plant_purchase_solution (conditions : ProblemConditions) 
  (h1 : conditions.first_phase.quantity_a = 30)
  (h2 : conditions.first_phase.quantity_b = 15)
  (h3 : total_cost conditions.first_phase = 675)
  (h4 : conditions.second_phase.quantity_a = 12)
  (h5 : conditions.second_phase.quantity_b = 5)
  (h6 : conditions.total_cost_both_phases = 940)
  (h7 : conditions.first_phase.price_a = conditions.second_phase.price_a)
  (h8 : conditions.first_phase.price_b = conditions.second_phase.price_b) :
  ∃ (optimal_plan : PlantPurchase),
    conditions.first_phase.price_a = 20 ∧
    conditions.first_phase.price_b = 5 ∧
    optimal_plan.quantity_a + optimal_plan.quantity_b = 31 ∧
    optimal_plan.quantity_b < 2 * optimal_plan.quantity_a ∧
    total_cost optimal_plan = 320 ∧
    ∀ (other_plan : PlantPurchase),
      other_plan.quantity_a + other_plan.quantity_b = 31 →
      other_plan.quantity_b < 2 * other_plan.quantity_a →
      total_cost other_plan ≥ total_cost optimal_plan := by
  sorry

end NUMINAMATH_CALUDE_plant_purchase_solution_l2868_286837


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l2868_286864

theorem largest_prime_factors_difference (n : Nat) (h : n = 242858) :
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p ∣ n ∧ q ∣ n ∧
  (∀ r : Nat, Nat.Prime r → r ∣ n → r ≤ p ∧ r ≤ q) ∧
  p ≠ q ∧ p - q = 80 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l2868_286864


namespace NUMINAMATH_CALUDE_frank_candy_count_l2868_286863

theorem frank_candy_count (bags : ℕ) (pieces_per_bag : ℕ) 
  (h1 : bags = 26) (h2 : pieces_per_bag = 33) : 
  bags * pieces_per_bag = 858 := by
  sorry

end NUMINAMATH_CALUDE_frank_candy_count_l2868_286863


namespace NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_l2868_286817

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

-- Define the hyperbola equation
def hyperbola (x y m : ℝ) : Prop := x^2 - m*(y+1)^2 = 1

-- Define the tangency condition
def are_tangent (m : ℝ) : Prop :=
  ∃ x y, ellipse x y ∧ hyperbola x y m

-- Theorem statement
theorem tangent_ellipse_hyperbola :
  ∀ m, are_tangent m → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_l2868_286817


namespace NUMINAMATH_CALUDE_maximum_at_one_implies_a_greater_than_neg_one_l2868_286868

/-- The function f(x) = ln x - (1/2)ax² - bx has a maximum at x = 1 -/
def has_maximum_at_one (a b : ℝ) : Prop :=
  ∀ x, x > 0 → (Real.log x - (1/2) * a * x^2 - b * x) ≤ (Real.log 1 - (1/2) * a * 1^2 - b * 1)

/-- If f(x) = ln x - (1/2)ax² - bx has a maximum at x = 1, then a > -1 -/
theorem maximum_at_one_implies_a_greater_than_neg_one (a b : ℝ) :
  has_maximum_at_one a b → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_maximum_at_one_implies_a_greater_than_neg_one_l2868_286868


namespace NUMINAMATH_CALUDE_polynomial_sum_property_l2868_286843

/-- Generate all words of length n using letters A and B -/
def generateWords (n : ℕ) : List String :=
  sorry

/-- Convert a word to a polynomial by replacing A with x and B with (1-x) -/
def wordToPolynomial (word : String) : ℝ → ℝ :=
  sorry

/-- Sum the first k polynomials -/
def sumPolynomials (n : ℕ) (k : ℕ) : ℝ → ℝ :=
  sorry

/-- A function is increasing on [0,1] -/
def isIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y

/-- A function is constant on [0,1] -/
def isConstant (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → f x = f y

theorem polynomial_sum_property (n : ℕ) (k : ℕ) (h : 1 ≤ k ∧ k ≤ 2^n) :
  let f := sumPolynomials n k
  isConstant f ∨ isIncreasing f :=
sorry

end NUMINAMATH_CALUDE_polynomial_sum_property_l2868_286843


namespace NUMINAMATH_CALUDE_insect_jump_coordinates_l2868_286806

/-- A point in a 2D plane represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a jump to the right by a certain distance -/
def jumpRight (p : Point) (distance : ℝ) : Point :=
  ⟨p.x + distance, p.y⟩

theorem insect_jump_coordinates :
  let A : Point := ⟨-2, 1⟩
  let B : Point := jumpRight A 4
  B.x = 2 ∧ B.y = 1 := by
  sorry

end NUMINAMATH_CALUDE_insect_jump_coordinates_l2868_286806


namespace NUMINAMATH_CALUDE_function_inequality_implies_squares_inequality_l2868_286809

theorem function_inequality_implies_squares_inequality 
  (a b A B : ℝ) 
  (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) : 
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_squares_inequality_l2868_286809


namespace NUMINAMATH_CALUDE_platform_length_l2868_286805

/-- Given a train with speed 72 km/hr and length 220 m, crossing a platform in 26 seconds,
    the length of the platform is 300 m. -/
theorem platform_length (train_speed : ℝ) (train_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 →
  train_length = 220 →
  crossing_time = 26 →
  (train_speed * (5/18) * crossing_time) - train_length = 300 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l2868_286805


namespace NUMINAMATH_CALUDE_car_speed_problem_l2868_286898

theorem car_speed_problem (distance : ℝ) (speed_difference : ℝ) (time_difference : ℝ) :
  distance = 300 →
  speed_difference = 10 →
  time_difference = 2 →
  ∃ (speed_r : ℝ),
    speed_r > 0 ∧
    distance / speed_r - time_difference = distance / (speed_r + speed_difference) ∧
    speed_r = 30 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2868_286898


namespace NUMINAMATH_CALUDE_N_bounds_l2868_286884

/-- The number of divisors of n -/
def d (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The number of ordered pairs (x,y) satisfying the given conditions -/
def N (p : ℕ) : ℕ := (Finset.filter (fun pair : ℕ × ℕ =>
  1 ≤ pair.1 ∧ pair.1 ≤ p * (p - 1) ∧
  1 ≤ pair.2 ∧ pair.2 ≤ p * (p - 1) ∧
  (pair.1 ^ pair.2) % p = 1 ∧
  (pair.2 ^ pair.1) % p = 1
) (Finset.product (Finset.range (p * (p - 1) + 1)) (Finset.range (p * (p - 1) + 1)))).card

theorem N_bounds (p : ℕ) (h : Nat.Prime p) :
  (Nat.totient (p - 1) * d (p - 1))^2 ≤ N p ∧ N p ≤ ((p - 1) * d (p - 1))^2 := by
  sorry

end NUMINAMATH_CALUDE_N_bounds_l2868_286884


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2868_286828

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (h : a > b) :
  (a + b) / 2 > Real.sqrt (a * b) ∧
  (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2868_286828


namespace NUMINAMATH_CALUDE_repeating_decimal_reciprocal_l2868_286873

/-- The repeating decimal 0.363636... as a rational number -/
def repeating_decimal : ℚ := 4 / 11

/-- The reciprocal of the repeating decimal 0.363636... -/
def reciprocal : ℚ := 11 / 4

theorem repeating_decimal_reciprocal :
  (repeating_decimal)⁻¹ = reciprocal := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_reciprocal_l2868_286873


namespace NUMINAMATH_CALUDE_opposite_of_83_is_84_l2868_286802

/-- Represents a circle with 100 equally spaced points numbered from 1 to 100. -/
structure NumberedCircle where
  numbers : Fin 100 → Fin 100
  bijective : Function.Bijective numbers

/-- Checks if numbers less than k are evenly distributed on both sides of the diameter through k. -/
def evenlyDistributed (c : NumberedCircle) (k : Fin 100) : Prop :=
  ∀ m < k, (c.numbers m < k ∧ c.numbers (m + 50) ≥ k) ∨
           (c.numbers m ≥ k ∧ c.numbers (m + 50) < k)

/-- The main theorem stating that if numbers are evenly distributed for all k,
    then the number opposite to 83 is 84. -/
theorem opposite_of_83_is_84 (c : NumberedCircle) 
  (h : ∀ k, evenlyDistributed c k) : 
  c.numbers (Fin.ofNat 33) = Fin.ofNat 84 :=
sorry

end NUMINAMATH_CALUDE_opposite_of_83_is_84_l2868_286802


namespace NUMINAMATH_CALUDE_hyperbola_focus_k_value_l2868_286850

/-- Given a hyperbola with equation x^2 - ky^2 = 1 and one focus at (3,0), prove that k = 1/8 -/
theorem hyperbola_focus_k_value (k : ℝ) : 
  (∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 - k*(y t)^2 = 1) →  -- Hyperbola equation
  (∃ (x₀ y₀ : ℝ), x₀^2 - k*y₀^2 = 1 ∧ x₀ = 3 ∧ y₀ = 0) →  -- Focus at (3,0)
  k = 1/8 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_k_value_l2868_286850


namespace NUMINAMATH_CALUDE_union_of_sets_l2868_286827

theorem union_of_sets (M N : Set ℕ) (hM : M = {1, 2}) (hN : N = {2, 3}) :
  M ∪ N = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l2868_286827


namespace NUMINAMATH_CALUDE_simple_interest_time_calculation_l2868_286881

theorem simple_interest_time_calculation (P : ℝ) (h1 : P > 0) : ∃ T : ℝ,
  (P * 5 * T) / 100 = P / 5 ∧ T = 4 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_time_calculation_l2868_286881


namespace NUMINAMATH_CALUDE_point_movement_l2868_286846

theorem point_movement (A B : ℝ × ℝ) : 
  A = (-3, 2) → 
  B.1 = A.1 + 1 → 
  B.2 = A.2 - 2 → 
  B = (-2, 0) := by
sorry

end NUMINAMATH_CALUDE_point_movement_l2868_286846


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l2868_286831

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ := a^2 + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- Theorem statement
theorem diamond_equation_solution :
  ∃ h : ℝ, diamond 3 h = 12 ∧ h = 6 := by sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l2868_286831


namespace NUMINAMATH_CALUDE_total_profit_calculation_l2868_286800

-- Define the investments and A's profit share
def investment_A : ℕ := 6300
def investment_B : ℕ := 4200
def investment_C : ℕ := 10500
def A_profit_share : ℕ := 4080

-- Define the total investment
def total_investment : ℕ := investment_A + investment_B + investment_C

-- Define A's investment ratio
def A_investment_ratio : ℚ := investment_A / total_investment

-- Theorem to prove
theorem total_profit_calculation : 
  ∃ (total_profit : ℕ), 
    (A_investment_ratio * total_profit = A_profit_share) ∧
    (total_profit = 13600) :=
by sorry

end NUMINAMATH_CALUDE_total_profit_calculation_l2868_286800


namespace NUMINAMATH_CALUDE_total_votes_polled_l2868_286825

-- Define the total number of votes
variable (V : ℝ)

-- Define the number of votes for each candidate
variable (T S R F : ℝ)

-- Define the conditions
def condition1 : Prop := T = S + 0.15 * V
def condition2 : Prop := S = R + 0.05 * V
def condition3 : Prop := R = F + 0.07 * V
def condition4 : Prop := T + S + R + F = V
def condition5 : Prop := T - 2500 - 2000 = S + 2500
def condition6 : Prop := S + 2500 = R + 2000 + 0.05 * V

-- State the theorem
theorem total_votes_polled
  (h1 : condition1 V T S)
  (h2 : condition2 V S R)
  (h3 : condition3 V R F)
  (h4 : condition4 V T S R F)
  (h5 : condition5 T S)
  (h6 : condition6 V S R) :
  V = 30000 := by
  sorry


end NUMINAMATH_CALUDE_total_votes_polled_l2868_286825


namespace NUMINAMATH_CALUDE_num_keepers_is_correct_l2868_286822

/-- The number of keepers in a caravan with hens, goats, and camels. -/
def num_keepers : ℕ :=
  let num_hens : ℕ := 50
  let num_goats : ℕ := 45
  let num_camels : ℕ := 8
  let hen_feet : ℕ := 2
  let goat_feet : ℕ := 4
  let camel_feet : ℕ := 4
  let keeper_feet : ℕ := 2
  let total_animal_feet : ℕ := num_hens * hen_feet + num_goats * goat_feet + num_camels * camel_feet
  let total_animal_heads : ℕ := num_hens + num_goats + num_camels
  let extra_feet : ℕ := 224
  15

theorem num_keepers_is_correct : num_keepers = 15 := by
  sorry

#eval num_keepers

end NUMINAMATH_CALUDE_num_keepers_is_correct_l2868_286822


namespace NUMINAMATH_CALUDE_min_sum_squares_l2868_286880

theorem min_sum_squares (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_3 : a + b + c = 3) : 
  ∀ m : ℝ, m = a^2 + b^2 + c^2 → m ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2868_286880


namespace NUMINAMATH_CALUDE_sqrt_a_minus_two_real_l2868_286803

theorem sqrt_a_minus_two_real (a : ℝ) : (∃ x : ℝ, x^2 = a - 2) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_minus_two_real_l2868_286803


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_d_value_l2868_286890

/-- 
Given a polynomial of the form 3x^3 + dx + 9 with a factor x^2 + qx + 1,
prove that d = -24.
-/
theorem polynomial_factor_implies_d_value :
  ∀ d q : ℝ,
  (∃ c : ℝ, ∀ x : ℝ, 3*x^3 + d*x + 9 = (x^2 + q*x + 1) * (3*x + c)) →
  d = -24 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_d_value_l2868_286890


namespace NUMINAMATH_CALUDE_computation_problems_count_l2868_286861

theorem computation_problems_count (total_problems : ℕ) (comp_points : ℕ) (word_points : ℕ) (total_points : ℕ) :
  total_problems = 30 →
  comp_points = 3 →
  word_points = 5 →
  total_points = 110 →
  ∃ (comp_count : ℕ) (word_count : ℕ),
    comp_count + word_count = total_problems ∧
    comp_count * comp_points + word_count * word_points = total_points ∧
    comp_count = 20 :=
by sorry

end NUMINAMATH_CALUDE_computation_problems_count_l2868_286861


namespace NUMINAMATH_CALUDE_raisin_cost_fraction_l2868_286893

/-- Represents the cost ratio of raisins to the total mixture -/
def cost_ratio (raisin_weight : ℚ) (nut_weight : ℚ) (nut_cost_ratio : ℚ) : ℚ :=
  (raisin_weight) / (raisin_weight + nut_weight * nut_cost_ratio)

/-- Theorem stating that the cost of raisins is 3/19 of the total mixture cost -/
theorem raisin_cost_fraction :
  cost_ratio 3 4 4 = 3 / 19 := by
sorry

end NUMINAMATH_CALUDE_raisin_cost_fraction_l2868_286893


namespace NUMINAMATH_CALUDE_equation_describes_cone_l2868_286826

/-- Spherical coordinates -/
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Definition of a cone in spherical coordinates -/
def IsCone (c : ℝ) (f : SphericalCoordinates → Prop) : Prop :=
  ∀ p : SphericalCoordinates, f p ↔ p.ρ = c * Real.sin p.φ

/-- The main theorem: the equation ρ = c * sin φ describes a cone -/
theorem equation_describes_cone (c : ℝ) (hc : c > 0) :
  IsCone c (fun p => p.ρ = c * Real.sin p.φ) :=
sorry

end NUMINAMATH_CALUDE_equation_describes_cone_l2868_286826


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2868_286835

def f (a : ℝ) (x : ℝ) : ℝ := (x - a)^2

theorem sufficient_not_necessary_condition :
  (∀ a : ℝ, a = 1 → (∀ x y : ℝ, 1 < x → x < y → f a x < f a y)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ (∀ x y : ℝ, 1 < x → x < y → f a x < f a y)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2868_286835


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_two_l2868_286895

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q ∧ a n > 0

/-- Theorem: For a geometric sequence with positive terms satisfying 2a_1 + a_2 = a_3, 
    the common ratio is 2 -/
theorem geometric_sequence_ratio_two (a : ℕ → ℝ) (q : ℝ) 
    (h_geom : GeometricSequence a q)
    (h_eq : 2 * a 1 + a 2 = a 3) : q = 2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_ratio_two_l2868_286895


namespace NUMINAMATH_CALUDE_project_estimated_hours_l2868_286833

/-- The number of extra hours Anie needs to work each day -/
def extra_hours : ℕ := 5

/-- The number of hours in Anie's normal work schedule each day -/
def normal_hours : ℕ := 10

/-- The number of days it would take Anie to finish the job -/
def days_to_finish : ℕ := 100

/-- The total number of hours Anie works each day -/
def total_hours_per_day : ℕ := normal_hours + extra_hours

/-- Theorem: The project is estimated to take 1500 hours -/
theorem project_estimated_hours : 
  days_to_finish * total_hours_per_day = 1500 := by
  sorry


end NUMINAMATH_CALUDE_project_estimated_hours_l2868_286833


namespace NUMINAMATH_CALUDE_prime_product_sum_relation_l2868_286899

theorem prime_product_sum_relation : 
  ∀ (a b c d : ℕ), 
    Prime a → Prime b → Prime c → Prime d →
    a * b * c * d = 11 * (a + b + c + d) →
    a + b + c + d = 20 := by
sorry

end NUMINAMATH_CALUDE_prime_product_sum_relation_l2868_286899


namespace NUMINAMATH_CALUDE_intensity_after_three_plates_l2868_286814

/-- The intensity of light after passing through a number of glass plates -/
def intensity (a : ℝ) (n : ℕ) : ℝ :=
  a * (0.9 ^ n)

/-- Theorem: The intensity of light with original intensity a after passing through 3 glass plates is 0.729a -/
theorem intensity_after_three_plates (a : ℝ) :
  intensity a 3 = 0.729 * a := by
  sorry

end NUMINAMATH_CALUDE_intensity_after_three_plates_l2868_286814


namespace NUMINAMATH_CALUDE_arc_length_ninety_degrees_radius_three_l2868_286851

/-- The arc length of a sector with a central angle of 90° and a radius of 3 is equal to (3/2)π. -/
theorem arc_length_ninety_degrees_radius_three :
  let central_angle : ℝ := 90
  let radius : ℝ := 3
  let arc_length : ℝ := (central_angle * π * radius) / 180
  arc_length = (3/2) * π := by sorry

end NUMINAMATH_CALUDE_arc_length_ninety_degrees_radius_three_l2868_286851


namespace NUMINAMATH_CALUDE_part_one_part_two_l2868_286815

/-- The function f(x) = |a-4x| + |2a+x| -/
def f (a : ℝ) (x : ℝ) : ℝ := |a - 4*x| + |2*a + x|

/-- Part I: When a = 1, f(x) ≥ 3 if and only if x ≤ 0 or x ≥ 2/5 -/
theorem part_one : 
  ∀ x : ℝ, f 1 x ≥ 3 ↔ x ≤ 0 ∨ x ≥ 2/5 := by sorry

/-- Part II: For all x ≠ 0 and all a, f(x) + f(-1/x) ≥ 10 -/
theorem part_two : 
  ∀ a x : ℝ, x ≠ 0 → f a x + f a (-1/x) ≥ 10 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2868_286815


namespace NUMINAMATH_CALUDE_shekars_english_marks_l2868_286866

/-- Represents the marks scored in each subject -/
structure Marks where
  mathematics : ℕ
  science : ℕ
  socialStudies : ℕ
  biology : ℕ
  english : ℕ

/-- Calculates the average of a list of natural numbers -/
def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

/-- Theorem: Given Shekar's marks in other subjects and his average, his English marks are 67 -/
theorem shekars_english_marks (m : Marks) (h1 : m.mathematics = 76) (h2 : m.science = 65)
    (h3 : m.socialStudies = 82) (h4 : m.biology = 75)
    (h5 : average [m.mathematics, m.science, m.socialStudies, m.biology, m.english] = 73) :
    m.english = 67 := by
  sorry

#check shekars_english_marks

end NUMINAMATH_CALUDE_shekars_english_marks_l2868_286866


namespace NUMINAMATH_CALUDE_g_zero_value_l2868_286847

-- Define polynomials f, g, and h
variable (f g h : ℝ[X])

-- Define the relationship between h, f, and g
axiom h_eq_f_mul_g : h = f * g

-- Define the constant term of f
axiom f_const_term : f.coeff 0 = 6

-- Define the constant term of h
axiom h_const_term : h.coeff 0 = -18

-- Theorem to prove
theorem g_zero_value : g.coeff 0 = -3 := by sorry

end NUMINAMATH_CALUDE_g_zero_value_l2868_286847


namespace NUMINAMATH_CALUDE_min_jumps_proof_l2868_286848

/-- The distance of each jump in millimeters -/
def jump_distance : ℝ := 19

/-- The distance between points A and B in centimeters -/
def total_distance : ℝ := 1812

/-- The minimum number of jumps required -/
def min_jumps : ℕ := 954

/-- Theorem stating the minimum number of jumps required -/
theorem min_jumps_proof :
  ∃ (n : ℕ), n = min_jumps ∧ 
  (n : ℝ) * jump_distance ≥ total_distance * 10 ∧
  ∀ (m : ℕ), (m : ℝ) * jump_distance ≥ total_distance * 10 → m ≥ n :=
sorry

end NUMINAMATH_CALUDE_min_jumps_proof_l2868_286848


namespace NUMINAMATH_CALUDE_triangle_height_l2868_286894

theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 9.31 → base = 4.9 → height = (2 * area) / base → height = 3.8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l2868_286894


namespace NUMINAMATH_CALUDE_sequence_properties_l2868_286879

theorem sequence_properties (a : ℕ+ → ℤ) (S : ℕ+ → ℤ) :
  (∀ n : ℕ+, S n = -n.val^2 + 24*n.val) →
  (∀ n : ℕ+, a n = S n - S (n-1)) →
  (∀ n : ℕ+, a n = -2*n.val + 25) ∧
  (∀ n : ℕ+, S n ≤ S 12) ∧
  (S 12 = 144) := by
sorry

end NUMINAMATH_CALUDE_sequence_properties_l2868_286879


namespace NUMINAMATH_CALUDE_function_symmetry_l2868_286854

/-- Given a function f(x) = ax³ + bx + c*sin(x) - 2 where f(-2) = 8, prove that f(2) = -12 -/
theorem function_symmetry (a b c : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^3 + b * x + c * Real.sin x - 2
  f (-2) = 8 → f 2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l2868_286854


namespace NUMINAMATH_CALUDE_jerry_shelf_difference_l2868_286852

/-- Calculates the difference between action figures and books on Jerry's shelf -/
def action_figure_book_difference (
  initial_books : ℕ
  ) (initial_action_figures : ℕ)
  (added_action_figures : ℕ) : ℕ :=
  (initial_action_figures + added_action_figures) - initial_books

/-- Proves that the difference between action figures and books is 3 -/
theorem jerry_shelf_difference :
  action_figure_book_difference 3 4 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jerry_shelf_difference_l2868_286852


namespace NUMINAMATH_CALUDE_integer_partition_impossibility_l2868_286839

theorem integer_partition_impossibility : 
  ¬ (∃ (A B C : Set Int), 
    (∀ (n : Int), n ∈ A ∨ n ∈ B ∨ n ∈ C) ∧ 
    (A ∪ B ∪ C = Set.univ) ∧
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅) ∧
    (∀ (n : Int), 
      ((n ∈ A ∧ (n - 50) ∈ B ∧ (n + 1987) ∈ C) ∨
       (n ∈ A ∧ (n - 50) ∈ C ∧ (n + 1987) ∈ B) ∨
       (n ∈ B ∧ (n - 50) ∈ A ∧ (n + 1987) ∈ C) ∨
       (n ∈ B ∧ (n - 50) ∈ C ∧ (n + 1987) ∈ A) ∨
       (n ∈ C ∧ (n - 50) ∈ A ∧ (n + 1987) ∈ B) ∨
       (n ∈ C ∧ (n - 50) ∈ B ∧ (n + 1987) ∈ A)))) :=
by sorry

end NUMINAMATH_CALUDE_integer_partition_impossibility_l2868_286839


namespace NUMINAMATH_CALUDE_equation_proof_l2868_286824

theorem equation_proof : (15 : ℝ) ^ 3 * 7 ^ 4 / 5670 = 1428.75 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2868_286824


namespace NUMINAMATH_CALUDE_lees_initial_money_l2868_286891

def friends_money : ℕ := 8
def meal_cost : ℕ := 15
def total_paid : ℕ := 18

theorem lees_initial_money :
  ∃ (lees_money : ℕ), lees_money + friends_money = total_paid ∧ lees_money = 10 := by
sorry

end NUMINAMATH_CALUDE_lees_initial_money_l2868_286891


namespace NUMINAMATH_CALUDE_boat_speed_problem_l2868_286886

/-- Proves that given a lake of width 60 miles, a boat traveling at 30 mph,
    and a waiting time of 3 hours for another boat to arrive,
    the speed of the second boat is 12 mph. -/
theorem boat_speed_problem (lake_width : ℝ) (janet_speed : ℝ) (waiting_time : ℝ) :
  lake_width = 60 →
  janet_speed = 30 →
  waiting_time = 3 →
  ∃ (sister_speed : ℝ),
    sister_speed = lake_width / (lake_width / janet_speed + waiting_time) ∧
    sister_speed = 12 := by sorry

end NUMINAMATH_CALUDE_boat_speed_problem_l2868_286886


namespace NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l2868_286841

/-- Prove that sin 40° * (tan 10° - √3) = -1 -/
theorem sin_40_tan_10_minus_sqrt_3 :
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l2868_286841


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l2868_286889

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a survey with its characteristics --/
structure Survey where
  population_size : ℕ
  sample_size : ℕ
  has_subgroups : Bool
  is_uniform : Bool

/-- Determines the most appropriate sampling method for a given survey --/
def best_sampling_method (s : Survey) : SamplingMethod :=
  if s.has_subgroups then SamplingMethod.Stratified
  else if s.is_uniform && s.population_size > s.sample_size * 10 then SamplingMethod.Systematic
  else SamplingMethod.SimpleRandom

/-- The three surveys described in the problem --/
def survey1 : Survey := { population_size := 10, sample_size := 3, has_subgroups := false, is_uniform := true }
def survey2 : Survey := { population_size := 32 * 40, sample_size := 32, has_subgroups := false, is_uniform := true }
def survey3 : Survey := { population_size := 160, sample_size := 20, has_subgroups := true, is_uniform := false }

theorem correct_sampling_methods :
  best_sampling_method survey1 = SamplingMethod.SimpleRandom ∧
  best_sampling_method survey2 = SamplingMethod.Systematic ∧
  best_sampling_method survey3 = SamplingMethod.Stratified :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l2868_286889


namespace NUMINAMATH_CALUDE_complex_multiplication_l2868_286823

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2868_286823


namespace NUMINAMATH_CALUDE_quadratic_root_coefficients_l2868_286801

theorem quadratic_root_coefficients :
  ∀ (r s : ℝ),
  (∃ x : ℂ, 3 * x^2 + r * x + s = 0 ∧ x = 2 + Complex.I * Real.sqrt 3) →
  r = -12 ∧ s = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_coefficients_l2868_286801


namespace NUMINAMATH_CALUDE_expression_evaluation_l2868_286813

theorem expression_evaluation :
  let a : ℚ := 1/3
  let b : ℤ := -1
  4 * (3 * a^2 * b - a * b^2) - (2 * a * b^2 + 3 * a^2 * b) = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2868_286813


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_l2868_286887

/-- Given an ellipse with equation x²/16 + y²/9 = 1, 
    the slope of any chord with midpoint (1,2) is -9/32 -/
theorem ellipse_chord_slope :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  (x₁^2 / 16 + y₁^2 / 9 = 1) →
  (x₂^2 / 16 + y₂^2 / 9 = 1) →
  ((x₁ + x₂) / 2 = 1) →
  ((y₁ + y₂) / 2 = 2) →
  (y₂ - y₁) / (x₂ - x₁) = -9/32 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_l2868_286887


namespace NUMINAMATH_CALUDE_pure_imaginary_a_value_l2868_286818

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The complex number z -/
def z (a : ℝ) : ℂ := (a + i) * (3 + 2*i)

/-- A complex number is pure imaginary if its real part is zero -/
def is_pure_imaginary (c : ℂ) : Prop := c.re = 0 ∧ c.im ≠ 0

theorem pure_imaginary_a_value :
  ∃ (a : ℝ), is_pure_imaginary (z a) ∧ a = 2/3 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_a_value_l2868_286818


namespace NUMINAMATH_CALUDE_range_of_m_l2868_286875

-- Define a decreasing function on (-∞, 0)
def DecreasingOnNegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y < 0 → f x > f y

-- Define the theorem
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h1 : DecreasingOnNegative f) 
  (h2 : f (1 - m) < f (m - 3)) : 
  1 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2868_286875


namespace NUMINAMATH_CALUDE_max_alternations_theorem_l2868_286892

/-- Represents a painter's strategy for painting fence sections -/
def PainterStrategy := ℕ → Bool

/-- Represents the state of the fence after painting -/
def FenceState := List Bool

/-- Counts the number of color alternations in a fence state -/
def countAlternations (fence : FenceState) : ℕ := sorry

/-- Simulates the painting process and returns the final fence state -/
def paintFence (strategy1 strategy2 : PainterStrategy) : FenceState := sorry

/-- The maximum number of alternations the first painter can guarantee -/
def maxGuaranteedAlternations : ℕ := sorry

/-- Theorem stating the maximum number of alternations the first painter can guarantee -/
theorem max_alternations_theorem :
  ∀ (strategy2 : PainterStrategy),
  ∃ (strategy1 : PainterStrategy),
  countAlternations (paintFence strategy1 strategy2) ≥ 49 ∧
  maxGuaranteedAlternations = 49 := by sorry

end NUMINAMATH_CALUDE_max_alternations_theorem_l2868_286892


namespace NUMINAMATH_CALUDE_expression_simplification_l2868_286877

theorem expression_simplification (a b : ℝ) : 
  (8 * a^3 * b) * (4 * a * b^2) * (1 / (2 * a * b)^3) = 4 * a := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2868_286877


namespace NUMINAMATH_CALUDE_larger_number_proof_l2868_286832

theorem larger_number_proof (x y : ℝ) : 
  y > x → 4 * y = 3 * x → y - x = 12 → y = -36 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2868_286832


namespace NUMINAMATH_CALUDE_largest_decimal_l2868_286862

theorem largest_decimal : 
  let a := 0.987
  let b := 0.9861
  let c := 0.98709
  let d := 0.968
  let e := 0.96989
  (c ≥ a) ∧ (c ≥ b) ∧ (c ≥ d) ∧ (c ≥ e) := by
  sorry

end NUMINAMATH_CALUDE_largest_decimal_l2868_286862


namespace NUMINAMATH_CALUDE_x_intercepts_difference_l2868_286871

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the x-intercepts
variable (x₁ x₂ x₃ x₄ : ℝ)

-- State the conditions
axiom g_def : ∀ x, g x = 2 * f (200 - x)
axiom vertex_condition : ∃ v, f v = 0 ∧ g v = 0
axiom x_intercepts_order : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄
axiom x_intercepts_f : f x₁ = 0 ∧ f x₄ = 0
axiom x_intercepts_g : g x₂ = 0 ∧ g x₃ = 0
axiom x_diff : x₃ - x₂ = 300

-- Theorem to prove
theorem x_intercepts_difference : x₄ - x₁ = 600 := by sorry

end NUMINAMATH_CALUDE_x_intercepts_difference_l2868_286871


namespace NUMINAMATH_CALUDE_henry_book_count_l2868_286810

/-- Calculates the number of books Henry has after donating and picking up new books -/
def final_book_count (initial_books : ℕ) (box_count : ℕ) (books_per_box : ℕ) 
  (room_books : ℕ) (coffee_table_books : ℕ) (kitchen_books : ℕ) (new_books : ℕ) : ℕ :=
  initial_books - (box_count * books_per_box + room_books + coffee_table_books + kitchen_books) + new_books

/-- Theorem stating that Henry ends up with 23 books -/
theorem henry_book_count : 
  final_book_count 99 3 15 21 4 18 12 = 23 := by
  sorry

end NUMINAMATH_CALUDE_henry_book_count_l2868_286810


namespace NUMINAMATH_CALUDE_sum_of_five_cubes_l2868_286819

theorem sum_of_five_cubes (n : ℤ) : ∃ (a b c d e : ℤ), n = a^3 + b^3 + c^3 + d^3 + e^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_cubes_l2868_286819


namespace NUMINAMATH_CALUDE_bouncy_balls_per_package_l2868_286885

/-- The number of bouncy balls in each package -/
def balls_per_package : ℝ := 10

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs : ℝ := 8.0

/-- The number of packs of green bouncy balls Maggie gave away -/
def green_packs_given : ℝ := 4.0

/-- The number of packs of green bouncy balls Maggie bought -/
def green_packs_bought : ℝ := 4.0

/-- The total number of bouncy balls Maggie kept -/
def total_balls : ℕ := 80

theorem bouncy_balls_per_package :
  yellow_packs * balls_per_package = total_balls := by sorry

end NUMINAMATH_CALUDE_bouncy_balls_per_package_l2868_286885


namespace NUMINAMATH_CALUDE_no_solutions_lcm_gcd_equation_l2868_286836

theorem no_solutions_lcm_gcd_equation :
  ¬∃ (n : ℕ), n > 0 ∧ Nat.lcm n 150 = Nat.gcd n 150 + 600 := by
sorry

end NUMINAMATH_CALUDE_no_solutions_lcm_gcd_equation_l2868_286836


namespace NUMINAMATH_CALUDE_jhons_total_pay_l2868_286856

/-- Calculates the total pay for a worker given their work schedule and pay rates. -/
def calculate_total_pay (total_days : ℕ) (present_days : ℕ) (present_rate : ℚ) (absent_rate : ℚ) : ℚ :=
  let absent_days := total_days - present_days
  present_days * present_rate + absent_days * absent_rate

/-- Proves that Jhon's total pay is $320.00 given the specified conditions. -/
theorem jhons_total_pay :
  calculate_total_pay 60 35 7 3 = 320 := by
  sorry

end NUMINAMATH_CALUDE_jhons_total_pay_l2868_286856
