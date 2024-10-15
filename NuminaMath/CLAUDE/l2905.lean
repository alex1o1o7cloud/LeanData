import Mathlib

namespace NUMINAMATH_CALUDE_wren_population_decline_l2905_290580

theorem wren_population_decline (n : ℕ) : (∀ k : ℕ, k < n → (0.7 : ℝ) ^ k ≥ 0.1) ∧ (0.7 : ℝ) ^ n < 0.1 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_wren_population_decline_l2905_290580


namespace NUMINAMATH_CALUDE_probability_divisible_by_2_3_5_or_7_l2905_290564

theorem probability_divisible_by_2_3_5_or_7 : 
  let S : Finset ℕ := Finset.range 120
  let A : Finset ℕ := S.filter (fun n => n % 2 = 0)
  let B : Finset ℕ := S.filter (fun n => n % 3 = 0)
  let C : Finset ℕ := S.filter (fun n => n % 5 = 0)
  let D : Finset ℕ := S.filter (fun n => n % 7 = 0)
  (A ∪ B ∪ C ∪ D).card / S.card = 13 / 15 := by
sorry


end NUMINAMATH_CALUDE_probability_divisible_by_2_3_5_or_7_l2905_290564


namespace NUMINAMATH_CALUDE_felix_axe_sharpening_cost_l2905_290538

/-- Calculates the total cost of axe sharpening given the number of trees chopped,
    trees per sharpening, and cost per sharpening. -/
def axeSharpeningCost (treesChopped : ℕ) (treesPerSharpening : ℕ) (costPerSharpening : ℕ) : ℕ :=
  ((treesChopped - 1) / treesPerSharpening + 1) * costPerSharpening

/-- Proves that given the conditions, the total cost of axe sharpening is $35. -/
theorem felix_axe_sharpening_cost :
  ∀ (treesChopped : ℕ),
    treesChopped ≥ 91 →
    axeSharpeningCost treesChopped 13 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_felix_axe_sharpening_cost_l2905_290538


namespace NUMINAMATH_CALUDE_A₁_Aₒ₂_independent_l2905_290563

/-- A bag containing black and white balls -/
structure Bag where
  black : ℕ
  white : ℕ

/-- An event in the probability space of drawing balls from the bag -/
structure Event (bag : Bag) where
  prob : ℝ
  nonneg : 0 ≤ prob
  le_one : prob ≤ 1

/-- Drawing a ball from the bag with replacement -/
def draw (bag : Bag) : Event bag := sorry

/-- The event of drawing a black ball -/
def black_ball (bag : Bag) : Event bag := sorry

/-- The event of drawing a white ball -/
def white_ball (bag : Bag) : Event bag := sorry

/-- The probability of an event -/
def P (bag : Bag) (e : Event bag) : ℝ := e.prob

/-- Two events are independent if the probability of their intersection
    is equal to the product of their individual probabilities -/
def independent (bag : Bag) (e1 e2 : Event bag) : Prop :=
  P bag (draw bag) = P bag e1 * P bag e2

/-- A₁: The event of drawing a black ball on the first draw -/
def A₁ (bag : Bag) : Event bag := black_ball bag

/-- A₂: The event of drawing a black ball on the second draw -/
def A₂ (bag : Bag) : Event bag := black_ball bag

/-- Aₒ₂: The complement of A₂ (drawing a white ball on the second draw) -/
def Aₒ₂ (bag : Bag) : Event bag := white_ball bag

/-- Theorem: A₁ and Aₒ₂ are independent events when drawing with replacement -/
theorem A₁_Aₒ₂_independent (bag : Bag) : independent bag (A₁ bag) (Aₒ₂ bag) := by
  sorry

end NUMINAMATH_CALUDE_A₁_Aₒ₂_independent_l2905_290563


namespace NUMINAMATH_CALUDE_plant_original_price_l2905_290588

/-- Given a 10% discount on a plant and a final price of $9, prove that the original price was $10. -/
theorem plant_original_price (discount_percentage : ℚ) (discounted_price : ℚ) : 
  discount_percentage = 10 →
  discounted_price = 9 →
  (1 - discount_percentage / 100) * 10 = discounted_price := by
  sorry

end NUMINAMATH_CALUDE_plant_original_price_l2905_290588


namespace NUMINAMATH_CALUDE_pet_shop_total_cost_l2905_290536

/-- Represents the cost of purchasing all pets in a pet shop given specific conditions. -/
def total_cost_of_pets (num_puppies num_kittens num_parakeets : ℕ) 
  (parakeet_cost : ℚ) 
  (puppy_parakeet_ratio kitten_parakeet_ratio : ℚ) : ℚ :=
  let puppy_cost := puppy_parakeet_ratio * parakeet_cost
  let kitten_cost := kitten_parakeet_ratio * parakeet_cost
  (num_puppies : ℚ) * puppy_cost + (num_kittens : ℚ) * kitten_cost + (num_parakeets : ℚ) * parakeet_cost

/-- Theorem stating that under given conditions, the total cost of pets is $130. -/
theorem pet_shop_total_cost : 
  total_cost_of_pets 2 2 3 10 3 2 = 130 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_total_cost_l2905_290536


namespace NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l2905_290511

theorem arithmetic_progression_of_primes (a : ℕ → ℕ) (d : ℕ) :
  (∀ i ∈ Finset.range 15, Nat.Prime (a i)) →
  (∀ i ∈ Finset.range 14, a (i + 1) = a i + d) →
  d > 0 →
  a 0 > 15 →
  d > 30000 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l2905_290511


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l2905_290529

theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := fun (x : ℝ) ↦ a^x + 1
  f 0 = 2 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l2905_290529


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l2905_290581

theorem quadratic_inequality_properties (a b c : ℝ) :
  (∀ x, (a * x^2 + b * x + c ≤ 0) ↔ (x ≤ -2 ∨ x ≥ 3)) →
  (a < 0 ∧
   (∀ x, (a * x + c > 0) ↔ x < 6) ∧
   (∀ x, (c * x^2 + b * x + a < 0) ↔ (-1/2 < x ∧ x < 1/3))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l2905_290581


namespace NUMINAMATH_CALUDE_trains_meeting_time_l2905_290595

/-- The time taken for two trains to meet under specific conditions -/
theorem trains_meeting_time : 
  let train1_length : ℝ := 300
  let train1_crossing_time : ℝ := 20
  let train2_length : ℝ := 450
  let train2_speed_kmh : ℝ := 90
  let train1_speed : ℝ := train1_length / train1_crossing_time
  let train2_speed : ℝ := train2_speed_kmh * 1000 / 3600
  let relative_speed : ℝ := train1_speed + train2_speed
  let total_distance : ℝ := train1_length + train2_length
  let meeting_time : ℝ := total_distance / relative_speed
  meeting_time = 18.75 := by sorry

end NUMINAMATH_CALUDE_trains_meeting_time_l2905_290595


namespace NUMINAMATH_CALUDE_odd_function_extrema_l2905_290590

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

-- State the theorem
theorem odd_function_extrema :
  ∀ (a b c : ℝ),
  (∀ x, f a b c (-x) = -(f a b c x)) →  -- f is odd
  (f a b c 1 = 2) →                     -- maximum value of 2 at x = 1
  (∀ x, f a b c x ≤ f a b c 1) →        -- global maximum at x = 1
  (∃ (f_max f_min : ℝ),
    (∀ x ∈ Set.Icc (-4) 3, f (-1) 0 3 x ≤ f_max) ∧
    (∀ x ∈ Set.Icc (-4) 3, f_min ≤ f (-1) 0 3 x) ∧
    f_max = 52 ∧ f_min = -18) :=
by sorry


end NUMINAMATH_CALUDE_odd_function_extrema_l2905_290590


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l2905_290560

theorem zeros_before_first_nonzero_digit (n : ℕ) (d : ℕ) (h : d = 64000) :
  (∃ k : ℕ, (7 : ℚ) / d = k / 10^(n + 1) ∧ k % 10 ≠ 0 ∧ k < 10^n) → n = 4 :=
sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l2905_290560


namespace NUMINAMATH_CALUDE_valid_y_characterization_l2905_290598

/-- The set of y values in [0, 2π] for which sin(x+y) ≥ sin(x) - sin(y) holds for all x in [0, 2π] -/
def valid_y_set : Set ℝ :=
  {y | 0 ≤ y ∧ y ≤ 2 * Real.pi ∧ 
    ∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi → Real.sin (x + y) ≥ Real.sin x - Real.sin y}

theorem valid_y_characterization :
  valid_y_set = {0, 2 * Real.pi} := by sorry

end NUMINAMATH_CALUDE_valid_y_characterization_l2905_290598


namespace NUMINAMATH_CALUDE_teal_survey_result_l2905_290591

/-- Represents the survey results about teal color perception -/
structure TealSurvey where
  total : ℕ
  more_blue : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the number of people who believe teal is "more green" -/
def more_green (survey : TealSurvey) : ℕ :=
  survey.total - (survey.more_blue - survey.both) - survey.both - survey.neither

/-- Theorem stating the result of the teal color survey -/
theorem teal_survey_result : 
  let survey : TealSurvey := {
    total := 150,
    more_blue := 90,
    both := 40,
    neither := 20
  }
  more_green survey = 80 := by sorry

end NUMINAMATH_CALUDE_teal_survey_result_l2905_290591


namespace NUMINAMATH_CALUDE_simplify_expression_l2905_290571

theorem simplify_expression : 2 - (2 / (2 + 2 * Real.sqrt 2)) + (2 / (2 - 2 * Real.sqrt 2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2905_290571


namespace NUMINAMATH_CALUDE_parabola_shift_correct_l2905_290516

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = 4x^2 -/
def original_parabola : Parabola := { a := 4, b := 0, c := 0 }

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h
  , c := p.a * h^2 + p.c + v }

/-- The resulting parabola after shifting -/
def shifted_parabola : Parabola := shift_parabola original_parabola 9 6

theorem parabola_shift_correct :
  shifted_parabola = { a := 4, b := -72, c := 330 } := by sorry

end NUMINAMATH_CALUDE_parabola_shift_correct_l2905_290516


namespace NUMINAMATH_CALUDE_expansion_equality_l2905_290549

theorem expansion_equality (x : ℝ) : (x - 2) * (x + 2) * (x^2 + 4) = x^4 - 16 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l2905_290549


namespace NUMINAMATH_CALUDE_total_games_theorem_l2905_290503

/-- The total number of games played by Frankie and Carla -/
def total_games (carla_games frankie_games : ℕ) : ℕ := carla_games + frankie_games

/-- Theorem: Given that Carla won 20 games and Frankie won half as many games as Carla,
    the total number of games played is 30. -/
theorem total_games_theorem :
  ∀ (carla_games frankie_games : ℕ),
    carla_games = 20 →
    frankie_games = carla_games / 2 →
    total_games carla_games frankie_games = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_games_theorem_l2905_290503


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l2905_290512

theorem regular_polygon_interior_angle_sum 
  (n : ℕ) 
  (h_exterior : (360 : ℝ) / n = 45) : 
  (n - 2) * 180 = 1080 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l2905_290512


namespace NUMINAMATH_CALUDE_book_purchase_with_discount_l2905_290546

/-- Calculates the total cost of books with a discount applied -/
theorem book_purchase_with_discount 
  (book_price : ℝ) 
  (quantity : ℕ) 
  (discount_per_book : ℝ) 
  (h1 : book_price = 5) 
  (h2 : quantity = 10) 
  (h3 : discount_per_book = 0.5) : 
  (book_price - discount_per_book) * quantity = 45 := by
sorry

end NUMINAMATH_CALUDE_book_purchase_with_discount_l2905_290546


namespace NUMINAMATH_CALUDE_candy_distribution_l2905_290514

theorem candy_distribution (total_candy : ℕ) (family_members : ℕ) 
  (h1 : total_candy = 45) (h2 : family_members = 5) : 
  total_candy % family_members = 0 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2905_290514


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l2905_290500

theorem existence_of_special_integers : ∃ (a b c : ℤ), 
  (a > 2011) ∧ (b > 2011) ∧ (c > 2011) ∧
  ∃ (n : ℕ), (((a + Real.sqrt b)^c : ℝ) / 10000 - n : ℝ) = 0.20102011 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l2905_290500


namespace NUMINAMATH_CALUDE_variance_transformation_l2905_290553

/-- Given a sample of 10 data points, this function represents their variance. -/
def sample_variance (x : Fin 10 → ℝ) : ℝ := sorry

/-- Given a sample of 10 data points, this function represents the variance of the transformed data. -/
def transformed_variance (x : Fin 10 → ℝ) : ℝ := 
  sample_variance (fun i => 2 * x i - 1)

/-- Theorem stating the relationship between the original variance and the transformed variance. -/
theorem variance_transformation (x : Fin 10 → ℝ) 
  (h : sample_variance x = 8) : transformed_variance x = 32 := by
  sorry

end NUMINAMATH_CALUDE_variance_transformation_l2905_290553


namespace NUMINAMATH_CALUDE_work_problem_solution_l2905_290526

def work_problem (a_rate b_rate : ℝ) (combined_days : ℝ) : Prop :=
  a_rate = 2 * b_rate →
  combined_days = 6 →
  b_rate * (a_rate + b_rate)⁻¹ * combined_days = 18

theorem work_problem_solution :
  ∀ (a_rate b_rate combined_days : ℝ),
    work_problem a_rate b_rate combined_days :=
by
  sorry

end NUMINAMATH_CALUDE_work_problem_solution_l2905_290526


namespace NUMINAMATH_CALUDE_adjacent_sum_6_l2905_290525

/-- Represents a 3x3 table filled with numbers from 1 to 9 --/
def Table := Fin 3 → Fin 3 → Fin 9

/-- Checks if a table is valid according to the given conditions --/
def is_valid_table (t : Table) : Prop :=
  (∀ i j, t i j ≠ 0) ∧  -- All cells are filled
  (∀ x, ∃! i j, t i j = x) ∧  -- Each number appears exactly once
  t 0 0 = 1 ∧ t 2 0 = 2 ∧ t 0 2 = 3 ∧ t 2 2 = 4 ∧  -- Given positions
  (∃ i j, t i j = 5 ∧ 
    (t (i-1) j + t (i+1) j + t i (j-1) + t i (j+1) : ℕ) = 9)  -- Sum around 5 is 9

/-- Sum of adjacent numbers to a given position --/
def adjacent_sum (t : Table) (i j : Fin 3) : ℕ :=
  (t (i-1) j + t (i+1) j + t i (j-1) + t i (j+1) : ℕ)

/-- The main theorem --/
theorem adjacent_sum_6 (t : Table) (h : is_valid_table t) :
  ∃ i j, t i j = 6 ∧ adjacent_sum t i j = 29 :=
sorry

end NUMINAMATH_CALUDE_adjacent_sum_6_l2905_290525


namespace NUMINAMATH_CALUDE_inequality_proof_l2905_290517

theorem inequality_proof (x₁ x₂ x₃ : ℝ) 
  (h_pos₁ : x₁ > 0) (h_pos₂ : x₂ > 0) (h_pos₃ : x₃ > 0)
  (h_sum : x₁ + x₂ + x₃ = 1) : 
  x₂^2 / x₁ + x₃^2 / x₂ + x₁^2 / x₃ ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2905_290517


namespace NUMINAMATH_CALUDE_womens_average_age_l2905_290599

theorem womens_average_age (n : ℕ) (A : ℝ) (W₁ W₂ : ℝ) : 
  n = 6 ∧ 
  n * A - 10 - 12 + W₁ + W₂ = n * (A + 2) → 
  (W₁ + W₂) / 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_womens_average_age_l2905_290599


namespace NUMINAMATH_CALUDE_calculate_expression_l2905_290535

theorem calculate_expression : -3^2 + |(-5)| - 18 * (-1/3)^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2905_290535


namespace NUMINAMATH_CALUDE_quadratic_single_solution_l2905_290587

theorem quadratic_single_solution (p : ℝ) : 
  (∃! y : ℝ, 2 * y^2 - 8 * y = p) → p = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_single_solution_l2905_290587


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2905_290528

/-- Given a quadratic inequality ax^2 + bx + 1 > 0 with solution set (-1, 1/3),
    prove that a - b = -1 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x + 1 > 0 ↔ -1 < x ∧ x < 1/3) → 
  a - b = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2905_290528


namespace NUMINAMATH_CALUDE_school_play_ticket_ratio_l2905_290589

theorem school_play_ticket_ratio :
  ∀ (total_tickets student_tickets adult_tickets : ℕ),
    total_tickets = 366 →
    adult_tickets = 122 →
    total_tickets = student_tickets + adult_tickets →
    ∃ (k : ℕ), student_tickets = k * adult_tickets →
    (student_tickets : ℚ) / (adult_tickets : ℚ) = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_school_play_ticket_ratio_l2905_290589


namespace NUMINAMATH_CALUDE_group_size_calculation_l2905_290509

theorem group_size_calculation (n : ℕ) : 
  (n * 14 + 34) / (n + 1) = 16 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l2905_290509


namespace NUMINAMATH_CALUDE_expression_equals_one_l2905_290584

theorem expression_equals_one (a : ℝ) (h : a = Real.sqrt 2) : 
  ((a + 1) / (a + 2) + 1 / (a - 2)) / (2 / (a^2 - 4)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2905_290584


namespace NUMINAMATH_CALUDE_dirichlet_approximation_l2905_290522

theorem dirichlet_approximation (x : ℝ) (h_irr : Irrational x) (h_pos : 0 < x) :
  ∀ N : ℕ, ∃ p q : ℤ, N < q ∧ 0 < q ∧ |x - (p : ℝ) / (q : ℝ)| < 1 / (q : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_dirichlet_approximation_l2905_290522


namespace NUMINAMATH_CALUDE_range_of_g_minus_x_l2905_290561

def g (x : ℝ) : ℝ := x^2 - 3*x + 4

theorem range_of_g_minus_x :
  Set.range (fun x => g x - x) ∩ Set.Icc (-2 : ℝ) 2 = Set.Icc 0 16 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_minus_x_l2905_290561


namespace NUMINAMATH_CALUDE_distance_to_school_l2905_290597

def walking_speed : ℝ := 80
def travel_time : ℝ := 28

theorem distance_to_school :
  walking_speed * travel_time = 2240 := by sorry

end NUMINAMATH_CALUDE_distance_to_school_l2905_290597


namespace NUMINAMATH_CALUDE_account_balance_after_transfer_l2905_290531

/-- Given an initial account balance and an amount transferred out, 
    calculate the final account balance. -/
def final_balance (initial : ℕ) (transferred : ℕ) : ℕ :=
  initial - transferred

theorem account_balance_after_transfer :
  final_balance 27004 69 = 26935 := by
  sorry

end NUMINAMATH_CALUDE_account_balance_after_transfer_l2905_290531


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l2905_290545

theorem gcd_from_lcm_and_ratio (A B : ℕ+) 
  (h_lcm : Nat.lcm A B = 180)
  (h_ratio : A * 6 = B * 5) :
  Nat.gcd A B = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l2905_290545


namespace NUMINAMATH_CALUDE_power_of_product_l2905_290548

theorem power_of_product (a b : ℝ) : (3 * a * b)^2 = 9 * a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l2905_290548


namespace NUMINAMATH_CALUDE_exists_valid_configuration_l2905_290573

/-- A point in a plane represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points form an isosceles triangle -/
def isIsosceles (p1 p2 p3 : Point) : Prop :=
  let d12 := (p1.x - p2.x)^2 + (p1.y - p2.y)^2
  let d23 := (p2.x - p3.x)^2 + (p2.y - p3.y)^2
  let d31 := (p3.x - p1.x)^2 + (p3.y - p1.y)^2
  d12 = d23 ∨ d23 = d31 ∨ d31 = d12

/-- A configuration of five points in a plane -/
def Configuration := Fin 5 → Point

/-- Check if a configuration satisfies the isosceles condition for all triplets -/
def validConfiguration (config : Configuration) : Prop :=
  ∀ i j k, i < j → j < k → isIsosceles (config i) (config j) (config k)

/-- There exists a configuration of five points satisfying the isosceles condition -/
theorem exists_valid_configuration : ∃ (config : Configuration), validConfiguration config := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_configuration_l2905_290573


namespace NUMINAMATH_CALUDE_irrational_sum_product_theorem_l2905_290551

-- Define the property of being irrational
def IsIrrational (x : ℝ) : Prop := ¬ (∃ (q : ℚ), (q : ℝ) = x)

-- State the theorem
theorem irrational_sum_product_theorem (a : ℝ) (h : IsIrrational a) :
  ∃ (b b' : ℝ), IsIrrational b ∧ IsIrrational b' ∧
    (∃ (q1 q2 : ℚ), (a + b : ℝ) = q1 ∧ (a * b' : ℝ) = q2) ∧
    IsIrrational (a * b) ∧ IsIrrational (a + b') := by
  sorry


end NUMINAMATH_CALUDE_irrational_sum_product_theorem_l2905_290551


namespace NUMINAMATH_CALUDE_min_perimeter_l2905_290501

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  equalSide : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.equalSide + t.base

/-- Represents the pair of isosceles triangles in the problem -/
structure TrianglePair where
  t1 : IsoscelesTriangle
  t2 : IsoscelesTriangle

/-- The conditions given in the problem -/
def satisfiesConditions (pair : TrianglePair) : Prop :=
  let t1 := pair.t1
  let t2 := pair.t2
  -- Same perimeter
  perimeter t1 = perimeter t2 ∧
  -- Ratio of bases is 10:9
  10 * t2.base = 9 * t1.base ∧
  -- Base relations
  t1.base = 2 * t1.equalSide - 12 ∧
  t2.base = 3 * t2.equalSide - 30 ∧
  -- Non-congruent
  t1 ≠ t2

theorem min_perimeter (pair : TrianglePair) :
  satisfiesConditions pair → perimeter pair.t1 ≥ 228 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_l2905_290501


namespace NUMINAMATH_CALUDE_jake_buys_three_packages_l2905_290506

/-- Represents the number of sausage packages Jake buys -/
def num_packages : ℕ := 3

/-- Represents the weight of each sausage package in pounds -/
def package_weight : ℕ := 2

/-- Represents the price per pound of sausages in dollars -/
def price_per_pound : ℕ := 4

/-- Represents the total amount Jake pays in dollars -/
def total_paid : ℕ := 24

/-- Theorem stating that Jake buys 3 packages of sausages -/
theorem jake_buys_three_packages : 
  num_packages * package_weight * price_per_pound = total_paid :=
by sorry

end NUMINAMATH_CALUDE_jake_buys_three_packages_l2905_290506


namespace NUMINAMATH_CALUDE_stating_least_possible_area_l2905_290558

/-- Represents the length of a side of a square in centimeters. -/
def SideLength : ℝ := 5

/-- The lower bound of the actual side length when measured to the nearest centimeter. -/
def LowerBound : ℝ := SideLength - 0.5

/-- Calculates the area of a square given its side length. -/
def SquareArea (side : ℝ) : ℝ := side * side

/-- 
Theorem stating that the least possible area of a square with sides measured as 5 cm 
to the nearest centimeter is 20.25 cm².
-/
theorem least_possible_area :
  SquareArea LowerBound = 20.25 := by sorry

end NUMINAMATH_CALUDE_stating_least_possible_area_l2905_290558


namespace NUMINAMATH_CALUDE_increasing_function_condition_l2905_290577

theorem increasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → ((a - 1) * x + 2) < ((a - 1) * y + 2)) →
  a > 1 := by sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l2905_290577


namespace NUMINAMATH_CALUDE_opposite_of_negative_2022_l2905_290596

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_2022 : opposite (-2022) = 2022 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2022_l2905_290596


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_47_l2905_290544

theorem smallest_four_digit_divisible_by_47 :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 47 = 0 → 1034 ≤ n) ∧
  1000 ≤ 1034 ∧ 1034 < 10000 ∧ 1034 % 47 = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_47_l2905_290544


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2905_290555

theorem chess_tournament_games (n : ℕ) (h : n = 20) : 
  (n * (n - 1)) = 380 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2905_290555


namespace NUMINAMATH_CALUDE_smallest_number_of_students_l2905_290594

/-- Represents the number of students in each grade --/
structure Students where
  ninth : ℕ
  eighth : ℕ
  seventh : ℕ

/-- The ratio of 9th-graders to 7th-graders is 4:5 --/
def ratio_ninth_to_seventh (s : Students) : Prop :=
  5 * s.ninth = 4 * s.seventh

/-- The ratio of 9th-graders to 8th-graders is 7:6 --/
def ratio_ninth_to_eighth (s : Students) : Prop :=
  6 * s.ninth = 7 * s.eighth

/-- The total number of students --/
def total_students (s : Students) : ℕ :=
  s.ninth + s.eighth + s.seventh

/-- The statement to be proved --/
theorem smallest_number_of_students :
  ∃ (s : Students),
    ratio_ninth_to_seventh s ∧
    ratio_ninth_to_eighth s ∧
    total_students s = 87 ∧
    (∀ (t : Students),
      ratio_ninth_to_seventh t ∧
      ratio_ninth_to_eighth t →
      total_students t ≥ 87) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_students_l2905_290594


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2905_290547

def A (a : ℝ) : Set ℝ := {2, a^2 - a + 2, 1 - a}

theorem possible_values_of_a (a : ℝ) : 4 ∈ A a → a = -3 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2905_290547


namespace NUMINAMATH_CALUDE_max_y_coordinate_ellipse_l2905_290572

theorem max_y_coordinate_ellipse :
  let f (x y : ℝ) := x^2 / 49 + (y + 3)^2 / 25
  ∀ x y : ℝ, f x y = 1 → y ≤ 2 ∧ ∃ x₀ : ℝ, f x₀ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_y_coordinate_ellipse_l2905_290572


namespace NUMINAMATH_CALUDE_smallest_difference_is_one_l2905_290502

/-- Triangle with integer side lengths and specific ordering -/
structure OrderedTriangle where
  de : ℕ
  ef : ℕ
  fd : ℕ
  de_lt_ef : de < ef
  ef_le_fd : ef ≤ fd

/-- The perimeter of the triangle is 2050 -/
def hasPerimeter2050 (t : OrderedTriangle) : Prop :=
  t.de + t.ef + t.fd = 2050

/-- The triangle inequality holds -/
def satisfiesTriangleInequality (t : OrderedTriangle) : Prop :=
  t.de + t.ef > t.fd ∧ t.ef + t.fd > t.de ∧ t.fd + t.de > t.ef

theorem smallest_difference_is_one :
  ∃ (t : OrderedTriangle), 
    hasPerimeter2050 t ∧ 
    satisfiesTriangleInequality t ∧
    (∀ (u : OrderedTriangle), 
      hasPerimeter2050 u → satisfiesTriangleInequality u → 
      u.ef - u.de ≥ t.ef - t.de) ∧
    t.ef - t.de = 1 :=
  sorry

end NUMINAMATH_CALUDE_smallest_difference_is_one_l2905_290502


namespace NUMINAMATH_CALUDE_milk_water_ratio_after_addition_l2905_290507

theorem milk_water_ratio_after_addition 
  (initial_volume : ℝ) 
  (initial_milk_ratio : ℝ) 
  (initial_water_ratio : ℝ) 
  (added_water : ℝ) :
  initial_volume = 45 ∧ 
  initial_milk_ratio = 4 ∧ 
  initial_water_ratio = 1 ∧ 
  added_water = 3 →
  let initial_total_ratio := initial_milk_ratio + initial_water_ratio
  let initial_milk_volume := (initial_milk_ratio / initial_total_ratio) * initial_volume
  let initial_water_volume := (initial_water_ratio / initial_total_ratio) * initial_volume
  let final_milk_volume := initial_milk_volume
  let final_water_volume := initial_water_volume + added_water
  let final_ratio := final_milk_volume / final_water_volume
  final_ratio = 3 := by
sorry

end NUMINAMATH_CALUDE_milk_water_ratio_after_addition_l2905_290507


namespace NUMINAMATH_CALUDE_root_in_interval_l2905_290562

noncomputable def f (x : ℝ) := 2^x - 3*x

theorem root_in_interval :
  ∃ (r : ℝ), r ∈ Set.Ioo 3 4 ∧ f r = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l2905_290562


namespace NUMINAMATH_CALUDE_discriminant_positive_roots_difference_implies_m_values_l2905_290530

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 + (m + 3) * x + m + 1

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (m + 3)^2 - 4 * (m + 1)

-- Theorem 1: The discriminant is always positive for any real m
theorem discriminant_positive (m : ℝ) : discriminant m > 0 := by sorry

-- Define the roots of the quadratic equation
noncomputable def α (m : ℝ) : ℝ := sorry
noncomputable def β (m : ℝ) : ℝ := sorry

-- Theorem 2: If α - β = 2√2, then m = -3 or m = 1
theorem roots_difference_implies_m_values (m : ℝ) (h : α m - β m = 2 * Real.sqrt 2) : 
  m = -3 ∨ m = 1 := by sorry

end NUMINAMATH_CALUDE_discriminant_positive_roots_difference_implies_m_values_l2905_290530


namespace NUMINAMATH_CALUDE_distance_to_optimal_shooting_point_l2905_290513

/-- Given a field with width 2b, a goal with width 2a, and a distance c to the sideline,
    prove that the distance x satisfying the conditions is √((b-c)^2 - a^2). -/
theorem distance_to_optimal_shooting_point (b a c x : ℝ) 
  (h1 : b > 0)
  (h2 : a > 0)
  (h3 : c ≥ 0)
  (h4 : c < b)
  (h5 : (b - c)^2 = a^2 + x^2) :
  x = Real.sqrt ((b - c)^2 - a^2) := by
sorry

end NUMINAMATH_CALUDE_distance_to_optimal_shooting_point_l2905_290513


namespace NUMINAMATH_CALUDE_manny_has_more_ten_bills_l2905_290550

-- Define the number of bills each person has
def mandy_twenty_bills : ℕ := 3
def manny_fifty_bills : ℕ := 2

-- Define the value of each bill type
def twenty_bill_value : ℕ := 20
def fifty_bill_value : ℕ := 50
def ten_bill_value : ℕ := 10

-- Calculate the total value for each person
def mandy_total : ℕ := mandy_twenty_bills * twenty_bill_value
def manny_total : ℕ := manny_fifty_bills * fifty_bill_value

-- Calculate the number of $10 bills each person can get
def mandy_ten_bills : ℕ := mandy_total / ten_bill_value
def manny_ten_bills : ℕ := manny_total / ten_bill_value

-- State the theorem
theorem manny_has_more_ten_bills : manny_ten_bills - mandy_ten_bills = 4 := by
  sorry

end NUMINAMATH_CALUDE_manny_has_more_ten_bills_l2905_290550


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2905_290579

-- Define the function f(x) = -x^3 + 3x^2
def f (x : ℝ) : ℝ := -x^3 + 3*x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x

-- Theorem statement
theorem tangent_line_equation (a : ℝ) :
  ∃ b : ℝ, f' a = 3 ∧ f a = 3*a + b :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2905_290579


namespace NUMINAMATH_CALUDE_cement_warehouse_distribution_l2905_290556

theorem cement_warehouse_distribution (total : ℕ) (extra : ℕ) (multiplier : ℕ) 
  (warehouseA : ℕ) (warehouseB : ℕ) : 
  total = 462 → 
  extra = 32 → 
  multiplier = 4 →
  total = warehouseA + warehouseB → 
  warehouseA = multiplier * warehouseB + extra →
  warehouseA = 376 ∧ warehouseB = 86 := by
  sorry

end NUMINAMATH_CALUDE_cement_warehouse_distribution_l2905_290556


namespace NUMINAMATH_CALUDE_f_nonnegative_when_a_is_one_f_two_extreme_points_condition_l2905_290566

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.cos x - x

-- Theorem 1: When a = 1, f(x) ≥ 0 for all x
theorem f_nonnegative_when_a_is_one :
  ∀ x : ℝ, f 1 x ≥ 0 := by sorry

-- Theorem 2: f(x) has two extreme points in (0, π) iff 0 < a < e^(-π)
theorem f_two_extreme_points_condition (a : ℝ) :
  (∃ x y : ℝ, 0 < x ∧ x < y ∧ y < π ∧
   (∀ z : ℝ, 0 < z ∧ z < π → f a z ≤ f a x ∨ f a z ≤ f a y) ∧
   (∀ w : ℝ, x < w ∧ w < y → f a w > f a x ∧ f a w > f a y)) ↔
  (0 < a ∧ a < Real.exp (-π)) := by sorry

end NUMINAMATH_CALUDE_f_nonnegative_when_a_is_one_f_two_extreme_points_condition_l2905_290566


namespace NUMINAMATH_CALUDE_projection_region_area_l2905_290586

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ

/-- The region inside the trapezoid with the given projection property -/
def ProjectionRegion (t : IsoscelesTrapezoid) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The main theorem -/
theorem projection_region_area (t : IsoscelesTrapezoid) 
  (h1 : t.base1 = 1) (h2 : t.base2 = 2) (h3 : t.height = 1) : 
  area (ProjectionRegion t) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_projection_region_area_l2905_290586


namespace NUMINAMATH_CALUDE_max_digit_diff_l2905_290565

/-- Two-digit number representation -/
def two_digit_number (tens units : Nat) : Nat :=
  10 * tens + units

/-- The difference between two two-digit numbers -/
def digit_diff (a b : Nat) : Int :=
  (two_digit_number a b : Int) - (two_digit_number b a)

theorem max_digit_diff :
  ∀ a b : Nat,
    a ≠ b →
    a ≠ 0 →
    b ≠ 0 →
    a ≤ 9 →
    b ≤ 9 →
    digit_diff a b ≤ 72 ∧
    ∃ a b : Nat, a ≠ b ∧ a ≠ 0 ∧ b ≠ 0 ∧ a ≤ 9 ∧ b ≤ 9 ∧ digit_diff a b = 72 :=
by sorry

end NUMINAMATH_CALUDE_max_digit_diff_l2905_290565


namespace NUMINAMATH_CALUDE_remainder_proof_l2905_290593

theorem remainder_proof (a b c : ℕ) : 
  a < 10 → b < 10 → c < 10 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 10 = 2 →
  (7 * c) % 10 = 3 →
  (8 * b) % 10 = (4 + b) % 10 →
  (2 * a + b + 3 * c) % 10 = 1 :=
by sorry

end NUMINAMATH_CALUDE_remainder_proof_l2905_290593


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l2905_290533

theorem min_value_of_fraction (a : ℝ) (h : a > 1) : 
  ∀ x : ℝ, x > 1 → (x^2 - x + 1) / (x - 1) ≥ (a^2 - a + 1) / (a - 1) → 
  (a^2 - a + 1) / (a - 1) = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l2905_290533


namespace NUMINAMATH_CALUDE_total_age_in_three_years_l2905_290592

def age_problem (sam sue kendra : ℕ) : Prop :=
  sam = 2 * sue ∧ 
  kendra = 3 * sam ∧ 
  kendra = 18

theorem total_age_in_three_years 
  (sam sue kendra : ℕ) 
  (h : age_problem sam sue kendra) : 
  (sue + 3) + (sam + 3) + (kendra + 3) = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_age_in_three_years_l2905_290592


namespace NUMINAMATH_CALUDE_fourth_vertex_exists_l2905_290568

-- Define a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a quadrilateral
structure Quadrilateral :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

-- Define properties of the quadrilateral
def is_cyclic (q : Quadrilateral) : Prop := sorry

def is_tangential (q : Quadrilateral) : Prop := sorry

def is_convex (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem fourth_vertex_exists 
  (A B C : Point) 
  (h_convex : is_convex ⟨A, B, C, C⟩) 
  (h_cyclic : ∀ D, is_cyclic ⟨A, B, C, D⟩) 
  (h_tangential : ∀ D, is_tangential ⟨A, B, C, D⟩) : 
  ∃ D, is_cyclic ⟨A, B, C, D⟩ ∧ is_tangential ⟨A, B, C, D⟩ ∧ is_convex ⟨A, B, C, D⟩ :=
sorry

end NUMINAMATH_CALUDE_fourth_vertex_exists_l2905_290568


namespace NUMINAMATH_CALUDE_olympiad_survey_l2905_290583

theorem olympiad_survey (P : ℝ) (a b c d : ℝ) 
  (h1 : (a + b + d) / P = 0.9)
  (h2 : (a + c + d) / P = 0.6)
  (h3 : (b + c + d) / P = 0.9)
  (h4 : a + b + c + d = P)
  (h5 : P > 0) :
  d / P = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_olympiad_survey_l2905_290583


namespace NUMINAMATH_CALUDE_remainder_a_fourth_plus_four_l2905_290510

theorem remainder_a_fourth_plus_four (a : ℤ) (h : ¬ (5 ∣ a)) : (a^4 + 4) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_a_fourth_plus_four_l2905_290510


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2905_290534

theorem no_integer_solutions : ¬∃ (x y : ℤ), 0 < x ∧ x < y ∧ Real.sqrt 4096 = Real.sqrt x + Real.sqrt y + Real.sqrt (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2905_290534


namespace NUMINAMATH_CALUDE_initial_speed_calculation_l2905_290575

theorem initial_speed_calculation (distance : ℝ) (fast_speed : ℝ) (time_diff : ℝ) 
  (h1 : distance = 24)
  (h2 : fast_speed = 12)
  (h3 : time_diff = 2/3) : 
  ∃ v : ℝ, v > 0 ∧ distance / v - distance / fast_speed = time_diff ∧ v = 9 := by
sorry

end NUMINAMATH_CALUDE_initial_speed_calculation_l2905_290575


namespace NUMINAMATH_CALUDE_ab_range_l2905_290524

def f (x : ℝ) : ℝ := |2 - x^2|

theorem ab_range (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  0 < a * b ∧ a * b < 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_range_l2905_290524


namespace NUMINAMATH_CALUDE_seating_arrangement_l2905_290518

/-- The number of ways to arrange n people in n chairs -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of chairs in the row -/
def total_chairs : ℕ := 7

/-- The number of people to be seated -/
def people_to_seat : ℕ := 5

/-- The number of chairs that must remain empty -/
def empty_chairs : ℕ := 2

theorem seating_arrangement :
  permutations (total_chairs - empty_chairs) = 120 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_l2905_290518


namespace NUMINAMATH_CALUDE_sue_fill_time_l2905_290578

def jim_rate : ℚ := 1 / 30
def tony_rate : ℚ := 1 / 90
def combined_rate : ℚ := 1 / 15

def sue_time : ℚ := 45

theorem sue_fill_time (sue_rate : ℚ) 
  (h1 : sue_rate = 1 / sue_time)
  (h2 : jim_rate + sue_rate + tony_rate = combined_rate) : 
  sue_time = 45 := by sorry

end NUMINAMATH_CALUDE_sue_fill_time_l2905_290578


namespace NUMINAMATH_CALUDE_factorization_proof_l2905_290569

theorem factorization_proof (m n : ℝ) : 4 * m^2 * n - 4 * n^3 = 4 * n * (m + n) * (m - n) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2905_290569


namespace NUMINAMATH_CALUDE_high_octane_half_cost_l2905_290552

/-- Represents the composition and cost of a fuel mixture -/
structure FuelMixture where
  high_octane : ℚ
  regular_octane : ℚ
  cost_ratio : ℚ
  total : ℚ

/-- Calculates the fraction of total cost due to high octane fuel -/
def high_octane_cost_fraction (fuel : FuelMixture) : ℚ :=
  (fuel.high_octane * fuel.cost_ratio) / ((fuel.high_octane * fuel.cost_ratio) + fuel.regular_octane)

/-- Theorem: For a fuel mixture with 15 parts high octane and 45 parts regular octane,
    where high octane costs 3 times as much as regular octane,
    the fraction of the total cost due to high octane is 1/2 -/
theorem high_octane_half_cost (fuel : FuelMixture)
  (h1 : fuel.high_octane = 15)
  (h2 : fuel.regular_octane = 45)
  (h3 : fuel.cost_ratio = 3)
  (h4 : fuel.total = fuel.high_octane + fuel.regular_octane) :
  high_octane_cost_fraction fuel = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_high_octane_half_cost_l2905_290552


namespace NUMINAMATH_CALUDE_amber_amethyst_ratio_l2905_290557

/-- Given a necklace with 40 beads, 7 amethyst beads, and 19 turquoise beads,
    prove that the ratio of amber beads to amethyst beads is 2:1. -/
theorem amber_amethyst_ratio (total : ℕ) (amethyst : ℕ) (turquoise : ℕ) 
  (h1 : total = 40)
  (h2 : amethyst = 7)
  (h3 : turquoise = 19) :
  (total - amethyst - turquoise) / amethyst = 2 := by
  sorry

end NUMINAMATH_CALUDE_amber_amethyst_ratio_l2905_290557


namespace NUMINAMATH_CALUDE_annual_bill_calculation_correct_l2905_290585

/-- Calculates the total annual bill for Noah's calls to his Grammy -/
def annual_bill_calculation : ℝ :=
  let weekday_duration : ℝ := 25
  let weekend_duration : ℝ := 45
  let holiday_duration : ℝ := 60
  
  let total_weekdays : ℝ := 260
  let total_weekends : ℝ := 104
  let total_holidays : ℝ := 11
  
  let intl_weekdays : ℝ := 130
  let intl_weekends : ℝ := 52
  let intl_holidays : ℝ := 6
  
  let local_weekday_rate : ℝ := 0.05
  let local_weekend_rate : ℝ := 0.06
  let local_holiday_rate : ℝ := 0.07
  
  let intl_weekday_rate : ℝ := 0.09
  let intl_weekend_rate : ℝ := 0.11
  let intl_holiday_rate : ℝ := 0.12
  
  let tax_rate : ℝ := 0.10
  let monthly_service_fee : ℝ := 2.99
  let intl_holiday_discount : ℝ := 0.05
  
  let local_weekday_cost := (total_weekdays - intl_weekdays) * weekday_duration * local_weekday_rate
  let local_weekend_cost := (total_weekends - intl_weekends) * weekend_duration * local_weekend_rate
  let local_holiday_cost := (total_holidays - intl_holidays) * holiday_duration * local_holiday_rate
  
  let intl_weekday_cost := intl_weekdays * weekday_duration * intl_weekday_rate
  let intl_weekend_cost := intl_weekends * weekend_duration * intl_weekend_rate
  let intl_holiday_cost := intl_holidays * holiday_duration * intl_holiday_rate * (1 - intl_holiday_discount)
  
  let total_call_cost := local_weekday_cost + local_weekend_cost + local_holiday_cost + 
                         intl_weekday_cost + intl_weekend_cost + intl_holiday_cost
  
  let total_tax := total_call_cost * tax_rate
  let total_service_fee := monthly_service_fee * 12
  
  total_call_cost + total_tax + total_service_fee

theorem annual_bill_calculation_correct : 
  annual_bill_calculation = 1042.20 := by sorry

end NUMINAMATH_CALUDE_annual_bill_calculation_correct_l2905_290585


namespace NUMINAMATH_CALUDE_sum_of_ages_sum_of_ages_proof_l2905_290540

theorem sum_of_ages : ℕ → ℕ → Prop :=
  fun john_age father_age =>
    (john_age = 15) →
    (father_age = 2 * john_age + 32) →
    (john_age + father_age = 77)

-- Proof
theorem sum_of_ages_proof : sum_of_ages 15 62 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_sum_of_ages_proof_l2905_290540


namespace NUMINAMATH_CALUDE_train_passing_time_l2905_290527

/-- The length of train A in meters -/
def train_a_length : ℝ := 150

/-- The length of train B in meters -/
def train_b_length : ℝ := 200

/-- The time (in seconds) it takes for a passenger on train A to see train B pass by -/
def time_a_sees_b : ℝ := 10

/-- The time (in seconds) it takes for a passenger on train B to see train A pass by -/
def time_b_sees_a : ℝ := 7.5

theorem train_passing_time :
  (train_b_length / time_a_sees_b) = (train_a_length / time_b_sees_a) :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l2905_290527


namespace NUMINAMATH_CALUDE_line_contains_point_l2905_290537

/-- A line in the xy-plane represented by the equation 2 - kx = 5y -/
def line (k : ℝ) (x y : ℝ) : Prop := 2 - k * x = 5 * y

/-- The point (2, -1) -/
def point : ℝ × ℝ := (2, -1)

/-- Theorem: The line contains the point (2, -1) if and only if k = 7/2 -/
theorem line_contains_point :
  ∀ k : ℝ, line k point.1 point.2 ↔ k = 7/2 := by sorry

end NUMINAMATH_CALUDE_line_contains_point_l2905_290537


namespace NUMINAMATH_CALUDE_first_player_min_score_l2905_290542

/-- Represents a game state with remaining numbers -/
def GameState := List Nat

/-- Removes a list of numbers from the game state -/
def removeNumbers (state : GameState) (toRemove : List Nat) : GameState :=
  state.filter (λ n => n ∉ toRemove)

/-- Calculates the score based on the two remaining numbers -/
def calculateScore (state : GameState) : Nat :=
  if state.length = 2 then
    state.maximum.getD 0 - state.minimum.getD 0
  else
    0

/-- Represents a player's strategy -/
def Strategy := GameState → List Nat

/-- Simulates a game given two strategies -/
def playGame (player1 : Strategy) (player2 : Strategy) : Nat :=
  let initialState : GameState := List.range 101
  let finalState := (List.range 11).foldl
    (λ state round =>
      let state' := removeNumbers state (player1 state)
      removeNumbers state' (player2 state'))
    initialState
  calculateScore finalState

/-- Theorem: The first player can always ensure a score of at least 52 -/
theorem first_player_min_score :
  ∃ (player1 : Strategy), ∀ (player2 : Strategy), playGame player1 player2 ≥ 52 := by
  sorry


end NUMINAMATH_CALUDE_first_player_min_score_l2905_290542


namespace NUMINAMATH_CALUDE_kindergarten_craft_problem_l2905_290505

theorem kindergarten_craft_problem :
  ∃ (scissors glue_sticks crayons : ℕ),
    scissors + glue_sticks + crayons = 26 ∧
    2 * scissors + 3 * glue_sticks + 4 * crayons = 24 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_craft_problem_l2905_290505


namespace NUMINAMATH_CALUDE_family_reunion_ratio_l2905_290541

theorem family_reunion_ratio (male_adults female_adults children total_adults total_people : ℕ) : 
  female_adults = male_adults + 50 →
  male_adults = 100 →
  total_adults = male_adults + female_adults →
  total_people = 750 →
  total_people = total_adults + children →
  (children : ℚ) / total_adults = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_family_reunion_ratio_l2905_290541


namespace NUMINAMATH_CALUDE_estimate_fish_population_l2905_290554

/-- Estimate the total number of fish in a pond using mark-recapture method -/
theorem estimate_fish_population (initially_caught marked_in_second_catch second_catch : ℕ) :
  initially_caught = 30 →
  marked_in_second_catch = 2 →
  second_catch = 50 →
  (initially_caught * second_catch) / marked_in_second_catch = 750 :=
by sorry

end NUMINAMATH_CALUDE_estimate_fish_population_l2905_290554


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l2905_290519

theorem range_of_a_minus_b (a b : ℝ) (ha : -1 < a ∧ a < 1) (hb : 1 < b ∧ b < 3) :
  -4 < a - b ∧ a - b < 0 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l2905_290519


namespace NUMINAMATH_CALUDE_binary_multiplication_l2905_290515

-- Define binary numbers as natural numbers
def binary_1101 : ℕ := 13  -- 1101₂ in decimal
def binary_111 : ℕ := 7    -- 111₂ in decimal

-- Define the expected result
def expected_result : ℕ := 79  -- 1001111₂ in decimal

-- Theorem statement
theorem binary_multiplication :
  binary_1101 * binary_111 = expected_result := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_l2905_290515


namespace NUMINAMATH_CALUDE_bus_speed_problem_l2905_290508

/-- Given two buses traveling in opposite directions, this theorem proves
    the speed of the second bus given the conditions of the problem. -/
theorem bus_speed_problem (east_speed : ℝ) (time : ℝ) (total_distance : ℝ)
  (h1 : east_speed = 55)
  (h2 : time = 4)
  (h3 : total_distance = 460) :
  ∃ west_speed : ℝ, 
    west_speed * time + east_speed * time = total_distance ∧
    west_speed = 60 :=
by sorry

end NUMINAMATH_CALUDE_bus_speed_problem_l2905_290508


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2905_290559

theorem expand_and_simplify (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2905_290559


namespace NUMINAMATH_CALUDE_function_characterization_l2905_290504

/-- A continuous monotonic function satisfying the given inequality is equal to x + 1 -/
theorem function_characterization (f : ℝ → ℝ) 
  (hcont : Continuous f) 
  (hmono : Monotone f) 
  (h0 : f 0 = 1) 
  (hineq : ∀ x y : ℝ, f (x + y) ≥ f x * f y - f (x * y) + 1) :
  ∀ x : ℝ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_characterization_l2905_290504


namespace NUMINAMATH_CALUDE_high_jump_probabilities_l2905_290532

/-- The probability of clearing the height in a single jump -/
def p : ℝ := 0.8

/-- The probability of clearing the height on two consecutive jumps -/
def prob_two_consecutive : ℝ := p * p

/-- The probability of clearing the height for the first time on the third attempt -/
def prob_third_attempt : ℝ := (1 - p) * (1 - p) * p

/-- The minimum number of attempts required to clear the height with a 99% probability -/
def min_attempts : ℕ := 3

/-- Theorem stating the probabilities and minimum attempts -/
theorem high_jump_probabilities :
  prob_two_consecutive = 0.64 ∧
  prob_third_attempt = 0.032 ∧
  min_attempts = 3 ∧
  (1 - (1 - p) ^ min_attempts ≥ 0.99) :=
by sorry

end NUMINAMATH_CALUDE_high_jump_probabilities_l2905_290532


namespace NUMINAMATH_CALUDE_one_positive_integer_solution_l2905_290576

theorem one_positive_integer_solution : 
  ∃! (x : ℕ), x > 0 ∧ 24 - 6 * x > 12 :=
by sorry

end NUMINAMATH_CALUDE_one_positive_integer_solution_l2905_290576


namespace NUMINAMATH_CALUDE_markers_count_l2905_290574

/-- Given a ratio of pens : pencils : markers as 2 : 2 : 5, and 10 pens, the number of markers is 25. -/
theorem markers_count (pens pencils markers : ℕ) : 
  pens = 10 → 
  pens * 5 = markers * 2 → 
  markers = 25 := by
  sorry

end NUMINAMATH_CALUDE_markers_count_l2905_290574


namespace NUMINAMATH_CALUDE_hash_nested_20_l2905_290539

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.5 * N + 2

-- State the theorem
theorem hash_nested_20 : hash (hash (hash (hash 20))) = 5 := by sorry

end NUMINAMATH_CALUDE_hash_nested_20_l2905_290539


namespace NUMINAMATH_CALUDE_joan_remaining_oranges_l2905_290570

theorem joan_remaining_oranges (joan_initial : ℕ) (sara_sold : ℕ) (joan_remaining : ℕ) : 
  joan_initial = 37 → sara_sold = 10 → joan_remaining = joan_initial - sara_sold → joan_remaining = 27 := by
  sorry

end NUMINAMATH_CALUDE_joan_remaining_oranges_l2905_290570


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_l2905_290567

/-- The number of books in the 'crazy silly school' series -/
def total_books : ℕ := 21

/-- The number of books that have been read -/
def books_read : ℕ := 13

/-- The number of books yet to be read -/
def books_unread : ℕ := 8

/-- Theorem: The total number of books is equal to the sum of read and unread books -/
theorem crazy_silly_school_books : total_books = books_read + books_unread := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_l2905_290567


namespace NUMINAMATH_CALUDE_opposite_signs_and_larger_absolute_value_l2905_290520

theorem opposite_signs_and_larger_absolute_value (a b : ℚ) 
  (h1 : a * b < 0) (h2 : a + b > 0) : 
  (a < 0 ∧ b > 0) ∨ (a > 0 ∧ b < 0) ∧ 
  (max (abs a) (abs b) = abs (max a b)) :=
sorry

end NUMINAMATH_CALUDE_opposite_signs_and_larger_absolute_value_l2905_290520


namespace NUMINAMATH_CALUDE_glass_pane_area_is_4900_l2905_290521

/-- The area of a square glass pane inside a square frame -/
def glass_pane_area (frame_side_length : ℝ) (frame_width : ℝ) : ℝ :=
  (frame_side_length - 2 * frame_width) ^ 2

/-- Theorem: The area of the square glass pane is 4900 cm² -/
theorem glass_pane_area_is_4900 :
  glass_pane_area 100 15 = 4900 := by
  sorry

end NUMINAMATH_CALUDE_glass_pane_area_is_4900_l2905_290521


namespace NUMINAMATH_CALUDE_atomic_numbers_descending_l2905_290543

/-- Atomic number of Chlorine -/
def atomic_number_Cl : ℕ := 17

/-- Atomic number of Oxygen -/
def atomic_number_O : ℕ := 8

/-- Atomic number of Lithium -/
def atomic_number_Li : ℕ := 3

/-- Theorem stating that the atomic numbers of Cl, O, and Li are in descending order -/
theorem atomic_numbers_descending :
  atomic_number_Cl > atomic_number_O ∧ atomic_number_O > atomic_number_Li :=
sorry

end NUMINAMATH_CALUDE_atomic_numbers_descending_l2905_290543


namespace NUMINAMATH_CALUDE_intersection_of_sets_l2905_290582

theorem intersection_of_sets : 
  let A : Set ℤ := {1, 2, 3}
  let B : Set ℤ := {-2, 2}
  A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l2905_290582


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2905_290523

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n : ℚ) / d = 0.714714714 ∧ 
  (∀ (n' d' : ℕ), (n' : ℚ) / d' = 0.714714714 → n ≤ n' ∧ d ≤ d') ∧
  n + d = 571 :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2905_290523
