import Mathlib

namespace NUMINAMATH_CALUDE_x_minus_y_equals_four_l73_7359

theorem x_minus_y_equals_four (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares_eq : x^2 - y^2 = 40) : 
  x - y = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_four_l73_7359


namespace NUMINAMATH_CALUDE_math_contest_problem_count_l73_7321

/-- Represents the number of problems solved by each participant -/
structure ParticipantSolutions where
  neznayka : ℕ
  pilyulkin : ℕ
  knopochka : ℕ
  vintik : ℕ
  znayka : ℕ

/-- Defines the conditions of the math contest -/
def MathContest (n : ℕ) (solutions : ParticipantSolutions) : Prop :=
  solutions.neznayka = 6 ∧
  solutions.znayka = 10 ∧
  solutions.pilyulkin > solutions.neznayka ∧
  solutions.pilyulkin < solutions.znayka ∧
  solutions.knopochka > solutions.neznayka ∧
  solutions.knopochka < solutions.znayka ∧
  solutions.vintik > solutions.neznayka ∧
  solutions.vintik < solutions.znayka ∧
  solutions.neznayka + solutions.pilyulkin + solutions.knopochka + solutions.vintik + solutions.znayka = 4 * n

theorem math_contest_problem_count (solutions : ParticipantSolutions) :
  ∃ n : ℕ, MathContest n solutions → n = 10 :=
by sorry

end NUMINAMATH_CALUDE_math_contest_problem_count_l73_7321


namespace NUMINAMATH_CALUDE_interest_percentage_of_face_value_l73_7313

-- Define the bond parameters
def face_value : ℝ := 5000
def selling_price : ℝ := 6153.846153846153
def interest_rate_of_selling_price : ℝ := 0.065

-- Define the theorem
theorem interest_percentage_of_face_value :
  let interest := interest_rate_of_selling_price * selling_price
  let interest_percentage_of_face := (interest / face_value) * 100
  interest_percentage_of_face = 8 := by sorry

end NUMINAMATH_CALUDE_interest_percentage_of_face_value_l73_7313


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l73_7394

theorem arithmetic_sequence_general_term (a : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2) →
  (∀ n : ℕ, n ≥ 1 → a n = 2 * n - 1) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l73_7394


namespace NUMINAMATH_CALUDE_smallest_covering_radius_l73_7345

theorem smallest_covering_radius :
  let r : ℝ := Real.sqrt 3 / 2
  ∀ s : ℝ, s < r → ¬(∃ (c₁ c₂ c₃ : ℝ × ℝ),
    (∀ p : ℝ × ℝ, Real.sqrt ((p.1 - 0)^2 + (p.2 - 0)^2) ≤ 1 →
      Real.sqrt ((p.1 - c₁.1)^2 + (p.2 - c₁.2)^2) ≤ s ∨
      Real.sqrt ((p.1 - c₂.1)^2 + (p.2 - c₂.2)^2) ≤ s ∨
      Real.sqrt ((p.1 - c₃.1)^2 + (p.2 - c₃.2)^2) ≤ s)) ∧
  ∃ (c₁ c₂ c₃ : ℝ × ℝ),
    (∀ p : ℝ × ℝ, Real.sqrt ((p.1 - 0)^2 + (p.2 - 0)^2) ≤ 1 →
      Real.sqrt ((p.1 - c₁.1)^2 + (p.2 - c₁.2)^2) ≤ r ∨
      Real.sqrt ((p.1 - c₂.1)^2 + (p.2 - c₂.2)^2) ≤ r ∨
      Real.sqrt ((p.1 - c₃.1)^2 + (p.2 - c₃.2)^2) ≤ r) :=
by sorry

end NUMINAMATH_CALUDE_smallest_covering_radius_l73_7345


namespace NUMINAMATH_CALUDE_barefoot_kids_count_l73_7301

theorem barefoot_kids_count (total kids_with_socks kids_with_shoes kids_with_both : ℕ) :
  total = 22 ∧ kids_with_socks = 12 ∧ kids_with_shoes = 8 ∧ kids_with_both = 6 →
  total - ((kids_with_socks - kids_with_both) + (kids_with_shoes - kids_with_both) + kids_with_both) = 8 := by
sorry

end NUMINAMATH_CALUDE_barefoot_kids_count_l73_7301


namespace NUMINAMATH_CALUDE_laptop_price_l73_7336

theorem laptop_price (deposit : ℝ) (deposit_percentage : ℝ) (full_price : ℝ) 
  (h1 : deposit = 400)
  (h2 : deposit_percentage = 25)
  (h3 : deposit = (deposit_percentage / 100) * full_price) :
  full_price = 1600 := by
sorry

end NUMINAMATH_CALUDE_laptop_price_l73_7336


namespace NUMINAMATH_CALUDE_base9_addition_theorem_l73_7334

/-- Addition of numbers in base 9 --/
def base9_add (a b c : ℕ) : ℕ :=
  sorry

/-- Conversion from base 9 to base 10 --/
def base9_to_base10 (n : ℕ) : ℕ :=
  sorry

theorem base9_addition_theorem :
  base9_add 2175 1714 406 = 4406 :=
by sorry

end NUMINAMATH_CALUDE_base9_addition_theorem_l73_7334


namespace NUMINAMATH_CALUDE_walnuts_amount_l73_7356

/-- The amount of walnuts in the trail mix -/
def walnuts : ℝ := sorry

/-- The total amount of nuts in the trail mix -/
def total_nuts : ℝ := 0.5

/-- The amount of almonds in the trail mix -/
def almonds : ℝ := 0.25

/-- Theorem stating that the amount of walnuts is 0.25 cups -/
theorem walnuts_amount : walnuts = 0.25 := by sorry

end NUMINAMATH_CALUDE_walnuts_amount_l73_7356


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l73_7312

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a^2 < 0 → a < b) ∧
  (∃ a b : ℝ, a < b ∧ (a - b) * a^2 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l73_7312


namespace NUMINAMATH_CALUDE_fiftieth_parenthesis_sum_l73_7325

def sequence_term (n : ℕ) : ℕ := 24 * ((n - 1) / 4) + 1

def parenthesis_sum (n : ℕ) : ℕ :=
  if n % 4 = 1 then sequence_term n
  else if n % 4 = 2 then sequence_term n + (sequence_term n + 2)
  else if n % 4 = 3 then sequence_term n + (sequence_term n + 2) + (sequence_term n + 4)
  else sequence_term n

theorem fiftieth_parenthesis_sum : parenthesis_sum 50 = 392 := by sorry

end NUMINAMATH_CALUDE_fiftieth_parenthesis_sum_l73_7325


namespace NUMINAMATH_CALUDE_final_apple_count_l73_7342

def apples_on_tree (initial : ℕ) (picked : ℕ) (new_growth : ℕ) : ℕ :=
  initial - picked + new_growth

theorem final_apple_count :
  apples_on_tree 11 7 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_final_apple_count_l73_7342


namespace NUMINAMATH_CALUDE_presidentAndCommittee_ten_l73_7396

/-- The number of ways to choose a president and a 3-person committee from a group of n people,
    where the president cannot be on the committee and the order of choosing committee members does not matter. -/
def presidentAndCommittee (n : ℕ) : ℕ :=
  n * (n - 1).choose 3

/-- Theorem stating that for a group of 10 people, there are 840 ways to choose a president
    and a 3-person committee under the given conditions. -/
theorem presidentAndCommittee_ten :
  presidentAndCommittee 10 = 840 := by
  sorry

end NUMINAMATH_CALUDE_presidentAndCommittee_ten_l73_7396


namespace NUMINAMATH_CALUDE_range_of_a_l73_7316

theorem range_of_a (x a : ℝ) : 
  (∀ x, (|x + a| < 3 ↔ 2 < x ∧ x < 3)) → 
  -5 ≤ a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l73_7316


namespace NUMINAMATH_CALUDE_books_ratio_proof_l73_7330

def books_problem (initial_books : ℕ) (rebecca_books : ℕ) (remaining_books : ℕ) : Prop :=
  let mara_books := initial_books - remaining_books - rebecca_books
  mara_books / rebecca_books = 3

theorem books_ratio_proof (initial_books : ℕ) (rebecca_books : ℕ) (remaining_books : ℕ)
  (h1 : initial_books = 220)
  (h2 : rebecca_books = 40)
  (h3 : remaining_books = 60) :
  books_problem initial_books rebecca_books remaining_books :=
by
  sorry

end NUMINAMATH_CALUDE_books_ratio_proof_l73_7330


namespace NUMINAMATH_CALUDE_louisa_second_day_travel_l73_7362

/-- Proves that Louisa traveled 280 miles on the second day of her vacation --/
theorem louisa_second_day_travel :
  ∀ (first_day_distance second_day_distance : ℝ) 
    (speed : ℝ) 
    (time_difference : ℝ),
  first_day_distance = 160 →
  speed = 40 →
  time_difference = 3 →
  first_day_distance / speed + time_difference = second_day_distance / speed →
  second_day_distance = 280 := by
sorry

end NUMINAMATH_CALUDE_louisa_second_day_travel_l73_7362


namespace NUMINAMATH_CALUDE_apples_in_basket_A_l73_7344

/-- The number of baskets -/
def num_baskets : ℕ := 5

/-- The average number of fruits per basket -/
def avg_fruits_per_basket : ℕ := 25

/-- The number of mangoes in basket B -/
def mangoes_in_B : ℕ := 30

/-- The number of peaches in basket C -/
def peaches_in_C : ℕ := 20

/-- The number of pears in basket D -/
def pears_in_D : ℕ := 25

/-- The number of bananas in basket E -/
def bananas_in_E : ℕ := 35

/-- The number of apples in basket A -/
def apples_in_A : ℕ := num_baskets * avg_fruits_per_basket - (mangoes_in_B + peaches_in_C + pears_in_D + bananas_in_E)

theorem apples_in_basket_A : apples_in_A = 15 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_basket_A_l73_7344


namespace NUMINAMATH_CALUDE_jaden_initial_cars_l73_7378

/-- The number of toy cars Jaden had initially -/
def initial_cars : ℕ := 14

/-- The number of cars Jaden bought -/
def bought_cars : ℕ := 28

/-- The number of cars Jaden received as gifts -/
def gift_cars : ℕ := 12

/-- The number of cars Jaden gave to his sister -/
def sister_cars : ℕ := 8

/-- The number of cars Jaden gave to his friend -/
def friend_cars : ℕ := 3

/-- The number of cars Jaden has left -/
def remaining_cars : ℕ := 43

theorem jaden_initial_cars :
  initial_cars + bought_cars + gift_cars - sister_cars - friend_cars = remaining_cars :=
by sorry

end NUMINAMATH_CALUDE_jaden_initial_cars_l73_7378


namespace NUMINAMATH_CALUDE_peanut_butter_jar_size_l73_7357

/-- Calculates the size of the third jar given the total amount of peanut butter,
    the sizes of two jars, and the total number of jars. -/
def third_jar_size (total_peanut_butter : ℕ) (jar1_size jar2_size : ℕ) (total_jars : ℕ) : ℕ :=
  let jars_per_size := total_jars / 3
  let remaining_peanut_butter := total_peanut_butter - (jar1_size + jar2_size) * jars_per_size
  remaining_peanut_butter / jars_per_size

/-- Proves that given the conditions, the size of the third jar is 40 ounces. -/
theorem peanut_butter_jar_size :
  third_jar_size 252 16 28 9 = 40 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_jar_size_l73_7357


namespace NUMINAMATH_CALUDE_johns_expenses_exceed_earnings_l73_7391

/-- Represents the percentage of John's earnings spent on each category -/
structure Expenses where
  rent : ℝ
  dishwasher : ℝ
  groceries : ℝ

/-- Calculates John's expenses based on the given conditions -/
def calculate_expenses (rent_percent : ℝ) : Expenses :=
  { rent := rent_percent,
    dishwasher := rent_percent - (0.3 * rent_percent),
    groceries := rent_percent + (0.15 * rent_percent) }

/-- Theorem stating that John's expenses exceed his earnings -/
theorem johns_expenses_exceed_earnings (rent_percent : ℝ) 
  (h1 : rent_percent = 0.4)  -- John spent 40% of his earnings on rent
  (h2 : rent_percent > 0)    -- Rent percentage is positive
  (h3 : rent_percent < 1)    -- Rent percentage is less than 100%
  : (calculate_expenses rent_percent).rent + 
    (calculate_expenses rent_percent).dishwasher + 
    (calculate_expenses rent_percent).groceries > 1 := by
  sorry

#check johns_expenses_exceed_earnings

end NUMINAMATH_CALUDE_johns_expenses_exceed_earnings_l73_7391


namespace NUMINAMATH_CALUDE_simplify_expression_l73_7361

theorem simplify_expression (x : ℝ) : 
  x * (3 * x^2 - 2) - 5 * (x^2 - 2*x + 7) = 3 * x^3 - 5 * x^2 + 8 * x - 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l73_7361


namespace NUMINAMATH_CALUDE_max_product_constraint_l73_7304

theorem max_product_constraint (x y : ℝ) (h : x + y = 1) : x * y ≤ 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l73_7304


namespace NUMINAMATH_CALUDE_negation_of_exists_square_nonpositive_l73_7368

theorem negation_of_exists_square_nonpositive :
  (¬ ∃ x : ℝ, x^2 ≤ 0) ↔ (∀ x : ℝ, x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exists_square_nonpositive_l73_7368


namespace NUMINAMATH_CALUDE_certain_number_proof_l73_7346

theorem certain_number_proof : ∃ X : ℝ, 
  0.8 * X = 0.7 * 60.00000000000001 + 30 ∧ X = 90.00000000000001 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l73_7346


namespace NUMINAMATH_CALUDE_farmers_herd_size_l73_7383

theorem farmers_herd_size :
  ∀ (total : ℚ),
  (2 / 5 : ℚ) * total + (1 / 5 : ℚ) * total + (1 / 10 : ℚ) * total + 9 = total →
  total = 30 := by
sorry

end NUMINAMATH_CALUDE_farmers_herd_size_l73_7383


namespace NUMINAMATH_CALUDE_hat_pairs_l73_7318

/-- Given a group of 12 people where exactly 4 are wearing hats, 
    the number of pairs where at least one person is wearing a hat is 38. -/
theorem hat_pairs (total : ℕ) (hat_wearers : ℕ) (h1 : total = 12) (h2 : hat_wearers = 4) :
  (total.choose 2) - ((total - hat_wearers).choose 2) = 38 := by
  sorry

end NUMINAMATH_CALUDE_hat_pairs_l73_7318


namespace NUMINAMATH_CALUDE_moon_speed_km_per_hour_l73_7372

/-- The speed of the moon in kilometers per second -/
def moon_speed_km_per_sec : ℝ := 0.2

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem: The moon's speed in kilometers per hour -/
theorem moon_speed_km_per_hour :
  moon_speed_km_per_sec * (seconds_per_hour : ℝ) = 720 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_km_per_hour_l73_7372


namespace NUMINAMATH_CALUDE_january_salary_l73_7329

/-- Given the average salaries for two sets of four months and the salary for May,
    calculate the salary for January. -/
theorem january_salary
  (avg_jan_to_apr : ℝ)
  (avg_feb_to_may : ℝ)
  (may_salary : ℝ)
  (h1 : avg_jan_to_apr = 8000)
  (h2 : avg_feb_to_may = 8500)
  (h3 : may_salary = 6500) :
  ∃ (jan feb mar apr : ℝ),
    (jan + feb + mar + apr) / 4 = avg_jan_to_apr ∧
    (feb + mar + apr + may_salary) / 4 = avg_feb_to_may ∧
    jan = 4500 := by
  sorry

#check january_salary

end NUMINAMATH_CALUDE_january_salary_l73_7329


namespace NUMINAMATH_CALUDE_jemma_grasshopper_count_l73_7302

/-- The number of grasshoppers Jemma saw on her African daisy plant -/
def grasshoppers_on_plant : ℕ := 7

/-- The number of dozens of baby grasshoppers Jemma found under the plant -/
def dozens_of_baby_grasshoppers : ℕ := 2

/-- The number of grasshoppers in a dozen -/
def grasshoppers_per_dozen : ℕ := 12

/-- The total number of grasshoppers Jemma found -/
def total_grasshoppers : ℕ := grasshoppers_on_plant + dozens_of_baby_grasshoppers * grasshoppers_per_dozen

theorem jemma_grasshopper_count : total_grasshoppers = 31 := by
  sorry

end NUMINAMATH_CALUDE_jemma_grasshopper_count_l73_7302


namespace NUMINAMATH_CALUDE_prime_power_fraction_implies_prime_l73_7341

theorem prime_power_fraction_implies_prime (n : ℕ) (h1 : n ≥ 2) :
  (∃ b : ℕ+, ∃ p : ℕ, ∃ k : ℕ, Prime p ∧ (b^n - 1) / (b - 1) = p^k) →
  Prime n :=
by sorry

end NUMINAMATH_CALUDE_prime_power_fraction_implies_prime_l73_7341


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l73_7333

theorem sum_of_fifth_powers (a b c : ℝ) (h : a + b + c = 0) :
  2 * (a^5 + b^5 + c^5) = 5 * a * b * c * (a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l73_7333


namespace NUMINAMATH_CALUDE_largest_circle_area_l73_7366

theorem largest_circle_area (length width : ℝ) (h1 : length = 18) (h2 : width = 8) :
  let perimeter := 2 * (length + width)
  let radius := perimeter / (2 * Real.pi)
  (Real.pi * radius ^ 2) = 676 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_largest_circle_area_l73_7366


namespace NUMINAMATH_CALUDE_probability_at_least_one_female_l73_7338

theorem probability_at_least_one_female (total : ℕ) (male : ℕ) (female : ℕ) (selected : ℕ) :
  total = male + female →
  selected = 3 →
  male = 6 →
  female = 4 →
  (1 - (Nat.choose male selected / Nat.choose total selected : ℚ)) = 5/6 :=
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_female_l73_7338


namespace NUMINAMATH_CALUDE_farm_legs_count_l73_7354

/-- Represents the number of legs for each animal type -/
def legs_per_animal (animal : String) : Nat :=
  match animal with
  | "chicken" => 2
  | "buffalo" => 4
  | _ => 0

/-- Calculates the total number of legs in the farm -/
def total_legs (total_animals : Nat) (chickens : Nat) : Nat :=
  let buffalos := total_animals - chickens
  chickens * legs_per_animal "chicken" + buffalos * legs_per_animal "buffalo"

/-- Theorem: In a farm with 9 animals, including 5 chickens and the rest buffalos, there are 26 legs in total -/
theorem farm_legs_count : total_legs 9 5 = 26 := by
  sorry

end NUMINAMATH_CALUDE_farm_legs_count_l73_7354


namespace NUMINAMATH_CALUDE_calculate_speed_l73_7351

/-- Given two people moving in opposite directions, calculate the speed of one person given the other's speed and their final distance after a certain time. -/
theorem calculate_speed 
  (roja_speed : ℝ) 
  (time : ℝ) 
  (final_distance : ℝ) 
  (h1 : roja_speed = 7)
  (h2 : time = 4)
  (h3 : final_distance = 40) :
  ∃ (pooja_speed : ℝ), pooja_speed = 3 ∧ (roja_speed + pooja_speed) * time = final_distance :=
by sorry

end NUMINAMATH_CALUDE_calculate_speed_l73_7351


namespace NUMINAMATH_CALUDE_sum_of_four_squares_equals_prime_multiple_l73_7317

theorem sum_of_four_squares_equals_prime_multiple (p : Nat) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃ (m : Nat) (x₁ x₂ x₃ x₄ : Int), 
    m < p ∧ 
    x₁^2 + x₂^2 + x₃^2 + x₄^2 = m * p ∧ 
    (∀ (m' : Nat) (y₁ y₂ y₃ y₄ : Int), 
      m' < p → 
      y₁^2 + y₂^2 + y₃^2 + y₄^2 = m' * p → 
      m ≤ m') ∧
    m = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_squares_equals_prime_multiple_l73_7317


namespace NUMINAMATH_CALUDE_percentage_of_singles_is_70_percent_l73_7385

def total_hits : ℕ := 50
def home_runs : ℕ := 3
def triples : ℕ := 2
def doubles : ℕ := 10

def singles : ℕ := total_hits - (home_runs + triples + doubles)

theorem percentage_of_singles_is_70_percent :
  (singles : ℚ) / total_hits * 100 = 70 := by sorry

end NUMINAMATH_CALUDE_percentage_of_singles_is_70_percent_l73_7385


namespace NUMINAMATH_CALUDE_ratio_problem_l73_7322

theorem ratio_problem (a b c : ℝ) : 
  a / b = 2 / 3 ∧ b / c = 3 / 4 ∧ a^2 + c^2 = 180 → b = 9 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l73_7322


namespace NUMINAMATH_CALUDE_equation_equals_twentyfour_l73_7353

theorem equation_equals_twentyfour : 8 / (3 - 8 / 3) = 24 := by
  sorry

#check equation_equals_twentyfour

end NUMINAMATH_CALUDE_equation_equals_twentyfour_l73_7353


namespace NUMINAMATH_CALUDE_circular_rug_middle_ring_area_l73_7376

theorem circular_rug_middle_ring_area :
  ∀ (inner_radius middle_radius outer_radius : ℝ)
    (inner_area middle_area outer_area : ℝ),
  inner_radius = 1 →
  middle_radius = inner_radius + 1 →
  outer_radius = middle_radius + 1 →
  inner_area = π * inner_radius^2 →
  middle_area = π * middle_radius^2 →
  outer_area = π * outer_radius^2 →
  middle_area - inner_area = 3 * π := by
sorry

end NUMINAMATH_CALUDE_circular_rug_middle_ring_area_l73_7376


namespace NUMINAMATH_CALUDE_paper_width_calculation_l73_7393

theorem paper_width_calculation (w : ℝ) : 
  (2 * w * 17 = 2 * 8.5 * 11 + 100) → w = 287 / 34 := by
  sorry

end NUMINAMATH_CALUDE_paper_width_calculation_l73_7393


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_of_72_l73_7308

theorem smallest_non_factor_product_of_72 (x y : ℕ+) : 
  x ≠ y →
  x ∣ 72 →
  y ∣ 72 →
  ¬(x * y ∣ 72) →
  (∀ a b : ℕ+, a ≠ b → a ∣ 72 → b ∣ 72 → ¬(a * b ∣ 72) → x * y ≤ a * b) →
  x * y = 32 := by
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_of_72_l73_7308


namespace NUMINAMATH_CALUDE_inverse_89_mod_90_l73_7363

theorem inverse_89_mod_90 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 89 ∧ (89 * x) % 90 = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_89_mod_90_l73_7363


namespace NUMINAMATH_CALUDE_thursday_tuesday_difference_l73_7326

/-- The amount of money Max's mom gave him on Tuesday -/
def tuesday_amount : ℕ := 8

/-- The amount of money Max's mom gave him on Wednesday -/
def wednesday_amount : ℕ := 5 * tuesday_amount

/-- The amount of money Max's mom gave him on Thursday -/
def thursday_amount : ℕ := wednesday_amount + 9

/-- The theorem stating the difference between Thursday's and Tuesday's amounts -/
theorem thursday_tuesday_difference : thursday_amount - tuesday_amount = 41 := by
  sorry

end NUMINAMATH_CALUDE_thursday_tuesday_difference_l73_7326


namespace NUMINAMATH_CALUDE_M_intersect_N_is_empty_l73_7371

-- Define the set M
def M : Set ℝ := {y | ∃ x, y = x + 2}

-- Define the set N
def N : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

-- Theorem statement
theorem M_intersect_N_is_empty : M ∩ (N.image Prod.snd) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_is_empty_l73_7371


namespace NUMINAMATH_CALUDE_constant_sum_inverse_lengths_l73_7343

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

-- Define point F
def F : ℝ × ℝ := (1, 0)

-- Define a line passing through F
def line_through_F (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define a perpendicular line passing through F
def perp_line_through_F (k : ℝ) (x y : ℝ) : Prop := y = -(1/k) * (x - 1)

-- Define the theorem
theorem constant_sum_inverse_lengths 
  (k : ℝ) 
  (A B C D : ℝ × ℝ) 
  (hA : curve A.1 A.2) (hB : curve B.1 B.2) (hC : curve C.1 C.2) (hD : curve D.1 D.2)
  (hAB : line_through_F k A.1 A.2 ∧ line_through_F k B.1 B.2)
  (hCD : perp_line_through_F k C.1 C.2 ∧ perp_line_through_F k D.1 D.2)
  (hk : k ≠ 0) :
  1 / Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) + 
  1 / Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 17/48 :=
sorry

end NUMINAMATH_CALUDE_constant_sum_inverse_lengths_l73_7343


namespace NUMINAMATH_CALUDE_albino_deer_antlers_l73_7335

theorem albino_deer_antlers (total_deer : ℕ) (albino_deer : ℕ) 
  (h1 : total_deer = 920)
  (h2 : albino_deer = 23)
  (h3 : albino_deer = (total_deer * 10 / 100) / 4) : 
  albino_deer = 23 := by
  sorry

end NUMINAMATH_CALUDE_albino_deer_antlers_l73_7335


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l73_7390

/-- The number of y-intercepts of the parabola x = 3y^2 - 4y + 2 -/
theorem parabola_y_intercepts :
  let f : ℝ → ℝ := fun y => 3 * y^2 - 4 * y + 2
  (∃ y, f y = 0) = false :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l73_7390


namespace NUMINAMATH_CALUDE_square_inequality_l73_7349

theorem square_inequality (a b : ℝ) : a > b ∧ b > 0 → a^2 > b^2 ∧ ¬(∀ x y : ℝ, x^2 > y^2 → x > y ∧ y > 0) := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l73_7349


namespace NUMINAMATH_CALUDE_cubic_root_reciprocal_sum_l73_7347

theorem cubic_root_reciprocal_sum (p q r : ℝ) : 
  p^3 - 9*p^2 + 8*p + 2 = 0 →
  q^3 - 9*q^2 + 8*q + 2 = 0 →
  r^3 - 9*r^2 + 8*r + 2 = 0 →
  p ≠ q → p ≠ r → q ≠ r →
  1/p^2 + 1/q^2 + 1/r^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_reciprocal_sum_l73_7347


namespace NUMINAMATH_CALUDE_cards_given_to_jeff_l73_7314

def initial_cards : ℕ := 573
def bought_cards : ℕ := 127
def cards_to_john : ℕ := 195
def cards_to_jimmy : ℕ := 75
def percentage_to_jeff : ℚ := 6 / 100
def final_cards : ℕ := 210

theorem cards_given_to_jeff :
  let total_cards := initial_cards + bought_cards
  let cards_after_john_jimmy := total_cards - (cards_to_john + cards_to_jimmy)
  let cards_to_jeff := (percentage_to_jeff * cards_after_john_jimmy).ceil
  cards_to_jeff = 26 ∧ 
  final_cards + cards_to_jeff = cards_after_john_jimmy :=
sorry

end NUMINAMATH_CALUDE_cards_given_to_jeff_l73_7314


namespace NUMINAMATH_CALUDE_ellipse_condition_l73_7300

/-- Represents an ellipse with foci on the y-axis -/
def is_ellipse_y_axis (m n : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ m = 1 / (a ^ 2) ∧ n = 1 / (b ^ 2)

/-- The main theorem stating that m > n > 0 is necessary and sufficient for mx^2 + ny^2 = 1 
    to represent an ellipse with foci on the y-axis -/
theorem ellipse_condition (m n : ℝ) : 
  (m > n ∧ n > 0) ↔ is_ellipse_y_axis m n := by sorry

end NUMINAMATH_CALUDE_ellipse_condition_l73_7300


namespace NUMINAMATH_CALUDE_largest_c_for_max_function_l73_7377

open Real

/-- Given positive real numbers a and b, the largest real c such that 
    c ≤ max(ax + 1/(ax), bx + 1/(bx)) for all positive real x is 2. -/
theorem largest_c_for_max_function (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∃ c : ℝ, ∀ x : ℝ, x > 0 → c ≤ max (a * x + 1 / (a * x)) (b * x + 1 / (b * x))) ∧
  (∀ c : ℝ, (∀ x : ℝ, x > 0 → c ≤ max (a * x + 1 / (a * x)) (b * x + 1 / (b * x))) → c ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_largest_c_for_max_function_l73_7377


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l73_7388

theorem inverse_proportion_problem (k : ℝ) (a b : ℝ → ℝ) :
  (∀ x, a x * (b x)^2 = k) →  -- Inverse proportion relationship
  (∃ x, a x = 40) →           -- a = 40 for some value of b
  (a (b 10) = 10) →           -- When a = 10
  b 10 = 2                    -- b = 2
:= by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l73_7388


namespace NUMINAMATH_CALUDE_complex_equation_solution_l73_7369

theorem complex_equation_solution :
  ∀ z : ℂ, (z - Complex.I) / z = Complex.I → z = -1/2 + Complex.I/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l73_7369


namespace NUMINAMATH_CALUDE_fraction_multiplication_equality_l73_7327

theorem fraction_multiplication_equality : (1 / 2) * (1 / 3) * (1 / 4) * (1 / 6) * 144 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_equality_l73_7327


namespace NUMINAMATH_CALUDE_number_difference_l73_7305

theorem number_difference (L S : ℕ) (hL : L > S) (hDiv : L = 6 * S + 20) (hLValue : L = 1634) : 
  L - S = 1365 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l73_7305


namespace NUMINAMATH_CALUDE_solution_set_for_t_equals_one_range_of_t_l73_7309

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := |x - t| + |x + t|

-- Part 1
theorem solution_set_for_t_equals_one :
  {x : ℝ | f 1 x ≤ 8 - x^2} = Set.Icc (-2) 2 := by sorry

-- Part 2
theorem range_of_t (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 4) :
  (∀ x : ℝ, f t x = (4 * m^2 + n) / (m * n)) →
  t ∈ Set.Iic (-9/8) ∪ Set.Ici (9/8) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_t_equals_one_range_of_t_l73_7309


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l73_7352

theorem isosceles_triangle_base_angle (α : ℝ) (h1 : α = 42) :
  let β := (180 - α) / 2
  (α = β) ∨ (β = 69) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l73_7352


namespace NUMINAMATH_CALUDE_toms_promotion_expenses_l73_7364

/-- Represents the problem of calculating Tom's promotion expenses --/
def TomsDoughBallPromotion (flour_needed : ℕ) (flour_bag_size : ℕ) (flour_bag_cost : ℕ)
  (salt_needed : ℕ) (salt_cost_per_pound : ℚ) (tickets_sold : ℕ) (ticket_price : ℕ) (profit : ℕ) : Prop :=
  let flour_bags := flour_needed / flour_bag_size
  let flour_cost := flour_bags * flour_bag_cost
  let salt_cost := (salt_needed : ℚ) * salt_cost_per_pound
  let revenue := tickets_sold * ticket_price
  let promotion_cost := revenue - profit - flour_cost - (salt_cost.num / salt_cost.den)
  promotion_cost = 1000

/-- The theorem stating that Tom's promotion expenses are $1000 --/
theorem toms_promotion_expenses :
  TomsDoughBallPromotion 500 50 20 10 (1/5) 500 20 8798 :=
sorry

end NUMINAMATH_CALUDE_toms_promotion_expenses_l73_7364


namespace NUMINAMATH_CALUDE_garden_trees_l73_7355

/-- The number of trees in a garden with specific planting conditions. -/
def number_of_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  yard_length / tree_distance + 1

/-- Theorem stating that the number of trees in the garden is 26. -/
theorem garden_trees :
  number_of_trees 800 32 = 26 := by
  sorry

end NUMINAMATH_CALUDE_garden_trees_l73_7355


namespace NUMINAMATH_CALUDE_basketball_probabilities_l73_7360

/-- Represents a basketball player's shooting accuracy -/
structure Player where
  accuracy : ℝ
  accuracy_nonneg : 0 ≤ accuracy
  accuracy_le_one : accuracy ≤ 1

/-- The probability of a player hitting at least one shot in two attempts -/
def prob_hit_at_least_one (player : Player) : ℝ :=
  1 - (1 - player.accuracy)^2

/-- The probability of two players making exactly three shots in four attempts -/
def prob_three_out_of_four (player_a player_b : Player) : ℝ :=
  2 * (player_a.accuracy * (1 - player_a.accuracy) * player_b.accuracy^2 +
       player_b.accuracy * (1 - player_b.accuracy) * player_a.accuracy^2)

theorem basketball_probabilities 
  (player_a : Player) 
  (player_b : Player) 
  (h_a : player_a.accuracy = 1/2) 
  (h_b : (1 - player_b.accuracy)^2 = 1/16) :
  prob_hit_at_least_one player_a = 3/4 ∧ 
  prob_three_out_of_four player_a player_b = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_basketball_probabilities_l73_7360


namespace NUMINAMATH_CALUDE_passing_percentage_is_30_l73_7350

def max_marks : ℕ := 600
def student_marks : ℕ := 80
def fail_margin : ℕ := 100

def passing_percentage : ℚ :=
  (student_marks + fail_margin : ℚ) / max_marks * 100

theorem passing_percentage_is_30 : passing_percentage = 30 := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_is_30_l73_7350


namespace NUMINAMATH_CALUDE_topsoil_cost_l73_7315

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of topsoil -/
def cubic_yards : ℝ := 8

/-- The total cost of topsoil in dollars -/
def total_cost : ℝ := cost_per_cubic_foot * cubic_feet_per_cubic_yard * cubic_yards

theorem topsoil_cost : total_cost = 1728 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_l73_7315


namespace NUMINAMATH_CALUDE_angle_D_measure_l73_7311

-- Define a scalene triangle DEF
structure ScaleneTriangle where
  D : ℝ
  E : ℝ
  F : ℝ
  scalene : D ≠ E ∧ E ≠ F ∧ D ≠ F
  sum_180 : D + E + F = 180

-- Theorem statement
theorem angle_D_measure (t : ScaleneTriangle) 
  (h1 : t.D = 2 * t.E) 
  (h2 : t.F = t.E - 20) : 
  t.D = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l73_7311


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l73_7395

/-- The total surface area of a cylinder with height 12 and radius 5 is 170π. -/
theorem cylinder_surface_area : 
  let h : ℝ := 12
  let r : ℝ := 5
  let lateral_area := 2 * Real.pi * r * h
  let base_area := 2 * Real.pi * r^2
  lateral_area + base_area = 170 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l73_7395


namespace NUMINAMATH_CALUDE_tim_golf_balls_l73_7319

/-- The number of items in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of golf balls Tim has -/
def golf_ball_dozens : ℕ := 13

/-- The total number of golf balls Tim has -/
def total_golf_balls : ℕ := golf_ball_dozens * dozen

theorem tim_golf_balls : total_golf_balls = 156 := by
  sorry

end NUMINAMATH_CALUDE_tim_golf_balls_l73_7319


namespace NUMINAMATH_CALUDE_jaime_savings_time_l73_7382

/-- Calculates the number of weeks needed to save a target amount given weekly savings and bi-weekly expenses -/
def weeksToSave (weeklySavings : ℚ) (biWeeklyExpense : ℚ) (targetAmount : ℚ) : ℚ :=
  let netBiWeeklySavings := 2 * weeklySavings - biWeeklyExpense
  let netWeeklySavings := netBiWeeklySavings / 2
  targetAmount / netWeeklySavings

/-- Proves that it takes 5 weeks to save $135 with $50 weekly savings and $46 bi-weekly expense -/
theorem jaime_savings_time : weeksToSave 50 46 135 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jaime_savings_time_l73_7382


namespace NUMINAMATH_CALUDE_unique_odd_number_with_remainder_l73_7397

theorem unique_odd_number_with_remainder : 
  ∃! n : ℕ, 30 < n ∧ n < 50 ∧ n % 2 = 1 ∧ n % 7 = 2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_odd_number_with_remainder_l73_7397


namespace NUMINAMATH_CALUDE_square_k_ascending_range_l73_7306

/-- A function f is k-ascending on a set M if for all x in M, f(x + k) ≥ f(x) --/
def IsKAscending (f : ℝ → ℝ) (k : ℝ) (M : Set ℝ) : Prop :=
  ∀ x ∈ M, f (x + k) ≥ f x

theorem square_k_ascending_range {k : ℝ} (hk : k ≠ 0) :
  IsKAscending (fun x ↦ x^2) k (Set.Ioi (-1)) → k ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_k_ascending_range_l73_7306


namespace NUMINAMATH_CALUDE_mice_pairing_impossible_l73_7310

/-- Represents the number of mice in the family -/
def total_mice : ℕ := 24

/-- Represents the number of mice that go to the warehouse each night -/
def mice_per_night : ℕ := 4

/-- Represents the number of new pairings a mouse makes each night -/
def new_pairings_per_night : ℕ := mice_per_night - 1

/-- Represents the number of pairings each mouse needs to make -/
def required_pairings : ℕ := total_mice - 1

/-- Theorem stating that it's impossible for each mouse to pair with every other mouse exactly once -/
theorem mice_pairing_impossible : 
  ¬(required_pairings % new_pairings_per_night = 0) := by sorry

end NUMINAMATH_CALUDE_mice_pairing_impossible_l73_7310


namespace NUMINAMATH_CALUDE_field_trip_attendance_calculation_l73_7339

/-- The number of people on a field trip -/
def field_trip_attendance (num_vans : ℕ) (num_buses : ℕ) (people_per_van : ℕ) (people_per_bus : ℕ) : ℕ :=
  num_vans * people_per_van + num_buses * people_per_bus

/-- Theorem stating the total number of people on the field trip -/
theorem field_trip_attendance_calculation :
  field_trip_attendance 6 8 6 18 = 180 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_attendance_calculation_l73_7339


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l73_7392

def initial_bottle_caps : ℕ := 6
def found_bottle_caps : ℕ := 22

theorem danny_bottle_caps :
  initial_bottle_caps + found_bottle_caps = 28 :=
by sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l73_7392


namespace NUMINAMATH_CALUDE_factorial_starts_with_1966_l73_7365

theorem factorial_starts_with_1966 : ∃ k : ℕ, ∃ n : ℕ, 
  1966 * 10^n ≤ k! ∧ k! < 1967 * 10^n :=
sorry

end NUMINAMATH_CALUDE_factorial_starts_with_1966_l73_7365


namespace NUMINAMATH_CALUDE_smallest_other_integer_l73_7328

theorem smallest_other_integer (m n x : ℕ+) : 
  m = 36 → 
  Nat.gcd m n = x + 5 → 
  Nat.lcm m n = x * (x + 5) → 
  ∃ (n_min : ℕ+), n_min ≤ n ∧ n_min = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l73_7328


namespace NUMINAMATH_CALUDE_third_year_planting_l73_7340

def initial_planting : ℝ := 10000
def annual_increase : ℝ := 0.2

def acres_planted (year : ℕ) : ℝ :=
  initial_planting * (1 + annual_increase) ^ (year - 1)

theorem third_year_planting :
  acres_planted 3 = 14400 := by sorry

end NUMINAMATH_CALUDE_third_year_planting_l73_7340


namespace NUMINAMATH_CALUDE_factor_count_8100_l73_7384

def number_to_factor : ℕ := 8100

/-- The number of positive factors of a natural number n -/
def count_factors (n : ℕ) : ℕ := sorry

theorem factor_count_8100 : count_factors number_to_factor = 45 := by sorry

end NUMINAMATH_CALUDE_factor_count_8100_l73_7384


namespace NUMINAMATH_CALUDE_quadratic_solution_l73_7398

theorem quadratic_solution (m : ℝ) : (2 : ℝ)^2 + m * 2 + 2 = 0 → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l73_7398


namespace NUMINAMATH_CALUDE_original_number_equation_l73_7307

/-- Given a number x, prove that when it's doubled, 15 is added, and the result is trebled, it equals 75 -/
theorem original_number_equation (x : ℝ) : 3 * (2 * x + 15) = 75 ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_equation_l73_7307


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_b_terms_is_one_l73_7399

/-- The sequence b_n defined as n! + n^2 + 1 -/
def b (n : ℕ) : ℕ := n.factorial + n^2 + 1

/-- The theorem stating that the maximum GCD of consecutive terms in the sequence is 1 -/
theorem max_gcd_consecutive_b_terms_is_one :
  ∀ n : ℕ, ∃ m : ℕ, m ≥ n → (∀ k ≥ m, Nat.gcd (b k) (b (k + 1)) = 1) ∧
    (∀ i j : ℕ, i ≥ n → j = i + 1 → Nat.gcd (b i) (b j) ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_b_terms_is_one_l73_7399


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_l73_7324

theorem square_plus_inverse_square (x : ℝ) (h : x^2 - 3*x + 1 = 0) :
  x^2 + 1/x^2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_l73_7324


namespace NUMINAMATH_CALUDE_solve_for_p_l73_7331

theorem solve_for_p (n m p : ℚ) : 
  (5 / 6 : ℚ) = n / 72 ∧ 
  (5 / 6 : ℚ) = (m + n) / 84 ∧ 
  (5 / 6 : ℚ) = (p - m) / 120 → 
  p = 110 := by
sorry

end NUMINAMATH_CALUDE_solve_for_p_l73_7331


namespace NUMINAMATH_CALUDE_angle_between_vectors_l73_7370

/-- The angle between two vectors in degrees -/
def angle_between (u v : ℝ × ℝ) : ℝ := sorry

/-- The dot product of two 2D vectors -/
def dot_product (u v : ℝ × ℝ) : ℝ := sorry

/-- The magnitude (length) of a 2D vector -/
def magnitude (u : ℝ × ℝ) : ℝ := sorry

theorem angle_between_vectors :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, -4)
  ∀ c : ℝ × ℝ,
    magnitude c = Real.sqrt 5 →
    dot_product (a.1 + b.1, a.2 + b.2) c = 5/2 →
    angle_between a c = 120 := by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l73_7370


namespace NUMINAMATH_CALUDE_convex_polygon_sides_l73_7348

theorem convex_polygon_sides (sum_except_one : ℝ) (missing_angle : ℝ) : 
  sum_except_one = 2970 ∧ missing_angle = 150 → 
  (∃ (n : ℕ), n = 20 ∧ 180 * (n - 2) = sum_except_one + missing_angle) :=
by sorry

end NUMINAMATH_CALUDE_convex_polygon_sides_l73_7348


namespace NUMINAMATH_CALUDE_sarah_picked_45_apples_l73_7367

/-- The number of apples Sarah's brother picked -/
def brother_apples : ℝ := 9.0

/-- The factor by which Sarah picked more apples than her brother -/
def sarah_factor : ℕ := 5

/-- The number of apples Sarah picked -/
def sarah_apples : ℝ := brother_apples * sarah_factor

theorem sarah_picked_45_apples : sarah_apples = 45 := by
  sorry

end NUMINAMATH_CALUDE_sarah_picked_45_apples_l73_7367


namespace NUMINAMATH_CALUDE_max_value_theorem_l73_7380

theorem max_value_theorem (a : ℝ) (h : 8 * a^2 + 6 * a + 2 = 0) :
  ∃ (max_val : ℝ), max_val = 1/4 ∧ (3 * a + 1 ≤ max_val) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l73_7380


namespace NUMINAMATH_CALUDE_tissue_paper_usage_l73_7337

theorem tissue_paper_usage (initial : ℕ) (remaining : ℕ) (used : ℕ) : 
  initial = 97 → remaining = 93 → used = initial - remaining → used = 4 := by
  sorry

end NUMINAMATH_CALUDE_tissue_paper_usage_l73_7337


namespace NUMINAMATH_CALUDE_no_sum_2017_double_digits_l73_7374

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating the impossibility of expressing 2017 as the sum of two natural numbers
    where the sum of digits of one is twice the sum of digits of the other -/
theorem no_sum_2017_double_digits : ¬ ∃ (A B : ℕ), 
  (A + B = 2017) ∧ (sumOfDigits A = 2 * sumOfDigits B) := by
  sorry

end NUMINAMATH_CALUDE_no_sum_2017_double_digits_l73_7374


namespace NUMINAMATH_CALUDE_quadratic_coefficient_determination_l73_7386

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  f : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- Theorem: If a quadratic function has vertex at (2, 5) and passes through (1, 2), then a = -3 -/
theorem quadratic_coefficient_determination (q : QuadraticFunction) 
  (vertex : q.f 2 = 5) 
  (point : q.f 1 = 2) : 
  q.a = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_determination_l73_7386


namespace NUMINAMATH_CALUDE_direction_vector_c_value_l73_7303

/-- Given a line passing through two points and a direction vector, prove the value of c. -/
theorem direction_vector_c_value (p1 p2 : ℝ × ℝ) (h : p1 = (-3, 1) ∧ p2 = (0, 4)) :
  let v : ℝ × ℝ := (p2.1 - p1.1, p2.2 - p1.2)
  v.1 = 3 → v = (3, 3) :=
by sorry

end NUMINAMATH_CALUDE_direction_vector_c_value_l73_7303


namespace NUMINAMATH_CALUDE_tan_alpha_minus_2beta_l73_7381

theorem tan_alpha_minus_2beta (α β : Real) 
  (h1 : Real.tan (α - β) = 2/5)
  (h2 : Real.tan β = 1/2) :
  Real.tan (α - 2*β) = -1/12 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_2beta_l73_7381


namespace NUMINAMATH_CALUDE_integer_roots_l73_7373

/-- A polynomial of degree 3 with integer coefficients -/
def polynomial (a₂ a₁ : ℤ) (x : ℤ) : ℤ := x^3 + a₂ * x^2 + a₁ * x - 13

/-- The set of possible integer roots -/
def possible_roots : Set ℤ := {-13, -1, 1, 13}

/-- Theorem stating that the possible integer roots of the polynomial are -13, -1, 1, and 13 -/
theorem integer_roots (a₂ a₁ : ℤ) :
  ∀ x : ℤ, polynomial a₂ a₁ x = 0 → x ∈ possible_roots :=
by sorry

end NUMINAMATH_CALUDE_integer_roots_l73_7373


namespace NUMINAMATH_CALUDE_price_per_ring_is_correct_l73_7320

/-- Calculates the price per pineapple ring given the following conditions:
  * Number of pineapples bought
  * Cost per pineapple
  * Number of rings per pineapple
  * Number of rings sold together
  * Total profit
-/
def price_per_ring (num_pineapples : ℕ) (cost_per_pineapple : ℚ) 
                   (rings_per_pineapple : ℕ) (rings_per_set : ℕ) 
                   (total_profit : ℚ) : ℚ :=
  let total_cost := num_pineapples * cost_per_pineapple
  let total_rings := num_pineapples * rings_per_pineapple
  let total_revenue := total_cost + total_profit
  let num_sets := total_rings / rings_per_set
  let price_per_set := total_revenue / num_sets
  price_per_set / rings_per_set

/-- Theorem stating that the price per pineapple ring is $1.25 under the given conditions -/
theorem price_per_ring_is_correct : 
  price_per_ring 6 3 12 4 72 = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_price_per_ring_is_correct_l73_7320


namespace NUMINAMATH_CALUDE_function_comparison_l73_7379

theorem function_comparison
  (a : ℝ)
  (h_a_lower : -3 < a)
  (h_a_upper : a < 0)
  (x₁ x₂ : ℝ)
  (h_x_order : x₁ < x₂)
  (h_x_sum : x₁ + x₂ ≠ 1 + a)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^2 + 2 * a * x + 4)
  : f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_function_comparison_l73_7379


namespace NUMINAMATH_CALUDE_sin_cos_identity_l73_7375

theorem sin_cos_identity (x : ℝ) (h : 3 * Real.sin x + Real.cos x = 0) :
  Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + Real.cos x ^ 2 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l73_7375


namespace NUMINAMATH_CALUDE_santiago_has_58_roses_l73_7332

/-- The number of red roses Mrs. Garrett has -/
def garrett_roses : ℕ := 24

/-- The number of additional roses Mrs. Santiago has compared to Mrs. Garrett -/
def additional_roses : ℕ := 34

/-- The total number of red roses Mrs. Santiago has -/
def santiago_roses : ℕ := garrett_roses + additional_roses

/-- Theorem stating that Mrs. Santiago has 58 red roses -/
theorem santiago_has_58_roses : santiago_roses = 58 := by
  sorry

end NUMINAMATH_CALUDE_santiago_has_58_roses_l73_7332


namespace NUMINAMATH_CALUDE_cos_difference_given_sum_l73_7358

theorem cos_difference_given_sum (A B : Real) 
  (h1 : Real.sin A + Real.sin B = 0.75)
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = -0.21875 := by
sorry

end NUMINAMATH_CALUDE_cos_difference_given_sum_l73_7358


namespace NUMINAMATH_CALUDE_summer_jolly_degrees_l73_7387

/-- The combined number of degrees for Summer and Jolly -/
def combined_degrees (summer_degrees : ℕ) (difference : ℕ) : ℕ :=
  summer_degrees + (summer_degrees - difference)

/-- Theorem: Given Summer has 150 degrees and 5 more degrees than Jolly,
    the combined number of degrees they both have is 295. -/
theorem summer_jolly_degrees :
  combined_degrees 150 5 = 295 := by
  sorry

end NUMINAMATH_CALUDE_summer_jolly_degrees_l73_7387


namespace NUMINAMATH_CALUDE_repeating_decimal_56_equals_fraction_l73_7389

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

theorem repeating_decimal_56_equals_fraction :
  RepeatingDecimal 5 6 = 56 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_56_equals_fraction_l73_7389


namespace NUMINAMATH_CALUDE_school_gender_ratio_l73_7323

/-- Given a school with a 5:4 ratio of boys to girls and 1500 boys, prove there are 1200 girls -/
theorem school_gender_ratio (num_boys : ℕ) (num_girls : ℕ) : 
  num_boys = 1500 →
  (5 : ℚ) / 4 = num_boys / num_girls →
  num_girls = 1200 := by
sorry

end NUMINAMATH_CALUDE_school_gender_ratio_l73_7323
