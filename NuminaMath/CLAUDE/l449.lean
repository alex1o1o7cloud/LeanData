import Mathlib

namespace NUMINAMATH_CALUDE_animal_food_cost_l449_44945

/-- The total weekly cost for both animals' food -/
def total_weekly_cost : ℕ := 30

/-- The weekly cost of rabbit food -/
def rabbit_weekly_cost : ℕ := 12

/-- The number of weeks Julia has had the rabbit -/
def rabbit_weeks : ℕ := 5

/-- The number of weeks Julia has had the parrot -/
def parrot_weeks : ℕ := 3

/-- The total amount Julia has spent on animal food -/
def total_spent : ℕ := 114

theorem animal_food_cost :
  total_weekly_cost = rabbit_weekly_cost + (total_spent - rabbit_weekly_cost * rabbit_weeks) / parrot_weeks :=
by sorry

end NUMINAMATH_CALUDE_animal_food_cost_l449_44945


namespace NUMINAMATH_CALUDE_nell_initial_cards_l449_44953

/-- The number of cards Nell gave away -/
def cards_given_away : ℕ := 276

/-- The number of cards Nell has left -/
def cards_left : ℕ := 252

/-- Nell's initial number of cards -/
def initial_cards : ℕ := cards_given_away + cards_left

theorem nell_initial_cards : initial_cards = 528 := by
  sorry

end NUMINAMATH_CALUDE_nell_initial_cards_l449_44953


namespace NUMINAMATH_CALUDE_last_four_average_l449_44962

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 65 →
  (list.take 3).sum / 3 = 60 →
  (list.drop 3).sum / 4 = 68.75 := by
  sorry

end NUMINAMATH_CALUDE_last_four_average_l449_44962


namespace NUMINAMATH_CALUDE_smallest_height_of_special_triangle_l449_44957

/-- Given a scalene triangle with integer side lengths a, b, c satisfying 
    the relation (a^2/c) - (a-c)^2 = (b^2/c) - (b-c)^2, 
    the smallest height of the triangle is 12/5. -/
theorem smallest_height_of_special_triangle (a b c : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hscalene : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (hrelation : (a^2 : ℚ)/c - (a-c)^2 = (b^2 : ℚ)/c - (b-c)^2) :
  ∃ h : ℚ, h = 12/5 ∧ h = min (2 * (a * b) / (2 * a)) (min (2 * (b * c) / (2 * b)) (2 * (a * c) / (2 * c))) :=
sorry

end NUMINAMATH_CALUDE_smallest_height_of_special_triangle_l449_44957


namespace NUMINAMATH_CALUDE_perimeter_difference_is_zero_l449_44921

/-- The perimeter of a rectangle -/
def rectangle_perimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- The perimeter of Shape 1 -/
def shape1_perimeter : ℕ :=
  rectangle_perimeter 4 3

/-- The perimeter of Shape 2 -/
def shape2_perimeter : ℕ :=
  rectangle_perimeter 6 1

/-- The positive difference between the perimeters of Shape 1 and Shape 2 -/
def perimeter_difference : ℕ :=
  Int.natAbs (shape1_perimeter - shape2_perimeter)

theorem perimeter_difference_is_zero : perimeter_difference = 0 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_is_zero_l449_44921


namespace NUMINAMATH_CALUDE_triangle_area_l449_44998

/-- The area of a triangle with base 30 inches and height 18 inches is 270 square inches. -/
theorem triangle_area (base height : ℝ) (h1 : base = 30) (h2 : height = 18) :
  (1 / 2 : ℝ) * base * height = 270 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l449_44998


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l449_44996

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), a ≡ 61 [ZMOD 70] ∧ b ≡ 43 [ZMOD 70] ∧ (3 * a + 9 * b) ≡ 0 [ZMOD 70] := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l449_44996


namespace NUMINAMATH_CALUDE_average_cookies_per_package_l449_44915

def cookie_counts : List Nat := [9, 11, 13, 19, 23, 27]

theorem average_cookies_per_package : 
  (List.sum cookie_counts) / (List.length cookie_counts) = 17 := by
  sorry

end NUMINAMATH_CALUDE_average_cookies_per_package_l449_44915


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l449_44905

theorem max_sum_of_squares (m n : ℕ) : 
  1 ≤ m ∧ m ≤ 1981 ∧ 1 ≤ n ∧ n ≤ 1981 →
  (n^2 - m*n - m^2)^2 = 1 →
  m^2 + n^2 ≤ 3524578 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l449_44905


namespace NUMINAMATH_CALUDE_angle_triple_complement_l449_44974

theorem angle_triple_complement (x : ℝ) : x = 3 * (90 - x) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l449_44974


namespace NUMINAMATH_CALUDE_sum_of_arguments_l449_44982

def complex_pow_eq (z : ℂ) : Prop := z^5 = -32 * Complex.I

theorem sum_of_arguments (z₁ z₂ z₃ z₄ z₅ : ℂ) 
  (h₁ : complex_pow_eq z₁) (h₂ : complex_pow_eq z₂) (h₃ : complex_pow_eq z₃) 
  (h₄ : complex_pow_eq z₄) (h₅ : complex_pow_eq z₅) 
  (distinct : z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₁ ≠ z₅ ∧ 
              z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₂ ≠ z₅ ∧ 
              z₃ ≠ z₄ ∧ z₃ ≠ z₅ ∧ 
              z₄ ≠ z₅) :
  Complex.arg z₁ + Complex.arg z₂ + Complex.arg z₃ + Complex.arg z₄ + Complex.arg z₅ = 
  990 * (π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_arguments_l449_44982


namespace NUMINAMATH_CALUDE_problem_statement_l449_44954

theorem problem_statement (x y : ℝ) (h1 : x * y = 12) (h2 : x + y = -8) :
  y * Real.sqrt (x / y) + x * Real.sqrt (y / x) = -4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l449_44954


namespace NUMINAMATH_CALUDE_inequality_solution_set_l449_44923

def inequality (a x : ℝ) : Prop := a * x^2 - (a + 1) * x + 1 > 0

def solution_set (a : ℝ) : Set ℝ :=
  if a < 0 then Set.Ioo (1/a) 1
  else if 0 < a ∧ a < 1 then Set.Iio 1 ∪ Set.Ioi (1/a)
  else if a = 1 then Set.Iio 1 ∪ Set.Ioi 1
  else Set.Iio (1/a) ∪ Set.Ioi 1

theorem inequality_solution_set (a : ℝ) (h : a ≠ 0) :
  {x : ℝ | inequality a x} = solution_set a :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l449_44923


namespace NUMINAMATH_CALUDE_triangle_right_angle_l449_44959

theorem triangle_right_angle (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- sides are positive
  0 < A ∧ 0 < B ∧ 0 < C →  -- angles are positive
  A + B + C = π →  -- sum of angles in a triangle
  a * Real.cos A + b * Real.cos B = c * Real.cos C →  -- given condition
  a^2 = b^2 + c^2  -- conclusion: right triangle with a as hypotenuse
  := by sorry

end NUMINAMATH_CALUDE_triangle_right_angle_l449_44959


namespace NUMINAMATH_CALUDE_x_less_than_y_l449_44902

theorem x_less_than_y (n : ℕ) (x y : ℝ) 
  (hn : n > 2) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hxn : x^n = x + 1) 
  (hyn : y^(n+1) = y^3 + 1) : 
  x < y :=
by sorry

end NUMINAMATH_CALUDE_x_less_than_y_l449_44902


namespace NUMINAMATH_CALUDE_lottery_increment_proof_l449_44983

/-- Represents the increment in the price of each successive ticket -/
def increment : ℝ := 1

/-- The number of lottery tickets -/
def num_tickets : ℕ := 5

/-- The price of the first ticket -/
def first_ticket_price : ℝ := 1

/-- The profit Lily plans to keep -/
def profit : ℝ := 4

/-- The prize money for the lottery winner -/
def prize : ℝ := 11

/-- The total amount collected from selling all tickets -/
def total_collected (x : ℝ) : ℝ :=
  first_ticket_price + (first_ticket_price + x) + (first_ticket_price + 2*x) + 
  (first_ticket_price + 3*x) + (first_ticket_price + 4*x)

theorem lottery_increment_proof :
  total_collected increment = profit + prize :=
sorry

end NUMINAMATH_CALUDE_lottery_increment_proof_l449_44983


namespace NUMINAMATH_CALUDE_prime_factors_count_four_equals_two_squared_seven_is_prime_eleven_is_prime_l449_44924

/-- The total number of prime factors in the expression (4)^11 × (7)^5 × (11)^2 -/
def totalPrimeFactors : ℕ := 29

/-- The exponent of 4 in the expression -/
def exponent4 : ℕ := 11

/-- The exponent of 7 in the expression -/
def exponent7 : ℕ := 5

/-- The exponent of 11 in the expression -/
def exponent11 : ℕ := 2

theorem prime_factors_count :
  totalPrimeFactors = 2 * exponent4 + exponent7 + exponent11 := by
  sorry

/-- 4 is equal to 2^2 -/
theorem four_equals_two_squared : (4 : ℕ) = 2^2 := by
  sorry

/-- 7 is a prime number -/
theorem seven_is_prime : Nat.Prime 7 := by
  sorry

/-- 11 is a prime number -/
theorem eleven_is_prime : Nat.Prime 11 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_count_four_equals_two_squared_seven_is_prime_eleven_is_prime_l449_44924


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l449_44948

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

-- Define the function g
def g (a b k x : ℝ) : ℝ := f a b x - k * x

-- State the theorem
theorem quadratic_function_properties (a b : ℝ) (ha : a ≠ 0) :
  (f a b (-1) = 0) →
  (∀ x : ℝ, f a b x ≥ 0) →
  (a = 1 ∧ b = 2) ∧
  (∀ k : ℝ, (∀ x ∈ Set.Icc (-2) 2, Monotone (g a b k)) ↔ k ≤ -2 ∨ k ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l449_44948


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_x_plus_y_8_l449_44993

theorem coefficient_x3y5_in_x_plus_y_8 :
  Finset.sum (Finset.range 9) (λ k => Nat.choose 8 k * (if k = 3 then 1 else 0)) = 56 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_x_plus_y_8_l449_44993


namespace NUMINAMATH_CALUDE_minimum_additional_games_l449_44901

def initial_games : ℕ := 3
def initial_wins : ℕ := 2
def target_percentage : ℚ := 9/10

def winning_percentage (additional_games : ℕ) : ℚ :=
  (initial_wins + additional_games) / (initial_games + additional_games)

theorem minimum_additional_games :
  ∃ N : ℕ, (∀ n : ℕ, n < N → winning_percentage n < target_percentage) ∧
            winning_percentage N ≥ target_percentage ∧
            N = 7 :=
by sorry

end NUMINAMATH_CALUDE_minimum_additional_games_l449_44901


namespace NUMINAMATH_CALUDE_inequality_solution_l449_44990

theorem inequality_solution (x : ℝ) : 
  (x / 4 ≤ 3 + 2 * x ∧ 3 + 2 * x < -3 * (1 + 2 * x)) ↔ 
  x ∈ Set.Icc (-12/7) (-3/4) ∧ x ≠ -3/4 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l449_44990


namespace NUMINAMATH_CALUDE_simplify_expression_l449_44952

theorem simplify_expression (a b c : ℝ) 
  (h : Real.sqrt (a - 5) + (b - 3)^2 = Real.sqrt (c - 4) + Real.sqrt (4 - c)) :
  Real.sqrt c / (Real.sqrt a - Real.sqrt b) = Real.sqrt 5 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l449_44952


namespace NUMINAMATH_CALUDE_graces_pool_filling_time_l449_44978

/-- The problem of filling Grace's pool --/
theorem graces_pool_filling_time 
  (pool_capacity : ℝ) 
  (first_hose_rate : ℝ) 
  (second_hose_rate : ℝ) 
  (additional_time : ℝ) 
  (h : pool_capacity = 390) 
  (r1 : first_hose_rate = 50) 
  (r2 : second_hose_rate = 70) 
  (t : additional_time = 2) :
  ∃ (wait_time : ℝ), 
    wait_time * first_hose_rate + 
    additional_time * (first_hose_rate + second_hose_rate) = 
    pool_capacity ∧ wait_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_graces_pool_filling_time_l449_44978


namespace NUMINAMATH_CALUDE_mrs_hilt_total_chapters_l449_44956

/-- The total number of chapters Mrs. Hilt has read -/
def total_chapters_read : ℕ :=
  let last_month_17ch := 4 * 17
  let last_month_25ch := 3 * 25
  let last_month_30ch := 2 * 30
  let this_month_book1 := 18
  let this_month_book2 := 24
  last_month_17ch + last_month_25ch + last_month_30ch + this_month_book1 + this_month_book2

/-- Theorem stating that Mrs. Hilt has read 245 chapters in total -/
theorem mrs_hilt_total_chapters : total_chapters_read = 245 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_total_chapters_l449_44956


namespace NUMINAMATH_CALUDE_sum_u_v_l449_44994

theorem sum_u_v (u v : ℚ) (h1 : 5 * u - 6 * v = 19) (h2 : 3 * u + 5 * v = -1) : 
  u + v = 27 / 43 := by
  sorry

end NUMINAMATH_CALUDE_sum_u_v_l449_44994


namespace NUMINAMATH_CALUDE_participating_countries_form_set_l449_44942

/-- A type representing countries --/
structure Country where
  name : String

/-- A type representing a specific event --/
structure Event where
  name : String
  year : Nat

/-- A predicate that determines if a country participated in an event --/
def participated (country : Country) (event : Event) : Prop := sorry

/-- Definition of a set with definite elements --/
def isDefiniteSet (S : Set α) : Prop :=
  ∀ x, (x ∈ S) ∨ (x ∉ S)

/-- Theorem stating that countries participating in a specific event form a definite set --/
theorem participating_countries_form_set (event : Event) :
  isDefiniteSet {country : Country | participated country event} := by
  sorry

end NUMINAMATH_CALUDE_participating_countries_form_set_l449_44942


namespace NUMINAMATH_CALUDE_ratio_is_one_to_two_l449_44946

/-- Represents a co-ed softball team -/
structure CoedSoftballTeam where
  men : ℕ
  women : ℕ
  total_players : ℕ
  women_more_than_men : women = men + 5
  total_is_sum : total_players = men + women

/-- The ratio of men to women in a co-ed softball team -/
def ratio_men_to_women (team : CoedSoftballTeam) : ℚ × ℚ :=
  (team.men, team.women)

theorem ratio_is_one_to_two (team : CoedSoftballTeam) 
    (h : team.total_players = 15) : 
    ratio_men_to_women team = (1, 2) := by
  sorry

#check ratio_is_one_to_two

end NUMINAMATH_CALUDE_ratio_is_one_to_two_l449_44946


namespace NUMINAMATH_CALUDE_trapezoid_cd_length_l449_44909

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  /-- Length of side BD -/
  bd : ℝ
  /-- Angle DBA in radians -/
  angle_dba : ℝ
  /-- Angle BDC in radians -/
  angle_bdc : ℝ
  /-- Ratio of BC to AD -/
  ratio_bc_ad : ℝ
  /-- AD is parallel to BC -/
  ad_parallel_bc : True
  /-- BD equals 3 -/
  bd_eq_three : bd = 3
  /-- Angle DBA equals 30 degrees (π/6 radians) -/
  angle_dba_eq_thirty_deg : angle_dba = Real.pi / 6
  /-- Angle BDC equals 60 degrees (π/3 radians) -/
  angle_bdc_eq_sixty_deg : angle_bdc = Real.pi / 3
  /-- Ratio of BC to AD is 7:4 -/
  ratio_bc_ad_eq_seven_four : ratio_bc_ad = 7 / 4

/-- Theorem: In the given trapezoid, CD equals 9/4 -/
theorem trapezoid_cd_length (t : Trapezoid) : ∃ cd : ℝ, cd = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_cd_length_l449_44909


namespace NUMINAMATH_CALUDE_pet_store_kittens_l449_44951

theorem pet_store_kittens (initial : ℕ) (final : ℕ) (new : ℕ) : 
  initial = 6 → final = 9 → new = final - initial → new = 3 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_kittens_l449_44951


namespace NUMINAMATH_CALUDE_atMostOneHead_atLeastTwoHeads_mutually_exclusive_atMostOneHead_atLeastTwoHeads_cover_all_l449_44987

-- Define the sample space for tossing two coins
inductive CoinToss
  | HH -- Two heads
  | HT -- Head then tail
  | TH -- Tail then head
  | TT -- Two tails

-- Define the events
def atMostOneHead (outcome : CoinToss) : Prop :=
  outcome = CoinToss.HT ∨ outcome = CoinToss.TH ∨ outcome = CoinToss.TT

def atLeastTwoHeads (outcome : CoinToss) : Prop :=
  outcome = CoinToss.HH

-- Theorem: The events are mutually exclusive
theorem atMostOneHead_atLeastTwoHeads_mutually_exclusive :
  ∀ (outcome : CoinToss), ¬(atMostOneHead outcome ∧ atLeastTwoHeads outcome) :=
by
  sorry

-- Theorem: The events cover all possible outcomes
theorem atMostOneHead_atLeastTwoHeads_cover_all :
  ∀ (outcome : CoinToss), atMostOneHead outcome ∨ atLeastTwoHeads outcome :=
by
  sorry

end NUMINAMATH_CALUDE_atMostOneHead_atLeastTwoHeads_mutually_exclusive_atMostOneHead_atLeastTwoHeads_cover_all_l449_44987


namespace NUMINAMATH_CALUDE_juanita_dessert_cost_l449_44931

/-- Calculates the cost of Juanita's dessert given the prices of individual items --/
def dessert_cost (brownie_price ice_cream_price syrup_price nuts_price : ℚ) : ℚ :=
  brownie_price + 2 * ice_cream_price + 2 * syrup_price + nuts_price

/-- Proves that Juanita's dessert costs $7.00 given the prices of individual items --/
theorem juanita_dessert_cost :
  dessert_cost 2.5 1 0.5 1.5 = 7 := by
  sorry

#eval dessert_cost 2.5 1 0.5 1.5

end NUMINAMATH_CALUDE_juanita_dessert_cost_l449_44931


namespace NUMINAMATH_CALUDE_classroom_boys_count_l449_44991

theorem classroom_boys_count (initial_girls : ℕ) : 
  let initial_boys := initial_girls + 5
  let final_girls := initial_girls + 10
  let final_boys := initial_boys + 3
  final_girls = 22 →
  final_boys = 20 := by
sorry

end NUMINAMATH_CALUDE_classroom_boys_count_l449_44991


namespace NUMINAMATH_CALUDE_selling_price_theorem_l449_44936

/-- Calculates the selling price per tire given production costs and profit -/
def selling_price_per_tire (cost_per_batch : ℝ) (cost_per_tire : ℝ) (batch_size : ℕ) (profit_per_tire : ℝ) : ℝ :=
  cost_per_tire + profit_per_tire

/-- Theorem: The selling price per tire is the sum of cost per tire and profit per tire -/
theorem selling_price_theorem (cost_per_batch : ℝ) (cost_per_tire : ℝ) (batch_size : ℕ) (profit_per_tire : ℝ) :
  selling_price_per_tire cost_per_batch cost_per_tire batch_size profit_per_tire = cost_per_tire + profit_per_tire :=
by sorry

#eval selling_price_per_tire 22500 8 15000 10.5

end NUMINAMATH_CALUDE_selling_price_theorem_l449_44936


namespace NUMINAMATH_CALUDE_max_omega_for_increasing_g_l449_44966

/-- Given a function f and its translation g, proves that the maximum value of ω is 2 
    when g is increasing on [0, π/4] -/
theorem max_omega_for_increasing_g (ω : ℝ) (f g : ℝ → ℝ) : 
  ω > 0 → 
  (∀ x, f x = 2 * Real.sin (ω * x - π / 8)) →
  (∀ x, g x = f (x + π / (8 * ω))) →
  (∀ x ∈ Set.Icc 0 (π / 4), Monotone g) →
  ω ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_omega_for_increasing_g_l449_44966


namespace NUMINAMATH_CALUDE_negative_two_inequality_l449_44908

theorem negative_two_inequality (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_two_inequality_l449_44908


namespace NUMINAMATH_CALUDE_martha_turtles_l449_44960

theorem martha_turtles (martha_turtles : ℕ) (marion_turtles : ℕ) : 
  marion_turtles = martha_turtles + 20 →
  martha_turtles + marion_turtles = 100 →
  martha_turtles = 40 := by
sorry

end NUMINAMATH_CALUDE_martha_turtles_l449_44960


namespace NUMINAMATH_CALUDE_unique_integer_factorial_division_l449_44919

theorem unique_integer_factorial_division : 
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 60 ∧ (∃ k : ℕ, k * (Nat.factorial n)^(n + 2) = Nat.factorial (n^2)) :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_factorial_division_l449_44919


namespace NUMINAMATH_CALUDE_canoe_kayak_difference_is_five_l449_44933

/-- Represents the rental information for canoes and kayaks --/
structure RentalInfo where
  canoe_cost : ℕ
  kayak_cost : ℕ
  canoe_kayak_ratio : Rat
  total_revenue : ℕ

/-- Calculates the difference between canoes and kayaks rented --/
def canoe_kayak_difference (info : RentalInfo) : ℕ :=
  let kayaks := (info.total_revenue : ℚ) * 3 / (11 * 4 + 16 * 3)
  let canoes := kayaks * info.canoe_kayak_ratio
  (canoes - kayaks).ceil.toNat

/-- Theorem stating the difference between canoes and kayaks rented --/
theorem canoe_kayak_difference_is_five (info : RentalInfo) 
  (h1 : info.canoe_cost = 11)
  (h2 : info.kayak_cost = 16)
  (h3 : info.canoe_kayak_ratio = 4 / 3)
  (h4 : info.total_revenue = 460) :
  canoe_kayak_difference info = 5 := by
  sorry

#eval canoe_kayak_difference { 
  canoe_cost := 11, 
  kayak_cost := 16, 
  canoe_kayak_ratio := 4 / 3, 
  total_revenue := 460 
}

end NUMINAMATH_CALUDE_canoe_kayak_difference_is_five_l449_44933


namespace NUMINAMATH_CALUDE_evaluate_expression_l449_44904

theorem evaluate_expression : 2001^3 - 2000 * 2001^2 - 2000^2 * 2001 + 2 * 2000^3 = 24008004001 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l449_44904


namespace NUMINAMATH_CALUDE_smallest_common_pet_count_l449_44937

theorem smallest_common_pet_count : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → 
    (∃ (x : ℕ), x > 1 ∧ 2 ∣ m ∧ x ∣ m) → 
    n ≤ m) ∧ 
  (∃ (x : ℕ), x > 1 ∧ 2 ∣ n ∧ x ∣ n) ∧
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_pet_count_l449_44937


namespace NUMINAMATH_CALUDE_wall_area_calculation_l449_44907

/-- The area of a rectangular wall with width and height of 4 feet is 16 square feet. -/
theorem wall_area_calculation : 
  ∀ (width height area : ℝ), 
  width = 4 → 
  height = 4 → 
  area = width * height → 
  area = 16 :=
by sorry

end NUMINAMATH_CALUDE_wall_area_calculation_l449_44907


namespace NUMINAMATH_CALUDE_unique_prime_root_equation_l449_44938

theorem unique_prime_root_equation :
  ∀ p q n : ℕ,
    Prime p → Prime q → n > 0 →
    (p + q : ℝ) ^ (1 / n : ℝ) = p - q →
    p = 5 ∧ q = 3 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_root_equation_l449_44938


namespace NUMINAMATH_CALUDE_students_playing_sports_l449_44976

theorem students_playing_sports (A B : Finset ℕ) : 
  A.card = 7 → B.card = 8 → (A ∩ B).card = 3 → (A ∪ B).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_sports_l449_44976


namespace NUMINAMATH_CALUDE_problem_solution_l449_44947

theorem problem_solution (p q r u v w : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l449_44947


namespace NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l449_44999

theorem cube_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 10) 
  (h2 : x * y = 12) : 
  x^3 + y^3 = 640 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l449_44999


namespace NUMINAMATH_CALUDE_quadratic_function_with_specific_properties_l449_44968

theorem quadratic_function_with_specific_properties :
  ∀ (a b x₁ x₂ : ℝ),
    a < 0 →
    b > 0 →
    x₁ ≠ x₂ →
    x₁^2 + a*x₁ + b = 0 →
    x₂^2 + a*x₂ + b = 0 →
    ((x₁ - (-2) = x₂ - x₁) ∨ (x₁ / (-2) = x₂ / x₁)) →
    (∀ x, x^2 + a*x + b = x^2 - 5*x + 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_with_specific_properties_l449_44968


namespace NUMINAMATH_CALUDE_jake_not_dropping_coffee_percentage_l449_44944

-- Define the probabilities
def trip_probability : ℝ := 0.4
def drop_coffee_when_tripping_probability : ℝ := 0.25

-- Theorem to prove
theorem jake_not_dropping_coffee_percentage :
  1 - (trip_probability * drop_coffee_when_tripping_probability) = 0.9 :=
by sorry

end NUMINAMATH_CALUDE_jake_not_dropping_coffee_percentage_l449_44944


namespace NUMINAMATH_CALUDE_terminal_zeros_125_360_l449_44997

def number_of_terminal_zeros (a b : ℕ) : ℕ :=
  sorry

theorem terminal_zeros_125_360 : 
  let a := 125
  let b := 360
  number_of_terminal_zeros a b = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_terminal_zeros_125_360_l449_44997


namespace NUMINAMATH_CALUDE_sin_arccos_circle_l449_44934

theorem sin_arccos_circle (x y : ℝ) :
  y = Real.sin (Real.arccos x) ↔ x^2 + y^2 = 1 ∧ x ∈ Set.Icc (-1) 1 ∧ y ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_arccos_circle_l449_44934


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_l449_44927

/-- A triangular pyramid with specific properties -/
structure SpecialPyramid where
  -- Base side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Condition: base side lengths are √85, √58, and √45
  ha : a = Real.sqrt 85
  hb : b = Real.sqrt 58
  hc : c = Real.sqrt 45
  -- Condition: lateral edges are mutually perpendicular
  lateral_edges_perpendicular : Bool

/-- The sphere inscribed in the special pyramid -/
structure InscribedSphere (p : SpecialPyramid) where
  -- The radius of the sphere
  radius : ℝ
  -- Condition: The sphere touches all lateral faces
  touches_all_faces : Bool
  -- Condition: The center of the sphere lies on the base
  center_on_base : Bool

/-- The main theorem stating the radius of the inscribed sphere -/
theorem inscribed_sphere_radius (p : SpecialPyramid) (s : InscribedSphere p) :
  s.touches_all_faces ∧ s.center_on_base ∧ p.lateral_edges_perpendicular →
  s.radius = 14 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_l449_44927


namespace NUMINAMATH_CALUDE_dance_partners_exist_l449_44973

-- Define the types for boys and girls
variable {Boy Girl : Type}

-- Define the dance relation
variable (danced : Boy → Girl → Prop)

-- Define the conditions
variable (h1 : ∀ b : Boy, ∃ g : Girl, ¬danced b g)
variable (h2 : ∀ g : Girl, ∃ b : Boy, danced b g)

-- State the theorem
theorem dance_partners_exist :
  ∃ (f f' : Boy) (g g' : Girl), f ≠ f' ∧ g ≠ g' ∧ danced f g ∧ danced f' g' :=
sorry

end NUMINAMATH_CALUDE_dance_partners_exist_l449_44973


namespace NUMINAMATH_CALUDE_range_of_inequality_l449_44958

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def monotonic_increasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y → f x < f y

def even_function_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 2) = f (-x - 2)

-- State the theorem
theorem range_of_inequality (h1 : monotonic_increasing_on_interval f) 
                            (h2 : even_function_shifted f) :
  {x : ℝ | f (2 * x) < f (x + 2)} = Set.Ioo (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_inequality_l449_44958


namespace NUMINAMATH_CALUDE_broadway_ticket_price_l449_44916

theorem broadway_ticket_price (num_adults num_children : ℕ) (total_amount : ℚ) :
  num_adults = 400 →
  num_children = 200 →
  total_amount = 16000 →
  ∃ (adult_price child_price : ℚ),
    adult_price = 2 * child_price ∧
    num_adults * adult_price + num_children * child_price = total_amount ∧
    adult_price = 32 := by
  sorry

end NUMINAMATH_CALUDE_broadway_ticket_price_l449_44916


namespace NUMINAMATH_CALUDE_base_for_256_with_4_digits_l449_44972

theorem base_for_256_with_4_digits : ∃ (b : ℕ), b = 5 ∧ b^3 ≤ 256 ∧ 256 < b^4 ∧ ∀ (x : ℕ), x < b → (x^3 ≤ 256 → 256 ≥ x^4) := by
  sorry

end NUMINAMATH_CALUDE_base_for_256_with_4_digits_l449_44972


namespace NUMINAMATH_CALUDE_committee_formation_count_l449_44935

theorem committee_formation_count :
  let dept_A : Finset ℕ := Finset.range 6
  let dept_B : Finset ℕ := Finset.range 7
  let dept_C : Finset ℕ := Finset.range 5
  (dept_A.card * dept_B.card * dept_C.card : ℕ) = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l449_44935


namespace NUMINAMATH_CALUDE_existence_of_n_l449_44988

theorem existence_of_n (k : ℕ+) : ∃ n : ℤ, 
  Real.sqrt (n + 1981^k.val : ℝ) + Real.sqrt (n : ℝ) = (Real.sqrt 1982 + 1)^k.val := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_l449_44988


namespace NUMINAMATH_CALUDE_x_plus_y_value_l449_44967

theorem x_plus_y_value (x y : ℝ) (hx : |x| = 2) (hy : |y| = 5) (hxy : x < y) :
  x + y = 7 ∨ x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l449_44967


namespace NUMINAMATH_CALUDE_g_neither_even_nor_odd_l449_44941

noncomputable def g (x : ℝ) : ℝ := Real.log (x + 2 + Real.sqrt (1 + (x + 2)^2))

theorem g_neither_even_nor_odd : 
  (∀ x, g (-x) = g x) ∧ (∀ x, g (-x) = -g x) → False := by
  sorry

end NUMINAMATH_CALUDE_g_neither_even_nor_odd_l449_44941


namespace NUMINAMATH_CALUDE_difference_of_squares_l449_44911

theorem difference_of_squares (a : ℝ) : a * a - (a - 1) * (a + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l449_44911


namespace NUMINAMATH_CALUDE_one_shot_each_probability_l449_44975

def yao_rate : ℝ := 0.8
def mcgrady_rate : ℝ := 0.7

theorem one_shot_each_probability :
  let yao_one_shot := 2 * yao_rate * (1 - yao_rate)
  let mcgrady_one_shot := 2 * mcgrady_rate * (1 - mcgrady_rate)
  yao_one_shot * mcgrady_one_shot = 0.1344 := by
sorry

end NUMINAMATH_CALUDE_one_shot_each_probability_l449_44975


namespace NUMINAMATH_CALUDE_difference_of_sums_l449_44964

/-- Sum of first n odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n^2

/-- Sum of first n even numbers -/
def sumEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- The number of odd numbers from 1 to 2011 -/
def oddCount : ℕ := 1006

/-- The number of even numbers from 2 to 2010 -/
def evenCount : ℕ := 1005

theorem difference_of_sums : sumOddNumbers oddCount - sumEvenNumbers evenCount = 1006 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_sums_l449_44964


namespace NUMINAMATH_CALUDE_equation_real_roots_range_l449_44984

-- Define the equation
def equation (x m : ℝ) : ℝ := 25 - |x + 1| - 4 * 5 - |x + 1| - m

-- Define the property of having real roots
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, equation x m = 0

-- Theorem statement
theorem equation_real_roots_range :
  ∀ m : ℝ, has_real_roots m ↔ m ∈ Set.Ioo (-3 : ℝ) 0 :=
sorry

end NUMINAMATH_CALUDE_equation_real_roots_range_l449_44984


namespace NUMINAMATH_CALUDE_simplify_fraction_l449_44950

theorem simplify_fraction : (144 : ℚ) / 12672 = 1 / 88 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l449_44950


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l449_44989

def income : ℕ := 21000
def savings : ℕ := 3000
def expenditure : ℕ := income - savings

def ratio_income_expenditure : ℚ := income / expenditure

theorem income_expenditure_ratio :
  ratio_income_expenditure = 7 / 6 := by sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l449_44989


namespace NUMINAMATH_CALUDE_sum_abcd_equals_negative_fourteen_thirds_l449_44912

theorem sum_abcd_equals_negative_fourteen_thirds 
  (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 7) : 
  a + b + c + d = -14/3 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_negative_fourteen_thirds_l449_44912


namespace NUMINAMATH_CALUDE_unique_solution_l449_44900

def is_valid_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

def satisfies_equation (Θ : ℕ) : Prop :=
  is_valid_digit Θ ∧ 
  (198 : ℚ) / Θ = (40 : ℚ) + 2 * Θ

theorem unique_solution : 
  ∃! Θ : ℕ, satisfies_equation Θ ∧ Θ = 4 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l449_44900


namespace NUMINAMATH_CALUDE_xy_plus_y_squared_l449_44913

theorem xy_plus_y_squared (x y : ℝ) (h : x * (x + y) = x^2 + 3*y + 12) :
  x*y + y^2 = y^2 + 3*y + 12 := by
  sorry

end NUMINAMATH_CALUDE_xy_plus_y_squared_l449_44913


namespace NUMINAMATH_CALUDE_divisibility_by_133_l449_44910

theorem divisibility_by_133 (n : ℕ) : ∃ k : ℤ, 11^(n+2) + 12^(2*n+1) = 133 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_133_l449_44910


namespace NUMINAMATH_CALUDE_spade_nested_calculation_l449_44970

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_nested_calculation : spade 5 (spade 3 (spade 8 12)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_spade_nested_calculation_l449_44970


namespace NUMINAMATH_CALUDE_max_binder_price_l449_44914

/-- Proves that the maximum whole-dollar price of a binder is $7 given the conditions --/
theorem max_binder_price (total_money : ℕ) (num_binders : ℕ) (entrance_fee : ℕ) (tax_rate : ℚ) : 
  total_money = 160 →
  num_binders = 18 →
  entrance_fee = 5 →
  tax_rate = 8 / 100 →
  ∃ (price : ℕ), price = 7 ∧ 
    price = ⌊(total_money - entrance_fee) / ((1 + tax_rate) * num_binders)⌋ ∧
    ∀ (p : ℕ), p > price → 
      p * num_binders * (1 + tax_rate) + entrance_fee > total_money :=
by
  sorry

#check max_binder_price

end NUMINAMATH_CALUDE_max_binder_price_l449_44914


namespace NUMINAMATH_CALUDE_problem_solution_l449_44906

theorem problem_solution (x : ℝ) : 0.8 * x - 20 = 60 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l449_44906


namespace NUMINAMATH_CALUDE_triangular_arrangement_rows_l449_44961

/-- The number of cans in a triangular arrangement with n rows -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The proposition to be proved -/
theorem triangular_arrangement_rows : 
  ∃ (n : ℕ), triangular_sum n = 480 - 15 ∧ n = 30 := by
  sorry

end NUMINAMATH_CALUDE_triangular_arrangement_rows_l449_44961


namespace NUMINAMATH_CALUDE_pattern_theorem_l449_44930

/-- Function to create a number with the first n digits of 123456... -/
def firstNDigits (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (firstNDigits (n-1)) * 10 + n

/-- Function to create a number with n ones -/
def nOnes (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (nOnes (n-1)) * 10 + 1

/-- Theorem stating the pattern observed in the problem -/
theorem pattern_theorem (n : ℕ) : 
  (firstNDigits n) * 9 + (n + 1) = nOnes (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_pattern_theorem_l449_44930


namespace NUMINAMATH_CALUDE_quadrilateral_angle_measure_l449_44918

theorem quadrilateral_angle_measure (W X Y Z : ℝ) : 
  W > 0 ∧ X > 0 ∧ Y > 0 ∧ Z > 0 →
  W = 3 * X ∧ W = 4 * Y ∧ W = 6 * Z →
  W + X + Y + Z = 360 →
  W = 1440 / 7 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_measure_l449_44918


namespace NUMINAMATH_CALUDE_probability_at_least_three_aces_l449_44995

/-- The number of cards in a standard deck without jokers -/
def deck_size : ℕ := 52

/-- The number of Aces in a standard deck -/
def num_aces : ℕ := 4

/-- The number of cards drawn -/
def draw_size : ℕ := 5

/-- The probability of drawing at least 3 Aces when randomly selecting 5 cards from a standard 52-card deck (without jokers) -/
theorem probability_at_least_three_aces :
  (Nat.choose num_aces 3 * Nat.choose (deck_size - num_aces) 2 +
   Nat.choose num_aces 4 * Nat.choose (deck_size - num_aces) 1) /
  Nat.choose deck_size draw_size =
  (Nat.choose 4 3 * Nat.choose 48 2 + Nat.choose 4 4 * Nat.choose 48 1) /
  Nat.choose 52 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_three_aces_l449_44995


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l449_44977

theorem integer_pairs_satisfying_equation :
  ∀ (x y : ℤ), x^5 + y^5 = (x + y)^3 ↔
    ((x = 0 ∧ y = 1) ∨
     (x = 1 ∧ y = 0) ∨
     (x = 0 ∧ y = -1) ∨
     (x = -1 ∧ y = 0) ∨
     (x = 2 ∧ y = 2) ∨
     (x = -2 ∧ y = -2) ∨
     (∃ (a : ℤ), x = a ∧ y = -a)) :=
by sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l449_44977


namespace NUMINAMATH_CALUDE_distance_PQ_l449_44971

def P : ℝ × ℝ := (-1, 2)
def Q : ℝ × ℝ := (3, 0)

theorem distance_PQ : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_PQ_l449_44971


namespace NUMINAMATH_CALUDE_min_value_of_sum_l449_44940

theorem min_value_of_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (sum_eq : a + b + c = 3) (prod_sum_eq : a * b + b * c + a * c = 2) :
  a + b ≥ (6 - 2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l449_44940


namespace NUMINAMATH_CALUDE_jonah_aquarium_fish_count_l449_44949

/-- The number of fish in Jonah's aquarium after all the changes -/
def final_fish_count (initial_fish : ℕ) (added_fish : ℕ) (exchanged_fish : ℕ) (x : ℕ) : ℤ :=
  (initial_fish + added_fish : ℤ) - 2 * x + exchanged_fish

theorem jonah_aquarium_fish_count :
  final_fish_count 14 2 3 x = 19 - 2 * x :=
by sorry

end NUMINAMATH_CALUDE_jonah_aquarium_fish_count_l449_44949


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l449_44929

/-- Given two vectors a and b in R², where a is parallel to (a - b), 
    prove that the magnitude of their sum is 3√5/2. -/
theorem vector_sum_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 1]
  (∃ (k : ℝ), a = k • (a - b)) → 
  ‖a + b‖ = (3 * Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l449_44929


namespace NUMINAMATH_CALUDE_odd_function_negative_x_l449_44925

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_x 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_positive : ∀ x > 0, f x = -x + 1) :
  ∀ x < 0, f x = -x - 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_x_l449_44925


namespace NUMINAMATH_CALUDE_calories_per_dollar_difference_l449_44965

-- Define the given conditions
def burrito_count : ℕ := 10
def burrito_price : ℚ := 6
def burrito_calories : ℕ := 120
def burger_count : ℕ := 5
def burger_price : ℚ := 8
def burger_calories : ℕ := 400

-- Define the theorem
theorem calories_per_dollar_difference :
  (burger_count * burger_calories : ℚ) / burger_price -
  (burrito_count * burrito_calories : ℚ) / burrito_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_calories_per_dollar_difference_l449_44965


namespace NUMINAMATH_CALUDE_problem_solution_l449_44928

theorem problem_solution : 2.017 * 2016 - 10.16 * 201.7 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l449_44928


namespace NUMINAMATH_CALUDE_distinguishable_triangles_count_l449_44992

/-- The number of available colors for the triangles -/
def num_colors : ℕ := 8

/-- The number of corner triangles in the large triangle -/
def num_corners : ℕ := 3

/-- The total number of small triangles in the large triangle -/
def total_triangles : ℕ := num_corners + 1

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of distinguishable large triangles -/
def num_distinguishable_triangles : ℕ :=
  (num_colors + 
   num_colors * (num_colors - 1) + 
   choose num_colors num_corners) * num_colors

theorem distinguishable_triangles_count :
  num_distinguishable_triangles = 960 :=
sorry

end NUMINAMATH_CALUDE_distinguishable_triangles_count_l449_44992


namespace NUMINAMATH_CALUDE_average_text_messages_l449_44979

/-- Calculate the average number of text messages sent over 5 days -/
theorem average_text_messages 
  (day1 : ℕ) 
  (day2 : ℕ) 
  (day3_to_5 : ℕ) 
  (h1 : day1 = 220) 
  (h2 : day2 = day1 / 2) 
  (h3 : day3_to_5 = 50) :
  (day1 + day2 + 3 * day3_to_5) / 5 = 96 := by
  sorry

end NUMINAMATH_CALUDE_average_text_messages_l449_44979


namespace NUMINAMATH_CALUDE_floor_of_negative_three_point_seven_l449_44943

theorem floor_of_negative_three_point_seven :
  ⌊(-3.7 : ℝ)⌋ = -4 := by sorry

end NUMINAMATH_CALUDE_floor_of_negative_three_point_seven_l449_44943


namespace NUMINAMATH_CALUDE_existence_of_critical_point_and_positive_function_l449_44922

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp (m * x) - Real.log x - 2

theorem existence_of_critical_point_and_positive_function :
  (∃ t : ℝ, t ∈ Set.Ioo (1/2) 1 ∧ ∀ y : ℝ, y ∈ Set.Ioo (1/2) 1 → (deriv (f 1)) t = 0 ∧ (deriv (f 1)) y = 0 → y = t) ∧
  (∃ m : ℝ, m ∈ Set.Ioo 0 1 ∧ ∀ x : ℝ, x > 0 → f m x > 0) :=
sorry

end NUMINAMATH_CALUDE_existence_of_critical_point_and_positive_function_l449_44922


namespace NUMINAMATH_CALUDE_new_average_income_after_death_l449_44926

/-- Calculates the new average income after a member's death -/
def new_average_income (original_members : ℕ) (original_average : ℚ) (deceased_income : ℚ) : ℚ :=
  (original_members * original_average - deceased_income) / (original_members - 1)

/-- Theorem: The new average income after a member's death is 650 -/
theorem new_average_income_after_death :
  new_average_income 4 782 1178 = 650 := by
  sorry

end NUMINAMATH_CALUDE_new_average_income_after_death_l449_44926


namespace NUMINAMATH_CALUDE_divisible_by_eight_l449_44917

theorem divisible_by_eight (b n : ℕ) (h1 : Even b) (h2 : b > 0) (h3 : n > 1)
  (h4 : ∃ k : ℕ, (b^n - 1) / (b - 1) = k^2) : 
  8 ∣ b := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eight_l449_44917


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_divisibility_l449_44955

theorem infinitely_many_pairs_divisibility :
  ∀ k : ℕ, ∃ n m : ℕ, (n + m)^2 / (n + 7) = k :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_divisibility_l449_44955


namespace NUMINAMATH_CALUDE_average_speed_round_trip_l449_44932

theorem average_speed_round_trip (speed_xy speed_yx : ℝ) (h1 : speed_xy = 43) (h2 : speed_yx = 34) :
  (2 * speed_xy * speed_yx) / (speed_xy + speed_yx) = 38 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_round_trip_l449_44932


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l449_44963

def p (x : ℂ) : ℂ := 5 * x^5 + 18 * x^3 - 45 * x^2 + 30 * x

theorem roots_of_polynomial :
  ∀ x : ℂ, p x = 0 ↔ x = 0 ∨ x = 1/5 ∨ x = Complex.I * Real.sqrt 3 ∨ x = -Complex.I * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l449_44963


namespace NUMINAMATH_CALUDE_craig_appliance_sales_l449_44980

/-- The number of appliances sold by Craig in a week -/
def num_appliances : ℕ := 6

/-- The total selling price of appliances in dollars -/
def total_selling_price : ℚ := 3620

/-- The total commission Craig earned in dollars -/
def total_commission : ℚ := 662

/-- The fixed commission per appliance in dollars -/
def fixed_commission : ℚ := 50

/-- The percentage of selling price Craig receives as commission -/
def commission_rate : ℚ := 1/10

theorem craig_appliance_sales :
  num_appliances = 6 ∧
  (num_appliances : ℚ) * fixed_commission + commission_rate * total_selling_price = total_commission :=
sorry

end NUMINAMATH_CALUDE_craig_appliance_sales_l449_44980


namespace NUMINAMATH_CALUDE_F_is_odd_l449_44981

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define F in terms of f
def F (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f x - f (-x)

-- Theorem: F is an odd function
theorem F_is_odd (f : ℝ → ℝ) : ∀ x : ℝ, F f (-x) = -(F f x) :=
by
  sorry

end NUMINAMATH_CALUDE_F_is_odd_l449_44981


namespace NUMINAMATH_CALUDE_three_card_selections_standard_deck_l449_44985

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (red_suits : Nat)
  (black_suits : Nat)

/-- A standard deck of cards -/
def standard_deck : Deck :=
  { total_cards := 52
  , num_suits := 4
  , cards_per_suit := 13
  , red_suits := 2
  , black_suits := 2 }

/-- The number of ways to choose three different cards in a specific order -/
def three_card_selections (d : Deck) : Nat :=
  d.total_cards * (d.total_cards - 1) * (d.total_cards - 2)

/-- Theorem stating that the number of ways to choose three different cards
    in a specific order from a standard deck is 132600 -/
theorem three_card_selections_standard_deck :
  three_card_selections standard_deck = 132600 := by
  sorry

end NUMINAMATH_CALUDE_three_card_selections_standard_deck_l449_44985


namespace NUMINAMATH_CALUDE_smallest_gcd_qr_l449_44986

theorem smallest_gcd_qr (p q r : ℕ+) 
  (h1 : Nat.gcd p q = 540)
  (h2 : Nat.gcd p r = 1080) :
  ∃ (m : ℕ+), 
    (∀ (q' r' : ℕ+), Nat.gcd p q' = 540 → Nat.gcd p r' = 1080 → m ≤ Nat.gcd q' r') ∧
    Nat.gcd q r = m :=
  sorry

end NUMINAMATH_CALUDE_smallest_gcd_qr_l449_44986


namespace NUMINAMATH_CALUDE_transport_theorem_l449_44920

-- Define the capacity of a worker per hour
def worker_capacity : ℝ := 30

-- Define the capacity of a robot per hour
def robot_capacity : ℝ := 450

-- Define the number of robots
def num_robots : ℕ := 3

-- Define the total amount to be transported
def total_amount : ℝ := 3600

-- Define the time limit
def time_limit : ℝ := 2

-- Define the function to calculate the minimum number of additional workers
def min_additional_workers : ℕ := 15

theorem transport_theorem :
  -- Condition 1: Robot carries 420kg more than a worker
  (robot_capacity = worker_capacity + 420) →
  -- Condition 2: Time for robot to carry 900kg equals time for 10 workers to carry 600kg
  (900 / robot_capacity = 600 / (10 * worker_capacity)) →
  -- Condition 3 & 4 are implicitly used in the conclusion
  -- Conclusion 1: Robot capacity is 450kg per hour
  (robot_capacity = 450) ∧
  -- Conclusion 2: Worker capacity is 30kg per hour
  (worker_capacity = 30) ∧
  -- Conclusion 3: Minimum additional workers needed is 15
  (min_additional_workers = 15 ∧
   robot_capacity * num_robots * time_limit + worker_capacity * min_additional_workers * time_limit ≥ total_amount ∧
   ∀ n : ℕ, n < 15 → robot_capacity * num_robots * time_limit + worker_capacity * n * time_limit < total_amount) :=
by sorry

end NUMINAMATH_CALUDE_transport_theorem_l449_44920


namespace NUMINAMATH_CALUDE_mapping_A_to_B_l449_44969

def f (x : ℝ) : ℝ := 2 * x - 1

def B : Set ℝ := {-3, -1, 3}

theorem mapping_A_to_B :
  ∃ A : Set ℝ, (∀ x ∈ A, f x ∈ B) ∧ (∀ y ∈ B, ∃ x ∈ A, f x = y) ∧ A = {-1, 0, 2} := by
  sorry

end NUMINAMATH_CALUDE_mapping_A_to_B_l449_44969


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l449_44903

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  a : ℤ  -- First term
  d : ℤ  -- Common difference

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.a + seq.d * (n - 1)

/-- Sum of the first n terms of an arithmetic sequence -/
def ArithmeticSequence.sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (2 * seq.a + seq.d * (n - 1)) / 2

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  seq.nthTerm 7 = 4 ∧ seq.nthTerm 8 = 10 ∧ seq.nthTerm 9 = 16 →
  seq.sumFirstN 5 = -100 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l449_44903


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l449_44939

/-- The value of d for which the line y = 3x + d is tangent to the parabola y² = 12x -/
theorem tangent_line_to_parabola : ∃! d : ℝ,
  ∀ x y : ℝ, (y = 3 * x + d ∧ y^2 = 12 * x) →
  (∃! x₀ y₀ : ℝ, y₀ = 3 * x₀ + d ∧ y₀^2 = 12 * x₀) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l449_44939
