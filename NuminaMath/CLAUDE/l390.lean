import Mathlib

namespace f_composition_at_one_over_e_l390_39057

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.exp x else Real.log x

-- State the theorem
theorem f_composition_at_one_over_e :
  f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end f_composition_at_one_over_e_l390_39057


namespace michael_fish_count_l390_39028

theorem michael_fish_count (initial_fish : Float) (given_fish : Float) : 
  initial_fish = 49.0 → given_fish = 18.0 → initial_fish + given_fish = 67.0 := by
  sorry

end michael_fish_count_l390_39028


namespace fraction_comparison_l390_39072

theorem fraction_comparison (a b c d : ℚ) 
  (h1 : a / b < c / d) 
  (h2 : b > d) 
  (h3 : d > 0) : 
  (a + c) / (b + d) < (1 / 2) * (a / b + c / d) := by
  sorry

end fraction_comparison_l390_39072


namespace cyclic_quadrilateral_area_l390_39061

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
def CyclicQuadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

/-- The distance between two points in a 2D plane. -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- The area of a quadrilateral given its four vertices. -/
def quadrilateralArea (A B C D : ℝ × ℝ) : ℝ := sorry

theorem cyclic_quadrilateral_area 
  (A B C D : ℝ × ℝ) 
  (h_cyclic : CyclicQuadrilateral A B C D)
  (h_AB : distance A B = 2)
  (h_BC : distance B C = 6)
  (h_CD : distance C D = 4)
  (h_DA : distance D A = 4) :
  quadrilateralArea A B C D = 8 * Real.sqrt 3 := by
  sorry

end cyclic_quadrilateral_area_l390_39061


namespace quadratic_inequality_necessary_condition_l390_39058

theorem quadratic_inequality_necessary_condition (m : ℝ) :
  (∀ x : ℝ, x^2 - x + m > 0) → m > 0 := by
  sorry

end quadratic_inequality_necessary_condition_l390_39058


namespace geometric_sequence_ratio_l390_39046

/-- For a geometric sequence with positive terms and common ratio q where q^2 = 4,
    the ratio (a_3 + a_4) / (a_4 + a_5) equals 1/2. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- common ratio is q
  q^2 = 4 →
  (a 3 + a 4) / (a 4 + a 5) = 1/2 :=
by sorry

end geometric_sequence_ratio_l390_39046


namespace sequence_square_l390_39091

theorem sequence_square (a : ℕ → ℕ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n - 1) →
  ∀ n : ℕ, n ≥ 1 → a n = n^2 :=
by sorry

end sequence_square_l390_39091


namespace eric_erasers_l390_39041

/-- Given that Eric shares his erasers among 99 friends and each friend gets 94 erasers,
    prove that Eric has 9306 erasers in total. -/
theorem eric_erasers (num_friends : ℕ) (erasers_per_friend : ℕ) 
    (h1 : num_friends = 99) (h2 : erasers_per_friend = 94) : 
    num_friends * erasers_per_friend = 9306 := by
  sorry

end eric_erasers_l390_39041


namespace complex_value_theorem_l390_39069

theorem complex_value_theorem (z : ℂ) (h : (1 - z) / (1 + z) = I) : 
  Complex.abs (z + 1) = Real.sqrt 2 := by sorry

end complex_value_theorem_l390_39069


namespace smallest_perfect_square_multiplier_l390_39010

theorem smallest_perfect_square_multiplier : ∃ (n : ℕ), 
  (7 * n = 7 * 7) ∧ 
  (∃ (m : ℕ), m * m = 7 * n) ∧
  (∀ (k : ℕ), k < 7 → ¬∃ (m : ℕ), m * m = k * n) := by
  sorry

end smallest_perfect_square_multiplier_l390_39010


namespace chimney_bricks_proof_l390_39087

/-- The number of hours it takes Brenda to build the chimney alone -/
def brenda_time : ℝ := 8

/-- The number of hours it takes Brandon to build the chimney alone -/
def brandon_time : ℝ := 12

/-- The decrease in combined output when working together (in bricks per hour) -/
def output_decrease : ℝ := 15

/-- The number of hours it takes Brenda and Brandon to build the chimney together -/
def combined_time : ℝ := 6

/-- The number of bricks in the chimney -/
def chimney_bricks : ℝ := 360

theorem chimney_bricks_proof : 
  combined_time * ((chimney_bricks / brenda_time + chimney_bricks / brandon_time) - output_decrease) = chimney_bricks := by
  sorry

end chimney_bricks_proof_l390_39087


namespace max_fly_path_2x1x1_box_l390_39024

/-- Represents a rectangular box with dimensions a, b, and c -/
structure Box where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the maximum path length for a fly in the given box -/
def maxFlyPathLength (box : Box) : ℝ :=
  sorry

/-- Theorem stating the maximum fly path length for a 2x1x1 box -/
theorem max_fly_path_2x1x1_box :
  let box : Box := { a := 2, b := 1, c := 1 }
  maxFlyPathLength box = 4 + 4 * Real.sqrt 5 + Real.sqrt 6 :=
by sorry

end max_fly_path_2x1x1_box_l390_39024


namespace certain_number_divisibility_l390_39008

theorem certain_number_divisibility (m : ℕ+) 
  (h1 : ∃ (k : ℕ+), m = 8 * k) 
  (h2 : ∀ (d : ℕ+), d ∣ m → d ≤ 8) : 
  64 ∣ m^2 ∧ ∀ (n : ℕ+), (∀ (k : ℕ+), n ∣ (8*k)^2) → n ≤ 64 :=
by sorry

end certain_number_divisibility_l390_39008


namespace circle_diameter_from_area_l390_39004

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) :
  A = 4 * Real.pi →
  A = Real.pi * r^2 →
  d = 2 * r →
  d = 4 := by
  sorry

end circle_diameter_from_area_l390_39004


namespace rachel_money_theorem_l390_39011

def rachel_money_left (initial_amount : ℚ) (lunch_fraction : ℚ) (dvd_fraction : ℚ) : ℚ :=
  initial_amount - (lunch_fraction * initial_amount) - (dvd_fraction * initial_amount)

theorem rachel_money_theorem :
  rachel_money_left 200 (1/4) (1/2) = 50 := by
  sorry

end rachel_money_theorem_l390_39011


namespace factor_expression_l390_39030

theorem factor_expression (x : ℝ) : 4 * x * (x - 2) + 6 * (x - 2) = 2 * (x - 2) * (2 * x + 3) := by
  sorry

end factor_expression_l390_39030


namespace polynomial_product_l390_39034

-- Define the polynomials f, g, and h
def f (x : ℝ) : ℝ := x^4 - x^3 - 1
def g (x : ℝ) : ℝ := x^8 - x^6 - 2*x^4 + 1
def h (x : ℝ) : ℝ := x^4 + x^3 - 1

-- State the theorem
theorem polynomial_product :
  (∀ x, g x = f x * h x) → (∀ x, h x = x^4 + x^3 - 1) := by
  sorry

end polynomial_product_l390_39034


namespace sum_of_fifth_powers_l390_39078

theorem sum_of_fifth_powers (ζ₁ ζ₂ ζ₃ : ℂ) 
  (sum_1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (sum_2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 6)
  (sum_3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 8) :
  ζ₁^5 + ζ₂^5 + ζ₃^5 = 20 := by
  sorry

end sum_of_fifth_powers_l390_39078


namespace intersection_complement_theorem_l390_39022

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 6}
def B : Set Nat := {2, 3, 4}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {1, 6} := by
  sorry

end intersection_complement_theorem_l390_39022


namespace lowest_cost_scheme_l390_39054

-- Define excavator types
inductive ExcavatorType
| A
| B

-- Define the excavation capacity for each type
def excavation_capacity (t : ExcavatorType) : ℝ :=
  match t with
  | ExcavatorType.A => 30
  | ExcavatorType.B => 15

-- Define the hourly cost for each type
def hourly_cost (t : ExcavatorType) : ℝ :=
  match t with
  | ExcavatorType.A => 300
  | ExcavatorType.B => 180

-- Define the total excavation function
def total_excavation (a b : ℕ) : ℝ :=
  4 * (a * excavation_capacity ExcavatorType.A + b * excavation_capacity ExcavatorType.B)

-- Define the total cost function
def total_cost (a b : ℕ) : ℝ :=
  4 * (a * hourly_cost ExcavatorType.A + b * hourly_cost ExcavatorType.B)

-- Theorem statement
theorem lowest_cost_scheme :
  ∀ a b : ℕ,
    a + b = 12 →
    total_excavation a b ≥ 1080 →
    total_cost a b ≤ 12960 →
    total_cost 7 5 ≤ total_cost a b ∧
    total_cost 7 5 = 12000 :=
sorry

end lowest_cost_scheme_l390_39054


namespace base_number_proof_l390_39003

theorem base_number_proof (x : ℝ) : x^8 = 4^16 → x = 16 := by
  sorry

end base_number_proof_l390_39003


namespace warehouse_weight_limit_l390_39040

theorem warehouse_weight_limit (P : ℕ) (certain_weight : ℝ) : 
  (P : ℝ) * 0.3 < 75 ∧ 
  (P : ℝ) * 0.2 = 48 ∧ 
  (P : ℝ) * 0.8 ≥ certain_weight ∧ 
  24 ≥ certain_weight ∧ 24 < 75 →
  certain_weight = 75 := by
sorry

end warehouse_weight_limit_l390_39040


namespace second_employee_hourly_rate_l390_39099

/-- Proves that the hourly rate of the second employee before subsidy is $22 -/
theorem second_employee_hourly_rate 
  (first_employee_rate : ℝ)
  (subsidy : ℝ)
  (weekly_savings : ℝ)
  (hours_per_week : ℝ)
  (h1 : first_employee_rate = 20)
  (h2 : subsidy = 6)
  (h3 : weekly_savings = 160)
  (h4 : hours_per_week = 40)
  : ∃ (second_employee_rate : ℝ), 
    hours_per_week * first_employee_rate - hours_per_week * (second_employee_rate - subsidy) = weekly_savings ∧ 
    second_employee_rate = 22 :=
by sorry

end second_employee_hourly_rate_l390_39099


namespace range_of_a_l390_39092

theorem range_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 * Real.exp (y/x) - a * y^3 = 0) : a ≥ Real.exp 3 / 27 := by
  sorry

end range_of_a_l390_39092


namespace expression_simplification_l390_39006

theorem expression_simplification (x y z : ℝ) :
  (x - (2 * y + z)) - ((x + 2 * y) - 3 * z) = -4 * y + 2 * z := by
  sorry

end expression_simplification_l390_39006


namespace moms_approach_is_sampling_survey_l390_39016

/-- Represents a method of data collection. -/
inductive DataCollectionMethod
| Census
| SamplingSurvey

/-- Represents the action of tasting food. -/
structure TastingAction where
  dish : String
  portion : String

/-- Determines the data collection method based on the tasting action. -/
def determineMethod (action : TastingAction) : DataCollectionMethod :=
  if action.portion = "entire" then DataCollectionMethod.Census
  else DataCollectionMethod.SamplingSurvey

theorem moms_approach_is_sampling_survey :
  let momsTasting : TastingAction := { dish := "cooking dish", portion := "little bit" }
  determineMethod momsTasting = DataCollectionMethod.SamplingSurvey := by
  sorry


end moms_approach_is_sampling_survey_l390_39016


namespace final_pen_count_l390_39032

def pen_collection (initial_pens : ℕ) (mike_gives : ℕ) (sharon_takes : ℕ) : ℕ :=
  let after_mike := initial_pens + mike_gives
  let after_cindy := 2 * after_mike
  after_cindy - sharon_takes

theorem final_pen_count :
  pen_collection 7 22 19 = 39 := by
  sorry

end final_pen_count_l390_39032


namespace set_membership_implies_value_l390_39074

theorem set_membership_implies_value (a : ℝ) : 
  3 ∈ ({1, a, a - 2} : Set ℝ) → a = 5 := by
  sorry

end set_membership_implies_value_l390_39074


namespace total_oil_leaked_equals_11687_l390_39036

/-- The amount of oil leaked before repairs, in liters -/
def oil_leaked_before : ℕ := 6522

/-- The amount of oil leaked during repairs, in liters -/
def oil_leaked_during : ℕ := 5165

/-- The total amount of oil leaked, in liters -/
def total_oil_leaked : ℕ := oil_leaked_before + oil_leaked_during

theorem total_oil_leaked_equals_11687 : total_oil_leaked = 11687 := by
  sorry

end total_oil_leaked_equals_11687_l390_39036


namespace weight_of_new_person_l390_39089

theorem weight_of_new_person (initial_count : ℕ) (weight_increase : ℝ) (leaving_weight : ℝ) :
  initial_count = 12 →
  weight_increase = 4 →
  leaving_weight = 58 →
  (initial_count : ℝ) * weight_increase + leaving_weight = 106 :=
by
  sorry

end weight_of_new_person_l390_39089


namespace profit_percentage_calculation_l390_39007

def selling_price : ℝ := 900
def profit : ℝ := 300

theorem profit_percentage_calculation : 
  (profit / (selling_price - profit)) * 100 = 50 := by
sorry

end profit_percentage_calculation_l390_39007


namespace special_square_numbers_l390_39050

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- A function that returns the first two digits of a six-digit number -/
def first_two_digits (n : ℕ) : ℕ :=
  n / 10000

/-- A function that returns the middle two digits of a six-digit number -/
def middle_two_digits (n : ℕ) : ℕ :=
  (n / 100) % 100

/-- A function that returns the last two digits of a six-digit number -/
def last_two_digits (n : ℕ) : ℕ :=
  n % 100

/-- A function that checks if all digits of a six-digit number are non-zero -/
def all_digits_nonzero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ [n / 100000, (n / 10000) % 10, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10] → d ≠ 0

/-- The main theorem stating that there are exactly 2 special square numbers -/
theorem special_square_numbers :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, 100000 ≤ n ∧ n < 1000000 ∧
              all_digits_nonzero n ∧
              is_perfect_square n ∧
              is_perfect_square (first_two_digits n) ∧
              is_perfect_square (middle_two_digits n) ∧
              is_perfect_square (last_two_digits n)) ∧
    s.card = 2 := by
  sorry


end special_square_numbers_l390_39050


namespace mod_congruence_unique_solution_l390_39064

theorem mod_congruence_unique_solution : ∃! n : ℕ, n ≤ 19 ∧ n ≡ -5678 [ZMOD 20] ∧ n = 2 := by
  sorry

end mod_congruence_unique_solution_l390_39064


namespace original_ribbon_length_is_correct_l390_39080

/-- The length of ribbon tape used for one gift in meters -/
def ribbon_per_gift : ℝ := 0.84

/-- The number of gifts prepared -/
def num_gifts : ℕ := 10

/-- The length of leftover ribbon tape in meters -/
def leftover_ribbon : ℝ := 0.5

/-- The total length of the original ribbon tape in meters -/
def original_ribbon_length : ℝ := ribbon_per_gift * num_gifts + leftover_ribbon

theorem original_ribbon_length_is_correct :
  original_ribbon_length = 8.9 := by sorry

end original_ribbon_length_is_correct_l390_39080


namespace total_wheat_weight_l390_39084

def wheat_weights : List ℝ := [91, 91, 91.5, 89, 91.2, 91.3, 88.7, 88.8, 91.8, 91.1]
def standard_weight : ℝ := 90

theorem total_wheat_weight :
  (wheat_weights.sum) = 905.4 := by
  sorry

end total_wheat_weight_l390_39084


namespace packaging_cost_per_cake_l390_39068

/-- Proves that the cost of packaging per cake is $1 -/
theorem packaging_cost_per_cake
  (ingredient_cost_two_cakes : ℝ)
  (selling_price_per_cake : ℝ)
  (profit_per_cake : ℝ)
  (h1 : ingredient_cost_two_cakes = 12)
  (h2 : selling_price_per_cake = 15)
  (h3 : profit_per_cake = 8) :
  selling_price_per_cake - profit_per_cake - (ingredient_cost_two_cakes / 2) = 1 := by
  sorry

end packaging_cost_per_cake_l390_39068


namespace prince_cd_spend_l390_39098

/-- Calculates the amount spent on CDs given the total number of CDs,
    percentage of expensive CDs, prices, and buying pattern. -/
def calculate_cd_spend (total_cds : ℕ) (expensive_percentage : ℚ) 
                       (expensive_price cheap_price : ℚ) 
                       (expensive_bought_ratio : ℚ) : ℚ :=
  let expensive_cds : ℚ := expensive_percentage * total_cds
  let cheap_cds : ℚ := (1 - expensive_percentage) * total_cds
  let expensive_bought : ℚ := expensive_bought_ratio * expensive_cds
  expensive_bought * expensive_price + cheap_cds * cheap_price

/-- Proves that Prince spent $1000 on CDs given the problem conditions. -/
theorem prince_cd_spend : 
  calculate_cd_spend 200 (40/100) 10 5 (1/2) = 1000 := by
  sorry

end prince_cd_spend_l390_39098


namespace sum_of_squares_l390_39081

theorem sum_of_squares (x y z : ℝ) :
  (x + y + z) / 3 = 10 →
  (x * y * z) ^ (1/3 : ℝ) = 7 →
  3 / (1/x + 1/y + 1/z) = 4 →
  x^2 + y^2 + z^2 = 385.5 := by
  sorry

end sum_of_squares_l390_39081


namespace nines_in_hundred_l390_39063

def count_nines (n : ℕ) : ℕ :=
  (n / 10) + (n / 10)

theorem nines_in_hundred : count_nines 100 = 20 := by sorry

end nines_in_hundred_l390_39063


namespace trig_identity_l390_39067

theorem trig_identity (x z : ℝ) : 
  (Real.sin x)^2 + (Real.sin (x + z))^2 - 2 * (Real.sin x) * (Real.sin z) * (Real.sin (x + z)) = (Real.sin z)^2 := by
  sorry

end trig_identity_l390_39067


namespace function_composition_ratio_l390_39090

/-- Given two functions f and g, prove that f(g(f(3))) / g(f(g(3))) = 59/19 -/
theorem function_composition_ratio
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = 3 * x + 2)
  (hg : ∀ x, g x = 2 * x - 3) :
  f (g (f 3)) / g (f (g 3)) = 59 / 19 :=
by sorry

end function_composition_ratio_l390_39090


namespace cube_surface_area_l390_39005

/-- The surface area of a cube with edge length 5 cm is 150 cm². -/
theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 5) :
  6 * edge_length ^ 2 = 150 := by
  sorry

end cube_surface_area_l390_39005


namespace min_value_of_expression_l390_39002

theorem min_value_of_expression (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 9) :
  ∃ (m : ℝ), m = 36 ∧ ∀ (a b : ℝ),
    (a = (Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12))^2 ∧
     b = (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))^2) →
    a - b ≥ m :=
by sorry

end min_value_of_expression_l390_39002


namespace minimum_candies_in_can_l390_39044

theorem minimum_candies_in_can (red green : ℕ) : 
  (red > 0) →
  (green > 0) →
  ((3 * red) / 5 : ℚ) = (3 / 8) * ((3 * red) / 5 + (2 * green) / 5) →
  (∀ r g : ℕ, r > 0 ∧ g > 0 ∧ ((3 * r) / 5 : ℚ) = (3 / 8) * ((3 * r) / 5 + (2 * g) / 5) → r + g ≥ red + green) →
  red + green = 35 :=
by sorry

end minimum_candies_in_can_l390_39044


namespace factor_expression_l390_39045

theorem factor_expression (x : ℝ) : 75*x + 45 = 15*(5*x + 3) := by
  sorry

end factor_expression_l390_39045


namespace real_part_of_z_l390_39079

def i : ℂ := Complex.I

theorem real_part_of_z (z : ℂ) (h : (1 + 2*i)*z = 3 + 4*i) : 
  Complex.re z = 11/5 := by
  sorry

end real_part_of_z_l390_39079


namespace hyperbola_asymptotes_l390_39019

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, real axis length 2, and focal distance 4,
    prove that its asymptotes are y = ±√3 x -/
theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (2 * a = 2) →  -- real axis length is 2
  (4 = 2 * Real.sqrt (a^2 + b^2)) →  -- focal distance is 4
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ (y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x)) :=
by sorry

end hyperbola_asymptotes_l390_39019


namespace parabola_translation_l390_39059

-- Define the initial parabola
def initial_parabola (x : ℝ) : ℝ := (x - 1)^2 - 4

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := (x - 4)^2 - 2

-- Define the translation
def translation_right : ℝ := 3
def translation_up : ℝ := 2

-- Theorem statement
theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = initial_parabola (x - translation_right) + translation_up :=
by sorry

end parabola_translation_l390_39059


namespace peters_mothers_age_l390_39051

/-- Proves that Peter's mother's age is 60 given the problem conditions -/
theorem peters_mothers_age :
  ∀ (harriet_age peter_age mother_age : ℕ),
    harriet_age = 13 →
    peter_age + 4 = 2 * (harriet_age + 4) →
    peter_age = mother_age / 2 →
    mother_age = 60 := by
  sorry

end peters_mothers_age_l390_39051


namespace emily_spent_28_dollars_l390_39048

/-- Calculates the total cost of Emily's flower purchase --/
def emily_flower_cost (rose_price daisy_price tulip_price lily_price : ℕ)
  (rose_qty daisy_qty tulip_qty lily_qty : ℕ) : ℕ :=
  rose_price * rose_qty + daisy_price * daisy_qty + tulip_price * tulip_qty + lily_price * lily_qty

/-- Proves that Emily spent 28 dollars on flowers --/
theorem emily_spent_28_dollars :
  emily_flower_cost 4 3 5 6 2 3 1 1 = 28 := by
  sorry

end emily_spent_28_dollars_l390_39048


namespace key_lime_juice_yield_l390_39073

def recipe_amount : ℚ := 1/4
def tablespoons_per_cup : ℕ := 16
def key_limes_needed : ℕ := 8

theorem key_lime_juice_yield : 
  let doubled_amount : ℚ := 2 * recipe_amount
  let total_tablespoons : ℚ := doubled_amount * tablespoons_per_cup
  let juice_per_lime : ℚ := total_tablespoons / key_limes_needed
  juice_per_lime = 1 := by sorry

end key_lime_juice_yield_l390_39073


namespace quadratic_roots_property_l390_39052

theorem quadratic_roots_property : ∀ x₁ x₂ : ℝ, 
  (x₁^2 + 2019*x₁ + 1 = 0) → 
  (x₂^2 + 2019*x₂ + 1 = 0) → 
  (x₁ ≠ x₂) →
  (x₁*x₂ - x₁ - x₂ = 2020) := by
sorry

end quadratic_roots_property_l390_39052


namespace expression_value_l390_39085

theorem expression_value (a b : ℝ) (h1 : a = Real.sqrt 3 - Real.sqrt 2) (h2 : b = Real.sqrt 3 + Real.sqrt 2) :
  a^2 + 3*a*b + b^2 - a + b = 13 + 2 * Real.sqrt 2 := by
  sorry

end expression_value_l390_39085


namespace total_tickets_sold_l390_39009

def student_tickets : ℕ := 90
def non_student_tickets : ℕ := 60

theorem total_tickets_sold : student_tickets + non_student_tickets = 150 := by
  sorry

end total_tickets_sold_l390_39009


namespace greatest_solution_sin_cos_equation_l390_39000

theorem greatest_solution_sin_cos_equation :
  ∃ (x : ℝ),
    x ∈ Set.Icc 0 (10 * Real.pi) ∧
    |2 * Real.sin x - 1| + |2 * Real.cos (2 * x) - 1| = 0 ∧
    (∀ (y : ℝ), y ∈ Set.Icc 0 (10 * Real.pi) →
      |2 * Real.sin y - 1| + |2 * Real.cos (2 * y) - 1| = 0 → y ≤ x) ∧
    x = 61 * Real.pi / 6 :=
by sorry

end greatest_solution_sin_cos_equation_l390_39000


namespace parabola_properties_incorrect_statement_l390_39038

-- Define the parabola
def parabola (x : ℝ) : ℝ := -(x - 1)^2 + 4

-- Statements to prove
theorem parabola_properties :
  -- The parabola opens downwards
  (∀ x : ℝ, parabola x ≤ parabola 1) ∧
  -- The shape is the same as y = x^2
  (∃ c : ℝ, ∀ x : ℝ, parabola x = c - x^2) ∧
  -- The vertex is (1,4)
  (parabola 1 = 4 ∧ ∀ x : ℝ, parabola x ≤ 4) ∧
  -- The axis of symmetry is the line x = 1
  (∀ x : ℝ, parabola (1 + x) = parabola (1 - x)) :=
by sorry

-- Statement C is incorrect
theorem incorrect_statement :
  ¬(parabola (-1) = 4 ∧ ∀ x : ℝ, parabola x ≤ 4) :=
by sorry

end parabola_properties_incorrect_statement_l390_39038


namespace range_of_m_l390_39097

-- Define the sets P and M
def P : Set ℝ := {x | x^2 ≤ 4}
def M (m : ℝ) : Set ℝ := {m}

-- State the theorem
theorem range_of_m (m : ℝ) : (P ∩ M m = M m) → m ∈ Set.Icc (-2) 2 := by
  sorry

end range_of_m_l390_39097


namespace first_bakery_sacks_proof_l390_39020

/-- The number of sacks the second bakery needs per week -/
def second_bakery_sacks : ℕ := 4

/-- The number of sacks the third bakery needs per week -/
def third_bakery_sacks : ℕ := 12

/-- The total number of weeks -/
def total_weeks : ℕ := 4

/-- The total number of sacks needed for all bakeries in 4 weeks -/
def total_sacks : ℕ := 72

/-- The number of sacks the first bakery needs per week -/
def first_bakery_sacks : ℕ := 2

theorem first_bakery_sacks_proof :
  first_bakery_sacks * total_weeks + 
  second_bakery_sacks * total_weeks + 
  third_bakery_sacks * total_weeks = total_sacks :=
by sorry

end first_bakery_sacks_proof_l390_39020


namespace f_has_real_roots_a_range_l390_39043

-- Define the quadratic function f
def f (a x : ℝ) : ℝ := x^2 + (2*a - 1)*x + 1 - 2*a

-- Theorem 1: For all a ∈ ℝ, f(x) = 1 has real roots
theorem f_has_real_roots (a : ℝ) : ∃ x : ℝ, f a x = 1 := by sorry

-- Theorem 2: If f has zero points in (-1,0) and (0,1/2), then 1/2 < a < 3/4
theorem a_range (a : ℝ) (h1 : f a (-1) > 0) (h2 : f a 0 < 0) (h3 : f a (1/2) > 0) :
  1/2 < a ∧ a < 3/4 := by sorry

end f_has_real_roots_a_range_l390_39043


namespace C_power_50_l390_39096

def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 4; -8, -10]

theorem C_power_50 : C^50 = !![201, 200; -400, -449] := by sorry

end C_power_50_l390_39096


namespace regression_line_properties_l390_39049

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear regression model -/
structure LinearRegression where
  b : ℝ  -- slope
  a : ℝ  -- intercept

/-- Represents a dataset -/
structure Dataset where
  points : List Point
  centroid : Point

/-- Checks if a point lies on the regression line -/
def pointOnLine (model : LinearRegression) (p : Point) : Prop :=
  p.y = model.b * p.x + model.a

/-- The main theorem stating that the regression line passes through the centroid
    but not necessarily through all data points -/
theorem regression_line_properties (data : Dataset) (model : LinearRegression) :
  (pointOnLine model data.centroid) ∧
  (∃ p ∈ data.points, ¬pointOnLine model p) := by
  sorry


end regression_line_properties_l390_39049


namespace logarithm_product_theorem_l390_39083

theorem logarithm_product_theorem (c d : ℕ+) : 
  (d - c = 840) →
  (Real.log d / Real.log c = 3) →
  (c + d : ℕ) = 1010 := by sorry

end logarithm_product_theorem_l390_39083


namespace ball_placement_theorem_l390_39066

/-- The number of ways to place n different balls into m different boxes --/
def placeWays (n m : ℕ) : ℕ := sorry

/-- The number of ways to place n different balls into m different boxes, leaving k boxes empty --/
def placeWaysWithEmpty (n m k : ℕ) : ℕ := sorry

theorem ball_placement_theorem :
  (placeWaysWithEmpty 4 4 1 = 144) ∧ (placeWaysWithEmpty 4 4 2 = 84) := by sorry

end ball_placement_theorem_l390_39066


namespace line_slope_l390_39031

theorem line_slope (x y : ℝ) : x + 2 * y - 6 = 0 → (y - 3) = (-1/2) * x := by
  sorry

end line_slope_l390_39031


namespace eggs_scrambled_l390_39055

-- Define the parameters
def total_time : ℕ := 39
def time_per_sausage : ℕ := 5
def num_sausages : ℕ := 3
def time_per_egg : ℕ := 4

-- Define the theorem
theorem eggs_scrambled :
  ∃ (num_eggs : ℕ),
    num_eggs * time_per_egg = total_time - (num_sausages * time_per_sausage) ∧
    num_eggs = 6 :=
by
  sorry

end eggs_scrambled_l390_39055


namespace sin_alpha_value_l390_39093

theorem sin_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = Real.sqrt 5 / 3 := by
  sorry

end sin_alpha_value_l390_39093


namespace cos_sum_thirteenth_l390_39027

theorem cos_sum_thirteenth : 
  Real.cos (3 * Real.pi / 13) + Real.cos (5 * Real.pi / 13) + Real.cos (7 * Real.pi / 13) = (Real.sqrt 13 - 1) / 4 := by
  sorry

end cos_sum_thirteenth_l390_39027


namespace line_transformation_l390_39065

/-- Given a line ax + y - 7 = 0 transformed by matrix A to 9x + y - 91 = 0, prove a = 2 and b = 13 -/
theorem line_transformation (a b : ℝ) :
  (∀ x y : ℝ, a * x + y - 7 = 0 →
    ∃ x' y' : ℝ, x' = 3 * x ∧ y' = -x + b * y ∧ 9 * x' + y' - 91 = 0) →
  a = 2 ∧ b = 13 := by
  sorry


end line_transformation_l390_39065


namespace bank_account_withdrawal_l390_39015

theorem bank_account_withdrawal (initial_balance : ℚ) : 
  (initial_balance > 0) →
  (initial_balance - 200 + (1/2) * (initial_balance - 200) = 450) →
  (200 / initial_balance = 2/5) := by
sorry

end bank_account_withdrawal_l390_39015


namespace product_sum_of_digits_l390_39037

def repeat_digit (d : Nat) (n : Nat) : Nat :=
  d * ((10^n - 1) / 9)

def sum_of_digits (n : Nat) : Nat :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

theorem product_sum_of_digits :
  sum_of_digits (repeat_digit 4 2012 * repeat_digit 9 2012) = 18108 := by
  sorry

end product_sum_of_digits_l390_39037


namespace power_product_simplification_l390_39060

theorem power_product_simplification (a : ℝ) : (36 * a^9)^4 * (63 * a^9)^4 = a^4 := by
  sorry

end power_product_simplification_l390_39060


namespace ratio_of_probabilities_l390_39017

/-- The number of rational terms in the expansion -/
def rational_terms : ℕ := 5

/-- The number of irrational terms in the expansion -/
def irrational_terms : ℕ := 4

/-- The total number of terms in the expansion -/
def total_terms : ℕ := rational_terms + irrational_terms

/-- The probability of having rational terms adjacent -/
def p : ℚ := (Nat.factorial rational_terms * Nat.factorial rational_terms) / Nat.factorial total_terms

/-- The probability of having no two rational terms adjacent -/
def q : ℚ := (Nat.factorial irrational_terms * Nat.factorial rational_terms) / Nat.factorial total_terms

theorem ratio_of_probabilities : p / q = 5 := by
  sorry

end ratio_of_probabilities_l390_39017


namespace set_intersection_example_l390_39082

theorem set_intersection_example :
  let M : Set ℕ := {2, 3, 4, 5}
  let N : Set ℕ := {3, 4, 5}
  M ∩ N = {3, 4, 5} := by
sorry

end set_intersection_example_l390_39082


namespace divisible_by_three_l390_39088

theorem divisible_by_three (a b : ℕ) (h : 3 ∣ (a * b)) : 3 ∣ a ∨ 3 ∣ b := by
  sorry

end divisible_by_three_l390_39088


namespace A_profit_share_l390_39056

-- Define the capital shares of partners A, B, C, and D
def share_A : ℚ := 1/3
def share_B : ℚ := 1/4
def share_C : ℚ := 1/5
def share_D : ℚ := 1 - (share_A + share_B + share_C)

-- Define the total profit
def total_profit : ℕ := 2445

-- Theorem statement
theorem A_profit_share :
  (share_A * total_profit : ℚ) = 815 := by
  sorry

end A_profit_share_l390_39056


namespace cash_count_correction_l390_39012

/-- Represents the correction needed for a cash count error -/
def correction_needed (q d n c x : ℕ) : ℤ :=
  let initial_count := 25 * q + 10 * d + 5 * n + c
  let corrected_count := 25 * (q - x) + 10 * (d - x) + 5 * (n + x) + (c + x)
  corrected_count - initial_count

/-- 
Theorem: Given a cash count with q quarters, d dimes, n nickels, c cents,
and x nickels mistakenly counted as quarters and x dimes as cents,
the correction needed is to add 11x cents.
-/
theorem cash_count_correction (q d n c x : ℕ) :
  correction_needed q d n c x = 11 * x := by
  sorry

end cash_count_correction_l390_39012


namespace sum_lower_bound_l390_39021

theorem sum_lower_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : 1/a + 4/b = 1) :
  ∀ c : ℝ, c < 9 → a + b > c :=
by
  sorry

end sum_lower_bound_l390_39021


namespace midpoint_trajectory_l390_39047

/-- The trajectory of the midpoint of chords passing through the origin of a circle --/
theorem midpoint_trajectory (x y : ℝ) :
  (0 < x) → (x ≤ 1) →
  (∃ (a b : ℝ), (a - 1)^2 + b^2 = 1 ∧ (x = a/2) ∧ (y = b/2)) →
  (x - 1/2)^2 + y^2 = 1 := by sorry

end midpoint_trajectory_l390_39047


namespace choose_3_from_10_l390_39095

theorem choose_3_from_10 : (Nat.choose 10 3) = 120 := by
  sorry

end choose_3_from_10_l390_39095


namespace distance_between_points_l390_39029

/-- The distance between points (3, 5) and (-4, 1) is √65 -/
theorem distance_between_points : Real.sqrt 65 = Real.sqrt ((3 - (-4))^2 + (5 - 1)^2) := by
  sorry

end distance_between_points_l390_39029


namespace no_prime_intercept_lines_through_point_l390_39071

-- Define a prime number
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define a line with intercepts
def Line (a b : ℕ) := {(x, y) : ℝ × ℝ | x / a + y / b = 1}

-- Theorem statement
theorem no_prime_intercept_lines_through_point :
  ¬∃ (a b : ℕ), isPrime a ∧ isPrime b ∧ (6, 5) ∈ Line a b := by
  sorry

end no_prime_intercept_lines_through_point_l390_39071


namespace inequality_solution_set_l390_39026

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x, |x + a| < b ↔ 2 < x ∧ x < 4) → a * b = -3 := by
  sorry

end inequality_solution_set_l390_39026


namespace smallest_exceeding_day_l390_39018

def tea_intake (n : ℕ) : ℚ := (n * (n + 1) * (n + 2)) / 3

theorem smallest_exceeding_day : 
  (∀ k < 13, tea_intake k ≤ 900) ∧ tea_intake 13 > 900 := by sorry

end smallest_exceeding_day_l390_39018


namespace alice_paid_percentage_l390_39075

def suggested_retail_price : ℝ → ℝ := id

def marked_price (P : ℝ) : ℝ := 0.6 * P

def alice_paid (P : ℝ) : ℝ := 0.4 * marked_price P

theorem alice_paid_percentage (P : ℝ) (h : P > 0) :
  alice_paid P / suggested_retail_price P = 0.24 := by
  sorry

end alice_paid_percentage_l390_39075


namespace scientific_notation_proof_l390_39086

theorem scientific_notation_proof : 
  (55000000 : ℝ) = 5.5 * (10 ^ 7) := by sorry

end scientific_notation_proof_l390_39086


namespace standard_deviation_decreases_after_correction_l390_39001

/-- Represents a class with test scores -/
structure TestScores where
  size : ℕ
  average : ℝ
  standardDev : ℝ

/-- Represents a score correction -/
structure ScoreCorrection where
  oldScore : ℝ
  newScore : ℝ

/-- The main theorem stating that the original standard deviation is greater than the new one after corrections -/
theorem standard_deviation_decreases_after_correction 
  (original : TestScores)
  (correction1 correction2 : ScoreCorrection)
  (new_std_dev : ℝ)
  (h_size : original.size = 50)
  (h_avg : original.average = 70)
  (h_correction1 : correction1.oldScore = 50 ∧ correction1.newScore = 80)
  (h_correction2 : correction2.oldScore = 100 ∧ correction2.newScore = 70)
  : original.standardDev > new_std_dev := by
  sorry

end standard_deviation_decreases_after_correction_l390_39001


namespace sequence_problem_l390_39025

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

/-- The main theorem -/
theorem sequence_problem (a b : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a)
    (h_geom : geometric_sequence b)
    (h_eq : 2 * a 3 - (a 8)^2 + 2 * a 13 = 0)
    (h_b8 : b 8 = a 8) :
  b 4 * b 12 = 16 := by
  sorry

end sequence_problem_l390_39025


namespace difference_calculation_l390_39023

theorem difference_calculation (total : ℝ) (h : total = 6000) : 
  (1 / 10 * total) - (1 / 1000 * total) = 594 := by
  sorry

end difference_calculation_l390_39023


namespace solution_of_fraction_equation_l390_39042

theorem solution_of_fraction_equation :
  ∃ x : ℝ, (3 - x) / (4 + 2*x) = 0 ∧ x = 3 :=
by sorry

end solution_of_fraction_equation_l390_39042


namespace donated_books_count_l390_39033

/-- Represents the number of books in the library over time --/
structure LibraryBooks where
  initial_old : ℕ
  bought_two_years_ago : ℕ
  bought_last_year : ℕ
  current_total : ℕ

/-- Calculates the number of old books donated --/
def books_donated (lib : LibraryBooks) : ℕ :=
  lib.initial_old + lib.bought_two_years_ago + lib.bought_last_year - lib.current_total

/-- Theorem stating the number of old books donated --/
theorem donated_books_count (lib : LibraryBooks) 
  (h1 : lib.initial_old = 500)
  (h2 : lib.bought_two_years_ago = 300)
  (h3 : lib.bought_last_year = lib.bought_two_years_ago + 100)
  (h4 : lib.current_total = 1000) :
  books_donated lib = 200 := by
  sorry

#eval books_donated ⟨500, 300, 400, 1000⟩

end donated_books_count_l390_39033


namespace expression_simplification_l390_39013

theorem expression_simplification (a b : ℝ) (h1 : a = 2) (h2 : b = -1) :
  2 * (-a^2 + 2*a*b) - 3 * (a*b - a^2) = 2 := by
  sorry

end expression_simplification_l390_39013


namespace applicant_overall_score_l390_39094

/-- Calculates the overall score given written test and interview scores and their weights -/
def overall_score (written_score interview_score : ℝ) (written_weight interview_weight : ℝ) : ℝ :=
  written_score * written_weight + interview_score * interview_weight

/-- Theorem stating that the overall score is 72 points given the specific scores and weights -/
theorem applicant_overall_score :
  let written_score : ℝ := 80
  let interview_score : ℝ := 60
  let written_weight : ℝ := 0.6
  let interview_weight : ℝ := 0.4
  overall_score written_score interview_score written_weight interview_weight = 72 := by
  sorry

#eval overall_score 80 60 0.6 0.4

end applicant_overall_score_l390_39094


namespace sqrt_625_equals_5_to_m_l390_39076

theorem sqrt_625_equals_5_to_m (m : ℝ) : (625 : ℝ)^(1/2) = 5^m → m = 2 := by
  sorry

end sqrt_625_equals_5_to_m_l390_39076


namespace alice_bob_meet_l390_39070

/-- The number of points on the circular path -/
def n : ℕ := 18

/-- Alice's clockwise movement per turn -/
def alice_move : ℕ := 7

/-- Bob's counterclockwise movement per turn -/
def bob_move : ℕ := 11

/-- The number of turns after which Alice and Bob meet -/
def meeting_turns : ℕ := 9

/-- Function to calculate the position after a certain number of moves -/
def position_after_moves (start : ℕ) (move : ℕ) (turns : ℕ) : ℕ :=
  (start + move * turns - 1) % n + 1

theorem alice_bob_meet :
  position_after_moves n alice_move meeting_turns =
  position_after_moves n (n - bob_move) meeting_turns :=
sorry

end alice_bob_meet_l390_39070


namespace some_T_divisible_by_3_l390_39077

def T : Set ℤ := {x | ∃ n : ℤ, x = (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2}

theorem some_T_divisible_by_3 : ∃ x ∈ T, 3 ∣ x := by
  sorry

end some_T_divisible_by_3_l390_39077


namespace simplified_expression_constant_expression_l390_39053

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 2 * x^2 + 4 * x * y - 2 * x - 3
def B (x y : ℝ) : ℝ := -x^2 + x * y + 2

-- Theorem 1: Prove the simplified expression for 3A - 2(A + 2B)
theorem simplified_expression (x y : ℝ) :
  3 * A x y - 2 * (A x y + 2 * B x y) = 6 * x^2 - 2 * x - 11 := by sorry

-- Theorem 2: Prove the value of y when B + (1/2)A is constant for any x
theorem constant_expression (y : ℝ) :
  (∀ x : ℝ, ∃ c : ℝ, B x y + (1/2) * A x y = c) ↔ y = 1/3 := by sorry

end simplified_expression_constant_expression_l390_39053


namespace fourth_power_of_square_of_fourth_smallest_prime_l390_39039

-- Define a function to get the nth smallest prime number
def nthSmallestPrime (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem fourth_power_of_square_of_fourth_smallest_prime :
  (nthSmallestPrime 4)^2^4 = 5764801 := by sorry

end fourth_power_of_square_of_fourth_smallest_prime_l390_39039


namespace paved_road_time_l390_39062

/-- Calculates the time spent on a paved road given total trip distance,
    dirt road travel time and speed, and speed difference between paved and dirt roads. -/
theorem paved_road_time (total_distance : ℝ) (dirt_time : ℝ) (dirt_speed : ℝ) (speed_diff : ℝ) :
  total_distance = 200 →
  dirt_time = 3 →
  dirt_speed = 32 →
  speed_diff = 20 →
  (total_distance - dirt_time * dirt_speed) / (dirt_speed + speed_diff) = 2 := by
  sorry

#check paved_road_time

end paved_road_time_l390_39062


namespace complement_of_M_in_U_l390_39014

universe u

def U : Finset ℕ := {4,5,6,8,9}
def M : Finset ℕ := {5,6,8}

theorem complement_of_M_in_U :
  (U \ M) = {4,9} := by sorry

end complement_of_M_in_U_l390_39014


namespace opposite_direction_speed_l390_39035

/-- Given two people moving in opposite directions, this theorem proves
    the speed of the second person given the conditions of the problem. -/
theorem opposite_direction_speed
  (time : ℝ)
  (distance : ℝ)
  (speed1 : ℝ)
  (h1 : time = 4)
  (h2 : distance = 28)
  (h3 : speed1 = 4)
  (h4 : distance = time * (speed1 + speed2)) :
  speed2 = 3 := by
  sorry


end opposite_direction_speed_l390_39035
