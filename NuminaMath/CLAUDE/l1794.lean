import Mathlib

namespace NUMINAMATH_CALUDE_floor_plus_x_eq_seventeen_fourths_l1794_179426

theorem floor_plus_x_eq_seventeen_fourths :
  ∃! (x : ℚ), ⌊x⌋ + x = 17/4 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_floor_plus_x_eq_seventeen_fourths_l1794_179426


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l1794_179464

-- Define the structure for an isosceles triangle
structure IsoscelesTriangle where
  side : ℕ  -- Equal sides
  base : ℕ  -- Base

-- Define the theorem
theorem min_perimeter_isosceles_triangles 
  (t1 t2 : IsoscelesTriangle) 
  (h1 : t1 ≠ t2)  -- Noncongruent triangles
  (h2 : 2 * t1.side + t1.base = 2 * t2.side + t2.base)  -- Same perimeter
  (h3 : t1.side * t1.base = t2.side * t2.base)  -- Same area (simplified)
  (h4 : 9 * t1.base = 8 * t2.base)  -- Ratio of bases
  : 2 * t1.side + t1.base ≥ 868 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l1794_179464


namespace NUMINAMATH_CALUDE_video_game_lives_l1794_179448

theorem video_game_lives (initial_lives gained_lives final_lives : ℕ) 
  (h1 : initial_lives = 47)
  (h2 : gained_lives = 46)
  (h3 : final_lives = 70) :
  initial_lives - (final_lives - gained_lives) = 23 := by
  sorry

end NUMINAMATH_CALUDE_video_game_lives_l1794_179448


namespace NUMINAMATH_CALUDE_sphere_surface_area_doubling_l1794_179461

theorem sphere_surface_area_doubling (r : ℝ) :
  (4 * Real.pi * r^2 = 2464) →
  (4 * Real.pi * (2*r)^2 = 39376) :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_doubling_l1794_179461


namespace NUMINAMATH_CALUDE_sum_of_seventh_powers_squared_l1794_179438

theorem sum_of_seventh_powers_squared (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (sum_zero : a + b + c = 0) :
  (a^7 + b^7 + c^7)^2 / ((a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) * (a^4 + b^4 + c^4) * (a^5 + b^5 + c^5)) = 49/60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seventh_powers_squared_l1794_179438


namespace NUMINAMATH_CALUDE_select_five_from_eight_l1794_179436

/-- The number of combinations of n items taken r at a time -/
def combination (n r : ℕ) : ℕ := sorry

/-- Theorem stating that selecting 5 items from 8 items results in 56 combinations -/
theorem select_five_from_eight : combination 8 5 = 56 := by sorry

end NUMINAMATH_CALUDE_select_five_from_eight_l1794_179436


namespace NUMINAMATH_CALUDE_x_asymptotics_l1794_179488

/-- The Lambert W function -/
noncomputable def W : ℝ → ℝ := sorry

/-- Asymptotic equivalence -/
def asymptotic_equiv (f g : ℕ → ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ) (N : ℕ), c₁ > 0 ∧ c₂ > 0 ∧ ∀ n ≥ N, c₁ * g n ≤ f n ∧ f n ≤ c₂ * g n

theorem x_asymptotics (n : ℕ) (x : ℝ) (h : x^x = n) :
  asymptotic_equiv (λ n => x) (λ n => Real.log n / Real.log (Real.log n)) :=
sorry

end NUMINAMATH_CALUDE_x_asymptotics_l1794_179488


namespace NUMINAMATH_CALUDE_power_inequality_l1794_179473

def S : Set ℤ := {-2, -1, 0, 1, 2, 3}

theorem power_inequality (n : ℤ) :
  n ∈ S → ((-1/2 : ℚ)^n > (-1/5 : ℚ)^n ↔ n = -1 ∨ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_power_inequality_l1794_179473


namespace NUMINAMATH_CALUDE_aleesia_weight_loss_l1794_179401

/-- Aleesia's weekly weight loss problem -/
theorem aleesia_weight_loss 
  (aleesia_weeks : ℕ) 
  (alexei_weeks : ℕ) 
  (alexei_weekly_loss : ℝ) 
  (total_loss : ℝ) 
  (h1 : aleesia_weeks = 10) 
  (h2 : alexei_weeks = 8) 
  (h3 : alexei_weekly_loss = 2.5) 
  (h4 : total_loss = 35) :
  ∃ (aleesia_weekly_loss : ℝ), 
    aleesia_weekly_loss * aleesia_weeks + alexei_weekly_loss * alexei_weeks = total_loss ∧ 
    aleesia_weekly_loss = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_aleesia_weight_loss_l1794_179401


namespace NUMINAMATH_CALUDE_thousand_factorization_sum_l1794_179419

/-- Checks if a positive integer contains zero in its decimal representation -/
def containsZero (n : Nat) : Bool :=
  n.repr.contains '0'

/-- Theorem stating the existence of two positive integers satisfying the given conditions -/
theorem thousand_factorization_sum :
  ∃ (a b : Nat), a * b = 1000 ∧ ¬containsZero a ∧ ¬containsZero b ∧ a + b = 133 := by
  sorry

end NUMINAMATH_CALUDE_thousand_factorization_sum_l1794_179419


namespace NUMINAMATH_CALUDE_sqrt_x_minus_two_defined_l1794_179472

theorem sqrt_x_minus_two_defined (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_two_defined_l1794_179472


namespace NUMINAMATH_CALUDE_multiply_72519_by_9999_l1794_179416

theorem multiply_72519_by_9999 : 72519 * 9999 = 725117481 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72519_by_9999_l1794_179416


namespace NUMINAMATH_CALUDE_exponent_product_equals_twentyfive_l1794_179441

theorem exponent_product_equals_twentyfive :
  (5 ^ 0.4) * (5 ^ 0.6) * (5 ^ 0.2) * (5 ^ 0.3) * (5 ^ 0.5) = 25 := by
  sorry

end NUMINAMATH_CALUDE_exponent_product_equals_twentyfive_l1794_179441


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l1794_179404

theorem smaller_number_in_ratio (a b c x y : ℝ) : 
  0 < a → a < b → x > 0 → y > 0 → x / y = (a + 1) / (b + 1) → x + y = 2 * c →
  min x y = (2 * c * (a + 1)) / (a + b + 2) := by
sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l1794_179404


namespace NUMINAMATH_CALUDE_range_of_a_l1794_179474

theorem range_of_a (p q : Prop) (hp : ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0)
  (hq : ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) (hpq : p ∧ q) :
  a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1794_179474


namespace NUMINAMATH_CALUDE_clothes_expenditure_fraction_l1794_179458

def salary : ℝ := 190000

theorem clothes_expenditure_fraction 
  (food_fraction : ℝ) 
  (rent_fraction : ℝ) 
  (remaining : ℝ) 
  (h1 : food_fraction = 1/5)
  (h2 : rent_fraction = 1/10)
  (h3 : remaining = 19000)
  (h4 : ∃ (clothes_fraction : ℝ), 
    salary * (1 - food_fraction - rent_fraction - clothes_fraction) = remaining) :
  ∃ (clothes_fraction : ℝ), clothes_fraction = 3/5 := by
sorry

end NUMINAMATH_CALUDE_clothes_expenditure_fraction_l1794_179458


namespace NUMINAMATH_CALUDE_problem_1_l1794_179494

theorem problem_1 : 3^2 * (-1 + 3) - (-16) / 8 = 20 := by sorry

end NUMINAMATH_CALUDE_problem_1_l1794_179494


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1794_179467

/-- Two points are symmetric about the origin if their coordinates have opposite signs -/
def symmetric_about_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

/-- If points M(3,a-2) and N(b,a) are symmetric about the origin, then a + b = -2 -/
theorem symmetric_points_sum (a b : ℝ) :
  symmetric_about_origin 3 (a - 2) b a → a + b = -2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1794_179467


namespace NUMINAMATH_CALUDE_floor_painting_theorem_l1794_179493

/-- The number of integer pairs (a, b) satisfying the floor painting conditions -/
def floor_painting_solutions : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let a := p.1
    let b := p.2
    b > a ∧ b % 3 = 0 ∧ (a - 6) * (b - 6) = 36 ∧ a > 6 ∧ b > 6)
    (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- Theorem stating that there are exactly 2 solutions to the floor painting problem -/
theorem floor_painting_theorem : floor_painting_solutions = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_painting_theorem_l1794_179493


namespace NUMINAMATH_CALUDE_circle_area_equality_l1794_179460

theorem circle_area_equality (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 15) (h₂ : r₂ = 25) (h₃ : r₃ = 20) :
  π * r₃^2 = π * r₂^2 - π * r₁^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_equality_l1794_179460


namespace NUMINAMATH_CALUDE_kaili_circle_method_l1794_179487

theorem kaili_circle_method (S : ℝ) (V : ℝ) (h : S = 4 * Real.pi / 9) :
  (2/3)^3 = 16 * V / 9 :=
sorry

end NUMINAMATH_CALUDE_kaili_circle_method_l1794_179487


namespace NUMINAMATH_CALUDE_total_envelopes_l1794_179434

def blue_envelopes : ℕ := 120
def yellow_envelopes : ℕ := blue_envelopes - 25
def green_envelopes : ℕ := 5 * yellow_envelopes

theorem total_envelopes : blue_envelopes + yellow_envelopes + green_envelopes = 690 := by
  sorry

end NUMINAMATH_CALUDE_total_envelopes_l1794_179434


namespace NUMINAMATH_CALUDE_inverse_expression_equals_one_thirteenth_l1794_179466

theorem inverse_expression_equals_one_thirteenth :
  (3 - 5 * (3 - 4)⁻¹ * 2)⁻¹ = (1 : ℚ) / 13 := by
  sorry

end NUMINAMATH_CALUDE_inverse_expression_equals_one_thirteenth_l1794_179466


namespace NUMINAMATH_CALUDE_child_growth_proof_l1794_179478

/-- Calculates the growth in height given current and previous heights -/
def heightGrowth (currentHeight previousHeight : Float) : Float :=
  currentHeight - previousHeight

theorem child_growth_proof :
  let currentHeight : Float := 41.5
  let previousHeight : Float := 38.5
  heightGrowth currentHeight previousHeight = 3 := by
  sorry

end NUMINAMATH_CALUDE_child_growth_proof_l1794_179478


namespace NUMINAMATH_CALUDE_distinct_roots_sum_bound_l1794_179407

theorem distinct_roots_sum_bound (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ → 
  r₁^2 + p*r₁ + 9 = 0 → 
  r₂^2 + p*r₂ + 9 = 0 → 
  |r₁ + r₂| > 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_distinct_roots_sum_bound_l1794_179407


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l1794_179411

theorem quadratic_roots_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    x₁^2 + p*x₁ + q = 0 ∧
    x₂^2 + p*x₂ + q = 0 ∧
    |x₁ - x₂| = 2) →
  p = 2 * Real.sqrt (q + 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l1794_179411


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l1794_179403

theorem geometric_sequence_second_term (b : ℝ) (h1 : b > 0) :
  (∃ r : ℝ, r > 0 ∧ b = 15 * r ∧ 45/4 = b * r) → b = 15 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l1794_179403


namespace NUMINAMATH_CALUDE_probability_even_sum_two_wheels_l1794_179406

theorem probability_even_sum_two_wheels (wheel1_total : ℕ) (wheel1_even : ℕ) (wheel2_total : ℕ) (wheel2_even : ℕ) :
  wheel1_total = 2 * wheel1_even ∧ 
  wheel2_total = 5 ∧ 
  wheel2_even = 2 →
  (wheel1_even : ℚ) / wheel1_total * (wheel2_even : ℚ) / wheel2_total + 
  (wheel1_even : ℚ) / wheel1_total * ((wheel2_total - wheel2_even) : ℚ) / wheel2_total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_sum_two_wheels_l1794_179406


namespace NUMINAMATH_CALUDE_pauls_crayons_given_to_friends_l1794_179446

/-- Given information about Paul's crayons --/
structure CrayonInfo where
  initial : ℕ  -- Initial number of crayons
  lost_difference : ℕ  -- Difference between lost and given crayons
  total_gone : ℕ  -- Total number of crayons no longer in possession

/-- Calculate the number of crayons given to friends --/
def crayons_given_to_friends (info : CrayonInfo) : ℕ :=
  (info.total_gone - info.lost_difference) / 2

/-- Theorem stating the number of crayons Paul gave to his friends --/
theorem pauls_crayons_given_to_friends :
  let info : CrayonInfo := {
    initial := 110,
    lost_difference := 322,
    total_gone := 412
  }
  crayons_given_to_friends info = 45 := by
  sorry

end NUMINAMATH_CALUDE_pauls_crayons_given_to_friends_l1794_179446


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1794_179439

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

theorem symmetric_points_sum (a b : ℝ) :
  symmetric_wrt_origin (-1, a) (b, 2) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1794_179439


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l1794_179497

/-- Proves that the weight of a replaced person is 66 kg given the conditions of the problem -/
theorem weight_of_replaced_person
  (n : ℕ) -- number of people in the initial group
  (avg_increase : ℝ) -- increase in average weight
  (new_weight : ℝ) -- weight of the new person
  (h1 : n = 8) -- there are 8 persons initially
  (h2 : avg_increase = 2.5) -- the average weight increases by 2.5 kg
  (h3 : new_weight = 86) -- the weight of the new person is 86 kg
  : ∃ (replaced_weight : ℝ), replaced_weight = 66 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l1794_179497


namespace NUMINAMATH_CALUDE_binomial_coefficient_8_4_l1794_179465

theorem binomial_coefficient_8_4 : Nat.choose 8 4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_8_4_l1794_179465


namespace NUMINAMATH_CALUDE_sean_purchase_cost_l1794_179486

/-- The cost of items in Sean's purchase -/
def CostCalculation (soda_price : ℝ) : Prop :=
  let soup_price := 3 * soda_price
  let sandwich_price := 3 * soup_price
  (3 * soda_price) + (2 * soup_price) + sandwich_price = 18

/-- Theorem stating the total cost of Sean's purchase -/
theorem sean_purchase_cost :
  CostCalculation 1 := by
  sorry

end NUMINAMATH_CALUDE_sean_purchase_cost_l1794_179486


namespace NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l1794_179427

theorem sin_sum_of_complex_exponentials (θ φ : ℝ) :
  Complex.exp (θ * I) = (1 / 5 : ℂ) + (2 * Real.sqrt 6 / 5 : ℂ) * I ∧
  Complex.exp (φ * I) = (-5 / 13 : ℂ) - (12 / 13 : ℂ) * I →
  Real.sin (θ + φ) = -(12 - 10 * Real.sqrt 6) / 65 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l1794_179427


namespace NUMINAMATH_CALUDE_students_liking_new_menu_l1794_179430

theorem students_liking_new_menu (total_students : ℕ) (disliking_students : ℕ) 
  (h1 : total_students = 400) 
  (h2 : disliking_students = 165) : 
  total_students - disliking_students = 235 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_new_menu_l1794_179430


namespace NUMINAMATH_CALUDE_boat_current_rate_l1794_179435

/-- Proves that given a boat with a speed of 42 km/hr in still water,
    traveling 35.2 km downstream in 44 minutes, the rate of the current is 6 km/hr. -/
theorem boat_current_rate 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 42)
  (h2 : distance = 35.2)
  (h3 : time = 44 / 60) : 
  ∃ (current_rate : ℝ), 
    current_rate = 6 ∧ 
    distance = (boat_speed + current_rate) * time :=
by sorry

end NUMINAMATH_CALUDE_boat_current_rate_l1794_179435


namespace NUMINAMATH_CALUDE_meal_combinations_l1794_179420

/-- Number of meat options -/
def meatOptions : ℕ := 4

/-- Number of vegetable options -/
def vegOptions : ℕ := 5

/-- Number of dessert options -/
def dessertOptions : ℕ := 5

/-- Number of meat choices -/
def meatChoices : ℕ := 2

/-- Number of vegetable choices -/
def vegChoices : ℕ := 3

/-- Number of dessert choices -/
def dessertChoices : ℕ := 1

/-- The total number of meal combinations -/
theorem meal_combinations : 
  (meatOptions.choose meatChoices) * (vegOptions.choose vegChoices) * dessertOptions = 300 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_l1794_179420


namespace NUMINAMATH_CALUDE_newspaper_subscription_cost_l1794_179498

theorem newspaper_subscription_cost (discount_rate : ℝ) (discounted_price : ℝ) (normal_price : ℝ) : 
  discount_rate = 0.45 →
  discounted_price = 44 →
  normal_price * (1 - discount_rate) = discounted_price →
  normal_price = 80 := by
sorry

end NUMINAMATH_CALUDE_newspaper_subscription_cost_l1794_179498


namespace NUMINAMATH_CALUDE_zara_age_l1794_179402

def guesses : List Nat := [26, 29, 31, 34, 37, 39, 42, 45, 47, 50, 52]

def is_prime (n : Nat) : Prop := Nat.Prime n

def more_than_half_low (age : Nat) : Prop :=
  (guesses.filter (· < age)).length > guesses.length / 2

def three_off_by_one (age : Nat) : Prop :=
  (guesses.filter (fun x => x = age - 1 ∨ x = age + 1)).length = 3

theorem zara_age : ∃! age : Nat, 
  age ∈ guesses ∧
  is_prime age ∧
  more_than_half_low age ∧
  three_off_by_one age ∧
  age = 47 :=
sorry

end NUMINAMATH_CALUDE_zara_age_l1794_179402


namespace NUMINAMATH_CALUDE_product_of_roots_equals_one_l1794_179414

theorem product_of_roots_equals_one :
  let A := Real.sqrt 2019 + Real.sqrt 2020 + 1
  let B := -Real.sqrt 2019 - Real.sqrt 2020 - 1
  let C := Real.sqrt 2019 - Real.sqrt 2020 + 1
  let D := Real.sqrt 2020 - Real.sqrt 2019 - 1
  A * B * C * D = 1 := by sorry

end NUMINAMATH_CALUDE_product_of_roots_equals_one_l1794_179414


namespace NUMINAMATH_CALUDE_row_swap_property_l1794_179437

def row_swap_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 1, 0]

theorem row_swap_property (A : Matrix (Fin 2) (Fin 2) ℝ) :
  row_swap_matrix * A = Matrix.of (λ i j => A (1 - i) j) := by
  sorry

end NUMINAMATH_CALUDE_row_swap_property_l1794_179437


namespace NUMINAMATH_CALUDE_cubic_inequality_and_fraction_inequality_l1794_179425

theorem cubic_inequality_and_fraction_inequality 
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  (x^3 + y^3 ≥ x^2*y + x*y^2) ∧ 
  ((x/(y*z) + y/(z*x) + z/(x*y)) ≥ (1/x + 1/y + 1/z)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_and_fraction_inequality_l1794_179425


namespace NUMINAMATH_CALUDE_simplify_expression_l1794_179450

theorem simplify_expression (a b : ℝ) : 6*a - 8*b - 2*(3*a + b) = -10*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1794_179450


namespace NUMINAMATH_CALUDE_museum_ticket_cost_l1794_179462

def entrance_ticket_cost (num_students : ℕ) (num_teachers : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / (num_students + num_teachers)

theorem museum_ticket_cost :
  entrance_ticket_cost 20 3 115 = 5 := by
sorry

end NUMINAMATH_CALUDE_museum_ticket_cost_l1794_179462


namespace NUMINAMATH_CALUDE_largest_multiple_of_11_less_than_neg_150_l1794_179415

theorem largest_multiple_of_11_less_than_neg_150 :
  ∀ n : ℤ, n * 11 < -150 → n * 11 ≤ -154 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_11_less_than_neg_150_l1794_179415


namespace NUMINAMATH_CALUDE_division_theorem_l1794_179499

theorem division_theorem (A B : ℕ) : 23 = 6 * A + B ∧ B < 6 → A = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_theorem_l1794_179499


namespace NUMINAMATH_CALUDE_circle_fits_in_triangle_l1794_179455

theorem circle_fits_in_triangle (a b c : ℝ) (S : ℝ) : 
  a = 3 ∧ b = 4 ∧ c = 5 → S = 25 / 8 →
  ∃ (r R : ℝ), r = (a + b - c) / 2 ∧ S = π * R^2 ∧ R < r := by
  sorry

end NUMINAMATH_CALUDE_circle_fits_in_triangle_l1794_179455


namespace NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l1794_179418

theorem no_solution_quadratic_inequality :
  ¬∃ (x : ℝ), 3 * x^2 + 9 * x ≤ -12 := by sorry

end NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l1794_179418


namespace NUMINAMATH_CALUDE_complement_of_union_l1794_179468

def U : Set Nat := {1,2,3,4,5}
def M : Set Nat := {1,3}
def N : Set Nat := {1,2}

theorem complement_of_union (U M N : Set Nat) : 
  U \ (M ∪ N) = {4,5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l1794_179468


namespace NUMINAMATH_CALUDE_triangle_division_theorem_l1794_179484

theorem triangle_division_theorem (A B C : ℝ) :
  A + B + C = 180 →
  B = 120 →
  (∃ D : ℝ, (A = D ∧ B / 2 = D) ∨ (C = D ∧ B / 2 = D) ∨ (A = D ∧ C = D)) →
  ((A = 40 ∧ C = 20) ∨ (A = 45 ∧ C = 15) ∨ (A = 20 ∧ C = 40) ∨ (A = 15 ∧ C = 45)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_division_theorem_l1794_179484


namespace NUMINAMATH_CALUDE_political_science_majors_l1794_179408

/-- The number of applicants who majored in political science -/
def P : ℕ := 15

theorem political_science_majors :
  (40 : ℕ) = P + 15 + 10 ∧ 
  (20 : ℕ) = 5 + 15 ∧
  (10 : ℕ) = 40 - (P + 20) :=
by sorry

end NUMINAMATH_CALUDE_political_science_majors_l1794_179408


namespace NUMINAMATH_CALUDE_mia_average_first_four_days_l1794_179457

theorem mia_average_first_four_days 
  (total_distance : ℝ) 
  (race_days : ℕ) 
  (jesse_avg_first_three : ℝ) 
  (jesse_day_four : ℝ) 
  (combined_avg_last_three : ℝ) 
  (h1 : total_distance = 30)
  (h2 : race_days = 7)
  (h3 : jesse_avg_first_three = 2/3)
  (h4 : jesse_day_four = 10)
  (h5 : combined_avg_last_three = 6) :
  ∃ mia_avg_first_four : ℝ,
    mia_avg_first_four = 3 ∧
    mia_avg_first_four * 4 + combined_avg_last_three * 3 = total_distance ∧
    jesse_avg_first_three * 3 + jesse_day_four + combined_avg_last_three * 3 = total_distance :=
by sorry

end NUMINAMATH_CALUDE_mia_average_first_four_days_l1794_179457


namespace NUMINAMATH_CALUDE_zero_point_condition_l1794_179423

theorem zero_point_condition (a : ℝ) : 
  (∃ x ∈ Set.Ioo (-1 : ℝ) 1, 3 * a * x + 1 - 2 * a = 0) ↔ 
  (a < -1 ∨ a > 1/5) := by sorry

end NUMINAMATH_CALUDE_zero_point_condition_l1794_179423


namespace NUMINAMATH_CALUDE_triangle_3_7_triangle_3_neg4_triangle_neg4_3_triangle_not_commutative_l1794_179459

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ := -2 * a * b - b + 1

-- Theorem statements
theorem triangle_3_7 : triangle 3 7 = -48 := by sorry

theorem triangle_3_neg4 : triangle 3 (-4) = 29 := by sorry

theorem triangle_neg4_3 : triangle (-4) 3 = 22 := by sorry

theorem triangle_not_commutative : ∃ a b : ℚ, triangle a b ≠ triangle b a := by sorry

end NUMINAMATH_CALUDE_triangle_3_7_triangle_3_neg4_triangle_neg4_3_triangle_not_commutative_l1794_179459


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1794_179432

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (a₁ : ℝ), r ≠ 0 ∧ ∀ n : ℕ, a n = a₁ * r^(n-1)

theorem geometric_sequence_sixth_term 
  (a : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a) 
  (h_sum1 : a 2 + a 4 = 20) 
  (h_sum2 : a 3 + a 5 = 40) : 
  a 6 = 64 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1794_179432


namespace NUMINAMATH_CALUDE_ratio_x_to_y_is_eight_l1794_179476

theorem ratio_x_to_y_is_eight (x y : ℝ) (h : y = 0.125 * x) : x / y = 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_is_eight_l1794_179476


namespace NUMINAMATH_CALUDE_complex_number_problem_l1794_179412

theorem complex_number_problem (α β : ℂ) :
  (α - β).re > 0 →
  (2 * Complex.I * (α + β)).re > 0 →
  β = 4 + Complex.I →
  α = -4 + Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1794_179412


namespace NUMINAMATH_CALUDE_binomial_9_choose_3_l1794_179495

theorem binomial_9_choose_3 : Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_binomial_9_choose_3_l1794_179495


namespace NUMINAMATH_CALUDE_complex_modulus_l1794_179475

theorem complex_modulus (z : ℂ) : z = -1 + Complex.I * Real.sqrt 3 → Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1794_179475


namespace NUMINAMATH_CALUDE_triangle_area_l1794_179469

-- Define the point P
def P : ℝ × ℝ := (2, 5)

-- Define the slopes of the two lines
def slope1 : ℝ := -1
def slope2 : ℝ := 1.5

-- Define Q and R as the x-intercepts of the lines
def Q : ℝ × ℝ := (-3, 0)
def R : ℝ × ℝ := (5.33, 0)

-- Theorem statement
theorem triangle_area : 
  let triangle_area := (1/2) * (R.1 - Q.1) * P.2
  triangle_area = 20.825 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1794_179469


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l1794_179456

theorem distance_from_origin_to_point :
  let x : ℝ := 8
  let y : ℝ := 15
  Real.sqrt (x^2 + y^2) = 17 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l1794_179456


namespace NUMINAMATH_CALUDE_power_function_k_values_l1794_179489

/-- A function is a power function if it has the form f(x) = ax^n where a is a non-zero constant and n is a real number. -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^n

/-- The main theorem: if y=(k^2-k-5)x^3 is a power function, then k = 3 or k = -2 -/
theorem power_function_k_values (k : ℝ) :
  is_power_function (λ x => (k^2 - k - 5) * x^3) → k = 3 ∨ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_k_values_l1794_179489


namespace NUMINAMATH_CALUDE_fraction_product_equality_l1794_179463

theorem fraction_product_equality : 
  (3 / 4) * (36 / 60) * (10 / 4) * (14 / 28) * (9 / 3)^2 * (45 / 15) * (12 / 18) * (20 / 40)^3 = 27 / 32 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equality_l1794_179463


namespace NUMINAMATH_CALUDE_A_equals_one_two_l1794_179483

def A : Set ℤ := {x | 0 < x ∧ x ≤ 2}

theorem A_equals_one_two : A = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_A_equals_one_two_l1794_179483


namespace NUMINAMATH_CALUDE_problem_solution_l1794_179453

theorem problem_solution (x : ℝ) (h1 : Real.sqrt ((3 * x) / 7) = x) (h2 : x ≠ 0) : x = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1794_179453


namespace NUMINAMATH_CALUDE_custom_mult_identity_l1794_179482

/-- Custom multiplication operation -/
noncomputable def customMult (a b c : ℝ) (x y : ℝ) : ℝ := a * x + b * y + c * x * y

theorem custom_mult_identity {a b c : ℝ} (h1 : customMult a b c 1 2 = 4) (h2 : customMult a b c 2 3 = 6) :
  ∃ m : ℝ, m ≠ 0 ∧ (∀ x : ℝ, customMult a b c x m = x) → m = 5 := by
  sorry


end NUMINAMATH_CALUDE_custom_mult_identity_l1794_179482


namespace NUMINAMATH_CALUDE_platform_length_platform_length_proof_l1794_179480

/-- Calculates the length of a platform given train parameters -/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let total_distance := train_speed_mps * crossing_time
  total_distance - train_length

/-- Proves that the platform length is 50 meters given the specific conditions -/
theorem platform_length_proof : 
  platform_length 250 72 15 = 50 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_platform_length_proof_l1794_179480


namespace NUMINAMATH_CALUDE_same_root_implies_a_equals_three_l1794_179431

theorem same_root_implies_a_equals_three (a : ℝ) : 
  (∃ x : ℝ, 3 * x - 2 * a = 0 ∧ 2 * x + 3 * a - 13 = 0) → a = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_same_root_implies_a_equals_three_l1794_179431


namespace NUMINAMATH_CALUDE_monster_count_theorem_l1794_179452

/-- Calculates the total number of monsters after 5 days given the initial count and daily growth factors -/
def total_monsters (initial : ℕ) (factor2 factor3 factor4 factor5 : ℕ) : ℕ :=
  initial + 
  initial * factor2 + 
  initial * factor2 * factor3 + 
  initial * factor2 * factor3 * factor4 + 
  initial * factor2 * factor3 * factor4 * factor5

/-- Theorem stating that given the specific initial count and growth factors, the total number of monsters after 5 days is 872 -/
theorem monster_count_theorem : total_monsters 2 3 4 5 6 = 872 := by
  sorry

end NUMINAMATH_CALUDE_monster_count_theorem_l1794_179452


namespace NUMINAMATH_CALUDE_president_and_vp_from_seven_l1794_179410

/-- The number of ways to choose a President and a Vice-President from a group of n people,
    where the two positions must be held by different people. -/
def choose_president_and_vp (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: There are 42 ways to choose a President and a Vice-President from a group of 7 people,
    where the two positions must be held by different people. -/
theorem president_and_vp_from_seven : choose_president_and_vp 7 = 42 := by
  sorry

end NUMINAMATH_CALUDE_president_and_vp_from_seven_l1794_179410


namespace NUMINAMATH_CALUDE_point_coordinates_l1794_179491

/-- The coordinates of a point A(a,b) satisfying given conditions -/
theorem point_coordinates :
  ∀ (a b : ℝ),
    (|b| = 3) →  -- Distance from A to x-axis is 3
    (|a| = 4) →  -- Distance from A to y-axis is 4
    (a > b) →    -- Given condition a > b
    ((a = 4 ∧ b = -3) ∨ (a = 4 ∧ b = 3)) := by
  sorry


end NUMINAMATH_CALUDE_point_coordinates_l1794_179491


namespace NUMINAMATH_CALUDE_bracket_equation_solution_l1794_179445

theorem bracket_equation_solution (x : ℝ) : 
  45 - (28 - (37 - (15 - x))) = 59 → x = 20 := by
sorry

end NUMINAMATH_CALUDE_bracket_equation_solution_l1794_179445


namespace NUMINAMATH_CALUDE_second_pipe_filling_time_l1794_179428

/-- Given a pool that can be filled by one pipe in 10 hours and by both pipes in 3.75 hours,
    prove that the second pipe alone takes 6 hours to fill the pool. -/
theorem second_pipe_filling_time
  (time_pipe1 : ℝ) (time_both : ℝ) (time_pipe2 : ℝ)
  (h1 : time_pipe1 = 10)
  (h2 : time_both = 3.75)
  (h3 : 1 / time_pipe1 + 1 / time_pipe2 = 1 / time_both) :
  time_pipe2 = 6 :=
sorry

end NUMINAMATH_CALUDE_second_pipe_filling_time_l1794_179428


namespace NUMINAMATH_CALUDE_triangle_inequality_l1794_179409

/-- For any triangle with side lengths a, b, and c, 
    3(b+c-a)(c+a-b)(a+b-c) ≤ a²(b+c-a) + b²(c+a-b) + c²(a+b-c) holds. -/
theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  3 * (b + c - a) * (c + a - b) * (a + b - c) ≤ 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1794_179409


namespace NUMINAMATH_CALUDE_equation_solution_l1794_179496

theorem equation_solution :
  ∃! x : ℚ, (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ∧ x = -7/6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1794_179496


namespace NUMINAMATH_CALUDE_river_depth_l1794_179481

/-- The depth of a river given its width, flow rate, and volume of water per minute. -/
theorem river_depth (width : ℝ) (flow_rate : ℝ) (volume_per_minute : ℝ) :
  width = 45 →
  flow_rate = 5 →
  volume_per_minute = 7500 →
  (volume_per_minute / (width * (flow_rate * 1000 / 60))) = 2 := by
  sorry


end NUMINAMATH_CALUDE_river_depth_l1794_179481


namespace NUMINAMATH_CALUDE_chocolate_distribution_l1794_179400

theorem chocolate_distribution (x y : ℕ) (h1 : y = x + 1) (h2 : ∃ z : ℕ, y = (x - 1) * z + z) : 
  ∃ z : ℕ, y = (x - 1) * z + z ∧ z = 2 := by
sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l1794_179400


namespace NUMINAMATH_CALUDE_sum_difference_is_4750_l1794_179442

/-- Rounds a number to the nearest multiple of 5, rounding 2.5s up -/
def roundToNearestFive (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

/-- Sums all integers from 1 to n -/
def sumToN (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Sums all integers from 1 to n after rounding each to the nearest multiple of 5 -/
def sumRoundedToN (n : ℕ) : ℕ :=
  (n / 5) * (2 * 0 + 3 * 5)

theorem sum_difference_is_4750 :
  sumToN 100 - sumRoundedToN 100 = 4750 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_is_4750_l1794_179442


namespace NUMINAMATH_CALUDE_school_bus_time_theorem_l1794_179405

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Converts 24-hour format to 12-hour format -/
def to12HourFormat (t : Time) : Time :=
  if t.hours ≤ 12 then t else { hours := t.hours - 12, minutes := t.minutes }

/-- Calculates the time difference in minutes between two Time values -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t2.hours * 60 + t2.minutes) - (t1.hours * 60 + t1.minutes)

theorem school_bus_time_theorem :
  let schoolEndTime : Time := { hours := 16, minutes := 45 }
  let arrivalTime : Time := { hours := 17, minutes := 20 }
  (to12HourFormat schoolEndTime = { hours := 4, minutes := 45 }) ∧
  (timeDifference schoolEndTime arrivalTime = 35) :=
by sorry

end NUMINAMATH_CALUDE_school_bus_time_theorem_l1794_179405


namespace NUMINAMATH_CALUDE_gcf_of_30_90_75_l1794_179429

theorem gcf_of_30_90_75 : Nat.gcd 30 (Nat.gcd 90 75) = 15 := by sorry

end NUMINAMATH_CALUDE_gcf_of_30_90_75_l1794_179429


namespace NUMINAMATH_CALUDE_salem_women_count_l1794_179471

/-- Proves the number of women in Salem after population change -/
theorem salem_women_count (leesburg_population : ℕ) (salem_multiplier : ℕ) (people_moving_out : ℕ) :
  leesburg_population = 58940 →
  salem_multiplier = 15 →
  people_moving_out = 130000 →
  (salem_multiplier * leesburg_population - people_moving_out) / 2 = 377050 := by
sorry

end NUMINAMATH_CALUDE_salem_women_count_l1794_179471


namespace NUMINAMATH_CALUDE_barts_earnings_l1794_179490

/-- Represents the earnings for a single day --/
structure DayEarnings where
  rate : Rat
  questionsPerSurvey : Nat
  surveysCompleted : Nat

/-- Calculates the total earnings for a given day --/
def calculateDayEarnings (day : DayEarnings) : Rat :=
  day.rate * day.questionsPerSurvey * day.surveysCompleted

/-- Calculates the total earnings for three days --/
def calculateTotalEarnings (day1 day2 day3 : DayEarnings) : Rat :=
  calculateDayEarnings day1 + calculateDayEarnings day2 + calculateDayEarnings day3

/-- Theorem statement for Bart's earnings over three days --/
theorem barts_earnings :
  let monday : DayEarnings := { rate := 1/5, questionsPerSurvey := 10, surveysCompleted := 3 }
  let tuesday : DayEarnings := { rate := 1/4, questionsPerSurvey := 12, surveysCompleted := 4 }
  let wednesday : DayEarnings := { rate := 1/10, questionsPerSurvey := 15, surveysCompleted := 5 }
  calculateTotalEarnings monday tuesday wednesday = 51/2 := by
  sorry

end NUMINAMATH_CALUDE_barts_earnings_l1794_179490


namespace NUMINAMATH_CALUDE_trajectory_equation_of_P_l1794_179485

/-- The trajectory equation of point P on the xOy plane, given its distance from A(0,0,4) -/
theorem trajectory_equation_of_P (P : ℝ × ℝ) (d : ℝ → ℝ → ℝ → ℝ → ℝ) :
  (∀ z, d P.1 P.2 0 z = d P.1 P.2 0 0) →  -- P is on the xOy plane
  d P.1 P.2 0 4 = 5 →                     -- distance between P and A is 5
  P.1^2 + P.2^2 = 9 :=                    -- trajectory equation
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_of_P_l1794_179485


namespace NUMINAMATH_CALUDE_sufficient_conditions_for_inequality_l1794_179422

theorem sufficient_conditions_for_inequality (f : ℝ → ℝ) :
  (((∀ x y : ℝ, x < y → f x > f y) ∧ (∀ x : ℝ, f x > 0)) →
    (∃ a : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f (x + a) < f x + f a)) ∧
  ((∀ x y : ℝ, x < y → f x < f y) ∧ (∃ x₀ : ℝ, x₀ < 0 ∧ f x₀ = 0) →
    (∃ a : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f (x + a) < f x + f a)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_conditions_for_inequality_l1794_179422


namespace NUMINAMATH_CALUDE_democrat_ratio_l1794_179443

/-- Represents the number of participants in a meeting with democrats -/
structure Meeting where
  total : ℕ
  female : ℕ
  male : ℕ
  femaleDemocrats : ℕ
  maleDemocrats : ℕ

/-- The properties of the meeting as described in the problem -/
def meetingProperties (m : Meeting) : Prop :=
  m.total = 750 ∧
  m.female + m.male = m.total ∧
  m.femaleDemocrats = m.female / 2 ∧
  m.maleDemocrats = m.male / 4 ∧
  m.femaleDemocrats = 125

/-- The theorem stating that the ratio of democrats to total participants is 1:3 -/
theorem democrat_ratio (m : Meeting) (h : meetingProperties m) :
  (m.femaleDemocrats + m.maleDemocrats) * 3 = m.total := by
  sorry


end NUMINAMATH_CALUDE_democrat_ratio_l1794_179443


namespace NUMINAMATH_CALUDE_prob_two_odd_dice_l1794_179413

/-- The number of faces on a standard die -/
def num_faces : ℕ := 6

/-- The number of odd faces on a standard die -/
def num_odd_faces : ℕ := 3

/-- The total number of possible outcomes when throwing two dice -/
def total_outcomes : ℕ := num_faces * num_faces

/-- The number of outcomes where both dice show odd numbers -/
def favorable_outcomes : ℕ := num_odd_faces * num_odd_faces

/-- The probability of rolling two odd numbers when throwing two dice simultaneously -/
theorem prob_two_odd_dice : 
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_odd_dice_l1794_179413


namespace NUMINAMATH_CALUDE_smallest_multiple_of_2_3_4_5_7_l1794_179454

theorem smallest_multiple_of_2_3_4_5_7 : ∃ n : ℕ, 
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(2 ∣ m ∧ 3 ∣ m ∧ 4 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m)) ∧ 
  (2 ∣ n ∧ 3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) :=
by
  use 420
  sorry

#eval 420 % 2
#eval 420 % 3
#eval 420 % 4
#eval 420 % 5
#eval 420 % 7

end NUMINAMATH_CALUDE_smallest_multiple_of_2_3_4_5_7_l1794_179454


namespace NUMINAMATH_CALUDE_no_real_roots_x_squared_plus_four_l1794_179451

theorem no_real_roots_x_squared_plus_four :
  ¬ ∃ (x : ℝ), x^2 + 4 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_x_squared_plus_four_l1794_179451


namespace NUMINAMATH_CALUDE_return_journey_time_l1794_179449

/-- Proves that given a round trip with specified conditions, the return journey takes 7 hours -/
theorem return_journey_time 
  (total_distance : ℝ) 
  (outbound_time : ℝ) 
  (average_speed : ℝ) 
  (h1 : total_distance = 2000) 
  (h2 : outbound_time = 10) 
  (h3 : average_speed = 142.85714285714286) : 
  (total_distance / average_speed) - outbound_time = 7 := by
  sorry

end NUMINAMATH_CALUDE_return_journey_time_l1794_179449


namespace NUMINAMATH_CALUDE_inequality_proof_l1794_179444

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1/8) :
  a^2 + b^2 + c^2 + a^2*b^2 + b^2*c^2 + c^2*a^2 ≥ 15/16 ∧
  (a^2 + b^2 + c^2 + a^2*b^2 + b^2*c^2 + c^2*a^2 = 15/16 ↔ a = 1/2 ∧ b = 1/2 ∧ c = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1794_179444


namespace NUMINAMATH_CALUDE_simplify_expression_l1794_179424

theorem simplify_expression (n : ℕ) : 
  (3^(n+5) - 3*(3^n)) / (3*(3^(n+4))) = 80 / 27 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l1794_179424


namespace NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l1794_179447

-- Define the parabola equation
def parabola (y : ℝ) : ℝ := -3 * y^2 + 4 * y + 2

-- State the theorem
theorem parabola_has_one_x_intercept :
  ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l1794_179447


namespace NUMINAMATH_CALUDE_quadratic_equation_b_value_l1794_179470

theorem quadratic_equation_b_value 
  (b : ℝ) 
  (h1 : 2 * (5 : ℝ)^2 + b * 5 - 65 = 0) : 
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_b_value_l1794_179470


namespace NUMINAMATH_CALUDE_geometric_sequence_special_case_l1794_179477

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_special_case (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 3)^2 - 4*(a 3) + 3 = 0 →
  (a 7)^2 - 4*(a 7) + 3 = 0 →
  a 5 = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_case_l1794_179477


namespace NUMINAMATH_CALUDE_smallest_positive_angle_solution_l1794_179433

/-- The equation that needs to be satisfied -/
def equation (y : ℝ) : Prop :=
  6 * Real.sin y * (Real.cos y)^3 - 6 * (Real.sin y)^3 * Real.cos y = 1

/-- The smallest positive angle in degrees that satisfies the equation -/
def smallest_angle : ℝ := 10.4525

theorem smallest_positive_angle_solution :
  equation (smallest_angle * π / 180) ∧
  ∀ y, 0 < y ∧ y < smallest_angle * π / 180 → ¬equation y :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_solution_l1794_179433


namespace NUMINAMATH_CALUDE_lcm_132_315_l1794_179440

theorem lcm_132_315 : Nat.lcm 132 315 = 13860 := by sorry

end NUMINAMATH_CALUDE_lcm_132_315_l1794_179440


namespace NUMINAMATH_CALUDE_parallelogram_grid_non_congruent_triangles_l1794_179492

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the parallelogram grid array -/
def ParallelogramGrid : List Point := [
  ⟨0, 0⟩,   -- Point 1
  ⟨1, 0⟩,   -- Point 2
  ⟨1.5, 0.5⟩, -- Point 3
  ⟨2.5, 0.5⟩, -- Point 4
  ⟨0.5, 0.25⟩, -- Point 5 (midpoint)
  ⟨1.75, 0.25⟩, -- Point 6 (midpoint)
  ⟨1.75, 0⟩, -- Point 7 (midpoint)
  ⟨1.25, 0.25⟩  -- Point 8 (center)
]

/-- Determines if two triangles are congruent -/
def areTrianglesCongruent (t1 t2 : List Point) : Bool :=
  sorry -- Implementation details omitted

/-- Counts the number of non-congruent triangles in the grid -/
def countNonCongruentTriangles (grid : List Point) : Nat :=
  sorry -- Implementation details omitted

/-- Theorem: The number of non-congruent triangles in the parallelogram grid is 9 -/
theorem parallelogram_grid_non_congruent_triangles :
  countNonCongruentTriangles ParallelogramGrid = 9 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_grid_non_congruent_triangles_l1794_179492


namespace NUMINAMATH_CALUDE_stone_splitting_properties_l1794_179479

/-- Represents the state of stone piles -/
structure PileState :=
  (piles : List Nat)
  (valid : piles.sum = 100)

/-- Represents a single move in the stone-splitting process -/
def split_move (s : PileState) : PileState → Prop :=
  sorry

/-- Represents the complete process of splitting stones -/
def splitting_process (initial : PileState) (final : PileState) : Prop :=
  sorry

theorem stone_splitting_properties 
  (initial : PileState)
  (final : PileState)
  (h_initial : initial.piles = [100])
  (h_final : final.piles.all (· = 1) ∧ final.piles.length = 100)
  (h_process : splitting_process initial final) :
  (∃ s : PileState, splitting_process initial s ∧ 
    (∃ sub : List Nat, sub ⊆ s.piles ∧ sub.length = 30 ∧ sub.sum = 60)) ∧
  (∃ s : PileState, splitting_process initial s ∧ 
    (∃ sub : List Nat, sub ⊆ s.piles ∧ sub.length = 20 ∧ sub.sum = 60)) ∧
  (∃ f : PileState → PileState, 
    splitting_process initial (f final) ∧
    ∀ s, splitting_process initial s → splitting_process s (f final) →
      ¬∃ sub : List Nat, sub ⊆ s.piles ∧ sub.length = 19 ∧ sub.sum = 60) :=
sorry

end NUMINAMATH_CALUDE_stone_splitting_properties_l1794_179479


namespace NUMINAMATH_CALUDE_xy_value_l1794_179417

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 + y^2 = 58) : x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1794_179417


namespace NUMINAMATH_CALUDE_perpendicular_line_to_plane_is_perpendicular_to_all_lines_in_plane_l1794_179421

-- Define a plane
structure Plane :=
  (α : Type*)

-- Define a line
structure Line :=
  (l : Type*)

-- Define perpendicular relation between a line and a plane
def perpendicular_to_plane (l : Line) (α : Plane) : Prop :=
  sorry

-- Define a line being contained within a plane
def contained_in_plane (m : Line) (α : Plane) : Prop :=
  sorry

-- Define perpendicular relation between two lines
def perpendicular_lines (l m : Line) : Prop :=
  sorry

-- Theorem statement
theorem perpendicular_line_to_plane_is_perpendicular_to_all_lines_in_plane
  (l m : Line) (α : Plane)
  (h1 : perpendicular_to_plane l α)
  (h2 : contained_in_plane m α) :
  perpendicular_lines l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_plane_is_perpendicular_to_all_lines_in_plane_l1794_179421
