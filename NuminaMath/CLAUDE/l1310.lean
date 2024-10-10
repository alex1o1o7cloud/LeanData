import Mathlib

namespace train_ride_nap_time_l1310_131023

theorem train_ride_nap_time (total_time reading_time eating_time movie_time : ℕ) 
  (h1 : total_time = 9)
  (h2 : reading_time = 2)
  (h3 : eating_time = 1)
  (h4 : movie_time = 3) :
  total_time - (reading_time + eating_time + movie_time) = 3 :=
by sorry

end train_ride_nap_time_l1310_131023


namespace box_stacking_comparison_l1310_131083

/-- Represents the height of a stack of boxes -/
def stack_height (box_height : ℝ) (num_floors : ℕ) : ℝ :=
  box_height * (num_floors : ℝ)

/-- The problem statement -/
theorem box_stacking_comparison : 
  let box_a_height : ℝ := 3
  let box_b_height : ℝ := 3.5
  let taehyung_floors : ℕ := 16
  let yoongi_floors : ℕ := 14
  
  stack_height box_b_height yoongi_floors - stack_height box_a_height taehyung_floors = 1 := by
  sorry

end box_stacking_comparison_l1310_131083


namespace pat_stickers_l1310_131028

theorem pat_stickers (initial_stickers given_away_stickers : ℝ) 
  (h1 : initial_stickers = 39.0)
  (h2 : given_away_stickers = 22.0) :
  initial_stickers - given_away_stickers = 17.0 := by
  sorry

end pat_stickers_l1310_131028


namespace min_value_product_l1310_131065

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a/b + b/c + c/a + b/a + c/b + a/c = 10)
  (h2 : a^2 + b^2 + c^2 = 9) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) ≥ 91/2 :=
by sorry

end min_value_product_l1310_131065


namespace opposite_def_opposite_of_two_l1310_131072

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_def (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of 2 is -2 -/
theorem opposite_of_two : opposite 2 = -2 := by sorry

end opposite_def_opposite_of_two_l1310_131072


namespace quadratic_monotone_decreasing_m_range_l1310_131093

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 1

-- State the theorem
theorem quadratic_monotone_decreasing_m_range :
  ∀ m : ℝ, (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 1 → f m x₁ > f m x₂) →
  m ∈ Set.Ici 1 := by
  sorry

end quadratic_monotone_decreasing_m_range_l1310_131093


namespace expected_value_is_four_thirds_l1310_131021

/-- The expected value of a biased coin flip --/
def expected_value_biased_coin : ℚ :=
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let value_heads : ℤ := 5
  let value_tails : ℤ := -6
  p_heads * value_heads + p_tails * value_tails

/-- Theorem: The expected value of the biased coin flip is 4/3 --/
theorem expected_value_is_four_thirds :
  expected_value_biased_coin = 4/3 := by
  sorry

end expected_value_is_four_thirds_l1310_131021


namespace second_division_divisor_l1310_131046

theorem second_division_divisor (x y : ℕ) (h1 : x > 0) (h2 : x % 10 = 3) (h3 : x / 10 = y)
  (h4 : ∃ k : ℕ, (2 * x) % k = 1 ∧ (2 * x) / k = 3 * y) (h5 : 11 * y - x = 2) :
  ∃ k : ℕ, (2 * x) % k = 1 ∧ (2 * x) / k = 3 * y ∧ k = 7 :=
by sorry

end second_division_divisor_l1310_131046


namespace ceiling_product_equation_solution_l1310_131094

theorem ceiling_product_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ (⌈x⌉ : ℝ) * x = 210 ∧ x = 14 := by sorry

end ceiling_product_equation_solution_l1310_131094


namespace max_value_expression_l1310_131049

theorem max_value_expression :
  ∃ (x y : ℝ),
    ∀ (a b : ℝ),
      (Real.sqrt (9 - Real.sqrt 7) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 1) *
      (3 + 2 * Real.sqrt (13 - Real.sqrt 7) * Real.cos y - Real.cos (2 * y)) ≤
      (Real.sqrt (9 - Real.sqrt 7) * Real.sin a - Real.sqrt (2 * (1 + Real.cos (2 * a))) - 1) *
      (3 + 2 * Real.sqrt (13 - Real.sqrt 7) * Real.cos b - Real.cos (2 * b)) →
      (Real.sqrt (9 - Real.sqrt 7) * Real.sin a - Real.sqrt (2 * (1 + Real.cos (2 * a))) - 1) *
      (3 + 2 * Real.sqrt (13 - Real.sqrt 7) * Real.cos b - Real.cos (2 * b)) = 24 - 2 * Real.sqrt 7 :=
by sorry

end max_value_expression_l1310_131049


namespace complex_sum_parts_l1310_131088

theorem complex_sum_parts (z : ℂ) (h : z / (1 + 2*I) = 2 + I) : 
  (z + 5).re + (z + 5).im = 0 := by
  sorry

end complex_sum_parts_l1310_131088


namespace brick_surface_area_l1310_131067

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 10 cm x 4 cm x 3 cm brick is 164 cm² -/
theorem brick_surface_area :
  surface_area 10 4 3 = 164 := by
  sorry

end brick_surface_area_l1310_131067


namespace prob_less_than_5_eq_half_l1310_131081

/-- A fair 8-sided die -/
def fair_8_sided_die : Finset ℕ := Finset.range 8

/-- The probability of an event occurring when rolling a fair 8-sided die -/
def prob (event : Finset ℕ) : ℚ := (event.card : ℚ) / (fair_8_sided_die.card : ℚ)

/-- The event of rolling a number less than 5 -/
def less_than_5 : Finset ℕ := Finset.filter (λ x => x < 5) fair_8_sided_die

theorem prob_less_than_5_eq_half : 
  prob less_than_5 = 1/2 := by sorry

end prob_less_than_5_eq_half_l1310_131081


namespace gross_revenue_increase_l1310_131032

theorem gross_revenue_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (price_reduction_percent : ℝ) 
  (quantity_increase_percent : ℝ) 
  (h1 : price_reduction_percent = 20) 
  (h2 : quantity_increase_percent = 50) :
  let new_price := original_price * (1 - price_reduction_percent / 100)
  let new_quantity := original_quantity * (1 + quantity_increase_percent / 100)
  let original_gross := original_price * original_quantity
  let new_gross := new_price * new_quantity
  (new_gross - original_gross) / original_gross * 100 = 20 := by
sorry

end gross_revenue_increase_l1310_131032


namespace range_of_a_l1310_131029

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 5, a * x^2 - x - 4 > 0) → a > 5 := by
  sorry

end range_of_a_l1310_131029


namespace compound_carbon_count_l1310_131041

/-- Represents the number of atoms of a given element in a compound -/
structure AtomCount where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Represents the atomic weights of elements -/
structure AtomicWeights where
  carbon : ℝ
  hydrogen : ℝ
  oxygen : ℝ

/-- Calculates the molecular weight of a compound given its atom count and atomic weights -/
def molecularWeight (count : AtomCount) (weights : AtomicWeights) : ℝ :=
  count.carbon * weights.carbon +
  count.hydrogen * weights.hydrogen +
  count.oxygen * weights.oxygen

/-- The theorem to be proved -/
theorem compound_carbon_count (weights : AtomicWeights)
    (h_carbon : weights.carbon = 12.01)
    (h_hydrogen : weights.hydrogen = 1.01)
    (h_oxygen : weights.oxygen = 16.00) :
    ∃ (count : AtomCount),
      count.hydrogen = 8 ∧
      count.oxygen = 2 ∧
      molecularWeight count weights = 88 ∧
      count.carbon = 4 := by
  sorry

end compound_carbon_count_l1310_131041


namespace volume_of_midpoint_set_l1310_131062

/-- A regular tetrahedron -/
structure RegularTetrahedron :=
  (edge_length : ℝ)

/-- The set of midpoints of segments whose endpoints belong to different tetrahedra -/
def midpoint_set (t1 t2 : RegularTetrahedron) : Set (Fin 3 → ℝ) :=
  sorry

/-- The volume of a set in ℝ³ -/
noncomputable def volume (s : Set (Fin 3 → ℝ)) : ℝ :=
  sorry

/-- Central symmetry transformation -/
def central_symmetry (t : RegularTetrahedron) : RegularTetrahedron :=
  sorry

theorem volume_of_midpoint_set :
  ∀ t : RegularTetrahedron,
  t.edge_length = Real.sqrt 2 →
  volume (midpoint_set t (central_symmetry t)) = 5/6 :=
by sorry

end volume_of_midpoint_set_l1310_131062


namespace quadratic_equation_root_zero_l1310_131008

theorem quadratic_equation_root_zero (a : ℝ) :
  (∀ x, x^2 + x + a^2 - 1 = 0 → x = 0 ∨ x ≠ 0) →
  (∃ x, x^2 + x + a^2 - 1 = 0 ∧ x = 0) →
  a = 1 ∨ a = -1 := by
sorry

end quadratic_equation_root_zero_l1310_131008


namespace pre_bought_tickets_l1310_131077

/-- The number of people who pre-bought plane tickets -/
def num_pre_buyers : ℕ := 20

/-- The price of a pre-bought ticket -/
def pre_bought_price : ℕ := 155

/-- The price of a ticket bought at the gate -/
def gate_price : ℕ := 200

/-- The number of people who bought tickets at the gate -/
def num_gate_buyers : ℕ := 30

/-- The difference in total amount paid between gate buyers and pre-buyers -/
def price_difference : ℕ := 2900

theorem pre_bought_tickets : 
  num_pre_buyers * pre_bought_price + price_difference = num_gate_buyers * gate_price := by
  sorry

end pre_bought_tickets_l1310_131077


namespace f_value_at_2_l1310_131006

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_value_at_2 (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8)
  (h2 : f (-2) = 10) : 
  f 2 = -26 := by
  sorry

end f_value_at_2_l1310_131006


namespace tangency_triangle_area_for_given_radii_l1310_131005

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents three mutually externally tangent circles -/
structure TangentCircles where
  c1 : Circle
  c2 : Circle
  c3 : Circle

/-- The area of the triangle formed by the points of tangency of three mutually externally tangent circles -/
def tangencyTriangleArea (tc : TangentCircles) : ℝ := sorry

/-- Theorem stating that for the given radii, the area of the tangency triangle is 120/25 -/
theorem tangency_triangle_area_for_given_radii :
  ∀ tc : TangentCircles,
    tc.c1.radius = 5 ∧ tc.c2.radius = 12 ∧ tc.c3.radius = 13 →
    tangencyTriangleArea tc = 120 / 25 := by
  sorry

end tangency_triangle_area_for_given_radii_l1310_131005


namespace smallest_solution_quadratic_l1310_131080

theorem smallest_solution_quadratic :
  ∃ (x : ℝ), x = 2/3 ∧ 6*x^2 - 19*x + 10 = 0 ∧ ∀ (y : ℝ), 6*y^2 - 19*y + 10 = 0 → x ≤ y :=
by sorry

end smallest_solution_quadratic_l1310_131080


namespace solution_in_interval_l1310_131011

def f (x : ℝ) := x^2 + 12*x - 15

theorem solution_in_interval :
  ∃ x : ℝ, x ∈ (Set.Ioo 1.1 1.2) ∧ f x = 0 :=
by
  have h1 : f 1.1 < 0 := by sorry
  have h2 : f 1.2 > 0 := by sorry
  sorry

end solution_in_interval_l1310_131011


namespace even_iff_mod_two_eq_zero_l1310_131047

theorem even_iff_mod_two_eq_zero (x : Int) : Even x ↔ x % 2 = 0 := by sorry

end even_iff_mod_two_eq_zero_l1310_131047


namespace vampire_population_after_two_nights_l1310_131042

def vampire_growth (initial_vampires : ℕ) (new_vampires_per_night : ℕ) (nights : ℕ) : ℕ :=
  initial_vampires * (new_vampires_per_night + 1)^nights

theorem vampire_population_after_two_nights :
  vampire_growth 3 7 2 = 192 :=
by sorry

end vampire_population_after_two_nights_l1310_131042


namespace quadratic_inequality_l1310_131091

theorem quadratic_inequality (x : ℝ) : x^2 - 9*x + 14 < 0 ↔ 2 < x ∧ x < 7 := by
  sorry

end quadratic_inequality_l1310_131091


namespace f_81_product_remainder_l1310_131012

def p : ℕ := 2^16 + 1

-- S is implicitly defined as the set of positive integers not divisible by p

def is_in_S (x : ℕ) : Prop := x > 0 ∧ ¬(p ∣ x)

axiom p_is_prime : Nat.Prime p

axiom f_exists : ∃ (f : ℕ → ℕ), 
  (∀ x, is_in_S x → f x < p) ∧
  (∀ x y, is_in_S x → is_in_S y → (f x * f y) % p = (f (x * y) + f (x * y^(p-2))) % p) ∧
  (∀ x, is_in_S x → f (x + p) = f x)

def N : ℕ := sorry  -- Definition of N as the product of nonzero f(81) values

theorem f_81_product_remainder : N % p = 16384 := by sorry

end f_81_product_remainder_l1310_131012


namespace ratio_B_to_C_l1310_131037

def total_amount : ℕ := 578
def share_A : ℕ := 408
def share_B : ℕ := 102
def share_C : ℕ := 68

theorem ratio_B_to_C :
  (share_B : ℚ) / share_C = 3 / 2 := by sorry

end ratio_B_to_C_l1310_131037


namespace book_distribution_theorem_l1310_131024

/-- The number of ways to distribute books to people -/
def distribute_books (total_people : ℕ) (math_books : ℕ) (chinese_books : ℕ) : ℕ :=
  Nat.choose total_people chinese_books

/-- Theorem stating that the number of ways to distribute 6 math books and 3 Chinese books
    to 9 people is equal to C(9,3) -/
theorem book_distribution_theorem :
  distribute_books 9 6 3 = Nat.choose 9 3 := by
  sorry

end book_distribution_theorem_l1310_131024


namespace divisible_by_101_exists_l1310_131071

theorem divisible_by_101_exists (n : ℕ) (h : n ≥ 10^2018) : 
  ∃ k : ℕ, ∃ m : ℕ, m ≥ n ∧ m = n + k ∧ m % 101 = 0 :=
by sorry

end divisible_by_101_exists_l1310_131071


namespace alex_martin_games_l1310_131076

/-- The number of players in the four-square league --/
def total_players : ℕ := 12

/-- The number of players in each game --/
def players_per_game : ℕ := 6

/-- The number of players to be chosen after Alex and Martin are included --/
def players_to_choose : ℕ := players_per_game - 2

/-- The number of remaining players after Alex and Martin are excluded --/
def remaining_players : ℕ := total_players - 2

/-- The number of times Alex plays in the same game as Martin --/
def games_together : ℕ := Nat.choose remaining_players players_to_choose

theorem alex_martin_games :
  games_together = 210 :=
sorry

end alex_martin_games_l1310_131076


namespace train_crossing_time_l1310_131058

/-- The time taken for a train to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length > 0 → train_speed_kmh > 0 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 
  (train_length / (train_speed_kmh * (5 / 18))) :=
by sorry

end train_crossing_time_l1310_131058


namespace trees_died_in_typhoon_l1310_131066

theorem trees_died_in_typhoon (initial_trees left_trees : ℕ) : 
  initial_trees = 20 → left_trees = 4 → initial_trees - left_trees = 16 := by
  sorry

end trees_died_in_typhoon_l1310_131066


namespace september_first_is_wednesday_l1310_131040

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- The number of lessons for each day of the week -/
def lessonsPerDay (d : DayOfWeek) : Nat :=
  match d with
  | .Monday => 1
  | .Tuesday => 2
  | .Wednesday => 3
  | .Thursday => 4
  | .Friday => 5
  | .Saturday => 0
  | .Sunday => 0

/-- The total number of lessons in a week -/
def lessonsPerWeek : Nat :=
  (lessonsPerDay .Monday) +
  (lessonsPerDay .Tuesday) +
  (lessonsPerDay .Wednesday) +
  (lessonsPerDay .Thursday) +
  (lessonsPerDay .Friday) +
  (lessonsPerDay .Saturday) +
  (lessonsPerDay .Sunday)

/-- The function to determine the day of the week for September 1 -/
def septemberFirstDay (totalLessons : Nat) : DayOfWeek :=
  sorry

/-- The theorem stating that September 1 falls on a Wednesday -/
theorem september_first_is_wednesday :
  septemberFirstDay 64 = DayOfWeek.Wednesday :=
sorry

end september_first_is_wednesday_l1310_131040


namespace average_of_quadratic_solutions_l1310_131057

/-- Given a quadratic equation ax² - 4ax + b = 0 with two real solutions,
    prove that the average of these solutions is 2. -/
theorem average_of_quadratic_solutions (a b : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 - 4 * a * x + b
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) → (x₁ + x₂) / 2 = 2 := by
  sorry

end average_of_quadratic_solutions_l1310_131057


namespace set_operations_l1310_131009

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 2}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 4}

-- State the theorem
theorem set_operations :
  (A ∩ B = {x | 2 ≤ x ∧ x ≤ 4}) ∧
  (A ∪ B = {x | x > 1}) := by
  sorry

end set_operations_l1310_131009


namespace smallest_four_digit_divisible_by_40_l1310_131059

theorem smallest_four_digit_divisible_by_40 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 40 = 0 → n ≥ 1000 :=
by
  sorry

end smallest_four_digit_divisible_by_40_l1310_131059


namespace rational_function_positivity_l1310_131001

theorem rational_function_positivity (x : ℝ) :
  (x^2 - 9) / (x^2 - 16) > 0 ↔ x < -4 ∨ x > 4 := by
  sorry

end rational_function_positivity_l1310_131001


namespace quadratic_function_properties_l1310_131017

def f (x : ℝ) := x^2 - 4*x + 3
def g (x : ℝ) := -3*x + 3

theorem quadratic_function_properties :
  (∃ (x : ℝ), g x = 0 ∧ f x = 0) ∧
  (g 0 = f 0) ∧
  (∀ (x : ℝ), f x ≥ -1) ∧
  (∃ (x : ℝ), f x = -1) := by
sorry

end quadratic_function_properties_l1310_131017


namespace not_all_same_color_probability_l1310_131064

def num_people : ℕ := 3
def num_colors : ℕ := 5

theorem not_all_same_color_probability :
  (num_colors^num_people - num_colors) / num_colors^num_people = 24 / 25 := by
  sorry

end not_all_same_color_probability_l1310_131064


namespace five_balls_three_boxes_l1310_131010

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to distribute 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 243 := by
  sorry

end five_balls_three_boxes_l1310_131010


namespace polynomial_intercept_nonzero_coeff_l1310_131086

theorem polynomial_intercept_nonzero_coeff 
  (a b c d e f : ℝ) 
  (Q : ℝ → ℝ) 
  (h_Q : ∀ x, Q x = x^6 + a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) 
  (h_roots : ∃ p q r s t : ℝ, p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧ t ≠ 0 ∧ 
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
    q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
    r ≠ s ∧ r ≠ t ∧ 
    s ≠ t ∧
    Q p = 0 ∧ Q q = 0 ∧ Q r = 0 ∧ Q s = 0 ∧ Q t = 0)
  (h_zero_root : Q 0 = 0) :
  d ≠ 0 := by
sorry

end polynomial_intercept_nonzero_coeff_l1310_131086


namespace ben_sandwich_options_l1310_131085

/-- Represents the number of different types for each sandwich component -/
structure SandwichOptions where
  bread : Nat
  meat : Nat
  cheese : Nat

/-- Represents specific sandwich combinations that are not allowed -/
structure ForbiddenCombinations where
  beef_swiss : Nat
  rye_turkey : Nat
  turkey_swiss : Nat

/-- Calculates the number of sandwich options given the available choices and forbidden combinations -/
def calculate_sandwich_options (options : SandwichOptions) (forbidden : ForbiddenCombinations) : Nat :=
  options.bread * options.meat * options.cheese - (forbidden.beef_swiss + forbidden.rye_turkey + forbidden.turkey_swiss)

/-- The main theorem stating the number of different sandwiches Ben could order -/
theorem ben_sandwich_options :
  let options : SandwichOptions := { bread := 5, meat := 7, cheese := 6 }
  let forbidden : ForbiddenCombinations := { beef_swiss := 5, rye_turkey := 6, turkey_swiss := 5 }
  calculate_sandwich_options options forbidden = 194 := by
  sorry

end ben_sandwich_options_l1310_131085


namespace hourly_rate_approximation_l1310_131050

/-- Calculates the hourly rate based on given salary information and work schedule. -/
def calculate_hourly_rate (base_salary : ℚ) (commission_rate : ℚ) (total_sales : ℚ) 
  (performance_bonus : ℚ) (deductions : ℚ) (hours_per_day : ℕ) (days_per_week : ℕ) 
  (weeks_per_month : ℕ) : ℚ :=
  let total_earnings := base_salary + (commission_rate * total_sales) + performance_bonus - deductions
  let total_hours := hours_per_day * days_per_week * weeks_per_month
  total_earnings / total_hours

/-- Proves that the hourly rate is approximately $3.86 given the specified conditions. -/
theorem hourly_rate_approximation :
  let base_salary : ℚ := 576
  let commission_rate : ℚ := 3 / 100
  let total_sales : ℚ := 4000
  let performance_bonus : ℚ := 75
  let deductions : ℚ := 30
  let hours_per_day : ℕ := 8
  let days_per_week : ℕ := 6
  let weeks_per_month : ℕ := 4
  let hourly_rate := calculate_hourly_rate base_salary commission_rate total_sales 
    performance_bonus deductions hours_per_day days_per_week weeks_per_month
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |hourly_rate - 386/100| < ε :=
by
  sorry


end hourly_rate_approximation_l1310_131050


namespace fibonacci_sum_theorem_l1310_131048

/-- Definition of the Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The sum of the Fibonacci series divided by powers of 10 -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / (10 : ℝ) ^ n

/-- Theorem stating that the sum of Fₙ/10ⁿ from n=0 to infinity equals 10/89 -/
theorem fibonacci_sum_theorem : fibSum = 10 / 89 := by
  sorry

end fibonacci_sum_theorem_l1310_131048


namespace balanced_leaving_probability_formula_l1310_131014

/-- The probability that 3n students leaving from 3 rows of n students, one at a time
    with all leaving orders equally likely, such that there are never two rows where
    the number of students remaining differs by 2 or more. -/
def balanced_leaving_probability (n : ℕ) : ℚ :=
  (6 * n * (n.factorial ^ 3 : ℚ)) / ((3 * n).factorial : ℚ)

/-- Theorem stating that the probability of balanced leaving for 3n students
    in 3 rows of n is equal to (6n * (n!)^3) / (3n)! -/
theorem balanced_leaving_probability_formula (n : ℕ) (h : n ≥ 1) :
  balanced_leaving_probability n = (6 * n * (n.factorial ^ 3 : ℚ)) / ((3 * n).factorial : ℚ) :=
by sorry

end balanced_leaving_probability_formula_l1310_131014


namespace cheaper_to_buy_more_count_l1310_131004

def C (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 15 then 15 * n + 20
  else if 16 ≤ n ∧ n ≤ 30 then 13 * n
  else if 31 ≤ n ∧ n ≤ 45 then 11 * n + 50
  else 9 * n

theorem cheaper_to_buy_more_count :
  (∃ s : Finset ℕ, s.card = 4 ∧ ∀ n ∈ s, C (n + 1) < C n) ∧
  ¬(∃ s : Finset ℕ, s.card > 4 ∧ ∀ n ∈ s, C (n + 1) < C n) :=
by sorry

end cheaper_to_buy_more_count_l1310_131004


namespace gcd_840_1785_f_2_equals_62_l1310_131043

-- Define the polynomial f(x) = 2x⁴ + 3x³ + 5x - 4
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

-- Theorem for the GCD of 840 and 1785
theorem gcd_840_1785 : Nat.gcd 840 1785 = 105 := by sorry

-- Theorem for the value of f(2)
theorem f_2_equals_62 : f 2 = 62 := by sorry

end gcd_840_1785_f_2_equals_62_l1310_131043


namespace sqrt_eight_and_nine_sixteenths_l1310_131056

theorem sqrt_eight_and_nine_sixteenths :
  Real.sqrt (8 + 9 / 16) = Real.sqrt 137 / 4 := by
  sorry

end sqrt_eight_and_nine_sixteenths_l1310_131056


namespace isosceles_trapezoid_rotation_l1310_131092

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  height : ℝ
  is_isosceles : True

/-- Represents a geometric solid -/
inductive Solid
  | Cylinder
  | Cone
  | Frustum

/-- The result of rotating an isosceles trapezoid -/
def rotate_isosceles_trapezoid (t : IsoscelesTrapezoid) : List Solid :=
  sorry

/-- Theorem stating that rotating an isosceles trapezoid around its longer base
    results in one cylinder and two cones -/
theorem isosceles_trapezoid_rotation 
  (t : IsoscelesTrapezoid) : 
  rotate_isosceles_trapezoid t = [Solid.Cylinder, Solid.Cone, Solid.Cone] :=
sorry

end isosceles_trapezoid_rotation_l1310_131092


namespace quadratic_function_derivative_l1310_131035

theorem quadratic_function_derivative (a c : ℝ) :
  (∀ x, deriv (fun x => a * x^2 + c) x = 2 * a * x) →
  deriv (fun x => a * x^2 + c) 1 = 2 →
  a = 1 := by
  sorry

end quadratic_function_derivative_l1310_131035


namespace polynomial_expansion_l1310_131039

theorem polynomial_expansion (x : ℝ) : 
  (1 - x^3) * (1 + x^4 - x^5) = 1 - x^3 + x^4 - x^5 - x^7 + x^8 := by
  sorry

end polynomial_expansion_l1310_131039


namespace number_problem_l1310_131090

theorem number_problem (x : ℤ) : x + 12 - 27 = 24 → x = 39 := by
  sorry

end number_problem_l1310_131090


namespace prob_red_ball_l1310_131033

def urn1_red : ℚ := 3 / 8
def urn1_total : ℚ := 8
def urn2_red : ℚ := 1 / 2
def urn2_total : ℚ := 8
def urn3_red : ℚ := 0
def urn3_total : ℚ := 8

def prob_urn_selection : ℚ := 1 / 3

theorem prob_red_ball : 
  prob_urn_selection * (urn1_red * (urn1_total / urn1_total) + 
                        urn2_red * (urn2_total / urn2_total) + 
                        urn3_red * (urn3_total / urn3_total)) = 7 / 24 := by
  sorry

end prob_red_ball_l1310_131033


namespace combination_formula_l1310_131070

/-- The number of combinations of n things taken k at a time -/
def binomial (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem combination_formula (n m : ℕ) (h : n ≥ m - 1) :
  binomial n (m - 1) = Nat.factorial n / (Nat.factorial (m - 1) * Nat.factorial (n - m + 1)) := by
  sorry

end combination_formula_l1310_131070


namespace consecutive_integers_problem_l1310_131019

theorem consecutive_integers_problem (x y z : ℤ) :
  (x = y + 1) →
  (y = z + 1) →
  (x > y) →
  (y > z) →
  (2 * x + 3 * y + 3 * z = 5 * y + 8) →
  z = 2 := by
  sorry

end consecutive_integers_problem_l1310_131019


namespace no_equal_coin_exchange_l1310_131003

theorem no_equal_coin_exchange : ¬ ∃ (n : ℕ), n > 0 ∧ n * (1 + 2 + 3 + 5) = 500 := by
  sorry

end no_equal_coin_exchange_l1310_131003


namespace abc_acute_angle_implies_m_values_l1310_131079

def OA : Fin 2 → ℝ := ![3, -4]
def OB : Fin 2 → ℝ := ![6, -3]
def OC (m : ℝ) : Fin 2 → ℝ := ![5 - m, -3 - m]

def BA : Fin 2 → ℝ := ![OA 0 - OB 0, OA 1 - OB 1]
def BC (m : ℝ) : Fin 2 → ℝ := ![OC m 0 - OB 0, OC m 1 - OB 1]

def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0 * w 0) + (v 1 * w 1)

def is_acute_angle (m : ℝ) : Prop := dot_product BA (BC m) > 0

theorem abc_acute_angle_implies_m_values :
  ∀ m : ℝ, is_acute_angle m → (m = 0 ∨ m = 1) :=
by sorry

end abc_acute_angle_implies_m_values_l1310_131079


namespace product_of_digits_not_divisible_by_four_l1310_131044

def numbers : List Nat := [4628, 4638, 4648, 4658, 4662]

theorem product_of_digits_not_divisible_by_four : 
  ∃ n ∈ numbers, 
    ¬(n % 4 = 0) ∧ 
    ((n % 100) % 10 * ((n % 100) / 10 % 10) = 24) := by
  sorry

end product_of_digits_not_divisible_by_four_l1310_131044


namespace common_solution_y_value_l1310_131026

theorem common_solution_y_value (x y : ℝ) 
  (eq1 : x^2 + y^2 = 25) 
  (eq2 : x^2 + y = 10) : 
  y = (1 - Real.sqrt 61) / 2 := by
  sorry

end common_solution_y_value_l1310_131026


namespace isabel_music_purchase_l1310_131036

theorem isabel_music_purchase (country_albums : ℕ) (pop_albums : ℕ) (songs_per_album : ℕ) : 
  country_albums = 4 → pop_albums = 5 → songs_per_album = 8 → 
  (country_albums + pop_albums) * songs_per_album = 72 := by
sorry

end isabel_music_purchase_l1310_131036


namespace song_length_proof_l1310_131061

/-- Proves that given the conditions, each song on the album is 3.5 minutes long -/
theorem song_length_proof 
  (jumps_per_second : ℕ) 
  (total_songs : ℕ) 
  (total_jumps : ℕ) 
  (h1 : jumps_per_second = 1)
  (h2 : total_songs = 10)
  (h3 : total_jumps = 2100) :
  (total_jumps : ℚ) / (jumps_per_second * 60 * total_songs) = 3.5 := by
  sorry

end song_length_proof_l1310_131061


namespace unique_solution_l1310_131063

/-- The # operation as defined in the problem -/
def hash (a b : ℝ) : ℝ := a * b - 2 * a - 2 * b + 6

/-- Statement of the problem -/
theorem unique_solution : ∃! (x : ℝ), x > 0 ∧ hash (hash x 7) x = 82 := by
  sorry

end unique_solution_l1310_131063


namespace third_visit_next_month_l1310_131053

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a schedule of pool visits -/
structure PoolSchedule :=
  (visit_days : List DayOfWeek)

/-- Represents a month's pool visits -/
structure MonthVisits :=
  (count : Nat)

/-- Function to calculate the date of the nth visit in the next month -/
def nextMonthVisitDate (schedule : PoolSchedule) (current_month : MonthVisits) (n : Nat) : Nat :=
  sorry

/-- Theorem statement -/
theorem third_visit_next_month 
  (schedule : PoolSchedule)
  (current_month : MonthVisits)
  (h1 : schedule.visit_days = [DayOfWeek.Wednesday, DayOfWeek.Friday])
  (h2 : current_month.count = 10) :
  nextMonthVisitDate schedule current_month 3 = 12 :=
sorry

end third_visit_next_month_l1310_131053


namespace solution_set_f_geq_1_range_of_m_l1310_131027

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f_geq_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} :=
sorry

-- Theorem for the range of m
theorem range_of_m :
  {m : ℝ | ∀ x, |m - 2| ≥ |f x|} = {m : ℝ | m ≥ 5 ∨ m ≤ -1} :=
sorry

end solution_set_f_geq_1_range_of_m_l1310_131027


namespace height_difference_l1310_131038

/-- The height difference between the tallest and shortest players on a basketball team. -/
theorem height_difference (tallest_height shortest_height : ℝ) 
  (h_tallest : tallest_height = 77.75)
  (h_shortest : shortest_height = 68.25) : 
  tallest_height - shortest_height = 9.5 := by
  sorry

end height_difference_l1310_131038


namespace sum_equals_221_2357_l1310_131055

theorem sum_equals_221_2357 : 217 + 2.017 + 0.217 + 2.0017 = 221.2357 := by
  sorry

end sum_equals_221_2357_l1310_131055


namespace prob_only_one_AB_qualifies_prob_at_least_one_qualifies_l1310_131000

-- Define the probabilities for each student passing each round
def prob_written_A : ℚ := 2/3
def prob_written_B : ℚ := 1/2
def prob_written_C : ℚ := 3/4
def prob_interview_A : ℚ := 1/2
def prob_interview_B : ℚ := 2/3
def prob_interview_C : ℚ := 1/3

-- Define the probability of each student qualifying for the finals
def prob_qualify_A : ℚ := prob_written_A * prob_interview_A
def prob_qualify_B : ℚ := prob_written_B * prob_interview_B
def prob_qualify_C : ℚ := prob_written_C * prob_interview_C

-- Theorem for the first question
theorem prob_only_one_AB_qualifies :
  (prob_qualify_A * (1 - prob_qualify_B) + (1 - prob_qualify_A) * prob_qualify_B) = 4/9 := by
  sorry

-- Theorem for the second question
theorem prob_at_least_one_qualifies :
  (1 - (1 - prob_qualify_A) * (1 - prob_qualify_B) * (1 - prob_qualify_C)) = 2/3 := by
  sorry

end prob_only_one_AB_qualifies_prob_at_least_one_qualifies_l1310_131000


namespace solution_set_part1_range_of_a_part2_l1310_131002

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 1 ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} := by sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∃ x, f x a < 2*a} = {a : ℝ | a > 3} := by sorry

end solution_set_part1_range_of_a_part2_l1310_131002


namespace arithmetic_square_root_property_l1310_131020

theorem arithmetic_square_root_property (π : ℝ) : 
  Real.sqrt ((π - 4)^2) = 4 - π := by sorry

end arithmetic_square_root_property_l1310_131020


namespace cafeteria_pies_l1310_131098

theorem cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) 
  (h1 : initial_apples = 50)
  (h2 : handed_out = 5)
  (h3 : apples_per_pie = 5) :
  (initial_apples - handed_out) / apples_per_pie = 9 := by
  sorry

end cafeteria_pies_l1310_131098


namespace fixed_point_quadratic_l1310_131016

theorem fixed_point_quadratic (p : ℝ) : 
  9 * (5 : ℝ)^2 + p * 5 - 5 * p = 225 := by sorry

end fixed_point_quadratic_l1310_131016


namespace turn_over_five_most_effective_l1310_131051

-- Define the type for card sides
inductive CardSide
| Letter (c : Char)
| Number (n : Nat)

-- Define a card as a pair of sides
def Card := (CardSide × CardSide)

-- Define a function to check if a character is a vowel
def isVowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U']

-- Define a function to check if a number is even
def isEven (n : Nat) : Bool :=
  n % 2 = 0

-- Define Jane's claim as a function
def janesClaimHolds (card : Card) : Bool :=
  match card with
  | (CardSide.Letter c, CardSide.Number n) => ¬(isVowel c) ∨ isEven n
  | (CardSide.Number n, CardSide.Letter c) => ¬(isVowel c) ∨ isEven n
  | _ => true

-- Define the set of cards on the table
def cardsOnTable : List Card := [
  (CardSide.Letter 'A', CardSide.Number 0),  -- 0 is a placeholder
  (CardSide.Letter 'T', CardSide.Number 0),
  (CardSide.Letter 'U', CardSide.Number 0),
  (CardSide.Number 5, CardSide.Letter ' '),  -- ' ' is a placeholder
  (CardSide.Number 8, CardSide.Letter ' '),
  (CardSide.Number 10, CardSide.Letter ' '),
  (CardSide.Number 14, CardSide.Letter ' ')
]

-- Theorem: Turning over the card with 5 is the most effective way to potentially disprove Jane's claim
theorem turn_over_five_most_effective :
  ∃ (card : Card), card ∈ cardsOnTable ∧ 
  (∃ (c : Char), card = (CardSide.Number 5, CardSide.Letter c)) ∧
  (∀ (otherCard : Card), otherCard ∈ cardsOnTable → otherCard ≠ card →
    (∃ (possibleChar : Char), 
      ¬(janesClaimHolds (CardSide.Number 5, CardSide.Letter possibleChar)) →
      (janesClaimHolds otherCard ∨ 
       ∀ (possibleNum : Nat), janesClaimHolds (CardSide.Letter possibleChar, CardSide.Number possibleNum))))
  := by sorry


end turn_over_five_most_effective_l1310_131051


namespace isosceles_triangles_105_similar_l1310_131073

-- Define an isosceles triangle with a specific angle
structure IsoscelesTriangle :=
  (base_angle : ℝ)
  (vertex_angle : ℝ)
  (is_isosceles : base_angle * 2 + vertex_angle = 180)

-- Define similarity for isosceles triangles
def are_similar (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.base_angle = t2.base_angle ∧ t1.vertex_angle = t2.vertex_angle

-- Theorem statement
theorem isosceles_triangles_105_similar :
  ∀ (t1 t2 : IsoscelesTriangle),
  t1.vertex_angle = 105 → t2.vertex_angle = 105 →
  are_similar t1 t2 :=
sorry

end isosceles_triangles_105_similar_l1310_131073


namespace ticket_cost_proof_l1310_131095

def adult_price : ℕ := 12
def child_price : ℕ := 10
def senior_price : ℕ := 8
def student_price : ℕ := 9

def num_parents : ℕ := 2
def num_grandparents : ℕ := 2
def num_sisters : ℕ := 3
def num_cousins : ℕ := 1
def num_uncle_aunt : ℕ := 2

def total_cost : ℕ :=
  num_parents * adult_price +
  num_grandparents * senior_price +
  num_sisters * child_price +
  num_cousins * student_price +
  num_uncle_aunt * adult_price

theorem ticket_cost_proof : total_cost = 103 := by
  sorry

end ticket_cost_proof_l1310_131095


namespace expression_value_l1310_131052

theorem expression_value (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = 1) :
  3 * x - 2 * y + 4 * z = 9 := by
  sorry

end expression_value_l1310_131052


namespace unique_four_digit_int_l1310_131013

/-- Represents a four-digit positive integer --/
structure FourDigitInt where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  positive : 1000 ≤ a * 1000 + b * 100 + c * 10 + d

/-- The conditions given in the problem --/
def satisfiesConditions (n : FourDigitInt) : Prop :=
  n.a + n.b + n.c + n.d = 17 ∧
  n.b + n.c = 9 ∧
  n.a - n.d = 2 ∧
  (n.a * 1000 + n.b * 100 + n.c * 10 + n.d) % 9 = 0

/-- The theorem to be proved --/
theorem unique_four_digit_int :
  ∃! (n : FourDigitInt), satisfiesConditions n ∧ n.a = 5 ∧ n.b = 4 ∧ n.c = 5 ∧ n.d = 3 :=
sorry

end unique_four_digit_int_l1310_131013


namespace not_obtainable_2013201420152016_l1310_131034

/-- Represents the state of the board -/
structure Board :=
  (left : ℕ)
  (right : ℕ)

/-- Represents a single operation on the board -/
def operate (b : Board) : Board :=
  { left := b.left * b.right,
    right := b.left^3 + b.right^3 }

/-- Checks if a number is obtainable on the board -/
def is_obtainable (n : ℕ) : Prop :=
  ∃ (b : Board), ∃ (k : ℕ), 
    (Nat.iterate operate k { left := 21, right := 8 }).left = n ∨
    (Nat.iterate operate k { left := 21, right := 8 }).right = n

/-- The main theorem stating that 2013201420152016 is not obtainable -/
theorem not_obtainable_2013201420152016 : 
  ¬ is_obtainable 2013201420152016 := by
  sorry


end not_obtainable_2013201420152016_l1310_131034


namespace smallest_prime_after_seven_nonprimes_l1310_131022

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def has_seven_consecutive_nonprimes (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ ∀ i : ℕ, i ≥ k ∧ i < k + 7 → ¬(is_prime i)

theorem smallest_prime_after_seven_nonprimes :
  (is_prime 97) ∧ 
  (has_seven_consecutive_nonprimes 90) ∧
  (∀ p : ℕ, p < 97 → ¬(is_prime p ∧ has_seven_consecutive_nonprimes (p - 7))) :=
by sorry

end smallest_prime_after_seven_nonprimes_l1310_131022


namespace quadratic_set_equality_l1310_131082

theorem quadratic_set_equality (p : ℝ) : 
  ({x : ℝ | x^2 - 5*x + p ≥ 0} = {x : ℝ | x ≤ -1 ∨ x ≥ 6}) → p = -6 :=
by sorry

end quadratic_set_equality_l1310_131082


namespace curve_k_values_l1310_131097

-- Define the curve equation
def curve_equation (x y k : ℝ) : Prop :=
  5 * x^2 - k * y^2 = 5

-- Define the focal length
def focal_length : ℝ := 4

-- Theorem statement
theorem curve_k_values :
  ∃ k : ℝ, (k = 5/3 ∨ k = -1) ∧
  ∀ x y : ℝ, curve_equation x y k ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ curve_equation x y k) ∧
    (max a b - min a b) / 2 = focal_length) :=
sorry

end curve_k_values_l1310_131097


namespace ellipse_properties_l1310_131054

-- Define the ellipse E
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection of a line with the ellipse
def line_ellipse_intersection (k : ℝ) (a b : ℝ) (x : ℝ) : Prop :=
  (3 + 4*k^2) * x^2 - 8*k^2 * x + (4*k^2 - 12) = 0

-- Main theorem
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ (x y : ℝ), ellipse a b x y ∧ parabola x y) →
  (∃ (x1 y1 x2 y2 : ℝ), ellipse a b x1 y1 ∧ ellipse a b x2 y2 ∧ 
    parabola x1 y1 ∧ parabola x2 y2 ∧ 
    ((x2 - x1)^2 + (y2 - y1)^2)^(1/2 : ℝ) = 3) →
  (a = 2 ∧ b = (3 : ℝ)^(1/2 : ℝ)) ∧
  (∀ (k : ℝ), k ≠ 0 →
    (∃ (x1 x2 x3 x4 : ℝ), 
      line_ellipse_intersection k a b x1 ∧
      line_ellipse_intersection k a b x2 ∧
      line_ellipse_intersection (-1/k) a b x3 ∧
      line_ellipse_intersection (-1/k) a b x4 ∧
      (∃ (r : ℝ), 
        (x1 - 1)^2 + (k*(x1 - 1))^2 = r^2 ∧
        (x2 - 1)^2 + (k*(x2 - 1))^2 = r^2 ∧
        (x3 - 1)^2 + (-1/k*(x3 - 1))^2 = r^2 ∧
        (x4 - 1)^2 + (-1/k*(x4 - 1))^2 = r^2)) ↔
    (k = 1 ∨ k = -1)) :=
by sorry

end ellipse_properties_l1310_131054


namespace parallel_vectors_x_value_l1310_131069

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, -4)
  parallel a b → x = -2 :=
by
  sorry

end parallel_vectors_x_value_l1310_131069


namespace proposition_is_true_l1310_131025

theorem proposition_is_true : ∀ (x y : ℝ), x + 2*y ≠ 5 → x ≠ 1 ∨ y ≠ 2 := by sorry

end proposition_is_true_l1310_131025


namespace circle_triangle_intersection_l1310_131078

/-- Given an equilateral triangle intersected by a circle at six points, 
    this theorem proves the length of DE based on other given lengths. -/
theorem circle_triangle_intersection (AG GF FC HJ : ℝ) (h1 : AG = 2) (h2 : GF = 13) 
  (h3 : FC = 1) (h4 : HJ = 7) : ∃ (DE : ℝ), DE = 2 * Real.sqrt 22 := by
  sorry

end circle_triangle_intersection_l1310_131078


namespace total_distance_is_1734_l1310_131068

/-- The number of trees in the row -/
def num_trees : ℕ := 18

/-- The interval between adjacent trees in meters -/
def tree_interval : ℕ := 3

/-- Calculate the total distance walked to water all trees -/
def total_distance : ℕ :=
  -- Sum of distances for each tree
  (Finset.range num_trees).sum (fun i => 2 * i * tree_interval)

/-- Theorem stating the total distance walked -/
theorem total_distance_is_1734 : total_distance = 1734 := by
  sorry

end total_distance_is_1734_l1310_131068


namespace intersection_locus_is_ellipse_l1310_131084

/-- The locus of points (x, y) satisfying a system of equations forms an ellipse -/
theorem intersection_locus_is_ellipse :
  ∀ (s x y : ℝ), 
  (2 * s * x - 3 * y - 4 * s = 0) → 
  (x - 3 * s * y + 4 = 0) → 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end intersection_locus_is_ellipse_l1310_131084


namespace part_a_part_b_l1310_131087

-- Define the set M of functions satisfying the given conditions
def M : Set (ℤ → ℝ) :=
  {f | f 0 ≠ 0 ∧ ∀ n m : ℤ, f n * f m = f (n + m) + f (n - m)}

-- Theorem for part (a)
theorem part_a (f : ℤ → ℝ) (hf : f ∈ M) (h1 : f 1 = 5/2) :
  ∀ n : ℤ, f n = 2^n + 2^(-n) := by sorry

-- Theorem for part (b)
theorem part_b (f : ℤ → ℝ) (hf : f ∈ M) (h1 : f 1 = Real.sqrt 3) :
  ∀ n : ℤ, f n = 2 * Real.cos (π * n / 6) := by sorry

end part_a_part_b_l1310_131087


namespace sum_of_digits_of_f_l1310_131089

/-- The number of digits in (10^2020 + 2020)^2 when written out in full -/
def num_digits : ℕ := 4041

/-- The function that calculates (10^2020 + 2020)^2 -/
def f : ℕ := (10^2020 + 2020)^2

/-- The sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_f : sum_of_digits f = 25 := by sorry

end sum_of_digits_of_f_l1310_131089


namespace perfect_square_trinomial_m_value_l1310_131075

theorem perfect_square_trinomial_m_value (m : ℝ) : 
  (∃ k : ℝ, ∀ y : ℝ, 4*y^2 - m*y + 25 = (2*y - k)^2) → 
  (m = 20 ∨ m = -20) := by
sorry

end perfect_square_trinomial_m_value_l1310_131075


namespace sum_of_solutions_quadratic_l1310_131060

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 9*x - 20) → (∃ y : ℝ, y^2 = 9*y - 20 ∧ x + y = 9) :=
by sorry

end sum_of_solutions_quadratic_l1310_131060


namespace equation_solution_solution_set_l1310_131099

theorem equation_solution (x : ℝ) : 
  x ≠ 7 → (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 3*x - 18) / (x^2 - 4*x - 21) :=
by sorry

theorem solution_set : 
  {x : ℝ | (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 3*x - 18) / (x^2 - 4*x - 21)} = {x : ℝ | x ≠ 7} :=
by sorry

end equation_solution_solution_set_l1310_131099


namespace star_operation_result_l1310_131045

def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 4}

def star_operation (X Y : Set Nat) : Set Nat :=
  {x | x ∈ X ∧ x ∉ Y}

theorem star_operation_result :
  star_operation A B = {1, 3} := by
  sorry

end star_operation_result_l1310_131045


namespace linear_function_range_l1310_131007

theorem linear_function_range (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (m + 2) * x₁ + (1 - m) > (m + 2) * x₂ + (1 - m)) →
  (∃ x : ℝ, x > 0 ∧ (m + 2) * x + (1 - m) = 0) →
  m < -2 := by
sorry

end linear_function_range_l1310_131007


namespace right_triangle_sides_l1310_131015

theorem right_triangle_sides (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_ratio : ∃ (k : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k) 
  (h_area : a * b / 2 = 24) :
  a = 6 ∧ b = 8 ∧ c = 10 := by
sorry

end right_triangle_sides_l1310_131015


namespace non_zero_vector_positive_norm_l1310_131096

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem non_zero_vector_positive_norm (a b : V) 
  (h_a : a ≠ 0) (h_b : ‖b‖ = 1) : 
  ‖a‖ > 0 := by sorry

end non_zero_vector_positive_norm_l1310_131096


namespace nabla_calculation_l1310_131074

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem nabla_calculation : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end nabla_calculation_l1310_131074


namespace find_A_l1310_131031

theorem find_A : ∃ A : ℤ, A + 19 = 47 ∧ A = 28 := by
  sorry

end find_A_l1310_131031


namespace percentage_greater_l1310_131018

theorem percentage_greater (A B : ℝ) (y : ℝ) (h1 : A > B) (h2 : B > 0) : 
  let C := A + B
  y = 100 * ((C - B) / B) → y = 100 * (A / B) := by
sorry

end percentage_greater_l1310_131018


namespace arthur_purchase_cost_l1310_131030

/-- The cost of Arthur's purchases on two days -/
theorem arthur_purchase_cost
  (hamburger_cost : ℝ)
  (hot_dog_cost : ℝ)
  (day1_total : ℝ)
  (h_hot_dog_cost : hot_dog_cost = 1)
  (h_day1_equation : 3 * hamburger_cost + 4 * hot_dog_cost = day1_total)
  (h_day1_total : day1_total = 10) :
  2 * hamburger_cost + 3 * hot_dog_cost = 7 :=
by sorry

end arthur_purchase_cost_l1310_131030
