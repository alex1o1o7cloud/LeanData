import Mathlib

namespace NUMINAMATH_CALUDE_grunters_win_probability_l668_66866

def number_of_games : ℕ := 5
def win_probability : ℚ := 3/5

theorem grunters_win_probability :
  let p := win_probability
  let n := number_of_games
  (n.choose 4 * p^4 * (1-p)^1) + p^n = 1053/3125 := by sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l668_66866


namespace NUMINAMATH_CALUDE_fraction_problem_l668_66893

theorem fraction_problem (x : ℚ) : 
  x / (4 * x + 5) = 3 / 7 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l668_66893


namespace NUMINAMATH_CALUDE_glucose_solution_volume_l668_66873

/-- Given a glucose solution where 500 cubic centimeters contain 10 grams of glucose,
    this theorem proves that the volume of solution containing 20 grams of glucose
    is 1000 cubic centimeters. -/
theorem glucose_solution_volume :
  let volume_500cc : ℝ := 500
  let glucose_500cc : ℝ := 10
  let glucose_target : ℝ := 20
  let volume_target : ℝ := (glucose_target * volume_500cc) / glucose_500cc
  volume_target = 1000 := by
  sorry

end NUMINAMATH_CALUDE_glucose_solution_volume_l668_66873


namespace NUMINAMATH_CALUDE_water_mass_in_range_l668_66821

/-- Represents the thermodynamic properties of a substance -/
structure ThermodynamicProperties where
  specific_heat_capacity : Real
  specific_latent_heat : Real

/-- Represents the initial state of a substance -/
structure InitialState where
  mass : Real
  temperature : Real

/-- Calculates the range of added water mass given the initial conditions and final temperature -/
def calculate_water_mass_range (ice_props : ThermodynamicProperties)
                               (water_props : ThermodynamicProperties)
                               (ice_initial : InitialState)
                               (water_initial : InitialState)
                               (final_temp : Real) : Set Real :=
  sorry

/-- Theorem stating that the mass of added water lies within the calculated range -/
theorem water_mass_in_range :
  let ice_props : ThermodynamicProperties := {
    specific_heat_capacity := 2100,
    specific_latent_heat := 3.3e5
  }
  let water_props : ThermodynamicProperties := {
    specific_heat_capacity := 4200,
    specific_latent_heat := 0
  }
  let ice_initial : InitialState := {
    mass := 0.1,
    temperature := -5
  }
  let water_initial : InitialState := {
    mass := 0,  -- mass to be determined
    temperature := 10
  }
  let final_temp : Real := 0
  let water_mass_range := calculate_water_mass_range ice_props water_props ice_initial water_initial final_temp
  ∀ m ∈ water_mass_range, 0.0028 ≤ m ∧ m ≤ 0.8119 :=
by sorry

end NUMINAMATH_CALUDE_water_mass_in_range_l668_66821


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l668_66855

theorem contrapositive_equivalence (a b : ℝ) : 
  (¬(a - 1 > b - 2) → ¬(a > b)) ↔ (a - 1 ≤ b - 2 → a ≤ b) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l668_66855


namespace NUMINAMATH_CALUDE_function_not_in_first_quadrant_l668_66812

-- Define the linear function
def f (x : ℝ) : ℝ := -3 * x - 2

-- Theorem: The function f does not pass through the first quadrant
theorem function_not_in_first_quadrant :
  ∀ x y : ℝ, f x = y → ¬(x > 0 ∧ y > 0) := by
  sorry

end NUMINAMATH_CALUDE_function_not_in_first_quadrant_l668_66812


namespace NUMINAMATH_CALUDE_candies_needed_to_fill_bags_l668_66867

theorem candies_needed_to_fill_bags (total_candies : ℕ) (bag_capacity : ℕ) (h1 : total_candies = 254) (h2 : bag_capacity = 30) : 
  (bag_capacity - (total_candies % bag_capacity)) = 16 := by
sorry

end NUMINAMATH_CALUDE_candies_needed_to_fill_bags_l668_66867


namespace NUMINAMATH_CALUDE_purely_imaginary_fraction_l668_66804

theorem purely_imaginary_fraction (a : ℝ) : 
  (∃ k : ℝ, (a - I) / (1 + I) = k * I) → a = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_fraction_l668_66804


namespace NUMINAMATH_CALUDE_two_a_minus_a_equals_a_l668_66834

theorem two_a_minus_a_equals_a (a : ℝ) : 2 * a - a = a := by
  sorry

end NUMINAMATH_CALUDE_two_a_minus_a_equals_a_l668_66834


namespace NUMINAMATH_CALUDE_locus_not_hyperbola_ellipse_intersection_l668_66870

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def tangent (c1 c2 : Circle) : Prop :=
  dist c1.center c2.center = c1.radius + c2.radius ∨
  dist c1.center c2.center = |c1.radius - c2.radius|

def locus (O₁ O₂ : Circle) : Set (ℝ × ℝ) :=
  {P | ∃ (r : ℝ), tangent O₁ ⟨P, r⟩ ∧ tangent O₂ ⟨P, r⟩}

def hyperbola (f₁ f₂ : ℝ × ℝ) (a : ℝ) : Set (ℝ × ℝ) :=
  {P | |dist P f₁ - dist P f₂| = 2 * a}

def ellipse (f₁ f₂ : ℝ × ℝ) (a : ℝ) : Set (ℝ × ℝ) :=
  {P | dist P f₁ + dist P f₂ = 2 * a}

theorem locus_not_hyperbola_ellipse_intersection
  (O₁ O₂ : Circle) (f₁ f₂ g₁ g₂ : ℝ × ℝ) (a b : ℝ) :
  locus O₁ O₂ ≠ hyperbola f₁ f₂ a ∩ ellipse g₁ g₂ b :=
sorry

end NUMINAMATH_CALUDE_locus_not_hyperbola_ellipse_intersection_l668_66870


namespace NUMINAMATH_CALUDE_matrix_equality_proof_l668_66863

open Matrix

-- Define the condition for matrix congruence modulo 3
def congruent_mod_3 (X Y : Matrix (Fin 6) (Fin 6) ℤ) : Prop :=
  ∀ i j, (X i j - Y i j) % 3 = 0

-- Main theorem statement
theorem matrix_equality_proof (A B : Matrix (Fin 6) (Fin 6) ℤ)
  (h1 : congruent_mod_3 A (1 : Matrix (Fin 6) (Fin 6) ℤ))
  (h2 : congruent_mod_3 B (1 : Matrix (Fin 6) (Fin 6) ℤ))
  (h3 : A ^ 3 * B ^ 3 * A ^ 3 = B ^ 3) :
  A = 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equality_proof_l668_66863


namespace NUMINAMATH_CALUDE_triangle_area_l668_66839

/-- Given a triangle ABC where BC = 10 cm and the height from A to BC is 12 cm,
    prove that the area of triangle ABC is 60 square centimeters. -/
theorem triangle_area (BC height : ℝ) (h1 : BC = 10) (h2 : height = 12) :
  (1 / 2) * BC * height = 60 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l668_66839


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l668_66824

/-- A hyperbola with center at the origin and foci on the x-axis -/
structure Hyperbola where
  center : ℝ × ℝ
  foci_on_x_axis : Bool

/-- The length of a line segment -/
def length (a b : ℝ × ℝ) : ℝ := sorry

/-- The real axis of a hyperbola -/
def real_axis (h : Hyperbola) : ℝ := sorry

theorem hyperbola_real_axis_length
  (C : Hyperbola)
  (h_center : C.center = (0, 0))
  (h_foci : C.foci_on_x_axis = true)
  (A B : ℝ × ℝ)
  (h_intersect : A.1 = -4 ∧ B.1 = -4)
  (h_distance : length A B = 4) :
  real_axis C = 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l668_66824


namespace NUMINAMATH_CALUDE_smallest_root_of_equation_l668_66811

theorem smallest_root_of_equation (x : ℝ) : 
  (|x - 1| / x^2 = 6) → (x = -1/2 ∨ x = 1/3) ∧ (-1/2 < 1/3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_root_of_equation_l668_66811


namespace NUMINAMATH_CALUDE_rectangle_bisector_slope_l668_66842

/-- The slope of a line passing through the origin and the center of a rectangle
    with vertices (1, 0), (5, 0), (1, 2), and (5, 2) is 1/3. -/
theorem rectangle_bisector_slope :
  let vertices : List (ℝ × ℝ) := [(1, 0), (5, 0), (1, 2), (5, 2)]
  let center : ℝ × ℝ := (
    (vertices[0].1 + vertices[3].1) / 2,
    (vertices[0].2 + vertices[3].2) / 2
  )
  let slope : ℝ := center.2 / center.1
  slope = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_bisector_slope_l668_66842


namespace NUMINAMATH_CALUDE_total_toes_is_164_l668_66878

/-- Represents a race on planet Popton -/
inductive Race
| Hoopit
| Neglart

/-- Number of hands for each race -/
def hands (r : Race) : Nat :=
  match r with
  | Race.Hoopit => 4
  | Race.Neglart => 5

/-- Number of toes per hand for each race -/
def toes_per_hand (r : Race) : Nat :=
  match r with
  | Race.Hoopit => 3
  | Race.Neglart => 2

/-- Number of students of each race on the bus -/
def students (r : Race) : Nat :=
  match r with
  | Race.Hoopit => 7
  | Race.Neglart => 8

/-- Total number of toes for a single being of a given race -/
def toes_per_being (r : Race) : Nat :=
  hands r * toes_per_hand r

/-- Total number of toes on the bus for a given race -/
def total_toes_per_race (r : Race) : Nat :=
  students r * toes_per_being r

/-- Total number of toes on the Popton school bus -/
def total_toes_on_bus : Nat :=
  total_toes_per_race Race.Hoopit + total_toes_per_race Race.Neglart

theorem total_toes_is_164 : total_toes_on_bus = 164 := by
  sorry

end NUMINAMATH_CALUDE_total_toes_is_164_l668_66878


namespace NUMINAMATH_CALUDE_expression_evaluation_l668_66879

theorem expression_evaluation (a b : ℚ) (h1 : a = -1) (h2 : b = 1/2) :
  ((2*a + b)^2 - (2*a + b)*(2*a - b)) / (2*b) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l668_66879


namespace NUMINAMATH_CALUDE_pages_left_to_read_is_1000_l668_66890

/-- Calculates the number of pages left to read in a book series -/
def pagesLeftToRead (totalBooks : ℕ) (pagesPerBook : ℕ) (readFirstMonth : ℕ) : ℕ :=
  let remainingAfterFirstMonth := totalBooks - readFirstMonth
  let readSecondMonth := remainingAfterFirstMonth / 2
  let totalRead := readFirstMonth + readSecondMonth
  let pagesLeft := (totalBooks - totalRead) * pagesPerBook
  pagesLeft

/-- Theorem: Given the specified reading pattern, 1000 pages are left to read -/
theorem pages_left_to_read_is_1000 :
  pagesLeftToRead 14 200 4 = 1000 := by
  sorry

#eval pagesLeftToRead 14 200 4

end NUMINAMATH_CALUDE_pages_left_to_read_is_1000_l668_66890


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l668_66825

theorem smallest_x_absolute_value_equation :
  ∃ x : ℝ, (∀ y : ℝ, |y + 3| = 15 → x ≤ y) ∧ |x + 3| = 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l668_66825


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l668_66891

-- Define set A
def A : Set ℝ := {x | x^2 < 4}

-- Define set B
def B : Set ℝ := {x | -3 < x ∧ x ≤ 1}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l668_66891


namespace NUMINAMATH_CALUDE_tan_difference_pi_4_minus_theta_l668_66872

theorem tan_difference_pi_4_minus_theta (θ : Real) (h : Real.tan θ = 1/2) :
  Real.tan (π/4 - θ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_pi_4_minus_theta_l668_66872


namespace NUMINAMATH_CALUDE_weight_of_doubled_cube_l668_66859

/-- Given two cubes of the same material, if the second cube has sides twice as long
    as the first cube, and the first cube weighs 4 pounds, then the second cube weighs 32 pounds. -/
theorem weight_of_doubled_cube (s : ℝ) (weight_first : ℝ) (volume_first : ℝ) (weight_second : ℝ) :
  s > 0 →
  weight_first = 4 →
  volume_first = s^3 →
  weight_first / volume_first = weight_second / ((2*s)^3) →
  weight_second = 32 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_doubled_cube_l668_66859


namespace NUMINAMATH_CALUDE_ab_value_l668_66823

theorem ab_value (a b : ℝ) (h1 : a^2 + b^2 = 2) (h2 : a^4 + b^4 = 31/16) : 
  a * b = Real.sqrt (33/32) := by
sorry

end NUMINAMATH_CALUDE_ab_value_l668_66823


namespace NUMINAMATH_CALUDE_least_k_cubed_divisible_by_336_l668_66869

theorem least_k_cubed_divisible_by_336 :
  ∃ (k : ℕ), k > 0 ∧ k^3 % 336 = 0 ∧ ∀ (m : ℕ), m > 0 → m^3 % 336 = 0 → k ≤ m :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_least_k_cubed_divisible_by_336_l668_66869


namespace NUMINAMATH_CALUDE_pocket_money_problem_l668_66830

theorem pocket_money_problem (older_initial : ℕ) (younger_initial : ℕ) (difference : ℕ) (amount_given : ℕ) : 
  older_initial = 2800 →
  younger_initial = 1500 →
  older_initial - amount_given = younger_initial + amount_given + difference →
  difference = 360 →
  amount_given = 470 := by
  sorry

end NUMINAMATH_CALUDE_pocket_money_problem_l668_66830


namespace NUMINAMATH_CALUDE_number_of_baskets_l668_66815

theorem number_of_baskets (green_per_basket : ℕ) (total_green : ℕ) (h1 : green_per_basket = 2) (h2 : total_green = 14) :
  total_green / green_per_basket = 7 := by
sorry

end NUMINAMATH_CALUDE_number_of_baskets_l668_66815


namespace NUMINAMATH_CALUDE_nathaniel_best_friends_l668_66827

/-- Given that Nathaniel has 37 tickets initially, gives 5 tickets to each best friend,
    and ends up with 2 tickets, prove that he has 7 best friends. -/
theorem nathaniel_best_friends :
  let initial_tickets : ℕ := 37
  let tickets_per_friend : ℕ := 5
  let remaining_tickets : ℕ := 2
  let best_friends : ℕ := (initial_tickets - remaining_tickets) / tickets_per_friend
  best_friends = 7 := by
sorry


end NUMINAMATH_CALUDE_nathaniel_best_friends_l668_66827


namespace NUMINAMATH_CALUDE_max_knights_between_knights_is_32_l668_66877

/-- Represents a seating arrangement of knights and samurais around a round table. -/
structure SeatingArrangement where
  total_knights : ℕ
  total_samurais : ℕ
  knights_with_samurai_right : ℕ

/-- The maximum number of knights that could be seated next to two other knights. -/
def max_knights_between_knights (arrangement : SeatingArrangement) : ℕ :=
  arrangement.total_knights - (arrangement.knights_with_samurai_right + 1)

/-- Theorem stating the maximum number of knights that could be seated next to two other knights
    in the given arrangement. -/
theorem max_knights_between_knights_is_32 (arrangement : SeatingArrangement) 
  (h1 : arrangement.total_knights = 40)
  (h2 : arrangement.total_samurais = 10)
  (h3 : arrangement.knights_with_samurai_right = 7) :
  max_knights_between_knights arrangement = 32 := by
  sorry

#check max_knights_between_knights_is_32

end NUMINAMATH_CALUDE_max_knights_between_knights_is_32_l668_66877


namespace NUMINAMATH_CALUDE_percent_composition_l668_66846

-- Define the % operations
def percent_right (x : ℤ) : ℤ := 8 - x
def percent_left (x : ℤ) : ℤ := x - 8

-- Theorem statement
theorem percent_composition : percent_left (percent_right 10) = -10 := by
  sorry

end NUMINAMATH_CALUDE_percent_composition_l668_66846


namespace NUMINAMATH_CALUDE_cookie_price_is_two_l668_66887

/-- The price of each cookie in dollars, given the baking and sales conditions -/
def cookie_price (clementine_cookies jake_cookies tory_cookies total_revenue : ℕ) : ℚ :=
  total_revenue / (clementine_cookies + jake_cookies + tory_cookies)

theorem cookie_price_is_two :
  let clementine_cookies : ℕ := 72
  let jake_cookies : ℕ := 2 * clementine_cookies
  let tory_cookies : ℕ := (clementine_cookies + jake_cookies) / 2
  let total_revenue : ℕ := 648
  cookie_price clementine_cookies jake_cookies tory_cookies total_revenue = 2 := by
sorry

#eval cookie_price 72 144 108 648

end NUMINAMATH_CALUDE_cookie_price_is_two_l668_66887


namespace NUMINAMATH_CALUDE_parabola_line_slope_l668_66803

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - focus.1)

-- Define the condition for a point to be on both the line and the parabola
def intersection_point (k : ℝ) (x y : ℝ) : Prop :=
  parabola x y ∧ line_through_focus k x y

-- Define the ratio condition
def ratio_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 - focus.1)^2 + (A.2 - focus.2)^2 = 16 * ((B.1 - focus.1)^2 + (B.2 - focus.2)^2)

theorem parabola_line_slope (k : ℝ) (A B : ℝ × ℝ) :
  intersection_point k A.1 A.2 →
  intersection_point k B.1 B.2 →
  A ≠ B →
  ratio_condition A B →
  k = 4/3 ∨ k = -4/3 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_slope_l668_66803


namespace NUMINAMATH_CALUDE_sum_of_twelve_terms_special_case_l668_66809

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- The common difference
  h : ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence. -/
def sum_of_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1 : ℚ) * seq.d)

theorem sum_of_twelve_terms_special_case (seq : ArithmeticSequence) 
  (h₁ : seq.a 5 = 1)
  (h₂ : seq.a 17 = 18) :
  sum_of_terms seq 12 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_twelve_terms_special_case_l668_66809


namespace NUMINAMATH_CALUDE_theater_ticket_difference_l668_66848

theorem theater_ticket_difference :
  ∀ (orchestra_tickets balcony_tickets : ℕ),
    orchestra_tickets + balcony_tickets = 370 →
    12 * orchestra_tickets + 8 * balcony_tickets = 3320 →
    balcony_tickets - orchestra_tickets = 190 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_difference_l668_66848


namespace NUMINAMATH_CALUDE_investment_growth_l668_66874

/-- Calculates the total amount after compound interest is applied for a given number of periods -/
def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

/-- The problem statement -/
theorem investment_growth : compound_interest 300 0.1 2 = 363 := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l668_66874


namespace NUMINAMATH_CALUDE_cosine_product_equals_one_eighth_two_minus_sqrt_two_l668_66833

theorem cosine_product_equals_one_eighth_two_minus_sqrt_two :
  (1 + Real.cos (π / 9)) * (1 + Real.cos (4 * π / 9)) *
  (1 + Real.cos (5 * π / 9)) * (1 + Real.cos (8 * π / 9)) =
  1 / 8 * (2 - Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_cosine_product_equals_one_eighth_two_minus_sqrt_two_l668_66833


namespace NUMINAMATH_CALUDE_max_value_of_four_numbers_l668_66880

theorem max_value_of_four_numbers
  (a b c d : ℝ)
  (h_positive : 0 < d ∧ 0 < c ∧ 0 < b ∧ 0 < a)
  (h_order : d ≤ c ∧ c ≤ b ∧ b ≤ a)
  (h_sum : a + b + c + d = 4)
  (h_sum_squares : a^2 + b^2 + c^2 + d^2 = 8) :
  a ≤ 1 + Real.sqrt 3 ∧ ∃ (a₀ b₀ c₀ d₀ : ℝ),
    0 < d₀ ∧ d₀ ≤ c₀ ∧ c₀ ≤ b₀ ∧ b₀ ≤ a₀ ∧
    a₀ + b₀ + c₀ + d₀ = 4 ∧
    a₀^2 + b₀^2 + c₀^2 + d₀^2 = 8 ∧
    a₀ = 1 + Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_four_numbers_l668_66880


namespace NUMINAMATH_CALUDE_xyz_product_l668_66810

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + 1/y = 5) (eq2 : y + 1/z = 2) (eq3 : z + 2/x = 10/3) :
  x * y * z = (21 + Real.sqrt 433) / 2 := by
  sorry

end NUMINAMATH_CALUDE_xyz_product_l668_66810


namespace NUMINAMATH_CALUDE_max_y_over_x_l668_66841

-- Define the feasible region
def FeasibleRegion (x y : ℝ) : Prop :=
  x + y ≥ 3 ∧ x - y ≥ -1 ∧ 2*x - y ≤ 3

-- State the theorem
theorem max_y_over_x :
  ∃ (max : ℝ), max = 2 ∧
  ∀ (x y : ℝ), FeasibleRegion x y → y / x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_y_over_x_l668_66841


namespace NUMINAMATH_CALUDE_city_birth_rate_l668_66807

/-- Represents the birth rate problem in a city --/
theorem city_birth_rate 
  (death_rate : ℕ) 
  (net_increase : ℕ) 
  (intervals_per_day : ℕ) 
  (h1 : death_rate = 3)
  (h2 : net_increase = 129600)
  (h3 : intervals_per_day = 43200) :
  ∃ (birth_rate : ℕ), 
    birth_rate = 6 ∧ 
    (birth_rate - death_rate) * intervals_per_day = net_increase :=
by sorry

end NUMINAMATH_CALUDE_city_birth_rate_l668_66807


namespace NUMINAMATH_CALUDE_sum_digits_base8_888_l668_66899

/-- Converts a natural number to its base 8 representation as a list of digits -/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  digits.sum

theorem sum_digits_base8_888 : sumDigits (toBase8 888) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_base8_888_l668_66899


namespace NUMINAMATH_CALUDE_eighth_hexagonal_number_l668_66895

/-- Definition of hexagonal numbers -/
def hexagonal (n : ℕ) : ℕ := n * (2 * n - 1)

/-- The 8th hexagonal number is 120 -/
theorem eighth_hexagonal_number : hexagonal 8 = 120 := by
  sorry

end NUMINAMATH_CALUDE_eighth_hexagonal_number_l668_66895


namespace NUMINAMATH_CALUDE_equation_satisfied_l668_66849

theorem equation_satisfied (a b c : ℤ) (h1 : a = b) (h2 : b = c + 1) :
  a * (a - b) + b * (b - c) + c * (c - a) = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_l668_66849


namespace NUMINAMATH_CALUDE_boatman_distance_against_current_l668_66898

/-- Represents the speed of a boat in different water conditions -/
structure BoatSpeed where
  stillWater : ℝ
  current : ℝ

/-- Calculates the distance traveled given speed and time -/
def distanceTraveled (speed time : ℝ) : ℝ := speed * time

/-- Represents the problem of a boatman traveling in a stream -/
theorem boatman_distance_against_current 
  (boat : BoatSpeed)
  (h1 : distanceTraveled (boat.stillWater + boat.current) (1/3) = 1)
  (h2 : distanceTraveled boat.stillWater 3 = 6)
  (h3 : boat.stillWater > boat.current)
  (h4 : boat.current > 0) :
  distanceTraveled (boat.stillWater - boat.current) 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_boatman_distance_against_current_l668_66898


namespace NUMINAMATH_CALUDE_parallelogram_vertex_d_l668_66886

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Theorem: Given a parallelogram ABCD with vertices A(-1,-2), B(3,-1), and C(5,6), 
    the coordinates of vertex D are (1,5) -/
theorem parallelogram_vertex_d (ABCD : Parallelogram) 
    (h1 : ABCD.A = ⟨-1, -2⟩) 
    (h2 : ABCD.B = ⟨3, -1⟩) 
    (h3 : ABCD.C = ⟨5, 6⟩) : 
    ABCD.D = ⟨1, 5⟩ := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_vertex_d_l668_66886


namespace NUMINAMATH_CALUDE_solution_set_properties_inequality_properties_l668_66892

/-- The function f(x) = x² - ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 3

/-- Part I: Solution set properties -/
theorem solution_set_properties (a b : ℝ) :
  (∀ x, f a x ≤ -3 ↔ b ≤ x ∧ x ≤ 3) →
  a = 5 ∧ b = 2 :=
sorry

/-- Part II: Inequality properties -/
theorem inequality_properties (a : ℝ) :
  (∀ x, x ≥ 1/2 → f a x ≥ 1 - x^2) →
  a ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_properties_inequality_properties_l668_66892


namespace NUMINAMATH_CALUDE_upper_limit_of_b_l668_66897

theorem upper_limit_of_b (a b : ℤ) (h1 : 9 ≤ a ∧ a ≤ 14) (h2 : b ≥ 7) 
  (h3 : (14 : ℚ) / 7 - (9 : ℚ) / b = 1.55) : b ≤ 19 := by
  sorry

end NUMINAMATH_CALUDE_upper_limit_of_b_l668_66897


namespace NUMINAMATH_CALUDE_max_b_theorem_l668_66875

def is_lattice_point (x y : ℤ) : Prop := True

def line_equation (m : ℚ) (x : ℚ) : ℚ := m * x + 3

def no_lattice_points (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x ∧ x ≤ 200 → ¬(is_lattice_point x y ∧ line_equation m x = y)

def max_b : ℚ := 68 / 203

theorem max_b_theorem :
  (∀ m : ℚ, 1/3 < m → m < max_b → no_lattice_points m) ∧
  (∀ b : ℚ, b > max_b → ∃ m : ℚ, 1/3 < m ∧ m < b ∧ ¬(no_lattice_points m)) :=
sorry

end NUMINAMATH_CALUDE_max_b_theorem_l668_66875


namespace NUMINAMATH_CALUDE_inscribed_square_area_l668_66816

/-- The area of a square inscribed in the ellipse x^2/5 + y^2/10 = 1, with its diagonals parallel to the coordinate axes, is 40/3. -/
theorem inscribed_square_area (x y : ℝ) :
  (x^2 / 5 + y^2 / 10 = 1) →  -- ellipse equation
  (∃ (a : ℝ), x = a ∧ y = a) →  -- square vertices on the ellipse
  (40 : ℝ) / 3 = 4 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l668_66816


namespace NUMINAMATH_CALUDE_power_of_power_l668_66835

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l668_66835


namespace NUMINAMATH_CALUDE_y_squared_value_l668_66832

theorem y_squared_value (y : ℝ) (h : Real.sqrt (y + 16) - Real.sqrt (y - 16) = 2) : 
  y^2 = 9216 := by
sorry

end NUMINAMATH_CALUDE_y_squared_value_l668_66832


namespace NUMINAMATH_CALUDE_max_ratio_squared_l668_66854

theorem max_ratio_squared (a b c y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≥ b) (hbc : b ≥ c)
  (hy : 0 ≤ y ∧ y < a) (hz : 0 ≤ z ∧ z < c)
  (heq : a^2 + z^2 = c^2 + y^2 ∧ c^2 + y^2 = (a - y)^2 + (c - z)^2) :
  (a / c)^2 ≤ 4/3 :=
sorry

end NUMINAMATH_CALUDE_max_ratio_squared_l668_66854


namespace NUMINAMATH_CALUDE_discount_percentage_l668_66814

def coffee_cost : ℝ := 6
def cheesecake_cost : ℝ := 10
def discounted_price : ℝ := 12

theorem discount_percentage : 
  (1 - discounted_price / (coffee_cost + cheesecake_cost)) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l668_66814


namespace NUMINAMATH_CALUDE_pie_division_l668_66876

theorem pie_division (total_pie : ℚ) (people : ℕ) : 
  total_pie = 8 / 9 → people = 3 → total_pie / people = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_pie_division_l668_66876


namespace NUMINAMATH_CALUDE_solution_set_equality_l668_66853

-- Define the set S
def S : Set ℝ := {x : ℝ | |x + 3| - |x - 2| ≥ 3}

-- State the theorem
theorem solution_set_equality : S = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l668_66853


namespace NUMINAMATH_CALUDE_diagonal_bd_length_l668_66894

/-- A trapezoid with specific properties -/
structure SpecialTrapezoid where
  /-- The length of base AD -/
  ad : ℝ
  /-- The length of base BC -/
  bc : ℝ
  /-- The length of diagonal AC -/
  ac : ℝ
  /-- The circles on AB, BC, and CD as diameters intersect at one point -/
  circles_intersect : Prop

/-- The theorem about the length of diagonal BD in a special trapezoid -/
theorem diagonal_bd_length (t : SpecialTrapezoid)
    (h_ad : t.ad = 20)
    (h_bc : t.bc = 14)
    (h_ac : t.ac = 16) :
  ∃ (bd : ℝ), bd = 30 ∧ bd * bd = t.ac * t.ac + (t.ad - t.bc) * (t.ad - t.bc) / 4 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_bd_length_l668_66894


namespace NUMINAMATH_CALUDE_expression_evaluation_l668_66856

theorem expression_evaluation : 
  81 + (128 / 16) + (15 * 12) - 250 - (180 / 3)^2 = -3581 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l668_66856


namespace NUMINAMATH_CALUDE_even_sum_not_both_odd_l668_66883

theorem even_sum_not_both_odd (n m : ℤ) (h : Even (n^2 + m + n * m)) :
  ¬(Odd n ∧ Odd m) := by
  sorry

end NUMINAMATH_CALUDE_even_sum_not_both_odd_l668_66883


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l668_66852

theorem arithmetic_sequence_sum (x y z : ℤ) : 
  (x + y + z = 72) →
  (∃ (x y z : ℤ), x + y + z = 72 ∧ y - x = 1) ∧
  (∃ (x y z : ℤ), x + y + z = 72 ∧ y - x = 2) ∧
  (¬ ∃ (x y z : ℤ), x + y + z = 72 ∧ y - x = 2 ∧ Odd x) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l668_66852


namespace NUMINAMATH_CALUDE_mean_of_readings_l668_66806

def readings : List ℝ := [2, 2.1, 2, 2.2]

theorem mean_of_readings (x : ℝ) (mean : ℝ) : 
  readings.length = 4 →
  mean = (readings.sum + x) / 5 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_readings_l668_66806


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l668_66817

/-- 
Given an arithmetic sequence with first term 5, last term 50, and sum of all terms 330,
prove that the common difference is 45/11.
-/
theorem arithmetic_sequence_common_difference 
  (a₁ : ℚ) (aₙ : ℚ) (S : ℚ) (n : ℕ) (d : ℚ) :
  a₁ = 5 →
  aₙ = 50 →
  S = 330 →
  S = n / 2 * (a₁ + aₙ) →
  aₙ = a₁ + (n - 1) * d →
  d = 45 / 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l668_66817


namespace NUMINAMATH_CALUDE_geometric_sequence_101st_term_l668_66836

def geometricSequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * r^(n - 1)

theorem geometric_sequence_101st_term :
  let a := 12
  let second_term := -36
  let r := second_term / a
  let n := 101
  geometricSequence a r n = 12 * 3^100 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_101st_term_l668_66836


namespace NUMINAMATH_CALUDE_complex_number_identity_l668_66800

theorem complex_number_identity (a b c : ℂ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : c ≠ 0) 
  (h4 : a + b + c = 15) 
  (h5 : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2*a*b*c) : 
  (a^3 + b^3 + c^3) / (a*b*c) = 18 := by
sorry

end NUMINAMATH_CALUDE_complex_number_identity_l668_66800


namespace NUMINAMATH_CALUDE_moses_esther_difference_l668_66885

theorem moses_esther_difference (total : ℝ) (moses_percentage : ℝ) : 
  total = 50 ∧ moses_percentage = 0.4 → 
  let moses_share := moses_percentage * total
  let remainder := total - moses_share
  let esther_share := remainder / 2
  moses_share - esther_share = 5 := by
sorry

end NUMINAMATH_CALUDE_moses_esther_difference_l668_66885


namespace NUMINAMATH_CALUDE_marble_ratio_l668_66860

def dans_marbles : ℕ := 5
def marys_marbles : ℕ := 10

theorem marble_ratio : 
  (marys_marbles : ℚ) / (dans_marbles : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_l668_66860


namespace NUMINAMATH_CALUDE_odd_function_and_inequality_l668_66819

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - 4 / (2 * a^x + a)

theorem odd_function_and_inequality 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : ∀ x, f a x = -f a (-x)) -- odd function condition
  (h4 : ∀ x, f a x ∈ Set.univ) -- defined on (-∞, +∞)
  : 
  (a = 2) ∧ 
  (∀ t : ℝ, (∀ x ∈ Set.Ioc 0 1, t * f a x ≥ 2^x - 2) ↔ t ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_odd_function_and_inequality_l668_66819


namespace NUMINAMATH_CALUDE_green_marbles_taken_l668_66858

theorem green_marbles_taken (initial_green : ℝ) (remaining_green : ℝ) 
  (h1 : initial_green = 32.0)
  (h2 : remaining_green = 9.0) :
  initial_green - remaining_green = 23.0 := by
  sorry

end NUMINAMATH_CALUDE_green_marbles_taken_l668_66858


namespace NUMINAMATH_CALUDE_coefficient_x21_eq_932_l668_66868

open Nat BigOperators Finset

/-- The coefficient of x^21 in the expansion of (1 + x + x^2 + ... + x^20)(1 + x + x^2 + ... + x^10)^3 -/
def coefficient_x21 : ℕ :=
  (Finset.range 22).sum (λ i => i.choose 3) -
  3 * ((Finset.range 15).sum (λ i => i.choose 3)) +
  1

/-- The geometric series (1 + x + x^2 + ... + x^n) -/
def geometric_sum (n : ℕ) (x : ℝ) : ℝ :=
  (Finset.range (n + 1)).sum (λ i => x ^ i)

theorem coefficient_x21_eq_932 :
  coefficient_x21 = 932 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x21_eq_932_l668_66868


namespace NUMINAMATH_CALUDE_permutation_residue_system_bound_l668_66801

/-- A permutation of (1, 2, ..., n) -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The set {pᵢ + i | 1 ≤ i ≤ n} is a complete residue system modulo n -/
def IsSumCompleteResidue (n : ℕ) (p : Permutation n) : Prop :=
  ∀ k : Fin n, ∃ i : Fin n, (p i + i : ℕ) % n = k

/-- The set {pᵢ - i | 1 ≤ i ≤ n} is a complete residue system modulo n -/
def IsDiffCompleteResidue (n : ℕ) (p : Permutation n) : Prop :=
  ∀ k : Fin n, ∃ i : Fin n, ((p i : ℕ) - (i : ℕ) + n) % n = k

/-- Main theorem: If n satisfies the conditions, then n ≥ 4 -/
theorem permutation_residue_system_bound (n : ℕ) :
  (∃ p : Permutation n, IsSumCompleteResidue n p ∧ IsDiffCompleteResidue n p) →
  n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_permutation_residue_system_bound_l668_66801


namespace NUMINAMATH_CALUDE_oh_squared_value_l668_66813

/-- Given a triangle ABC with circumcenter O, orthocenter H, and circumradius R -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  O : ℝ × ℝ
  H : ℝ × ℝ
  R : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem states that for a triangle with R = 10 and a^2 + b^2 + c^2 = 50, OH^2 = 850 -/
theorem oh_squared_value (t : Triangle) 
  (h1 : t.R = 10) 
  (h2 : t.a^2 + t.b^2 + t.c^2 = 50) : 
  (t.O.1 - t.H.1)^2 + (t.O.2 - t.H.2)^2 = 850 := by
  sorry

end NUMINAMATH_CALUDE_oh_squared_value_l668_66813


namespace NUMINAMATH_CALUDE_car_trip_duration_l668_66845

/-- Represents the duration of a car trip with varying speeds -/
def car_trip (first_speed second_speed average_speed : ℝ) (first_duration : ℝ) : ℝ → Prop :=
  λ total_duration : ℝ =>
    let second_duration := total_duration - first_duration
    let total_distance := first_speed * first_duration + second_speed * second_duration
    (total_distance / total_duration = average_speed) ∧
    (total_duration > first_duration) ∧
    (first_duration > 0) ∧
    (second_duration > 0)

/-- Theorem stating that the car trip with given parameters lasts 7.5 hours -/
theorem car_trip_duration :
  car_trip 30 42 34 5 7.5 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_duration_l668_66845


namespace NUMINAMATH_CALUDE_pyramid_properties_l668_66805

/-- Represents a right octagonal pyramid -/
structure RightOctagonalPyramid where
  base_area : ℝ
  cross_section_area1 : ℝ
  cross_section_area2 : ℝ
  cross_section_distance : ℝ

/-- Calculates the distance of the larger cross section from the apex -/
def larger_cross_section_distance (p : RightOctagonalPyramid) : ℝ := sorry

/-- Calculates the total height of the pyramid -/
def total_height (p : RightOctagonalPyramid) : ℝ := sorry

/-- Theorem stating the properties of the specific pyramid -/
theorem pyramid_properties (p : RightOctagonalPyramid) 
  (h1 : p.base_area = 1200)
  (h2 : p.cross_section_area1 = 300 * Real.sqrt 2)
  (h3 : p.cross_section_area2 = 675 * Real.sqrt 2)
  (h4 : p.cross_section_distance = 10) :
  larger_cross_section_distance p = 30 ∧ total_height p = 40 := by sorry

end NUMINAMATH_CALUDE_pyramid_properties_l668_66805


namespace NUMINAMATH_CALUDE_marching_band_members_l668_66838

theorem marching_band_members :
  ∃! n : ℕ, 100 < n ∧ n < 200 ∧
  n % 4 = 1 ∧ n % 5 = 2 ∧ n % 7 = 3 ∧
  n = 157 := by sorry

end NUMINAMATH_CALUDE_marching_band_members_l668_66838


namespace NUMINAMATH_CALUDE_troll_ratio_l668_66826

/-- The number of trolls hiding by the path in the forest -/
def trolls_by_path : ℕ := 6

/-- The total number of trolls counted -/
def total_trolls : ℕ := 33

/-- The number of trolls hiding under the bridge -/
def trolls_under_bridge : ℕ := 18

/-- The number of trolls hiding in the plains -/
def trolls_in_plains : ℕ := trolls_under_bridge / 2

theorem troll_ratio : 
  trolls_by_path + trolls_under_bridge + trolls_in_plains = total_trolls ∧ 
  trolls_under_bridge / trolls_by_path = 3 := by
  sorry

end NUMINAMATH_CALUDE_troll_ratio_l668_66826


namespace NUMINAMATH_CALUDE_car_production_total_l668_66882

theorem car_production_total (north_america europe asia south_america : ℕ) 
  (h1 : north_america = 3884)
  (h2 : europe = 2871)
  (h3 : asia = 5273)
  (h4 : south_america = 1945) :
  north_america + europe + asia + south_america = 13973 :=
by sorry

end NUMINAMATH_CALUDE_car_production_total_l668_66882


namespace NUMINAMATH_CALUDE_pet_weights_l668_66831

/-- Given the weights of pets owned by Evan, Ivan, and Kara, prove their total weight -/
theorem pet_weights (evan_dog : ℕ) (ivan_dog : ℕ) (kara_cat : ℕ)
  (h1 : evan_dog = 63)
  (h2 : evan_dog = 7 * ivan_dog)
  (h3 : kara_cat = 5 * (evan_dog + ivan_dog)) :
  evan_dog + ivan_dog + kara_cat = 432 := by
  sorry

end NUMINAMATH_CALUDE_pet_weights_l668_66831


namespace NUMINAMATH_CALUDE_equation_satisfied_at_x_equals_4_l668_66802

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- State the theorem
theorem equation_satisfied_at_x_equals_4 :
  2 * (f 4) - 19 = f (4 - 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_satisfied_at_x_equals_4_l668_66802


namespace NUMINAMATH_CALUDE_john_savings_proof_l668_66865

/-- Calculates the monthly savings amount given the total savings period, amount spent, and remaining amount. -/
def monthly_savings (savings_period_years : ℕ) (amount_spent : ℕ) (amount_remaining : ℕ) : ℚ :=
  let total_saved : ℕ := amount_spent + amount_remaining
  let total_months : ℕ := savings_period_years * 12
  (total_saved : ℚ) / total_months

/-- Proves that given a savings period of 2 years, $400 spent, and $200 remaining, the monthly savings amount is $25. -/
theorem john_savings_proof :
  monthly_savings 2 400 200 = 25 := by
  sorry

end NUMINAMATH_CALUDE_john_savings_proof_l668_66865


namespace NUMINAMATH_CALUDE_cars_to_sell_l668_66871

/-- The number of cars each client selected -/
def cars_per_client : ℕ := 3

/-- The number of times each car was selected -/
def selections_per_car : ℕ := 3

/-- The number of clients who visited the garage -/
def num_clients : ℕ := 15

/-- The number of cars the seller has to sell -/
def num_cars : ℕ := 15

theorem cars_to_sell :
  num_cars * selections_per_car = num_clients * cars_per_client :=
by sorry

end NUMINAMATH_CALUDE_cars_to_sell_l668_66871


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l668_66844

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of our geometric sequence -/
def a : ℚ := 1/4

/-- The common ratio of our geometric sequence -/
def r : ℚ := 1/4

/-- The number of terms we're summing -/
def n : ℕ := 6

theorem geometric_sequence_sum : 
  geometricSum a r n = 4095/12288 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l668_66844


namespace NUMINAMATH_CALUDE_circle_coverage_fraction_l668_66851

/-- The fraction of a smaller circle's area not covered by a larger circle when placed inside it -/
theorem circle_coverage_fraction (dX dY : ℝ) (h_dX : dX = 16) (h_dY : dY = 18) (h_inside : dX < dY) :
  (π * (dY / 2)^2 - π * (dX / 2)^2) / (π * (dX / 2)^2) = 17 / 64 := by
  sorry

end NUMINAMATH_CALUDE_circle_coverage_fraction_l668_66851


namespace NUMINAMATH_CALUDE_combination_equality_implies_seven_l668_66884

theorem combination_equality_implies_seven (n : ℕ) : 
  (n.choose 3) = ((n-1).choose 3) + ((n-1).choose 4) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_implies_seven_l668_66884


namespace NUMINAMATH_CALUDE_shooting_probabilities_l668_66822

/-- Two shooters independently shoot at a target -/
structure ShootingScenario where
  /-- Probability of shooter A hitting the target -/
  prob_A : ℝ
  /-- Probability of shooter B hitting the target -/
  prob_B : ℝ
  /-- Assumption that probabilities are between 0 and 1 -/
  h_prob_A : 0 ≤ prob_A ∧ prob_A ≤ 1
  h_prob_B : 0 ≤ prob_B ∧ prob_B ≤ 1

/-- The probability that the target is hit in one shooting attempt -/
def prob_hit (s : ShootingScenario) : ℝ :=
  s.prob_A + s.prob_B - s.prob_A * s.prob_B

/-- The probability that the target is hit exactly by shooter A -/
def prob_hit_A (s : ShootingScenario) : ℝ :=
  s.prob_A * (1 - s.prob_B)

theorem shooting_probabilities (s : ShootingScenario) 
  (h_A : s.prob_A = 0.95) (h_B : s.prob_B = 0.9) : 
  prob_hit s = 0.995 ∧ prob_hit_A s = 0.095 := by
  sorry

#eval prob_hit ⟨0.95, 0.9, by norm_num, by norm_num⟩
#eval prob_hit_A ⟨0.95, 0.9, by norm_num, by norm_num⟩

end NUMINAMATH_CALUDE_shooting_probabilities_l668_66822


namespace NUMINAMATH_CALUDE_prime_conditions_theorem_l668_66808

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def satisfies_conditions (A : ℕ) : Prop :=
  is_prime A ∧
  A < 100 ∧
  is_prime (A + 10) ∧
  is_prime (A - 20) ∧
  is_prime (A + 30) ∧
  is_prime (A + 60) ∧
  is_prime (A + 70)

theorem prime_conditions_theorem :
  ∀ A : ℕ, satisfies_conditions A ↔ (A = 37 ∨ A = 43 ∨ A = 79) :=
by sorry

end NUMINAMATH_CALUDE_prime_conditions_theorem_l668_66808


namespace NUMINAMATH_CALUDE_small_tile_position_l668_66896

/-- Represents a tile on the grid -/
inductive Tile
| Small : Tile  -- 1x1 tile
| Large : Tile  -- 1x3 tile

/-- Represents a position on the 7x7 grid -/
structure Position :=
  (row : Fin 7)
  (col : Fin 7)

/-- Checks if a position is on the border of the grid -/
def is_border (p : Position) : Prop :=
  p.row = 0 ∨ p.row = 6 ∨ p.col = 0 ∨ p.col = 6

/-- Checks if a position is in the center of the grid -/
def is_center (p : Position) : Prop :=
  p.row = 3 ∧ p.col = 3

/-- Represents the state of the grid -/
structure GridState :=
  (tiles : List (Tile × Position))
  (small_tile_count : Nat)
  (large_tile_count : Nat)

/-- Checks if a GridState is valid according to the problem conditions -/
def is_valid_state (state : GridState) : Prop :=
  state.small_tile_count = 1 ∧
  state.large_tile_count = 16 ∧
  state.tiles.length = 17

/-- The main theorem to prove -/
theorem small_tile_position (state : GridState) :
  is_valid_state state →
  ∃ (p : Position), (Tile.Small, p) ∈ state.tiles ∧ (is_border p ∨ is_center p) :=
sorry

end NUMINAMATH_CALUDE_small_tile_position_l668_66896


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l668_66828

theorem consecutive_integers_problem (a b c : ℕ) : 
  a.succ = b → b.succ = c → 
  a > 0 → b > 0 → c > 0 → 
  a^2 = 97344 → c^2 = 98596 → 
  b = 313 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l668_66828


namespace NUMINAMATH_CALUDE_chord_length_in_isosceles_trapezoid_l668_66843

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : Bool
  /-- The circle is inscribed in the trapezoid -/
  isInscribed : Bool

/-- The theorem stating the length of the chord connecting the tangent points -/
theorem chord_length_in_isosceles_trapezoid 
  (t : IsoscelesTrapezoidWithInscribedCircle) 
  (h1 : t.r = 3)
  (h2 : t.area = 48)
  (h3 : t.isIsosceles = true)
  (h4 : t.isInscribed = true) :
  ∃ (chord_length : ℝ), chord_length = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_in_isosceles_trapezoid_l668_66843


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_l668_66818

open Set

-- Define the universe set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 < 4}

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | -2 < x ∧ x ≤ 3} := by sorry

-- Theorem for the complement of A with respect to U
theorem complement_A : (U \ A) = {x : ℝ | x < -1 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_l668_66818


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l668_66829

theorem theater_ticket_sales (total_tickets : ℕ) (advanced_price door_price : ℚ) (total_revenue : ℚ) 
  (h1 : total_tickets = 800)
  (h2 : advanced_price = 14.5)
  (h3 : door_price = 22)
  (h4 : total_revenue = 16640) :
  ∃ (door_tickets : ℕ), 
    door_tickets = 672 ∧ 
    (total_tickets - door_tickets) * advanced_price + door_tickets * door_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l668_66829


namespace NUMINAMATH_CALUDE_binomial_p_value_l668_66840

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial random variable -/
def expected_value (X : BinomialRV) : ℝ := X.n * X.p

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_p_value (X : BinomialRV) 
  (h2 : expected_value X = 30)
  (h3 : variance X = 20) : 
  X.p = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_p_value_l668_66840


namespace NUMINAMATH_CALUDE_translation_coordinates_l668_66847

/-- Given a point A(-1, 2) in the Cartesian coordinate system,
    translated 4 units to the right and 2 units down to obtain point A₁,
    the coordinates of A₁ are (3, 0). -/
theorem translation_coordinates :
  let A : ℝ × ℝ := (-1, 2)
  let right_translation : ℝ := 4
  let down_translation : ℝ := 2
  let A₁ : ℝ × ℝ := (A.1 + right_translation, A.2 - down_translation)
  A₁ = (3, 0) := by
sorry

end NUMINAMATH_CALUDE_translation_coordinates_l668_66847


namespace NUMINAMATH_CALUDE_inequality_proof_l668_66861

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * (a + b) + b * c * (b + c) + a * c * (a + c) - 6 * a * b * c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l668_66861


namespace NUMINAMATH_CALUDE_robins_haircut_l668_66888

/-- Given Robin's initial hair length and current hair length, 
    prove the amount of hair cut is the difference between these lengths. -/
theorem robins_haircut (initial_length current_length : ℕ) 
  (h1 : initial_length = 17)
  (h2 : current_length = 13) :
  initial_length - current_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_robins_haircut_l668_66888


namespace NUMINAMATH_CALUDE_max_imaginary_part_at_84_degrees_l668_66862

/-- The polynomial whose roots we're investigating -/
def f (z : ℂ) : ℂ := z^12 - z^9 + z^6 - z^3 + 1

/-- The set of roots of the polynomial -/
def roots : Set ℂ := {z : ℂ | f z = 0}

/-- The set of angles corresponding to the roots -/
def root_angles : Set Real := {θ : Real | ∃ z ∈ roots, z = Complex.exp (θ * Complex.I)}

/-- The theorem stating the maximum imaginary part occurs at 84 degrees -/
theorem max_imaginary_part_at_84_degrees :
  ∃ θ ∈ root_angles,
    θ * Real.pi / 180 = 84 * Real.pi / 180 ∧
    ∀ φ ∈ root_angles, -Real.pi/2 ≤ φ ∧ φ ≤ Real.pi/2 →
      Complex.abs (Complex.sin (Complex.ofReal φ)) ≤ 
      Complex.abs (Complex.sin (Complex.ofReal (θ * Real.pi / 180))) :=
sorry

end NUMINAMATH_CALUDE_max_imaginary_part_at_84_degrees_l668_66862


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l668_66864

theorem sin_2alpha_value (α : Real) 
  (h : (1 - Real.tan α) / (1 + Real.tan α) = 3 - 2 * Real.sqrt 2) : 
  Real.sin (2 * α) = (2 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l668_66864


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l668_66837

/-- A quadratic function f(x) = x^2 + (k+2)x + k + 5 -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 + (k+2)*x + k + 5

theorem quadratic_roots_condition (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ f k x₁ = 0 ∧ f k x₂ = 0) →
  -5 < k ∧ k < -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l668_66837


namespace NUMINAMATH_CALUDE_cubic_roots_arithmetic_imply_p_eq_two_l668_66850

/-- A cubic polynomial with coefficient p -/
def cubic_poly (p : ℝ) (x : ℝ) : ℝ := x^3 - 6*p*x^2 + 5*p*x + 88

/-- The roots of the cubic polynomial form an arithmetic sequence -/
def roots_form_arithmetic_sequence (p : ℝ) : Prop :=
  ∃ (a d : ℝ), Set.range (λ i : Fin 3 => a + i.val * d) = {x | cubic_poly p x = 0}

/-- If the roots of x³ - 6px² + 5px + 88 = 0 form an arithmetic sequence, then p = 2 -/
theorem cubic_roots_arithmetic_imply_p_eq_two :
  ∀ p : ℝ, roots_form_arithmetic_sequence p → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_arithmetic_imply_p_eq_two_l668_66850


namespace NUMINAMATH_CALUDE_orange_harvest_difference_l668_66857

theorem orange_harvest_difference (ripe_sacks unripe_sacks : ℕ) 
  (h1 : ripe_sacks = 44) 
  (h2 : unripe_sacks = 25) : 
  ripe_sacks - unripe_sacks = 19 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_difference_l668_66857


namespace NUMINAMATH_CALUDE_arrangements_with_constraints_l668_66820

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def adjacent_arrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

def doubly_adjacent_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 2)

theorem arrangements_with_constraints (n : ℕ) (h : n = 5) : 
  total_arrangements n - 2 * adjacent_arrangements n + doubly_adjacent_arrangements n = 36 := by
  sorry

#check arrangements_with_constraints

end NUMINAMATH_CALUDE_arrangements_with_constraints_l668_66820


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l668_66881

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the set B
def B : Set ℝ := {1, 2}

-- Theorem statement
theorem intersection_of_A_and_B 
  (A : Set ℝ) 
  (h1 : ∀ y ∈ B, ∃ x ∈ A, f x = y) :
  (A ∩ B = ∅) ∨ (A ∩ B = {1}) :=
sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l668_66881


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l668_66889

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 75 →
  E = 4 * F + 30 →
  D + E + F = 180 →
  F = 15 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l668_66889
