import Mathlib

namespace tan_315_degrees_l2077_207778

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end tan_315_degrees_l2077_207778


namespace sin_graph_translation_l2077_207746

open Real

theorem sin_graph_translation (a : ℝ) (h1 : 0 < a) (h2 : a < π) :
  (∀ x, sin (2 * (x - a) + π / 3) = sin (2 * x)) → a = π / 6 := by
  sorry

end sin_graph_translation_l2077_207746


namespace contrapositive_equivalence_sufficient_not_necessary_negation_equivalence_disjunction_not_both_true_l2077_207753

-- 1. Contrapositive
theorem contrapositive_equivalence :
  (∀ x : ℝ, x ≠ 2 → x^2 - 5*x + 6 ≠ 0) ↔
  (∀ x : ℝ, x^2 - 5*x + 6 = 0 → x = 2) := by sorry

-- 2. Sufficient but not necessary condition
theorem sufficient_not_necessary :
  (∀ x : ℝ, x < 1 → x^2 - 3*x + 2 > 0) ∧
  (∃ x : ℝ, x^2 - 3*x + 2 > 0 ∧ x ≥ 1) := by sorry

-- 3. Negation of universal quantifier
theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≠ 0) ↔
  (∃ x : ℝ, x^2 + x + 1 = 0) := by sorry

-- 4. Disjunction does not imply both true
theorem disjunction_not_both_true :
  ∃ (p q : Prop), (p ∨ q) ∧ ¬(p ∧ q) := by sorry

end contrapositive_equivalence_sufficient_not_necessary_negation_equivalence_disjunction_not_both_true_l2077_207753


namespace hyperbola_asymptote_angle_l2077_207783

/-- Proves that for a hyperbola x²/a² - y²/b² = 1 with a > b, 
    if the angle between its asymptotes is 45°, then a/b = 1 + √2 -/
theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  (Real.pi / 4 = Real.arctan ((b/a - (-b/a)) / (1 + (b/a) * (-b/a)))) →
  a / b = 1 + Real.sqrt 2 := by
sorry

end hyperbola_asymptote_angle_l2077_207783


namespace base_conversion_sum_fraction_l2077_207767

/-- Given that 546 in base 7 is equal to xy9 in base 10, where x and y are single digits,
    prove that (x + y + 9) / 21 = 6 / 7 -/
theorem base_conversion_sum_fraction :
  ∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ 
  (5 * 7^2 + 4 * 7 + 6 : ℕ) = x * 100 + y * 10 + 9 →
  (x + y + 9 : ℚ) / 21 = 6 / 7 := by
sorry

end base_conversion_sum_fraction_l2077_207767


namespace divisor_count_equality_l2077_207775

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Theorem: For all positive integers n and k, there exists a positive integer s
    such that the number of positive divisors of sn equals the number of positive divisors of sk
    if and only if n does not divide k and k does not divide n -/
theorem divisor_count_equality (n k : ℕ+) :
  (∃ s : ℕ+, num_divisors (s * n) = num_divisors (s * k)) ↔ (¬(n ∣ k) ∧ ¬(k ∣ n)) :=
sorry

end divisor_count_equality_l2077_207775


namespace constant_term_expansion_l2077_207752

theorem constant_term_expansion (a : ℝ) : 
  (∃ k : ℝ, k = -40 ∧ k = (a * 40 + 1 * 1)) → a = -1 := by
  sorry

end constant_term_expansion_l2077_207752


namespace hannah_measuring_spoons_l2077_207764

/-- The number of measuring spoons Hannah bought -/
def num_measuring_spoons : ℕ := 2

/-- The price of each cookie in dollars -/
def cookie_price : ℚ := 4/5

/-- The number of cookies sold -/
def num_cookies : ℕ := 40

/-- The price of each cupcake in dollars -/
def cupcake_price : ℚ := 2

/-- The number of cupcakes sold -/
def num_cupcakes : ℕ := 30

/-- The price of each measuring spoon in dollars -/
def spoon_price : ℚ := 13/2

/-- The amount of money left after buying measuring spoons in dollars -/
def money_left : ℚ := 79

theorem hannah_measuring_spoons :
  (num_cookies * cookie_price + num_cupcakes * cupcake_price - money_left) / spoon_price = num_measuring_spoons := by
  sorry

end hannah_measuring_spoons_l2077_207764


namespace wrench_force_problem_l2077_207739

/-- The force required to loosen a nut varies inversely with the length of the wrench handle -/
def inverse_variation (force : ℝ) (length : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ force * length = k

theorem wrench_force_problem (force₁ : ℝ) (length₁ : ℝ) (force₂ : ℝ) (length₂ : ℝ) :
  inverse_variation force₁ length₁ →
  inverse_variation force₂ length₂ →
  force₁ = 300 →
  length₁ = 12 →
  length₂ = 18 →
  force₂ = 200 := by
  sorry

end wrench_force_problem_l2077_207739


namespace final_paycheck_amount_l2077_207731

def biweekly_gross_pay : ℝ := 1120
def retirement_rate : ℝ := 0.25
def tax_deduction : ℝ := 100

theorem final_paycheck_amount :
  biweekly_gross_pay * (1 - retirement_rate) - tax_deduction = 740 := by
  sorry

end final_paycheck_amount_l2077_207731


namespace total_is_99_l2077_207779

/-- The total number of ducks and ducklings in a flock --/
def total_ducks_and_ducklings : ℕ → ℕ → ℕ → ℕ := fun a b c => 
  (2 + 6 + 9) + (2 * a + 6 * b + 9 * c)

/-- Theorem: The total number of ducks and ducklings is 99 --/
theorem total_is_99 : total_ducks_and_ducklings 5 3 6 = 99 := by
  sorry

end total_is_99_l2077_207779


namespace deposit_percentage_l2077_207704

def deposit : ℝ := 120
def remaining : ℝ := 1080

theorem deposit_percentage :
  (deposit / (deposit + remaining)) * 100 = 10 := by sorry

end deposit_percentage_l2077_207704


namespace consecutive_color_groups_probability_l2077_207763

-- Define the number of pencils of each color
def green_pencils : ℕ := 4
def orange_pencils : ℕ := 3
def blue_pencils : ℕ := 5

-- Define the total number of pencils
def total_pencils : ℕ := green_pencils + orange_pencils + blue_pencils

-- Define the probability of the specific selection
def probability_consecutive_color_groups : ℚ :=
  (Nat.factorial 3 * Nat.factorial green_pencils * Nat.factorial orange_pencils * Nat.factorial blue_pencils) /
  Nat.factorial total_pencils

-- Theorem statement
theorem consecutive_color_groups_probability :
  probability_consecutive_color_groups = 1 / 4620 :=
sorry

end consecutive_color_groups_probability_l2077_207763


namespace percentage_calculation_l2077_207713

theorem percentage_calculation (whole : ℝ) (part : ℝ) :
  whole = 475.25 →
  part = 129.89 →
  (part / whole) * 100 = 27.33 :=
by sorry

end percentage_calculation_l2077_207713


namespace multiplication_addition_difference_l2077_207757

theorem multiplication_addition_difference : 
  (2 : ℚ) / 3 * (3 : ℚ) / 2 - ((2 : ℚ) / 3 + (3 : ℚ) / 2) = -(7 : ℚ) / 6 :=
by sorry

end multiplication_addition_difference_l2077_207757


namespace hyperbola_eccentricity_l2077_207715

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptote y = x -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b = a) : 
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 2 := by
  sorry

end hyperbola_eccentricity_l2077_207715


namespace length_of_BC_l2077_207786

-- Define the parabola
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) (b c : ℝ) : Prop :=
  t.A = (0, 0) ∧
  t.B = (-b, parabola (-b)) ∧
  t.C = (c, parabola c) ∧
  b > 0 ∧
  c > 0 ∧
  t.B.2 = t.C.2 ∧  -- BC is parallel to x-axis
  (1/2 * (c + b) * (parabola (-b))) = 96  -- Area of the triangle is 96

-- Theorem to prove
theorem length_of_BC (t : Triangle) (b c : ℝ) 
  (h : triangle_conditions t b c) : 
  (t.C.1 - t.B.1) = 59/9 := by sorry

end length_of_BC_l2077_207786


namespace locust_jump_equivalence_l2077_207780

/-- A type representing the position of a locust on a line -/
def Position := ℝ

/-- A type representing a configuration of locusts -/
def Configuration := List Position

/-- A function that represents a jump to the right -/
def jumpRight (config : Configuration) (i j : ℕ) : Configuration :=
  sorry

/-- A function that represents a jump to the left -/
def jumpLeft (config : Configuration) (i j : ℕ) : Configuration :=
  sorry

/-- A predicate that checks if all locusts are 1 unit apart -/
def isUnitApart (config : Configuration) : Prop :=
  sorry

theorem locust_jump_equivalence (initial : Configuration) 
  (h : ∃ (final : Configuration), (∀ i j, jumpRight initial i j = final) ∧ isUnitApart final) :
  ∃ (final : Configuration), (∀ i j, jumpLeft initial i j = final) ∧ isUnitApart final :=
sorry

end locust_jump_equivalence_l2077_207780


namespace candidate_vote_percentage_l2077_207769

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (loss_margin : ℕ) 
  (candidate_percentage : ℚ) :
  total_votes = 4400 →
  loss_margin = 1760 →
  candidate_percentage = total_votes.cast⁻¹ * (total_votes - loss_margin) / 2 →
  candidate_percentage = 30 / 100 :=
by sorry

end candidate_vote_percentage_l2077_207769


namespace peters_to_amandas_flower_ratio_l2077_207795

theorem peters_to_amandas_flower_ratio : 
  ∀ (amanda_flowers peter_flowers peter_flowers_after : ℕ),
    amanda_flowers = 20 →
    peter_flowers = peter_flowers_after + 15 →
    peter_flowers_after = 45 →
    peter_flowers = 3 * amanda_flowers :=
by
  sorry

end peters_to_amandas_flower_ratio_l2077_207795


namespace probability_theorem_l2077_207760

/-- The probability that the straight-line distance between two randomly chosen points
    on the sides of a square with side length 2 is at least 1 -/
def probability_distance_at_least_one (S : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- A square with side length 2 -/
def square_side_two : Set (ℝ × ℝ) :=
  sorry

theorem probability_theorem :
  probability_distance_at_least_one square_side_two = (26 - Real.pi) / 32 := by
  sorry

end probability_theorem_l2077_207760


namespace water_jars_count_l2077_207788

/-- Represents the number of jars of each size -/
def num_jars : ℕ := 4

/-- Represents the total volume of water in gallons -/
def total_water : ℕ := 7

/-- Represents the volume of water in quarts -/
def water_in_quarts : ℕ := total_water * 4

theorem water_jars_count :
  num_jars * 3 = 12 ∧
  num_jars * (1 + 2 + 4) = water_in_quarts :=
by sorry

#check water_jars_count

end water_jars_count_l2077_207788


namespace hundred_guests_at_reunions_l2077_207755

/-- The number of guests attending at least one of two reunions -/
def guests_at_reunions (oates_guests yellow_guests both_guests : ℕ) : ℕ :=
  oates_guests + yellow_guests - both_guests

/-- Theorem: Given the conditions of the problem, 100 guests attend at least one reunion -/
theorem hundred_guests_at_reunions :
  let oates_guests : ℕ := 42
  let yellow_guests : ℕ := 65
  let both_guests : ℕ := 7
  guests_at_reunions oates_guests yellow_guests both_guests = 100 := by
  sorry

end hundred_guests_at_reunions_l2077_207755


namespace combined_shoe_size_l2077_207720

-- Define Jasmine's shoe size
def jasmine_size : ℕ := 7

-- Define the relationship between Alexa's and Jasmine's shoe sizes
def alexa_size : ℕ := 2 * jasmine_size

-- Define the combined shoe size
def combined_size : ℕ := jasmine_size + alexa_size

-- Theorem to prove
theorem combined_shoe_size : combined_size = 21 := by
  sorry

end combined_shoe_size_l2077_207720


namespace bumper_car_line_after_three_rounds_l2077_207762

def bumper_car_line (initial_people : ℕ) (capacity : ℕ) (leave_once : ℕ) (priority_join : ℕ) (rounds : ℕ) : ℕ :=
  let first_round := initial_people - capacity - leave_once + priority_join
  let subsequent_rounds := first_round - (rounds - 1) * capacity + (rounds - 1) * priority_join
  subsequent_rounds

theorem bumper_car_line_after_three_rounds :
  bumper_car_line 30 5 10 5 3 = 20 := by sorry

end bumper_car_line_after_three_rounds_l2077_207762


namespace dog_grooming_time_l2077_207730

theorem dog_grooming_time :
  let short_hair_time : ℕ := 10 -- Time to dry a short-haired dog in minutes
  let full_hair_time : ℕ := 2 * short_hair_time -- Time to dry a full-haired dog
  let short_hair_count : ℕ := 6 -- Number of short-haired dogs
  let full_hair_count : ℕ := 9 -- Number of full-haired dogs
  let total_time : ℕ := short_hair_time * short_hair_count + full_hair_time * full_hair_count
  total_time / 60 = 4 -- Total time in hours
  := by sorry

end dog_grooming_time_l2077_207730


namespace roots_sum_bound_l2077_207717

theorem roots_sum_bound (u v : ℂ) : 
  u ≠ v → 
  u^2023 = 1 → 
  v^2023 = 1 → 
  Complex.abs (u + v) < Real.sqrt (2 + Real.sqrt 5) := by
sorry

end roots_sum_bound_l2077_207717


namespace candy_bar_weight_reduction_l2077_207712

/-- Represents the change in weight and price of a candy bar -/
structure CandyBar where
  original_weight : ℝ
  new_weight : ℝ
  price : ℝ
  price_per_ounce_increase : ℝ

/-- The theorem stating the relationship between weight reduction and price per ounce increase -/
theorem candy_bar_weight_reduction (c : CandyBar) 
  (h1 : c.price_per_ounce_increase = 2/3)
  (h2 : c.price > 0)
  (h3 : c.original_weight > 0)
  (h4 : c.new_weight > 0)
  (h5 : c.new_weight < c.original_weight) :
  (c.original_weight - c.new_weight) / c.original_weight = 0.4 := by
sorry

end candy_bar_weight_reduction_l2077_207712


namespace unique_triple_solution_l2077_207702

theorem unique_triple_solution (a b c : ℝ) : 
  a > 5 ∧ b > 5 ∧ c > 5 ∧
  ((a + 3)^2 / (b + c - 5) + (b + 5)^2 / (c + a - 7) + (c + 7)^2 / (a + b - 9) = 49) →
  a = 13 ∧ b = 9 ∧ c = 6 := by
  sorry

end unique_triple_solution_l2077_207702


namespace arccos_one_over_sqrt_two_l2077_207792

theorem arccos_one_over_sqrt_two (π : Real) :
  Real.arccos (1 / Real.sqrt 2) = π / 4 := by sorry

end arccos_one_over_sqrt_two_l2077_207792


namespace circle_point_selection_eq_258_l2077_207716

/-- The number of ways to select 8 points from 24 equally spaced points on a circle,
    such that no two selected points have an arc length of 3 or 8 between them. -/
def circle_point_selection : ℕ :=
  2^8 + 2

/-- Proves that the number of valid selections is 258. -/
theorem circle_point_selection_eq_258 : circle_point_selection = 258 := by
  sorry

end circle_point_selection_eq_258_l2077_207716


namespace nth_equation_proof_l2077_207770

theorem nth_equation_proof (n : ℕ) : 
  n^2 + (n+1)^2 = (n*(n+1)+1)^2 - (n*(n+1))^2 := by
  sorry

end nth_equation_proof_l2077_207770


namespace business_value_l2077_207785

/-- Proves the value of a business given partial ownership and sale information -/
theorem business_value (
  total_shares : ℚ)
  (owner_share : ℚ)
  (sold_fraction : ℚ)
  (sale_price : ℚ)
  (h1 : owner_share = 1 / 3)
  (h2 : sold_fraction = 3 / 5)
  (h3 : sale_price = 15000) :
  total_shares = 75000 := by
  sorry

end business_value_l2077_207785


namespace sum_is_composite_l2077_207705

theorem sum_is_composite (a b c d : ℕ+) 
  (h : a^2 + b^2 + a*b = c^2 + d^2 + c*d) : 
  ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ (a : ℕ) + b + c + d = k * m :=
sorry

end sum_is_composite_l2077_207705


namespace cards_distribution_l2077_207772

theorem cards_distribution (total_cards : ℕ) (num_people : ℕ) (h1 : total_cards = 60) (h2 : num_people = 9) :
  let cards_per_person : ℕ := total_cards / num_people
  let remaining_cards : ℕ := total_cards % num_people
  let people_with_extra : ℕ := remaining_cards
  (num_people - people_with_extra) = 3 :=
by sorry

end cards_distribution_l2077_207772


namespace four_integers_with_many_divisors_l2077_207726

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem four_integers_with_many_divisors :
  ∃ (a b c d : ℕ),
    a > 0 ∧ a ≤ 70000 ∧ count_divisors a > 100 ∧
    b > 0 ∧ b ≤ 70000 ∧ count_divisors b > 100 ∧
    c > 0 ∧ c ≤ 70000 ∧ count_divisors c > 100 ∧
    d > 0 ∧ d ≤ 70000 ∧ count_divisors d > 100 :=
by
  use 69300, 50400, 60480, 55440
  sorry

end four_integers_with_many_divisors_l2077_207726


namespace stamp_book_gcd_l2077_207773

theorem stamp_book_gcd : Nat.gcd (Nat.gcd 1260 1470) 1890 = 210 := by
  sorry

end stamp_book_gcd_l2077_207773


namespace star_difference_l2077_207744

def star (x y : ℤ) : ℤ := x * y - 3 * x

theorem star_difference : (star 6 2) - (star 2 6) = -12 := by
  sorry

end star_difference_l2077_207744


namespace vertical_line_no_slope_l2077_207793

/-- A line parallel to the y-axis has no defined slope -/
theorem vertical_line_no_slope (a : ℝ) : 
  ¬ ∃ (m : ℝ), ∀ (x y : ℝ), x = a → (∀ ε > 0, ∃ δ > 0, ∀ x' y', |x' - x| < δ → |y' - y| < ε * |x' - x|) :=
by
  sorry

end vertical_line_no_slope_l2077_207793


namespace kyle_age_l2077_207791

/-- Given the ages of several people and their relationships, prove Kyle's age --/
theorem kyle_age (david sandra casey fiona julian shelley kyle frederick tyson : ℕ) 
  (h1 : shelley = kyle - 3)
  (h2 : shelley = julian + 4)
  (h3 : julian = frederick - 20)
  (h4 : julian = fiona + 5)
  (h5 : frederick = 2 * tyson)
  (h6 : tyson = 2 * casey)
  (h7 : casey = fiona - 2)
  (h8 : 2 * casey = sandra)
  (h9 : sandra = david + 4)
  (h10 : david = 16) : 
  kyle = 23 := by sorry

end kyle_age_l2077_207791


namespace range_of_a_l2077_207756

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |1/2 * x^3 - a*x| ≤ 1) ↔ -1/2 ≤ a ∧ a ≤ 3/2 := by
  sorry

end range_of_a_l2077_207756


namespace interior_angles_sum_l2077_207725

theorem interior_angles_sum (n : ℕ) : 
  (180 * (n - 2) = 3240) → (180 * ((n + 4) - 2) = 3960) := by
  sorry

end interior_angles_sum_l2077_207725


namespace cord_lengths_l2077_207743

theorem cord_lengths (total_length : ℝ) (a b c : ℝ) : 
  total_length = 60 → -- Total length is 60 decimeters
  a + b + c = total_length * 10 → -- Sum of parts equals total length in cm
  b = a + 1 → -- Second part is 1 cm more than first
  c = b + 1 → -- Third part is 1 cm more than second
  (a, b, c) = (199, 200, 201) := by sorry

end cord_lengths_l2077_207743


namespace perpendicular_vectors_l2077_207784

-- Define the vectors a and b
def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (6, -4)

-- Define the dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the scalar multiplication for 2D vectors
def scalar_mult (t : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (t * v.1, t * v.2)

-- Define vector addition for 2D vectors
def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Theorem statement
theorem perpendicular_vectors (t : ℝ) : 
  dot_product a (vector_add (scalar_mult t a) b) = 0 → t = -5 := by sorry

end perpendicular_vectors_l2077_207784


namespace hyperbola_equation_l2077_207782

/-- Given a hyperbola with asymptote x + √3y = 0 and one focus at (4, 0),
    its standard equation is x²/12 - y²/4 = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧
    (x + Real.sqrt 3 * y = 0 → y = -(1 / Real.sqrt 3) * x) ∧  -- Asymptote condition
    c = 4 ∧                                                   -- Focus condition
    c^2 = a^2 + b^2 ∧                                         -- Hyperbola property
    b/a = Real.sqrt 3 / 3) →                                  -- Derived from asymptote
  x^2 / 12 - y^2 / 4 = 1 :=
by sorry

end hyperbola_equation_l2077_207782


namespace complex_number_modulus_l2077_207729

theorem complex_number_modulus (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 3)
  (h3 : Complex.abs (z - w) = 1) :
  Complex.abs z = 4 * Real.sqrt 7 := by
  sorry

end complex_number_modulus_l2077_207729


namespace melissa_points_per_game_l2077_207708

/-- Given Melissa's game scoring information, calculate her points per game without bonus -/
theorem melissa_points_per_game (bonus_per_game : ℕ) (total_points : ℕ) (num_games : ℕ) 
  (h1 : bonus_per_game = 82)
  (h2 : total_points = 15089)
  (h3 : num_games = 79) :
  (total_points - bonus_per_game * num_games) / num_games = 109 := by
  sorry

end melissa_points_per_game_l2077_207708


namespace max_value_of_f_l2077_207738

/-- The function f(x) = -5x^2 + 25x - 15 -/
def f (x : ℝ) : ℝ := -5 * x^2 + 25 * x - 15

/-- Theorem stating that the maximum value of f(x) is 750 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = 750 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end max_value_of_f_l2077_207738


namespace octavia_photos_count_l2077_207787

/-- Represents the number of photographs in a photography exhibition --/
structure PhotoExhibition where
  total : ℕ
  octavia_photos : ℕ
  jack_framed : ℕ
  jack_framed_octavia : ℕ
  jack_framed_others : ℕ

/-- The photography exhibition satisfies the given conditions --/
def exhibition_conditions (e : PhotoExhibition) : Prop :=
  e.jack_framed_octavia = 24 ∧
  e.jack_framed_others = 12 ∧
  e.jack_framed = e.jack_framed_octavia + e.jack_framed_others ∧
  e.total = 48 ∧
  e.total = e.octavia_photos + e.jack_framed - e.jack_framed_octavia

/-- Theorem stating that under the given conditions, Octavia took 36 photographs --/
theorem octavia_photos_count (e : PhotoExhibition) 
  (h : exhibition_conditions e) : e.octavia_photos = 36 := by
  sorry


end octavia_photos_count_l2077_207787


namespace intersection_implies_m_values_l2077_207796

theorem intersection_implies_m_values (m : ℝ) : 
  let M : Set ℝ := {4, 5, -3*m}
  let N : Set ℝ := {-9, 3}
  (M ∩ N).Nonempty → m = 3 ∨ m = -1 := by
  sorry

end intersection_implies_m_values_l2077_207796


namespace permutations_6_3_l2077_207718

/-- The number of permutations of k elements chosen from a set of n elements -/
def permutations (n k : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - k)

/-- Theorem: The number of permutations of 3 elements chosen from a set of 6 elements is 120 -/
theorem permutations_6_3 : permutations 6 3 = 120 := by
  sorry

end permutations_6_3_l2077_207718


namespace sum_of_coefficients_l2077_207751

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 3^5 - 1 := by
sorry

end sum_of_coefficients_l2077_207751


namespace restaurant_bill_proof_l2077_207741

theorem restaurant_bill_proof : 
  ∀ (total_bill : ℝ),
  (∃ (individual_share : ℝ),
    -- 9 friends initially splitting the bill equally
    individual_share = total_bill / 9 ∧ 
    -- 8 friends each paying an extra $3.00
    8 * (individual_share + 3) = total_bill) →
  total_bill = 216 := by
sorry

end restaurant_bill_proof_l2077_207741


namespace fencing_cost_calculation_l2077_207710

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (length breadth fencing_cost_per_meter : ℝ) : ℝ :=
  2 * (length + breadth) * fencing_cost_per_meter

/-- Proves that the total cost of fencing the given rectangular plot is 5300 currency units -/
theorem fencing_cost_calculation :
  let length : ℝ := 75
  let breadth : ℝ := 25
  let fencing_cost_per_meter : ℝ := 26.50
  (length = breadth + 50) →
  total_fencing_cost length breadth fencing_cost_per_meter = 5300 := by
  sorry

#eval total_fencing_cost 75 25 26.50

end fencing_cost_calculation_l2077_207710


namespace afternoon_bags_count_l2077_207703

def morning_bags : ℕ := 29
def bag_weight : ℕ := 7
def total_weight : ℕ := 322

def afternoon_bags : ℕ := (total_weight - morning_bags * bag_weight) / bag_weight

theorem afternoon_bags_count : afternoon_bags = 17 := by
  sorry

end afternoon_bags_count_l2077_207703


namespace equation_solutions_l2077_207774

theorem equation_solutions : 
  let f (x : ℝ) := (x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 3) * (x - 5) * (x - 1)
  let g (x : ℝ) := (x - 3) * (x - 7) * (x - 3)
  { x : ℝ | x ≠ 3 ∧ x ≠ 7 ∧ f x / g x = 1 } = { 3 + Real.sqrt 3, 3 + Real.sqrt 5, 3 - Real.sqrt 5 } :=
by sorry

end equation_solutions_l2077_207774


namespace absolute_value_inequality_l2077_207709

theorem absolute_value_inequality (x y : ℝ) : 
  |y - 3*x| < 2*x ↔ x > 0 ∧ x < y ∧ y < 5*x :=
sorry

end absolute_value_inequality_l2077_207709


namespace inequality_proof_l2077_207706

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 := by
  sorry

end inequality_proof_l2077_207706


namespace impossible_coverage_l2077_207707

/-- Represents a rectangular paper strip -/
structure PaperStrip where
  width : ℕ
  length : ℕ

/-- Represents a cube -/
structure Cube where
  sideLength : ℕ

/-- Represents the configuration of paper strips on cube faces -/
def CubeConfiguration := Cube → List PaperStrip

/-- Checks if a configuration covers exactly three faces sharing a vertex -/
def coversThreeFaces (config : CubeConfiguration) (cube : Cube) : Prop :=
  sorry

/-- Checks if strips in a configuration overlap -/
def hasOverlap (config : CubeConfiguration) : Prop :=
  sorry

/-- Checks if a configuration leaves any gaps -/
def hasGaps (config : CubeConfiguration) (cube : Cube) : Prop :=
  sorry

/-- Main theorem: It's impossible to cover three faces of a 4x4x4 cube with 16 1x3 strips -/
theorem impossible_coverage : 
  ∀ (config : CubeConfiguration),
    let cube := Cube.mk 4
    let strips := List.replicate 16 (PaperStrip.mk 1 3)
    (coversThreeFaces config cube) → 
    (¬ hasOverlap config) → 
    (¬ hasGaps config cube) → 
    False :=
  sorry

end impossible_coverage_l2077_207707


namespace m_range_l2077_207766

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x^2 - 4*x + 3)}

-- Define set B
def B (m : ℝ) : Set ℝ := {y | ∃ x ∈ (Set.univ \ A), y = x + m/x ∧ m > 0}

-- Theorem statement
theorem m_range (m : ℝ) : (2 * Real.sqrt m ∈ B m) ↔ (1 < m ∧ m < 9) := by
  sorry

end m_range_l2077_207766


namespace quadratic_expression_equals_64_l2077_207700

theorem quadratic_expression_equals_64 (x : ℝ) : 
  (2*x + 3)^2 + 2*(2*x + 3)*(5 - 2*x) + (5 - 2*x)^2 = 64 := by
  sorry

end quadratic_expression_equals_64_l2077_207700


namespace small_pizza_has_four_slices_l2077_207758

/-- The number of slices in a small pizza -/
def small_pizza_slices : ℕ := sorry

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := 8

/-- The number of small pizzas purchased -/
def small_pizzas_bought : ℕ := 3

/-- The number of large pizzas purchased -/
def large_pizzas_bought : ℕ := 2

/-- The number of slices George eats -/
def george_slices : ℕ := 3

/-- The number of slices Bob eats -/
def bob_slices : ℕ := george_slices + 1

/-- The number of slices Susie eats -/
def susie_slices : ℕ := bob_slices / 2

/-- The number of slices Bill eats -/
def bill_slices : ℕ := 3

/-- The number of slices Fred eats -/
def fred_slices : ℕ := 3

/-- The number of slices Mark eats -/
def mark_slices : ℕ := 3

/-- The number of slices left over -/
def leftover_slices : ℕ := 10

theorem small_pizza_has_four_slices : small_pizza_slices = 4 := by
  sorry

end small_pizza_has_four_slices_l2077_207758


namespace expression_evaluation_l2077_207745

theorem expression_evaluation :
  let a : ℝ := 1
  let b : ℝ := 2
  2*(a+b)*(a-b) - (a+b)^2 + a*(2*a+b) = -11 := by
sorry

end expression_evaluation_l2077_207745


namespace discount_problem_l2077_207740

/-- Given a purchase with a 25% discount where the discount amount is $40, 
    prove that the total amount paid is $120. -/
theorem discount_problem (original_price : ℝ) (discount_rate : ℝ) (discount_amount : ℝ) (total_paid : ℝ) : 
  discount_rate = 0.25 →
  discount_amount = 40 →
  discount_amount = discount_rate * original_price →
  total_paid = original_price - discount_amount →
  total_paid = 120 := by
sorry

end discount_problem_l2077_207740


namespace debt_doubling_time_l2077_207711

theorem debt_doubling_time (interest_rate : ℝ) (doubling_factor : ℝ) : 
  interest_rate = 0.06 → doubling_factor = 2 → 
  (∀ t : ℕ, t < 12 → (1 + interest_rate)^t ≤ doubling_factor) ∧ 
  (1 + interest_rate)^12 > doubling_factor := by
  sorry

end debt_doubling_time_l2077_207711


namespace symmetric_points_sum_l2077_207781

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are equal -/
def symmetricYAxis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem symmetric_points_sum (a b : ℝ) :
  symmetricYAxis (a, 3) (4, b) → a + b = -1 := by
  sorry

end symmetric_points_sum_l2077_207781


namespace smallest_with_twelve_divisors_l2077_207732

/-- A function that returns the number of positive integer divisors of a given positive integer -/
def numberOfDivisors (n : ℕ+) : ℕ := sorry

/-- A function that checks if a given positive integer has exactly 12 positive integer divisors -/
def hasTwelveDivisors (n : ℕ+) : Prop :=
  numberOfDivisors n = 12

/-- Theorem stating that 108 is the smallest positive integer with exactly 12 positive integer divisors -/
theorem smallest_with_twelve_divisors :
  (∀ m : ℕ+, m < 108 → ¬(hasTwelveDivisors m)) ∧ hasTwelveDivisors 108 := by
  sorry

end smallest_with_twelve_divisors_l2077_207732


namespace odd_function_property_l2077_207789

/-- Given a function f(x) = x^5 + ax^3 + bx, where a and b are real constants,
    if f(-2) = 10, then f(2) = -10 -/
theorem odd_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^5 + a*x^3 + b*x
  f (-2) = 10 → f 2 = -10 := by
sorry

end odd_function_property_l2077_207789


namespace equation_one_integral_root_l2077_207701

/-- The equation has exactly one integral root -/
theorem equation_one_integral_root :
  ∃! (x : ℤ), x - 8 / (x - 3) = 2 - 8 / (x - 3) :=
by
  sorry

end equation_one_integral_root_l2077_207701


namespace remainder_of_sum_times_three_div_six_l2077_207714

/-- The sum of an arithmetic sequence with first term a, common difference d, and n terms -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- The number of terms in the arithmetic sequence with first term 2, common difference 6, and last term 266 -/
def n : ℕ := 45

/-- The first term of the arithmetic sequence -/
def a : ℕ := 2

/-- The common difference of the arithmetic sequence -/
def d : ℕ := 6

/-- The last term of the arithmetic sequence -/
def last_term : ℕ := 266

theorem remainder_of_sum_times_three_div_six :
  (3 * arithmetic_sum a d n) % 6 = 0 :=
sorry

end remainder_of_sum_times_three_div_six_l2077_207714


namespace gcd_12547_23791_l2077_207798

theorem gcd_12547_23791 : Nat.gcd 12547 23791 = 1 := by
  sorry

end gcd_12547_23791_l2077_207798


namespace clock_malfunction_proof_l2077_207748

/-- Represents a time in HH:MM format -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Represents a single digit change due to malfunction -/
inductive DigitChange
  | Increase
  | Decrease
  | NoChange

/-- Applies a digit change to a number -/
def applyDigitChange (n : Nat) (change : DigitChange) : Nat :=
  match change with
  | DigitChange.Increase => (n + 1) % 10
  | DigitChange.Decrease => (n + 9) % 10
  | DigitChange.NoChange => n

/-- Applies changes to all digits of a time -/
def applyChanges (t : Time) (h1 h2 m1 m2 : DigitChange) : Time :=
  let newHours := applyDigitChange (t.hours / 10) h1 * 10 + applyDigitChange (t.hours % 10) h2
  let newMinutes := applyDigitChange (t.minutes / 10) m1 * 10 + applyDigitChange (t.minutes % 10) m2
  ⟨newHours, newMinutes, sorry⟩

theorem clock_malfunction_proof :
  ∃ (original : Time) (h1 h2 m1 m2 : DigitChange),
    applyChanges original h1 h2 m1 m2 = ⟨20, 50, sorry⟩ ∧
    original = ⟨19, 49, sorry⟩ :=
  sorry

end clock_malfunction_proof_l2077_207748


namespace concert_ticket_revenue_l2077_207750

theorem concert_ticket_revenue 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (full_price : ℕ) 
  (discounted_price : ℕ) 
  (full_price_tickets : ℕ) 
  (discounted_tickets : ℕ) :
  total_tickets = 200 →
  total_revenue = 2800 →
  discounted_price = (3 * full_price) / 4 →
  total_tickets = full_price_tickets + discounted_tickets →
  total_revenue = full_price * full_price_tickets + discounted_price * discounted_tickets →
  full_price_tickets * full_price = 680 :=
by sorry

end concert_ticket_revenue_l2077_207750


namespace unique_four_digit_number_l2077_207735

theorem unique_four_digit_number : ∃! (abcd : ℕ), 
  (1000 ≤ abcd ∧ abcd < 10000) ∧  -- 4-digit number
  (abcd % 11 = 0) ∧  -- multiple of 11
  (((abcd / 1000) * 10 + ((abcd / 100) % 10)) % 7 = 0) ∧  -- ac is multiple of 7
  ((abcd / 1000) + ((abcd / 100) % 10) + ((abcd / 10) % 10) + (abcd % 10) = (abcd % 10)^2) ∧  -- sum of digits equals square of last digit
  abcd = 3454 := by
sorry

end unique_four_digit_number_l2077_207735


namespace class_composition_unique_l2077_207727

/-- Represents a pair of numbers written by a student -/
structure Answer :=
  (classmates : Nat)
  (girls : Nat)

/-- Represents the class composition -/
structure ClassComposition :=
  (boys : Nat)
  (girls : Nat)

/-- Checks if an answer is valid given the actual class composition -/
def isValidAnswer (actual : ClassComposition) (answer : Answer) : Prop :=
  (answer.classmates = actual.boys + actual.girls - 1 ∧ 
   (answer.girls = actual.girls ∨ answer.girls = actual.girls + 4 ∨ answer.girls = actual.girls - 4)) ∨
  (answer.girls = actual.girls ∧ 
   (answer.classmates = actual.boys + actual.girls - 1 ∨ 
    answer.classmates = actual.boys + actual.girls + 3 ∨ 
    answer.classmates = actual.boys + actual.girls - 5))

theorem class_composition_unique :
  ∃! comp : ClassComposition,
    isValidAnswer comp ⟨15, 18⟩ ∧
    isValidAnswer comp ⟨15, 10⟩ ∧
    isValidAnswer comp ⟨12, 13⟩ ∧
    comp.boys = 16 ∧
    comp.girls = 14 := by sorry

end class_composition_unique_l2077_207727


namespace overall_profit_calculation_john_profit_is_50_l2077_207777

/-- Calculates the overall profit from selling two items with given costs and profit/loss percentages -/
theorem overall_profit_calculation 
  (grinder_cost mobile_cost : ℕ) 
  (grinder_loss_percent mobile_profit_percent : ℚ) : ℕ :=
  let grinder_selling_price := grinder_cost - (grinder_cost * grinder_loss_percent).floor
  let mobile_selling_price := mobile_cost + (mobile_cost * mobile_profit_percent).ceil
  let total_selling_price := grinder_selling_price + mobile_selling_price
  let total_cost := grinder_cost + mobile_cost
  (total_selling_price - total_cost).toNat

/-- Proves that given the specific costs and percentages, the overall profit is 50 -/
theorem john_profit_is_50 : 
  overall_profit_calculation 15000 8000 (5/100) (10/100) = 50 := by
  sorry

end overall_profit_calculation_john_profit_is_50_l2077_207777


namespace polynomial_transformation_l2077_207723

theorem polynomial_transformation (x y : ℝ) (hx : x ≠ 0) :
  y = x + 1/x →
  (x^4 - x^3 - 6*x^2 - x + 1 = 0) ↔ (x^2*(y^2 - y - 8) = 0) :=
by sorry

end polynomial_transformation_l2077_207723


namespace sum_of_alpha_beta_l2077_207733

/-- Given constants α and β satisfying the rational equation, prove their sum is 176 -/
theorem sum_of_alpha_beta (α β : ℝ) 
  (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 102*x + 2021) / (x^2 + 89*x - 3960)) : 
  α + β = 176 := by
  sorry

end sum_of_alpha_beta_l2077_207733


namespace quadratic_inequality_solution_l2077_207747

-- Define the quadratic function
def f (x : ℝ) := x^2 + x - 2

-- Define the solution set
def solution_set := {x : ℝ | x ≤ -2 ∨ x ≥ 1}

-- Theorem statement
theorem quadratic_inequality_solution :
  ∀ x : ℝ, f x ≥ 0 ↔ x ∈ solution_set :=
by sorry

end quadratic_inequality_solution_l2077_207747


namespace capri_sun_cost_per_pouch_l2077_207761

/-- Calculates the cost per pouch in cents given the number of boxes, pouches per box, and total cost in dollars. -/
def cost_per_pouch (boxes : ℕ) (pouches_per_box : ℕ) (total_cost_dollars : ℕ) : ℕ :=
  (total_cost_dollars * 100) / (boxes * pouches_per_box)

/-- Proves that for 10 boxes with 6 pouches each, costing $12 in total, each pouch costs 20 cents. -/
theorem capri_sun_cost_per_pouch :
  cost_per_pouch 10 6 12 = 20 := by
  sorry

end capri_sun_cost_per_pouch_l2077_207761


namespace expression_simplification_l2077_207799

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 + 4 / (x - 3)) / ((x^2 + 2*x + 1) / (2*x - 6)) = Real.sqrt 2 := by
  sorry

end expression_simplification_l2077_207799


namespace parabola_intersection_points_l2077_207719

/-- The x-coordinates of the intersection points of two parabolas -/
theorem parabola_intersection_points (x : ℝ) :
  (3 * x^2 - 4 * x + 7 = 6 * x^2 + x + 3) ↔ 
  (x = (5 + Real.sqrt 73) / -6 ∨ x = (5 - Real.sqrt 73) / -6) := by
sorry

end parabola_intersection_points_l2077_207719


namespace molecular_weight_calculation_l2077_207728

/-- The atomic weight of nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of nitrogen atoms in the compound -/
def num_N : ℕ := 2

/-- The number of oxygen atoms in the compound -/
def num_O : ℕ := 5

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := (num_N : ℝ) * atomic_weight_N + (num_O : ℝ) * atomic_weight_O

theorem molecular_weight_calculation :
  molecular_weight = 108.02 := by sorry

end molecular_weight_calculation_l2077_207728


namespace probability_three_unused_rockets_expected_targets_hit_l2077_207724

/-- Represents a rocket artillery system -/
structure RocketSystem where
  totalRockets : ℕ
  maxShotsPerTarget : ℕ
  hitProbability : ℝ

/-- Calculates the probability of having exactly 3 unused rockets after firing at 5 targets -/
def probabilityThreeUnusedRockets (system : RocketSystem) : ℝ :=
  10 * system.hitProbability^3 * (1 - system.hitProbability)^2

/-- Calculates the expected number of targets hit when firing at 9 targets -/
def expectedTargetsHit (system : RocketSystem) : ℝ :=
  10 * system.hitProbability - system.hitProbability^10

/-- Theorem stating the probability of having exactly 3 unused rockets after firing at 5 targets -/
theorem probability_three_unused_rockets 
  (system : RocketSystem) 
  (h1 : system.totalRockets = 10) 
  (h2 : system.maxShotsPerTarget = 2) :
  probabilityThreeUnusedRockets system = 10 * system.hitProbability^3 * (1 - system.hitProbability)^2 := by
  sorry

/-- Theorem stating the expected number of targets hit when firing at 9 targets -/
theorem expected_targets_hit 
  (system : RocketSystem) 
  (h1 : system.totalRockets = 10) 
  (h2 : system.maxShotsPerTarget = 2) :
  expectedTargetsHit system = 10 * system.hitProbability - system.hitProbability^10 := by
  sorry

end probability_three_unused_rockets_expected_targets_hit_l2077_207724


namespace min_workers_for_profit_is_16_l2077_207768

/-- Represents the minimum number of workers required for a manufacturing plant to make a profit -/
def min_workers_for_profit (
  maintenance_cost : ℕ)  -- Daily maintenance cost in dollars
  (hourly_wage : ℕ)      -- Hourly wage per worker in dollars
  (widgets_per_hour : ℕ) -- Number of widgets produced per worker per hour
  (widget_price : ℕ)     -- Selling price of each widget in dollars
  (work_hours : ℕ)       -- Number of work hours per day
  : ℕ :=
  16

/-- Theorem stating that given the specific conditions, the minimum number of workers for profit is 16 -/
theorem min_workers_for_profit_is_16 :
  min_workers_for_profit 600 20 4 4 10 = 16 := by
  sorry

#eval min_workers_for_profit 600 20 4 4 10

end min_workers_for_profit_is_16_l2077_207768


namespace alien_abduction_l2077_207771

theorem alien_abduction (P : ℕ) : 
  (80 : ℚ) / 100 * P + 40 = P → P = 200 := by
  sorry

end alien_abduction_l2077_207771


namespace rulers_added_l2077_207765

theorem rulers_added (initial_rulers : ℕ) (final_rulers : ℕ) (added_rulers : ℕ) : 
  initial_rulers = 46 → final_rulers = 71 → added_rulers = final_rulers - initial_rulers → 
  added_rulers = 25 := by
  sorry

end rulers_added_l2077_207765


namespace cos_75_cos_15_plus_sin_75_sin_15_l2077_207749

theorem cos_75_cos_15_plus_sin_75_sin_15 :
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) +
  Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 1 / 2 := by
  sorry

end cos_75_cos_15_plus_sin_75_sin_15_l2077_207749


namespace opposite_numbers_with_equation_l2077_207742

theorem opposite_numbers_with_equation (x y : ℝ) : 
  x + y = 0 → (x + 2)^2 - (y + 2)^2 = 4 → x = 1/2 ∧ y = -1/2 := by
  sorry

end opposite_numbers_with_equation_l2077_207742


namespace quadratic_sum_of_solutions_l2077_207737

theorem quadratic_sum_of_solutions : ∃ a b : ℝ, 
  (∀ x : ℝ, x^2 - 6*x + 11 = 25 ↔ (x = a ∨ x = b)) ∧ 
  a ≥ b ∧ 
  3*a + 2*b = 15 + Real.sqrt 92 / 2 := by
  sorry

end quadratic_sum_of_solutions_l2077_207737


namespace leftHandedJazzLoversCount_l2077_207790

/-- Represents a club with members having different characteristics -/
structure Club where
  total : ℕ
  leftHanded : ℕ
  jazzLovers : ℕ
  rightHandedNonJazz : ℕ

/-- The number of left-handed jazz lovers in the club -/
def leftHandedJazzLovers (c : Club) : ℕ :=
  c.total - (c.leftHanded + c.jazzLovers - c.rightHandedNonJazz)

/-- Theorem stating the number of left-handed jazz lovers in the given club -/
theorem leftHandedJazzLoversCount (c : Club) 
  (h1 : c.total = 20)
  (h2 : c.leftHanded = 8)
  (h3 : c.jazzLovers = 15)
  (h4 : c.rightHandedNonJazz = 2) :
  leftHandedJazzLovers c = 5 := by
  sorry

#eval leftHandedJazzLovers { total := 20, leftHanded := 8, jazzLovers := 15, rightHandedNonJazz := 2 }

end leftHandedJazzLoversCount_l2077_207790


namespace no_integer_solutions_for_P_x_eq_x_l2077_207776

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- Property: for any integers a and b, b - a divides P(b) - P(a) -/
def IntegerCoefficientProperty (P : IntPolynomial) : Prop :=
  ∀ a b : ℤ, (b - a) ∣ (P b - P a)

theorem no_integer_solutions_for_P_x_eq_x
  (P : IntPolynomial)
  (h_int_coeff : IntegerCoefficientProperty P)
  (h_P_3 : P 3 = 4)
  (h_P_4 : P 4 = 3) :
  ¬∃ x : ℤ, P x = x :=
by sorry

end no_integer_solutions_for_P_x_eq_x_l2077_207776


namespace largest_multiple_of_15_under_500_l2077_207797

theorem largest_multiple_of_15_under_500 : ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 → n ≤ 495 := by
  sorry

end largest_multiple_of_15_under_500_l2077_207797


namespace point_in_second_quadrant_implies_a_range_l2077_207722

/-- A point P(x, y) is in the second quadrant if x < 0 and y > 0 -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The x-coordinate of point P as a function of a -/
def x_coord (a : ℝ) : ℝ := 2 * a + 1

/-- The y-coordinate of point P as a function of a -/
def y_coord (a : ℝ) : ℝ := 1 - a

/-- Theorem: If P(2a+1, 1-a) is in the second quadrant, then a < -1/2 -/
theorem point_in_second_quadrant_implies_a_range (a : ℝ) :
  in_second_quadrant (x_coord a) (y_coord a) → a < -1/2 := by
  sorry

end point_in_second_quadrant_implies_a_range_l2077_207722


namespace arithmetic_seq_common_diff_l2077_207759

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  seq_def : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- 
If for an arithmetic sequence, 2S₃ = 3S₂ + 6, 
then the common difference is 2 
-/
theorem arithmetic_seq_common_diff 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : 
  seq.d = 2 := by sorry

end arithmetic_seq_common_diff_l2077_207759


namespace min_value_theorem_l2077_207721

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 3) + 2 / (y + 3) = 1 / 4) :
  2 * x + 3 * y ≥ 16 * Real.sqrt 3 - 16 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧
    1 / (x₀ + 3) + 2 / (y₀ + 3) = 1 / 4 ∧
    2 * x₀ + 3 * y₀ = 16 * Real.sqrt 3 - 16 :=
by sorry

end min_value_theorem_l2077_207721


namespace equation_solution_l2077_207754

theorem equation_solution :
  let f : ℂ → ℂ := λ x => x^3 + 4*x^2*Real.sqrt 3 + 12*x + 4*Real.sqrt 3 + x^2 - 1
  ∀ x : ℂ, f x = 0 ↔ x = 0 ∨ x = -Real.sqrt 3 ∨ x = (-Real.sqrt 3 + Complex.I)/2 ∨ x = (-Real.sqrt 3 - Complex.I)/2 :=
by
  sorry

end equation_solution_l2077_207754


namespace max_sum_in_t_grid_l2077_207736

/-- A T-shaped grid represented as a list of 5 integers -/
def TGrid := List Int

/-- Check if a T-shaped grid is valid (contains exactly the numbers 2, 5, 8, 11, 14) -/
def isValidTGrid (grid : TGrid) : Prop :=
  grid.length = 5 ∧ grid.toFinset = {2, 5, 8, 11, 14}

/-- Calculate the vertical sum of a T-shaped grid -/
def verticalSum (grid : TGrid) : Int :=
  match grid with
  | [a, b, c, _, _] => a + b + c
  | _ => 0

/-- Calculate the horizontal sum of a T-shaped grid -/
def horizontalSum (grid : TGrid) : Int :=
  match grid with
  | [_, b, _, d, e] => b + d + e
  | _ => 0

/-- Check if a T-shaped grid satisfies the sum condition -/
def satisfiesSumCondition (grid : TGrid) : Prop :=
  verticalSum grid = horizontalSum grid

/-- The main theorem: The maximum sum in a valid T-shaped grid is 33 -/
theorem max_sum_in_t_grid :
  ∀ (grid : TGrid),
    isValidTGrid grid →
    satisfiesSumCondition grid →
    (verticalSum grid ≤ 33 ∧ horizontalSum grid ≤ 33) :=
by sorry

end max_sum_in_t_grid_l2077_207736


namespace find_other_number_l2077_207734

theorem find_other_number (a b : ℤ) (h1 : a - b = 8) (h2 : a = 16) : b = 8 := by
  sorry

end find_other_number_l2077_207734


namespace rug_area_is_24_l2077_207794

/-- Calculates the area of a rug on a rectangular floor with uncovered strips along the edges -/
def rugArea (floorLength floorWidth stripWidth : ℝ) : ℝ :=
  (floorLength - 2 * stripWidth) * (floorWidth - 2 * stripWidth)

/-- Theorem stating that the area of the rug is 24 square meters given the specific dimensions -/
theorem rug_area_is_24 :
  rugArea 10 8 2 = 24 := by
  sorry

#eval rugArea 10 8 2

end rug_area_is_24_l2077_207794
