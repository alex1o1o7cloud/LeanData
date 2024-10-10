import Mathlib

namespace kaleb_initial_savings_l2656_265646

/-- The amount of money Kaleb had initially saved up. -/
def initial_savings : ℕ := sorry

/-- The cost of each toy. -/
def toy_cost : ℕ := 6

/-- The number of toys Kaleb can buy after receiving his allowance. -/
def num_toys : ℕ := 6

/-- The amount of allowance Kaleb received. -/
def allowance : ℕ := 15

/-- Theorem stating that Kaleb's initial savings were $21. -/
theorem kaleb_initial_savings :
  initial_savings = 21 :=
by
  sorry

end kaleb_initial_savings_l2656_265646


namespace fraction_equality_l2656_265664

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 8 / 5) : 
  (2 * m * r - 3 * n * t) / (5 * n * t - 4 * m * r) = -5 / 11 := by
  sorry

end fraction_equality_l2656_265664


namespace largest_multiple_of_15_under_500_l2656_265635

theorem largest_multiple_of_15_under_500 : 
  ∀ n : ℕ, n * 15 < 500 → n * 15 ≤ 495 :=
sorry

end largest_multiple_of_15_under_500_l2656_265635


namespace surface_area_increase_after_removal_l2656_265642

/-- Represents a rectangular solid with length, width, and height -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Represents the change in surface area after removal of a smaller prism -/
def surfaceAreaChange (larger : RectangularSolid) (smaller : RectangularSolid) : ℝ :=
  (smaller.length * smaller.width + smaller.length * smaller.height + smaller.width * smaller.height) * 2 -
  smaller.length * smaller.width

theorem surface_area_increase_after_removal :
  let larger := RectangularSolid.mk 5 3 2
  let smaller := RectangularSolid.mk 2 1 1
  surfaceAreaChange larger smaller = 4 := by
  sorry


end surface_area_increase_after_removal_l2656_265642


namespace bernoulli_inequality_l2656_265665

theorem bernoulli_inequality (p : ℝ) (k : ℚ) (hp : p > 0) (hk : k > 1) :
  (1 + p)^(k : ℝ) > 1 + p * k := by
  sorry

end bernoulli_inequality_l2656_265665


namespace g_at_negative_two_l2656_265658

/-- The function g(x) = 2x^2 + 3x + 1 -/
def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

/-- Theorem: g(-2) = 3 -/
theorem g_at_negative_two : g (-2) = 3 := by
  sorry

end g_at_negative_two_l2656_265658


namespace function_ordering_l2656_265652

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem function_ordering (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : ∀ x₁ x₂, x₁ ≤ -1 ∧ x₂ ≤ -1 → (x₂ - x₁) * (f x₂ - f x₁) < 0) :
  f (-1) < f (-3/2) ∧ f (-3/2) < f 2 :=
sorry

end function_ordering_l2656_265652


namespace arithmetic_geometric_sequence_ratio_l2656_265620

/-- An arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def is_geometric_sequence (b : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, b (n + 1) = q * b n

/-- The theorem statement -/
theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  is_arithmetic_sequence a →
  is_geometric_sequence (λ n => a (2*n - 1) - (2*n - 1)) q →
  q = 1 := by
  sorry

end arithmetic_geometric_sequence_ratio_l2656_265620


namespace part1_part2_l2656_265676

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 3| + |x - a|

-- Part 1
theorem part1 (x : ℝ) :
  f 4 x = 7 → -3 ≤ x ∧ x ≤ 4 :=
by sorry

-- Part 2
theorem part2 (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2}) → a = 1 :=
by sorry

end part1_part2_l2656_265676


namespace real_roots_range_roots_condition_m_value_l2656_265666

/-- Quadratic equation parameters -/
def a : ℝ := 1
def b (m : ℝ) : ℝ := 2 * (m - 1)
def c (m : ℝ) : ℝ := m^2 + 2

/-- Discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ := (b m)^2 - 4 * a * (c m)

/-- Theorem stating the range of m for real roots -/
theorem real_roots_range (m : ℝ) :
  (∃ x : ℝ, a * x^2 + (b m) * x + (c m) = 0) ↔ m ≤ -1/2 := by sorry

/-- Theorem stating the value of m when the roots satisfy the given condition -/
theorem roots_condition_m_value (m : ℝ) (x₁ x₂ : ℝ) 
  (hroots : a * x₁^2 + (b m) * x₁ + (c m) = 0 ∧ a * x₂^2 + (b m) * x₂ + (c m) = 0)
  (hcond : (x₁ - x₂)^2 = 18 - x₁ * x₂) :
  m = -2 := by sorry

end real_roots_range_roots_condition_m_value_l2656_265666


namespace magazines_to_boxes_l2656_265653

theorem magazines_to_boxes (total_magazines : ℕ) (magazines_per_box : ℕ) (h1 : total_magazines = 63) (h2 : magazines_per_box = 9) :
  total_magazines / magazines_per_box = 7 := by
  sorry

end magazines_to_boxes_l2656_265653


namespace mass_percentage_Al_approx_l2656_265661

-- Define atomic masses
def atomic_mass_Al : ℝ := 26.98
def atomic_mass_S : ℝ := 32.06
def atomic_mass_Ca : ℝ := 40.08
def atomic_mass_C : ℝ := 12.01
def atomic_mass_O : ℝ := 16.00
def atomic_mass_K : ℝ := 39.10
def atomic_mass_Cl : ℝ := 35.45

-- Define molar masses of compounds
def molar_mass_Al2S3 : ℝ := 2 * atomic_mass_Al + 3 * atomic_mass_S
def molar_mass_CaCO3 : ℝ := atomic_mass_Ca + atomic_mass_C + 3 * atomic_mass_O
def molar_mass_KCl : ℝ := atomic_mass_K + atomic_mass_Cl

-- Define moles of compounds in the mixture
def moles_Al2S3 : ℝ := 2
def moles_CaCO3 : ℝ := 3
def moles_KCl : ℝ := 5

-- Define total mass of the mixture
def total_mass : ℝ := moles_Al2S3 * molar_mass_Al2S3 + moles_CaCO3 * molar_mass_CaCO3 + moles_KCl * molar_mass_KCl

-- Define mass of Al in the mixture
def mass_Al : ℝ := 2 * moles_Al2S3 * atomic_mass_Al

-- Theorem: The mass percentage of Al in the mixture is approximately 11.09%
theorem mass_percentage_Al_approx (ε : ℝ) (h : ε > 0) : 
  ∃ δ : ℝ, δ > 0 ∧ |mass_Al / total_mass * 100 - 11.09| < δ :=
sorry

end mass_percentage_Al_approx_l2656_265661


namespace quadrilateral_diagonal_length_l2656_265696

/-- Given a quadrilateral ABCD with diagonals intersecting at O, this theorem proves
    that under specific conditions, the length of AD is 2√57. -/
theorem quadrilateral_diagonal_length
  (A B C D O : ℝ × ℝ) -- Points in 2D space
  (h_intersect : (A.1 - C.1) * (B.2 - D.2) = (A.2 - C.2) * (B.1 - D.1)) -- Diagonals intersect
  (h_BO : dist B O = 5)
  (h_OD : dist O D = 7)
  (h_AO : dist A O = 9)
  (h_OC : dist O C = 4)
  (h_AB : dist A B = 6)
  (h_BD : dist B D = 6) :
  dist A D = 2 * Real.sqrt 57 :=
sorry

end quadrilateral_diagonal_length_l2656_265696


namespace heathers_oranges_l2656_265604

/-- The total number of oranges Heather has after receiving oranges from Russell -/
def total_oranges (initial : ℝ) (received : ℝ) : ℝ :=
  initial + received

/-- Theorem stating that Heather's total oranges is 96.3 given the initial and received amounts -/
theorem heathers_oranges :
  total_oranges 60.5 35.8 = 96.3 := by
  sorry

end heathers_oranges_l2656_265604


namespace number_of_blue_balls_l2656_265608

/-- Given a set of balls with red, blue, and green colors, prove the number of blue balls. -/
theorem number_of_blue_balls
  (total : ℕ)
  (green : ℕ)
  (h1 : total = 40)
  (h2 : green = 7)
  (h3 : ∃ (blue : ℕ), total = green + blue + 2 * blue) :
  ∃ (blue : ℕ), blue = 11 ∧ total = green + blue + 2 * blue :=
sorry

end number_of_blue_balls_l2656_265608


namespace probability_same_number_four_dice_l2656_265692

/-- The number of sides on a standard die -/
def standardDieSides : ℕ := 6

/-- The number of dice being rolled -/
def numberOfDice : ℕ := 4

/-- The probability of all dice showing the same number -/
def probabilitySameNumber : ℚ := 1 / (standardDieSides ^ (numberOfDice - 1))

theorem probability_same_number_four_dice :
  probabilitySameNumber = 1 / 216 := by
  sorry

end probability_same_number_four_dice_l2656_265692


namespace rectangles_with_one_gray_cell_l2656_265603

/-- The number of rectangles containing exactly one gray cell in a 2x20 grid --/
def num_rectangles_with_one_gray_cell (total_gray_cells : ℕ) 
  (blue_cells : ℕ) (red_cells : ℕ) : ℕ :=
  blue_cells * 4 + red_cells * 8

/-- Theorem stating the number of rectangles with one gray cell in the given grid --/
theorem rectangles_with_one_gray_cell :
  num_rectangles_with_one_gray_cell 40 36 4 = 176 := by
  sorry

#eval num_rectangles_with_one_gray_cell 40 36 4

end rectangles_with_one_gray_cell_l2656_265603


namespace quadratic_sum_l2656_265699

/-- A quadratic function f(x) = ax^2 + bx + c with vertex (3, -2) and f(0) = 0 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x = a * (x - 3)^2 - 2) →  -- vertex form
  QuadraticFunction a b c 0 = 0 →                         -- passes through (0, 0)
  a + b + c = -10/9 := by
  sorry

end quadratic_sum_l2656_265699


namespace pool_supply_problem_l2656_265697

theorem pool_supply_problem (x : ℕ) (h1 : x + 3 * x = 800) : x = 266 := by
  sorry

end pool_supply_problem_l2656_265697


namespace max_diff_squares_consecutive_integers_l2656_265618

theorem max_diff_squares_consecutive_integers (n : ℤ) : 
  n + (n + 1) < 150 → (n + 1)^2 - n^2 ≤ 149 := by
  sorry

end max_diff_squares_consecutive_integers_l2656_265618


namespace power_of_same_base_power_of_different_base_l2656_265662

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Theorem 1: Condition for representing a^n as (a^p)^q
theorem power_of_same_base (a n : ℕ) (h : n > 1) :
  (∃ p q : ℕ, p > 1 ∧ q > 1 ∧ a^n = (a^p)^q) ↔ ¬(isPrime n) :=
sorry

-- Theorem 2: Condition for representing a^n as b^m with a different base
theorem power_of_different_base (a n : ℕ) (h : n > 0) :
  (∃ b m : ℕ, b ≠ a ∧ m > 0 ∧ a^n = b^m) ↔
  (∃ k : ℕ, k > 0 ∧ ∃ b : ℕ, b ≠ a ∧ a^n = (b^k)^(n/k)) :=
sorry

end power_of_same_base_power_of_different_base_l2656_265662


namespace tangent_line_equation_l2656_265600

def S (x : ℝ) : ℝ := 3*x - x^3

theorem tangent_line_equation (x₀ y₀ : ℝ) (h : y₀ = S x₀) (h₀ : x₀ = 2) (h₁ : y₀ = -2) :
  ∃ (m b : ℝ), (∀ x y, y = m*x + b → (x = x₀ ∧ y = y₀) ∨ (y - y₀ = m*(x - x₀))) ∧
  ((m = -9 ∧ b = 16) ∨ (m = 0 ∧ b = -2)) :=
sorry

end tangent_line_equation_l2656_265600


namespace lcm_520_693_l2656_265673

theorem lcm_520_693 : Nat.lcm 520 693 = 360360 := by
  sorry

end lcm_520_693_l2656_265673


namespace x_value_l2656_265614

theorem x_value (x : ℝ) (h : x ∈ ({1, x^2} : Set ℝ)) : x = 0 := by
  sorry

end x_value_l2656_265614


namespace band_members_count_l2656_265688

theorem band_members_count (flute trumpet trombone drummer clarinet french_horn saxophone piano violin guitar : ℕ) : 
  flute = 5 →
  trumpet = 3 * flute →
  trombone = trumpet - 8 →
  drummer = trombone + 11 →
  clarinet = 2 * flute →
  french_horn = trombone + 3 →
  saxophone = (trumpet + trombone) / 2 →
  piano = drummer + 2 →
  violin = french_horn - clarinet →
  guitar = 3 * flute →
  flute + trumpet + trombone + drummer + clarinet + french_horn + saxophone + piano + violin + guitar = 111 := by
sorry

end band_members_count_l2656_265688


namespace handshake_count_l2656_265698

/-- The number of handshakes in a convention of gremlins and imps -/
theorem handshake_count (num_gremlins num_imps : ℕ) : 
  num_gremlins = 20 →
  num_imps = 15 →
  (num_gremlins * (num_gremlins - 1)) / 2 + num_gremlins * num_imps = 490 := by
  sorry

#check handshake_count

end handshake_count_l2656_265698


namespace no_real_numbers_with_integer_roots_l2656_265644

theorem no_real_numbers_with_integer_roots : 
  ¬ ∃ (a b c : ℝ), 
    (∃ (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) ∧
    (∃ (y₁ y₂ : ℤ), y₁ ≠ y₂ ∧ (a+1) * y₁^2 + (b+1) * y₁ + (c+1) = 0 ∧ (a+1) * y₂^2 + (b+1) * y₂ + (c+1) = 0) :=
by
  sorry


end no_real_numbers_with_integer_roots_l2656_265644


namespace inequality_proof_l2656_265654

theorem inequality_proof (a b : ℝ) : 
  |a + b| / (1 + |a + b|) ≤ |a| / (1 + |a|) + |b| / (1 + |b|) := by
  sorry

end inequality_proof_l2656_265654


namespace b_value_when_square_zero_l2656_265677

theorem b_value_when_square_zero (b : ℝ) : (b + 5)^2 = 0 → b = -5 := by
  sorry

end b_value_when_square_zero_l2656_265677


namespace largest_integer_negative_quadratic_l2656_265669

theorem largest_integer_negative_quadratic : 
  (∀ m : ℤ, m > 7 → m^2 - 11*m + 24 ≥ 0) ∧ 
  (7^2 - 11*7 + 24 < 0) := by
  sorry

end largest_integer_negative_quadratic_l2656_265669


namespace repeating_decimal_equals_fraction_l2656_265626

/-- The repeating decimal 0.464646... expressed as a real number -/
def repeating_decimal : ℚ := 46 / 99

/-- The theorem stating that the repeating decimal 0.464646... is equal to 46/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = 46 / 99 := by
  sorry

end repeating_decimal_equals_fraction_l2656_265626


namespace license_plate_difference_l2656_265622

/-- The number of possible letters in a license plate position -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate position -/
def num_digits : ℕ := 10

/-- The number of possible Florida license plates -/
def florida_plates : ℕ := num_letters^6 * num_digits^2

/-- The number of possible Texas license plates -/
def texas_plates : ℕ := num_letters^3 * num_digits^4

/-- The difference between Florida and Texas license plate possibilities -/
def plate_difference : ℕ := florida_plates - texas_plates

theorem license_plate_difference :
  plate_difference = 54293545536 := by
  sorry

end license_plate_difference_l2656_265622


namespace quadratic_rational_root_even_coeff_l2656_265606

theorem quadratic_rational_root_even_coeff
  (a b c : ℤ) (x : ℚ)
  (h_a_nonzero : a ≠ 0)
  (h_root : a * x^2 + b * x + c = 0) :
  Even a ∨ Even b ∨ Even c :=
sorry

end quadratic_rational_root_even_coeff_l2656_265606


namespace append_12_to_three_digit_number_l2656_265643

theorem append_12_to_three_digit_number (h t u : ℕ) :
  let original := 100 * h + 10 * t + u
  let new_number := original * 100 + 12
  new_number = 10000 * h + 1000 * t + 100 * u + 12 :=
by sorry

end append_12_to_three_digit_number_l2656_265643


namespace ranch_feed_corn_cost_l2656_265668

/-- Represents the ranch with its animals and pasture. -/
structure Ranch where
  sheep : ℕ
  cattle : ℕ
  pasture_acres : ℕ

/-- Represents the feed requirements and costs. -/
structure FeedInfo where
  cow_grass_per_month : ℕ
  sheep_grass_per_month : ℕ
  corn_bag_cost : ℕ
  cow_corn_months_per_bag : ℕ
  sheep_corn_months_per_bag : ℕ

/-- Calculates the annual cost of feed corn for the ranch. -/
def annual_feed_corn_cost (ranch : Ranch) (feed : FeedInfo) : ℕ :=
  sorry

/-- Theorem stating the annual feed corn cost for the given ranch and feed information. -/
theorem ranch_feed_corn_cost :
  let ranch := Ranch.mk 8 5 144
  let feed := FeedInfo.mk 2 1 10 1 2
  annual_feed_corn_cost ranch feed = 360 :=
sorry

end ranch_feed_corn_cost_l2656_265668


namespace new_bus_distance_l2656_265627

theorem new_bus_distance (old_distance : ℝ) (percentage_increase : ℝ) (new_distance : ℝ) : 
  old_distance = 300 →
  percentage_increase = 0.30 →
  new_distance = old_distance * (1 + percentage_increase) →
  new_distance = 390 := by
sorry

end new_bus_distance_l2656_265627


namespace sum_of_three_squares_l2656_265683

theorem sum_of_three_squares (square triangle : ℚ) : 
  (square + triangle + 2 * square + triangle = 34) →
  (triangle + square + triangle + 3 * square = 40) →
  (3 * square = 66 / 7) := by
  sorry

end sum_of_three_squares_l2656_265683


namespace circle_equation_to_circle_params_l2656_265689

/-- A circle in the 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in the form ax² + by² + cx + dy + e = 0. -/
def CircleEquation (a b c d e : ℝ) : (ℝ × ℝ) → Prop :=
  fun p => a * p.1^2 + b * p.2^2 + c * p.1 + d * p.2 + e = 0

theorem circle_equation_to_circle_params :
  ∃! (circle : Circle),
    (∀ p, CircleEquation 1 1 (-4) 2 0 p ↔ (p.1 - circle.center.1)^2 + (p.2 - circle.center.2)^2 = circle.radius^2) ∧
    circle.center = (2, -1) ∧
    circle.radius = Real.sqrt 5 := by
  sorry

end circle_equation_to_circle_params_l2656_265689


namespace orange_juice_mix_l2656_265621

/-- Given the conditions for preparing orange juice, prove that 3 cans of water are needed per can of concentrate. -/
theorem orange_juice_mix (servings : ℕ) (serving_size : ℚ) (concentrate_cans : ℕ) (concentrate_size : ℚ) 
  (h1 : servings = 200)
  (h2 : serving_size = 6)
  (h3 : concentrate_cans = 60)
  (h4 : concentrate_size = 5) : 
  (servings * serving_size - concentrate_cans * concentrate_size) / (concentrate_cans * concentrate_size) = 3 := by
  sorry

end orange_juice_mix_l2656_265621


namespace sqrt_2023_bounds_l2656_265680

theorem sqrt_2023_bounds : 40 < Real.sqrt 2023 ∧ Real.sqrt 2023 < 45 := by
  have h1 : 1600 < 2023 := by sorry
  have h2 : 2023 < 2025 := by sorry
  sorry

end sqrt_2023_bounds_l2656_265680


namespace order_of_6_l2656_265675

def f (x : ℕ) : ℕ := x^2 % 13

def is_periodic (f : ℕ → ℕ) (x : ℕ) (period : ℕ) : Prop :=
  ∀ n, f^[n + period] x = f^[n] x

theorem order_of_6 (h : is_periodic f 6 72) :
  ∀ k, 0 < k → k < 72 → ¬ is_periodic f 6 k :=
sorry

end order_of_6_l2656_265675


namespace difference_three_fifths_l2656_265672

theorem difference_three_fifths (x : ℝ) : x - (3/5) * x = 145 → x = 362.5 := by
  sorry

end difference_three_fifths_l2656_265672


namespace mushroom_collection_problem_l2656_265670

theorem mushroom_collection_problem :
  ∃ (x₁ x₂ x₃ x₄ : ℕ),
    x₁ + x₂ = 7 ∧
    x₁ + x₃ = 9 ∧
    x₁ + x₄ = 10 ∧
    x₂ + x₃ = 10 ∧
    x₂ + x₄ = 11 ∧
    x₃ + x₄ = 13 ∧
    x₁ ≤ x₂ ∧ x₂ ≤ x₃ ∧ x₃ ≤ x₄ :=
by sorry

end mushroom_collection_problem_l2656_265670


namespace abs_sum_inequality_l2656_265610

theorem abs_sum_inequality (x : ℝ) : 
  |x - 1| + |x - 2| > 3 ↔ x < 0 ∨ x > 3 := by sorry

end abs_sum_inequality_l2656_265610


namespace angle_between_vectors_solution_l2656_265695

def angle_between_vectors (problem : Unit) : Prop :=
  ∃ (a b : ℝ × ℝ),
    let dot_product := (a.1 * b.1 + a.2 * b.2)
    let magnitude := fun v : ℝ × ℝ => Real.sqrt (v.1^2 + v.2^2)
    let angle := Real.arccos (dot_product / (magnitude a * magnitude b))
    (a.1 * a.1 + a.2 * a.2 - 2 * (a.1 * b.1 + a.2 * b.2) = 3) ∧
    (magnitude a = 1) ∧
    (b = (1, 1)) ∧
    (angle = 3 * Real.pi / 4)

theorem angle_between_vectors_solution : angle_between_vectors () := by
  sorry

end angle_between_vectors_solution_l2656_265695


namespace noras_age_l2656_265612

/-- Represents a person's age --/
structure Person where
  age : ℕ

/-- Proves that Nora's current age is 10 years old --/
theorem noras_age (terry nora : Person) : 
  (terry.age + 10 = 4 * nora.age) → 
  (terry.age = 30) → 
  (nora.age = 10) := by
sorry

end noras_age_l2656_265612


namespace reading_difference_l2656_265667

theorem reading_difference (min_assigned : ℕ) (harrison_extra : ℕ) (sam_pages : ℕ) :
  min_assigned = 25 →
  harrison_extra = 10 →
  sam_pages = 100 →
  ∃ (pam_pages : ℕ) (harrison_pages : ℕ),
    pam_pages = sam_pages / 2 ∧
    harrison_pages = min_assigned + harrison_extra ∧
    pam_pages > harrison_pages ∧
    pam_pages - harrison_pages = 15 :=
by sorry

end reading_difference_l2656_265667


namespace distance_between_parallel_lines_l2656_265649

/-- The distance between two parallel lines -/
theorem distance_between_parallel_lines :
  let l₁ : ℝ → ℝ → Prop := fun x y ↦ x - 2 * y + 1 = 0
  let l₂ : ℝ → ℝ → Prop := fun x y ↦ x - 2 * y - 4 = 0
  ∃ d : ℝ, d = Real.sqrt 5 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ), l₁ x₁ y₁ → l₂ x₂ y₂ →
      ((x₂ - x₁)^2 + (y₂ - y₁)^2 : ℝ) ≥ d^2 :=
by sorry


end distance_between_parallel_lines_l2656_265649


namespace percentage_calculation_l2656_265607

theorem percentage_calculation : 
  (0.2 * 120 + 0.25 * 250 + 0.15 * 80) - 0.1 * 600 = 38.5 := by
  sorry

end percentage_calculation_l2656_265607


namespace sum_product_inequalities_l2656_265616

theorem sum_product_inequalities (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((a + b) * (1/a + 1/b) ≥ 4) ∧ ((a + b + c) * (1/a + 1/b + 1/c) ≥ 9) := by sorry

end sum_product_inequalities_l2656_265616


namespace kids_meals_sold_l2656_265624

theorem kids_meals_sold (kids_meals : ℕ) (adult_meals : ℕ) : 
  (kids_meals : ℚ) / (adult_meals : ℚ) = 2 / 1 →
  kids_meals + adult_meals = 12 →
  kids_meals = 8 := by
sorry

end kids_meals_sold_l2656_265624


namespace sports_conference_games_l2656_265638

/-- Calculates the total number of games in a sports conference season --/
def total_games (total_teams : ℕ) (teams_per_division : ℕ) (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let games_per_team := (teams_per_division - 1) * intra_division_games + teams_per_division * inter_division_games
  (total_teams * games_per_team) / 2

theorem sports_conference_games : 
  total_games 12 6 3 2 = 162 := by sorry

end sports_conference_games_l2656_265638


namespace max_b_squared_l2656_265615

theorem max_b_squared (a b : ℤ) : 
  (a + b)^2 + a*(a + b) + b = 0 → b^2 ≤ 81 :=
by sorry

end max_b_squared_l2656_265615


namespace cornelia_age_proof_l2656_265601

/-- Cornelia's current age -/
def cornelia_age : ℕ := 80

/-- Kilee's current age -/
def kilee_age : ℕ := 20

/-- In 10 years, Cornelia will be three times as old as Kilee -/
theorem cornelia_age_proof :
  cornelia_age + 10 = 3 * (kilee_age + 10) :=
by sorry

end cornelia_age_proof_l2656_265601


namespace sin_15_cos_15_eq_quarter_l2656_265623

theorem sin_15_cos_15_eq_quarter : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end sin_15_cos_15_eq_quarter_l2656_265623


namespace polynomial_factorization_l2656_265684

theorem polynomial_factorization (x y : ℝ) : 
  x^3 - 4*x^2*y + 4*x*y^2 = x*(x - 2*y)^2 := by
  sorry

end polynomial_factorization_l2656_265684


namespace probability_distinct_numbers_value_l2656_265611

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The probability of rolling six distinct numbers with six eight-sided dice -/
def probability_distinct_numbers : ℚ :=
  (num_sides.factorial / (num_sides - num_dice).factorial) / num_sides ^ num_dice

theorem probability_distinct_numbers_value :
  probability_distinct_numbers = 315 / 4096 := by
  sorry

end probability_distinct_numbers_value_l2656_265611


namespace gumball_count_l2656_265629

/-- Represents a gumball machine with red, green, and blue gumballs. -/
structure GumballMachine where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Creates a gumball machine with the given conditions. -/
def createMachine (redCount : ℕ) : GumballMachine :=
  let blueCount := redCount / 2
  let greenCount := blueCount * 4
  { red := redCount, blue := blueCount, green := greenCount }

/-- Calculates the total number of gumballs in the machine. -/
def totalGumballs (machine : GumballMachine) : ℕ :=
  machine.red + machine.blue + machine.green

/-- Theorem stating that a machine with 16 red gumballs has 56 gumballs in total. -/
theorem gumball_count : totalGumballs (createMachine 16) = 56 := by
  sorry

end gumball_count_l2656_265629


namespace robin_camera_pictures_l2656_265630

/-- The number of pictures Robin uploaded from her camera -/
def camera_pictures (phone_pictures total_albums pictures_per_album : ℕ) : ℕ :=
  total_albums * pictures_per_album - phone_pictures

/-- Proof that Robin uploaded 5 pictures from her camera -/
theorem robin_camera_pictures :
  camera_pictures 35 5 8 = 5 := by
  sorry

end robin_camera_pictures_l2656_265630


namespace constant_speed_motion_not_correlation_l2656_265691

/-- Definition of a correlation relationship -/
def correlation_relationship (X Y : Type) (f : X → Y) :=
  ∃ (pattern : X → Set Y), ∀ x : X, f x ∈ pattern x ∧ ¬ (∃ y : Y, pattern x = {y})

/-- Definition of a functional relationship -/
def functional_relationship (X Y : Type) (f : X → Y) :=
  ∀ x : X, ∃! y : Y, f x = y

/-- Distance as a function of time for constant speed motion -/
def distance (v : ℝ) (t : ℝ) : ℝ := v * t

theorem constant_speed_motion_not_correlation :
  ∀ v : ℝ, v > 0 → ¬ (correlation_relationship ℝ ℝ (distance v)) :=
sorry

end constant_speed_motion_not_correlation_l2656_265691


namespace geometric_series_sum_l2656_265632

theorem geometric_series_sum : 
  let a : ℝ := (Real.sqrt 2 + 1) / (Real.sqrt 2 - 1)
  let q : ℝ := (Real.sqrt 2 - 1) / Real.sqrt 2
  let S : ℝ := a / (1 - q)
  S = 6 + 4 * Real.sqrt 2 := by
  sorry

end geometric_series_sum_l2656_265632


namespace january_salary_l2656_265655

/-- Represents the salary structure for a person over 5 months -/
structure SalaryStructure where
  jan : ℝ
  feb : ℝ
  mar : ℝ
  apr : ℝ
  may : ℝ
  bonus : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem january_salary (s : SalaryStructure) 
  (avg_jan_apr : (s.jan + s.feb + s.mar + s.apr) / 4 = 8000)
  (avg_feb_may : (s.feb + s.mar + s.apr + s.may) / 4 = 8400)
  (may_salary : s.may = 6500)
  (apr_raise : s.apr = 1.05 * s.feb)
  (mar_bonus : s.mar = s.feb + s.bonus) :
  s.jan = 4900 := by
  sorry

end january_salary_l2656_265655


namespace sin_cos_fourth_power_sum_l2656_265685

theorem sin_cos_fourth_power_sum (θ : Real) (h : Real.sin (2 * θ) = 1 / 2) :
  Real.sin θ ^ 4 + Real.cos θ ^ 4 = 7 / 8 := by
  sorry

end sin_cos_fourth_power_sum_l2656_265685


namespace number_comparison_l2656_265671

theorem number_comparison : ∃ (a b c : ℝ), 
  a = 7^(0.3 : ℝ) ∧ 
  b = (0.3 : ℝ)^7 ∧ 
  c = Real.log 0.3 ∧ 
  a > b ∧ b > c := by
  sorry

end number_comparison_l2656_265671


namespace parallel_perpendicular_lines_l2656_265609

/-- Given a point P and a line l, prove the equations of parallel and perpendicular lines through P -/
theorem parallel_perpendicular_lines
  (P : ℝ × ℝ)  -- Point P
  (l : ℝ → ℝ → Prop)  -- Line l
  (hl : ∀ x y, l x y ↔ 3 * x - 2 * y - 7 = 0)  -- Equation of line l
  (hP : P = (-4, 2))  -- Coordinates of point P
  : 
  -- 1. Equation of parallel line through P
  (∀ x y, (3 * x - 2 * y + 16 = 0) ↔ 
    (∃ k, k ≠ 0 ∧ ∀ a b, l a b → (3 * x - 2 * y = 3 * a - 2 * b + k))) ∧
    (3 * P.1 - 2 * P.2 + 16 = 0) ∧

  -- 2. Equation of perpendicular line through P
  (∀ x y, (2 * x + 3 * y + 2 = 0) ↔ 
    (∀ a b, l a b → (3 * (x - a) + 2 * (y - b) = 0))) ∧
    (2 * P.1 + 3 * P.2 + 2 = 0) :=
by sorry

end parallel_perpendicular_lines_l2656_265609


namespace cylinder_cone_lateral_area_ratio_l2656_265641

/-- The ratio of lateral surface areas of a cylinder and a cone with equal slant heights and base radii -/
theorem cylinder_cone_lateral_area_ratio 
  (r : ℝ) -- base radius
  (l : ℝ) -- slant height
  (h_pos_r : r > 0)
  (h_pos_l : l > 0) :
  (2 * π * r * l) / (π * r * l) = 2 := by
  sorry

#check cylinder_cone_lateral_area_ratio

end cylinder_cone_lateral_area_ratio_l2656_265641


namespace quadratic_solution_sum_l2656_265619

theorem quadratic_solution_sum (a b : ℝ) : 
  (5 * (a + b * Complex.I)^2 + 4 * (a + b * Complex.I) + 1 = 0 ∧
   5 * (a - b * Complex.I)^2 + 4 * (a - b * Complex.I) + 1 = 0) →
  a + b^2 = -9/25 := by
sorry

end quadratic_solution_sum_l2656_265619


namespace simplify_expression_l2656_265659

theorem simplify_expression (a b m : ℝ) (h1 : a + b = m) (h2 : a * b = -4) :
  (a - 2) * (b - 2) = -2 * m := by sorry

end simplify_expression_l2656_265659


namespace find_b_value_l2656_265657

theorem find_b_value (a b : ℚ) (h1 : 3 * a - 2 = 1) (h2 : 2 * b - 3 * a = 2) : b = 5/2 := by
  sorry

end find_b_value_l2656_265657


namespace winnie_lollipop_distribution_l2656_265690

theorem winnie_lollipop_distribution (total_lollipops : ℕ) (friends : ℕ) 
  (h1 : total_lollipops = 37 + 108 + 8 + 254) 
  (h2 : friends = 13) : 
  total_lollipops % friends = 4 := by
  sorry

end winnie_lollipop_distribution_l2656_265690


namespace journey_distance_l2656_265639

/-- Prove that given a journey with specified conditions, the total distance traveled is 270 km. -/
theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  total_time = 15 →
  speed1 = 45 →
  speed2 = 30 →
  (total_time * speed1 * speed2) / (speed1 + speed2) = 270 := by
  sorry

#check journey_distance

end journey_distance_l2656_265639


namespace square_circle_area_ratio_l2656_265694

theorem square_circle_area_ratio (s r : ℝ) (h : 4 * s = 2 * Real.pi * r) :
  s^2 / (Real.pi * r^2) = 4 / Real.pi := by
  sorry

end square_circle_area_ratio_l2656_265694


namespace greatest_integer_x_prime_l2656_265681

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def f (x : ℤ) : ℤ := |6 * x^2 - 47 * x + 15|

theorem greatest_integer_x_prime :
  ∀ x : ℤ, (is_prime (f x).toNat → x ≤ 8) ∧
  (is_prime (f 8).toNat) :=
sorry

end greatest_integer_x_prime_l2656_265681


namespace board_game_cost_l2656_265656

def number_of_games : ℕ := 6
def total_paid : ℕ := 100
def change_bill_value : ℕ := 5
def number_of_change_bills : ℕ := 2

theorem board_game_cost :
  (total_paid - (change_bill_value * number_of_change_bills)) / number_of_games = 15 := by
  sorry

end board_game_cost_l2656_265656


namespace remainder_problem_l2656_265634

theorem remainder_problem (x R : ℤ) : 
  (∃ k : ℤ, x = 82 * k + R) → 
  (∃ m : ℤ, x + 17 = 41 * m + 22) → 
  R = 5 := by
sorry

end remainder_problem_l2656_265634


namespace exists_table_with_square_corner_sums_l2656_265663

/-- Represents a 100 x 100 table of natural numbers -/
def Table := Fin 100 → Fin 100 → ℕ

/-- Checks if all numbers in the same row or column are different -/
def all_different (t : Table) : Prop :=
  ∀ i j i' j', i ≠ i' ∨ j ≠ j' → t i j ≠ t i' j'

/-- Checks if the sum of numbers in angle cells of a square submatrix is a square number -/
def corner_sum_is_square (t : Table) : Prop :=
  ∀ i j n, ∃ k : ℕ, 
    t i j + t i (j + n) + t (i + n) j + t (i + n) (j + n) = k * k

/-- The main theorem stating the existence of a table satisfying all conditions -/
theorem exists_table_with_square_corner_sums : 
  ∃ t : Table, all_different t ∧ corner_sum_is_square t := by
  sorry

end exists_table_with_square_corner_sums_l2656_265663


namespace laborer_income_l2656_265637

/-- Represents the financial situation of a laborer over a 10-month period. -/
structure LaborerFinances where
  monthly_income : ℝ
  initial_expenditure : ℝ
  reduced_expenditure : ℝ
  initial_period : ℕ
  reduced_period : ℕ
  savings : ℝ

/-- The laborer's finances satisfy the given conditions. -/
def satisfies_conditions (f : LaborerFinances) : Prop :=
  f.initial_expenditure = 75 ∧
  f.reduced_expenditure = 60 ∧
  f.initial_period = 6 ∧
  f.reduced_period = 4 ∧
  f.savings = 30 ∧
  f.initial_period * f.monthly_income < f.initial_period * f.initial_expenditure ∧
  f.reduced_period * f.monthly_income = f.reduced_period * f.reduced_expenditure + 
    (f.initial_period * f.initial_expenditure - f.initial_period * f.monthly_income) + f.savings

/-- Theorem stating that if the laborer's finances satisfy the given conditions, 
    then their monthly income is 72. -/
theorem laborer_income (f : LaborerFinances) 
  (h : satisfies_conditions f) : f.monthly_income = 72 := by
  sorry

end laborer_income_l2656_265637


namespace james_milk_consumption_l2656_265617

/-- The amount of milk James drank, given his initial amount, the conversion rate from gallons to ounces, and the remaining amount. -/
def milk_drank (initial_gallons : ℕ) (ounces_per_gallon : ℕ) (remaining_ounces : ℕ) : ℕ :=
  initial_gallons * ounces_per_gallon - remaining_ounces

/-- Theorem stating that James drank 13 ounces of milk. -/
theorem james_milk_consumption :
  milk_drank 3 128 371 = 13 := by
  sorry

end james_milk_consumption_l2656_265617


namespace range_of_a_l2656_265679

/-- The range of values for a given the conditions -/
theorem range_of_a (p q : ℝ → Prop) (a : ℝ) : 
  (∀ x, p x → q x) →  -- p is sufficient for q
  (∃ x, q x ∧ ¬(p x)) →  -- p is not necessary for q
  (∀ x, p x ↔ (x^2 - 2*x - 3 < 0)) →  -- definition of p
  (∀ x, q x ↔ (x > a)) →  -- definition of q
  a ≤ -1 :=
sorry

end range_of_a_l2656_265679


namespace school_visit_arrangements_l2656_265648

/-- Represents the number of days available for scheduling -/
def num_days : ℕ := 5

/-- Represents the number of schools to be scheduled -/
def num_schools : ℕ := 3

/-- Calculates the number of permutations of r items chosen from n items -/
def permutations (n r : ℕ) : ℕ :=
  if r > n then 0
  else Nat.factorial n / Nat.factorial (n - r)

/-- Calculates the number of valid arrangements for the school visits -/
def count_arrangements : ℕ :=
  permutations 4 2 + permutations 3 2 + permutations 2 2

/-- Theorem stating that the number of valid arrangements is 20 -/
theorem school_visit_arrangements :
  count_arrangements = 20 :=
sorry

end school_visit_arrangements_l2656_265648


namespace second_round_difference_l2656_265640

/-- Bowling game results -/
structure BowlingGame where
  patrick_first : ℕ
  richard_first : ℕ
  patrick_second : ℕ
  richard_second : ℕ

/-- Conditions of the bowling game -/
def bowling_conditions (game : BowlingGame) : Prop :=
  game.patrick_first = 70 ∧
  game.richard_first = game.patrick_first + 15 ∧
  game.patrick_second = 2 * game.richard_first ∧
  game.richard_second < game.patrick_second ∧
  game.richard_first + game.richard_second = game.patrick_first + game.patrick_second + 12

/-- Theorem: The difference between Patrick's and Richard's knocked down pins in the second round is 3 -/
theorem second_round_difference (game : BowlingGame) 
  (h : bowling_conditions game) : 
  game.patrick_second - game.richard_second = 3 := by
  sorry

end second_round_difference_l2656_265640


namespace cube_face_sum_l2656_265631

theorem cube_face_sum (a b c d e f : ℕ+) :
  (a * b * c + a * e * c + a * b * f + a * e * f +
   d * b * c + d * e * c + d * b * f + d * e * f) = 1287 →
  (a + d) + (b + e) + (c + f) = 33 := by
sorry

end cube_face_sum_l2656_265631


namespace like_terms_exponent_sum_l2656_265602

/-- Given that 3x^(2m)y^3 and -2x^2y^n are like terms, prove that m + n = 4 -/
theorem like_terms_exponent_sum (m n : ℕ) : 
  (∀ x y : ℝ, 3 * x^(2*m) * y^3 = -2 * x^2 * y^n) → m + n = 4 := by
  sorry

end like_terms_exponent_sum_l2656_265602


namespace tangent_circle_equation_l2656_265674

-- Define the given circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 4

-- Define the point of tangency
def tangent_point : ℝ × ℝ := (4, -1)

-- Define the radius of the new circle
def new_radius : ℝ := 1

-- Define the possible equations of the new circle
def new_circle_1 (x y : ℝ) : Prop := (x - 5)^2 + (y + 1)^2 = 1
def new_circle_2 (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 1

-- Theorem statement
theorem tangent_circle_equation : 
  ∃ (x y : ℝ), (circle_C x y ∧ (x, y) = tangent_point) → 
  (new_circle_1 x y ∨ new_circle_2 x y) :=
sorry

end tangent_circle_equation_l2656_265674


namespace tim_earnings_l2656_265660

/-- Calculates the total money earned by Tim given the number of coins received from various sources. -/
def total_money_earned (shine_pennies shine_nickels shine_dimes shine_quarters : ℕ)
                       (tip_pennies tip_nickels tip_dimes tip_half_dollars : ℕ)
                       (stranger_pennies stranger_quarters : ℕ) : ℚ :=
  let penny_value : ℚ := 1 / 100
  let nickel_value : ℚ := 5 / 100
  let dime_value : ℚ := 10 / 100
  let quarter_value : ℚ := 25 / 100
  let half_dollar_value : ℚ := 50 / 100

  let shine_total : ℚ := shine_pennies * penny_value + shine_nickels * nickel_value +
                         shine_dimes * dime_value + shine_quarters * quarter_value
  let tip_total : ℚ := tip_pennies * penny_value + tip_nickels * nickel_value +
                       tip_dimes * dime_value + tip_half_dollars * half_dollar_value
  let stranger_total : ℚ := stranger_pennies * penny_value + stranger_quarters * quarter_value

  shine_total + tip_total + stranger_total

/-- Theorem stating that Tim's total earnings equal $9.79 given the specified coin counts. -/
theorem tim_earnings :
  total_money_earned 4 3 13 6 15 12 7 9 10 3 = 979 / 100 := by
  sorry

end tim_earnings_l2656_265660


namespace primitive_root_existence_l2656_265650

theorem primitive_root_existence (p : Nat) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ g : Nat, 1 < g ∧ g < p ∧ ∀ n : Nat, n > 0 → IsPrimitiveRoot g (p^n) :=
by sorry

/- Definitions used:
Nat.Prime: Prime number predicate
IsPrimitiveRoot: Predicate for primitive root
-/

end primitive_root_existence_l2656_265650


namespace paddington_washington_goats_difference_l2656_265645

theorem paddington_washington_goats_difference 
  (washington_goats : ℕ) 
  (total_goats : ℕ) 
  (h1 : washington_goats = 140)
  (h2 : total_goats = 320)
  (h3 : washington_goats < total_goats - washington_goats) : 
  total_goats - washington_goats - washington_goats = 40 := by
  sorry

end paddington_washington_goats_difference_l2656_265645


namespace sum_of_cubes_representable_l2656_265686

theorem sum_of_cubes_representable (a b : ℤ) 
  (h1 : ∃ (x1 y1 : ℤ), a = x1^2 + 3*y1^2) 
  (h2 : ∃ (x2 y2 : ℤ), b = x2^2 + 3*y2^2) : 
  ∃ (x3 y3 : ℤ), a^3 + b^3 = x3^2 + 3*y3^2 := by
  sorry

end sum_of_cubes_representable_l2656_265686


namespace negative_values_iff_a_outside_interval_l2656_265693

/-- A quadratic function f(x) = x^2 - ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

/-- The function f takes negative values -/
def takes_negative_values (a : ℝ) : Prop :=
  ∃ x, f a x < 0

/-- The main theorem: f takes negative values iff a > 2 or a < -2 -/
theorem negative_values_iff_a_outside_interval :
  ∀ a : ℝ, takes_negative_values a ↔ (a > 2 ∨ a < -2) :=
sorry

end negative_values_iff_a_outside_interval_l2656_265693


namespace proposition_form_l2656_265628

theorem proposition_form : 
  ∃ (p q : Prop), (12 % 4 = 0 ∧ 12 % 3 = 0) ↔ (p ∧ q) :=
by sorry

end proposition_form_l2656_265628


namespace arithmetic_sequence_problem_l2656_265682

theorem arithmetic_sequence_problem (a : ℚ) : 
  a > 0 ∧ 
  (∃ d : ℚ, 140 + d = a ∧ a + d = 45/28) → 
  a = 3965/56 := by
sorry

end arithmetic_sequence_problem_l2656_265682


namespace two_numbers_problem_l2656_265605

theorem two_numbers_problem (a b : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) 
  (h4 : a + b = 6) (h5 : a / b = 6) : a * b - (a - b) = 6 / 49 := by
  sorry

end two_numbers_problem_l2656_265605


namespace square_sum_geq_product_sum_l2656_265651

theorem square_sum_geq_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end square_sum_geq_product_sum_l2656_265651


namespace diamond_calculation_l2656_265647

def diamond (A B : ℚ) : ℚ := (A - B) / 5

theorem diamond_calculation : (diamond (diamond 7 15) 2) = -18/25 := by
  sorry

end diamond_calculation_l2656_265647


namespace problem_solution_l2656_265613

theorem problem_solution : 
  (12345679^2 * 81 - 1) / 11111111 / 10 * 9 - 8 = 10000000000 := by sorry

end problem_solution_l2656_265613


namespace treys_total_time_l2656_265687

/-- The number of tasks to clean the house -/
def clean_house_tasks : ℕ := 7

/-- The number of tasks to take a shower -/
def shower_tasks : ℕ := 1

/-- The number of tasks to make dinner -/
def dinner_tasks : ℕ := 4

/-- The time in minutes to complete each task -/
def time_per_task : ℕ := 10

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem: Given the conditions, the total time to complete Trey's list is 2 hours -/
theorem treys_total_time : 
  (clean_house_tasks + shower_tasks + dinner_tasks) * time_per_task / minutes_per_hour = 2 := by
  sorry

end treys_total_time_l2656_265687


namespace system_of_equations_l2656_265678

theorem system_of_equations (x y k : ℝ) : 
  (2 * x + y = 1) → 
  (x + 2 * y = k - 2) → 
  (x - y = 2) → 
  (k = 1) := by
sorry

end system_of_equations_l2656_265678


namespace opposite_of_2023_l2656_265625

theorem opposite_of_2023 :
  ∃ y : ℤ, (2023 : ℤ) + y = 0 ∧ y = -2023 :=
by sorry

end opposite_of_2023_l2656_265625


namespace horizontal_asymptote_of_f_l2656_265633

noncomputable def f (x : ℝ) : ℝ := (8 * x^2 - 4) / (4 * x^2 + 8 * x + 3)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N, ∀ x > N, |f x - 2| < ε :=
sorry

end horizontal_asymptote_of_f_l2656_265633


namespace homework_submission_negation_l2656_265636

variable (Student : Type)
variable (inClass : Student → Prop)
variable (submittedHomework : Student → Prop)

theorem homework_submission_negation :
  (¬ ∀ s : Student, inClass s → submittedHomework s) ↔
  (∃ s : Student, inClass s ∧ ¬ submittedHomework s) :=
by sorry

end homework_submission_negation_l2656_265636
