import Mathlib

namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l506_50661

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    (4 * π * r^2 = 256 * π) →
    ((4 / 3) * π * r^3 = (2048 / 3) * π) :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l506_50661


namespace NUMINAMATH_CALUDE_smallest_four_digit_remainder_five_mod_six_l506_50684

theorem smallest_four_digit_remainder_five_mod_six : 
  ∃ (n : ℕ), 
    (1000 ≤ n ∧ n ≤ 9999) ∧ 
    (n % 6 = 5) ∧
    (∀ m, (1000 ≤ m ∧ m ≤ 9999) → (m % 6 = 5) → n ≤ m) ∧
    n = 1001 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_remainder_five_mod_six_l506_50684


namespace NUMINAMATH_CALUDE_smallest_solution_equation_l506_50663

theorem smallest_solution_equation (x : ℝ) :
  (3 * x) / (x - 2) + (2 * x^2 - 28) / x = 11 →
  x ≥ ((-1 : ℝ) - Real.sqrt 17) / 2 ∧
  (3 * (((-1 : ℝ) - Real.sqrt 17) / 2)) / (((-1 : ℝ) - Real.sqrt 17) / 2 - 2) +
  (2 * (((-1 : ℝ) - Real.sqrt 17) / 2)^2 - 28) / (((-1 : ℝ) - Real.sqrt 17) / 2) = 11 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_equation_l506_50663


namespace NUMINAMATH_CALUDE_billy_coin_count_l506_50601

/-- Represents the number of piles for each coin type -/
structure PileCount where
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat

/-- Represents the number of coins in each pile for each coin type -/
structure CoinsPerPile where
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat

/-- Calculates the total number of coins given the pile counts and coins per pile -/
def totalCoins (piles : PileCount) (coinsPerPile : CoinsPerPile) : Nat :=
  piles.quarters * coinsPerPile.quarters +
  piles.dimes * coinsPerPile.dimes +
  piles.nickels * coinsPerPile.nickels +
  piles.pennies * coinsPerPile.pennies

/-- Billy's coin sorting problem -/
theorem billy_coin_count :
  let piles : PileCount := { quarters := 3, dimes := 2, nickels := 4, pennies := 6 }
  let coinsPerPile : CoinsPerPile := { quarters := 5, dimes := 7, nickels := 3, pennies := 9 }
  totalCoins piles coinsPerPile = 95 := by
  sorry

end NUMINAMATH_CALUDE_billy_coin_count_l506_50601


namespace NUMINAMATH_CALUDE_ninth_term_is_negative_256_l506_50632

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℤ
  is_geometric : ∀ n : ℕ, ∃ q : ℤ, a (n + 1) = a n * q
  a2a5 : a 2 * a 5 = -32
  a3a4_sum : a 3 + a 4 = 4

/-- The 9th term of the geometric sequence is -256 -/
theorem ninth_term_is_negative_256 (seq : GeometricSequence) : seq.a 9 = -256 := by
  sorry

#check ninth_term_is_negative_256

end NUMINAMATH_CALUDE_ninth_term_is_negative_256_l506_50632


namespace NUMINAMATH_CALUDE_area_left_of_y_axis_is_half_l506_50621

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- Calculates the area of a parallelogram left of the y-axis -/
def areaLeftOfYAxis (p : Parallelogram) : ℝ := sorry

/-- The main theorem stating that the area left of the y-axis is half the total area -/
theorem area_left_of_y_axis_is_half (p : Parallelogram) 
  (h1 : p.A = ⟨3, 4⟩) 
  (h2 : p.B = ⟨-2, 1⟩) 
  (h3 : p.C = ⟨-5, -2⟩) 
  (h4 : p.D = ⟨0, 1⟩) : 
  areaLeftOfYAxis p = (1 / 2) * area p := by
  sorry

end NUMINAMATH_CALUDE_area_left_of_y_axis_is_half_l506_50621


namespace NUMINAMATH_CALUDE_complement_A_in_U_l506_50642

-- Define the universal set U
def U : Set ℝ := {x | x^2 ≥ 1}

-- Define set A
def A : Set ℝ := {x | Real.log (x - 1) ≤ 0}

-- State the theorem
theorem complement_A_in_U : 
  (U \ A) = {x | x ≤ -1 ∨ x = 1 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l506_50642


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l506_50647

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y - 2 = 0

-- Define the center of the circle
def center : ℝ × ℝ := (-1, 1)

-- Define the radius of the circle
def radius : ℝ := 2

-- Theorem stating that the given center and radius satisfy the circle equation
theorem circle_center_and_radius :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry


end NUMINAMATH_CALUDE_circle_center_and_radius_l506_50647


namespace NUMINAMATH_CALUDE_parabola_coefficients_l506_50668

/-- A parabola with given properties has specific coefficients -/
theorem parabola_coefficients :
  ∀ (a b c : ℝ),
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (3 = a * 4^2 + b * 4 + c) →
  (4 = -b / (2 * a)) →
  (7 = a * 2^2 + b * 2 + c) →
  a = 1 ∧ b = -8 ∧ c = 19 :=
by sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l506_50668


namespace NUMINAMATH_CALUDE_sum_of_digits_94_eights_times_94_sevens_l506_50619

/-- Represents a number with 94 repeated digits --/
def RepeatedDigit (d : ℕ) : ℕ := 
  d * (10^94 - 1) / 9

/-- Sum of digits of a natural number --/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem --/
theorem sum_of_digits_94_eights_times_94_sevens : 
  sumOfDigits (RepeatedDigit 8 * RepeatedDigit 7) = 1034 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_94_eights_times_94_sevens_l506_50619


namespace NUMINAMATH_CALUDE_tv_price_before_tax_l506_50611

theorem tv_price_before_tax (P : ℝ) : P + 0.15 * P = 1955 → P = 1700 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_before_tax_l506_50611


namespace NUMINAMATH_CALUDE_dormitory_to_city_distance_l506_50696

theorem dormitory_to_city_distance :
  ∀ (D : ℝ),
  (1 / 3 : ℝ) * D + (3 / 5 : ℝ) * D + 2 = D →
  D = 30 := by
sorry

end NUMINAMATH_CALUDE_dormitory_to_city_distance_l506_50696


namespace NUMINAMATH_CALUDE_geometric_sequence_terms_l506_50673

/-- 
Given a geometric sequence where:
- The first term is 9/8
- The last term is 1/3
- The common ratio is 2/3
This theorem proves that the number of terms in the sequence is 4.
-/
theorem geometric_sequence_terms : 
  ∀ (a : ℚ) (r : ℚ) (last : ℚ) (n : ℕ),
  a = 9/8 → r = 2/3 → last = 1/3 →
  last = a * r^(n-1) →
  n = 4 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_terms_l506_50673


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l506_50679

theorem fraction_product_simplification :
  (150 : ℚ) / 12 * 7 / 140 * 6 / 5 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l506_50679


namespace NUMINAMATH_CALUDE_total_different_movies_l506_50659

-- Define the number of people
def num_people : ℕ := 5

-- Define the number of movies watched by each person
def dalton_movies : ℕ := 15
def hunter_movies : ℕ := 19
def alex_movies : ℕ := 25
def bella_movies : ℕ := 21
def chris_movies : ℕ := 11

-- Define the number of movies watched together
def all_together : ℕ := 5
def dalton_hunter_alex : ℕ := 3
def bella_chris : ℕ := 2

-- Theorem to prove
theorem total_different_movies : 
  dalton_movies + hunter_movies + alex_movies + bella_movies + chris_movies
  - (num_people - 1) * all_together
  - (3 - 1) * dalton_hunter_alex
  - (2 - 1) * bella_chris = 63 := by
sorry

end NUMINAMATH_CALUDE_total_different_movies_l506_50659


namespace NUMINAMATH_CALUDE_grid_bottom_right_value_l506_50633

/-- Represents a 4x4 grid of rational numbers -/
def Grid := Fin 4 → Fin 4 → ℚ

/-- Checks if a sequence of 4 rational numbers forms an arithmetic progression -/
def isArithmeticSequence (s : Fin 4 → ℚ) : Prop :=
  ∃ d : ℚ, ∀ i : Fin 3, s (i + 1) - s i = d

/-- A grid satisfying the problem conditions -/
def validGrid (g : Grid) : Prop :=
  (∀ i : Fin 4, isArithmeticSequence (λ j ↦ g i j)) ∧  -- Each row is an arithmetic sequence
  (∀ j : Fin 4, isArithmeticSequence (λ i ↦ g i j)) ∧  -- Each column is an arithmetic sequence
  g 0 0 = 1 ∧ g 1 0 = 4 ∧ g 2 0 = 7 ∧ g 3 0 = 10 ∧     -- First column values
  g 2 3 = 25 ∧ g 3 2 = 36                              -- Given values in the grid

theorem grid_bottom_right_value (g : Grid) (h : validGrid g) : g 3 3 = 37 := by
  sorry

end NUMINAMATH_CALUDE_grid_bottom_right_value_l506_50633


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l506_50687

/-- The repeating decimal 0.5̄10 as a rational number -/
def repeating_decimal : ℚ := 0.5 + 0.01 / (1 - 1/100)

/-- The theorem stating that 0.5̄10 is equal to 101/198 -/
theorem repeating_decimal_as_fraction : repeating_decimal = 101 / 198 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l506_50687


namespace NUMINAMATH_CALUDE_choose_three_from_seven_l506_50669

theorem choose_three_from_seven (n : ℕ) (k : ℕ) : n = 7 ∧ k = 3 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_seven_l506_50669


namespace NUMINAMATH_CALUDE_power_product_equals_four_digit_l506_50656

/-- Given that 2^x × 9^y equals the four-digit number 2x9y, prove that x^2 * y^3 = 200 -/
theorem power_product_equals_four_digit (x y : ℕ) : 
  (2^x * 9^y = 2000 + 100*x + 10*y + 9) → 
  (1000 ≤ 2000 + 100*x + 10*y + 9) → 
  (2000 + 100*x + 10*y + 9 < 10000) → 
  x^2 * y^3 = 200 := by sorry

end NUMINAMATH_CALUDE_power_product_equals_four_digit_l506_50656


namespace NUMINAMATH_CALUDE_surface_area_after_corner_removal_l506_50651

/-- The surface area of a cube after removing smaller cubes from its corners --/
theorem surface_area_after_corner_removal (edge_length original_cube_edge : ℝ) 
  (h1 : original_cube_edge = 4)
  (h2 : edge_length = 2) :
  6 * original_cube_edge^2 = 
  6 * original_cube_edge^2 - 8 * (3 * edge_length^2 - 3 * edge_length^2) :=
by sorry

end NUMINAMATH_CALUDE_surface_area_after_corner_removal_l506_50651


namespace NUMINAMATH_CALUDE_coneSurface_is_cone_l506_50644

/-- A surface in spherical coordinates (ρ, θ, φ) defined by ρ = c sin φ, where c is a positive constant -/
def coneSurface (c : ℝ) (h : c > 0) (ρ θ φ : ℝ) : Prop :=
  ρ = c * Real.sin φ

/-- The shape described by the coneSurface is a cone -/
theorem coneSurface_is_cone (c : ℝ) (h : c > 0) :
  ∃ (cone : Set (ℝ × ℝ × ℝ)), ∀ (ρ θ φ : ℝ),
    coneSurface c h ρ θ φ ↔ (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ) ∈ cone :=
sorry

end NUMINAMATH_CALUDE_coneSurface_is_cone_l506_50644


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l506_50667

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 < 0} = Set.Ioo (-2 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l506_50667


namespace NUMINAMATH_CALUDE_part_one_part_two_l506_50612

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := |k * x - 1|

-- Part I
theorem part_one (k : ℝ) :
  (∀ x, f k x ≤ 3 ↔ x ∈ Set.Icc (-2) 1) → k = -2 := by sorry

-- Part II
theorem part_two (m : ℝ) :
  (∀ x, f 1 (x + 2) - f 1 (2 * x + 1) ≤ 3 - 2 * m) → m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l506_50612


namespace NUMINAMATH_CALUDE_problem1_l506_50657

theorem problem1 (a b : ℝ) (ha : a = -Real.sqrt 2) (hb : b = Real.sqrt 6) :
  (a + b) * (a - b) + b * (a + 2 * b) - (a + b)^2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem1_l506_50657


namespace NUMINAMATH_CALUDE_initial_water_amount_l506_50666

/-- Given a container with alcohol and water, prove the initial amount of water -/
theorem initial_water_amount (initial_alcohol : ℝ) (added_water : ℝ) (ratio_alcohol : ℝ) (ratio_water : ℝ) :
  initial_alcohol = 4 →
  added_water = 2.666666666666667 →
  ratio_alcohol = 3 →
  ratio_water = 5 →
  ratio_alcohol / ratio_water = initial_alcohol / (initial_alcohol + added_water + x) →
  x = 4 :=
by sorry

end NUMINAMATH_CALUDE_initial_water_amount_l506_50666


namespace NUMINAMATH_CALUDE_incorrect_statement_identification_l506_50627

theorem incorrect_statement_identification :
  ((-64 : ℚ)^(1/3) = -4) ∧ 
  ((49 : ℚ)^(1/2) = 7) ∧ 
  ((1/27 : ℚ)^(1/3) = 1/3) →
  ¬((1/16 : ℚ)^(1/2) = 1/4) :=
by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_identification_l506_50627


namespace NUMINAMATH_CALUDE_intersection_difference_l506_50604

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 6
def parabola2 (x : ℝ) : ℝ := -2 * x^2 + 2 * x + 6

theorem intersection_difference :
  ∃ (a c : ℝ),
    (∀ x : ℝ, parabola1 x = parabola2 x → x = a ∨ x = c) ∧
    c ≥ a ∧
    c - a = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_difference_l506_50604


namespace NUMINAMATH_CALUDE_equation_solutions_count_l506_50609

theorem equation_solutions_count :
  ∃! (s : Finset ℝ), 
    (∀ θ ∈ s, 0 < θ ∧ θ ≤ π ∧ 4 - 2 * Real.sin θ + 3 * Real.cos (2 * θ) = 0) ∧
    s.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l506_50609


namespace NUMINAMATH_CALUDE_shopkeeper_stock_worth_l506_50638

def item_A_profit_percentage : Real := 0.15
def item_A_loss_percentage : Real := 0.10
def item_A_profit_portion : Real := 0.25
def item_A_loss_portion : Real := 0.75

def item_B_profit_percentage : Real := 0.20
def item_B_loss_percentage : Real := 0.05
def item_B_profit_portion : Real := 0.30
def item_B_loss_portion : Real := 0.70

def item_C_profit_percentage : Real := 0.10
def item_C_loss_percentage : Real := 0.08
def item_C_profit_portion : Real := 0.40
def item_C_loss_portion : Real := 0.60

def tax_rate : Real := 0.12
def net_loss : Real := 750

def cost_price_ratio_A : Real := 2
def cost_price_ratio_B : Real := 3
def cost_price_ratio_C : Real := 4

theorem shopkeeper_stock_worth (x : Real) :
  let cost_A := cost_price_ratio_A * x
  let cost_B := cost_price_ratio_B * x
  let cost_C := cost_price_ratio_C * x
  let profit_loss_A := item_A_profit_portion * cost_A * item_A_profit_percentage - 
                       item_A_loss_portion * cost_A * item_A_loss_percentage
  let profit_loss_B := item_B_profit_portion * cost_B * item_B_profit_percentage - 
                       item_B_loss_portion * cost_B * item_B_loss_percentage
  let profit_loss_C := item_C_profit_portion * cost_C * item_C_profit_percentage - 
                       item_C_loss_portion * cost_C * item_C_loss_percentage
  let total_profit_loss := profit_loss_A + profit_loss_B + profit_loss_C
  total_profit_loss = -net_loss →
  cost_A = 46875 ∧ cost_B = 70312.5 ∧ cost_C = 93750 := by
sorry


end NUMINAMATH_CALUDE_shopkeeper_stock_worth_l506_50638


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_five_l506_50625

theorem floor_ceiling_sum_five (x : ℝ) :
  (⌊x⌋ + ⌈x⌉ = 5) ↔ (2 < x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_five_l506_50625


namespace NUMINAMATH_CALUDE_gcd_fraction_equality_l506_50645

theorem gcd_fraction_equality (a b c d : ℕ) (h : a * b = c * d) :
  (Nat.gcd a c * Nat.gcd a d) / Nat.gcd (Nat.gcd (Nat.gcd a b) c) d = a := by
  sorry

end NUMINAMATH_CALUDE_gcd_fraction_equality_l506_50645


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l506_50688

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a * b ≤ ((a + b) / 2) * Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l506_50688


namespace NUMINAMATH_CALUDE_grocery_store_soda_bottles_l506_50630

/-- 
Given a grocery store with regular and diet soda bottles, this theorem proves 
the number of diet soda bottles, given the number of regular soda bottles and 
the difference between regular and diet soda bottles.
-/
theorem grocery_store_soda_bottles 
  (regular_soda : ℕ) 
  (difference : ℕ) 
  (h1 : regular_soda = 67)
  (h2 : regular_soda = difference + diet_soda) : 
  diet_soda = 9 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_soda_bottles_l506_50630


namespace NUMINAMATH_CALUDE_rectangle_area_l506_50685

/-- Given a square with side length 15 and a rectangle with length 18 and diagonal 27,
    prove that the area of the rectangle is 216 when its perimeter equals the square's perimeter. -/
theorem rectangle_area (square_side : ℝ) (rect_length rect_diagonal : ℝ) :
  square_side = 15 →
  rect_length = 18 →
  rect_diagonal = 27 →
  4 * square_side = 2 * rect_length + 2 * (rect_diagonal ^ 2 - rect_length ^ 2).sqrt →
  rect_length * (rect_diagonal ^ 2 - rect_length ^ 2).sqrt = 216 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l506_50685


namespace NUMINAMATH_CALUDE_coin_black_region_probability_l506_50643

/-- The probability of a coin partially covering a black region on a specially painted square. -/
theorem coin_black_region_probability : 
  let square_side : ℝ := 10
  let triangle_leg : ℝ := 3
  let diamond_side : ℝ := 3 * Real.sqrt 2
  let coin_diameter : ℝ := 2
  let valid_region_side : ℝ := square_side - coin_diameter
  let valid_region_area : ℝ := valid_region_side ^ 2
  let triangle_area : ℝ := 1/2 * triangle_leg ^ 2
  let diamond_area : ℝ := diamond_side ^ 2
  let overlap_area : ℝ := 48 + 4 * Real.sqrt 2 + 2 * Real.pi
  overlap_area / valid_region_area = (48 + 4 * Real.sqrt 2 + 2 * Real.pi) / 64 := by
  sorry

end NUMINAMATH_CALUDE_coin_black_region_probability_l506_50643


namespace NUMINAMATH_CALUDE_dvd_discount_l506_50639

/-- The discount on each pack of DVDs, given the original price and the discounted price for multiple packs. -/
theorem dvd_discount (original_price : ℕ) (num_packs : ℕ) (total_price : ℕ) : 
  original_price = 107 → num_packs = 93 → total_price = 93 → 
  (original_price - (total_price / num_packs) : ℕ) = 106 :=
by sorry

end NUMINAMATH_CALUDE_dvd_discount_l506_50639


namespace NUMINAMATH_CALUDE_antonios_meatballs_l506_50653

/-- Given the conditions of Antonio's meatball preparation, prove the amount of hamburger per meatball. -/
theorem antonios_meatballs (family_members : ℕ) (total_hamburger : ℝ) (antonios_meatballs : ℕ) :
  family_members = 8 →
  total_hamburger = 4 →
  antonios_meatballs = 4 →
  (total_hamburger / (family_members * antonios_meatballs) : ℝ) = 0.125 := by
  sorry

#check antonios_meatballs

end NUMINAMATH_CALUDE_antonios_meatballs_l506_50653


namespace NUMINAMATH_CALUDE_function_and_tangent_line_l506_50641

-- Define the function f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

-- State the theorem
theorem function_and_tangent_line 
  (a b c : ℝ) 
  (h1 : ∃ (x : ℝ), x = 2 ∧ (3 * a * x^2 + 2 * b * x + c = 0)) -- extremum at x = 2
  (h2 : f a b c 2 = -6) -- f(2) = -6
  (h3 : c = -4) -- f'(0) = -4
  : 
  (∀ x, f a b c x = x^3 - 2*x^2 - 4*x + 2) ∧ -- f(x) = x³ - 2x² - 4x + 2
  (∃ (m n : ℝ), m = 3 ∧ n = 6 ∧ ∀ x y, y = (f a b c (-1)) + (3 * a * (-1)^2 + 2 * b * (-1) + c) * (x - (-1)) ↔ m * x - y + n = 0) -- Tangent line equation at x = -1
  := by sorry

end NUMINAMATH_CALUDE_function_and_tangent_line_l506_50641


namespace NUMINAMATH_CALUDE_geometric_series_sum_l506_50694

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  let a : ℚ := 2
  let r : ℚ := 2/5
  let n : ℕ := 5
  geometric_sum a r n = 10310/3125 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l506_50694


namespace NUMINAMATH_CALUDE_lamp_game_solvable_l506_50680

/-- Represents a move in the lamp game -/
inductive Move
  | row (r : Nat) (start : Nat)
  | col (c : Nat) (start : Nat)

/-- The lamp game state -/
def LampGame (n m : Nat) :=
  { grid : Fin n → Fin n → Bool // n > 0 ∧ m > 0 }

/-- Applies a move to the game state -/
def applyMove (game : LampGame n m) (move : Move) : LampGame n m :=
  sorry

/-- Checks if all lamps are on -/
def allOn (game : LampGame n m) : Prop :=
  ∀ i j, game.val i j = true

/-- Main theorem: all lamps can be turned on iff m divides n -/
theorem lamp_game_solvable (n m : Nat) :
  (∃ (game : LampGame n m) (moves : List Move), allOn (moves.foldl applyMove game)) ↔ m ∣ n :=
  sorry

end NUMINAMATH_CALUDE_lamp_game_solvable_l506_50680


namespace NUMINAMATH_CALUDE_apps_deleted_proof_l506_50698

/-- The number of apps Dave had at the start -/
def initial_apps : ℕ := 23

/-- The number of apps Dave had after deleting some -/
def remaining_apps : ℕ := 5

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := initial_apps - remaining_apps

theorem apps_deleted_proof : deleted_apps = 18 := by
  sorry

end NUMINAMATH_CALUDE_apps_deleted_proof_l506_50698


namespace NUMINAMATH_CALUDE_largest_number_in_systematic_sample_l506_50655

/-- The largest number in a systematic sample --/
theorem largest_number_in_systematic_sample :
  let population_size : ℕ := 60
  let sample_size : ℕ := 10
  let remainder : ℕ := 3
  let divisor : ℕ := 6
  let sampling_interval : ℕ := population_size / sample_size
  let first_sample : ℕ := remainder
  let last_sample : ℕ := first_sample + sampling_interval * (sample_size - 1)
  last_sample = 57 := by sorry

end NUMINAMATH_CALUDE_largest_number_in_systematic_sample_l506_50655


namespace NUMINAMATH_CALUDE_expression_simplification_l506_50624

theorem expression_simplification (x y z : ℝ) : 
  ((x + z) - (y - 2*z)) - ((x - 2*z) - (y + z)) = 6*z := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l506_50624


namespace NUMINAMATH_CALUDE_cody_candy_count_l506_50637

/-- The number of boxes of chocolate candy Cody bought -/
def chocolate_boxes : ℕ := 7

/-- The number of boxes of caramel candy Cody bought -/
def caramel_boxes : ℕ := 3

/-- The number of candy pieces in each box -/
def pieces_per_box : ℕ := 8

/-- The total number of candy pieces Cody has -/
def total_candy : ℕ := (chocolate_boxes + caramel_boxes) * pieces_per_box

theorem cody_candy_count : total_candy = 80 := by
  sorry

end NUMINAMATH_CALUDE_cody_candy_count_l506_50637


namespace NUMINAMATH_CALUDE_fraction_subtraction_l506_50623

theorem fraction_subtraction : (9 : ℚ) / 23 - (5 : ℚ) / 69 = (22 : ℚ) / 69 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l506_50623


namespace NUMINAMATH_CALUDE_inequality_proof_l506_50681

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (sum_eq_two : a + b + c + d = 2) :
  (a^2 / (a^2 + 1)^2) + (b^2 / (b^2 + 1)^2) + 
  (c^2 / (c^2 + 1)^2) + (d^2 / (d^2 + 1)^2) ≤ 16/25 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l506_50681


namespace NUMINAMATH_CALUDE_max_value_product_max_value_achieved_l506_50649

theorem max_value_product (x y z w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_w : w > 0)
  (sum_one : x + y + z + w = 1) :
  x^2 * y^2 * z^2 * w ≤ 64 / 823543 := by
sorry

theorem max_value_achieved (x y z w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_w : w > 0)
  (sum_one : x + y + z + w = 1) :
  ∃ x y z w, x^2 * y^2 * z^2 * w = 64 / 823543 ∧ 
             x + y + z + w = 1 ∧ 
             x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 := by
sorry

end NUMINAMATH_CALUDE_max_value_product_max_value_achieved_l506_50649


namespace NUMINAMATH_CALUDE_combined_height_sara_joe_l506_50678

/-- The combined height of Sara and Joe is 120 inches -/
theorem combined_height_sara_joe : 
  ∀ (sara_height joe_height : ℕ),
  joe_height = 2 * sara_height + 6 →
  joe_height = 82 →
  sara_height + joe_height = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_combined_height_sara_joe_l506_50678


namespace NUMINAMATH_CALUDE_square_area_with_four_circles_l506_50693

theorem square_area_with_four_circles (r : ℝ) (h : r = 7) : 
  let side_length := 4 * r
  (side_length ^ 2 : ℝ) = 784 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_four_circles_l506_50693


namespace NUMINAMATH_CALUDE_chairs_to_hall_l506_50631

/-- Calculates the total number of chairs taken to the hall given the number of students,
    chairs per trip, and number of trips. -/
def totalChairs (students : ℕ) (chairsPerTrip : ℕ) (numTrips : ℕ) : ℕ :=
  students * chairsPerTrip * numTrips

/-- Proves that 5 students, each carrying 5 chairs per trip and making 10 trips,
    will take a total of 250 chairs to the hall. -/
theorem chairs_to_hall :
  totalChairs 5 5 10 = 250 := by
  sorry

#eval totalChairs 5 5 10

end NUMINAMATH_CALUDE_chairs_to_hall_l506_50631


namespace NUMINAMATH_CALUDE_triangle_area_ratio_l506_50628

/-- Given a right triangle with a point on its hypotenuse and lines drawn parallel to the legs,
    dividing it into a rectangle and two smaller right triangles, this theorem states the
    relationship between the areas of the smaller triangles and the rectangle. -/
theorem triangle_area_ratio (a b m : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_m : m > 0) :
  let rectangle_area := a * b
  let small_triangle1_area := m * rectangle_area
  let small_triangle2_area := (b^2) / (4 * m)
  (small_triangle2_area / rectangle_area) = b / (4 * m * a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_l506_50628


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l506_50614

/-- The radius of an inscribed circle in a sector that is one-third of a larger circle -/
theorem inscribed_circle_radius (R : ℝ) (h : R = 6) :
  let sector_angle : ℝ := 2 * Real.pi / 3
  let inscribed_radius : ℝ := R * (Real.sqrt 2 - 1)
  inscribed_radius = R * (Real.sqrt 2 - 1) := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l506_50614


namespace NUMINAMATH_CALUDE_probability_green_ball_l506_50640

/-- The probability of drawing a green ball from a bag with specified contents -/
theorem probability_green_ball (green black red : ℕ) : 
  green = 3 → black = 3 → red = 6 → 
  (green : ℚ) / (green + black + red : ℚ) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_green_ball_l506_50640


namespace NUMINAMATH_CALUDE_meeting_point_x_coordinate_l506_50602

-- Define the river boundaries
def river_left : ℝ := 0
def river_right : ℝ := 25

-- Define the current speed
def current_speed : ℝ := 2

-- Define the starting positions
def mallard_start : ℝ × ℝ := (0, 0)
def wigeon_start : ℝ × ℝ := (25, 0)

-- Define the meeting point y-coordinate
def meeting_y : ℝ := 22

-- Define the speeds relative to water
def mallard_speed : ℝ := 4
def wigeon_speed : ℝ := 3

-- Theorem statement
theorem meeting_point_x_coordinate :
  ∃ (x : ℝ), 
    x > river_left ∧ 
    x < river_right ∧ 
    (∃ (t : ℝ), t > 0 ∧
      (mallard_start.1 + mallard_speed * t * Real.cos (Real.arctan ((meeting_y - mallard_start.2) / (x - mallard_start.1))) = x) ∧
      (wigeon_start.1 - wigeon_speed * t * Real.cos (Real.arctan ((meeting_y - wigeon_start.2) / (wigeon_start.1 - x))) = x) ∧
      (mallard_start.2 + (mallard_speed * Real.sin (Real.arctan ((meeting_y - mallard_start.2) / (x - mallard_start.1))) + current_speed) * t = meeting_y) ∧
      (wigeon_start.2 + (wigeon_speed * Real.sin (Real.arctan ((meeting_y - wigeon_start.2) / (wigeon_start.1 - x))) + current_speed) * t = meeting_y)) ∧
    x = 100 / 7 := by
  sorry

end NUMINAMATH_CALUDE_meeting_point_x_coordinate_l506_50602


namespace NUMINAMATH_CALUDE_triangle_minimum_area_l506_50613

theorem triangle_minimum_area :
  ∀ (S : ℝ), 
  (∀ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 →
    a + b > c ∧ b + c > a ∧ c + a > b →
    (∃ (h : ℝ), h ≤ 1 ∧ S = 1/2 * (a * h)) →
    (∀ (w : ℝ), w < 1 → 
      ¬(∃ (h : ℝ), h ≤ w ∧ S = 1/2 * (a * h)))) →
  S ≥ 1 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_minimum_area_l506_50613


namespace NUMINAMATH_CALUDE_ladies_walking_distance_l506_50650

/-- The total distance walked by a group of ladies over a period of days. -/
def total_distance (
  group_size : ℕ
  ) (
  group_distance : ℝ
  ) (
  jamie_extra : ℝ
  ) (
  sue_extra : ℝ
  ) (
  days : ℕ
  ) : ℝ :=
  group_size * group_distance * days + jamie_extra * days + sue_extra * days

/-- Proof that the total distance walked by the ladies is 36 miles. -/
theorem ladies_walking_distance :
  let group_size : ℕ := 5
  let group_distance : ℝ := 3
  let jamie_extra : ℝ := 2
  let sue_extra : ℝ := jamie_extra / 2
  let days : ℕ := 6
  total_distance group_size group_distance jamie_extra sue_extra days = 36 := by
  sorry


end NUMINAMATH_CALUDE_ladies_walking_distance_l506_50650


namespace NUMINAMATH_CALUDE_lcm_e_n_l506_50665

theorem lcm_e_n (e n : ℕ) (h1 : e > 0) (h2 : n ≥ 100 ∧ n ≤ 999) 
  (h3 : ¬(3 ∣ n)) (h4 : ¬(2 ∣ e)) (h5 : n = 230) : 
  Nat.lcm e n = 230 := by
  sorry

end NUMINAMATH_CALUDE_lcm_e_n_l506_50665


namespace NUMINAMATH_CALUDE_min_k_for_three_or_more_intersections_range_of_ratio_for_four_intersections_l506_50636

-- Define the curve M and line l
def curve_M (x y : ℝ) : Prop := (x^2 = -y) ∨ (x^2 = 4*y)
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*x - 3

-- Define the number of intersection points
def intersection_points (k : ℝ) : ℕ := sorry

-- Theorem 1: Minimum value of k when m ≥ 3
theorem min_k_for_three_or_more_intersections :
  ∀ k : ℝ, k > 0 → intersection_points k ≥ 3 → k ≥ Real.sqrt 3 := by sorry

-- Theorem 2: Range of |AB|/|CD| when m = 4
theorem range_of_ratio_for_four_intersections :
  ∀ k : ℝ, k > 0 → intersection_points k = 4 →
  ∃ r : ℝ, 0 < r ∧ r < 4 ∧
  (∃ A B C D : ℝ × ℝ,
    curve_M A.1 A.2 ∧ curve_M B.1 B.2 ∧ curve_M C.1 C.2 ∧ curve_M D.1 D.2 ∧
    line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧ line_l k C.1 C.2 ∧ line_l k D.1 D.2 ∧
    r = (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) /
        (Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2))) := by sorry

end NUMINAMATH_CALUDE_min_k_for_three_or_more_intersections_range_of_ratio_for_four_intersections_l506_50636


namespace NUMINAMATH_CALUDE_insurance_cost_calculation_l506_50670

/-- Calculates the total annual insurance cost given quarterly, monthly, annual, and semi-annual payments -/
def total_annual_insurance_cost (car_quarterly : ℕ) (home_monthly : ℕ) (health_annual : ℕ) (life_semiannual : ℕ) : ℕ :=
  car_quarterly * 4 + home_monthly * 12 + health_annual + life_semiannual * 2

/-- Theorem stating that given specific insurance costs, the total annual cost is 8757 -/
theorem insurance_cost_calculation :
  total_annual_insurance_cost 378 125 5045 850 = 8757 := by
  sorry

end NUMINAMATH_CALUDE_insurance_cost_calculation_l506_50670


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l506_50603

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : ArithmeticSequence a)
  (h_condition : a 1 + 3 * a 8 + a 15 = 120) :
  2 * a 6 - a 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l506_50603


namespace NUMINAMATH_CALUDE_service_charge_percentage_l506_50671

theorem service_charge_percentage (salmon_cost black_burger_cost chicken_katsu_cost : ℝ)
  (tip_percentage : ℝ) (total_paid change : ℝ) :
  salmon_cost = 40 →
  black_burger_cost = 15 →
  chicken_katsu_cost = 25 →
  tip_percentage = 0.05 →
  total_paid = 100 →
  change = 8 →
  let food_cost := salmon_cost + black_burger_cost + chicken_katsu_cost
  let tip := food_cost * tip_percentage
  let service_charge := total_paid - change - food_cost - tip
  service_charge / food_cost = 0.1 := by
sorry

end NUMINAMATH_CALUDE_service_charge_percentage_l506_50671


namespace NUMINAMATH_CALUDE_semicircle_area_ratio_l506_50691

/-- The ratio of the combined areas of two semicircles to the area of their circumscribing circle -/
theorem semicircle_area_ratio (r : ℝ) (h : r > 0) :
  let semicircle_area := π * (r / 2)^2 / 2
  let circle_area := π * r^2
  2 * semicircle_area / circle_area = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_ratio_l506_50691


namespace NUMINAMATH_CALUDE_problem_G10_1_l506_50615

theorem problem_G10_1 (a : ℝ) : 
  (6 * Real.sqrt 3) / (3 * Real.sqrt 2 - 2 * Real.sqrt 3) = 3 * Real.sqrt a + 6 → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_G10_1_l506_50615


namespace NUMINAMATH_CALUDE_max_value_a_l506_50692

theorem max_value_a (a : ℝ) : 
  (∀ x k, x ∈ Set.Ioo 0 6 → k ∈ Set.Icc (-1) 1 → 
    6 * Real.log x + x^2 - 8*x + a ≤ k*x) → 
  a ≤ 6 - 6 * Real.log 6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_l506_50692


namespace NUMINAMATH_CALUDE_opposite_of_2023_l506_50676

theorem opposite_of_2023 : 
  ∃ x : ℤ, (2023 + x = 0) ∧ (x = -2023) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l506_50676


namespace NUMINAMATH_CALUDE_function_composition_ratio_l506_50672

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := 3 * x - 2

-- State the theorem
theorem function_composition_ratio :
  (f (g (f 3))) / (g (f (g 3))) = 53 / 49 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_ratio_l506_50672


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l506_50682

/-- A hyperbola with the property that the distance from its vertex to its asymptote
    is 1/4 of the length of its imaginary axis has eccentricity 2. -/
theorem hyperbola_eccentricity (a b : ℝ) (h : a > 0) (k : b > 0) :
  (a * b / Real.sqrt (a^2 + b^2) = 1/4 * (2*b)) → (Real.sqrt (a^2 + b^2) / a = 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l506_50682


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l506_50660

/-- Calculates the perimeter of a T-shaped figure formed by two rectangles -/
def t_perimeter (rect1_width rect1_height rect2_width rect2_height overlap : ℕ) : ℕ :=
  2 * (rect1_width + rect1_height) + 2 * (rect2_width + rect2_height) - 2 * overlap

/-- The perimeter of the T-shaped figure is 26 inches -/
theorem t_shape_perimeter : t_perimeter 3 5 2 5 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_l506_50660


namespace NUMINAMATH_CALUDE_square_diagonal_l506_50648

/-- The diagonal of a square with perimeter 800 cm is 200√2 cm. -/
theorem square_diagonal (perimeter : ℝ) (side : ℝ) (diagonal : ℝ) : 
  perimeter = 800 →
  side = perimeter / 4 →
  diagonal = side * Real.sqrt 2 →
  diagonal = 200 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_l506_50648


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_total_bill_is_140_l506_50620

/-- Calculates the total bill for a restaurant order with given conditions -/
theorem restaurant_bill_calculation 
  (tax_rate : ℝ) 
  (striploin_cost : ℝ) 
  (wine_cost : ℝ) 
  (gratuities : ℝ) : ℝ :=
  let total_before_tax := striploin_cost + wine_cost
  let tax_amount := tax_rate * total_before_tax
  let total_after_tax := total_before_tax + tax_amount
  let total_bill := total_after_tax + gratuities
  total_bill

/-- Proves that the total bill is $140 given the specified conditions -/
theorem total_bill_is_140 : 
  restaurant_bill_calculation 0.1 80 10 41 = 140 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_total_bill_is_140_l506_50620


namespace NUMINAMATH_CALUDE_jessica_scores_mean_l506_50664

def jessica_scores : List ℝ := [87, 94, 85, 92, 90, 88]

theorem jessica_scores_mean :
  (jessica_scores.sum / jessica_scores.length : ℝ) = 89.3333333333333 := by
  sorry

end NUMINAMATH_CALUDE_jessica_scores_mean_l506_50664


namespace NUMINAMATH_CALUDE_base_c_is_seven_l506_50629

theorem base_c_is_seven (c : ℕ) (h : c > 1) : 
  (3 * c + 2)^2 = c^3 + 2 * c^2 + 6 * c + 4 → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_base_c_is_seven_l506_50629


namespace NUMINAMATH_CALUDE_rectangle_coverage_l506_50605

/-- A shape composed of 6 unit squares -/
structure Shape :=
  (area : ℕ)
  (h_area : area = 6)

/-- A rectangle with dimensions m × n -/
structure Rectangle (m n : ℕ) :=
  (width : ℕ)
  (height : ℕ)
  (h_width : width = m)
  (h_height : height = n)

/-- Predicate for a rectangle that can be covered by shapes -/
def is_coverable (m n : ℕ) : Prop :=
  (3 ∣ m ∧ 4 ∣ n) ∨ (3 ∣ n ∧ 4 ∣ m) ∨ (12 ∣ m ∧ 12 ∣ n)

theorem rectangle_coverage (m n : ℕ) (hm : m ≠ 1 ∧ m ≠ 2 ∧ m ≠ 5) (hn : n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 5) :
  ∃ (s : Shape), ∃ (r : Rectangle m n), is_coverable m n ↔ 
    (∃ (arrangement : ℕ → ℕ → Shape), 
      (∀ i j, i < m ∧ j < n → (arrangement i j).area = 6) ∧
      (∀ i j, i < m ∧ j < n → ∃ k l, k < m ∧ l < n ∧ arrangement i j = arrangement k l) ∧
      (∀ i j k l, i < m ∧ j < n ∧ k < m ∧ l < n → 
        (i ≠ k ∨ j ≠ l) → arrangement i j ≠ arrangement k l)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_coverage_l506_50605


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_integer_l506_50616

theorem sum_and_reciprocal_integer (a : ℝ) (h : ∃ k : ℤ, a + 1/a = k) :
  (∃ m : ℤ, a^2 + 1/a^2 = m) ∧ (∀ n : ℕ, ∃ l : ℤ, a^n + 1/a^n = l) := by
  sorry


end NUMINAMATH_CALUDE_sum_and_reciprocal_integer_l506_50616


namespace NUMINAMATH_CALUDE_binomial_27_6_l506_50697

theorem binomial_27_6 (h1 : Nat.choose 26 4 = 14950)
                      (h2 : Nat.choose 26 5 = 65780)
                      (h3 : Nat.choose 26 6 = 230230) :
  Nat.choose 27 6 = 296010 := by
  sorry

end NUMINAMATH_CALUDE_binomial_27_6_l506_50697


namespace NUMINAMATH_CALUDE_renovation_project_dirt_required_l506_50635

theorem renovation_project_dirt_required (sand cement total : ℚ)
  (h1 : sand = 0.16666666666666666)
  (h2 : cement = 0.16666666666666666)
  (h3 : total = 0.6666666666666666) :
  total - (sand + cement) = 0.3333333333333333 :=
by sorry

end NUMINAMATH_CALUDE_renovation_project_dirt_required_l506_50635


namespace NUMINAMATH_CALUDE_max_value_d_l506_50683

theorem max_value_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10)
  (sum_prod_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 ∧ 
  ∃ a b c d, a + b + c + d = 10 ∧ 
             a*b + a*c + a*d + b*c + b*d + c*d = 20 ∧
             d = (5 + Real.sqrt 105) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_d_l506_50683


namespace NUMINAMATH_CALUDE_infinite_squares_l506_50699

theorem infinite_squares (k : ℕ) (hk : k ≥ 2) :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ n ∈ S, ∃ (u v : ℕ), k * n + 1 = u^2 ∧ (k + 1) * n + 1 = v^2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_squares_l506_50699


namespace NUMINAMATH_CALUDE_golf_carts_needed_l506_50690

def patrons_from_cars : ℕ := 12
def patrons_from_bus : ℕ := 27
def golf_cart_capacity : ℕ := 3

theorem golf_carts_needed : 
  (patrons_from_cars + patrons_from_bus + golf_cart_capacity - 1) / golf_cart_capacity = 13 := by
  sorry

end NUMINAMATH_CALUDE_golf_carts_needed_l506_50690


namespace NUMINAMATH_CALUDE_modular_inverse_15_mod_17_l506_50618

theorem modular_inverse_15_mod_17 : ∃ x : ℕ, x ≤ 16 ∧ (15 * x) % 17 = 1 :=
by
  use 9
  sorry

end NUMINAMATH_CALUDE_modular_inverse_15_mod_17_l506_50618


namespace NUMINAMATH_CALUDE_distance_per_block_l506_50626

/-- Proves that the distance of each block is 1/8 mile -/
theorem distance_per_block (total_time : ℚ) (total_blocks : ℕ) (speed : ℚ) :
  total_time = 10 / 60 →
  total_blocks = 16 →
  speed = 12 →
  (speed * total_time) / total_blocks = 1 / 8 := by
  sorry

#check distance_per_block

end NUMINAMATH_CALUDE_distance_per_block_l506_50626


namespace NUMINAMATH_CALUDE_stock_price_loss_l506_50686

theorem stock_price_loss (n : ℕ) (P : ℝ) (h : P > 0) : 
  P * (1.1 ^ n) * (0.9 ^ n) < P := by
  sorry

#check stock_price_loss

end NUMINAMATH_CALUDE_stock_price_loss_l506_50686


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l506_50662

theorem solution_set_quadratic_inequality (a : ℝ) :
  {x : ℝ | x^2 - 2*a + a^2 - 1 < 0} = {x : ℝ | a - 1 < x ∧ x < a + 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l506_50662


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l506_50652

/-- The number of available colors for glass panes -/
def num_colors : ℕ := 10

/-- The number of panes in the window frame -/
def num_panes : ℕ := 4

/-- A function that calculates the number of valid arrangements -/
def valid_arrangements (colors : ℕ) (panes : ℕ) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that the number of valid arrangements is 3430 -/
theorem valid_arrangements_count :
  valid_arrangements num_colors num_panes = 3430 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l506_50652


namespace NUMINAMATH_CALUDE_f_min_value_f_max_value_tangent_line_equation_l506_50607

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem for the minimum value
theorem f_min_value : ∃ x₀ ∈ Set.Icc (-2 : ℝ) 1, ∀ x ∈ Set.Icc (-2 : ℝ) 1, f x₀ ≤ f x ∧ f x₀ = -2 := by sorry

-- Theorem for the maximum value
theorem f_max_value : ∃ x₀ ∈ Set.Icc (-2 : ℝ) 1, ∀ x ∈ Set.Icc (-2 : ℝ) 1, f x ≤ f x₀ ∧ f x₀ = 2 := by sorry

-- Theorem for the tangent line equation
theorem tangent_line_equation : 
  let P : ℝ × ℝ := (2, -6)
  let tangent_line (x y : ℝ) : Prop := 24 * x - y - 54 = 0
  ∀ x y : ℝ, tangent_line x y ↔ (y - f P.1 = (3 * P.1^2 - 3) * (x - P.1)) := by sorry

end NUMINAMATH_CALUDE_f_min_value_f_max_value_tangent_line_equation_l506_50607


namespace NUMINAMATH_CALUDE_center_of_mass_position_l506_50608

/-- A system of disks with specific properties -/
structure DiskSystem where
  -- The ratio of radii of two adjacent disks
  ratio : ℝ
  -- The radius of the largest disk
  largest_radius : ℝ
  -- Assertion that the ratio is 1/2
  ratio_is_half : ratio = 1/2
  -- Assertion that the largest radius is 2 meters
  largest_radius_is_two : largest_radius = 2

/-- The center of mass of the disk system -/
noncomputable def center_of_mass (ds : DiskSystem) : ℝ := sorry

/-- Theorem stating that the center of mass is at 6/7 meters from the largest disk's center -/
theorem center_of_mass_position (ds : DiskSystem) : 
  center_of_mass ds = 6/7 := by sorry

end NUMINAMATH_CALUDE_center_of_mass_position_l506_50608


namespace NUMINAMATH_CALUDE_expression_value_l506_50695

theorem expression_value : (3^2 - 3) - (5^2 - 5) * 2 + (6^2 - 6) = -4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l506_50695


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l506_50634

theorem hot_dogs_remainder : 34582918 % 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l506_50634


namespace NUMINAMATH_CALUDE_no_xyz_single_double_triple_digits_l506_50646

theorem no_xyz_single_double_triple_digits :
  ¬∃ (x y z : ℕ+),
    (1 : ℚ) / x = 1 / y + 1 / z ∧
    ((1 ≤ x ∧ x < 10) ∨ (1 ≤ y ∧ y < 10) ∨ (1 ≤ z ∧ z < 10)) ∧
    ((10 ≤ x ∧ x < 100) ∨ (10 ≤ y ∧ y < 100) ∨ (10 ≤ z ∧ z < 100)) ∧
    ((100 ≤ x ∧ x < 1000) ∨ (100 ≤ y ∧ y < 1000) ∨ (100 ≤ z ∧ z < 1000)) :=
by sorry

end NUMINAMATH_CALUDE_no_xyz_single_double_triple_digits_l506_50646


namespace NUMINAMATH_CALUDE_high_school_total_students_l506_50675

/-- Represents a high school with three grades in its senior section. -/
structure HighSchool where
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ

/-- Represents a stratified sample from the high school. -/
structure Sample where
  total : ℕ
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ

/-- The total number of students in the high school section. -/
def total_students (hs : HighSchool) : ℕ :=
  hs.first_grade + hs.second_grade + hs.third_grade

/-- The theorem stating the total number of students in the high school section. -/
theorem high_school_total_students 
  (hs : HighSchool)
  (sample : Sample)
  (h1 : hs.first_grade = 400)
  (h2 : sample.total = 45)
  (h3 : sample.second_grade = 15)
  (h4 : sample.third_grade = 10)
  (h5 : sample.first_grade = sample.total - sample.second_grade - sample.third_grade)
  (h6 : sample.first_grade * hs.first_grade = sample.total * 20) :
  total_students hs = 900 := by
  sorry


end NUMINAMATH_CALUDE_high_school_total_students_l506_50675


namespace NUMINAMATH_CALUDE_common_tangents_count_l506_50610

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 16*y - 48 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 8*y - 44 = 0

-- Define the function to count common tangents
def count_common_tangents (c1 c2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  count_common_tangents circle1 circle2 = 2 := by sorry

end NUMINAMATH_CALUDE_common_tangents_count_l506_50610


namespace NUMINAMATH_CALUDE_marble_probability_difference_l506_50689

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1500

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 2000

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def P_s : ℚ := (red_marbles * (red_marbles - 1) + black_marbles * (black_marbles - 1)) / (total_marbles * (total_marbles - 1))

/-- The probability of drawing two marbles of different colors -/
def P_d : ℚ := (2 * red_marbles * black_marbles) / (total_marbles * (total_marbles - 1))

/-- Theorem stating that the absolute difference between P_s and P_d is 1/50 -/
theorem marble_probability_difference : |P_s - P_d| = 1 / 50 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_difference_l506_50689


namespace NUMINAMATH_CALUDE_tan_30_plus_4cos_30_l506_50622

theorem tan_30_plus_4cos_30 : Real.tan (30 * π / 180) + 4 * Real.cos (30 * π / 180) = 7 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_30_plus_4cos_30_l506_50622


namespace NUMINAMATH_CALUDE_other_number_is_eight_l506_50654

theorem other_number_is_eight (x y : ℤ) (h1 : 2 * x + 3 * y = 100) 
  (h2 : x = 28 ∨ y = 28) : (x ≠ 28 → x = 8) ∧ (y ≠ 28 → y = 8) := by
  sorry

end NUMINAMATH_CALUDE_other_number_is_eight_l506_50654


namespace NUMINAMATH_CALUDE_vincent_stickers_l506_50606

theorem vincent_stickers (yesterday : ℕ) (extra_today : ℕ) : 
  yesterday = 15 → extra_today = 10 → yesterday + (yesterday + extra_today) = 40 := by
  sorry

end NUMINAMATH_CALUDE_vincent_stickers_l506_50606


namespace NUMINAMATH_CALUDE_specific_arrangement_probability_l506_50600

def total_lamps : ℕ := 8
def red_lamps : ℕ := 4
def blue_lamps : ℕ := 4
def lamps_on : ℕ := 4

def ways_to_arrange_colors : ℕ := Nat.choose total_lamps red_lamps
def ways_to_turn_on : ℕ := Nat.choose total_lamps lamps_on

def remaining_positions : ℕ := 5
def remaining_red : ℕ := 3
def remaining_blue : ℕ := 2
def remaining_on : ℕ := 2

def ways_to_arrange_remaining : ℕ := Nat.choose remaining_positions remaining_red
def ways_to_turn_on_remaining : ℕ := Nat.choose remaining_positions remaining_on

theorem specific_arrangement_probability :
  (ways_to_arrange_remaining * ways_to_turn_on_remaining : ℚ) / 
  (ways_to_arrange_colors * ways_to_turn_on) = 1 / 49 := by
  sorry

end NUMINAMATH_CALUDE_specific_arrangement_probability_l506_50600


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l506_50658

theorem solve_quadratic_equation (x : ℝ) : 3 * (x + 1)^2 = 27 → x = 2 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l506_50658


namespace NUMINAMATH_CALUDE_ball_distribution_proof_l506_50677

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem ball_distribution_proof (total_balls : ℕ) (box1_num box2_num : ℕ) : 
  total_balls = 7 ∧ box1_num = 2 ∧ box2_num = 3 →
  (choose total_balls box1_num.succ.succ) + 
  (choose total_balls box1_num.succ) + 
  (choose total_balls box1_num) = 91 := by
sorry

end NUMINAMATH_CALUDE_ball_distribution_proof_l506_50677


namespace NUMINAMATH_CALUDE_approx_625_to_four_fifths_l506_50674

-- Define the problem
theorem approx_625_to_four_fifths : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  |Real.rpow 625 (4/5) - 238| < ε :=
sorry

end NUMINAMATH_CALUDE_approx_625_to_four_fifths_l506_50674


namespace NUMINAMATH_CALUDE_trains_crossing_time_l506_50617

/-- The time taken for two trains to cross each other -/
theorem trains_crossing_time (length1 length2 speed1 speed2 : ℝ) : 
  length1 = 500 → 
  length2 = 750 → 
  speed1 = 60 → 
  speed2 = 40 → 
  (((length1 + length2) / ((speed1 + speed2) * (5/18))) : ℝ) = 45 := by
  sorry

#check trains_crossing_time

end NUMINAMATH_CALUDE_trains_crossing_time_l506_50617
