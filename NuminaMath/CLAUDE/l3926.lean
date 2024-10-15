import Mathlib

namespace NUMINAMATH_CALUDE_water_speed_calculation_l3926_392648

/-- The speed of water in a river where a person who can swim at 12 km/h in still water
    takes 1 hour to swim 10 km against the current. -/
def water_speed : ℝ := 2

theorem water_speed_calculation (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) 
  (h1 : still_water_speed = 12)
  (h2 : distance = 10)
  (h3 : time = 1)
  (h4 : distance / time = still_water_speed - water_speed) : 
  water_speed = 2 := by
  sorry

#check water_speed_calculation

end NUMINAMATH_CALUDE_water_speed_calculation_l3926_392648


namespace NUMINAMATH_CALUDE_tangent_line_and_root_range_l3926_392655

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 6 * x^2 - 6 * x

theorem tangent_line_and_root_range :
  -- Part 1: Tangent line equation
  (∀ x y : ℝ, y = f x → (x = 2 → 12 * x - y - 17 = 0)) ∧
  -- Part 2: Range of m for three distinct real roots
  (∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f x₁ + m = 0 ∧ f x₂ + m = 0 ∧ f x₃ + m = 0) ↔ -3 < m ∧ m < -2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_root_range_l3926_392655


namespace NUMINAMATH_CALUDE_marble_173_is_gray_l3926_392641

/-- Represents the color of a marble -/
inductive MarbleColor
| Gray
| White
| Black

/-- Defines the pattern of marbles -/
def marblePattern : List MarbleColor :=
  List.replicate 6 MarbleColor.Gray ++
  List.replicate 3 MarbleColor.White ++
  List.replicate 5 MarbleColor.Black

/-- Determines the color of the nth marble in the sequence -/
def nthMarbleColor (n : Nat) : MarbleColor :=
  let patternLength := marblePattern.length
  let indexInPattern := (n - 1) % patternLength
  marblePattern[indexInPattern]'
    (by
      have h : indexInPattern < patternLength := Nat.mod_lt _ (Nat.zero_lt_of_ne_zero (by decide))
      exact h
    )

/-- Theorem: The 173rd marble is gray -/
theorem marble_173_is_gray : nthMarbleColor 173 = MarbleColor.Gray := by
  sorry

end NUMINAMATH_CALUDE_marble_173_is_gray_l3926_392641


namespace NUMINAMATH_CALUDE_fruit_filling_probability_is_five_eighths_l3926_392620

/-- The number of fruit types available -/
def num_fruits : ℕ := 5

/-- The number of meat types available -/
def num_meats : ℕ := 4

/-- The number of ingredient types required for a filling -/
def ingredients_per_filling : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of making a mooncake with fruit filling -/
def fruit_filling_probability : ℚ :=
  choose num_fruits ingredients_per_filling /
  (choose num_fruits ingredients_per_filling + choose num_meats ingredients_per_filling)

theorem fruit_filling_probability_is_five_eighths :
  fruit_filling_probability = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fruit_filling_probability_is_five_eighths_l3926_392620


namespace NUMINAMATH_CALUDE_apple_basket_problem_l3926_392605

theorem apple_basket_problem (n : ℕ) (h1 : n > 1) : 
  (2 : ℝ) / n = (2 : ℝ) / 5 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_apple_basket_problem_l3926_392605


namespace NUMINAMATH_CALUDE_determinant_of_cubic_roots_l3926_392633

theorem determinant_of_cubic_roots (p q r : ℝ) (a b c : ℝ) : 
  (∀ x : ℝ, x^3 + 3*p*x^2 + q*x + r = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  let matrix := !![a, b, c; b, c, a; c, a, b]
  Matrix.det matrix = 3*p*q := by
sorry

end NUMINAMATH_CALUDE_determinant_of_cubic_roots_l3926_392633


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3926_392650

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 11) (h2 : a * b = 21) :
  a^3 + b^3 = 638 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3926_392650


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3926_392654

/-- Given a triangle with sides a, b, and c, where a = 3, b = 5, and c is a root of x^2 - 5x + 4 = 0
    that satisfies the triangle inequality, prove that the perimeter is 12. -/
theorem triangle_perimeter (a b c : ℝ) : 
  a = 3 → b = 5 → c^2 - 5*c + 4 = 0 → 
  a + b > c ∧ a + c > b ∧ b + c > a →
  a + b + c = 12 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3926_392654


namespace NUMINAMATH_CALUDE_distance_before_collision_value_l3926_392603

/-- Two boats moving towards each other -/
structure BoatSystem where
  boat1_speed : ℝ
  boat2_speed : ℝ
  initial_distance : ℝ

/-- Calculate the distance between boats one minute before collision -/
def distance_before_collision (bs : BoatSystem) : ℝ :=
  sorry

/-- Theorem stating the distance between boats one minute before collision -/
theorem distance_before_collision_value (bs : BoatSystem)
  (h1 : bs.boat1_speed = 5)
  (h2 : bs.boat2_speed = 25)
  (h3 : bs.initial_distance = 20) :
  distance_before_collision bs = 0.5 :=
sorry

end NUMINAMATH_CALUDE_distance_before_collision_value_l3926_392603


namespace NUMINAMATH_CALUDE_cubes_with_five_neighbors_count_l3926_392614

/-- Represents a large cube assembled from unit cubes -/
structure LargeCube where
  sideLength : ℕ

/-- The number of unit cubes with exactly 4 neighbors in the large cube -/
def cubesWithFourNeighbors (c : LargeCube) : ℕ := 12 * (c.sideLength - 2)

/-- The number of unit cubes with exactly 5 neighbors in the large cube -/
def cubesWithFiveNeighbors (c : LargeCube) : ℕ := 6 * (c.sideLength - 2)^2

/-- Theorem stating the relationship between cubes with 4 and 5 neighbors -/
theorem cubes_with_five_neighbors_count (c : LargeCube) 
  (h : cubesWithFourNeighbors c = 132) : 
  cubesWithFiveNeighbors c = 726 := by
  sorry

end NUMINAMATH_CALUDE_cubes_with_five_neighbors_count_l3926_392614


namespace NUMINAMATH_CALUDE_two_black_balls_probability_l3926_392687

/-- The probability of drawing two black balls without replacement from a box containing 8 white balls and 7 black balls is 1/5. -/
theorem two_black_balls_probability :
  let total_balls : ℕ := 8 + 7
  let black_balls : ℕ := 7
  let prob_first_black : ℚ := black_balls / total_balls
  let prob_second_black : ℚ := (black_balls - 1) / (total_balls - 1)
  prob_first_black * prob_second_black = 1 / 5 := by
sorry


end NUMINAMATH_CALUDE_two_black_balls_probability_l3926_392687


namespace NUMINAMATH_CALUDE_impossible_corner_cut_l3926_392627

theorem impossible_corner_cut (a b c : ℝ) : 
  a^2 + b^2 = 25 ∧ b^2 + c^2 = 36 ∧ c^2 + a^2 = 64 → False :=
by
  sorry

#check impossible_corner_cut

end NUMINAMATH_CALUDE_impossible_corner_cut_l3926_392627


namespace NUMINAMATH_CALUDE_deepak_present_age_l3926_392685

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's present age. -/
theorem deepak_present_age 
  (ratio_rahul : ℕ) 
  (ratio_deepak : ℕ) 
  (rahul_future_age : ℕ) 
  (years_to_future : ℕ) :
  ratio_rahul = 4 →
  ratio_deepak = 3 →
  rahul_future_age = 26 →
  years_to_future = 10 →
  ∃ (x : ℕ), 
    ratio_rahul * x + years_to_future = rahul_future_age ∧
    ratio_deepak * x = 12 :=
by sorry

end NUMINAMATH_CALUDE_deepak_present_age_l3926_392685


namespace NUMINAMATH_CALUDE_T_is_three_rays_l3926_392608

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 
    (5 = p.1 + 3 ∧ 5 ≥ p.2 - 6) ∨
    (5 = p.2 - 6 ∧ 5 ≥ p.1 + 3) ∨
    (p.1 + 3 = p.2 - 6 ∧ 5 ≥ p.1 + 3)}

-- Define the three rays
def ray1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2 ∧ p.2 ≤ 11}
def ray2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≤ 2 ∧ p.2 = 11}
def ray3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 9 ∧ p.1 ≤ 2 ∧ p.2 ≤ 11}

-- Theorem statement
theorem T_is_three_rays : T = ray1 ∪ ray2 ∪ ray3 := by
  sorry

end NUMINAMATH_CALUDE_T_is_three_rays_l3926_392608


namespace NUMINAMATH_CALUDE_rectangle_diagonal_estimate_l3926_392612

theorem rectangle_diagonal_estimate (length width diagonal : ℝ) : 
  length = 3 → width = 2 → diagonal^2 = length^2 + width^2 → 
  3.6 < diagonal ∧ diagonal < 3.7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_estimate_l3926_392612


namespace NUMINAMATH_CALUDE_fraction_is_standard_notation_l3926_392688

-- Define what it means for an expression to be in standard algebraic notation
def is_standard_algebraic_notation (expr : ℚ) : Prop :=
  ∃ (n m : ℤ), m ≠ 0 ∧ expr = n / m

-- Define our fraction
def our_fraction (n m : ℤ) : ℚ := n / m

-- Theorem statement
theorem fraction_is_standard_notation (n m : ℤ) (h : m ≠ 0) :
  is_standard_algebraic_notation (our_fraction n m) :=
sorry

end NUMINAMATH_CALUDE_fraction_is_standard_notation_l3926_392688


namespace NUMINAMATH_CALUDE_count_hexagons_l3926_392642

/-- The number of regular hexagons in a larger hexagon -/
def num_hexagons (n : ℕ+) : ℚ :=
  (n^2 + n : ℚ)^2 / 4

/-- Theorem: The number of regular hexagons with vertices among the vertices of equilateral triangles
    in a regular hexagon of side length n is (n² + n)² / 4 -/
theorem count_hexagons (n : ℕ+) :
  num_hexagons n = (n^2 + n : ℚ)^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_count_hexagons_l3926_392642


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l3926_392665

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∀ n : ℕ, n < 1000 → n % 5 = 0 → n % 6 = 0 → n ≤ 990 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l3926_392665


namespace NUMINAMATH_CALUDE_money_division_l3926_392690

theorem money_division (p q r : ℕ) (total : ℝ) : 
  p + q + r = 22 →  -- Ratio sum: 3 + 7 + 12 = 22
  (7 / 22 * total - 3 / 22 * total = 4000) →
  (12 / 22 * total - 7 / 22 * total = 5000) :=
by
  sorry

end NUMINAMATH_CALUDE_money_division_l3926_392690


namespace NUMINAMATH_CALUDE_emma_bank_account_l3926_392613

theorem emma_bank_account (initial_amount : ℝ) : 
  let withdrawal := 60
  let deposit := 2 * withdrawal
  let final_balance := 290
  (initial_amount - withdrawal + deposit = final_balance) → initial_amount = 230 := by
sorry

end NUMINAMATH_CALUDE_emma_bank_account_l3926_392613


namespace NUMINAMATH_CALUDE_jordan_income_proof_l3926_392670

-- Define the daily incomes and work days
def terry_daily_income : ℝ := 24
def work_days : ℕ := 7
def weekly_income_difference : ℝ := 42

-- Define Jordan's daily income as a variable
def jordan_daily_income : ℝ := 30

-- Theorem to prove
theorem jordan_income_proof :
  jordan_daily_income * work_days - terry_daily_income * work_days = weekly_income_difference :=
by sorry

end NUMINAMATH_CALUDE_jordan_income_proof_l3926_392670


namespace NUMINAMATH_CALUDE_function_domain_range_implies_b_equals_two_l3926_392607

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Define the properties of the function
def has_domain_range (b : ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc 1 b ↔ f x ∈ Set.Icc 1 b) ∧
  (∀ y ∈ Set.Icc 1 b, ∃ x ∈ Set.Icc 1 b, f x = y)

-- Theorem statement
theorem function_domain_range_implies_b_equals_two :
  ∃ b : ℝ, has_domain_range b → b = 2 := by sorry

end NUMINAMATH_CALUDE_function_domain_range_implies_b_equals_two_l3926_392607


namespace NUMINAMATH_CALUDE_parabola_directrix_intersection_l3926_392675

/-- The parabola equation: x^2 = 4y -/
def parabola_equation (x y : ℝ) : Prop := x^2 = 4*y

/-- The directrix equation for a parabola with equation x^2 = 4ay -/
def directrix_equation (a y : ℝ) : Prop := y = -a

/-- The y-axis equation -/
def y_axis (x : ℝ) : Prop := x = 0

theorem parabola_directrix_intersection :
  ∃ (a : ℝ), a = 1 ∧
  (∀ x y : ℝ, parabola_equation x y ↔ x^2 = 4*a*y) ∧
  (∃ y : ℝ, directrix_equation a y ∧ y_axis 0 ∧ y = -1) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_intersection_l3926_392675


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l3926_392649

/-- The surface area of a cube after removing smaller cubes from its corners --/
def surface_area_after_removal (cube_size : ℝ) (corner_size : ℝ) : ℝ :=
  6 * cube_size^2

/-- Theorem: The surface area remains unchanged after corner removal --/
theorem surface_area_unchanged
  (cube_size : ℝ)
  (corner_size : ℝ)
  (h1 : cube_size = 4)
  (h2 : corner_size = 1.5)
  : surface_area_after_removal cube_size corner_size = 96 := by
  sorry

#check surface_area_unchanged

end NUMINAMATH_CALUDE_surface_area_unchanged_l3926_392649


namespace NUMINAMATH_CALUDE_real_part_of_z_l3926_392611

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  (z.re : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3926_392611


namespace NUMINAMATH_CALUDE_triangle_cosine_problem_l3926_392622

theorem triangle_cosine_problem (A B C : ℝ) (a b c : ℝ) (D : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sqrt 3 * Real.sin (2018 * Real.pi - x) * Real.sin (3 * Real.pi / 2 + x) - Real.cos x ^ 2 + 1
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- D is on angle bisector of A
  2 * Real.cos (A / 2) * Real.sin (B / 2) = Real.sin (C / 2) →
  -- f(A) = 3/2
  f A = 3 / 2 →
  -- AD = √2 BD = 2
  2 * Real.sin (B / 2) = Real.sqrt 2 * Real.sin (C / 2) ∧
  2 * Real.sin (B / 2) = 2 * Real.sin ((B + C) / 2) →
  -- Conclusion
  Real.cos C = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_problem_l3926_392622


namespace NUMINAMATH_CALUDE_march_largest_drop_l3926_392630

/-- Represents the months in the first half of 1994 -/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June

/-- Returns the price change for a given month -/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.January  => -1.00
  | Month.February => 0.50
  | Month.March    => -3.00
  | Month.April    => 2.00
  | Month.May      => -1.50
  | Month.June     => -0.75

/-- Determines if a given month has the largest price drop -/
def has_largest_drop (m : Month) : Prop :=
  ∀ (other : Month), price_change m ≤ price_change other

theorem march_largest_drop :
  has_largest_drop Month.March :=
sorry

end NUMINAMATH_CALUDE_march_largest_drop_l3926_392630


namespace NUMINAMATH_CALUDE_max_value_of_f_l3926_392682

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + 1/2 * x^2 + 2*a*x

theorem max_value_of_f (a : ℝ) (h1 : 0 < a) (h2 : a < 2) 
  (h3 : ∃ x ∈ Set.Icc 1 4, ∀ y ∈ Set.Icc 1 4, f a x ≤ f a y) 
  (h4 : ∃ x ∈ Set.Icc 1 4, f a x = -16/3) :
  ∃ x ∈ Set.Icc 1 4, ∀ y ∈ Set.Icc 1 4, f a y ≤ f a x ∧ f a x = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3926_392682


namespace NUMINAMATH_CALUDE_jack_life_timeline_l3926_392686

theorem jack_life_timeline (jack_lifetime : ℝ) 
  (h1 : jack_lifetime = 84)
  (adolescence : ℝ) (h2 : adolescence = (1/6) * jack_lifetime)
  (facial_hair : ℝ) (h3 : facial_hair = (1/12) * jack_lifetime)
  (marriage : ℝ) (h4 : marriage = (1/7) * jack_lifetime)
  (son_birth : ℝ) (h5 : son_birth = 5)
  (son_lifetime : ℝ) (h6 : son_lifetime = (1/2) * jack_lifetime) :
  jack_lifetime - (adolescence + facial_hair + marriage + son_birth + son_lifetime) = 4 := by
sorry

end NUMINAMATH_CALUDE_jack_life_timeline_l3926_392686


namespace NUMINAMATH_CALUDE_distance_between_A_and_B_l3926_392600

-- Define the position of point A
def A : ℝ := 3

-- Define the possible positions of point B
def B : Set ℝ := {-9, 9}

-- Define the distance function
def distance (x y : ℝ) : ℝ := |x - y|

-- Theorem statement
theorem distance_between_A_and_B :
  ∀ b ∈ B, distance A b = 6 ∨ distance A b = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_A_and_B_l3926_392600


namespace NUMINAMATH_CALUDE_arthur_leftover_is_four_l3926_392667

/-- The amount of money Arthur has leftover after selling his basketball cards and buying comic books -/
def arthursLeftover (cardValue : ℚ) (numCards : ℕ) (comicBookPrice : ℚ) : ℚ :=
  let totalCardValue := cardValue * numCards
  let numComicBooks := (totalCardValue / comicBookPrice).floor
  totalCardValue - numComicBooks * comicBookPrice

/-- Theorem stating that Arthur will have $4 leftover -/
theorem arthur_leftover_is_four :
  arthursLeftover (5/100) 2000 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arthur_leftover_is_four_l3926_392667


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l3926_392651

/-- If x² + 2x + m > 0 for all real x, then m > 1 -/
theorem quadratic_always_positive (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + m > 0) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l3926_392651


namespace NUMINAMATH_CALUDE_tourist_tax_calculation_l3926_392674

theorem tourist_tax_calculation (tax_free_amount tax_rate total_tax : ℝ) 
  (h1 : tax_free_amount = 600)
  (h2 : tax_rate = 0.07)
  (h3 : total_tax = 78.4) : 
  ∃ (total_value : ℝ), 
    total_value > tax_free_amount ∧ 
    tax_rate * (total_value - tax_free_amount) = total_tax ∧ 
    total_value = 1720 := by
  sorry

end NUMINAMATH_CALUDE_tourist_tax_calculation_l3926_392674


namespace NUMINAMATH_CALUDE_natural_number_pair_product_sum_gcd_lcm_l3926_392672

theorem natural_number_pair_product_sum_gcd_lcm : 
  ∀ a b : ℕ, 
    a > 0 ∧ b > 0 → 
    (a * b - (a + b) = Nat.gcd a b + Nat.lcm a b) ↔ 
    ((a = 6 ∧ b = 3) ∨ (a = 6 ∧ b = 4) ∨ (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 6)) :=
by sorry

end NUMINAMATH_CALUDE_natural_number_pair_product_sum_gcd_lcm_l3926_392672


namespace NUMINAMATH_CALUDE_blue_garden_yield_l3926_392634

/-- Calculates the expected potato yield from a rectangular garden --/
def expected_potato_yield (length_steps : ℕ) (width_steps : ℕ) (feet_per_step : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (length_steps : ℝ) * feet_per_step * (width_steps : ℝ) * feet_per_step * yield_per_sqft

theorem blue_garden_yield :
  expected_potato_yield 18 25 3 (3/4) = 3037.5 := by
  sorry

end NUMINAMATH_CALUDE_blue_garden_yield_l3926_392634


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3926_392621

theorem quadratic_inequality (x : ℝ) : x^2 - 8*x + 12 < 0 ↔ 2 < x ∧ x < 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3926_392621


namespace NUMINAMATH_CALUDE_smallest_number_with_all_factors_l3926_392695

def alice_number : ℕ := 90

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ n → p ∣ m)

theorem smallest_number_with_all_factors :
  ∃ m : ℕ, m > 0 ∧ has_all_prime_factors alice_number m ∧
  ∀ k : ℕ, k > 0 → has_all_prime_factors alice_number k → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_all_factors_l3926_392695


namespace NUMINAMATH_CALUDE_sum_expression_value_l3926_392645

theorem sum_expression_value (a b c : ℝ) 
  (h1 : a + b = 8) 
  (h2 : a * b = c^2 + 16) : 
  a + 2*b + 3*c = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_expression_value_l3926_392645


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l3926_392647

theorem geometric_sequence_middle_term 
  (a b c : ℝ) 
  (h_seq : ∃ r : ℝ, b = a * r ∧ c = b * r) 
  (h_a : a = 7 + 4 * Real.sqrt 3) 
  (h_c : c = 7 - 4 * Real.sqrt 3) : 
  b = 1 ∨ b = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l3926_392647


namespace NUMINAMATH_CALUDE_negation_of_r_not_p_is_true_p_or_r_is_false_p_and_q_is_false_l3926_392691

-- Define the propositions
def p : Prop := ∃ x₀ : ℝ, x₀ > -2 ∧ 6 + |x₀| = 5

def q : Prop := ∀ x : ℝ, x < 0 → x^2 + 4/x^2 ≥ 4

def r : Prop := ∀ x y : ℝ, |x| + |y| ≤ 1 → |y| / (|x| + 2) ≤ 1/2

-- Theorem statements
theorem negation_of_r : 
  (¬r) ↔ (∃ x y : ℝ, |x| + |y| > 1 ∧ |y| / (|x| + 2) > 1/2) :=
sorry

theorem not_p_is_true : ¬p :=
sorry

theorem p_or_r_is_false : ¬(p ∨ r) :=
sorry

theorem p_and_q_is_false : ¬(p ∧ q) :=
sorry

end NUMINAMATH_CALUDE_negation_of_r_not_p_is_true_p_or_r_is_false_p_and_q_is_false_l3926_392691


namespace NUMINAMATH_CALUDE_solution_set_equality_l3926_392601

theorem solution_set_equality : {x : ℝ | x^2 - 2*x + 1 = 0} = {1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3926_392601


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l3926_392640

/-- Represents different sampling methods --/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Represents income levels --/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents a community with different income levels --/
structure Community where
  highIncome : Nat
  middleIncome : Nat
  lowIncome : Nat

/-- Represents a school class --/
structure SchoolClass where
  totalStudents : Nat
  specialtyType : String

/-- Determines the most appropriate sampling method for a community survey --/
def communitySamplingMethod (community : Community) (sampleSize : Nat) : SamplingMethod :=
  sorry

/-- Determines the most appropriate sampling method for a school class survey --/
def schoolClassSamplingMethod (schoolClass : SchoolClass) (sampleSize : Nat) : SamplingMethod :=
  sorry

/-- Theorem stating the appropriate sampling methods for the given surveys --/
theorem appropriate_sampling_methods
  (community : Community)
  (schoolClass : SchoolClass) :
  communitySamplingMethod {highIncome := 125, middleIncome := 280, lowIncome := 95} 100 = SamplingMethod.Stratified ∧
  schoolClassSamplingMethod {totalStudents := 15, specialtyType := "art"} 3 = SamplingMethod.SimpleRandom :=
  sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l3926_392640


namespace NUMINAMATH_CALUDE_soap_box_height_l3926_392663

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Theorem: Given the dimensions of a carton and soap boxes, and the maximum number of soap boxes
    that can fit in the carton, prove that the height of a soap box is 1 inch. -/
theorem soap_box_height
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (max_boxes : ℕ)
  (h_carton_length : carton.length = 30)
  (h_carton_width : carton.width = 42)
  (h_carton_height : carton.height = 60)
  (h_soap_length : soap.length = 7)
  (h_soap_width : soap.width = 6)
  (h_max_boxes : max_boxes = 360)
  : soap.height = 1 :=
by sorry

end NUMINAMATH_CALUDE_soap_box_height_l3926_392663


namespace NUMINAMATH_CALUDE_stratified_sampling_result_count_l3926_392696

def junior_population : ℕ := 400
def senior_population : ℕ := 200
def total_sample_size : ℕ := 60

def stratified_proportional_sample_count (n1 n2 k : ℕ) : ℕ :=
  Nat.choose n1 ((k * n1) / (n1 + n2)) * Nat.choose n2 ((k * n2) / (n1 + n2))

theorem stratified_sampling_result_count :
  stratified_proportional_sample_count junior_population senior_population total_sample_size =
  Nat.choose junior_population 40 * Nat.choose senior_population 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_result_count_l3926_392696


namespace NUMINAMATH_CALUDE_largest_equal_cost_number_l3926_392668

/-- Sum of digits in decimal representation -/
def sumOfDecimalDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDecimalDigits (n / 10)

/-- Sum of digits in binary representation -/
def sumOfBinaryDigits (n : Nat) : Nat :=
  if n = 0 then 0 else (n % 2) + sumOfBinaryDigits (n / 2)

/-- Cost calculation for Option 1 -/
def option1Cost (n : Nat) : Nat :=
  2 * sumOfDecimalDigits n

/-- Cost calculation for Option 2 -/
def option2Cost (n : Nat) : Nat :=
  sumOfBinaryDigits n

theorem largest_equal_cost_number :
  ∀ n : Nat, n < 2000 → n > 1023 →
    option1Cost n ≠ option2Cost n ∧
    option1Cost 1023 = option2Cost 1023 :=
by sorry

end NUMINAMATH_CALUDE_largest_equal_cost_number_l3926_392668


namespace NUMINAMATH_CALUDE_profit_achieved_l3926_392609

/-- The number of pencils purchased -/
def num_purchased : ℕ := 1800

/-- The cost of each pencil when purchased -/
def cost_per_pencil : ℚ := 15 / 100

/-- The selling price of each pencil -/
def selling_price : ℚ := 30 / 100

/-- The desired profit -/
def desired_profit : ℚ := 150

/-- The number of pencils that must be sold to make the desired profit -/
def num_sold : ℕ := 1400

theorem profit_achieved : 
  (num_sold : ℚ) * selling_price - (num_purchased : ℚ) * cost_per_pencil = desired_profit := by
  sorry

end NUMINAMATH_CALUDE_profit_achieved_l3926_392609


namespace NUMINAMATH_CALUDE_parallelogram_area_10_20_l3926_392692

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 10 cm and height 20 cm is 200 square centimeters -/
theorem parallelogram_area_10_20 :
  parallelogram_area 10 20 = 200 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_10_20_l3926_392692


namespace NUMINAMATH_CALUDE_all_statements_false_l3926_392615

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Define the theorem
theorem all_statements_false :
  ¬(∀ (m n : Line) (α : Plane), 
    parallel_line_plane m α → parallel_line_plane n α → parallel_lines m n) ∧
  ¬(∀ (m n : Line) (α : Plane), 
    perpendicular_line_plane m α → perpendicular_lines m n → parallel_line_plane n α) ∧
  ¬(∀ (m n : Line) (α β : Plane), 
    perpendicular_line_plane m α → perpendicular_line_plane n β → 
    perpendicular_lines m n → perpendicular_planes α β) ∧
  ¬(∀ (m : Line) (α β : Plane), 
    line_in_plane m β → parallel_planes α β → parallel_line_plane m α) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_false_l3926_392615


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3926_392659

/-- The constant term in the expansion of (x - 3/x^2)^6 -/
def constant_term : ℕ := 135

/-- The binomial coefficient function -/
def binomial_coeff (n k : ℕ) : ℕ := sorry

/-- Theorem: The constant term in the expansion of (x - 3/x^2)^6 is 135 -/
theorem constant_term_expansion :
  constant_term = binomial_coeff 6 2 * 3^2 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l3926_392659


namespace NUMINAMATH_CALUDE_not_all_cells_tetraploid_l3926_392671

/-- Represents a watermelon plant --/
structure WatermelonPlant where
  /-- The number of chromosome sets in somatic cells --/
  somaticChromosomeSets : ℕ
  /-- The number of chromosome sets in root cells --/
  rootChromosomeSets : ℕ

/-- Represents the process of culturing and treating watermelon plants --/
def cultureAndTreat (original : WatermelonPlant) : WatermelonPlant :=
  { somaticChromosomeSets := 2 * original.somaticChromosomeSets,
    rootChromosomeSets := original.rootChromosomeSets }

/-- Theorem: Not all cells in a watermelon plant obtained from treating diploid seedlings
    with colchicine contain four sets of chromosomes --/
theorem not_all_cells_tetraploid (original : WatermelonPlant)
    (h_diploid : original.somaticChromosomeSets = 2)
    (h_root_untreated : (cultureAndTreat original).rootChromosomeSets = original.rootChromosomeSets) :
    ∃ (cell_type : WatermelonPlant → ℕ),
      cell_type (cultureAndTreat original) ≠ 4 :=
  sorry


end NUMINAMATH_CALUDE_not_all_cells_tetraploid_l3926_392671


namespace NUMINAMATH_CALUDE_triangle_inequality_l3926_392618

open Real

theorem triangle_inequality (A B C : ℝ) (R r : ℝ) :
  R > 0 ∧ r > 0 →
  (3 * Real.sqrt 3 * r^2) / (2 * R^2) ≤ Real.sin A * Real.sin B * Real.sin C ∧
  Real.sin A * Real.sin B * Real.sin C ≤ (3 * Real.sqrt 3 * r) / (4 * R) ∧
  (3 * Real.sqrt 3 * r) / (4 * R) ≤ 3 * Real.sqrt 3 / 8 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3926_392618


namespace NUMINAMATH_CALUDE_inequality_solution_parity_of_f_l3926_392661

noncomputable section

variable (x : ℝ) (a : ℝ)

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a/x

theorem inequality_solution :
  (∀ x, 0 < x ∧ x < 1 ↔ f x 2 - f (x-1) 2 > 2*x - 1) :=
sorry

theorem parity_of_f :
  (∀ x ≠ 0, f (-x) 0 = f x 0) ∧
  (∀ a ≠ 0, ∃ x ≠ 0, f (-x) a ≠ f x a ∧ f (-x) a ≠ -f x a) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_parity_of_f_l3926_392661


namespace NUMINAMATH_CALUDE_expression_simplification_l3926_392606

theorem expression_simplification (x y : ℝ) :
  (x^3 - 9*x*y^2) / (9*y^2 + x^2) * ((x + 3*y) / (x^2 - 3*x*y) + (x - 3*y) / (x^2 + 3*x*y)) = x - 3*y :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3926_392606


namespace NUMINAMATH_CALUDE_final_state_is_blue_l3926_392689

/-- Represents the color of a sheep -/
inductive SheepColor
  | Red
  | Green
  | Blue

/-- Represents the state of the sheep population -/
structure SheepState where
  red : Nat
  green : Nat
  blue : Nat

/-- The color-changing rule for sheep meetings -/
def changeColor (c1 c2 : SheepColor) : SheepColor :=
  match c1, c2 with
  | SheepColor.Red, SheepColor.Green => SheepColor.Blue
  | SheepColor.Red, SheepColor.Blue => SheepColor.Green
  | SheepColor.Green, SheepColor.Blue => SheepColor.Red
  | SheepColor.Green, SheepColor.Red => SheepColor.Blue
  | SheepColor.Blue, SheepColor.Red => SheepColor.Green
  | SheepColor.Blue, SheepColor.Green => SheepColor.Red
  | _, _ => c1  -- If same color, no change

/-- The invariant property of the sheep population -/
def invariant (state : SheepState) : Bool :=
  (state.red - state.green) % 3 = 0 ∧
  (state.green - state.blue) % 3 = 2 ∧
  (state.blue - state.red) % 3 = 1

/-- The initial state of the sheep population -/
def initialState : SheepState :=
  { red := 18, green := 15, blue := 22 }

/-- Theorem: The only possible final state is all sheep being blue -/
theorem final_state_is_blue (state : SheepState) :
  invariant initialState →
  invariant state →
  (state.red + state.green + state.blue = initialState.red + initialState.green + initialState.blue) →
  (state.red = 0 ∧ state.green = 0 ∧ state.blue = 55) :=
sorry

end NUMINAMATH_CALUDE_final_state_is_blue_l3926_392689


namespace NUMINAMATH_CALUDE_average_pencils_per_box_l3926_392676

theorem average_pencils_per_box : 
  let pencil_counts : List Nat := [12, 14, 14, 15, 15, 15, 16, 16, 17, 18]
  let total_boxes : Nat := pencil_counts.length
  let total_pencils : Nat := pencil_counts.sum
  (total_pencils : ℚ) / total_boxes = 15.2 := by
  sorry

end NUMINAMATH_CALUDE_average_pencils_per_box_l3926_392676


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3926_392694

theorem sum_of_coefficients (x : ℝ) :
  ∃ (A B C D E : ℝ),
    125 * x^3 + 64 = (A * x + B) * (C * x^2 + D * x + E) ∧
    A + B + C + D + E = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3926_392694


namespace NUMINAMATH_CALUDE_problem_solution_l3926_392637

theorem problem_solution (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h1 : a^b = b^a) (h2 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3926_392637


namespace NUMINAMATH_CALUDE_cyclic_ratio_inequality_l3926_392623

theorem cyclic_ratio_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  a / b + b / c + c / d + d / a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_cyclic_ratio_inequality_l3926_392623


namespace NUMINAMATH_CALUDE_theater_probability_ratio_l3926_392660

theorem theater_probability_ratio : 
  let n : ℕ := 4  -- number of sections and acts
  let p : ℝ := 1 / 4  -- probability of moving in a given act
  let q : ℝ := 1 - p  -- probability of not moving in a given act
  let prob_move_once : ℝ := n * p * q^(n-1)  -- probability of moving exactly once
  let prob_move_twice : ℝ := (n.choose 2) * p^2 * q^(n-2)  -- probability of moving exactly twice
  prob_move_twice / prob_move_once = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_theater_probability_ratio_l3926_392660


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_value_l3926_392625

theorem intersection_nonempty_implies_a_value (a : ℝ) : 
  let P : Set ℝ := {0, a}
  let Q : Set ℝ := {1, 2}
  (P ∩ Q).Nonempty → a = 1 ∨ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_value_l3926_392625


namespace NUMINAMATH_CALUDE_solution_set_inequalities_l3926_392693

theorem solution_set_inequalities (a b : ℝ) 
  (h : ∃ x, x > a ∧ x < b) : 
  {x : ℝ | x < 1 - a ∧ x < 1 - b} = {x : ℝ | x < 1 - b} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequalities_l3926_392693


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3926_392681

theorem coin_flip_probability (oliver_prob jayden_prob mia_prob : ℚ) :
  oliver_prob = 1/3 →
  jayden_prob = 1/4 →
  mia_prob = 1/5 →
  (∑' n : ℕ, (1 - oliver_prob)^(n-1) * oliver_prob *
              (1 - jayden_prob)^(n-1) * jayden_prob *
              (1 - mia_prob)^(n-1) * mia_prob) = 1/36 := by
  sorry

#check coin_flip_probability

end NUMINAMATH_CALUDE_coin_flip_probability_l3926_392681


namespace NUMINAMATH_CALUDE_claire_earnings_l3926_392656

-- Define the given quantities
def total_flowers : ℕ := 400
def tulips : ℕ := 120
def white_roses : ℕ := 80
def small_red_roses : ℕ := 40
def medium_red_roses : ℕ := 60

-- Define the prices
def price_small : ℚ := 3/4
def price_medium : ℚ := 1
def price_large : ℚ := 5/4

-- Calculate the number of roses and red roses
def roses : ℕ := total_flowers - tulips
def red_roses : ℕ := roses - white_roses

-- Calculate the number of large red roses
def large_red_roses : ℕ := red_roses - small_red_roses - medium_red_roses

-- Define the function to calculate earnings
def earnings : ℚ :=
  (small_red_roses / 2 : ℚ) * price_small +
  (medium_red_roses / 2 : ℚ) * price_medium +
  (large_red_roses / 2 : ℚ) * price_large

-- Theorem statement
theorem claire_earnings : earnings = 215/2 := by sorry

end NUMINAMATH_CALUDE_claire_earnings_l3926_392656


namespace NUMINAMATH_CALUDE_lakeisha_lawn_size_l3926_392684

/-- The size of each lawn LaKeisha has already mowed -/
def lawn_size : ℝ := sorry

/-- LaKeisha's charge per square foot -/
def charge_per_sqft : ℝ := 0.10

/-- Cost of the book set -/
def book_cost : ℝ := 150

/-- Number of lawns already mowed -/
def lawns_mowed : ℕ := 3

/-- Additional square feet to mow -/
def additional_sqft : ℝ := 600

theorem lakeisha_lawn_size :
  lawn_size = 300 ∧
  charge_per_sqft * (lawns_mowed * lawn_size + additional_sqft) = book_cost :=
sorry

end NUMINAMATH_CALUDE_lakeisha_lawn_size_l3926_392684


namespace NUMINAMATH_CALUDE_max_sum_is_24_l3926_392616

def numbers : Finset ℕ := {1, 4, 7, 10, 13}

def valid_arrangement (a b c d e : ℕ) : Prop :=
  a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ d ∈ numbers ∧ e ∈ numbers ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  a + b + e = a + c + e

def sum_of_arrangement (a b c d e : ℕ) : ℕ := a + b + e

theorem max_sum_is_24 :
  ∀ a b c d e : ℕ, valid_arrangement a b c d e →
    sum_of_arrangement a b c d e ≤ 24 :=
sorry

end NUMINAMATH_CALUDE_max_sum_is_24_l3926_392616


namespace NUMINAMATH_CALUDE_largest_non_representable_integer_l3926_392636

/-- 
Given positive integers a, b, and c with no two having a common divisor greater than 1,
2abc-ab-bc-ca is the largest integer that cannot be expressed as xbc+yca+zab 
for non-negative integers x, y, z
-/
theorem largest_non_representable_integer (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : Nat.gcd a b = 1) (hbc : Nat.gcd b c = 1) (hca : Nat.gcd c a = 1) :
  (∀ x y z : ℕ, 2*a*b*c - a*b - b*c - c*a ≠ x*b*c + y*c*a + z*a*b) ∧
  (∀ n : ℕ, n > 2*a*b*c - a*b - b*c - c*a → 
    ∃ x y z : ℕ, n = x*b*c + y*c*a + z*a*b) := by
  sorry

end NUMINAMATH_CALUDE_largest_non_representable_integer_l3926_392636


namespace NUMINAMATH_CALUDE_female_officers_on_duty_percentage_l3926_392635

/-- Calculates the percentage of female officers on duty -/
def percentage_female_officers_on_duty (total_on_duty : ℕ) (female_ratio_on_duty : ℚ) (total_female_officers : ℕ) : ℚ :=
  (female_ratio_on_duty * total_on_duty : ℚ) / total_female_officers * 100

/-- Theorem stating that the percentage of female officers on duty is 20% -/
theorem female_officers_on_duty_percentage 
  (total_on_duty : ℕ) 
  (female_ratio_on_duty : ℚ) 
  (total_female_officers : ℕ) 
  (h1 : total_on_duty = 100)
  (h2 : female_ratio_on_duty = 1/2)
  (h3 : total_female_officers = 250) :
  percentage_female_officers_on_duty total_on_duty female_ratio_on_duty total_female_officers = 20 :=
sorry

end NUMINAMATH_CALUDE_female_officers_on_duty_percentage_l3926_392635


namespace NUMINAMATH_CALUDE_max_value_when_a_is_one_range_of_a_for_two_roots_l3926_392624

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 + 4 * x - 3 - a

-- Theorem for the maximum value of f when a = 1
theorem max_value_when_a_is_one :
  ∃ (max : ℝ), max = 2 ∧ ∀ x ∈ Set.Icc (-1) 1, f 1 x ≤ max :=
sorry

-- Theorem for the range of a when f has two distinct roots
theorem range_of_a_for_two_roots :
  ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ 
    a ∈ Set.Ioi 0 ∪ Set.Ioo (-1) 0 ∪ Set.Iic (-2) :=
sorry

end NUMINAMATH_CALUDE_max_value_when_a_is_one_range_of_a_for_two_roots_l3926_392624


namespace NUMINAMATH_CALUDE_circle_diameter_from_triangle_l3926_392652

/-- Theorem: The diameter of a circle inscribing a right triangle with area 150 and one leg 30 is 10√10 -/
theorem circle_diameter_from_triangle (triangle_area : ℝ) (leg : ℝ) (diameter : ℝ) : 
  triangle_area = 150 →
  leg = 30 →
  diameter = 10 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_triangle_l3926_392652


namespace NUMINAMATH_CALUDE_probabilities_correct_l3926_392677

/-- Represents the color of a ball -/
inductive Color
  | Black
  | White

/-- Represents a bag containing balls -/
structure Bag where
  black : ℕ
  white : ℕ

/-- Calculate the probability of drawing a ball of a specific color from a bag -/
def prob_color (b : Bag) (c : Color) : ℚ :=
  match c with
  | Color.Black => b.black / (b.black + b.white)
  | Color.White => b.white / (b.black + b.white)

/-- The contents of bag A -/
def bag_A : Bag := ⟨2, 2⟩

/-- The contents of bag B -/
def bag_B : Bag := ⟨2, 1⟩

theorem probabilities_correct :
  (prob_color bag_A Color.Black * prob_color bag_B Color.Black = 1/3) ∧
  (prob_color bag_A Color.White * prob_color bag_B Color.White = 1/6) ∧
  (prob_color bag_A Color.White * prob_color bag_B Color.White +
   prob_color bag_A Color.White * prob_color bag_B Color.Black +
   prob_color bag_A Color.Black * prob_color bag_B Color.White = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_probabilities_correct_l3926_392677


namespace NUMINAMATH_CALUDE_power_mod_thirteen_l3926_392604

theorem power_mod_thirteen : 2^2010 ≡ 12 [ZMOD 13] := by sorry

end NUMINAMATH_CALUDE_power_mod_thirteen_l3926_392604


namespace NUMINAMATH_CALUDE_fruit_cost_problem_l3926_392644

/-- The cost of fruits problem -/
theorem fruit_cost_problem (apple_price pear_price mango_price : ℝ) 
  (h1 : 5 * apple_price + 4 * pear_price = 48)
  (h2 : 2 * apple_price + 3 * mango_price = 33)
  (h3 : mango_price = pear_price + 2.5) :
  3 * apple_price + 3 * pear_price = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_fruit_cost_problem_l3926_392644


namespace NUMINAMATH_CALUDE_binomial_20_choose_6_l3926_392662

theorem binomial_20_choose_6 : Nat.choose 20 6 = 38760 := by sorry

end NUMINAMATH_CALUDE_binomial_20_choose_6_l3926_392662


namespace NUMINAMATH_CALUDE_nonagon_side_length_l3926_392638

/-- A regular nonagon with perimeter 171 cm has sides of length 19 cm -/
theorem nonagon_side_length : ∀ (perimeter side_length : ℝ),
  perimeter = 171 →
  side_length * 9 = perimeter →
  side_length = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_nonagon_side_length_l3926_392638


namespace NUMINAMATH_CALUDE_probability_three_two_color_l3926_392697

/-- The probability of drawing 3 balls of one color and 2 of the other from a bin with 10 black and 10 white balls -/
theorem probability_three_two_color (total_balls : ℕ) (black_balls white_balls : ℕ) (drawn_balls : ℕ) : 
  total_balls = black_balls + white_balls →
  black_balls = 10 →
  white_balls = 10 →
  drawn_balls = 5 →
  (Nat.choose total_balls drawn_balls : ℚ) * (30 : ℚ) / (43 : ℚ) = 
    (Nat.choose black_balls 3 * Nat.choose white_balls 2 + 
     Nat.choose black_balls 2 * Nat.choose white_balls 3 : ℚ) :=
by sorry

#check probability_three_two_color

end NUMINAMATH_CALUDE_probability_three_two_color_l3926_392697


namespace NUMINAMATH_CALUDE_meat_cost_per_pound_l3926_392683

/-- The cost of meat per pound given the total cost, rice quantity, rice price, and meat quantity -/
theorem meat_cost_per_pound 
  (total_cost : ℝ)
  (rice_quantity : ℝ)
  (rice_price_per_kg : ℝ)
  (meat_quantity : ℝ)
  (h1 : total_cost = 25)
  (h2 : rice_quantity = 5)
  (h3 : rice_price_per_kg = 2)
  (h4 : meat_quantity = 3)
  : (total_cost - rice_quantity * rice_price_per_kg) / meat_quantity = 5 := by
  sorry

end NUMINAMATH_CALUDE_meat_cost_per_pound_l3926_392683


namespace NUMINAMATH_CALUDE_license_plate_count_l3926_392602

/-- Number of digits in the license plate -/
def num_digits : ℕ := 5

/-- Number of letters in the license plate -/
def num_letters : ℕ := 3

/-- Number of possible digits (0-9) -/
def digit_choices : ℕ := 10

/-- Number of possible letters (A-Z) -/
def letter_choices : ℕ := 26

/-- Number of positions where the consecutive letters can be placed -/
def letter_positions : ℕ := num_digits + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := letter_positions * digit_choices^num_digits * letter_choices^num_letters

theorem license_plate_count : total_license_plates = 105456000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3926_392602


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l3926_392680

def number : Nat := 219257

theorem largest_prime_factors_difference (p q : Nat) : 
  Nat.Prime p ∧ Nat.Prime q ∧ 
  p ∣ number ∧ q ∣ number ∧
  ∀ r, Nat.Prime r → r ∣ number → r ≤ p ∧ r ≤ q →
  p - q = 144 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l3926_392680


namespace NUMINAMATH_CALUDE_fraction_multiplication_addition_l3926_392678

theorem fraction_multiplication_addition : 
  (1 / 3 : ℚ) * (1 / 4 : ℚ) * (1 / 5 : ℚ) + (1 / 2 : ℚ) = 31 / 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_addition_l3926_392678


namespace NUMINAMATH_CALUDE_alice_savings_l3926_392626

/-- Alice's savings problem -/
theorem alice_savings (total_days : ℕ) (total_dimes : ℕ) (dime_value : ℚ) (daily_savings : ℚ) : 
  total_days = 40 →
  total_dimes = 4 →
  dime_value = 1/10 →
  daily_savings = (total_dimes : ℚ) * dime_value / total_days →
  daily_savings = 1/100 := by
  sorry

#check alice_savings

end NUMINAMATH_CALUDE_alice_savings_l3926_392626


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3926_392699

theorem geometric_sequence_problem (a : ℝ) (h : a > 0) :
  let r : ℝ := 1/2
  let n : ℕ := 6
  let sum : ℝ := a * (1 - r^n) / (1 - r)
  sum = 189 → a * r = 48 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3926_392699


namespace NUMINAMATH_CALUDE_product_of_numbers_l3926_392628

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 42) (h2 : |x - y| = 4) : x * y = 437 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3926_392628


namespace NUMINAMATH_CALUDE_pentagon_triangle_intersections_pentagon_quadrilateral_intersections_l3926_392643

/-- A polygon in a plane -/
class Polygon :=
  (sides : ℕ)

/-- A pentagon is a polygon with 5 sides -/
def Pentagon : Polygon :=
  { sides := 5 }

/-- A triangle is a polygon with 3 sides -/
def Triangle : Polygon :=
  { sides := 3 }

/-- A quadrilateral is a polygon with 4 sides -/
def Quadrilateral : Polygon :=
  { sides := 4 }

/-- The maximum number of intersection points between the sides of two polygons -/
def maxIntersections (P Q : Polygon) : ℕ := sorry

/-- Theorem: Maximum intersections between a pentagon and a triangle -/
theorem pentagon_triangle_intersections :
  maxIntersections Pentagon Triangle = 10 := by sorry

/-- Theorem: Maximum intersections between a pentagon and a quadrilateral -/
theorem pentagon_quadrilateral_intersections :
  maxIntersections Pentagon Quadrilateral = 16 := by sorry

end NUMINAMATH_CALUDE_pentagon_triangle_intersections_pentagon_quadrilateral_intersections_l3926_392643


namespace NUMINAMATH_CALUDE_integral_comparison_l3926_392664

theorem integral_comparison : ∫ x in (0:ℝ)..1, x > ∫ x in (0:ℝ)..1, x^3 := by
  sorry

end NUMINAMATH_CALUDE_integral_comparison_l3926_392664


namespace NUMINAMATH_CALUDE_point_on_double_angle_l3926_392658

/-- Given a point P(-1, 2) on the terminal side of angle α, 
    prove that the point (-3, -4) lies on the terminal side of angle 2α. -/
theorem point_on_double_angle (α : ℝ) :
  let P : ℝ × ℝ := (-1, 2)
  let r : ℝ := Real.sqrt (P.1^2 + P.2^2)
  let cos_α : ℝ := P.1 / r
  let sin_α : ℝ := P.2 / r
  let cos_2α : ℝ := cos_α^2 - sin_α^2
  let sin_2α : ℝ := 2 * sin_α * cos_α
  let Q : ℝ × ℝ := (-3, -4)
  (∃ k : ℝ, k > 0 ∧ Q.1 = k * cos_2α ∧ Q.2 = k * sin_2α) :=
by
  sorry

end NUMINAMATH_CALUDE_point_on_double_angle_l3926_392658


namespace NUMINAMATH_CALUDE_congruence_solution_l3926_392657

theorem congruence_solution : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 14567 [MOD 16] ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l3926_392657


namespace NUMINAMATH_CALUDE_no_intersection_l3926_392646

/-- Represents a 2D point or vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Represents a parametric line in 2D -/
structure ParamLine where
  origin : Vec2D
  direction : Vec2D

/-- The first line -/
def line1 : ParamLine :=
  { origin := { x := 1, y := 4 }
    direction := { x := -2, y := 6 } }

/-- The second line -/
def line2 : ParamLine :=
  { origin := { x := 3, y := 10 }
    direction := { x := -1, y := 3 } }

/-- Checks if two parametric lines intersect -/
def linesIntersect (l1 l2 : ParamLine) : Prop :=
  ∃ (s t : ℝ), l1.origin.x + s * l1.direction.x = l2.origin.x + t * l2.direction.x ∧
                l1.origin.y + s * l1.direction.y = l2.origin.y + t * l2.direction.y

theorem no_intersection : ¬ linesIntersect line1 line2 := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_l3926_392646


namespace NUMINAMATH_CALUDE_electricity_consumption_for_2_75_yuan_l3926_392610

-- Define the relationship between electricity consumption and charges
def electricity_charge (consumption : ℝ) : ℝ := 0.55 * consumption

-- Theorem statement
theorem electricity_consumption_for_2_75_yuan :
  ∃ (consumption : ℝ), electricity_charge consumption = 2.75 ∧ consumption = 5 :=
sorry

end NUMINAMATH_CALUDE_electricity_consumption_for_2_75_yuan_l3926_392610


namespace NUMINAMATH_CALUDE_worker_payment_schedule_l3926_392669

/-- Proves that the amount to return for each day not worked is $25 --/
theorem worker_payment_schedule (total_days : Nat) (days_not_worked : Nat) (payment_per_day : Nat) (total_earnings : Nat) :
  total_days = 30 →
  days_not_worked = 24 →
  payment_per_day = 100 →
  total_earnings = 0 →
  (total_days - days_not_worked) * payment_per_day = days_not_worked * 25 := by
  sorry

end NUMINAMATH_CALUDE_worker_payment_schedule_l3926_392669


namespace NUMINAMATH_CALUDE_vector_properties_l3926_392653

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (1, 2)

theorem vector_properties :
  (a.1 * b.1 + a.2 * b.2 = 4) ∧
  ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2 = 2) ∧
  ((a.1 + b.1) * c.1 + (a.2 + b.2) * c.2 = 0) :=
sorry

end NUMINAMATH_CALUDE_vector_properties_l3926_392653


namespace NUMINAMATH_CALUDE_A_closed_under_mult_l3926_392639

/-- The set A of quadratic forms over integers -/
def A : Set ℤ := {n : ℤ | ∃ (a b k : ℤ), n = a^2 + k*a*b + b^2}

/-- A is closed under multiplication -/
theorem A_closed_under_mult :
  ∀ (x y : ℤ), x ∈ A → y ∈ A → (x * y) ∈ A := by
  sorry

end NUMINAMATH_CALUDE_A_closed_under_mult_l3926_392639


namespace NUMINAMATH_CALUDE_cubic_function_uniqueness_l3926_392629

/-- Given a cubic function f(x) = ax^3 - 3x^2 + x + b with a ≠ 0, 
    if the tangent line at x = 1 is 2x + y + 1 = 0, 
    then f(x) = x^3 - 3x^2 + x - 2 -/
theorem cubic_function_uniqueness (a b : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - 3 * x^2 + x + b
  let f' : ℝ → ℝ := λ x ↦ 3 * a * x^2 - 6 * x + 1
  (f' 1 = -2 ∧ f 1 = -3) → f = λ x ↦ x^3 - 3 * x^2 + x - 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_uniqueness_l3926_392629


namespace NUMINAMATH_CALUDE_journey_portions_l3926_392679

/-- Proves that the journey is divided into 5 portions given the conditions -/
theorem journey_portions (total_distance : ℝ) (speed : ℝ) (time : ℝ) (portions_covered : ℕ) :
  total_distance = 35 →
  speed = 40 →
  time = 0.7 →
  portions_covered = 4 →
  (speed * time) / portions_covered = total_distance / 5 :=
by sorry

end NUMINAMATH_CALUDE_journey_portions_l3926_392679


namespace NUMINAMATH_CALUDE_train_length_calculation_l3926_392617

theorem train_length_calculation (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  bridge_length = 300 ∧ crossing_time = 45 ∧ train_speed = 55.99999999999999 →
  2220 = train_speed * crossing_time - bridge_length :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3926_392617


namespace NUMINAMATH_CALUDE_slope_of_line_l_l3926_392666

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 5

-- Define the center of the circle
def center : ℝ × ℝ := (3, 5)

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y - 5 = k * (x - 3)

-- Define points A, B, and P
variables (A B P : ℝ × ℝ)

-- State that A and B are on the circle C
axiom A_on_circle : circle_C A.1 A.2
axiom B_on_circle : circle_C B.1 B.2

-- State that A, B, and P are on line l
axiom A_on_line : ∃ k, line_l k A.1 A.2
axiom B_on_line : ∃ k, line_l k B.1 B.2
axiom P_on_line : ∃ k, line_l k P.1 P.2

-- State that P is on the y-axis
axiom P_on_y_axis : P.1 = 0

-- State the vector relationship
axiom vector_relation : 2 * (A.1 - P.1, A.2 - P.2) = (B.1 - P.1, B.2 - P.2)

-- Theorem to prove
theorem slope_of_line_l : ∃ k, (k = 2 ∨ k = -2) ∧ line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧ line_l k P.1 P.2 :=
sorry

end NUMINAMATH_CALUDE_slope_of_line_l_l3926_392666


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3926_392619

theorem complex_equation_solution (x₁ x₂ A : ℂ) (h_distinct : x₁ ≠ x₂)
  (h_eq1 : x₁ * (x₁ + 1) = A)
  (h_eq2 : x₂ * (x₂ + 1) = A)
  (h_eq3 : x₁^4 + 3*x₁^3 + 5*x₁ = x₂^4 + 3*x₂^3 + 5*x₂) :
  A = -7 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3926_392619


namespace NUMINAMATH_CALUDE_circle_line_intersection_and_min_chord_l3926_392673

/-- Circle C: x^2 + y^2 - 4x - 2y - 20 = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 20 = 0

/-- Line l: mx - y - m + 3 = 0 (m ∈ ℝ) -/
def line_l (m x y : ℝ) : Prop := m*x - y - m + 3 = 0

theorem circle_line_intersection_and_min_chord :
  (∀ m : ℝ, ∃ x y : ℝ, circle_C x y ∧ line_l m x y) ∧
  (∃ min_length : ℝ, min_length = 4 * Real.sqrt 5 ∧
    ∀ m : ℝ, ∀ x₁ y₁ x₂ y₂ : ℝ,
      circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l m x₁ y₁ ∧ line_l m x₂ y₂ →
      Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≥ min_length) ∧
  (∃ x y : ℝ, circle_C x y ∧ x - 2*y + 5 = 0 ∧
    ∀ x' y' : ℝ, circle_C x' y' ∧ x' - 2*y' + 5 = 0 →
      Real.sqrt ((x - x')^2 + (y - y')^2) = 4 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_and_min_chord_l3926_392673


namespace NUMINAMATH_CALUDE_fraction_sum_reciprocal_l3926_392631

theorem fraction_sum_reciprocal (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  1 / x + 1 / y = 1 / z → z = (x * y) / (y + x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_reciprocal_l3926_392631


namespace NUMINAMATH_CALUDE_problem_l3926_392632

/-- Given m > 0, prove the following statements -/
theorem problem (m : ℝ) (hm : m > 0) :
  /- If (x+2)(x-6) ≤ 0 implies 2-m ≤ x ≤ 2+m for all x, then m ≥ 4 -/
  ((∀ x, (x + 2) * (x - 6) ≤ 0 → 2 - m ≤ x ∧ x ≤ 2 + m) → m ≥ 4) ∧
  /- If m = 5, and for all x, ((x+2)(x-6) ≤ 0) ∨ (-3 ≤ x ≤ 7) is true, 
     and ((x+2)(x-6) ≤ 0) ∧ (-3 ≤ x ≤ 7) is false, 
     then x ∈ [-3,-2) ∪ (6,7] -/
  (m = 5 → 
    (∀ x, ((x + 2) * (x - 6) ≤ 0 ∨ (-3 ≤ x ∧ x ≤ 7)) ∧
           ¬((x + 2) * (x - 6) ≤ 0 ∧ -3 ≤ x ∧ x ≤ 7)) →
    (∀ x, x ∈ Set.Ioo (-3) (-2) ∪ Set.Ioc 6 7)) :=
by sorry

end NUMINAMATH_CALUDE_problem_l3926_392632


namespace NUMINAMATH_CALUDE_least_valid_number_l3926_392698

def is_valid (n : ℕ) : Prop :=
  n > 1 ∧
  n % 4 = 3 ∧
  n % 5 = 3 ∧
  n % 7 = 3 ∧
  n % 10 = 3 ∧
  n % 11 = 3

theorem least_valid_number : 
  is_valid 1543 ∧ ∀ m : ℕ, m < 1543 → ¬(is_valid m) :=
sorry

end NUMINAMATH_CALUDE_least_valid_number_l3926_392698
