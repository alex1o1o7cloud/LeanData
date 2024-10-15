import Mathlib

namespace NUMINAMATH_CALUDE_table_tennis_match_probability_l1080_108056

/-- The probability of Player A winning a single game -/
def p_A : ℝ := 0.6

/-- The probability of Player B winning a single game -/
def p_B : ℝ := 0.4

/-- The probability of Player A winning the match in a best-of-three format -/
def p_A_wins_match : ℝ := p_A * p_A + p_A * p_B * p_A + p_B * p_A * p_A

theorem table_tennis_match_probability :
  p_A + p_B = 1 →
  p_A_wins_match = 0.648 := by
  sorry

end NUMINAMATH_CALUDE_table_tennis_match_probability_l1080_108056


namespace NUMINAMATH_CALUDE_strawberries_eaten_l1080_108033

-- Define the initial number of strawberries
def initial_strawberries : ℕ := 35

-- Define the remaining number of strawberries
def remaining_strawberries : ℕ := 33

-- Theorem to prove
theorem strawberries_eaten : initial_strawberries - remaining_strawberries = 2 := by
  sorry

end NUMINAMATH_CALUDE_strawberries_eaten_l1080_108033


namespace NUMINAMATH_CALUDE_polynomial_sum_equality_l1080_108026

theorem polynomial_sum_equality : 
  let p (x : ℝ) := 4 * x^2 - 2 * x + 1
  let q (x : ℝ) := -3 * x^2 + x - 5
  let r (x : ℝ) := 2 * x^2 - 4 * x + 3
  ∀ x, p x + q x + r x = 3 * x^2 - 5 * x - 1 := by
    sorry

end NUMINAMATH_CALUDE_polynomial_sum_equality_l1080_108026


namespace NUMINAMATH_CALUDE_greatest_b_quadratic_inequality_l1080_108081

theorem greatest_b_quadratic_inequality :
  ∃ b : ℝ, b^2 - 10*b + 24 ≤ 0 ∧ ∀ x : ℝ, x^2 - 10*x + 24 ≤ 0 → x ≤ b :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_b_quadratic_inequality_l1080_108081


namespace NUMINAMATH_CALUDE_x_value_l1080_108060

theorem x_value (x y : ℚ) (h1 : x / y = 8 / 3) (h2 : y = 27) : x = 72 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1080_108060


namespace NUMINAMATH_CALUDE_whitney_purchase_cost_is_445_62_l1080_108007

/-- Calculates the total cost of Whitney's purchase given the specified conditions --/
def whitneyPurchaseCost : ℝ :=
  let whaleBookCount : ℕ := 15
  let fishBookCount : ℕ := 12
  let sharkBookCount : ℕ := 5
  let magazineCount : ℕ := 8
  let whaleBookPrice : ℝ := 14
  let fishBookPrice : ℝ := 13
  let sharkBookPrice : ℝ := 10
  let magazinePrice : ℝ := 3
  let fishBookDiscount : ℝ := 0.1
  let salesTaxRate : ℝ := 0.05

  let whaleBooksCost := whaleBookCount * whaleBookPrice
  let fishBooksCost := fishBookCount * fishBookPrice * (1 - fishBookDiscount)
  let sharkBooksCost := sharkBookCount * sharkBookPrice
  let magazinesCost := magazineCount * magazinePrice

  let totalBeforeTax := whaleBooksCost + fishBooksCost + sharkBooksCost + magazinesCost
  let salesTax := totalBeforeTax * salesTaxRate
  let totalCost := totalBeforeTax + salesTax

  totalCost

/-- Theorem stating that Whitney's total purchase cost is $445.62 --/
theorem whitney_purchase_cost_is_445_62 : whitneyPurchaseCost = 445.62 := by
  sorry


end NUMINAMATH_CALUDE_whitney_purchase_cost_is_445_62_l1080_108007


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l1080_108070

/-- Given an arithmetic progression where the sum of n terms is 2n^2 + 3n for every n,
    this function represents the r-th term of the progression. -/
def arithmeticProgressionTerm (r : ℕ) : ℕ := 4 * r + 1

/-- The sum of the first n terms of the arithmetic progression. -/
def arithmeticProgressionSum (n : ℕ) : ℕ := 2 * n^2 + 3 * n

/-- Theorem stating that the r-th term of the arithmetic progression is 4r + 1,
    given that the sum of n terms is 2n^2 + 3n for every n. -/
theorem arithmetic_progression_rth_term (r : ℕ) :
  arithmeticProgressionTerm r = arithmeticProgressionSum r - arithmeticProgressionSum (r - 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l1080_108070


namespace NUMINAMATH_CALUDE_sarahs_trip_distance_l1080_108022

theorem sarahs_trip_distance :
  ∀ y : ℚ, (y / 4 + 25 + y / 6 = y) → y = 300 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_trip_distance_l1080_108022


namespace NUMINAMATH_CALUDE_hexagon_perimeter_hexagon_perimeter_proof_l1080_108029

/-- The perimeter of a regular hexagon with side length 8 is 48. -/
theorem hexagon_perimeter : ℕ → ℕ
  | 6 => 48
  | _ => 0

#check hexagon_perimeter
-- hexagon_perimeter : ℕ → ℕ

theorem hexagon_perimeter_proof (n : ℕ) (h : n = 6) : 
  hexagon_perimeter n = 8 * n :=
by sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_hexagon_perimeter_proof_l1080_108029


namespace NUMINAMATH_CALUDE_streetlight_shadow_indeterminate_l1080_108004

-- Define persons A and B
def Person : Type := String

-- Define shadow length under sunlight
def sunShadowLength (p : Person) : ℝ := sorry

-- Define shadow length under streetlight
def streetShadowLength (p : Person) (distance : ℝ) : ℝ := sorry

-- Define the problem conditions
axiom longer_sun_shadow (A B : Person) : sunShadowLength A > sunShadowLength B

-- Theorem stating that the relative shadow lengths under streetlight cannot be determined
theorem streetlight_shadow_indeterminate (A B : Person) :
  ∃ (d1 d2 : ℝ), 
    (streetShadowLength A d1 > streetShadowLength B d2) ∧
    (∃ (d3 d4 : ℝ), streetShadowLength A d3 < streetShadowLength B d4) ∧
    (∃ (d5 d6 : ℝ), streetShadowLength A d5 = streetShadowLength B d6) :=
sorry

end NUMINAMATH_CALUDE_streetlight_shadow_indeterminate_l1080_108004


namespace NUMINAMATH_CALUDE_triangle_angle_from_sides_l1080_108043

theorem triangle_angle_from_sides : 
  ∀ (a b c : ℝ), 
    a = 1 → 
    b = Real.sqrt 7 → 
    c = Real.sqrt 3 → 
    ∃ (A B C : ℝ), 
      A + B + C = π ∧ 
      0 < A ∧ A < π ∧ 
      0 < B ∧ B < π ∧ 
      0 < C ∧ C < π ∧ 
      b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) ∧ 
      B = 5*π/6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_from_sides_l1080_108043


namespace NUMINAMATH_CALUDE_alice_savings_l1080_108037

/-- Alice's savings problem -/
theorem alice_savings (B : ℕ) : 
  let month1 : ℕ := 10
  let month2 : ℕ := month1 + 30 + B
  let month3 : ℕ := month1 + 60
  month1 + month2 + month3 = 120 + B := by
  sorry

end NUMINAMATH_CALUDE_alice_savings_l1080_108037


namespace NUMINAMATH_CALUDE_fraction_equality_implies_x_value_l1080_108015

theorem fraction_equality_implies_x_value :
  ∀ x : ℝ, (4 + 2*x) / (6 + 3*x) = (3 + 2*x) / (5 + 3*x) → x = -2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_x_value_l1080_108015


namespace NUMINAMATH_CALUDE_lineup_combinations_l1080_108040

def team_size : ℕ := 12
def strong_players : ℕ := 4
def positions_to_fill : ℕ := 5

theorem lineup_combinations : 
  (strong_players * (strong_players - 1) * 
   (team_size - 2) * (team_size - 3) * (team_size - 4)) = 8640 := by
  sorry

end NUMINAMATH_CALUDE_lineup_combinations_l1080_108040


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1080_108061

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Part 1
theorem solution_set_part1 :
  ∀ x : ℝ, f (-12) (-2) x < 0 ↔ -1/2 < x ∧ x < 1/3 := by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x : ℝ, f a (-1) x ≥ 0) ↔ a ≥ 1/8 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1080_108061


namespace NUMINAMATH_CALUDE_kendalls_quarters_l1080_108044

/-- Represents the number of coins of each type -/
structure CoinCounts where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ

/-- Calculates the total value of coins in dollars -/
def totalValue (c : CoinCounts) : ℚ :=
  c.quarters * (1/4) + c.dimes * (1/10) + c.nickels * (1/20)

theorem kendalls_quarters :
  ∃ (c : CoinCounts), c.dimes = 12 ∧ c.nickels = 6 ∧ totalValue c = 4 ∧ c.quarters = 10 := by
  sorry

end NUMINAMATH_CALUDE_kendalls_quarters_l1080_108044


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_m_l1080_108084

-- Define the function f(x) = |2x-1|
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < -1/2 ∨ x > 3/2} := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f x + 2 * |x + 3| - 4 > m * x) → m ≤ -11 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_m_l1080_108084


namespace NUMINAMATH_CALUDE_f_value_at_7_5_l1080_108085

def f_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x, f (x + 2) = -f x) ∧  -- f(x+2) = -f(x)
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x)  -- f(x) = x for 0 ≤ x ≤ 1

theorem f_value_at_7_5 (f : ℝ → ℝ) (h : f_conditions f) : f 7.5 = -0.5 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_7_5_l1080_108085


namespace NUMINAMATH_CALUDE_quadratic_equation_one_solution_positive_n_value_l1080_108008

theorem quadratic_equation_one_solution (n : ℝ) : 
  (∃! x : ℝ, 4 * x^2 + n * x + 16 = 0) → n = 16 ∨ n = -16 :=
by sorry

theorem positive_n_value (n : ℝ) :
  (∃! x : ℝ, 4 * x^2 + n * x + 16 = 0) ∧ n > 0 → n = 16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_one_solution_positive_n_value_l1080_108008


namespace NUMINAMATH_CALUDE_town_businesses_town_businesses_proof_l1080_108034

theorem town_businesses : ℕ → Prop :=
  fun total_businesses =>
    let fired := total_businesses / 2
    let quit := total_businesses / 3
    let can_apply := 12
    fired + quit + can_apply = total_businesses ∧ total_businesses = 72

-- Proof
theorem town_businesses_proof : ∃ n : ℕ, town_businesses n := by
  sorry

end NUMINAMATH_CALUDE_town_businesses_town_businesses_proof_l1080_108034


namespace NUMINAMATH_CALUDE_larger_cube_surface_area_l1080_108021

theorem larger_cube_surface_area (small_cube_surface_area : ℝ) (num_small_cubes : ℕ) :
  small_cube_surface_area = 24 →
  num_small_cubes = 125 →
  ∃ (larger_cube_surface_area : ℝ), larger_cube_surface_area = 600 := by
  sorry

end NUMINAMATH_CALUDE_larger_cube_surface_area_l1080_108021


namespace NUMINAMATH_CALUDE_prob_less_than_three_heads_in_eight_flips_prob_less_than_three_heads_in_eight_flips_proof_l1080_108078

/-- The probability of getting fewer than 3 heads in 8 fair coin flips -/
theorem prob_less_than_three_heads_in_eight_flips : ℚ :=
  37 / 256

/-- Proof that the probability of getting fewer than 3 heads in 8 fair coin flips is 37/256 -/
theorem prob_less_than_three_heads_in_eight_flips_proof :
  prob_less_than_three_heads_in_eight_flips = 37 / 256 := by
  sorry


end NUMINAMATH_CALUDE_prob_less_than_three_heads_in_eight_flips_prob_less_than_three_heads_in_eight_flips_proof_l1080_108078


namespace NUMINAMATH_CALUDE_opposite_roots_iff_ab_eq_c_l1080_108072

-- Define the cubic polynomial f(x) = x^3 + a x^2 + b x + c
def f (a b c x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

-- Define a predicate for when two roots are opposite numbers
def has_opposite_roots (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, f a b c x = 0 ∧ f a b c y = 0 ∧ y = -x

-- State the theorem
theorem opposite_roots_iff_ab_eq_c (a b c : ℝ) (h : b ≤ 0) :
  has_opposite_roots a b c ↔ a * b = c :=
sorry

end NUMINAMATH_CALUDE_opposite_roots_iff_ab_eq_c_l1080_108072


namespace NUMINAMATH_CALUDE_krishan_money_l1080_108020

/-- Represents the money ratios and changes for Ram, Gopal, Shyam, and Krishan --/
structure MoneyProblem where
  ram_initial : ℚ
  gopal_initial : ℚ
  shyam_initial : ℚ
  krishan_ratio : ℚ
  ram_increase_percent : ℚ
  shyam_decrease_percent : ℚ
  ram_final : ℚ
  shyam_final : ℚ

/-- Theorem stating that given the conditions, Krishan's money is 3400 --/
theorem krishan_money (p : MoneyProblem)
  (h1 : p.ram_initial = 7)
  (h2 : p.gopal_initial = 17)
  (h3 : p.shyam_initial = 10)
  (h4 : p.krishan_ratio = 16)
  (h5 : p.ram_increase_percent = 18.5)
  (h6 : p.shyam_decrease_percent = 20)
  (h7 : p.ram_final = 699.8)
  (h8 : p.shyam_final = 800)
  (h9 : p.gopal_initial / p.ram_initial = 8 / p.krishan_ratio)
  (h10 : p.gopal_initial / p.shyam_initial = 8 / 9) :
  ∃ (x : ℚ), x * p.krishan_ratio = 3400 := by
  sorry


end NUMINAMATH_CALUDE_krishan_money_l1080_108020


namespace NUMINAMATH_CALUDE_total_ways_to_draw_balls_l1080_108096

-- Define the total number of balls
def total_balls : ℕ := 8

-- Define the number of black balls
def black_balls : ℕ := 2

-- Define the number of white balls
def white_balls : ℕ := 6

-- Define a function to calculate the number of ways to draw balls
def ways_to_draw_balls : ℕ := 
  -- Sum of ways for each possible draw (1st, 2nd, 3rd, and 4th)
  1 + 2 + 3 + 4

-- Theorem statement
theorem total_ways_to_draw_balls : 
  ways_to_draw_balls = 10 :=
by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_total_ways_to_draw_balls_l1080_108096


namespace NUMINAMATH_CALUDE_movie_concessions_cost_l1080_108071

/-- Calculates the amount spent on concessions given the total cost of a movie trip and ticket prices. -/
theorem movie_concessions_cost 
  (total_cost : ℝ) 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (adult_ticket_price : ℝ) 
  (child_ticket_price : ℝ) 
  (h1 : total_cost = 76) 
  (h2 : num_adults = 5) 
  (h3 : num_children = 2) 
  (h4 : adult_ticket_price = 10) 
  (h5 : child_ticket_price = 7) : 
  total_cost - (num_adults * adult_ticket_price + num_children * child_ticket_price) = 12 := by
sorry


end NUMINAMATH_CALUDE_movie_concessions_cost_l1080_108071


namespace NUMINAMATH_CALUDE_sine_three_fourths_pi_minus_alpha_l1080_108017

theorem sine_three_fourths_pi_minus_alpha (α : ℝ) 
  (h : Real.sin (π / 4 + α) = 3 / 5) : 
  Real.sin (3 * π / 4 - α) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sine_three_fourths_pi_minus_alpha_l1080_108017


namespace NUMINAMATH_CALUDE_yellow_parrots_count_l1080_108064

theorem yellow_parrots_count (total : ℕ) (red_fraction : ℚ) : 
  total = 108 → red_fraction = 5/6 → (1 - red_fraction) * total = 18 := by
  sorry

end NUMINAMATH_CALUDE_yellow_parrots_count_l1080_108064


namespace NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l1080_108039

/-- Given a cube with edge length 4, the surface area of its inscribed sphere is 16π. -/
theorem inscribed_sphere_surface_area (edge_length : ℝ) (h : edge_length = 4) :
  let radius : ℝ := edge_length / 2
  4 * π * radius^2 = 16 * π :=
by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l1080_108039


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_l1080_108052

-- Define a triangle with side lengths a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_angle_ratio (t : Triangle) 
  (h1 : t.a^2 = t.b * (t.b + t.c)) -- Given condition
  (h2 : t.A > 0 ∧ t.B > 0 ∧ t.C > 0) -- Angles are positive
  (h3 : t.A + t.B + t.C = π) -- Sum of angles in a triangle
  (h4 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0) -- Side lengths are positive
  : t.B / t.A = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_ratio_l1080_108052


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1080_108010

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | |x - 1| > 1}

-- State the theorem
theorem complement_of_A_in_U : 
  (Set.univ \ A) = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1080_108010


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_bound_l1080_108024

theorem quadratic_roots_sum_bound (p : ℝ) (r₁ r₂ : ℝ) : 
  (r₁ ≠ r₂) →
  (r₁^2 + p*r₁ + 7 = 0) →
  (r₂^2 + p*r₂ + 7 = 0) →
  |r₁ + r₂| > 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_bound_l1080_108024


namespace NUMINAMATH_CALUDE_a_four_plus_b_four_l1080_108035

theorem a_four_plus_b_four (a b : ℝ) (h1 : a^2 - b^2 = 10) (h2 : a * b = 8) : a^4 + b^4 = 548 := by
  sorry

end NUMINAMATH_CALUDE_a_four_plus_b_four_l1080_108035


namespace NUMINAMATH_CALUDE_triangle_properties_l1080_108053

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sin t.A + Real.sqrt 3 * Real.cos t.A = 0)
  (h2 : t.a = 2 * Real.sqrt 7)
  (h3 : t.b = 2) :
  t.A = 2 * Real.pi / 3 ∧ 
  t.c = 4 ∧ 
  1/2 * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1080_108053


namespace NUMINAMATH_CALUDE_carriage_equation_correct_l1080_108031

/-- Represents the scenario of people and carriages as described in the ancient Chinese problem --/
def carriage_problem (x : ℕ) : Prop :=
  -- Three people sharing a carriage leaves two carriages empty
  (3 * (x - 2) : ℤ) = (3 * x - 6 : ℤ) ∧
  -- Two people sharing a carriage leaves nine people walking
  (2 * x + 9 : ℤ) = (3 * x - 6 : ℤ)

/-- The equation 3(x-2) = 2x + 9 correctly represents the carriage problem --/
theorem carriage_equation_correct (x : ℕ) :
  carriage_problem x ↔ (3 * (x - 2) : ℤ) = (2 * x + 9 : ℤ) :=
sorry

end NUMINAMATH_CALUDE_carriage_equation_correct_l1080_108031


namespace NUMINAMATH_CALUDE_equation_solution_l1080_108080

theorem equation_solution (n : ℚ) : 
  (2 / (n + 2) + 3 / (n + 2) + (2 * n) / (n + 2) = 4) → n = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1080_108080


namespace NUMINAMATH_CALUDE_horner_rule_operations_l1080_108014

/-- Horner's Rule representation of a polynomial -/
def horner_representation (coeffs : List ℤ) : ℤ → ℤ :=
  fun x => coeffs.foldl (fun acc a => acc * x + a) 0

/-- Count of operations in Horner's Rule evaluation -/
def horner_operation_count (coeffs : List ℤ) : ℕ × ℕ :=
  (coeffs.length - 1, coeffs.length - 1)

/-- The polynomial f(x) = 4x^5 - 3x^4 + 6x - 9 -/
def f : List ℤ := [4, -3, 0, 0, 6, -9]

theorem horner_rule_operations :
  horner_operation_count f = (5, 3) := by sorry

end NUMINAMATH_CALUDE_horner_rule_operations_l1080_108014


namespace NUMINAMATH_CALUDE_mens_wages_l1080_108093

/-- Proves that the total wages of men is Rs. 30 given the problem conditions -/
theorem mens_wages (W : ℕ) (total_earnings : ℕ) : 
  (5 : ℕ) = W →  -- 5 men are equal to W women
  W = 8 →        -- W women are equal to 8 boys
  total_earnings = 90 →  -- Total earnings of all people is Rs. 90
  (5 : ℕ) * (total_earnings / 15) = 30 := by
sorry

end NUMINAMATH_CALUDE_mens_wages_l1080_108093


namespace NUMINAMATH_CALUDE_not_cube_of_integer_l1080_108045

theorem not_cube_of_integer : ¬ ∃ (k : ℤ), 10^202 + 5 * 10^101 + 1 = k^3 := by sorry

end NUMINAMATH_CALUDE_not_cube_of_integer_l1080_108045


namespace NUMINAMATH_CALUDE_stadium_length_yards_l1080_108000

-- Define the length of the stadium in feet
def stadium_length_feet : ℕ := 186

-- Define the number of feet in a yard
def feet_per_yard : ℕ := 3

-- Theorem to prove the length of the stadium in yards
theorem stadium_length_yards : 
  stadium_length_feet / feet_per_yard = 62 := by
  sorry

end NUMINAMATH_CALUDE_stadium_length_yards_l1080_108000


namespace NUMINAMATH_CALUDE_f_properties_l1080_108062

def f (x : ℝ) := x^3 - 6*x + 5

theorem f_properties :
  let sqrt2 := Real.sqrt 2
  ∀ x y : ℝ,
  (∀ x ∈ Set.Ioo (-sqrt2) sqrt2, ∀ y ∈ Set.Ioo (-sqrt2) sqrt2, x < y → f x > f y) ∧
  (∀ x ∈ Set.Iic (-sqrt2), ∀ y ∈ Set.Iic (-sqrt2), x < y → f x < f y) ∧
  (∀ x ∈ Set.Ioi sqrt2, ∀ y ∈ Set.Ioi sqrt2, x < y → f x < f y) ∧
  (f (-sqrt2) = 5 + 4*sqrt2) ∧
  (f sqrt2 = 5 - 4*sqrt2) ∧
  (∀ a : ℝ, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = a ∧ f y = a ∧ f z = a) ↔
    5 - 4*sqrt2 < a ∧ a < 5 + 4*sqrt2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1080_108062


namespace NUMINAMATH_CALUDE_largest_value_l1080_108018

theorem largest_value (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b > 1/2 ∧ b > a^2 + b^2 ∧ b > 2*a*b := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l1080_108018


namespace NUMINAMATH_CALUDE_range_of_a_for_M_subset_N_l1080_108090

/-- The set of real numbers m for which x^2 - x - m = 0 has solutions in (-1, 1) -/
def M : Set ℝ :=
  {m : ℝ | ∃ x, -1 < x ∧ x < 1 ∧ x^2 - x - m = 0}

/-- The solution set of (x - a)(x + a - 2) < 0 -/
def N (a : ℝ) : Set ℝ :=
  {x : ℝ | (x - a) * (x + a - 2) < 0}

/-- The theorem stating the range of a values for which M ⊆ N(a) -/
theorem range_of_a_for_M_subset_N :
  {a : ℝ | M ⊆ N a} = {a : ℝ | a < -1/4 ∨ a > 9/4} := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_M_subset_N_l1080_108090


namespace NUMINAMATH_CALUDE_hannah_total_spent_l1080_108086

def sweatshirt_count : ℕ := 3
def tshirt_count : ℕ := 2
def sweatshirt_price : ℕ := 15
def tshirt_price : ℕ := 10

theorem hannah_total_spent :
  sweatshirt_count * sweatshirt_price + tshirt_count * tshirt_price = 65 :=
by sorry

end NUMINAMATH_CALUDE_hannah_total_spent_l1080_108086


namespace NUMINAMATH_CALUDE_largest_common_divisor_360_315_l1080_108001

theorem largest_common_divisor_360_315 : Nat.gcd 360 315 = 45 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_360_315_l1080_108001


namespace NUMINAMATH_CALUDE_correct_oranges_put_back_l1080_108002

/-- Represents the fruit selection problem with given prices and quantities -/
structure FruitSelection where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  desired_avg_price : ℚ

/-- Calculates the number of oranges to put back to achieve the desired average price -/
def oranges_to_put_back (fs : FruitSelection) : ℕ :=
  2

/-- Theorem stating that putting back the calculated number of oranges achieves the desired average price -/
theorem correct_oranges_put_back (fs : FruitSelection)
  (h1 : fs.apple_price = 40/100)
  (h2 : fs.orange_price = 60/100)
  (h3 : fs.total_fruits = 10)
  (h4 : fs.initial_avg_price = 48/100)
  (h5 : fs.desired_avg_price = 45/100) :
  let num_oranges_back := oranges_to_put_back fs
  let remaining_fruits := fs.total_fruits - num_oranges_back
  let num_apples := 6  -- Derived from the problem's solution
  let num_oranges := 4 -- Derived from the problem's solution
  fs.apple_price * num_apples + fs.orange_price * (num_oranges - num_oranges_back) =
    fs.desired_avg_price * remaining_fruits :=
by
  sorry

end NUMINAMATH_CALUDE_correct_oranges_put_back_l1080_108002


namespace NUMINAMATH_CALUDE_outfits_count_l1080_108036

/-- Number of red shirts -/
def red_shirts : ℕ := 4

/-- Number of green shirts -/
def green_shirts : ℕ := 4

/-- Number of blue shirts -/
def blue_shirts : ℕ := 4

/-- Number of pants -/
def pants : ℕ := 7

/-- Number of green hats -/
def green_hats : ℕ := 6

/-- Number of red hats -/
def red_hats : ℕ := 6

/-- Number of blue hats -/
def blue_hats : ℕ := 6

/-- Calculate the number of outfits with different colored shirts and hats -/
def outfits : ℕ := 
  (red_shirts * pants * (green_hats + blue_hats)) +
  (green_shirts * pants * (red_hats + blue_hats)) +
  (blue_shirts * pants * (red_hats + green_hats))

theorem outfits_count : outfits = 1008 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l1080_108036


namespace NUMINAMATH_CALUDE_total_spent_is_23_88_l1080_108041

def green_grape_price : ℝ := 2.79
def red_grape_price : ℝ := 3.25
def regular_cherry_price : ℝ := 4.90
def organic_cherry_price : ℝ := 5.75

def green_grape_weight : ℝ := 2.5
def red_grape_weight : ℝ := 1.8
def regular_cherry_weight : ℝ := 1.2
def organic_cherry_weight : ℝ := 0.9

def total_spent : ℝ :=
  green_grape_price * green_grape_weight +
  red_grape_price * red_grape_weight +
  regular_cherry_price * regular_cherry_weight +
  organic_cherry_price * organic_cherry_weight

theorem total_spent_is_23_88 : total_spent = 23.88 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_23_88_l1080_108041


namespace NUMINAMATH_CALUDE_total_amount_proof_l1080_108003

/-- Given an amount divided into two parts, where one part is invested at 3% and the other at 5%,
    this theorem proves that the total amount is 4000 when the first part is 2800 and
    the total annual interest is 144. -/
theorem total_amount_proof (T A : ℝ) : 
  A = 2800 → 
  0.03 * A + 0.05 * (T - A) = 144 → 
  T = 4000 := by
sorry

end NUMINAMATH_CALUDE_total_amount_proof_l1080_108003


namespace NUMINAMATH_CALUDE_three_W_four_l1080_108068

-- Define the operation W
def W (a b : ℤ) : ℤ := b + 5*a - 3*a^2

-- Theorem statement
theorem three_W_four : W 3 4 = -8 := by sorry

end NUMINAMATH_CALUDE_three_W_four_l1080_108068


namespace NUMINAMATH_CALUDE_point_A_coordinates_l1080_108057

-- Define the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

-- Define the translation operations
def translateLeft (p : Point) (d : ℝ) : Point :=
  { x := p.x - d, y := p.y }

def translateUp (p : Point) (d : ℝ) : Point :=
  { x := p.x, y := p.y + d }

-- Theorem statement
theorem point_A_coordinates 
  (A : Point) 
  (B : Point)
  (C : Point)
  (hB : ∃ d : ℝ, translateLeft A d = B)
  (hC : ∃ d : ℝ, translateUp A d = C)
  (hBcoord : B.x = 1 ∧ B.y = 2)
  (hCcoord : C.x = 3 ∧ C.y = 4) :
  A.x = 3 ∧ A.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l1080_108057


namespace NUMINAMATH_CALUDE_arcsin_one_half_equals_pi_over_six_l1080_108050

theorem arcsin_one_half_equals_pi_over_six : 
  Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_equals_pi_over_six_l1080_108050


namespace NUMINAMATH_CALUDE_custom_op_equation_solution_l1080_108077

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := a^2 + b^2 - a*b

-- State the theorem
theorem custom_op_equation_solution :
  ∀ x : ℝ, custom_op x (x - 1) = 3 ↔ x = 2 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_equation_solution_l1080_108077


namespace NUMINAMATH_CALUDE_fourth_root_simplification_l1080_108009

theorem fourth_root_simplification (x : ℝ) (hx : x > 0) :
  (x^3 * (x^5)^(1/2))^(1/4) = x^(11/8) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_simplification_l1080_108009


namespace NUMINAMATH_CALUDE_vehicle_speeds_l1080_108095

/-- Proves that given the conditions of the problem, the bus speed is 20 km/h and the car speed is 60 km/h -/
theorem vehicle_speeds (distance : ℝ) (bus_speed : ℝ) (car_speed : ℝ) (bus_departure : ℝ) (car_departure : ℝ) (arrival_difference : ℝ) :
  distance = 80 ∧
  car_departure = bus_departure + 3 ∧
  car_speed = 3 * bus_speed ∧
  arrival_difference = 1/3 ∧
  distance / bus_speed = distance / car_speed + (car_departure - bus_departure) - arrival_difference →
  bus_speed = 20 ∧ car_speed = 60 := by
sorry


end NUMINAMATH_CALUDE_vehicle_speeds_l1080_108095


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l1080_108067

/-- A color type representing red, white, and blue -/
inductive Color
  | Red
  | White
  | Blue

/-- A point in the grid -/
structure Point where
  x : Nat
  y : Nat

/-- A coloring function that assigns a color to each point in the grid -/
def Coloring := Point → Color

/-- A rectangle in the grid -/
structure Rectangle where
  topLeft : Point
  bottomRight : Point

/-- Theorem stating the existence of a monochromatic rectangle in a 12x12 grid -/
theorem monochromatic_rectangle_exists (coloring : Coloring) :
  ∃ (rect : Rectangle) (c : Color),
    rect.topLeft.x ≤ 12 ∧ rect.topLeft.y ≤ 12 ∧
    rect.bottomRight.x ≤ 12 ∧ rect.bottomRight.y ≤ 12 ∧
    coloring rect.topLeft = c ∧
    coloring { x := rect.topLeft.x, y := rect.bottomRight.y } = c ∧
    coloring { x := rect.bottomRight.x, y := rect.topLeft.y } = c ∧
    coloring rect.bottomRight = c := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l1080_108067


namespace NUMINAMATH_CALUDE_phase_shift_sine_function_l1080_108074

/-- The phase shift of the function y = 4 sin(3x - π/4) is π/12 to the right -/
theorem phase_shift_sine_function :
  let f : ℝ → ℝ := λ x => 4 * Real.sin (3 * x - π / 4)
  ∃ (shift : ℝ), shift = π / 12 ∧ 
    ∀ x, f (x + shift) = 4 * Real.sin (3 * x) :=
by sorry

end NUMINAMATH_CALUDE_phase_shift_sine_function_l1080_108074


namespace NUMINAMATH_CALUDE_ratio_equality_l1080_108087

theorem ratio_equality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hbc : b/c = 2005) (hcb : c/b = 2005) :
  (b + c) / (a + b) = 2005 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1080_108087


namespace NUMINAMATH_CALUDE_grant_total_sales_l1080_108058

def baseball_cards_price : ℝ := 25
def baseball_bat_price : ℝ := 10
def baseball_glove_original_price : ℝ := 30
def baseball_glove_discount : ℝ := 0.2
def baseball_cleats_price : ℝ := 10
def baseball_cleats_count : ℕ := 2

def total_sales : ℝ :=
  baseball_cards_price +
  baseball_bat_price +
  (baseball_glove_original_price * (1 - baseball_glove_discount)) +
  (baseball_cleats_price * baseball_cleats_count)

theorem grant_total_sales :
  total_sales = 79 := by sorry

end NUMINAMATH_CALUDE_grant_total_sales_l1080_108058


namespace NUMINAMATH_CALUDE_original_equals_scientific_l1080_108094

/-- Represent a number in scientific notation -/
structure ScientificNotation where
  mantissa : ℝ
  exponent : ℤ
  mantissa_range : 1 ≤ mantissa ∧ mantissa < 10

/-- The given number -/
def original_number : ℕ := 3010000000

/-- The scientific notation representation of the original number -/
def scientific_representation : ScientificNotation := {
  mantissa := 3.01,
  exponent := 9,
  mantissa_range := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_representation.mantissa * (10 : ℝ) ^ scientific_representation.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l1080_108094


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l1080_108054

theorem line_intercepts_sum (c : ℝ) : 
  (∃ (x y : ℝ), 6*x + 9*y + c = 0 ∧ x + y = 30) → c = -108 := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l1080_108054


namespace NUMINAMATH_CALUDE_minimum_rows_needed_l1080_108089

structure School where
  students : ℕ
  h1 : 1 ≤ students
  h2 : students ≤ 39

def City := List School

def totalStudents (city : City) : ℕ :=
  city.map (λ s => s.students) |>.sum

theorem minimum_rows_needed (city : City) 
  (h_total : totalStudents city = 1990) 
  (h_seats_per_row : ℕ := 199) : ℕ :=
  12

#check minimum_rows_needed

end NUMINAMATH_CALUDE_minimum_rows_needed_l1080_108089


namespace NUMINAMATH_CALUDE_cow_manure_plant_height_l1080_108047

/-- The height of the cow manure plant given the heights of control and bone meal plants -/
theorem cow_manure_plant_height
  (control_height : ℝ)
  (bone_meal_percentage : ℝ)
  (cow_manure_percentage : ℝ)
  (h1 : control_height = 36)
  (h2 : bone_meal_percentage = 1.25)
  (h3 : cow_manure_percentage = 2) :
  control_height * bone_meal_percentage * cow_manure_percentage = 90 := by
  sorry

#check cow_manure_plant_height

end NUMINAMATH_CALUDE_cow_manure_plant_height_l1080_108047


namespace NUMINAMATH_CALUDE_defective_pens_l1080_108030

/-- The number of defective pens in a box of 12 pens, given the probability of selecting two non-defective pens. -/
theorem defective_pens (total : ℕ) (prob : ℚ) (h_total : total = 12) (h_prob : prob = 22727272727272727 / 100000000000000000) :
  ∃ (defective : ℕ), defective = 6 ∧ 
    (prob = (↑(total - defective) / ↑total) * (↑(total - defective - 1) / ↑(total - 1))) :=
by sorry

end NUMINAMATH_CALUDE_defective_pens_l1080_108030


namespace NUMINAMATH_CALUDE_race_outcome_l1080_108023

/-- Represents the distance traveled by an animal at a given time --/
structure DistanceTime where
  distance : ℝ
  time : ℝ

/-- Represents the race between a tortoise and a hare --/
structure Race where
  tortoise : List DistanceTime
  hare : List DistanceTime

/-- Checks if a list of DistanceTime points represents a steady pace --/
def isSteadyPace (points : List DistanceTime) : Prop := sorry

/-- Checks if a list of DistanceTime points has exactly two stops --/
def hasTwoStops (points : List DistanceTime) : Prop := sorry

/-- Checks if the first point in a list finishes before the first point in another list --/
def finishesFirst (winner loser : List DistanceTime) : Prop := sorry

/-- Theorem representing the race conditions and outcome --/
theorem race_outcome (race : Race) : 
  isSteadyPace race.tortoise ∧ 
  hasTwoStops race.hare ∧ 
  finishesFirst race.tortoise race.hare := by
  sorry

#check race_outcome

end NUMINAMATH_CALUDE_race_outcome_l1080_108023


namespace NUMINAMATH_CALUDE_A_intersect_B_l1080_108012

def A : Set ℕ := {1, 2, 4, 6, 8}

def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}

theorem A_intersect_B : A ∩ B = {2, 4, 8} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1080_108012


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l1080_108042

/-- Given a line L1: 2x + 3y = 12, and a perpendicular line L2 with y-intercept -1,
    the x-intercept of L2 is 2/3. -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 2 * x + 3 * y = 12
  let m1 : ℝ := -2 / 3  -- slope of L1
  let m2 : ℝ := 3 / 2   -- slope of L2 (perpendicular to L1)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ y = m2 * x - 1  -- equation of L2
  (∀ x y, L2 x y → (x = 0 → y = -1)) →  -- y-intercept of L2 is -1
  (∀ x, L2 x 0 → x = 2/3) :=  -- x-intercept of L2 is 2/3
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l1080_108042


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l1080_108049

/-- Given two lines that intersect at x = -10, prove the value of k -/
theorem intersection_point_k_value :
  let line1 : ℝ → ℝ → ℝ := λ x y => -3 * x + y
  let line2 : ℝ → ℝ → ℝ := λ x y => 0.75 * x + y
  let k : ℝ := line1 (-10) (line2 (-10) 20)
  k = 57.5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l1080_108049


namespace NUMINAMATH_CALUDE_floor_equality_condition_l1080_108066

theorem floor_equality_condition (a b : ℝ) :
  (∀ n : ℕ+, a * ⌊b * n⌋ = b * ⌊a * n⌋) ↔ (a = b ∨ a = 0 ∨ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_floor_equality_condition_l1080_108066


namespace NUMINAMATH_CALUDE_part1_part2_l1080_108055

-- Define propositions p, q, and r as functions of a
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + (a - 1) * x + a^2 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (2 * a^2 - a)^x < (2 * a^2 - a)^y

def r (a : ℝ) : Prop := (2 * a - 1) / (a - 2) ≤ 1

-- Define the range of a for part 1
def range_a (a : ℝ) : Prop := (a ≥ -1 ∧ a < -1/2) ∨ (a > 1/3 ∧ a ≤ 1)

-- Theorem for part 1
theorem part1 (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_a a := by sorry

-- Theorem for part 2
theorem part2 : (∀ a : ℝ, ¬(p a) → r a) ∧ ¬(∀ a : ℝ, r a → ¬(p a)) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1080_108055


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_l1080_108027

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |x - 4| - a

-- Theorem for the minimum value of f when a = 1
theorem min_value_when_a_is_one :
  ∃ (m : ℝ), m = 4 ∧ ∀ (x : ℝ), f 1 x ≥ m :=
sorry

-- Theorem for the range of a
theorem range_of_a :
  ∀ (a : ℝ), (∀ (x : ℝ), f a x ≥ 4/a + 1) ↔ (a < 0 ∨ a = 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_l1080_108027


namespace NUMINAMATH_CALUDE_chess_match_max_ab_l1080_108011

theorem chess_match_max_ab (a b c : ℝ) : 
  0 ≤ a ∧ a < 1 ∧
  0 ≤ b ∧ b < 1 ∧
  0 ≤ c ∧ c < 1 ∧
  a + b + c = 1 ∧
  3*a + b = 1 →
  a * b ≤ 1/12 := by
sorry

end NUMINAMATH_CALUDE_chess_match_max_ab_l1080_108011


namespace NUMINAMATH_CALUDE_optimal_price_and_profit_l1080_108005

/-- Represents the profit function for a product with given pricing conditions -/
def profit_function (x : ℝ) : ℝ := -x^2 + 140*x - 4000

/-- Represents the valid range for the selling price -/
def valid_price_range (x : ℝ) : Prop := 50 ≤ x ∧ x ≤ 100

theorem optimal_price_and_profit :
  ∃ (x : ℝ), 
    valid_price_range x ∧ 
    (∀ y, valid_price_range y → profit_function y ≤ profit_function x) ∧
    x = 70 ∧ 
    profit_function x = 900 :=
sorry

end NUMINAMATH_CALUDE_optimal_price_and_profit_l1080_108005


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1080_108013

/-- Given two vectors a and b in ℝ², where a = (4, 8) and b = (x, 4),
    if a is perpendicular to b, then x = -8. -/
theorem perpendicular_vectors_x_value :
  let a : ℝ × ℝ := (4, 8)
  let b : ℝ × ℝ := (x, 4)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = -8 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1080_108013


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l1080_108065

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with axes parallel to coordinate axes -/
structure Ellipse where
  center : Point
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  ((p.x - e.center.x)^2 / e.semi_major_axis^2) + ((p.y - e.center.y)^2 / e.semi_minor_axis^2) = 1

theorem ellipse_minor_axis_length : 
  ∀ (e : Ellipse),
    let p1 : Point := ⟨-2, 1⟩
    let p2 : Point := ⟨0, 0⟩
    let p3 : Point := ⟨0, 3⟩
    let p4 : Point := ⟨4, 0⟩
    let p5 : Point := ⟨4, 3⟩
    (¬ collinear p1 p2 p3 ∧ ¬ collinear p1 p2 p4 ∧ ¬ collinear p1 p2 p5 ∧ 
     ¬ collinear p1 p3 p4 ∧ ¬ collinear p1 p3 p5 ∧ ¬ collinear p1 p4 p5 ∧ 
     ¬ collinear p2 p3 p4 ∧ ¬ collinear p2 p3 p5 ∧ ¬ collinear p2 p4 p5 ∧ 
     ¬ collinear p3 p4 p5) →
    (pointOnEllipse p1 e ∧ pointOnEllipse p2 e ∧ pointOnEllipse p3 e ∧ 
     pointOnEllipse p4 e ∧ pointOnEllipse p5 e) →
    2 * e.semi_minor_axis = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l1080_108065


namespace NUMINAMATH_CALUDE_two_people_completion_time_l1080_108083

/-- Represents the amount of work done on a given day -/
def work_on_day (n : ℕ) : ℕ := 2^(n-1)

/-- Represents the total amount of work done up to and including a given day -/
def total_work (n : ℕ) : ℕ := 2^n - 1

/-- The number of days it takes one person to complete the job -/
def days_for_one_person : ℕ := 12

/-- The theorem stating that two people working together will complete the job in 11 days -/
theorem two_people_completion_time :
  ∃ (n : ℕ), n = 11 ∧ total_work n = total_work days_for_one_person := by
  sorry

end NUMINAMATH_CALUDE_two_people_completion_time_l1080_108083


namespace NUMINAMATH_CALUDE_coefficient_x_squared_l1080_108097

theorem coefficient_x_squared (m n : ℕ+) : 
  (2 * m.val + 3 * n.val = 13) → 
  (∃ k, k = Nat.choose m.val 2 * 2^2 + Nat.choose n.val 2 * 3^2 ∧ (k = 31 ∨ k = 40)) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_l1080_108097


namespace NUMINAMATH_CALUDE_geometric_sequence_shift_l1080_108076

theorem geometric_sequence_shift (a : ℕ → ℝ) (q c : ℝ) :
  (q ≠ 1) →
  (∀ n, a (n + 1) = q * a n) →
  (∃ r, ∀ n, (a (n + 1) + c) = r * (a n + c)) →
  c = 0 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_shift_l1080_108076


namespace NUMINAMATH_CALUDE_prime_divisor_greater_than_exponent_l1080_108038

theorem prime_divisor_greater_than_exponent (p q : ℕ) : 
  Prime p → Prime q → q > 5 → q ∣ (2^p + 3^p) → q > p := by
  sorry

end NUMINAMATH_CALUDE_prime_divisor_greater_than_exponent_l1080_108038


namespace NUMINAMATH_CALUDE_ben_hours_per_shift_l1080_108046

/-- Represents the time it takes Ben to build one rocking chair -/
def time_per_chair : ℕ := 5

/-- Represents the number of chairs Ben builds in 10 days -/
def chairs_in_ten_days : ℕ := 16

/-- Represents the number of days Ben works -/
def work_days : ℕ := 10

/-- Represents the number of shifts Ben works per day -/
def shifts_per_day : ℕ := 1

/-- Theorem stating that Ben works 8 hours per shift -/
theorem ben_hours_per_shift : 
  (chairs_in_ten_days * time_per_chair) / work_days = 8 := by
  sorry

end NUMINAMATH_CALUDE_ben_hours_per_shift_l1080_108046


namespace NUMINAMATH_CALUDE_person_2019_chooses_left_l1080_108059

def chocolate_distribution (L M R : ℕ+) (n : ℕ) : ℕ :=
  let total := L + M + R
  let full_rounds := n / total
  let remainder := n % total
  let left_count := full_rounds * L.val + min remainder L.val
  let middle_count := full_rounds * M.val + min (remainder - left_count) M.val
  let right_count := full_rounds * R.val + (remainder - left_count - middle_count)
  if (L.val : ℚ) / (left_count + 1) ≥ max ((M.val : ℚ) / (middle_count + 1)) ((R.val : ℚ) / (right_count + 1))
  then 0  -- Left table
  else if (M.val : ℚ) / (middle_count + 1) > (R.val : ℚ) / (right_count + 1)
  then 1  -- Middle table
  else 2  -- Right table

theorem person_2019_chooses_left (L M R : ℕ+) (h1 : L = 9) (h2 : M = 19) (h3 : R = 25) :
  chocolate_distribution L M R 2019 = 0 :=
sorry

end NUMINAMATH_CALUDE_person_2019_chooses_left_l1080_108059


namespace NUMINAMATH_CALUDE_peggy_record_profit_difference_l1080_108091

/-- Represents the profit difference between two offers for a record collection. -/
def profit_difference (total_records : ℕ) (sammy_price : ℕ) (bryan_price_high : ℕ) (bryan_price_low : ℕ) : ℕ :=
  let sammy_offer := total_records * sammy_price
  let bryan_offer := (total_records / 2) * bryan_price_high + (total_records / 2) * bryan_price_low
  sammy_offer - bryan_offer

/-- Theorem stating the profit difference for Peggy's record collection. -/
theorem peggy_record_profit_difference :
  profit_difference 200 4 6 1 = 100 := by
  sorry

end NUMINAMATH_CALUDE_peggy_record_profit_difference_l1080_108091


namespace NUMINAMATH_CALUDE_scale_length_theorem_l1080_108048

/-- Given a scale divided into equal parts, this function calculates its total length -/
def scaleLength (numParts : ℕ) (partLength : ℕ) : ℕ :=
  numParts * partLength

/-- Theorem stating that a scale with 4 parts of 20 inches each has a total length of 80 inches -/
theorem scale_length_theorem :
  scaleLength 4 20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_scale_length_theorem_l1080_108048


namespace NUMINAMATH_CALUDE_dimes_count_l1080_108019

/-- Represents the number of coins of each type --/
structure CoinCounts where
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat

/-- Calculates the total value in cents for a given set of coin counts --/
def totalValue (coins : CoinCounts) : Nat :=
  coins.quarters * 25 + coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- Theorem: Given the total amount and the number of other coins, the number of dimes is 3 --/
theorem dimes_count (coins : CoinCounts) :
  coins.quarters = 10 ∧ coins.nickels = 3 ∧ coins.pennies = 5 ∧ totalValue coins = 300 →
  coins.dimes = 3 := by
  sorry


end NUMINAMATH_CALUDE_dimes_count_l1080_108019


namespace NUMINAMATH_CALUDE_tan_equality_proof_l1080_108073

theorem tan_equality_proof (n : Int) :
  -180 < n ∧ n < 180 → Real.tan (n * π / 180) = Real.tan (210 * π / 180) → n = 30 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_proof_l1080_108073


namespace NUMINAMATH_CALUDE_adult_meals_count_l1080_108028

/-- The number of meals that can feed children -/
def childMeals : ℕ := 90

/-- The number of adults who have their meal -/
def adultsMealed : ℕ := 35

/-- The number of children that can be fed with remaining food after some adults eat -/
def remainingChildMeals : ℕ := 45

/-- The number of meals initially available for adults -/
def adultMeals : ℕ := 80

theorem adult_meals_count :
  adultMeals = childMeals - remainingChildMeals + adultsMealed :=
by sorry

end NUMINAMATH_CALUDE_adult_meals_count_l1080_108028


namespace NUMINAMATH_CALUDE_triangle_area_l1080_108088

/-- Given a triangle with perimeter 28 cm and inradius 2.5 cm, its area is 35 cm² -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 28 → inradius = 2.5 → area = inradius * (perimeter / 2) → area = 35 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1080_108088


namespace NUMINAMATH_CALUDE_no_solution_exists_l1080_108075

theorem no_solution_exists : ¬∃ x : ℝ, 2 < 3 * x ∧ 3 * x < 4 ∧ 1 < 5 * x ∧ 5 * x < 3 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1080_108075


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1080_108051

theorem quadratic_roots_sum (m n : ℝ) : 
  m^2 + 2*m - 2022 = 0 → 
  n^2 + 2*n - 2022 = 0 → 
  m^2 + 3*m + n = 2020 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1080_108051


namespace NUMINAMATH_CALUDE_revenue_change_l1080_108016

/-- Proves that when the price increases by 50% and the quantity sold decreases by 20%, the revenue increases by 20% -/
theorem revenue_change 
  (P Q : ℝ) 
  (P' : ℝ) (hP' : P' = 1.5 * P) 
  (Q' : ℝ) (hQ' : Q' = 0.8 * Q) : 
  P' * Q' = 1.2 * (P * Q) := by
  sorry

#check revenue_change

end NUMINAMATH_CALUDE_revenue_change_l1080_108016


namespace NUMINAMATH_CALUDE_smallest_sum_of_five_consecutive_primes_l1080_108063

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- A function that returns true if five consecutive primes starting from the nth prime sum to a multiple of 3, false otherwise -/
def sumDivisibleByThree (n : ℕ) : Prop :=
  (nthPrime n + nthPrime (n+1) + nthPrime (n+2) + nthPrime (n+3) + nthPrime (n+4)) % 3 = 0

/-- The index of the first prime in the sequence of five consecutive primes that sum to 39 -/
def firstPrimeIndex : ℕ := sorry

theorem smallest_sum_of_five_consecutive_primes :
  (∀ k < firstPrimeIndex, ¬sumDivisibleByThree k) ∧
  sumDivisibleByThree firstPrimeIndex ∧
  nthPrime firstPrimeIndex + nthPrime (firstPrimeIndex+1) + nthPrime (firstPrimeIndex+2) +
  nthPrime (firstPrimeIndex+3) + nthPrime (firstPrimeIndex+4) = 39 := by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_five_consecutive_primes_l1080_108063


namespace NUMINAMATH_CALUDE_polynomial_coefficient_product_l1080_108099

theorem polynomial_coefficient_product (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (x^2 - 2) * (x - 1)^7 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + 
                                     a₄*(x-1)^4 + a₅*(x-1)^5 + a₆*(x-1)^6 + 
                                     a₇*(x-1)^7 + a₈*(x-1)^8 + a₉*(x-1)^9) →
  (a₁ + a₃ + a₅ + a₇ + a₉ + 2) * (a₂ + a₄ + a₆ + a₈) = 4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_product_l1080_108099


namespace NUMINAMATH_CALUDE_inverse_proportion_point_ordering_l1080_108032

/-- Given three points A(-3, y₁), B(-2, y₂), C(3, y₃) on the graph of y = -2/x,
    prove that y₃ < y₁ < y₂ -/
theorem inverse_proportion_point_ordering (y₁ y₂ y₃ : ℝ) : 
  y₁ = -2 / (-3) → y₂ = -2 / (-2) → y₃ = -2 / 3 → y₃ < y₁ ∧ y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_ordering_l1080_108032


namespace NUMINAMATH_CALUDE_no_rational_solution_for_odd_coeff_quadratic_l1080_108098

theorem no_rational_solution_for_odd_coeff_quadratic
  (a b c : ℤ)
  (ha : Odd a)
  (hb : Odd b)
  (hc : Odd c) :
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_for_odd_coeff_quadratic_l1080_108098


namespace NUMINAMATH_CALUDE_martha_exceptional_savings_l1080_108006

/-- Represents Martha's savings over a week -/
def MarthaSavings (daily_allowance : ℚ) (regular_fraction : ℚ) (exceptional_fraction : ℚ) : ℚ :=
  6 * (daily_allowance * regular_fraction) + (daily_allowance * exceptional_fraction)

/-- Theorem stating the fraction Martha saved on the exceptional day -/
theorem martha_exceptional_savings :
  ∀ (daily_allowance : ℚ),
  daily_allowance = 12 →
  MarthaSavings daily_allowance (1/2) (1/4) = 39 :=
by
  sorry


end NUMINAMATH_CALUDE_martha_exceptional_savings_l1080_108006


namespace NUMINAMATH_CALUDE_log_product_interval_l1080_108082

theorem log_product_interval :
  1 < Real.log 6 / Real.log 5 * Real.log 7 / Real.log 6 * Real.log 8 / Real.log 7 ∧
  Real.log 6 / Real.log 5 * Real.log 7 / Real.log 6 * Real.log 8 / Real.log 7 < 2 :=
by sorry

end NUMINAMATH_CALUDE_log_product_interval_l1080_108082


namespace NUMINAMATH_CALUDE_hari_join_time_l1080_108025

/-- Represents the number of months in a year -/
def monthsInYear : ℕ := 12

/-- Praveen's initial investment in rupees -/
def praveenInvestment : ℚ := 3500

/-- Hari's investment in rupees -/
def hariInvestment : ℚ := 9000.000000000002

/-- Profit sharing ratio for Praveen -/
def praveenShare : ℚ := 2

/-- Profit sharing ratio for Hari -/
def hariShare : ℚ := 3

/-- Theorem stating when Hari joined the business -/
theorem hari_join_time : 
  ∃ (x : ℕ), x < monthsInYear ∧ 
  (praveenInvestment * monthsInYear) / (hariInvestment * (monthsInYear - x)) = praveenShare / hariShare ∧
  x = 5 := by sorry

end NUMINAMATH_CALUDE_hari_join_time_l1080_108025


namespace NUMINAMATH_CALUDE_percent_relation_l1080_108069

theorem percent_relation (a b : ℝ) (h : a = 1.2 * b) : 
  2 * b / a = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_percent_relation_l1080_108069


namespace NUMINAMATH_CALUDE_puzzle_solution_l1080_108092

theorem puzzle_solution (a b c d : ℕ+) 
  (h1 : a^6 = b^5) 
  (h2 : c^4 = d^3) 
  (h3 : c - a = 25) : 
  d - b = 561 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l1080_108092


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_less_than_zero_l1080_108079

theorem negation_of_absolute_value_less_than_zero :
  (¬ ∀ x : ℝ, |x| < 0) ↔ (∃ x₀ : ℝ, |x₀| ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_less_than_zero_l1080_108079
