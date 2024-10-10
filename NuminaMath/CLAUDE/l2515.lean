import Mathlib

namespace zeros_of_f_l2515_251543

noncomputable section

-- Define the piecewise function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1/2 then x - 2/x else x^2 + 2*x + a - 1

-- Define the set of zeros for f
def zeros (a : ℝ) : Set ℝ :=
  {x : ℝ | f a x = 0}

-- Theorem statement
theorem zeros_of_f (a : ℝ) (h : a > 0) :
  zeros a = 
    if a > 2 then {Real.sqrt 2}
    else if a = 2 then {Real.sqrt 2, -1}
    else {Real.sqrt 2, -1 + Real.sqrt (2-a), -1 - Real.sqrt (2-a)} :=
by sorry

end

end zeros_of_f_l2515_251543


namespace function_non_negative_iff_k_geq_neg_one_l2515_251562

/-- The function f(x) = |x^2 - 1| + x^2 + kx is non-negative on (0, +∞) if and only if k ≥ -1 -/
theorem function_non_negative_iff_k_geq_neg_one (k : ℝ) :
  (∀ x > 0, |x^2 - 1| + x^2 + k*x ≥ 0) ↔ k ≥ -1 :=
by sorry

end function_non_negative_iff_k_geq_neg_one_l2515_251562


namespace joint_probability_female_literate_l2515_251509

/-- Represents the total number of employees -/
def total_employees : ℕ := 1400

/-- Represents the proportion of female employees -/
def female_ratio : ℚ := 3/5

/-- Represents the proportion of male employees -/
def male_ratio : ℚ := 2/5

/-- Represents the proportion of engineers in the workforce -/
def engineer_ratio : ℚ := 7/20

/-- Represents the proportion of managers in the workforce -/
def manager_ratio : ℚ := 1/4

/-- Represents the proportion of support staff in the workforce -/
def support_ratio : ℚ := 2/5

/-- Represents the overall computer literacy rate -/
def overall_literacy_rate : ℚ := 31/50

/-- Represents the computer literacy rate for male engineers -/
def male_engineer_literacy : ℚ := 4/5

/-- Represents the computer literacy rate for female engineers -/
def female_engineer_literacy : ℚ := 3/4

/-- Represents the computer literacy rate for male managers -/
def male_manager_literacy : ℚ := 11/20

/-- Represents the computer literacy rate for female managers -/
def female_manager_literacy : ℚ := 3/5

/-- Represents the computer literacy rate for male support staff -/
def male_support_literacy : ℚ := 2/5

/-- Represents the computer literacy rate for female support staff -/
def female_support_literacy : ℚ := 1/2

/-- Theorem stating that the joint probability of a randomly selected employee being both female and computer literate is equal to 36.75% -/
theorem joint_probability_female_literate : 
  (female_ratio * engineer_ratio * female_engineer_literacy + 
   female_ratio * manager_ratio * female_manager_literacy + 
   female_ratio * support_ratio * female_support_literacy) = 147/400 := by
  sorry

end joint_probability_female_literate_l2515_251509


namespace complex_fraction_evaluation_l2515_251540

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^15 + b^15) / (a + b)^15 = -2 := by
  sorry

end complex_fraction_evaluation_l2515_251540


namespace polar_sin_is_circle_l2515_251502

-- Define the polar coordinate equation
def polar_equation (ρ θ : ℝ) : Prop := ρ = Real.sin θ

-- Define the transformation from polar to Cartesian coordinates
def to_cartesian (x y ρ θ : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Define a circle in Cartesian coordinates
def is_circle (x y : ℝ) (h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem polar_sin_is_circle :
  ∃ (h k r : ℝ), ∀ (x y ρ θ : ℝ),
    polar_equation ρ θ → to_cartesian x y ρ θ →
    is_circle x y h k r :=
sorry

end polar_sin_is_circle_l2515_251502


namespace product_approx_six_times_number_l2515_251534

-- Define a function to check if two numbers are approximately equal
def approx_equal (x y : ℝ) : Prop := abs (x - y) ≤ 1

-- Theorem 1: The product of 198 × 2 is approximately 400
theorem product_approx : approx_equal (198 * 2) 400 := by sorry

-- Theorem 2: If twice a number is 78, then six times that number is 240
theorem six_times_number (x : ℝ) (h : 2 * x = 78) : 6 * x = 240 := by sorry

end product_approx_six_times_number_l2515_251534


namespace minimum_value_theorem_l2515_251573

theorem minimum_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geometric_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^(2*b))) :
  (∀ x y, x > 0 → y > 0 → 2/x + 1/y ≥ 2/a + 1/b) → 2/a + 1/b = 8 :=
by sorry

end minimum_value_theorem_l2515_251573


namespace total_turtles_l2515_251598

def turtle_problem (lucas rebecca miguel tran pedro kristen kris trey : ℕ) : Prop :=
  lucas = 8 ∧
  rebecca = 2 * lucas ∧
  miguel = rebecca + 10 ∧
  tran = miguel + 5 ∧
  pedro = 2 * tran ∧
  kristen = 3 * pedro ∧
  kris = kristen / 4 ∧
  trey = 5 * kris ∧
  lucas + rebecca + miguel + tran + pedro + kristen + kris + trey = 605

theorem total_turtles :
  ∃ (lucas rebecca miguel tran pedro kristen kris trey : ℕ),
    turtle_problem lucas rebecca miguel tran pedro kristen kris trey :=
by
  sorry

end total_turtles_l2515_251598


namespace cubic_root_product_l2515_251531

theorem cubic_root_product (p q r : ℝ) : 
  (3 * p^3 - 9 * p^2 + 5 * p - 15 = 0) ∧ 
  (3 * q^3 - 9 * q^2 + 5 * q - 15 = 0) ∧ 
  (3 * r^3 - 9 * r^2 + 5 * r - 15 = 0) → 
  p * q * r = 5 := by
sorry

end cubic_root_product_l2515_251531


namespace domino_puzzle_l2515_251599

theorem domino_puzzle (visible_points : ℕ) (num_tiles : ℕ) (grid_size : ℕ) :
  visible_points = 37 →
  num_tiles = 8 →
  grid_size = 4 →
  ∃ (missing_points : ℕ),
    (visible_points + missing_points) % grid_size = 0 ∧
    missing_points ≤ 3 ∧
    ∀ (m : ℕ), m > missing_points →
      (visible_points + m) % grid_size ≠ 0 :=
by sorry


end domino_puzzle_l2515_251599


namespace joey_fraction_of_ethan_time_l2515_251584

def alexa_vacation_days : ℕ := 7 + 2  -- 1 week and 2 days

def joey_learning_days : ℕ := 6

def alexa_vacation_fraction : ℚ := 3/4

theorem joey_fraction_of_ethan_time : 
  (joey_learning_days : ℚ) / ((alexa_vacation_days : ℚ) / alexa_vacation_fraction) = 1/2 := by
  sorry

end joey_fraction_of_ethan_time_l2515_251584


namespace good_price_after_discounts_l2515_251522

theorem good_price_after_discounts (P : ℝ) : 
  P * (1 - 0.20) * (1 - 0.10) * (1 - 0.05) = 6700 → P = 9798.25 := by
  sorry

end good_price_after_discounts_l2515_251522


namespace equation_solution_l2515_251582

theorem equation_solution :
  let f (x : ℝ) := x * ((6 - x) / (x + 1)) * ((6 - x) / (x + 1) + x)
  ∀ x : ℝ, f x = 8 ↔ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 := by
sorry

end equation_solution_l2515_251582


namespace painter_problem_l2515_251594

theorem painter_problem (total_rooms : ℕ) (time_per_room : ℕ) (time_left : ℕ) 
  (h1 : total_rooms = 9)
  (h2 : time_per_room = 8)
  (h3 : time_left = 32) :
  total_rooms - (time_left / time_per_room) = 5 := by
sorry

end painter_problem_l2515_251594


namespace boat_license_count_l2515_251518

/-- Represents the set of possible letters for a boat license -/
def BoatLicenseLetter : Finset Char := {'A', 'M'}

/-- Represents the set of possible digits for a boat license -/
def BoatLicenseDigit : Finset Nat := Finset.range 10

/-- The number of digits in a boat license -/
def BoatLicenseDigitCount : Nat := 5

/-- Calculates the total number of possible boat licenses -/
def TotalBoatLicenses : Nat :=
  BoatLicenseLetter.card * (BoatLicenseDigit.card ^ BoatLicenseDigitCount)

/-- Theorem stating that the total number of boat licenses is 200,000 -/
theorem boat_license_count :
  TotalBoatLicenses = 200000 := by
  sorry

end boat_license_count_l2515_251518


namespace team_combinations_l2515_251553

theorem team_combinations (n k : ℕ) (h1 : n = 7) (h2 : k = 4) :
  Nat.choose n k = 35 := by
  sorry

end team_combinations_l2515_251553


namespace complement_P_subset_Q_l2515_251555

open Set Real

theorem complement_P_subset_Q : 
  let P : Set ℝ := {x | x < 1}
  let Q : Set ℝ := {x | x > -1}
  (compl P : Set ℝ) ⊆ Q := by
  sorry

end complement_P_subset_Q_l2515_251555


namespace enclosed_area_is_3600_l2515_251549

/-- The equation defining the graph -/
def graph_equation (x y : ℝ) : Prop :=
  |x - 120| + |y| = |x/3|

/-- The set of points satisfying the graph equation -/
def graph_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | graph_equation p.1 p.2}

/-- The area enclosed by the graph -/
noncomputable def enclosed_area : ℝ := sorry

/-- Theorem stating that the enclosed area is 3600 -/
theorem enclosed_area_is_3600 : enclosed_area = 3600 := by sorry

end enclosed_area_is_3600_l2515_251549


namespace deepak_third_period_profit_l2515_251537

def anand_investment : ℕ := 22500
def deepak_investment : ℕ := 35000
def total_investment : ℕ := anand_investment + deepak_investment

def first_period_profit : ℕ := 9600
def second_period_profit : ℕ := 12800
def third_period_profit : ℕ := 18000

def profit_share (investment : ℕ) (total_profit : ℕ) : ℚ :=
  (investment : ℚ) / (total_investment : ℚ) * (total_profit : ℚ)

theorem deepak_third_period_profit :
  profit_share deepak_investment third_period_profit = 10960 := by
  sorry

end deepak_third_period_profit_l2515_251537


namespace domain_of_f_l2515_251515

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x^2 - 1)
def domain_f_squared (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | -Real.sqrt 3 ≤ x ∧ x ≤ Real.sqrt 3}

-- Theorem statement
theorem domain_of_f (f : ℝ → ℝ) :
  (∀ x, x ∈ domain_f_squared f → f (x^2 - 1) ≠ 0) →
  (∀ y, f y ≠ 0 → -1 ≤ y ∧ y ≤ 2) :=
sorry

end domain_of_f_l2515_251515


namespace root_magnitude_of_quadratic_l2515_251524

theorem root_magnitude_of_quadratic (z : ℂ) : z^2 + z + 1 = 0 → Complex.abs z = 1 := by
  sorry

end root_magnitude_of_quadratic_l2515_251524


namespace closest_integer_to_cube_root_l2515_251595

theorem closest_integer_to_cube_root : ∃ n : ℤ, 
  n = 10 ∧ ∀ m : ℤ, |n - (7^3 + 9^3)^(1/3)| ≤ |m - (7^3 + 9^3)^(1/3)| := by
  sorry

end closest_integer_to_cube_root_l2515_251595


namespace min_blocks_correct_l2515_251588

/-- A list of positive integer weights representing ice blocks -/
def IceBlocks := List Nat

/-- Predicate to check if a list of weights can satisfy any demand (p, q) where p + q ≤ 2016 -/
def CanSatisfyDemand (blocks : IceBlocks) : Prop :=
  ∀ p q : Nat, p + q ≤ 2016 → ∃ (subsetP subsetQ : List Nat),
    subsetP.Disjoint subsetQ ∧
    subsetP.sum = p ∧
    subsetQ.sum = q ∧
    (subsetP ++ subsetQ).Sublist blocks

/-- The minimum number of ice blocks needed -/
def MinBlocks : Nat := 18

/-- Theorem stating that MinBlocks is the minimum number of ice blocks needed -/
theorem min_blocks_correct :
  (∃ (blocks : IceBlocks), blocks.length = MinBlocks ∧ blocks.all (· > 0) ∧ CanSatisfyDemand blocks) ∧
  (∀ (blocks : IceBlocks), blocks.length < MinBlocks → ¬CanSatisfyDemand blocks) := by
  sorry

#check min_blocks_correct

end min_blocks_correct_l2515_251588


namespace yield_fertilization_correlation_l2515_251571

/-- Represents the yield of crops -/
def CropYield : Type := ℝ

/-- Represents the amount of fertilization -/
def Fertilization : Type := ℝ

/-- Defines the relationship between crop yield and fertilization -/
def dependsOn (y : CropYield) (f : Fertilization) : Prop := ∃ (g : Fertilization → CropYield), y = g f

/-- Defines correlation between two variables -/
def correlated (X Y : Type) : Prop := ∃ (f : X → Y), Function.Injective f ∨ Function.Surjective f

/-- Theorem stating that if crop yield depends on fertilization, then they are correlated -/
theorem yield_fertilization_correlation :
  (∀ y : CropYield, ∀ f : Fertilization, dependsOn y f) →
  correlated Fertilization CropYield :=
by sorry

end yield_fertilization_correlation_l2515_251571


namespace min_contribution_problem_l2515_251500

/-- Proves that given 12 people contributing a total of $20.00, with a maximum individual contribution of $9.00, the minimum amount each person must have contributed is $1.00. -/
theorem min_contribution_problem (n : ℕ) (total : ℚ) (max_contrib : ℚ) (h1 : n = 12) (h2 : total = 20) (h3 : max_contrib = 9) : 
  ∃ (min_contrib : ℚ), 
    min_contrib = 1 ∧ 
    n * min_contrib ≤ total ∧ 
    ∀ (individual_contrib : ℚ), individual_contrib ≤ max_contrib → 
      (n - 1) * min_contrib + individual_contrib ≤ total :=
by sorry

end min_contribution_problem_l2515_251500


namespace brother_d_payment_l2515_251544

theorem brother_d_payment (n : ℕ) (a₁ d : ℚ) (h₁ : n = 5) (h₂ : a₁ = 300) 
  (h₃ : n / 2 * (2 * a₁ + (n - 1) * d) = 1000) : a₁ + 3 * d = 450 := by
  sorry

end brother_d_payment_l2515_251544


namespace circle_slope_range_l2515_251597

theorem circle_slope_range (x y : ℝ) (h : x^2 + (y - 3)^2 = 1) :
  ∃ (k : ℝ), k ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) ∧ y = k * x :=
sorry

end circle_slope_range_l2515_251597


namespace isosceles_triangle_sides_l2515_251514

/-- An isosceles triangle with given leg and base lengths -/
structure IsoscelesTriangle where
  leg : ℝ
  base : ℝ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.leg + t.base

/-- The perimeter of a quadrilateral formed by joining two isosceles triangles along their legs -/
def perimeterQuadLeg (t : IsoscelesTriangle) : ℝ := 2 * t.base + 2 * t.leg

/-- The perimeter of a quadrilateral formed by joining two isosceles triangles along their bases -/
def perimeterQuadBase (t : IsoscelesTriangle) : ℝ := 4 * t.leg

theorem isosceles_triangle_sides (t : IsoscelesTriangle) :
  perimeter t = 100 ∧
  perimeterQuadLeg t + 4 = perimeterQuadBase t →
  t.leg = 34 ∧ t.base = 32 := by
  sorry

end isosceles_triangle_sides_l2515_251514


namespace polynomial_expansion_l2515_251538

theorem polynomial_expansion (x : ℝ) : 
  (3 * x^2 - 4 * x + 3) * (-4 * x^2 + 2 * x - 6) = 
  -12 * x^4 + 22 * x^3 - 38 * x^2 + 30 * x - 18 := by sorry

end polynomial_expansion_l2515_251538


namespace circle_center_l2515_251517

/-- Given a circle with equation (x-2)^2 + (y+1)^2 = 3, prove that its center is at (2, -1) -/
theorem circle_center (x y : ℝ) : (x - 2)^2 + (y + 1)^2 = 3 → (2, -1) = (x, y) := by
  sorry

end circle_center_l2515_251517


namespace min_value_x_plus_y_l2515_251578

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 4/b = 1 → x + y ≤ a + b ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 4/y = 1 ∧ x + y = 9 := by
  sorry

end min_value_x_plus_y_l2515_251578


namespace gan_is_axisymmetric_l2515_251559

/-- A figure is axisymmetric if it can be folded along a line so that the parts on both sides of the line coincide. -/
def is_axisymmetric (figure : Type*) : Prop :=
  ∃ (line : Set figure), ∀ (point : figure), 
    ∃ (reflected_point : figure), 
      point ≠ reflected_point ∧ 
      (point ∈ line ∨ reflected_point ∈ line) ∧
      (∀ (p : figure), p ∉ line → (p = point ↔ p = reflected_point))

/-- The Chinese character "干" -/
def gan : Type* := sorry

/-- Theorem: The Chinese character "干" is an axisymmetric figure -/
theorem gan_is_axisymmetric : is_axisymmetric gan := by sorry

end gan_is_axisymmetric_l2515_251559


namespace find_number_l2515_251503

theorem find_number : ∃ x : ℝ, (38 + 2 * x = 124) ∧ (x = 43) := by
  sorry

end find_number_l2515_251503


namespace large_puzzle_cost_l2515_251512

theorem large_puzzle_cost (small large : ℝ) 
  (h1 : small + large = 23)
  (h2 : large + 3 * small = 39) : 
  large = 15 := by
  sorry

end large_puzzle_cost_l2515_251512


namespace factorial_ratio_72_l2515_251585

theorem factorial_ratio_72 : ∃! (n : ℕ), (Nat.factorial (n + 2)) / (Nat.factorial n) = 72 :=
by
  -- Proof goes here
  sorry

end factorial_ratio_72_l2515_251585


namespace small_birdhouse_price_is_seven_l2515_251504

/-- Represents the price of birdhouses and sales information. -/
structure BirdhouseSales where
  large_price : ℕ
  medium_price : ℕ
  large_sold : ℕ
  medium_sold : ℕ
  small_sold : ℕ
  total_sales : ℕ

/-- Calculates the price of small birdhouses given the sales information. -/
def small_birdhouse_price (sales : BirdhouseSales) : ℕ :=
  (sales.total_sales - (sales.large_price * sales.large_sold + sales.medium_price * sales.medium_sold)) / sales.small_sold

/-- Theorem stating that the price of small birdhouses is $7 given the specific sales information. -/
theorem small_birdhouse_price_is_seven :
  let sales := BirdhouseSales.mk 22 16 2 2 3 97
  small_birdhouse_price sales = 7 := by
  sorry

#eval small_birdhouse_price (BirdhouseSales.mk 22 16 2 2 3 97)

end small_birdhouse_price_is_seven_l2515_251504


namespace power_function_m_value_l2515_251527

/-- A power function that passes through (2, 16) and (1/2, m) -/
def power_function (x : ℝ) : ℝ := x ^ 4

theorem power_function_m_value :
  let f := power_function
  (f 2 = 16) ∧ (∃ m, f (1/2) = m) →
  f (1/2) = 1/16 := by
sorry

end power_function_m_value_l2515_251527


namespace game_not_fair_l2515_251583

/-- Represents the game described in the problem -/
structure Game where
  deck_size : ℕ
  named_cards : ℕ
  win_amount : ℚ
  lose_amount : ℚ

/-- Calculates the expected winnings for the guessing player -/
def expected_winnings (g : Game) : ℚ :=
  let p_named := g.named_cards / g.deck_size
  let p_not_named := 1 - p_named
  let max_cards_per_suit := g.deck_size / 4
  let p_correct_guess_not_named := max_cards_per_suit / (g.deck_size - g.named_cards)
  let expected_case1 := p_named * g.win_amount
  let expected_case2 := p_not_named * (p_correct_guess_not_named * g.win_amount - (1 - p_correct_guess_not_named) * g.lose_amount)
  expected_case1 + expected_case2

/-- The theorem stating that the expected winnings for the guessing player are 1/8 Ft -/
theorem game_not_fair (g : Game) (h1 : g.deck_size = 32) (h2 : g.named_cards = 4) 
    (h3 : g.win_amount = 2) (h4 : g.lose_amount = 1) : 
  expected_winnings g = 1/8 := by
  sorry

end game_not_fair_l2515_251583


namespace euler_family_mean_age_l2515_251579

def euler_family_ages : List ℕ := [6, 6, 6, 6, 12, 14, 14, 16]

theorem euler_family_mean_age : 
  (euler_family_ages.sum / euler_family_ages.length : ℚ) = 10 := by
  sorry

end euler_family_mean_age_l2515_251579


namespace max_squirrel_attacks_l2515_251561

theorem max_squirrel_attacks (N : ℕ+) (a b c : ℤ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a - c = N) : 
  (∃ k : ℕ, k ≤ N ∧ 
    (∀ m : ℕ, m < k → ∃ a' b' c' : ℤ, 
      a' > b' ∧ b' ≥ c' ∧ a' - c' ≤ N - m) ∧
    (∃ a' b' c' : ℤ, a' = b' ∧ b' ≥ c' ∧ a' - c' ≤ N - k)) ∧
  (∀ k : ℕ, k > N → 
    ¬(∀ m : ℕ, m < k → ∃ a' b' c' : ℤ, 
      a' > b' ∧ b' ≥ c' ∧ a' - c' ≤ N - m)) :=
by sorry

end max_squirrel_attacks_l2515_251561


namespace fraction_equality_l2515_251563

theorem fraction_equality : (1 : ℝ) / (2 - Real.sqrt 3) = 2 + Real.sqrt 3 := by
  sorry

end fraction_equality_l2515_251563


namespace sin_two_theta_value_l2515_251567

theorem sin_two_theta_value (θ : Real) 
  (h1 : Real.cos (π/4 - θ) * Real.cos (π/4 + θ) = Real.sqrt 2 / 6)
  (h2 : 0 < θ)
  (h3 : θ < π/2) :
  Real.sin (2 * θ) = Real.sqrt 7 / 3 := by
  sorry

end sin_two_theta_value_l2515_251567


namespace rectangle_area_l2515_251516

/-- The area of a rectangle with perimeter 40 feet and length twice its width is 800/9 square feet. -/
theorem rectangle_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) 
  (h1 : perimeter = 40)
  (h2 : length = 2 * width)
  (h3 : perimeter = 2 * length + 2 * width)
  (h4 : area = length * width) :
  area = 800 / 9 := by
  sorry


end rectangle_area_l2515_251516


namespace triangle_sine_cosine_identity_l2515_251572

/-- For angles A, B, and C of a triangle, sin A + sin B + sin C = 4 cos(A/2) cos(B/2) cos(C/2). -/
theorem triangle_sine_cosine_identity (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin A + Real.sin B + Real.sin C = 4 * Real.cos (A/2) * Real.cos (B/2) * Real.cos (C/2) := by
  sorry

end triangle_sine_cosine_identity_l2515_251572


namespace milk_distribution_l2515_251519

theorem milk_distribution (boxes : Nat) (bottles_per_box : Nat) (eaten : Nat) (people : Nat) :
  boxes = 7 →
  bottles_per_box = 9 →
  eaten = 7 →
  people = 8 →
  (boxes * bottles_per_box - eaten) / people = 7 := by
  sorry

end milk_distribution_l2515_251519


namespace g_continuity_condition_l2515_251576

/-- The function g(x) = 5x - 3 -/
def g (x : ℝ) : ℝ := 5 * x - 3

/-- The statement is true if and only if d ≤ c/5 -/
theorem g_continuity_condition (c d : ℝ) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, |x - 1| < d → |g x - 1| < c) ↔ d ≤ c / 5 := by
  sorry

end g_continuity_condition_l2515_251576


namespace set_A_equality_l2515_251577

def A : Set ℕ := {x | x ≤ 4}

theorem set_A_equality : A = {0, 1, 2, 3, 4} := by sorry

end set_A_equality_l2515_251577


namespace max_university_students_l2515_251511

theorem max_university_students (j m : ℕ) : 
  m = 2 * j + 100 →  -- Max's university has twice Julie's students plus 100
  m + j = 5400 →     -- Total students in both universities
  m = 3632           -- Number of students in Max's university
  := by sorry

end max_university_students_l2515_251511


namespace complete_square_ratio_l2515_251541

/-- Represents a quadratic expression in the form ak² + bk + c -/
structure QuadraticExpression (α : Type*) [Ring α] where
  a : α
  b : α
  c : α

/-- Represents a quadratic expression in completed square form a(k + b)² + c -/
structure CompletedSquareForm (α : Type*) [Ring α] where
  a : α
  b : α
  c : α

/-- Function to convert a QuadraticExpression to CompletedSquareForm -/
def completeSquare {α : Type*} [Field α] (q : QuadraticExpression α) : CompletedSquareForm α :=
  sorry

theorem complete_square_ratio {α : Type*} [Field α] :
  let q : QuadraticExpression α := ⟨4, -8, 16⟩
  let csf := completeSquare q
  csf.c / csf.b = -12 := by sorry

end complete_square_ratio_l2515_251541


namespace line_slope_l2515_251591

/-- The slope of the line (x/2) + (y/3) = 2 is -3/2 -/
theorem line_slope (x y : ℝ) :
  (x / 2 + y / 3 = 2) → (∃ b : ℝ, y = (-3/2) * x + b) := by
  sorry

end line_slope_l2515_251591


namespace f_value_at_7_l2515_251586

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 5

-- State the theorem
theorem f_value_at_7 (a b : ℝ) :
  f a b (-7) = 7 → f a b 7 = -17 := by
  sorry

end f_value_at_7_l2515_251586


namespace exponent_multiplication_l2515_251506

theorem exponent_multiplication (x : ℝ) : x^5 * x^3 = x^8 := by
  sorry

end exponent_multiplication_l2515_251506


namespace chocolate_probability_theorem_not_always_between_probabilities_l2515_251535

structure ChocolateBox where
  white : ℕ
  total : ℕ
  h_total_pos : total > 0

def probability (box : ChocolateBox) : ℚ :=
  box.white / box.total

theorem chocolate_probability_theorem 
  (box1 box2 : ChocolateBox) :
  ∃ (combined : ChocolateBox),
    probability combined > min (probability box1) (probability box2) ∧
    probability combined < max (probability box1) (probability box2) ∧
    combined.white = box1.white + box2.white ∧
    combined.total = box1.total + box2.total :=
sorry

theorem not_always_between_probabilities 
  (box1 box2 : ChocolateBox) :
  ¬ ∀ (combined : ChocolateBox),
    (combined.white = box1.white + box2.white ∧
     combined.total = box1.total + box2.total) →
    (probability combined > min (probability box1) (probability box2) ∧
     probability combined < max (probability box1) (probability box2)) :=
sorry

end chocolate_probability_theorem_not_always_between_probabilities_l2515_251535


namespace two_digit_number_five_times_sum_of_digits_l2515_251575

theorem two_digit_number_five_times_sum_of_digits : 
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ n = 5 * (n / 10 + n % 10) :=
by
  sorry

end two_digit_number_five_times_sum_of_digits_l2515_251575


namespace sufficient_but_not_necessary_l2515_251556

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 5 → (x - 5) * (x + 1) < 0) ∧
  (∃ x, (x - 5) * (x + 1) < 0 ∧ (x < -1 ∨ x > 5)) :=
by sorry

end sufficient_but_not_necessary_l2515_251556


namespace cosine_values_l2515_251501

def terminalPoint : ℝ × ℝ := (-3, 4)

theorem cosine_values (α : ℝ) (h : terminalPoint ∈ {p : ℝ × ℝ | ∃ t, p.1 = t * Real.cos α ∧ p.2 = t * Real.sin α}) :
  Real.cos α = -3/5 ∧ Real.cos (2 * α) = -7/25 := by
  sorry

end cosine_values_l2515_251501


namespace tangent_product_theorem_l2515_251574

theorem tangent_product_theorem : 
  Real.tan (20 * π / 180) * Real.tan (40 * π / 180) * Real.tan (60 * π / 180) * Real.tan (80 * π / 180) = 3 := by
  sorry

end tangent_product_theorem_l2515_251574


namespace system_solution_l2515_251592

theorem system_solution (x y z t : ℤ) :
  (3 * x - 2 * y + 4 * z + 2 * t = 19) →
  (5 * x + 6 * y - 2 * z + 3 * t = 23) →
  (x = 16 * z - 18 * y - 11 ∧ t = 28 * y - 26 * z + 26) :=
by sorry

end system_solution_l2515_251592


namespace initial_cards_calculation_l2515_251521

/-- The number of Pokemon cards Jason gave away -/
def cards_given_away : ℕ := 9

/-- The number of Pokemon cards Jason has left -/
def cards_left : ℕ := 4

/-- The initial number of Pokemon cards Jason had -/
def initial_cards : ℕ := cards_given_away + cards_left

theorem initial_cards_calculation :
  initial_cards = cards_given_away + cards_left :=
by sorry

end initial_cards_calculation_l2515_251521


namespace max_e_value_l2515_251566

def b (n : ℕ) : ℕ := 120 + n^2

def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

theorem max_e_value :
  ∃ (k : ℕ), e k = 5 ∧ ∀ (n : ℕ), e n ≤ 5 :=
sorry

end max_e_value_l2515_251566


namespace pattern_repeats_proof_l2515_251554

/-- The number of beads in one pattern -/
def beads_per_pattern : ℕ := 14

/-- The number of beads in one bracelet -/
def beads_per_bracelet : ℕ := 42

/-- The total number of beads for 1 bracelet and 10 necklaces -/
def total_beads : ℕ := 742

/-- The number of times the pattern repeats per necklace -/
def pattern_repeats_per_necklace : ℕ := 5

/-- Theorem stating that the pattern repeats 5 times per necklace -/
theorem pattern_repeats_proof : 
  beads_per_bracelet + 10 * pattern_repeats_per_necklace * beads_per_pattern = total_beads :=
by sorry

end pattern_repeats_proof_l2515_251554


namespace trig_identity_on_line_l2515_251590

/-- If the terminal side of angle α lies on the line y = 2x, 
    then sin²α - cos²α + sin α * cos α = 1 -/
theorem trig_identity_on_line (α : Real) 
  (h : Real.tan α = 2) : 
  Real.sin α ^ 2 - Real.cos α ^ 2 + Real.sin α * Real.cos α = 1 := by
  sorry

end trig_identity_on_line_l2515_251590


namespace triangle_angle_relation_l2515_251569

theorem triangle_angle_relation (P Q R : Real) (h1 : 5 * Real.sin P + 2 * Real.cos Q = 5) 
  (h2 : 2 * Real.sin Q + 5 * Real.cos P = 3) (h3 : P + Q + R = π) : Real.sin R = 1/20 := by
  sorry

end triangle_angle_relation_l2515_251569


namespace complex_square_l2515_251552

theorem complex_square : (1 + Complex.I) ^ 2 = 2 * Complex.I := by
  sorry

end complex_square_l2515_251552


namespace cracker_ratio_is_one_l2515_251565

/-- The number of crackers Marcus has -/
def marcus_crackers : ℕ := 27

/-- The number of crackers Mona has -/
def mona_crackers : ℕ := marcus_crackers

/-- The ratio of Marcus's crackers to Mona's crackers -/
def cracker_ratio : ℚ := marcus_crackers / mona_crackers

theorem cracker_ratio_is_one : cracker_ratio = 1 := by
  sorry

end cracker_ratio_is_one_l2515_251565


namespace roots_of_cubic_polynomial_l2515_251508

theorem roots_of_cubic_polynomial :
  let p : ℝ → ℝ := λ x => x^3 + x^2 - 6*x - 6
  (∀ x : ℝ, p x = 0 ↔ x = -1 ∨ x = 3 ∨ x = -2) ∧
  (p (-1) = 0) ∧ (p 3 = 0) ∧ (p (-2) = 0) :=
by sorry

end roots_of_cubic_polynomial_l2515_251508


namespace algorithm_uniqueness_false_l2515_251548

-- Define the concept of an algorithm
structure Algorithm where
  finite : Bool
  determinate : Bool
  outputProperty : Bool

-- Define the property of uniqueness for algorithms
def isUnique (problemClass : Type) (alg : Algorithm) : Prop :=
  ∀ (otherAlg : Algorithm), alg = otherAlg

-- Theorem statement
theorem algorithm_uniqueness_false :
  ∃ (problemClass : Type) (alg1 alg2 : Algorithm),
    alg1.finite ∧ alg1.determinate ∧ alg1.outputProperty ∧
    alg2.finite ∧ alg2.determinate ∧ alg2.outputProperty ∧
    alg1 ≠ alg2 :=
sorry

end algorithm_uniqueness_false_l2515_251548


namespace cubic_sum_minus_product_l2515_251525

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 12) 
  (sum_product_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 648 := by sorry

end cubic_sum_minus_product_l2515_251525


namespace divisibility_implies_one_l2515_251536

theorem divisibility_implies_one (n : ℕ+) (h : n ∣ 2^n.val - 1) : n = 1 := by
  sorry

end divisibility_implies_one_l2515_251536


namespace correct_num_tables_l2515_251560

/-- The number of tables in the lunchroom -/
def num_tables : ℕ := 6

/-- The initial number of students per table -/
def initial_students_per_table : ℚ := 6

/-- The desired number of students per table -/
def desired_students_per_table : ℚ := 17 / 3

/-- Theorem stating that the number of tables is correct -/
theorem correct_num_tables :
  (initial_students_per_table * num_tables : ℚ) =
  (desired_students_per_table * num_tables : ℚ) :=
by sorry

end correct_num_tables_l2515_251560


namespace least_three_digit_7_heavy_l2515_251513

def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_7_heavy : 
  (∀ m : ℕ, is_three_digit m → is_7_heavy m → 104 ≤ m) ∧ 
  is_three_digit 104 ∧ 
  is_7_heavy 104 :=
sorry

end least_three_digit_7_heavy_l2515_251513


namespace root_fraction_to_power_l2515_251545

theorem root_fraction_to_power : (81 ^ (1/3)) / (81 ^ (1/4)) = 81 ^ (1/12) := by
  sorry

end root_fraction_to_power_l2515_251545


namespace number_equation_solution_l2515_251589

theorem number_equation_solution : 
  ∃ x : ℝ, (2 * x = 3 * x - 25) ∧ x = 25 := by
  sorry

end number_equation_solution_l2515_251589


namespace HE_in_possible_values_l2515_251551

/-- A quadrilateral with side lengths satisfying certain conditions -/
structure Quadrilateral :=
  (EF : ℝ)
  (FG : ℝ)
  (GH : ℝ)
  (HE : ℤ)
  (ef_eq : EF = 7)
  (fg_eq : FG = 21)
  (gh_eq : GH = 7)

/-- The possible values for HE in the quadrilateral -/
def possible_HE (q : Quadrilateral) : Set ℤ :=
  {n : ℤ | 15 ≤ n ∧ n ≤ 27}

/-- The theorem stating that HE must be in the set of possible values -/
theorem HE_in_possible_values (q : Quadrilateral) : q.HE ∈ possible_HE q := by
  sorry

end HE_in_possible_values_l2515_251551


namespace sqrt_x_plus_sqrt_x_equals_y_infinitely_many_pairs_l2515_251557

theorem sqrt_x_plus_sqrt_x_equals_y (m : ℕ) :
  ∃ (x y : ℚ), (x + Real.sqrt x).sqrt = y ∧
    y = Real.sqrt (m * (m + 1)) ∧
    x = (2 * y^2 + 1 + Real.sqrt (4 * y^2 + 1)) / 2 :=
by sorry

theorem infinitely_many_pairs :
  ∀ n : ℕ, ∃ (S : Finset (ℚ × ℚ)), S.card = n ∧
    ∀ (x y : ℚ), (x, y) ∈ S → (x + Real.sqrt x).sqrt = y :=
by sorry

end sqrt_x_plus_sqrt_x_equals_y_infinitely_many_pairs_l2515_251557


namespace tourist_distribution_l2515_251539

theorem tourist_distribution (total_tourists : ℕ) (h1 : total_tourists = 737) :
  ∃! (num_cars tourists_per_car : ℕ),
    num_cars * tourists_per_car = total_tourists ∧
    num_cars > 0 ∧
    tourists_per_car > 0 :=
by
  sorry

end tourist_distribution_l2515_251539


namespace balloon_difference_balloon_difference_proof_l2515_251593

/-- Proves that the difference between the combined total of Amy, Felix, and Olivia's balloons
    and James' balloons is 373. -/
theorem balloon_difference : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun james_balloons amy_balloons felix_balloons olivia_balloons =>
    james_balloons = 1222 ∧
    amy_balloons = 513 ∧
    felix_balloons = 687 ∧
    olivia_balloons = 395 →
    (amy_balloons + felix_balloons + olivia_balloons) - james_balloons = 373

-- The proof is omitted
theorem balloon_difference_proof :
  balloon_difference 1222 513 687 395 := by sorry

end balloon_difference_balloon_difference_proof_l2515_251593


namespace arithmetic_sequence_properties_l2515_251520

/-- An arithmetic sequence with sum of first n terms Sₙ -/
structure ArithmeticSequence where
  S : ℕ → ℝ
  is_arithmetic : ∃ (a₁ d : ℝ), ∀ n : ℕ, S n = n * a₁ + (n * (n - 1) / 2) * d

/-- Properties of a specific arithmetic sequence -/
def SpecificSequence (seq : ArithmeticSequence) : Prop :=
  seq.S 10 = 0 ∧ seq.S 15 = 25

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : SpecificSequence seq) : 
  (∃ a₅ : ℝ, a₅ = -1/3 ∧ ∀ n : ℕ, seq.S n = n * a₅ + (n * (n - 1) / 2) * (2/3)) ∧ 
  (∀ n : ℕ, seq.S n ≥ seq.S 5) ∧
  (∃ min_value : ℝ, min_value = -49 ∧ ∀ n : ℕ, n * seq.S n ≥ min_value) ∧
  (¬∃ max_value : ℝ, ∀ n : ℕ, seq.S n / n ≤ max_value) := by
  sorry

end arithmetic_sequence_properties_l2515_251520


namespace marcel_potatoes_l2515_251532

/-- Given the conditions of Marcel and Dale's grocery shopping, prove that Marcel bought 4 potatoes. -/
theorem marcel_potatoes :
  ∀ (marcel_corn dale_corn marcel_potatoes dale_potatoes total_vegetables : ℕ),
  marcel_corn = 10 →
  dale_corn = marcel_corn / 2 →
  dale_potatoes = 8 →
  total_vegetables = 27 →
  total_vegetables = marcel_corn + dale_corn + marcel_potatoes + dale_potatoes →
  marcel_potatoes = 4 :=
by
  sorry


end marcel_potatoes_l2515_251532


namespace range_of_u_l2515_251529

theorem range_of_u (x y : ℝ) (h : x^2 / 3 + y^2 = 1) :
  let u := |2*x + y - 4| + |3 - x - 2*y|
  1 ≤ u ∧ u ≤ 13 :=
sorry

end range_of_u_l2515_251529


namespace equal_area_rectangles_l2515_251542

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem equal_area_rectangles (r1 r2 : Rectangle) 
  (h1 : r1.length = 4)
  (h2 : r1.width = 30)
  (h3 : r2.width = 15)
  (h4 : area r1 = area r2) :
  r2.length = 8 := by
  sorry

end equal_area_rectangles_l2515_251542


namespace shortest_distance_specific_rectangle_l2515_251564

/-- A rectangle on a cube face with given dimensions -/
structure RectangleOnCube where
  pq : ℝ
  qr : ℝ
  is_vertex_q : Bool
  is_vertex_s : Bool
  on_adjacent_faces : Bool

/-- The shortest distance between two points through a cube -/
def shortest_distance_through_cube (r : RectangleOnCube) : ℝ :=
  sorry

/-- Theorem stating the shortest distance for the given rectangle -/
theorem shortest_distance_specific_rectangle :
  let r : RectangleOnCube := {
    pq := 20,
    qr := 15,
    is_vertex_q := true,
    is_vertex_s := true,
    on_adjacent_faces := true
  }
  shortest_distance_through_cube r = Real.sqrt 337 := by
  sorry

end shortest_distance_specific_rectangle_l2515_251564


namespace largest_number_less_than_150_divisible_by_3_l2515_251596

theorem largest_number_less_than_150_divisible_by_3 :
  ∃ (x : ℕ), x = 12 ∧
  (∀ (y : ℕ), 11 * y < 150 ∧ 3 ∣ y → y ≤ x) ∧
  11 * x < 150 ∧ 3 ∣ x :=
by sorry

end largest_number_less_than_150_divisible_by_3_l2515_251596


namespace spade_calculation_l2515_251570

/-- Define the ⊙ operation for real numbers -/
def spade (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- Theorem stating that 7 ⊙ (2 ⊙ 3) = 24 -/
theorem spade_calculation : spade 7 (spade 2 3) = 24 := by
  sorry

end spade_calculation_l2515_251570


namespace heart_ratio_theorem_l2515_251507

def heart (n m : ℕ) : ℕ := n^3 + m^2

theorem heart_ratio_theorem : (heart 3 5 : ℚ) / (heart 5 3 : ℚ) = 26 / 67 := by
  sorry

end heart_ratio_theorem_l2515_251507


namespace dulce_has_three_points_l2515_251587

-- Define the points for each person and the team
def max_points : ℕ := 5
def dulce_points : ℕ := 3  -- This is what we want to prove
def val_points (d : ℕ) : ℕ := 2 * (max_points + d)
def team_total (d : ℕ) : ℕ := max_points + d + val_points d

-- Define the opponent's points and the point difference
def opponent_points : ℕ := 40
def point_difference : ℕ := 16

-- Theorem to prove
theorem dulce_has_three_points : 
  team_total dulce_points = opponent_points - point_difference := by
  sorry


end dulce_has_three_points_l2515_251587


namespace festival_lineup_solution_valid_l2515_251558

/-- Represents the minimum number of Gennadys required for the festival lineup -/
def min_gennadys (alexanders borises vasilies : ℕ) : ℕ :=
  max 0 (borises - 1 - alexanders - vasilies)

/-- Theorem stating the minimum number of Gennadys required for the given problem -/
theorem festival_lineup (alexanders borises vasilies : ℕ) 
  (h_alex : alexanders = 45)
  (h_boris : borises = 122)
  (h_vasily : vasilies = 27) :
  min_gennadys alexanders borises vasilies = 49 := by
  sorry

/-- Verifies that the solution satisfies the problem constraints -/
theorem solution_valid (alexanders borises vasilies gennadys : ℕ)
  (h_alex : alexanders = 45)
  (h_boris : borises = 122)
  (h_vasily : vasilies = 27)
  (h_gennady : gennadys = min_gennadys alexanders borises vasilies) :
  alexanders + borises + vasilies + gennadys ≥ borises + (borises - 1) := by
  sorry

end festival_lineup_solution_valid_l2515_251558


namespace quadratic_roots_sum_product_l2515_251546

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (m^2 + 4*m - 1 = 0) → 
  (n^2 + 4*n - 1 = 0) → 
  m + n + m*n = -5 := by
sorry

end quadratic_roots_sum_product_l2515_251546


namespace joeys_swimming_time_l2515_251510

theorem joeys_swimming_time (ethan_time : ℝ) 
  (h1 : ethan_time > 0)
  (h2 : 3/4 * ethan_time = 9)
  (h3 : ethan_time = 12) :
  1/2 * ethan_time = 6 := by
  sorry

end joeys_swimming_time_l2515_251510


namespace sum_of_smaller_radii_eq_twice_original_radius_l2515_251526

/-- Represents a tetrahedron with an insphere and four smaller tetrahedrons -/
structure Tetrahedron where
  r : ℝ  -- radius of the insphere of the original tetrahedron
  r₁ : ℝ  -- radius of the insphere of the first smaller tetrahedron
  r₂ : ℝ  -- radius of the insphere of the second smaller tetrahedron
  r₃ : ℝ  -- radius of the insphere of the third smaller tetrahedron
  r₄ : ℝ  -- radius of the insphere of the fourth smaller tetrahedron

/-- The sum of the radii of the inspheres of the four smaller tetrahedrons is equal to twice the radius of the insphere of the original tetrahedron -/
theorem sum_of_smaller_radii_eq_twice_original_radius (t : Tetrahedron) :
  t.r₁ + t.r₂ + t.r₃ + t.r₄ = 2 * t.r := by
  sorry

end sum_of_smaller_radii_eq_twice_original_radius_l2515_251526


namespace ice_cream_volume_l2515_251528

/-- The volume of ice cream in a cone with a hemispherical top -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1 / 3) * π * r^2 * h
  let hemisphere_volume := (2 / 3) * π * r^3
  h = 10 ∧ r = 3 → cone_volume + hemisphere_volume = 48 * π := by sorry

end ice_cream_volume_l2515_251528


namespace circle_symmetry_l2515_251580

/-- Given a circle with center (1,2) and radius 1, symmetric about the line y = x + b,
    prove that b = 1 -/
theorem circle_symmetry (b : ℝ) : 
  (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 1 ↔ (y - x = b ∧ (x + y - 3)^2 + (y - x - b)^2 / 4 = 1)) →
  b = 1 := by
sorry

end circle_symmetry_l2515_251580


namespace grisha_remaining_money_l2515_251523

-- Define the given constants
def initial_money : ℕ := 5000
def bunny_price : ℕ := 45
def bag_price : ℕ := 30
def bunnies_per_bag : ℕ := 30

-- Define the function to calculate the remaining money
def remaining_money : ℕ :=
  let full_bag_cost := bag_price + bunnies_per_bag * bunny_price
  let full_bags := initial_money / full_bag_cost
  let money_after_full_bags := initial_money - full_bags * full_bag_cost
  let additional_bag_cost := bag_price
  let money_for_extra_bunnies := money_after_full_bags - additional_bag_cost
  let extra_bunnies := money_for_extra_bunnies / bunny_price
  initial_money - (full_bags * full_bag_cost + additional_bag_cost + extra_bunnies * bunny_price)

-- The theorem to prove
theorem grisha_remaining_money :
  remaining_money = 20 := by sorry

end grisha_remaining_money_l2515_251523


namespace three_numbers_problem_l2515_251547

theorem three_numbers_problem :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  x = 1.4 * y ∧
  x / z = 14 / 11 ∧
  z - y = 0.125 * (x + y) - 40 ∧
  x = 280 ∧ y = 200 ∧ z = 220 := by
sorry

end three_numbers_problem_l2515_251547


namespace triangle_area_l2515_251581

theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 3) (h2 : b = 5) (h3 : Real.cos C = -3/5) :
  let S := (1/2) * a * b * Real.sin C
  S = 6 := by
  sorry

end triangle_area_l2515_251581


namespace dining_bill_calculation_l2515_251505

theorem dining_bill_calculation (total : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) 
  (h_total : total = 184.80)
  (h_tax : tax_rate = 0.10)
  (h_tip : tip_rate = 0.20) : 
  ∃ (food_price : ℝ), 
    food_price * (1 + tax_rate) * (1 + tip_rate) = total ∧ 
    food_price = 140 := by
  sorry

end dining_bill_calculation_l2515_251505


namespace distance_between_vertices_parabolas_distance_l2515_251530

-- Define the equation
def equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + abs (y - 2) = 4

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop :=
  y = -(1/12) * x^2 + 3

def parabola2 (x y : ℝ) : Prop :=
  y = (1/4) * x^2 - 1

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 3)
def vertex2 : ℝ × ℝ := (0, -1)

-- Theorem: The distance between the vertices is 4
theorem distance_between_vertices : 
  Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = 4 := by
  sorry

-- Main theorem
theorem parabolas_distance : ∃ (x1 y1 x2 y2 : ℝ),
  equation x1 y1 ∧ equation x2 y2 ∧
  parabola1 x1 y1 ∧ parabola2 x2 y2 ∧
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 4 := by
  sorry

end distance_between_vertices_parabolas_distance_l2515_251530


namespace no_positive_rational_solution_l2515_251550

theorem no_positive_rational_solution (n : ℕ+) :
  ¬∃ (x y : ℚ), 0 < x ∧ 0 < y ∧ x + y + 1/x + 1/y = 3*n := by
  sorry

end no_positive_rational_solution_l2515_251550


namespace geometric_sequence_implies_geometric_subsequences_exists_non_geometric_sequence_with_geometric_subsequences_l2515_251568

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def odd_subsequence (a : ℕ → ℝ) : ℕ → ℝ :=
  λ k => a (2 * k - 1)

def even_subsequence (a : ℕ → ℝ) : ℕ → ℝ :=
  λ k => a (2 * k)

theorem geometric_sequence_implies_geometric_subsequences
  (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  is_geometric_sequence (odd_subsequence a) ∧
  is_geometric_sequence (even_subsequence a) :=
sorry

theorem exists_non_geometric_sequence_with_geometric_subsequences :
  ∃ a : ℕ → ℝ,
    is_geometric_sequence (odd_subsequence a) ∧
    is_geometric_sequence (even_subsequence a) ∧
    ¬is_geometric_sequence a :=
sorry

end geometric_sequence_implies_geometric_subsequences_exists_non_geometric_sequence_with_geometric_subsequences_l2515_251568


namespace pet_store_theorem_l2515_251533

/-- Given a ratio of cats to dogs to birds and the number of cats, 
    calculate the number of dogs and birds -/
def pet_store_count (cat_ratio dog_ratio bird_ratio num_cats : ℕ) : ℕ × ℕ :=
  let scale_factor := num_cats / cat_ratio
  (dog_ratio * scale_factor, bird_ratio * scale_factor)

/-- Theorem: Given the ratio 2:3:4 for cats:dogs:birds and 20 cats, 
    there are 30 dogs and 40 birds -/
theorem pet_store_theorem : 
  pet_store_count 2 3 4 20 = (30, 40) := by
  sorry

end pet_store_theorem_l2515_251533
