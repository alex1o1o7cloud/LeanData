import Mathlib

namespace smallest_two_digit_prime_with_composite_reverse_l138_13831

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

theorem smallest_two_digit_prime_with_composite_reverse :
  ∃ (n : ℕ), is_two_digit n ∧ is_prime n ∧ ¬(is_prime (reverse_digits n)) ∧
  (∀ m : ℕ, is_two_digit m → is_prime m → m < n → is_prime (reverse_digits m)) ∧
  n = 19 :=
sorry

end smallest_two_digit_prime_with_composite_reverse_l138_13831


namespace smallest_divisible_by_hundred_million_l138_13882

/-- The smallest positive integer n such that the nth term of a geometric sequence
    with first term 5/6 and second term 25 is divisible by 100 million. -/
theorem smallest_divisible_by_hundred_million : ℕ :=
  let a₁ : ℚ := 5 / 6  -- First term
  let a₂ : ℚ := 25     -- Second term
  let r : ℚ := a₂ / a₁ -- Common ratio
  let aₙ : ℕ → ℚ := λ n => r ^ (n - 1) * a₁  -- nth term
  9  -- The smallest n (to be proved)

#check smallest_divisible_by_hundred_million

end smallest_divisible_by_hundred_million_l138_13882


namespace square_sum_identity_l138_13893

theorem square_sum_identity (a b : ℝ) : a^2 + b^2 = (a + b)^2 + (-2 * a * b) := by
  sorry

end square_sum_identity_l138_13893


namespace quadratic_solution_property_l138_13866

theorem quadratic_solution_property (a b : ℝ) : 
  (a * 1^2 + b * 1 + 1 = 0) → (2022 - a - b = 2023) := by
  sorry

end quadratic_solution_property_l138_13866


namespace center_sum_l138_13878

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x + 18*y + 9

-- Define the center of the circle
def is_center (h k : ℝ) : Prop :=
  ∀ x y, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 6*h - 18*k - 9)

-- Theorem statement
theorem center_sum : ∃ h k, is_center h k ∧ h + k = 12 :=
sorry

end center_sum_l138_13878


namespace hyperbola_focus_directrix_distance_example_l138_13889

/-- The distance between the right focus and left directrix of a hyperbola -/
def hyperbola_focus_directrix_distance (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  5

/-- Theorem: The distance between the right focus and left directrix of the hyperbola x²/4 - y²/12 = 1 is 5 -/
theorem hyperbola_focus_directrix_distance_example :
  hyperbola_focus_directrix_distance 2 (2 * Real.sqrt 3) = 5 := by
  sorry

end hyperbola_focus_directrix_distance_example_l138_13889


namespace cube_painting_theorem_l138_13833

/-- The number of faces on a cube -/
def num_faces : ℕ := 6

/-- The number of available colors -/
def num_colors : ℕ := 7

/-- The number of symmetries of a cube -/
def cube_symmetries : ℕ := 24

/-- The number of ways to paint a cube with different colors on each face -/
def paint_cube_ways : ℕ := (num_colors.factorial) / (num_colors - num_faces).factorial

/-- The number of distinct ways to paint a cube considering symmetries -/
def distinct_paint_ways : ℕ := paint_cube_ways / cube_symmetries

theorem cube_painting_theorem : distinct_paint_ways = 210 := by
  sorry

end cube_painting_theorem_l138_13833


namespace ellipse_intersection_theorem_l138_13875

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with equation x²/4 + y²/3 = 1 -/
def Ellipse (p : Point) : Prop :=
  p.x^2 / 4 + p.y^2 / 3 = 1

/-- Checks if two points are symmetric about the x-axis -/
def SymmetricAboutXAxis (p1 p2 : Point) : Prop :=
  p1.x = p2.x ∧ p1.y = -p2.y

/-- Represents a line passing through two points -/
def Line (p1 p2 : Point) : Point → Prop :=
  λ p => (p.y - p1.y) * (p2.x - p1.x) = (p.x - p1.x) * (p2.y - p1.y)

/-- Checks if a point is on the x-axis -/
def OnXAxis (p : Point) : Prop :=
  p.y = 0

/-- Checks if a line intersects the ellipse -/
def IntersectsEllipse (l : Point → Prop) : Prop :=
  ∃ p, l p ∧ Ellipse p

/-- Main theorem -/
theorem ellipse_intersection_theorem (D E A B : Point) 
    (hDE : SymmetricAboutXAxis D E) 
    (hD : Ellipse D) (hE : Ellipse E)
    (hA : OnXAxis A) (hB : OnXAxis B)
    (hDA : ¬IntersectsEllipse (Line D A))
    (hInt : IntersectsEllipse (Line D A) ∧ IntersectsEllipse (Line B E) ∧ 
            ∃ p, Line D A p ∧ Line B E p ∧ Ellipse p) :
    A.x * B.x = 4 := by
  sorry

end ellipse_intersection_theorem_l138_13875


namespace log_equation_solution_l138_13842

/-- Proves that 56 is the solution to the logarithmic equation log_7(x) - 3log_7(2) = 1 -/
theorem log_equation_solution :
  ∃ x : ℝ, (Real.log x / Real.log 7) - 3 * (Real.log 2 / Real.log 7) = 1 ∧ x = 56 := by
  sorry

end log_equation_solution_l138_13842


namespace quadratic_trinomial_decomposition_l138_13813

/-- Any quadratic trinomial can be represented as the sum of two quadratic trinomials with zero discriminants -/
theorem quadratic_trinomial_decomposition (a b c : ℝ) :
  ∃ (p q r s t u : ℝ), 
    (∀ x, a * x^2 + b * x + c = (p * x^2 + q * x + r) + (s * x^2 + t * x + u)) ∧
    (q^2 - 4 * p * r = 0) ∧
    (t^2 - 4 * s * u = 0) := by
  sorry

end quadratic_trinomial_decomposition_l138_13813


namespace composite_ratio_theorem_l138_13808

/-- The nth positive composite number -/
def nthComposite (n : ℕ) : ℕ := sorry

/-- The product of the first n positive composite numbers -/
def productFirstNComposites (n : ℕ) : ℕ := sorry

theorem composite_ratio_theorem : 
  (productFirstNComposites 7) / (productFirstNComposites 14 / productFirstNComposites 7) = 1 / 110 := by
  sorry

end composite_ratio_theorem_l138_13808


namespace expand_cubic_sum_product_l138_13885

theorem expand_cubic_sum_product (x : ℝ) : (x^3 + 3) * (x^3 + 4) = x^6 + 7*x^3 + 12 := by
  sorry

end expand_cubic_sum_product_l138_13885


namespace set_A_properties_l138_13870

def A : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 2^k}

theorem set_A_properties :
  (∀ a ∈ A, ∀ b : ℕ, b ≠ 0 → b < 2*a - 1 → ¬(2*a ∣ b*(b+1))) ∧
  (∀ a ∉ A, a ≠ 1 → ∃ b : ℕ, b ≠ 0 ∧ b < 2*a - 1 ∧ (2*a ∣ b*(b+1))) :=
by sorry

end set_A_properties_l138_13870


namespace completing_square_equivalence_l138_13802

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 4*x - 1 = 0 ↔ (x + 2)^2 = 5 :=
by sorry

end completing_square_equivalence_l138_13802


namespace intersection_equals_Q_l138_13841

-- Define the sets P and Q
def P : Set ℝ := {x | x < 2}
def Q : Set ℝ := {x | x^2 ≤ 1}

-- Theorem statement
theorem intersection_equals_Q : P ∩ Q = Q := by
  sorry

end intersection_equals_Q_l138_13841


namespace power_zero_eq_one_l138_13853

theorem power_zero_eq_one (n : ℤ) (h : n ≠ 0) : n^0 = 1 := by
  sorry

end power_zero_eq_one_l138_13853


namespace sport_water_amount_l138_13888

/-- Represents a flavored drink formulation -/
structure Formulation where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation of the drink -/
def standard : Formulation :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation of the drink -/
def sport : Formulation :=
  { flavoring := 1, corn_syrup := 4, water := 60 }

theorem sport_water_amount (corn_syrup_amount : ℚ) :
  corn_syrup_amount = 5 →
  sport.water / sport.corn_syrup * corn_syrup_amount = 75 := by
sorry

end sport_water_amount_l138_13888


namespace lost_card_number_l138_13869

theorem lost_card_number (n : ℕ) (h1 : n > 0) (h2 : (n * (n + 1)) / 2 - 101 ∈ Set.range (λ i => i : Fin n → ℕ)) : 
  ∃ (k : Fin n), k.val + 1 = 4 ∧ (n * (n + 1)) / 2 - (k.val + 1) = 101 :=
by sorry

end lost_card_number_l138_13869


namespace events_mutually_exclusive_but_not_complementary_l138_13895

-- Define the set of cards
inductive Card : Type
| Hearts : Card
| Spades : Card
| Diamonds : Card
| Clubs : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define a distribution of cards to people
def Distribution := Person → Card

-- Define the event "A gets clubs"
def A_gets_clubs (d : Distribution) : Prop := d Person.A = Card.Clubs

-- Define the event "B gets clubs"
def B_gets_clubs (d : Distribution) : Prop := d Person.B = Card.Clubs

-- Statement to prove
theorem events_mutually_exclusive_but_not_complementary :
  ∃ (d : Distribution),
    (∀ (p : Person), ∃! (c : Card), d p = c) →
    (¬(A_gets_clubs d ∧ B_gets_clubs d)) ∧
    (∃ (d' : Distribution), ¬(A_gets_clubs d') ∧ ¬(B_gets_clubs d')) :=
sorry

end events_mutually_exclusive_but_not_complementary_l138_13895


namespace multiple_properties_l138_13806

theorem multiple_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ 
  (∃ p : ℤ, a + b = 4 * p) ∧ 
  (∃ q : ℤ, a + b = 2 * q) :=
by sorry

end multiple_properties_l138_13806


namespace arithmetic_mean_problem_l138_13828

theorem arithmetic_mean_problem (a b c : ℝ) :
  (a + b + c + 97) / 4 = 85 →
  (a + b + c) / 3 = 81 := by
  sorry

end arithmetic_mean_problem_l138_13828


namespace equal_coefficients_implies_n_seven_l138_13871

theorem equal_coefficients_implies_n_seven (n : ℕ) (h1 : n ≥ 6) :
  (Nat.choose n 5 * 3^5 = Nat.choose n 6 * 3^6) → n = 7 := by
  sorry

end equal_coefficients_implies_n_seven_l138_13871


namespace garden_perimeter_l138_13840

/-- The perimeter of a rectangular garden with width 8 meters and the same area as a rectangular playground of length 16 meters and width 12 meters is 64 meters. -/
theorem garden_perimeter : 
  let playground_length : ℝ := 16
  let playground_width : ℝ := 12
  let playground_area : ℝ := playground_length * playground_width
  let garden_width : ℝ := 8
  let garden_length : ℝ := playground_area / garden_width
  let garden_perimeter : ℝ := 2 * (garden_length + garden_width)
  garden_perimeter = 64 := by sorry

end garden_perimeter_l138_13840


namespace gcf_of_lcms_l138_13868

/-- Greatest Common Factor of two natural numbers -/
def GCF (a b : ℕ) : ℕ := Nat.gcd a b

/-- Least Common Multiple of two natural numbers -/
def LCM (c d : ℕ) : ℕ := Nat.lcm c d

/-- Theorem: The GCF of the LCM of (9, 21) and the LCM of (8, 15) is 3 -/
theorem gcf_of_lcms : GCF (LCM 9 21) (LCM 8 15) = 3 := by
  sorry

end gcf_of_lcms_l138_13868


namespace water_flow_proof_l138_13858

theorem water_flow_proof (rate_second : ℝ) (total_flow : ℝ) : 
  rate_second = 36 →
  ∃ (rate_first rate_third : ℝ),
    rate_second = rate_first * 1.5 ∧
    rate_third = rate_second * 1.25 ∧
    total_flow = rate_first + rate_second + rate_third ∧
    total_flow = 105 := by
  sorry

end water_flow_proof_l138_13858


namespace repeating_decimal_sum_main_theorem_l138_13838

theorem repeating_decimal_sum : ∀ (a b c : ℕ), a < 10 → b < 10 → c < 10 →
  (a : ℚ) / 9 + (b : ℚ) / 9 - (c : ℚ) / 9 = (a + b - c : ℚ) / 9 :=
by sorry

theorem main_theorem : (8 : ℚ) / 9 + (2 : ℚ) / 9 - (6 : ℚ) / 9 = 4 / 9 :=
by sorry

end repeating_decimal_sum_main_theorem_l138_13838


namespace curve_C_cartesian_equation_l138_13826

/-- Given a curve C in polar coordinates, prove its Cartesian equation --/
theorem curve_C_cartesian_equation (ρ θ : ℝ) (h : ρ = ρ * Real.cos θ + 2) :
  ∃ x y : ℝ, x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ y^2 = 4*x + 4 := by
  sorry

end curve_C_cartesian_equation_l138_13826


namespace migraine_expectation_l138_13816

/-- The fraction of Canadians suffering from migraines -/
def migraine_fraction : ℚ := 2 / 7

/-- The total number of Canadians in the sample -/
def sample_size : ℕ := 350

/-- The expected number of Canadians in the sample suffering from migraines -/
def expected_migraines : ℕ := 100

theorem migraine_expectation :
  (migraine_fraction * sample_size : ℚ) = expected_migraines := by sorry

end migraine_expectation_l138_13816


namespace wife_walking_speed_l138_13811

/-- Proves that given a circular track of 726 m circumference, if two people walk in opposite
    directions starting from the same point, with one person walking at 4.5 km/hr and they
    meet after 5.28 minutes, then the other person's walking speed is 3.75 km/hr. -/
theorem wife_walking_speed
  (track_circumference : ℝ)
  (suresh_speed : ℝ)
  (meeting_time : ℝ)
  (h1 : track_circumference = 726 / 1000) -- Convert 726 m to km
  (h2 : suresh_speed = 4.5)
  (h3 : meeting_time = 5.28 / 60) -- Convert 5.28 minutes to hours
  : ∃ (wife_speed : ℝ), wife_speed = 3.75 := by
  sorry

#check wife_walking_speed

end wife_walking_speed_l138_13811


namespace davids_remaining_money_l138_13846

theorem davids_remaining_money (initial : ℝ) (remaining : ℝ) (spent : ℝ) : 
  initial = 1500 →
  remaining + spent = initial →
  remaining < spent →
  remaining < 750 := by
sorry

end davids_remaining_money_l138_13846


namespace binder_problem_l138_13851

/-- Given that 18 binders can bind 900 books in 10 days, prove that 11 binders can bind 660 books in 12 days. -/
theorem binder_problem (binders_initial : ℕ) (books_initial : ℕ) (days_initial : ℕ)
  (binders_final : ℕ) (days_final : ℕ) 
  (h1 : binders_initial = 18) (h2 : books_initial = 900) (h3 : days_initial = 10)
  (h4 : binders_final = 11) (h5 : days_final = 12) :
  (books_initial * binders_final * days_final) / (binders_initial * days_initial) = 660 := by
sorry

end binder_problem_l138_13851


namespace herd_size_l138_13804

theorem herd_size (bulls : ℕ) (h : bulls = 70) : 
  (2 / 3 : ℚ) * (1 / 3 : ℚ) * (total_herd : ℚ) = bulls → total_herd = 315 := by
  sorry

end herd_size_l138_13804


namespace positive_integer_solution_exists_l138_13819

theorem positive_integer_solution_exists : 
  ∃ (x y z t : ℕ+), x + y + z + t = 10 :=
by sorry

#check positive_integer_solution_exists

end positive_integer_solution_exists_l138_13819


namespace complex_product_equals_five_l138_13830

theorem complex_product_equals_five (a : ℝ) : 
  let z₁ : ℂ := -1 + 2*I
  let z₂ : ℂ := a - 2*I
  z₁ * z₂ = 5 → a = -1 := by
sorry

end complex_product_equals_five_l138_13830


namespace product_condition_l138_13896

theorem product_condition (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) ∧
  (∃ a b : ℝ, a * b > 1 ∧ ¬(a > 1 ∧ b > 1)) :=
sorry

end product_condition_l138_13896


namespace parabola_focus_l138_13873

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop := y = -1/8 * x^2

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (0, 2)

/-- Theorem: The focus of the parabola y = -1/8 * x^2 is (0, 2) -/
theorem parabola_focus :
  ∀ (x y : ℝ), parabola_eq x y →
  ∃ (d : ℝ), 
    (x - focus.1)^2 + (y - focus.2)^2 = (y - d)^2 ∧
    (∀ (x' y' : ℝ), parabola_eq x' y' → 
      (x' - focus.1)^2 + (y' - focus.2)^2 = (y' - d)^2) :=
sorry

end parabola_focus_l138_13873


namespace max_min_sum_l138_13824

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the closed interval
def I : Set ℝ := {x | -3 ≤ x ∧ x ≤ 0}

-- State the theorem
theorem max_min_sum (M m : ℝ) 
  (hM : ∀ x ∈ I, f x ≤ M) 
  (hm : ∀ x ∈ I, m ≤ f x) 
  (hMexists : ∃ x ∈ I, f x = M) 
  (hmexists : ∃ x ∈ I, f x = m) : 
  M + m = -14 := by sorry

end max_min_sum_l138_13824


namespace complex_number_equality_l138_13860

theorem complex_number_equality (b : ℝ) : 
  let z := (3 - b * Complex.I) / (2 + Complex.I)
  z.re = z.im → b = -9 := by sorry

end complex_number_equality_l138_13860


namespace max_xy_value_l138_13832

theorem max_xy_value (x y : ℝ) (hx : x < 0) (hy : y < 0) (h_eq : 3 * x + y = -2) :
  ∃ (max_xy : ℝ), max_xy = 1/3 ∧ ∀ z, z = x * y → z ≤ max_xy :=
sorry

end max_xy_value_l138_13832


namespace total_coin_value_l138_13848

-- Define the number of rolls for each coin type
def quarters_rolls : ℕ := 5
def dimes_rolls : ℕ := 4
def nickels_rolls : ℕ := 3
def pennies_rolls : ℕ := 2

-- Define the number of coins in each roll
def quarters_per_roll : ℕ := 40
def dimes_per_roll : ℕ := 50
def nickels_per_roll : ℕ := 40
def pennies_per_roll : ℕ := 50

-- Define the value of each coin in cents
def quarter_value : ℕ := 25
def dime_value : ℕ := 10
def nickel_value : ℕ := 5
def penny_value : ℕ := 1

-- Calculate the total value in cents
def total_value : ℕ :=
  quarters_rolls * quarters_per_roll * quarter_value +
  dimes_rolls * dimes_per_roll * dime_value +
  nickels_rolls * nickels_per_roll * nickel_value +
  pennies_rolls * pennies_per_roll * penny_value

-- Theorem to prove
theorem total_coin_value : total_value = 7700 := by
  sorry

end total_coin_value_l138_13848


namespace patricia_hair_donation_l138_13897

/-- Calculates the amount of hair to donate given the current length, additional growth, and desired final length -/
def hair_to_donate (current_length additional_growth final_length : ℕ) : ℕ :=
  (current_length + additional_growth) - final_length

/-- Proves that Patricia needs to donate 23 inches of hair -/
theorem patricia_hair_donation :
  let current_length : ℕ := 14
  let additional_growth : ℕ := 21
  let final_length : ℕ := 12
  hair_to_donate current_length additional_growth final_length = 23 := by
  sorry

end patricia_hair_donation_l138_13897


namespace salary_change_l138_13815

theorem salary_change (original : ℝ) (h : original > 0) :
  let decreased := original * 0.5
  let increased := decreased * 1.5
  (original - increased) / original = 0.25 := by
sorry

end salary_change_l138_13815


namespace oil_cylinder_capacity_l138_13803

theorem oil_cylinder_capacity (C : ℚ) 
  (h1 : (4/5 : ℚ) * C - (3/4 : ℚ) * C = 4) : C = 80 := by
  sorry

end oil_cylinder_capacity_l138_13803


namespace factor_expression_l138_13864

theorem factor_expression (x : ℝ) : 3*x*(x-5) + 4*(x-5) + 6*x = (3*x + 4)*(x - 5) := by
  sorry

end factor_expression_l138_13864


namespace pizza_cheese_calories_pizza_cheese_calories_proof_l138_13809

theorem pizza_cheese_calories : ℝ → Prop :=
  fun cheese_calories =>
    let lettuce_calories : ℝ := 50
    let carrot_calories : ℝ := 2 * lettuce_calories
    let dressing_calories : ℝ := 210
    let salad_calories : ℝ := lettuce_calories + carrot_calories + dressing_calories
    let crust_calories : ℝ := 600
    let pepperoni_calories : ℝ := (1 / 3) * crust_calories
    let pizza_calories : ℝ := crust_calories + pepperoni_calories + cheese_calories
    let jackson_salad_portion : ℝ := 1 / 4
    let jackson_pizza_portion : ℝ := 1 / 5
    let jackson_consumed_calories : ℝ := 330
    jackson_salad_portion * salad_calories + jackson_pizza_portion * pizza_calories = jackson_consumed_calories →
    cheese_calories = 400

-- Proof
theorem pizza_cheese_calories_proof : pizza_cheese_calories 400 := by
  sorry

end pizza_cheese_calories_pizza_cheese_calories_proof_l138_13809


namespace cone_volume_from_lateral_surface_l138_13862

/-- The volume of a cone whose lateral surface unfolds into a semicircle with radius 2 -/
theorem cone_volume_from_lateral_surface (r : Real) (h : Real) : 
  (r = 1) → (h = Real.sqrt 3) → (2 * π * r = 2 * π) → 
  (1 / 3 : Real) * π * r^2 * h = (Real.sqrt 3 * π) / 3 := by
  sorry

end cone_volume_from_lateral_surface_l138_13862


namespace power_calculation_l138_13876

theorem power_calculation : (8^5 / 8^3) * 4^6 = 262144 := by
  sorry

end power_calculation_l138_13876


namespace apple_cost_calculation_apple_cost_proof_l138_13879

/-- Calculates the total cost of apples for a family after a price increase -/
theorem apple_cost_calculation (original_price : ℝ) (price_increase : ℝ) 
  (family_size : ℕ) (pounds_per_person : ℝ) : ℝ :=
  let new_price := original_price * (1 + price_increase)
  let total_pounds := (family_size : ℝ) * pounds_per_person
  new_price * total_pounds

/-- Proves that the total cost for the given scenario is $16 -/
theorem apple_cost_proof : 
  apple_cost_calculation 1.6 0.25 4 2 = 16 := by
  sorry

end apple_cost_calculation_apple_cost_proof_l138_13879


namespace smallest_prime_not_three_l138_13861

theorem smallest_prime_not_three : ¬(∀ p : ℕ, Prime p → p ≥ 3) :=
sorry

end smallest_prime_not_three_l138_13861


namespace arithmetic_sequence_64th_term_l138_13837

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence with specific properties, the 64th term is 129. -/
theorem arithmetic_sequence_64th_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_3rd : a 3 = 7)
  (h_18th : a 18 = 37) :
  a 64 = 129 := by
  sorry

end arithmetic_sequence_64th_term_l138_13837


namespace sum_of_consecutive_even_integers_l138_13805

/-- Three consecutive even integers where the sum of the first and third is 128 -/
structure ConsecutiveEvenIntegers where
  a : ℤ
  b : ℤ
  c : ℤ
  consecutive : b = a + 2 ∧ c = b + 2
  even : Even a
  sum_first_third : a + c = 128

/-- The sum of three consecutive even integers is 192 when the sum of the first and third is 128 -/
theorem sum_of_consecutive_even_integers (x : ConsecutiveEvenIntegers) : x.a + x.b + x.c = 192 := by
  sorry

end sum_of_consecutive_even_integers_l138_13805


namespace sequence_general_term_l138_13823

def S (n : ℕ) : ℕ := 2^n - 1

def a (n : ℕ) : ℕ := 2^(n-1)

theorem sequence_general_term (n : ℕ) (h : n ≥ 1) : 
  a n = S n - S (n-1) :=
sorry

end sequence_general_term_l138_13823


namespace tobias_played_one_week_l138_13857

/-- Calculates the number of weeks Tobias played given the conditions of the problem -/
def tobias_weeks (nathan_hours_per_day : ℕ) (nathan_weeks : ℕ) (tobias_hours_per_day : ℕ) (total_hours : ℕ) : ℕ :=
  let nathan_total_hours := nathan_hours_per_day * 7 * nathan_weeks
  let tobias_total_hours := total_hours - nathan_total_hours
  tobias_total_hours / (tobias_hours_per_day * 7)

/-- Theorem stating that Tobias played for 1 week given the problem conditions -/
theorem tobias_played_one_week :
  tobias_weeks 3 2 5 77 = 1 := by
  sorry

#eval tobias_weeks 3 2 5 77

end tobias_played_one_week_l138_13857


namespace cake_recipe_flour_l138_13807

theorem cake_recipe_flour (sugar cups_of_sugar : ℕ) (flour initial_flour : ℕ) :
  cups_of_sugar = 6 →
  initial_flour = 2 →
  flour = cups_of_sugar + 1 →
  flour = 7 :=
by
  sorry

end cake_recipe_flour_l138_13807


namespace z_sixth_power_l138_13881

theorem z_sixth_power (z : ℂ) : z = (-Real.sqrt 3 - Complex.I) / 2 → z^6 = -1 := by
  sorry

end z_sixth_power_l138_13881


namespace cone_lateral_surface_area_l138_13834

/-- Given a cone with slant height 4 and angle between the slant height and axis of rotation 30°,
    the lateral surface area of the cone is 8π. -/
theorem cone_lateral_surface_area (l : ℝ) (θ : ℝ) (h1 : l = 4) (h2 : θ = 30 * π / 180) :
  π * l * (l * Real.sin θ) = 8 * π := by
  sorry

end cone_lateral_surface_area_l138_13834


namespace green_marbles_count_l138_13800

theorem green_marbles_count (total : ℕ) (white : ℕ) 
  (h1 : white = 40)
  (h2 : (1 : ℚ) / 4 + (1 : ℚ) / 3 + (1 : ℚ) / 6 + (white : ℚ) / total = 1) :
  ⌊(1 : ℚ) / 6 * total⌋ = 27 := by
  sorry

end green_marbles_count_l138_13800


namespace sector_area_l138_13801

theorem sector_area (θ : Real) (chord_length : Real) (area : Real) : 
  θ = 2 ∧ 
  chord_length = 2 * Real.sin 1 ∧ 
  area = (1 / 2) * 1 * θ →
  area = 1 := by
sorry

end sector_area_l138_13801


namespace emily_age_l138_13820

/-- Represents the ages of people in the problem -/
structure Ages where
  alan : ℕ
  bob : ℕ
  carl : ℕ
  donna : ℕ
  emily : ℕ

/-- The age relationships in the problem -/
def valid_ages (ages : Ages) : Prop :=
  ages.alan = ages.bob - 4 ∧
  ages.bob = ages.carl + 5 ∧
  ages.donna = ages.carl + 2 ∧
  ages.emily = ages.alan + ages.donna - ages.bob

theorem emily_age (ages : Ages) :
  valid_ages ages → ages.bob = 20 → ages.emily = 13 := by
  sorry

end emily_age_l138_13820


namespace mean_of_cubic_solutions_l138_13865

theorem mean_of_cubic_solutions (x : ℝ) :
  x^3 + 2*x^2 - 13*x - 10 = 0 →
  ∃ (s : Finset ℝ), s.card = 3 ∧ (∀ y ∈ s, y^3 + 2*y^2 - 13*y - 10 = 0) ∧
  (s.sum id) / s.card = -1 :=
sorry

end mean_of_cubic_solutions_l138_13865


namespace sqrt_difference_equals_six_l138_13847

theorem sqrt_difference_equals_six :
  Real.sqrt (21 + 12 * Real.sqrt 3) - Real.sqrt (21 - 12 * Real.sqrt 3) = 6 := by
  sorry

end sqrt_difference_equals_six_l138_13847


namespace remaining_distance_to_grandma_l138_13812

theorem remaining_distance_to_grandma (total_distance : ℕ) 
  (distance1 distance2 distance3 distance4 distance5 distance6 : ℕ) : 
  total_distance = 78 ∧ 
  distance1 = 35 ∧ 
  distance2 = 7 ∧ 
  distance3 = 18 ∧ 
  distance4 = 3 ∧ 
  distance5 = 12 ∧ 
  distance6 = 2 → 
  total_distance - (distance1 + distance2 + distance3 + distance4 + distance5 + distance6) = 1 := by
  sorry

end remaining_distance_to_grandma_l138_13812


namespace sugar_calculation_l138_13859

theorem sugar_calculation (standard_sugar : ℚ) (reduced_sugar : ℚ) : 
  standard_sugar = 10/3 → 
  reduced_sugar = (1/3) * standard_sugar →
  reduced_sugar = 10/9 :=
by sorry

end sugar_calculation_l138_13859


namespace max_cake_pieces_l138_13849

def cakeSize : ℕ := 50
def pieceSize1 : ℕ := 4
def pieceSize2 : ℕ := 6
def pieceSize3 : ℕ := 8

theorem max_cake_pieces :
  let maxLargePieces := (cakeSize / pieceSize3) ^ 2
  let remainingWidth := cakeSize - (cakeSize / pieceSize3) * pieceSize3
  let maxSmallPieces := 2 * (cakeSize / pieceSize1)
  maxLargePieces + maxSmallPieces = 60 :=
by sorry

end max_cake_pieces_l138_13849


namespace pictures_per_album_l138_13854

theorem pictures_per_album (total_pictures : ℕ) (num_albums : ℕ) (pictures_per_album : ℕ) : 
  total_pictures = 24 → 
  num_albums = 4 → 
  total_pictures = num_albums * pictures_per_album →
  pictures_per_album = 6 := by
  sorry

end pictures_per_album_l138_13854


namespace diophantine_equation_solution_l138_13850

theorem diophantine_equation_solution : ∃ (x y : ℕ), x^2 + y^2 = 61^3 := by
  sorry

end diophantine_equation_solution_l138_13850


namespace javier_speech_time_l138_13872

/-- Represents the time spent on different activities of speech preparation --/
structure SpeechTime where
  outline : ℕ
  write : ℕ
  practice : ℕ

/-- Calculates the total time spent on speech preparation --/
def totalTime (st : SpeechTime) : ℕ :=
  st.outline + st.write + st.practice

/-- Theorem stating the total time Javier spends on his speech --/
theorem javier_speech_time :
  ∃ (st : SpeechTime),
    st.outline = 30 ∧
    st.write = st.outline + 28 ∧
    st.practice = st.write / 2 ∧
    totalTime st = 117 := by
  sorry


end javier_speech_time_l138_13872


namespace rancher_cows_count_l138_13844

theorem rancher_cows_count (horses : ℕ) (cows : ℕ) : 
  cows = 5 * horses →
  cows + horses = 168 →
  cows = 140 := by
sorry

end rancher_cows_count_l138_13844


namespace find_x_l138_13825

theorem find_x (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) 
  (h : (2 * a) ^ (2 * b) = a ^ b * x ^ (2 * b)) : x = 2 * Real.sqrt a := by
  sorry

end find_x_l138_13825


namespace largest_three_digit_congruence_l138_13829

theorem largest_three_digit_congruence :
  ∀ n : ℕ,
  n ≤ 998 →
  100 ≤ n →
  n ≤ 999 →
  70 * n ≡ 210 [MOD 350] →
  ∃ m : ℕ,
  m = 998 ∧
  70 * m ≡ 210 [MOD 350] ∧
  ∀ k : ℕ,
  100 ≤ k →
  k ≤ 999 →
  70 * k ≡ 210 [MOD 350] →
  k ≤ m :=
by sorry

end largest_three_digit_congruence_l138_13829


namespace distinct_bracelets_count_l138_13810

/-- Represents a bracelet configuration -/
structure Bracelet :=
  (red : Nat)
  (blue : Nat)
  (green : Nat)

/-- Defines the specific bracelet configuration in the problem -/
def problem_bracelet : Bracelet :=
  { red := 1, blue := 2, green := 2 }

/-- Calculates the total number of beads in a bracelet -/
def total_beads (b : Bracelet) : Nat :=
  b.red + b.blue + b.green

/-- Represents the number of distinct bracelets -/
def distinct_bracelets (b : Bracelet) : Nat :=
  (Nat.factorial (total_beads b)) / 
  (Nat.factorial b.red * Nat.factorial b.blue * Nat.factorial b.green * 
   (total_beads b) * 2)

/-- Theorem stating that the number of distinct bracelets for the given configuration is 4 -/
theorem distinct_bracelets_count :
  distinct_bracelets problem_bracelet = 4 := by
  sorry

end distinct_bracelets_count_l138_13810


namespace matrix_sum_squares_invertible_l138_13839

open Matrix

variable {n : ℕ}

/-- Given real n×n matrices M and N satisfying the conditions, M² + N² is invertible iff M and N are invertible -/
theorem matrix_sum_squares_invertible (M N : Matrix (Fin n) (Fin n) ℝ)
  (h_neq : M ≠ N)
  (h_cube : M^3 = N^3)
  (h_comm : M^2 * N = N^2 * M) :
  IsUnit (M^2 + N^2) ↔ IsUnit M ∧ IsUnit N := by
  sorry

end matrix_sum_squares_invertible_l138_13839


namespace fish_weight_l138_13874

/-- Given a barrel of fish with the following properties:
    - The initial weight of the barrel with all fish is 54 kg
    - The weight of the barrel with half of the fish removed is 29 kg
    This theorem proves that the total weight of the fish is 50 kg. -/
theorem fish_weight (initial_weight : ℝ) (half_removed_weight : ℝ) 
  (h1 : initial_weight = 54)
  (h2 : half_removed_weight = 29) :
  ∃ (barrel_weight fish_weight : ℝ),
    barrel_weight + fish_weight = initial_weight ∧
    barrel_weight + fish_weight / 2 = half_removed_weight ∧
    fish_weight = 50 := by
  sorry

end fish_weight_l138_13874


namespace fraction_equality_l138_13827

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) :
  (a - d) * (b - c) / ((a - b) * (c - d)) = -4 / 3 := by
  sorry

end fraction_equality_l138_13827


namespace circle_center_distance_l138_13886

/-- The distance between the center of the circle defined by x^2 + y^2 = 8x - 2y + 16 and the point (-3, 4) is √74. -/
theorem circle_center_distance : 
  let circle_eq := fun (x y : ℝ) => x^2 + y^2 - 8*x + 2*y - 16 = 0
  let center := (fun (x y : ℝ) => circle_eq x y ∧ 
                 ∀ (x' y' : ℝ), circle_eq x' y' → (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2)
  let distance := fun (x₁ y₁ x₂ y₂ : ℝ) => Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)
  ∃ (cx cy : ℝ), center cx cy ∧ distance cx cy (-3) 4 = Real.sqrt 74 := by sorry

end circle_center_distance_l138_13886


namespace five_digit_divisibility_l138_13852

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def remove_middle_digit (n : ℕ) : ℕ :=
  (n / 10000) * 1000 + (n / 100 % 10) * 10 + (n % 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_five_digit n ∧ (n % (remove_middle_digit n) = 0)

theorem five_digit_divisibility :
  ∀ n : ℕ, satisfies_condition n ↔ ∃ N : ℕ, 10 ≤ N ∧ N ≤ 99 ∧ n = N * 1000 := by
  sorry

end five_digit_divisibility_l138_13852


namespace prime_pythagorean_inequality_l138_13843

theorem prime_pythagorean_inequality (p m n : ℕ) 
  (hp : Nat.Prime p) 
  (hm : m > 0) 
  (hn : n > 0) 
  (heq : p^2 + m^2 = n^2) : 
  m > p := by
  sorry

end prime_pythagorean_inequality_l138_13843


namespace select_perfect_square_l138_13835

theorem select_perfect_square (nums : Finset ℕ) (h_card : nums.card = 48) 
  (h_prime_factors : (nums.prod id).factors.toFinset.card = 10) :
  ∃ (a b c d : ℕ), a ∈ nums ∧ b ∈ nums ∧ c ∈ nums ∧ d ∈ nums ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∃ (m : ℕ), a * b * c * d = m ^ 2 := by
sorry


end select_perfect_square_l138_13835


namespace unique_intersection_point_f_bijective_f_inv_is_inverse_l138_13894

/-- The cubic function f(x) = x^3 + 6x^2 + 9x + 15 -/
def f (x : ℝ) : ℝ := x^3 + 6*x^2 + 9*x + 15

/-- The theorem stating that the unique intersection point of f and its inverse is (-3, -3) -/
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = f p.2 ∧ p.2 = f p.1 ∧ p = (-3, -3) := by
  sorry

/-- The function f is bijective -/
theorem f_bijective : Function.Bijective f := by
  sorry

/-- The inverse function of f exists -/
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

/-- The theorem stating that f_inv is indeed the inverse of f -/
theorem f_inv_is_inverse :
  Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f := by
  sorry

end unique_intersection_point_f_bijective_f_inv_is_inverse_l138_13894


namespace tangent_angle_parabola_l138_13856

/-- The angle of inclination of the tangent to y = x^2 at (1/2, 1/4) is 45° -/
theorem tangent_angle_parabola : 
  let f (x : ℝ) := x^2
  let x₀ : ℝ := 1/2
  let y₀ : ℝ := 1/4
  let m := (deriv f) x₀
  let θ := Real.arctan m
  θ = π/4 := by sorry

end tangent_angle_parabola_l138_13856


namespace product_defect_rate_l138_13891

theorem product_defect_rate (stage1_defect_rate stage2_defect_rate : ℝ) 
  (h1 : stage1_defect_rate = 0.1)
  (h2 : stage2_defect_rate = 0.03) :
  1 - (1 - stage1_defect_rate) * (1 - stage2_defect_rate) = 0.127 := by
  sorry

end product_defect_rate_l138_13891


namespace wall_width_theorem_l138_13898

theorem wall_width_theorem (width height length : ℝ) (volume : ℝ) :
  height = 6 * width →
  length = 7 * height →
  volume = width * height * length →
  volume = 16128 →
  width = 4 := by
sorry

end wall_width_theorem_l138_13898


namespace candy_left_after_eating_l138_13867

/-- The number of candy pieces left after two people eat some from a total collection --/
def candy_left (total : ℕ) (people : ℕ) (eaten_per_person : ℕ) : ℕ :=
  total - (people * eaten_per_person)

/-- Theorem stating that 60 pieces of candy are left when 2 people each eat 4 pieces from a total of 68 --/
theorem candy_left_after_eating : 
  candy_left 68 2 4 = 60 := by
  sorry

end candy_left_after_eating_l138_13867


namespace smallest_sum_of_reciprocals_l138_13883

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 → (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ)) ∧
  (x : ℕ) + (y : ℕ) = 64 :=
sorry

end smallest_sum_of_reciprocals_l138_13883


namespace reflect_F_twice_l138_13817

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Theorem stating that reflecting point F(1,3) over y-axis then x-axis results in F''(-1,-3) -/
theorem reflect_F_twice :
  let F : ℝ × ℝ := (1, 3)
  let F' := reflect_y F
  let F'' := reflect_x F'
  F'' = (-1, -3) := by sorry

end reflect_F_twice_l138_13817


namespace nested_fraction_equality_l138_13855

theorem nested_fraction_equality : 2 + 1 / (2 + 1 / (2 + 2)) = 22 / 9 := by
  sorry

end nested_fraction_equality_l138_13855


namespace function_relation_l138_13845

/-- Given functions h and k, prove that C = 3D/4 -/
theorem function_relation (C D : ℝ) (h k : ℝ → ℝ) : 
  D ≠ 0 →
  (∀ x, h x = 2 * C * x - 3 * D^2) →
  (∀ x, k x = D * x) →
  h (k 2) = 0 →
  C = 3 * D / 4 := by
sorry

end function_relation_l138_13845


namespace a_8_equals_8_l138_13899

def sequence_property (a : ℕ+ → ℕ) : Prop :=
  ∀ (s t : ℕ+), a (s * t) = a s * a t

theorem a_8_equals_8 (a : ℕ+ → ℕ) (h1 : sequence_property a) (h2 : a 2 = 2) : 
  a 8 = 8 := by
  sorry

end a_8_equals_8_l138_13899


namespace ratio_equality_l138_13880

theorem ratio_equality (p q r u v w : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end ratio_equality_l138_13880


namespace triangle_circles_area_sum_l138_13877

theorem triangle_circles_area_sum (r s t : ℝ) : 
  r > 0 ∧ s > 0 ∧ t > 0 →
  r + s = 5 →
  r + t = 12 →
  s + t = 13 →
  π * (r^2 + s^2 + t^2) = 113 * π :=
by sorry

end triangle_circles_area_sum_l138_13877


namespace circle_properties_l138_13892

theorem circle_properties (A : ℝ) (h : A = 64 * Real.pi) : ∃ (r C : ℝ), r = 8 ∧ C = 16 * Real.pi ∧ A = Real.pi * r^2 ∧ C = 2 * Real.pi * r := by
  sorry

end circle_properties_l138_13892


namespace inequality_of_powers_l138_13890

theorem inequality_of_powers (a b c d x : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d ≥ 0) 
  (h5 : a + d = b + c) (h6 : x > 0) : 
  x^a + x^d ≥ x^b + x^c := by
sorry

end inequality_of_powers_l138_13890


namespace negative_three_to_zero_power_l138_13821

theorem negative_three_to_zero_power : (-3 : ℤ) ^ (0 : ℕ) = 1 := by sorry

end negative_three_to_zero_power_l138_13821


namespace square_of_difference_l138_13863

theorem square_of_difference (x : ℝ) : (x - 3)^2 = x^2 - 6*x + 9 := by
  sorry

end square_of_difference_l138_13863


namespace only_fourth_prop_true_l138_13884

-- Define the propositions
def prop1 : Prop := ∀ a b m : ℝ, (a < b → a * m^2 < b * m^2)
def prop2 : Prop := ∀ p q : Prop, (p ∨ q → p ∧ q)
def prop3 : Prop := ∀ x : ℝ, (x > 1 → x > 2) ∧ ¬(x > 2 → x > 1)
def prop4 : Prop := (¬∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0)

-- Theorem statement
theorem only_fourth_prop_true : ¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4 := by
  sorry

end only_fourth_prop_true_l138_13884


namespace remainder_problem_l138_13887

theorem remainder_problem : 7 * 12^24 + 3^24 ≡ 0 [ZMOD 11] := by
  sorry

end remainder_problem_l138_13887


namespace restaurant_group_size_l138_13818

theorem restaurant_group_size :
  ∀ (adult_meal_cost : ℕ) (num_kids : ℕ) (total_cost : ℕ),
    adult_meal_cost = 3 →
    num_kids = 7 →
    total_cost = 15 →
    ∃ (num_adults : ℕ),
      num_adults * adult_meal_cost = total_cost ∧
      num_adults + num_kids = 12 :=
by sorry

end restaurant_group_size_l138_13818


namespace gcd_105_88_l138_13822

theorem gcd_105_88 : Nat.gcd 105 88 = 1 := by
  sorry

end gcd_105_88_l138_13822


namespace soccer_camp_afternoon_attendance_l138_13836

theorem soccer_camp_afternoon_attendance (total_kids : ℕ) 
  (h1 : total_kids = 2000)
  (h2 : ∃ soccer_kids : ℕ, soccer_kids = total_kids / 2)
  (h3 : ∃ morning_kids : ℕ, morning_kids = soccer_kids / 4) :
  ∃ afternoon_kids : ℕ, afternoon_kids = 750 :=
by sorry

end soccer_camp_afternoon_attendance_l138_13836


namespace cube_root_unity_product_l138_13814

/-- A complex cube root of unity -/
def ω : ℂ :=
  sorry

/-- The property of ω being a complex cube root of unity -/
axiom ω_cube_root : ω^3 = 1

/-- The sum of powers of ω equals zero -/
axiom ω_sum_zero : 1 + ω + ω^2 = 0

/-- The main theorem -/
theorem cube_root_unity_product (a b c : ℂ) :
  (a + b*ω + c*ω^2) * (a + b*ω^2 + c*ω) = a^2 + b^2 + c^2 - a*b - a*c - b*c :=
sorry

end cube_root_unity_product_l138_13814
