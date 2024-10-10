import Mathlib

namespace sqrt_19_between_4_and_5_l2127_212761

theorem sqrt_19_between_4_and_5 : 4 < Real.sqrt 19 ∧ Real.sqrt 19 < 5 := by
  sorry

end sqrt_19_between_4_and_5_l2127_212761


namespace polynomial_division_remainder_l2127_212753

/-- The remainder when x^4 + 2x^3 is divided by x^2 + 3x + 2 is x^2 + 2x -/
theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 + 2*x^3 = (x^2 + 3*x + 2) * q + (x^2 + 2*x) := by
  sorry

end polynomial_division_remainder_l2127_212753


namespace isosceles_triangle_base_length_l2127_212733

/-- An isosceles triangle with congruent sides of length 7 cm and perimeter 23 cm has a base of length 9 cm. -/
theorem isosceles_triangle_base_length : 
  ∀ (base congruent_side perimeter : ℝ),
  congruent_side = 7 →
  perimeter = 23 →
  perimeter = 2 * congruent_side + base →
  base = 9 := by
sorry

end isosceles_triangle_base_length_l2127_212733


namespace quadratic_root_difference_l2127_212726

theorem quadratic_root_difference (C : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₁ - x₂ = 5.5 ∧ 2 * x₁^2 + 5 * x₁ = C ∧ 2 * x₂^2 + 5 * x₂ = C) → 
  C = 12 := by
sorry

end quadratic_root_difference_l2127_212726


namespace salary_increase_after_employee_reduction_l2127_212789

theorem salary_increase_after_employee_reduction (E : ℝ) (S : ℝ) (h1 : E > 0) (h2 : S > 0) :
  let new_E := 0.9 * E
  let new_S := (E * S) / new_E
  (new_S - S) / S = 1 / 9 := by sorry

end salary_increase_after_employee_reduction_l2127_212789


namespace eight_digit_number_divisibility_l2127_212730

/-- Represents an eight-digit number in the form 757AB384 -/
def EightDigitNumber (A B : ℕ) : ℕ := 757000000 + A * 10000 + B * 1000 + 384

/-- The number is divisible by 357 -/
def IsDivisibleBy357 (n : ℕ) : Prop := ∃ k : ℕ, n = 357 * k

theorem eight_digit_number_divisibility :
  ∀ A : ℕ, (A < 10) →
    (IsDivisibleBy357 (EightDigitNumber A 5) ∧
     ∀ B : ℕ, B < 10 → B ≠ 5 → ¬IsDivisibleBy357 (EightDigitNumber A B)) :=
by sorry

end eight_digit_number_divisibility_l2127_212730


namespace problem_solution_l2127_212721

-- Define what it means for a number to be a factor of another
def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

-- Define what it means for a number to be a divisor of another
def is_divisor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem problem_solution :
  (is_factor 5 25) ∧
  (is_divisor 19 209 ∧ ¬ is_divisor 19 63) ∧
  (is_factor 9 180) := by
  sorry


end problem_solution_l2127_212721


namespace cos_330_degrees_l2127_212700

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_degrees_l2127_212700


namespace gcd_equality_exists_l2127_212796

theorem gcd_equality_exists : ∃ k : ℕ+, 
  Nat.gcd 2012 2020 = Nat.gcd (2012 + k) 2020 ∧
  Nat.gcd 2012 2020 = Nat.gcd 2012 (2020 + k) ∧
  Nat.gcd 2012 2020 = Nat.gcd (2012 + k) (2020 + k) := by
  sorry

end gcd_equality_exists_l2127_212796


namespace two_thousand_eight_times_two_thousand_six_l2127_212742

theorem two_thousand_eight_times_two_thousand_six (n : ℕ) :
  (2 * 2006 = 1) →
  (∀ n : ℕ, (2*n + 2) * 2006 = 3 * ((2*n) * 2006)) →
  2008 * 2006 = 3^1003 := by
sorry

end two_thousand_eight_times_two_thousand_six_l2127_212742


namespace sweater_markup_l2127_212773

theorem sweater_markup (wholesale_cost : ℝ) (retail_price : ℝ) :
  retail_price > 0 →
  wholesale_cost > 0 →
  (retail_price * 0.4 = wholesale_cost * 1.35) →
  ((retail_price - wholesale_cost) / wholesale_cost) * 100 = 237.5 := by
sorry

end sweater_markup_l2127_212773


namespace john_metal_purchase_cost_l2127_212711

/-- Calculates the total cost of John's metal purchases in USD -/
def total_cost (silver_oz gold_oz platinum_oz palladium_oz : ℝ)
               (silver_price_usd gold_price_multiplier : ℝ)
               (platinum_price_gbp palladium_price_eur : ℝ)
               (usd_gbp_rate1 usd_gbp_rate2 : ℝ)
               (usd_eur_rate1 usd_eur_rate2 : ℝ)
               (silver_gold_discount platinum_tax : ℝ) : ℝ :=
  sorry

theorem john_metal_purchase_cost :
  total_cost 2.5 3.5 4.5 5.5 25 60 80 100 1.3 1.4 1.15 1.2 0.05 0.08 = 6184.815 := by
  sorry

end john_metal_purchase_cost_l2127_212711


namespace polynomial_expansion_equality_l2127_212712

theorem polynomial_expansion_equality (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  (a₀ + a₂ + a₄ + a₆)^2 - (a₁ + a₃ + a₅)^2 = 1 := by
  sorry

end polynomial_expansion_equality_l2127_212712


namespace original_number_is_sixty_l2127_212779

theorem original_number_is_sixty : 
  ∀ x : ℝ, (0.5 * x = 30) → x = 60 := by
sorry

end original_number_is_sixty_l2127_212779


namespace arithmetic_sequence_sum_l2127_212723

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  is_arithmetic_sequence a →
  a 0 = 3 →
  a 1 = 9 →
  a 2 = 15 →
  (∃ n : ℕ, a (n + 2) = 33 ∧ a (n + 1) = y ∧ a n = x) →
  x + y = 48 := by sorry

end arithmetic_sequence_sum_l2127_212723


namespace interval_intersection_l2127_212778

theorem interval_intersection (x : ℝ) : 
  (2 < 3*x ∧ 3*x < 5 ∧ 2 < 4*x ∧ 4*x < 5) ↔ (2/3 < x ∧ x < 5/4) :=
by sorry

end interval_intersection_l2127_212778


namespace quadratic_inequality_minimum_l2127_212705

theorem quadratic_inequality_minimum (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → x^2 + 2*x + a ≥ 0) ↔ a ≥ 1 := by
  sorry

end quadratic_inequality_minimum_l2127_212705


namespace jones_elementary_population_l2127_212794

/-- The total number of students at Jones Elementary School -/
def total_students : ℕ := 150

/-- The number of boys at Jones Elementary School -/
def num_boys : ℕ := (60 * total_students) / 100

/-- Theorem stating that 90 students represent some percentage of the boys,
    and boys make up 60% of the total school population of 150 students -/
theorem jones_elementary_population :
  90 * total_students = 60 * num_boys :=
sorry

end jones_elementary_population_l2127_212794


namespace quadratic_roots_theorem_l2127_212758

theorem quadratic_roots_theorem (r₁ r₂ : ℝ) (p q : ℝ) : 
  (r₁^2 - 5*r₁ + 6 = 0) →
  (r₂^2 - 5*r₂ + 6 = 0) →
  (r₁^2 + p*r₁^2 + q = 0) →
  (r₂^2 + p*r₂^2 + q = 0) →
  p = -13 ∧ q = 36 := by
sorry

end quadratic_roots_theorem_l2127_212758


namespace complement_A_inter_B_l2127_212747

def U : Set Int := {-3, -2, -1, 0, 1, 2, 3}
def A : Set Int := {-3, -2, 2, 3}
def B : Set Int := {-3, 0, 1, 2}

theorem complement_A_inter_B :
  (U \ A) ∩ B = {0, 1} := by sorry

end complement_A_inter_B_l2127_212747


namespace xiaoming_characters_proof_l2127_212764

theorem xiaoming_characters_proof : 
  ∀ (N : ℕ),
  (N / 2 - 50 : ℕ) + -- Day 1
  ((N / 2 + 50) / 2 - 20 : ℕ) + -- Day 2
  (((N / 4 + 45 : ℕ) / 2 + 10) : ℕ) + -- Day 3
  60 + -- Day 4
  40 = N → -- Remaining characters
  N = 700 := by
sorry

end xiaoming_characters_proof_l2127_212764


namespace point_locations_l2127_212792

def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

def is_in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_locations (x y : ℝ) (h : |3*x + 2| + |2*y - 1| = 0) :
  is_in_second_quadrant x y ∧ is_in_fourth_quadrant (x + 1) (y - 2) := by
  sorry

end point_locations_l2127_212792


namespace kho_kho_players_l2127_212706

theorem kho_kho_players (total : ℕ) (kabadi : ℕ) (both : ℕ) (kho_kho : ℕ) : 
  total = 25 → kabadi = 10 → both = 5 → kho_kho = total - kabadi + both := by
  sorry

end kho_kho_players_l2127_212706


namespace paul_frosting_needs_l2127_212739

/-- Represents the amount of frosting needed for different baked goods -/
structure FrostingNeeds where
  layer_cake : ℚ
  single_cake : ℚ
  brownies : ℚ
  cupcakes : ℚ

/-- Calculates the total cans of frosting needed -/
def total_frosting_needed (f : FrostingNeeds) (layer_cakes single_cakes brownies cupcake_dozens : ℕ) : ℚ :=
  f.layer_cake * layer_cakes + 
  f.single_cake * single_cakes + 
  f.brownies * brownies + 
  f.cupcakes * cupcake_dozens

/-- Theorem stating that Paul needs 21 cans of frosting -/
theorem paul_frosting_needs : 
  let f : FrostingNeeds := { 
    layer_cake := 1,
    single_cake := 1/2,
    brownies := 1/2,
    cupcakes := 1/2
  }
  total_frosting_needed f 3 12 18 6 = 21 := by
  sorry

end paul_frosting_needs_l2127_212739


namespace min_set_size_l2127_212772

theorem min_set_size (n : ℕ) 
  (h1 : ∃ (s : Finset ℝ), s.card = 2*n + 1)
  (h2 : ∃ (s1 s2 : Finset ℝ), s1.card = n + 1 ∧ s2.card = n ∧ 
        (∀ x ∈ s1, x ≥ 10) ∧ (∀ x ∈ s2, x ≥ 1))
  (h3 : ∃ (s : Finset ℝ), s.card = 2*n + 1 ∧ 
        (Finset.sum s id) / (2*n + 1 : ℝ) = 6) :
  n ≥ 4 := by
sorry

end min_set_size_l2127_212772


namespace second_point_y_coordinate_l2127_212738

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x = 2 * y + 5

-- Define the two points
def point1 (m n : ℝ) : ℝ × ℝ := (m, n)
def point2 (m n k : ℝ) : ℝ × ℝ := (m + 1, n + k)

-- Theorem statement
theorem second_point_y_coordinate 
  (m n : ℝ) 
  (h1 : line_equation m n) 
  (h2 : line_equation (m + 1) (n + 0.5)) : 
  (point2 m n 0.5).2 = n + 0.5 := by
  sorry

end second_point_y_coordinate_l2127_212738


namespace right_triangle_legs_l2127_212791

theorem right_triangle_legs (a b : ℝ) : 
  a > 0 → b > 0 →
  (a^2 + b^2 = 100) →  -- Pythagorean theorem (hypotenuse = 10)
  (a + b = 14) →       -- Derived from inradius and semiperimeter
  (a = 6 ∧ b = 8) ∨ (a = 8 ∧ b = 6) := by
sorry

end right_triangle_legs_l2127_212791


namespace original_model_cost_l2127_212701

/-- The original cost of a model before the price increase -/
def original_cost : ℝ := sorry

/-- The amount Kirsty saved -/
def saved_amount : ℝ := 30 * original_cost

/-- The new cost of a model after the price increase -/
def new_cost : ℝ := original_cost + 0.50

theorem original_model_cost :
  (saved_amount = 27 * new_cost) → original_cost = 0.45 := by
  sorry

end original_model_cost_l2127_212701


namespace quadratic_opens_upwards_l2127_212729

/-- A quadratic function f(x) = ax² + bx + c -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The quadratic function opens upwards if a > 0 -/
def opens_upwards (a b c : ℝ) : Prop := a > 0

theorem quadratic_opens_upwards (a b c : ℝ) 
  (h1 : f a b c (-1) = 10)
  (h2 : f a b c 0 = 5)
  (h3 : f a b c 1 = 2)
  (h4 : f a b c 2 = 1)
  (h5 : f a b c 3 = 2) :
  opens_upwards a b c := by sorry

end quadratic_opens_upwards_l2127_212729


namespace no_solution_to_inequalities_l2127_212709

theorem no_solution_to_inequalities : ¬∃ x : ℝ, (6*x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 9*x - 5) := by
  sorry

end no_solution_to_inequalities_l2127_212709


namespace max_value_theorem_l2127_212703

theorem max_value_theorem (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 10) :
  ∃ (M : ℝ), M = 100 ∧ ∀ (x y z w : ℝ), x^2 + y^2 + z^2 + w^2 = 10 → x^4 + y^2 + z^2 + w^2 ≤ M :=
sorry

end max_value_theorem_l2127_212703


namespace maria_cookies_left_maria_cookies_problem_l2127_212766

theorem maria_cookies_left (initial_cookies : ℕ) (friend_cookies : ℕ) (eat_cookies : ℕ) : ℕ :=
  let remaining_after_friend := initial_cookies - friend_cookies
  let family_cookies := remaining_after_friend / 2
  let remaining_after_family := remaining_after_friend - family_cookies
  let final_cookies := remaining_after_family - eat_cookies
  final_cookies

theorem maria_cookies_problem :
  maria_cookies_left 19 5 2 = 5 := by
  sorry

end maria_cookies_left_maria_cookies_problem_l2127_212766


namespace cubic_root_sum_cubes_l2127_212786

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (4 * a^3 + 2023 * a + 4012 = 0) ∧ 
  (4 * b^3 + 2023 * b + 4012 = 0) ∧ 
  (4 * c^3 + 2023 * c + 4012 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 3009 := by
sorry

end cubic_root_sum_cubes_l2127_212786


namespace x_plus_ten_equals_forty_l2127_212719

theorem x_plus_ten_equals_forty (x y : ℝ) (h1 : x / y = 6 / 3) (h2 : y = 15) : x + 10 = 40 := by
  sorry

end x_plus_ten_equals_forty_l2127_212719


namespace binomial_coefficient_equality_l2127_212725

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 28 x = Nat.choose 28 (2*x - 1)) → x = 1 := by
  sorry

end binomial_coefficient_equality_l2127_212725


namespace roots_expression_value_l2127_212776

theorem roots_expression_value (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 3 * x₁ + 1 = 0) → 
  (2 * x₂^2 - 3 * x₂ + 1 = 0) → 
  ((x₁ + x₂) / (1 + x₁ * x₂) = 1) :=
by sorry

end roots_expression_value_l2127_212776


namespace max_value_and_inequality_l2127_212780

def f (x m : ℝ) : ℝ := |x - m| - |x + 2*m|

theorem max_value_and_inequality (m : ℝ) (hm : m > 0) 
  (hmax : ∀ x, f x m ≤ 3) :
  m = 1 ∧ 
  ∀ a b : ℝ, a * b > 0 → a^2 + b^2 = m^2 → a^3 / b + b^3 / a ≥ 1 := by
  sorry


end max_value_and_inequality_l2127_212780


namespace quadratic_factorization_l2127_212777

theorem quadratic_factorization (h b c d : ℤ) : 
  (∀ x : ℝ, 6 * x^2 + x - 12 = (h * x + b) * (c * x + d)) → 
  |h| + |b| + |c| + |d| = 12 := by
  sorry

end quadratic_factorization_l2127_212777


namespace minimum_bottles_l2127_212745

def bottle_capacity : ℕ := 15
def minimum_volume : ℕ := 150

theorem minimum_bottles : 
  ∀ n : ℕ, (n * bottle_capacity ≥ minimum_volume ∧ 
  ∀ m : ℕ, m < n → m * bottle_capacity < minimum_volume) → n = 10 :=
by sorry

end minimum_bottles_l2127_212745


namespace ellipse_product_l2127_212774

/-- Represents an ellipse with center O, major axis AB, minor axis CD, and focus F. -/
structure Ellipse where
  a : ℝ  -- Length of semi-major axis
  b : ℝ  -- Length of semi-minor axis
  f : ℝ  -- Distance from center to focus

/-- Conditions for the ellipse problem -/
def EllipseProblem (e : Ellipse) : Prop :=
  e.f = 6 ∧ e.a - e.b = 4 ∧ e.a^2 - e.b^2 = e.f^2

theorem ellipse_product (e : Ellipse) (h : EllipseProblem e) : (2 * e.a) * (2 * e.b) = 65 := by
  sorry

#check ellipse_product

end ellipse_product_l2127_212774


namespace smallest_n_congruence_l2127_212785

theorem smallest_n_congruence : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → (7 ^ m.val) % 3 ≠ (m.val ^ 4) % 3) ∧ 
  (7 ^ n.val) % 3 = (n.val ^ 4) % 3 ∧
  n = 1 := by
sorry

end smallest_n_congruence_l2127_212785


namespace smallest_rectangle_cover_l2127_212783

/-- A point in the unit square -/
def Point := Fin 2 → Real

/-- The unit square -/
def UnitSquare : Set Point :=
  {p | ∀ i, 0 ≤ p i ∧ p i ≤ 1}

/-- The interior of the unit square -/
def InteriorUnitSquare : Set Point :=
  {p | ∀ i, 0 < p i ∧ p i < 1}

/-- A rectangle with sides parallel to the unit square -/
structure Rectangle where
  x1 : Real
  x2 : Real
  y1 : Real
  y2 : Real
  h1 : x1 < x2
  h2 : y1 < y2

/-- A point is in the interior of a rectangle -/
def isInterior (p : Point) (r : Rectangle) : Prop :=
  r.x1 < p 0 ∧ p 0 < r.x2 ∧ r.y1 < p 1 ∧ p 1 < r.y2

/-- The theorem statement -/
theorem smallest_rectangle_cover (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, k = 2 * n + 2 ∧
  (∀ S : Set Point, Finite S → S.ncard = n → S ⊆ InteriorUnitSquare →
    ∃ R : Set Rectangle, R.ncard = k ∧
      (∀ p ∈ S, ∀ r ∈ R, ¬isInterior p r) ∧
      (∀ p ∈ InteriorUnitSquare \ S, ∃ r ∈ R, isInterior p r)) ∧
  (∀ k' : ℕ, k' < k →
    ∃ S : Set Point, Finite S ∧ S.ncard = n ∧ S ⊆ InteriorUnitSquare ∧
      ∀ R : Set Rectangle, R.ncard = k' →
        (∃ p ∈ S, ∃ r ∈ R, isInterior p r) ∨
        (∃ p ∈ InteriorUnitSquare \ S, ∀ r ∈ R, ¬isInterior p r)) :=
by sorry

end smallest_rectangle_cover_l2127_212783


namespace factorial_properties_l2127_212790

-- Define ord_p
def ord_p (p : ℕ) (n : ℕ) : ℕ := sorry

-- Define S_p
def S_p (p : ℕ) (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem factorial_properties (n : ℕ) (p : ℕ) (m : ℕ) 
  (h_prime : Nat.Prime p) 
  (h_div : p ^ m ∣ n ∧ ¬(p ^ (m + 1) ∣ n)) 
  (h_ord : ord_p p n = m) : 
  (ord_p p (n.factorial) = (n - S_p p n) / (p - 1)) ∧ 
  (∃ k : ℕ, (2 * n).factorial = k * n.factorial * (n + 1).factorial) ∧
  (Nat.Coprime m (n + 1) → 
    ∃ k : ℕ, (m * n + n).factorial = k * (m * n).factorial * (n + 1).factorial) := by
  sorry

end factorial_properties_l2127_212790


namespace prob_red_ball_l2127_212768

/-- The probability of drawing a red ball from a bag with 2 red balls and 1 white ball is 2/3 -/
theorem prob_red_ball (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = 3 →
  red_balls = 2 →
  white_balls = 1 →
  red_balls + white_balls = total_balls →
  (red_balls : ℚ) / total_balls = 2 / 3 := by
  sorry

#check prob_red_ball

end prob_red_ball_l2127_212768


namespace initial_mean_calculation_l2127_212732

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (corrected_mean : ℝ) :
  n = 50 ∧ 
  wrong_value = 21 ∧ 
  correct_value = 48 ∧ 
  corrected_mean = 36.54 →
  ∃ initial_mean : ℝ, 
    initial_mean * n + (correct_value - wrong_value) = corrected_mean * n ∧ 
    initial_mean = 36 :=
by sorry

end initial_mean_calculation_l2127_212732


namespace trig_identity_l2127_212770

theorem trig_identity : Real.sin (68 * π / 180) * Real.sin (67 * π / 180) - 
  Real.sin (23 * π / 180) * Real.cos (68 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end trig_identity_l2127_212770


namespace complex_power_2017_l2127_212715

theorem complex_power_2017 : 
  let z : ℂ := (1 - Complex.I) / (1 + Complex.I)
  z^2017 = -Complex.I := by
sorry

end complex_power_2017_l2127_212715


namespace triangle_problem_l2127_212759

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → A < π →
  B > 0 → B < π →
  C > 0 → C < π →
  (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * (b^2 + c^2 - a^2) →
  (1/2) * b * c * Real.sin A = 3/2 →
  (A = π/3) ∧
  ((b*c - 4*Real.sqrt 3) * Real.cos A + a*c * Real.cos B) / (a^2 - b^2) = 1 := by
  sorry

end triangle_problem_l2127_212759


namespace least_candies_to_remove_l2127_212743

theorem least_candies_to_remove (total : Nat) (sisters : Nat) (to_remove : Nat) : 
  total = 24 → 
  sisters = 5 → 
  (total - to_remove) % sisters = 0 → 
  ∀ x : Nat, x < to_remove → (total - x) % sisters ≠ 0 →
  to_remove = 4 := by
  sorry

end least_candies_to_remove_l2127_212743


namespace initial_cost_calculation_l2127_212708

/-- Represents the car rental cost structure and usage --/
structure CarRental where
  initialCost : ℝ
  costPerMile : ℝ
  milesDriven : ℝ
  totalCost : ℝ

/-- Theorem stating the initial cost of the car rental --/
theorem initial_cost_calculation (rental : CarRental) 
    (h1 : rental.costPerMile = 0.50)
    (h2 : rental.milesDriven = 1364)
    (h3 : rental.totalCost = 832) :
    rental.initialCost = 150 := by
  sorry


end initial_cost_calculation_l2127_212708


namespace quadratic_inequality_and_condition_l2127_212750

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x + 1

-- Define the proposition p
def p (a : ℝ) : Prop := ∀ x, f a x > 0

-- Define the proposition q
def q (x : ℝ) : Prop := x^2 - 2*x - 8 > 0

theorem quadratic_inequality_and_condition :
  (∃ a, a ∈ Set.Icc 0 4 ∧ p a) ∧
  (∀ x, q x → x > 5) ∧
  (∃ x, q x ∧ x ≤ 5) :=
sorry

end quadratic_inequality_and_condition_l2127_212750


namespace triangle_angle_inequality_l2127_212746

theorem triangle_angle_inequality (a : ℝ) : 
  (∃ (α β : ℝ), 0 < α ∧ 0 < β ∧ α + β < π ∧ 
    Real.cos (Real.sqrt α) + Real.cos (Real.sqrt β) > a + Real.cos (Real.sqrt (α * β))) 
  → a < 1 := by
  sorry

end triangle_angle_inequality_l2127_212746


namespace tower_surface_area_is_1207_l2127_212755

def cube_volumes : List ℕ := [1, 27, 64, 125, 216, 343, 512, 729]

def cube_side_lengths : List ℕ := [1, 3, 4, 5, 6, 7, 8, 9]

def visible_faces : List ℕ := [6, 4, 4, 4, 4, 4, 4, 5]

def tower_surface_area (volumes : List ℕ) (side_lengths : List ℕ) (faces : List ℕ) : ℕ :=
  (List.zip (List.zip side_lengths faces) volumes).foldr
    (fun ((s, f), v) acc => acc + f * s * s)
    0

theorem tower_surface_area_is_1207 :
  tower_surface_area cube_volumes cube_side_lengths visible_faces = 1207 := by
  sorry

end tower_surface_area_is_1207_l2127_212755


namespace set_inclusion_implies_m_values_l2127_212754

def A (m : ℝ) : Set ℝ := {1, 3, 2*m+3}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem set_inclusion_implies_m_values (m : ℝ) : B m ⊆ A m → m = 1 ∨ m = 3 := by
  sorry

end set_inclusion_implies_m_values_l2127_212754


namespace divides_implies_equal_l2127_212707

theorem divides_implies_equal (a b : ℕ+) : 
  (4 * a * b - 1) ∣ ((4 * a^2 - 1)^2) → a = b := by
  sorry

end divides_implies_equal_l2127_212707


namespace division_problem_l2127_212751

theorem division_problem (n : ℕ) : 
  n / 20 = 10 ∧ n % 20 = 10 → n = 210 := by
  sorry

end division_problem_l2127_212751


namespace min_value_of_sum_of_roots_min_value_achievable_l2127_212781

theorem min_value_of_sum_of_roots (x : ℝ) :
  Real.sqrt (x^2 + (x - 2)^2) + Real.sqrt ((x - 2)^2 + (x + 2)^2) ≥ 2 * Real.sqrt 5 :=
by sorry

theorem min_value_achievable :
  ∃ x : ℝ, Real.sqrt (x^2 + (x - 2)^2) + Real.sqrt ((x - 2)^2 + (x + 2)^2) = 2 * Real.sqrt 5 :=
by sorry

end min_value_of_sum_of_roots_min_value_achievable_l2127_212781


namespace rectangleChoicesIs54_l2127_212702

/-- The number of ways to choose 4 lines (2 horizontal and 2 vertical) from 5 horizontal
    and 4 vertical lines to form a rectangle, without selecting the first and fifth
    horizontal lines together. -/
def rectangleChoices : ℕ := by
  -- Define the number of horizontal and vertical lines
  let horizontalLines : ℕ := 5
  let verticalLines : ℕ := 4

  -- Calculate the total number of ways to choose 2 horizontal lines
  let totalHorizontalChoices : ℕ := Nat.choose horizontalLines 2

  -- Calculate the number of choices that include both first and fifth horizontal lines
  let invalidHorizontalChoices : ℕ := 1

  -- Calculate the number of valid horizontal line choices
  let validHorizontalChoices : ℕ := totalHorizontalChoices - invalidHorizontalChoices

  -- Calculate the number of ways to choose 2 vertical lines
  let verticalChoices : ℕ := Nat.choose verticalLines 2

  -- Calculate the total number of valid choices
  exact validHorizontalChoices * verticalChoices

/-- Theorem stating that the number of valid rectangle choices is 54 -/
theorem rectangleChoicesIs54 : rectangleChoices = 54 := by sorry

end rectangleChoicesIs54_l2127_212702


namespace residue_of_11_pow_2016_mod_19_l2127_212757

theorem residue_of_11_pow_2016_mod_19 : 11^2016 % 19 = 17 := by
  sorry

end residue_of_11_pow_2016_mod_19_l2127_212757


namespace white_balls_count_l2127_212710

/-- Proves that in a bag with 3 red balls and x white balls, 
    if the probability of drawing a red ball is 1/4, then x = 9. -/
theorem white_balls_count (x : ℕ) : 
  (3 : ℚ) / (3 + x) = 1 / 4 → x = 9 := by
  sorry

end white_balls_count_l2127_212710


namespace ten_thousandths_place_of_5_32_l2127_212748

theorem ten_thousandths_place_of_5_32 : ∃ (n : ℕ), (5 : ℚ) / 32 = (n * 10000 + 5) / 100000 :=
by sorry

end ten_thousandths_place_of_5_32_l2127_212748


namespace xy_square_plus_2xy_plus_1_l2127_212749

theorem xy_square_plus_2xy_plus_1 (x y : ℝ) 
  (h : x^2 - 2*x + y^2 - 6*y + 10 = 0) : 
  x^2 * y^2 + 2*x*y + 1 = 16 := by
  sorry

end xy_square_plus_2xy_plus_1_l2127_212749


namespace apple_arrangements_l2127_212765

def word : String := "apple"

/-- The number of distinct letters in the word -/
def distinctLetters : Nat := 4

/-- The total number of letters in the word -/
def totalLetters : Nat := 5

/-- The frequency of the letter 'p' in the word -/
def frequencyP : Nat := 2

/-- The frequency of the letter 'a' in the word -/
def frequencyA : Nat := 1

/-- The frequency of the letter 'l' in the word -/
def frequencyL : Nat := 1

/-- The frequency of the letter 'e' in the word -/
def frequencyE : Nat := 1

/-- The number of distinct arrangements of the letters in the word -/
def distinctArrangements : Nat := 60

theorem apple_arrangements :
  distinctArrangements = Nat.factorial totalLetters / 
    (Nat.factorial frequencyP * Nat.factorial frequencyA * 
     Nat.factorial frequencyL * Nat.factorial frequencyE) := by
  sorry

end apple_arrangements_l2127_212765


namespace protest_jail_time_ratio_l2127_212744

theorem protest_jail_time_ratio : 
  let days_of_protest : ℕ := 30
  let num_cities : ℕ := 21
  let arrests_per_day : ℕ := 10
  let days_before_trial : ℕ := 4
  let sentence_weeks : ℕ := 2
  let total_jail_weeks : ℕ := 9900
  let total_arrests : ℕ := days_of_protest * num_cities * arrests_per_day
  let weeks_before_trial : ℕ := total_arrests * days_before_trial / 7
  let weeks_after_trial : ℕ := total_jail_weeks - weeks_before_trial
  let total_possible_weeks : ℕ := total_arrests * sentence_weeks
  (weeks_after_trial : ℚ) / total_possible_weeks = 1 / 2 := by
  sorry

#check protest_jail_time_ratio

end protest_jail_time_ratio_l2127_212744


namespace snow_clearing_time_l2127_212795

/-- Calculates the number of hours required to clear snow given the total volume,
    initial shoveling capacity, and hourly decrease in capacity. -/
def snow_clearing_hours (total_volume : ℕ) (initial_capacity : ℕ) (hourly_decrease : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

theorem snow_clearing_time :
  snow_clearing_hours 216 25 2 = 21 := by
  sorry

end snow_clearing_time_l2127_212795


namespace regular_polygon_sides_l2127_212797

theorem regular_polygon_sides (n : ℕ) (n_pos : 0 < n) : 
  (∀ θ : ℝ, θ = 156 → (n : ℝ) * θ = 180 * ((n : ℝ) - 2)) → n = 15 := by
  sorry

end regular_polygon_sides_l2127_212797


namespace negative_square_power_equality_l2127_212767

theorem negative_square_power_equality (a b : ℝ) : (-a^2 * b^3)^2 = a^4 * b^6 := by
  sorry

end negative_square_power_equality_l2127_212767


namespace warm_production_time_l2127_212728

/-- Represents the production time for flower pots -/
structure PotProduction where
  cold_time : ℕ  -- Time to produce a pot when machine is cold (in minutes)
  warm_time : ℕ  -- Time to produce a pot when machine is warm (in minutes)
  hour_length : ℕ  -- Length of a production hour (in minutes)
  extra_pots : ℕ  -- Additional pots produced in the last hour compared to the first

/-- Theorem stating the warm production time given the conditions -/
theorem warm_production_time (p : PotProduction) 
  (h1 : p.cold_time = 6)
  (h2 : p.hour_length = 60)
  (h3 : p.extra_pots = 2)
  (h4 : p.hour_length / p.cold_time + p.extra_pots = p.hour_length / p.warm_time) :
  p.warm_time = 5 := by
  sorry

#check warm_production_time

end warm_production_time_l2127_212728


namespace shopping_time_calculation_l2127_212798

-- Define the total shopping trip time in minutes
def total_shopping_time : ℕ := 90

-- Define the waiting times
def wait_for_cart : ℕ := 3
def wait_for_employee : ℕ := 13
def wait_for_restock : ℕ := 14
def wait_in_line : ℕ := 18

-- Define the theorem
theorem shopping_time_calculation :
  total_shopping_time - (wait_for_cart + wait_for_employee + wait_for_restock + wait_in_line) = 42 := by
  sorry

end shopping_time_calculation_l2127_212798


namespace trigonometric_simplification_l2127_212784

theorem trigonometric_simplification (α : ℝ) :
  2 * Real.sin (2 * α) ^ 2 + Real.sqrt 3 * Real.sin (4 * α) -
  (4 * Real.tan (2 * α) * (1 - Real.tan (2 * α) ^ 2)) /
  (Real.sin (8 * α) * (1 + Real.tan (2 * α) ^ 2) ^ 2) =
  2 * Real.sin (4 * α - π / 6) :=
by sorry

end trigonometric_simplification_l2127_212784


namespace wrong_number_correction_l2127_212756

theorem wrong_number_correction (n : ℕ) (initial_avg correct_avg : ℚ) 
  (first_error second_correct : ℤ) : 
  n = 10 → 
  initial_avg = 40.2 → 
  correct_avg = 40.3 → 
  first_error = 19 → 
  second_correct = 31 → 
  ∃ (second_error : ℤ), 
    (n : ℚ) * initial_avg - first_error - second_error + second_correct = (n : ℚ) * correct_avg ∧ 
    second_error = 11 := by
  sorry

end wrong_number_correction_l2127_212756


namespace sandwich_jam_cost_l2127_212782

theorem sandwich_jam_cost (N B J : ℕ) : 
  N > 1 → 
  B > 0 → 
  J > 0 → 
  N * (4 * B + 6 * J) = 351 → 
  N * J * 6 = 162 :=
by sorry

end sandwich_jam_cost_l2127_212782


namespace manolo_face_masks_l2127_212771

/-- Calculates the number of face-masks Manolo makes in a four-hour shift -/
def face_masks_in_shift (first_hour_rate : ℕ) (other_hours_rate : ℕ) (shift_duration : ℕ) : ℕ :=
  let first_hour_masks := 60 / first_hour_rate
  let other_hours_masks := (shift_duration - 1) * 60 / other_hours_rate
  first_hour_masks + other_hours_masks

/-- Theorem stating that Manolo makes 45 face-masks in a four-hour shift -/
theorem manolo_face_masks :
  face_masks_in_shift 4 6 4 = 45 :=
by sorry

end manolo_face_masks_l2127_212771


namespace complex_sum_theorem_l2127_212714

theorem complex_sum_theorem :
  let A : ℂ := 3 + 2*I
  let B : ℂ := -3 + I
  let C : ℂ := 1 - 2*I
  let D : ℂ := 4 + 3*I
  A + B + C + D = 5 + 4*I :=
by sorry

end complex_sum_theorem_l2127_212714


namespace arrangement_count_correct_l2127_212735

/-- The number of ways to arrange 7 people in a row with 2 people between A and B -/
def arrangement_count : ℕ := 960

/-- The total number of people -/
def total_people : ℕ := 7

/-- The number of people between A and B -/
def people_between : ℕ := 2

/-- Theorem stating that the arrangement count is correct -/
theorem arrangement_count_correct : 
  arrangement_count = 
    (Nat.factorial 2) *  -- Ways to arrange A and B
    (Nat.choose (total_people - 2) people_between) *  -- Ways to choose people between A and B
    (Nat.factorial people_between) *  -- Ways to arrange people between A and B
    (Nat.factorial (total_people - people_between - 2))  -- Ways to arrange remaining people
  := by sorry

end arrangement_count_correct_l2127_212735


namespace cube_volume_problem_l2127_212716

theorem cube_volume_problem (a : ℝ) (h : a > 0) :
  (a - 2) * a * (a + 2) = a^3 - 12 → a^3 = 27 := by
  sorry

end cube_volume_problem_l2127_212716


namespace range_of_m_range_of_x_l2127_212775

-- Part 1
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - 2 * m * x - 1 < 0) → m ∈ Set.Ioc (-1) 0 :=
sorry

-- Part 2
theorem range_of_x (x : ℝ) :
  (∀ m : ℝ, |m| ≤ 1 → m * x^2 - 2 * m * x - 1 < 0) → 
  x ∈ Set.Ioo (1 - Real.sqrt 2) 1 ∪ Set.Ioo 1 (1 + Real.sqrt 2) :=
sorry

end range_of_m_range_of_x_l2127_212775


namespace vector_sum_magnitude_constraint_l2127_212717

/-- Given vectors a and b, if the magnitude of their sum does not exceed 5,
    then the second component of b is in the range [-6, 2]. -/
theorem vector_sum_magnitude_constraint (a b : ℝ × ℝ) (h : ‖a + b‖ ≤ 5) :
  a = (-2, 2) → b.1 = 5 → -6 ≤ b.2 ∧ b.2 ≤ 2 := by
  sorry

#check vector_sum_magnitude_constraint

end vector_sum_magnitude_constraint_l2127_212717


namespace expression_simplification_l2127_212724

theorem expression_simplification (m n : ℚ) (hm : m = 1) (hn : n = -3) :
  2/3 * (6*m - 9*m*n) - (n^2 - 6*m*n) = -5 := by
  sorry

end expression_simplification_l2127_212724


namespace arithmetic_geometric_sequence_l2127_212740

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric sequence -/
def geometric_seq (x y z : ℝ) : Prop :=
  y / x = z / y ∧ y ≠ 0

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  arithmetic_seq a →
  geometric_seq (a 1) (a 3) (a 4) →
  a 2 = -6 := by
  sorry

end arithmetic_geometric_sequence_l2127_212740


namespace karen_is_ten_l2127_212793

def sisters : Finset ℕ := {2, 4, 6, 8, 10, 12, 14}

def park_pair (a b : ℕ) : Prop :=
  a ∈ sisters ∧ b ∈ sisters ∧ a ≠ b ∧ a + b = 20

def pool_pair (a b : ℕ) : Prop :=
  a ∈ sisters ∧ b ∈ sisters ∧ a ≠ b ∧ 3 < a ∧ a < 9 ∧ 3 < b ∧ b < 9

def karen_age (k : ℕ) : Prop :=
  k ∈ sisters ∧ k ≠ 4 ∧
  ∃ (p1 p2 s1 s2 : ℕ),
    park_pair p1 p2 ∧
    pool_pair s1 s2 ∧
    p1 ≠ s1 ∧ p1 ≠ s2 ∧ p2 ≠ s1 ∧ p2 ≠ s2 ∧
    k ≠ p1 ∧ k ≠ p2 ∧ k ≠ s1 ∧ k ≠ s2

theorem karen_is_ten : ∃! k, karen_age k ∧ k = 10 := by
  sorry

end karen_is_ten_l2127_212793


namespace last_three_digits_of_2_to_15000_l2127_212720

theorem last_three_digits_of_2_to_15000 (h : 2^500 ≡ 1 [ZMOD 1250]) :
  2^15000 ≡ 1 [ZMOD 1000] := by
sorry

end last_three_digits_of_2_to_15000_l2127_212720


namespace pencils_given_l2127_212713

theorem pencils_given (initial : ℕ) (total : ℕ) (given : ℕ) : 
  initial = 9 → total = 65 → given = total - initial :=
by
  sorry

end pencils_given_l2127_212713


namespace correct_quantities_correct_min_discount_l2127_212727

-- Define the problem parameters
def total_cost : ℝ := 25000
def total_profit : ℝ := 11700
def ornament_cost : ℝ := 40
def ornament_price : ℝ := 58
def pendant_cost : ℝ := 30
def pendant_price : ℝ := 45
def second_profit_goal : ℝ := 10800

-- Define the quantities of ornaments and pendants
def ornaments : ℕ := 400
def pendants : ℕ := 300

-- Define the theorem for part 1
theorem correct_quantities :
  ornament_cost * ornaments + pendant_cost * pendants = total_cost ∧
  (ornament_price - ornament_cost) * ornaments + (pendant_price - pendant_cost) * pendants = total_profit :=
sorry

-- Define the minimum discount percentage
def min_discount_percentage : ℝ := 20

-- Define the theorem for part 2
theorem correct_min_discount :
  let new_pendant_price := pendant_price * (1 - min_discount_percentage / 100)
  (ornament_price - ornament_cost) * ornaments + (new_pendant_price - pendant_cost) * (2 * pendants) ≥ second_profit_goal ∧
  ∀ d : ℝ, d < min_discount_percentage →
    let price := pendant_price * (1 - d / 100)
    (ornament_price - ornament_cost) * ornaments + (price - pendant_cost) * (2 * pendants) < second_profit_goal :=
sorry

end correct_quantities_correct_min_discount_l2127_212727


namespace shaded_region_perimeter_l2127_212737

theorem shaded_region_perimeter (r : ℝ) : 
  r > 0 →
  2 * Real.pi * r = 24 →
  (3 : ℝ) * (1 / 6 : ℝ) * (2 * Real.pi * r) = 12 :=
by sorry

end shaded_region_perimeter_l2127_212737


namespace employed_males_percentage_l2127_212736

/-- Given a population where 60% are employed and 20% of the employed are females,
    prove that 48% of the population are employed males. -/
theorem employed_males_percentage
  (total_population : ℕ) 
  (employed_percentage : ℚ) 
  (employed_females_percentage : ℚ) 
  (h1 : employed_percentage = 60 / 100)
  (h2 : employed_females_percentage = 20 / 100) :
  (employed_percentage - employed_percentage * employed_females_percentage) * 100 = 48 := by
  sorry

end employed_males_percentage_l2127_212736


namespace cannot_compare_full_mark_students_l2127_212760

/-- Represents a school with a total number of students and full-mark scorers -/
structure School where
  total_students : ℕ
  full_mark_students : ℕ
  h_full_mark_valid : full_mark_students ≤ total_students

/-- The percentage of full-mark scorers in a school -/
def full_mark_percentage (s : School) : ℚ :=
  (s.full_mark_students : ℚ) / (s.total_students : ℚ) * 100

theorem cannot_compare_full_mark_students
  (school_A school_B : School)
  (h_A : full_mark_percentage school_A = 1)
  (h_B : full_mark_percentage school_B = 2) :
  ¬ (∀ (s₁ s₂ : School),
    full_mark_percentage s₁ = 1 →
    full_mark_percentage s₂ = 2 →
    (s₁.full_mark_students < s₂.full_mark_students ∨
     s₁.full_mark_students > s₂.full_mark_students ∨
     s₁.full_mark_students = s₂.full_mark_students)) :=
by
  sorry

end cannot_compare_full_mark_students_l2127_212760


namespace negation_of_existence_negation_of_quadratic_inequality_l2127_212752

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x : ℝ, f x ≤ 0) ↔ (∀ x : ℝ, f x > 0) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l2127_212752


namespace pie_distribution_l2127_212718

theorem pie_distribution (total_slices : ℕ) (carl_portion : ℚ) (nancy_slices : ℕ) :
  total_slices = 8 →
  carl_portion = 1 / 4 →
  nancy_slices = 2 →
  (total_slices - carl_portion * total_slices - nancy_slices : ℚ) / total_slices = 1 / 2 :=
by sorry

end pie_distribution_l2127_212718


namespace max_min_product_l2127_212769

def digits : List Nat := [2, 4, 6, 8]

def makeNumber (a b c : Nat) : Nat := 100 * a + 10 * b + c

def product (a b c d : Nat) : Nat := (makeNumber a b c) * d

theorem max_min_product :
  (∀ (a b c d : Nat), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits →
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    product a b c d ≤ product 8 6 4 2) ∧
  (∀ (a b c d : Nat), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits →
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    product 2 4 6 8 ≤ product a b c d) :=
by sorry

end max_min_product_l2127_212769


namespace some_number_value_l2127_212787

theorem some_number_value (X : ℝ) :
  2 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * X)) = 1600.0000000000002 →
  X = 1.25 := by
sorry

end some_number_value_l2127_212787


namespace prob_two_queens_or_at_least_one_king_is_34_221_l2127_212741

/-- Represents a standard deck of cards -/
structure StandardDeck :=
  (total_cards : ℕ)
  (num_kings : ℕ)
  (num_queens : ℕ)
  (h_total : total_cards = 52)
  (h_kings : num_kings = 4)
  (h_queens : num_queens = 4)

/-- Calculates the probability of drawing either two queens or at least 1 king -/
def prob_two_queens_or_at_least_one_king (deck : StandardDeck) : ℚ :=
  34 / 221

/-- Theorem stating that the probability of drawing either two queens or at least 1 king is 34/221 -/
theorem prob_two_queens_or_at_least_one_king_is_34_221 (deck : StandardDeck) :
  prob_two_queens_or_at_least_one_king deck = 34 / 221 := by
  sorry

end prob_two_queens_or_at_least_one_king_is_34_221_l2127_212741


namespace church_member_percentage_l2127_212722

theorem church_member_percentage (total_members : ℕ) (adult_members : ℕ) (child_members : ℕ) : 
  total_members = 120 →
  child_members = adult_members + 24 →
  total_members = adult_members + child_members →
  (adult_members : ℚ) / (total_members : ℚ) = 2/5 :=
by sorry

end church_member_percentage_l2127_212722


namespace right_triangle_hypotenuse_l2127_212704

theorem right_triangle_hypotenuse (PQ PR : ℝ) (h1 : PQ = 15) (h2 : PR = 20) :
  Real.sqrt (PQ^2 + PR^2) = 25 := by
sorry

end right_triangle_hypotenuse_l2127_212704


namespace tenth_vertex_label_l2127_212731

/-- Regular 2012-gon with vertices labeled according to specific conditions -/
structure Polygon2012 where
  /-- The labeling function for vertices -/
  label : Fin 2012 → Fin 2012
  /-- The first vertex is labeled A₁ -/
  first_vertex : label 0 = 0
  /-- The second vertex is labeled A₄ -/
  second_vertex : label 1 = 3
  /-- If k+ℓ and m+n have the same remainder mod 2012, then AₖAₗ and AₘAₙ don't intersect -/
  non_intersecting_chords : ∀ k ℓ m n : Fin 2012, 
    (k + ℓ) % 2012 = (m + n) % 2012 → 
    (label k + label ℓ) % 2012 ≠ (label m + label n) % 2012

/-- The label of the tenth vertex in a Polygon2012 is A₂₈ -/
theorem tenth_vertex_label (p : Polygon2012) : p.label 9 = 27 := by
  sorry

end tenth_vertex_label_l2127_212731


namespace cube_sum_problem_l2127_212788

theorem cube_sum_problem (a b c : ℝ) 
  (sum_eq : a + b + c = 5)
  (sum_prod_eq : b * c + c * a + a * b = 7)
  (prod_eq : a * b * c = 2) :
  a^3 + b^3 + c^3 = 26 := by
  sorry

end cube_sum_problem_l2127_212788


namespace celsius_to_fahrenheit_l2127_212734

theorem celsius_to_fahrenheit (C F : ℝ) : 
  C = (4/7) * (F - 40) → C = 35 → F = 101.25 := by
sorry

end celsius_to_fahrenheit_l2127_212734


namespace joel_puzzles_l2127_212799

/-- The number of puzzles Joel collected -/
def puzzles : ℕ := sorry

/-- The number of toys Joel's sister donated -/
def sister_toys : ℕ := sorry

/-- The total number of toys Joel donated -/
def total_toys : ℕ := 108

/-- The number of stuffed animals Joel collected -/
def stuffed_animals : ℕ := 18

/-- The number of action figures Joel collected -/
def action_figures : ℕ := 42

/-- The number of board games Joel collected -/
def board_games : ℕ := 2

/-- The number of toys Joel added from his own closet -/
def joel_toys : ℕ := 22

theorem joel_puzzles :
  puzzles = 13 ∧
  sister_toys * 2 = joel_toys ∧
  stuffed_animals + action_figures + board_games + puzzles + sister_toys + joel_toys = total_toys :=
sorry

end joel_puzzles_l2127_212799


namespace z_in_second_quadrant_l2127_212762

def z₁ : ℂ := -3 + Complex.I
def z₂ : ℂ := 1 - Complex.I
def z : ℂ := z₁ - z₂

theorem z_in_second_quadrant : 
  z.re < 0 ∧ z.im > 0 := by sorry

end z_in_second_quadrant_l2127_212762


namespace cubic_equation_root_l2127_212763

theorem cubic_equation_root (a b : ℚ) :
  ((-2 : ℝ) - 5 * Real.sqrt 3) ^ 3 + a * ((-2 : ℝ) - 5 * Real.sqrt 3) ^ 2 + 
  b * ((-2 : ℝ) - 5 * Real.sqrt 3) + 49 = 0 →
  a = 235 / 71 := by
sorry

end cubic_equation_root_l2127_212763
